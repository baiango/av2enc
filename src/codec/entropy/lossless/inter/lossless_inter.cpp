// AV2 v1.0.0 lossless inter encoder/reader.  See lossless_inter.h.
//
// The code is written from the specification.  It does not reuse the private
// tile/coefficient helpers of `encoder/lossless`; only the public reversible
// WHT/DPCM helpers and the committed §9.3 CDF tables are shared.

#include "codec/entropy/lossless/inter/lossless_inter.h"

#include "core/bitstream/bitwriter.h"
#include "codec/intra/modes/intra_modes.h"
#include "codec/entropy/lossless/inter/lossless_inter_cdf_tables.h"
#include "codec/entropy/lossless/lossless_cdf_tables.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"
#include "codec/inter/multiref/gate/compound_cdf_tables.h"
#include "codec/inter/tlayers/tlayers.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace av2enc::lossless::inter {
namespace {

using av2enc::av2::Decoder;
using av2enc::av2::Encoder;
using av2enc::av2::Status;
namespace tl = ::av2enc::inter::tlayers;
using partition::BlockSize;
using partition::PartitionType;

constexpr int kDcPred = 0;
constexpr int kV = 1;
constexpr int kH = 2;
constexpr int kIntraModesCount = 13;
constexpr int kSentinel256 = static_cast<int>(BlockSize::k256x256);

// §9.2 Fsc_Bsize_Groups / Size_Group.
constexpr int kFscBsizeGroups[29] = {
    0, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 3, 3, 4, 4, 6, 6, 4, 4, 6, 6};
constexpr int kSizeGroup[29] = {
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2};

// §8.3.2 coefficient context offsets.
constexpr int kSigRefDiff[5][2] = {{0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

int clip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }
int abs_int(int v) { return v < 0 ? -v : v; }

int floor_log2(int v) {
  int r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

bool y_mode_is_directional(int mode) { return mode >= kV && mode <= 8; }

template <typename T, typename U, std::size_t N>
void copy_array(T (&dst)[N], const U (&src)[N]) {
  static_assert(sizeof(T) == sizeof(U), "CDF element size mismatch");
  std::memcpy(dst, src, sizeof(dst));
}

// ---------------------------------------------------------------------------
// Reversible transforms (§7.15.2.2, §7.15.4)
// ---------------------------------------------------------------------------

}  // namespace

void inverse_wht4(const int in[4], int shift, int out[4]) {
  int a = in[0] >> shift;
  int c = in[1] >> shift;
  int d = in[2] >> shift;
  int b = in[3] >> shift;
  a += c;
  d -= b;
  const int e = (a - d) >> 1;
  b = e - b;
  c = e - c;
  a -= b;
  d += c;
  out[0] = a;
  out[1] = b;
  out[2] = c;
  out[3] = d;
}

void forward_wht4(const int in[4], int out[4]) {
  const int A = in[0];
  const int B = in[1];
  const int C = in[2];
  const int D = in[3];
  const int a1 = A + B;
  const int d1 = D - C;
  const int e = (a1 - d1) >> 1;
  const int b0 = e - B;
  const int c0 = e - C;
  const int a0 = a1 - c0;
  const int d0 = d1 + b0;
  out[0] = a0;
  out[1] = c0;
  out[2] = d0;
  out[3] = b0;
}

namespace {

bool is_pow2_dim(int v) { return v == 4 || v == 8 || v == 16 || v == 32; }

void forward_wht_2d(const int* residual, int w, int h, int* levels) {
  std::vector<int> tmp(static_cast<std::size_t>(w) * h);
  for (int c = 0; c < w; ++c) {
    int col[4];
    int out[4];
    for (int r = 0; r < h; r += 4) {
      for (int k = 0; k < 4; ++k) col[k] = residual[(r + k) * w + c];
      forward_wht4(col, out);
      for (int k = 0; k < 4; ++k) tmp[(r + k) * w + c] = out[k];
    }
  }
  for (int r = 0; r < h; ++r) {
    int row[4];
    int out[4];
    for (int c = 0; c < w; c += 4) {
      for (int k = 0; k < 4; ++k) row[k] = tmp[r * w + c + k];
      forward_wht4(row, out);
      for (int k = 0; k < 4; ++k) levels[r * w + c + k] = out[k];
    }
  }
}

void inverse_wht_2d(const int* levels, int w, int h, int* residual) {
  std::vector<int> tmp(static_cast<std::size_t>(w) * h);
  for (int r = 0; r < h; ++r) {
    int row[4];
    int out[4];
    for (int c = 0; c < w; c += 4) {
      for (int k = 0; k < 4; ++k) row[k] = levels[r * w + c + k] * 8;
      inverse_wht4(row, 3, out);
      for (int k = 0; k < 4; ++k) tmp[r * w + c + k] = out[k];
    }
  }
  for (int c = 0; c < w; ++c) {
    int col[4];
    int out[4];
    for (int r = 0; r < h; r += 4) {
      for (int k = 0; k < 4; ++k) col[k] = tmp[(r + k) * w + c];
      inverse_wht4(col, 0, out);
      for (int k = 0; k < 4; ++k) residual[(r + k) * w + c] = out[k];
    }
  }
}

}  // namespace

void forward_lossless_block(const int* residual, int w, int h, bool dct_dct,
                            int* levels) {
  if (dct_dct) {
    assert(is_pow2_dim(w) && is_pow2_dim(h));
    forward_wht_2d(residual, w, h, levels);
    return;
  }
  std::memcpy(levels, residual, static_cast<std::size_t>(w) * h * sizeof(int));
}

void inverse_lossless_block(const int* levels, int w, int h, bool dct_dct,
                            int* residual) {
  if (dct_dct) {
    assert(is_pow2_dim(w) && is_pow2_dim(h));
    inverse_wht_2d(levels, w, h, residual);
    return;
  }
  std::memcpy(residual, levels, static_cast<std::size_t>(w) * h * sizeof(int));
}

namespace {

void dpcm_difference(const int* residual, int w, int h, bool vertical,
                     int* out) {
  std::memcpy(out, residual, static_cast<std::size_t>(w) * h * sizeof(int));
  if (vertical) {
    for (int j = 0; j < w; ++j) {
      for (int i = 1; i < h; ++i) {
        out[i * w + j] = residual[i * w + j] - residual[(i - 1) * w + j];
      }
    }
  } else {
    for (int i = 0; i < h; ++i) {
      for (int j = 1; j < w; ++j) {
        out[i * w + j] = residual[i * w + j] - residual[i * w + (j - 1)];
      }
    }
  }
}

void dpcm_cumsum(const int* in, int w, int h, bool vertical, int* out) {
  std::memcpy(out, in, static_cast<std::size_t>(w) * h * sizeof(int));
  if (vertical) {
    for (int j = 0; j < w; ++j) {
      for (int i = 1; i < h; ++i) out[i * w + j] += out[(i - 1) * w + j];
    }
  } else {
    for (int i = 0; i < h; ++i) {
      for (int j = 1; j < w; ++j) out[i * w + j] += out[i * w + (j - 1)];
    }
  }
}

}  // namespace

void dpcm_difference_block(const int* residual, int w, int h, bool vertical,
                           int* out) {
  dpcm_difference(residual, w, h, vertical, out);
}

void dpcm_cumsum_block(const int* in, int w, int h, bool vertical, int* out) {
  dpcm_cumsum(in, w, h, vertical, out);
}

// ---------------------------------------------------------------------------
// §5.18.2 lossless flags
// ---------------------------------------------------------------------------

FrameLossless derive_lossless(const FrameQuantInputs& inputs,
                              bool frame_is_inter) {
  (void)frame_is_inter;  // §5.18.2 is independent of the frame type.
  FrameLossless out;
  out.coded_lossless = true;
  out.num_segments =
      std::max(1, std::min(inputs.num_segments <= 0 ? kBaseSegments
                                                    : inputs.num_segments,
                           kMaxSegments));
  for (int s = 0; s < out.num_segments; ++s) {
    int qindex = inputs.base_q_idx;
    if (inputs.alt_q_enabled[s]) qindex += inputs.alt_q[s];
    qindex = clip3(0, 255, qindex);
    out.qindex[s] = qindex;
    const bool lossless =
        qindex == 0 && !inputs.delta_q_present &&
        inputs.delta_q_y_dc + inputs.base_y_dc_delta_q <= 0 &&
        inputs.delta_q_u_dc + inputs.base_uv_dc_delta_q <= 0 &&
        inputs.delta_q_v_dc + inputs.base_uv_dc_delta_q <= 0 &&
        inputs.delta_q_u_ac + inputs.base_uv_ac_delta_q <= 0 &&
        inputs.delta_q_v_ac + inputs.base_uv_ac_delta_q <= 0;
    out.lossless[s] = lossless;
    if (lossless) {
      out.has_lossless_segment = true;
    } else {
      out.coded_lossless = false;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

bool make_geometry(int width, int height, ChromaFormat format, Geometry* out,
                   std::string* error) {
  if (width <= 0 || height <= 0 || (width & 7) != 0 || (height & 7) != 0) {
    *error = "lossless inter requires width/height positive multiples of 8";
    return false;
  }
  Geometry g;
  g.width = width;
  g.height = height;
  g.format = format;
  switch (format) {
    case ChromaFormat::k420:
      g.sub_x = 1;
      g.sub_y = 1;
      break;
    case ChromaFormat::k422:
      g.sub_x = 1;
      g.sub_y = 0;
      break;
    case ChromaFormat::k444:
      g.sub_x = 0;
      g.sub_y = 0;
      break;
    case ChromaFormat::k400:
      // §6.4.2 Table 6.2: CHROMA_FORMAT_400 derives SubsamplingX/Y = 1/1 but
      // has no chroma planes.
      g.sub_x = 1;
      g.sub_y = 1;
      g.has_chroma = false;
      break;
  }
  g.mi_cols = 2 * ((width + 7) >> 3);
  g.mi_rows = 2 * ((height + 7) >> 3);
  g.chroma_w = g.has_chroma ? (width >> g.sub_x) : 0;
  g.chroma_h = g.has_chroma ? (height >> g.sub_y) : 0;
  g.chroma_mi_cols = g.has_chroma ? (g.mi_cols >> g.sub_x) : 0;
  g.chroma_mi_rows = g.has_chroma ? (g.mi_rows >> g.sub_y) : 0;
  g.sb_cols = (g.mi_cols + 15) >> 4;
  g.sb_rows = (g.mi_rows + 15) >> 4;
  *out = g;
  return true;
}

// ---------------------------------------------------------------------------
// §5.20.6.1 TX size selection
// ---------------------------------------------------------------------------

bool find_tx_size(int coded_w, int coded_h, int* tx_w, int* tx_h) {
  int w = std::min(32, coded_w);
  int h = std::min(32, coded_h);
  if (!is_pow2_dim(w) || !is_pow2_dim(h)) return false;
  *tx_w = w;
  *tx_h = h;
  return true;
}

TxSizeDecision select_tx_size(int block_w, int block_h, bool is_inter,
                              bool fsc_mode, bool allow_select,
                              bool want_large) {
  TxSizeDecision d;
  const bool blk4 = block_w == 4 && block_h == 4;
  if (blk4 || (!is_inter && !fsc_mode) || !allow_select) {
    d.coded_w = 4;
    d.coded_h = 4;
    return d;
  }
  d.signal_lossless_tx_size = true;
  d.lossless_tx_size = want_large;
  if (want_large && find_tx_size(block_w, block_h, &d.coded_w, &d.coded_h)) {
    return d;
  }
  d.coded_w = 4;
  d.coded_h = 4;
  d.lossless_tx_size = false;
  return d;
}

// ---------------------------------------------------------------------------
// CDF state
// ---------------------------------------------------------------------------

namespace {

struct CoeffCdfs {
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_16[3][6];
  std::uint16_t eob_pt_32[3][7];
  std::uint16_t eob_pt_64[3][8];
  std::uint16_t eob_pt_128[3][9];
  std::uint16_t eob_pt_256[3][9];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_extra[3];
  std::uint16_t coeff_base[5][20][2][5];
  std::uint16_t coeff_base_lf[5][33][2][7];
  std::uint16_t coeff_base_eob[5][4][4];
  std::uint16_t coeff_base_lf_eob[5][4][6];
  std::uint16_t coeff_base_uv[12][5];
  std::uint16_t coeff_base_lf_uv[12][7];
  std::uint16_t coeff_base_eob_uv[4][4];
  std::uint16_t coeff_base_lf_eob_uv[4][6];
  std::uint16_t coeff_br[7][5];
  std::uint16_t coeff_br_lf[14][5];
  std::uint16_t coeff_br_uv[4][5];
  std::uint16_t dc_sign[2][2][3][3];
  std::uint16_t coeff_base_bob[3][3][4];
  std::uint16_t coeff_base_idtx[3][7][5];
  std::uint16_t coeff_br_idtx[3][7][5];
  std::uint16_t idtx_sign[3][9][3];
};

struct CdfState {
  std::uint16_t do_split[64][3];
  std::uint16_t rect_type[64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
  std::uint16_t fsc_mode[4][6][3];
  std::uint16_t use_dpcm_y[3];
  std::uint16_t dpcm_mode_y[3];
  std::uint16_t use_dpcm_uv[3];
  std::uint16_t dpcm_mode_uv[3];
  std::uint16_t lossless_tx_size[4][2][3];
  std::uint16_t lossless_inter_tx_type[3];
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t single_mode[5][4];
  // §5.20.7.10-.12 two-reference syntax (B frames).
  std::uint16_t comp_mode[5][3];
  std::uint16_t comp_ref0[3][6][3];
  std::uint16_t single_ref[3][6][3];
  std::uint16_t is_joint[2][3];
  std::uint16_t compound_mode_non_joint[5][6];
  CoeffCdfs coeff;
};

void init_cdfs(CdfState& c) {
  std::memcpy(c.do_split, av2enc::basic::kDoSplitCdf[0], sizeof(c.do_split));
  std::memcpy(c.rect_type, av2enc::basic::kRectTypeCdf[0], sizeof(c.rect_type));
  copy_array(c.y_mode_set, av2enc::basic::kYModeSetCdf);
  copy_array(c.y_mode_index, av2enc::basic::kYModeIndexCdf);
  copy_array(c.uv_mode, av2enc::basic::kUvModeCflNotAllowedCdf);
  copy_array(c.fsc_mode, kFscModeCdf);
  copy_array(c.use_dpcm_y, kUseDpcmYCdf);
  copy_array(c.dpcm_mode_y, kDpcmModeYCdf);
  copy_array(c.use_dpcm_uv, kUseDpcmUvCdf);
  copy_array(c.dpcm_mode_uv, kDpcmModeUvCdf);
  copy_array(c.lossless_tx_size, kLosslessTxSizeCdf);
  copy_array(c.lossless_inter_tx_type, kLosslessInterTxTypeCdf);
  copy_array(c.is_inter, av2enc::video::kIsInterCdf);
  copy_array(c.skip, av2enc::video::kSkipCdf);
  copy_array(c.single_mode, av2enc::video::kSingleModeCdf);
  copy_array(c.comp_mode, av2enc::video::multiref::gate::kCompModeCdf);
  copy_array(c.comp_ref0, av2enc::video::multiref::gate::kCompRef0Cdf);
  copy_array(c.single_ref, av2enc::video::kSingleRefCdf);
  copy_array(c.is_joint, av2enc::video::multiref::gate::kIsJointCdf);
  copy_array(c.compound_mode_non_joint,
             av2enc::video::multiref::gate::kCompoundModeNonJointCdf);

  const auto& q = av2enc::still::qctx::tables_for_slice(0);
  copy_array(c.coeff.txb_skip, *q.txb_skip);
  copy_array(c.coeff.v_txb_skip, *q.v_txb_skip);
  copy_array(c.coeff.eob_pt_16, *q.eob_pt_16);
  copy_array(c.coeff.eob_pt_32, *q.eob_pt_32);
  copy_array(c.coeff.eob_pt_64, *q.eob_pt_64);
  copy_array(c.coeff.eob_pt_128, *q.eob_pt_128);
  copy_array(c.coeff.eob_pt_256, *q.eob_pt_256);
  copy_array(c.coeff.eob_pt_512, *q.eob_pt_512);
  copy_array(c.coeff.eob_pt_1024, *q.eob_pt_1024);
  copy_array(c.coeff.eob_extra, *q.eob_extra);
  copy_array(c.coeff.coeff_base, *q.coeff_base);
  copy_array(c.coeff.coeff_base_lf, *q.coeff_base_lf);
  copy_array(c.coeff.coeff_base_eob, *q.coeff_base_eob);
  copy_array(c.coeff.coeff_base_lf_eob, *q.coeff_base_lf_eob);
  copy_array(c.coeff.coeff_base_uv, *q.coeff_base_uv);
  copy_array(c.coeff.coeff_base_lf_uv, *q.coeff_base_lf_uv);
  copy_array(c.coeff.coeff_base_eob_uv, *q.coeff_base_eob_uv);
  copy_array(c.coeff.coeff_base_lf_eob_uv, *q.coeff_base_lf_eob_uv);
  copy_array(c.coeff.coeff_br, *q.coeff_br);
  copy_array(c.coeff.coeff_br_lf, *q.coeff_br_lf);
  copy_array(c.coeff.coeff_br_uv, *q.coeff_br_uv);
  copy_array(c.coeff.dc_sign, *q.dc_sign);
  copy_array(c.coeff.coeff_base_bob, *q.coeff_base_bob);
  copy_array(c.coeff.coeff_base_idtx, *q.coeff_base_idtx);
  copy_array(c.coeff.coeff_br_idtx, *q.coeff_br_idtx);
  copy_array(c.coeff.idtx_sign, *q.idtx_sign);
}

// ---------------------------------------------------------------------------
// Symbol writer / reader
// ---------------------------------------------------------------------------

class SymWriter {
 public:
  bool init(std::uint8_t* out, std::size_t capacity, bool adaptive,
            std::string* error) {
    enc_.disable_cdf_update = adaptive ? 0 : 1;
    const Status s = enc_.init_symbol(out, capacity);
    if (s != Status::kOk) {
      *error = std::string("entropy init: ") + av2enc::av2::status_name(s);
      return false;
    }
    return true;
  }
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const Status s = enc_.encode_symbol(cdf, n, symbol);
    if (s != Status::kOk) {
      error_ = std::string("encode_symbol: ") + av2enc::av2::status_name(s);
      ok_ = false;
      return false;
    }
    return true;
  }
  bool bit(bool b) {
    const Status s = enc_.encode_bool(b);
    if (s != Status::kOk) {
      error_ = std::string("encode_bool: ") + av2enc::av2::status_name(s);
      ok_ = false;
      return false;
    }
    return true;
  }
  bool literal(int n, std::uint32_t value) {
    for (int i = n - 1; i >= 0; --i) {
      if (!bit(((value >> i) & 1u) != 0u)) return false;
    }
    return true;
  }
  bool finish(std::size_t* bytes) {
    const Status s = enc_.exit_symbol(bytes);
    if (s != Status::kOk) {
      error_ = std::string("exit_symbol: ") + av2enc::av2::status_name(s);
      ok_ = false;
      return false;
    }
    return true;
  }
  bool ok() const { return ok_; }
  const std::string& error() const { return error_; }

 private:
  Encoder enc_{};
  bool ok_ = true;
  std::string error_;
};

class SymReader {
 public:
  bool init(const std::uint8_t* data, std::size_t size, std::size_t sz,
            bool adaptive, std::string* error) {
    dec_.disable_cdf_update = adaptive ? 0 : 1;
    const Status s = dec_.init_symbol(data, size, sz);
    if (s != Status::kOk) {
      *error = std::string("entropy init: ") + av2enc::av2::status_name(s);
      return false;
    }
    return true;
  }
  bool sym(std::uint16_t* cdf, int n, int* symbol) {
    const Status s = dec_.decode_symbol(cdf, n, symbol);
    if (s != Status::kOk) {
      error_ = std::string("decode_symbol: ") + av2enc::av2::status_name(s);
      ok_ = false;
      return false;
    }
    return true;
  }
  bool bit(bool* b) {
    const Status s = dec_.read_bool(b);
    if (s != Status::kOk) {
      error_ = std::string("read_bool: ") + av2enc::av2::status_name(s);
      ok_ = false;
      return false;
    }
    return true;
  }
  bool literal(int n, std::uint32_t* value) {
    std::uint32_t v = 0;
    for (int i = 0; i < n; ++i) {
      bool b = false;
      if (!bit(&b)) return false;
      v = (v << 1) | (b ? 1u : 0u);
    }
    *value = v;
    return true;
  }
  bool finish() {
    const Status s = dec_.exit_symbol();
    if (s != Status::kOk) {
      error_ = std::string("exit_symbol: ") + av2enc::av2::status_name(s);
      ok_ = false;
      return false;
    }
    return true;
  }
  bool ok() const { return ok_; }
  const std::string& error() const { return error_; }

 private:
  Decoder dec_{};
  bool ok_ = true;
  std::string error_;
};

// ---------------------------------------------------------------------------
// Plane and prediction
// ---------------------------------------------------------------------------

struct Plane {
  int w = 0;
  int h = 0;
  // Samples are stored as uint16_t at every bit depth (8/10/12).  The stream
  // syntax is depth-scaled; only the reconstruction clip changes.
  std::vector<std::uint16_t> data;

  void resize(int width, int height) {
    w = width;
    h = height;
    data.assign(static_cast<std::size_t>(w) * h, 0);
  }
  std::uint16_t* row(int y) {
    return data.data() + static_cast<std::size_t>(y) * w;
  }
  const std::uint16_t* row(int y) const {
    return data.data() + static_cast<std::size_t>(y) * w;
  }
};

// §7.13.3.22 Div_Lut, used by the §7.13.2.11 DC approx_divide().
constexpr int kLosslessDivLutBits = 7;
constexpr int kLosslessDivLutPrecBits = 9;
constexpr int kLosslessDivLut[129] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465,
    462, 458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423,
    420, 417, 415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388,
    386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358,
    356, 354, 352, 350, 349, 347, 345, 343, 341, 340, 338, 336, 334, 333,
    331, 329, 328, 326, 324, 323, 321, 320, 318, 317, 315, 314, 312, 311,
    309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295, 294, 293, 291,
    290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 277, 275, 274,
    273, 272, 271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259,
    258, 257, 256};

int round2_shift(int value, int shift) {
  if (shift <= 0) return value;
  return (value + (1 << (shift - 1))) >> shift;
}

int resolve_divisor(int divisor) {
  const int magnitude = std::abs(divisor);
  const int n = floor_log2(magnitude);
  const int e = magnitude - (1 << n);
  const int f = n > kLosslessDivLutBits
                    ? round2_shift(e, n - kLosslessDivLutBits)
                    : (e << (kLosslessDivLutBits - n));
  return kLosslessDivLut[f] * (divisor < 0 ? -1 : 1);
}

int divisor_shift(int divisor) {
  return floor_log2(std::abs(divisor)) + kLosslessDivLutPrecBits;
}

int approx_divide_lossless(int numerator, int denominator) {
  return round2_shift(numerator * resolve_divisor(denominator),
                      divisor_shift(denominator));
}

int clip1_depth(int value, int bit_depth) {
  return clip3(0, (1 << bit_depth) - 1, value);
}

// §7.13.2.1 edge arrays for a 10/12-bit plane.  The module codes only DC, V
// and H, so the directional padding / edge filtering of the shared 8-bit
// intra component is not needed; the unavailable-neighbour fillers (mid - 1
// for above, mid + 1 for left) match §7.13.2.1 exactly.
struct Edges16 {
  std::array<int, 129> above{};
  std::array<int, 129> left{};
};

void prepare_edges16(const Plane& plane, int x, int y, int w, int h,
                     int bit_depth, bool have_left, bool have_above,
                     Edges16* edges) {
  const int mid = 1 << (bit_depth - 1);
  const int num_needed = w + h;
  const int max_x = plane.w - 1;
  const int max_y = plane.h - 1;
  for (int i = 0; i < num_needed; ++i) {
    if (!have_left && have_above) {
      edges->left[i] = plane.data[static_cast<std::size_t>(y - 1) * plane.w + x];
    } else if (!have_left) {
      edges->left[i] = mid + 1;
    } else {
      const int row = std::min(y + i, max_y);
      edges->left[i] =
          plane.data[static_cast<std::size_t>(row) * plane.w + (x - 1)];
    }
  }
  for (int i = 0; i < num_needed; ++i) {
    if (!have_above && have_left) {
      edges->above[i] =
          plane.data[static_cast<std::size_t>(y) * plane.w + (x - 1)];
    } else if (!have_above) {
      edges->above[i] = mid - 1;
    } else {
      const int col = std::min(x + i, max_x);
      edges->above[i] =
          plane.data[static_cast<std::size_t>(y - 1) * plane.w + col];
    }
  }
}

bool predict_intra_block(const Plane& plane, int x, int y, int w, int h,
                         int mode, bool have_left, bool have_above,
                         int bit_depth, std::uint16_t* dst,
                         std::size_t dst_stride) {
  if (static_cast<int>(dst_stride) < w) return false;
  Edges16 edges;
  prepare_edges16(plane, x, y, w, h, bit_depth, have_left, have_above, &edges);
  switch (mode) {
    case kDcPred: {
      int value = 0;
      if (have_left && have_above) {
        int sum = 0;
        for (int k = 0; k < h; ++k) sum += edges.left[k];
        for (int k = 0; k < w; ++k) sum += edges.above[k];
        value = clip1_depth(approx_divide_lossless(sum, w + h), bit_depth);
      } else if (have_left) {
        int sum = 0;
        for (int k = 0; k < h; ++k) sum += edges.left[k];
        value = round2_shift(sum, floor_log2(h));
      } else if (have_above) {
        int sum = 0;
        for (int k = 0; k < w; ++k) sum += edges.above[k];
        value = round2_shift(sum, floor_log2(w));
      } else {
        value = 1 << (bit_depth - 1);
      }
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          dst[static_cast<std::size_t>(i) * dst_stride + j] =
              static_cast<std::uint16_t>(value);
        }
      }
      return true;
    }
    case kV:
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          dst[static_cast<std::size_t>(i) * dst_stride + j] =
              static_cast<std::uint16_t>(edges.above[j]);
        }
      }
      return true;
    case kH:
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          dst[static_cast<std::size_t>(i) * dst_stride + j] =
              static_cast<std::uint16_t>(edges.left[i]);
        }
      }
      return true;
    default:
      return false;
  }
}

void predict_inter_zero(const Plane& ref, int x, int y, int w, int h,
                        std::uint16_t* dst, std::size_t dst_stride) {
  for (int i = 0; i < h; ++i) {
    const std::uint16_t* src = ref.data.data() +
                               static_cast<std::size_t>(y + i) * ref.w + x;
    std::uint16_t* d = dst + static_cast<std::size_t>(i) * dst_stride;
    std::memcpy(d, src, static_cast<std::size_t>(w) * sizeof(std::uint16_t));
  }
}

// §7.13.3.14 COMPOUND_AVERAGE at CWP_EQUAL with two zero-MV EIGHTTAP
// predictions: InterRound1 = 7 makes each Preds[] entry a 1/16-precision
// sample, and the blend is Round2(8 * p0 + 8 * p1, 4 + InterPostRound) with
// InterPostRound = 4, i.e. Round2(ref0 + ref1, 1) after Clip1.
void predict_inter_compound(const Plane& ref0, const Plane& ref1, int x,
                            int y, int w, int h, int bit_depth,
                            std::uint16_t* dst, std::size_t dst_stride) {
  for (int i = 0; i < h; ++i) {
    const std::uint16_t* a = ref0.data.data() +
                             static_cast<std::size_t>(y + i) * ref0.w + x;
    const std::uint16_t* b = ref1.data.data() +
                             static_cast<std::size_t>(y + i) * ref1.w + x;
    std::uint16_t* d = dst + static_cast<std::size_t>(i) * dst_stride;
    for (int j = 0; j < w; ++j) {
      d[j] = static_cast<std::uint16_t>(
          clip1_depth((static_cast<int>(a[j]) + static_cast<int>(b[j]) + 1) >>
                          1,
                      bit_depth));
    }
  }
}

void reconstruct_block(const int* levels, int w, int h, bool dct_dct, bool dpcm,
                       bool vertical, const std::uint16_t* pred,
                       std::size_t pred_stride, std::uint16_t* recon,
                       std::size_t recon_stride, int bit_depth) {
  std::vector<int> tmp(static_cast<std::size_t>(w) * h);
  std::vector<int> residual(static_cast<std::size_t>(w) * h);
  inverse_lossless_block(levels, w, h, dct_dct, tmp.data());
  if (dpcm) {
    dpcm_cumsum(tmp.data(), w, h, vertical, residual.data());
  } else {
    std::memcpy(residual.data(), tmp.data(),
                static_cast<std::size_t>(w) * h * sizeof(int));
  }
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int p = pred[static_cast<std::size_t>(i) * pred_stride + j];
      recon[static_cast<std::size_t>(i) * recon_stride + j] =
          static_cast<std::uint16_t>(
              clip1_depth(p + residual[i * w + j], bit_depth));
    }
  }
}

// ---------------------------------------------------------------------------
// Tile context
// ---------------------------------------------------------------------------

struct TileContext {
  const Geometry* g = nullptr;
  std::vector<int> above_mi;
  std::vector<int> left_mi;
  std::array<std::vector<std::uint8_t>, 3> above_level;
  std::array<std::vector<std::uint8_t>, 3> left_level;
  std::array<std::vector<std::uint8_t>, 3> above_dc;
  std::array<std::vector<std::uint8_t>, 3> left_dc;
  std::vector<std::uint8_t> intra_joint;
  std::vector<std::uint8_t> fsc_modes;
  std::vector<std::uint8_t> is_inter;
  std::vector<std::uint8_t> ymode;
  std::vector<std::uint8_t> skip;
  // §5.20.7.10 ref state per mi cell for the comp_mode context.  0xff marks
  // "no reference list entry" (intra or single prediction list 1).
  std::vector<std::uint8_t> ref0;
  std::vector<std::uint8_t> ref1;
  int eob_u = 0;
  // Frame distances of the current reference list (list 0 / list 1), used by
  // the §5.20.7.6 is_joint context and §5.20.7.10 comp_mode backward test.
  int frame_distance[2] = {1, 0};
  int num_total_refs = 1;

  void reset(const Geometry& geo) {
    g = &geo;
    above_mi.assign(static_cast<std::size_t>(geo.mi_cols) + 16, kSentinel256);
    left_mi.assign(static_cast<std::size_t>(geo.mi_rows) + 16, kSentinel256);
    for (int p = 0; p < 3; ++p) {
      const int w = p == 0 ? geo.mi_cols : geo.chroma_mi_cols;
      const int h = p == 0 ? geo.mi_rows : geo.chroma_mi_rows;
      above_level[p].assign(static_cast<std::size_t>(w) + 16, 0);
      left_level[p].assign(static_cast<std::size_t>(h) + 16, 0);
      above_dc[p].assign(static_cast<std::size_t>(w) + 16, 0);
      left_dc[p].assign(static_cast<std::size_t>(h) + 16, 0);
    }
    const std::size_t cells =
        static_cast<std::size_t>(geo.mi_cols) * geo.mi_rows;
    intra_joint.assign(cells, 0);
    fsc_modes.assign(cells, 0);
    is_inter.assign(cells, 0);
    ymode.assign(cells, 0);
    skip.assign(cells, 0);
    ref0.assign(cells, 0xff);
    ref1.assign(cells, 0xff);
    eob_u = 0;
    clear_above();
  }

  void clear_above() {
    for (int p = 0; p < 3; ++p) {
      std::fill(above_level[p].begin(), above_level[p].end(), 0);
      std::fill(above_dc[p].begin(), above_dc[p].end(), 0);
    }
    std::fill(above_mi.begin(), above_mi.end(), kSentinel256);
  }
  void clear_left() {
    for (int p = 0; p < 3; ++p) {
      std::fill(left_level[p].begin(), left_level[p].end(), 0);
      std::fill(left_dc[p].begin(), left_dc[p].end(), 0);
    }
    std::fill(left_mi.begin(), left_mi.end(), kSentinel256);
  }

  bool inside(int r, int c) const {
    return r >= 0 && r < g->mi_rows && c >= 0 && c < g->mi_cols;
  }
  std::size_t cell(int r, int c) const {
    return static_cast<std::size_t>(r) * g->mi_cols + c;
  }

  int joint_mode(int r, int c, int w4, int h4, int dir) const {
    int mv_row = 0;
    int mv_col = 0;
    if (dir != 0) {
      mv_row = r - 1;
      mv_col = c + w4 - 1;
    } else {
      mv_col = c - 1;
      mv_row = r + h4 - 1;
    }
    if (inside(mv_row, mv_col)) return intra_joint[cell(mv_row, mv_col)];
    return 0;
  }

  int neighbor_positions(int pos[2][2], int r, int c, int w4, int h4) const {
    int n = 0;
    auto add = [&](int nr, int nc) {
      if (n < 2 && inside(nr, nc)) {
        pos[n][0] = nr;
        pos[n][1] = nc;
        ++n;
      }
    };
    add(r + h4 - 1, c - 1);
    add(r - 1, c + w4 - 1);
    add(r, c - 1);
    add(r - 1, c);
    return n;
  }

  int is_inter_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    auto intra_at = [&](int i) { return is_inter[cell(pos[i][0], pos[i][1])] == 0; };
    if (n == 2) {
      return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
    }
    if (n == 1) return 2 * (intra_at(0) ? 1 : 0);
    return 0;
  }

  int skip_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int ctx = 0;
    for (int i = 0; i < n; ++i) {
      ctx += skip[cell(pos[i][0], pos[i][1])] != 0 ? 1 : 0;
    }
    return ctx;
  }

  int new_mv_context(int r, int c, int w4, int h4) const {
    int new_mv_count = 0;
    const bool sb_border = (r & 15) == 0;
    auto scan = [&](int dr, int dc) -> int {
      const int mr = r + dr;
      const int mc = c + dc;
      if (!inside(mr, mc)) return 0;
      const std::size_t idx = cell(mr, mc);
      if (is_inter[idx] == 0) return 0;
      if (ymode[idx] == 16) new_mv_count = std::min(3, new_mv_count + 1);
      return 1;
    };
    const int left_a = scan(h4 - 1, -1);
    const int above_a = scan(-1, w4 - 1);
    const int left_b = scan(0, -1);
    const int above_b = (w4 >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
    const int nearest = ((above_a || above_b) ? 1 : 0) +
                        ((left_a || left_b) ? 1 : 0);
    return nearest + (new_mv_count > 0 ? 2 : 0);
  }

  // §8.3.2 check_backward().
  bool ref_backward(int ref) const {
    return ref >= 0 && ref < num_total_refs && frame_distance[ref] < 0;
  }

  // §8.3.2 comp_mode context (p595).
  int comp_mode_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    auto single_at = [&](int i) {
      return ref1[cell(pos[i][0], pos[i][1])] == 0xff;
    };
    auto intra_at = [&](int i) {
      return is_inter[cell(pos[i][0], pos[i][1])] == 0;
    };
    auto ref0_at = [&](int i) {
      return static_cast<int>(ref0[cell(pos[i][0], pos[i][1])]);
    };
    if (n == 2) {
      if (single_at(0) && single_at(1)) {
        return (ref_backward(ref0_at(0)) ? 1 : 0) ^
               (ref_backward(ref0_at(1)) ? 1 : 0);
      }
      if (single_at(0)) {
        return 2 + ((ref_backward(ref0_at(0)) || intra_at(0)) ? 1 : 0);
      }
      if (single_at(1)) {
        return 2 + ((ref_backward(ref0_at(1)) || intra_at(1)) ? 1 : 0);
      }
      return 4;
    }
    if (n == 1) {
      if (single_at(0)) return ref_backward(ref0_at(0)) ? 1 : 0;
      return 3;
    }
    return 1;
  }

  // §8.3.2 comp_ref / single_ref context (count_refs comparison).  With
  // NumTotalRefs == 2 it compares count_refs(0) against count_refs(1) over
  // the two neighbour cells.
  int ref_count_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int this_ref = 0;
    int next_ref = 0;
    for (int i = 0; i < n; ++i) {
      const std::size_t idx = cell(pos[i][0], pos[i][1]);
      if (is_inter[idx] == 0) continue;
      if (ref0[idx] == 0) ++this_ref;
      if (ref1[idx] == 0) ++this_ref;
      if (ref0[idx] == 1) ++next_ref;
      if (ref1[idx] == 1) ++next_ref;
    }
    if (this_ref == next_ref) return 1;
    return this_ref < next_ref ? 0 : 2;
  }

  // §8.3.2 is_joint context (p597).
  int is_joint_context(int d0, int d1) const {
    const int first = abs_int(d0);
    const int second = abs_int(d1);
    const bool same_side = (d0 < 0 && d1 < 0) || (d0 > 0 && d1 > 0);
    return (same_side || first != second) ? 1 : 0;
  }
};

// ---------------------------------------------------------------------------
// Partition helpers (§5.20.3.2)
// ---------------------------------------------------------------------------

bool is_chroma_offset_for_subsize(BlockSize sub) {
  return partition::mi_height_log2(sub) == 0 ||
         partition::mi_width_log2(sub) == 0;
}

bool is_chroma_offset_for_partition(PartitionType p, BlockSize bsize) {
  const BlockSize sub = partition::partition_subsize(p, bsize);
  if (sub == BlockSize::kInvalid) return false;
  return is_chroma_offset_for_subsize(sub);
}

bool plane_residual_valid(BlockSize sub) {
  return partition::block_width(sub) >= 8 &&
         partition::block_height(sub) >= 8;
}

bool is_partition_allowed(int r, int c, BlockSize bsize, PartitionType p,
                          int chroma_offset, bool has_chroma, int mi_rows,
                          int mi_cols) {
  const BlockSize sub = partition::partition_subsize(p, bsize);
  if (sub == BlockSize::kInvalid) return false;
  if (has_chroma && !chroma_offset) {
    chroma_offset = is_chroma_offset_for_partition(p, bsize) ? 1 : 0;
  }
  if (has_chroma && !chroma_offset && !plane_residual_valid(sub)) return false;
  if (p == PartitionType::kNone) {
    const int half_w = partition::num_4x4_blocks_wide(bsize) >> 1;
    const int half_h = partition::num_4x4_blocks_high(bsize) >> 1;
    return (r + half_h) < mi_rows && (c + half_w) < mi_cols;
  }
  if (has_chroma && chroma_offset) {
    const int half_w = partition::num_4x4_blocks_wide(bsize) >> 1;
    const int half_h = partition::num_4x4_blocks_high(bsize) >> 1;
    if (p == PartitionType::kHorz) return (r + half_h) < mi_rows;
    if (p == PartitionType::kVert) return (c + half_w) < mi_cols;
  }
  return true;
}

struct AllowedPartitions {
  int count = 0;
  bool none = false;
  bool horz = false;
  bool vert = false;
};

AllowedPartitions init_allowed(int r, int c, BlockSize bsize,
                               int chroma_offset, bool has_chroma, int mi_rows,
                               int mi_cols) {
  AllowedPartitions a;
  a.none = is_partition_allowed(r, c, bsize, PartitionType::kNone,
                                chroma_offset, has_chroma, mi_rows, mi_cols);
  a.horz = is_partition_allowed(r, c, bsize, PartitionType::kHorz,
                                chroma_offset, has_chroma, mi_rows, mi_cols);
  a.vert = is_partition_allowed(r, c, bsize, PartitionType::kVert,
                                chroma_offset, has_chroma, mi_rows, mi_cols);
  a.count = (a.none ? 1 : 0) + (a.horz ? 1 : 0) + (a.vert ? 1 : 0);
  if (a.count == 0) {
    a.none = true;
    a.count = 1;
  }
  return a;
}

struct ImpliedPartition {
  bool implied = false;
  PartitionType partition = PartitionType::kNone;
};

ImpliedPartition partition_implied_at_boundary(int r, int c, BlockSize bsize,
                                               int mi_rows, int mi_cols) {
  ImpliedPartition out;
  const int num_w = partition::num_4x4_blocks_wide(bsize);
  const int num_h = partition::num_4x4_blocks_high(bsize);
  const bool has_rows = (r + (num_h >> 1)) < mi_rows;
  const bool has_cols = (c + (num_w >> 1)) < mi_cols;
  if (has_rows && has_cols) return out;
  PartitionType implied = PartitionType::kNone;
  if (num_w == num_h) {
    implied = has_rows ? PartitionType::kVert : PartitionType::kHorz;
  } else if (num_h > num_w) {
    if (!has_rows) {
      implied = PartitionType::kHorz;
    } else {
      const bool sub_has_cols = (c + (num_w >> 2)) < mi_cols;
      if (num_w >= 4 && !sub_has_cols) implied = PartitionType::kHorz;
    }
  } else {
    if (!has_cols) {
      implied = PartitionType::kVert;
    } else {
      const bool sub_has_rows = (r + (num_h >> 2)) < mi_rows;
      if (num_h >= 4 && !sub_has_rows) implied = PartitionType::kVert;
    }
  }
  out.implied = implied != PartitionType::kNone;
  out.partition = implied;
  return out;
}

ImpliedPartition partition_implied(int r, int c, BlockSize bsize, int mi_rows,
                                   int mi_cols) {
  ImpliedPartition out;
  if (bsize == BlockSize::k4x4 || static_cast<int>(bsize) >= 25) return out;
  return partition_implied_at_boundary(r, c, bsize, mi_rows, mi_cols);
}

int do_split_context(const TileContext& ctx, int r, int c, BlockSize bsize) {
  const int bsw = std::max(partition::mi_width_log2(bsize), 1);
  const int bsh = std::max(partition::mi_height_log2(bsize), 1);
  const int ctx1 = partition::mi_height_log2(
                       static_cast<BlockSize>(ctx.left_mi[r])) < bsh
                       ? 1
                       : 0;
  const int ctx2 = partition::mi_width_log2(
                       static_cast<BlockSize>(ctx.above_mi[c])) < bsw
                       ? 1
                       : 0;
  return partition::partition_size_adjust(bsize) * 4 + ctx1 * 2 + ctx2;
}

int rect_type_context(const TileContext& ctx, int r, int c, BlockSize bsize) {
  const int bsw = std::max(partition::mi_width_log2(bsize), 1);
  const int bsh = std::max(partition::mi_height_log2(bsize), 1);
  const int ctx1 = partition::mi_height_log2(
                       static_cast<BlockSize>(ctx.left_mi[r])) < bsh
                       ? 1
                       : 0;
  const int ctx2 = partition::mi_width_log2(
                       static_cast<BlockSize>(ctx.above_mi[c])) < bsw
                       ? 1
                       : 0;
  return partition::partition_size_adjust_rect_type(bsize) * 4 + ctx1 * 2 +
         ctx2;
}

bool rect_type_implied(BlockSize bsize, PartitionType* rect) {
  const auto implied = partition::rect_type_implied_by_bsize(bsize);
  if (!implied.implied) return false;
  *rect = implied.partition;
  return true;
}

// ---------------------------------------------------------------------------
// Coefficient syntax helpers
// ---------------------------------------------------------------------------

int tx_size_ctx(int w, int h) {
  const int log2w = floor_log2(w);
  const int log2h = floor_log2(h);
  const int sqr = std::min(log2w, log2h) - 2;
  const int sqr_up = std::max(log2w, log2h) - 2;
  return (sqr + sqr_up + 1) >> 1;
}

int eob_multisize(int w, int h) {
  return std::min(floor_log2(w), 5) + std::min(floor_log2(h), 5) - 4;
}

void build_scan(int w, int h, std::vector<int>* out) {
  out->assign(static_cast<std::size_t>(w) * h, 0);
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    (*out)[c] = y * w + x;
    x += 1;
    y -= 1;
    if (y < 0 || x >= w) {
      x += 1;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
}

int eob_pt_from_eob(int eob) {
  if (eob <= 1) return 1;
  if (eob == 2) return 2;
  return floor_log2(eob - 1) + 2;
}

int eob_from_eob_pt(int eob_pt) {
  return eob_pt < 2 ? eob_pt : ((1 << (eob_pt - 2)) + 1);
}

int lf_base_levels(bool is_lf) { return is_lf ? 4 : 2; }
int lf_base_max(bool is_lf) { return is_lf ? 5 : 3; }
int max_level_for(bool is_lf, int plane) {
  if (is_lf) return plane == 0 ? 8 : 5;
  return 6;
}

int dc_sign_context(const TileContext& ctx, int x4, int y4, int w4, int h4) {
  int sign = 0;
  for (int k = 0; k < w4; ++k) {
    const int c = x4 + k;
    if (c < ctx.g->mi_cols) {
      const int s = ctx.above_dc[0][static_cast<std::size_t>(c)];
      if (s == 1) --sign;
      else if (s == 2) ++sign;
    }
  }
  for (int k = 0; k < h4; ++k) {
    const int rr = y4 + k;
    if (rr < ctx.g->mi_rows) {
      const int s = ctx.left_dc[0][static_cast<std::size_t>(rr)];
      if (s == 1) --sign;
      else if (s == 2) ++sign;
    }
  }
  return sign < 0 ? 1 : (sign > 0 ? 2 : 0);
}

struct CoeffResult {
  int eob = 0;
  int cul_level = 0;
  int dc_category = 0;
};

int all_zero_context(const TileContext& ctx, int plane, int x4, int y4, int w4,
                     int h4, int bsize_w, int bsize_h, int w, int h,
                     bool fsc_mode, int eob_u) {
  if (plane == 0) {
    int top = 0;
    int left = 0;
    const int max_x = ctx.g->mi_cols;
    const int max_y = ctx.g->mi_rows;
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < max_x) top |= ctx.above_level[0][x4 + k];
    }
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < max_y) left |= ctx.left_level[0][y4 + k];
    }
    top = std::min(top, 4);
    left = std::min(left, 4);
    if (fsc_mode) return 9;
    if (bsize_w == w && bsize_h == h) return 0;
    return (top + left + 3) >> 1;
  }
  int above = 0;
  int left = 0;
  const int max_x = ctx.g->chroma_mi_cols;
  const int max_y = ctx.g->chroma_mi_rows;
  for (int i = 0; i < w4; ++i) {
    if (x4 + i < max_x) {
      above |= ctx.above_level[plane][x4 + i];
      above |= ctx.above_dc[plane][x4 + i];
    }
  }
  for (int i = 0; i < h4; ++i) {
    if (y4 + i < max_y) {
      left |= ctx.left_level[plane][y4 + i];
      left |= ctx.left_dc[plane][y4 + i];
    }
  }
  int c = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
  if (plane == 2) {
    if (bsize_w * bsize_h > w * h) c += 3;
    if (eob_u != 0) c += 6;
  } else {
    c += 6;
  }
  return c;
}

void clear_coeff_context(TileContext& ctx, int plane, int x4, int y4, int w4,
                         int h4) {
  for (int i = 0; i < w4; ++i) {
    ctx.above_level[plane][x4 + i] = 0;
    ctx.above_dc[plane][x4 + i] = 0;
  }
  for (int i = 0; i < h4; ++i) {
    ctx.left_level[plane][y4 + i] = 0;
    ctx.left_dc[plane][y4 + i] = 0;
  }
}

void update_coeff_context(TileContext& ctx, int plane, int x4, int y4, int w4,
                          int h4, int cul_level, int dc_category) {
  for (int i = 0; i < w4; ++i) {
    ctx.above_level[plane][x4 + i] = static_cast<std::uint8_t>(cul_level);
    ctx.above_dc[plane][x4 + i] = static_cast<std::uint8_t>(dc_category);
  }
  for (int i = 0; i < h4; ++i) {
    ctx.left_level[plane][y4 + i] = static_cast<std::uint8_t>(cul_level);
    ctx.left_dc[plane][y4 + i] = static_cast<std::uint8_t>(dc_category);
  }
}

bool write_eob(CdfState& c, SymWriter& sw, int eob, int plane, int w, int h,
               bool is_inter) {
  const int multisize = eob_multisize(w, h);
  const int eob_ctx = plane > 0 ? 2 : (is_inter ? 1 : 0);
  std::uint16_t* cdf = nullptr;
  int symbols = 0;
  switch (multisize) {
    case 0:
      cdf = &c.coeff.eob_pt_16[eob_ctx][0];
      symbols = 5;
      break;
    case 1:
      cdf = &c.coeff.eob_pt_32[eob_ctx][0];
      symbols = 6;
      break;
    case 2:
      cdf = &c.coeff.eob_pt_64[eob_ctx][0];
      symbols = 7;
      break;
    case 3:
      cdf = &c.coeff.eob_pt_128[eob_ctx][0];
      symbols = 8;
      break;
    case 4:
      cdf = &c.coeff.eob_pt_256[eob_ctx][0];
      symbols = 8;
      break;
    case 5:
      cdf = &c.coeff.eob_pt_512[eob_ctx][0];
      symbols = 8;
      break;
    default:
      cdf = &c.coeff.eob_pt_1024[eob_ctx][0];
      symbols = 8;
      break;
  }
  int eob_pt = eob_pt_from_eob(eob);
  int symbol = eob_pt - 1;
  int extra_escape = 0;
  if (eob_pt > symbols) {
    symbol = symbols - 1;
    extra_escape = eob_pt - symbols;
  }
  if (!sw.sym(cdf, symbols, symbol)) return false;
  if (multisize >= 4 && symbol == symbols - 1) {
    if (multisize == 4) {
      if (!sw.bit((extra_escape & 1) != 0)) return false;
    } else {
      if (!sw.bit((extra_escape & 2) != 0)) return false;
      if (!sw.bit((extra_escape & 1) != 0)) return false;
    }
  }
  if (eob_pt >= 3) {
    const int base = eob_from_eob_pt(eob_pt);
    const int extra = eob - base;
    const int extra_bit = (extra >> (eob_pt - 3)) & 1;
    if (!sw.sym(&c.coeff.eob_extra[0], 2, extra_bit)) return false;
    for (int i = eob_pt - 4; i >= 0; --i) {
      if (!sw.bit(((extra >> i) & 1) != 0)) return false;
    }
  }
  return true;
}

bool read_eob(CdfState& c, SymReader& r, int plane, int w, int h, bool is_inter,
              int* eob) {
  const int multisize = eob_multisize(w, h);
  const int eob_ctx = plane > 0 ? 2 : (is_inter ? 1 : 0);
  std::uint16_t* cdf = nullptr;
  int symbols = 0;
  switch (multisize) {
    case 0:
      cdf = &c.coeff.eob_pt_16[eob_ctx][0];
      symbols = 5;
      break;
    case 1:
      cdf = &c.coeff.eob_pt_32[eob_ctx][0];
      symbols = 6;
      break;
    case 2:
      cdf = &c.coeff.eob_pt_64[eob_ctx][0];
      symbols = 7;
      break;
    case 3:
      cdf = &c.coeff.eob_pt_128[eob_ctx][0];
      symbols = 8;
      break;
    case 4:
      cdf = &c.coeff.eob_pt_256[eob_ctx][0];
      symbols = 8;
      break;
    case 5:
      cdf = &c.coeff.eob_pt_512[eob_ctx][0];
      symbols = 8;
      break;
    default:
      cdf = &c.coeff.eob_pt_1024[eob_ctx][0];
      symbols = 8;
      break;
  }
  int symbol = 0;
  if (!r.sym(cdf, symbols, &symbol)) return false;
  int eob_pt = symbol + 1;
  if (multisize >= 4 && symbol == symbols - 1) {
    int escape = 0;
    bool b = false;
    if (multisize == 4) {
      if (!r.bit(&b)) return false;
      escape = b ? 1 : 0;
    } else {
      if (!r.bit(&b)) return false;
      escape = b ? 2 : 0;
      if (!r.bit(&b)) return false;
      escape |= b ? 1 : 0;
    }
    eob_pt = symbols + escape;
  }
  int value = eob_from_eob_pt(eob_pt);
  if (eob_pt >= 3) {
    int extra = 0;
    if (!r.sym(&c.coeff.eob_extra[0], 2, &extra)) return false;
    if (extra) value += 1 << (eob_pt - 3);
    for (int i = eob_pt - 4; i >= 0; --i) {
      bool bb = false;
      if (!r.bit(&bb)) return false;
      if (bb) value += 1 << i;
    }
  }
  *eob = value;
  return true;
}

bool write_quant_escape(SymWriter& sw, int x, int& hr_level_avg) {
  const int pred_level = hr_level_avg;
  const int m = clip3(1, 6, pred_level <= 0 ? 0 : floor_log2(pred_level));
  const int k = m + 1;
  const int c_max = std::min(m + 4, 6);
  if (x < (c_max << m)) {
    const int q = x >> m;
    for (int i = 0; i < q; ++i) {
      if (!sw.bit(false)) return false;
    }
    if (!sw.bit(true)) return false;
    if (!sw.literal(m, static_cast<std::uint32_t>(x & ((1 << m) - 1)))) {
      return false;
    }
  } else {
    for (int i = 0; i < c_max; ++i) {
      if (!sw.bit(false)) return false;
    }
    int g = 0;
    while (true) {
      const int length = k + g;
      const int x_base = (c_max << m) + (1 << length) - (1 << k);
      if (x < x_base + (1 << length)) {
        for (int i = 0; i < g; ++i) {
          if (!sw.bit(false)) return false;
        }
        if (!sw.bit(true)) return false;
        if (!sw.literal(length, static_cast<std::uint32_t>(x - x_base))) {
          return false;
        }
        break;
      }
      ++g;
    }
  }
  hr_level_avg = (x + hr_level_avg) >> 1;
  return true;
}

bool read_quant_escape(SymReader& r, int& hr_level_avg, int* x) {
  const int pred_level = hr_level_avg;
  const int m = clip3(1, 6, pred_level <= 0 ? 0 : floor_log2(pred_level));
  const int k = m + 1;
  const int c_max = std::min(m + 4, 6);
  int q = 0;
  while (q < c_max) {
    bool b = false;
    if (!r.bit(&b)) return false;
    if (b) break;
    ++q;
  }
  int length = 0;
  int x_base = 0;
  if (q == c_max) {
    length = -1;
    while (true) {
      bool b = false;
      if (!r.bit(&b)) return false;
      ++length;
      if (b) break;
    }
    length += k;
    x_base = (q << m) + (1 << length) - (1 << k);
  } else {
    length = m;
    x_base = q << m;
  }
  std::uint32_t rem = 0;
  if (!r.literal(length, &rem)) return false;
  const int value = x_base + static_cast<int>(rem);
  hr_level_avg = (value + hr_level_avg) >> 1;
  *x = value;
  return true;
}

// The §5.20.8.2 lossless inter transform-type symbol.  Only TX_4X4 codes it,
// and it uses Default_Lossless_Inter_Tx_Type_Cdf (two symbols).
bool write_lossless_inter_tx_type(CdfState& c, SymWriter& sw, int plane, int w,
                                  int h, bool is_inter, bool fsc_mode,
                                  bool tx_idtx) {
  if (plane != 0 || !is_inter || fsc_mode || w != 4 || h != 4) return true;
  return sw.sym(&c.lossless_inter_tx_type[0], 2, tx_idtx ? 1 : 0);
}

bool read_lossless_inter_tx_type(CdfState& c, SymReader& r, int plane, int w,
                                 int h, bool is_inter, bool fsc_mode,
                                 bool* tx_idtx) {
  if (plane != 0 || !is_inter || fsc_mode || w != 4 || h != 4) return true;
  int symbol = 0;
  if (!r.sym(&c.lossless_inter_tx_type[0], 2, &symbol)) return false;
  *tx_idtx = symbol != 0;
  return true;
}

// Normal (non-FSC) coefficient writer.  Works for DCT_DCT and IDTX because the
// coefficient syntax is the same; only the transform-type bit differs.
CoeffResult write_coeff_normal(CdfState& c, SymWriter& sw, TileContext& ctx,
                               int plane, int x4, int y4, int w4, int h4,
                               int bsize_w, int bsize_h, int w, int h,
                               const int* levels, bool is_inter, bool fsc_mode,
                               bool tx_idtx, int eob_u) {
  CoeffResult out;
  const int tsc = tx_size_ctx(w, h);
  const int seg_eob = std::min(32, w) * std::min(32, h);
  std::vector<int> scan;
  build_scan(w, h, &scan);
  int eob = 0;
  for (int i = seg_eob - 1; i >= 0; --i) {
    if (levels[scan[i]] != 0) {
      eob = i + 1;
      break;
    }
  }
  const int ctx0 = all_zero_context(ctx, plane, x4, y4, w4, h4, bsize_w,
                                    bsize_h, w, h, fsc_mode, eob_u);
  std::uint16_t* skip_cdf =
      plane == 2 ? &c.coeff.v_txb_skip[ctx0][0]
                 : &c.coeff.txb_skip[(is_inter || fsc_mode) ? 1 : 0][tsc]
                                   [ctx0][0];
  if (!sw.sym(skip_cdf, 2, eob == 0 ? 1 : 0)) return out;
  if (eob == 0) {
    clear_coeff_context(ctx, plane, x4, y4, w4, h4);
    return out;
  }
  if (!write_eob(c, sw, eob, plane, w, h, is_inter)) return out;
  if (!write_lossless_inter_tx_type(c, sw, plane, w, h, is_inter, fsc_mode,
                                   tx_idtx)) {
    return out;
  }

  std::vector<int> level_pre(static_cast<std::size_t>(w) * h, 0);
  int hr_level_avg = 0;

  for (int ci = eob - 1; ci >= 0; --ci) {
    const int pos = scan[ci];
    const int row = pos / w;
    const int col = pos % w;
    const bool lf = plane == 0 ? ((row + col) < 4) : ((row + col) < 1);
    const int base_levels = lf_base_levels(lf);
    const int base_max = lf_base_max(lf);
    const int br_max = (lf && plane > 0) ? 0 : 3;
    const int pre_max = base_max + br_max;
    const int absv = abs_int(levels[pos]);
    int base_level = 0;
    int br_level = 0;
    if (absv <= base_levels) {
      base_level = absv;
    } else {
      base_level = std::min(base_levels + 1, base_max);
      br_level = std::min(absv, pre_max) - base_level;
    }

    int ctx_eob = 0;
    if (ci != 0) {
      if (ci <= seg_eob / 8) ctx_eob = 1;
      else if (ci <= seg_eob / 4) ctx_eob = 2;
      else ctx_eob = 3;
    }

    if (ci == eob - 1) {
      const int symbol = base_level - 1;
      if (plane == 0) {
        if (lf) {
          if (!sw.sym(&c.coeff.coeff_base_lf_eob[tsc][ctx_eob][0], 5,
                      symbol)) return out;
        } else {
          if (!sw.sym(&c.coeff.coeff_base_eob[tsc][ctx_eob][0], 3, symbol)) {
            return out;
          }
        }
      } else {
        if (lf) {
          if (!sw.sym(&c.coeff.coeff_base_lf_eob_uv[ctx_eob][0], 5, symbol)) {
            return out;
          }
        } else {
          if (!sw.sym(&c.coeff.coeff_base_eob_uv[ctx_eob][0], 3, symbol)) {
            return out;
          }
        }
      }
    } else {
      int mag = 0;
      const int num_ref = plane == 0 ? 5 : 3;
      for (int idx = 0; idx < num_ref; ++idx) {
        const int rr = row + kSigRefDiff[idx][0];
        const int cc = col + kSigRefDiff[idx][1];
        if (rr < h && cc < w) {
          const int limit = lf ? 5 : 3;
          mag += std::min(level_pre[rr * w + cc], limit);
        }
      }
      const int mctx = (mag + 1) >> 1;
      if (plane == 0) {
        if (lf) {
          int lf_ctx = 0;
          if (ci == 0) lf_ctx = std::min(mctx, 8);
          else if (row + col < 2) lf_ctx = std::min(mctx, 6) + 9;
          else lf_ctx = std::min(mctx, 4) + 16;
          if (!sw.sym(&c.coeff.coeff_base_lf[tsc][lf_ctx][0][0], 6,
                      base_level)) return out;
        } else {
          const int ctx2 = std::min(mctx, 4);
          int hf_ctx = ctx2;
          if (row + col >= 6 && row + col < 8) hf_ctx = ctx2 + 5;
          else if (row + col >= 8) hf_ctx = ctx2 + 10;
          if (!sw.sym(&c.coeff.coeff_base[tsc][hf_ctx][0][0], 4, base_level)) {
            return out;
          }
        }
      } else {
        const int ctx2 = std::min(mctx, 3);
        const int uv_ctx = plane == 1 ? ctx2 : ctx2 + 4;
        if (lf) {
          if (!sw.sym(&c.coeff.coeff_base_lf_uv[uv_ctx][0], 6, base_level)) {
            return out;
          }
        } else {
          if (!sw.sym(&c.coeff.coeff_base_uv[uv_ctx][0], 4, base_level)) {
            return out;
          }
        }
      }
    }

    const int pre_level = base_level + br_level;
    if (pre_level > base_levels && !(lf && plane > 0)) {
      int mag = 0;
      for (int idx = 0; idx < 3; ++idx) {
        const int rr = row + kMagRef[idx][0];
        const int cc = col + kMagRef[idx][1];
        if (rr < h && cc < w) {
          mag += std::min(level_pre[rr * w + cc], 5);
        }
      }
      mag = std::min((mag + 1) >> 1, 6);
      if (plane == 0) {
        int br_ctx = mag;
        if (lf) {
          if (pos != 0) br_ctx = mag + 7;
          if (!sw.sym(&c.coeff.coeff_br_lf[br_ctx][0], 4, br_level)) {
            return out;
          }
        } else {
          if (!sw.sym(&c.coeff.coeff_br[mag][0], 4, br_level)) return out;
        }
      } else {
        if (!sw.sym(&c.coeff.coeff_br_uv[std::min(mag, 3)][0], 4, br_level)) {
          return out;
        }
      }
    }
    level_pre[pos] = pre_level;
  }

  for (int ci = eob - 1; ci >= 0; --ci) {
    const int pos = scan[ci];
    const int row = pos / w;
    const int col = pos % w;
    const int pre_level = level_pre[pos];
    const int absv = abs_int(levels[pos]);
    const bool negative = levels[pos] < 0;
    if (pre_level != 0) {
      if (row == 0 && col == 0 && plane == 0) {
        const int sign_ctx = dc_sign_context(ctx, x4, y4, w4, h4);
        if (!sw.sym(&c.coeff.dc_sign[0][0][sign_ctx][0], 2,
                    negative ? 1 : 0)) return out;
      } else {
        if (!sw.bit(negative)) return out;
      }
      const bool lf = plane == 0 ? ((row + col) < 4) : ((row + col) < 1);
      const int max_level = max_level_for(lf, plane);
      if (pre_level >= max_level) {
        if (!write_quant_escape(sw, absv - pre_level, hr_level_avg)) {
          return out;
        }
      }
      out.cul_level = std::min(4, out.cul_level + absv);
      if (pos == 0) out.dc_category = negative ? 1 : 2;
    }
  }
  out.eob = eob;
  update_coeff_context(ctx, plane, x4, y4, w4, h4, out.cul_level,
                       out.dc_category);
  return out;
}

// FSC luma coefficient writer (§5.20.7.27 `useFsc` path).  Generic for every
// lossless transform size, 4x4..32x32, rectangular included.
CoeffResult write_coeff_fsc(CdfState& c, SymWriter& sw, TileContext& ctx,
                            int x4, int y4, int w4, int h4, int bsize_w,
                            int bsize_h, int w, int h, const int* levels,
                            bool is_inter, bool fsc_mode, bool tx_idtx) {
  CoeffResult out;
  const int tsc = tx_size_ctx(w, h);
  const int seg_eob = std::min(32, w) * std::min(32, h);
  std::vector<int> scan;
  build_scan(w, h, &scan);
  int first = -1;
  for (int i = 0; i < seg_eob; ++i) {
    if (levels[scan[i]] != 0) {
      first = i;
      break;
    }
  }
  const int ctx0 = all_zero_context(ctx, 0, x4, y4, w4, h4, bsize_w, bsize_h,
                                    w, h, fsc_mode, 0);
  const bool all_zero = first < 0;
  if (!sw.sym(&c.coeff.txb_skip[1][tsc][ctx0][0], 2, all_zero ? 1 : 0)) {
    return out;
  }
  if (all_zero) {
    clear_coeff_context(ctx, 0, x4, y4, w4, h4);
    return out;
  }
  const int bob = first;
  const int eob = seg_eob - bob;
  if (!write_eob(c, sw, eob, 0, w, h, is_inter)) return out;
  if (!write_lossless_inter_tx_type(c, sw, 0, w, h, is_inter, fsc_mode, tx_idtx)) {
    return out;
  }

  std::vector<int> level_grid(static_cast<std::size_t>(w) * h, 0);
  std::vector<int> sign_grid(static_cast<std::size_t>(w) * h, 0);
  int hr_level_avg = 0;
  for (int ci = bob; ci < seg_eob; ++ci) {
    const int pos = scan[ci];
    const int row = pos / w;
    const int col = pos % w;
    const int absv = abs_int(levels[pos]);
    int base_symbol = 0;
    int coarse = 0;
    if (ci == bob) {
      base_symbol = absv >= 3 ? 2 : std::max(0, absv - 1);
      coarse = base_symbol + 1;
      const int bob_ctx =
          bob <= (seg_eob >> 3) ? 0 : (bob <= (seg_eob >> 2) ? 1 : 2);
      if (!sw.sym(&c.coeff.coeff_base_bob[std::min(2, tsc)][bob_ctx][0], 3,
                  base_symbol)) {
        return out;
      }
    } else {
      int mag = 0;
      if (col > 0) mag += std::min(3, level_grid[row * w + col - 1]);
      if (row > 0) mag += std::min(3, level_grid[(row - 1) * w + col]);
      base_symbol = absv >= 3 ? 3 : absv;
      coarse = base_symbol;
      if (!sw.sym(&c.coeff.coeff_base_idtx[std::min(2, tsc)][mag][0], 4,
                  base_symbol)) {
        return out;
      }
    }
    if (coarse > 2) {
      int mag = 0;
      if (col > 0) mag += std::min(5, level_grid[row * w + col - 1]);
      if (row > 0) mag += std::min(5, level_grid[(row - 1) * w + col]);
      mag = std::min(mag, 6);
      const int br = std::min(3, std::max(0, absv - coarse));
      if (!sw.sym(&c.coeff.coeff_br_idtx[std::min(2, tsc)][mag][0], 4, br)) {
        return out;
      }
      coarse += br;
    }
    level_grid[row * w + col] = coarse;
  }

  for (int ci = 0; ci < seg_eob; ++ci) {
    const int pos = scan[ci];
    const int row = pos / w;
    const int col = pos % w;
    const int coarse = level_grid[row * w + col];
    if (coarse == 0) continue;
    int signc = 0;
    if (col > 0) signc += sign_grid[row * w + col - 1];
    if (row > 0) signc += sign_grid[(row - 1) * w + col];
    if (col > 0 && row > 0) signc += sign_grid[(row - 1) * w + col - 1];
    int sign_ctx = 0;
    if (signc > 2) sign_ctx = 5;
    else if (signc < -2) sign_ctx = 6;
    else if (signc > 0) sign_ctx = 1;
    else if (signc < 0) sign_ctx = 2;
    else sign_ctx = 0;
    if (coarse > 3 && sign_ctx != 0) sign_ctx += 2;
    const int absv = abs_int(levels[pos]);
    const bool negative = levels[pos] < 0;
    if (!sw.sym(&c.coeff.idtx_sign[std::min(2, tsc)][sign_ctx][0], 2,
                negative ? 1 : 0)) {
      return out;
    }
    sign_grid[row * w + col] = negative ? -1 : 1;
    if (absv >= 6) {
      if (!write_quant_escape(sw, absv - coarse, hr_level_avg)) return out;
    }
    out.cul_level = std::min(4, out.cul_level + absv);
    if (pos == 0) out.dc_category = negative ? 1 : 2;
  }
  out.eob = eob;
  update_coeff_context(ctx, 0, x4, y4, w4, h4, out.cul_level,
                       out.dc_category);
  return out;
}

// Mode helpers
// ---------------------------------------------------------------------------

int uv_mode_index(int uv_mode, int y_mode) {
  static constexpr int kDefaultUv[kIntraModesCount] = {
      kDcPred, 9, 10, 11, 12, kV, kH, 3, 4, 8, 5, 6, 7};
  int index = 0;
  if (y_mode_is_directional(y_mode)) {
    if (uv_mode == y_mode) return 0;
    index = 1;
  }
  for (int i = 0; i < kIntraModesCount; ++i) {
    const int mode = kDefaultUv[i];
    if (mode == y_mode && y_mode_is_directional(y_mode)) continue;
    if (mode == uv_mode) return index;
    ++index;
  }
  return 0;
}

int uv_mode_from_index(int idx, int y_mode) {
  static constexpr int kDefaultUv[kIntraModesCount] = {
      kDcPred, 9, 10, 11, 12, kV, kH, 3, 4, 8, 5, 6, 7};
  if (y_mode_is_directional(y_mode)) {
    if (idx == 0) return y_mode;
    --idx;
  }
  int count = 0;
  for (int i = 0; i < kIntraModesCount; ++i) {
    const int mode = kDefaultUv[i];
    if (mode == y_mode && y_mode_is_directional(y_mode)) continue;
    if (count == idx) return mode;
    ++count;
  }
  return kDcPred;
}

int fsc_context(const TileContext& ctx, int r, int c, int w4, int h4,
                int sb_size_mi, bool frame_is_intra) {
  // §8.3.2: inter frames (no intra region) use the fixed context 3.
  if (!frame_is_intra) return 3;
  int nnum = 0;
  int ctx_sum = 0;
  auto add = [&](int nr, int nc) {
    if (nnum >= 2) return;
    if (!ctx.inside(nr, nc)) return;
    const int sb_shift = floor_log2(sb_size_mi);
    if ((r >> sb_shift) != (nr >> sb_shift)) return;
    ctx_sum += ctx.fsc_modes[ctx.cell(nr, nc)];
    ++nnum;
  };
  add(r + h4 - 1, c - 1);
  add(r - 1, c + w4 - 1);
  add(r, c - 1);
  add(r - 1, c);
  return ctx_sum;
}

// Combined reader for the §5.20.7.27 coefficient block.  The FSC path is
// selected only after the eob and (for TX_4X4 inter) the
// lossless_inter_tx_type bit have been read, because useFsc depends on the
// resulting PlaneTxType.
bool read_coeff_block(CdfState& c, SymReader& r, TileContext& ctx, int plane,
                      int x4, int y4, int w4, int h4, int bsize_w,
                      int bsize_h, int w, int h, bool is_inter, bool fsc_mode,
                      bool enable_fsc, bool* tx_idtx, int eob_u,
                      int* levels_raster, CoeffResult* out) {
  std::fill(levels_raster, levels_raster + w * h, 0);
  *out = {};
  const int tsc = tx_size_ctx(w, h);
  const int seg_eob = std::min(32, w) * std::min(32, h);
  std::vector<int> scan;
  build_scan(w, h, &scan);
  const int ctx0 = all_zero_context(ctx, plane, x4, y4, w4, h4, bsize_w,
                                    bsize_h, w, h, fsc_mode, eob_u);
  std::uint16_t* skip_cdf =
      plane == 2 ? &c.coeff.v_txb_skip[ctx0][0]
                 : &c.coeff.txb_skip[(is_inter || fsc_mode) ? 1 : 0][tsc]
                                   [ctx0][0];
  int all_zero = 0;
  if (!r.sym(skip_cdf, 2, &all_zero)) return false;
  if (all_zero) {
    clear_coeff_context(ctx, plane, x4, y4, w4, h4);
    return true;
  }
  int eob = 0;
  if (!read_eob(c, r, plane, w, h, is_inter, &eob)) return false;
  if (eob <= 0 || eob > seg_eob) return false;
  if (!read_lossless_inter_tx_type(c, r, plane, w, h, is_inter, fsc_mode,
                                  tx_idtx)) {
    return false;
  }
  const bool use_fsc =
      enable_fsc && *tx_idtx && plane == 0 && (fsc_mode || is_inter);

  if (use_fsc) {
    const int bob = seg_eob - eob;
    std::vector<int> level_grid(static_cast<std::size_t>(w) * h, 0);
    std::vector<int> sign_grid(static_cast<std::size_t>(w) * h, 0);
    int hr_level_avg = 0;
    for (int ci = bob; ci < seg_eob; ++ci) {
      const int pos = scan[ci];
      const int row = pos / w;
      const int col = pos % w;
      int coarse = 0;
      if (ci == bob) {
        int symbol = 0;
        const int bob_ctx =
            bob <= (seg_eob >> 3) ? 0 : (bob <= (seg_eob >> 2) ? 1 : 2);
        if (!r.sym(&c.coeff.coeff_base_bob[std::min(2, tsc)][bob_ctx][0], 3,
                   &symbol)) return false;
        coarse = symbol + 1;
      } else {
        int mag = 0;
        if (col > 0) mag += std::min(3, level_grid[row * w + col - 1]);
        if (row > 0) mag += std::min(3, level_grid[(row - 1) * w + col]);
        int symbol = 0;
        if (!r.sym(&c.coeff.coeff_base_idtx[std::min(2, tsc)][mag][0], 4,
                   &symbol)) return false;
        coarse = symbol;
      }
      if (coarse > 2) {
        int mag = 0;
        if (col > 0) mag += std::min(5, level_grid[row * w + col - 1]);
        if (row > 0) mag += std::min(5, level_grid[(row - 1) * w + col]);
        mag = std::min(mag, 6);
        int br = 0;
        if (!r.sym(&c.coeff.coeff_br_idtx[std::min(2, tsc)][mag][0], 4, &br)) {
          return false;
        }
        coarse += br;
      }
      level_grid[row * w + col] = coarse;
    }
    for (int ci = 0; ci < seg_eob; ++ci) {
      const int pos = scan[ci];
      const int row = pos / w;
      const int col = pos % w;
      const int coarse = level_grid[row * w + col];
      if (coarse == 0) continue;
      int signc = 0;
      if (col > 0) signc += sign_grid[row * w + col - 1];
      if (row > 0) signc += sign_grid[(row - 1) * w + col];
      if (col > 0 && row > 0) signc += sign_grid[(row - 1) * w + col - 1];
      int sign_ctx = 0;
      if (signc > 2) sign_ctx = 5;
      else if (signc < -2) sign_ctx = 6;
      else if (signc > 0) sign_ctx = 1;
      else if (signc < 0) sign_ctx = 2;
      else sign_ctx = 0;
      if (coarse > 3 && sign_ctx != 0) sign_ctx += 2;
      int sign = 0;
      if (!r.sym(&c.coeff.idtx_sign[std::min(2, tsc)][sign_ctx][0], 2,
                 &sign)) return false;
      const bool negative = sign != 0;
      sign_grid[row * w + col] = negative ? -1 : 1;
      int absv = coarse;
      if (coarse >= 6) {
        int x = 0;
        if (!read_quant_escape(r, hr_level_avg, &x)) return false;
        absv += x;
      }
      levels_raster[pos] = negative ? -absv : absv;
      out->cul_level = std::min(4, out->cul_level + absv);
      if (pos == 0) out->dc_category = negative ? 1 : 2;
    }
    out->eob = eob;
    update_coeff_context(ctx, 0, x4, y4, w4, h4, out->cul_level,
                         out->dc_category);
    return true;
  }

  std::vector<int> level_pre(static_cast<std::size_t>(w) * h, 0);
  int hr_level_avg = 0;
  for (int ci = eob - 1; ci >= 0; --ci) {
    const int pos = scan[ci];
    const int row = pos / w;
    const int col = pos % w;
    const bool lf = plane == 0 ? ((row + col) < 4) : ((row + col) < 1);
    const int base_levels = lf_base_levels(lf);
    int base_level = 0;
    int ctx_eob = 0;
    if (ci != 0) {
      if (ci <= seg_eob / 8) ctx_eob = 1;
      else if (ci <= seg_eob / 4) ctx_eob = 2;
      else ctx_eob = 3;
    }
    if (ci == eob - 1) {
      int symbol = 0;
      if (plane == 0) {
        if (lf) {
          if (!r.sym(&c.coeff.coeff_base_lf_eob[tsc][ctx_eob][0], 5,
                     &symbol)) return false;
        } else {
          if (!r.sym(&c.coeff.coeff_base_eob[tsc][ctx_eob][0], 3, &symbol)) {
            return false;
          }
        }
      } else {
        if (lf) {
          if (!r.sym(&c.coeff.coeff_base_lf_eob_uv[ctx_eob][0], 5, &symbol)) {
            return false;
          }
        } else {
          if (!r.sym(&c.coeff.coeff_base_eob_uv[ctx_eob][0], 3, &symbol)) {
            return false;
          }
        }
      }
      base_level = symbol + 1;
    } else {
      int mag = 0;
      const int num_ref = plane == 0 ? 5 : 3;
      for (int idx = 0; idx < num_ref; ++idx) {
        const int rr = row + kSigRefDiff[idx][0];
        const int cc = col + kSigRefDiff[idx][1];
        if (rr < h && cc < w) {
          const int limit = lf ? 5 : 3;
          mag += std::min(level_pre[rr * w + cc], limit);
        }
      }
      const int mctx = (mag + 1) >> 1;
      int symbol = 0;
      if (plane == 0) {
        if (lf) {
          int lf_ctx = 0;
          if (ci == 0) lf_ctx = std::min(mctx, 8);
          else if (row + col < 2) lf_ctx = std::min(mctx, 6) + 9;
          else lf_ctx = std::min(mctx, 4) + 16;
          if (!r.sym(&c.coeff.coeff_base_lf[tsc][lf_ctx][0][0], 6, &symbol)) {
            return false;
          }
        } else {
          const int ctx2 = std::min(mctx, 4);
          int hf_ctx = ctx2;
          if (row + col >= 6 && row + col < 8) hf_ctx = ctx2 + 5;
          else if (row + col >= 8) hf_ctx = ctx2 + 10;
          if (!r.sym(&c.coeff.coeff_base[tsc][hf_ctx][0][0], 4, &symbol)) {
            return false;
          }
        }
      } else {
        const int ctx2 = std::min(mctx, 3);
        const int uv_ctx = plane == 1 ? ctx2 : ctx2 + 4;
        if (lf) {
          if (!r.sym(&c.coeff.coeff_base_lf_uv[uv_ctx][0], 6, &symbol)) {
            return false;
          }
        } else {
          if (!r.sym(&c.coeff.coeff_base_uv[uv_ctx][0], 4, &symbol)) {
            return false;
          }
        }
      }
      base_level = symbol;
    }
    int pre_level = base_level;
    if (base_level > base_levels && !(lf && plane > 0)) {
      int mag = 0;
      for (int idx = 0; idx < 3; ++idx) {
        const int rr = row + kMagRef[idx][0];
        const int cc = col + kMagRef[idx][1];
        if (rr < h && cc < w) {
          mag += std::min(level_pre[rr * w + cc], 5);
        }
      }
      mag = std::min((mag + 1) >> 1, 6);
      int br = 0;
      if (plane == 0) {
        int br_ctx = mag;
        if (lf) {
          if (pos != 0) br_ctx = mag + 7;
          if (!r.sym(&c.coeff.coeff_br_lf[br_ctx][0], 4, &br)) return false;
        } else {
          if (!r.sym(&c.coeff.coeff_br[mag][0], 4, &br)) return false;
        }
      } else {
        if (!r.sym(&c.coeff.coeff_br_uv[std::min(mag, 3)][0], 4, &br)) {
          return false;
        }
      }
      pre_level += br;
    }
    level_pre[pos] = pre_level;
  }

  for (int ci = eob - 1; ci >= 0; --ci) {
    const int pos = scan[ci];
    const int row = pos / w;
    const int col = pos % w;
    const int pre_level = level_pre[pos];
    if (pre_level == 0) continue;
    const bool lf = plane == 0 ? ((row + col) < 4) : ((row + col) < 1);
    bool negative = false;
    if (row == 0 && col == 0 && plane == 0) {
      const int sign_ctx = dc_sign_context(ctx, x4, y4, w4, h4);
      int sign = 0;
      if (!r.sym(&c.coeff.dc_sign[0][0][sign_ctx][0], 2, &sign)) {
        return false;
      }
      negative = sign != 0;
    } else {
      if (!r.bit(&negative)) return false;
    }
    const int max_level = max_level_for(lf, plane);
    int absv = pre_level;
    if (pre_level >= max_level) {
      int x = 0;
      if (!read_quant_escape(r, hr_level_avg, &x)) return false;
      absv += x;
    }
    levels_raster[pos] = negative ? -absv : absv;
    out->cul_level = std::min(4, out->cul_level + absv);
    if (pos == 0) out->dc_category = negative ? 1 : 2;
  }
  out->eob = eob;
  update_coeff_context(ctx, plane, x4, y4, w4, h4, out->cul_level,
                       out->dc_category);
  return true;
}

// ---------------------------------------------------------------------------
// Lossless tile writer
// ---------------------------------------------------------------------------

struct BlockPlan {
  bool is_inter = false;
  // Two-reference B block: codes comp_mode = COMPOUND_REFERENCE and blends
  // ref0/ref1 with COMPOUND_AVERAGE (GLOBAL_GLOBALMV, zero MVs).
  bool compound = false;
  int y_mode = kDcPred;
  bool use_dpcm_y = false;
  bool dpcm_y_vertical = false;
  bool fsc = false;
  int uv_mode = kDcPred;
  bool use_dpcm_uv = false;
  bool dpcm_uv_vertical = false;
  int tx_w = 4;
  int tx_h = 4;
  bool tx_idtx = true;
  bool signal_tx = false;
  bool lossless_tx_size = false;
};

struct PredictionScratch {
  std::array<std::uint16_t, 1024> pred{};
  std::array<int, 1024> residual{};
  std::array<int, 1024> tmp{};
  std::array<int, 1024> levels{};
};

class TileWriter {
 public:
  TileWriter(const Geometry& g, const Options& opt, bool keyframe,
             CdfState& cdfs, TileContext& ctx, Plane* src[3], Plane* recon[3],
             const Plane* ref[3], const Plane* ref1[3], int num_refs,
             const int frame_distance[2], FrameStats* stats)
      : g_(g), opt_(opt), keyframe_(keyframe), cdfs_(cdfs), ctx_(ctx),
        stats_(stats), num_refs_(num_refs) {
    for (int p = 0; p < 3; ++p) {
      src_[p] = src[p];
      recon_[p] = recon[p];
      ref_[p] = ref[p];
      ref1_[p] = ref1[p];
    }
    ctx_.num_total_refs = num_refs;
    ctx_.frame_distance[0] = frame_distance[0];
    ctx_.frame_distance[1] = frame_distance[1];
  }

  bool encode_tile(std::uint8_t* out, std::size_t capacity,
                   std::size_t* tile_bytes, std::string& error) {
    if (!w_.init(out, capacity, opt_.adaptive_cdf, &error)) return false;
    const int sb_mi = 16;
    for (int sb_r = 0; sb_r < g_.sb_rows; ++sb_r) {
      ctx_.clear_left();
      for (int sb_c = 0; sb_c < g_.sb_cols; ++sb_c) {
        const int r = sb_r * sb_mi;
        const int c = sb_c * sb_mi;
        if (r >= g_.mi_rows || c >= g_.mi_cols) continue;
        if (!write_partition(r, c, BlockSize::k64x64, 0, true, &error)) {
          return false;
        }
      }
    }
    if (!w_.finish(tile_bytes)) {
      error = w_.error();
      return false;
    }
    return true;
  }

 private:
  bool full_block(int r, int c, BlockSize bsize) const {
    return r + partition::num_4x4_blocks_high(bsize) <= g_.mi_rows &&
           c + partition::num_4x4_blocks_wide(bsize) <= g_.mi_cols;
  }

  bool write_partition(int r, int c, BlockSize bsize, int chroma_offset,
                       bool has_chroma, std::string* error) {
    if (r >= g_.mi_rows || c >= g_.mi_cols) return true;
    const AllowedPartitions allowed =
        init_allowed(r, c, bsize, chroma_offset, has_chroma, g_.mi_rows,
                     g_.mi_cols);
    const ImpliedPartition implied =
        partition_implied(r, c, bsize, g_.mi_rows, g_.mi_cols);
    const int num_w = partition::num_4x4_blocks_wide(bsize);
    const int num_h = partition::num_4x4_blocks_high(bsize);
    const bool full = full_block(r, c, bsize);
    PartitionType part = PartitionType::kNone;
    bool emit = false;

    if (implied.implied &&
        ((implied.partition == PartitionType::kNone && allowed.none) ||
         (implied.partition == PartitionType::kHorz && allowed.horz) ||
         (implied.partition == PartitionType::kVert && allowed.vert))) {
      part = implied.partition;
    } else if (full && allowed.none &&
               partition::block_width(bsize) <= opt_.block_size &&
               partition::block_height(bsize) <= opt_.block_size) {
      part = PartitionType::kNone;
      emit = allowed.count > 1;
    } else if (allowed.count == 1) {
      part = allowed.none
                 ? PartitionType::kNone
                 : (allowed.horz ? PartitionType::kHorz : PartitionType::kVert);
    } else {
      part = num_h > num_w ? PartitionType::kHorz : PartitionType::kVert;
      if (part == PartitionType::kHorz && !allowed.horz) {
        part = PartitionType::kVert;
      } else if (part == PartitionType::kVert && !allowed.vert) {
        part = PartitionType::kHorz;
      }
      if ((part == PartitionType::kHorz && !allowed.horz) ||
          (part == PartitionType::kVert && !allowed.vert)) {
        *error = "lossless inter: no legal partition";
        return false;
      }
      emit = true;
    }

    if (emit) {
      if (allowed.none) {
        const int dc = do_split_context(ctx_, r, c, bsize);
        if (!w_.sym(&cdfs_.do_split[dc][0], 2,
                    part == PartitionType::kNone ? 0 : 1)) {
          *error = w_.error();
          return false;
        }
      }
      if (part == PartitionType::kHorz || part == PartitionType::kVert) {
        PartitionType rect = PartitionType::kNone;
        if (!rect_type_implied(bsize, &rect)) {
          const int symbol = part == PartitionType::kHorz ? 0 : 1;
          const int rc = rect_type_context(ctx_, r, c, bsize);
          if (!w_.sym(&cdfs_.rect_type[rc][0],
                      2, symbol)) {
            *error = w_.error();
            return false;
          }
        }
      }
    }

    if (part == PartitionType::kNone) return write_block(r, c, bsize, error);

    const int half_w = num_w >> 1;
    const int half_h = num_h >> 1;
    const BlockSize sub = partition::partition_subsize(part, bsize);
    const int child_chroma_offset =
        (!chroma_offset && has_chroma &&
         is_chroma_offset_for_partition(part, bsize))
            ? 1
            : chroma_offset;
    if (part == PartitionType::kHorz) {
      if (!write_partition(r, c, sub, child_chroma_offset, has_chroma, error))
        return false;
      if (!write_partition(r + half_h, c, sub, child_chroma_offset, has_chroma,
                           error))
        return false;
    } else {
      if (!write_partition(r, c, sub, child_chroma_offset, has_chroma, error))
        return false;
      if (!write_partition(r, c + half_w, sub, child_chroma_offset, has_chroma,
                           error))
        return false;
    }
    return true;
  }

  BlockPlan choose_plan(int r, int c, BlockSize bsize) {
    (void)r;
    (void)c;
    BlockPlan plan;
    const int bw = partition::block_width(bsize);
    const int bh = partition::block_height(bsize);
    plan.is_inter =
        !keyframe_ && (opt_.intra_period <= 0 ||
                       (block_index_ % opt_.intra_period) != 0);
    // A two-reference B frame blends both sides on every inter leaf.  Blocks
    // smaller than 8x8 cannot signal comp_mode (§5.20.5.9
    // is_comp_ref_allowed) and stay single-reference; this writer's minimum
    // leaf is 8x8.
    plan.compound = plan.is_inter && num_refs_ > 1 && bw >= 8 && bh >= 8;
    if (!plan.is_inter) {
      const int m = block_index_ % 3;
      if (opt_.enable_dpcm && m == 1) {
        plan.y_mode = kV;
        plan.use_dpcm_y = true;
        plan.dpcm_y_vertical = true;
        plan.uv_mode = kV;
        plan.use_dpcm_uv = true;
        plan.dpcm_uv_vertical = true;
      } else if (opt_.enable_dpcm && m == 2) {
        plan.y_mode = kH;
        plan.use_dpcm_y = true;
        plan.dpcm_y_vertical = false;
        plan.uv_mode = kH;
        plan.use_dpcm_uv = true;
        plan.dpcm_uv_vertical = false;
      } else {
        plan.y_mode = kDcPred;
        plan.uv_mode = kDcPred;
      }
      plan.fsc = opt_.enable_fsc &&
                 (opt_.force_intra_fsc || ((block_index_ & 1) == 0)) &&
                 bw <= 32 && bh <= 32;
      if (plan.fsc) {
        TxSizeDecision d =
            select_tx_size(bw, bh, false, true, true, true);
        plan.tx_w = d.coded_w;
        plan.tx_h = d.coded_h;
        plan.signal_tx = d.signal_lossless_tx_size;
        plan.lossless_tx_size = d.lossless_tx_size;
        plan.tx_idtx = true;
      } else {
        plan.tx_w = 4;
        plan.tx_h = 4;
        plan.tx_idtx = false;
      }
      return plan;
    }

    const bool want_large =
        opt_.inter_4x4_only
            ? false
            : (opt_.alternate_inter_tx ? ((block_index_ & 1) != 0) : true);
    TxSizeDecision d = select_tx_size(bw, bh, true, false, true, want_large);
    plan.tx_w = d.coded_w;
    plan.tx_h = d.coded_h;
    plan.signal_tx = d.signal_lossless_tx_size;
    plan.lossless_tx_size = d.lossless_tx_size;
    plan.tx_idtx = true;
    return plan;
  }

  bool write_intra_mode(int r, int c, BlockSize bsize, const BlockPlan& plan,
                        std::string* error) {
    const int w4 = partition::num_4x4_blocks_wide(bsize);
    const int h4 = partition::num_4x4_blocks_high(bsize);
    const int sb_mi = 16;
    if (plan.use_dpcm_y) {
      if (!w_.sym(&cdfs_.use_dpcm_y[0], 2, 1)) {
        *error = w_.error();
        return false;
      }
      if (!w_.sym(&cdfs_.dpcm_mode_y[0], 2, plan.dpcm_y_vertical ? 0 : 1)) {
        *error = w_.error();
        return false;
      }
    } else {
      if (!w_.sym(&cdfs_.use_dpcm_y[0], 2, 0)) {
        *error = w_.error();
        return false;
      }
      const int jctx =
          (ctx_.joint_mode(r, c, w4, h4, 0) >= 5 ? 1 : 0) +
          (ctx_.joint_mode(r, c, w4, h4, 1) >= 5 ? 1 : 0);
      if (!w_.sym(&cdfs_.y_mode_set[0], 4, 0)) {
        *error = w_.error();
        return false;
      }
      if (!w_.sym(&cdfs_.y_mode_index[jctx][0], 8, 0)) {
        *error = w_.error();
        return false;
      }
    }
    if (opt_.enable_fsc) {
      const int fctx = fsc_context(ctx_, r, c, w4, h4, sb_mi, keyframe_);
      const int group = kFscBsizeGroups[static_cast<int>(bsize)];
      if (!w_.sym(&cdfs_.fsc_mode[fctx][group][0], 2, plan.fsc ? 1 : 0)) {
        *error = w_.error();
        return false;
      }
    }
    if (g_.has_chroma) {
      if (plan.use_dpcm_uv) {
        if (!w_.sym(&cdfs_.use_dpcm_uv[0], 2, 1)) {
          *error = w_.error();
          return false;
        }
        if (!w_.sym(&cdfs_.dpcm_mode_uv[0], 2,
                    plan.dpcm_uv_vertical ? 0 : 1)) {
          *error = w_.error();
          return false;
        }
      } else {
        if (!w_.sym(&cdfs_.use_dpcm_uv[0], 2, 0)) {
          *error = w_.error();
          return false;
        }
        const int uv_ctx = y_mode_is_directional(plan.y_mode) ? 1 : 0;
        const int idx = uv_mode_index(kDcPred, plan.y_mode);
        if (idx <= 6) {
          if (!w_.sym(&cdfs_.uv_mode[uv_ctx][0], 8, idx)) {
            *error = w_.error();
            return false;
          }
        } else {
          if (!w_.sym(&cdfs_.uv_mode[uv_ctx][0], 8, 7)) {
            *error = w_.error();
            return false;
          }
          if (!w_.literal(3, static_cast<std::uint32_t>(idx - 7))) {
            *error = w_.error();
            return false;
          }
        }
      }
    }
    if (plan.signal_tx) {
      const int group = kSizeGroup[static_cast<int>(bsize)];
      if (!w_.sym(&cdfs_.lossless_tx_size[group][0][0], 2,
                  plan.lossless_tx_size ? 1 : 0)) {
        *error = w_.error();
        return false;
      }
    }
    return true;
  }

  bool write_inter_mode(int r, int c, BlockSize bsize, const BlockPlan& plan,
                        std::string* error) {
    const int w4 = partition::num_4x4_blocks_wide(bsize);
    const int h4 = partition::num_4x4_blocks_high(bsize);
    const int sctx = ctx_.skip_context(r, c, w4, h4);
    if (!w_.sym(&cdfs_.skip[sctx][0], 2, 0)) {
      *error = w_.error();
      return false;
    }
    const int bw = partition::block_width(bsize);
    const int bh = partition::block_height(bsize);
    if (num_refs_ > 1) {
      // §5.20.7.10 read_ref_frames(): NumTotalRefs == 2.  comp_mode is coded
      // only when is_comp_ref_allowed() (8x8 or thin 4xN/Nx4).
      if (bw >= 8 && bh >= 8) {
        const int cmctx = ctx_.comp_mode_context(r, c, w4, h4);
        if (!w_.sym(&cdfs_.comp_mode[cmctx][0], 2,
                    plan.compound ? 1 : 0)) {
          *error = w_.error();
          return false;
        }
      }
      const int rctx = ctx_.ref_count_context(r, c, w4, h4);
      if (plan.compound) {
        // §5.20.7.11: NumTotalRefs == 2 and NumSameRefCompound == 0 hit the
        // `ref + 1 == NumTotalRefs - 1` branch, which infers comp_ref = 1 for
        // the (0, 1) pair and codes no symbol.
        (void)rctx;
        // §5.20.7.6: non-joint GLOBAL_GLOBALMV (compound_mode_non_joint = 3).
        const int jctx = ctx_.is_joint_context(ctx_.frame_distance[0],
                                               ctx_.frame_distance[1]);
        if (!w_.sym(&cdfs_.is_joint[jctx][0], 2, 0)) {
          *error = w_.error();
          return false;
        }
        const int mctx = ctx_.new_mv_context(r, c, w4, h4);
        if (!w_.sym(&cdfs_.compound_mode_non_joint[mctx][0], 5, 3)) {
          *error = w_.error();
          return false;
        }
      } else {
        // §5.20.7.12 read_single_ref(): one symbol selects list 0.
        if (!w_.sym(&cdfs_.single_ref[rctx][0][0], 2, 1)) {
          *error = w_.error();
          return false;
        }
        const int mctx = ctx_.new_mv_context(r, c, w4, h4);
        if (!w_.sym(&cdfs_.single_mode[mctx][0], 3, 1)) {  // GLOBALMV
          *error = w_.error();
          return false;
        }
      }
    } else {
      const int mctx = ctx_.new_mv_context(r, c, w4, h4);
      if (!w_.sym(&cdfs_.single_mode[mctx][0], 3, 1)) {  // GLOBALMV
        *error = w_.error();
        return false;
      }
    }
    if (plan.signal_tx) {
      const int group = kSizeGroup[static_cast<int>(bsize)];
      if (!w_.sym(&cdfs_.lossless_tx_size[group][0][0], 2,
                  plan.lossless_tx_size ? 1 : 0)) {
        *error = w_.error();
        return false;
      }
    }
    return true;
  }

  bool write_block(int r, int c, BlockSize bsize, std::string* error) {
    const BlockPlan plan = choose_plan(r, c, bsize);
    if (!keyframe_) {
      const int w4 = partition::num_4x4_blocks_wide(bsize);
      const int h4 = partition::num_4x4_blocks_high(bsize);
      const int ictx = ctx_.is_inter_context(r, c, w4, h4);
      if (!w_.sym(&cdfs_.is_inter[ictx][0], 2, plan.is_inter ? 1 : 0)) {
        *error = w_.error();
        return false;
      }
    }
    if (plan.is_inter) {
      if (!write_inter_mode(r, c, bsize, plan, error)) return false;
    } else {
      if (!write_intra_mode(r, c, bsize, plan, error)) return false;
    }

    const int w4 = partition::num_4x4_blocks_wide(bsize);
    const int h4 = partition::num_4x4_blocks_high(bsize);
    const int joint = plan.use_dpcm_y ? (plan.dpcm_y_vertical ? 22 : 50) : 0;
    for (int y = 0; y < h4; ++y) {
      for (int x = 0; x < w4; ++x) {
        const std::size_t idx = ctx_.cell(r + y, c + x);
        ctx_.intra_joint[idx] = static_cast<std::uint8_t>(joint);
        ctx_.fsc_modes[idx] = static_cast<std::uint8_t>(plan.fsc ? 1 : 0);
        ctx_.is_inter[idx] = static_cast<std::uint8_t>(plan.is_inter ? 1 : 0);
        ctx_.ymode[idx] = static_cast<std::uint8_t>(
            plan.is_inter ? (plan.compound ? 22 : 15) : plan.y_mode);
        ctx_.ref0[idx] =
            static_cast<std::uint8_t>(plan.is_inter ? 0 : 0xff);
        ctx_.ref1[idx] = static_cast<std::uint8_t>(
            plan.is_inter && plan.compound ? 1 : 0xff);
        ctx_.skip[idx] = 0;
        ctx_.left_mi[r + y] = static_cast<int>(bsize);
        ctx_.above_mi[c + x] = static_cast<int>(bsize);
      }
    }

    std::vector<std::uint8_t> luma_idtx;
    if (!write_luma(r, c, bsize, plan, &luma_idtx, error)) return false;
    if (!write_chroma(r, c, bsize, plan, luma_idtx, error)) return false;

    if (plan.is_inter) ++stats_->inter_blocks;
    else ++stats_->intra_blocks;
    if (plan.is_inter) {
      if (plan.lossless_tx_size) ++stats_->lossless_tx_size_1;
      else ++stats_->lossless_tx_size_0;
    }
    if (plan.fsc) ++stats_->fsc_luma_blocks;
    if (plan.use_dpcm_y) ++stats_->dpcm_y_blocks;
    if (plan.use_dpcm_uv) ++stats_->dpcm_uv_blocks;
    ++block_index_;
    return true;
  }

  bool write_luma(int r, int c, BlockSize bsize, const BlockPlan& plan,
                  std::vector<std::uint8_t>* luma_idtx, std::string* error) {
    const int bw = partition::block_width(bsize);
    const int bh = partition::block_height(bsize);
    const int x = c * 4;
    const int y = r * 4;
    const int tw4 = bw / 4;
    const int th4 = bh / 4;
    luma_idtx->assign(static_cast<std::size_t>(tw4) * th4, 0);
    PredictionScratch s;

    auto code_one = [&](int tx, int ty, int w, int h, bool idtx,
                        bool dpcm, bool vertical) -> bool {
      const int px = x + tx;
      const int py = y + ty;
      if (plan.is_inter) {
        if (plan.compound) {
          predict_inter_compound(*ref_[0], *ref1_[0], px, py, w, h,
                                 opt_.bit_depth, s.pred.data(), w);
        } else {
          predict_inter_zero(*ref_[0], px, py, w, h, s.pred.data(), w);
        }
      } else if (!predict_intra_block(*recon_[0], px, py, w, h, plan.y_mode,
                                      px > 0, py > 0, opt_.bit_depth, s.pred.data(), w)) {
        *error = "luma intra prediction failed";
        return false;
      }
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          const int src = src_[0]->data[static_cast<std::size_t>(py + i) *
                                            src_[0]->w + (px + j)];
          s.residual[i * w + j] = src - s.pred[i * w + j];
        }
      }
      if (dpcm) {
        dpcm_difference(s.residual.data(), w, h, vertical, s.tmp.data());
      } else {
        std::memcpy(s.tmp.data(), s.residual.data(),
                    static_cast<std::size_t>(w) * h * sizeof(int));
      }
      forward_lossless_block(s.tmp.data(), w, h, !idtx, s.levels.data());
      const int x4 = px >> 2;
      const int y4 = py >> 2;
      const bool use_fsc = !plan.is_inter && plan.fsc;
      const bool fsc_syntax =
          opt_.enable_fsc && idtx && (plan.fsc || plan.is_inter);
      if (fsc_syntax) {
        write_coeff_fsc(cdfs_, w_, ctx_, x4, y4, w >> 2, h >> 2, bw, bh, w, h,
                        s.levels.data(), plan.is_inter, use_fsc, idtx);
      } else {
        write_coeff_normal(cdfs_, w_, ctx_, 0, x4, y4, w >> 2, h >> 2, bw, bh,
                           w, h, s.levels.data(), plan.is_inter, use_fsc, idtx,
                           0);
      }
      if (!w_.ok()) {
        *error = w_.error();
        return false;
      }
      reconstruct_block(s.levels.data(), w, h, !idtx, dpcm, vertical,
                        s.pred.data(), w,
                        recon_[0]->data.data() +
                            static_cast<std::size_t>(py) * recon_[0]->w + px,
                        recon_[0]->w, opt_.bit_depth);
      return true;
    };

    if (plan.is_inter && plan.tx_w == 4 && plan.tx_h == 4) {
      int sub = 0;
      for (int ty = 0; ty < bh; ty += 4) {
        for (int tx = 0; tx < bw; tx += 4) {
          const bool idtx = opt_.inter_4x4_only
                                ? opt_.inter_4x4_idtx
                                : (opt_.alternate_inter_tx
                                       ? (((sub + block_index_) & 1) != 0)
                                       : true);
          if (!code_one(tx, ty, 4, 4, idtx, false, false)) return false;
          // §5.20.7.27: an all-zero luma block codes no lossless_inter_tx_type
          // symbol and the decoder infers TxTypes = DCT_DCT.  Store the
          // effective type so the chroma tx type derivation matches.
          bool nonzero = false;
          for (int q = 0; q < 16; ++q) nonzero |= s.levels[q] != 0;
          const bool effective_idtx = nonzero && idtx;
          (*luma_idtx)[static_cast<std::size_t>(ty / 4) * tw4 + tx / 4] =
              effective_idtx ? 1 : 0;
          if (effective_idtx) ++stats_->idtx_4x4;
          else ++stats_->dct_dct_4x4;
          ++sub;
        }
      }
      return true;
    }

    if (!plan.is_inter && !plan.fsc) {
      for (int ty = 0; ty < bh; ty += 4) {
        for (int tx = 0; tx < bw; tx += 4) {
          if (!code_one(tx, ty, 4, 4, false, plan.use_dpcm_y,
                        plan.dpcm_y_vertical)) {
            return false;
          }
        }
      }
      return true;
    }

    const int tw = plan.tx_w;
    const int th = plan.tx_h;
    const bool idtx = plan.tx_idtx;
    if (!code_one(0, 0, tw, th, idtx,
                  !plan.is_inter && plan.use_dpcm_y,
                  plan.dpcm_y_vertical)) {
      return false;
    }
    for (std::size_t i = 0; i < luma_idtx->size(); ++i) {
      (*luma_idtx)[i] = idtx ? 1 : 0;
    }
    return true;
  }

  bool write_chroma(int r, int c, BlockSize bsize, const BlockPlan& plan,
                    const std::vector<std::uint8_t>& luma_idtx,
                    std::string* error) {
    if (!g_.has_chroma) return true;
    const int bw = partition::block_width(bsize);
    const int bh = partition::block_height(bsize);
    const int cw = bw >> g_.sub_x;
    const int ch = bh >> g_.sub_y;
    const int x = (c * 4) >> g_.sub_x;
    const int y = (r * 4) >> g_.sub_y;
    const int luma_w4 = bw / 4;
    PredictionScratch s;
    int eob_u = 0;
    for (int plane = 1; plane < 3; ++plane) {
      for (int cy = 0; cy < ch; cy += 4) {
        for (int cx = 0; cx < cw; cx += 4) {
          const int px = x + cx;
          const int py = y + cy;
          bool idtx = false;
          if (plan.is_inter) {
            if (plan.tx_w != 4 || plan.tx_h != 4) {
              idtx = true;
            } else {
              const int lx = (cx << g_.sub_x) >> 2;
              const int ly = (cy << g_.sub_y) >> 2;
              const std::size_t li =
                  static_cast<std::size_t>(ly) * luma_w4 + lx;
              idtx = li < luma_idtx.size() && luma_idtx[li] != 0;
            }
          } else {
            idtx = plan.fsc;
          }
          if (plan.is_inter) {
            if (plan.compound) {
              predict_inter_compound(*ref_[plane], *ref1_[plane], px, py, 4, 4,
                                     opt_.bit_depth, s.pred.data(), 4);
            } else {
              predict_inter_zero(*ref_[plane], px, py, 4, 4, s.pred.data(), 4);
            }
          } else if (!predict_intra_block(*recon_[plane], px, py, 4, 4,
                                          plan.uv_mode, px > 0, py > 0,
                                          opt_.bit_depth, s.pred.data(), 4)) {
            *error = "chroma intra prediction failed";
            return false;
          }
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              const int src = src_[plane]->data[static_cast<std::size_t>(py + i) *
                                                    src_[plane]->w + (px + j)];
              s.residual[i * 4 + j] = src - s.pred[i * 4 + j];
            }
          }
          const bool dpcm = !plan.is_inter && plan.use_dpcm_uv;
          if (dpcm) {
            dpcm_difference(s.residual.data(), 4, 4, plan.dpcm_uv_vertical,
                            s.tmp.data());
          } else {
            std::memcpy(s.tmp.data(), s.residual.data(),
                        16 * sizeof(int));
          }
          forward_lossless_block(s.tmp.data(), 4, 4, !idtx, s.levels.data());
          const int prev_eob_u = ctx_.eob_u;
          const CoeffResult cr = write_coeff_normal(
              cdfs_, w_, ctx_, plane, px >> 2, py >> 2, 1, 1, cw, ch, 4, 4,
              s.levels.data(), plan.is_inter, plan.fsc, idtx,
              plane == 2 ? prev_eob_u : 0);
          if (!w_.ok()) {
            *error = w_.error();
            return false;
          }
          if (plane == 1) {
            eob_u = cr.eob;
            ctx_.eob_u = eob_u;
          }
          reconstruct_block(s.levels.data(), 4, 4, !idtx, dpcm,
                            plan.dpcm_uv_vertical, s.pred.data(), 4,
                            recon_[plane]->data.data() +
                                static_cast<std::size_t>(py) * recon_[plane]->w +
                                px,
                            recon_[plane]->w, opt_.bit_depth);
        }
      }
    }
    ctx_.eob_u = 0;
    return true;
  }

  const Geometry& g_;
  const Options& opt_;
  bool keyframe_;
  CdfState& cdfs_;
  TileContext& ctx_;
  FrameStats* stats_;
  Plane* src_[3] = {};
  Plane* recon_[3] = {};
  const Plane* ref_[3] = {};
  const Plane* ref1_[3] = {};
  int num_refs_ = 1;
  int block_index_ = 0;
  SymWriter w_{};
};

// ---------------------------------------------------------------------------
// Lossless tile reader
// ---------------------------------------------------------------------------

class TileReader {
 public:
  TileReader(const Geometry& g, const Options& opt, bool keyframe,
             CdfState& cdfs, TileContext& ctx, Plane* recon[3],
             const Plane* ref[3], const Plane* ref1[3], int num_refs,
             const int frame_distance[2], FrameStats* stats)
      : g_(g), opt_(opt), keyframe_(keyframe), cdfs_(cdfs), ctx_(ctx),
        stats_(stats), num_refs_(num_refs) {
    for (int p = 0; p < 3; ++p) {
      recon_[p] = recon[p];
      ref_[p] = ref[p];
      ref1_[p] = ref1[p];
    }
    ctx_.num_total_refs = num_refs;
    ctx_.frame_distance[0] = frame_distance[0];
    ctx_.frame_distance[1] = frame_distance[1];
  }

  bool decode_tile(const std::uint8_t* data, std::size_t size, std::size_t sz,
                   std::string& error) {
    if (!r_.init(data, size, sz, opt_.adaptive_cdf, &error)) return false;
    const int sb_mi = 16;
    for (int sb_r = 0; sb_r < g_.sb_rows; ++sb_r) {
      ctx_.clear_left();
      for (int sb_c = 0; sb_c < g_.sb_cols; ++sb_c) {
        const int r = sb_r * sb_mi;
        const int c = sb_c * sb_mi;
        if (r >= g_.mi_rows || c >= g_.mi_cols) continue;
        if (!read_partition(r, c, BlockSize::k64x64, 0, true, &error)) {
          return false;
        }
      }
    }
    if (!r_.finish()) {
      error = r_.error();
      return false;
    }
    return true;
  }

 private:

  bool read_partition(int r, int c, BlockSize bsize, int chroma_offset,
                      bool has_chroma, std::string* error) {
    if (r >= g_.mi_rows || c >= g_.mi_cols) return true;
    const AllowedPartitions allowed =
        init_allowed(r, c, bsize, chroma_offset, has_chroma, g_.mi_rows,
                     g_.mi_cols);
    const ImpliedPartition implied =
        partition_implied(r, c, bsize, g_.mi_rows, g_.mi_cols);
    const int num_w = partition::num_4x4_blocks_wide(bsize);
    const int num_h = partition::num_4x4_blocks_high(bsize);
    PartitionType part = PartitionType::kNone;
    bool need_symbols = false;

    if (implied.implied &&
        ((implied.partition == PartitionType::kNone && allowed.none) ||
         (implied.partition == PartitionType::kHorz && allowed.horz) ||
         (implied.partition == PartitionType::kVert && allowed.vert))) {
      part = implied.partition;
    } else if (allowed.count == 1) {
      part = allowed.none
                 ? PartitionType::kNone
                 : (allowed.horz ? PartitionType::kHorz : PartitionType::kVert);
    } else {
      need_symbols = true;
    }

    if (need_symbols) {
      int do_split = 0;
      if (allowed.none) {
        const int dc = do_split_context(ctx_, r, c, bsize);
        if (!r_.sym(&cdfs_.do_split[dc][0], 2, &do_split)) {
          *error = r_.error();
          return false;
        }
        if (!do_split) part = PartitionType::kNone;
      }
      if (do_split || !allowed.none) {
        PartitionType rect = PartitionType::kNone;
        if (rect_type_implied(bsize, &rect)) {
          part = rect;
        } else {
          int symbol = 0;
          const int rc = rect_type_context(ctx_, r, c, bsize);
          if (!r_.sym(&cdfs_.rect_type[rc][0], 2, &symbol)) {
            *error = r_.error();
            return false;
          }
          part = symbol == 0 ? PartitionType::kHorz : PartitionType::kVert;
        }
      }
    }

    if (part == PartitionType::kNone) return read_block(r, c, bsize, error);

    const int half_w = num_w >> 1;
    const int half_h = num_h >> 1;
    const BlockSize sub = partition::partition_subsize(part, bsize);
    const int child_chroma_offset =
        (!chroma_offset && has_chroma &&
         is_chroma_offset_for_partition(part, bsize))
            ? 1
            : chroma_offset;
    if (part == PartitionType::kHorz) {
      if (!read_partition(r, c, sub, child_chroma_offset, has_chroma, error))
        return false;
      if (!read_partition(r + half_h, c, sub, child_chroma_offset, has_chroma,
                          error))
        return false;
    } else {
      if (!read_partition(r, c, sub, child_chroma_offset, has_chroma, error))
        return false;
      if (!read_partition(r, c + half_w, sub, child_chroma_offset, has_chroma,
                          error))
        return false;
    }
    return true;
  }

  bool read_intra_mode(int r, int c, BlockSize bsize, BlockPlan* plan,
                       std::string* error) {
    const int w4 = partition::num_4x4_blocks_wide(bsize);
    const int h4 = partition::num_4x4_blocks_high(bsize);
    const int sb_mi = 16;
    int use_dpcm_y = 0;
    if (!r_.sym(&cdfs_.use_dpcm_y[0], 2, &use_dpcm_y)) {
      *error = r_.error();
      return false;
    }
    if (use_dpcm_y) {
      int mode = 0;
      if (!r_.sym(&cdfs_.dpcm_mode_y[0], 2, &mode)) {
        *error = r_.error();
        return false;
      }
      plan->use_dpcm_y = true;
      plan->dpcm_y_vertical = mode == 0;
      plan->y_mode = plan->dpcm_y_vertical ? kV : kH;
    } else {
      int set = 0;
      if (!r_.sym(&cdfs_.y_mode_set[0], 4, &set) || set != 0) {
        *error = "lossless inter reader: non-DC y_mode_set unsupported";
        return false;
      }
      const int jctx =
          (ctx_.joint_mode(r, c, w4, h4, 0) >= 5 ? 1 : 0) +
          (ctx_.joint_mode(r, c, w4, h4, 1) >= 5 ? 1 : 0);
      int index = 0;
      if (!r_.sym(&cdfs_.y_mode_index[jctx][0], 8, &index) || index != 0) {
        *error = "lossless inter reader: non-DC y_mode unsupported";
        return false;
      }
      plan->y_mode = kDcPred;
    }
    plan->fsc = false;
    if (opt_.enable_fsc) {
      const int fctx = fsc_context(ctx_, r, c, w4, h4, sb_mi, keyframe_);
      const int group = kFscBsizeGroups[static_cast<int>(bsize)];
      int symbol = 0;
      if (!r_.sym(&cdfs_.fsc_mode[fctx][group][0], 2, &symbol)) {
        *error = r_.error();
        return false;
      }
      plan->fsc = symbol != 0;
    }
    if (g_.has_chroma) {
      int use_dpcm_uv = 0;
      if (!r_.sym(&cdfs_.use_dpcm_uv[0], 2, &use_dpcm_uv)) {
        *error = r_.error();
        return false;
      }
      if (use_dpcm_uv) {
        int mode = 0;
        if (!r_.sym(&cdfs_.dpcm_mode_uv[0], 2, &mode)) {
          *error = r_.error();
          return false;
        }
        plan->use_dpcm_uv = true;
        plan->dpcm_uv_vertical = mode == 0;
        plan->uv_mode = plan->dpcm_uv_vertical ? kV : kH;
      } else {
        const int uv_ctx = y_mode_is_directional(plan->y_mode) ? 1 : 0;
        int idx = 0;
        if (!r_.sym(&cdfs_.uv_mode[uv_ctx][0], 8, &idx)) {
          *error = r_.error();
          return false;
        }
        if (idx == 7) {
          std::uint32_t extra = 0;
          if (!r_.literal(3, &extra)) {
            *error = r_.error();
            return false;
          }
          idx += static_cast<int>(extra);
        }
        plan->uv_mode = uv_mode_from_index(idx, plan->y_mode);
      }
    }
    if (plan->fsc) {
      const int group = kSizeGroup[static_cast<int>(bsize)];
      int bit = 0;
      if (!r_.sym(&cdfs_.lossless_tx_size[group][0][0], 2, &bit)) {
        *error = r_.error();
        return false;
      }
      plan->lossless_tx_size = bit != 0;
      plan->signal_tx = true;
      if (bit) {
        if (!find_tx_size(partition::block_width(bsize),
                          partition::block_height(bsize), &plan->tx_w,
                          &plan->tx_h)) {
          *error = "lossless inter reader: bad fsc tx size";
          return false;
        }
      } else {
        plan->tx_w = 4;
        plan->tx_h = 4;
      }
      plan->tx_idtx = true;
    } else {
      plan->tx_w = 4;
      plan->tx_h = 4;
      plan->tx_idtx = false;
    }
    return true;
  }

  bool read_block(int r, int c, BlockSize bsize, std::string* error) {
    BlockPlan plan;
    if (!keyframe_) {
      const int w4 = partition::num_4x4_blocks_wide(bsize);
      const int h4 = partition::num_4x4_blocks_high(bsize);
      int is_inter = 0;
      const int ictx = ctx_.is_inter_context(r, c, w4, h4);
      const int sctx = ctx_.skip_context(r, c, w4, h4);
      if (!r_.sym(&cdfs_.is_inter[ictx][0], 2, &is_inter)) {
        *error = r_.error();
        return false;
      }
      if (is_inter) {
        plan.is_inter = true;
        int skip = 0;
        if (!r_.sym(&cdfs_.skip[sctx][0], 2, &skip)) {
          *error = r_.error();
          return false;
        }
        if (skip) {
          *error = "lossless inter reader: skip_flag=1 not supported";
          return false;
        }
        const int bw = partition::block_width(bsize);
        const int bh = partition::block_height(bsize);
        if (num_refs_ > 1) {
          int comp_mode = 0;
          if (bw >= 8 && bh >= 8) {
            const int cmctx = ctx_.comp_mode_context(r, c, w4, h4);
            if (!r_.sym(&cdfs_.comp_mode[cmctx][0], 2, &comp_mode)) {
              *error = r_.error();
              return false;
            }
          }
          const int rctx = ctx_.ref_count_context(r, c, w4, h4);
          if (comp_mode) {
            plan.compound = true;
            // §5.20.7.11 with NumTotalRefs == 2, NumSameRefCompound == 0: the
            // (0, 1) pair is implied by the `ref + 1 == NumTotalRefs - 1`
            // branch and no comp_ref symbol is present.
            const int jctx = ctx_.is_joint_context(ctx_.frame_distance[0],
                                                   ctx_.frame_distance[1]);
            int is_joint = 0;
            if (!r_.sym(&cdfs_.is_joint[jctx][0], 2, &is_joint) ||
                is_joint != 0) {
              *error = "lossless inter reader: joint compound is not supported";
              return false;
            }
            const int mctx = ctx_.new_mv_context(r, c, w4, h4);
            int compound_mode = 0;
            if (!r_.sym(&cdfs_.compound_mode_non_joint[mctx][0], 5,
                        &compound_mode) ||
                compound_mode != 3) {
              *error = "lossless inter reader: only GLOBAL_GLOBALMV compound "
                       "is supported";
              return false;
            }
          } else {
            int single_ref = 0;
            if (!r_.sym(&cdfs_.single_ref[rctx][0][0], 2, &single_ref) ||
                single_ref != 1) {
              *error = "lossless inter reader: only reference list 0 is "
                       "supported for single prediction";
              return false;
            }
            const int mctx = ctx_.new_mv_context(r, c, w4, h4);
            int single_mode = 0;
            if (!r_.sym(&cdfs_.single_mode[mctx][0], 3, &single_mode) ||
                single_mode != 1) {
              *error = "lossless inter reader: only GLOBALMV is supported";
              return false;
            }
          }
        } else {
          const int mctx = ctx_.new_mv_context(r, c, w4, h4);
          int single_mode = 0;
          if (!r_.sym(&cdfs_.single_mode[mctx][0], 3, &single_mode)) {
            *error = r_.error();
            return false;
          }
          if (single_mode != 1) {
            *error = "lossless inter reader: only GLOBALMV is supported (r=" +
                     std::to_string(r) + " c=" + std::to_string(c) +
                     " mode=" + std::to_string(single_mode) + ")";
            return false;
          }
        }
        plan.signal_tx = partition::block_width(bsize) > 4 ||
                         partition::block_height(bsize) > 4;
        if (plan.signal_tx) {
          const int group = kSizeGroup[static_cast<int>(bsize)];
          int bit = 0;
          if (!r_.sym(&cdfs_.lossless_tx_size[group][0][0], 2, &bit)) {
            *error = r_.error();
            return false;
          }
          plan.lossless_tx_size = bit != 0;
          if (bit) {
            if (!find_tx_size(partition::block_width(bsize),
                              partition::block_height(bsize), &plan.tx_w,
                              &plan.tx_h)) {
              *error = "lossless inter reader: bad inter tx size";
              return false;
            }
          } else {
            plan.tx_w = 4;
            plan.tx_h = 4;
          }
        }
        plan.tx_idtx = true;
      } else if (!read_intra_mode(r, c, bsize, &plan, error)) {
        return false;
      }
    } else if (!read_intra_mode(r, c, bsize, &plan, error)) {
      return false;
    }

    const int w4 = partition::num_4x4_blocks_wide(bsize);
    const int h4 = partition::num_4x4_blocks_high(bsize);
    const int joint = plan.use_dpcm_y ? (plan.dpcm_y_vertical ? 22 : 50) : 0;
    for (int y = 0; y < h4; ++y) {
      for (int x = 0; x < w4; ++x) {
        const std::size_t idx = ctx_.cell(r + y, c + x);
        ctx_.intra_joint[idx] = static_cast<std::uint8_t>(joint);
        ctx_.fsc_modes[idx] = static_cast<std::uint8_t>(plan.fsc ? 1 : 0);
        ctx_.is_inter[idx] = static_cast<std::uint8_t>(plan.is_inter ? 1 : 0);
        ctx_.ymode[idx] = static_cast<std::uint8_t>(
            plan.is_inter ? (plan.compound ? 22 : 15) : plan.y_mode);
        ctx_.ref0[idx] =
            static_cast<std::uint8_t>(plan.is_inter ? 0 : 0xff);
        ctx_.ref1[idx] = static_cast<std::uint8_t>(
            plan.is_inter && plan.compound ? 1 : 0xff);
        ctx_.skip[idx] = 0;
        ctx_.left_mi[r + y] = static_cast<int>(bsize);
        ctx_.above_mi[c + x] = static_cast<int>(bsize);
      }
    }

    std::vector<std::uint8_t> luma_idtx;
    if (!read_luma(r, c, bsize, plan, &luma_idtx, error)) return false;
    if (!read_chroma(r, c, bsize, plan, luma_idtx, error)) return false;

    if (plan.is_inter) ++stats_->inter_blocks;
    else ++stats_->intra_blocks;
    if (plan.is_inter) {
      if (plan.lossless_tx_size) ++stats_->lossless_tx_size_1;
      else ++stats_->lossless_tx_size_0;
    }
    if (plan.fsc) ++stats_->fsc_luma_blocks;
    if (plan.use_dpcm_y) ++stats_->dpcm_y_blocks;
    if (plan.use_dpcm_uv) ++stats_->dpcm_uv_blocks;
    ++block_index_;
    return true;
  }

  bool read_luma(int r, int c, BlockSize bsize, const BlockPlan& plan,
                 std::vector<std::uint8_t>* luma_idtx, std::string* error) {
    const int bw = partition::block_width(bsize);
    const int bh = partition::block_height(bsize);
    const int x = c * 4;
    const int y = r * 4;
    const int tw4 = bw / 4;
    const int th4 = bh / 4;
    luma_idtx->assign(static_cast<std::size_t>(tw4) * th4, 0);
    PredictionScratch s;
    std::vector<int> levels(static_cast<std::size_t>(bw) * bh, 0);

    auto read_one = [&](int tx, int ty, int w, int h, bool* idtx,
                        bool dpcm, bool vertical) -> bool {
      const int px = x + tx;
      const int py = y + ty;
      if (plan.is_inter) {
        if (plan.compound) {
          predict_inter_compound(*ref_[0], *ref1_[0], px, py, w, h,
                                 opt_.bit_depth, s.pred.data(), w);
        } else {
          predict_inter_zero(*ref_[0], px, py, w, h, s.pred.data(), w);
        }
      } else if (!predict_intra_block(*recon_[0], px, py, w, h, plan.y_mode,
                                      px > 0, py > 0, opt_.bit_depth, s.pred.data(), w)) {
        *error = "luma intra prediction failed";
        return false;
      }
      CoeffResult cr;
      if (!read_coeff_block(cdfs_, r_, ctx_, 0, px >> 2, py >> 2, w >> 2,
                            h >> 2, bw, bh, w, h, plan.is_inter, plan.fsc,
                            opt_.enable_fsc, idtx, 0, levels.data(), &cr)) {
        *error = r_.error();
        return false;
      }
      // §5.20.7.27: all_zero leaves TxTypes at DCT_DCT; the inter tx-type
      // symbol is only present for a non-zero block.
      if (cr.eob == 0) *idtx = false;
      reconstruct_block(levels.data(), w, h, !*idtx, dpcm, vertical,
                        s.pred.data(), w,
                        recon_[0]->data.data() +
                            static_cast<std::size_t>(py) * recon_[0]->w + px,
                        recon_[0]->w, opt_.bit_depth);
      return true;
    };

    if (plan.is_inter && plan.tx_w == 4 && plan.tx_h == 4) {
      for (int ty = 0; ty < bh; ty += 4) {
        for (int tx = 0; tx < bw; tx += 4) {
          bool idtx = true;
          if (!read_one(tx, ty, 4, 4, &idtx, false, false)) return false;
          (*luma_idtx)[static_cast<std::size_t>(ty / 4) * tw4 + tx / 4] =
              idtx ? 1 : 0;
          if (idtx) ++stats_->idtx_4x4;
          else ++stats_->dct_dct_4x4;
        }
      }
      return true;
    }

    if (!plan.is_inter && !plan.fsc) {
      for (int ty = 0; ty < bh; ty += 4) {
        for (int tx = 0; tx < bw; tx += 4) {
          bool idtx = false;
          if (!read_one(tx, ty, 4, 4, &idtx, plan.use_dpcm_y,
                        plan.dpcm_y_vertical)) {
            return false;
          }
        }
      }
      return true;
    }

    bool idtx = plan.tx_idtx;
    if (!read_one(0, 0, plan.tx_w, plan.tx_h, &idtx,
                  !plan.is_inter && plan.use_dpcm_y, plan.dpcm_y_vertical)) {
      return false;
    }
    for (std::size_t i = 0; i < luma_idtx->size(); ++i) {
      (*luma_idtx)[i] = idtx ? 1 : 0;
    }
    return true;
  }

  bool read_chroma(int r, int c, BlockSize bsize, const BlockPlan& plan,
                   const std::vector<std::uint8_t>& luma_idtx,
                   std::string* error) {
    if (!g_.has_chroma) return true;
    const int bw = partition::block_width(bsize);
    const int bh = partition::block_height(bsize);
    const int cw = bw >> g_.sub_x;
    const int ch = bh >> g_.sub_y;
    const int x = (c * 4) >> g_.sub_x;
    const int y = (r * 4) >> g_.sub_y;
    const int luma_w4 = bw / 4;
    PredictionScratch s;
    std::vector<int> levels(64, 0);
    for (int plane = 1; plane < 3; ++plane) {
      for (int cy = 0; cy < ch; cy += 4) {
        for (int cx = 0; cx < cw; cx += 4) {
          const int px = x + cx;
          const int py = y + cy;
          bool idtx = false;
          if (plan.is_inter) {
            if (plan.tx_w != 4 || plan.tx_h != 4) {
              idtx = true;
            } else {
              const int lx = (cx << g_.sub_x) >> 2;
              const int ly = (cy << g_.sub_y) >> 2;
              const std::size_t li =
                  static_cast<std::size_t>(ly) * luma_w4 + lx;
              idtx = li < luma_idtx.size() && luma_idtx[li] != 0;
            }
          } else {
            idtx = plan.fsc;
          }
          if (plan.is_inter) {
            if (plan.compound) {
              predict_inter_compound(*ref_[plane], *ref1_[plane], px, py, 4, 4,
                                     opt_.bit_depth, s.pred.data(), 4);
            } else {
              predict_inter_zero(*ref_[plane], px, py, 4, 4, s.pred.data(), 4);
            }
          } else if (!predict_intra_block(*recon_[plane], px, py, 4, 4,
                                          plan.uv_mode, px > 0, py > 0,
                                          opt_.bit_depth, s.pred.data(), 4)) {
            *error = "chroma intra prediction failed";
            return false;
          }
          CoeffResult cr;
          const int prev_eob_u = ctx_.eob_u;
          if (!read_coeff_block(cdfs_, r_, ctx_, plane, px >> 2, py >> 2, 1, 1,
                                cw, ch, 4, 4, plan.is_inter, plan.fsc,
                                opt_.enable_fsc, &idtx,
                                plane == 2 ? prev_eob_u : 0, levels.data(),
                                &cr)) {
            *error = r_.error();
            return false;
          }
          if (plane == 1) ctx_.eob_u = cr.eob;
          const bool dpcm = !plan.is_inter && plan.use_dpcm_uv;
          reconstruct_block(levels.data(), 4, 4, !idtx, dpcm,
                            plan.dpcm_uv_vertical, s.pred.data(), 4,
                            recon_[plane]->data.data() +
                                static_cast<std::size_t>(py) * recon_[plane]->w +
                                px,
                            recon_[plane]->w, opt_.bit_depth);
        }
      }
    }
    ctx_.eob_u = 0;
    return true;
  }

  const Geometry& g_;
  const Options& opt_;
  bool keyframe_;
  CdfState& cdfs_;
  TileContext& ctx_;
  FrameStats* stats_;
  Plane* recon_[3] = {};
  const Plane* ref_[3] = {};
  const Plane* ref1_[3] = {};
  int num_refs_ = 1;
  int block_index_ = 0;
  SymReader r_{};
};

// ---------------------------------------------------------------------------
// OBU and header syntax
// ---------------------------------------------------------------------------

void put_uvlc(av2enc::BitWriter& w, std::uint32_t value) {
  unsigned lz = 0;
  while (value >= ((1ULL << (lz + 1)) - 1ULL)) ++lz;
  w.put_bits(0, lz);
  w.put_bit(true);
  w.put_bits(static_cast<std::uint64_t>(value) - ((1ULL << lz) - 1ULL), lz);
}

void put_ns(av2enc::BitWriter& w, std::uint32_t n, std::uint32_t value) {
  unsigned bits = 0;
  while ((1U << bits) < n) ++bits;
  const std::uint32_t m = (1U << bits) - n;
  if (value < m) {
    w.put_bits(value, bits - 1);
  } else {
    const std::uint32_t x = value + m;
    w.put_bits(x >> 1, bits - 1);
    w.put_bit((x & 1) != 0);
  }
}

void put_trailing_bits(av2enc::BitWriter& w) {
  w.put_bit(true);
  w.align_to_byte();
}

void append_leb128(std::vector<std::uint8_t>& out, std::uint32_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7f);
    value >>= 7;
    if (value != 0) byte |= 0x80;
    out.push_back(byte);
  } while (value != 0);
}

std::vector<std::uint8_t> make_obu(std::uint8_t type,
                                   std::span<const std::uint8_t> payload,
                                   std::uint8_t tlayer_id = 0) {
  std::vector<std::uint8_t> body;
  body.push_back(static_cast<std::uint8_t>(((type & 0x1f) << 2) |
                                           (tlayer_id & 0x03)));
  body.insert(body.end(), payload.begin(), payload.end());
  std::vector<std::uint8_t> out;
  append_leb128(out, static_cast<std::uint32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

struct ObuView {
  std::uint8_t type = 0;
  const std::uint8_t* payload = nullptr;
  std::size_t size = 0;
};

bool parse_obus(const std::uint8_t* data, std::size_t size,
                std::vector<ObuView>& out, std::string* error) {
  std::size_t pos = 0;
  while (pos < size) {
    std::uint32_t len = 0;
    unsigned shift = 0;
    std::uint8_t byte = 0;
    do {
      if (pos >= size || shift > 28) {
        *error = "truncated/invalid leb128 OBU size";
        return false;
      }
      byte = data[pos++];
      len |= static_cast<std::uint32_t>(byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);
    if (len == 0 || pos + len > size) {
      *error = "OBU size exceeds temporal unit";
      return false;
    }
    const std::uint8_t header = data[pos];
    const bool ext = (header & 0x80) != 0;
    const std::uint8_t type = (header >> 2) & 0x1f;
    const std::size_t header_bytes = ext ? 2 : 1;
    if (len < header_bytes) {
      *error = "OBU shorter than its header";
      return false;
    }
    ObuView view;
    view.type = type;
    view.payload = data + pos + header_bytes;
    view.size = len - header_bytes;
    out.push_back(view);
    pos += len;
  }
  return true;
}

unsigned ceil_log2(unsigned v) {
  unsigned bits = 0;
  unsigned x = v - 1;
  while (x != 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1;
    ++bits;
  }
  return bits;
}

std::uint8_t chroma_format_idc(ChromaFormat f) {
  switch (f) {
    case ChromaFormat::k420:
      return 0;
    case ChromaFormat::k444:
      return 2;
    case ChromaFormat::k422:
      return 3;
    case ChromaFormat::k400:
      return 1;
  }
  return 0;
}

std::uint8_t sequence_profile(ChromaFormat f) {
  switch (f) {
    case ChromaFormat::k420:
      return 0;
    case ChromaFormat::k422:
      return 3;
    case ChromaFormat::k444:
      return 4;
    case ChromaFormat::k400:
      return 1;
  }
  return 31;
}

// §5.4.1 fields that depend on the frame schedule.  The plain key+P chain
// keeps max_tlayer_id = 0, monotonic output and NumRefFrames = 2; the
// hierarchical-P schedule raises them from the tlayers plan.
struct ScheduleInfo {
  int max_tlayer_id = 0;
  bool monotonic_output_order = true;
  int order_hint_bits = 4;
  int num_ref_frames = 2;
};

void write_sequence_header(av2enc::BitWriter& w, const Options& opt,
                           const Geometry& g, const ScheduleInfo& schedule) {
  put_uvlc(w, 0);  // seq_header_id
  w.put_bits(sequence_profile(g.format), 5);
  w.put_bit(false);  // single_picture_header_flag
  w.put_bits(static_cast<std::uint64_t>(opt.seq_level_idx), 5);
  if (opt.seq_level_idx > 3) w.put_bit(false);  // seq_tier
  put_uvlc(w, chroma_format_idc(g.format));
  // §5.4.1 bit_depth_idc: 0 = 10-bit, 1 = 8-bit (12-bit is not signalable).
  put_uvlc(w, opt.bit_depth == 8 ? 1 : 0);
  w.put_bits(0, 3);  // seq_lcr_id
  w.put_bit(false);  // still_picture
  w.put_bits(static_cast<std::uint64_t>(schedule.max_tlayer_id), 2);
  w.put_bits(0, 3);  // max_mlayer_id
  w.put_bit(schedule.monotonic_output_order);

  unsigned wb = bit_width(static_cast<std::uint32_t>(g.width) - 1);
  unsigned hb = bit_width(static_cast<std::uint32_t>(g.height) - 1);
  if (wb == 0) wb = 1;
  if (hb == 0) hb = 1;
  w.put_bits(wb - 1, 4);
  w.put_bits(hb - 1, 4);
  w.put_bits(static_cast<std::uint64_t>(g.width) - 1, wb);
  w.put_bits(static_cast<std::uint64_t>(g.height) - 1, hb);
  w.put_bit(false);  // seq_cropping_window_present_flag
  w.put_bit(false);  // seq_initial_display_delay_present_flag
  w.put_bit(false);  // decoder_model_info_present_flag

  // §5.4.1 TLayerDependencyMap: present only when max_tlayer_id > 0.  The
  // hierarchical-P plan uses the default map (ref_tlayer <= obu_tlayer_id),
  // so the explicit map flag is 0.
  if (schedule.max_tlayer_id > 0) {
    w.put_bit(false);  // tlayer_dependency_present_flag = 0
  }

  // §5.4.3 sequence_partition_config()
  w.put_bit(false);  // use_256x256_superblock
  w.put_bit(false);  // use_128x128_superblock
  if (g.has_chroma) {
    w.put_bit(false);  // enable_sdp (inferred 0 for Monochrome)
  }
  w.put_bit(false);  // enable_ext_partitions
  w.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config()
  w.put_bit(false);  // enable_ext_seg -> MaxSegments = 8
  w.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config()
  w.put_bit(false);  // enable_dip
  w.put_bit(false);  // enable_intra_edge_filter
  w.put_bit(false);  // enable_mrls
  w.put_bit(false);  // enable_cfl_intra
  if (g.has_chroma) {
    w.put_bits(0, 2);  // cfl_ds_filter_index (not coded for Monochrome)
  }
  w.put_bit(false);  // enable_mhccp
  w.put_bit(false);  // enable_ibp

  // §5.4.6 sequence_inter_config(): 4 motion modes, no motion-mode present.
  for (int i = 0; i < 4; ++i) w.put_bit(false);
  w.put_bit(false);  // enable_masked_compound
  w.put_bit(false);  // enable_ref_frame_mvs
  w.put_bits(static_cast<std::uint64_t>(schedule.order_hint_bits - 1), 4);
  w.put_bit(false);  // enable_refmvbank
  w.put_bit(true);   // disable_drl_reorder
  w.put_bit(true);   // explicit_ref_frame_map
  w.put_bit(true);   // explicit_num_ref_frames
  w.put_bits(static_cast<std::uint64_t>(schedule.num_ref_frames - 1), 4);
  w.put_bits(0, 3);  // long_term_frame_id_bits
  put_ns(w, 5, 0);   // seq_max_drl_bits_minus_1
  w.put_bit(false);  // allow_frame_max_drl_bits
  put_ns(w, 3, 0);   // seq_max_bvp_drl_bits_minus_1
  w.put_bit(false);  // allow_frame_max_bvp_drl_bits
  w.put_bits(0, 2);  // num_same_ref_compound
  w.put_bit(false);  // enable_tip
  w.put_bit(false);  // enable_mv_traj
  w.put_bit(false);  // enable_bawp
  w.put_bit(false);  // enable_cwp
  w.put_bit(false);  // enable_imp_msk_bld
  w.put_bit(false);  // enable_df_sub_pu
  w.put_bits(0, 2);  // enable_opfl_refine = REFINE_NONE
  w.put_bit(false);  // enable_refinemv
  w.put_bit(false);  // enable_bru
  w.put_bit(false);  // enable_adaptive_mvd
  w.put_bit(false);  // enable_mvd_sign_derive
  w.put_bit(false);  // enable_flex_mvres
  w.put_bit(false);  // enable_global_motion (identity)
  w.put_bit(false);  // enable_short_refresh_frame_flags

  // §5.4.7 sequence_scc_config()
  w.put_bit(false);  // seq_choose_screen_content_tools
  w.put_bit(false);  // seq_force_screen_content_tools = 0

  // §5.4.8 sequence_transform_quant_entropy_config()
  w.put_bit(opt.enable_fsc);
  if (!opt.enable_fsc) w.put_bit(false);  // enable_idtx_intra
  w.put_bit(false);  // enable_intra_ist
  w.put_bit(false);  // enable_inter_ist
  if (g.has_chroma) w.put_bit(false);  // enable_chroma_dctonly
  w.put_bit(false);  // enable_inter_ddt
  w.put_bit(false);  // reduced_tx_part_set
  if (g.has_chroma) w.put_bit(false);  // enable_cctx
  w.put_bit(false);  // enable_tcq
  w.put_bit(false);  // enable_parity_hiding
  w.put_bit(false);  // enable_avg_cdf
  if (g.has_chroma) w.put_bit(false);  // separate_uv_delta_q
  w.put_bit(true);   // equal_ac_dc_q
  if (g.has_chroma) {
    w.put_bits(23, 5); // base_uv_ac_delta_q -> 0
    w.put_bit(false);  // uv_ac_delta_q_enabled
  }

  // §5.4.10 sequence_filter_config()
  w.put_bit(false);  // disable_loopfilters_across_tiles
  w.put_bit(false);  // enable_cdef
  w.put_bit(false);  // enable_gdf
  w.put_bit(false);  // enable_restoration
  w.put_bit(false);  // enable_ccso
  w.put_bit(false);  // cdef_on_skip_txfm_always_on
  w.put_bit(true);   // cdef_on_skip_txfm_disabled
  w.put_bits(0, 2);  // df_par_bits_minus_2
  w.put_bit(false);  // seq_tile_info_present_flag
  w.put_bit(false);  // film_grain_params_present
}

struct BitReader {
  const std::uint8_t* data = nullptr;
  std::size_t bits = 0;
  std::size_t pos = 0;

  bool read_bit(bool* out) {
    if (pos >= bits) return false;
    *out = ((data[pos >> 3] >> (7 - (pos & 7))) & 1) != 0;
    ++pos;
    return true;
  }
  bool read_bits(unsigned n, std::uint32_t* out) {
    std::uint32_t v = 0;
    for (unsigned i = 0; i < n; ++i) {
      bool b = false;
      if (!read_bit(&b)) return false;
      v = (v << 1) | (b ? 1u : 0u);
    }
    *out = v;
    return true;
  }
  bool uvlc(std::uint32_t* out) {
    unsigned lz = 0;
    while (true) {
      bool b = false;
      if (!read_bit(&b)) return false;
      if (b) break;
      ++lz;
      if (lz > 31) return false;
    }
    std::uint64_t payload = 0;
    for (unsigned i = 0; i < lz; ++i) {
      bool b = false;
      if (!read_bit(&b)) return false;
      payload = (payload << 1) | (b ? 1u : 0u);
    }
    *out = static_cast<std::uint32_t>(((1ULL << lz) - 1ULL) + payload);
    return true;
  }
  bool ns(unsigned n, std::uint32_t* out) {
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    std::uint32_t v = 0;
    if (!read_bits(w - 1, &v)) return false;
    if (v < m) {
      *out = v;
      return true;
    }
    bool b = false;
    if (!read_bit(&b)) return false;
    *out = (v << 1) + (b ? 1u : 0u) - m;
    return true;
  }
  bool align_zero(std::string* error) {
    while ((pos & 7) != 0) {
      bool b = false;
      if (!read_bit(&b)) return false;
      if (b) {
        *error = "non-zero byte alignment bit";
        return false;
      }
    }
    return true;
  }
};

struct SeqInfo {
  int width = 0;
  int height = 0;
  ChromaFormat format = ChromaFormat::k420;
  int seq_level_idx = 0;
  bool enable_fsc = false;
  int order_hint_bits = 0;
  int num_ref_frames = 0;
  int bit_depth = 8;
  int max_tlayer_id = 0;
};

bool parse_sequence_header(const ObuView& obu, SeqInfo* info,
                           std::string* error) {
  BitReader r{obu.payload, obu.size * 8, 0};
  std::uint32_t v = 0;
  if (!r.uvlc(&v) || v != 0) {
    *error = "sequence header: seq_header_id";
    return false;
  }
  std::uint32_t profile = 0;
  if (!r.read_bits(5, &profile)) return false;
  bool single = false;
  if (!r.read_bit(&single) || single) {
    *error = "sequence header: expected multi-picture header";
    return false;
  }
  if (!r.read_bits(5, &v)) return false;
  info->seq_level_idx = static_cast<int>(v);
  if (info->seq_level_idx > 3) {
    bool tier = false;
    if (!r.read_bit(&tier)) return false;
  }
  std::uint32_t chroma = 0;
  if (!r.uvlc(&chroma) || chroma > 3) {
    *error = "sequence header: chroma_format_idc";
    return false;
  }
  info->format = static_cast<ChromaFormat>(chroma);
  const bool monochrome = info->format == ChromaFormat::k400;
  std::uint32_t depth = 0;
  if (!r.uvlc(&depth) || depth > 1) {
    *error = "sequence header: bit_depth_idc";
    return false;
  }
  info->bit_depth = depth == 1 ? 8 : 10;
  if (!r.read_bits(3, &v)) return false;  // seq_lcr_id
  bool still = false;
  if (!r.read_bit(&still)) return false;
  if (!r.read_bits(2, &v)) return false;
  info->max_tlayer_id = static_cast<int>(v);
  if (!r.read_bits(3, &v)) return false;
  if (v != 0) {
    *error = "sequence header: multi-layer streams unsupported";
    return false;
  }
  bool monotonic = false;
  if (!r.read_bit(&monotonic)) return false;

  std::uint32_t wb = 0;
  std::uint32_t hb = 0;
  if (!r.read_bits(4, &wb) || !r.read_bits(4, &hb)) return false;
  ++wb;
  ++hb;
  std::uint32_t wm1 = 0;
  std::uint32_t hm1 = 0;
  if (!r.read_bits(wb, &wm1) || !r.read_bits(hb, &hm1)) return false;
  info->width = static_cast<int>(wm1) + 1;
  info->height = static_cast<int>(hm1) + 1;

  bool crop = false;
  if (!r.read_bit(&crop) || crop) {
    *error = "sequence header: cropping unsupported";
    return false;
  }
  bool delay = false;
  bool dmodel = false;
  if (!r.read_bit(&delay) || !r.read_bit(&dmodel) || delay || dmodel) {
    *error = "sequence header: decoder model unsupported";
    return false;
  }
  if (info->max_tlayer_id > 0) {
    bool tlayer_map = false;
    if (!r.read_bit(&tlayer_map) || tlayer_map) {
      *error = "sequence header: explicit TLayerDependencyMap unsupported";
      return false;
    }
  }
  bool b256 = false, b128 = false, sdp = false, extpart = false,
       aspect = false;
  if (!r.read_bit(&b256) || !r.read_bit(&b128)) {
    return false;
  }
  if (!monochrome) {
    if (!r.read_bit(&sdp)) return false;
  }
  if (!r.read_bit(&extpart) || !r.read_bit(&aspect)) {
    return false;
  }
  if (b256 || b128 || sdp || extpart || aspect) {
    *error = "sequence header: partition config unsupported";
    return false;
  }
  bool extseg = false, seqseginfo = false;
  if (!r.read_bit(&extseg) || !r.read_bit(&seqseginfo) || extseg ||
      seqseginfo) {
    *error = "sequence header: segment config unsupported";
    return false;
  }
  bool dip = false, edge = false, mrls = false, cfl = false, mhccp = false,
       ibp = false;
  if (!r.read_bit(&dip) || !r.read_bit(&edge) || !r.read_bit(&mrls) ||
      !r.read_bit(&cfl)) {
    return false;
  }
  if (!monochrome) {
    if (!r.read_bits(2, &v)) return false;  // cfl_ds_filter_index
  }
  if (!r.read_bit(&mhccp) || !r.read_bit(&ibp)) {
    return false;
  }
  for (int i = 0; i < 4; ++i) {
    bool mode = false;
    if (!r.read_bit(&mode) || mode) {
      *error = "sequence header: motion modes unsupported";
      return false;
    }
  }
  bool masked = false, refmvs = false;
  if (!r.read_bit(&masked) || !r.read_bit(&refmvs)) return false;
  if (masked || refmvs) {
    *error = "sequence header: inter config unsupported";
    return false;
  }
  std::uint32_t oh = 0;
  if (!r.read_bits(4, &oh)) return false;
  info->order_hint_bits = static_cast<int>(oh) + 1;
  bool refmvbank = false, drl = false;
  if (!r.read_bit(&refmvbank) || !r.read_bit(&drl)) return false;
  if (!drl) {
    bool constrain = false;
    if (!r.read_bit(&constrain)) return false;
  }
  bool explicit_map = false, explicit_num = false;
  if (!r.read_bit(&explicit_map) || !r.read_bit(&explicit_num)) return false;
  if (!explicit_map || !explicit_num) {
    *error = "sequence header: requires explicit ref map/count";
    return false;
  }
  std::uint32_t nrf = 0;
  if (!r.read_bits(4, &nrf)) return false;
  info->num_ref_frames = static_cast<int>(nrf) + 1;
  if (!r.read_bits(3, &v)) return false;  // long_term_frame_id_bits
  if (!r.ns(5, &v)) return false;         // seq_max_drl
  bool allow_drl = false;
  if (!r.read_bit(&allow_drl)) return false;
  if (!r.ns(3, &v)) return false;  // seq_max_bvp
  bool allow_bvp = false;
  if (!r.read_bit(&allow_bvp)) return false;
  if (!r.read_bits(2, &v)) return false;  // num_same_ref_compound
  bool tip = false;
  if (!r.read_bit(&tip)) return false;
  if (tip) {
    *error = "sequence header: TIP unsupported";
    return false;
  }
  bool mvtraj = false, bawp = false, cwp = false, imp = false, dfsub = false;
  if (!r.read_bit(&mvtraj) || !r.read_bit(&bawp) || !r.read_bit(&cwp) ||
      !r.read_bit(&imp) || !r.read_bit(&dfsub)) {
    return false;
  }
  if (!r.read_bits(2, &v)) return false;  // enable_opfl_refine
  bool refinemv = false;
  if (!r.read_bit(&refinemv)) return false;
  bool bru = false, amvd = false, mvsign = false, flex = false, gm = false,
       shortrefresh = false;
  if (!r.read_bit(&bru) || !r.read_bit(&amvd) || !r.read_bit(&mvsign) ||
      !r.read_bit(&flex) || !r.read_bit(&gm) || !r.read_bit(&shortrefresh)) {
    return false;
  }
  bool choose_sct = false, force_sct = false;
  if (!r.read_bit(&choose_sct) || !r.read_bit(&force_sct)) return false;
  bool fsc = false;
  if (!r.read_bit(&fsc)) return false;
  info->enable_fsc = fsc;
  if (!fsc) {
    bool idtx = false;
    if (!r.read_bit(&idtx)) return false;
  } else {
    info->enable_fsc = true;
  }
  bool ist_i = false, ist_e = false, chroma_dct = false, ddt = false,
       reduced_part = false, cctx = false, tcq = false;
  if (!r.read_bit(&ist_i) || !r.read_bit(&ist_e)) return false;
  if (!monochrome) {
    if (!r.read_bit(&chroma_dct)) return false;
  }
  if (!r.read_bit(&ddt) || !r.read_bit(&reduced_part)) return false;
  if (!monochrome) {
    if (!r.read_bit(&cctx)) return false;
  }
  if (!r.read_bit(&tcq)) {
    return false;
  }
  bool parity = false;
  if (!r.read_bit(&parity)) return false;
  bool avg = false;
  if (!r.read_bit(&avg)) return false;
  if (avg) {
    bool avgtype = false;
    if (!r.read_bit(&avgtype)) return false;
  }
  bool sep = false;
  if (!monochrome) {
    if (!r.read_bit(&sep)) return false;  // separate_uv_delta_q
  }
  bool equal = false;
  if (!r.read_bit(&equal)) return false;
  if (!equal) {
    if (!r.read_bits(5, &v)) return false;
    bool ydc = false;
    if (!r.read_bit(&ydc)) return false;
    if (!monochrome) {
      if (!r.read_bits(5, &v)) return false;
      bool udc = false;
      if (!r.read_bit(&udc)) return false;
    }
  }
  if (!monochrome) {
    if (!r.read_bits(5, &v)) return false;  // base_uv_ac_delta_q
    bool uvac = false;
    if (!r.read_bit(&uvac)) return false;
  }
  bool noloop = false, cdef = false, gdf = false;
  if (!r.read_bit(&noloop) || !r.read_bit(&cdef) || !r.read_bit(&gdf)) {
    return false;
  }
  bool rest = false;
  if (!r.read_bit(&rest) || rest) {
    *error = "sequence header: restoration unsupported";
    return false;
  }
  bool ccso = false;
  if (!r.read_bit(&ccso) || ccso) {
    *error = "sequence header: CCSO unsupported";
    return false;
  }
  bool cdefon = false;
  if (!r.read_bit(&cdefon)) return false;
  if (!cdefon) {
    bool cdefoff = false;
    if (!r.read_bit(&cdefoff)) return false;
  }
  if (!r.read_bits(2, &v)) return false;  // df_par_bits_minus_2
  bool tileinfo = false;
  if (!r.read_bit(&tileinfo) || tileinfo) {
    *error = "sequence header: seq tile info unsupported";
    return false;
  }
  bool grain = false;
  if (!r.read_bit(&grain) || grain) {
    *error = "sequence header: film grain unsupported";
    return false;
  }
  return true;
}

struct FrameHeaderInfo {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_flags = 0;
  std::uint32_t num_total_refs = 1;
  std::uint32_t ref_idx[2] = {0, 0};
  bool reference_select = false;
  bool disable_cdf_update = true;
  std::uint32_t base_q_idx = 0;
  bool segmentation_enabled = false;
  bool using_qmatrix = false;
  std::uint32_t reduced_tx_set = 0;
  // §5.18.2 output flags.  The plain chain is always immediate output; the
  // hierarchical-P schedule codes every frame after the first as an implicit
  // output picture because monotonic_output_order_flag = 0.
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
  int order_hint_bits = 4;
};

void write_frame_header(av2enc::BitWriter& w, const Options& opt,
                        const FrameHeaderInfo& h, const Geometry& g,
                        int num_ref_frames) {
  put_uvlc(w, 0);  // cur_mfh_id
  put_uvlc(w, 0);  // seq_header_id_in_frame_header
  if (!h.keyframe) w.put_bit(true);     // frame_is_inter
  w.put_bit(h.immediate_output_frame);
  if (!h.immediate_output_frame) {
    w.put_bit(h.implicit_output_frame);
  }
  w.put_bit(false);                     // frame_size_override_flag
  w.put_bits(h.order_hint, static_cast<unsigned>(h.order_hint_bits));
  if (h.keyframe) {
    // OBU_CLOSED_LOOP_KEY + max_mlayer_id == 0: refresh_frame_flags inferred.
  } else {
    w.put_bit(false);  // signal_primary_ref_frame
    w.put_bit(true);   // disable_cross_frame_cdf_init
    w.put_bits(h.refresh_flags, static_cast<unsigned>(num_ref_frames));
    w.put_bit(true);                  // frame_explicit_ref_frame_map
    w.put_bits(h.num_total_refs, 3);  // num_total_refs
    const unsigned ref_bits =
        ceil_log2(static_cast<unsigned>(num_ref_frames));
    for (std::uint32_t i = 0; i < h.num_total_refs && i < 2; ++i) {
      w.put_bits(h.ref_idx[i], ref_bits);  // ref_frame_idx[i]
    }
  }
  if (h.keyframe) {
    w.put_bit(false);  // allow_intrabc
  } else {
    w.put_bit(false);  // allow_intrabc
    w.put_bit(false);  // use_qtr_precision_mv
    w.put_bit(true);   // allow_high_precision_mv -> 1/8 pel
    w.put_bit(false);  // is_filter_switchable
    w.put_bits(0, 2);  // interpolation_filter = EIGHTTAP
  }
  w.put_bit(opt.adaptive_cdf ? false : true);  // disable_cdf_update
  const int sb_cols = (g.mi_cols + 15) >> 4;
  const int sb_rows = (g.mi_rows + 15) >> 4;
  w.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1) w.put_bit(false);
  if (sb_rows > 1) w.put_bit(false);
  w.put_bits(h.base_q_idx, opt.bit_depth == 8 ? 8 : 9);
  w.put_bit(h.segmentation_enabled);
  w.put_bit(h.using_qmatrix);
  if (!h.keyframe) {
    w.put_bit(h.reference_select);  // reference_select
    w.put_bit(false);               // skip_mode_present
  }
  w.put_bits(h.reduced_tx_set, 2);
}

bool read_frame_header(const ObuView& obu, bool keyframe, const SeqInfo& seq,
                       const Geometry& g, FrameHeaderInfo* out,
                       std::size_t* header_bytes, std::string* error) {
  BitReader r{obu.payload, obu.size * 8, 0};
  out->keyframe = keyframe;
  bool first_tile_group = false;
  if (!r.read_bit(&first_tile_group) || !first_tile_group) {
    *error = "frame header: expected is_first_tile_group";
    return false;
  }
  std::uint32_t v = 0;
  if (!r.uvlc(&v) || v != 0) {
    *error = "frame header: cur_mfh_id";
    return false;
  }
  if (!r.uvlc(&v) || v != 0) {
    *error = "frame header: seq_header_id";
    return false;
  }
  if (!keyframe) {
    bool inter = false;
    if (!r.read_bit(&inter) || !inter) {
      *error = "frame header: expected INTER_FRAME";
      return false;
    }
  }
  bool immediate = false;
  if (!r.read_bit(&immediate)) return false;
  if (!immediate) {
    // monotonic_output_order_flag = 0 (hierarchical-P schedule) codes the
    // implicit output picture flag right after it.
    bool implicit = false;
    if (!r.read_bit(&implicit) || !implicit) {
      *error = "frame header: hidden picture";
      return false;
    }
  }
  bool override = false;
  if (!r.read_bit(&override) || override) {
    *error = "frame header: frame_size_override unsupported";
    return false;
  }
  if (!r.read_bits(static_cast<unsigned>(seq.order_hint_bits),
                   &out->order_hint)) {
    return false;
  }
  if (!keyframe) {
    bool signal_primary = false;
    if (!r.read_bit(&signal_primary) || signal_primary) {
      *error = "frame header: primary ref signalling unsupported";
      return false;
    }
    bool disable_cross = false;
    if (!r.read_bit(&disable_cross) || !disable_cross) {
      *error = "frame header: cross-frame CDF init unsupported";
      return false;
    }
    if (!r.read_bits(static_cast<unsigned>(seq.num_ref_frames), &v)) {
      return false;
    }
    out->refresh_flags = v;
    bool explicit_map = false;
    if (!r.read_bit(&explicit_map) || !explicit_map) {
      *error = "frame header: explicit ref map";
      return false;
    }
    if (!r.read_bits(3, &v)) return false;
    out->num_total_refs = v;
    if (out->num_total_refs != 1 && out->num_total_refs != 2) {
      *error = "frame header: only one or two total refs are supported";
      return false;
    }
    const unsigned ref_bits =
        ceil_log2(static_cast<unsigned>(seq.num_ref_frames));
    for (std::uint32_t i = 0; i < out->num_total_refs; ++i) {
      if (!r.read_bits(ref_bits, &out->ref_idx[i])) return false;
    }
  }
  if (keyframe) {
    bool intrabc = false;
    if (!r.read_bit(&intrabc) || intrabc) {
      *error = "frame header: IntrABC unsupported";
      return false;
    }
  } else {
    bool intrabc = false;
    if (!r.read_bit(&intrabc) || intrabc) {
      *error = "frame header: IntrABC unsupported";
      return false;
    }
    bool qtr = false;
    if (!r.read_bit(&qtr) || qtr) {
      *error = "frame header: quarter-pel precision unsupported";
      return false;
    }
    bool high = false;
    if (!r.read_bit(&high) || !high) {
      *error = "frame header: expected 1/8 pel precision";
      return false;
    }
    bool switchable = false;
    if (!r.read_bit(&switchable) || switchable) {
      *error = "frame header: switchable interpolation filter";
      return false;
    }
    std::uint32_t filt = 0;
    if (!r.read_bits(2, &filt) || filt != 0) {
      *error = "frame header: expected EIGHTTAP";
      return false;
    }
  }
  bool disable_cdf = false;
  if (!r.read_bit(&disable_cdf)) return false;
  out->disable_cdf_update = disable_cdf;
  bool uniform = false;
  if (!r.read_bit(&uniform) || !uniform) {
    *error = "frame header: expected uniform tile spacing";
    return false;
  }
  const int sb_cols = (g.mi_cols + 15) >> 4;
  const int sb_rows = (g.mi_rows + 15) >> 4;
  if (sb_cols > 1) {
    if (!r.read_bits(1, &v) || v != 0) {
      *error = "frame header: tile cols increment";
      return false;
    }
  }
  if (sb_rows > 1) {
    if (!r.read_bits(1, &v) || v != 0) {
      *error = "frame header: tile rows increment";
      return false;
    }
  }
  if (!r.read_bits(seq.bit_depth == 8 ? 8 : 9, &out->base_q_idx)) {
    return false;
  }
  bool seg = false;
  if (!r.read_bit(&seg)) return false;
  out->segmentation_enabled = seg;
  if (seg) {
    *error = "frame header: segmentation syntax is not written";
    return false;
  }
  bool qm = false;
  if (!r.read_bit(&qm)) return false;
  out->using_qmatrix = qm;
  if (qm) {
    *error = "frame header: QM unsupported";
    return false;
  }
  if (!keyframe) {
    bool reference_select = false;
    if (!r.read_bit(&reference_select)) {
      *error = "frame header: reference_select";
      return false;
    }
    out->reference_select = reference_select;
    bool skip_mode = false;
    if (!r.read_bit(&skip_mode) || skip_mode) {
      *error = "frame header: skip_mode_present";
      return false;
    }
  }
  if (!r.read_bits(2, &out->reduced_tx_set)) return false;
  if (!r.align_zero(error)) return false;
  *header_bytes = r.pos / 8;
  return true;
}

bool planes_from_picture(const Picture& p, const Geometry& g, Plane* out,
                         std::string* error) {
  const std::size_t y_size = static_cast<std::size_t>(g.width) * g.height;
  const std::size_t c_size =
      static_cast<std::size_t>(g.chroma_w) * g.chroma_h;
  if (p.y.size() < y_size) {
    *error = "picture plane spans are smaller than the geometry";
    return false;
  }
  out[0].w = g.width;
  out[0].h = g.height;
  out[0].data.assign(p.y.begin(), p.y.begin() + y_size);
  if (!g.has_chroma) {
    for (int plane = 1; plane < 3; ++plane) {
      out[plane].w = 0;
      out[plane].h = 0;
      out[plane].data.clear();
    }
    return true;
  }
  if (p.u.size() < c_size || p.v.size() < c_size) {
    *error = "picture chroma spans are smaller than the geometry";
    return false;
  }
  out[1].w = g.chroma_w;
  out[1].h = g.chroma_h;
  out[1].data.assign(p.u.begin(), p.u.begin() + c_size);
  out[2].w = g.chroma_w;
  out[2].h = g.chroma_h;
  out[2].data.assign(p.v.begin(), p.v.begin() + c_size);
  return true;
}

Picture picture_from_planes(const Geometry& g, const Plane planes[3],
                           int bit_depth) {
  Picture p;
  p.width = g.width;
  p.height = g.height;
  p.format = g.format;
  p.bit_depth = bit_depth;
  p.y = planes[0].data;
  if (g.has_chroma) {
    p.u = planes[1].data;
    p.v = planes[2].data;
  }
  return p;
}

std::uint64_t plane_error(const Plane& a, const Plane& b) {
  std::uint64_t err = 0;
  for (std::size_t i = 0; i < a.data.size() && i < b.data.size(); ++i) {
    err = std::max<std::uint64_t>(err, static_cast<std::uint64_t>(
                                          abs_int(static_cast<int>(a.data[i]) -
                                                  static_cast<int>(b.data[i]))));
  }
  return err;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool encode_video(std::span<const Picture> pictures, const Options& options,
                  EncodeResult* result, std::string* error) {
  *result = {};
  if (pictures.empty()) {
    *error = "encode_video requires at least one picture";
    return false;
  }
  if (options.sb_size != 64) {
    *error = "lossless inter supports sb_size = 64 only";
    return false;
  }
  const Picture& first = pictures.front();
  if (first.format != ChromaFormat::k420 &&
      first.format != ChromaFormat::k422 &&
      first.format != ChromaFormat::k444 &&
      first.format != ChromaFormat::k400) {
    *error = "unsupported chroma format";
    return false;
  }
  Geometry g;
  if (!make_geometry(first.width, first.height, first.format, &g, error)) {
    return false;
  }
  for (const Picture& p : pictures) {
    if (p.width != first.width || p.height != first.height ||
        p.format != first.format) {
      *error = "all pictures must share geometry and format";
      return false;
    }
  }

  // Frame schedule.  temporal_layers == 0 is the historical key + flat P
  // chain (every P references the previous coded picture, refresh flags 1,
  // NumRefFrames 2).  temporal_layers = 4/8/16 is the same hierarchical-P
  // schedule --video-tlayers uses: the plan's coding order, one single
  // reference, implicit output pictures and per-OBU obu_tlayer_id.
  ScheduleInfo schedule;
  tl::Plan layer_plan;
  const bool layered = options.temporal_layers > 0;
  if (layered) {
    tl::PlanConfig pc;
    pc.gop_size = options.temporal_layers;
    pc.schedule = options.b_schedule ? tl::Schedule::kHierarchicalB
                                     : tl::Schedule::kHierarchicalP;
    if (options.b_schedule && options.temporal_layers > 8) {
      *error = "lossless B schedule supports a GOP of 4 or 8 pictures";
      return false;
    }
    pc.dependency = tl::DependencyMode::kAllLower;
    pc.explicit_dependency_map = false;  // pinned dav2d default map
    pc.order_hint_bits = options.temporal_layers >= 16 ? 5 : 4;
    pc.num_slots = options.temporal_layers;
    pc.base_q_idx = 0;
    tl::PlanError plan_error = tl::PlanError::kOk;
    if (!tl::build_plan(pc, layer_plan, &plan_error)) {
      *error = std::string("temporal-layer plan: ") +
               tl::plan_error_name(plan_error);
      return false;
    }
    if (static_cast<int>(pictures.size()) != options.temporal_layers) {
      *error = "--video-lossless --video-tlayers codes one closed GOP: the "
               "input must have exactly the GOP size in pictures";
      return false;
    }
    tl::PlanError validation = tl::PlanError::kOk;
    int bad_display = -1;
    int bad_coding = -1;
    if (!tl::validate_plan(layer_plan, &validation, &bad_display,
                           &bad_coding)) {
      *error = std::string("temporal-layer plan invalid: ") +
               tl::plan_error_name(validation);
      return false;
    }
    schedule.max_tlayer_id = layer_plan.max_tlayer_id;
    schedule.monotonic_output_order = false;
    schedule.order_hint_bits = pc.order_hint_bits;
    schedule.num_ref_frames = layer_plan.num_slots;
  }

  av2enc::BitWriter seq;
  write_sequence_header(seq, options, g, schedule);
  seq.put_bit(false);  // obu_extension_flag
  put_trailing_bits(seq);
  const std::vector<std::uint8_t> seq_obu = make_obu(1, seq.take_bytes());

  const std::vector<std::uint8_t> td_obu = make_obu(2, {});

  std::vector<int> coding_order;
  coding_order.reserve(pictures.size());
  if (layered) {
    coding_order = layer_plan.coding_order;
  } else {
    for (std::size_t i = 0; i < pictures.size(); ++i) {
      coding_order.push_back(static_cast<int>(i));
    }
  }
  std::vector<Picture> recon_by_display(pictures.size());
  std::uint64_t max_err = 0;

  for (std::size_t ci = 0; ci < coding_order.size(); ++ci) {
    const int display = coding_order[ci];
    const bool keyframe = ci == 0;
    const tl::PlannedFrame* pf =
        layered ? &layer_plan.frames[static_cast<std::size_t>(display)]
                : nullptr;
    Plane src[3];
    if (!planes_from_picture(pictures[static_cast<std::size_t>(display)], g,
                             src, error)) {
      return false;
    }
    Plane recon[3];
    recon[0].resize(g.width, g.height);
    recon[1].resize(g.chroma_w, g.chroma_h);
    recon[2].resize(g.chroma_w, g.chroma_h);

    Plane ref_storage[2][3];
    const Plane* ref[3] = {nullptr, nullptr, nullptr};
    const Plane* ref1[3] = {nullptr, nullptr, nullptr};
    int num_refs = 1;
    int frame_distance[2] = {1, 0};
    if (!keyframe) {
      num_refs = pf != nullptr ? pf->num_refs : 1;
      for (int r = 0; r < num_refs && r < 2; ++r) {
        const int ref_display =
            pf != nullptr ? pf->ref_display[r] : coding_order[ci - 1];
        const Picture& ref_picture =
            recon_by_display[static_cast<std::size_t>(ref_display)];
        if (ref_picture.y.empty()) {
          *error = "lossless reference picture was not coded";
          return false;
        }
        if (!planes_from_picture(ref_picture, g, ref_storage[r], error)) {
          return false;
        }
        if (r == 0) {
          ref[0] = &ref_storage[0][0];
          ref[1] = &ref_storage[0][1];
          ref[2] = &ref_storage[0][2];
        } else {
          ref1[0] = &ref_storage[1][0];
          ref1[1] = &ref_storage[1][1];
          ref1[2] = &ref_storage[1][2];
        }
        if (pf != nullptr) {
          frame_distance[r] = clip3(
              -127, 127,
              static_cast<int>(pf->display_order_hint) -
                  static_cast<int>(pf->ref_order_hint[r]));
        } else {
          frame_distance[r] = 1;
        }
      }
    }

    CdfState cdfs;
    init_cdfs(cdfs);
    TileContext ctx;
    ctx.reset(g);
    FrameStats stats;
    Plane* src_ptr[3] = {&src[0], &src[1], &src[2]};
    Plane* recon_ptr[3] = {&recon[0], &recon[1], &recon[2]};
    std::vector<std::uint8_t> tile(16U * 1024U * 1024U);
    std::size_t tile_bytes = 0;
    TileWriter writer(g, options, keyframe, cdfs, ctx, src_ptr, recon_ptr, ref,
                      ref1, num_refs, frame_distance, &stats);
    if (!writer.encode_tile(tile.data(), tile.size(), &tile_bytes, *error)) {
      return false;
    }
    tile.resize(tile_bytes);

    FrameHeaderInfo header;
    header.keyframe = keyframe;
    header.num_total_refs = static_cast<std::uint32_t>(num_refs);
    header.reference_select = num_refs > 1;
    header.base_q_idx = 0;
    header.reduced_tx_set = 0;
    header.disable_cdf_update = !options.adaptive_cdf;
    header.order_hint_bits = schedule.order_hint_bits;
    if (pf != nullptr) {
      header.order_hint = pf->order_hint_lsbs;
      header.refresh_flags = pf->refresh_frame_flags;
      for (int r = 0; r < num_refs && r < 2; ++r) {
        header.ref_idx[r] = static_cast<std::uint32_t>(
            pf->ref_slot[r] < 0 ? 0 : pf->ref_slot[r]);
      }
      header.immediate_output_frame = pf->immediate_output_frame;
      header.implicit_output_frame = pf->implicit_output_frame;
    } else {
      header.order_hint = static_cast<std::uint32_t>(display);
      header.refresh_flags = 1;
      header.ref_idx[0] = 0;
      header.immediate_output_frame = true;
      header.implicit_output_frame = false;
    }

    av2enc::BitWriter fw;
    fw.put_bit(true);  // is_first_tile_group
    write_frame_header(fw, options, header, g, schedule.num_ref_frames);
    fw.align_to_byte();
    std::vector<std::uint8_t> payload = fw.take_bytes();
    payload.insert(payload.end(), tile.begin(), tile.end());
    const std::uint8_t obu_type = keyframe ? 4 : 7;
    const std::uint8_t obu_tlayer =
        pf != nullptr ? static_cast<std::uint8_t>(pf->obu_tlayer_id()) : 0;
    const std::vector<std::uint8_t> frame_obu =
        make_obu(obu_type, payload, obu_tlayer);

    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td_obu.begin(), td_obu.end());
    if (keyframe) unit.insert(unit.end(), seq_obu.begin(), seq_obu.end());
    unit.insert(unit.end(), frame_obu.begin(), frame_obu.end());
    result->units.push_back(std::move(unit));

    std::uint64_t err = 0;
    for (int p = 0; p < 3; ++p) {
      err = std::max(err, plane_error(recon[p], src[p]));
    }
    stats.max_abs_recon_error = err;
    stats.exact = err == 0;
    max_err = std::max(max_err, err);
    result->stats.push_back(stats);
    recon_by_display[static_cast<std::size_t>(display)] =
        picture_from_planes(g, recon, options.bit_depth);
  }
  result->recon.reserve(pictures.size());
  for (Picture& picture : recon_by_display) {
    result->recon.push_back(std::move(picture));
  }

  result->max_abs_recon_error = max_err;
  result->exact = max_err == 0;
  if (options.verify_exact && !result->exact) {
    *error = "lossless inter reconstruction is not bit exact";
    return false;
  }
  return true;
}

bool decode_video(std::span<const std::vector<std::uint8_t>> units,
                  DecodeResult* result, std::string* error) {
  *result = {};
  SeqInfo seq;
  bool have_seq = false;
  std::vector<FrameStats> stats_out;
  std::vector<Picture> slots;
  std::vector<bool> slot_valid;
  std::vector<std::uint32_t> slot_order_hint;
  std::vector<std::pair<std::uint32_t, Picture>> display_order;

  for (const auto& unit : units) {
    std::vector<ObuView> obus;
    if (!parse_obus(unit.data(), unit.size(), obus, error)) return false;
    for (const ObuView& obu : obus) {
      if (obu.type == 1) {
        if (!parse_sequence_header(obu, &seq, error)) return false;
        have_seq = true;
      }
    }
    if (!have_seq) {
      *error = "temporal unit is missing a sequence header";
      return false;
    }
    if (slots.size() != static_cast<std::size_t>(seq.num_ref_frames)) {
      slots.assign(static_cast<std::size_t>(seq.num_ref_frames), Picture{});
      slot_valid.assign(static_cast<std::size_t>(seq.num_ref_frames), false);
      slot_order_hint.assign(static_cast<std::size_t>(seq.num_ref_frames), 0);
    }
    Geometry g;
    if (!make_geometry(seq.width, seq.height, seq.format, &g, error)) {
      return false;
    }
    const ObuView* frame_obu = nullptr;
    bool keyframe = false;
    for (const ObuView& obu : obus) {
      if (obu.type == 4) {
        frame_obu = &obu;
        keyframe = true;
      } else if (obu.type == 7) {
        frame_obu = &obu;
        keyframe = false;
      }
    }
    if (frame_obu == nullptr) {
      *error = "temporal unit is missing a frame OBU";
      return false;
    }
    FrameHeaderInfo header;
    std::size_t header_bytes = 0;
    if (!read_frame_header(*frame_obu, keyframe, seq, g, &header,
                           &header_bytes, error)) {
      return false;
    }
    if (header_bytes > frame_obu->size) {
      *error = "frame header larger than the OBU payload";
      return false;
    }
    const std::uint8_t* tile = frame_obu->payload + header_bytes;
    const std::size_t tile_size = frame_obu->size - header_bytes;

    Plane ref_storage[2][3];
    const Plane* ref[3] = {nullptr, nullptr, nullptr};
    const Plane* ref1[3] = {nullptr, nullptr, nullptr};
    int num_refs = 1;
    int frame_distance[2] = {1, 0};
    if (!keyframe) {
      num_refs = static_cast<int>(header.num_total_refs);
      for (int r = 0; r < num_refs && r < 2; ++r) {
        const std::uint32_t slot = header.ref_idx[r];
        if (slot >= slots.size() || !slot_valid[slot]) {
          *error = "frame header references an invalid slot";
          return false;
        }
        if (!planes_from_picture(slots[slot], g, ref_storage[r], error)) {
          return false;
        }
        if (r == 0) {
          ref[0] = &ref_storage[0][0];
          ref[1] = &ref_storage[0][1];
          ref[2] = &ref_storage[0][2];
        } else {
          ref1[0] = &ref_storage[1][0];
          ref1[1] = &ref_storage[1][1];
          ref1[2] = &ref_storage[1][2];
        }
        frame_distance[r] = clip3(
            -127, 127,
            static_cast<int>(header.order_hint) -
                static_cast<int>(slot_order_hint[slot]));
      }
    }

    Plane recon[3];
    recon[0].resize(g.width, g.height);
    recon[1].resize(g.chroma_w, g.chroma_h);
    recon[2].resize(g.chroma_w, g.chroma_h);
    Options opt;
    opt.adaptive_cdf = !header.disable_cdf_update;
    opt.enable_fsc = seq.enable_fsc;
    opt.bit_depth = seq.bit_depth;
    CdfState cdfs;
    init_cdfs(cdfs);
    TileContext ctx;
    ctx.reset(g);
    FrameStats stats;
    Plane* recon_ptr[3] = {&recon[0], &recon[1], &recon[2]};
    TileReader reader(g, opt, keyframe, cdfs, ctx, recon_ptr, ref, ref1,
                      num_refs, frame_distance, &stats);
    if (!reader.decode_tile(tile, tile_size, tile_size, *error)) return false;

    Picture pic = picture_from_planes(g, recon, seq.bit_depth);
    stats_out.push_back(stats);
    // §7.23 refresh.  A CLK frame refreshes every slot (refresh_frame_flags is
    // inferred allFrames); later frames refresh the slots in the header.
    if (keyframe) {
      for (std::uint32_t s = 0; s < slots.size(); ++s) {
        slots[s] = pic;
        slot_valid[s] = true;
        slot_order_hint[s] = header.order_hint;
      }
    } else {
      for (std::uint32_t s = 0; s < slots.size(); ++s) {
        if ((header.refresh_flags & (1U << s)) == 0) continue;
        slots[s] = pic;
        slot_valid[s] = true;
        slot_order_hint[s] = header.order_hint;
      }
    }
    display_order.emplace_back(header.order_hint, std::move(pic));
  }

  // §7.21 output order follows the order hints, not coding order.
  std::sort(display_order.begin(), display_order.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<Picture> frames;
  frames.reserve(display_order.size());
  for (auto& entry : display_order) frames.push_back(std::move(entry.second));
  result->frames = std::move(frames);
  result->stats = std::move(stats_out);
  return true;
}

// ---------------------------------------------------------------------------
// FSC coefficient syntax round trip
// ---------------------------------------------------------------------------

bool fsc_coeff_roundtrip(const int* levels, const CoeffContext& context,
                         int* decoded, std::string* error) {
  const int w = context.coded_w;
  const int h = context.coded_h;
  if (w <= 0 || h <= 0 || w > 32 || h > 32) {
    *error = "FSC roundtrip requires a 4x4..32x32 transform";
    return false;
  }
  CdfState cdfs;
  init_cdfs(cdfs);

  // Neighbour grids for the synthetic frame.  Width in 4x4 units.
  std::vector<std::uint8_t> above_level(64, 0);
  std::vector<std::uint8_t> left_level(64, 0);
  std::vector<std::uint8_t> above_dc(64, 0);
  std::vector<std::uint8_t> left_dc(64, 0);

  CoeffContext ctx = context;
  ctx.mi_cols = std::max(ctx.mi_cols, 64);
  ctx.mi_rows = std::max(ctx.mi_rows, 64);
  ctx.above_level = above_level.data();
  ctx.left_level = left_level.data();
  ctx.above_dc = above_dc.data();
  ctx.left_dc = left_dc.data();

  TileContext tc;
  Geometry g;
  g.mi_cols = 64;
  g.mi_rows = 64;
  g.chroma_mi_cols = 64;
  g.chroma_mi_rows = 64;
  tc.reset(g);

  const int w4 = w >> 2;
  const int h4 = h >> 2;
  std::vector<std::uint8_t> out(16U * 1024U * 1024U);
  SymWriter sw;
  if (!sw.init(out.data(), out.size(), false, error)) return false;
  const int max_eob = std::min(32, w) * std::min(32, h);
  bool any = false;
  for (int i = 0; i < max_eob; ++i) any |= levels[i] != 0;
  if (any) {
    write_coeff_fsc(cdfs, sw, tc, ctx.x4, ctx.y4, w4, h4, w4 * 4, h4 * 4, w,
                    h, levels, ctx.is_inter, ctx.fsc_mode,
                    /*tx_idtx=*/true);
  } else {
    write_coeff_normal(cdfs, sw, tc, ctx.plane, ctx.x4, ctx.y4, w4, h4, w4 * 4,
                       h4 * 4, w, h, levels, ctx.is_inter, ctx.fsc_mode, true,
                       ctx.eob_u);
  }
  if (!sw.ok()) {
    *error = sw.error();
    return false;
  }
  std::size_t bytes = 0;
  if (!sw.finish(&bytes)) {
    *error = sw.error();
    return false;
  }

  CdfState rcdfs;
  init_cdfs(rcdfs);
  TileContext rtc;
  rtc.reset(g);
  SymReader sr;
  if (!sr.init(out.data(), out.size(), bytes, false, error)) return false;
  std::vector<int> got(static_cast<std::size_t>(w) * h, 0);
  CoeffResult cr;
  bool idtx = true;
  if (any) {
    if (!read_coeff_block(rcdfs, sr, rtc, ctx.plane, ctx.x4, ctx.y4, w4, h4,
                          w4 * 4, h4 * 4, w, h, ctx.is_inter, ctx.fsc_mode,
                          true, &idtx, ctx.eob_u, got.data(), &cr)) {
      *error = "FSC reader failed: " + sr.error();
      return false;
    }
  } else {
    if (!read_coeff_block(rcdfs, sr, rtc, ctx.plane, ctx.x4, ctx.y4, w4, h4,
                          w4 * 4, h4 * 4, w, h, ctx.is_inter, ctx.fsc_mode,
                          true, &idtx, ctx.eob_u, got.data(), &cr)) {
      *error = "FSC reader failed: " + sr.error();
      return false;
    }
  }
  if (!sr.finish()) {
    *error = sr.error();
    return false;
  }
  std::memcpy(decoded, got.data(),
              static_cast<std::size_t>(w) * h * sizeof(int));
  return true;
}

// Pinned goldens are filled in by the test harness after a deterministic build.
std::span<const GoldenCase> golden_cases() {
  static const GoldenCase kCases[] = {
      {"64x64_420_b16_p4_fsc", 64, 64, ChromaFormat::k420, 16, 4, true,
       "f8a75d8c5eb952c586c900938d8e31ab"},
      {"136x88_420_b8_p0_fsc", 136, 88, ChromaFormat::k420, 8, 0, true,
       "b12472f61da0ac3205c05ca97efd7034"},
      {"640x360_420_b16_p4_fsc", 640, 360, ChromaFormat::k420, 16, 4, true,
       "c63c0f423b07b554cdb1a200fea6a0ef"},
      {"64x64_444_b16_p4_fsc", 64, 64, ChromaFormat::k444, 16, 4, true,
       "7750883142da08ee2a9a17e6c0224eed"},
      {"64x64_422_b16_p4_fsc", 64, 64, ChromaFormat::k422, 16, 4, true,
       "deed7cba67c3383ee9d2a4be3d4e5201"},
  };
  return std::span<const GoldenCase>(kCases);
}

}  // namespace av2enc::lossless::inter

