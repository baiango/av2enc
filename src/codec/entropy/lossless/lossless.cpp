// AV2 v1.0.0 lossless still-picture encoder/reader.
//
// Everything in this file is written from the specification.  No AVM, SVT,
// clave, av2-turbo or other codec encoder code was consulted.
//
// Layout of this translation unit:
//   1. constants and small tables
//   2. reversible transforms / DPCM helpers
//   3. §5.18.2 lossless flag derivation
//   4. CDF state and OBU/syntax helpers
//   5. tile context (partition/mode/coefficient neighbour state)
//   6. coefficient syntax writer and reader (normal + FSC)
//   7. lossless tile writer (encoder)
//   8. lossless tile reader (decoder)
//   9. OBU assembly / parsing and public API

#include "codec/entropy/lossless/lossless.h"

#include "core/bitstream/bitwriter.h"
#include "codec/intra/modes/intra_modes.h"
#include "codec/entropy/lossless/lossless_cdf_tables.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/still_cdf_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::lossless {
namespace {

using av2enc::av2::Decoder;
using av2enc::av2::Encoder;
using av2enc::av2::Status;
using partition::BlockSize;
using partition::PartitionType;

constexpr int kDcPred = 0;
constexpr int kV = 1;
constexpr int kH = 2;
constexpr int kIntraModesCount = 13;
constexpr int kTx4x4 = 0;
constexpr int kTx8x8 = 1;
constexpr int kSentinel256 = static_cast<int>(BlockSize::k256x256);

// §9.2 Fsc_Bsize_Groups[BLOCK_SIZES].
constexpr int kFscBsizeGroups[29] = {
    0, 1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 3, 3, 4, 4, 6, 6, 4, 4, 6, 6};

// §9.2 Size_Group[BLOCK_SIZES].
constexpr int kSizeGroup[29] = {
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2};

// §5.20.7.30 get_scan(2D) for the two transform sizes this encoder uses.
constexpr int kScan4[16] = {0, 4, 1, 8, 5, 2, 12, 9,
                            6, 3, 13, 10, 7, 14, 11, 15};
constexpr int kScan8[64] = {
    0,  8,  1,  16, 9,  2,  24, 17, 10, 3,  32, 25, 18, 11, 4,  40,
    33, 26, 19, 12, 5,  48, 41, 34, 27, 20, 13, 6,  56, 49, 42, 35,
    28, 21, 14, 7,  57, 50, 43, 36, 29, 22, 15, 58, 51, 44, 37, 30,
    23, 59, 52, 45, 38, 31, 60, 53, 46, 39, 61, 54, 47, 62, 55, 63};

// §8.3.2 coefficient context offsets.
constexpr int kSigRefDiff[5][2] = {{0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

int clip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }

int floor_log2(int v) {
  int r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

int abs_int(int v) { return v < 0 ? -v : v; }

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

bool y_mode_is_directional(int mode) { return mode >= kV && mode <= 8; }

// §5.20.5.6 get_intra_uv_mode_set inversion for the 13-mode, CfL-not-allowed
// alphabet.  The default list is DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH,
// V, H, D45, D135, D67, D113, D157, D203.
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

// Exact inverse of the §7.15.2.2 lifting steps.  The inverse WHT is *not*
// self-inverse when (a - d) is odd because of the >>1 lifting step; this is
// the algebraic inverse that keeps the encoder bit exact.
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
  // Input order is src[0]=a, src[1]=c, src[2]=d, src[3]=b.
  out[0] = a0;
  out[1] = c0;
  out[2] = d0;
  out[3] = b0;
}

void forward_lossless_4x4(const int residual[16], int levels[16]) {
  // Decoder order is inverse-WHT rows then inverse-WHT columns, so the
  // encoder applies the exact inverse in the opposite order: columns first.
  int tmp[16] = {};
  for (int c = 0; c < 4; ++c) {
    int col[4] = {residual[0 * 4 + c], residual[1 * 4 + c],
                  residual[2 * 4 + c], residual[3 * 4 + c]};
    int out[4] = {};
    forward_wht4(col, out);
    for (int r = 0; r < 4; ++r) tmp[r * 4 + c] = out[r];
  }
  for (int r = 0; r < 4; ++r) {
    int row[4] = {tmp[r * 4 + 0], tmp[r * 4 + 1], tmp[r * 4 + 2],
                  tmp[r * 4 + 3]};
    int out[4] = {};
    forward_wht4(row, out);
    for (int c = 0; c < 4; ++c) levels[r * 4 + c] = out[c];
  }
}

void inverse_lossless_4x4(const int levels[16], int residual[16]) {
  int tmp[16] = {};
  // Decoder row pass: Dequant = level * 8, inverse WHT with shift 3.
  for (int r = 0; r < 4; ++r) {
    int row[4] = {levels[r * 4 + 0] * 8, levels[r * 4 + 1] * 8,
                  levels[r * 4 + 2] * 8, levels[r * 4 + 3] * 8};
    int out[4] = {};
    inverse_wht4(row, 3, out);
    for (int c = 0; c < 4; ++c) tmp[r * 4 + c] = out[c];
  }
  // Decoder column pass: inverse WHT with shift 0.
  for (int c = 0; c < 4; ++c) {
    int col[4] = {tmp[0 * 4 + c], tmp[1 * 4 + c], tmp[2 * 4 + c],
                  tmp[3 * 4 + c]};
    int out[4] = {};
    inverse_wht4(col, 0, out);
    for (int r = 0; r < 4; ++r) residual[r * 4 + c] = out[r];
  }
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
      for (int i = 1; i < h; ++i) {
        out[i * w + j] += out[(i - 1) * w + j];
      }
    }
  } else {
    for (int i = 0; i < h; ++i) {
      for (int j = 1; j < w; ++j) {
        out[i * w + j] += out[i * w + (j - 1)];
      }
    }
  }
}

}  // namespace

void dpcm_difference_4x4(const int residual[16], bool vertical, int out[16]) {
  dpcm_difference(residual, 4, 4, vertical, out);
}
void dpcm_cumsum_4x4(const int in[16], bool vertical, int out[16]) {
  dpcm_cumsum(in, 4, 4, vertical, out);
}
void dpcm_difference_8x8(const int residual[64], bool vertical, int out[64]) {
  dpcm_difference(residual, 8, 8, vertical, out);
}
void dpcm_cumsum_8x8(const int in[64], bool vertical, int out[64]) {
  dpcm_cumsum(in, 8, 8, vertical, out);
}

// ---------------------------------------------------------------------------
// §5.18.2 lossless flags
// ---------------------------------------------------------------------------

int qindex_for_segment(const LosslessInputs& inputs, int segment_id) {
  const int seg = clip3(0, kMaxSegments - 1, segment_id);
  int qindex = inputs.base_q_idx;
  if (inputs.alt_q_enabled[seg]) {
    qindex = inputs.base_q_idx + inputs.alt_q[seg];
  }
  return clip3(0, 255, qindex);
}

LosslessFlags compute_lossless_flags(const LosslessInputs& inputs) {
  LosslessFlags out;
  out.coded_lossless = true;
  const int segments =
      std::max(1, std::min(inputs.num_segments <= 0 ? kBaseSegments
                                                    : inputs.num_segments,
                           kMaxSegments));
  for (int s = 0; s < segments; ++s) {
    const int qindex = qindex_for_segment(inputs, s);
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
// CDF state
// ---------------------------------------------------------------------------

namespace {

template <typename T, std::size_t N>
void copy_from(const T (&src)[N], T (&dst)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

struct CoeffCdfs {
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_16[3][6];
  std::uint16_t eob_pt_32[3][7];
  std::uint16_t eob_pt_64[3][8];
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
  // FSC / IDTX paths.
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
  CoeffCdfs coeff;
};

void init_cdfs(CdfState& c) {
  std::memcpy(c.do_split, av2enc::basic::kDoSplitCdf[0], sizeof(c.do_split));
  std::memcpy(c.rect_type, av2enc::basic::kRectTypeCdf[0], sizeof(c.rect_type));
  copy_from(av2enc::basic::kYModeSetCdf, c.y_mode_set);
  copy_from(av2enc::basic::kYModeIndexCdf, c.y_mode_index);
  copy_from(av2enc::basic::kUvModeCflNotAllowedCdf, c.uv_mode);
  copy_from(kFscModeCdf, c.fsc_mode);
  copy_from(kUseDpcmYCdf, c.use_dpcm_y);
  copy_from(kDpcmModeYCdf, c.dpcm_mode_y);
  copy_from(kUseDpcmUvCdf, c.use_dpcm_uv);
  copy_from(kDpcmModeUvCdf, c.dpcm_mode_uv);
  copy_from(kLosslessTxSizeCdf, c.lossless_tx_size);

  const auto& q = av2enc::still::qctx::tables_for_slice(0);
  copy_from(*q.txb_skip, c.coeff.txb_skip);
  copy_from(*q.v_txb_skip, c.coeff.v_txb_skip);
  copy_from(*q.eob_pt_16, c.coeff.eob_pt_16);
  copy_from(*q.eob_pt_32, c.coeff.eob_pt_32);
  copy_from(*q.eob_pt_64, c.coeff.eob_pt_64);
  copy_from(*q.eob_extra, c.coeff.eob_extra);
  copy_from(*q.coeff_base, c.coeff.coeff_base);
  copy_from(*q.coeff_base_lf, c.coeff.coeff_base_lf);
  copy_from(*q.coeff_base_eob, c.coeff.coeff_base_eob);
  copy_from(*q.coeff_base_lf_eob, c.coeff.coeff_base_lf_eob);
  copy_from(*q.coeff_base_uv, c.coeff.coeff_base_uv);
  copy_from(*q.coeff_base_lf_uv, c.coeff.coeff_base_lf_uv);
  copy_from(*q.coeff_base_eob_uv, c.coeff.coeff_base_eob_uv);
  copy_from(*q.coeff_base_lf_eob_uv, c.coeff.coeff_base_lf_eob_uv);
  copy_from(*q.coeff_br, c.coeff.coeff_br);
  copy_from(*q.coeff_br_lf, c.coeff.coeff_br_lf);
  copy_from(*q.coeff_br_uv, c.coeff.coeff_br_uv);
  copy_from(*q.dc_sign, c.coeff.dc_sign);
  copy_from(*q.coeff_base_bob, c.coeff.coeff_base_bob);
  copy_from(*q.coeff_base_idtx, c.coeff.coeff_base_idtx);
  copy_from(*q.coeff_br_idtx, c.coeff.coeff_br_idtx);
  copy_from(*q.idtx_sign, c.coeff.idtx_sign);
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

struct Geometry {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int chroma_w = 0;
  int chroma_h = 0;
  int chroma_mi_cols = 0;
  int chroma_mi_rows = 0;
  int sb_cols = 0;
  int sb_rows = 0;
};

bool make_geometry(int width, int height, Geometry& g, std::string& error) {
  if (width <= 0 || height <= 0 || (width & 7) != 0 || (height & 7) != 0) {
    error = "lossless reference requires width/height positive multiples of 8";
    return false;
  }
  g.width = width;
  g.height = height;
  g.mi_cols = 2 * ((width + 7) >> 3);
  g.mi_rows = 2 * ((height + 7) >> 3);
  g.chroma_w = width / 2;
  g.chroma_h = height / 2;
  g.chroma_mi_cols = g.mi_cols / 2;
  g.chroma_mi_rows = g.mi_rows / 2;
  g.sb_cols = (g.mi_cols + 15) >> 4;
  g.sb_rows = (g.mi_rows + 15) >> 4;
  return true;
}

// ---------------------------------------------------------------------------
// Tile context state (§5.20.9.x availability / §8.3.2 neighbour contexts)
// ---------------------------------------------------------------------------

struct TileContext {
  int mi_cols = 0;
  int mi_rows = 0;
  int chroma_mi_cols = 0;
  int chroma_mi_rows = 0;
  std::vector<int> above_mi;
  std::vector<int> left_mi;
  std::array<std::vector<std::uint8_t>, 3> above_level;
  std::array<std::vector<std::uint8_t>, 3> left_level;
  std::array<std::vector<std::uint8_t>, 3> above_dc;
  std::array<std::vector<std::uint8_t>, 3> left_dc;
  std::vector<std::uint8_t> intra_joint;  // luma mi grid
  std::vector<std::uint8_t> fsc_modes;    // luma mi grid
  int eob_u = 0;                          // plane-1 EOB for plane-2 ctx

  void reset(const Geometry& g) {
    mi_cols = g.mi_cols;
    mi_rows = g.mi_rows;
    chroma_mi_cols = g.chroma_mi_cols;
    chroma_mi_rows = g.chroma_mi_rows;
    above_mi.assign(static_cast<std::size_t>(mi_cols) + 16, kSentinel256);
    left_mi.assign(static_cast<std::size_t>(mi_rows) + 16, kSentinel256);
    for (int p = 0; p < 3; ++p) {
      const int w = p == 0 ? mi_cols : chroma_mi_cols;
      const int h = p == 0 ? mi_rows : chroma_mi_rows;
      above_level[p].assign(static_cast<std::size_t>(w) + 16, 0);
      left_level[p].assign(static_cast<std::size_t>(h) + 16, 0);
      above_dc[p].assign(static_cast<std::size_t>(w) + 16, 0);
      left_dc[p].assign(static_cast<std::size_t>(h) + 16, 0);
    }
    intra_joint.assign(static_cast<std::size_t>(mi_cols) * mi_rows, 0);
    fsc_modes.assign(static_cast<std::size_t>(mi_cols) * mi_rows, 0);
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
    return r >= 0 && r < mi_rows && c >= 0 && c < mi_cols;
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
    if (inside(mv_row, mv_col)) {
      return intra_joint[static_cast<std::size_t>(mv_row) * mi_cols + mv_col];
    }
    return 0;
  }
};

// ---------------------------------------------------------------------------
// Symbol writer / reader wrappers
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
// Plane buffer and prediction
// ---------------------------------------------------------------------------

struct Plane {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> data;

  void resize(int width, int height) {
    w = width;
    h = height;
    data.assign(static_cast<std::size_t>(w) * h, 0);
  }
  std::uint8_t* row(int y) { return data.data() + static_cast<std::size_t>(y) * w; }
  const std::uint8_t* row(int y) const {
    return data.data() + static_cast<std::size_t>(y) * w;
  }
};

bool predict_tx(const Plane& plane, int x, int y, int w, int h, int mode,
                bool have_left, bool have_above, int bit_depth,
                std::uint8_t* dst, std::size_t dst_stride) {
  using intra::Mode;
  intra::NeighborRequest req;
  req.recon.data = plane.data.data();
  req.recon.stride = plane.w;
  req.x = x;
  req.y = y;
  req.w = w;
  req.h = h;
  req.max_x = plane.w - 1;
  req.max_y = plane.h - 1;
  req.have_left = have_left;
  req.have_above = have_above;
  req.num4_above_right = 0;
  req.num4_below_left = 0;
  req.color_plane = 0;
  intra::PredictParams params;
  switch (mode) {
    case kDcPred:
      params.mode = Mode::kDc;
      break;
    case kV:
      params.mode = Mode::kV;
      break;
    case kH:
      params.mode = Mode::kH;
      break;
    default:
      return false;
  }
  params.angle_delta = 0;
  params.enable_edge_filter = false;
  return intra::predict_block(req, bit_depth, params, dst, dst_stride) ==
         intra::IntraStatus::kOk;
}

// Reconstruct one transform from coefficient levels using the exact decoder
// path: inverse WHT (or identity for lossless IDTX), optional DPCM cumulative
// sum, then prediction add and Clip1.
void reconstruct_tx(const int* levels, int w, int h, bool idtx, bool dpcm,
                    bool vertical, const std::uint8_t* pred,
                    std::size_t pred_stride, std::uint8_t* recon,
                    std::size_t recon_stride) {
  int tmp[64] = {};
  int residual[64] = {};
  if (idtx) {
    for (int i = 0; i < w * h; ++i) tmp[i] = levels[i];
  } else {
    // Only 4x4 DCT_DCT is used for lossless non-FSC blocks.
    inverse_lossless_4x4(levels, tmp);
  }
  if (dpcm) {
    dpcm_cumsum(tmp, w, h, vertical, residual);
  } else {
    std::memcpy(residual, tmp, static_cast<std::size_t>(w) * h * sizeof(int));
  }
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int p = pred[static_cast<std::size_t>(i) * pred_stride + j];
      recon[static_cast<std::size_t>(i) * recon_stride + j] =
          static_cast<std::uint8_t>(clip3(0, 255, p + residual[i * w + j]));
    }
  }
}

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
  // 4:2:0 residual block, always at least 4x4.  The only invalid E/W combos
  // for the sizes produced by this encoder are those with a 4x4 chroma axis
  // from a 4x4 luma axis, which the tree never creates.
  return partition::block_width(sub) >= 8 && partition::block_height(sub) >= 8;
}

bool is_partition_allowed(int r, int c, BlockSize bsize, PartitionType p,
                          int chroma_offset, bool has_chroma,
                          int mi_rows, int mi_cols) {
  const BlockSize sub = partition::partition_subsize(p, bsize);
  if (sub == BlockSize::kInvalid) return false;

  // FrameIsIntra -> the inter-only BLOCK_4X4 restriction does not apply.
  if (has_chroma && !chroma_offset) {
    chroma_offset = is_chroma_offset_for_partition(p, bsize) ? 1 : 0;
  }
  if (has_chroma && !chroma_offset) {
    if (!plane_residual_valid(sub)) return false;
  }
  if (p == PartitionType::kNone) {
    const int half_w = partition::num_4x4_blocks_wide(bsize) >> 1;
    const int half_h = partition::num_4x4_blocks_high(bsize) >> 1;
    const bool has_rows = r + half_h < mi_rows;
    const bool has_cols = c + half_w < mi_cols;
    if (!has_rows || !has_cols) return false;
  }
  if (has_chroma && chroma_offset) {
    const int half_w = partition::num_4x4_blocks_wide(bsize) >> 1;
    const int half_h = partition::num_4x4_blocks_high(bsize) >> 1;
    if (p == PartitionType::kHorz) {
      return (r + half_h) < mi_rows;
    }
    if (p == PartitionType::kVert) {
      return (c + half_w) < mi_cols;
    }
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
                               int chroma_offset, bool has_chroma,
                               int mi_rows, int mi_cols) {
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
  if (has_rows && has_cols) {
    out.partition = PartitionType::kNone;
    return out;
  }
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
  // bSize == BLOCK_4X4, or bSize >= BLOCK_4X32 (enum >= 25), is implied NONE.
  if (bsize == BlockSize::k4x4 || static_cast<int>(bsize) >= 25) {
    out.partition = PartitionType::kNone;
    return out;
  }
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
// Coefficient syntax
// ---------------------------------------------------------------------------

int tx_size_ctx(int w, int h) {
  // Tx_Size_Sqr / Tx_Size_Sqr_Up for square 4x4 and 8x8.
  const int log2w = floor_log2(w);
  const int log2h = floor_log2(h);
  const int sqr = std::min(log2w, log2h) - 2;
  const int sqr_up = std::max(log2w, log2h) - 2;
  return (sqr + sqr_up + 1) >> 1;
}

int eob_multisize(int w, int h) {
  return std::min(floor_log2(w), 5) + std::min(floor_log2(h), 5) - 4;
}

const int* scan_for(int w, int h) {
  if (w == 4 && h == 4) return kScan4;
  if (w == 8 && h == 8) return kScan8;
  return nullptr;
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

bool write_eob(CdfState& c, SymWriter& sw, int eob, int plane, int w4, int h4,
               int w, int h) {
  (void)w4;
  (void)h4;
  const int multisize = eob_multisize(w, h);
  const int eob_ctx = plane > 0 ? 2 : 0;
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
    default:
      return false;
  }
  int eob_pt = eob_pt_from_eob(eob);
  if (eob_pt > symbols) return false;
  if (!sw.sym(cdf, symbols, eob_pt - 1)) return false;
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

bool read_eob(CdfState& c, SymReader& r, int plane, int w, int h, int* eob) {
  const int multisize = eob_multisize(w, h);
  const int eob_ctx = plane > 0 ? 2 : 0;
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
    default:
      return false;
  }
  int symbol = 0;
  if (!r.sym(cdf, symbols, &symbol)) return false;
  const int eob_pt = symbol + 1;
  int value = eob_from_eob_pt(eob_pt);
  if (eob_pt >= 3) {
    int extra = 0;
    if (!r.sym(&c.coeff.eob_extra[0], 2, &extra)) return false;
    if (extra) value += 1 << (eob_pt - 3);
    for (int i = eob_pt - 4; i >= 0; --i) {
      bool b = false;
      if (!r.bit(&b)) return false;
      if (b) value += 1 << i;
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

int dc_sign_context(const TileContext& ctx, int x4, int y4, int w4, int h4) {
  int sign = 0;
  for (int k = 0; k < w4; ++k) {
    const int c = x4 + k;
    if (c < ctx.mi_cols) {
      const int s = ctx.above_dc[0][static_cast<std::size_t>(c)];
      if (s == 1) --sign;
      else if (s == 2) ++sign;
    }
  }
  for (int k = 0; k < h4; ++k) {
    const int rr = y4 + k;
    if (rr < ctx.mi_rows) {
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

// Common all_zero context derivation.
int all_zero_context(const TileContext& ctx, int plane, int x4, int y4, int w4,
                     int h4, int bsize_w, int bsize_h, int w, int h,
                     bool fsc_mode, int eob_u) {
  if (plane == 0) {
    int top = 0;
    int left = 0;
    const int max_x = ctx.mi_cols;
    const int max_y = ctx.mi_rows;
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < max_x) top |= ctx.above_level[0][x4 + k];
    }
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < max_y) left |= ctx.left_level[0][y4 + k];
    }
    top = std::min(top, 4);
    left = std::min(left, 4);
    if (fsc_mode) return 9;  // TXB_SKIP_CONTEXTS - 1
    if (bsize_w == w && bsize_h == h) return 0;
    return (top + left + 3) >> 1;
  }
  int above = 0;
  int left = 0;
  const int max_x = ctx.chroma_mi_cols;
  const int max_y = ctx.chroma_mi_rows;
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

// FSC luma coefficient writer.  `levels` is in raster order.
CoeffResult write_coeff_fsc(CdfState& c, SymWriter& sw, TileContext& ctx,
                            int x4, int y4, int w4, int h4, int bsize_w,
                            int bsize_h, int w, int h, const int* levels,
                            bool fsc_mode) {
  CoeffResult out;
  const int tsc = tx_size_ctx(w, h);
  const int seg_eob = std::min(32, w) * std::min(32, h);
  const int* scan = scan_for(w, h);
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
  if (!write_eob(c, sw, eob, 0, w4, h4, w, h)) return out;

  int level_grid[64] = {};
  int sign_grid[64] = {};
  int hr_level_avg = 0;
  // First pass: base and br levels for every coded scan position.
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

  // Second pass: signs and read_quant residuals, in scan order from c=0.
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

// Normal coefficient writer (luma/chroma).
CoeffResult write_coeff_normal(CdfState& c, SymWriter& sw, TileContext& ctx,
                               int plane, int x4, int y4, int w4, int h4,
                               int bsize_w, int bsize_h, int w, int h,
                               const int* levels, bool fsc_block, int eob_u) {
  CoeffResult out;
  const int tsc = tx_size_ctx(w, h);
  const int seg_eob = std::min(32, w) * std::min(32, h);
  const int* scan = scan_for(w, h);
  int eob = 0;
  for (int i = seg_eob - 1; i >= 0; --i) {
    if (levels[scan[i]] != 0) {
      eob = i + 1;
      break;
    }
  }
  const int ctx0 = all_zero_context(ctx, plane, x4, y4, w4, h4, bsize_w,
                                    bsize_h, w, h, fsc_block, eob_u);
  std::uint16_t* skip_cdf =
      plane == 2 ? &c.coeff.v_txb_skip[ctx0][0]
                 : &c.coeff.txb_skip[fsc_block ? 1 : 0][tsc][ctx0][0];
  if (!sw.sym(skip_cdf, 2, eob == 0 ? 1 : 0)) return out;
  if (eob == 0) {
    clear_coeff_context(ctx, plane, x4, y4, w4, h4);
    return out;
  }
  if (!write_eob(c, sw, eob, plane, w4, h4, w, h)) return out;

  int level_pre[64] = {};
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
        if (!write_quant_escape(sw, absv - pre_level, hr_level_avg)) return out;
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

// Reader counterparts.  They return the raster-order levels.
bool read_coeff_normal(CdfState& c, SymReader& r, TileContext& ctx, int plane,
                       int x4, int y4, int w4, int h4, int bsize_w,
                       int bsize_h, int w, int h, bool fsc_block, int eob_u,
                       int* levels_raster, CoeffResult* out) {
  std::fill(levels_raster, levels_raster + w * h, 0);
  const int tsc = tx_size_ctx(w, h);
  const int seg_eob = std::min(32, w) * std::min(32, h);
  const int* scan = scan_for(w, h);
  const int ctx0 = all_zero_context(ctx, plane, x4, y4, w4, h4, bsize_w,
                                    bsize_h, w, h, fsc_block, eob_u);
  std::uint16_t* skip_cdf =
      plane == 2 ? &c.coeff.v_txb_skip[ctx0][0]
                 : &c.coeff.txb_skip[fsc_block ? 1 : 0][tsc][ctx0][0];
  int all_zero = 0;
  if (!r.sym(skip_cdf, 2, &all_zero)) return false;
  if (all_zero) {
    clear_coeff_context(ctx, plane, x4, y4, w4, h4);
    *out = {};
    return true;
  }
  int eob = 0;
  if (!read_eob(c, r, plane, w, h, &eob)) return false;
  if (eob <= 0 || eob > seg_eob) return false;

  int level_pre[64] = {};
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

  // §5.20.7.27: signs and the read_quant residual are a second pass over the
  // same scan-order coefficients.
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

bool read_coeff_fsc(CdfState& c, SymReader& r, TileContext& ctx, int x4,
                    int y4, int w4, int h4, int bsize_w, int bsize_h, int w,
                    int h, bool fsc_mode, int* levels_raster,
                    CoeffResult* out) {
  std::fill(levels_raster, levels_raster + w * h, 0);
  const int tsc = tx_size_ctx(w, h);
  const int seg_eob = std::min(32, w) * std::min(32, h);
  const int* scan = scan_for(w, h);
  const int ctx0 = all_zero_context(ctx, 0, x4, y4, w4, h4, bsize_w, bsize_h,
                                    w, h, fsc_mode, 0);
  int all_zero = 0;
  if (!r.sym(&c.coeff.txb_skip[1][tsc][ctx0][0], 2, &all_zero)) return false;
  if (all_zero) {
    clear_coeff_context(ctx, 0, x4, y4, w4, h4);
    *out = {};
    return true;
  }
  int eob = 0;
  if (!read_eob(c, r, 0, w, h, &eob)) return false;
  if (eob <= 0 || eob > seg_eob) return false;
  const int bob = seg_eob - eob;
  int level_grid[64] = {};
  int sign_grid[64] = {};
  int hr_level_avg = 0;
  // First pass: base and br levels.
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

  // Second pass: signs and read_quant, in scan order from c=0.
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
    if (!r.sym(&c.coeff.idtx_sign[std::min(2, tsc)][sign_ctx][0], 2, &sign)) {
      return false;
    }
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

// ---------------------------------------------------------------------------
// Mode syntax helpers
// ---------------------------------------------------------------------------

int fsc_context(const TileContext& ctx, int r, int c, int w4, int h4,
                int sb_size_mi) {
  int nnum = 0;
  int ctx_sum = 0;
  auto add = [&](int nr, int nc) {
    if (nnum >= 2) return;
    if (!ctx.inside(nr, nc)) return;
    const int sb_shift = floor_log2(sb_size_mi);
    if ((r >> sb_shift) != (nr >> sb_shift)) return;
    ctx_sum += ctx.fsc_modes[static_cast<std::size_t>(nr) * ctx.mi_cols + nc];
    ++nnum;
  };
  add(r + h4 - 1, c - 1);
  add(r - 1, c + w4 - 1);
  add(r, c - 1);
  add(r - 1, c);
  return ctx_sum;
}

// ---------------------------------------------------------------------------
// Lossless tile writer
// ---------------------------------------------------------------------------

struct BlockPlan {
  int y_mode = kDcPred;
  bool use_dpcm_y = false;
  bool dpcm_y_vertical = false;
  bool fsc = false;
  int uv_mode = kDcPred;
  bool use_dpcm_uv = false;
  bool dpcm_uv_vertical = false;
};

struct PredictionScratch {
  std::uint8_t pred[64] = {};
  int residual[64] = {};
  int tmp[64] = {};
  int levels[64] = {};
};

class TileWriter {
 public:
  TileWriter(const Geometry& g, const Options& opt, CdfState& cdfs,
             TileContext& ctx, Plane* src[3], Plane* recon[3], Stats* stats)
      : g_(g), opt_(opt), cdfs_(cdfs), ctx_(ctx), stats_(stats) {
    for (int p = 0; p < 3; ++p) {
      src_[p] = src[p];
      recon_[p] = recon[p];
    }
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
  bool write_partition(int r, int c, BlockSize bsize, int chroma_offset,
                       bool has_chroma, std::string* error) {
    if (r >= g_.mi_rows || c >= g_.mi_cols) return true;
    const AllowedPartitions allowed =
        init_allowed(r, c, bsize, chroma_offset, has_chroma, g_.mi_rows,
                     g_.mi_cols);
    const ImpliedPartition implied =
        partition_implied(r, c, bsize, g_.mi_rows, g_.mi_cols);
    PartitionType part = PartitionType::kNone;
    bool emit = false;
    if (implied.implied &&
        ((implied.partition == PartitionType::kNone && allowed.none) ||
         (implied.partition == PartitionType::kHorz && allowed.horz) ||
         (implied.partition == PartitionType::kVert && allowed.vert))) {
      part = implied.partition;
    } else if (allowed.count == 1) {
      part = allowed.none ? PartitionType::kNone
                          : (allowed.horz ? PartitionType::kHorz
                                          : PartitionType::kVert);
    } else {
      const int bw = partition::block_width(bsize);
      const int bh = partition::block_height(bsize);
      PartitionType preferred =
          bh > bw ? PartitionType::kHorz : PartitionType::kVert;
      if (preferred == PartitionType::kHorz && !allowed.horz) {
        preferred = PartitionType::kVert;
      } else if (preferred == PartitionType::kVert && !allowed.vert) {
        preferred = PartitionType::kHorz;
      }
      const bool preferred_ok =
          (preferred == PartitionType::kHorz && allowed.horz) ||
          (preferred == PartitionType::kVert && allowed.vert);
      if (preferred_ok && bsize != BlockSize::k8x8) {
        part = preferred;
      } else if (bsize != BlockSize::k8x8) {
        *error = "unsupported lossless partition geometry (no 8x8 leaf)";
        return false;
      } else {
        part = PartitionType::kNone;
      }
      emit = true;
    }
    if (emit) {
      if (allowed.none && part == PartitionType::kNone) {
        if (!w_.sym(&cdfs_.do_split[do_split_context(ctx_, r, c, bsize)][0], 2,
                    0)) {
          *error = w_.error();
          return false;
        }
      } else if (allowed.none) {
        if (!w_.sym(&cdfs_.do_split[do_split_context(ctx_, r, c, bsize)][0], 2,
                    1)) {
          *error = w_.error();
          return false;
        }
      }
      if (part == PartitionType::kHorz || part == PartitionType::kVert) {
        PartitionType rect = PartitionType::kNone;
        if (!rect_type_implied(bsize, &rect)) {
          const int symbol = part == PartitionType::kHorz ? 0 : 1;
          if (!w_.sym(&cdfs_.rect_type[rect_type_context(ctx_, r, c, bsize)][0],
                      2, symbol)) {
            *error = w_.error();
            return false;
          }
        }
      }
    }
    if (part == PartitionType::kNone) {
      if (bsize != BlockSize::k8x8) {
        *error = "lossless reference only codes 8x8 coding blocks";
        return false;
      }
      return write_block(r, c, error);
    }
    const int half_w = partition::num_4x4_blocks_wide(bsize) >> 1;
    const int half_h = partition::num_4x4_blocks_high(bsize) >> 1;
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

  int luma_cost(const BlockPlan& plan, int r, int c) {
    // Use the source as the reference; lossless reconstruction equals source,
    // so this is exactly the decoder's prediction reference.
    PredictionScratch s;
    const Plane& p = *src_[0];
    const int base_x = c * 4;
    const int base_y = r * 4;
    int total = 0;
    for (int ty = 0; ty < 2; ++ty) {
      for (int tx = 0; tx < 2; ++tx) {
        const int x = base_x + tx * 4;
        const int y = base_y + ty * 4;
        const bool have_left = (c > 0) || (tx > 0);
        const bool have_above = (r > 0) || (ty > 0);
        if (!predict_tx(p, x, y, 4, 4, plan.y_mode, have_left, have_above, 8,
                        s.pred, 4)) {
          return std::numeric_limits<int>::max();
        }
        for (int i = 0; i < 16; ++i) {
          const int src = p.data[static_cast<std::size_t>(y + i / 4) * p.w +
                                (x + i % 4)];
          s.residual[i] = src - s.pred[i];
        }
        if (plan.use_dpcm_y) {
          dpcm_difference(s.residual, 4, 4, plan.dpcm_y_vertical, s.tmp);
        } else {
          std::memcpy(s.tmp, s.residual, sizeof(s.tmp));
        }
        forward_lossless_4x4(s.tmp, s.levels);
        for (int i = 0; i < 16; ++i) total += abs_int(s.levels[i]);
      }
    }
    return total;
  }

  int chroma_cost(const BlockPlan& plan, int r, int c, int plane) {
    PredictionScratch s;
    const Plane& p = *src_[plane];
    const int x = c * 2;
    const int y = r * 2;
    const bool have_left = c > 0;
    const bool have_above = r > 0;
    if (!predict_tx(p, x, y, 4, 4, plan.uv_mode, have_left, have_above, 8,
                    s.pred, 4)) {
      return std::numeric_limits<int>::max();
    }
    for (int i = 0; i < 16; ++i) {
      const int src = p.data[static_cast<std::size_t>(y + i / 4) * p.w +
                            (x + i % 4)];
      s.residual[i] = src - s.pred[i];
    }
    if (plan.use_dpcm_uv) {
      dpcm_difference(s.residual, 4, 4, plan.dpcm_uv_vertical, s.tmp);
    } else {
      std::memcpy(s.tmp, s.residual, sizeof(s.tmp));
    }
    forward_lossless_4x4(s.tmp, s.levels);
    int total = 0;
    for (int i = 0; i < 16; ++i) total += abs_int(s.levels[i]);
    return total;
  }

  BlockPlan choose_plan(int r, int c) {
    BlockPlan best;
    best.y_mode = kDcPred;
    best.use_dpcm_y = false;
    best.uv_mode = kDcPred;
    best.use_dpcm_uv = false;

    // Luma candidate search.
    const bool force_fsc = opt_.fsc == FscMode::kForce;
    if (!force_fsc) {
      int best_cost = luma_cost(best, r, c);
      if (opt_.enable_dpcm) {
        BlockPlan v;
        v.y_mode = kV;
        v.use_dpcm_y = true;
        v.dpcm_y_vertical = true;
        const int vc = luma_cost(v, r, c);
        if (vc < best_cost) {
          best_cost = vc;
          best = v;
        }
        BlockPlan h;
        h.y_mode = kH;
        h.use_dpcm_y = true;
        h.dpcm_y_vertical = false;
        const int hc = luma_cost(h, r, c);
        if (hc < best_cost) {
          best_cost = hc;
          best = h;
        }
      }
    }

    // Chroma candidate search.  Keep the luma decision.
    const int base_chroma = chroma_cost(best, r, c, 1) +
                            chroma_cost(best, r, c, 2);
    int best_chroma_cost = base_chroma;
    if (opt_.enable_dpcm) {
      BlockPlan v = best;
      v.use_dpcm_uv = true;
      v.dpcm_uv_vertical = true;
      const int vc = chroma_cost(v, r, c, 1) + chroma_cost(v, r, c, 2);
      if (vc < best_chroma_cost) {
        best_chroma_cost = vc;
        best.use_dpcm_uv = true;
        best.dpcm_uv_vertical = true;
        best.uv_mode = kV;
      }
      BlockPlan h = best;
      h.use_dpcm_uv = true;
      h.dpcm_uv_vertical = false;
      const int hc = chroma_cost(h, r, c, 1) + chroma_cost(h, r, c, 2);
      if (hc < best_chroma_cost) {
        best.use_dpcm_uv = true;
        best.dpcm_uv_vertical = false;
        best.uv_mode = kH;
      }
    }

    // FSC decision.  Kept simple and explicit: force -> every block; auto ->
    // only when the residual has a long zero prefix in the 2D scan (the FSC
    // code then skips those coefficients).
    if (opt_.fsc == FscMode::kForce) {
      best.fsc = true;
      best.use_dpcm_y = false;
      best.y_mode = kDcPred;
    } else if (opt_.fsc == FscMode::kAuto) {
      PredictionScratch s;
      const Plane& p = *src_[0];
      const int x = c * 4;
      const int y = r * 4;
      if (predict_tx(p, x, y, 8, 8, kDcPred, c > 0, r > 0, 8, s.pred, 8)) {
        for (int i = 0; i < 64; ++i) {
          const int src = p.data[static_cast<std::size_t>(y + i / 8) * p.w +
                                (x + i % 8)];
          s.residual[i] = src - s.pred[i];
        }
        int first = -1;
        for (int i = 0; i < 64; ++i) {
          if (s.residual[kScan8[i]] != 0) {
            first = i;
            break;
          }
        }
        if (first > 0) best.fsc = true;
      }
    }
    return best;
  }

  bool write_block(int r, int c, std::string* error) {
    // External block stage (CUDA `still_lossless`): use its per-block modes and
    // levels instead of the host DPCM/FSC search and forward transform.  The
    // prediction below still reads the reconstruction, whose decoded samples
    // equal the source for a lossless frame, so the GPU computed the same
    // residual from the source plane.
    const BlockCoding* pre = nullptr;
    BlockPlan plan;
    if (opt_.precomputed != nullptr) {
      const int blocks_x = g_.mi_cols / 2;
      const int index = (r / 2) * blocks_x + (c / 2);
      pre = &opt_.precomputed[index];
      plan.y_mode = pre->y_mode;
      plan.use_dpcm_y = pre->use_dpcm_y != 0;
      plan.dpcm_y_vertical = pre->dpcm_y_vertical != 0;
      plan.fsc = pre->fsc != 0;
      plan.uv_mode = pre->uv_mode;
      plan.use_dpcm_uv = pre->use_dpcm_uv != 0;
      plan.dpcm_uv_vertical = pre->dpcm_uv_vertical != 0;
    } else {
      plan = choose_plan(r, c);
    }
    const int w4 = 2;
    const int h4 = 2;
    const int sb_mi = 16;
    // --- mode info ---
    // Luma: lossless use_dpcm_y syntax, then optional fsc_mode.
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
    const bool fsc_allowed = opt_.fsc != FscMode::kOff;
    if (fsc_allowed) {
      const int fctx = fsc_context(ctx_, r, c, w4, h4, sb_mi);
      const int group = kFscBsizeGroups[static_cast<int>(BlockSize::k8x8)];
      if (!w_.sym(&cdfs_.fsc_mode[fctx][group][0], 2, plan.fsc ? 1 : 0)) {
        *error = w_.error();
        return false;
      }
    }

    // Chroma mode.
    if (plan.use_dpcm_uv) {
      if (!w_.sym(&cdfs_.use_dpcm_uv[0], 2, 1)) {
        *error = w_.error();
        return false;
      }
      if (!w_.sym(&cdfs_.dpcm_mode_uv[0], 2, plan.dpcm_uv_vertical ? 0 : 1)) {
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

    // --- tx size (lossless_tx_size only for fsc blocks) ---
    if (plan.fsc) {
      const int group = kSizeGroup[static_cast<int>(BlockSize::k8x8)];
      if (!w_.sym(&cdfs_.lossless_tx_size[group][0][0], 2, 1)) {
        *error = w_.error();
        return false;
      }
    }

    // Store mode/joint/fsc context and partition context exactly where
    // decode_block does.
    const int joint = plan.use_dpcm_y ? (plan.dpcm_y_vertical ? 22 : 50) : 0;
    for (int y = 0; y < h4; ++y) {
      for (int x = 0; x < w4; ++x) {
        ctx_.intra_joint[static_cast<std::size_t>(r + y) * g_.mi_cols +
                         (c + x)] = static_cast<std::uint8_t>(joint);
        ctx_.fsc_modes[static_cast<std::size_t>(r + y) * g_.mi_cols + (c + x)] =
            static_cast<std::uint8_t>(plan.fsc ? 1 : 0);
        ctx_.left_mi[r + y] = static_cast<int>(BlockSize::k8x8);
        ctx_.above_mi[c + x] = static_cast<int>(BlockSize::k8x8);
      }
    }

    // --- luma residual ---
    if (plan.fsc) {
      if (!write_luma_fsc(r, c, plan, pre, error)) return false;
    } else {
      if (!write_luma_4x4(r, c, plan, pre, error)) return false;
    }

    // --- chroma residual ---
    if (!write_chroma(r, c, plan, pre, error)) return false;

    ++stats_->luma_blocks;
    if (plan.fsc) ++stats_->fsc_blocks;
    if (plan.use_dpcm_y) ++stats_->dpcm_y_blocks;
    if (plan.use_dpcm_uv) ++stats_->dpcm_uv_blocks;
    return true;
  }

  bool write_luma_4x4(int r, int c, const BlockPlan& plan,
                      const BlockCoding* pre, std::string* error) {
    PredictionScratch s;
    const Plane& srcp = *src_[0];
    const Plane& predp = *recon_[0];
    const int base_x = c * 4;
    const int base_y = r * 4;
    for (int ty = 0; ty < 2; ++ty) {
      for (int tx = 0; tx < 2; ++tx) {
        const int x = base_x + tx * 4;
        const int y = base_y + ty * 4;
        const bool have_left = (c > 0) || (tx > 0);
        const bool have_above = (r > 0) || (ty > 0);
        if (!predict_tx(predp, x, y, 4, 4, plan.y_mode, have_left, have_above,
                        8, s.pred, 4)) {
          *error = "luma prediction failed";
          return false;
        }
        for (int i = 0; i < 16; ++i) {
          const int src = srcp.data[static_cast<std::size_t>(y + i / 4) *
                                        srcp.w + (x + i % 4)];
          s.residual[i] = src - s.pred[i];
        }
        if (pre != nullptr) {
          std::memcpy(s.levels, pre->luma[ty * 2 + tx], sizeof(s.levels));
        } else {
          if (plan.use_dpcm_y) {
            dpcm_difference(s.residual, 4, 4, plan.dpcm_y_vertical, s.tmp);
          } else {
            std::memcpy(s.tmp, s.residual, sizeof(s.tmp));
          }
          forward_lossless_4x4(s.tmp, s.levels);
        }
        const int x4 = (x >> 2);
        const int y4 = (y >> 2);
        write_coeff_normal(cdfs_, w_, ctx_, 0, x4, y4, 1, 1, 8, 8, 4, 4,
                           s.levels, false, 0);
        if (!w_.ok()) {
          *error = w_.error();
          return false;
        }
        // Verify by running the exact decoder path.
        reconstruct_tx(s.levels, 4, 4, false, plan.use_dpcm_y,
                       plan.dpcm_y_vertical, s.pred, 4,
                       recon_[0]->data.data() +
                           static_cast<std::size_t>(y) * recon_[0]->w + x,
                       recon_[0]->w);
      }
    }
    return true;
  }

  bool write_luma_fsc(int r, int c, const BlockPlan& plan,
                      const BlockCoding* pre, std::string* error) {
    PredictionScratch s;
    const Plane& srcp = *src_[0];
    const Plane& predp = *recon_[0];
    const int x = c * 4;
    const int y = r * 4;
    if (!predict_tx(predp, x, y, 8, 8, plan.y_mode, c > 0, r > 0, 8, s.pred,
                    8)) {
      *error = "luma FSC prediction failed";
      return false;
    }
    for (int i = 0; i < 64; ++i) {
      const int src = srcp.data[static_cast<std::size_t>(y + i / 8) * srcp.w +
                                (x + i % 8)];
      s.residual[i] = src - s.pred[i];
    }
    // FSC blocks use IDTX: levels are the residual (or its DPCM difference).
    if (pre != nullptr) {
      std::memcpy(s.levels, pre->luma_fsc, sizeof(s.levels));
    } else if (plan.use_dpcm_y) {
      dpcm_difference(s.residual, 8, 8, plan.dpcm_y_vertical, s.levels);
    } else {
      std::memcpy(s.levels, s.residual, sizeof(s.levels));
    }
    const int x4 = x >> 2;
    const int y4 = y >> 2;
    write_coeff_fsc(cdfs_, w_, ctx_, x4, y4, 2, 2, 8, 8, 8, 8, s.levels,
                    true);
    if (!w_.ok()) {
      *error = w_.error();
      return false;
    }
    // FSC luma chroma transform is derived as IDTX; reconstruction is
    // identity + optional DPCM.
    reconstruct_tx(s.levels, 8, 8, true, plan.use_dpcm_y,
                   plan.dpcm_y_vertical, s.pred, 8,
                   recon_[0]->data.data() +
                       static_cast<std::size_t>(y) * recon_[0]->w + x,
                   recon_[0]->w);
    return true;
  }

  bool write_chroma(int r, int c, const BlockPlan& plan,
                    const BlockCoding* pre, std::string* error) {
    const int cx = c * 2;
    const int cy = r * 2;
    int eob1 = 0;
    for (int plane = 1; plane < 3; ++plane) {
      PredictionScratch s;
      const Plane& srcp = *src_[plane];
      const Plane& predp = *recon_[plane];
      if (!predict_tx(predp, cx, cy, 4, 4, plan.uv_mode, c > 0, r > 0, 8,
                      s.pred, 4)) {
        *error = "chroma prediction failed";
        return false;
      }
      for (int i = 0; i < 16; ++i) {
        const int src = srcp.data[static_cast<std::size_t>(cy + i / 4) *
                                      srcp.w + (cx + i % 4)];
        s.residual[i] = src - s.pred[i];
      }
      if (pre != nullptr) {
        std::memcpy(s.levels, pre->chroma[plane - 1], sizeof(s.levels));
      } else {
        if (plan.use_dpcm_uv) {
          dpcm_difference(s.residual, 4, 4, plan.dpcm_uv_vertical, s.tmp);
        } else {
          std::memcpy(s.tmp, s.residual, sizeof(s.tmp));
        }
        if (plan.fsc) {
          std::memcpy(s.levels, s.tmp, sizeof(s.levels));
        } else {
          forward_lossless_4x4(s.tmp, s.levels);
        }
      }
      const int x4 = cx >> 2;
      const int y4 = cy >> 2;
      const int prev_eob_u = ctx_.eob_u;
      const CoeffResult cr = write_coeff_normal(
          cdfs_, w_, ctx_, plane, x4, y4, 1, 1, 4, 4, 4, 4, s.levels,
          plan.fsc, plane == 2 ? prev_eob_u : 0);
      if (!w_.ok()) {
        *error = w_.error();
        return false;
      }
      if (plane == 1) {
        eob1 = cr.eob;
        ctx_.eob_u = eob1;
      }
      reconstruct_tx(s.levels, 4, 4, plan.fsc, plan.use_dpcm_uv,
                     plan.dpcm_uv_vertical, s.pred, 4,
                     recon_[plane]->data.data() +
                         static_cast<std::size_t>(cy) * recon_[plane]->w + cx,
                     recon_[plane]->w);
    }
    ctx_.eob_u = 0;
    return true;
  }

  const Geometry& g_;
  const Options& opt_;
  CdfState& cdfs_;
  TileContext& ctx_;
  Stats* stats_;
  Plane* src_[3] = {};
  Plane* recon_[3] = {};
  SymWriter w_{};
};

// ---------------------------------------------------------------------------
// Lossless tile reader
// ---------------------------------------------------------------------------

class TileReader {
 public:
  TileReader(const Geometry& g, const Options& opt, CdfState& cdfs,
             TileContext& ctx, Plane* recon[3], Stats* stats)
      : g_(g), opt_(opt), cdfs_(cdfs), ctx_(ctx), stats_(stats) {
    for (int p = 0; p < 3; ++p) recon_[p] = recon[p];
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
    PartitionType part = PartitionType::kNone;
    bool need_symbols = false;
    if (implied.implied &&
        ((implied.partition == PartitionType::kNone && allowed.none) ||
         (implied.partition == PartitionType::kHorz && allowed.horz) ||
         (implied.partition == PartitionType::kVert && allowed.vert))) {
      part = implied.partition;
    } else if (allowed.count == 1) {
      part = allowed.none ? PartitionType::kNone
                          : (allowed.horz ? PartitionType::kHorz
                                          : PartitionType::kVert);
    } else {
      need_symbols = true;
    }
    if (need_symbols) {
      int do_split = 0;
      if (allowed.none) {
        if (!r_.sym(&cdfs_.do_split[do_split_context(ctx_, r, c, bsize)][0], 2,
                    &do_split)) {
          *error = r_.error();
          return false;
        }
        if (!do_split) {
          part = PartitionType::kNone;
        }
      }
      if (do_split || !allowed.none) {
        PartitionType rect = PartitionType::kNone;
        if (rect_type_implied(bsize, &rect)) {
          part = rect;
        } else {
          int symbol = 0;
          if (!r_.sym(&cdfs_.rect_type[rect_type_context(ctx_, r, c, bsize)][0],
                      2, &symbol)) {
            *error = r_.error();
            return false;
          }
          part = symbol == 0 ? PartitionType::kHorz : PartitionType::kVert;
        }
      }
    }
    if (part == PartitionType::kNone) {
      if (bsize != BlockSize::k8x8) {
        *error = "lossless reader only supports 8x8 coding blocks (r=" +
                 std::to_string(r) + " c=" + std::to_string(c) +
                 " bsize=" + std::to_string(static_cast<int>(bsize)) + ")";
        return false;
      }
      return read_block(r, c, error);
    }
    const int half_w = partition::num_4x4_blocks_wide(bsize) >> 1;
    const int half_h = partition::num_4x4_blocks_high(bsize) >> 1;
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

  bool read_block(int r, int c, std::string* error) {
    const int w4 = 2;
    const int h4 = 2;
    const int sb_mi = 16;

    int use_dpcm_y = 0;
    if (!r_.sym(&cdfs_.use_dpcm_y[0], 2, &use_dpcm_y)) {
      *error = r_.error();
      return false;
    }
    int y_mode = kDcPred;
    bool dpcm_y_vertical = false;
    if (use_dpcm_y) {
      int mode = 0;
      if (!r_.sym(&cdfs_.dpcm_mode_y[0], 2, &mode)) {
        *error = r_.error();
        return false;
      }
      dpcm_y_vertical = mode == 0;
      y_mode = dpcm_y_vertical ? kV : kH;
    } else {
      int set = 0;
      if (!r_.sym(&cdfs_.y_mode_set[0], 4, &set)) {
        *error = r_.error();
        return false;
      }
      if (set != 0) {
        *error = "lossless reader: non-DC y_mode_set unsupported";
        return false;
      }
      const int jctx =
          (ctx_.joint_mode(r, c, w4, h4, 0) >= 5 ? 1 : 0) +
          (ctx_.joint_mode(r, c, w4, h4, 1) >= 5 ? 1 : 0);
      int index = 0;
      if (!r_.sym(&cdfs_.y_mode_index[jctx][0], 8, &index)) {
        *error = r_.error();
        return false;
      }
      if (index != 0) {
        *error = "lossless reader: only DC y_mode_index unsupported";
        return false;
      }
      y_mode = kDcPred;
    }
    bool fsc = false;
    if (opt_.fsc != FscMode::kOff) {
      const int fctx = fsc_context(ctx_, r, c, w4, h4, sb_mi);
      const int group = kFscBsizeGroups[static_cast<int>(BlockSize::k8x8)];
      int symbol = 0;
      if (!r_.sym(&cdfs_.fsc_mode[fctx][group][0], 2, &symbol)) {
        *error = r_.error();
        return false;
      }
      fsc = symbol != 0;
    }

    int use_dpcm_uv = 0;
    if (!r_.sym(&cdfs_.use_dpcm_uv[0], 2, &use_dpcm_uv)) {
      *error = r_.error();
      return false;
    }
    int uv_mode = kDcPred;
    bool dpcm_uv_vertical = false;
    if (use_dpcm_uv) {
      int mode = 0;
      if (!r_.sym(&cdfs_.dpcm_mode_uv[0], 2, &mode)) {
        *error = r_.error();
        return false;
      }
      dpcm_uv_vertical = mode == 0;
      uv_mode = dpcm_uv_vertical ? kV : kH;
    } else {
      const int uv_ctx = y_mode_is_directional(y_mode) ? 1 : 0;
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
      static constexpr int kDefaultUv[kIntraModesCount] = {
          kDcPred, 9, 10, 11, 12, kV, kH, 3, 4, 8, 5, 6, 7};
      int resolved = -1;
      int count = 0;
      if (y_mode_is_directional(y_mode)) {
        if (idx == 0) resolved = y_mode;
        else {
          count = 1;
          for (int i = 0; i < kIntraModesCount; ++i) {
            if (kDefaultUv[i] == y_mode) continue;
            if (count == idx) {
              resolved = kDefaultUv[i];
              break;
            }
            ++count;
          }
        }
      } else {
        for (int i = 0; i < kIntraModesCount; ++i) {
          if (count == idx) {
            resolved = kDefaultUv[i];
            break;
          }
          ++count;
        }
      }
      if (resolved < 0) {
        *error = "lossless reader: invalid uv_mode index";
        return false;
      }
      uv_mode = resolved;
    }

    int luma_tx = kTx4x4;
    if (fsc) {
      const int group = kSizeGroup[static_cast<int>(BlockSize::k8x8)];
      int lossless_tx_size = 0;
      if (!r_.sym(&cdfs_.lossless_tx_size[group][0][0], 2,
                  &lossless_tx_size)) {
        *error = r_.error();
        return false;
      }
      luma_tx = lossless_tx_size ? kTx8x8 : kTx4x4;
    }

    const int joint = use_dpcm_y ? (dpcm_y_vertical ? 22 : 50) : 0;
    for (int y = 0; y < h4; ++y) {
      for (int x = 0; x < w4; ++x) {
        ctx_.intra_joint[static_cast<std::size_t>(r + y) * g_.mi_cols +
                         (c + x)] = static_cast<std::uint8_t>(joint);
        ctx_.fsc_modes[static_cast<std::size_t>(r + y) * g_.mi_cols + (c + x)] =
            static_cast<std::uint8_t>(fsc ? 1 : 0);
        ctx_.left_mi[r + y] = static_cast<int>(BlockSize::k8x8);
        ctx_.above_mi[c + x] = static_cast<int>(BlockSize::k8x8);
      }
    }

    if (fsc) {
      if (luma_tx != kTx8x8) {
        *error = "lossless reader: FSC 4x4 luma tx unsupported";
        return false;
      }
      int levels[64] = {};
      CoeffResult cr;
      if (!read_coeff_fsc(cdfs_, r_, ctx_, c * 4 / 4, r * 4 / 4, 2, 2, 8, 8,
                          8, 8, true, levels, &cr)) {
        *error = r_.error();
        return false;
      }
      if (!recon_luma(r, c, y_mode, use_dpcm_y, dpcm_y_vertical, true, 8, 8,
                      levels)) {
        *error = "luma FSC reconstruction failed";
        return false;
      }
    } else {
      for (int ty = 0; ty < 2; ++ty) {
        for (int tx = 0; tx < 2; ++tx) {
          int levels[16] = {};
          CoeffResult cr;
          const int x4 = c + tx;
          const int y4 = r + ty;
          if (!read_coeff_normal(cdfs_, r_, ctx_, 0, x4, y4, 1, 1, 8, 8, 4,
                                 4, false, 0, levels, &cr)) {
            *error = r_.error();
            return false;
          }
          if (!recon_luma_4x4(r, c, tx, ty, y_mode, use_dpcm_y,
                              dpcm_y_vertical, levels)) {
            *error = "luma reconstruction failed";
            return false;
          }
        }
      }
    }

    // Chroma.
    int eob1 = 0;
    for (int plane = 1; plane < 3; ++plane) {
      int levels[16] = {};
      CoeffResult cr;
      const int prev_eob_u = ctx_.eob_u;
      if (!read_coeff_normal(cdfs_, r_, ctx_, plane, c / 2, r / 2, 1, 1, 4, 4,
                             4, 4, fsc, plane == 2 ? prev_eob_u : 0, levels,
                             &cr)) {
        *error = r_.error();
        return false;
      }
      if (plane == 1) {
        eob1 = cr.eob;
        ctx_.eob_u = eob1;
      }
      if (!recon_chroma(r, c, plane, uv_mode, use_dpcm_uv, dpcm_uv_vertical,
                        fsc, levels)) {
        *error = "chroma reconstruction failed";
        return false;
      }
    }
    ctx_.eob_u = 0;

    ++stats_->luma_blocks;
    if (fsc) ++stats_->fsc_blocks;
    if (use_dpcm_y) ++stats_->dpcm_y_blocks;
    if (use_dpcm_uv) ++stats_->dpcm_uv_blocks;
    return true;
  }

  bool recon_luma_4x4(int r, int c, int tx, int ty, int y_mode, bool dpcm,
                      bool vertical, const int levels[16]) {
    PredictionScratch s;
    const Plane& recp = *recon_[0];
    const int x = c * 4 + tx * 4;
    const int y = r * 4 + ty * 4;
    const bool have_left = (c > 0) || (tx > 0);
    const bool have_above = (r > 0) || (ty > 0);
    if (!predict_tx(recp, x, y, 4, 4, y_mode, have_left, have_above, 8,
                    s.pred, 4)) {
      return false;
    }
    reconstruct_tx(levels, 4, 4, false, dpcm, vertical, s.pred, 4,
                   recon_[0]->data.data() +
                       static_cast<std::size_t>(y) * recon_[0]->w + x,
                   recon_[0]->w);
    return true;
  }

  bool recon_luma(int r, int c, int y_mode, bool dpcm, bool vertical,
                  bool idtx, int w, int h, const int* levels) {
    PredictionScratch s;
    const Plane& recp = *recon_[0];
    const int x = c * 4;
    const int y = r * 4;
    if (!predict_tx(recp, x, y, w, h, y_mode, c > 0, r > 0, 8, s.pred, w)) {
      return false;
    }
    reconstruct_tx(levels, w, h, idtx, dpcm, vertical, s.pred, w,
                   recon_[0]->data.data() +
                       static_cast<std::size_t>(y) * recon_[0]->w + x,
                   recon_[0]->w);
    return true;
  }

  bool recon_chroma(int r, int c, int plane, int uv_mode, bool dpcm,
                    bool vertical, bool idtx, const int levels[16]) {
    PredictionScratch s;
    const Plane& recp = *recon_[plane];
    const int x = c * 2;
    const int y = r * 2;
    if (!predict_tx(recp, x, y, 4, 4, uv_mode, c > 0, r > 0, 8, s.pred, 4)) {
      return false;
    }
    reconstruct_tx(levels, 4, 4, idtx, dpcm, vertical, s.pred, 4,
                   recon_[plane]->data.data() +
                       static_cast<std::size_t>(y) * recon_[plane]->w + x,
                   recon_[plane]->w);
    return true;
  }

  const Geometry& g_;
  const Options& opt_;
  CdfState& cdfs_;
  TileContext& ctx_;
  Stats* stats_;
  Plane* recon_[3] = {};
  SymReader r_{};
};

// ---------------------------------------------------------------------------
// OBU / header syntax
// ---------------------------------------------------------------------------

void put_uvlc(av2enc::BitWriter& w, std::uint32_t value) {
  unsigned lz = 0;
  while (value >= ((1ULL << (lz + 1)) - 1ULL)) ++lz;
  w.put_bits(0, lz);
  w.put_bit(true);
  w.put_bits(static_cast<std::uint64_t>(value) - ((1ULL << lz) - 1ULL), lz);
}

bool get_uvlc(const std::uint8_t* data, std::size_t bits, std::size_t* pos,
              std::uint32_t* value) {
  unsigned lz = 0;
  while (true) {
    if (*pos >= bits) return false;
    const bool bit = ((data[*pos >> 3] >> (7 - (*pos & 7))) & 1) != 0;
    ++*pos;
    if (bit) break;
    ++lz;
    if (lz > 31) return false;
  }
  std::uint64_t payload = 0;
  for (unsigned i = 0; i < lz; ++i) {
    if (*pos >= bits) return false;
    const std::uint64_t bit =
        ((data[*pos >> 3] >> (7 - (*pos & 7))) & 1) != 0;
    ++*pos;
    payload = (payload << 1) | bit;
  }
  *value = static_cast<std::uint32_t>(((1ULL << lz) - 1ULL) + payload);
  return true;
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
                                   std::span<const std::uint8_t> payload) {
  // §5.2.2: obu_header_extension_flag f(1), obu_type f(5), tlayer f(2).
  std::vector<std::uint8_t> body;
  body.push_back(static_cast<std::uint8_t>((type & 0x1f) << 2));
  body.insert(body.end(), payload.begin(), payload.end());
  std::vector<std::uint8_t> out;
  append_leb128(out, static_cast<std::uint32_t>(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::size_t write_sequence_header(av2enc::BitWriter& w, const Options& opt,
                                  int width, int height) {
  const std::size_t start = w.bit_size();
  put_uvlc(w, 0);              // seq_header_id
  w.put_bits(0, 5);            // seq_profile_idc = 0
  w.put_bit(true);             // single_picture_header_flag
  w.put_bits(static_cast<std::uint64_t>(opt.seq_level_idx), 5);
  put_uvlc(w, 0);              // chroma_format_idc = 4:2:0
  put_uvlc(w, 1);              // bit_depth_idc = 1 -> 8-bit
  unsigned wb = 1;
  while ((1U << wb) < static_cast<unsigned>(width)) ++wb;
  unsigned hb = 1;
  while ((1U << hb) < static_cast<unsigned>(height)) ++hb;
  w.put_bits(wb - 1, 4);
  w.put_bits(hb - 1, 4);
  w.put_bits(static_cast<std::uint64_t>(width) - 1, wb);
  w.put_bits(static_cast<std::uint64_t>(height) - 1, hb);
  w.put_bit(false);  // seq_cropping_window_present_flag

  w.put_bit(false);  // use_256x256_superblock
  w.put_bit(false);  // use_128x128_superblock
  w.put_bit(false);  // enable_sdp
  w.put_bit(false);  // enable_ext_partitions
  w.put_bit(false);  // reduce_pb_aspect_ratio
  w.put_bit(false);  // enable_ext_seg
  w.put_bit(false);  // seq_seg_info_present_flag
  w.put_bit(false);  // enable_dip
  w.put_bit(opt.enable_intra_edge_filter);
  w.put_bit(false);  // enable_mrls
  w.put_bit(false);  // enable_cfl_intra
  w.put_bits(0, 2);  // cfl_ds_filter_index
  w.put_bit(false);  // enable_mhccp
  w.put_bit(false);  // enable_ibp
  w.put_bit(false);  // enable_refmvbank
  w.put_bit(true);   // disable_drl_reorder
  put_ns(w, 3, 0);   // seq_max_bvp_drl_bits_minus_1
  w.put_bit(false);  // allow_frame_max_bvp_drl_bits
  w.put_bit(false);  // enable_bawp

  const bool fsc = opt.fsc != FscMode::kOff;
  w.put_bit(fsc);    // enable_fsc
  if (!fsc) w.put_bit(false);  // enable_idtx_intra
  w.put_bit(false);  // enable_intra_ist
  w.put_bit(false);  // enable_inter_ist
  w.put_bit(false);  // enable_chroma_dctonly
  w.put_bit(false);  // reduced_tx_part_set
  w.put_bit(false);  // enable_cctx
  w.put_bit(false);  // enable_tcq
  w.put_bit(false);  // enable_parity_hiding
  w.put_bit(false);  // separate_uv_delta_q
  w.put_bit(true);   // equal_ac_dc_q
  w.put_bits(23, 5); // base_uv_ac_delta_q -> 0
  w.put_bit(false);  // uv_ac_delta_q_enabled

  w.put_bit(false);  // disable_loopfilters_across_tiles
  w.put_bit(false);  // enable_cdef
  w.put_bit(false);  // enable_gdf
  w.put_bit(false);  // enable_restoration
  w.put_bit(false);  // enable_ccso
  w.put_bits(0, 2);  // df_par_bits_minus_2
  w.put_bit(false);  // seq_tile_info_present_flag
  w.put_bit(false);  // film_grain_params_present
  return w.bit_size() - start;
}

void write_frame_header(av2enc::BitWriter& w, const Options& opt,
                        int mi_cols, int mi_rows) {
  put_uvlc(w, 0);  // cur_mfh_id
  put_uvlc(w, 0);  // seq_header_id_in_frame_header
  w.put_bit(false);  // allow_screen_content_tools
  w.put_bit(false);  // allow_intrabc
  w.put_bit(!opt.adaptive_cdf);  // disable_cdf_update

  const int sb_cols = (mi_cols + 15) >> 4;
  const int sb_rows = (mi_rows + 15) >> 4;
  w.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1) w.put_bit(false);  // increment_tile_cols_log2 = 0
  if (sb_rows > 1) w.put_bit(false);  // increment_tile_rows_log2 = 0

  w.put_bits(0, 8);  // base_q_idx = 0 (lossless)
  w.put_bit(false);  // segmentation_enabled
  w.put_bit(false);  // using_qmatrix
  // delta_q_params(): base_q_idx == 0 -> no delta_q_present bit.
  // CodedLossless -> allow_tcq/allow_parity_hiding inferred 0.
  // deblocking/gdf/cdef/lr/ccso return before coding any bits.
  // read_tx_mode(): CodedLossless -> TxMode = ONLY_4X4, no bit.
  // frame_reference_mode()/skip_mode_params(): FrameIsIntra -> no bits.
  w.put_bits(0, 2);  // reduced_tx_set
  // global_motion_params(), film_grain_config(): intra / absent -> no bits.
}

struct ObuView {
  std::uint8_t type = 0;
  const std::uint8_t* payload = nullptr;
  std::size_t size = 0;
};

bool parse_obus(const std::uint8_t* data, std::size_t size,
                std::vector<ObuView>& out, std::string& error) {
  std::size_t pos = 0;
  while (pos < size) {
    std::uint32_t len = 0;
    unsigned shift = 0;
    std::uint8_t byte = 0;
    do {
      if (pos >= size || shift > 28) {
        error = "truncated/invalid leb128 OBU size";
        return false;
      }
      byte = data[pos++];
      len |= static_cast<std::uint32_t>(byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);
    if (len == 0 || pos + len > size) {
      error = "OBU size exceeds temporal unit";
      return false;
    }
    const std::uint8_t header = data[pos];
    const bool ext = (header & 0x80) != 0;
    const std::uint8_t type = (header >> 2) & 0x1f;
    const std::size_t header_bytes = ext ? 2 : 1;
    if (len < header_bytes) {
      error = "OBU shorter than its header";
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

// Minimal sequence-header parser for the fields this writer emits.
struct SeqInfo {
  int width = 0;
  int height = 0;
  int seq_level_idx = 0;
  bool enable_fsc = false;
  bool enable_intra_edge_filter = false;
};

bool parse_sequence_header(const ObuView& obu, SeqInfo& info,
                           std::string& error) {
  std::size_t pos = 0;
  const std::size_t bits = obu.size * 8;
  std::uint32_t v = 0;
  if (!get_uvlc(obu.payload, bits, &pos, &v)) {
    error = "sequence header: seq_header_id";
    return false;
  }
  auto read_bits = [&](unsigned n, std::uint32_t* out) {
    std::uint32_t x = 0;
    for (unsigned i = 0; i < n; ++i) {
      if (pos >= bits) return false;
      const std::uint32_t b =
          (obu.payload[pos >> 3] >> (7 - (pos & 7))) & 1;
      ++pos;
      x = (x << 1) | b;
    }
    *out = x;
    return true;
  };
  std::uint32_t profile = 0;
  if (!read_bits(5, &profile) || profile != 0) {
    error = "sequence header: profile";
    return false;
  }
  std::uint32_t single = 0;
  if (!read_bits(1, &single) || single != 1) {
    error = "sequence header: single_picture_header_flag";
    return false;
  }
  if (!read_bits(5, &v)) return false;
  info.seq_level_idx = static_cast<int>(v);
  if (!get_uvlc(obu.payload, bits, &pos, &v) || v != 0) {
    error = "sequence header: chroma_format_idc";
    return false;
  }
  if (!get_uvlc(obu.payload, bits, &pos, &v) || v != 1) {
    error = "sequence header: bit_depth_idc";
    return false;
  }
  std::uint32_t wb = 0;
  std::uint32_t hb = 0;
  if (!read_bits(4, &wb) || !read_bits(4, &hb)) return false;
  ++wb;
  ++hb;
  std::uint32_t wm1 = 0;
  std::uint32_t hm1 = 0;
  if (!read_bits(wb, &wm1) || !read_bits(hb, &hm1)) return false;
  info.width = static_cast<int>(wm1) + 1;
  info.height = static_cast<int>(hm1) + 1;

  // The remaining bits are fixed for this writer.  Read them only to reach
  // enable_fsc / enable_intra_edge_filter.
  auto skip = [&](unsigned n) { pos += n; };
  skip(1);  // cropping
  skip(5);  // 256/128/sdp/ext/reduce
  skip(2);  // ext_seg/seq_seg_info
  skip(1);  // dip
  std::uint32_t edge = 0;
  if (!read_bits(1, &edge)) return false;
  info.enable_intra_edge_filter = edge != 0;
  skip(1 + 1 + 2 + 1 + 1);  // mrls,cfl,cfl_ds,mhccp,ibp
  skip(1 + 1);              // refmvbank,disable_drl_reorder
  // ns(3) for seq_max_bvp_drl_bits_minus_1 (value 0 -> one bit '0').
  skip(1);
  skip(1 + 1);  // allow_frame_max_bvp_drl_bits, enable_bawp
  std::uint32_t fsc = 0;
  if (!read_bits(1, &fsc)) return false;
  info.enable_fsc = fsc != 0;
  // We do not need the rest for decoding the tile.
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool encode_still(const StillPicture& picture, const Options& options,
                  EncodeResult& result, std::string& error) {
  result = {};
  if (options.sb_size != 64) {
    error = "lossless reference supports sb_size = 64 only";
    return false;
  }
  Geometry g;
  if (!make_geometry(static_cast<int>(picture.width),
                     static_cast<int>(picture.height), g, error)) {
    return false;
  }
  const std::size_t y_size = static_cast<std::size_t>(g.width) * g.height;
  const std::size_t c_size =
      static_cast<std::size_t>(g.chroma_w) * g.chroma_h;
  if (picture.y.size() < y_size || picture.u.size() < c_size ||
      picture.v.size() < c_size) {
    error = "source plane spans are smaller than the picture geometry";
    return false;
  }

  Plane src[3];
  Plane recon[3];
  src[0].w = g.width;
  src[0].h = g.height;
  src[0].data.assign(picture.y.begin(), picture.y.begin() + y_size);
  src[1].w = g.chroma_w;
  src[1].h = g.chroma_h;
  src[1].data.assign(picture.u.begin(), picture.u.begin() + c_size);
  src[2].w = g.chroma_w;
  src[2].h = g.chroma_h;
  src[2].data.assign(picture.v.begin(), picture.v.begin() + c_size);
  for (int p = 0; p < 3; ++p) recon[p].resize(src[p].w, src[p].h);

  CdfState cdfs;
  init_cdfs(cdfs);
  TileContext ctx;
  ctx.reset(g);

  Stats stats;
  std::vector<std::uint8_t> tile(16U * 1024U * 1024U);
  std::size_t tile_bytes = 0;
  Plane* src_ptr[3] = {&src[0], &src[1], &src[2]};
  Plane* recon_ptr[3] = {&recon[0], &recon[1], &recon[2]};
  TileWriter writer(g, options, cdfs, ctx, src_ptr, recon_ptr, &stats);
  if (!writer.encode_tile(tile.data(), tile.size(), &tile_bytes, error)) {
    return false;
  }
  tile.resize(tile_bytes);
  stats.tile_bytes = static_cast<std::uint32_t>(tile_bytes);

  // Assemble TD + SEQ + CLK.
  av2enc::BitWriter td;
  const std::vector<std::uint8_t> td_obu =
      make_obu(2, std::span<const std::uint8_t>());
  (void)td;

  av2enc::BitWriter seq;
  write_sequence_header(seq, options, g.width, g.height);
  seq.put_bit(false);  // obu_extension_flag
  put_trailing_bits(seq);
  const std::vector<std::uint8_t> seq_bytes = seq.take_bytes();
  const std::vector<std::uint8_t> seq_obu = make_obu(1, seq_bytes);

  av2enc::BitWriter clk;
  clk.put_bit(true);  // is_first_tile_group
  write_frame_header(clk, options, g.mi_cols, g.mi_rows);
  clk.align_to_byte();
  std::vector<std::uint8_t> clk_payload = clk.take_bytes();
  clk_payload.insert(clk_payload.end(), tile.begin(), tile.end());
  const std::vector<std::uint8_t> clk_obu = make_obu(4, clk_payload);

  result.temporal_unit.clear();
  result.temporal_unit.insert(result.temporal_unit.end(), td_obu.begin(),
                              td_obu.end());
  result.temporal_unit.insert(result.temporal_unit.end(), seq_obu.begin(),
                              seq_obu.end());
  result.temporal_unit.insert(result.temporal_unit.end(), clk_obu.begin(),
                              clk_obu.end());

  result.recon_y = recon[0].data;
  result.recon_u = recon[1].data;
  result.recon_v = recon[2].data;

  std::uint64_t max_err = 0;
  for (int p = 0; p < 3; ++p) {
    for (std::size_t i = 0; i < recon[p].data.size(); ++i) {
      const int a = recon[p].data[i];
      const int b = src[p].data[i];
      max_err = std::max<std::uint64_t>(
          max_err, static_cast<std::uint64_t>(abs_int(a - b)));
    }
  }
  stats.max_abs_recon_error = max_err;
  stats.exact = max_err == 0;
  stats.frame_bytes =
      static_cast<std::uint32_t>(result.temporal_unit.size());
  result.stats = stats;
  if (options.verify_exact && !stats.exact) {
    error = "lossless reconstruction is not bit exact";
    return false;
  }
  return true;
}

bool decode_temporal_unit(std::span<const std::uint8_t> temporal_unit,
                          DecodeResult& result, std::string& error) {
  result = {};
  std::vector<ObuView> obus;
  if (!parse_obus(temporal_unit.data(), temporal_unit.size(), obus, error)) {
    return false;
  }
  SeqInfo seq;
  bool have_seq = false;
  const ObuView* clk = nullptr;
  for (const auto& obu : obus) {
    if (obu.type == 1) {
      if (!parse_sequence_header(obu, seq, error)) return false;
      have_seq = true;
    } else if (obu.type == 4) {
      clk = &obu;
    }
  }
  if (!have_seq || clk == nullptr) {
    error = "temporal unit is missing a sequence header or CLK frame";
    return false;
  }

  Geometry g;
  if (!make_geometry(seq.width, seq.height, g, error)) return false;

  // Parse the CLK header.  Our writer always emits the same prefix, so a
  // bit reader over the payload is enough.
  const std::uint8_t* p = clk->payload;
  const std::size_t bits = clk->size * 8;
  std::size_t pos = 0;
  auto read_bit = [&](bool* b) {
    if (pos >= bits) return false;
    *b = ((p[pos >> 3] >> (7 - (pos & 7))) & 1) != 0;
    ++pos;
    return true;
  };
  auto read_bits = [&](unsigned n, std::uint32_t* out) {
    std::uint32_t x = 0;
    for (unsigned i = 0; i < n; ++i) {
      bool b = false;
      if (!read_bit(&b)) return false;
      x = (x << 1) | (b ? 1u : 0u);
    }
    *out = x;
    return true;
  };
  bool first_tile_group = false;
  if (!read_bit(&first_tile_group) || !first_tile_group) {
    error = "CLK is not the first tile group";
    return false;
  }
  std::uint32_t v = 0;
  if (!get_uvlc(p, bits, &pos, &v) || v != 0) {
    error = "frame header: cur_mfh_id";
    return false;
  }
  if (!get_uvlc(p, bits, &pos, &v) || v != 0) {
    error = "frame header: seq_header_id";
    return false;
  }
  std::uint32_t allow_sct = 0;
  std::uint32_t allow_ibc = 0;
  if (!read_bits(1, &allow_sct) || allow_sct != 0 ||
      !read_bits(1, &allow_ibc) || allow_ibc != 0) {
    error = "frame header: unsupported screen-content/intrabc flags";
    return false;
  }
  std::uint32_t disable_cdf = 0;
  if (!read_bits(1, &disable_cdf)) return false;
  std::uint32_t uniform = 0;
  if (!read_bits(1, &uniform) || uniform != 1) {
    error = "frame header: expected uniform tile spacing";
    return false;
  }
  const int sb_cols = (g.mi_cols + 15) >> 4;
  const int sb_rows = (g.mi_rows + 15) >> 4;
  if (sb_cols > 1) {
    if (!read_bits(1, &v) || v != 0) {
      error = "frame header: tile cols increment";
      return false;
    }
  }
  if (sb_rows > 1) {
    if (!read_bits(1, &v) || v != 0) {
      error = "frame header: tile rows increment";
      return false;
    }
  }
  std::uint32_t base_q = 0;
  if (!read_bits(8, &base_q) || base_q != 0) {
    error = "frame header: expected base_q_idx = 0";
    return false;
  }
  std::uint32_t seg = 0;
  std::uint32_t qm = 0;
  if (!read_bits(1, &seg) || seg != 0 || !read_bits(1, &qm) || qm != 0) {
    error = "frame header: segmentation/QM not supported";
    return false;
  }
  // base_q_idx == 0 -> no delta_q_present.  CodedLossless infers the rest.
  std::uint32_t reduced = 0;
  if (!read_bits(2, &reduced)) return false;
  // byte_alignment() zeros.
  while ((pos & 7) != 0) {
    bool b = false;
    if (!read_bit(&b) || b) {
      error = "frame header: non-zero byte alignment bits";
      return false;
    }
  }
  const std::size_t header_bytes = pos / 8;
  if (header_bytes > clk->size) {
    error = "frame header larger than CLK payload";
    return false;
  }
  const std::uint8_t* tile = clk->payload + header_bytes;
  const std::size_t tile_size = clk->size - header_bytes;

  Options opt;
  opt.adaptive_cdf = disable_cdf == 0;
  opt.enable_intra_edge_filter = seq.enable_intra_edge_filter;
  opt.fsc = seq.enable_fsc ? FscMode::kAuto : FscMode::kOff;

  CdfState cdfs;
  init_cdfs(cdfs);
  TileContext ctx;
  ctx.reset(g);
  Plane recon[3];
  recon[0].resize(g.width, g.height);
  recon[1].resize(g.chroma_w, g.chroma_h);
  recon[2].resize(g.chroma_w, g.chroma_h);
  Stats stats;
  Plane* recon_ptr[3] = {&recon[0], &recon[1], &recon[2]};
  TileReader reader(g, opt, cdfs, ctx, recon_ptr, &stats);
  if (!reader.decode_tile(tile, tile_size, tile_size, error)) return false;

  result.width = static_cast<std::uint32_t>(g.width);
  result.height = static_cast<std::uint32_t>(g.height);
  result.y = recon[0].data;
  result.u = recon[1].data;
  result.v = recon[2].data;
  stats.tile_bytes = static_cast<std::uint32_t>(tile_size);
  stats.frame_bytes = static_cast<std::uint32_t>(clk->size);
  result.stats = stats;
  return true;
}

}  // namespace av2enc::lossless
