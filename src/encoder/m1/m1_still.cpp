// Integrated M1 still-picture encoder.
//
// This file is the integration point for the wave-1 components:
//   partition (src/encoder/partition)
//   full intra prediction (src/encoder/intra_modes)
//   transform variants (src/encoder/tx_variants)
//   CfL (src/encoder/cfl)
//   deblocking (src/encoder/filters/deblock)
//
// It keeps the M0 path in src/encoder/m0_still.cpp untouched and uses the
// normative §8.2 entropy coder from src/entropy/av2 unchanged.
#include "encoder/m1/m1_still.h"

#include "encoder/cfl/cfl.h"
#include "encoder/filters/deblock/deblock.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m0_cdf_tables.h"
#include "encoder/m0_tables.h"
#include "encoder/m1/m1_cdf_tables.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"
#include "entropy/av2/av2_entropy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::m1 {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;

// YMode numeric values (§6.19.5.5).
constexpr int kDcPred = 0;
constexpr int kV = 1;
constexpr int kH = 2;
constexpr int kD45 = 3;
constexpr int kD135 = 4;
constexpr int kD113 = 5;
constexpr int kD157 = 6;
constexpr int kD203 = 7;
constexpr int kD67 = 8;
constexpr int kSmooth = 9;
constexpr int kSmoothV = 10;
constexpr int kSmoothH = 11;
constexpr int kPaeth = 12;

// Reordered_Y_Mode[INTRA_MODES] (§5.20.5.5).
constexpr int kReorderedY[13] = {kDcPred, kSmooth, kSmoothV, kSmoothH, kPaeth,
                                 kD45,    kD67,    kV,      kD113,  kD135,
                                 kD157,   kH,      kD203};

// Default_Mode_List_Y[DIRECTIONAL_MODES_COUNT] (§5.20.5.5).
constexpr int kDefaultModeListY[56] = {
    17, 45, 3,  10, 24, 31, 38, 52, 15, 19, 43, 47, 1,  5,  8,  12,
    22, 26, 29, 33, 36, 40, 50, 54, 16, 18, 44, 46, 2,  4,  9,  11,
    23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0,  6,  7,  13,
    21, 27, 28, 34, 35, 41, 49, 55};

// Default_Mode_List_Uv[UV_INTRA_MODES_CFL_NOT_ALLOWED] (§5.20.5.6).
constexpr int kDefaultModeListUv[13] = {kDcPred, kSmooth, kSmoothV, kSmoothH,
                                        kPaeth,  kV,      kH,       kD45,
                                        kD135,   kD67,    kD113,    kD157,
                                        kD203};

constexpr int kRectHorz = 0;
constexpr int kRectVert = 1;
constexpr std::size_t kTileCapacityBytes = 16U * 1024U * 1024U;

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
int clip1(int value) { return clip3(0, 255, value); }
int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

int av2_y_mode(intra::Mode mode) {
  switch (mode) {
    case intra::Mode::kDc: return kDcPred;
    case intra::Mode::kV: return kV;
    case intra::Mode::kH: return kH;
    case intra::Mode::kD45: return kD45;
    case intra::Mode::kD135: return kD135;
    case intra::Mode::kD113: return kD113;
    case intra::Mode::kD157: return kD157;
    case intra::Mode::kD203: return kD203;
    case intra::Mode::kD67: return kD67;
    case intra::Mode::kSmooth: return kSmooth;
    case intra::Mode::kSmoothV: return kSmoothV;
    case intra::Mode::kSmoothH: return kSmoothH;
    case intra::Mode::kPaeth: return kPaeth;
  }
  return kDcPred;
}

bool y_mode_is_directional(int mode) { return mode >= kV && mode <= kD67; }
bool y_mode_is_smooth(int mode) {
  return mode == kSmooth || mode == kSmoothV || mode == kSmoothH;
}

// Reordered index of a YMode (0..12).
int reordered_index(int mode) {
  for (int i = 0; i < 13; ++i) {
    if (kReorderedY[i] == mode) return i;
  }
  return 0;
}

// IntraJointMode and y_modeIndex for a (YMode, angleDelta) pair.
struct YModeCode {
  int joint = 0;
  int mode_index = 0;
};

YModeCode y_mode_code(int mode, int angle_delta) {
  YModeCode out;
  const int r = reordered_index(mode);
  if (r < 5) {
    out.joint = r;
    out.mode_index = r;
    return out;
  }
  const int base = r - 5;  // 0..7
  const int dir_index = base * 7 + (angle_delta + 3);  // 0..55
  out.joint = 5 + dir_index;
  int pos = 0;
  for (int i = 0; i < 56; ++i) {
    if (kDefaultModeListY[i] == dir_index) {
      pos = i;
      break;
    }
  }
  out.mode_index = 5 + pos;
  return out;
}

int uv_mode_index(int uv_mode, int y_mode) {
  int index = 0;
  if (y_mode_is_directional(y_mode)) {
    if (uv_mode == y_mode) return 0;
    index = 1;
  }
  for (int i = 0; i < 13; ++i) {
    const int mode = kDefaultModeListUv[i];
    if (mode == y_mode && y_mode_is_directional(y_mode)) continue;
    if (mode == uv_mode) return index;
    ++index;
  }
  return 0;
}

int tx_sz_ctx(int tx_size) {
  return (kTxSizeSqr[tx_size] + kTxSizeSqrUp[tx_size] + 1) >> 1;
}

// Returns true when transform_type() codes an intra_tx_type symbol for this
// luma TX block. Intra with sqrUp < 32x32 uses TX_SET_INTRA_1; larger sizes are
// TX_SET_DCTONLY (no symbol).
bool luma_tx_type_signalled(int tx_size, int eob) {
  return kTxSizeSqrUp[tx_size] < 3 && eob != 1 && eob != 0;
}

struct CdfState {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t y_mode_offset[3][7];
  std::uint16_t uv_mode[2][9];
  std::uint16_t is_cfl[3][3];
  std::uint16_t cfl_index[3];
  std::uint16_t cfl_sign[9];
  std::uint16_t cfl_alpha[6][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t eob_pt_256[3][9];
  std::uint16_t eob_pt_128[3][9];
  std::uint16_t eob_pt_64[3][8];
  std::uint16_t eob_extra[3];
  std::uint16_t coeff_base[5][20][2][5];
  std::uint16_t coeff_base_lf[5][33][2][7];
  std::uint16_t coeff_base_eob[5][4][4];
  std::uint16_t coeff_base_lf_eob[5][4][6];
  std::uint16_t coeff_br[7][5];
  std::uint16_t coeff_br_lf[14][5];
  std::uint16_t coeff_base_uv[12][5];
  std::uint16_t coeff_base_lf_uv[12][7];
  std::uint16_t coeff_base_eob_uv[4][4];
  std::uint16_t coeff_base_lf_eob_uv[4][6];
  std::uint16_t coeff_br_uv[4][5];
  std::uint16_t dc_sign[2][2][3][3];
  std::uint16_t intra_tx_type_set1[3][8];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(CdfState& c) {
  copy_array(c.do_split, av2enc::m0::kDoSplitCdf);
  copy_array(c.rect_type, av2enc::m0::kRectTypeCdf);
  copy_array(c.y_mode_set, kYModeSetCdf);
  copy_array(c.y_mode_index, kYModeIndexCdf);
  copy_array(c.y_mode_offset, kYModeOffsetCdf);
  copy_array(c.uv_mode, kUvModeCflNotAllowedCdf);
  copy_array(c.is_cfl, kIsCflCdf);
  copy_array(c.cfl_index, kCflIndexCdf);
  copy_array(c.cfl_sign, kCflSignCdf);
  copy_array(c.cfl_alpha, kCflAlphaCdf);
  copy_array(c.txb_skip, av2enc::m0::kTxbSkipCdf);
  copy_array(c.v_txb_skip, av2enc::m0::kVTxbSkipCdf);
  copy_array(c.eob_pt_1024, av2enc::m0::kEobPt1024Cdf);
  copy_array(c.eob_pt_512, av2enc::m0::kEobPt512Cdf);
  copy_array(c.eob_pt_256, av2enc::m0::kEobPt256Cdf);
  copy_array(c.eob_pt_128, av2enc::m0::kEobPt128Cdf);
  copy_array(c.eob_pt_64, av2enc::m0::kEobPt64Cdf);
  copy_array(c.eob_extra, av2enc::m0::kEobExtraCdf);
  copy_array(c.coeff_base, av2enc::m0::kCoeffBaseCdf);
  copy_array(c.coeff_base_lf, av2enc::m0::kCoeffBaseLfCdf);
  copy_array(c.coeff_base_eob, av2enc::m0::kCoeffBaseEobCdf);
  copy_array(c.coeff_base_lf_eob, av2enc::m0::kCoeffBaseLfEobCdf);
  copy_array(c.coeff_br, av2enc::m0::kCoeffBrCdf);
  copy_array(c.coeff_br_lf, av2enc::m0::kCoeffBrLfCdf);
  copy_array(c.coeff_base_uv, av2enc::m0::kCoeffBaseUvCdf);
  copy_array(c.coeff_base_lf_uv, av2enc::m0::kCoeffBaseLfUvCdf);
  copy_array(c.coeff_base_eob_uv, av2enc::m0::kCoeffBaseEobUvCdf);
  copy_array(c.coeff_base_lf_eob_uv, av2enc::m0::kCoeffBaseLfEobUvCdf);
  copy_array(c.coeff_br_uv, av2enc::m0::kCoeffBrUvCdf);
  copy_array(c.dc_sign, av2enc::m0::kDcSignCdf);
  copy_array(c.intra_tx_type_set1, av2enc::m0::kIntraTxTypeSet1Cdf);
}

struct Plane {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                   value);
  }
  std::uint8_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
};

struct TxPlan {
  int plane = 0;
  int tx_size = 0;
  int coded_w = 0;
  int coded_h = 0;
  int adj_w = 0;
  int adj_h = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;
  std::vector<std::uint8_t> pred;
  std::vector<std::uint8_t> recon;
};

struct ModeChoice {
  intra::Mode mode = intra::Mode::kDc;
  int angle_delta = 0;
  int av2_mode = kDcPred;
  std::int64_t sse = 0;
};

struct UvChoice {
  bool is_cfl = false;
  intra::Mode mode = intra::Mode::kDc;
  int angle_delta = 0;
  int av2_mode = kDcPred;
  std::int64_t sse = 0;
};

// Per-4x4 decoded flags for count_top_right_avail / count_bottom_left_avail.
struct DecodedMap {
  int cols = 0;
  int rows = 0;
  std::vector<std::uint8_t> cells;

  void reset(int c, int r) {
    cols = c;
    rows = r;
    cells.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(r), 0);
  }
  bool at(int x, int y) const {
    if (x < 0 || y < 0 || x >= cols || y >= rows) return false;
    return cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(x)] != 0;
  }
  void fill(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < cols && y + j < rows) {
          cells[static_cast<std::size_t>(y + j) *
                    static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(x + i)] = 1;
        }
      }
    }
  }
};

class Encoder {
 public:
  Encoder(const StillPicture& picture, const M1Options& options)
      : picture_(picture), options_(options) {
    width_ = static_cast<int>(picture.width);
    height_ = static_cast<int>(picture.height);
    chroma_width_ = (width_ + 1) / 2;
    chroma_height_ = (height_ + 1) / 2;

    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = 64;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;

    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    luma_stride_ = mi_cols_ * 4;
    luma_height_ = mi_rows_ * 4;
    chroma_stride_ = luma_stride_ / 2;
    chroma_padded_height_ = luma_height_ / 2;

    init_cdfs(cdfs_);
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update =
        (options_.adaptive_cdf ? 0 : 1);

    recon_[0].resize(luma_stride_, luma_height_, 128);
    recon_[1].resize(chroma_stride_, chroma_padded_height_, 128);
    recon_[2].resize(chroma_stride_, chroma_padded_height_, 128);

    for (int plane = 0; plane < 3; ++plane) {
      const int n = (plane == 0 ? mi_cols_ : mi_cols_ / 2) + 16;
      const int m = (plane == 0 ? mi_rows_ : mi_rows_ / 2) + 16;
      above_level_[plane].assign(static_cast<std::size_t>(n), 0);
      left_level_[plane].assign(static_cast<std::size_t>(m), 0);
      above_dc_[plane].assign(static_cast<std::size_t>(n), 0);
      left_dc_[plane].assign(static_cast<std::size_t>(m), 0);
    }
    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    intra_joint_.assign(
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_),
        0);
    uv_smooth_.assign(
        static_cast<std::size_t>(mi_rows_ / 2) *
            static_cast<std::size_t>(mi_cols_ / 2),
        0);
    uvcfl_.assign(uv_smooth_.size(), 0);

    for (int plane = 0; plane < 3; ++plane) {
      decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                            plane == 0 ? mi_rows_ : mi_rows_ / 2);
    }
    clear_above_context();

    if (options_.enable_deblock) {
      for (int plane = 0; plane < 3; ++plane) {
        deblock_mi_[plane].assign(
            static_cast<std::size_t>(mi_cols_) *
                static_cast<std::size_t>(mi_rows_),
            deblock::DeblockMi{});
      }
    }
  }

  bool encode(std::vector<std::uint8_t>* tile_payload, std::string& error) {
    FrameGeometry geometry = geometry_;
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = options_.min_block_size;
    poptions.max_depth = options_.max_partition_depth;
    poptions.allow_rect = true;
    poptions.allow_split = false;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }

    const double lambda = 0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;

    const auto cost_fn = [&](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = static_cast<double>(
          block_variance_cost_at(info.mi_row, info.mi_col, info.size));
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };

    const partition::FrameDecision frame =
        partition::decide_frame(geometry, cost_fn, poptions);
    if (frame.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame.error;
      return false;
    }
    if (!partition::validate_frame(geometry, frame, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }

    clear_above_context();
    for (int sbi = 0; sbi < static_cast<int>(frame.superblocks.size()); ++sbi) {
      const partition::PartitionNode& sb = frame.superblocks[sbi];
      const int r0 = static_cast<int>(sbi / partition::sb_cols(geometry)) * 16;
      (void)r0;
      if (sb.mi_row == 0) {
        // Frame start: above context stays cleared.
      }
      if (sb.mi_col == 0) {
        clear_left_context();
      }
      // Reset SB-local decoded flags exactly like clear_block_decoded_flags().
      for (int plane = 0; plane < 3; ++plane) {
        decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                              plane == 0 ? mi_rows_ : mi_rows_ / 2);
      }
      sb_mi_row_ = sb.mi_row;
      sb_mi_col_ = sb.mi_col;
      sb_cols_plane_[0] = std::min(16, mi_cols_ - sb.mi_col);
      sb_rows_plane_[0] = std::min(16, mi_rows_ - sb.mi_row);
      sb_cols_plane_[1] = std::min(8, mi_cols_ / 2 - sb.mi_col / 2);
      sb_rows_plane_[1] = std::min(8, mi_rows_ / 2 - sb.mi_row / 2);
      sb_cols_plane_[2] = sb_cols_plane_[1];
      sb_rows_plane_[2] = sb_rows_plane_[1];

      if (!encode_node(sb)) {
        error = error_;
        return false;
      }
    }

    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      error = std::string("entropy exit_symbol failed: ") +
              av2::status_name(status);
      return false;
    }
    tile_payload->assign(tile_buffer_.begin(),
                         tile_buffer_.begin() +
                             static_cast<std::ptrdiff_t>(payload_bytes));
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  Plane& recon_mut(int plane) { return recon_[plane]; }
  int mi_cols() const { return mi_cols_; }
  int mi_rows() const { return mi_rows_; }
  const deblock::DeblockMi* deblock_mi(int plane) const {
    return deblock_mi_[plane].data();
  }

 private:
  // ---------------------------------------------------------------------
  // Entropy helpers
  // ---------------------------------------------------------------------
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_symbol failed: ") +
               av2::status_name(status);
      return false;
    }
    return true;
  }
  bool raw_bit(bool bit) {
    const av2::Status status = enc_.encode_bool(bit);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_bool failed: ") +
               av2::status_name(status);
      return false;
    }
    return true;
  }
  bool write_literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

  // ---------------------------------------------------------------------
  // Context grids
  // ---------------------------------------------------------------------
  void clear_above_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(above_level_[plane].begin(), above_level_[plane].end(), 0);
      std::fill(above_dc_[plane].begin(), above_dc_[plane].end(), 0);
    }
    std::fill(above_mi_.begin(), above_mi_.end(), 6);
  }
  void clear_left_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(left_level_[plane].begin(), left_level_[plane].end(), 0);
      std::fill(left_dc_[plane].begin(), left_dc_[plane].end(), 0);
    }
    std::fill(left_mi_.begin(), left_mi_.end(), 6);
  }

  int& above_mi(int x) { return above_mi_[static_cast<std::size_t>(x)]; }
  int& left_mi(int y) { return left_mi_[static_cast<std::size_t>(y)]; }
  std::uint8_t& above_level(int plane, int x) {
    return above_level_[plane][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_level(int plane, int y) {
    return left_level_[plane][static_cast<std::size_t>(y)];
  }
  std::uint8_t& above_dc(int plane, int x) {
    return above_dc_[plane][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_dc(int plane, int y) {
    return left_dc_[plane][static_cast<std::size_t>(y)];
  }
  int& intra_joint(int r, int c) {
    return intra_joint_[static_cast<std::size_t>(r) *
                            static_cast<std::size_t>(mi_cols_) +
                        static_cast<std::size_t>(c)];
  }
  std::uint8_t& uv_smooth(int r, int c) {
    return uv_smooth_[static_cast<std::size_t>(r) *
                          static_cast<std::size_t>(mi_cols_ / 2) +
                      static_cast<std::size_t>(c)];
  }
  std::uint8_t& uvcfl(int r, int c) {
    return uvcfl_[static_cast<std::size_t>(r) *
                      static_cast<std::size_t>(mi_cols_ / 2) +
                  static_cast<std::size_t>(c)];
  }
  bool inside_mi(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }
  int get_joint_mode(int r, int c, int w4, int h4, int dir) const {
    int mv_row = 0;
    int mv_col = 0;
    if (dir != 0) {
      mv_row = r - 1;
      mv_col = c + w4 - 1;
    } else {
      mv_col = c - 1;
      mv_row = r + h4 - 1;
    }
    if (inside_mi(mv_row, mv_col)) {
      return intra_joint_[static_cast<std::size_t>(mv_row) *
                              static_cast<std::size_t>(mi_cols_) +
                          static_cast<std::size_t>(mv_col)];
    }
    return 0;
  }

  // ---------------------------------------------------------------------
  // Partition syntax
  // ---------------------------------------------------------------------
  bool code_do_split(PartitionType partition, int r, int c, BlockSize size) {
    const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
    const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
    const int ctx = kPartitionSizeAdjust[static_cast<int>(size)] * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.do_split[0][ctx][0], 2,
               partition == PartitionType::kNone ? 0 : 1);
  }

  bool code_rect_type(PartitionType partition, int r, int c, BlockSize size) {
    const int bsw = std::max(kMiWidthLog2[static_cast<int>(size)], 1);
    const int bsh = std::max(kMiHeightLog2[static_cast<int>(size)], 1);
    const int ctx = kPartitionSizeAdjustRect[static_cast<int>(size)] * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.rect_type[0][ctx][0], 2,
               partition == PartitionType::kVert ? kRectVert : kRectHorz);
  }

  bool encode_node(const partition::PartitionNode& node) {
    if (trace_) {
      std::cerr << "m1 node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " part=" << partition::partition_name(node.partition)
                << " forced=" << (node.forced ? 1 : 0) << " allowed="
                << node.allowed.allowed[0] << node.allowed.allowed[1]
                << node.allowed.allowed[2] << node.allowed.allowed[3] << "\n";
    }
    if (!node.forced) {
      if (node.allowed.is_allowed(PartitionType::kNone)) {
        if (!code_do_split(node.partition, node.mi_row, node.mi_col,
                           node.size)) {
          return false;
        }
        if (node.partition == PartitionType::kNone) {
          return encode_block(node);
        }
      }
      if (node.partition == PartitionType::kNone) {
        // No do_split symbol was coded (NONE is the implied partition), but
        // the tree solver only reaches this for a forced node.
        return encode_block(node);
      }
      const partition::RectTypeImplied implied =
          partition::rect_type_implied_by_bsize(node.size);
      bool need_rect = !implied.implied;
      if (need_rect) {
        const bool allow_h = node.allowed.is_allowed(PartitionType::kHorz);
        const bool allow_v = node.allowed.is_allowed(PartitionType::kVert);
        if (!allow_h || !allow_v) {
          need_rect = false;
        }
      }
      if (need_rect) {
        if (!code_rect_type(node.partition, node.mi_row, node.mi_col,
                            node.size)) {
          return false;
        }
      }
    }
    if (node.partition == PartitionType::kNone) {
      return encode_block(node);
    }
    for (const partition::PartitionNode& child : node.children) {
      if (!encode_node(child)) {
        return false;
      }
    }
    return true;
  }

  // ---------------------------------------------------------------------
  // Source / reconstruction helpers
  // ---------------------------------------------------------------------
  std::uint8_t source_sample(int plane, int x, int y) const {
    const std::span<const std::uint8_t> data =
        plane == 0 ? picture_.y : (plane == 1 ? picture_.u : picture_.v);
    const int w = plane == 0 ? width_ : chroma_width_;
    const int h = plane == 0 ? height_ : chroma_height_;
    const int cx = clip3(0, w - 1, x);
    const int cy = clip3(0, h - 1, y);
    const int stride = plane == 0 ? width_ : chroma_width_;
    return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(cx)];
  }

  void fill_source(int plane, int x0, int y0, int w, int h,
                   std::vector<std::uint8_t>& out) const {
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        out[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
            static_cast<std::size_t>(x)] =
            source_sample(plane, x0 + x, y0 + y);
      }
    }
  }

  int count_top_right(int plane, int abs_x4, int abs_y4, int w4) const {
    const int sub = plane == 0 ? 0 : 1;
    const int sb4p = 16 >> sub;
    const int sb_c0 = sb_mi_col_ >> sub;
    const int sb_r0 = sb_mi_row_ >> sub;
    const int rel_x = abs_x4 - sb_c0;
    const int rel_y = abs_y4 - sb_r0;
    const int sb_w = sb_cols_plane_[plane];
    int count = 0;
    for (int i = 0; i < w4; ++i) {
      const int x = rel_x + w4 + i;
      if (x >= sb_w) break;
      if (rel_y - 1 >= 0 &&
          !decoded_[plane].at(sb_c0 + x, sb_r0 + rel_y - 1)) {
        break;
      }
      ++count;
    }
    (void)sb4p;
    return count;
  }

  int count_bottom_left(int plane, int abs_x4, int abs_y4, int h4) const {
    const int sub = plane == 0 ? 0 : 1;
    const int sb_c0 = sb_mi_col_ >> sub;
    const int sb_r0 = sb_mi_row_ >> sub;
    const int rel_x = abs_x4 - sb_c0;
    const int rel_y = abs_y4 - sb_r0;
    const int sb_h = sb_rows_plane_[plane];
    int count = 0;
    for (int i = 0; i < h4; ++i) {
      const int y = rel_y + h4 + i;
      if (y >= sb_h) break;
      if (rel_x - 1 >= 0 &&
          !decoded_[plane].at(sb_c0 + rel_x - 1, sb_r0 + y)) {
        break;
      }
      ++count;
    }
    return count;
  }

  // ---------------------------------------------------------------------
  // Prediction
  // ---------------------------------------------------------------------
  bool prepare_edges(int plane, int x0, int y0, int w, int h,
                     intra::IntraEdges& edges) const {
    intra::NeighborRequest request;
    request.recon = intra::PlaneView{recon_[plane].samples.data(),
                                     recon_[plane].width};
    request.x = x0;
    request.y = y0;
    request.w = w;
    request.h = h;
    request.max_x = recon_[plane].width - 1;
    request.max_y = recon_[plane].height - 1;
    request.have_left = x0 > 0;
    request.have_above = y0 > 0;
    const int sub = plane == 0 ? 0 : 1;
    request.num4_above_right =
        count_top_right(plane, x0 >> 2, y0 >> 2, w >> 2);
    request.num4_below_left =
        count_bottom_left(plane, x0 >> 2, y0 >> 2, h >> 2);
    (void)sub;
    request.color_plane = plane;
    return intra::prepare_edges(request, 8, edges) == intra::IntraStatus::kOk;
  }

  bool neighborhood_smooth(int plane, int r, int c, int w4, int h4,
                           bool& above_smooth, bool& left_smooth) const {
    above_smooth = false;
    left_smooth = false;
    if (plane == 0) {
      if (r > 0 && c < mi_cols_) {
        above_smooth = y_mode_is_smooth(
            intra_joint_[static_cast<std::size_t>(r - 1) *
                             static_cast<std::size_t>(mi_cols_) +
                         static_cast<std::size_t>(c)]);
      }
      if (c > 0 && r < mi_rows_) {
        left_smooth = y_mode_is_smooth(
            intra_joint_[static_cast<std::size_t>(r) *
                             static_cast<std::size_t>(mi_cols_) +
                         static_cast<std::size_t>(c - 1)]);
      }
    } else {
      const int cr = r / 2;
      const int cc = c / 2;
      const int cmi_cols = mi_cols_ / 2;
      if (cr > 0 && cc < cmi_cols) {
        above_smooth =
            uv_smooth_[static_cast<std::size_t>(cr - 1) *
                           static_cast<std::size_t>(cmi_cols) +
                       static_cast<std::size_t>(cc)] != 0;
      }
      if (cc > 0 && cr < mi_rows_ / 2) {
        left_smooth =
            uv_smooth_[static_cast<std::size_t>(cr) *
                           static_cast<std::size_t>(cmi_cols) +
                       static_cast<std::size_t>(cc - 1)] != 0;
      }
    }
    (void)w4;
    (void)h4;
    return true;
  }

  bool predict_block(int plane, int x0, int y0, int w, int h,
                     intra::Mode mode, int angle_delta,
                     std::vector<std::uint8_t>& out) {
    intra::IntraEdges edges;
    if (!prepare_edges(plane, x0, y0, w, h, edges)) {
      error_ = "prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    neighborhood_smooth(plane, y0 >> 2, x0 >> 2, w >> 2, h >> 2,
                        above_smooth, left_smooth);
    intra::PredictParams params;
    params.mode = mode;
    params.angle_delta = angle_delta;
    params.enable_edge_filter = options_.enable_intra_edge_filter;
    params.above_neighbor_is_smooth = above_smooth;
    params.left_neighbor_is_smooth = left_smooth;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (intra::predict_intra(edges, params, out.data(),
                             static_cast<std::size_t>(w)) !=
        intra::IntraStatus::kOk) {
      error_ = "predict_intra failed";
      return false;
    }
    return true;
  }

  std::int64_t sse_against_source(int plane, int x0, int y0, int w, int h,
                                  const std::vector<std::uint8_t>& pred) const {
    std::int64_t sse = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)];
        const int d = src - p;
        sse += static_cast<std::int64_t>(d) * d;
      }
    }
    return sse;
  }

  bool choose_luma_mode(int x0, int y0, int w, int h, ModeChoice& best) {
    if (options_.force_dc) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    intra::IntraEdges edges;
    if (!prepare_edges(0, x0, y0, w, h, edges)) {
      error_ = "luma prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    neighborhood_smooth(0, y0 >> 2, x0 >> 2, w >> 2, h >> 2, above_smooth,
                        left_smooth);
    std::array<intra::ModeCandidate, 61> candidates;
    const int count = intra::build_mode_candidates(candidates, true);
    if (count <= 0) {
      error_ = "build_mode_candidates failed";
      return false;
    }
    std::vector<std::uint8_t> pred;
    pred.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    best.sse = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < count; ++i) {
      intra::PredictParams params;
      params.mode = candidates[static_cast<std::size_t>(i)].mode;
      params.angle_delta = candidates[static_cast<std::size_t>(i)].angle_delta;
      params.enable_edge_filter = options_.enable_intra_edge_filter;
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      if (intra::predict_intra(edges, params, pred.data(),
                               static_cast<std::size_t>(w)) !=
          intra::IntraStatus::kOk) {
        error_ = "luma predict_intra failed";
        return false;
      }
      const std::int64_t sse = sse_against_source(0, x0, y0, w, h, pred);
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = params.mode;
        best.angle_delta = params.angle_delta;
        best.av2_mode = av2_y_mode(params.mode);
      }
    }
    return true;
  }

  std::int64_t block_variance_cost_at(int mi_row, int mi_col,
                                      BlockSize size) const {
    const partition::BlockRect rect =
        partition::block_rect(geometry_, mi_row, mi_col, size);
    const int w = rect.visible_width;
    const int h = rect.visible_height;
    if (w <= 0 || h <= 0) return 0.0;
    std::int64_t sum = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        sum += source_sample(0, rect.x + x, rect.y + y);
      }
    }
    const double mean =
        static_cast<double>(sum) / static_cast<double>(w * h);
    double sse = 0.0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const double d = static_cast<double>(source_sample(
                            0, rect.x + x, rect.y + y)) -
                         mean;
        sse += d * d;
      }
    }
    return static_cast<std::int64_t>(sse);
  }

  // ---------------------------------------------------------------------
  // Transform / reconstruction
  // ---------------------------------------------------------------------
  bool compute_tx(int plane, int x0, int y0, int cw, int ch,
                  const std::vector<std::uint8_t>& pred, TxPlan& plan) {
    const auto params_opt =
        txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error_ = "unsupported transform size";
      return false;
    }
    const txv::TxParams& params = *params_opt;
    plan.plane = plane;
    plan.tx_size = params.tx_size;
    plan.coded_w = cw;
    plan.coded_h = ch;
    plan.adj_w = params.w;
    plan.adj_h = params.h;
    plan.pred = pred;
    const int coded_pels = cw * ch;
    std::vector<std::int16_t> residual_coded(
        static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        residual_coded[static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(cw) +
                       static_cast<std::size_t>(x)] =
            static_cast<std::int16_t>(src - p);
      }
    }
    std::vector<std::int16_t> residual_adj(
        static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
    txv::downsample_residual(residual_coded.data(), params,
                             residual_adj.data());
    std::vector<double> coeff(
        static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
    plan.levels.assign(coeff.size(), 0);
    txv::forward_transform(residual_adj.data(), params, coeff.data(),
                           plan.levels.data());

    const int area = params.w * params.h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    build_scan(params.w, params.h, scan);
    plan.eob = 0;
    for (int c = area - 1; c >= 0; --c) {
      if (plan.levels[scan[static_cast<std::size_t>(c)]] != 0) {
        plan.eob = c + 1;
        break;
      }
    }

    std::vector<std::int32_t> dequant(
        static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
    std::vector<std::int16_t> recon_residual(
        static_cast<std::size_t>(coded_pels));
    txv::inverse_transform(plan.levels.data(), params, dequant.data(),
                           recon_residual.data());
    plan.recon.resize(static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        const int r =
            recon_residual[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(cw) +
                           static_cast<std::size_t>(x)];
        plan.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(clip1(p + r));
      }
    }
    return true;
  }

  void build_scan(int w, int h, std::vector<std::uint16_t>& scan) const {
    scan.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    int x = 0;
    int y = 0;
    for (int c = 0; c < w * h; ++c) {
      scan[static_cast<std::size_t>(c)] =
          static_cast<std::uint16_t>(y * w + x);
      ++x;
      --y;
      if (y < 0 || x >= w) {
        ++x;
        const int s = std::min(x, h - 1 - y);
        x -= s;
        y += s;
      }
    }
  }

  void write_recon(int plane, int x0, int y0, int cw, int ch,
                   const TxPlan& plan) {
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        recon_[plane].at(x0 + x, y0 + y) =
            plan.recon[static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(cw) +
                       static_cast<std::size_t>(x)];
      }
    }
  }

  // ---------------------------------------------------------------------
  // Mode info syntax
  // ---------------------------------------------------------------------
  int joint_context() const {
    // Uses the block-start MiRow/MiCol of the current node.
    return (get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, 0) >= 5 ? 1 : 0) +
           (get_joint_mode(cur_r_, cur_c_, cur_w4_, cur_h4_, 1) >= 5 ? 1 : 0);
  }

  bool code_y_mode(const ModeChoice& choice) {
    const YModeCode code = y_mode_code(choice.av2_mode, choice.angle_delta);
    const int ctx = joint_context();
    if (code.mode_index < 13) {
      if (!sym(&cdfs_.y_mode_set[0], 4, 0)) return false;
      if (code.mode_index <= 6) {
        return sym(&cdfs_.y_mode_index[ctx][0], 8, code.mode_index);
      }
      if (!sym(&cdfs_.y_mode_index[ctx][0], 8, 7)) return false;
      return sym(&cdfs_.y_mode_offset[ctx][0], 6, code.mode_index - 7);
    }
    const int set = 1 + (code.mode_index - 13) / 16;
    const int second = (code.mode_index - 13) % 16;
    if (!sym(&cdfs_.y_mode_set[0], 4, set)) return false;
    return write_literal(4, static_cast<std::uint32_t>(second));
  }

  bool code_uv_mode(bool is_cfl, int uv_mode, int y_mode) {
    if (is_cfl) {
      // is_cfl derives its context from neighbouring chroma CfL flags.
      const int cr = cur_r_ / 2;
      const int cc = cur_c_ / 2;
      const int cmi_cols = mi_cols_ / 2;
      int ctx = 0;
      if (cr > 0 && cc < cmi_cols &&
          uvcfl_[static_cast<std::size_t>(cr - 1) *
                     static_cast<std::size_t>(cmi_cols) +
                 static_cast<std::size_t>(cc)] != 0) {
        ++ctx;
      }
      if (cc > 0 && cr < mi_rows_ / 2 &&
          uvcfl_[static_cast<std::size_t>(cr) *
                     static_cast<std::size_t>(cmi_cols) +
                 static_cast<std::size_t>(cc - 1)] != 0) {
        ++ctx;
      }
      if (!sym(&cdfs_.is_cfl[ctx][0], 2, 1)) return false;
      // CFL_DERIVED_ALPHA = 1.
      return sym(&cdfs_.cfl_index[0], 2, 1);
    }
    const int ctx = y_mode_is_directional(y_mode) ? 1 : 0;
    // is_cfl = 0 is only coded when CfL is allowed.
    if (options_.enable_cfl) {
      const int cr = cur_r_ / 2;
      const int cc = cur_c_ / 2;
      const int cmi_cols = mi_cols_ / 2;
      int cctx = 0;
      if (cr > 0 && cc < cmi_cols &&
          uvcfl_[static_cast<std::size_t>(cr - 1) *
                     static_cast<std::size_t>(cmi_cols) +
                 static_cast<std::size_t>(cc)] != 0) {
        ++cctx;
      }
      if (cc > 0 && cr < mi_rows_ / 2 &&
          uvcfl_[static_cast<std::size_t>(cr) *
                     static_cast<std::size_t>(cmi_cols) +
                 static_cast<std::size_t>(cc - 1)] != 0) {
        ++cctx;
      }
      if (!sym(&cdfs_.is_cfl[cctx][0], 2, 0)) return false;
    }
    const int idx = uv_mode_index(uv_mode, y_mode);
    if (idx <= 6) {
      return sym(&cdfs_.uv_mode[ctx][0], 8, idx);
    }
    if (!sym(&cdfs_.uv_mode[ctx][0], 8, 7)) return false;
    return write_literal(3, static_cast<std::uint32_t>(idx - 7));
  }

  // ---------------------------------------------------------------------
  // Residual syntax
  // ---------------------------------------------------------------------
  bool write_eob(int eob, int plane, int coded_w, int coded_h) {
    int eob_pt = 1;
    if (eob == 2) {
      eob_pt = 2;
    } else if (eob >= 3) {
      eob_pt = floor_log2(eob - 1) + 2;
    }
    const int multisize = std::min(5, floor_log2(coded_w)) +
                          std::min(5, floor_log2(coded_h)) - 4;
    std::uint16_t* cdf = nullptr;
    int escape_bits = 0;
    int cdf_n = 8;
    const int cdf_ctx = plane > 0 ? 2 : 0;
    switch (multisize) {
      case 0:
        error_ = "EOB multisize 0 is not supported by M1 transform sizes";
        return false;
      case 1:
        cdf = &cdfs_.eob_pt_64[cdf_ctx][0];
        escape_bits = 0;
        cdf_n = 7;
        break;
      case 2:
        cdf = &cdfs_.eob_pt_64[cdf_ctx][0];
        escape_bits = 0;
        cdf_n = 7;
        break;
      case 3:
        cdf = &cdfs_.eob_pt_128[cdf_ctx][0];
        break;
      case 4:
        cdf = &cdfs_.eob_pt_256[cdf_ctx][0];
        escape_bits = 1;
        break;
      case 5:
        cdf = &cdfs_.eob_pt_512[cdf_ctx][0];
        escape_bits = 2;
        break;
      default:
        cdf = &cdfs_.eob_pt_1024[cdf_ctx][0];
        escape_bits = 2;
        break;
    }
    int symbol = 0;
    int escape = 0;
    if (eob_pt <= 8) {
      symbol = eob_pt - 1;
    } else {
      symbol = 7;
      escape = eob_pt - 8;
    }
    if (!sym(cdf, cdf_n, symbol)) return false;
    if (symbol == 7 && escape_bits > 0) {
      if (!write_literal(escape_bits, static_cast<std::uint32_t>(escape))) {
        return false;
      }
    }
    if (eob_pt >= 3) {
      const int base = (1 << (eob_pt - 2)) + 1;
      const int extra = eob - base;
      const int eob_extra = (extra >> (eob_pt - 3)) & 1;
      if (!sym(&cdfs_.eob_extra[0], 2, eob_extra)) return false;
      for (int i = eob_pt - 4; i >= 0; --i) {
        if (!raw_bit(((extra >> i) & 1) != 0)) return false;
      }
    }
    return true;
  }

  bool write_quant_escape(int x, int& hr_level_avg) {
    const int pred_level = hr_level_avg;
    const int m = clip3(1, 6, pred_level <= 0 ? 0 : floor_log2(pred_level));
    const int k = m + 1;
    const int c_max = std::min(m + 4, 6);
    if (x < (c_max << m)) {
      const int q = x >> m;
      for (int i = 0; i < q; ++i) {
        if (!raw_bit(false)) return false;
      }
      if (!raw_bit(true)) return false;
      if (!write_literal(m, static_cast<std::uint32_t>(x & ((1 << m) - 1)))) {
        return false;
      }
    } else {
      for (int i = 0; i < c_max; ++i) {
        if (!raw_bit(false)) return false;
      }
      int g = 0;
      while (true) {
        const int length = k + g;
        const int x_base = (c_max << m) + (1 << length) - (1 << k);
        if (x < x_base + (1 << length)) {
          for (int i = 0; i < g; ++i) {
            if (!raw_bit(false)) return false;
          }
          if (!raw_bit(true)) return false;
          if (!write_literal(length,
                             static_cast<std::uint32_t>(x - x_base))) {
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

  int dc_sign_context(int x4, int y4, int w4, int h4) const {
    int dc_sign = 0;
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < mi_cols_) {
        const int sign =
            above_dc_[0][static_cast<std::size_t>(x4 + k)];
        if (sign == 1) --dc_sign;
        else if (sign == 2) ++dc_sign;
      }
    }
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < mi_rows_) {
        const int sign =
            left_dc_[0][static_cast<std::size_t>(y4 + k)];
        if (sign == 1) --dc_sign;
        else if (sign == 2) ++dc_sign;
      }
    }
    if (dc_sign < 0) return 1;
    if (dc_sign > 0) return 2;
    return 0;
  }

  bool code_tx(const TxPlan& plan, int mi_row, int mi_col, int bsize_w,
               int bsize_h) {
    const int plane = plan.plane;
    const int adj_w = plan.adj_w;
    const int adj_h = plan.adj_h;
    const int area = adj_w * adj_h;
    const int w4 = plan.coded_w >> 2;
    const int h4 = plan.coded_h >> 2;
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int x4 = mi_col >> sub_x;
    const int y4 = mi_row >> sub_y;
    const int max_x4 = plane == 0 ? mi_cols_ : mi_cols_ / 2;
    const int max_y4 = plane == 0 ? mi_rows_ : mi_rows_ / 2;

    int ctx = 0;
    if (plane == 0) {
      int top = 0;
      int left = 0;
      for (int k = 0; k < w4; ++k) {
        if (x4 + k < max_x4) top |= above_level(0, x4 + k);
      }
      for (int k = 0; k < h4; ++k) {
        if (y4 + k < max_y4) left |= left_level(0, y4 + k);
      }
      top = std::min(top, 4);
      left = std::min(left, 4);
      if (bsize_w == plan.coded_w && bsize_h == plan.coded_h) {
        ctx = 0;
      } else {
        ctx = (top + left + 3) >> 1;
      }
    } else {
      int above = 0;
      int left = 0;
      for (int i = 0; i < w4; ++i) {
        if (x4 + i < max_x4) {
          above |= above_level(plane, x4 + i);
          above |= above_dc(plane, x4 + i);
        }
      }
      for (int i = 0; i < h4; ++i) {
        if (y4 + i < max_y4) {
          left |= left_level(plane, y4 + i);
          left |= left_dc(plane, y4 + i);
        }
      }
      ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
      if (plane == 2) {
        if (bsize_w * bsize_h > plan.coded_w * plan.coded_h) ctx += 3;
        if (eob_u_ != 0) ctx += 6;
      } else {
        ctx += 6;
      }
    }

    const int tsc = tx_sz_ctx(plan.tx_size);
    std::uint16_t* skip_cdf = plane == 2 ? &cdfs_.v_txb_skip[ctx][0]
                                         : &cdfs_.txb_skip[0][tsc][ctx][0];
    if (!sym(skip_cdf, 2, plan.eob == 0 ? 1 : 0)) return false;
    if (plan.eob == 0) {
      for (int i = 0; i < w4; ++i) {
        above_level(plane, x4 + i) = 0;
        above_dc(plane, x4 + i) = 0;
      }
      for (int i = 0; i < h4; ++i) {
        left_level(plane, y4 + i) = 0;
        left_dc(plane, y4 + i) = 0;
      }
      if (plane == 1) eob_u_ = 0;
      return true;
    }

    if (!write_eob(plan.eob, plane, plan.coded_w, plan.coded_h)) return false;
    if (plane == 0 && luma_tx_type_signalled(plan.tx_size, plan.eob)) {
      // DCT_DCT is symbol 0 in every TX_SET_INTRA_1 mode table.
      if (!sym(&cdfs_.intra_tx_type_set1[kTxSizeSqr[plan.tx_size]][0], 7,
               0)) {
        return false;
      }
    }

    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    build_scan(adj_w, adj_h, scan);
    std::vector<int> level_pre(static_cast<std::size_t>(area), 0);
    for (int c = plan.eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / adj_w;
      const int col = pos % adj_w;
      const int abs_value = std::abs(plan.levels[static_cast<std::size_t>(pos)]);
      const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
      const int base_levels = is_lf ? 4 : 2;
      const int base_max = is_lf ? 5 : 3;
      const int br_max = (is_lf && plane > 0) ? 0 : 3;
      const int pre_max = base_max + br_max;
      int base_level = 0;
      int br_level = 0;
      if (abs_value <= base_levels) {
        base_level = abs_value;
      } else {
        base_level = std::min(base_levels + 1, base_max);
        const int pre = std::min(abs_value, pre_max);
        br_level = pre - base_level;
      }

      int ctx_eob = 0;
      if (c != 0) {
        if (c <= area / 8) ctx_eob = 1;
        else if (c <= area / 4) ctx_eob = 2;
        else ctx_eob = 3;
      }

      if (c == plan.eob - 1) {
        const int symbol = base_level - 1;
        if (plane == 0) {
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_eob[tsc][ctx_eob][0], 5,
                     symbol)) return false;
          } else {
            if (!sym(&cdfs_.coeff_base_eob[tsc][ctx_eob][0], 3, symbol)) {
              return false;
            }
          }
        } else {
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_eob_uv[ctx_eob][0], 5, symbol)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob_uv[ctx_eob][0], 3, symbol)) {
              return false;
            }
          }
        }
      } else {
        int mag = 0;
        const int num_ref = plane == 0 ? 5 : 3;
        for (int idx = 0; idx < num_ref; ++idx) {
          const int rr = row + kSigRef[idx][0];
          const int cc = col + kSigRef[idx][1];
          if (rr < adj_h && cc < adj_w) {
            const int limit = is_lf ? 5 : 3;
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * adj_w + cc], limit);
          }
        }
        const int mctx = (mag + 1) >> 1;
        if (plane == 0) {
          if (is_lf) {
            int lf_ctx = 0;
            if (c == 0) lf_ctx = std::min(mctx, 8);
            else if (row + col < 2) lf_ctx = std::min(mctx, 6) + 9;
            else lf_ctx = std::min(mctx, 4) + 16;
            if (!sym(&cdfs_.coeff_base_lf[tsc][lf_ctx][0][0], 6, base_level)) {
              return false;
            }
          } else {
            const int ctx2 = std::min(mctx, 4);
            int hf_ctx = ctx2;
            if (row + col >= 6 && row + col < 8) hf_ctx = ctx2 + 5;
            else if (row + col >= 8) hf_ctx = ctx2 + 10;
            if (!sym(&cdfs_.coeff_base[tsc][hf_ctx][0][0], 4, base_level)) {
              return false;
            }
          }
        } else {
          const int ctx2 = std::min(mctx, 3);
          const int uv_ctx = plane == 1 ? ctx2 : ctx2 + 4;
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_uv[uv_ctx][0], 6, base_level)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_uv[uv_ctx][0], 4, base_level)) {
              return false;
            }
          }
        }
      }

      const int pre_level = base_level + br_level;
      if (pre_level > base_levels && !(is_lf && plane > 0)) {
        int mag = 0;
        for (int idx = 0; idx < 3; ++idx) {
          const int rr = row + kMagRef[idx][0];
          const int cc = col + kMagRef[idx][1];
          if (rr < adj_h && cc < adj_w) {
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * adj_w + cc], 5);
          }
        }
        mag = std::min((mag + 1) >> 1, 6);
        if (plane == 0) {
          int br_ctx = mag;
          if (is_lf) {
            if (pos != 0) br_ctx = mag + 7;
            if (!sym(&cdfs_.coeff_br_lf[br_ctx][0], 4, br_level)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_br[mag][0], 4, br_level)) return false;
          }
        } else {
          if (!sym(&cdfs_.coeff_br_uv[std::min(mag, 3)][0], 4, br_level)) {
            return false;
          }
        }
      }
      level_pre[static_cast<std::size_t>(pos)] = pre_level;
    }

    int cul_level = 0;
    int dc_category = 0;
    int hr_level_avg = 0;
    for (int c = plan.eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / adj_w;
      const int col = pos % adj_w;
      const int pre_level = level_pre[static_cast<std::size_t>(pos)];
      const int abs_value = std::abs(plan.levels[static_cast<std::size_t>(pos)]);
      const bool negative = plan.levels[static_cast<std::size_t>(pos)] < 0;
      if (pre_level != 0) {
        if (row == 0 && col == 0 && plane == 0) {
          const int sign_ctx = dc_sign_context(x4, y4, w4, h4);
          if (!sym(&cdfs_.dc_sign[0][0][sign_ctx][0], 2,
                   negative ? 1 : 0)) {
            return false;
          }
        } else {
          if (!raw_bit(negative)) return false;
        }
        const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
        const int max_level = is_lf ? (plane == 0 ? 8 : 5) : 6;
        if (abs_value >= max_level) {
          if (!write_quant_escape(abs_value - pre_level, hr_level_avg)) {
            return false;
          }
        }
      }
      if (pos == 0 && abs_value > 0) dc_category = negative ? 1 : 2;
      cul_level = std::min(4, cul_level + abs_value);
    }

    for (int i = 0; i < w4; ++i) {
      above_level(plane, x4 + i) = static_cast<std::uint8_t>(cul_level);
      above_dc(plane, x4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    for (int i = 0; i < h4; ++i) {
      left_level(plane, y4 + i) = static_cast<std::uint8_t>(cul_level);
      left_dc(plane, y4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    if (plane == 1) eob_u_ = 1;
    return true;
  }

  // ---------------------------------------------------------------------
  // Coding a block
  // ---------------------------------------------------------------------
  bool encode_block(const partition::PartitionNode& node) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    cur_r_ = r;
    cur_c_ = c;
    cur_w4_ = w4;
    cur_h4_ = h4;

    ModeChoice y;
    if (!choose_luma_mode(x0, y0, w, h, y)) return false;

    std::vector<std::uint8_t> luma_pred;
    if (!predict_block(0, x0, y0, w, h, y.mode, y.angle_delta, luma_pred)) {
      return false;
    }
    TxPlan luma_plan;
    if (!compute_tx(0, x0, y0, w, h, luma_pred, luma_plan)) return false;
    write_recon(0, x0, y0, w, h, luma_plan);

    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, node.size);
    if (!partition::is_valid_block_size(chroma_size)) {
      error_ = "invalid chroma residual size";
      return false;
    }
    const int cw = partition::block_width(chroma_size);
    const int ch = partition::block_height(chroma_size);
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;

    UvChoice uv;
    if (!choose_uv_mode(cx0, cy0, cw, ch, y.av2_mode, y.angle_delta, uv)) {
      return false;
    }

    std::vector<std::uint8_t> u_pred;
    std::vector<std::uint8_t> v_pred;
    bool cfl = uv.is_cfl;
    if (cfl) {
      // The CfL component expects the DC prediction in the chroma plane and
      // reconstructed above/left neighbours. Derive the implicit alpha and
      // write the CfL prediction into the planes.
      if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_pred)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_pred)) {
        return false;
      }
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          recon_[1].at(cx0 + xx, cy0 + yy) =
              u_pred[static_cast<std::size_t>(yy) * cw + xx];
          recon_[2].at(cx0 + xx, cy0 + yy) =
              v_pred[static_cast<std::size_t>(yy) * cw + xx];
        }
      }
      cfl::Scratch scratch;
      cfl::Block ub;
      cfl::Block vb;
      make_cfl_block(1, cx0, cy0, cw, ch, ub);
      make_cfl_block(2, cx0, cy0, cw, ch, vb);
      cfl::compute_luma(ub, scratch);
      const int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
      cfl::compute_luma(vb, scratch);
      const int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
      cfl::apply_prediction(ub, alpha_u, scratch);
      cfl::apply_prediction(vb, alpha_v, scratch);
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          u_pred[static_cast<std::size_t>(yy) * cw + xx] =
              recon_[1].at(cx0 + xx, cy0 + yy);
          v_pred[static_cast<std::size_t>(yy) * cw + xx] =
              recon_[2].at(cx0 + xx, cy0 + yy);
        }
      }
    } else {
      if (!predict_block(1, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                         u_pred)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                         v_pred)) {
        return false;
      }
    }

    TxPlan u_plan;
    TxPlan v_plan;
    if (!compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan)) return false;
    if (!compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) return false;
    write_recon(1, cx0, cy0, cw, ch, u_plan);
    write_recon(2, cx0, cy0, cw, ch, v_plan);

    if (trace_) {
      std::cerr << "m1 block r=" << r << " c=" << c
                << " size=" << partition::block_size_name(node.size)
                << " y=" << y.av2_mode << "/" << y.angle_delta
                << " uv=" << (cfl ? -1 : uv.av2_mode) << "/"
                << uv.angle_delta << " cfl=" << (cfl ? 1 : 0)
                << " tx=" << luma_plan.coded_w << "x" << luma_plan.coded_h
                << " eob=" << luma_plan.eob << "\n";
    }

    // Syntax order: mode_info() then residual().
    if (!code_y_mode(y)) return false;
    if (!code_uv_mode(cfl, uv.av2_mode, y.av2_mode)) return false;
    if (!code_tx(luma_plan, r, c, w, h)) return false;
    if (!code_tx(u_plan, r, c, cw, ch)) return false;
    if (!code_tx(v_plan, r, c, cw, ch)) return false;

    // Update partition/mode state after the block.
    for (int yy = 0; yy < h4; ++yy) {
      left_mi(r + yy) = kMiHeightLog2[static_cast<int>(node.size)];
    }
    for (int xx = 0; xx < w4; ++xx) {
      above_mi(c + xx) = kMiWidthLog2[static_cast<int>(node.size)];
    }
    const YModeCode ycode = y_mode_code(y.av2_mode, y.angle_delta);
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        intra_joint(r + yy, c + xx) = ycode.joint;
      }
    }
    const int cr = r / 2;
    const int cc = c / 2;
    const int cw4 = cw / 4;
    const int ch4 = ch / 4;
    for (int yy = 0; yy < ch4; ++yy) {
      for (int xx = 0; xx < cw4; ++xx) {
        uv_smooth(cr + yy, cc + xx) =
            (!cfl && y_mode_is_smooth(uv.av2_mode)) ? 1 : 0;
        uvcfl(cr + yy, cc + xx) = cfl ? 1 : 0;
      }
    }

    for (int plane = 0; plane < 3; ++plane) {
      decoded_[plane].fill(
          plane == 0 ? c : c / 2, plane == 0 ? r : r / 2,
          plane == 0 ? w4 : cw4, plane == 0 ? h4 : ch4);
    }

    if (options_.enable_deblock) {
      fill_deblock_mi(node, cw, ch, luma_plan.tx_size, u_plan.tx_size);
    }
    return true;
  }

  void make_cfl_block(int plane, int cx0, int cy0, int cw, int ch,
                      cfl::Block& block) {
    block.luma = recon_[0].samples.data();
    block.luma_stride = recon_[0].width;
    block.luma_max_x = recon_[0].width - 1;
    block.luma_max_y = recon_[0].height - 1;
    block.sb_top_y = (cy0 * 2 / 64) * 64;
    block.chroma = recon_[plane].samples.data();
    block.chroma_stride = recon_[plane].width;
    block.x = cx0;
    block.y = cy0;
    block.w = cw;
    block.h = ch;
    block.coded_width = mi_cols_ * 2;
    block.coded_height = mi_rows_ * 2;
    block.have_above = cfl::cfl_avail_above(cy0, 1) ? 1 : 0;
    block.have_left = cfl::cfl_avail_left(cx0, 1) ? 1 : 0;
    block.bit_depth = 8;
    block.ds_filter = 0;
    block.sub_x = 1;
    block.sub_y = 1;
  }

  bool choose_uv_mode(int cx0, int cy0, int cw, int ch, int y_mode,
                      int y_angle_delta, UvChoice& best) {
    best.is_cfl = false;
    best.sse = std::numeric_limits<std::int64_t>::max();
    if (options_.force_dc || !options_.enable_cfl) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    std::vector<std::uint8_t> pred;

    // Try the 13 base modes with angle delta 0, plus the luma mode with the
    // luma angle delta (so AngleDeltaUV can be inherited).
    std::array<std::pair<intra::Mode, int>, 14> candidates{};
    int count = 0;
    static const intra::Mode kBase[13] = {
        intra::Mode::kDc,      intra::Mode::kV,       intra::Mode::kH,
        intra::Mode::kD45,     intra::Mode::kD135,    intra::Mode::kD113,
        intra::Mode::kD157,    intra::Mode::kD203,    intra::Mode::kD67,
        intra::Mode::kSmooth,  intra::Mode::kSmoothV, intra::Mode::kSmoothH,
        intra::Mode::kPaeth};
    for (int i = 0; i < 13; ++i) {
      candidates[static_cast<std::size_t>(count++)] = {kBase[i], 0};
    }
    if (y_mode_is_directional(y_mode) && y_angle_delta != 0) {
      candidates[static_cast<std::size_t>(count++)] = {
          static_cast<intra::Mode>(y_mode), y_angle_delta};
    }

    for (int i = 0; i < count; ++i) {
      const intra::Mode mode = candidates[static_cast<std::size_t>(i)].first;
      const int delta = candidates[static_cast<std::size_t>(i)].second;
      if (!predict_block(1, cx0, cy0, cw, ch, mode, delta, pred)) return false;
      const std::int64_t sse = sse_against_source(1, cx0, cy0, cw, ch, pred);
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = mode;
        best.angle_delta = delta;
        best.av2_mode = av2_y_mode(mode);
      }
    }

    if (options_.enable_cfl) {
      std::vector<std::uint8_t> u_dc;
      std::vector<std::uint8_t> v_dc;
      if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_dc)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_dc)) {
        return false;
      }
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          recon_[1].at(cx0 + xx, cy0 + yy) =
              u_dc[static_cast<std::size_t>(yy) * cw + xx];
          recon_[2].at(cx0 + xx, cy0 + yy) =
              v_dc[static_cast<std::size_t>(yy) * cw + xx];
        }
      }
      cfl::Scratch scratch;
      cfl::Block ub;
      cfl::Block vb;
      make_cfl_block(1, cx0, cy0, cw, ch, ub);
      make_cfl_block(2, cx0, cy0, cw, ch, vb);
      cfl::compute_luma(ub, scratch);
      const int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
      cfl::compute_luma(vb, scratch);
      const int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
      cfl::apply_prediction(ub, alpha_u, scratch);
      cfl::apply_prediction(vb, alpha_v, scratch);
      std::int64_t sse = 0;
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          const int su = source_sample(1, cx0 + xx, cy0 + yy);
          const int pu = recon_[1].at(cx0 + xx, cy0 + yy);
          const int du = su - pu;
          sse += static_cast<std::int64_t>(du) * du;
          const int sv = source_sample(2, cx0 + xx, cy0 + yy);
          const int pv = recon_[2].at(cx0 + xx, cy0 + yy);
          const int dv = sv - pv;
          sse += static_cast<std::int64_t>(dv) * dv;
        }
      }
      if (sse < best.sse) {
        best.sse = sse;
        best.is_cfl = true;
        best.mode = intra::Mode::kDc;
        best.angle_delta = 0;
        best.av2_mode = kDcPred;
      }
    }
    return true;
  }

  void fill_deblock_mi(const partition::PartitionNode& node, int cw, int ch,
                       int luma_tx_size, int chroma_tx_size) {
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w / 4;
    const int h4 = h / 4;
    const TxSize luma = static_cast<TxSize>(luma_tx_size);
    const TxSize chroma = static_cast<TxSize>(chroma_tx_size);
    (void)cw;
    (void)ch;
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        const int idx = (node.mi_row + yy) * mi_cols_ + (node.mi_col + xx);
        deblock::DeblockMi mi;
        mi.block_col_base = static_cast<std::uint16_t>(node.mi_col);
        mi.block_row_base = static_cast<std::uint16_t>(node.mi_row);
        mi.block_w4 = static_cast<std::uint8_t>(w4);
        mi.block_h4 = static_cast<std::uint8_t>(h4);
        mi.sub_pu_size = luma;
        mi.tx_size = luma;
        mi.tx_col_base = static_cast<std::uint16_t>(node.mi_col);
        mi.tx_row_base = static_cast<std::uint16_t>(node.mi_row);
        mi.sub_pu_col_base = static_cast<std::uint16_t>(node.mi_col);
        mi.sub_pu_row_base = static_cast<std::uint16_t>(node.mi_row);
        mi.skip = 0;
        mi.region_intra = 1;
        mi.lossless = 0;
        mi.qindex = options_.base_q_idx;
        deblock_mi_[0][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[1][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[2][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[1][static_cast<std::size_t>(idx)].tx_size = chroma;
        deblock_mi_[2][static_cast<std::size_t>(idx)].tx_size = chroma;
      }
    }
  }

  // Members
  const StillPicture& picture_;
  const M1Options& options_;
  FrameGeometry geometry_;
  int width_ = 0;
  int height_ = 0;
  int chroma_width_ = 0;
  int chroma_height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int luma_stride_ = 0;
  int luma_height_ = 0;
  int chroma_stride_ = 0;
  int chroma_padded_height_ = 0;

  CdfState cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_buffer_;
  std::string error_;
  Plane recon_[3];

  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::array<std::vector<std::uint8_t>, 3> above_dc_;
  std::array<std::vector<std::uint8_t>, 3> left_dc_;
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
  std::vector<int> intra_joint_;
  std::vector<std::uint8_t> uv_smooth_;
  std::vector<std::uint8_t> uvcfl_;
  std::array<DecodedMap, 3> decoded_;

  int sb_mi_row_ = 0;
  int sb_mi_col_ = 0;
  int sb_cols_plane_[3] = {0, 0, 0};
  int sb_rows_plane_[3] = {0, 0, 0};

  int cur_r_ = 0;
  int cur_c_ = 0;
  int cur_w4_ = 0;
  int cur_h4_ = 0;
  int eob_u_ = 0;
  bool trace_ = std::getenv("AV2ENC_M1_TRACE") != nullptr;

  std::array<std::vector<deblock::DeblockMi>, 3> deblock_mi_;

  using TxSize = deblock::TxSize;
};

}  // namespace

bool encode_m1_still_picture(const StillPicture& picture,
                             const M1Options& options, M1Result& result,
                             std::string& error) {
  error.clear();
  if (picture.width == 0 || picture.height == 0) {
    error = "empty picture";
    return false;
  }
  if ((picture.width & 1U) != 0U || (picture.height & 1U) != 0U) {
    error = "M1 4:2:0 requires even dimensions";
    return false;
  }
  if (options.base_q_idx != 90) {
    error = "M1 pins base_q_idx = 90 (coefficient CDF qctx slice 0)";
    return false;
  }
  const std::size_t luma_size = static_cast<std::size_t>(picture.width) *
                                static_cast<std::size_t>(picture.height);
  const std::size_t chroma_size =
      static_cast<std::size_t>((picture.width + 1U) / 2U) *
      static_cast<std::size_t>((picture.height + 1U) / 2U);
  if (picture.y.size() < luma_size || picture.u.size() < chroma_size ||
      picture.v.size() < chroma_size) {
    error = "picture planes are too small";
    return false;
  }

  Encoder encoder(picture, options);
  std::vector<std::uint8_t> tile_payload;
  if (!encoder.encode(&tile_payload, error)) {
    return false;
  }

  M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_intra_edge_filter = options.enable_intra_edge_filter;
  config.enable_cfl_intra = options.enable_cfl;
  config.enable_chroma_dctonly = true;
  config.tx_mode_select = false;
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u = options.enable_deblock;
  config.apply_deblock_v = options.enable_deblock;

  result.frame = assemble_m1_still_picture(config, tile_payload);
  result.tile_bytes = static_cast<std::uint32_t>(tile_payload.size());
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());

  // Deblock the reconstruction before reporting it. Intra prediction already
  // used the unfiltered reconstruction, matching normative filter order.
  if (options.enable_deblock) {
    deblock::DeblockFrame frame;
    for (int plane = 0; plane < 3; ++plane) {
      frame.planes[plane].data = encoder.recon_mut(plane).samples.data();
      frame.planes[plane].stride = encoder.recon_mut(plane).width;
      frame.mi[plane] = encoder.deblock_mi(plane);
    }
    frame.mi_cols = encoder.mi_cols();
    frame.mi_rows = encoder.mi_rows();
    frame.subsampling_x = 1;
    frame.subsampling_y = 1;
    deblock::DeblockOptions dopts;
    for (int i = 0; i < 4; ++i) dopts.apply_deblocking_filter[i] = true;
    dopts.frame_is_intra = true;
    dopts.enable_sdp = false;
    const deblock::DeblockStatus status =
        deblock::deblock_frame(frame, dopts);
    if (status != deblock::DeblockStatus::kOk) {
      error = std::string("deblock failed: ") +
              deblock::deblock_status_string(status);
      return false;
    }
  }

  const auto crop = [](const Plane& plane, int width, int height) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
      std::copy_n(plane.samples.begin() +
                      static_cast<std::ptrdiff_t>(y) * plane.width,
                  width,
                  out.begin() + static_cast<std::ptrdiff_t>(y) * width);
    }
    return out;
  };
  result.recon_y = crop(encoder.recon(0), static_cast<int>(picture.width),
                        static_cast<int>(picture.height));
  result.recon_u =
      crop(encoder.recon(1), static_cast<int>((picture.width + 1U) / 2U),
           static_cast<int>((picture.height + 1U) / 2U));
  result.recon_v =
      crop(encoder.recon(2), static_cast<int>((picture.width + 1U) / 2U),
           static_cast<int>((picture.height + 1U) / 2U));
  return true;
}

}  // namespace av2enc::m1
