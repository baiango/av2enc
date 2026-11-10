#include "encoder/m0_still.h"

#include "encoder/m0_cdf_tables.h"
#include "encoder/m0_tables.h"
#include "entropy/av2/av2_entropy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::m0 {
namespace {

// Block sizes from Table 6.22 (the enum order is normative; see
// spec/tables/block_tables.json `enum_block_size`).
constexpr int kBlock32x32 = 9;
constexpr int kBlock32x64 = 10;
constexpr int kBlock64x32 = 11;
constexpr int kBlock64x64 = 12;



// RECT_HORZ / RECT_VERT (§6.19.3.1).
constexpr int kRectHorz = 0;
constexpr int kRectVert = 1;

// Partition_Size_Adjust and Partition_Size_Adjust_Rect_Type (§9.2) for the
// four sizes the M0 tree uses.
constexpr int kSizeAdjust[4] = {2, 3, 3, 3};      // 32x32, 32x64, 64x64, 64x32
constexpr int kRectSizeAdjust[4] = {3, 4, 6, 5};  // same order

// Mi_Width_Log2 / Mi_Height_Log2 for the same sizes.
constexpr int kMiLog2W[4] = {3, 3, 4, 4};
constexpr int kMiLog2H[4] = {3, 4, 4, 3};

constexpr int kSizeIndex(int block_size) {
  switch (block_size) {
    case kBlock32x32:
      return 0;
    case kBlock32x64:
      return 1;
    case kBlock64x32:
      return 3;
    default:
      return 2;
  }
}

constexpr std::size_t kTileCapacityBytes = 4U * 1024U * 1024U;

// The generated header carries qctx slice 0, i.e. base_q_idx <= 90.
constexpr std::uint8_t kPinnedBaseQ = 90;

struct CdfState {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_pt_256[3][9];
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

void init_cdfs(CdfState& cdfs) {
  copy_array(cdfs.do_split, kDoSplitCdf);
  copy_array(cdfs.rect_type, kRectTypeCdf);
  copy_array(cdfs.y_mode_set, kYModeSetCdf);
  copy_array(cdfs.y_mode_index, kYModeIndexCdf);
  copy_array(cdfs.uv_mode, kUvModeCflNotAllowedCdf);
  copy_array(cdfs.txb_skip, kTxbSkipCdf);
  copy_array(cdfs.v_txb_skip, kVTxbSkipCdf);
  copy_array(cdfs.eob_pt_1024, kEobPt1024Cdf);
  copy_array(cdfs.eob_pt_256, kEobPt256Cdf);
  copy_array(cdfs.eob_extra, kEobExtraCdf);
  copy_array(cdfs.coeff_base, kCoeffBaseCdf);
  copy_array(cdfs.coeff_base_lf, kCoeffBaseLfCdf);
  copy_array(cdfs.coeff_base_eob, kCoeffBaseEobCdf);
  copy_array(cdfs.coeff_base_lf_eob, kCoeffBaseLfEobCdf);
  copy_array(cdfs.coeff_br, kCoeffBrCdf);
  copy_array(cdfs.coeff_br_lf, kCoeffBrLfCdf);
  copy_array(cdfs.coeff_base_uv, kCoeffBaseUvCdf);
  copy_array(cdfs.coeff_base_lf_uv, kCoeffBaseLfUvCdf);
  copy_array(cdfs.coeff_base_eob_uv, kCoeffBaseEobUvCdf);
  copy_array(cdfs.coeff_base_lf_eob_uv, kCoeffBaseLfEobUvCdf);
  copy_array(cdfs.coeff_br_uv, kCoeffBrUvCdf);
  copy_array(cdfs.dc_sign, kDcSignCdf);
  copy_array(cdfs.intra_tx_type_set1, kIntraTxTypeSet1Cdf);
}

struct Plane {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), value);
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

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int clip1(int value) { return clip3(0, 255, value); }

// §4.8: Round2(x, n) = (x + (1 << (n - 1))) >> n (arithmetic shift).
int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §7.13.2.11 / §7.13.3.22.  All M0 denominators are powers of two (16, 32,
// 64), for which resolve_divisor() yields f = 0, Div_Lut[0] = 512.
int approx_divide(int numerator, int denominator) {
  const int shift = 9 + floor_log2(denominator);
  return round2(numerator * 512, shift);
}

class TileEncoder {
 public:
  TileEncoder(const StillPicture& picture, const StillPictureConfig& config,
              M0PartitionMode partition_mode)
      : picture_(picture), config_(config), partition_mode_(partition_mode) {
    mi_cols_ = 2 * ((picture.width + 7U) >> 3U);
    mi_rows_ = 2 * ((picture.height + 7U) >> 3U);
    chroma_width_ = (picture.width + 1U) / 2U;
    chroma_height_ = (picture.height + 1U) / 2U;

    init_cdfs(cdfs_);
    real_coeffs_ = partition_mode == M0PartitionMode::k32x32 ||
                   partition_mode == M0PartitionMode::k32x32HorzFirst;
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = config.disable_cdf_update ? 1 : 0;

    recon_[0].resize(static_cast<int>(picture.width), static_cast<int>(picture.height), 128);
    recon_[1].resize(static_cast<int>(chroma_width_), static_cast<int>(chroma_height_), 128);
    recon_[2].resize(static_cast<int>(chroma_width_), static_cast<int>(chroma_height_), 128);

    for (int plane = 0; plane < 3; ++plane) {
      above_level_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 0);
      left_level_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 0);
      above_dc_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 0);
      left_dc_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 0);
    }
    for (int plane = 0; plane < 2; ++plane) {
      above_mi_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 6);
      left_mi_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 6);
    }
    intra_joint_.assign(static_cast<std::size_t>(mi_rows_) *
                            static_cast<std::size_t>(mi_cols_),
                        0);
    clear_above_context();
  }

  bool encode(std::vector<std::uint8_t>* tile_payload, std::string& error) {
    clear_above_context();
    for (int r = 0; r < mi_rows_; r += 16) {
      clear_left_context();
      for (int c = 0; c < mi_cols_; c += 16) {
        bool ok = false;
        switch (partition_mode_) {
          case M0PartitionMode::k64x64Single:
            ok = encode_partition_none_64(r, c);
            break;
          case M0PartitionMode::k64x64VertNone:
            ok = encode_partition_64_rect(r, c, kRectVert, false);
            break;
          case M0PartitionMode::k64x64HorzNone:
            ok = encode_partition_64_rect(r, c, kRectHorz, false);
            break;
          case M0PartitionMode::k32x32HorzFirst:
            ok = encode_partition_64_rect(r, c, kRectHorz, true);
            break;
          case M0PartitionMode::k32x32:
          default:
            ok = encode_partition_64_rect(r, c, kRectVert, true);
            break;
        }
        if (!ok) {
          error = error_;
          return false;
        }
      }
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      error = std::string("entropy exit_symbol failed: ") + av2::status_name(status);
      return false;
    }
    tile_payload->assign(tile_buffer_.begin(),
                         tile_buffer_.begin() + static_cast<std::ptrdiff_t>(payload_bytes));
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }

 private:
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_symbol failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

  bool raw_bit(bool bit) {
    const av2::Status status = enc_.encode_bool(bit);
    if (status != av2::Status::kOk) {
      error_ = std::string("entropy encode_bool failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

  void clear_above_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(above_level_[plane].begin(), above_level_[plane].end(), 0);
      std::fill(above_dc_[plane].begin(), above_dc_[plane].end(), 0);
    }
    for (int plane = 0; plane < 2; ++plane) {
      std::fill(above_mi_[plane].begin(), above_mi_[plane].end(), 6);
    }
  }

  void clear_left_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(left_level_[plane].begin(), left_level_[plane].end(), 0);
      std::fill(left_dc_[plane].begin(), left_dc_[plane].end(), 0);
    }
    for (int plane = 0; plane < 2; ++plane) {
      std::fill(left_mi_[plane].begin(), left_mi_[plane].end(), 6);
    }
  }

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
  std::uint8_t& above_mi(int plane, int x) {
    return above_mi_[plane][static_cast<std::size_t>(x)];
  }
  std::uint8_t& left_mi(int plane, int y) {
    return left_mi_[plane][static_cast<std::size_t>(y)];
  }
  std::uint8_t& intra_joint(int r, int c) {
    return intra_joint_[static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
                        static_cast<std::size_t>(c)];
  }

  bool is_inside(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }

  // get_joint_mode(dir) (§5.20.5.5).
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
    if (is_inside(mv_row, mv_col)) {
      return intra_joint_[static_cast<std::size_t>(mv_row) *
                              static_cast<std::size_t>(mi_cols_) +
                          static_cast<std::size_t>(mv_col)];
    }
    return 0;  // DC_PRED
  }

  bool encode_partition_64_rect(int r, int c, int rect_type,
                                bool split_children) {
    const int idx = kSizeIndex(kBlock64x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 1)) {
      return false;
    }
    const int rect_ctx =
        kRectSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, rect_type)) {
      return false;
    }
    if (rect_type == kRectVert) {
      // PARTITION_VERT -> two 32x64 blocks at (r, c) and (r, c + 8).
      if (split_children) {
        return encode_partition_32x64(r, c) && encode_partition_32x64(r, c + 8);
      }
      return encode_partition_none_32x64(r, c) &&
             encode_partition_none_32x64(r, c + 8);
    }
    // PARTITION_HORZ -> two 64x32 blocks at (r, c) and (r + 8, c).
    if (split_children) {
      return encode_partition_64x32(r, c) && encode_partition_64x32(r + 8, c);
    }
    return encode_partition_none_64x32(r, c) &&
           encode_partition_none_64x32(r + 8, c);
  }

  bool encode_partition_none_32x64(int r, int c) {
    const int idx = kSizeIndex(kBlock32x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int ctx = kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
                    (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_32x64(r, c);
  }

  bool encode_partition_none_64x32(int r, int c) {
    const int idx = kSizeIndex(kBlock64x32);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int ctx = kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
                    (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_64x32(r, c);
  }

  bool encode_partition_64x32(int r, int c) {
    // 64x32 -> two 32x32 blocks by VERT.
    const int idx = kSizeIndex(kBlock64x32);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int ctx = kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
                    (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][ctx][0], 2, 1)) {
      return false;
    }
    const int rect_ctx = kRectSizeAdjust[idx] * 4 +
                         (left_mi(0, r) < bsh ? 2 : 0) +
                         (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, kRectVert)) {
      return false;
    }
    return encode_partition_32x32(r, c) && encode_partition_32x32(r, c + 8);
  }

  bool encode_partition_32x64(int r, int c) {
    const int idx = kSizeIndex(kBlock32x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 1)) {
      return false;
    }
    const int rect_ctx =
        kRectSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, kRectHorz)) {
      return false;
    }
    // PARTITION_HORZ -> two 32x32 blocks at (r, c) and (r + 8, c).
    return encode_partition_32x32(r, c) && encode_partition_32x32(r + 8, c);
  }

  bool encode_partition_32x32(int r, int c) {
    const int idx = kSizeIndex(kBlock32x32);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_32x32(r, c);
  }

  bool encode_partition_none_64(int r, int c) {
    const int idx = kSizeIndex(kBlock64x64);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int do_split_ctx =
        kSizeAdjust[idx] * 4 + (left_mi(0, r) < bsh ? 2 : 0) +
        (above_mi(0, c) < bsw ? 1 : 0);
    if (!sym(&cdfs_.do_split[0][do_split_ctx][0], 2, 0)) {
      return false;
    }
    return encode_block_64x64(r, c);
  }

  bool encode_mode_info_dc(int r, int c, int w4, int h4) {
    // mode_info() -> intra_frame_mode_info() (§5.20.5.3) for DC_PRED.
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) {
      return false;
    }
    const int joint_ctx = (get_joint_mode(r, c, w4, h4, 0) >= 5 ? 1 : 0) +
                          (get_joint_mode(r, c, w4, h4, 1) >= 5 ? 1 : 0);
    if (!sym(&cdfs_.y_mode_index[joint_ctx][0], 8, 0)) {
      return false;
    }
    return sym(&cdfs_.uv_mode[0][0], 8, 0);
  }

  bool encode_block_32x32(int r, int c) {
    if (!encode_mode_info_dc(r, c, 8, 8)) {
      return false;
    }

    // read_block_tx_size(): TxMode == TX_MODE_LARGEST, no syntax.
    // residual(): plane 0 is one TX_32X32; chroma planes are one TX_16X16.
    if (!encode_residual_plane(0, r, c, 32, 32, 3, 32, 32)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 16, 16, 2, 16, 16)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 16, 16, 2, 16, 16)) {
      return false;
    }

    // decode_block() stores MiSize for the mode/partition contexts.
    for (int y = 0; y < 8; ++y) {
      left_mi(0, r + y) = 3;  // Mi_Height_Log2[BLOCK_32X32]
      above_mi(0, c + y) = 3;  // Mi_Width_Log2[BLOCK_32X32]
    }
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        intra_joint(r + y, c + x) = 0;  // DC_PRED
      }
    }
    return true;
  }

  bool encode_block_32x64(int r, int c) {
    if (!encode_mode_info_dc(r, c, 8, 16)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_32X64 luma, TX_16X32 chroma.
    if (!encode_residual_plane(0, r, c, 32, 64, 4, 32, 64)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 16, 32, 3, 16, 32)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 16, 32, 3, 16, 32)) {
      return false;
    }
    for (int y = 0; y < 16; ++y) {
      left_mi(0, r + y) = 4;  // Mi_Height_Log2[BLOCK_32X64]
    }
    for (int x = 0; x < 8; ++x) {
      above_mi(0, c + x) = 3;  // Mi_Width_Log2[BLOCK_32X64]
    }
    for (int y = 0; y < 16; ++y) {
      for (int x = 0; x < 8; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  bool encode_block_64x32(int r, int c) {
    if (!encode_mode_info_dc(r, c, 16, 8)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_64X32 luma, TX_32X16 chroma.
    if (!encode_residual_plane(0, r, c, 64, 32, 4, 64, 32)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 32, 16, 3, 32, 16)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 32, 16, 3, 32, 16)) {
      return false;
    }
    for (int y = 0; y < 8; ++y) {
      left_mi(0, r + y) = 3;   // Mi_Height_Log2[BLOCK_64X32]
    }
    for (int x = 0; x < 16; ++x) {
      above_mi(0, c + x) = 4;  // Mi_Width_Log2[BLOCK_64X32]
    }
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 16; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  bool encode_block_64x64(int r, int c) {
    if (!encode_mode_info_dc(r, c, 16, 16)) {
      return false;
    }
    // TX_MODE_LARGEST -> TX_64X64 luma, TX_32X32 chroma.
    if (!encode_residual_plane(0, r, c, 64, 64, 4, 64, 64)) {
      return false;
    }
    if (!encode_residual_plane(1, r, c, 32, 32, 3, 32, 32)) {
      return false;
    }
    if (!encode_residual_plane(2, r, c, 32, 32, 3, 32, 32)) {
      return false;
    }
    for (int y = 0; y < 16; ++y) {
      left_mi(0, r + y) = 4;   // Mi_Height_Log2[BLOCK_64X64]
      above_mi(0, c + y) = 4;  // Mi_Width_Log2[BLOCK_64X64]
    }
    for (int y = 0; y < 16; ++y) {
      for (int x = 0; x < 16; ++x) {
        intra_joint(r + y, c + x) = 0;
      }
    }
    return true;
  }

  std::uint8_t source_sample(int plane, int x, int y) const {
    const std::span<const std::uint8_t> data =
        plane == 0 ? picture_.y : (plane == 1 ? picture_.u : picture_.v);
    const int stride = plane == 0 ? static_cast<int>(picture_.width)
                                  : static_cast<int>(chroma_width_);
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }

  bool write_literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

  // Encoder inverse of read_quant() (§5.20.7.28) for allowTcq = 0 and
  // isHidden = 0. `x` is the value added to the pre-escape level; for a
  // DC-only block there is at most one escape, so hr_level_avg_ starts at 0.
  bool write_quant_escape(int x) {
    const int pred_level = hr_level_avg_;
    const int m = clip3(1, 6, pred_level <= 0 ? 0 : floor_log2(pred_level));
    const int k = m + 1;
    const int c_max = std::min(m + 4, 6);
    if (x < (c_max << m)) {
      const int q = x >> m;
      for (int i = 0; i < q; ++i) {
        if (!raw_bit(false)) {
          return false;
        }
      }
      if (!raw_bit(true)) {
        return false;
      }
      if (!write_literal(m, static_cast<std::uint32_t>(x & ((1 << m) - 1)))) {
        return false;
      }
    } else {
      for (int i = 0; i < c_max; ++i) {
        if (!raw_bit(false)) {
          return false;
        }
      }
      int g = 0;
      while (true) {
        const int length = k + g;
        const int x_base = (c_max << m) + (1 << length) - (1 << k);
        if (x < x_base + (1 << length)) {
          for (int i = 0; i < g; ++i) {
            if (!raw_bit(false)) {
              return false;
            }
          }
          if (!raw_bit(true)) {
            return false;
          }
          if (!write_literal(length, static_cast<std::uint32_t>(x - x_base))) {
            return false;
          }
          break;
        }
        ++g;
      }
    }
    hr_level_avg_ = (x + hr_level_avg_) >> 1;
    return true;
  }

  // Orthonormal 2D DCT-II, computed through the normative integer kernel:
  // Dct_Kernel{n}[j][i] = 64 * sqrt(n) * B[j][i].
  void forward_dct(const std::vector<int>& block, int n,
                   std::vector<double>* out) const {
    const std::int16_t* kernel =
        n == 32 ? &kDct32[0][0] : &kDct16[0][0];
    const double scale = 1.0 / (64.0 * std::sqrt(static_cast<double>(n)));
    std::vector<double> tmp(static_cast<std::size_t>(n) * n, 0.0);
    out->assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int v = 0; v < n; ++v) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
          sum += static_cast<double>(kernel[v * n + j]) *
                 static_cast<double>(block[static_cast<std::size_t>(i) * n + j]);
        }
        tmp[static_cast<std::size_t>(i) * n + v] = sum;
      }
    }
    for (int u = 0; u < n; ++u) {
      for (int v = 0; v < n; ++v) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
          sum += static_cast<double>(kernel[u * n + i]) *
                 tmp[static_cast<std::size_t>(i) * n + v];
        }
        (*out)[static_cast<std::size_t>(u) * n + v] = sum * scale * scale;
      }
    }
  }

  // Normative 2D inverse matrix transform (§7.15.4.1) for TX_32X32/TX_16X16.
  void inverse_dct(const std::vector<int>& dequant, int n,
                   std::vector<int>* residual) const {
    const std::int16_t* kernel =
        n == 32 ? &kDct32[0][0] : &kDct16[0][0];
    std::vector<int> intermediate(static_cast<std::size_t>(n) * n, 0);
    for (int i = 0; i < n; ++i) {
      for (int oj = 0; oj < n; ++oj) {
        std::int64_t sum = 0;
        for (int j = 0; j < n; ++j) {
          sum += static_cast<std::int64_t>(kernel[j * n + oj]) *
                 dequant[static_cast<std::size_t>(i) * n + j];
        }
        intermediate[static_cast<std::size_t>(i) * n + oj] = clip3(
            -(1 << 15), (1 << 15) - 1,
            round2(static_cast<int>(sum), 6));
      }
    }
    residual->assign(static_cast<std::size_t>(n) * n, 0);
    for (int oi = 0; oi < n; ++oi) {
      for (int j = 0; j < n; ++j) {
        std::int64_t sum = 0;
        for (int i = 0; i < n; ++i) {
          sum += static_cast<std::int64_t>(kernel[i * n + oi]) *
                 intermediate[static_cast<std::size_t>(i) * n + j];
        }
        (*residual)[static_cast<std::size_t>(oi) * n + j] =
            clip3(-256, 255, round2(static_cast<int>(sum), 13));
      }
    }
  }

  bool write_eob(int eob, int plane, int n) {
    int eob_pt = 1;
    if (eob == 2) {
      eob_pt = 2;
    } else if (eob >= 3) {
      eob_pt = floor_log2(eob - 1) + 2;
    }
    const bool is_1024 = n == 32;
    const int max_extra_bits = is_1024 ? 2 : 1;
    int symbol = 0;
    int escape = 0;
    if (eob_pt <= (is_1024 ? 7 : 7)) {
      symbol = eob_pt - 1;
    } else {
      symbol = 7;
      escape = eob_pt - 8;
    }
    if (!sym(is_1024 ? &cdfs_.eob_pt_1024[plane > 0 ? 2 : 0][0]
                     : &cdfs_.eob_pt_256[2][0],
             8, symbol)) {
      return false;
    }
    if (symbol == 7) {
      if (!write_literal(max_extra_bits, static_cast<std::uint32_t>(escape))) {
        return false;
      }
    }
    if (eob_pt >= 3) {
      const int base = (1 << (eob_pt - 2)) + 1;
      const int extra = eob - base;
      const int eob_extra = (extra >> (eob_pt - 3)) & 1;
      if (!sym(&cdfs_.eob_extra[0], 2, eob_extra)) {
        return false;
      }
      for (int i = eob_pt - 4; i >= 0; --i) {
        if (!raw_bit(((extra >> i) & 1) != 0)) {
          return false;
        }
      }
    }
    return true;
  }

  int dc_sign_context(int x4, int y4, int w4, int h4) const {
    int dc_sign = 0;
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < mi_cols_) {
        const int sign = above_dc_[0][static_cast<std::size_t>(x4 + k)];
        if (sign == 1) {
          --dc_sign;
        } else if (sign == 2) {
          ++dc_sign;
        }
      }
    }
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < mi_rows_) {
        const int sign = left_dc_[0][static_cast<std::size_t>(y4 + k)];
        if (sign == 1) {
          --dc_sign;
        } else if (sign == 2) {
          ++dc_sign;
        }
      }
    }
    if (dc_sign < 0) {
      return 1;
    }
    if (dc_sign > 0) {
      return 2;
    }
    return 0;
  }

  bool encode_residual_plane(int plane, int mi_row, int mi_col, int width,
                             int height, int tx_sz_ctx, int bsize_w,
                             int bsize_h) {
    const int w4 = width >> 2;
    const int h4 = height >> 2;
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int x4 = (mi_col >> sub_x);
    const int y4 = (mi_row >> sub_y);

    int ctx = 0;
    if (plane == 0) {
      int top = 0;
      int left = 0;
      for (int k = 0; k < w4; ++k) {
        if (x4 + k < mi_cols_) {
          top |= above_level(0, x4 + k);
        }
      }
      for (int k = 0; k < h4; ++k) {
        if (y4 + k < mi_rows_) {
          left |= left_level(0, y4 + k);
        }
      }
      top = std::min(top, 4);
      left = std::min(left, 4);
      // With TX_MODE_LARGEST the residual block always equals the transform
      // block for M0 luma, so the `bw == w && bh == h` branch wins.
      if (bsize_w == width && bsize_h == height) {
        ctx = 0;
      } else {
        ctx = (top + left + 3) >> 1;
      }
    } else {
      int above = 0;
      int left = 0;
      for (int i = 0; i < w4; ++i) {
        if (x4 + i < mi_cols_ / 2) {
          above |= above_level(plane, x4 + i);
          above |= above_dc(plane, x4 + i);
        }
      }
      for (int i = 0; i < h4; ++i) {
        if (y4 + i < mi_rows_ / 2) {
          left |= left_level(plane, y4 + i);
          left |= left_dc(plane, y4 + i);
        }
      }
      ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
      if (plane == 2) {
        if (bsize_w * bsize_h > width * height) {
          ctx += 3;
        }
        if (eob_u_ != 0) {
          ctx += 6;
        }
      } else {
        ctx += 6;
      }
    }

    hr_level_avg_ = 0;
    std::uint16_t* skip_cdf = nullptr;
    if (plane == 2) {
      skip_cdf = &cdfs_.v_txb_skip[ctx][0];
    } else {
      skip_cdf = &cdfs_.txb_skip[0][tx_sz_ctx][ctx][0];
    }

    if (!real_coeffs_) {
      // Diagnostic partition modes: zero residual, output is DC prediction.
      if (!sym(skip_cdf, 2, 1)) {
        return false;
      }
      for (int i = 0; i < w4; ++i) {
        above_level(plane, x4 + i) = 0;
        above_dc(plane, x4 + i) = 0;
      }
      for (int i = 0; i < h4; ++i) {
        left_level(plane, y4 + i) = 0;
        left_dc(plane, y4 + i) = 0;
      }
      if (plane == 1) {
        eob_u_ = 0;
      }
      return true;
    }

    // Prediction (DC_PRED, §7.13.2.10) from already reconstructed samples.
    int x0 = 0;
    int y0 = 0;
    if (plane == 0) {
      x0 = mi_col * 4;
      y0 = mi_row * 4;
    } else {
      x0 = (mi_col >> 1) * 4;
      y0 = (mi_row >> 1) * 4;
    }
    const bool have_left = is_inside(mi_row, mi_col - 1);
    const bool have_above = is_inside(mi_row - 1, mi_col);
    int sum_left = 0;
    int sum_above = 0;
    if (have_left) {
      for (int i = 0; i < height; ++i) {
        sum_left += recon_[plane].at(x0 - 1, y0 + i);
      }
    }
    if (have_above) {
      for (int j = 0; j < width; ++j) {
        sum_above += recon_[plane].at(x0 + j, y0 - 1);
      }
    }
    int pred_value = 128;
    if (have_left && have_above) {
      pred_value = clip1(approx_divide(sum_left + sum_above, width + height));
    } else if (have_left) {
      pred_value = round2(sum_left, floor_log2(height));
    } else if (have_above) {
      pred_value = round2(sum_above, floor_log2(width));
    }

    constexpr int kQuantStep = 528;
    constexpr int kQuantTableBits = 3;  // QUANT_TABLE_BITS
    const int dq_denom = plane == 0 ? 2 : 1;
    const int n = width;

    std::vector<int> residual_block(static_cast<std::size_t>(n) * n, 0);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        residual_block[static_cast<std::size_t>(j) * n + i] =
            static_cast<int>(source_sample(plane, x0 + i, y0 + j)) - pred_value;
      }
    }
    std::vector<double> coeff;
    forward_dct(residual_block, n, &coeff);
    std::vector<int> quantized(static_cast<std::size_t>(n) * n, 0);
    for (int pos = 0; pos < n * n; ++pos) {
      quantized[static_cast<std::size_t>(pos)] = static_cast<int>(
          std::lround(coeff[static_cast<std::size_t>(pos)] * 64.0 /
                      static_cast<double>(kQuantStep)));
    }
    const std::uint16_t* scan = n == 32 ? kScan32 : kScan16;
    int eob = 0;
    for (int c = n * n - 1; c >= 0; --c) {
      if (quantized[scan[c]] != 0) {
        eob = c + 1;
        break;
      }
    }

    if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) {
      return false;
    }
    if (eob == 0) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          recon_[plane].at(x0 + i, y0 + j) = static_cast<std::uint8_t>(pred_value);
        }
      }
      for (int i = 0; i < w4; ++i) {
        above_level(plane, x4 + i) = 0;
        above_dc(plane, x4 + i) = 0;
      }
      for (int i = 0; i < h4; ++i) {
        left_level(plane, y4 + i) = 0;
        left_dc(plane, y4 + i) = 0;
      }
      if (plane == 1) {
        eob_u_ = 0;
      }
      return true;
    }

    if (!write_eob(eob, plane, n)) {
      return false;
    }
    // transform_type() is luma-only. For chroma, compute_tx_type() derives
    // DCT_DCT from UVMode = DC_PRED and codes no intra_tx_type symbol.

    // Sig_Ref_Diff_Offset[TX_CLASS_2D] and
    // Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D] (§8.3.2).
    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    std::vector<int> level_pre(static_cast<std::size_t>(n) * n, 0);
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[c];
      const int row = pos / n;
      const int col = pos % n;
      const int abs_value = std::abs(quantized[pos]);
      const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
      const int base_levels = is_lf ? 4 : 2;
      const int base_max = is_lf ? 5 : 3;
      const int br_max = (is_lf && plane > 0) ? 0 : 3;
      const int pre_max = base_max + br_max;
      int base_level = 0;
      int br_level = 0;
      int escape_x = 0;
      if (abs_value <= base_levels) {
        base_level = abs_value;
      } else {
        base_level = std::min(base_levels + 1, base_max);
        const int pre = std::min(abs_value, pre_max);
        br_level = pre - base_level;
        escape_x = abs_value - pre;
      }

      int ctx_eob = 0;
      if (c != 0) {
        const int area = n << floor_log2(n);
        if (c <= area / 8) {
          ctx_eob = 1;
        } else if (c <= area / 4) {
          ctx_eob = 2;
        } else {
          ctx_eob = 3;
        }
      }

      if (c == eob - 1) {
        const int symbol = base_level - 1;
        if (plane == 0) {
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_eob[tx_sz_ctx][ctx_eob][0], 5,
                     symbol)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob[tx_sz_ctx][ctx_eob][0], 3, symbol)) {
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
          if (rr < n && cc < n) {
            const int limit = is_lf ? 5 : 3;
            mag += std::min(level_pre[static_cast<std::size_t>(rr) * n + cc],
                            limit);
          }
        }
        const int ctx = (mag + 1) >> 1;
        if (plane == 0) {
          if (is_lf) {
            int lf_ctx = 0;
            if (c == 0) {
              lf_ctx = std::min(ctx, 8);
            } else if (row + col < 2) {
              lf_ctx = std::min(ctx, 6) + 9;
            } else {
              lf_ctx = std::min(ctx, 4) + 16;
            }
            if (!sym(&cdfs_.coeff_base_lf[tx_sz_ctx][lf_ctx][0][0], 6,
                     base_level)) {
              return false;
            }
          } else {
            const int ctx2 = std::min(ctx, 4);
            int hf_ctx = ctx2;
            if (row + col >= 6 && row + col < 8) {
              hf_ctx = ctx2 + 5;
            } else if (row + col >= 8) {
              hf_ctx = ctx2 + 10;
            }
            if (!sym(&cdfs_.coeff_base[tx_sz_ctx][hf_ctx][0][0], 4, base_level)) {
              return false;
            }
          }
        } else {
          const int ctx2 = std::min(ctx, 3);
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
          if (rr < n && cc < n) {
            mag += std::min(level_pre[static_cast<std::size_t>(rr) * n + cc],
                            5);
          }
        }
        mag = std::min((mag + 1) >> 1, 6);
        if (plane == 0) {
          int br_ctx = mag;
          if (is_lf) {
            if (pos != 0) {
              br_ctx = mag + 7;
            }
            if (!sym(&cdfs_.coeff_br_lf[br_ctx][0], 4, br_level)) {
              return false;
            }
          } else {
            if (!sym(&cdfs_.coeff_br[mag][0], 4, br_level)) {
              return false;
            }
          }
        } else {
          if (!sym(&cdfs_.coeff_br_uv[std::min(mag, 3)][0], 4, br_level)) {
            return false;
          }
        }
      }
      level_pre[static_cast<std::size_t>(pos)] = pre_level;
      (void)escape_x;
    }

    // Sign and read_quant() pass, in reverse scan order (§5.20.7.27).
    int cul_level = 0;
    int dc_category = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[c];
      const int row = pos / n;
      const int col = pos % n;
      const int pre_level = level_pre[static_cast<std::size_t>(pos)];
      const int abs_value = std::abs(quantized[pos]);
      const bool negative = quantized[pos] < 0;
      if (pre_level != 0) {
        if (row == 0 && col == 0 && plane == 0) {
          const int sign_ctx = dc_sign_context(x4, y4, w4, h4);
          if (!sym(&cdfs_.dc_sign[0][0][sign_ctx][0], 2, negative ? 1 : 0)) {
            return false;
          }
        } else {
          if (!raw_bit(negative)) {
            return false;
          }
        }
        const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
        const int max_level = is_lf ? (plane == 0 ? 8 : 5) : 6;
        if (abs_value >= max_level) {
          if (!write_quant_escape(abs_value - pre_level)) {
            return false;
          }
        }
      }
      if (pos == 0 && abs_value > 0) {
        dc_category = negative ? 1 : 2;
      }
      cul_level = std::min(4, cul_level + abs_value);
    }

    // Dequant + normative 2D inverse transform, then add the prediction.
    std::vector<int> dequant(static_cast<std::size_t>(n) * n, 0);
    for (int pos = 0; pos < n * n; ++pos) {
      const int qc = quantized[pos];
      if (qc == 0) {
        continue;
      }
      const int dq_high = (std::abs(qc) * kQuantStep) & 0xFFFFFF;
      const int dq = round2(dq_high, kQuantTableBits);
      const int value = dq / dq_denom;
      dequant[static_cast<std::size_t>(pos)] = qc < 0 ? -value : value;
    }
    std::vector<int> residual;
    inverse_dct(dequant, n, &residual);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        recon_[plane].at(x0 + i, y0 + j) = static_cast<std::uint8_t>(
            clip1(pred_value +
                  residual[static_cast<std::size_t>(j) * n + i]));
      }
    }

    for (int i = 0; i < w4; ++i) {
      above_level(plane, x4 + i) = static_cast<std::uint8_t>(cul_level);
      above_dc(plane, x4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    for (int i = 0; i < h4; ++i) {
      left_level(plane, y4 + i) = static_cast<std::uint8_t>(cul_level);
      left_dc(plane, y4 + i) = static_cast<std::uint8_t>(dc_category);
    }
    if (plane == 1) {
      eob_u_ = 1;
    }
    return true;
  }

  const StillPicture& picture_;
  const StillPictureConfig& config_;
  M0PartitionMode partition_mode_ = M0PartitionMode::k32x32;
  bool real_coeffs_ = false;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  std::uint32_t chroma_width_ = 0;
  std::uint32_t chroma_height_ = 0;

  CdfState cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_buffer_;
  std::string error_;

  Plane recon_[3];

  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::array<std::vector<std::uint8_t>, 3> above_dc_;
  std::array<std::vector<std::uint8_t>, 3> left_dc_;
  std::array<std::vector<std::uint8_t>, 2> above_mi_;
  std::array<std::vector<std::uint8_t>, 2> left_mi_;
  std::vector<std::uint8_t> intra_joint_;
  int eob_u_ = 0;
  int hr_level_avg_ = 0;
};

}  // namespace

bool encode_still_picture(const StillPicture& picture,
                          const StillPictureConfig& config,
                          StillEncodeResult& result,
                          std::string& error,
                          M0PartitionMode partition_mode) {
  error.clear();
  if (picture.width == 0 || picture.height == 0) {
    error = "empty picture";
    return false;
  }
  if ((picture.width & 63U) != 0U || (picture.height & 63U) != 0U) {
    error = "M0 requires dimensions that are multiples of 64";
    return false;
  }
  if (picture.width != config.max_frame_width ||
      picture.height != config.max_frame_height) {
    error = "StillPictureConfig frame size does not match the picture";
    return false;
  }
  if (config.base_q_idx != kPinnedBaseQ) {
    error = "M0 pins base_q_idx = 90 (coefficient CDF tables are qctx slice 0)";
    return false;
  }
  const std::size_t luma_size =
      static_cast<std::size_t>(picture.width) * static_cast<std::size_t>(picture.height);
  const std::size_t chroma_size =
      static_cast<std::size_t>((picture.width + 1U) / 2U) *
      static_cast<std::size_t>((picture.height + 1U) / 2U);
  if (picture.y.size() < luma_size || picture.u.size() < chroma_size ||
      picture.v.size() < chroma_size) {
    error = "picture planes are too small";
    return false;
  }

  TileEncoder encoder(picture, config, partition_mode);
  std::vector<std::uint8_t> tile_payload;
  if (!encoder.encode(&tile_payload, error)) {
    return false;
  }

  result.frame = assemble_m0_still_picture(config, tile_payload);
  result.tile_bytes = static_cast<std::uint32_t>(tile_payload.size());
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.recon_y = encoder.recon(0).samples;
  result.recon_u = encoder.recon(1).samples;
  result.recon_v = encoder.recon(2).samples;
  return true;
}

}  // namespace av2enc::m0
