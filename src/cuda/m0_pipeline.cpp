// Host orchestration for the CUDA M0 still-picture path.
//
// Responsibilities:
//   1. Reproduce the exact M0 partition tree from src/encoder/m0_still.cpp and
//      record the partition-syntax operations in coding order.
//   2. Hand the resulting raster block plan to the GPU for DC prediction,
//      residual generation, forward DCT, quantisation, dequantisation, inverse
//      DCT and reconstruction.
//   3. Run the §5.20 entropy syntax writer on the CPU over the GPU-produced
//      (prediction, quantised coefficient) pairs, then assemble the OBU frame.
//
// The syntax writer below is a host-side port of TileEncoder::encode_* in
// src/encoder/m0_still.cpp, restricted to what the GPU job returns.  It keeps
// the exact CDF order, context derivation and coefficient passes so the
// bitstream is byte-identical to the CPU reference.
//
// TODO(components): when the shared entropy component (src/cuda/entropy.*)
// exposes a host syntax layer, replace HostEntropyWriter with it.  Until then
// this is the only place that knows §5.20 syntax; all transform/prediction
// arithmetic is delegated to the shared components in m0_pipeline.cu.

#include "cuda/m0_pipeline.h"

#include "encoder/m0_cdf_tables.h"
#include "encoder/m0_tables.h"
#include "entropy/av2/av2_entropy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::cuda {
namespace {

// ---------------------------------------------------------------------------
// M0 block-size constants and partition tables (mirrored from m0_still.cpp).
// ---------------------------------------------------------------------------

constexpr int kBlock8x16 = 4;
constexpr int kBlock16x8 = 5;
constexpr int kBlock16x32 = 7;
constexpr int kBlock16x4 = 20;
constexpr int kBlock32x8 = 22;
constexpr int kBlock32x16 = 8;
constexpr int kBlock32x32 = 9;
constexpr int kBlock32x64 = 10;
constexpr int kBlock64x32 = 11;
constexpr int kBlock64x64 = 12;

constexpr int kRectHorz = 0;
constexpr int kRectVert = 1;

constexpr int kSizeAdjust[11] = {2, 3, 3, 3, 2, 1, 1, 1, 1, 13, 11};
constexpr int kRectSizeAdjust[11] = {3, 4, 6, 5, 2, 1, 2, 1, 0, 14, 14};
constexpr int kMiLog2W[11] = {3, 3, 4, 4, 3, 2, 2, 1, 2, 3, 2};
constexpr int kMiLog2H[11] = {3, 4, 4, 3, 2, 3, 1, 2, 2, 1, 0};

constexpr std::size_t kTileCapacityBytes = 4U * 1024U * 1024U;

constexpr int kSizeIndex(int block_size) {
  switch (block_size) {
    case kBlock32x32:
      return 0;
    case kBlock32x64:
      return 1;
    case kBlock64x32:
      return 3;
    case kBlock32x16:
      return 4;
    case kBlock16x32:
      return 5;
    case kBlock16x8:
      return 6;
    case kBlock8x16:
      return 7;
    case 6:  // BLOCK_16X16
      return 8;
    case kBlock32x8:
      return 9;
    case kBlock16x4:
      return 10;
    default:
      return 2;
  }
}

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

// ---------------------------------------------------------------------------
// Partition plan
// ---------------------------------------------------------------------------

enum class OpKind : std::uint8_t {
  kClearAbove = 0,
  kClearLeft = 1,
  kDoSplit = 2,
  kRectType = 3,
  kBlock = 4,
};

struct Op {
  OpKind kind = OpKind::kBlock;
  int ctx = 0;
  int symbol = 0;
  int block = 0;
};

class PartitionPlanner {
 public:
  PartitionPlanner(int mi_cols, int mi_rows, m0::M0PartitionMode mode)
      : mi_cols_(mi_cols),
        mi_rows_(mi_rows),
        mode_(mode),
        real_coeffs_(mode == m0::M0PartitionMode::k32x32 ||
                     mode == m0::M0PartitionMode::k32x32HorzFirst) {
    for (int plane = 0; plane < 2; ++plane) {
      above_mi_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 6);
      left_mi_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 6);
    }
  }

  void build() {
    push(OpKind::kClearAbove, 0, 0, 0);
    clear_above();
    for (int r = 0; r < mi_rows_; r += 16) {
      push(OpKind::kClearLeft, 0, 0, 0);
      clear_left();
      for (int c = 0; c < mi_cols_; c += 16) {
        const bool edge_sb = (r + 16 > mi_rows_) || (c + 16 > mi_cols_);
        if (mode_ == m0::M0PartitionMode::k32x32 && edge_sb) {
          partition_none_64(r, c);
        } else {
          switch (mode_) {
            case m0::M0PartitionMode::k64x64Single:
              partition_none_64(r, c);
              break;
            case m0::M0PartitionMode::k64x64VertNone:
              partition_64_rect(r, c, kRectVert, false);
              break;
            case m0::M0PartitionMode::k64x64HorzNone:
              partition_64_rect(r, c, kRectHorz, false);
              break;
            case m0::M0PartitionMode::k32x32HorzFirst:
              partition_64_rect(r, c, kRectHorz, true);
              break;
            case m0::M0PartitionMode::k32x32:
            default:
              partition_64_rect(r, c, kRectVert, true);
              break;
          }
        }
      }
    }
  }

  const std::vector<Op>& ops() const { return ops_; }
  const std::vector<detail::BlockDesc>& blocks() const { return blocks_; }
  std::size_t total_coeffs() const { return total_coeffs_; }
  int real_blocks() const { return real_blocks_; }

 private:
  void push(OpKind kind, int ctx, int symbol, int block) {
    Op op;
    op.kind = kind;
    op.ctx = ctx;
    op.symbol = symbol;
    op.block = block;
    ops_.push_back(op);
  }

  void clear_above() {
    for (int plane = 0; plane < 2; ++plane) {
      std::fill(above_mi_[plane].begin(), above_mi_[plane].end(), 6);
    }
  }

  void clear_left() {
    for (int plane = 0; plane < 2; ++plane) {
      std::fill(left_mi_[plane].begin(), left_mi_[plane].end(), 6);
    }
  }

  std::uint8_t left_mi(int r) const {
    return left_mi_[0][static_cast<std::size_t>(r)];
  }
  std::uint8_t above_mi(int c) const {
    return above_mi_[0][static_cast<std::size_t>(c)];
  }

  int do_split_ctx(int idx, int r, int c) const {
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    return kSizeAdjust[idx] * 4 + (left_mi(r) < bsh ? 2 : 0) +
           (above_mi(c) < bsw ? 1 : 0);
  }

  int rect_ctx(int idx, int r, int c) const {
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    return kRectSizeAdjust[idx] * 4 + (left_mi(r) < bsh ? 2 : 0) +
           (above_mi(c) < bsw ? 1 : 0);
  }

  // §5.20.3.2 partition_implied() for the square/rectangular sizes M0 uses.
  bool partition_implied(int r, int c, int block_size, int* implied) const {
    const int idx = kSizeIndex(block_size);
    const int w4 = 1 << kMiLog2W[idx];
    const int h4 = 1 << kMiLog2H[idx];
    const bool has_rows = (r + (h4 >> 1)) < mi_rows_;
    const bool has_cols = (c + (w4 >> 1)) < mi_cols_;
    if (has_rows && has_cols) {
      *implied = 0;
      return false;
    }
    int part = 0;
    if (w4 == h4) {
      part = has_rows ? kRectVert : kRectHorz;
    } else if (h4 > w4) {
      if (!has_rows) {
        part = kRectHorz;
      } else {
        const bool sub_has_cols = (c + (w4 >> 2)) < mi_cols_;
        if (w4 >= 4 && !sub_has_cols) {
          part = kRectHorz;
        }
      }
    } else {
      if (!has_cols) {
        part = kRectVert;
      } else {
        const bool sub_has_rows = (r + (h4 >> 2)) < mi_rows_;
        if (h4 >= 4 && !sub_has_rows) {
          part = kRectVert;
        }
      }
    }
    *implied = part;
    return part != 0;
  }

  detail::BlockDesc make_block(int r, int c, int luma_w, int luma_h,
                               int chroma_w, int chroma_h, int upscale,
                               int luma_ctx, int chroma_ctx, int w4, int h4,
                               int mi_w_log2, int mi_h_log2) {
    detail::BlockDesc b;
    b.mi_row = r;
    b.mi_col = c;
    b.luma_w = luma_w;
    b.luma_h = luma_h;
    b.upscale = upscale;
    b.tx_w = upscale == 2 ? luma_w / 2 : luma_w;
    b.tx_h = upscale == 2 ? luma_h / 2 : luma_h;
    b.chroma_w = chroma_w;
    b.chroma_h = chroma_h;
    b.luma_pels = b.tx_w * b.tx_h;
    b.chroma_pels = chroma_w * chroma_h;
    b.luma_tx_ctx = luma_ctx;
    b.chroma_tx_ctx = chroma_ctx;
    b.real_coeffs = real_coeffs_ ? 1 : 0;
    b.coeff_offset = static_cast<int>(total_coeffs_);
    total_coeffs_ += static_cast<std::size_t>(b.luma_pels) +
                     2U * static_cast<std::size_t>(b.chroma_pels);
    if (b.real_coeffs != 0) {
      ++real_blocks_;
    }

    // decode_block() stores MiSize for the mode/partition contexts.
    for (int y = 0; y < h4; ++y) {
      left_mi_[0][static_cast<std::size_t>(r + y)] =
          static_cast<std::uint8_t>(mi_h_log2);
    }
    for (int x = 0; x < w4; ++x) {
      above_mi_[0][static_cast<std::size_t>(c + x)] =
          static_cast<std::uint8_t>(mi_w_log2);
    }
    return b;
  }

  void add_block(int r, int c, int luma_w, int luma_h, int chroma_w,
                 int chroma_h, int upscale, int luma_ctx, int chroma_ctx,
                 int w4, int h4, int mi_w_log2, int mi_h_log2) {
    blocks_.push_back(make_block(r, c, luma_w, luma_h, chroma_w, chroma_h,
                                 upscale, luma_ctx, chroma_ctx, w4, h4,
                                 mi_w_log2, mi_h_log2));
    push(OpKind::kBlock, 0, 0, static_cast<int>(blocks_.size()) - 1);
  }

  void partition_none_64(int r, int c) {
    const int idx = kSizeIndex(kBlock64x64);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 0, 0);
    add_block(r, c, 64, 64, 32, 32, 2, 4, 3, 16, 16, 4, 4);
  }

  void partition_64_rect(int r, int c, int rect_type, bool split_children) {
    const int idx = kSizeIndex(kBlock64x64);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 1, 0);
    push(OpKind::kRectType, rect_ctx(idx, r, c), rect_type, 0);
    if (rect_type == kRectVert) {
      if (split_children) {
        partition_32x64(r, c);
        partition_32x64(r, c + 8);
      } else {
        partition_none_32x64(r, c);
        partition_none_32x64(r, c + 8);
      }
    } else {
      if (split_children) {
        partition_64x32(r, c);
        partition_64x32(r + 8, c);
      } else {
        partition_none_64x32(r, c);
        partition_none_64x32(r + 8, c);
      }
    }
  }

  void partition_none_32x64(int r, int c) {
    const int idx = kSizeIndex(kBlock32x64);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 0, 0);
    add_block(r, c, 32, 64, 16, 32, 1, 4, 3, 8, 16, 3, 4);
  }

  void partition_none_64x32(int r, int c) {
    const int idx = kSizeIndex(kBlock64x32);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 0, 0);
    add_block(r, c, 64, 32, 32, 16, 1, 4, 3, 16, 8, 4, 3);
  }

  void partition_64x32(int r, int c) {
    const int idx = kSizeIndex(kBlock64x32);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 1, 0);
    push(OpKind::kRectType, rect_ctx(idx, r, c), kRectVert, 0);
    partition_32x32(r, c);
    partition_32x32(r, c + 8);
  }

  void partition_32x64(int r, int c) {
    const int idx = kSizeIndex(kBlock32x64);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 1, 0);
    push(OpKind::kRectType, rect_ctx(idx, r, c), kRectHorz, 0);
    partition_32x32(r, c);
    partition_32x32(r + 8, c);
  }

  void partition_32x32(int r, int c) {
    int implied = 0;
    if (partition_implied(r, c, kBlock32x32, &implied)) {
      if (implied == kRectHorz) {
        if (r < mi_rows_) {
          partition_32x16(r, c);
        }
        if (r + 4 < mi_rows_) {
          partition_32x16(r + 4, c);
        }
        return;
      }
      if (c < mi_cols_) {
        partition_16x32(r, c);
      }
      if (c + 4 < mi_cols_) {
        partition_16x32(r, c + 4);
      }
      return;
    }
    const int idx = kSizeIndex(kBlock32x32);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 0, 0);
    add_block(r, c, 32, 32, 16, 16, 1, 3, 2, 8, 8, 3, 3);
  }

  void partition_32x16(int r, int c) {
    const int idx = kSizeIndex(kBlock32x16);
    const int bsh = kMiLog2H[idx];
    const int bsw = kMiLog2W[idx];
    const int w4 = 1 << bsw;
    const int h4 = 1 << bsh;
    const bool has_rows = (r + (h4 >> 1)) < mi_rows_;
    const bool has_cols = (c + (w4 >> 1)) < mi_cols_;
    if (!has_rows && has_cols) {
      push(OpKind::kRectType, rect_ctx(idx, r, c), kRectHorz, 0);
      if (r < mi_rows_) {
        partition_32x8(r, c);
      }
      if (r + 2 < mi_rows_) {
        partition_32x8(r + 2, c);
      }
      return;
    }
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 0, 0);
    add_block(r, c, 32, 16, 16, 8, 1, 3, 2, 8, 4, 3, 2);
  }

  void partition_32x8(int r, int c) {
    const int idx = kSizeIndex(kBlock32x8);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 0, 0);
    add_block(r, c, 32, 8, 16, 4, 1, 2, 1, 8, 2, 3, 1);
  }

  void partition_16x32(int r, int c) {
    const int idx = kSizeIndex(kBlock16x32);
    push(OpKind::kDoSplit, do_split_ctx(idx, r, c), 0, 0);
    add_block(r, c, 16, 32, 8, 16, 1, 3, 2, 4, 8, 2, 3);
  }

  int mi_cols_ = 0;
  int mi_rows_ = 0;
  m0::M0PartitionMode mode_ = m0::M0PartitionMode::k32x32;
  bool real_coeffs_ = false;
  std::array<std::vector<std::uint8_t>, 2> above_mi_;
  std::array<std::vector<std::uint8_t>, 2> left_mi_;
  std::vector<Op> ops_;
  std::vector<detail::BlockDesc> blocks_;
  std::size_t total_coeffs_ = 0;
  int real_blocks_ = 0;
};

// ---------------------------------------------------------------------------
// §5.20 syntax writer (CPU side)
// ---------------------------------------------------------------------------

struct CdfState {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
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

void init_cdfs(CdfState& cdfs) {
  copy_array(cdfs.do_split, m0::kDoSplitCdf);
  copy_array(cdfs.rect_type, m0::kRectTypeCdf);
  copy_array(cdfs.y_mode_set, m0::kYModeSetCdf);
  copy_array(cdfs.y_mode_index, m0::kYModeIndexCdf);
  copy_array(cdfs.uv_mode, m0::kUvModeCflNotAllowedCdf);
  copy_array(cdfs.txb_skip, m0::kTxbSkipCdf);
  copy_array(cdfs.v_txb_skip, m0::kVTxbSkipCdf);
  copy_array(cdfs.eob_pt_1024, m0::kEobPt1024Cdf);
  copy_array(cdfs.eob_pt_512, m0::kEobPt512Cdf);
  copy_array(cdfs.eob_pt_256, m0::kEobPt256Cdf);
  copy_array(cdfs.eob_pt_128, m0::kEobPt128Cdf);
  copy_array(cdfs.eob_pt_64, m0::kEobPt64Cdf);
  copy_array(cdfs.eob_extra, m0::kEobExtraCdf);
  copy_array(cdfs.coeff_base, m0::kCoeffBaseCdf);
  copy_array(cdfs.coeff_base_lf, m0::kCoeffBaseLfCdf);
  copy_array(cdfs.coeff_base_eob, m0::kCoeffBaseEobCdf);
  copy_array(cdfs.coeff_base_lf_eob, m0::kCoeffBaseLfEobCdf);
  copy_array(cdfs.coeff_br, m0::kCoeffBrCdf);
  copy_array(cdfs.coeff_br_lf, m0::kCoeffBrLfCdf);
  copy_array(cdfs.coeff_base_uv, m0::kCoeffBaseUvCdf);
  copy_array(cdfs.coeff_base_lf_uv, m0::kCoeffBaseLfUvCdf);
  copy_array(cdfs.coeff_base_eob_uv, m0::kCoeffBaseEobUvCdf);
  copy_array(cdfs.coeff_base_lf_eob_uv, m0::kCoeffBaseLfEobUvCdf);
  copy_array(cdfs.coeff_br_uv, m0::kCoeffBrUvCdf);
  copy_array(cdfs.dc_sign, m0::kDcSignCdf);
  copy_array(cdfs.intra_tx_type_set1, m0::kIntraTxTypeSet1Cdf);
}

class HostEntropyWriter {
 public:
  HostEntropyWriter(const m0::StillPictureConfig& config, int mi_cols,
                    int mi_rows)
      : mi_cols_(mi_cols), mi_rows_(mi_rows) {
    init_cdfs(cdfs_);
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = config.disable_cdf_update ? 1 : 0;
    for (int plane = 0; plane < 3; ++plane) {
      above_level_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 0);
      left_level_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 0);
      above_dc_[plane].assign(static_cast<std::size_t>(mi_cols_) + 8U, 0);
      left_dc_[plane].assign(static_cast<std::size_t>(mi_rows_) + 8U, 0);
    }
  }

  bool write(const std::vector<Op>& ops,
             const std::vector<detail::BlockDesc>& blocks,
             const std::int32_t* preds, const std::int32_t* coeffs,
             std::vector<std::uint8_t>* payload, std::string& error) {
    for (const Op& op : ops) {
      switch (op.kind) {
        case OpKind::kClearAbove:
          clear_above_context();
          break;
        case OpKind::kClearLeft:
          clear_left_context();
          break;
        case OpKind::kDoSplit:
          if (!sym(&cdfs_.do_split[0][op.ctx][0], 2, op.symbol)) {
            return fail(error);
          }
          break;
        case OpKind::kRectType:
          if (!sym(&cdfs_.rect_type[0][op.ctx][0], 2, op.symbol)) {
            return fail(error);
          }
          break;
        case OpKind::kBlock:
          if (!encode_block(op.block, blocks, preds, coeffs)) {
            return fail(error);
          }
          break;
      }
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      error = std::string("entropy exit_symbol failed: ") +
              av2::status_name(status);
      return false;
    }
    payload->assign(tile_buffer_.begin(),
                    tile_buffer_.begin() +
                        static_cast<std::ptrdiff_t>(payload_bytes));
    return true;
  }

 private:
  bool fail(std::string& error) {
    error = error_;
    return false;
  }

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

  void clear_above_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(above_level_[plane].begin(), above_level_[plane].end(), 0);
      std::fill(above_dc_[plane].begin(), above_dc_[plane].end(), 0);
    }
  }

  void clear_left_context() {
    for (int plane = 0; plane < 3; ++plane) {
      std::fill(left_level_[plane].begin(), left_level_[plane].end(), 0);
      std::fill(left_dc_[plane].begin(), left_dc_[plane].end(), 0);
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

  bool write_literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

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
          if (!write_literal(length,
                             static_cast<std::uint32_t>(x - x_base))) {
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

  bool write_eob(int eob, int plane, int w, int h) {
    int eob_pt = 1;
    if (eob == 2) {
      eob_pt = 2;
    } else if (eob >= 3) {
      eob_pt = floor_log2(eob - 1) + 2;
    }
    const int multisize = std::min(5, floor_log2(w)) +
                          std::min(5, floor_log2(h)) - 4;
    std::uint16_t* cdf = nullptr;
    int escape_bits = 0;
    int cdf_n = 8;
    const int cdf_ctx = plane > 0 ? 2 : 0;
    switch (multisize) {
      case 2:
        cdf = &cdfs_.eob_pt_64[cdf_ctx][0];
        escape_bits = 0;
        cdf_n = 7;
        break;
      case 3:
        cdf = &cdfs_.eob_pt_128[cdf_ctx][0];
        escape_bits = 0;
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
    if (!sym(cdf, cdf_n, symbol)) {
      return false;
    }
    if (symbol == 7 && escape_bits > 0) {
      if (!write_literal(escape_bits, static_cast<std::uint32_t>(escape))) {
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

  const std::vector<std::uint16_t>& scan_for(int w, int h) {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(w)) << 32U) |
        static_cast<std::uint32_t>(h);
    auto it = scans_.find(key);
    if (it != scans_.end()) {
      return it->second;
    }
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(w) *
                                    static_cast<std::size_t>(h));
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
    return scans_.emplace(key, std::move(scan)).first->second;
  }

  bool encode_mode_info_dc() {
    // y_mode_set, y_mode_index[joint_ctx = 0], uv_mode.  Every M0 block is
    // DC_PRED and intra_joint[] stays zero, so the joint context is always 0.
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) {
      return false;
    }
    if (!sym(&cdfs_.y_mode_index[0][0], 8, 0)) {
      return false;
    }
    return sym(&cdfs_.uv_mode[0][0], 8, 0);
  }

  bool encode_block(int block_index,
                    const std::vector<detail::BlockDesc>& blocks,
                    const std::int32_t* preds, const std::int32_t* coeffs) {
    const detail::BlockDesc& b = blocks[static_cast<std::size_t>(block_index)];
    if (!encode_mode_info_dc()) {
      return false;
    }
    const std::int32_t* base = coeffs + b.coeff_offset;
    const std::int32_t* u = base + b.luma_pels;
    const std::int32_t* v = u + b.chroma_pels;
    const std::int32_t* pred = preds + static_cast<std::size_t>(block_index) * 3;
    if (!encode_residual_plane(0, b, pred[0], base, b.luma_w, b.luma_h, b.tx_w,
                               b.tx_h, b.luma_tx_ctx)) {
      return false;
    }
    if (!encode_residual_plane(1, b, pred[1], u, b.chroma_w, b.chroma_h,
                               b.chroma_w, b.chroma_h, b.chroma_tx_ctx)) {
      return false;
    }
    return encode_residual_plane(2, b, pred[2], v, b.chroma_w, b.chroma_h,
                                 b.chroma_w, b.chroma_h, b.chroma_tx_ctx);
  }

  bool encode_residual_plane(int plane, const detail::BlockDesc& b, int pred,
                             const std::int32_t* quant, int width, int height,
                             int tw, int th, int tx_sz_ctx) {
    const int w4 = width >> 2;
    const int h4 = height >> 2;
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int x4 = b.mi_col >> sub_x;
    const int y4 = b.mi_row >> sub_y;

    int ctx = 0;
    if (plane == 0) {
      // With TX_MODE_LARGEST the residual block equals the transform block for
      // M0 luma, so the CPU's `bsize_w == width && bsize_h == height` branch
      // always wins and ctx is 0.
      ctx = 0;
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

    if (b.real_coeffs == 0) {
      if (!sym(skip_cdf, 2, 1)) {
        return false;
      }
      clear_plane_levels(plane, x4, y4, w4, h4);
      if (plane == 1) {
        eob_u_ = 0;
      }
      return true;
    }

    const std::vector<std::uint16_t>& scan = scan_for(tw, th);
    const int area = tw * th;
    int eob = 0;
    for (int c = area - 1; c >= 0; --c) {
      if (quant[scan[static_cast<std::size_t>(c)]] != 0) {
        eob = c + 1;
        break;
      }
    }

    if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) {
      return false;
    }
    if (eob == 0) {
      clear_plane_levels(plane, x4, y4, w4, h4);
      if (plane == 1) {
        eob_u_ = 0;
      }
      return true;
    }

    if (!write_eob(eob, plane, tw, th)) {
      return false;
    }

    // Sig_Ref_Diff_Offset[TX_CLASS_2D] and
    // Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D] (§8.3.2).
    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    std::vector<int> level_pre(static_cast<std::size_t>(area), 0);
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / tw;
      const int col = pos % tw;
      const int abs_value = std::abs(static_cast<int>(quant[pos]));
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
            if (!sym(&cdfs_.coeff_base_eob[tx_sz_ctx][ctx_eob][0], 3,
                     symbol)) {
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
          if (rr < th && cc < tw) {
            const int limit = is_lf ? 5 : 3;
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * tw + cc], limit);
          }
        }
        const int mag_ctx = (mag + 1) >> 1;
        if (plane == 0) {
          if (is_lf) {
            int lf_ctx = 0;
            if (c == 0) {
              lf_ctx = std::min(mag_ctx, 8);
            } else if (row + col < 2) {
              lf_ctx = std::min(mag_ctx, 6) + 9;
            } else {
              lf_ctx = std::min(mag_ctx, 4) + 16;
            }
            if (!sym(&cdfs_.coeff_base_lf[tx_sz_ctx][lf_ctx][0][0], 6,
                     base_level)) {
              return false;
            }
          } else {
            const int ctx2 = std::min(mag_ctx, 4);
            int hf_ctx = ctx2;
            if (row + col >= 6 && row + col < 8) {
              hf_ctx = ctx2 + 5;
            } else if (row + col >= 8) {
              hf_ctx = ctx2 + 10;
            }
            if (!sym(&cdfs_.coeff_base[tx_sz_ctx][hf_ctx][0][0], 4,
                     base_level)) {
              return false;
            }
          }
        } else {
          const int ctx2 = std::min(mag_ctx, 3);
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
          if (rr < th && cc < tw) {
            mag += std::min(
                level_pre[static_cast<std::size_t>(rr) * tw + cc], 5);
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
    }

    // Sign and read_quant() pass, in reverse scan order (§5.20.7.27).
    int cul_level = 0;
    int dc_category = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / tw;
      const int col = pos % tw;
      const int pre_level = level_pre[static_cast<std::size_t>(pos)];
      const int abs_value = std::abs(static_cast<int>(quant[pos]));
      const bool negative = quant[pos] < 0;
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
    (void)pred;
    return true;
  }

  void clear_plane_levels(int plane, int x4, int y4, int w4, int h4) {
    for (int i = 0; i < w4; ++i) {
      above_level(plane, x4 + i) = 0;
      above_dc(plane, x4 + i) = 0;
    }
    for (int i = 0; i < h4; ++i) {
      left_level(plane, y4 + i) = 0;
      left_dc(plane, y4 + i) = 0;
    }
  }

  int mi_cols_ = 0;
  int mi_rows_ = 0;
  CdfState cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_buffer_;
  std::string error_;
  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::array<std::vector<std::uint8_t>, 3> above_dc_;
  std::array<std::vector<std::uint8_t>, 3> left_dc_;
  std::map<std::uint64_t, std::vector<std::uint16_t>> scans_;
  int eob_u_ = 0;
  int hr_level_avg_ = 0;
};

std::vector<std::uint8_t> crop_plane(const std::vector<std::uint8_t>& plane,
                                     int stride, int width, int height) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    std::copy_n(plane.begin() + static_cast<std::ptrdiff_t>(y) * stride, width,
                out.begin() + static_cast<std::ptrdiff_t>(y) * width);
  }
  return out;
}

}  // namespace

bool encode_still_picture_cuda(const m0::StillPicture& picture,
                               const m0::StillPictureConfig& config,
                               m0::StillEncodeResult& result,
                               std::string& error,
                               m0::M0PartitionMode partition_mode,
                               const PipelineOptions& options,
                               PipelineTimings* timings) {
  const auto total_start = std::chrono::steady_clock::now();
  error.clear();
  result = {};
  if (picture.width == 0 || picture.height == 0) {
    error = "empty picture";
    return false;
  }
  if ((picture.width & 1U) != 0U || (picture.height & 1U) != 0U) {
    error = "M0 4:2:0 requires even dimensions";
    return false;
  }
  if (picture.width != config.max_frame_width ||
      picture.height != config.max_frame_height) {
    error = "StillPictureConfig frame size does not match the picture";
    return false;
  }
  if (config.base_q_idx != 90) {
    error = "M0 pins base_q_idx = 90 (coefficient CDF tables are qctx slice 0)";
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

  const int width = static_cast<int>(picture.width);
  const int height = static_cast<int>(picture.height);
  const int chroma_width = (width + 1) / 2;
  const int chroma_height = (height + 1) / 2;
  const int mi_cols = 2 * ((width + 7) >> 3);
  const int mi_rows = 2 * ((height + 7) >> 3);
  const int padded_width = (width + 63) & ~63;
  const int padded_height = (height + 63) & ~63;
  const int chroma_padded_width = (padded_width + 1) / 2;
  const int chroma_padded_height = (padded_height + 1) / 2;

  PartitionPlanner planner(mi_cols, mi_rows, partition_mode);
  planner.build();
  const std::vector<Op>& ops = planner.ops();
  const std::vector<detail::BlockDesc>& blocks = planner.blocks();
  const std::size_t total_coeffs = planner.total_coeffs();

  if (options.verbose) {
    std::fprintf(stderr,
                 "m0_pipeline: %dx%d, %zu blocks (%d real), %zu coeffs, "
                 "%zu syntax ops\n",
                 width, height, blocks.size(), planner.real_blocks(),
                 total_coeffs, ops.size());
  }

  // Padded reconstruction planes, initialised to 128 like m0_still.cpp.
  std::vector<std::uint8_t> recon_y(
      static_cast<std::size_t>(padded_width) * padded_height, 128);
  std::vector<std::uint8_t> recon_u(
      static_cast<std::size_t>(chroma_padded_width) * chroma_padded_height,
      128);
  std::vector<std::uint8_t> recon_v = recon_u;

  std::vector<std::int32_t> coeffs(total_coeffs, 0);
  std::vector<std::int32_t> preds(blocks.size() * 3, 0);

  detail::GpuTimings gpu_timings;
  if (planner.real_blocks() > 0) {
    detail::GpuJob job;
    job.y = picture.y.data();
    job.u = picture.u.data();
    job.v = picture.v.data();
    job.width = width;
    job.height = height;
    job.chroma_width = chroma_width;
    job.chroma_height = chroma_height;
    job.padded_width = padded_width;
    job.padded_height = padded_height;
    job.chroma_padded_width = chroma_padded_width;
    job.chroma_padded_height = chroma_padded_height;
    job.blocks = blocks.data();
    job.block_count = static_cast<int>(blocks.size());
    job.mi_cols = mi_cols;
    job.mi_rows = mi_rows;
    job.recon_y = recon_y.data();
    job.recon_u = recon_u.data();
    job.recon_v = recon_v.data();
    job.coeffs = coeffs.data();
    job.preds = preds.data();
    if (!detail::run_gpu_job(job, options.device, error, &gpu_timings)) {
      return false;
    }
  }

  const auto entropy_start = std::chrono::steady_clock::now();
  HostEntropyWriter writer(config, mi_cols, mi_rows);
  std::vector<std::uint8_t> tile_payload;
  if (!writer.write(ops, blocks, preds.data(), coeffs.data(), &tile_payload,
                    error)) {
    return false;
  }
  const auto entropy_end = std::chrono::steady_clock::now();

  result.frame = m0::assemble_m0_still_picture(config, tile_payload);
  result.tile_bytes = static_cast<std::uint32_t>(tile_payload.size());
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.recon_y = crop_plane(recon_y, padded_width, width, height);
  result.recon_u = crop_plane(recon_u, chroma_padded_width, chroma_width,
                              chroma_height);
  result.recon_v = crop_plane(recon_v, chroma_padded_width, chroma_width,
                              chroma_height);

  if (timings != nullptr) {
    timings->upload_ms = gpu_timings.upload_ms;
    timings->kernel_ms = gpu_timings.kernel_ms;
    timings->download_ms = gpu_timings.download_ms;
    timings->entropy_ms =
        std::chrono::duration<double, std::milli>(entropy_end - entropy_start)
            .count();
    timings->total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start)
            .count();
    timings->block_count = static_cast<int>(blocks.size());
    timings->real_blocks = planner.real_blocks();
    timings->kernel_launches = gpu_timings.launches;
  }
  return true;
}

}  // namespace av2enc::cuda
