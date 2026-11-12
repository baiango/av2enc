// Entropy + OBU assembly for the standalone 10-bit (M10) still picture.
//
// The symbol layout is the M0 syntax subset that dav2d already accepts, with
// the bit-depth-dependent header fields owned by `high_bitdepth_headers`:
//
//   §5.20.3.2  partition syntax (VERT/HORZ only; NONE at partial 64x64 SBs)
//   §5.20.5    DC intra mode info
//   §5.20.7    TX_MODE_LARGEST residual, EOB and coefficient coding
//   §8.2       normative symbol coder (`src/entropy/av2`)
//   §8.3.1     coefficient CDF slice from the *raw* base_q_idx
//
// No M1 decision code is used.  This writer only consumes the block/level
// stream produced by the CPU reference or the GPU block stage.
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "formats/format.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_headers.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/entropy/av2/av2_entropy.h"
#include "codec/entropy/qctx/qctx.h"

namespace av2enc::high_bitdepth {
namespace {

constexpr int kBlock32 = 32;
constexpr int kBlock64 = 64;
constexpr std::size_t kTileCapacityBytes = 16U * 1024U * 1024U;

// §5.20.3.2 context tables for the three sizes the M10 tree reaches.
struct SizeInfo {
  int size_adjust;
  int rect_adjust;
  int w_log2;
  int h_log2;
};
SizeInfo size_info(int luma_size) {
  if (luma_size == 64) return SizeInfo{3, 6, 4, 4};   // 64x64
  if (luma_size == 32) return SizeInfo{2, 3, 3, 3};   // 32x32
  return SizeInfo{3, 4, 3, 4};                        // 32x64 / 64x32
}

// TX_SIZE indices used by the M10 tree are derived per plane from the coded
// transform dimensions, so extra chroma formats do not need their own list.

int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §5.20.7.27 txSzCtx from the nominal transform dimensions.
int tx_size_context_from_dims(int w, int h) {
  const int sqr = std::max(0, floor_log2(w) - 2);
  const int sqr_up = std::max(0, floor_log2(h) - 2);
  const int lo = std::min(sqr, sqr_up);
  const int hi = std::max(sqr, sqr_up);
  return (lo + hi + 1) >> 1;
}

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

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
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

// qctx is the §8.3.1 coefficient-CDF slice (0..3), selected from the *raw*
// frame base_q_idx by the caller.  Slice 0 is byte-identical to the qctx-0
// tables the M10 still path has always used, so q <= 90 streams do not move.
void init_cdfs(CdfState& c, int qctx) {
  using namespace av2enc::basic;
  copy_array(c.do_split, kDoSplitCdf);
  copy_array(c.rect_type, kRectTypeCdf);
  copy_array(c.y_mode_set, kYModeSetCdf);
  copy_array(c.y_mode_index, kYModeIndexCdf);
  copy_array(c.uv_mode, kUvModeCflNotAllowedCdf);

  const av2enc::still::qctx::CoeffCdfSlice& q =
      av2enc::still::qctx::tables_for_slice(qctx);
  copy_array(c.txb_skip, *q.txb_skip);
  copy_array(c.v_txb_skip, *q.v_txb_skip);
  copy_array(c.eob_pt_1024, *q.eob_pt_1024);
  copy_array(c.eob_pt_512, *q.eob_pt_512);
  copy_array(c.eob_pt_256, *q.eob_pt_256);
  copy_array(c.eob_pt_128, *q.eob_pt_128);
  copy_array(c.eob_pt_64, *q.eob_pt_64);
  copy_array(c.eob_extra, *q.eob_extra);
  copy_array(c.coeff_base, *q.coeff_base);
  copy_array(c.coeff_base_lf, *q.coeff_base_lf);
  copy_array(c.coeff_base_eob, *q.coeff_base_eob);
  copy_array(c.coeff_base_lf_eob, *q.coeff_base_lf_eob);
  copy_array(c.coeff_br, *q.coeff_br);
  copy_array(c.coeff_br_lf, *q.coeff_br_lf);
  copy_array(c.coeff_base_uv, *q.coeff_base_uv);
  copy_array(c.coeff_base_lf_uv, *q.coeff_base_lf_uv);
  copy_array(c.coeff_base_eob_uv, *q.coeff_base_eob_uv);
  copy_array(c.coeff_base_lf_eob_uv, *q.coeff_base_lf_eob_uv);
  copy_array(c.coeff_br_uv, *q.coeff_br_uv);
  copy_array(c.dc_sign, *q.dc_sign);
}

class Writer {
 public:
  Writer(const fmt::FrameGeometry& geometry, const HighBitDepthBlock* blocks,
         std::size_t block_count, const std::int32_t* levels,
         std::size_t level_count, const HighBitDepthOptions& options)
      : blocks_(blocks),
        block_count_(block_count),
        levels_(levels),
        level_count_(level_count),
        options_(options) {
    mi_cols_ = static_cast<int>(geometry.mi_cols);
    mi_rows_ = static_cast<int>(geometry.mi_rows);
    sub_x_ = geometry.sub_x;
    sub_y_ = geometry.sub_y;
    monochrome_ = !fmt::has_chroma(geometry.format.chroma);
    const int qctx = options_.qctx >= 0
                         ? options_.qctx
                         : fmt::cdf_q_context(options_.base_q_idx);
    init_cdfs(cdfs_, qctx);
    tile_buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(tile_buffer_.data(), tile_buffer_.size());
    enc_.disable_cdf_update = options_.disable_cdf_update ? 1 : 0;

    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16U, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16U, 6);
    for (int plane = 0; plane < 3; ++plane) {
      const int shift_x = plane == 0 ? 0 : sub_x_;
      const int shift_y = plane == 0 ? 0 : sub_y_;
      const int n = (mi_cols_ >> shift_x) + 16;
      const int m = (mi_rows_ >> shift_y) + 16;
      above_level_[plane].assign(static_cast<std::size_t>(n), 0);
      left_level_[plane].assign(static_cast<std::size_t>(m), 0);
      above_dc_[plane].assign(static_cast<std::size_t>(n), 0);
      left_dc_[plane].assign(static_cast<std::size_t>(m), 0);
    }
    clear_above_context();
  }

  bool run(std::uint32_t* tile_bytes, std::string* error) {
    std::size_t block_index = 0;

    for (int r = 0; r < mi_rows_; r += 16) {
      clear_left_context();
      for (int c = 0; c < mi_cols_; c += 16) {
        const bool edge = (r + 16 > mi_rows_) || (c + 16 > mi_cols_);
        if (!edge) {
          if (!code_partition_64_vert(r, c, error)) return false;
          // Decode order: left 32x64 (top, bottom), then right 32x64.
          for (int cc = c; cc < c + 16; cc += 8) {
            if (!code_partition_32x64(r, cc, error)) return false;
            for (int rr = r; rr < r + 16; rr += 8) {
              if (block_index >= block_count_) {
                return fail(error, "block stream ended early");
              }
              const HighBitDepthBlock& block = blocks_[block_index++];
              if (block.mi_row != rr || block.mi_col != cc ||
                  block.luma_size != kBlock32) {
                return fail(error, "block stream does not follow the 32x32 tree");
              }
              // Every 32x32 leaf codes do_split = 0 in the M0-proven tree.
              if (!code_do_split(rr, cc, kBlock32, false)) {
                return fail(error, error_);
              }
              if (!write_block(block, error)) return false;
            }
          }
        } else {
          if (!code_partition_64_none(r, c, error)) return false;
          if (block_index >= block_count_) {
            return fail(error, "block stream ended early");
          }
          const HighBitDepthBlock& block = blocks_[block_index++];
          if (block.mi_row != r || block.mi_col != c ||
              block.luma_size != kBlock64) {
            return fail(error, "edge superblock is not a 64x64 leaf");
          }
          if (!write_block(block, error)) return false;
        }
      }
    }

    if (block_index != block_count_) {
      return fail(error, "block stream has trailing blocks");
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("entropy exit_symbol failed: ") +
                             av2::status_name(status));
    }
    payload_.assign(tile_buffer_.begin(),
                    tile_buffer_.begin() +
                        static_cast<std::ptrdiff_t>(payload_bytes));
    if (tile_bytes != nullptr) *tile_bytes = static_cast<std::uint32_t>(payload_bytes);
    return true;
  }

  const std::vector<std::uint8_t>& payload() const { return payload_; }

 private:
  static bool fail(std::string* error, const std::string& text) {
    if (error != nullptr) *error = text;
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
  bool write_literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

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
  std::uint8_t& above_mi(int x) { return above_mi_[static_cast<std::size_t>(x)]; }
  std::uint8_t& left_mi(int y) { return left_mi_[static_cast<std::size_t>(y)]; }

  // -------------------------------------------------------------------
  // Partition syntax
  // -------------------------------------------------------------------
  bool code_do_split(int r, int c, int luma_size, bool split) {
    const SizeInfo info = size_info(luma_size);
    const int ctx = info.size_adjust * 4 +
                    (left_mi(r) < info.h_log2 ? 2 : 0) +
                    (above_mi(c) < info.w_log2 ? 1 : 0);
    return sym(&cdfs_.do_split[0][ctx][0], 2, split ? 1 : 0);
  }
  bool code_rect_type(int r, int c, int luma_size, bool vertical) {
    const SizeInfo info = size_info(luma_size);
    const int ctx = info.rect_adjust * 4 +
                    (left_mi(r) < info.h_log2 ? 2 : 0) +
                    (above_mi(c) < info.w_log2 ? 1 : 0);
    return sym(&cdfs_.rect_type[0][ctx][0], 2, vertical ? 1 : 0);
  }

  bool code_partition_64_vert(int r, int c, std::string* error) {
    if (!code_do_split(r, c, kBlock64, true) ||
        !code_rect_type(r, c, kBlock64, true)) {
      return fail(error, error_);
    }
    return true;
  }
  bool code_partition_64_none(int r, int c, std::string* error) {
    if (!code_do_split(r, c, kBlock64, false)) return fail(error, error_);
    return true;
  }
  bool code_partition_32x64(int r, int c, std::string* error) {
    if (!code_do_split(r, c, 0 /*32x64 marker*/, true)) {
      return fail(error, error_);
    }
    // 32x64 is not square: use the rectangular 32x64 SizeInfo.
    const SizeInfo info = size_info(0);
    const int ctx = info.rect_adjust * 4 +
                    (left_mi(r) < info.h_log2 ? 2 : 0) +
                    (above_mi(c) < info.w_log2 ? 1 : 0);
    if (!sym(&cdfs_.rect_type[0][ctx][0], 2, 0 /*RECT_HORZ*/)) {
      return fail(error, error_);
    }
    return true;
  }

  // -------------------------------------------------------------------
  // Mode info (DC only)
  // -------------------------------------------------------------------
  bool code_mode_info_dc() {
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) return false;
    // Every block is DC, so both neighbour joint modes are 0 -> context 0.
    if (!sym(&cdfs_.y_mode_index[0][0], 8, 0)) return false;
    if (monochrome_) return true;  // §5.20.5.3: no intra_uv_mode symbol
    return sym(&cdfs_.uv_mode[0][0], 8, 0);
  }

  // -------------------------------------------------------------------
  // Residual syntax
  // -------------------------------------------------------------------
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
    int cdf_n = 8;
    int escape_bits = 0;
    const int cdf_ctx = plane > 0 ? 2 : (plane < 0 ? 1 : 0);
    switch (multisize) {
      case 2:
        cdf = &cdfs_.eob_pt_64[cdf_ctx][0];
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
    if (eob_pt <= cdf_n) {
      symbol = eob_pt - 1;
    } else {
      symbol = cdf_n - 1;
      escape = eob_pt - cdf_n;
    }
    if (!sym(cdf, cdf_n, symbol)) return false;
    if (symbol == cdf_n - 1 && escape_bits > 0) {
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

  bool write_quant_escape(int x) {
    const int pred_level = hr_level_avg_;
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
    hr_level_avg_ = (x + hr_level_avg_) >> 1;
    return true;
  }

  int dc_sign_context(int x4, int y4, int w4, int h4) const {
    int dc_sign = 0;
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < mi_cols_) {
        const int sign = above_dc_[0][static_cast<std::size_t>(x4 + k)];
        if (sign == 1) --dc_sign;
        if (sign == 2) ++dc_sign;
      }
    }
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < mi_rows_) {
        const int sign = left_dc_[0][static_cast<std::size_t>(y4 + k)];
        if (sign == 1) --dc_sign;
        if (sign == 2) ++dc_sign;
      }
    }
    return dc_sign < 0 ? 1 : (dc_sign > 0 ? 2 : 0);
  }

  bool encode_residual_plane(const HighBitDepthBlock& block, int plane, int coded_w,
                             int coded_h, int tx_sz_ctx, int bsize_w,
                             int bsize_h, std::size_t level_base,
                             std::string* error) {
    const int w4 = coded_w >> 2;
    const int h4 = coded_h >> 2;
    const int sx = plane > 0 ? sub_x_ : 0;
    const int sy = plane > 0 ? sub_y_ : 0;
    const int x4 = (block.mi_col >> sx);
    const int y4 = (block.mi_row >> sy);
    const int max_x4 = mi_cols_ >> sx;
    const int max_y4 = mi_rows_ >> sy;

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
      ctx = (bsize_w == coded_w && bsize_h == coded_h) ? 0
                                                       : ((top + left + 3) >> 1);
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
        if (bsize_w * bsize_h > coded_w * coded_h) ctx += 3;
        if (eob_u_ != 0) ctx += 6;
      } else {
        ctx += 6;
      }
    }

    hr_level_avg_ = 0;
    std::uint16_t* skip_cdf =
        plane == 2 ? &cdfs_.v_txb_skip[ctx][0]
                   : &cdfs_.txb_skip[0][tx_sz_ctx][ctx][0];
    const int eob = static_cast<int>(block.eob[plane]);
    if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) return fail(error, error_);
    if (eob == 0) {
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

    const auto params_opt = txv::make_tx_params(
        static_cast<txv::TxSize>(block.tx_size[plane]), txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      return fail(error, "invalid transform size");
    }
    const txv::TxParams& params = *params_opt;
    const int tw = params.w;
    const int th = params.h;
    const int area = tw * th;
    if (eob < 0 || eob > area) {
      return fail(error, "EOB outside the transform area");
    }
    if (level_base + static_cast<std::size_t>(eob) > level_count_) {
      return fail(error, "coefficient levels overflow");
    }
    if (!write_eob(eob, plane, params.coded_w, params.coded_h)) {
      return fail(error, error_);
    }

    static constexpr int kSigRef[5][2] = {
        {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
    static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    hbd::build_scan(tw, th, scan.data());
    std::vector<int> level_pre(static_cast<std::size_t>(area), 0);
    const auto level_at = [&](int c) {
      return static_cast<int>(
          levels_[level_base + static_cast<std::size_t>(c)]);
    };

    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / tw;
      const int col = pos % tw;
      const int abs_value = std::abs(level_at(c));
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
              return fail(error, error_);
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob[tx_sz_ctx][ctx_eob][0], 3,
                     symbol)) {
              return fail(error, error_);
            }
          }
        } else {
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_eob_uv[ctx_eob][0], 5, symbol)) {
              return fail(error, error_);
            }
          } else {
            if (!sym(&cdfs_.coeff_base_eob_uv[ctx_eob][0], 3, symbol)) {
              return fail(error, error_);
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
        const int mctx = (mag + 1) >> 1;
        if (plane == 0) {
          if (is_lf) {
            int lf_ctx = 0;
            if (c == 0) {
              lf_ctx = std::min(mctx, 8);
            } else if (row + col < 2) {
              lf_ctx = std::min(mctx, 6) + 9;
            } else {
              lf_ctx = std::min(mctx, 4) + 16;
            }
            if (!sym(&cdfs_.coeff_base_lf[tx_sz_ctx][lf_ctx][0][0], 6,
                     base_level)) {
              return fail(error, error_);
            }
          } else {
            const int ctx2 = std::min(mctx, 4);
            int hf_ctx = ctx2;
            if (row + col >= 6 && row + col < 8) {
              hf_ctx = ctx2 + 5;
            } else if (row + col >= 8) {
              hf_ctx = ctx2 + 10;
            }
            if (!sym(&cdfs_.coeff_base[tx_sz_ctx][hf_ctx][0][0], 4,
                     base_level)) {
              return fail(error, error_);
            }
          }
        } else {
          const int ctx2 = std::min(mctx, 3);
          const int uv_ctx = plane == 1 ? ctx2 : ctx2 + 4;
          if (is_lf) {
            if (!sym(&cdfs_.coeff_base_lf_uv[uv_ctx][0], 6, base_level)) {
              return fail(error, error_);
            }
          } else {
            if (!sym(&cdfs_.coeff_base_uv[uv_ctx][0], 4, base_level)) {
              return fail(error, error_);
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
            if (pos != 0) br_ctx = mag + 7;
            if (!sym(&cdfs_.coeff_br_lf[br_ctx][0], 4, br_level)) {
              return fail(error, error_);
            }
          } else {
            if (!sym(&cdfs_.coeff_br[mag][0], 4, br_level)) {
              return fail(error, error_);
            }
          }
        } else {
          if (!sym(&cdfs_.coeff_br_uv[std::min(mag, 3)][0], 4, br_level)) {
            return fail(error, error_);
          }
        }
      }
      level_pre[static_cast<std::size_t>(pos)] = pre_level;
    }

    int cul_level = 0;
    int dc_category = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / tw;
      const int col = pos % tw;
      const int pre_level = level_pre[static_cast<std::size_t>(pos)];
      const int level = level_at(c);
      const int abs_value = std::abs(level);
      const bool negative = level < 0;
      if (pre_level != 0) {
        if (row == 0 && col == 0 && plane == 0) {
          const int sign_ctx = dc_sign_context(x4, y4, w4, h4);
          if (!sym(&cdfs_.dc_sign[0][0][sign_ctx][0], 2, negative ? 1 : 0)) {
            return fail(error, error_);
          }
        } else {
          if (!raw_bit(negative)) return fail(error, error_);
        }
        const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
        const int max_level = is_lf ? (plane == 0 ? 8 : 5) : 6;
        if (abs_value >= max_level) {
          if (!write_quant_escape(abs_value - pre_level)) {
            return fail(error, error_);
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

  bool write_block(const HighBitDepthBlock& block, std::string* error) {
    const int r = block.mi_row;
    const int c = block.mi_col;
    const int luma = block.luma_size;
    if (luma != kBlock32 && luma != kBlock64) {
      return fail(error, "unsupported M10 luma block size");
    }
    const int w = luma;
    const int h = luma;
    const int w4 = w >> 2;
    const int h4 = h >> 2;

    // TX_MODE_LARGEST: luma TX == coding block; chroma is subsampled by the
    // picture's chroma format (4:2:0 halves both, 4:2:2 half width, 4:4:4
    // identity, 4:0:0 has no plane).
    const int coded_w[3] = {w, w >> sub_x_, w >> sub_x_};
    const int coded_h[3] = {h, h >> sub_y_, h >> sub_y_};
    for (int plane = 0; plane < 3; ++plane) {
      if (plane > 0 && monochrome_) break;
      const int expected = txv::tx_size_index(coded_w[plane], coded_h[plane]);
      if (expected < 0 || block.tx_size[plane] != expected) {
        return fail(error, "TX_MODE_LARGEST block size mismatch");
      }
    }
    const int tx_ctx[3] = {
        tx_size_context_from_dims(coded_w[0], coded_h[0]),
        tx_size_context_from_dims(coded_w[1], coded_h[1]),
        tx_size_context_from_dims(coded_w[2], coded_h[2])};

    if (!code_mode_info_dc()) return fail(error, error_);

    std::size_t level_base = block.levels_offset;
    const int planes = monochrome_ ? 1 : 3;
    for (int plane = 0; plane < planes; ++plane) {
      if (!encode_residual_plane(block, plane, coded_w[plane],
                                 coded_h[plane], tx_ctx[plane],
                                 coded_w[plane], coded_h[plane], level_base,
                                 error)) {
        return false;
      }
      level_base += block.eob[plane];
    }

    // Partition/mode contexts for later blocks.
    for (int y = 0; y < h4; ++y) left_mi(r + y) = h4 > 8 ? 4 : 3;
    for (int x = 0; x < w4; ++x) above_mi(c + x) = w4 > 8 ? 4 : 3;
    return true;
  }

  const HighBitDepthBlock* blocks_ = nullptr;
  std::size_t block_count_ = 0;
  const std::int32_t* levels_ = nullptr;
  std::size_t level_count_ = 0;
  const HighBitDepthOptions& options_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int sub_x_ = 1;
  int sub_y_ = 1;
  bool monochrome_ = false;

  CdfState cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_buffer_;
  std::vector<std::uint8_t> payload_;
  std::string error_;

  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::array<std::vector<std::uint8_t>, 3> above_dc_;
  std::array<std::vector<std::uint8_t>, 3> left_dc_;
  std::vector<std::uint8_t> above_mi_;
  std::vector<std::uint8_t> left_mi_;
  int eob_u_ = 0;
  int hr_level_avg_ = 0;
};

}  // namespace

bool assemble_high_bitdepth_tile_payload(
    const fmt::Picture& picture, const HighBitDepthOptions& options,
    const HighBitDepthBlock* blocks, std::size_t block_count,
    const std::int32_t* levels, std::size_t level_count,
    std::vector<std::uint8_t>* payload, std::uint32_t* tile_bytes,
    std::string* error) {
  if (error != nullptr) error->clear();
  if (payload == nullptr || blocks == nullptr ||
      (levels == nullptr && level_count != 0)) {
    if (error != nullptr) *error = "null stream argument";
    return false;
  }
  payload->clear();
  if (picture.format.depth != fmt::BitDepth::k10) {
    if (error != nullptr) *error = "high bit depth requires 10-bit samples";
    return false;
  }
  if (fmt::chroma_format_idc(picture.format.chroma) < 0) {
    if (error != nullptr) *error = "invalid chroma format";
    return false;
  }
  if (picture.geometry.mi_cols == 0 || picture.geometry.mi_rows == 0) {
    if (error != nullptr) *error = "empty picture geometry";
    return false;
  }
  if (!high_bitdepth_geometry_supported(picture.format.width,
                                        picture.format.height, error)) {
    return false;
  }

  Writer writer(picture.geometry, blocks, block_count, levels, level_count,
                options);
  std::uint32_t payload_bytes = 0;
  if (!writer.run(&payload_bytes, error)) return false;
  *payload = writer.payload();
  if (tile_bytes != nullptr) *tile_bytes = payload_bytes;
  return true;
}

bool assemble_high_bitdepth_stream(const fmt::Picture& picture, const HighBitDepthOptions& options,
                         const HighBitDepthBlock* blocks, std::size_t block_count,
                         const std::int32_t* levels, std::size_t level_count,
                         std::vector<std::uint8_t>* out,
                         std::uint32_t* tile_bytes, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) {
    if (error != nullptr) *error = "null stream argument";
    return false;
  }
  out->clear();
  if (picture.format.chroma != fmt::ChromaFormat::k420) {
    if (error != nullptr) *error = "M10 header is 4:2:0 only; use hbd_headers";
    return false;
  }

  HighBitDepthConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.format.height);
  config.depth = fmt::BitDepth::k10;
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;
  std::string config_error;
  if (!validate_config(config, &config_error)) {
    if (error != nullptr) *error = config_error;
    return false;
  }

  std::vector<std::uint8_t> payload;
  std::uint32_t payload_bytes = 0;
  if (!assemble_high_bitdepth_tile_payload(picture, options, blocks, block_count,
                                           levels, level_count, &payload,
                                           &payload_bytes, error)) {
    return false;
  }
  *out = assemble_high_bitdepth_still_picture(config, payload);
  if (tile_bytes != nullptr) *tile_bytes = payload_bytes;
  return true;
}

}  // namespace av2enc::high_bitdepth
