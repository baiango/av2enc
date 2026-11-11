#include "encoder/palette/scc_stream.h"

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m0_cdf_tables.h"
#include "encoder/m1/m1_cdf_tables.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/partition/partition.h"
#include "encoder/palette/scc_cdf.h"
#include "encoder/tx_variants/tx_variants.h"
#include "entropy/av2/av2_entropy.h"
#include "io/ivf.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace av2enc::scc {
namespace {

using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

int mi_cols(const SccConfig& c) {
  return 2 * ((c.width + 7) / 8);
}
int mi_rows(const SccConfig& c) { return 2 * ((c.height + 7) / 8); }

// §5.18.1/§5.18.2 frame header, identical to M1's writer except for the two
// screen-content bits (and the IBC allow flags that follow them).
void write_frame_header(BitWriter& writer, const SccConfig& c) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  // screen_content_params(): single_picture_header_flag makes
  // seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS, so the
  // frame codes allow_screen_content_tools.  seq_force_integer_mv is also
  // SELECT_INTEGER_MV, so force_integer_mv is present when it is 1.
  writer.put_bit(c.allow_screen_content_tools);
  if (c.allow_screen_content_tools) {
    writer.put_bit(true);  // force_integer_mv (IBC/CRC use integer vectors)
  }

  // intrabc_params().
  writer.put_bit(c.allow_intrabc);
  if (c.allow_intrabc) {
    writer.put_bit(c.allow_global_intrabc);
    if (c.allow_global_intrabc) {
      writer.put_bit(c.allow_local_intrabc);
    }
    // max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1;
    // allow_frame_max_bvp_drl_bits == 0, so no change_bvp_drl syntax.
  }

  writer.put_bit(true);  // disable_cdf_update

  // tile_info(): one tile (uniform spacing, no increments).
  const int cols = (mi_cols(c) + 15) / 16;
  const int rows = (mi_rows(c) + 15) / 16;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (cols > 1) writer.put_bit(false);
  if (rows > 1) writer.put_bit(false);

  writer.put_bits(c.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);              // segmentation_enabled
  writer.put_bit(false);              // using_qmatrix
  writer.put_bit(false);              // delta_q_present
  // deblocking_filter_params().
  writer.put_bit(c.apply_deblock_y_vert);
  writer.put_bit(c.apply_deblock_y_horz);
  const bool have_chroma_deblock =
      c.apply_deblock_y_vert || c.apply_deblock_y_horz;
  if (have_chroma_deblock) {
    writer.put_bit(c.apply_deblock_u);
    writer.put_bit(c.apply_deblock_v);
  }
  if (c.apply_deblock_y_vert) writer.put_bit(false);
  if (c.apply_deblock_y_horz) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_u) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_v) writer.put_bit(false);
  writer.put_bit(false);  // tx_mode_select = 0 (TX_MODE_LARGEST)
  writer.put_bits(0U, 2U);  // reduced_tx_set
}

struct TileCdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t skip[6][3];
  std::uint16_t is_cfl[3][3];
  std::uint16_t intrabc[3][3];
  std::uint16_t intrabc_mode[3];
  palette::PaletteCdfState palette;
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(TileCdfs& c) {
  copy_array(c.do_split, av2enc::m0::kDoSplitCdf);
  copy_array(c.rect_type, av2enc::m0::kRectTypeCdf);
  copy_array(c.y_mode_set, av2enc::m1::kYModeSetCdf);
  copy_array(c.y_mode_index, av2enc::m1::kYModeIndexCdf);
  copy_array(c.uv_mode, av2enc::m1::kUvModeCflNotAllowedCdf);
  copy_array(c.txb_skip, av2enc::m0::kTxbSkipCdf);
  copy_array(c.v_txb_skip, av2enc::m0::kVTxbSkipCdf);
  copy_array(c.is_cfl, av2enc::m1::kIsCflCdf);
  copy_array(c.skip, scc::kSkipCdf);
  copy_array(c.intrabc, scc::kIntrabcCdf);
  copy_array(c.intrabc_mode, scc::kIntrabcModeCdf);
  palette::init_palette_cdfs(c.palette);
}

int tx_sz_ctx(int tx_size) {
  return (m1::kTxSizeSqr[tx_size] + m1::kTxSizeSqrUp[tx_size] + 1) >> 1;
}

class TileWriter {
 public:
  TileWriter(const SccFramePlan& plan, const SccConfig& config)
      : plan_(plan),
        config_(config),
        mi_cols_(mi_cols(config)),
        mi_rows_(mi_rows(config)) {
    init_cdfs(cdfs_);
    tile_.resize(kTileCapacity);
    enc_.init_symbol(tile_.data(), tile_.size());
    enc_.disable_cdf_update = 1;
    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    ibc_.assign(static_cast<std::size_t>(mi_cols_) *
                    static_cast<std::size_t>(mi_rows_), 0);
    skips_.assign(static_cast<std::size_t>(mi_cols_) *
                      static_cast<std::size_t>(mi_rows_), 0);
    for (int plane = 0; plane < 3; ++plane) {
      const int n = (plane == 0 ? mi_cols_ : mi_cols_ / 2) + 16;
      const int m = (plane == 0 ? mi_rows_ : mi_rows_ / 2) + 16;
      above_level_[static_cast<std::size_t>(plane)].assign(
          static_cast<std::size_t>(n), 0);
      left_level_[static_cast<std::size_t>(plane)].assign(
          static_cast<std::size_t>(m), 0);
    }
    palette_state_.reset(mi_cols_, mi_rows_);
    recon_y_.assign(static_cast<std::size_t>(config.width) *
                        static_cast<std::size_t>(config.height), 128);
    recon_u_.assign(static_cast<std::size_t>(config.width / 2) *
                        static_cast<std::size_t>(config.height / 2), 128);
    recon_v_.assign(recon_u_.size(), 128);
  }

  bool run(SccEncodeResult& out, std::string& error) {
    const int sb_cols = (mi_cols_ + 15) / 16;
    const int sb_rows = (mi_rows_ + 15) / 16;
    if (static_cast<int>(plan_.sbs.size()) != sb_cols * sb_rows) {
      error = "superblock plan count does not match the frame geometry";
      return false;
    }
    for (int sb_index = 0; sb_index < sb_cols * sb_rows; ++sb_index) {
      const int expect_row = (sb_index / sb_cols) * 16;
      const int expect_col = (sb_index % sb_cols) * 16;
      if (expect_col == 0) clear_left_context();
      const SbPlan& sb = plan_.sbs[static_cast<std::size_t>(sb_index)];
      if (!write_sb(sb, expect_row, expect_col, error)) return false;
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = enc_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      error = std::string("entropy exit_symbol failed: ") +
              av2::status_name(status);
      return false;
    }
    out.tile_bytes = payload_bytes;
    out.recon_y = recon_y_;
    out.recon_u = recon_u_;
    out.recon_v = recon_v_;
    out.frame = assemble(std::span<const std::uint8_t>(
        tile_.data(), static_cast<std::size_t>(payload_bytes)));
    return true;
  }

 private:
  static constexpr std::size_t kTileCapacity = 16U * 1024U * 1024U;

  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("encode_symbol failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

  bool raw_bit(bool bit) {
    const av2::Status status = enc_.encode_bool(bit);
    if (status != av2::Status::kOk) {
      error_ = std::string("encode_bool failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

  void clear_left_context() {
    std::fill(left_mi_.begin(), left_mi_.end(), 6);
    for (auto& level : left_level_) std::fill(level.begin(), level.end(), 0);
  }

  // §5.20.3.2 partition_implied_at_boundary for a 64x64 root.
  SbPlan::Partition implied_partition(int mi_row, int mi_col) const {
    const int half = 8;  // 64x64 has 16x16 MI; half is 8
    const bool has_rows = mi_row + half < mi_rows_;
    const bool has_cols = mi_col + half < mi_cols_;
    if (has_rows && has_cols) return SbPlan::Partition::kNone;
    return has_rows ? SbPlan::Partition::kVert : SbPlan::Partition::kHorz;
  }

  bool child_partition_implied(int bsize, int mi_row, int mi_col) const {
    const int half_w = 1 << (std::max(m1::kMiWidthLog2[bsize], 1) - 1);
    const int half_h = 1 << (std::max(m1::kMiHeightLog2[bsize], 1) - 1);
    return !((mi_row + half_h < mi_rows_) && (mi_col + half_w < mi_cols_));
  }

  bool write_sb(const SbPlan& sb, int mi_row, int mi_col, std::string& error) {
    const int bsize = 12;  // BLOCK_64X64 at the superblock root
    const int bsw = std::max(m1::kMiWidthLog2[bsize], 1);
    const int bsh = std::max(m1::kMiHeightLog2[bsize], 1);
    const SbPlan::Partition implied = implied_partition(mi_row, mi_col);
    const bool split = sb.partition != SbPlan::Partition::kNone;
    if (implied != SbPlan::Partition::kNone) {
      if (sb.partition != implied) {
        error = "superblock partition does not match the implied boundary cut";
        return false;
      }
    } else {
      const int ctx = m1::kPartitionSizeAdjust[bsize] * 4 +
                      (left_mi_[static_cast<std::size_t>(mi_row)] < bsh ? 2 : 0) +
                      (above_mi_[static_cast<std::size_t>(mi_col)] < bsw ? 1 : 0);
      if (!sym(&cdfs_.do_split[0][ctx][0], 2, split ? 1 : 0)) {
        error = error_;
        return false;
      }
    }
    if (split && implied == SbPlan::Partition::kNone) {
      const int rect_ctx =
          m1::kPartitionSizeAdjustRect[bsize] * 4 +
          (left_mi_[static_cast<std::size_t>(mi_row)] < bsh ? 2 : 0) +
          (above_mi_[static_cast<std::size_t>(mi_col)] < bsw ? 1 : 0);
      const int rect = sb.partition == SbPlan::Partition::kVert ? 1 : 0;
      if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, rect)) {
        error = error_;
        return false;
      }
    }
    std::size_t expected = 1U;
    if (split) {
      expected = 0;
      if (sb.partition == SbPlan::Partition::kVert) {
        expected = (mi_col + 0 < mi_cols_ ? 1U : 0U) +
                   (mi_col + 8 < mi_cols_ ? 1U : 0U);
      } else {
        expected = (mi_row + 0 < mi_rows_ ? 1U : 0U) +
                   (mi_row + 8 < mi_rows_ ? 1U : 0U);
      }
    }
    if (sb.leaves.size() != expected) {
      error = "superblock partition has the wrong number of leaves";
      return false;
    }
    const int child_bsize =
        !split ? 12 : (sb.partition == SbPlan::Partition::kVert ? 10 : 11);
    for (const BlockPlan& leaf : sb.leaves) {
      if (leaf.bsize != child_bsize) {
        error = "leaf block size does not match the partition subsize";
        return false;
      }
      if (split && !child_partition_implied(child_bsize, leaf.mi_row,
                                            leaf.mi_col)) {
        // decode_partition() codes the child's own do_split unless the
        // boundary implies a partition.  This gate writer only emits NONE
        // children.
        const int cbw = std::max(m1::kMiWidthLog2[child_bsize], 1);
        const int cbh = std::max(m1::kMiHeightLog2[child_bsize], 1);
        const int cctx =
            m1::kPartitionSizeAdjust[child_bsize] * 4 +
            (left_mi_[static_cast<std::size_t>(leaf.mi_row)] < cbh ? 2 : 0) +
            (above_mi_[static_cast<std::size_t>(leaf.mi_col)] < cbw ? 1 : 0);
        if (!sym(&cdfs_.do_split[0][cctx][0], 2, 0)) {
          error = error_;
          return false;
        }
      }
      if (!write_block(leaf, error)) return false;
    }
    return true;
  }

  bool write_block(const BlockPlan& block, std::string& error) {
    const auto bsize = static_cast<partition::BlockSize>(block.bsize);
    const int bw = partition::block_width(bsize);
    const int bh = partition::block_height(bsize);
    if (bw <= 0 || bh <= 0) {
      error = "invalid block size";
      return false;
    }
    const int x = block.mi_col * 4;
    const int y = block.mi_row * 4;
    if (x < 0 || y < 0 || x + bw > config_.width ||
        y + bh > config_.height) {
      error = "block is outside the frame";
      return false;
    }
    const bool ibc_allowed = config_.allow_intrabc && bw <= 64 && bh <= 64 &&
                             block.bsize != 12;

    // --- intra_frame_mode_info() ---
    if (ibc_allowed) {
      const int ibc_ctx =
          (block.mi_row > 0
               ? ibc_[static_cast<std::size_t>(block.mi_row - 1) *
                          static_cast<std::size_t>(mi_cols_) +
                      static_cast<std::size_t>(block.mi_col)]
               : 0) +
          (block.mi_col > 0
               ? ibc_[static_cast<std::size_t>(block.mi_row) *
                          static_cast<std::size_t>(mi_cols_) +
                      static_cast<std::size_t>(block.mi_col - 1)]
               : 0);
      const bool is_ibc = block.kind == BlockKind::kIbc;
      if (!sym(&cdfs_.intrabc[std::min(ibc_ctx, 2)][0], 2, is_ibc ? 1 : 0)) {
        error = error_;
        return false;
      }
      if (is_ibc) {
        if (!write_ibc_block(block, bw, bh, error)) return false;
      } else {
        if (!write_intra_modes(block, bw, bh, error)) return false;
      }
    } else {
      if (block.kind == BlockKind::kIbc) {
        error = "IBC block is not signallable for this size/config";
        return false;
      }
      if (!write_intra_modes(block, bw, bh, error)) return false;
    }

    // --- mark contexts and reconstruct ---
    const int w4 = bw / 4;
    const int h4 = bh / 4;
    for (int yy = 0; yy < h4; ++yy) {
      const int r = block.mi_row + yy;
      if (r >= mi_rows_) break;
      left_mi_[static_cast<std::size_t>(r)] = m1::kMiHeightLog2[block.bsize];
      for (int xx = 0; xx < w4; ++xx) {
        const int c = block.mi_col + xx;
        if (c >= mi_cols_) break;
        const std::size_t idx = static_cast<std::size_t>(r) *
                                    static_cast<std::size_t>(mi_cols_) +
                                static_cast<std::size_t>(c);
        ibc_[idx] = block.kind == BlockKind::kIbc ? 1 : 0;
        skips_[idx] = (block.kind == BlockKind::kIbc && block.skip) ? 1 : 0;
      }
    }
    for (int xx = 0; xx < w4; ++xx) {
      const int c = block.mi_col + xx;
      if (c >= mi_cols_) break;
      above_mi_[static_cast<std::size_t>(c)] = m1::kMiWidthLog2[block.bsize];
    }
    return true;
  }

  bool write_intra_modes(const BlockPlan& block, int bw, int bh,
                         std::string& error) {
    // read_intra_y_mode(): DC is mode index 0 in both the mode set and the
    // default list, so it is independent of neighbour joint modes.
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.y_mode_index[0][0], 8, 0)) {
      error = error_;
      return false;
    }
    // read_intra_uv_mode(): is_cfl is present only when the sequence header
    // enables CfL.  DC is uv_mode index 0 for a DC Y mode.
    if (config_.enable_cfl_intra && !sym(&cdfs_.is_cfl[0][0], 2, 0)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.uv_mode[0][0], 8, 0)) {
      error = error_;
      return false;
    }

    // palette_mode_info() for every non-IBC intra block when screen tools are
    // allowed (MiSize >= 8x8, w/h <= 64).
    const bool palette_allowed = config_.allow_screen_content_tools &&
                                 block.bsize >= 3 && bw <= 64 && bh <= 64;
    if (palette_allowed) {
      std::uint8_t cache[16] = {};
      const int cache_n = palette::get_palette_cache(
          palette_state_, block.mi_row, block.mi_col, block.mi_col > 0, cache);
      if (block.kind == BlockKind::kPalette) {
        if (!palette::encode_palette_y(enc_, cdfs_.palette, block.palette, bw,
                                       bh, bw, bh, cache, cache_n, error)) {
          return false;
        }
        palette::store_palette_state(palette_state_, block.mi_row,
                                     block.mi_col, bw, bh, block.palette);
      } else {
        if (!sym(&cdfs_.palette.y_mode[0], 2, 0)) {
          error = error_;
          return false;
        }
      }
    } else if (block.kind == BlockKind::kPalette) {
      error = "palette is not signallable for this block size";
      return false;
    }

    // Residual: three planes with EOB=0 (transform-skip / palette-only).
    if (!write_eob_zero(block, bw, bh, error)) return false;

    // Reconstruction.
    recon_intra(block, bw, bh, error);
    return error.empty();
  }

  bool write_ibc_block(const BlockPlan& block, int bw, int bh,
                       std::string& error) {
    // read_skip() is signalled for intra-frame IBC blocks only.
    const int skip_ctx =
        (block.mi_row > 0
             ? skips_[static_cast<std::size_t>(block.mi_row - 1) *
                           static_cast<std::size_t>(mi_cols_) +
                       static_cast<std::size_t>(block.mi_col)]
             : 0) +
        (block.mi_col > 0
             ? skips_[static_cast<std::size_t>(block.mi_row) *
                           static_cast<std::size_t>(mi_cols_) +
                       static_cast<std::size_t>(block.mi_col - 1)]
             : 0);
    if (!sym(&cdfs_.skip[std::min(skip_ctx, 2)][0], 2, block.skip ? 1 : 0)) {
      error = error_;
      return false;
    }
    if (!block.skip) {
      error = "the gate writer only supports IBC with skip_flag=1";
      return false;
    }

    // read_intrabc_info(): NEARMV (intrabc_mode=1) or NEWMV (0, not wired).
    if (!sym(&cdfs_.intrabc_mode[0], 2, block.intrabc_mode ? 1 : 0)) {
      error = error_;
      return false;
    }
    if (!block.intrabc_mode) {
      error = "the gate writer only supports intrabc_mode=1 (NEARMV)";
      return false;
    }
    const int m = config_.seq_max_bvp_drl_bits_minus_1 + 1;
    for (int idx = 0; idx < m; ++idx) {
      const bool more = idx < block.ref_mv_idx;
      if (!raw_bit(more)) {
        error = error_;
        return false;
      }
      if (!more) break;
    }
    if (block.ref_mv_idx > m) {
      error = "ref_mv_idx is outside the BVP DRL window";
      return false;
    }

    // Validate the BV against the local/global constraints before writing it.
    if (block.ref_mv_idx == 1) {
      // Default IBC candidates: index 0 = (0, -SbHeight),
      // index 1 = (-SbWidth - INTRABC_DELAY_PIXELS, 0) = (-320, 0) for 64 SB.
      const int expected_row = 0;
      const int expected_col = -(64 + 256) * 8;
      if (block.bv_row != expected_row || block.bv_col != expected_col) {
        error = "ref_mv_idx 1 requires the default (-320,0) BV";
        return false;
      }
    } else if (block.ref_mv_idx == 0) {
      const int expected_row = -64 * 8;
      if (block.bv_row != expected_row || block.bv_col != 0) {
        error = "ref_mv_idx 0 requires the default (0,-64) BV";
        return false;
      }
    } else {
      error = "only the two default IBC candidates are supported";
      return false;
    }
    std::string bv_error;
    if (!validate_global_bv(config_, block.mi_row, block.mi_col, block.bsize,
                            block.bv_row, block.bv_col, bv_error)) {
      error = bv_error;
      return false;
    }

    // skip_flag == 1: no residual symbols.  Prediction is a copy.
    ibc_copy(block, bw, bh, error);
    return error.empty();
  }

  bool write_eob_zero(const BlockPlan& block, int bw, int bh,
                      std::string& error) {
    (void)block;
    for (int plane = 0; plane < 3; ++plane) {
      const int cw = plane == 0 ? bw : bw / 2;
      const int ch = plane == 0 ? bh : bh / 2;
      const auto params_opt =
          txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        error = "unsupported residual transform size";
        return false;
      }
      const int tsc = tx_sz_ctx(static_cast<int>(params_opt->tx_size));
      std::uint16_t* cdf = nullptr;
      if (plane == 0) {
        cdf = &cdfs_.txb_skip[0][tsc][0][0];
      } else if (plane == 1) {
        cdf = &cdfs_.txb_skip[0][tsc][6][0];
      } else {
        // plane 2: above/left are zero, no extra 3, EobU == 0 => ctx 0.
        cdf = &cdfs_.v_txb_skip[0][0];
      }
      if (!sym(cdf, 2, 1)) {  // 1 = EOB 0 / transform skip
        error = error_;
        return false;
      }
    }
    return true;
  }

  void recon_intra(const BlockPlan& block, int bw, int bh,
                   std::string& /*error*/) {
    const int x = block.mi_col * 4;
    const int y = block.mi_row * 4;
    if (block.kind == BlockKind::kPalette) {
      palette::predict_palette_block(block.palette, bw, bh,
                                     recon_y_.data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(config_.width) +
                      static_cast<std::size_t>(x),
                                     config_.width);
    } else {
      intra::NeighborRequest req;
      req.recon = intra::PlaneView{recon_y_.data(), config_.width};
      req.x = x;
      req.y = y;
      req.w = bw;
      req.h = bh;
      req.max_x = config_.width - 1;
      req.max_y = config_.height - 1;
      req.have_left = x > 0;
      req.have_above = y > 0;
      req.color_plane = 0;
      intra::PredictParams params;
      params.mode = intra::Mode::kDc;
      params.enable_edge_filter = false;
      intra::predict_block(req, 8, params,
                           recon_y_.data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(config_.width) +
                      static_cast<std::size_t>(x),
                           static_cast<std::size_t>(config_.width));
    }
    // Chroma: DC with the same rule (palette is luma-only in AV2 v1.0.0).
    for (int plane = 1; plane < 3; ++plane) {
      std::vector<std::uint8_t>& dst = plane == 1 ? recon_u_ : recon_v_;
      const int stride = config_.width / 2;
      const int cx = x / 2;
      const int cy = y / 2;
      const int cw = bw / 2;
      const int ch = bh / 2;
      intra::NeighborRequest req;
      req.recon = intra::PlaneView{dst.data(), stride};
      req.x = cx;
      req.y = cy;
      req.w = cw;
      req.h = ch;
      req.max_x = stride - 1;
      req.max_y = config_.height / 2 - 1;
      req.have_left = cx > 0;
      req.have_above = cy > 0;
      req.color_plane = plane;
      intra::PredictParams params;
      params.mode = intra::Mode::kDc;
      params.enable_edge_filter = false;
      intra::predict_block(req, 8, params, dst.data() +
                           static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
                           static_cast<std::size_t>(cx),
                           static_cast<std::size_t>(stride));
    }
  }

  bool ibc_copy(const BlockPlan& block, int bw, int bh, std::string& error) {
    const int x = block.mi_col * 4;
    const int y = block.mi_row * 4;
    const int dx = ibc_plane_offset(block.bv_col, 0);
    const int dy = ibc_plane_offset(block.bv_row, 0);
    const int sx = x + dx;
    const int sy = y + dy;
    if (sx < 0 || sy < 0 || sx + bw > config_.width ||
        sy + bh > config_.height) {
      error = "IBC source block is outside the frame";
      return false;
    }
    for (int i = 0; i < bh; ++i) {
      std::memcpy(recon_y_.data() +
                      static_cast<std::size_t>(y + i) *
                          static_cast<std::size_t>(config_.width) +
                      static_cast<std::size_t>(x),
                  recon_y_.data() +
                      static_cast<std::size_t>(sy + i) *
                          static_cast<std::size_t>(config_.width) +
                      static_cast<std::size_t>(sx),
                  static_cast<std::size_t>(bw));
    }
    const int cdx = ibc_plane_offset(block.bv_col, 1);
    const int cdy = ibc_plane_offset(block.bv_row, 1);
    const int cstride = config_.width / 2;
    const int cx = x / 2;
    const int cy = y / 2;
    const int cw = bw / 2;
    const int ch = bh / 2;
    const int csx = cx + cdx;
    const int csy = cy + cdy;
    if (csx < 0 || csy < 0 || csx + cw > cstride ||
        csy + ch > config_.height / 2) {
      error = "IBC chroma source block is outside the frame";
      return false;
    }
    for (int plane = 1; plane < 3; ++plane) {
      std::vector<std::uint8_t>& dst = plane == 1 ? recon_u_ : recon_v_;
      for (int i = 0; i < ch; ++i) {
        std::memcpy(dst.data() +
                        static_cast<std::size_t>(cy + i) *
                            static_cast<std::size_t>(cstride) +
                        static_cast<std::size_t>(cx),
                    dst.data() +
                        static_cast<std::size_t>(csy + i) *
                            static_cast<std::size_t>(cstride) +
                        static_cast<std::size_t>(csx),
                    static_cast<std::size_t>(cw));
      }
    }
    return true;
  }

  std::vector<std::uint8_t> assemble(std::span<const std::uint8_t> payload) {
    BitWriter writer;
    writer.put_bit(true);  // is_first_tile_group
    write_frame_header(writer, config_);
    writer.align_to_byte();
    std::vector<std::uint8_t> frame_payload = writer.take_bytes();
    frame_payload.insert(frame_payload.end(), payload.begin(), payload.end());

    m1::M1Config seq_config;
    seq_config.max_frame_width = static_cast<std::uint16_t>(config_.width);
    seq_config.max_frame_height = static_cast<std::uint16_t>(config_.height);
    seq_config.base_q_idx = config_.base_q_idx;
    seq_config.enable_intra_edge_filter = config_.enable_intra_edge_filter;
    seq_config.enable_cfl_intra = config_.enable_cfl_intra;
    seq_config.enable_chroma_dctonly = true;
    seq_config.tx_mode_select = false;
    seq_config.apply_deblock_y_vert = config_.apply_deblock_y_vert;
    seq_config.apply_deblock_y_horz = config_.apply_deblock_y_horz;
    seq_config.apply_deblock_u = config_.apply_deblock_u;
    seq_config.apply_deblock_v = config_.apply_deblock_v;

    const std::vector<std::uint8_t> seq = m1::make_sequence_header_obu(seq_config);
    const std::vector<std::uint8_t> td =
        av2enc::m0::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                             std::span<const std::uint8_t>{});
    const std::vector<std::uint8_t> clk =
        av2enc::m0::make_obu(ObuHeader{ObuType::kClosedLoopKey}, frame_payload);
    std::vector<std::uint8_t> out;
    out.reserve(td.size() + seq.size() + clk.size());
    out.insert(out.end(), td.begin(), td.end());
    out.insert(out.end(), seq.begin(), seq.end());
    out.insert(out.end(), clk.begin(), clk.end());
    return out;
  }

  const SccFramePlan& plan_;
  SccConfig config_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  TileCdfs cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_;
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::vector<std::uint8_t> ibc_;
  std::vector<std::uint8_t> skips_;
  std::vector<std::uint8_t> recon_y_;
  std::vector<std::uint8_t> recon_u_;
  std::vector<std::uint8_t> recon_v_;
  palette::PaletteState palette_state_;
  std::string error_;
};

}  // namespace

int ibc_plane_offset(int bv8, int subsampling) {
  // §7.13.3.17.  startX/startY are in 1/1024 sample units; the +32 is the
  // SCALE_SUBPEL -> filter-tap rounding offset.  The current-frame x/y cancel,
  // leaving a pure offset.
  if (subsampling < 0 || subsampling > 3) return 0;
  const long long two = 2LL * static_cast<long long>(bv8);
  long long v = 0;
  if (two >= 0) {
    v = two >> subsampling;
  } else {
    v = -(((-two) + (1LL << subsampling) - 1) >> subsampling);
  }
  const long long num = v * 64 + 32;
  if (num >= 0) return static_cast<int>(num / 1024);
  return static_cast<int>(-((-num + 1023) / 1024));
}

bool validate_global_bv(const SccConfig& config, int mi_row, int mi_col,
                        int bsize, int bv_row, int bv_col,
                        std::string& error) {
  error.clear();
  const auto block = static_cast<partition::BlockSize>(bsize);
  const int bw = partition::block_width(block);
  const int bh = partition::block_height(block);
  if (bw <= 0 || bh <= 0) {
    error = "invalid IBC block size";
    return false;
  }
  const int bottom_border = (bv_row & 7) != 0 ? 1 : 0;
  const int right_border = (bv_col & 7) != 0 ? 1 : 0;
  const int delta_row = bv_row >> 3;
  const int delta_col = bv_col >> 3;
  int src_top = mi_row * 4 + delta_row;
  int src_left = mi_col * 4 + delta_col;
  const int src_bottom = src_top + bh + bottom_border;
  const int src_right = src_left + bw + right_border;
  // Chroma override (§6.19.7.12): the top-left is expressed in chroma
  // coordinates.  For 4:2:0 the chroma origin is half the luma origin.
  // ChromaMiRow/ChromaMiCol are in luma MI units for a shared partition
  // (the SDP chroma-partition override does not apply here).
  src_left = mi_col * 4 + delta_col;
  src_top = mi_row * 4 + delta_row;

  const int tile_top = 0;
  const int tile_left = 0;
  const int tile_bottom = mi_rows(config) * 4;
  const int tile_right = mi_cols(config) * 4;
  if (src_top < tile_top || src_left < tile_left || src_bottom > tile_bottom ||
      src_right > tile_right) {
    error = "IBC block vector points outside the tile";
    return false;
  }
  if (!config.allow_global_intrabc) {
    error = "global intra block copy is disabled";
    return false;
  }
  const int sb_h = 64;
  const int active_sb_row = (mi_row * 4) / sb_h;
  const int active_sb64_col = (mi_col * 4) >> 6;
  const int src_sb_row = (src_bottom - 1) / sb_h;
  const int src_sb64_col = (src_right - 1) >> 6;
  const int total_sb64_per_row = ((mi_cols(config) - 1) >> 4) + 1;
  const int active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
  const int src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
  const int sb64_residual = 0;  // AllowExtraIBCRange is 0 for 64x64 SBs
  if (src_sb64 >= active_sb64 - 4 - sb64_residual) {
    error = "IBC source is inside the INTRABC_DELAY_SB64 window";
    return false;
  }
  const int gradient = 4 + (64 / 64);
  const int wf_offset = gradient * (active_sb_row - src_sb_row);
  if (src_sb_row > active_sb_row ||
      src_sb64_col >= active_sb64_col - 4 + wf_offset - sb64_residual) {
    error = "IBC source violates the wavefront constraint";
    return false;
  }
  return true;
}

bool encode_scc_frame(const SccFramePlan& plan, SccEncodeResult& out,
                      std::string& error) {
  error.clear();
  out = SccEncodeResult{};
  if (plan.config.width <= 0 || plan.config.height <= 0 ||
      (plan.config.width & 1) != 0 || (plan.config.height & 1) != 0) {
    error = "SCC gate requires even positive dimensions";
    return false;
  }
  if (plan.config.width > 4096 || plan.config.height > 4096) {
    error = "SCC gate dimensions are unreasonably large";
    return false;
  }
  if (plan.config.allow_intrabc &&
      !(plan.config.allow_global_intrabc || plan.config.allow_local_intrabc)) {
    error = "allow_intrabc requires a reference region";
    return false;
  }
  TileWriter writer(plan, plan.config);
  return writer.run(out, error);
}

bool write_scc_y4m(const SccEncodeResult& result, const SccConfig& config,
                   const std::string& path, std::string& error) {
  error.clear();
  const int w = config.width;
  const int h = config.height;
  if (w <= 0 || h <= 0 || (w & 1) != 0 || (h & 1) != 0) {
    error = "invalid Y4M dimensions";
    return false;
  }
  if (static_cast<int>(result.recon_y.size()) != w * h ||
      static_cast<int>(result.recon_u.size()) != (w / 2) * (h / 2) ||
      static_cast<int>(result.recon_v.size()) != (w / 2) * (h / 2)) {
    error = "reconstruction size does not match the frame";
    return false;
  }
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    error = "cannot open " + path;
    return false;
  }
  std::fprintf(f, "YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n", w, h);
  std::fputs("FRAME\n", f);
  std::fwrite(result.recon_y.data(), 1, result.recon_y.size(), f);
  std::fwrite(result.recon_u.data(), 1, result.recon_u.size(), f);
  std::fwrite(result.recon_v.data(), 1, result.recon_v.size(), f);
  const bool ok = std::ferror(f) == 0;
  std::fclose(f);
  if (!ok) error = "Y4M write failed";
  return ok;
}

bool write_scc_ivf(const SccEncodeResult& result, const SccConfig& config,
                   const std::string& path, std::string& error) {
  error.clear();
  IvfWriter writer;
  IvfHeader header;
  header.width = static_cast<std::uint16_t>(config.width);
  header.height = static_cast<std::uint16_t>(config.height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(std::filesystem::path(path), header, error)) return false;
  if (!writer.write_frame(result.frame, 0, error)) return false;
  return writer.close(error);
}

}  // namespace av2enc::scc
