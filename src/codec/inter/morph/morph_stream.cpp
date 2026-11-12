#include "codec/inter/morph/morph_stream.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/intra/modes/intra_modes.h"
#include "codec/inter/morph/morph_common.h"
#include "codec/inter/morph/morph_syntax.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "codec/intra/ibc/stream/ibc_cdfs.h"
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "pipeline/still/still_cdf_tables.h"
#include "pipeline/still/still_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/entropy/av2/av2_entropy.h"
#include "core/io/ivf.h"

namespace av2enc::inter::morph {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

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

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  unsigned w = 0;
  while ((1U << w) < n) {
    ++w;
  }
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

int frame_mi_cols(const MorphStreamConfig& c) {
  return 2 * ((c.width + 7) / 8);
}
int frame_mi_rows(const MorphStreamConfig& c) {
  return 2 * ((c.height + 7) / 8);
}

// §5.4.1–§5.4.10 sequence header. Identical to av2enc::still::write_sequence_header()
// except enable_bawp = 1. single_picture_header_flag = 1, so the §5.4.6
// single-picture branch applies: enable_bawp is coded, enable_imp_msk_bld is
// inferred 0.
void write_sequence_header(BitWriter& writer, const MorphStreamConfig& c) {
  put_uvlc(writer, 0);                    // seq_header_id
  writer.put_bits(0U, 5U);                // seq_profile_idc
  writer.put_bit(true);                   // single_picture_header_flag
  writer.put_bits(c.seq_level_idx, 5U);   // seq_level_idx
  put_uvlc(writer, 0);                    // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                    // bit_depth_idc = 1 => 8-bit

  unsigned width_bits = bit_width(static_cast<std::uint32_t>(c.width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(c.height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.height) - 1U, height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag

  // §5.4.3 sequence_partition_config(): 64x64 SB, no SDP / extended parts.
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config().
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config().
  writer.put_bit(false);                 // enable_dip
  writer.put_bit(c.enable_intra_edge_filter);
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(false);  // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp

  // §5.4.6 sequence_inter_config(), single_picture_header_flag branch.
  writer.put_bit(false);   // enable_refmvbank
  writer.put_bit(true);    // disable_drl_reorder
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);   // allow_frame_max_bvp_drl_bits
  writer.put_bit(c.enable_bawp);  // morph_pred is signallable when 1

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(true);   // enable_chroma_dctonly
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(false);  // film_grain_params_present
}

std::vector<std::uint8_t> make_sequence_header_obu(const MorphStreamConfig& c) {
  BitWriter writer;
  write_sequence_header(writer, c);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

// §5.18.2 frame_header_info(): screen-content/intrabc bits enabled.
void write_frame_header(BitWriter& writer, const MorphStreamConfig& c) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  writer.put_bit(c.allow_screen_content_tools);
  if (c.allow_screen_content_tools) {
    writer.put_bit(true);  // force_integer_mv
  }
  writer.put_bit(c.allow_intrabc);
  if (c.allow_intrabc) {
    writer.put_bit(true);   // allow_global_intrabc
    writer.put_bit(false);  // allow_local_intrabc
  }

  writer.put_bit(c.disable_cdf_update);

  const int cols = (frame_mi_cols(c) + 15) / 16;
  const int rows = (frame_mi_rows(c) + 15) / 16;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (cols > 1) writer.put_bit(false);
  if (rows > 1) writer.put_bit(false);

  writer.put_bits(c.base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]

  writer.put_bit(false);  // tx_mode_select = TX_MODE_LARGEST
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
  av2enc::still::ibc_stream::IbcCdfs ibc;
  MorphCdfs morph;
};

void init_tile_cdfs(TileCdfs& c) {
  std::memcpy(c.do_split, av2enc::basic::kDoSplitCdf, sizeof(c.do_split));
  std::memcpy(c.rect_type, av2enc::basic::kRectTypeCdf, sizeof(c.rect_type));
  std::memcpy(c.y_mode_set, av2enc::still::kYModeSetCdf, sizeof(c.y_mode_set));
  std::memcpy(c.y_mode_index, av2enc::still::kYModeIndexCdf,
              sizeof(c.y_mode_index));
  std::memcpy(c.uv_mode, av2enc::still::kUvModeCflNotAllowedCdf,
              sizeof(c.uv_mode));
  std::memcpy(c.txb_skip, av2enc::basic::kTxbSkipCdf, sizeof(c.txb_skip));
  std::memcpy(c.v_txb_skip, av2enc::basic::kVTxbSkipCdf, sizeof(c.v_txb_skip));
  av2enc::still::ibc_stream::init_ibc_cdfs(c.ibc);
  init_morph_cdfs(c.morph);
}

int tx_sz_ctx(int tx_size) {
  return (av2enc::still::kTxSizeSqr[tx_size] + av2enc::still::kTxSizeSqrUp[tx_size] + 1) >> 1;
}

// §5.20.4.1 context list. `buffered == false` is NPos (same superblock row and
// at most two entries); `buffered == true` is NPosBuf (may cross a superblock
// row). This mirrors the list used by the IBC adapter but stays local so this
// component does not link the whole IBC stream writer.
struct NeighbourList {
  int count = 0;
  int row[2] = {0, 0};
  int col[2] = {0, 0};
};

NeighbourList neighbours(int mi_row, int mi_col, int bw4, int bh4, int mi_cols,
                         int mi_rows, int sb_size, bool buffered) {
  NeighbourList out;
  int sb_mi = std::max(1, sb_size / 4);
  int sb_log2 = 0;
  while (sb_mi > 1) {
    sb_mi >>= 1;
    ++sb_log2;
  }
  const auto inside = [&](int r, int c) {
    return r >= 0 && c >= 0 && r < mi_rows && c < mi_cols;
  };
  const auto add = [&](int r, int c) {
    if (out.count >= 2 || !inside(r, c)) return;
    if (!buffered && ((mi_row >> sb_log2) != (r >> sb_log2))) return;
    out.row[out.count] = r;
    out.col[out.count] = c;
    ++out.count;
  };
  add(mi_row + bh4 - 1, mi_col - 1);
  add(mi_row - 1, mi_col + bw4 - 1);
  add(mi_row, mi_col - 1);
  add(mi_row - 1, mi_col);
  return out;
}

int context_sum(const std::vector<std::uint8_t>& grid, int mi_cols,
                int mi_rows, int mi_row, int mi_col, int bw4, int bh4,
                int sb_size, bool buffered) {
  const NeighbourList n =
      neighbours(mi_row, mi_col, bw4, bh4, mi_cols, mi_rows, sb_size, buffered);
  int ctx = 0;
  for (int i = 0; i < n.count; ++i) {
    if (grid[static_cast<std::size_t>(n.row[i]) *
                 static_cast<std::size_t>(mi_cols) +
             static_cast<std::size_t>(n.col[i])] != 0) {
      ++ctx;
    }
  }
  return ctx;
}

class TileWriter {
 public:
  TileWriter(const MorphStreamPlan& plan, const MorphStreamResult& /*unused*/)
      : plan_(plan),
        config_(plan.config),
        mi_cols_(frame_mi_cols(plan.config)),
        mi_rows_(frame_mi_rows(plan.config)) {
    init_tile_cdfs(cdfs_);
    tile_.resize(kTileCapacity);
    enc_.init_symbol(tile_.data(), tile_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;
    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    ibc_.assign(static_cast<std::size_t>(mi_cols_) *
                    static_cast<std::size_t>(mi_rows_),
                0);
    skips_.assign(ibc_.size(), 0);
    morph_preds_.assign(ibc_.size(), 0);
    neighbours_.reset(mi_cols_, mi_rows_);
    buffer_.reset(mi_cols_, mi_rows_);
    ibc_config_.sb_size = 64;
    ibc_config_.allow_global = true;
    ibc_config_.allow_local = false;
    ibc_config_.max_bvp_drl_bits_minus_1 = 0;
    recon_y_.assign(static_cast<std::size_t>(config_.width) *
                        static_cast<std::size_t>(config_.height),
                    128);
    recon_u_.assign(static_cast<std::size_t>(config_.width / 2) *
                        static_cast<std::size_t>(config_.height / 2),
                    128);
    recon_v_.assign(recon_u_.size(), 128);
  }

  bool run(MorphStreamResult& out, std::string& error) {
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
      const MorphStreamSb& sb = plan_.sbs[static_cast<std::size_t>(sb_index)];
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
    out.ibc_blocks = ibc_blocks_;
    out.morph_symbols = morph_symbols_;
    out.morph_blocks = morph_blocks_;
    for (int i = 0; i < 3; ++i) {
      out.morph_ctx[i] = morph_ctx_[i];
      out.morph_ctx_value[i][0] = morph_ctx_value_[i][0];
      out.morph_ctx_value[i][1] = morph_ctx_value_[i][1];
    }
    out.morph_fitted = morph_fitted_;
    out.morph_alpha_min = morph_alpha_min_;
    out.morph_alpha_max = morph_alpha_max_;
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
  }

  MorphStreamSb::Partition implied_partition(int mi_row, int mi_col) const {
    const int half = 8;
    const bool has_rows = mi_row + half < mi_rows_;
    const bool has_cols = mi_col + half < mi_cols_;
    if (has_rows && has_cols) return MorphStreamSb::Partition::kNone;
    return has_rows ? MorphStreamSb::Partition::kVert
                    : MorphStreamSb::Partition::kHorz;
  }

  bool child_partition_implied(int bsize, int mi_row, int mi_col) const {
    const int half_w = 1 << (std::max(av2enc::still::kMiWidthLog2[bsize], 1) - 1);
    const int half_h = 1 << (std::max(av2enc::still::kMiHeightLog2[bsize], 1) - 1);
    return !((mi_row + half_h < mi_rows_) && (mi_col + half_w < mi_cols_));
  }

  bool write_sb(const MorphStreamSb& sb, int mi_row, int mi_col,
                std::string& error) {
    const int bsize = 12;  // BLOCK_64X64
    const int bsw = std::max(av2enc::still::kMiWidthLog2[bsize], 1);
    const int bsh = std::max(av2enc::still::kMiHeightLog2[bsize], 1);
    const MorphStreamSb::Partition implied = implied_partition(mi_row, mi_col);
    const bool split = sb.partition != MorphStreamSb::Partition::kNone;
    if (implied != MorphStreamSb::Partition::kNone) {
      if (sb.partition != implied) {
        error = "superblock partition does not match the implied boundary cut";
        return false;
      }
    } else {
      const int ctx = av2enc::still::kPartitionSizeAdjust[bsize] * 4 +
                      (left_mi_[static_cast<std::size_t>(mi_row)] < bsh ? 2 : 0) +
                      (above_mi_[static_cast<std::size_t>(mi_col)] < bsw ? 1 : 0);
      if (!sym(&cdfs_.do_split[0][ctx][0], 2, split ? 1 : 0)) {
        error = error_;
        return false;
      }
    }
    if (split && implied == MorphStreamSb::Partition::kNone) {
      const int rect_ctx =
          av2enc::still::kPartitionSizeAdjustRect[bsize] * 4 +
          (left_mi_[static_cast<std::size_t>(mi_row)] < bsh ? 2 : 0) +
          (above_mi_[static_cast<std::size_t>(mi_col)] < bsw ? 1 : 0);
      const int rect = sb.partition == MorphStreamSb::Partition::kVert ? 1 : 0;
      if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, rect)) {
        error = error_;
        return false;
      }
    }
    std::size_t expected = 1U;
    if (split) {
      expected = 0;
      if (sb.partition == MorphStreamSb::Partition::kVert) {
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
        !split ? 12
               : (sb.partition == MorphStreamSb::Partition::kVert ? 10 : 11);
    for (const MorphStreamBlock& leaf : sb.leaves) {
      if (leaf.bsize != child_bsize) {
        error = "leaf block size does not match the partition subsize";
        return false;
      }
      if (split &&
          !child_partition_implied(child_bsize, leaf.mi_row, leaf.mi_col)) {
        const int cbw = std::max(av2enc::still::kMiWidthLog2[child_bsize], 1);
        const int cbh = std::max(av2enc::still::kMiHeightLog2[child_bsize], 1);
        const int cctx =
            av2enc::still::kPartitionSizeAdjust[child_bsize] * 4 +
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

  bool write_intra_modes(int bsize, int bw, int bh, int y_mode,
                         std::string& error) {
    // Reordered_Y_Mode index: DC_PRED -> 0, SMOOTH_PRED -> 1. Both are
    // non-directional, so y_mode_index context is 0 and no offset follows.
    int mode_index = -1;
    if (y_mode == 0) {
      mode_index = 0;  // DC_PRED
    } else if (y_mode == 9) {
      mode_index = 1;  // SMOOTH_PRED
    } else {
      error = "gate writer supports only DC_PRED and SMOOTH_PRED intra modes";
      return false;
    }
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.y_mode_index[0][0], 8, mode_index)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.uv_mode[0][0], 8, 0)) {
      error = error_;
      return false;
    }
    // §5.20.8.1 palette_mode_info(): PlaneStart == 0 && YMode == DC_PRED.
    if (config_.allow_screen_content_tools && y_mode == 0 &&
        bsize >= static_cast<int>(partition::BlockSize::k8x8) && bw <= 64 &&
        bh <= 64) {
      // has_palette_y = 0.
      if (!sym(&cdfs_.ibc.palette_y_mode[0], 2, 0)) {
        error = error_;
        return false;
      }
    }
    return write_eob_zero(bw, bh, false, error);
  }

  bool write_eob_zero(int bw, int bh, bool is_inter, std::string& error) {
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
        cdf = &cdfs_.txb_skip[is_inter ? 1 : 0][tsc][0][0];
        if (!sym(cdf, 2, 1)) {
          error = error_;
          return false;
        }
      } else if (plane == 1) {
        cdf = &cdfs_.txb_skip[is_inter ? 1 : 0][tsc][6][0];
        if (!sym(cdf, 2, 1)) {
          error = error_;
          return false;
        }
      } else {
        cdf = &cdfs_.v_txb_skip[0][0];
        if (!sym(cdf, 2, 1)) {
          error = error_;
          return false;
        }
      }
    }
    return true;
  }

  bool write_block(const MorphStreamBlock& block, std::string& error) {
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
    const bool ibc_allowed =
        config_.allow_intrabc &&
        av2enc::still::ibc_stream::intrabc_available(true, block.bsize);
    if (block.ibc && !ibc_allowed) {
      error = "IBC block is not signallable for this size/config";
      return false;
    }

    av2enc::still::ibc_stream::begin_ibc_block(buffer_, block.mi_row, block.mi_col);

    const int bw4 = std::max(1, bw / 4);
    const int bh4 = std::max(1, bh / 4);
    // §5.20.5.3: use_intrabc is coded for every block where it is available,
    // including the zero value on intra blocks.
    if (ibc_allowed) {
      const int ibc_ctx =
          std::min(context_sum(ibc_, mi_cols_, mi_rows_, block.mi_row,
                               block.mi_col, bw4, bh4, 64, false),
                   2);
      if (!av2enc::still::ibc_stream::write_use_intrabc(enc_, cdfs_.ibc, ibc_ctx,
                                             block.ibc, error)) {
        return false;
      }
    }

    ibc::BlockVector chosen_bv{0, 0};
    bool is_ibc = false;
    if (block.ibc) {
      const int skip_ctx =
          std::min(context_sum(skips_, mi_cols_, mi_rows_, block.mi_row,
                               block.mi_col, bw4, bh4, 64, true),
                   2);
      if (!av2enc::still::ibc_stream::write_skip_flag(enc_, cdfs_.ibc, skip_ctx,
                                           block.skip, error)) {
        return false;
      }
      const av2enc::still::ibc_stream::BvStack stack = av2enc::still::ibc_stream::build_bv_stack(
          frame_view(), ibc_config_, buffer_, block.mi_row, block.mi_col,
          block.bsize, neighbours_);
      if (config_.strict_border) {
        std::string why;
        if (!av2enc::still::ibc_stream::is_bv_encodable(
                frame_view(), ibc_config_, buffer_, block.mi_row, block.mi_col,
                block.bsize, block.bv, &why)) {
          error = "IBC block " + std::to_string(block.mi_row) + "," +
                  std::to_string(block.mi_col) + ": " + why;
          return false;
        }
      }
      if (!av2enc::still::ibc_stream::write_intrabc_info(enc_, cdfs_.ibc, content_config(),
                                              stack, block.syntax, block.bv,
                                              error)) {
        return false;
      }

      // --- morph_pred S() ------------------------------------------------
      MorphPredGate gate;
      gate.enable_bawp = config_.enable_bawp;                    // sequence
      gate.allow_screen_content_tools =
          config_.allow_screen_content_tools;                    // frame
      gate.frame_is_intra = true;
      gate.use_intrabc = true;
      gate.enable_imp_msk_bld = false;          // inferred in single-picture
      const bool morph_present =
          config_.enable_bawp && config_.allow_screen_content_tools;
      if (!morph_present && block.morph_pred) {
        error = "morph_pred is not signallable in this configuration";
        return false;
      }
      const int morph_ctx = morph_symbol_context(
          morph_preds_.data(), mi_cols_, mi_rows_, block.mi_row, block.mi_col,
          bw4, bh4, 64);
      if (morph_present) {
        ++morph_ctx_[morph_ctx];
        ++morph_ctx_value_[morph_ctx][block.morph_pred ? 1 : 0];
      }
      if (block.morph_pred) {
        // §6.19.5.4: always check the -1,-1 offset, independent of the
        // encoder's optional strict-border mode.
        std::string why;
        const bool offset_valid = av2enc::still::ibc_stream::is_bv_offset_valid(
            frame_view(), ibc_config_, buffer_, block.mi_row, block.mi_col,
            block.bsize, block.bv, -1, -1, &why);
        if (!morph_pred_allowed(gate, offset_valid, &why)) {
          error = "IBC block " + std::to_string(block.mi_row) + "," +
                  std::to_string(block.mi_col) + ": " + why;
          return false;
        }
        // §7.13.3.25 conformance box (template reads outside the block).
        MorphContext mctx;
        mctx.mi_col = block.mi_col;
        mctx.mi_row = block.mi_row;
        mctx.mi_cols = mi_cols_;
        mctx.mi_rows = mi_rows_;
        MorphRect rect{x, y, bw, bh};
        int rx = 0;
        int ry = 0;
        int rbw = 0;
        int rbh = 0;
        if (!morph_reference_geometry(mctx, rect, block.bv.row, block.bv.col,
                                      &rx, &ry, &rbw, &rbh) ||
            !morph_reference_in_bounds(rx, ry, rbw, rbh, mi_cols_ * 4,
                                       mi_rows_ * 4)) {
          error = "morph_pred reference block violates the §7.13.3.25 box";
          return false;
        }
      }
      if (morph_present) {
        if (!write_morph_pred(enc_, cdfs_.morph, morph_ctx, block.morph_pred,
                              error)) {
          return false;
        }
        ++morph_symbols_;
      }
      if (!block.skip && !write_eob_zero(bw, bh, true, error)) return false;

      chosen_bv = block.bv;
      is_ibc = true;
      ++ibc_blocks_;
      if (!ibc_copy(block, bw, bh, error)) return false;
      if (block.morph_pred) {
        MorphContext mctx;
        mctx.mi_col = block.mi_col;
        mctx.mi_row = block.mi_row;
        mctx.mi_cols = mi_cols_;
        mctx.mi_rows = mi_rows_;
        mctx.avail_u = block.mi_row > 0;
        mctx.avail_l = block.mi_col > 0;
        MorphPlane plane{recon_y_.data(), config_.width};
        MorphRect rect{x, y, bw, bh};
        MorphFit fit{};
        if (!morph_predict(mctx, plane, rect, block.bv.row, block.bv.col,
                           &fit)) {
          error = "morph prediction failed after the symbol was written";
          return false;
        }
        ++morph_blocks_;
        if (fit.alpha != 256 || fit.beta != 0) ++morph_fitted_;
        morph_alpha_min_ = std::min(morph_alpha_min_, fit.alpha);
        morph_alpha_max_ = std::max(morph_alpha_max_, fit.alpha);
      }
    } else {
      if (!write_intra_modes(block.bsize, bw, bh, block.y_mode, error)) {
        return false;
      }
      recon_intra(block, bw, bh);
    }

    const int w4 = bw / 4;
    const int h4 = bh / 4;
    for (int yy = 0; yy < h4; ++yy) {
      const int r = block.mi_row + yy;
      if (r >= mi_rows_) break;
      left_mi_[static_cast<std::size_t>(r)] =
          av2enc::still::kMiHeightLog2[block.bsize];
      for (int xx = 0; xx < w4; ++xx) {
        const int c = block.mi_col + xx;
        if (c >= mi_cols_) break;
        const std::size_t idx = static_cast<std::size_t>(r) *
                                    static_cast<std::size_t>(mi_cols_) +
                                static_cast<std::size_t>(c);
        ibc_[idx] = is_ibc ? 1 : 0;
        skips_[idx] = (is_ibc && block.skip) ? 1 : 0;
        morph_preds_[idx] = (is_ibc && block.morph_pred) ? 1 : 0;
      }
    }
    for (int xx = 0; xx < w4; ++xx) {
      const int c = block.mi_col + xx;
      if (c >= mi_cols_) break;
      above_mi_[static_cast<std::size_t>(c)] =
          av2enc::still::kMiWidthLog2[block.bsize];
    }
    av2enc::still::ibc_stream::finish_ibc_block(buffer_, neighbours_, block.mi_row, block.mi_col,
                     block.bsize, is_ibc, chosen_bv);
    return true;
  }

  ibc::FrameView frame_view() const {
    ibc::FrameView view;
    view.y = ibc::PlaneView{recon_y_.data(), config_.width, config_.width,
                            config_.height};
    view.u = ibc::PlaneView{recon_u_.data(), config_.width / 2,
                            config_.width / 2, config_.height / 2};
    view.v = ibc::PlaneView{recon_v_.data(), config_.width / 2,
                            config_.width / 2, config_.height / 2};
    view.has_chroma = true;
    view.sub_x = 1;
    view.sub_y = 1;
    return view;
  }

  av2enc::still::ibc_stream::ScreenContentConfig content_config() const {
    av2enc::still::ibc_stream::ScreenContentConfig c;
    c.force_integer_mv = true;
    c.enable_mvd_sign_derive = false;
    c.max_bvp_drl_bits_minus_1 = 0;
    return c;
  }

  void recon_intra(const MorphStreamBlock& block, int bw, int bh) {
    const int x = block.mi_col * 4;
    const int y = block.mi_row * 4;
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
    params.mode = block.y_mode == 9 ? intra::Mode::kSmooth : intra::Mode::kDc;
    params.enable_edge_filter = false;
    intra::predict_block(req, 8, params,
                         recon_y_.data() +
                             static_cast<std::size_t>(y) *
                                 static_cast<std::size_t>(config_.width) +
                             static_cast<std::size_t>(x),
                         static_cast<std::size_t>(config_.width));
    for (int plane = 1; plane < 3; ++plane) {
      std::vector<std::uint8_t>& dst = plane == 1 ? recon_u_ : recon_v_;
      const int stride = config_.width / 2;
      const int cx = x / 2;
      const int cy = y / 2;
      const int cw = bw / 2;
      const int ch = bh / 2;
      intra::NeighborRequest creq;
      creq.recon = intra::PlaneView{dst.data(), stride};
      creq.x = cx;
      creq.y = cy;
      creq.w = cw;
      creq.h = ch;
      creq.max_x = stride - 1;
      creq.max_y = config_.height / 2 - 1;
      creq.have_left = cx > 0;
      creq.have_above = cy > 0;
      creq.color_plane = plane;
      intra::PredictParams cparams;
      cparams.mode = intra::Mode::kDc;
      cparams.enable_edge_filter = false;
      intra::predict_block(creq, 8, cparams,
                           dst.data() +
                               static_cast<std::size_t>(cy) *
                                   static_cast<std::size_t>(stride) +
                               static_cast<std::size_t>(cx),
                           static_cast<std::size_t>(stride));
    }
  }

  bool ibc_copy(const MorphStreamBlock& block, int bw, int bh,
                std::string& error) {
    const int x = block.mi_col * 4;
    const int y = block.mi_row * 4;
    const int cstride = config_.width / 2;
    const int cx = x / 2;
    const int cy = y / 2;
    const ibc::FrameView view = frame_view();
    if (!ibc::copy_block(
            view, block.mi_row, block.mi_col, block.bsize, block.bv,
            recon_y_.data() +
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(config_.width) +
                static_cast<std::size_t>(x),
            config_.width,
            recon_u_.data() +
                static_cast<std::size_t>(cy) * static_cast<std::size_t>(cstride) +
                static_cast<std::size_t>(cx),
            cstride,
            recon_v_.data() +
                static_cast<std::size_t>(cy) * static_cast<std::size_t>(cstride) +
                static_cast<std::size_t>(cx),
            cstride, bw, bh)) {
      error = "IBC copy source is outside the plane";
      return false;
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

    const std::vector<std::uint8_t> seq = make_sequence_header_obu(config_);
    const std::vector<std::uint8_t> td =
        av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                             std::span<const std::uint8_t>{});
    const std::vector<std::uint8_t> clk =
        av2enc::basic::make_obu(ObuHeader{ObuType::kClosedLoopKey},
                             frame_payload);
    std::vector<std::uint8_t> out;
    out.reserve(td.size() + seq.size() + clk.size());
    out.insert(out.end(), td.begin(), td.end());
    out.insert(out.end(), seq.begin(), seq.end());
    out.insert(out.end(), clk.begin(), clk.end());
    return out;
  }

  const MorphStreamPlan& plan_;
  MorphStreamConfig config_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  TileCdfs cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_;
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
  std::vector<std::uint8_t> ibc_;
  std::vector<std::uint8_t> skips_;
  std::vector<std::uint8_t> morph_preds_;
  av2enc::still::ibc_stream::BvNeighbourMap neighbours_;
  ibc::IbcBufferState buffer_;
  ibc::IbcConfig ibc_config_;
  std::vector<std::uint8_t> recon_y_;
  std::vector<std::uint8_t> recon_u_;
  std::vector<std::uint8_t> recon_v_;
  std::string error_;
  int ibc_blocks_ = 0;
  int morph_symbols_ = 0;
  int morph_blocks_ = 0;
  int morph_ctx_[3] = {0, 0, 0};
  int morph_ctx_value_[3][2] = {};
  int morph_fitted_ = 0;
  int morph_alpha_min_ = 256;
  int morph_alpha_max_ = 256;
};

}  // namespace

bool encode_morph_stream(const MorphStreamPlan& plan, MorphStreamResult& out,
                         std::string& error) {
  error.clear();
  out = MorphStreamResult{};
  const MorphStreamConfig& c = plan.config;
  if (c.width <= 0 || c.height <= 0 || (c.width & 1) != 0 ||
      (c.height & 1) != 0) {
    error = "stream requires even positive dimensions";
    return false;
  }
  if (c.width > 4096 || c.height > 4096) {
    error = "stream dimensions are unreasonably large";
    return false;
  }
  if (c.base_q_idx == 0) {
    error = "base_q_idx = 0 would make the frame lossless";
    return false;
  }
  if (c.allow_intrabc && !c.allow_screen_content_tools) {
    error = "allow_intrabc requires allow_screen_content_tools";
    return false;
  }
  TileWriter writer(plan, out);
  return writer.run(out, error);
}

bool write_morph_y4m(const MorphStreamResult& result,
                     const MorphStreamConfig& config, const std::string& path,
                     std::string& error) {
  error.clear();
  const int w = config.width;
  const int h = config.height;
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

bool write_morph_ivf(const MorphStreamResult& result,
                     const MorphStreamConfig& config, const std::string& path,
                     std::string& error) {
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

}  // namespace av2enc::inter::morph
