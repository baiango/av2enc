#include "codec/intra/ibc/stream/still_ibc_stream.h"

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
#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/still_cdf_tables.h"
#include "pipeline/still/still_headers.h"
#include "pipeline/still/still_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/entropy/av2/av2_entropy.h"
#include "core/io/ivf.h"

namespace av2enc::still::ibc_stream {
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

int frame_mi_cols(const StreamConfig& c) {
  return 2 * ((c.width + 7) / 8);
}
int frame_mi_rows(const StreamConfig& c) {
  return 2 * ((c.height + 7) / 8);
}

// §5.18.1/§5.18.2 frame header with the screen-content bits enabled.  This is
// the same layout as the existing M1 writer for the force_integer_mv=1 path.
void write_frame_header(BitWriter& writer, const StreamConfig& c) {
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  writer.put_bit(c.allow_screen_content_tools);
  if (c.allow_screen_content_tools) {
    writer.put_bit(c.force_integer_mv);
  }

  writer.put_bit(c.allow_intrabc);
  if (c.allow_intrabc) {
    writer.put_bit(c.allow_global_intrabc);
    if (c.allow_global_intrabc) {
      writer.put_bit(c.allow_local_intrabc);
    }
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

  writer.put_bit(false);  // tx_mode_select = TX_MODE_LARGEST
  writer.put_bits(static_cast<std::uint64_t>(c.reduced_tx_set), 2U);
}

struct TileCdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t uv_mode[2][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  IbcCdfs ibc;
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
  init_ibc_cdfs(c.ibc);
}

int tx_sz_ctx(int tx_size) {
  return (av2enc::still::kTxSizeSqr[tx_size] + av2enc::still::kTxSizeSqrUp[tx_size] + 1) >> 1;
}

class TileWriter {
 public:
  TileWriter(const StreamPlan& plan, const StreamResult& /*unused*/)
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
    neighbours_.reset(mi_cols_, mi_rows_);
    buffer_.reset(mi_cols_, mi_rows_);
    ibc_config_.sb_size = 64;
    ibc_config_.allow_global = config_.allow_global_intrabc;
    ibc_config_.allow_local = config_.allow_local_intrabc;
    ibc_config_.max_bvp_drl_bits_minus_1 =
        config_.seq_max_bvp_drl_bits_minus_1;
    recon_y_.assign(static_cast<std::size_t>(config_.width) *
                        static_cast<std::size_t>(config_.height),
                    128);
    recon_u_.assign(static_cast<std::size_t>(config_.width / 2) *
                        static_cast<std::size_t>(config_.height / 2),
                    128);
    recon_v_.assign(recon_u_.size(), 128);
  }

  bool run(StreamResult& out, std::string& error) {
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
      const StreamSb& sb = plan_.sbs[static_cast<std::size_t>(sb_index)];
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
    out.newmv_blocks = newmv_blocks_;
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

  StreamSb::Partition implied_partition(int mi_row, int mi_col) const {
    const int half = 8;
    const bool has_rows = mi_row + half < mi_rows_;
    const bool has_cols = mi_col + half < mi_cols_;
    if (has_rows && has_cols) return StreamSb::Partition::kNone;
    return has_rows ? StreamSb::Partition::kVert : StreamSb::Partition::kHorz;
  }

  bool child_partition_implied(int bsize, int mi_row, int mi_col) const {
    const int half_w = 1 << (std::max(av2enc::still::kMiWidthLog2[bsize], 1) - 1);
    const int half_h = 1 << (std::max(av2enc::still::kMiHeightLog2[bsize], 1) - 1);
    return !((mi_row + half_h < mi_rows_) && (mi_col + half_w < mi_cols_));
  }

  bool write_sb(const StreamSb& sb, int mi_row, int mi_col,
                std::string& error) {
    const int bsize = 12;  // BLOCK_64X64
    const int bsw = std::max(av2enc::still::kMiWidthLog2[bsize], 1);
    const int bsh = std::max(av2enc::still::kMiHeightLog2[bsize], 1);
    const StreamSb::Partition implied = implied_partition(mi_row, mi_col);
    const bool split = sb.partition != StreamSb::Partition::kNone;
    if (implied != StreamSb::Partition::kNone) {
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
    if (split && implied == StreamSb::Partition::kNone) {
      const int rect_ctx =
          av2enc::still::kPartitionSizeAdjustRect[bsize] * 4 +
          (left_mi_[static_cast<std::size_t>(mi_row)] < bsh ? 2 : 0) +
          (above_mi_[static_cast<std::size_t>(mi_col)] < bsw ? 1 : 0);
      const int rect = sb.partition == StreamSb::Partition::kVert ? 1 : 0;
      if (!sym(&cdfs_.rect_type[0][rect_ctx][0], 2, rect)) {
        error = error_;
        return false;
      }
    }
    std::size_t expected = 1U;
    if (split) {
      expected = 0;
      if (sb.partition == StreamSb::Partition::kVert) {
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
               : (sb.partition == StreamSb::Partition::kVert ? 10 : 11);
    for (const StreamBlock& leaf : sb.leaves) {
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

  bool write_intra_modes(int bw, int bh, std::string& error) {
    // read_intra_y_mode(): DC is mode index 0 in both lists.
    if (!sym(&cdfs_.y_mode_set[0], 4, 0)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.y_mode_index[0][0], 8, 0)) {
      error = error_;
      return false;
    }
    if (config_.enable_cfl_intra) {
      error = "stream writer keeps CfL disabled so is_cfl is not coded";
      return false;
    }
    if (!sym(&cdfs_.uv_mode[0][0], 8, 0)) {
      error = error_;
      return false;
    }
    if (config_.allow_screen_content_tools && bw <= 64 && bh <= 64) {
      // §5.20.8.1 palette_mode_info(): has_palette_y = 0.
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

  bool write_block(const StreamBlock& block, std::string& error) {
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
    if (block.ibc && !config_.allow_intrabc) {
      error = "IBC block is not signallable with allow_intrabc=0";
      return false;
    }
    if (block.ibc && !config_.allow_screen_content_tools) {
      error = "IBC block requires allow_screen_content_tools=1";
      return false;
    }
    const bool ibc_allowed = config_.allow_intrabc &&
                             intrabc_available(true, block.bsize);
    if (block.ibc && !ibc_allowed) {
      error = "IBC block is not signallable for this size/config";
      return false;
    }

    begin_ibc_block(buffer_, block.mi_row, block.mi_col);

    BlockVector chosen_bv{0, 0};
    bool is_ibc = false;
    if (ibc_allowed) {
      const int ibc_ctx = std::min(
          ibc_symbol_context(ibc_, mi_cols_, mi_rows_, block.mi_row,
                             block.mi_col, block.bsize, 64),
          2);
      if (!write_use_intrabc(enc_, cdfs_.ibc, ibc_ctx, block.ibc, error)) {
        return false;
      }
      if (block.ibc) {
        const int skip_ctx = std::min(
            skip_symbol_context(skips_, mi_cols_, mi_rows_, block.mi_row,
                                block.mi_col, block.bsize, 64),
            2);
        if (!write_skip_flag(enc_, cdfs_.ibc, skip_ctx, block.skip, error)) {
          return false;
        }
        const BvStack stack = build_bv_stack(
            frame_view(), ibc_config_, buffer_, block.mi_row, block.mi_col,
            block.bsize, neighbours_);
        if (config_.strict_border) {
          std::string why;
          if (!is_bv_encodable(frame_view(), ibc_config_, buffer_,
                               block.mi_row, block.mi_col, block.bsize,
                               block.bv, &why)) {
            error = "IBC block " + std::to_string(block.mi_row) + "," +
                    std::to_string(block.mi_col) + ": " + why;
            return false;
          }
        } else if (!is_bv_valid(frame_view(), ibc_config_, buffer_,
                                block.mi_row, block.mi_col, block.bsize,
                                block.bv, nullptr)) {
          error = "IBC block vector is invalid";
          return false;
        }
        if (!write_intrabc_info(enc_, cdfs_.ibc, content_config(), stack,
                                block.syntax, block.bv, error)) {
          return false;
        }
        if (!block.skip && !write_eob_zero(bw, bh, true, error)) return false;
        chosen_bv = block.bv;
        is_ibc = true;
        ++ibc_blocks_;
        if (block.syntax.mode == IntrabcMode::kNewMv) ++newmv_blocks_;
        if (!ibc_copy(block, bw, bh, error)) return false;
      } else {
        if (!write_intra_modes(bw, bh, error)) return false;
        recon_intra(block, bw, bh);
      }
    } else {
      if (block.ibc) {
        error = "IBC block is not signallable for this size/config";
        return false;
      }
      if (!write_intra_modes(bw, bh, error)) return false;
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
      }
    }
    for (int xx = 0; xx < w4; ++xx) {
      const int c = block.mi_col + xx;
      if (c >= mi_cols_) break;
      above_mi_[static_cast<std::size_t>(c)] =
          av2enc::still::kMiWidthLog2[block.bsize];
    }
    finish_ibc_block(buffer_, neighbours_, block.mi_row, block.mi_col,
                     block.bsize, is_ibc, chosen_bv);
    return true;
  }

  FrameView frame_view() const {
    FrameView view;
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

  ScreenContentConfig content_config() const {
    ScreenContentConfig c;
    c.force_integer_mv = config_.force_integer_mv;
    c.enable_mvd_sign_derive = false;
    c.max_bvp_drl_bits_minus_1 = config_.seq_max_bvp_drl_bits_minus_1;
    return c;
  }

  void recon_intra(const StreamBlock& block, int bw, int bh) {
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
    params.mode = intra::Mode::kDc;
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

  bool ibc_copy(const StreamBlock& block, int bw, int bh, std::string& error) {
    const int x = block.mi_col * 4;
    const int y = block.mi_row * 4;
    const int cstride = config_.width / 2;
    const int cx = x / 2;
    const int cy = y / 2;
    const FrameView view = frame_view();
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

    av2enc::still::StillConfig seq_config;
    seq_config.max_frame_width = static_cast<std::uint16_t>(config_.width);
    seq_config.max_frame_height = static_cast<std::uint16_t>(config_.height);
    seq_config.seq_level_idx = config_.seq_level_idx;
    seq_config.base_q_idx = config_.base_q_idx;
    seq_config.disable_cdf_update = config_.disable_cdf_update;
    seq_config.enable_intra_edge_filter = config_.enable_intra_edge_filter;
    seq_config.enable_cfl_intra = config_.enable_cfl_intra;
    seq_config.enable_chroma_dctonly = config_.enable_chroma_dctonly;
    seq_config.tx_mode_select = false;
    seq_config.reduced_tx_set = config_.reduced_tx_set;
    seq_config.apply_deblock_y_vert = config_.apply_deblock_y_vert;
    seq_config.apply_deblock_y_horz = config_.apply_deblock_y_horz;
    seq_config.apply_deblock_u = config_.apply_deblock_u;
    seq_config.apply_deblock_v = config_.apply_deblock_v;

    const std::vector<std::uint8_t> seq =
        av2enc::still::make_sequence_header_obu(seq_config);
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

  const StreamPlan& plan_;
  StreamConfig config_;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  TileCdfs cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> tile_;
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
  std::vector<std::uint8_t> ibc_;
  std::vector<std::uint8_t> skips_;
  BvNeighbourMap neighbours_;
  IbcBufferState buffer_;
  IbcConfig ibc_config_;
  std::vector<std::uint8_t> recon_y_;
  std::vector<std::uint8_t> recon_u_;
  std::vector<std::uint8_t> recon_v_;
  std::string error_;
  int ibc_blocks_ = 0;
  int newmv_blocks_ = 0;
};

}  // namespace

MiNeighbours mode_info_neighbours(int mi_row, int mi_col, int bsize,
                                  int mi_cols, int mi_rows, int sb_size,
                                  bool buffered) noexcept {
  MiNeighbours out;
  const auto size = static_cast<partition::BlockSize>(bsize);
  const int bw4 = std::max(1, partition::block_width(size) / 4);
  const int bh4 = std::max(1, partition::block_height(size) / 4);
  int sb_log2 = 0;
  int sb_mi = std::max(1, sb_size / 4);
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
  // §5.20.4.1 decode_block(): NPos/NPosBuf insertion order.
  add(mi_row + bh4 - 1, mi_col - 1);
  add(mi_row - 1, mi_col + bw4 - 1);
  add(mi_row, mi_col - 1);
  add(mi_row - 1, mi_col);
  return out;
}

int ibc_symbol_context(const std::vector<std::uint8_t>& grid, int mi_cols,
                       int mi_rows, int mi_row, int mi_col, int bsize,
                       int sb_size) noexcept {
  const MiNeighbours n =
      mode_info_neighbours(mi_row, mi_col, bsize, mi_cols, mi_rows, sb_size,
                           /*buffered=*/false);
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

int skip_symbol_context(const std::vector<std::uint8_t>& grid, int mi_cols,
                        int mi_rows, int mi_row, int mi_col, int bsize,
                        int sb_size) noexcept {
  const MiNeighbours n =
      mode_info_neighbours(mi_row, mi_col, bsize, mi_cols, mi_rows, sb_size,
                           /*buffered=*/true);
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

bool encode_ibc_stream(const StreamPlan& plan, StreamResult& out,
                       std::string& error) {
  error.clear();
  out = StreamResult{};
  const StreamConfig& c = plan.config;
  if (c.width <= 0 || c.height <= 0 || (c.width & 1) != 0 ||
      (c.height & 1) != 0) {
    error = "stream requires even positive dimensions";
    return false;
  }
  if (c.width > 4096 || c.height > 4096) {
    error = "stream dimensions are unreasonably large";
    return false;
  }
  if (c.allow_intrabc &&
      !(c.allow_global_intrabc || c.allow_local_intrabc)) {
    error = "allow_intrabc requires a usable reference region";
    return false;
  }
  if (c.allow_intrabc && !c.allow_screen_content_tools) {
    error = "allow_intrabc requires allow_screen_content_tools";
    return false;
  }
  if (!c.force_integer_mv) {
    error = "the minimal stream writer only supports force_integer_mv=1";
    return false;
  }
  if (c.enable_cfl_intra) {
    error = "the minimal stream writer keeps CfL disabled";
    return false;
  }
  TileWriter writer(plan, out);
  return writer.run(out, error);
}

bool write_ibc_y4m(const StreamResult& result, const StreamConfig& config,
                   const std::string& path, std::string& error) {
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

bool write_ibc_ivf(const StreamResult& result, const StreamConfig& config,
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

}  // namespace av2enc::still::ibc_stream
