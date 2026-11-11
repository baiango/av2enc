#include "encoder/m2/post/loop_filters.h"

#include "encoder/filters/restoration/restoration_common.h"

#include <algorithm>
#include <cstring>

namespace av2enc::m2::post {
namespace {

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

int ceil_div(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

}  // namespace

const char* post_filter_stage_name(M2PostFilterStage stage) noexcept {
  switch (stage) {
    case M2PostFilterStage::kDeblock:
      return "deblock";
    case M2PostFilterStage::kCdef:
      return "cdef";
    case M2PostFilterStage::kCcso:
      return "ccso";
  }
  return "unknown";
}

int M2PostFilterFrame::plane_width(int plane) const {
  return (mi_cols_ * 4) >> (plane == 0 ? 0 : sub_x_);
}

int M2PostFilterFrame::plane_height(int plane) const {
  return (mi_rows_ * 4) >> (plane == 0 ? 0 : sub_y_);
}

bool M2PostFilterFrame::init_from_recon(const M2ReconFrame& recon,
                                        int base_q_idx, bool frame_is_intra,
                                        std::string* error) {
  if (recon.width == 0 || recon.height == 0) {
    return fail(error, "m2 post: empty reconstruction");
  }
  if ((recon.width & 7U) != 0 || (recon.height & 7U) != 0) {
    return fail(error,
                "m2 post: reconstruction dimensions must be divisible by 8");
  }
  const std::size_t luma = static_cast<std::size_t>(recon.width) * recon.height;
  const std::size_t chroma =
      static_cast<std::size_t>(recon.width / 2) * (recon.height / 2);
  if (recon.y.size() != luma || recon.u.size() != chroma ||
      recon.v.size() != chroma) {
    return fail(error, "m2 post: reconstruction plane size mismatch");
  }

  mi_cols_ = static_cast<int>(recon.width) / 4;
  mi_rows_ = static_cast<int>(recon.height) / 4;
  sub_x_ = 1;
  sub_y_ = 1;
  bit_depth_ = 8;
  base_q_idx_ = base_q_idx;
  frame_is_intra_ = frame_is_intra;
  sb_size_ = 64;

  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane_width(plane);
    const int h = plane_height(plane);
    stride_[plane] = w;
    plane_[plane].assign(static_cast<std::size_t>(w) *
                            static_cast<std::size_t>(h), 0);
    const std::vector<std::uint8_t>& src =
        plane == 0 ? recon.y : (plane == 1 ? recon.u : recon.v);
    const int src_stride =
        plane == 0 ? static_cast<int>(recon.width)
                   : static_cast<int>(recon.width / 2);
    for (int y = 0; y < h; ++y) {
      std::memcpy(plane_[plane].data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(w),
                  src.data() + static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(src_stride),
                  static_cast<std::size_t>(w));
    }
    output_[plane].clear();
    cdef_out_[plane].clear();
  }

  const std::size_t mi_cells =
      static_cast<std::size_t>(mi_cols_) * static_cast<std::size_t>(mi_rows_);
  for (int plane = 0; plane < 3; ++plane) {
    deblock_mi_[plane].assign(mi_cells, deblock::DeblockMi{});
    for (int r = 0; r < mi_rows_; ++r) {
      for (int c = 0; c < mi_cols_; ++c) {
        deblock::DeblockMi& mi =
            deblock_mi_[plane][static_cast<std::size_t>(r) *
                                   static_cast<std::size_t>(mi_cols_) +
                               static_cast<std::size_t>(c)];
        mi.block_col_base = static_cast<std::uint16_t>(c);
        mi.block_row_base = static_cast<std::uint16_t>(r);
        mi.block_w4 = 1;
        mi.block_h4 = 1;
        mi.sub_pu_size = deblock::TxSize::k4x4;
        mi.tx_size = deblock::TxSize::k4x4;
        mi.tx_col_base = static_cast<std::uint16_t>(c);
        mi.tx_row_base = static_cast<std::uint16_t>(r);
        mi.sub_pu_col_base = static_cast<std::uint16_t>(c);
        mi.sub_pu_row_base = static_cast<std::uint16_t>(r);
        mi.skip = 0;
        mi.region_intra = frame_is_intra_ ? 1 : 0;
        mi.lossless = 0;
        mi.qindex = static_cast<std::uint8_t>(
            std::min(std::max(base_q_idx, 0), 255));
      }
    }
  }

  skip_.assign(mi_cells, 0);
  lossless_.assign(mi_cells, 0);
  lossless_chroma_.assign(mi_cells, 0);
  n64_cols_ = ceil_div(mi_cols_, 16);
  n64_rows_ = ceil_div(mi_rows_, 16);
  cdef_idx_.assign(static_cast<std::size_t>(n64_cols_) *
                       static_cast<std::size_t>(n64_rows_),
                   0);

  for (int plane = 0; plane < 3; ++plane) {
    int cols = 0;
    int rows = 0;
    restoration::ccso_blk_grid_size(
        plane_width(plane), plane_height(plane), 6,
        plane == 0 ? 0 : sub_x_, plane == 0 ? 0 : sub_y_, &cols, &rows);
    ccso_blk_cols_[plane] = cols;
    ccso_blk_rows_[plane] = rows;
    ccso_blk_[plane].assign(static_cast<std::size_t>(cols) *
                                static_cast<std::size_t>(rows), 0);
  }

  tile_col_starts_.clear();
  tile_row_starts_.clear();
  mi_col_start_.clear();
  mi_col_end_.clear();
  mi_row_start_.clear();
  mi_row_end_.clear();
  return true;
}

bool M2PostFilterFrame::set_tile_layout(const tiles::TileLayout& layout,
                                        std::string* error) {
  if (mi_cols_ == 0 || mi_rows_ == 0) {
    return fail(error, "m2 post: frame is not initialised");
  }
  if (layout.width != mi_cols_ * 4 || layout.height != mi_rows_ * 4) {
    return fail(error, "m2 post: tile layout geometry does not match the frame");
  }
  if (layout.col_starts_sb.empty() || layout.row_starts_sb.empty()) {
    return fail(error, "m2 post: tile layout has no starts");
  }
  const int sb_mi = layout.sb_size / 4;
  tile_col_starts_.clear();
  tile_row_starts_.clear();
  for (int sb : layout.col_starts_sb) {
    tile_col_starts_.push_back(static_cast<std::uint16_t>(sb * sb_mi));
  }
  for (int sb : layout.row_starts_sb) {
    tile_row_starts_.push_back(static_cast<std::uint16_t>(sb * sb_mi));
  }

  const std::size_t mi_cells =
      static_cast<std::size_t>(mi_cols_) * static_cast<std::size_t>(mi_rows_);
  mi_col_start_.assign(mi_cells, 0);
  mi_col_end_.assign(mi_cells, static_cast<std::uint16_t>(mi_cols_));
  mi_row_start_.assign(mi_cells, 0);
  mi_row_end_.assign(mi_cells, static_cast<std::uint16_t>(mi_rows_));
  for (int r = 0; r < mi_rows_; ++r) {
    int row_start = 0;
    int row_end = mi_rows_;
    for (std::size_t i = 0; i < tile_row_starts_.size(); ++i) {
      const int start = tile_row_starts_[i];
      const int end = i + 1 < tile_row_starts_.size()
                          ? tile_row_starts_[i + 1]
                          : mi_rows_;
      if (r >= start && r < end) {
        row_start = start;
        row_end = end;
        break;
      }
    }
    for (int c = 0; c < mi_cols_; ++c) {
      int col_start = 0;
      int col_end = mi_cols_;
      for (std::size_t i = 0; i < tile_col_starts_.size(); ++i) {
        const int start = tile_col_starts_[i];
        const int end = i + 1 < tile_col_starts_.size()
                            ? tile_col_starts_[i + 1]
                            : mi_cols_;
        if (c >= start && c < end) {
          col_start = start;
          col_end = end;
          break;
        }
      }
      const std::size_t idx =
          static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
          static_cast<std::size_t>(c);
      mi_col_start_[idx] = static_cast<std::uint16_t>(col_start);
      mi_col_end_[idx] = static_cast<std::uint16_t>(col_end);
      mi_row_start_[idx] = static_cast<std::uint16_t>(row_start);
      mi_row_end_[idx] = static_cast<std::uint16_t>(row_end);
    }
  }
  return true;
}

bool M2PostFilterFrame::set_deblock_block(
    int plane, int mi_row, int mi_col, int w4, int h4, int sub_pu_w4,
    int sub_pu_h4, bool skip, bool region_intra, int qindex,
    deblock::TxSize tx_size) {
  if (plane < 0 || plane > 2 || w4 <= 0 || h4 <= 0) return false;
  if (mi_row < 0 || mi_col < 0) return false;
  const int max_w = std::min(w4, mi_cols_ - mi_col);
  const int max_h = std::min(h4, mi_rows_ - mi_row);
  if (max_w <= 0 || max_h <= 0) return false;
  const int q = std::min(std::max(qindex, 0), 255);
  const int tx_w4 =
      std::max(1, deblock::tx_width(tx_size) / deblock::kMiSize);
  const int tx_h4 =
      std::max(1, deblock::tx_height(tx_size) / deblock::kMiSize);
  for (int r = 0; r < max_h; ++r) {
    for (int c = 0; c < max_w; ++c) {
      deblock::DeblockMi& mi =
          deblock_mi_[plane][static_cast<std::size_t>(mi_row + r) *
                                   static_cast<std::size_t>(mi_cols_) +
                               static_cast<std::size_t>(mi_col + c)];
      mi.block_col_base = static_cast<std::uint16_t>(mi_col);
      mi.block_row_base = static_cast<std::uint16_t>(mi_row);
      mi.block_w4 = static_cast<std::uint8_t>(std::min(w4, 255));
      mi.block_h4 = static_cast<std::uint8_t>(std::min(h4, 255));
      mi.sub_pu_size = tx_size;
      mi.tx_size = tx_size;
      // Every 4x4 cell points at the transform block that contains it.  A
      // 4x4 TX therefore makes every MI edge a transform edge.
      mi.tx_col_base = static_cast<std::uint16_t>(
          mi_col + (c / tx_w4) * tx_w4);
      mi.tx_row_base = static_cast<std::uint16_t>(
          mi_row + (r / tx_h4) * tx_h4);
      mi.sub_pu_col_base = static_cast<std::uint16_t>(mi_col);
      mi.sub_pu_row_base = static_cast<std::uint16_t>(mi_row);
      mi.skip = skip ? 1 : 0;
      mi.region_intra = region_intra ? 1 : 0;
      mi.qindex = static_cast<std::uint8_t>(q);
    }
  }
  // sub_pu_w4/h4 are accepted for future sub-PU filtering; the current
  // DeblockMi stores them implicitly in sub_pu_size/sub_pu bases.
  (void)sub_pu_w4;
  (void)sub_pu_h4;
  return true;
}

void M2PostFilterFrame::fill_deblock_rect(int mi_row, int mi_col, int w4,
                                          int h4, bool skip,
                                          bool region_intra, int qindex) {
  for (int plane = 0; plane < 3; ++plane) {
    (void)set_deblock_block(plane, mi_row, mi_col, w4, h4, w4, h4, skip,
                            region_intra, qindex, deblock::TxSize::k4x4);
  }
}

bool run_m2_post_filters(M2PostFilterFrame& frame,
                         const M2PostFilterConfig& config,
                         M2PostFilterReport& report, std::string* error) {
  report = M2PostFilterReport{};
  report.bit_depth = frame.bit_depth_;
  if (frame.mi_cols_ <= 0 || frame.mi_rows_ <= 0) {
    return fail(error, "m2 post: frame is not initialised");
  }
  if (frame.bit_depth_ != 8) {
    return fail(error,
                "m2 post: deblock/CDEF/CCSO are 8-bit only; keep loop filters "
                "off for a 10-bit frame");
  }
  if ((frame.mi_cols_ & 1) != 0 || (frame.mi_rows_ & 1) != 0) {
    return fail(error, "m2 post: the 4:2:0 MI grid must be even");
  }

  // Start from the reconstruction (or the previous stage's output).
  for (int plane = 0; plane < 3; ++plane) {
    frame.output_[plane] = frame.plane_[plane];
    frame.cdef_out_[plane].clear();
  }

  // --- 1. deblock (in place on the reconstruction) -----------------------
  if (config.enable_deblock) {
    deblock::DeblockFrame df;
    for (int plane = 0; plane < 3; ++plane) {
      df.planes[plane].data = frame.plane_[plane].data();
      df.planes[plane].stride = frame.stride_[plane];
      df.mi[plane] = frame.deblock_mi_[plane].data();
    }
    df.mi_cols = frame.mi_cols_;
    df.mi_rows = frame.mi_rows_;
    df.subsampling_x = frame.sub_x_;
    df.subsampling_y = frame.sub_y_;
    if (!frame.tile_col_starts_.empty()) {
      df.tile_col_starts = frame.tile_col_starts_.data();
      df.num_tile_cols = static_cast<int>(frame.tile_col_starts_.size());
    }
    if (!frame.tile_row_starts_.empty()) {
      df.tile_row_starts = frame.tile_row_starts_.data();
      df.num_tile_rows = static_cast<int>(frame.tile_row_starts_.size());
    }
    deblock::DeblockOptions options = config.deblock;
    options.disable_loopfilters_across_tiles =
        config.disable_loopfilters_across_tiles;
    options.bit_depth = 8;
    const deblock::DeblockStatus status = deblock::deblock_frame(df, options);
    if (status != deblock::DeblockStatus::kOk) {
      return fail(error, std::string("m2 post: deblock failed: ") +
                             deblock::deblock_status_string(status));
    }
    for (int plane = 0; plane < 3; ++plane) {
      frame.output_[plane] = frame.plane_[plane];
    }
    report.deblock_ran = true;
    report.order.push_back(M2PostFilterStage::kDeblock);
  }

  // --- 2. CDEF (reads post-deblock, writes a separate buffer) ------------
  if (config.enable_cdef) {
    cdef::CdefFrame cf;
    for (int plane = 0; plane < 3; ++plane) {
      cf.planes[plane].src = frame.plane_[plane].data();
      frame.cdef_out_[plane].assign(frame.plane_[plane].size(), 0);
      cf.planes[plane].dst = frame.cdef_out_[plane].data();
      cf.planes[plane].stride = frame.stride_[plane];
    }
    cf.mi_cols = frame.mi_cols_;
    cf.mi_rows = frame.mi_rows_;
    cf.subsampling_x = frame.sub_x_;
    cf.subsampling_y = frame.sub_y_;
    cf.skip = frame.skip_.data();
    cf.lossless = frame.lossless_.data();
    cf.lossless_chroma = frame.lossless_chroma_.data();
    cf.cdef_idx = frame.cdef_idx_.data();
    cf.n64_cols = frame.n64_cols_;
    cf.n64_rows = frame.n64_rows_;
    if (!frame.tile_col_starts_.empty()) {
      cf.tile_col_starts = frame.tile_col_starts_.data();
      cf.num_tile_cols = static_cast<int>(frame.tile_col_starts_.size());
    }
    if (!frame.tile_row_starts_.empty()) {
      cf.tile_row_starts = frame.tile_row_starts_.data();
      cf.num_tile_rows = static_cast<int>(frame.tile_row_starts_.size());
    }
    cdef::CdefOptions options = config.cdef;
    options.disable_loopfilters_across_tiles =
        config.disable_loopfilters_across_tiles;
    options.bit_depth = 8;
    const cdef::CdefStatus status = cdef::cdef_frame(cf, options);
    if (status != cdef::CdefStatus::kOk) {
      return fail(error, std::string("m2 post: CDEF failed: ") +
                             cdef::cdef_status_string(status));
    }
    report.cdef_ran = true;
    report.order.push_back(M2PostFilterStage::kCdef);
  }

  // --- 3. CCSO (post-CDEF buffer, in place) ------------------------------
  if (config.enable_ccso) {
    if (config.disable_loopfilters_across_tiles &&
        frame.mi_col_start_.empty()) {
      return fail(error,
                  "m2 post: disable_loopfilters_across_tiles needs "
                  "set_tile_layout() for the CCSO MiCol/MiRow bounds");
    }
    if (!report.cdef_ran) {
      // CCSO needs the post-CDEF buffer; with CDEF disabled the normative
      // buffer is the deblocked frame, so make an explicit copy.
      for (int plane = 0; plane < 3; ++plane) {
        frame.cdef_out_[plane] = frame.plane_[plane];
      }
    }
    restoration::RestorationFrame rf;
    for (int plane = 0; plane < 3; ++plane) {
      rf.curr[plane].data = frame.plane_[plane].data();
      rf.curr[plane].stride = frame.stride_[plane];
      rf.cdef[plane].data = frame.cdef_out_[plane].data();
      rf.cdef[plane].stride = frame.stride_[plane];
    }
    rf.mi_cols = frame.mi_cols_;
    rf.mi_rows = frame.mi_rows_;
    rf.subsampling_x = frame.sub_x_;
    rf.subsampling_y = frame.sub_y_;
    rf.bit_depth = 8;
    rf.sb_width = 64;
    rf.sb_height = 64;
    rf.sb_shift = 4;
    rf.disable_loopfilters_across_tiles =
        config.disable_loopfilters_across_tiles ? 1 : 0;
    if (rf.disable_loopfilters_across_tiles != 0) {
      rf.mi_col_start = frame.mi_col_start_.data();
      rf.mi_col_end = frame.mi_col_end_.data();
      rf.mi_row_start = frame.mi_row_start_.data();
      rf.mi_row_end = frame.mi_row_end_.data();
    }

    restoration::CcsoFrameParams params = config.ccso;
    for (int plane = 0; plane < 3; ++plane) {
      if (params.blks[plane] == nullptr) {
        params.blks[plane] = frame.ccso_blk_[plane].data();
        params.blk_cols[plane] = frame.ccso_blk_cols_[plane];
        params.blk_rows[plane] = frame.ccso_blk_rows_[plane];
      }
    }
    const restoration::RestorationStatus status =
        restoration::ccso_frame(rf, params);
    if (status != restoration::RestorationStatus::kOk) {
      return fail(error, std::string("m2 post: CCSO failed: ") +
                             restoration::restoration_status_string(status));
    }
    report.ccso_ran = true;
    report.order.push_back(M2PostFilterStage::kCcso);
  }

  // Publish the output: post-CDEF/CCSO when either ran, else post-deblock.
  if (report.cdef_ran || report.ccso_ran) {
    for (int plane = 0; plane < 3; ++plane) {
      frame.output_[plane] = frame.cdef_out_[plane];
    }
  }
  return true;
}

std::size_t write_m2_deblock_params(
    BitWriter& writer, const bool apply_deblocking_filter[4],
    const bool df_delta_q_present[4], const int df_delta_q[4],
    int df_par_bits_minus_2, std::string* error) {
  if (apply_deblocking_filter == nullptr) {
    return 0;
  }
  if (df_par_bits_minus_2 < 0 || df_par_bits_minus_2 > 3) {
    fail(error, "m2 post: df_par_bits_minus_2 is f(2), so it must be 0..3");
    return 0;
  }
  const int par_bits = df_par_bits_minus_2 + 2;
  const int bias = 1 << (par_bits - 1);
  const int min_delta = -bias;
  const int max_delta = bias - 1;
  const bool have_chroma =
      apply_deblocking_filter[0] || apply_deblocking_filter[1];
  for (int plane = 0; plane < 4; ++plane) {
    const bool enabled =
        plane < 2 ? apply_deblocking_filter[plane]
                  : (have_chroma && apply_deblocking_filter[plane]);
    const bool present =
        enabled && df_delta_q_present != nullptr &&
        df_delta_q_present[plane];
    if (!enabled && df_delta_q_present != nullptr &&
        df_delta_q_present[plane]) {
      fail(error,
           "m2 post: df_delta_q_present is only coded for an enabled plane");
      return 0;
    }
    if (present) {
      if (df_delta_q == nullptr) {
        fail(error, "m2 post: df_delta_q_present needs df_delta_q values");
        return 0;
      }
      if (df_delta_q[plane] < min_delta || df_delta_q[plane] > max_delta) {
        fail(error,
             "m2 post: df_delta_q is outside the dfParBits range for this "
             "sequence");
        return 0;
      }
    }
  }
  // Validation is complete; now write the exact §5.18.7.9 row.  The spec
  // interleaves present/delta per plane, so a present bit is always followed
  // by its delta before the next plane.
  const std::size_t start = writer.bit_size();
  writer.put_bit(apply_deblocking_filter[0]);
  writer.put_bit(apply_deblocking_filter[1]);
  if (have_chroma) {
    writer.put_bit(apply_deblocking_filter[2]);
    writer.put_bit(apply_deblocking_filter[3]);
  }
  for (int plane = 0; plane < 4; ++plane) {
    const bool enabled =
        plane < 2 ? apply_deblocking_filter[plane]
                  : (have_chroma && apply_deblocking_filter[plane]);
    if (!enabled) continue;
    const bool present =
        df_delta_q_present != nullptr && df_delta_q_present[plane];
    writer.put_bit(present);
    if (present) {
      const int coded = df_delta_q[plane] + bias;
      writer.put_bits(static_cast<std::uint64_t>(coded),
                      static_cast<unsigned>(par_bits));
    }
  }
  return writer.bit_size() - start;
}

std::size_t write_m2_deblock_params(
    BitWriter& writer, const bool apply_deblocking_filter[4],
    const bool df_delta_q_present[4], std::string* error) {
  static const int kZeroDelta[4] = {0, 0, 0, 0};
  return write_m2_deblock_params(writer, apply_deblocking_filter,
                                 df_delta_q_present, kZeroDelta, 0, error);
}

}  // namespace av2enc::m2::post
