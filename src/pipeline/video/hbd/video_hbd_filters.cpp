#include "pipeline/video/hbd/video_hbd_filters.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/deblock/deblock.h"
#include "codec/partition/tree/partition.h"
#include "pipeline/still/still_tables.h"
#include "pipeline/video/post/deblock_grid.h"
#include "formats/high_bitdepth/hbd_ccso.h"
#include "formats/high_bitdepth/hbd_cdef.h"
#include "formats/high_bitdepth/hbd_gdf.h"
#include "formats/high_bitdepth/hbd_lr.h"

namespace av2enc::video::hbd {
namespace {

using av2enc::fmt::Format;

constexpr int kMiSize = 4;

int plane_shift(int plane, int sub) { return plane == 0 ? 0 : sub; }

// §7.17 metadata grid for one key frame.  The key frame has no skipped leaves,
// so every cell is an intra region with the leaf's transform size.
bool build_deblock_grids(const Format& format,
                         std::span<const syntax::BlockSyntax> blocks,
                         int base_q_idx, bool frame_is_intra, int mi_cols,
                         int mi_rows, std::vector<deblock::DeblockMi> mi[3],
                         deblock::DeblockMi* grids[3], std::string& error) {
  const int num_planes = fmt::num_planes(format.chroma);
  if (num_planes < 1 || num_planes > 3) {
    error = "video deblock: invalid plane count";
    return false;
  }
  const std::size_t count =
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows);
  for (int p = 0; p < 3; ++p) {
    mi[p].assign(p < num_planes ? count : 0, deblock::DeblockMi{});
    grids[p] = p < num_planes ? mi[p].data() : nullptr;
  }
  for (const syntax::BlockSyntax& block : blocks) {
    if (block.partition !=
        static_cast<std::uint8_t>(partition::PartitionType::kNone)) {
      continue;
    }
    const auto bsize = static_cast<partition::BlockSize>(block.bsize);
    const int w4 = partition::block_width(bsize);
    const int h4 = partition::block_height(bsize);
    if (w4 <= 0 || h4 <= 0) continue;
    const int bx = block.mi_col;
    const int by = block.mi_row;
    const int clamped_w4 = std::min(w4, mi_cols - bx);
    const int clamped_h4 = std::min(h4, mi_rows - by);
    const bool inter_skip =
        !frame_is_intra && block.is_inter != 0 && block.skip_flag != 0;
    int tx_size[3] = {block.tx_size[0], block.tx_size[1], block.tx_size[2]};
    if (inter_skip) {
      // §7.17.2: a skipped inter leaf uses Max_Tx_Size_Rect for the luma block
      // and for the chroma block its residual size derives from the format.
      tx_size[0] = post::video_deblock_max_tx(block.bsize);
      const auto chroma_size =
          fmt::chroma_residual_size(static_cast<fmt::BlockSize>(block.bsize),
                                    format.chroma);
      const int chroma_max =
          chroma_size ? post::video_deblock_max_tx(static_cast<int>(*chroma_size))
                      : tx_size[0];
      tx_size[1] = chroma_max;
      tx_size[2] = chroma_max;
    }
    if (tx_size[0] < 0) {
      error = "video deblock: no Max_Tx_Size_Rect for a leaf";
      return false;
    }
    for (int p = 0; p < num_planes; ++p) {
      deblock::DeblockMi entry;
      entry.block_col_base = static_cast<std::uint16_t>(bx);
      entry.block_row_base = static_cast<std::uint16_t>(by);
      entry.block_w4 = static_cast<std::uint8_t>(clamped_w4);
      entry.block_h4 = static_cast<std::uint8_t>(clamped_h4);
      entry.tx_size = static_cast<deblock::TxSize>(tx_size[p]);
      entry.sub_pu_size = entry.tx_size;
      entry.tx_col_base = static_cast<std::uint16_t>(bx);
      entry.tx_row_base = static_cast<std::uint16_t>(by);
      entry.sub_pu_col_base = static_cast<std::uint16_t>(bx);
      entry.sub_pu_row_base = static_cast<std::uint16_t>(by);
      entry.skip = inter_skip ? 1 : 0;
      entry.region_intra =
          (frame_is_intra || block.is_inter == 0) ? 1 : 0;
      entry.lossless = 0;
      entry.qindex = block.qindex >= 0
                         ? static_cast<std::uint16_t>(block.qindex)
                         : static_cast<std::uint16_t>(base_q_idx);
      for (int yy = 0; yy < w4 && by + yy < mi_rows; ++yy) {
        for (int xx = 0; xx < w4 && bx + xx < mi_cols; ++xx) {
          grids[p][static_cast<std::size_t>(by + yy) *
                        static_cast<std::size_t>(mi_cols) +
                    static_cast<std::size_t>(bx + xx)] = entry;
        }
      }
    }
  }
  return true;
}

void fill_cdef_plan(const av2enc::hbd::HbdCdefPlan& src,
                    post::VideoCdefPlan* dst) {
  dst->enabled = src.enabled;
  dst->damping = src.damping;
  dst->num_strengths = src.num_strengths;
  dst->on_skip_txfm = false;
  for (int i = 0; i < src.num_strengths && i < 8; ++i) {
    dst->strength[i].y_pri = src.strengths[i].y_pri;
    dst->strength[i].y_sec = src.strengths[i].y_sec;
    dst->strength[i].uv_pri = src.strengths[i].uv_pri;
    dst->strength[i].uv_sec = src.strengths[i].uv_sec;
  }
  dst->n64_cols = src.n64_cols;
  dst->n64_rows = src.n64_rows;
  dst->index = src.index;
}

void fill_ccso_plan(const av2enc::hbd::HbdCcsoConfig& src,
                    post::VideoCcsoPlan* dst) {
  dst->enabled = src.enabled;
  dst->ccso_luma_size_log2 = src.ccso_luma_size_log2;
  for (int p = 0; p < 3; ++p) {
    post::VideoCcsoPlane& out = dst->plane[p];
    const av2enc::hbd::HbdCcsoPlaneConfig& in = src.plane[p];
    out.enabled = in.enabled;
    out.bo_only = in.bo_only;
    out.scale_idx = in.scale_idx;
    out.quant_idx = in.quant_idx;
    out.ext_filter = in.ext_filter;
    out.edge_clf = in.edge_clf;
    out.max_band_log2 = in.max_band_log2;
    out.offset_idx = in.offset_idx;
    out.blks = src.blks[p];
    out.blk_cols = src.blk_cols[p];
    out.blk_rows = src.blk_rows[p];
  }
}

void fill_lr_plan(const av2enc::hbd::HbdLrConfig& src, int base_q_idx,
                  post::VideoLrPlan* dst) {
  dst->enabled = src.enabled;
  dst->pc_wiener =
      src.plane[0].frame_type == av2enc::restoration::kRestorePcWiener;
  dst->tx_skip = src.lr_tx_skip;
  dst->base_q_idx = base_q_idx;
  dst->tx_skip_cols = src.tx_skip_cols;
  dst->tx_skip_rows = src.tx_skip_rows;
  dst->luma_shift = src.luma_shift;
  dst->chroma_shift = src.chroma_shift;
  for (int p = 0; p < 3; ++p) {
    post::VideoLrPlane& out = dst->plane[p];
    const av2enc::hbd::HbdLrPlaneConfig& in = src.plane[p];
    out.enabled = in.enabled;
    out.restore_type = static_cast<std::uint8_t>(in.frame_type);
    out.unit_size = in.unit_size;
    out.shift = in.shift;
    out.unit_rows = in.unit_rows;
    out.unit_cols = in.unit_cols;
    out.unit_type = in.unit_type;
    out.unit_coeffs = in.unit_coeffs;
  }
}

void fill_gdf_plan(const av2enc::hbd::HbdGdfConfig& src, int base_q_idx,
                   post::VideoGdfPlan* dst) {
  dst->enabled = src.enabled;
  dst->qc_idx = src.pic_qc_idx;
  dst->scale_idx = src.pic_scale_idx;
  dst->per_block = false;
  dst->frame_is_intra = true;
  dst->base_q_idx = base_q_idx;
}

}  // namespace

bool apply_frame_filters(
    const Format& format, int mi_cols, int mi_rows, const int visible_w[3],
    const int visible_h[3], std::uint16_t* const recon[3],
    const int recon_stride[3], const FramePlanes& source,
    std::span<const syntax::BlockSyntax> blocks, int base_q_idx,
    bool frame_is_intra, bool enable_deblock,
    const post::VideoLoopFilterOptions& options,
    post::VideoFilterPlan* plan, std::string& error) {
  error.clear();
  if (plan == nullptr) {
    error = "video filters: null plan";
    return false;
  }
  *plan = post::VideoFilterPlan{};
  const int num_planes = fmt::num_planes(format.chroma);
  const int sub_x = fmt::subsampling_x(format.chroma);
  const int sub_y = fmt::subsampling_y(format.chroma);
  const int bit_depth = fmt::bit_depth_value(format.depth);
  if (num_planes < 1 || num_planes > 3) {
    error = "video filters: invalid chroma format";
    return false;
  }
  // §7.18-§7.20.5 are planned on the intra key frame.  An inter frame keeps
  // them disabled in the frame header so no skip grid is needed.
  const post::VideoLoopFilterOptions tools =
      frame_is_intra ? options : post::VideoLoopFilterOptions{};
  const bool any = enable_deblock || tools.any();
  if (!any) return true;

  const bool need_curr = frame_is_intra &&
                         (tools.ccso || tools.cdef || tools.wiener ||
                          tools.chroma_lr || tools.pc_wiener || tools.gdf);

  // §7.17 deblock, in place.  Runs before the copy so §7.19/§7.20 sample the
  // post-deblock CurrFrame.
  if (enable_deblock) {
    std::vector<deblock::DeblockMi> mi[3];
    deblock::DeblockMi* grids[3] = {nullptr, nullptr, nullptr};
    if (!build_deblock_grids(format, blocks, base_q_idx, frame_is_intra,
                             mi_cols, mi_rows, mi, grids, error)) {
      return false;
    }
    deblock::DeblockFrame16 frame{};
    for (int p = 0; p < num_planes; ++p) {
      frame.planes[p].data = recon[p];
      frame.planes[p].stride = recon_stride[p];
      frame.mi[p] = grids[p];
    }
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.subsampling_x = sub_x;
    frame.subsampling_y = sub_y;
    frame.num_planes = num_planes;
    deblock::DeblockOptions deblock_options;
    deblock_options.apply_deblocking_filter[0] = true;
    deblock_options.apply_deblocking_filter[1] = true;
    deblock_options.apply_deblocking_filter[2] = num_planes > 1;
    deblock_options.apply_deblocking_filter[3] = num_planes > 2;
    deblock_options.frame_is_intra = true;
    deblock_options.bit_depth = bit_depth;
    const deblock::DeblockStatus status =
        deblock::deblock_frame16(frame, deblock_options);
    if (status != deblock::DeblockStatus::kOk) {
      error = std::string("video deblock failed: ") +
              deblock::deblock_status_string(status);
      return false;
    }
  }

  if (!need_curr) return true;

  std::vector<std::uint16_t> curr[3];
  const int coded_w[3] = {
      (mi_cols * kMiSize), (mi_cols * kMiSize) >> sub_x,
      (mi_cols * kMiSize) >> sub_x};
  const int coded_h[3] = {
      (mi_rows * kMiSize), (mi_rows * kMiSize) >> sub_y,
      (mi_rows * kMiSize) >> sub_y};
  for (int p = 0; p < num_planes; ++p) {
    curr[p].assign(static_cast<std::size_t>(coded_w[p]) *
                       static_cast<std::size_t>(coded_h[p]),
                   0);
    for (int y = 0; y < coded_h[p]; ++y) {
      for (int x = 0; x < coded_w[p]; ++x) {
        curr[p][static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(coded_w[p]) +
                static_cast<std::size_t>(x)] =
            recon[p][static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(recon_stride[p]) +
                     static_cast<std::size_t>(x)];
      }
    }
  }

  // §7.18 CDEF.
  std::vector<std::uint16_t> cdef_samples[3];
  av2enc::hbd::HbdCdefPlan cdef_plan;
  if (tools.cdef) {
    av2enc::hbd::HbdFilterFrame16 frame;
    for (int p = 0; p < num_planes; ++p) {
      frame.src[p] = curr[p].data();
      frame.ref[p] = source.plane[p].data;
      frame.src_stride[p] = coded_w[p];
      frame.ref_stride[p] = source.plane[p].stride;
      frame.visible_w[p] = visible_w[p];
      frame.visible_h[p] = visible_h[p];
    }
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.sub_x = sub_x;
    frame.sub_y = sub_y;
    frame.num_planes = num_planes;
    frame.bit_depth = bit_depth;
    if (!av2enc::hbd::plan_cdef(frame, tools.cdef_damping,
                                tools.cdef_num_strengths, &cdef_plan,
                                &error)) {
      return false;
    }
    const std::uint16_t* src_planes[3] = {nullptr, nullptr, nullptr};
    std::uint16_t* dst_planes[3] = {nullptr, nullptr, nullptr};
    int stride[3] = {0, 0, 0};
    for (int p = 0; p < num_planes; ++p) {
      src_planes[p] = curr[p].data();
      stride[p] = coded_w[p];
      cdef_samples[p].assign(
          static_cast<std::size_t>(coded_w[p]) *
              static_cast<std::size_t>(coded_h[p]),
          0);
      dst_planes[p] = cdef_samples[p].data();
    }
    if (!av2enc::hbd::apply_cdef16(cdef_plan, format, src_planes, dst_planes,
                                   stride, mi_cols, mi_rows, &error)) {
      return false;
    }
    for (int p = 0; p < num_planes; ++p) {
      for (int y = 0; y < coded_h[p]; ++y) {
        for (int x = 0; x < coded_w[p]; ++x) {
          recon[p][static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(recon_stride[p]) +
                   static_cast<std::size_t>(x)] =
              cdef_samples[p][static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(coded_w[p]) +
                              static_cast<std::size_t>(x)];
        }
      }
    }
    fill_cdef_plan(cdef_plan, &plan->cdef);
  }

  // §7.19 CCSO.
  av2enc::hbd::HbdCcsoPlan ccso_plan;
  if (tools.ccso) {
    std::vector<std::uint16_t> cdef16[3];
    for (int p = 0; p < num_planes; ++p) {
      cdef16[p].assign(
          static_cast<std::size_t>(coded_w[p]) *
              static_cast<std::size_t>(coded_h[p]),
          0);
      for (int y = 0; y < coded_h[p]; ++y) {
        for (int x = 0; x < coded_w[p]; ++x) {
          cdef16[p][static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(coded_w[p]) +
                    static_cast<std::size_t>(x)] =
              recon[p][static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(recon_stride[p]) +
                       static_cast<std::size_t>(x)];
        }
      }
    }
    av2enc::hbd::HbdCcsoFrame16 frame;
    const std::uint16_t* curr_planes[3] = {nullptr, nullptr, nullptr};
    std::uint16_t* out_planes[3] = {nullptr, nullptr, nullptr};
    int stride[3] = {0, 0, 0};
    for (int p = 0; p < num_planes; ++p) {
      frame.curr[p] = curr[p].data();
      frame.cdef[p] = cdef16[p].data();
      frame.ref[p] = source.plane[p].data;
      frame.curr_stride[p] = coded_w[p];
      frame.cdef_stride[p] = coded_w[p];
      frame.ref_stride[p] = source.plane[p].stride;
      frame.visible_w[p] = visible_w[p];
      frame.visible_h[p] = visible_h[p];
      curr_planes[p] = curr[p].data();
      out_planes[p] = cdef16[p].data();
      stride[p] = coded_w[p];
    }
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.sub_x = sub_x;
    frame.sub_y = sub_y;
    frame.num_planes = num_planes;
    frame.bit_depth = bit_depth;
    // `--video-ccso-bands` selects the searched band range in the M2 8-bit
    // planner; the shared still planner searches its own candidate ladder and
    // takes only the CCSO unit size (6 = the 64x64 superblock).
    (void)tools.ccso_bands_log2;
    if (!av2enc::hbd::plan_ccso(frame, 6, false, &ccso_plan, &error)) {
      return false;
    }
    if (!av2enc::hbd::apply_ccso16(ccso_plan, format, curr_planes, out_planes,
                                   stride, stride, mi_cols, mi_rows, &error)) {
      return false;
    }
    for (int p = 0; p < num_planes; ++p) {
      for (int y = 0; y < coded_h[p]; ++y) {
        for (int x = 0; x < coded_w[p]; ++x) {
          recon[p][static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(recon_stride[p]) +
                   static_cast<std::size_t>(x)] =
              cdef16[p][static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(coded_w[p]) +
                        static_cast<std::size_t>(x)];
        }
      }
    }
    fill_ccso_plan(ccso_plan.config, &plan->ccso);
  }

  // §7.20 loop restoration + §7.20.5 GDF.
  const bool any_lr =
      tools.wiener || tools.chroma_lr || tools.pc_wiener;
  if (any_lr || tools.gdf) {
    av2enc::hbd::HbdLrFrame16 lr_frame;
    for (int p = 0; p < num_planes; ++p) {
      lr_frame.curr[p] = curr[p].data();
      lr_frame.cdef[p] = recon[p];
      lr_frame.ref[p] = source.plane[p].data;
      lr_frame.curr_stride[p] = coded_w[p];
      lr_frame.cdef_stride[p] = recon_stride[p];
      lr_frame.ref_stride[p] = source.plane[p].stride;
      lr_frame.visible_w[p] = visible_w[p];
      lr_frame.visible_h[p] = visible_h[p];
    }
    lr_frame.mi_cols = mi_cols;
    lr_frame.mi_rows = mi_rows;
    lr_frame.sub_x = sub_x;
    lr_frame.sub_y = sub_y;
    lr_frame.num_planes = num_planes;
    lr_frame.bit_depth = bit_depth;
    lr_frame.base_q_idx = base_q_idx;
    lr_frame.cfl_ds_filter_index = tools.cfl_ds_filter_index;

    // §7.20.1 LrTxSkip: every key-frame leaf with luma eob == 0.
    std::vector<std::uint8_t> lr_tx_skip(
        static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows),
        0);
    if (tools.pc_wiener) {
      for (const syntax::BlockSyntax& block : blocks) {
        if (block.partition !=
            static_cast<std::uint8_t>(partition::PartitionType::kNone)) {
          continue;
        }
        if (block.eob[0] != 0) continue;
        const auto bsize = static_cast<partition::BlockSize>(block.bsize);
        const int w4 = partition::block_width(bsize);
        const int h4 = partition::block_height(bsize);
        for (int yy = 0; yy < h4; ++yy) {
          for (int xx = 0; xx < w4; ++xx) {
            const int r = block.mi_row + yy;
            const int c = block.mi_col + xx;
            if (r < mi_rows && c < mi_cols) {
              lr_tx_skip[static_cast<std::size_t>(r) *
                             static_cast<std::size_t>(mi_cols) +
                         static_cast<std::size_t>(c)] = 1;
            }
          }
        }
      }
    }

    av2enc::hbd::HbdLrPlan lr_plan;
    if (any_lr) {
      if (!av2enc::hbd::plan_lr(lr_frame, tools.wiener, tools.chroma_lr,
                                tools.pc_wiener, base_q_idx, &lr_plan,
                                &error)) {
        return false;
      }
      lr_plan.config.lr_tx_skip = lr_tx_skip;
      lr_plan.config.tx_skip_cols = mi_cols;
      lr_plan.config.tx_skip_rows = mi_rows;
      if (tools.pc_wiener) {
        std::vector<std::uint16_t> scratch(
            static_cast<std::size_t>(coded_w[0]) *
            static_cast<std::size_t>(coded_h[0]));
        av2enc::hbd::plan_pc_wiener16(lr_frame, &lr_plan, lr_tx_skip,
                                      scratch.data());
      }
    }

    std::vector<std::uint16_t> lr16[3];
    std::uint16_t* lr_ptr[3] = {nullptr, nullptr, nullptr};
    int lr_stride[3] = {0, 0, 0};
    for (int p = 0; p < num_planes; ++p) {
      lr16[p].assign(
          static_cast<std::size_t>(coded_w[p]) *
              static_cast<std::size_t>(coded_h[p]),
          0);
      lr_ptr[p] = lr16[p].data();
      lr_stride[p] = coded_w[p];
    }
    if (any_lr && lr_plan.config.enabled) {
      if (!av2enc::hbd::apply_wiener16(lr_frame, lr_plan.config, lr_ptr,
                                       lr_stride, &error)) {
        return false;
      }
      if (tools.pc_wiener && lr_plan.config.plane[0].enabled) {
        av2enc::hbd::apply_pc_wiener16(lr_frame, lr_plan.config,
                                       lr_plan.config.lr_tx_skip, lr_ptr[0],
                                       lr_stride[0]);
      }
    } else {
      for (int p = 0; p < num_planes; ++p) {
        for (int y = 0; y < coded_h[p]; ++y) {
          for (int x = 0; x < coded_w[p]; ++x) {
            lr16[p][static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(coded_w[p]) +
                    static_cast<std::size_t>(x)] =
                recon[p][static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(recon_stride[p]) +
                         static_cast<std::size_t>(x)];
          }
        }
      }
    }

    av2enc::hbd::HbdGdfPlan gdf_plan;
    if (tools.gdf) {
      av2enc::hbd::HbdGdfFrame16 gdf_frame;
      gdf_frame.curr = curr[0].data();
      gdf_frame.cdef = recon[0];
      gdf_frame.ref = source.plane[0].data;
      gdf_frame.curr_stride = coded_w[0];
      gdf_frame.cdef_stride = recon_stride[0];
      gdf_frame.ref_stride = source.plane[0].stride;
      gdf_frame.width = visible_w[0];
      gdf_frame.height = visible_h[0];
      gdf_frame.mi_cols = mi_cols;
      gdf_frame.mi_rows = mi_rows;
      gdf_frame.bit_depth = bit_depth;
      gdf_frame.base_q_idx = base_q_idx;
      if (!av2enc::hbd::plan_gdf(gdf_frame, &gdf_plan, &error)) return false;
      if (!av2enc::hbd::apply_gdf16(gdf_plan, gdf_frame, lr_ptr[0],
                                    lr_stride[0], &error)) {
        return false;
      }
    }

    for (int p = 0; p < num_planes; ++p) {
      for (int y = 0; y < coded_h[p]; ++y) {
        for (int x = 0; x < coded_w[p]; ++x) {
          recon[p][static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(recon_stride[p]) +
                   static_cast<std::size_t>(x)] =
              lr16[p][static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(coded_w[p]) +
                      static_cast<std::size_t>(x)];
        }
      }
    }
    if (lr_plan.config.enabled) fill_lr_plan(lr_plan.config, base_q_idx,
                                             &plan->lr);
    if (gdf_plan.config.enabled) fill_gdf_plan(gdf_plan.config, base_q_idx,
                                               &plan->gdf);
  }

  return true;
}

}  // namespace av2enc::video::hbd
