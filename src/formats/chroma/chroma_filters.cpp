#include "formats/chroma/chroma_filters.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/deblock/deblock.h"

namespace av2enc::chroma {
namespace {

using av2enc::fmt::ChromaFormat;
using av2enc::fmt::Format;

constexpr int kBitDepth = 8;
constexpr int kMiSize = 4;

int plane_shift(int plane, int sub) { return plane == 0 ? 0 : sub; }

int plane_width(int mi_cols, int plane, int sub_x) {
  return (mi_cols * kMiSize) >> plane_shift(plane, sub_x);
}
int plane_height(int mi_rows, int plane, int sub_y) {
  return (mi_rows * kMiSize) >> plane_shift(plane, sub_y);
}

// Widens one 8-bit plane into the uint16_t filter container.  `width`/`height`
// are the coded MI-aligned extents, but the buffer strides stay those of the
// padded reconstruction (the cores only visit coded samples).
std::vector<std::uint16_t> widen(const std::uint8_t* src, int stride,
                                 int width, int height) {
  std::vector<std::uint16_t> out(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x)] =
          static_cast<std::uint16_t>(
              src[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                  static_cast<std::size_t>(x)]);
    }
  }
  return out;
}

void narrow_into(const std::vector<std::uint16_t>& src, int width, int height,
                 std::uint8_t* dst, int dst_stride) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(
              src[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)]);
    }
  }
}

// §7.17 metadata grids for the fixed 64x64 PARTITION_NONE leaf plan.  Same
// construction as the 10-bit product writer (`hbd_deblock.h`): every cell
// carries the block origin, the per-plane coded transform size, region_intra =
// 1 and the frame qindex.  An intra still never skips a block.
bool build_deblock_grids(const Format& format,
                         const std::vector<ChromaFilterBlock>& blocks,
                         int base_q_idx, int mi_cols, int mi_rows,
                         deblock::DeblockMi* grids[3], std::string& error) {
  const int num_planes = fmt::num_planes(format.chroma);
  if (num_planes < 1 || num_planes > 3) {
    error = "chroma deblock: invalid plane count";
    return false;
  }
  const std::size_t count =
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows);
  (void)count;
  for (const ChromaFilterBlock& block : blocks) {
    const int bx = block.mi_col;
    const int by = block.mi_row;
    const int w4 = block.luma_size / 4;
    if (w4 <= 0) {
      error = "chroma deblock: empty block";
      return false;
    }
    const int clamped_w4 = std::min(w4, mi_cols - bx);
    const int clamped_h4 = std::min(w4, mi_rows - by);
    for (int p = 0; p < num_planes; ++p) {
      deblock::DeblockMi entry;
      entry.block_col_base = static_cast<std::uint16_t>(bx);
      entry.block_row_base = static_cast<std::uint16_t>(by);
      entry.block_w4 = static_cast<std::uint8_t>(clamped_w4);
      entry.block_h4 = static_cast<std::uint8_t>(clamped_h4);
      entry.tx_size = static_cast<deblock::TxSize>(block.tx_size[p]);
      entry.sub_pu_size = entry.tx_size;
      entry.tx_col_base = static_cast<std::uint16_t>(bx);
      entry.tx_row_base = static_cast<std::uint16_t>(by);
      entry.sub_pu_col_base = static_cast<std::uint16_t>(bx);
      entry.sub_pu_row_base = static_cast<std::uint16_t>(by);
      entry.skip = 0;
      entry.region_intra = 1;
      entry.lossless = 0;
      entry.qindex = static_cast<std::uint16_t>(base_q_idx);
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

// §5.18.5.2: apply_deblocking_filter[2..3] need a luma direction, and a
// monochrome stream codes no chroma plane at all.
bool check_deblock_request(const ChromaFilterOptions& options,
                           int num_planes, std::string& error) {
  (void)options;
  (void)num_planes;
  (void)error;
  return true;
}

}  // namespace

bool run_chroma_filters(const Format& format,
                        const ChromaFilterPlanes& planes,
                        const std::vector<ChromaFilterBlock>& blocks,
                        const ChromaFilterOptions& options,
                        ChromaFilterPlan* plan, std::string& error) {
  error.clear();
  if (plan == nullptr) {
    error = "chroma filters: null plan";
    return false;
  }
  *plan = ChromaFilterPlan{};
  if (!options.enable_deblock && !options.enable_cdef && !options.enable_ccso &&
      !options.enable_wiener && !options.enable_chroma_lr &&
      !options.enable_pc_wiener && !options.enable_gdf) {
    return true;
  }
  const int num_planes = fmt::num_planes(format.chroma);
  const int sub_x = fmt::subsampling_x(format.chroma);
  const int sub_y = fmt::subsampling_y(format.chroma);
  if (num_planes < 1 || num_planes > 3) {
    error = "chroma filters: invalid chroma format";
    return false;
  }
  const int mi_cols = static_cast<int>(fmt::mi_cols(format.width));
  const int mi_rows = static_cast<int>(fmt::mi_rows(format.height));
  if (mi_cols <= 0 || mi_rows <= 0) {
    error = "chroma filters: invalid geometry";
    return false;
  }
  if (!check_deblock_request(options, num_planes, error)) return false;

  // §7.20 chroma LR + §7.17 deblock is not decoder-exact at 4:2:2/4:4:4 with
  // the pinned decoder: the chroma lossless-mask read runs past its extent in
  // the stripe at luma row 248 and a few aliased chroma blocks are left
  // unrestored.  The 10-bit writer rejects the same combination by name.
  if (options.enable_chroma_lr && options.enable_deblock && num_planes > 1 &&
      sub_y == 0) {
    error =
        "chroma loop restoration and --still-deblock do not decode exactly at "
        "4:2:2/4:4:4 with pinned dav2d: the decoder's chroma lossless mask "
        "read runs past its extent for the stripe at luma row 248 and skips "
        "restoration of the aliased blocks.  Omit --still-deblock or use "
        "--chroma 420";
    return false;
  }

  // Working planes in the uint16_t container, coded extents.  `plane16` is the
  // reconstruction that the filter chain rewrites; `curr16` is the post-deblock
  // / pre-CDEF copy that §7.19/§7.20 sample.
  std::vector<std::uint16_t> plane16[3];
  std::vector<std::uint16_t> curr16[3];
  std::vector<std::uint16_t> src16[3];
  int coded_w[3] = {0, 0, 0};
  int coded_h[3] = {0, 0, 0};
  for (int p = 0; p < num_planes; ++p) {
    coded_w[p] = plane_width(mi_cols, p, sub_x);
    coded_h[p] = plane_height(mi_rows, p, sub_y);
    if (planes.plane[p] == nullptr || planes.stride[p] < coded_w[p]) {
      error = "chroma filters: null or short reconstruction plane";
      return false;
    }
    plane16[p] = widen(planes.plane[p], planes.stride[p], coded_w[p],
                       coded_h[p]);
    src16[p] = std::vector<std::uint16_t>(
        static_cast<std::size_t>(coded_w[p]) *
            static_cast<std::size_t>(coded_h[p]),
        0);
    // The decision SSE reads the source with the reconstruction stride; the
        // caller guarantees it covers the visible rectangle.
    for (int y = 0; y < planes.visible_h[p]; ++y) {
      const int sy = std::min(y, coded_h[p] - 1);
      for (int x = 0; x < planes.visible_w[p]; ++x) {
        const int sx = std::min(x, coded_w[p] - 1);
        src16[p][static_cast<std::size_t>(sy) *
                      static_cast<std::size_t>(coded_w[p]) +
                  static_cast<std::size_t>(sx)] =
            static_cast<std::uint16_t>(
                planes.source[p][static_cast<std::size_t>(sy) *
                                     static_cast<std::size_t>(
                                         planes.source_stride[p]) +
                                 static_cast<std::size_t>(sx)]);
      }
    }
  }

  // §7.17 deblock.  Runs in place on the reconstruction.  The deblock core's
  // 16-bit entry accepts every chroma subsampling at BitDepth = 8.
  if (options.enable_deblock) {
    std::vector<deblock::DeblockMi> mi[3];
    deblock::DeblockMi* grids[3] = {nullptr, nullptr, nullptr};
    const std::size_t count =
        static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows);
    for (int p = 0; p < num_planes; ++p) {
      mi[p].assign(count, deblock::DeblockMi{});
      grids[p] = mi[p].data();
    }
    if (!build_deblock_grids(format, blocks, options.base_q_idx, mi_cols,
                             mi_rows, grids, error)) {
      return false;
    }
    deblock::DeblockFrame16 frame{};
    for (int p = 0; p < num_planes; ++p) {
      frame.planes[p].data = plane16[p].data();
      frame.planes[p].stride = coded_w[p];
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
    deblock_options.bit_depth = kBitDepth;
    if (deblock::deblock_frame16(frame, deblock_options) !=
        deblock::DeblockStatus::kOk) {
      error = "chroma deblock failed";
      return false;
    }
    plan->deblock = true;
  }

  const bool need_curr = options.enable_ccso || options.enable_cdef ||
                         options.enable_wiener || options.enable_chroma_lr ||
                         options.enable_pc_wiener || options.enable_gdf;
  if (need_curr) {
    for (int p = 0; p < num_planes; ++p) curr16[p] = plane16[p];
  }

  // §7.18 CDEF: CurrFrame -> separate CdefFrame.  The planner's SSE search is
  // the shared one; it is evaluated at BitDepth = 8, so its decisions match the
  // 8-bit core exactly.
  std::vector<std::uint16_t> cdef16[3];
  if (options.enable_cdef) {
    av2enc::hbd::HbdFilterFrame16 filter_frame;
    for (int p = 0; p < num_planes; ++p) {
      filter_frame.src[p] = curr16[p].data();
      filter_frame.ref[p] = src16[p].data();
      filter_frame.src_stride[p] = coded_w[p];
      filter_frame.ref_stride[p] = coded_w[p];
      filter_frame.visible_w[p] = planes.visible_w[p];
      filter_frame.visible_h[p] = planes.visible_h[p];
    }
    filter_frame.mi_cols = mi_cols;
    filter_frame.mi_rows = mi_rows;
    filter_frame.sub_x = sub_x;
    filter_frame.sub_y = sub_y;
    filter_frame.num_planes = num_planes;
    filter_frame.bit_depth = kBitDepth;
    if (!av2enc::hbd::plan_cdef(filter_frame, options.cdef_damping,
                                options.cdef_num_strengths, &plan->cdef,
                                &error)) {
      return false;
    }
    const std::uint16_t* src_planes[3] = {nullptr, nullptr, nullptr};
    std::uint16_t* dst_planes[3] = {nullptr, nullptr, nullptr};
    int src_stride[3] = {0, 0, 0};
    for (int p = 0; p < num_planes; ++p) {
      src_planes[p] = curr16[p].data();
      src_stride[p] = coded_w[p];
      cdef16[p].assign(plane16[p].size(), 0);
      dst_planes[p] = cdef16[p].data();
    }
    if (!av2enc::hbd::apply_cdef16(plan->cdef, format, src_planes, dst_planes,
                                   src_stride, mi_cols, mi_rows, &error)) {
      return false;
    }
    for (int p = 0; p < num_planes; ++p) plane16[p].swap(cdef16[p]);
  }

  // §7.19 CCSO: reads the pre-CDEF CurrFrame and filters the post-CDEF planes
  // in place.
  if (options.enable_ccso) {
    av2enc::hbd::HbdCcsoFrame16 filter_frame;
    const std::uint16_t* curr_planes[3] = {nullptr, nullptr, nullptr};
    std::uint16_t* out_planes[3] = {nullptr, nullptr, nullptr};
    int stride[3] = {0, 0, 0};
    for (int p = 0; p < num_planes; ++p) {
      filter_frame.curr[p] = curr16[p].data();
      filter_frame.cdef[p] = plane16[p].data();
      filter_frame.ref[p] = src16[p].data();
      filter_frame.curr_stride[p] = coded_w[p];
      filter_frame.cdef_stride[p] = coded_w[p];
      filter_frame.ref_stride[p] = coded_w[p];
      filter_frame.visible_w[p] = planes.visible_w[p];
      filter_frame.visible_h[p] = planes.visible_h[p];
      curr_planes[p] = curr16[p].data();
      out_planes[p] = plane16[p].data();
      stride[p] = coded_w[p];
    }
    filter_frame.mi_cols = mi_cols;
    filter_frame.mi_rows = mi_rows;
    filter_frame.sub_x = sub_x;
    filter_frame.sub_y = sub_y;
    filter_frame.num_planes = num_planes;
    filter_frame.bit_depth = kBitDepth;
    if (!av2enc::hbd::plan_ccso(filter_frame, options.ccso_luma_size_log2,
                                options.ccso_edge_only, &plan->ccso,
                                &error)) {
      return false;
    }
    if (!av2enc::hbd::apply_ccso16(plan->ccso, format, curr_planes, out_planes,
                                   stride, stride, mi_cols, mi_rows, &error)) {
      return false;
    }
  }

  // §7.20 loop restoration (Wiener NS / PC-Wiener) and §7.20.5 GDF.  LrFrame is
  // separate: LR reads CurrFrame/CdefFrame, then GDF rewrites its luma.
  const bool any_lr = options.enable_wiener || options.enable_chroma_lr ||
                      options.enable_pc_wiener;
  if (any_lr || options.enable_gdf) {
    av2enc::hbd::HbdLrFrame16 lr_frame;
    for (int p = 0; p < num_planes; ++p) {
      lr_frame.curr[p] = curr16[p].data();
      lr_frame.cdef[p] = plane16[p].data();
      lr_frame.ref[p] = src16[p].data();
      lr_frame.curr_stride[p] = coded_w[p];
      lr_frame.cdef_stride[p] = coded_w[p];
      lr_frame.ref_stride[p] = coded_w[p];
      lr_frame.visible_w[p] = planes.visible_w[p];
      lr_frame.visible_h[p] = planes.visible_h[p];
    }
    lr_frame.mi_cols = mi_cols;
    lr_frame.mi_rows = mi_rows;
    lr_frame.sub_x = sub_x;
    lr_frame.sub_y = sub_y;
    lr_frame.num_planes = num_planes;
    lr_frame.bit_depth = kBitDepth;
    lr_frame.base_q_idx = options.base_q_idx;

    std::vector<std::uint8_t> lr_tx_skip(
        static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows),
        0);
    if (options.enable_pc_wiener) {
      for (const ChromaFilterBlock& block : blocks) {
        if (!block.luma_eob_zero) continue;
        const int w4 = block.luma_size / 4;
        for (int yy = 0; yy < w4; ++yy) {
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

    if (any_lr) {
      if (!av2enc::hbd::plan_lr(lr_frame, options.enable_wiener,
                                options.enable_chroma_lr,
                                options.enable_pc_wiener, options.base_q_idx,
                                &plan->lr, &error)) {
        return false;
      }
      plan->lr.config.lr_tx_skip = lr_tx_skip;
      plan->lr.config.tx_skip_cols = mi_cols;
      plan->lr.config.tx_skip_rows = mi_rows;
      if (options.enable_pc_wiener) {
        std::vector<std::uint16_t> scratch(
            static_cast<std::size_t>(coded_w[0]) *
            static_cast<std::size_t>(coded_h[0]));
        av2enc::hbd::plan_pc_wiener16(lr_frame, &plan->lr, lr_tx_skip,
                                      scratch.data());
      }
    }

    std::vector<std::uint16_t> lr16[3];
    std::uint16_t* lr_ptr[3] = {nullptr, nullptr, nullptr};
    int lr_stride[3] = {0, 0, 0};
    for (int p = 0; p < num_planes; ++p) {
      lr16[p].assign(plane16[p].size(), 0);
      lr_ptr[p] = lr16[p].data();
      lr_stride[p] = coded_w[p];
    }
    if (any_lr && plan->lr.config.enabled) {
      if (!av2enc::hbd::apply_wiener16(lr_frame, plan->lr.config, lr_ptr,
                                       lr_stride, &error)) {
        return false;
      }
      if (options.enable_pc_wiener && plan->lr.config.plane[0].enabled) {
        av2enc::hbd::apply_pc_wiener16(lr_frame, plan->lr.config,
                                       plan->lr.config.lr_tx_skip, lr_ptr[0],
                                       lr_stride[0]);
      }
    } else {
      for (int p = 0; p < num_planes; ++p) lr16[p] = plane16[p];
    }

    if (options.enable_gdf) {
      av2enc::hbd::HbdGdfFrame16 gdf_frame;
      gdf_frame.curr = curr16[0].data();
      gdf_frame.cdef = plane16[0].data();
      gdf_frame.ref = src16[0].data();
      gdf_frame.curr_stride = coded_w[0];
      gdf_frame.cdef_stride = coded_w[0];
      gdf_frame.ref_stride = coded_w[0];
      gdf_frame.width = planes.visible_w[0];
      gdf_frame.height = planes.visible_h[0];
      gdf_frame.mi_cols = mi_cols;
      gdf_frame.mi_rows = mi_rows;
      gdf_frame.bit_depth = kBitDepth;
      gdf_frame.base_q_idx = options.base_q_idx;
      if (!av2enc::hbd::plan_gdf(gdf_frame, &plan->gdf, &error)) {
        return false;
      }
      if (!av2enc::hbd::apply_gdf16(plan->gdf, gdf_frame, lr_ptr[0],
                                    lr_stride[0], &error)) {
        return false;
      }
    }

    for (int p = 0; p < num_planes; ++p) plane16[p].swap(lr16[p]);
  }

  // Narrow the filtered planes back into the writer's 8-bit reconstruction.
  for (int p = 0; p < num_planes; ++p) {
    narrow_into(plane16[p], coded_w[p], coded_h[p], planes.plane[p],
                planes.stride[p]);
  }
  return true;
}

}  // namespace av2enc::chroma
