// M1 GPU in-loop restoration stage — implementation.

#include "still_restoration.h"

#include <cstddef>

#include "../../../codec/filters/loop_restoration/ccso.h"
#include "../../../codec/filters/loop_restoration/wiener.h"

namespace av2enc::cuda {
namespace still {
namespace {

cudaError_t status_to_cuda(restoration::RestorationStatus st) {
  if (st == restoration::RestorationStatus::kOk) {
    return cudaSuccess;
  }
  if (st == restoration::RestorationStatus::kUnsupportedGdf ||
      st == restoration::RestorationStatus::kUnsupportedPcWiener ||
      st == restoration::RestorationStatus::kUnsupportedBitDepth) {
    return cudaErrorNotSupported;
  }
  return cudaErrorInvalidValue;
}

restoration::CcsoFrameParams host_ccso_params(
    const RestorationCcsoStage& p) {
  restoration::CcsoFrameParams out{};
  out.ccso_luma_size_log2 = p.ccso_luma_size_log2;
  for (int plane = 0; plane < 3; ++plane) {
    out.plane[plane] = p.plane[plane];
    // The inline host `offsets` table is not used by validation; the kernel
    // gets `offsets_dev` through restoration_stage_ccso_core().
    out.blks[plane] = p.blks[plane];
    out.blk_cols[plane] = p.blk_cols[plane];
    out.blk_rows[plane] = p.blk_rows[plane];
  }
  return out;
}

}  // namespace

restoration::CcsoCoreFrame restoration_stage_ccso_core(
    const restoration::RestorationFrame& frame,
    const RestorationCcsoStage& params) {
  restoration::CcsoCoreFrame core{};
  for (int plane = 0; plane < 3; ++plane) {
    restoration::CcsoCorePlane& c = core.plane[plane];
    c.curr_luma = frame.curr[0].data;
    c.curr_luma_stride = frame.curr[0].stride;
    c.dst = frame.cdef[plane].data;
    c.dst_stride = frame.cdef[plane].stride;
    c.blks = params.blks[plane];
    c.blk_cols = params.blk_cols[plane];
    c.blk_rows = params.blk_rows[plane];
    c.enabled = params.plane[plane].enabled;
    c.scale_idx = params.plane[plane].scale_idx;
    c.quant_idx = params.plane[plane].quant_idx;
    c.ext_filter = params.plane[plane].ext_filter;
    c.edge_clf = params.plane[plane].edge_clf;
    c.max_band_log2 = params.plane[plane].max_band_log2;
    c.bo_only = params.plane[plane].bo_only;
    c.offsets = params.offsets_dev[plane];
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.subsampling_x;
  core.sub_y = frame.subsampling_y;
  core.bit_depth = frame.bit_depth;
  core.ccso_luma_size_log2 = params.ccso_luma_size_log2;
  core.sb_width = frame.sb_width;
  core.sb_height = frame.sb_height;
  core.bru_modes = frame.bru_modes;
  core.segment_ids = frame.segment_ids;
  core.chroma_segment_ids = frame.chroma_segment_ids;
  core.lossless = frame.lossless;
  core.mi_col_start = frame.mi_col_start;
  core.mi_col_end = frame.mi_col_end;
  core.mi_row_start = frame.mi_row_start;
  core.mi_row_end = frame.mi_row_end;
  core.disable_loopfilters_across_tiles =
      frame.disable_loopfilters_across_tiles;
  return core;
}

cudaError_t restoration_stage_validate(
    const restoration::RestorationFrame& frame,
    const RestorationStageParams& params) {
  // CCSO: scalar/range/grid-shape validation only (no device dereference).
  const restoration::CcsoFrameParams host = host_ccso_params(params.ccso);
  cudaError_t err =
      status_to_cuda(restoration::ccso_validate(frame, host));
  if (err != cudaSuccess) {
    return err;
  }
  for (int p = 0; p < 3; ++p) {
    if (!restoration::ccso_header_conforms(params.ccso.plane[p])) {
      return cudaErrorInvalidValue;
    }
  }

  // Loop restoration: pointer/shape validation only.  Host-only grid checks
  // (tile alignment, PC-Wiener in a switchable unit) are not performed here
  // because those grids are device memory; see the README contract.
  err = status_to_cuda(
      restoration::loop_restoration_validate(frame, params.lr));
  if (err != cudaSuccess) {
    return err;
  }

  // CCSO reads CurrFrame[0] (pre-CDEF/deblocked luma) while writing
  // CdefFrame.  If a plane is CCSO-modified, the same plane's CurrFrame must be
  // a distinct pre-CCSO snapshot because loop restoration sources samples
  // outside the current stripe from CurrFrame.
  for (int p = 0; p < 3; ++p) {
    if (!params.ccso.plane[p].enabled) {
      continue;
    }
    if (params.ccso.offsets_dev[p] == nullptr) {
      return cudaErrorInvalidValue;
    }
    if (frame.curr[p].data == frame.cdef[p].data) {
      return cudaErrorInvalidValue;
    }
  }
  // CCSO's edge classifier always reads luma from CurrFrame[0].
  if (params.ccso.any_enabled() &&
      frame.curr[0].data == frame.cdef[0].data) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t restoration_stage_apply(
    const restoration::RestorationFrame& frame,
    const RestorationStageParams& params, cudaStream_t stream) {
  const cudaError_t err = restoration_stage_validate(frame, params);
  if (err != cudaSuccess) {
    return err;
  }

  // 1. CCSO modifies CdefFrame in place (§7.19).
  if (params.ccso.any_enabled()) {
    const restoration::CcsoCoreFrame core =
        restoration_stage_ccso_core(frame, params.ccso);
    const cudaError_t ccso_err = ccso_frame(core, stream);
    if (ccso_err != cudaSuccess) {
      return ccso_err;
    }
  }

  // 2. §7.20: LrFrame := CdefFrame, then Wiener NS where LrType selects it.
  //    The launcher performs the copy even when every type is RESTORE_NONE.
  return loop_restoration_frame(frame, params.lr, stream);
}

}  // namespace still
}  // namespace av2enc::cuda
