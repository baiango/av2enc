// Device-ready helpers for CompoundBatchItem.
//
// This header is standard-library free and safe to include from a CUDA
// translation unit: it only fills an adv::CompoundJob from a flat item plus
// base pointers.  The blending itself is adv::run_compound_job(), which is
// already AV2ENC_ADV_HD and is shared by src/cuda/inter/advanced.
#pragma once

#include "encoder/inter/advanced/compound_common.h"
#include "encoder/m2/multiref/gpu_batch.h"

#if defined(__CUDACC__)
#define AV2ENC_MULTIREF_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_MULTIREF_HD inline
#endif

namespace av2enc::m2::multiref {

// Fills the job in place.  `pred0`/`pred1`/`dst`/`masks` are the pool bases the
// item offsets refer to.
AV2ENC_MULTIREF_HD void load_compound_job(const CompoundBatchItem& item,
                                          const std::int16_t* pred0,
                                          const std::int16_t* pred1,
                                          std::uint8_t* dst,
                                          std::uint8_t* masks,
                                          adv::CompoundJob& job) {
  job = adv::CompoundJob{};
  job.plane = item.plane;
  job.sub_x = item.sub_x;
  job.sub_y = item.sub_y;
  job.bsize = item.bsize;
  job.bit_depth = 8;
  job.rect = adv::BlockRect{item.rect_x, item.rect_y, item.rect_w,
                            item.rect_h};
  job.p0 = pred0 + item.p0_offset;
  job.p0_stride = item.p0_stride;
  job.p1 = pred1 + item.p1_offset;
  job.p1_stride = item.p1_stride;
  job.dst = dst + item.dst_offset;
  job.dst_stride = item.dst_stride;
  job.type = static_cast<adv::CompoundType>(item.type);
  job.cwp_weight = item.cwp_weight;
  job.wedge_index = item.wedge_index;
  job.wedge_sign = item.wedge_sign;
  job.mask_type = item.mask_type;
  job.imp_mask_eligible = item.imp_mask_eligible;
  job.ref_start_x[0] = item.ref_start_x[0];
  job.ref_start_x[1] = item.ref_start_x[1];
  job.ref_start_y[0] = item.ref_start_y[0];
  job.ref_start_y[1] = item.ref_start_y[1];
  job.mi_cols = item.mi_cols;
  job.mi_rows = item.mi_rows;
  job.mask_out =
      item.mask_stride > 0 ? masks + item.mask_offset : nullptr;
  job.mask_out_stride = item.mask_stride;
  job.luma_mask =
      item.luma_mask_stride > 0 ? masks + item.luma_mask_offset : nullptr;
  job.luma_mask_stride = item.luma_mask_stride;
}

// Per-sample blending entry point shared with a future CUDA kernel.
AV2ENC_MULTIREF_HD int compound_sample_from_item(const CompoundBatchItem& item,
                                                 const adv::CompoundJob& job,
                                                 int i, int j, int pred0,
                                                 int pred1) {
  (void)item;
  return adv::compound_sample_at(job, i, j, pred0, pred1, 0, 0);
}

}  // namespace av2enc::m2::multiref
