// GPU-ready compound blend batch.
//
// The CPU reference in compound_predict.cpp calls adv::apply_compound() once per
// plane.  A CUDA implementation wants one flat, pointer-free work description
// that it can upload with a single cudaMemcpy and execute with one thread per
// sample.  This file defines that description:
//
//   * CompoundBatchItem is trivially copyable and contains only ints and
//     offsets (no pointers, no std::vector), so it is device-memcpy ready;
//   * predictions, destinations, sources and masks live in flat pools;
//   * run_compound_batch_cpu() fills an adv::CompoundJob from the item and
//     calls the shared per-sample code, so the CPU batch is bit-exact with
//     both the single-job path and (by construction) the same kernel body;
//   * compound_batch_core.h exposes load_compound_job(), the only device-side
//     helper a kernel needs, marked AV2ENC_MULTIREF_HD.
//
// The existing CUDA ownership rules apply: a kernel must allocate these pools
// from src/cuda/arena.*, not cudaMalloc.
#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "codec/inter/advanced/compound.h"

namespace av2enc::video::multiref {

namespace adv = av2enc::inter::adv;

// One compound blend job.  All offsets are element offsets into the pools of
// CompoundBatch; strides are in samples.
struct CompoundBatchItem {
  int plane = 0;
  int rect_x = 0;
  int rect_y = 0;
  int rect_w = 0;
  int rect_h = 0;
  int bsize = 0;
  int sub_x = 0;
  int sub_y = 0;

  // 1/16-sample predictions (block-local).
  int p0_offset = 0;
  int p0_stride = 0;
  int p1_offset = 0;
  int p1_stride = 0;

  int dst_offset = 0;
  int dst_stride = 0;
  int src_offset = 0;
  int src_stride = 0;

  // Plane-0 mask output and chroma luma-mask input.
  int mask_offset = 0;
  int mask_stride = 0;
  int luma_mask_offset = 0;
  int luma_mask_stride = 0;

  // adv::CompoundType.
  int type = static_cast<int>(adv::CompoundType::kAverage);
  int cwp_weight = 8;
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
  int imp_mask_eligible = 0;

  int ref_start_x[2] = {0, 0};
  int ref_start_y[2] = {0, 0};
  int mi_cols = 0;
  int mi_rows = 0;

  // 1 -> write the luma SSE against `src` into sse_out.
  int compute_sse = 0;
  std::int64_t sse_out = 0;
};

static_assert(std::is_trivially_copyable_v<CompoundBatchItem>,
              "CompoundBatchItem must be device-memcpy ready");
static_assert(sizeof(CompoundBatchItem) % 8 == 0,
              "keep the item 8-byte aligned for vector loads");

struct CompoundBatch {
  // Frame plane geometry and strides (samples).
  int plane_width[3] = {0, 0, 0};
  int plane_height[3] = {0, 0, 0};
  int plane_stride[3] = {0, 0, 0};
  int plane_dst_offset[3] = {0, 0, 0};
  int plane_src_offset[3] = {0, 0, 0};

  std::vector<std::int16_t> pred0;  // 1/16 predictions, list 0
  std::vector<std::int16_t> pred1;  // 1/16 predictions, list 1
  std::vector<std::uint8_t> dst;    // three planes, destrided by plane_stride
  std::vector<std::uint8_t> src;    // three planes
  // Plane-0 mask pool.  Chroma wedge/diffwtd items reference their block's
  // luma mask through luma_mask_offset, so a batch must be ordered with each
  // block's plane-0 item before its chroma items (run_compound_batch_cpu()
  // enforces this by executing the list in order).
  std::vector<std::uint8_t> masks;

  std::vector<CompoundBatchItem> items;

  void reset(int width, int height);
  // Reserves pool space and returns the element offsets.
  int alloc_pred(std::size_t samples);
  int alloc_mask(std::size_t samples);
};

// CPU reference executor for the batch.  Returns false on the first invalid
// item (leaving previous items applied).
bool run_compound_batch_cpu(CompoundBatch& batch, std::string& error);

// Validates that every offset points inside its pool.
bool validate_compound_batch(const CompoundBatch& batch, std::string& error);

}  // namespace av2enc::video::multiref
