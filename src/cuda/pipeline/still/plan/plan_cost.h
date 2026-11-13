// Device leaf-variance cost kernel for the M1 GPU frame planner.
//
// The planner's decision model uses the same truncated block-variance cost as
// the CPU M1 encoder.  This kernel computes that cost for an explicit list of
// visible rectangles so a plan can be searched from device costs without
// changing the resulting tree.  It allocates nothing: the caller passes arena
// buffers and the orchestrator stream.
//
// The arithmetic mirrors src/cuda/pipeline/still/plan/plan.cpp's
// block_variance_cost_at(): row-major int64 sum, double mean, row-major double
// SSE accumulation, truncated to int64.  One thread handles one rectangle so
// the summation order is identical to the CPU reference; compile the kernel
// with -fmad=false (or keep the explicit __d* intrinsics).
#pragma once

#include <cstdint>
#include <string>

namespace av2enc::still::plan {

// Visible rectangle in luma samples.  `w`/`h` are the clipped visible extents,
// not the coded block size.
struct PlanCostRect {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::uint16_t w = 0;
  std::uint16_t h = 0;
  std::uint16_t reserved = 0;
};

// Launch the variance kernel on `count` rectangles.  `d_luma` is a device
// plane with `luma_stride` samples per row; `d_costs` receives one int64 per
// rectangle.  Returns false with a non-empty `error` on a launch failure.  No
// synchronization is performed.
bool plan_cost_launch_variance(const std::uint8_t* d_luma, int luma_stride,
                               const PlanCostRect* d_rects,
                               std::int64_t* d_costs, int count, void* stream,
                               std::string* error);

}  // namespace av2enc::still::plan
