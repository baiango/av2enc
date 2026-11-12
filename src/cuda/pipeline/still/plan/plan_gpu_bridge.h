// Optional bridge from the plan component's host PODs to the orchestrator's
// shared FramePlan / BlockPlan contract in src/cuda/pipeline/still/still_gpu.h.
//
// The planner owns the real partition tree and leaf order; the orchestrator
// owns the device PODs and buffer sizing.  This bridge performs the field
// mapping only.  It is compiled only when the contract header is present, so
// the plan tests still build on a tree without the CUDA M1 orchestrator.
//
// Integration note: the orchestrator must size `blocks` for the *leaf* count,
// not for the superblock count.  A depth-4 640x360 tree has ~247 leaves versus
// 60 superblocks; the current stage::build_plan hook passes a superblock-sized
// span, so it cannot accept this plan until the caller grows that allocation.

#pragma once

#include "cuda/pipeline/still/plan/plan.h"

#include <span>
#include <string>
#include <vector>

#if __has_include("cuda/pipeline/still/still_gpu.h")
#include "cuda/pipeline/still/still_gpu.h"
#if __has_include("pipeline/still/stream/still_stream_bridge.h")
#include "pipeline/still/stream/still_stream_bridge.h"
#endif

namespace av2enc::still::plan {

// Builds the frame plan and fills the orchestrator PODs.  `blocks` is resized
// to the leaf count; `frame.blocks` points at it, so both must outlive use.
bool build_gpu_frame_plan(int width, int height,
                          std::span<const std::uint8_t> luma,
                          const PlanOptions& options,
                          std::vector<cuda::still_gpu::BlockPlan>& blocks,
                          cuda::still_gpu::FramePlan& frame, std::string& error);

#if __has_include("pipeline/still/stream/still_stream_bridge.h")
// Same, plus the preorder tree in the exact shape `stream::pack_gpu_frame()`
// consumes.  `tree` is resized to the node count and left in preorder.
bool build_gpu_frame_plan(int width, int height,
                          std::span<const std::uint8_t> luma,
                          const PlanOptions& options,
                          std::vector<cuda::still_gpu::BlockPlan>& blocks,
                          cuda::still_gpu::FramePlan& frame,
                          std::vector<stream::TreeNode>& tree,
                          std::string& error);
#endif

}  // namespace av2enc::still::plan

#endif  // __has_include("cuda/pipeline/still/still_gpu.h")
