// Adapter from the shared M1 GPU contract (src/cuda/m1/m1_gpu.h) to this
// scheduler's host-only plan.
//
// The orchestrator owns m1gpu::FramePlan/BlockPlan.  This module only reads
// them: geometry and the syntax-order leaf list come from the shared PODs, and
// the per-block decisions (today from the CPU plan stage, later from GPU
// search) are optional.  Availability counts are not copied because the
// scheduler replays clear_block_decoded_flags() itself, which is exact for the
// component's num4AboveRight/num4BelowLeft inputs.
#ifndef AV2ENC_CUDA_M1_SCHED_BRIDGE_H_
#define AV2ENC_CUDA_M1_SCHED_BRIDGE_H_

#include <string>
#include <vector>

#include "cuda/m1/m1_gpu.h"
#include "cuda/m1/sched/sched.h"

namespace av2enc::cuda::m1::sched {

// Owns the converted leaf list and the FramePlan view over it.  Move-only:
// FramePlan::blocks points into `blocks`, so copying would dangle.
class AdaptedFramePlan {
 public:
  AdaptedFramePlan() = default;
  // `decisions` may be null; when non-null it must hold src.block_count
  // entries in the same syntax order.
  AdaptedFramePlan(const av2enc::cuda::m1gpu::FramePlan& src,
                   const av2enc::cuda::m1gpu::BlockDecision* decisions);
  AdaptedFramePlan(const AdaptedFramePlan&) = delete;
  AdaptedFramePlan& operator=(const AdaptedFramePlan&) = delete;
  AdaptedFramePlan(AdaptedFramePlan&&) noexcept = default;
  AdaptedFramePlan& operator=(AdaptedFramePlan&&) noexcept = default;

  const FramePlan& frame() const { return frame_; }
  const std::vector<BlockPlan>& blocks() const { return blocks_; }

 private:
  FramePlan frame_{};
  std::vector<BlockPlan> blocks_;
};

// Builds the wavefront plan directly from the shared PODs.
bool build_m1_plan(const av2enc::cuda::m1gpu::FramePlan& src,
                   const av2enc::cuda::m1gpu::BlockDecision* decisions,
                   Plan* out, std::string& error);

Plan build_m1_plan(const av2enc::cuda::m1gpu::FramePlan& src,
                   const av2enc::cuda::m1gpu::BlockDecision* decisions);

inline Plan build_m1_plan(const av2enc::cuda::m1gpu::FramePlan& src) {
  return build_m1_plan(src, nullptr);
}

}  // namespace av2enc::cuda::m1::sched

#endif  // AV2ENC_CUDA_M1_SCHED_BRIDGE_H_
