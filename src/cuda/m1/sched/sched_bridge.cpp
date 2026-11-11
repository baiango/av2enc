// Shared-contract adapter; see sched_bridge.h.

#include "cuda/m1/sched/sched_bridge.h"

#include <cstddef>

namespace av2enc::cuda::m1::sched {

AdaptedFramePlan::AdaptedFramePlan(
    const av2enc::cuda::m1gpu::FramePlan& src,
    const av2enc::cuda::m1gpu::BlockDecision* decisions) {
  frame_.width = static_cast<int>(src.width);
  frame_.height = static_cast<int>(src.height);
  frame_.chroma_width = (frame_.width + 1) / 2;
  frame_.chroma_height = (frame_.height + 1) / 2;
  frame_.padded_luma_width =
      static_cast<int>(src.y_stride != 0 ? src.y_stride : src.padded_width);
  frame_.padded_luma_height = static_cast<int>(src.padded_height);
  frame_.padded_chroma_width =
      static_cast<int>(src.u_stride != 0 ? src.u_stride
                                         : src.padded_width / 2);
  frame_.padded_chroma_height = frame_.padded_luma_height / 2;
  frame_.mi_cols = static_cast<int>(src.mi_cols);
  frame_.mi_rows = static_cast<int>(src.mi_rows);

  blocks_.reserve(src.blocks.size());
  for (std::size_t i = 0; i < src.blocks.size(); ++i) {
    const av2enc::cuda::m1gpu::BlockPlan& b = src.blocks[i];
    BlockPlan out;
    out.mi_row = static_cast<int>(b.mi_row);
    out.mi_col = static_cast<int>(b.mi_col);
    out.luma_w = static_cast<int>(b.luma_w);
    out.luma_h = static_cast<int>(b.luma_h);
    out.chroma_w = static_cast<int>(b.chroma_w);
    out.chroma_h = static_cast<int>(b.chroma_h);
    out.has_chroma = b.has_chroma != 0;
    if (decisions != nullptr) {
      const av2enc::cuda::m1gpu::BlockDecision& d = decisions[i];
      out.luma_mode = static_cast<int>(d.luma_mode);
      out.luma_angle = static_cast<int>(d.luma_angle_delta);
      out.is_cfl = d.is_cfl != 0 ? 1 : 0;
      out.chroma_mode = out.is_cfl != 0 ? 0 : static_cast<int>(d.uv_mode);
      out.chroma_angle = 0;
    }
    blocks_.push_back(out);
  }
  frame_.blocks = blocks_.empty() ? nullptr : blocks_.data();
  frame_.block_count = static_cast<int>(blocks_.size());
}

bool build_m1_plan(const av2enc::cuda::m1gpu::FramePlan& src,
                   const av2enc::cuda::m1gpu::BlockDecision* decisions,
                   Plan* out, std::string& error) {
  if (out == nullptr) {
    error = "build_m1_plan: null output plan";
    return false;
  }
  const AdaptedFramePlan adapted(src, decisions);
  return out->build(adapted.frame(), error);
}

Plan build_m1_plan(const av2enc::cuda::m1gpu::FramePlan& src,
                   const av2enc::cuda::m1gpu::BlockDecision* decisions) {
  Plan out;
  std::string error;
  build_m1_plan(src, decisions, &out, error);
  return out;
}

}  // namespace av2enc::cuda::m1::sched
