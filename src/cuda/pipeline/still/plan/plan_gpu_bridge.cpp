// Bridge: plan::FramePlan -> av2enc::cuda::still_gpu::FramePlan.  See the header.

#include "cuda/pipeline/still/plan/plan_gpu_bridge.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#if __has_include("cuda/m1/still_gpu.h")

namespace av2enc::still::plan {
namespace {

using GpuBlock = cuda::still_gpu::BlockPlan;

std::uint8_t avail_mask(const NeighborAvailability& avail) {
  std::uint8_t mask = 0;
  if (avail.left) mask |= cuda::still_gpu::kAvailLeft;
  if (avail.above) mask |= cuda::still_gpu::kAvailAbove;
  if (avail.above_left) mask |= cuda::still_gpu::kAvailAboveLeft;
  if (avail.above_right) mask |= cuda::still_gpu::kAvailAboveRight;
  if (avail.below_left) mask |= cuda::still_gpu::kAvailBelowLeft;
  return mask;
}

bool upscale_ratio(int coded, int adjusted, std::uint8_t& out,
                   std::string& error) {
  if (adjusted <= 0 || coded % adjusted != 0) {
    error = "transform size is not an integer upscale of the adjusted size";
    return false;
  }
  const int ratio = coded / adjusted;
  if (ratio < 1 || ratio > 2) {
    error = "unsupported coded/adjusted transform upscale ratio";
    return false;
  }
  out = static_cast<std::uint8_t>(ratio);
  return true;
}

bool narrow_u8(int value, std::uint8_t& out, const char* what,
               std::string& error) {
  if (value < 0 || value > std::numeric_limits<std::uint8_t>::max()) {
    error = std::string(what) + " does not fit the orchestrator POD";
    return false;
  }
  out = static_cast<std::uint8_t>(value);
  return true;
}

bool map_gpu_frame(const FramePlan& plan, std::vector<GpuBlock>& blocks,
                   cuda::still_gpu::FramePlan& frame, std::string& error) {
  if (plan.blocks.size() > std::numeric_limits<std::uint32_t>::max()) {
    error = "leaf count does not fit the orchestrator POD";
    return false;
  }

  blocks.assign(plan.blocks.size(), GpuBlock{});
  std::uint8_t max_chroma_w = 0;
  std::uint8_t max_chroma_h = 0;
  for (std::size_t i = 0; i < plan.blocks.size(); ++i) {
    const BlockPlan& src = plan.blocks[i];
    GpuBlock& dst = blocks[i];
    if (src.mi_row > std::numeric_limits<std::uint16_t>::max() ||
        src.mi_col > std::numeric_limits<std::uint16_t>::max()) {
      error = "block origin does not fit the orchestrator POD";
      return false;
    }
    dst.mi_row = static_cast<std::uint16_t>(src.mi_row);
    dst.mi_col = static_cast<std::uint16_t>(src.mi_col);
    if (!narrow_u8(src.luma_w, dst.luma_w, "luma width", error) ||
        !narrow_u8(src.luma_h, dst.luma_h, "luma height", error) ||
        !narrow_u8(src.chroma_w, dst.chroma_w, "chroma width", error) ||
        !narrow_u8(src.chroma_h, dst.chroma_h, "chroma height", error) ||
        !narrow_u8(src.luma.adj_w, dst.tx_w, "TX width", error) ||
        !narrow_u8(src.luma.adj_h, dst.tx_h, "TX height", error) ||
        !narrow_u8(src.u.adj_w, dst.chroma_tx_w, "chroma TX width", error) ||
        !narrow_u8(src.u.adj_h, dst.chroma_tx_h, "chroma TX height", error)) {
      return false;
    }
    if (!upscale_ratio(src.luma_w, src.luma.adj_w, dst.tx_upscale_w, error) ||
        !upscale_ratio(src.luma_h, src.luma.adj_h, dst.tx_upscale_h, error)) {
      return false;
    }
    dst.has_chroma = src.has_chroma ? 1 : 0;
    dst.avail = avail_mask(src.luma_avail);
    dst.tree_depth = src.depth;

    // Padded reconstruction-plane sample offsets (see still_encoder.cpp strides).
    dst.plane_offset_y =
        static_cast<std::uint32_t>(src.y * plan.luma_stride + src.x);
    dst.plane_offset_u = static_cast<std::uint32_t>(
        src.chroma_y * plan.chroma_stride + src.chroma_x);
    dst.plane_offset_v = dst.plane_offset_u;

    dst.coeff_offset_y = src.luma.coeff_offset;
    dst.coeff_offset_u = src.u.coeff_offset;
    dst.coeff_offset_v = src.v.coeff_offset;

    max_chroma_w = std::max(max_chroma_w, dst.chroma_w);
    max_chroma_h = std::max(max_chroma_h, dst.chroma_h);
  }

  frame = cuda::still_gpu::FramePlan{};
  frame.width = static_cast<std::uint32_t>(plan.width);
  frame.height = static_cast<std::uint32_t>(plan.height);
  frame.padded_width = static_cast<std::uint32_t>(plan.luma_stride);
  frame.padded_height = static_cast<std::uint32_t>(plan.luma_padded_height);
  frame.y_stride = static_cast<std::uint32_t>(plan.luma_stride);
  frame.u_stride = static_cast<std::uint32_t>(plan.chroma_stride);
  frame.v_stride = static_cast<std::uint32_t>(plan.chroma_stride);
  frame.mi_cols = static_cast<std::uint16_t>(plan.mi_cols);
  frame.mi_rows = static_cast<std::uint16_t>(plan.mi_rows);
  frame.sb_cols = static_cast<std::uint16_t>(plan.sb_cols);
  frame.sb_rows = static_cast<std::uint16_t>(plan.sb_rows);
  frame.sb_size = 64;
  frame.chroma_shift_x = 1;
  frame.chroma_shift_y = 1;
  frame.is_placeholder = 0;
  frame.max_chroma_w = max_chroma_w;
  frame.max_chroma_h = max_chroma_h;
  frame.block_count = static_cast<std::uint32_t>(blocks.size());
  frame.leaf_count = static_cast<std::uint32_t>(plan.leaves);
  frame.coeff_count_y = plan.luma_coeff_count;
  frame.coeff_count_u = plan.u_coeff_count;
  frame.coeff_count_v = plan.v_coeff_count;
  frame.blocks = std::span<const GpuBlock>(blocks);
  return true;
}

#if __has_include("encoder/m1/stream/still_stream_bridge.h")
bool map_stream_tree(const FramePlan& plan,
                     std::vector<stream::TreeNode>& tree,
                     std::string& error) {
  tree.assign(plan.tree.size(), stream::TreeNode{});
  for (std::size_t i = 0; i < plan.tree.size(); ++i) {
    const TreeNode& src = plan.tree[i];
    stream::TreeNode& dst = tree[i];
    if (src.mi_row < 0 || src.mi_col < 0 ||
        src.mi_row > std::numeric_limits<std::uint16_t>::max() ||
        src.mi_col > std::numeric_limits<std::uint16_t>::max()) {
      error = "tree node origin does not fit the stream POD";
      return false;
    }
    dst.mi_row = static_cast<std::uint16_t>(src.mi_row);
    dst.mi_col = static_cast<std::uint16_t>(src.mi_col);
    dst.bsize = src.block_size;
    dst.partition = src.partition;
    dst.allowed_mask = src.allowed_mask;
    dst.flags = static_cast<std::uint8_t>(
        (src.forced ? syntax::kFlagForced : 0) |
        (src.sb_root ? syntax::kFlagSbRoot : 0));
    dst.child_count = src.child_count;
    dst.leaf_index = src.leaf_index == kNoTreeNode ? 0 : src.leaf_index;
  }
  return true;
}
#endif

}  // namespace

bool build_gpu_frame_plan(int width, int height,
                          std::span<const std::uint8_t> luma,
                          const PlanOptions& options,
                          std::vector<GpuBlock>& blocks,
                          cuda::still_gpu::FramePlan& frame, std::string& error) {
  FramePlan plan;
  if (!build_frame_plan(width, height, luma, options, plan, error)) {
    return false;
  }
  return map_gpu_frame(plan, blocks, frame, error);
}

#if __has_include("encoder/m1/stream/still_stream_bridge.h")
bool build_gpu_frame_plan(int width, int height,
                          std::span<const std::uint8_t> luma,
                          const PlanOptions& options,
                          std::vector<GpuBlock>& blocks,
                          cuda::still_gpu::FramePlan& frame,
                          std::vector<stream::TreeNode>& tree,
                          std::string& error) {
  FramePlan plan;
  if (!build_frame_plan(width, height, luma, options, plan, error)) {
    return false;
  }
  if (!map_gpu_frame(plan, blocks, frame, error)) {
    return false;
  }
  return map_stream_tree(plan, tree, error);
}
#endif

}  // namespace av2enc::still::plan

#endif  // __has_include("cuda/m1/still_gpu.h")
