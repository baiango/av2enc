// M1 GPU still-picture orchestrator.
//
// This file owns the persistent CUDA runtime (one Context, one Arena, pinned
// staging) and the data-flow between the staged components under
// `src/cuda/pipeline/still/<stage>/`.  It never allocates in the block loop and performs at
// most one stream synchronisation per frame.
//
// Until a component exists, the corresponding work is produced by the CPU M1
// reference encoder (`av2enc::still::encode_still_picture`), which is also the source
// of the frame bytes while the §8.2 syntax writer has no GPU bridge.  The
// fallback is reported per stage through `GpuResult::stages` and is not hidden.
#include "cuda/pipeline/still/still_gpu.h"

#include "cuda/arena.h"
#include "cuda/context.h"
#include "cuda/formats/chroma/chroma_formats.h"
#include "cuda/formats/high_bitdepth_gpu.h"
#include "formats/chroma/chroma_residual.h"
#include "formats/chroma/chroma_still.h"
#include "formats/format.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

#include <cuda_runtime.h>

// Stage component headers, included as their workstreams land.  Everything is
// guarded so this file keeps building before a component exists; the plan
// bridge is the only component with an orchestrator-facing adapter today.
#if __has_include("cuda/pipeline/still/plan/plan_gpu_bridge.h")
#include "cuda/pipeline/still/plan/plan_gpu_bridge.h"
#define AV2ENC_STILL_GPU_HAVE_PLAN_BRIDGE 1
#else
#define AV2ENC_STILL_GPU_HAVE_PLAN_BRIDGE 0
#endif
#if __has_include("cuda/pipeline/still/search/luma_search.h")
#include "cuda/pipeline/still/search/luma_search.h"
#endif
#if __has_include("cuda/pipeline/still/intra/intra_extra.h")
#include "cuda/pipeline/still/intra/intra_extra.h"
#define AV2ENC_STILL_GPU_HAVE_INTRA_EXTRA 1
#else
#define AV2ENC_STILL_GPU_HAVE_INTRA_EXTRA 0
#endif
#if __has_include("cuda/pipeline/still/tx/still_tx.h")
#include "cuda/pipeline/still/tx/still_tx.h"
#endif
#if __has_include("cuda/codec/transform/tx_type/tx_type.h")
#include "cuda/codec/transform/tx_type/tx_type.h"
#define AV2ENC_STILL_GPU_HAVE_TX_TYPE 1
#else
#define AV2ENC_STILL_GPU_HAVE_TX_TYPE 0
#endif
#if __has_include("cuda/codec/transform/secondary/tx_secondary_gpu.h")
#include "cuda/codec/transform/secondary/tx_secondary_gpu.h"
#include "codec/transform/secondary/tx_secondary.h"
#include "codec/transform/secondary/tx_secondary_tables_dev.cuh"
#define AV2ENC_STILL_GPU_HAVE_SECONDARY 1
#else
#define AV2ENC_STILL_GPU_HAVE_SECONDARY 0
#endif
#include "codec/rate/rdo/rdo.h"
#include "codec/intra/mrl/stream/mrl_interaction.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/tcq/cctx.h"
#if __has_include("cuda/codec/transform/tcq/tcq_device.cuh")
#include "cuda/codec/transform/tcq/tcq_device.cuh"
#define AV2ENC_STILL_GPU_HAVE_TCQ_CORE 1
#else
#define AV2ENC_STILL_GPU_HAVE_TCQ_CORE 0
#endif
#if __has_include("cuda/pipeline/still/recon/recon.h")
#include "cuda/pipeline/still/recon/recon.h"
#endif
#if __has_include("cuda/pipeline/still/chroma/still_chroma.h")
#include "cuda/pipeline/still/chroma/still_chroma.h"
#endif
#if __has_include("cuda/pipeline/still/cfl/still_cfl.h")
#include "cuda/pipeline/still/cfl/still_cfl.h"
#endif
#if __has_include("cuda/pipeline/still/sched/sched_bridge.h")
#include "cuda/pipeline/still/sched/sched_bridge.h"
#define AV2ENC_STILL_GPU_HAVE_SCHED_BRIDGE 1
#else
#define AV2ENC_STILL_GPU_HAVE_SCHED_BRIDGE 0
#endif
#if __has_include("pipeline/still/stream/still_stream_bridge.h")
#include "pipeline/still/stream/still_stream_bridge.h"
#define AV2ENC_STILL_GPU_HAVE_STREAM_BRIDGE 1
#else
#define AV2ENC_STILL_GPU_HAVE_STREAM_BRIDGE 0
#endif
#if __has_include("cuda/codec/intra/modes/intra_modes.h")
#include "cuda/codec/intra/modes/intra_modes.h"
#define AV2ENC_STILL_GPU_HAVE_INTRA_MODES 1
#else
#define AV2ENC_STILL_GPU_HAVE_INTRA_MODES 0
#endif
#if __has_include("cuda/codec/intra/palette/palette.h")
#include "cuda/codec/intra/palette/palette.h"
#define AV2ENC_STILL_GPU_HAVE_PALETTE 1
#else
#define AV2ENC_STILL_GPU_HAVE_PALETTE 0
#endif
#if __has_include("cuda/codec/intra/ibc/ibc.h")
#include "cuda/codec/intra/ibc/ibc.h"
#define AV2ENC_STILL_GPU_HAVE_IBC 1
#else
#define AV2ENC_STILL_GPU_HAVE_IBC 0
#endif
#include "codec/transform/qm/qm.h"
#include "codec/rate/rc/rc.h"
#if __has_include("cuda/pipeline/still/cdef/still_cdef_cuda.h")
#include "cuda/pipeline/still/cdef/still_cdef.h"
#include "cuda/pipeline/still/cdef/still_cdef_cuda.h"
#define AV2ENC_STILL_GPU_HAVE_CDEF 1
#else
#define AV2ENC_STILL_GPU_HAVE_CDEF 0
#endif
#if __has_include("cuda/codec/filters/loop_restoration/ccso.h")
#include "codec/filters/loop_restoration/restoration.h"
#include "cuda/codec/filters/loop_restoration/ccso.h"
#define AV2ENC_STILL_GPU_HAVE_CCSO 1
#else
#define AV2ENC_STILL_GPU_HAVE_CCSO 0
#endif
#if __has_include("cuda/codec/filters/loop_restoration/wiener.h")
#include "cuda/codec/filters/loop_restoration/wiener.h"
#define AV2ENC_STILL_GPU_HAVE_WIENER 1
#else
#define AV2ENC_STILL_GPU_HAVE_WIENER 0
#endif
#if __has_include("cuda/codec/filters/loop_restoration/chroma/lr_chroma.h")
#include "cuda/codec/filters/loop_restoration/chroma/lr_chroma.h"
#define AV2ENC_STILL_GPU_HAVE_LR_CHROMA 1
#else
#define AV2ENC_STILL_GPU_HAVE_LR_CHROMA 0
#endif
#if __has_include("cuda/codec/filters/pc_wiener/pc_wiener.h")
#include "cuda/codec/filters/pc_wiener/pc_wiener.h"
#define AV2ENC_STILL_GPU_HAVE_PC_WIENER 1
#else
#define AV2ENC_STILL_GPU_HAVE_PC_WIENER 0
#endif
#if __has_include("cuda/codec/filters/gdf/gdf.h")
#include "cuda/codec/filters/gdf/gdf.h"
#define AV2ENC_STILL_GPU_HAVE_GDF 1
#else
#define AV2ENC_STILL_GPU_HAVE_GDF 0
#endif

// Full GPU frame pipeline: real plan tree + luma search + chroma/CfL + TX +
// reconstruction + deblock + GPU entropy assembly.  Every component must be
// present; otherwise the frame falls back to the CPU reference.
#if AV2ENC_STILL_GPU_HAVE_PLAN_BRIDGE && AV2ENC_STILL_GPU_HAVE_INTRA_MODES && \
    AV2ENC_STILL_GPU_HAVE_SCHED_BRIDGE && AV2ENC_STILL_GPU_HAVE_STREAM_BRIDGE && \
    __has_include("cuda/pipeline/still/search/luma_search.h") && \
    __has_include("cuda/pipeline/still/intra/intra_extra.h") && \
    __has_include("cuda/pipeline/still/chroma/still_chroma.h") && \
    __has_include("cuda/pipeline/still/tx/still_tx.h") && \
    __has_include("cuda/pipeline/still/recon/recon.h") && \
    __has_include("cuda/codec/intra/palette/palette.h") && \
    __has_include("cuda/codec/intra/ibc/ibc.h")
#define AV2ENC_STILL_GPU_HAVE_FULL 1
#else
#define AV2ENC_STILL_GPU_HAVE_FULL 0
#endif
// Keep the old name as an alias for the stage-1-only build path.
#define AV2ENC_STILL_GPU_HAVE_STAGE1 AV2ENC_STILL_GPU_HAVE_FULL

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::cuda::still_gpu {
namespace {

// The TX/recon components live in the sibling `av2enc::cuda::still` namespace and
// the shared predictors in `av2enc::cuda::intra_modes`; unqualified lookup from
// `still_gpu` would not find them.
#if AV2ENC_STILL_GPU_HAVE_FULL
namespace cm1 = ::av2enc::cuda::still;
namespace im = ::av2enc::cuda::intra_modes;
namespace srch = ::av2enc::cuda::still::search;
namespace iext = ::av2enc::cuda::still::intra_extra;
namespace sch = ::av2enc::cuda::still::sched;
namespace still_stream = ::av2enc::still::stream;
#endif

constexpr std::uint64_t kMiB = 1024ull * 1024ull;
// The M1 still-picture working set is a few MiB.  Keep the arena small while
// other agents share the GPU; it is still device-derived (see ensure_runtime).
constexpr std::uint64_t kArenaFloorBytes = 256ull * kMiB;
constexpr std::uint64_t kDeviceHeadroomBytes = 512ull * kMiB;
constexpr std::uint64_t kArenaSlackBytes = 16ull * kMiB;

// ---------------------------------------------------------------------------
// Geometry helpers (mirror the CPU M1 recon planes)
// ---------------------------------------------------------------------------

struct Geometry {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t padded_width = 0;
  std::uint32_t padded_height = 0;
  std::uint32_t u_stride = 0;
  std::uint32_t chroma_padded_height = 0;
  std::uint16_t mi_cols = 0;
  std::uint16_t mi_rows = 0;
  std::uint16_t sb_cols = 0;
  std::uint16_t sb_rows = 0;
};

bool make_geometry(std::uint32_t width, std::uint32_t height,
                   Geometry& geometry) {
  if (width == 0 || height == 0 || width > 65535 || height > 65535) {
    return false;
  }
  geometry.width = width;
  geometry.height = height;
  // §5.18.4.4: MiCols/MiRows are rounded to 8 pixels, not to 4.
  const std::uint32_t mi_cols = 2u * ((width + 7u) >> 3);
  const std::uint32_t mi_rows = 2u * ((height + 7u) >> 3);
  const std::uint32_t coded_w = mi_cols * 4u;
  const std::uint32_t coded_h = mi_rows * 4u;
  // Whole 64x64 superblocks, as in src/encoder/m1/still_encoder.cpp.
  geometry.padded_width = (coded_w + 63u) & ~63u;
  geometry.padded_height = (coded_h + 63u) & ~63u;
  geometry.u_stride = geometry.padded_width / 2u;
  geometry.chroma_padded_height = geometry.padded_height / 2u;
  geometry.mi_cols = static_cast<std::uint16_t>(mi_cols);
  geometry.mi_rows = static_cast<std::uint16_t>(mi_rows);
  geometry.sb_cols =
      static_cast<std::uint16_t>((width + 63u) / 64u);
  geometry.sb_rows =
      static_cast<std::uint16_t>((height + 63u) / 64u);
  return true;
}

std::size_t luma_plane_bytes(const Geometry& g) {
  return static_cast<std::size_t>(g.padded_width) *
         static_cast<std::size_t>(g.padded_height);
}

std::size_t chroma_plane_bytes(const Geometry& g) {
  return static_cast<std::size_t>(g.u_stride) *
         static_cast<std::size_t>(g.chroma_padded_height);
}

std::size_t visible_chroma_pixels(std::uint32_t width, std::uint32_t height) {
  return static_cast<std::size_t>((width + 1u) / 2u) *
         static_cast<std::size_t>((height + 1u) / 2u);
}

// Largest valid M1 transform side <= `dim`.  64 is signalled as the adjusted
// 32x32 transform, so 32 is the cap.
std::uint8_t pick_tx_side(std::uint32_t dim) {
  std::uint32_t tx = 32;
  while (tx > dim && tx > 4) {
    tx >>= 1;
  }
  return static_cast<std::uint8_t>(tx);
}

// ---------------------------------------------------------------------------
// Placeholder plan
//
// One PARTITION_NONE block per 64x64 superblock.  This is deliberately not the
// M1 search tree (that is `src/cuda/m1/plan/`); it only gives the data-flow
// test a consistent geometry and buffer layout while the plan stage is absent.
// ---------------------------------------------------------------------------

#if !AV2ENC_STILL_GPU_HAVE_FULL
bool build_placeholder_plan(const ::av2enc::still::StillPicture& picture,
                            FramePlan& plan,
                            std::span<BlockPlan> blocks,
                            std::string& error) {
  Geometry g;
  if (!make_geometry(picture.width, picture.height, g)) {
    error = "unsupported picture dimensions";
    return false;
  }
  const std::uint32_t block_count =
      static_cast<std::uint32_t>(g.sb_cols) *
      static_cast<std::uint32_t>(g.sb_rows);
  if (blocks.size() < block_count) {
    error = "placeholder plan: block array too small";
    return false;
  }

  plan = FramePlan{};
  plan.width = g.width;
  plan.height = g.height;
  plan.padded_width = g.padded_width;
  plan.padded_height = g.padded_height;
  plan.y_stride = g.padded_width;
  plan.u_stride = g.u_stride;
  plan.v_stride = g.u_stride;
  plan.mi_cols = g.mi_cols;
  plan.mi_rows = g.mi_rows;
  plan.sb_cols = g.sb_cols;
  plan.sb_rows = g.sb_rows;
  plan.sb_size = 64;
  plan.chroma_shift_x = 1;
  plan.chroma_shift_y = 1;
  plan.is_placeholder = 1;
  plan.max_chroma_w = 32;
  plan.max_chroma_h = 32;
  plan.block_count = block_count;
  plan.leaf_count = block_count;
  plan.blocks = blocks.first(block_count);

  std::uint32_t coeff_y = 0;
  std::uint32_t coeff_u = 0;
  std::uint32_t coeff_v = 0;
  std::uint32_t index = 0;
  for (std::uint32_t sy = 0; sy < g.sb_rows; ++sy) {
    for (std::uint32_t sx = 0; sx < g.sb_cols; ++sx) {
      const std::uint32_t x = sx * 64u;
      const std::uint32_t y = sy * 64u;
      const std::uint32_t w = std::min<std::uint32_t>(64u, g.width - x);
      const std::uint32_t h = std::min<std::uint32_t>(64u, g.height - y);
      const std::uint32_t cw = (w + 1u) / 2u;
      const std::uint32_t ch = (h + 1u) / 2u;

      BlockPlan& block = blocks[index];
      block = BlockPlan{};
      block.mi_row = static_cast<std::uint16_t>(y / 4u);
      block.mi_col = static_cast<std::uint16_t>(x / 4u);
      block.luma_w = static_cast<std::uint8_t>(w);
      block.luma_h = static_cast<std::uint8_t>(h);
      block.chroma_w = static_cast<std::uint8_t>(cw);
      block.chroma_h = static_cast<std::uint8_t>(ch);
      block.tx_w = pick_tx_side(w);
      block.tx_h = pick_tx_side(h);
      block.chroma_tx_w = pick_tx_side(cw);
      block.chroma_tx_h = pick_tx_side(ch);
      block.tx_upscale_w = (w > block.tx_w) ? 2 : 1;
      block.tx_upscale_h = (h > block.tx_h) ? 2 : 1;
      block.has_chroma = (cw >= 4u && ch >= 4u) ? 1 : 0;
      block.tree_depth = 0;
      block.plane_offset_y =
          y * g.padded_width + x;
      block.plane_offset_u =
          (y / 2u) * g.u_stride + (x / 2u);
      block.plane_offset_v = block.plane_offset_u;
      block.coeff_offset_y = coeff_y;
      block.coeff_offset_u = coeff_u;
      block.coeff_offset_v = coeff_v;

      std::uint8_t avail = 0;
      if (block.mi_col > 0) {
        avail = static_cast<std::uint8_t>(avail | kAvailLeft);
      }
      if (block.mi_row > 0) {
        avail = static_cast<std::uint8_t>(avail | kAvailAbove);
      }
      if (block.mi_row > 0 && block.mi_col > 0) {
        avail = static_cast<std::uint8_t>(avail | kAvailAboveLeft);
      }
      const std::uint32_t mi_w = (w + 3u) / 4u;
      const std::uint32_t mi_h = (h + 3u) / 4u;
      if (block.mi_col + mi_w < g.mi_cols) {
        avail = static_cast<std::uint8_t>(avail | kAvailAboveRight);
      }
      if (block.mi_row + mi_h < g.mi_rows) {
        avail = static_cast<std::uint8_t>(avail | kAvailBelowLeft);
      }
      block.avail = avail;

      coeff_y += static_cast<std::uint32_t>(block.tx_w) * block.tx_h;
      if (block.has_chroma != 0) {
        coeff_u += static_cast<std::uint32_t>(block.chroma_tx_w) *
                   block.chroma_tx_h;
        coeff_v += static_cast<std::uint32_t>(block.chroma_tx_w) *
                   block.chroma_tx_h;
      }
      ++index;
    }
  }

  plan.coeff_count_y = coeff_y;
  plan.coeff_count_u = coeff_u;
  plan.coeff_count_v = coeff_v;
  return true;
}
#endif  // !AV2ENC_STILL_GPU_HAVE_FULL

// Validate a plan produced by the plan stage.  Catches geometry/offset bugs
// before they become kernel out-of-bounds reads.
bool validate_plan(const FramePlan& plan, std::string& error) {
  Geometry g;
  if (!make_geometry(plan.width, plan.height, g)) {
    error = "plan: bad dimensions";
    return false;
  }
  if (plan.padded_width != g.padded_width ||
      plan.padded_height != g.padded_height ||
      plan.y_stride != g.padded_width || plan.u_stride != g.u_stride ||
      plan.v_stride != g.u_stride || plan.mi_cols != g.mi_cols ||
      plan.mi_rows != g.mi_rows) {
    error = "plan: geometry disagrees with the M1 padded layout";
    return false;
  }
  if (plan.blocks.size() != plan.block_count || plan.block_count == 0) {
    error = "plan: block_count does not match the block array";
    return false;
  }
  const std::size_t luma_bytes = luma_plane_bytes(g);
  const std::size_t chroma_bytes = chroma_plane_bytes(g);
  std::uint32_t last_coeff_y = 0;
  std::uint32_t last_coeff_u = 0;
  std::uint32_t last_coeff_v = 0;
  for (std::uint32_t i = 0; i < plan.block_count; ++i) {
    const BlockPlan& block = plan.blocks[i];
    const bool has_luma = block.luma_w != 0 && block.luma_h != 0;
    if (!has_luma && block.has_chroma == 0) {
      error = "plan: block codes neither luma nor chroma";
      return false;
    }
    if (has_luma && (block.tx_w == 0 || block.tx_h == 0)) {
      error = "plan: zero-sized luma transform";
      return false;
    }
    if (has_luma) {
      const std::size_t luma_end =
          static_cast<std::size_t>(block.plane_offset_y) +
          static_cast<std::size_t>(block.luma_h - 1u) * plan.y_stride +
          block.luma_w;
      if (luma_end > luma_bytes) {
        error = "plan: luma block outside the padded plane";
        return false;
      }
    }
    if (block.has_chroma != 0) {
      const std::size_t chroma_end =
          static_cast<std::size_t>(block.plane_offset_u) +
          static_cast<std::size_t>(block.chroma_h - 1u) * plan.u_stride +
          block.chroma_w;
      if (chroma_end > chroma_bytes) {
        error = "plan: chroma block outside the padded plane";
        return false;
      }
    }
    // Extended/uneven partitions can own chroma blocks whose syntax order
    // differs from the per-plane coefficient allocation order, so offsets are
    // not necessarily monotonic.  The stream bridge validates that the
    // declared per-plane ranges do not overlap.
    last_coeff_y = std::max(last_coeff_y, block.coeff_offset_y);
    last_coeff_u = std::max(last_coeff_u, block.coeff_offset_u);
    last_coeff_v = std::max(last_coeff_v, block.coeff_offset_v);
  }
  if ((plan.coeff_count_y > 0 && last_coeff_y >= plan.coeff_count_y) ||
      (plan.coeff_count_u > 0 && last_coeff_u >= plan.coeff_count_u) ||
      (plan.coeff_count_v > 0 && last_coeff_v >= plan.coeff_count_v)) {
    error = "plan: coefficient counts are too small";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// 8-bit <-> 16-bit plane conversions for the uint16_t filter cores
// ---------------------------------------------------------------------------

__global__ void u8_to_u16_plane_kernel(const std::uint8_t* src, int src_stride,
                                       std::uint16_t* dst, int dst_stride,
                                       int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  dst[static_cast<std::size_t>(y) * dst_stride + x] =
      static_cast<std::uint16_t>(
          src[static_cast<std::size_t>(y) * src_stride + x]);
}

// PC-Wiener/GDF write uint16 samples; the 8-bit still stream clips them to
// [0, 255] exactly as the CPU reference does.
__global__ void u16_to_u8_plane_kernel(const std::uint16_t* src,
                                       int src_stride, std::uint8_t* dst,
                                       int dst_stride, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int v = static_cast<int>(
      src[static_cast<std::size_t>(y) * src_stride + x]);
  dst[static_cast<std::size_t>(y) * dst_stride + x] =
      static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

void launch_u8_to_u16(const std::uint8_t* src, int src_stride,
                      std::uint16_t* dst, int dst_stride, int width, int height,
                      cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid(static_cast<unsigned>((width + 31) / 32),
                  static_cast<unsigned>((height + 7) / 8));
  u8_to_u16_plane_kernel<<<grid, block, 0, stream>>>(src, src_stride, dst,
                                                     dst_stride, width, height);
}

void launch_u16_to_u8(const std::uint16_t* src, int src_stride,
                      std::uint8_t* dst, int dst_stride, int width, int height,
                      cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid(static_cast<unsigned>((width + 31) / 32),
                  static_cast<unsigned>((height + 7) / 8));
  u16_to_u8_plane_kernel<<<grid, block, 0, stream>>>(src, src_stride, dst,
                                                     dst_stride, width, height);
}

// ---------------------------------------------------------------------------
// Host buffers
// ---------------------------------------------------------------------------

struct HostBuffers {
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::vector<std::int32_t> coeff_y;
  std::vector<std::int32_t> coeff_u;
  std::vector<std::int32_t> coeff_v;
  std::vector<std::uint8_t> pred_y;
  std::vector<std::uint8_t> pred_u;
  std::vector<std::uint8_t> pred_v;
  std::vector<BlockDecision> decisions;
};

bool allocate_host_buffers(const FramePlan& plan, HostBuffers& host,
                           std::string& error) {
  Geometry g;
  if (!make_geometry(plan.width, plan.height, g)) {
    error = "host buffers: bad plan dimensions";
    return false;
  }
  const std::size_t luma_bytes = luma_plane_bytes(g);
  const std::size_t chroma_bytes = chroma_plane_bytes(g);
  host.recon_y.assign(luma_bytes, 0);
  host.recon_u.assign(chroma_bytes, 0);
  host.recon_v.assign(chroma_bytes, 0);
  host.pred_y.assign(luma_bytes, 0);
  host.pred_u.assign(chroma_bytes, 0);
  host.pred_v.assign(chroma_bytes, 0);
  host.coeff_y.assign(plan.coeff_count_y, 0);
  host.coeff_u.assign(plan.coeff_count_u, 0);
  host.coeff_v.assign(plan.coeff_count_v, 0);
  host.decisions.assign(plan.block_count, BlockDecision{});
  return true;
}

// ---------------------------------------------------------------------------
// Persistent device runtime
// ---------------------------------------------------------------------------

struct DeviceBuffers {
  U8Planes src;
  U8Planes recon;
  I32Planes coeff;
  U8Planes pred;
  BlockPlan* blocks = nullptr;
  BlockDecision* decisions = nullptr;
  std::size_t blocks_bytes = 0;
  std::size_t decisions_bytes = 0;
  std::uint64_t total_bytes = 0;

  // Full GPU pipeline scratch (packed per-block prediction / inverse residual,
  // search and TX/recon descriptors, per-block results, reconstruction state).
#if AV2ENC_STILL_GPU_HAVE_FULL
  std::uint8_t* pred_packed[3] = {nullptr, nullptr, nullptr};
  std::size_t pred_packed_bytes[3] = {0, 0, 0};
  std::int16_t* residual_packed[3] = {nullptr, nullptr, nullptr};
  std::size_t residual_packed_bytes[3] = {0, 0, 0};
  srch::LumaSearchBlock* luma_blocks = nullptr;
  std::size_t luma_blocks_bytes = 0;
  srch::LumaSearchResult* luma_results = nullptr;
  std::size_t luma_results_bytes = 0;
  // Per-candidate base SSE (search debug) for the MRL ordered combine, and the
  // MRL/DIP choice plus their chosen-prediction descriptors/scratch.
  std::int64_t* candidate_sse = nullptr;
  std::size_t candidate_sse_bytes = 0;
  iext::Choice* intra_choices = nullptr;
  std::size_t intra_choices_bytes = 0;
  mrl::MrlJob* mrl_jobs = nullptr;
  std::size_t mrl_jobs_bytes = 0;
  dip::DipJob* dip_jobs = nullptr;
  std::size_t dip_jobs_bytes = 0;
  ibp::IbpJob* ibp_jobs = nullptr;
  std::size_t ibp_jobs_bytes = 0;
  std::uint8_t* mrl_pred = nullptr;
  std::uint8_t* dip_pred = nullptr;
  std::size_t extra_pred_bytes = 0;
  im::IntraJob* luma_jobs = nullptr;
  std::size_t luma_jobs_bytes = 0;
  // Chroma decision descriptors.  They are built on the host but uploaded once
  // and patched on the device per block, so the full-UV candidate inputs
  // (luma mode / AngleDeltaY, UVSmooth neighbour flags) come from the device
  // decision state instead of a host round trip.
  cm1::StillChromaJob* chroma_jobs = nullptr;
  std::size_t chroma_jobs_bytes = 0;
  cm1::StillChromaResult* chroma_results = nullptr;
  std::size_t chroma_results_bytes = 0;
  // Per-MI record of the final luma mode/angle, used by CHROMA_PART leaves
  // (§5.20.5.6 reads the co-located luma-tree YMode).
  std::int8_t* mi_luma_mode = nullptr;
  std::int8_t* mi_luma_angle = nullptr;
  std::size_t mi_luma_bytes = 0;
  // §5.20.8 palette: one 8-entry colour table per leaf (indexed by leaf) and a
  // 64x64 scratch colour map for the single-block prediction kernel.
  std::uint8_t* palette_colors = nullptr;
  std::size_t palette_colors_bytes = 0;
  std::uint8_t* palette_map = nullptr;
  std::size_t palette_map_bytes = 0;
  std::int32_t* eob = nullptr;  // 3 * block_count (Y, U, V per block)
  std::size_t eob_bytes = 0;
  StillTxJob* tx_jobs = nullptr;  // 3 * block_count
  std::size_t tx_jobs_bytes = 0;
  int* tx_errors = nullptr;
#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
  // §5.20.7.27/§9.6 intra TX-type candidate search (one block at a time).
  std::int16_t* tx_cand_adj = nullptr;      // candidates * kTxTypeAdjElems
  std::size_t tx_cand_adj_bytes = 0;
  double* tx_cand_coeff = nullptr;          // candidates * kTxTypeAdjElems
  std::size_t tx_cand_coeff_bytes = 0;
  std::int32_t* tx_cand_levels = nullptr;   // candidates * kTxTypeAdjElems
  std::size_t tx_cand_levels_bytes = 0;
  std::int16_t* tx_cand_recon = nullptr;    // candidates * kTxTypeCodedElems
  std::size_t tx_cand_recon_bytes = 0;
  TxTypeCandJob* tx_cand_jobs = nullptr;
  std::size_t tx_cand_jobs_bytes = 0;
  TxBlockJob* tx_cand_tx_jobs = nullptr;
  std::size_t tx_cand_tx_jobs_bytes = 0;
  TxTypeBlockJob* tx_choose_block = nullptr;
  std::size_t tx_choose_block_bytes = 0;
  TxTypeCandidateMetrics* tx_metrics = nullptr;
  std::size_t tx_metrics_bytes = 0;
  TxTypeDecision* tx_decision = nullptr;
  std::size_t tx_decision_bytes = 0;
  TxParams* tx_param_table = nullptr;
  std::size_t tx_param_table_bytes = 0;
  StillTxJob* tx_probe_job = nullptr;
  std::size_t tx_probe_job_bytes = 0;
  std::int16_t* tx_residual_coded = nullptr;
  std::size_t tx_residual_coded_bytes = 0;
  std::int16_t* tx_residual_adj = nullptr;
  std::size_t tx_residual_adj_bytes = 0;
  // Per-block chosen luma TX type (3 slots per block: Y/U/V).
  std::int32_t* tx_types = nullptr;
  std::size_t tx_types_bytes = 0;
  // §5.20.8.2 candidate tables, one entry per TX size (16 TYPE values each),
  // plus the per-size mapped-mode / legality / symbol-bit tables.  Built once
  // per frame; the per-block prepass only rewrites `type_bits`.
  TxTypeCandJob* tx_type_cands = nullptr;
  std::size_t tx_type_cands_bytes = 0;
  TxBlockJob* tx_type_jobs = nullptr;
  std::size_t tx_type_jobs_bytes = 0;
  TxTypeBlockJob* tx_type_blocks = nullptr;
  std::size_t tx_type_blocks_bytes = 0;
  std::int8_t* tx_type_map = nullptr;
  std::size_t tx_type_map_bytes = 0;
  std::uint8_t* tx_type_legal = nullptr;
  std::size_t tx_type_legal_bytes = 0;
  double* tx_type_bits = nullptr;
  std::size_t tx_type_bits_bytes = 0;
#endif
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
  // §7.15.3 IST candidate search: 3 sets x 3 sec_tx_type slots per block.
  std::int32_t* ist_levels = nullptr;     // 9 * kTxTypeAdjElems
  std::size_t ist_levels_bytes = 0;
  std::int32_t* ist_dequant = nullptr;    // 9 * kTxTypeAdjElems
  std::size_t ist_dequant_bytes = 0;
  std::int16_t* ist_residual = nullptr;   // 9 * kTxTypeCodedElems
  std::size_t ist_residual_bytes = 0;
  double* ist_sse = nullptr;              // 9
  std::size_t ist_sse_bytes = 0;
  std::int32_t* ist_eob = nullptr;        // 9
  std::size_t ist_eob_bytes = 0;
  std::uint8_t* ist_valid = nullptr;      // 9
  std::size_t ist_valid_bytes = 0;
  double* ist_bits = nullptr;             // 9
  std::size_t ist_bits_bytes = 0;
  TxSecondaryJob* ist_jobs = nullptr;     // 9
  std::size_t ist_jobs_bytes = 0;
  TxBlockJob* ist_block_jobs = nullptr;   // 9
  std::size_t ist_block_jobs_bytes = 0;
  std::int32_t* ist_secs = nullptr;       // per block
  std::size_t ist_secs_bytes = 0;
  std::int32_t* ist_sets = nullptr;       // per block
  std::size_t ist_sets_bytes = 0;
  double* ist_base_sse = nullptr;         // 1
  std::size_t ist_base_sse_bytes = 0;
  std::int32_t* ist_primary_type = nullptr;  // 1
  std::size_t ist_primary_type_bytes = 0;
  // Per-size/type/mapped-mode/set/sec candidate table.
  std::int8_t* ist_map = nullptr;              // 25 * 13 * 7 * 4
  std::size_t ist_map_bytes = 0;
  TxSecondaryJob* ist_job_table = nullptr;     // 25 * 2 * 13 * 9
  std::size_t ist_job_table_bytes = 0;
  TxBlockJob* ist_block_table = nullptr;       // 25 * 2 * 9
  std::size_t ist_block_table_bytes = 0;
  double* ist_bit_table = nullptr;             // 25 * 2 * 13 * 9
  std::size_t ist_bit_table_bytes = 0;
  std::uint8_t* ist_set_count = nullptr;       // 25 * 2 * 13
  std::size_t ist_set_count_bytes = 0;
#endif
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
  // §7.14.6 cross-component transform search (7 types x U/V).
  std::int32_t* cctx_levels = nullptr;    // 7 * 2 * kTxTypeAdjElems
  std::size_t cctx_levels_bytes = 0;
  std::int32_t* cctx_dequant = nullptr;
  std::size_t cctx_dequant_bytes = 0;
  std::int16_t* cctx_residual = nullptr;  // 7 * 2 * kTxTypeCodedElems
  std::size_t cctx_residual_bytes = 0;
  double* cctx_u_coeff = nullptr;
  std::size_t cctx_u_coeff_bytes = 0;
  double* cctx_v_coeff = nullptr;
  std::size_t cctx_v_coeff_bytes = 0;
  double* cctx_sse = nullptr;             // 7
  std::size_t cctx_sse_bytes = 0;
  std::int32_t* cctx_eob = nullptr;       // 7 * 2
  std::size_t cctx_eob_bytes = 0;
  TxBlockJob* cctx_jobs = nullptr;        // 7 * 2
  std::size_t cctx_jobs_bytes = 0;
  std::int32_t* cctx_types = nullptr;     // per block
  std::size_t cctx_types_bytes = 0;
#endif
  cm1::ReconWriteJob* recon_jobs = nullptr;  // 3 * block_count
  std::size_t recon_jobs_bytes = 0;
  cm1::ReconBlockInfo* recon_infos = nullptr;  // block_count
  std::size_t recon_infos_bytes = 0;
  void* search_scratch = nullptr;
  std::size_t search_scratch_bytes = 0;
  deblock::DeblockMi* deblock_mi[3] = {nullptr, nullptr, nullptr};
  std::size_t deblock_mi_bytes[3] = {0, 0, 0};
  std::uint8_t* decoded[3] = {nullptr, nullptr, nullptr};
  std::size_t decoded_bytes[3] = {0, 0, 0};
  int* intra_joint = nullptr;
  std::size_t intra_joint_bytes = 0;
  std::uint8_t* uv_smooth = nullptr;
  std::size_t uv_smooth_bytes = 0;
  // Post-deblock in-loop filters (§7.18 CDEF).  CDEF needs a destination
  // distinct from the unfiltered reconstruction, so it writes its own plane
  // set; the orchestrator downloads whichever set is current.
  U8Planes filter;
  std::uint8_t* cdef_skip = nullptr;             // luma MI grid, all zero
  std::uint8_t* cdef_lossless = nullptr;         // luma MI grid, all zero
  std::uint8_t* cdef_lossless_chroma = nullptr;  // chroma MI grid, all zero
  std::size_t cdef_mi_bytes = 0;
  std::size_t cdef_chroma_mi_bytes = 0;
  std::int8_t* cdef_idx = nullptr;               // per-64x64 grid
  std::size_t cdef_idx_bytes = 0;
  // §7.19 CCSO: per-plane unit-enable grids and decoded offset tables.
  std::uint8_t* ccso_blks[3] = {nullptr, nullptr, nullptr};
  std::size_t ccso_blk_bytes[3] = {0, 0, 0};
  std::int16_t* ccso_offsets[3] = {nullptr, nullptr, nullptr};
  std::size_t ccso_offset_bytes[3] = {0, 0, 0};
  // §7.20 loop restoration.  `lr` receives the final LrFrame; `curr` for the
  // stage is the deblocked reconstruction (`recon`) and `cdef` is `filter`
  // once CDEF/CCSO ran.  The per-plane unit descriptor arrays are uploaded
  // from the CPU reference's §5.20.10.5/.6 decision.
  U8Planes lr;
  std::uint8_t* lr_unit_type[3] = {nullptr, nullptr, nullptr};
  std::size_t lr_unit_type_bytes[3] = {0, 0, 0};
  std::int16_t* lr_unit_coeffs[3] = {nullptr, nullptr, nullptr};
  std::size_t lr_unit_coeffs_bytes[3] = {0, 0, 0};
  // §7.20.1 LrTxSkip luma MI grid for PC-Wiener (per-4x4 eob == 0).
  std::uint8_t* lr_tx_skip = nullptr;
  std::size_t lr_tx_skip_bytes = 0;
  // §5.18.6.2 QM: per-block adjusted-size q2 matrices.
  std::int32_t* q2 = nullptr;
  std::size_t q2_bytes = 0;
  // §5.4.9 / §5.18.7.1 device segment map for --still-per-segment-q.  One
  // entry per superblock plus the variance scratch used to derive it.
  std::uint8_t* sb_segment = nullptr;
  std::size_t sb_segment_bytes = 0;
  std::uint8_t* sb_qindex = nullptr;
  std::size_t sb_qindex_bytes = 0;
  std::uint64_t* sb_variance = nullptr;
  std::size_t sb_variance_bytes = 0;
  // Luma-only 16-bit planes for PC-Wiener (§7.20.4) and GDF (§7.20.5).
  U16Planes luma16_curr;
  U16Planes luma16_cdef;
  U16Planes luma16_lr;
#endif
};

// Persistent per-frame filter workspaces.  These own arena memory for the
// components that expose a workspace (chroma LR, PC-Wiener, GDF tables); they
// are recreated at a frame boundary whenever the arena is rebuilt.
struct FilterWorkspaces {
#if AV2ENC_STILL_GPU_HAVE_LR_CHROMA
  lr_chroma::LrChromaWorkspace chroma_lr;
#endif
#if AV2ENC_STILL_GPU_HAVE_PC_WIENER
  pc_wiener::PcWienerWorkspace pc_wiener;
#endif
#if AV2ENC_STILL_GPU_HAVE_GDF
  gdf::GdfDeviceBuffers gdf;
  bool gdf_ready = false;
#endif
  int mi_cols = 0;
  int mi_rows = 0;
  // Frame SbSize the workspaces were sized for.  A later encode with a
  // different --still-sb128/--still-sb256 mode has different LR unit sizes.
  int sb_size = 0;

  void release() {
#if AV2ENC_STILL_GPU_HAVE_LR_CHROMA
    chroma_lr.reset();
#endif
#if AV2ENC_STILL_GPU_HAVE_PC_WIENER
    pc_wiener.reset();
#endif
#if AV2ENC_STILL_GPU_HAVE_GDF
    gdf = gdf::GdfDeviceBuffers{};
    gdf_ready = false;
#endif
    mi_cols = 0;
    mi_rows = 0;
    sb_size = 0;
  }
};

struct GpuRuntime {
  bool init_attempted = false;
  bool ready = false;
  std::string init_error;
  int device = -1;
  Context context;
  Arena arena;
  StagingBuffer upload;
  StagingBuffer download;
  // Per-block descriptor staging for the TX decision stages (one unique slot
  // per block so an async H2D is never overwritten while pending).
  StagingBuffer tx_stage;
  DeviceBuffers buffers;
  FilterWorkspaces filters;
  std::uint64_t arena_capacity = 0;

  ~GpuRuntime() {
    filters.release();
    buffers = DeviceBuffers{};
    tx_stage.release();
    download.release();
    upload.release();
    arena.release();
    context.shutdown();
  }
};

GpuRuntime& runtime() {
  static GpuRuntime instance;
  return instance;
}

std::size_t coeff_bytes(std::uint32_t count) {
  return static_cast<std::size_t>(count) * sizeof(std::int32_t);
}

std::size_t packed_plane_pixels(const FramePlan& plan, int plane) {
  std::size_t total = 0;
  for (const BlockPlan& block : plan.blocks) {
    if (plane == 0) {
      total += static_cast<std::size_t>(block.luma_w) * block.luma_h;
    } else if (block.has_chroma != 0) {
      total += static_cast<std::size_t>(block.chroma_w) * block.chroma_h;
    }
  }
  return total;
}

#if AV2ENC_STILL_GPU_HAVE_FULL
constexpr int kSearchMaxBlocks = 1;
constexpr int kSearchCandidatesPerChunk = 61;
constexpr int kSearchMaxBlockArea = 64 * 64;
// One candidate per AV2 TX_TYPE for the §5.20.7.27 intra search.
constexpr int kTxTypeSearchSlots = 16;

// Upper bound on the §5.20.10.4 loop-restoration unit grid: the smallest
// normative luma unit is 64 samples (16 MI cells) and the chroma grid at
// 4:2:0/32 is the same shape, so one slot per 16x16 MI tile covers both.
std::size_t lr_unit_slots(const FramePlan& plan) {
  const std::size_t cols =
      (static_cast<std::size_t>(plan.mi_cols) + 15u) / 16u;
  const std::size_t rows =
      (static_cast<std::size_t>(plan.mi_rows) + 15u) / 16u;
  return cols * rows;
}

// Superblocks in the frame plan's effective (intra-clamped) SbSize grid.
std::size_t sb_slot_count(const FramePlan& plan) {
  return static_cast<std::size_t>(plan.sb_cols) *
         static_cast<std::size_t>(plan.sb_rows);
}

// Adjusted-size element count of the QM q2 matrices: one per TX block/plane.
std::size_t q2_elems(const FramePlan& plan) {  std::size_t total = 0;
  for (const BlockPlan& block : plan.blocks) {
    total += static_cast<std::size_t>(block.tx_w) * block.tx_h;
    if (block.has_chroma != 0) {
      total += 2u * static_cast<std::size_t>(block.chroma_tx_w) *
               block.chroma_tx_h;
    }
  }
  return total;
}

std::size_t full_device_bytes(const FramePlan& plan) {
  const std::size_t n = plan.block_count;
  const std::size_t mi_cells = static_cast<std::size_t>(plan.mi_cols) *
                               static_cast<std::size_t>(plan.mi_rows);
  const std::size_t chroma_mi =
      static_cast<std::size_t>(plan.mi_cols / 2u) *
      static_cast<std::size_t>(plan.mi_rows / 2u);
  std::size_t total = 0;
  for (int p = 0; p < 3; ++p) {
    const std::size_t packed = packed_plane_pixels(plan, p);
    total += packed;                            // prediction
    total += packed * sizeof(std::int16_t);     // inverse residual
  }
  total += n * sizeof(srch::LumaSearchBlock);
  total += n * sizeof(srch::LumaSearchResult);
  total += n * kSearchCandidatesPerChunk * sizeof(std::int64_t);
  total += n * sizeof(iext::Choice);
  total += n * sizeof(mrl::MrlJob);
  total += n * sizeof(dip::DipJob);
  total += n * sizeof(ibp::IbpJob);
  total += 2u * kSearchMaxBlockArea;  // MRL/DIP chosen-prediction scratch
  total += n * sizeof(im::IntraJob);
  total += n * sizeof(cm1::StillChromaJob);
  total += n * sizeof(cm1::StillChromaResult);
  total += n * sizeof(cm1::ReconBlockInfo);
  total += 3u * n * sizeof(StillTxJob);
  total += 3u * n * sizeof(cm1::ReconWriteJob);
  total += 3u * n * sizeof(std::int32_t);
  total += sizeof(int);  // tx_errors
#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
  // TX-type candidate search scratch (one block at a time).
  total += static_cast<std::size_t>(kTxTypeSearchSlots) * kTxTypeAdjElems *
           (sizeof(std::int16_t) + sizeof(double) + sizeof(std::int32_t));
  total += static_cast<std::size_t>(kTxTypeSearchSlots) *
           kTxTypeCodedElems * sizeof(std::int16_t);
  total += kTxTypeSearchSlots *
           (sizeof(TxTypeCandJob) + sizeof(TxBlockJob) +
            sizeof(TxTypeCandidateMetrics));
  total += sizeof(TxTypeBlockJob) + sizeof(TxTypeDecision) +
           sizeof(StillTxJob) + 16u * sizeof(TxParams);
  total += kTxTypeCodedElems * sizeof(std::int16_t);
  total += kTxTypeAdjElems * sizeof(std::int16_t);
  total += 3u * n * sizeof(std::int32_t);  // tx_types
  total += 25u * 16u * (sizeof(TxTypeCandJob) + sizeof(TxBlockJob)) +
           25u * sizeof(TxTypeBlockJob);
  total += 25u * 13u * 7u * 4u;                       // mapped-mode table
  total += 25u * 13u * 16u;                           // legality
  total += 25u * 13u * 16u * sizeof(double);          // symbol bits
#endif
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
  total += 9u * kTxTypeAdjElems * sizeof(std::int32_t) * 2u;
  total += 9u * kTxTypeCodedElems * sizeof(std::int16_t);
  total += 9u * (sizeof(double) + sizeof(std::int32_t) + sizeof(uint8_t) +
                 sizeof(double) + sizeof(TxSecondaryJob) + sizeof(TxBlockJob));
  total += 2u * n * sizeof(std::int32_t);
  total += sizeof(double) + sizeof(std::int32_t);  // ist_base_sse/type
  total += 25u * 13u * 7u * 4u;                       // ist_map
  total += 25u * 2u * 13u * 9u * sizeof(TxSecondaryJob);
  total += 25u * 2u * 9u * sizeof(TxBlockJob);
  total += 25u * 2u * 13u * 9u * sizeof(double);
  total += 25u * 2u * 13u * sizeof(std::uint8_t);
#endif
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
  total += 7u * 2u * kTxTypeAdjElems * sizeof(std::int32_t) * 2u;
  total += 7u * 2u * kTxTypeCodedElems * sizeof(std::int16_t);
  total += 2u * kTxTypeAdjElems * sizeof(double);
  total += 7u * sizeof(double) + 7u * 2u * sizeof(std::int32_t) +
           7u * 2u * sizeof(TxBlockJob) + n * sizeof(std::int32_t);
#endif
  total += srch::luma_search_scratch_bytes(
      kSearchMaxBlocks, kSearchMaxBlockArea, kSearchCandidatesPerChunk);
  total += mi_cells * sizeof(deblock::DeblockMi) * 3u;
  total += mi_cells + chroma_mi * 2u;  // decoded[3]
  total += mi_cells * sizeof(int);     // intra_joint
  total += chroma_mi;                  // uv_smooth
  total += mi_cells * 2u;              // luma mode/angle MI grids
  total += n * 8u;                     // palette colour tables
  total += kSearchMaxBlockArea;        // palette colour-map scratch
  // In-loop filter (§7.18) destination planes and metadata grids.
  const std::size_t plane_y =
      static_cast<std::size_t>(plan.y_stride) * plan.padded_height;
  const std::size_t plane_c =
      static_cast<std::size_t>(plan.u_stride) * (plan.padded_height / 2u);
  total += plane_y + plane_c * 2u;     // filter planes
  // skip/lossless/lossless_chroma all live on the luma MI grid.
  total += mi_cells * 3u;
  const std::size_t n64_cols = (plan.mi_cols + 15u) / 16u;
  const std::size_t n64_rows = (plan.mi_rows + 15u) / 16u;
  total += n64_cols * n64_rows;        // cdef_idx (int8)
  // §7.19 CCSO: three unit-enable grids (worst case luma grid) and three
  // decoded offset tables (576 int16 per plane).
  total += 3u * n64_cols * n64_rows;
  total += 3u * 576u * sizeof(std::int16_t);
  // §7.20 loop restoration: final LrFrame planes, per-plane unit type grids
  // and coefficient arrays, plus the luma-only 16-bit curr/cdef/lr planes used
  // by PC-Wiener and GDF.
  const std::size_t unit_slots = lr_unit_slots(plan);
  total += plane_y + plane_c * 2u;
  total += 3u * unit_slots;
  total += 3u * unit_slots * 18u * sizeof(std::int16_t);
  total += 3u * plane_y * sizeof(std::uint16_t);
  total += mi_cells;  // PC-Wiener LrTxSkip grid
  total += q2_elems(plan) * sizeof(std::int32_t);  // QM q2 matrices
  total += sb_slot_count(plan) * 2u;               // segment id + qindex
  total += sb_slot_count(plan) * sizeof(std::uint64_t);  // variance scratch
  return total;
}
#endif

bool buffers_fit(const FramePlan& plan, const DeviceBuffers& buffers,
                 bool full) {
  Geometry g;
  if (!make_geometry(plan.width, plan.height, g)) {
    return false;
  }
  const std::size_t luma_bytes = luma_plane_bytes(g);
  const std::size_t chroma_bytes = chroma_plane_bytes(g);
  const bool base =
      buffers.src.y_bytes >= luma_bytes &&
      buffers.src.u_bytes >= chroma_bytes &&
      buffers.src.v_bytes >= chroma_bytes &&
      buffers.recon.y_bytes >= luma_bytes &&
      buffers.recon.u_bytes >= chroma_bytes &&
      buffers.recon.v_bytes >= chroma_bytes &&
      buffers.pred.y_bytes >= luma_bytes &&
      buffers.pred.u_bytes >= chroma_bytes &&
      buffers.pred.v_bytes >= chroma_bytes &&
      buffers.coeff.y_bytes >= coeff_bytes(plan.coeff_count_y) &&
      buffers.coeff.u_bytes >= coeff_bytes(plan.coeff_count_u) &&
      buffers.coeff.v_bytes >= coeff_bytes(plan.coeff_count_v) &&
      buffers.blocks_bytes >=
          static_cast<std::size_t>(plan.block_count) * sizeof(BlockPlan) &&
      buffers.decisions_bytes >= static_cast<std::size_t>(plan.block_count) *
                                     sizeof(BlockDecision);
  if (!base || !full) {
    return base;
  }
#if AV2ENC_STILL_GPU_HAVE_FULL
  const std::size_t n = plan.block_count;
  const std::size_t mi_cells = static_cast<std::size_t>(plan.mi_cols) *
                               static_cast<std::size_t>(plan.mi_rows);
  const std::size_t chroma_mi =
      static_cast<std::size_t>(plan.mi_cols / 2u) *
      static_cast<std::size_t>(plan.mi_rows / 2u);
  for (int p = 0; p < 3; ++p) {
    const std::size_t packed = packed_plane_pixels(plan, p);
    if (buffers.pred_packed_bytes[p] < packed ||
        buffers.residual_packed_bytes[p] < packed * sizeof(std::int16_t)) {
      return false;
    }
  }
  const std::size_t n64_cols = (plan.mi_cols + 15u) / 16u;
  const std::size_t n64_rows = (plan.mi_rows + 15u) / 16u;
  const std::size_t unit_slots = lr_unit_slots(plan);
  return buffers.filter.y_bytes >= luma_bytes &&
         buffers.filter.u_bytes >= chroma_bytes &&
         buffers.filter.v_bytes >= chroma_bytes &&
         buffers.lr.y_bytes >= luma_bytes &&
         buffers.lr.u_bytes >= chroma_bytes &&
         buffers.lr.v_bytes >= chroma_bytes &&
         buffers.lr_unit_type_bytes[0] >= unit_slots &&
         buffers.lr_unit_type_bytes[1] >= unit_slots &&
         buffers.lr_unit_type_bytes[2] >= unit_slots &&
         buffers.lr_unit_coeffs_bytes[0] >= unit_slots * 18u * sizeof(std::int16_t) &&
         buffers.lr_unit_coeffs_bytes[1] >= unit_slots * 18u * sizeof(std::int16_t) &&
         buffers.lr_unit_coeffs_bytes[2] >= unit_slots * 18u * sizeof(std::int16_t) &&
         buffers.luma16_curr.y_bytes >= luma_bytes * sizeof(std::uint16_t) &&
         buffers.luma16_cdef.y_bytes >= luma_bytes * sizeof(std::uint16_t) &&
         buffers.luma16_lr.y_bytes >= luma_bytes * sizeof(std::uint16_t) &&
         buffers.lr_tx_skip_bytes >= mi_cells &&
         buffers.sb_segment_bytes >= sb_slot_count(plan) &&
         buffers.sb_qindex_bytes >= sb_slot_count(plan) &&
         buffers.sb_variance_bytes >= sb_slot_count(plan) * sizeof(std::uint64_t) &&
         buffers.q2_bytes >= q2_elems(plan) * sizeof(std::int32_t) &&
         buffers.cdef_mi_bytes >= mi_cells &&
         buffers.cdef_chroma_mi_bytes >= mi_cells &&
         buffers.cdef_idx_bytes >= n64_cols * n64_rows &&
         buffers.ccso_blk_bytes[0] >= n64_cols * n64_rows &&
         buffers.ccso_blk_bytes[1] >= n64_cols * n64_rows &&
         buffers.ccso_blk_bytes[2] >= n64_cols * n64_rows &&
         buffers.ccso_offset_bytes[0] >= 576u * sizeof(std::int16_t) &&
         buffers.ccso_offset_bytes[1] >= 576u * sizeof(std::int16_t) &&
         buffers.ccso_offset_bytes[2] >= 576u * sizeof(std::int16_t) &&
         buffers.luma_blocks_bytes >= n * sizeof(srch::LumaSearchBlock) &&
         buffers.luma_results_bytes >= n * sizeof(srch::LumaSearchResult) &&
         buffers.luma_jobs_bytes >= n * sizeof(im::IntraJob) &&
         buffers.chroma_jobs_bytes >= n * sizeof(cm1::StillChromaJob) &&
         buffers.chroma_results_bytes >= n * sizeof(cm1::StillChromaResult) &&
         buffers.eob_bytes >= 3u * n * sizeof(std::int32_t) &&
         buffers.tx_jobs_bytes >= 3u * n * sizeof(StillTxJob) &&
         buffers.recon_jobs_bytes >= 3u * n * sizeof(cm1::ReconWriteJob) &&
         buffers.recon_infos_bytes >= n * sizeof(cm1::ReconBlockInfo) &&
         buffers.search_scratch_bytes >=
             srch::luma_search_scratch_bytes(kSearchMaxBlocks,
                                             kSearchMaxBlockArea,
                                             kSearchCandidatesPerChunk) &&
         buffers.tx_errors != nullptr &&
         buffers.deblock_mi_bytes[0] >= mi_cells * sizeof(deblock::DeblockMi) &&
         buffers.deblock_mi_bytes[1] >= mi_cells * sizeof(deblock::DeblockMi) &&
         buffers.deblock_mi_bytes[2] >= mi_cells * sizeof(deblock::DeblockMi) &&
         buffers.decoded_bytes[0] >= mi_cells &&
         buffers.decoded_bytes[1] >= chroma_mi &&
         buffers.decoded_bytes[2] >= chroma_mi &&
         buffers.intra_joint_bytes >= mi_cells * sizeof(int) &&
         buffers.uv_smooth_bytes >= chroma_mi &&
         buffers.mi_luma_bytes >= mi_cells &&
         buffers.palette_colors_bytes >= n * 8u &&
         buffers.palette_map_bytes >= kSearchMaxBlockArea
#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
         && buffers.tx_cand_adj != nullptr && buffers.tx_cand_coeff != nullptr &&
         buffers.tx_cand_levels != nullptr && buffers.tx_cand_recon != nullptr &&
         buffers.tx_cand_jobs != nullptr && buffers.tx_cand_tx_jobs != nullptr &&
         buffers.tx_choose_block != nullptr && buffers.tx_metrics != nullptr &&
         buffers.tx_decision != nullptr && buffers.tx_param_table != nullptr &&
         buffers.tx_probe_job != nullptr && buffers.tx_residual_coded != nullptr &&
         buffers.tx_residual_adj != nullptr &&
         buffers.tx_types_bytes >= 3u * n * sizeof(std::int32_t)
#endif
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
         && buffers.ist_levels != nullptr && buffers.ist_dequant != nullptr &&
         buffers.ist_residual != nullptr && buffers.ist_sse != nullptr &&
         buffers.ist_eob != nullptr && buffers.ist_valid != nullptr &&
         buffers.ist_bits != nullptr && buffers.ist_jobs != nullptr &&
         buffers.ist_block_jobs != nullptr &&
         buffers.ist_secs_bytes >= n * sizeof(std::int32_t) &&
         buffers.ist_sets_bytes >= n * sizeof(std::int32_t) &&
         buffers.ist_map != nullptr && buffers.ist_job_table != nullptr &&
         buffers.ist_block_table != nullptr &&
         buffers.ist_bit_table != nullptr &&
         buffers.ist_set_count != nullptr &&
         buffers.ist_base_sse != nullptr &&
         buffers.ist_primary_type != nullptr
#endif
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
         && buffers.cctx_levels != nullptr && buffers.cctx_dequant != nullptr &&
         buffers.cctx_residual != nullptr && buffers.cctx_u_coeff != nullptr &&
         buffers.cctx_v_coeff != nullptr && buffers.cctx_sse != nullptr &&
         buffers.cctx_eob != nullptr && buffers.cctx_jobs != nullptr &&
         buffers.cctx_types_bytes >= n * sizeof(std::int32_t)
#endif
         ;
#else
  (void)full;
  return base;
#endif
}

bool allocate_device_buffers(const FramePlan& plan, bool full,
                             DeviceBuffers& out, std::string& error) {
  Geometry g;
  if (!make_geometry(plan.width, plan.height, g)) {
    error = "device buffers: bad plan dimensions";
    return false;
  }
  Arena& arena = runtime().arena;
  const std::uint64_t mark = arena.mark();
  out = DeviceBuffers{};

  const std::size_t luma_bytes = luma_plane_bytes(g);
  const std::size_t chroma_bytes = chroma_plane_bytes(g);
  const auto bump = [&](std::size_t bytes, std::size_t align) -> void* {
    if (bytes == 0) {
      return nullptr;
    }
    void* p = nullptr;
    if (!is_ok(arena.allocate_checked(bytes, &p, align))) {
      return nullptr;
    }
    return p;
  };
  const auto bump_u8 = [&](std::size_t bytes, std::uint8_t** dst) -> bool {
    void* p = bump(bytes, Arena::kAlignment);
    *dst = static_cast<std::uint8_t*>(p);
    return bytes == 0 || p != nullptr;
  };
  const auto bump_i32 = [&](std::size_t bytes, std::int32_t** dst) -> bool {
    void* p = bump(bytes, Arena::kAlignment);
    *dst = static_cast<std::int32_t*>(p);
    return bytes == 0 || p != nullptr;
  };

  bool ok = true;
  ok = ok && bump_u8(luma_bytes, &out.src.y);
  ok = ok && bump_u8(chroma_bytes, &out.src.u);
  ok = ok && bump_u8(chroma_bytes, &out.src.v);
  ok = ok && bump_u8(luma_bytes, &out.recon.y);
  ok = ok && bump_u8(chroma_bytes, &out.recon.u);
  ok = ok && bump_u8(chroma_bytes, &out.recon.v);
  ok = ok && bump_u8(luma_bytes, &out.pred.y);
  ok = ok && bump_u8(chroma_bytes, &out.pred.u);
  ok = ok && bump_u8(chroma_bytes, &out.pred.v);
  ok = ok && bump_i32(coeff_bytes(plan.coeff_count_y), &out.coeff.y);
  ok = ok && bump_i32(coeff_bytes(plan.coeff_count_u), &out.coeff.u);
  ok = ok && bump_i32(coeff_bytes(plan.coeff_count_v), &out.coeff.v);
  out.src.y_bytes = luma_bytes;
  out.src.u_bytes = chroma_bytes;
  out.src.v_bytes = chroma_bytes;
  out.recon.y_bytes = luma_bytes;
  out.recon.u_bytes = chroma_bytes;
  out.recon.v_bytes = chroma_bytes;
  out.pred.y_bytes = luma_bytes;
  out.pred.u_bytes = chroma_bytes;
  out.pred.v_bytes = chroma_bytes;
  out.coeff.y_bytes = coeff_bytes(plan.coeff_count_y);
  out.coeff.u_bytes = coeff_bytes(plan.coeff_count_u);
  out.coeff.v_bytes = coeff_bytes(plan.coeff_count_v);
  out.blocks_bytes =
      static_cast<std::size_t>(plan.block_count) * sizeof(BlockPlan);
  out.decisions_bytes =
      static_cast<std::size_t>(plan.block_count) * sizeof(BlockDecision);
  if (ok) {
    void* p = bump(out.blocks_bytes, Arena::kAlignment);
    out.blocks = static_cast<BlockPlan*>(p);
    ok = out.blocks != nullptr;
  }
  if (ok) {
    void* p = bump(out.decisions_bytes, Arena::kAlignment);
    out.decisions = static_cast<BlockDecision*>(p);
    ok = out.decisions != nullptr;
  }

#if AV2ENC_STILL_GPU_HAVE_FULL
  if (ok && full) {
    const std::size_t n = plan.block_count;
    const std::size_t mi_cells = static_cast<std::size_t>(plan.mi_cols) *
                                 static_cast<std::size_t>(plan.mi_rows);
    const std::size_t chroma_mi =
        static_cast<std::size_t>(plan.mi_cols / 2u) *
        static_cast<std::size_t>(plan.mi_rows / 2u);
    for (int p = 0; p < 3; ++p) {
      const std::size_t packed = packed_plane_pixels(plan, p);
      out.pred_packed_bytes[p] = packed;
      out.residual_packed_bytes[p] = packed * sizeof(std::int16_t);
      out.pred_packed[p] =
          static_cast<std::uint8_t*>(bump(out.pred_packed_bytes[p], 256));
      out.residual_packed[p] =
          static_cast<std::int16_t*>(bump(out.residual_packed_bytes[p], 256));
    }
    out.luma_blocks_bytes = n * sizeof(srch::LumaSearchBlock);
    out.luma_results_bytes = n * sizeof(srch::LumaSearchResult);
    out.candidate_sse_bytes = n * kSearchCandidatesPerChunk * sizeof(std::int64_t);
    out.intra_choices_bytes = n * sizeof(iext::Choice);
    out.mrl_jobs_bytes = n * sizeof(mrl::MrlJob);
    out.dip_jobs_bytes = n * sizeof(dip::DipJob);
    out.ibp_jobs_bytes = n * sizeof(ibp::IbpJob);
    out.extra_pred_bytes = kSearchMaxBlockArea;
    out.luma_jobs_bytes = n * sizeof(im::IntraJob);
    out.chroma_jobs_bytes = n * sizeof(cm1::StillChromaJob);
    out.chroma_results_bytes = n * sizeof(cm1::StillChromaResult);
    out.eob_bytes = 3u * n * sizeof(std::int32_t);
    out.tx_jobs_bytes = 3u * n * sizeof(StillTxJob);
    out.recon_jobs_bytes = 3u * n * sizeof(cm1::ReconWriteJob);
    out.recon_infos_bytes = n * sizeof(cm1::ReconBlockInfo);
    out.search_scratch_bytes = srch::luma_search_scratch_bytes(
        kSearchMaxBlocks, kSearchMaxBlockArea, kSearchCandidatesPerChunk);
    out.decoded_bytes[0] = mi_cells;
    out.decoded_bytes[1] = chroma_mi;
    out.decoded_bytes[2] = chroma_mi;
    out.deblock_mi_bytes[0] = mi_cells * sizeof(deblock::DeblockMi);
    out.deblock_mi_bytes[1] = out.deblock_mi_bytes[0];
    out.deblock_mi_bytes[2] = out.deblock_mi_bytes[0];
    out.intra_joint_bytes = mi_cells * sizeof(int);
    out.uv_smooth_bytes = chroma_mi;
    out.mi_luma_bytes = mi_cells;
    out.palette_colors_bytes = n * 8u;
    out.palette_map_bytes = kSearchMaxBlockArea;

    out.luma_blocks = static_cast<srch::LumaSearchBlock*>(
        bump(out.luma_blocks_bytes, 256));
    out.luma_results = static_cast<srch::LumaSearchResult*>(
        bump(out.luma_results_bytes, 256));
    out.candidate_sse = static_cast<std::int64_t*>(
        bump(out.candidate_sse_bytes, 256));
    out.intra_choices =
        static_cast<iext::Choice*>(bump(out.intra_choices_bytes, 256));
    out.mrl_jobs = static_cast<mrl::MrlJob*>(bump(out.mrl_jobs_bytes, 256));
    out.dip_jobs = static_cast<dip::DipJob*>(bump(out.dip_jobs_bytes, 256));
    out.ibp_jobs = static_cast<ibp::IbpJob*>(bump(out.ibp_jobs_bytes, 256));
    out.mrl_pred = static_cast<std::uint8_t*>(bump(out.extra_pred_bytes, 256));
    out.dip_pred = static_cast<std::uint8_t*>(bump(out.extra_pred_bytes, 256));
    out.luma_jobs =
        static_cast<im::IntraJob*>(bump(out.luma_jobs_bytes, 256));
    out.chroma_jobs = static_cast<cm1::StillChromaJob*>(
        bump(out.chroma_jobs_bytes, 256));
    out.chroma_results = static_cast<cm1::StillChromaResult*>(
        bump(out.chroma_results_bytes, 256));
    out.eob = static_cast<std::int32_t*>(bump(out.eob_bytes, 256));
    out.tx_jobs = static_cast<StillTxJob*>(bump(out.tx_jobs_bytes, 256));
    out.recon_jobs = static_cast<cm1::ReconWriteJob*>(
        bump(out.recon_jobs_bytes, 256));
    out.recon_infos = static_cast<cm1::ReconBlockInfo*>(
        bump(out.recon_infos_bytes, 256));
    out.search_scratch = bump(out.search_scratch_bytes, 256);
    out.tx_errors = static_cast<int*>(bump(sizeof(int), 256));
#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
    out.tx_cand_adj_bytes = static_cast<std::size_t>(kTxTypeSearchSlots) *
                            kTxTypeAdjElems * sizeof(std::int16_t);
    out.tx_cand_coeff_bytes = static_cast<std::size_t>(kTxTypeSearchSlots) *
                              kTxTypeAdjElems * sizeof(double);
    out.tx_cand_levels_bytes = static_cast<std::size_t>(kTxTypeSearchSlots) *
                               kTxTypeAdjElems * sizeof(std::int32_t);
    out.tx_cand_recon_bytes = static_cast<std::size_t>(kTxTypeSearchSlots) *
                              kTxTypeCodedElems * sizeof(std::int16_t);
    out.tx_cand_jobs_bytes =
        static_cast<std::size_t>(kTxTypeSearchSlots) * sizeof(TxTypeCandJob);
    out.tx_cand_tx_jobs_bytes =
        static_cast<std::size_t>(kTxTypeSearchSlots) * sizeof(TxBlockJob);
    out.tx_choose_block_bytes = sizeof(TxTypeBlockJob);
    out.tx_metrics_bytes =
        static_cast<std::size_t>(kTxTypeSearchSlots) *
        sizeof(TxTypeCandidateMetrics);
    out.tx_decision_bytes = sizeof(TxTypeDecision);
    out.tx_param_table_bytes = 16u * sizeof(TxParams);
    out.tx_probe_job_bytes = sizeof(StillTxJob);
    out.tx_residual_coded_bytes = kTxTypeCodedElems * sizeof(std::int16_t);
    out.tx_residual_adj_bytes = kTxTypeAdjElems * sizeof(std::int16_t);
    out.tx_types_bytes = 3u * n * sizeof(std::int32_t);
    out.tx_cand_adj = static_cast<std::int16_t*>(
        bump(out.tx_cand_adj_bytes, 256));
    out.tx_cand_coeff = static_cast<double*>(
        bump(out.tx_cand_coeff_bytes, 256));
    out.tx_cand_levels = static_cast<std::int32_t*>(
        bump(out.tx_cand_levels_bytes, 256));
    out.tx_cand_recon = static_cast<std::int16_t*>(
        bump(out.tx_cand_recon_bytes, 256));
    out.tx_cand_jobs = static_cast<TxTypeCandJob*>(
        bump(out.tx_cand_jobs_bytes, 256));
    out.tx_cand_tx_jobs = static_cast<TxBlockJob*>(
        bump(out.tx_cand_tx_jobs_bytes, 256));
    out.tx_choose_block = static_cast<TxTypeBlockJob*>(
        bump(out.tx_choose_block_bytes, 256));
    out.tx_metrics = static_cast<TxTypeCandidateMetrics*>(
        bump(out.tx_metrics_bytes, 256));
    out.tx_decision = static_cast<TxTypeDecision*>(
        bump(out.tx_decision_bytes, 256));
    out.tx_param_table = static_cast<TxParams*>(
        bump(out.tx_param_table_bytes, 256));
    out.tx_probe_job = static_cast<StillTxJob*>(
        bump(out.tx_probe_job_bytes, 256));
    out.tx_residual_coded = static_cast<std::int16_t*>(
        bump(out.tx_residual_coded_bytes, 256));
    out.tx_residual_adj = static_cast<std::int16_t*>(
        bump(out.tx_residual_adj_bytes, 256));
    out.tx_types = static_cast<std::int32_t*>(bump(out.tx_types_bytes, 256));
    out.tx_type_cands_bytes =
        25u * 16u * sizeof(TxTypeCandJob);
    out.tx_type_jobs_bytes = 25u * 16u * sizeof(TxBlockJob);
    out.tx_type_blocks_bytes = 25u * sizeof(TxTypeBlockJob);
    out.tx_type_map_bytes = 25u * 13u * 7u * 4u;
    out.tx_type_legal_bytes = 25u * 13u * 16u;
    out.tx_type_bits_bytes = 25u * 13u * 16u * sizeof(double);
    out.tx_type_cands = static_cast<TxTypeCandJob*>(
        bump(out.tx_type_cands_bytes, 256));
    out.tx_type_jobs = static_cast<TxBlockJob*>(
        bump(out.tx_type_jobs_bytes, 256));
    out.tx_type_blocks = static_cast<TxTypeBlockJob*>(
        bump(out.tx_type_blocks_bytes, 256));
    out.tx_type_map = static_cast<std::int8_t*>(
        bump(out.tx_type_map_bytes, 256));
    out.tx_type_legal = static_cast<std::uint8_t*>(
        bump(out.tx_type_legal_bytes, 256));
    out.tx_type_bits = static_cast<double*>(
        bump(out.tx_type_bits_bytes, 256));
#endif
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
    out.ist_levels_bytes =
        9u * kTxTypeAdjElems * sizeof(std::int32_t);
    out.ist_dequant_bytes = out.ist_levels_bytes;
    out.ist_residual_bytes = 9u * kTxTypeCodedElems * sizeof(std::int16_t);
    out.ist_sse_bytes = 9u * sizeof(double);
    out.ist_eob_bytes = 9u * sizeof(std::int32_t);
    out.ist_valid_bytes = 9u;
    out.ist_bits_bytes = 9u * sizeof(double);
    out.ist_jobs_bytes = 9u * sizeof(TxSecondaryJob);
    out.ist_block_jobs_bytes = 9u * sizeof(TxBlockJob);
    out.ist_secs_bytes = n * sizeof(std::int32_t);
    out.ist_sets_bytes = n * sizeof(std::int32_t);
    out.ist_base_sse_bytes = sizeof(double);
    out.ist_primary_type_bytes = sizeof(std::int32_t);
    out.ist_map_bytes = 25u * 13u * 7u * 4u;
    out.ist_job_table_bytes =
        25u * 2u * 13u * 9u * sizeof(TxSecondaryJob);
    out.ist_block_table_bytes = 25u * 2u * 9u * sizeof(TxBlockJob);
    out.ist_bit_table_bytes = 25u * 2u * 13u * 9u * sizeof(double);
    out.ist_set_count_bytes = 25u * 2u * 13u * sizeof(std::uint8_t);
    out.ist_map = static_cast<std::int8_t*>(bump(out.ist_map_bytes, 256));
    out.ist_job_table = static_cast<TxSecondaryJob*>(
        bump(out.ist_job_table_bytes, 256));
    out.ist_block_table = static_cast<TxBlockJob*>(
        bump(out.ist_block_table_bytes, 256));
    out.ist_bit_table = static_cast<double*>(
        bump(out.ist_bit_table_bytes, 256));
    out.ist_set_count = static_cast<std::uint8_t*>(
        bump(out.ist_set_count_bytes, 256));
    out.ist_levels = static_cast<std::int32_t*>(bump(out.ist_levels_bytes, 256));
    out.ist_dequant = static_cast<std::int32_t*>(bump(out.ist_dequant_bytes, 256));
    out.ist_residual = static_cast<std::int16_t*>(bump(out.ist_residual_bytes, 256));
    out.ist_sse = static_cast<double*>(bump(out.ist_sse_bytes, 256));
    out.ist_eob = static_cast<std::int32_t*>(bump(out.ist_eob_bytes, 256));
    out.ist_valid = static_cast<std::uint8_t*>(bump(out.ist_valid_bytes, 256));
    out.ist_bits = static_cast<double*>(bump(out.ist_bits_bytes, 256));
    out.ist_jobs = static_cast<TxSecondaryJob*>(bump(out.ist_jobs_bytes, 256));
    out.ist_block_jobs = static_cast<TxBlockJob*>(bump(out.ist_block_jobs_bytes, 256));
    out.ist_secs = static_cast<std::int32_t*>(bump(out.ist_secs_bytes, 256));
    out.ist_sets = static_cast<std::int32_t*>(bump(out.ist_sets_bytes, 256));
    out.ist_base_sse = static_cast<double*>(bump(out.ist_base_sse_bytes, 256));
    out.ist_primary_type = static_cast<std::int32_t*>(
        bump(out.ist_primary_type_bytes, 256));
#endif
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
    out.cctx_levels_bytes =
        7u * 2u * kTxTypeAdjElems * sizeof(std::int32_t);
    out.cctx_dequant_bytes = out.cctx_levels_bytes;
    out.cctx_residual_bytes =
        7u * 2u * kTxTypeCodedElems * sizeof(std::int16_t);
    out.cctx_u_coeff_bytes = kTxTypeAdjElems * sizeof(double);
    out.cctx_v_coeff_bytes = out.cctx_u_coeff_bytes;
    out.cctx_sse_bytes = 7u * sizeof(double);
    out.cctx_eob_bytes = 7u * 2u * sizeof(std::int32_t);
    out.cctx_jobs_bytes = 7u * 2u * sizeof(TxBlockJob);
    out.cctx_types_bytes = n * sizeof(std::int32_t);
    out.cctx_levels = static_cast<std::int32_t*>(
        bump(out.cctx_levels_bytes, 256));
    out.cctx_dequant = static_cast<std::int32_t*>(
        bump(out.cctx_dequant_bytes, 256));
    out.cctx_residual = static_cast<std::int16_t*>(
        bump(out.cctx_residual_bytes, 256));
    out.cctx_u_coeff = static_cast<double*>(bump(out.cctx_u_coeff_bytes, 256));
    out.cctx_v_coeff = static_cast<double*>(bump(out.cctx_v_coeff_bytes, 256));
    out.cctx_sse = static_cast<double*>(bump(out.cctx_sse_bytes, 256));
    out.cctx_eob = static_cast<std::int32_t*>(bump(out.cctx_eob_bytes, 256));
    out.cctx_jobs = static_cast<TxBlockJob*>(bump(out.cctx_jobs_bytes, 256));
    out.cctx_types = static_cast<std::int32_t*>(bump(out.cctx_types_bytes, 256));
#endif
    for (int p = 0; p < 3; ++p) {
      out.deblock_mi[p] = static_cast<deblock::DeblockMi*>(
          bump(out.deblock_mi_bytes[p], 256));
      out.decoded[p] = static_cast<std::uint8_t*>(
          bump(out.decoded_bytes[p], 256));
    }
    out.intra_joint = static_cast<int*>(bump(out.intra_joint_bytes, 256));
    out.uv_smooth = static_cast<std::uint8_t*>(bump(out.uv_smooth_bytes, 256));
    out.mi_luma_mode =
        static_cast<std::int8_t*>(bump(out.mi_luma_bytes, 256));
    out.mi_luma_angle =
        static_cast<std::int8_t*>(bump(out.mi_luma_bytes, 256));
    out.palette_colors =
        static_cast<std::uint8_t*>(bump(out.palette_colors_bytes, 256));
    out.palette_map =
        static_cast<std::uint8_t*>(bump(out.palette_map_bytes, 256));
    // In-loop filter destination planes + metadata grids (§7.18).
    out.filter.y_bytes = luma_bytes;
    out.filter.u_bytes = chroma_bytes;
    out.filter.v_bytes = chroma_bytes;
    ok = ok && bump_u8(out.filter.y_bytes, &out.filter.y);
    ok = ok && bump_u8(out.filter.u_bytes, &out.filter.u);
    ok = ok && bump_u8(out.filter.v_bytes, &out.filter.v);
    out.cdef_mi_bytes = mi_cells;
    // §7.18.1 indexes LosslessArray[ChromaSegmentIds] on the luma MI grid.
    out.cdef_chroma_mi_bytes = mi_cells;
    out.cdef_skip = static_cast<std::uint8_t*>(
        bump(out.cdef_mi_bytes, 256));
    out.cdef_lossless = static_cast<std::uint8_t*>(
        bump(out.cdef_mi_bytes, 256));
    out.cdef_lossless_chroma = static_cast<std::uint8_t*>(
        bump(out.cdef_chroma_mi_bytes, 256));
    const std::size_t n64_cols = (plan.mi_cols + 15u) / 16u;
    const std::size_t n64_rows = (plan.mi_rows + 15u) / 16u;
    out.cdef_idx_bytes = n64_cols * n64_rows;
    out.cdef_idx = static_cast<std::int8_t*>(bump(out.cdef_idx_bytes, 256));
    for (int plane = 0; plane < 3; ++plane) {
      out.ccso_blk_bytes[plane] = n64_cols * n64_rows;
      out.ccso_blks[plane] = static_cast<std::uint8_t*>(
          bump(out.ccso_blk_bytes[plane], 256));
      out.ccso_offset_bytes[plane] = 576u * sizeof(std::int16_t);
      out.ccso_offsets[plane] = static_cast<std::int16_t*>(
          bump(out.ccso_offset_bytes[plane], 256));
      ok = ok && out.ccso_blks[plane] != nullptr &&
           out.ccso_offsets[plane] != nullptr;
    }
    // §7.20 loop restoration output and its per-plane unit descriptors.
    out.lr.y_bytes = luma_bytes;
    out.lr.u_bytes = chroma_bytes;
    out.lr.v_bytes = chroma_bytes;
    ok = ok && bump_u8(out.lr.y_bytes, &out.lr.y);
    ok = ok && bump_u8(out.lr.u_bytes, &out.lr.u);
    ok = ok && bump_u8(out.lr.v_bytes, &out.lr.v);
    const std::size_t unit_slots = lr_unit_slots(plan);
    for (int plane = 0; plane < 3; ++plane) {
      out.lr_unit_type_bytes[plane] = unit_slots;
      out.lr_unit_coeffs_bytes[plane] =
          unit_slots * 18u * sizeof(std::int16_t);
      out.lr_unit_type[plane] = static_cast<std::uint8_t*>(
          bump(out.lr_unit_type_bytes[plane], 256));
      out.lr_unit_coeffs[plane] = static_cast<std::int16_t*>(
          bump(out.lr_unit_coeffs_bytes[plane], 256));
      ok = ok && out.lr_unit_type[plane] != nullptr &&
           out.lr_unit_coeffs[plane] != nullptr;
    }
    out.luma16_curr.y_bytes = luma_bytes * sizeof(std::uint16_t);
    out.luma16_cdef.y_bytes = out.luma16_curr.y_bytes;
    out.luma16_lr.y_bytes = out.luma16_curr.y_bytes;
    out.luma16_curr.y = static_cast<std::uint16_t*>(
        bump(out.luma16_curr.y_bytes, 256));
    out.luma16_cdef.y = static_cast<std::uint16_t*>(
        bump(out.luma16_cdef.y_bytes, 256));
    out.luma16_lr.y = static_cast<std::uint16_t*>(
        bump(out.luma16_lr.y_bytes, 256));
    ok = ok && out.luma16_curr.y != nullptr &&
         out.luma16_cdef.y != nullptr && out.luma16_lr.y != nullptr;
    out.lr_tx_skip_bytes = mi_cells;
    out.lr_tx_skip = static_cast<std::uint8_t*>(
        bump(out.lr_tx_skip_bytes, 256));
    ok = ok && out.lr_tx_skip != nullptr;
    out.q2_bytes = q2_elems(plan) * sizeof(std::int32_t);
    out.q2 = static_cast<std::int32_t*>(bump(out.q2_bytes, 256));
    out.sb_segment_bytes = sb_slot_count(plan);
    out.sb_qindex_bytes = sb_slot_count(plan);
    out.sb_variance_bytes = sb_slot_count(plan) * sizeof(std::uint64_t);
    out.sb_segment = static_cast<std::uint8_t*>(bump(out.sb_segment_bytes, 256));
    out.sb_qindex = static_cast<std::uint8_t*>(bump(out.sb_qindex_bytes, 256));
    out.sb_variance = static_cast<std::uint64_t*>(
        bump(out.sb_variance_bytes, 256));
    ok = ok && (out.sb_segment_bytes == 0 || out.sb_segment != nullptr) &&
         (out.sb_qindex_bytes == 0 || out.sb_qindex != nullptr) &&
         (out.sb_variance_bytes == 0 || out.sb_variance != nullptr);
    ok = ok && (out.q2_bytes == 0 || out.q2 != nullptr);
    ok = ok && out.lr.y != nullptr && out.lr.u != nullptr &&
         out.lr.v != nullptr;
    ok = ok && out.filter.y != nullptr && out.filter.u != nullptr &&
         out.filter.v != nullptr && out.cdef_skip != nullptr &&
         out.cdef_lossless != nullptr &&
         out.cdef_lossless_chroma != nullptr && out.cdef_idx != nullptr;
    ok = ok && out.pred_packed[0] != nullptr &&
         out.pred_packed[1] != nullptr && out.pred_packed[2] != nullptr &&
         out.residual_packed[0] != nullptr &&
         out.residual_packed[1] != nullptr &&
         out.residual_packed[2] != nullptr && out.luma_blocks != nullptr &&
         out.luma_results != nullptr && out.luma_jobs != nullptr &&
         out.candidate_sse != nullptr && out.intra_choices != nullptr &&
         out.mrl_jobs != nullptr && out.dip_jobs != nullptr &&
         out.ibp_jobs != nullptr &&
         out.mrl_pred != nullptr && out.dip_pred != nullptr &&
         out.chroma_jobs != nullptr && out.chroma_results != nullptr &&
         out.eob != nullptr &&
         out.tx_jobs != nullptr && out.recon_jobs != nullptr &&
         out.recon_infos != nullptr && out.search_scratch != nullptr &&
         out.tx_errors != nullptr && out.deblock_mi[0] != nullptr &&         out.deblock_mi[1] != nullptr && out.deblock_mi[2] != nullptr &&
         out.decoded[0] != nullptr && out.decoded[1] != nullptr &&
         out.decoded[2] != nullptr && out.intra_joint != nullptr &&
         out.uv_smooth != nullptr && out.mi_luma_mode != nullptr &&
         out.mi_luma_angle != nullptr && out.palette_colors != nullptr &&
         out.palette_map != nullptr;
#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
    ok = ok && out.tx_cand_adj != nullptr && out.tx_cand_coeff != nullptr &&
         out.tx_cand_levels != nullptr && out.tx_cand_recon != nullptr &&
         out.tx_cand_jobs != nullptr && out.tx_cand_tx_jobs != nullptr &&
         out.tx_choose_block != nullptr && out.tx_metrics != nullptr &&
         out.tx_decision != nullptr && out.tx_param_table != nullptr &&
         out.tx_probe_job != nullptr && out.tx_residual_coded != nullptr &&
         out.tx_residual_adj != nullptr && out.tx_types != nullptr &&
         out.tx_type_cands != nullptr && out.tx_type_jobs != nullptr &&
         out.tx_type_blocks != nullptr && out.tx_type_map != nullptr &&
         out.tx_type_legal != nullptr && out.tx_type_bits != nullptr;
#endif
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
    ok = ok && out.ist_levels != nullptr && out.ist_dequant != nullptr &&
         out.ist_residual != nullptr && out.ist_sse != nullptr &&
         out.ist_eob != nullptr && out.ist_valid != nullptr &&
         out.ist_bits != nullptr && out.ist_jobs != nullptr &&
         out.ist_block_jobs != nullptr && out.ist_secs != nullptr &&
         out.ist_sets != nullptr && out.ist_map != nullptr &&
         out.ist_job_table != nullptr && out.ist_block_table != nullptr &&
         out.ist_bit_table != nullptr && out.ist_set_count != nullptr &&
         out.ist_base_sse != nullptr && out.ist_primary_type != nullptr;
#endif
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
    ok = ok && out.cctx_levels != nullptr && out.cctx_dequant != nullptr &&
         out.cctx_residual != nullptr && out.cctx_u_coeff != nullptr &&
         out.cctx_v_coeff != nullptr && out.cctx_sse != nullptr &&
         out.cctx_eob != nullptr && out.cctx_jobs != nullptr &&
         out.cctx_types != nullptr;
#endif
  }
#else
  (void)full;
#endif

  if (!ok) {
    arena.rewind(mark);
    error = "device arena: out of memory for the M1 working set";
    return false;
  }

  out.total_bytes =
      static_cast<std::uint64_t>(luma_bytes) * 3u +
      static_cast<std::uint64_t>(chroma_bytes) * 6u +
      static_cast<std::uint64_t>(coeff_bytes(plan.coeff_count_y)) +
      static_cast<std::uint64_t>(coeff_bytes(plan.coeff_count_u)) +
      static_cast<std::uint64_t>(coeff_bytes(plan.coeff_count_v)) +
      static_cast<std::uint64_t>(out.blocks_bytes) +
      static_cast<std::uint64_t>(out.decisions_bytes);
#if AV2ENC_STILL_GPU_HAVE_FULL
  if (full) {
    out.total_bytes = static_cast<std::uint64_t>(out.total_bytes) +
                      static_cast<std::uint64_t>(full_device_bytes(plan));
  }
#endif
  return true;
}

enum class RuntimeState { kReady, kCpuFallback, kError };

// Persistent bytes the filter workspaces will request from the arena.  Kept
// separate from DeviceBuffers because the components own their own layout.
std::uint64_t filter_workspace_bytes(const FramePlan& plan,
                                     const GpuOptions& options) {
  std::uint64_t bytes = 0;
  // §5.18.7.11: the luma/chroma unit shift is capped by the frame SbSize.
  // The still planner forces shift 3 at 64x64 and shift 2 at 128x128, so the
  // component workspaces must follow plan.sb_size, not a fixed 64.
  const int lr_shift = plan.sb_size == 128 ? 2 : 3;
  const int sb_shift = plan.sb_size == 128 ? 5 : 4;
#if AV2ENC_STILL_GPU_HAVE_LR_CHROMA
  if (options.enable_chroma_lr) {
    lr_chroma::LrChromaConfig cfg;
    cfg.mi_cols = plan.mi_cols;
    cfg.mi_rows = plan.mi_rows;
    cfg.sub_x = 1;
    cfg.sub_y = 1;
    cfg.bit_depth = 8;
    cfg.unit_size = (512 >> 1) >> lr_shift;  // 32 at 64 SB, 64 at 128 SB
    cfg.cfl_ds_filter_index = 0;
    cfg.sb_shift = sb_shift;
    bytes += lr_chroma::required_device_bytes(cfg);
  }
#endif
#if AV2ENC_STILL_GPU_HAVE_PC_WIENER
  if (options.enable_pc_wiener) {
    pc_wiener::PcWienerConfig cfg;
    cfg.width = plan.mi_cols * 4;
    cfg.height = plan.mi_rows * 4;
    cfg.bit_depth = 8;
    cfg.unit_size = 512 >> lr_shift;  // 64 at 64 SB, 128 at 128 SB
    bytes += pc_wiener::pc_wiener_required_device_bytes(cfg);
  }
#endif
#if AV2ENC_STILL_GPU_HAVE_GDF
  if (options.enable_gdf) {
    // §5.18.7.9 GdfBlkSize = Max(Block_Width[SbSize], 128), i.e. 128 for both
    // the 64x64 and 128x128 frame superblocks.
    const int gdf_unit = plan.sb_size > 128 ? plan.sb_size : 128;
    const int cols = (plan.mi_cols * 4 + gdf_unit - 1) / gdf_unit;
    const int rows = (plan.mi_rows * 4 + gdf_unit - 1) / gdf_unit;
    gdf::GdfConfig cfg;
    cfg.max_units = gdf::gdf_grid_slots(cols, rows);
    bytes += gdf::required_device_bytes(cfg);
  }
#endif
  return bytes;
}

// Allocates the component workspaces once per arena/geometry.  Called at a
// frame boundary only, after the arena and DeviceBuffers exist.
bool prepare_filter_workspaces(const FramePlan& plan, const GpuOptions& options,
                               std::string& error) {
  FilterWorkspaces& ws = runtime().filters;
  if (ws.mi_cols != plan.mi_cols || ws.mi_rows != plan.mi_rows ||
      ws.sb_size != plan.sb_size) {
    ws.release();
    ws.mi_cols = plan.mi_cols;
    ws.mi_rows = plan.mi_rows;
    ws.sb_size = plan.sb_size;
  }
  Arena& arena = runtime().arena;
  (void)arena;
  const int lr_shift = plan.sb_size == 128 ? 2 : 3;
  const int sb_shift = plan.sb_size == 128 ? 5 : 4;
#if AV2ENC_STILL_GPU_HAVE_LR_CHROMA
  if (options.enable_chroma_lr && !ws.chroma_lr.initialized()) {
    lr_chroma::LrChromaConfig cfg;
    cfg.mi_cols = plan.mi_cols;
    cfg.mi_rows = plan.mi_rows;
    cfg.sub_x = 1;
    cfg.sub_y = 1;
    cfg.bit_depth = 8;
    cfg.unit_size = (512 >> 1) >> lr_shift;
    cfg.cfl_ds_filter_index = 0;
    cfg.sb_shift = sb_shift;
    std::string detail;
    if (lr_chroma::allocate_checked(arena, cfg, &ws.chroma_lr, nullptr,
                                    &detail) != Status::kOk) {
      error = "M1 GPU: chroma loop-restoration workspace: " + detail;
      return false;
    }
  }
#endif
#if AV2ENC_STILL_GPU_HAVE_PC_WIENER
  if (options.enable_pc_wiener && !ws.pc_wiener.initialized()) {
    pc_wiener::PcWienerConfig cfg;
    cfg.width = plan.mi_cols * 4;
    cfg.height = plan.mi_rows * 4;
    cfg.bit_depth = 8;
    cfg.unit_size = 512 >> lr_shift;
    std::string detail;
    if (ws.pc_wiener.create(arena, cfg, &detail) != Status::kOk) {
      error = "M1 GPU: PC-Wiener workspace: " + detail;
      return false;
    }
  }
#endif
#if AV2ENC_STILL_GPU_HAVE_GDF
  if (options.enable_gdf && !ws.gdf_ready) {
    const int gdf_unit = plan.sb_size > 128 ? plan.sb_size : 128;
    const int cols = (plan.mi_cols * 4 + gdf_unit - 1) / gdf_unit;
    const int rows = (plan.mi_rows * 4 + gdf_unit - 1) / gdf_unit;
    gdf::GdfConfig cfg;
    cfg.max_units = gdf::gdf_grid_slots(cols, rows);
    std::string detail;
    if (gdf::gdf_buffers_init(arena, cfg, &ws.gdf, nullptr, &detail) !=
        Status::kOk) {
      error = "M1 GPU: GDF workspace: " + detail;
      return false;
    }
    ws.gdf_ready = true;
  }
#endif
  return true;
}

RuntimeState ensure_runtime(const GpuOptions& options, const FramePlan* plan,
                            std::uint64_t need_device,
                            std::uint64_t need_upload,
                            std::uint64_t need_download, bool stage1,
                            std::string& error) {
  GpuRuntime& gpu = runtime();
  if (!gpu.init_attempted) {
    gpu.init_attempted = true;
    gpu.device = options.device;
    std::string init_error;
    if (!is_ok(gpu.context.init(options.device, &init_error))) {
      gpu.init_error = init_error.empty() ? "CUDA context init failed"
                                          : init_error;
      gpu.ready = false;
      if (options.allow_cpu_fallback) {
        return RuntimeState::kCpuFallback;
      }
      error = "M1 GPU: " + gpu.init_error;
      return RuntimeState::kError;
    }
  }
  if (!gpu.context.initialized()) {
    if (options.allow_cpu_fallback) {
      return RuntimeState::kCpuFallback;
    }
    error = "M1 GPU: " + gpu.init_error;
    return RuntimeState::kError;
  }
  if (gpu.device != options.device) {
    error = "M1 GPU: one device per process (already initialized device " +
            std::to_string(gpu.device) + ")";
    return RuntimeState::kError;
  }

  // Device-derived sizing rule: min(total, free) - 512 MiB, optionally bounded
  // by an explicit cap.  The M1 working set only needs a few MiB, so the
  // reservation is additionally clamped to a small multiple of that working
  // set; it never assumes a particular card.
  const std::uint64_t total = gpu.context.total_vram_bytes();
  const std::uint64_t free_bytes = gpu.context.free_vram_bytes();
  std::uint64_t budget = 0;
  if (total > 0 && free_bytes > 0) {
    const std::uint64_t available = std::min(total, free_bytes);
    budget = (available > kDeviceHeadroomBytes)
                 ? available - kDeviceHeadroomBytes
                 : 0;
  }
  const std::uint64_t cap_bytes =
      options.vram_cap_mb > 0
          ? static_cast<std::uint64_t>(options.vram_cap_mb) * kMiB
          : 0;
  if (cap_bytes > 0 && cap_bytes < need_device) {
    const std::uint64_t need_mib = (need_device + kMiB - 1u) / kMiB;
    error = "M1 GPU: --vram-cap " + std::to_string(options.vram_cap_mb) +
            " MiB is below the required working set (" +
            std::to_string(need_mib) + " MiB)";
    return RuntimeState::kError;
  }
  std::uint64_t desired = std::max(
      kArenaFloorBytes, need_device + need_device / 2u + kArenaSlackBytes);
  if (cap_bytes > 0) {
    desired = std::min(desired, cap_bytes);
  }
  if (budget > 0) {
    desired = std::min(desired, budget);
  }
  if (desired < need_device) {
    if (options.allow_cpu_fallback) {
      return RuntimeState::kCpuFallback;
    }
    error = "M1 GPU: the device budget cannot hold the working set (" +
            std::to_string(need_device) + " bytes)";
    return RuntimeState::kError;
  }

  // An explicit cap also forces a shrink if a previous frame reserved more.
  const bool over_cap = cap_bytes > 0 && gpu.arena.initialized() &&
                        gpu.arena.capacity() > cap_bytes;
  const bool need_grow = !gpu.arena.initialized() ||
                         desired > gpu.arena.capacity() || over_cap;
  const bool need_buffers =
      plan != nullptr && (need_device > 0) &&
      !buffers_fit(*plan, gpu.buffers, stage1);
  if (need_grow || need_buffers) {
    // Releasing and re-initialising is a frame-boundary operation, never a
    // block-loop one.  With an explicit cap the reservation must not exceed it;
    // otherwise stay grow-only across frames.
    const std::uint64_t capacity =
        (gpu.arena.initialized() && !over_cap)
            ? std::max(desired, gpu.arena.capacity())
            : desired;
    gpu.filters.release();
    gpu.buffers = DeviceBuffers{};
    gpu.upload.release();
    gpu.download.release();
    gpu.arena.release();
    std::string arena_error;
    if (!is_ok(gpu.arena.init(capacity, Arena::kAlignment, &arena_error))) {
      gpu.init_error =
          arena_error.empty() ? "arena init failed" : arena_error;
      gpu.ready = false;
      if (options.allow_cpu_fallback) {
        return RuntimeState::kCpuFallback;
      }
      error = "M1 GPU: " + gpu.init_error;
      return RuntimeState::kError;
    }
    gpu.arena_capacity = gpu.arena.capacity();
    if (plan != nullptr && need_device > 0) {
      if (!allocate_device_buffers(*plan, stage1, gpu.buffers, error)) {
        if (options.allow_cpu_fallback) {
          return RuntimeState::kCpuFallback;
        }
        return RuntimeState::kError;
      }
    }
  }
  if (plan != nullptr && need_device > 0) {
    if (!prepare_filter_workspaces(*plan, options, error)) {
      if (options.allow_cpu_fallback) {
        return RuntimeState::kCpuFallback;
      }
      return RuntimeState::kError;
    }
  }
  if (need_upload > gpu.upload.capacity()) {
    gpu.upload.release();
    std::string stage_error;
    if (!is_ok(gpu.upload.init(need_upload, &stage_error))) {
      gpu.init_error =
          stage_error.empty() ? "pinned upload init failed" : stage_error;
      if (options.allow_cpu_fallback) {
        return RuntimeState::kCpuFallback;
      }
      error = "M1 GPU: " + gpu.init_error;
      return RuntimeState::kError;
    }
  }
  if (need_download > gpu.download.capacity()) {
    gpu.download.release();
    std::string stage_error;
    if (!is_ok(gpu.download.init(need_download, &stage_error))) {
      gpu.init_error =
          stage_error.empty() ? "pinned download init failed" : stage_error;
      if (options.allow_cpu_fallback) {
        return RuntimeState::kCpuFallback;
      }
      error = "M1 GPU: " + gpu.init_error;
      return RuntimeState::kError;
    }
  }
  gpu.ready = true;
  return RuntimeState::kReady;
}

// ---------------------------------------------------------------------------
// Recon plane helpers
// ---------------------------------------------------------------------------

bool blit_visible_to_padded(std::span<const std::uint8_t> src,
                            std::uint32_t width, std::uint32_t height,
                            std::uint32_t stride,
                            std::span<std::uint8_t> dst) {
  if (stride < width || dst.size() < static_cast<std::size_t>(stride) * height ||
      src.size() < static_cast<std::size_t>(width) * height) {
    return false;
  }
  for (std::uint32_t y = 0; y < height; ++y) {
    std::memcpy(dst.data() + static_cast<std::size_t>(y) * stride,
                src.data() + static_cast<std::size_t>(y) * width, width);
  }
  return true;
}

bool crop_padded_to_visible(std::span<const std::uint8_t> src,
                            std::uint32_t width, std::uint32_t height,
                            std::uint32_t stride,
                            std::vector<std::uint8_t>& dst) {
  if (stride < width || src.size() < static_cast<std::size_t>(stride) * height) {
    return false;
  }
  dst.resize(static_cast<std::size_t>(width) * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    std::memcpy(dst.data() + static_cast<std::size_t>(y) * width,
                src.data() + static_cast<std::size_t>(y) * stride, width);
  }
  return true;
}

bool compare_plane(std::span<const std::uint8_t> gpu,
                   std::span<const std::uint8_t> cpu, std::uint32_t width,
                   std::uint32_t height, std::uint32_t stride,
                   const char* plane, std::string& error) {
  if (gpu.size() < static_cast<std::size_t>(stride) * height ||
      cpu.size() < static_cast<std::size_t>(width) * height) {
    error = std::string(plane) + " plane is too small";
    return false;
  }
  std::uint64_t mismatches = 0;
  std::string first;
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint8_t got =
          gpu[static_cast<std::size_t>(y) * stride + x];
      const std::uint8_t want =
          cpu[static_cast<std::size_t>(y) * width + x];
      if (got != want) {
        if (first.empty()) {
          first = std::string(plane) + " mismatch at (" +
                  std::to_string(x) + ", " + std::to_string(y) + "): got " +
                  std::to_string(static_cast<int>(got)) + ", want " +
                  std::to_string(static_cast<int>(want));
        }
        ++mismatches;
      }
    }
  }
  if (mismatches != 0) {
    error = first + " (" + std::to_string(mismatches) + " of " +
            std::to_string(static_cast<std::uint64_t>(width) * height) +
            " samples)";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Options / small utilities
// ---------------------------------------------------------------------------

// The mapping lives in still_gpu.h so host-side tests and this orchestrator share
// exactly one implementation; every M3 flag is carried, not just the ones the
// GPU stages currently consume.
::av2enc::still::StillOptions to_cpu_options(const GpuOptions& options) {
  return gpu_options_to_m1(options);
}

// Reads a (possibly weak) hook address without the compiler proving it null in
// builds where the same translation unit defines the hook.
template <typename Fn>
bool hook_present(Fn fn) {
  Fn volatile copy = fn;
  return copy != nullptr;
}

bool validate_options(const GpuOptions& options,
                      const ::av2enc::still::StillPicture& picture,
                      std::string& error) {
  if ((picture.width & 1u) != 0u || (picture.height & 1u) != 0u) {
    error = "M1 4:2:0 requires even dimensions";
    return false;
  }
  if (options.base_q_idx != 90) {
    error = "M1 GPU pins base_q_idx = 90; use --still for QP 1..255";
    return false;
  }
  if (options.min_block_size < 4 || options.min_block_size > 64) {
    error = "M1 min_block_size must be in [4, 64]";
    return false;
  }
  if (options.max_partition_depth < 0 || options.max_partition_depth > 32) {
    error = "M1 max_partition_depth must be in [0, 32]";
    return false;
  }
  if (options.cdef_num_strengths < 1 || options.cdef_num_strengths > 8) {
    error = "M1 GPU: --still-cdef-strengths must be in [1, 8]";
    return false;
  }
  if (options.cdef_damping < 3 || options.cdef_damping > 6) {
    error = "M1 GPU: CDEF damping must be in [3, 6]";
    return false;
  }
  if (options.ccso_bands_log2 < 0 || options.ccso_bands_log2 > 6) {
    error = "M1 GPU: --still-ccso-bands must be in [0, 6]";
    return false;
  }
  // Every M3 tool flag is either implemented by a GPU stage or rejected by
  // name here.  None is accepted-and-ignored.
  if (!gpu_m3_options_supported(options, &error)) {
    return false;
  }
  const std::size_t luma = static_cast<std::size_t>(picture.width) *
                           static_cast<std::size_t>(picture.height);
  const std::size_t chroma = visible_chroma_pixels(picture.width, picture.height);
  if (picture.y.size() < luma || picture.u.size() < chroma ||
      picture.v.size() < chroma) {
    error = "picture planes are too small";
    return false;
  }
  return true;
}

bool fill_packed_offsets(FramePlan& plan, std::vector<BlockPlan>& blocks) {
  std::uint64_t pred_y = 0;
  std::uint64_t pred_u = 0;
  std::uint64_t pred_v = 0;
  std::uint64_t resid_y = 0;
  std::uint64_t resid_u = 0;
  std::uint64_t resid_v = 0;
  for (BlockPlan& block : blocks) {
    const std::uint64_t luma_area =
        static_cast<std::uint64_t>(block.luma_w) * block.luma_h;
    const std::uint64_t chroma_area =
        static_cast<std::uint64_t>(block.chroma_w) * block.chroma_h;
    block.pred_offset_y = static_cast<std::uint32_t>(pred_y);
    block.residual_offset_y = static_cast<std::uint32_t>(resid_y);
    pred_y += luma_area;
    resid_y += luma_area;
    if (block.has_chroma != 0) {
      block.pred_offset_u = static_cast<std::uint32_t>(pred_u);
      block.pred_offset_v = static_cast<std::uint32_t>(pred_v);
      block.residual_offset_u = static_cast<std::uint32_t>(resid_u);
      block.residual_offset_v = static_cast<std::uint32_t>(resid_v);
      pred_u += chroma_area;
      pred_v += chroma_area;
      resid_u += chroma_area;
      resid_v += chroma_area;
    } else {
      block.pred_offset_u = static_cast<std::uint32_t>(pred_u);
      block.pred_offset_v = static_cast<std::uint32_t>(pred_v);
      block.residual_offset_u = static_cast<std::uint32_t>(resid_u);
      block.residual_offset_v = static_cast<std::uint32_t>(resid_v);
    }
  }
  plan.block_count = static_cast<std::uint32_t>(blocks.size());
  const std::uint64_t limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
  return pred_y <= limit && pred_u <= limit && pred_v <= limit &&
         resid_y <= limit && resid_u <= limit && resid_v <= limit;
}

#if AV2ENC_STILL_GPU_HAVE_FULL
enum class GpuRunStatus { kOk, kUnsupported, kError };

// ---------------------------------------------------------------------------
// Descriptor plumbing for the ordered block pipeline.
//
// The search and prediction components take plain POD descriptors.  Mode
// decisions are produced on the device, so these three one-thread kernels move
// the decision fields between device arrays without a host round trip.
// ---------------------------------------------------------------------------

// CPU `neighborhood_smooth()` for luma: above/left neighbour uses SMOOTH,
// SMOOTH_V or SMOOTH_H (IntraJointMode 1..3).
__global__ void prepare_search_block_kernel(srch::LumaSearchBlock* blocks,
                                            const int* intra_joint, int mi_cols,
                                            int mi_rows, int index) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  srch::LumaSearchBlock& block = blocks[index];
  const int r = block.y0 >> 2;
  const int c = block.x0 >> 2;
  int above = 0;
  int left = 0;
  if (r > 0 && c < mi_cols) {
    const int joint = intra_joint[(r - 1) * mi_cols + c];
    above = (joint >= 1 && joint <= 3) ? 1 : 0;
  }
  if (c > 0 && r < mi_rows) {
    const int joint = intra_joint[r * mi_cols + (c - 1)];
    left = (joint >= 1 && joint <= 3) ? 1 : 0;
  }
  block.filter_type_above = above;
  block.filter_type_left = left;
}

// Applies the chosen mode/angle and the neighbour smooth flags to the packed
// predictor job for block `index`.
__global__ void prepare_prediction_kernel(
    im::IntraJob* jobs, const srch::LumaSearchResult* results,
    const srch::LumaSearchBlock* blocks, int index) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  im::IntraBlock& block = jobs[index].block;
  block.mode = static_cast<im::IntraMode>(results[index].mode);
  block.angle_delta = results[index].angle_delta;
  block.filter_type_above = blocks[index].filter_type_above;
  block.filter_type_left = blocks[index].filter_type_left;
}

// Writes the IntraJointMode the reconstruction state needs.  Mirrors
// syntax::y_mode_code(): reordered index 0..4 for non-directional modes,
// 5 + dir_index for directional ones.
__global__ void prepare_commit_kernel(cm1::ReconBlockInfo* infos,
                                      const srch::LumaSearchResult* results,
                                      const cm1::StillChromaResult* chroma,
                                      int index) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  // Position of each AV2 YMode in kReorderedY.
  static constexpr int kReorderedIndex[13] = {0, 7, 11, 5, 9, 8, 10,
                                              12, 6, 1, 2, 3, 4};
  const int mode = results[index].mode;
  int reordered = 0;
  if (mode >= 0 && mode < 13) {
    reordered = kReorderedIndex[mode];
  }
  const int angle = results[index].angle_delta;
  int joint = reordered;
  if (reordered >= 5) {
    joint = 5 + (reordered - 5) * 7 + (angle + 3);
  }
  infos[index].joint = static_cast<std::uint8_t>(joint);
  // §7.13.2.15/.16: record whether the chosen UV mode is SMOOTH/SMOOTH_V/
  // SMOOTH_H (9..11).  A CfL block never is.
  const int uv_mode = chroma[index].uv_mode;
  infos[index].uv_smooth =
      (chroma[index].is_cfl == 0 && uv_mode >= 9 && uv_mode <= 11) ? 1 : 0;
}

// Writes a planned luma decision (palette / IBC force DC with the planned
// score).  Used when the GPU search is skipped for a screen-content leaf.
__global__ void set_luma_result_kernel(srch::LumaSearchResult* results,
                                       int index, int mode, int angle,
                                       std::int64_t sse) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  results[index].mode = mode;
  results[index].angle_delta = angle;
  results[index].reserved = 0;
  results[index].sse = sse;
}

// IBC replaces the chroma mode decision: read_intrabc_info() forces DC on both
// planes and clears CfL/palette state (§5.20.5.4).
__global__ void set_chroma_dc_kernel(cm1::StillChromaResult* results,
                                     int index) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  cm1::StillChromaResult& r = results[index];
  r = cm1::StillChromaResult{};
  r.winner = cm1::kM1ChromaDc;
  r.uv_mode = 0;
  r.angle_delta = 0;
  r.is_cfl = 0;
  r.cfl_index = 1;
}

// Records block `index`'s final luma mode/angle on the MI grid.  A later
// CHROMA_PART leaf (§5.20.5.6) codes uv_mode against the co-located luma-tree
// YMode, which is only available on the device until the frame sync.
__global__ void record_luma_mode_kernel(const BlockPlan* blocks,
                                        const srch::LumaSearchResult* results,
                                        std::int8_t* mi_mode,
                                        std::int8_t* mi_angle, int mi_cols,
                                        int mi_rows, int index) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const BlockPlan& block = blocks[index];
  const int w4 = block.luma_w / 4;
  const int h4 = block.luma_h / 4;
  if (w4 <= 0 || h4 <= 0) {
    return;
  }
  const int mode = results[index].mode;
  const int angle = results[index].angle_delta;
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = block.mi_row + yy;
    if (rr >= mi_rows) break;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = block.mi_col + xx;
      if (cc >= mi_cols) break;
      const std::size_t idx =
          static_cast<std::size_t>(rr) * static_cast<std::size_t>(mi_cols) +
          static_cast<std::size_t>(cc);
      mi_mode[idx] = static_cast<std::int8_t>(mode);
      mi_angle[idx] = static_cast<std::int8_t>(angle);
    }
  }
}

// Fills the full-UV chroma inputs from device state: the leaf's own (or the
// co-located luma-tree) YMode/AngleDeltaY and the §7.13.2.15/.16 UVSmooth
// neighbour flags.  Mirrors `still_encoder.cpp`'s `neighborhood_smooth()` for
// plane > 0 and the `uv_y_mode` selection for CHROMA_PART leaves.
__global__ void prepare_chroma_job_kernel(cm1::StillChromaJob* jobs,
                                          const BlockPlan* blocks,
                                          const srch::LumaSearchResult* luma,
                                          const std::int8_t* mi_mode,
                                          const std::int8_t* mi_angle,
                                          const std::uint8_t* uv_smooth,
                                          int mi_cols, int mi_rows,
                                          int enable_edge_filter, int index) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  cm1::StillChromaJob& job = jobs[index];
  const BlockPlan& block = blocks[index];
  int y_mode = 0;
  int y_angle = 0;
  if (block.luma_w != 0 && block.luma_h != 0) {
    y_mode = luma[index].mode;
    y_angle = luma[index].angle_delta;
  } else {
    const int r = block.chroma_mi_row;
    const int c = block.chroma_mi_col;
    if (r >= 0 && r < mi_rows && c >= 0 && c < mi_cols) {
      const std::size_t idx =
          static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols) +
          static_cast<std::size_t>(c);
      y_mode = mi_mode[idx];
      y_angle = mi_angle[idx];
    }
  }
  job.y_mode = y_mode;
  job.y_angle_delta = y_angle;
  job.enable_edge_filter = enable_edge_filter;
  const int cmi_cols = mi_cols / 2;
  const int cmi_rows = mi_rows / 2;
  const int cr = job.y >> 2;
  const int cc = job.x >> 2;
  job.filter_type_above =
      (cr > 0 && cc < cmi_cols)
          ? (uv_smooth[static_cast<std::size_t>(cr - 1) * cmi_cols + cc] != 0
                 ? 1
                 : 0)
          : 0;
  job.filter_type_left =
      (cc > 0 && cr < cmi_rows)
          ? (uv_smooth[static_cast<std::size_t>(cr) * cmi_cols + (cc - 1)] != 0
                 ? 1
                 : 0)
          : 0;
}

// ---------------------------------------------------------------------------
// §5.4.9 / §5.18.7.1 device segment state
// ---------------------------------------------------------------------------
//
// `--still-per-segment-q` splits the frame into two used segments: segment 0
// keeps base_q_idx and segment 1 carries a SEG_LVL_ALT_Q feature.  The table
// is the spec's full 8-segment state (enable_ext_seg == 0), so the per-SB
// qindex derivation is a device-side get_qindex() that consults the same
// feature array the header writer signals.
struct DeviceSegmentState {
  std::int32_t base_q_idx = 0;
  std::int32_t num_segments = 8;
  std::int32_t sb_px = 64;
  std::int32_t sb_cols = 0;
  std::int32_t sb_rows = 0;
  std::int32_t width = 0;
  std::int32_t height = 0;
  std::int32_t alt_q[8] = {};
  std::uint8_t alt_q_active[8] = {};
};

// §7.14.2 get_qindex(false, segment_id) for the no-UV-delta still case.
__device__ int segment_block_qindex(const DeviceSegmentState& state,
                                    int segment_id) {
  int q = state.base_q_idx;
  if (segment_id >= 0 && segment_id < state.num_segments &&
      state.alt_q_active[segment_id] != 0) {
    q += state.alt_q[segment_id];
  }
  if (q < 1) q = 1;
  if (q > 255) q = 255;
  return q;
}

// Builds the per-SB segment map with the CPU planner's deterministic rule:
// variance = (sumsq*n - sum*sum) / (n*n) over the *visible* luma samples of
// each superblock; a superblock at or below the frame mean variance joins
// segment 1.  The same kernel then patches the per-SB qindex into the TX and
// reconstruction descriptors, so no host round trip is needed before the
// block loop.
__global__ void derive_segment_map_kernel(
    const std::uint8_t* src, DeviceSegmentState state,
    std::uint64_t* variance, std::uint8_t* segment, std::uint8_t* qindex,
    const BlockPlan* blocks, StillTxJob* tx_jobs,
    cm1::ReconBlockInfo* recon_infos, int nblocks) {
  const int sb_count = state.sb_cols * state.sb_rows;
  __shared__ unsigned long long reduce[256];
  for (int sb = static_cast<int>(threadIdx.x); sb < sb_count;
       sb += static_cast<int>(blockDim.x)) {
    const int sby = sb / state.sb_cols;
    const int sbx = sb - sby * state.sb_cols;
    const int x0 = sbx * state.sb_px;
    const int y0 = sby * state.sb_px;
    const int w = state.sb_px < state.width - x0 ? state.sb_px
                                                 : state.width - x0;
    const int h = state.sb_px < state.height - y0 ? state.sb_px
                                                  : state.height - y0;
    std::uint64_t sum = 0;
    std::uint64_t sumsq = 0;
    for (int y = 0; y < h; ++y) {
      const std::uint8_t* row =
          src + static_cast<std::size_t>(y0 + y) * state.width + x0;
      for (int x = 0; x < w; ++x) {
        const std::uint64_t v = row[x];
        sum += v;
        sumsq += v * v;
      }
    }
    const std::uint64_t n = static_cast<std::uint64_t>(w) * h;
    variance[sb] = n == 0 ? 0 : (sumsq * n - sum * sum) / (n * n);
  }
  __syncthreads();

  unsigned long long local = 0;
  for (int sb = static_cast<int>(threadIdx.x); sb < sb_count;
       sb += static_cast<int>(blockDim.x)) {
    local += variance[sb];
  }
  reduce[threadIdx.x] = local;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
       stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const std::uint64_t mean =
      sb_count == 0 ? 0 : reduce[0] / static_cast<std::uint64_t>(sb_count);
  for (int sb = static_cast<int>(threadIdx.x); sb < sb_count;
       sb += static_cast<int>(blockDim.x)) {
    const int seg = variance[sb] <= mean ? 1 : 0;
    segment[sb] = static_cast<std::uint8_t>(seg);
    qindex[sb] = static_cast<std::uint8_t>(segment_block_qindex(state, seg));
  }
  __syncthreads();

  const int sb_mi = state.sb_px / 4;
  for (int j = static_cast<int>(threadIdx.x); j < nblocks * 3;
       j += static_cast<int>(blockDim.x)) {
    const int i = j / 3;
    const BlockPlan& block = blocks[i];
    const int sb = (block.mi_row / sb_mi) * state.sb_cols +
                   block.mi_col / sb_mi;
    StillTxJob& job = tx_jobs[j];
    job.use_deltaq = 1;
    job.ratio_scale = 1;  // CPU apply_qindex_step() order
    job.qindex = qindex[sb];
    job.delta_dc = 0;
    job.delta_ac = 0;
    job.plane = j - i * 3;
  }
  for (int i = static_cast<int>(threadIdx.x); i < nblocks;
       i += static_cast<int>(blockDim.x)) {
    const BlockPlan& block = blocks[i];
    const int sb = (block.mi_row / sb_mi) * state.sb_cols +
                   block.mi_col / sb_mi;
    recon_infos[i].qindex = qindex[sb];
  }
}

::av2enc::still::StillConfig make_still_config(const GpuOptions& options,
                                      const ::av2enc::still::StillPicture& picture) {  ::av2enc::still::StillConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_intra_edge_filter =
      options.enable_edge_filter && std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr;
  config.enable_cfl_intra = options.enable_cfl;
  config.cfl_ds_filter_index = options.cfl_ds_filter;
  config.enable_ibp = options.enable_ibp;
  config.enable_mrls = options.enable_mrl;
  config.enable_dip = options.enable_dip;
  config.enable_mhccp = options.mhccp;
  config.enable_palette = options.palette;
  config.allow_screen_content_tools = options.palette || options.ibc;
  config.force_integer_mv = true;
  config.allow_intrabc = options.ibc;
  config.allow_global_intrabc = options.ibc;
  config.allow_local_intrabc = false;
  config.enable_ext_partitions = options.ext_partitions;
  config.enable_uneven_4way_partitions = options.ext_partitions;
  config.enable_sdp = options.sdp;
  // §5.4.3 sequence_partition_config(): a still frame uses the matching
  // SbSize; 256x256 is clamped to 128x128 for an intra frame (§5.18.1).
  config.use_128x128_superblock = options.sb_size == 128;
  config.use_256x256_superblock = options.sb_size == 256;
  config.enable_chroma_dctonly = true;
  config.tx_mode_select = false;
  // §5.4.8 enable_idtx_intra / enable_intra_ist / enable_cctx / enable_tcq.
  // These change sequence/frame header bits and the syntax writer's symbol
  // sets, so mirror the CPU reference's flags exactly.
  config.enable_idtx_intra = options.idtx;
  config.enable_intra_ist = options.ist;
  config.enable_cctx = options.cctx;
  config.enable_tcq = options.tcq;
  config.tx_mode_select = options.tx_partition;
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u = options.enable_deblock;
  config.apply_deblock_v = options.enable_deblock;
  // §5.18.7.1 / §5.18.7.8 neutral delta-Q / segmentation syntax.  The
  // per-segment-Q table is installed by run_gpu_frame (it needs a live
  // SegmentationParams), so only the neutral flags are set here.
  config.per_sb_delta_q.segmentation_enabled = options.enable_segmentation;
  config.per_sb_delta_q.delta_q_present = options.enable_delta_q;
  config.per_sb_delta_q.delta_q_res =
      options.enable_delta_q ? options.delta_q_res : 0;
  return config;
}

#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
// ---------------------------------------------------------------------------
// §5.20.7.27/§9.6 intra TX-type search
// ---------------------------------------------------------------------------
//
// The candidate set depends on the *decided* intra mode, which lives on the
// device.  Instead of a per-block descriptor upload, the search runs over the
// full 16-TYPE superset for the block's TX size and a device prepass rewrites
// each candidate's `type_bits` from the per-size legality/symbol-bit tables:
// an illegal candidate gets an impossible cost, so the shared decide kernel
// still picks exactly the CPU `intra_tx_candidates()` winner.
constexpr int kTxTypeSlots = kTxTypeSearchSlots;
constexpr int kTxTypeModeCount = 13;
constexpr int kTxTypeAngleCount = 7;   // -3..3
constexpr int kTxTypeMrlCount = 4;     // 0..3
constexpr int kTxTypeMapStride =
    kTxTypeModeCount * kTxTypeAngleCount * kTxTypeMrlCount;

struct TxTypeHostTables {
  std::vector<TxTypeCandJob> cands;    // 25 * 16
  std::vector<TxBlockJob> jobs;        // 25 * 16
  std::vector<TxTypeBlockJob> blocks;  // 25
  std::vector<std::int8_t> map;        // 25 * kTxTypeMapStride
  std::vector<std::uint8_t> legal;     // 25 * 13 * 16
  std::vector<double> bits;            // 25 * 13 * 16
};

bool build_tx_type_tables(bool allow_idtx, TxTypeHostTables* out,
                          std::string& error) {
  out->cands.assign(25u * kTxTypeSlots, TxTypeCandJob{});
  out->jobs.assign(25u * kTxTypeSlots, TxBlockJob{});
  out->blocks.assign(25u, TxTypeBlockJob{});
  out->map.assign(25u * kTxTypeMapStride, -1);
  out->legal.assign(25u * kTxTypeModeCount * kTxTypeSlots, 0);
  out->bits.assign(25u * kTxTypeModeCount * kTxTypeSlots, 0.0);
  for (int size = 0; size < 25; ++size) {
    const auto dct = txv::make_tx_params(static_cast<txv::TxSize>(size),
                                         txv::TxType::kDctDct);
    if (!dct.has_value()) {
      error = "tx_type: bad TX size index " + std::to_string(size);
      return false;
    }
    TxTypeBlockJob& block = out->blocks[static_cast<std::size_t>(size)];
    block.cand_begin = 0;
    block.cand_count = kTxTypeSlots;
    block.coded = 0;
    block.coded_w = dct->coded_w;
    block.coded_h = dct->coded_h;
    for (int type = 0; type < kTxTypeSlots; ++type) {
      const auto params = txv::make_tx_params(
          static_cast<txv::TxSize>(size),
          static_cast<txv::TxType>(type));
      if (!params.has_value()) {
        error = "tx_type: unsupported candidate type";
        return false;
      }
      const std::size_t slot =
          static_cast<std::size_t>(size) * kTxTypeSlots + type;
      TxTypeCandJob& cand = out->cands[slot];
      cand.tx.params = *params;
      cand.tx.in_offset = static_cast<std::int32_t>(type) * kTxTypeAdjElems;
      cand.tx.out_offset =
          static_cast<std::int32_t>(type) * kTxTypeCodedElems;
      cand.adj_src = 0;
      cand.block = 0;
      cand.type = type;
      cand.candidate = type;
      cand.type_bits = 0.0;
      out->jobs[slot] = cand.tx;
    }
    // §5.20.7.29 mapped intra mode for every (mode, angle, mrl) at this size.
    for (int mode = 0; mode < kTxTypeModeCount; ++mode) {
      for (int angle = -3; angle <= 3; ++angle) {
        for (int mrl = 0; mrl < kTxTypeMrlCount; ++mrl) {
          const int mapped = ::av2enc::still::mrl_stream::
              mapped_intra_mode_for_mrl(mode, angle, mrl, dct->coded_w,
                                        dct->coded_h);
          out->map[static_cast<std::size_t>(size) * kTxTypeMapStride +
                   static_cast<std::size_t>(
                       (mode * kTxTypeAngleCount + (angle + 3)) *
                           kTxTypeMrlCount +
                       mrl)] = static_cast<std::int8_t>(mapped);
        }
      }
    }
    // §5.20.8.2 candidate set + §9.3 symbol cost per mapped mode.
    for (int mapped = 0; mapped < kTxTypeModeCount; ++mapped) {
      txtype::TxTypeQuery query;
      query.coded_w = dct->coded_w;
      query.coded_h = dct->coded_h;
      query.mapped_mode = mapped;
      query.reduced_tx_set = 0;
      query.dct_only = false;
      txtype::TxTypeSet set;
      if (!txtype::intra_tx_candidates(query, &set)) {
        continue;
      }
      for (int i = 0; i < set.count; ++i) {
        const txtype::TxTypeCandidate& candidate = set.candidates[i];
        txv::Tx1dType row = txv::Tx1dType::kDct;
        txv::Tx1dType col = txv::Tx1dType::kDct;
        txv::tx_1d_types(candidate.type, &row, &col);
        const bool is_idtx = row == txv::Tx1dType::kIdt &&
                             col == txv::Tx1dType::kIdt;
        const bool mixed_identity =
            (row == txv::Tx1dType::kIdt) != (col == txv::Tx1dType::kIdt);
        // CPU `compute_tx`: the enable_idtx_intra=0 path filters every
        // identity-based 1-D type; with the flag on only pure IDTX is enabled.
        if (mixed_identity) continue;
        if (is_idtx && !allow_idtx) continue;
        const int type = static_cast<int>(candidate.type);
        if (type < 0 || type >= kTxTypeSlots) continue;
        const std::size_t idx =
            (static_cast<std::size_t>(size) * kTxTypeModeCount + mapped) *
                kTxTypeSlots +
            type;
        out->legal[idx] = 1;
        out->bits[idx] = txtype::static_type_bits(query, candidate);
      }
    }
  }
  return true;
}

__global__ void tx_type_prepare_candidates_kernel(
    const srch::LumaSearchResult* __restrict__ results,
    const iext::Choice* __restrict__ choices,
    const std::int8_t* __restrict__ map_table,
    const std::uint8_t* __restrict__ legal_table,
    const double* __restrict__ bits_table,
    const TxTypeCandJob* __restrict__ src_cands,
    TxTypeCandJob* __restrict__ cands, int ncand) {
  int mode = results->mode;
  int angle = results->angle_delta;
  int mrl = choices->mrl_index;
  if (mode < 0 || mode >= kTxTypeModeCount) mode = 0;
  if (angle < -3) angle = -3;
  if (angle > 3) angle = 3;
  if (mrl < 0 || mrl >= kTxTypeMrlCount) mrl = 0;
  int mapped = static_cast<int>(
      map_table[(mode * kTxTypeAngleCount + (angle + 3)) * kTxTypeMrlCount +
                mrl]);
  if (mapped < 0 || mapped >= kTxTypeModeCount) mapped = 0;
  for (int j = static_cast<int>(threadIdx.x); j < ncand;
       j += static_cast<int>(blockDim.x)) {
    TxTypeCandJob cand = src_cands[j];
    const int type = cand.type;
    const std::size_t idx =
        static_cast<std::size_t>(mapped) * kTxTypeSlots + type;
    cand.type_bits =
        (type >= 0 && type < kTxTypeSlots && legal_table[idx] != 0)
            ? bits_table[idx]
            : 1.0e9;
    cands[j] = cand;
  }
}

__global__ void tx_type_apply_decision_kernel(
    const TxTypeDecision* __restrict__ decision,
    const TxTypeCandJob* __restrict__ cands,
    StillTxJob* __restrict__ job, std::int32_t* __restrict__ types,
    int slot) {
  int index = decision->candidate_index;
  if (index < 0 || index >= kTxTypeSlots) {
    for (int j = 0; j < kTxTypeSlots; ++j) {
      if (cands[j].type == decision->type) {
        index = j;
        break;
      }
    }
  }
  if (index >= 0 && index < kTxTypeSlots &&
      cands[index].type == decision->type) {
    job->params = cands[index].tx.params;
  }
  types[slot] = decision->type;
}

// §7.15 residual = source - prediction for the coded block (source is clamped
// to the visible plane, exactly like `TileEncoder::source_sample`).
__global__ void still_residual_coded_kernel(
    const std::uint8_t* __restrict__ src, int src_stride, int src_max_x,
    int src_max_y, const std::uint8_t* __restrict__ pred, int w, int h,
    int x0, int y0, std::int16_t* __restrict__ out) {
  const int n = w * h;
  for (int i = static_cast<int>(threadIdx.x); i < n;
       i += static_cast<int>(blockDim.x)) {
    const int y = i / w;
    const int x = i - y * w;
    int px = x0 + x;
    int py = y0 + y;
    px = px < 0 ? 0 : (px > src_max_x ? src_max_x : px);
    py = py < 0 ? 0 : (py > src_max_y ? src_max_y : py);
    const int s = src[static_cast<std::size_t>(py) * src_stride + px];
    out[i] = static_cast<std::int16_t>(s - static_cast<int>(pred[i]));
  }
}
#endif  // AV2ENC_STILL_GPU_HAVE_TX_TYPE

#if AV2ENC_STILL_GPU_HAVE_SECONDARY
// ---------------------------------------------------------------------------
// §7.15.3 intra secondary transform (IST) candidate search
// ---------------------------------------------------------------------------
//
// Mirrors `TileEncoder::compute_tx`'s IST block: the incumbent is the primary
// reconstruction with no secondary transform; each (most_probable_stx_set,
// sec_tx_type) candidate projects the primary levels, dequantises, applies the
// normative §7.15.3 inverse and scores `SSE + lambda * symbol bits`.  All
// coefficient/pixel work stays on device; only the static §9.3 symbol costs and
// the kernel/transpose tables are precomputed on the host.
constexpr int kIstSets = 3;
constexpr int kIstSecTypes = 3;
constexpr int kIstSlots = kIstSets * kIstSecTypes;
constexpr int kIstTypeSels = 2;  // DCT_DCT(0), ADST_ADST(3)

struct IstHostTables {
  std::vector<TxSecondaryJob> jobs;    // 25 * 2 * 13 * 9
  std::vector<TxBlockJob> block_jobs;  // 25 * 2 * 9
  std::vector<double> bits;            // 25 * 2 * 13 * 9
  std::vector<std::uint8_t> set_count; // 25 * 2 * 13
  std::vector<std::int8_t> map;        // 25 * 13 * 7 * 4
};

bool build_ist_tables(IstHostTables* out, std::string& error) {
  const std::size_t map_stride = static_cast<std::size_t>(kTxTypeMapStride);
  out->jobs.assign(25u * kIstTypeSels * 13u * kIstSlots, TxSecondaryJob{});
  out->block_jobs.assign(25u * kIstTypeSels * kIstSlots, TxBlockJob{});
  out->bits.assign(25u * kIstTypeSels * 13u * kIstSlots, 0.0);
  out->set_count.assign(25u * kIstTypeSels * 13u, 0);
  out->map.assign(25u * map_stride, -1);
  for (int size = 0; size < 25; ++size) {
    const auto params = txv::make_tx_params(static_cast<txv::TxSize>(size),
                                            txv::TxType::kDctDct);
    if (!params.has_value()) {
      error = "ist: bad TX size index";
      return false;
    }
    const int cw = params->coded_w;
    const int ch = params->coded_h;
    for (int mode = 0; mode < kTxTypeModeCount; ++mode) {
      for (int angle = -3; angle <= 3; ++angle) {
        for (int mrl = 0; mrl < kTxTypeMrlCount; ++mrl) {
          out->map[static_cast<std::size_t>(size) * map_stride +
                   static_cast<std::size_t>(
                       (mode * kTxTypeAngleCount + (angle + 3)) *
                           kTxTypeMrlCount +
                       mrl)] = static_cast<std::int8_t>(
              txsec::map_y_mode(mode, angle, mrl, cw, ch));
        }
      }
    }
    for (int type_sel = 0; type_sel < kIstTypeSels; ++type_sel) {
      const txv::TxType type =
          type_sel == 0 ? txv::TxType::kDctDct : txv::TxType::kAdstAdst;
      const auto tparams = txv::make_tx_params(
          static_cast<txv::TxSize>(size), type);
      if (!tparams.has_value()) {
        error = "ist: bad candidate transform size";
        return false;
      }
      const int set_count =
          txsec::most_probable_stx_set_count(type, cw, ch);
      for (int mapped = 0; mapped < kTxTypeModeCount; ++mapped) {
        out->set_count[(static_cast<std::size_t>(size) * kIstTypeSels +
                        type_sel) *
                           13u +
                       mapped] = static_cast<std::uint8_t>(set_count);
        for (int set = 0; set < kIstSets; ++set) {
          int kernel = -1;
          if (set < set_count) {
            if (!txsec::kernel_for_syntax(mapped, type, set, false, cw, ch,
                                          &kernel)) {
              kernel = -1;
            }
          }
          const bool transpose = txsec::transpose_for_mode(mapped);
          for (int sec = 1; sec <= kIstSecTypes; ++sec) {
            const int j = set * kIstSecTypes + (sec - 1);
            const std::size_t slot =
                ((static_cast<std::size_t>(size) * kIstTypeSels + type_sel) *
                     13u +
                 mapped) *
                    kIstSlots +
                j;
            TxSecondaryJob& job = out->jobs[slot];
            job.tx_size = size;
            job.plane_tx_type = static_cast<int>(type);
            job.kernel = kernel;
            job.sec_tx_type = sec;
            job.transpose = transpose ? 1 : 0;
            job.bit_depth = 8;
            job.in_offset = j * kTxTypeAdjElems;
            job.out_offset = j * kTxTypeAdjElems;
            out->block_jobs[(static_cast<std::size_t>(size) * kIstTypeSels +
                             type_sel) *
                                kIstSlots +
                            j] = TxBlockJob{
                *tparams, j * kTxTypeAdjElems, j * kTxTypeCodedElems};
            if (kernel >= 0 && set < set_count) {
              out->bits[slot] =
                  txsec::sec_tx_type_bits(sec, size, false) +
                  txsec::most_probable_stx_set_bits(set, type, cw, ch);
            }
          }
        }
      }
    }
  }
  return true;
}

struct IstDevTables {
  const std::int8_t* ist4;
  const std::int8_t* ist8;
  const std::uint8_t* scan_map;
  const std::uint8_t* tx_width;
  const std::uint8_t* tx_height;
  const std::uint8_t* tx_width_log2;
  const std::uint8_t* scan_order4;
  const std::uint8_t* scan_order8;
};

struct IstSizeInfo {
  int w = 0;
  int h = 0;
  int n = 0;
  int n2 = 0;
  bool large = false;
};

__device__ __forceinline__ IstDevTables ist_device_tables() {
  IstDevTables t;
  t.ist4 = &av2enc::txsec::dev::kIst4x4Kernel[0][0][0][0];
  t.ist8 = &av2enc::txsec::dev::kIst8x8Kernel[0][0][0][0];
  t.scan_map = &av2enc::txsec::dev::kStxScanMap[0][0][0];
  t.tx_width = &av2enc::txsec::dev::kTxWidth[0];
  t.tx_height = &av2enc::txsec::dev::kTxHeight[0];
  t.tx_width_log2 = &av2enc::txsec::dev::kTxWidthLog2[0];
  t.scan_order4 = &av2enc::txsec::dev::kStxScanOrder4x4[0];
  t.scan_order8 = &av2enc::txsec::dev::kStxScanOrder8x8[0];
  return t;
}

// Device twin of `txsec::secondary_size_info()` (the component header's
// helpers are not inline and cannot be included in a second translation unit).
__device__ __forceinline__ bool ist_size_info(const IstDevTables& t,
                                              int tx_size, int plane_tx_type,
                                              IstSizeInfo* out) {
  if (out == nullptr || tx_size < 0 || tx_size >= 25) return false;
  IstSizeInfo info;
  info.w = t.tx_width[tx_size] < 32 ? t.tx_width[tx_size] : 32;
  info.h = t.tx_height[tx_size] < 32 ? t.tx_height[tx_size] : 32;
  info.large = info.w >= 8 && info.h >= 8;
  if (!info.large) {
    info.n = 8;
  } else if (tx_size == 1 || plane_tx_type == 3) {
    info.n = 20;
  } else {
    info.n = 32;
  }
  info.n2 = info.large ? 48 : 16;
  *out = info;
  return true;
}

// §5.20.7.30 diagonal scan (`syntax::build_scan` on the CPU).
__device__ __forceinline__ void ist_build_scan(int w, int h,
                                               int* scan) {
  int x = 0;
  int y = 0;
  const int n = w * h;
  for (int c = 0; c < n; ++c) {
    scan[c] = y * w + x;
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = x < h - 1 - y ? x : h - 1 - y;
      x -= s;
      y += s;
    }
  }
}

__global__ void ist_prepare_kernel(
    const srch::LumaSearchResult* __restrict__ results,
    const iext::Choice* __restrict__ choices, int size_idx,
    const std::int8_t* __restrict__ map_table,
    const TxSecondaryJob* __restrict__ job_table,
    const TxBlockJob* __restrict__ block_table,
    const double* __restrict__ bit_table,
    const std::int32_t* __restrict__ tx_types,
    TxSecondaryJob* __restrict__ jobs, TxBlockJob* __restrict__ block_jobs,
    std::uint8_t* __restrict__ valid, double* __restrict__ bits,
    int* __restrict__ primary_type) {
  const int mode = results->mode;
  const int angle = results->angle_delta;
  const int mrl = choices->mrl_index;
  const int type = tx_types[blockIdx.x];
  if (threadIdx.x == 0) {
    *primary_type = type;
  }
  int type_sel = -1;
  if (type == 0) type_sel = 0;
  if (type == 3) type_sel = 1;
  int mapped = -1;
  if (mode >= 0 && mode < kTxTypeModeCount && angle >= -3 && angle <= 3 &&
      mrl >= 0 && mrl < kTxTypeMrlCount) {
    mapped = map_table[(mode * kTxTypeAngleCount + (angle + 3)) *
                           kTxTypeMrlCount +
                       mrl];
  }
  const int j = static_cast<int>(threadIdx.x);
  if (j >= kIstSlots) return;
  std::uint8_t ok = 0;
  TxSecondaryJob job{};
  job.tx_size = size_idx;
  job.plane_tx_type = 0;
  job.kernel = -1;
  job.sec_tx_type = 1;
  job.transpose = 0;
  job.bit_depth = 8;
  job.in_offset = j * kTxTypeAdjElems;
  job.out_offset = j * kTxTypeAdjElems;
  TxBlockJob block_job{};
  block_job = block_table[static_cast<std::size_t>(size_idx) * kIstTypeSels *
                              kIstSlots +
                          j];
  double bits_value = 0.0;
  if (type_sel >= 0 && mapped >= 0 && mapped < kTxTypeModeCount) {
    const std::size_t slot =
        ((static_cast<std::size_t>(size_idx) * kIstTypeSels + type_sel) * 13u +
         mapped) *
            kIstSlots +
        j;
    job = job_table[slot];
    bits_value = bit_table[slot];
    ok = job.kernel >= 0 ? 1 : 0;
    if (ok != 0) {
      block_job =
          block_table[(static_cast<std::size_t>(size_idx) * kIstTypeSels +
                       type_sel) *
                          kIstSlots +
                      j];
    }
  }
  jobs[j] = job;
  block_jobs[j] = block_job;
  valid[j] = ok;
  bits[j] = bits_value;
}

// Encoder-side analysis projection, exactly `project_ist_levels()`.
__global__ void ist_project_kernel(const std::int32_t* __restrict__ primary,
                                   const TxSecondaryJob* __restrict__ jobs,
                                   std::int32_t* __restrict__ levels) {
  const int j = static_cast<int>(blockIdx.x);
  const TxSecondaryJob job = jobs[j];
  if (job.kernel < 0 || job.sec_tx_type < 1 || job.sec_tx_type > 3) return;
  const IstDevTables t = ist_device_tables();
  IstSizeInfo info;
  if (!ist_size_info(t, job.tx_size, job.plane_tx_type, &info)) {
    return;
  }
  const int elems = info.w * info.h;
  std::int32_t* slot = levels + job.out_offset;
  for (int i = 0; i < elems; ++i) slot[i] = primary[i];
  const bool large = info.large;
  const int kernel_width = large ? 48 : 16;
  const int kernel_index = job.kernel * 3 + (job.sec_tx_type - 1);
  const std::int8_t* matrix =
      large ? t.ist8 + kernel_index * 32 * 48
            : t.ist4 + kernel_index * 8 * 16;
  const int scan_bwl = large ? 3 : 2;
  const int scan_mask = (1 << scan_bwl) - 1;
  int scan[1024];
  ist_build_scan(info.w, info.h, scan);
  int out_pos[48];
  std::int32_t x[48];
  for (int i = 0; i < info.n2; ++i) {
    const int pos =
        large ? t.scan_order8[t.scan_map[kernel_index * 48 + i]]
              : t.scan_order4[i];
    const int sx = pos & scan_mask;
    const int sy = pos >> scan_bwl;
    const int index = job.transpose ? sx * info.w + sy : sy * info.w + sx;
    out_pos[i] = index;
    x[i] = slot[index];
  }
  std::int32_t sec[32];
  for (int i = 0; i < info.n; ++i) {
    std::int64_t sum = 0;
    for (int k = 0; k < info.n2; ++k) {
      sum += static_cast<std::int64_t>(x[k]) *
             static_cast<std::int64_t>(matrix[i * kernel_width + k]);
    }
    const std::int64_t magnitude = sum < 0 ? -sum : sum;
    const std::int64_t rounded =
        (magnitude + (static_cast<std::int64_t>(1) << 6)) >> 7;
    sec[i] = static_cast<std::int32_t>(sum < 0 ? -rounded : rounded);
  }
  for (int i = 0; i < info.n2; ++i) {
    const int index = out_pos[i];
    bool in_first_n = false;
    for (int k = 0; k < info.n; ++k) {
      if (scan[k] == index) {
        in_first_n = true;
        break;
      }
    }
    if (!in_first_n) slot[index] = 0;
  }
  for (int i = 0; i < info.n; ++i) {
    slot[scan[i]] = sec[i];
  }
}

__global__ void ist_eob_kernel(const TxSecondaryJob* __restrict__ jobs,
                               const std::int32_t* __restrict__ levels,
                               std::int32_t* __restrict__ eob) {
  const int j = static_cast<int>(blockIdx.x);
  const TxSecondaryJob job = jobs[j];
  if (job.kernel < 0 || job.sec_tx_type < 1 || job.sec_tx_type > 3) {
    eob[j] = 0;
    return;
  }
  const IstDevTables t = ist_device_tables();
  IstSizeInfo info;
  if (!ist_size_info(t, job.tx_size, job.plane_tx_type, &info)) {
    eob[j] = -1;
    return;
  }
  int area = info.w * info.h;
  int scan[1024];
  ist_build_scan(info.w, info.h, scan);
  const std::int32_t* slot = levels + job.out_offset;
  int value = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (slot[scan[c]] != 0) {
      value = c + 1;
      break;
    }
  }
  eob[j] = value;
}

__device__ __forceinline__ int ist_clip1(int v, int bit_depth) {
  const int high = (1 << bit_depth) - 1;
  return v < 0 ? 0 : (v > high ? high : v);
}

__global__ void ist_sse_kernel(
    const std::uint8_t* __restrict__ src, int src_stride, int src_max_x,
    int src_max_y, const std::uint8_t* __restrict__ pred,
    const TxBlockJob* __restrict__ block_jobs,
    const std::int16_t* __restrict__ residual, int x0, int y0,
    double* __restrict__ sse) {
  const int j = static_cast<int>(blockIdx.x);
  const TxBlockJob job = block_jobs[j];
  const int w = job.params.coded_w;
  const int h = job.params.coded_h;
  const std::int16_t* rr = residual + job.out_offset;
  double sum = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int px = x0 + x;
      int py = y0 + y;
      px = px < 0 ? 0 : (px > src_max_x ? src_max_x : px);
      py = py < 0 ? 0 : (py > src_max_y ? src_max_y : py);
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      const int v = ist_clip1(static_cast<int>(pred[i]) +
                                  static_cast<int>(rr[i]), 8);
      const double d =
          static_cast<double>(src[static_cast<std::size_t>(py) * src_stride + px]) -
          static_cast<double>(v);
      sum += d * d;
    }
  }
  sse[j] = sum;
}

__global__ void ist_base_sse_kernel(
    const std::uint8_t* __restrict__ src, int src_stride, int src_max_x,
    int src_max_y, const std::uint8_t* __restrict__ pred,
    const std::int16_t* __restrict__ residual, int w, int h, int x0, int y0,
    double* __restrict__ sse) {
  double sum = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int px = x0 + x;
      int py = y0 + y;
      px = px < 0 ? 0 : (px > src_max_x ? src_max_x : px);
      py = py < 0 ? 0 : (py > src_max_y ? src_max_y : py);
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      const int v = ist_clip1(static_cast<int>(pred[i]) +
                                  static_cast<int>(residual[i]), 8);
      const double d =
          static_cast<double>(src[static_cast<std::size_t>(py) * src_stride + px]) -
          static_cast<double>(v);
      sum += d * d;
    }
  }
  *sse = sum;
}

__global__ void ist_decide_kernel(
    const std::int32_t* __restrict__ primary_eob_addr,
    const std::int32_t* __restrict__ type_addr,
    const std::uint8_t* __restrict__ valid,
    const std::int32_t* __restrict__ eob, const double* __restrict__ sse,
    const double* __restrict__ bits, const double* __restrict__ base_sse_addr, double lambda,
    int coded_w, int coded_h, std::int32_t* __restrict__ primary_levels,
    std::int32_t* __restrict__ primary_eob,
    std::int16_t* __restrict__ primary_residual,
    const std::int32_t* __restrict__ candidate_levels,
    const std::int16_t* __restrict__ candidate_residual,
    std::int32_t* __restrict__ sec_out, std::int32_t* __restrict__ set_out,
    int adj_elems, int coded_elems) {
  const int primary = *primary_eob_addr;
  const int type = *type_addr;
  const double base_sse = *base_sse_addr;
  const bool large = coded_w >= 8 && coded_h >= 8;
  int limit = 0;
  if (large) {
    limit = ((coded_w == 8 && coded_h == 8) || type == 3) ? 20 : 32;
  } else {
    limit = 8;
  }
  double best_cost = base_sse;
  int best_j = -1;
  int best_sec = 0;
  int best_set = 0;
  int best_eob = primary;
  if (primary >= 2 && primary <= limit) {
    for (int set = 0; set < kIstSets; ++set) {
      for (int sec = 1; sec <= kIstSecTypes; ++sec) {
        const int j = set * kIstSecTypes + (sec - 1);
        if (valid[j] == 0) continue;
        const int ev = eob[j];
        if (ev < 2 || ev > limit) continue;
        const double cost = sse[j] + lambda * bits[j];
        if (cost < best_cost) {
          best_cost = cost;
          best_j = j;
          best_sec = sec;
          best_set = set;
          best_eob = ev;
        }
      }
    }
  }
  *sec_out = best_sec;
  *set_out = best_set;
  if (best_sec != 0 && best_j >= 0) {
    for (int i = 0; i < adj_elems; ++i) {
      primary_levels[i] = candidate_levels[best_j * kTxTypeAdjElems + i];
    }
    for (int i = 0; i < coded_elems; ++i) {
      primary_residual[i] = candidate_residual[best_j * kTxTypeCodedElems + i];
    }
    *primary_eob = best_eob;
  }
}
#endif  // AV2ENC_STILL_GPU_HAVE_SECONDARY

#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
// ---------------------------------------------------------------------------
// §7.14.6 cross-component transform (CCTX) chroma search
// ---------------------------------------------------------------------------
//
// The seven rotations are evaluated on device from the forward chroma
// coefficients; the decision mirrors `TileEncoder::compute_chroma_cctx`:
// reconstruct both planes with the §7.14.4 dequant and the §7.14.6 inverse
// rotation, score serial `SSE(U)+SSE(V) + lambda * cctx_type bits`, and reject
// a non-NONE rotation with U eob <= 1 (it is not signallable).
constexpr int kCctxTypes = ::av2enc::tcq::kCctxTypeCount;  // 7
constexpr int kCctxSlots = kCctxTypes * 2;                 // U and V per type

struct CctxBits {
  double value[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

__device__ __forceinline__ void cctx_build_scan(int w, int h, int* scan) {
  int x = 0;
  int y = 0;
  const int n = w * h;
  for (int c = 0; c < n; ++c) {
    scan[c] = y * w + x;
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = x < h - 1 - y ? x : h - 1 - y;
      x -= s;
      y += s;
    }
  }
}

__global__ void cctx_levels_kernel(const double* __restrict__ u_coeff,
                                   const double* __restrict__ v_coeff,
                                   double quant_step,
                                   std::int32_t* __restrict__ levels,
                                   std::int32_t* __restrict__ eob, int area,
                                   int w, int h) {
  namespace td = ::av2enc::cuda::tcq::detail;
  const int t = static_cast<int>(blockIdx.x);
  std::int32_t* u = levels + (t * 2) * kTxTypeAdjElems;
  std::int32_t* v = levels + (t * 2 + 1) * kTxTypeAdjElems;
  int eob_u = 0;
  int eob_v = 0;
  const double int32_min_plus_one = -2147483647.0;
  const double int32_max = 2147483647.0;
  for (int i = 0; i < area; ++i) {
    double rot_u = 0.0;
    double rot_v = 0.0;
    td::tcq_cctx_forward_rotate(u_coeff[i], v_coeff[i], t, &rot_u, &rot_v);
    const double u_q = td::tcq_round(rot_u / quant_step);
    const double v_q = td::tcq_round(rot_v / quant_step);
    const double uc = u_q < int32_min_plus_one
                          ? int32_min_plus_one
                          : (u_q > int32_max ? int32_max : u_q);
    const double vc = v_q < int32_min_plus_one
                          ? int32_min_plus_one
                          : (v_q > int32_max ? int32_max : v_q);
    const std::int32_t ul = static_cast<std::int32_t>(uc);
    const std::int32_t vl = static_cast<std::int32_t>(vc);
    u[i] = ul;
    v[i] = vl;
    if (ul != 0) eob_u = i + 1;
    if (vl != 0) eob_v = i + 1;
  }
  // Convert raster eob to the §5.20.7.30 scan eob the syntax needs.
  int scan[1024];
  cctx_build_scan(w, h, scan);
  int scan_u = 0;
  int scan_v = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (u[scan[c]] != 0) {
      scan_u = c + 1;
      break;
    }
  }
  for (int c = area - 1; c >= 0; --c) {
    if (v[scan[c]] != 0) {
      scan_v = c + 1;
      break;
    }
  }
  eob[t * 2] = scan_u;
  eob[t * 2 + 1] = scan_v;
  (void)eob_u;
  (void)eob_v;
}

__global__ void cctx_rotate_kernel(std::int32_t* __restrict__ dequant,
                                   int area) {
  namespace td = ::av2enc::cuda::tcq::detail;
  const int t = static_cast<int>(blockIdx.x);
  std::int32_t* u = dequant + (t * 2) * kTxTypeAdjElems;
  std::int32_t* v = dequant + (t * 2 + 1) * kTxTypeAdjElems;
  for (int i = 0; i < area; ++i) {
    const td::TcQCctxPair pair =
        td::tcq_cctx_dequant_pair(u[i], v[i], t, 8);
    u[i] = pair.u;
    v[i] = pair.v;
  }
}

__global__ void cctx_sse_kernel(
    const std::uint8_t* __restrict__ src_u, const std::uint8_t* __restrict__ src_v,
    int src_stride, int src_max_x, int src_max_y,
    const std::uint8_t* __restrict__ pred_u,
    const std::uint8_t* __restrict__ pred_v,
    const std::int16_t* __restrict__ residual, int w, int h, int x0, int y0,
    double* __restrict__ sse) {
  const int t = static_cast<int>(blockIdx.x);
  double sum = 0.0;
  for (int plane = 0; plane < 2; ++plane) {
    const std::uint8_t* src = plane == 0 ? src_u : src_v;
    const std::uint8_t* pred = plane == 0 ? pred_u : pred_v;
    const std::int16_t* rr =
        residual + (t * 2 + plane) * kTxTypeCodedElems;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int px = x0 + x;
        int py = y0 + y;
        px = px < 0 ? 0 : (px > src_max_x ? src_max_x : px);
        py = py < 0 ? 0 : (py > src_max_y ? src_max_y : py);
        const std::size_t i = static_cast<std::size_t>(y) * w + x;
        const int pv = ist_clip1(static_cast<int>(pred[i]) + rr[i], 8);
        const double d =
            static_cast<double>(src[static_cast<std::size_t>(py) * src_stride + px]) -
            static_cast<double>(pv);
        sum += d * d;
      }
    }
  }
  sse[t] = sum;
}

__global__ void cctx_decide_kernel(
    const double* __restrict__ sse, const std::int32_t* __restrict__ eob,
    CctxBits bits, double lambda,
    std::int32_t* __restrict__ out_type,
    std::int32_t* __restrict__ u_levels, std::int32_t* __restrict__ v_levels,
    std::int32_t* __restrict__ u_eob, std::int32_t* __restrict__ v_eob,
    std::int16_t* __restrict__ u_res, std::int16_t* __restrict__ v_res,
    const std::int32_t* __restrict__ cand_levels,
    const std::int16_t* __restrict__ cand_res, int area, int coded_elems) {
  int best = -1;
  double best_cost = 0.0;
  int best_eob_u = 0;
  int best_eob_v = 0;
  for (int t = 0; t < kCctxTypes; ++t) {
    const int eob_u = eob[t * 2];
    if (t != 0 && eob_u <= 1) continue;
    const double b = eob_u > 1 ? bits.value[t] : 0.0;
    const double cost = sse[t] + lambda * b;
    if (best < 0 || cost < best_cost) {
      best = t;
      best_cost = cost;
      best_eob_u = eob_u;
      best_eob_v = eob[t * 2 + 1];
    }
  }
  if (best < 0) {
    *out_type = 0;
    return;
  }
  *out_type = best;
  *u_eob = best_eob_u;
  *v_eob = best_eob_v;
  for (int i = 0; i < area; ++i) {
    u_levels[i] = cand_levels[(best * 2) * kTxTypeAdjElems + i];
    v_levels[i] = cand_levels[(best * 2 + 1) * kTxTypeAdjElems + i];
  }
  for (int i = 0; i < coded_elems; ++i) {
    u_res[i] = cand_res[(best * 2) * kTxTypeCodedElems + i];
    v_res[i] = cand_res[(best * 2 + 1) * kTxTypeCodedElems + i];
  }
}
#endif  // AV2ENC_STILL_GPU_HAVE_TCQ_CORE

// Runs the full GPU frame pipeline in syntax order:
//
//   search -> chosen-mode prediction -> luma TX/recon -> chroma decide (DC/CfL)
//   -> chroma TX/recon -> commit, then deblock and the CPU syntax bridge.
//
// Dependencies stay on one stream, so no per-block host synchronisation is
// needed.  The frame downloads once (recon, levels, EOB, decisions) and the
// caller compares/assembles after the single sync.
GpuRunStatus run_gpu_frame(const ::av2enc::still::StillPicture& picture,
                           const GpuOptions& options, const FramePlan& plan,
                           const std::vector<still_stream::TreeNode>& tree,
                           HostBuffers& host,
                           std::vector<BlockDecision>& decisions,
                           GpuResult& result,
                           const ::av2enc::still::StillResult* cpu_plan,
                           bool& synced, std::string& error) {
  synced = false;
  GpuRuntime& gpu = runtime();
  DeviceBuffers& buffers = gpu.buffers;
  const cudaStream_t stream = gpu.context.stream();
  const int blocks = static_cast<int>(plan.block_count);
  const int mi_cols = plan.mi_cols;
  const int mi_rows = plan.mi_rows;
  const int coded_w = mi_cols * 4;
  const int coded_h = mi_rows * 4;
  const int chroma_coded_w = mi_cols * 2;
  const int chroma_coded_h = mi_rows * 2;
  const int chroma_vis_w = static_cast<int>((picture.width + 1u) / 2u);
  const int chroma_vis_h = static_cast<int>((picture.height + 1u) / 2u);
  // In-loop filters: the CPU reference plans the decisions/parameters (index
  // grids, offsets, unit grids), the GPU applies the normative pixel kernels to
  // the deblocked reconstruction in §7.18/§7.19 order.
  const ::av2enc::still::StillConfig* filter_config =
      cpu_plan != nullptr ? &cpu_plan->config : nullptr;
  const bool cdef_active =
#if AV2ENC_STILL_GPU_HAVE_CDEF
      filter_config != nullptr && filter_config->cdef.enable &&
      cpu_plan != nullptr && !cpu_plan->cdef_idx.empty();
#else
      false;
#endif
  const bool ccso_active =
#if AV2ENC_STILL_GPU_HAVE_CCSO
      filter_config != nullptr && filter_config->ccso.enable;
#else
      false;
#endif
  // §7.20 loop restoration: the CPU reference plans the per-unit decisions and
  // coefficient tables; the GPU applies the normative pixel kernels.  Chroma
  // LR (and the luma LrFrame copy when only chroma is enabled) is part of the
  // same stage.  PC-Wiener replaces luma restoration with §7.20.4.
  const bool pcw_active =
      filter_config != nullptr && filter_config->lr.enable &&
      filter_config->lr.pc_wiener;
  const bool lr_wiener_active =
      filter_config != nullptr && filter_config->lr.enable &&
      !filter_config->lr.pc_wiener;
  const bool lr_active = lr_wiener_active || pcw_active;
  const bool lr_chroma_active =
      lr_active && (filter_config->lr.plane[1].enabled ||
                    filter_config->lr.plane[2].enabled);
  const bool gdf_active =
      filter_config != nullptr && filter_config->gdf.enable;
  // §5.20.8 / §5.20.5.3-.4 screen-content tools: the CPU reference plans the
  // palette colour set/direction and the exact IBC block vector (same
  // leaf-order contract as the filter decision grids); the GPU applies the
  // palette prediction/map kernel and the IBC copy kernel.
  const bool screen_plan =
      cpu_plan != nullptr && (options.palette || options.ibc);
  if (blocks <= 0 || buffers.tx_jobs == nullptr) {
    error = "gpu: device buffers are not prepared";
    return GpuRunStatus::kError;
  }
  for (const BlockPlan& block : plan.blocks) {
    const bool has_luma = block.luma_w != 0 && block.luma_h != 0;
    if (!has_luma && block.has_chroma == 0) {
      error = "unsupported: leaf has neither luma nor chroma";
      return GpuRunStatus::kUnsupported;
    }
    if (block.has_chroma != 0 &&
        (block.chroma_w < 4 || block.chroma_h < 4)) {
      error = "unsupported: chroma block smaller than 4x4";
      return GpuRunStatus::kUnsupported;
    }
  }
  if (std::getenv("AV2ENC_STILL_FORCE_Y") != nullptr ||
      std::getenv("AV2ENC_STILL_FORCE_Y2") != nullptr) {
    error = "unsupported: AV2ENC_STILL_FORCE_Y* debug override";
    return GpuRunStatus::kUnsupported;
  }
  if (screen_plan) {
    if (options.palette &&
        cpu_plan->palette_index_by_leaf.size() !=
            static_cast<std::size_t>(blocks)) {
      error = "gpu: palette plan does not cover the leaf list";
      return GpuRunStatus::kError;
    }
    if (options.ibc &&
        cpu_plan->ibc_by_leaf.size() != static_cast<std::size_t>(blocks)) {
      error = "gpu: IBC plan does not cover the leaf list";
      return GpuRunStatus::kError;
    }
  }

#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
  // ---- §5.20.7.27/§9.6 TX-type search tables ------------------------------
  // Built once per frame for all 25 transform sizes.  The per-block prepass
  // only rewrites the 16 candidate `type_bits` for the decided intra mode.
  const bool tx_type_search = options.tx_type || options.idtx;
  const bool ist_active = options.ist;
  const bool cctx_active = options.cctx;
  const bool tx_type_tables_needed =
      tx_type_search || ist_active || cctx_active;
  TxTypeHostTables tx_tables;
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
  IstHostTables ist_tables;
#endif
  std::size_t tx_stage_off = 0;
  if (tx_type_tables_needed) {
    if (!build_tx_type_tables(options.idtx, &tx_tables, error)) {
      return GpuRunStatus::kError;
    }
    std::size_t table_bytes =
        tx_tables.cands.size() * sizeof(TxTypeCandJob) +
        tx_tables.jobs.size() * sizeof(TxBlockJob) +
        tx_tables.blocks.size() * sizeof(TxTypeBlockJob) +
        tx_tables.map.size() * sizeof(std::int8_t) +
        tx_tables.legal.size() * sizeof(std::uint8_t) +
        tx_tables.bits.size() * sizeof(double);
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
    if (ist_active) {
      if (!build_ist_tables(&ist_tables, error)) {
        return GpuRunStatus::kError;
      }
      table_bytes +=
          ist_tables.jobs.size() * sizeof(TxSecondaryJob) +
          ist_tables.block_jobs.size() * sizeof(TxBlockJob) +
          ist_tables.bits.size() * sizeof(double) +
          ist_tables.set_count.size() * sizeof(std::uint8_t) +
          ist_tables.map.size() * sizeof(std::int8_t);
    }
#endif
      const std::size_t need_stage =
          table_bytes + static_cast<std::size_t>(blocks) * 4096u + 16384u;
    if (gpu.tx_stage.capacity() < need_stage) {
      gpu.tx_stage.release();
      std::string stage_error;
      if (!is_ok(gpu.tx_stage.init(need_stage, &stage_error))) {
        error = "gpu: TX decision staging init failed: " + stage_error;
        return GpuRunStatus::kError;
      }
    }
    auto* stage = static_cast<std::uint8_t*>(gpu.tx_stage.data());
    std::size_t off = 0;
    const auto stage_copy = [&](void* dst, const void* src,
                                std::size_t bytes) -> bool {
      std::memcpy(stage + off, src, bytes);
      const cudaError_t st = cudaMemcpyAsync(dst, stage + off, bytes,
                                             cudaMemcpyHostToDevice, stream);
      off += (bytes + 255u) & ~static_cast<std::size_t>(255u);
      return st == cudaSuccess;
    };
    bool uploaded =
        stage_copy(buffers.tx_type_cands, tx_tables.cands.data(),
                   tx_tables.cands.size() * sizeof(TxTypeCandJob)) &&
        stage_copy(buffers.tx_type_jobs, tx_tables.jobs.data(),
                   tx_tables.jobs.size() * sizeof(TxBlockJob)) &&
        stage_copy(buffers.tx_type_blocks, tx_tables.blocks.data(),
                   tx_tables.blocks.size() * sizeof(TxTypeBlockJob)) &&
        stage_copy(buffers.tx_type_map, tx_tables.map.data(),
                   tx_tables.map.size() * sizeof(std::int8_t)) &&
        stage_copy(buffers.tx_type_legal, tx_tables.legal.data(),
                   tx_tables.legal.size() * sizeof(std::uint8_t)) &&
        stage_copy(buffers.tx_type_bits, tx_tables.bits.data(),
                   tx_tables.bits.size() * sizeof(double));
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
    if (uploaded && ist_active) {
      uploaded =
          stage_copy(buffers.ist_job_table, ist_tables.jobs.data(),
                     ist_tables.jobs.size() * sizeof(TxSecondaryJob)) &&
          stage_copy(buffers.ist_block_table, ist_tables.block_jobs.data(),
                     ist_tables.block_jobs.size() * sizeof(TxBlockJob)) &&
          stage_copy(buffers.ist_bit_table, ist_tables.bits.data(),
                     ist_tables.bits.size() * sizeof(double)) &&
          stage_copy(buffers.ist_set_count, ist_tables.set_count.data(),
                     ist_tables.set_count.size() * sizeof(std::uint8_t)) &&
          stage_copy(buffers.ist_map, ist_tables.map.data(),
                     ist_tables.map.size() * sizeof(std::int8_t));
    }
#endif
    if (!uploaded ||
        cudaMemsetAsync(buffers.tx_types, 0,
                        static_cast<std::size_t>(blocks) * 3u *
                            sizeof(std::int32_t),
                        stream) != cudaSuccess) {
      error = "gpu: TX decision table upload failed";
      return GpuRunStatus::kError;
    }
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
    if (cctx_active &&
        cudaMemsetAsync(buffers.cctx_types, 0,
                        static_cast<std::size_t>(blocks) *
                            sizeof(std::int32_t),
                        stream) != cudaSuccess) {
      error = "gpu: CCTX decision reset failed";
      return GpuRunStatus::kError;
    }
#endif
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
    if (ist_active &&
        (cudaMemsetAsync(buffers.ist_secs, 0,
                         static_cast<std::size_t>(blocks) *
                             sizeof(std::int32_t),
                         stream) != cudaSuccess ||
         cudaMemsetAsync(buffers.ist_sets, 0,
                         static_cast<std::size_t>(blocks) *
                             sizeof(std::int32_t),
                         stream) != cudaSuccess)) {
      error = "gpu: IST decision reset failed";
      return GpuRunStatus::kError;
    }
#endif
    tx_stage_off = off;
  }
#else
  const bool ist_active = false;
#endif
#if !AV2ENC_STILL_GPU_HAVE_TCQ_CORE
  const bool cctx_active = false;
#endif

  // ---- availability counts (from the frame plan) -------------------------
  // The plan exports the exact superblock-local availability replay for its
  // frame SbSize (64 or 128), including ChromaMi ownership for extended and
  // sub-8 leaves.  The rect-only scheduler model cannot represent non-owning
  // chroma leaves, so it is not used for availability here.

  // ---- static descriptor arrays -----------------------------------------
  std::vector<srch::LumaSearchBlock> luma_blocks(static_cast<std::size_t>(blocks));
  std::vector<im::IntraJob> luma_jobs(static_cast<std::size_t>(blocks));
  std::vector<StillTxJob> tx_jobs(static_cast<std::size_t>(blocks) * 3u);
  std::vector<cm1::ReconWriteJob> recon_jobs(static_cast<std::size_t>(blocks) * 3u);
  std::vector<cm1::ReconBlockInfo> recon_infos(static_cast<std::size_t>(blocks));
  std::vector<cm1::StillChromaJob> chroma_jobs(static_cast<std::size_t>(blocks));
  std::vector<mrl::MrlJob> mrl_jobs(static_cast<std::size_t>(blocks));
  std::vector<dip::DipJob> dip_jobs(static_cast<std::size_t>(blocks));
  std::vector<ibp::IbpJob> ibp_jobs(static_cast<std::size_t>(blocks));

  for (int i = 0; i < blocks; ++i) {
    const BlockPlan& block = plan.blocks[static_cast<std::size_t>(i)];
    const int x0 = block.mi_col * 4;
    const int y0 = block.mi_row * 4;
    // An owning extended leaf codes U/V at its ChromaMi origin, which can
    // differ from the luma leaf origin.
    const int cx = (block.chroma_mi_size != 0 ? block.chroma_mi_col
                                              : block.mi_col) *
                   2;
    const int cy = (block.chroma_mi_size != 0 ? block.chroma_mi_row
                                              : block.mi_row) *
                   2;

    srch::LumaSearchBlock& search = luma_blocks[static_cast<std::size_t>(i)];
    search.recon_plane = buffers.recon.y;
    search.recon_stride = static_cast<int>(plan.y_stride);
    search.x0 = x0;
    search.y0 = y0;
    search.width = block.luma_w;
    search.height = block.luma_h;
    search.max_x = coded_w - 1;
    search.max_y = coded_h - 1;
    search.have_left = x0 > 0 ? 1 : 0;
    search.have_above = y0 > 0 ? 1 : 0;
    search.num4_above_right = block.n4_above_right;
    search.num4_below_left = block.n4_below_left;
    search.source_plane = buffers.src.y;
    search.source_stride = static_cast<int>(picture.width);
    search.source_max_x = static_cast<int>(picture.width) - 1;
    search.source_max_y = static_cast<int>(picture.height) - 1;

    im::IntraJob& pred = luma_jobs[static_cast<std::size_t>(i)];
    pred.block = im::IntraBlock{};
    pred.block.recon_plane = buffers.recon.y;
    pred.block.recon_stride = static_cast<int>(plan.y_stride);
    pred.block.x0 = x0;
    pred.block.y0 = y0;
    pred.block.width = block.luma_w;
    pred.block.height = block.luma_h;
    pred.block.max_x = coded_w - 1;
    pred.block.max_y = coded_h - 1;
    pred.block.have_left = search.have_left;
    pred.block.have_above = search.have_above;
    pred.block.num4_above_right = search.num4_above_right;
    pred.block.num4_below_left = search.num4_below_left;
    pred.block.plane = 0;
    pred.block.mode = im::kDcPred;
    pred.block.angle_delta = 0;
    pred.block.enable_edge_filter = options.enable_edge_filter ? 1 : 0;
    pred.pred = buffers.pred_packed[0] + block.pred_offset_y;
    pred.residual = nullptr;

    const auto init_tx = [&](StillTxJob& job, int coded_width, int coded_height,
                             int src_stride, int sx, int sy, int src_w,
                             int src_h, int pred_off, int residual_off,
                             int levels_off, int eob_off) -> bool {
      if (!still_tx_init_job(&job, coded_width, coded_height, TxType::kDctDct)) {
        error = "unsupported: TX size " + std::to_string(coded_width) + "x" +
                std::to_string(coded_height);
        return false;
      }
      job.input = static_cast<std::uint8_t>(StillTxInput::kSourcePred);
      job.src_packed = 0;
      job.src_x = static_cast<std::int16_t>(sx);
      job.src_y = static_cast<std::int16_t>(sy);
      job.src_width = static_cast<std::int16_t>(src_w);
      job.src_height = static_cast<std::int16_t>(src_h);
      job.src_offset = 0;
      job.src_stride = static_cast<std::int32_t>(src_stride);
      job.pred_offset = pred_off;
      job.residual_out_offset = residual_off;
      job.levels_offset = levels_off;
      job.eob_offset = eob_off;
      job.residual_adj_offset = levels_off;
      job.coeff_offset = levels_off;
      job.dequant_offset = levels_off;
      return true;
    };

    if (block.luma_w != 0 && block.luma_h != 0) {
      if (!init_tx(tx_jobs[static_cast<std::size_t>(i) * 3u], block.luma_w,
                   block.luma_h, static_cast<int>(picture.width), x0, y0,
                   static_cast<int>(picture.width),
                   static_cast<int>(picture.height),
                   static_cast<int>(block.pred_offset_y),
                   static_cast<int>(block.residual_offset_y),
                   static_cast<int>(block.coeff_offset_y), i * 3)) {
        return GpuRunStatus::kUnsupported;
      }
    }
    if (block.has_chroma != 0) {
      if (!init_tx(tx_jobs[static_cast<std::size_t>(i) * 3u + 1u],
                   block.chroma_w, block.chroma_h, chroma_vis_w, cx, cy,
                   chroma_vis_w, chroma_vis_h,
                   static_cast<int>(block.pred_offset_u),
                   static_cast<int>(block.residual_offset_u),
                   static_cast<int>(block.coeff_offset_u), i * 3 + 1)) {
        return GpuRunStatus::kUnsupported;
      }
      if (!init_tx(tx_jobs[static_cast<std::size_t>(i) * 3u + 2u],
                   block.chroma_w, block.chroma_h, chroma_vis_w, cx, cy,
                   chroma_vis_w, chroma_vis_h,
                   static_cast<int>(block.pred_offset_v),
                   static_cast<int>(block.residual_offset_v),
                   static_cast<int>(block.coeff_offset_v), i * 3 + 2)) {
        return GpuRunStatus::kUnsupported;
      }
    }
    if (cpu_plan != nullptr && cpu_plan->delta_q_enabled) {
      for (int plane = 0; plane < 3; ++plane) {
        StillTxJob& job = tx_jobs[static_cast<std::size_t>(i) * 3u +
                                  static_cast<std::size_t>(plane)];
        job.use_deltaq = 1;
        job.qindex = cpu_plan->delta_q_index;
        job.delta_dc = cpu_plan->delta_q_dc[plane];
        job.delta_ac = cpu_plan->delta_q_ac[plane];
        job.plane = plane;
      }
    }

    cm1::ReconWriteJob& write_y = recon_jobs[static_cast<std::size_t>(i) * 3u];
    write_y.pred = buffers.pred_packed[0] + block.pred_offset_y;
    write_y.residual = buffers.residual_packed[0] + block.residual_offset_y;
    write_y.plane = buffers.recon.y;
    write_y.stride = static_cast<int>(plan.y_stride);
    write_y.height = static_cast<int>(plan.padded_height);
    write_y.x0 = x0;
    write_y.y0 = y0;
    write_y.w = block.luma_w;
    write_y.h = block.luma_h;

    cm1::ReconWriteJob& write_u =
        recon_jobs[static_cast<std::size_t>(i) * 3u + 1u];
    write_u.pred = buffers.pred_packed[1] + block.pred_offset_u;
    write_u.residual = buffers.residual_packed[1] + block.residual_offset_u;
    write_u.plane = buffers.recon.u;
    write_u.stride = static_cast<int>(plan.u_stride);
    write_u.height = static_cast<int>(plan.padded_height / 2u);
    write_u.x0 = cx;
    write_u.y0 = cy;
    write_u.w = block.chroma_w;
    write_u.h = block.chroma_h;
    cm1::ReconWriteJob& write_v =
        recon_jobs[static_cast<std::size_t>(i) * 3u + 2u];
    write_v = write_u;
    write_v.pred = buffers.pred_packed[2] + block.pred_offset_v;
    write_v.residual = buffers.residual_packed[2] + block.residual_offset_v;
    write_v.plane = buffers.recon.v;

    cm1::ReconBlockInfo& info = recon_infos[static_cast<std::size_t>(i)];
    info.mi_row = block.mi_row;
    info.mi_col = block.mi_col;
    info.w4 = block.luma_w / 4;
    info.h4 = block.luma_h / 4;
    info.chroma_w4 = block.has_chroma != 0 ? block.chroma_w / 4 : 0;
    info.chroma_h4 = block.has_chroma != 0 ? block.chroma_h / 4 : 0;
    info.chroma_mi_row = block.chroma_mi_size != 0
                             ? static_cast<int>(block.chroma_mi_row)
                             : static_cast<int>(block.mi_row);
    info.chroma_mi_col = block.chroma_mi_size != 0
                             ? static_cast<int>(block.chroma_mi_col)
                             : static_cast<int>(block.mi_col);
    info.chroma_mi_w4 = block.chroma_mi_w4 != 0
                            ? static_cast<int>(block.chroma_mi_w4)
                            : block.luma_w / 4;
    info.chroma_mi_h4 = block.chroma_mi_h4 != 0
                            ? static_cast<int>(block.chroma_mi_h4)
                            : block.luma_h / 4;
    info.joint = 0;
    info.uv_smooth = 0;
    info.qindex = options.base_q_idx;
    const auto luma_params = block.luma_w != 0
                                 ? txv::make_tx_params(block.luma_w,
                                                       block.luma_h,
                                                       txv::TxType::kDctDct)
                                 : std::optional<txv::TxParams>{};
    const auto chroma_params =
        txv::make_tx_params(block.chroma_w, block.chroma_h,
                            txv::TxType::kDctDct);
    info.tx_size_luma =
        luma_params.has_value()
            ? static_cast<std::uint8_t>(luma_params->tx_size)
            : static_cast<std::uint8_t>(0);
    info.tx_size_chroma =
        chroma_params.has_value()
            ? static_cast<std::uint8_t>(chroma_params->tx_size)
            : static_cast<std::uint8_t>(0);
    info.skip = 0;
    info.region_intra = 1;
    info.lossless = 0;

    cm1::StillChromaJob& chroma = chroma_jobs[static_cast<std::size_t>(i)];
    chroma.luma = buffers.recon.y;
    chroma.luma_stride = static_cast<int>(plan.y_stride);
    chroma.luma_max_x = coded_w - 1;
    chroma.luma_max_y = coded_h - 1;
    chroma.sb_top_y =
        ((block.chroma_mi_size != 0 ? block.chroma_mi_row * 4 : y0) /
         plan.sb_size) *
        plan.sb_size;
    chroma.chroma_u = buffers.recon.u;
    chroma.chroma_v = buffers.recon.v;
    chroma.chroma_stride = static_cast<int>(plan.u_stride);
    chroma.pred_u = buffers.pred_packed[1] + block.pred_offset_u;
    chroma.pred_v = buffers.pred_packed[2] + block.pred_offset_v;
    chroma.pred_stride = block.chroma_w;
    chroma.src_u = buffers.src.u;
    chroma.src_v = buffers.src.v;
    chroma.src_stride_u = chroma_vis_w;
    chroma.src_stride_v = chroma_vis_w;
    chroma.src_max_x = chroma_vis_w - 1;
    chroma.src_max_y = chroma_vis_h - 1;
    chroma.num4_above_right = block.chroma_n4_above_right;
    chroma.num4_below_left = block.chroma_n4_below_left;
    chroma.x = cx;
    chroma.y = cy;
    chroma.w = block.chroma_w;
    chroma.h = block.chroma_h;
    chroma.coded_width = chroma_coded_w;
    chroma.coded_height = chroma_coded_h;

    mrl::MrlJob& mrl_job = mrl_jobs[static_cast<std::size_t>(i)];
    mrl_job.block.recon = buffers.recon.y;
    mrl_job.block.recon_stride = static_cast<int>(plan.y_stride);
    mrl_job.block.x = x0;
    mrl_job.block.y = y0;
    mrl_job.block.w = block.luma_w;
    mrl_job.block.h = block.luma_h;
    mrl_job.block.max_x = coded_w - 1;
    mrl_job.block.max_y = coded_h - 1;
    mrl_job.block.have_left = search.have_left;
    mrl_job.block.have_above = search.have_above;
    mrl_job.block.num4_above_right = search.num4_above_right;
    mrl_job.block.num4_below_left = search.num4_below_left;
    mrl_job.block.plane = 0;
    mrl_job.block.sb_height = plan.sb_size;
    mrl_job.pred = buffers.mrl_pred;

    dip::DipJob& dip_job = dip_jobs[static_cast<std::size_t>(i)];
    dip_job.block.recon = buffers.recon.y;
    dip_job.block.recon_stride = static_cast<int>(plan.y_stride);
    dip_job.block.x = x0;
    dip_job.block.y = y0;
    dip_job.block.w = block.luma_w;
    dip_job.block.h = block.luma_h;
    dip_job.block.max_x = coded_w - 1;
    dip_job.block.max_y = coded_h - 1;
    dip_job.block.have_left = search.have_left;
    dip_job.block.have_above = search.have_above;
    dip_job.block.num4_above_right = search.num4_above_right;
    dip_job.block.num4_below_left = search.num4_below_left;
    dip_job.pred = buffers.dip_pred;

    ibp::IbpJob& ibp_job = ibp_jobs[static_cast<std::size_t>(i)];
    ibp_job.block.intra.recon_plane = buffers.recon.y;
    ibp_job.block.intra.recon_stride = static_cast<int>(plan.y_stride);
    ibp_job.block.intra.x0 = x0;
    ibp_job.block.intra.y0 = y0;
    ibp_job.block.intra.width = block.luma_w;
    ibp_job.block.intra.height = block.luma_h;
    ibp_job.block.intra.max_x = coded_w - 1;
    ibp_job.block.intra.max_y = coded_h - 1;
    ibp_job.block.intra.have_left = search.have_left;
    ibp_job.block.intra.have_above = search.have_above;
    ibp_job.block.intra.num4_above_right = search.num4_above_right;
    ibp_job.block.intra.num4_below_left = search.num4_below_left;
    ibp_job.block.intra.plane = 0;
    ibp_job.block.intra.mode = im::kDcPred;
    ibp_job.block.enable_ibp = 1;
    ibp_job.block.uv_is_cfl = 0;
    ibp_job.pred = buffers.pred_packed[0] + block.pred_offset_y;
  }

  // ---- QM q2 matrices ----------------------------------------------------
  std::vector<std::int32_t> q2_host;
  std::size_t q2_upload_bytes = 0;
  if (cpu_plan != nullptr && cpu_plan->qm_enabled) {
    q2_host.assign(q2_elems(plan), 0);
    std::size_t q2_offset = 0;
    for (int i = 0; i < blocks; ++i) {
      for (int plane = 0; plane < 3; ++plane) {
        StillTxJob& job = tx_jobs[static_cast<std::size_t>(i) * 3u +
                                  static_cast<std::size_t>(plane)];
        const TxParams& p = job.params;
        qm::QmDequantParams qp;
        qp.state = &cpu_plan->qm_state;
        qp.tx_size = p.tx_size;
        qp.plane = plane;
        qp.tx_type = p.tx_type;
        qp.segment_id = 0;
        qp.dc_quant = av2enc::txv::kQuantStep;
        qp.ac_quant = av2enc::txv::kQuantStep;
        qp.bit_depth = 8;
        const std::size_t n =
            static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h);
        if (q2_offset + n > q2_host.size() ||
            !qm::build_q2_matrix(qp, q2_host.data() + q2_offset)) {
          error = "gpu: QM q2 matrix build failed";
          return GpuRunStatus::kUnsupported;
        }
        job.use_deltaq = 1;  // shared device quantizer
        job.qindex = options.base_q_idx;
        job.delta_dc = 0;
        job.delta_ac = 0;
        job.plane = plane;
        job.q2_offset = static_cast<std::int32_t>(q2_offset);
        q2_offset += n;
      }
    }
    q2_upload_bytes = q2_host.size() * sizeof(std::int32_t);
  }

  // ---- upload ------------------------------------------------------------
  const std::size_t src_y_bytes =
      static_cast<std::size_t>(picture.width) * picture.height;
  const std::size_t src_c_bytes =
      static_cast<std::size_t>(chroma_vis_w) * chroma_vis_h;
  const std::size_t luma_blocks_bytes =
      luma_blocks.size() * sizeof(srch::LumaSearchBlock);
  const std::size_t luma_jobs_bytes = luma_jobs.size() * sizeof(im::IntraJob);
  const std::size_t chroma_jobs_bytes =
      chroma_jobs.size() * sizeof(cm1::StillChromaJob);
  const std::size_t tx_jobs_bytes = tx_jobs.size() * sizeof(StillTxJob);
  const std::size_t recon_jobs_bytes =
      recon_jobs.size() * sizeof(cm1::ReconWriteJob);
  const std::size_t recon_infos_bytes =
      recon_infos.size() * sizeof(cm1::ReconBlockInfo);
  const std::size_t mrl_jobs_bytes = mrl_jobs.size() * sizeof(mrl::MrlJob);
  const std::size_t dip_jobs_bytes = dip_jobs.size() * sizeof(dip::DipJob);
  const std::size_t ibp_jobs_bytes = ibp_jobs.size() * sizeof(ibp::IbpJob);
  // The per-SB segment kernel reads the block origins on the device.
  const std::size_t plan_blocks_bytes =
      options.per_segment_q
          ? static_cast<std::size_t>(blocks) * sizeof(BlockPlan)
          : 0;
  // §7.19 CCSO metadata: per-plane unit-enable grid + decoded offset table.
  // The decoded tables come from the CPU reference's parameters so the GPU
  // kernel sees exactly the values the bitstream signals.
  std::size_t ccso_blk_bytes[3] = {0, 0, 0};
  std::size_t ccso_off_bytes[3] = {0, 0, 0};
  std::int16_t ccso_offset_tables[3][576] = {};
  const std::size_t cdef_idx_bytes =
      cdef_active ? cpu_plan->cdef_idx.size() : 0;
  const std::size_t lr_tx_skip_bytes =
      pcw_active && cpu_plan != nullptr ? cpu_plan->lr_tx_skip.size() : 0;
  // §7.20 luma Wiener NS unit descriptors (device copies).  Chroma descriptors
  // are consumed by the lr_chroma workspace through upload_units() below.
  std::size_t lr_type_bytes[3] = {0, 0, 0};
  std::size_t lr_coeff_bytes[3] = {0, 0, 0};
  if (lr_active) {
    const ::av2enc::still::StillLrPlane& plane = filter_config->lr.plane[0];
    if (plane.enabled) {
      lr_type_bytes[0] = plane.unit_type.size();
      lr_coeff_bytes[0] = plane.unit_coeffs.size() * sizeof(std::int16_t);
    }
  }
#if AV2ENC_STILL_GPU_HAVE_CCSO
  if (ccso_active) {
    for (int plane = 0; plane < 3; ++plane) {
      const ::av2enc::still::StillCcsoPlane& cfg = filter_config->ccso.plane[plane];
      if (!cfg.enabled) {
        continue;
      }
      ::av2enc::restoration::CcsoParams params;
      params.enabled = true;
      params.bo_only = cfg.bo_only;
      params.scale_idx = cfg.scale_idx;
      params.quant_idx = cfg.quant_idx;
      params.ext_filter = cfg.ext_filter;
      params.edge_clf = cfg.edge_clf;
      params.max_band_log2 = cfg.max_band_log2;
      const int count = ::av2enc::restoration::ccso_offset_count(params);
      if (count <= 0 || count > 576 ||
          !::av2enc::restoration::ccso_set_offsets(params, cfg.offset_idx,
                                                   static_cast<std::size_t>(
                                                       count))) {
        error = "gpu: CCSO offset table is not signallable";
        return GpuRunStatus::kError;
      }
      std::memcpy(ccso_offset_tables[plane], params.offsets,
                  sizeof(ccso_offset_tables[plane]));
      ccso_off_bytes[plane] = sizeof(ccso_offset_tables[plane]);
      ccso_blk_bytes[plane] =
          cpu_plan->ccso_blks[plane].empty()
              ? 0
              : cpu_plan->ccso_blks[plane].size() * sizeof(std::uint8_t);
    }
  }
#endif
  const std::size_t upload_total = src_y_bytes + src_c_bytes * 2u +
                                   luma_blocks_bytes + luma_jobs_bytes +
                                   chroma_jobs_bytes +
                                   tx_jobs_bytes + recon_jobs_bytes +
                                   recon_infos_bytes + mrl_jobs_bytes +
                                   dip_jobs_bytes + ibp_jobs_bytes +
                                   plan_blocks_bytes +
                                   (screen_plan && options.palette
                                        ? static_cast<std::size_t>(blocks) * 8u
                                        : 0u) +
                                   ccso_blk_bytes[0] +
                                   ccso_blk_bytes[1] + ccso_blk_bytes[2] +
                                   ccso_off_bytes[0] + ccso_off_bytes[1] +
                                   ccso_off_bytes[2] + cdef_idx_bytes +
                                   lr_type_bytes[0] + lr_coeff_bytes[0] +
                                   lr_tx_skip_bytes + q2_upload_bytes;
  auto* upload = static_cast<std::uint8_t*>(gpu.upload.data());
  if (upload == nullptr || gpu.upload.capacity() < upload_total) {
    error = "gpu: pinned upload staging is too small";
    return GpuRunStatus::kError;
  }
  std::size_t up = 0;
  std::memcpy(upload + up, picture.y.data(), src_y_bytes);
  up += src_y_bytes;
  std::memcpy(upload + up, picture.u.data(), src_c_bytes);
  up += src_c_bytes;
  std::memcpy(upload + up, picture.v.data(), src_c_bytes);
  up += src_c_bytes;
  std::memcpy(upload + up, luma_blocks.data(), luma_blocks_bytes);
  const std::uint8_t* host_luma_blocks = upload + up;
  up += luma_blocks_bytes;
  std::memcpy(upload + up, luma_jobs.data(), luma_jobs_bytes);
  const std::uint8_t* host_luma_jobs = upload + up;
  up += luma_jobs_bytes;
  std::memcpy(upload + up, chroma_jobs.data(), chroma_jobs_bytes);
  const std::uint8_t* host_chroma_jobs = upload + up;
  up += chroma_jobs_bytes;
  std::memcpy(upload + up, tx_jobs.data(), tx_jobs_bytes);
  const std::uint8_t* host_tx_jobs = upload + up;
  up += tx_jobs_bytes;
  std::memcpy(upload + up, recon_jobs.data(), recon_jobs_bytes);
  const std::uint8_t* host_recon_jobs = upload + up;
  up += recon_jobs_bytes;
  std::memcpy(upload + up, recon_infos.data(), recon_infos_bytes);
  const std::uint8_t* host_recon_infos = upload + up;
  up += recon_infos_bytes;
  const std::uint8_t* host_plan_blocks = nullptr;
  if (plan_blocks_bytes != 0) {
    std::memcpy(upload + up, plan.blocks.data(), plan_blocks_bytes);
    host_plan_blocks = upload + up;
    up += plan_blocks_bytes;
  }
  std::memcpy(upload + up, mrl_jobs.data(), mrl_jobs_bytes);
  const std::uint8_t* host_mrl_jobs = upload + up;
  up += mrl_jobs_bytes;
  std::memcpy(upload + up, dip_jobs.data(), dip_jobs_bytes);
  const std::uint8_t* host_dip_jobs = upload + up;
  up += dip_jobs_bytes;
  std::memcpy(upload + up, ibp_jobs.data(), ibp_jobs_bytes);
  const std::uint8_t* host_ibp_jobs = upload + up;
  up += ibp_jobs_bytes;
  const std::uint8_t* host_palette_colors = nullptr;
  if (screen_plan && options.palette) {
    std::uint8_t* colors = upload + up;
    std::memset(colors, 0, static_cast<std::size_t>(blocks) * 8u);
    for (int i = 0; i < blocks; ++i) {
      const std::uint32_t index =
          cpu_plan->palette_index_by_leaf[static_cast<std::size_t>(i)];
      if (index == ::av2enc::still::StillResult::kNoPaletteLeaf) continue;
      const ::av2enc::palette::PaletteResult& result =
          cpu_plan->palette_results[index];
      for (int k = 0; k < 8; ++k) {
        colors[static_cast<std::size_t>(i) * 8u + static_cast<std::size_t>(k)] =
            k < result.size ? result.colors[static_cast<std::size_t>(k)] : 0;
      }
    }
    host_palette_colors = colors;
    up += static_cast<std::size_t>(blocks) * 8u;
  }
  const std::uint8_t* host_ccso_blks[3] = {nullptr, nullptr, nullptr};
  const std::int16_t* host_ccso_offsets[3] = {nullptr, nullptr, nullptr};
  const std::int8_t* host_cdef_idx = nullptr;
  const std::uint8_t* host_lr_type = nullptr;
  const std::int16_t* host_lr_coeff = nullptr;
  if (lr_type_bytes[0] != 0) {
    std::memcpy(upload + up, filter_config->lr.plane[0].unit_type.data(),
                lr_type_bytes[0]);
    host_lr_type = upload + up;
    up += lr_type_bytes[0];
  }
  if (lr_coeff_bytes[0] != 0) {
    std::memcpy(upload + up, filter_config->lr.plane[0].unit_coeffs.data(),
                lr_coeff_bytes[0]);
    host_lr_coeff =
        reinterpret_cast<const std::int16_t*>(upload + up);
    up += lr_coeff_bytes[0];
  }
  const std::uint8_t* host_lr_tx_skip = nullptr;
  if (lr_tx_skip_bytes != 0) {
    std::memcpy(upload + up, cpu_plan->lr_tx_skip.data(), lr_tx_skip_bytes);
    host_lr_tx_skip = upload + up;
    up += lr_tx_skip_bytes;
  }
  const std::int32_t* host_q2 = nullptr;
  if (q2_upload_bytes != 0) {
    std::memcpy(upload + up, q2_host.data(), q2_upload_bytes);
    host_q2 = reinterpret_cast<const std::int32_t*>(upload + up);
    up += q2_upload_bytes;
  }
  if (cdef_idx_bytes != 0) {
    std::memcpy(upload + up, cpu_plan->cdef_idx.data(), cdef_idx_bytes);
    host_cdef_idx = reinterpret_cast<const std::int8_t*>(upload + up);
    up += cdef_idx_bytes;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (ccso_blk_bytes[plane] != 0) {
      std::memcpy(upload + up, cpu_plan->ccso_blks[plane].data(),
                  ccso_blk_bytes[plane]);
      host_ccso_blks[plane] = upload + up;
      up += ccso_blk_bytes[plane];
    }
    if (ccso_off_bytes[plane] != 0) {
      std::memcpy(upload + up, ccso_offset_tables[plane],
                  ccso_off_bytes[plane]);
      host_ccso_offsets[plane] = reinterpret_cast<const std::int16_t*>(
          upload + up);
      up += ccso_off_bytes[plane];
    }
  }

  const auto enqueue_upload = [&](void* dst, const void* src,
                                  std::size_t bytes) -> bool {
    return is_ok(upload_async(stream, dst, src, bytes));
  };
  if (!enqueue_upload(buffers.src.y, upload, src_y_bytes) ||
      !enqueue_upload(buffers.src.u, upload + src_y_bytes, src_c_bytes) ||
      !enqueue_upload(buffers.src.v, upload + src_y_bytes + src_c_bytes,
                      src_c_bytes) ||
      !enqueue_upload(buffers.luma_blocks, host_luma_blocks,
                      luma_blocks_bytes) ||
      !enqueue_upload(buffers.luma_jobs, host_luma_jobs, luma_jobs_bytes) ||
      !enqueue_upload(buffers.chroma_jobs, host_chroma_jobs,
                      chroma_jobs_bytes) ||
      !enqueue_upload(buffers.tx_jobs, host_tx_jobs, tx_jobs_bytes) ||
      !enqueue_upload(buffers.recon_jobs, host_recon_jobs, recon_jobs_bytes) ||
      !enqueue_upload(buffers.recon_infos, host_recon_infos,
                      recon_infos_bytes) ||
      (plan_blocks_bytes != 0 &&
       !enqueue_upload(buffers.blocks, host_plan_blocks,
                       plan_blocks_bytes)) ||
      !enqueue_upload(buffers.mrl_jobs, host_mrl_jobs, mrl_jobs_bytes) ||
      !enqueue_upload(buffers.dip_jobs, host_dip_jobs, dip_jobs_bytes) ||
      !enqueue_upload(buffers.ibp_jobs, host_ibp_jobs, ibp_jobs_bytes) ||
      (host_palette_colors != nullptr &&
       !enqueue_upload(buffers.palette_colors, host_palette_colors,
                       static_cast<std::size_t>(blocks) * 8u)) ||
      cudaMemsetAsync(buffers.tx_errors, 0, sizeof(int), stream) !=
          cudaSuccess ||
      cudaMemsetAsync(buffers.mi_luma_mode, 0, buffers.mi_luma_bytes,
                      stream) != cudaSuccess ||
      cudaMemsetAsync(buffers.mi_luma_angle, 0, buffers.mi_luma_bytes,
                      stream) != cudaSuccess ||
      cudaMemsetAsync(buffers.luma_results, 0, buffers.luma_results_bytes,
                      stream) != cudaSuccess ||
      cudaMemsetAsync(buffers.eob, 0, buffers.eob_bytes, stream) !=
          cudaSuccess) {
    error = "gpu: upload failed";
    return GpuRunStatus::kError;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (host_ccso_blks[plane] != nullptr &&
        !enqueue_upload(buffers.ccso_blks[plane], host_ccso_blks[plane],
                        ccso_blk_bytes[plane])) {
      error = "gpu: CCSO grid upload failed";
      return GpuRunStatus::kError;
    }
    if (host_ccso_offsets[plane] != nullptr &&
        !enqueue_upload(buffers.ccso_offsets[plane], host_ccso_offsets[plane],
                        ccso_off_bytes[plane])) {
      error = "gpu: CCSO offset upload failed";
      return GpuRunStatus::kError;
    }
  }
  if (cdef_active) {
    // M1 is intra/QP90: no skipped and no lossless MI cells, so the metadata
    // grids are all zero.  The index grid is uploaded from the CPU plan.
    if (!enqueue_upload(buffers.cdef_idx, host_cdef_idx,
                        cdef_idx_bytes) ||
        cudaMemsetAsync(buffers.cdef_skip, 0, buffers.cdef_mi_bytes, stream) !=
            cudaSuccess ||
        cudaMemsetAsync(buffers.cdef_lossless, 0, buffers.cdef_mi_bytes,
                        stream) != cudaSuccess ||
        cudaMemsetAsync(buffers.cdef_lossless_chroma, 0,
                        buffers.cdef_chroma_mi_bytes, stream) != cudaSuccess) {
      error = "gpu: CDEF metadata upload failed";
      return GpuRunStatus::kError;
    }
  }
  if (lr_type_bytes[0] != 0 &&
      !enqueue_upload(buffers.lr_unit_type[0], host_lr_type,
                      lr_type_bytes[0])) {
    error = "gpu: loop-restoration unit type upload failed";
    return GpuRunStatus::kError;
  }
  if (lr_coeff_bytes[0] != 0 &&
      !enqueue_upload(buffers.lr_unit_coeffs[0], host_lr_coeff,
                      lr_coeff_bytes[0])) {
    error = "gpu: loop-restoration coefficient upload failed";
    return GpuRunStatus::kError;
  }
  if (lr_tx_skip_bytes != 0 &&
      !enqueue_upload(buffers.lr_tx_skip, host_lr_tx_skip,
                      lr_tx_skip_bytes)) {
    error = "gpu: PC-Wiener LrTxSkip upload failed";
    return GpuRunStatus::kError;
  }
  if (q2_upload_bytes != 0 &&
      !enqueue_upload(buffers.q2, host_q2, q2_upload_bytes)) {
    error = "gpu: QM q2 matrix upload failed";
    return GpuRunStatus::kError;
  }
  if (cudaMemsetAsync(buffers.intra_choices, 0,
                      static_cast<std::size_t>(blocks) *
                          sizeof(iext::Choice),
                      stream) != cudaSuccess) {
    error = "gpu: intra choice reset failed";
    return GpuRunStatus::kError;
  }

  // §5.4.9 / §5.18.7.1 real per-SB segmentation: derive the segment map and
  // per-SB qindex on the device, then patch the TX/recon descriptors.  The
  // variance rule and integer arithmetic match the CPU still planner.
  if (options.per_segment_q) {
    DeviceSegmentState seg_state;
    seg_state.base_q_idx = options.base_q_idx;
    seg_state.num_segments = 8;  // enable_ext_seg == 0
    seg_state.sb_px = plan.sb_size;
    seg_state.sb_cols = plan.sb_cols;
    seg_state.sb_rows = plan.sb_rows;
    seg_state.width = static_cast<int>(picture.width);
    seg_state.height = static_cast<int>(picture.height);
    for (int id = 0; id < 8; ++id) {
      seg_state.alt_q[id] = 0;
      seg_state.alt_q_active[id] = 0;
    }
    // Segment 1 carries the signalled SEG_LVL_ALT_Q; segment 0 is base_q_idx.
    seg_state.alt_q[1] = options.per_segment_q_delta;
    seg_state.alt_q_active[1] = 1;
    derive_segment_map_kernel<<<1, 256, 0, stream>>>(
        buffers.src.y, seg_state, buffers.sb_variance, buffers.sb_segment,
        buffers.sb_qindex, buffers.blocks, buffers.tx_jobs,
        buffers.recon_infos, blocks);
  }

  // ---- reconstruction state ---------------------------------------------
  const int chroma_height = static_cast<int>(plan.padded_height / 2u);
  cm1::ReconFrame recon_frame;
  recon_frame.plane[0] = cm1::ReconPlane{buffers.recon.y,
                                         static_cast<int>(plan.y_stride),
                                         static_cast<int>(plan.padded_height)};
  recon_frame.plane[1] = cm1::ReconPlane{buffers.recon.u,
                                         static_cast<int>(plan.u_stride),
                                         chroma_height};
  recon_frame.plane[2] = cm1::ReconPlane{buffers.recon.v,
                                         static_cast<int>(plan.v_stride),
                                         chroma_height};
  recon_frame.mi_cols = mi_cols;
  recon_frame.mi_rows = mi_rows;
  for (int p = 0; p < 3; ++p) {
    recon_frame.deblock_mi[p] = buffers.deblock_mi[p];
  }
  recon_frame.base_q_idx = options.base_q_idx;

  cm1::ReconPredState state;
  state.mi_cols = mi_cols;
  state.mi_rows = mi_rows;
  state.decoded[0] = buffers.decoded[0];
  state.decoded[1] = buffers.decoded[1];
  state.decoded[2] = buffers.decoded[2];
  state.decoded_cols[0] = mi_cols;
  state.decoded_rows[0] = mi_rows;
  state.decoded_cols[1] = mi_cols / 2;
  state.decoded_rows[1] = mi_rows / 2;
  state.decoded_cols[2] = mi_cols / 2;
  state.decoded_rows[2] = mi_rows / 2;
  state.intra_joint = buffers.intra_joint;
  state.uv_smooth = buffers.uv_smooth;

  if (cm1::recon_init_frame(recon_frame, stream) != cudaSuccess ||
      cm1::recon_init_state(state, stream) != cudaSuccess) {
    error = "gpu: reconstruction state init failed";
    return GpuRunStatus::kError;
  }

  // ---- stage configuration ---------------------------------------------
  srch::LumaSearchGates gates;
  if (options.luma_dc_only || options.force_dc ||
      std::getenv("AV2ENC_STILL_LUMA_DC") != nullptr) {
    // DIP (and palette on the CPU path) need the real DC score even in the
    // DC-only fast path, so the search must predict DC instead of returning
    // sse = 0.
    gates.flags |= options.enable_dip ? srch::kGateDcOnlyPredict
                                      : srch::kGateDcOnly;
  }
  if (std::getenv("AV2ENC_STILL_BASE_ONLY") != nullptr) {
    gates.flags |= srch::kGateBaseOnly;
  }
  if (std::getenv("AV2ENC_STILL_NO_DIR") != nullptr) {
    gates.flags |= srch::kGateNoDirectional;
  }
  gates.enable_edge_filter =
      (options.enable_edge_filter &&
       std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr)
          ? 1
          : 0;

  const srch::LumaSearchLayout search_layout = srch::luma_search_layout(
      kSearchMaxBlocks, kSearchMaxBlockArea, kSearchCandidatesPerChunk);
  const srch::LumaSearchScratch search_scratch =
      srch::luma_search_make_scratch(buffers.search_scratch, search_layout);

  cm1::StillChromaOptions chroma_options;
  chroma_options.enable_cfl =
      (options.enable_cfl && !options.force_dc &&
       std::getenv("AV2ENC_STILL_UV_DC") == nullptr)
          ? 1
          : 0;
  chroma_options.enable_uv_full = options.uv_full ? 1 : 0;
  chroma_options.ds_filter = options.cfl_ds_filter;
  chroma_options.enable_ibp = options.enable_ibp ? 1 : 0;
  chroma_options.cfl_explicit = options.cfl_explicit ? 1 : 0;
  chroma_options.enable_mhccp = options.mhccp ? 1 : 0;

  StillTxBuffers tx_buffers;
  tx_buffers.src = buffers.src.y;
  tx_buffers.pred = buffers.pred_packed[0];
  tx_buffers.levels = buffers.coeff.y;
  tx_buffers.residual_out = buffers.residual_packed[0];
  tx_buffers.eob = buffers.eob;
  tx_buffers.q2 = buffers.q2;
  tx_buffers.error_count = buffers.tx_errors;

  // ---- ordered block loop ----------------------------------------------
  int last_sb_row = -1;
  int last_sb_col = -1;
  for (int i = 0; i < blocks; ++i) {
    const BlockPlan& block = plan.blocks[static_cast<std::size_t>(i)];
    const int x0 = block.mi_col * 4;
    const int y0 = block.mi_row * 4;
    const int sb_mi = plan.sb_size / 4;
    const int sb_row = (block.mi_row / sb_mi) * sb_mi;
    const int sb_col = (block.mi_col / sb_mi) * sb_mi;
    if (sb_row != last_sb_row || sb_col != last_sb_col) {
      if (cm1::recon_begin_sb(state, sb_row, sb_col, stream, sb_mi) !=
          cudaSuccess) {
        error = "gpu: recon_begin_sb failed";
        return GpuRunStatus::kError;
      }
      last_sb_row = sb_row;
      last_sb_col = sb_col;
    }

    const bool has_luma = block.luma_w != 0 && block.luma_h != 0;
    const bool leaf_palette =
        screen_plan && options.palette && has_luma &&
        cpu_plan->palette_index_by_leaf[static_cast<std::size_t>(i)] !=
            ::av2enc::still::StillResult::kNoPaletteLeaf;
    const bool leaf_ibc =
        screen_plan && options.ibc && has_luma &&
        cpu_plan->ibc_by_leaf[static_cast<std::size_t>(i)].used != 0;
    if (has_luma) {
    if (leaf_palette || leaf_ibc) {
      // §5.20.8 / §5.20.5.3-.4: the CPU-planned decision forces DC luma and
      // no residual; the GPU applies the palette map/prediction or block copy.
      const std::int64_t planned_sse =
          leaf_palette
              ? cpu_plan->palette_results[cpu_plan->palette_index_by_leaf
                                              [static_cast<std::size_t>(i)]]
                    .sse
              : 0;
      set_luma_result_kernel<<<1, 1, 0, stream>>>(
          buffers.luma_results, i, /*mode=*/0, /*angle=*/0, planned_sse);
      record_luma_mode_kernel<<<1, 1, 0, stream>>>(
          buffers.blocks, buffers.luma_results, buffers.mi_luma_mode,
          buffers.mi_luma_angle, mi_cols, mi_rows, i);
      const int luma_visible_w =
          std::min<int>(block.luma_w, coded_w - x0);
      const int luma_visible_h =
          std::min<int>(block.luma_h, coded_h - y0);
      if (leaf_palette) {
        const ::av2enc::palette::PaletteResult& pal =
            cpu_plan->palette_results
                [cpu_plan->palette_index_by_leaf[static_cast<std::size_t>(i)]];
        av2enc::cuda::PaletteDeviceBlock pb;
        pb.src = buffers.src.y + static_cast<std::size_t>(y0) * picture.width +
                 static_cast<std::size_t>(x0);
        pb.src_stride = static_cast<int>(picture.width);
        pb.block_w = block.luma_w;
        pb.block_h = block.luma_h;
        pb.visible_w = luma_visible_w;
        pb.visible_h = luma_visible_h;
        pb.colors = buffers.palette_colors + static_cast<std::size_t>(i) * 8u;
        pb.size = pal.size;
        pb.map = buffers.palette_map;
        pb.map_stride = block.luma_w;
        pb.dst = buffers.pred_packed[0] + block.pred_offset_y;
        pb.dst_stride = block.luma_w;
        pb.dither = false;
        if (pal.size < 2 || pal.size > 8 ||
            av2enc::cuda::palette_build_map(pb, stream) != cudaSuccess ||
            av2enc::cuda::palette_predict(pb, stream) != cudaSuccess) {
          error = "gpu: palette prediction failed";
          return GpuRunStatus::kError;
        }
      } else {
        const ::av2enc::still::StillResult::IbcPlan& ibc =
            cpu_plan->ibc_by_leaf[static_cast<std::size_t>(i)];
        av2enc::cuda::IbcDeviceFrame frame;
        frame.y = buffers.recon.y;
        frame.u = buffers.recon.u;
        frame.v = buffers.recon.v;
        frame.stride_y = static_cast<int>(plan.y_stride);
        frame.stride_u = static_cast<int>(plan.u_stride);
        frame.stride_v = static_cast<int>(plan.v_stride);
        frame.width = coded_w;
        frame.height = coded_h;
        frame.has_chroma = block.has_chroma != 0;
        frame.sub_x = 1;
        frame.sub_y = 1;
        const av2enc::ibc::BlockVector bv{ibc.bv_row, ibc.bv_col};
        if (av2enc::cuda::ibc_copy_block(
                frame, block.mi_row, block.mi_col, block.bsize, bv,
                buffers.pred_packed[0] + block.pred_offset_y, block.luma_w,
                nullptr, 0, nullptr, 0, luma_visible_w, luma_visible_h,
                stream) != cudaSuccess) {
          error = "gpu: IBC luma copy failed (bsize " +
                  std::to_string(block.bsize) + ", " +
                  std::to_string(block.luma_w) + "x" +
                  std::to_string(block.luma_h) + " at " +
                  std::to_string(x0) + "," + std::to_string(y0) + " bv " +
                  std::to_string(ibc.bv_row) + "," +
                  std::to_string(ibc.bv_col) + ")";
          return GpuRunStatus::kError;
        }
      }
      const std::size_t luma_pixels =
          static_cast<std::size_t>(block.luma_w) * block.luma_h;
      if (cudaMemsetAsync(buffers.residual_packed[0] + block.residual_offset_y,
                          0, luma_pixels * sizeof(std::int16_t),
                          stream) != cudaSuccess ||
          cm1::recon_write_blocks(
              buffers.recon_jobs + static_cast<std::size_t>(i) * 3u, 1,
              stream) != cudaSuccess) {
        error = "gpu: screen-content luma reconstruction failed";
        return GpuRunStatus::kError;
      }
    } else {
    prepare_search_block_kernel<<<1, 1, 0, stream>>>(
        buffers.luma_blocks, buffers.intra_joint, mi_cols, mi_rows, i);
    srch::LumaSearchDebug search_debug{};
    if (options.enable_mrl) {
      search_debug.d_candidate_sse =
          buffers.candidate_sse +
          static_cast<std::size_t>(i) * srch::kMaxLumaCandidates;
      search_debug.stride = srch::kMaxLumaCandidates;
    }
    if (options.enable_ibp) {
      if (iext::ibp_luma_search(
              buffers.luma_blocks + i, plan.sb_size, gates.flags,
              gates.enable_edge_filter, buffers.luma_results + i,
              options.enable_mrl
                  ? buffers.candidate_sse +
                        static_cast<std::size_t>(i) *
                            srch::kMaxLumaCandidates
                  : nullptr,
              stream) != cudaSuccess) {
        error = "gpu: IBP luma search failed";
        return GpuRunStatus::kError;
      }
    } else if (srch::luma_search(buffers.luma_blocks + i, 1, gates,
                                 search_scratch, buffers.luma_results + i,
                                 &search_debug, stream) != cudaSuccess) {
      error = "gpu: luma search failed";
      return GpuRunStatus::kError;
    }
    if (options.enable_mrl &&
        iext::mrl_search(
            buffers.luma_blocks + i, plan.sb_size,
            gates.enable_edge_filter,
            buffers.candidate_sse +
                static_cast<std::size_t>(i) * srch::kMaxLumaCandidates,
            gates.flags, buffers.luma_results + i, buffers.intra_choices + i,
            stream) != cudaSuccess) {
      error = "gpu: MRL search failed";
      return GpuRunStatus::kError;
    }
    if (options.enable_dip &&
        iext::dip_search(buffers.luma_blocks + i, plan.sb_size,
                         gates.enable_edge_filter, options.enable_dip ? 1 : 0,
                         std::getenv("AV2ENC_STILL_BASE_ONLY") != nullptr ? 1 : 0,
                         buffers.luma_results + i, buffers.intra_choices + i,
                         stream) != cudaSuccess) {
      error = "gpu: DIP search failed";
      return GpuRunStatus::kError;
    }
    if ((options.enable_mrl || options.enable_dip || options.enable_ibp) &&
        iext::prepare_jobs(buffers.mrl_jobs, buffers.dip_jobs,
                           buffers.ibp_jobs,
                           buffers.luma_blocks, buffers.luma_results,
                           buffers.intra_choices, i, gates.enable_edge_filter,
                           stream) != cudaSuccess) {
      error = "gpu: intra extra job preparation failed";
      return GpuRunStatus::kError;
    }
    prepare_prediction_kernel<<<1, 1, 0, stream>>>(
        buffers.luma_jobs, buffers.luma_results, buffers.luma_blocks, i);
    if (im::intra_predict_batch(buffers.luma_jobs + i, 1, stream) !=
        cudaSuccess) {
      error = "gpu: chosen-mode prediction failed";
      return GpuRunStatus::kError;
    }
    if (options.enable_ibp &&
        ibp::ibp_predict_batch(buffers.ibp_jobs + i, 1, stream) !=
            cudaSuccess) {
      error = "gpu: IBP prediction failed";
      return GpuRunStatus::kError;
    }
    if (options.enable_mrl &&
        mrl::mrl_predict_batch(buffers.mrl_jobs + i, 1, stream) !=
            cudaSuccess) {
      error = "gpu: MRL prediction failed";
      return GpuRunStatus::kError;
    }
    if (options.enable_dip &&
        dip::dip_predict_batch(buffers.dip_jobs + i, 1, stream) !=
            cudaSuccess) {
      error = "gpu: DIP prediction failed";
      return GpuRunStatus::kError;
    }
    if ((options.enable_mrl || options.enable_dip) &&
        iext::select_prediction(
            buffers.intra_choices + i,
            options.enable_mrl ? buffers.mrl_pred : nullptr,
            options.enable_dip ? buffers.dip_pred : nullptr,
            buffers.pred_packed[0] + block.pred_offset_y,
            block.luma_w * block.luma_h, stream) != cudaSuccess) {
      error = "gpu: intra extra prediction selection failed";
      return GpuRunStatus::kError;
    }
    record_luma_mode_kernel<<<1, 1, 0, stream>>>(
        buffers.blocks, buffers.luma_results, buffers.mi_luma_mode,
        buffers.mi_luma_angle, mi_cols, mi_rows, i);

#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
    if (tx_type_search) {
      const int size_idx =
          txv::tx_size_index(block.luma_w, block.luma_h);
      if (size_idx < 0 || size_idx >= 25) {
        error = "unsupported: TX size for the type search";
        return GpuRunStatus::kUnsupported;
      }
      if (tx_stage_off + 256u >
          gpu.tx_stage.capacity()) {
        error = "gpu: TX decision staging exhausted";
        return GpuRunStatus::kError;
      }
      // Probe the block's DCT adjusted residual; the candidate kernel reuses
      // it for every TX_TYPE.
      StillTxJob probe = tx_jobs[static_cast<std::size_t>(i) * 3u];
      probe.residual_adj_offset = 0;
      probe.coeff_offset = 0;
      probe.levels_offset = 0;
      probe.dequant_offset = 0;
      probe.residual_out_offset = 0;
      probe.eob_offset = 0;
      auto* stage = static_cast<std::uint8_t*>(gpu.tx_stage.data());
      std::memcpy(stage + tx_stage_off, &probe, sizeof(probe));
      const cudaError_t probe_copy = cudaMemcpyAsync(
          buffers.tx_probe_job, stage + tx_stage_off, sizeof(probe),
          cudaMemcpyHostToDevice, stream);
      tx_stage_off += 256u;
      if (probe_copy != cudaSuccess) {
        error = "gpu: TX-type probe upload failed";
        return GpuRunStatus::kError;
      }
      still_residual_coded_kernel<<<1, 256, 0, stream>>>(
          buffers.src.y, static_cast<int>(picture.width),
          static_cast<int>(picture.width) - 1,
          static_cast<int>(picture.height) - 1,
          buffers.pred_packed[0] + block.pred_offset_y, block.luma_w,
          block.luma_h, block.mi_col * 4, block.mi_row * 4,
          buffers.tx_residual_coded);
      StillTxBuffers probe_buffers;
      probe_buffers.src = buffers.src.y;
      probe_buffers.pred = buffers.pred_packed[0];
      probe_buffers.residual_adj = buffers.tx_residual_adj;
      probe_buffers.coeff = buffers.tx_cand_coeff;
      probe_buffers.levels = buffers.tx_cand_levels;
      probe_buffers.error_count = buffers.tx_errors;
      if (still_tx_batch(buffers.tx_probe_job, 1, probe_buffers, 0, stream) !=
              cudaSuccess ||
          cudaGetLastError() != cudaSuccess) {
        error = "gpu: TX-type probe failed";
        return GpuRunStatus::kError;
      }
      const std::size_t map_off =
          static_cast<std::size_t>(size_idx) * kTxTypeMapStride;
      const std::size_t mode_off =
          static_cast<std::size_t>(size_idx) * kTxTypeModeCount *
          kTxTypeSlots;
      tx_type_prepare_candidates_kernel<<<1, 32, 0, stream>>>(
          buffers.luma_results + i, buffers.intra_choices + i,
          buffers.tx_type_map + map_off, buffers.tx_type_legal + mode_off,
          buffers.tx_type_bits + mode_off,
          buffers.tx_type_cands +
              static_cast<std::size_t>(size_idx) * kTxTypeSlots,
          buffers.tx_cand_jobs, kTxTypeSlots);
      TxTypeCostModel cost;
      cost.coeff_bits = 1.0;
      cost.use_type_bits = true;
      cost.lambda = ::av2enc::rdo::lambda_sse_from_quant_step(
          1.0 / probe.params.quant_factor, options.tx_type_sse_scale);
      if (tx_type_choose_batch(
              buffers.tx_residual_coded, buffers.tx_residual_adj,
              buffers.tx_type_blocks + size_idx, 1, buffers.tx_cand_jobs,
              kTxTypeSlots,
              buffers.tx_type_jobs +
                  static_cast<std::size_t>(size_idx) * kTxTypeSlots,
              buffers.tx_cand_adj, buffers.tx_cand_coeff,
              buffers.tx_cand_levels, buffers.tx_cand_recon,
              buffers.tx_metrics, buffers.tx_decision, cost, stream) !=
          cudaSuccess) {
        error = "gpu: TX-type search failed";
        return GpuRunStatus::kError;
      }
      tx_type_apply_decision_kernel<<<1, 1, 0, stream>>>(
          buffers.tx_decision, buffers.tx_cand_jobs,
          buffers.tx_jobs + static_cast<std::size_t>(i) * 3u,
          buffers.tx_types, i * 3);
      if (cudaGetLastError() != cudaSuccess) {
        error = "gpu: TX-type decision apply failed";
        return GpuRunStatus::kError;
      }
    }
#endif

    tx_buffers.src = buffers.src.y;
    tx_buffers.pred = buffers.pred_packed[0];
    tx_buffers.levels = buffers.coeff.y;
    tx_buffers.residual_out = buffers.residual_packed[0];
    tx_buffers.eob = buffers.eob;
    if (still_tx_batch(buffers.tx_jobs + static_cast<std::size_t>(i) * 3u, 1,
                    tx_buffers, 0, stream) != cudaSuccess) {
      error = "gpu: luma TX failed";
      return GpuRunStatus::kError;
    }
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
    if (ist_active) {
      const int size_idx = txv::tx_size_index(block.luma_w, block.luma_h);
      if (size_idx < 0 || size_idx >= 25) {
        error = "unsupported: TX size for the IST search";
        return GpuRunStatus::kUnsupported;
      }
      const int adj_elems = static_cast<int>(block.tx_w) * block.tx_h;
      const int coded_elems = static_cast<int>(block.luma_w) * block.luma_h;
      const auto host_params = txv::make_tx_params(
          block.luma_w, block.luma_h, txv::TxType::kDctDct);
      if (!host_params.has_value()) {
        error = "unsupported: IST transform size";
        return GpuRunStatus::kUnsupported;
      }
      auto* primary_levels =
          buffers.coeff.y + block.coeff_offset_y;
      auto* primary_residual =
          buffers.residual_packed[0] + block.residual_offset_y;
      const std::size_t map_off =
          static_cast<std::size_t>(size_idx) * kTxTypeMapStride;
      ist_prepare_kernel<<<1, 32, 0, stream>>>(
          buffers.luma_results + i, buffers.intra_choices + i, size_idx,
          buffers.ist_map + map_off, buffers.ist_job_table,
          buffers.ist_block_table, buffers.ist_bit_table,
          buffers.tx_types + i * 3,
          buffers.ist_jobs,
          buffers.ist_block_jobs, buffers.ist_valid, buffers.ist_bits,
          buffers.ist_primary_type);
      ist_project_kernel<<<kIstSlots, 1, 0, stream>>>(
          primary_levels, buffers.ist_jobs, buffers.ist_levels);
      ist_eob_kernel<<<kIstSlots, 1, 0, stream>>>(
          buffers.ist_jobs, buffers.ist_levels, buffers.ist_eob);
      bool ist_ok = cudaGetLastError() == cudaSuccess;
      for (int j = 0; j < kIstSlots && ist_ok; ++j) {
        ist_ok = txv_dequantize(buffers.ist_levels + j * kTxTypeAdjElems,
                                buffers.ist_dequant + j * kTxTypeAdjElems,
                                *host_params, 1, stream) == cudaSuccess;
      }
      if (ist_ok) {
        ist_ok = txsec_inverse_dequant(
                     buffers.ist_dequant, buffers.ist_dequant,
                     buffers.ist_jobs, kIstSlots, stream) == cudaSuccess;
      }
      if (ist_ok) {
        ist_ok = txv_inverse_block_multi(
                     buffers.ist_dequant, buffers.ist_residual,
                     buffers.ist_block_jobs, kIstSlots, stream) == cudaSuccess;
      }
      if (!ist_ok) {
        error = "gpu: IST candidate evaluation failed";
        return GpuRunStatus::kError;
      }
      const int x0 = block.mi_col * 4;
      const int y0 = block.mi_row * 4;
      ist_sse_kernel<<<kIstSlots, 1, 0, stream>>>(
          buffers.src.y, static_cast<int>(picture.width),
          static_cast<int>(picture.width) - 1,
          static_cast<int>(picture.height) - 1,
          buffers.pred_packed[0] + block.pred_offset_y,
          buffers.ist_block_jobs, buffers.ist_residual, x0, y0,
          buffers.ist_sse);
      ist_base_sse_kernel<<<1, 1, 0, stream>>>(
          buffers.src.y, static_cast<int>(picture.width),
          static_cast<int>(picture.width) - 1,
          static_cast<int>(picture.height) - 1,
          buffers.pred_packed[0] + block.pred_offset_y, primary_residual,
          block.luma_w, block.luma_h, x0, y0, buffers.ist_base_sse);
      const double lambda = ::av2enc::rdo::lambda_sse_from_quant_step(
          1.0 / host_params->quant_factor, options.ist_sse_scale);
      ist_decide_kernel<<<1, 1, 0, stream>>>(
          buffers.eob + static_cast<std::size_t>(i) * 3u,
          buffers.tx_types + static_cast<std::size_t>(i) * 3u,
          buffers.ist_valid, buffers.ist_eob, buffers.ist_sse,
          buffers.ist_bits, buffers.ist_base_sse, lambda, block.luma_w,
          block.luma_h, primary_levels,
          buffers.eob + static_cast<std::size_t>(i) * 3u, primary_residual,
          buffers.ist_levels, buffers.ist_residual,
          buffers.ist_secs + i, buffers.ist_sets + i, adj_elems,
          coded_elems);
      if (cudaGetLastError() != cudaSuccess) {
        error = "gpu: IST decision failed";
        return GpuRunStatus::kError;
      }
    }
#endif
    if (cm1::recon_write_blocks(
            buffers.recon_jobs + static_cast<std::size_t>(i) * 3u, 1,
            stream) != cudaSuccess) {
      error = "gpu: luma reconstruction failed";
      return GpuRunStatus::kError;
    }
    }  // ordinary intra luma path
    }  // has_luma

    if (block.has_chroma != 0) {
      if (leaf_ibc) {
        const ::av2enc::still::StillResult::IbcPlan& ibc =
            cpu_plan->ibc_by_leaf[static_cast<std::size_t>(i)];
        set_chroma_dc_kernel<<<1, 1, 0, stream>>>(buffers.chroma_results, i);
        av2enc::cuda::IbcDeviceFrame frame;
        frame.y = buffers.recon.y;
        frame.u = buffers.recon.u;
        frame.v = buffers.recon.v;
        frame.stride_y = static_cast<int>(plan.y_stride);
        frame.stride_u = static_cast<int>(plan.u_stride);
        frame.stride_v = static_cast<int>(plan.v_stride);
        frame.width = coded_w;
        frame.height = coded_h;
        frame.has_chroma = 1;
        frame.sub_x = 1;
        frame.sub_y = 1;
        const av2enc::ibc::BlockVector bv{ibc.bv_row, ibc.bv_col};
        if (av2enc::cuda::ibc_copy_block(
                frame, block.mi_row, block.mi_col, block.bsize, bv,
                buffers.pred_packed[0] + block.pred_offset_y, block.luma_w,
                buffers.pred_packed[1] + block.pred_offset_u, block.chroma_w,
                buffers.pred_packed[2] + block.pred_offset_v, block.chroma_w,
                std::min<int>(block.luma_w, coded_w - x0),
                std::min<int>(block.luma_h, coded_h - y0),
                stream) != cudaSuccess) {
          error = "gpu: IBC chroma copy failed";
          return GpuRunStatus::kError;
        }
        const std::size_t chroma_pixels =
            static_cast<std::size_t>(block.chroma_w) * block.chroma_h;
        for (int p = 1; p <= 2; ++p) {
          const std::uint8_t* pred =
              (p == 1) ? buffers.pred_packed[1] : buffers.pred_packed[2];
          const std::size_t pred_off =
              (p == 1) ? block.pred_offset_u : block.pred_offset_v;
          (void)pred;
          (void)pred_off;
          if (cudaMemsetAsync(
                  buffers.residual_packed[p] +
                      ((p == 1) ? block.residual_offset_u
                                : block.residual_offset_v),
                  0, chroma_pixels * sizeof(std::int16_t), stream) !=
              cudaSuccess) {
            error = "gpu: IBC chroma residual reset failed";
            return GpuRunStatus::kError;
          }
        }
        if (cm1::recon_write_blocks(
                buffers.recon_jobs + static_cast<std::size_t>(i) * 3u + 1u, 2,
                stream) != cudaSuccess) {
          error = "gpu: IBC chroma reconstruction failed";
          return GpuRunStatus::kError;
        }
      } else {
      prepare_chroma_job_kernel<<<1, 1, 0, stream>>>(
          buffers.chroma_jobs, buffers.blocks, buffers.luma_results,
          buffers.mi_luma_mode, buffers.mi_luma_angle, buffers.uv_smooth,
          mi_cols, mi_rows, gates.enable_edge_filter, i);
      if (cm1::chroma_decide_batch(
              buffers.chroma_jobs + i, 1, chroma_options,
              buffers.chroma_results + i, stream) != cudaSuccess) {
        error = "gpu: chroma decision failed";
        return GpuRunStatus::kError;
      }

      for (int p = 1; p <= 2; ++p) {
        tx_buffers.src = (p == 1) ? buffers.src.u : buffers.src.v;
        tx_buffers.pred = buffers.pred_packed[p];
        tx_buffers.levels = (p == 1) ? buffers.coeff.u : buffers.coeff.v;
        tx_buffers.residual_out = buffers.residual_packed[p];
        tx_buffers.eob = buffers.eob;
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
        if (cctx_active) {
          continue;
        }
#endif
        if (still_tx_batch(
                buffers.tx_jobs + static_cast<std::size_t>(i) * 3u + p, 1,
                tx_buffers, 0, stream) != cudaSuccess) {
          error = "gpu: chroma TX failed";
          return GpuRunStatus::kError;
        }
      }
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
      if (cctx_active) {
        const auto cparams = txv::make_tx_params(
            block.chroma_w, block.chroma_h, txv::TxType::kDctDct);
        if (!cparams.has_value()) {
          error = "unsupported: CCTX chroma transform size";
          return GpuRunStatus::kUnsupported;
        }
        const int area = static_cast<int>(cparams->w) * cparams->h;
        const int coded_elems =
            static_cast<int>(block.chroma_w) * block.chroma_h;
        const int cx = (block.chroma_mi_size != 0 ? block.chroma_mi_col
                                                  : block.mi_col) *
                       2;
        const int cy = (block.chroma_mi_size != 0 ? block.chroma_mi_row
                                                  : block.mi_row) *
                       2;
        const bool has_stage =
            tx_stage_off + 3u * 4096u <= gpu.tx_stage.capacity();
        if (!has_stage) {
          error = "gpu: CCTX staging exhausted";
          return GpuRunStatus::kError;
        }
        auto* stage = static_cast<std::uint8_t*>(gpu.tx_stage.data());
        StillTxJob probe;
        if (!still_tx_init_job(&probe, block.chroma_w, block.chroma_h,
                               txv::TxType::kDctDct)) {
          error = "unsupported: CCTX chroma TX size";
          return GpuRunStatus::kUnsupported;
        }
        probe.input = static_cast<std::uint8_t>(StillTxInput::kSourcePred);
        probe.src_packed = 0;
        probe.src_x = static_cast<std::int16_t>(cx);
        probe.src_y = static_cast<std::int16_t>(cy);
        probe.src_width = static_cast<std::int16_t>(chroma_vis_w);
        probe.src_height = static_cast<std::int16_t>(chroma_vis_h);
        probe.src_stride = static_cast<std::int32_t>(chroma_vis_w);
        probe.residual_adj_offset = 0;
        probe.coeff_offset = 0;
        probe.levels_offset = 0;
        probe.dequant_offset = 0;
        probe.residual_out_offset = 0;
        probe.eob_offset = 0;
        bool cctx_ok = true;
        for (int p = 1; p <= 2 && cctx_ok; ++p) {
          probe.pred_offset = static_cast<std::int32_t>(
              p == 1 ? block.pred_offset_u : block.pred_offset_v);
          std::memcpy(stage + tx_stage_off, &probe, sizeof(probe));
          cctx_ok = cudaMemcpyAsync(buffers.tx_probe_job,
                                    stage + tx_stage_off, sizeof(probe),
                                    cudaMemcpyHostToDevice, stream) ==
                    cudaSuccess;
          tx_stage_off += 256u;
          StillTxBuffers pb;
          pb.src = (p == 1) ? buffers.src.u : buffers.src.v;
          pb.pred = buffers.pred_packed[p];
          pb.residual_adj = buffers.tx_residual_adj;
          pb.levels = buffers.ist_levels;
          pb.coeff = (p == 1) ? buffers.cctx_u_coeff : buffers.cctx_v_coeff;
          pb.error_count = buffers.tx_errors;
          cctx_ok = cctx_ok &&
                    still_tx_batch(buffers.tx_probe_job, 1, pb, 0, stream) ==
                        cudaSuccess;
        }
        if (!cctx_ok) {
          error = "gpu: CCTX forward transform failed";
          return GpuRunStatus::kError;
        }
        cctx_levels_kernel<<<kCctxTypes, 1, 0, stream>>>(
            buffers.cctx_u_coeff, buffers.cctx_v_coeff,
            1.0 / cparams->quant_factor, buffers.cctx_levels,
            buffers.cctx_eob, area, cparams->w, cparams->h);
        for (int slot = 0; slot < kCctxSlots && cctx_ok; ++slot) {
          cctx_ok = txv_dequantize(buffers.cctx_levels + slot * kTxTypeAdjElems,
                                   buffers.cctx_dequant + slot * kTxTypeAdjElems,
                                   *cparams, 1, stream) == cudaSuccess;
        }
        if (cctx_ok) {
          cctx_rotate_kernel<<<kCctxTypes, 1, 0, stream>>>(
              buffers.cctx_dequant, area);
        }
        // 14 mixed jobs: every (type, plane) has its own in/out slot.
        TxBlockJob cctx_jobs[kCctxSlots];
        for (int slot = 0; slot < kCctxSlots; ++slot) {
          cctx_jobs[slot].params = *cparams;
          cctx_jobs[slot].in_offset = slot * kTxTypeAdjElems;
          cctx_jobs[slot].out_offset = slot * kTxTypeCodedElems;
        }
        std::memcpy(stage + tx_stage_off, cctx_jobs, sizeof(cctx_jobs));
        cctx_ok = cctx_ok &&
                  cudaMemcpyAsync(buffers.cctx_jobs, stage + tx_stage_off,
                                  sizeof(cctx_jobs), cudaMemcpyHostToDevice,
                                  stream) == cudaSuccess;
        tx_stage_off += 1024u;
        if (cctx_ok) {
          cctx_ok = txv_inverse_block_multi(
                        buffers.cctx_dequant, buffers.cctx_residual,
                        buffers.cctx_jobs, kCctxSlots, stream) == cudaSuccess;
        }
        if (!cctx_ok) {
          error = "gpu: CCTX candidate evaluation failed";
          return GpuRunStatus::kError;
        }
        cctx_sse_kernel<<<kCctxTypes, 1, 0, stream>>>(
            buffers.src.u, buffers.src.v, chroma_vis_w, chroma_vis_w - 1,
            chroma_vis_h - 1, buffers.pred_packed[1] + block.pred_offset_u,
            buffers.pred_packed[2] + block.pred_offset_v,
            buffers.cctx_residual, block.chroma_w, block.chroma_h, cx, cy,
            buffers.cctx_sse);
        CctxBits bits;
        for (int t = 0; t < kCctxTypes; ++t) {
          bits.value[t] = ::av2enc::tcq::cctx_type_bits(
              static_cast<::av2enc::tcq::CctxType>(t));
        }
        const double cctx_lambda = ::av2enc::rdo::lambda_sse_from_quant_step(
            1.0 / cparams->quant_factor, 0.125);
        cctx_decide_kernel<<<1, 1, 0, stream>>>(
            buffers.cctx_sse, buffers.cctx_eob, bits, cctx_lambda,
            buffers.cctx_types + i,
            buffers.coeff.u + block.coeff_offset_u,
            buffers.coeff.v + block.coeff_offset_v,
            buffers.eob + static_cast<std::size_t>(i) * 3u + 1u,
            buffers.eob + static_cast<std::size_t>(i) * 3u + 2u,
            buffers.residual_packed[1] + block.residual_offset_u,
            buffers.residual_packed[2] + block.residual_offset_v,
            buffers.cctx_levels, buffers.cctx_residual, area, coded_elems);
        if (cudaGetLastError() != cudaSuccess) {
          error = "gpu: CCTX decision failed";
          return GpuRunStatus::kError;
        }
      }
#endif
      if (cm1::recon_write_blocks(
              buffers.recon_jobs + static_cast<std::size_t>(i) * 3u + 1u, 2,
              stream) != cudaSuccess) {
        error = "gpu: chroma reconstruction failed";
        return GpuRunStatus::kError;
      }
      }  // ordinary chroma path
    }  // has_chroma

    prepare_commit_kernel<<<1, 1, 0, stream>>>(
        buffers.recon_infos, buffers.luma_results, buffers.chroma_results, i);
    if (cm1::recon_commit_blocks(buffers.recon_infos + i, 1, state,
                                 recon_frame, stream) != cudaSuccess) {
      error = "gpu: reconstruction commit failed";
      return GpuRunStatus::kError;
    }
  }

  // ---- deblock ----------------------------------------------------------
  if (options.enable_deblock) {
    deblock::DeblockOptions deblock_options;
    for (int i = 0; i < 4; ++i) {
      deblock_options.apply_deblocking_filter[i] = true;
    }
    deblock_options.frame_is_intra = true;
    deblock_options.enable_sdp = false;
    if (cpu_plan != nullptr && cpu_plan->delta_q_enabled) {
      // §7.17.6: the chroma filter strength uses the effective UV AC delta.
      // The CPU plan carries the resolved totals (frame + sequence base), so
      // they are passed as the frame delta with a zero base.
      deblock_options.delta_q_u_ac = cpu_plan->delta_q_ac[1];
      deblock_options.delta_q_v_ac = cpu_plan->delta_q_ac[2];
      deblock_options.base_uv_ac_delta_q = 0;
    }
    if (cm1::recon_deblock_frame(recon_frame, deblock_options, stream) !=
        cudaSuccess) {
      error = "gpu: deblock failed";
      return GpuRunStatus::kError;
    }
  }

  // ---- in-loop filters (§7.18 CDEF) -------------------------------------
  bool filtered = false;
#if AV2ENC_STILL_GPU_HAVE_CDEF
  if (cdef_active) {
    ::av2enc::cdef::CdefOptions cdef_options;
    const ::av2enc::still::StillCdefConfig& cfg = filter_config->cdef;
    cdef_options.damping = cfg.damping;
    cdef_options.num_strengths = cfg.num_strengths;
    cdef_options.on_skip_txfm = cfg.on_skip_txfm;
    cdef_options.disable_loopfilters_across_tiles = false;
    cdef_options.bit_depth = 8;
    for (int i = 0; i < cfg.num_strengths && i < ::av2enc::cdef::kMaxStrengths;
         ++i) {
      cdef_options.strengths[i].y_pri = cfg.strengths[i].y_pri;
      cdef_options.strengths[i].y_sec = cfg.strengths[i].y_sec;
      cdef_options.strengths[i].uv_pri = cfg.strengths[i].uv_pri;
      cdef_options.strengths[i].uv_sec = cfg.strengths[i].uv_sec;
    }
    cm1::CdefDeviceFrame df;
    df.src[0] = buffers.recon.y;
    df.src[1] = buffers.recon.u;
    df.src[2] = buffers.recon.v;
    df.dst[0] = buffers.filter.y;
    df.dst[1] = buffers.filter.u;
    df.dst[2] = buffers.filter.v;
    df.stride[0] = static_cast<int>(plan.y_stride);
    df.stride[1] = static_cast<int>(plan.u_stride);
    df.stride[2] = static_cast<int>(plan.u_stride);
    df.width = static_cast<int>(picture.width);
    df.height = static_cast<int>(picture.height);
    df.mi_cols = mi_cols;
    df.mi_rows = mi_rows;
    df.skip = buffers.cdef_skip;
    df.lossless = buffers.cdef_lossless;
    df.lossless_chroma = buffers.cdef_lossless_chroma;
    df.cdef_idx = buffers.cdef_idx;
    df.n64_cols = (mi_cols + 15) / 16;
    df.n64_rows = (mi_rows + 15) / 16;
    std::string cdef_error;
    if (!cm1::still_cdef_validate_device_frame(df, cdef_options,
                                               &cdef_error)) {
      error = "gpu: CDEF frame rejected: " + cdef_error;
      return GpuRunStatus::kUnsupported;
    }
    if (cm1::still_cdef_filter(df, cdef_options, stream) != cudaSuccess) {
      error = "gpu: CDEF filter failed";
      return GpuRunStatus::kError;
    }
    filtered = true;
  }
#endif

  // ---- in-loop filters (§7.19 CCSO) -------------------------------------
#if AV2ENC_STILL_GPU_HAVE_CCSO
  if (ccso_active) {
    // CCSO modifies CdefFrame in place.  Without CDEF the CDEF frame is the
    // deblocked reconstruction, so seed the destination from it first.
    if (!filtered) {
      const std::size_t plane_y =
          static_cast<std::size_t>(plan.y_stride) * plan.padded_height;
      const std::size_t plane_c =
          static_cast<std::size_t>(plan.u_stride) * (plan.padded_height / 2u);
      if (cudaMemcpyAsync(buffers.filter.y, buffers.recon.y, plane_y,
                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
          cudaMemcpyAsync(buffers.filter.u, buffers.recon.u, plane_c,
                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
          cudaMemcpyAsync(buffers.filter.v, buffers.recon.v, plane_c,
                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        error = "gpu: CCSO input copy failed";
        return GpuRunStatus::kError;
      }
    }
    ::av2enc::restoration::CcsoCoreFrame core;
    core.mi_cols = mi_cols;
    core.mi_rows = mi_rows;
    core.sub_x = 1;
    core.sub_y = 1;
    core.bit_depth = 8;
    // §5.18.7.12: CcsoLumaSizeLog2 = Mi_Width_Log2[SbSize] + 2 (6 at 64x64,
    // 7 at the 128x128 intra frame superblock; 8 is legal for 256).
    const int ccso_sb_log2 = plan.sb_size == 128 ? 5 : 4;
    core.ccso_luma_size_log2 = ccso_sb_log2 + 2;
    core.sb_width = plan.sb_size;
    core.sb_height = plan.sb_size;
    std::uint8_t* dst_planes[3] = {buffers.filter.y, buffers.filter.u,
                                   buffers.filter.v};
    const int dst_strides[3] = {static_cast<int>(plan.y_stride),
                                static_cast<int>(plan.u_stride),
                                static_cast<int>(plan.u_stride)};
    for (int plane = 0; plane < 3; ++plane) {
      const ::av2enc::still::StillCcsoPlane& cfg =
          filter_config->ccso.plane[plane];
      ::av2enc::restoration::CcsoCorePlane& out = core.plane[plane];
      out.enabled = cfg.enabled;
      out.curr_luma = buffers.recon.y;
      out.curr_luma_stride = static_cast<int>(plan.y_stride);
      out.dst = dst_planes[plane];
      out.dst_stride = dst_strides[plane];
      out.blks = buffers.ccso_blks[plane];
      out.blk_cols = cpu_plan->ccso_blk_cols[plane];
      out.blk_rows = cpu_plan->ccso_blk_rows[plane];
      out.scale_idx = cfg.scale_idx;
      out.quant_idx = cfg.quant_idx;
      out.ext_filter = cfg.ext_filter;
      out.edge_clf = cfg.edge_clf;
      out.max_band_log2 = cfg.max_band_log2;
      out.bo_only = cfg.bo_only;
      out.offsets = buffers.ccso_offsets[plane];
    }
    if (av2enc::cuda::ccso_frame(core, stream) != cudaSuccess) {
      error = "gpu: CCSO filter failed";
      return GpuRunStatus::kError;
    }
    filtered = true;
  }
#endif

  // ---- in-loop filters (§7.20 loop restoration) -------------------------
  bool lr_filtered = false;
#if AV2ENC_STILL_GPU_HAVE_WIENER
  if (lr_active || gdf_active) {
    // CdefFrame is the post-CDEF/CCSO picture; without either filter it is the
    // deblocked reconstruction.  LrFrame must be a distinct buffer.
    const U8Planes& cdef_planes = filtered ? buffers.filter : buffers.recon;
    ::av2enc::restoration::RestorationFrame frame{};
    for (int plane = 0; plane < 3; ++plane) {
      frame.curr[plane].data =
          plane == 0 ? buffers.recon.y
                     : (plane == 1 ? buffers.recon.u : buffers.recon.v);
      frame.curr[plane].stride =
          static_cast<int>(plane == 0 ? plan.y_stride : plan.u_stride);
      frame.cdef[plane].data =
          plane == 0 ? cdef_planes.y
                     : (plane == 1 ? cdef_planes.u : cdef_planes.v);
      frame.cdef[plane].stride =
          static_cast<int>(plane == 0 ? plan.y_stride : plan.u_stride);
      frame.lr[plane].data =
          plane == 0 ? buffers.lr.y
                     : (plane == 1 ? buffers.lr.u : buffers.lr.v);
      frame.lr[plane].stride =
          static_cast<int>(plane == 0 ? plan.y_stride : plan.u_stride);
    }
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.subsampling_x = 1;
    frame.subsampling_y = 1;
    frame.bit_depth = 8;
    // §7.20.1: sbShift = Mi_Width_Log2[SbSize]; an intra frame uses the
    // 128x128 frame superblock under --still-sb128/--still-sb256.
    frame.sb_width = plan.sb_size;
    frame.sb_height = plan.sb_size;
    frame.sb_shift = plan.sb_size == 128 ? 5 : 4;

    ::av2enc::restoration::LoopRestorationParams lr_params{};
    const ::av2enc::still::StillLrPlane& luma_plane =
        filter_config->lr.plane[0];
    lr_params.frame_type[0] =
        (luma_plane.enabled && !pcw_active)
            ? ::av2enc::restoration::kRestoreWienerNs
            : ::av2enc::restoration::kRestoreNone;
    lr_params.unit_size[0] = 512 >> luma_plane.shift;
    // Chroma is applied by the lr_chroma workspace (it carries the
    // §5.20.10.6 subset/uv_sym/bank syntax that the generic Wiener launcher
    // does not model); the generic launcher still performs the §7.20
    // LrFrame := CdefFrame copy for every plane.
    lr_params.frame_type[1] = ::av2enc::restoration::kRestoreNone;
    lr_params.frame_type[2] = ::av2enc::restoration::kRestoreNone;
    lr_params.unit_size[1] = (512 >> 1) >> filter_config->lr.plane[1].shift;
    lr_params.unit_size[2] = (512 >> 1) >> filter_config->lr.plane[2].shift;
    if (luma_plane.enabled) {
      lr_params.unit_type[0] = buffers.lr_unit_type[0];
      lr_params.unit_rows[0] = luma_plane.unit_rows;
      lr_params.unit_cols[0] = luma_plane.unit_cols;
      lr_params.unit_coeffs[0] = buffers.lr_unit_coeffs[0];
    }
    const cudaError_t lr_err =
        ::av2enc::cuda::loop_restoration_frame(frame, lr_params, stream);
    if (lr_err != cudaSuccess) {
      error = std::string("gpu: loop restoration failed: ") +
              cudaGetErrorName(lr_err);
      return GpuRunStatus::kError;
    }
    lr_filtered = true;
  }
#endif
#if AV2ENC_STILL_GPU_HAVE_PC_WIENER
  if (pcw_active) {
    FilterWorkspaces& workspaces = gpu.filters;
    if (!workspaces.pc_wiener.initialized()) {
      error = "gpu: PC-Wiener workspace is not initialized";
      return GpuRunStatus::kError;
    }
    const U8Planes& cdef_planes = filtered ? buffers.filter : buffers.recon;
    const int luma_w = coded_w;
    const int luma_h = coded_h;
    const int luma_stride = static_cast<int>(plan.y_stride);
    launch_u8_to_u16(buffers.recon.y, luma_stride, buffers.luma16_curr.y,
                     luma_stride, luma_w, luma_h, stream);
    launch_u8_to_u16(cdef_planes.y, luma_stride, buffers.luma16_cdef.y,
                     luma_stride, luma_w, luma_h, stream);
    // §7.20.1 LrTxSkip is the CPU plan's per-4x4 eob==0 map; its grid is
    // uploaded above so the device classifier matches the reference.
    pc_wiener::PcWienerSource src{};
    src.curr = buffers.luma16_curr.y;
    src.curr_stride = luma_stride;
    src.cdef = buffers.luma16_cdef.y;
    src.cdef_stride = luma_stride;
    src.luma_start_x = 0;
    src.luma_end_x = luma_w - 1;
    src.luma_start_y = 0;
    src.luma_end_y = luma_h - 1;
    src.stripe_start_y = 0;
    src.stripe_end_y = luma_h - 1;
    pc_wiener::PcWienerParams params{};
    params.base_q_idx = options.base_q_idx;
    params.bit_depth = 8;
    params.mi_col_end = mi_cols;
    params.mi_row_start = 0;
    params.mi_row_end = mi_rows;
    params.lr_tx_skip = buffers.lr_tx_skip;
    params.lr_tx_skip_stride = mi_cols;
    std::string detail;
    if (!is_ok(pc_wiener::pc_wiener_classify(workspaces.pc_wiener, src, params,
                                             stream))) {
      error = "gpu: PC-Wiener classify failed";
      return GpuRunStatus::kError;
    }
    if (!is_ok(pc_wiener::pc_wiener_filter_units(
            workspaces.pc_wiener, src, params, nullptr,
            buffers.lr_unit_type[0], buffers.luma16_lr.y, luma_stride,
            stream))) {
      error = "gpu: PC-Wiener filter failed";
      return GpuRunStatus::kError;
    }
    launch_u16_to_u8(buffers.luma16_lr.y, luma_stride, buffers.lr.y, luma_stride,
                     luma_w, luma_h, stream);
    lr_filtered = true;
  }
#endif
#if AV2ENC_STILL_GPU_HAVE_LR_CHROMA
  if (lr_chroma_active) {
    FilterWorkspaces& workspaces = gpu.filters;
    if (!workspaces.chroma_lr.initialized()) {
      error = "gpu: chroma loop-restoration workspace is not initialized";
      return GpuRunStatus::kError;
    }
    lr_chroma::LrChromaFrame frame{};
    const U8Planes& cdef_planes = filtered ? buffers.filter : buffers.recon;
    frame.chroma[0].curr = buffers.recon.u;
    frame.chroma[0].curr_stride = static_cast<int>(plan.u_stride);
    frame.chroma[0].cdef = cdef_planes.u;
    frame.chroma[0].cdef_stride = static_cast<int>(plan.u_stride);
    frame.chroma[0].dst = buffers.lr.u;
    frame.chroma[0].dst_stride = static_cast<int>(plan.u_stride);
    frame.chroma[1].curr = buffers.recon.v;
    frame.chroma[1].curr_stride = static_cast<int>(plan.v_stride);
    frame.chroma[1].cdef = cdef_planes.v;
    frame.chroma[1].cdef_stride = static_cast<int>(plan.v_stride);
    frame.chroma[1].dst = buffers.lr.v;
    frame.chroma[1].dst_stride = static_cast<int>(plan.v_stride);
    frame.luma.curr = buffers.recon.y;
    frame.luma.curr_stride = static_cast<int>(plan.y_stride);
    frame.luma.cdef = cdef_planes.y;
    frame.luma.cdef_stride = static_cast<int>(plan.y_stride);
    frame.luma.dst = buffers.lr.y;
    frame.luma.dst_stride = static_cast<int>(plan.y_stride);
    frame.mi_cols = mi_cols;
    frame.mi_rows = mi_rows;
    frame.sub_x = 1;
    frame.sub_y = 1;
    frame.bit_depth = 8;
    frame.cfl_ds_filter_index = 0;
    frame.disable_loopfilters_across_tiles = 0;

    for (int chroma = 0; chroma < 2; ++chroma) {
      const ::av2enc::still::StillLrPlane& plane =
          filter_config->lr.plane[chroma + 1];
      const int units = workspaces.chroma_lr.unit_count(chroma);
      if (units <= 0 ||
          static_cast<int>(plane.unit_type.size()) != units ||
          static_cast<int>(plane.unit_coeffs.size()) !=
              units * 18) {
        error = "gpu: chroma loop-restoration unit plan mismatch";
        return GpuRunStatus::kError;
      }
      std::vector<lr_chroma::LrChromaUnit> host_units(
          static_cast<std::size_t>(units));
      for (int i = 0; i < units; ++i) {
        lr_chroma::LrChromaUnit& unit = host_units[static_cast<std::size_t>(i)];
        unit.on = plane.unit_type[static_cast<std::size_t>(i)] ==
                          ::av2enc::restoration::kRestoreWienerNs
                      ? 1
                      : 0;
        if (i < static_cast<int>(plane.unit_merged.size())) {
          unit.merged = plane.unit_merged[static_cast<std::size_t>(i)] != 0;
          unit.bank_k = plane.unit_bank_k[static_cast<std::size_t>(i)];
          unit.subset = plane.unit_subset[static_cast<std::size_t>(i)];
          unit.uv_sym = plane.unit_uv_sym[static_cast<std::size_t>(i)] != 0;
        }
        std::memcpy(unit.coeffs,
                    plane.unit_coeffs.data() + static_cast<std::size_t>(i) * 18u,
                    18u * sizeof(std::int16_t));
      }
      std::string detail;
      const Status upload = lr_chroma::upload_units(
          workspaces.chroma_lr, chroma, host_units.data(), units, stream,
          &detail);
      if (!is_ok(upload)) {
        error = "gpu: chroma loop-restoration unit upload failed: " + detail;
        return GpuRunStatus::kError;
      }
    }
    std::string detail;
    if (!is_ok(lr_chroma::resolve_units(workspaces.chroma_lr, stream,
                                        &detail))) {
      error = "gpu: chroma loop-restoration resolve failed: " + detail;
      return GpuRunStatus::kError;
    }
    if (!is_ok(lr_chroma::apply(workspaces.chroma_lr, frame, stream,
                                &detail))) {
      error = "gpu: chroma loop-restoration apply failed: " + detail;
      return GpuRunStatus::kError;
    }
    lr_filtered = true;
  }
#endif
#if AV2ENC_STILL_GPU_HAVE_GDF
  if (gdf_active) {
    FilterWorkspaces& workspaces = gpu.filters;
    if (!workspaces.gdf_ready) {
      error = "gpu: GDF workspace is not initialized";
      return GpuRunStatus::kError;
    }
    const U8Planes& cdef_planes = filtered ? buffers.filter : buffers.recon;
    const int luma_w = coded_w;
    const int luma_h = coded_h;
    const int luma_stride = static_cast<int>(plan.y_stride);
    launch_u8_to_u16(buffers.recon.y, luma_stride, buffers.luma16_curr.y,
                     luma_stride, luma_w, luma_h, stream);
    launch_u8_to_u16(cdef_planes.y, luma_stride, buffers.luma16_cdef.y,
                     luma_stride, luma_w, luma_h, stream);
    // GDF rewrites the LrFrame produced by Wiener/PC-Wiener (or the CdefFrame
    // copy when neither ran); only the filtered units change.
    launch_u8_to_u16(buffers.lr.y, luma_stride, buffers.luma16_lr.y,
                     luma_stride, luma_w, luma_h, stream);
    const int gdf_cols = filter_config->gdf.cols;
    const int gdf_rows = filter_config->gdf.rows;
    if (gdf_cols <= 0 || gdf_rows <= 0 ||
        gdf_cols * gdf_rows > workspaces.gdf.max_units) {
      error = "gpu: GDF unit grid exceeds the workspace";
      return GpuRunStatus::kError;
    }
    std::vector<std::uint8_t> use_gdf(
        static_cast<std::size_t>(gdf_cols) * gdf_rows, 1);
    std::string detail;
    if (!is_ok(gdf::gdf_upload_use_gdf(workspaces.gdf, use_gdf.data(),
                                       gdf_cols * gdf_rows, stream, &detail))) {
      error = "gpu: GDF use_gdf upload failed: " + detail;
      return GpuRunStatus::kError;
    }
    gdf::GdfFrameGrid grid{};
    grid.src.curr = buffers.luma16_curr.y;
    grid.src.curr_stride = luma_stride;
    grid.src.cdef = buffers.luma16_cdef.y;
    grid.src.cdef_stride = luma_stride;
    grid.src.start_x = 0;
    grid.src.end_x = luma_w - 1;
    grid.src.start_y = 0;
    grid.src.end_y = luma_h - 1;
    grid.src.stripe_start_y = 0;
    grid.src.stripe_end_y = luma_h - 1;
    grid.lr = buffers.luma16_lr.y;
    grid.lr_stride = luma_stride;
    grid.bit_depth = 8;
    grid.pix_scale = filter_config->gdf.pix_scale;
    grid.unit_size = filter_config->gdf.blk_size;
    grid.cols = gdf_cols;
    grid.rows = gdf_rows;
    grid.width = luma_w;
    grid.height = luma_h;
    grid.mi_rows = mi_rows;
    grid.index.frame_is_intra = 1;
    grid.index.base_q_idx = options.base_q_idx;
    grid.index.bit_depth = 8;
    grid.index.gdf_pic_qc_idx = filter_config->gdf.pic_qc_idx;
    grid.index.order_hints = nullptr;
    grid.index.frame_distance = nullptr;
    grid.index.num_total_refs = 0;
    grid.index.restricted_oh = 0;
    if (!is_ok(gdf::gdf_apply_grid(grid, workspaces.gdf, stream, &detail))) {
      error = "gpu: GDF apply failed: " + detail;
      return GpuRunStatus::kError;
    }
    launch_u16_to_u8(buffers.luma16_lr.y, luma_stride, buffers.lr.y, luma_stride,
                     luma_w, luma_h, stream);
    lr_filtered = true;
  }
#endif

  // ---- download + the one synchronisation ------------------------------
  const std::size_t recon_y_bytes =
      static_cast<std::size_t>(plan.y_stride) * plan.padded_height;
  const std::size_t recon_c_bytes =
      static_cast<std::size_t>(plan.u_stride) * (plan.padded_height / 2u);
  const std::size_t coeff_y_bytes = coeff_bytes(plan.coeff_count_y);
  const std::size_t coeff_u_bytes = coeff_bytes(plan.coeff_count_u);
  const std::size_t coeff_v_bytes = coeff_bytes(plan.coeff_count_v);
  const std::size_t eob_bytes =
      static_cast<std::size_t>(blocks) * 3u * sizeof(std::int32_t);
  const std::size_t luma_result_bytes =
      static_cast<std::size_t>(blocks) * sizeof(srch::LumaSearchResult);
  const std::size_t chroma_result_bytes =
      static_cast<std::size_t>(blocks) * sizeof(cm1::StillChromaResult);
  const std::size_t choice_bytes =
      static_cast<std::size_t>(blocks) * sizeof(iext::Choice);
  const std::size_t sb_seg_bytes =
      options.per_segment_q ? sb_slot_count(plan) : 0;
#if AV2ENC_STILL_GPU_HAVE_TX_TYPE
  const std::size_t tx_types_bytes =
      (options.tx_type || options.idtx)
          ? static_cast<std::size_t>(blocks) * 3u * sizeof(std::int32_t)
          : 0u;
#else
  const std::size_t tx_types_bytes = 0;
#endif
#if AV2ENC_STILL_GPU_HAVE_SECONDARY
  const std::size_t ist_bytes =
      options.ist ? static_cast<std::size_t>(blocks) * sizeof(std::int32_t)
                  : 0u;
#else
  const std::size_t ist_bytes = 0;
#endif
#if AV2ENC_STILL_GPU_HAVE_TCQ_CORE
  const std::size_t cctx_bytes =
      options.cctx ? static_cast<std::size_t>(blocks) * sizeof(std::int32_t)
                   : 0u;
#else
  const std::size_t cctx_bytes = 0;
#endif
  const std::size_t download_total =
      recon_y_bytes + recon_c_bytes * 2u + coeff_y_bytes + coeff_u_bytes +
      coeff_v_bytes + eob_bytes + luma_result_bytes + chroma_result_bytes +
      choice_bytes + sb_seg_bytes + tx_types_bytes + 2u * ist_bytes +
      cctx_bytes + sizeof(int);
  auto* download = static_cast<std::uint8_t*>(gpu.download.data());
  if (download == nullptr || gpu.download.capacity() < download_total) {
    error = "gpu: pinned download staging is too small";
    return GpuRunStatus::kError;
  }
  std::size_t down = 0;
  const auto enqueue_download = [&](const void* src, std::size_t bytes) -> bool {
    if (!is_ok(download_async(stream, download + down, src, bytes))) {
      return false;
    }
    down += bytes;
    return true;
  };
  const U8Planes& out_planes = lr_filtered
                                   ? buffers.lr
                                   : (filtered ? buffers.filter : buffers.recon);
  if (!enqueue_download(out_planes.y, recon_y_bytes) ||
      !enqueue_download(out_planes.u, recon_c_bytes) ||
      !enqueue_download(out_planes.v, recon_c_bytes) ||
      !enqueue_download(buffers.coeff.y, coeff_y_bytes) ||
      !enqueue_download(buffers.coeff.u, coeff_u_bytes) ||
      !enqueue_download(buffers.coeff.v, coeff_v_bytes) ||
      !enqueue_download(buffers.eob, eob_bytes) ||
      !enqueue_download(buffers.luma_results, luma_result_bytes) ||
      !enqueue_download(buffers.chroma_results, chroma_result_bytes) ||
      !enqueue_download(buffers.intra_choices, choice_bytes) ||
      (sb_seg_bytes != 0 &&
       !enqueue_download(buffers.sb_segment, sb_seg_bytes)) ||
      (tx_types_bytes != 0 &&
       !enqueue_download(buffers.tx_types, tx_types_bytes)) ||
      (ist_bytes != 0 &&
       !enqueue_download(buffers.ist_secs, ist_bytes)) ||
      (ist_bytes != 0 &&
       !enqueue_download(buffers.ist_sets, ist_bytes)) ||
      (cctx_bytes != 0 &&
       !enqueue_download(buffers.cctx_types, cctx_bytes)) ||
      !enqueue_download(buffers.tx_errors, sizeof(int))) {
    error = "gpu: result download failed";
    return GpuRunStatus::kError;
  }
  if (!is_ok(gpu.context.synchronize())) {
    error = "gpu: synchronisation failed";
    return GpuRunStatus::kError;
  }
  synced = true;

  int tx_errors = 0;
  std::size_t read = recon_y_bytes + recon_c_bytes * 2u + coeff_y_bytes +
                     coeff_u_bytes + coeff_v_bytes + eob_bytes +
                     luma_result_bytes + chroma_result_bytes + choice_bytes +
                     sb_seg_bytes + tx_types_bytes + 2u * ist_bytes +
                     cctx_bytes;
  std::memcpy(&tx_errors, download + read, sizeof(int));
  if (tx_errors != 0) {
    error = "unsupported: TX rejected " + std::to_string(tx_errors) +
            " job(s)";
    return GpuRunStatus::kUnsupported;
  }

  read = 0;
  std::memcpy(host.recon_y.data(), download + read, recon_y_bytes);
  read += recon_y_bytes;
  std::memcpy(host.recon_u.data(), download + read, recon_c_bytes);
  read += recon_c_bytes;
  std::memcpy(host.recon_v.data(), download + read, recon_c_bytes);
  read += recon_c_bytes;
  std::memcpy(host.coeff_y.data(), download + read, coeff_y_bytes);
  read += coeff_y_bytes;
  std::memcpy(host.coeff_u.data(), download + read, coeff_u_bytes);
  read += coeff_u_bytes;
  std::memcpy(host.coeff_v.data(), download + read, coeff_v_bytes);
  read += coeff_v_bytes;
  std::vector<std::int32_t> eob(static_cast<std::size_t>(blocks) * 3u);
  std::memcpy(eob.data(), download + read, eob_bytes);
  read += eob_bytes;
  std::vector<srch::LumaSearchResult> luma_results(
      static_cast<std::size_t>(blocks));
  std::memcpy(luma_results.data(), download + read, luma_result_bytes);
  read += luma_result_bytes;
  std::vector<cm1::StillChromaResult> chroma_results(
      static_cast<std::size_t>(blocks));
  std::memcpy(chroma_results.data(), download + read, chroma_result_bytes);
  read += chroma_result_bytes;
  std::vector<iext::Choice> intra_choices(static_cast<std::size_t>(blocks));
  std::memcpy(intra_choices.data(), download + read, choice_bytes);
  read += choice_bytes;
  std::vector<std::uint8_t> sb_segment(sb_seg_bytes);
  if (sb_seg_bytes != 0) {
    std::memcpy(sb_segment.data(), download + read, sb_seg_bytes);
    read += sb_seg_bytes;
  }
  std::vector<std::int32_t> tx_types(static_cast<std::size_t>(blocks) * 3u, 0);
  if (tx_types_bytes != 0) {
    std::memcpy(tx_types.data(), download + read, tx_types_bytes);
    read += tx_types_bytes;
  }
  std::vector<std::int32_t> ist_secs(static_cast<std::size_t>(blocks), 0);
  std::vector<std::int32_t> ist_sets(static_cast<std::size_t>(blocks), 0);
  if (ist_bytes != 0) {
    std::memcpy(ist_secs.data(), download + read, ist_bytes);
    read += ist_bytes;
    std::memcpy(ist_sets.data(), download + read, ist_bytes);
    read += ist_bytes;
  }
  std::vector<std::int32_t> cctx_types(static_cast<std::size_t>(blocks), 0);
  if (cctx_bytes != 0) {
    std::memcpy(cctx_types.data(), download + read, cctx_bytes);
    read += cctx_bytes;
  }

  decisions.assign(static_cast<std::size_t>(blocks), BlockDecision{});
  // §5.20.8 palette side array: the writer indexes PaletteResult entries by
  // BlockSyntax::palette_index, so build the compact array once and remember
  // each leaf's slot.
  std::vector<::av2enc::palette::PaletteResult> palette_side;
  std::vector<std::uint32_t> palette_side_index(
      static_cast<std::size_t>(blocks),
      ::av2enc::still::StillResult::kNoPaletteLeaf);
  if (screen_plan && options.palette) {
    for (int i = 0; i < blocks; ++i) {
      const std::uint32_t index =
          cpu_plan->palette_index_by_leaf[static_cast<std::size_t>(i)];
      if (index == ::av2enc::still::StillResult::kNoPaletteLeaf) continue;
      if (index >= cpu_plan->palette_results.size()) {
        error = "gpu: palette plan index out of range";
        return GpuRunStatus::kError;
      }
      palette_side_index[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(
          palette_side.size());
      palette_side.push_back(cpu_plan->palette_results[index]);
    }
  }
  // §5.20.5.6: a CHROMA_PART leaf codes uv_mode/uv_mode_index against the
  // co-located luma-tree YMode.  Record each luma leaf's decision on the MI
  // grid so chroma-only leaves can read it.
  std::vector<std::int8_t> mi_luma_mode(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows), 0);
  std::vector<std::int8_t> mi_luma_angle(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows), 0);
  for (int i = 0; i < blocks; ++i) {
    const BlockPlan& b = plan.blocks[static_cast<std::size_t>(i)];
    if (b.luma_w == 0 || b.luma_h == 0) continue;
    const srch::LumaSearchResult& r =
        luma_results[static_cast<std::size_t>(i)];
    const int w4 = b.luma_w / 4;
    const int h4 = b.luma_h / 4;
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = b.mi_row + yy;
      if (rr >= mi_rows) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = b.mi_col + xx;
        if (cc >= mi_cols) break;
        const std::size_t idx = static_cast<std::size_t>(rr) * mi_cols + cc;
        mi_luma_mode[idx] = static_cast<std::int8_t>(r.mode);
        mi_luma_angle[idx] = static_cast<std::int8_t>(r.angle_delta);
      }
    }
  }
  for (int i = 0; i < blocks; ++i) {
    BlockDecision& decision = decisions[static_cast<std::size_t>(i)];
    const srch::LumaSearchResult& luma =
        luma_results[static_cast<std::size_t>(i)];
    const cm1::StillChromaResult& chroma =
        chroma_results[static_cast<std::size_t>(i)];
    const BlockPlan& plan_block = plan.blocks[static_cast<std::size_t>(i)];
    if (sb_seg_bytes != 0) {
      const int sb_mi = plan.sb_size / 4;
      const std::size_t sb_index =
          static_cast<std::size_t>((plan_block.mi_row / sb_mi) *
                                   plan.sb_cols +
                                   plan_block.mi_col / sb_mi);
      if (sb_index < sb_segment.size()) {
        decision.segment_id = sb_segment[sb_index];
      }
    }
    const bool block_has_luma = plan_block.luma_w != 0 && plan_block.luma_h != 0;
    if (block_has_luma) {
      decision.luma_mode = static_cast<std::uint8_t>(luma.mode);
      decision.luma_angle_delta = static_cast<std::int8_t>(luma.angle_delta);
    } else {
      const int mr = plan_block.chroma_mi_row;
      const int mc = plan_block.chroma_mi_col;
      std::uint8_t mode = 0;
      std::int8_t angle = 0;
      if (mr >= 0 && mr < mi_rows && mc >= 0 && mc < mi_cols) {
        const std::size_t idx =
            static_cast<std::size_t>(mr) * mi_cols + mc;
        mode = static_cast<std::uint8_t>(mi_luma_mode[idx]);
        angle = mi_luma_angle[idx];
      }
      decision.luma_mode = mode;
      decision.luma_angle_delta = angle;
    }
    decision.sse = (!block_has_luma)
                       ? 0u
                       : (luma.sse > 0 &&
                    static_cast<std::uint64_t>(luma.sse) <
                        std::numeric_limits<std::uint32_t>::max())
                       ? static_cast<std::uint32_t>(luma.sse)
                       : (luma.sse > 0
                              ? std::numeric_limits<std::uint32_t>::max()
                              : 0u);
    decision.uv_mode = static_cast<std::uint8_t>(chroma.uv_mode);
    decision.is_cfl = chroma.is_cfl != 0 ? 1 : 0;
    decision.cfl_alpha_u = static_cast<std::int16_t>(chroma.alpha_u);
    decision.cfl_alpha_v = static_cast<std::int16_t>(chroma.alpha_v);
    decision.cfl_index = static_cast<std::uint8_t>(chroma.cfl_index);
    decision.cfl_alpha_signs =
        static_cast<std::uint8_t>(chroma.alpha_signs);
    decision.cfl_alpha_u_syntax =
        static_cast<std::uint8_t>(chroma.alpha_u_syntax);
    decision.cfl_alpha_v_syntax =
        static_cast<std::uint8_t>(chroma.alpha_v_syntax);
    decision.cfl_mh_dir = static_cast<std::uint8_t>(chroma.cfl_mh_dir);
    decision.tx_type_y = static_cast<std::uint8_t>(
        std::max<std::int32_t>(0, tx_types[static_cast<std::size_t>(i) * 3u]));
    decision.tx_type_u = 0;
    decision.tx_type_v = 0;
    decision.sec_tx_type = static_cast<std::uint8_t>(
        std::max<std::int32_t>(0, ist_secs[static_cast<std::size_t>(i)]));
    decision.stx_set = static_cast<std::uint8_t>(
        std::max<std::int32_t>(0, ist_sets[static_cast<std::size_t>(i)]));
    decision.cctx_type = static_cast<std::uint8_t>(
        std::max<std::int32_t>(0, cctx_types[static_cast<std::size_t>(i)]));
    const iext::Choice& extra = intra_choices[static_cast<std::size_t>(i)];
    decision.mrl_index = static_cast<std::uint8_t>(extra.mrl_index);
    decision.mrl_sec_index = static_cast<std::uint8_t>(extra.mrl_sec_index);
    decision.use_dip = static_cast<std::uint8_t>(extra.use_dip);
    decision.dip_mode = static_cast<std::uint8_t>(extra.dip_mode);
    decision.dip_transpose = static_cast<std::uint8_t>(extra.dip_transpose);
    decision.eob_y = static_cast<std::uint16_t>(
        std::max<std::int32_t>(0, eob[static_cast<std::size_t>(i) * 3u]));
    decision.eob_u = static_cast<std::uint16_t>(std::max<std::int32_t>(
        0, eob[static_cast<std::size_t>(i) * 3u + 1u]));
    decision.eob_v = static_cast<std::uint16_t>(std::max<std::int32_t>(
        0, eob[static_cast<std::size_t>(i) * 3u + 2u]));
    if (screen_plan) {
      if (options.palette) {
        const std::uint32_t side =
            palette_side_index[static_cast<std::size_t>(i)];
        if (side != ::av2enc::still::StillResult::kNoPaletteLeaf) {
          decision.is_palette = 1;
          decision.palette_index = side;
          // Palette forces YMode DC and clears MRL/DIP.
          decision.luma_mode = 0;
          decision.luma_angle_delta = 0;
          decision.mrl_index = 0;
          decision.mrl_sec_index = 0;
          decision.use_dip = 0;
          decision.dip_mode = 0;
          decision.dip_transpose = 0;
        }
      }
      if (options.ibc) {
        const ::av2enc::still::StillResult::IbcPlan& ibc =
            cpu_plan->ibc_by_leaf[static_cast<std::size_t>(i)];
        if (ibc.used != 0) {
          decision.is_ibc = 1;
          decision.ibc_skip = 1;
          decision.ibc_mode = ibc.mode;
          decision.ibc_ref_mv_idx = ibc.ref_mv_idx;
          decision.ibc_bv_row = ibc.bv_row;
          decision.ibc_bv_col = ibc.bv_col;
          decision.luma_mode = 0;
          decision.luma_angle_delta = 0;
          decision.mrl_index = 0;
          decision.mrl_sec_index = 0;
          decision.use_dip = 0;
          decision.dip_mode = 0;
          decision.dip_transpose = 0;
          decision.uv_mode = 0;
          decision.is_cfl = 0;
          decision.cfl_index = 1;
        }
      }
    }
  }

  // ---- GPU entropy assembly --------------------------------------------
  still_stream::GpuCoefficients coeffs{host.coeff_y, host.coeff_u, host.coeff_v};
  // With an in-loop filter the CPU reference's exact header configuration is
  // reused: the filter parameters are a planning output, so the GPU cannot
  // re-derive them independently without risking a header mismatch.
  //
  // §5.4.9 per-segment Q needs a live SegmentationParams for the CPU writer's
  // seg_info table, so it is built here (the CPU reference's copy points at a
  // stack local that does not outlive encode_still_picture).
  rc::SegmentationParams segment_params;
  if (options.per_segment_q) {
    std::string seg_error;
    if (!rc::set_segment_alt_q(segment_params, /*segment_id=*/1,
                               options.per_segment_q_delta, &seg_error)) {
      error = "gpu: per-segment Q table: " + seg_error;
      return GpuRunStatus::kError;
    }
    segment_params.enabled = true;
    segment_params.update_map = true;      // key frame: inferred 1
    segment_params.temporal_update = false;
  }
  ::av2enc::still::StillConfig config =
      (filter_config != nullptr) ? *filter_config
                                 : make_still_config(options, picture);
  config.per_sb_delta_q.segmentation_enabled =
      options.enable_segmentation || options.per_segment_q;
  config.per_sb_delta_q.delta_q_present = options.enable_delta_q;
  config.per_sb_delta_q.delta_q_res =
      options.enable_delta_q ? options.delta_q_res : 0;
  config.per_sb_delta_q.segmentation =
      options.per_segment_q ? &segment_params : nullptr;
  still_stream::FilterGrids grids;
  if (cdef_active) {
    grids.cdef_idx = cpu_plan->cdef_idx.data();
    grids.cdef_n64_cols = (mi_cols + 15) / 16;
    grids.cdef_n64_rows = (mi_rows + 15) / 16;
  }
  if (ccso_active) {
    for (int plane = 0; plane < 3; ++plane) {
      grids.ccso_blks[plane] = cpu_plan->ccso_blks[plane].empty()
                                   ? nullptr
                                   : cpu_plan->ccso_blks[plane].data();
      grids.ccso_blk_cols[plane] = cpu_plan->ccso_blk_cols[plane];
      grids.ccso_blk_rows[plane] = cpu_plan->ccso_blk_rows[plane];
    }
    std::fprintf(stderr, "ccso cpu grid: %d x %d (enable=%d)\n",
                 cpu_plan->ccso_blk_cols[0], cpu_plan->ccso_blk_rows[0],
                 filter_config->ccso.enable ? 1 : 0);
  }
  std::vector<std::uint8_t> frame;
  std::size_t tile_bytes = 0;
  if (!still_stream::assemble_gpu_frame(result.plan, tree, result.blocks, decisions,
                                    coeffs, config, frame, error,
                                    &tile_bytes, grids,
                                    palette_side)) {
    error = "gpu: entropy assembly failed: " + error;
    return GpuRunStatus::kError;
  }
  result.frame = std::move(frame);
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.tile_bytes = static_cast<std::uint32_t>(tile_bytes);
  result.frame_bytes_from_cpu = false;
  return GpuRunStatus::kOk;
}
#endif  // AV2ENC_STILL_GPU_HAVE_FULL

void log_stage(const GpuOptions& options, Stage stage, StageState state) {
  if (!options.verbose) {
    return;
  }
  std::fprintf(stderr, "m1-gpu: %-8s %s\n", stage_name(stage),
               stage_state_name(state));
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage adapters
//
// The plan workstream ships a host-side bridge to the PODs in this header.  It
// is the only component with an orchestrator-facing entry point so far; the
// remaining stage hooks stay unset until their owners provide the same shape of
// adapter (see README.md for the exact blocker per stage).
// ---------------------------------------------------------------------------
namespace stage {

#if AV2ENC_STILL_GPU_HAVE_PLAN_BRIDGE
bool build_plan(const ::av2enc::still::StillPicture& picture,
                const GpuOptions& options, FramePlan& plan,
                std::vector<BlockPlan>& blocks, std::string& error) {
  ::av2enc::still::plan::PlanOptions plan_options;
  plan_options.sb_size = 64;
  plan_options.base_q_idx = options.base_q_idx;
  plan_options.min_block_size = options.min_block_size;
  plan_options.max_partition_depth = options.max_partition_depth;
  plan_options.allow_rect = true;
  plan_options.allow_split = false;
  plan_options.force_single_block = options.force_single_block;
  ::av2enc::still::plan::apply_debug_env(plan_options);

  const std::size_t luma_bytes =
      static_cast<std::size_t>(picture.width) * picture.height;
  return ::av2enc::still::plan::build_gpu_frame_plan(
      static_cast<int>(picture.width), static_cast<int>(picture.height),
      picture.y.first(luma_bytes), plan_options, blocks, plan, error);
}
#endif

}  // namespace stage

StageState GpuStageStatus::get(Stage stage) const noexcept {
  switch (stage) {
    case Stage::kPlan:
      return plan;
    case Stage::kSearch:
      return search;
    case Stage::kTx:
      return tx;
    case Stage::kRecon:
      return recon;
    case Stage::kChroma:
      return chroma;
    case Stage::kCfl:
      return cfl;
    case Stage::kDeblock:
      return deblock;
    case Stage::kFilters:
      return filters;
    case Stage::kSched:
      return sched;
    case Stage::kCount:
      break;
  }
  return StageState::kStub;
}

int linked_stage_count() noexcept {
  int count = 0;
  count += (hook_present(stage::build_plan)) ? 1 : 0;
  count += (hook_present(stage::search)) ? 1 : 0;
  count += (hook_present(stage::transform)) ? 1 : 0;
  count += (hook_present(stage::reconstruct)) ? 1 : 0;
  count += (hook_present(stage::chroma)) ? 1 : 0;
  count += (hook_present(stage::cfl)) ? 1 : 0;
  count += (hook_present(stage::deblock)) ? 1 : 0;
  count += (hook_present(stage::schedule)) ? 1 : 0;
#if AV2ENC_STILL_GPU_HAVE_FULL
  // Full pipeline compiled into this translation unit: plan, search, TX,
  // recon, chroma (DC/CfL), deblock and the scheduler availability replay.
  // The weak hooks above are not used by the ordered executor.
  count = 7;
#endif
  return count;
}

bool encode_format_gpu(const ::av2enc::fmt::Picture& picture,
                       const ::av2enc::high_bitdepth::HighBitDepthOptions& options,
                       ::av2enc::high_bitdepth::HighBitDepthResult& result,
                       std::string& error) {
  error.clear();
  if (picture.format.depth != ::av2enc::fmt::BitDepth::k10) {
    error = "--still-gpu format dispatch currently serves 10-bit pictures";
    return false;
  }
  return ::av2enc::cuda::high_bitdepth::encode_high_bitdepth_gpu(
      picture, options, result, &error);
}

// ---------------------------------------------------------------------------
// 8-bit chroma product block executor
// ---------------------------------------------------------------------------
namespace {

__global__ void fill_u16_kernel(std::uint16_t* data, std::size_t count,
                                std::uint16_t value) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) data[index] = value;
}

class ChromaGpuExecutorImpl final
    : public ::av2enc::chroma::ChromaGpuExecutor {
 public:
  bool begin(const ::av2enc::chroma::ChromaPicture& picture,
             std::string& error) override {
    error.clear();
    config_ = ::av2enc::cuda::chroma::Config{};
    config_.width = picture.width;
    config_.height = picture.height;
    config_.format = static_cast<::av2enc::cuda::chroma::Format>(
        picture.chroma);
    config_.bit_depth = 8;
    config_.max_blocks = 1;

    const std::uint64_t need =
        ::av2enc::cuda::chroma::required_device_bytes(config_);
    std::string arena_error;
    if (!::av2enc::cuda::is_ok(arena_.init(
            need, ::av2enc::cuda::Arena::kAlignment, &arena_error))) {
      error = arena_error.empty() ? "chroma GPU arena init failed" : arena_error;
      return false;
    }
    const ::av2enc::cuda::chroma::ChromaStatus status =
        workspace_.init(arena_, config_, &error);
    if (!status.ok()) {
      if (error.empty()) error = "chroma GPU workspace init failed";
      return false;
    }

    const ::av2enc::cuda::chroma::FrameGeom& geom = workspace_.geometry();
    for (int plane = 0; plane < 3; ++plane) {
      const ::av2enc::cuda::chroma::PlaneGeom& pg = geom.planes[plane];
      if (!pg.present) continue;
      const std::size_t count =
          static_cast<std::size_t>(pg.padded_w) * pg.padded_h;
      const unsigned blocks =
          static_cast<unsigned>((count + 255) / 256);
      fill_u16_kernel<<<blocks, 256>>>(workspace_.recon_plane(plane), count,
                                      128);
      if (cudaGetLastError() != cudaSuccess) {
        error = "chroma GPU recon initialization failed";
        return false;
      }
      // The device source planes are uint16_t containers; widen the 8-bit
      // product plane once per picture.
      std::vector<std::uint16_t> host(static_cast<std::size_t>(pg.padded_w) *
                                      pg.padded_h);
      for (int y = 0; y < pg.padded_h; ++y) {
        for (int x = 0; x < pg.padded_w; ++x) {
          host[static_cast<std::size_t>(y) * pg.padded_w + x] =
              picture.planes[plane][static_cast<std::size_t>(y) *
                                        picture.plane_width[plane] +
                                    static_cast<std::size_t>(x)];
        }
      }
      if (cudaMemcpy2D(workspace_.src_plane(plane),
                       static_cast<std::size_t>(pg.padded_w) *
                           sizeof(std::uint16_t),
                       host.data(),
                       static_cast<std::size_t>(pg.padded_w) *
                           sizeof(std::uint16_t),
                       static_cast<std::size_t>(pg.padded_w) *
                           sizeof(std::uint16_t),
                       static_cast<std::size_t>(pg.padded_h),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
        error = "chroma GPU source upload failed";
        return false;
      }
    }
    return true;
  }

  bool forward(int plane, int x0, int y0, int nominal_w, int nominal_h,
               const ::av2enc::chroma::TxShape& /*cpu_tx*/, bool have_left,
               bool have_above, int* pred_value, int* levels, int* eob,
               std::string& error) override {
    using namespace ::av2enc::cuda::chroma;
    error.clear();
    TxShape shape{};
    if (!make_tx_shape(nominal_w, nominal_h, &shape)) {
      error = "chroma GPU: unsupported transform shape";
      return false;
    }
    const FrameGeom& geom = workspace_.geometry();
    const PlaneGeom& pg = geom.planes[plane];

    PredictJob predict{};
    predict.recon = workspace_.recon_plane(plane);
    predict.stride = pg.padded_w;
    predict.max_x = pg.coded_w - 1;
    predict.max_y = pg.coded_h - 1;
    predict.x0 = x0;
    predict.y0 = y0;
    predict.nominal_w = nominal_w;
    predict.nominal_h = nominal_h;
    predict.have_left = have_left ? 1 : 0;
    predict.have_above = have_above ? 1 : 0;
    predict.bit_depth = 8;
    predict.pred = workspace_.pred_block(0);
    ChromaStatus status = workspace_.predict(&predict, 1, nullptr, &error);
    if (!status.ok()) return false;

    std::uint16_t pred_scalar = 0;
    if (cudaMemcpy(&pred_scalar, workspace_.pred_block(0),
                   sizeof(pred_scalar), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      error = "chroma GPU: prediction download failed";
      return false;
    }
    *pred_value = static_cast<int>(pred_scalar);

    ResidualJob residual{};
    residual.src = workspace_.src_plane(plane);
    residual.src_stride = pg.padded_w;
    residual.max_x = pg.padded_w - 1;
    residual.max_y = pg.padded_h - 1;
    residual.x0 = x0;
    residual.y0 = y0;
    residual.pred = *pred_value;
    residual.bit_depth = 8;
    residual.shape = shape;
    residual.residual_adj = workspace_.residual_adj(0);
    status = workspace_.make_residuals(&residual, 1, nullptr, &error);
    if (!status.ok()) return false;

    ForwardJob forward{};
    forward.shape = shape;
    forward.residual_adj = workspace_.residual_adj(0);
    forward.coeff = workspace_.coeff(0);
    forward.levels = workspace_.levels(0);
    status = workspace_.forward(&forward, 1, nullptr, &error);
    if (!status.ok()) return false;

    const std::size_t area =
        static_cast<std::size_t>(shape.adj_w) * shape.adj_h;
    std::vector<std::int32_t> host_levels(area);
    if (cudaMemcpy(host_levels.data(), workspace_.levels(0),
                   area * sizeof(std::int32_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      error = "chroma GPU: level download failed";
      return false;
    }
    std::vector<std::uint16_t> scan;
    ::av2enc::chroma::build_diagonal_scan(shape.adj_w, shape.adj_h, scan);
    int found = 0;
    for (int c = shape.seg_eob - 1; c >= 0; --c) {
      if (host_levels[scan[static_cast<std::size_t>(c)]] != 0) {
        found = c + 1;
        break;
      }
    }
    *eob = found;
    for (std::size_t i = 0; i < area; ++i) {
      levels[i] = host_levels[i];
    }
    return true;
  }

  bool reconstruct(int plane, int x0, int y0, int nominal_w, int nominal_h,
                   const ::av2enc::chroma::TxShape& /*cpu_tx*/, int pred_value,
                   const int* levels, int eob, std::string& error) override {
    using namespace ::av2enc::cuda::chroma;
    error.clear();
    TxShape shape{};
    if (!make_tx_shape(nominal_w, nominal_h, &shape)) {
      error = "chroma GPU: unsupported transform shape";
      return false;
    }
    const FrameGeom& geom = workspace_.geometry();
    const PlaneGeom& pg = geom.planes[plane];
    const std::size_t area =
        static_cast<std::size_t>(shape.adj_w) * shape.adj_h;

    if (eob == 0) {
      if (cudaMemset(workspace_.residual_adj(0), 0,
                     area * sizeof(std::int16_t)) != cudaSuccess) {
        error = "chroma GPU: residual clear failed";
        return false;
      }
    } else {
      std::vector<std::int32_t> host_levels(area);
      for (std::size_t i = 0; i < area; ++i) {
        host_levels[i] = levels[i];
      }
      if (cudaMemcpy(workspace_.levels(0), host_levels.data(),
                     area * sizeof(std::int32_t), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        error = "chroma GPU: level upload failed";
        return false;
      }
      DequantJob dequant{};
      dequant.shape = shape;
      dequant.bit_depth = 8;
      dequant.levels = workspace_.levels(0);
      dequant.dequant = workspace_.dequant(0);
      ChromaStatus status =
          workspace_.dequantize(&dequant, 1, nullptr, &error);
      if (!status.ok()) return false;

      InverseJob inverse{};
      inverse.shape = shape;
      inverse.bit_depth = 8;
      inverse.dequant = workspace_.dequant(0);
      inverse.residual_adj = workspace_.residual_adj(0);
      status = workspace_.inverse(&inverse, 1, nullptr, &error);
      if (!status.ok()) return false;
    }

    ReconstructJob reconstruct{};
    reconstruct.shape = shape;
    reconstruct.bit_depth = 8;
    reconstruct.pred = pred_value;
    reconstruct.residual_adj = workspace_.residual_adj(0);
    reconstruct.dst = workspace_.recon_plane(plane);
    reconstruct.dst_stride = pg.padded_w;
    reconstruct.max_x = pg.padded_w - 1;
    reconstruct.max_y = pg.padded_h - 1;
    reconstruct.x0 = x0;
    reconstruct.y0 = y0;
    const ChromaStatus status =
        workspace_.reconstruct(&reconstruct, 1, nullptr, &error);
    return status.ok();
  }

  bool download_recon(int plane, std::uint8_t* out,
                      std::string& error) override {
    error.clear();
    const ::av2enc::cuda::chroma::FrameGeom& geom = workspace_.geometry();
    const ::av2enc::cuda::chroma::PlaneGeom& pg = geom.planes[plane];
    if (!pg.present) return true;
    const std::size_t count =
        static_cast<std::size_t>(pg.padded_w) * pg.padded_h;
    std::vector<std::uint16_t> host(count);
    if (cudaMemcpy(host.data(), workspace_.recon_plane(plane),
                   count * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      error = "chroma GPU: reconstruction download failed";
      return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = static_cast<std::uint8_t>(host[i] & 0xFFU);
    }
    return true;
  }

 private:
  ::av2enc::cuda::Arena arena_;
  ::av2enc::cuda::chroma::Config config_{};
  ::av2enc::cuda::chroma::Workspace workspace_;
};

}  // namespace

std::unique_ptr<::av2enc::chroma::ChromaGpuExecutor>
make_chroma_gpu_executor() {
  return std::make_unique<ChromaGpuExecutorImpl>();
}

bool encode_still_gpu(const ::av2enc::still::StillPicture& picture,
                   const GpuOptions& options, GpuResult& result,
                   std::string& error) {
  const auto start_time = std::chrono::steady_clock::now();
  error.clear();
  result = GpuResult{};

  if (!validate_options(options, picture, error)) {
    return false;
  }
  Geometry geometry;
  if (!make_geometry(picture.width, picture.height, geometry)) {
    error = "unsupported picture dimensions";
    return false;
  }

  // ---- plan -------------------------------------------------------------
#if AV2ENC_STILL_GPU_HAVE_FULL
  std::vector<still_stream::TreeNode> tree;
  {
    ::av2enc::still::plan::PlanOptions plan_options;
    plan_options.sb_size = options.sb_size;
    plan_options.base_q_idx = options.base_q_idx;
    plan_options.min_block_size = options.min_block_size;
    plan_options.max_partition_depth = options.max_partition_depth;
    plan_options.allow_rect = true;
    plan_options.allow_split = false;
    plan_options.force_single_block = options.force_single_block;
    plan_options.enable_ext_partitions = options.ext_partitions;
    plan_options.enable_uneven_4way_partitions = options.ext_partitions;
    plan_options.sdp = options.sdp;
    plan_options.sub8 = options.sub8;
    ::av2enc::still::plan::apply_debug_env(plan_options);
    const std::size_t luma_bytes =
        static_cast<std::size_t>(picture.width) * picture.height;
    if (!::av2enc::still::plan::build_gpu_frame_plan(
            static_cast<int>(picture.width), static_cast<int>(picture.height),
            picture.y.first(luma_bytes), plan_options, result.blocks,
            result.plan, tree, error)) {
      error = "M1 GPU: plan stage failed: " + error;
      return false;
    }
    result.plan.is_placeholder = 0;
    result.rebind_plan();
  }
#else
  {
    const std::size_t block_capacity =
        static_cast<std::size_t>(geometry.sb_cols) *
        static_cast<std::size_t>(geometry.sb_rows);
    if (stage::build_plan != nullptr) {
      if (!stage::build_plan(picture, options, result.plan, result.blocks,
                             error)) {
        error = "M1 GPU: plan stage failed: " + error;
        return false;
      }
      result.plan.is_placeholder = 0;
    } else {
      result.blocks.assign(block_capacity, BlockPlan{});
      if (!build_placeholder_plan(picture, result.plan, result.blocks,
                                  error)) {
        return false;
      }
    }
    result.rebind_plan();
  }
#endif

  if (result.plan.block_count == 0 ||
      result.plan.block_count != result.blocks.size()) {
    error = "M1 GPU: plan returned an invalid block count";
    return false;
  }
  if (!validate_plan(result.plan, error)) {
    return false;
  }
  if (!fill_packed_offsets(result.plan, result.blocks)) {
    error = "M1 GPU: packed scratch layout overflows 32-bit offsets";
    return false;
  }
  result.rebind_plan();

  // ---- host buffers -----------------------------------------------------
  HostBuffers host;
  if (!allocate_host_buffers(result.plan, host, error)) {
    return false;
  }
  result.decisions.assign(result.plan.block_count, BlockDecision{});

  // In-loop filter tools need the CPU reference's planning output (header
  // parameters + cdef_idx grid) before the GPU frame runs; the CPU encode is
  // also the parity oracle, so this does not add a second encode later.
  // In-loop filters change the reconstruction and header; film grain changes
  // only the §5.14/§5.18.10 metadata.  All reuse the CPU reference's exact
  // config and planned decision grids.
  const bool filter_tools = options.enable_cdef || options.enable_ccso ||
                            options.ccso_edge || options.enable_grain ||
                            options.enable_wiener || options.enable_chroma_lr ||
                            options.enable_pc_wiener || options.enable_gdf ||
                            options.delta_q || options.qm || options.ist ||
                            options.cctx;
  const bool screen_tools = options.palette || options.ibc;
  ::av2enc::still::StillResult cpu;
  bool cpu_ready = false;
  if (filter_tools || screen_tools) {
    std::string cpu_error;
    if (!::av2enc::still::encode_still_picture(
            picture, to_cpu_options(options), cpu, cpu_error)) {
      error = "M1 CPU reference: " + cpu_error;
      return false;
    }
    cpu_ready = true;
  }

  // ---- GPU execution ----------------------------------------------------
  bool gpu_ok = false;
#if AV2ENC_STILL_GPU_HAVE_FULL
  const bool gpu_requested = true;
  const bool tree_ready = !tree.empty();
#else
  const bool gpu_requested = false;
  const bool tree_ready = false;
#endif

  const std::uint64_t luma_bytes = luma_plane_bytes(geometry);
  const std::uint64_t chroma_bytes = chroma_plane_bytes(geometry);
  const std::uint64_t src_bytes = luma_bytes + chroma_bytes * 2u;
  const std::uint64_t coeff_total =
      static_cast<std::uint64_t>(coeff_bytes(result.plan.coeff_count_y)) +
      static_cast<std::uint64_t>(coeff_bytes(result.plan.coeff_count_u)) +
      static_cast<std::uint64_t>(coeff_bytes(result.plan.coeff_count_v));
  const std::uint64_t recon_total = luma_bytes * 2u + chroma_bytes * 4u;
  const std::uint64_t decision_bytes =
      static_cast<std::uint64_t>(result.plan.block_count) *
      sizeof(BlockDecision);
  const std::uint64_t block_bytes =
      static_cast<std::uint64_t>(result.plan.block_count) * sizeof(BlockPlan);
  std::uint64_t need_device =
      src_bytes + recon_total + coeff_total + decision_bytes + block_bytes;
  std::uint64_t need_upload = src_bytes;
  std::uint64_t need_download = recon_total + coeff_total + decision_bytes;
  if (filter_tools) {
    // CDEF index grid + CCSO unit grids/offset tables are staged through the
    // same pinned upload buffer.  Loop-restoration unit descriptors come from
    // the CPU plan as well (luma unit grid/coefficients, chroma via the
    // component's upload_units()).
    const std::uint64_t n64_cols =
        (static_cast<std::uint64_t>(result.plan.mi_cols) + 15u) / 16u;
    const std::uint64_t n64_rows =
        (static_cast<std::uint64_t>(result.plan.mi_rows) + 15u) / 16u;
    need_upload += n64_cols * n64_rows;                       // cdef_idx
    need_upload += 3u * n64_cols * n64_rows;                  // ccso_blks
    need_upload += 3u * 576u * sizeof(std::int16_t);          // ccso offsets
    need_upload += n64_cols * n64_rows;                       // lr unit types
    need_upload += n64_cols * n64_rows * 18u * sizeof(std::int16_t);
    need_upload += static_cast<std::uint64_t>(result.plan.mi_cols) *
                   static_cast<std::uint64_t>(result.plan.mi_rows);  // LrTxSkip
    if (options.qm) {
      need_upload += static_cast<std::uint64_t>(q2_elems(result.plan)) *
                     sizeof(std::int32_t);
    }
    need_device += filter_workspace_bytes(result.plan, options);
  }

#if AV2ENC_STILL_GPU_HAVE_FULL
  if (gpu_requested) {
    const std::size_t n = result.plan.block_count;
    need_device += static_cast<std::uint64_t>(full_device_bytes(result.plan));
    need_upload +=
        n * (sizeof(srch::LumaSearchBlock) + sizeof(im::IntraJob) +
             sizeof(cm1::StillChromaJob) + 3u * sizeof(StillTxJob) +
             3u * sizeof(cm1::ReconWriteJob) +
             sizeof(cm1::ReconBlockInfo) + sizeof(mrl::MrlJob) +
             sizeof(dip::DipJob) + sizeof(ibp::IbpJob));
    if (options.palette) {
      need_upload += static_cast<std::uint64_t>(n) * 8u;
    }
    need_download +=
        3u * n * sizeof(std::int32_t) +
        n * (sizeof(srch::LumaSearchResult) + sizeof(cm1::StillChromaResult) +
             sizeof(iext::Choice)) +
        sizeof(int);
    if (options.per_segment_q) {
      // Device segment map metadata: one BlockPlan upload for the origin
      // lookup, one segment id per superblock downloaded for the syntax.
      need_upload += n * sizeof(BlockPlan);
      need_download +=
          static_cast<std::uint64_t>(result.plan.sb_cols) *
          static_cast<std::uint64_t>(result.plan.sb_rows);
    }
    if (options.tx_type || options.idtx) {
      need_download += 3u * n * sizeof(std::int32_t);
    }
    if (options.ist) {
      need_download += 2u * n * sizeof(std::int32_t);
    }
    if (options.cctx) {
      need_download += n * sizeof(std::int32_t);
    }
  }
#endif

  if (gpu_requested && tree_ready) {
    const RuntimeState state = ensure_runtime(
        options, &result.plan, need_device, need_upload, need_download, true,
        error);
    if (state == RuntimeState::kError) {
      return false;
    }
    if (state == RuntimeState::kCpuFallback) {
      if (!options.allow_cpu_fallback) {
        return false;
      }
      if (options.verbose) {
        std::fprintf(stderr, "m1-gpu: runtime fallback: %s\n",
                     error.empty() ? "(no detail)" : error.c_str());
      }
      error.clear();
    } else {
#if AV2ENC_STILL_GPU_HAVE_FULL
      bool synced = false;
      const ::av2enc::still::StillResult* cpu_plan =
          (filter_tools || screen_tools) && cpu_ready ? &cpu : nullptr;
      const GpuRunStatus status = run_gpu_frame(
          picture, options, result.plan, tree, host, result.decisions, result,
          cpu_plan, synced, error);
      if (status == GpuRunStatus::kError) {
        return false;
      }
      if (status == GpuRunStatus::kOk) {
        gpu_ok = true;
        result.stats.sync_count = 1;
        result.stats.tx_jobs =
            static_cast<int>(result.plan.block_count) * 3;
        result.stats.kernel_launches =
            7 * static_cast<int>(result.plan.block_count) + 2 +
            (options.enable_cdef ? 1 : 0) +    // §7.18 CDEF filter
            (options.enable_ccso || options.ccso_edge ? 1 : 0) +  // §7.19
            (options.enable_wiener ? 1 : 0) +   // §7.20.3 luma Wiener NS
            (options.enable_chroma_lr ? 2 : 0) +  // §7.20.3 chroma LR
            (options.enable_pc_wiener ? 2 : 0) +  // §7.20.4 classify + filter
            (options.enable_gdf ? 2 : 0);  // §7.20.5 derive + apply
        result.stats.host_upload_bytes = need_upload;
        result.stats.host_download_bytes = need_download;
      } else {
        if (synced) {
          result.stats.sync_count = 1;
        }
        if (!options.allow_cpu_fallback) {
          error = "M1 GPU: " + error;
          return false;
        }
        if (options.verbose ||
            std::getenv("AV2ENC_STILL_GPU_VERBOSE") != nullptr) {
          std::fprintf(stderr, "m1-gpu: GPU path unsupported, CPU fallback: %s\n",
                       error.c_str());
        }
        error.clear();
      }
#endif
    }
  } else if (!gpu_requested && !options.allow_cpu_fallback) {
    error = "M1 GPU: the full GPU pipeline is not compiled in";
    return false;
  }

  if (gpu_ok) {
    result.stages.plan = StageState::kGpu;
    result.stages.search = StageState::kGpu;
    result.stages.tx = StageState::kGpu;
    result.stages.recon = StageState::kGpu;
    result.stages.chroma = StageState::kGpu;
    result.stages.cfl = options.enable_cfl ? StageState::kGpu
                                           : StageState::kCpuReference;
    result.stages.deblock = options.enable_deblock ? StageState::kGpu
                                                   : StageState::kStub;
    result.stages.filters =
        (options.enable_cdef || options.enable_ccso || options.ccso_edge ||
         options.enable_wiener || options.enable_chroma_lr ||
         options.enable_pc_wiener || options.enable_gdf)
            ? StageState::kGpu
            : StageState::kStub;
    result.stages.sched = StageState::kGpu;
    for (Stage stage_id : {Stage::kPlan, Stage::kSearch, Stage::kTx,
                           Stage::kRecon, Stage::kChroma, Stage::kCfl,
                           Stage::kDeblock, Stage::kFilters, Stage::kSched}) {
      log_stage(options, stage_id, result.stages.get(stage_id));
    }
  } else if (gpu_requested && !tree_ready) {
    result.stages.plan = StageState::kCpuReference;
    result.stages.search = StageState::kCpuReference;
    result.stages.tx = StageState::kCpuReference;
    result.stages.recon = StageState::kCpuReference;
    result.stages.chroma = StageState::kCpuReference;
    result.stages.cfl = StageState::kCpuReference;
    result.stages.deblock = StageState::kCpuReference;
    result.stages.filters =
        (options.enable_cdef || options.enable_ccso || options.ccso_edge ||
         options.enable_wiener || options.enable_chroma_lr ||
         options.enable_pc_wiener || options.enable_gdf)
            ? StageState::kCpuReference
            : StageState::kStub;
    result.stages.sched = StageState::kStub;
  } else {
    result.stages.plan = StageState::kCpuReference;
    result.stages.search = StageState::kCpuReference;
    result.stages.tx = StageState::kCpuReference;
    result.stages.recon = StageState::kCpuReference;
    result.stages.chroma = StageState::kCpuReference;
    result.stages.cfl = StageState::kCpuReference;
    result.stages.deblock = StageState::kCpuReference;
    result.stages.filters =
        (options.enable_cdef || options.enable_ccso || options.ccso_edge ||
         options.enable_wiener || options.enable_chroma_lr ||
         options.enable_pc_wiener || options.enable_gdf)
            ? StageState::kCpuReference
            : StageState::kStub;
    result.stages.sched = StageState::kStub;
    for (Stage stage_id : {Stage::kPlan, Stage::kSearch, Stage::kTx,
                           Stage::kRecon, Stage::kChroma, Stage::kCfl,
                           Stage::kDeblock, Stage::kFilters, Stage::kSched}) {
      log_stage(options, stage_id, result.stages.get(stage_id));
    }
  }

  // ---- CPU reference: fallback bytes and the parity oracle --------------
  const bool need_cpu = !gpu_ok || options.verify_against_cpu;
  if (need_cpu) {
    if (!cpu_ready) {
      std::string cpu_error;
      if (!::av2enc::still::encode_still_picture(
              picture, to_cpu_options(options), cpu, cpu_error)) {
        error = "M1 CPU reference: " + cpu_error;
        return false;
      }
      cpu_ready = true;
    }
    if (gpu_ok) {
      if (options.verify_against_cpu) {
        if (cpu.frame.size() != result.frame.size() ||
            (!cpu.frame.empty() &&
             std::memcmp(cpu.frame.data(), result.frame.data(),
                         cpu.frame.size()) != 0)) {
          error = "M1 GPU/CPU frame mismatch: gpu " +
                  std::to_string(result.frame.size()) + " bytes, cpu " +
                  std::to_string(cpu.frame.size()) + " bytes";
          return false;
        }
        const std::uint32_t cw = (picture.width + 1u) / 2u;
        const std::uint32_t ch = (picture.height + 1u) / 2u;
        if (!compare_plane(host.recon_y, cpu.recon_y, picture.width,
                           picture.height, result.plan.y_stride, "Y", error) ||
            !compare_plane(host.recon_u, cpu.recon_u, cw, ch,
                           result.plan.u_stride, "U", error) ||
            !compare_plane(host.recon_v, cpu.recon_v, cw, ch,
                           result.plan.u_stride, "V", error)) {
          error = "M1 GPU/CPU recon parity failed: " + error;
          return false;
        }
        result.stats.gpu_frame_verified = true;
      }
      result.stats.leaf_count = cpu.leaves;
    } else {
      result.frame = std::move(cpu.frame);
      result.frame_bytes = cpu.frame_bytes;
      result.tile_bytes = cpu.tile_bytes;
      result.frame_bytes_from_cpu = true;
      result.stats.leaf_count = cpu.leaves;
      result.stats.block_count = static_cast<int>(result.plan.block_count);
      // Marshal the CPU recon through the padded host planes.
      if (!blit_visible_to_padded(cpu.recon_y, picture.width, picture.height,
                                  result.plan.y_stride, host.recon_y) ||
          !blit_visible_to_padded(cpu.recon_u, (picture.width + 1u) / 2u,
                                  (picture.height + 1u) / 2u,
                                  result.plan.u_stride, host.recon_u) ||
          !blit_visible_to_padded(cpu.recon_v, (picture.width + 1u) / 2u,
                                  (picture.height + 1u) / 2u,
                                  result.plan.u_stride, host.recon_v)) {
        error = "M1 GPU: CPU recon marshalling failed";
        return false;
      }
    }
  }

  // ---- result planes ----------------------------------------------------
  const bool crop_ok =
      crop_padded_to_visible(host.recon_y, picture.width, picture.height,
                             result.plan.y_stride, result.recon_y) &&
      crop_padded_to_visible(host.recon_u, (picture.width + 1u) / 2u,
                             (picture.height + 1u) / 2u, result.plan.u_stride,
                             result.recon_u) &&
      crop_padded_to_visible(host.recon_v, (picture.width + 1u) / 2u,
                             (picture.height + 1u) / 2u, result.plan.v_stride,
                             result.recon_v);
  if (!crop_ok) {
    error = "M1 GPU: recon crop failed";
    return false;
  }
  if (gpu_ok) {
    result.coeff_y = std::move(host.coeff_y);
    result.coeff_u = std::move(host.coeff_u);
    result.coeff_v = std::move(host.coeff_v);
  } else {
    result.coeff_y.clear();
    result.coeff_u.clear();
    result.coeff_v.clear();
  }
  result.stats.block_count = static_cast<int>(result.plan.block_count);
  if (result.stats.leaf_count == 0) {
    result.stats.leaf_count = static_cast<int>(result.plan.leaf_count);
  }

  const GpuRuntime& gpu = runtime();
  result.stats.gpu_runtime_ready = gpu.ready;
  result.stats.arena_capacity_bytes =
      gpu.arena.initialized() ? gpu.arena.capacity() : 0;
  result.stats.arena_used_bytes =
      gpu.arena.initialized() ? gpu.arena.used() : 0;
  result.stats.device_total_bytes =
      gpu.context.initialized() ? gpu.context.total_vram_bytes() : 0;
  result.stats.device_free_bytes =
      gpu.context.initialized() ? gpu.context.free_vram_bytes() : 0;
  if (gpu.context.initialized()) {
    const char* name = gpu.context.device().name;
    const std::size_t name_len = std::strlen(name);
    const std::size_t copy_len =
        std::min(name_len, sizeof(result.stats.device_name) - 1u);
    std::memcpy(result.stats.device_name, name, copy_len);
    result.stats.device_name[copy_len] = '\0';
  }
  const auto end_time = std::chrono::steady_clock::now();
  result.stats.total_ms =
      std::chrono::duration<double, std::milli>(end_time - start_time).count();
  result.rebind_plan();
  return true;
}

}  // namespace av2enc::cuda::still_gpu
