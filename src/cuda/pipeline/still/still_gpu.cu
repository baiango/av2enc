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
#if __has_include("cuda/pipeline/still/tx/still_tx.h")
#include "cuda/pipeline/still/tx/still_tx.h"
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

// Full GPU frame pipeline: real plan tree + luma search + chroma/CfL + TX +
// reconstruction + deblock + GPU entropy assembly.  Every component must be
// present; otherwise the frame falls back to the CPU reference.
#if AV2ENC_STILL_GPU_HAVE_PLAN_BRIDGE && AV2ENC_STILL_GPU_HAVE_INTRA_MODES && \
    AV2ENC_STILL_GPU_HAVE_SCHED_BRIDGE && AV2ENC_STILL_GPU_HAVE_STREAM_BRIDGE && \
    __has_include("cuda/pipeline/still/search/luma_search.h") && \
    __has_include("cuda/pipeline/still/chroma/still_chroma.h") && \
    __has_include("cuda/pipeline/still/tx/still_tx.h") && \
    __has_include("cuda/pipeline/still/recon/recon.h")
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
    if (block.luma_w == 0 || block.luma_h == 0 || block.tx_w == 0 ||
        block.tx_h == 0) {
      error = "plan: zero-sized block or transform";
      return false;
    }
    const std::size_t luma_end =
        static_cast<std::size_t>(block.plane_offset_y) +
        static_cast<std::size_t>(block.luma_h - 1u) * plan.y_stride +
        block.luma_w;
    if (luma_end > luma_bytes) {
      error = "plan: luma block outside the padded plane";
      return false;
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
    if (block.coeff_offset_y < last_coeff_y ||
        block.coeff_offset_u < last_coeff_u ||
        block.coeff_offset_v < last_coeff_v) {
      error = "plan: coefficient offsets are not monotonic";
      return false;
    }
    last_coeff_y = block.coeff_offset_y;
    last_coeff_u = block.coeff_offset_u;
    last_coeff_v = block.coeff_offset_v;
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
  im::IntraJob* luma_jobs = nullptr;
  std::size_t luma_jobs_bytes = 0;
  cm1::StillChromaResult* chroma_results = nullptr;
  std::size_t chroma_results_bytes = 0;
  std::int32_t* eob = nullptr;  // 3 * block_count (Y, U, V per block)
  std::size_t eob_bytes = 0;
  StillTxJob* tx_jobs = nullptr;  // 3 * block_count
  std::size_t tx_jobs_bytes = 0;
  int* tx_errors = nullptr;
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
#endif
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
  DeviceBuffers buffers;
  std::uint64_t arena_capacity = 0;

  ~GpuRuntime() {
    buffers = DeviceBuffers{};
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
  total += n * sizeof(im::IntraJob);
  total += n * sizeof(cm1::StillChromaResult);
  total += n * sizeof(cm1::ReconBlockInfo);
  total += 3u * n * sizeof(StillTxJob);
  total += 3u * n * sizeof(cm1::ReconWriteJob);
  total += 3u * n * sizeof(std::int32_t);
  total += sizeof(int);  // tx_errors
  total += srch::luma_search_scratch_bytes(
      kSearchMaxBlocks, kSearchMaxBlockArea, kSearchCandidatesPerChunk);
  total += mi_cells * sizeof(deblock::DeblockMi) * 3u;
  total += mi_cells + chroma_mi * 2u;  // decoded[3]
  total += mi_cells * sizeof(int);     // intra_joint
  total += chroma_mi;                  // uv_smooth
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
  return buffers.filter.y_bytes >= luma_bytes &&
         buffers.filter.u_bytes >= chroma_bytes &&
         buffers.filter.v_bytes >= chroma_bytes &&
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
         buffers.uv_smooth_bytes >= chroma_mi;
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
    out.luma_jobs_bytes = n * sizeof(im::IntraJob);
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

    out.luma_blocks = static_cast<srch::LumaSearchBlock*>(
        bump(out.luma_blocks_bytes, 256));
    out.luma_results = static_cast<srch::LumaSearchResult*>(
        bump(out.luma_results_bytes, 256));
    out.luma_jobs =
        static_cast<im::IntraJob*>(bump(out.luma_jobs_bytes, 256));
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
    for (int p = 0; p < 3; ++p) {
      out.deblock_mi[p] = static_cast<deblock::DeblockMi*>(
          bump(out.deblock_mi_bytes[p], 256));
      out.decoded[p] = static_cast<std::uint8_t*>(
          bump(out.decoded_bytes[p], 256));
    }
    out.intra_joint = static_cast<int*>(bump(out.intra_joint_bytes, 256));
    out.uv_smooth = static_cast<std::uint8_t*>(bump(out.uv_smooth_bytes, 256));
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
         out.chroma_results != nullptr && out.eob != nullptr &&
         out.tx_jobs != nullptr && out.recon_jobs != nullptr &&
         out.recon_infos != nullptr && out.search_scratch != nullptr &&
         out.tx_errors != nullptr && out.deblock_mi[0] != nullptr &&
         out.deblock_mi[1] != nullptr && out.deblock_mi[2] != nullptr &&
         out.decoded[0] != nullptr && out.decoded[1] != nullptr &&
         out.decoded[2] != nullptr && out.intra_joint != nullptr &&
         out.uv_smooth != nullptr;
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
}

::av2enc::still::StillConfig make_still_config(const GpuOptions& options,
                                      const ::av2enc::still::StillPicture& picture) {
  ::av2enc::still::StillConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_intra_edge_filter =
      options.enable_edge_filter && std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr;
  config.enable_cfl_intra = options.enable_cfl;
  config.enable_chroma_dctonly = true;
  config.tx_mode_select = false;
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u = options.enable_deblock;
  config.apply_deblock_v = options.enable_deblock;
  return config;
}

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
  if (blocks <= 0 || buffers.tx_jobs == nullptr) {
    error = "gpu: device buffers are not prepared";
    return GpuRunStatus::kError;
  }
  for (const BlockPlan& block : plan.blocks) {
    if (block.has_chroma == 0 || block.chroma_w < 4 || block.chroma_h < 4) {
      error = "unsupported: chroma-less M1 block";
      return GpuRunStatus::kUnsupported;
    }
  }
  if (std::getenv("AV2ENC_STILL_FORCE_Y") != nullptr ||
      std::getenv("AV2ENC_STILL_FORCE_Y2") != nullptr) {
    error = "unsupported: AV2ENC_STILL_FORCE_Y* debug override";
    return GpuRunStatus::kUnsupported;
  }

  // ---- availability counts (scheduler replay, host-only) ----------------
  sch::Plan wave;
  if (!sch::build_still_plan(plan, nullptr, &wave, error)) {
    error = "gpu: scheduler plan failed: " + error;
    return GpuRunStatus::kError;
  }
  std::vector<int> n4_above_right(static_cast<std::size_t>(blocks), 0);
  std::vector<int> n4_below_left(static_cast<std::size_t>(blocks), 0);
  for (const sch::PlaneJob& job : wave.jobs()) {
    if (job.plane == 0 && job.block >= 0 && job.block < blocks) {
      n4_above_right[static_cast<std::size_t>(job.block)] = job.n4_above_right;
      n4_below_left[static_cast<std::size_t>(job.block)] = job.n4_below_left;
    }
  }

  // ---- static descriptor arrays -----------------------------------------
  std::vector<srch::LumaSearchBlock> luma_blocks(static_cast<std::size_t>(blocks));
  std::vector<im::IntraJob> luma_jobs(static_cast<std::size_t>(blocks));
  std::vector<StillTxJob> tx_jobs(static_cast<std::size_t>(blocks) * 3u);
  std::vector<cm1::ReconWriteJob> recon_jobs(static_cast<std::size_t>(blocks) * 3u);
  std::vector<cm1::ReconBlockInfo> recon_infos(static_cast<std::size_t>(blocks));
  std::vector<cm1::StillChromaJob> chroma_jobs(static_cast<std::size_t>(blocks));

  for (int i = 0; i < blocks; ++i) {
    const BlockPlan& block = plan.blocks[static_cast<std::size_t>(i)];
    const int x0 = block.mi_col * 4;
    const int y0 = block.mi_row * 4;
    const int cx = block.mi_col * 2;
    const int cy = block.mi_row * 2;

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
    search.num4_above_right = n4_above_right[static_cast<std::size_t>(i)];
    search.num4_below_left = n4_below_left[static_cast<std::size_t>(i)];
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

    if (!init_tx(tx_jobs[static_cast<std::size_t>(i) * 3u], block.luma_w,
                 block.luma_h, static_cast<int>(picture.width), x0, y0,
                 static_cast<int>(picture.width),
                 static_cast<int>(picture.height),
                 static_cast<int>(block.pred_offset_y),
                 static_cast<int>(block.residual_offset_y),
                 static_cast<int>(block.coeff_offset_y), i * 3)) {
      return GpuRunStatus::kUnsupported;
    }
    if (!init_tx(tx_jobs[static_cast<std::size_t>(i) * 3u + 1u], block.chroma_w,
                 block.chroma_h, chroma_vis_w, cx, cy, chroma_vis_w,
                 chroma_vis_h,
                 static_cast<int>(block.pred_offset_u),
                 static_cast<int>(block.residual_offset_u),
                 static_cast<int>(block.coeff_offset_u), i * 3 + 1)) {
      return GpuRunStatus::kUnsupported;
    }
    if (!init_tx(tx_jobs[static_cast<std::size_t>(i) * 3u + 2u], block.chroma_w,
                 block.chroma_h, chroma_vis_w, cx, cy, chroma_vis_w,
                 chroma_vis_h,
                 static_cast<int>(block.pred_offset_v),
                 static_cast<int>(block.residual_offset_v),
                 static_cast<int>(block.coeff_offset_v), i * 3 + 2)) {
      return GpuRunStatus::kUnsupported;
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
    info.chroma_w4 = block.chroma_w / 4;
    info.chroma_h4 = block.chroma_h / 4;
    info.joint = 0;
    info.uv_smooth = 0;
    info.qindex = options.base_q_idx;
    const auto luma_params =
        txv::make_tx_params(block.luma_w, block.luma_h, txv::TxType::kDctDct);
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
    chroma.sb_top_y = (y0 / 64) * 64;
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
    chroma.x = cx;
    chroma.y = cy;
    chroma.w = block.chroma_w;
    chroma.h = block.chroma_h;
    chroma.coded_width = chroma_coded_w;
    chroma.coded_height = chroma_coded_h;
  }

  // ---- upload ------------------------------------------------------------
  const std::size_t src_y_bytes =
      static_cast<std::size_t>(picture.width) * picture.height;
  const std::size_t src_c_bytes =
      static_cast<std::size_t>(chroma_vis_w) * chroma_vis_h;
  const std::size_t luma_blocks_bytes =
      luma_blocks.size() * sizeof(srch::LumaSearchBlock);
  const std::size_t luma_jobs_bytes = luma_jobs.size() * sizeof(im::IntraJob);
  const std::size_t tx_jobs_bytes = tx_jobs.size() * sizeof(StillTxJob);
  const std::size_t recon_jobs_bytes =
      recon_jobs.size() * sizeof(cm1::ReconWriteJob);
  const std::size_t recon_infos_bytes =
      recon_infos.size() * sizeof(cm1::ReconBlockInfo);
  // §7.19 CCSO metadata: per-plane unit-enable grid + decoded offset table.
  // The decoded tables come from the CPU reference's parameters so the GPU
  // kernel sees exactly the values the bitstream signals.
  std::size_t ccso_blk_bytes[3] = {0, 0, 0};
  std::size_t ccso_off_bytes[3] = {0, 0, 0};
  std::int16_t ccso_offset_tables[3][576] = {};
  const std::size_t cdef_idx_bytes =
      cdef_active ? cpu_plan->cdef_idx.size() : 0;
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
                                   tx_jobs_bytes + recon_jobs_bytes +
                                   recon_infos_bytes + ccso_blk_bytes[0] +
                                   ccso_blk_bytes[1] + ccso_blk_bytes[2] +
                                   ccso_off_bytes[0] + ccso_off_bytes[1] +
                                   ccso_off_bytes[2] + cdef_idx_bytes;
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
  std::memcpy(upload + up, tx_jobs.data(), tx_jobs_bytes);
  const std::uint8_t* host_tx_jobs = upload + up;
  up += tx_jobs_bytes;
  std::memcpy(upload + up, recon_jobs.data(), recon_jobs_bytes);
  const std::uint8_t* host_recon_jobs = upload + up;
  up += recon_jobs_bytes;
  std::memcpy(upload + up, recon_infos.data(), recon_infos_bytes);
  const std::uint8_t* host_recon_infos = upload + up;
  up += recon_infos_bytes;
  const std::uint8_t* host_ccso_blks[3] = {nullptr, nullptr, nullptr};
  const std::int16_t* host_ccso_offsets[3] = {nullptr, nullptr, nullptr};
  const std::int8_t* host_cdef_idx = nullptr;
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
      !enqueue_upload(buffers.tx_jobs, host_tx_jobs, tx_jobs_bytes) ||
      !enqueue_upload(buffers.recon_jobs, host_recon_jobs, recon_jobs_bytes) ||
      !enqueue_upload(buffers.recon_infos, host_recon_infos,
                      recon_infos_bytes) ||
      cudaMemsetAsync(buffers.tx_errors, 0, sizeof(int), stream) !=
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
    gates.flags |= srch::kGateDcOnly;
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

  StillTxBuffers tx_buffers;
  tx_buffers.src = buffers.src.y;
  tx_buffers.pred = buffers.pred_packed[0];
  tx_buffers.levels = buffers.coeff.y;
  tx_buffers.residual_out = buffers.residual_packed[0];
  tx_buffers.eob = buffers.eob;
  tx_buffers.error_count = buffers.tx_errors;

  // ---- ordered block loop ----------------------------------------------
  int last_sb_row = -1;
  int last_sb_col = -1;
  for (int i = 0; i < blocks; ++i) {
    const BlockPlan& block = plan.blocks[static_cast<std::size_t>(i)];
    const int sb_row = (block.mi_row / 16) * 16;
    const int sb_col = (block.mi_col / 16) * 16;
    if (sb_row != last_sb_row || sb_col != last_sb_col) {
      if (cm1::recon_begin_sb(state, sb_row, sb_col, stream) != cudaSuccess) {
        error = "gpu: recon_begin_sb failed";
        return GpuRunStatus::kError;
      }
      last_sb_row = sb_row;
      last_sb_col = sb_col;
    }

    prepare_search_block_kernel<<<1, 1, 0, stream>>>(
        buffers.luma_blocks, buffers.intra_joint, mi_cols, mi_rows, i);
    if (srch::luma_search(buffers.luma_blocks + i, 1, gates, search_scratch,
                          buffers.luma_results + i, nullptr,
                          stream) != cudaSuccess) {
      error = "gpu: luma search failed";
      return GpuRunStatus::kError;
    }
    prepare_prediction_kernel<<<1, 1, 0, stream>>>(
        buffers.luma_jobs, buffers.luma_results, buffers.luma_blocks, i);
    if (im::intra_predict_batch(buffers.luma_jobs + i, 1, stream) !=
        cudaSuccess) {
      error = "gpu: chosen-mode prediction failed";
      return GpuRunStatus::kError;
    }

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
    if (cm1::recon_write_blocks(
            buffers.recon_jobs + static_cast<std::size_t>(i) * 3u, 1,
            stream) != cudaSuccess) {
      error = "gpu: luma reconstruction failed";
      return GpuRunStatus::kError;
    }

    if (cm1::chroma_decide(chroma_jobs[static_cast<std::size_t>(i)],
                           chroma_options, buffers.chroma_results + i,
                           stream) != cudaSuccess) {
      error = "gpu: chroma decision failed";
      return GpuRunStatus::kError;
    }

    for (int p = 1; p <= 2; ++p) {
      tx_buffers.src = (p == 1) ? buffers.src.u : buffers.src.v;
      tx_buffers.pred = buffers.pred_packed[p];
      tx_buffers.levels = (p == 1) ? buffers.coeff.u : buffers.coeff.v;
      tx_buffers.residual_out = buffers.residual_packed[p];
      tx_buffers.eob = buffers.eob;
      if (still_tx_batch(
              buffers.tx_jobs + static_cast<std::size_t>(i) * 3u + p, 1,
              tx_buffers, 0, stream) != cudaSuccess) {
        error = "gpu: chroma TX failed";
        return GpuRunStatus::kError;
      }
    }
    if (cm1::recon_write_blocks(
            buffers.recon_jobs + static_cast<std::size_t>(i) * 3u + 1u, 2,
            stream) != cudaSuccess) {
      error = "gpu: chroma reconstruction failed";
      return GpuRunStatus::kError;
    }

    prepare_commit_kernel<<<1, 1, 0, stream>>>(
        buffers.recon_infos, buffers.luma_results, i);
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
    core.ccso_luma_size_log2 = 6;
    core.sb_width = 64;
    core.sb_height = 64;
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
  const std::size_t download_total =
      recon_y_bytes + recon_c_bytes * 2u + coeff_y_bytes + coeff_u_bytes +
      coeff_v_bytes + eob_bytes + luma_result_bytes + chroma_result_bytes +
      sizeof(int);
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
  const U8Planes& out_planes = filtered ? buffers.filter : buffers.recon;
  if (!enqueue_download(out_planes.y, recon_y_bytes) ||
      !enqueue_download(out_planes.u, recon_c_bytes) ||
      !enqueue_download(out_planes.v, recon_c_bytes) ||
      !enqueue_download(buffers.coeff.y, coeff_y_bytes) ||
      !enqueue_download(buffers.coeff.u, coeff_u_bytes) ||
      !enqueue_download(buffers.coeff.v, coeff_v_bytes) ||
      !enqueue_download(buffers.eob, eob_bytes) ||
      !enqueue_download(buffers.luma_results, luma_result_bytes) ||
      !enqueue_download(buffers.chroma_results, chroma_result_bytes) ||
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
                     luma_result_bytes + chroma_result_bytes;
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

  decisions.assign(static_cast<std::size_t>(blocks), BlockDecision{});
  for (int i = 0; i < blocks; ++i) {
    BlockDecision& decision = decisions[static_cast<std::size_t>(i)];
    const srch::LumaSearchResult& luma =
        luma_results[static_cast<std::size_t>(i)];
    const cm1::StillChromaResult& chroma =
        chroma_results[static_cast<std::size_t>(i)];
    decision.luma_mode = static_cast<std::uint8_t>(luma.mode);
    decision.luma_angle_delta = static_cast<std::int8_t>(luma.angle_delta);
    decision.sse = (luma.sse > 0 &&
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
    decision.tx_type_y = 0;
    decision.tx_type_u = 0;
    decision.tx_type_v = 0;
    decision.eob_y = static_cast<std::uint16_t>(
        std::max<std::int32_t>(0, eob[static_cast<std::size_t>(i) * 3u]));
    decision.eob_u = static_cast<std::uint16_t>(std::max<std::int32_t>(
        0, eob[static_cast<std::size_t>(i) * 3u + 1u]));
    decision.eob_v = static_cast<std::uint16_t>(std::max<std::int32_t>(
        0, eob[static_cast<std::size_t>(i) * 3u + 2u]));
  }

  // ---- GPU entropy assembly --------------------------------------------
  still_stream::GpuCoefficients coeffs{host.coeff_y, host.coeff_u, host.coeff_v};
  // With an in-loop filter the CPU reference's exact header configuration is
  // reused: the filter parameters are a planning output, so the GPU cannot
  // re-derive them independently without risking a header mismatch.
  const ::av2enc::still::StillConfig config =
      (filter_config != nullptr) ? *filter_config
                                 : make_still_config(options, picture);
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
  }
  std::vector<std::uint8_t> frame;
  std::size_t tile_bytes = 0;
  if (!still_stream::assemble_gpu_frame(result.plan, tree, result.blocks, decisions,
                                    coeffs, config, frame, error,
                                    &tile_bytes, grids)) {
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
                            options.ccso_edge || options.enable_grain;
  ::av2enc::still::StillResult cpu;
  bool cpu_ready = false;
  if (filter_tools) {
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
    // same pinned upload buffer.
    const std::uint64_t n64_cols =
        (static_cast<std::uint64_t>(result.plan.mi_cols) + 15u) / 16u;
    const std::uint64_t n64_rows =
        (static_cast<std::uint64_t>(result.plan.mi_rows) + 15u) / 16u;
    need_upload += n64_cols * n64_rows;                       // cdef_idx
    need_upload += 3u * n64_cols * n64_rows;                  // ccso_blks
    need_upload += 3u * 576u * sizeof(std::int16_t);          // ccso offsets
  }

#if AV2ENC_STILL_GPU_HAVE_FULL
  if (gpu_requested) {
    const std::size_t n = result.plan.block_count;
    need_device += static_cast<std::uint64_t>(full_device_bytes(result.plan));
    need_upload +=
        n * (sizeof(srch::LumaSearchBlock) + sizeof(im::IntraJob) +
             3u * sizeof(StillTxJob) + 3u * sizeof(cm1::ReconWriteJob) +
             sizeof(cm1::ReconBlockInfo));
    need_download +=
        3u * n * sizeof(std::int32_t) +
        n * (sizeof(srch::LumaSearchResult) + sizeof(cm1::StillChromaResult)) +
        sizeof(int);
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
          filter_tools && cpu_ready ? &cpu : nullptr;
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
            (options.enable_ccso || options.ccso_edge ? 1 : 0);  // §7.19
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
        if (options.verbose) {
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
        (options.enable_cdef || options.enable_ccso || options.ccso_edge)
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
        (options.enable_cdef || options.enable_ccso || options.ccso_edge)
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
        (options.enable_cdef || options.enable_ccso || options.ccso_edge)
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
