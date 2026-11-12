#ifndef AV2ENC_CUDA_WARP_STACK_WARP_STACK_TYPES_H_
#define AV2ENC_CUDA_WARP_STACK_WARP_STACK_TYPES_H_

// Shared POD types for the CUDA warp-stack component.
//
// This header is deliberately free of <cuda_runtime.h> so the same job and
// result structs can be used by:
//   * the CUDA translation unit (`warp_stack.cu`),
//   * the host emulation of the shared device header (bit-exact CPU side),
//   * the GPU parity test.
//
// `WarpPredictJob` carries every input the §7.13.3.19/.20 predictors need. All
// pointers inside a job are *device* pointers when the job is uploaded to the
// GPU; during host emulation the caller substitutes host pointers.
//
// Model provenance ("6-param and derived models"):
//   * `params` is a Q16 6-parameter model, either supplied directly or taken
//     from `WarpCornerResult::corner.wmmat` / `WarpEstimateJob` output;
//   * §7.13.3.23 `reduce_warp_model` and `get_warp_translation` live in the
//     shared `advanced/warp_common.h` and are called from the device header,
//     so a reduced/clamped model is bit-identical on both sides.
//
// Stack *ordering* never happens here: the CPU builder consumes the corner and
// sample tables this module produces and still calls the existing
// `inter::warp::insert_warp_candidate()` in order.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "codec/inter/advanced/warp.h"
#include "codec/inter/advanced/warp_common.h"
#include "codec/inter/warp_stack/warp_stack.h"

namespace av2enc::cuda::warpstack {

namespace ia = inter::adv;

// One CUDA block per prediction job. 128 threads is enough to saturate the
// 8x8/4x4 section work of a 64x64 block without spilling shared memory.
inline constexpr int kPredictThreads = 128;
// Shared intermediate sized for the extended warp (64 rows * 4 columns); the
// block-warp layout (15 * 8) fits inside the same array.
inline constexpr int kWarpIntermediateInts = ia::kExtWarpRows * ia::kExtWarpCols;
// `Arena::kAlignment`; duplicated here so this header stays CUDA-free.
inline constexpr std::size_t kWarpStackAlignment = 256;

// ---------------------------------------------------------------------------
// §7.13.3.19 block warp / §7.13.3.20 extended warp prediction
// ---------------------------------------------------------------------------

// `path` selects the predictor:
//   0 = auto: §7.13.3.14's block->extended fallback, exactly
//       adv::warp_needs_extended() (!shearValid || w < 8 || h < 8 || scaled);
//   1 = force block warp (an invalid shear then writes nothing, matching
//       adv::warp_predict());
//   2 = force extended warp.
inline constexpr int kWarpPathAuto = 0;
inline constexpr int kWarpPathBlock = 1;
inline constexpr int kWarpPathExtended = 2;

struct WarpPredictJob {
  ia::WarpParams params{};  // Q16 6-parameter model
  const void* ref = nullptr;  // uint8_t* when geom.bit_depth == 8, else uint16_t*
  int ref_stride = 0;
  int ref_max_x = 0;  // inclusive padded reference bounds (§7.13.3.19 lastX)
  int ref_max_y = 0;
  ia::WarpBlockGeometry geom{};
  int compound = 0;  // 0 -> clip to geom.bit_depth; 1 -> 1/16 sample precision
  int path = kWarpPathAuto;
  std::int16_t* dst = nullptr;
  int dst_stride = 0;
};

// Filled by the kernel; lets a caller verify the fallback decision and the
// invalid-shear rejection without re-deriving them on the host.
struct WarpPredictResult {
  int path = 0;         // 1 block, 2 extended, 0 not run
  int sections = 0;     // sections actually written
  int shear_valid = 0;  // adv::setup_shear(params).valid
  int status = 0;       // 0 ok, 1 invalid job (nothing written)
};

// ---------------------------------------------------------------------------
// Corner point generation (§7.12.2.3/.4)
// ---------------------------------------------------------------------------

struct WarpCornerJob {
  inter::warp::WarpGrid grid{};  // `mi` is a device pointer in GPU use
  int sb_h4 = 16;
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  std::int8_t ref_frame0 = inter::warp::kRefNone;
  int iter = 0;  // the `adjust_col` argument of generate_points_from_corners
};

struct WarpCornerResult {
  inter::warp::CornerResult corner{};
  int model_valid = 0;  // 1 when a reduced model was built (CPU inserts it)
};

// ---------------------------------------------------------------------------
// Warp sample collection (§7.12.3)
// ---------------------------------------------------------------------------

struct WarpSampleJob {
  inter::warp::WarpSampleParams params{};
};

// ---------------------------------------------------------------------------
// Least-squares model derivation (§7.13.3.23)
// ---------------------------------------------------------------------------

struct WarpEstimateJob {
  inter::warp::WarpSampleSet samples{};
  int mid_x = 0;
  int mid_y = 0;
  int block_mv[2] = {0, 0};
};

// ---------------------------------------------------------------------------
// Arena sizing
// ---------------------------------------------------------------------------

// One workspace covers a whole batch epoch. Counts are per stage; a zero count
// costs nothing. `max_mi_cells` reserves an optional device WarpMi grid so the
// sample/corner tables can point at arena memory instead of a caller buffer.
struct WarpStackConfig {
  int max_predict_jobs = 0;
  int max_corner_jobs = 0;
  int max_sample_jobs = 0;
  int max_estimate_jobs = 0;
  int max_mi_cells = 0;
};

// The arena tables are filled with cudaMemcpy, so every descriptor/result must
// be trivially copyable. This also keeps host and device layouts identical.
static_assert(std::is_trivially_copyable_v<WarpPredictJob>);
static_assert(std::is_trivially_copyable_v<WarpPredictResult>);
static_assert(std::is_trivially_copyable_v<WarpCornerJob>);
static_assert(std::is_trivially_copyable_v<WarpCornerResult>);
static_assert(std::is_trivially_copyable_v<WarpSampleJob>);
static_assert(std::is_trivially_copyable_v<WarpEstimateJob>);
static_assert(std::is_trivially_copyable_v<inter::warp::WarpSampleSet>);
static_assert(std::is_trivially_copyable_v<ia::WarpParams>);
static_assert(std::is_trivially_copyable_v<inter::warp::WarpMi>);

struct WarpStackPlan {
  std::uint64_t predict_jobs = 0;
  std::uint64_t predict_results = 0;
  std::uint64_t corner_jobs = 0;
  std::uint64_t corner_results = 0;
  std::uint64_t sample_jobs = 0;
  std::uint64_t sample_sets = 0;
  std::uint64_t estimate_jobs = 0;
  std::uint64_t estimate_params = 0;
  std::uint64_t mi_grid = 0;
  std::uint64_t total = 0;
};

namespace detail {

inline std::uint64_t table_bytes_impl(std::uint64_t count, std::uint64_t element_size) {
  if (count == 0) {
    return 0;
  }
  const std::uint64_t raw = count * element_size;
  const std::uint64_t mask = static_cast<std::uint64_t>(kWarpStackAlignment) - 1;
  return (raw + mask) & ~mask;
}

template <typename T>
inline std::uint64_t table_bytes(int count) {
  static_assert(alignof(T) <= kWarpStackAlignment,
                "Arena rounds to max(alignof(T), kAlignment); the plan must too");
  if (count <= 0) {
    return 0;
  }
  return table_bytes_impl(static_cast<std::uint64_t>(count), sizeof(T));
}

}  // namespace detail

// Per-table byte plan. This is exactly what `Workspace::create()` allocates, in
// the same order, with `Arena`'s 256-byte rounding.
inline WarpStackPlan plan_device_bytes(const WarpStackConfig& config) {
  WarpStackPlan plan;
  plan.predict_jobs = detail::table_bytes<WarpPredictJob>(config.max_predict_jobs);
  plan.predict_results = detail::table_bytes<WarpPredictResult>(config.max_predict_jobs);
  plan.corner_jobs = detail::table_bytes<WarpCornerJob>(config.max_corner_jobs);
  plan.corner_results = detail::table_bytes<WarpCornerResult>(config.max_corner_jobs);
  plan.sample_jobs = detail::table_bytes<WarpSampleJob>(config.max_sample_jobs);
  plan.sample_sets =
      detail::table_bytes<inter::warp::WarpSampleSet>(config.max_sample_jobs);
  plan.estimate_jobs = detail::table_bytes<WarpEstimateJob>(config.max_estimate_jobs);
  plan.estimate_params = detail::table_bytes<ia::WarpParams>(config.max_estimate_jobs);
  plan.mi_grid = detail::table_bytes<inter::warp::WarpMi>(config.max_mi_cells);
  plan.total = plan.predict_jobs + plan.predict_results + plan.corner_jobs +
               plan.corner_results + plan.sample_jobs + plan.sample_sets +
               plan.estimate_jobs + plan.estimate_params + plan.mi_grid;
  return plan;
}

// Bytes a workspace needs for the configured batch envelope. This is the
// "largest launch" superset: descriptors + result tables + warp sample tables
// + optional MI grid. A caller sizes the Arena from this value plus whatever
// input/output planes it also needs.
inline std::uint64_t required_device_bytes(const WarpStackConfig& config) {
  return plan_device_bytes(config).total;
}

}  // namespace av2enc::cuda::warpstack

#endif  // AV2ENC_CUDA_WARP_STACK_WARP_STACK_TYPES_H_
