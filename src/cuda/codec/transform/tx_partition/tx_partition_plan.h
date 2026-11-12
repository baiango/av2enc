// Host-side plan/arena layer for the CUDA TX_MODE_SELECT transform partition.
//
// Every device byte used by this component comes from the caller's persistent
// `av2enc::cuda::Arena`.  There is no direct device allocation anywhere in the
// module and no allocation inside a launcher.  Sizing is explicit:
//
//   const LaunchConfig config{max_plans};
//   const std::uint64_t bytes = required_device_bytes(config);
//   // caller: arena.release(); arena.init(bytes + slack);
//   Workspace ws;
//   Status s = ws.init(arena, config, &bytes, &error);
//
// `Workspace` holds the one-time persistent tables (the [16][25] TxParams table
// and the 25x3 scan table).  Each launch asks `acquire_scratch()` for the plan,
// job and resolved-part arrays, then `release_scratch()` rewinds the arena to
// its mark.  The files in this directory only ever call `allocate_array()` /
// `allocate_checked()` / `mark()` / `rewind()` on the arena.
//
// A short arena returns `Status::kArenaExhausted` with `required_bytes` set to
// the exact `required_device_bytes(config)` value, so the caller can grow the
// arena and retry.  There is no allocation fallback.
#ifndef AV2ENC_CUDA_TX_PARTITION_PLAN_H_
#define AV2ENC_CUDA_TX_PARTITION_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/codec/transform/tx_partition/tx_partition_device.cuh"
#include "codec/transform/tx_partition/tx_partition.h"

namespace av2enc::cuda::txpart {

// Largest number of plans (coding-block partition decisions) one launch may
// carry.  Each plan has at most kMaxParts TX parts.  `max_plans` bounds the
// per-launch scratch arrays exactly.
struct LaunchConfig {
  int max_plans = 1;
};

// Exact arena bytes for one Workspace plus the largest launch of `config`.
// Every array is rounded to Arena::kAlignment, matching the bump allocator.
std::uint64_t required_device_bytes(const LaunchConfig& config);

// Split of required_device_bytes() for diagnostics/tests.
std::uint64_t persistent_device_bytes(const LaunchConfig& config);
std::uint64_t scratch_device_bytes(const LaunchConfig& config);

// Persistent device tables.  All pointers are arena-resident and read-only
// after init().  `host_params` is the host mirror used by the emulation test
// and by build_host_jobs(); it is not device memory.
struct Workspace {
  TxParams* d_params = nullptr;                // [kTxTypeCount][kTxSizeCount]
  std::uint16_t* d_scan_data = nullptr;        // [kScanTableCount][kMaxScanArea]
  std::uint16_t* d_scan_length = nullptr;      // [kScanTableCount]
  std::vector<TxParams> host_params;           // host mirror of d_params
  int max_plans = 0;
  std::uint64_t bytes = 0;                     // required_device_bytes(config)
  LaunchConfig config{};

  // Allocates the persistent tables from `arena` exactly once.  Also verifies
  // that `config`'s per-launch scratch fits in the remaining space.  On
  // kArenaExhausted, *required_bytes is the full required_device_bytes(config).
  Status init(Arena& arena, const LaunchConfig& config_in,
              std::uint64_t* required_bytes = nullptr,
              std::string* error = nullptr);

  // Forgets the pointers without freeing anything (the arena owns the memory).
  void reset() noexcept;

  bool valid() const noexcept {
    return max_plans > 0 && d_params != nullptr && d_scan_data != nullptr;
  }

  ScanTableView scan_view() const noexcept {
    ScanTableView view;
    view.data = d_scan_data;
    view.length = d_scan_length;
    return view;
  }
};

// Per-launch arena scratch.  The GPU launchers fill this with the plan upload,
// the device-resolved jobs, and the device-resolved part origins.
inline constexpr std::uint64_t kNoArenaMark = ~static_cast<std::uint64_t>(0);

struct LaunchScratch {
  PlanDesc* d_plans = nullptr;
  TxBlockJob* d_jobs = nullptr;
  ResolvedPart* d_parts = nullptr;
  int plan_capacity = 0;
  int part_capacity = 0;
  std::uint64_t mark = kNoArenaMark;
};

// Bump-allocates per-launch scratch between arena.mark() and the eventual
// release_scratch().  Sizes are plan_count plans and plan_count*4 parts, the
// worst-case partition.  `required_bytes` (when non-null) receives
// required_device_bytes(workspace.config).
Status acquire_scratch(Arena& arena, const Workspace& workspace, int plan_count,
                       LaunchScratch* scratch,
                       std::uint64_t* required_bytes = nullptr,
                       std::string* error = nullptr);

// Rewinds the arena to the mark taken by acquire_scratch().  Safe to call once.
void release_scratch(Arena& arena, LaunchScratch& scratch) noexcept;

// ---------------------------------------------------------------------------
// Host-side plan construction.  These mirror the device geometry bit for bit;
// the tests compare them against the §5.20.6.3 CPU implementation.
// ---------------------------------------------------------------------------

// One partition decision for one <= 64x64 region.  `tx_type` is the luma
// TX_TYPE (0..15) applied to every part of the plan.
struct PlanRequest {
  int mi_size = -1;
  int partition = kPartNone;
  int mi_row = 0;
  int mi_col = 0;
  int tx_type = 0;
};

// A batch of plans with prefix-summed element bases.  `plans` can be uploaded
// verbatim; its `part_base` / `in_base` / `out_base` fields are absolute.
struct PlanBatch {
  std::vector<PlanDesc> plans;
  int total_parts = 0;
  std::int32_t total_adj_elems = 0;
  std::int32_t total_coded_elems = 0;
};

// Validates every request and fills the prefix offsets.  Returns
// kInvalidArgument with a descriptive error for an illegal partition.
Status make_batch(const PlanRequest* requests, int count, PlanBatch* batch,
                  std::string* error = nullptr);

// Convenience overload for one plan.
Status make_batch(const PlanRequest& request, PlanBatch* batch,
                  std::string* error = nullptr);

// Resolves `batch` on the host (same code as the device kernel) into
// absolute-indexed TxBlockJobs.  `capacity` must be >= batch.total_parts.
Status build_host_jobs(const Workspace& workspace, const PlanBatch& batch,
                       TxBlockJob* jobs, int capacity,
                       std::string* error = nullptr);

// Same for the resolved part origins.
Status build_host_parts(const PlanBatch& batch, ResolvedPart* parts,
                        int capacity, std::string* error = nullptr);

// LumaTxSizes / LumaTxMiddle / LumaTxScanOrder builder, identical to the CPU
// driver's set_tx_size() walk.  Grid cells outside the parts stay 0.
av2enc::video::txpartition::LumaTxGrid build_luma_grid(const PlanBatch& batch,
                                            int mi_rows, int mi_cols);

}  // namespace av2enc::cuda::txpart

#endif  // AV2ENC_CUDA_TX_PARTITION_PLAN_H_
