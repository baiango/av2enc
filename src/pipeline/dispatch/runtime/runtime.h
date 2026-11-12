// Multi-GPU runtime: device selection, per-GPU arenas, tile dispatch, merge,
// and reference exchange, built on `src/multi_gpu/planner.*`.
//
// This layer does not encode. A caller supplies a FrameWorker, which writes
// one compressed payload per local tile and a reconstruction region into the
// device buffers described by GpuFrameContext. The runtime schedules uploads,
// compute streams, downloads, merge, and reference fill; it never reorders the
// bitstream away from raster TileNum order.
//
// Hardware access is through DeviceBackend. On a CUDA host use
// make_cuda_backend() (cuda_backend.h); on a CPU-only host link the mock
// backend from tests/. No public header includes cuda_runtime.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../planner.h"
#include "device_backend.h"
#include "dispatch.h"
#include "reference.h"

namespace av2enc::multigpu::runtime {

inline constexpr std::uint64_t kDefaultDeviceHeadroomBytes =
    512ull * 1024ull * 1024ull;
inline constexpr std::uint32_t kDefaultPayloadStrideBytes = 1u << 20;

struct RuntimeConfig {
  // Empty means 0..planner.gpu_count-1. Physical CUDA ordinals otherwise.
  std::vector<int> gpu_ordinals;
  // Already queried/dropped devices can be excluded during a deterministic
  // re-plan (used by Runtime::initialize on fallback).
  std::vector<int> excluded_ordinals;

  PlannerConfig planner;
  // Required when auto_layout is true.
  int width = 0;
  int height = 0;
  bool auto_layout = true;
  // Used when auto_layout is false; must satisfy the planner's geometry rules.
  tiles::TileLayout layout;

  int min_gpus = 1;
  bool require_homogeneous = true;
  // Both are startup-only policies. Never applied in the middle of a frame.
  bool allow_drop_missing = false;
  bool allow_drop_failed = false;

  std::uint64_t device_headroom_bytes = kDefaultDeviceHeadroomBytes;
  std::uint64_t vram_cap_bytes = 0;  // 0 = device-derived, no fixed preset
  int arena_init_attempts = 2;       // initial request + halving retries
  std::uint32_t payload_stride_bytes = kDefaultPayloadStrideBytes;
};

// Device-derived arena capacity. There is deliberately no fixed 7,500 MiB
// fallback: capacity = floor_to_MiB(min(total, free) - headroom), optionally
// bounded by `cap_bytes`. A cap that leaves less than 1 MiB is invalid.
struct ArenaCapacity {
  bool valid = false;
  std::string error;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_bytes = 0;
  std::uint64_t device_derived_bytes = 0;
  std::uint64_t cap_bytes = 0;
  std::uint64_t capacity_bytes = 0;
  bool capped = false;
};

ArenaCapacity compute_arena_capacity(const DeviceProperties& device,
                                     std::uint64_t headroom_bytes,
                                     std::uint64_t cap_bytes);

struct DeviceSlot {
  int ordinal = -1;
  DeviceProperties props;
  ArenaCapacity arena;
  double required_live_mb = 0.0;
  double capacity_mb = 0.0;
  std::string note;
};

struct Selection {
  bool valid = false;
  std::string error;
  std::vector<DeviceSlot> slots;
  std::vector<int> requested_ordinals;
  std::vector<int> dropped_ordinals;
  std::vector<std::string> warnings;
  tiles::TileLayout layout;
  MultiGpuPlan plan;
  bool fallback_used = false;
};

// Query devices, enforce homogeneity, choose the default layout for the
// surviving GPU count, run plan_multi_gpu(), and verify each GPU's planned live
// VRAM against its own device-derived capacity.
//
// With allow_drop_missing/allow_drop_failed the first unusable device is dropped
// deterministically (request order) and the plan is rebuilt for the remaining
// GPUs. Dropping below min_gpus fails. Without those flags the function fails
// with the offending device ordinal; it never silently shrinks the run.
Status select_devices_and_plan(DeviceBackend& backend,
                               const RuntimeConfig& config, Selection& out,
                               std::string* error = nullptr);

struct GpuFrameContext {
  DeviceBackend* backend = nullptr;
  int slot = -1;
  int ordinal = -1;
  int frame_index = 0;
  int reference_index = 0;
  int frame_width = 0;
  int frame_height = 0;
  int bit_depth = 8;
  const tiles::TileLayout* layout = nullptr;

  StreamHandle compute_stream = kInvalidStream;
  StreamHandle copy_stream = kInvalidStream;

  // GPU-local tiles in raster order. Slot i's payload goes to
  // device_payload + i * payload_stride_bytes.
  const int* tile_indices = nullptr;
  int tile_count = 0;
  std::uint32_t payload_stride_bytes = 0;

  // Full-frame device mirrors owned by the GPU arena. `device_reference` points
  // at the current reference slot; for frame 0 there is no previous
  // reconstruction, so its contents are unspecified.
  std::uint8_t* device_source = nullptr;
  std::uint8_t* device_payload = nullptr;
  std::uint8_t* device_recon = nullptr;
  std::uint8_t* device_reference = nullptr;

  // Host array of tile_count entries. The worker writes the compressed byte
  // count for local tile i after encoding it. Values must be in [1, stride].
  std::uint32_t* tile_sizes_out = nullptr;

  std::size_t source_bytes = 0;
  std::size_t recon_bytes = 0;
};

class FrameWorker {
 public:
  virtual ~FrameWorker() = default;
  // Enqueue compute work on ctx.compute_stream. Must not synchronize the
  // device or reorder the raster tile list. On failure return non-kOk and set
  // *error; the runtime aborts the whole frame.
  virtual Status encode(const GpuFrameContext& ctx, std::string* error) = 0;
};

struct FrameRequest {
  int frame_index = 0;
  const void* source = nullptr;  // full 4:2:0 frame, size source_bytes
  std::size_t source_bytes = 0;
  bool capture_recon = false;    // copy the assembled host frame into result
};

struct FrameResult {
  bool valid = false;
  std::string error;
  int frame_index = -1;
  int failed_slot = -1;
  int tile_count = 0;
  // Strict raster TileNum order.
  std::vector<std::vector<std::uint8_t>> tile_payloads;
  std::vector<std::uint32_t> tile_sizes;
  std::uint64_t payload_bytes = 0;
  // Source upload + payload D2H + recon gather D2H + reference fill H2D.
  std::uint64_t host_tx_bytes = 0;
  std::uint64_t reference_fill_bytes = 0;
  bool reference_fill_pending = false;
  std::vector<std::uint8_t> recon;  // only when capture_recon
};

struct GpuRuntimeInfo {
  int slot = -1;
  int ordinal = -1;
  int tile_count = 0;
  std::uint64_t arena_capacity_bytes = 0;
  std::uint64_t arena_used_bytes = 0;
  std::uint64_t device_source_bytes = 0;
  std::uint64_t device_recon_bytes = 0;
  std::uint64_t device_reference_bytes = 0;
  std::uint64_t device_payload_bytes = 0;
  bool reference_fill_pending = false;
};

struct RuntimeStats {
  std::uint64_t frames = 0;
  std::uint64_t host_tx_bytes = 0;
  std::uint64_t merged_payload_bytes = 0;
  std::uint64_t reference_fill_bytes = 0;
  std::uint64_t source_upload_bytes = 0;
  int prefetches = 0;
  int device_drops = 0;
  int fallback_replans = 0;
};

class Runtime {
 public:
  Runtime();
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  // Selects devices, initializes one context + arena + two streams + events per
  // GPU, allocates the persistent device buffers and pinned host staging, and
  // builds the dispatch table. On failure nothing is left initialized.
  Status initialize(DeviceBackend& backend, RuntimeConfig config,
                    std::string* error = nullptr);
  void shutdown() noexcept;
  bool initialized() const noexcept;

  const Selection& selection() const noexcept;
  const MultiGpuPlan& plan() const noexcept;
  const DispatchTable& dispatch_table() const noexcept;
  const ReferenceSchedule& reference_schedule() const noexcept;
  const std::vector<GpuRuntimeInfo>& gpu_runtime() const noexcept;
  const RuntimeStats& stats() const noexcept;
  bool reference_fill_pending() const noexcept;

  // Optional: enqueue this frame's source upload on every GPU copy stream and
  // record the upload event without dispatching compute. dispatch_frame() then
  // waits on that event instead of re-uploading. Only one prefetch may be
  // outstanding. The source is copied into pinned staging immediately.
  Status prefetch_source(int frame_index, const void* host_source,
                         std::size_t bytes, std::string* error = nullptr);

  // Enqueue and complete the frame through reference fill. Payloads are
  // downloaded and merged in raster order; reconstruction is gathered and the
  // next reference slot is filled. The reference fill itself is left pending
  // and gated into the next frame's compute stream.
  Status dispatch_frame(const FrameRequest& request, FrameWorker& worker,
                        FrameResult& result, std::string* error = nullptr);

  // Wait for pending reference-fill events. Not needed in the per-frame path;
  // use before shutdown or when a caller wants a fully drained timeline.
  Status synchronize(std::string* error = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace av2enc::multigpu::runtime
