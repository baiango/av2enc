#include "runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

#include "../planner.h"

namespace av2enc::multigpu::runtime {
namespace {

constexpr std::uint64_t kMib = 1024ull * 1024ull;
constexpr std::uint64_t kMinArenaBytes = kMib;
constexpr std::size_t kArenaAlignment = 256;

void set_error(std::string* error, const std::string& text) {
  if (error != nullptr) {
    *error = text;
  }
}

std::uint64_t floor_mib(std::uint64_t bytes) {
  return bytes & ~(kMib - 1);
}

bool same_device_class(const DeviceProperties& a, const DeviceProperties& b) {
  if (a.cc_major != b.cc_major || a.cc_minor != b.cc_minor) {
    return false;
  }
  if (a.sm_count != b.sm_count) {
    return false;
  }
  const std::uint64_t large = std::max(a.total_vram_bytes, b.total_vram_bytes);
  const std::uint64_t small = std::min(a.total_vram_bytes, b.total_vram_bytes);
  if (small == 0 || large == 0) {
    return false;
  }
  // 5% class tolerance: two 3060 12 GB cards pass, a 12 GB / 8 GB mix does not.
  return static_cast<double>(large) <= static_cast<double>(small) * 1.05;
}

bool contains(const std::vector<int>& values, int value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

std::string ordinal_text(int ordinal) {
  return std::to_string(ordinal);
}

bool choose_layout(const RuntimeConfig& config, int gpu_count,
                   tiles::TileLayout& out, std::string& error) {
  if (!config.auto_layout) {
    std::string layout_error;
    if (!tiles::validate_tile_layout(config.layout, layout_error)) {
      error = "explicit layout is invalid: " + layout_error;
      return false;
    }
    if (config.layout.tile_count() < gpu_count) {
      std::ostringstream text;
      text << "explicit layout has " << config.layout.tile_count()
           << " tiles for " << gpu_count << " GPUs";
      error = text.str();
      return false;
    }
    out = config.layout;
    return true;
  }

  if (config.width <= 0 || config.height <= 0) {
    error = "auto layout needs positive width and height";
    return false;
  }

  const tiles::TileLayoutPreset presets[] = {
      default_layout_for_gpus(gpu_count), tiles::TileLayoutPreset::k16x2,
      tiles::TileLayoutPreset::k8x2,     tiles::TileLayoutPreset::k4x2,
      tiles::TileLayoutPreset::k2x2,     tiles::TileLayoutPreset::k2x1,
      tiles::TileLayoutPreset::k1x1,
  };
  for (tiles::TileLayoutPreset preset : presets) {
    const tiles::TileLayoutDims dims = tiles::preset_dims(preset);
    const tiles::LayoutResult result =
        tiles::plan_tile_layout(config.width, config.height, dims.cols,
                                dims.rows);
    if (!result.ok()) {
      continue;
    }
    if (result.layout.tile_count() < gpu_count) {
      continue;
    }
    out = result.layout;
    return true;
  }
  std::ostringstream text;
  text << "no representable default layout at " << config.width << "x"
       << config.height << " for " << gpu_count << " GPUs";
  error = text.str();
  return false;
}

struct GpuState {
  GpuRuntimeInfo info;
  ArenaHandle arena = kInvalidArena;
  StreamHandle compute = kInvalidStream;
  StreamHandle copy = kInvalidStream;
  EventHandle upload_done[2] = {kInvalidEvent, kInvalidEvent};
  EventHandle compute_done = kInvalidEvent;
  EventHandle gather_done = kInvalidEvent;
  EventHandle fill_done = kInvalidEvent;

  std::uint8_t* device_source[2] = {nullptr, nullptr};
  std::uint8_t* device_payload = nullptr;
  std::uint8_t* device_recon = nullptr;
  std::uint8_t* device_references = nullptr;

  std::uint8_t* host_source[2] = {nullptr, nullptr};
  std::uint8_t* host_payload = nullptr;
  std::uint8_t* host_recon = nullptr;

  std::vector<std::uint32_t> tile_sizes;
  bool fill_pending = false;
  bool context_ready = false;
};

void release_gpu(DeviceBackend& backend, GpuState& gpu) {
  // Teardown order: staging, events/streams, arena, context.
  for (std::uint8_t*& ptr : gpu.host_source) {
    if (ptr != nullptr) {
      backend.pinned_free(ptr);
      ptr = nullptr;
    }
  }
  if (gpu.host_payload != nullptr) {
    backend.pinned_free(gpu.host_payload);
    gpu.host_payload = nullptr;
  }
  if (gpu.host_recon != nullptr) {
    backend.pinned_free(gpu.host_recon);
    gpu.host_recon = nullptr;
  }
  if (gpu.upload_done[0] != kInvalidEvent) {
    backend.event_destroy(gpu.upload_done[0]);
    gpu.upload_done[0] = kInvalidEvent;
  }
  if (gpu.upload_done[1] != kInvalidEvent) {
    backend.event_destroy(gpu.upload_done[1]);
    gpu.upload_done[1] = kInvalidEvent;
  }
  if (gpu.compute_done != kInvalidEvent) {
    backend.event_destroy(gpu.compute_done);
    gpu.compute_done = kInvalidEvent;
  }
  if (gpu.gather_done != kInvalidEvent) {
    backend.event_destroy(gpu.gather_done);
    gpu.gather_done = kInvalidEvent;
  }
  if (gpu.fill_done != kInvalidEvent) {
    backend.event_destroy(gpu.fill_done);
    gpu.fill_done = kInvalidEvent;
  }
  if (gpu.compute != kInvalidStream) {
    backend.stream_destroy(gpu.compute);
    gpu.compute = kInvalidStream;
  }
  if (gpu.copy != kInvalidStream) {
    backend.stream_destroy(gpu.copy);
    gpu.copy = kInvalidStream;
  }
  if (gpu.arena != kInvalidArena) {
    backend.arena_release(gpu.arena);
    gpu.arena = kInvalidArena;
  }
  if (gpu.context_ready) {
    backend.context_shutdown(gpu.info.ordinal);
    gpu.context_ready = false;
  }
  gpu = GpuState{};
}

}  // namespace

ArenaCapacity compute_arena_capacity(const DeviceProperties& device,
                                     std::uint64_t headroom_bytes,
                                     std::uint64_t cap_bytes) {
  ArenaCapacity capacity;
  capacity.device_total_bytes = device.total_vram_bytes;
  capacity.device_free_bytes = device.free_vram_bytes;
  capacity.cap_bytes = cap_bytes;

  const std::uint64_t usable =
      std::min(device.total_vram_bytes, device.free_vram_bytes);
  if (usable <= headroom_bytes) {
    std::ostringstream text;
    text << "device " << device.ordinal << " has " << usable
         << " usable bytes, not above the " << headroom_bytes
         << "-byte device headroom";
    capacity.error = text.str();
    return capacity;
  }
  const std::uint64_t derived = floor_mib(usable - headroom_bytes);
  capacity.device_derived_bytes = derived;
  if (derived < kMinArenaBytes) {
    capacity.error = "device-derived arena capacity is below 1 MiB";
    return capacity;
  }

  std::uint64_t effective = derived;
  if (cap_bytes != 0) {
    effective = std::min(effective, cap_bytes);
    effective = floor_mib(effective);
    capacity.capped = cap_bytes < derived;
  }
  if (effective < kMinArenaBytes) {
    capacity.error = "VRAM cap leaves less than 1 MiB for the arena";
    return capacity;
  }
  capacity.capacity_bytes = effective;
  capacity.valid = true;
  return capacity;
}

Status select_devices_and_plan(DeviceBackend& backend,
                               const RuntimeConfig& config, Selection& out,
                               std::string* error) {
  out = Selection{};

  std::vector<int> requested;
  if (!config.gpu_ordinals.empty()) {
    requested = config.gpu_ordinals;
  } else {
    if (config.planner.gpu_count < 1 || config.planner.gpu_count > 8) {
      out.error = "GPU count must be 1..8";
      set_error(error, out.error);
      return Status::kInvalidArgument;
    }
    for (int i = 0; i < config.planner.gpu_count; ++i) {
      requested.push_back(i);
    }
  }
  out.requested_ordinals = requested;
  if (requested.empty() || requested.size() > 8) {
    out.error = "requested GPU list must have 1..8 ordinals";
    set_error(error, out.error);
    return Status::kInvalidArgument;
  }
  if (config.min_gpus < 1 || config.min_gpus > 8) {
    out.error = "min_gpus must be 1..8";
    set_error(error, out.error);
    return Status::kInvalidArgument;
  }
  for (std::size_t i = 0; i < requested.size(); ++i) {
    if (requested[i] < 0) {
      out.error = "GPU ordinals must be non-negative";
      set_error(error, out.error);
      return Status::kInvalidArgument;
    }
    for (std::size_t j = i + 1; j < requested.size(); ++j) {
      if (requested[i] == requested[j]) {
        out.error = "GPU ordinal appears twice in the request";
        set_error(error, out.error);
        return Status::kInvalidArgument;
      }
    }
  }
  if (config.payload_stride_bytes < 2) {
    out.error = "payload stride must be at least 2 bytes";
    set_error(error, out.error);
    return Status::kInvalidArgument;
  }

  int device_count = 0;
  Status status = backend.device_count(&device_count);
  if (!is_ok(status)) {
    out.error = "device enumeration failed";
    set_error(error, out.error);
    return status;
  }

  std::vector<DeviceSlot> candidates;
  for (int ordinal : requested) {
    if (contains(config.excluded_ordinals, ordinal)) {
      out.dropped_ordinals.push_back(ordinal);
      out.warnings.push_back("GPU " + ordinal_text(ordinal) +
                             " excluded by a previous fallback");
      continue;
    }
    if (ordinal >= device_count) {
      if (config.allow_drop_missing) {
        out.dropped_ordinals.push_back(ordinal);
        out.warnings.push_back("GPU " + ordinal_text(ordinal) +
                               " is not present; dropped at startup");
        continue;
      }
      std::ostringstream text;
      text << "GPU " << ordinal << " is not present (" << device_count
           << " device(s) visible)";
      out.error = text.str();
      set_error(error, out.error);
      return Status::kDeviceUnavailable;
    }
    DeviceProperties props;
    props.ordinal = ordinal;
    status = backend.query_device(ordinal, &props);
    if (!is_ok(status)) {
      if (config.allow_drop_failed) {
        out.dropped_ordinals.push_back(ordinal);
        out.warnings.push_back("GPU " + ordinal_text(ordinal) +
                               " could not be queried; dropped at startup");
        continue;
      }
      std::ostringstream text;
      text << "GPU " << ordinal << " query failed: " << status_name(status);
      out.error = text.str();
      set_error(error, out.error);
      return status;
    }
    if (props.ordinal != ordinal) {
      props.ordinal = ordinal;
    }
    DeviceSlot slot;
    slot.ordinal = ordinal;
    slot.props = std::move(props);
    candidates.push_back(std::move(slot));
  }

  if (config.require_homogeneous && candidates.size() > 1) {
    const DeviceProperties& reference = candidates[0].props;
    for (std::size_t i = 1; i < candidates.size();) {
      if (same_device_class(reference, candidates[i].props)) {
        if (reference.name != candidates[i].props.name) {
          out.warnings.push_back(
              "GPU names differ but compute capability, SM count, and VRAM "
              "class match; running as homogeneous");
        }
        ++i;
        continue;
      }
      if (config.allow_drop_failed) {
        out.dropped_ordinals.push_back(candidates[i].ordinal);
        out.warnings.push_back(
            "GPU " + ordinal_text(candidates[i].ordinal) +
            " does not match GPU " + ordinal_text(reference.ordinal) +
            " (compute capability / SM count / VRAM class); dropped at "
            "startup");
        candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(i));
        continue;
      }
      std::ostringstream text;
      text << "GPU " << candidates[i].ordinal << " does not match GPU "
           << reference.ordinal
           << " (compute capability / SM count / VRAM class)";
      out.error = text.str();
      set_error(error, out.error);
      return Status::kDeviceMismatch;
    }
  }

  for (;;) {
    const int gpu_count = static_cast<int>(candidates.size());
    if (gpu_count < config.min_gpus) {
      out.error = "fewer usable GPUs than min_gpus";
      set_error(error, out.error);
      return Status::kNoDevices;
    }

    tiles::TileLayout layout;
    std::string layout_error;
    if (!choose_layout(config, gpu_count, layout, layout_error)) {
      out.error = layout_error;
      set_error(error, out.error);
      return Status::kPlanInvalid;
    }

    PlannerConfig planner_config = config.planner;
    planner_config.gpu_count = gpu_count;
    MultiGpuPlan plan;
    std::string plan_error;
    if (!plan_multi_gpu(layout, planner_config, plan, plan_error)) {
      out.error = "planner rejected the configuration: " + plan_error;
      set_error(error, out.error);
      return Status::kPlanInvalid;
    }

    int unfit = -1;
    Status unfit_status = Status::kOutOfMemory;
    std::string unfit_note;
    for (int slot_index = 0; slot_index < gpu_count; ++slot_index) {
      DeviceSlot& slot = candidates[static_cast<std::size_t>(slot_index)];
      slot.arena = compute_arena_capacity(slot.props,
                                          config.device_headroom_bytes,
                                          config.vram_cap_bytes);
      if (!slot.arena.valid) {
        unfit = slot_index;
        unfit_status = Status::kOutOfMemory;
        unfit_note = slot.arena.error;
        break;
      }
      slot.required_live_mb =
          plan.gpus[static_cast<std::size_t>(slot_index)].vram.live_mb;
      slot.capacity_mb =
          static_cast<double>(slot.arena.capacity_bytes) / static_cast<double>(kMib);
      if (slot.required_live_mb > slot.capacity_mb + 1.0e-9) {
        std::ostringstream text;
        text << "GPU " << slot.ordinal << " needs " << slot.required_live_mb
             << " MiB but its device-derived arena is " << slot.capacity_mb
             << " MiB";
        unfit = slot_index;
        unfit_status = Status::kOutOfMemory;
        unfit_note = text.str();
        break;
      }
    }

    if (unfit < 0) {
      out.slots = std::move(candidates);
      out.layout = std::move(layout);
      out.plan = std::move(plan);
      out.fallback_used = !out.dropped_ordinals.empty();
      out.valid = true;
      if (error != nullptr) {
        error->clear();
      }
      return Status::kOk;
    }

    if (config.allow_drop_failed && gpu_count - 1 >= config.min_gpus) {
      const int bad_ordinal =
          candidates[static_cast<std::size_t>(unfit)].ordinal;
      out.dropped_ordinals.push_back(bad_ordinal);
      out.warnings.push_back("GPU " + ordinal_text(bad_ordinal) +
                             " cannot fit its planned live set (" +
                             unfit_note + "); dropped at startup");
      candidates.erase(candidates.begin() + unfit);
      continue;
    }
    out.error = unfit_note;
    set_error(error, out.error);
    return unfit_status;
  }
}

struct Runtime::Impl {
  DeviceBackend* backend = nullptr;
  RuntimeConfig config;
  Selection selection;
  DispatchTable dispatch;
  ReferenceSchedule reference_schedule;
  std::vector<GpuState> gpus;
  // Contiguous GPU-local tile index arrays handed to FrameWorker.
  std::vector<std::vector<int>> tile_indices;
  // Stable storage for the public per-GPU view.
  std::vector<GpuRuntimeInfo> info;
  RuntimeStats stats;
  bool initialized = false;
  int reference_index = 0;
  int prefetched_frame = -1;
  std::uint64_t frame_bytes = 0;
  std::uint32_t payload_stride = kDefaultPayloadStrideBytes;
  int bit_depth = 8;
  std::uint8_t* host_recon[2] = {nullptr, nullptr};

  void refresh_info() {
    info.clear();
    info.reserve(gpus.size());
    for (const GpuState& gpu : gpus) {
      GpuRuntimeInfo view = gpu.info;
      view.reference_fill_pending = gpu.fill_pending;
      info.push_back(view);
    }
  }

  bool any_fill_pending() const {
    for (const GpuState& gpu : gpus) {
      if (gpu.fill_pending) {
        return true;
      }
    }
    return false;
  }

  void reset() {
    for (GpuState& gpu : gpus) {
      if (gpu.context_ready && backend != nullptr) {
        (void)backend->device_synchronize(gpu.info.ordinal);
      }
      if (backend != nullptr) {
        release_gpu(*backend, gpu);
      }
    }
    gpus.clear();
    for (std::uint8_t*& ptr : host_recon) {
      if (ptr != nullptr && backend != nullptr) {
        backend->pinned_free(ptr);
      }
      ptr = nullptr;
    }
    selection = Selection{};
    dispatch = DispatchTable{};
    reference_schedule = ReferenceSchedule{};
    tile_indices.clear();
    info.clear();
    stats = RuntimeStats{};
    reference_index = 0;
    prefetched_frame = -1;
    frame_bytes = 0;
    payload_stride = kDefaultPayloadStrideBytes;
    bit_depth = 8;
    initialized = false;
  }
};

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {}
Runtime::~Runtime() { shutdown(); }

bool Runtime::initialized() const noexcept { return impl_->initialized; }

const Selection& Runtime::selection() const noexcept { return impl_->selection; }
const MultiGpuPlan& Runtime::plan() const noexcept {
  return impl_->selection.plan;
}
const DispatchTable& Runtime::dispatch_table() const noexcept {
  return impl_->dispatch;
}
const ReferenceSchedule& Runtime::reference_schedule() const noexcept {
  return impl_->reference_schedule;
}
const std::vector<GpuRuntimeInfo>& Runtime::gpu_runtime() const noexcept {
  return impl_->info;
}
const RuntimeStats& Runtime::stats() const noexcept { return impl_->stats; }
bool Runtime::reference_fill_pending() const noexcept {
  return impl_->any_fill_pending();
}

Status Runtime::initialize(DeviceBackend& backend, RuntimeConfig config,
                           std::string* error) {
  if (impl_->initialized) {
    set_error(error, "runtime is already initialized");
    return Status::kAlreadyInitialized;
  }
  impl_->reset();
  impl_->backend = &backend;
  impl_->config = config;

  if (config.payload_stride_bytes < 2) {
    set_error(error, "payload stride must be at least 2 bytes");
    return Status::kInvalidArgument;
  }

  std::vector<int> excluded = config.excluded_ordinals;
  Selection selection;
  std::vector<GpuState> gpus;

  for (;;) {
    RuntimeConfig attempt_config = config;
    attempt_config.excluded_ordinals = excluded;
    std::string select_error;
    Status status =
        select_devices_and_plan(backend, attempt_config, selection, &select_error);
    if (!is_ok(status)) {
      set_error(error, select_error.empty() ? status_name(status) : select_error);
      return status;
    }

    const int gpu_count = static_cast<int>(selection.slots.size());
    std::vector<GpuState> candidate(static_cast<std::size_t>(gpu_count));
    int failed_slot = -1;
    Status failed_status = Status::kOutOfMemory;
    std::string failed_error;

    for (int slot = 0; slot < gpu_count && failed_slot < 0; ++slot) {
      GpuState& gpu = candidate[static_cast<std::size_t>(slot)];
      gpu.info.slot = slot;
      gpu.info.ordinal = selection.slots[static_cast<std::size_t>(slot)].ordinal;
      std::string context_error;
      status = backend.context_init(gpu.info.ordinal, &context_error);
      if (!is_ok(status)) {
        failed_slot = slot;
        failed_status = status;
        failed_error = context_error.empty() ? status_name(status) : context_error;
        break;
      }
      gpu.context_ready = true;

      std::uint64_t capacity =
          selection.slots[static_cast<std::size_t>(slot)].arena.capacity_bytes;
      ArenaHandle arena = kInvalidArena;
      std::string arena_error;
      Status arena_status = Status::kOutOfMemory;
      const int attempts = std::max(1, config.arena_init_attempts);
      for (int attempt = 0; attempt < attempts; ++attempt) {
        arena_status = backend.arena_init(gpu.info.ordinal, capacity, &arena,
                                          &arena_error);
        if (is_ok(arena_status)) {
          break;
        }
        const std::uint64_t next = floor_mib(capacity / 2);
        if (next < kMinArenaBytes || next == capacity) {
          break;
        }
        capacity = next;
      }
      if (!is_ok(arena_status)) {
        failed_slot = slot;
        failed_status = arena_status;
        failed_error = arena_error.empty() ? status_name(arena_status)
                                           : arena_error;
        break;
      }
      gpu.arena = arena;
      gpu.info.arena_capacity_bytes = backend.arena_capacity(arena);
      const double actual_capacity_mb =
          static_cast<double>(gpu.info.arena_capacity_bytes) /
          static_cast<double>(kMib);
      const double required_mb =
          selection.plan.gpus[static_cast<std::size_t>(slot)].vram.live_mb;
      if (actual_capacity_mb + 1.0e-9 < required_mb) {
        failed_slot = slot;
        failed_status = Status::kOutOfMemory;
        std::ostringstream text;
        text << "GPU " << gpu.info.ordinal << " arena shrank to "
             << actual_capacity_mb << " MiB but the planned live set is "
             << required_mb << " MiB";
        failed_error = text.str();
        break;
      }
    }

    if (failed_slot >= 0) {
      for (GpuState& gpu : candidate) {
        if (gpu.context_ready || gpu.arena != kInvalidArena) {
          release_gpu(backend, gpu);
        }
      }
      if (config.allow_drop_failed && gpu_count - 1 >= config.min_gpus) {
        const int bad_ordinal =
            selection.slots[static_cast<std::size_t>(failed_slot)].ordinal;
        excluded.push_back(bad_ordinal);
        impl_->stats.device_drops += 1;
        impl_->stats.fallback_replans += 1;
        continue;
      }
      set_error(error, failed_error);
      return failed_status;
    }

    gpus = std::move(candidate);
    break;
  }

  impl_->selection = std::move(selection);
  impl_->gpus = std::move(gpus);
  impl_->config = config;
  impl_->stats.device_drops = std::max(
      0, static_cast<int>(impl_->selection.requested_ordinals.size()) -
             static_cast<int>(impl_->selection.slots.size()));
  if (!impl_->selection.dropped_ordinals.empty()) {
    impl_->stats.fallback_replans = std::max(impl_->stats.fallback_replans, 1);
  }

  impl_->frame_bytes = impl_->selection.plan.memory.frame_bytes;
  impl_->payload_stride = config.payload_stride_bytes;
  impl_->bit_depth = config.planner.bit_depth;
  if (impl_->frame_bytes == 0) {
    set_error(error, "planner returned a zero frame size");
    impl_->reset();
    return Status::kPlanInvalid;
  }
  if (impl_->frame_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    set_error(error, "frame size does not fit size_t");
    impl_->reset();
    return Status::kInvalidArgument;
  }

  std::vector<int> ordinals;
  ordinals.reserve(impl_->selection.slots.size());
  for (const DeviceSlot& slot : impl_->selection.slots) {
    ordinals.push_back(slot.ordinal);
  }
  std::string dispatch_error;
  impl_->dispatch = build_dispatch_table(impl_->selection.plan, ordinals,
                                         &dispatch_error);
  if (!impl_->dispatch.valid) {
    set_error(error, dispatch_error);
    impl_->reset();
    return Status::kPlanInvalid;
  }
  impl_->tile_indices.resize(impl_->dispatch.by_gpu.size());
  for (std::size_t slot = 0; slot < impl_->dispatch.by_gpu.size(); ++slot) {
    std::vector<int>& indices = impl_->tile_indices[slot];
    indices.reserve(impl_->dispatch.by_gpu[slot].size());
    for (const DispatchEntry& entry : impl_->dispatch.by_gpu[slot]) {
      indices.push_back(entry.tile_index);
    }
  }

  const std::size_t frame_size =
      static_cast<std::size_t>(impl_->frame_bytes);
  const int reference_frames =
      std::max(1, impl_->config.planner.reference_frames);
  const std::uint64_t reference_bytes =
      static_cast<std::uint64_t>(reference_frames) * impl_->frame_bytes;
  const int selected_gpu_count = static_cast<int>(impl_->gpus.size());

  // Stream/event/allocation failures are still startup failures. If the caller
  // allowed degraded startup, drop the offending GPU and rebuild the plan
  // deterministically instead of leaving a half-initialized runtime.
  auto allocation_failure = [&](int ordinal, Status status) -> Status {
    const std::string text =
        (error != nullptr && !error->empty()) ? *error
                                              : std::string(status_name(status));
    impl_->reset();
    if (config.allow_drop_failed &&
        selected_gpu_count - 1 >= config.min_gpus) {
      RuntimeConfig retry = config;
      retry.excluded_ordinals = config.excluded_ordinals;
      retry.excluded_ordinals.push_back(ordinal);
      return initialize(backend, retry, error);
    }
    set_error(error, text);
    return status;
  };

  for (int slot = 0; slot < static_cast<int>(impl_->gpus.size()); ++slot) {
    GpuState& gpu = impl_->gpus[static_cast<std::size_t>(slot)];
    const int tile_count =
        static_cast<int>(impl_->dispatch.by_gpu[static_cast<std::size_t>(slot)].size());
    gpu.info.tile_count = tile_count;
    gpu.tile_sizes.assign(static_cast<std::size_t>(tile_count), 0u);

    std::string gpu_error;
    Status status = backend.stream_create(gpu.info.ordinal, &gpu.compute,
                                          &gpu_error);
    if (!is_ok(status)) {
      set_error(error, "GPU " + ordinal_text(gpu.info.ordinal) +
                           " compute stream: " + gpu_error);
      return allocation_failure(gpu.info.ordinal, status);
    }
    status = backend.stream_create(gpu.info.ordinal, &gpu.copy, &gpu_error);
    if (!is_ok(status)) {
      set_error(error, "GPU " + ordinal_text(gpu.info.ordinal) +
                           " copy stream: " + gpu_error);
      return allocation_failure(gpu.info.ordinal, status);
    }
    for (int parity = 0; parity < 2; ++parity) {
      status = backend.event_create(gpu.info.ordinal, false,
                                    &gpu.upload_done[parity], &gpu_error);
      if (!is_ok(status)) {
        set_error(error, "GPU " + ordinal_text(gpu.info.ordinal) +
                             " upload event: " + gpu_error);
        return allocation_failure(gpu.info.ordinal, status);
      }
    }
    EventHandle* single_events[] = {&gpu.compute_done, &gpu.gather_done,
                                    &gpu.fill_done};
    for (EventHandle* handle : single_events) {
      status = backend.event_create(gpu.info.ordinal, false, handle,
                                    &gpu_error);
      if (!is_ok(status)) {
        set_error(error, "GPU " + ordinal_text(gpu.info.ordinal) +
                             " scheduling event: " + gpu_error);
        return allocation_failure(gpu.info.ordinal, status);
      }
    }

    auto allocate_device = [&](std::size_t bytes, std::uint8_t** out,
                               const char* what) -> Status {
      void* pointer = nullptr;
      const Status result =
          backend.arena_allocate(gpu.arena, bytes, kArenaAlignment, &pointer,
                                 &gpu_error);
      if (!is_ok(result)) {
        set_error(error, "GPU " + ordinal_text(gpu.info.ordinal) + " " + what +
                             ": " + gpu_error);
        return result;
      }
      *out = static_cast<std::uint8_t*>(pointer);
      return Status::kOk;
    };

    if (frame_size >
        std::numeric_limits<std::size_t>::max() / 2) {
      set_error(error, "source double buffer overflows size_t");
      impl_->reset();
      return Status::kInvalidArgument;
    }
    status = allocate_device(frame_size, &gpu.device_source[0], "source A");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    status = allocate_device(frame_size, &gpu.device_source[1], "source B");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    status = allocate_device(frame_size, &gpu.device_recon, "reconstruction");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    if (reference_bytes >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      set_error(error, "reference pool overflows size_t");
      impl_->reset();
      return Status::kInvalidArgument;
    }
    status = allocate_device(static_cast<std::size_t>(reference_bytes),
                             &gpu.device_references, "reference pool");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    const std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(tile_count) *
        static_cast<std::uint64_t>(impl_->payload_stride);
    if (payload_bytes == 0 ||
        payload_bytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      set_error(error, "payload buffer size is invalid");
      impl_->reset();
      return Status::kInvalidArgument;
    }
    status = allocate_device(static_cast<std::size_t>(payload_bytes),
                             &gpu.device_payload, "payload slots");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    gpu.info.device_source_bytes = 2 * impl_->frame_bytes;
    gpu.info.device_recon_bytes = impl_->frame_bytes;
    gpu.info.device_reference_bytes = reference_bytes;
    gpu.info.device_payload_bytes = payload_bytes;

    auto allocate_host = [&](std::size_t bytes, std::uint8_t** out,
                             const char* what) -> Status {
      void* pointer = nullptr;
      const Status result = backend.pinned_alloc(bytes, &pointer, &gpu_error);
      if (!is_ok(result)) {
        set_error(error, "GPU " + ordinal_text(gpu.info.ordinal) + " " + what +
                             " pinned staging: " + gpu_error);
        return result;
      }
      *out = static_cast<std::uint8_t*>(pointer);
      return Status::kOk;
    };
    status = allocate_host(frame_size, &gpu.host_source[0], "source A");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    status = allocate_host(frame_size, &gpu.host_source[1], "source B");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    status = allocate_host(static_cast<std::size_t>(payload_bytes),
                           &gpu.host_payload, "payload");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
    status = allocate_host(frame_size, &gpu.host_recon, "reconstruction");
    if (!is_ok(status)) {
      return allocation_failure(gpu.info.ordinal, status);
    }
  }

  for (std::uint8_t*& ptr : impl_->host_recon) {
    void* pointer = nullptr;
    std::string pinned_error;
    const Status status = backend.pinned_alloc(frame_size, &pointer,
                                               &pinned_error);
    if (!is_ok(status)) {
      set_error(error, "full-frame recon staging: " + pinned_error);
      impl_->reset();
      return status;
    }
    ptr = static_cast<std::uint8_t*>(pointer);
  }

  std::vector<int> physical_ordinals;
  physical_ordinals.reserve(impl_->selection.slots.size());
  for (const DeviceSlot& slot : impl_->selection.slots) {
    physical_ordinals.push_back(slot.ordinal);
  }
  impl_->reference_schedule = plan_reference_schedule(
      impl_->selection.plan, physical_ordinals,
      std::span<const std::uint64_t>());
  impl_->refresh_info();
  impl_->initialized = true;
  return Status::kOk;
}

void Runtime::shutdown() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  impl_->reset();
}

Status Runtime::prefetch_source(int frame_index, const void* host_source,
                                std::size_t bytes, std::string* error) {
  if (!impl_->initialized) {
    set_error(error, "runtime is not initialized");
    return Status::kNotInitialized;
  }
  if (host_source == nullptr || bytes != impl_->frame_bytes) {
    set_error(error, "prefetch source size does not match the frame size");
    return Status::kInvalidArgument;
  }
  if (frame_index < 0) {
    set_error(error, "frame index must be non-negative");
    return Status::kInvalidArgument;
  }
  if (impl_->prefetched_frame == frame_index) {
    set_error(error, "this frame is already prefetched");
    return Status::kAlreadyInitialized;
  }
  if (impl_->prefetched_frame >= 0) {
    set_error(error, "another frame is already prefetched");
    return Status::kBusy;
  }

  const int parity = frame_index & 1;
  auto fail_prefetch = [&](Status status, const std::string& text) -> Status {
    // Drain the copy streams so a partially submitted prefetch cannot race the
    // next upload into the same parity buffer.
    for (GpuState& gpu : impl_->gpus) {
      if (gpu.copy != kInvalidStream) {
        (void)impl_->backend->stream_synchronize(gpu.copy);
      }
    }
    impl_->prefetched_frame = -1;
    set_error(error, text);
    return status;
  };
  for (GpuState& gpu : impl_->gpus) {
    std::memcpy(gpu.host_source[parity], host_source, bytes);
    Status status = impl_->backend->copy_h2d_async(
        gpu.copy, gpu.device_source[parity], gpu.host_source[parity], bytes);
    if (!is_ok(status)) {
      std::ostringstream text;
      text << "GPU " << gpu.info.ordinal << " prefetch upload failed: "
           << status_name(status);
      return fail_prefetch(status, text.str());
    }
    status = impl_->backend->event_record(gpu.upload_done[parity], gpu.copy);
    if (!is_ok(status)) {
      return fail_prefetch(status, "prefetch event record failed");
    }
  }
  impl_->prefetched_frame = frame_index;
  impl_->stats.prefetches += 1;
  const std::uint64_t upload_bytes =
      static_cast<std::uint64_t>(impl_->gpus.size()) * impl_->frame_bytes;
  impl_->stats.source_upload_bytes += upload_bytes;
  impl_->stats.host_tx_bytes += upload_bytes;
  if (error != nullptr) {
    error->clear();
  }
  return Status::kOk;
}

Status Runtime::dispatch_frame(const FrameRequest& request, FrameWorker& worker,
                               FrameResult& result, std::string* error) {
  result = FrameResult{};
  result.frame_index = request.frame_index;
  if (!impl_->initialized) {
    result.error = "runtime is not initialized";
    set_error(error, result.error);
    return Status::kNotInitialized;
  }
  if (request.frame_index < 0) {
    result.error = "frame index must be non-negative";
    set_error(error, result.error);
    return Status::kInvalidArgument;
  }
  if (request.source == nullptr ||
      request.source_bytes != impl_->frame_bytes) {
    result.error = "source frame size does not match the planned frame size";
    set_error(error, result.error);
    return Status::kInvalidArgument;
  }
  if (impl_->prefetched_frame >= 0 &&
      impl_->prefetched_frame != request.frame_index) {
    std::ostringstream text;
    text << "outstanding prefetch for frame " << impl_->prefetched_frame
         << " must be dispatched first";
    result.error = text.str();
    set_error(error, result.error);
    return Status::kBusy;
  }

  bool drained = false;
  auto fail = [&](int slot, Status status, const std::string& text) -> Status {
    // A failed frame leaves work in flight on some streams. Drain the compute
    // and copy streams before returning so the next frame cannot race with the
    // abandoned payload/recon downloads. This is an error-path sync only; the
    // normal frame path never calls it.
    if (!drained) {
      drained = true;
      for (GpuState& gpu : impl_->gpus) {
        if (gpu.compute != kInvalidStream) {
          (void)impl_->backend->stream_synchronize(gpu.compute);
        }
        if (gpu.copy != kInvalidStream) {
          (void)impl_->backend->stream_synchronize(gpu.copy);
        }
      }
    }
    result.valid = false;
    result.failed_slot = slot;
    result.error = text;
    result.tile_payloads.clear();
    result.tile_sizes.clear();
    result.recon.clear();
    impl_->refresh_info();
    set_error(error, text);
    return status;
  };

  const bool prefetched = impl_->prefetched_frame == request.frame_index;
  const int parity = request.frame_index & 1;
  const int recon_parity = request.frame_index & 1;

  // Gate this frame's compute behind the previous frame's reference fill.
  for (GpuState& gpu : impl_->gpus) {
    if (gpu.fill_pending) {
      const Status status =
          impl_->backend->event_wait(gpu.fill_done, gpu.compute);
      if (!is_ok(status)) {
        return fail(gpu.info.slot, status,
                    "failed to gate compute on the previous reference fill");
      }
    }
  }

  for (GpuState& gpu : impl_->gpus) {
    if (prefetched) {
      const Status status =
          impl_->backend->event_wait(gpu.upload_done[parity], gpu.compute);
      if (!is_ok(status)) {
        return fail(gpu.info.slot, status, "failed to wait for prefetched source");
      }
      continue;
    }
    std::memcpy(gpu.host_source[parity], request.source,
                static_cast<std::size_t>(impl_->frame_bytes));
    Status status = impl_->backend->copy_h2d_async(
        gpu.copy, gpu.device_source[parity], gpu.host_source[parity],
        static_cast<std::size_t>(impl_->frame_bytes));
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "source upload failed");
    }
    status = impl_->backend->event_record(gpu.upload_done[parity], gpu.copy);
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "source upload event failed");
    }
    status = impl_->backend->event_wait(gpu.upload_done[parity], gpu.compute);
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "compute wait on source upload failed");
    }
  }
  if (prefetched) {
    impl_->prefetched_frame = -1;
  }

  std::uint64_t payload_bytes = 0;
  std::uint64_t recon_gather_bytes = 0;
  std::uint64_t reference_fill_bytes = 0;
  const tiles::TileLayout& layout = impl_->selection.layout;

  for (GpuState& gpu : impl_->gpus) {
    std::fill(gpu.tile_sizes.begin(), gpu.tile_sizes.end(), 0u);
    const std::vector<DispatchEntry>& local =
        impl_->dispatch.by_gpu[static_cast<std::size_t>(gpu.info.slot)];

    GpuFrameContext context;
    context.backend = impl_->backend;
    context.slot = gpu.info.slot;
    context.ordinal = gpu.info.ordinal;
    context.frame_index = request.frame_index;
    context.reference_index = impl_->reference_index;
    context.frame_width = layout.width;
    context.frame_height = layout.height;
    context.bit_depth = impl_->bit_depth;
    context.layout = &layout;
    context.compute_stream = gpu.compute;
    context.copy_stream = gpu.copy;
    const std::vector<int>& indices =
        impl_->tile_indices[static_cast<std::size_t>(gpu.info.slot)];
    context.tile_indices = indices.empty() ? nullptr : indices.data();
    context.tile_count = static_cast<int>(indices.size());
    context.payload_stride_bytes = impl_->payload_stride;
    context.device_source = gpu.device_source[parity];
    context.device_payload = gpu.device_payload;
    context.device_recon = gpu.device_recon;
    context.device_reference =
        gpu.device_references +
        static_cast<std::uint64_t>(impl_->reference_index) * impl_->frame_bytes;
    context.tile_sizes_out = gpu.tile_sizes.data();
    context.source_bytes = static_cast<std::size_t>(impl_->frame_bytes);
    context.recon_bytes = static_cast<std::size_t>(impl_->frame_bytes);

    std::string worker_error;
    const Status encode_status = worker.encode(context, &worker_error);
    if (!is_ok(encode_status)) {
      return fail(gpu.info.slot, Status::kDispatchFailed,
                  worker_error.empty() ? "GPU worker failed" : worker_error);
    }
    Status status = impl_->backend->event_record(gpu.compute_done, gpu.compute);
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "compute completion event failed");
    }
    status = impl_->backend->event_wait(gpu.compute_done, gpu.copy);
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "copy wait on compute failed");
    }

    const std::uint32_t stride = impl_->payload_stride;
    for (int index = 0; index < context.tile_count; ++index) {
      const std::uint32_t size =
          gpu.tile_sizes[static_cast<std::size_t>(index)];
      if (size < 2 || size > stride) {
        std::ostringstream text;
        text << "tile "
             << local[static_cast<std::size_t>(index)].tile_index
             << " reported " << size << " payload bytes; expected 2.."
             << stride;
        return fail(gpu.info.slot, Status::kDispatchFailed, text.str());
      }
      status = impl_->backend->copy_d2h_async(
          gpu.copy, gpu.host_payload + static_cast<std::size_t>(index) * stride,
          gpu.device_payload + static_cast<std::size_t>(index) * stride, size);
      if (!is_ok(status)) {
        return fail(gpu.info.slot, status, "payload download failed");
      }
      payload_bytes += size;
    }

    for (const DispatchEntry& entry : local) {
      const tiles::TileRect& rect =
          layout.tiles[static_cast<std::size_t>(entry.tile_index)];
      status = enqueue_tile_region_copy(
          *impl_->backend, CopyDirection::kDeviceToHost, gpu.copy, gpu.host_recon,
          gpu.device_recon, 0, layout, rect, impl_->bit_depth);
      if (!is_ok(status)) {
        return fail(gpu.info.slot, status, "reconstruction gather failed");
      }
      recon_gather_bytes +=
          tile_region_bytes(layout, rect, impl_->bit_depth);
    }

    status = impl_->backend->event_record(gpu.gather_done, gpu.copy);
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "gather completion event failed");
    }
  }

  for (GpuState& gpu : impl_->gpus) {
    const Status status = impl_->backend->event_synchronize(gpu.gather_done);
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "waiting for payload gather failed");
    }
  }

  std::vector<TilePayloadView> views;
  views.reserve(static_cast<std::size_t>(impl_->dispatch.tile_count));
  for (int tile = 0; tile < impl_->dispatch.tile_count; ++tile) {
    const DispatchEntry& entry =
        impl_->dispatch.by_tile[static_cast<std::size_t>(tile)];
    const GpuState& gpu = impl_->gpus[static_cast<std::size_t>(entry.gpu_slot)];
    const std::uint32_t size =
        gpu.tile_sizes[static_cast<std::size_t>(entry.local_index)];
    const std::uint8_t* data =
        gpu.host_payload +
        static_cast<std::size_t>(entry.local_index) * impl_->payload_stride;
    views.push_back(TilePayloadView{
        tile, std::span<const std::uint8_t>(data, static_cast<std::size_t>(size))});
  }
  MergedFrame merged;
  std::string merge_error;
  if (!merge_payloads(impl_->selection.plan, views, merged, &merge_error)) {
    return fail(-1, Status::kDispatchFailed, merge_error);
  }

  std::uint8_t* assembled = impl_->host_recon[recon_parity];
  for (int tile = 0; tile < impl_->dispatch.tile_count; ++tile) {
    const DispatchEntry& entry =
        impl_->dispatch.by_tile[static_cast<std::size_t>(tile)];
    const GpuState& gpu = impl_->gpus[static_cast<std::size_t>(entry.gpu_slot)];
    const tiles::TileRect& rect = layout.tiles[static_cast<std::size_t>(tile)];
    const Status status =
        copy_tile_region_host(assembled, gpu.host_recon, layout, rect,
                              impl_->bit_depth);
    if (!is_ok(status)) {
      return fail(entry.gpu_slot, status, "host reconstruction assembly failed");
    }
  }

  const int reference_frames =
      std::max(1, impl_->config.planner.reference_frames);
  const int next_reference =
      (impl_->reference_index + 1) % reference_frames;

  for (GpuState& gpu : impl_->gpus) {
    std::uint8_t* slot_base =
        gpu.device_references +
        static_cast<std::uint64_t>(next_reference) * impl_->frame_bytes;
    for (int tile = 0; tile < impl_->dispatch.tile_count; ++tile) {
      const DispatchEntry& entry =
          impl_->dispatch.by_tile[static_cast<std::size_t>(tile)];
      const tiles::TileRect& rect =
          layout.tiles[static_cast<std::size_t>(tile)];
      Status tile_status = Status::kOk;
      if (entry.gpu_slot == gpu.info.slot) {
        tile_status = enqueue_tile_region_copy(
            *impl_->backend, CopyDirection::kDeviceToDevice, gpu.copy, slot_base,
            gpu.device_recon, 0, layout, rect, impl_->bit_depth);
      } else {
        tile_status = enqueue_tile_region_copy(
            *impl_->backend, CopyDirection::kHostToDevice, gpu.copy, slot_base,
            assembled, 0, layout, rect, impl_->bit_depth);
        reference_fill_bytes +=
            tile_region_bytes(layout, rect, impl_->bit_depth);
      }
      if (!is_ok(tile_status)) {
        return fail(gpu.info.slot, tile_status, "reference fill failed");
      }
    }
    const Status status =
        impl_->backend->event_record(gpu.fill_done, gpu.copy);
    if (!is_ok(status)) {
      return fail(gpu.info.slot, status, "reference fill event failed");
    }
    gpu.fill_pending = true;
  }
  impl_->reference_index = next_reference;

  result.valid = true;
  result.frame_index = request.frame_index;
  result.tile_count = merged.tile_count;
  result.tile_payloads = std::move(merged.tiles);
  result.tile_sizes = std::move(merged.tile_sizes);
  result.payload_bytes = merged.total_bytes;
  result.reference_fill_bytes = reference_fill_bytes;
  result.reference_fill_pending = true;
  const std::uint64_t source_bytes =
      static_cast<std::uint64_t>(impl_->gpus.size()) * impl_->frame_bytes;
  result.host_tx_bytes = source_bytes + payload_bytes + recon_gather_bytes +
                         reference_fill_bytes;
  if (request.capture_recon) {
    result.recon.assign(assembled, assembled + static_cast<std::size_t>(impl_->frame_bytes));
  }

  impl_->stats.frames += 1;
  impl_->stats.merged_payload_bytes += merged.total_bytes;
  impl_->stats.reference_fill_bytes += reference_fill_bytes;
  if (!prefetched) {
    impl_->stats.source_upload_bytes += source_bytes;
    impl_->stats.host_tx_bytes += source_bytes;
  }
  impl_->stats.host_tx_bytes += payload_bytes + recon_gather_bytes +
                                reference_fill_bytes;
  impl_->refresh_info();
  if (error != nullptr) {
    error->clear();
  }
  return Status::kOk;
}

Status Runtime::synchronize(std::string* error) {
  if (!impl_->initialized) {
    set_error(error, "runtime is not initialized");
    return Status::kNotInitialized;
  }
  for (GpuState& gpu : impl_->gpus) {
    if (!gpu.fill_pending) {
      continue;
    }
    const Status status = impl_->backend->event_synchronize(gpu.fill_done);
    if (!is_ok(status)) {
      set_error(error, "waiting for reference fill failed");
      return status;
    }
    gpu.fill_pending = false;
  }
  impl_->refresh_info();
  if (error != nullptr) {
    error->clear();
  }
  return Status::kOk;
}

}  // namespace av2enc::multigpu::runtime
