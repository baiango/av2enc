// Host plan/arena implementation for the CUDA TX_MODE_SELECT transform
// partition.  No direct device allocation: every pointer comes from the
// caller's `av2enc::cuda::Arena` (arena.h).
#include "cuda/codec/transform/tx_partition/tx_partition_plan.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace av2enc::cuda::txpart {
namespace {

// Sentinel so release_scratch() can never rewind to a stale mark 0 that would
// discard the persistent Workspace tables.
constexpr std::uint64_t kNoMark = kNoArenaMark;

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept {
  const std::uint64_t mask = alignment - 1;
  return (value + mask) & ~mask;
}

template <typename T>
std::uint64_t array_bytes(std::uint64_t count) noexcept {
  return align_up(static_cast<std::uint64_t>(sizeof(T)) * count,
                  static_cast<std::uint64_t>(Arena::kAlignment));
}

std::uint64_t config_persistent_bytes(const LaunchConfig& config) {
  if (config.max_plans <= 0) return 0;
  std::uint64_t total = 0;
  total += array_bytes<TxParams>(kTxTypeCount * kTxSizeCount);
  total += array_bytes<std::uint16_t>(kScanTableCount * kMaxScanArea);
  total += array_bytes<std::uint16_t>(kScanTableCount);
  return total;
}

std::uint64_t config_scratch_bytes(const LaunchConfig& config) {
  if (config.max_plans <= 0) return 0;
  const std::uint64_t plans = static_cast<std::uint64_t>(config.max_plans);
  std::uint64_t total = 0;
  total += array_bytes<PlanDesc>(plans);
  total += array_bytes<TxBlockJob>(plans * kMaxParts);
  total += array_bytes<ResolvedPart>(plans * kMaxParts);
  return total;
}

std::uint64_t scratch_bytes_for_plans(int plan_count) {
  const std::uint64_t plans = static_cast<std::uint64_t>(plan_count);
  std::uint64_t total = 0;
  total += array_bytes<PlanDesc>(plans);
  total += array_bytes<TxBlockJob>(plans * kMaxParts);
  total += array_bytes<ResolvedPart>(plans * kMaxParts);
  return total;
}

bool set_error(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

}  // namespace

std::uint64_t required_device_bytes(const LaunchConfig& config) {
  return config_persistent_bytes(config) + config_scratch_bytes(config);
}

std::uint64_t persistent_device_bytes(const LaunchConfig& config) {
  return config_persistent_bytes(config);
}

std::uint64_t scratch_device_bytes(const LaunchConfig& config) {
  return config_scratch_bytes(config);
}

void Workspace::reset() noexcept {
  d_params = nullptr;
  d_scan_data = nullptr;
  d_scan_length = nullptr;
  host_params.clear();
  max_plans = 0;
  bytes = 0;
  config = LaunchConfig{};
}

Status Workspace::init(Arena& arena, const LaunchConfig& config_in,
                       std::uint64_t* required_bytes, std::string* error) {
  const std::uint64_t required = required_device_bytes(config_in);
  if (required_bytes != nullptr) *required_bytes = required;
  reset();

  if (config_in.max_plans <= 0 || config_in.max_plans > (1 << 20)) {
    set_error(error, "tx_partition workspace: max_plans must be in [1, 1048576]");
    return Status::kInvalidArgument;
  }
  if (!arena.initialized()) {
    set_error(error, "tx_partition workspace: arena not initialized");
    return Status::kNotInitialized;
  }
  if (arena.remaining() < required) {
    set_error(error, "tx_partition workspace: need " +
                         std::to_string(required) + " bytes, arena has " +
                         std::to_string(arena.remaining()));
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  // Keep this order in sync with config_persistent_bytes().
  d_params = arena.allocate_array<TxParams>(kTxTypeCount * kTxSizeCount);
  d_scan_data =
      arena.allocate_array<std::uint16_t>(kScanTableCount * kMaxScanArea);
  d_scan_length = arena.allocate_array<std::uint16_t>(kScanTableCount);
  if (d_params == nullptr || d_scan_data == nullptr ||
      d_scan_length == nullptr) {
    arena.rewind(mark);
    reset();
    set_error(error, "tx_partition workspace: arena allocation failed after "
                     "the size check");
    return Status::kArenaExhausted;
  }

  // The persistent tables must leave room for the largest configured launch.
  const std::uint64_t scratch = scratch_device_bytes(config_in);
  if (arena.remaining() < scratch) {
    arena.rewind(mark);
    reset();
    set_error(error, "tx_partition workspace: no room for per-launch scratch");
    return Status::kArenaExhausted;
  }

  // Host mirror of the [tx_type][tx_size] parameter table.
  host_params.resize(static_cast<std::size_t>(kTxTypeCount * kTxSizeCount));
  for (int tx_type = 0; tx_type < kTxTypeCount; ++tx_type) {
    for (int tx_size = 0; tx_size < kTxSizeCount; ++tx_size) {
      const auto params = av2enc::txv::make_tx_params(
          static_cast<av2enc::txv::TxSize>(tx_size),
          static_cast<av2enc::txv::TxType>(tx_type));
      if (!params.has_value()) {
        arena.rewind(mark);
        reset();
        set_error(error, "tx_partition workspace: invalid TxParams entry");
        return Status::kInvalidArgument;
      }
      host_params[static_cast<std::size_t>(tx_type * kTxSizeCount + tx_size)] =
          *params;
    }
  }

  // §5.20.7.30 scans, generated once by the same shared code as the device
  // kernel/emulation and uploaded.
  std::vector<std::uint16_t> scan_data(
      static_cast<std::size_t>(kScanTableCount * kMaxScanArea));
  std::vector<std::uint16_t> scan_length(
      static_cast<std::size_t>(kScanTableCount));
  ScanTableView host_view;
  host_view.data = scan_data.data();
  host_view.length = scan_length.data();
  for (int tx_size = 0; tx_size < kTxSizeCount; ++tx_size) {
    for (int cls = 0; cls < kScanClasses; ++cls) {
      if (!scan_table_fill_one(host_view, tx_size, cls)) {
        arena.rewind(mark);
        reset();
        set_error(error, "tx_partition workspace: scan generation failed");
        return Status::kInvalidArgument;
      }
    }
  }

  cudaError_t cuda_error =
      cudaMemcpy(d_params, host_params.data(),
                 sizeof(TxParams) *
                     static_cast<std::size_t>(kTxTypeCount * kTxSizeCount),
                 cudaMemcpyHostToDevice);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(
        d_scan_data, scan_data.data(),
        sizeof(std::uint16_t) *
            static_cast<std::size_t>(kScanTableCount * kMaxScanArea),
        cudaMemcpyHostToDevice);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(
        d_scan_length, scan_length.data(),
        sizeof(std::uint16_t) * static_cast<std::size_t>(kScanTableCount),
        cudaMemcpyHostToDevice);
  }
  if (cuda_error != cudaSuccess) {
    arena.rewind(mark);
    reset();
    set_error(error, "tx_partition workspace: table upload failed");
    return status_from_cuda(cuda_error);
  }

  max_plans = config_in.max_plans;
  bytes = required;
  config = config_in;
  return Status::kOk;
}

Status acquire_scratch(Arena& arena, const Workspace& workspace, int plan_count,
                       LaunchScratch* scratch,
                       std::uint64_t* required_bytes, std::string* error) {
  const std::uint64_t required = required_device_bytes(workspace.config);
  if (required_bytes != nullptr) *required_bytes = required;
  if (scratch == nullptr) {
    set_error(error, "tx_partition scratch: null output");
    return Status::kInvalidArgument;
  }
  scratch->d_plans = nullptr;
  scratch->d_jobs = nullptr;
  scratch->d_parts = nullptr;
  scratch->plan_capacity = 0;
  scratch->part_capacity = 0;
  scratch->mark = kNoMark;

  if (!workspace.valid()) {
    set_error(error, "tx_partition scratch: workspace not initialized");
    return Status::kNotInitialized;
  }
  if (!arena.initialized()) {
    set_error(error, "tx_partition scratch: arena not initialized");
    return Status::kNotInitialized;
  }
  if (plan_count <= 0 || plan_count > workspace.max_plans) {
    set_error(error, "tx_partition scratch: plan_count outside workspace "
                     "max_plans");
    return Status::kInvalidArgument;
  }
  const std::uint64_t needed = scratch_bytes_for_plans(plan_count);
  if (arena.remaining() < needed) {
    set_error(error, "tx_partition scratch: need " +
                         std::to_string(needed) + " bytes, arena has " +
                         std::to_string(arena.remaining()));
    return Status::kArenaExhausted;
  }

  const std::uint64_t mark = arena.mark();
  const std::size_t parts =
      static_cast<std::size_t>(plan_count) * static_cast<std::size_t>(kMaxParts);
  // Keep this order in sync with scratch_bytes_for_plans().
  scratch->d_plans =
      arena.allocate_array<PlanDesc>(static_cast<std::size_t>(plan_count));
  scratch->d_jobs = arena.allocate_array<TxBlockJob>(parts);
  scratch->d_parts = arena.allocate_array<ResolvedPart>(parts);
  if (scratch->d_plans == nullptr || scratch->d_jobs == nullptr ||
      scratch->d_parts == nullptr) {
    arena.rewind(mark);
    scratch->d_plans = nullptr;
    scratch->d_jobs = nullptr;
    scratch->d_parts = nullptr;
    set_error(error, "tx_partition scratch: arena allocation failed after the "
                     "size check");
    return Status::kArenaExhausted;
  }
  scratch->plan_capacity = plan_count;
  scratch->part_capacity = static_cast<int>(parts);
  scratch->mark = mark;
  return Status::kOk;
}

void release_scratch(Arena& arena, LaunchScratch& scratch) noexcept {
  if (scratch.mark == kNoMark) return;
  (void)arena.rewind(scratch.mark);
  scratch.d_plans = nullptr;
  scratch.d_jobs = nullptr;
  scratch.d_parts = nullptr;
  scratch.plan_capacity = 0;
  scratch.part_capacity = 0;
  scratch.mark = kNoMark;
}

Status make_batch(const PlanRequest* requests, int count, PlanBatch* batch,
                  std::string* error) {
  if (batch == nullptr) {
    set_error(error, "tx_partition batch: null output");
    return Status::kInvalidArgument;
  }
  *batch = PlanBatch{};
  if (requests == nullptr || count <= 0) {
    set_error(error, "tx_partition batch: empty request list");
    return Status::kInvalidArgument;
  }

  std::int64_t wide_adj = 0;
  std::int64_t wide_coded = 0;
  int total_parts = 0;
  batch->plans.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const PlanRequest& request = requests[i];
    if (request.tx_type < 0 || request.tx_type >= kTxTypeCount) {
      set_error(error, "tx_partition batch: tx_type outside [0,16)");
      return Status::kInvalidArgument;
    }
    PlanGeometry geom;
    if (!plan_geometry(request.mi_size, request.partition, request.mi_row,
                       request.mi_col, &geom)) {
      set_error(error, "tx_partition batch: request " + std::to_string(i) +
                           " is not signallable (mi_size " +
                           std::to_string(request.mi_size) + ", partition " +
                           std::to_string(request.partition) + ")");
      return Status::kInvalidArgument;
    }
    PlanDesc desc;
    desc.mi_size = request.mi_size;
    desc.partition = request.partition;
    desc.mi_row = request.mi_row;
    desc.mi_col = request.mi_col;
    desc.tx_type = request.tx_type;
    desc.part_base = total_parts;
    desc.in_base = static_cast<int>(wide_adj);
    desc.out_base = static_cast<int>(wide_coded);
    for (int p = 0; p < geom.count; ++p) {
      const int tx_size = geom.parts[p].tx_size;
      wide_adj += static_cast<std::int64_t>(scan_width(tx_size)) *
                  scan_height(tx_size);
      wide_coded += static_cast<std::int64_t>(tx_width(tx_size)) *
                    tx_height(tx_size);
    }
    if (wide_adj > 0x7FFFFFFF || wide_coded > 0x7FFFFFFF) {
      set_error(error, "tx_partition batch: element offsets overflow int32");
      return Status::kInvalidArgument;
    }
    total_parts += geom.count;
    batch->plans.push_back(desc);
  }
  batch->total_parts = total_parts;
  batch->total_adj_elems = static_cast<std::int32_t>(wide_adj);
  batch->total_coded_elems = static_cast<std::int32_t>(wide_coded);
  return Status::kOk;
}

Status make_batch(const PlanRequest& request, PlanBatch* batch,
                  std::string* error) {
  return make_batch(&request, 1, batch, error);
}

Status build_host_jobs(const Workspace& workspace, const PlanBatch& batch,
                       TxBlockJob* jobs, int capacity, std::string* error) {
  if (jobs == nullptr) {
    set_error(error, "tx_partition jobs: null output");
    return Status::kInvalidArgument;
  }
  if (!workspace.valid()) {
    set_error(error, "tx_partition jobs: workspace not initialized");
    return Status::kNotInitialized;
  }
  if (capacity < batch.total_parts) {
    set_error(error, "tx_partition jobs: capacity below total_parts");
    return Status::kInvalidArgument;
  }
  for (const PlanDesc& plan : batch.plans) {
    if (!plan_resolve(workspace.host_params.data(), plan, jobs, capacity,
                      nullptr, 0)) {
      set_error(error, "tx_partition jobs: plan resolution failed");
      return Status::kInvalidArgument;
    }
  }
  return Status::kOk;
}

Status build_host_parts(const PlanBatch& batch, ResolvedPart* parts,
                        int capacity, std::string* error) {
  if (parts == nullptr) {
    set_error(error, "tx_partition parts: null output");
    return Status::kInvalidArgument;
  }
  if (capacity < batch.total_parts) {
    set_error(error, "tx_partition parts: capacity below total_parts");
    return Status::kInvalidArgument;
  }
  for (const PlanDesc& plan : batch.plans) {
    PlanGeometry geom;
    if (!plan_geometry(plan.mi_size, plan.partition, plan.mi_row, plan.mi_col,
                       &geom)) {
      set_error(error, "tx_partition parts: plan geometry failed");
      return Status::kInvalidArgument;
    }
    if (plan.part_base < 0 ||
        plan.part_base + geom.count > capacity) {
      set_error(error, "tx_partition parts: part_base outside capacity");
      return Status::kInvalidArgument;
    }
    for (int i = 0; i < geom.count; ++i) {
      ResolvedPart& out = parts[plan.part_base + i];
      out.mi_row = geom.parts[i].mi_row;
      out.mi_col = geom.parts[i].mi_col;
      out.tx_size = geom.parts[i].tx_size;
    }
  }
  return Status::kOk;
}

av2enc::video::txpartition::LumaTxGrid build_luma_grid(const PlanBatch& batch,
                                            int mi_rows, int mi_cols) {
  av2enc::video::txpartition::LumaTxGrid grid;
  grid.reset(mi_rows, mi_cols);
  TxGridView view;
  view.tx_size = grid.tx_size.data();
  view.middle = grid.middle.data();
  view.scan_order = grid.scan_order.data();
  view.mi_rows = grid.mi_rows;
  view.mi_cols = grid.mi_cols;
  for (const PlanDesc& plan : batch.plans) {
    PlanGeometry geom;
    if (!plan_geometry(plan.mi_size, plan.partition, plan.mi_row, plan.mi_col,
                       &geom)) {
      continue;
    }
    for (int i = 0; i < geom.count; ++i) {
      const int tx_size = geom.parts[i].tx_size;
      (void)set_tx_size(view, geom.parts[i].mi_row, geom.parts[i].mi_col,
                        tx_height(tx_size) / kMiSizePx,
                        tx_width(tx_size) / kMiSizePx, 0, 0);
    }
  }
  return grid;
}

}  // namespace av2enc::cuda::txpart
