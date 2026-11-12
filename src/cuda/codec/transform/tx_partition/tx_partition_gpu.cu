// CUDA kernels and launchers for the TX_MODE_SELECT transform partition.
// See tx_partition_gpu.h for the arena contract; the transforms themselves are
// the `src/cuda/tx_variants/` `_multi` kernels (one block per TX part).
#include "cuda/codec/transform/tx_partition/tx_partition_gpu.h"

#include <cstddef>
#include <cstdint>

#include "cuda/codec/transform/variants/tx_variants.h"

namespace av2enc::cuda::txpart {
namespace {

// Per-plan geometry resolution.  One CUDA block per plan; only lane 0 writes,
// because plan_resolve() fills up to four jobs in decode order.
__global__ void resolve_kernel(const PlanDesc* __restrict__ plans, int nplans,
                               const TxParams* __restrict__ params,
                               TxBlockJob* __restrict__ jobs, int job_capacity,
                               ResolvedPart* __restrict__ parts,
                               int part_capacity) {
  if (threadIdx.x != 0 || blockIdx.x >= static_cast<unsigned>(nplans)) return;
  (void)plan_resolve(params, plans[blockIdx.x], jobs, job_capacity, parts,
                     part_capacity);
}

__device__ __forceinline__ int dequant_round2(int value, int n) {
  return (value + (1 << (n - 1))) >> n;
}

// Same 528 / 3-bit dequantization as src/cuda/tx_variants/tx_variants.cu and
// src/encoder/tx_variants/tx_variants.cpp.  One element per thread, so
// `d_dequant == d_levels` is a safe in-place form.
__global__ void dequant_kernel(const TxBlockJob* __restrict__ jobs,
                               int njobs, const std::int32_t* __restrict__ levels,
                               std::int32_t* __restrict__ dequant) {
  const unsigned job_index = blockIdx.x;
  if (job_index >= static_cast<unsigned>(njobs)) return;
  const TxBlockJob job = jobs[job_index];
  const int elems = job.params.w * job.params.h;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= elems) return;
  const int index = job.in_offset + tid;
  const int qc = levels[index];
  const int abs_qc = qc < 0 ? -qc : qc;
  const int dq_high = (abs_qc * 528) & 0xFFFFFF;
  const int dq = dequant_round2(dq_high, 3) / job.params.dq_denom;
  const int clipped = dq < -(1 << 15) ? -(1 << 15)
                      : (dq > (1 << 15) - 1 ? (1 << 15) - 1 : dq);
  dequant[index] = qc < 0 ? -clipped : clipped;
}

__global__ void scan_table_kernel(ScanTableView view) {
  const int index = static_cast<int>(threadIdx.x);
  if (index >= kScanTableCount) return;
  const int tx_size = index / kScanClasses;
  const int cls = index - tx_size * kScanClasses;
  (void)scan_table_fill_one(view, tx_size, cls);
}

__global__ void grid_kernel(const PlanDesc* __restrict__ plans, int nplans,
                            const ResolvedPart* __restrict__ parts,
                            TxGridView grid) {
  if (blockIdx.x >= static_cast<unsigned>(nplans)) return;
  if (threadIdx.x != 0) return;
  const PlanDesc plan = plans[blockIdx.x];
  PlanGeometry geom;
  if (!plan_geometry(plan.mi_size, plan.partition, plan.mi_row, plan.mi_col,
                     &geom)) {
    return;
  }
  for (int i = 0; i < geom.count; ++i) {
    const ResolvedPart part = parts[plan.part_base + i];
    if (part.tx_size < 0 || part.tx_size >= kTxSizeCount) continue;
    (void)set_tx_size(grid, part.mi_row, part.mi_col,
                      tx_height(part.tx_size) / kMiSizePx,
                      tx_width(part.tx_size) / kMiSizePx, 0, 0);
  }
}

Status resolve_plans(const PlanBatch& batch, const Workspace& workspace,
                     const LaunchScratch& scratch, cudaStream_t stream,
                     std::string* error) {
  const int plan_count = static_cast<int>(batch.plans.size());
  if (plan_count > scratch.plan_capacity) {
    if (error != nullptr) *error = "tx_partition: plan capacity exceeded";
    return Status::kInvalidArgument;
  }
  const cudaError_t copy_error = cudaMemcpyAsync(
      scratch.d_plans, batch.plans.data(),
      sizeof(PlanDesc) * static_cast<std::size_t>(plan_count),
      cudaMemcpyHostToDevice, stream);
  if (copy_error != cudaSuccess) return status_from_cuda(copy_error);
  resolve_kernel<<<static_cast<unsigned>(plan_count), 4, 0, stream>>>(
      scratch.d_plans, plan_count, workspace.d_params, scratch.d_jobs,
      scratch.part_capacity, scratch.d_parts, scratch.part_capacity);
  return status_from_cuda(cudaGetLastError());
}

bool batch_fits(const Workspace& workspace, const PlanBatch& batch,
                std::string* error) {
  if (!workspace.valid()) {
    if (error != nullptr) *error = "tx_partition: workspace not initialized";
    return false;
  }
  const int plan_count = static_cast<int>(batch.plans.size());
  if (plan_count <= 0 || plan_count > workspace.max_plans) {
    if (error != nullptr) {
      *error = "tx_partition: plan count outside workspace max_plans";
    }
    return false;
  }
  if (batch.total_parts <= 0 ||
      batch.total_parts > plan_count * kMaxParts) {
    if (error != nullptr) *error = "tx_partition: invalid total_parts";
    return false;
  }
  return true;
}

}  // namespace

Status forward_batch(Arena& arena, Workspace& workspace,
                     const PlanBatch& batch,
                     const std::int16_t* d_residual_adj, double* d_coeff,
                     std::int32_t* d_levels, cudaStream_t stream,
                     std::uint64_t* required_bytes, std::string* error) {
  if (!batch_fits(workspace, batch, error)) return Status::kInvalidArgument;
  if (d_residual_adj == nullptr || d_coeff == nullptr || d_levels == nullptr) {
    if (error != nullptr) *error = "tx_partition: null transform buffer";
    return Status::kInvalidArgument;
  }
  LaunchScratch scratch;
  const Status acquired =
      acquire_scratch(arena, workspace, static_cast<int>(batch.plans.size()),
                      &scratch, required_bytes, error);
  if (acquired != Status::kOk) return acquired;
  Status status = resolve_plans(batch, workspace, scratch, stream, error);
  if (status == Status::kOk) {
    status = status_from_cuda(txv_forward_transform_multi(
        d_residual_adj, d_coeff, d_levels, scratch.d_jobs, batch.total_parts,
        stream));
  }
  release_scratch(arena, scratch);
  return status;
}

Status inverse_batch(Arena& arena, Workspace& workspace,
                     const PlanBatch& batch, const std::int32_t* d_levels,
                     std::int16_t* d_residual_coded, cudaStream_t stream,
                     std::uint64_t* required_bytes, std::string* error) {
  if (!batch_fits(workspace, batch, error)) return Status::kInvalidArgument;
  if (d_levels == nullptr || d_residual_coded == nullptr) {
    if (error != nullptr) *error = "tx_partition: null transform buffer";
    return Status::kInvalidArgument;
  }
  LaunchScratch scratch;
  const Status acquired =
      acquire_scratch(arena, workspace, static_cast<int>(batch.plans.size()),
                      &scratch, required_bytes, error);
  if (acquired != Status::kOk) return acquired;
  Status status = resolve_plans(batch, workspace, scratch, stream, error);
  if (status == Status::kOk) {
    status = status_from_cuda(txv_inverse_transform_multi(
        d_levels, d_residual_coded, scratch.d_jobs, batch.total_parts, stream));
  }
  release_scratch(arena, scratch);
  return status;
}

Status dequantize_batch(Arena& arena, Workspace& workspace,
                        const PlanBatch& batch, const std::int32_t* d_levels,
                        std::int32_t* d_dequant, cudaStream_t stream,
                        std::uint64_t* required_bytes, std::string* error) {
  if (!batch_fits(workspace, batch, error)) return Status::kInvalidArgument;
  if (d_levels == nullptr || d_dequant == nullptr) {
    if (error != nullptr) *error = "tx_partition: null dequant buffer";
    return Status::kInvalidArgument;
  }
  LaunchScratch scratch;
  const Status acquired =
      acquire_scratch(arena, workspace, static_cast<int>(batch.plans.size()),
                      &scratch, required_bytes, error);
  if (acquired != Status::kOk) return acquired;
  Status status = resolve_plans(batch, workspace, scratch, stream, error);
  if (status == Status::kOk) {
    dequant_kernel<<<static_cast<unsigned>(batch.total_parts), 1024, 0,
                     stream>>>(scratch.d_jobs, batch.total_parts, d_levels,
                               d_dequant);
    status = status_from_cuda(cudaGetLastError());
  }
  release_scratch(arena, scratch);
  return status;
}

Status inverse_dequant_batch(Arena& arena, Workspace& workspace,
                             const PlanBatch& batch,
                             const std::int32_t* d_levels,
                             std::int32_t* d_dequant,
                             std::int16_t* d_residual_coded,
                             cudaStream_t stream,
                             std::uint64_t* required_bytes,
                             std::string* error) {
  if (!batch_fits(workspace, batch, error)) return Status::kInvalidArgument;
  if (d_levels == nullptr || d_dequant == nullptr ||
      d_residual_coded == nullptr) {
    if (error != nullptr) *error = "tx_partition: null transform buffer";
    return Status::kInvalidArgument;
  }
  LaunchScratch scratch;
  const Status acquired =
      acquire_scratch(arena, workspace, static_cast<int>(batch.plans.size()),
                      &scratch, required_bytes, error);
  if (acquired != Status::kOk) return acquired;
  Status status = resolve_plans(batch, workspace, scratch, stream, error);
  if (status == Status::kOk) {
    dequant_kernel<<<static_cast<unsigned>(batch.total_parts), 1024, 0,
                     stream>>>(scratch.d_jobs, batch.total_parts, d_levels,
                               d_dequant);
    status = status_from_cuda(cudaGetLastError());
  }
  if (status == Status::kOk) {
    status = status_from_cuda(txv_inverse_block_multi(
        d_dequant, d_residual_coded, scratch.d_jobs, batch.total_parts,
        stream));
  }
  release_scratch(arena, scratch);
  return status;
}

Status build_scan_tables_device(const ScanTableView& view,
                                cudaStream_t stream) {
  if (view.data == nullptr || view.length == nullptr) {
    return Status::kInvalidArgument;
  }
  scan_table_kernel<<<1, 128, 0, stream>>>(view);
  return status_from_cuda(cudaGetLastError());
}

Status build_grid_device(Arena& arena, Workspace& workspace,
                         const PlanBatch& batch, const TxGridView& grid,
                         cudaStream_t stream, std::uint64_t* required_bytes,
                         std::string* error) {
  if (!batch_fits(workspace, batch, error)) return Status::kInvalidArgument;
  if (grid.tx_size == nullptr || grid.middle == nullptr ||
      grid.scan_order == nullptr || grid.mi_rows <= 0 || grid.mi_cols <= 0) {
    if (error != nullptr) *error = "tx_partition: invalid grid view";
    return Status::kInvalidArgument;
  }
  LaunchScratch scratch;
  const Status acquired =
      acquire_scratch(arena, workspace, static_cast<int>(batch.plans.size()),
                      &scratch, required_bytes, error);
  if (acquired != Status::kOk) return acquired;
  Status status = resolve_plans(batch, workspace, scratch, stream, error);
  if (status == Status::kOk) {
    grid_kernel<<<static_cast<unsigned>(batch.plans.size()), 1, 0, stream>>>(
        scratch.d_plans, static_cast<int>(batch.plans.size()), scratch.d_parts,
        grid);
    status = status_from_cuda(cudaGetLastError());
  }
  release_scratch(arena, scratch);
  return status;
}

}  // namespace av2enc::cuda::txpart
