// CUDA TX_MODE_SELECT transform-partition launchers.
//
// One CUDA block per TX part executes the `src/cuda/tx_variants/` kernels; the
// plan geometry, element offsets and LumaTxSizes grid are produced on the
// device from the shared `tx_partition_device.cuh` core.
//
// Arena rules (mandatory):
//   * no direct device allocation anywhere in this component;
//   * `Workspace` persistent tables are allocated once by `Workspace::init()`;
//   * every launcher takes the caller's `Arena&`, bracket-allocs its plan/job
//     scratch with `acquire_scratch()` (arena mark) and rewinds it with
//     `release_scratch()`;
//   * the arena scratch is reused by later launches only through stream-ordered
//     copies and kernels, so a launcher never synchronises.
//
// Entry points return `Status`.  A short arena yields
// `Status::kArenaExhausted` with `*required_bytes == required_device_bytes()`.
#ifndef AV2ENC_CUDA_TX_PARTITION_GPU_H_
#define AV2ENC_CUDA_TX_PARTITION_GPU_H_

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cuda/codec/transform/tx_partition/tx_partition_plan.h"

namespace av2enc::cuda::txpart {

// residual_adjusted (int16) -> coeff (double) + levels (int32), one job per TX
// part.  `d_coeff` and `d_levels` are laid out at each job's `in_offset`.
Status forward_batch(Arena& arena, Workspace& workspace,
                     const PlanBatch& batch,
                     const std::int16_t* d_residual_adj, double* d_coeff,
                     std::int32_t* d_levels, cudaStream_t stream = nullptr,
                     std::uint64_t* required_bytes = nullptr,
                     std::string* error = nullptr);

// levels (int32) -> coded residual (int16), one job per TX part.  The fused
// tx_variants inverse dequantizes inside the kernel.
Status inverse_batch(Arena& arena, Workspace& workspace,
                     const PlanBatch& batch, const std::int32_t* d_levels,
                     std::int16_t* d_residual_coded,
                     cudaStream_t stream = nullptr,
                     std::uint64_t* required_bytes = nullptr,
                     std::string* error = nullptr);

// levels (int32) -> dequant (int32).  `d_dequant == d_levels` is the supported
// in-place form (each thread reads then writes its own element).
Status dequantize_batch(Arena& arena, Workspace& workspace,
                        const PlanBatch& batch, const std::int32_t* d_levels,
                        std::int32_t* d_dequant, cudaStream_t stream = nullptr,
                        std::uint64_t* required_bytes = nullptr,
                        std::string* error = nullptr);

// levels -> dequant (optionally in place) -> coded residual (int16).  Both the
// dequant and the inverse run over the same device-resolved job array.
Status inverse_dequant_batch(Arena& arena, Workspace& workspace,
                             const PlanBatch& batch,
                             const std::int32_t* d_levels,
                             std::int32_t* d_dequant,
                             std::int16_t* d_residual_coded,
                             cudaStream_t stream = nullptr,
                             std::uint64_t* required_bytes = nullptr,
                             std::string* error = nullptr);

// Boots the §5.20.7.30 scan table on the device into a caller-provided view
// (typically arena scratch).  This is the device twin of the host generation
// used by Workspace::init(); the GPU test compares both against the CPU scan.
Status build_scan_tables_device(const ScanTableView& view,
                                cudaStream_t stream = nullptr);

// Device LumaTxSizes grid builder.  `grid` points at caller/arena buffers of
// grid.mi_rows * grid.mi_cols bytes each; nothing is allocated here.
Status build_grid_device(Arena& arena, Workspace& workspace,
                         const PlanBatch& batch, const TxGridView& grid,
                         cudaStream_t stream = nullptr,
                         std::uint64_t* required_bytes = nullptr,
                         std::string* error = nullptr);

}  // namespace av2enc::cuda::txpart

#endif  // AV2ENC_CUDA_TX_PARTITION_GPU_H_
