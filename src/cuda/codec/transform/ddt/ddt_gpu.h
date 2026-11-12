// CUDA inter DDT (Data Dependent Transform) — arena-backed batch API.
//
// CPU reference: src/encoder/m2/ddt/.  This component implements the
// decoder-normative inverse (§7.15.2.1 + §7.15.4.1 + §9.6 Ddtx kernels) and
// the encoder forward from that reference, bit-exactly, on the GPU.
//
// Memory rule (docs/cuda-interface.md): every device byte comes from one
// `av2enc::cuda::Arena`.  This module never calls cudaMalloc/cudaFree.  The
// host-facing batch entry points
//
//   1. compute `required_device_bytes(config)`,
//   2. bump-allocate each device array with `Arena::allocate_checked()`,
//   3. run between `Arena::mark()` / `Arena::rewind()`, and
//   4. return `BatchStatus::kArenaTooSmall` with the exact required size when
//      the arena cannot provide it.
//
// Device-pointer launchers (`forward_batch_device` / `inverse_batch_device`)
// let an integration path keep its own arena-resident buffers; they allocate
// nothing and launch exactly one kernel.
//
// Batch layout: one `DdtJob` per transform block.  `in_offset` / `out_offset`
// are element offsets into the shared input/output buffers, so a mixed-size
// candidate batch can be packed with host-side prefix sums (the same
// convention as `cuda/tx_variants/`).  For the inverse, `out_offset` indexes
// the coded-size residual (`coded_w * coded_h` per block).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/arena.h"
#include "cuda/error.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::cuda::ddt {

using av2enc::txv::TxParams;

// One block's transform parameters plus the DDT decoder state and buffer
// offsets.  `bit_depth` affects only the inverse clips (8 or 10).
struct DdtJob {
  TxParams params{};
  std::int32_t in_offset = 0;
  std::int32_t out_offset = 0;
  std::uint8_t enable_inter_ddt = 0;
  std::uint8_t is_inter = 0;
  std::uint8_t use_intrabc = 0;
  std::uint8_t bit_depth = 8;
};

enum class Direction : std::uint8_t {
  kForward = 0,
  kInverse = 1,
};

// Sizing request.  `max_blocks` is the largest batch a caller intends to pass;
// `with_levels` reserves the forward level output; `inverse_from_levels` is
// accepted for symmetry (the inverse input is int32 either way, so it does not
// change the byte count).
struct BatchConfig {
  Direction direction = Direction::kForward;
  int max_blocks = 1;
  bool with_levels = false;
  bool inverse_from_levels = false;
};

namespace detail {

constexpr std::uint64_t align_up_arena(std::uint64_t bytes) noexcept {
  constexpr std::uint64_t kMask =
      static_cast<std::uint64_t>(Arena::kAlignment) - 1;
  return (bytes + kMask) & ~kMask;
}

struct WorkspaceBytes {
  std::uint64_t jobs = 0;
  std::uint64_t input = 0;
  std::uint64_t coeff = 0;
  std::uint64_t levels = 0;
  std::uint64_t coded = 0;
};

inline WorkspaceBytes workspace_bytes(const BatchConfig& config) noexcept {
  WorkspaceBytes bytes;
  if (config.max_blocks <= 0) return bytes;
  const std::uint64_t blocks = static_cast<std::uint64_t>(config.max_blocks);
  bytes.jobs = static_cast<std::uint64_t>(sizeof(DdtJob)) * blocks;
  if (config.direction == Direction::kForward) {
    bytes.input = 2ull * 1024ull * blocks;   // int16 adjusted residual
    bytes.coeff = 4ull * 1024ull * blocks;   // int32 coefficients
    if (config.with_levels) {
      bytes.levels = 4ull * 1024ull * blocks;
    }
  } else {
    bytes.input = 4ull * 1024ull * blocks;   // int32 dequant or levels
    bytes.coded = 2ull * 4096ull * blocks;   // int16 coded residual
  }
  return bytes;
}

}  // namespace detail

// Exact arena bytes for the largest batch described by `config`.  Includes the
// per-array 256-byte alignment the bump allocator applies, so
// `arena.init(required_device_bytes(config))` is sufficient and one byte less
// is not.
inline std::uint64_t required_device_bytes(const BatchConfig& config) noexcept {
  const detail::WorkspaceBytes b = detail::workspace_bytes(config);
  std::uint64_t total = 0;
  const auto add = [&total](std::uint64_t bytes) {
    if (bytes != 0) total += detail::align_up_arena(bytes);
  };
  add(b.jobs);
  add(b.input);
  add(b.coeff);
  add(b.levels);
  add(b.coded);
  return total;
}

// Split for diagnostics/tests.
inline std::uint64_t required_device_bytes_split(const BatchConfig& config,
                                                 std::uint64_t* jobs,
                                                 std::uint64_t* input,
                                                 std::uint64_t* output) noexcept {
  const detail::WorkspaceBytes b = detail::workspace_bytes(config);
  if (jobs != nullptr) *jobs = detail::align_up_arena(b.jobs);
  if (input != nullptr) *input = detail::align_up_arena(b.input);
  if (output != nullptr) {
    *output = detail::align_up_arena(b.coeff) +
              detail::align_up_arena(b.levels) +
              detail::align_up_arena(b.coded);
  }
  return required_device_bytes(config);
}

enum class BatchStatus : int {
  kOk = 0,
  kInvalidArgument,
  kArenaTooSmall,
  kCudaError,
};

const char* batch_status_name(BatchStatus status);

// Encoder forward for a mixed batch.  `host_residual_adj` holds each block's
// adjusted-size residual at `job.in_offset`; `host_coeff` receives the integer
// coefficients at `job.out_offset`.  When `host_levels` is non-null it also
// receives `level_from_coeff(coeff)` at the same offsets.
//
// The call uploads, launches one kernel, downloads and rewinds the arena, so it
// synchronizes `stream` (like the other host-in/host-out convenience paths).
// Device-resident callers should use forward_batch_device().
BatchStatus forward_batch(Arena& arena, const DdtJob* jobs, int job_count,
                          const std::int16_t* host_residual_adj,
                          std::int32_t* host_coeff,
                          std::int32_t* host_levels = nullptr,
                          cudaStream_t stream = nullptr,
                          std::uint64_t* required_bytes = nullptr,
                          std::string* error = nullptr);

// Decoder inverse for a mixed batch.  `host_input` holds int32 dequant values
// at `job.in_offset` when `input_is_levels == false`, or quantised levels when
// it is true (the kernel then applies §7.14.4 dequant).  `host_residual_coded`
// receives `coded_w * coded_h` int16 samples at `job.out_offset`.
BatchStatus inverse_batch(Arena& arena, const DdtJob* jobs, int job_count,
                          const std::int32_t* host_input,
                          std::int16_t* host_residual_coded,
                          bool input_is_levels = false,
                          cudaStream_t stream = nullptr,
                          std::uint64_t* required_bytes = nullptr,
                          std::string* error = nullptr);

// Pure launchers over caller-owned device buffers.  No allocation, no
// synchronization, one kernel launch; return cudaGetLastError().
cudaError_t forward_batch_device(const std::int16_t* d_residual_adj,
                                 std::int32_t* d_coeff, std::int32_t* d_levels,
                                 const DdtJob* d_jobs, int job_count,
                                 cudaStream_t stream = nullptr);

cudaError_t inverse_batch_device(const std::int32_t* d_input,
                                 std::int16_t* d_residual_coded,
                                 const DdtJob* d_jobs, int job_count,
                                 bool input_is_levels = false,
                                 cudaStream_t stream = nullptr);

// True when every field of `job` is a legal AV2 transform configuration for
// the requested direction (used by the arena entry points).  When `error` is
// non-null it receives a description of the first failure.
bool valid_job(const DdtJob& job, Direction direction, std::string* error);

}  // namespace av2enc::cuda::ddt
