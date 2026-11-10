// CUDA AV2 §8.2 entropy stage: kernels + host launch helpers.
//
// One block per tile.  Lane 0 runs the serial range coder; the other 31 lanes
// only participate in the block-level CDF staging copy before and after the
// serial loop.  No allocation, no synchronization, no launch-loop host work.
//
// See entropy.h for the API contract and README-entropy.md for the measured
// latency/occupancy results.

#include "entropy.h"

namespace av2enc::cuda {
namespace {

constexpr uint32_t kDefaultSmemStageBytes = kEntropyStageBytes;
constexpr uint32_t kDefaultCdfStageEntries = kEntropyCdfStageEntries;

__device__ __forceinline__ uint32_t status_code(Status status) noexcept {
  return static_cast<uint32_t>(status);
}

// Default frozen.  Adaptation requires the explicit opt-in flag and is
// overridden by an explicit freeze.
__device__ __forceinline__ bool cdf_updates_enabled(uint32_t flags) noexcept {
  return ((flags & kEntropyEnableCdfUpdate) != 0u) &&
         ((flags & kEntropyDisableCdfUpdate) == 0u);
}

__device__ __forceinline__ bool cdf_staging_enabled(
    const EntropyEncodeDesc& desc) noexcept {
  if ((desc.flags & kEntropyDisableStaging) != 0u || desc.cdfs == nullptr ||
      desc.cdf_count == 0u || desc.cdf_entries == 0u) {
    return false;
  }
  const uint64_t total =
      static_cast<uint64_t>(desc.cdf_count) * desc.cdf_entries;
  return total <= kDefaultCdfStageEntries;
}

__device__ __forceinline__ bool cdf_staging_enabled(
    const EntropyDecodeDesc& desc) noexcept {
  if ((desc.flags & kEntropyDisableStaging) != 0u || desc.cdfs == nullptr ||
      desc.cdf_count == 0u || desc.cdf_entries == 0u) {
    return false;
  }
  const uint64_t total =
      static_cast<uint64_t>(desc.cdf_count) * desc.cdf_entries;
  return total <= kDefaultCdfStageEntries;
}

// Fill any unused descriptor padding with zero so the kernel reads a fully
// defined POD (host side does the same in the launch paths).
__device__ __forceinline__ void zero_result(EntropyTileResult* r) noexcept {
  r->status = status_code(Status::kOk);
  r->payload_bytes = 0;
  r->symbol_count = 0;
  r->bool_count = 0;
  r->tape_entries_used = 0;
  r->cycles = 0;
}

}  // namespace

__global__ void __launch_bounds__(kEntropyBlockThreads, 8)
entropy_encode_kernel(const EntropyEncodeDesc* __restrict__ tiles,
                      EntropyTileResult* __restrict__ results,
                      uint32_t tile_count) {
  __shared__ uint8_t stage[kDefaultSmemStageBytes];
  __shared__ uint16_t cdf_stage[kDefaultCdfStageEntries];

  const uint32_t tile = blockIdx.x;
  if (tile >= tile_count) {
    return;  // block-uniform: whole block exits
  }
  const EntropyEncodeDesc desc = tiles[tile];

  const bool staged = cdf_staging_enabled(desc);
  if (staged) {
    const uint32_t total = desc.cdf_count * desc.cdf_entries;
    for (uint32_t i = threadIdx.x; i < total; i += kEntropyBlockThreads) {
      cdf_stage[i] = desc.cdfs[i];
    }
  }
  // Zero the output staging window cooperatively; the serial lane then skips
  // its own 4 KiB clear (a single-lane clear was ~130k cycles per tile).
  const bool use_out_stage = (desc.flags & kEntropyDisableStaging) == 0u;
  if (use_out_stage) {
    for (uint32_t i = threadIdx.x; i < kDefaultSmemStageBytes;
         i += kEntropyBlockThreads) {
      stage[i] = 0;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    EntropyTileResult result;
    zero_result(&result);

    const uint64_t t0 = clock64();
    TileEncoder encoder;
    encoder.disable_cdf_update =
        cdf_updates_enabled(desc.flags) ? 0u : 1u;

    uint8_t* out_stage = use_out_stage ? stage : nullptr;
    const uint32_t out_stage_bytes =
        (out_stage != nullptr) ? kDefaultSmemStageBytes : 0u;

    Status status = encoder.init(desc.out, desc.out_capacity, out_stage,
                                 out_stage_bytes, /*stage_zeroed=*/true);
    uint32_t used = 0;
    if (status == Status::kOk) {
      if (desc.tape == nullptr && desc.tape_entries != 0u) {
        status = Status::kInvalidArgument;
      } else if (staged) {
        // Pass the shared array directly: a runtime-selected generic pointer
        // here makes every CDF access a generic load (~2x slower on sm_86).
        status = detail::encode_tape(encoder, desc.tape, desc.tape_entries,
                                     cdf_stage, desc.cdf_count,
                                     desc.cdf_entries, &used);
      } else {
        status = detail::encode_tape(encoder, desc.tape, desc.tape_entries,
                                     desc.cdfs, desc.cdf_count,
                                     desc.cdf_entries, &used);
      }
    }
    size_t payload_bytes = 0;
    if (status == Status::kOk) {
      status = encoder.exit_symbol(&payload_bytes);
    }
    const uint64_t t1 = clock64();

    result.status = status_code(status);
    result.payload_bytes =
        (status == Status::kOk) ? static_cast<uint32_t>(payload_bytes) : 0u;
    result.symbol_count = encoder.symbol_count;
    result.bool_count = encoder.bool_count;
    result.tape_entries_used = static_cast<uint64_t>(used);
    result.cycles = t1 - t0;
    results[tile] = result;
  }
  __syncthreads();

  if (staged) {
    const uint32_t total = desc.cdf_count * desc.cdf_entries;
    for (uint32_t i = threadIdx.x; i < total; i += kEntropyBlockThreads) {
      desc.cdfs[i] = cdf_stage[i];
    }
  }
}

__global__ void __launch_bounds__(kEntropyBlockThreads, 8)
entropy_decode_kernel(const EntropyDecodeDesc* __restrict__ tiles,
                      EntropyTileResult* __restrict__ results,
                      uint32_t tile_count) {
  __shared__ uint16_t cdf_stage[kDefaultCdfStageEntries];

  const uint32_t tile = blockIdx.x;
  if (tile >= tile_count) {
    return;  // block-uniform: whole block exits
  }
  const EntropyDecodeDesc desc = tiles[tile];

  const bool staged = cdf_staging_enabled(desc);
  if (staged) {
    const uint32_t total = desc.cdf_count * desc.cdf_entries;
    for (uint32_t i = threadIdx.x; i < total; i += kEntropyBlockThreads) {
      cdf_stage[i] = desc.cdfs[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    EntropyTileResult result;
    zero_result(&result);

    const uint64_t t0 = clock64();
    TileDecoder decoder;
    decoder.disable_cdf_update =
        cdf_updates_enabled(desc.flags) ? 0u : 1u;

    Status status = decoder.init(desc.in, desc.backing_bytes,
                                 desc.payload_bytes);
    uint32_t used = 0;
    if (status == Status::kOk) {
      if (desc.tape == nullptr && desc.tape_entries != 0u) {
        status = Status::kInvalidArgument;
      } else if (staged) {
        status = detail::decode_tape(decoder, desc.tape, desc.tape_entries,
                                     cdf_stage, desc.cdf_count,
                                     desc.cdf_entries, &used);
      } else {
        status = detail::decode_tape(decoder, desc.tape, desc.tape_entries,
                                     desc.cdfs, desc.cdf_count,
                                     desc.cdf_entries, &used);
      }
    }
    if (status == Status::kOk) {
      status = decoder.exit_symbol();
    }
    const uint64_t t1 = clock64();

    result.status = status_code(status);
    result.payload_bytes = desc.payload_bytes;
    result.symbol_count = decoder.symbol_count;
    result.bool_count = decoder.bool_count;
    result.tape_entries_used = static_cast<uint64_t>(used);
    result.cycles = t1 - t0;
    results[tile] = result;
  }
  __syncthreads();

  if (staged) {
    const uint32_t total = desc.cdf_count * desc.cdf_entries;
    for (uint32_t i = threadIdx.x; i < total; i += kEntropyBlockThreads) {
      desc.cdfs[i] = cdf_stage[i];
    }
  }
}

cudaError_t launch_entropy_encode(const EntropyEncodeDesc* d_tiles,
                                  EntropyTileResult* d_results,
                                  uint32_t tile_count,
                                  cudaStream_t stream) {
  if (tile_count == 0u) {
    return cudaSuccess;
  }
  if (d_tiles == nullptr || d_results == nullptr ||
      tile_count > 0x7FFFFFFFu) {
    return cudaErrorInvalidValue;
  }
  entropy_encode_kernel<<<tile_count, kEntropyBlockThreads, 0, stream>>>(
      d_tiles, d_results, tile_count);
  return cudaGetLastError();
}

cudaError_t launch_entropy_decode(const EntropyDecodeDesc* d_tiles,
                                  EntropyTileResult* d_results,
                                  uint32_t tile_count,
                                  cudaStream_t stream) {
  if (tile_count == 0u) {
    return cudaSuccess;
  }
  if (d_tiles == nullptr || d_results == nullptr ||
      tile_count > 0x7FFFFFFFu) {
    return cudaErrorInvalidValue;
  }
  entropy_decode_kernel<<<tile_count, kEntropyBlockThreads, 0, stream>>>(
      d_tiles, d_results, tile_count);
  return cudaGetLastError();
}

}  // namespace av2enc::cuda
