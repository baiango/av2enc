// CUDA §7.13.3.15 sub-PU grid — public API.
//
// This component builds the decoder's SubPuSize / SubPuColBase / SubPuRowBase
// grids, the metadata §7.17 deblocking reads (SubPuSize in filt_max_size,
// SubPuColBase/SubPuRowBase for the sub-PU edge test) and the reference-MV
// sub-PU path records through §7.13.3.14.  The arithmetic lives in the shared
// `subpu_device.cuh`, which is also compiled by the host for parity.
//
// Scope:
//   * `find_tx_size`, Tx_Width/Tx_Height and the TxSize ordering;
//   * SubPuGrid construction (reset) and the §7.13.3.15 write process;
//   * luma-MI indexing and independent chroma subsampling in x/y;
//   * the §7.13.3.14 early return when the region equals the PU.
//
// Memory contract (mandatory, project-wide)
// -----------------------------------------
// Every device byte comes from the single persistent `av2enc::cuda::Arena`.
// There is no cudaMalloc / cudaFree / cudaMallocHost in this module or its
// launchers.  `required_device_bytes(config)` is the exact arena requirement:
//
//   persistent : two storage planes of size/col_base/row_base, one job staging
//                array, one per-job result array, and the Tx table
//
// `allocate_checked()` takes one `Arena::mark()`, bump-allocates exactly
// `required_device_bytes(config)` and rewinds on failure.  A short arena
// returns `SubPuStatus::kArenaTooSmall` with `*required_bytes` set to the exact
// requirement and the arena bump pointer unchanged.  `mark()` / `rewind()` are
// exposed for callers that want to bracket a larger scratch region themselves;
// `deallocate()` rewinds to the allocation mark.
//
// Launchers never allocate and never grow the arena.  `save_batch()` stages
// descriptors into the persistent job array and owns its stream
// synchronisation, so two back-to-back launches leave `used()` and
// `peak_used()` flat.
#ifndef AV2ENC_CUDA_SUBPU_SUBPU_H_
#define AV2ENC_CUDA_SUBPU_SUBPU_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/codec/filters/deblock/subpu/subpu_device.cuh"

namespace av2enc::cuda::subpu {

// Largest number of sub-PU jobs one batch may carry.  Bounds the staging and
// result arrays in required_device_bytes().
inline constexpr int kSubPuMaxJobs = 1 << 20;

// Sentinel for "this buffer set was never allocated".
inline constexpr std::uint64_t kNoArenaMark = ~static_cast<std::uint64_t>(0);

// Sizing request.  `max_jobs` is the batch capacity; `mi_rows`/`mi_cols` are
// the luma MI grid the two storage planes are indexed on.
struct SubPuConfig {
  int max_jobs = 1;
  int mi_rows = 0;
  int mi_cols = 0;
};

// Exact arena bytes for one buffer set.  Every array is rounded to
// Arena::kAlignment, matching the bump allocator.  Returns 0 for an invalid
// configuration.
std::uint64_t required_device_bytes(const SubPuConfig& config) noexcept;

// Arena watermark helpers re-exported so callers can bracket their own scratch
// with the same alignment contract as this component.
std::uint64_t mark(const Arena& arena) noexcept;
bool rewind(Arena& arena, std::uint64_t mark_value) noexcept;

// One §7.13.3.15 invocation.  Mirrors the CPU reference arguments exactly.
struct SubPuJob {
  int plane = 0;          // 0 luma, 1 U, 2 V
  int x = 0;              // region origin, plane samples
  int y = 0;
  int w = 0;              // region extent, plane samples
  int h = 0;
  int subsampling_x = 0;  // only used when plane > 0
  int subsampling_y = 0;
  int pu_width = 0;       // PuWidth / PuHeight, plane samples
  int pu_height = 0;
};

// Per-job outcome (one of SubPuCode).  Transport errors are reported by the
// return status; a job the §7.13.3.15 process rejects is not a launch failure.
struct SubPuResult {
  int code = kSubPuOk;
};

// Device-resident grids.  Both slots are mi_rows*mi_cols entries.
struct SubPuDeviceGrid {
  std::uint8_t* size[kSubPuPlaneCount] = {nullptr, nullptr};
  std::int32_t* col_base[kSubPuPlaneCount] = {nullptr, nullptr};
  std::int32_t* row_base[kSubPuPlaneCount] = {nullptr, nullptr};
  int mi_rows = 0;
  int mi_cols = 0;

  SubPuGridView view() const noexcept {
    SubPuGridView out;
    for (int plane = 0; plane < kSubPuPlaneCount; ++plane) {
      out.size[plane] = size[plane];
      out.col_base[plane] = col_base[plane];
      out.row_base[plane] = row_base[plane];
    }
    out.mi_rows = mi_rows;
    out.mi_cols = mi_cols;
    return out;
  }
};

// Arena-owned buffers for one configuration.  All pointers are device
// pointers; `d_jobs` and `d_codes` are fixed-capacity staging arrays.
struct SubPuBuffers {
  SubPuDeviceGrid grid;
  SubPuJob* d_jobs = nullptr;
  SubPuResult* d_codes = nullptr;
  SubPuTxEntry* d_tx_table = nullptr;
  int max_jobs = 0;
  std::uint64_t persistent_mark = kNoArenaMark;
};

// Host mirror of the grid, used by tests and by callers that need to inspect
// metadata on the CPU.  Size and bases are per storage plane (0 luma,
// 1 chroma), both on the luma MI grid.
struct SubPuHostGrid {
  int mi_rows = 0;
  int mi_cols = 0;
  std::vector<std::uint8_t> size[kSubPuPlaneCount];
  std::vector<std::int32_t> col_base[kSubPuPlaneCount];
  std::vector<std::int32_t> row_base[kSubPuPlaneCount];

  void reset(int rows, int cols) {
    mi_rows = rows;
    mi_cols = cols;
    const std::size_t cells =
        static_cast<std::size_t>(rows < 0 ? 0 : rows) *
        static_cast<std::size_t>(cols < 0 ? 0 : cols);
    for (int plane = 0; plane < kSubPuPlaneCount; ++plane) {
      size[plane].assign(cells, 0);
      col_base[plane].assign(cells, 0);
      row_base[plane].assign(cells, 0);
    }
  }
};

// Distinct outcome of the allocating entry point.  kArenaTooSmall carries the
// number of bytes the arena must provide in `required_bytes`.
enum class SubPuStatus : int {
  kOk = 0,
  kInvalidArgument,
  kNotInitialized,
  kArenaTooSmall,
  kCudaError,
};

const char* subpu_status_name(SubPuStatus status) noexcept;

// Allocates the persistent grid and staging arrays from `arena` exactly once.
// On kArenaTooSmall the bump pointer is restored and `*required_bytes` is the
// exact required_device_bytes(config); no allocation fallback exists.
SubPuStatus allocate_checked(Arena& arena, const SubPuConfig& config,
                             SubPuBuffers* out,
                             std::uint64_t* required_bytes = nullptr,
                             std::string* error = nullptr);

// Rewinds the arena to the allocation mark and clears the pointers.  Safe to
// call on a default-constructed/unallocated buffer set.
void deallocate(Arena& arena, SubPuBuffers& buffers) noexcept;

// Zeroes both storage planes (SubPuSize/ColBase/RowBase), matching
// SubPuGrid::reset().
SubPuStatus reset_grid(const SubPuBuffers& buffers,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);

// Applies `count` §7.13.3.15 invocations in descriptor order and fills
// `host_results[i]` with the SubPuCode of job i.  `host_results` may be null.
// The entry point synchronises on `stream`; jobs that target overlapping cells
// within one batch must not be used unless their write values agree.
SubPuStatus save_batch(const SubPuBuffers& buffers, const SubPuJob* host_jobs,
                       int count, SubPuResult* host_results = nullptr,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);

// Convenience form of save_batch() for one job.
SubPuStatus save_one(const SubPuBuffers& buffers, const SubPuJob& job,
                     SubPuResult* result = nullptr,
                     cudaStream_t stream = nullptr,
                     std::string* error = nullptr);

// Fills the device Tx table with (size, Tx_Width, Tx_Height, find_tx_size
// round-trip) for all 25 TX sizes.
SubPuStatus build_tx_table(const SubPuBuffers& buffers,
                           cudaStream_t stream = nullptr,
                           std::string* error = nullptr);

// Copies the 25-row device Tx table into `out` (`capacity >= 25`).
SubPuStatus download_tx_table(const SubPuBuffers& buffers, SubPuTxEntry* out,
                              int capacity, cudaStream_t stream = nullptr,
                              std::string* error = nullptr);

// Copies the full grid (both storage planes) into `out`, resizing its vectors
// to mi_rows*mi_cols.  Synchronises on `stream`.
SubPuStatus download_grid(const SubPuBuffers& buffers, SubPuHostGrid* out,
                          cudaStream_t stream = nullptr,
                          std::string* error = nullptr);

}  // namespace av2enc::cuda::subpu

#endif  // AV2ENC_CUDA_SUBPU_SUBPU_H_
