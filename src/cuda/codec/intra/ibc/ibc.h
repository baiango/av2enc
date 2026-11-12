// AV2 intra block copy — CUDA device API.
//
// Device twin of the CPU reference in src/encoder/ibc/.  The kernels call the
// same host/device core (`ibc_core.h`): plane scaling, geometry/global BV
// validity, block copy and SAD.
//
// Scope: 8-bit 4:2:0, 64x64 superblocks, integer BVs.  Global IBC validity is
// evaluated on the device; the local four-buffer ring is small and stays on
// the CPU (the GPU copy kernel only needs the already-validated BV).  No
// hidden synchronisation and no global state.
//
// Allocation: ibc_search() stages one packed (SAD, candidate-index) word in
// the caller's Arena.  Everything else operates on caller-owned buffers, so
// there is no cudaMalloc here.  The search word is transient: it is released
// with Arena::rewind() before the call returns.

#ifndef AV2ENC_CUDA_IBC_IBC_H_
#define AV2ENC_CUDA_IBC_IBC_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "codec/intra/ibc/core/ibc.h"

namespace av2enc::cuda {

class Arena;

struct IbcDeviceFrame {
  const std::uint8_t* y = nullptr;
  const std::uint8_t* u = nullptr;
  const std::uint8_t* v = nullptr;
  int stride_y = 0;
  int stride_u = 0;
  int stride_v = 0;
  int width = 0;
  int height = 0;
  bool has_chroma = true;
  int sub_x = 1;
  int sub_y = 1;
};

struct IbcDeviceGeometry {
  int mi_cols = 0;
  int mi_rows = 0;
  int sb_size = 64;
  bool allow_global = true;
  bool allow_extra_ibc_range = false;
};

// Copy the source block into the destination planes (transform-skip
// prediction).  `visible_w/h` clamp the current block at the frame edge.
cudaError_t ibc_copy_block(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                           int bsize, av2enc::ibc::BlockVector bv,
                           std::uint8_t* dst_y, int dst_stride_y,
                           std::uint8_t* dst_u, int dst_stride_u,
                           std::uint8_t* dst_v, int dst_stride_v, int visible_w,
                           int visible_h, cudaStream_t stream = nullptr);

// SAD of `n` candidate BVs against the current block.  `bvs` and `sads` are
// device arrays; one CUDA block per candidate.
cudaError_t ibc_sad_batch(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                          int bsize, const av2enc::ibc::BlockVector* bvs,
                          int n, std::int64_t* sads,
                          cudaStream_t stream = nullptr);

// Global-only §6.19.7.12 validity for n BVs (`bvs`/`valid` are device arrays).
// Local buffers are a CPU concern; with allow_global=false this returns
// cudaErrorInvalidValue.
cudaError_t ibc_validate_global(const IbcDeviceGeometry& geometry, int mi_row,
                                int mi_col, int bsize,
                                const av2enc::ibc::BlockVector* bvs, int n,
                                std::uint8_t* valid,
                                cudaStream_t stream = nullptr);

// Brute-force search over a `step`-pixel grid of previously reconstructed
// positions.  Writes the best (row-major) BV and SAD.  Only global IBC
// validity is applied; the caller must reject local-only BVs if needed.
//
// Device scratch: one packed word, bump-allocated from `arena` and rewound
// before return.  `ibc_required_device_bytes()` reports the capacity that
// scratch needs (including arena alignment).
struct IbcSearchConfig {
  int width = 0;
  int height = 0;
  int bsize = 0;
  int step = 1;
};

std::uint64_t ibc_required_device_bytes(const IbcSearchConfig& config);

// Distinct outcome for the allocating entry point.  kArenaTooSmall carries the
// number of bytes the arena must be able to provide in `required_bytes`.
enum class IbcSearchStatus : int {
  kOk = 0,
  kInvalidArgument,
  kArenaTooSmall,
  kCudaError,
};

IbcSearchStatus ibc_search(Arena& arena, const IbcDeviceFrame& frame,
                           const IbcDeviceGeometry& geometry, int mi_row,
                           int mi_col, int bsize, int step,
                           av2enc::ibc::BlockVector* best,
                           std::int64_t* best_sad,
                           cudaStream_t stream = nullptr,
                           std::uint64_t* required_bytes = nullptr);

}  // namespace av2enc::cuda

#endif  // AV2ENC_CUDA_IBC_IBC_H_
