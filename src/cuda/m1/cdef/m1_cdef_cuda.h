// M1 GPU path — CDEF device entry points.
//
// Included only by CUDA translation units (.cu) and the CUDA parity test; the
// stage contract in `m1_cdef.h` stays CUDA-free.
//
// Both entry points are self-contained: they launch kernels defined in
// `m1_cdef.cu` and do not require `src/cuda/cdef/cdef.cu` to be part of the
// link.  The arithmetic is shared with the CPU reference through
// `encoder/filters/cdef/cdef_core.h` / `cdef_search.h`, so the outputs are
// bit-exact for identical inputs.
//
// Contract: no allocations, no synchronisation, no global state.  The caller
// owns the arena, the uploads and the stream.
#ifndef AV2ENC_CUDA_M1_CDEF_CUDA_H_
#define AV2ENC_CUDA_M1_CDEF_CUDA_H_

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda/m1/cdef/m1_cdef.h"

namespace av2enc::cuda::m1 {

// §7.18 filter pass over every 8x8 luma CDEF block.  `frame.dst` receives the
// CdefFrame (copy for skip/lossless/`cdef_idx == -1`, filtered otherwise).
// One launch, 64 threads per 8x8 block.
cudaError_t m1_cdef_filter(const CdefDeviceFrame& frame,
                           const cdef::CdefOptions& options,
                           cudaStream_t stream = nullptr);

// Strength-search SSE pass (encoder side, §7.18 block filter used as the
// distortion metric).  One launch; fills `d_sse` with
// `candidate_count * n64_cols * n64_rows` int64 values laid out
// `[candidate][block]` with blocks in raster order -- the same table
// `cdef::cdef_search_plan()` builds on the host, so
// `cdef::search_detail::select()` produces the identical decision.
//
// `ref` / `ref_stride` are device pointers to the original picture planes
// (tight or padded, visible area only is compared).  The SSE includes both
// chroma planes.  No host synchronisation happens here.
cudaError_t m1_cdef_search_sse(const CdefDeviceFrame& frame,
                               const std::uint8_t* const ref[3],
                               const int ref_stride[3],
                               const cdef::CdefStrength* candidates,
                               int candidate_count, int damping,
                               bool on_skip_txfm,
                               bool disable_loopfilters_across_tiles,
                               std::int64_t* d_sse,
                               cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::m1

#endif  // AV2ENC_CUDA_M1_CDEF_CUDA_H_
