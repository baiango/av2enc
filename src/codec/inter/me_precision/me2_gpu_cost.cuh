// Warp-cooperative cost functor shared by the me2 CUDA kernel and the
// host-emulation parity test.
//
// The arithmetic is the same as src/cuda/inter/me/me.cu: SAD rows (or 8x8 SATD
// tiles) are split across the 32 lanes and reduced with integer warp
// shuffles.  Integer addition is associative, so the reduced result equals
// me::me_cost_plane() exactly.  The emulation test sums the per-lane partials
// on the host and compares both the costs and the full search results.

#ifndef AV2ENC_ENCODER_INTER_ME2_ME2_GPU_COST_CUH_
#define AV2ENC_ENCODER_INTER_ME2_ME2_GPU_COST_CUH_

#include <cstdint>

#include "codec/inter/me/me_core.h"

#if defined(__CUDACC__)
#define AV2ENC_ME2_GPU_HD __device__ __forceinline__
#else
#define AV2ENC_ME2_GPU_HD inline
#endif

namespace av2enc::me2 {
namespace gpu {

// One lane's partial cost.  `lanes` is 32 in the warp kernel; the emulation
// test calls this for lane = 0..lanes-1 and sums the results.
AV2ENC_ME2_GPU_HD std::int32_t me2_cost_lane(
    const me::MeCorePlane& cur, const me::MeCorePlane& ref, int x, int y,
    int w, int h, std::int32_t mv_x, std::int32_t mv_y, int metric,
    int filter, int lane, int lanes) {
  std::int32_t local = 0;
  if (metric == me::kMeSad) {
    if (me::me_is_integer_mv(mv_x, mv_y)) {
      local = me::me_sad_rows(cur, ref, x, y, w, mv_x,
                              mv_y >> me::kSubpelBits, lane, h, lanes);
    } else {
      local = me::me_sad_subpel_rows(cur, ref, x, y, w, mv_x, mv_y, filter,
                                     lane, h, lanes);
    }
  } else {
    const int tiles_x = (w + 7) >> 3;
    const int tiles_y = (h + 7) >> 3;
    const int units = tiles_x * tiles_y;
    for (int u = lane; u < units; u += lanes) {
      const int tx = (u % tiles_x) << 3;
      const int ty = (u / tiles_x) << 3;
      local += me::me_satd_unit(cur, ref, x, y, w, h, mv_x, mv_y, filter, tx,
                                ty);
    }
  }
  return local;
}

#if defined(__CUDACC__)

__device__ __forceinline__ std::int32_t me2_warp_sum(std::int32_t v) {
  v += __shfl_down_sync(0xffffffffu, v, 16);
  v += __shfl_down_sync(0xffffffffu, v, 8);
  v += __shfl_down_sync(0xffffffffu, v, 4);
  v += __shfl_down_sync(0xffffffffu, v, 2);
  v += __shfl_down_sync(0xffffffffu, v, 1);
  return __shfl_sync(0xffffffffu, v, 0);
}

struct Me2WarpCost {
  __device__ __noinline__ std::int32_t operator()(
      const me::MeCorePlane& cur, const me::MeCorePlane& ref, int x, int y,
      int w, int h, std::int32_t mv_x, std::int32_t mv_y, int metric,
      int filter) const {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    return me2_warp_sum(me2_cost_lane(cur, ref, x, y, w, h, mv_x, mv_y, metric,
                                      filter, lane, 32));
  }
};

#endif  // __CUDACC__

}  // namespace gpu
}  // namespace av2enc::me2

#endif  // AV2ENC_ENCODER_INTER_ME2_ME2_GPU_COST_CUH_
