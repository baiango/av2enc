// Host-only helpers for the M2 video intra candidate generator.
//
// Split from video_intra.cu so the host-emulation test can compile and test
// the candidate expansion and §5.20.9.1 availability derivation without nvcc.
#ifndef AV2ENC_CUDA_PIPELINE_VIDEO_INTRA_VIDEO_INTRA_HOST_H_
#define AV2ENC_CUDA_PIPELINE_VIDEO_INTRA_VIDEO_INTRA_HOST_H_

#include <algorithm>

#include "cuda/pipeline/video/intra/video_intra.h"

namespace av2enc::cuda::video::intra {

inline const Candidate* luma_candidates() noexcept {
  static const struct Table {
    Candidate list[kMaxCandidates];
    int count = 0;
    Table() {
      count = build_candidates(0, kAllModesMask, 1, list, kMaxCandidates);
    }
  } table;
  return table.list;
}

inline int luma_candidate_count() noexcept { return kMaxCandidates; }

inline int build_candidates(int plane, std::uint32_t mode_mask,
                            int include_angle_deltas, Candidate* out,
                            int capacity) noexcept {
  if (capacity < 0 || (capacity > 0 && out == nullptr)) {
    return -1;
  }
  if (plane < 0 || plane > 2) {
    return -1;
  }
  if (plane > 0) {
    // Chroma is DC only (M2 scope): the shared predictor applies the chroma
    // derivation; the full UV directional search is not wired.
    if (capacity < 1) {
      return -1;
    }
    out[0] = Candidate{};
    return 1;
  }
  if (mode_mask == 0 || (mode_mask & ~kAllModesMask) != 0) {
    return -1;
  }
  int count = 0;
  for (int mode = 0; mode < kNumBaseModes; ++mode) {
    if ((mode_mask & (1u << static_cast<unsigned>(mode))) == 0) {
      continue;
    }
    if (count >= capacity) {
      return -1;
    }
    out[count].mode = static_cast<std::uint8_t>(mode);
    out[count].angle_delta = 0;
    ++count;
  }
  if (include_angle_deltas != 0) {
    constexpr int kDeltas[6] = {-3, -2, -1, 1, 2, 3};
    for (int mode = static_cast<int>(intra_modes::kVpred);
         mode <= static_cast<int>(intra_modes::kD67Pred); ++mode) {
      if ((mode_mask & (1u << static_cast<unsigned>(mode))) == 0) {
        continue;
      }
      for (int delta : kDeltas) {
        if (count >= capacity) {
          return -1;
        }
        out[count].mode = static_cast<std::uint8_t>(mode);
        out[count].angle_delta = static_cast<std::int8_t>(delta);
        ++count;
      }
    }
  }
  return count;
}

inline Availability derive_availability(const AvailRequest& request) noexcept {
  Availability out;
  if (request.plane_w <= 0 || request.plane_h <= 0 || request.w <= 0 ||
      request.h <= 0) {
    return out;
  }
  const int tile_x0 = std::max(0, request.tile_x0);
  const int tile_y0 = std::max(0, request.tile_y0);
  const int tile_x1 =
      request.tile_x1 > 0 ? std::min(request.tile_x1, request.plane_w)
                          : request.plane_w;
  const int tile_y1 =
      request.tile_y1 > 0 ? std::min(request.tile_y1, request.plane_h)
                          : request.plane_h;
  out.max_x = tile_x1 - 1;
  out.max_y = tile_y1 - 1;
  if (out.max_x < request.x || out.max_y < request.y) {
    return out;
  }
  // §5.20.9.1: availability is tile-relative.
  out.have_left = (request.x > tile_x0 && request.x > 0) ? 1 : 0;
  out.have_above = (request.y > tile_y0 && request.y > 0) ? 1 : 0;

  // Above-right: in raster order the above row is reconstructed to the visible
  // tile edge.  §7.13.2.1 reads at most x0+w+4*num4-1, and the count itself is
  // bounded by w/4 4x4 units.
  if (out.have_above != 0) {
    const int right_px = out.max_x - (request.x + request.w) + 1;
    const int geometric = right_px > 0 ? right_px / 4 : 0;
    out.num4_above_right = std::min(geometric, request.w / 4);
  }
  if (request.decoded_above_right4 >= 0) {
    out.num4_above_right =
        std::min(out.num4_above_right, request.decoded_above_right4);
  }

  // Below-left: the row below the leaf is not reconstructed in raster order,
  // so the geometric default is 0.  A caller replaying §6.19.2.3's
  // BlockDecoded map can pass the measured count (bounded by h/4).
  if (out.have_left != 0 && request.decoded_below_left4 < 0) {
    out.num4_below_left = 0;
  } else if (out.have_left != 0) {
    const int below_px = out.max_y - (request.y + request.h) + 1;
    const int geometric = below_px > 0 ? below_px / 4 : 0;
    out.num4_below_left =
        std::min(request.decoded_below_left4, std::min(geometric, request.h / 4));
  }
  return out;
}

}  // namespace av2enc::cuda::video::intra

#endif  // AV2ENC_CUDA_PIPELINE_VIDEO_INTRA_VIDEO_INTRA_HOST_H_
