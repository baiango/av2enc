// Host-only full intra mode search shared by the CPU M2 reference and the
// CPU-emulation backend.
//
// The M2 video coder used to hard-code every intra leaf to DC_PRED.  With
// `VideoOptions::enable_intra_full` (CLI `--video-intra-full`) the luma leaf
// runs the still product's 61-candidate search:
//
//   intra::build_mode_candidates(out, /*include_angle_deltas=*/true)
//     = DC, V, H, D45, D135, D113, D157, D203, D67, SMOOTH, SMOOTH_V,
//       SMOOTH_H, PAETH, then the six non-zero angle deltas for each of the
//       eight directional modes.
//
// Cost model and tie-break are the still encoder's: SSE of the prediction
// against the (clamped) source, strict `<` in candidate order, so the earlier
// candidate wins a tie.  Chroma stays DC_PRED (the M2 syntax writer cannot
// signal a chroma directional search, see docs/negative-results.md).
//
// Availability is §5.20.9.1 `is_inside()` plus the §7.13.2.1 extension bounds,
// derived from a §6.19.2.3 `BlockDecoded` replay for the current superblock
// (the still encoder's `count_top_right()` / `count_bottom_left()` rule).  The
// M2 backend's reconstruction planes are tile-local and multi-tile is rejected,
// so the tile origin is (0, 0) and the plane extents are MI-aligned.
//
// This header has no CUDA dependency; it is compiled into both the CPU
// reference tree (`src/pipeline/video/video_frame.cpp`) and the emulation
// backend (`src/cuda/pipeline/video/video_backend_cpu.cpp`).  The real CUDA
// backend consumes the verified device module
// (`src/cuda/pipeline/video/intra/`) instead, so no predictor math is
// transcribed here: every sample comes from `av2enc::intra::predict_intra`.
#ifndef AV2ENC_PIPELINE_VIDEO_VIDEO_INTRA_SEARCH_H_
#define AV2ENC_PIPELINE_VIDEO_VIDEO_INTRA_SEARCH_H_

#include <algorithm>
#include <array>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "codec/intra/ibp/ibp.h"
#include "codec/intra/modes/intra_modes.h"

namespace av2enc::video::intra_search {

struct Availability {
  int max_x = 0;  // last visible column in the plane (inclusive)
  int max_y = 0;
  int have_left = 0;
  int have_above = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;
};

// §6.19.2.3 BlockDecoded replay for one plane.  The decoder resets the map at
// every superblock (`clear_block_decoded_flags`) and only the current
// superblock's cells are meaningful, so the encoders reset it when they enter a
// superblock root and fill each coded leaf.  This is the still encoder's
// `DecodedMap` (src/pipeline/still/still_encoder.cpp).
class BlockDecoded {
 public:
  void reset(int cols, int rows) {
    cols_ = cols;
    rows_ = rows;
    cells_.assign(static_cast<std::size_t>(cols) *
                      static_cast<std::size_t>(rows),
                  0);
  }
  bool at(int x, int y) const {
    if (x < 0 || y < 0 || x >= cols_ || y >= rows_) return false;
    return cells_[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(cols_) +
                  static_cast<std::size_t>(x)] != 0;
  }
  void fill(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < cols_ && y + j < rows_) {
          cells_[static_cast<std::size_t>(y + j) *
                      static_cast<std::size_t>(cols_) +
                  static_cast<std::size_t>(x + i)] = 1;
        }
      }
    }
  }

 private:
  int cols_ = 0;
  int rows_ = 0;
  std::vector<std::uint8_t> cells_;
};

// §5.20.9.1 availability plus the §7.13.2.1 extension counts for one tile-local
// plane, derived exactly like the still encoder's `count_top_right()` /
// `count_bottom_left()` from the current superblock's BlockDecoded map.
//
//   x4/y4/w4/h4   leaf origin/size in 4x4 units of this plane
//   plane_mi_*    plane extent in 4x4 units
//   sb_mi         superblock size in 4x4 units (16 luma, 8 chroma)
//
// `max_x`/`max_y` are the MI-aligned reconstruction plane extents, matching
// the still encoder (`mi_cols * 4 - 1`).
inline Availability availability(const BlockDecoded& map, int x4, int y4,
                                 int w4, int h4, int plane_mi_cols,
                                 int plane_mi_rows, int sb_mi) noexcept {
  Availability out;
  if (plane_mi_cols <= 0 || plane_mi_rows <= 0 || w4 <= 0 || h4 <= 0 ||
      sb_mi <= 0) {
    return out;
  }
  out.max_x = plane_mi_cols * 4 - 1;
  out.max_y = plane_mi_rows * 4 - 1;
  if (x4 < 0 || y4 < 0 || x4 >= plane_mi_cols || y4 >= plane_mi_rows) {
    return out;
  }
  if (sb_mi <= 0) return out;
  const int sb_c0 = (x4 / sb_mi) * sb_mi;
  const int sb_r0 = (y4 / sb_mi) * sb_mi;
  const int sb_w = std::min(sb_mi, plane_mi_cols - sb_c0);
  const int sb_h = std::min(sb_mi, plane_mi_rows - sb_r0);
  const int rel_x = x4 - sb_c0;
  const int rel_y = y4 - sb_r0;
  // §5.20.9.1: availability is tile-relative (the planes are tile-local).
  out.have_left = (x4 > 0) ? 1 : 0;
  out.have_above = (y4 > 0) ? 1 : 0;

  // Above-right, exactly count_top_right(): the top border of an SB's
  // BlockDecoded map extends two superblocks to the right only on the SB's own
  // top row; otherwise availability stops at the SB's right edge.
  if (out.have_above != 0) {
    const int border_w = std::min(2 * sb_mi, plane_mi_cols - sb_c0);
    int count = 0;
    for (int i = 0; i < w4; ++i) {
      const int x = rel_x + w4 + i;
      if (x >= (rel_y == 0 ? border_w : sb_w)) break;
      if (rel_y - 1 >= 0 && !map.at(sb_c0 + x, sb_r0 + rel_y - 1)) break;
      ++count;
    }
    out.num4_above_right = count;
  }
  // Below-left, exactly count_bottom_left(): never crosses into the next SB
  // row, and only cells already coded in this SB count (a taller left
  // neighbour can make below-left available in the middle of an SB).
  if (out.have_left != 0) {
    int count = 0;
    for (int i = 0; i < h4; ++i) {
      const int y = rel_y + h4 + i;
      if (y >= sb_h) break;
      if (rel_x - 1 >= 0 && !map.at(sb_c0 + rel_x - 1, sb_r0 + y)) break;
      ++count;
    }
    out.num4_below_left = count;
  }
  return out;
}

// §7.13.2.15/.16 filterTypeAbove/Left: the neighbouring coding block codes a
// SMOOTH/SMOOTH_V/SMOOTH_H mode.  `joint` is the §5.20.5.5 IntraJointMode
// (DC=0, SMOOTH=1, SMOOTH_V=2, SMOOTH_H=3, PAETH=4, 5+ = directional).
inline bool joint_is_smooth(int joint) noexcept {
  return joint >= 1 && joint <= 3;
}

struct LumaChoice {
  int mode = 0;         // av2enc::intra::Mode / AV2 YMode value
  int angle_delta = 0;  // -3..+3
  std::int64_t sse = 0;
};

namespace detail {

inline int source_sample(const std::uint8_t* source, int stride, int max_x,
                         int max_y, int x, int y) noexcept {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x > max_x) x = max_x;
  if (y > max_y) y = max_y;
  return source[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                 static_cast<std::size_t>(x)];
}

inline std::int64_t sse_against_source(const std::uint8_t* source, int stride,
                                       int max_x, int max_y, int x0, int y0,
                                       int w, int h,
                                       const std::uint8_t* pred) noexcept {
  std::int64_t sse = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int src = source_sample(source, stride, max_x, max_y, x0 + x,
                                    y0 + y);
      const int d = src - static_cast<int>(
                               pred[static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(x)]);
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

}  // namespace detail

// Runs the 61-candidate luma search.  `pred` receives the winning w*h
// predictor (stride w).  Returns false only when the component rejects the
// geometry or a mode (it should not for a legal leaf).
//
// With `enable_ibp` every candidate is evaluated with the §7.13.2 IBP
// predictor (`ibp::predict`).  The decoder derives applyIbp/useIBP from the
// sequence flag and the signalled (mode, angle_delta), so the search cannot
// mix IBP and non-IBP candidates: the whole candidate set must agree with the
// reconstruction path.  The 61 candidates already contain every pair that
// triggers the two-angle blend for a non-4x4 luma block.
inline bool search_luma(const std::uint8_t* recon, int recon_stride, int x0,
                        int y0, int w, int h,
                        const Availability& avail, bool above_smooth,
                        bool left_smooth, bool enable_edge_filter,
                        const std::uint8_t* source, int source_stride,
                        int source_max_x, int source_max_y, std::uint8_t* pred,
                        LumaChoice& choice, bool enable_ibp = false) noexcept {
  if (recon == nullptr || source == nullptr || pred == nullptr || w <= 0 ||
      h <= 0 || w > intra::kMaxBlockDim || h > intra::kMaxBlockDim) {
    return false;
  }
  intra::NeighborRequest request;
  request.recon = intra::PlaneView{recon, recon_stride};
  request.x = x0;
  request.y = y0;
  request.w = w;
  request.h = h;
  request.max_x = avail.max_x;
  request.max_y = avail.max_y;
  request.have_left = avail.have_left != 0;
  request.have_above = avail.have_above != 0;
  request.num4_above_right = avail.num4_above_right;
  request.num4_below_left = avail.num4_below_left;
  request.color_plane = 0;

  intra::IntraEdges edges;
  if (intra::prepare_edges(request, 8, edges) != intra::IntraStatus::kOk) {
    return false;
  }
  std::array<intra::ModeCandidate, intra::kMaxModeCandidates> candidates;
  int count = intra::build_mode_candidates(candidates, true);
  if (count <= 0) return false;

  auto predict_with = [&](intra::Mode mode, int angle_delta,
                          std::uint8_t* dst) noexcept -> bool {
    if (enable_ibp) {
      ibp::PredictParams params;
      params.mode = mode;
      params.angle_delta = angle_delta;
      params.enable_ibp = true;
      params.enable_edge_filter = enable_edge_filter;
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      return ibp::predict(edges, params, dst, static_cast<std::size_t>(w)) ==
             intra::IntraStatus::kOk;
    }
    intra::PredictParams params;
    params.mode = mode;
    params.angle_delta = angle_delta;
    params.enable_edge_filter = enable_edge_filter;
    params.above_neighbor_is_smooth = above_smooth;
    params.left_neighbor_is_smooth = left_smooth;
    return intra::predict_intra(edges, params, dst,
                                static_cast<std::size_t>(w)) ==
           intra::IntraStatus::kOk;
  };

  choice.sse = 0;
  bool have_best = false;
  for (int i = 0; i < count; ++i) {
    const intra::ModeCandidate& candidate =
        candidates[static_cast<std::size_t>(i)];
    if (!predict_with(candidate.mode, candidate.angle_delta, pred)) {
      return false;
    }
    const std::int64_t sse = detail::sse_against_source(
        source, source_stride, source_max_x, source_max_y, x0, y0, w, h, pred);
    if (!have_best || sse < choice.sse) {
      have_best = true;
      choice.sse = sse;
      choice.mode = static_cast<int>(candidate.mode);
      choice.angle_delta = candidate.angle_delta;
    }
  }
  if (!have_best) return false;
  // `pred` currently holds the last evaluated candidate; write the winner so
  // the caller transforms the prediction that was actually selected.
  if (!predict_with(static_cast<intra::Mode>(choice.mode), choice.angle_delta,
                    pred)) {
    return false;
  }
  return true;
}

}  // namespace av2enc::video::intra_search

#endif  // AV2ENC_PIPELINE_VIDEO_VIDEO_INTRA_SEARCH_H_
