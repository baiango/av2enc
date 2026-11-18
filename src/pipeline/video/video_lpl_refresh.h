// Two-pass refresh allocation for the opt-in lpl36 perceptual refresh
// pipeline (`AV2ENC_PIPE_LPL_REFRESH`).
//
// The drift-refresh trigger in `video_frame.cpp` refreshes a frame when the
// primary reference's luma PSNR is below an anchor.  This planner replaces
// both the trigger and the selection rule:
//
//   * pass 1 codes the sequence with no refresh and records, per frame, the
//     measured lpl36 error of the reconstruction against its source and the
//     frame payload;
//   * pass 2 codes every inter frame as a refresh (forced residual, same QP
//     drop as the drift-refresh plumbing) and records the same quantities, so
//     the planner has a measured *refresh cost* (payload difference) and a
//     measured *refresh floor* (the lpl36 error a refreshed frame actually
//     achieves) for every candidate;
//   * a dynamic program chooses the refresh set that minimises the predicted
//     sum of per-frame lpl36 error under the payload budget.  After a refresh
//     at i the predicted error at frame j starts at the measured all-refresh
//     floor b1[j] and closes the gap to the no-refresh curve b0[j] with 0.8
//     survival per frame; the output pass can be followed by one refinement
//     pass when the probe's payload estimate under-priced the isolated
//     refreshes.
//
// The decision therefore uses only the perceptual metric and priced payloads;
// no PSNR term is involved.  The output schedule is applied by the existing
// drift-refresh plumbing (frame QP drop + `force_inter_residual`), so no new
// syntax is emitted.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace av2enc::video::lpl_refresh {

struct Measurements {
  // Per-frame lpl36 error (lower is better), reconstruction against source.
  std::vector<double> error_no_refresh;
  std::vector<double> error_all_refresh;
  // Per-frame tile payload (bytes) in the same two passes.
  std::vector<std::uint32_t> payload_no_refresh;
  std::vector<std::uint32_t> payload_all_refresh;
};

struct Plan {
  std::vector<std::uint8_t> schedule;  // one entry per coded frame
  std::vector<int> refresh_frames;
  std::vector<int> costs;  // per-frame estimated extra bytes
  // Predicted mean lpl36 error of the planned chain and of the no-refresh
  // chain, from the same growth model.  Reported in the trace; not a decision
  // input.
  double predicted_mean_error = 0.0;
  double predicted_no_refresh_mean_error = 0.0;
  std::int64_t estimated_bytes = 0;
};

// Per-frame survival of inherited drift.  A refresh drops the frame to the
// measured refresh floor; the no-refresh error then re-accumulates.  With
// `kDriftCarry = 0.8` the predicted error closes ~83% of the gap to the
// no-refresh curve after eight frames, which matches the measured decay of
// the PSNR-triggered refresh in docs/video-drift-refresh.md (the benefit is
// mostly gone after ~8 frames).  This is a model of error propagation, not a
// threshold tuned on the measured clip: the planner still chooses every frame
// from the measured lpl36 error and measured payload.
inline constexpr double kDriftCarry = 0.8;

inline double predicted_error(const std::vector<double>& no_refresh,
                              const std::vector<double>& all_refresh, std::size_t refreshed_at,
                              std::size_t frame) {
  const std::size_t age = frame - refreshed_at;
  const double gap = std::max(0.0, no_refresh[frame] - all_refresh[frame]);
  double carry = 1.0;
  for (std::size_t k = 0; k < age; ++k) carry *= kDriftCarry;
  return all_refresh[frame] + gap * (1.0 - carry);
}

// `budget_bytes` is the payload allowance for the refresh extras.  The DP
// uses the measured per-candidate extra `payload_all_refresh[i] -
// payload_no_refresh[i]`, clamped at zero; a refresh that saves payload is
// free to the allocator.
inline Plan plan(const Measurements& m, int budget_bytes, double cost_scale = 1.0) {
  Plan out;
  const std::size_t n = m.error_no_refresh.size();
  if (n == 0) return out;
  out.schedule.assign(n, 0);
  out.costs.assign(n, 0);
  if (m.error_all_refresh.size() != n || m.payload_no_refresh.size() != n ||
      m.payload_all_refresh.size() != n) {
    return out;
  }

  std::vector<double> b0 = m.error_no_refresh;
  std::vector<double> b1 = m.error_all_refresh;
  b1[0] = b0[0];  // the key picture is the start state, not a refresh
  const double scale = std::max(1.0, cost_scale);
  std::vector<int> cost(n, 0);
  for (std::size_t i = 1; i < n; ++i) {
    std::int64_t extra = static_cast<std::int64_t>(m.payload_all_refresh[i]) -
                         static_cast<std::int64_t>(m.payload_no_refresh[i]);
    extra = std::max<std::int64_t>(0, extra);
    extra = static_cast<std::int64_t>(std::llround(static_cast<double>(extra) * scale));
    extra = std::min<std::int64_t>(extra, budget_bytes);
    cost[i] = static_cast<int>(extra);
    out.costs[i] = cost[i];
  }

  // pred[i][j] = predicted error at frame j if the last refresh was frame i.
  // The refresh floor is the measured all-refresh curve; the no-refresh gap
  // re-accumulates with kDriftCarry.  Segment sums are precomputed so the DP
  // inner loop is cheap.
  std::vector<std::vector<double>> seg(n, std::vector<double>(n + 1, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      seg[i][j + 1] = seg[i][j] + predicted_error(b0, b1, i, j);
    }
  }

  const int budget = std::max(0, budget_bytes);
  const double kInf = std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> g(
      n, std::vector<double>(static_cast<std::size_t>(budget) + 1, kInf));
  std::vector<std::vector<int>> choice(n,
                                       std::vector<int>(static_cast<std::size_t>(budget) + 1, -1));

  for (std::size_t ii = n; ii-- > 0;) {
    for (int b = 0; b <= budget; ++b) {
      double best = seg[ii][n];
      int best_k = -1;
      for (std::size_t k = ii + 1; k < n; ++k) {
        const int c = cost[k];
        if (c > b) continue;
        const double v = seg[ii][k] + g[k][static_cast<std::size_t>(b - c)];
        if (v < best - 1e-9) {
          best = v;
          best_k = static_cast<int>(k);
        }
      }
      g[ii][static_cast<std::size_t>(b)] = best;
      choice[ii][static_cast<std::size_t>(b)] = best_k;
    }
  }

  std::size_t i = 0;
  int b = budget;
  std::int64_t estimated = 0;
  while (i < n) {
    const int k = choice[i][static_cast<std::size_t>(b)];
    if (k <= 0) break;
    const std::size_t ku = static_cast<std::size_t>(k);
    out.schedule[ku] = 1;
    out.refresh_frames.push_back(k);
    estimated += cost[ku];
    b -= cost[ku];
    i = ku;
  }
  out.estimated_bytes = estimated;
  out.predicted_mean_error = g[0][static_cast<std::size_t>(budget)] / static_cast<double>(n);
  out.predicted_no_refresh_mean_error = seg[0][n] / static_cast<double>(n);
  return out;
}

}  // namespace av2enc::video::lpl_refresh
