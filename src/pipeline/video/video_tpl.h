// TPL (two-pass look-ahead) complexity model — motion-compensated.
//
// The rate controller allocates per-frame QP from `rc::FrameComplexity`, which
// `plan_video_rate_control` currently fills with
// `complexity_from_u8_planes(frame, reference)` — a **motion-free** residual.
// On moving content that number is dominated by motion, not by coding cost: a
// pan of one pixel moves most of the picture, so a *cheap* frame measures as
// expensive and the allocator spends its budget in the wrong place.  That is
// why the per-clip key offset had to be tuned by hand and did not transfer
// (docs: +0.22 JOD on the orbital plate, -0.10 on loral).
//
// This gate replaces the proxy with a real look-ahead first pass:
//
//   1. per 16x16 block, integer motion search against the frame's actual
//      reference (reusing the ARNR search), then
//   2. the motion-compensated residual MAD/MSE of that frame, and
//   3. a temporal propagation weight: a picture every later picture references
//      carries its own error forward, so its cost is scaled by one plus the
//      number of pictures that (transitively) depend on it.
//
// Only the *estimate* changes; the allocation policy, the bitrate model and the
// bitstream are untouched, so TPL cannot alter syntax or pinned streams.
//
// Gate (env only, default off):
//   AV2ENC_TPL=1                 enable the MC complexity pass
//   AV2ENC_TPL_SEARCH=<0..16>    integer ME radius (default 8)
//   AV2ENC_TPL_PROPAGATE=<0..1>  apply the temporal dependency weight (default 1)
//   AV2ENC_TPL_STATS=1           one stderr line per frame
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include "codec/rate/rc/rc.h"
#include "pipeline/video/video_arnr.h"
#include "pipeline/video/video_frame.h"

namespace av2enc::video::tpl {

inline bool parse_off(const char* v) {
  if (v == nullptr || *v == '\0') return true;
  if (std::strcmp(v, "0") == 0) return true;
  auto eq = [v](const char* w) {
    for (std::size_t i = 0; w[i] != '\0'; ++i) {
      if (std::tolower(static_cast<unsigned char>(v[i])) != w[i]) return false;
    }
    return v[std::strlen(w)] == '\0';
  };
  return eq("false") || eq("off") || eq("no");
}

inline bool enabled() { return !parse_off(std::getenv("AV2ENC_TPL")); }

inline int env_int(const char* name, int fallback, int lo, int hi) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') return fallback;
  return static_cast<int>(std::clamp<long>(parsed, lo, hi));
}

inline int search() { return env_int("AV2ENC_TPL_SEARCH", 8, 0, 16); }
inline int propagate() { return env_int("AV2ENC_TPL_PROPAGATE", 1, 0, 1); }
inline int gain() { return env_int("AV2ENC_TPL_GAIN", 2, 0, 12); }
inline bool stats_on() { return !parse_off(std::getenv("AV2ENC_TPL_STATS")); }

namespace detail {

// Motion-compensated residual statistics of `cur` predicted from `ref`.
// Blocks that no legal vector can predict stay in the sum at their collocated
// error, so the measure never understates cost.
inline void mc_residual(const std::uint8_t* cur, const std::uint8_t* ref,
                        int width, int height, int range, double* mad,
                        double* mse) {
  std::int64_t abs_sum = 0;
  std::int64_t sq_sum = 0;
  std::int64_t count = 0;
  const int bw = 16;
  const int bh = 16;
  for (int by = 0; by + bh <= height; by += bh) {
    for (int bx = 0; bx + bw <= width; bx += bw) {
      const arnr::detail::Mv mv = arnr::detail::estimate_mv(
          cur, ref, width, width, height, bx, by, bw, bh, range);
      const std::uint8_t* c = cur + static_cast<std::ptrdiff_t>(by) * width + bx;
      const std::uint8_t* r =
          ref + static_cast<std::ptrdiff_t>(by + mv.dy) * width + (bx + mv.dx);
      for (int row = 0; row < bh; ++row) {
        const std::uint8_t* cr = c + static_cast<std::ptrdiff_t>(row) * width;
        const std::uint8_t* rr = r + static_cast<std::ptrdiff_t>(row) * width;
        for (int col = 0; col < bw; ++col) {
          const int d = static_cast<int>(cr[col]) - static_cast<int>(rr[col]);
          abs_sum += d < 0 ? -d : d;
          sq_sum += static_cast<std::int64_t>(d) * d;
          ++count;
        }
      }
    }
  }
  if (count == 0) {
    *mad = 0.0;
    *mse = 0.0;
    return;
  }
  *mad = static_cast<double>(abs_sum) / static_cast<double>(count);
  *mse = static_cast<double>(sq_sum) / static_cast<double>(count);
}

}  // namespace detail

// Builds the look-ahead complexity table.  `primary_ref[i]` is the display
// index of picture i's reference, or -1 for an intra picture.  The result is
// indexed by display order, matching `plan_video_rate_control`.
inline void compute(
    std::span<const VideoInputFrame> inputs,
    std::span<const int> primary_ref,
    std::vector<av2enc::rc::FrameComplexity>* complexity) {
  const std::size_t n = inputs.size();
  complexity->assign(n, av2enc::rc::FrameComplexity{});
  if (n == 0) return;
  const int width = static_cast<int>(inputs[0].width);
  const int height = static_cast<int>(inputs[0].height);
  const int range = search();

  // 1. Per-picture motion-compensated residual.
  std::vector<double> mad(n, 0.0);
  std::vector<double> mse(n, 0.0);
  std::vector<int> intra(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    int ref = primary_ref.empty()
                  ? (i == 0 ? -1 : static_cast<int>(i) - 1)
                  : primary_ref[i];
    if (ref < 0 || ref >= static_cast<int>(n)) {
      intra[i] = 1;
      // Intra cost: distance from the frame's own flat prediction is a poor
      // proxy, so use the local-activity of the picture (high-pass energy),
      // which is what an intra frame actually has to spend bits on.
      std::int64_t abs_sum = 0;
      std::int64_t count = 0;
      for (int y = 0; y + 1 < height; y += 2) {
        const std::uint8_t* row =
            inputs[i].y.data() + static_cast<std::ptrdiff_t>(y) * width;
        const std::uint8_t* next = row + width;
        for (int x = 0; x + 1 < width; x += 2) {
          const int d = static_cast<int>(row[x]) - static_cast<int>(next[x + 1]);
          abs_sum += d < 0 ? -d : d;
          ++count;
        }
      }
      mad[i] = count ? static_cast<double>(abs_sum) / static_cast<double>(count)
                     : 0.0;
      mse[i] = mad[i] * mad[i];
      continue;
    }
    detail::mc_residual(inputs[i].y.data(),
                        inputs[static_cast<std::size_t>(ref)].y.data(), width,
                        height, range, &mad[i], &mse[i]);
  }

  // 2. Temporal propagation: a picture that later pictures reference carries
  //    its error forward, so it is worth more than its own residual says.
  std::vector<double> dependents(n, 0.0);
  if (propagate() != 0) {
    for (std::size_t i = 0; i < n; ++i) {
      int ref = primary_ref.empty()
                    ? (i == 0 ? -1 : static_cast<int>(i) - 1)
                    : primary_ref[i];
      // Walk the reference chain upwards: each hop adds one dependent step.
      double weight = 1.0;
      int cur = ref;
      std::size_t guard = 0;
      while (cur >= 0 && cur < static_cast<int>(n) && guard < n) {
        dependents[static_cast<std::size_t>(cur)] += weight;
        ref = primary_ref.empty() ? (cur == 0 ? -1 : cur - 1)
                                  : primary_ref[static_cast<std::size_t>(cur)];
        cur = ref;
        ++guard;
        if (cur == static_cast<int>(i)) break;  // closed loop guard
      }
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    av2enc::rc::FrameComplexity& c = (*complexity)[i];
    c.width = width;
    c.height = height;
    // Propagated cost: a picture with d dependents carries its own error
    // forward d times, so it is priced as if it were 1+d times as expensive.
    const double weight = propagate() != 0 ? 1.0 + dependents[i] : 1.0;
    c.residual_mad = mad[i] * weight;
    c.residual_mse = mse[i] * weight;
    c.intra_fraction = intra[i] != 0 ? 1.0 : 0.0;
    if (stats_on()) {
      std::fprintf(stderr,
                   "tpl: i=%zu intra=%d mad=%.3f mse=%.2f dependents=%.1f\n", i,
                   intra[i], mad[i], mse[i], dependents[i]);
    }
  }
}

// Allocates a per-frame QP plan at the *same estimated rate* as a flat QP
// stream, which is what makes this a pure reallocation: the total target is
// whatever `base_qp` would have spent, so a win cannot come from spending more
// bits.  Frames whose motion-compensated residual is expensive get a finer QP;
// frames that predict almost for free get a coarser one.  `gain` scales how far
// the plan departs from flat, and a global shift re-centres the plan so the
// estimated total matches the flat-QP total.
inline void plan_qp(std::span<const av2enc::rc::FrameComplexity> complexity,
                    int base_qp, int gain, std::vector<std::uint8_t>* frame_qp) {
  const std::size_t n = complexity.size();
  frame_qp->assign(n, static_cast<std::uint8_t>(std::clamp(base_qp, 1, 255)));
  if (n == 0 || gain <= 0) return;

  // Flat-QP estimated total: the budget.
  double budget = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    budget += av2enc::rc::estimate_frame_bits_model(complexity[i], base_qp);
  }
  if (!(budget > 0.0)) return;

  // Relative cost of each frame against the mean, in log2 so the weighting is
  // scale-free: a frame 4x as expensive moves gain*2 QP steps finer.
  double mean = 0.0;
  for (std::size_t i = 0; i < n; ++i) mean += complexity[i].residual_mad;
  mean /= static_cast<double>(n);
  if (!(mean > 0.0)) return;

  std::vector<int> offset(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const double rel = complexity[i].residual_mad / mean;
    const double steps = rel > 0.0 ? std::log2(rel) : 0.0;
    // Negative offset = finer QP for expensive frames.
    offset[i] = static_cast<int>(std::lround(-gain * steps));
  }

  // Re-centre on the budget: raise QP everywhere until the estimate fits.
  auto total = [&](int shift) {
    double bits = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const int q = std::clamp(base_qp + offset[i] + shift, 1, 255);
      bits += av2enc::rc::estimate_frame_bits_model(complexity[i], q);
    }
    return bits;
  };
  int shift = 0;
  while (shift < 40 && total(shift) > budget) ++shift;
  while (shift > -40 && total(shift) < budget) --shift;
  if (shift != 0 && std::abs(shift) > 0) {
    // Step back if the integer shift overshot the budget.
    if (total(shift) > budget && shift > 0) --shift;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const int q = std::clamp(base_qp + offset[i] + shift, 1, 255);
    (*frame_qp)[i] = static_cast<std::uint8_t>(q);
  }
  if (stats_on()) {
    std::fprintf(stderr,
                 "tpl: plan base_qp=%d gain=%d shift=%d budget=%.0f "
                 "planned=%.0f\n",
                 base_qp, gain, shift, budget, total(shift));
    for (std::size_t i = 0; i < n && i < 12; ++i) {
      std::fprintf(stderr, "tpl:   i=%zu qp=%d (off %+d) mad=%.2f\n", i,
                   static_cast<int>((*frame_qp)[i]), offset[i],
                   complexity[i].residual_mad);
    }
  }
}

}  // namespace av2enc::video::tpl
