// AV2ENC_PIPE_BUDGET: budget-closed per-superblock allocation (see the header).
//
// This file only holds the pieces that are independent of the encoder class:
// the greedy allocator and the table bookkeeping.  The frame/leaf loops live
// in video_frame_decision.cpp (`pipe_budget_decide_frame`) because the leaf
// body is the encoder's own coder.
#include "pipeline/video/video_pipe_budget.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace av2enc::video::pipe_budget {

namespace {

// One superblock's measured rate/distortion curve over the qindex grid.
struct Curve {
  std::vector<CurvePoint> points;  // ascending qindex
};

}  // namespace

// Picks one qindex per superblock so that the summed bytes land under
// `budget_bytes` while the summed distortion is as small as possible.
//
// Greedy marginal gain: start every superblock at the coarsest point that
// fits (the last one), then repeatedly move the superblock whose next finer
// step buys the most distortion per byte (gain / bytes^gamma) until the
// budget is spent or no superblock can move finer.  The table accessors keep
// the loop N-independent of the leaf coder.
std::vector<int> allocate(const std::vector<std::vector<CurvePoint>>& curves,
                          long long budget_bytes, double gamma) {
  const std::size_t n = curves.size();
  std::vector<int> pick(n, 0);
  std::vector<long long> bytes(n, 0);
  long long total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (curves[i].empty()) continue;
    pick[i] = static_cast<int>(curves[i].size()) - 1;
    bytes[i] = curves[i][static_cast<std::size_t>(pick[i])].bytes;
    total += bytes[i];
  }
  if (budget_bytes <= 0) return pick;
  if (const char* dbg = std::getenv("AV2ENC_PIPE_BUDGET_STATS")) {
    (void)dbg;
    std::fprintf(stderr, "pipe-budget alloc: n=%zu coarsest_total=%lld "
                         "budget=%lld room=%lld\n", n, total, budget_bytes,
                 budget_bytes - total);
  }
  for (;;) {
    int best = -1;
    double best_gain = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (curves[i].empty() || pick[i] == 0) continue;
      const CurvePoint& here = curves[i][static_cast<std::size_t>(pick[i])];
      const CurvePoint& finer = curves[i][static_cast<std::size_t>(pick[i] - 1)];
      const long long dbytes = finer.bytes - here.bytes;
      if (dbytes <= 0) continue;  // cannot afford a non-monotone point
      if (total + dbytes > budget_bytes) continue;
      const double grain = here.distortion - finer.distortion;
      if (!(grain > 0.0)) continue;
      const double score = grain / std::pow(static_cast<double>(dbytes), gamma);
      if (score > best_gain) {
        best_gain = score;
        best = static_cast<int>(i);
      }
    }
    if (best < 0) break;
    const std::size_t b = static_cast<std::size_t>(best);
    const long long dbytes =
        curves[b][static_cast<std::size_t>(pick[b] - 1)].bytes -
        curves[b][static_cast<std::size_t>(pick[b])].bytes;
    total += dbytes;
    bytes[b] += dbytes;
    --pick[b];
  }
  return pick;
}

// Marginal-cost allocation: `ref_bytes[i]` is superblock i's absolute probe
// size at the reference qindex.  Every point is re-expressed as a delta from
// that reference, so the per-superblock header the probe pays is cancelled and
// the budget closes on the FRAME's bytes.
std::vector<int> allocate_marginal(
    std::vector<std::vector<CurvePoint>> curves,
    const std::vector<long long>& ref_bytes, long long budget_bytes,
    double gamma) {
  const std::size_t n = curves.size();
  // Ascending bytes.  Each point's distortion is the J the driver charged at
  // that qindex (lower is better), so the curves are already comparable.
  for (std::size_t i = 0; i < n; ++i) {
    std::sort(curves[i].begin(), curves[i].end(),
              [](const CurvePoint& a, const CurvePoint& b) {
                return a.bytes < b.bytes;
              });
  }
  std::vector<int> pick(n, 0);
  long long total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (curves[i].empty()) continue;
    // Start every superblock at the point closest to the reference size.
    int best = 0;
    long long best_gap = -1;
    for (std::size_t k = 0; k < curves[i].size(); ++k) {
      const long long gap = std::llabs(curves[i][k].bytes - ref_bytes[i]);
      if (best_gap < 0 || gap < best_gap) {
        best_gap = gap;
        best = static_cast<int>(k);
      }
    }
    pick[i] = best;
    total += curves[i][static_cast<std::size_t>(best)].bytes;
  }
  if (budget_bytes <= 0) return pick;
  // Two-sided greedy.  Each step moves one superblock by one grid point:
  //   refine (k-1) buys distortion and costs bytes,
  //   coarsen (k+1) frees bytes and costs distortion.
  // Strategy per iteration: if any refinement fits the remaining room at a
  // better gain/byte than the best available coarsening's price, take it;
  // otherwise coarsen the cheapest superblock to make room; stop when neither
  // is possible.
  for (;;) {
    int refine_sb = -1;
    double refine_score = 0.0;
    int coarsen_sb = -1;
    double coarsen_price = 0.0;
    const long long room = budget_bytes - total;
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t sz = curves[i].size();
      if (sz < 2) continue;
      const std::size_t k = static_cast<std::size_t>(pick[i]);
      const CurvePoint& here = curves[i][k];
      if (k > 0) {
        const CurvePoint& fine = curves[i][k - 1];
        const long long dbytes = fine.bytes - here.bytes;  // > 0
        const double grain = here.distortion - fine.distortion;  // > 0
        if (dbytes > 0 && grain > 0.0 && dbytes <= room) {
          const double score =
              grain / std::pow(static_cast<double>(dbytes), gamma);
          if (refine_sb < 0 || score > refine_score) {
            refine_sb = static_cast<int>(i);
            refine_score = score;
          }
        }
      }
      if (k + 1 < sz) {
        const CurvePoint& coarse = curves[i][k + 1];
        const long long freed = here.bytes - coarse.bytes;  // > 0
        const double cost = coarse.distortion - here.distortion;  // > 0
        if (freed > 0 && cost > 0.0) {
          const double price =
              cost / std::pow(static_cast<double>(freed), gamma);
          if (coarsen_sb < 0 || price < coarsen_price) {
            coarsen_sb = static_cast<int>(i);
            coarsen_price = price;
          }
        }
      }
    }
    const bool have_refine = refine_sb >= 0;
    const bool have_coarsen = coarsen_sb >= 0;
    if (!have_refine && !have_coarsen) break;
    // Refine when it fits and there is no need to free bytes; otherwise free
    // bytes first when the coarsening is cheaper per byte than the refinement
    // is worth.
    const bool take_refine =
        have_refine && (!have_coarsen || refine_score >= coarsen_price);
    if (take_refine) {
      const std::size_t i = static_cast<std::size_t>(refine_sb);
      const std::size_t k = static_cast<std::size_t>(pick[i]);
      total += curves[i][k - 1].bytes - curves[i][k].bytes;
      pick[i] = static_cast<int>(k - 1);
    } else if (have_coarsen) {
      if (room >= 0 && refine_sb < 0) {
        // Nothing fits; free bytes.
      }
      const std::size_t i = static_cast<std::size_t>(coarsen_sb);
      const std::size_t k = static_cast<std::size_t>(pick[i]);
      total += curves[i][k + 1].bytes - curves[i][k].bytes;
      pick[i] = static_cast<int>(k + 1);
    } else {
      break;
    }
  }
  return pick;
}

void print_summary(const std::vector<std::vector<CurvePoint>>& curves,
                   const std::vector<int>& pick, long long budget) {
  long long bytes = 0;
  double d = 0.0;
  int used = 0;
  std::printf("pipe-budget: superblocks=%zu budget=%lld\n", curves.size(),
              budget);
  for (std::size_t i = 0; i < curves.size() && i < pick.size(); ++i) {
    if (curves[i].empty()) continue;
    const CurvePoint& c = curves[i][static_cast<std::size_t>(pick[i])];
    bytes += c.bytes;
    d += c.distortion;
    ++used;
  }
  std::printf("pipe-budget: chosen=%d bytes=%lld distortion=%.6e\n", used,
              bytes, d);
  std::fflush(stdout);
}

}  // namespace av2enc::video::pipe_budget
