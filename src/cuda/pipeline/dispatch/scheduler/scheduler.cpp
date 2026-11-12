// Host-only wavefront plan construction for the M0 DC-intra block path.
//
// Nothing in this translation unit touches CUDA, so the dependency analysis,
// wavefront levels, batching plan and launch counters can be unit-tested on a
// host without a GPU (scheduler_plan_test.cpp).
//
// Dependency model: a plane block reads the reconstructed column at x0-1 over
// rows [y0, y0+h) and the reconstructed row at y0-1 over columns [x0, x0+w).
// Both reads are clamped to the visible plane bounds, so every sample that is
// read either belongs to an already-covered plane cell or lies outside the
// visible bounds (initialised to 128 and never written).  The plan records an
// edge for every block that owns one of those cells, which is a superset of the
// distinct reconstructed blocks the predictor actually reads after clamping.

#include "cuda/pipeline/dispatch/scheduler/scheduler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::cuda::sched {
namespace {

constexpr int kCellLuma = 4;    // samples per coverage cell, luma plane
constexpr int kCellChroma = 2;  // chroma block origins are 2-sample aligned

std::uint64_t fnv1a(std::uint64_t hash, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

bool validate(const FrameJob& job, std::string& error) {
  const FrameGeometry& g = job.geometry;
  if (g.width <= 0 || g.height <= 0) {
    error = "frame geometry has an empty picture";
    return false;
  }
  const int expect_cw = (g.width + 1) / 2;
  const int expect_ch = (g.height + 1) / 2;
  if (g.chroma_width != expect_cw || g.chroma_height != expect_ch) {
    error = "chroma geometry is not M0 4:2:0 for " + std::to_string(g.width) +
            "x" + std::to_string(g.height);
    return false;
  }
  if (g.padded_width < g.width || g.padded_height < g.height ||
      (g.padded_width % 64) != 0 || (g.padded_height % 64) != 0) {
    error = "padded geometry must be 64-aligned and cover the picture";
    return false;
  }
  if (g.chroma_padded_width != (g.padded_width + 1) / 2 ||
      g.chroma_padded_height != (g.padded_height + 1) / 2) {
    error = "chroma padded geometry does not match the luma padded geometry";
    return false;
  }
  if (g.mi_cols != 2 * ((g.width + 7) >> 3) ||
      g.mi_rows != 2 * ((g.height + 7) >> 3)) {
    error = "mi_cols/mi_rows do not match the M0 derivation";
    return false;
  }
  if (job.block_count < 0 || (job.block_count > 0 && job.blocks == nullptr)) {
    error = "invalid block list";
    return false;
  }
  return true;
}

}  // namespace

const char* plane_name(int plane) {
  switch (plane) {
    case 0:
      return "Y";
    case 1:
      return "U";
    case 2:
      return "V";
    default:
      return "?";
  }
}

std::uint64_t coding_blocks_signature(const CodingBlock* blocks, int count) {
  std::uint64_t hash = 1469598103934665603ull;
  hash = fnv1a(hash, &count, sizeof(count));
  if (blocks != nullptr && count > 0) {
    hash = fnv1a(hash, blocks,
                 static_cast<std::size_t>(count) * sizeof(CodingBlock));
  }
  return hash;
}

void Plan::clear() {
  geometry_ = {};
  jobs_.clear();
  groups_.clear();
  classes_.clear();
  level_begin_.clear();
  levels_ = 0;
  counters_ = {};
  blocks_signature_ = 0;
}

bool Plan::build(const FrameJob& job, std::string& error) {
  clear();
  error.clear();
  if (!validate(job, error)) {
    return false;
  }

  const FrameGeometry& g = job.geometry;
  geometry_ = g;
  blocks_signature_ = coding_blocks_signature(job.blocks, job.block_count);
  counters_.coding_blocks = job.block_count;

  // -------------------------------------------------------------------------
  // 1. Expand coding blocks into plane blocks.
  // -------------------------------------------------------------------------
  for (int i = 0; i < job.block_count; ++i) {
    const CodingBlock& b = job.blocks[i];
    if (b.mi_row < 0 || b.mi_col < 0) {
      error = "negative block position";
      return false;
    }
    if (b.luma_w <= 0 || b.luma_h <= 0 || b.chroma_w <= 0 || b.chroma_h <= 0) {
      error = "block " + std::to_string(i) + " has a non-positive size";
      return false;
    }
    if (b.upscale != 1 && b.upscale != 2) {
      error = "block " + std::to_string(i) + " has an invalid upscale factor";
      return false;
    }
    if (b.upscale == 2 && (b.luma_w != 64 || b.luma_h != 64)) {
      error = "block " + std::to_string(i) +
              " requests the 64x64 adjusted transform on a non-64x64 block";
      return false;
    }
    const int expect_luma =
        (b.luma_w / b.upscale) * (b.luma_h / b.upscale);
    if (b.luma_pels != expect_luma) {
      error = "block " + std::to_string(i) + " luma_pels mismatch";
      return false;
    }
    if (b.chroma_pels != b.chroma_w * b.chroma_h) {
      error = "block " + std::to_string(i) + " chroma_pels mismatch";
      return false;
    }
    if (b.real_coeffs == 0) {
      continue;  // M0 diagnostic block: no GPU work, recon stays at 128
    }
    ++counters_.real_blocks;

    if (b.luma_w == 64 && b.luma_h == 64) {
      ++counters_.luma_64_jobs;
    }

    const int chroma_x = b.mi_col * 2;
    const int chroma_y = b.mi_row * 2;
    const int luma_x = b.mi_col * 4;
    const int luma_y = b.mi_row * 4;

    PlaneJob y;
    y.plane = 0;
    y.block = i;
    y.x = luma_x;
    y.y = luma_y;
    y.w = b.luma_w;
    y.h = b.luma_h;
    y.tx_w = b.luma_w / b.upscale;
    y.tx_h = b.luma_h / b.upscale;
    y.upscale = b.upscale;
    y.coeff_count = b.luma_pels;
    y.max_x = g.width - 1;
    y.max_y = g.height - 1;
    y.pred_index = 3 * i;
    y.coeff_offset = b.coeff_offset;
    y.scratch_slot = (b.upscale == 2) ? counters_.luma_64_jobs - 1 : -1;

    PlaneJob u = y;
    u.plane = 1;
    u.x = chroma_x;
    u.y = chroma_y;
    u.w = b.chroma_w;
    u.h = b.chroma_h;
    u.tx_w = b.chroma_w;
    u.tx_h = b.chroma_h;
    u.upscale = 1;
    u.coeff_count = b.chroma_pels;
    u.max_x = g.chroma_width - 1;
    u.max_y = g.chroma_height - 1;
    u.pred_index = 3 * i + 1;
    u.coeff_offset = b.coeff_offset + b.luma_pels;
    u.scratch_slot = -1;

    PlaneJob v = u;
    v.plane = 2;
    v.pred_index = 3 * i + 2;
    v.coeff_offset = u.coeff_offset + b.chroma_pels;

    const std::int64_t end =
        static_cast<std::int64_t>(v.coeff_offset) + b.chroma_pels;
    if (job.total_coeffs > 0 && end > job.total_coeffs) {
      error = "block " + std::to_string(i) +
              " coefficient range exceeds total_coeffs";
      return false;
    }

    jobs_.push_back(y);
    jobs_.push_back(u);
    jobs_.push_back(v);
  }
  counters_.plane_jobs = static_cast<int>(jobs_.size());
  if (jobs_.empty()) {
    counters_.levels = 0;
    return true;
  }

  // Canonical coefficient ranges must not overlap, or the scatter into the
  // caller's layout would silently merge two blocks.
  {
    std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
    ranges.reserve(static_cast<std::size_t>(job.block_count));
    for (int i = 0; i < job.block_count; ++i) {
      const CodingBlock& b = job.blocks[i];
      if (b.real_coeffs == 0) {
        continue;
      }
      const std::int64_t begin = b.coeff_offset;
      const std::int64_t end = begin + b.luma_pels + 2 * b.chroma_pels;
      ranges.emplace_back(begin, end);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i) {
      if (ranges[i].first < ranges[i - 1].second) {
        error = "canonical coefficient ranges overlap";
        clear();
        return false;
      }
    }
  }

  const int n = static_cast<int>(jobs_.size());

  // -------------------------------------------------------------------------
  // 2. Coverage map + prerequisite edges per plane.
  // -------------------------------------------------------------------------
  const std::vector<std::vector<int>> deps = [&]() {
    std::vector<std::vector<int>> edges(static_cast<std::size_t>(n));
    for (int plane = 0; plane < 3; ++plane) {
      const int cell = (plane == 0) ? kCellLuma : kCellChroma;
      const int pw =
          (plane == 0) ? g.padded_width : g.chroma_padded_width;
      const int ph =
          (plane == 0) ? g.padded_height : g.chroma_padded_height;
      const int cells_x = pw / cell;
      const int cells_y = ph / cell;
      std::vector<int> cover(
          static_cast<std::size_t>(cells_x) * static_cast<std::size_t>(cells_y),
          -1);

      for (int idx = 0; idx < n; ++idx) {
        const PlaneJob& j = jobs_[static_cast<std::size_t>(idx)];
        if (j.plane != plane) {
          continue;
        }
        if (j.x < 0 || j.y < 0 || j.x + j.w > pw || j.y + j.h > ph ||
            (j.x % cell) != 0 || (j.y % cell) != 0 || (j.w % cell) != 0 ||
            (j.h % cell) != 0) {
          error = std::string("plane ") + plane_name(plane) + " block at (" +
                  std::to_string(j.x) + "," + std::to_string(j.y) +
                  ") is outside the padded plane";
          return edges;  // caller checks error
        }
        for (int cy = j.y / cell; cy < (j.y + j.h) / cell; ++cy) {
          for (int cx = j.x / cell; cx < (j.x + j.w) / cell; ++cx) {
            int& cell_id = cover[static_cast<std::size_t>(cy) *
                                     static_cast<std::size_t>(cells_x) +
                                 static_cast<std::size_t>(cx)];
            if (cell_id >= 0) {
              error = std::string("plane ") + plane_name(plane) +
                      " blocks overlap at cell (" + std::to_string(cx) + "," +
                      std::to_string(cy) + ")";
              return edges;
            }
            cell_id = idx;
          }
        }
      }

      const auto add_edge = [&](std::vector<int>& out, int id) {
        if (id < 0) {
          return;
        }
        if (std::find(out.begin(), out.end(), id) == out.end()) {
          out.push_back(id);
        }
      };

      for (int idx = 0; idx < n; ++idx) {
        const PlaneJob& j = jobs_[static_cast<std::size_t>(idx)];
        if (j.plane != plane) {
          continue;
        }
        std::vector<int>& out = edges[static_cast<std::size_t>(idx)];
        if (j.x > 0) {
          const int cx = j.x / cell - 1;
          for (int cy = j.y / cell; cy < (j.y + j.h) / cell; ++cy) {
            add_edge(out, cover[static_cast<std::size_t>(cy) *
                                    static_cast<std::size_t>(cells_x) +
                                static_cast<std::size_t>(cx)]);
          }
        }
        if (j.y > 0) {
          const int cy = j.y / cell - 1;
          for (int cx = j.x / cell; cx < (j.x + j.w) / cell; ++cx) {
            add_edge(out, cover[static_cast<std::size_t>(cy) *
                                    static_cast<std::size_t>(cells_x) +
                                static_cast<std::size_t>(cx)]);
          }
        }
      }
    }
    return edges;
  }();

  if (!error.empty()) {
    clear();
    return false;
  }

  // -------------------------------------------------------------------------
  // 3. Wavefront levels (longest path over the prerequisite DAG).
  // -------------------------------------------------------------------------
  std::vector<int> indegree(static_cast<std::size_t>(n), 0);
  std::vector<std::vector<int>> consumers(static_cast<std::size_t>(n));
  long long edge_count = 0;
  for (int idx = 0; idx < n; ++idx) {
    const std::vector<int>& in = deps[static_cast<std::size_t>(idx)];
    edge_count += static_cast<long long>(in.size());
    for (int d : in) {
      consumers[static_cast<std::size_t>(d)].push_back(idx);
      ++indegree[static_cast<std::size_t>(idx)];
    }
  }
  counters_.dependency_edges = edge_count;

  std::vector<int> level(static_cast<std::size_t>(n), 0);
  std::vector<int> ready;
  ready.reserve(static_cast<std::size_t>(n));
  for (int idx = 0; idx < n; ++idx) {
    if (indegree[static_cast<std::size_t>(idx)] == 0) {
      ready.push_back(idx);
    }
  }
  int processed = 0;
  while (!ready.empty()) {
    const int u = ready.back();
    ready.pop_back();
    ++processed;
    for (int v : consumers[static_cast<std::size_t>(u)]) {
      int& lv = level[static_cast<std::size_t>(v)];
      lv = std::max(lv, level[static_cast<std::size_t>(u)] + 1);
      if (--indegree[static_cast<std::size_t>(v)] == 0) {
        ready.push_back(v);
      }
    }
  }
  if (processed != n) {
    counters_.cyclic = true;
    error = "DC-prediction prerequisite graph contains a cycle";
    clear();
    return false;
  }

  int max_level = 0;
  for (int idx = 0; idx < n; ++idx) {
    jobs_[static_cast<std::size_t>(idx)].level = level[static_cast<std::size_t>(idx)];
    max_level = std::max(max_level, level[static_cast<std::size_t>(idx)]);
  }
  levels_ = max_level + 1;

  // Defensive check: every edge must cross a wavefront boundary.  A violation
  // would mean two dependent blocks share a level; the level construction above
  // makes that impossible, but the counter is part of the contract.
  for (int idx = 0; idx < n; ++idx) {
    for (int d : deps[static_cast<std::size_t>(idx)]) {
      if (level[static_cast<std::size_t>(d)] >= level[static_cast<std::size_t>(idx)]) {
        ++counters_.dependency_violations;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 4. Transform classes: blocks that can share one batched launch.
  //    make_tx_params(coded_w, coded_h, luma) depends only on the adjusted
  //    transform size and the 64x64-luma upscale flag, so (tx_w, tx_h,
  //    upscale) identifies the TxParams exactly for the M0 geometry.
  // -------------------------------------------------------------------------
  const auto class_for = [&](int tx_w, int tx_h, int upscale) {
    for (int c = 0; c < static_cast<int>(classes_.size()); ++c) {
      const TransformClass& existing = classes_[static_cast<std::size_t>(c)];
      if (existing.tx_w == tx_w && existing.tx_h == tx_h &&
          existing.upscale == upscale) {
        return c;
      }
    }
    TransformClass created;
    created.tx_w = tx_w;
    created.tx_h = tx_h;
    created.upscale = upscale;
    created.pels = tx_w * tx_h;
    classes_.push_back(created);
    return static_cast<int>(classes_.size()) - 1;
  };
  for (PlaneJob& j : jobs_) {
    j.cls = class_for(j.tx_w, j.tx_h, j.upscale);
  }
  for (const PlaneJob& j : jobs_) {
    ++classes_[static_cast<std::size_t>(j.cls)].job_count;
  }

  // -------------------------------------------------------------------------
  // 5. Order: wavefront major, then transform class, then plane, then raster.
  //    Within a class the slot assignment follows the same order, so every
  //    (level, class) group is a contiguous run in the job array *and* in the
  //    class buffers.
  // -------------------------------------------------------------------------
  std::sort(jobs_.begin(), jobs_.end(),
            [](const PlaneJob& a, const PlaneJob& b) {
              if (a.level != b.level) {
                return a.level < b.level;
              }
              if (a.cls != b.cls) {
                return a.cls < b.cls;
              }
              if (a.plane != b.plane) {
                return a.plane < b.plane;
              }
              if (a.y != b.y) {
                return a.y < b.y;
              }
              return a.x < b.x;
            });

  std::vector<int> next_slot(classes_.size(), 0);
  for (PlaneJob& j : jobs_) {
    j.slot = next_slot[static_cast<std::size_t>(j.cls)]++;
  }

  // -------------------------------------------------------------------------
  // 6. Groups and per-level job ranges.
  // -------------------------------------------------------------------------
  for (int idx = 0; idx < n; ++idx) {
    const PlaneJob& j = jobs_[static_cast<std::size_t>(idx)];
    if (groups_.empty() || groups_.back().level != j.level ||
        groups_.back().cls != j.cls) {
      WaveGroup group;
      group.level = j.level;
      group.cls = j.cls;
      group.job_begin = idx;
      group.job_end = idx + 1;
      group.slot_begin = j.slot;
      group.slot_end = j.slot + 1;
      groups_.push_back(group);
    } else {
      WaveGroup& group = groups_.back();
      if (j.slot != group.slot_end) {
        error = "internal: class slots are not contiguous inside a wave group";
        clear();
        return false;
      }
      group.job_end = idx + 1;
      group.slot_end = j.slot + 1;
    }
  }

  level_begin_.assign(static_cast<std::size_t>(levels_) + 1u,
                      static_cast<int>(jobs_.size()));
  {
    int cursor = 0;
    for (int lv = 0; lv < levels_; ++lv) {
      level_begin_[static_cast<std::size_t>(lv)] = cursor;
      while (cursor < n &&
             jobs_[static_cast<std::size_t>(cursor)].level == lv) {
        ++cursor;
      }
    }
    level_begin_[static_cast<std::size_t>(levels_)] = cursor;
  }

  // -------------------------------------------------------------------------
  // 7. Launch accounting.
  // -------------------------------------------------------------------------
  counters_.levels = levels_;
  counters_.groups = static_cast<int>(groups_.size());
  counters_.transform_classes = static_cast<int>(classes_.size());

  long long scheduled = 1;  // canonical coefficient scatter
  for (int lv = 0; lv < levels_; ++lv) {
    if (level_begin_[static_cast<std::size_t>(lv)] <
        level_begin_[static_cast<std::size_t>(lv) + 1]) {
      ++scheduled;  // batched prediction + residual for the level
    }
  }
  for (const WaveGroup& group : groups_) {
    scheduled += 5;  // forward DCT, quantise, dequantise, inverse DCT, recon
    const TransformClass& cls = classes_[static_cast<std::size_t>(group.cls)];
    if (cls.upscale == 2) {
      scheduled += group.job_end - group.job_begin;  // 64x64 residual downsample
    }
  }
  counters_.scheduled_kernel_launches = scheduled;

  long long naive_calls = 0;
  long long naive_kernels = 0;
  for (int i = 0; i < job.block_count; ++i) {
    const CodingBlock& b = job.blocks[i];
    if (b.real_coeffs == 0) {
      continue;
    }
    const bool luma_64 = (b.luma_w == 64 && b.luma_h == 64);
    naive_calls += luma_64 ? 6 : 5;
    naive_kernels += luma_64 ? 8 : 7;
    naive_calls += 8;    // chroma pair
    naive_kernels += 10;  // chroma pair
  }
  counters_.naive_component_calls = naive_calls;
  counters_.naive_kernel_launches = naive_kernels;
  return true;
}

int Plan::level_begin(int level) const {
  if (level < 0) {
    return 0;
  }
  if (level >= levels_) {
    return static_cast<int>(jobs_.size());
  }
  return level_begin_[static_cast<std::size_t>(level)];
}

int Plan::level_end(int level) const {
  if (level < 0) {
    return 0;
  }
  if (level >= levels_) {
    return static_cast<int>(jobs_.size());
  }
  return level_begin_[static_cast<std::size_t>(level) + 1];
}

}  // namespace av2enc::cuda::sched
