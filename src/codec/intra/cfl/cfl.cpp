// CPU reference for AV2 v1.0.0 §7.13.5 chroma-from-luma.
//
// Spec-only: every branch below is a transcription of the pseudocode in
// §7.13.5, using the shared integer helpers in cfl_common.h. The two
// deliberate deviations are bookkeeping, not arithmetic:
//
//   * Reads of reconstructed luma/chroma are clamped to the caller-supplied
//     plane bounds. The spec's CurrFrame is a padded array; the caller passes
//     the padded reconstruction bounds, so the clamp is a no-op for valid
//     streams and a safety net (ASan-clean) for odd edge geometry.
//   * The sequential prevSample chains in the luma-average collection are
//     evaluated directly: for i >= limit, t is the value at limit - 1 (or the
//     midpoint when limit == 0), which is identical to carrying prevSample.
//     The random-block test cross-checks this against a literal sequential
//     transcription.

#include "cfl.h"

#include <algorithm>
#include <cstddef>

namespace av2enc::cfl {
namespace {

int clamp_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

std::size_t plane_index(int stride, int row, int col) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(col);
}

// cfl_ds_filter_index == 3 behaves exactly like 0 (§5.4.5 note).
int normalised_filter(const Block& block) {
  return block.ds_filter == 3 ? 0 : block.ds_filter;
}

// Number of chroma samples of a block row/column that are inside the coded
// chroma grid (MiCols/MiRows * MI_SIZE >> sub). Used for §7.13.5's
// prevSample replication.
int valid_columns(const Block& block) {
  return clamp_int(0, block.w, block.coded_width - block.x);
}

int valid_rows(const Block& block) {
  return clamp_int(0, block.h, block.coded_height - block.y);
}

// One filtered luma sample on the above neighbour line, before >> 3.
int filtered_above(const Block& block, int filter, int i) {
  const int sub_x = block.sub_x;
  const int sub_y = block.sub_y;
  const int x = block.x << sub_x;
  const int y = block.y << sub_y;
  int t = 0;
  for (int dy = -sub_y; dy <= sub_y; ++dy) {
    const int row = clamp_int(
        0, block.luma_max_y,
        std::max(block.sb_top_y - 1, y - 1 - sub_y + dy));
    for (int dx = -sub_x; dx <= sub_x; ++dx) {
      const int offset = std::max(0, (i << sub_x) + dx);
      const int col = clamp_int(0, block.luma_max_x, x + offset);
      const int weight = ds_weight(filter, dy, dx, sub_x, sub_y);
      if (weight != 0) {
        t += weight * static_cast<int>(
                         block.luma[plane_index(block.luma_stride, row, col)]);
      }
    }
  }
  return t;
}

// One filtered luma sample on the left neighbour column, before >> 3.
int filtered_left(const Block& block, int filter, int j) {
  const int sub_x = block.sub_x;
  const int sub_y = block.sub_y;
  const int x = block.x << sub_x;
  const int y = block.y << sub_y;
  int t = 0;
  for (int dy = -sub_y; dy <= sub_y; ++dy) {
    const int offset = std::max(0, (j << sub_y) + dy);
    const int row = clamp_int(0, block.luma_max_y, y + offset);
    for (int dx = -sub_x; dx <= sub_x; ++dx) {
      const int col =
          clamp_int(0, block.luma_max_x, x - 1 - sub_x + dx);
      const int weight = ds_weight(filter, dy, dx, sub_x, sub_y);
      if (weight != 0) {
        t += weight * static_cast<int>(
                         block.luma[plane_index(block.luma_stride, row, col)]);
      }
    }
  }
  return t;
}

// Chroma neighbour used by the derived-alpha correlation, with §7.13.5's
// prevSample replication at the right/bottom coded edge.
int chroma_sample_with_replication(const Block& block, int row, int col,
                                   int valid) {
  const int init = 1 << (block.bit_depth - 1);
  if (col >= 0 && col < valid) {
    return static_cast<int>(
        block.chroma[plane_index(block.chroma_stride, row, block.x + col)]);
  }
  if (valid > 0) {
    return static_cast<int>(block.chroma[plane_index(
        block.chroma_stride, row, block.x + valid - 1)]);
  }
  return init;
}

struct PlaneBest {
  Alpha alpha{};
  std::int64_t sse = 0;
};

// All 17 values a single plane's alpha can take: zero, then +1..+8 eighths,
// then -1..-8 eighths. Ties keep the earlier (larger-magnitude-positive)
// candidate except that DC is evaluated first, so zero wins exact ties.
PlaneBest best_alpha_exhaustive(const Block& block, const Scratch& scratch,
                                const SourcePlane& source, bool nonzero_only) {
  PlaneBest best{};
  bool have_best = false;
  const auto consider = [&](Alpha candidate) {
    const std::int64_t sse =
        prediction_sse(block, alpha_q8(candidate), scratch, source);
    if (!have_best || sse < best.sse) {
      have_best = true;
      best.alpha = candidate;
      best.sse = sse;
    }
  };
  if (!nonzero_only) {
    consider(Alpha{kSignZero, 0});
  }
  for (int magnitude = 1; magnitude <= kAlphabetSize; ++magnitude) {
    consider(Alpha{kSignPos, static_cast<std::int8_t>(magnitude)});
    consider(Alpha{kSignNeg, static_cast<std::int8_t>(magnitude)});
  }
  return best;
}

bool same_geometry(const Block& a, const Block& b) {
  return a.luma == b.luma && a.luma_stride == b.luma_stride &&
         a.luma_max_x == b.luma_max_x && a.luma_max_y == b.luma_max_y &&
         a.sb_top_y == b.sb_top_y && a.x == b.x && a.y == b.y &&
         a.w == b.w && a.h == b.h && a.coded_width == b.coded_width &&
         a.coded_height == b.coded_height && a.bit_depth == b.bit_depth &&
         a.ds_filter == b.ds_filter && a.sub_x == b.sub_x &&
         a.sub_y == b.sub_y;
}

}  // namespace

bool valid(const Block& block) {
  if (block.luma == nullptr || block.chroma == nullptr) {
    return false;
  }
  if (block.w <= 0 || block.h <= 0 || block.w > kMaxDim ||
      block.h > kMaxDim) {
    return false;
  }
  if (block.luma_stride <= 0 || block.chroma_stride <= 0) {
    return false;
  }
  if (block.luma_max_x < 0 || block.luma_max_y < 0) {
    return false;
  }
  if (block.x < 0 || block.y < 0) {
    return false;
  }
  if (block.chroma_stride < block.x + block.w) {
    return false;
  }
  if (block.bit_depth < 8 || block.bit_depth > 12) {
    return false;
  }
  if (block.sub_x < 0 || block.sub_x > 1 || block.sub_y < 0 ||
      block.sub_y > 1) {
    return false;
  }
  if (block.ds_filter < 0 || block.ds_filter > 3) {
    return false;
  }
  if (block.have_above != 0 && block.y <= 0) {
    return false;
  }
  if (block.have_left != 0 && block.x <= 0) {
    return false;
  }
  return true;
}

void compute_luma(const Block& block, Scratch& scratch) {
  const int filter = normalised_filter(block);
  const int sub_x = block.sub_x;
  const int sub_y = block.sub_y;
  const int init = 1 << (block.bit_depth - 1);
  const int w = block.w;
  const int h = block.h;
  const int step_w = w > 32 ? 2 : 1;
  const int step_h = h > 32 ? 2 : 1;

  int luma_sum = 0;
  int luma_count = 0;

  // --- §7.13.5 lumaAvg: above neighbour line ------------------------------
  if (block.have_above != 0) {
    const int limit = valid_columns(block);
    for (int i = 0; i < w; ++i) {
      int t;
      if (i < limit) {
        t = filtered_above(block, filter, i);
      } else if (limit > 0) {
        t = filtered_above(block, filter, limit - 1);
      } else {
        t = init;
      }
      if (i % step_w == 0) {
        luma_sum += t;
      }
      scratch.luma_above[i] = t >> 3;
    }
    luma_count += w / step_w;
  }

  // --- §7.13.5 lumaAvg: left neighbour column ------------------------------
  if (block.have_left != 0) {
    const int limit = valid_rows(block);
    for (int j = 0; j < h; ++j) {
      int t;
      if (j < limit) {
        t = filtered_left(block, filter, j);
      } else if (limit > 0) {
        t = filtered_left(block, filter, limit - 1);
      } else {
        t = init;
      }
      if (j % step_h == 0) {
        luma_sum += t;
      }
      scratch.luma_left[j] = t >> 3;
    }
    luma_count += h / step_h;
  }

  scratch.luma_avg = 8 << (block.bit_depth - 1);
  if (luma_count > 0) {
    const int average = approx_divide(luma_sum, luma_count);
    const int maximum = (8 << block.bit_depth) - 1;
    scratch.luma_avg = average < maximum ? average : maximum;
  }

  // --- §7.13.5 array L ----------------------------------------------------
  for (int i = 0; i < h; ++i) {
    const int luma_y = (block.y + i) << sub_y;
    const bool clamp_y = (i == 0) || (luma_y % 64 == 0);
    for (int j = 0; j < w; ++j) {
      const int luma_x = (block.x + j) << sub_x;
      const bool clamp_x = (j == 0) || (luma_x % 64 == 0);
      int t = 0;
      for (int dy = -sub_y; dy <= sub_y; ++dy) {
        const int clamp_dy = clamp_y ? (dy > 0 ? dy : 0) : dy;
        const int row = clamp_int(0, block.luma_max_y, luma_y + clamp_dy);
        for (int dx = -sub_x; dx <= sub_x; ++dx) {
          const int clamp_dx = clamp_x ? (dx > 0 ? dx : 0) : dx;
          const int col = clamp_int(0, block.luma_max_x, luma_x + clamp_dx);
          const int weight = ds_weight(filter, dy, dx, sub_x, sub_y);
          if (weight != 0) {
            t += weight * static_cast<int>(block.luma[plane_index(
                             block.luma_stride, row, col)]);
          }
        }
      }
      scratch.luma[i * w + j] = t;
    }
  }
}

int derive_implicit_alpha(const Block& block, const Scratch& scratch) {
  const int w = block.w;
  const int h = block.h;

  int num_above = 0;
  int num_left = 0;
  if (block.have_above != 0 && block.have_left != 0) {
    if (w > h * 2) {
      num_above = kNumRefSamCfl;
    } else if (h > w * 2) {
      num_left = kNumRefSamCfl;
    } else {
      num_above = kNumRefSamCfl >> 1;
      num_left = kNumRefSamCfl >> 1;
    }
  } else {
    num_above = block.have_above != 0 ? kNumRefSamCfl : 0;
    num_left = block.have_left != 0 ? kNumRefSamCfl : 0;
  }
  num_above = std::min(num_above, w);
  num_left = std::min(num_left, h);

  int count = 0;
  std::int64_t sum_x = 0;
  std::int64_t sum_y = 0;
  std::int64_t sum_xy = 0;
  std::int64_t sum_xx = 0;

  int samples[kMaxDim]{};
  if (num_above > 0) {
    const int step = w / num_above;
    const int limit = valid_columns(block);
    const int row = block.y - 1;
    for (int i = 0; i < w; ++i) {
      samples[i] = chroma_sample_with_replication(block, row, i, limit);
    }
    for (int i = step >> 1; i < w; i += step) {
      const int lx = scratch.luma_above[i];
      const int sample = samples[i];
      sum_x += lx;
      sum_y += sample;
      sum_xy += static_cast<std::int64_t>(lx) * sample;
      sum_xx += static_cast<std::int64_t>(lx) * lx;
      ++count;
    }
  }
  if (num_left > 0) {
    const int step = h / num_left;
    const int limit = valid_rows(block);
    const int col = block.x - 1;
    for (int j = 0; j < h; ++j) {
      if (j < limit) {
        samples[j] = static_cast<int>(block.chroma[plane_index(
            block.chroma_stride, block.y + j, col)]);
      } else if (limit > 0) {
        samples[j] = static_cast<int>(block.chroma[plane_index(
            block.chroma_stride, block.y + limit - 1, col)]);
      } else {
        samples[j] = 1 << (block.bit_depth - 1);
      }
    }
    for (int j = step >> 1; j < h; j += step) {
      const int lx = scratch.luma_left[j];
      const int sample = samples[j];
      sum_x += lx;
      sum_y += sample;
      sum_xy += static_cast<std::int64_t>(lx) * sample;
      sum_xx += static_cast<std::int64_t>(lx) * lx;
      ++count;
    }
  }

  int implicit_alpha = 0;
  if (count > 0) {
    const std::int64_t der =
        sum_xx - (sum_x * sum_x) / static_cast<std::int64_t>(count);
    const std::int64_t nor =
        sum_xy - (sum_x * sum_y) / static_cast<std::int64_t>(count);
    if (der != 0 && nor != 0) {
      implicit_alpha = resolve_division(static_cast<int>(nor),
                                        static_cast<int>(der), 8);
    }
  }
  return implicit_alpha;
}

void apply_prediction(const Block& block, int alpha_q8_value,
                      const Scratch& scratch) {
  const int w = block.w;
  const int h = block.h;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int luma_value = scratch.luma[i * w + j];
      const int scaled =
          round2_signed(alpha_q8_value * (luma_value - scratch.luma_avg),
                        kScaleShift);
      const std::size_t index =
          plane_index(block.chroma_stride, block.y + i, block.x + j);
      const int dc = static_cast<int>(block.chroma[index]);
      block.chroma[index] = static_cast<std::uint8_t>(
          clip1(dc + scaled, block.bit_depth));
    }
  }
}

int predict_explicit(const Block& block, Alpha alpha, Scratch& scratch) {
  compute_luma(block, scratch);
  const int value = alpha_q8(alpha);
  apply_prediction(block, value, scratch);
  return value;
}

int predict_derived(const Block& block, Scratch& scratch) {
  compute_luma(block, scratch);
  const int value = derive_implicit_alpha(block, scratch);
  apply_prediction(block, value, scratch);
  return value;
}

std::int64_t prediction_sse(const Block& block, int alpha_q8_value,
                            const Scratch& scratch,
                            const SourcePlane& source) {
  if (source.plane == nullptr || source.stride <= 0) {
    return 0;
  }
  const int w = block.w;
  const int h = block.h;
  std::int64_t sse = 0;
  for (int i = 0; i < h; ++i) {
    const int source_y = clamp_int(0, source.max_y, block.y + i);
    for (int j = 0; j < w; ++j) {
      const int luma_value = scratch.luma[i * w + j];
      const int scaled =
          round2_signed(alpha_q8_value * (luma_value - scratch.luma_avg),
                        kScaleShift);
      const std::size_t index =
          plane_index(block.chroma_stride, block.y + i, block.x + j);
      const int dc = static_cast<int>(block.chroma[index]);
      const int prediction = clip1(dc + scaled, block.bit_depth);
      const int source_x = clamp_int(0, source.max_x, block.x + j);
      const int sample = static_cast<int>(
          source.plane[plane_index(source.stride, source_y, source_x)]);
      const int diff = prediction - sample;
      sse += static_cast<std::int64_t>(diff) * diff;
    }
  }
  return sse;
}

std::optional<SearchResult> search_alpha(const Block& u, const Block& v,
                                         const SourcePlane& source_u,
                                         const SourcePlane& source_v,
                                         SearchMode mode, Scratch& scratch) {
  if (!valid(u) || !valid(v) || !same_geometry(u, v)) {
    return std::nullopt;
  }
  compute_luma(u, scratch);

  SearchResult result{};
  const bool have_sources =
      source_u.plane != nullptr && source_v.plane != nullptr;

  if (mode == SearchMode::kExhaustiveSse) {
    if (!have_sources) {
      return std::nullopt;
    }
    const PlaneBest best_u =
        best_alpha_exhaustive(u, scratch, source_u, false);
    const PlaneBest best_v =
        best_alpha_exhaustive(v, scratch, source_v, false);
    result.alpha = AlphaPair{best_u.alpha, best_v.alpha};
    result.sse_u = best_u.sse;
    result.sse_v = best_v.sse;
    if (!signalable(result.alpha)) {
      // Both planes chose DC; the bitstream forbids (zero, zero). Force the
      // plane whose resulting *pair* cost is lower. best_u.sse/best_v.sse are
      // the DC (zero-alpha) costs in this branch.
      const PlaneBest nonzero_u =
          best_alpha_exhaustive(u, scratch, source_u, true);
      const PlaneBest nonzero_v =
          best_alpha_exhaustive(v, scratch, source_v, true);
      if (nonzero_u.sse + best_v.sse <= best_u.sse + nonzero_v.sse) {
        result.alpha.u = nonzero_u.alpha;
        result.sse_u = nonzero_u.sse;
      } else {
        result.alpha.v = nonzero_v.alpha;
        result.sse_v = nonzero_v.sse;
      }
    }
  } else {
    const Alpha snapped_u = snap_alpha_q8(derive_implicit_alpha(u, scratch));
    const Alpha snapped_v = snap_alpha_q8(derive_implicit_alpha(v, scratch));
    result.alpha = AlphaPair{snapped_u, snapped_v};
    if (!signalable(result.alpha)) {
      if (have_sources) {
        const PlaneBest nonzero_u =
            best_alpha_exhaustive(u, scratch, source_u, true);
        const PlaneBest nonzero_v =
            best_alpha_exhaustive(v, scratch, source_v, true);
        const std::int64_t zero_u = prediction_sse(u, 0, scratch, source_u);
        const std::int64_t zero_v = prediction_sse(v, 0, scratch, source_v);
        if (nonzero_u.sse + zero_v <= zero_u + nonzero_v.sse) {
          result.alpha.u = nonzero_u.alpha;
        } else {
          result.alpha.v = nonzero_v.alpha;
        }
      } else {
        // No source planes: pick the smallest signalable non-zero alpha on U.
        result.alpha.u = Alpha{kSignPos, 1};
      }
    }
    if (have_sources) {
      result.sse_u = prediction_sse(u, alpha_q8(result.alpha.u), scratch,
                                    source_u);
      result.sse_v = prediction_sse(v, alpha_q8(result.alpha.v), scratch,
                                    source_v);
    }
  }

  if (!signalable(result.alpha)) {
    return std::nullopt;
  }
  result.syntax = explicit_signalling(result.alpha);
  return result;
}

void apply_prediction_pair(const Block& u, const Block& v,
                           const AlphaPair& alpha, const Scratch& scratch) {
  apply_prediction(u, alpha_q8(alpha.u), scratch);
  apply_prediction(v, alpha_q8(alpha.v), scratch);
}

bool cfl_allowed(int chroma_w, int chroma_h, bool enable_cfl_intra,
                 bool is_chroma_part, bool frame_is_intra,
                 bool cfl_allowed_in_sdp, bool lossless) {
  if (chroma_w <= 0 || chroma_h <= 0) {
    return false;
  }
  if (!enable_cfl_intra) {
    return false;
  }
  if (is_chroma_part && frame_is_intra && !cfl_allowed_in_sdp) {
    return false;
  }
  if (lossless) {
    return chroma_w == 4 && chroma_h == 4;
  }
  return chroma_w <= 64 && chroma_h <= 64;
}

}  // namespace av2enc::cfl
