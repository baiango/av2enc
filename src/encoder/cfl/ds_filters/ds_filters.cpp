// AV2 v1.0.0 §7.13.5 luma down-sampling filters for CfL.
//
// The arithmetic below is a direct transcription of the §7.13.5 pseudocode
// with the same bookkeeping conventions as src/encoder/cfl/cfl.cpp:
//
//   * the above luma row is clamped to max(sbTop - 1, y - 1 - subY + dy),
//   * plane reads are clamped to the caller's reconstructed-plane bounds,
//   * the luma-average loops replicate the last valid sample when the block
//     crosses the coded right/bottom edge,
//   * array L clamps negative taps on the first block row/column and on a
//     64-sample luma boundary.
//
// The deliberate duplication is what makes the component independently
// testable: ds_filters_test.cpp asserts byte equality with cfl::compute_luma /
// apply_prediction for every filter and edge case.

#include "ds_filters.h"

#include <algorithm>
#include <cstddef>

namespace av2enc::cfl::ds {
namespace {

int clamp_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

std::size_t plane_index(int stride, int row, int col) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(col);
}

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
      const int w = weight(filter, sub_x, sub_y, dy, dx);
      if (w != 0) {
        t += w * static_cast<int>(
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
      const int col = clamp_int(0, block.luma_max_x, x - 1 - sub_x + dx);
      const int w = weight(filter, sub_x, sub_y, dy, dx);
      if (w != 0) {
        t += w * static_cast<int>(
                     block.luma[plane_index(block.luma_stride, row, col)]);
      }
    }
  }
  return t;
}

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

}  // namespace

void compute_luma(const Block& block, int cfl_ds_filter_index,
                  LumaScratch& scratch) {
  const int filter = normalise_index(cfl_ds_filter_index);
  const int sub_x = block.sub_x;
  const int sub_y = block.sub_y;
  const int init = 1 << (block.bit_depth - 1);
  const int w = block.w;
  const int h = block.h;
  const int step_w = w > 32 ? 2 : 1;
  const int step_h = h > 32 ? 2 : 1;

  for (int i = 0; i < w; ++i) {
    scratch.above[i] = 0;
  }
  for (int j = 0; j < h; ++j) {
    scratch.left[j] = 0;
  }

  int luma_sum = 0;
  int luma_count = 0;

  // --- lumaAvg: above neighbour line --------------------------------------
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
      scratch.above[i] = t >> 3;
    }
    luma_count += w / step_w;
  }

  // --- lumaAvg: left neighbour column -------------------------------------
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
      scratch.left[j] = t >> 3;
    }
    luma_count += h / step_h;
  }

  scratch.luma_avg = 8 << (block.bit_depth - 1);
  if (luma_count > 0) {
    const int average = approx_divide(luma_sum, luma_count);
    const int maximum = (8 << block.bit_depth) - 1;
    scratch.luma_avg = average < maximum ? average : maximum;
  }

  // --- array L ------------------------------------------------------------
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
          const int wgt = weight(filter, sub_x, sub_y, dy, dx);
          if (wgt != 0) {
            t += wgt * static_cast<int>(block.luma[plane_index(
                             block.luma_stride, row, col)]);
          }
        }
      }
      scratch.l[i * w + j] = t;
    }
  }
}

void compute_luma(const Block& block, LumaScratch& scratch) {
  compute_luma(block, block.ds_filter, scratch);
}

int derive_implicit_alpha(const Block& block, const LumaScratch& scratch) {
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
      const int lx = scratch.above[i];
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
      const int lx = scratch.left[j];
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

void apply_prediction(const Block& block, int alpha_q8,
                      const LumaScratch& scratch) {
  const int w = block.w;
  const int h = block.h;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int luma_value = scratch.l[i * w + j];
      const int scaled =
          round2_signed(alpha_q8 * (luma_value - scratch.luma_avg),
                        kScaleShift);
      const std::size_t index =
          plane_index(block.chroma_stride, block.y + i, block.x + j);
      const int dc = static_cast<int>(block.chroma[index]);
      block.chroma[index] = static_cast<std::uint8_t>(
          clip1(dc + scaled, block.bit_depth));
    }
  }
}

std::int64_t prediction_sse(const Block& block, int alpha_q8,
                            const LumaScratch& scratch,
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
      const int luma_value = scratch.l[i * w + j];
      const int scaled =
          round2_signed(alpha_q8 * (luma_value - scratch.luma_avg),
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

std::optional<FilterEvaluation> evaluate_filter(
    const Block& u, const Block& v, const SourcePlane& source_u,
    const SourcePlane& source_v, int cfl_ds_filter_index, SearchMode mode,
    Scratch& alpha_scratch) {
  if (!valid_index(cfl_ds_filter_index)) {
    return std::nullopt;
  }
  Block uu = u;
  Block vv = v;
  uu.ds_filter = cfl_ds_filter_index;
  vv.ds_filter = cfl_ds_filter_index;
  const std::optional<SearchResult> result =
      search_alpha(uu, vv, source_u, source_v, mode, alpha_scratch);
  if (!result.has_value()) {
    return std::nullopt;
  }
  FilterEvaluation out{};
  out.index = normalise_index(cfl_ds_filter_index);
  out.alpha = result->alpha;
  out.syntax = result->syntax;
  out.sse_u = result->sse_u;
  out.sse_v = result->sse_v;
  return out;
}

std::optional<FilterChoice> search_filter(
    const Block& u, const Block& v, const SourcePlane& source_u,
    const SourcePlane& source_v, const SearchOptions& options, SearchMode mode,
    Scratch& alpha_scratch) {
  FilterChoice choice{};
  bool have_evaluation = false;
  double best_cost = 0.0;
  for (int index = 0; index < kNumDistinctFilters; ++index) {
    const std::optional<FilterEvaluation> evaluation =
        evaluate_filter(u, v, source_u, source_v, index, mode, alpha_scratch);
    if (!evaluation.has_value()) {
      continue;
    }
    FilterCost cost{};
    cost.index = index;
    cost.sse_u = evaluation->sse_u;
    cost.sse_v = evaluation->sse_v;
    cost.blocks = 1;
    cost.rate_bits = options.rate_bits[static_cast<std::size_t>(index)];
    cost.rd = static_cast<double>(cost.total_sse());
    if (options.rate_distortion) {
      cost.rd += options.lambda * cost.rate_bits;
    }
    choice.costs[static_cast<std::size_t>(index)] = cost;
    if (!have_evaluation || cost.rd < best_cost) {
      have_evaluation = true;
      best_cost = cost.rd;
      choice.index = index;
      choice.evaluation = *evaluation;
    }
  }
  if (!have_evaluation) {
    return std::nullopt;
  }
  return choice;
}

bool SequenceFilterSearch::add_block(const Block& u, const Block& v,
                                     const SourcePlane& source_u,
                                     const SourcePlane& source_v,
                                     SearchMode mode,
                                     Scratch& alpha_scratch) {
  const std::optional<FilterChoice> choice =
      search_filter(u, v, source_u, source_v, SearchOptions{}, mode,
                    alpha_scratch);
  if (!choice.has_value()) {
    return false;
  }
  for (int index = 0; index < kNumDistinctFilters; ++index) {
    const FilterCost& c = choice->costs[static_cast<std::size_t>(index)];
    if (c.blocks == 0) {
      continue;
    }
    accumulators_[static_cast<std::size_t>(index)].index = index;
    accumulators_[static_cast<std::size_t>(index)].sse_u += c.sse_u;
    accumulators_[static_cast<std::size_t>(index)].sse_v += c.sse_v;
    accumulators_[static_cast<std::size_t>(index)].blocks += 1;
  }
  ++blocks_;
  return true;
}

FilterCost SequenceFilterSearch::cost(int cfl_ds_filter_index,
                                      const SearchOptions& options) const {
  const int index = clamp_sequence_index(cfl_ds_filter_index);
  FilterCost out = accumulators_[static_cast<std::size_t>(index)];
  out.index = index;
  out.rate_bits = options.rate_bits[static_cast<std::size_t>(index)];
  out.rd = static_cast<double>(out.total_sse());
  if (options.rate_distortion) {
    out.rd += options.lambda * out.rate_bits;
  }
  return out;
}

int SequenceFilterSearch::choose(const SearchOptions& options) const {
  int best_index = 0;
  double best_cost = 0.0;
  bool have_best = false;
  for (int index = 0; index < kNumDistinctFilters; ++index) {
    if (accumulators_[static_cast<std::size_t>(index)].blocks == 0) {
      continue;
    }
    const FilterCost c = cost(index, options);
    if (!have_best || c.rd < best_cost) {
      have_best = true;
      best_cost = c.rd;
      best_index = index;
    }
  }
  return best_index;
}

}  // namespace av2enc::cfl::ds
