// AV2 §7.18 CDEF — encoder-side strength search (host + device core).
//
// The decoder-side filter (`cdef.h` / `cdef_core.h`) evaluates a *given*
// parameter set.  This header adds the encoder-side decision: for every 64x64
// luma block pick one of `CdefStrengths` legal strength sets to minimise the
// squared error against the original picture.
//
// Two things make the search non-obvious and are handled here:
//
//   1. `cdef_idx` has no "off" syntax once the frame enables CDEF and the block
//      is not skipped: §5.20.10.1 always writes an index in
//      [0, CdefStrengths).  A block can therefore only be left unfiltered if
//      the header carries an all-zero strength set and the search picks it.
//      `CdefSearchOptions::include_off` adds that set (index 0).
//   2. With CdefStrengths == 1 the decoder reads *no* index symbol at all
//      (§5.20.10.1 takes the `CdefStrengths == 1` branch), so every block is
//      forced to index 0.  The search then has to choose the one best global
//      set; per-block selection is not signalable.
//
// The evaluation re-uses the exact §7.18 filter arithmetic from cdef_core.h, so
// the SSE the search sees is the SSE of the samples the decoder will produce.
// All arithmetic is integer; SSE sums are exact and order-independent, which is
// what makes the CPU and CUDA searches comparable bit-for-bit.
//
// Scope: 8-bit 4:2:0, MI-aligned planes, caller-owned memory.

#ifndef AV2ENC_ENCODER_FILTERS_CDEF_CDEF_SEARCH_H_
#define AV2ENC_ENCODER_FILTERS_CDEF_CDEF_SEARCH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <algorithm>

#include "codec/filters/cdef/cdef.h"
#include "codec/filters/cdef/cdef_core.h"

namespace av2enc::cdef {

// Default candidate ladder, weak -> strong.  Index 0 is the all-zero set, i.e.
// "copy" (a legal strength set that leaves the block unfiltered).  These are
// syntax-value strengths: pri 0..15, sec in {0,1,2,4}.
inline constexpr int kCdefDefaultCandidateCount = 10;
AV2ENC_CDEF_TABLE CdefStrength kCdefDefaultCandidates[kCdefDefaultCandidateCount] = {
    {0, 0, 0, 0},    // off / copy
    {1, 0, 0, 0},    // very weak luma
    {2, 0, 0, 0},    // weak luma
    {4, 0, 0, 0},    // luma only
    {4, 2, 0, 0},    // luma primary + secondary
    {4, 2, 2, 0},    // + weak chroma primary
    {4, 2, 4, 2},    // luma + full chroma
    {6, 2, 4, 2},    //
    {8, 4, 4, 2},    //
    {12, 4, 8, 4},   // strong
};

// Compact candidate ladder for the bounded video search
// (`--video-cdef-search ladder|fast`).  Four sets instead of ten: the identity
// set plus the weak / moderate / strong rungs the default ladder's winner
// normally lands on.  Standard syntax-value strengths.
inline constexpr int kCdefFastCandidateCount = 4;
AV2ENC_CDEF_TABLE CdefStrength kCdefFastCandidates[kCdefFastCandidateCount] = {
    {0, 0, 0, 0},   // off / copy
    {2, 0, 0, 0},   // weak luma
    {4, 2, 0, 0},   // luma primary + secondary
    {8, 4, 4, 2},   // strong luma + full chroma
};

AV2ENC_CDEF_HD constexpr bool cdef_is_zero_strength(const CdefStrength& s) {
  return s.y_pri == 0 && s.y_sec == 0 && s.uv_pri == 0 && s.uv_sec == 0;
}

AV2ENC_CDEF_HD constexpr bool cdef_valid_strength(const CdefStrength& s) {
  return s.y_pri >= 0 && s.y_pri <= 15 && s.uv_pri >= 0 && s.uv_pri <= 15 &&
         (s.y_sec == 0 || s.y_sec == 1 || s.y_sec == 2 || s.y_sec == 4) &&
         (s.uv_sec == 0 || s.uv_sec == 1 || s.uv_sec == 2 || s.uv_sec == 4);
}

// One input plane for the search.  `src` is the unfiltered reconstruction;
// `ref` the original picture used only for the SSE metric.
struct CdefSearchPlane {
  const std::uint8_t* data = nullptr;
  int stride = 0;
};

// Everything the search needs.  Grids are in luma MI units unless noted.
struct CdefSearchFrame {
  CdefSearchPlane src[3];
  CdefSearchPlane ref[3];
  int width = 0;   // visible luma width
  int height = 0;  // visible luma height
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  const std::uint8_t* skip = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint8_t* lossless_chroma = nullptr;
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Squared error of one 8x8 CDEF block, split by plane group.  `luma` counts the
// 64 luma samples, `chroma` the two 4x4 chroma blocks (16 samples each).
struct CdefBlockSse {
  std::int64_t luma = 0;
  std::int64_t chroma = 0;
};

// Evaluates one 8x8 luma CDEF block (plus its chroma blocks unless skipped) for
// a strength set whose block-derived inputs (`y_dir`, `var`, `skip`,
// `skip_chroma`) were already computed.  Those four values do not depend on the
// strength set, so the search computes them once per block and reuses them
// across the whole candidate ladder; `cdef_eval_block()` below is the
// self-contained wrapper.
AV2ENC_CDEF_HD CdefBlockSse cdef_eval_block_params(
    const CdefSearchFrame& frame, const CdefStrength& strength, int damping,
    bool disable_loopfilters_across_tiles, int r, int c, int y_dir,
    std::int64_t var, bool skip, bool skip_chroma,
    std::uint8_t* const dst[3]) {
  CdefBlockSse sse;
  const int sub_x = frame.subsampling_x;
  const int sub_y = frame.subsampling_y;
  const CdefRegion region = cdef_filter_region(
      frame.mi_cols, frame.mi_rows, frame.tile_col_starts, frame.num_tile_cols,
      frame.tile_row_starts, frame.num_tile_rows,
      disable_loopfilters_across_tiles, r, c);

  CdefFilterParams luma{0, 0, 0, damping};
  CdefFilterParams chroma{0, 0, 0, damping - 1};
  if (!skip) {
    luma = cdef_luma_params(strength.y_pri, strength.y_sec, y_dir, var, damping);
    chroma = cdef_chroma_params(strength.uv_pri, strength.uv_sec, y_dir, sub_x,
                                sub_y, damping);
  }

  const int x0 = c * kMiSize;
  const int y0 = r * kMiSize;
  const int vis_w = (frame.width + 1) >> 1;   // ceil(width / 2)
  const int vis_h = (frame.height + 1) >> 1;  // ceil(height / 2)

  // Luma.
  {
    const std::uint8_t* src = frame.src[0].data;
    const std::size_t stride = static_cast<std::size_t>(frame.src[0].stride);
    const std::uint8_t* ref = frame.ref[0].data;
    const std::size_t ref_stride = static_cast<std::size_t>(frame.ref[0].stride);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const std::size_t o = static_cast<std::size_t>(y0 + i) * stride +
                              static_cast<std::size_t>(x0 + j);
        int value = src[o];
        if (!skip) {
          value = cdef_filter_sample(src, stride, x0, y0, i, j, luma.dir,
                                     luma.pri, luma.sec, luma.damping, 0, 0,
                                     region);
        }
        if (dst[0] != nullptr) {
          dst[0][o] = static_cast<std::uint8_t>(value);
        }
        if (x0 + j < frame.width && y0 + i < frame.height) {
          const int diff =
              value - static_cast<int>(ref[static_cast<std::size_t>(y0 + i) *
                                               ref_stride +
                                           static_cast<std::size_t>(x0 + j)]);
          sse.luma += static_cast<std::int64_t>(diff) * diff;
        }
      }
    }
  }

  // Chroma.
  {
    const bool filter_chroma = !skip && !skip_chroma;
    const int cx0 = x0 >> sub_x;
    const int cy0 = y0 >> sub_y;
    const int cw = 8 >> sub_x;
    const int ch = 8 >> sub_y;
    for (int plane = 1; plane < 3; ++plane) {
      const std::uint8_t* src = frame.src[plane].data;
      const std::size_t stride =
          static_cast<std::size_t>(frame.src[plane].stride);
      const std::uint8_t* ref = frame.ref[plane].data;
      const std::size_t ref_stride =
          static_cast<std::size_t>(frame.ref[plane].stride);
      for (int i = 0; i < ch; ++i) {
        for (int j = 0; j < cw; ++j) {
          const std::size_t o = static_cast<std::size_t>(cy0 + i) * stride +
                                static_cast<std::size_t>(cx0 + j);
          int value = src[o];
          if (filter_chroma) {
            value = cdef_filter_sample(src, stride, cx0, cy0, i, j, chroma.dir,
                                       chroma.pri, chroma.sec, chroma.damping,
                                       sub_x, sub_y, region);
          }
          if (dst[plane] != nullptr) {
            dst[plane][o] = static_cast<std::uint8_t>(value);
          }
          if (cx0 + j < vis_w && cy0 + i < vis_h) {
            const int diff = value - static_cast<int>(
                                         ref[static_cast<std::size_t>(cy0 + i) *
                                                 ref_stride +
                                             static_cast<std::size_t>(cx0 + j)]);
            sse.chroma += static_cast<std::int64_t>(diff) * diff;
          }
        }
      }
    }
  }
  return sse;
}

// Self-contained single-block evaluation: derives the block skip flags and the
// §7.18.2 direction/variance, then runs the filter.  Convenience wrapper around
// cdef_eval_block_params(); the search uses the split form to compute those
// block-derived values once per block instead of once per candidate.
AV2ENC_CDEF_HD CdefBlockSse cdef_eval_block(
    const CdefSearchFrame& frame, const CdefStrength& strength, int damping,
    bool on_skip_txfm, bool disable_loopfilters_across_tiles, int r, int c,
    std::uint8_t* const dst[3]) {
  bool skip = false;
  bool skip_chroma = false;
  cdef_block_skip_flags(frame.skip, frame.lossless, frame.lossless_chroma,
                        frame.mi_cols, r, c, on_skip_txfm, &skip,
                        &skip_chroma);
  int y_dir = 0;
  std::int64_t var = 0;
  if (!skip) {
    cdef_direction(frame.src[0].data,
                   static_cast<std::size_t>(frame.src[0].stride),
                   c * kMiSize, r * kMiSize, 8, &y_dir, &var);
  }
  return cdef_eval_block_params(frame, strength, damping,
                                disable_loopfilters_across_tiles, r, c, y_dir,
                                var, skip, skip_chroma, dst);
}

// ---------------------------------------------------------------------------
// Host-only planning
// ---------------------------------------------------------------------------

struct CdefSearchOptions {
  int damping = 3;
  // Header CdefStrengths.  1 => no index syntax, one global set.
  int num_strengths = 1;
  bool on_skip_txfm = false;
  bool disable_loopfilters_across_tiles = false;
  // Add an all-zero (copy) strength set to the candidate pool.  Without it no
  // block can be left unfiltered in a signalled stream.  When false, identity
  // sets are removed from the pool (they are still evaluated for reporting).
  bool include_off = true;
  // Candidate ladder; null/0 selects kCdefDefaultCandidates.
  const CdefStrength* candidates = nullptr;
  int candidate_count = 0;
  // Evaluate every `eval_step`-th 8x8 block row and column (2 = 2x2 MI
  // decimation: a quarter of the blocks) for the frame-level strength choice.
  // Only valid with num_strengths == 1: with a per-block index grid every block
  // needs its own exact SSE table.  1 evaluates the whole frame.
  int eval_step = 1;
  // Weight applied to chroma SSE when ranking.  1 = sample-weighted total SSE.
  std::int64_t chroma_weight = 1;
};

struct CdefSearchResult {
  int num_strengths = 0;                 // actual CdefStrengths (<= requested)
  CdefStrength strengths[kMaxStrengths] = {};
  CdefOptions options;                   // ready for cdef_frame()/the kernel
  int n64_cols = 0;
  int n64_rows = 0;
  // One entry per 64x64 luma block, row-major.  Always in [0, num_strengths):
  // a signalled stream has no -1 index.
  std::vector<std::int8_t> cdef_idx;
  std::vector<std::int64_t> block_sse;      // chosen SSE per block (luma+chroma)
  std::vector<std::int64_t> block_sse_off;  // identity-set SSE per block
  std::vector<std::int8_t> block_choice;    // header strength index per block
  std::int64_t sse_total = 0;               // chosen
  std::int64_t sse_off_total = 0;           // unfiltered
  int blocks_off = 0;                       // blocks that picked an all-zero set
  // True when CdefStrengths == 1: the stream cannot signal per-block indices,
  // so cdef_idx must be treated as "all zero" by the writer.
  bool forced_single_strength = false;
};

namespace search_detail {

inline int grid_cols(const CdefSearchFrame& f) {
  return (f.mi_cols + kCdefSize4 - 1) / kCdefSize4;
}
inline int grid_rows(const CdefSearchFrame& f) {
  return (f.mi_rows + kCdefSize4 - 1) / kCdefSize4;
}

inline bool valid_frame(const CdefSearchFrame& f, std::string* error) {
  auto fail = [error](const char* text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  if (f.width <= 0 || f.height <= 0 || f.mi_cols <= 0 || f.mi_rows <= 0) {
    return fail("invalid frame geometry");
  }
  if ((f.mi_cols & 1) != 0 || (f.mi_rows & 1) != 0) {
    return fail("odd MI grid");
  }
  if (f.subsampling_x != 1 || f.subsampling_y != 1) {
    return fail("only 4:2:0 is supported");
  }
  if (f.width > f.mi_cols * kMiSize || f.height > f.mi_rows * kMiSize) {
    return fail("visible area exceeds the MI-aligned planes");
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (f.src[plane].data == nullptr || f.ref[plane].data == nullptr) {
      return fail("null source or reference plane");
    }
    const int sub = (plane == 0) ? 0 : f.subsampling_x;
    const int width = (f.mi_cols * kMiSize) >> sub;
    if (f.src[plane].stride < width || f.ref[plane].stride < width) {
      return fail("plane stride shorter than the MI-aligned plane");
    }
  }
  if ((f.num_tile_cols > 0 && f.tile_col_starts == nullptr) ||
      (f.num_tile_rows > 0 && f.tile_row_starts == nullptr)) {
    return fail("missing tile start array");
  }
  return true;
}

// Normalises the candidate ladder: appends the default ladder when none was
// supplied, inserts the identity set when requested, and removes it when not.
inline bool build_pool(const CdefSearchOptions& options,
                       std::vector<CdefStrength>* pool, std::string* error) {
  auto fail = [error](const char* text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  pool->clear();
  if (options.candidates != nullptr && options.candidate_count > 0) {
    pool->assign(options.candidates,
                 options.candidates + options.candidate_count);
  } else {
    pool->assign(kCdefDefaultCandidates,
                 kCdefDefaultCandidates + kCdefDefaultCandidateCount);
  }
  for (const CdefStrength& s : *pool) {
    if (!cdef_valid_strength(s)) {
      return fail("illegal candidate strength set");
    }
  }
  bool have_off = false;
  for (const CdefStrength& s : *pool) {
    have_off = have_off || cdef_is_zero_strength(s);
  }
  if (options.include_off && !have_off) {
    pool->insert(pool->begin(), CdefStrength{0, 0, 0, 0});
  }
  if (!options.include_off) {
    pool->erase(std::remove_if(pool->begin(), pool->end(),
                               [](const CdefStrength& s) {
                                 return cdef_is_zero_strength(s);
                               }),
                pool->end());
  }
  // Drop duplicate strength sets: selecting two identical sets would spend an
  // index symbol for no distortion gain.
  std::vector<CdefStrength> unique;
  unique.reserve(pool->size());
  for (const CdefStrength& s : *pool) {
    bool duplicate = false;
    for (const CdefStrength& u : unique) {
      duplicate = duplicate || (u.y_pri == s.y_pri && u.y_sec == s.y_sec &&
                                u.uv_pri == s.uv_pri && u.uv_sec == s.uv_sec);
    }
    if (!duplicate) {
      unique.push_back(s);
    }
  }
  *pool = std::move(unique);
  if (pool->empty()) {
    return fail("empty candidate pool");
  }
  if (pool->size() > 32) {
    return fail("too many candidates");
  }
  return true;
}

// Selection: turns a per-(candidate, 64x64 block) SSE table into the header
// strength list and the per-block index grid.  `sse` is laid out
// `[candidate][block]`, block-major in raster order; `off_sse` is the identity
// set's SSE per block (reporting only).
//
// The CUDA search produces the same table, so this function is the single
// decision rule for both paths.
inline bool select(const std::vector<CdefStrength>& pool, int n64_cols,
                   int n64_rows, const std::vector<std::int64_t>& sse,
                   const std::vector<std::int64_t>& off_sse,
                   const CdefSearchOptions& options,
                   CdefSearchResult* result, std::string* error) {
  auto fail = [error](const char* text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  const int nblocks = n64_cols * n64_rows;
  const int ncand = static_cast<int>(pool.size());
  if (nblocks <= 0 || ncand <= 0) {
    return fail("empty search grid");
  }
  const std::size_t nb = static_cast<std::size_t>(nblocks);
  if (sse.size() != static_cast<std::size_t>(ncand) * nb ||
      off_sse.size() != nb) {
    return fail("search table size mismatch");
  }
  auto at = [&sse, nb](int k, int block) -> std::int64_t {
    return sse[static_cast<std::size_t>(k) * nb +
               static_cast<std::size_t>(block)];
  };

  const int requested = options.num_strengths;
  std::vector<int> chosen;
  if (requested == 1) {
    int best = 0;
    std::int64_t best_total = -1;
    for (int k = 0; k < ncand; ++k) {
      std::int64_t sum = 0;
      for (int b = 0; b < nblocks; ++b) {
        sum += at(k, b);
      }
      if (best_total < 0 || sum < best_total) {
        best_total = sum;
        best = k;
      }
    }
    chosen.push_back(best);
  } else {
    // Greedy: keep the best achievable per-block SSE and add the candidate
    // that reduces the total the most.  Start from the identity set when it is
    // in the pool (it is the only way to leave a block unfiltered).
    std::vector<std::int64_t> best(nb, 0);
    bool have_baseline = false;
    for (int k = 0; k < ncand; ++k) {
      if (cdef_is_zero_strength(pool[static_cast<std::size_t>(k)])) {
        chosen.push_back(k);
        for (int b = 0; b < nblocks; ++b) {
          best[static_cast<std::size_t>(b)] = at(k, b);
        }
        have_baseline = true;
        break;
      }
    }
    if (!have_baseline) {
      // No identity set: seed with the best single candidate.
      int best_k = 0;
      std::int64_t best_total = -1;
      for (int k = 0; k < ncand; ++k) {
        std::int64_t sum = 0;
        for (int b = 0; b < nblocks; ++b) {
          sum += at(k, b);
        }
        if (best_total < 0 || sum < best_total) {
          best_total = sum;
          best_k = k;
        }
      }
      chosen.push_back(best_k);
      for (int b = 0; b < nblocks; ++b) {
        best[static_cast<std::size_t>(b)] = at(best_k, b);
      }
      have_baseline = true;
    }
    while (static_cast<int>(chosen.size()) < requested) {
      int best_k = -1;
      std::int64_t best_gain = 0;
      for (int k = 0; k < ncand; ++k) {
        bool already = false;
        for (int picked : chosen) {
          already = already || picked == k;
        }
        if (already) {
          continue;
        }
        std::int64_t gain = 0;
        for (int b = 0; b < nblocks; ++b) {
          const std::int64_t value = at(k, b);
          if (value < best[static_cast<std::size_t>(b)]) {
            gain += best[static_cast<std::size_t>(b)] - value;
          }
        }
        if (gain > best_gain) {  // strict: ties keep the lower candidate index
          best_gain = gain;
          best_k = k;
        }
      }
      if (best_k < 0) {
        break;
      }
      chosen.push_back(best_k);
      for (int b = 0; b < nblocks; ++b) {
        const std::int64_t value = at(best_k, b);
        if (value < best[static_cast<std::size_t>(b)]) {
          best[static_cast<std::size_t>(b)] = value;
        }
      }
    }
  }

  // Per-block assignment over the chosen header list (lowest index wins ties).
  result->num_strengths = static_cast<int>(chosen.size());
  for (int i = 0; i < result->num_strengths; ++i) {
    result->strengths[i] =
        pool[static_cast<std::size_t>(chosen[static_cast<std::size_t>(i)])];
  }
  result->options.damping = options.damping;
  result->options.num_strengths = result->num_strengths;
  for (int i = 0; i < result->num_strengths; ++i) {
    result->options.strengths[i] = result->strengths[i];
  }
  result->options.on_skip_txfm = options.on_skip_txfm;
  result->options.disable_loopfilters_across_tiles =
      options.disable_loopfilters_across_tiles;
  result->options.bit_depth = 8;

  result->n64_cols = n64_cols;
  result->n64_rows = n64_rows;
  result->cdef_idx.assign(nb, 0);
  result->block_sse.assign(nb, 0);
  result->block_sse_off.assign(off_sse.begin(), off_sse.end());
  result->block_choice.assign(nb, 0);
  result->blocks_off = 0;
  std::int64_t chosen_total = 0;
  std::int64_t off_total = 0;
  for (int b = 0; b < nblocks; ++b) {
    int best_i = 0;
    std::int64_t best_v = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < result->num_strengths; ++i) {
      const std::int64_t v = at(chosen[static_cast<std::size_t>(i)], b);
      if (v < best_v) {
        best_v = v;
        best_i = i;
      }
    }
    result->block_choice[static_cast<std::size_t>(b)] =
        static_cast<std::int8_t>(best_i);
    result->cdef_idx[static_cast<std::size_t>(b)] =
        static_cast<std::int8_t>(best_i);
    result->block_sse[static_cast<std::size_t>(b)] = best_v;
    chosen_total += best_v;
    off_total += off_sse[static_cast<std::size_t>(b)];
    if (cdef_is_zero_strength(
            result->strengths[static_cast<std::size_t>(best_i)])) {
      ++result->blocks_off;
    }
  }
  // Prune strength sets that no block selected.  This matters: the identity
  // set is the greedy baseline, and if filtering wins everywhere it is never
  // used.  Dropping it can turn a two-set header into CdefStrengths == 1,
  // which removes the cdef_idx symbols from the tile entirely.
  std::vector<int> used;
  for (int i = 0; i < result->num_strengths; ++i) {
    bool picked = false;
    for (int b = 0; b < nblocks; ++b) {
      picked = picked ||
               result->block_choice[static_cast<std::size_t>(b)] == i;
    }
    if (picked) {
      used.push_back(i);
    }
  }
  if (!used.empty() &&
      static_cast<int>(used.size()) < result->num_strengths) {
    CdefStrength pruned[kMaxStrengths] = {};
    for (std::size_t i = 0; i < used.size(); ++i) {
      pruned[i] = result->strengths[used[i]];
    }
    result->num_strengths = static_cast<int>(used.size());
    for (int i = 0; i < result->num_strengths; ++i) {
      result->strengths[i] = pruned[i];
    }
    result->options.num_strengths = result->num_strengths;
    for (int i = 0; i < result->num_strengths; ++i) {
      result->options.strengths[i] = result->strengths[i];
    }
    for (int i = result->num_strengths; i < kMaxStrengths; ++i) {
      result->options.strengths[i] = CdefStrength{};
    }
    for (int b = 0; b < nblocks; ++b) {
      const int old_i = result->block_choice[static_cast<std::size_t>(b)];
      int new_i = 0;
      for (std::size_t i = 0; i < used.size(); ++i) {
        if (used[i] == old_i) {
          new_i = static_cast<int>(i);
          break;
        }
      }
      result->block_choice[static_cast<std::size_t>(b)] =
          static_cast<std::int8_t>(new_i);
      result->cdef_idx[static_cast<std::size_t>(b)] =
          static_cast<std::int8_t>(new_i);
    }
    result->blocks_off = 0;
    for (int b = 0; b < nblocks; ++b) {
      if (cdef_is_zero_strength(
              result->strengths[static_cast<std::size_t>(
                  result->block_choice[static_cast<std::size_t>(b)])])) {
        ++result->blocks_off;
      }
    }
  }
  result->sse_total = chosen_total;
  result->sse_off_total = off_total;
  result->forced_single_strength = result->num_strengths == 1;
  return true;
}

}  // namespace search_detail

// Evaluates every candidate over the frame, then selects the header strength
// list and the per-64x64 index grid.  Deterministic: ties keep the lowest
// candidate/header index.  Returns false with a reason on invalid input.
inline bool cdef_search_plan(const CdefSearchFrame& frame,
                             const CdefSearchOptions& options,
                             CdefSearchResult* result, std::string* error) {
  auto fail = [error](const std::string& text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  if (result == nullptr) {
    return fail("null result");
  }
  *result = CdefSearchResult{};
  if (!search_detail::valid_frame(frame, error)) {
    return false;
  }
  if (options.damping < 3 || options.damping > 6) {
    return fail("damping outside 3..6");
  }
  if (options.num_strengths < 1 || options.num_strengths > kMaxStrengths) {
    return fail("num_strengths outside 1..8");
  }
  if (options.chroma_weight < 0) {
    return fail("negative chroma weight");
  }
  if (options.eval_step < 1) {
    return fail("eval_step must be >= 1");
  }
  if (options.eval_step > 1 && options.num_strengths != 1) {
    return fail(
        "decimated CDEF search requires CdefStrengths == 1 (a per-block index "
        "grid needs every block's SSE)");
  }
  std::vector<CdefStrength> pool;
  if (!search_detail::build_pool(options, &pool, error)) {
    return false;
  }

  const int n64_cols = search_detail::grid_cols(frame);
  const int n64_rows = search_detail::grid_rows(frame);
  const int nblocks = n64_cols * n64_rows;
  const int ncand = static_cast<int>(pool.size());
  const std::int64_t chroma_weight = options.chroma_weight;
  const std::size_t nb = static_cast<std::size_t>(nblocks);

  // Evaluate every candidate over every 8x8 block once.  sse[k*nblocks + b].
  std::vector<std::int64_t> sse(static_cast<std::size_t>(ncand) * nb, 0);
  CdefStrength identity{0, 0, 0, 0};
  std::vector<std::int64_t> off_sse(nb, 0);
  int off_index = -1;
  for (int k = 0; k < ncand; ++k) {
    if (cdef_is_zero_strength(pool[static_cast<std::size_t>(k)])) {
      off_index = k;
      break;
    }
  }

  const int eval_step = options.eval_step;
  const bool decimate = eval_step > 1;

  for (int r = 0; r < frame.mi_rows; r += kCdefBlock4) {
    const int br = r / kCdefBlock4;
    if (decimate && (br % eval_step) != 0) continue;
    for (int c = 0; c < frame.mi_cols; c += kCdefBlock4) {
      const int bc = c / kCdefBlock4;
      if (decimate && (bc % eval_step) != 0) continue;
      const int block = (r / kCdefSize4) * n64_cols + (c / kCdefSize4);
      std::uint8_t* const no_dst[3] = {nullptr, nullptr, nullptr};
      // The §7.18.1 skip flags and the §7.18.2 direction/variance depend only on
      // the block, not on the strength set: derive them once per block.
      bool skip = false;
      bool skip_chroma = false;
      cdef_block_skip_flags(frame.skip, frame.lossless, frame.lossless_chroma,
                            frame.mi_cols, r, c, options.on_skip_txfm, &skip,
                            &skip_chroma);
      int y_dir = 0;
      std::int64_t var = 0;
      if (!skip) {
        cdef_direction(frame.src[0].data,
                       static_cast<std::size_t>(frame.src[0].stride),
                       c * kMiSize, r * kMiSize, 8, &y_dir, &var);
      }
      if (off_index < 0) {
        // Identity set not in the pool: evaluate it for reporting only.
        const CdefBlockSse off = cdef_eval_block_params(
            frame, identity, options.damping,
            options.disable_loopfilters_across_tiles, r, c, y_dir, var, skip,
            skip_chroma, no_dst);
        off_sse[static_cast<std::size_t>(block)] +=
            off.luma + chroma_weight * off.chroma;
      }
      for (int k = 0; k < ncand; ++k) {
        const CdefBlockSse one = cdef_eval_block_params(
            frame, pool[static_cast<std::size_t>(k)], options.damping,
            options.disable_loopfilters_across_tiles, r, c, y_dir, var, skip,
            skip_chroma, no_dst);
        sse[static_cast<std::size_t>(k) * nb +
            static_cast<std::size_t>(block)] +=
            one.luma + chroma_weight * one.chroma;
      }
    }
  }

  if (off_index >= 0) {
    const std::size_t row = static_cast<std::size_t>(off_index) * nb;
    for (int b = 0; b < nblocks; ++b) {
      off_sse[static_cast<std::size_t>(b)] =
          sse[row + static_cast<std::size_t>(b)];
    }
  }
  if (!search_detail::select(pool, n64_cols, n64_rows, sse, off_sse, options,
                             result, error)) {
    return false;
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

}  // namespace av2enc::cdef

#endif  // AV2ENC_ENCODER_FILTERS_CDEF_CDEF_SEARCH_H_
