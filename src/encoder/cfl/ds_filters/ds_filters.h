#ifndef AV2ENC_ENCODER_CFL_DS_FILTERS_DS_FILTERS_H_
#define AV2ENC_ENCODER_CFL_DS_FILTERS_DS_FILTERS_H_

// AV2 v1.0.0 §5.4.5 / §7.13.5 luma down-sampling filters for CfL.
//
// `cfl_ds_filter_index` is a sequence-level f(2) field.  The prediction
// process uses it in three places (§7.13.5):
//
//   * the above/left luma-average collection (lumaAbove / lumaLeft),
//   * the array L used for the chroma prediction,
//   * nothing in the alpha signalling (the index is not repeated per block).
//
// The normative tap tables are:
//
//   Cfl_Filters_420[3][3][3] = {{{0,0,0},{0,2,2},{0,2,2}},
//                               {{0,0,0},{1,2,1},{1,2,1}},
//                               {{0,1,0},{1,4,1},{0,1,0}}}
//   Cfl_Filters_422[3][3]    = {{0,4,4},{2,4,2},{0,8,0}}
//
// and index 3 behaves exactly like index 0 (§6.4.5 note).  All arithmetic is
// integer; there is no floating point in this module's prediction path.
//
// This component is standalone: it re-implements the filter/edge arithmetic
// rather than changing src/encoder/cfl/cfl.cpp or src/cuda/cfl/.  The tests
// cross-check it against both existing implementations.

#include "../cfl.h"

#include <array>
#include <cstdint>
#include <optional>

namespace av2enc::cfl::ds {

// cfl_ds_filter_index is f(2) (§5.4.5).
inline constexpr int kSyntaxBits = 2;
inline constexpr int kNumIndices = 1 << kSyntaxBits;
// Indices 0, 1, 2 select distinct filters; index 3 aliases 0.
inline constexpr int kNumDistinctFilters = 3;

// §5.4.5 / §6.4.5: 3 is legal on the wire but behaves as 0.
constexpr int normalise_index(int cfl_ds_filter_index) {
  return cfl_ds_filter_index == 3 ? 0 : cfl_ds_filter_index;
}

// True for every value an f(2) field can carry.
constexpr bool valid_index(int cfl_ds_filter_index) {
  return cfl_ds_filter_index >= 0 && cfl_ds_filter_index < kNumIndices;
}

// Per-sequence choice.  Only 0, 1 and 2 can be selected by an encoder; 3 is
// accepted on input (stream compatibility) and normalised to 0.
constexpr int clamp_sequence_index(int cfl_ds_filter_index) {
  return cfl_ds_filter_index < 0 || cfl_ds_filter_index >= kNumIndices
             ? 0
             : normalise_index(cfl_ds_filter_index);
}

// ---------------------------------------------------------------------------
// Normative taps
// ---------------------------------------------------------------------------

// Tap weight for 4:2:0. `dy`/`dx` are in {-1, 0, 1}; the index is normalised
// internally so callers may pass 3.
constexpr int weight_420(int filter, int dy, int dx) {
  const int f = normalise_index(filter);
  if (dy < -1 || dy > 1 || dx < -1 || dx > 1) {
    return 0;
  }
  if (f == 0) {
    return (dy >= 0 && dx >= 0) ? 2 : 0;
  }
  if (f == 1) {
    if (dy < 0) {
      return 0;
    }
    return dx == 0 ? 2 : 1;
  }
  // f == 2
  if (dy == 0 && dx == 0) {
    return 4;
  }
  if (dy == 0 || dx == 0) {
    return 1;
  }
  return 0;
}

// Tap weight for 4:2:2. §7.13.5 indexes Cfl_Filters_422 by dx, not dy.
constexpr int weight_422(int filter, int dx) {
  const int f = normalise_index(filter);
  if (dx < -1 || dx > 1) {
    return 0;
  }
  if (f == 0) {
    return dx < 0 ? 0 : 4;
  }
  if (f == 1) {
    return dx == 0 ? 4 : 2;
  }
  return dx == 0 ? 8 : 0;  // f == 2
}

// One tap for the current chroma format.  sub_x/sub_y are SubsamplingX/Y.
constexpr int weight(int filter, int sub_x, int sub_y, int dy, int dx) {
  if (sub_x != 0 && sub_y != 0) {
    return weight_420(filter, dy, dx);
  }
  if (sub_x != 0) {
    // 4:2:2 => sub_y == 0, so the dy loop only visits dy == 0.
    return dy == 0 ? weight_422(filter, dx) : 0;
  }
  // 4:4:4 (and the spec's 4:4:0 fallback): t = 8 * v.
  return (dx == 0 && dy == 0) ? 8 : 0;
}

// The full 3x3 table for one index, for exhaustive table tests.
struct Taps420 {
  int w[3][3];
};

constexpr Taps420 taps_420(int filter) {
  Taps420 t{};
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      t.w[dy + 1][dx + 1] = weight_420(filter, dy, dx);
    }
  }
  return t;
}

struct Taps422 {
  int w[3];
};

constexpr Taps422 taps_422(int filter) {
  Taps422 t{};
  for (int dx = -1; dx <= 1; ++dx) {
    t.w[dx + 1] = weight_422(filter, dx);
  }
  return t;
}

// Number of non-zero taps, useful for diagnostics and edge tests.
constexpr int num_taps_420(int filter) {
  const Taps420 t = taps_420(filter);
  int n = 0;
  for (int dy = 0; dy < 3; ++dy) {
    for (int dx = 0; dx < 3; ++dx) {
      n += t.w[dy][dx] != 0 ? 1 : 0;
    }
  }
  return n;
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

// Independent copy of the §7.13.5 scratch, so tests can compare it with the
// existing cfl::Scratch without sharing storage.
struct LumaScratch {
  int luma_avg = 0;  // 3 fractional bits
  int above[kMaxDim]{};
  int left[kMaxDim]{};
  int l[kMaxDim * kMaxDim]{};
};

// §7.13.5 lumaAvg / lumaAbove / lumaLeft / L for an explicit filter index.
// `block.ds_filter` is ignored; pass the index the sequence will signal.
void compute_luma(const Block& block, int cfl_ds_filter_index,
                  LumaScratch& scratch);

// Uses block.ds_filter.
void compute_luma(const Block& block, LumaScratch& scratch);

// §7.13.5 derived alpha (Q8, 0..511).  Requires compute_luma() first.
int derive_implicit_alpha(const Block& block, const LumaScratch& scratch);

// §7.13.5:
//   pred = Clip1(dc + Round2Signed(alpha * (L - lumaAvg), 11))
// `block.chroma` holds the DC prediction on input.  Writes w*h samples.
void apply_prediction(const Block& block, int alpha_q8,
                      const LumaScratch& scratch);

// SSE of the prediction against `source` for one alpha and one filter.
std::int64_t prediction_sse(const Block& block, int alpha_q8,
                            const LumaScratch& scratch,
                            const SourcePlane& source);

// ---------------------------------------------------------------------------
// Encoder index selection
// ---------------------------------------------------------------------------

// Cost of one candidate index for one block or one accumulated sequence.
struct FilterCost {
  int index = 0;
  std::int64_t sse_u = 0;
  std::int64_t sse_v = 0;
  std::int64_t blocks = 0;
  double rate_bits = 0.0;  // sequence-header / side-info rate attributed here
  double rd = 0.0;         // sse + lambda * rate_bits (or sse in SSE mode)
  std::int64_t total_sse() const { return sse_u + sse_v; }
};

// Everything needed to reproduce a per-block decision.
struct FilterEvaluation {
  int index = 0;
  AlphaPair alpha{};
  Signalling syntax{};
  std::int64_t sse_u = 0;
  std::int64_t sse_v = 0;
  std::int64_t total_sse() const { return sse_u + sse_v; }
};

struct SearchOptions {
  // false: minimise chroma SSE.  true: minimise SSE + lambda * rate_bits.
  // cfl_ds_filter_index is sequence-level, so for a single sequence its rate
  // is constant across candidates and SSE and RD choose the same filter unless
  // the caller supplies a non-constant rate model.
  bool rate_distortion = false;
  double lambda = 0.0;
  // Optional side-info rate per index, in bits per sequence.
  std::array<double, kNumIndices> rate_bits{};
};

// Evaluate one explicit index.  Alpha decision is the existing exhaustive SSE
// search (src/encoder/cfl/cfl.cpp), so this component does not fork the
// bitstream's alpha selection.  Returns nullopt for invalid input.
std::optional<FilterEvaluation> evaluate_filter(
    const Block& u, const Block& v, const SourcePlane& source_u,
    const SourcePlane& source_v, int cfl_ds_filter_index, SearchMode mode,
    Scratch& alpha_scratch);

// Per-block search over the distinct indices 0,1,2.  Deterministic tie-break:
// strictly lower cost wins, so equal costs always select the lowest index.
struct FilterChoice {
  int index = 0;
  FilterEvaluation evaluation{};
  std::array<FilterCost, kNumDistinctFilters> costs{};
};

std::optional<FilterChoice> search_filter(
    const Block& u, const Block& v, const SourcePlane& source_u,
    const SourcePlane& source_v, const SearchOptions& options, SearchMode mode,
    Scratch& alpha_scratch);

// Accumulates per-block candidate SSE so a caller can choose one index for a
// whole sequence (the syntax element is per sequence, not per block).
class SequenceFilterSearch {
 public:
  // Returns false when the block could not be evaluated (invalid geometry or
  // missing sources in the exhaustive mode).
  bool add_block(const Block& u, const Block& v, const SourcePlane& source_u,
                 const SourcePlane& source_v, SearchMode mode,
                 Scratch& alpha_scratch);

  // Number of blocks that contributed complete costs.
  std::int64_t blocks() const { return blocks_; }

  // Cost of index `i` (0..2; 3 normalises to 0).  `rate_bits` is per
  // sequence, not per block.
  FilterCost cost(int cfl_ds_filter_index,
                  const SearchOptions& options) const;

  // Lowest RD (or SSE) cost, ties to the lowest index.
  int choose(const SearchOptions& options) const;

  const std::array<FilterCost, kNumDistinctFilters>& accumulators() const {
    return accumulators_;
  }

 private:
  std::array<FilterCost, kNumDistinctFilters> accumulators_{};
  std::int64_t blocks_ = 0;
};

}  // namespace av2enc::cfl::ds

#endif  // AV2ENC_ENCODER_CFL_DS_FILTERS_DS_FILTERS_H_
