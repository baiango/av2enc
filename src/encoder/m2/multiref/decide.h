// Compound mode / reference decision API.
//
// Input:  a small set of candidate predictions (reference pair + MV pair, or a
//         precomputed CompoundPrediction) and the source block.
// Output: the chosen reference pair, compound mode, mask/wedge parameters, and
//         the signalled values a tile writer needs.
//
// All search is exhaustive over a fixed candidate list with strict `<`
// comparisons, so the result is deterministic: the same inputs always produce
// the same mode, references and mask.
//
// Cost model: SSE + lambda * rate_proxy.  The rate proxy counts the §5.20.7.6
// and §5.20.7.16 symbols only (comp_mode, mode, comp_group_idx, compound_type,
// wedge bits, mask_type, cwp_idx) plus a caller-supplied MV cost.  It is not a
// range-coder measurement; real entropy cost belongs to the M2 RDO layer.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "encoder/m2/multiref/compound_predict.h"
#include "encoder/m2/multiref/ref_map.h"

namespace av2enc::m2::multiref {

struct CompoundCandidateInput {
  // RefFrame indices (0-based into RefFrameMap::ref_frame_idx).
  int ref0 = 0;
  int ref1 = 1;
  mc::Mv mv[2] = {};
  // Encoder-side MV cost proxy (MVD bits, DRL bits); 0 for merge/global.
  int mv_bits = 0;
  // True when the chosen YMode will be GLOBAL_GLOBALMV (no MVD/DRL).  The
  // implicit-mask blend is disabled for that mode (§7.13.3.14).
  bool global_global_mv = false;
  // Optional precomputed predictions.  When absent, decide_compound() builds
  // them from the RefFrames in the input.
  bool has_prediction = false;
  CompoundPrediction prediction;
};

struct SingleCandidateInput {
  int ref = 0;
  mc::Mv mv = {};
  // 0 = GLOBALMV, 1 = NEARMV/merge, 2 = NEWMV.
  int mode = 0;
  int drl_index = 0;
  int mv_bits = 0;
  std::int64_t sse = std::numeric_limits<std::int64_t>::max();
};

struct DecisionOptions {
  bool allow_single = true;
  bool allow_compound = true;
  bool enable_average = true;
  bool enable_cwp = true;
  bool enable_wedge = true;
  bool enable_diffwtd = true;
  bool enable_imp_mask = true;
  // Sequence-level tools the decision must respect.
  bool seq_enable_imp_msk_bld = false;
  bool num_same_ref_compound = false;
  // cost = sse + lambda * bits
  double lambda = 0.0;
};

enum class DecisionKind : std::uint8_t { kNone, kSingle, kCompound };

struct DecisionResult {
  DecisionKind kind = DecisionKind::kNone;
  int ref0 = -1;
  int ref1 = -1;
  mc::Mv mv[2] = {};
  // Single-reference fields (mode: 0 GLOBALMV, 1 NEARMV, 2 NEWMV).
  int single_mode = 0;
  int drl_index = 0;
  CompoundModeParams compound;
  std::int64_t sse = std::numeric_limits<std::int64_t>::max();
  int rate_bits = 0;
  double cost = std::numeric_limits<double>::infinity();
};

struct DecisionInput {
  // References used to build predictions when a candidate does not carry one.
  const mc::RefFrame* refs[2] = {nullptr, nullptr};
  int bsize = 0;
  mc::BlockRect luma_rect{};
  int mi_cols = 0;
  int mi_rows = 0;
  const std::uint8_t* source = nullptr;
  int source_stride = 0;
  const RefFrameMap* map = nullptr;
  std::vector<CompoundCandidateInput> compound_candidates;
  std::vector<SingleCandidateInput> single_candidates;
  DecisionOptions options;
};

// Evaluates every enabled compound mode for one prediction and reports the
// measured SSE.  Used by tests and by the decision itself.
struct EvaluatedMode {
  CompoundModeParams params;
  std::int64_t sse = std::numeric_limits<std::int64_t>::max();
};

void evaluate_compound_modes(const CompoundPrediction& prediction,
                             const RefFrameMap* map, int ref0, int ref1,
                             const std::uint8_t* source, int source_stride,
                             const DecisionOptions& options,
                             std::vector<EvaluatedMode>& out);

// Main decision entry point.
bool decide_compound(const DecisionInput& input, DecisionResult& result,
                     std::string& error);

}  // namespace av2enc::m2::multiref
