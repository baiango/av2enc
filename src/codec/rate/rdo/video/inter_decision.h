// Inter candidate cost and deterministic selection for the M2 P-frame path.
//
// Candidate cost:
//
//     cost = distortion + lambda * (mode + ref + MV + residual) bits
//
// `score_inter_candidate()` obtains the residual term from
// `residual::decide_inter_block()` (the writer's own symbol cost) and the
// mode/MV/ref term from `InterRateModel`.  `decide_inter_candidates()` feeds
// the results to `av2enc::rdo::CandidateSelector`, so tie-breaking is the
// shared, documented total order: cost, then distortion/rate per the config,
// then CandidateKind, id, aux, and finally generation order.
//
// Candidate order used by the integration:
//   skip GLOBALMV  (kind 0)
//   skip NEARMV    (kind 1)
//   skip NEWMV     (kind 2)
//   intra fallback (kind 3)
// Residual variants keep the mode kind and set `skip_flag = false`; `aux`
// orders skip before residual for the same DRL index.
//
// The layer is pure: it takes predictions and source-plane views, returns a
// decision, and never writes a bitstream or mutates a reference frame.

#pragma once

#include "codec/inter/mc/mc.h"
#include "codec/rate/rdo/video/config.h"
#include "codec/rate/rdo/video/rate_model.h"
#include "codec/transform/inter_residual/inter_residual.h"

#include <cstdint>
#include <span>
#include <string>

namespace av2enc::video::rdo {

enum class InterCandidateKind : std::uint8_t {
  kSkipGlobal = 0,  // GLOBALMV, the zero-MV skip candidate
  kMerge = 1,       // NEARMV, the spatial-stack merge candidate
  kNewMv = 2,       // NEWMV, ME + signalled MVD
  kIntra = 3,       // intra fallback
};

struct InterCandidateSpec {
  InterCandidateKind kind = InterCandidateKind::kSkipGlobal;
  int inter_mode = 0;  // 0 GLOBALMV, 1 NEARMV, 2 NEWMV
  int drl_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  int ref = 0;
  bool force_residual = false;
  // Optional pre-selection distortion (for example SAD or SATD).  When unset,
  // the distortion comes from the residual decision (total Y+U+V SSE).
  bool has_distortion_override = false;
  double distortion_override = 0.0;
};

struct InterCandidate {
  InterCandidateKind kind = InterCandidateKind::kSkipGlobal;
  int inter_mode = 0;
  int drl_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  int ref = 0;
  bool skip_flag = true;
  double distortion = 0.0;
  double mode_bits = 0.0;
  double residual_bits = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  bool valid = false;
};

struct ScoredInterCandidate {
  InterCandidate candidate;
  residual::BlockDecision residual;
  bool has_residual = false;
  bool valid = false;
};

// Evaluates one inter candidate: residual skip decision, mode/MV/ref rate, and
// the Lagrangian cost.  `evaluate_residual=false` still measures the skip
// distortion (Y+U+V SSE) and skip_flag cost through the shared residual path,
// so every candidate uses the same distortion units.
bool score_inter_candidate(const InterCandidateSpec& spec,
                           const residual::BlockInput& block,
                           bool evaluate_residual,
                           const RateContext& rate_context,
                           const RdoConfig& config,
                           const InterRateModel& model,
                           ScoredInterCandidate& out,
                           std::string* error = nullptr);

// Intra fallback candidate.  `coefficient_bits` comes from the M1
// TransformStreamRate adapter (`intra_coefficient_bits()`), so this layer does
// not reimplement coefficient rate.
InterCandidate make_intra_candidate(double distortion, double coefficient_bits,
                                    int y_mode, int y_angle_delta, int uv_mode,
                                    const RateContext& rate_context,
                                    const RdoConfig& config,
                                    const InterRateModel& model);

struct InterDecision {
  bool found = false;
  int best_index = -1;  // index into the input span
  InterCandidate candidate;
  double cost = 0.0;
  double margin = 0.0;  // best-to-second cost gap, 0 with < 2 candidates
  int candidate_count = 0;
};

// Deterministic selection over a fixed candidate list.  Invalid candidates
// are ignored; if the list is empty or the config is invalid, `found` is
// false and `error` (when supplied) is set.  At most 16 valid candidates are
// accepted (the M2 path generates at most four inter + one intra).
InterDecision decide_inter_candidates(
    std::span<const InterCandidate> candidates, const RdoConfig& config,
    std::string* error = nullptr);

}  // namespace av2enc::video::rdo
