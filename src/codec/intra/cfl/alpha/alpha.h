#ifndef AV2ENC_ENCODER_CFL_ALPHA_ALPHA_H_
#define AV2ENC_ENCODER_CFL_ALPHA_ALPHA_H_

// Explicit CfL alpha path (AV2 v1.0.0 §5.20.7.32, §6.19.7.25, §7.13.5).
//
// The shared CfL core (../cfl.h) implements the §7.13.5 prediction and a
// per-plane exhaustive SSE search. This component adds the *bitstream* side of
// CFL_EXPLICIT and the encoder decision between it and CFL_DERIVED_ALPHA:
//
//   * quantisation of an arbitrary Q8 slope to the signalled 1..8 eighths;
//   * joint sign combination search (cfl_alpha_signs 0..7, never (zero,zero));
//   * exact Signalling fields cfl_index / cfl_alpha_signs / cfl_alpha_u/v;
//   * exhaustive explicit search and derived-vs-explicit selection by SSE or
//     RD, with a documented deterministic tie-break;
//   * exact writers/readers and pinned default CDFs in syntax.h.
//
// Prediction itself is not reimplemented: apply_candidate() calls the core's
// apply_prediction(), so CPU and CUDA (src/cuda/cfl) stay bit-exact.
//
// Rate model
// ----------
// `rate_q8` is in 1/256 bit units and is the exact §8.2 model cost of the CfL
// alpha syntax under the caller's current CDF copy (the preceding is_cfl
// symbol is common to every candidate and is excluded). `lambda_q8` is
// lambda * 256, and
//
//   cost_q16 = sse * 65536 + lambda_q8 * rate_q8 = 65536 * (sse + lambda*bits)
//
// so the comparison is exact integer arithmetic. With `use_rate == false` the
// selection is SSE-only.

#include "syntax.h"

#include "../cfl.h"

#include <cstdint>
#include <optional>

namespace av2enc::cfl::alpha {

// Number of signallable values for one plane: zero, +1..+8, -1..-8.
inline constexpr int kPerPlaneAlphas = 1 + 2 * kAlphabetSize;  // 17

// Canonical enumeration order, matching the core's exhaustive search so a
// tie keeps the same candidate: 0 = zero, 1..8 = +1..+8, 9..16 = -1..-8.
constexpr Alpha per_plane_alpha(int index) {
  if (index <= 0) {
    return Alpha{kSignZero, 0};
  }
  if (index <= kAlphabetSize) {
    return Alpha{kSignPos, static_cast<std::int8_t>(index)};
  }
  return Alpha{kSignNeg,
               static_cast<std::int8_t>(index - kAlphabetSize)};
}

// Inverse of per_plane_alpha(). Returns -1 for a non-signallable Alpha.
constexpr int per_plane_index(Alpha a) {
  if (a.sign == kSignZero) {
    return a.magnitude == 0 ? 0 : -1;
  }
  if (!alpha_signalable(a)) {
    return -1;
  }
  return a.sign == kSignPos ? static_cast<int>(a.magnitude)
                            : kAlphabetSize + static_cast<int>(a.magnitude);
}

// §5.20.7.32: signU = (signs + 1) / 3, signV = (signs + 1) % 3.
constexpr int joint_sign_u(int alpha_signs) {
  return (alpha_signs + 1) / 3;
}
constexpr int joint_sign_v(int alpha_signs) {
  return (alpha_signs + 1) % 3;
}

// Canonical pair for one joint-sign symbol and two magnitudes. `magnitude_u/v`
// are the signalled 1..8 values (CflAlpha), not the 0..7 symbols. A zero sign
// forces its magnitude to zero.
constexpr AlphaPair pair_from_joint(int alpha_signs, int magnitude_u,
                                    int magnitude_v) {
  const int sign_u = joint_sign_u(alpha_signs);
  const int sign_v = joint_sign_v(alpha_signs);
  AlphaPair p{};
  p.u.sign = static_cast<std::int8_t>(sign_u);
  p.u.magnitude =
      static_cast<std::int8_t>(sign_u == kSignZero ? 0 : magnitude_u);
  p.v.sign = static_cast<std::int8_t>(sign_v);
  p.v.magnitude =
      static_cast<std::int8_t>(sign_v == kSignZero ? 0 : magnitude_v);
  return p;
}

// Nearest signallable explicit alpha for an arbitrary Q8 slope. Ties (exactly
// halfway between two magnitudes) round away from zero; magnitudes clamp to
// 1..8 and a zero slope maps to CFL_SIGN_ZERO.
constexpr Alpha quantise_q8(int q8) { return snap_alpha_q8(q8); }

// Quantise a pair of Q8 slopes. If both quantise to zero the pair is not
// signallable; the plane with the larger magnitude keeps its sign and is
// forced to magnitude 1 (U on an exact tie, U positive when both are zero).
AlphaPair quantise_pair(int q8_u, int q8_v);

// Quantise the two planes' §7.13.5 derived slopes (implicitAlpha, Q8) to the
// nearest signallable pair. Cheap explicit candidate without reading the
// source; `scratch` must already hold compute_luma() for this geometry.
AlphaPair quantise_derived_pair(const Block& u, const Block& v,
                                const Scratch& scratch);

// Canonical Signalling for an explicit pair. Returns a Signalling with
// cfl_index == -1 when `p` is not signalable.
Signalling signalling_for(AlphaPair p);

// Strict inverse of signalling_for(): validates cfl_index, the sign symbol,
// the per-plane symbol ranges, and the (zero,zero) prohibition.
std::optional<AlphaPair> pair_from_signalling_checked(const Signalling& s);

struct CostModel {
  bool use_rate = false;
  std::int64_t lambda_q8 = 0;  // lambda * 256

  std::int64_t cost_q16(std::int64_t sse, int rate_q8) const {
    const std::int64_t base = sse * 65536;
    if (!use_rate) {
      return base;
    }
    return base + lambda_q8 * rate_q8;
  }
};

enum class TieBreak {
  // Keep the existing derived-alpha stream when cost and SSE tie.
  kPreferDerived,
  // Prefer explicit when cost and SSE tie (e.g. to exercise the new path).
  kPreferExplicit,
};

struct Config {
  CostModel cost{};
  bool allow_derived = true;
  bool allow_explicit = true;
  TieBreak tie_break = TieBreak::kPreferDerived;
};

// One evaluated CfL alpha mode.
struct Candidate {
  bool explicit_alpha = false;
  Signalling syntax{};
  AlphaPair explicit_pair{};  // meaningful only when explicit_alpha
  int q8_u = 0;               // exact alpha applied to U (Q8)
  int q8_v = 0;               // exact alpha applied to V (Q8)
  std::int64_t sse_u = 0;
  std::int64_t sse_v = 0;
  int rate_q8 = 0;  // CfL alpha syntax model cost, 1/256 bit

  std::int64_t total_sse() const { return sse_u + sse_v; }
  std::int64_t cost_q16(const CostModel& model) const {
    return model.cost_q16(total_sse(), rate_q8);
  }
};

struct Decision {
  Candidate chosen{};
  Candidate derived{};
  Candidate explicit_alpha{};
  bool has_derived = false;
  bool has_explicit = false;
};

// Evaluates one Signalling against the source planes. For explicit syntax the
// pair is applied; for derived syntax the per-plane implicit alpha is used
// exactly (not quantised). Returns nullopt for an invalid/non-signalable
// Signalling or invalid geometry/sources. `scratch` is filled by the call.
std::optional<Candidate> evaluate_signalling(const Signalling& syntax,
                                             const Block& u, const Block& v,
                                             const SourcePlane& source_u,
                                             const SourcePlane& source_v,
                                             const CflCdfs& cdfs,
                                             Scratch& scratch);

// Exhaustive explicit search: every joint sign combination (8) and every
// magnitude 1..8 for each non-zero sign. `config.allow_derived` is ignored.
// Returns nullopt for invalid input or missing source planes. Ties within a
// sign keep the lower magnitude; ties across signs keep the lower
// cfl_alpha_signs.
std::optional<Candidate> search_explicit(const Block& u, const Block& v,
                                         const SourcePlane& source_u,
                                         const SourcePlane& source_v,
                                         const Config& config,
                                         const CflCdfs& cdfs,
                                         Scratch& scratch);

// Derived-vs-explicit selection. Requires both source planes (this is an
// encoder RD/SSE decision). Deterministic tie-break:
//   1. lower cost_q16
//   2. lower total SSE
//   3. Config::tie_break (derived by default)
//   4. lower cfl_alpha_signs, then alpha_u, then alpha_v (explicit only)
std::optional<Decision> decide_cfl_alpha(const Block& u, const Block& v,
                                         const SourcePlane& source_u,
                                         const SourcePlane& source_v,
                                         const Config& config,
                                         const CflCdfs& cdfs,
                                         Scratch& scratch);

// Applies the chosen candidate to both chroma planes with the shared core.
// `scratch` must come from evaluate_signalling/search_explicit/decide (same
// geometry). Derived candidates apply their exact per-plane Q8 alpha.
void apply_candidate(const Block& u, const Block& v, const Candidate& candidate,
                     const Scratch& scratch);

}  // namespace av2enc::cfl::alpha

#endif  // AV2ENC_ENCODER_CFL_ALPHA_ALPHA_H_
