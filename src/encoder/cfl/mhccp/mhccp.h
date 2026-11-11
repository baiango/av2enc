#ifndef AV2ENC_ENCODER_CFL_MHCCP_H_
#define AV2ENC_ENCODER_CFL_MHCCP_H_

// CPU reference API for AV2 v1.0.0 §7.13.6-§7.13.8 MHCCP.
//
// The arithmetic lives in mhccp_common.h so the CPU reference, the host
// emulation test and the CUDA kernel share one transcription. This header adds
// the encoder-facing pieces: structural validation, source-SSE evaluation,
// direction/parameter search, and a unified explicit / derived / multi
// hypothesis decision built on top of src/encoder/cfl/cfl.h.
//
// Conventions:
//   * `u` and `v` describe the same luma geometry and must expose the same
//     reconstructed luma. Their `chroma` pointers differ.
//   * On input `chroma` holds the DC prediction for the current block. The
//     reconstructed chroma above/left neighbours must already be present; they
//     are the CflRef[1] least-squares targets.
//   * Prediction writes the full coded w*h block in place.

#include "mhccp_common.h"

#include "../cfl.h"

#include <cstdint>
#include <optional>

namespace av2enc::cfl::mhccp {

// True when both descriptors are valid and describe the same luma geometry.
bool valid_pair(const Block& u, const Block& v);

// Sum of squared error of the MHCCP prediction against `source`.
// `refs` must come from capture_refs() on `block`.
std::int64_t prediction_sse(const Block& block, const MultiParams& params,
                            int cfl_mh_dir, const Refs& refs,
                            const SourcePlane& source);

// Apply one plane's model in place.
void apply_mhccp(const Block& block, const MultiParams& params,
                 int cfl_mh_dir, const Refs& refs);

// Apply the U and V models (same direction, per-plane parameters).
void apply_mhccp_pair(const Block& u, const Block& v, int cfl_mh_dir,
                      const MultiParams& params_u, const MultiParams& params_v);

// One evaluated direction.
struct Candidate {
  int mh_dir = 0;
  MultiParams params_u{};
  MultiParams params_v{};
  int count_u = 0;
  int count_v = 0;
  std::int64_t sse_u = 0;
  std::int64_t sse_v = 0;
  std::int64_t total_sse() const { return sse_u + sse_v; }
};

// Evaluate a fixed cfl_mh_dir for both planes.
std::optional<Candidate> evaluate_mhccp(const Block& u, const Block& v,
                                        const SourcePlane& source_u,
                                        const SourcePlane& source_v,
                                        int cfl_mh_dir);

// Try cfl_mh_dir 0..2 and return the lowest U+V SSE. Ties keep the lower
// direction. Returns std::nullopt for invalid input or missing source planes.
std::optional<Candidate> search_mhccp(const Block& u, const Block& v,
                                      const SourcePlane& source_u,
                                      const SourcePlane& source_v);

// Unified encoder hypothesis decision. This is an encoder-only decision: the
// returned syntax values are the exact S() symbols a writer must code.
enum class HypothesisKind : std::uint8_t {
  kExplicit = 0,
  kDerivedAlpha = 1,
  kMulti = 2,
};

struct Decision {
  HypothesisKind kind = HypothesisKind::kDerivedAlpha;
  int cfl_index = kIndexDerivedAlpha;
  int cfl_mh_dir = -1;  // valid for kMulti only
  cfl::AlphaPair alpha{};       // valid for kExplicit
  cfl::Signalling syntax{};     // explicit signalling (cfl_index filled in)
  int alpha_q8_u = 0;           // resolved prediction weight, all kinds
  int alpha_q8_v = 0;
  MultiParams params_u{};
  MultiParams params_v{};
  // Geometry used by kMulti (the same luma, per-plane chroma). The other
  // hypothesis kinds ignore these fields.
  Block multi_u{};
  Block multi_v{};
  std::int64_t sse_u = 0;
  std::int64_t sse_v = 0;
  std::int64_t total_sse() const { return sse_u + sse_v; }
};

struct DecisionOptions {
  bool allow_explicit = true;
  bool allow_derived = true;
  bool allow_multi = true;
  bool enable_cfl_intra = true;
  bool enable_mhccp = true;
  bool is_chroma_part = false;
  bool frame_is_intra = true;
  bool cfl_allowed_in_sdp = false;
  bool lossless = false;
};

// Search explicit (quantised 1/8 alpha), derived, and MHCCP hypotheses and
// return the lowest U+V SSE. Candidate order on an exact tie is explicit,
// derived, multi; strict '<' keeps the earlier candidate.
std::optional<Decision> decide_cfl(const cfl::Block& u, const cfl::Block& v,
                                   const cfl::SourcePlane& source_u,
                                   const cfl::SourcePlane& source_v,
                                   int num4_above_right,
                                   int num4_below_left,
                                   int tile_col_end, int tile_row_end,
                                   const DecisionOptions& options,
                                   cfl::Scratch& scratch);

// Apply a decision to both planes in place.
void apply_decision(const cfl::Block& u, const cfl::Block& v,
                    const Decision& decision, cfl::Scratch& scratch);

// Convert an existing CfL block descriptor into an MHCCP descriptor.
//
// `tile_col_end` / `tile_row_end` are MiColEnd*MI_SIZE / MiRowEnd*MI_SIZE in
// luma samples; pass 0 for a single-tile stream (derive from coded_width /
// coded_height). `chroma_max_x/y` default to the coded chroma grid; callers
// with a padded reconstruction plane should overwrite those two fields after
// conversion.
Block make_block(const cfl::Block& base, int num4_above_right,
                 int num4_below_left, int tile_col_end = 0,
                 int tile_row_end = 0);

}  // namespace av2enc::cfl::mhccp

#endif  // AV2ENC_ENCODER_CFL_MHCCP_H_
