#ifndef AV2ENC_ENCODER_CFL_H_
#define AV2ENC_ENCODER_CFL_H_

// CPU reference implementation of AV2 v1.0.0 §7.13.5 chroma-from-luma.
//
// The CPU path is the correctness oracle for src/cuda/cfl/. Both paths share
// the integer arithmetic in cfl_common.h, so "CPU vs GPU" is a launch/thread
// mapping test, not a second spec transcription.
//
// Usage sketch (intra/RDO integration):
//
//   cfl::Block u = ...;   // U plane, chroma already holds DC prediction
//   cfl::Block v = ...;   // V plane, same geometry
//   cfl::Scratch scratch;
//   cfl::search_alpha(u, v, src_u, src_v, cfl::SearchMode::kExhaustiveSse,
//                     scratch, result);
//   cfl::apply_prediction(u, cfl::alpha_q8(result.alpha.u), scratch);
//   cfl::apply_prediction(v, cfl::alpha_q8(result.alpha.v), scratch);
//
// or, for the derived-alpha mode the bitstream signals as
// cfl_index == CFL_DERIVED_ALPHA:
//
//   cfl::compute_luma(u, scratch);
//   int alpha = cfl::derive_implicit_alpha(u, scratch);
//   cfl::apply_prediction(u, alpha, scratch);
//
// Nothing here writes bits or touches CDFs. `explicit_signalling()` returns
// the exact symbol values the entropy pass must code later.

#include "cfl_common.h"

#include <cstdint>
#include <optional>

namespace av2enc::cfl {

// Per-block working set. `luma` holds L in the layout L[i * w + j] with three
// fractional bits (§7.13.5). Reuse one Scratch across blocks; the encoder path
// must not allocate per block.
struct Scratch {
  int luma_avg = 0;  // three fractional bits; 8 << (BitDepth - 1) when empty
  int luma_above[kMaxDim]{};
  int luma_left[kMaxDim]{};
  int luma[kMaxDim * kMaxDim]{};
};

// Source chroma used by the encoder alpha search (not needed for prediction).
struct SourcePlane {
  const std::uint8_t* plane = nullptr;
  int stride = 0;
  int max_x = 0;  // inclusive
  int max_y = 0;
};

enum class SearchMode {
  // Use the §7.13.5 derived-alpha correlation on the above/left neighbours,
  // then snap to the nearest explicit alpha. Cheap; no source reads.
  kNeighbourLeastSquares,
  // Exhaustive SSE over the 17 signallable alpha values per plane, using the
  // source chroma. Encoder-only decision; the bitstream is unchanged.
  kExhaustiveSse,
};

struct SearchResult {
  AlphaPair alpha{};
  Signalling syntax{};
  std::int64_t sse_u = 0;
  std::int64_t sse_v = 0;
  std::int64_t total_sse() const { return sse_u + sse_v; }
};

// Structural validation. Returns false for a descriptor the reference does not
// handle (null plane, block larger than 64x64, bad bit depth / subsampling).
bool valid(const Block& block);

// §7.13.5: fills scratch.luma_avg, scratch.luma_above[0..w),
// scratch.luma_left[0..h) and scratch.luma[i*w+j]. Call once per block; both
// planes share the result.
void compute_luma(const Block& block, Scratch& scratch);

// §7.13.5 CFL_DERIVED_ALPHA: implicitAlpha in Q8 (0..511). Uses the chroma
// above/left reconstructed neighbours, so `block.chroma` must be valid.
// Call compute_luma() first.
int derive_implicit_alpha(const Block& block, const Scratch& scratch);

// Applies §7.13.5 to one plane:
//   pred = Clip1(dc + Round2Signed(alpha * (L - lumaAvg), 11))
// `block.chroma` must already contain the DC prediction. Writes all w*h coded
// samples (the decoder reconstructs the full coded block, padding included).
void apply_prediction(const Block& block, int alpha_q8_value,
                      const Scratch& scratch);

// Convenience wrappers returning the Q8 alpha that was applied.
int predict_explicit(const Block& block, Alpha alpha, Scratch& scratch);
int predict_derived(const Block& block, Scratch& scratch);

// Sum of squared error between the CfL prediction for `alpha_q8_value` and the
// source plane. Uses scratch.luma (call compute_luma first).
std::int64_t prediction_sse(const Block& block, int alpha_q8_value,
                            const Scratch& scratch, const SourcePlane& source);

// Encoder alpha decision for both planes. `u` and `v` must describe the same
// luma geometry and share the same `luma` pointer; only their `chroma`
// pointers differ. On return the planes are unchanged. `scratch` is filled by
// the call (compute_luma on the shared geometry).
//
// kNeighbourLeastSquares never reads `source_*`; kExhaustiveSse requires valid
// source planes. Returns std::nullopt for invalid input.
std::optional<SearchResult> search_alpha(const Block& u, const Block& v,
                                         const SourcePlane& source_u,
                                         const SourcePlane& source_v,
                                         SearchMode mode, Scratch& scratch);

// Applies the chosen pair to both planes. `scratch` must have been filled by
// search_alpha (or by compute_luma with the same geometry).
void apply_prediction_pair(const Block& u, const Block& v,
                           const AlphaPair& alpha, const Scratch& scratch);

// §5.20.5.6 cflAllowed derivation for the current chroma residual size.
// `chroma_w`/`chroma_h` are the plane residual dimensions (Block_Width/
// Block_Height of planeSz). Lossless CfL is limited to 4x4.
bool cfl_allowed(int chroma_w, int chroma_h, bool enable_cfl_intra,
                 bool is_chroma_part, bool frame_is_intra,
                 bool cfl_allowed_in_sdp, bool lossless);

}  // namespace av2enc::cfl

#endif  // AV2ENC_ENCODER_CFL_H_
