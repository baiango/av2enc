#ifndef AV2ENC_VIDEO_ADVANCED_DECISION_H_
#define AV2ENC_VIDEO_ADVANCED_DECISION_H_

// Encoder-side decisions for the AV2 advanced inter tools owned by the M2
// advanced layer.  The decisions produce exactly the syntax values that
// advanced_syntax.h writes; they do not write bits themselves.
//
// Rate is deliberately absent: src/encoder/rdo/ owns lambda.  The searches here
// return SSE so the RDO layer can add its rate term and compare against SIMPLE /
// compound / intra candidates.
//
// Reused components (never duplicated):
//   * src/encoder/inter/advanced/warp.*      warp prediction, estimation,
//                                            reduce_warp_model, setup_shear
//   * src/encoder/inter/advanced/compound.*  inter-intra search and masks
//   * src/encoder/inter/advanced/bawp.*      fitted / explicit BAWP and chroma

#include "codec/inter/advanced/bawp.h"
#include "codec/inter/advanced/compound.h"
#include "codec/inter/advanced/warp.h"
#include "codec/inter/advanced/video/advanced_syntax.h"

#include <cstdint>
#include <limits>
#include <string>

namespace av2enc::video::adv {

// ---------------------------------------------------------------------------
// Warped motion / affine
// ---------------------------------------------------------------------------

// `wm_params` is the six-parameter Q16 affine model used by the warp tools
// (§7.13.3.19–.24).  Default_Warp_Params is the identity model.
using WmParams = ia::WarpParams;
inline WmParams default_wm_params() { return WmParams{}; }
inline WmParams reduce_wm_params(const WmParams& params) {
  return ia::reduce_warp_model(params);
}

// The model the decoder actually uses for WARPMV: read_warp_delta() reduces the
// candidate model and re-derives its translation from the assigned block MV.
// `block_mv` must be the final MV (1/8 luma samples) after assign_mv().
WmParams warpmv_local_model(const WmParams& candidate, int mid_x, int mid_y,
                            const int block_mv[2]);

// Width of one signalled warp-delta step for the two precisions.
inline constexpr int kWarpDeltaStepStandard = 1 << (kWarpDeltaStepBits + 1);
inline constexpr int kWarpDeltaStepHigh = 1 << kWarpDeltaStepBits;

// Signed parameter value represented by one read_warp_delta_param() result.
int warp_delta_value(const WarpDeltaSymbols& symbols, int index);

// `apply_warp_delta` is the exact read_warp_delta() + reduce + translation
// sequence.  `base` is WarpParamStack[RefWarpIdx]; when `six_param` is false,
// parameters 4/5 are re-derived from 2/3 as the syntax requires.
ia::WarpParams apply_warp_delta(const ia::WarpParams& base,
                                const WarpDeltaSymbols& symbols,
                                bool six_param, int mid_x, int mid_y,
                                const int block_mv[2]);

// Quantises `target - base` for one precision.  `model` is what the decoder
// will reconstruct (including reduce_warp_model/get_warp_translation).
struct WarpDeltaQuantization {
  WarpDeltaSymbols symbols{};
  ia::WarpParams model{};
  std::int64_t param_error = 0;  // squared error over parameters 2..5
};

WarpDeltaQuantization quantize_warp_delta(const ia::WarpParams& base,
                                          const ia::WarpParams& target,
                                          bool high_precision, bool six_param,
                                          int mid_x, int mid_y,
                                          const int block_mv[2]);

struct WarpDeltaChoice {
  bool high_precision = false;
  WarpDeltaQuantization quantization{};
};

// Standard precision wins ties: it costs fewer symbols.
WarpDeltaChoice choose_warp_delta_precision(const ia::WarpParams& base,
                                            const ia::WarpParams& target,
                                            bool six_param, int mid_x, int mid_y,
                                            const int block_mv[2]);

struct WarpSearchInput {
  // WarpParamStack[0..3] from the decoder's §7.12.2 warp candidate scan.
  const ia::WarpParams* candidates = nullptr;
  int num_candidates = 0;  // 0..MAX_WARP_REF_CANDIDATES

  // Optional local/global model to encode as a delta from a candidate.
  const ia::WarpParams* target = nullptr;

  ia::PlaneView ref{};
  ia::WarpBlockGeometry geom{};
  bool compound = false;
  const std::uint8_t* source = nullptr;
  int source_stride = 0;
  std::int16_t* scratch = nullptr;  // block-local, at least w*h
  int scratch_stride = 0;

  bool enable_six_param = false;
  int mid_x = 0;
  int mid_y = 0;
  const int* block_mv = nullptr;  // [2], 1/8 luma samples
};

struct WarpCodingDecision {
  bool use_warp = false;
  bool warpmv = false;  // true = WARPMV (candidate model, no delta)
  int warp_idx = 3;     // RefWarpIdx 0..3
  bool six_param = false;
  bool high_precision = false;
  WarpDeltaSymbols delta{};
  ia::WarpParams model{};
  std::int64_t sse = std::numeric_limits<std::int64_t>::max();
};

// Exhaustive SSE search over the candidate models and, when `target` is set,
// over the quantised deltas (standard/high precision, 4/6 parameters).
// `use_warp` is left true whenever any candidate evaluated; the caller compares
// `sse` with its SIMPLE-mode SSE before committing.
WarpCodingDecision select_warp_coding(const WarpSearchInput& in);

// True when the decoder's §7.13.3.14 fallback forces the extended warp pass.
bool warp_requires_extended(const ia::WarpParams& params,
                            const ia::WarpBlockGeometry& geom);

// ---------------------------------------------------------------------------
// Inter-intra
// ---------------------------------------------------------------------------

struct InterIntraCodingInput {
  ia::InterIntraSearchInput search{};
  bool is_warp = false;   // YMode == WARPMV -> read_interintra_mode(1)
  bool warp_inter_intra_allowed = true;
  std::int64_t inter_sse = std::numeric_limits<std::int64_t>::max();
};

struct InterIntraCodingDecision {
  bool enabled = false;
  int interintra_mode = 0;  // 0..3
  bool wedge_interintra = false;
  int wedge_index = 0;
  int wedge_sign = 0;  // read_interintra_mode() forces 0
  std::int64_t sse = std::numeric_limits<std::int64_t>::max();
};

// Fills the decision fields of MotionModeSymbols.  The caller still sets
// `extend_allowed`, `local_allowed`, `inter_intra_allowed` and the neighbour
// contexts.
void fill_inter_intra_symbols(const InterIntraCodingDecision& decision,
                              MotionModeSymbols* symbols);

// Wraps search_inter_intra() and maps the result to syntax values.  The
// decision is enabled only when the best inter-intra SSE beats `inter_sse`.
InterIntraCodingDecision decide_inter_intra(const InterIntraCodingInput& in);

// §7.13.3.30 inter-intra weights (0..64) for one plane-sized block.
void build_inter_intra_weights(int mode, int w, int h, std::uint8_t* out,
                               int stride);

// §7.13.3.27 wedge weights (0..64).
bool build_wedge_weights(int bsize, int wedge_sign, int wedge_index,
                         std::uint8_t* out, int stride);

// ---------------------------------------------------------------------------
// BAWP
// ---------------------------------------------------------------------------

struct BawpDecisionInput {
  ia::BawpJob job{};
  const std::uint8_t* source = nullptr;  // luma source block
  int source_stride = 0;
  // Scratch destination in the same coordinate system as `job.dst`; the
  // decision copies the prediction into it before each candidate.
  std::uint8_t* scratch = nullptr;
  int scratch_stride = 0;

  bool allow_bawp = true;      // frame-level allow_bawp
  bool ref_scaled = false;     // §5.20.7.6 !is_scaled(RefFrame[0], 1)
  bool frame_is_switch = false;
  int y_mode = static_cast<int>(YMode::kNearMv);
  bool use_amvd = false;
  int order_hint = 0;
  int ref_order_hint = 0;

  // Optional chroma evaluation.  `chroma_job.cur` must already contain the
  // chroma inter prediction; its `dst` is overwritten by the decision.
  const ia::BawpJob* chroma_job = nullptr;
  const std::uint8_t* chroma_source = nullptr;
  int chroma_source_stride = 0;
  std::uint8_t* chroma_scratch = nullptr;
  int chroma_scratch_stride = 0;
  bool has_chroma = false;
};

struct BawpDecision {
  bool use_bawp = false;
  bool explicit_bawp = false;
  bool explicit_bawp_scale = false;
  bool use_bawp_chroma = false;
  int explicit_ctx = 2;
  int list_index = 2;
  int first_ref_dist = 0;
  int luma_alpha = 256;    // BawpAlpha
  int chroma_alpha = 256;  // plane > 0 uses BawpAlpha (or 256 when count == 0)
  std::int64_t sse_off = 0;
  std::int64_t sse_best = 0;
};

// Evaluates {off, fitted, explicit sign 0, explicit sign 1} and, when a chroma
// job is supplied, whether chroma BAWP is worth signalling.
BawpDecision decide_bawp(const BawpDecisionInput& in);

// Copies the signalled flags/contexts into the syntax struct.
BawpSymbols bawp_symbols_from_decision(const BawpDecision& decision);

// Applies the decision to the luma job (explicit flags + list index + distance)
// or to a chroma job (`bawp_alpha = decision.luma_alpha`; chroma always uses the
// luma BawpAlpha and the fitted beta).
void apply_bawp_decision_to_job(const BawpDecision& decision, bool chroma,
                                ia::BawpJob* job);

// §7.13.3.25 explicit-alpha rule, exported for tests.
int explicit_bawp_alpha(int list_index, int first_ref_dist,
                        bool explicit_bawp_scale);

// §7.13.3.25 chroma rule: plane > 0 uses BawpAlpha; a zero-sample template
// forces alpha = 1 << 8.
int bawp_alpha_for_plane(int plane, int luma_alpha, int count);

}  // namespace av2enc::video::adv

#endif  // AV2ENC_VIDEO_ADVANCED_DECISION_H_
