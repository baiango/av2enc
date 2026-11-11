// AV2 §7.13.3 inter motion compensation — CPU reference API.
//
// Scope (M2 P-frame baseline, docs/tool-roadmap.md §5):
//   * single-reference translational MC, 8-bit 4:2:0
//   * normative 8-tap / 4-tap switchable subpel interpolation
//   * skip / merge candidate selection (spatial MV stack subset)
//   * NEWMV / NEARMV motion-vector prediction hooks
//   * reference-frame inputs and partial (edge) blocks
//
// Explicitly out of scope: compound prediction, B frames, global/local warp,
// OBMC, inter-intra, BAWP, TIP, OPFL, refine-MV, temporal MV projection, the
// ref-MV bank, and any bitstream syntax.  Motion estimation is a separate
// component (src/encoder/inter/me/); this module consumes its MVs through
// ModeDecision::me_mv / me_absolute.
//
// This is a correctness oracle, not a performance target (docs/architecture.md).
// The CUDA counterpart lives in src/cuda/inter/mc/ and must match bit-exactly.

#ifndef AV2ENC_ENCODER_INTER_MC_MC_H_
#define AV2ENC_ENCODER_INTER_MC_MC_H_

#include <cstdint>

#include "encoder/inter/mc/mc_core.h"
#include "encoder/inter/mc/mc_types.h"

namespace av2enc::inter::mc {

enum class McStatus : std::int32_t {
  kOk = 0,
  kInvalidArgument,
  kUnsupportedSubsampling,
  kBlockTooLarge,
};

const char* mc_status_string(McStatus status);

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

// Predicts one plane of a block.
//
// `rect` is in plane samples and must lie inside the MI-aligned reference
// plane (RefMiCols*4 >> sub_x by RefMiRows*4 >> sub_y).  Edge blocks pass their
// clipped size; the function never reads or writes outside that rectangle.
// `mv` is the final block MV in 1/8 luma samples, identical for all samples
// (SPLITMV/OBMC are out of scope).
//
// `dst` is a uint8 plane with `dst_stride >= rect.w`; samples are written with
// the normative Clip1(Round2(..., InterRound1)).
McStatus predict_plane(const RefFrame& ref, int plane, const BlockRect& rect,
                       Mv mv, InterpFilter filter, std::uint8_t* dst,
                       int dst_stride);

// Convenience wrapper for 4:2:0: predicts Y, U and V for a luma block.
// `luma_rect` must be even in x/y/w/h (the M2 partition grid satisfies this).
// Chroma destinations are half the luma width/height.
McStatus predict_block_420(const RefFrame& ref, const BlockRect& luma_rect,
                           Mv mv, InterpFilter filter, MutablePlane dst[3]);

// ---------------------------------------------------------------------------
// Motion-vector prediction (AV2 §7.12.2, spatial subset)
// ---------------------------------------------------------------------------

// The ME component (src/encoder/inter/me/) reports 1/16-luma-sample results.
// AV2 signals at most 1/8-luma-sample precision (MvPrecision =
// MV_PRECISION_EIGHTH_PEL for the M2 baseline), so quantise before resolving a
// mode.  This is an encoder-side choice, not a normative process: Round2Signed
// rounds ties away from zero.
inline Mv mv_from_me_1_16(std::int32_t mv_y, std::int32_t mv_x) {
  return Mv{mc_round2_signed(mv_y, 1), mc_round2_signed(mv_x, 1)};
}

// DrlReorder (§6.17.2).  The default M2 setting is kConstraint.
enum class DrlReorder : std::uint8_t {
  kNever = 0,
  kConstraint = 1,
  kAlways = 2,
};

struct MvStackParams {
  // Current block, in luma MI units.
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_h4 = 16;  // superblock height in MI units (64x64 -> 16)
  int mi_cols = 0;
  int mi_rows = 0;
  // Row-major mi_cols x mi_rows grid.  Entries outside the coded area have
  // ref[] == kRefNone and are skipped by the scan.
  const MiInfo* mi = nullptr;
  std::int8_t ref_frame = kRefLast;
  // GlobalMvs[0] (identity = 0,0).  Encoders without global motion pass 0.
  Mv global_mv{};
  bool use_intrabc = false;
  DrlReorder reorder = DrlReorder::kConstraint;
};

struct StackCandidate {
  Mv mv[2];  // list 0 / list 1 (single-ref: list 1 mirrors list 0)
  int weight = 0;
  int row_offset = 0;  // candidate MI row offset from the current block
  int col_offset = 0;
};

struct MvStack {
  StackCandidate cand[kMaxRefMvStackSize];
  int count = 0;
  int num_nearest = 0;  // §7.12.2 step 14 boundary between NEAR and others
};

// Builds the spatial MV stack for a single-reference block.  Implemented:
// global-MV setup, spatial scan points (§7.12.2 steps 7-13), the col scan
// (step 15), weight sorting (steps 16-17) and the extra-search / MVP
// combination (steps 20-22).  Temporal projection, the derived-SMVP stack,
// the ref-MV bank and warp candidates are not implemented; their slots are
// simply absent, as for a frame with those tools disabled.
McStatus build_mv_stack(const MvStackParams& params, MvStack* stack);

// §5.20.9.4 / §5.20.9.5 picture-bound clipping for a block MV.
Mv clamp_mv_to_picture(Mv mv, const MvStackParams& params);

enum class InterMode : std::uint8_t {
  kSkipMerge = 0,  // skip / merge: MV = RefStackMv[drl_index][0], no MVD
  kNearmv = 1,     // NEARMV: MV = RefStackMv[drl_index][0]
  kNewmv = 2,      // NEWMV:   MV = RefStackMv[drl_index][0] + MVD
};

struct ModeDecision {
  InterMode mode = InterMode::kSkipMerge;
  int drl_index = 0;
  Mv mvd{};
  // NEWMV hook for the ME component: when me_absolute is true the input MV is
  // treated as the desired absolute MV and the MVD is derived as
  // me_mv - pred_mv.  Otherwise `mvd` is used directly.
  Mv me_mv{};
  bool me_absolute = false;
  std::int8_t ref_frame = kRefLast;
  InterpFilter filter = InterpFilter::kEighttap;
};

struct ResolvedMode {
  Mv mv{};
  Mv pred_mv{};
  Mv mvd{};
  int drl_index = 0;
  std::int8_t ref_frame = kRefLast;
  InterpFilter filter = InterpFilter::kEighttap;
};

// Resolves the final MV for a mode decision.  This is the hook the encoder
// calls after the ME/RDO stage has chosen between SKIP/NEARMV/NEWMV.  The
// result is what the bitstream must signal: for skip/merge and NEARMV the DRL
// index, for NEWMV the MVD.
McStatus resolve_mode(const MvStack& stack, const MvStackParams& params,
                      const ModeDecision& decision, ResolvedMode* out);

// ---------------------------------------------------------------------------
// Encoder-side candidate selection (not normative)
// ---------------------------------------------------------------------------

struct MergeScore {
  int drl_index = -1;
  std::int64_t sad = 0;
};

// Predicts the luma block for every stack candidate and returns the one with
// the lowest SAD against `src`.  This is an encoder decision: the chosen index
// must still be signalled as a NEARMV/NEAR_NEARMV DRL index (index 0 is the
// normative skip/merge default).  `scratch` must hold luma_rect.w *
// luma_rect.h samples.
McStatus select_merge_candidate(const MvStack& stack, const RefFrame& ref,
                                std::int8_t ref_frame,
                                const BlockRect& luma_rect,
                                InterpFilter filter, const std::uint8_t* src,
                                int src_stride, std::uint8_t* scratch,
                                int scratch_stride, MergeScore* out);

// Sum of absolute differences between two uint8 blocks.  Exposed for tests and
// for encoders that want to reuse the merge scoring.
std::int64_t sad_block(const std::uint8_t* a, int a_stride,
                       const std::uint8_t* b, int b_stride, int w, int h);

}  // namespace av2enc::inter::mc

#endif  // AV2ENC_ENCODER_INTER_MC_MC_H_
