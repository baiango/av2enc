// AV2 v1.0.0 §7.12.2 complete motion-vector prediction stack.
//
// Scope
// -----
// This is the spec-only completion of the M2 MV stack.  It matches
// `av2enc::video::merge::build_mv_stack()` (the wired component) candidate-for-candidate
// whenever the extra branches are inactive, and adds the normative branches
// that component omits:
//
//   * §7.12.2.1/.2  setup global MV (`get_warp_motion_vector`);
//   * §7.12.2.3/.4  generate points from corners / warp corner process;
//   * §7.12.2.5/.6  scan col / scan point, including the §7.12.2.9 add-warp
//                   call and the §7.12.2.11 insert-warp-candidate process;
//   * §7.12.2.10    the complete add-reference-MV dispatch: single, compound,
//                   TIP single-ref, TIP add-derived and the warped call;
//   * §7.12.2.12-.18 search / compound search / compound-add-derived /
//                   derive-ref-MV-from-TIP / single-add-derived (with the
//                   `use_ref_frame_mvs && enable_mv_traj` trajectory branch) /
//                   derive-single-ref-MV-from-TIP / TIP-add-derived;
//   * §7.12.2.19    the normative one-move-to-front sorting process;
//   * §7.12.2.20    the extra search process *exactly*: the pre-clamp of every
//                   candidate with clamp_mv_row/col, the global-MV candidate
//                   (with the `!use_intrabc` guard and PruneCount budget), the
//                   `Block_Width > 32 && Block_Height > 32` MVP-combination
//                   candidates, and the DeriveWrl warp candidates
//                   (warp bank newest-first, `gm_params`, two
//                   Default_Warp_Params) with the MAX_WARP_REF_CANDIDATES cap;
//   * §7.12.2.22    fill MVP from derived SMVP for both the compound (step 18)
//                   and single (step 20) positions;
//   * §7.12.2.23    final clamping process.
//
// Deliberately out of scope: the §7.12.2.21 ref-MV bank (a separate component
// owns it) and anything that changes the §7.9 motion field (the temporal scan
// is injected through `StackParams::temporal_scan` so the ordering is exact).
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).  No encoder code from AVM, SVT-AV1/AV2, clave
// or av2-turbo was consulted.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/advanced/warp_common.h"
#include "codec/inter/mc/mc_types.h"
#include "codec/partition/tree/partition.h"

namespace av2enc::inter::mv_stack_full {

namespace adv = inter::adv;
using inter::mc::Mv;

// ---------------------------------------------------------------------------
// §9.2 constants
// ---------------------------------------------------------------------------
inline constexpr int kMaxRefMvStackSize = 6;   // MAX_REF_MV_STACK_SIZE
inline constexpr int kMaxPrNum = 16;           // MAX_PR_NUM
inline constexpr int kMaxDrStackSize = 4;      // MAX_DR_STACK_SIZE
inline constexpr int kMaxDrPrNum = 2;          // MAX_DR_PR_NUM
inline constexpr int kMaxDrlIndex = kMaxRefMvStackSize - 1;
inline constexpr int kMaxWarpRefCandidates = 4;  // MAX_WARP_REF_CANDIDATES
inline constexpr int kWarpParamBankSize = 4;     // WARP_PARAM_BANK_SIZE
inline constexpr int kRefsPerFrame = 7;          // REFS_PER_FRAME
inline constexpr int kMotionModes = 5;           // MOTION_MODES
inline constexpr int kCwpEqual = 8;              // CWP_EQUAL
inline constexpr int kMiSize = 4;                // MI_SIZE
inline constexpr int kMvBorder = inter::mc::kMvBorder;
inline constexpr int kMvLow = inter::mc::kMvLow;
inline constexpr int kMvUpp = inter::mc::kMvUpp;

// §6.19.7.5 YMode values.
inline constexpr int kYModeNearMv = 14;
inline constexpr int kYModeGlobalMv = 15;
inline constexpr int kYModeNewMv = 16;
inline constexpr int kYModeWarpMv = 17;
inline constexpr int kYModeWarpNewMv = 18;
inline constexpr int kYModeNearNearMv = 19;
inline constexpr int kYModeNearNewMv = 20;
inline constexpr int kYModeNewNearMv = 21;
inline constexpr int kYModeGlobalGlobalMv = 22;
inline constexpr int kYModeNewNewMv = 23;
inline constexpr int kYModeJointNewMv = 24;

// §6.19.7.13 MotionModes values.
inline constexpr int kMotionSimple = 0;
inline constexpr int kMotionInterIntra = 1;
inline constexpr int kMotionLocalWarp = 2;
inline constexpr int kMotionDeltaWarp = 3;
inline constexpr int kMotionExtendWarp = 4;

// RefFrame values.  Inter references are 0..6 (REFS_PER_FRAME), TIP_FRAME is
// 7, INTRA_FRAME is 8 and NONE is -1.  (This is the `av2enc::video::merge` convention,
// deliberately not `mc_types.h`'s 0 == INTRA convention.)
inline constexpr std::int8_t kRefNone = -1;
inline constexpr std::int8_t kTipFrame = 7;
inline constexpr std::int8_t kIntraFrame = 8;

inline constexpr bool is_inter_ref_frame(std::int8_t ref) {
  return ref != kIntraFrame && ref != kRefNone;
}
inline constexpr bool is_tip_ref(std::int8_t ref) { return ref == kTipFrame; }

// §5.18.2 MV_PRECISION_* values (0..6; 6 == eighth pel).
inline constexpr int kMvPrecisionEighthPel = 6;

inline constexpr int kGmIdentity = 0;  // §5.18.9.1 IDENTITY

enum class Reorder : std::uint8_t {
  kDisabled = 0,    // DRL_REORDER_DISABLED
  kConstraint = 1,  // DRL_REORDER_CONSTRAINT
  kAlways = 2,      // DRL_REORDER_ALWAYS
};

enum class Status : std::int32_t {
  kOk = 0,
  kInvalidArgument,
  kInvalidBlockSize,
  kInvalidReference,
  kInvalidStackParams,
  kUnsupportedFeature,
};

const char* status_string(Status status);

// ---------------------------------------------------------------------------
// Mode-info grid
// ---------------------------------------------------------------------------
//
// One cell per 4x4 luma MI.  Every field the normative processes read is
// present, separately from `mc::MiInfo` (which lacks SubMvs, motion modes and
// warp parameters):
//   * `written` models "RefFrames[mvRow][mvCol][0] has been written for this
//     frame"; write it for intra cells too (they carry ref[0] == kIntraFrame).
//   * `mv` is Mvs (block MV), `sub_mv` is SubMvs.
//   * `col_base`/`row_base` are MiColBase/MiRowBase.
//   * `warp[list]` is WarpParams[mvRow][mvCol][list].
//   * `tip_16x16` is TipSizes16x16[mvRow][mvCol].
struct MiCell {
  std::int8_t ref[2] = {kRefNone, kRefNone};
  std::int32_t mv[2][2] = {{0, 0}, {0, 0}};      // [list][comp], 1/8 pel
  std::int32_t sub_mv[2][2] = {{0, 0}, {0, 0}};  // [list][comp], 1/8 pel
  std::int8_t is_inter = 0;
  std::int8_t written = 0;
  std::int8_t w4 = 0;  // Num_4x4_Blocks_Wide[MiSize]
  std::int8_t h4 = 0;
  std::int16_t col_base = 0;
  std::int16_t row_base = 0;
  std::uint8_t y_mode = 0;
  std::uint8_t motion_mode = 0;
  std::int8_t cwp = kCwpEqual;
  std::uint8_t tip_16x16 = 0;  // 0 => 8x8 candidate unit, 1 => 16x16
  adv::WarpParams warp[2]{};
};

struct MiGrid {
  int cols = 0;
  int rows = 0;
  const MiCell* mi = nullptr;  // row-major, rows * cols

  bool inside(int row, int col) const {
    return mi != nullptr && row >= 0 && col >= 0 && row < rows && col < cols;
  }
  const MiCell& at(int row, int col) const {
    return mi[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(cols) +
              static_cast<std::size_t>(col)];
  }
};

// ---------------------------------------------------------------------------
// TIP motion field (§7.10 / §7.12.2.15/.17/.18)
// ---------------------------------------------------------------------------
//
// MotionFieldValid/Mvs/MotionFieldOffset/TrajValid/TrajMv are 8x8 arrays:
// index [y8][x8], component [comp].  `mv`/`traj_mv` are interleaved pairs.
struct TipField {
  int cols8 = 0;
  int rows8 = 0;
  const std::uint8_t* valid = nullptr;      // MotionFieldValid[y8][x8]
  const std::int32_t* mv = nullptr;         // MotionFieldMvs[y8][x8][2]
  const std::int32_t* offset = nullptr;     // MotionFieldOffset[y8][x8]
  const std::uint8_t* traj_valid = nullptr; // TrajValid[ref][y8][x8] per ref
  const std::int32_t* traj_mv = nullptr;    // TrajMv[ref][y8][x8][2] per ref
  int traj_refs = 0;
};

// Per-reference warp parameter bank (§7.12.2.20 / §7.12.2.11).  The bank is
// frame state: a LOCALWARP block records its chosen model.  Reads are
// newest-first through `at()`.
class WarpParamBank {
 public:
  void reset();
  void record(const adv::WarpParams& params);
  int size() const { return size_; }
  int start() const { return start_; }
  const adv::WarpParams& slot(int physical) const;
  const adv::WarpParams& at(int index) const {
    return slot((start_ + index) % kWarpParamBankSize);
  }

 private:
  adv::WarpParams params_[kWarpParamBankSize]{};
  int start_ = 0;
  int size_ = 0;
};

// ---------------------------------------------------------------------------
// Stack parameters
// ---------------------------------------------------------------------------

struct Stack;  // defined below; the temporal-scan hook updates it in place

struct StackParams {
  MiGrid grid{};
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_h4 = 16;  // Num_4x4_Blocks_High[SbSize]

  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
  bool is_compound = false;
  bool skip_mode = false;
  bool derive_wrl = false;
  bool use_intrabc = false;
  bool force_integer_mv = false;
  int frame_mv_precision = kMvPrecisionEighthPel;

  // §7.12.2.1/.2 global motion.  `gm_type_by_ref` is GmType[] indexed by
  // RefFrame (0..6); `global_mv` is GlobalMvs[list]; `gm_params[list]` is
  // gm_params[RefFrame[list]].
  int gm_type_by_ref[kRefsPerFrame] = {kGmIdentity, kGmIdentity, kGmIdentity,
                                       kGmIdentity, kGmIdentity, kGmIdentity,
                                       kGmIdentity};
  Mv global_mv[2]{};
  adv::WarpParams gm_params[2]{};
  bool gm_params_present = true;

  // §7.12.2 step 12 (temporal half).  `temporal_scan` is invoked at the exact
  // normative point with `temporal_opaque`; it must update the stack using the
  // receiver's own §7.12.2.7/.8 state.  When null, temporal scan is a no-op
  // (correct for use_ref_frame_mvs == 0).
  bool use_ref_frame_mvs = false;
  bool enable_mv_traj = false;
  bool use_temporal_first = false;  // computed by build(); input ignored
  void (*temporal_scan)(void* opaque, Stack* stack, int* prune_count) = nullptr;
  void* temporal_opaque = nullptr;

  // FrameDistance[]/OrderHints[] indexed by RefFrame 0..num_refs-1.
  const int* frame_distance = nullptr;
  const int* order_hints = nullptr;
  int num_refs = 0;
  int order_hint = 0;
  std::int8_t closest_past = kRefNone;
  std::int8_t closest_future = kRefNone;
  int num_past_refs = 0;
  int num_future_refs = 0;
  bool closest_past_restricted = false;
  bool closest_future_restricted = false;

  Reorder reorder = Reorder::kDisabled;
  int max_drl_bits_minus_1 = 4;

  // §7.12.2.20 warp candidates.  `warp_bank` covers RefFrame[0]'s bank.
  const WarpParamBank* warp_bank = nullptr;

  const TipField* tip = nullptr;
};

struct Candidate {
  Mv mv[2];  // list 0 / list 1
  int weight = 0;
  int row_offset = 0;
  int col_offset = 0;
  std::int8_t cwp = kCwpEqual;

  bool same(const Candidate& other, bool compound) const {
    return mv[0] == other.mv[0] && (!compound || mv[1] == other.mv[1]);
  }
};

struct Stack {
  Candidate cand[kMaxRefMvStackSize];
  int count = 0;
  int num_nearest = 0;
  bool sorted = false;

  // §7.12.2.11 WarpParamStack / NumWarpFound.
  adv::WarpParams warp_params[kMaxWarpRefCandidates];
  int num_warp_found = 0;

  // NumMvFound after step 20 and before the §7.12.2.20 extra search.  An index
  // >= scanned_count therefore names a candidate that only exists because of
  // the global / MVP-combination / derived insertions.
  int scanned_count = 0;

  // Build bookkeeping (also the state the §7.12.2.20 PruneCount budget uses).
  int prune_count = 0;
  int single_prune_count = 0;
  int derived_prune_count = 0;
  int derived_count = 0;
};

// Trace of which §7.12.2.10 branch fired; used by the tests and the mirror's
// structural checks.  Index 0/1 = candList.
struct DispatchTrace {
  int search_stack[2] = {0, 0};
  int tip_single_derived[2] = {0, 0};
  int tip_add_derived = 0;
  int single_add_derived[2] = {0, 0};
  int compound_tip = 0;
  int compound_search = 0;
  int compound_add_derived = 0;
  int warp_inserted = 0;
};

// Full build state.  `build()` accepts an optional pointer so callers can see
// the derived stack and dispatch trace without re-running the process.
struct BuildState {
  Stack stack{};
  DispatchTrace trace{};
  struct Derived {
    Mv mv[2];
  } derived[kMaxDrStackSize]{};
  struct Single {
    std::int8_t ref = kRefNone;
    Mv mv;
  } single[kMaxDrStackSize]{};
  int derived_count = 0;
  int single_count = 0;
  int prune_count = 0;
  int single_prune_count = 0;
  int derived_prune_count = 0;
  bool use_temporal = false;
  bool use_temporal_first = false;
  bool is_sb_border = false;
};

// ---------------------------------------------------------------------------
// Stack API
// ---------------------------------------------------------------------------

// Runs §7.12.2 steps 1-22 in normative order.
Status build(const StackParams& params, Stack* stack,
             BuildState* state = nullptr);

// §7.12.2.20 in isolation.  `state` must hold the post-step-20 stack and the
// derived/single bookkeeping.  Exposed because the clamp order and the
// `!use_intrabc` global-candidate guard are the audit's specific findings.
Status extra_search(const StackParams& params, BuildState* state);

// §7.12.2.22.
Status fill_derived(const StackParams& params, BuildState* state);

// §7.12.2.23.
Status clamp_stack(const StackParams& params, Stack* stack);

// §7.12.2.12 / .13 / .10.
Status search_stack(const StackParams& params, BuildState* state, int mv_row,
                    int mv_col, int weight, int cand_list);
Status compound_search_stack(const StackParams& params, BuildState* state,
                             int mv_row, int mv_col, int weight);
Status add_reference_mv(const StackParams& params, BuildState* state,
                        int mv_row, int mv_col, int weight);

// §7.12.2.9 / .11.
Status add_warp_motion_vector(const StackParams& params, BuildState* state,
                              int mv_row, int mv_col);
Status insert_warp_candidate(const adv::WarpParams& params, Stack* stack);

// §7.12.2.6 / .5.  `scan_point_if_valid` is folded into scan_col.
Status scan_point(const StackParams& params, BuildState* state, int delta_row,
                  int delta_col);
Status scan_col(const StackParams& params, BuildState* state, int delta_col);

// §7.12.2.16 / .14 / .15 / .17 / .18.
Status single_add_derived(const StackParams& params, BuildState* state,
                          int mv_row, int mv_col, int cand_list);
Status compound_add_derived(const StackParams& params, BuildState* state,
                            int mv_row, int mv_col);
Status derive_ref_mv_candidate_from_tip_mode(const StackParams& params,
                                             BuildState* state, int mv_row,
                                             int mv_col, int weight);
Status derive_single_ref_mv_candidate_from_tip_mode(
    const StackParams& params, BuildState* state, int mv_row, int mv_col,
    int weight, int cand_list);
Status tip_add_derived(const StackParams& params, BuildState* state, int mv_row,
                       int mv_col);

// §7.12.2.7 (single ref-list sweep) and .8 (sampled projection).
struct MvPair {
  Mv a;
  Mv b;
};

// §7.12.2.15/.17/.18 helper: get_tip_cand(candRow, candCol).
Status get_tip_cand(const StackParams& params, int cand_row, int cand_col,
                    Mv out[2]);
// get_tip_offsets(): (refOffset, pastOffset, futureOffset), clamped.
Status get_tip_offsets(const StackParams& params, int* ref_offset,
                       int* past_offset, int* future_offset);
// §7.12.2.8 get_motion_field_mv().
Status get_motion_field_mv(const StackParams& params, std::int8_t dst, int y8,
                           int x8, Mv* out);

// §7.12.2.19.
void sort_stack_range(const StackParams& params, Stack* stack, int start,
                      int end);

// §7.12.2.1/.2.
Status setup_global_mv(const StackParams& params, int ref_list, Mv* out);
// §7.12.2.2 (public for tests).  x/y are Q16 luma; returns 1/8-pel MV.
Mv get_warp_motion_vector(const adv::WarpParams& warp, int mi_col, int mi_row,
                          int bw4, int bh4, int mi_cols, int mi_rows,
                          int mv_precision, bool force_integer_mv);
// §7.12.2.4 helper; pos_row/pos_col are MI coordinates.
Mv get_warp_motion_vector_xy_pos(const adv::WarpParams& warp, int pos_row,
                                 int pos_col, int mi_rows, int mi_cols);

// §7.12.2.3/.4.
struct CornerResult {
  int corners_found = 0;
  bool all_mvs_same = true;
  bool negative_ref_pts = false;
  bool inserted = false;
  int pts[3][2] = {{0, 0}, {0, 0}, {0, 0}};   // [row,col] luma
  Mv mvs[3];                                   // 1/8 pel
  int ref_pts[3][2] = {{0, 0}, {0, 0}, {0, 0}};  // Q16
  adv::WarpParams wmmat{};
};
Status generate_points_from_corners(const StackParams& params,
                                    BuildState* state, int iter,
                                    CornerResult* out);

// §5.20.9.4/.5.
std::int32_t clamp_mv_row(const StackParams& params, std::int32_t value);
std::int32_t clamp_mv_col(const StackParams& params, std::int32_t value);

// §7.11.4 WarpMvCount (and WarpSampleFound) for the current block.  Used by the
// `is_warp` CDF row and by the gate's context derivation.
struct WarpContextResult {
  int warp_mv_count = 0;
  bool warp_sample_found[2] = {false, false};
  int extend_delta_row = 0;
  int extend_delta_col = 0;
};
Status compute_warp_context(const StackParams& params,
                            WarpContextResult* out);

// ---------------------------------------------------------------------------
// Predicates and small helpers
// ---------------------------------------------------------------------------
bool is_thin_4xn_nx4_block(partition::BlockSize size);
bool is_comp_ref_allowed(partition::BlockSize size);
// §7.12.2.20: `Block_Width > 32 && Block_Height > 32`.
bool mvp_combinations_allowed(int width, int height);

bool has_newmv_mode(int y_mode);
bool has_newmv_for_list(int y_mode, int ref_list);
bool has_nearmv_mode(int y_mode);

// §5.20.7.13 lower_mv_precision.
void lower_mv_precision(int precision, Mv* mv);

// §5.20.7.13 mv_clamp_to_integer (used by the mode resolver).
std::int32_t clamp_mv_to_integer(std::int32_t v);
Mv clamp_mv_to_integer(Mv mv);

// §7.12.2.8 helper: get_mv_projection (§7.9.4) with the trajectory clamp.
Mv project_mv(Mv mv, int numerator, int denominator);

// Structural validation of the parameters (used by build()).
Status validate(const StackParams& params, std::string* why = nullptr);

// Structural equality (all candidate fields, weights, offsets, cwp).
bool equal(const Stack& a, const Stack& b);
int first_divergence(const Stack& a, const Stack& b);

}  // namespace av2enc::inter::mv_stack_full
