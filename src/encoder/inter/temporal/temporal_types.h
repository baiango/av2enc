// AV2 v1.0.0 temporal inter tools — shared POD types and constants.
//
// Scope (AV2 Bitstream & Decoding Process Specification v1.0.0, 2026-05-28,
// commit 38f28e7):
//
//   §7.9      motion field estimation / temporal MV projection
//   §7.10     TIP motion-field setup and build-TIP structures
//   §7.12.2.7/.8   temporal scan / temporal sample
//   §7.12.2.21     reference-MV bank fill
//   §7.13.3.2      predict TIP
//   §7.13.3.3–.6   refine-MV prediction / reference area / search
//   §7.13.3.8–.12  optical-flow refinement (OPFL)
//   §7.22     motion field motion vector storage
//
// This component is a CPU correctness reference.  It does not write the
// M0/M1/M2 bitstream; see temporal_syntax.h for the temporal syntax bits and
// README.md for the exact M2 integration points.
//
// Conventions:
//   * Sample coordinates are row/column.
//   * Motion vectors are 1/8 luma samples, `mv[0] = row`, `mv[1] = column`.
//   * Motion-field locations are 8x8 luma samples: `x8`, `y8`.
//   * Mode-info locations are 4x4 luma samples: `mi_row`, `mi_col`.
//   * Predicted samples that feed a compound blend are 1/16 sample precision
//     (4 fractional bits, `sample << 4`), matching `Preds[]` after
//     `block_inter_prediction()` with `isCompound = 1`.
//
// Everything here is fixed-size and standard-library-free so the same
// arithmetic can be reused by a CUDA translation unit.

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_TYPES_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_TYPES_H_

#include <cstdint>

#include "encoder/inter/mc/mc_types.h"

namespace av2enc::inter::temporal {

enum class TemporalStatus : std::int32_t {
  kOk = 0,
  kInvalidArgument,
  kUnsupported,
};

const char* temporal_status_string(TemporalStatus status);

// ---------------------------------------------------------------------------
// §9.2 constants used by the temporal tools.
// ---------------------------------------------------------------------------

inline constexpr int kMiSize = 4;
inline constexpr int kMiSizeLog2 = 2;
inline constexpr int kMaxRefFrames = 16;
inline constexpr int kRefsPerFrame = 7;
inline constexpr int kNumRefs = 7;  // ref_frame values 0..6
inline constexpr int kRestrictedOrderHint = -1;

// Reference frame values (§5.20.7.10 Table: ref_frame).
inline constexpr int kRefNone = -1;
inline constexpr int kTipFrame = 7;
inline constexpr int kIntraFrame = 8;
inline constexpr int kFirstInterRef = 0;
inline constexpr int kLastInterRef = 6;

inline constexpr int kMaxFrameDistance = 31;
inline constexpr int kRefMvsLimit = 2047;  // REFMVS_LIMIT
inline constexpr int kMvLow = -(1 << 16);
inline constexpr int kMvUpp = (1 << 16);

inline constexpr int kFilterBits = 7;
inline constexpr int kSubpelBits = 4;
inline constexpr int kScaleSubpelBits = 10;
inline constexpr int kInterRound0 = 3;
inline constexpr int kInterRound1Single = 11;
inline constexpr int kInterRound1Compound = 7;
inline constexpr int kInterPostRoundCompound = 2 * kFilterBits - (kInterRound0 + kInterRound1Compound);

inline constexpr int kOpflGradUnit = 16;
inline constexpr int kOpflGradUnitLog2 = 4;
inline constexpr int kOpflMvDeltaLimit = 16;  // 1 << MV_REFINE_PREC_BITS
inline constexpr int kMvRefinePrecBits = 4;
inline constexpr int kMaxLsBits = 26;

inline constexpr int kRefMvBankSize = 4;
inline constexpr int kBankRefsPerFrame = 9;
inline constexpr int kMaxRmbSbHits = 64;

inline constexpr int kMaxPrNum = 16;
inline constexpr int kMaxRefMvStackSize = 6;
inline constexpr int kMaxDrStackSize = 4;
inline constexpr int kMaxDrPrNum = 2;
inline constexpr int kTipMfmvStackSize = 3;
inline constexpr int kMfmvStackSize = 4;
inline constexpr int kTipContexts = 3;

inline constexpr int kCwpEqual = 8;

inline constexpr bool is_inter_ref(int ref) {
  return ref >= kFirstInterRef && ref <= kLastInterRef;
}

inline constexpr bool is_valid_mv_component(int mv) {
  return mv > kMvLow && mv < kMvUpp;
}

// §6.19.7.11 `FrameType` values.
enum class FrameType : std::int32_t {
  kKey = 0,
  kInter = 1,
  kIntraOnly = 2,
  kSwitch = 3,
};

// §6.20 Table 6.20 TipFrameMode values.
enum class TipFrameMode : std::int32_t {
  kDisabled = 0,
  kAsReference = 1,
  kAsOutput = 2,
};

// §6.4 Table 6.5 enable_opfl_refine values.
enum class RefineType : std::int32_t {
  kNone = 0,
  kSwitchable = 1,
  kAll = 2,
  kAuto = 3,  // sequence-header only; never a frame opfl_refine_type
};

// A motion vector in 1/8 luma samples.  Row first to match spec `mv[0]`.
struct Mv {
  std::int32_t row = 0;
  std::int32_t col = 0;
};

inline constexpr bool operator==(Mv a, Mv b) {
  return a.row == b.row && a.col == b.col;
}
inline constexpr bool operator!=(Mv a, Mv b) { return !(a == b); }

struct MvPair {
  Mv mv[2];  // list 0 / list 1
};

// AV2 interp filters (§5.18.5.1).  The values match `Subpel_Filters` rows, so
// mc::kSubpelFilters can be indexed directly.
using InterpFilter = mc::InterpFilter;

// A read-only 8-bit plane.  `width`/`height` are the MI-aligned plane extent
// that the decoder may read (RefMiCols*4 >> sub, RefMiRows*4 >> sub).
struct PlaneView {
  const std::uint8_t* data = nullptr;
  int stride = 0;
  int width = 0;
  int height = 0;
};

// One reference-frame slot as seen by the temporal tools.
struct RefFrameInfo {
  int slot = -1;                     // ref_frame_idx[rf]
  FrameType type = FrameType::kInter;  // RefFrameType[slot]
  int width = 0;                     // RefFrameWidth[slot]
  int height = 0;                    // RefFrameHeight[slot]
  int order_hint = kRestrictedOrderHint;  // RefOrderHint[slot]
  int num_saved_refs = 0;            // RefNumTotalRefs[slot]
  // SavedOrderHints[slot][k], k = 0..num_saved_refs-1.
  int saved_order_hint[kRefsPerFrame] = {
      kRestrictedOrderHint, kRestrictedOrderHint, kRestrictedOrderHint,
      kRestrictedOrderHint, kRestrictedOrderHint, kRestrictedOrderHint,
      kRestrictedOrderHint};
};

// Saved motion field of one reference frame (§7.22 storage output).
//
// `mv` is [y8][x8][list][comp], compressed with compression_mv().
// `ref` is [y8][x8][list] and holds the *reference index* (0..6) that the
// saved frame used, i.e. an index into `SavedOrderHints` of that frame.
struct SavedMvField {
  int w8 = 0;
  int h8 = 0;
  const std::int16_t* mv = nullptr;   // compressed MVs
  const std::int8_t* ref = nullptr;   // per-list reference index or kRefNone
};

// Motion field produced by §7.9 and consumed by §7.10 / §7.12.2.7.
struct MotionField {
  int w8 = 0;
  int h8 = 0;
  std::int32_t* mv = nullptr;      // [y8][x8][2]
  std::uint8_t* valid = nullptr;   // [y8][x8]
  std::int32_t* offset = nullptr;  // [y8][x8], MotionFieldOffset
};

// Optional trajectory arrays (§7.9.2 / §7.12.2.8).  Only used when
// `enable_mv_traj` is set.
struct TrajectoryField {
  int w8 = 0;
  int h8 = 0;
  int num_refs = 0;
  std::int32_t* mv = nullptr;       // [rf][y8][x8][2], clipped to ±REFMVS_LIMIT
  std::uint8_t* valid = nullptr;    // [rf][y8][x8]
  std::int16_t* pos = nullptr;      // [k][rf][y8][x8][2]
  std::uint8_t* pos_valid = nullptr;  // [k][rf][y8][x8]
};

// Scratch for the TIP block-average filter.  Supplied by the caller so the
// core stays allocation-free and usable from a CUDA kernel.
struct MotionFieldScratch {
  std::int32_t* avg_mv = nullptr;     // [y8][x8][2]
  std::uint8_t* avg_valid = nullptr;  // [y8][x8]
};

struct MotionFieldConfig {
  int mi_cols = 0;
  int mi_rows = 0;
  int proj_step = 1;          // ProjStep
  bool sb_64x64 = true;       // SbSize == BLOCK_64X64
  bool enable_tip = false;
  bool enable_mv_traj = false;
  bool reduced_ref_frame_mvs_mode = false;
  bool use_bru = false;       // BRU is not part of this component
  int bru_ref = -1;
  int order_hint = 0;         // OrderHint of the current frame
  int num_total_refs = 0;     // NumTotalRefs
  int num_past_refs = 0;      // NumPastRefs (§7.8)
  int num_future_refs = 0;    // NumFutureRefs (§7.8)
  int frame_width = 0;
  int frame_height = 0;
  // ref_order_hint[rf] = OrderHints[rf]; refs[rf] describes ref_frame_idx[rf].
  int ref_order_hint[kRefsPerFrame] = {
      kRestrictedOrderHint, kRestrictedOrderHint, kRestrictedOrderHint,
      kRestrictedOrderHint, kRestrictedOrderHint, kRestrictedOrderHint,
      kRestrictedOrderHint};
  RefFrameInfo refs[kRefsPerFrame];
  SavedMvField saved[kRefsPerFrame];  // saved field of refs[rf].slot
};

struct MotionFieldResult {
  int closest_past = kRefNone;
  int closest_future = kRefNone;
  int projection_count = 0;
};

// ---------------------------------------------------------------------------
// Reference-MV bank (§7.12.2.21, §5.20.2.2).
// ---------------------------------------------------------------------------

// RefMvBankParams[ref][idx][0..5] = cwpIdx, key, mv0 row, mv0 col,
//                                  mv1 row, mv1 col.
struct RefMvBankEntry {
  std::int32_t p[6] = {0, 0, 0, 0, 0, 0};
};

struct RefMvBank {
  RefMvBankEntry bank[kBankRefsPerFrame][kRefMvBankSize];
  int size[kBankRefsPerFrame] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int start[kBankRefsPerFrame] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int bank_hits = 0;    // RefMvBankHits
  int remain_hits = 0;  // RefMvRemainHits
  int unit_hits = 0;    // RefMvUnitHits
};

// One row source for reset_ref_mv_bank() (§5.20.2.2).  The caller walks the
// already-coded row above the superblock.
struct RefMvBankRowSource {
  std::uint8_t is_inter = 0;
  std::int8_t ref[2] = {kRefNone, kRefNone};
  std::int32_t mv[2][2] = {{0, 0}, {0, 0}};
  std::int32_t cwp_idx = kCwpEqual;
  std::uint8_t motion_mode = 0;  // >= 2 is LOCALWARP
  int bw4 = 1;                   // Num_4x4_Blocks_Wide of this block
};

// A stack entry shared with the MV-prediction stack.  `cwp` is CWP_EQUAL for
// bank-derived candidates unless the stored bank entry says otherwise.
struct StackMv {
  Mv mv[2];
  int cwp = kCwpEqual;
  int weight = 1;
};

// ---------------------------------------------------------------------------
// TIP (§7.10, §7.13.3.2).
// ---------------------------------------------------------------------------

// Tip_Weighting_Factor[8] (§7.13.3.1).
inline constexpr int kTipWeightingFactor[8] = {8, 12, 16, 18, 20, 4, 6, -4};

struct TipOffsets {
  int ref_offset = 0;     // distance between ClosestFuture and ClosestPast
  int past_offset = 0;    // current frame to ClosestPast
  int future_offset = 0;  // current frame to ClosestFuture
};

struct TipConfig {
  TipFrameMode mode = TipFrameMode::kDisabled;
  int tip_global_wtd_index = 0;
  Mv tip_global_mv{};
  InterpFilter interp_filter = InterpFilter::kEighttapSharp;
  bool enable_tip_refinemv = false;
  bool enable_refinemv = false;
  RefineType opfl_refine_type = RefineType::kNone;
  bool allow_tip_hole_fill = false;
  int num_past_refs = 0;
  int num_future_refs = 0;
  int order_hint = 0;
  int closest_past_order_hint = kRestrictedOrderHint;
  int closest_future_order_hint = kRestrictedOrderHint;
};

// One build-TIP block.  `bsize` is the coded block size name; only the MI
// extent matters to callers of this component.
struct TipBlock {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 2;  // 8x8 default
  int bh4 = 2;
  Mv mv[2];     // final per-list MVs, 1/8 luma samples
  bool motion_field_valid = false;
};

// TIP structures for a whole frame.  The caller owns the storage; this
// component fills it.  `mv_grid` is [mi_rows][mi_cols][2] so the MV can be
// stored per 4x4 through §7.22.
struct TipPlan {
  TipOffsets offsets{};
  int weight = kCwpEqual;
  int num_blocks = 0;  // capacity on input, block count on output
  TipBlock* blocks = nullptr;
  Mv* mv_grid = nullptr;   // [mi_rows][mi_cols], list 0 (Mvs[..][0])
  Mv* mv_grid1 = nullptr;  // optional [mi_rows][mi_cols], list 1 (Mvs[..][1])
};

// Inputs to get_tip_cand() (§7.22/§7.13.3.2).  `MiRowBase`/`MiColBase` and
// `TipSizes16x16` are block-grid state owned by the caller.
struct TipCandInput {
  int cand_row = 0;  // MI units
  int cand_col = 0;
  int base_row = 0;  // MiRowBase[0][cand_row][cand_col]
  int base_col = 0;  // MiColBase[0][cand_row][cand_col]
  bool is_16x16 = false;
  Mv base_mv{};  // Mvs[candRow][candCol][0]
  const MotionField* field = nullptr;
};

// ---------------------------------------------------------------------------
// Refine-MV (§7.13.3.3–.6).
// ---------------------------------------------------------------------------

struct RefArea {
  int first_x = 0;
  int first_y = 0;
  int last_x = 0;
  int last_y = 0;
};

struct RefineMvConfig {
  bool enable_refinemv = false;
  RefineType opfl_refine_type = RefineType::kNone;
  bool frame_is_switch = false;
  bool is_compound = false;
  bool skip_mode = false;
  bool use_optflow = false;
  bool is_global_global_mv = false;
  bool motion_mode_simple = true;
  bool has_newmv = false;
  bool y_mode_near_near_mv = false;
  bool y_mode_joint_new_mv = false;
  bool ref0_scaled = false;
  bool ref1_scaled = false;
  int block_width = 0;
  int block_height = 0;
  int order_hint = 0;
  int ref0_order_hint = kRestrictedOrderHint;
  int ref1_order_hint = kRestrictedOrderHint;
  // Signalled bit for the switchable case (ignored otherwise).
  bool use_refinemv_bit = true;
};

struct RefineMvSyntax {
  bool use_refinemv = false;
  bool decided_against_refinemv = false;
  bool switchable = false;  // whether the bit is present
};

// ---------------------------------------------------------------------------
// OPFL (§7.13.3.8–.12).
// ---------------------------------------------------------------------------

struct OpflInput {
  int w = 0;  // region width, multiple of 4
  int h = 0;
  const std::int16_t* p0 = nullptr;  // 1/16-precision predictions, list 0
  int p0_stride = 0;
  const std::int16_t* p1 = nullptr;  // list 1
  int p1_stride = 0;
  Mv mvs[2];       // 1/8 luma samples
  int dist[2] = {0, 0};  // scaled order-hint distances, set by opfl_scale_distances
  int n = 8;             // optical-flow block size: 8 or 4
  int opt_x = 0;         // subblock offset within the inter block (samples)
  int opt_y = 0;
};

struct OpflOutput {
  // [y4][x4][list][comp], 1/16 luma samples (OpflMvs)
  std::int16_t* opfl_mvs = nullptr;
  int opfl_stride4 = 0;  // number of 4x4 columns in the destination
  // [y4][x4][list][comp], 1/8 luma samples (MvDeltas)
  std::int16_t* mv_deltas = nullptr;
  int delta_stride4 = 0;
};

// ---------------------------------------------------------------------------
// Temporal syntax configuration (see temporal_syntax.h).
// ---------------------------------------------------------------------------

struct TemporalSequenceConfig {
  // §5.4.6 sequence_inter_config(), temporal slice.  The non-temporal fields
  // between them are carried so the bit positions match the full syntax.
  bool enable_ref_frame_mvs = false;
  bool reduced_ref_frame_mvs_mode = false;
  bool enable_refmvbank = false;
  bool enable_bawp = false;
  bool enable_cwp = false;
  bool enable_imp_msk_bld = false;
  bool enable_df_sub_pu = false;
  bool enable_tip = false;
  bool disable_tip_output = false;
  bool enable_tip_hole_fill = false;
  bool enable_mv_traj = false;
  RefineType enable_opfl_refine = RefineType::kNone;
  bool enable_refinemv = false;
  bool enable_tip_refinemv = false;
  bool enable_tip_explicit_qp = false;  // derived/pass-through
};

struct TemporalFrameConfig {
  bool use_ref_frame_mvs = false;
  int num_total_refs = 0;
  bool sb_64x64 = true;
  bool enable_tip = false;
  bool enable_tip_refinemv = false;
  RefineType enable_opfl_refine = RefineType::kNone;
  bool enable_tip_hole_fill = false;
  int num_past_refs = 0;
  int num_future_refs = 0;
  bool enable_refinemv = false;
  bool is_tip_frame = false;  // is_tip_frame() / OBU_LEADING_TIP or REGULAR_TIP
  bool enable_tip_output = true;  // EnableTipOutput from the sequence header

  // Outputs / signalled values.
  int tmvp_sample_step_minus_1 = 0;  // ProjStep = +1
  TipConfig tip{};                   // mode, weight, global MV, filter
  RefineType opfl_refine_type = RefineType::kNone;
  bool tip_sharp = true;
  bool tip_regular = false;
};

struct TipBlockSyntaxConfig {
  TipFrameMode tip_frame_mode = TipFrameMode::kDisabled;
  bool skip_mode = false;
  int bw4 = 2;
  int bh4 = 2;
  bool mi_size_matches_chroma = true;
  // Outputs.
  bool tip_mode = false;
  int tip_pred_mode = 0;  // 0 = NEARMV, 1 = NEWMV
};

struct RefineMvBlockSyntaxConfig {
  RefineMvConfig ctx{};
  // Outputs.
  RefineMvSyntax syntax{};
};

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_TYPES_H_
