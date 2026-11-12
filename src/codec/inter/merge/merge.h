// AV2 skip / merge / DRL semantics for the M2 P-frame path.
//
// Scope
// -----
// This is the M2-owned correctness layer for the inter block-mode decision:
//
//   * frame-level skip_mode_params() and block-level skip/merge eligibility
//     (§5.18.8.2, §5.20.5.9, §5.20.7.1, §5.20.7.6);
//   * single_mode decoding into YMode = NEARMV + single_mode
//     (§5.20.7.6, §5.20.7.8, §6.19.7.5);
//   * read_drl_idx() / DRL index derivation and the DRL_REORDER_* stack order
//     (§5.20.7.8, §7.12.2 steps 16-17 and §7.12.2.19);
//   * a spec-exact spatial MV stack builder (§7.12.2) that exposes the
//     §7.12.2.20 MVP-combination gate that the shared `inter::mc` component
//     currently omits; plus an audit wrapper that compares the shared
//     component against this reference without modifying it;
//   * skip vs merge vs NEWMV resolution and syntax validation.
//
// The M2 core owns bitstream assembly.  This module produces the resolved
// mode/MV/reference pair and the DRL/`single_mode` symbols; it never writes a
// bitstream and never mutates a reference frame.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.18.8.2, §5.20.5.9, §5.20.7.1-.13,
// §6.19.7.5, §7.11.2-.3, §7.12.2 (spatial subset), §7.12.2.19-.20,
// §7.12.2.23.
//
// Deliberately out of scope (each is disabled by the M2 sequence header):
// temporal MV projection, the ref-MV bank, derived-SMVP tools that need
// temporal MVs, TIP, warp/OBMC/inter-intra, BAWP, use_optflow, refine-MV,
// adaptive MVD, flex MV resolution, global motion (GmType > IDENTITY), and
// CWP weights other than CWP_EQUAL.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/mc/mc.h"
#include "codec/partition/tree/partition.h"

namespace av2enc::video::merge {

using partition::BlockSize;

// §9.2 constants used by the stack and DRL.
inline constexpr int kMaxRefMvStackSize = 6;  // MAX_REF_MV_STACK_SIZE
inline constexpr int kMaxPrNum = 16;          // MAX_PR_NUM
inline constexpr int kMaxDrStackSize = 4;     // MAX_DR_STACK_SIZE
inline constexpr int kMaxDrPrNum = 2;         // MAX_DR_PR_NUM
inline constexpr int kMaxDrlIndex = kMaxRefMvStackSize - 1;
inline constexpr int kMaxSupportedRefs = 4;   // M2 project limit (spec: 16)
inline constexpr int kMaxBlockMi = 16;        // 64x64 luma block
inline constexpr int kMinBlockMi = 1;         // 4x4 luma block

// AV2 RefFrame values (§6.19.7.9).  Inter references are 0..6 (the M2
// project uses 0..3); TIP_FRAME is 7, INTRA_FRAME is 8 and NONE is -1.
// Anything that is not kRefNone is treated as "written" by the stack scan.
inline constexpr std::int8_t kRefNone = -1;
inline constexpr std::int8_t kTipFrame = 7;
inline constexpr std::int8_t kIntraFrame = 8;

enum class Status : std::int32_t {
  kOk = 0,
  kInvalidArgument,
  kInvalidBlockSize,
  kInvalidRefCount,
  kInvalidDrlIndex,
  kTruncatedDrlBits,
  kInvalidSkipMode,
  kUnsupportedFeature,
};

const char* status_string(Status status);

// DrlReorder (§5.4.6 / Table 6.4).
enum class DrlReorder : std::uint8_t {
  kDisabled = 0,    // DRL_REORDER_DISABLED
  kConstraint = 1,  // DRL_REORDER_CONSTRAINT
  kAlways = 2,      // DRL_REORDER_ALWAYS
};

// YMode values (§6.19.7.5).
enum YMode : int {
  kYModeNearMv = 14,
  kYModeGlobalMv = 15,
  kYModeNewMv = 16,
  kYModeWarpMv = 17,
  kYModeWarpNewMv = 18,
  kYModeNearNearMv = 19,
  kYModeNearNewMv = 20,
  kYModeNewNearMv = 21,
  kYModeGlobalGlobalMv = 22,
  kYModeNewNewMv = 23,
  kYModeJointNewMv = 24,
};

// ---------------------------------------------------------------------------
// Small normative predicates
// ---------------------------------------------------------------------------

bool is_inter_ref(std::int8_t ref);
bool is_thin_4xn_nx4_block(BlockSize size);
// is_comp_ref_allowed(): Min(w,h) >= 8 or a thin 4xN/Nx4 block.
bool is_comp_ref_allowed(BlockSize size);
// §7.12.2.20: MVP-combination candidates are inserted only when both block
// dimensions are strictly greater than 32 luma samples.
bool mvp_combinations_allowed(int width, int height);
bool mvp_combinations_allowed(BlockSize size);

// §5.18.3.1.  `restricted` inputs are represented by RefOrderHint below rather
// than a magic value; this helper is the unrestricted form.
int relative_dist(int a, int b);

// ---------------------------------------------------------------------------
// Frame-level skip_mode_params() (§5.18.8.2)
// ---------------------------------------------------------------------------

struct RefOrderHint {
  int order_hint = 0;
  // The saved reference order hint equals RESTRICTED_OH.
  bool restricted = false;
  bool valid = false;
};

struct SkipModeFrameParams {
  bool frame_is_intra = false;
  bool switch_frame = false;
  int num_total_refs = 1;  // 1..kMaxSupportedRefs
  int order_hint = 0;      // current frame order hint
  bool order_hint_restricted = false;
  RefOrderHint refs[kMaxSupportedRefs]{};
};

struct SkipModeFrameResult {
  bool allowed = false;
  int num_total_refs = 1;
  // Ref list indices, not frame-type enums.
  std::int8_t frame[2] = {0, 0};
};

Status derive_skip_mode_params(const SkipModeFrameParams& params,
                               SkipModeFrameResult* result);

// Checks the syntax gate: skip_mode_present may be 1 only when allowed.
Status validate_skip_mode_present(const SkipModeFrameResult& result,
                                  bool skip_mode_present);

// Neighbour reference frames used by skip_mode_frames().  `count` is NNumBuf
// (0..2), and ref[n][list] is NRefFrame[n][list].
struct NeighborRefInfo {
  int count = 0;
  std::int8_t ref[2][2] = {{kRefNone, kRefNone}, {kRefNone, kRefNone}};
};

struct SkipModeFrameInputs {
  NeighborRefInfo neighbours{};
  std::int8_t closest_past = kRefNone;
  std::int8_t closest_future = kRefNone;
};

// §5.20.7.10 skip_mode_frames(): neighbour pair override, TIP override, then
// the frame-level SkipModeFrame[] fallback.
Status derive_skip_mode_frames(const SkipModeFrameResult& result,
                               const SkipModeFrameInputs& inputs,
                               std::int8_t out[2]);

// Block-level gate for read_skip_mode(): the syntax is present only when
// skip_mode_present, no SEG_LVL_SKIP / SEG_LVL_GLOBALMV feature is active,
// the region is not intra, and is_comp_ref_allowed() is true.
struct BlockSkipModeInputs {
  bool skip_mode_present = false;
  bool seg_lvl_skip = false;
  bool seg_lvl_globalmv = false;
  bool region_intra = false;
  BlockSize size = BlockSize::kInvalid;
};

Status skip_mode_syntax_allowed(const BlockSkipModeInputs& inputs,
                                bool* allowed);

// ---------------------------------------------------------------------------
// DRL index syntax (§5.20.7.8)
// ---------------------------------------------------------------------------
//
// read_drl_idx(start, m):
//   for (idx = start; idx < m; idx++) { drl_mode; if (!drl_mode) return idx; }
//   return m;
//
// The exhaustion value is `m`, so it is *not* encoded with a terminating zero.
// `m` is max_drl_bits_minus_1 + 1 and is at most kMaxDrlIndex (5).  `start`
// may be RefMvIdx0 + 1 for same-reference NEAR_NEARMV; when start >= m the
// process returns m without reading any bit.

std::vector<std::uint8_t> encode_drl_index(int index, int start, int m);

Status decode_drl_index(std::span<const std::uint8_t> bits, int start, int m,
                        int* index, std::size_t* consumed);

// CDF row for drl_mode: Min(idx, 2).
int drl_cdf_row(int idx);

// has_second_drl(YMode) (§5.20.7.6).
bool has_second_drl(int y_mode, bool skip_mode, bool use_optflow);

// start value for the second read_drl_idx() call.
Status derive_second_drl_start(int y_mode, bool same_refs, int refmv_idx0,
                               int* start);

// ---------------------------------------------------------------------------
// single_mode decoding (§5.20.7.6)
// ---------------------------------------------------------------------------

enum class SingleMode : std::uint8_t {
  kNearMv = 0,    // YMode NEARMV
  kGlobalMv = 1,  // YMode GLOBALMV
  kNewMv = 2,     // YMode NEWMV
};

Status decode_single_mode(int single_mode, SingleMode* out);
int single_mode_y_mode(int single_mode);  // 14 + single_mode
bool single_mode_has_drl(SingleMode mode);
bool single_mode_has_mvd(SingleMode mode);

// ---------------------------------------------------------------------------
// Mode contexts (§7.11.2, §9.3 CDF selection)
// ---------------------------------------------------------------------------

struct NeighborMode {
  bool available = false;
  bool is_inter = false;
  std::uint8_t y_mode = 0;
  std::int8_t ref[2] = {kRefNone, kRefNone};
  bool skip_mode = false;
  bool skip_flag = false;
};

struct NeighborModes {
  NeighborMode above{};
  NeighborMode left{};
};

// TileSkipModeCdf[ctx]: sum of neighbour SkipModes (0..2).
int skip_mode_context(const NeighborModes& neighbours);
// TileSkipCdf[ctx]: sum of neighbour Skips + 3 when the current block has
// skip_mode == 1 (SKIP_CONTEXTS >> 1).
int skip_flag_context(const NeighborModes& neighbours, bool skip_mode);

// Full §7.11.2 NewMvContext derivation.  `y_mode` is a per-MI-cell YMode grid;
// pass nullptr to treat every neighbour as non-NEWMV (only useful for tests).
struct NewMvContextParams {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_h4 = 16;
  int mi_cols = 0;
  int mi_rows = 0;
  const inter::mc::MiInfo* mi = nullptr;
  const std::uint8_t* y_mode = nullptr;
  bool is_compound = false;
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
};

Status derive_new_mv_context(const NewMvContextParams& params, int* ctx);

// ---------------------------------------------------------------------------
// Reference MV stack (§7.12.2 spatial subset, spec-exact)
// ---------------------------------------------------------------------------

struct StackParams {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_h4 = 16;  // 16 for a 64x64 superblock
  int mi_cols = 0;
  int mi_rows = 0;
  // Row-major mi_cols x mi_rows grid.  `ref[0] == kRefNone` marks an unwritten
  // cell.  Intra cells must use kIntraFrame (8), not 0: 0 is a valid inter
  // reference index.
  const inter::mc::MiInfo* mi = nullptr;

  // RefFrame[0] / RefFrame[1].  Use kRefNone for list 1 of a single block.
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
  bool is_compound = false;

  // GlobalMvs[0] / GlobalMvs[1] (identity -> {0,0}).
  inter::mc::Mv global_mv[2]{};

  DrlReorder reorder = DrlReorder::kDisabled;
  // useTemporalFirst can only be true when ref_frame_mvs is enabled.  M2 has
  // enable_ref_frame_mvs = 0, so this is false.
  bool use_temporal_first = false;
  // max_drl_bits_minus_1 from the sequence/frame header.  The derived-SMVP
  // fill cap is max_drl_bits_minus_1 + 2.  The M2 baseline uses 4.
  int max_drl_bits_minus_1 = 4;

  // Out of scope and rejected when true: intra block copy.
  bool use_intrabc = false;

  // FrameDistance[ref] for ref labels 0..num_refs-1.  Required for the
  // single-add-derived projection when a neighbouring list references a
  // different reference frame.  With a single reference frame no projection
  // is possible, so nullptr is fine.
  const int* frame_distance = nullptr;
  int num_refs = 0;
};

struct StackCandidate {
  inter::mc::Mv mv[2];  // list 0 / list 1
  int weight = 0;
  int row_offset = 0;
  int col_offset = 0;
  std::int8_t cwp = 8;  // CWP_EQUAL; M2 has enable_cwp = 0
};

struct MvStack {
  StackCandidate cand[kMaxRefMvStackSize];
  int count = 0;
  int num_nearest = 0;
  bool sorted = false;
};

// Spec-exact spatial stack.  Unlike inter::mc::build_mv_stack this applies the
// §7.12.2.20 `Block_Width > 32 && Block_Height > 32` MVP-combination gate and
// supports the compound (two-list) case needed by skip_mode.
Status build_mv_stack(const StackParams& params, MvStack* stack);

// ---------------------------------------------------------------------------
// Shared-component audit (does not modify inter::mc)
// ---------------------------------------------------------------------------

struct McStackAudit {
  bool mc_supported = false;  // false for compound / unsupported params
  bool mc_call_ok = false;
  bool matches_spec = false;
  int mc_count = -1;
  int expected_count = -1;
  int first_divergence = -1;  // candidate index, or -1

  // Root causes observed when matches_spec is false.
  bool mvp_combinations_not_gated = false;
  bool above_left_sb_border_scan_diff = false;
  bool derived_candidates_missing = false;
  bool extra_mv_low_clamp = false;
};

// Runs inter::mc::build_mv_stack for a single-reference block and compares it
// with this module's spec-exact stack (and with instrumented variants).  The
// call is read-only.
Status audit_mc_stack(const StackParams& params, McStackAudit* audit);

// Always produces the spec-exact stack.  This is the API the M2 core should
// use until inter::mc::build_mv_stack is fixed.  `audit` is optional.
Status build_stack_checked(const StackParams& params, MvStack* stack,
                           McStackAudit* audit = nullptr);

// ---------------------------------------------------------------------------
// Mode resolution: skip_mode (compound merge), NEARMV merge, GLOBALMV, NEWMV
// ---------------------------------------------------------------------------

struct ModeSyntax {
  bool is_inter = true;
  bool skip_flag = false;  // no residual; independent of skip_mode
  bool skip_mode = false;  // compound merge mode
  SingleMode single_mode = SingleMode::kNearMv;
  int y_mode = kYModeNearMv;
  int drl_index = 0;
  inter::mc::Mv mvd{};
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
};

struct ResolvedMode {
  bool compound = false;
  bool skip_mode = false;
  bool skip_flag = false;
  int y_mode = kYModeNearMv;
  int drl_index = 0;
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
  inter::mc::Mv pred_mv[2];
  inter::mc::Mv mvd[2];
  inter::mc::Mv mv[2];
  std::int8_t cwp = 8;
};

// Structural checks that do not depend on a particular stack instance.
Status validate_mode_syntax(const ModeSyntax& syntax, const StackParams& params,
                            const MvStack& stack, std::string* why = nullptr);

// Resolves the final MVs.  skip_mode is the compound merge case: both list MVs
// come from RefStackMv[RefMvIdx], no MVD.  Single-mode NEARMV is a merge with
// no MVD; GLOBALMV uses GlobalMvs[0]; NEWMV adds the signalled MVD.
Status resolve_mode(const MvStack& stack, const StackParams& params,
                    const ModeSyntax& syntax, ResolvedMode* out);

Status resolve_single_mode(const MvStack& stack, const StackParams& params,
                           SingleMode mode, int drl_index,
                           inter::mc::Mv mvd, ResolvedMode* out);

// Convenience for the compound merge case.  Derives the two reference frames
// with skip_mode_frames() (including the neighbour override) and resolves the
// two-list stack entry at `drl_index`.  `params` must have been built with that
// same reference pair.
Status resolve_skip_mode(const MvStack& stack, const StackParams& params,
                         const SkipModeFrameResult& skip_params,
                         const SkipModeFrameInputs& skip_inputs, int drl_index,
                         bool skip_flag, ResolvedMode* out);

// Encoder convenience: first stack entry in [start, Min(m, count-1)] whose
// list-0 MV equals `mv`.  Deterministic (lowest index wins).
Status choose_drl_index_for_mv(const MvStack& stack, inter::mc::Mv mv,
                               int start, int m, int* index, bool* found);

// §5.20.7.13 mv_clamp_to_integer, exposed for tests.
std::int32_t clamp_mv_to_integer(std::int32_t v);
inter::mc::Mv clamp_mv_to_integer(inter::mc::Mv mv);

}  // namespace av2enc::video::merge
