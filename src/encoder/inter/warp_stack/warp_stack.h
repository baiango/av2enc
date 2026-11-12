#ifndef AV2ENC_INTER_WARP_STACK_WARP_STACK_H_
#define AV2ENC_INTER_WARP_STACK_WARP_STACK_H_

// AV2 v1.0.0 §7.11.4 + §7.12.2.3/.4/.9/.11 + §7.12.3 — warp machinery.
//
// This component implements the *state* needed by warped motion:
//
//   * §7.11.4  scan point warp context: WarpMvCount, WarpSampleFound[2] and
//              the ExtendDeltaRow/Col of the first matching neighbour,
//              including the even-column alignment used on the top superblock
//              row;
//   * §7.12.2.3/.4  generate points from corners (CornersFound, allMvsSame,
//              the projective wmmat, reduce_warp_model, translation clamp) and
//              the warp corner process, including
//              get_warp_motion_vector_xy_pos();
//   * §7.12.2.9/.11  add warp motion vector, insert warp candidate and the
//              four-entry WarpParamStack (WarpParamBank + gm_params + two
//              Default_Warp_Params, newest bank entry first);
//   * §7.12.3  find warp samples + add sample (CandList, NumSamples,
//              above_sample_stored), and the §7.13.3.23 least-squares model
//              built from that candidate list.
//
// The pixel side is NOT reimplemented here: `estimate_local_warp()` converts
// the normative candidate list into adv::WarpSample and calls
// adv::warp_estimate() (src/encoder/inter/advanced/warp_common.h), which is the
// same transcription the CUDA kernel and the existing warp predictor use.
//
// Relationship to the wired M2 stack
// ----------------------------------
// `build_warp_stack()` is a self-contained §7.12.2 single-reference stack whose
// only difference from `m2::merge::build_mv_stack()` is the warp half.  With
// `derive_wrl == false` it must be candidate-for-candidate identical to the
// wired stack; `warp_stack_test.cpp` asserts that over randomized grids.  This
// module owns no M2 code and is not on the M2 encode path; it exists so the
// warp half can be integrated without duplicating the scan logic.
//
// Deliberately out of scope (each requires syntax the M2 core does not emit):
// temporal scan (§7.12.2.7/.8), derived-SMVP (§7.12.2.14/.16), ref-MV bank
// (§7.12.2.21) and TIP (§7.12.2.15/.17/.18).  Compound stack construction is
// rejected: DERIVEWRL is 0 for compound blocks (§5.20.7.6), so warp candidates
// cannot be inserted there.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/advanced/warp_common.h"
#include "encoder/inter/mc/mc_types.h"

namespace av2enc::inter::warp {

namespace ia = adv;

// ---------------------------------------------------------------------------
// §9.2 constants (values pinned by spec/tables/constants.csv).
// ---------------------------------------------------------------------------

inline constexpr int kMaxRefMvStackSize = 6;      // MAX_REF_MV_STACK_SIZE
inline constexpr int kMaxWarpRefCandidates = ia::kMaxWarpRefCandidates;  // 4
inline constexpr int kWarpParamBankSize = 4;      // WARP_PARAM_BANK_SIZE
inline constexpr int kLeastSquaresSamplesMax = ia::kLeastSquaresSamplesMax;  // 8
inline constexpr int kMaxPrNum = 16;              // MAX_PR_NUM
inline constexpr int kGmTransOnlyPrecDiff = 13;   // GM_TRANS_ONLY_PREC_DIFF
inline constexpr int kLsMvMax = ia::kLsMvMax;     // 256
inline constexpr int kMiSize = 4;                 // MI_SIZE
inline constexpr int kCwpEqual = ia::kCwpEqual;   // 8

// §6.19.7.5 YMode values used by the warp path.
inline constexpr int kYModeNearMv = 14;
inline constexpr int kYModeGlobalMv = 15;
inline constexpr int kYModeNewMv = 16;
inline constexpr int kYModeWarpMv = 17;
inline constexpr int kYModeWarpNewMv = 18;
inline constexpr int kYModeNearNearMv = 19;
inline constexpr int kYModeNewNewMv = 23;
inline constexpr int kYModeGlobalGlobalMv = 22;

// §6.19.7.13 motion modes.  `>= kMotionLocalWarp` is the spec's
// `MotionModes[..] >= LOCALWARP` test.
inline constexpr int kMotionSimple = 0;
inline constexpr int kMotionInterIntra = 1;
inline constexpr int kMotionLocalWarp = 2;
inline constexpr int kMotionDeltaWarp = 3;
inline constexpr int kMotionExtendWarp = 4;

// RefFrame sentinels (§5.20.7.5).  Values 0..6 are inter reference slots.
inline constexpr std::int8_t kRefNone = -1;
inline constexpr std::int8_t kRefTip = 7;
inline constexpr std::int8_t kRefIntra = 8;

inline constexpr bool is_inter_ref_value(std::int8_t ref) {
  return ref != kRefNone && ref != kRefIntra;
}

// DrlReorder (§5.4.6).  The M2 sequence header writes DISABLED.
enum class DrlReorder : std::uint8_t {
  kDisabled = 0,
  kConstraint = 1,
  kAlways = 2,
};

enum class Status : std::int32_t {
  kOk = 0,
  kInvalidArgument,
  kInvalidBlockSize,
  kUnsupportedCompound,
  kUnsupportedTemporal,
  kUnsupportedIntrabc,
};

const char* status_string(Status status);

// ---------------------------------------------------------------------------
// Mode-info grid
// ---------------------------------------------------------------------------
//
// One cell per 4x4 luma MI.  Every field the normative processes read is
// present:
//   * `written` models "RefFrames[mvRow][mvCol][0] has been written for this
//     frame"; write it for intra cells too (they have ref[0] == kRefIntra).
//   * `is_inter` is IsInters.
//   * `mv` is Mvs (block MV, replicated over the block) and is what §7.12.3
//     samples.  `sub_mv` is SubMvs: the block MV for a normal block, but for a
//     LOCALWARP block each 8x8 section stores its warp-projected MV
//     (§7.13.3.19).  The MV stack, corners' non-warp path and compound-add-
//     derived read SubMvs; using the block MV there desynchronises the stack.
//   * `w4`/`h4`/`col_base`/`row_base` are Num_4x4_Blocks_Wide/High[] and
//     MiColBase/MiRowBase for the block containing the cell.
//   * `warp[ref]` is WarpParams[mvRow][mvCol][ref].
struct WarpMi {
  std::int8_t ref[2] = {kRefNone, kRefNone};
  std::int32_t mv[2][2] = {{0, 0}, {0, 0}};      // Mvs[..][list][comp]
  std::int32_t sub_mv[2][2] = {{0, 0}, {0, 0}};  // SubMvs[..][list][comp]
  std::int8_t is_inter = 0;
  std::int8_t written = 0;
  std::int8_t w4 = 0;
  std::int8_t h4 = 0;
  std::int16_t col_base = 0;
  std::int16_t row_base = 0;
  std::uint8_t y_mode = 0;
  std::uint8_t motion_mode = 0;  // MotionModes[]
  ia::WarpParams warp[2]{};
};

struct WarpGrid {
  int mi_cols = 0;
  int mi_rows = 0;
  const WarpMi* mi = nullptr;  // row-major, mi_rows * mi_cols

  bool inside(int row, int col) const {
    return mi != nullptr && row >= 0 && col >= 0 && row < mi_rows && col < mi_cols;
  }
  const WarpMi& at(int row, int col) const {
    return mi[static_cast<std::size_t>(row) * static_cast<std::size_t>(mi_cols) +
              static_cast<std::size_t>(col)];
  }
};

// ---------------------------------------------------------------------------
// §7.11.4 scan point warp context
// ---------------------------------------------------------------------------

// ExtendDeltaRow/ExtendDeltaCol: where the first usable extended-warp
// neighbour was found.  The names come from the spec; `ExtraWarpSample` is the
// value object this API returns (the audit calls the pair "ExtraWarpSample").
struct ExtraWarpSample {
  bool found = false;
  int delta_row = 0;
  int delta_col = 0;
};

struct WarpContext {
  int warp_mv_count = 0;                 // WarpMvCount (TileIsWarpCdf row)
  bool warp_sample_found[2] = {false, false};
  ExtraWarpSample extra;                 // ExtendDeltaRow/Col
};

struct WarpContextParams {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_h4 = 16;
  WarpGrid grid{};
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
};

// Runs the four §7.11.2 scan points through §7.11.4.
WarpContext compute_warp_context(const WarpContextParams& params);

// A single §7.11.4 invocation.  `delta_col` is the pre-alignment value.
void scan_point_warp_context(const WarpContextParams& params, int delta_row,
                             int delta_col, WarpContext* out);

// ---------------------------------------------------------------------------
// §7.12.2.3 / §7.12.2.4 generate points from corners
// ---------------------------------------------------------------------------

struct WarpStack;

struct CornerResult {
  int corners_found = 0;
  bool all_mvs_same = true;
  bool negative_ref_pts = false;
  bool inserted = false;
  int corner_pts[3][2] = {{0, 0}, {0, 0}, {0, 0}};  // [row,col] in luma
  int corner_mvs[3][2] = {{0, 0}, {0, 0}, {0, 0}};  // [mv_y,mv_x], 1/8
  int ref_pts[3][2] = {{0, 0}, {0, 0}, {0, 0}};     // [row,col], Q16
  ia::WarpParams wmmat{};                            // reduced model + clamped t
};

// §7.12.2.4.  `adjust_col` is the `iter` argument of generate_points_from_corners.
bool warp_corner_process(const WarpGrid& grid, int sb_h4, int mi_row, int mi_col,
                         int bw4, int bh4, std::int8_t ref_frame0, int delta_row,
                         int delta_col, int adjust_col, CornerResult* out);

// §7.12.2.3.  When `stack` is non-null and the model is valid, the result is
// inserted with §7.12.2.11.  Returning the intermediate values keeps the
// process testable without a stack.
CornerResult generate_points_from_corners(const WarpGrid& grid, int sb_h4,
                                          int mi_row, int mi_col, int bw4,
                                          int bh4, std::int8_t ref_frame0,
                                          int iter, WarpStack* stack);

// §7.13.3.19 sub-block MV that a LOCALWARP block stores into SubMvs for the
// 8x8 section (i8, j8), covering MI cells (row + i8*2 + i, col + j8*2 + j).
// `x`/`y` are the block origin in luma samples.  The MV is (mv_y, mv_x) in
// 1/8 luma samples and is clipped to MV_LOW+1..MV_UPP-1 (no picture clamp).
inter::mc::Mv warp_sub_block_mv(const ia::WarpParams& params, int x, int y,
                                int i8, int j8);

// §7.12.2.4 helper.  `pos_row`/`pos_col` are MI coordinates; the returned MV is
// (mv_y, mv_x) in 1/8 luma samples.
inter::mc::Mv get_warp_motion_vector_xy_pos(const ia::WarpParams& params,
                                            int pos_row, int pos_col, int mi_row,
                                            int mi_col, int bw4, int bh4,
                                            int mi_rows, int mi_cols);

// ---------------------------------------------------------------------------
// Warp parameter bank (§7.12.2.20)
// ---------------------------------------------------------------------------
//
// A four-entry ring per reference slot.  The spec reads it newest first:
//   for (i = WarpBankSize[ref]-1; i >= 0; --i)
//       idx = (WarpBankStart[ref] + i) % WARP_PARAM_BANK_SIZE;
//
// The bank maintenance itself is a property of the frame/block loop (a
// LOCALWARP block registers its chosen model); `record()` is that hook.  Later
// writes overwrite the oldest entry.
class WarpParamBank {
 public:
  void reset();
  void record(const ia::WarpParams& params);
  int size() const { return size_; }
  int start() const { return start_; }
  // Physical slot 0..3.
  const ia::WarpParams& slot(int physical) const;
  // Logical entry `index` counted from the oldest (`index == 0`) to the newest
  // (`index == size()-1`).  §7.12.2.20 walks `size()-1` down to 0.
  const ia::WarpParams& at(int index) const {
    return slot((start_ + index) % kWarpParamBankSize);
  }

 private:
  ia::WarpParams params_[kWarpParamBankSize]{};
  int start_ = 0;
  int size_ = 0;
};

// ---------------------------------------------------------------------------
// §7.12.2 single-reference MV stack with the warp half
// ---------------------------------------------------------------------------

struct WarpStackParams {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_h4 = 16;  // 16 for a 64x64 superblock
  int mi_cols = 0;
  int mi_rows = 0;
  const WarpMi* mi = nullptr;

  // RefFrame[0] / RefFrame[1].  Single prediction: list 1 == kRefNone.
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
  bool is_compound = false;  // rejected (§7.12.2.9 only runs when !isCompound)

  bool derive_wrl = false;
  bool force_integer_mv = false;
  bool use_intrabc = false;
  // Temporal candidates: this builder implements the spatial subset only.
  bool use_ref_frame_mvs = false;

  DrlReorder reorder = DrlReorder::kDisabled;
  bool use_temporal_first = false;  // cannot be true without ref-frame MVs
  int max_drl_bits_minus_1 = 4;

  // GlobalMvs[0] (and [1] for a compound extension).
  inter::mc::Mv global_mv[2]{};
  // True when GmType[RefFrame[0]] > IDENTITY (get_mv() then returns
  // GlobalMvs instead of SubMvs for GLOBAL* candidate modes).
  bool global_mv_nonidentity = false;
  // gm_params[RefFrame[0]] for the §7.12.2.20 insertion.
  ia::WarpParams gm_params{};
  // Warp bank for RefFrame[0]; null means empty.
  const WarpParamBank* warp_bank = nullptr;

  WarpGrid grid() const { return WarpGrid{mi_cols, mi_rows, mi}; }
};

struct WarpCandidate {
  inter::mc::Mv mv[2];  // list 0 / list 1
  int weight = 0;
  int row_offset = 0;
  int col_offset = 0;
  std::int8_t cwp = kCwpEqual;
};

struct WarpStack {
  WarpCandidate cand[kMaxRefMvStackSize];
  int count = 0;
  int num_nearest = 0;
  bool sorted = false;
  ia::WarpParams warp_params[kMaxWarpRefCandidates];  // WarpParamStack
  int num_warp_found = 0;
};

// Builds the stack.  `derive_wrl == false` produces the same candidates,
// offsets and weights as the wired M2 stack; `derive_wrl == true` additionally
// runs §7.12.2.3/.4 (corner warp models) and §7.12.2.9/.11/.20 (warp stack).
Status build_warp_stack(const WarpStackParams& params, WarpStack* stack);

// §7.12.2.11.  No deduplication: the spec caps at four entries and appends.
void insert_warp_candidate(const ia::WarpParams& params, WarpStack* stack);

// §7.12.2.9.
void add_warp_motion_vector(const WarpGrid& grid, int mi_row, int mi_col,
                            std::int8_t ref_frame0, bool derive_wrl,
                            int mv_row, int mv_col, WarpStack* stack);

// §7.12.2.6 scan point (exposed for tests) and §7.12.2.5 scan col.
struct ScanState {
  int prune_count = 0;
  int stack_count = 0;
};
void scan_point(const WarpStackParams& params, int delta_row, int delta_col,
                WarpStack* stack, int* prune_count);
void scan_col(const WarpStackParams& params, int delta_col, WarpStack* stack,
              int* prune_count);

// §7.12.2.10 single-prediction subset.
void add_reference_mv(const WarpStackParams& params, int mv_row, int mv_col,
                      int weight, WarpStack* stack, int* prune_count);

// §7.12.2.19 sorting process: one move-to-front over [start, end).
void sort_stack_range(WarpStack* stack, int start, int end);

// §7.12.2.23 clamping process.
void clamp_stack(const WarpStackParams& params, WarpStack* stack);

// ---------------------------------------------------------------------------
// §7.12.3 find warp samples + add sample
// ---------------------------------------------------------------------------

struct WarpSampleSet {
  int num_samples = 0;
  // CandList[i] = {src_y, src_x, dst_y, dst_x} in 1/8 luma samples, exactly
  // the spec's cand[0..3].
  int cand[kLeastSquaresSamplesMax][4] = {};
};

struct WarpSampleParams {
  int mi_row = 0;
  int mi_col = 0;
  int w4 = 1;
  int h4 = 1;
  int mi_cols = 0;
  int mi_rows = 0;
  int sb_h4 = 16;
  WarpGrid grid{};
  // RefFrame[0..1] of the current block and which of them to sample.
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
  int ref = 0;  // 0 -> RefFrame[0], 1 -> RefFrame[1]
  bool avail_u = false;  // AvailU
  bool avail_l = false;  // AvailL
};

// §7.12.3. Returns true when at least one sample was produced.
bool find_warp_samples(const WarpSampleParams& params, WarpSampleSet* out);

// §7.12.3.2 add sample process.  `delta_row`/`delta_col` are MI offsets.
void add_warp_sample(const WarpSampleParams& params, int delta_row, int delta_col,
                     WarpSampleSet* out);

// §7.12.3 helper above_sample_stored(deltaCol).
bool above_sample_stored(const WarpSampleParams& params, int delta_col);

// Inputs for the full local-warp derivation of §7.13.3.23.
struct LocalWarpParams {
  int mi_row = 0;
  int mi_col = 0;
  int w4 = 1;
  int h4 = 1;
  int mi_cols = 0;
  int mi_rows = 0;
  int sb_h4 = 16;
  WarpGrid grid{};
  std::int8_t ref_frame[2] = {kRefNone, kRefNone};
  bool avail_u = false;
  bool avail_l = false;
  inter::mc::Mv block_mv{};  // BlockMvs[ref]
};

struct LocalWarpResult {
  ia::WarpParams params{};      // LocalWarpParams[ref]
  WarpSampleSet samples;
  bool det_zero = false;
};

// §7.12.3 + §7.13.3.23.  `samples` may be null.  Returns nullopt when the
// process cannot run (no samples and a degenerate block MV is still a valid
// translation model, so this only fails on invalid geometry).
std::optional<LocalWarpResult> derive_local_warp(const LocalWarpParams& params,
                                                 int ref);

// §7.13.3.23 least squares over the candidate list.  This is not
// adv::warp_estimate(): that helper narrows the normal-equation determinant to
// int, which changes the model for 64x64 blocks (det > 2^31).  The pixel side
// still uses adv::warp_predict()/warp_predict_ext().
ia::WarpParams estimate_warp_from_samples(const WarpSampleSet& samples, int mid_x,
                                          int mid_y, const int block_mv[2]);

// §7.13.3.23 `get_warp_translation` wrapper used by the tests.
ia::WarpParams warp_model_with_translation(ia::WarpParams params, int mid_x,
                                           int mid_y, const int block_mv[2]);

}  // namespace av2enc::inter::warp

#endif  // AV2ENC_INTER_WARP_STACK_WARP_STACK_H_
