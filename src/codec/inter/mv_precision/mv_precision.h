// AV2 v1.0.0 motion-vector precision and adaptive-MVD signalling.
//
// This module owns the block/frame/sequence syntax that the M2 baseline does
// not emit:
//
//   * §5.4.6  seq_enabled_motion_modes / seq_frame_motion_modes_present_flag /
//             enable_six_param_warp_delta
//   * §5.4.6  enable_adaptive_mvd / enable_mvd_sign_derive / enable_flex_mvres
//   * §5.18.2 frame_enabled_motion_modes, use_qtr_precision_mv /
//             allow_high_precision_mv, read_interpolation_filter()
//   * §5.20.7.6 use_amvd, use_most_probable_precision, pb_mv_precision,
//             interp_filter
//   * §5.20.7.13 assign_mv() joint-MVD projection (projMv)
//   * §5.20.7.20 AMVD mv_joint, §5.20.7.21 amvd_index, jmvd_scale_mode
//   * §5.20.7.13 / §5.20.7.20 mv_sign with enable_mvd_sign_derive
//
// The *precision model* (FrameMvPrecision -> per-block MvPrecision, the
// normative lower_mv_precision() rounding, and the context helpers) lives in
// `src/encoder/inter/me2/me2_precision.h`.  This module consumes that header
// read-only and adds the bitstream layer on top of it, so the two cannot drift.
//
// Writer and reader are deliberately separate functions.  Sharing an
// "encode/decode" helper would hide a symmetric off-by-one; the unit test
// drives the two against each other, against pinned goldens, and finally
// through `dav2d` in the gate.
//
// CDF selection (§8.3):
//   use_amvd                  TileUseAmvdCdf[index][ctx]
//   mv_joint                  TileMvJointAdaptiveCdf
//   amvd_index                TileAmvdIndicesCdf[comp]
//   jmvd_scale_mode           TileJmvd[Adaptive]ScaleModeCdf
//   use_most_probable_precision  TileUseMostProbablePrecisionCdf[ctx]
//   pb_mv_precision           TilePbMvPrecisionCdf[ctx][FrameMvPrecision-HALF]
//   interp_filter             TileInterpFilterCdf[ctx]
//
// The state is one copy per tile: the AV2 entropy coder mutates a CDF after
// every symbol, and tiles must not share state.

#ifndef AV2ENC_ENCODER_INTER_MV_PRECISION_MV_PRECISION_H_
#define AV2ENC_ENCODER_INTER_MV_PRECISION_MV_PRECISION_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "core/bitstream/bitwriter.h"
#include "codec/inter/mc/mc_types.h"
#include "codec/inter/me_precision/me2_precision.h"
#include "codec/inter/mv_precision/mv_precision_cdf_tables.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::inter::mvprec {

// ---------------------------------------------------------------------------
// A small MSB-first bit reader for the raw f(n)/L(n) fields below.
//
// The entropy-coded symbols use av2::Decoder; this reader exists so the frame
// and sequence helpers can be tested without a tile payload.
// ---------------------------------------------------------------------------

class BitReader {
 public:
  BitReader(std::span<const std::uint8_t> bytes, std::size_t bit_offset = 0)
      : bytes_(bytes), bit_pos_(bit_offset) {}

  bool read_bit(bool* bit) {
    if (bit == nullptr || bit_pos_ >= bytes_.size() * 8U) return false;
    *bit = ((bytes_[bit_pos_ >> 3U] >> (7U - (bit_pos_ & 7U))) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool read_bits(int count, std::uint32_t* value) {
    if (value == nullptr || count < 0 || count > 32) return false;
    std::uint32_t out = 0;
    for (int i = 0; i < count; ++i) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      out = (out << 1U) | (bit ? 1U : 0U);
    }
    *value = out;
    return true;
  }

  std::size_t bit_pos() const { return bit_pos_; }
  std::size_t bits_left() const { return bytes_.size() * 8U - bit_pos_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

// ---------------------------------------------------------------------------
// §5.4.6 / §5.18.2 motion modes.
// ---------------------------------------------------------------------------

enum MotionMode : std::int32_t {
  kMotionSimple = 0,
  kMotionInterIntra = 1,
  kMotionLocalWarp = 2,
  kMotionDeltaWarp = 3,
  kMotionExtendWarp = 4,
  kMotionModes = 5,
};

struct MotionModeSet {
  // Indexed by MotionMode.  SIMPLE is never coded; it is kept so a caller can
  // round-trip the whole array without a second representation.
  bool enabled[kMotionModes] = {false, false, false, false, false};

  bool any() const {
    for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
      if (enabled[mode]) return true;
    }
    return false;
  }
  bool operator==(const MotionModeSet& other) const {
    for (int mode = 0; mode < kMotionModes; ++mode) {
      if (enabled[mode] != other.enabled[mode]) return false;
    }
    return true;
  }
};

struct SequenceMotionModeSyntax {
  MotionModeSet seq_enabled;
  // Derived by the writer from seq_enabled; on read it is returned as coded.
  bool seq_frame_motion_modes_present_flag = false;
  // Present only when seq_enabled[DELTAWARP] is set.
  bool enable_six_param_warp_delta = false;
};

// Writes one `sequence_inter_config()` motion-mode slice (§5.4.6), in order:
//   seq_enabled_motion_modes[INTERINTRA..EXTENDWARP]
//   seq_frame_motion_modes_present_flag      (iff any mode enabled)
//   enable_six_param_warp_delta              (iff DELTAWARP enabled)
bool write_sequence_motion_modes(BitWriter& writer,
                                 const SequenceMotionModeSyntax& syntax,
                                 std::string* error);
bool read_sequence_motion_modes(BitReader& reader,
                                SequenceMotionModeSyntax* syntax,
                                std::string* error);

// Writes the §5.18.2 per-frame loop.  When
// seq_frame_motion_modes_present_flag == 0 the frame values are inferred from
// the sequence and no bits are coded; when it is 1 only the sequence-enabled
// modes code a bit.  `frame` modes that the sequence disabled are forced 0.
bool write_frame_motion_modes(BitWriter& writer,
                              const SequenceMotionModeSyntax& sequence,
                              const MotionModeSet& frame,
                              std::string* error);
bool read_frame_motion_modes(BitReader& reader,
                             const SequenceMotionModeSyntax& sequence,
                             MotionModeSet* frame, std::string* error);

// Sequence-level interaction flags (§5.4.6), coded in this order.
struct SequenceMvResSyntax {
  bool enable_adaptive_mvd = false;
  bool enable_mvd_sign_derive = false;
  bool enable_flex_mvres = false;
};

bool write_sequence_mv_res(BitWriter& writer,
                           const SequenceMvResSyntax& syntax,
                           std::string* error);
bool read_sequence_mv_res(BitReader& reader, SequenceMvResSyntax* syntax,
                          std::string* error);

// ---------------------------------------------------------------------------
// §5.18.2 frame MV precision.
//
// FrameMvPrecision derivation is implemented by me2
// (`me2_frame_precision_from_bits` / `me2_frame_precision_to_bits`); this
// module only writes and reads the two header bits.
// ---------------------------------------------------------------------------

// `force_integer_mv` comes from screen_content_params().  It removes both bits
// and pins FrameMvPrecision to MV_PRECISION_ONE_PEL.
bool write_frame_mv_precision(BitWriter& writer,
                              const me2::FramePrecisionSyntax& syntax,
                              int force_integer_mv, std::string* error);
bool read_frame_mv_precision(BitReader& reader, int force_integer_mv,
                             int enable_flex_mvres,
                             me2::FramePrecisionSyntax* syntax,
                             std::string* error);

// §5.18.5.1 read_interpolation_filter().  `filter` is 0..3 (EIGHTTAP,
// EIGHTTAP_SMOOTH, EIGHTTAP_SHARP, BILINEAR).  A switchable frame codes one
// bit and leaves `filter` at SWITCHABLE (4).
inline constexpr int kInterpSwitchable = 4;

bool write_frame_interp_filter(BitWriter& writer, bool is_switchable,
                               int filter, std::string* error);
bool read_frame_interp_filter(BitReader& reader, bool* is_switchable,
                              int* filter, std::string* error);

// ---------------------------------------------------------------------------
// §5.20.7.6 YMode values needed by AMVD / flex MV resolution.
// ---------------------------------------------------------------------------

enum YMode : std::int32_t {
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

bool allow_amvd_mode(int y_mode);
bool has_newmv(int y_mode);
bool has_nearmv(int y_mode);
bool is_joint_mvd_coding_mode(int y_mode);

// §5.20.7.20 constants.
inline constexpr int kMvJoints = 4;
inline constexpr int kMvJointZero = 0;
inline constexpr int kMvJointHnzvz = 1;   // column only
inline constexpr int kMvJointHzvnz = 2;   // row only
inline constexpr int kMvJointHnzvnz = 3;  // both

inline constexpr int kMaxAmvdIndex = 8;
inline constexpr std::int32_t kAmvdIndexToMvd[kMaxAmvdIndex] = {
    2, 4, 6, 8, 16, 32, 64, 128};

inline constexpr int kJointNewMvScaleFactorCnt = 5;
inline constexpr int kJointAmvdScaleFactorCnt = 3;

// ---------------------------------------------------------------------------
// §9.3 CDF state, one independent copy per tile.
// ---------------------------------------------------------------------------

struct MvPrecisionCdfState {
  // TileUseAmvdCdf[9][3][2]
  std::uint16_t use_amvd[9][3][3];
  // TileMvJointAdaptiveCdf
  std::uint16_t mv_joint[5];
  // TileAmvdIndicesCdf[comp][MAX_AMVD_INDEX]
  std::uint16_t amvd_index[2][9];
  // TileJmvdScaleModeCdf / TileJmvdAdaptiveScaleModeCdf
  std::uint16_t jmvd_scale_mode[6];
  std::uint16_t jmvd_adaptive_scale_mode[4];
  // TileUseMostProbablePrecisionCdf[3][2]
  std::uint16_t use_most_probable_precision[3][3];
  // TilePbMvPrecisionCdf[2][3][3]
  std::uint16_t pb_mv_precision[2][3][4];
  // TileInterpFilterCdf[16][3]
  std::uint16_t interp_filter[16][4];

  // Shell-coded MVD tables, used by the gate's flex-precision blocks.
  std::uint16_t shell_set[3];
  std::uint16_t shell_last_two[3];
  std::uint16_t shell_offset_low[2][3];
  std::uint16_t shell_offset_class2[3];
  std::uint16_t shell_offset_other[16][3];
  std::uint16_t col_mv_greater[2][3];
  std::uint16_t col_mv_index[4][3];
  // Indexed [precision][shell_set]; only precisions 0,1,3,4,5,6 exist.
  std::uint16_t shell_class[7][2][9];
};

// Copies every §9.3 default.  Call once per tile before the first block.
void init_default_mv_precision_cdfs(MvPrecisionCdfState* state);

// ---------------------------------------------------------------------------
// §8.3 neighbour contexts.
//
// The spec distinguishes two neighbour lists:
//   NPos/NNum     same superblock row only  (precision contexts)
//   NPosBuf/NNumBuf  may cross the SB row    (AMVD context, interp filter)
// `NeighborSet` carries both so a caller cannot accidentally use the wrong
// one.
// ---------------------------------------------------------------------------

struct NeighborMvInfo {
  bool available = false;
  // INTRA_FRAME = -2, NONE = -1, TIP_FRAME = -3, otherwise a ref slot.
  std::int8_t ref[2] = {-1, -1};
  bool uses_amvd = false;
  bool use_most_probable_precision = false;
  int mv_precision = me2::kMvPrecisionEighthPel;
  int y_mode = kYModeNearMv;
  // InterpFilters value; 3 means "unavailable" in the §8.3 formula.
  int interp_filter = 3;
};

struct NeighborSet {
  NeighborMvInfo npos[2];
  int num_npos = 0;
  NeighborMvInfo nbuf[2];
  int num_nbuf = 0;
};

// §5.20.7.6 order: add_neighbor() with the NPos
// same-superblock-row restriction for `npos`, unrestricted for `nbuf`.
// Positions are in 4x4 units; `mi_row`/`mi_col` are the current block origin,
// `bw4`/`bh4` its size, `sb_h4` the superblock height in MI units (16 for a
// 64x64 SB).  `mi` is the row-major neighbour grid; entries whose `available`
// is false are ignored.  `mi_cols`/`mi_rows` bound the frame.
NeighborSet collect_neighbors(const NeighborMvInfo* mi, int mi_cols,
                              int mi_rows, int mi_row, int mi_col, int bw4,
                              int bh4, int sb_h4);

// TileUseAmvdCdf index from §8.3.
int use_amvd_cdf_index(int y_mode, bool use_optflow);

// TileUseAmvdCdf ctx: number of NPosBuf neighbours that carry RefFrame[0] and
// have UsesAmvds == 1.
int use_amvd_ctx(const NeighborSet& neighbours, int ref0);

// TileUseMostProbablePrecisionCdf ctx: NPos neighbours whose flag is set.
int use_most_probable_precision_ctx(const NeighborSet& neighbours);

// TilePbMvPrecisionCdf ctx: 1 if any NPos neighbour has a coarser (numerically
// smaller) MvPrecision than FrameMvPrecision.
int pb_mv_precision_ctx(const NeighborSet& neighbours, int frame_precision);

// TileInterpFilterCdf ctx (§8.3), 0..15.  `is_compound` sets the base 4.
int interp_filter_ctx(const NeighborSet& neighbours, int ref0,
                      bool is_compound);

// §5.20.7.6 needs_interp_filter().
bool needs_interp_filter(bool large, int y_mode, int motion_mode);

// Legal block-level interp_filter values.  BILINEAR (3) is a frame-header
// value and is not in TileInterpFilterCdf's three-symbol alphabet.
bool block_interp_filter_legal(int filter);
bool frame_interp_filter_legal(int filter, bool is_switchable);

// ---------------------------------------------------------------------------
// Per-block precision plan (§5.20.7.6).
// ---------------------------------------------------------------------------

struct MvResConfig {
  bool enable_adaptive_mvd = false;
  bool enable_mvd_sign_derive = false;
  bool enable_flex_mvres = false;
};

// Checks the three sequence flags and documents the interactions that the
// precision plan relies on.  Every combination of the three flags is legal
// AV2; the function returns false only for an internally impossible state
// (none today), and exists so a future core integration has one place to add
// encoder-side policy.
bool validate_mv_res_config(const MvResConfig& config, std::string* error);

struct BlockPrecisionSyntax {
  // IsAdaptiveMvd = enable_adaptive_mvd && use_amvd.
  bool is_adaptive_mvd = false;
  // Present iff enable_flex_mvres && UsePerBlockMvPrecision && has_newmv.
  bool code_use_most_probable_precision = false;
  bool use_most_probable_precision = true;
  bool code_pb_mv_precision = false;
  int pb_mv_precision = -1;
  // TilePbMvPrecisionCdf row = FrameMvPrecision - MV_PRECISION_HALF_PEL.
  int pb_cdf_row = 0;
  // The resolved MvPrecision for this block.
  int mv_precision = me2::kMvPrecisionEighthPel;
};

// `frame` must already carry the header-derived FrameMvPrecision and
// UsePerBlockMvPrecision.  `target_precision` is the desired block precision;
// when the block cannot signal it this returns false rather than silently
// emitting a different precision.
bool plan_block_precision(const MvResConfig& config,
                          const me2::FramePrecisionSyntax& frame,
                          int y_mode, bool use_amvd, int target_precision,
                          BlockPrecisionSyntax* plan, std::string* error);

bool write_block_precision(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                           int most_probable_ctx, int pb_ctx,
                           const BlockPrecisionSyntax& plan,
                           std::string* error);
bool read_block_precision(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                          int most_probable_ctx, int pb_ctx,
                          const me2::FramePrecisionSyntax& frame,
                          int y_mode, bool use_amvd,
                          bool enable_adaptive_mvd, bool enable_flex_mvres,
                          BlockPrecisionSyntax* plan, std::string* error);

// ---------------------------------------------------------------------------
// §5.20.7.6 / §5.20.7.20 / §5.20.7.21 AMVD symbols.
// ---------------------------------------------------------------------------

bool write_use_amvd(av2::Encoder& enc, MvPrecisionCdfState& cdfs, int index,
                    int ctx, bool value, std::string* error);
bool read_use_amvd(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int index,
                   int ctx, bool* value, std::string* error);

bool write_mv_joint(av2::Encoder& enc, MvPrecisionCdfState& cdfs, int joint,
                    std::string* error);
bool read_mv_joint(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int* joint,
                   std::string* error);

bool write_amvd_index(av2::Encoder& enc, MvPrecisionCdfState& cdfs, int comp,
                      int index, std::string* error);
bool read_amvd_index(av2::Decoder& dec, MvPrecisionCdfState& cdfs, int comp,
                     int* index, std::string* error);

// One AMVD MVD as read by read_mv(): mv_joint plus one amvd_index per nonzero
// component.  Components are magnitudes in 1/8-sample units and may differ.
struct AmvdMvSyntax {
  int joint = kMvJointZero;
  int index[2] = {0, 0};
  std::int32_t magnitude[2] = {0, 0};
};

// Inverse table lookup for one component.  Returns false when `value` is not
// in Amvd_Index_To_Mvd (an AMVD block cannot signal any other magnitude).
bool amvd_index_for_magnitude(std::int32_t value, int* index);

// Selects the legal (mv_joint, index, magnitude) triple for `mvd`.
bool amvd_mv_from_mv(mc::Mv mvd, AmvdMvSyntax* syntax, std::string* error);

bool write_amvd_mv(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                   const AmvdMvSyntax& syntax, std::string* error);
bool read_amvd_mv(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                  AmvdMvSyntax* syntax, std::string* error);

bool write_jmvd_scale_mode(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                           bool use_amvd, int mode, std::string* error);
bool read_jmvd_scale_mode(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                          bool use_amvd, int* mode, std::string* error);

// ---------------------------------------------------------------------------
// §5.20.7.13 joint-MVD projection (projMv).
// ---------------------------------------------------------------------------

struct JointMvdPlan {
  int base_list = 0;
  int first_distance = 0;
  int second_distance = 0;
};

// assign_mv()'s pre-loop selection.  `distance0/1` are FrameDistance[] values
// (signed); `restricted0/1` mark the RESTRICTED_OH order hint.
JointMvdPlan joint_mvd_plan(int distance0, int distance1, bool restricted0,
                            bool restricted1);

// §7.9.4 get_mv_projection(), numerator = second_distance, denominator =
// first_distance.  Inputs/outputs are 1/8-sample units.
mc::Mv project_joint_mvd(mc::Mv diff, int second_distance,
                         int first_distance);

// The scaling applied to projMv in assign_mv():
//   use_amvd: jmvd_scale_mode 1 -> *2, 2 -> /2 (whole vector)
//   otherwise: mode>0 scales one component ((mode-1)&1) by *2 (mode<=2) or /2
mc::Mv scale_joint_projection(mc::Mv proj, int jmvd_scale_mode, bool use_amvd);

// Computes BlockMvs for JOINT_NEWMV.  `diff` is the MVD of the base list;
// `pred` are PredMvs[list] in 1/8 units.  Applies mv_clamp_to_integer().
void assign_joint_mvd(mc::Mv pred[2], mc::Mv diff, const JointMvdPlan& plan,
                      int jmvd_scale_mode, bool use_amvd, mc::Mv out[2]);

// ---------------------------------------------------------------------------
// §5.20.7.13 mv_sign and enable_mvd_sign_derive.
// ---------------------------------------------------------------------------

struct MvdSignConfig {
  bool use_intrabc = false;
  bool enable_mvd_sign_derive = false;
  int motion_mode = 0;  // SIMPLE == 0
  bool is_adaptive_mvd = false;
  bool skip_mode = false;
  bool allow_screen_content_tools = false;
  int frame_precision = me2::kMvPrecisionEighthPel;
  int mv_precision = me2::kMvPrecisionEighthPel;
  int y_mode = kYModeNewMv;
  bool is_compound = false;
  int ref_mv_idx = 0;
};

bool mvd_sign_derive_allowed(const MvdSignConfig& config);

// Number of mv_sign L(1) bits emitted for these magnitudes.  With derivation
// enabled the last nonzero component is omitted.
int mvd_sign_symbol_count(const MvdSignConfig& config, const mc::Mv mvd[2],
                          int lists);

// Writes the sign bits.  When the last sign is derived it must equal
// `lastSign & 1`; otherwise this returns false (the caller would emit a
// different MV than it predicted).
bool write_mvd_signs(av2::Encoder& enc, const MvdSignConfig& config,
                     const mc::Mv mvd[2], int lists, std::string* error);

// Reads the sign bits into `mvd` (which arrives with unsigned magnitudes).
bool read_mvd_signs(av2::Decoder& dec, const MvdSignConfig& config,
                    mc::Mv mvd[2], int lists, std::string* error);

// ---------------------------------------------------------------------------
// §5.20.7.6 switchable interp_filter.
// ---------------------------------------------------------------------------

bool write_block_interp_filter(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                               int ctx, int filter, std::string* error);
bool read_block_interp_filter(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                              int ctx, int* filter, std::string* error);

// ---------------------------------------------------------------------------
// Generic shell-coded MVD (§5.20.7.20), precision-aware.
//
// M2 ships only the 1/8-pel writer.  A flex-precision block codes shell
// symbols in its own precision units, so the gate and the tests need the
// general form.
// ---------------------------------------------------------------------------

// `dy`/`dx` are in 1/8-sample units and must be divisible by the precision
// radix.  Signs are not written here; use write_mvd_signs() after this call,
// matching assign_mv()'s mv_sign loop.
bool write_shell_mv(av2::Encoder& enc, MvPrecisionCdfState& cdfs,
                    std::int32_t dy, std::int32_t dx, int precision,
                    std::string* error);
bool read_shell_mv(av2::Decoder& dec, MvPrecisionCdfState& cdfs,
                   int precision, mc::Mv* mvd, std::string* error);

// §5.20.7.13 mv_clamp_to_integer().
std::int32_t mv_clamp_to_integer(std::int32_t value);

}  // namespace av2enc::inter::mvprec

#endif  // AV2ENC_ENCODER_INTER_MV_PRECISION_MV_PRECISION_H_
