#ifndef AV2ENC_VIDEO_ADVANCED_SYNTAX_H_
#define AV2ENC_VIDEO_ADVANCED_SYNTAX_H_

// AV2 v1.0.0 advanced-inter block syntax for the M2 encoder.
//
// This layer owns the *bit-level* symbols of the three advanced inter tools
// that the M2 baseline disables:
//
//   * warped motion / affine (§5.20.7.1–.7, §5.20.7.14, §7.12.2.9–.11,
//     §7.13.3.19–.24): is_warp, warp_mv, warp_idx, warpmv_with_mvd,
//     warp_delta_precision, warp_delta_param_{low,high,sign}, and the
//     extended/local motion-mode bits.
//   * inter-intra (§5.20.7.14, §5.20.7.15, §7.13.3.29–.30): inter_intra /
//     warp_inter_intra, interintra_mode, wedge_interintra and the
//     read_wedge_mode() symbols.
//   * BAWP (§5.20.7.6, §7.13.3.25): use_bawp, explicit_bawp,
//     explicit_bawp_scale and use_bawp_chroma.
//
// Every writer is the exact inverse of the normative syntax.  The matching
// readers exist so the unit tests can round-trip through the normative
// `av2::Decoder` state machine (§8.2), not through a private coder.
//
// AV2 has no OBMC motion mode (MOTION_MODES is SIMPLE, INTERINTRA, LOCALWARP,
// DELTAWARP, EXTENDWARP).  Nothing here can signal or decode OBMC.
//
// The caller owns the `av2::Encoder`/`av2::Decoder` and one `AdvancedCdfState`
// per tile.  The M2 tile writer keeps its own CDF struct; a conforming
// integration either adds these arrays to that struct or calls the staged
// writers below with this state.  `AdvancedCdfState::reset()` copies the §9.3
// defaults, which is what `disable_cross_frame_cdf_init = 1` needs.

#include "core/bitstream/bitwriter.h"
#include "codec/inter/advanced/bawp.h"
#include "codec/inter/advanced/compound.h"
#include "codec/inter/advanced/tables/wedge_tables.h"
#include "codec/inter/advanced/warp.h"
#include "codec/inter/advanced/video/advanced_cdf_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::video::adv {

namespace ia = inter::adv;
namespace wt = inter::adv::tables;

// §6.19.7.5 Table: YMode values used by this layer.
enum class YMode : std::uint8_t {
  kNearMv = 14,
  kGlobalMv = 15,
  kNewMv = 16,
  kWarpMv = 17,
  kWarpNewMv = 18,
};

// §9.2 constants that the syntax needs locally.
inline constexpr int kBlockSizes = 29;
inline constexpr int kMiSize = 4;
inline constexpr int kWarpDeltaNumSymbolsLow = 8;
inline constexpr int kWarpDeltaNumSymbolsHigh = 8;
inline constexpr int kWarpDeltaStepBits = 10;
inline constexpr int kMaxWarpRefCandidates = 4;
inline constexpr int kMotionModes = 5;

// Reuse the generated §9.2 tables rather than duplicating them.
inline constexpr const std::int8_t (&kSizeGroup)[kBlockSizes] = cdf::kSizeGroup;
inline constexpr const std::int8_t (&kWedgeBits)[kBlockSizes] = wt::kWedgeBits;

// ---------------------------------------------------------------------------
// §9.3 CDF state for one tile.
// ---------------------------------------------------------------------------

struct AdvancedCdfState {
  std::uint16_t is_warp[5][3];
  std::uint16_t warp_mv[3];
  std::uint16_t warp_idx[3][3];
  std::uint16_t warp_with_mvd[3];
  std::uint16_t warp_precision[kBlockSizes][3];
  std::uint16_t warp_delta_low[2][9];
  std::uint16_t warp_delta_high[2][9];
  std::uint16_t warp_delta_sign[3];
  std::uint16_t inter_intra[4][3];
  std::uint16_t warp_inter_intra[4][3];
  std::uint16_t inter_intra_mode[4][5];
  std::uint16_t wedge_inter_intra[3];
  std::uint16_t wedge_quad[5];
  std::uint16_t wedge_angle[4][6];
  std::uint16_t wedge_dist1[5];
  std::uint16_t wedge_dist2[4];
  std::uint16_t use_bawp[3];
  std::uint16_t use_bawp_chroma[3];
  std::uint16_t explicit_bawp[3][3];
  std::uint16_t explicit_bawp_scale[3];
  std::uint16_t use_local_warp[4][3];
  std::uint16_t use_extend_warp[3][3];

  // Copies the §9.3 defaults. Call once per tile before the first block when
  // `primary_ref_frame == PRIMARY_REF_NONE || disable_cross_frame_cdf_init`.
  void reset();
};

// ---------------------------------------------------------------------------
// §7.11.2 / §7.11.4 neighbour context.
// ---------------------------------------------------------------------------

// One decoded/written mode-info cell.  `written = false` models the spec's
// "RefFrames[][] has been written for this frame" guard.  RefFrame values are
// INTRA_FRAME = -2, NONE = -1, TIP_FRAME = -3, otherwise 0..NUM_REF_FRAMES-1.
struct NeighborMi {
  bool written = false;
  bool is_inter = false;
  int ref[2] = {-1, -1};
  int y_mode = static_cast<int>(YMode::kGlobalMv);
  int motion_mode = 0;       // MotionModes[] 0..4
  int interp_filter = 0;
};

struct NeighborGrid {
  int mi_cols = 0;
  int mi_rows = 0;
  const NeighborMi* mi = nullptr;  // row-major, mi_rows * mi_cols

  bool inside(int row, int col) const {
    return mi != nullptr && row >= 0 && col >= 0 && row < mi_rows && col < mi_cols;
  }
  const NeighborMi& at(int row, int col) const {
    return mi[static_cast<std::size_t>(row) * static_cast<std::size_t>(mi_cols) +
              static_cast<std::size_t>(col)];
  }
};

struct ModeContext {
  int warp_mv_count = 0;
  bool warp_sample_found[2] = {false, false};
  int extend_delta_row = 0;
  int extend_delta_col = 0;
  int local_warp_ctx = 0;
  int extend_warp_ctx = 0;
  int npos_count = 0;
  int npos[2][2] = {{0, 0}, {0, 0}};
  int local_warp_ctx_npos = 0;
  int extend_warp_ctx_npos = 0;
};

// §7.11.2 find_mode_ctx + §7.11.4 scan_point_warp_ctx.
// `sb_mi_size` is Num_4x4_Blocks_High[SbSize] (16 for a 64x64 SB).
// `ref0`/`ref1` are the block's RefFrame[0]/RefFrame[1] (NONE = -1).
ModeContext compute_mode_context(const NeighborGrid& grid, int mi_row, int mi_col,
                                 int bsize, int sb_mi_size, int ref0, int ref1);

// Context-only helpers for integrations that already track the neighbour data
// in another layout.
int warp_mv_context(const NeighborGrid& grid, int mi_row, int mi_col, int bsize,
                    int sb_mi_size, int ref0, int ref1);

// ---------------------------------------------------------------------------
// §§5.20.7.14–.15 motion mode / inter-intra / wedge plans.
// ---------------------------------------------------------------------------

inline bool has_newmv(YMode mode) {
  return mode == YMode::kNewMv || mode == YMode::kWarpNewMv;
}
inline bool has_nearmv(YMode mode) { return mode == YMode::kNearMv; }
inline bool is_warp_mode(YMode mode) {
  return mode == YMode::kWarpMv || mode == YMode::kWarpNewMv;
}

// read_warp_delta_param() symbol values. `magnitude[i]` is the unsigned value
// before the sign bit; index 0..3 maps to warp parameters 2..5.
struct WarpDeltaSymbols {
  bool high_precision = false;
  int magnitude[4] = {0, 0, 0, 0};  // 0..7 standard, 0..14 high precision
  bool negative[4] = {false, false, false, false};
};

// read_wedge_mode() symbols for one WedgeIndex.
struct WedgeModeSymbols {
  int wedge_index = 0;
  int wedge_quad = 0;
  int wedge_angle = 0;
  int wedge_dist1 = 0;
  int wedge_dist2 = 0;
  bool use_dist2 = false;
};

// Bijection with `Wedge_Angle_Dist_2_Index` (§5.20.7.18).  Returns false when
// `wedge_index` is not signallable (68..127, or -1).
bool wedge_mode_symbols(int wedge_index, WedgeModeSymbols* out);
bool wedge_index_from_symbols(const WedgeModeSymbols& symbols, int* wedge_index);

struct BawpSymbols {
  bool use_bawp = false;
  bool explicit_bawp = false;
  bool explicit_bawp_scale = false;
  bool use_bawp_chroma = false;
  // (YMode == NEARMV) ? 0 : (YMode == NEWMV && use_amvd ? 1 : 2)
  int explicit_ctx = 2;
};

// ---------------------------------------------------------------------------
// Stage writers.  They emit only the advanced symbols; the M2 base writer keeps
// is_inter/skip/single_mode/DRL/MV.  The comments give the exact insertion
// point in `inter_block_mode_info()`.
// ---------------------------------------------------------------------------

// `is_warp` is present iff allow_warpmv_mode && min(w,h) >= 8.
bool write_is_warp(av2::Encoder& enc, AdvancedCdfState& cdfs, int warp_mv_ctx,
                   bool value, std::string* error);

// `warp_mv` is present only when is_warp == 1 and !force_integer_mv.
bool write_warp_mv(av2::Encoder& enc, AdvancedCdfState& cdfs, bool value,
                   std::string* error);

// BAWP flags, immediately after `single_mode` (§5.20.7.6).
bool write_bawp_symbols(av2::Encoder& enc, AdvancedCdfState& cdfs,
                        const BawpSymbols& symbols, bool has_chroma,
                        std::string* error);

// read_motion_mode(): exactly one of the extended/local/inter-intra paths.
// `is_warpmv`/`is_warp_newmv` select the WARPMV / WARP_NEWMV branches.
// `inter_intra_allowed`, `local_allowed`, `extend_allowed` mirror
// AllowedMotionModes[] && frame_enabled_motion_modes[].
struct MotionModeSymbols {
  bool is_warp_newmv = false;
  bool extend_allowed = false;
  bool local_allowed = false;
  bool use_extend_warp = false;
  bool use_local_warp = false;
  int extend_warp_ctx = 0;  // §8.3.2 TileUseExtendWarpCdf[ctx]
  int local_warp_ctx = 0;   // §8.3.2 TileUseLocalWarpCdf[ctx]
  // Inter-intra payload (only when !is_warp_newmv and allowed).
  bool inter_intra_allowed = false;
  bool inter_intra = false;
  int interintra_mode = 0;
  bool wedge_allowed = false;
  bool wedge_interintra = false;
  int wedge_index = 0;
};

bool write_motion_mode(av2::Encoder& enc, AdvancedCdfState& cdfs, int mi_size,
                       const MotionModeSymbols& symbols, std::string* error);

// `warp_idx` loop.  `ref_warp_idx` is 0..3; 3 encodes "no candidate" (three
// one-bits).  Present for YMode == WARPMV or motion_mode == DELTAWARP.
bool write_warp_idx(av2::Encoder& enc, AdvancedCdfState& cdfs, int ref_warp_idx,
                    std::string* error);

// `warpmv_with_mvd` (only YMode == WARPMV && RefWarpIdx < 2).
bool write_warpmv_with_mvd(av2::Encoder& enc, AdvancedCdfState& cdfs,
                           bool value, std::string* error);

// `read_warp_delta()` (only motion_mode == DELTAWARP and YMode == WARP_NEWMV).
// `six_param` must equal enable_six_param_warp_delta && warp_idx == 1.
bool write_warp_delta(av2::Encoder& enc, AdvancedCdfState& cdfs, int mi_size,
                      const WarpDeltaSymbols& symbols, bool six_param,
                      std::string* error);

// `read_interintra_mode(isWarp)` when YMode == WARPMV (isWarp = 1) or the
// INTERINTRA branch of read_motion_mode() (isWarp = 0).  When
// `symbols.inter_intra` is false nothing else is written.
bool write_inter_intra(av2::Encoder& enc, AdvancedCdfState& cdfs, bool is_warp,
                       int mi_size, const MotionModeSymbols& symbols,
                       std::string* error);

// read_wedge_mode() without the trailing wedge_sign bit.
bool write_wedge_mode(av2::Encoder& enc, AdvancedCdfState& cdfs,
                      const WedgeModeSymbols& symbols, std::string* error);

// L(1) wedge_sign.  Inter-intra forces 0 and never calls this.
bool write_wedge_sign(av2::Encoder& enc, bool sign, std::string* error);

// ---------------------------------------------------------------------------
// Staged *readers* — exact inverses, used by the round-trip tests.
// ---------------------------------------------------------------------------

bool read_is_warp(av2::Decoder& dec, AdvancedCdfState& cdfs, int warp_mv_ctx,
                  bool* value, std::string* error);
bool read_warp_mv(av2::Decoder& dec, AdvancedCdfState& cdfs, bool* value,
                  std::string* error);
bool read_bawp_symbols(av2::Decoder& dec, AdvancedCdfState& cdfs,
                       int explicit_ctx, bool has_chroma, BawpSymbols* out,
                       std::string* error);
bool read_motion_mode(av2::Decoder& dec, AdvancedCdfState& cdfs, int size_group,
                      bool is_warp_newmv, bool extend_allowed, bool local_allowed,
                      int extend_warp_ctx, int local_warp_ctx,
                      bool inter_intra_allowed, bool wedge_allowed,
                      MotionModeSymbols* out, std::string* error);
bool read_warp_idx(av2::Decoder& dec, AdvancedCdfState& cdfs, int* ref_warp_idx,
                   std::string* error);
bool read_warpmv_with_mvd(av2::Decoder& dec, AdvancedCdfState& cdfs, bool* value,
                          std::string* error);
bool read_warp_delta(av2::Decoder& dec, AdvancedCdfState& cdfs, int mi_size,
                     bool six_param, WarpDeltaSymbols* out, std::string* error);
bool read_inter_intra(av2::Decoder& dec, AdvancedCdfState& cdfs, bool is_warp,
                      int mi_size, bool wedge_allowed, bool* inter_intra,
                      int* interintra_mode, bool* wedge_interintra,
                      int* wedge_index, std::string* error);
bool read_wedge_mode(av2::Decoder& dec, AdvancedCdfState& cdfs,
                     WedgeModeSymbols* out, std::string* error);
bool read_wedge_sign(av2::Decoder& dec, int* sign, std::string* error);


// ---------------------------------------------------------------------------
// Full ordered block writer/reader.
//
// This is the exact §5.20.7.6 ordering for a single-reference inter block.  The
// advanced symbols are interleaved with the M2-owned baseline symbols through
// callbacks, so an integration can pass its existing `code_single_mode`,
// `code_drl_index` and `write_mv` implementations without duplicating them.
//
// Call order (the numbers are the syntax order):
//   1  is_warp (if allow_warpmv_mode && min(w,h) >= 8)
//   2  warp_mv (if is_warp && !force_integer_mv)
//   3  single_mode (if !is_warp) then use_bawp/explicit/scale/chroma
//   4  use_extend_warp / use_local_warp / inter_intra payload
//   5  warp_idx loop (YMode == WARPMV || motion_mode == DELTAWARP)
//   6  warpmv_with_mvd (YMode == WARPMV && RefWarpIdx < 2)
//   7  drl_idx (NEARMV / NEWMV / WARP_NEWMV)
//   8  MVD (NEWMV / WARP_NEWMV / WARPMV + MVD)
//   9  warp_delta (motion_mode == DELTAWARP && YMode == WARP_NEWMV)
//  10  warp_inter_intra + inter-intra payload (YMode == WARPMV)
//
// YMode is derived: is_warp ? (warp_mv ? WARPMV : WARP_NEWMV)
//                            : NEARMV + single_mode.
// Motion mode is derived from the emitted bits; the writer validates the plan.

struct BaselineInterWriter {
  void* ctx = nullptr;
  bool (*single_mode)(void* ctx, int new_mv_ctx, int symbol) = nullptr;
  bool (*drl_index)(void* ctx, int new_mv_ctx, int index) = nullptr;
  bool (*mvd)(void* ctx, int mvd_y, int mvd_x) = nullptr;
};

struct BaselineInterReader {
  void* ctx = nullptr;
  bool (*single_mode)(void* ctx, int new_mv_ctx, int* symbol) = nullptr;
  bool (*drl_index)(void* ctx, int new_mv_ctx, int* index) = nullptr;
  bool (*mvd)(void* ctx, int* mvd_y, int* mvd_x) = nullptr;
};

struct AdvancedBlockCoding {
  // Inputs (both directions).
  int mi_size = 12;
  int new_mv_ctx = 0;
  bool force_integer_mv = false;
  bool allow_warpmv_mode = false;
  bool allow_bawp = false;
  bool has_chroma = false;
  bool ref_scaled = false;
  bool frame_is_switch = false;
  int warp_mv_ctx = 0;

  // Writer inputs / reader outputs.
  bool is_warp = false;
  bool warp_mv = false;
  int single_mode = 1;  // 0 NEARMV, 1 GLOBALMV, 2 NEWMV
  BawpSymbols bawp{};
  MotionModeSymbols motion{};
  int warp_idx = 3;
  bool warpmv_with_mvd = false;
  WarpDeltaSymbols warp_delta{};
  bool six_param = false;
  int drl_index = 0;
  int mvd_y = 0;
  int mvd_x = 0;

  // Derived outputs (filled by both directions).
  int y_mode = static_cast<int>(YMode::kGlobalMv);
  int motion_mode = 0;  // MotionMode value
};

bool write_advanced_inter_block(av2::Encoder& enc, AdvancedCdfState& cdfs,
                                const AdvancedBlockCoding& block,
                                const BaselineInterWriter& baseline,
                                std::string* error);

bool read_advanced_inter_block(av2::Decoder& dec, AdvancedCdfState& cdfs,
                               const AdvancedBlockCoding& gates,
                               const BaselineInterReader& baseline,
                               AdvancedBlockCoding* out, std::string* error);

// ---------------------------------------------------------------------------
// Sequence/frame header bits.
// ---------------------------------------------------------------------------

struct SequenceAdvancedConfig {
  // §5.4.6 sequence_inter_config().  The four motion modes are coded in
  // INTERINTRA..EXTENDWARP order.
  bool seq_enabled_motion_modes[kMotionModes] = {false, false, false, false,
                                                 false};
  bool seq_frame_motion_modes_present_flag = false;
  bool enable_six_param_warp_delta = false;
  bool enable_masked_compound = false;
  bool enable_bawp = false;
  bool enable_cwp = false;
  bool enable_imp_msk_bld = false;
  bool enable_global_motion = false;
};

// Writes the advanced subset of sequence_inter_config() in normative order:
//   seq_enabled_motion_modes[INTERINTRA..EXTENDWARP]
//   seq_frame_motion_modes_present_flag (iff any mode enabled)
//   enable_six_param_warp_delta (iff seq_enabled_motion_modes[DELTAWARP])
//   enable_masked_compound
//   ... callers interleave the M2-owned fields between these two groups;
//   enable_bawp, enable_cwp, enable_imp_msk_bld, enable_global_motion are
//   emitted by the three helpers below at their exact M2 insertion points.
//
// Returns false only for a self-inconsistent config.
bool write_sequence_motion_mode_config(av2enc::BitWriter* writer,
                                       const SequenceAdvancedConfig& config,
                                       std::string* error);

// The three tail fields, in order, placed after `enable_tip`/`enable_mv_traj`.
bool write_sequence_bawp_cwp_bits(av2enc::BitWriter* writer,
                                  const SequenceAdvancedConfig& config,
                                  std::string* error);

// Placed with `enable_global_motion` in the sequence tail (after
// enable_flex_mvres).  Kept separate because M2 currently hard-codes 0.
bool write_sequence_global_motion_bit(av2enc::BitWriter* writer,
                                      const SequenceAdvancedConfig& config,
                                      std::string* error);

struct FrameAdvancedConfig {
  bool frame_is_intra = false;
  bool seq_frame_motion_modes_present_flag = false;
  bool seq_enabled_motion_modes[kMotionModes] = {false, false, false, false,
                                                 false};
  // frame_enabled_motion_modes[] after read_interpolation_filter().
  bool frame_enabled_motion_modes[kMotionModes] = {false, false, false, false,
                                                   false};
  bool enable_bawp = false;            // sequence-level enable_bawp
  bool allow_bawp = false;             // frame-level decision
  bool allow_warpmv_mode = false;      // frame-level decision
};

// §5.18.2 after read_interpolation_filter(): one bit per sequence-enabled
// motion mode (INTERINTRA..EXTENDWARP) when
// seq_frame_motion_modes_present_flag is 1.
bool write_frame_motion_mode_bits(av2enc::BitWriter* writer,
                                  const FrameAdvancedConfig& config,
                                  std::string* error);

// §5.18.2 after skip_mode_params():
//   if (!FrameIsIntra && enable_bawp) f(1) allow_bawp
//   if (!FrameIsIntra && frame_enabled_motion_modes[DELTAWARP])
//       f(1) allow_warpmv_mode
bool write_frame_bawp_warp_bits(av2enc::BitWriter* writer,
                                const FrameAdvancedConfig& config,
                                std::string* error);

}  // namespace av2enc::video::adv

#endif  // AV2ENC_VIDEO_ADVANCED_SYNTAX_H_
