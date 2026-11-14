// AV2 delta-Q and segmentation bitstream syntax (§5.18.7.1, §5.18.7.8,
// §5.20.5.7/.8/.11/.12) plus the coefficient-CDF slice selector (§6.17.2).
//
// Scope
// -----
// This module is the *bitstream side* only.  It owns:
//   * frame-header syntax: segmentation_params(), setup_qm_params()'s
//     using_qmatrix bit, delta_q_params();
//   * the derived frame state that follows from those fields
//     (LastActiveSegId, SegIdPreSkip, HasLosslessSegment, update_map
//     inference for PRIMARY_REF_NONE);
//   * tile/block syntax: CurrentQIndex reset per tile, ReadDeltas per
//     superblock, read_delta_qindex() and read_segment_id() /
//     intra_segment_id() / inter_segment_id();
//   * the §8.3 CDF selection for delta_q_abs, segment_id, seg_id_ext_flag,
//     seg_id_predicted, and the four COEFF_CDF_Q_CTXS slices of
//     init_coeff_cdfs().
//
// It does not decide quantizers, partition trees, or modes.  The rate-control
// planner in src/encoder/rc/ produces `rc::DeltaQPlan` and
// `rc::SegmentationMap`; this module converts those decisions into syntax and
// gives the caller the exact state updates a decoder performs.
//
// Normative references (AV2 Bitstream & Decoding Process Specification v1.0.0)
// ---------------------------------------------------------------------------
//   §5.4.4  sequence_segment_config(): enable_ext_seg, MaxSegments
//   §5.4.9  seg_info(numSegments)
//   §5.18.2 frame_header_info() call order:
//             quantization_params(); set_primary_ref_frame_and_ctx(1);
//             segmentation_params(); setup_qm_params(); delta_q_params();
//   §5.18.7.1 segmentation_params()
//   §5.18.7.8 delta_q_params()  (AV2 has no delta_q_update_map syntax)
//   §5.20.1  CurrentQIndex = base_q_idx at each tile
//   §5.20.2.1 ReadDeltas = delta_q_present at each superblock
//   §5.20.5.7 intra_segment_id()
//   §5.20.5.8 read_segment_id(); §8.3.2 segment_id ctx
//   §5.20.5.11 read_delta_qindex()
//   §5.20.5.12 seg_feature_active()
//   §6.17.2 init_coeff_cdfs() (slice from raw base_q_idx)
//   §7.14.2 get_qindex()
//
// Important deviations from AV1 that this module implements deliberately:
//   * AV2 `delta_q_params()` has only `delta_q_present` and `delta_q_res`.
//     There is no `delta_q_update_map` field; CurrentQIndex is reset to
//     base_q_idx at every tile and only carries across superblocks in a tile.
//   * `segmentation_update_map` and `segmentation_temporal_update` are *not*
//     coded when DerivedPrimaryRefFrame == PRIMARY_REF_NONE (key frames):
//     they are inferred to 1 and 0 respectively.  They are coded on inter
//     frames.
//   * `enable_ext_seg == 0` means 8 segments, not 16 (MaxSegments).
#pragma once

#include "core/bitstream/bitwriter.h"
#include "codec/entropy/deltaq/still/bitreader.h"
#include "codec/entropy/deltaq/still/deltaq_cdf_tables.h"
#include "codec/rate/rc/rc.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::deltaq {

// ---------------------------------------------------------------------------
// Constants (§9.1 / spec constants)
// ---------------------------------------------------------------------------

inline constexpr int kMaxSegmentsSpec = 16;  // MAX_SEGMENTS
inline constexpr int kMaxSegmentsBase = 8;   // enable_ext_seg == 0
inline constexpr int kSegLvlMax = 3;         // SEG_LVL_MAX
inline constexpr int kSegLvlAltQ = 0;        // SEG_LVL_ALT_Q
inline constexpr int kSegLvlSkip = 1;        // SEG_LVL_SKIP
inline constexpr int kSegLvlGlobalMv = 2;    // SEG_LVL_GLOBALMV
inline constexpr int kSegAltQBits = 9;       // Segmentation_Feature_Bits[ALT_Q]
inline constexpr int kSegAltQLimit = 351;    // Segmentation_Feature_Max[ALT_Q]
inline constexpr int kDeltaQSmall = 7;       // DELTA_Q_SMALL
inline constexpr int kDeltaQAlphabet = 8;    // Default_Delta_Q_Cdf symbols
inline constexpr int kDeltaQMaxAbs = 516;    // largest escape-coded magnitude
inline constexpr int kQctxCount = 4;         // COEFF_CDF_Q_CTXS

// Number of segment ids available for `enable_ext_seg` (§5.4.4).
int max_segments(bool enable_ext_seg) noexcept;

// §5.4.9 seg_info(MaxSegments), the segment feature table shared by
// sequence_segment_config() (seq_seg_info_present_flag = 1) and
// segmentation_params() (when the frame does not reuse the sequence table).
bool validate_segment_table(const rc::SegmentationParams& params,
                            int num_segments, std::string* error = nullptr);
void write_segment_table(BitWriter& writer,
                         const rc::SegmentationParams& params,
                         int num_segments);
bool read_segment_table(BitReader& reader, int num_segments,
                        rc::SegmentationParams& params,
                        std::string* error = nullptr);

// Clause 6.17.2 slice index.  The input is the *raw* base_q_idx.
int coeff_cdf_qctx(int base_q_idx) noexcept;

// One coefficient CDF per selected slice.  The pointed-to objects have the
// post-slice dimensions of the normative arrays.  This is the value a caller
// installs into its `Tile*Cdf` copies; it must be selected from raw
// base_q_idx, never from an effective/delta/segment qindex.
using TxbSkipCdf = std::uint16_t[2][5][10][3];
using VTxbSkipCdf = std::uint16_t[12][3];
using EobPt16Cdf = std::uint16_t[3][6];
using EobPt32Cdf = std::uint16_t[3][7];
using EobPt64Cdf = std::uint16_t[3][8];
using EobPt128Cdf = std::uint16_t[3][9];
using EobPt256Cdf = std::uint16_t[3][9];
using EobPt512Cdf = std::uint16_t[3][9];
using EobPt1024Cdf = std::uint16_t[3][9];
using EobExtraCdf = std::uint16_t[3];
using DcSignCdf = std::uint16_t[2][2][3][3];
using CoeffBaseCdf = std::uint16_t[5][20][2][5];
using CoeffBaseLfCdf = std::uint16_t[5][33][2][7];
using CoeffBaseEobCdf = std::uint16_t[5][4][4];
using CoeffBaseLfEobCdf = std::uint16_t[5][4][6];
using CoeffBrCdf = std::uint16_t[7][5];
using CoeffBrLfCdf = std::uint16_t[14][5];
using CoeffBaseUvCdf = std::uint16_t[12][5];
using CoeffBaseLfUvCdf = std::uint16_t[12][7];
using CoeffBaseEobUvCdf = std::uint16_t[4][4];
using CoeffBaseLfEobUvCdf = std::uint16_t[4][6];
using CoeffBrUvCdf = std::uint16_t[4][5];

struct CoeffCdfSlice {
  const TxbSkipCdf* txb_skip = nullptr;
  const VTxbSkipCdf* v_txb_skip = nullptr;
  const EobPt16Cdf* eob_pt_16 = nullptr;
  const EobPt32Cdf* eob_pt_32 = nullptr;
  const EobPt64Cdf* eob_pt_64 = nullptr;
  const EobPt128Cdf* eob_pt_128 = nullptr;
  const EobPt256Cdf* eob_pt_256 = nullptr;
  const EobPt512Cdf* eob_pt_512 = nullptr;
  const EobPt1024Cdf* eob_pt_1024 = nullptr;
  const EobExtraCdf* eob_extra = nullptr;
  const DcSignCdf* dc_sign = nullptr;
  const CoeffBaseCdf* coeff_base = nullptr;
  const CoeffBaseLfCdf* coeff_base_lf = nullptr;
  const CoeffBaseEobCdf* coeff_base_eob = nullptr;
  const CoeffBaseLfEobCdf* coeff_base_lf_eob = nullptr;
  const CoeffBrCdf* coeff_br = nullptr;
  const CoeffBrLfCdf* coeff_br_lf = nullptr;
  const CoeffBaseUvCdf* coeff_base_uv = nullptr;
  const CoeffBaseLfUvCdf* coeff_base_lf_uv = nullptr;
  const CoeffBaseEobUvCdf* coeff_base_eob_uv = nullptr;
  const CoeffBaseLfEobUvCdf* coeff_base_lf_eob_uv = nullptr;
  const CoeffBrUvCdf* coeff_br_uv = nullptr;
};

CoeffCdfSlice coeff_cdf_slice(int base_q_idx) noexcept;

// ---------------------------------------------------------------------------
// CDF state for the non-coefficient delta-Q / segmentation symbols
// ---------------------------------------------------------------------------
//
// These are copies of the §9.3 defaults.  A tile owns one copy; the coder
// mutates it in place unless disable_cdf_update is set.
struct BlockCdfs {
  std::uint16_t delta_q[9] = {};                 // TileDeltaQCdf, N = 8
  std::uint16_t segment_id[3][9] = {};           // TileSegmentIdCdf, N = 8
  std::uint16_t segment_id_ext[3][9] = {};       // TileSegmentIdExtCdf, N = 8
  std::uint16_t seg_id_ext_flag[3][3] = {};      // TileSegIdExtFlagCdf, N = 2
  std::uint16_t segment_id_predicted[3][3] = {}; // N = 2

  void init() noexcept;
};

// ---------------------------------------------------------------------------
// Frame header: segmentation_params() + setup_qm_params() + delta_q_params()
// ---------------------------------------------------------------------------

struct FrameQpHeader {
  // DerivedPrimaryRefFrame == PRIMARY_REF_NONE (key frame, and any frame whose
  // primary reference is none).  When true, update_map/temporal_update are
  // inferred rather than coded.
  bool primary_ref_is_none = true;
  // Sequence-level enable_ext_seg (§5.4.4).  false => MaxSegments = 8.
  bool enable_ext_seg = false;
  // FrameType == SWITCH_FRAME.  Only affects whether temporal_update is coded
  // when update_map == 1.
  bool frame_type_switch = false;
  // setup_qm_params()'s using_qmatrix bit.  The M1/M2 writers never enable
  // quantization matrices; a true value is rejected because the QM payload is
  // not part of this module.
  bool using_qmatrix = false;
  // §5.4.4 sequence_segment_config(): seq_seg_info_present_flag and
  // seq_allow_seg_info_change.  When the sequence carries the table the frame
  // may reuse it instead of coding seg_info(); with allow_change = 0 the reuse
  // is inferred and no reuse_seg_info bit is coded.
  bool seq_seg_info_present = false;
  bool seq_allow_seg_info_change = false;
  // Frame request: reuse_seg_info (only meaningful when the sequence present
  // and allow_change are both 1).  The frame's `params` must still carry the
  // same table so the derived state and block decisions match the decoder.
  bool reuse_seg_info = true;
};

// Writes segmentation_params() (§5.18.7.1).  Returns the number of bits.
// `error` is set when the parameters cannot be represented (feature data out
// of range, too many segments for enable_ext_seg, temporal_update on a key
// frame, ...).  Returns 0 on error.
std::size_t write_segmentation_params(BitWriter& writer,
                                      const rc::SegmentationParams& params,
                                      const FrameQpHeader& header,
                                      std::string* error = nullptr);

// Writes delta_q_params() (§5.18.7.8).  `base_q_idx == 0` implies
// delta_q_present = 0 and codes no bits.
std::size_t write_delta_q_params(BitWriter& writer,
                                 const rc::DeltaQParams& params,
                                 int base_q_idx,
                                 std::string* error = nullptr);

// Writes the three consecutive frame-header pieces in normative order:
//   segmentation_params(); setup_qm_params(); delta_q_params()
// This is the exact replacement for the three existing bits
// (segmentation_enabled, using_qmatrix, delta_q_present) in the M1/M2 header
// writers.  Returns the number of bits written, or 0 on error.
std::size_t write_segmentation_and_delta_q(BitWriter& writer,
                                           const rc::SegmentationParams& seg,
                                           const rc::DeltaQParams& delta_q,
                                           int base_q_idx,
                                           const FrameQpHeader& header,
                                           std::string* error = nullptr);

bool read_segmentation_params(BitReader& reader,
                              const FrameQpHeader& header,
                              rc::SegmentationParams& params,
                              std::string* error = nullptr);

bool read_delta_q_params(BitReader& reader,
                         int base_q_idx,
                         rc::DeltaQParams& params,
                         std::string* error = nullptr);

// Reads the three pieces in normative order.  `using_qmatrix` is returned so
// the caller can reject an unsupported QM payload.
bool read_segmentation_and_delta_q(BitReader& reader,
                                   int base_q_idx,
                                   const FrameQpHeader& header,
                                   rc::SegmentationParams& seg,
                                   rc::DeltaQParams& delta_q,
                                   bool& using_qmatrix,
                                   std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Derived frame state (§5.18.7.1, §6.17.2, §7.14.2)
// ---------------------------------------------------------------------------

struct FrameSegmentation {
  bool enabled = false;
  bool update_map = false;       // effective value (inferred 1 for key frames)
  bool temporal_update = false;  // effective value (inferred 0 for key frames)
  bool enable_ext_seg = false;
  int num_segments = kMaxSegmentsBase;
  int last_active_seg_id = 0;    // highest id with any enabled feature
  bool seg_id_pre_skip = false;  // any SEG_LVL_SKIP/GLOBALMV feature enabled
  bool has_lossless_segment = false;
  // Feature state for ids 0 .. num_segments-1 (ids >= num_segments are empty).
  std::array<rc::SegmentFeatures, kMaxSegmentsSpec> features{};

  bool alt_q_active(int segment_id) const noexcept {
    return enabled && segment_id >= 0 && segment_id < kMaxSegmentsSpec &&
           features[static_cast<std::size_t>(segment_id)]
               .enabled[static_cast<std::size_t>(kSegLvlAltQ)];
  }
  bool skip_active(int segment_id) const noexcept {
    return enabled && segment_id >= 0 && segment_id < kMaxSegmentsSpec &&
           features[static_cast<std::size_t>(segment_id)]
               .enabled[static_cast<std::size_t>(kSegLvlSkip)];
  }
  bool globalmv_active(int segment_id) const noexcept {
    return enabled && segment_id >= 0 && segment_id < kMaxSegmentsSpec &&
           features[static_cast<std::size_t>(segment_id)]
               .enabled[static_cast<std::size_t>(kSegLvlGlobalMv)];
  }
  std::int32_t alt_q(int segment_id) const noexcept {
    return features[static_cast<std::size_t>(segment_id)]
        .data[static_cast<std::size_t>(kSegLvlAltQ)];
  }
};

// Builds the derived state exactly as the decoder does.  `base_q_idx` and
// `delta_q` are needed for HasLosslessSegment (§5.18.2, get_qindex(1, id)).
bool derive_frame_segmentation(const rc::SegmentationParams& params,
                               const rc::DeltaQParams& delta_q,
                               int base_q_idx,
                               const FrameQpHeader& header,
                               FrameSegmentation& out,
                               std::string* error = nullptr);

// get_qindex(ignoreDeltaQ, segmentId) §7.14.2 for the states this module
// supports (no UV/DC deltas).
bool effective_qindex(const FrameSegmentation& seg,
                      int base_q_idx,
                      int current_q_index,
                      bool delta_q_present,
                      int segment_id,
                      bool ignore_delta_q,
                      int& qindex,
                      std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Tile-level CurrentQIndex / ReadDeltas state (§5.20.1, §5.20.2.1)
// ---------------------------------------------------------------------------

struct TileQp {
  int base_q_idx = 0;
  rc::DeltaQParams delta_q{};
  int current_q_index = 0;  // CurrentQIndex
  bool read_deltas = false; // ReadDeltas
  bool initialized = false;
};

// CurrentQIndex = base_q_idx at the start of each tile.  Call this once per
// tile; never carry state across a tile boundary.
bool begin_tile(TileQp& state, int base_q_idx, const rc::DeltaQParams& delta_q,
                std::string* error = nullptr);

// ReadDeltas = delta_q_present at the start of each superblock.
void begin_superblock(TileQp& state) noexcept;

// `!(MiSize == SbSize && skip_flag) && ReadDeltas`
bool codes_delta_q(const TileQp& state, bool mi_size_is_sb,
                   bool skip_flag) noexcept;

// Nearest reduced delta for a qindex target: round half away from zero,
// step = 1 << res.  This is the same rule as rc::plan_delta_q().
int reduced_delta_for_target(int current_q_index, int target_q_index,
                             int res) noexcept;

// Applies one coded reduced delta: Clip3(1, 255, cur + (delta << res)).
bool apply_reduced_delta(TileQp& state, int reduced_delta,
                         std::string* error = nullptr);

// Writes/reads the delta_q_abs symbol (alphabet 8, DELTA_Q_SMALL escape) and
// the §5.20.5.11 sign/state update.  `cdf` is TileDeltaQCdf (9 entries).
//
// Both functions clear ReadDeltas afterwards, exactly like the spec's
// `ReadDeltas = 0` statement after read_delta_qindex().
bool write_delta_qindex(av2::Encoder& encoder, std::uint16_t cdf[9],
                        TileQp& state, bool mi_size_is_sb, bool skip_flag,
                        int reduced_delta, std::string* error = nullptr);

bool read_delta_qindex(av2::Decoder& decoder, std::uint16_t cdf[9],
                       TileQp& state, bool mi_size_is_sb, bool skip_flag,
                       int& reduced_delta, std::string* error = nullptr);

// Low-level symbol coding, exported for tests and for callers that manage
// ReadDeltas themselves.  `value` is the signed reduced delta
// (reducedDeltaQIndex), not the shifted qindex offset.
bool write_delta_q_abs(av2::Encoder& encoder, std::uint16_t cdf[9], int value,
                       std::string* error = nullptr);
bool read_delta_q_abs(av2::Decoder& decoder, std::uint16_t cdf[9], int& value,
                      std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Segmentation map state (§5.20.5.7/.8, §5.20.7.2/.4)
// ---------------------------------------------------------------------------

// MI-grid state for one tile.  `current` is SegmentIds for this frame,
// `previous` is PrevSegmentIds.  `above_pred` / `left_pred` are
// AboveSegPredContext / LeftSegPredContext.
struct SegmentGrid {
  int mi_cols = 0;
  int mi_rows = 0;
  int tile_mi_col_start = 0;
  int tile_mi_col_end = 0;
  int tile_mi_row_start = 0;
  int tile_mi_row_end = 0;
  bool enable_ext_seg = false;
  int num_segments = kMaxSegmentsBase;
  std::vector<std::uint8_t> current;
  std::vector<std::uint8_t> previous;
  std::vector<std::uint8_t> above_pred;
  std::vector<std::uint8_t> left_pred;
};

bool init_segment_grid(SegmentGrid& grid, int mi_cols, int mi_rows,
                       int tile_mi_col_start, int tile_mi_col_end,
                       int tile_mi_row_start, int tile_mi_row_end,
                       bool enable_ext_seg, int num_segments,
                       std::span<const std::uint8_t> previous_ids,
                       std::string* error = nullptr);

// clear_above_context() / clear_left_context() for the segment-prediction
// arrays.  Call the first once per tile and the second at each superblock row,
// like the decoder's decode_tile().
void clear_above_context(SegmentGrid& grid) noexcept;
void clear_left_context(SegmentGrid& grid) noexcept;

// Replaces PrevSegmentIds (for example from the previous frame's map).
bool set_previous_ids(SegmentGrid& grid, std::span<const std::uint8_t> ids,
                      std::string* error = nullptr);

// read_segment_id() neighbour prediction from the *current* frame map.
int segment_prediction(const SegmentGrid& grid, int mi_row, int mi_col) noexcept;
// §8.3.2 context for the segment_id symbol.
int segment_id_context(int prev_ul, int prev_u, int prev_l) noexcept;

// get_segment_id() §5.20.7.4: minimum PrevSegmentIds over the block.
int previous_segment_prediction(const SegmentGrid& grid, int mi_row, int mi_col,
                                int bw4, int bh4) noexcept;

// neg_deinterleave() (§5.20.5.8) and its inverse.  The brute-force inverse
// scans [0, max), which is at most 16 entries.
int neg_deinterleave(int diff, int ref, int max) noexcept;
bool neg_interleave(int target, int ref, int max, int& coded) noexcept;

struct SegmentBlock {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;  // block width in 4x4 luma units
  int bh4 = 1;  // block height in 4x4 luma units
};

// Fills SegmentIds for the block after its mode info has been decoded.
bool commit_segment_block(SegmentGrid& grid, const SegmentBlock& block,
                          int segment_id, std::string* error = nullptr);

// intra_segment_id() (§5.20.5.7): used by M1 key frames.
bool write_intra_segment_id(av2::Encoder& encoder, BlockCdfs& cdfs,
                            const FrameSegmentation& seg, SegmentGrid& grid,
                            const SegmentBlock& block, int segment_id,
                            bool skip_flag, std::string* error = nullptr);
bool read_intra_segment_id(av2::Decoder& decoder, BlockCdfs& cdfs,
                           const FrameSegmentation& seg, SegmentGrid& grid,
                           const SegmentBlock& block, bool skip_flag,
                           int& segment_id, std::string* error = nullptr);

// inter_segment_id(preSkip) (§5.20.7.2): used by M2 P frames (also for the
// intra blocks inside a P frame).  `pre_skip` selects the call before or after
// read_skip().  At the pre-skip call the caller must pass skip_flag = 0
// because read_skip() has not happened yet.  `coded_symbols` reports whether
// any bits were written/read.
bool write_inter_segment_id(av2::Encoder& encoder, BlockCdfs& cdfs,
                            const FrameSegmentation& seg, SegmentGrid& grid,
                            const SegmentBlock& block, bool pre_skip,
                            int segment_id, bool skip_flag,
                            bool& coded_symbols,
                            std::string* error = nullptr);
bool read_inter_segment_id(av2::Decoder& decoder, BlockCdfs& cdfs,
                           const FrameSegmentation& seg, SegmentGrid& grid,
                           const SegmentBlock& block, bool pre_skip,
                           bool skip_flag, int& segment_id,
                           bool& coded_symbols,
                           std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Rate-control planner integration (src/encoder/rc/)
// ---------------------------------------------------------------------------

// A checked view over rc::DeltaQPlan for one tile.  rc plans one entry per
// superblock in raster order, and the decoder resets CurrentQIndex at every
// tile, so a plan never spans tiles.
struct DeltaQPlanView {
  int sb_cols = 0;
  int sb_rows = 0;
  const rc::DeltaQPlan* plan = nullptr;
};

bool make_delta_q_plan_view(const rc::DeltaQPlan& plan, int sb_cols,
                            int sb_rows, DeltaQPlanView& view,
                            std::string* error = nullptr);

const rc::DeltaQPlanStep* delta_q_plan_step(const DeltaQPlanView& view, int sb_x,
                                            int sb_y) noexcept;

// Verifies that the current TileQp state matches the plan step's
// qindex_before and that applying the step yields qindex_after.
bool check_plan_step(const TileQp& state, const rc::DeltaQPlanStep& step,
                     std::string* error = nullptr);

// Copies an rc::SegmentationMap into PrevSegmentIds (the previous-frame map).
bool load_segmentation_map(const rc::SegmentationMap& map, SegmentGrid& grid,
                           std::string* error = nullptr);

// Reports the qindex the encoder must use for a block, given the planner's
// segment id and the current tile QP state.
int block_qindex(const FrameSegmentation& seg, const TileQp& qp,
                 int segment_id) noexcept;

}  // namespace av2enc::deltaq
