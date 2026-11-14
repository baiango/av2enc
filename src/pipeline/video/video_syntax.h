// AV2 M2 P-frame tile syntax writer.
//
// Structure follows the M1 still-picture writer (partition tree, intra modes,
// transform-tree, EOB/coefficient coding) and adds the inter-frame block layer:
//
//   inter_frame_mode_info()   §5.20.7.1
//     read_is_inter()         §5.20.7.3
//     read_skip()             §5.20.5.10
//     inter_block_mode_info() §5.20.7.6
//       read_ref_frames()     §5.20.7.10
//       single_mode / drl_idx / read_mv (shell coding) §5.20.7.8, §5.20.7.20
//
// Intra blocks inside the P frame reuse the M1 intra mode and residual syntax.
// Inter blocks default to skip_flag = 1 (no residual).  With feature 1 enabled
// the writer emits the inter transform block: TX_MODE_LARGEST transform sizes,
// the mandatory inter luma TX_TYPE (DCT_DCT), and §8.3.2 inter coefficient
// contexts.
//
// The writer derives every entropy context from the block stream exactly as
// the decoder does. The producer supplies decisions and reconstruction data,
// not CDF state.
#pragma once

#include "pipeline/video/video_headers.h"
#include "pipeline/video/post/video_filter_plan.h"
#include "codec/entropy/avg_cdf/avg_cdf_bank.h"
#include "codec/inter/merge/merge.h"
#include "codec/inter/temporal/temporal_types.h"
#include "codec/intra/palette/core/palette.h"
#include "codec/rate/rc/rc.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <string>
#include <vector>

namespace av2enc::video::syntax {

inline constexpr int kPlaneCount = 3;

enum BlockFlags : std::uint8_t {
  kFlagForced = 1u << 0,
  kFlagIsCfl = 1u << 1,
  kFlagSbRoot = 1u << 2,
  // §5.20.5.3 use_intrabc = 1 on an intra frame.  The block carries the BV and
  // (optionally) morph_pred; it codes no intra mode or residual when skip.
  kFlagIsIbc = 1u << 3,
  // The chroma-ownership fields (has_chroma / chroma_bsize / chroma_mi_row /
  // chroma_mi_col) are valid.  Cleared by every producer that predates SDP,
  // which keeps the historical "derive chroma from the luma leaf" behaviour.
  kFlagHasChromaInfo = 1u << 4,
  // §5.20.3.1 intra SDP split node.  It codes no partition syntax; its two
  // children are the luma tree root and then the chroma tree root.
  kFlagSdpSplit = 1u << 5,
};

// Inter prediction mode as signalled through single_mode:
//   0 -> NEARMV (YMode 14), 1 -> NEWMV (16), 2 -> GLOBALMV (15)
enum InterMode : std::uint8_t {
  kInterGlobalMv = 0,
  kInterNearMv = 1,
  kInterNewMv = 2,
};

// §5.20.5.5 IntraJointMode value for a coded (YMode, AngleDeltaY): 0..4 for
// the reordered non-directional modes (DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH)
// and 5 + directional index otherwise.  The encoder needs it to derive the
// §7.13.2.15/.16 filterTypeAbove/Left of the following blocks; it is the same
// mapping the syntax writer tracks internally.
int intra_joint_mode(int y_mode, int angle_delta);

struct BlockSyntax {
  // --- tree ---
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t bsize = 0;
  std::uint8_t partition = 0;
  // Bit p = allowed[PartitionType p].  Bit 3 exceeds 4 bits for the extended
  // family, so this is 16 bits (10 spec partition values).
  std::uint16_t allowed_mask = 0;
  std::uint8_t flags = 0;
  std::uint16_t subtree = 0;

  // --- §5.20.3.1 SDP tree state (only when the frame enables SDP) ---
  // `tree_type` is the §6.19.3.1 TreeType (0 SHARED, 1 LUMA, 2 CHROMA).
  // `has_chroma` is the decoded HasChroma (ownership: 0 on a LUMA_PART leaf);
  // `chroma_bsize` is the owned ChromaMiSize in luma 4x4 units and
  // `chroma_mi_row/col` its luma MI origin.  `chroma_partition_known` /
  // `implied_luma_partition` are the §5.20.3.2 chroma-root bookkeeping.
  std::uint8_t tree_type = 0;
  std::uint8_t has_chroma = 1;
  std::uint8_t chroma_offset = 0;
  std::uint8_t chroma_bsize = 0;
  std::uint16_t chroma_mi_row = 0;
  std::uint16_t chroma_mi_col = 0;
  std::uint8_t chroma_partition_known = 0;
  std::uint8_t implied_luma_partition = 0;
  // The §5.20.3.2 partition symbols for this node, derived by the reference
  // `sdp` planner on the encoder side (contexts from the luma/chroma MiSize
  // grids it maintains).  The writer emits them verbatim, so it can never
  // disagree with the solver about implied/allowed sets or contexts.
  struct SdpSymbol {
    std::uint8_t kind = 0;
    std::uint8_t plane = 0;
    std::uint8_t context = 0;
    std::uint8_t value = 0;
  };
  std::uint8_t sdp_symbol_count = 0;
  SdpSymbol sdp_symbols[4];

  // --- leaf mode info ---
  std::uint8_t is_inter = 0;
  std::uint8_t skip_flag = 0;   // inter only; 1 means no residual
  std::uint8_t inter_mode = kInterGlobalMv;
  std::uint8_t drl_index = 0;
  // §5.18.2/§5.20.7.10 RefFrame[0] index into ref_frame_idx[] (0..6).
  std::uint8_t ref_index = 0;
  // Compound block (§5.20.7.10 comp_mode / §5.20.7.6).  `ref_index1` is
  // RefFrame[1]; `compound_joint` selects is_joint; `compound_mode` is the
  // compound_mode_non_joint symbol (YMode = NEAR_NEARMV + compound_mode).
  std::uint8_t compound = 0;
  std::uint8_t ref_index1 = 0;
  std::uint8_t compound_joint = 0;
  std::uint8_t compound_mode = 3;  // 0 = NEAR_NEARMV, 3 = GLOBAL_GLOBALMV
  // §5.20.7.6 compound MVs.  NEAR_NEARMV resolves list 0/1 independently:
  // `drl_index`/`mv_y,mv_x` are list 0 and `drl_index1`/`mv1_y,mv1_x` list 1.
  // With use_optflow the block codes a single RefMvIdx into the two-list
  // stack, so `drl_index1` is unused and both MVs come from entry
  // `drl_index` of that stack.
  std::uint8_t drl_index1 = 0;
  std::int32_t mv1_y = 0;
  std::int32_t mv1_x = 0;
  // §5.20.7.6 use_optflow (REFINE_SWITCHABLE only) and §5.20.7.17 use_refinemv.
  std::uint8_t use_optflow = 0;
  std::uint8_t use_refinemv = 0;
  // §5.20.7.16 masked compound (enable_masked_compound = 1).  `compound_masked`
  // is comp_group_idx; when set, `compound_type` selects the mask
  // (0 = WEDGE, 1 = DIFFWTD) and the matching symbols follow.
  std::uint8_t compound_masked = 0;
  std::uint8_t compound_type = 0;
  std::uint8_t wedge_index = 0;
  std::uint8_t wedge_sign = 0;
  std::uint8_t mask_type = 0;
  // §5.20.7.6 CwpIdx (enable_cwp = 1).  `cwp_index` is the coded index; 0 is
  // CWP_EQUAL.  The weight is derived by the decoder from is_same_side().
  // `cwp_weight` is carried for the prediction and is never coded directly.
  std::uint8_t cwp_index = 0;
  std::int8_t cwp_weight = 8;
  std::int32_t mv_y = 0;   // resolved MV, 1/8 luma samples
  std::int32_t mv_x = 0;
  std::int32_t mvd_y = 0;  // signalled MVD for NEWMV, 1/8 luma samples
  std::int32_t mvd_x = 0;
  // §5.20.7.6 MV-resolution plan for a NEWMV leaf.
  //   `use_amvd` selects the §5.20.7.20/.21 mv_joint/amvd_index MVD form and
  //   pins MvPrecision to the frame value.
  //   `block_mv_precision` is the target me2::kMvPrecision* (6=EIGHTH,
  //   5=QUARTER, 4=HALF, 3=ONE, ...); -1 means "frame precision, no coarser
  //   per-block symbol was decided".  With --video-flex-mvres the writer still
  //   codes use_most_probable_precision for every eligible NEWMV block.
  //   `mvd_sign_derived` records the §5.20.7.13 derived-sign decision and is
  //   used by the trace/gate; the writer derives it again from the magnitudes.
  std::uint8_t use_amvd = 0;
  std::int8_t block_mv_precision = -1;
  std::uint8_t mvd_sign_derived = 0;

  std::uint8_t y_mode = 0;
  std::int8_t y_angle_delta = 0;  // §5.20.5.5 mrl_index / mrl_sec_index (enable_mrls).  Luma only; 0 when MRL
  // is disabled or the block is non-directional.
  std::uint8_t mrl_index = 0;
  std::uint8_t mrl_sec_index = 0;
  // §5.20.7.9 dip_mode_info() (--video-dip).  Signalled only for luma DC_PRED
  // keyframe blocks that satisfy the size predicate; `use_dip = 0` otherwise.
  std::uint8_t use_dip = 0;
  std::uint8_t dip_transpose = 0;
  std::uint8_t dip_mode = 0;  // 0..5
  std::uint8_t uv_mode = 0;
  // §5.20.5.7/.8, §5.20.7.2 segment id signalled for this leaf.  A skipped
  // inter leaf does not code the symbol; the encoder replays the decoder's
  // §5.20.7.4 neighbour prediction and stores that value here.  Unused (0)
  // when the frame does not enable segmentation.
  std::uint8_t segment_id = 0;
  // §7.14.2 get_qindex() for this leaf (effective quantizer after
  // delta-Q/segmentation).  -1 means "not supplied": the deblock grid then
  // falls back to the frame base_q_idx (the historical default path).
  std::int16_t qindex = -1;

  // --- §5.20.5.3/.4 intra block copy (kFlagIsIbc) --------------------------
  std::uint8_t ibc_skip = 1;
  std::uint8_t ibc_mode = 0;
  std::uint8_t ibc_ref_mv_idx = 0;
  std::int32_t ibc_bv_row = 0;
  std::int32_t ibc_bv_col = 0;
  std::uint8_t morph_pred = 0;

  // --- §5.20.8 luma palette (--video-palette) ------------------------------
  // Index into FramePlan::palettes for a palette leaf; kNoPalette otherwise.
  // PaletteModeInfo is only reachable when y_mode == DC_PRED, so a palette
  // leaf always carries y_mode = 0.
  static constexpr std::uint32_t kNoPalette = 0xFFFFFFFFu;
  std::uint32_t palette_index = kNoPalette;

  // --- §5.20.7.14/.15 motion mode (INTERINTRA) -----------------------------
  std::uint8_t motion_mode = 0;
  std::uint8_t interintra_mode = 0;
  std::uint8_t wedge_interintra = 0;
  // §5.20.7.32 cfl_index when kFlagIsCfl is set (1 = CFL_DERIVED_ALPHA,
  // 2 = CFL_MULTI/MHCCP).
  std::uint8_t cfl_index = 1;
  std::uint8_t cfl_mh_dir = 0;
  std::uint8_t cfl_alpha_signs = 0;
  std::uint8_t cfl_alpha_u = 0;
  std::uint8_t cfl_alpha_v = 0;

  // --- §5.20.7.6 / §7.13.3.25 BAWP (--video-bawp) -------------------------
  // `use_bawp` is coded after single_mode for a single-prediction block that
  // passes the §5.20.7.6 gate.  The encoder uses the fitted (not explicit)
  // form; `bawp_alpha` is the luma alpha the decoder derives, and chroma BAWP
  // is never signalled (use_bawp_chroma = 0).
  std::uint8_t use_bawp = 0;
  std::uint8_t explicit_bawp = 0;
  std::uint8_t explicit_bawp_scale = 0;
  std::uint8_t use_bawp_chroma = 0;
  std::int16_t bawp_alpha = 256;

  // --- leaf residual ---
  std::uint8_t tx_size[kPlaneCount] = {0, 0, 0};
  std::uint32_t eob[kPlaneCount] = {0, 0, 0};
  std::uint32_t levels_offset = 0;
  // §5.20.7.27 cctx_type of the leaf's chroma pair (0 = CCTX_NONE).  Coded for
  // plane 1 when the sequence/frame enables CCTX and (is_inter || U eob != 1).
  std::uint8_t cctx_type = 0;

  // --- opt-in TX_MODE_SELECT (§5.20.6) luma plan -------------------------
  // `tx_partition` is a txpartition::TxPartition value for the coding block and
  // `tx_part_count` the number of transform leaves it produced (1 when there is
  // no partition syntax).  `tx_size[0]/eob[0]` still carry the first leaf so a
  // TX_MODE_LARGEST consumer keeps working; when tx_part_count != 0 the arrays
  // below describe every luma leaf in decode order and are authoritative.
  std::uint8_t tx_type[kPlaneCount] = {0, 0, 0};
  // §5.20.8.2 IST symbols for the luma TX block.  Chroma entries stay 0.
  std::uint8_t sec_tx_type[kPlaneCount] = {0, 0, 0};
  std::uint8_t most_probable_stx_set[kPlaneCount] = {0, 0, 0};
  std::uint8_t tx_partition = 0;
  std::uint8_t tx_part_count = 0;
  std::uint8_t tx_part_tx_size[4] = {0, 0, 0, 0};
  std::uint8_t tx_part_tx_type[4] = {0, 0, 0, 0};
  std::uint8_t tx_part_sec_tx_type[4] = {0, 0, 0, 0};
  std::uint8_t tx_part_stx_set[4] = {0, 0, 0, 0};
  std::int16_t tx_part_mi_row[4] = {0, 0, 0, 0};
  std::int16_t tx_part_mi_col[4] = {0, 0, 0, 0};
  std::uint32_t tx_part_eob[4] = {0, 0, 0, 0};
  std::uint32_t tx_part_levels_offset[4] = {0, 0, 0, 0};
};

// §7.9/§7.12.2 frame-level temporal state the syntax writer needs to rebuild
// the same MV stack the decision path used.  `field` is owned by the caller and
// must stay alive for the whole write() call.
struct TemporalFramePlan {
  bool use_ref_frame_mvs = false;
  int order_hint = 0;
  int num_past_refs = 0;
  int num_future_refs = 0;
  int closest_past = -1;    // §7.8 OrigClosestPast ref label, -1 = NONE
  int closest_future = -1;  // §7.8 OrigClosestFuture ref label, -1 = NONE
  // OrderHints[ref] for ref labels 0..num_total_refs-1.
  std::array<int, 7> ref_order_hints{-1, -1, -1, -1, -1, -1, -1};
  // merge::TemporalScanFn supplied by video_frame.cpp; nullptr keeps the stack
  // builder temporal-inert without a link dependency on motion_field.cpp.
  merge::TemporalScanFn scan = nullptr;
  const inter::temporal::MotionField* field = nullptr;
  // §7.9.2 trajectory field (enable_mv_traj); nullptr keeps the derived
  // candidates projection-only.
  const inter::temporal::TrajectoryField* mv_traj = nullptr;
};

struct FramePlan {
  int width = 0;
  int height = 0;
  int sb_size = 64;  // false for the intra keyframe (intra_frame_mode_info), true for P frames
  // §5.4.3 MaxPbAspectRatio the sequence header derived (8, or 2/4 with
  // reduce_pb_aspect_ratio).  The partition search and the tile writer's
  // §6.19.3.2 recomputation must use the same value.
  int max_pb_aspect_ratio = 8;
  // (inter_frame_mode_info, which codes is_inter/skip).
  bool inter_frame = true;
  // §5.4.3 enable_sdp.  The sequence-header bit is signalled for the whole
  // sequence; `frame_sdp` is set only on the intra keyframe, where §5.20.3.1
  // replaces each 64x64 SHARED node with a luma tree and a chroma tree.
  bool enable_sdp = false;
  bool frame_sdp = false;
  // §5.18.2 allow_bawp: code use_bawp after single_mode for every eligible
  // single-prediction block.  Off keeps every historical stream byte-identical.
  bool allow_bawp = false;
  // §5.20.5.3/.4 intra block copy + §7.13.3.26 morph_pred on an intra frame.
  // `allow_screen_content_tools` also makes palette_mode_info() present for
  // every eligible non-IBC block.  `allow_intrabc` codes use_intrabc; it is
  // independent of `morph_pred_present` (that symbol needs enable_bawp).
  bool allow_screen_content_tools = false;
  bool allow_intrabc = false;
  // IBC block-vector precision (force_integer_mv on the frame that codes
  // use_intrabc).  Independent of the inter-frame precision pin below.
  bool force_integer_mv = true;
  // §5.4.7 seq_force_integer_mv: the sequence pins FrameMvPrecision to one pel
  // for every frame, so the inter writer codes no precision pair and the
  // MVDs use one-pel units.  Off keeps every historical inter stream.
  bool frame_force_integer_mv = false;
  bool morph_pred_present = false;
  // §5.20.8 palette analysis results, indexed by BlockSyntax::palette_index.
  // Empty on every stream that does not select a palette.
  std::vector<palette::PaletteResult> palettes;
  // §5.20.7.14/.15 single-reference motion-mode syntax.  When set, every
  // eligible inter block codes inter_intra (and the inter-intra payload when
  // set).
  bool enable_interintra = false;
  // §5.4.6 temporal refinement tools.  `opfl_refine_type` mirrors the sequence
  // value (0 NONE, 1 SWITCHABLE, 2 ALL).  SWITCHABLE codes use_optflow on every
  // eligible compound block; `enable_refinemv` makes §5.20.7.17 derive
  // use_refinemv for eligible NEAR_NEARMV compound blocks (no bit: the
  // non-GLOBALMV form is not switchable).
  int opfl_refine_type = 0;
  bool enable_refinemv = false;
  // §5.18.2 FrameType == SWITCH_FRAME (refine-MV/OPFL are disabled on it).
  bool frame_is_switch = false;
  // §5.18.2 per-frame reference map.  num_total_refs > 1 lists the explicit
  // map; reference_select enables per-block comp_mode (compound).
  int num_total_refs = 1;
  bool reference_select = false;
  // §5.4.6 / §5.20.7.16 masked compound.  When on, every compound leaf codes
  // comp_group_idx and (when set) compound_type with the wedge/diffwtd
  // symbols.  Off keeps the historical AVERAGE-only byte stream.
  bool enable_masked_compound = false;
  // §5.4.6 enable_cwp / §5.20.7.6 cwp_idx.  When on, every eligible
  // NEAR_NEARMV COMPOUND_AVERAGE block codes cwp_idx after the compound type.
  bool enable_cwp = false;
  // Frame MV precision: 0 = 1/8, 1 = 1/4, 2 = 1/2 pel.
  int mv_precision = 0;
  // §5.4.6 MV-resolution tools.  Each defaults off; the block writer then
  // emits exactly the historical single_mode/DRL/shell-MVD syntax.
  bool enable_adaptive_mvd = false;
  bool enable_mvd_sign_derive = false;
  bool enable_flex_mvres = false;
  // FrameDistance[ref] for the MV stack's derived candidates.
  std::array<int, 7> frame_distance{0, 0, 0, 0, 0, 0, 0};
  // §7.12.2 temporal scan state (use_ref_frame_mvs + the §7.9 field).
  TemporalFramePlan temporal{};
  // §5.18.9 global motion parameters for the MV-stack GlobalMvs.  Identity
  // models (the default) reproduce the historical zero GlobalMvs.  The writer
  // projects the centre MV per block with the frame precision.
  std::array<inter::gme::GmParams, 7> gm_params{};
  int gm_num_refs = 0;
  // 0 = DRL_REORDER_DISABLED, 1 = CONSTRAINT, 2 = ALWAYS.
  int drl_reorder = 0;
  // Opt-in §5.18.8.1 TX_MODE_SELECT (--video-tx-search).  Off keeps
  // tx_mode_select = 0 and TX_MODE_LARGEST, i.e. every historical stream is
  // byte-identical.  When on, every leaf's luma residual follows
  // write_tx_partition()/per-part transform blocks and the stored per-part
  // TX types.
  bool tx_mode_select = false;
  bool reduced_tx_part_set = false;
  // Opt-in §5.18.2 allow_tcq (--video-tcq).  When the sequence selected
  // choose_tcq_per_frame = 1 this mirrors the per-frame allow_tcq bit; the
  // coefficient writer applies the §5.20.7.27 TCQ syntax and state to every
  // luma TX_CLASS_2D block.
  bool allow_tcq = false;
  // Opt-in §5.4.8 enable_cctx (--video-cctx).  When set, plane 1 codes
  // cctx_type after its EOB and the U/V pair was quantised jointly.
  bool enable_cctx = false;
  // §5.18.2 frame-header reduced_tx_set (0 keeps the full TX-type sets).
  int reduced_tx_set = 0;
  // Multi-tile view: `width`/`height` and the block list are tile-local, while
  // the filter plan is frame-level.  `frame_mi_cols`/`frame_mi_rows` default to
  // this tile's grid (single-tile streams), and `tile_mi_row`/`tile_mi_col` are
  // the tile origin in frame MI coordinates.
  int frame_mi_cols = 0;
  int frame_mi_rows = 0;
  int tile_mi_row = 0;
  int tile_mi_col = 0;

  // §5.18.7.1/.8 opt-in segmentation + delta-Q state for this tile.  The
  // fields are inert (disabled/empty) on every default stream, so the writer
  // emits exactly the historical bits.  `delta_steps` holds one planned step
  // per superblock in raster order (the decoder resets CurrentQIndex at every
  // tile, so the plan never spans tiles).
  rc::SegmentationParams segmentation{};
  rc::DeltaQParams delta_q{};
  bool primary_ref_is_none = true;
  // §5.4.4 MaxSegments = 16 for this sequence (--video-segments 9..16).
  bool enable_ext_seg = false;
  // §5.20.7.2 previous-frame SegmentIds (PrevSegmentIds) for the temporal
  // segmentation update.  Empty means "all zero".
  std::vector<std::uint8_t> prev_segment_ids;
  std::vector<rc::DeltaQPlanStep> delta_steps;

  int sb_cols() const;
  int sb_rows() const;
  int superblock_count() const;
  // True when the tile codes the §5.20.5.7/.11 segmentation/delta-Q symbols.
  bool qp_enabled() const { return segmentation.enabled || delta_q.present; }
};

// Emit the range-coded tile payload for one frame (key or inter). The caller
// wraps it in a CLK or regular tile-group OBU with av2enc::video::make_frame_obu().
bool assemble_video_frame(const FramePlan& plan,
                       std::span<const BlockSyntax> blocks,
                       std::span<const std::int32_t> levels,
                       const VideoConfig& config,
                       std::vector<std::uint8_t>& tile_payload,
                       std::string& error, std::size_t* tile_bytes = nullptr,
                       std::span<const std::uint16_t> cdf_seed = {},
                       std::vector<std::uint16_t>* cdf_final = nullptr,
                       const post::VideoFilterPlan* filters = nullptr);

// Opaque packed §8.2 CDF image (no neighbour/context state).  Used by the
// cross-frame continuity wiring in video_frame.cpp: the encoder seeds a frame
// from the primary reference's saved CDFs and exports the adapted tile CDFs so
// the average-CDF bank can run §8.2.4 / §7.5.
std::size_t cdf_image_words();
bool cdf_image_defaults_for_base_q_idx(int base_q_idx,
                                       std::vector<std::uint16_t>& out,
                                       std::string* error = nullptr);
bool cdf_image_tables(std::vector<std::uint16_t>& frame,
                      std::vector<std::uint16_t>& tile,
                      std::vector<std::uint16_t>& saved,
                      std::vector<std::uint16_t>& defaults,
                      std::vector<entropy::avg_cdf::CdfTable>& tables,
                      std::string* error = nullptr);

int y_mode_joint(int av2_mode, int angle_delta);
void build_scan(int w, int h, std::span<std::uint16_t> scan);

}  // namespace av2enc::video::syntax
