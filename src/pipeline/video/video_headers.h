#pragma once

#include "core/bitstream/bitwriter.h"
#include "codec/rate/rc/rc.h"
#include "codec/transform/delta_q/dcdeltaq.h"
#include "formats/format.h"
#include "pipeline/video/post/video_filter_plan.h"
#include "pipeline/video/post/video_loop_filters.h"
#include "pipeline/video/video_gm_types.h"

#include <cstdint>
#include <span>
#include <vector>

// AV2 v1.0.0 OBU/header writer for the M2 P-frame path.
//
// This is deliberately a *new* writer rather than an edit to the M1 header
// writer: M1 is a single still picture (single_picture_header_flag = 1), while
// a P-frame stream needs the multi-frame sequence header, order hints,
// reference slots and the inter-frame header fields.  The keyframe tile
// payload is still produced by the M1 still encoder and wrapped here.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.4.1, §5.4.6, §5.18.2, §5.19.
namespace av2enc::video {

struct VideoConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  // §5.4.1 seq_tier (present only when seq_level_idx > 3 and
  // single_picture_header_flag == 0).  Off is main tier, the historical value;
  // --video-tier high selects the high tier for levels >= 4.0.
  bool seq_tier = false;
  // §5.4.1 seq_header_id.  The default 0 is the historical single sequence
  // header; --video-multi-sequence carries a second header with id 1.
  std::uint32_t seq_header_id = 0;
  // §5.4.1 seq_profile_idc override. -1 derives the profile from the pixel
  // format (the historical path). The SVC embedded-layer ladder raises it
  // because Annex A caps seq_max_mlayer_cnt per profile (0 -> 1 embedded
  // layer, 1 -> 2, 2 -> 3).
  int seq_profile_override = -1;
  // §6.4.1 Table 6.2 / 6.3.  The defaults are the historical 8-bit 4:2:0
  // header (`chroma_format_idc = 0`, `bit_depth_idc = 1`).
  fmt::ChromaFormat chroma = fmt::ChromaFormat::k420;
  fmt::BitDepth depth = fmt::BitDepth::k8;
  // §5.18.6.1 base_q_idx: f(8) at 8-bit, f(9) at 10-bit.
  std::uint16_t base_q_idx = 90;   // §8.3.1 qctx slice from raw base_q_idx
  // §5.4.8 sequence delta-Q fields (--video-uv-delta-q).  The default is the
  // historical group: separate_uv_delta_q = 0, equal_ac_dc_q = 1,
  // base_uv_ac_delta_q = 0 (raw 23) and uv_ac_delta_q_enabled = 0.  A non-zero
  // UV/DC delta enables the full §5.4.8 group and §5.18.6.1 frame fields.
  dcdeltaq::SequenceDeltaQ seq_delta_q{};
  // §5.4.4 sequence_segment_config().  enable_ext_seg raises MaxSegments from
  // 8 to 16 (--video-segments 9..16).  Off keeps the historical zero bit.
  bool enable_ext_seg = false;
  // --video-seq-seg-info: seq_seg_info_present_flag = 1 writes
  // seq_allow_seg_info_change plus the full seg_info(MaxSegments) table, so
  // every frame can reuse it instead of coding a per-frame table.  Off keeps
  // the historical zero bit.
  bool seq_seg_info_present = false;
  bool seq_allow_seg_info_change = true;
  // The sequence segment table (SEG_LVL_* features, MaxSegments entries).
  rc::SegmentationParams seq_seg_info{};
  // §5.4.6 order_hint_bits_minus_1 + 1. Four bits covers a 16-frame GOP.
  std::uint8_t order_hint_bits = 4;
  // §5.4.6 explicit_num_ref_frames / num_ref_frames_minus_1.
  std::uint8_t num_ref_frames = 2;
  // §5.4.6 explicit_ref_frame_map.  1 keeps the historical per-frame
  // frame_explicit_ref_frame_map choice (the M2 explicit map); 0 makes every
  // inter frame use the §7.7 implicit mapping and codes no map bits at all.
  bool explicit_ref_frame_map = true;
  // §5.18.2 frame_explicit_ref_frame_map / num_total_refs.
  std::uint8_t num_total_refs = 1;
  // §5.4.6 seq_max_drl_bits_minus_1. m = value + 1 DRL indices are codable;
  // 4 -> m = 5, enough for the six-entry spatial stack (index 5 is the
  // read_drl_idx() exhaustion value).
  std::uint8_t max_drl_bits_minus_1 = 4;
  // §5.4.1 temporal layers.  The hierarchical-P mode (--video-tlayers) raises
  // max_tlayer_id and turns monotonic_output_order_flag off so frame OBUs can
  // carry obu_tlayer_id and code implicit_output_frame.  Defaults keep the
  // single-layer byte-identical stream.
  std::uint8_t max_tlayer_id = 0;
  bool monotonic_output_order_flag = true;
  bool tlayer_dependency_present_flag = false;
  // tlayer_dependency[curr][ref] for curr = 1..max_tlayer_id.
  bool tlayer_dependency[4][4] = {{false, false, false, false},
                                  {false, false, false, false},
                                  {false, false, false, false},
                                  {false, false, false, false}};
  // §5.4.6 compound tools.  --video-compound enables reference_select per
  // frame; the encoder only signals the AVERAGE (comp_group_idx == 0) blend, so
  // masked compound and CWP stay off in the sequence header.
  bool enable_masked_compound = false;
  bool enable_cwp = false;
  bool enable_imp_msk_bld = false;
  // §5.4.6 enable_df_sub_pu: when set, every inter frame header codes
  // allow_df_sub_pu inside deblocking_filter_params() (§5.18.7.9).  Off keeps
  // the historical inferred 0 and every existing header byte-identical.
  bool enable_df_sub_pu = false;
  // §5.4.6 enable_bawp (opt-in --video-bawp).  Off keeps the historical
  // sequence header.  When on, every inter frame codes allow_bawp and eligible
  // single-prediction blocks code use_bawp (§5.20.7.6).
  bool enable_bawp = false;
  // §5.4.6 motion-vector resolution tool set.  Each bit is off by default, so
  // the historical header writes the same three zero bits.  The block-level
  // syntax lives in mv_precision (§5.20.7.6); enable_flex_mvres additionally
  // makes the frame-header derivation set UsePerBlockMvPrecision = 1.
  bool enable_adaptive_mvd = false;
  bool enable_mvd_sign_derive = false;
  bool enable_flex_mvres = false;
  // §5.4.6 enable_opfl_refine: 0 = REFINE_NONE, 1 = REFINE_SWITCHABLE,
  // 2 = REFINE_ALL.  A non-AUTO sequence value means the frame header codes no
  // frame_opfl_refine_type() bits (§5.18.3.2).  --video-opfl signals
  // REFINE_SWITCHABLE so each eligible compound block codes use_optflow.
  int enable_opfl_refine = 0;
  // §5.4.6 enable_refinemv (--video-refine-mv).  With a non-GLOBALMV compound
  // block the decision is not switchable (§5.20.7.17), so every eligible
  // NEAR_NEARMV block uses §7.13.3.3 refine-MV.
  bool enable_refinemv = false;
  std::uint8_t num_same_ref_compound = 0;
  // §5.4.6 enable_refmvbank (--video-refmvbank): §7.12.2.21 bank candidates
  // are inserted at step 19 and the per-superblock bank is maintained from the
  // coded mode info.  Off keeps the historical sequence bit and stack.
  bool enable_refmvbank = false;
  // §5.4.6 enable_mv_traj (--video-mv-traj): §7.9.2 fills the trajectory MV
  // gaps and §7.12.2.14/.16 use TrajMv for derived candidates.  Off keeps the
  // historical sequence bit and the projection-only derived subset.
  bool enable_mv_traj = false;
  // §5.4.6 seq_enabled_motion_modes[INTERINTRA] (--video-compound-interintra).
  // When set, seq_frame_motion_modes_present_flag = 0, so every inter frame
  // inherits the sequence value and codes no extra bit; each eligible
  // single-reference block then codes the §5.20.7.14 motion-mode symbol.
  bool enable_interintra = false;
  // §5.4.3 sequence_partition_config() superblock size.  `use_128x128_superblock`
  // is only coded when `use_256x256_superblock` is 0.  The defaults are the
  // historical 64x64 stream.
  bool use_128x128_superblock = false;
  bool use_256x256_superblock = false;
  // §5.4.3 extended-partition tools (--video-ext-partitions).
  bool enable_ext_partitions = false;
  bool enable_uneven_4way_partitions = false;
  // §5.4.3 reduce_pb_aspect_ratio / max_pb_aspect_ratio_log2_minus_1
  // (--video-reduce-pb-aspect).  When set, MaxPbAspectRatio is
  // 1 << (log2_minus_1 + 1) = 2 or 4 instead of 8; the partition search runs
  // with the same limit so every coded leaf satisfies §6.19.4.1.
  bool reduce_pb_aspect_ratio = false;
  int max_pb_aspect_ratio_log2_minus_1 = 1;
  // §5.4.7 screen-content config.  `seq_screen_content_tools_present` writes
  // seq_choose_screen_content_tools = seq_choose_integer_mv = 1, so the frame
  // header decides allow_screen_content_tools / force_integer_mv.  Off keeps
  // the historical two zero bits and no frame-header screen syntax.
  bool seq_screen_content_tools_present = false;
  // --video-scc-force: seq_choose_screen_content_tools = 0 with
  // seq_force_screen_content_tools = 1, so every frame infers
  // allow_screen_content_tools = 1 and codes no allow bit.  seq_choose_integer_mv
  // stays SELECT unless `seq_force_integer_mv` is also set, so the keyframe
  // still decides force_integer_mv with one frame-header bit.
  bool seq_force_screen_content_tools = false;
  // --video-force-integer-mv (requires seq_force_screen_content_tools):
  // seq_choose_integer_mv = 0 / seq_force_integer_mv = 1.  Every frame infers
  // force_integer_mv = 1, the inter frame header codes no
  // use_qtr_precision_mv/allow_high_precision_mv pair, and every inter MV is on
  // the one-pel grid (§5.18.2 FrameMvPrecision = MV_PRECISION_ONE_PEL).
  bool seq_force_integer_mv = false;

  // True when the sequence enables the screen-content tools at all.
  bool screen_content_tools_enabled() const {
    return seq_screen_content_tools_present || seq_force_screen_content_tools;
  }
  bool disable_cdf_update = true;
  // §5.18.7.9 deblocking_filter_params(): when enabled the frame header codes
  // all four apply_deblocking_filter flags with DfDeltaQ = 0 (df_par_bits =
  // 2 from the sequence header).  Off keeps the historical all-zero row.
  bool enable_deblock = false;
  // §5.18.8.1 read_tx_mode(): opt-in TX_MODE_SELECT (--video-tx-search).  Off
  // keeps the historical tx_mode_select = 0 / TX_MODE_LARGEST bitstream.
  bool tx_mode_select = false;
  // §5.4.8 average-CDF config.  Opt-in (--avg-cdf); both bits are written only
  // in the non-single-picture header, so the pinned --video stream is
  // unchanged while the flags stay off.
  bool enable_avg_cdf = false;
  bool avg_cdf_type = true;
  // §5.4.1 timing block, non-single-picture only (--video-timing).  This is
  // the pinned-decoder-compatible form: decoder_model_info_present_flag = 1
  // with num_units_in_decoding_tick, and seq_decoder_model_info_present_flag
  // = 0 (pinned dav2d f9dafbc does not parse the nested §5.4.13 block).
  bool initial_display_delay_present = false;
  std::uint8_t initial_display_delay_minus_1 = 0;
  bool decoder_model_info_present = false;
  std::uint32_t num_units_in_decoding_tick = 0;
  // §5.4.13 nested seq_decoder_model_info() block.  Opt-in via
  // --video-decoder-model; the default form keeps the block absent so the
  // pinned dav2d timing stream is byte-identical.
  bool seq_decoder_model_info_present = false;
  std::uint32_t decoder_buffer_delay = 70000;  // 1/90000 s
  std::uint32_t encoder_buffer_delay = 20000;  // 1/90000 s
  bool low_delay_mode_flag = false;
  // §5.7 multi-frame header: 0 = no MFH (seq_header_id_in_frame_header is
  // coded); >0 makes every frame header reference that MFH and omit the
  // sequence-header id.  Opt-in via --video-mfh.
  std::uint32_t cur_mfh_id = 0;
  // §5.4.6 DRL reordering.  Default is DRL_REORDER_DISABLED.
  bool disable_drl_reorder = true;
  bool constrain_drl_reorder = true;
  // Sequence-level intra tool used by the M1 keyframe tile payload and by the
  // M2 intra fallback blocks.
  bool enable_cfl_intra = true;
  // §5.4.5 cfl_ds_filter_index (f(2), non-monochrome only).  Only the
  // chroma-from-luma loop-restoration taps read it on the format video core.
  int cfl_ds_filter_index = 0;
  // §5.4.5 enable_mhccp (--video-mhccp): CFL_MULTI hypotheses (cfl_index 2).
  bool enable_mhccp = false;
  // §5.4.5 enable_ibp (--video-ibp): the intra keyframe and every intra leaf
  // use the IBP predictor (modified edge filtering, the two-angle blend and
  // the §7.13.2.12 DC post-filter).  Off keeps the historical bit 0.
  bool enable_ibp = false;
  // §5.18.6.2 using_qmatrix (--video-qm) with the §9.4 default matrices.
  bool using_qmatrix = false;
  // §5.13 user-defined QM carrier (--video-user-qm): a
  // quantizer_matrix_obu() for level 0 precedes the key frame, so the decoder
  // sets QmDataPresent[0] and §7.14.4 reads UserQm[0] for <=8x8 blocks.
  bool qm_user_data = false;
  // §5.4.10 sequence_filter_config() flags for the opt-in M2 loop filters.
  // All default to the historical values, so a default stream keeps its bytes.
  bool enable_cdef = false;
  bool cdef_on_skip_txfm_always_on = false;
  bool cdef_on_skip_txfm_disabled = true;
  // CdefOnSkipTxfm mode the frame header derives: 0 DISABLED, 1 ALWAYS_ON,
  // 2 ADAPTIVE.  Mirrors the pair above.
  int cdef_on_skip_txfm_mode = 0;
  // §5.4.10 CCSO flags.  Off by default; `ccso_unit_matches_sb_size` is only
  // coded when `enable_ccso` is set.
  bool enable_ccso = false;
  bool ccso_unit_matches_sb_size = true;
  // §5.4.10 loop-restoration flags (non-separable Wiener / chroma LR).
  // §5.4.10 GDF flags.  Header-only mode (gdf_per_block = 0).
  bool enable_gdf = false;
  bool gdf_unit_matches_sb_size = true;
  bool enable_restoration = false;
  bool lr_disable_luma_pc_wiener = true;
  bool lr_disable_luma_wiener_ns = false;
  bool lr_uv_present = false;
  bool lr_disable_chroma_wiener_ns = false;
  // §5.4.1 layer limits for the SVC path.  The defaults reproduce the
  // single-layer header byte-for-byte (seq_lcr_id = 0, M = T = 1).
  std::uint8_t seq_lcr_id = 0;          // f(3), 1..7 with a global LCR
  std::uint8_t max_mlayer_id = 0;       // f(3)
  std::uint8_t seq_max_mlayer_cnt = 1;  // seq_max_mlayer_cnt_minus_1 + 1
  // Dependency-present flags.  The maps written are the spec defaults
  // (ref <= curr), so the bits are explicit but semantically inert.
  bool mlayer_dependency_present = true;
  // §5.4.6 enable_short_refresh_frame_flags (--video-short-refresh).  When
  // set, every non-key/non-switch frame codes has_refresh_frame_flags and, if
  // set, one `frame_to_refresh` slot index instead of the full mask.
  bool enable_short_refresh_frame_flags = false;
  // §5.4.6 enable_global_motion (opt-in --video-gme).  Off keeps the
  // historical sequence header and makes global_motion_params() a no-op.
  bool enable_global_motion = false;
  // §5.4.6 temporal-inter flags for the opt-in TIP path (--video-tip).  Off
  // keeps the historical sequence header bit-for-bit (both bits were already
  // written as 0, and the conditional TIP fields are absent when disabled).
  // enable_ref_frame_mvs gates the per-frame use_ref_frame_mvs bit; enable_tip
  // enables OBU_REGULAR_TIP frames and the TipFrameMode syntax slice.
  bool enable_ref_frame_mvs = false;
  // §5.4.6 enable_tip ...
  bool enable_tip = false;
  bool disable_tip_output = false;
  bool enable_tip_hole_fill = false;
  // §5.14/§5.18.10 film grain signalling (--video-grain).  Off keeps the
  // historical sequence/header bits (film_grain_params_present = 0) and every
  // frame header byte-identical.
  bool film_grain_params_present = false;
  // §5.4.8 enable_idtx_intra (--video-idtx): allows the 2-D identity TX type
  // (type 9) in the intra TX-type search.  The one-axis 1-D types (10..15)
  // remain unsignalled (pinned dav2d rejects negative H/V_DCT coefficients).
  bool enable_idtx_intra = false;
  // §5.4.5 enable_mrls (--video-mrl): multiple reference line selection for
  // directional intra luma.  Off keeps the historical sequence bit and every
  // block codes no mrl_index (the bit is absent).
  bool enable_mrls = false;
  // §5.4.5 enable_dip (--video-dip): data-driven intra prediction as a luma
  // DC_PRED alternative on the intra keyframe.  Off keeps the historical
  // sequence bit and codes no dip_mode_info().
  bool enable_dip = false;
  // §5.4.8 enable_intra_ist / enable_inter_ist (--video-ist).  The CPU video
  // writer codes the intra secondary transform on luma blocks; inter IST stays
  // unsignalled because the inter residual path has no sec_tx_type search.
  bool enable_intra_ist = false;
  bool enable_inter_ist = false;
  // §5.4.8 enable_cctx (--video-cctx): enables §7.14.6 chroma CCTX.
  bool enable_cctx = false;
  // §5.4.8 enable_tcq / choose_tcq_per_frame (--video-tcq).  The non-single
  // picture writer selects choose_tcq_per_frame = 1, so every frame header
  // codes allow_tcq (VideoFrameHeader::allow_tcq).  Parity hiding stays 0:
  // §5.4.8 forces it off when choose_tcq_per_frame is 0 and this encoder never
  // codes the hidden-parity coefficient symbols.
  bool enable_tcq = false;
  bool choose_tcq_per_frame = false;
  bool enable_parity_hiding = false;
  // §5.4.3 enable_sdp / enable_extended_sdp (--video-sdp).  The intra keyframe
  // codes each 64x64 SHARED node as a luma tree followed by a chroma tree.
  // enable_extended_sdp is 0: the encoder does not drive inter region_type.
  bool enable_sdp = false;
  bool enable_extended_sdp = false;
  // §5.4.8 enable_inter_ddt (--video-ddt): the decoder swaps ADST/FDST 1-D
  // kernels for DDTX/FDDT on inter blocks (length != 4).  There is no frame or
  // block syntax; the flag changes the transform the encoder must use.
  bool enable_inter_ddt = false;};

// §5.18.7.1 segmentation_params() / §5.18.7.8 delta_q_params() state for one
// frame.  Every field defaults to the historical disabled stream, so the bit
// writer emits exactly the previous three zero bits when neither tool is on.
struct VideoFrameQp {
  rc::SegmentationParams segmentation{};
  rc::DeltaQParams delta_q{};
  // §5.18.6.1 frame-level DC/AC deltas (--video-uv-delta-q).  The default
  // FrameDeltaQ codes no fields because SequenceDeltaQ enables none.
  dcdeltaq::FrameDeltaQ frame_delta_q{};
  // DerivedPrimaryRefFrame == PRIMARY_REF_NONE for this frame.  This is NOT
  // `keyframe`: a P frame whose only reference is an INTRA/key frame also
  // derives PRIMARY_REF_NONE, so segmentation_update_map /
  // segmentation_temporal_update are not coded for it (see
  // src/codec/entropy/deltaq/video/README.md).
  bool primary_ref_is_none = true;
  // §5.4.4 enable_ext_seg for this sequence: 1 raises MaxSegments to 16
  // (--video-segments 9..16).  It is not a frame-level bit; the frame header
  // writer uses it only to size seg_info().
  bool enable_ext_seg = false;
  // §5.4.4 sequence segment table (--video-seq-seg-info).  When present the
  // frame reuses the sequence seg_info(); with allow_change = 0 the reuse is
  // inferred and no reuse_seg_info bit is coded.
  bool seq_seg_info_present = false;
  bool seq_allow_seg_info_change = false;
  bool reuse_seg_info = true;
};

struct VideoFrameHeader {
  bool keyframe = false;
  // §5.18.2 seq_header_id_in_frame_header (uvlc).  Always 0 unless
  // --video-multi-sequence selects the second bank entry.
  std::uint32_t seq_header_id = 0;
  // Coded order_hint. The key frame is 0; P frames increment by one.
  std::uint32_t order_hint = 0;
  // Reference slot used by every block in this frame (single reference).
  std::uint8_t ref_slot = 0;
  // §5.18.2 explicit reference map.  For num_total_refs > 1 the slots are
  // listed newest first; every block then selects one with single_ref syntax.
  // `frame_explicit_ref_frame_map` is coded only when the sequence header sets
  // explicit_ref_frame_map; 0 selects the §7.7 implicit mapping, so
  // num_total_refs/ref_frame_idx carry the derived values for the block syntax
  // but no map bits are written.
  bool frame_explicit_ref_frame_map = true;
  std::uint8_t num_total_refs = 1;
  std::uint8_t ref_frame_idx[7] = {0, 0, 0, 0, 0, 0, 0};
  // §5.18.2 use_ref_frame_mvs.  Only coded when the sequence enables
  // enable_ref_frame_mvs; a TIP frame requires it (and the TIP motion-field
  // estimation, §7.9).  Absent (inferred 0) otherwise, so every historical
  // header keeps its bytes.
  bool use_ref_frame_mvs = false;
  // TIP-as-output frame (OBU_REGULAR_TIP).  The TIP block is written in place
  // of the tile-coded tail (§7.10.1); there is no tile payload.
  bool tip_frame = false;
  // §5.18.2 TipFrameMode == TIP_FRAME_AS_REF for a tile-coded inter frame
  // (--video-tip-block-ref).  The header codes the mode bit, the optional
  // allow_tip_hole_fill bit and tip_global_wtd_index; the tile then may code
  // RefFrame[0] == TIP_FRAME leaves.  Off keeps the historical
  // TIP_FRAME_DISABLED bit for every non-TIP frame.
  bool tip_as_ref = false;
  std::uint8_t tip_global_wtd_index = 2;  // Tip_Weighting_Factor index
  std::int16_t tip_global_mv_row = 0;     // TipGlobalMv, 1/8 luma samples
  std::int16_t tip_global_mv_col = 0;
  bool tip_sharp = true;                   // TipInterpFilter
  // §5.18.2 allow_tip_hole_fill: written only for a non-disabled
  // TipFrameMode when the sequence enables §5.4.6 enable_tip_hole_fill.
  bool allow_tip_hole_fill = false;
  // §5.18.2 frame MV precision.  Defaults are the M2 1/8-pel pair.
  bool use_qtr_precision_mv = false;
  bool allow_high_precision_mv = true;
  // §5.18.2 refresh_frame_flags. The key frame refreshes all slots.
  std::uint32_t refresh_frame_flags = 1;
  // §5.18.2 compact refresh signalling, coded only when the sequence sets
  // enable_short_refresh_frame_flags and the frame is neither SWITCH nor KEY.
  // `has_refresh_frame_flags = 1` codes one `frame_to_refresh` slot index;
  // 0 refreshes nothing.  Both fields are inert when the sequence bit is off,
  // so every historical header keeps the full f(NumRefFrames) mask.
  bool has_refresh_frame_flags = true;
  std::uint8_t frame_to_refresh = 0;
  // §5.18.3.3/.4 screen-content and IBC bits.  Only coded when the sequence
  // selected the per-frame form (VideoConfig::seq_screen_content_tools_present).
  // The morph carrier sets all of them on the intra keyframe.
  bool allow_screen_content_tools = false;
  bool force_integer_mv = false;
  bool allow_intrabc = false;
  bool allow_global_intrabc = true;
  bool allow_local_intrabc = false;
  // §5.18.2 immediate_output_frame.  Clearing it codes a hidden picture, which
  // a later show-existing-frame OBU can re-show (derive_sef_order_hint = 1).
  // monotonic_output_order_flag = 1 means implicit_output_frame stays absent.
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
  // §5.18.2 reference_select: 1 lets a block code comp_mode and two refs.
  bool reference_select = false;
  // §5.18.7.9 allow_df_sub_pu: coded only for an inter frame when the sequence
  // sets enable_df_sub_pu.  It makes §7.17 filter the §7.13.3.15 sub-PU edges.
  bool allow_df_sub_pu = false;
  // §5.18.2 allow_tcq.  Coded only when the sequence sets enable_tcq with
  // choose_tcq_per_frame = 1 (--video-tcq).  The encoder enables TCQ on every
  // frame, so the default is 1; with the sequence bit off the field is inert.
  bool allow_tcq = true;
  // §5.18.2 cross-frame CDF initialisation.  The defaults reproduce the
  // historical inter header (signal off, reset on).  The continuity path
  // signals primary_ref_frame = 0 (ref_frame_idx[0]) and clears the reset bit.
  bool signal_primary_ref_frame = false;
  bool disable_cross_frame_cdf_init = true;
  std::uint8_t primary_ref_frame = 0;
  // §5.18.9 global motion for this frame.  Defaults to "no model", which
  // writes `use_global_motion = 0` when the sequence bit is enabled and writes
  // nothing when it is disabled.
  GmFramePlan gm{};
  // §5.18.7.10 cdef_params() for this frame.  `enabled` is the
  // cdef_frame_enable bit; the strengths are only coded when it is set.
  post::VideoCdefPlan cdef;
  // §5.18.7.12 ccso_params() for this frame.
  post::VideoCcsoPlan ccso;
  // §5.18.7.11 lr_params() for this frame.
  post::VideoLrPlan lr;
  // §5.18.7.9 gdf_params() for this frame.
  post::VideoGdfPlan gdf;
  // §5.18.10.1 film_grain_config(): only coded when the sequence sets
  // film_grain_params_present.  Non-single-picture frames code apply_grain
  // f(1), then fgm_id uvlc() + grain_seed f(16) when it is 1.
  bool apply_grain = false;
  std::uint8_t fgm_id = 0;
  std::uint16_t grain_seed = 0;
  // §5.18.7.1/.8 quantization tail: segmentation table and delta-Q parameters.
  VideoFrameQp qp;
};

// §5.4.10 filter sequence flags for the opt-in M2 loop-filter chain.  Every
// key+P path (plain, B frames, temporal layers) must signal exactly the tools
// its per-frame plans use; the helper keeps those headers identical.
void apply_video_loop_filter_sequence_config(
    VideoConfig& config, const post::VideoLoopFilterOptions& loop_filters);

// Copies one frame's filter plan into the frame header (cdef_params(),
// ccso_params(), lr_params(), gdf_params()).
void apply_video_loop_filter_frame_header(const post::VideoFilterPlan& plan,
                                          VideoFrameHeader& header);

// sequence_header_obu() for single_picture_header_flag = 0, wrapped as an
// Annex B length-delimited OBU. The sequence enables only the tools the M1
// still tile payload and the M2 inter tile payload use.
std::vector<std::uint8_t> make_sequence_header_obu(const VideoConfig& config);

// frame_header_info() syntax bits for one frame, without the leading
// is_first_tile_group bit and without byte_alignment(). Returns the number of
// bits written.
std::size_t write_frame_header(BitWriter& writer, const VideoConfig& config,
                               const VideoFrameHeader& header);

// §5.19 tile_group_obu() header for a single tile:
//   is_first_tile_group f(1)
//   frame_header(isFirst = 1)
//   byte_alignment()
// The caller appends the tile payload.
std::size_t write_tile_group_header(BitWriter& writer, const VideoConfig& config,
                                    const VideoFrameHeader& header);

// Complete OBU_CLOSED_LOOP_KEY / OBU_REGULAR_TILE_GROUP payload, including
// the tile payload. The tile payload must already contain the §8.2.4
// exit_symbol() trailing bits.  `obu_tlayer_id` is the §5.2.2 layer id of the
// frame OBU (0 for a single-layer stream and always 0 for key/switch/RAS).
std::vector<std::uint8_t> make_frame_obu(const VideoConfig& config,
                                         const VideoFrameHeader& header,
                                         std::span<const std::uint8_t> tile_payload,
                                         std::uint8_t obu_tlayer_id = 0);

// Empty temporal delimiter OBU.
std::vector<std::uint8_t> make_temporal_delimiter_obu();

// Complete OBU_REGULAR_TIP payload (§6.2.2 type 14): frame_header(isFirst = 0)
// followed by §5.2.3 trailing bits.  There is no is_first_tile_group bit and
// no tile payload; §5.18.2 runs decode_frame_wrapup() directly.  `header`
// must have `tip_frame = true`.  Returns an empty vector on error.
std::vector<std::uint8_t> make_tip_frame_obu(const VideoConfig& config,
                                             const VideoFrameHeader& header,
                                             std::string& error);

}  // namespace av2enc::video
