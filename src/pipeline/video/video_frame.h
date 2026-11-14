// M2 two-frame P-frame integration on top of the M1 still encoder.
//
// Frame 0 is an intra keyframe. The M2-owned copy of the M1 intra syntax
// writer emits it (the M1 still integration TU was not linkable in this tree
// snapshot); the M1 tables, partition solver, intra prediction, transform and
// entropy coder are reused unchanged. Frame 1..N-1 are P frames that reference
// slot 0 and refresh slot 0.
//
// The P-frame decision path is deliberately small:
//   * partition tree from src/encoder/partition (same rules as M1)
//   * per-leaf inter candidates: GLOBALMV (zero MV), NEARMV (spatial MV stack)
//     and NEWMV (src/encoder/inter/me search quantised to 1/8 pel)
//   * per-leaf intra fallback: DC intra + DCT residual (M1 transform path)
//   * inter leaves default to skip_flag = 1; the gated `enable_inter_residual`
//     option codes a quantised inter residual (feature 1)
//
// This is a CPU correctness reference. The CUDA M2 tree is intentionally
// separate and not required for this gate.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "formats/format.h"
#include "codec/transform/delta_q/dcdeltaq.h"
#include "pipeline/video/post/video_loop_filters.h"

namespace av2enc::video {

// §5.18.2 REFS_PER_FRAME: NumTotalRefs is a 3-bit value, so one frame can
// address at most seven explicit reference slots.
inline constexpr int kRefsPerFrame = 7;
// §5.4.6 explicit_num_ref_frames max (4-bit `num_ref_frames_minus_1`).
inline constexpr int kMaxNumRefFrames = 16;
// The largest DPB this encoder model manages (the flat P chain and the frame
// type plans both stop at eight slots).
inline constexpr int kMaxRefSlots = 8;

// Pixel format of one sequence.  The default is the historical 8-bit 4:2:0
// stream, which keeps every existing bitstream byte-identical.  Any other
// format goes through the high-bit-depth / chroma video core
// (`src/pipeline/video/hbd/`).
struct VideoInputFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  // 8-bit packed planes (used when `format.depth == BitDepth::k8`).
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
  // High-depth packed `uint16_t` planes, LSB-aligned, used for 10-bit.
  std::span<const std::uint16_t> y16;
  std::span<const std::uint16_t> u16;
  std::span<const std::uint16_t> v16;
  fmt::Format format{};
};

struct VideoReconFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  // High-depth reconstruction planes for the format video core.
  std::vector<std::uint16_t> y16;
  std::vector<std::uint16_t> u16;
  std::vector<std::uint16_t> v16;
  fmt::Format format{};
};

struct VideoOptions {
  // Pixel format.  `{k420, k8}` is the historical M2 core; every other
  // combination is served by the format video core (10-bit / 4:2:2 / 4:4:4 /
  // 4:0:0).  The CLI only sets it for verified formats.
  fmt::Format format{};
  // §5.4.1 seq_profile_idc override (-1 = derive from the format).  The SVC
  // embedded-layer ladder needs the profile whose Annex A mlayer cap covers
  // the configured embedded layer count.
  int seq_profile_override = -1;
  // §5.18.6.1 base_q_idx.  MaxQ is 255 at 8-bit and 303 at 10-bit.
  std::uint16_t base_q_idx = 90;
  // Optional per-frame base_q_idx (one entry per input frame). Empty keeps the
  // fixed `base_q_idx` for every frame, which is the default byte-identical
  // path. Used by the opt-in --rc crf/vbr path; the frame header and the
  // coefficient-CDF qctx slice both follow the per-frame value.
  std::vector<std::uint8_t> frame_base_q_idx;
  // Opt-in §7.17 deblocking of every reconstructed frame (--video-deblock).
  // Off keeps every historical stream byte-identical.  When on, the frame
  // header signals all four apply flags with zero df_delta_q, and the
  // reconstructed frame is deblocked before it is emitted or stored as a
  // reference, so encoder and decoder references match.
  bool enable_deblock = false;
  bool enable_intra_fallback = true;
  // Opt-in full luma intra search (--video-intra-full): the still product's
  // 61-candidate SSE search instead of the DC-only fallback.  Off by default so
  // the shipped streams stay byte-identical.  Chroma stays DC_PRED.
  bool enable_intra_full = false;
  // Opt-in bottom-edge inter (--video-edge-inter): reconstruct a bottom leaf
  // whose coded extent runs past MiRows*4 into a padded reconstruction, and
  // allow inter prediction for the visible part.  The decoder's reference is
  // edge-extended past the visible picture, so MC reads the same padded
  // samples.  Off keeps every historical stream byte-identical.
  bool enable_edge_inter = false;
  // Opt-in §7.13.5 CfL chroma decision (--video-cfl): the M1 still product's
  // {DC, CfL} candidate set with derived alpha.  Off by default so every
  // historical stream stays byte-identical.
  bool enable_cfl = false;
  // Opt-in §7.13.6–.8 MHCCP (--video-mhccp): signals enable_mhccp and searches
  // the three CFL_MULTI directions after the derived-alpha hypothesis.  Needs
  // enable_cfl; off keeps every historical stream byte-identical.
  bool enable_mhccp = false;
  // Opt-in §5.18.6.2 default-matrix QM (--video-qm): using_qmatrix = 1 with
  // level 0 (the §9.4 default matrices, no user QM OBU).  The transform chain
  // dequantizes with the per-coefficient q2, so reconstruction stays exact.
  bool enable_qm = false;
  // Opt-in §5.4.8 / §5.20.7.27 trellis-coded quantisation (--video-tcq).  The
  // sequence selects choose_tcq_per_frame = 1 (so every frame codes allow_tcq)
  // and enable_parity_hiding = 0.  Luma TX_CLASS_2D blocks only.
  bool enable_tcq = false;
  // Opt-in §5.4.8 / §7.14.6 cross-component transform (--video-cctx).  U/V are
  // quantised jointly and plane 1 codes cctx_type.
  bool enable_cctx = false;
  // §5.13 user-defined QM carrier (--video-user-qm): implies enable_qm and
  // emits one quantizer_matrix_obu() with an identity (all-32) user matrix for
  // level 0 before the key frame.  §7.14.4 then reads UserQm[0] for every
  // <=8x8 transform.  Pinned dav2d rejects QM OBUs; gate with avmdec.
  bool user_qm = false;
  bool enable_newmv = true;
  bool enable_nearmv = true;
  // Opt-in §5.4.6 enable_ref_frame_mvs / §5.18.2 use_ref_frame_mvs and the
  // §7.9 + §7.12.2.7/.8 temporal MV candidates.  Off keeps every historical
  // stream byte-identical (the sequence header omits the bit).
  bool enable_ref_frame_mvs = false;
  // Opt-in §5.4.6 enable_short_refresh_frame_flags (--video-short-refresh):
  // every P frame codes has_refresh_frame_flags + frame_to_refresh instead of
  // the full mask.  Off keeps the historical sequence bit and frame headers
  // byte-identical.
  bool short_refresh = false;
  // Opt-in §5.4.6 enable_refmvbank (--video-refmvbank): maintain the §7.12.2.21
  // per-reference bank and insert its candidates at step 19.
  bool refmvbank = false;
  // Opt-in §5.4.6 enable_mv_traj (--video-mv-traj): §7.9.2 trajectory fill and
  // the §7.12.2.14/.16 TrajMv derived candidates.
  bool mv_traj = false;
  // Opt-in §5.4.6 num_same_ref_compound > 0 (--video-same-ref-compound):
  // compound blocks may code the same reference in both lists.  Requires
  // --video-compound and two references.
  bool same_ref_compound = false;
  // Gate stress mode: prefer a NEARMV candidate inserted by the temporal scan
  // whenever one exists.  Proves the coded stream actually depends on the
  // §7.12.2.7/.8 entries (never used for product quality measurements).
  bool ref_mvs_stress = false;
  // Opt-in compound refinement carrier.  `compound_near_near` adds the
  // §5.20.7.6 NEAR_NEARMV candidate on a two-reference B frame (the
  // GLOBAL_GLOBALMV subset is the historical --video-compound surface).
  // `opfl_refine_type` mirrors §5.4.6 enable_opfl_refine (0 NONE,
  // 1 SWITCHABLE, 2 ALL) and `enable_refinemv` mirrors enable_refinemv.
  // --video-opfl / --video-refine-mv own this pair; both are CPU-only.
  bool compound_near_near = false;
  int opfl_refine_type = 0;
  bool enable_refinemv = false;
  // §5.4.6 / §5.20.7.6 / §7.13.3.14 compound prediction-surface overlays on
  // the NEAR_NEARMV carrier.  `enable_cwp_weighted` signals enable_cwp and
  // searches the non-equal Cwp_Weighting_Factor rows; `enable_imp_msk_bld`
  // signals enable_imp_msk_bld and derives the §7.13.3.14 implicit mask.
  // Both default off, so every historical stream is byte-identical.
  bool enable_cwp_weighted = false;
  bool enable_imp_msk_bld = false;
  // §5.4.6 enable_df_sub_pu / §5.18.2 allow_df_sub_pu: code the frame-level
  // sub-PU deblocking bit and feed the §7.13.3.15 sub-PU grid to §7.17.
  bool enable_df_sub_pu = false;
  bool force_global_mv = false;
  bool force_intra = false;
  bool force_single_block = false;
  // Feature 1: inter residual.  Off by default; the baseline is skip-only.
  // With it on, the skip/residual and intra/inter decisions use a rate-aware
  // RD cost weighted by inter_rd_lambda instead of the historical SSE-only
  // comparison.
  // When true, a non-skip inter leaf codes transform coefficients for all
  // three planes.  force_inter_residual bypasses the rate heuristic and codes
  // residual on every inter leaf (gate stress mode).
  bool enable_inter_residual = false;
  // Distortion-per-bit weight for the opt-in residual RD decision.  Tuned on
  // 360p content: 4 keeps the default-path quality while cutting the stream
  // roughly 3x; the historical implicit value was ~0.5*(qlookup/8)^2 (2178
  // at QP 90), which made every residual look unaffordable.
  double inter_rd_lambda = 4.0;
  bool force_inter_residual = false;
  // Opt-in §5.18.8.1 TX_MODE_SELECT luma TX size/type/partition search
  // (--video-tx-search).  Off keeps every historical stream byte-identical;
  // when on, each coded luma residual evaluates the §5.20.6/§5.20.8 candidate
  // sets and signals the RD-best partition/type through the existing writer.
  bool enable_tx_search = false;
  // Price TX-search candidates with the §8.3.2 coefficient-context model.
  // Set by --video-rd-preset quality (or its explicit
  // --video-tx-context-rate spelling).  Off keeps the historical
  // `1 bit per nonzero` rate for standalone --video-tx-search, which is what
  // the CUDA device TX module still implements, so CPU/GPU parity holds.
  bool tx_context_rate = false;
  // SSE-per-bit weight for the TX search (Cost = SSE + lambda * rate_bits).
  // The module's rate is a real bit count, so this is much larger than
  // inter_rd_lambda (which weights the historical estimate_tx_bits heuristic).
  // 256 measured best on the 360p sample at QP 90: with the inter residual
  // tool it lowers rate and raises PSNR slightly; the default skip-only path
  // is rate-neutral because it has no inter residual leaves to search.
  double tx_rd_lambda = 256.0;
  // Opt-in rate-aware §5.20.3.1 partition tree (--video-partition-rd, also set
  // by --video-rd-preset quality).  Off keeps the historical cost, which mixes
  // a source-variance leaf distortion with a SAD-domain signal cost; that
  // signal cost exceeds the maximum 64x64 variance reduction above q≈163, so
  // every optional split is unaffordable and the P-frame tree collapses to one
  // 64x64 leaf per superblock.  When on, every candidate is costed in the SSE
  // domain and each leaf also pays an estimated mode/MV/skip syntax rate:
  //   leaf = variance + lambda_sse * leaf_bits,  split = lambda_sse * 2 bits
  // with lambda_sse calibrated to the measured local RD slope at the ~100 kbps
  // operating point.  All pinned default streams keep the historical path.
  bool partition_rd = false;
  int min_block_size = 16;
  int max_partition_depth = 4;
  // §5.4.3 sequence_partition_config() superblock size (--video-sb128 /
  // --video-sb256).  `sb256` implies a 128x128 SbSize for intra frames and
  // 256x256 for inter frames; the encoder keeps every leaf <= 64x64 under both
  // flags so the existing leaf coder is untouched.  Both false keeps the
  // historical 64x64 stream byte-identical.
  bool sb128 = false;
  bool sb256 = false;
  // §5.4.3 enable_ext_partitions + enable_uneven_4way_partitions
  // (--video-ext-partitions).  The search may use HORZ_3 / VERT_3 and the
  // uneven 4-way families; leaves still own their chroma and stay >= 8x8.
  bool ext_partitions = false;
  int superblock_pixels() const {
    return sb256 ? 256 : (sb128 ? 128 : 64);
  }
  // ME integer search radius in luma samples.
  int me_range = 16;
  // Opt-in ME refinement: add the normative 1/8-pel subpel stage.
  bool me_subpel_eighth = false;
  // References per P frame (NumTotalRefs).  1 keeps the historical M2 stream;
  // 2..7 enables per-block single_ref selection over the most recent slots.
  int num_total_refs = 1;
  // Fixed refresh/reference slot for the single-reference P chain.  -1 keeps
  // the historical rotating `i % NumRefFrames` slot.  The SVC embedded-layer
  // path pins the base chain to slot 0 so the refinement layers own slots
  // 1..M-1 and never collide with a rotated base picture (a collision makes an
  // m0 frame reference a slot holding m1, which §6.17.2 forbids).
  int fixed_write_slot = -1;
  // §5.4.6 NumRefFrames: the size of the reference-slot pool.  0 derives the
  // historical value (2 for the single-reference chain, 8 for the
  // multi-reference model), so every existing stream stays byte-identical.
  // An explicit 4..8 selects the spec multi-slot surface and must be at least
  // num_total_refs.
  int num_ref_frames = 0;
  // §5.4.6/§5.18.2 implicit reference map (--video-implicit-ref-map).  Sets
  // sequence explicit_ref_frame_map = 0, so every inter frame uses the §7.7
  // derivation (NumTotalRefs and ref_frame_idx vary per frame from the slot
  // state) and codes no map bits.  The flat P chain, the B-frame schedule and
  // the hierarchical-P temporal layers derive their block-level reference list
  // from it; the frame-type wrappers (TIP/SWITCH/RAS/LTR/OLK/SEF/leading),
  // tiles, compound and the non-8-bit format cores keep the explicit map and
  // are rejected by name when combined with this option.
  bool implicit_ref_map = false;
  // B frames: N consecutive B frames per GOP (0 = the flat key + P chain).
  // Requires num_total_refs == 1 (the bframes plan owns the per-frame refs).
  int b_frames = 0;
  // Number of consecutive closed B-frame GOPs (default 1).  The input is
  // gops * (b_frames + 2) frames; every GOP starts with a CLK that resets the
  // §7.23 reference store.
  int video_gops = 1;
  // Compound prediction (reference_select = 1).  Requires num_total_refs == 2;
  // the gated subset is non-joint GLOBAL_GLOBALMV with the AVERAGE blend.
  bool compound = false;
  // Masked compound (§5.20.7.16 / §7.13.3.27-.30), gated behind `compound`.
  // `compound_wedge` searches the 68x2 wedge masks; `compound_diffwtd`
  // searches both difference-weight mask types.  Both default off so the
  // historical AVERAGE-only streams stay byte-identical.
  bool compound_wedge = false;
  bool compound_diffwtd = false;
  // Frame MV precision: 0 = 1/8 pel, 1 = 1/4 pel, 2 = 1/2 pel.
  int mv_precision = 0;
  // §5.4.6 adaptive MVD (--video-adaptive-mvd): every NEWMV block codes the
  // use_amvd symbol and, when set, the §5.20.7.20/.21 mv_joint/amvd_index form
  // instead of the shell MVD.  IsAdaptiveMvd pins MvPrecision to the frame
  // value and suppresses per-block precision symbols and sign derivation.
  bool enable_adaptive_mvd = false;
  // §5.4.6 MVD sign derivation (--video-mvd-sign-derive).  Eligible NEWMV
  // blocks omit the last nonzero mv_sign; §5.20.7.13 derives it from the
  // parity of the coded magnitudes, so the encoder forces that sign.
  bool enable_mvd_sign_derive = false;
  // §5.4.6 flexible MV resolution (--video-flex-mvres): UsePerBlockMvPrecision
  // = 1, and each NEWMV block may code use_most_probable_precision /
  // pb_mv_precision to select a coarser precision than FrameMvPrecision.
  bool enable_flex_mvres = false;
  // Normative DRL reordering (§7.12.2.19).  Off = DRL_REORDER_DISABLED.
  bool drl_reorder = false;
  bool drl_reorder_always = false;
  // Long-term references (§5.4.6 long_term_frame_id_bits = 1, §5.18.2
  // long_term_id_plus_1).  Off keeps the historical header (bits = 0).
  bool long_term = false;
  // Open random access: CLK + open-loop key + long-term reference retained
  // across the RAP (§7.4.4).  Implies long_term; requires exactly 4 frames.
  bool olk = false;
  // Open-GOP leading pictures between the OLK and the first regular frame.
  // 0 = off.  N needs at least N+2 input frames.
  int leading_frames = 0;
  // Re-show the last coded picture with a show-existing-frame OBU: the last
  // picture is coded hidden (immediate_output_frame = 0) and the SEF unit that
  // follows outputs it with derive_sef_order_hint = 1 (§5.18.2, §6.17.2).
  bool sef = false;
  // Random-access switch frame: frame 1 is OBU_SWITCH (§6.2.2 type 10).
  // Requires exactly 2 input frames (CLK + SWITCH).
  bool ra_switch = false;
  // Random access with a long-term reference: CLK + SWITCH + OBU_RAS_FRAME
  // (§6.2.2 type 21, §7.4.5).  Requires exactly 3 input frames.
  bool ras = false;
  // Hierarchical-P temporal layers (§5.4.1 max_tlayer_id, §5.2.2
  // obu_tlayer_id).  0 = off.  N = one closed GOP of N frames (4 or 8).
  int temporal_layers = 0;
  // Emit only the temporal layers <= this id (a decodable substream).  -1
  // keeps every layer.
  int tlayer_drop = -1;
  // Internal: code the last picture hidden.  Only the SEF wrapper in
  // video_frames.cpp sets this; it is not a CLI option.
  bool hide_last_picture = false;
  // Opt-in uniform tile grid (--video --tiles). 0/0 (the default) keeps the
  // single-tile writer and the byte-identical stream.  Every frame is split on
  // superblock boundaries and re-emitted through
  // post::assemble_video_tile_group_from_payloads(); intra prediction is
  // clamped to the containing tile so the reconstruction stays decoder-exact.
  int tile_cols = 0;
  int tile_rows = 0;
  // Host worker count for the CPU reference encoder's deterministic
  // parallel-for loops (independent tile encoders and batched motion search).
  // 1 (the default) keeps the historical serial path; 0 means "auto"
  // (std::thread::hardware_concurrency).  Results are byte-identical for
  // every value: work is partitioned up front and merged in index order.
  unsigned cpu_threads = 1;
  // SVC sequence-header layer limits (§5.4.1).  Defaults reproduce the
  // single-layer header (seq_lcr_id = 0, M = T = 1).
  std::uint8_t seq_lcr_id = 0;
  std::uint8_t max_tlayer_id = 0;
  std::uint8_t max_mlayer_id = 0;
  std::uint8_t seq_max_mlayer_cnt = 1;
  // §5.4.1 high tier (--video-tier high).  Off keeps seq_tier = 0.
  bool high_tier = false;
  // §5.4.1/§5.18.2 multi-sequence surface (--video-multi-sequence): carry a
  // second sequence header with id 1 and select it with
  // seq_header_id_in_frame_header.  Off keeps the single id-0 header.
  bool multi_sequence = false;
  // §5.8/§5.10/§5.11 full LCR/OPS carrier (--video-lcr-full): activates a
  // global LCR (seq_lcr_id = 1) and emits a global LCR, a local LCR for
  // xlayer 0, a global OPS and a local OPS before the key frame.
  bool lcr_full = false;
  // Opt-in §8.2.4 average-CDF / §7.5 frame-end bookkeeping (--avg-cdf).
  // enable_avg_cdf/avg_cdf_type are then signalled in the sequence header.
  bool avg_cdf = false;
  // Opt-in §8.2.6 adaptation in the video tile writer (--adaptive-cdf with
  // --avg-cdf).  Off keeps disable_cdf_update = 1.
  bool adaptive_cdf = false;
  // Opt-in §5.18.2 cross-frame CDF carry (--cdf-continuity): each P frame
  // installs the primary reference's saved CDFs, signals primary_ref_frame = 0
  // and clears disable_cross_frame_cdf_init, so the decoder loads the same
  // context.
  bool cdf_continuity = false;
  // Opt-in §5.4.1/§5.4.12/§5.4.13 timing: decoder_model_info_present_flag with
  // num_units_in_decoding_tick plus a content-interpretation OBU carrying
  // timing_info().  Pinned dav2d accepts this form (the nested
  // seq_decoder_model_info block and BRT are rejected).
  bool video_timing = false;
  std::uint32_t timing_num_units_in_display_tick = 0;
  std::uint32_t timing_time_scale = 0;
  // Opt-in §5.4.13 seq_decoder_model_info() nested block
  // (--video-decoder-model).  Requires video_timing; the delays are 1/90000 s.
  bool video_decoder_model = false;
  std::uint32_t decoder_buffer_delay = 70000;
  std::uint32_t encoder_buffer_delay = 20000;
  bool low_delay_mode = false;
  // Opt-in §5.7 multi-frame header on the non-single-picture path
  // (--video-mfh).  One MFH (id 1) in the key temporal unit; every frame
  // header then sets cur_mfh_id = 1 and omits seq_header_id_in_frame_header.
  bool mfh = false;
  // Global motion (§5.18.9, §7.12.2, §7.13.3.19-.24).  Off by default: the
  // historical stream keeps `enable_global_motion = 0` and every GLOBALMV
  // candidate uses a zero MV.  `enable_gme` estimates one model per explicit
  // reference from a deterministic integer block-MV grid and signals it in the
  // frame header.  `enable_warp` additionally predicts GLOBALMV blocks with the
  // §7.13.3.19/.20 warped predictor when the model is non-identity (otherwise
  // the projected §7.12.2.2 centre MV drives ordinary translational MC).
  bool enable_gme = false;
  bool enable_warp = false;
  // §5.20.7.6 / §7.13.3.25 block adaptive weighted prediction (--video-bawp).
  // Off keeps every historical stream byte-identical; when on, eligible
  // single-prediction luma blocks evaluate the fitted BAWP predictor and code
  // use_bawp.  Chroma BAWP is never signalled (use_bawp_chroma = 0).
  bool enable_bawp = false;
  // §5.20.5.4 / §7.13.3.26 morphological prediction (--video-morph).  Codes
  // the screen-content + intra-block-copy mode-info path on the intra
  // keyframe, signals `morph_pred`, and adapts the IBC copy with the BAWP
  // template fit.  Requires enable_bawp (the sequence bit that makes
  // `morph_pred` signallable); off keeps every historical stream unchanged.
  bool enable_morph = false;
  // §5.20.8 luma palette selection on the intra keyframe (--video-palette).
  // Signals the §5.4.7 per-frame screen-content form and codes
  // palette_mode_info()/palette_tokens() for every leaf the analysis selects
  // (palette leaves are DC + eob=0).  Off keeps every historical stream
  // byte-identical.
  bool enable_palette = false;
  // §5.20.5.3/.4 general intra block copy on the intra keyframe (--video-ibc).
  // Same mode-info path as the morph carrier without morph_pred.  Off keeps
  // every historical stream byte-identical.
  bool enable_ibc = false;
  // --video-scc-force: write the §5.4.7 forced screen-content form
  // (seq_choose_screen_content_tools = 0, seq_force_screen_content_tools = 1).
  // Every frame infers allow_screen_content_tools = 1, so palette/IBC are
  // signalled without a per-frame allow bit and without the single-picture
  // inference.  Off keeps the historical two zero bits.
  bool seq_force_screen_content_tools = false;
  // --video-force-integer-mv (requires seq_force_screen_content_tools): write
  // seq_choose_integer_mv = 0 / seq_force_integer_mv = 1.  Every frame then
  // derives force_integer_mv = 1 and FrameMvPrecision = one pel; the inter
  // search quantises MVs to the integer grid and the header codes no precision
  // pair.  Off leaves integer-MV choice to each frame.
  bool seq_force_integer_mv = false;
  // §5.4.3 sequence_partition_config() reduce_pb_aspect_ratio
  // (--video-reduce-pb-aspect).  When set, the header codes
  // max_pb_aspect_ratio_log2_minus_1 (0 -> 2:1, 1 -> 4:1) and the partition
  // search runs with that MaxPbAspectRatio.  Off keeps the historical zero bit
  // and MaxPbAspectRatio = 8.
  bool reduce_pb_aspect_ratio = false;
  int max_pb_aspect_ratio_log2_minus_1 = 1;
  // §5.20.7.14/.15 / §7.13.3.29/.30 inter-intra (--video-compound-interintra).
  // Codes seq_enabled_motion_modes[INTERINTRA], the single-reference
  // motion-mode/interintra_mode/wedge_interintra syntax, and blends the inter
  // prediction with an intra prediction.  Off keeps every historical stream
  // byte-identical.
  bool enable_interintra = false;
  // Opt-in TIP sequence (--video-tip, §7.10/§7.13.3.2).  Codes one future
  // anchor P and the remaining pictures as TIP_FRAME_AS_OUTPUT frames that
  // reference the anchor plus the previous picture.  Requires 3..8 frames, a
  // single tile and none of the frame-type/loop-filter/CDF-chain modes; off
  // keeps every existing stream byte-identical.
  bool enable_tip = false;
  // §5.4.6 enable_tip_hole_fill plus the per-frame allow_tip_hole_fill bit
  // (§5.18.2).  On for the product TIP stream: the encoder's §7.10.3/.4 setup
  // and the pinned decoder's post-estimation validity mask agree cell for cell
  // on a coded anchor field, so both sides fill the same holes.  The
  // --video-tip-no-hole-fill control keeps the pre-fix stream for A/B gates.
  bool tip_hole_fill = true;
  // §5.20.7.10 block-level TIP reference (--video-tip-block-ref): the tail P
  // of the --video-tip --video-ref-mvs schedule is coded with TipFrameMode =
  // TIP_FRAME_AS_REF and its last leaf as RefFrame[0] == TIP_FRAME.  Requires
  // the single-64x64-leaf partition so the block is deterministic.  Off keeps
  // the shipped TIP stream byte-identical.
  bool tip_block_ref = false;
  // Opt-in §5.14/§5.18.10 film grain signalling (--video-grain): sequence
  // header film_grain_params_present = 1, a film grain model OBU before a
  // frame whose model is new, and per-frame film_grain_config().  Composes
  // with the plain/B/temporal/TIP/tiled schedules and the frame-type wrappers
  // (long-term/OLK/leading/SEF/SWITCH/RAS).  Off keeps every historical
  // stream byte-identical.
  bool film_grain = false;
  // §5.4.8 enable_idtx_intra (--video-idtx): opt-in 2-D identity TX type in
  // the intra TX-type search.  Off keeps the sequence bit 0 and the historical
  // candidate filter, so --video-tx-search streams are unchanged.
  bool enable_idtx_intra = false;
  // §5.4.5 enable_mrls (--video-mrl): directional intra luma searches
  // mrl_index 1..3 with the §7.13.2.1 extra reference lines.  Off keeps every
  // historical stream byte-identical.
  bool enable_mrl = false;
  // §5.4.5 enable_ibp (--video-ibp): every intra leaf is predicted with the
  // §7.13.2 IBP semantics (modified edge filtering, two-angle blend, DC
  // post-filter).  There is no per-block IBP syntax bit; the decoder derives
  // the decision from the sequence flag, mode and angle_delta.  Off keeps the
  // historical bit 0 and every stream byte-identical.
  bool enable_ibp = false;
  // §5.4.5 enable_dip (--video-dip): a luma DC_PRED keyframe block may signal
  // §5.20.7.9 use_dip/dip_transpose/dip_mode and reconstruct with §7.13.2.3.
  // Off keeps every historical stream byte-identical.
  bool enable_dip = false;
  // §5.4.8 enable_intra_ist (--video-ist): luma intra residuals code
  // sec_tx_type / most_probable_stx_set and the encoder applies the §7.15.3
  // inverse during reconstruction.  Off keeps every historical stream
  // byte-identical.
  bool enable_ist = false;
  // §5.4.3 enable_sdp (--video-sdp): the intra keyframe codes a luma tree and a
  // chroma tree for every 64x64 superblock.  Off keeps the historical SHARED
  // tree and the enable_sdp = 0 sequence bit.
  bool enable_sdp = false;
  // §5.4.8 enable_inter_ddt (--video-ddt): the decoder reconstructs an inter
  // ADST/FDST 1-D component at length 8/16 with the DDTX/FDDT kernel, so the
  // encoder must select and apply the same transform.  Off keeps the bit 0.
  bool enable_inter_ddt = false;  // Opt-in M2 loop filters (--video-cdef, ...).  All default off, so a default
  // stream is byte-identical.  They run on the reconstructed frame before it
  // becomes the output picture and the reference for later frames.
  post::VideoLoopFilterOptions loop_filters;
  // Opt-in §5.18.7.1/§5.18.7.8 segmentation and delta-Q on the CPU --video
  // writer (--video-segmentation / --video-delta-q).  Both default off and
  // keep the historical (segmentation_enabled, using_qmatrix, delta_q_present)
  // = (0, 0, 0) header, so every existing stream stays byte-identical.
  //
  // enable_segmentation codes the §5.20.5.7/.8 / §5.20.7.2 segment id for
  // every leaf from a deterministic per-superblock luma-variance split.  The
  // table carries SEG_LVL_ALT_Q = 0 for segment 1, so the map is exercised
  // without moving the quantizer; per_segment_q supplies a real value.
  // enable_delta_q signals §5.18.7.8 delta_q_present plus per-superblock
  // CurrentQIndex updates on a raster-parity staircase.
  bool enable_segmentation = false;
  bool enable_delta_q = false;
  int delta_q_res = 0;  // 0..3
  // Opt-in real SEG_LVL_ALT_Q planning (--video-per-segment-q): segment 1 is
  // the low-variance superblocks and uses base_q_idx + seg_q_delta.
  // The block quantizer, reconstruction and deblocking metadata all use the
  // derived §7.14.2 effective qindex in inter and intra leaves.
  bool per_segment_q = false;
  // Qindex delta used by --video-per-segment-q and the --video-delta-q
  // staircase (--video-seg-delta).
  int seg_q_delta = 8;
  // §5.4.4 seq_seg_info_present_flag (--video-seq-seg-info): the sequence
  // header carries the full segment table and every frame reuses it (or, with
  // allow_change, codes reuse_seg_info).  Off keeps the historical zero bit.
  bool seq_seg_info_present = false;
  bool seq_allow_seg_info_change = true;
  // Opt-in §5.4.8/§5.18.6.1 separate UV/DC delta-Q (--video-uv-delta-q D).
  // U uses +D and V uses -D for both DC and AC, so separate_uv_delta_q = 1
  // codes diff_uv_delta = 1 and every chroma plane/component exercises its own
  // §7.14.2 quantizer.  Off keeps the historical sequence/frame header and a
  // single qlookup() step for all planes.
  bool enable_uv_delta_q = false;
  int uv_delta_q = 0;
  // §5.4.4 enable_ext_seg / MaxSegments = 16 (--video-segments 9..16).
  bool enable_ext_seg = false;
  int num_segments = 2;
  // Opt-in segmentation extensions.  `seg_q_delta` remains the ALT_Q value;
  // the skip/globalmv flags select the lowest-variance bucket for their
  // feature, and temporal_update is coded on P frames whose primary reference
  // is an earlier P frame.
  bool enable_seg_skip = false;
  bool enable_seg_globalmv = false;
  bool enable_seg_temporal = false;
  // Resolved §5.4.8/§5.18.6.1 delta-Q state.  Both default to the historical
  // disabled values.  The encoder derives the per-plane DC/AC quantizers from
  // these and the frame header writes them through the shared component.
  dcdeltaq::SequenceDeltaQ seq_delta_q{};
  dcdeltaq::FrameDeltaQ frame_delta_q{};
  // Debug trace to stderr.
  bool trace = false;
};

struct VideoSequenceResult {
  // One temporal unit per input frame (TD + [SEQ] + frame OBU).
  std::vector<std::vector<std::uint8_t>> units;
  // Decoder output pictures, in display/output order.  Frame types that code
  // non-output pictures (OLK, non-output leading) leave those pictures out, so
  // this can be shorter than the coded frame count.
  std::vector<VideoReconFrame> recon;
  // One entry per **coded** frame unit in decode order (an SEF unit counts as
  // one): index into `recon` when the unit outputs a picture, -1 when it is
  // decoded but never displayed.
  std::vector<std::int32_t> frame_output_index;
  std::vector<std::uint32_t> frame_bytes;
  std::vector<std::uint32_t> tile_bytes;
  std::vector<std::uint32_t> inter_blocks;
  std::vector<std::uint32_t> intra_blocks;
  // Number of §7.12.2.7/.8 temporal candidates the stack builder inserted
  // across the whole sequence (0 unless enable_ref_frame_mvs was on).
  std::uint64_t temporal_candidates = 0;
  // Number of coded inter leaves whose selected NEARMV candidate or NEWMV
  // predictor came from a §7.12.2.7/.8 temporal entry (gate evidence).
  std::uint64_t temporal_selected = 0;
  // Number of coded compound leaves that selected a same-reference (§5.4.6
  // num_same_ref_compound > 0) pair.  Zero unless --video-same-ref-compound.
  std::uint64_t same_ref_blocks = 0;
  // §7.12.2.21 bank candidates inserted / selected (0 unless
  // --video-refmvbank).
  std::uint64_t refmvbank_candidates = 0;
  std::uint64_t refmvbank_selected = 0;
  // §7.12.2.14/.16 trajectory-derived candidate insertions (0 unless
  // --video-mv-traj).
  std::uint64_t mv_traj_derived = 0;
  // P frames coded with the compact §5.18.2 refresh form.
  std::uint64_t short_refresh_frames = 0;
  // §5.20.7.6 MV-resolution decisions on coded NEWMV leaves: AMVD, a coarser
  // per-block precision, and a §5.20.7.13 derived MVD sign (gate evidence).
  std::uint64_t amvd_selected = 0;
  std::uint64_t flex_precision_selected = 0;
  std::uint64_t derived_sign_selected = 0;
  // §7.22 store evidence: 8x8 cells that needed the per-sub-block warp
  // projection, and wedge cells where the allow-list dropped a list.
  std::uint64_t ref_mvs_warp_cells = 0;
  std::uint64_t ref_mvs_wedge_dropped = 0;
  // Coded leaves with RefFrame[0] == TIP_FRAME (§5.20.7.10 block form, gate
  // evidence for --video-tip-block-ref).
  std::uint64_t tip_block_refs = 0;
};

// §5.4.6 NumRefFrames actually signalled for these options: the explicit
// `num_ref_frames`, otherwise 8 for the multi-reference model and 2 for the
// historical single-reference chain.
int resolved_num_ref_frames(const VideoOptions& options);

bool encode_video_sequence(std::span<const VideoInputFrame> frames,
                        const VideoOptions& options, VideoSequenceResult& result,
                        std::string& error);

// Cross-embedded-layer refinement sequence (SVC `--svc-embedded`, §5.4.1
// MLayerDependencyMap / §6.17.2 reference conformance).
//
// Codes every input frame as an inter picture that predicts from
// `references[i]`, the decoded picture of the lower embedded layer at the same
// display index.  `references` must have one same-resolution entry per frame.
// Each frame header references `reference_slot` and refreshes `refresh_slot`,
// so the lower layer's picture survives in the DPB while the refinement layer
// keeps its own slot.  The first frame is inter: §7.3.6 only allows the CLK in
// the lowest embedded layer of a temporal unit, and the caller emits every
// lower mlayer before this one, so the reference is already valid.
struct RefinementOptions {
  int reference_slot = 0;
  int refresh_slot = 1;
  // §5.4.6 NumRefFrames.  Must be > max(reference_slot, refresh_slot) and at
  // most 16.
  int num_ref_frames = 8;
};

bool encode_video_refinement_sequence(
    std::span<const VideoInputFrame> frames,
    std::span<const VideoReconFrame> references, const VideoOptions& options,
    const RefinementOptions& refinement, VideoSequenceResult& result,
    std::string& error);

// Format video core dispatch (10-bit 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit
// 4:2:2/4:4:4/4:0:0).  `encode_video_sequence()` calls this when
// `options.format` is not the historical 8-bit 4:2:0.  The CUDA sibling is
// `encode_video_sequence_gpu()` in the video GPU target.
bool encode_video_sequence_format(std::span<const VideoInputFrame> frames,
                                  const VideoOptions& options,
                                  VideoSequenceResult& result,
                                  std::string& error);

}  // namespace av2enc::video
