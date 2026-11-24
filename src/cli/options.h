// CLI options shared by the argument parser, validation, help and the
// still/video front ends.  Extracted verbatim from src/main.cpp: the field
// defaults, comments and the order of `specified` are part of the CLI
// contract, so this struct is moved, never rewritten.
#ifndef AV2ENC_CLI_OPTIONS_H
#define AV2ENC_CLI_OPTIONS_H

#include "frame/metadata/types/metadata_types.h"
#include "pipeline/video/video_frame.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  std::uint64_t max_frames = 0;  // 0 = all frames
  int qp = 90;
  bool adaptive_cdf = false;
  // §8.2.4/§7.5 average-CDF bookkeeping (opt-in; multi-tile stills now, video
  // once the cross-frame seed export is wired).
  bool avg_cdf = false;
  // §5.18.2 cross-frame CDF carry on --video (requires --avg-cdf).
  bool cdf_continuity = false;
  // Opt-in per-frame QP support for the average-CDF chain
  // (--video-cdf-per-frame-qp): the §8.2.2 CDF image resets from each frame's
  // coefficient qctx slice instead of one fixed base_q_idx.
  bool video_cdf_per_frame_qp = false;
  // §5.4.1/§5.4.12/§5.4.13 timing on --video.
  bool video_timing = false;
  // §5.7 multi-frame header on --video.
  bool video_mfh = false;
  // §5.4.1 high tier on --video (--video-tier high).
  bool video_tier_high = false;
  // §5.4.1/§5.18.2 multi-sequence bank on --video.
  bool video_multi_sequence = false;
  // §5.8/§5.10 full LCR/OPS carrier on --video.
  bool video_lcr_full = false;
  bool m1 = false;
  bool still_gpu = false;
  bool still_cfl = true;
  bool still_deblock = true;
  bool still_single = false;
  bool still_dc = false;
  bool still_full_intra = false;
  bool still_uv_full = false;
  bool still_rdo_intra = false;
  double still_rdo_scale = 0.125;
  bool still_rdoq = false;
  double still_rdoq_scale = 0.125;
  bool still_tx_type = false;
  double still_tx_type_scale = 0.125;
  bool still_cdef = false;
  bool still_cdef_on_skip_txfm = false;
  int still_cdef_strengths = 4;
  bool still_ccso = false;
  int still_ccso_bands = 6;
  bool still_sb128 = false;
  bool still_sb256 = false;
  bool still_ext_partitions = false;
  bool still_sub8 = false;
  bool still_sdp = false;
  bool still_segmentation = false;
  bool still_delta_q = false;
  int still_delta_q_res = 0;
  // Real per-SB segmentation with SEG_LVL_ALT_Q (opt-in).
  bool still_per_segment_q = false;
  int still_seg_delta = 8;
  bool lossless = false;
  bool lossless_force_fsc = false;
  bool still_content_interpretation = false;
  // Neutral metadata OBUs (§5.17).  All default off; the payloads are the
  // all-zero/minimal spec-valid forms and do not affect reconstruction.
  bool metadata_timecode = false;
  bool metadata_scan_type = false;
  // §6.16.5 content light level (`--metadata-cll [max:fall]`).
  bool metadata_cll = false;
  std::uint16_t metadata_max_cll = 0;
  std::uint16_t metadata_max_fall = 0;
  // §6.16.6 mastering display colour volume (`--metadata-mdcv [preset]`).
  bool metadata_mdcv = false;
  av2enc::hdr::MasteringDisplayColorVolume metadata_mdcv_value{};
  // §6.16.9 ICC profile (`--metadata-icc [file]`).
  bool metadata_icc = false;
  std::vector<std::uint8_t> metadata_icc_data;
  // §6.16.11 temporal point info (`--metadata-temporal-point [ticks]`).
  bool metadata_temporal_point = false;
  std::uint32_t metadata_frame_presentation_time = 0;
  // §6.16.13 decoded frame hash (`--metadata-frame-hash [per-plane]`).
  bool metadata_frame_hash = false;
  bool metadata_frame_hash_per_plane = false;
  // §6.16.8 banding hints (`--metadata-banding [neutral|source]`).
  bool metadata_banding = false;
  av2enc::metadata::BandingHints metadata_banding_value{};
  // §6.16.12 user data unregistered (`--metadata-user-data [uuid[:hex]]`).
  bool metadata_user_data = false;
  av2enc::metadata::UserDataUnregistered metadata_user_data_value{};
  // §5.4.13/§6.11 decoder model info and buffer removal timing (--video only
  // for the nested block; BRT is carried per temporal unit).
  bool video_decoder_model = false;
  std::uint32_t decoder_buffer_delay = 70000;
  std::uint32_t encoder_buffer_delay = 20000;
  bool low_delay_mode = false;
  bool video_brt = false;
  std::uint32_t brt_time = 0;
  bool still_qm = false;
  int still_qm_level = 3;
  bool still_user_qm = false;
  bool still_ist = false;
  double still_ist_scale = 0.125;
  bool still_tcq = false;
  double still_tcq_scale = 0.125;
  bool still_cctx = false;
  bool still_tx_partition = false;
  bool still_idtx = false;
  bool still_ddt = false;
  bool still_dc_ac_delta_q = false;
  bool still_ibp = false;
  bool still_mrl = false;
  int still_cfl_ds = 0;
  bool still_cfl_explicit = false;
  bool still_dip = false;
  bool still_palette = false;
  bool still_ibc = false;
  int still_ibc_search_step = 8;
  bool still_mhccp = false;
  bool still_ccso_edge = false;
  bool still_wiener = false;
  bool still_chroma_lr = false;
  bool still_pc_wiener = false;
  bool still_gdf = false;
  bool still_grain = false;
  bool still_grain_adaptive = false;
  int still_depth = -1;  // -1 = component default
  // Multi-frame inter (video) path.  Opt-in; every default still/basic stream
  // is untouched.
  bool video = false;
  // Opt-in CUDA M2 backend for --video (--video-gpu).  Implies --video.
  bool video_gpu = false;
  std::uint64_t p_frames = 0;  // 0 = every frame after the keyframe
  std::uint64_t b_frames = 0;  // B frames per GOP (0 = key + P chain)
  int video_refs = 1;          // NumTotalRefs (1..7)
  // §5.4.6 NumRefFrames (4..8).  0 = the derived historical value (2 for the
  // single-reference chain, 8 for the multi-reference model).
  int video_num_ref_frames = 0;
  // Documented rejected carrier for the §7.8 implicit reference lists.
  bool video_implicit_ref_map = false;
  bool video_compound = false;  // reference_select compound (needs refs 2)
  // §7.13.3.27/.28 masked compound, opt-in behind --video-compound.
  bool video_compound_wedge = false;
  bool video_compound_diffwtd = false;
  // §5.4.6 / §5.20.7.6 / §7.13.3.14 compound prediction-surface overlays on the
  // NEAR_NEARMV carrier: non-equal CWP weights, the derived implicit mask blend
  // and the §7.13.3.15 / §7.17 sub-PU deblock syntax.  All off by default.
  bool video_cwp_weighted = false;
  bool video_imp_mask = false;
  bool video_df_sub_pu = false;
  // Rejected carrier: §7.13.3.29-.30 inter-intra is a single-reference motion
  // mode (§5.20.7.14 motion_mode_allowed() requires !isCompound).
  bool video_compound_interintra = false;
  bool video_inter_residual = false;
  // Opt-in bottom-edge inter (--video-edge-inter).  CPU single-tile P chain
  // only; off keeps every historical stream byte-identical.
  bool video_edge_inter = false;
  // Opt-in bottom-edge residual mask (--video-edge-mask): zero the encoder-side
  // residual of the coded rows below the visible bottom edge.  Needs
  // edge-inter.
  bool video_edge_mask = false;
  // Opt-in leaf RD cost (--video-leaf-rd): one QP-derived lambda and real-ish
  // intra/inter syntax bits for the leaf decision.  Off keeps every historical
  // stream byte-identical.
  bool video_leaf_rd = false;
  // --fast-decode: the latest measured CPU-only algorithmic RD aggregate
  // (renamed from --nightly 2026-11-20, which stays accepted as an alias).
  // It is deliberately not the default and is rejected on --video-gpu, so the
  // GPU path keeps the historical CPU decisions and stream bytes.
  bool fast_decode = false;
  // --nightly-b: the measured chained single-key hierarchical-B aggregate
  // (edge path, key-QP offset, BAWP, partition-RD, per-layer QP, anchor
  // refresh).  Implies the chained --video-tlayers 8 form; the caller still
  // supplies --video-gops and the frame count.  Research opt-in, validated
  // with tools/validate_avmdec.py while the dav2d pin has the §7.21.6 drain
  // defect.
  bool nightly_b = false;
  // The chained tlayers form / B block selection. --nightly-b sets both; the
  // AV2ENC_TLAYERS_CHAIN / _B env gates remain for the raw spike flags.
  bool video_tlayers_chain = false;
  bool video_tlayers_b = false;
  // Sequential key-frame RD driver (AV2ENC_RD_DRIVER=1 is the env form; this
  // is the CLI/product form, set by --nightly-b). Key picture only.
  bool video_rd_driver = false;
  // Exact-rate / BA-shaped perceptual pipe (AV2ENC_EFF_PERC) and the
  // head/tail exact-rate TX windows (AV2ENC_RD_HEAD_TX / _TAIL_TX) as CLI
  // forms; --nightly-b sets the measured pipe + tail.
  bool video_eff_perc = false;
  bool video_tail_tx = false;
  bool video_head_tx = false;
  // Opt-in measured CPU RD preset (--video-rd-preset <name>).  Empty means no
  // expansion, so the default stream stays byte-identical.  The expansion only
  // sets existing CLI options and then the normal validation runs.
  std::string video_rd_preset;
  // Opt-in §5.18.8.1 TX_MODE_SELECT luma TX size/type/partition search.
  bool video_tx_search = false;
  // Opt-in keyframe intra luma TX partition with per-part prediction
  // (--video-key-intra-txpart).  Requires --video-tx-search.
  bool video_key_intra_txpart = false;
  // Price TX-search candidates with the §8.3.2 coefficient-context model.
  // The measured RD preset turns this on; standalone --video-tx-search keeps
  // the historical heuristic so the CUDA device TX module (which has no
  // context model yet) still matches the CPU reference byte for byte.
  bool video_tx_context_rate = false;
  double video_tx_lambda = 256.0;
  double video_key_tx_lambda = 0.0;  // 0 inherits --video-tx-lambda, or the
                                     // key fit under --video-qp-lambda
  double video_leaf_rd_lambda = 125.0;
  // Opt-in QP-scaled RD lambdas (--video-qp-lambda): replace the leaf, TX,
  // key-TX, RDOQ-scale and partition-fallback constants with per-site power
  // laws measured at both operating points.  Off = historical constants.
  bool video_qp_lambda = false;
  // Opt-in rate-aware §5.20.3.1 partition decision (CPU reference).  See
  // VideoOptions::partition_rd; --nightly enables it.
  bool video_partition_rd = false;
  // SSE-per-bit scale for the measured partition probe (lambda = scale*step²).
  double video_partition_lambda = 1.1e-5;
  // Probe depth bound (1 = superblock vs its first split level).
  int video_partition_depth = 1;
  // Opt-in §7.17 deblocking of every reconstructed video frame.
  bool video_deblock = false;
  // Opt-in §5.18.7.1/.8 segmentation + per-superblock delta-Q (CPU --video).
  bool video_segmentation = false;
  bool video_delta_q = false;
  int video_delta_q_res = 0;
  // Opt-in per-superblock adaptive quantisation (--video-aq): an
  // activity-driven delta-Q plan (implies delta_q_present).
  bool video_aq = false;
  double video_aq_strength = 1.0;
  int video_aq_clamp = 3;
  int video_aq_bias = 0;
  bool video_aq_bias_given = false;
  bool video_per_segment_q = false;
  int video_seg_delta = 8;
  // Opt-in §5.4.8/§5.18.6.1 separate UV/DC delta-Q (--video-uv-delta-q D).
  bool video_uv_delta_q = false;
  int video_uv_delta_q_value = 4;
  // Opt-in symmetric chroma quantiser offset (--video-uv-qp-offset N): both U
  // and V get the SAME delta, so negative N codes chroma finer than luma at the
  // same luma QP.  Unlike --video-uv-delta-q this does not swap bits between the
  // planes: it changes the chroma step itself, which is the only lever that
  // addresses chroma error concentrated in dark saturated regions (large DC
  // offsets where a coarse step reads as a visible hue shift).
  bool video_uv_qp_offset_given = false;
  int video_uv_qp_offset = 0;
  // Opt-in segmentation extensions (--video-seg-skip/--video-seg-globalmv/
  // --video-seg-temporal/--video-segments N).  All default off.
  bool video_seg_skip = false;
  bool video_seg_globalmv = false;
  bool video_seg_temporal = false;
  int video_segments = 0;  // 0 = historical two-segment planner
  bool video_gme = false;
  bool video_warp = false;
  bool video_bawp = false;  // opt-in §7.13.3.25 block adaptive weighted pred
  // Opt-in CPU-only BAWP carrier.  §5.20.7.6 does not code use_bawp on a
  // GLOBALMV leaf, so this re-expresses a GLOBALMV winner as NEARMV/NEWMV at
  // the same MV and reference before fitting.  It also permits
  // --video-inter-residual on the fitted predictor.  Kept separate from plain
  // --video-bawp because the CUDA BAWP backend is skip-only (CPU/CUDA parity).
  bool video_bawp_carrier = false;
  // Opt-in §5.4.6 enable_ref_frame_mvs + §7.9/§7.12.2.7/.8 temporal MVs.
  bool video_ref_mvs = false;
  bool video_ref_mvs_stress = false;
  // Opt-in §5.4.6 reference-management tools on the CPU --video writer.
  bool video_short_refresh = false;
  // Opt-in §5.18.2 compact reference signalling (--video-ref-signal-compact):
  // NumRefFrames = 1 + explicit_ref_frame_map = 0 + no primary-ref block, so a
  // single-reference P chain codes no reference fields at all.  See
  // docs/video-ref-signalling.md.
  bool video_ref_signal_compact = false;
  bool video_refmvbank = false;
  bool video_mv_traj = false;
  bool video_same_ref_compound = false;
  // Documented rejected carrier: §7.12.2/§7.22 BRU (pinned dav2d does not
  // parse enable_bru/use_bru).
  bool video_bru = false;
  // Documented rejected carriers.  Both tools decode with the pinned dav2d
  // through their component gates, but neither is reachable from the M2
  // inter product path (see the parse errors).
  bool video_morph = false;
  // §7.13.3.8-.12 / §7.13.3.3-.6 temporal refinement on the NEAR_NEARMV
  // compound carrier (CPU 8-bit 4:2:0; gate_video_opfl_refine.sh).
  bool video_opfl = false;
  bool video_refine_mv = false;
  // Opt-in screen-content tools on the CPU `--video` intra keyframe.  Both
  // default off so every historical stream is byte-identical.
  bool video_palette = false;     // §5.20.8 luma palette selection
  bool video_ibc = false;         // §5.20.5.3/.4 general intra block copy
  // §5.4.7 forced screen-content form on the palette/IBC/morph carriers.
  bool video_scc_force = false;
  // §5.4.7 seq_force_integer_mv (requires --video-scc-force).
  bool video_force_integer_mv = false;
  // §5.4.3 reduce_pb_aspect_ratio with max_pb_aspect_ratio_log2_minus_1.
  bool video_reduce_pb_aspect = false;
  // §5.4.4 seq_seg_info_present_flag (--video-seq-seg-info).  The frozen form
  // codes seq_allow_seg_info_change = 0, so every frame reuses the table with
  // no reuse_seg_info bit.
  bool video_seq_seg_info = false;
  bool video_seq_seg_frozen = false;
  // §5.4.2 seq_tile_info_present_flag (--video-tile-group-compact): the
  // sequence header carries the one-tile tile_params() and each frame infers
  // reuse_tile_info = 1 (§5.18.7.2), dropping 3 bits per frame header.
  bool video_tile_group_compact = false;
  int video_pb_aspect_log2_minus_1 = 1;  // 0 -> 2:1, 1 -> 4:1
  bool video_compound_near_near = false;  // §5.20.7.6 NEAR_NEARMV carrier
  bool video_tip = false;
  // A/B control for the §5.4.6 enable_tip_hole_fill bring-up: keeps the
  // pre-fix sequence header (hole fill off) so a gate can compare both masks.
  bool video_tip_no_hole_fill = false;
  // §5.20.7.10 block-level TIP reference carrier.
  bool video_tip_block_ref = false;
  bool video_intra_full = false;  // opt-in 61-candidate luma intra search
  bool video_cfl = false;         // opt-in §7.13.5 CfL chroma decision
  bool video_mhccp = false;       // opt-in §7.13.6-.8 MHCCP (implies CfL)
  bool video_qm = false;          // opt-in default-matrix QM
  bool video_user_qm = false;     // opt-in user-defined QM OBU (implies QM)
  bool video_tcq = false;         // opt-in §5.20.7.27 trellis TCQ
  bool video_rdoq = false;        // opt-in luma inter coefficient RDOQ
  // RDOQ SSE-domain lambda calibration.
  double video_rdoq_scale = av2enc::video::kVideoRdoqSseScale;
  bool video_cctx = false;        // opt-in §7.14.6 cross-component transform
  bool video_grain = false;       // opt-in §5.14/§5.18.10 film grain signalling
  bool video_lossless = false;    // opt-in Q=0 lossless P-frame writer
  int video_lossless_intra_period = 0;  // 1/N blocks intra (0 = all inter)
  bool video_idtx = false;        // §5.4.8 enable_idtx_intra (2-D IDTX)
  bool video_mrl = false;         // §5.4.5 enable_mrls (directional luma)
  bool video_ibp = false;         // §5.4.5 enable_ibp (intra bi-prediction)
  bool video_dip = false;         // §5.4.5 enable_dip (luma DC alternative)
  bool video_ist = false;         // §5.4.8 enable_intra_ist (luma intra)
  bool video_idtx_1d = false;     // documented rejected 1-D intra carrier
  bool video_sdp = false;         // §5.4.3 enable_sdp intra luma/chroma trees
  bool video_ddt = false;         // §5.4.8 enable_inter_ddt
  bool video_subpel_eighth = false;
  int video_mv_precision = 0;  // 0 = 1/8, 1 = 1/4, 2 = 1/2 pel
  bool video_adaptive_mvd = false;    // §5.4.6 enable_adaptive_mvd
  bool video_mvd_sign_derive = false;  // §5.4.6 enable_mvd_sign_derive
  bool video_flex_mvres = false;       // §5.4.6 enable_flex_mvres
  bool video_drl_reorder = false;
  bool video_drl_reorder_always = false;
  bool video_long_term = false;
  bool video_olk = false;
  int video_leading = 0;  // 0 = off
  bool video_sef = false;
  bool video_switch = false;
  bool video_ras = false;
  int video_tlayers = 0;   // 0 = off; N = one closed hierarchical-P GOP
  // Per-temporal-layer base_q_idx offsets (layers 0..3). Default off; the
  // chained --video-tlayers --video-gops form applies them in coding order.
  std::array<int, 4> video_tlayer_qp_offset = {0, 0, 0, 0};
  bool video_tlayer_qp_given = false;
  // Force the inter residual on the chained layer-0 anchors (drift refresh
  // probe; default off).
  bool video_tlayer_anchor_residual = false;
  // Selective anchor refresh: trigger the forced residual only when the
  // anchor's primary reference PSNR is below the key anchor by this threshold,
  // within a byte budget. 0 threshold = force every layer-0 anchor.
  double video_tlayer_refresh_threshold = 0.0;
  int video_tlayer_refresh_budget = 3250;
  bool video_lossless_b = false;  // --video-lossless: use the B schedule
  int video_tlayer_drop = -1;  // -1 = keep every layer
  int video_gops = 1;      // consecutive closed B-frame GOPs (default 1)
  bool video_force_single = false;
  bool video_skip_collapse = false;
  bool video_sb128 = false;
  bool video_sb256 = false;
  bool video_ext_partitions = false;
  int video_min_block = 16;
  int video_max_depth = 4;
  int video_me_range = 16;
  bool video_cdef = false;
  int video_cdef_strength = 1;  // CdefStrengths 1..8
  int video_cdef_damping = 3;   // CdefDamping 3..6
  // Price the §5.18.7.10 params row in the per-frame CDEF decision
  // (--video-cdef-rd).  Frames whose filter cannot pay for its own header code
  // the one-bit `cdef_frame_enable = 0` form instead.
  bool video_cdef_rd = false;
  double video_cdef_rd_lambda = 0.0;  // SSE per bit; 0 = zero-gain frames only
  // CDEF strength-search budget (0 full, 1 ladder, 2 step, 3 fast).  Only the
  // decision search changes; the filter and syntax are untouched.
  int video_cdef_search = 0;
  bool video_ccso = false;
  int video_ccso_bands = 4;  // CcsoMaxBandLog2 (band-offset-only form)
  // Rate-aware §7.19 CCSO decision (opt-in; the distortion-only fit used by
  // --video-ccso is untouched when this is false).
  bool video_ccso_rd = false;
  double video_ccso_lambda = 64.0;
  // Extra exact table refit + re-prune passes for --video-ccso-rd.
  int video_ccso_refit_passes = 0;
  // Price of one ccso_blk unit symbol in the prune decision.
  double video_ccso_unit_bits = 1.0;
  // Price the whole §5.18.7.12 row (fixed flags + the tu(7) table) in the CCSO
  // fit and choose ccso_max_band_log2 with the table (--video-ccso-header-rd).
  bool video_ccso_header_rd = false;
  // §5.4.5 cfl_ds_filter_index (0..3; 3 aliases 0).  Drives the chroma-from-
  // luma taps of the chroma loop restoration.
  int video_cfl_ds_filter = 0;
  bool video_wiener = false;
  bool video_pc_wiener = false;
  bool video_chroma_lr = false;
  bool video_gdf = false;
  bool video_trace = false;
  std::filesystem::path video_recon;
  // Per-frame rate control for --video. Default "cqp" keeps the fixed --qp
  // stream byte-identical; crf/vbr plan a base_q_idx per frame.
  std::string rc_mode = "cqp";
  bool rc_given = false;
  double rc_crf = 32.0;
  bool rc_crf_given = false;
  double rc_bitrate_kbps = 0.0;
  bool rc_bitrate_given = false;
  // Opt-in reference-intro QP offset (--video-key-qp-offset).  The key frame
  // takes --qp + N; a later frame whose source is pixel-identical to its
  // predecessor inherits that predecessor's QP, so a duplicated opening pair
  // (f1 == f0) is coded at one QP instead of splitting it.  A zero offset
  // reproduces the fixed-QP stream byte for byte.
  bool video_key_qp_offset_given = false;
  int video_key_qp_offset = 0;
  // Opt-in per-frame drift refresh (--video-drift-refresh).  The flat key+P
  // chain codes almost every P leaf as skip, so a prediction error the frame
  // inherits is never repaid.  When enabled, each inter frame measures its
  // primary reference's reconstruction error and, if it is below the
  // threshold, codes itself as a refresh (frame QP - N, residual on every
  // inter leaf) within one per-sequence payload allowance.  Off by default:
  // every pinned stream stays byte-identical.  CPU-only, single-tile,
  // low-delay flat chain, --rc cqp only.
  bool video_drift_refresh = false;
  double video_drift_refresh_threshold = 2.5;  // dB below the key
  int video_drift_refresh_qp = 0;
  int video_drift_refresh_budget = 1200;  // bytes per sequence
  // Opt-in §8.2.4 minimum tile termination (--video-tile-flush-min).  The
  // trailing '1' stays where §8.2.4 puts it; the fourteen zeros after it and
  // the byte padding are not written, which §8.2.4 permits down to
  // SymbolMaxBits == -14.  Saves 1-2 bytes per coded frame; off by default so
  // every pinned stream stays byte-identical.
  bool video_tile_flush_min = false;
  // Explicit off-switches for the overhead levers --nightly enables, so the
  // gates keep a real A/B control and a user can drop any of them.
  bool video_tile_flush_off = false;
  bool video_ref_signal_off = false;
  bool video_tile_group_off = false;
  bool video_ccso_header_off = false;
  // Ablation switches for the remaining --nightly-implied tools that a
  // reference-structure experiment has to drop (each owns a header field or a
  // block mode that the reference tools reject by name).  Off flags keep the
  // default and every pinned stream byte-identical; see
  // docs/reference-structure.md.
  bool video_edge_inter_off = false;
  bool video_edge_mask_off = false;
  bool video_bawp_off = false;
  // --video-compound-off: the A/B control for the compound arm that
  // --nightly-b enables (the pre-2026-11-21 85.472 B baseline).
  bool video_compound_off = false;
  // --video-compound-depth-off: the A/B control for the NEAR_NEARMV/CWP depth
  // arm on top of compound (restores the 86.805 TIP-only preset).
  bool video_compound_depth_off = false;
  bool video_drift_refresh_off = false;
  bool video_cdf_chain_off = false;
  bool video_key_qp_offset_off = false;
  bool video_deblock_off = false;
  bool video_loop_filters_off = false;
  bool video_tx_search_off = false;
  bool video_rdoq_off = false;
  bool video_partition_rd_off = false;
  std::uint64_t vram_cap_mib = 0;  // 0 = largest reservation the device can serve
  int bit_depth = 8;               // 8 (default) or 10; 12-bit is not signalable
  std::string chroma = "420";      // 420 (default), 422, 444, 400
  std::filesystem::path recon;     // optional encoder-reconstruction dump
  // Multi-tile stills. "1x1" (default) keeps the single-tile writer; every
  // other value selects the tiled writer and requires --no-deblock.
  std::string tiles = "1x1";
  // Host CPU workers for the CPU reference video encoder's deterministic
  // parallel-for loops (--cpu-threads).  1 (default) is the serial path;
  // 0 means auto (hardware concurrency).  Ignored by the GPU path.
  unsigned cpu_threads = 1;
  bool cpu_threads_given = false;
  // Host-only multi-GPU tile-plan dump. 0 = off (the default); never allocates
  // device memory and never writes a stream.
  int gpu_plan = 0;
  // Measured effort ranking (x264-style preset).  Every field is inert
  // unless --effort or --cpu-used is given; the default stream is
  // byte-identical because the selector is never called.
  bool effort_given = false;
  int effort_budget = 0;
  bool cpu_used_given = false;
  int cpu_used = -1;
  bool effort_plan = false;
  bool effort_active = false;
  std::string effort_report;
  // Option tokens seen on the command line (flags only, not their values).
  // Validation uses this to reject combinations in which a flag would
  // otherwise be accepted and silently ignored.
  std::vector<std::string_view> specified;

  // SVC layer encode (opt-in).  Any --svc-* flag sets svc_given; --oppoint,
  // --alllayers and --spatial-layers select the output subset.
  std::string svc_layers;
  bool svc_given = false;
  int svc_temporal = 1;
  int svc_embedded = 1;
  int svc_lcr_id = 1;
  int svc_oppoint = -1;
  bool svc_all_layers = false;
  int svc_spatial_layers = -1;
  bool help = false;
};

inline bool flag_given(const Options& options, std::string_view flag) {
  return std::find(options.specified.begin(), options.specified.end(), flag) !=
         options.specified.end();
}

enum class ParseStatus {
  kOk,
  kHelp,
  kError,
};

// Formats the format video core has been verified and gated on.  Everything
// else keeps the historical `--video is 8-bit 4:2:0` rejection, by name.
// Order of landing: 10-bit 4:2:0 first, then the remaining 10-bit chroma
// formats, then the 8-bit chroma formats.
inline bool video_format_wired(int bit_depth, const std::string& chroma) {
  if (bit_depth != 8 && bit_depth != 10) return false;
  return chroma == "420" || chroma == "422" || chroma == "444" ||
         chroma == "400";
}

// The chained single-key tlayers form: --nightly-b sets the option; the
// AV2ENC_TLAYERS_CHAIN env gate keeps the raw spike flags off the default
// surface.  The B block selection has the same two entry points.
inline bool tlayers_chain_enabled(const Options& options) {
  if (options.video_tlayers_chain) return true;
  const char* value = std::getenv("AV2ENC_TLAYERS_CHAIN");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

inline bool tlayers_b_enabled(const Options& options) {
  if (options.video_tlayers_b) return true;
  const char* value = std::getenv("AV2ENC_TLAYERS_B");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// cli/parse.cpp
ParseStatus parse_arguments(int argc, char** argv, Options& options,
                            std::string& error);

// cli/nightly.cpp
[[nodiscard]] bool apply_video_rd_preset(Options& options, std::string& error);
[[nodiscard]] bool apply_video_nightly(Options& options, std::string& error);
[[nodiscard]] bool apply_video_nightly_b(Options& options, std::string& error);

// cli/validate.cpp
ParseStatus validate_options(Options& options, std::string& error);

// cli/help.cpp
void print_usage(std::ostream& out);

#endif  // AV2ENC_CLI_OPTIONS_H
