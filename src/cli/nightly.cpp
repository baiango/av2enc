// --video-rd-preset and --fast-decode expansions.  Both only set existing CLI
// options; moved verbatim from src/main.cpp so the expanded flag set (and
// therefore the stream) is unchanged.
#include "cli/options.h"

#include <cstdlib>
#include <string>

// Expansion of the measured video RD presets (--video-rd-preset <name>).  The
// preset only sets existing CLI options; the caller then runs the normal
// combination validation, so a preset cannot reach a tool pairing that the
// individual flags reject.  No preset leaves every field untouched, so the
// historical default stream is byte-identical.
//
// The 'quality' row is measured on the CPU --video reference at the 240p30 /
// 100 kbps point (see docs/video-rd-preset.md): a single tile, P-only chain.
// The 'best' row is the measured flat lead stack: the whole --fast-decode
// aggregate plus the nine env-only RD tools, the drift-refresh allocation
// point and --video-key-qp-offset -39.  It is a LEAD (K 86.33, +0.097 over the
// 86.232 flat aggregate), not a KEEP.
bool apply_video_rd_preset(Options& options, std::string& error) {
  if (options.video_rd_preset.empty()) return true;
  const bool best = options.video_rd_preset == "best";
  if (options.video_rd_preset != "quality" && !best) {
    error = "--video-rd-preset expects 'quality' or 'best' (the measured rows)";
    return false;
  }
  // 'best' is the flat --fast-decode aggregate plus the env-only lead tools, so
  // it implies the video path; 'quality' is the plain preset and still requires
  // an explicit --video.
  if (!options.video && !best) {
    error = "--video-rd-preset applies to the CPU --video encoder; pass "
            "--video (the --still writer pins its own tool set)";
    return false;
  }
  if (options.video_gpu) {
    error = "--video-rd-preset is the measured CPU tool set; drop --video-gpu "
            "(its decisions are calibrated separately)";
    return false;
  }
  options.video_inter_residual = true;
  options.video_deblock = true;
  options.video_ccso = true;
  options.video_tx_search = true;
  options.video_intra_full = true;
  if (!best) return true;

  // 'best' also expands the --fast-decode aggregate through its one
  // implementation: this sets the flag, so the caller runs
  // apply_video_nightly() next, and that expansion accepts 'best' as its
  // preset row.
  if (options.rc_mode != "cqp") {
    error = "--video-rd-preset best is measured on the fixed --qp stream; use "
            "--rc cqp";
    return false;
  }
  options.video = true;
  options.fast_decode = true;
  // Reference-intro allocation: the lead stack's funded key spend.  An
  // explicit --video-key-qp-offset wins (0 is a valid ablation) and
  // --video-key-qp-offset-off drops it, exactly like --nightly-b.
  if (!options.video_key_qp_offset_given &&
      !options.video_key_qp_offset_off) {
    options.video_key_qp_offset = -39;
    options.video_key_qp_offset_given = true;
  }
  // The measured head/tail allocation point the lead stack was measured at;
  // an explicit CLI value wins.
  if (!flag_given(options, "--video-drift-refresh-threshold")) {
    options.video_drift_refresh_threshold = 1.5;
  }
  if (!flag_given(options, "--video-drift-refresh-budget")) {
    options.video_drift_refresh_budget = 3250;
  }
  // The five env-only tools.  They are read lazily on the first decision,
  // after this expansion and before the first encoder runs, so the static
  // configs they feed see the preset values.  An explicit env value wins,
  // including "0" for a caller ablation, so the stack stays inspectable.
  const auto set_gate = [](const char* name, const char* value) {
    if (std::getenv(name) == nullptr) {
      setenv(name, value, 1);
    }
  };
  set_gate("AV2ENC_RD_UVFULL", "1");
  set_gate("AV2ENC_RD_UVLEVEL", "1");
  set_gate("AV2ENC_RD_KEY_TX", "1");
  set_gate("AV2ENC_TX_DETAIL_SEARCH_TAIL", "1");
  set_gate("AV2ENC_TX_DETAIL_MIN", "0.50");
  // The measured joint P plan solve: one objective over partition and
  // TX/levels collapses the probe's over-priced coarse blocks.  +0.68 K
  // bracket / +0.92 fit at 100 kbps, but the serial raster walk costs
  // ~12 min per 60-frame 240p encode, so this is an opt-in research gate,
  // not the fast path (--fast-decode is unchanged).  An explicit env value
  // (including 0) wins and restores the 86.33 lead stack.
  set_gate("AV2ENC_RD_JOINT_P", "1");
  // The four CLI-form lead tools: the exact-rate pipe, the key-frame RD
  // driver and the display-order head/tail TX windows.
  options.video_eff_perc = true;
  options.video_rd_driver = true;
  options.video_head_tx = true;
  options.video_tail_tx = true;
  return true;
}

// --fast-decode expands the measured CPU-only algorithmic aggregate (renamed
// from --nightly 2026-11-20; --nightly is accepted as an alias and expands the
// same flag set).  It is the only place the new decision models are enabled
// together; the default path, --video-rd-preset quality and every GPU path
// keep the historical decisions.
bool apply_video_nightly(Options& options, std::string& error) {
  if (!options.fast_decode) return true;
  if (options.m1 || options.still_gpu) {
    error = "--fast-decode is a CPU --video mode; drop --still/--still-gpu";
    return false;
  }
  if (options.video_gpu) {
    error = "--fast-decode is CPU-only: the CUDA backend keeps the historical "
            "decisions so its streams stay CPU-reference exact (drop "
            "--video-gpu)";
    return false;
  }
  options.video = true;
  if (options.video_rd_preset.empty()) {
    options.video_rd_preset = "quality";
  } else if (options.video_rd_preset != "quality" &&
             options.video_rd_preset != "best") {
    error = "--fast-decode wraps the measured 'quality' preset; drop the other "
            "--video-rd-preset value";
    return false;
  }
  if (!apply_video_rd_preset(options, error)) return false;
  // Explicit ablation switches (docs/reference-structure.md).  They only
  // suppress the matching element below and are off by default, so every
  // pinned nightly stream is byte-identical.
  if (options.video_deblock_off) options.video_deblock = false;
  // Only the decisions that measured positive in combination on the CPU
  // benchmark.  The TX context rate stays a separate opt-in: it helps the old
  // 64x64 tree but measured negative when stacked on the bottom-edge path (see
  // docs/video-nightly.md).  The measured partition probe replaces the
  // variance tree with a per-leaf NEWMV + quantised-residual RD cost; its
  // quarter-scale lambda and first-level bound are the measured combination.
  options.video_ccso_rd = true;
  // The compound/ref-mvs arms cannot carry the bottom-edge inter path, the
  // compound arms cannot carry BAWP, and the reference-management tools
  // reject the carried-CDF chain; those entries below read the same switches.
  options.video_edge_inter = !options.video_edge_inter_off;
  options.video_partition_rd = true;
  if (!flag_given(options, "--video-partition-rd-lambda")) {
    options.video_partition_lambda = 2.75e-6;
  }
  if (!flag_given(options, "--video-partition-rd-depth")) {
    options.video_partition_depth = 1;
  }
  // Reference-intro allocation: the key frame spends -25 QP (better key and
  // matching first P reference, so the P chain starts from a cleaner picture);
  // a duplicated opening pair keeps one QP.  The adaptive-CDF rate saving below
  // is what pays for the extra key bits, so the two move together.  Measured
  // at the 240p30 / 100 kbps point; see docs/video-nightly.md.  A
  // caller-supplied offset wins, and the rate-controlled paths (--rc crf/vbr)
  // own base_q_idx, so they are skipped.
  if (!options.video_key_qp_offset_given && options.rc_mode == "cqp" &&
      !options.video_key_qp_offset_off) {
    options.video_key_qp_offset = -27;
    options.video_key_qp_offset_given = true;
  }
  // Luma inter coefficient RDOQ at its measured lambda calibration.
  options.video_rdoq = true;
  // Keyframe intra luma TX partitioning with per-part prediction/recon.
  options.video_key_intra_txpart = true;
  // Adaptive coefficient CDFs and the drift-triggered refresh.  The CDF chain
  // is a pure entropy win at fixed decisions (-13.9% rate), and its saving is
  // what the -25 key offset spends; the refresh forces the inter residual on
  // the frames whose reference has drifted more than 2.5 dB below the key.
  // At QP110 the combination is 24,904 B / 99.616 kbps / 9.7069 JOD /
  // 43.8667 dB (8, 6.9 fps).  The rate-controlled paths keep the historical
  // decisions because they own base_q_idx.
  if (options.rc_mode == "cqp" && !options.video_cdf_chain_off) {
    options.avg_cdf = true;
    options.adaptive_cdf = true;
    // --cdf-continuity carries the §7.4.4/§7.5 image instead of re-deriving
    // the §9.3 one per frame; it is exact now and 2.77% smaller, so the
    // aggregate uses the carry and does not imply the reset-per-frame opt-in.
    // An explicit --video-cdf-per-frame-qp keeps the reset-per-frame image,
    // which is the A/B control the carry gate pins.
    const bool user_reset = flag_given(options, "--video-cdf-per-frame-qp");
    options.cdf_continuity = !user_reset;
    options.video_cdf_per_frame_qp = user_reset;
    options.video_drift_refresh = !options.video_drift_refresh_off;
  }
  // Measured with the final aggregate: 1/8-pel ME refinement and the
  // RDOQ-interaction TX-search lambda.  At QP117 this is 24,939 B /
  // 99.756 kbps / 9.5868 JOD / 41.6953 dB.
  options.video_subpel_eighth = true;
  // The TX lambda is QP-scaled when --video-qp-lambda is on; the historical
  // 136 stays the value for the constant-lambda path.
  if (!options.video_qp_lambda && !flag_given(options, "--video-tx-lambda")) {
    options.video_tx_lambda = 136.0;
  }
  // The bottom-edge residual mask: hidden decoder-invisible rows get no
  // residual and the bottom-leaf decisions score only the visible rows.
  options.video_edge_mask = !options.video_edge_mask_off;
  // §7.18 CDEF on every reconstructed frame.  Re-measured on top of the edge +
  // rate-aware-CCSO path: it is the largest algorithmic gain left and, unlike
  // the old --video-cdef --video-gdf pair, affordable (docs/video-nightly.md).
  options.video_cdef = true;
  // The §5.18.7.10 params row is re-signalled on every CDEF-enabled frame, so a
  // frame whose filter changes no sample still pays for it.  Price that row in
  // the decision: such frames code the one-bit `cdef_frame_enable = 0` form
  // (identical reconstruction, ≈2 B/frame smaller header).  See
  // docs/video-nightly.md.
  options.video_cdef_rd = true;
  // QP-scaled RD lambdas: five sites used constants where the tradeoff scales
  // with QP.  At 240p this is +0.7 dB PSNR / +1.1 K at iso-rate (CVVDP alone
  // called it ripple; Butteraugli -11% and SSIMULACRA2 +1.6 show the win);
  // at 720p it is +0.046 JOD.  See docs/qp-scaled-lambda.md.
  options.video_qp_lambda = true;

  // Overhead wave (docs/video-nightly.md): all four leave the decoded picture
  // byte-identical at the same QP except the CCSO row pricing, which trades
  // ~2.4% of the rate for a measured +0.015 JOD at iso-rate.
  options.video_tile_flush_min = !options.video_tile_flush_off;
  if (!options.video_ref_signal_off) options.video_ref_signal_compact = true;
  if (!options.video_tile_group_off) options.video_tile_group_compact = true;
  if (!options.video_ccso_header_off) options.video_ccso_header_rd = true;
  // Rate-aware CCSO is tuned on the nightly measurement point: 64 bands, a
  // table refitted once on the units that survive pruning, and a unit-symbol
  // price above one bit because enabling a unit also changes the coded
  // residual.  An explicit CLI value wins, so the knobs stay composable; the
  // documented expansion types the resolved values out.
  if (!flag_given(options, "--video-ccso-bands")) {
    options.video_ccso_bands = 6;
  }
  if (!flag_given(options, "--video-ccso-lambda")) {
    options.video_ccso_lambda = 22.0;
  }
  if (!flag_given(options, "--video-ccso-refit")) {
    options.video_ccso_refit_passes = 2;
  }
  if (!flag_given(options, "--video-ccso-unit-bits")) {
    options.video_ccso_unit_bits = 2.0;
  }
  // §7.13.3.25 fitted BAWP over the inter-residual path, with the CPU carrier
  // that re-expresses a GLOBALMV winner so use_bawp can be coded.  Measured
  // positive at iso-rate on the 240p fade clip; see docs/video-nightly.md.
  options.video_bawp = !options.video_bawp_off;
  options.video_bawp_carrier = !options.video_bawp_off;
  // One calibrated leaf cost (--video-leaf-rd) for the inter-candidate search,
  // the skip/residual decision and the intra/inter leaf decision.  Re-measured
  // on top of the aggregate above: +0.054 JOD / +0.21 dB at the 100 kbps point
  // and ≈-8% BD-rate; see docs/video-nightly.md.
  options.video_leaf_rd = true;
  // Ablation switches that must win over the assignments above: RDOQ, the
  // partition probe, TX search and the loop-filter stack are all rejected by
  // the frame-type wrappers, and the compound arms drop the CDF chain in the
  // earlier block.  Applied last so an unset switch changes nothing.
  if (options.video_rdoq_off) options.video_rdoq = false;
  if (options.video_partition_rd_off) options.video_partition_rd = false;
  if (options.video_tx_search_off) {
    options.video_tx_search = false;
    options.video_key_intra_txpart = false;
  }
  if (options.video_loop_filters_off) {
    options.video_deblock = false;
    options.video_cdef = false;
    options.video_cdef_rd = false;
    options.video_ccso = false;
    options.video_ccso_rd = false;
    options.video_ccso_header_rd = false;
  }
  // The §8.3.2 context rate is a separate opt-in: it helps the old tree but
  // measured worse when stacked on the full nightly aggregate.
  return true;
}

// --nightly-b: the measured chained single-key hierarchical-B aggregate.
// Expands the reordering-legal tool set plus the ports measured on the chain
// (edge path, key-QP offset, BAWP/carrier, partition-RD, anchor residual).
// The caller still supplies --video-gops and the frame count.  Research
// opt-in: decoder-exact under tools/validate_avmdec.py while the pinned dav2d
// has the §7.21.6 drain defect.  It is NOT the default and NOT implied by
// --fast-decode.
bool apply_video_nightly_b(Options& options, std::string& error) {
  if (!options.nightly_b) return true;
  if (options.m1 || options.still_gpu) {
    error = "--nightly-b is a CPU --video mode; drop --still/--still-gpu";
    return false;
  }
  if (options.video_gpu) {
    error = "--nightly-b is CPU-only: the CUDA backend keeps the historical "
            "decisions (drop --video-gpu)";
    return false;
  }
  if (options.fast_decode) {
    error = "--nightly-b and --fast-decode are separate measured aggregates; "
            "pick one";
    return false;
  }
  if (options.rc_mode != "cqp") {
    error = "--nightly-b layers on the fixed --qp stream; use --rc cqp";
    return false;
  }
  // --video-gops M > 1 is the chained form.  A single-picture stream is legal
  // without it (--frames 1): the schedule reduces to a lone key, which is what
  // a one-picture stream is anyway.  Requiring a chain for a still forced the
  // caller to pad it with 7 duplicate frames.
  if (options.video_gops < 1) {
    error = "--nightly-b needs --video-gops M >= 1 (M 8-frame B blocks); "
            "omit it only for a single picture (--frames 1)";
    return false;
  }
  options.video = true;
  if (options.video_rd_preset.empty()) {
    options.video_rd_preset = "quality";
  } else if (options.video_rd_preset != "quality") {
    error = "--nightly-b wraps the measured 'quality' preset; drop the other "
            "--video-rd-preset value";
    return false;
  }
  if (!apply_video_rd_preset(options, error)) return false;

  // Sequential key-frame RD driver: the one key is measured in the
  // reconstruction its layout builds before the block anchors predict from it.
  options.video_rd_driver = true;
  // Exact-rate / BA-shaped pipe and the tail exact-rate TX window: the
  // measured head/tail delta on the B carrier is the tail alone.
  options.video_eff_perc = true;
  options.video_tail_tx = true;

  // The chained 8-frame dyadic B block with one key.
  options.video_tlayers = 8;
  options.video_tlayers_chain = true;
  options.video_tlayers_b = true;

  // The measured chained-form ports.  These are measured ON A CHAIN and the
  // validation rejects them without one, so a single-picture stream
  // (--frames 1, no --video-gops) keeps them off; nothing here can act across
  // time on a lone key.
  // The bottom-edge path is not only an inter tool: it reconstructs and pads
  // the hidden rows of the last superblock row, which a lone key also straddles
  // (docs/negative-results.md, "Single-picture encodes miss the bottom-edge
  // path": single 27.30 dB vs chain key 35.39 dB at equal bytes/qindex, with
  // rows 0..180 identical and rows 180..240 3.11x/4.55x worse without it).
  // It was gated on video_gops > 1 while the preset sets tlayers = 8 even for a
  // lone key, so --frames 1 was both *not given* it and *rejected* if asked.
  // Now always set, mirroring apply_video_nightly; the _off switches opt out.
  options.video_edge_inter = !options.video_edge_inter_off;
  options.video_edge_mask = !options.video_edge_mask_off;
  // --video-edge-mask is only legal with --video-edge-inter, so a caller who
  // turns the inter half off means "the whole edge path off"; without this the
  // lone switch would be rejected by validation instead of disabling it.
  if (options.video_edge_inter_off) options.video_edge_mask = false;
  // A single picture has no inter chain, so the key-offset allocation (a
  // reference-intro spend) and BAWP (an inter tool) do not apply and are
  // rejected by validation without a chain.
  const bool single_picture = options.video_gops <= 1;
  if (!single_picture && !options.video_key_qp_offset_given &&
      !options.video_key_qp_offset_off) {
    // -37 is the re-swept optimum on the final stack (compound + TIP + CWP
    // depth + exact compound rate + joint plan). The -50 value was tuned
    // before those landed; at matched 100 kbps -37 measures K 88.207 against
    // 87.513 (+0.694) with both arms avmdec 60/60 recon-exact. See
    // docs/rd-alloc-post-joint.md.
    options.video_key_qp_offset = -37;
    options.video_key_qp_offset_given = true;
  }
  options.video_bawp = !single_picture && !options.video_bawp_off;
  options.video_bawp_carrier = !single_picture && !options.video_bawp_off;
  // AVERAGE compound on the interior B pictures: the plan's (0,1) past/future
  // pair is a true two-reference pair, and §5.20.7.6 codes use_bawp only on
  // single-prediction leaves, so the BAWP carrier above coexists per leaf.
  // The depth arm adds the first NEAR_NEARMV stack pair per list with the
  // five CWP weights; it is +0.218 over TIP (86.805 -> 87.023).  The implicit
  // mask is NOT in the measured arm: it selects no different block but does
  // change the reconstructed blend, so it re-keys the 16/60-picture pins at
  // the same byte lengths and was excluded from the 87.023 measurement
  // (docs/compound-depth.md).  Wedge/diffwtd, OPFL and refine-MV stay off.
  // --video-compound-off drops compound and the depth arm; the TIP-only
  // control is --video-compound-depth-off.  Explicit depth flags win.
  // Compound and the depth arm need two references (past + future), so a lone
  // key keeps both off.
  options.video_compound = !single_picture && !options.video_compound_off;
  const bool compound_depth =
      options.video_compound && !options.video_compound_depth_off;
  if (!flag_given(options, "--video-compound-near-near")) {
    options.video_compound_near_near = compound_depth;
  }
  if (!flag_given(options, "--video-cwp-weighted")) {
    options.video_cwp_weighted = compound_depth;
  }
  // The partition probe is an inter-tree tool and is rejected without a chain.
  options.video_partition_rd =
      !single_picture && !options.video_partition_rd_off;
  // Nightly base lambda; with --video-qp-lambda the probe uses the measured
  // QP fit (the explicit flag would pin the constant instead).
  if (!flag_given(options, "--video-partition-rd-lambda")) {
    options.video_partition_lambda = 2.75e-6;
  }
  if (!flag_given(options, "--video-partition-rd-depth")) {
    options.video_partition_depth = 1;
  }
  // Drift refresh: force the inter residual on the chained layer-0 anchors.
  // No anchors exist in a single-picture stream, so keep it off there.
  options.video_tlayer_anchor_residual = !single_picture;

  // B_LEGAL: the reordering-legal measured tool set (docs/bframes.md).
  options.video_ccso_rd = true;
  options.video_ccso_bands = 6;
  options.video_ccso_lambda = 22.0;
  options.video_ccso_refit_passes = 2;
  options.video_ccso_unit_bits = 2.0;
  options.video_rdoq = true;
  // Ablation switch, applied immediately so an explicit --video-rdoq-off can
  // reach the RDOQ/TCQ exclusion (the two own the same levels).  This mirrors
  // apply_video_nightly's last-wins handling of the same flag; without it the
  // chained preset has no way to price the TCQ arm at all.
  if (options.video_rdoq_off) options.video_rdoq = false;
  options.video_cdef = true;
  options.video_cdef_rd = true;
  options.video_leaf_rd = true;
  options.video_subpel_eighth = true;
  options.video_key_intra_txpart = true;
  options.video_qp_lambda = true;
  options.avg_cdf = true;
  options.adaptive_cdf = true;
  options.cdf_continuity = true;
  options.video_tile_flush_min = true;
  options.video_ccso_header_rd = true;

  // The measured env-only research gates.  They are read lazily on the first
  // decision, after this expansion.  An explicit env value wins, including "0"
  // for a caller ablation, so the aggregate stays inspectable.
  const auto set_gate = [](const char* name, const char* value) {
    if (std::getenv(name) == nullptr) {
      setenv(name, value, 1);
    }
  };
  set_gate("AV2ENC_RD_KEY_TX", "1");
  set_gate("AV2ENC_RD_UVFULL", "1");
  set_gate("AV2ENC_RD_UVLEVEL", "1");
  set_gate("AV2ENC_TX_DETAIL_SEARCH_TAIL", "1");
  set_gate("AV2ENC_TX_DETAIL_MIN", "0.50");
  // TIP output on the chained-B dead ends (docs/tlayers-chain.md): the leaves
  // no planned frame references become TIP_FRAME_AS_OUTPUT pictures.  Mode 1
  // is the measured arm, and the coded-leaf probe keeps TIP only when its luma
  // PSNR is within 0.30 dB of the tile-coded alternative.  Field guard 3
  // falls back to the coded B when a reference §7.22 field carries a CWP
  // compound block or a compound cell with a non-zero MV (the unguarded
  // composition is not decoder-exact).  Gated on compound:
  // --video-compound-off is therefore the exact 85.472 control.  An explicit
  // env value (including 0) wins.
  if (options.video_compound) {
    set_gate("AV2ENC_TIP_LEAF", "1");
    set_gate("AV2ENC_TIP_LEAF_PSNR_MARGIN", "0.30");
    if (compound_depth) {
      set_gate("AV2ENC_TIP_LEAF_FIELD_GUARD", "3");
      set_gate("AV2ENC_COMPOUND_RATE", "1");
      set_gate("AV2ENC_COMPOUND_PAIR_LIMIT", "1");
      // Exact compound symbol pricing in the eff-perc leaf RD: comp_mode /
      // is_joint / compound_mode / jmvd_scale_mode / both DRLs /
      // comp_group_idx / cwp_idx / the compound filter.  Measured +0.261 over
      // the heuristic price (87.023 -> 87.284).  The compound NEWMV modes are
      // NOT enabled (AV2ENC_COMPOUND_NEWMV is left unset; they measure +0.054,
      // below the gate).
      set_gate("AV2ENC_COMPOUND_EXACT_RATE", "1");
      // The Phase-2 joint plan solve on the chained-B carrier: the same
      // one-objective tree search with the compound leaf priced exactly.
      // +0.286 K bracket / +0.278 fit at 100 kbps; the serial raster walk is
      // ~12 min per 60-frame 240p encode, so it is a research gate, not the
      // fast path.  An explicit env value (including 0) wins and restores
      // the 87.284 exact-rate arm.
      set_gate("AV2ENC_RD_JOINT_P", "1");
      set_gate("AV2ENC_RD_JOINT_P_TARGET", "8");
    }
  }

  // Symmetric §5.4.8 chroma quantizer offset.  Measured on the 720p waterfall
  // still (docs/negative-results.md): the frame's chroma error is concentrated
  // in dark saturated content, where a single coarse chroma step over a large
  // DC offset reads as a visible hue shift.
  //
  // Judge this by chroma fidelity, not by the PSNR-like aggregates: at the
  // same luma QP, coding both planes 8 qindex steps finer costs ~2.1 % of the
  // payload and cuts the visible hue error -- the dark foliage above the falls
  // goes 8.68 -> 7.82 LSB rms and the worst pixel 35.0 -> 27.0, and the
  // moss/rock band between the falls and the leaves goes 4.40 -> 4.04 rms with
  // the peak 26.2 -> 24.5.  SSIMULACRA2 and butteraugli are reported in the
  // docs for continuity but are not the reason for this default: they weight
  // chroma like PSNR and disagree with each other here.
  //
  // An explicit --video-uv-qp-offset N always wins, including 0 (the opt-out
  // that restores the historical single-step stream).
  if (!options.video_uv_qp_offset_given && !options.video_uv_delta_q) {
    options.video_uv_qp_offset = -8;
  }
  // Ablation switch, applied last so it wins over the assignments above (the
  // RD preset sets video_deblock/video_ccso and this function sets cdef/ccso_rd
  // earlier).  Mirrors apply_video_nightly's --video-loop-filters-off list
  // exactly, so the two aggregates treat the flag the same way; without it the
  // flag was silently ignored under --nightly-b because only the other
  // expansion ever read it.
  if (options.video_loop_filters_off) {
    options.video_deblock = false;
    options.video_cdef = false;
    options.video_cdef_rd = false;
    options.video_ccso = false;
    options.video_ccso_rd = false;
    options.video_ccso_header_rd = false;
  }
  return true;
}
