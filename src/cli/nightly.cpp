// --video-rd-preset and --fast-decode expansions.  Both only set existing CLI
// options; moved verbatim from src/main.cpp so the expanded flag set (and
// therefore the stream) is unchanged.
#include "cli/options.h"

#include <string>

// Expansion of the measured video RD presets (--video-rd-preset <name>).  The
// preset only sets existing CLI options; the caller then runs the normal
// combination validation, so a preset cannot reach a tool pairing that the
// individual flags reject.  No preset leaves every field untouched, so the
// historical default stream is byte-identical.
//
// The 'quality' row is measured on the CPU --video reference at the 240p30 /
// 100 kbps point (see docs/video-rd-preset.md): a single tile, P-only chain.
bool apply_video_rd_preset(Options& options, std::string& error) {
  if (options.video_rd_preset.empty()) return true;
  if (options.video_rd_preset != "quality") {
    error = "--video-rd-preset expects 'quality' (the only measured row)";
    return false;
  }
  if (!options.video) {
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
  } else if (options.video_rd_preset != "quality") {
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

// --nightly-b: the chained single-key B structure (docs/tlayers-chain.md).
// Research opt-in: it is decoder-exact under tools/validate_avmdec.py while
// the pinned dav2d has the §7.21.6 drain defect, and it measured -42.9 K
// against the flat aggregate at 100 kbps because every reordering schedule
// rejects --video-edge-inter.  It is NOT the default and NOT implied by
// --fast-decode.  It sets the tlayers/B chain gates and validates as a
// single-tile, single-key stream; the caller passes --video-gops.
bool apply_video_nightly_b(Options& options, std::string& error) {
  if (!options.nightly_b) return true;
  if (options.m1 || options.still_gpu) {
    error = "--nightly-b is a CPU --video mode; drop --still/--still-gpu";
    return false;
  }
  if (options.video_gpu) {
    error = "--nightly-b is CPU-only: drop --video-gpu";
    return false;
  }
  options.video = true;
  if (options.video_tlayers == 0) {
    error = "--nightly-b codes the chained B structure: pass --video-tlayers 8 "
            "(and --video-gops M for M 8-frame blocks)";
    return false;
  }
  return true;
}
