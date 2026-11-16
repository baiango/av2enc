// Combination and per-flag validation.  Moved verbatim from src/main.cpp: the
// order of the checks decides which error is reported first, so the blocks are
// neither reordered nor merged.  apply_effort_ranking() runs before them, as
// before, so a ranked flag obeys the same rejection rules as a typed one.
#include "cli/gpu_callable.h"
#include "cli/options.h"

#include "pipeline/planner/effort.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// Expansion of the measured feature ranking into existing CLI options.  Called
// after option parsing and before validation, so the selected flags obey the
// same rejection rules as flags typed by the user.  Only --still (the measured
// CPU reference path) is calibrated; no GPU/video row exists yet.
bool apply_effort_ranking(Options& options, std::string& error) {
  const auto reject = [&](std::string message) {
    error = std::move(message);
    return false;
  };
  if (options.video || options.lossless || options.still_gpu ||
      options.gpu_plan != 0) {
    return reject("--effort/--cpu-used is calibrated for --still only; there "
                  "is no measured video, lossless, GPU or gpu-plan row yet");
  }
  if (options.tiles != "1x1") {
    return reject("--effort/--cpu-used cannot be combined with --tiles: the "
                  "tiled writer pins its tool set");
  }
  if (options.bit_depth != 8 || options.chroma != "420") {
    return reject("--effort/--cpu-used is calibrated on 8-bit 4:2:0 stills; "
                  "the --bit-depth/--chroma format paths pin their tool set");
  }

  const int budget_sources = (options.effort_given ? 1 : 0) +
                             (options.cpu_used_given ? 1 : 0);
  if (budget_sources == 0) {
    return reject("--effort-plan requires --effort or --cpu-used");
  }
  if (budget_sources > 1) {
    return reject("pass only one of --effort or --cpu-used");
  }

  double budget = 0.0;
  std::string source;
  if (options.effort_given) {
    budget = static_cast<double>(options.effort_budget);
    source = "--effort " + std::to_string(options.effort_budget);
  } else {
    budget = av2enc::effort::budget_for_cpu_used(options.cpu_used);
    source = "--cpu-used " + std::to_string(options.cpu_used);
  }

  av2enc::effort::Constraints constraints;
  constraints.cfl = options.still_cfl;
  constraints.sb128 = options.still_sb128;
  constraints.sb256 = options.still_sb256;
  constraints.qm = options.still_qm;
  constraints.dc_ac_delta_q = options.still_dc_ac_delta_q;
  constraints.still_single = options.still_single;
  constraints.mrl = options.still_mrl;
  constraints.wiener = options.still_wiener;
  constraints.chroma_lr = options.still_chroma_lr;
  constraints.pc_wiener = options.still_pc_wiener;
  constraints.gdf = options.still_gdf;

  av2enc::effort::Selection selection;
  if (!av2enc::effort::select(budget, av2enc::effort::Mode::kStill,
                              options.specified, constraints, selection,
                              error)) {
    return false;
  }
  selection.budget_source = source;

  struct FlagMap {
    std::string_view flag;
    bool Options::*member;
  };
  static constexpr FlagMap kFlagMap[] = {
      {"--still-tx-type", &Options::still_tx_type},
      {"--still-ist", &Options::still_ist},
      {"--still-dip", &Options::still_dip},
      {"--still-rdoq", &Options::still_rdoq},
      {"--still-uv-full", &Options::still_uv_full},
      {"--still-cdef", &Options::still_cdef},
      {"--still-ccso", &Options::still_ccso},
      {"--still-mhccp", &Options::still_mhccp},
      {"--still-ext-partitions", &Options::still_ext_partitions},
      {"--still-ibp", &Options::still_ibp},
      {"--still-pc-wiener", &Options::still_pc_wiener},
      {"--still-rdo-intra", &Options::still_rdo_intra},
      {"--still-full-intra", &Options::still_full_intra},
      {"--still-mrl", &Options::still_mrl},
  };
  for (const std::string& flag : selection.flags) {
    bool found = false;
    for (const FlagMap& map : kFlagMap) {
      if (flag == map.flag) {
        options.*(map.member) = true;
        found = true;
        break;
      }
    }
    if (!found) {
      return reject("effort table selected '" + flag +
                    "', which this CLI does not implement");
    }
  }
  options.m1 = true;  // the measured ranking is a Still-path preset
  options.effort_active = true;
  options.effort_report = av2enc::effort::format_selection(selection);
  return true;
}
}  // namespace

ParseStatus validate_options(Options& options, std::string& error) {
  if (options.effort_given || options.cpu_used_given || options.effort_plan) {
    if (!apply_effort_ranking(options, error)) {
      return ParseStatus::kError;
    }
  }

  if (options.nightly) {
    if (!apply_video_nightly(options, error)) {
      return ParseStatus::kError;
    }
  } else if (!apply_video_rd_preset(options, error)) {
    return ParseStatus::kError;
  }

  if (options.input.empty() && !options.effort_plan) {
    error = "no input file specified";
    return ParseStatus::kError;
  }
  if (options.output.empty() && options.gpu_plan == 0 &&
      !options.effort_plan) {
    error = "no output file specified (use -o <output.ivf>)";
    return ParseStatus::kError;
  }
  const auto reject = [&](std::string message) {
    error = std::move(message);
    return ParseStatus::kError;
  };

  if (options.video_edge_inter) {
    if (!options.video) {
      error = "--video-edge-inter applies to the CPU --video encoder; pass "
              "--video";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-edge-inter is CPU-only: the CUDA M2 backend has no "
              "padded bottom-edge reference path";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--video-edge-inter needs the single-tile writer; the tiled "
              "path keeps its tile-local reconstruction";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0 || options.video_tlayers != 0 ||
        options.video_long_term || options.video_olk ||
        options.video_leading != 0 || options.video_sef ||
        options.video_switch || options.video_ras || options.video_tip) {
      error = "--video-edge-inter is measured on the low-delay P chain; drop "
              "the B-frame / reorder / temporal-layer mode";
      return ParseStatus::kError;
    }
    if (options.video_ref_mvs) {
      error = "--video-edge-inter cannot carry the temporal MV field: the "
              "bottom SVB rows have no coded MI grid";
      return ParseStatus::kError;
    }
  }

  // --video-qp-lambda: CPU reference decisions only.
  if (options.video_qp_lambda) {
    // An explicit numeric lambda still wins over the scaled value for that
    // site (the documented escape hatch).  The nightly expansion therefore
    // omits --video-tx-lambda and --video-partition-rd-lambda: listing them
    // would pin those two sites by hand and quietly change the stream.
    if (!options.video || options.m1 || options.still_gpu) {
      error = "--video-qp-lambda is a --video feature: pass --video";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-qp-lambda is CPU-only: the CUDA M2 backend keeps the "
              "historical RDO decisions as its parity reference";
      return ParseStatus::kError;
    }
    if (options.rc_mode != "cqp") {
      error = "--video-qp-lambda needs a fixed --qp stream: the "
              "rate-controlled paths own base_q_idx and would re-scale every "
              "frame";
      return ParseStatus::kError;
    }
  }

  // --video-tile-flush-min block: the CPU video tile writer only.
  if (options.video_tile_flush_min) {
    if (!options.video || options.m1 || options.still_gpu) {
      error = "--video-tile-flush-min is a --video feature: pass --video";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-tile-flush-min is CPU-only: the CUDA M2 backend keeps "
              "the historical tile termination as its parity reference";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--video-tile-flush-min needs the single-tile writer";
      return ParseStatus::kError;
    }
    if (options.lossless) {
      error = "--video-tile-flush-min applies to the lossy video writer";
      return ParseStatus::kError;
    }
  }
  // --video-tile-flush-min block end.

  if (options.video_key_qp_offset_given) {
    if (!options.video || options.m1 || options.still_gpu) {
      error = "--video-key-qp-offset applies to the CPU --video encoder; "
              "pass --video";
      return ParseStatus::kError;
    }
    if (options.rc_mode != "cqp") {
      error = "--video-key-qp-offset layers on the fixed --qp stream; use "
              "--rc cqp (the --rc crf/vbr planners own base_q_idx)";
      return ParseStatus::kError;
    }
    if (options.svc_given) {
      error = "--video-key-qp-offset is wired for the plain --video chain, "
              "not the SVC layers";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0 || options.video_tlayers != 0 ||
        options.video_tip || options.video_long_term || options.video_olk ||
        options.video_leading != 0 || options.video_sef ||
        options.video_switch || options.video_ras) {
      error = "--video-key-qp-offset is wired for the flat key+P chain; drop "
              "the B-frame / reorder / temporal-layer / frame-type wrapper "
              "mode";
      return ParseStatus::kError;
    }
    if (options.avg_cdf || options.adaptive_cdf || options.cdf_continuity) {
      // --cdf-continuity does not need the reset: the §7.4.4/§7.5 image
      // follows the symbols actually coded, and only a frame the decoder
      // resets (§5.18.2) derives its §9.3 defaults from its own base_q_idx.
      if (!options.video_cdf_per_frame_qp && !options.cdf_continuity) {
        error = "--video-key-qp-offset needs a fixed base_q_idx; --avg-cdf/"
                "--adaptive-cdf are not compatible with per-frame QP (pass "
                "--video-cdf-per-frame-qp to reset the CDF image per frame, or "
                "--cdf-continuity to carry it)";
        return ParseStatus::kError;
      }
    }
  }

  // --video-drift-refresh: implementation, then docs.  Delimited block so it
  // can be merged without collisions.  The mechanism itself lives in
  // src/pipeline/video/video_frame.cpp (the measured reference-error trigger).
  if (options.video_drift_refresh) {
    if (!options.video || options.m1 || options.still_gpu) {
      error = "--video-drift-refresh applies to the CPU --video encoder; "
              "pass --video";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-drift-refresh is CPU-only: the CUDA backend keeps the "
              "historical skip chain";
      return ParseStatus::kError;
    }
    if (options.rc_mode != "cqp") {
      error = "--video-drift-refresh layers a per-frame QP plan on the fixed "
              "--qp stream; use --rc cqp (the --rc crf/vbr planners own "
              "base_q_idx)";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--video-drift-refresh is wired for a single tile";
      return ParseStatus::kError;
    }
    if (options.svc_given) {
      error = "--video-drift-refresh is wired for the plain --video chain, "
              "not the SVC layers";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0 || options.video_tlayers != 0 ||
        options.video_tip || options.video_long_term || options.video_olk ||
        options.video_leading != 0 || options.video_sef ||
        options.video_switch || options.video_ras) {
      error = "--video-drift-refresh is wired for the flat key+P chain; drop "
              "the B-frame / reorder / temporal-layer / frame-type wrapper "
              "mode";
      return ParseStatus::kError;
    }
    if (options.bit_depth != 8 || options.chroma != "420") {
      error = "--video-drift-refresh is wired for 8-bit 4:2:0";
      return ParseStatus::kError;
    }
    // A run-time base_q_idx change would desynchronise the §9.3 image from
    // the static per-frame plan, so the CDF chain is only allowed while the
    // refresh keeps every frame's QP (--video-drift-refresh-qp 0).  The forced
    // residual is decision-level and does not touch the CDF seeding.
    if ((options.avg_cdf || options.adaptive_cdf || options.cdf_continuity) &&
        options.video_drift_refresh_qp != 0) {
      error = "--video-drift-refresh composes with the CDF chain only at "
              "--video-drift-refresh-qp 0: a run-time base_q_idx change would "
              "desynchronise the per-frame §9.3 CDF image";
      return ParseStatus::kError;
    }
    // --cdf-continuity composes with --video-drift-refresh: the carried image
    // follows the symbols actually coded, so a refresh frame's extra residual
    // needs no §9.3 re-derivation.  The per-frame-QP reset form above is the
    // one that depends on base_q_idx.
  }
  // --video-drift-refresh block end.

  if (options.video_edge_mask) {
    if (!options.video || options.m1 || options.still_gpu) {
      error = "--video-edge-mask applies to the CPU --video encoder; pass "
              "--video";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-edge-mask is CPU-only: the CUDA M2 backend keeps the "
              "historical bottom-edge residual";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--video-edge-mask needs the whole-frame single-tile writer";
      return ParseStatus::kError;
    }
    if (!options.video_edge_inter) {
      error = "--video-edge-mask needs --video-edge-inter: only that path "
              "reconstructs and reference-pads the hidden bottom rows";
      return ParseStatus::kError;
    }
  }

  if (options.cpu_threads_given) {
    if (!options.video) {
      error = "--cpu-threads applies to the CPU --video reference encoder";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--cpu-threads is a CPU reference knob; the GPU backend "
              "(--video-gpu) has its own scheduling";
      return ParseStatus::kError;
    }
  }

  if (options.gpu_plan != 0) {
    if (options.video || options.m1 || options.still_gpu || options.lossless) {
      error = "--gpu-plan is a host-planning dry run; do not combine it with "
              "--video, --still, --still-gpu or --lossless";
      return ParseStatus::kError;
    }
    // The plan reads the input header, the tile layout and the optional VRAM
    // cap. Every other flag would be accepted without an effect, so name it
    // instead of ignoring it.
    for (const std::string_view flag : options.specified) {
      if (flag == "--gpu-plan" || flag == "--tiles" || flag == "--vram-cap" ||
          flag == "-o" || flag == "--output") {
        continue;
      }
      return reject("--gpu-plan is a host-only dry run; " + std::string(flag) +
                    " has no effect on the plan");
    }
  }
  if (options.video_cdf_per_frame_qp) {
    if (!options.video || options.m1 || options.still_gpu ||
        options.lossless) {
      error = "--video-cdf-per-frame-qp applies to the CPU --video encoder; "
              "pass --video";
      return ParseStatus::kError;
    }
    if (!options.avg_cdf || !options.adaptive_cdf) {
      error = "--video-cdf-per-frame-qp needs --avg-cdf and --adaptive-cdf: "
              "it only relaxes the §8.2.2 CDF-image reset for a per-frame QP "
              "plan";
      return ParseStatus::kError;
    }
    if (options.cdf_continuity) {
      error = "--video-cdf-per-frame-qp re-initialises the CDF image from "
              "each frame's slice; it does not compose with --cdf-continuity";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--video-cdf-per-frame-qp is wired for a single tile";
      return ParseStatus::kError;
    }
  }

  if (options.video) {
    if (options.m1 || options.still_gpu || options.lossless) {
      error = "--video cannot be combined with --still, --still-gpu or "
              "--lossless";
      return ParseStatus::kError;
    }
    if (options.adaptive_cdf && !options.avg_cdf) {
      error = "--adaptive-cdf on the video path needs --avg-cdf: the "
              "average-CDF bank owns the §8.2.2 tile seed and §7.5 frame-end "
              "bookkeeping";
      return ParseStatus::kError;
    }
    if (options.cdf_continuity &&
        !(options.avg_cdf && options.adaptive_cdf)) {
      error = "--cdf-continuity requires --avg-cdf and --adaptive-cdf";
      return ParseStatus::kError;
    }
    // The encoder side is wired (seed export + §7.4.4 reference load + §7.5
    // frame end), but pinned dav2d f9dafbc rejects a frame whose
    // disable_cross_frame_cdf_init is 0: "Error parsing frame header / Error
    // parsing OBU data".  Reproducer (documented in docs/negative-results.md):
    //   av2enc in.y4m -o out.ivf --video --p-frames 2 --avg-cdf
    //       --adaptive-cdf --cdf-continuity
    // Set the env escape below only for encoder-side experiments; a conforming
    // decoder is required to gate it.
    if ((options.video_timing || options.video_mfh) &&
        (options.video_long_term || options.video_olk ||
         options.video_leading > 0 || options.video_sef ||
         options.video_switch || options.video_ras)) {
      error = "--video-timing/--video-mfh are wired into the plain --video "
              "sequence header only; the frame-type modes (long-term/OLK/"
              "leading/SEF/SWITCH/RAS) build their own headers";
      return ParseStatus::kError;
    }
    if ((options.bit_depth != 8 || options.chroma != "420") &&
        !video_format_wired(options.bit_depth, options.chroma)) {
      error = "--video supports 8-bit 4:2:0 and 10-bit 4:2:0; the remaining "
              "--bit-depth/--chroma combinations are not verified on the "
              "video path yet";
      return ParseStatus::kError;
    }
    // Rate-control combination errors are usage errors: name them here so the
    // caller exits 2 before the output file is opened (and therefore never
    // truncates an existing -o stream).
    if (options.rc_given && options.rc_mode == "vbr" &&
        !options.rc_bitrate_given) {
      error = "--rc vbr requires --bitrate <kbps>";
      return ParseStatus::kError;
    }
    if (options.rc_crf_given && options.rc_mode != "crf") {
      error = "--crf requires --rc crf";
      return ParseStatus::kError;
    }
    if (options.rc_bitrate_given && options.rc_mode != "vbr") {
      error = "--bitrate requires --rc vbr";
      return ParseStatus::kError;
    }
    if (options.rc_mode != "cqp" &&
        (options.video_olk || options.video_leading != 0 ||
         options.video_ras)) {
      error = "--rc " + options.rc_mode +
              " does not compose with --video-olk, --video-leading or "
              "--video-ras yet: those modes split the sequence into short "
              "sub-encodes whose per-frame QP count differs from the input";
      return ParseStatus::kError;
    }
    if (options.rc_mode != "cqp" && !options.video_cdf_per_frame_qp &&
        (options.avg_cdf || options.adaptive_cdf ||
         options.cdf_continuity)) {
      error = "--rc " + options.rc_mode +
              " needs a fixed base_q_idx; --avg-cdf/--adaptive-cdf/"
              "--cdf-continuity are not compatible with per-frame QP (pass "
              "--video-cdf-per-frame-qp to reset the CDF image per frame)";
      return ParseStatus::kError;
    }
    // CRF derives base_q_idx from --crf only (qcomp is 0, so the quantiser is
    // constant across frames).  An explicit --qp would be accepted and ignored,
    // so reject it by name; --qp remains the VBR statistics-pass anchor.
    if (options.rc_mode == "crf" && flag_given(options, "--qp")) {
      error = "--rc crf owns the quantizer; drop --qp or use --rc cqp/--rc "
              "vbr (where --qp is the statistics-pass anchor)";
      return ParseStatus::kError;
    }
    // --video-drl-reorder is decoder-gated: the sequence header codes
    // disable_drl_reorder = 0 / constrain_drl_reorder = 1.  The pinned dav2d
    // suppresses the §7.12.2.19 sort for a single reference within two POCs
    // (its high-priority-TMVP term is not gated by use_ref_frame_mvs), so the
    // CONSTRAINT mode is bit-exact but inert on the low-delay P surface;
    // --video-drl-reorder-always exercises the transform.  Both are pinned by
    // src/pipeline/video/gate_video_drl_reorder.sh.
    if (options.video_morph) {
      // §5.20.5.4/§7.13.3.26: the morph carrier signals the screen-content +
      // IBC mode-info path on the intra keyframe, so it needs a single-tile
      // keyframe, frozen per-tile CDFs and no loop filters (the IBC copy and
      // morph fit read the unfiltered reconstruction).  Every other composition
      // is rejected by name instead of being silently ignored.
      if (options.video_gpu) {
        error = "--video-morph is CPU-only: the CUDA M2 backend has no IBC "
                "mode-info path";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-morph is gated on the plain key+P chain: it cannot "
                "be combined with --b-frames, --video-tlayers, --video-tip or "
                "the frame-type wrapper modes";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-morph is gated on a single tile";
        return ParseStatus::kError;
      }
      if (options.video_deblock || options.video_cdef || options.video_ccso ||
          options.video_wiener || options.video_pc_wiener ||
          options.video_chroma_lr || options.video_gdf) {
        error = "--video-morph does not compose with the loop filters: the "
                "IBC copy/morph template must read the unfiltered "
                "reconstruction";
        return ParseStatus::kError;
      }
      if (options.avg_cdf || options.cdf_continuity) {
        error = "--video-morph uses the per-tile §9.3 IBC/morph CDFs; it is "
                "not wired with --avg-cdf/--cdf-continuity";
        return ParseStatus::kError;
      }
      if (options.video_drl_reorder_always) {
        error = "--video-morph codes use_intrabc and §7.12.2.19 "
                "DRL_REORDER_ALWAYS reorders the decoder's BV stack, which "
                "the IBC stack builder does not implement (CONSTRAINT is "
                "exact)";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_per_segment_q ||
          options.video_uv_delta_q) {
        error = "--video-morph is not wired with "
                "--video-segmentation/--video-per-segment-q/--video-uv-delta-q";
        return ParseStatus::kError;
      }
    }
    // §5.20.8 palette / §5.20.5.3-.4 general IBC on the intra keyframe.  Both
    // share the single-tile screen-content carrier and are rejected by name on
    // surfaces where their cache/ring or CDF state would not match the
    // decoder.
    if (options.video_palette || options.video_ibc) {
      const char* tool = options.video_palette ? "--video-palette"
                                               : "--video-ibc";
      if (options.video_gpu) {
        error = std::string(tool) +
                " is CPU-only: the CUDA M2 backend has no screen-content "
                "mode-info path";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = std::string(tool) +
                " is not applied by the --video-lossless writer";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = std::string(tool) + " is gated on a single tile";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = std::string(tool) +
                " is gated on the plain key+P chain: it cannot be combined "
                "with --b-frames, --video-tlayers, --video-tip or the "
                "frame-type wrapper modes";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = std::string(tool) +
                " is gated on 8-bit 4:2:0 (the format video core has no "
                "screen-content syntax)";
        return ParseStatus::kError;
      }
      if (options.avg_cdf || options.cdf_continuity) {
        error = std::string(tool) +
                " uses per-frame §9.3 palette/IBC CDFs; it is not wired with "
                "--avg-cdf/--cdf-continuity";
        return ParseStatus::kError;
      }
      if (options.video_palette && options.video_tx_search) {
        error = "--video-palette codes eob=0 palette leaves and is not wired "
                "with --video-tx-search (TX_MODE_SELECT needs a real luma TX "
                "partition per leaf)";
        return ParseStatus::kError;
      }
      if (options.video_ibc && options.video_drl_reorder_always) {
        error = "--video-ibc codes use_intrabc and §7.12.2.19 "
                "DRL_REORDER_ALWAYS also reorders the decoder's BV stack, "
                "which the IBC stack builder does not implement "
                "(--video-drl-reorder CONSTRAINT is exact)";
        return ParseStatus::kError;
      }
      if (options.video_ibc &&
          (options.video_segmentation || options.video_per_segment_q)) {
        error = "--video-ibc is not wired with "
                "--video-segmentation/--video-per-segment-q (the per-SB "
                "delta-Q state and the IBC skip block disagree)";
        return ParseStatus::kError;
      }
    }
    if (options.video_scc_force) {
      // §5.4.7 forced form: only meaningful on the screen-content carriers.
      // The per-tool gates above cover the tile count, frame-type chain,
      // format and CDF surface; here we only require a carrier and CPU.
      if (!(options.video_palette || options.video_ibc ||
            options.video_morph)) {
        error = "--video-scc-force needs one of --video-palette/--video-ibc/"
                "--video-morph: the forced sequence form only changes how "
                "those tools are signalled";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-scc-force is CPU-only: the CUDA M2 backend has no "
                "screen-content mode-info path";
        return ParseStatus::kError;
      }
    }
    if (options.video_force_integer_mv) {
      if (!options.video_scc_force) {
        error = "--video-force-integer-mv needs --video-scc-force";
        return ParseStatus::kError;
      }
      // force_integer_mv pins FrameMvPrecision to one pel, so a frame
      // precision flag or a per-block precision tool would be silently
      // ignored (or desynchronise the MVD units).
      if (options.video_mv_precision != 0) {
        error = "--video-force-integer-mv pins FrameMvPrecision to one pel; "
                "drop --video-mv-precision";
        return ParseStatus::kError;
      }
      if (options.video_flex_mvres || options.video_adaptive_mvd ||
          options.video_mvd_sign_derive) {
        error = "--video-force-integer-mv is not wired with "
                "--video-flex-mvres/--video-adaptive-mvd/"
                "--video-mvd-sign-derive (pb_mv_precision and the AMVD/sign "
                "units assume a signalled frame precision)";
        return ParseStatus::kError;
      }
      if (options.video_gme || options.video_warp) {
        error = "--video-force-integer-mv is not wired with "
                "--video-gme/--video-warp (the projected global MVs are not "
                "quantised to the one-pel frame grid)";
        return ParseStatus::kError;
      }
    }
    if (options.video_reduce_pb_aspect) {
      // §5.4.3: the tiled writer has its own minimal sequence header that does
      // not carry the partition config; keep the tiled path byte-identical.
      if (options.tiles != "1x1") {
        error = "--video-reduce-pb-aspect is gated on a single tile (the "
                "tiled sequence-header copy does not carry the partition "
                "config)";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-reduce-pb-aspect is gated on the plain key+P chain: "
                "the B/TIP/frame-type wrappers build their own sequence header "
                "and partition plan";
        return ParseStatus::kError;
      }
    }
    if (options.video_seq_seg_info) {
      if (!options.video_segmentation && !options.video_per_segment_q) {
        error = "--video-seq-seg-info needs --video-segmentation or "
                "--video-per-segment-q";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-seq-seg-info is gated on a single tile (the tiled "
                "sequence-header copy does not carry the segment table)";
        return ParseStatus::kError;
      }
      if (options.video_seq_seg_frozen &&
          (options.video_seg_skip || options.video_seg_globalmv)) {
        error = "--video-seq-seg-info-frozen is not wired with "
                "--video-seg-skip/--video-seg-globalmv: those features make "
                "the P-frame table differ from the frozen sequence table";
        return ParseStatus::kError;
      }
    }
    if (options.video_tile_group_compact) {
      // §5.4.2: only the single-tile plain chain knows the sequence grid up
      // front; the tiled and frame-type wrappers build their own sequence
      // headers and would ignore the flag.
      if (options.tiles != "1x1") {
        error = "--video-tile-group-compact is gated on a single tile (the "
                "tiled sequence-header copy carries no sequence tile_params)";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-tile-group-compact is gated on the plain key+P chain: "
                "the B/TIP/frame-type wrappers build their own sequence header";
        return ParseStatus::kError;
      }
    }
    if (options.video_bru) {
      // §7.12.2/§7.22 BRU needs enable_bru/use_bru and the BRU tile mode.
      // The component writer builds a valid stream, but the pinned decoder
      // rejects it at the frame header (repro:
      // src/pipeline/video/video_bru_reject_gate.sh with the pinned dav2d).
      error = "--video-bru is not decoder-gated: the pinned dav2d does not "
              "parse enable_bru/use_bru (Error parsing frame header).  The "
              "BRU component stream and the reproducer are in "
              "src/pipeline/video/video_bru_reject_gate.sh";
      return ParseStatus::kError;
    }
    if (options.video_ref_mvs) {
      if (!options.video) {
        error = "--video-ref-mvs requires --video";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-ref-mvs is gated on a single tile: the §7.22 field "
                "is built from one whole-frame MI grid";
        return ParseStatus::kError;
      }
      if (options.video_mv_precision != 0) {
        error = "--video-ref-mvs currently requires eighth-pel MVs";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-ref-mvs is gated on 8-bit 4:2:0";        return ParseStatus::kError;
      }
    }
    if (options.video_compound_near_near) {
      if (!options.video_compound) {
        error = "--video-compound-near-near needs --video-compound";
        return ParseStatus::kError;
      }
      if (options.b_frames == 0) {
        error = "--video-compound-near-near is gated on the two-reference "
                "B-frame path (one past and one future reference): pass "
                "--b-frames";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-compound-near-near is CPU-only: the CUDA compound "
                "backend implements the GLOBAL_GLOBALMV subset";        return ParseStatus::kError;
      }
    }
    if (options.video_cwp_weighted || options.video_imp_mask) {
      // Both derive from the NEAR_NEARMV compound carrier: §5.20.7.6 codes
      // cwp_idx only for a NEAR_NEARMV AVERAGE block, and the §7.13.3.14
      // implicit mask excludes GLOBAL_GLOBALMV.
      if (!options.video_compound || !options.video_compound_near_near) {
        error = "--video-cwp-weighted/--video-imp-mask need "
                "--video-compound --video-compound-near-near (the "
                "NEAR_NEARMV AVERAGE carrier)";
        return ParseStatus::kError;
      }
      if (options.video_opfl || options.video_refine_mv) {
        // §5.20.7.6 skips cwp_idx for use_optflow/use_refinemv, and the
        // refinement predictors hard-code CWP_EQUAL.  Keep the overlays apart.
        error = "--video-cwp-weighted/--video-imp-mask do not compose with "
                "--video-opfl/--video-refine-mv";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-cwp-weighted/--video-imp-mask are CPU-only";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-cwp-weighted/--video-imp-mask are gated on 8-bit "
                "4:2:0";
        return ParseStatus::kError;
      }
    }
    if (options.video_df_sub_pu) {
      // §5.18.2 allow_df_sub_pu exists only for an inter frame; every other
      // frame type leaves it inferred 0.  The TIP output branch codes its own
      // allow_df_sub_pu/apply_deblocking_filter_tip pair, so keep the carrier
      // on the ordinary P/B tile path.
      if (options.video_gpu) {
        error = "--video-df-sub-pu is CPU-only";
        return ParseStatus::kError;
      }
      if (options.video_tip || options.video_bru) {
        error = "--video-df-sub-pu does not compose with --video-tip/"
                "--video-bru: those TIP output frames code their own "
                "allow_df_sub_pu pair";
        return ParseStatus::kError;
      }
      if (options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras ||
          options.video_tlayers != 0) {
        error = "--video-df-sub-pu is gated on the plain key+P/B chain: the "
                "frame-type wrappers and --video-tlayers need their own "
                "allow_df_sub_pu handling";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        // deblocking_filter_params() returns before allow_df_sub_pu when
        // CodedLossless is set.
        error = "--video-df-sub-pu does not compose with --video-lossless";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-df-sub-pu is gated on 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-df-sub-pu is gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if ((options.video_opfl || options.video_refine_mv) &&
        !options.video_compound_near_near && !options.video_gpu) {
      // Both tools predict from a non-GLOBALMV compound block
      // (§5.20.7.6 use_optflow, §5.20.7.17 refine-MV), which only the
      // NEAR_NEARMV carrier codes.  --video-gpu reports its own by-name
      // rejection below instead.
      error = "--video-opfl/--video-refine-mv need --video-compound-near-near: "
              "the temporal refinement tools only run on a NEAR_NEARMV "
              "compound block";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0 && options.b_frames > 6) {
      error = "--b-frames supports 1..6 B frames per GOP";
      return ParseStatus::kError;
    }
    if (options.video_ref_signal_compact) {
      // §5.18.2 compact reference signalling rewrites the sequence header and
      // the per-frame reference fields, so it is gated on the stream shape it
      // was measured on.  The deeper combination checks live in
      // validate_video_options() (video_frame.cpp).
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-ref-signal-compact is gated on the 8-bit 4:2:0 core "
                "(the format video core keeps the historical reference "
                "fields)";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1) {
        error = "--video-ref-signal-compact needs --video-refs 1: the §7.7 "
                "derived map must hold exactly the one live slot";
        return ParseStatus::kError;
      }
    }
    if (options.video_implicit_ref_map) {
      // §5.4.6/§7.7 implicit reference maps are wired on the CPU 8-bit 4:2:0
      // flat-P, B-frame and hierarchical-P paths.  Every tool that owns an
      // explicit map (or a separate format core) is rejected by name.
      if (options.video_tip) {
        error = "--video-implicit-ref-map does not compose with --video-tip: "
                "TIP needs the explicit past/future anchor pair";
        return ParseStatus::kError;
      }
      if (options.video_compound || options.video_compound_wedge ||
          options.video_compound_diffwtd) {
        error = "--video-implicit-ref-map does not compose with "
                "--video-compound: reference_select and the compound ref pairs "
                "require an explicit two-reference map";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-implicit-ref-map does not compose with --tiles: the "
                "tiled frame header signals a single explicit reference slot";
        return ParseStatus::kError;
      }
      if (options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-implicit-ref-map does not compose with the "
                "long-term/OLK/leading/SEF/SWITCH/RAS frame-type wrappers";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-implicit-ref-map is wired for the 8-bit 4:2:0 video "
                "core only; the --bit-depth/--chroma format cores keep the "
                "explicit map";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = "--video-implicit-ref-map does not compose with --svc-layers";
        return ParseStatus::kError;
      }
    }
    if (options.video_num_ref_frames != 0 &&
        (options.b_frames != 0 || options.video_tlayers != 0 ||
         options.video_long_term || options.video_olk ||
         options.video_leading != 0 || options.video_sef ||
         options.video_switch || options.video_ras)) {
      error = "--video-num-ref-frames applies to the flat P chain only; the "
              "B-frame / temporal-layer / frame-type modes own NumRefFrames";
      return ParseStatus::kError;
    }
    if (options.video_num_ref_frames != 0 &&
        options.video_refs > options.video_num_ref_frames) {
      error = "--video-refs " + std::to_string(options.video_refs) +
              " exceeds --video-num-ref-frames " +
              std::to_string(options.video_num_ref_frames);
      return ParseStatus::kError;
    }
    if (options.video_olk && options.video_refs != 1) {
      error = "--video-olk needs --video-refs 1";
      return ParseStatus::kError;
    }
    if ((options.video_long_term || options.video_olk) &&
        options.video_mv_precision != 0) {
      error = "--video-long-term/--video-olk need eighth-pel MV precision";
      return ParseStatus::kError;
    }
    if (options.video_leading != 0) {
      if (options.video_long_term || options.video_olk) {
        error = "--video-leading cannot be combined with --video-long-term or "
                "--video-olk";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1 || options.video_mv_precision != 0) {
        error = "--video-leading needs --video-refs 1 and eighth-pel MV "
                "precision";
        return ParseStatus::kError;
      }
    }
    if (options.video_sef &&
        (options.video_long_term || options.video_olk ||
         options.video_leading != 0)) {
      error = "--video-sef cannot be combined with the long-term or leading "
              "frame-type modes";
      return ParseStatus::kError;
    }
    const int frame_modes =
        ((options.video_long_term || options.video_olk) ? 1 : 0) +
        (options.video_leading != 0 ? 1 : 0) + (options.video_sef ? 1 : 0) +
        (options.video_switch ? 1 : 0) + (options.video_ras ? 1 : 0);
    if (frame_modes > 1) {
      error = "select only one frame-type mode: --video-long-term/--video-olk, "
              "--video-leading, --video-sef, --video-switch or --video-ras";
      return ParseStatus::kError;
    }
    if (options.video_deblock && frame_modes != 0) {
      error = "--video-deblock does not compose with --video-long-term/--video-"
              "olk, --video-leading, --video-sef, --video-switch or "
              "--video-ras: those header writers do not signal "
              "deblocking_filter_params()";
      return ParseStatus::kError;
    }
    if (options.video_deblock && options.tiles != "1x1") {
      error = "--video-deblock does not support --tiles > 1x1: the pinned "
              "decoder filters across interior tile edges, so the CPU "
              "reference cannot match it";
      return ParseStatus::kError;
    }
    // §5.4.6 adaptive MVD / MVD sign derivation / flexible MV resolution.
    // The three tools share the §5.20.7.6 block path and are gated only on the
    // flat CPU key+P carrier; every other composition is rejected by name so
    // the sequence header never advertises a tool the tile writer cannot code
    // exactly.
    const bool mv_res_tools = options.video_adaptive_mvd ||
                              options.video_mvd_sign_derive ||
                              options.video_flex_mvres;
    if (mv_res_tools) {
      const char* tool = options.video_adaptive_mvd
                             ? "--video-adaptive-mvd"
                             : (options.video_mvd_sign_derive
                                    ? "--video-mvd-sign-derive"
                                    : "--video-flex-mvres");
      if (!options.video) {
        error = std::string(tool) + " requires --video";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = std::string(tool) +
                " is CPU-only: the CUDA M2 backend has no per-block MV "
                "precision/AMVD syntax path";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = std::string(tool) + " is gated on 8-bit 4:2:0 inter coding";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = std::string(tool) +
                " is gated on a single tile (the §8.3 MV-precision contexts "
                "are tile-local)";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = std::string(tool) +
                " is not wired into the --video-lossless writer";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip) {
        error = std::string(tool) +
                " is gated on the flat key+P chain (no --b-frames, "
                "--video-tlayers or --video-tip)";
        return ParseStatus::kError;
      }
      if (frame_modes != 0) {
        error = std::string(tool) +
                " does not compose with the frame-type modes "
                "(--video-long-term/--video-olk, --video-leading, --video-sef, "
                "--video-switch, --video-ras)";
        return ParseStatus::kError;
      }
      if (options.video_compound || options.video_compound_interintra ||
          options.video_bawp || options.video_gme || options.video_warp ||
          options.video_opfl || options.video_refine_mv) {
        // §5.20.7.6 places use_amvd relative to use_bawp/motion_mode/compound
        // modes this writer does not code, so the combined order is not gated.
        error = std::string(tool) +
                " does not compose with --video-compound, "
                "--video-compound-interintra, --video-bawp, --video-gme/"
                "--video-warp or --video-opfl/--video-refine-mv yet "
                "(§5.20.7.6 symbol order)";
        return ParseStatus::kError;
      }
      if (options.video_mvd_sign_derive && options.video_mv_precision == 0) {
        error = "--video-mvd-sign-derive needs --video-mv-precision quarter "
                "or half: §5.20.7.13 derives a sign only when "
                "FrameMvPrecision <= QUARTER and MvPrecision < QUARTER";
        return ParseStatus::kError;
      }
      if (options.video_mvd_sign_derive && options.video_mv_precision == 1 &&
          !options.video_flex_mvres) {
        error = "--video-mvd-sign-derive with quarter-pel frames needs "
                "--video-flex-mvres so a block can select half-pel "
                "(MvPrecision < QUARTER); a quarter-pel block derives no sign";
        return ParseStatus::kError;
      }
    }
    if ((options.video_wiener || options.video_chroma_lr) &&
        options.tiles != "1x1") {
      error = "--video-wiener/--video-chroma-lr do not support --tiles > "
              "1x1: the pinned decoder's loop-restoration unit and window "
              "handling at shared tile edges does not match a frame-level "
              "filter (docs/negative-results.md)";
      return ParseStatus::kError;
    }
    // Every §5.18.7/§5.4.8 segmentation or delta-Q surface shares the same
    // header/decision restrictions (single tile, fixed CQP, no CDF chain,
    // no frame-schedule wrappers, no QM).
    const bool video_qp_tools =
        options.video_segmentation || options.video_delta_q ||
        options.video_per_segment_q || options.video_uv_delta_q ||
        options.video_seg_skip || options.video_seg_globalmv ||
        options.video_seg_temporal || options.video_segments != 0;
    if (flag_given(options, "--video-delta-q-res") && !options.video_delta_q) {
      error = "--video-delta-q-res requires --video-delta-q";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q &&
        (options.bit_depth != 8 || options.chroma != "420")) {
      error = "--video-uv-delta-q is only wired on the 8-bit 4:2:0 video core";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q && options.video_tcq) {
      error = "--video-uv-delta-q does not compose with --video-tcq: the "
              "per-block plane quantiser differs from the scalar TCQ step";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q && options.video_cctx) {
      error = "--video-uv-delta-q does not compose with --video-cctx: the "
              "joint U/V search uses one scalar chroma step";
      return ParseStatus::kError;
    }
    if (options.video_uv_delta_q && options.video_qm) {
      error = "--video-uv-delta-q does not compose with --video-qm: the "
              "per-coefficient QM quantiser is not the scalar DC/AC step";
      return ParseStatus::kError;
    }
    if ((options.video_seg_skip || options.video_seg_globalmv) &&
        options.video_segments == 1) {
      error = "--video-seg-skip/--video-seg-globalmv need at least two "
              "segments";
      return ParseStatus::kError;
    }
    if (options.video_seg_skip && options.video_seg_globalmv &&
        options.video_segments != 0 && options.video_segments < 3) {
      error = "--video-seg-skip and --video-seg-globalmv need --video-segments "
              ">= 3 so the features occupy distinct segment ids";
      return ParseStatus::kError;
    }
    if (options.video_seg_globalmv && options.video_per_segment_q) {
      error = "--video-seg-globalmv does not compose with "
              "--video-per-segment-q: an intra leaf cannot stay in the forced "
              "GLOBALMV segment, and reassigning it would change the "
              "quantizer the block was decided with";
      return ParseStatus::kError;
    }
    if (options.video_seg_temporal && options.b_frames != 0) {
      error = "--video-seg-temporal does not compose with --b-frames";
      return ParseStatus::kError;
    }
    if ((options.video_segmentation || options.video_per_segment_q ||
         options.video_seg_skip || options.video_seg_globalmv ||
         options.video_seg_temporal) &&
        options.video_refs != 1) {
      error = "segmentation needs a single reference: PrevSegmentIds and "
              "SkipSegFrame are replayed from ref 0";
      return ParseStatus::kError;
    }
    if ((options.video_seg_skip || options.video_seg_globalmv) &&
        (options.video_compound || options.video_compound_wedge ||
         options.video_compound_diffwtd || options.video_compound_interintra ||
         options.video_cwp_weighted || options.video_imp_mask)) {
      error = "--video-seg-skip/--video-seg-globalmv do not compose with the "
              "compound prediction surfaces (forced GLOBALMV is single-ref)";
      return ParseStatus::kError;
    }
    if (flag_given(options, "--video-seg-delta") && !options.video_per_segment_q &&
        !options.video_delta_q) {
      error = "--video-seg-delta requires --video-per-segment-q or "
              "--video-delta-q";
      return ParseStatus::kError;
    }
    if (options.video_per_segment_q && options.video_delta_q) {
      error = "--video-per-segment-q does not compose with --video-delta-q";
      return ParseStatus::kError;
    }
    if (options.video_per_segment_q) {
      const int segment_q = options.qp + options.video_seg_delta;
      if (segment_q < 1 || segment_q > 255) {
        error = "per-segment Q: --qp + --video-seg-delta must be in [1, 255] "
                "(the video transform path has no lossless segment)";
        return ParseStatus::kError;
      }
    }
    if (video_qp_tools && options.rc_mode != "cqp") {
      error = "segmentation/delta-Q requires a fixed base_q_idx (--rc cqp); "
              "the CRF/VBR per-frame QP plan is not composed with these tools";
      return ParseStatus::kError;
    }
    if (video_qp_tools && options.video_qm) {
      error = "segmentation/delta-Q does not compose with --video-qm: the "
              "frame-header writer owns the using_qmatrix bit";
      return ParseStatus::kError;
    }
    if (video_qp_tools &&
        (options.avg_cdf || options.adaptive_cdf || options.cdf_continuity)) {
      error = "segmentation/delta-Q does not compose with the CDF chain: the "
              "per-tile delta/segment CDFs are not part of the §8.2.4 image";
      return ParseStatus::kError;
    }
    if (video_qp_tools &&
        (frame_modes != 0 || options.b_frames > 0 ||
         options.video_tlayers > 0 || options.video_tip)) {
      error = "segmentation/delta-Q does not compose with the B/temporal-"
              "layer/random-access/TIP frame schedules";
      return ParseStatus::kError;
    }
    if (video_qp_tools && options.tiles != "1x1") {
      error = "segmentation/delta-Q does not support --tiles > 1x1 yet: the "
              "tiled frame-header writer does not signal §5.18.7";
      return ParseStatus::kError;
    }
    if ((options.video_cdef || options.video_ccso || options.video_wiener ||
         options.video_chroma_lr || options.video_gdf) &&
        frame_modes != 0) {
      error = "video loop filters do not compose with the frame-type wrappers "
              "(their component header writers do not signal the filter rows)";
      return ParseStatus::kError;
    }
    if (flag_given(options, "--video-tx-lambda") && !options.video_tx_search) {
      error = "--video-tx-lambda requires --video-tx-search";
      return ParseStatus::kError;
    }
    if (options.video_tx_context_rate && !options.video_tx_search) {
      error = "--video-tx-context-rate requires --video-tx-search";
      return ParseStatus::kError;
    }
    if (options.video_key_intra_txpart && !options.video_tx_search) {
      error = "--video-key-intra-txpart requires --video-tx-search (its "
              "bitstream is TX_MODE_SELECT)";
      return ParseStatus::kError;
    }
    if (options.video_key_intra_txpart && options.video_gpu) {
      error = "--video-key-intra-txpart is CPU-only: the CUDA video TX module "
              "has the single-predictor contract, so per-part intra prediction "
              "is not wired there";
      return ParseStatus::kError;
    }
    if (options.video_key_intra_txpart &&
        (options.bit_depth != 8 || options.chroma != "420")) {
      error = "--video-key-intra-txpart is wired for the 8-bit 4:2:0 CPU "
              "video core";
      return ParseStatus::kError;
    }
    if (options.video_tx_context_rate && options.video_gpu) {
      error = "--video-tx-context-rate is CPU-only: the CUDA TX module still "
              "uses the historical rate model, so --video-gpu parity would "
              "break (use --video)";
      return ParseStatus::kError;
    }
    if (options.video_ccso_rd &&
        (options.bit_depth != 8 || options.chroma != "420")) {
      error = "--video-ccso-rd is wired for the 8-bit 4:2:0 CPU CCSO planner";
      return ParseStatus::kError;
    }
    if (options.video_ccso_header_rd && options.video_gpu) {
      error = "--video-ccso-header-rd is a CPU-measured CCSO decision model; "
              "the CUDA video stream is not calibrated against it (use "
              "--video)";
      return ParseStatus::kError;
    }
    if (options.video_mhccp && (options.avg_cdf || options.adaptive_cdf)) {
      error = "--video-mhccp does not compose with --avg-cdf/--adaptive-cdf: "
              "the §8.2.4 bank does not carry the §5.20.7.32 cfl_mhccp/"
              "cfl_mh_dir CDFs";
      return ParseStatus::kError;
    }    if (options.video_tx_search && frame_modes != 0) {
      error = "--video-tx-search does not compose with the frame-type wrapper "
              "modes (their header writers do not signal tx_mode_select)";
      return ParseStatus::kError;
    }
    if (options.video_pc_wiener && options.video_wiener) {
      error = "--video-pc-wiener replaces the luma Wiener NS decision; do not "
              "combine the two luma restoration tools";
      return ParseStatus::kError;
    }
    if (options.video_qm && options.qp != 90) {
      error = "--video-qm is gated at --qp 90: the shared device QM stage "
              "builds q2 from the pinned 528 quantizer step";
      return ParseStatus::kError;
    }
    if (options.video_user_qm && options.video_gpu) {
      error = "--video-user-qm is CPU-only: the CUDA video backend does not "
              "emit the §5.13 QM OBU";
      return ParseStatus::kError;
    }
    if (options.video_grain && options.video_gpu) {
      error = "--video-grain is CPU-only: the CUDA video backend does not "
              "write §5.14 film grain OBUs or frame headers";
      return ParseStatus::kError;
    }
    if (options.video_idtx_1d) {
      error =
          "--video-idtx-1d is rejected: the one-axis V/H_DCT, V/H_ADST and "
          "V/H_FLIPADST intra types need dc_sign_horz_vert (§5.20.7.27) and "
          "the class-specific scans, and pinned dav2d f9dafbc rejects every "
          "negative H_DCT/V_DCT coefficient under --strict 1 (repro: "
          "hv_neg.ivf from src/codec/intra/supplemental/, strict 0 decodes, "
          "strict 1 fails; avmdec fails too).  Use --video-idtx for the 2-D "
          "identity type";
      return ParseStatus::kError;
    }
    if (options.video_idtx) {
      if (!options.video_tx_search) {
        error = "--video-idtx requires --video-tx-search (without the TX "
                "search every intra residual is DCT_DCT)";
        return ParseStatus::kError;
      }
      if (options.video_gpu) {
        error = "--video-idtx is CPU-only: the CUDA TX search pins "
                "allow_idtx = 0";
        return ParseStatus::kError;
      }
    }
    if (options.video_mrl || options.video_ist) {
      if (options.video_gpu) {
        error = "--video-mrl/--video-ist are CPU-only: the CUDA video backend "
                "does not signal the sequence bits or the per-block syntax";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-mrl/--video-ist are wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-mrl/--video-ist do not compose with --tiles: the "
                "tiled sequence-header writer is a separate copy";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = "--video-mrl/--video-ist do not compose with --svc-layers";
        return ParseStatus::kError;
      }
    }
    if (options.video_ist && options.video_qm) {
      error = "--video-ist does not compose with --video-qm: the §7.15.3 "
              "inverse needs the per-coefficient q2 dequantizer";
      return ParseStatus::kError;
    }
    if (options.video_ibp) {
      // §5.4.5 enable_ibp changes the prediction of every intra leaf, not
      // only the keyframe: the decoder derives applyIbp/useIBP from the
      // sequence flag, YMode and angle_delta.  The CPU writer uses the
      // verified src/codec/intra/ibp component on both the luma search and
      // the DC/chroma paths; combinations that predict through a different
      // path are rejected by name instead of emitting a diverging stream.
      if (options.video_gpu) {
        error = "--video-ibp is CPU-only: the CUDA video backend does not "
                "apply the §7.13.2 IBP predictor";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-ibp is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-ibp does not compose with --tiles: the tiled "
                "sequence-header writer is a separate copy";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = "--video-ibp does not compose with --svc-layers";
        return ParseStatus::kError;
      }
      if (options.video_cfl || options.video_mhccp) {
        error = "--video-ibp requires the plain DC chroma path: "
                "--video-cfl/--video-mhccp do not apply the §7.13.2.12 IBP "
                "DC post-filter";
        return ParseStatus::kError;
      }
      if (options.video_mrl) {
        error = "--video-ibp does not compose with --video-mrl: the IBP "
                "component implements MrlIndex = 0 only";
        return ParseStatus::kError;
      }
      if (options.video_palette || options.video_ibc || options.video_morph) {
        const char* tool = options.video_palette
                               ? "--video-palette"
                               : (options.video_ibc ? "--video-ibc"
                                                   : "--video-morph");
        error = std::string(tool) +
                " is not wired with --video-ibp: the screen-content keyframe "
                "mode-info path does not run the §7.13.2 IBP prediction";
        return ParseStatus::kError;
      }
      if (options.video_compound_interintra) {
        error = "--video-ibp does not compose with "
                "--video-compound-interintra: the §7.13.3.29 inter-intra "
                "predictor does not run the IBP edge-filter/blend path";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-ibp is gated on the plain key+P chain: it cannot be "
                "combined with --b-frames, --video-tlayers, --video-tip or "
                "the frame-type wrapper modes";
        return ParseStatus::kError;
      }
    }
    if (options.video_dip) {
      if (options.video_gpu) {
        error = "--video-dip is CPU-only: the CUDA video backend does not "
                "signal enable_dip or the per-block dip_mode_info()";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-dip is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-dip does not compose with --tiles: the tiled "
                "sequence-header writer is a separate copy";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras || options.video_tip ||
          options.svc_given) {
        error = "--video-dip does not compose with the B/temporal-layer/"
                "random-access/TIP/SVC frame schedules";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = "--video-dip does not compose with --video-lossless: the "
                "lossless writer has its own keyframe mode-info path";
        return ParseStatus::kError;
      }
      if (options.video_sdp) {
        error = "--video-dip does not compose with --video-sdp: the SDP "
                "luma/chroma tree mode-info path is not gated with DIP";        return ParseStatus::kError;
      }
    }
    if (options.video_sdp) {
      if (options.video_gpu) {
        error = "--video-sdp is CPU-only: the CUDA video backend does not "
                "emit the SDP luma/chroma trees";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-sdp is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-sdp is wired for a single tile";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras || options.video_tip ||
          options.svc_given) {
        error = "--video-sdp does not compose with the B/temporal-layer/"
                "random-access/TIP/SVC frame schedules";
        return ParseStatus::kError;
      }
      if (options.video_morph || options.video_cfl || options.video_mhccp) {
        error = "--video-sdp does not compose with --video-morph/--video-cfl/"
                "--video-mhccp: the SDP keyframe chroma tree has its own mode "
                "and CflAllowedInSdp gating";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q || options.video_uv_delta_q) {
        error = "--video-sdp does not compose with segmentation/delta-Q";
        return ParseStatus::kError;
      }
    }
    if (options.video_ddt) {
      if (options.video_gpu) {
        error = "--video-ddt is CPU-only: the CUDA video backend does not "
                "apply the DDTX/FDDT kernels";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-ddt is wired for 8-bit 4:2:0";
        return ParseStatus::kError;
      }
      if (options.video_lossless) {
        error = "--video-ddt does not compose with --video-lossless: a "
                "lossless block never signals ADST/FDST";
        return ParseStatus::kError;
      }
      if (options.video_tip || options.svc_given) {
        error = "--video-ddt does not compose with --video-tip or "
                "--svc-layers: those header writers do not signal "
                "enable_inter_ddt";
        return ParseStatus::kError;
      }    }
    if (options.video_lossless) {
      if (options.video_gpu) {
        error = "--video-lossless is CPU-only";
        return ParseStatus::kError;
      }
      if (flag_given(options, "--qp") && options.qp != 0) {
        error = "--video-lossless is Q=0; drop --qp or pass --qp 0";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 && options.bit_depth != 10) {
        error = "--video-lossless supports 8-bit and 10-bit (12-bit is not "
                "signalable in AV2 v1.0.0)";
        return ParseStatus::kError;
      }
      if (options.video_lossless_b &&
          options.video_tlayers != 4 && options.video_tlayers != 8) {
        error = "--video-lossless-b codes one closed B GOP: pass "
                "--video-tlayers 4 or 8 (GOP size in pictures)";
        return ParseStatus::kError;
      }
      if (flag_given(options, "--video-min-block")) {
        if (options.video_min_block != 8 && options.video_min_block != 16 &&
            options.video_min_block != 32) {
          error = "--video-lossless uses --video-min-block as its coding "
                  "block size; pass 8, 16 or 32";
          return ParseStatus::kError;
        }
      }
      const std::pair<bool, const char*> not_applied[] = {
          {options.video_qm, "--video-qm"},
          {options.video_user_qm, "--video-user-qm"},
          {options.video_tcq, "--video-tcq"},
          {options.video_rdoq, "--video-rdoq"},
          {options.video_cctx, "--video-cctx"},
          {options.video_grain, "--video-grain"},
          {options.video_tx_search, "--video-tx-search"},
          {options.video_key_intra_txpart, "--video-key-intra-txpart"},
          {options.video_partition_rd, "--video-partition-rd"},
          {options.video_intra_full, "--video-intra-full"},
          {options.video_cfl, "--video-cfl"},
          {options.video_mhccp, "--video-mhccp"},
          {options.video_deblock, "--video-deblock"},
          {options.video_cdef, "--video-cdef"},
          {options.video_ccso, "--video-ccso"},
          {options.video_wiener, "--video-wiener"},
          {options.video_pc_wiener, "--video-pc-wiener"},
          {options.video_chroma_lr, "--video-chroma-lr"},
          {options.video_gdf, "--video-gdf"},
          {options.video_gme, "--video-gme"},
          {options.video_warp, "--video-warp"},
          {options.video_bawp, "--video-bawp"},
          {options.video_tip, "--video-tip"},
          {options.video_mrl, "--video-mrl"},
          {options.video_ibp, "--video-ibp"},
          {options.video_dip, "--video-dip"},
          {options.video_ist, "--video-ist"},
          {options.video_uv_delta_q, "--video-uv-delta-q"},
          {options.video_segmentation, "--video-segmentation"},
          {options.video_delta_q, "--video-delta-q"},
          {options.video_per_segment_q, "--video-per-segment-q"},
          {options.video_seg_skip, "--video-seg-skip"},
          {options.video_seg_globalmv, "--video-seg-globalmv"},
          {options.video_seg_temporal, "--video-seg-temporal"},
          {options.video_segments != 0, "--video-segments"},
          {options.video_tip_no_hole_fill, "--video-tip-no-hole-fill"},
          {options.video_compound, "--video-compound"},
          {options.video_inter_residual, "--video-inter-residual"},
          {options.video_leaf_rd, "--video-leaf-rd"},
          {options.video_subpel_eighth, "--video-subpel-eighth"},
          {options.video_refs != 1, "--video-refs"},
          {options.b_frames != 0, "--b-frames"},
          {options.video_long_term, "--video-long-term"},
          {options.video_olk, "--video-olk"},
          {options.video_leading != 0, "--video-leading"},
          {options.video_sef, "--video-sef"},
          {options.video_switch, "--video-switch"},
          {options.video_ras, "--video-ras"},
          {options.tiles != "1x1", "--tiles"},
          {options.rc_mode != "cqp" || options.rc_given, "--rc"},
      };
      for (const auto& [set, flag] : not_applied) {
        if (!set) continue;
        error = std::string(flag) +
                " is not applied by the --video-lossless writer";
        return ParseStatus::kError;
      }
    }
    if (options.video_tcq || options.video_cctx) {
      const char* tool = options.video_tcq ? "--video-tcq" : "--video-cctx";
      if (options.video_gpu) {
        error = std::string(tool) + " is CPU-only: the CUDA video backend has "
                "no TCQ/CCTX entropy path";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = std::string(tool) + " is wired for the 8-bit 4:2:0 video core "
                "only; the --bit-depth/--chroma format cores keep the "
                "historical sequence header";
        return ParseStatus::kError;
      }
      if (options.video_qm || options.video_user_qm) {
        error = std::string(tool) + " does not compose with --video-qm: the "
                "per-coefficient QM quantiser is not the scalar step the "
                "TCQ/CCTX search reconstructs with";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q || options.video_uv_delta_q) {
        error = std::string(tool) + " does not compose with segmentation/"
                "delta-Q: the per-block quantiser differs from the frame "
                "base_q_idx the TCQ/CCTX decision uses";
        return ParseStatus::kError;
      }
      if (options.video_lossless || options.qp == 0) {
        error = std::string(tool) + " does not compose with lossless "
                "quantisation (CodedLossless forces allow_tcq = 0 and "
                "is_cctx_allowed() = 0)";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = std::string(tool) + " does not compose with --tiles > 1x1: "
                "the tiled sequence/frame header writer does not signal the "
                "TCQ/CCTX flags";
        return ParseStatus::kError;
      }
      if (frame_modes != 0 || options.video_tip) {
        error = std::string(tool) + " does not compose with the frame-type "
                "wrapper modes: their header writers do not signal allow_tcq "
                "or the CCTX sequence bit";
        return ParseStatus::kError;
      }
      if (options.svc_given) {
        error = std::string(tool) + " does not compose with --svc-layers";
        return ParseStatus::kError;
      }
    }
    if (options.video_leaf_rd && options.video_gpu) {
      error = "--video-leaf-rd is CPU-only: the CUDA video backend keeps the "
              "historical leaf decisions so its streams stay CPU-reference "
              "exact (drop --video-gpu)";
      return ParseStatus::kError;
    }
    if (options.video_skip_collapse) {
      if (options.video_gpu) {
        error = "--video-skip-collapse is a CPU partition decision: the CUDA "
                "backend solves its own tree (drop --video-gpu)";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-skip-collapse needs the whole-frame single-tile "
                "decision pass (--tiles 1x1)";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q) {
        error = "--video-skip-collapse does not compose with the per-segment "
                "quantiser tools: a full-SB skip leaf cannot code a delta";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-skip-collapse is wired for the 8-bit 4:2:0 video "
                "core only";
        return ParseStatus::kError;
      }
    }
    if (options.video_qm && options.video_tx_search) {
      error = "--video-qm does not compose with --video-tx-search: the device "
              "TX search module has its own scalar quantizer";
      return ParseStatus::kError;
    }
    if (options.video_rdoq) {
      if (options.video_gpu) {
        error = "--video-rdoq is CPU-only: the CUDA video backend has no "
                "coefficient trellis";
        return ParseStatus::kError;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        error = "--video-rdoq is wired for the 8-bit 4:2:0 video core only";
        return ParseStatus::kError;
      }
      if (!options.video_inter_residual) {
        error = "--video-rdoq needs --video-inter-residual: the default video "
                "chain codes no inter residual to quantise";
        return ParseStatus::kError;
      }
      if (options.video_tcq) {
        error = "--video-rdoq replaces the scalar quantiser, it does not "
                "compose with --video-tcq (its half-step trellis owns the "
                "levels)";
        return ParseStatus::kError;
      }
      if (options.video_qm || options.video_user_qm) {
        error = "--video-rdoq does not compose with --video-qm: the "
                "per-coefficient QM quantiser is not the scalar step the "
                "trellis distorts against";
        return ParseStatus::kError;
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q || options.video_uv_delta_q) {
        error = "--video-rdoq does not compose with segmentation/delta-Q: the "
                "trellis uses the frame base_q_idx CDF slice";
        return ParseStatus::kError;
      }
      if (options.video_lossless || options.qp == 0) {
        error = "--video-rdoq does not compose with lossless quantisation";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1" || frame_modes != 0 || options.video_tip ||
          options.svc_given) {
        error = "--video-rdoq is wired on the single-tile plain P chain only";
        return ParseStatus::kError;
      }
    }
    if (options.video_mhccp && frame_modes != 0) {
      error = "--video-mhccp does not compose with the frame-type wrapper "
              "modes (their sequence headers do not signal enable_mhccp)";
      return ParseStatus::kError;
    }
    if (options.video_tx_search && options.svc_given) {
      error = "--video-tx-search does not compose with --svc-layers";
      return ParseStatus::kError;
    }
    if ((options.video_switch || options.video_ras) &&
        (options.video_refs != 1 || options.video_mv_precision != 0)) {
      error = "--video-switch/--video-ras need --video-refs 1 and eighth-pel "
              "MV precision";
      return ParseStatus::kError;
    }
    if (options.video_gme) {
      // The gated global-motion path is the plain P chain: one model per
      // explicit reference, GLOBALMV projection/warp, no SWITCH/RAS special
      // header form, no per-frame B/temporal plan and no uniform tile grid.
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          frame_modes != 0 || options.video_compound ||
          options.video_inter_residual || options.video_drl_reorder) {
        error = "--video-gme/--video-warp are gated on the plain skip-only P "
                "chain: they cannot be combined with --b-frames, "
                "--video-tlayers, a frame-type wrapper, --video-compound, "
                "--video-inter-residual or --video-drl-reorder";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-gme/--video-warp are gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if (options.video_bawp_carrier) {
      // CPU-only: the CUDA BAWP backend is skip-only and does not carry the
      // GLOBALMV->NEARMV mode swap.  Same plain-chain surface as --video-bawp.
      if (options.video_gpu) {
        error = "--video-bawp-carrier is CPU-only: the CUDA BAWP backend is "
                "skip-only and keeps the historical BAWP decisions";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-bawp-carrier is gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if (options.video_bawp) {
      // BAWP is a single-prediction luma tool.  The gated product surface is
      // the plain P chain with no warp/compound/temporal-frame wrapper and a
      // single tile (the fitted template needs whole-frame coordinates).
      //
      // BAWP composes with --video-inter-residual through the CPU-only
      // carrier: the fitted predictor is the §7.13.3.25 prediction the
      // residual is coded against, so the decoder reconstructs BAWP +
      // residual exactly.  Plain --video-bawp stays skip-only.
      const bool residual_ok =
          !options.video_inter_residual || options.video_bawp_carrier;
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          frame_modes != 0 || options.video_compound ||
          options.video_gme || options.video_warp ||
          options.video_switch || options.video_ras || options.video_olk ||
          options.video_leading != 0 || options.video_sef || !residual_ok) {
        error = "--video-bawp is gated on the plain "
                "single-prediction P chain: it cannot be combined with "
                "--b-frames, --video-tlayers, a frame-type wrapper, "
                "--video-compound or --video-gme/--video-warp"
                " (and --video-inter-residual needs "
                "--video-bawp-carrier)";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-bawp is gated on a single tile";
        return ParseStatus::kError;
      }
    }
    if (options.video_compound_interintra) {
      // §5.20.7.14 motion_mode_allowed() sets AllowedMotionModes[INTERINTRA]
      // only for !isCompound, so the §7.13.3.29/.30 mask blend rides the
      // single-reference blocks.  The flag is kept under its historical
      // --video-compound-interintra name; the syntax is not compound.
      if (options.video_gpu) {
        error = "--video-compound-interintra is CPU-only: the CUDA M2 backend "
                "does not code the §5.20.7.14 motion-mode syntax";
        return ParseStatus::kError;
      }
      if (options.video_compound || options.video_compound_wedge ||
          options.video_compound_diffwtd) {
        error = "--video-compound-interintra is the single-reference "
                "COMPOUND_INTRA path; it cannot be combined with "
                "--video-compound (reference_select makes the block compound, "
                "which forbids INTERINTRA)";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1) {
        error = "--video-compound-interintra requires --video-refs 1: "
                "§5.20.7.14 allows INTERINTRA only for a single-reference "
                "block";
        return ParseStatus::kError;
      }
      if (options.video_bawp || options.video_morph) {
        error = "--video-compound-interintra cannot be combined with "
                "--video-bawp/--video-morph: §5.20.7.14 returns before "
                "AllowedMotionModes[INTERINTRA] when use_bawp is set";
        return ParseStatus::kError;
      }
      if (options.b_frames != 0 || options.video_tlayers != 0 ||
          options.video_tip || options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras) {
        error = "--video-compound-interintra is gated on the plain key+P "
                "chain: it cannot be combined with --b-frames, "
                "--video-tlayers, --video-tip or the frame-type wrappers";
        return ParseStatus::kError;
      }
      if (options.tiles != "1x1") {
        error = "--video-compound-interintra is gated on a single tile";
        return ParseStatus::kError;
      }
      if (options.avg_cdf || options.cdf_continuity) {
        error = "--video-compound-interintra uses the per-tile §9.3 "
                "inter-intra CDFs; it is not wired with "
                "--avg-cdf/--cdf-continuity";
        return ParseStatus::kError;
      }
    }
    if (options.video_compound) {
      if (options.b_frames == 0 && options.video_refs != 2) {
        error = "--video-compound needs --video-refs 2 (the (0,1) pair is "
                "implied by NumTotalRefs == 2)";
        return ParseStatus::kError;
      }
      if (options.video_drl_reorder || options.video_long_term ||
          options.video_olk || options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras ||
          options.video_tlayers != 0) {
        error = "--video-compound cannot be combined with the frame-type or "
                "DRL-reorder modes";
        return ParseStatus::kError;
      }
    } else if (options.video_compound_wedge ||
               options.video_compound_diffwtd) {
      error = "--video-compound-wedge/--video-compound-diffwtd need "
              "--video-compound";
      return ParseStatus::kError;
    }
    if (options.video_gops != 1 && options.b_frames == 0) {
      error = "--video-gops only applies to the --b-frames schedule";
      return ParseStatus::kError;
    }
    if (options.b_frames != 0) {
      if (options.video_tlayers != 0 || frame_modes != 0) {
        error = "--b-frames cannot be combined with the temporal-layer or "
                "frame-type modes";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1) {
        error = "--b-frames owns the per-frame reference list; keep "
                "--video-refs 1";
        return ParseStatus::kError;
      }
      if (options.video_drl_reorder) {
        error = "--b-frames cannot be combined with --video-drl-reorder";
        return ParseStatus::kError;
      }
    }
    if (options.video_tlayers != 0) {
      if (options.video_tlayers != 4 && options.video_tlayers != 8) {
        error = "--video-tlayers supports 4 or 8 frames (one closed GOP)";
        return ParseStatus::kError;
      }
      if (frame_modes != 0 || options.video_leading != 0) {
        error = "--video-tlayers cannot be combined with the frame-type modes";
        return ParseStatus::kError;
      }
      if (options.video_refs != 1 || options.video_mv_precision != 0) {
        error = "--video-tlayers needs --video-refs 1 and eighth-pel MV "
                "precision";
        return ParseStatus::kError;
      }
      if (options.video_tlayer_drop > 3) {
        error = "--video-tlayer-drop is at most 3 (obsolete: max_tlayer_id "
                "is f(2))";
        return ParseStatus::kError;
      }
      // The hierarchical-P plan stops at max_tlayer_id = 1 for N = 4 and 2
      // for N = 8. Reject a higher drop here instead of failing mid-encode.
      const int plan_max_tlayer = options.video_tlayers == 4 ? 1 : 2;
      if (options.video_tlayer_drop > plan_max_tlayer) {
        error = "--video-tlayer-drop " +
                std::to_string(options.video_tlayer_drop) + " is above the " +
                std::to_string(options.video_tlayers) +
                "-frame plan's max_tlayer_id " +
                std::to_string(plan_max_tlayer);
        return ParseStatus::kError;
      }
    } else if (options.video_tlayer_drop >= 0) {
      error = "--video-tlayer-drop requires --video-tlayers";
      return ParseStatus::kError;
    }

    if (options.video_tip_no_hole_fill && !options.video_tip) {
      error = "--video-tip-no-hole-fill is the A/B control for --video-tip";
      return ParseStatus::kError;
    }

    if (options.video_tip_block_ref && !options.video_tip) {
      error =
          "--video-tip-block-ref is the §5.20.7.10 RefFrame[0] == "
          "TIP_FRAME carrier for --video-tip --video-ref-mvs";
      return ParseStatus::kError;
    }
    if (options.video_tip_block_ref && !options.video_ref_mvs) {
      error = "--video-tip-block-ref requires --video-ref-mvs";
      return ParseStatus::kError;
    }

    if (options.video_gpu) {
      // The CUDA M2 backend mirrors the flat one-reference P chain.  Every
      // option that builds a different ref/header model is rejected by name
      // instead of being accepted and silently ignored (or falling back to the
      // CPU encoder).  --video without this flag keeps the full CPU surface.
#if !AV2ENC_VIDEO_GPU_CALLABLE
#if AV2ENC_HAVE_CUDA
      error = "--video-gpu is not available in this build: the M2 video GPU "
              "entry point (src/cuda/pipeline/video/) is not linked. Use "
              "--video for the CPU reference path.";
#else
      error = "--video-gpu requires a CUDA build; this is a CPU-only reference "
              "build. Use --video for the CPU reference path.";
#endif
      return ParseStatus::kError;
#else
      const auto reject_gpu = [&](const char* flag) {
        error = std::string("--video-gpu does not implement ") + flag +
                " yet; use --video for the CPU reference path";
        return ParseStatus::kError;
      };
      if (options.video_cdf_per_frame_qp) {
        return reject_gpu("--video-cdf-per-frame-qp (per-frame CDF image "
                          "reset)");
      }
      if (options.video_tip) {
        return reject_gpu("--video-tip (TIP frame scheduling)");
      }
      if (options.video_ref_mvs) {
        return reject_gpu("--video-ref-mvs (§7.9/§7.22 temporal MVs)");
      }
      if (options.video_bru) {
        return reject_gpu("--video-bru (backwards reference update)");
      }
      if (options.video_opfl) {
        return reject_gpu("--video-opfl (CPU §7.13.3.8-.12 OPFL)");
      }
      if (options.video_refine_mv) {
        return reject_gpu("--video-refine-mv (CPU §7.13.3.3-.6 refine-MV)");
      }
      if (options.cdf_continuity) {
        return reject_gpu("--cdf-continuity (cross-frame CDF carry)");
      }
      if (options.video_ref_signal_compact) {
        return reject_gpu("--video-ref-signal-compact (CPU §5.18.2 compact "
                          "reference signalling)");
      }
      if (options.video_segmentation || options.video_delta_q ||
          options.video_per_segment_q || options.video_uv_delta_q ||
          options.video_seg_skip || options.video_seg_globalmv ||
          options.video_seg_temporal || options.video_segments != 0) {
        return reject_gpu("--video-segmentation/--video-delta-q/"
                          "--video-per-segment-q/--video-uv-delta-q "
                          "(CPU reference only)");
      }
      if (options.svc_given) return reject_gpu("--svc-layers");
      if (options.video_implicit_ref_map) {
        return reject_gpu("--video-implicit-ref-map (§7.7 implicit maps)");
      }
      if (options.video_morph) {
        return reject_gpu("--video-morph (IBC keyframe mode info)");
      }
      if (options.video_palette) {
        return reject_gpu("--video-palette (palette keyframe mode info)");
      }
      if (options.video_ibc) {
        return reject_gpu("--video-ibc (IBC keyframe mode info)");
      }
      if (options.video_ibp) {
        return reject_gpu("--video-ibp (§7.13.2 intra bi-prediction)");
      }
      if (options.video_key_intra_txpart) {
        return reject_gpu("--video-key-intra-txpart (per-part intra "
                          "prediction/recon feedback)");
      }
      if (options.video_compound_interintra) {
        return reject_gpu("--video-compound-interintra (§5.20.7.14 motion "
                          "mode)");
      }
#endif
    }
  } else {
    if (flag_given(options, "--p-frames") ||
        flag_given(options, "--b-frames") ||
        flag_given(options, "--video-refs") ||
        flag_given(options, "--video-num-ref-frames") ||
        flag_given(options, "--video-implicit-ref-map") ||
        flag_given(options, "--video-subpel-eighth") ||
        flag_given(options, "--video-mv-precision") ||
        flag_given(options, "--video-inter-residual") ||
        flag_given(options, "--video-tx-search") ||
        flag_given(options, "--video-key-intra-txpart") ||
        flag_given(options, "--video-sdp") ||
        flag_given(options, "--video-ddt") ||
        flag_given(options, "--video-tx-lambda") ||
        flag_given(options, "--video-deblock") ||
        flag_given(options, "--video-segmentation") ||
        flag_given(options, "--video-delta-q") ||
        flag_given(options, "--video-delta-q-res") ||
        flag_given(options, "--video-per-segment-q") ||
        flag_given(options, "--video-seg-delta") ||
        flag_given(options, "--video-uv-delta-q") ||
        flag_given(options, "--video-seg-skip") ||
        flag_given(options, "--video-seg-globalmv") ||
        flag_given(options, "--video-seg-temporal") ||
        flag_given(options, "--video-segments") ||
        flag_given(options, "--video-gme") ||
        flag_given(options, "--video-warp") ||
        flag_given(options, "--video-bawp") ||
        flag_given(options, "--video-morph") ||
        flag_given(options, "--video-palette") ||
        flag_given(options, "--video-ibc") ||
        flag_given(options, "--video-opfl") ||
        flag_given(options, "--video-refine-mv") ||
        flag_given(options, "--video-compound-near-near") ||
        flag_given(options, "--video-tip") ||
        flag_given(options, "--video-tip-no-hole-fill") ||
        flag_given(options, "--video-intra-full") ||
        flag_given(options, "--video-cfl") ||
        flag_given(options, "--video-mrl") ||
        flag_given(options, "--video-ibp") ||
        flag_given(options, "--video-dip") ||
        flag_given(options, "--video-ist") ||
        flag_given(options, "--video-mhccp") ||
        flag_given(options, "--video-qm") ||
        flag_given(options, "--video-tlayers") ||
        flag_given(options, "--video-tlayer-drop") ||
        flag_given(options, "--video-compound") ||
        flag_given(options, "--video-drl-reorder") ||
        flag_given(options, "--video-drl-reorder-always") ||
        flag_given(options, "--video-long-term") ||
        flag_given(options, "--video-olk") ||
        flag_given(options, "--video-leading") ||
        flag_given(options, "--video-sef") ||
        flag_given(options, "--video-switch") ||
        flag_given(options, "--video-ras") ||
        flag_given(options, "--video-force-single") ||
        flag_given(options, "--video-skip-collapse") ||
        flag_given(options, "--video-sb128") ||
        flag_given(options, "--video-sb256") ||
        flag_given(options, "--video-ext-partitions") ||
        flag_given(options, "--video-min-block") ||
        flag_given(options, "--video-max-depth") ||
        flag_given(options, "--video-me-range") ||
        flag_given(options, "--video-cdef") ||
        flag_given(options, "--video-cdef-rd") ||
        flag_given(options, "--video-cdef-rd-lambda") ||
        flag_given(options, "--video-cdef-strength") ||
        flag_given(options, "--video-cdef-damping") ||
        flag_given(options, "--video-cdef-search") ||
        flag_given(options, "--video-ccso") ||
        flag_given(options, "--video-ccso-bands") ||
        flag_given(options, "--video-ccso-rd") ||
        flag_given(options, "--video-ccso-header-rd") ||
        flag_given(options, "--video-wiener") ||
        flag_given(options, "--video-pc-wiener") ||
        flag_given(options, "--video-chroma-lr") ||
        flag_given(options, "--video-gdf") ||
        flag_given(options, "--video-trace") ||
        flag_given(options, "--video-recon")) {
      error = "video options require --video";
      return ParseStatus::kError;
    }
    if (options.rc_given) {
      error = "--rc is wired for --video only; the still paths keep the fixed "
              "--qp stream";
      return ParseStatus::kError;
    }
    if (options.rc_crf_given) {
      error = "--crf requires --video --rc crf";
      return ParseStatus::kError;
    }
    if (options.rc_bitrate_given) {
      error = "--bitrate requires --video --rc vbr";
      return ParseStatus::kError;
    }
  }
  if (options.lossless) {
    if (options.m1) {
      return reject("--lossless has its own still-picture writer; do not "
                    "combine it with --still");
    }
    for (const std::string_view flag : options.specified) {
      if ((flag.starts_with("--still-") && flag != "--still-gpu") ||
          flag == "--no-cfl" || flag == "--no-deblock") {
        return reject(std::string(flag) +
                      " is not applied by the --lossless writer");
      }
    }
  }
  // Every other --still tool is a modifier of the Still paths. Outside those
  // paths it would be accepted and ignored, so it is a usage error by name.
  if (!options.m1 && !options.still_gpu) {
    for (const std::string_view flag : options.specified) {
      // Multi-tile stills report their own, more specific mode error below.
      if (flag == "--no-deblock" && options.tiles != "1x1") continue;
      if ((flag.starts_with("--still-") && flag != "--still-gpu") ||
          flag == "--no-cfl" || flag == "--no-deblock") {
        return reject(std::string(flag) + " requires --still or --still-gpu");
      }
    }
  }
  if (!options.video && options.max_frames > 1) {
    return reject(
        "--frames > 1 requires --video; the still modes encode one frame");
  }
  if (options.video && !options.video_lossless && options.qp == 0) {
    return reject("--video does not support lossless QP 0; use "
                  "--video-lossless");
  }
  if (options.video && options.bit_depth == 8 && options.qp > 255) {
    return reject("--qp > 255 requires 10-bit: MaxQ is 255 at 8-bit");
  }
  if (options.video && options.bit_depth == 10 && options.qp > 303) {
    return reject("--qp > 303 exceeds the 10-bit MaxQ");
  }
  if (!options.video && options.qp == 0 && !options.lossless) {
    return reject("--qp 0 requires --lossless");
  }
  if (options.lossless && flag_given(options, "--qp") && options.qp != 0) {
    return reject("--lossless is Q=0; drop --qp or pass --qp 0");
  }
  // The CPU Still path selects the 8.3.1 coefficient-CDF qctx slice from the
  // raw base_q_idx and derives the quantiser from qlookup(base_q_idx), so QP
  // 1..255 is supported there. The 10-bit format path supports 1..303; 8-bit
  // 4:2:2/4:4:4/4:0:0 pins 90. Basic and the 8-bit 4:2:0 Still GPU path pin 90.
  const bool format_still =
      options.bit_depth != 8 || options.chroma != "420";
  const bool still_cpu_qp =
      (options.m1 && !options.still_gpu) ||
      (options.still_gpu && format_still);
  if (!options.lossless && !options.video && !still_cpu_qp &&
      options.qp != 90) {
    return reject(options.still_gpu
                      ? "--still-gpu pins --qp 90; use --still for QP 1..255"
                      : "Basic pins --qp 90; use --still for QP 1..255");
  }
  // Debug tuning knobs only modify the tool that consumes them.
  if (flag_given(options, "--still-rdo-scale") && !options.still_rdo_intra) {
    return reject("--still-rdo-scale requires --still-rdo-intra");
  }
  if (flag_given(options, "--still-rdoq-scale") && !options.still_rdoq) {
    return reject("--still-rdoq-scale requires --still-rdoq");
  }
  if (flag_given(options, "--still-tx-type-scale") && !options.still_tx_type) {
    return reject("--still-tx-type-scale requires --still-tx-type");
  }
  if (flag_given(options, "--still-ist-scale") && !options.still_ist) {
    return reject("--still-ist-scale requires --still-ist");
  }
  if (flag_given(options, "--still-tcq-scale") && !options.still_tcq) {
    return reject("--still-tcq-scale requires --still-tcq");
  }
  if (flag_given(options, "--still-cdef-strengths") && !options.still_cdef) {
    return reject("--still-cdef-strengths requires --still-cdef");
  }
  if (flag_given(options, "--still-ccso-bands") && !options.still_ccso) {
    return reject("--still-ccso-bands requires --still-ccso");
  }
  if (flag_given(options, "--still-delta-q-res") && !options.still_delta_q) {
    return reject("--still-delta-q-res requires --still-delta-q");
  }
  if (flag_given(options, "--still-qm-level") && !options.still_qm) {
    return reject("--still-qm-level requires --still-qm");
  }
  if (options.lossless_force_fsc && !options.lossless &&
      !options.video_lossless) {
    return reject("--lossless-fsc requires --lossless or --video-lossless");
  }
  // The CPU Still writer has its own tool-combination rules. Reject them at
  // the CLI so they exit 2 with the flag names instead of surfacing as an
  // encode failure after the output file has been opened.
  if (options.m1 && !options.still_gpu) {
    if (options.still_qm && options.still_rdoq) {
      return reject("--still-qm is not supported with --still-rdoq yet");
    }
    if (options.still_dc_ac_delta_q && options.still_qm) {
      return reject("--still-dc-ac-delta-q is not supported with --still-qm yet");
    }
    if (options.still_dc_ac_delta_q && options.still_rdoq) {
      return reject(
          "--still-dc-ac-delta-q is not supported with --still-rdoq yet");
    }
    if (options.still_mrl && options.still_ibp) {
      return reject("--still-mrl and --still-ibp cannot be combined");
    }
    if (options.still_mrl && options.still_rdo_intra) {
      return reject("--still-mrl and --still-rdo-intra cannot be combined");
    }
    if (options.still_mhccp && !options.still_cfl) {
      return reject("--still-mhccp requires CfL (do not combine with --no-cfl)");
    }
    // Only the luma restoration type is exclusive: PC-Wiener and Wiener NS
    // both own plane 0. GDF and chroma LR now compose with either.
    if (options.still_pc_wiener && options.still_wiener) {
      return reject("--still-pc-wiener cannot be combined with --still-wiener");
    }
    // The remaining pairs are the CPU Still writer's own rules
    // (src/pipeline/still/still_encoder.cpp). Mirror them here so they exit 2
    // before IvfWriter::open instead of emitting a header-only IVF.
    if (options.still_qm && options.still_tcq) {
      return reject("--still-qm is not supported with --still-tcq yet");
    }
    if (options.still_qm && options.still_cctx) {
      return reject("--still-qm is not supported with --still-cctx yet");
    }
    if (options.still_qm && options.still_tx_partition) {
      return reject(
          "--still-qm is not supported with --still-tx-partition yet");
    }
    if (options.still_qm && options.still_per_segment_q) {
      return reject(
          "--still-qm is not supported with --still-per-segment-q yet");
    }
    if (options.still_cctx && options.still_dc_ac_delta_q) {
      return reject(
          "--still-cctx is not supported with --still-dc-ac-delta-q yet");
    }
    if (options.still_tx_partition && options.still_dc_ac_delta_q) {
      return reject("--still-tx-partition is not supported with "
                    "--still-dc-ac-delta-q yet");
    }
    if (options.still_tx_partition && options.still_dip) {
      return reject(
          "--still-tx-partition is not supported with --still-dip yet");
    }
    if (options.still_tx_partition && options.still_mrl) {
      return reject(
          "--still-tx-partition is not supported with --still-mrl yet");
    }
    if (options.still_tx_partition && options.still_ibp) {
      return reject(
          "--still-tx-partition is not supported with --still-ibp yet");
    }
    if (options.still_tcq && options.still_rdoq) {
      return reject("--still-tcq is not supported with --still-rdoq yet");
    }
    if (options.still_tcq && options.still_dc_ac_delta_q) {
      return reject(
          "--still-tcq is not supported with --still-dc-ac-delta-q yet");
    }
    if (options.still_tcq && options.still_ist) {
      return reject("--still-tcq is not supported with --still-ist yet");
    }
    if (options.still_per_segment_q && options.still_ist) {
      return reject(
          "--still-per-segment-q is not supported with --still-ist yet");
    }
    if (options.still_ddt) {
      return reject("--still-ddt is not signallable in a still picture: DDT/"
                    "DDTX/FDDT is inter-only (use the inter residual path)");
    }
  }
  if (options.still_ext_partitions) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-ext-partitions requires --still";
      return ParseStatus::kError;
    }

    if (options.still_single) {
      error = "--still-ext-partitions cannot be combined with --still-single";
      return ParseStatus::kError;
    }
  }
  if (options.svc_given) {
    if (!options.video) {
      error = "--svc-* requires --video";
      return ParseStatus::kError;
    }
    if (options.svc_layers.empty()) {
      error = "--svc-* requires --svc-layers x0=WxH[,x1=WxH,...]";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--svc-layers cannot be combined with --tiles: layered tiling is "
              "not gated";
      return ParseStatus::kError;
    }

    const int selections = (options.svc_oppoint >= 0 ? 1 : 0) +
                           (options.svc_all_layers ? 1 : 0) +
                           (options.svc_spatial_layers >= 1 ? 1 : 0);
    if (selections > 1) {
      error = "select at most one of --oppoint, --alllayers or "
              "--spatial-layers";
      return ParseStatus::kError;
    }
    if (options.svc_temporal != 1 && options.svc_embedded != 1) {
      error = "--svc-temporal > 1 does not compose with --svc-embedded > 1: "
              "the embedded refinement ladder needs a cross-mlayer reference "
              "schedule inside the hierarchical GOP; encode one of them at a "
              "time";
      return ParseStatus::kError;
    }
    if (options.video_long_term || options.video_olk ||
        options.video_leading != 0 || options.video_sef ||
        options.video_switch || options.video_ras ||
        options.video_drl_reorder) {
      error = "--svc-layers cannot be combined with a frame-type mode";
      return ParseStatus::kError;
    }
    // run_svc_encode() rebuilds svc.video from a hand-picked field list: it
    // hard-codes num_total_refs = 1 and never copies temporal_layers /
    // tlayer_drop. Reject those flags by name instead of accepting them and
    // emitting a byte-identical stream.
    if (flag_given(options, "--video-refs") ||
        flag_given(options, "--video-num-ref-frames") ||
        flag_given(options, "--video-tlayers") ||
        flag_given(options, "--video-tlayer-drop")) {
      error = "--svc-layers codes one reference and one temporal layer per "
              "spatial layer; --video-refs / --video-tlayers / "
              "--video-tlayer-drop are not wired into the SVC path";
      return ParseStatus::kError;
    }
  }
  if (options.still_ext_partitions) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-ext-partitions requires --still";
      return ParseStatus::kError;
    }

    if (options.still_single) {
      error = "--still-ext-partitions cannot be combined with --still-single";
      return ParseStatus::kError;
    }
  }
  if (options.still_sub8) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-sub8 requires --still";
      return ParseStatus::kError;
    }
    if (options.still_single) {
      error = "--still-sub8 cannot be combined with --still-single";
      return ParseStatus::kError;
    }
    if (options.still_ext_partitions) {
      error = "--still-sub8 is not wired with --still-ext-partitions yet";
      return ParseStatus::kError;
    }
  }
  if (options.still_sdp) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-sdp requires --still";
      return ParseStatus::kError;
    }

    if (options.still_single) {
      error = "--still-sdp cannot be combined with --still-single";
      return ParseStatus::kError;
    }
    if (options.still_sb128 || options.still_sb256) {
      error = "--still-sdp is only wired for 64x64 superblocks";
      return ParseStatus::kError;
    }
    if (options.still_ext_partitions) {
      error = "--still-sdp is not wired with --still-ext-partitions yet";
      return ParseStatus::kError;
    }
    if (options.still_sub8) {
      error = "--still-sdp is not wired with --still-sub8 yet";
      return ParseStatus::kError;
    }
    if (options.still_cdef || options.still_ccso) {
      error = "--still-sdp with CDEF/CCSO is not wired yet";
      return ParseStatus::kError;
    }
    if (options.still_segmentation || options.still_delta_q) {
      error = "--still-sdp with segmentation/delta-Q is not wired yet";
      return ParseStatus::kError;
    }
  }
  if (options.still_sb128 || options.still_sb256) {
    if (options.still_sb128 && options.still_sb256) {
      error = "--still-sb128 and --still-sb256 are mutually exclusive";
      return ParseStatus::kError;
    }
    if (!options.m1 && !options.still_gpu) {
      error = "--still-sb128/--still-sb256 require --still";
      return ParseStatus::kError;
    }
    if (options.still_single) {
      error = "--still-single cannot be combined with a 128x128 superblock";
      return ParseStatus::kError;
    }
    if (options.still_depth == 0) {
      error = "--still-depth 0 cannot be combined with a 128x128 superblock "
              "(the 128 root must split)";
      return ParseStatus::kError;
    }
  }
  if (options.video_sb128 || options.video_sb256) {
    if (options.video_sb128 && options.video_sb256) {
      error = "--video-sb128 and --video-sb256 are mutually exclusive";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-sb128/--video-sb256 are CPU-only; --video-gpu keeps "
              "64x64 superblocks";
      return ParseStatus::kError;
    }
    if (options.svc_given) {
      error = "--video-sb128/--video-sb256 do not compose with SVC layers";
      return ParseStatus::kError;
    }
  }
  if (options.video_ext_partitions) {
    if (options.video_gpu) {
      error = "--video-ext-partitions is CPU-only; --video-gpu keeps the "
              "base partition set";
      return ParseStatus::kError;
    }
    if (options.svc_given) {
      error = "--video-ext-partitions does not compose with SVC layers";
      return ParseStatus::kError;
    }
    if (options.video_sb256) {
      error = "--video-ext-partitions does not compose with "
              "--video-sb256 (use --video-sb128)";
      return ParseStatus::kError;
    }
  }
  if (flag_given(options, "--video-partition-rd-lambda") &&
      !options.video_partition_rd) {
    error = "--video-partition-rd-lambda requires --video-partition-rd";
    return ParseStatus::kError;
  }
  if (flag_given(options, "--video-partition-rd-depth") &&
      !options.video_partition_rd) {
    error = "--video-partition-rd-depth requires --video-partition-rd";
    return ParseStatus::kError;
  }
  if (options.video_partition_rd) {
    if (!options.video) {
      error = "--video-partition-rd is the CPU --video partition decision; "
              "pass --video (the --still tree has its own cost)";
      return ParseStatus::kError;
    }
    if (options.video_gpu) {
      error = "--video-partition-rd is the CPU reference decision; --video-gpu "
              "has its own calibration";
      return ParseStatus::kError;
    }
    // --video-ext-partitions composes: the extended solver is priced by the
    // same leaf model as the rect tree (see InterFrameEncoder::decide) — the
    // variance + leaf-rate estimate for keyframes and the measured
    // NEWMV/residual probe for P frames.  The candidate set is the only
    // difference.
    if (options.video_sdp) {
      error = "--video-partition-rd does not compose with --video-sdp (the SDP "
              "tree has a separate cost model)";
      return ParseStatus::kError;
    }
    if (options.svc_given) {
      error = "--video-partition-rd does not compose with --svc-layers";
      return ParseStatus::kError;
    }
    if (options.video_lossless) {
      error = "--video-partition-rd needs the lossy quantiser; the lossless "
              "path codes a fixed block size";
      return ParseStatus::kError;
    }
    // The probe precomputes one frame-level pass on the worker pool and prices
    // inter leaves only, so it keeps the measured 8-bit 4:2:0 single-tile
    // low-delay P chain and the 64x64 superblock tree.
    if (options.bit_depth != 8 || options.chroma != "420") {
      error = "--video-partition-rd is wired for the 8-bit 4:2:0 CPU video "
              "core only";
      return ParseStatus::kError;
    }
    if (options.tiles != "1x1") {
      error = "--video-partition-rd supports one tile: the probe runs one "
              "frame-level pass and cannot nest the tiled worker pools";
      return ParseStatus::kError;
    }
    if (options.b_frames > 0 || options.video_tlayers > 0 ||
        options.video_tip || options.video_long_term || options.video_olk ||
        options.video_leading != 0 || options.video_sef ||
        options.video_switch || options.video_ras) {
      error = "--video-partition-rd does not compose with the B/temporal-"
              "layer/random-access/TIP frame schedules";
      return ParseStatus::kError;
    }
    if (options.video_sb128 || options.video_sb256) {
      error = "--video-partition-rd does not compose with the 128/256 "
              "superblock trees";
      return ParseStatus::kError;
    }
  }
  if (options.still_cdef_on_skip_txfm) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-cdef-on-skip-txfm requires --still";
      return ParseStatus::kError;
    }
    if (!options.still_cdef) {
      error = "--still-cdef-on-skip-txfm requires --still-cdef";
      return ParseStatus::kError;
    }
  }
  if (options.still_grain_adaptive) {
    if (!options.m1 && !options.still_gpu) {
      error = "--still-grain-adaptive requires --still";
      return ParseStatus::kError;
    }
    if (!options.still_grain) {
      error = "--still-grain-adaptive requires --still-grain";
      return ParseStatus::kError;
    }
  }
  return ParseStatus::kOk;
}
