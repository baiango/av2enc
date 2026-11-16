#include "pipeline/video/video_frame_internal.h"

namespace av2enc::video {
namespace vf_detail {
#if defined(AV2ENC_VIDEO_FRAMES)
// B frames (--b-frames N).  The schedule comes from
// src/codec/inter/bframes: kIpbb for N consecutive B frames (gop = N + 2), so
// coding order is I P B B ... while display order is I B B ... P.  Units are
// written in coding order; monotonic_output_order_flag = 0 plus
// implicit_output_frame makes the decoder emit display order.  With
// --video-compound the two-reference B frames also enable reference_select.
bool encode_b_sequence(std::span<const VideoInputFrame> frames,
                       const VideoOptions& options, int gop_size,
                       VideoSequenceResult& result, std::string& error) {
  namespace bf = inter::bframes;
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  const int num_gops = std::max(1, options.video_gops);
  const std::size_t expected =
      static_cast<std::size_t>(gop_size) * static_cast<std::size_t>(num_gops);
  if (frames.size() != expected) {
    error = "--b-frames codes " + std::to_string(num_gops) +
            " closed GOP(s): pass exactly " + std::to_string(expected) +
            " frames (N B frames + key + P anchor per GOP)";
    return false;
  }
  bf::SequenceConfig sc;
  sc.num_gops = num_gops;
  sc.gop_size = gop_size;
  sc.pattern = bf::GopPattern::kIpbb;
  // OrderHintBits must cover the whole sequence: two 8-frame GOPs need 16
  // distinct hints, so the historical 4 bits stay exact until 16 pictures.
  int order_hint_bits = 4;
  while ((1 << order_hint_bits) < static_cast<int>(frames.size())) {
    ++order_hint_bits;
  }
  sc.order_hint_bits = order_hint_bits;
  sc.num_slots = 16;
  sc.base_q_idx = options.base_q_idx;
  sc.leading_pictures = 0;
  bf::Plan plan;
  bf::PlanError plan_error = bf::PlanError::kOk;
  if (!bf::build_sequence(sc, plan, &plan_error)) {
    error = std::string("B-frame schedule: ") + bf::plan_error_name(plan_error);
    return false;
  }
  if (plan.display_count() != static_cast<int>(frames.size())) {
    error = "B-frame schedule display count mismatch";
    return false;
  }
  // A CLK that starts a later GOP refreshes every slot.  Its output process
  // flushes the previous GOP's buffered pictures (implicit output only retains
  // pictures in the reference store), so every key is immediate_output.
  for (bf::FramePlan& fp : plan.frames) {
    if (fp.is_key) {
      fp.immediate_output_frame = true;
      fp.implicit_output_frame = false;
    }
  }
  bf::ValidationReport report;
  if (!bf::validate_plan(plan, &report)) {
    error = std::string("B-frame plan invalid: ") +
            bf::plan_error_name(report.error) + " (coding " +
            std::to_string(report.error_coding_index) + ", display " +
            std::to_string(report.error_display_index) + ")";
    return false;
  }

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = static_cast<std::uint8_t>(plan.order_hint_bits);
  config.num_ref_frames = static_cast<std::uint8_t>(plan.num_slots);
  config.explicit_ref_frame_map = !options.implicit_ref_map;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.minimal_tile_trailer = options.minimal_tile_trailer;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.enable_masked_compound =
      options.compound && (options.compound_wedge || options.compound_diffwtd);
  config.enable_cwp = options.enable_cwp_weighted;
  config.enable_imp_msk_bld = options.enable_imp_msk_bld;
  config.enable_df_sub_pu = options.enable_df_sub_pu;
  config.num_same_ref_compound = 0;
  config.enable_opfl_refine = options.compound ? options.opfl_refine_type : 0;
  config.enable_refinemv = options.compound && options.enable_refinemv;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.enable_mhccp = options.enable_mhccp;
  config.enable_ibp = options.enable_ibp;
  config.enable_idtx_intra = options.enable_idtx_intra;
  config.enable_mrls = options.enable_mrl;
  config.enable_dip = options.enable_dip;
  config.enable_intra_ist = options.enable_ist;
  config.enable_inter_ist = false;
  config.enable_tcq = options.enable_tcq;
  config.choose_tcq_per_frame = options.enable_tcq;
  config.enable_cctx = options.enable_cctx;
  config.enable_sdp = options.enable_sdp;
  config.enable_extended_sdp = false;
  config.enable_inter_ddt = options.enable_inter_ddt;
  config.using_qmatrix = options.enable_qm;
  config.qm_user_data = options.user_qm;
  config.enable_deblock = options.enable_deblock;
  apply_video_loop_filter_sequence_config(config, options.loop_filters);
  config.tx_mode_select = options.enable_tx_search;
  config.enable_bawp = options.enable_bawp;
  config.enable_adaptive_mvd = options.enable_adaptive_mvd;
  config.enable_mvd_sign_derive = options.enable_mvd_sign_derive;
  config.enable_flex_mvres = options.enable_flex_mvres;
  config.enable_ref_frame_mvs = options.enable_ref_frame_mvs;
  config.max_tlayer_id = 0;
  config.monotonic_output_order_flag = false;
  config.film_grain_params_present = options.film_grain;

  const int mi_cols = 2 * static_cast<int>((width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((height + 7U) >> 3U);
  std::vector<VideoReconFrame> recon_by_display(
      static_cast<std::size_t>(plan.display_count()));
  std::vector<SlotField> b_fields(static_cast<std::size_t>(plan.num_slots));
  std::vector<bool> b_is_key(static_cast<std::size_t>(plan.num_slots), false);
  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;
  // §5.14 film grain: one planner over the whole CVS.  Every picture of an
  // I/P/B schedule is output (immediate or implicit), so every frame header
  // codes film_grain_config() and its unit may carry one model OBU.
  grain::GrainStreamPlanner grain_planner;

  bool wrote_sequence_header = false;
  // §7.23 mirror for the implicit map.  This path signals obu_tlayer_id = 0
  // for every frame (max_tlayer_id = 0), so every RefTLayerId is 0 and the
  // default TLayerDependencyMap admits all of them.
  inter::mv_stack_full::RefSlotBank ref_bank;
  ref_bank.num_ref_frames = config.num_ref_frames;
  std::vector<int> slot_display(
      static_cast<std::size_t>(config.num_ref_frames), -1);
  const inter::mv_stack_full::DependencyMaps ref_deps =
      implicit_dependency_maps(config, 0);
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    const int display = plan.coding_order[ci];
    const bf::FramePlan* fp = plan.frame_at_display(display);
    if (fp == nullptr) {
      error = "B-frame plan is missing a display index";
      return false;
    }
    const bf::FrameHeaderFields hf = bf::header_fields(*fp);
    // Per-frame rate control: the planner is indexed by display order, so the
    // coding-order walk picks each picture's own base_q_idx.  Header and
    // coefficient-CDF qctx slice both follow it.
    const std::uint8_t base_q =
        options.frame_base_q_idx.empty()
            ? static_cast<std::uint8_t>(options.base_q_idx)
            : options.frame_base_q_idx[static_cast<std::size_t>(display)];
    frame_options.base_q_idx = base_q;
    VideoConfig frame_config = config;
    frame_config.base_q_idx = base_q;
    // Map slots (header ref_frame_idx) and the display picture each one holds.
    std::vector<int> map_slots;
    std::vector<int> ref_displays;
    std::array<int, 7> frame_distance{};
    std::array<int, 7> ref_order_hints{};
    if (options.implicit_ref_map && !fp->is_key) {
      inter::mv_stack_full::ImplicitRefMap implicit;
      if (!derive_implicit_fields(ref_bank, fp->display_order_hint,
                                  plan.order_hint_bits,
                                  static_cast<int>(width),
                                  static_cast<int>(height), 0, ref_deps,
                                  &implicit, &error)) {
        error = "B-frame display " + std::to_string(display) +
                " implicit ref map: " + error;
        return false;
      }
      const inter::mv_stack_full::RefMap& map = implicit.stage1;
      map_slots.reserve(static_cast<std::size_t>(map.num_total_refs));
      ref_displays.reserve(static_cast<std::size_t>(map.num_total_refs));
      for (int r = 0; r < map.num_total_refs; ++r) {
        const int slot = map.ref_frame_idx[r];
        if (slot < 0 || slot >= ref_bank.num_ref_frames ||
            slot_display[static_cast<std::size_t>(slot)] < 0) {
          error = "B-frame display " + std::to_string(display) +
                  " implicit map names a slot with no coded picture";
          return false;
        }
        map_slots.push_back(slot);
        ref_displays.push_back(slot_display[static_cast<std::size_t>(slot)]);
        frame_distance[static_cast<std::size_t>(r)] = map.scores_distance[r];
        ref_order_hints[static_cast<std::size_t>(r)] =
            ref_bank.slots[static_cast<std::size_t>(slot)].order_hint;
      }
    } else {
      for (int r = 0; r < hf.num_total_refs; ++r) {
        map_slots.push_back(hf.ref_frame_idx[r]);
        ref_displays.push_back(fp->ref_display[r]);
        frame_distance[static_cast<std::size_t>(r)] = static_cast<int>(
            relative_dist(static_cast<std::uint32_t>(fp->display_order_hint),
                          static_cast<std::uint32_t>(fp->ref_order_hint[r]),
                          plan.order_hint_bits));
        ref_order_hints[static_cast<std::size_t>(r)] = fp->ref_order_hint[r];
      }
    }
    const int num_refs = static_cast<int>(map_slots.size());
    if (options.trace && options.implicit_ref_map && !fp->is_key) {
      trace_implicit_map("B", static_cast<std::size_t>(display), num_refs,
                         map_slots);
    }
    std::vector<inter::mc::RefFrame> references;
    references.reserve(static_cast<std::size_t>(num_refs));
    for (int r = 0; r < num_refs; ++r) {
      const int ref_display = ref_displays[static_cast<std::size_t>(r)];
      if (ref_display < 0 ||
          recon_by_display[static_cast<std::size_t>(ref_display)].y.empty()) {
        error = "B-frame reference is not available";
        return false;
      }
      const VideoReconFrame& rf =
          recon_by_display[static_cast<std::size_t>(ref_display)];
      inter::mc::RefFrame ref;
      ref.plane[0] = inter::mc::PlaneView{rf.y.data(), static_cast<int>(width)};
      ref.plane[1] = inter::mc::PlaneView{
          rf.u.data(), static_cast<int>((width + 1U) / 2U)};
      ref.plane[2] = inter::mc::PlaneView{
          rf.v.data(), static_cast<int>((width + 1U) / 2U)};
      ref.mi_cols = mi_cols;
      ref.mi_rows = mi_rows;
      references.push_back(ref);
    }
    TemporalFrameContext temporal_ctx;
    MotionFieldFrame motion_field;
    if (options.enable_ref_frame_mvs && !fp->is_key) {
      SavedFieldRef ref_fields[kRefsPerFrame] = {};
      int ref_slots[kRefsPerFrame] = {};
      std::uint8_t ref_is_key[kRefsPerFrame] = {};
      for (int r = 0; r < num_refs; ++r) {
        const int slot = fp->ref_slot[r];
        if (slot < 0 || slot >= plan.num_slots) {
          error = "B-frame reference slot outside the encoder DPB";
          return false;
        }
        ref_fields[r] = make_saved_ref(b_fields[static_cast<std::size_t>(slot)]);
        ref_slots[r] = slot;
        ref_is_key[r] = b_is_key[static_cast<std::size_t>(slot)] ? 1U : 0U;
      }
      if (!estimate_motion_field(
              ref_fields, ref_slots, ref_order_hints.data(), ref_is_key,
              /*ref_valid=*/true, num_refs, fp->display_order_hint,
              static_cast<int>(width), static_cast<int>(height), mi_cols,
              mi_rows, /*proj_step=*/1, /*enable_mv_traj=*/false,
              &motion_field, error)) {
        error = "B-frame display " + std::to_string(display) + ": " + error;
        return false;
      }
      temporal_ctx.use_ref_frame_mvs = true;
      temporal_ctx.order_hint = fp->display_order_hint;
      temporal_ctx.num_past_refs = motion_field.config.num_past_refs;
      temporal_ctx.num_future_refs = motion_field.config.num_future_refs;
      temporal_ctx.closest_past = motion_field.result.closest_past;
      temporal_ctx.closest_future = motion_field.result.closest_future;
      temporal_ctx.field = &motion_field.field;
    }
    InterFrameEncoder encoder(frames[static_cast<std::size_t>(display)],
                              std::move(references), frame_distance,
                              std::max(1, num_refs), frame_options,
                              options.trace, fp->is_key, 0, 0, 0, 0, {}, nullptr,
                              nullptr, ref_order_hints,
                              /*primary_ref_is_none=*/true, &temporal_ctx);
    std::vector<std::uint8_t> tile;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "B-frame display " + std::to_string(display) + ": " + error;
      return false;
    }
    result.temporal_candidates +=
        static_cast<std::uint64_t>(encoder.temporal_added());
    result.temporal_selected +=
        static_cast<std::uint64_t>(encoder.temporal_selected());
    result.amvd_selected +=
        static_cast<std::uint64_t>(encoder.amvd_selected());
    result.flex_precision_selected +=
        static_cast<std::uint64_t>(encoder.flex_precision_selected());
    result.derived_sign_selected +=
        static_cast<std::uint64_t>(encoder.derived_sign_selected());

    VideoFrameHeader header;
    header.keyframe = hf.keyframe;
    header.order_hint = hf.order_hint_lsbs;
    header.num_total_refs = static_cast<std::uint8_t>(num_refs);
    for (int r = 0; r < num_refs; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(map_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    header.frame_explicit_ref_frame_map = !options.implicit_ref_map;
    header.refresh_frame_flags = hf.refresh_frame_flags;
    header.immediate_output_frame = hf.immediate_output_frame;
    header.implicit_output_frame = hf.implicit_output_frame;
    header.reference_select = hf.reference_select && options.compound &&
                              num_refs > 1;
    header.allow_df_sub_pu = options.enable_df_sub_pu && !hf.keyframe;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    header.use_ref_frame_mvs =
        options.enable_ref_frame_mvs && !hf.keyframe;
    apply_video_loop_filter_frame_header(encoder.filter_plan(), header);

    // §5.14: plan this picture's model in coding order before the frame header
    // is written; a changed model emits one film grain OBU in this unit.
    grain::GrainFramePlan grain_plan;
    if (options.film_grain &&
        !plan_grain_frame(frames[static_cast<std::size_t>(display)],
                          static_cast<std::uint32_t>(display), grain_planner,
                          grain_plan, error)) {
      error = "B-frame display " + std::to_string(display) + " film grain: " +
              error;
      return false;
    }
    if (options.film_grain) {
      header.apply_grain = grain_plan.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(grain_plan.fgm_id);
      header.grain_seed = grain_plan.grain_seed;
    }

    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const std::vector<std::uint8_t> obu =
        make_frame_obu(frame_config, header, tile, 0);
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain_obu(grain_plan, unit, error)) {
      error = "B-frame display " + std::to_string(display) + ": " + error;
      return false;
    }
    result.units.push_back(std::move(unit));

    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2), width, height);
    if (options.implicit_ref_map) {
      // §7.23: the next frame's derivation reads this frame's slot state.
      if (!ref_bank.refresh(hf.refresh_frame_flags, fp->display_order_hint,
                            static_cast<int>(width), static_cast<int>(height),
                            base_q, hf.keyframe, /*tlayer_id=*/0,
                            /*mlayer_id=*/0, hf.implicit_output_frame,
                            hf.immediate_output_frame, -1, &error)) {
        error = "B-frame display " + std::to_string(display) +
                " reference bank: " + error;
        return false;
      }
      int first_refreshed = -1;
      for (int slot = 0; slot < ref_bank.num_ref_frames; ++slot) {
        if (((hf.refresh_frame_flags >> slot) & 1U) != 0) {
          first_refreshed = slot;
          break;
        }
      }
      for (int slot = 0; slot < ref_bank.num_ref_frames; ++slot) {
        if (((hf.refresh_frame_flags >> slot) & 1U) == 0) continue;
        slot_display[static_cast<std::size_t>(slot)] =
            (hf.keyframe && slot != first_refreshed) ? -1 : display;
      }
    }


    // §7.22/§7.23: refresh the plan slot with this frame's motion field.
    if (options.enable_ref_frame_mvs) {
      if (fp->is_key) {
        // §7.23 refreshes every slot with the key frame (allFrames); its
        // field is empty for all of them.
        for (int slot = 0; slot < plan.num_slots; ++slot) {
          init_slot_field(&b_fields[static_cast<std::size_t>(slot)], mi_cols,
                          mi_rows);
          b_fields[static_cast<std::size_t>(slot)].order_hint =
              fp->display_order_hint;
          b_is_key[static_cast<std::size_t>(slot)] = true;
        }
      } else {
        const int slot = fp->slot;
        if (slot < 0 || slot >= plan.num_slots) {
          error = "B-frame plan slot outside the encoder DPB";
          return false;
        }
        SlotField& field = b_fields[static_cast<std::size_t>(slot)];
        RefMvsStoreStats store_stats;
        if (!store_slot_field(encoder, fp->display_order_hint,
                              ref_order_hints.data(), num_refs, &field,
                              error, &store_stats)) {
          error = "B-frame display " + std::to_string(display) + ": " + error;
          return false;
        }
        result.ref_mvs_warp_cells += store_stats.warp_cells;
        result.ref_mvs_wedge_dropped += store_stats.wedge_dropped;
        field.order_hint = fp->display_order_hint;
        field.num_saved_refs = num_refs;
        for (int r = 0; r < num_refs; ++r) {
          field.saved_order_hint[r] =
              ref_order_hints[static_cast<std::size_t>(r)];
        }
        b_is_key[static_cast<std::size_t>(slot)] = false;
      }
    }
  }

  std::vector<std::int32_t> rank(static_cast<std::size_t>(plan.display_count()),
                                 -1);
  for (int display = 0; display < plan.display_count(); ++display) {
    rank[static_cast<std::size_t>(display)] = display;
    result.recon.push_back(
        recon_by_display[static_cast<std::size_t>(display)]);
  }
  result.frame_output_index.resize(plan.coding_order.size());
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    result.frame_output_index[ci] =
        rank[static_cast<std::size_t>(plan.coding_order[ci])];
  }
  return true;
}

// Hierarchical-P temporal layers (--video-tlayers).  The plan is one closed
// GOP: frames are coded in plan.coding_order, each frame references exactly
// the previous coding-order frame (kHierarchicalP, single reference), and the
// frame OBU carries the plan's obu_tlayer_id.  The sequence header raises
// max_tlayer_id, clears monotonic_output_order_flag and (optionally) codes the
// TLayerDependencyMap; every frame after the first is an implicit-output
// picture, so the decoder emits display order even though units are written in
// coding order.
bool encode_temporal_sequence(std::span<const VideoInputFrame> frames,
                              const VideoOptions& options,
                              VideoSequenceResult& result, std::string& error) {
  namespace tl = inter::tlayers;
  const int gop = options.temporal_layers;
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  if (frames.size() != static_cast<std::size_t>(gop)) {
    error = "--video-tlayers codes one closed GOP: pass exactly " +
            std::to_string(gop) + " input frames";
    return false;
  }
  if (!options.frame_base_q_idx.empty() &&
      options.frame_base_q_idx.size() != frames.size()) {
    error = "per-frame rate control needs one base_q_idx per temporal-layer "
            "picture";
    return false;
  }

  tl::PlanConfig pc;
  pc.gop_size = gop;
  pc.schedule = tl::Schedule::kHierarchicalP;
  // Two references per interior frame (the base inter picture keeps the key
  // as its only reference).  The default TLayerDependencyMap admits both, and
  // the §7.22 fields then let §7.9 project real ref-MV candidates.
  pc.ref_mode = tl::RefMode::kTwo;
  pc.dependency = tl::DependencyMode::kAllLower;
  pc.explicit_dependency_map = false;  // pinned dav2d parses the default map
  // A 16-frame GOP needs OrderHintBits 5: the display hint of the anchor at
  // index 8 is a half-span away from the key's hint 0, which the §5.18.2
  // modulo reconstruction cannot resolve with 4 bits.
  pc.order_hint_bits = gop >= 16 ? 5 : 4;
  pc.num_slots = gop;
  pc.base_q_idx = options.base_q_idx;
  tl::Plan plan;
  tl::PlanError plan_error = tl::PlanError::kOk;
  if (!tl::build_plan(pc, plan, &plan_error)) {
    error = std::string("temporal-layer plan: ") +
            tl::plan_error_name(plan_error);
    return false;
  }
  tl::PlanError validation_error = tl::PlanError::kOk;
  int bad_display = -1;
  int bad_coding = -1;
  if (!tl::validate_plan(plan, &validation_error, &bad_display, &bad_coding)) {
    error = std::string("temporal-layer plan invalid: ") +
            tl::plan_error_name(validation_error) + " (display " +
            std::to_string(bad_display) + ", coding " +
            std::to_string(bad_coding) + ")";
    return false;
  }
  if (plan.display_count() != gop) {
    error = "temporal-layer plan display count mismatch";
    return false;
  }

  std::vector<int> coding = plan.coding_order;
  if (options.tlayer_drop >= 0) {
    if (options.tlayer_drop > plan.max_tlayer_id) {
      error = "--video-tlayer-drop is above the plan's max_tlayer_id";
      return false;
    }
    if (!tl::validate_layer_drop(plan, options.tlayer_drop, &plan_error)) {
      error = std::string("temporal-layer drop is not decodable: ") +
              tl::plan_error_name(plan_error);
      return false;
    }
    coding = plan.coding_order_up_to_layer(options.tlayer_drop);
    if (coding.empty()) {
      error = "--video-tlayer-drop removes every coded frame";
      return false;
    }
  }

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = static_cast<std::uint8_t>(pc.order_hint_bits);
  config.num_ref_frames = static_cast<std::uint8_t>(plan.num_slots);
  config.explicit_ref_frame_map = !options.implicit_ref_map;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.minimal_tile_trailer = options.minimal_tile_trailer;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.enable_mhccp = options.enable_mhccp;
  config.enable_ibp = options.enable_ibp;
  config.enable_idtx_intra = options.enable_idtx_intra;
  config.enable_mrls = options.enable_mrl;
  config.enable_dip = options.enable_dip;
  config.enable_intra_ist = options.enable_ist;
  config.enable_inter_ist = false;
  config.enable_tcq = options.enable_tcq;
  config.choose_tcq_per_frame = options.enable_tcq;
  config.enable_cctx = options.enable_cctx;
  config.enable_sdp = options.enable_sdp;
  config.enable_extended_sdp = false;
  config.enable_inter_ddt = options.enable_inter_ddt;
  config.using_qmatrix = options.enable_qm;
  config.qm_user_data = options.user_qm;
  config.enable_deblock = options.enable_deblock;
  apply_video_loop_filter_sequence_config(config, options.loop_filters);
  config.tx_mode_select = options.enable_tx_search;
  config.enable_bawp = options.enable_bawp;
  config.enable_adaptive_mvd = options.enable_adaptive_mvd;
  config.enable_mvd_sign_derive = options.enable_mvd_sign_derive;
  config.enable_flex_mvres = options.enable_flex_mvres;
  config.enable_ref_frame_mvs = options.enable_ref_frame_mvs;
  config.max_tlayer_id = static_cast<std::uint8_t>(plan.max_tlayer_id);
  config.monotonic_output_order_flag = false;
  // SVC layered encodes carry seq_lcr_id / max_mlayer_id from the layer plan;
  // the plain --video-tlayers path keeps the single-layer defaults (0/0/1) and
  // therefore stays byte-identical.
  config.seq_lcr_id = options.seq_lcr_id;
  config.seq_profile_override = options.seq_profile_override;
  config.max_mlayer_id = options.max_mlayer_id;
  config.seq_max_mlayer_cnt = options.seq_max_mlayer_cnt;
  config.tlayer_dependency_present_flag = plan.tlayer_dependency_present_flag;
  config.film_grain_params_present = options.film_grain;
  for (int curr = 0; curr <= plan.max_tlayer_id; ++curr) {
    for (int ref = 0; ref <= plan.max_tlayer_id; ++ref) {
      config.tlayer_dependency[curr][ref] =
          plan.tlayer_dependency[static_cast<std::size_t>(curr)]
                                [static_cast<std::size_t>(ref)];
    }
  }

  const int mi_cols = 2 * static_cast<int>((width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((height + 7U) >> 3U);
  std::vector<VideoReconFrame> recon_by_display(
      static_cast<std::size_t>(plan.display_count()));
  std::vector<SlotField> t_fields(static_cast<std::size_t>(plan.num_slots));
  std::vector<bool> t_is_key(static_cast<std::size_t>(plan.num_slots), false);

  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;

  // §5.14 film grain: every temporal-layer picture is output (immediate or
  // implicit), so every unit carries the §5.18.10.1 config and at most one
  // model OBU.
  grain::GrainStreamPlanner grain_planner;

  bool wrote_sequence_header = false;
  // §7.23 mirror for the implicit map; the OBU tlayer id is stored per slot so
  // the default TLayerDependencyMap (`ref_tlayer <= obu_tlayer_id`) applies.
  inter::mv_stack_full::RefSlotBank ref_bank;
  ref_bank.num_ref_frames = config.num_ref_frames;
  std::vector<int> slot_display(
      static_cast<std::size_t>(config.num_ref_frames), -1);
  for (std::size_t ci = 0; ci < coding.size(); ++ci) {
    const int display = coding[ci];
    const tl::PlannedFrame& pf =
        plan.frames[static_cast<std::size_t>(display)];
    if (pf.is_key && ci != 0) {
      error = "temporal-layer plan codes a key outside coding position 0";
      return false;
    }
    const int tlayer_id = pf.obu_tlayer_id();
    // Per-frame rate control follows display order (the planner's indexing),
    // while units are emitted in coding order.
    const std::uint8_t base_q =
        options.frame_base_q_idx.empty()
            ? static_cast<std::uint8_t>(options.base_q_idx)
            : options.frame_base_q_idx[static_cast<std::size_t>(display)];
    frame_options.base_q_idx = base_q;
    VideoConfig frame_config = config;
    frame_config.base_q_idx = base_q;

    std::vector<int> map_slots;
    std::vector<int> ref_displays;
    std::array<int, 7> frame_distance{};
    std::array<int, 7> ref_order_hints{};
    if (options.implicit_ref_map && !pf.is_key) {
      inter::mv_stack_full::ImplicitRefMap implicit;
      const inter::mv_stack_full::DependencyMaps deps =
          implicit_dependency_maps(config, tlayer_id);
      if (!derive_implicit_fields(ref_bank, pf.display_order_hint,
                                  static_cast<int>(pc.order_hint_bits),
                                  static_cast<int>(width),
                                  static_cast<int>(height), tlayer_id, deps,
                                  &implicit, &error)) {
        error = "temporal-layer frame " + std::to_string(display) +
                " implicit ref map: " + error;
        return false;
      }
      const inter::mv_stack_full::RefMap& map = implicit.stage1;
      map_slots.reserve(static_cast<std::size_t>(map.num_total_refs));
      ref_displays.reserve(static_cast<std::size_t>(map.num_total_refs));
      for (int r = 0; r < map.num_total_refs; ++r) {
        const int slot = map.ref_frame_idx[r];
        if (slot < 0 || slot >= ref_bank.num_ref_frames ||
            slot_display[static_cast<std::size_t>(slot)] < 0) {
          error = "temporal-layer frame " + std::to_string(display) +
                  " implicit map names a slot with no coded picture";
          return false;
        }
        map_slots.push_back(slot);
        ref_displays.push_back(slot_display[static_cast<std::size_t>(slot)]);
        frame_distance[static_cast<std::size_t>(r)] = map.scores_distance[r];
        ref_order_hints[static_cast<std::size_t>(r)] =
            ref_bank.slots[static_cast<std::size_t>(slot)].order_hint;
      }
    } else {
      for (int r = 0; r < pf.num_refs; ++r) {
        map_slots.push_back(pf.ref_slot[r]);
        ref_displays.push_back(pf.ref_display[r]);
        frame_distance[static_cast<std::size_t>(r)] =
            static_cast<int>(relative_dist(pf.display_order_hint,
                                           pf.ref_order_hint[r],
                                           static_cast<int>(pc.order_hint_bits)));
        ref_order_hints[static_cast<std::size_t>(r)] = pf.ref_order_hint[r];
      }
    }
    const int num_refs = static_cast<int>(map_slots.size());
    if (options.trace && options.implicit_ref_map && !pf.is_key) {
      trace_implicit_map("tl", static_cast<std::size_t>(display), num_refs,
                         map_slots);
    }

    std::vector<inter::mc::RefFrame> references;
    references.reserve(static_cast<std::size_t>(num_refs));
    for (int r = 0; r < num_refs; ++r) {
      const int ref_display = ref_displays[static_cast<std::size_t>(r)];
      if (ref_display < 0 ||
          recon_by_display[static_cast<std::size_t>(ref_display)].y.empty()) {
        error = "temporal-layer reference is not available";
        return false;
      }
      const VideoReconFrame& rf =
          recon_by_display[static_cast<std::size_t>(ref_display)];
      inter::mc::RefFrame ref;
      ref.plane[0] = inter::mc::PlaneView{rf.y.data(), static_cast<int>(width)};
      ref.plane[1] = inter::mc::PlaneView{
          rf.u.data(), static_cast<int>((width + 1U) / 2U)};
      ref.plane[2] = inter::mc::PlaneView{
          rf.v.data(), static_cast<int>((width + 1U) / 2U)};
      ref.mi_cols = mi_cols;
      ref.mi_rows = mi_rows;
      references.push_back(ref);
    }

    TemporalFrameContext temporal_ctx;
    MotionFieldFrame motion_field;
    if (options.enable_ref_frame_mvs && !pf.is_key) {
      SavedFieldRef ref_fields[kRefsPerFrame] = {};
      int ref_slots[kRefsPerFrame] = {};
      std::uint8_t ref_is_key[kRefsPerFrame] = {};
      for (int r = 0; r < pf.num_refs; ++r) {
        const int slot = pf.ref_slot[r];
        if (slot < 0 || slot >= plan.num_slots) {
          error = "temporal-layer reference slot outside the encoder DPB";
          return false;
        }
        ref_fields[r] = make_saved_ref(t_fields[static_cast<std::size_t>(slot)]);
        ref_slots[r] = slot;
        ref_is_key[r] = t_is_key[static_cast<std::size_t>(slot)] ? 1U : 0U;
      }
      if (!estimate_motion_field(
              ref_fields, ref_slots, ref_order_hints.data(), ref_is_key,
              /*ref_valid=*/true, pf.num_refs, pf.display_order_hint,
              static_cast<int>(width), static_cast<int>(height), mi_cols,
              mi_rows, /*proj_step=*/1, /*enable_mv_traj=*/false,
              &motion_field, error)) {
        error = "temporal-layer frame " + std::to_string(display) + ": " +
                error;
        return false;
      }
      temporal_ctx.use_ref_frame_mvs = true;
      temporal_ctx.order_hint = static_cast<int>(pf.display_order_hint);
      temporal_ctx.num_past_refs = motion_field.config.num_past_refs;
      temporal_ctx.num_future_refs = motion_field.config.num_future_refs;
      temporal_ctx.closest_past = motion_field.result.closest_past;
      temporal_ctx.closest_future = motion_field.result.closest_future;
      temporal_ctx.field = &motion_field.field;
    }
    InterFrameEncoder encoder(frames[static_cast<std::size_t>(display)],
                              std::move(references), frame_distance,
                              std::max(1, num_refs), frame_options,
                              options.trace, pf.is_key, 0, 0, 0, 0, {}, nullptr,
                              nullptr, ref_order_hints,
                              /*primary_ref_is_none=*/true, &temporal_ctx);
    std::vector<std::uint8_t> tile;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "temporal-layer frame " + std::to_string(display) + ": " + error;
      return false;
    }
    result.temporal_candidates +=
        static_cast<std::uint64_t>(encoder.temporal_added());
    result.temporal_selected +=
        static_cast<std::uint64_t>(encoder.temporal_selected());
    result.amvd_selected +=
        static_cast<std::uint64_t>(encoder.amvd_selected());
    result.flex_precision_selected +=
        static_cast<std::uint64_t>(encoder.flex_precision_selected());
    result.derived_sign_selected +=
        static_cast<std::uint64_t>(encoder.derived_sign_selected());

    VideoFrameHeader header;
    header.keyframe = pf.is_key;
    header.order_hint = pf.order_hint_lsbs;
    header.num_total_refs = static_cast<std::uint8_t>(num_refs);
    for (int r = 0; r < num_refs; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(map_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    header.frame_explicit_ref_frame_map = !options.implicit_ref_map;
    header.refresh_frame_flags = pf.refresh_frame_flags;
    header.immediate_output_frame = pf.immediate_output_frame;
    header.implicit_output_frame = pf.implicit_output_frame;
    // reference_select stays 0: the block layer uses one reference per block;
    // the two-entry map is what feeds the §7.9 field.
    header.reference_select = false;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    header.use_ref_frame_mvs =
        options.enable_ref_frame_mvs && !pf.is_key;
    apply_video_loop_filter_frame_header(encoder.filter_plan(), header);

    // §5.14: plan in coding order before the frame header; the model OBU is
    // inserted into this layer's temporal unit.
    grain::GrainFramePlan grain_plan;
    if (options.film_grain &&
        !plan_grain_frame(frames[static_cast<std::size_t>(display)],
                          static_cast<std::uint32_t>(display), grain_planner,
                          grain_plan, error, tlayer_id, /*mlayer_id=*/0)) {
      error = "temporal-layer frame " + std::to_string(display) +
              " film grain: " + error;
      return false;
    }
    if (options.film_grain) {
      header.apply_grain = grain_plan.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(grain_plan.fgm_id);
      header.grain_seed = grain_plan.grain_seed;
    }

    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const std::vector<std::uint8_t> obu =
        make_frame_obu(frame_config, header, tile,
                       static_cast<std::uint8_t>(pf.obu_tlayer_id()));
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain_obu(grain_plan, unit, error)) {
      error = "temporal-layer frame " + std::to_string(display) + ": " +
              error;
      return false;
    }
    result.units.push_back(std::move(unit));

    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2), width, height);
    if (options.implicit_ref_map) {
      if (!ref_bank.refresh(pf.refresh_frame_flags, pf.display_order_hint,
                            static_cast<int>(width), static_cast<int>(height),
                            base_q, pf.is_key, tlayer_id,
                            /*mlayer_id=*/0, pf.implicit_output_frame,
                            pf.immediate_output_frame, -1, &error)) {
        error = "temporal-layer frame " + std::to_string(display) +
                " reference bank: " + error;
        return false;
      }
      int first_refreshed = -1;
      for (int slot = 0; slot < ref_bank.num_ref_frames; ++slot) {
        if (((pf.refresh_frame_flags >> slot) & 1U) != 0) {
          first_refreshed = slot;
          break;
        }
      }
      for (int slot = 0; slot < ref_bank.num_ref_frames; ++slot) {
        if (((pf.refresh_frame_flags >> slot) & 1U) == 0) continue;
        slot_display[static_cast<std::size_t>(slot)] =
            (pf.is_key && slot != first_refreshed) ? -1 : display;
      }
    }


    // §7.22/§7.23: refresh the plan slot with this frame's motion field.
    if (options.enable_ref_frame_mvs) {
      if (pf.is_key) {
        for (int slot = 0; slot < plan.num_slots; ++slot) {
          init_slot_field(&t_fields[static_cast<std::size_t>(slot)], mi_cols,
                          mi_rows);
          t_fields[static_cast<std::size_t>(slot)].order_hint =
              static_cast<int>(pf.display_order_hint);
          t_is_key[static_cast<std::size_t>(slot)] = true;
        }
      } else if (pf.slot >= 0 && pf.slot < plan.num_slots) {
        SlotField& field = t_fields[static_cast<std::size_t>(pf.slot)];
        RefMvsStoreStats store_stats;
        if (!store_slot_field(encoder, static_cast<int>(pf.display_order_hint),
                              ref_order_hints.data(), pf.num_refs, &field,
                              error, &store_stats)) {
          error = "temporal-layer frame " + std::to_string(display) + ": " +
                  error;
          return false;
        }
        result.ref_mvs_warp_cells += store_stats.warp_cells;
        result.ref_mvs_wedge_dropped += store_stats.wedge_dropped;
        field.order_hint = static_cast<int>(pf.display_order_hint);
        field.num_saved_refs = pf.num_refs;
        for (int r = 0; r < pf.num_refs; ++r) {
          field.saved_order_hint[r] =
              ref_order_hints[static_cast<std::size_t>(r)];
        }
        t_is_key[static_cast<std::size_t>(pf.slot)] = false;
      }
    }
  }

  // Decoder output: display order over the retained frames.
  std::vector<int> retained(coding);
  std::sort(retained.begin(), retained.end());
  std::vector<std::int32_t> rank(static_cast<std::size_t>(plan.display_count()),
                                 -1);
  for (std::size_t i = 0; i < retained.size(); ++i) {
    const int display = retained[i];
    rank[static_cast<std::size_t>(display)] = static_cast<std::int32_t>(i);
    result.recon.push_back(
        recon_by_display[static_cast<std::size_t>(display)]);
  }
  result.frame_output_index.resize(coding.size());
  for (std::size_t ci = 0; ci < coding.size(); ++ci) {
    result.frame_output_index[ci] = rank[static_cast<std::size_t>(coding[ci])];
  }
  return true;
}
#endif  // AV2ENC_VIDEO_FRAMES
}  // namespace vf_detail


// ---------------------------------------------------------------------------
// Cross-embedded-layer refinement sequence (SVC --svc-embedded).
//
// Every frame is a regular inter picture that predicts from the lower
// embedded layer's decoded picture at the same display index.  §7.3.6 only
// permits the CLK in the lowest embedded layer of a temporal unit, so the
// first refinement frame is inter; the caller emits every lower mlayer of the
// same temporal unit first, which makes the reference slot valid (§5.18.2
// reference conformance, §7.23 slot refresh).  The frame header references
// `reference_slot` and refreshes `refresh_slot`; the lower layer keeps its own
// slot, so the next temporal unit's refinement frame still predicts from the
// lower layer, not from the previous refinement picture.
// ---------------------------------------------------------------------------
bool encode_video_refinement_sequence(
    std::span<const VideoInputFrame> frames,
    std::span<const VideoReconFrame> references, const VideoOptions& options,
    const RefinementOptions& refinement, VideoSequenceResult& result,
    std::string& error) {
  if (frames.empty()) {
    error = "refinement sequence has no input frames";
    return false;
  }
  if (references.size() != frames.size()) {
    error = "refinement sequence needs one reference picture per frame";
    return false;
  }
  if (refinement.reference_slot < 0 || refinement.refresh_slot < 0 ||
      refinement.reference_slot >= refinement.num_ref_frames ||
      refinement.refresh_slot >= refinement.num_ref_frames ||
      refinement.num_ref_frames > kMaxNumRefFrames) {
    error = "refinement slots are outside NumRefFrames";
    return false;
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  if (width == 0 || height == 0) {
    error = "refinement sequence needs a non-empty frame size";
    return false;
  }
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames[i].width != width || frames[i].height != height) {
      error = "refinement sequence frame dimensions differ";
      return false;
    }
    if (references[i].width != width || references[i].height != height ||
        references[i].y.empty() || references[i].u.empty() ||
        references[i].v.empty()) {
      error = "refinement reference " + std::to_string(i) +
              " has the wrong size or is not reconstructed";
      return false;
    }
  }
  if (options.enable_deblock || options.loop_filters.any() ||
      options.compound || options.enable_gme || options.enable_warp ||
      options.enable_tip || options.b_frames > 0) {
    // Deblock/loop filters need a picture-global filter plan per frame, and
    // the frame-type tools own their own reference schedules.  The SVC path
    // never sets them; reject instead of emitting an unfiltered picture.
    error = "embedded refinement does not compose with loop filters, "
            "compound prediction, global motion or the B/TIP frame types";
    return false;
  }

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = static_cast<std::uint8_t>(refinement.num_ref_frames);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.enable_mhccp = options.enable_mhccp;
  config.enable_ibp = options.enable_ibp;
  config.enable_tcq = options.enable_tcq;
  config.choose_tcq_per_frame = options.enable_tcq;
  config.enable_cctx = options.enable_cctx;
  config.using_qmatrix = options.enable_qm;
  config.enable_deblock = options.enable_deblock;
  config.enable_bawp = options.enable_bawp;
  config.enable_adaptive_mvd = options.enable_adaptive_mvd;
  config.enable_mvd_sign_derive = options.enable_mvd_sign_derive;
  config.enable_flex_mvres = options.enable_flex_mvres;
  config.tx_mode_select = options.enable_tx_search;
  config.enable_global_motion = false;
  // §5.4.1 layer limits from the SVC plan.  max_mlayer_id > 0 selects the
  // §5.4.1 dependency-map rows and the SeqMaxMlayerCnt field; the default
  // map (ref <= curr) is exactly what the cross-mlayer reference needs.
  config.seq_lcr_id = options.seq_lcr_id;
  config.seq_profile_override = options.seq_profile_override;
  config.max_tlayer_id = options.max_tlayer_id;
  config.max_mlayer_id = options.max_mlayer_id;
  config.seq_max_mlayer_cnt = options.seq_max_mlayer_cnt;
  config.mlayer_dependency_present = true;

  const int mi_cols = 2 * static_cast<int>((width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((height + 7U) >> 3U);

  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;
  frame_options.force_intra = false;

  if (!options.frame_base_q_idx.empty() &&
      options.frame_base_q_idx.size() != frames.size()) {
    error = "embedded refinement needs one base_q_idx per picture";
    return false;
  }
  bool wrote_sequence_header = false;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const std::uint8_t base_q =
        options.frame_base_q_idx.empty()
            ? static_cast<std::uint8_t>(options.base_q_idx)
            : options.frame_base_q_idx[i];
    frame_options.base_q_idx = base_q;
    VideoConfig frame_config = config;
    frame_config.base_q_idx = base_q;
    std::vector<inter::mc::RefFrame> refs;
    inter::mc::RefFrame ref;
    const VideoReconFrame& reference = references[i];
    ref.plane[0] = inter::mc::PlaneView{
        reference.y.data(), static_cast<int>(width)};
    ref.plane[1] = inter::mc::PlaneView{
        reference.u.data(), static_cast<int>((width + 1U) / 2U)};
    ref.plane[2] = inter::mc::PlaneView{
        reference.v.data(), static_cast<int>((width + 1U) / 2U)};
    ref.mi_cols = mi_cols;
    ref.mi_rows = mi_rows;
    refs.push_back(ref);

    std::array<int, 7> frame_distance{};
    frame_distance[0] = 0;  // same display time as the lower embedded layer
    std::array<int, 7> ref_order_hints{};
    ref_order_hints[0] = static_cast<int>(i);

    InterFrameEncoder encoder(frames[i], std::move(refs), frame_distance, 1,
                              frame_options, options.trace,
                              /*keyframe=*/false, 0, 0, 0, 0, {}, nullptr,
                              nullptr, ref_order_hints);
    std::vector<std::uint8_t> tile;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "refinement frame " + std::to_string(i) + ": " + error;
      return false;
    }

    VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = static_cast<std::uint32_t>(i);
    header.num_total_refs = 1;
    header.ref_frame_idx[0] = static_cast<std::uint8_t>(refinement.reference_slot);
    header.ref_slot = static_cast<std::uint8_t>(refinement.reference_slot);
    header.refresh_frame_flags =
        static_cast<std::uint8_t>(1U << refinement.refresh_slot);
    header.immediate_output_frame = true;
    header.implicit_output_frame = false;
    header.reference_select = false;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;

    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const std::vector<std::uint8_t> obu =
        make_frame_obu(frame_config, header, tile);
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.recon.push_back(to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2), width, height));
  }
  return true;
}

int resolved_num_ref_frames(const VideoOptions& options) {
  if (options.num_ref_frames != 0) return options.num_ref_frames;
  // The implicit map ranks every distinct slot up to REFS_PER_FRAME, so the
  // 2-slot historical default would cap it at two; give it the full pool.
  if (options.implicit_ref_map) return kMaxRefSlots;
  return options.num_total_refs > 1 ? kMaxRefSlots : 2;
}

// ---------------------------------------------------------------------------
// Opt-in TIP sequence (--video-tip, §7.10 / §7.13.3.2).
//
// Coding order is key, one future anchor P, then one TIP_FRAME_AS_OUTPUT per
// remaining input picture:
//
//   key    hint 0            slot 0   (intra)
//   anchor hint N-1          slot 1   (regular P referencing the key)
//   tip_i  hint i, i=1..N-2  slot i+1 (past = key or the previous TIP,
//                                      future = the anchor)
//
// The future anchor is decoded before the TIP frames, which is what makes
// §7.10 possible.  Every frame is immediate-output and the sequence keeps
// monotonic_output_order_flag = 1, so the decoder emits pictures in decode
// order (proven by src/codec/inter/tip/gate).  Each reference frame's §7.22
// field is stored from the real coded MV grid, and each TIP frame stores its
// own field so the next TIP in the chain can project it.
// ---------------------------------------------------------------------------
namespace vf_detail {
struct TipSlotState {
  bool valid = false;
  int order_hint = 0;
  int num_saved_refs = 0;
  int saved_order_hint[7] = {temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint};
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int8_t> field_ref;
  std::vector<std::int16_t> field_mv;
  Plane plane[3];

  temporal::SavedMvField saved() const {
    temporal::SavedMvField out;
    out.w8 = w8;
    out.h8 = h8;
    out.ref = field_ref.data();
    out.mv = field_mv.data();
    return out;
  }
};

// Allocates an empty §7.22 field (intra key / no coded inter blocks).
void init_empty_field(TipSlotState* slot, int mi_cols, int mi_rows) {
  slot->w8 = mi_cols / 2;
  slot->h8 = mi_rows / 2;
  slot->num_saved_refs = 0;
  slot->field_ref.assign(static_cast<std::size_t>(slot->w8) *
                             static_cast<std::size_t>(slot->h8) * 2U,
                         static_cast<std::int8_t>(temporal::kRefNone));
  slot->field_mv.assign(static_cast<std::size_t>(slot->w8) *
                            static_cast<std::size_t>(slot->h8) * 4U,
                        0);
}

// Stores one coded frame's §7.22 field from its per-4x4 mode/MV grid.
bool store_coded_field(const InterFrameEncoder& encoder, int order_hint,
                       const int* ref_order_hint, int num_refs,
                       TipSlotState* slot, std::string& error) {
  init_empty_field(slot, encoder.mv_mi_cols(), encoder.mv_mi_rows());
  temporal::RefMvField view;
  view.w8 = slot->w8;
  view.h8 = slot->h8;
  view.ref = slot->field_ref.data();
  view.mv = slot->field_mv.data();
  return store_field_grid(encoder, order_hint, ref_order_hint, num_refs, &view,
                          error);
}

// Picks the Tip_Weighting_Factor index whose factor is closest to the frame's
// temporal position between the past and future references.  Index 2 (16, the
// single-reference weight) is skipped: a two-sided TIP frame needs both lists.
int choose_tip_weight_index(int past_hint, int future_hint, int hint) {
  const int span = future_hint - past_hint;
  const int ideal =
      span > 0 ? (16 * (future_hint - hint) + span / 2) / span : 8;
  static constexpr int kIndices[] = {0, 1, 3, 4, 5, 6};
  int best = 0;
  int best_diff = 0x7fffffff;
  for (const int index : kIndices) {
    const int diff = std::abs(temporal::kTipWeightingFactor[index] - ideal);
    if (diff < best_diff) {
      best_diff = diff;
      best = index;
    }
  }
  return best;
}

bool encode_tip_sequence(std::span<const VideoInputFrame> frames,
                         const VideoOptions& options,
                         VideoSequenceResult& result, std::string& error) {
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  const int frame_count = static_cast<int>(frames.size());
  constexpr int kOrderHintBits = 4;
  const int num_ref_frames = kMaxRefSlots;  // 8 slots: key + anchor + TIPs
  const int recon_mi_cols = static_cast<int>(2U * ((width + 7U) >> 3U));
  const int recon_mi_rows = static_cast<int>(2U * ((height + 7U) >> 3U));

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = static_cast<std::uint8_t>(kOrderHintBits);
  config.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  config.explicit_ref_frame_map = !options.implicit_ref_map;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_ref_frame_mvs = true;
  config.enable_tip = true;
  config.disable_tip_output = false;
  // §5.4.6 enable_tip_hole_fill.  The sequence bit and the per-frame
  // allow_tip_hole_fill bit must agree with the §7.10.3/.4 setup the encoder
  // runs for this frame: signalling 0 makes the decoder predict a hole with
  // MV 0 while the encoder uses the neighbour-filled MV (sparse ±4 samples on
  // TIP frames).  Both masks are cell-identical on a coded anchor field when
  // the frame bit is written, so the product stream enables it.  See
  // docs/tip-decoder-gate.md.
  config.enable_tip_hole_fill = options.tip_hole_fill;
  config.enable_cfl_intra = true;
  config.enable_tcq = options.enable_tcq;
  config.choose_tcq_per_frame = options.enable_tcq;
  config.enable_cctx = options.enable_cctx;
  config.disable_drl_reorder = true;
  config.film_grain_params_present = options.film_grain;

  std::array<TipSlotState, kMaxRefSlots> slots;

  // §5.14 film grain: the key and the tile-coded P frames of this schedule are
  // ordinary frame headers and code film_grain_config().  A TIP OBU ends its
  // header at decode_frame_wrapup() (§5.18.2), so a TIP picture has no config
  // site: it is planned with apply_grain = 0 and carries no model OBU.
  grain::GrainStreamPlanner grain_planner;
  grain::GrainFramePlan key_grain;

  // --- Key frame -----------------------------------------------------------
  VideoFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.ref_slot = 0;
  key_header.refresh_frame_flags = (1U << num_ref_frames) - 1U;
  if (options.film_grain &&
      !plan_grain_frame(frames[0], 0, grain_planner, key_grain, error)) {
    error = "TIP keyframe film grain: " + error;
    return false;
  }
  if (options.film_grain) {
    key_header.apply_grain = key_grain.apply_grain;
    key_header.fgm_id = static_cast<std::uint8_t>(key_grain.fgm_id);
    key_header.grain_seed = key_grain.grain_seed;
  }

  std::vector<inter::mc::RefFrame> no_refs;
  std::array<int, 7> no_distance{};
  std::vector<std::uint8_t> key_payload;
  std::vector<syntax::BlockSyntax> key_blocks;
  std::vector<std::int32_t> key_levels;
  InterFrameEncoder key_encoder(frames[0], no_refs, no_distance, 1, options,
                                options.trace, /*keyframe=*/true);
  if (!key_encoder.encode(key_payload, key_blocks, key_levels, error)) {
    error = "TIP keyframe encode failed: " + error;
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) {
    slots[0].plane[plane] = key_encoder.recon(plane);
  }
  slots[0].valid = true;
  slots[0].order_hint = 0;
  init_empty_field(&slots[0], recon_mi_cols, recon_mi_rows);

  {
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
    const std::vector<std::uint8_t> obu =
        make_frame_obu(config, key_header, key_payload);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + seq.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq.begin(), seq.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain_obu(key_grain, unit, error)) {
      error = "TIP keyframe: " + error;
      return false;
    }
    result.units.push_back(std::move(unit));
    result.recon.push_back(to_recon_frame(slots[0].plane[0], slots[0].plane[1],
                                         slots[0].plane[2], width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(
        static_cast<std::uint32_t>(key_payload.size()));
    result.inter_blocks.push_back(0);
    result.intra_blocks.push_back(key_encoder.intra_blocks());
    result.frame_output_index.push_back(0);
  }

  // --- Future anchor P frame (hint N-1, slot 1) ----------------------------
  const int anchor_hint = frame_count - 1;
  {
    std::vector<inter::mc::RefFrame> references(1);
    references[0].plane[0] = inter::mc::PlaneView{
        slots[0].plane[0].samples.data(), slots[0].plane[0].stride};
    references[0].plane[1] = inter::mc::PlaneView{
        slots[0].plane[1].samples.data(), slots[0].plane[1].stride};
    references[0].plane[2] = inter::mc::PlaneView{
        slots[0].plane[2].samples.data(), slots[0].plane[2].stride};
    references[0].mi_cols = recon_mi_cols;
    references[0].mi_rows = recon_mi_rows;
    std::array<int, 7> distance{};
    distance[0] = relative_dist(static_cast<std::uint32_t>(anchor_hint), 0,
                                kOrderHintBits);
    std::array<int, 7> ref_order_hints{};
    ref_order_hints[0] = 0;

    std::vector<std::uint8_t> payload;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    InterFrameEncoder encoder(frames.back(), references, distance, 1, options,
                              options.trace, /*keyframe=*/false, 0, 0, 0, 0,
                              {}, nullptr, nullptr, ref_order_hints);
    if (!encoder.encode(payload, blocks, levels, error)) {
      error = "TIP anchor P encode failed: " + error;
      return false;
    }
    TipSlotState& slot = slots[1];
    for (int plane = 0; plane < 3; ++plane) {
      slot.plane[plane] = encoder.recon(plane);
    }
    slot.valid = true;
    slot.order_hint = anchor_hint;
    const int ref_hints[1] = {0};
    if (!store_coded_field(encoder, anchor_hint, ref_hints, 1, &slot, error)) {
      return false;
    }
    slot.num_saved_refs = 1;
    slot.saved_order_hint[0] = 0;

    VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = static_cast<std::uint32_t>(anchor_hint);
    header.num_total_refs = 1;
    header.ref_frame_idx[0] = 0;
    header.ref_slot = 0;
    header.refresh_frame_flags = 1U << 1;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    header.use_ref_frame_mvs = options.enable_ref_frame_mvs;

    grain::GrainFramePlan anchor_grain;
    if (options.film_grain &&
        !plan_grain_frame(frames.back(),
                          static_cast<std::uint32_t>(anchor_hint),
                          grain_planner, anchor_grain, error)) {
      error = "TIP anchor film grain: " + error;
      return false;
    }
    if (options.film_grain) {
      header.apply_grain = anchor_grain.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(anchor_grain.fgm_id);
      header.grain_seed = anchor_grain.grain_seed;
    }

    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu =
        make_frame_obu(config, header, payload);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain_obu(anchor_grain, unit, error)) {
      error = "TIP anchor: " + error;
      return false;
    }
    result.units.push_back(std::move(unit));
    result.recon.push_back(to_recon_frame(slot.plane[0], slot.plane[1],
                                         slot.plane[2], width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(payload.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    result.frame_output_index.push_back(1);
  }

  // --- TIP_FRAME_AS_OUTPUT frames -----------------------------------------
  // With --video-ref-mvs the last display picture is coded as a tile-coded P
  // that references the final TIP display frame, so the TIP slot's §7.22 field
  // feeds §7.9 estimation in a regular inter frame.
  const bool ref_mvs_tail = options.enable_ref_frame_mvs;
  const int tip_limit = ref_mvs_tail ? frame_count - 2 : frame_count - 1;
  std::vector<std::int16_t> pred_scratch(2U * 16U * 16U, 0);
  std::vector<std::uint8_t> block_scratch(16U * 16U, 0);
  for (int i = 1; i < tip_limit; ++i) {
    const int past_slot = i == 1 ? 0 : i;          // key, then previous TIP
    const int future_slot = 1;                     // the anchor
    TipSlotState& past = slots[static_cast<std::size_t>(past_slot)];
    TipSlotState& future = slots[future_slot];
    const int hint = i;

    temporal::MotionFieldConfig mf;
    mf.mi_cols = recon_mi_cols;
    mf.mi_rows = recon_mi_rows;
    mf.proj_step = 1;
    mf.sb_64x64 = true;
    mf.enable_tip = true;
    mf.order_hint = hint;
    mf.num_total_refs = 2;
    mf.num_past_refs = 1;
    mf.num_future_refs = 1;
    mf.frame_width = static_cast<int>(width);
    mf.frame_height = static_cast<int>(height);
    const int ref_slots[2] = {past_slot, future_slot};
    for (int rf = 0; rf < 2; ++rf) {
      const TipSlotState& ref = slots[static_cast<std::size_t>(ref_slots[rf])];
      mf.ref_order_hint[rf] = ref.order_hint;
      mf.refs[rf].slot = ref_slots[rf];
      mf.refs[rf].type = (ref_slots[rf] == 0) ? temporal::FrameType::kKey
                                              : temporal::FrameType::kInter;
      mf.refs[rf].width = static_cast<int>(width);
      mf.refs[rf].height = static_cast<int>(height);
      mf.refs[rf].order_hint = ref.order_hint;
      mf.refs[rf].num_saved_refs = ref.num_saved_refs;
      for (int k = 0; k < ref.num_saved_refs; ++k) {
        mf.refs[rf].saved_order_hint[k] = ref.saved_order_hint[k];
      }
      mf.saved[rf] = ref.saved();
    }

    temporal::TipConfig tip;
    tip.mode = temporal::TipFrameMode::kAsOutput;
    tip.tip_global_wtd_index =
        choose_tip_weight_index(past.order_hint, future.order_hint, hint);
    tip.tip_global_mv = temporal::Mv{0, 0};
    tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
    tip.allow_tip_hole_fill = options.tip_hole_fill;
    tip.enable_tip_refinemv = false;
    tip.enable_refinemv = false;
    tip.opfl_refine_type = temporal::RefineType::kNone;
    tip.num_past_refs = 1;
    tip.num_future_refs = 1;
    tip.order_hint = hint;

    TipSlotState& slot = slots[static_cast<std::size_t>(i + 1)];
    for (int plane = 0; plane < 3; ++plane) {
      const int pw = plane == 0 ? recon_mi_cols * 4 : recon_mi_cols * 2;
      const int ph = plane == 0 ? recon_mi_rows * 4 : recon_mi_rows * 2;
      slot.plane[plane].resize(pw, ph, 0);
    }

    std::vector<std::int32_t> field_mv(
        static_cast<std::size_t>(recon_mi_cols / 2) *
            static_cast<std::size_t>(recon_mi_rows / 2) * 2U, 0);
    std::vector<std::uint8_t> field_valid(
        static_cast<std::size_t>(recon_mi_cols / 2) *
            static_cast<std::size_t>(recon_mi_rows / 2), 0);
    std::vector<std::int32_t> field_offset(field_valid.size(), 0);
    std::vector<std::int32_t> avg_mv(field_mv.size(), 0);
    std::vector<std::uint8_t> avg_valid(field_valid.size(), 0);
    temporal::MotionField field;
    field.w8 = recon_mi_cols / 2;
    field.h8 = recon_mi_rows / 2;
    field.mv = field_mv.data();
    field.valid = field_valid.data();
    field.offset = field_offset.data();
    temporal::MotionFieldScratch scratch;
    scratch.avg_mv = avg_mv.data();
    scratch.avg_valid = avg_valid.data();

    const int capacity =
        ((recon_mi_cols + 3) / 4) * ((recon_mi_rows + 3) / 4);
    std::vector<temporal::TipBlock> blocks(static_cast<std::size_t>(capacity));
    std::vector<temporal::Mv> grid0(
        static_cast<std::size_t>(recon_mi_cols) *
        static_cast<std::size_t>(recon_mi_rows));
    std::vector<temporal::Mv> grid1(grid0.size());
    temporal::TipPlan plan;
    plan.num_blocks = capacity;
    plan.blocks = blocks.data();
    plan.mv_grid = grid0.data();
    plan.mv_grid1 = grid1.data();

    tip::TipBuildInput in;
    in.config = mf;
    in.tip = tip;
    in.field = &field;
    in.scratch = &scratch;
    in.plan = &plan;
    for (int plane = 0; plane < 3; ++plane) {
      in.ref[plane][0] = temporal::PlaneView{
          past.plane[plane].samples.data(), past.plane[plane].stride,
          past.plane[plane].width, past.plane[plane].height};
      in.ref[plane][1] = temporal::PlaneView{
          future.plane[plane].samples.data(), future.plane[plane].stride,
          future.plane[plane].width, future.plane[plane].height};
      in.dst[plane] = slot.plane[plane].samples.data();
      in.dst_stride[plane] = slot.plane[plane].stride;
      in.dst_width[plane] = plane == 0 ? static_cast<int>(width)
                                       : static_cast<int>(width) / 2;
      in.dst_height[plane] = plane == 0 ? static_cast<int>(height)
                                        : static_cast<int>(height) / 2;
      in.sub_x[plane] = plane == 0 ? 0 : 1;
      in.sub_y[plane] = plane == 0 ? 0 : 1;
    }
    in.pred_scratch = pred_scratch.data();
    in.block_scratch = block_scratch.data();

    tip::TipBuildOutput out;
    if (tip::build_tip_frame(in, &out) != temporal::TemporalStatus::kOk) {
      error = "TIP frame " + std::to_string(i) + ": build TIP failed";
      return false;
    }
    temporal::TipConfig used = tip;
    tip::fill_tip_closest_refs(mf, out.field_result, &used);

    slot.valid = true;
    slot.order_hint = hint;
    init_empty_field(&slot, recon_mi_cols, recon_mi_rows);
    slot.num_saved_refs = 2;
    slot.saved_order_hint[0] = past.order_hint;
    slot.saved_order_hint[1] = future.order_hint;
    {
      temporal::RefMvField stored;
      stored.w8 = slot.w8;
      stored.h8 = slot.h8;
      stored.ref = slot.field_ref.data();
      stored.mv = slot.field_mv.data();
      tip::TipStoreInput store;
      store.mi_cols = recon_mi_cols;
      store.mi_rows = recon_mi_rows;
      store.order_hint = hint;
      store.closest_past = out.field_result.closest_past;
      store.closest_future = out.field_result.closest_future;
      store.closest_past_order_hint = used.closest_past_order_hint;
      store.closest_future_order_hint = used.closest_future_order_hint;
      store.weight16 =
          temporal::kTipWeightingFactor[tip.tip_global_wtd_index] == 16;
      if (tip::store_tip_frame_motion_field(plan, store, &stored) !=
          temporal::TemporalStatus::kOk) {
        error = "TIP frame " + std::to_string(i) + ": §7.22 store failed";
        return false;
      }
    }

    VideoFrameHeader header;
    header.keyframe = false;
    header.tip_frame = true;
    header.order_hint = static_cast<std::uint32_t>(hint);
    header.num_total_refs = 2;
    header.ref_frame_idx[0] = static_cast<std::uint8_t>(past_slot);
    header.ref_frame_idx[1] = static_cast<std::uint8_t>(future_slot);
    header.ref_slot = header.ref_frame_idx[0];
    header.refresh_frame_flags = 1U << (i + 1);
    header.use_ref_frame_mvs = true;
    header.tip_global_wtd_index =
        static_cast<std::uint8_t>(tip.tip_global_wtd_index);
    header.tip_global_mv_row = 0;
    header.tip_global_mv_col = 0;
    header.tip_sharp = true;
    header.allow_tip_hole_fill = tip.allow_tip_hole_fill;

    // §5.14: a TIP picture codes film_grain_config() right after the TIP block
    // (the config is the only tail syntax the TIP header keeps).
    grain::GrainFramePlan tip_grain;
    if (options.film_grain &&
        !plan_grain_frame(frames[static_cast<std::size_t>(hint)],
                          static_cast<std::uint32_t>(hint), grain_planner,
                          tip_grain, error)) {
      error = "TIP frame " + std::to_string(i) + " film grain: " + error;
      return false;
    }
    if (options.film_grain) {
      header.apply_grain = tip_grain.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(tip_grain.fgm_id);
      header.grain_seed = tip_grain.grain_seed;
    }

    std::vector<std::uint8_t> obu = make_tip_frame_obu(config, header, error);
    if (obu.empty()) {
      error = "TIP frame " + std::to_string(i) + ": " + error;
      return false;
    }
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain_obu(tip_grain, unit, error)) {
      error = "TIP frame " + std::to_string(i) + ": " + error;
      return false;
    }
    result.units.push_back(std::move(unit));
    result.recon.push_back(to_recon_frame(slot.plane[0], slot.plane[1],
                                         slot.plane[2], width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(0);
    result.inter_blocks.push_back(static_cast<std::uint32_t>(out.num_blocks));
    result.intra_blocks.push_back(0);
    result.frame_output_index.push_back(i + 1);
  }

  if (ref_mvs_tail) {
    // --- Tail P referencing the TIP display frame (§7.9 over a TIP slot) ---
    //
    // The TIP frames above refresh slots 2..N-2 and store their own §7.22
    // fields.  This last input picture (display N-2) is coded as a regular
    // tile-coded P whose reference list is [final TIP (past), anchor
    // (future)], so the decoder runs §7.9 over the TIP frame's stored field.
    const int display = frame_count - 2;
    const int hint = display;
    const int tip_slot = display;        // slot i+1 with i = display - 1
    const int anchor_slot = 1;
    const int write_slot = frame_count - 1;
    TipSlotState& past = slots[static_cast<std::size_t>(tip_slot)];
    TipSlotState& future = slots[static_cast<std::size_t>(anchor_slot)];
    if (!past.valid || !future.valid) {
      error = "TIP tail P: reference slot was not coded";
      return false;
    }
    std::vector<inter::mc::RefFrame> references(2);
    const TipSlotState* ref_slots_state[2] = {&past, &future};
    for (int r = 0; r < 2; ++r) {
      for (int plane = 0; plane < 3; ++plane) {
        references[static_cast<std::size_t>(r)].plane[plane] =
            inter::mc::PlaneView{
                ref_slots_state[r]->plane[plane].samples.data(),
                ref_slots_state[r]->plane[plane].stride};
      }
      references[static_cast<std::size_t>(r)].mi_cols = recon_mi_cols;
      references[static_cast<std::size_t>(r)].mi_rows = recon_mi_rows;
    }
    std::array<int, 7> frame_distance{};
    frame_distance[0] = relative_dist(static_cast<std::uint32_t>(hint),
                                      static_cast<std::uint32_t>(past.order_hint),
                                      kOrderHintBits);
    frame_distance[1] = relative_dist(static_cast<std::uint32_t>(hint),
                                      static_cast<std::uint32_t>(future.order_hint),
                                      kOrderHintBits);
    std::array<int, 7> ref_order_hints{};
    ref_order_hints[0] = past.order_hint;
    ref_order_hints[1] = future.order_hint;

    auto saved_ref = [](const TipSlotState& slot) {
      SavedFieldRef out;
      out.num_saved_refs = slot.num_saved_refs;
      out.saved_order_hint = slot.saved_order_hint;
      out.field = slot.saved();
      return out;
    };
    SavedFieldRef ref_fields[kRefsPerFrame] = {};
    ref_fields[0] = saved_ref(past);
    ref_fields[1] = saved_ref(future);
    int ref_slot_ids[kRefsPerFrame] = {tip_slot, anchor_slot};
    std::uint8_t ref_is_key[kRefsPerFrame] = {0, 0};
    MotionFieldFrame motion_field;
    if (!estimate_motion_field(
            ref_fields, ref_slot_ids, ref_order_hints.data(), ref_is_key,
            /*ref_valid=*/true, 2, hint, static_cast<int>(width), static_cast<int>(height),
            recon_mi_cols, recon_mi_rows, /*proj_step=*/1,
            /*enable_mv_traj=*/false, &motion_field, error, options.tip_block_ref)) {
      error = "TIP tail P: " + error;
      return false;
    }
    TemporalFrameContext temporal_ctx;
    temporal_ctx.use_ref_frame_mvs = true;
    temporal_ctx.order_hint = hint;
    temporal_ctx.num_past_refs = motion_field.config.num_past_refs;
    temporal_ctx.num_future_refs = motion_field.config.num_future_refs;
    temporal_ctx.closest_past = motion_field.result.closest_past;
    temporal_ctx.closest_future = motion_field.result.closest_future;
    temporal_ctx.field = &motion_field.field;
    if (options.tip_block_ref) {
      // §5.18.2 TipFrameMode = TIP_FRAME_AS_REF.  The decoder runs the §7.10.2
      // temporal scale, hole fill and block-average filter on this frame's
      // own §7.9 field before decoding any tile, so the encoder must mutate
      // the same field before its block decisions.
      if (motion_field.result.closest_past < 0 || motion_field.result.closest_future < 0) {
        error = "TIP tail P: TIP block reference needs both closest refs";
        return false;
      }
      temporal::TipConfig tip;
      tip.mode = temporal::TipFrameMode::kAsReference;
      tip.tip_global_wtd_index = choose_tip_weight_index(past.order_hint, future.order_hint, hint);
      tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
      tip.enable_tip_refinemv = false;
      tip.enable_refinemv = false;
      tip.opfl_refine_type = temporal::RefineType::kNone;
      tip.allow_tip_hole_fill = options.tip_hole_fill;
      tip.num_past_refs = motion_field.config.num_past_refs;
      tip.num_future_refs = motion_field.config.num_future_refs;
      tip.order_hint = hint;
      tip.closest_past_order_hint = ref_order_hints[motion_field.result.closest_past];
      tip.closest_future_order_hint = ref_order_hints[motion_field.result.closest_future];
      const std::size_t cells = static_cast<std::size_t>(motion_field.mi_cols / 2) *
                                static_cast<std::size_t>(motion_field.mi_rows / 2);
      std::vector<std::int32_t> avg_mv(cells * 2, 0);
      std::vector<std::uint8_t> avg_valid(cells, 0);
      temporal::MotionFieldScratch scratch;
      scratch.avg_mv = avg_mv.data();
      scratch.avg_valid = avg_valid.data();
      if (temporal::setup_tip_motion_field(motion_field.config, tip, &motion_field.field, nullptr,
                                           &scratch,
                                           &motion_field.result) != temporal::TemporalStatus::kOk) {
        error = "TIP tail P: §7.10.2-.4 setup failed";
        return false;
      }
      if (options.trace) {
        int valid = 0;
        int nonzero = 0;
        for (std::size_t i = 0; i < cells; ++i) {
          if (motion_field.valid[i] == 0) continue;
          ++valid;
          if (motion_field.mv[i * 2 + 0] != 0 || motion_field.mv[i * 2 + 1] != 0) ++nonzero;
        }
        std::cerr << "tip_block_ref field valid=" << valid << " nonzero=" << nonzero << " of "
                  << cells << " offsets=(" << tip.closest_past_order_hint << ","
                  << tip.closest_future_order_hint << ")\n";
      }
      temporal_ctx.tip_as_ref = true;
      temporal_ctx.tip_weight_index = tip.tip_global_wtd_index;
      temporal_ctx.tip_allow_hole_fill = tip.allow_tip_hole_fill;
      temporal_ctx.tip_closest_past_order_hint = tip.closest_past_order_hint;
      temporal_ctx.tip_closest_future_order_hint = tip.closest_future_order_hint;
    }

    std::vector<std::uint8_t> payload;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    InterFrameEncoder encoder(frames[static_cast<std::size_t>(display)],
                              references, frame_distance, 2, options,
                              options.trace, /*keyframe=*/false, 0, 0, 0, 0,
                              {}, nullptr, nullptr, ref_order_hints,
                              /*primary_ref_is_none=*/false, &temporal_ctx);
    if (!encoder.encode(payload, blocks, levels, error)) {
      error = "TIP tail P encode failed: " + error;
      return false;
    }
    result.temporal_candidates +=
        static_cast<std::uint64_t>(encoder.temporal_added());
    result.temporal_selected +=
        static_cast<std::uint64_t>(encoder.temporal_selected());
    TipSlotState& slot = slots[static_cast<std::size_t>(write_slot)];
    for (int plane = 0; plane < 3; ++plane) {
      slot.plane[plane] = encoder.recon(plane);
    }
    slot.valid = true;
    slot.order_hint = hint;
    if (!store_coded_field(encoder, hint, ref_order_hints.data(), 2, &slot,
                           error)) {
      error = "TIP tail P: " + error;
      return false;
    }
    slot.num_saved_refs = 2;
    slot.saved_order_hint[0] = past.order_hint;
    slot.saved_order_hint[1] = future.order_hint;

    VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = static_cast<std::uint32_t>(hint);
    header.num_total_refs = 2;
    header.ref_frame_idx[0] = static_cast<std::uint8_t>(tip_slot);
    header.ref_frame_idx[1] = static_cast<std::uint8_t>(anchor_slot);
    header.ref_slot = header.ref_frame_idx[0];
    header.refresh_frame_flags = 1U << write_slot;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    header.use_ref_frame_mvs = true;
    if (options.tip_block_ref) {
      header.tip_as_ref = true;
      header.tip_global_wtd_index = static_cast<std::uint8_t>(temporal_ctx.tip_weight_index);
      header.allow_tip_hole_fill = temporal_ctx.tip_allow_hole_fill;
      result.tip_block_refs += static_cast<std::uint64_t>(encoder.tip_blocks());
      if (options.trace) {
        std::cerr << "tip_block_ref pictures=" << frame_count << " blocks=" << encoder.tip_blocks()
                  << " weight=" << temporal::kTipWeightingFactor[temporal_ctx.tip_weight_index]
                  << " order_hint=" << hint << " ref=(" << static_cast<int>(header.ref_frame_idx[0])
                  << "," << static_cast<int>(header.ref_frame_idx[1]) << ")\n";
      }
    }

    grain::GrainFramePlan tail_grain;
    if (options.film_grain &&
        !plan_grain_frame(frames[static_cast<std::size_t>(display)],
                          static_cast<std::uint32_t>(display), grain_planner,
                          tail_grain, error)) {
      error = "TIP tail P film grain: " + error;
      return false;
    }
    if (options.film_grain) {
      header.apply_grain = tail_grain.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(tail_grain.fgm_id);
      header.grain_seed = tail_grain.grain_seed;
    }

    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu = make_frame_obu(config, header, payload);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain_obu(tail_grain, unit, error)) {
      error = "TIP tail P: " + error;
      return false;
    }
    result.units.push_back(std::move(unit));
    result.recon.push_back(to_recon_frame(slot.plane[0], slot.plane[1],
                                         slot.plane[2], width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(payload.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    // The TIP schedule is immediate-output with monotonic_output_order_flag = 1,
    // so the decoder emits pictures in decode order: the tail P is the last
    // output picture, not display index N-2.
    result.frame_output_index.push_back(
        static_cast<std::int32_t>(result.recon.size()) - 1);
  }
  return true;
}

}  // namespace vf_detail
}  // namespace av2enc::video
