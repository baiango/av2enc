#include "pipeline/video/video_frame_internal.h"

#include "pipeline/video/video_interp_filter.h"

namespace av2enc::video {

void InterFrameEncoder::global_mv_for_block(int ref_index, int mi_row, int mi_col, int bw4,
                         int bh4, bool& use_warp,
                         inter::mc::Mv& mv) const {
  use_warp = false;
  mv = inter::mc::Mv{0, 0};
#if defined(AV2ENC_VIDEO_GME)
  if (gm_plan_ == nullptr || ref_index < 0 ||
      ref_index >= gm_plan_->num_refs) {
    return;
  }
  const inter::gme::GmParams& params =
      gm_plan_->params[static_cast<std::size_t>(ref_index)];
  if (params.type == inter::gme::GmType::kIdentity) return;
  inter::gme::MvPrecision precision = inter::gme::MvPrecision::kEighthPel;
  if (options_.mv_precision == 1) {
    precision = inter::gme::MvPrecision::kQuarterPel;
  } else if (options_.mv_precision == 2) {
    precision = inter::gme::MvPrecision::kHalfPel;
  }
  int projected[2] = {0, 0};
  inter::gme::get_warp_motion_vector(params, mi_col, mi_row, bw4, bh4,
                                     mi_cols_, mi_rows_, precision,
                                     projected);
  mv = inter::mc::Mv{projected[0], projected[1]};
  // §7.13.3.14 derives useWarp from the model, not from an encoder flag: a
  // non-identity GLOBALMV model always selects the §7.13.3.19/.20 warped
  // predictor for blocks at least 8x8 (an invalid shear falls back to the
  // extended warp).  --video-warp is therefore the explicit alias for
  // --video-gme; using translational MC here would desync the decoder.
  // The default plan never carries a warp model (make_gm_frame_plan flattens
  // it to its translation-only projection), so for translation models this
  // kernel projects the same constant centre MV every 8x8 and takes the
  // conformant rounding path the gated pan fixture already proves.
  use_warp = bw4 * 4 >= 8 && bh4 * 4 >= 8;
#else
  (void)ref_index;
  (void)mi_row;
  (void)mi_col;
  (void)bw4;
  (void)bh4;
#endif
}

merge::StackParams InterFrameEncoder::make_stack_params(int mi_row, int mi_col, int w4,
                                     int h4, int ref_index,
                                     SbSink::State& st) const {
  merge::StackParams params;
  params.mi_row = mi_row;
  params.mi_col = mi_col;
  params.bw4 = w4;
  params.bh4 = h4;
  params.sb_h4 = 16;
  params.mi_cols = mi_cols_;
  params.mi_rows = mi_rows_;
  params.mi = mv_grid_.data();
  // The merge layer uses AV2 RefFrame labels: 0..6 are reference indices,
  // INTRA = 8.
  params.ref_frame[0] = static_cast<std::int8_t>(ref_index);
  params.ref_frame[1] = merge::kRefNone;
  params.is_compound = false;
  bool gm_use_warp = false;
  inter::mc::Mv gm_mv{};
  global_mv_for_block(ref_index, mi_row, mi_col, w4, h4, gm_use_warp, gm_mv);
  params.global_mv[0] = gm_mv;
  params.global_mv[1] = gm_mv;
  // DRL reordering is not gated (see src/pipeline/video/README.md).
  params.reorder = merge::DrlReorder::kDisabled;
  params.max_drl_bits_minus_1 = 4;
  if (num_total_refs_ > 1) {
    params.num_refs = num_total_refs_;
    params.frame_distance = frame_distance_.data();
  }
  params.use_ref_frame_mvs = temporal_ctx_.use_ref_frame_mvs;
  params.motion_field = temporal_ctx_.field;
  params.mv_traj = temporal_ctx_.mv_traj;
  if (options_.mv_traj) {
    params.mv_traj_derived = &st.mv_traj_derived;
  }
  params.order_hint = temporal_ctx_.order_hint;
  params.order_hints = ref_order_hints_.data();
  params.num_past_refs = temporal_ctx_.num_past_refs;
  params.num_future_refs = temporal_ctx_.num_future_refs;
  params.closest_past = temporal_ctx_.closest_past;
  params.closest_future = temporal_ctx_.closest_future;
  params.tile_mi_col_start = tile_mi_col_;
  params.tile_mi_row_start = tile_mi_row_;
  params.tile_mi_col_end = frame_mi_cols_;
  params.tile_mi_row_end = frame_mi_rows_;
  params.temporal_added = &st.temporal_added;
  if (options_.refmvbank) {
    params.refmv_bank = &refmv_bank_;
    params.refmvbank_added = &st.refmvbank_added;
    attach_video_refmvbank(&params);
  }
  attach_video_temporal_scan(&params);
  return params;
}

bool InterFrameEncoder::drl_constraint_sort_suppressed(int ref_index) const {
  if (ref_index < 0 || ref_index > 6) return false;
  bool past = false;
  bool future = false;
  for (int r = 0; r < num_total_refs_ && r < 7; ++r) {
    const int d = frame_distance_[static_cast<std::size_t>(r)];
    if (d > 0) {
      past = true;
    } else if (d < 0) {
      future = true;
    }
  }
  if (past && future) return false;  // has_bothside_refs (§5.18.2)
  const int d = frame_distance_[static_cast<std::size_t>(ref_index)];
  return (d < 0 ? -d : d) <= 2;
}

bool InterFrameEncoder::load_stack(int mi_row, int mi_col, int w4, int h4, int ref_index,
                SbSink::State& st, merge::StackParams& params,
                merge::MvStack& stack) const {
  params = make_stack_params(mi_row, mi_col, w4, h4, ref_index, st);
  const drl::Mode mode =
      options_.drl_reorder
          ? (options_.drl_reorder_always ? drl::Mode::kAlways
                                         : drl::Mode::kConstraint)
          : drl::Mode::kDisabled;
  if (mode == drl::Mode::kDisabled) {
    return merge::build_stack_checked(params, &stack) ==
           merge::Status::kOk;
  }
  const bool suppress =
      mode == drl::Mode::kConstraint &&
      !params.is_compound && drl_constraint_sort_suppressed(ref_index);
  if (!trace_) {
    return drl::build_reordered_stack(params, mode, suppress, &stack) ==
           merge::Status::kOk;
  }
  // --video-trace: the transform is one move-to-front over
  // [0, numNearest), so report the blocks it actually reorders.  The gate
  // counts these lines to prove the reorder is exercised rather than merely
  // enabled in the sequence header.
  merge::MvStack pre;
  if (merge::build_stack_checked(params, &pre) != merge::Status::kOk) {
    return false;
  }
  if (drl::build_reordered_stack(params, mode, suppress, &stack) !=
      merge::Status::kOk) {
    return false;
  }
  bool swapped = false;
  for (int i = 0; i < stack.count && i < pre.count; ++i) {
    if (pre.cand[i].mv[0].y != stack.cand[i].mv[0].y ||
        pre.cand[i].mv[0].x != stack.cand[i].mv[0].x) {
      swapped = true;
      break;
    }
  }
  if (swapped) {
    std::cerr << "  drl reorder mi=(" << mi_row << "," << mi_col << ") "
              << (w4 * 4) << "x" << (h4 * 4) << " ref=" << ref_index
              << " nn=" << stack.num_nearest << " pre=";
    for (int i = 0; i < pre.count; ++i) {
      std::cerr << (i ? " " : "") << "w" << pre.cand[i].weight << "("
                << pre.cand[i].mv[0].y << "," << pre.cand[i].mv[0].x << ")";
    }
    std::cerr << " post=";
    for (int i = 0; i < stack.count; ++i) {
      std::cerr << (i ? " " : "") << "w" << stack.cand[i].weight << "("
                << stack.cand[i].mv[0].y << "," << stack.cand[i].mv[0].x
                << ")";
    }
    std::cerr << "\n";
  }
  return true;
}

void InterFrameEncoder::reset_refmvbank_for_sb(int mi_row, int mi_col) {
  if (!options_.refmvbank || keyframe_) return;
  // §6.19.2.1 clear_left_context() runs once per superblock row and clears
  // the bank sizes/starts.  The bank entries themselves are re-seeded from
  // the coded row above at each superblock; the pinned decoder also keeps
  // the size/start rings empty across the row boundary, so the clear must
  // happen here rather than only at the frame head.
  if (mi_row % (sb_size_ / 4) == 0 && mi_col == tile_mi_col_) {
    for (int ref = 0; ref < inter::temporal::kBankRefsPerFrame; ++ref) {
      refmv_bank_.size[ref] = 0;
      refmv_bank_.start[ref] = 0;
    }
  }
  std::vector<inter::temporal::RefMvBankRowSource> sources(
      static_cast<std::size_t>(mi_cols_));
  inter::temporal::RefMvBankRow row;
  row.mi_cols = mi_cols_;
  row.sources = sources.data();
  const bool top_row = (mi_row == tile_mi_row_);
  if (!top_row && mi_row > 0) {
    const int cand_row = mi_row - 1;
    for (int col = 0; col < mi_cols_; ++col) {
      const inter::mc::MiInfo& mi =
          mv_grid_[static_cast<std::size_t>(cand_row) *
                       static_cast<std::size_t>(mi_cols_) +
                   static_cast<std::size_t>(col)];
      inter::temporal::RefMvBankRowSource& src =
          sources[static_cast<std::size_t>(col)];
      src.is_inter = static_cast<std::uint8_t>(mi.is_inter != 0 ? 1 : 0);
      src.ref[0] = mi.ref[0];
      src.ref[1] = mi.ref[1];
      src.mv[0][0] = mi.mv[0][0];
      src.mv[0][1] = mi.mv[0][1];
      src.mv[1][0] = mi.mv[1][0];
      src.mv[1][1] = mi.mv[1][1];
      src.cwp_idx = 8;  // CWP_EQUAL: the product path codes no cwp_idx
      src.motion_mode = 0;
      src.bw4 = mi.w4 > 0 ? mi.w4 : 1;
    }
  } else {
    row.sources = nullptr;
  }
  inter::temporal::reset_ref_mv_bank(&refmv_bank_, mi_row, mi_col,
                                     sb_size_ / 4, /*frame_is_intra=*/false,
                                     top_row, row);
}

void InterFrameEncoder::update_refmvbank_leaf(const syntax::BlockSyntax& entry) {
  if (!options_.refmvbank || keyframe_) return;
  if (refmv_bank_.bank_hits >= inter::temporal::kMaxRmbSbHits) return;
  inter::temporal::RefMvBankBlockContext ctx;
  ctx.mi_row = entry.mi_row;
  ctx.mi_col = entry.mi_col;
  ctx.bw4 = partition::block_width(
                static_cast<partition::BlockSize>(entry.bsize)) >>
            2;
  ctx.bh4 = partition::block_height(
                static_cast<partition::BlockSize>(entry.bsize)) >>
            2;
  ctx.sb_size4 = sb_size_ / 4;
  ctx.chroma_part = false;
  if (entry.is_inter != 0) {
    // ref_index1 is a reference index (0..6), or 1 when not compound; it is
    // never the int8 sentinel kRefNone, so only kIntraFrame is a real guard.
    const bool compound =
        entry.compound != 0 &&
        entry.ref_index1 != static_cast<std::uint8_t>(merge::kIntraFrame);
    const int refs[2] = {
        static_cast<int>(entry.ref_index),
        compound ? static_cast<int>(entry.ref_index1)
                 : static_cast<int>(merge::kRefNone)};
    const inter::temporal::Mv mvs[2] = {
        inter::temporal::Mv{entry.mv_y, entry.mv_x},
        compound ? inter::temporal::Mv{entry.mv1_y, entry.mv1_x}
                 : inter::temporal::Mv{0, 0}};
    inter::temporal::update_ref_mv_bank(
        &refmv_bank_, ctx, refs, mvs, /*cwp_idx=*/8,
        /*from_within_sb=*/true);
  } else {
    inter::temporal::update_ref_mv_count(&refmv_bank_, ctx);
  }
}

bool InterFrameEncoder::predict_luma(int ref_index, inter::mc::Mv mv, int x0, int y0, int w,
                  int h, std::vector<std::uint8_t>& out) const {
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  const inter::mc::BlockRect rect{x0, y0, w, h};
  return inter::mc::predict_plane(
             references_[static_cast<std::size_t>(ref_index)], 0, rect, mv,
             leaf_interp_filter(), out.data(), w) ==
         inter::mc::McStatus::kOk;
}

bool InterFrameEncoder::predict_all(int ref_index, inter::mc::Mv mv, int x0, int y0, int w,
                 int h, std::vector<std::uint8_t>& luma,
                 std::vector<std::uint8_t>& u,
                 std::vector<std::uint8_t>& v) const {
  const int cw = w / 2;
  const int ch = h / 2;
  luma.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  inter::mc::MutablePlane dst[3];
  dst[0] = inter::mc::MutablePlane{luma.data(), w};
  dst[1] = inter::mc::MutablePlane{u.data(), cw};
  dst[2] = inter::mc::MutablePlane{v.data(), cw};
  const inter::mc::BlockRect rect{x0, y0, w, h};
  return inter::mc::predict_block_420(
             references_[static_cast<std::size_t>(ref_index)], rect, mv,
             leaf_interp_filter(), dst) ==
         inter::mc::McStatus::kOk;
}

bool InterFrameEncoder::run_me(int ref_index, int x0, int y0, int w, int h,
            inter::mc::Mv& mv, std::string& error) {
  if (!me_cache_.empty()) {
    const auto it = me_cache_.find(me_cache_key(ref_index, x0, y0, w, h));
    if (it != me_cache_.end()) {
      mv = it->second;
      return true;
    }
  }
  return search_me(ref_index, x0, y0, w, h, mv, error);
}

bool InterFrameEncoder::search_me(int ref_index, int x0, int y0, int w, int h,
               inter::mc::Mv& mv, std::string& error) const {
  const int frame_width = static_cast<int>(source_.width);
  const int frame_height = static_cast<int>(source_.height);
  const int frame_chroma_width = frame_width / 2;
  const int frame_chroma_height = frame_height / 2;
  me::MeCoreFrame cur;
  cur.width = frame_width;
  cur.height = frame_height;
  cur.plane[0] =
      me::MeCorePlane{source_.y.data(), frame_width, frame_width,
                      frame_height};
  cur.plane[1] = me::MeCorePlane{source_.u.data(), frame_chroma_width,
                                 frame_chroma_width, frame_chroma_height};
  cur.plane[2] = me::MeCorePlane{source_.v.data(), frame_chroma_width,
                                 frame_chroma_width, frame_chroma_height};
  me::MeCoreRef ref;
  ref.plane[0] = me::MeCorePlane{
      references_[static_cast<std::size_t>(ref_index)].plane[0].data,
      references_[static_cast<std::size_t>(ref_index)].plane[0].stride,
      frame_mi_cols_ * 4, frame_mi_rows_ * 4};
  ref.plane[1] = me::MeCorePlane{
      references_[static_cast<std::size_t>(ref_index)].plane[1].data,
      references_[static_cast<std::size_t>(ref_index)].plane[1].stride,
      frame_mi_cols_ * 2, frame_mi_rows_ * 2};
  ref.plane[2] = me::MeCorePlane{
      references_[static_cast<std::size_t>(ref_index)].plane[2].data,
      references_[static_cast<std::size_t>(ref_index)].plane[2].stride,
      frame_mi_cols_ * 2, frame_mi_rows_ * 2};

  me::MeCoreJob job;
  job.x = x0;
  job.y = y0;
  job.w = w;
  job.h = h;
  job.num_refs = 1;
  job.ref_indices[0] = 0;

  me::MeCoreOptions opts;
  opts.method = me::kMeDiamond;
  opts.metric = me::kMeSad;
  opts.range = options_.me_range;
  opts.subpel_stages = me::kMeSubpelHalf | me::kMeSubpelQuarter;
  if (options_.me_subpel_eighth) {
    opts.subpel_stages |= me::kMeSubpelEighth;
  }
  opts.subpel_interp = me::kMeInterpRegular;
  opts.use_predictors = 1;
  opts.use_merge_candidates = 0;

  me::MeCoreResult res;
  if (me::me_search_batch(cur, &ref, 1, &job, 1, opts, nullptr, &res) !=
      me::MeStatus::kOk) {
    error = "me_search_batch failed";
    return false;
  }
  mv = inter::mc::mv_from_me_1_16(res.mv_y, res.mv_x);
  return true;
}

std::uint64_t InterFrameEncoder::me_cache_key(int ref_index, int x0, int y0, int w,
                                  int h) {
  return (static_cast<std::uint64_t>(ref_index) & 0x7U) |
         ((static_cast<std::uint64_t>(x0) & 0xFFFFU) << 3U) |
         ((static_cast<std::uint64_t>(y0) & 0xFFFFU) << 19U) |
         ((static_cast<std::uint64_t>(w) & 0x3FFU) << 35U) |
         ((static_cast<std::uint64_t>(h) & 0x3FFU) << 45U);
}

void InterFrameEncoder::precompute_leaf_me(const std::vector<PartitionNode>& superblocks,
                        unsigned workers) {
  if (keyframe_ || !options_.enable_newmv || options_.force_global_mv) return;
  if (width_ != static_cast<int>(source_.width) ||
      height_ != static_cast<int>(source_.height)) {
    return;  // tiled: the tile encoders themselves run in parallel
  }
  if (parallel::resolve_workers(workers) <= 1) return;

  struct MeJob {
    int ref_index;
    int x;
    int y;
    int w;
    int h;
  };
  std::vector<MeJob> jobs;
  jobs.reserve(1024);
  const auto walk = [&](auto&& self, const PartitionNode& node) -> void {
    if (node.partition != PartitionType::kNone) {
      for (const PartitionNode& child : node.children) self(self, child);
      return;
    }
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    if (w < 8 || h < 8) return;  // encode_leaf() keeps sub-8 leaves intra
    const int x0 = node.mi_col * 4;
    const int y0 = node.mi_row * 4;
    if (x0 + w > mi_cols_ * 4 || y0 + h > mi_rows_ * 4) return;
    for (int ref = 0; ref < num_total_refs_; ++ref) {
      jobs.push_back(MeJob{ref, x0 + tile_x_, y0 + tile_y_, w, h});
    }
  };
  for (const PartitionNode& sb : superblocks) walk(walk, sb);
  if (jobs.empty()) return;

  std::vector<inter::mc::Mv> mvs(jobs.size());
  std::vector<char> ok(jobs.size(), 0);
  // Round-robin the raster-ordered job list over the fixed chunks: the leaf
  // search cost has a strong vertical gradient (flat sky vs textured
  // ground), so contiguous raster slices would leave workers idle.  The
  // permutation is built before the threads start and is a bijection, so
  // every job runs exactly once and the merge below stays in job order.
  const std::size_t chunks = std::min<std::size_t>(
      static_cast<std::size_t>(parallel::resolve_workers(workers)),
      jobs.size());
  const std::size_t chunk_size = (jobs.size() + chunks - 1) / chunks;
  std::vector<std::size_t> order;
  order.reserve(jobs.size());
  for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
    for (std::size_t rank = 0; rank < chunk_size; ++rank) {
      const std::size_t job_index = rank * chunks + chunk;
      if (job_index < jobs.size()) order.push_back(job_index);
    }
  }
  parallel::parallel_for(order.size(), workers, [&](std::size_t p) {
    const std::size_t job_index = order[p];
    const MeJob& job = jobs[job_index];
    std::string job_error;
    ok[job_index] = search_me(job.ref_index, job.x, job.y, job.w, job.h,
                              mvs[job_index], job_error)
                        ? 1
                        : 0;
  });
  // Merge in job order.  A failed job stays uncached and falls back to the
  // serial search, so the error text and behaviour are unchanged.
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    if (ok[i] == 0) continue;
    const MeJob& job = jobs[i];
    me_cache_.emplace(me_cache_key(job.ref_index, job.x, job.y, job.w,
                                   job.h),
                      mvs[i]);
  }
}

bool InterFrameEncoder::evaluate_inter_intra(int bsize, int x0, int y0, int w, int h,
                          const std::vector<std::uint8_t>& inter_y,
                          const std::vector<std::uint8_t>& inter_u,
                          const std::vector<std::uint8_t>& inter_v,
                          std::vector<std::uint8_t>& out_y,
                          std::vector<std::uint8_t>& out_u,
                          std::vector<std::uint8_t>& out_v, int& mode_out,
                          bool& wedge_out, int& index_out,
                          std::int64_t& sse_out, const SbSink::State& st,
                          std::string& error) {
  static constexpr intra::Mode kModes[4] = {
      intra::Mode::kDc, intra::Mode::kV, intra::Mode::kH,
      intra::Mode::kSmooth};
  bool above_smooth = false;
  bool left_smooth = false;
  luma_neighbor_smooth(y0 / 4, x0 / 4, above_smooth, left_smooth);
  std::vector<std::uint8_t> ii_y[4];
  std::vector<std::uint8_t> ii_u[4];
  std::vector<std::uint8_t> ii_v[4];
  for (int m = 0; m < 4; ++m) {
    if (!predict_intra_plane_mode(0, x0, y0, w, h, kModes[m], above_smooth,
                                  left_smooth, st, ii_y[m]) ||
        !predict_intra_plane_mode(1, x0 / 2, y0 / 2, w / 2, h / 2,
                                  kModes[m], false, false, st, ii_u[m]) ||
        !predict_intra_plane_mode(2, x0 / 2, y0 / 2, w / 2, h / 2,
                                  kModes[m], false, false, st, ii_v[m])) {
      error = "inter-intra intra prediction failed";
      return false;
    }
  }
  inter::adv::InterIntraSearchInput in{};
  in.bsize = bsize;
  in.rect = inter::adv::BlockRect{0, 0, w, h};
  for (int m = 0; m < 4; ++m) in.intra[m] = ii_y[m].data();
  in.intra_stride = w;
  std::vector<std::uint8_t> scratch(static_cast<std::size_t>(w) *
                                    static_cast<std::size_t>(h));
  in.scratch = scratch.data();
  in.scratch_stride = w;
  in.source = source_.y.data() +
              static_cast<std::size_t>(y0 + tile_y_) *
                  static_cast<std::size_t>(source_.width) +
              static_cast<std::size_t>(x0 + tile_x_);
  in.source_stride = static_cast<int>(source_.width);
  in.inter_pred = inter_y.data();
  in.inter_pred_stride = w;
  in.enable_wedge_interintra = true;
  in.wedge_sign = 0;
  const std::optional<inter::adv::InterIntraDecision> decision =
      inter::adv::search_inter_intra(in);
  if (!decision.has_value()) {
    error = "inter-intra search failed";
    return false;
  }
  // Wedge inter-intra codes read_wedge_mode() (wedge_quad/angle/dist), so it
  // is kept only when it beats the plain §7.13.3.29 mask in SSE.
  inter::adv::InterIntraDecision chosen = *decision;
  if (decision->wedge_interintra) {
    in.enable_wedge_interintra = false;
    const std::optional<inter::adv::InterIntraDecision> plain =
        inter::adv::search_inter_intra(in);
    if (plain.has_value() && plain->sse <= decision->sse) {
      chosen = *plain;
    }
    in.enable_wedge_interintra = true;
  }
  const int mode = static_cast<int>(chosen.mode);
  const bool wedge = chosen.wedge_interintra;
  const int index = chosen.wedge_index;
  out_y.resize(inter_y.size());
  std::memcpy(out_y.data(), in.inter_pred, inter_y.size());
  inter::adv::CompoundJob job{};
  job.rect = inter::adv::BlockRect{0, 0, w, h};
  job.bsize = in.bsize;
  job.is_inter_intra = 1;
  job.interintra_mode = chosen.mode;
  job.type = wedge ? inter::adv::CompoundType::kWedge
                   : inter::adv::CompoundType::kIntra;
  job.wedge_index = index;
  job.wedge_sign = 0;
  job.intra = ii_y[mode].data();
  job.intra_stride = w;
  job.dst = out_y.data();
  job.dst_stride = w;
  inter::adv::run_compound_job(job);
  // Luma mask for the chroma wedge form (§7.13.3.30 subsamples it).
  std::vector<std::uint8_t> luma_mask;
  if (wedge) {
    luma_mask.resize(static_cast<std::size_t>(w) *
                     static_cast<std::size_t>(h));
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        luma_mask[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(j)] =
            static_cast<std::uint8_t>(inter::adv::wedge_mask_value(
                in.bsize, 0, index, i, j));
      }
    }
  }
  const int cw = w / 2;
  const int ch = h / 2;
  const auto blend_chroma = [&](const std::vector<std::uint8_t>& inter,
                                const std::vector<std::uint8_t>& intra_pred,
                                std::vector<std::uint8_t>& out) {
    out = inter;
    inter::adv::CompoundJob cjob{};
    cjob.plane = 1;
    cjob.sub_x = 1;
    cjob.sub_y = 1;
    cjob.rect = inter::adv::BlockRect{0, 0, cw, ch};
    cjob.bsize = in.bsize;
    cjob.is_inter_intra = 1;
    cjob.interintra_mode = chosen.mode;
    cjob.type = wedge ? inter::adv::CompoundType::kWedge
                      : inter::adv::CompoundType::kIntra;
    cjob.wedge_index = index;
    cjob.wedge_sign = 0;
    cjob.luma_mask = wedge ? luma_mask.data() : nullptr;
    cjob.luma_mask_stride = w;
    cjob.intra = intra_pred.data();
    cjob.intra_stride = cw;
    cjob.dst = out.data();
    cjob.dst_stride = cw;
    inter::adv::run_compound_job(cjob);
  };
  blend_chroma(inter_u, ii_u[mode], out_u);
  blend_chroma(inter_v, ii_v[mode], out_v);
  const std::int64_t ii_sse =
      sse_against_source(source_, 0, x0 + tile_x_, y0 + tile_y_, w, h,
                         out_y) +
      sse_against_source(source_, 1, x0 / 2 + tile_x_ / 2,
                         y0 / 2 + tile_y_ / 2, cw, ch, out_u) +
      sse_against_source(source_, 2, x0 / 2 + tile_x_ / 2,
                         y0 / 2 + tile_y_ / 2, cw, ch, out_v);
  mode_out = mode;
  wedge_out = wedge;
  index_out = index;
  sse_out = ii_sse;
  return true;
}

temporal::PlaneView InterFrameEncoder::temporal_plane(const inter::mc::RefFrame& ref,
                                          int plane) {
  temporal::PlaneView view;
  view.data = ref.plane[plane].data;
  view.stride = ref.plane[plane].stride;
  view.width = ref.plane_width(plane);
  view.height = ref.plane_height(plane);
  return view;
}

temporal::RefArea InterFrameEncoder::full_area(const temporal::PlaneView& view) {
  return temporal::RefArea{0, 0, view.width - 1, view.height - 1};
}

bool InterFrameEncoder::temporal_ref_area(int sub_x, int sub_y, int x, int y, int w, int h,
                       const inter::mc::Mv mvs[2],
                       temporal::RefArea out[2]) const {
  temporal::RefAreaInput in;
  in.w = w;
  in.h = h;
  in.sub_x = sub_x;
  in.sub_y = sub_y;
  in.x = x;
  in.y = y;
  in.frame_w = static_cast<int>(source_.width);
  in.frame_h = static_cast<int>(source_.height);
  in.num_refs = 2;
  for (int list = 0; list < 2; ++list) {
    in.ref_mi_cols[list] = references_[static_cast<std::size_t>(list)].mi_cols;
    in.ref_mi_rows[list] = references_[static_cast<std::size_t>(list)].mi_rows;
    in.ref_luma_w[list] = in.frame_w;
    in.ref_luma_h[list] = in.frame_h;
    in.cand_mvs[list] = temporal::Mv{mvs[list].y, mvs[list].x};
  }
  return temporal::compute_ref_area(in, out) ==
         temporal::TemporalStatus::kOk;
}

bool InterFrameEncoder::build_compound_block(const inter::mc::Mv mvs[2], int bsize, int x0,
                          int y0, int w, int h,
                          multiref::CompoundPrediction& prediction,
                          std::string& error, int ref0, int ref1) {
  return multiref::build_compound_prediction(
      references_[static_cast<std::size_t>(ref0)],
      references_[static_cast<std::size_t>(ref1)],
      inter::mc::BlockRect{x0, y0, w, h}, bsize, mvs[0], mvs[1],
      leaf_interp_filter(),
      references_[static_cast<std::size_t>(ref0)].mi_cols,
      references_[static_cast<std::size_t>(ref0)].mi_rows, prediction, error);
}

bool InterFrameEncoder::store_compound_block(const multiref::CompoundPrediction& prediction,
                          const multiref::CompoundModeParams& params,
                          int x0, int y0, int w, int h,
                          std::vector<std::uint8_t>& y,
                          std::vector<std::uint8_t>& u,
                          std::vector<std::uint8_t>& v, std::string& error) {
  std::uint8_t* dst[3] = {compound_pred_[0].samples.data(),
                          compound_pred_[1].samples.data(),
                          compound_pred_[2].samples.data()};
  int dst_stride[3] = {compound_pred_[0].stride, compound_pred_[1].stride,
                       compound_pred_[2].stride};
  if (!multiref::apply_compound_prediction(prediction, params, dst, dst_stride,
                                           error)) {
    return false;
  }
  const int cw = w / 2;
  const int ch = h / 2;
  const int cx0 = x0 / 2;
  const int cy0 = y0 / 2;
  y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  for (int yy = 0; yy < h; ++yy) {
    std::memcpy(y.data() + static_cast<std::size_t>(yy) *
                               static_cast<std::size_t>(w),
                compound_pred_[0].samples.data() +
                    static_cast<std::size_t>(y0 + yy) *
                        static_cast<std::size_t>(compound_pred_[0].stride) +
                    x0,
                static_cast<std::size_t>(w));
  }
  for (int yy = 0; yy < ch; ++yy) {
    std::memcpy(u.data() + static_cast<std::size_t>(yy) *
                               static_cast<std::size_t>(cw),
                compound_pred_[1].samples.data() +
                    static_cast<std::size_t>(cy0 + yy) *
                        static_cast<std::size_t>(compound_pred_[1].stride) +
                    cx0,
                static_cast<std::size_t>(cw));
    std::memcpy(v.data() + static_cast<std::size_t>(yy) *
                               static_cast<std::size_t>(cw),
                compound_pred_[2].samples.data() +
                    static_cast<std::size_t>(cy0 + yy) *
                        static_cast<std::size_t>(compound_pred_[2].stride) +
                    cx0,
                static_cast<std::size_t>(cw));
  }
  return true;
}

bool InterFrameEncoder::compound_plain(const inter::mc::Mv mvs[2], int bsize, int x0, int y0,
                    int w, int h, std::vector<std::uint8_t>& y,
                    std::vector<std::uint8_t>& u,
                    std::vector<std::uint8_t>& v, std::string& error,
                    const multiref::CompoundModeParams* mode_params,
                    int ref0, int ref1) {
  multiref::CompoundPrediction prediction;
  if (!build_compound_block(mvs, bsize, x0, y0, w, h, prediction, error,
                            ref0, ref1)) {
    return false;
  }
  multiref::CompoundModeParams params;
  if (mode_params != nullptr) {
    params = *mode_params;
  } else {
    params.type = inter::adv::CompoundType::kAverage;
    params.cwp_weight = multiref::kCwpEqual;
  }
  return store_compound_block(prediction, params, x0, y0, w, h, y, u, v,
                              error);
}

bool InterFrameEncoder::compound_search_average(const inter::mc::Mv mvs[2], int bsize, int x0,
                             int y0, int w, int h,
                             std::vector<std::uint8_t>& y,
                             std::vector<std::uint8_t>& u,
                             std::vector<std::uint8_t>& v,
                             multiref::CompoundModeParams& chosen,
                             std::int64_t& chosen_sse, std::string& error) {
  multiref::CompoundPrediction prediction;
  if (!build_compound_block(mvs, bsize, x0, y0, w, h, prediction, error)) {
    return false;
  }
  const bool same_side = (frame_distance_[0] >= 0) ==
                         (frame_distance_[1] >= 0);
  std::vector<std::uint8_t> scratch(
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  inter::adv::CompoundSearchInput in{};
  in.bsize = bsize;
  in.rect = inter::adv::BlockRect{0, 0, w, h};
  in.p0 = prediction.pred[0][0].samples.data();
  in.p0_stride = prediction.pred[0][0].stride;
  in.p1 = prediction.pred[1][0].samples.data();
  in.p1_stride = prediction.pred[1][0].stride;
  in.source = source_.y.data() +
              static_cast<std::size_t>(y0) *
                  static_cast<std::size_t>(source_.width) +
              static_cast<std::size_t>(x0);
  in.source_stride = static_cast<int>(source_.width);
  in.scratch = scratch.data();
  in.scratch_stride = w;
  in.enable_average = true;
  in.enable_cwp = options_.enable_cwp_weighted;
  in.enable_wedge = false;
  in.enable_diffwtd = false;
  in.enable_imp_mask = options_.enable_imp_msk_bld;
  in.imp_mask_eligible = options_.enable_imp_msk_bld;
  in.mi_cols = prediction.mi_cols;
  in.mi_rows = prediction.mi_rows;
  in.ref_start_x[0] = prediction.ref_start_x[0][0];
  in.ref_start_x[1] = prediction.ref_start_x[1][0];
  in.ref_start_y[0] = prediction.ref_start_y[0][0];
  in.ref_start_y[1] = prediction.ref_start_y[1][0];
  // The CWP rows are §5.20.7.6 Cwp_Weighting_Factor[is_same_side]; index 0
  // is CWP_EQUAL in both.  search_compound() takes the weight values.
  static const int kOppositeCwp[multiref::kMaxCwpNum] = {8, 12, 4, 10, 6};
  static const int kSameCwp[multiref::kMaxCwpNum] = {8, 12, 4, 20, -4};
  const int* weights = same_side ? kSameCwp : kOppositeCwp;
  for (int i = 0; i < multiref::kMaxCwpNum; ++i) {
    in.cwp_candidates[i] = weights[i];
  }
  in.cwp_count = options_.enable_cwp_weighted ? multiref::kMaxCwpNum : 0;
  // §7.13.3.14: when the implicit mask is eligible and CWP is off, the
  // decoder derives get_mask() for the CWP_EQUAL average and can never
  // produce the plain equal-weight blend.  search_compound() would still
  // evaluate that impossible candidate, so it is handled explicitly instead
  // (the same guard multiref::decide_compound() uses).
  std::optional<inter::adv::CompoundDecision> decision;
  if (options_.enable_imp_msk_bld && !options_.enable_cwp_weighted) {
    inter::adv::CompoundJob job{};
    job.plane = 0;
    job.bsize = bsize;
    job.bit_depth = 8;
    job.rect = inter::adv::BlockRect{0, 0, w, h};
    job.p0 = prediction.pred[0][0].samples.data();
    job.p0_stride = prediction.pred[0][0].stride;
    job.p1 = prediction.pred[1][0].samples.data();
    job.p1_stride = prediction.pred[1][0].stride;
    job.dst = scratch.data();
    job.dst_stride = w;
    job.type = inter::adv::CompoundType::kAverage;
    job.cwp_weight = multiref::kCwpEqual;
    job.imp_mask_eligible = 1;
    job.mi_cols = prediction.mi_cols;
    job.mi_rows = prediction.mi_rows;
    job.ref_start_x[0] = prediction.ref_start_x[0][0];
    job.ref_start_x[1] = prediction.ref_start_x[1][0];
    job.ref_start_y[0] = prediction.ref_start_y[0][0];
    job.ref_start_y[1] = prediction.ref_start_y[1][0];
    if (!inter::adv::apply_compound(job)) {
      error = "compound implicit-mask prediction failed";
      return false;
    }
    inter::adv::CompoundDecision d{};
    d.type = inter::adv::CompoundType::kAverage;
    d.cwp_weight = multiref::kCwpEqual;
    d.use_imp_mask = true;
    d.sse = inter::adv::block_sse(
        scratch.data(), w, 0, 0, in.source, in.source_stride, 0, 0, w, h);
    decision = d;
  } else {
    decision = inter::adv::search_compound(in);
  }
  if (!decision.has_value()) {
    error = "compound CWP/implicit-mask search failed";
    return false;
  }
  chosen = multiref::CompoundModeParams{};
  chosen.type = inter::adv::CompoundType::kAverage;
  chosen.cwp_weight = decision->cwp_weight;
  chosen.use_imp_mask = decision->use_imp_mask;
  if (!decision->use_imp_mask && decision->cwp_weight != multiref::kCwpEqual) {
    chosen.cwp_index =
        multiref::cwp_index_from_weight(decision->cwp_weight, same_side);
    if (chosen.cwp_index < 0) {
      error = "CWP weight is not signallable for this reference side";
      return false;
    }
  } else {
    chosen.cwp_index = 0;
  }
  chosen_sse = decision->sse;
  return store_compound_block(prediction, chosen, x0, y0, w, h, y, u, v,
                              error);
}

void InterFrameEncoder::collect_sub_pus(SbSink& sink, int x0, int y0, int w, int h,
                     bool optflow, bool refine) {
  const int cw = w / 2;
  const int ch = h / 2;
  const int cx0 = x0 / 2;
  const int cy0 = y0 / 2;
  const auto add = [&](int plane, int px, int py, int pw, int ph,
                       int pu_w, int pu_h) {
    post::VideoSubPuRect rect;
    rect.x = px;
    rect.y = py;
    rect.w = pw;
    rect.h = ph;
    rect.pu_w = pu_w;
    rect.pu_h = pu_h;
    rect.plane = plane;
    sink.sub_pus.push_back(rect);
  };
  if (optflow) {
    // §7.13.3.8: n = 4 for a <=8x8 block, else 8; chroma uses 4x4.
    const int n = (w <= 8 && h <= 8) ? 4 : 8;
    for (int i = 0; i < h; i += n) {
      for (int j = 0; j < w; j += n) {
        add(0, x0 + j, y0 + i, std::min(n, w - j), std::min(n, h - i), w,
            h);
      }
    }
    const int csub = 4;
    for (int plane = 1; plane < 3; ++plane) {
      for (int i = 0; i < ch; i += csub) {
        for (int j = 0; j < cw; j += csub) {
          add(plane, cx0 + j, cy0 + i, std::min(csub, cw - j),
              std::min(csub, ch - i), cw, ch);
        }
      }
    }
    return;
  }
  if (!refine) return;
  // §7.13.3.1: tw = Min(w, 16), th = Min(h, 16); chroma halves both.
  const int tw = std::min(w, 16);
  const int th = std::min(h, 16);
  const int twc = tw / 2;
  const int thc = th / 2;
  for (int i = 0; i < h; i += th) {
    for (int j = 0; j < w; j += tw) {
      add(0, x0 + j, y0 + i, std::min(tw, w - j), std::min(th, h - i), w,
          h);
    }
  }
  for (int plane = 1; plane < 3; ++plane) {
    for (int i = 0; i < ch; i += thc) {
      for (int j = 0; j < cw; j += twc) {
        add(plane, cx0 + j, cy0 + i, std::min(twc, cw - j),
            std::min(thc, ch - i), cw, ch);
      }
    }
  }
}

bool InterFrameEncoder::predict_compound_optflow(SbSink& sink, const inter::mc::Mv mvs[2],
                              int x0, int y0, int w, int h,
                              std::string& error) {
  const int fw = static_cast<int>(source_.width);
  const int fh = static_cast<int>(source_.height);
  const temporal::PlaneView l0 = temporal_plane(references_[0], 0);
  const temporal::PlaneView l1 = temporal_plane(references_[1], 0);
  const temporal::RefArea fa0 = full_area(l0);
  const temporal::RefArea fa1 = full_area(l1);
  std::vector<std::uint8_t> ip0(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
  std::vector<std::uint8_t> ip1(ip0.size());
  if (!temporal::predict_plane_u8_ex(
          l0, x0, y0, w, h, temporal::Mv{mvs[0].y, mvs[0].x},
          inter::mc::InterpFilter::kBilinear, 0, 0, false, fw, fh, l0.width,
          l0.height, fw, fh, fa0, ip0.data(), w) ||
      !temporal::predict_plane_u8_ex(
          l1, x0, y0, w, h, temporal::Mv{mvs[1].y, mvs[1].x},
          inter::mc::InterpFilter::kBilinear, 0, 0, false, fw, fh, l1.width,
          l1.height, fw, fh, fa1, ip1.data(), w)) {
    error = "optflow initial prediction failed";
    return false;
  }
  // NOTE: §7.13.3.8 has a `sad < sadThresh` shortcut that skips the
  // refinement and predicts with the coded MVs.  The pinned decoder
  // implements that check only for TIP blocks (`skip_opfl_refine_with_tip`);
  // for an ordinary B-frame compound block it always refines.  Taking the
  // shortcut here is observable: on the 640x360 pan fixture the coded MVs
  // are integer in luma (a refined block's luma is identical either way) but
  // half/quarter pel in chroma, where the plain EIGHTTAP prediction differs
  // from the 4-tap refined subblock prediction by 1 LSB.  Do not shortcut.
  const int n = (w <= 8 && h <= 8) ? 4 : 8;
  const int w4 = w / 4;
  const int h4 = h / 4;
  std::vector<std::int16_t> opfl(static_cast<std::size_t>(w4) *
                                 static_cast<std::size_t>(h4) * 4);
  std::vector<std::int16_t> delta(opfl.size());
  temporal::OpflJob job;
  job.w = w;
  job.h = h;
  job.p0 = ip0.data();
  job.p0_stride = w;
  job.p1 = ip1.data();
  job.p1_stride = w;
  job.mvs[0] = temporal::Mv{mvs[0].y, mvs[0].x};
  job.mvs[1] = temporal::Mv{mvs[1].y, mvs[1].x};
  // The product B frame does not carry the absolute order hint into this
  // class; frame_distance_ is relative_dist(OrderHint, RefOrderHint), so a
  // synthetic origin reproduces the §7.13.3.9 distances exactly.  The base
  // must avoid kRestrictedOrderHint (-1).
  job.order_hint = 64;
  job.ref0_order_hint = 64 - frame_distance_[0];
  job.ref1_order_hint = 64 - frame_distance_[1];
  job.n = n;
  temporal::OpflOutput out;
  out.opfl_mvs = opfl.data();
  out.opfl_stride4 = w4;
  out.mv_deltas = delta.data();
  out.delta_stride4 = w4;
  if (temporal::opfl_compute(job, out) != temporal::TemporalStatus::kOk) {
    error = "opfl_compute failed";
    return false;
  }
  const auto cell = [&](int row4, int col4, int list, int comp) {
    return static_cast<std::size_t>(
        ((static_cast<std::size_t>(row4) * static_cast<std::size_t>(w4) +
          static_cast<std::size_t>(col4)) *
             2 +
         static_cast<std::size_t>(list)) *
            2 +
        static_cast<std::size_t>(comp));
  };
  sink.compound_y.assign(static_cast<std::size_t>(w) *
                             static_cast<std::size_t>(h),
                         0);
  const int cw = w / 2;
  const int ch = h / 2;
  sink.compound_u.assign(static_cast<std::size_t>(cw) *
                             static_cast<std::size_t>(ch),
                         0);
  sink.compound_v.assign(sink.compound_u.size(), 0);
  std::vector<std::int16_t> s0(static_cast<std::size_t>(n) *
                               static_cast<std::size_t>(n));
  std::vector<std::int16_t> s1(s0.size());
  for (int i2 = 0; i2 < h; i2 += n) {
    for (int j2 = 0; j2 < w; j2 += n) {
      temporal::MvPair mvs_off;
      mvs_off.mv[0] = temporal::Mv{
          opfl[cell((i2 >> 2), (j2 >> 2), 0, 0)],
          opfl[cell((i2 >> 2), (j2 >> 2), 0, 1)]};
      mvs_off.mv[1] = temporal::Mv{
          opfl[cell((i2 >> 2), (j2 >> 2), 1, 0)],
          opfl[cell((i2 >> 2), (j2 >> 2), 1, 1)]};
      // §7.13.3.8 setRefArea for an 8x8 luma subblock.
      temporal::RefArea a0 = fa0;
      temporal::RefArea a1 = fa1;
      if (n == 8) {
        temporal::RefArea area[2];
        if (!temporal_ref_area(0, 0, x0 + j2, y0 + i2, n, n, mvs, area)) {
          error = "optflow ref area failed";
          return false;
        }
        a0 = area[0];
        a1 = area[1];
      }
      if (!temporal::predict_plane_1_16_ex(
              l0, x0 + j2, y0 + i2, n, n, mvs_off.mv[0],
              inter::mc::InterpFilter::kEighttapSharp, 0, 0, true, fw, fh,
              l0.width, l0.height, fw, fh, a0, s0.data(), n) ||
          !temporal::predict_plane_1_16_ex(
              l1, x0 + j2, y0 + i2, n, n, mvs_off.mv[1],
              inter::mc::InterpFilter::kEighttapSharp, 0, 0, true, fw, fh,
              l1.width, l1.height, fw, fh, a1, s1.data(), n)) {
        error = "optflow subblock prediction failed";
        return false;
      }
      temporal::compound_average(
          s0.data(), n, s1.data(), n, n, n, multiref::kCwpEqual,
          sink.compound_y.data() +
              static_cast<std::size_t>(i2) * static_cast<std::size_t>(w) +
              static_cast<std::size_t>(j2),
          w);
    }
  }
  // §7.13.3.7: luma filled OpflMvs; chroma reapplies them at 4x4 (plane>0).
  const int cx0 = x0 / 2;
  const int cy0 = y0 / 2;
  const int csub = 4;  // Max(4, n >> 1)
  std::vector<std::int16_t> cs0(16);
  std::vector<std::int16_t> cs1(16);
  for (int plane = 1; plane < 3; ++plane) {
    const temporal::PlaneView c0 = temporal_plane(references_[0], plane);
    const temporal::PlaneView c1 = temporal_plane(references_[1], plane);
    std::vector<std::uint8_t>& dst =
        plane == 1 ? sink.compound_u : sink.compound_v;
    for (int i2 = 0; i2 < ch; i2 += csub) {
      for (int j2 = 0; j2 < cw; j2 += csub) {
        const int row4 = i2 >> 1;
        const int col4 = j2 >> 1;
        temporal::MvPair mvs_off;
        mvs_off.mv[0] = temporal::Mv{opfl[cell(row4, col4, 0, 0)],
                                     opfl[cell(row4, col4, 0, 1)]};
        mvs_off.mv[1] = temporal::Mv{opfl[cell(row4, col4, 1, 0)],
                                     opfl[cell(row4, col4, 1, 1)]};
        temporal::RefArea area[2];
        if (!temporal_ref_area(1, 1, cx0 + j2, cy0 + i2, csub, csub, mvs,
                               area)) {
          error = "optflow chroma ref area failed";
          return false;
        }
        if (!temporal::predict_plane_1_16_ex(
                c0, cx0 + j2, cy0 + i2, csub, csub, mvs_off.mv[0],
                inter::mc::InterpFilter::kEighttapSharp, 1, 1, true, fw, fh,
                c0.width, c0.height, fw, fh, area[0], cs0.data(), csub) ||
            !temporal::predict_plane_1_16_ex(
                c1, cx0 + j2, cy0 + i2, csub, csub, mvs_off.mv[1],
                inter::mc::InterpFilter::kEighttapSharp, 1, 1, true, fw, fh,
                c1.width, c1.height, fw, fh, area[1], cs1.data(), csub)) {
          error = "optflow chroma prediction failed";
          return false;
        }
        temporal::compound_average(
            cs0.data(), csub, cs1.data(), csub, csub, csub,
            multiref::kCwpEqual,
            dst.data() + static_cast<std::size_t>(i2) *
                             static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(j2),
            cw);
      }
    }
  }
  return true;
}

bool InterFrameEncoder::predict_compound_refinemv(SbSink& sink, const inter::mc::Mv mvs[2],
                               int x0, int y0, int w, int h,
                               std::string& error) {
  const int fw = static_cast<int>(source_.width);
  const int fh = static_cast<int>(source_.height);
  const temporal::PlaneView l0 = temporal_plane(references_[0], 0);
  const temporal::PlaneView l1 = temporal_plane(references_[1], 0);
  const temporal::PlaneView u0 = temporal_plane(references_[0], 1);
  const temporal::PlaneView u1 = temporal_plane(references_[1], 1);
  const temporal::PlaneView v0 = temporal_plane(references_[0], 2);
  const temporal::PlaneView v1 = temporal_plane(references_[1], 2);
  const int tw = std::min(w, 16);
  const int th = std::min(h, 16);
  const int cols = (w + tw - 1) / tw;
  const int rows = (h + th - 1) / th;
  const int cw = w / 2;
  const int ch = h / 2;
  const int twc = tw / 2;
  const int thc = th / 2;

  std::vector<temporal::MvPair> sb_mvs(static_cast<std::size_t>(rows) *
                                       static_cast<std::size_t>(cols));
  std::vector<std::uint8_t> s0(static_cast<std::size_t>(tw + 4) *
                               static_cast<std::size_t>(th + 4));
  std::vector<std::uint8_t> s1(s0.size());
  std::vector<std::int16_t> r0(static_cast<std::size_t>(tw) *
                               static_cast<std::size_t>(th));
  std::vector<std::int16_t> r1(r0.size());
  const auto predict_plane_refine = [&](int sub_x, int sub_y,
                                        const temporal::PlaneView& p0,
                                        const temporal::PlaneView& p1, int px,
                                        int py, int pw, int ph,
                                        const temporal::RefArea a[2],
                                        const temporal::MvPair& mvs_off,
                                        std::uint8_t* dst,
                                        int dst_stride) {
    temporal::RefinePredictInput in;
    in.ref[0] = p0;
    in.ref[1] = p1;
    in.num_refs = 2;
    in.x = px;
    in.y = py;
    in.w = pw;
    in.h = ph;
    in.sub_x = sub_x;
    in.sub_y = sub_y;
    in.frame_w = fw;
    in.frame_h = fh;
    for (int list = 0; list < 2; ++list) {
      in.ref_luma_w[list] = fw;
      in.ref_luma_h[list] = fh;
    }
    in.offset_mvs = mvs_off;
    in.cwp_weight = multiref::kCwpEqual;
    in.use_ref_area = true;
    in.area[0] = a[0];
    in.area[1] = a[1];
    in.dst = dst;
    in.dst_stride = dst_stride;
    in.scratch0 = r0.data();
    in.scratch1 = r1.data();
    in.scratch_stride = pw;
    return temporal::refine_predict_block(in) ==
           temporal::TemporalStatus::kOk;
  };

  sink.compound_y.assign(static_cast<std::size_t>(w) *
                             static_cast<std::size_t>(h),
                         0);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int x = x0 + c * tw;
      const int y = y0 + r * th;
      temporal::RefArea area[2];
      if (!temporal_ref_area(0, 0, x, y, tw, th, mvs, area)) {
        error = "refine-MV ref area failed";
        return false;
      }
      temporal::RefineSearchInput search;
      search.x = x;
      search.y = y;
      search.w = tw;
      search.h = th;
      search.frame_w = fw;
      search.frame_h = fh;
      for (int list = 0; list < 2; ++list) {
        search.ref_mi_cols[list] =
            references_[static_cast<std::size_t>(list)].mi_cols;
        search.ref_mi_rows[list] =
            references_[static_cast<std::size_t>(list)].mi_rows;
        search.ref_luma_w[list] = fw;
        search.ref_luma_h[list] = fh;
        search.cand_mvs[list] = temporal::Mv{mvs[list].y, mvs[list].x};
      }
      search.ref[0] = l0;
      search.ref[1] = l1;
      search.num_refs = 2;
      search.tip_pred = false;
      // §5.20.7.17: NEAR_NEARMV is not switchable.
      search.switchable = false;
      search.use_ref_area = true;
      search.area[0] = area[0];
      search.area[1] = area[1];
      search.scratch0 = s0.data();
      search.scratch1 = s1.data();
      search.scratch_stride = tw + 4;
      temporal::RefineSearchResult result;
      if (temporal::search_refinemv(search, &result) !=
          temporal::TemporalStatus::kOk) {
        error = "search_refinemv failed";
        return false;
      }
      sb_mvs[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
             static_cast<std::size_t>(c)] = result.offset_mvs;
      if (!predict_plane_refine(
              0, 0, l0, l1, x, y, tw, th, area, result.offset_mvs,
              sink.compound_y.data() +
                  static_cast<std::size_t>(r * th) *
                          static_cast<std::size_t>(w) +
                      static_cast<std::size_t>(c * tw),
              w)) {
        error = "refine-MV luma prediction failed";
        return false;
      }
    }
  }

  sink.compound_u.assign(static_cast<std::size_t>(cw) *
                             static_cast<std::size_t>(ch),
                         0);
  sink.compound_v.resize(sink.compound_u.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int x = c * twc;
      const int y = r * thc;
      temporal::RefArea carea[2];
      if (!temporal_ref_area(1, 1, x0 / 2 + x, y0 / 2 + y, twc, thc, mvs,
                             carea)) {
        error = "refine-MV chroma ref area failed";
        return false;
      }
      const temporal::MvPair& mvs_off =
          sb_mvs[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(c)];
      if (!predict_plane_refine(
              1, 1, u0, u1, x0 / 2 + x, y0 / 2 + y, twc, thc, carea, mvs_off,
              sink.compound_u.data() +
                  static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                  static_cast<std::size_t>(x),
              cw) ||
          !predict_plane_refine(
              1, 1, v0, v1, x0 / 2 + x, y0 / 2 + y, twc, thc, carea, mvs_off,
              sink.compound_v.data() +
                  static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                  static_cast<std::size_t>(x),
              cw)) {
        error = "refine-MV chroma prediction failed";
        return false;
      }
    }
  }
  return true;
}

void InterFrameEncoder::update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                    int ref_index, inter::mc::Mv mv,
                    int ref_index1,
                    const inter::mc::Mv* mv1,
                    bool warp0, bool warp1,
                    bool is_wedge, int wedge_index,
                    int wedge_sign, int bsize) {
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = r + yy;
    if (rr >= mi_rows_) break;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = c + xx;
      if (cc >= mi_cols_) break;
      inter::mc::MiInfo& mi =
          mv_grid_[static_cast<std::size_t>(rr) *
                   static_cast<std::size_t>(mi_cols_) +
               static_cast<std::size_t>(cc)];
      mi = inter::mc::MiInfo{};
      mi.ref[0] = is_inter ? static_cast<std::int8_t>(ref_index)
                           : kRefIntraAv2;
      mi.ref[1] = static_cast<std::int8_t>(ref_index1);
      mi.mv[0][0] = mv.y;
      mi.mv[0][1] = mv.x;
      mi.mv[1][0] = mv1 != nullptr ? mv1->y : mv.y;
      mi.mv[1][1] = mv1 != nullptr ? mv1->x : mv.x;
      mi.is_inter = static_cast<std::int8_t>(is_inter);
      mi.w4 = static_cast<std::int8_t>(w4);
      mi.h4 = static_cast<std::int8_t>(h4);
      mi.col_base = static_cast<std::int16_t>(c);
      mi.row_base = static_cast<std::int16_t>(r);
      mi.warp[0] = static_cast<std::int8_t>(warp0 && is_inter ? 1 : 0);
      mi.warp[1] = static_cast<std::int8_t>(warp1 && is_inter ? 1 : 0);
      mi.is_wedge = static_cast<std::int8_t>(is_wedge && is_inter ? 1 : 0);
      mi.wedge_index = static_cast<std::int8_t>(wedge_index);
      mi.wedge_sign = static_cast<std::int8_t>(wedge_sign);
      mi.bsize = static_cast<std::int16_t>(bsize);
    }
  }
}

// Lane-2 frame-horizon prediction tool: integer-MV copy with the reference
// extent clamp.  The decoder's §7.13.3.1 MC at MV = 0 reads the reference at
// the same position and clips every tap to the reference picture bounds, so a
// block whose coded extent runs past the MiRows/MiCols edge replicates the
// last row/column.  This primitive is the whole prediction for a zero-MV leaf
// (GLOBALMV), including the edge case, and is shared only with the new
// horizon pipeline.
void horizon_predict_zero_mv(const std::uint8_t* ref, int ref_stride,
                             int ref_width, int ref_height, int x0, int y0,
                             int w, int h, std::uint8_t* out) {
  if (ref == nullptr || out == nullptr || ref_width <= 0 || ref_height <= 0 ||
      w <= 0 || h <= 0) {
    return;
  }
  for (int y = 0; y < h; ++y) {
    const int sy = std::clamp(y0 + y, 0, ref_height - 1);
    const std::uint8_t* src_row =
        ref + static_cast<std::size_t>(sy) * static_cast<std::size_t>(ref_stride);
    std::uint8_t* dst_row = out + static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(w);
    for (int x = 0; x < w; ++x) {
      const int sx = std::clamp(x0 + x, 0, ref_width - 1);
      dst_row[x] = src_row[sx];
    }
  }
}

bool horizon_predict_mv(const HorizonRefPlanes& ref, int x0, int y0, int w,
                        int h, int mv_y, int mv_x, std::uint8_t* out[3]) {
  if (out == nullptr || out[0] == nullptr || out[1] == nullptr ||
      out[2] == nullptr) {
    return false;
  }
  if (ref.width <= 0 || ref.height <= 0 || (ref.width & 3) != 0 ||
      (ref.height & 3) != 0) {
    return false;
  }
  if (w <= 0 || h <= 0 || (w & 1) != 0 || (h & 1) != 0) return false;
  if (x0 < 0 || y0 < 0 || (x0 & 1) != 0 || (y0 & 1) != 0) return false;
  if (x0 + w > ref.width || y0 + h > ref.height) return false;
  inter::mc::RefFrame frame;
  for (int plane = 0; plane < 3; ++plane) {
    if (ref.data[plane] == nullptr || ref.stride[plane] <= 0) return false;
    frame.plane[plane].data = ref.data[plane];
    frame.plane[plane].stride = ref.stride[plane];
  }
  frame.mi_cols = ref.width / inter::mc::kMiSize;
  frame.mi_rows = ref.height / inter::mc::kMiSize;
  inter::mc::MutablePlane dst[3];
  dst[0] = inter::mc::MutablePlane{out[0], w};
  dst[1] = inter::mc::MutablePlane{out[1], w / 2};
  dst[2] = inter::mc::MutablePlane{out[2], w / 2};
  const inter::mc::BlockRect rect{x0, y0, w, h};
  const inter::mc::McStatus status = inter::mc::predict_block_420(
      frame, rect, inter::mc::Mv{mv_y, mv_x},
      inter::mc::InterpFilter::kEighttap, dst);
  return status == inter::mc::McStatus::kOk;
}

}  // namespace av2enc::video
