#include "pipeline/video/video_frame_internal.h"

namespace av2enc::video {

InterFrameEncoder::InterFrameEncoder(const VideoInputFrame& source,
                  std::vector<inter::mc::RefFrame> references,
                  std::array<int, 7> frame_distance, int num_total_refs,
                  const VideoOptions& options, bool trace, bool keyframe,
                  int tile_x, int tile_y, int tile_w,
                  int tile_h, std::span<const std::uint16_t> cdf_seed,
                  std::vector<std::uint16_t>* cdf_final,
                  const GmFramePlan* gm_plan,
                  std::array<int, 7> ref_order_hints,
                  bool primary_ref_is_none,
                  const TemporalFrameContext* temporal,
                  std::span<const std::uint8_t> prev_segment_ids)
    : source_(source),
      references_(std::move(references)),
      frame_distance_(frame_distance),
      num_total_refs_(num_total_refs),
      options_(options),
      trace_(trace),
      keyframe_(keyframe),
      tile_x_(tile_x),
      tile_y_(tile_y),
      tile_w_(tile_w),
      tile_h_(tile_h),
      cdf_seed_(cdf_seed),
      cdf_final_(cdf_final),
      gm_plan_(gm_plan),
      ref_order_hints_(ref_order_hints) {
#if defined(AV2ENC_VIDEO_DELTAQ)
  primary_ref_is_none_ = primary_ref_is_none;
  prev_segment_ids_.assign(prev_segment_ids.begin(), prev_segment_ids.end());
#else
  (void)primary_ref_is_none;
  (void)prev_segment_ids;
#endif
  if (temporal != nullptr) temporal_ctx_ = *temporal;
  // Coded (tile-local) geometry.  A zero tile_w/tile_h means "whole frame",
  // which keeps the historical single-tile path untouched.  Source reads and
  // motion-compensated predictions add tile_x_/tile_y_ so the reference frame
  // stays global, exactly as the decoder's tile-local walk sees it.
  width_ = tile_w > 0 ? tile_w : static_cast<int>(source.width);
  height_ = tile_h > 0 ? tile_h : static_cast<int>(source.height);
  tile_mi_row_ = tile_y_ / 4;
  tile_mi_col_ = tile_x_ / 4;
  chroma_width_ = width_ / 2;
  chroma_height_ = height_ / 2;

  geometry_.width = width_;
  geometry_.height = height_;
  // §5.18.1: use_256x256_superblock clamps an intra frame's coding SbSize to
  // 128x128; only inter frames get 256x256 roots.
  sb_size_ = (keyframe_ && options_.sb256)
                 ? 128
                 : options_.superblock_pixels();
  geometry_.sb_size = sb_size_;
  // §5.4.3 MaxPbAspectRatio: 8 unless reduce_pb_aspect_ratio selects the
  // 2:1 or 4:1 form.  The partition search and the §6.19.3.2 validation must
  // agree with the sequence header the decoder reads.
  geometry_.max_pb_aspect_ratio =
      options_.reduce_pb_aspect_ratio
          ? (1 << (options_.max_pb_aspect_ratio_log2_minus_1 + 1))
          : 8;
  geometry_.has_chroma = true;
  geometry_.chroma_subsampling_x = true;
  geometry_.chroma_subsampling_y = true;
  mi_cols_ = partition::mi_cols(geometry_);
  mi_rows_ = partition::mi_rows(geometry_);
  frame_mi_cols_ = static_cast<int>(2U * ((source.width + 7U) >> 3U));
  frame_mi_rows_ = static_cast<int>(2U * ((source.height + 7U) >> 3U));

  // Opt-in bottom-edge inter (--video-edge-inter).  A bottom leaf may code a
  // block whose coded extent runs past MiRows*4 (the partition rule only
  // requires its midpoint to be inside the picture).  The decoder
  // reconstructs the full block into its SB-aligned picture buffer and pads
  // the reference from the visible edge, so the encoder reconstructs into a
  // plane extended to the next superblock row.  Tiled coding keeps the
  // historical tile-local path.
  if (options_.enable_edge_inter &&
      width_ == static_cast<int>(source.width) &&
      height_ == static_cast<int>(source.height)) {
    const int base = mi_rows_ * 4;
    edge_pad_rows_ =
        ((base + sb_size_ - 1) / sb_size_) * sb_size_;
  }
  const int recon_rows =
      edge_pad_rows_ > 0 ? edge_pad_rows_ : mi_rows_ * 4;
  recon_[0].resize(mi_cols_ * 4, recon_rows, 0);
  recon_[1].resize(mi_cols_ * 2, recon_rows / 2, 0);
  recon_[2].resize(mi_cols_ * 2, recon_rows / 2, 0);
  compound_pred_[0].resize(mi_cols_ * 4, recon_rows, 0);
  compound_pred_[1].resize(mi_cols_ * 2, recon_rows / 2, 0);
  compound_pred_[2].resize(mi_cols_ * 2, recon_rows / 2, 0);

  quant_step_ = frame_quant_step(options_);
  current_qindex_ = static_cast<int>(options_.base_q_idx);
  uv_delta_enabled_ = options_.enable_uv_delta_q;
  resolved_delta_q_ = dcdeltaq::resolve_delta_q(options_.seq_delta_q,
                                                options_.frame_delta_q);
  frame_plane_quant(current_qindex_, plane_quant_, quant_step_);
  if (trace_ && uv_delta_enabled_) {
    std::cerr << "uvdq q=" << current_qindex_
              << " base_uv_ac=" << options_.seq_delta_q.base_uv_ac_delta_q
              << " base_uv_dc=" << options_.seq_delta_q.base_uv_dc_delta_q
              << " u_dc=" << resolved_delta_q_.u_dc
              << " v_dc=" << resolved_delta_q_.v_dc
              << " u_ac=" << resolved_delta_q_.u_ac
              << " v_ac=" << resolved_delta_q_.v_ac
              << " u_dc_q=" << plane_quant_[1].dc
              << " u_ac_q=" << plane_quant_[1].ac
              << " v_dc_q=" << plane_quant_[2].dc
              << " v_ac_q=" << plane_quant_[2].ac << "\n";
  }
  // §5.18.6.2 default-matrix QM (--video-qm): one picture-level set, level 0
  // (the spec-defined default matrices; no user QM OBU).
  if (options_.enable_qm) {
    qm_state_.using_qmatrix = true;
    qm_state_.pic_qm_num_minus_1 = 0;
    qm_state_.qm_y[0] = 0;
    qm_state_.qm_u[0] = 0;
    qm_state_.qm_v[0] = 0;
    for (int plane = 0; plane < 3; ++plane) {
      qm_state_.seg_qm_level[plane][0] = 0;
    }
    if (options_.user_qm) {
      // §5.4.11 user-defined matrix for level 0.  The all-32 matrix is the
      // identity (Round2(q * 32, 5) == q), so the encoder dequant follows
      // the same state the decoder loads from the QM OBU.
      qm_state_.qm_data_present[0] = true;
      qm::fill_user_qm_constant(&qm_state_.user_qm[0], 32);
    }
    qm_enabled_ = true;
  }
  mv_grid_.assign(static_cast<std::size_t>(mi_rows_) *
                      static_cast<std::size_t>(mi_cols_),
                  inter::mc::MiInfo{});
  // §5.20.5.5 IntraJointMode per 4x4 cell, used only by the opt-in full
  // intra search to derive §7.13.2.15/.16 filterTypeAbove/Left.  Inter
  // leaves write 0, exactly like the syntax writer's inter_block_mode_info().
  intra_joint_.assign(static_cast<std::size_t>(mi_rows_) *
                          static_cast<std::size_t>(mi_cols_),
                      0);
  // §6.19.2.3 BlockDecoded maps are superblock-local and now live in the
  // per-superblock sink state (SbSink::State::decoded), so there is no
  // frame-level map to size here.
  // §5.20.5.3/.4 IBC + §7.13.3.26 morph: only the intra keyframe carries
  // the screen-content mode-info path.  The BV ring/neighbour map is
  // advanced for every leaf, IBC or not.  `--video-ibc` enables the same IBC
  // mode-info path without morph_pred; `--video-palette` enables palette
  // selection on the same keyframe.
  plan_morph_ = keyframe_ && options_.enable_morph;
  ibc_enabled_ = keyframe_ && (options_.enable_morph || options_.enable_ibc);
  palette_enabled_ = keyframe_ && options_.enable_palette;
  if (ibc_enabled_) {
    ibc_buffer_.reset(mi_cols_, mi_rows_);
    ibc_neighbours_.reset(mi_cols_, mi_rows_);
    ibc_config_ = ibc::IbcConfig{};
    ibc_config_.sb_size = geometry_.sb_size;
    ibc_config_.allow_global = true;
    ibc_config_.allow_local = false;
    ibc_config_.max_bvp_drl_bits_minus_1 = 0;
  }
  if (palette_enabled_) {
    palette_state_.reset(mi_cols_, mi_rows_);
  }
}

std::int64_t InterFrameEncoder::block_variance_cost_at(
    const partition::PartitionNodeInfo& info) const {
  const partition::BlockRect rect = partition::block_rect(
      geometry_, info.mi_row, info.mi_col, info.size);
  const int w = rect.visible_width;
  const int h = rect.visible_height;
  if (w <= 0 || h <= 0) return 0;
#if defined(AV2ENC_PRIMITIVES)
  // `rect` is clipped to this tile's visible extent, so the absolute origin
  // is inside the source plane and source_sample()'s clamp never fires for an
  // in-frame partition node.  The kernel keeps the same int64 sum, binary64
  // mean/SSE order and int64 truncation (-ffp-contract=off, as required).
  const int stride = static_cast<int>(source_.width);
  const std::uint8_t* src =
      source_.y.data() + static_cast<std::size_t>(tile_y_ + rect.y) *
                             static_cast<std::size_t>(stride) +
      static_cast<std::size_t>(tile_x_ + rect.x);
  return primitives::kernels().variance(src, static_cast<std::size_t>(stride),
                                        w, h);
#else
  std::int64_t sum = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      sum += source_sample(source_, 0, tile_x_ + rect.x + x,
                           tile_y_ + rect.y + y);
    }
  }
  const double mean = static_cast<double>(sum) /
                      static_cast<double>(w * h);
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double d =
          static_cast<double>(source_sample(source_, 0, tile_x_ + rect.x + x,
                                            tile_y_ + rect.y + y)) -
          mean;
      sse += d * d;
    }
  }
  return static_cast<std::int64_t>(sse);
#endif
}

std::uint64_t InterFrameEncoder::partition_probe_key(int mi_row, int mi_col,
                                         partition::BlockSize size) {
  return (static_cast<std::uint64_t>(mi_row) & 0xFFFFU) |
         ((static_cast<std::uint64_t>(mi_col) & 0xFFFFU) << 16U) |
         ((static_cast<std::uint64_t>(size) & 0xFFU) << 32U);
}

bool InterFrameEncoder::partition_probe_fits(int mi_row, int mi_col,
                          partition::BlockSize size) const {
  const int bw = partition::block_width(size);
  const int bh = partition::block_height(size);
  if (bw < 8 || bh < 8 || bw > 64 || bh > 64) return false;
  if (width_ != static_cast<int>(source_.width) ||
      height_ != static_cast<int>(source_.height)) {
    return false;
  }
  const int x0 = mi_col * 4;
  const int y0 = mi_row * 4;
  if (x0 + bw > mi_cols_ * 4) return false;
  if (y0 + bh > mi_rows_ * 4 &&
      !(edge_pad_rows_ > 0 && y0 < mi_rows_ * 4)) {
    return false;
  }
  return true;
}

bool InterFrameEncoder::partition_probe_leaf_cost(const partition::PartitionNodeInfo& info,
                               double lambda, double& cost) const {
  if (!partition_probe_fits(info.mi_row, info.mi_col, info.size)) {
    return false;
  }
  const int bw = partition::block_width(info.size);
  const int bh = partition::block_height(info.size);
  const int ax0 = info.mi_col * 4 + tile_x_;
  const int ay0 = info.mi_row * 4 + tile_y_;
  std::string probe_error;
  inter::mc::Mv mv{};
  std::vector<std::uint8_t> pred;
  if (!search_me(0, ax0, ay0, bw, bh, mv, probe_error) ||
      !predict_luma(0, mv, ax0, ay0, bw, bh, pred)) {
    return false;
  }
  const double pred_sse = static_cast<double>(
      sse_against_source(source_, 0, ax0, ay0, bw, bh, pred));
  double best = pred_sse + lambda * kPartitionProbeSkipBits;
  TxPlan plan;
  if (compute_tx(source_, 0, ax0, ay0, bw, bh, pred, plane_quant_[0], plan,
                 probe_error, /*qm_state=*/nullptr, /*allow_tcq=*/false,
                 /*qctx_slice=*/0, /*is_inter=*/true)) {
    const double res_sse = static_cast<double>(
        sse_against_source(source_, 0, ax0, ay0, bw, bh, plan.recon));
    const double bits = kPartitionProbeModeBits + luma_plan_bits(plan, true);
    best = std::min(best, res_sse + lambda * bits);
  }
  cost = best;
  return true;
}

void InterFrameEncoder::collect_partition_probe_jobs(
    int mi_row, int mi_col, partition::BlockSize size, int depth,
    int max_depth, int floor_px,
    std::vector<PartitionProbeJob>& jobs) const {
  jobs.push_back(PartitionProbeJob{mi_row, mi_col, size});
  if (depth >= max_depth) return;
  const int w4 = partition::num_4x4_blocks_wide(size);
  const int h4 = partition::num_4x4_blocks_high(size);
  const auto visit = [&](int r, int c, partition::BlockSize child) {
    if (!partition::is_inside(geometry_, r, c)) return;
    if (partition::block_width(child) < floor_px ||
        partition::block_height(child) < floor_px) {
      return;
    }
    collect_partition_probe_jobs(r, c, child, depth + 1, max_depth,
                                 floor_px, jobs);
  };
  const partition::BlockSize horz =
      partition::partition_subsize(partition::PartitionType::kHorz, size);
  if (partition::is_valid_block_size(horz)) {
    visit(mi_row, mi_col, horz);
    visit(mi_row + h4 / 2, mi_col, horz);
  }
  const partition::BlockSize vert =
      partition::partition_subsize(partition::PartitionType::kVert, size);
  if (partition::is_valid_block_size(vert)) {
    visit(mi_row, mi_col, vert);
    visit(mi_row, mi_col + w4 / 2, vert);
  }
  const partition::BlockSize split =
      partition::partition_subsize(partition::PartitionType::kSplit, size);
  if (partition::is_valid_block_size(split)) {
    visit(mi_row, mi_col, split);
    visit(mi_row, mi_col + w4 / 2, split);
    visit(mi_row + h4 / 2, mi_col, split);
    visit(mi_row + h4 / 2, mi_col + w4 / 2, split);
  }
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  // The extended solver prices its 3-way / uneven 4-way children with this
  // same probe, so the precompute pass has to visit their geometry too.  The
  // child list comes from the spec's decode_partition() enumeration, so the
  // probe keys are exactly the (mi_row, mi_col, size) the solver will ask
  // for.  Partitions the solver rejects (min-block floor, chroma offset,
  // 4-wide vertical) simply never look their children up.
  if (options_.ext_partitions) {
    constexpr pe::PartitionType kExtTypes[] = {
        pe::PartitionType::kHorz3,  pe::PartitionType::kVert3,
        pe::PartitionType::kHorz4A, pe::PartitionType::kHorz4B,
        pe::PartitionType::kVert4A, pe::PartitionType::kVert4B};
    for (pe::PartitionType type : kExtTypes) {
      pe::ChildSlot slots[4];
      const int count = pe::enumerate_children(
          geometry_, mi_row, mi_col, size, type, /*has_chroma=*/true,
          /*chroma_offset=*/false, pe::ChromaBlock{mi_row, mi_col, size},
          slots);
      for (int i = 0; i < count; ++i) {
        if (!pe::child_is_coded(geometry_, slots[i])) continue;
        visit(slots[i].mi_row, slots[i].mi_col,
              static_cast<partition::BlockSize>(slots[i].size));
      }
    }
  }
#endif
}

std::unordered_map<std::uint64_t, double> InterFrameEncoder::build_partition_probe_costs(
    int max_depth, int floor_px, double lambda) const {
  std::vector<PartitionProbeJob> jobs;
  const int sb_mi = partition::sb_size_mi(geometry_);
  const int cols = partition::sb_cols(geometry_);
  const int rows = partition::sb_rows(geometry_);
  for (int sby = 0; sby < rows; ++sby) {
    for (int sbx = 0; sbx < cols; ++sbx) {
      const int mi_row = sby * sb_mi;
      const int mi_col = sbx * sb_mi;
      if (!partition::is_inside(geometry_, mi_row, mi_col)) continue;
      collect_partition_probe_jobs(
          mi_row, mi_col, partition::superblock_size(geometry_.sb_size), 0,
          max_depth, floor_px, jobs);
    }
  }
  std::vector<double> costs(jobs.size(), partition::kInfiniteCost);
  std::vector<char> modeled(jobs.size(), 0);
  const auto run = [&](std::size_t i) {
    partition::PartitionNodeInfo info;
    info.mi_row = jobs[i].mi_row;
    info.mi_col = jobs[i].mi_col;
    info.size = jobs[i].size;
    double cost = partition::kInfiniteCost;
    if (partition_probe_leaf_cost(info, lambda, cost)) {
      costs[i] = cost;
      modeled[i] = 1;
    }
  };
  const unsigned workers = parallel::resolve_workers(options_.cpu_threads);
  if (workers > 1 && jobs.size() > 1) {
    parallel::parallel_for(jobs.size(), workers, run);
  } else {
    for (std::size_t i = 0; i < jobs.size(); ++i) run(i);
  }
  std::unordered_map<std::uint64_t, double> out;
  out.reserve(jobs.size());
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    if (!modeled[i]) continue;
    out.emplace(partition_probe_key(jobs[i].mi_row, jobs[i].mi_col,
                                    jobs[i].size),
                costs[i]);
  }
  return out;
}

int InterFrameEncoder::visible_rows(int plane, int y0, int h) const {
  if (!options_.enable_edge_mask || edge_pad_rows_ <= 0) return h;
  const int vis_h = plane == 0 ? height_ : (height_ + 1) / 2;
  const int rows = vis_h - y0;
  if (rows >= h) return h;
  return rows > 0 ? rows : 0;
}

int InterFrameEncoder::qp_sb_mi() const { return partition::sb_size_mi(geometry_); }

int InterFrameEncoder::qp_sb_cols() const { return (mi_cols_ + qp_sb_mi() - 1) / qp_sb_mi(); }

int InterFrameEncoder::qp_sb_rows() const { return (mi_rows_ + qp_sb_mi() - 1) / qp_sb_mi(); }

int InterFrameEncoder::qp_sb_count() const { return qp_sb_cols() * qp_sb_rows(); }

int InterFrameEncoder::base_q_index() const { return static_cast<int>(options_.base_q_idx); }

bool InterFrameEncoder::qp_tools_active() const {
  return options_.enable_segmentation || options_.enable_delta_q ||
         options_.per_segment_q;
}

bool InterFrameEncoder::build_sb_maps(std::string& error) {
  const int cols = qp_sb_cols();
  const int rows = qp_sb_rows();
  const int count = cols * rows;
  if (count <= 0) {
    error = "delta-Q: empty superblock grid";
    return false;
  }
  sb_segment_.assign(static_cast<std::size_t>(count), 0);
  sb_target_q_.assign(static_cast<std::size_t>(count), base_q_index());
  std::vector<std::uint64_t> variance(static_cast<std::size_t>(count), 0);
  std::uint64_t total = 0;
  for (int sby = 0; sby < rows; ++sby) {
    for (int sbx = 0; sbx < cols; ++sbx) {
      const int x0 = sbx * sb_size_;
      const int y0 = sby * sb_size_;
      const int w = std::min(sb_size_, width_ - x0);
      const int h = std::min(sb_size_, height_ - y0);
      std::uint64_t sum = 0;
      std::uint64_t sumsq = 0;
      for (int y = 0; y < h; ++y) {
        const std::uint8_t* row =
            source_.y.data() +
            static_cast<std::size_t>(tile_y_ + y0 + y) * source_.width +
            static_cast<std::size_t>(tile_x_ + x0);
        for (int x = 0; x < w; ++x) {
          const std::uint64_t v = row[x];
          sum += v;
          sumsq += v * v;
        }
      }
      const std::uint64_t n =
          static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
      const std::uint64_t var =
          n == 0 ? 0 : (sumsq * n - sum * sum) / (n * n);
      variance[static_cast<std::size_t>(sby * cols + sbx)] = var;
      total += var;
    }
  }
  const std::uint64_t mean =
      total / static_cast<std::uint64_t>(count);
  const int delta = options_.seg_q_delta;
  const int num_segments = video_num_segments(options_);
  if (num_segments == 2) {
    // Historical planner: segment 1 is every superblock at or below the
    // frame mean variance.  Preserve it byte-for-byte for the gated cases.
    for (int i = 0; i < count; ++i) {
      const bool low = variance[static_cast<std::size_t>(i)] <= mean;
      sb_segment_[static_cast<std::size_t>(i)] = low ? 1 : 0;
    }
  } else if (num_segments > 2) {
    // Deterministic variance rank split: rank 0 is the flattest superblock
    // and maps to the highest segment id, so the skip/globalmv features that
    // occupy the last ids see the low-variance buckets.
    std::vector<int> order(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const std::uint64_t va = variance[static_cast<std::size_t>(a)];
      const std::uint64_t vb = variance[static_cast<std::size_t>(b)];
      if (va != vb) return va < vb;
      return a < b;
    });
    for (int rank = 0; rank < count; ++rank) {
      const int bucket =
          num_segments - 1 - (rank * num_segments) / count;
      sb_segment_[static_cast<std::size_t>(order[static_cast<std::size_t>(
          rank)])] = static_cast<std::uint8_t>(bucket);
    }
  } else {
    // One segment: every superblock is the baseline.
    std::fill(sb_segment_.begin(), sb_segment_.end(),
              static_cast<std::uint8_t>(0));
  }
  for (int i = 0; i < count; ++i) {
    if (options_.enable_delta_q) {
      // The delta-Q ramp is a deterministic per-superblock staircase so the
      // gate exercises a CurrentQIndex update at almost every superblock
      // (a content-driven map can be uniform on flat clips).  plan_delta_q()
      // rounds it to the nearest reachable step for delta_q_res.
      const bool high = (i % 2) != 0;
      sb_target_q_[static_cast<std::size_t>(i)] =
          high ? std::clamp(base_q_index() + delta, 1, 255)
               : base_q_index();
    }
  }
  return true;
}

bool InterFrameEncoder::build_segmentation(std::string& error) {
  seg_params_ = rc::SegmentationParams{};
  if (!seg_enabled_) return true;
  if (!make_video_segmentation(options_, primary_ref_is_none_, keyframe_,
                               seg_params_, error)) {
    return false;
  }
  return true;
}

bool InterFrameEncoder::prepare_qp_plan(std::string& error) {
  seg_enabled_ = options_.enable_segmentation || options_.per_segment_q;
  delta_q_enabled_ = options_.enable_delta_q;
  skip_seg_id_ = seg_enabled_ ? video_skip_segment(options_) : -1;
  globalmv_seg_id_ = seg_enabled_ ? video_globalmv_segment(options_) : -1;
  if (!seg_enabled_ && !delta_q_enabled_) return true;
  if (!build_sb_maps(error)) return false;
  if (!build_segmentation(error)) return false;
  dq_params_ = rc::DeltaQParams{};
  dq_params_.present = delta_q_enabled_;
  dq_params_.res = delta_q_enabled_ ? options_.delta_q_res : 0;
  deltaq::FrameQpHeader header;
  header.primary_ref_is_none = primary_ref_is_none_;
  header.enable_ext_seg = options_.enable_ext_seg;
  header.frame_type_switch = false;
  header.using_qmatrix = options_.enable_qm;
  if (!deltaq::derive_frame_segmentation(seg_params_, dq_params_,
                                         base_q_index(), header, enc_seg_,
                                         &error)) {
    error = "segmentation: " + error;
    return false;
  }
  if (seg_enabled_ &&
      !deltaq::init_segment_grid(
          enc_seg_grid_, mi_cols_, mi_rows_, /*tile_mi_col_start=*/0,
          /*tile_mi_col_end=*/mi_cols_, /*tile_mi_row_start=*/0,
          /*tile_mi_row_end=*/mi_rows_,
          /*enable_ext_seg=*/options_.enable_ext_seg,
          deltaq::max_segments(options_.enable_ext_seg),
          prev_segment_ids_, &error)) {
    error = "segmentation grid: " + error;
    return false;
  }
  sb_qindex_.assign(static_cast<std::size_t>(qp_sb_count()), base_q_index());
  sb_delta_steps_.clear();
  if (!delta_q_enabled_) {
    // Segmentation-only: get_qindex() is evaluated once per superblock.
    for (int i = 0; i < qp_sb_count(); ++i) {
      int q = base_q_index();
      if (seg_enabled_ &&
          !deltaq::effective_qindex(
              enc_seg_, base_q_index(), base_q_index(),
              /*delta_q_present=*/false,
              static_cast<int>(sb_segment_[static_cast<std::size_t>(i)]),
              /*ignore_delta_q=*/false, q, &error)) {
        return false;
      }
      if (q < 1 || q > 255) {
        error = "per-segment Q: effective qindex " + std::to_string(q) +
                " is outside [1, 255] (lossless segment)";
        return false;
      }
      sb_qindex_[static_cast<std::size_t>(i)] = q;
    }
  }
  return true;
}

bool InterFrameEncoder::plan_delta_q_steps(std::vector<int>& actual, std::string& error) {
  const int cols = qp_sb_cols();
  const int count = qp_sb_count();
  std::vector<int> targets = sb_target_q_;
  std::vector<char> codable(static_cast<std::size_t>(count), 1);
  for (const syntax::BlockSyntax& block : blocks_) {
    if ((block.flags & syntax::kFlagSbRoot) == 0) continue;
    const int index = (static_cast<int>(block.mi_row) / qp_sb_mi()) * cols +
                      static_cast<int>(block.mi_col) / qp_sb_mi();
    if (index < 0 || index >= count) continue;
    const bool leaf =
        block.partition ==
        static_cast<std::uint8_t>(partition::PartitionType::kNone);
    if (leaf && block.is_inter != 0 && block.skip_flag != 0) {
      codable[static_cast<std::size_t>(index)] = 0;
    }
  }
  for (int attempt = 0; attempt <= 2 * count + 4; ++attempt) {
    rc::DeltaQPlanConfig config;
    config.base_q_idx = base_q_index();
    config.delta_q_res = options_.delta_q_res;
    config.cols = cols;
    config.rows = qp_sb_rows();
    rc::DeltaQPlan plan;
    if (!rc::plan_delta_q(config, targets, plan, &error)) {
      error = "delta-Q plan: " + error;
      return false;
    }
    bool changed = false;
    int current = base_q_index();
    for (int i = 0; i < count; ++i) {
      const rc::DeltaQPlanStep& step =
          plan.steps[static_cast<std::size_t>(i)];
      if (step.qindex_before != current) {
        error = "delta-Q plan: step qindex_before does not chain";
        return false;
      }
      if (codable[static_cast<std::size_t>(i)] == 0 &&
          step.reduced_delta != 0) {
        targets[static_cast<std::size_t>(i)] = current;
        changed = true;
      }
      current = step.qindex_after;
    }
    if (changed) continue;
    sb_delta_steps_ = std::move(plan.steps);
    actual.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      actual[static_cast<std::size_t>(i)] =
          sb_delta_steps_[static_cast<std::size_t>(i)].qindex_after;
      if (trace_) {
        const rc::DeltaQPlanStep& step =
            sb_delta_steps_[static_cast<std::size_t>(i)];
        std::cerr << "deltaq sb=" << i
                  << " target=" << sb_target_q_[static_cast<std::size_t>(i)]
                  << " before=" << step.qindex_before
                  << " after=" << step.qindex_after
                  << " reduced=" << step.reduced_delta
                  << (codable[static_cast<std::size_t>(i)] == 0
                          ? " dropped(full-SB skip)"
                          : "")
                  << "\n";
      }
    }
    return true;
  }
  error = "delta-Q plan did not converge";
  return false;
}

void InterFrameEncoder::frame_plane_quant(int qindex, PlaneQuantStep out[3],
                       int& quant_step) const {
  const int q = std::clamp(qindex, 0, 255);
  PlaneQuantStep luma;
  luma.dc = luma.ac = rc::kQlookupQ0To255[q];
  out[0] = luma;
  quant_step = luma.ac;
  if (!uv_delta_enabled_) {
    out[1] = luma;
    out[2] = luma;
    return;
  }
  out[1].dc = dcdeltaq::get_q(q, resolved_delta_q_.u_dc);
  out[1].ac = dcdeltaq::get_q(q, resolved_delta_q_.u_ac);
  out[2].dc = dcdeltaq::get_q(q, resolved_delta_q_.v_dc);
  out[2].ac = dcdeltaq::get_q(q, resolved_delta_q_.v_ac);
}

void InterFrameEncoder::update_plane_quant(SbSink::State& st) const {
  frame_plane_quant(st.current_qindex, st.plane_quant, st.quant_step);
}

bool encode_video_sequence(std::span<const VideoInputFrame> frames,
                        const VideoOptions& options, VideoSequenceResult& result,
                        std::string& error) {
  // One persistent fork-join pool per encode, sized from --cpu-threads.  This
  // is the composition root: the pool lives for the whole sequence (never per
  // frame) and is only installed when there is more than one worker, so the
  // 1-thread path keeps its historical serial/per-call behaviour exactly.
  parallel::Pool worker_pool(parallel::resolve_workers(options.cpu_threads));
  parallel::PoolScope worker_scope(worker_pool);  if (options.sb128 && options.sb256) {
    error = "sb128 and sb256 are mutually exclusive";
    return false;
  }
  if (options.sb128 || options.sb256) {
    if (options.format.chroma != fmt::ChromaFormat::k420 ||
        options.format.depth != fmt::BitDepth::k8) {
      error = "--video-sb128/--video-sb256 are wired for 8-bit 4:2:0 only";
      return false;
    }
    if (options.tile_cols != 0 || options.tile_rows != 0) {
      error = "--video-sb128/--video-sb256 do not compose with --tiles yet";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras ||
        options.enable_tip) {
      error = "--video-sb128/--video-sb256 do not compose with the B/temporal/"
              "random-access/TIP frame schedules yet";
      return false;
    }
    if (options.loop_filters.any()) {
      error = "--video-sb128/--video-sb256 do not compose with the loop "
              "filters yet (the CDEF/CCSO/GDF/LR unit grids are 64-based); "
              "deblocking is supported";
      return false;
    }
    if (options.enable_segmentation || options.enable_delta_q ||
        options.per_segment_q) {
      error = "--video-sb128/--video-sb256 do not compose with "
              "segmentation/delta-Q yet";
      return false;
    }
    if (options.enable_ref_frame_mvs) {
      error = "--video-sb128/--video-sb256 do not compose with "
              "--video-ref-mvs yet";
      return false;
    }
  }
  if (options.ext_partitions) {
#if !defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    error = "--video-ext-partitions needs the full build (partition_ext "
            "adapter)";
    return false;
#else
    if (options.format.chroma != fmt::ChromaFormat::k420 ||
        options.format.depth != fmt::BitDepth::k8) {
      error = "--video-ext-partitions is wired for 8-bit 4:2:0 only";
      return false;
    }
    if (options.tile_cols != 0 || options.tile_rows != 0) {
      error = "--video-ext-partitions does not compose with --tiles yet";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras ||
        options.enable_tip) {
      error = "--video-ext-partitions does not compose with the B/temporal/"
              "random-access/TIP frame schedules yet";
      return false;
    }
    if (options.sb256) {
      error = "--video-ext-partitions does not compose with --video-sb256 "
              "(the extended solver is wired for 64x64/128x128 roots)";
      return false;
    }
    // The extended partition rows are part of the §8.2 CDF image (video
    // syntax CdfState::do_ext_partition / do_uneven_4way_partition), so the
    // --avg-cdf/--adaptive-cdf/--cdf-continuity carry chain is bit-exact with
    // the extended writer; nothing to reject here.
    if (options.enable_segmentation || options.enable_delta_q ||
        options.per_segment_q) {
      error = "--video-ext-partitions does not compose with "
              "segmentation/delta-Q yet";
      return false;
    }
    if (options.enable_ref_frame_mvs) {
      error = "--video-ext-partitions does not compose with --video-ref-mvs "
              "yet";
      return false;
    }
#endif  // AV2ENC_VIDEO_EXT_PARTITIONS
  }
  // §5.20.8 palette / §5.20.5.3-.4 general IBC are keyframe mode-info tools.
  // Keep them on the plain key+P chain (like the morph carrier): the frame-type
  // wrappers write their own sequence/key headers and would omit the
  // screen-content bits the tile payload depends on.
  if (options.enable_palette || options.enable_ibc) {
    const char* tool = options.enable_palette ? "palette" : "IBC";
    if (options.format.chroma != fmt::ChromaFormat::k420 ||
        options.format.depth != fmt::BitDepth::k8) {
      error = std::string("--video-") + tool +
              " is gated on 8-bit 4:2:0; the format video core has no "
              "screen-content syntax";
      return false;
    }
    if (options.b_frames != 0 || options.temporal_layers != 0 ||
        options.enable_tip || options.long_term || options.olk ||
        options.leading_frames != 0 || options.sef || options.ra_switch ||
        options.ras) {
      error = std::string("--video-") + tool +
              " is gated on the plain key+P chain: it cannot be combined with "
              "B frames, temporal layers, TIP, SEF or the frame-type wrappers";
      return false;
    }
    if (options.tile_cols != 0 || options.tile_rows != 0) {
      error = std::string("--video-") + tool +
              " is gated on a single tile (the palette cache and IBC ring are "
              "tile-local)";
      return false;
    }
    if (options.enable_palette && options.enable_tx_search) {
      error = "--video-palette is not wired with --video-tx-search";
      return false;
    }
    if (options.enable_ibc && options.drl_reorder_always) {
      error = "--video-ibc codes use_intrabc and §7.12.2.19 "
              "DRL_REORDER_ALWAYS reorders the decoder's BV stack, which the "
              "IBC stack builder does not implement";
      return false;
    }
    if (options.enable_ibc &&
        (options.enable_segmentation || options.per_segment_q ||
         options.enable_uv_delta_q)) {
      error = "--video-ibc is not wired with "
              "--video-segmentation/--video-per-segment-q/"
              "--video-uv-delta-q";
      return false;
    }
    if (options.avg_cdf || options.cdf_continuity) {
      error = std::string("--video-") + tool +
              " is not wired with --avg-cdf/--cdf-continuity";
      return false;
    }
  }  // Non-default formats use the depth/chroma-generic core.  The historical
  if ((options.enable_sdp || options.enable_inter_ddt) &&
      (options.format.chroma != fmt::ChromaFormat::k420 ||
       options.format.depth != fmt::BitDepth::k8)) {
    error = "--video-sdp/--video-ddt are wired for 8-bit 4:2:0";
    return false;
  }
  // §5.20.3.1 sub-8 luma leaves.  The rect tree carries ChromaMi ownership
  // (has_chroma/chroma_offset) and every leaf whose chroma residual would be
  // smaller than 4x4 shares the inherited ChromaMi block with the next leaf,
  // so the intra keyframe can code 4x4/4x8/8x4/... leaves.  Inter leaves stay
  // at 8x8 or above (the P-frame search floor is raised in decide()).
  if (options.min_block_size < 4 || options.min_block_size > 64) {
    error = "min_block_size must be in [4, 64]";
    return false;
  }
  if (options.min_block_size < 8 &&
      (options.format.chroma != fmt::ChromaFormat::k420 ||
       options.format.depth != fmt::BitDepth::k8)) {
    error = "--video-min-block 4 (shared-chroma sub-8 leaves) is wired for "
            "8-bit 4:2:0 only";
    return false;
  }
  // Pinned dav2d rejects the first P frame ("Invalid argument") when
  // enable_idtx_intra is 1 and the keyframe search reaches the sub-8/deep
  // leaves; the same failure appears at --video-min-block 8 --video-max-depth
  // 5, so it is a pre-existing idtx/small-leaf interaction rather than the
  // shared-chroma ownership.  Keep the named rejection for the new depth until
  // it is isolated.
  if (options.min_block_size < 8 && options.enable_idtx_intra) {
    error = "--video-min-block 4 does not compose with --video-idtx yet "
            "(the key+P chain is not decoder-exact at the sub-8 search depths)";
    return false;
  }
  // The SDP luma tree codes YMode through the same §5.20.5.5 dynamic list but
  // `--video-intra-full` streams from it are not decoder-exact at any
  // --video-min-block (pre-existing; `--video-sdp --video-mrl` is exact).
  if (options.enable_sdp && options.enable_intra_full) {
    error = "--video-sdp does not compose with --video-intra-full yet";
    return false;
  }
  // Non-default formats use the depth/chroma-generic core.  The historical
  // 8-bit 4:2:0 path below is untouched.  The four reference-management tools
  // own header/syntax fields that core does not write.
  if ((options.short_refresh || options.refmvbank || options.mv_traj ||
       options.same_ref_compound) &&
      (options.format.chroma != fmt::ChromaFormat::k420 ||
       options.format.depth != fmt::BitDepth::k8)) {
    error = "reference-management tools (--video-short-refresh/"
            "--video-refmvbank/--video-mv-traj/--video-same-ref-compound) are "
            "wired for the 8-bit 4:2:0 video core only";
    return false;
  }
  if (options.format.chroma != fmt::ChromaFormat::k420 ||
      options.format.depth != fmt::BitDepth::k8) {
    if (options.implicit_ref_map) {
      error = "the implicit reference map is wired for the 8-bit 4:2:0 video "
              "core only; the format cores keep the explicit map";
      return false;
    }
    if (options.drift_refresh) {
      error = "--video-drift-refresh is wired for the 8-bit 4:2:0 video core "
              "only; the format cores have no drift measurement";
      return false;
    }
    return encode_video_sequence_format(frames, options, result, error);
  }
  // --video-drift-refresh (see VideoOptions::drift_refresh): one per-frame QP
  // plan on a single-tile low-delay inter chain.  The frame-type wrappers, the
  // B/reorder schedules, the CDF chain and the tile writers own their own
  // base_q_idx handling, so they are named rejections rather than silently
  // unrefreshed pictures.
  if (options.drift_refresh) {
    if (options.drift_refresh_threshold_db <= 0.0 ||
        options.drift_refresh_threshold_db >= 30.0) {
      error = "drift_refresh_threshold_db must be in (0, 30)";
      return false;
    }
    if (options.drift_refresh_qp < 0 || options.drift_refresh_qp > 64) {
      error = "drift_refresh_qp must be in [0, 64]";
      return false;
    }
    if (options.drift_refresh_budget < 0) {
      error = "drift_refresh_budget must be >= 0";
      return false;
    }
    if (options.tile_cols > 0 || options.tile_rows > 0) {
      error = "--video-drift-refresh is wired for a single tile (the tiled "
              "frame header signals one base_q_idx for the whole tile group)";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras ||
        options.enable_tip) {
      error = "--video-drift-refresh is wired for the flat key+P chain; drop "
              "the B-frame / reorder / temporal-layer / TIP / frame-type "
              "wrapper mode";
      return false;
    }
    if (options.compound) {
      error = "--video-drift-refresh is wired for single-reference inter "
              "prediction";
      return false;
    }
    // The CDF chain composes only while the refresh keeps every frame's QP:
    // with drift_refresh_qp == 0 the §9.3 image still follows the static
    // per-frame plan, and the forced residual is decision-level only.
    if ((options.avg_cdf || options.adaptive_cdf ||
         options.cdf_continuity) &&
        options.drift_refresh_qp != 0) {
      error = "--video-drift-refresh composes with the CDF chain only at "
              "--video-drift-refresh-qp 0: a run-time base_q_idx change would "
              "desynchronise the per-frame §9.3 image";
      return false;
    }
    // --cdf-continuity composes with --video-drift-refresh: the carried image
    // follows the symbols actually coded, so a refresh frame's extra residual
    // needs no §9.3 re-derivation.  Only the per-frame-QP reset form depends on
    // base_q_idx, and that is checked above.
    if (options.enable_segmentation || options.per_segment_q ||
        options.enable_delta_q) {
      error = "--video-drift-refresh does not compose with segmentation/"
              "delta-Q: both own the per-frame effective qindex";
      return false;
    }
    if (options.sb128 || options.sb256) {
      error = "--video-drift-refresh is wired for the 64x64 root partition "
              "search";
      return false;
    }
  }
  error.clear();
  result = VideoSequenceResult{};
  if (frames.empty()) {
    error = "no input frames";
    return false;
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  // §5.18.4.4 codes MiCols x MiRows = 2*((W+7)>>3) x 2*((H+7)>>3) 4x4 units,
  // so any non-zero size is codable; only the 4:2:0 chroma layout constrains
  // the input.  The MI-aligned padded region is reconstructed and the output
  // picture is cropped back to width x height.
  if (width == 0 || height == 0) {
    error = "video requires non-zero width/height";
    return false;
  }
  if ((width & 1U) != 0 || (height & 1U) != 0) {
    error = "4:2:0 video requires even width/height, got " +
            std::to_string(width) + "x" + std::to_string(height);
    return false;
  }
  if (width > 0xFFFFU || height > 0xFFFFU) {
    error = "M2 dimensions exceed the sequence header field width";
    return false;
  }
  // §5.4.6 reference-management tools.  Each is wired on the flat key+P chain
  // with an explicit reference map and a single tile; every wrapper that owns
  // its own sequence/frame header is rejected by name rather than silently
  // omitting the tool.
  const bool ref_tool = options.short_refresh || options.refmvbank ||
                        options.mv_traj || options.same_ref_compound;
  if (ref_tool) {
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras ||
        options.enable_tip) {
      error = "reference-management tools (--video-short-refresh/"
              "--video-refmvbank/--video-mv-traj/--video-same-ref-compound) "
              "are gated on the plain key+P chain";
      return false;
    }
    if (options.tile_cols != 0 || options.tile_rows != 0) {
      error = "reference-management tools are gated on a single tile (the "
              "tiled header writer does not signal them)";
      return false;
    }
    if (options.implicit_ref_map) {
      error = "reference-management tools require the explicit reference map "
              "(--video-implicit-ref-map does not signal a stable ref pair)";
      return false;
    }
    if (options.avg_cdf || options.adaptive_cdf || options.cdf_continuity) {
      error = "reference-management tools are not wired with the average-CDF/"
              "CDF-continuity chain";
      return false;
    }
    if (options.mfh || options.video_timing) {
      error = "reference-management tools are not wired with --video-mfh/"
              "--video-timing";
      return false;
    }
  }
  if (options.same_ref_compound) {
    if (!options.compound) {
      error = "--video-same-ref-compound requires --video-compound";
      return false;
    }
    if (options.num_total_refs < 2) {
      error = "--video-same-ref-compound requires --video-refs >= 2";
      return false;
    }
    if (options.compound_wedge || options.compound_diffwtd) {
      error = "--video-same-ref-compound codes the AVERAGE blend only; "
              "masked compound is not wired for a same-reference pair";
      return false;
    }
    if (options.mv_traj) {
      error = "--video-same-ref-compound is not wired with --video-mv-traj "
              "(the trajectory rewrite of the compound derived stack is "
              "gated on two distinct references)";
      return false;
    }
  }
  if (options.mv_traj && options.num_total_refs < 2) {
    error = "--video-mv-traj needs at least two references (the trajectory "
            "is only consumed by the derived-candidate projection)";
    return false;
  }
  if (options.mv_traj && !options.enable_ref_frame_mvs) {
    error = "--video-mv-traj requires --video-ref-mvs (the trajectory field "
            "is only built by the §7.9 estimation path)";
    return false;
  }
  // §5.18.2 compact reference signalling (--video-ref-signal-compact).  It
  // rewrites the sequence header (NumRefFrames = 1, explicit_ref_frame_map =
  // 0) and the per-frame primary/map fields, so it is gated on the one stream
  // shape it was measured on: a single-reference low-delay key+P chain.
  if (options.ref_signal_compact) {
    if (options.num_total_refs != 1) {
      error = "--video-ref-signal-compact needs --video-refs 1 (the derived "
              "§7.7 list must hold exactly the one live slot)";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras ||
        options.enable_tip || options.enable_morph) {
      error = "--video-ref-signal-compact is gated on the plain key+P chain";
      return false;
    }
    if (options.tile_cols != 0 || options.tile_rows != 0) {
      error = "--video-ref-signal-compact is gated on a single tile (the "
              "tiled header writer has its own reference fields)";
      return false;
    }
    if (options.implicit_ref_map) {
      error = "--video-ref-signal-compact already selects the derived §7.7 "
              "map; drop --video-implicit-ref-map";
      return false;
    }
    if (options.short_refresh) {
      error = "--video-ref-signal-compact owns refresh_frame_flags; drop "
              "--video-short-refresh";
      return false;
    }
    if (options.num_ref_frames != 0) {
      error = "--video-ref-signal-compact sets NumRefFrames = 1; drop "
              "--video-num-ref-frames";
      return false;
    }
    if (options.mfh || options.video_timing || options.multi_sequence) {
      error = "--video-ref-signal-compact is not wired with --video-mfh/"
              "--video-timing/--video-multi-sequence";
      return false;
    }
  }
  // The §7.9.2 trajectory field feeds both the §7.12.2.8 temporal scan and the
  // §7.12.2.14/.16 derived candidates for any reference count.
  // The syntax writer selects the §8.3.1 qctx slice from base_q_idx and the
  // quantiser below uses qlookup(base_q_idx).  0 (lossless) is not implemented.
  if (options.base_q_idx == 0) {
    error = "M2 does not support lossless base_q_idx = 0";
    return false;
  }
  if (options.loop_filters.cfl_ds_filter_index != 0) {
    error = "--video-cfl-ds-filter is wired for the format video core "
            "(--bit-depth 10 or --chroma) only";
    return false;
  }
  if (options.enable_sdp) {
    if (options.tile_cols > 1 || options.tile_rows > 1) {
      error = "--video-sdp is wired for a single tile";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras ||
        options.enable_tip || options.enable_morph || options.enable_cfl ||
        options.enable_mhccp || options.enable_segmentation ||
        options.enable_delta_q || options.per_segment_q) {
      error = "--video-sdp does not compose with the frame-type wrappers, "
              "morph/CfL/MHCCP or segmentation/delta-Q";
      return false;
    }
  }
  if (options.enable_inter_ddt && options.enable_tip) {
    error = "--video-ddt does not compose with the TIP header writer";
    return false;
  }
  if (options.max_partition_depth < 0 || options.max_partition_depth > 8) {
    error = "max_partition_depth must be in [0, 8]";
    return false;
  }
  if (!options.frame_base_q_idx.empty() &&
      options.frame_base_q_idx.size() != frames.size()) {
    error = "frame_base_q_idx must be empty or have one entry per frame";
    return false;
  }
  for (const std::uint8_t qp : options.frame_base_q_idx) {
    if (qp == 0) {
      error = "frame_base_q_idx entries must be in [1, 255]";
      return false;
    }
  }
  // Per-frame quantizer selection: an empty vector means the historical
  // fixed-QP stream, so the default path does not construct any extra state.
  const auto frame_qp = [&options](std::size_t index) -> std::uint8_t {
    return static_cast<std::uint8_t>(
        options.frame_base_q_idx.empty() ? options.base_q_idx
                                         : options.frame_base_q_idx[index]);
  };
  // The opt-in loop filters run per whole frame.  Multi-tile filtering is not
  // wired yet, and the frame-type wrapper headers are written by the shared
  // random-access components which do not signal the filter parameters; both
  // are named rejections rather than silently unfiltered pictures.
  if (options.loop_filters.any()) {
    if ((options.loop_filters.wiener || options.loop_filters.chroma_lr) &&
        (options.tile_cols > 1 || options.tile_rows > 1)) {
      error = "video loop restoration (Wiener/chroma LR) does not compose "
              "with --tiles > 1x1 yet";
      return false;
    }
    if (options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras) {
      error = "video loop filters do not compose with the random-access "
              "frame-type modes yet";
      return false;
    }
  }
  // §5.13 user-defined QM (--video-user-qm) is emitted once, before the key
  // frame of the plain single-tile chain; every other composition writes its
  // own temporal units and would omit the OBU.
  if (options.user_qm) {
    if (!options.enable_qm) {
      error = "--video-user-qm requires --video-qm";
      return false;
    }
    if (options.tile_cols > 0 || options.tile_rows > 0) {
      error = "--video-user-qm does not compose with --tiles";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras) {
      error = "--video-user-qm does not compose with the B/temporal-layer/"
              "random-access frame schedules";
      return false;
    }
    if (options.enable_tip) {
      error = "--video-user-qm does not compose with --video-tip";
      return false;
    }
  }
  // §5.14 film grain is planned per output picture and inserted into the
  // temporal unit of every schedule.  The random-access frame-type wrappers
  // carry the sequence bit, the per-frame film_grain_config() and one type-23
  // model OBU through their own component writers (src/codec/inter/{sef,
  // random_access,longterm,leading}); the shared planner lives in
  // video_frames.cpp.
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const VideoInputFrame& f = frames[i];
    if (f.width != width || f.height != height) {
      error = "all frames must have the same dimensions";
      return false;
    }
    const std::size_t luma = static_cast<std::size_t>(width) * height;
    const std::size_t chroma =
        static_cast<std::size_t>((width + 1U) / 2U) * ((height + 1U) / 2U);
    if (f.y.size() < luma || f.u.size() < chroma || f.v.size() < chroma) {
      error = "input frame planes are too small";
      return false;
    }
  }

  // Frame 0: intra keyframe. The M1 integration TU is not linkable in this
  // tree snapshot, so the M2-owned copy of the M1 intra syntax writer emits
  // the keyframe. Decisions are DC intra + DCT; the syntax is the same
  // intra_frame_mode_info() path M1 uses.
  if (options.num_total_refs < 1 || options.num_total_refs > kRefsPerFrame) {
    error = "num_total_refs must be in [1, " +
            std::to_string(kRefsPerFrame) + "]";
    return false;
  }
  if (options.num_ref_frames != 0 &&
      (options.num_ref_frames < 4 || options.num_ref_frames > kMaxRefSlots)) {
    error = "num_ref_frames must be 0 (auto) or in [4, " +
            std::to_string(kMaxRefSlots) + "]";
    return false;
  }
  if (options.num_ref_frames != 0 &&
      options.num_total_refs > options.num_ref_frames) {
    error = "num_total_refs " + std::to_string(options.num_total_refs) +
            " exceeds NumRefFrames " + std::to_string(options.num_ref_frames);
    return false;
  }
  if (options.implicit_ref_map) {
    // §7.7 derives NumTotalRefs and ref_frame_idx from the slot state, so every
    // tool that owns its own reference map / block-level ref list stays on the
    // explicit form and is rejected by name here.
    if (options.enable_tip) {
      error = "--video-implicit-ref-map does not compose with --video-tip: "
              "TIP needs the explicit past/future anchor pair";
      return false;
    }
    if (options.compound) {
      error = "--video-implicit-ref-map does not compose with "
              "--video-compound: reference_select and the compound ref pairs "
              "require an explicit two-reference map";
      return false;
    }
    if (options.tile_cols > 0 || options.tile_rows > 0) {
      error = "--video-implicit-ref-map does not compose with --tiles: the "
              "tiled frame header signals a single explicit reference slot";
      return false;
    }
    if (options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras) {
      error = "--video-implicit-ref-map does not compose with the "
              "long-term/OLK/leading/SEF/SWITCH/RAS frame-type wrappers";
      return false;
    }
  }
  if (options.mv_precision < 0 || options.mv_precision > 2) {
    error = "mv_precision must be 0 (1/8), 1 (1/4) or 2 (1/2) pel";
    return false;
  }
  if (options.enable_tip) {
    // TIP_FRAME_AS_OUTPUT is frame-level, so it composes with no per-block
    // tool.  The schedule needs a future anchor that is decoded before the
    // TIP frames, which is why the supported surface is 3..8 input pictures
    // and the 8-slot NumRefFrames pool.
    if (frames.size() < 3 || frames.size() > 8) {
      error = "--video-tip supports 3..8 input frames (got " +
              std::to_string(frames.size()) + ")";
      return false;
    }
    if (options.enable_ref_frame_mvs && frames.size() < 4) {
      error = "--video-tip with --video-ref-mvs needs at least 4 input "
              "frames: the last picture is coded as a tile-coded P that "
              "references the TIP display frame";
      return false;
    }
    if (options.tip_block_ref) {
      // §5.20.7.10 block form.  The carrier needs the TIP-referencing tail P
      // (so the frame has two nearest refs and allow_ref_frame_mvs), a
      // deterministic last leaf (one 64x64 leaf per superblock) and the plain
      // 1/8-pel NEWMV MVD syntax.
      if (!options.enable_ref_frame_mvs) {
        error =
            "--video-tip-block-ref requires --video-ref-mvs: the "
            "RefFrame[0] == TIP_FRAME block lives in the TIP-referencing "
            "tail P and needs use_ref_frame_mvs = 1";
        return false;
      }
      if (!options.force_single_block) {
        error =
            "--video-tip-block-ref requires --video-force-single: the "
            "carrier converts the frame's last coded leaf, and only the "
            "single-64x64-leaf partition guarantees it is >= 8x8";
        return false;
      }
      if (options.mv_precision != 0 || options.seq_force_integer_mv) {
        error =
            "--video-tip-block-ref is wired for the 1/8-pel frame MV "
            "precision";
        return false;
      }
      if (options.enable_adaptive_mvd || options.enable_flex_mvres ||
          options.enable_mvd_sign_derive) {
        error =
            "--video-tip-block-ref does not compose with "
            "--video-adaptive-mvd/--video-flex-mvres/"
            "--video-mvd-sign-derive";
        return false;
      }
      if (options.enable_bawp || options.enable_morph || options.enable_interintra ||
          options.compound) {
        error =
            "--video-tip-block-ref does not compose with the BAWP/morph/"
            "inter-intra/compound carriers";
        return false;
      }
      if (options.refmvbank || options.mv_traj) {
        error =
            "--video-tip-block-ref does not compose with "
            "--video-refmvbank/--video-mv-traj: the tail P's TIP-scaled "
            "field and TIP block only model the plain §7.9 chain";
        return false;
      }
      if (options.enable_segmentation || options.enable_delta_q) {
        error =
            "--video-tip-block-ref does not compose with §5.18.7 "
            "segmentation/delta-Q";
        return false;
      }
    }
    if (options.tile_cols > 0 || options.tile_rows > 0) {
      error = "--video-tip does not compose with --tiles";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras) {
      error = "--video-tip does not compose with the B/temporal-layer/"
              "random-access frame schedules";
      return false;
    }
    if (options.loop_filters.any()) {
      error = "--video-tip does not compose with the loop filters yet";
      return false;
    }
    if (options.avg_cdf || options.adaptive_cdf || options.cdf_continuity) {
      error = "--video-tip does not compose with the CDF chain";
      return false;
    }
    if (options.compound) {
      error = "--video-tip does not compose with --video-compound";
      return false;
    }
    if (options.enable_gme || options.enable_warp) {
      error = "--video-tip does not compose with --video-gme/--video-warp";
      return false;
    }
    if (options.enable_tx_search || options.enable_deblock) {
      error = "--video-tip does not compose with --video-tx-search/"
              "--video-deblock";
      return false;
    }
    if (!options.frame_base_q_idx.empty()) {
      error = "--video-tip requires a fixed base_q_idx";
      return false;
    }
    if (options.mfh || options.video_timing) {
      error = "--video-tip does not compose with --video-mfh/--video-timing";
      return false;
    }
    return encode_tip_sequence(frames, options, result, error);
  }
  if (options.enable_deblock &&
      (options.long_term || options.olk || options.leading_frames > 0 ||
       options.sef || options.ra_switch || options.ras)) {
    // The frame-type wrappers assemble headers through their own modules, which
    // do not signal deblocking_filter_params(); filtering pixels without the
    // matching header would desynchronise the decoder.
    error = "--video-deblock does not compose with the frame-type wrappers "
            "(long-term/OLK/leading/SEF/SWITCH/RAS)";
    return false;
  }
  if (options.enable_deblock &&
      (options.tile_cols > 1 || options.tile_rows > 1)) {
    // Per-tile deblocking treats tile edges as frame edges, but pinned dav2d
    // still filters across interior tile edges even with
    // disable_loopfilters_across_tiles = 1 (docs/negative-results.md).
    error = "--video-deblock does not support multiple tiles: the pinned "
            "decoder filters across interior tile edges";
    return false;
  }
  // §5.18.7.1/.8 opt-in segmentation and delta-Q: CPU --video only.  Every
  // rejection below keeps either an unsupported header writer or a mode whose
  // block decision is not replayed by the segmentation planner out of the
  // gated surface.
  const bool qp_tools = options.enable_segmentation || options.enable_delta_q ||
                        options.per_segment_q || options.enable_uv_delta_q;
  if (qp_tools) {
#if !defined(AV2ENC_VIDEO_DELTAQ)
    error = "segmentation/delta-Q (--video-segmentation/--video-delta-q/"
            "--video-per-segment-q) needs the full build "
            "(src/codec/entropy/deltaq)";
    return false;
#else
    if (options.enable_delta_q && options.per_segment_q) {
      error = "--video-per-segment-q does not compose with --video-delta-q: "
              "the SEG_LVL_ALT_Q planner and the per-superblock delta-Q ramp "
              "are separate features";
      return false;
    }
    if (options.enable_qm) {
      error = "segmentation/delta-Q does not compose with --video-qm: the "
              "frame-header writer owns the using_qmatrix bit";
      return false;
    }
    if (options.delta_q_res < 0 || options.delta_q_res > 3) {
      error = "delta_q_res must be in [0, 3]";
      return false;
    }
    if (options.seg_q_delta < -351 ||
        options.seg_q_delta > 351) {
      error = "the per-segment Q delta must be in [-351, 351]";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras) {
      error = "segmentation/delta-Q does not compose with the B/temporal-layer/"
              "random-access frame schedules (their header writers do not own "
              "the §5.18.7.1/.8 rows)";
      return false;
    }
    if (options.enable_tip) {
      error = "segmentation/delta-Q does not compose with --video-tip";
      return false;
    }
    if (options.tile_cols > 1 || options.tile_rows > 1) {
      error = "segmentation/delta-Q does not support --tiles > 1x1 yet: the "
              "tiled frame-header writer does not signal the §5.18.7 rows";
      return false;
    }
    if (!options.frame_base_q_idx.empty()) {
      error = "segmentation/delta-Q requires a fixed base_q_idx "
              "(no --rc crf/vbr per-frame QP)";
      return false;
    }
    if (options.adaptive_cdf || options.cdf_continuity) {
      error = "segmentation/delta-Q does not compose with --adaptive-cdf/"
              "--cdf-continuity: the per-tile delta/segment CDFs are not part "
              "of the §8.2.4 CDF image";
      return false;
    }
#endif
  }
  const int num_total_refs = options.num_total_refs;
  // §5.18.2 compact reference signalling: NumRefFrames = 1 makes the §7.7
  // derived map hold exactly the one live slot, so the writer codes no
  // reference-map bits.  The order-hint width stays 4: the §5.18.3.1
  // get_relative_dist() is an exact difference (Clip3, no modulo), so a
  // narrower OrderHintBits would alias the distance at every wrap.
  const int num_ref_frames =
      options.ref_signal_compact
          ? 1
          : av2enc::video::resolved_num_ref_frames(options);
  const int order_hint_bits = 4;

  // B frames (coding order != display order, two references).
  if (options.b_frames > 0) {
#if defined(AV2ENC_VIDEO_FRAMES)
    const int gop_size = options.b_frames + 2;
    if (gop_size > 16) {
      error = "--b-frames supports at most 6 B frames per GOP";
      return false;
    }
    return encode_b_sequence(frames, options, gop_size, result, error);
#else
    error = "--b-frames needs the full build (bframes module)";
    return false;
#endif
  }

#if !defined(AV2ENC_VIDEO_FRAMES)
  if (options.compound) {
    error = "--video-compound needs the full build (compound module)";
    return false;
  }
#endif

  // Hierarchical-P temporal layers (plan/coding order, per-OBU tlayer id).
  if (options.temporal_layers > 0) {
#if defined(AV2ENC_VIDEO_FRAMES)
    return encode_temporal_sequence(frames, options, result, error);
#else
    error = "--video-tlayers needs the full build (tlayers module)";
    return false;
#endif
  }


  // Opt-in uniform tile grid for the video path.  Each tile is encoded as an
  // independent tile-local picture (fresh CDF, tile-local superblock origin,
  // cleared above/left contexts) but motion compensation still reads the
  // global reference frame, exactly like the decoder.  The post tile-group
  // writer then emits the §5.18.7.2 tile_info() and interleaved
  // le(TileSizeBytes) size fields.
  bool tiled = false;
  tiles::TileLayout tile_layout;
  post::VideoPostSequenceOptions tile_sequence;
  // Filters run on the completed frame, so the decoder must filter the shared
  // tile edges too (0 = filters may read across tiles).
  // No filters: keep the historical per-tile independence bit.  With filters
  // the picture-global pass filters the shared edges, so the decoder must too.
  tile_sequence.filters.disable_loopfilters_across_tiles =
      !options.loop_filters.any();
  tile_sequence.filters.enable_cdef = options.loop_filters.cdef;
  tile_sequence.filters.enable_gdf = options.loop_filters.gdf;
  tile_sequence.filters.enable_restoration =
      options.loop_filters.wiener || options.loop_filters.chroma_lr;
  tile_sequence.filters.lr_disable_luma_pc_wiener = true;
  tile_sequence.filters.lr_disable_luma_wiener_ns =
      !options.loop_filters.wiener;
  tile_sequence.filters.lr_uv_present = options.loop_filters.chroma_lr;
  tile_sequence.filters.lr_disable_chroma_wiener_ns =
      !options.loop_filters.chroma_lr;
  tile_sequence.filters.enable_ccso = options.loop_filters.ccso;
  tile_sequence.filters.ccso_unit_matches_sb_size = true;
  tile_sequence.filters.gdf_unit_matches_sb_size = true;
  tile_sequence.filters.cdef_on_skip_txfm_always_on =
      options.loop_filters.cdef_on_skip_txfm_mode == 1;
  tile_sequence.filters.cdef_on_skip_txfm_disabled =
      options.loop_filters.cdef_on_skip_txfm_mode == 0;
  // §5.4.8: the tiled frame writer owns its own sequence header, so it must
  // signal the average-CDF flags the per-tile bank below actually runs.
  tile_sequence.enable_avg_cdf = options.avg_cdf;
  tile_sequence.avg_cdf_type = true;
  tile_sequence.film_grain_params_present = options.film_grain;
  if (options.tile_cols != 0 || options.tile_rows != 0) {
    if (options.tile_cols < 1 || options.tile_rows < 1) {
      error = "tile_cols/tile_rows must both be positive";
      return false;
    }
    if (options.tile_cols > 1 || options.tile_rows > 1) {
      if (options.num_total_refs != 1) {
        error = "--tiles requires --video-refs 1: the tiled frame header "
                "signals a single explicit reference slot";
        return false;
      }
      if (options.mv_precision != 0) {
        error = "--tiles requires eighth-pel MV precision (the tiled frame "
                "header writer signals the 1/8-pel pair)";
        return false;
      }
      const post::VideoTilePlan planned =
          post::plan_video_tiles(static_cast<int>(width),
                                 static_cast<int>(height), options.tile_cols,
                                 options.tile_rows,
                                 tiles::kSuperblockPixels64, 7, 0);
      if (!planned.ok) {
        error = "--tiles " + std::to_string(options.tile_cols) + "x" +
                std::to_string(options.tile_rows) + " rejected: " +
                planned.error;
        return false;
      }
      tile_layout = planned.layout;
      tiled = true;
    }
  }

  struct CodedTile {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    std::vector<std::uint8_t> payload;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    std::uint32_t inter_blocks = 0;
    std::uint32_t intra_blocks = 0;
    post::VideoFilterPlan filter_plan;
  };
  std::vector<CodedTile> tile_sources;
  if (tiled) {
    for (const tiles::TileRect& tile : tile_layout.tiles) {
      CodedTile entry;
      entry.x = tile.x0;
      entry.y = tile.y0;
      entry.w = tile.pixel_width();
      entry.h = tile.pixel_height();
      tile_sources.push_back(std::move(entry));
    }
  } else {
    CodedTile whole;
    whole.w = static_cast<int>(width);
    whole.h = static_cast<int>(height);
    tile_sources.push_back(std::move(whole));
  }

  // Full-frame reconstruction planes.  Per-tile visible regions are pasted
  // into these once per frame; they are the decoder-visible frame and the
  // reference pool for the next P frame.
  const int recon_mi_cols =
      static_cast<int>(2U * ((width + 7U) >> 3U));
  const int recon_mi_rows =
      static_cast<int>(2U * ((height + 7U) >> 3U));
  // --video-edge-inter: the decoder keeps the SB-aligned reconstruction and
  // pads the reference past the visible bottom edge, so the frame buffer and
  // the reference extent include those rows.  The emitted/output picture is
  // still cropped to width x height.
  int recon_rows = recon_mi_rows * 4;
  int ref_mi_rows = recon_mi_rows;
  if (options.enable_edge_inter) {
    const int sb = options.superblock_pixels();
    const int aligned = ((recon_rows + sb - 1) / sb) * sb;
    if (aligned > recon_rows) {
      recon_rows = aligned;
      ref_mi_rows = aligned / 4;
    }
  }
  Plane frame_recon[3];
  frame_recon[0].resize(recon_mi_cols * 4, recon_rows, 0);
  frame_recon[1].resize(recon_mi_cols * 2, recon_rows / 2, 0);
  frame_recon[2].resize(recon_mi_cols * 2, recon_rows / 2, 0);

  // Copies a tile encoder's MI-aligned reconstruction into the frame buffer.
  // The coded tile extent is rounded up to MiCols/MiRows (§5.18.4.4), so the
  // copy may extend past the visible width/height at the bottom/right edge;
  // those padded samples are what the in-loop filters read and the decoder
  // reconstructs.  The destination is clipped to its padded extent.
  const auto paste_tile = [](const Plane& src, Plane& dst, int x0, int y0,
                             int w, int h) {
    const int copy_w = std::min(w, dst.stride - x0);
    const int copy_h = std::min(h, dst.height - y0);
    for (int y = 0; y < copy_h; ++y) {
      std::memcpy(dst.samples.data() +
                      static_cast<std::size_t>(y0 + y) *
                          static_cast<std::size_t>(dst.stride) +
                      static_cast<std::size_t>(x0),
                  src.samples.data() +
                      static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(src.stride),
                  static_cast<std::size_t>(copy_w));
    }
  };

  // --video-edge-inter: the decoder reconstructs the full bottom block but
  // extends the reference from the coded MI edge (§7.20-style edge
  // replication) before the frame becomes a reference.  That edge is
  // MiRows*4, not the visible height: for a height that is not a multiple of
  // 8 the last MI row is only partly visible and the decoder keeps its
  // reconstruction (e.g. 100x70: rows 70..71 are inside MiRows*4 = 72 and are
  // real samples; rows 72.. are the replicated ones).  Mirror that here so the
  // next frame's MC reads the same padded samples.  Runs after the in-loop
  // filters, which is where the decoder extends.
  const auto pad_recon_bottom = [&]() {
    // Only --video-edge-inter asks for the padded reference.  Without the
    // flag the MI-aligned rows below the visible picture keep their historical
    // contents (zero), which the CDEF/recon gates depend on for non-/8 sizes.
    if (!options.enable_edge_inter ||
        recon_rows <= recon_mi_rows * 4) {
      return;
    }
    for (int plane = 0; plane < 3; ++plane) {
      Plane& dst = frame_recon[plane];
      // Luma MI edge in rows; chroma is subsampled by 2 in both directions.
      const int mi_edge_h = plane == 0 ? recon_mi_rows * 4
                                       : recon_mi_rows * 2;
      if (mi_edge_h <= 0 || mi_edge_h > dst.height) continue;
      const std::uint8_t* src_row =
          dst.samples.data() +
          static_cast<std::size_t>(mi_edge_h - 1) *
              static_cast<std::size_t>(dst.stride);
      for (int y = mi_edge_h; y < dst.height; ++y) {
        std::memcpy(dst.samples.data() +
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(dst.stride),
                    src_row, static_cast<std::size_t>(dst.stride));
      }
    }
  };

  // Wave-3 frame-type modes are compiled only in the full build (see
  // video_frames.h): the standalone component gates link this TU alone.
  if (options.long_term || options.olk || options.leading_frames > 0 ||
      options.sef || options.ra_switch || options.ras) {
#if defined(AV2ENC_VIDEO_FRAMES)
    return frames::encode_special_sequence(frames, options, result, error);
#else
    error = "frame-type modes (--video-long-term/--video-olk/--video-leading/"
            "--video-sef/--video-switch/--video-ras) need the full build";
    return false;
#endif
  }

  std::vector<inter::mc::RefFrame> no_refs;
  std::array<int, 7> no_distance{};
  const auto options_for_frame = [&options, &frame_qp](std::size_t index) {
    VideoOptions copy = options;
    copy.base_q_idx = frame_qp(index);
    copy.frame_base_q_idx.clear();
    return copy;
  };
  // -----------------------------------------------------------------------
  // §8.2.4 average-CDF / §7.5 frame-end / §5.18.2 cross-frame CDF chain.
  // Off by default: cdf_chain stays false and none of the extra state exists.
  // -----------------------------------------------------------------------
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
  const bool cdf_chain =
      options.avg_cdf || options.adaptive_cdf || options.cdf_continuity;
#else
  // Standalone component gates compile this TU without the average-CDF bank
  // and the CI module; the opt-in chain is unavailable there.
  const bool cdf_chain = false;
#endif
  if (cdf_chain && !options.frame_base_q_idx.empty() &&
      !options.cdf_continuity) {
    // Reset-per-frame form: every frame installs the §9.3 image of its own
    // coefficient qctx slice, which is only legal while every frame shares
    // that slice (§8.3.1).  --cdf-continuity needs none of this: the image is
    // carried, and a reset frame (§5.18.2) derives its defaults from its own
    // base_q_idx, so the encoder and the decoder agree across slices.
    if (!options.cdf_per_frame_qp) {
      error = "average-CDF/continuity requires a fixed base_q_idx "
              "(no --rc crf/vbr per-frame QP); pass --video-cdf-per-frame-qp "
              "to re-initialise the CDF image from each frame's slice";
      return false;
    }
    // §8.3.1 selects the default coefficient CDFs from the frame's own
    // base_q_idx, so a plan that crosses qctx slices is legal as long as the
    // default image is re-derived per frame (seed_frame_cdfs below).  Pinned
    // dav2d accepts that form: disable_cross_frame_cdf_init stays 1, so the
    // decoder installs the same per-frame §9.3 image.
  }
  if (options.cdf_continuity && !(options.avg_cdf && options.adaptive_cdf)) {
    error = "--cdf-continuity requires --avg-cdf and --adaptive-cdf";
    return false;
  }
  std::vector<std::uint16_t> cdf_tile;
  std::vector<std::uint16_t> cdf_scratch;
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
  std::vector<std::uint16_t> cdf_frame;
  std::vector<std::uint16_t> cdf_saved;
  std::vector<std::uint16_t> cdf_defaults;
  std::vector<entropy::avg_cdf::CdfTable> cdf_bank_tables;
  entropy::avg_cdf::CdfBank cdf_bank;
  std::array<std::vector<std::uint16_t>, kMaxRefSlots> cdf_slots;
  if (cdf_chain) {
    const std::size_t words = syntax::cdf_image_words();
    cdf_frame.assign(words, 0);
    cdf_tile.assign(words, 0);
    cdf_saved.assign(words, 0);
    cdf_defaults.assign(words, 0);
    // §8.3.1: the default coefficient CDFs come from the frame's raw
    // base_q_idx.  With --video-cdf-per-frame-qp every frame was checked to
    // share one qctx slice above, so the key frame's slice is the whole
    // sequence's image; each frame still resets to it below.
    const int cdf_base_q_idx =
        options.frame_base_q_idx.empty()
            ? static_cast<int>(options.base_q_idx)
            : static_cast<int>(options.frame_base_q_idx.front());
    if (!syntax::cdf_image_defaults_for_base_q_idx(cdf_base_q_idx,
                                                   cdf_defaults, &error)) {
      return false;
    }
    if (!syntax::cdf_image_tables(cdf_frame, cdf_tile, cdf_saved, cdf_defaults,
                                  cdf_bank_tables, &error)) {
      return false;
    }
    entropy::avg_cdf::CdfBankConfig bank_config;
    bank_config.enable_avg_cdf = options.avg_cdf;
    bank_config.avg_cdf_type = true;
    bank_config.context_update_tile_id = 0;  // §5.18.7.2 fallback when !avg
    if (!cdf_bank.init(cdf_bank_tables, bank_config, &error)) {
      error = "average-CDF bank: " + error;
      return false;
    }
    // §8.3.1: the default coefficient CDFs come from the *current* frame's
    // base_q_idx, so a per-frame QP plan that crosses a qctx slice must
    // rebuild the flat default image for each frame.  The four images keep
    // their size, so every CdfTable pointer stays valid and begin_frame()
    // copies the new defaults into frame/tile/saved in place.
    if (!syntax::cdf_image_defaults_for_base_q_idx(
            static_cast<int>(frame_qp(0)), cdf_defaults, &error)) {
      return false;
    }
    // CLK resets to the §9.3 defaults (§5.18.2: FrameIsIntra,
    // primary_ref_frame = PRIMARY_REF_NONE).
    cdf_bank.begin_frame(/*reset_to_defaults=*/true);
  }
  // Re-derive the §9.3 image for one frame and reset the bank to it.
  const auto seed_frame_cdfs = [&](std::size_t frame_index) -> bool {
    if (!syntax::cdf_image_defaults_for_base_q_idx(
            static_cast<int>(frame_qp(frame_index)), cdf_defaults, &error)) {
      return false;
    }
    cdf_bank.begin_frame(/*reset_to_defaults=*/true);
    return true;
  };
#endif

  const VideoOptions key_options = options_for_frame(0);

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.use_128x128_superblock = options.sb128;
  config.use_256x256_superblock = options.sb256;
  config.enable_ext_partitions = options.ext_partitions;
  config.enable_uneven_4way_partitions = options.ext_partitions;
  config.reduce_pb_aspect_ratio = options.reduce_pb_aspect_ratio;
  config.max_pb_aspect_ratio_log2_minus_1 =
      options.max_pb_aspect_ratio_log2_minus_1;
  config.base_q_idx = options.base_q_idx;
  config.seq_delta_q = options.seq_delta_q;
  config.enable_ext_seg = options.enable_ext_seg;
  // §5.4.2: --video-tile-group-compact carries the one-tile grid in the
  // sequence header; the frame writer then codes no tile_info() bits.
  config.seq_tile_info_present = options.seq_tile_info_present;
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (options.seq_seg_info_present) {
    // --video-seq-seg-info: carry the §5.4.9 table in the sequence header so
    // every frame can reuse it.  The table is the key-frame form; P frames with
    // a SKIP/GLOBALMV feature code their own seg_info() when allow_change = 1.
    config.seq_seg_info_present = true;
    config.seq_allow_seg_info_change = options.seq_allow_seg_info_change;
    if (!make_video_segmentation(options, /*primary_ref_is_none=*/true,
                                 /*frame_is_intra=*/true, config.seq_seg_info,
                                 error)) {
      return false;
    }
  }
#endif
  config.order_hint_bits = order_hint_bits;
  config.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  // --video-ref-signal-compact: explicit_ref_frame_map = 0 is the §7.7
  // amendment that removes the per-frame frame_explicit_ref_frame_map /
  // num_total_refs / ref_frame_idx bits entirely.
  config.explicit_ref_frame_map =
      !options.implicit_ref_map && !options.ref_signal_compact;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.minimal_tile_trailer = options.minimal_tile_trailer;
  config.enable_masked_compound =
      options.compound && (options.compound_wedge || options.compound_diffwtd);
  config.enable_cwp = options.enable_cwp_weighted;
  config.enable_imp_msk_bld = options.enable_imp_msk_bld;
  config.enable_df_sub_pu = options.enable_df_sub_pu;
  config.num_same_ref_compound =
      options.same_ref_compound ? 1U : 0U;
  config.enable_refmvbank = options.refmvbank;
  config.enable_mv_traj = options.mv_traj;
  config.enable_short_refresh_frame_flags = options.short_refresh;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
  if (options.mfh) {
    // §5.7: every frame header references MFH id 1 and the sequence header is
    // loaded through MfhSeqHeaderId instead of seq_header_id_in_frame_header.
    config.cur_mfh_id = 1;
  }
#else
  if (options.mfh) {
    error = "MFH signalling needs the full build (frame/mfh/mfh.cpp)";
    return false;
  }
#endif
  if (options.video_timing) {
    // §5.4.1 decoder-compatible timing block.  The nested §5.4.13 block stays
    // off unless --video-decoder-model is set (pinned dav2d f9dafbc cannot
    // parse it; avmdec can).
    config.initial_display_delay_present = true;
    config.initial_display_delay_minus_1 = 0;
    config.decoder_model_info_present = true;
    config.num_units_in_decoding_tick =
        options.timing_num_units_in_display_tick;
    config.seq_decoder_model_info_present = options.video_decoder_model;
    config.decoder_buffer_delay = options.decoder_buffer_delay;
    config.encoder_buffer_delay = options.encoder_buffer_delay;
    config.low_delay_mode_flag = options.low_delay_mode;
  }
  config.disable_drl_reorder = !options.drl_reorder;
  config.constrain_drl_reorder = !options.drl_reorder_always;
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
  config.enable_global_motion = options.enable_gme;
  config.enable_bawp = options.enable_bawp || options.enable_morph;
  config.enable_adaptive_mvd = options.enable_adaptive_mvd;
  config.enable_mvd_sign_derive = options.enable_mvd_sign_derive;
  config.enable_flex_mvres = options.enable_flex_mvres;
  // §5.4.7/§5.20.5.4: the screen-content carriers select the per-frame form
  // and code the IBC/palette mode-info path on the intra keyframe;
  // --video-scc-force selects the forced sequence form instead.
  config.seq_screen_content_tools_present =
      !options.seq_force_screen_content_tools &&
      (options.enable_morph || options.enable_palette || options.enable_ibc);
  config.seq_force_screen_content_tools =
      options.seq_force_screen_content_tools;
  config.seq_force_integer_mv = options.seq_force_integer_mv;
  config.enable_interintra = options.enable_interintra;
  config.enable_ref_frame_mvs = options.enable_ref_frame_mvs;  // §5.4.10 CDEF/CCSO sequence flags for the opt-in M2 loop filters.
  if (options.loop_filters.cdef) {
    config.enable_cdef = true;
    config.cdef_on_skip_txfm_mode =
        options.loop_filters.cdef_on_skip_txfm_mode;
    config.cdef_on_skip_txfm_always_on =
        options.loop_filters.cdef_on_skip_txfm_mode == 1;
    config.cdef_on_skip_txfm_disabled =
        options.loop_filters.cdef_on_skip_txfm_mode == 0;
  }
  if (options.loop_filters.ccso) {
    config.enable_ccso = true;
    config.ccso_unit_matches_sb_size = true;
  }
  if (options.loop_filters.gdf) {
    config.enable_gdf = true;
    config.gdf_unit_matches_sb_size = true;
  }
  if (options.loop_filters.wiener || options.loop_filters.chroma_lr ||
      options.loop_filters.pc_wiener) {
    config.enable_restoration = true;
    config.lr_disable_luma_pc_wiener = !options.loop_filters.pc_wiener;
    config.lr_disable_luma_wiener_ns = !options.loop_filters.wiener;
    config.lr_uv_present = options.loop_filters.chroma_lr;
    config.lr_disable_chroma_wiener_ns = !options.loop_filters.chroma_lr;
  }
  config.tx_mode_select = options.enable_tx_search;
  config.seq_lcr_id = options.seq_lcr_id;
  config.seq_profile_override = options.seq_profile_override;
  config.max_tlayer_id = options.max_tlayer_id;
  config.max_mlayer_id = options.max_mlayer_id;
  config.seq_max_mlayer_cnt = options.seq_max_mlayer_cnt;
  config.film_grain_params_present = options.film_grain;
  // §5.4.1 high tier: seq_tier is only present when seq_level_idx > 3.
  config.seq_tier = options.high_tier;
  if (options.high_tier && config.seq_level_idx <= 3U) {
    error = "--video-tier high needs seq_level_idx > 3 (level 4.0 or above)";
    return false;
  }
  // §5.8/§5.10 full LCR/OPS: seq_lcr_id = 1 activates the global record.
  if (options.lcr_full) {
    config.seq_lcr_id = 1;
  }
  if (options.multi_sequence && frames.size() < 2) {
    error = "--video-multi-sequence needs at least 2 input pictures";
    return false;
  }

  // §5.13/§7.3.8.9 user-defined QM: a complete OBU_QUANTIZATION_MATRIX that is
  // available before the key frame header references level 0.  §6.12 resets
  // QmProtected/QmDataPresent at every CLK, so only the key temporal unit
  // carries the QM OBU.
  std::vector<std::uint8_t> qm_obu;
  if (options.enable_qm && options.user_qm) {
    qm::QmObuConfig qm_config;
    qm_config.bit_map = 1U;  // level 0
    qm_config.chroma_info_present = true;
    qm_config.levels[0].is_default = false;
    qm::fill_user_qm_constant(&qm_config.levels[0].user, 32);
    std::string qm_error;
    qm_obu = qm::make_quantizer_matrix_obu(qm_config, &qm_error);
    if (qm_obu.empty()) {
      error = "quantizer_matrix_obu failed: " + qm_error;
      return false;
    }
  }

  // §5.8/§5.10/§5.11 full LCR/OPS (--video-lcr-full): both records describe
  // the sequence header written below, so they are built after every config
  // field is final.
  std::vector<LcrOpsObu> lcr_obus;
  if (options.lcr_full) {
    if (!make_full_lcr_ops_obus(config, lcr_obus, error)) {
      error = "keyframe LCR/OPS: " + error;
      return false;
    }
  }
  // §5.4.1/§5.18.2 second sequence header for --video-multi-sequence.  Both
  // headers are carried before the key frame; every frame selects id 1.  Its
  // only difference is seq_header_id and OrderHintBits, so a decoder that
  // ignored the id and loaded header 0 would mis-parse order_hint.
  VideoConfig first_seq_config = config;
  VideoConfig second_seq_config;
  if (options.multi_sequence) {
    second_seq_config = config;
    second_seq_config.seq_header_id = 1;
    if (second_seq_config.order_hint_bits < 16U) {
      second_seq_config.order_hint_bits =
          static_cast<std::uint8_t>(second_seq_config.order_hint_bits + 1U);
    }
    // The active header for the whole CVS is id 1; id 0 stays in the bank.
    config.seq_header_id = second_seq_config.seq_header_id;
    config.order_hint_bits = second_seq_config.order_hint_bits;
  }

  VideoFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.ref_slot = 0;
  key_header.seq_header_id = config.seq_header_id;
  key_header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (!build_video_frame_qp(key_options,
                            static_cast<int>(key_options.base_q_idx),
                            /*primary_ref_is_none=*/true,
                            /*frame_is_intra=*/true, key_header, error)) {
    error = "keyframe: " + error;
    return false;
  }
#endif

  // §5.14: plan the key picture's model before its header is written.  A new
  // model emits one film grain OBU in the key temporal unit.
  grain::GrainStreamPlanner grain_planner;
  grain::GrainFramePlan key_grain;
  if (options.film_grain) {
    if (!plan_grain_frame(frames[0], 0, grain_planner, key_grain, error)) {
      error = "keyframe film grain: " + error;
      return false;
    }
    key_header.apply_grain = key_grain.apply_grain;
    key_header.fgm_id = static_cast<std::uint8_t>(key_grain.fgm_id);
    key_header.grain_seed = key_grain.grain_seed;
  }
  if (config.screen_content_tools_enabled()) {
    // §5.18.3.3/.4: the keyframe allows screen-content tools; IBC (and the
    // morph carrier) additionally allow global intra block copy with integer
    // block vectors.  Palette-only streams code allow_intrabc = 0.
    const bool intrabc = options.enable_morph || options.enable_ibc;
    key_header.allow_screen_content_tools = true;
    key_header.force_integer_mv = true;
    key_header.allow_intrabc = intrabc;
    key_header.allow_global_intrabc = intrabc;
    key_header.allow_local_intrabc = false;
  }

  // Key frame: intra keyframe coded once per tile.
  std::uint32_t key_payload_bytes = 0;
  std::uint32_t key_intra_blocks = 0;
  std::vector<std::uint16_t> cdf_seed;
  if (cdf_chain) cdf_seed = cdf_tile;  // §8.2.2 FrameCdf seed for every tile
  [[maybe_unused]] std::uint32_t cdf_tile_index = 0;
  // Independent tile encoders can run on the host worker pool: each tile owns
  // its reconstruction, decisions and MV grid, and references are read-only.
  // Entropy coding (encoder->write()) is never parallelised, and the CDF-chain
  // path stays serial because every tile shares cdf_tile/cdf_scratch.
  const unsigned tile_workers =
      (tile_sources.size() > 1 && !cdf_chain && !options.trace)
          ? parallel::resolve_workers(options.cpu_threads)
          : 1U;
  std::vector<std::unique_ptr<InterFrameEncoder>> key_encoders(
      tile_sources.size());
  std::vector<std::string> key_tile_errors(tile_sources.size());
  std::vector<std::uint32_t> key_tile_intra(tile_sources.size(), 0);
  parallel::parallel_for(tile_sources.size(), tile_workers,
                         [&](std::size_t t) {
    CodedTile& tile = tile_sources[t];
    if (cdf_chain) cdf_tile = cdf_seed;
    std::string& tile_error = key_tile_errors[t];
    auto encoder = std::make_unique<InterFrameEncoder>(
        frames[0], no_refs, no_distance, 1, key_options, options.trace,
        /*keyframe=*/true, tile.x, tile.y, tile.w, tile.h,
        cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                  : std::span<const std::uint16_t>(),
        cdf_chain ? &cdf_scratch : nullptr);
    if (!encoder->decide(tile_error) || !encoder->deblock(tile_error)) {
      return;
    }
    paste_tile(encoder->recon(0), frame_recon[0], tile.x, tile.y,
               encoder->recon(0).width, encoder->recon(0).height);
    paste_tile(encoder->recon(1), frame_recon[1], tile.x / 2, tile.y / 2,
               encoder->recon(1).width, encoder->recon(1).height);
    paste_tile(encoder->recon(2), frame_recon[2], tile.x / 2, tile.y / 2,
               encoder->recon(2).width, encoder->recon(2).height);
    key_tile_intra[t] = encoder->intra_blocks();
    key_encoders[t] = std::move(encoder);
  });
  // Merge per-tile state in tile order: the first failing tile keeps the
  // historical error text and the counts stay order-independent.
  for (std::size_t t = 0; t < tile_sources.size(); ++t) {
    if (!key_tile_errors[t].empty()) {
      error = "keyframe encode failed: " + key_tile_errors[t];
      return false;
    }
    key_intra_blocks += key_tile_intra[t];
  }
  post::VideoFilterPlan key_filters;
  if (options.loop_filters.any()) {
    // One frame-level pass after every tile is reconstructed: the decisions
    // (CDEF strengths, LR units, GDF) are picture-global, and the decoder
    // filters across the shared edges (disable_loopfilters_across_tiles = 0).
    post::VideoLoopFilterFrame lf;
    lf.width = static_cast<int>(width);
    lf.height = static_cast<int>(height);
    lf.mi_cols = recon_mi_cols;
    lf.mi_rows = recon_mi_rows;
    lf.frame_is_intra = true;
    lf.base_q_idx = options.base_q_idx;
    lf.num_total_refs = 0;
    lf.restricted_oh = -1;
    PaddedSourcePlanes key_source;
    pad_source_420(frames[0], recon_mi_cols, recon_mi_rows, &key_source);
    for (int plane = 0; plane < 3; ++plane) {
      lf.plane[plane] = frame_recon[plane].samples.data();
      lf.stride[plane] = frame_recon[plane].stride;
      lf.source[plane] = key_source.data[plane];
      lf.source_stride[plane] = key_source.stride[plane];
    }
    // §7.18.1 Skips[] / §7.20.1 LrTxSkip for the screen-content keyframe.  An
    // IBC skip leaf codes skip_flag = 1, and the real decoder leaves such a
    // block out of CDEF and LR; ordinary intra leaves (including palette
    // leaves, which code eob = 0) contribute nothing to Skips[].  Only build
    // the list on a screen-content carrier so every historical keyframe keeps
    // the previous all-zero grid.
    if (config.screen_content_tools_enabled()) {
      std::vector<post::VideoLoopFilterBlock> key_blocks;
      for (std::size_t t = 0; t < tile_sources.size(); ++t) {
        const CodedTile& tile = tile_sources[t];
        const int tile_mi_row = tile.y / 4;
        const int tile_mi_col = tile.x / 4;
        for (const syntax::BlockSyntax& leaf : key_encoders[t]->blocks()) {
          if (leaf.partition !=
              static_cast<std::uint8_t>(PartitionType::kNone)) {
            continue;
          }
          post::VideoLoopFilterBlock block;
          block.mi_row = tile_mi_row + static_cast<int>(leaf.mi_row);
          block.mi_col = tile_mi_col + static_cast<int>(leaf.mi_col);
          block.w4 = partition::block_width(
                         static_cast<partition::BlockSize>(leaf.bsize)) >>
                     2;
          block.h4 = partition::block_height(
                         static_cast<partition::BlockSize>(leaf.bsize)) >>
                     2;
          block.skip = (leaf.flags & syntax::kFlagIsIbc) != 0;
          // Palette / ordinary intra leaves keep LrTxSkip = 0 even at eob 0.
          block.luma_eob_zero = false;
          key_blocks.push_back(block);
        }
      }
      lf.blocks = key_blocks;
    }
    if (!post::run_video_loop_filters(lf, options.loop_filters, &key_filters,
                                      nullptr, &error)) {
      return false;
    }
  }
  pad_recon_bottom();
  for (std::size_t t = 0; t < tile_sources.size(); ++t) {
    CodedTile& tile = tile_sources[t];
    InterFrameEncoder& encoder = *key_encoders[t];
    // The syntax writer imports the CDF seed when it is constructed inside
    // write(); replay the frame seed here because the previous tile's final
    // CDFs replaced the shared buffer.
    if (cdf_chain) cdf_tile = cdf_seed;
    if (!encoder.write(tile.payload, tile.blocks, tile.levels,
                       key_filters.any() ? &key_filters : nullptr, error)) {
      error = "keyframe encode failed: " + error;
      return false;
    }
    tile.filter_plan = key_filters;
    if (tile.payload.empty()) {
      error = "keyframe produced an empty tile payload";
      return false;
    }
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
    if (cdf_chain) {
      std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
      if (!cdf_bank.save_tile_cdfs(
              static_cast<std::uint32_t>(t),
              static_cast<std::uint32_t>(tile_sources.size()))) {
        error = "average-CDF save_tile_cdfs rejected tile " +
                std::to_string(t);
        return false;
      }
    }
#endif
    key_payload_bytes += static_cast<std::uint32_t>(tile.payload.size());
  }
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
  if (cdf_chain) {
    // §7.5 frame_end_update_cdf(), then OBU_CLOSED_LOOP_KEY refreshes every
    // reference slot (§5.18.2).
    cdf_bank.average_at_frame_end();
    for (std::vector<std::uint16_t>& slot : cdf_slots) {
      slot.assign(cdf_frame.begin(), cdf_frame.end());
    }
  }
#endif

  std::vector<std::uint8_t> key_unit;
  {
    config.base_q_idx = frame_qp(0);
    apply_video_loop_filter_frame_header(tile_sources[0].filter_plan,
                                         key_header);
    std::vector<std::uint8_t> ci;
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
    if (options.video_timing) {
      // §5.15 content interpretation OBU carrying §5.4.12 timing_info().  It
      // is the first OBU of the coded output frame unit (§7.3.3), after the
      // temporal delimiter and sequence header.
      av2enc::ci::ContentInterpretation params =
          av2enc::ci::default_content_interpretation();
      params.has_timing_info = true;
      params.timing.num_units_in_display_tick =
          options.timing_num_units_in_display_tick;
      params.timing.time_scale = options.timing_time_scale;
      params.timing.equal_picture_interval = true;
      params.timing.num_ticks_per_picture_minus_1 = 0;
      ci = av2enc::ci::make_content_interpretation_obu(params, error);
      if (ci.empty()) {
        error = "content interpretation (timing) OBU: " + error;
        return false;
      }
    }
#endif
    std::vector<std::uint8_t> mfh_obu;
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
    if (options.mfh) {
      // One minimal MFH: sequence header 0, id 1, no frame size / deblocking
      // update / segmentation table.  Availability precedes the first frame
      // header that references it (§7.3.8.7).
      frame::mfh::MultiFrameHeader header;
      header.seq_header_id = 0;
      header.mfh_id = 1;
      header.frame_size_present = false;
      header.deblocking_filter_update = false;
      header.seg_info_present = false;
      std::string mfh_error;
      mfh_obu = frame::mfh::make_multi_frame_header_obu(header, &mfh_error);
      if (mfh_obu.empty()) {
        error = "MFH OBU: " + mfh_error;
        return false;
      }
    }
#endif
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    std::vector<std::uint8_t> seq;
    std::vector<std::uint8_t> obu;
    if (tiled) {
      seq = post::make_video_sequence_header_obu(config, tile_sequence, &error);
      if (seq.empty()) {
        error = "keyframe: tiled sequence header: " + error;
        return false;
      }
      post::VideoTiledFrameHeader header;
      header.keyframe = true;
      header.order_hint = key_header.order_hint;
      header.ref_slot = key_header.ref_slot;
      header.refresh_frame_flags = key_header.refresh_frame_flags;
      header.base_q_idx = config.base_q_idx;
      header.tx_mode_select = options.enable_tx_search;
      header.cdef = post::make_tiled_cdef_params(key_header.cdef);
      header.ccso = post::make_tiled_ccso_params(key_header.ccso);
      header.lr = key_header.lr;
      header.gdf = key_header.gdf;
      header.cdef_frame_enable = key_header.cdef.enabled;
      header.ccso_frame_flag = key_header.ccso.enabled;
      header.apply_grain = key_header.apply_grain;
      header.fgm_id = key_header.fgm_id;
      header.grain_seed = key_header.grain_seed;
      std::vector<std::vector<std::uint8_t>> payloads;
      payloads.reserve(tile_sources.size());
      for (const CodedTile& tile : tile_sources) {
        payloads.push_back(tile.payload);
      }
      post::VideoTiledFrame frame;
      if (!post::assemble_video_tile_group_from_payloads(
              config, tile_sequence, header, tile_layout, payloads, &frame,
              &error)) {
        error = "keyframe: " + error;
        return false;
      }
      obu = std::move(frame.obu);
    } else {
      seq = make_sequence_header_obu(first_seq_config);
      obu = make_frame_obu(config, key_header, tile_sources[0].payload);
    }
    std::vector<std::uint8_t> second_seq;
    if (options.multi_sequence) {
      second_seq = make_sequence_header_obu(second_seq_config);
    }
    key_unit.reserve(td.size() + seq.size() + second_seq.size() +
                     qm_obu.size() + ci.size() + mfh_obu.size() +
                     obu.size());
    key_unit.insert(key_unit.end(), td.begin(), td.end());
    for (const LcrOpsObu& lcr_obu : lcr_obus) {
      key_unit.insert(key_unit.end(), lcr_obu.bytes.begin(),
                      lcr_obu.bytes.end());
    }
    key_unit.insert(key_unit.end(), seq.begin(), seq.end());
    key_unit.insert(key_unit.end(), second_seq.begin(), second_seq.end());
    key_unit.insert(key_unit.end(), qm_obu.begin(), qm_obu.end());
    key_unit.insert(key_unit.end(), ci.begin(), ci.end());
    key_unit.insert(key_unit.end(), mfh_obu.begin(), mfh_obu.end());
    key_unit.insert(key_unit.end(), obu.begin(), obu.end());
  }
  if (options.film_grain &&
      !insert_grain_obu(key_grain, key_unit, error)) {
    error = "keyframe " + error;
    return false;
  }
  result.units.push_back(std::move(key_unit));
  result.recon.push_back(
      to_recon_frame(frame_recon[0], frame_recon[1], frame_recon[2], width,
                     height));
  result.frame_bytes.push_back(
      static_cast<std::uint32_t>(result.units.back().size()));
  result.tile_bytes.push_back(key_payload_bytes);
  result.inter_blocks.push_back(0);
  result.intra_blocks.push_back(key_intra_blocks);

  // Decoded picture buffer.  Key §7.23 keeps only the first refreshed slot
  // valid, so the first P frame has a single reference.  Later frames reference
  // the most recently decoded slots and refresh a rotating slot.
  std::array<RefSlot, kMaxRefSlots> slots;
  std::array<SlotField, kMaxRefSlots> slot_fields;
  // Input-picture index each slot was coded from.  The §5.18.2 order hint is
  // 4 bits in the flat chain, so it wraps every 16 frames and cannot index the
  // source array; --video-drift-refresh measures a reference against the
  // picture it was actually coded from.
  std::array<std::size_t, kMaxRefSlots> slot_source_index{};
  std::vector<int> recent_slots;
  slots[0].valid = true;
  slots[0].is_key = true;
  slots[0].order_hint = 0;
  for (int plane = 0; plane < 3; ++plane) {
    slots[0].plane[plane] = frame_recon[plane];
  }
  recent_slots.push_back(options.fixed_write_slot >= 0
                             ? options.fixed_write_slot
                             : 0);

  // §7.23 mirror the §7.7 derivation reads.  The key frame refreshes every
  // slot but only the first stays valid, so the first P frame sees one
  // candidate.  `num_ref_frames` is the signalled pool size.
  inter::mv_stack_full::RefSlotBank ref_bank;
  ref_bank.num_ref_frames = num_ref_frames;
  if (options.implicit_ref_map) {
    if (!ref_bank.refresh(static_cast<int>(key_header.refresh_frame_flags),
                          static_cast<int>(key_header.order_hint),
                          static_cast<int>(width),
                          static_cast<int>(height), frame_qp(0),
                          /*key_frame=*/true, 0, 0, true, true, -1, &error)) {
      error = "keyframe reference bank: " + error;
      return false;
    }
  }
  const inter::mv_stack_full::DependencyMaps ref_deps =
      implicit_dependency_maps(config, 0);
  init_slot_field(&slot_fields[0], recon_mi_cols, recon_mi_rows);

  const auto make_reference = [&](int slot) {
    inter::mc::RefFrame ref;
    ref.plane[0] = inter::mc::PlaneView{
        slots[static_cast<std::size_t>(slot)].plane[0].samples.data(),
        slots[static_cast<std::size_t>(slot)].plane[0].stride};
    ref.plane[1] = inter::mc::PlaneView{
        slots[static_cast<std::size_t>(slot)].plane[1].samples.data(),
        slots[static_cast<std::size_t>(slot)].plane[1].stride};
    ref.plane[2] = inter::mc::PlaneView{
        slots[static_cast<std::size_t>(slot)].plane[2].samples.data(),
        slots[static_cast<std::size_t>(slot)].plane[2].stride};
    ref.mi_cols = recon_mi_cols;
    ref.mi_rows = ref_mi_rows;
    return ref;
  };

  // §5.20.7.2 PrevSegmentIds carried from the primary reference frame for
  // segmentation_temporal_update.  Only the single-tile segmentation path
  // (which rejects --tiles > 1x1) uses it.
  std::vector<std::uint8_t> prev_segment_map;
  // --video-drift-refresh accounting.  `drift_mean_inter_payload` is the
  // running mean payload of the frames that were not refreshes, so a refresh
  // is charged only the bytes it added on top of the chain's normal cost.
  double drift_spent_bytes = 0.0;
  double drift_mean_inter_payload = 0.0;
  std::uint64_t drift_inter_count = 0;
  // The drift anchor is the key picture's own measured reconstruction PSNR, so
  // the allowance follows the operating point instead of an absolute dB.
  const double drift_anchor_psnr =
      options.drift_refresh
          ? reference_luma_psnr(slots[0].plane[0], frames[0],
                                static_cast<int>(width),
                                static_cast<int>(height))
          : 0.0;
  for (std::size_t i = 1; i < frames.size(); ++i) {
    const std::uint32_t order_hint =
        static_cast<std::uint32_t>(i) &
        static_cast<std::uint32_t>((1U << order_hint_bits) - 1U);
    // Reference list for this frame: the explicit recency list, or the §7.7
    // implicit mapping the decoder derives from the same slot state.
    std::vector<int> frame_ref_slots;
    std::array<int, 7> derived_distance{};
    if (options.implicit_ref_map) {
      inter::mv_stack_full::ImplicitRefMap implicit;
      if (!derive_implicit_fields(ref_bank, static_cast<int>(i),
                                  order_hint_bits, static_cast<int>(width),
                                  static_cast<int>(height), 0, ref_deps,
                                  &implicit, &error)) {
        error = "P-frame " + std::to_string(i) + " implicit ref map: " + error;
        return false;
      }
      const inter::mv_stack_full::RefMap& map = implicit.stage1;
      frame_ref_slots.reserve(static_cast<std::size_t>(map.num_total_refs));
      for (int r = 0; r < map.num_total_refs; ++r) {
        frame_ref_slots.push_back(map.ref_frame_idx[r]);
        derived_distance[static_cast<std::size_t>(r)] = map.scores_distance[r];
      }
    } else {
      const int explicit_count =
          std::min<int>(num_total_refs, static_cast<int>(recent_slots.size()));
      frame_ref_slots.assign(recent_slots.begin(),
                             recent_slots.begin() + explicit_count);
    }
    const int ref_count = static_cast<int>(frame_ref_slots.size());
    if (options.trace && options.implicit_ref_map) {
      trace_implicit_map("P", i, ref_count, frame_ref_slots);
    }
    // §5.18.2 choose_primary_secondary_ref_frame(): only a reference whose
    // FrameType is INTER_FRAME competes for primary_ref_frame.  When none
    // does, §5.18.2 forces primary_ref_frame = PRIMARY_REF_NONE and
    // disable_cross_frame_cdf_init = 1, so the decoder never runs §7.4.4
    // load_cdfs() for this frame and starts from the §9.3 image instead.
    bool derived_primary_none = true;
    for (int r = 0; r < ref_count; ++r) {
      const int ref_slot = frame_ref_slots[static_cast<std::size_t>(r)];
      if (!slots[static_cast<std::size_t>(ref_slot)].is_key) {
        derived_primary_none = false;
        break;
      }
    }
    std::vector<inter::mc::RefFrame> references;
    references.reserve(static_cast<std::size_t>(ref_count));
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < ref_count; ++r) {
      const int slot = frame_ref_slots[static_cast<std::size_t>(r)];
      references.push_back(make_reference(slot));
      frame_distance[static_cast<std::size_t>(r)] =
          options.implicit_ref_map
              ? derived_distance[static_cast<std::size_t>(r)]
              : relative_dist(
                    order_hint,
                    slots[static_cast<std::size_t>(slot)].order_hint,
                    order_hint_bits);
    }

    VideoOptions frame_options = options_for_frame(i);
    // --video-drift-refresh: measure the primary reference's actual
    // reconstruction error (stored reference plane against the source picture
    // that produced it) and, when it is below the threshold and the sequence
    // allowance is not exhausted, code this frame as a refresh: lower the
    // frame QP and code the residual of every inter leaf so the error does not
    // survive into the following frames.  `effective_frame_qp` is the single
    // source of truth for the frame header, the reference-bank record and the
    // quantiser; without the flag it is exactly frame_qp(i).
    int effective_frame_qp = static_cast<int>(frame_options.base_q_idx);
    bool drift_refresh_frame = false;
    if (options.drift_refresh && !frame_ref_slots.empty()) {
      const RefSlot& primary =
          slots[static_cast<std::size_t>(frame_ref_slots.front())];
      const std::size_t source_index =
          slot_source_index[static_cast<std::size_t>(frame_ref_slots.front())];
      double ref_psnr = 99.0;
      if (primary.valid && source_index < frames.size() &&
          frames[source_index].width == width &&
          frames[source_index].height == height) {
        ref_psnr = reference_luma_psnr(primary.plane[0], frames[source_index],
                                       static_cast<int>(width),
                                       static_cast<int>(height));
      }
      drift_refresh_frame =
          ref_psnr <
              drift_anchor_psnr - options.drift_refresh_threshold_db &&
          drift_spent_bytes <
              static_cast<double>(options.drift_refresh_budget);
      if (options.trace) {
        std::cerr << "drift frame=" << i << " slot=" << frame_ref_slots.front()
                  << " anchor=" << drift_anchor_psnr
                  << " ref_psnr=" << ref_psnr
                  << " spent=" << drift_spent_bytes
                  << " refresh=" << (drift_refresh_frame ? 1 : 0) << "\n";
      }
      if (drift_refresh_frame) {
        effective_frame_qp =
            std::max(1, std::min(255, effective_frame_qp -
                                          options.drift_refresh_qp));
        frame_options.base_q_idx =
            static_cast<std::uint16_t>(effective_frame_qp);
        // Code the residual of every inter leaf: the leaf RD price refuses it
        // at these QPs (the error is a few LSB per sample), so the refresh has
        // to bypass that decision for the frame whose job is to reset it.
        frame_options.force_inter_residual = true;
      }
    }
    // §5.14: plan this picture's grain model in coding order before the frame
    // header is written; a changed model emits one film grain OBU.
    //
    // A hidden picture (the SEF wrapper codes the last picture with
    // immediate_output_frame = 0) is not output, so §5.18.10.1 infers
    // apply_grain = 0 and codes no config bits.  Cancelling here also keeps the
    // planner's slot state free of models the decoder would never load.
    const bool grain_output =
        !(options.hide_last_picture && i + 1U == frames.size());
    grain::GrainFramePlan grain_plan;
    if (options.film_grain) {
      if (grain_output &&
          !plan_grain_frame(frames[i], static_cast<std::uint32_t>(i),
                            grain_planner, grain_plan, error)) {
        error = "P-frame " + std::to_string(i) + " film grain: " + error;
        return false;
      }
      if (!grain_output) {
        grain_plan = grain_planner.CancelFrame(static_cast<std::uint32_t>(i));
      }
    }
    // §5.18.9 global motion: estimate one model per explicit reference before
    // the block decisions so GLOBALMV and the header use the same parameters.
    // Off by default; `gm_plan` then stays the identity plan.
    GmFramePlan gm_plan;
    const GmFramePlan* gm_ptr = nullptr;
#if defined(AV2ENC_VIDEO_GME)
    if (options.enable_gme && !tiled) {
      std::vector<int> ref_order_hint(static_cast<std::size_t>(ref_count));
      std::vector<std::uint8_t> ref_slots(static_cast<std::size_t>(ref_count));
      std::vector<int> ref_distance(static_cast<std::size_t>(ref_count));
      for (int r = 0; r < ref_count; ++r) {
        const int slot = frame_ref_slots[static_cast<std::size_t>(r)];
        ref_order_hint[static_cast<std::size_t>(r)] = static_cast<int>(
            slots[static_cast<std::size_t>(slot)].order_hint);
        ref_slots[static_cast<std::size_t>(r)] =
            static_cast<std::uint8_t>(slot);
        ref_distance[static_cast<std::size_t>(r)] =
            frame_distance[static_cast<std::size_t>(r)];
      }
      if (!estimate_gm_plan_cpu(frames[i], references, ref_order_hint,
                                ref_slots, ref_distance, frame_options,
                                gm_plan, error)) {
        error = "P-frame " + std::to_string(i) + " global motion: " + error;
        return false;
      }
      gm_ptr = &gm_plan;
    }
#endif
    std::uint32_t payload_bytes = 0;
    std::uint32_t inter_blocks = 0;
    std::uint32_t intra_blocks = 0;
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
    if (cdf_chain) {
      if (options.cdf_continuity && !derived_primary_none) {
        // §7.4.4 load_cdfs(ref_frame_idx[primary_ref_frame]); we signal
        // primary_ref_frame = 0, so the first reference's saved CDFs are
        // installed into the frame state.
        const int primary_slot = frame_ref_slots.front();
        std::copy(cdf_slots[static_cast<std::size_t>(primary_slot)].begin(),
                  cdf_slots[static_cast<std::size_t>(primary_slot)].end(),
                  cdf_frame.begin());
        cdf_bank.begin_frame(/*reset_to_defaults=*/false);
      } else if (!seed_frame_cdfs(i)) {
        return false;
      }
    }
#endif
    std::vector<std::uint16_t> tile_cdf_seed;
    if (cdf_chain) tile_cdf_seed = cdf_tile;  // §8.2.2 FrameCdf seed per tile
    [[maybe_unused]] std::uint32_t tile_cdf_index = 0;
    // §7.20.1 needs the reference OrderHints for the GDF table derivation.
    std::array<int, 7> ref_order_hints{};
    for (int r = 0; r < ref_count; ++r) {
      ref_order_hints[static_cast<std::size_t>(r)] = static_cast<int>(
          slots[static_cast<std::size_t>(
              frame_ref_slots[static_cast<std::size_t>(r)])]
              .order_hint);
    }
    // §7.9 + §7.12.2.7/.8: project this frame's reference motion fields into
    // the current frame's motion field for the temporal MV candidates.
    TemporalFrameContext temporal_ctx;
    MotionFieldFrame motion_field;
    if (options.enable_ref_frame_mvs) {
      SavedFieldRef ref_fields[kRefsPerFrame] = {};
      int ref_slots[kRefsPerFrame] = {};
      std::uint8_t ref_is_key[kRefsPerFrame] = {};
      for (int r = 0; r < ref_count; ++r) {
        const int slot = recent_slots[static_cast<std::size_t>(r)];
        ref_fields[r] = make_saved_ref(slot_fields[static_cast<std::size_t>(slot)]);
        ref_slots[r] = slot;
        ref_is_key[r] = slots[static_cast<std::size_t>(slot)].is_key ? 1U : 0U;
      }
      if (!estimate_motion_field(
              ref_fields, ref_slots, ref_order_hints.data(), ref_is_key,
              /*ref_valid=*/true, ref_count, static_cast<int>(order_hint),
              static_cast<int>(width), static_cast<int>(height),
              recon_mi_cols, recon_mi_rows, /*proj_step=*/1,
              options.mv_traj, &motion_field, error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      temporal_ctx.use_ref_frame_mvs = true;
      temporal_ctx.order_hint = static_cast<int>(order_hint);
      temporal_ctx.num_past_refs = motion_field.config.num_past_refs;
      temporal_ctx.num_future_refs = motion_field.config.num_future_refs;
      temporal_ctx.closest_past = motion_field.result.closest_past;
      temporal_ctx.closest_future = motion_field.result.closest_future;
      temporal_ctx.field = &motion_field.field;
      temporal_ctx.mv_traj = options.mv_traj ? &motion_field.traj : nullptr;
    }
    std::vector<std::unique_ptr<InterFrameEncoder>> tile_encoders(
        tile_sources.size());
    std::vector<std::string> tile_errors(tile_sources.size());
    std::vector<std::uint32_t> tile_inter(tile_sources.size(), 0);
    std::vector<std::uint32_t> tile_intra(tile_sources.size(), 0);
    parallel::parallel_for(tile_sources.size(), tile_workers,
                           [&](std::size_t t) {
      CodedTile& tile = tile_sources[t];
      if (cdf_chain) cdf_tile = tile_cdf_seed;
      std::string& tile_error = tile_errors[t];
      auto encoder = std::make_unique<InterFrameEncoder>(
          frames[i], references, frame_distance, ref_count, frame_options,
          options.trace, /*keyframe=*/false, tile.x, tile.y, tile.w, tile.h,
          cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                    : std::span<const std::uint16_t>(),
          cdf_chain ? &cdf_scratch : nullptr, gm_ptr, ref_order_hints,
          /*primary_ref_is_none=*/i == 1, &temporal_ctx,
          i == 1 ? std::span<const std::uint8_t>()
                 : std::span<const std::uint8_t>(prev_segment_map));
      if (!encoder->decide(tile_error) || !encoder->deblock(tile_error)) {
        return;
      }
      tile_inter[t] = encoder->inter_blocks();
      tile_intra[t] = encoder->intra_blocks();
      paste_tile(encoder->recon(0), frame_recon[0], tile.x, tile.y,
                 encoder->recon(0).width, encoder->recon(0).height);
      paste_tile(encoder->recon(1), frame_recon[1], tile.x / 2, tile.y / 2,
                 encoder->recon(1).width, encoder->recon(1).height);
      paste_tile(encoder->recon(2), frame_recon[2], tile.x / 2, tile.y / 2,
                 encoder->recon(2).width, encoder->recon(2).height);
      tile_encoders[t] = std::move(encoder);
    });
    for (std::size_t t = 0; t < tile_sources.size(); ++t) {
      if (!tile_errors[t].empty()) {
        error = "P-frame " + std::to_string(i) + ": " + tile_errors[t];
        return false;
      }
      inter_blocks += tile_inter[t];
      intra_blocks += tile_intra[t];
    }
    for (const std::unique_ptr<InterFrameEncoder>& encoder : tile_encoders) {
      result.temporal_candidates +=
          static_cast<std::uint64_t>(encoder->temporal_added());
      result.temporal_selected +=
          static_cast<std::uint64_t>(encoder->temporal_selected());
      result.same_ref_blocks +=
          static_cast<std::uint64_t>(encoder->same_ref_blocks());
      result.refmvbank_candidates +=
          static_cast<std::uint64_t>(encoder->refmvbank_added());
      result.refmvbank_selected +=
          static_cast<std::uint64_t>(encoder->refmvbank_selected());
      result.mv_traj_derived +=
          static_cast<std::uint64_t>(encoder->mv_traj_derived());
      result.amvd_selected +=
          static_cast<std::uint64_t>(encoder->amvd_selected());
      result.flex_precision_selected +=
          static_cast<std::uint64_t>(encoder->flex_precision_selected());
      result.derived_sign_selected +=
          static_cast<std::uint64_t>(encoder->derived_sign_selected());
    }
#if defined(AV2ENC_VIDEO_DELTAQ)
    // Save this frame's §5.20.5.7 SegmentIds for the next frame's
    // segmentation_temporal_update prediction.  The segmentation path is
    // single-tile, so one map covers the frame.
    if (!tile_encoders.empty() && options.enable_segmentation) {
      prev_segment_map = tile_encoders[0]->segment_map();
    }
#endif
    post::VideoFilterPlan frame_filters;
    if (options.loop_filters.any()) {
      // One picture-global pass after every tile is reconstructed.  CDEF
      // strengths, LR units and GDF parameters are frame-level decisions, and
      // the tiled sequence signals disable_loopfilters_across_tiles = 0 so the
      // decoder filters the same shared edges.
      post::VideoLoopFilterFrame lf;
      lf.width = static_cast<int>(width);
      lf.height = static_cast<int>(height);
      lf.mi_cols = recon_mi_cols;
      lf.mi_rows = recon_mi_rows;
      lf.frame_is_intra = false;
      lf.base_q_idx = frame_options.base_q_idx;
      lf.num_total_refs = ref_count;
      lf.restricted_oh = -1;
      for (int r = 0; r < 2; ++r) {
        lf.ref_order_hints[r] = ref_order_hints[static_cast<std::size_t>(r)];
        lf.frame_distance[r] = frame_distance[static_cast<std::size_t>(r)];
      }
      std::vector<post::VideoLoopFilterBlock> filter_blocks;
      for (std::size_t t = 0; t < tile_sources.size(); ++t) {
        const CodedTile& tile = tile_sources[t];
        const int tile_mi_row = tile.y / 4;
        const int tile_mi_col = tile.x / 4;
        for (const syntax::BlockSyntax& leaf : tile_encoders[t]->blocks()) {
          if (leaf.partition !=
              static_cast<std::uint8_t>(PartitionType::kNone)) {
            continue;
          }
          post::VideoLoopFilterBlock block;
          block.mi_row = tile_mi_row + static_cast<int>(leaf.mi_row);
          block.mi_col = tile_mi_col + static_cast<int>(leaf.mi_col);
          block.w4 =
              partition::block_width(
                  static_cast<partition::BlockSize>(leaf.bsize)) >>
              2;
          block.h4 =
              partition::block_height(
                  static_cast<partition::BlockSize>(leaf.bsize)) >>
              2;
          // §7.18.1 Skips[]: an inter skip leaf and an IBC skip leaf both code
          // skip_flag = 1; with CdefOnSkipTxfm DISABLED the decoder leaves a
          // fully skipped 8x8 block unfiltered.
          block.skip = (leaf.is_inter && leaf.skip_flag != 0) ||
                       (leaf.flags & syntax::kFlagIsIbc) != 0;
          filter_blocks.push_back(block);
        }
      }
      lf.blocks = filter_blocks;
      PaddedSourcePlanes frame_source;
      pad_source_420(frames[i], recon_mi_cols, recon_mi_rows, &frame_source);
      for (int plane = 0; plane < 3; ++plane) {
        lf.plane[plane] = frame_recon[plane].samples.data();
        lf.stride[plane] = frame_recon[plane].stride;
        lf.source[plane] = frame_source.data[plane];
        lf.source_stride[plane] = frame_source.stride[plane];
      }
      if (!post::run_video_loop_filters(lf, options.loop_filters,
                                        &frame_filters, nullptr, &error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
    }
    pad_recon_bottom();
    for (std::size_t t = 0; t < tile_sources.size(); ++t) {
      CodedTile& tile = tile_sources[t];
      InterFrameEncoder& encoder = *tile_encoders[t];
      if (cdf_chain) cdf_tile = tile_cdf_seed;
      if (!encoder.write(tile.payload, tile.blocks, tile.levels,
                         frame_filters.any() ? &frame_filters : nullptr,
                         error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      tile.filter_plan = frame_filters;
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
      if (cdf_chain) {
        std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
        if (!cdf_bank.save_tile_cdfs(
                static_cast<std::uint32_t>(t),
                static_cast<std::uint32_t>(tile_sources.size()))) {
          error = "average-CDF save_tile_cdfs rejected tile " +
                  std::to_string(t);
          return false;
        }
      }
#endif
      payload_bytes += static_cast<std::uint32_t>(tile.payload.size());
    }

    const int write_slot = options.fixed_write_slot >= 0
                               ? options.fixed_write_slot
                               : static_cast<int>(
                                     i % static_cast<std::size_t>(
                                             num_ref_frames));
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
    if (cdf_chain) {
      // save_tile_cdfs() already ran per tile; §7.5 frame_end_update_cdf(),
      // then §7.23 save_cdfs(write_slot).
      cdf_bank.average_at_frame_end();
      std::vector<std::uint16_t>& saved_slot =
          cdf_slots[static_cast<std::size_t>(write_slot)];
      std::copy(cdf_frame.begin(), cdf_frame.end(), saved_slot.begin());
    }
#endif
    VideoFrameHeader header;
    header.keyframe = false;
    header.seq_header_id = config.seq_header_id;
    header.order_hint = order_hint;
    if (cdf_chain) {
      // --video-ref-signal-compact omits the primary-ref block: §5.18.2
      // choose_primary_secondary_ref_frame() resolves the one live reference
      // to slot 0 for every frame that has an INTER_FRAME reference, and
      // frame 1 (key-only reference) derives PRIMARY_REF_NONE either way, so
      // the coded CDF seed is unchanged.
      header.signal_primary_ref_frame =
          options.cdf_continuity && !options.ref_signal_compact;
      // §5.18.2 forces disable_cross_frame_cdf_init = 1 for a frame whose
      // reference list holds no INTER_FRAME (choose_primary_secondary_ref_frame
      // returns PRIMARY_REF_NONE).  The header codes the bit the decoder will
      // actually use, so the §6.17.2 reset and the §7.4.4 load agree.
      header.disable_cross_frame_cdf_init =
          !options.cdf_continuity || derived_primary_none;
      header.primary_ref_frame = 0;
    }
    header.num_total_refs = static_cast<std::uint8_t>(ref_count);
    for (int r = 0; r < ref_count; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(frame_ref_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    header.frame_explicit_ref_frame_map =
        !options.implicit_ref_map && !options.ref_signal_compact;
    header.refresh_frame_flags = 1U << write_slot;
    header.has_refresh_frame_flags = true;
    header.frame_to_refresh = static_cast<std::uint8_t>(write_slot);
    if (options.short_refresh) ++result.short_refresh_frames;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    // §5.18.3.3: with --video-scc-force the sequence infers
    // allow_screen_content_tools = 1, so the eligible intra leaves code
    // palette_mode_info(); force_integer_mv stays with the frame unless
    // --video-force-integer-mv pins it in the sequence.
    header.allow_screen_content_tools =
        config.seq_force_screen_content_tools;
    header.force_integer_mv = config.seq_force_integer_mv;
    header.use_ref_frame_mvs = options.enable_ref_frame_mvs;
    // SEF wrapper: code the last picture hidden so a following SEF unit can
    // re-show it; the decoder output count is unchanged.
    header.immediate_output_frame =
        !(options.hide_last_picture && i + 1U == frames.size());
    header.reference_select = options.compound && ref_count > 1;
    header.allow_df_sub_pu = options.enable_df_sub_pu;
    header.gm = gm_plan;
    if (options.film_grain) {
      header.apply_grain = grain_plan.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(grain_plan.fgm_id);
      header.grain_seed = grain_plan.grain_seed;
    }
    apply_video_loop_filter_frame_header(tile_sources[0].filter_plan, header);
#if defined(AV2ENC_VIDEO_DELTAQ)
    // §5.18.2 choose_primary_secondary_ref_frame(): only a reference whose
    // frame type is INTER_FRAME counts.  Frame 1 references the key frame, so
    // its derived primary reference is PRIMARY_REF_NONE; every later P frame
    // references an earlier P frame and codes update_map/temporal_update.
    if (!build_video_frame_qp(frame_options,
                              static_cast<int>(frame_options.base_q_idx),
                              /*primary_ref_is_none=*/i == 1,
                              /*frame_is_intra=*/false, header, error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }
#endif

    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    config.base_q_idx = static_cast<std::uint8_t>(effective_frame_qp);
    std::vector<std::uint8_t> obu;
    if (tiled) {
      post::VideoTiledFrameHeader tiled_header;
      tiled_header.keyframe = false;
      tiled_header.order_hint = header.order_hint;
      tiled_header.ref_slot = header.ref_slot;
      tiled_header.refresh_frame_flags = header.refresh_frame_flags;
      tiled_header.base_q_idx = config.base_q_idx;
      tiled_header.tx_mode_select = options.enable_tx_search;
      tiled_header.cdef = post::make_tiled_cdef_params(header.cdef);
      tiled_header.ccso = post::make_tiled_ccso_params(header.ccso);
      tiled_header.lr = header.lr;
      tiled_header.gdf = header.gdf;
      tiled_header.cdef_frame_enable = header.cdef.enabled;
      tiled_header.ccso_frame_flag = header.ccso.enabled;
      tiled_header.apply_grain = header.apply_grain;
      tiled_header.fgm_id = header.fgm_id;
      tiled_header.grain_seed = header.grain_seed;
      std::vector<std::vector<std::uint8_t>> payloads;
      payloads.reserve(tile_sources.size());
      for (const CodedTile& tile : tile_sources) {
        payloads.push_back(tile.payload);
      }
      post::VideoTiledFrame frame;
      if (!post::assemble_video_tile_group_from_payloads(
              config, tile_sequence, tiled_header, tile_layout, payloads,
              &frame, &error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      obu = std::move(frame.obu);
    } else {
      obu = make_frame_obu(config, header, tile_sources[0].payload);
    }
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain &&
        !insert_grain_obu(grain_plan, unit, error)) {
      error = "P-frame " + std::to_string(i) + " " + error;
      return false;
    }
    result.units.push_back(std::move(unit));

    result.recon.push_back(
        to_recon_frame(frame_recon[0], frame_recon[1], frame_recon[2], width,
                       height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(payload_bytes);
    result.inter_blocks.push_back(inter_blocks);
    result.intra_blocks.push_back(intra_blocks);
    if (options.drift_refresh) {
      if (drift_refresh_frame) {
        const double extra =
            static_cast<double>(payload_bytes) - drift_mean_inter_payload;
        if (extra > 0.0) drift_spent_bytes += extra;
        ++result.drift_refresh_frames;
        result.drift_refresh_bytes +=
            static_cast<std::uint64_t>(extra > 0.0 ? extra : 0.0);
      } else {
        drift_mean_inter_payload =
            (drift_mean_inter_payload *
                 static_cast<double>(drift_inter_count) +
             static_cast<double>(payload_bytes)) /
            static_cast<double>(drift_inter_count + 1U);
        ++drift_inter_count;
      }
    }

    // Store the reconstruction and refresh the rotating slot.
    slots[static_cast<std::size_t>(write_slot)].valid = true;
    slots[static_cast<std::size_t>(write_slot)].is_key = false;
    slots[static_cast<std::size_t>(write_slot)].order_hint = order_hint;
    slot_source_index[static_cast<std::size_t>(write_slot)] = i;
    for (int plane = 0; plane < 3; ++plane) {
      slots[static_cast<std::size_t>(write_slot)].plane[plane] =
          frame_recon[plane];
    }
    // §7.23: the next frame's implicit mapping must see this frame exactly as
    // the decoder stores it.
    if (options.implicit_ref_map &&
        !ref_bank.refresh(static_cast<int>(header.refresh_frame_flags),
                          static_cast<int>(order_hint),
                          static_cast<int>(width), static_cast<int>(height),
                          static_cast<std::uint8_t>(effective_frame_qp),
                          /*key_frame=*/false, 0, 0,
                          header.implicit_output_frame,
                          header.immediate_output_frame, -1, &error)) {
      error = "P-frame " + std::to_string(i) + " reference bank: " + error;
      return false;
    }
    // §7.22/§7.23: save this frame's motion field and reference order hints
    // for later §7.9 estimation.
    if (options.enable_ref_frame_mvs) {
      SlotField& field = slot_fields[static_cast<std::size_t>(write_slot)];
      RefMvsStoreStats store_stats;
      if (!store_slot_field(*tile_encoders[0], static_cast<int>(order_hint),
                            ref_order_hints.data(), ref_count, &field,
                            error, &store_stats)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      result.ref_mvs_warp_cells +=
          static_cast<std::uint64_t>(store_stats.warp_cells);
      result.ref_mvs_wedge_dropped +=
          static_cast<std::uint64_t>(store_stats.wedge_dropped);
      field.order_hint = static_cast<int>(order_hint);
      field.num_saved_refs = ref_count;
      for (int r = 0; r < ref_count; ++r) {
        field.saved_order_hint[r] = ref_order_hints[static_cast<std::size_t>(r)];
      }    }
    recent_slots.erase(
        std::remove(recent_slots.begin(), recent_slots.end(), write_slot),
        recent_slots.end());
    recent_slots.insert(recent_slots.begin(), write_slot);
    // A frame can address at most REFS_PER_FRAME slots, so the recent list
    // never needs more entries (it is also naturally bounded by NumRefFrames).
    if (recent_slots.size() > static_cast<std::size_t>(kRefsPerFrame)) {
      recent_slots.resize(static_cast<std::size_t>(kRefsPerFrame));
    }
  }

  // Default path: every coded picture is output in decode order.  A hidden
  // last picture (SEF wrapper) maps to -1; the SEF unit that follows carries
  // the display entry.
  result.frame_output_index.resize(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const bool hidden =
        options.hide_last_picture && i + 1U == frames.size();
    result.frame_output_index[i] =
        hidden ? -1 : static_cast<std::int32_t>(i);
  }
  return true;
}

}  // namespace av2enc::video
