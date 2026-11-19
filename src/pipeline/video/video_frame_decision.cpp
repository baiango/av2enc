#include "pipeline/video/video_frame_internal.h"

#include "codec/transform/tx_partition/tx_scan.h"
#include "pipeline/video/video_interp_filter.h"
#include "pipeline/video/video_horizon_port.h"
#include "pipeline/video/video_perceptual.h"
#include "pipeline/video/video_pipe_perc.h"
#include "pipeline/video/video_skip_drift.h"

namespace av2enc::video {

namespace {

// AV2ENC_PIPE_PERC coefficient price for one already-quantised plan.  The
// inter luma TX search already filled `rate_bits` from the same static §8.3.2
// context model, so that value is reused (it also carries the partition and
// TX-type symbols the search priced).  Every other plan (intra luma, chroma)
// is priced here from its scan-order levels; a model failure falls back to the
// historical 8 + 2.5*nz + 0.5*eob estimate for consistency.
double pipe_plan_coefficient_bits(
    const TxPlan& plan, int plane, bool is_inter, int qctx,
    const still::qctx::CoeffCdfSlice* cdf_slice = nullptr) {
  if (plan.searched && is_inter && plane == 0 && cdf_slice == nullptr) {
    return plan.rate_bits;
  }
  const auto part_bits = [&](int tx_size, int tx_type,
                             const std::int32_t* levels_scan, int area,
                             int eob) {
    if (eob == 0) {
      std::vector<std::int32_t> zeros(static_cast<std::size_t>(area), 0);
      double bits = 0.0;
      if (area > 0 &&
          txtsearch::scan_coefficient_bits(tx_size, tx_type, plane,
                                           zeros.data(), area, qctx, is_inter,
                                           cdf_slice, &bits)) {
        return bits;
      }
      return 1.0;
    }
    double bits = 0.0;
    if (area > 0 &&
        txtsearch::scan_coefficient_bits(tx_size, tx_type, plane, levels_scan,
                                         area, qctx, is_inter, cdf_slice,
                                         &bits)) {
      return bits;
    }
    int nonzero = 0;
    for (int c = 0; c < eob; ++c) {
      if (levels_scan[c] != 0) ++nonzero;
    }
    return 8.0 + 2.5 * static_cast<double>(nonzero) +
           0.5 * static_cast<double>(eob);
  };
  if (!plan.searched) {
    const int area = plan.adj_w * plan.adj_h;
    if (area <= 0) return plan.eob == 0 ? 1.0 : 0.0;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    if (!::av2enc::video::txpartition::get_scan(
            plan.tx_size, ::av2enc::video::txpartition::TxScanClass::k2D,
            scan)) {
      return plan.eob == 0 ? 1.0 : 0.0;
    }
    std::vector<std::int32_t> scan_levels(static_cast<std::size_t>(area), 0);
    for (int c = 0; c < area; ++c) {
      const std::size_t pos =
          static_cast<std::size_t>(c) < plan.levels.size()
              ? static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])
              : 0U;
      scan_levels[static_cast<std::size_t>(c)] =
          pos < plan.levels.size() ? plan.levels[pos] : 0;
    }
    double bits = part_bits(plan.tx_size, plan.tx_type, scan_levels.data(),
                            area, plan.eob);
    if (is_inter && plane == 0 && plan.eob > 0) bits += 4.0;
    return bits;
  }
  double bits = plan.parts.size() > 1 ? 3.0 : 0.0;
  for (const TxPlan::Part& part : plan.parts) {
    const int area =
        ::av2enc::video::txpartition::tx_scan_area(part.tx_size);
    bits += part_bits(part.tx_size, part.tx_type, part.levels.data(), area,
                      part.eob);
    if (is_inter && plane == 0 && part.eob > 0) bits += 4.0;
  }
  return bits;
}

}  // namespace

bool InterFrameEncoder::decide_sdp(std::string& error) {
  sdpp::Tools tools;
  tools.enable_sdp = true;
  tools.enable_extended_sdp = false;
  tools.frame_is_intra = true;
  tools.single_picture_header_flag = false;

  sdpp::DecisionOptions sopt;
  sopt.min_block_size_px = options_.min_block_size;
  sopt.max_depth = options_.max_partition_depth;
  sopt.allow_rect = true;
  sopt.allow_split = false;
  sopt.allow_3way = false;
  sopt.allow_uneven4 = false;
  sopt.allow_sdp = true;
  sopt.allow_extended_sdp = false;

  const double lambda =
      0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
  const double signal_cost = lambda * 2.0;
  const auto sdp_cost = [this, signal_cost](const sdpp::NodeInfo& info) {
    sdpp::NodeCosts costs;
    if (info.tree_type == sdpp::TreeType::kChromaPart) {
      // Decoder-gated policy: one chroma PARTITION_NONE block per SB.
      costs.leaf = 0.0;
    } else {
      partition::PartitionNodeInfo pinfo;
      pinfo.mi_row = info.mi_row;
      pinfo.mi_col = info.mi_col;
      pinfo.size = info.size;
      pinfo.depth = info.depth;
      pinfo.has_chroma = info.has_chroma;
      pinfo.chroma_offset = info.chroma_offset;
      costs.leaf = static_cast<double>(block_variance_cost_at(pinfo));
    }
    costs.none_signal = 0.0;
    costs.split_signal = sdpp::kInfiniteCost;
    costs.horz_signal = signal_cost;
    costs.vert_signal = signal_cost;
    costs.horz3_signal = sdpp::kInfiniteCost;
    costs.vert3_signal = sdpp::kInfiniteCost;
    costs.horz4a_signal = sdpp::kInfiniteCost;
    costs.horz4b_signal = sdpp::kInfiniteCost;
    costs.vert4a_signal = sdpp::kInfiniteCost;
    costs.vert4b_signal = sdpp::kInfiniteCost;
    costs.region_type[0] = sdpp::kInfiniteCost;
    costs.region_type[1] = sdpp::kInfiniteCost;
    return costs;
  };

  sdp_frame_ = sdpp::decide_frame(geometry_, tools, sdp_cost, sopt);
  if (sdp_frame_.status != sdpp::DecisionStatus::kOk) {
    error = "SDP partition decision failed: " + sdp_frame_.error;
    return false;
  }
  if (!sdpp::validate_frame(geometry_, tools, sdp_frame_, error)) {
    error = "SDP partition tree invalid: " + error;
    return false;
  }

  plan_sdp_pass();
  sb_sinks_.resize(sdp_frame_.superblocks.size());
  for (std::size_t sb = 0; sb < sdp_frame_.superblocks.size(); ++sb) {
    if (!run_one_sb(sb_sinks_[sb], sdp_frame_.superblocks[sb], error)) {
      return false;
    }
  }
  merge_sinks();
  return true;
}

bool InterFrameEncoder::decide(std::string& error) {
  // AV2ENC_TX_DETAIL_SEARCH_TAIL: one picture ordinal per input frame.  The
  // key picture is picture 0, every P picture advances it, the TIP second pass
  // re-decides the same picture, and tiled encodes (tile_w_ < the frame width)
  // leave the state alone.
  const bool whole_picture =
      tile_w_ <= 0 || tile_w_ >= static_cast<int>(source_.width);
  if (!tip_leaf_armed_ && whole_picture) tx_detail::begin_picture();
  if (options_.trace && whole_picture && !tip_leaf_armed_) {
    std::cerr << "  tx-detail picture " << tx_detail::picture_ordinal()
              << " tail=" << (tx_detail::tail_armed() ? 1 : 0)
              << " min=" << tx_detail::config().min_share << "\n";
  }
  if (keyframe_ && options_.enable_sdp) {
    if (!decide_sdp(error)) return false;
    dump_leaf_decisions();
    return true;
  }
  // §5.20.3.1 sub-8 leaves keep the intra keyframe only.  A P frame's inter
  // leaves must own a leaf-sized chroma block, so its search floor stays at
  // 8x8 even when --video-min-block 4 lowers the keyframe floor.
  const int search_min_block =
      keyframe_ ? options_.min_block_size
                : std::max(8, options_.min_block_size);
  partition::DecisionOptions poptions;
  poptions.min_block_size_px = search_min_block;
  poptions.max_depth = options_.max_partition_depth;
  poptions.allow_rect = true;
  // PARTITION_SPLIT only exists at a 128x128 (or 256x256) node.
  poptions.allow_split = sb_size_ > 64;
  if (options_.force_single_block) {
    // One 64x64 leaf per superblock: allow the root split chain but stop any
    // optional split that would create a block below 64x64.
    poptions.max_depth = 8;
    poptions.min_block_size_px = 64;
  }

  // Historical partition cost, kept byte-identical when the opt-in rate-aware
  // tree is off: a source-variance leaf distortion (SSE units) plus a
  // SAD-domain signal cost.  The two do not share a scale, which is the
  // reason the default tree never takes an optional split at high QP.
  const double legacy_lambda =
      0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
  // Rate-aware partition fallback.  It keeps the source variance (a
  // QP-independent proxy for the energy a leaf cannot predict) and charges a
  // rate term in the same SSE units:
  //   leaf  = variance + price * leaf_bits       (mode/MV/skip syntax)
  //   split = price * 2.0                        (partition symbol bits)
  // This is the whole cost for the keyframe and for P-frame leaves the
  // measured probe cannot model (sub-8, right/bottom edges, ME failure), and
  // it is the only cost when `partition_rd` is off (with the historical
  // SAD-domain price).  `price` is one estimated syntax bit in SSE units;
  // 700 with the 10-bit inter leaf term and 2-bit partition symbol gives a
  // ~8,400-SSE split penalty, calibrated from the measured local RD slope of
  // this clip at the ~100 kbps point (4,000-6,300 SSE/bit over QP 168..190;
  // see docs/video-rd-preset.md).  The measured probe supersedes it on the
  // P-frame path.
  constexpr double kPartitionSsePerSyntaxBit = 700.0;
  constexpr double kPartitionSignalBits = 2.0;
  constexpr double kPartitionLeafBitsInter = 10.0;
  constexpr double kPartitionLeafBitsIntra = 6.0;
  double syntax_bit_price =
      options_.partition_rd
          ? (options_.qp_lambda_scale
                 ? partition_model_price_at(options_, current_qindex_)
                 : kPartitionSsePerSyntaxBit)
          : legacy_lambda;
  const double signal_bits = kPartitionSignalBits;
  double leaf_bits = options_.partition_rd
                         ? (keyframe_ ? kPartitionLeafBitsIntra
                                      : kPartitionLeafBitsInter)
                         : 0.0;
  // Measured partition probe.  It replaces the variance leaf proxy with one
  // NEWMV prediction plus one quantised luma residual for every candidate
  // node, and is bounded to the first split level.  It prices leaves only,
  // so it is independent of which solver reaches them: the extended tree
  // gets exactly the cost the rect tree charges a same-shaped child.
  const bool partition_probe =
      options_.partition_rd && !keyframe_ && !options_.force_single_block &&
      !options_.enable_segmentation &&
      width_ == static_cast<int>(source_.width) &&
      height_ == static_cast<int>(source_.height);
  std::unordered_map<std::uint64_t, double> probe_costs;
  if (partition_probe) {
    syntax_bit_price = probe_syntax_price(0, 0);
    // The probe's bits are real (mode/skip syntax plus the residual rate),
    // so the leaf-level syntax estimate is inside the measured cost.
    leaf_bits = 0.0;
    poptions.max_depth =
        std::min(poptions.max_depth,
                 std::clamp(options_.partition_probe_depth, 1, 3));
    // AV2ENC_TX_DETAIL_SEARCH_TAIL: the measured probe clamps the P tree to
    // its first-level bound, which cannot reach a 16x16 coding block.  The
    // armed tail opens the four rect levels (64 -> 32 -> 16) it needs; the
    // per-node band gate below decides which nodes actually descend.
    if (tx_detail::tail_armed()) {
      poptions.max_depth =
          std::min(options_.max_partition_depth, 4);
    }
    probe_costs = build_partition_probe_costs(
        poptions.max_depth,
        std::max(poptions.min_block_size_px, partition::kMinBlockPixels));
  }
  const double signal_cost = syntax_bit_price * signal_bits;
  // AV2ENC_EXACT_RD partition refinement: when non-null, a node's leaf cost
  // comes from the measured (D, R) of the mode the leaf encoder actually chose
  // in the first pass.  Nodes the first pass did not encode (the alternative
  // side of a depth-1 split) keep the NEWMV probe as the early bound.  The
  // pointer is only ever set while a refinement solve runs, so the historical
  // probe-only path is bit-for-bit unchanged when the gate is off.
  const std::unordered_map<std::uint64_t, double>* exact_costs = nullptr;
  // AV2ENC_TX_DETAIL_SEARCH_TAIL: the band-gated descent.  `detail_descent`
  // is only true on an armed tail picture; `detail_max_depth` is the depth the
  // probe was just built for, and `detail_min_share` the swept threshold.
  const bool detail_descent = !keyframe_ && tx_detail::tail_armed();
  const int detail_max_depth = poptions.max_depth;
  const double detail_min_share = tx_detail::config().min_share;
  // One leaf model for both solvers.  It is exactly the historical keyframe
  // / P-frame estimate (variance + price * leaf_bits) or, when the measured
  // probe is active, the probe's measured residual RD on the node's own
  // geometry.  A 64x16 / 16x64 extended leaf is therefore priced by the same
  // code path that prices a NONE/HORZ/VERT/SPLIT child of the same shape,
  // including its leaf coder's transform size and chroma ownership.
  const auto partition_leaf_cost =
      [this, syntax_bit_price, leaf_bits, partition_probe, &probe_costs,
       &exact_costs, detail_descent, detail_max_depth,
       detail_min_share](const partition::PartitionNodeInfo& info) {
        // The leaf coder handles coding blocks up to 64x64.  A larger node
        // must split (the 128/256 root and its rectangular children); the
        // spec never implies NONE at those sizes, so the solver always has a
        // partition.
        const int bw = partition::block_width(info.size);
        const int bh = partition::block_height(info.size);
        if (bw > 64 || bh > 64) return partition::kInfiniteCost;
        // AV2ENC_TX_DETAIL_SEARCH_TAIL: in the armed tail, a node that is
        // strictly larger than 16x16 on at least one axis and carries the
        // 2-4 px source band is not allowed to stay a coding block.  The rect
        // solver then has to split it, so a band-rich region descends to the
        // 16x16 block whose legal §5.20.6.3 SPLIT is four 8x8 transforms.  The
        // gate is skipped when the remaining depth cannot place a 16x16 leaf
        // or when the spec allows no rect partition, so it can never make a
        // node unsolvable.
        if (detail_descent && bw >= 16 && bh >= 16 &&
            (bw > 16 || bh > 16) && info.depth + 1 <= detail_max_depth &&
            (info.allowed.is_allowed(partition::PartitionType::kHorz) ||
             info.allowed.is_allowed(partition::PartitionType::kVert))) {
          const partition::BlockRect rect = partition::block_rect(
              geometry_, info.mi_row, info.mi_col, info.size);
          if (rect.visible_width > 0 && rect.visible_height > 0 &&
              tx_detail::source_band_share(source_, tile_x_ + rect.x,
                                           tile_y_ + rect.y,
                                           rect.visible_width,
                                           rect.visible_height) >=
                  detail_min_share) {
            return partition::kInfiniteCost;
          }
        }
        double leaf = static_cast<double>(block_variance_cost_at(info)) +
                      syntax_bit_price * leaf_bits;
        if (partition_probe) {
          const std::uint64_t key =
              partition_probe_key(info.mi_row, info.mi_col, info.size);
          if (exact_costs != nullptr) {
            const auto exact = exact_costs->find(key);
            if (exact != exact_costs->end()) return exact->second;
          }
          const auto it = probe_costs.find(key);
          if (it != probe_costs.end()) {
            leaf = it->second;
          } else {
            double measured = 0.0;
            if (partition_probe_leaf_cost(
                    info, probe_syntax_price(info.mi_row, info.mi_col),
                    measured)) {
              leaf = measured;
            }
          }
        }
        return leaf;
      };
  const auto cost_fn = [partition_leaf_cost, signal_cost](
                           const partition::PartitionNodeInfo& info) {
    partition::NodeCosts costs;
    costs.leaf = partition_leaf_cost(info);
    costs.none_signal = 0.0;
    costs.split_signal = signal_cost;
    costs.horz_signal = signal_cost;
    costs.vert_signal = signal_cost;
    return costs;
  };

  std::vector<PartitionNode> base_superblocks;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  std::vector<pe::PartitionNode> ext_superblocks;
#endif
  // Re-runnable so the exact partition refinement can solve the same tree
  // with measured leaf costs.  Gate off: called exactly once, as before.
  const auto solve_partition = [&]() -> bool {
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  if (options_.ext_partitions) {
    pe::Tools tools;
    tools.enable_ext_partitions = true;
    tools.enable_uneven_4way_partitions = true;
    pe::DecisionOptions eopt;
    // Mirror the rect solver's bound exactly.  With --video-partition-rd the
    // probe clamps `poptions.max_depth` to the measured first level; the
    // extended tree must search the same depth or it would compare a deeper
    // tree against the shallow rect one and win on tiny leaves alone.
    eopt.min_block_size_px = poptions.min_block_size_px;
    eopt.max_depth = poptions.max_depth;
    eopt.allow_rect = true;
    eopt.allow_split = poptions.allow_split;
    eopt.allow_3way = true;
    eopt.allow_uneven4 = true;
    eopt.tie_break = pe::TieBreakOrder::kRectCompatOrder;
    const auto ext_cost_fn = [this, partition_leaf_cost, signal_cost](
                                 const pe::NodeInfo& info) {
      pe::NodeCosts costs;
      const int bw = partition::block_width(info.size);
      const int bh = partition::block_height(info.size);
      partition::PartitionNodeInfo binfo;
      binfo.mi_row = info.mi_row;
      binfo.mi_col = info.mi_col;
      binfo.size = static_cast<partition::BlockSize>(info.size);
      // Same leaf model the rect solver uses for a NONE/HORZ/VERT/SPLIT
      // child of this geometry: the syntax estimate (variance + leaf rate)
      // or the measured probe RD, selected by the same predicate.
      costs.leaf = (bw <= 64 && bh <= 64)
                       ? partition_leaf_cost(binfo)
                       : pe::kInfiniteCost;
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      // Keep every child >= 8x8 so the video leaf coder keeps owning its
      // chroma block (no shared ChromaMi).  HORZ_4A/B and VERT_4A/B emit an
      // 8x16 / 16x8 minimum slice, so they need the 64-pixel axis.
      costs.horz3_signal =
          (bh >= 32 && bw >= 8 && !(bw == 8 && bh == 32) &&
           !pe::chroma_offset_for_partition(
               geometry_, pe::PartitionType::kHorz3, info.size))
              ? signal_cost
              : pe::kInfiniteCost;
      costs.vert3_signal =
          (bw >= 32 && bh >= 8 && !(bw == 32 && bh == 8) &&
           !pe::chroma_offset_for_partition(
               geometry_, pe::PartitionType::kVert3, info.size))
              ? signal_cost
              : pe::kInfiniteCost;
      const double h4 = (bh >= 64 && bw >= 8) ? signal_cost
                                              : pe::kInfiniteCost;
      const double v4 = (bw >= 64 && bh >= 8) ? signal_cost
                                              : pe::kInfiniteCost;
      costs.horz4a_signal = h4;
      costs.horz4b_signal = h4;
      costs.vert4a_signal = v4;
      costs.vert4b_signal = v4;
      return costs;
    };
    const pe::FrameDecision frame =
        pe::decide_frame(geometry_, ext_cost_fn, tools, eopt);
    if (frame.status != pe::DecisionStatus::kOk) {
      error = "extended partition decision failed: " + frame.error;
      return false;
    }
    if (!pe::validate_frame(geometry_, tools, frame, error)) {
      error = "extended partition tree invalid: " + error;
      return false;
    }
    ext_superblocks = frame.superblocks;
    if (FILE* tree_file = part_tree_file()) {
      for (const auto& sb : ext_superblocks) {
        dump_partition_tree_ext(sb, tree_file);
      }
      std::fflush(tree_file);
    }
  } else
#endif
  {
    const partition::FrameDecision frame =
        partition::decide_frame(geometry_, cost_fn, poptions);
    if (frame.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame.error;
      return false;
    }
    if (!partition::validate_frame(geometry_, frame, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    base_superblocks = frame.superblocks;
    if (FILE* tree_file = part_tree_file()) {
      for (const auto& sb : base_superblocks) {
        dump_partition_tree(sb, tree_file);
      }
      std::fflush(tree_file);
    }
  }
  return true;
  };
  if (!solve_partition()) return false;

  // The partition tree is final, so the pure NEWMV searches can run ahead
  // of the coding-order walk on the host worker pool.  Serial at 1 thread.
  if (!base_superblocks.empty()) {
    precompute_leaf_me(base_superblocks, options_.cpu_threads);
  }

#if defined(AV2ENC_VIDEO_DELTAQ)
  if (!prepare_qp_plan(error)) return false;
#endif
  // One decision pass: frame-level plan, then one sink per superblock, then
  // the offset rebase.  `run_superblocks()` is the only place the schedule
  // lives: raster serial at --cpu-threads 1, the skew-2 wavefront above.
  const auto collect_frame = [&]() -> bool {
    plan_pass();
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    if (options_.ext_partitions) {
      sb_sinks_.resize(ext_superblocks.size());
      if (!run_superblocks(ext_superblocks, error)) return false;
    } else
#endif
    {
      sb_sinks_.resize(base_superblocks.size());
      if (!run_superblocks(base_superblocks, error)) return false;
    }
    merge_sinks();
    return true;
  };

  // AV2ENC_EXACT_RD partition refinement (one pass).  The historical probe
  // prices a node with one NEWMV + quantised residual before the mode search.
  // The first encode pass now records the real (D, R) of the mode each leaf
  // actually chose (encode_leaf -> sink.leaf_costs).  Re-solve the tree with
  // those exact costs where the node was encoded, keeping the probe only for
  // the alternative geometry the first pass did not visit (the early bound).
  // If the tree changes, encode once more with it.  This is deliberately not
  // iterated: a full exact cost per partition node would need every candidate
  // layout encoded, and the one refinement pass captures the dominant error
  // (the probe's NEWMV-only residual vs the leaf's chosen mode) at <=2x cost.
  const auto leaf_signature =
      [](const std::vector<PartitionNode>& superblocks) {
        std::vector<std::uint64_t> leaves;
        const auto walk = [&leaves](auto&& self,
                                    const PartitionNode& node) -> void {
          if (node.partition != partition::PartitionType::kNone) {
            for (const PartitionNode& child : node.children) {
              self(self, child);
            }
            return;
          }
          leaves.push_back(
              (static_cast<std::uint64_t>(node.mi_row) & 0xFFFFU) |
              ((static_cast<std::uint64_t>(node.mi_col) & 0xFFFFU) << 16U) |
              ((static_cast<std::uint64_t>(node.size) & 0xFFU) << 32U) |
              ((static_cast<std::uint64_t>(node.depth) & 0xFFU) << 40U));
        };
        for (const PartitionNode& sb : superblocks) walk(walk, sb);
        return leaves;
      };
  bool exact_partition_done = false;
  const auto collect_frame_refined = [&]() -> bool {
    if (!collect_frame()) return false;
    if (!exact_rd::partition_refine() || !partition_probe ||
        exact_partition_done || options_.ext_partitions ||
        qp_tools_active()) {
      return true;
    }
    exact_partition_done = true;
    std::unordered_map<std::uint64_t, double> measured;
    for (const SbSink& sink : sb_sinks_) {
      const std::size_t count = std::min(sink.blocks.size(),
                                         sink.leaf_costs.size());
      for (std::size_t i = 0; i < count; ++i) {
        const syntax::BlockSyntax& block = sink.blocks[i];
        const exact_rd::LeafCost& leaf = sink.leaf_costs[i];
        if (!leaf.valid) continue;
        if (block.partition !=
            static_cast<std::uint8_t>(partition::PartitionType::kNone)) {
          continue;
        }
        if (block.subtree != 0) continue;
        const partition::BlockSize size =
            static_cast<partition::BlockSize>(block.bsize);
        const double lambda =
            probe_syntax_price(block.mi_row, block.mi_col);
        measured[partition_probe_key(block.mi_row, block.mi_col, size)] =
            leaf.distortion + lambda * leaf.bits;
      }
    }
    if (measured.empty()) return true;
    ++exact_rd::counters().partition_refine_runs;
    exact_rd::counters().partition_overrides += measured.size();
    std::vector<PartitionNode> probe_tree = base_superblocks;
    exact_costs = &measured;
    const bool solved = solve_partition();
    exact_costs = nullptr;
    if (!solved) return false;
    if (leaf_signature(probe_tree) == leaf_signature(base_superblocks)) {
      base_superblocks = std::move(probe_tree);
      return true;
    }
    ++exact_rd::counters().partition_tree_changed;
    if (!base_superblocks.empty()) {
      precompute_leaf_me(base_superblocks, options_.cpu_threads);
    }
    return collect_frame();
  };
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (qp_tools_active()) {
    if (delta_q_enabled_) {
      std::vector<int> effective(
          static_cast<std::size_t>(qp_sb_count()), base_q_index());
      bool converged = false;
      for (int pass = 0; pass < 8; ++pass) {
        sb_qindex_ = effective;
        if (!collect_frame_refined()) return false;
        std::vector<int> actual;
        if (!plan_delta_q_steps(actual, error)) return false;
        if (actual == effective) {
          converged = true;
          break;
        }
        effective = std::move(actual);
      }
      if (!converged) {
        // A quantizer-dependent skip decision can oscillate for a full-SB
        // leaf (the delta is only codable while the block is not a full-SB
        // skip).  Fall back to the neutral ramp: every step is zero, so the
        // stream stays exact whatever the skip decisions are.
        sb_target_q_.assign(static_cast<std::size_t>(qp_sb_count()),
                            base_q_index());
        sb_qindex_.assign(static_cast<std::size_t>(qp_sb_count()),
                          base_q_index());
        if (!collect_frame_refined()) return false;
        std::vector<int> actual;
        if (!plan_delta_q_steps(actual, error)) return false;
        if (actual != sb_qindex_) {
          error = "delta-Q plan: neutral fallback did not produce a zero "
                  "ramp";
          return false;
        }
      }
    } else if (!collect_frame_refined()) {
      return false;
    }
  } else if (!collect_frame_refined()) {
    return false;
  }
#else
  if (!collect_frame_refined()) return false;
#endif
  // --video-skip-collapse: rewrite all-static superblocks to one skip leaf
  // and re-collect.  Default off (returns immediately).
  if (!collapse_static_superblocks(base_superblocks, poptions, collect_frame,
                                   error)) {
    return false;
  }
  dump_leaf_decisions();
  return true;
}

bool InterFrameEncoder::plan_sdp_symbols(const sdpp::Node& node, syntax::BlockSyntax& entry,
                      std::string& error) {
  sdpp::Node info;
  info.mi_row = node.mi_row;
  info.mi_col = node.mi_col;
  info.size = node.size;
  info.tree_type = node.tree_type;
  info.partition = node.partition;
  info.has_chroma = node.has_chroma;
  info.chroma_offset = node.chroma_offset;
  info.chroma_partition_known = node.chroma_partition_known;
  info.implied_luma_partition = node.implied_luma_partition;
  sdpp::SymbolContexts contexts;
  const sdpp::MiSizeGrid& grid =
      node.tree_type == sdpp::TreeType::kChromaPart ? sdp_chroma_sizes_
                                                    : sdp_luma_sizes_;
  contexts = sdpp::node_contexts(geometry_, info, grid);
  const sdpp::SymbolPlan plan = sdpp::plan_partition_symbols(
      node.boundary, node.allowed, node.size, node.tree_type, node.partition,
      contexts, /*bru_active=*/true);
  if (!plan.ok) {
    error = "SDP symbol plan at " + std::to_string(node.mi_row) + "," +
            std::to_string(node.mi_col) + ": " + plan.error;
    return false;
  }
  if (plan.symbols.size() > 4) {
    error = "SDP node codes more than four partition symbols";
    return false;
  }
  entry.sdp_symbol_count = static_cast<std::uint8_t>(plan.symbols.size());
  for (std::size_t i = 0; i < plan.symbols.size(); ++i) {
    const sdpp::Symbol& symbol = plan.symbols[i];
    entry.sdp_symbols[i].kind = static_cast<std::uint8_t>(symbol.kind);
    entry.sdp_symbols[i].plane = static_cast<std::uint8_t>(symbol.plane);
    entry.sdp_symbols[i].context =
        static_cast<std::uint8_t>(symbol.context);
    entry.sdp_symbols[i].value = static_cast<std::uint8_t>(symbol.value);
  }
  return true;
}

bool InterFrameEncoder::collect_sdp_node(SbSink& sink, const sdpp::Node& node,
                      std::string& error) {
  SbSink::State& st = sink.state;
  if (node.depth == 0) {
    // §6.19.2.3 clear_block_decoded_flags(): superblock-local.
    st.decoded[0].reset(mi_cols_, mi_rows_);
    st.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    st.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    reset_refmvbank_for_sb(node.mi_row, node.mi_col);
  }
  const std::size_t index = sink.blocks.size();
  sink.blocks.emplace_back();
  syntax::BlockSyntax& entry = sink.blocks.back();
  entry.levels_offset = kNoLevels;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  entry.tree_type = static_cast<std::uint8_t>(node.tree_type);
  entry.has_chroma = node.codes_chroma ? 1 : 0;
  entry.chroma_offset = node.chroma_offset ? 1 : 0;
  entry.chroma_bsize =
      node.chroma.size == BlockSize::kInvalid
          ? 0
          : static_cast<std::uint8_t>(node.chroma.size);
  entry.chroma_mi_row = static_cast<std::uint16_t>(node.chroma.mi_row);
  entry.chroma_mi_col = static_cast<std::uint16_t>(node.chroma.mi_col);
  entry.chroma_partition_known = node.chroma_partition_known ? 1 : 0;
  entry.implied_luma_partition =
      static_cast<std::uint8_t>(node.implied_luma_partition);
  entry.flags |= syntax::kFlagHasChromaInfo;
  std::uint16_t mask = 0;
  for (int p = 0; p < sdpp::kPartitionCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  if (node.forced) entry.flags |= syntax::kFlagForced;
  if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
  if (node.sdp_split) entry.flags |= syntax::kFlagSdpSplit;

  if (node.sdp_split) {
    if (node.children.size() != 2) {
      error = "SDP split node does not have exactly two tree children";
      return false;
    }
    for (const sdpp::Node& child : node.children) {
      if (!collect_sdp_node(sink, child, error)) return false;
    }
  } else if (node.partition == sdpp::PartitionType::kNone) {
    if (!plan_sdp_symbols(node, entry, error)) return false;
    PartitionNode pnode;
    pnode.mi_row = node.mi_row;
    pnode.mi_col = node.mi_col;
    pnode.size = node.size;
    pnode.partition = PartitionType::kNone;
    pnode.depth = node.depth;
    pnode.has_chroma = node.has_chroma;
    pnode.chroma_offset = node.chroma_offset;
    const std::size_t dump_level_begin = sink.levels.size();
    if (!encode_leaf(
            sink, pnode, index, error, static_cast<int>(node.tree_type),
            node.codes_chroma, node.chroma.mi_row, node.chroma.mi_col,
            node.chroma.size == BlockSize::kInvalid
                ? -1
                : static_cast<int>(node.chroma.size))) {
      return false;
    }
    record_leaf_dump(sink, sink.blocks[index], dump_level_begin);
    update_refmvbank_leaf(sink.blocks[index]);
    if (node.tree_type != sdpp::TreeType::kChromaPart) {
      sdp_luma_sizes_.fill_block(node.mi_row, node.mi_col, node.size);
    }
    if (node.codes_chroma) {
      sdp_chroma_sizes_.fill_block(node.chroma.mi_row, node.chroma.mi_col,
                                   node.chroma.size);
    }
  } else {
    if (!plan_sdp_symbols(node, entry, error)) return false;
    for (const sdpp::Node& child : node.children) {
      if (!collect_sdp_node(sink, child, error)) return false;
    }
  }
  const std::size_t subtree = sink.blocks.size() - index - 1;
  if (subtree > 0xFFFFU) {
    error = "partition subtree too large";
    return false;
  }
  sink.blocks[index].subtree = static_cast<std::uint16_t>(subtree);
  return true;
}

void InterFrameEncoder::record_leaf_dump(SbSink& sink, const syntax::BlockSyntax& entry,
                      std::size_t level_begin) {
  if (!block_dump_enabled()) return;
  LeafDumpRecord record;
  record.mi_row = entry.mi_row;
  record.mi_col = entry.mi_col;
  record.bsize = entry.bsize;
  record.partition = entry.partition;
  record.is_inter = entry.is_inter;
  record.mode = entry.is_inter ? entry.inter_mode : entry.y_mode;
  record.ref = entry.ref_index;
  record.drl_index = entry.drl_index;
  record.mv_x = entry.mv_x;
  record.mv_y = entry.mv_y;
  record.skip = entry.skip_flag;
  for (int plane = 0; plane < 3; ++plane) record.eob[plane] = entry.eob[plane];
  record.levels_digest = hash_levels(sink.levels, level_begin,
                                     sink.levels.size());
  record.palette_index = entry.palette_index;
  if (entry.palette_index != syntax::BlockSyntax::kNoPalette &&
      entry.palette_index < sink.palettes.size()) {
    const palette::PaletteResult& palette =
        sink.palettes[entry.palette_index];
    record.palette_size = palette.has_palette ? palette.size : 0;
    record.palette_direction = palette.direction;
    for (int i = 0; i < record.palette_size && i < 8; ++i) {
      record.palette_colors[static_cast<std::size_t>(i)] =
          palette.colors[static_cast<std::size_t>(i)];
    }
  }
  sink.leaf_dump.push_back(record);
}

void InterFrameEncoder::dump_leaf_decisions() {
  FILE* file = block_dump_file();
  if (file == nullptr) return;
  // Pre-order leaf order is raster order for a quadtree, so this sort is a
  // no-op on the serial walk and makes the text independent of the
  // completion order of a future wavefront.
  std::stable_sort(leaf_dump_.begin(), leaf_dump_.end(),
                   [](const LeafDumpRecord& a, const LeafDumpRecord& b) {
                     if (a.mi_row != b.mi_row) return a.mi_row < b.mi_row;
                     return a.mi_col < b.mi_col;
                   });
  for (const LeafDumpRecord& record : leaf_dump_) {
    std::fprintf(file, "%d %d %d %d %d %d %d %d %d %d %d %u %u %u %016llx",
                 record.mi_row, record.mi_col, record.bsize,
                 record.partition, record.is_inter, record.mode, record.ref,
                 record.drl_index, record.mv_x, record.mv_y, record.skip,
                 record.eob[0], record.eob[1], record.eob[2],
                 static_cast<unsigned long long>(record.levels_digest));
    if (record.palette_index != syntax::BlockSyntax::kNoPalette) {
      std::fprintf(file, " pal=%d,%d,", record.palette_size,
                   record.palette_direction);
      for (int i = 0; i < record.palette_size; ++i) {
        std::fprintf(file, "%s%02x", i == 0 ? "" : ",",
                     static_cast<unsigned>(
                         record.palette_colors[static_cast<std::size_t>(i)]));
      }
    }
    std::fputc('\n', file);
  }
  std::fflush(file);
  leaf_dump_.clear();
}

void InterFrameEncoder::merge_sinks() {
  std::size_t level_base = 0;
  // plan_pass() clears palettes_, but the SDP pass historically did not;
  // starting from the current size reproduces both.
  std::size_t palette_base = palettes_.size();
  for (SbSink& sink : sb_sinks_) {
    for (syntax::BlockSyntax& entry : sink.blocks) {
      if (entry.levels_offset == kNoLevels) {
        entry.levels_offset = 0;
      } else {
        entry.levels_offset += static_cast<std::uint32_t>(level_base);
      }
      for (int part = 0; part < entry.tx_part_count; ++part) {
        entry.tx_part_levels_offset[part] +=
            static_cast<std::uint32_t>(level_base);
      }
      if (entry.palette_index != syntax::BlockSyntax::kNoPalette) {
        entry.palette_index += static_cast<std::uint32_t>(palette_base);
      }
    }
    blocks_.insert(blocks_.end(), sink.blocks.begin(), sink.blocks.end());
    levels_.insert(levels_.end(), sink.levels.begin(), sink.levels.end());
    const std::size_t palette_count = sink.palettes.size();
    for (palette::PaletteResult& palette : sink.palettes) {
      palettes_.push_back(std::move(palette));
    }
    sub_pus_.insert(sub_pus_.end(), sink.sub_pus.begin(),
                    sink.sub_pus.end());
    leaf_dump_.insert(leaf_dump_.end(), sink.leaf_dump.begin(),
                      sink.leaf_dump.end());
    level_base += sink.levels.size();
    palette_base += palette_count;
    // Fold the unit-local counters back in.  Raster order is the historical
    // accumulation order; addition is associative, so a wavefront that
    // completes in any order still lands on the same totals.
    const SbSink::State& st = sink.state;
    inter_blocks_ += st.inter_blocks;
    intra_blocks_ += st.intra_blocks;
    temporal_added_ += st.temporal_added;
    temporal_selected_ += st.temporal_selected;
    same_ref_blocks_ += st.same_ref_blocks;
    refmvbank_added_ += st.refmvbank_added;
    refmvbank_selected_ += st.refmvbank_selected;
    mv_traj_derived_ += st.mv_traj_derived;
    amvd_selected_ += st.amvd_selected;
    flex_precision_selected_ += st.flex_precision_selected;
    derived_sign_selected_ += st.derived_sign_selected;
    tip_blocks_ += st.tip_blocks;
  }
}

bool InterFrameEncoder::run_one_sb(SbSink& sink, const PartitionNode& superblock,
                std::string& error) {
  sink.reset();
  seed_unit_state(sink);
  return collect_node(sink, superblock, error);
}

bool InterFrameEncoder::run_one_sb(SbSink& sink, const pe::PartitionNode& superblock,
                std::string& error) {
  sink.reset();
  seed_unit_state(sink);
  return collect_node_ext(sink, superblock, error);
}

bool InterFrameEncoder::run_one_sb(SbSink& sink, const sdpp::Node& superblock,
                std::string& error) {
  sink.reset();
  seed_unit_state(sink);
  return collect_sdp_node(sink, superblock, error);
}

void InterFrameEncoder::seed_unit_state(SbSink& sink) {
  sink.state.current_qindex = current_qindex_;
  sink.state.current_segment_id = 0;
  sink.state.quant_step = quant_step_;
  for (int plane = 0; plane < 3; ++plane) {
    sink.state.plane_quant[plane] = plane_quant_[plane];
  }
}

bool InterFrameEncoder::wavefront_diagonals(std::size_t count,
                         std::vector<std::vector<std::size_t>>& out) const {
  const int cols = partition::sb_cols(geometry_);
  const int rows = partition::sb_rows(geometry_);
  if (cols <= 0 || rows <= 0 ||
      static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows) !=
          count) {
    return false;
  }
  const int diagonals = 2 * (rows - 1) + cols;
  out.assign(static_cast<std::size_t>(diagonals), {});
  for (int sbr = rows - 1; sbr >= 0; --sbr) {
    for (int sbc = 0; sbc < cols; ++sbc) {
      const std::size_t index =
          static_cast<std::size_t>(sbr) * static_cast<std::size_t>(cols) +
          static_cast<std::size_t>(sbc);
      out[static_cast<std::size_t>(2 * sbr + sbc)].push_back(index);
    }
  }
  return true;
}

bool InterFrameEncoder::wavefront_allowed() const {
  if (keyframe_) return false;
  // AV2ENC_PIPE_PERC / AV2ENC_EFF_PERC / AV2ENC_PIPE_BA: the exact syntax probe
  // advances one shared CDF/grid state in coded order; concurrent superblocks
  // would race it.
  if (pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) return false;
  // R09: the syntax probe advances one shared CDF/grid state in coded order;
  // concurrent superblocks would race it.  Serial decisions keep the probe
  // deterministic and match the writer's serial tile walk.
  if (rate_hybrid::reporting()) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (seg_enabled_ || delta_q_enabled_) return false;
#endif
  if (options_.refmvbank) return false;
  // --video-trace writes one std::cerr line per leaf from inside run_one_sb.
  // Concurrent superblocks interleave those insertions, so lines are lost or
  // split and any gate that counts them becomes flaky: gate_video_edge_mask
  // observed 56 then 53 leaves on the same binary.  Tracing is diagnostic, so
  // keeping the superblock walk serial there restores deterministic output at
  // no cost to the measured paths.
  if (options_.trace) return false;
  return true;
}

double InterFrameEncoder::wavefront_width() const {
  const int cols = partition::sb_cols(geometry_);
  const int rows = partition::sb_rows(geometry_);
  if (cols <= 0 || rows <= 0) return 0.0;
  const double diagonals = static_cast<double>(2 * (rows - 1) + cols);
  return (static_cast<double>(rows) * static_cast<double>(cols)) / diagonals;
}

int InterFrameEncoder::wavefront_override() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_WAVEFRONT");
    if (env == nullptr) return -1;
    return std::string(env) == "0" ? 0 : 1;
  }();
  return value;
}

void InterFrameEncoder::plan_pass() {
  blocks_.clear();
  levels_.clear();
  palettes_.clear();
  leaf_dump_.clear();
  if (palette_enabled_) palette_state_.reset(mi_cols_, mi_rows_);
  // The merge stack reads the above-right / below-left MI grid, so stale
  // entries from the previous pass would make the encoder's stack differ
  // from the writer's.
  mv_grid_.assign(mv_grid_.size(), inter::mc::MiInfo{});
  intra_joint_.assign(intra_joint_.size(), 0);
  // AV2ENC_PIPE_BA: the Butteraugli-shaped frame mask (local-contrast masking
  // blended with the tool weight).  Built once per picture, key included, so
  // every surface that prices distortion reads one map.  With the tool weight
  // at 0 the map is neutral (256 everywhere) and the pipeline still runs its
  // exact-rate decisions: that is the gate's base arm.
  if (pipe_ba::enabled()) {
    pipe_ba::init();
    pipe_ba::build_frame_mask(source_.y.data(), static_cast<int>(source_.width),
                              static_cast<int>(source_.width),
                              static_cast<int>(source_.height), &pipe_ba_mask_);
    if (options_.trace) {
      std::cerr << "  pipe-ba mask " << pipe_ba_mask_.width << "x"
                << pipe_ba_mask_.height << " weight=" << pipe_ba::weight()
                << " low_band=" << pipe_ba::config().low_band_gain << "\n";
    }
  }
  // AV2ENC_PIPE_PERC / AV2ENC_EFF_PERC: source-driven detail weights for this
  // frame and a fresh exact-syntax probe seeded from the same CDF image the
  // writer starts from.  Both are skipped when the gates are off, so the
  // historical pass does no work.
  if ((pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) &&
      !keyframe_) {
    if (eff_perc::enabled()) {
      eff_perc::init();
      eff_perc::build_frame_weights(
          source_.y.data(), static_cast<int>(source_.width),
          static_cast<int>(source_.width), static_cast<int>(source_.height),
          &eff_weights_);
    }
    if (pipe_perc::enabled()) {
      pipe_perc::init();
      pipe_perc::build_frame_weights(
          source_.y.data(), static_cast<int>(source_.width),
          static_cast<int>(source_.width), static_cast<int>(source_.height),
          &pipe_weights_);
    }
    pipe_probe_ready_ = false;
    rate_hybrid::ProbeConfig config;
    config.base_q_idx = static_cast<int>(options_.base_q_idx);
    config.mi_cols = mi_cols_;
    config.mi_rows = mi_rows_;
    config.num_refs = num_total_refs_;
    config.max_drl_bits_minus_1 = 4;
    for (int i = 0; i < 7; ++i) {
      config.frame_distance[i] = frame_distance_[static_cast<std::size_t>(i)];
    }
    config.reference_select = options_.compound && num_total_refs_ > 1;
    config.allow_bawp =
        (options_.enable_bawp || options_.enable_morph) && !keyframe_;
    config.enable_interintra = options_.enable_interintra && !keyframe_;
    config.enable_adaptive_mvd = options_.enable_adaptive_mvd;
    config.enable_flex_mvres = options_.enable_flex_mvres;
    config.enable_mvd_sign_derive = options_.enable_mvd_sign_derive;
    config.mv_precision =
        options_.seq_force_integer_mv ? 3 : options_.mv_precision;
    config.disable_cdf_update = !options_.adaptive_cdf;
    pipe_probe_.reset(config, cdf_seed_);
    pipe_probe_ready_ = true;
  }
  // R09: every decision pass starts from the same frame/tile CDF seed and an
  // empty neighbour grid, exactly like the writer does.
  rate_probe_ready_ = false;
  if (rate_hybrid::reporting() && !keyframe_) {
    rate_hybrid::ProbeConfig config;
    config.base_q_idx = static_cast<int>(options_.base_q_idx);
    config.mi_cols = mi_cols_;
    config.mi_rows = mi_rows_;
    config.num_refs = num_total_refs_;
    config.max_drl_bits_minus_1 = 4;
    for (int i = 0; i < 7; ++i) {
      config.frame_distance[i] = frame_distance_[static_cast<std::size_t>(i)];
    }
    config.reference_select = options_.compound && num_total_refs_ > 1;
    config.allow_bawp =
        (options_.enable_bawp || options_.enable_morph) && !keyframe_;
    config.enable_interintra = options_.enable_interintra && !keyframe_;
    config.enable_adaptive_mvd = options_.enable_adaptive_mvd;
    config.enable_flex_mvres = options_.enable_flex_mvres;
    config.enable_mvd_sign_derive = options_.enable_mvd_sign_derive;
    config.mv_precision =
        options_.seq_force_integer_mv ? 3 : options_.mv_precision;
    config.disable_cdf_update = !options_.adaptive_cdf;
    rate_probe_.reset(config, cdf_seed_);
    rate_probe_ready_ = true;
  }
  for (int plane = 0; plane < 3; ++plane) {
    std::fill(recon_[plane].samples.begin(), recon_[plane].samples.end(),
              static_cast<std::uint8_t>(0));
  }
  inter_blocks_ = 0;
  intra_blocks_ = 0;
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (seg_enabled_) {
    deltaq::clear_above_context(enc_seg_grid_);
    deltaq::clear_left_context(enc_seg_grid_);
  }
#endif
}

void InterFrameEncoder::plan_sdp_pass() {
  blocks_.clear();
  levels_.clear();
  leaf_dump_.clear();
  sdp_luma_sizes_.reset(geometry_);
  sdp_chroma_sizes_.reset(geometry_);
  mv_grid_.assign(mv_grid_.size(), inter::mc::MiInfo{});
  intra_joint_.assign(intra_joint_.size(), 0);
  for (int plane = 0; plane < 3; ++plane) {
    std::fill(recon_[plane].samples.begin(), recon_[plane].samples.end(),
              static_cast<std::uint8_t>(0));
  }
  inter_blocks_ = 0;
  intra_blocks_ = 0;
}

bool InterFrameEncoder::collect_node(SbSink& sink, const PartitionNode& node,
                  std::string& error,
                  const ChromaInfo& inherited_chroma) {
  SbSink::State& st = sink.state;
  const ChromaInfo chroma = chroma_step(node, inherited_chroma);
  if (node.depth == 0) {
    // §6.19.2.3 clear_block_decoded_flags(): the map is superblock-local.
    st.decoded[0].reset(mi_cols_, mi_rows_);
    st.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    st.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    reset_refmvbank_for_sb(node.mi_row, node.mi_col);
#if defined(AV2ENC_VIDEO_DELTAQ)
    // §7.14.2 get_qindex() and §5.20.5.7 SegmentIds are superblock state for
    // this encoder: the syntax writer signals the same id and the delta at
    // the first leaf of the superblock.
    if (qp_tools_active()) {
      const int sb = (node.mi_row / qp_sb_mi()) * qp_sb_cols() +
                     node.mi_col / qp_sb_mi();
      if (sb < 0 || static_cast<std::size_t>(sb) >= sb_qindex_.size()) {
        error = "delta-Q: superblock index outside the plan";
        return false;
      }
      st.current_qindex = sb_qindex_[static_cast<std::size_t>(sb)];
      st.current_segment_id =
          seg_enabled_
              ? static_cast<int>(sb_segment_[static_cast<std::size_t>(sb)])
              : 0;
      st.quant_step =
          rc::kQlookupQ0To255[std::clamp(st.current_qindex, 0, 255)];
      update_plane_quant(sink.state);
      if (seg_enabled_ && node.mi_col == 0) {
        deltaq::clear_left_context(enc_seg_grid_);
      }
      if (trace_ && seg_enabled_) {
        std::cerr << "segq sb=" << sb
                  << " segment=" << st.current_segment_id
                  << " qindex=" << st.current_qindex << "\n";
      }
    }
#endif
  }
  const std::size_t index = sink.blocks.size();
  sink.blocks.emplace_back();
  syntax::BlockSyntax& entry = sink.blocks.back();
  entry.levels_offset = kNoLevels;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  std::uint8_t mask = 0;
  for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint8_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  if (node.forced) entry.flags |= syntax::kFlagForced;
  if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
  // Carry the decoded ownership arguments and the inherited ChromaMi block
  // so the writer codes chroma exactly once for a sub-8 group.  Every leaf
  // at 8x8 or above has ChromaMi == itself, which reproduces the historical
  // leaf-derived path.
  entry.flags |= syntax::kFlagHasChromaInfo;
  entry.has_chroma = node.has_chroma ? 1 : 0;
  entry.chroma_offset = node.chroma_offset ? 1 : 0;
  entry.chroma_bsize =
      chroma.valid ? static_cast<std::uint8_t>(chroma.size) : 0;
  entry.chroma_mi_row = static_cast<std::uint16_t>(chroma.mi_row);
  entry.chroma_mi_col = static_cast<std::uint16_t>(chroma.mi_col);

  if (node.partition == PartitionType::kNone) {
    const std::size_t dump_level_begin = sink.levels.size();
    if (!encode_leaf(sink, node, index, error, /*tree_type=*/0,
                     node.has_chroma, chroma.mi_row, chroma.mi_col,
                     chroma.valid ? static_cast<int>(chroma.size) : -1)) {
      return false;
    }
    record_leaf_dump(sink, sink.blocks[index], dump_level_begin);
    update_refmvbank_leaf(sink.blocks[index]);
  } else {
    for (const PartitionNode& child : node.children) {
      if (!collect_node(sink, child, error, chroma)) return false;
    }
  }
  const std::size_t subtree = sink.blocks.size() - index - 1;
  if (subtree > 0xFFFFU) {
    error = "partition subtree too large";
    return false;
  }
  sink.blocks[index].subtree = static_cast<std::uint16_t>(subtree);
  return true;
}

bool InterFrameEncoder::collect_node_ext(SbSink& sink, const pe::PartitionNode& node,
                      std::string& error) {
  SbSink::State& st = sink.state;
  if (node.depth == 0) {
    st.decoded[0].reset(mi_cols_, mi_rows_);
    st.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    st.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    reset_refmvbank_for_sb(node.mi_row, node.mi_col);
  }
  const std::size_t index = sink.blocks.size();
  sink.blocks.emplace_back();
  syntax::BlockSyntax& entry = sink.blocks.back();
  entry.levels_offset = kNoLevels;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  std::uint16_t mask = 0;
  for (int p = 0; p < pe::kPartitionCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  if (node.forced) entry.flags |= syntax::kFlagForced;
  if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
  entry.flags |= syntax::kFlagHasChromaInfo;
  entry.has_chroma = node.has_chroma ? 1 : 0;
  entry.chroma_offset = node.chroma_offset ? 1 : 0;
  entry.chroma_bsize =
      node.chroma.size == BlockSize::kInvalid
          ? 0
          : static_cast<std::uint8_t>(node.chroma.size);
  entry.chroma_mi_row = static_cast<std::uint16_t>(node.chroma.mi_row);
  entry.chroma_mi_col = static_cast<std::uint16_t>(node.chroma.mi_col);

  if (node.partition == pe::PartitionType::kNone) {
    // encode_leaf() only reads the origin/size, so a stack view is enough.
    PartitionNode leaf;
    leaf.mi_row = node.mi_row;
    leaf.mi_col = node.mi_col;
    leaf.size = static_cast<partition::BlockSize>(node.size);
    const std::size_t dump_level_begin = sink.levels.size();
    if (!encode_leaf(
            sink, leaf, index, error, /*tree_type=*/0, node.has_chroma,
            node.chroma.mi_row, node.chroma.mi_col,
            node.chroma.size == BlockSize::kInvalid
                ? -1
                : static_cast<int>(node.chroma.size))) {
      return false;
    }
    record_leaf_dump(sink, sink.blocks[index], dump_level_begin);
    update_refmvbank_leaf(sink.blocks[index]);
  } else {
    for (const pe::PartitionNode& child : node.children) {
      if (!collect_node_ext(sink, child, error)) return false;
    }
  }
  const std::size_t subtree = sink.blocks.size() - index - 1;
  if (subtree > 0xFFFFU) {
    error = "partition subtree too large";
    return false;
  }
  sink.blocks[index].subtree = static_cast<std::uint16_t>(subtree);
  return true;
}

bool InterFrameEncoder::encode_leaf(SbSink& sink, const PartitionNode& node, std::size_t index,
                 std::string& error, int tree_type,
                 bool leaf_has_chroma, int chroma_mi_row,
                 int chroma_mi_col, int chroma_bsize) {
  SbSink::State& st = sink.state;
  const int r = node.mi_row;
  const int c = node.mi_col;
  if (tip_leaf_armed_ && tree_type == 0 && r == tip_leaf_r_ && c == tip_leaf_c_) {
    return encode_tip_leaf(sink, node, index, error);
  }
  const int w = partition::block_width(node.size);
  const int h = partition::block_height(node.size);
  const int w4 = w >> 2;
  const int h4 = h >> 2;
  const int x0 = c * 4;
  const int y0 = r * 4;
  // §5.20.3.1 chroma ownership: an SDP leaf carries the decoded HasChroma
  // and the ChromaMi block it owns.  A CHROMA_PART leaf codes no luma; a
  // LUMA_PART leaf codes no chroma.
  const bool chroma_part = tree_type == 2;
  const bool luma_part = tree_type == 1;
  const bool owns_chroma = leaf_has_chroma && !luma_part;
  BlockSize chroma_size = node.size;
  if (chroma_bsize >= 0) {
    chroma_size = static_cast<BlockSize>(chroma_bsize);
  }
  // §5.20.3.1: a sub-8 leaf group shares one inherited ChromaMi block.  Its
  // chroma plane lives at the ChromaMi origin, not at this leaf's origin,
  // and is larger than leaf_w/2 x leaf_h/2.  Inter prediction (which derives
  // chroma from the luma leaf) and the screen-content tools are not wired
  // for that geometry, so such leaves stay intra.
  const bool shared_chroma =
      owns_chroma &&
      (chroma_mi_row != r || chroma_mi_col != c ||
       chroma_size != node.size);
  int cw = 0;
  int ch = 0;
  if (chroma_bsize >= 0) {
    const BlockSize chroma_residual =
        partition::chroma_plane_residual_size(geometry_, chroma_size);
    cw = partition::block_width(chroma_residual);
    ch = partition::block_height(chroma_residual);
  } else {
    cw = w / 2;
    ch = h / 2;
  }
  const int cx0 = chroma_bsize >= 0 ? (chroma_mi_col * 4) / 2 : x0 / 2;
  const int cy0 = chroma_bsize >= 0 ? (chroma_mi_row * 4) / 2 : y0 / 2;
  // Absolute luma/chroma coordinates for source reads and MC.  The coded
  // geometry and reconstruction stay tile-local.
  const int ax0 = x0 + tile_x_;
  const int ay0 = y0 + tile_y_;
  const int acx0 = cx0 + tile_x_ / 2;
  const int acy0 = cy0 + tile_y_ / 2;
  // --video-edge-mask: if this leaf's coded extent crosses the visible
  // bottom edge, the tail rows are decoder-invisible.  Both TX entry points
  // use the prediction as the source view there (zero residual) so no
  // levels are spent on samples the decoder discards after filtering.
  const int luma_mask_row = visible_rows(0, y0, h);
  const int chroma_mask_row = visible_rows(1, cy0, ch);
  // AV2ENC_PERCEPTUAL_RD: opt-in detail-aware TX distortion.  The weight map
  // is built from the source luma of this leaf's extent; strength 0 leaves it
  // invalid and both TX lambdas below pass through unchanged.  The proxy is
  // confined to the TX search: weighting the leaf-level candidate/skip SSE on
  // top of it measured worse, so only the TX cost is re-weighted (by dividing
  // lambda, which is the same as weighting the TX search's SSE).  A
  // CHROMA_PART leaf codes no luma, so it has no luma activity to score.  See
  // `video_perceptual.h` for the model and why the frame mean normalises it.
  const double perceptual_strength = perceptual::rd_strength();
  perceptual::LeafWeights perceptual_weights;
  if (perceptual_strength > 0.0 && !chroma_part &&
      source_.y.size() >= static_cast<std::size_t>(source_.width) *
                               static_cast<std::size_t>(source_.height)) {
    perceptual_weights = perceptual::make_leaf_weights(
        source_.y.data(), static_cast<int>(source_.width),
        static_cast<int>(source_.width), static_cast<int>(source_.height),
        ax0, ay0, w, h, perceptual_strength);
  }
  // --video-edge-mask: the decision metrics score only the rows the decoder
  // can see.  Removing the hidden-row bits is not enough on its own: the
  // skip/residual and mode comparisons would still charge a coded residual
  // for hidden-row SSD it can never fix (the source there is the clamped
  // last visible row, the prediction comes from the edge-padded reference),
  // which biases every bottom leaf toward skip.
  const auto sse_vis = [&](int plane, int sx, int sy, int sw, int sh,
                           const std::vector<std::uint8_t>& samples) {
    return sse_against_source(source_, plane, sx, sy, sw, sh, samples,
                              plane == 0 ? luma_mask_row : chroma_mask_row);
  };
  // AV2ENC_PIPE_PERC / AV2ENC_EFF_PERC: one objective for the leaf and the TX
  // search.
  //   D  = local perceptual luma SSE + weighted chroma SSE
  //   R  = exact inter syntax probe + coefficient model
  //   J  = D + lambda * R
  // `pipe_leaf_weights` is the per-pixel map handed to the TX candidate
  // search, so the TX size/type/partition choice minimizes the same J.
  // AV2ENC_PIPE_BA: the Butteraugli-shaped model replaces the CSF on the leaf
  // and TX distortion.  `ba_on` is the tool (weight > 0, mask built); the
  // pipeline itself (`pipe_on`) stays on at weight 0 so the gate's base arm
  // keeps the exact-rate decisions with plain SSE.
  const bool ba_on =
      pipe_ba::tool_on() && !keyframe_ && pipe_probe_ready_ && pipe_ba_mask_.valid();
  // PIPE_BA owns the distortion whenever it is on; the eff-perc branch only
  // runs for the EFF_PERC gate itself.
  const bool eff_on =
      !pipe_ba::enabled() && eff_perc::enabled() && !keyframe_ && pipe_probe_ready_;
  const bool pipe_on =
      (pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) &&
      !keyframe_ && pipe_probe_ready_;
  const double pipe_lambda =
      pipe_on
          ? leaf_rd_lambda(st.current_qindex, options_) *
                (pipe_ba::enabled() ? pipe_ba::config().lambda
                                    : (eff_on ? eff_perc::lambda_scale()
                                              : pipe_perc::lambda_scale()))
          : 0.0;
  const int pipe_qctx = ::av2enc::still::qctx::slice_for_base_q_idx(
      static_cast<int>(options_.base_q_idx));
  std::vector<std::uint16_t> pipe_leaf_weights;
  std::vector<std::uint16_t> pipe_chroma_weights;
  if (pipe_on && !chroma_part) {
    if (ba_on) {
      pipe_ba::fill_leaf_weights(pipe_ba_mask_, ax0, ay0, w, h,
                                 &pipe_leaf_weights);
      ++pipe_ba::counters().luma_leaves;
    } else if (eff_on) {
      eff_perc::fill_leaf_weights(eff_weights_, ax0, ay0, w, h,
                                  &pipe_leaf_weights);
    } else {
      pipe_perc::fill_leaf_weights(pipe_weights_, ax0, ay0, w, h,
                                   &pipe_leaf_weights);
    }
  }
  if (pipe_on && ba_on && !luma_part) {
    pipe_ba::fill_chroma_leaf_weights(pipe_ba_mask_, acx0, acy0, cw, ch,
                                      &pipe_chroma_weights);
    ++pipe_ba::counters().chroma_leaves;
  }
  const auto sse_pipe = [&](int plane, int sx, int sy, int sw, int sh,
                            const std::vector<std::uint8_t>& samples) {
    if (ba_on) {
      if (plane == 0) {
        return pipe_ba::plane_distortion(
            pipe_leaf_weights.empty() ? nullptr : pipe_leaf_weights.data(), sw,
            sh, source_.y.data(), static_cast<int>(source_.width), sx, sy,
            samples.data(), sw, 0, 0, luma_mask_row,
            pipe_ba::config().low_band_gain);
      }
      const std::span<const std::uint8_t> src_plane =
          plane == 1 ? source_.u : source_.v;
      if (src_plane.empty()) {
        return static_cast<double>(sse_vis(plane, sx, sy, sw, sh, samples));
      }
      return pipe_ba::plane_distortion(
                 pipe_chroma_weights.empty() ? nullptr
                                             : pipe_chroma_weights.data(),
                 sw, sh, src_plane.data(), static_cast<int>(source_.width) / 2,
                 sx, sy, samples.data(), sw, 0, 0, chroma_mask_row,
                 pipe_ba::config().chroma_low_band_gain) *
             pipe_ba::config().chroma_scale;
    }
    if (!pipe_on || pipe_leaf_weights.empty()) {
      return static_cast<double>(sse_vis(plane, sx, sy, sw, sh, samples));
    }
    if (plane != 0) {
      if (eff_on && eff_perc::csf_gain() > 0.0) {
        const std::span<const std::uint8_t> src_plane =
            plane == 1 ? source_.u : source_.v;
        if (!src_plane.empty()) {
          const double csf = eff_perc::weighted_sse(
              nullptr, sw, sh, src_plane.data(),
              static_cast<int>(source_.width) / 2, sx, sy, samples.data(), sw,
              0, 0, chroma_mask_row, eff_perc::csf_gain(),
              eff_perc::csf2_gain());
          return csf * eff_perc::chroma_weight();
        }
      }
      const double plain =
          static_cast<double>(sse_vis(plane, sx, sy, sw, sh, samples));
      return eff_on ? plain * eff_perc::chroma_weight() : plain;
    }
    if (eff_on) {
      return eff_perc::weighted_sse(
          pipe_leaf_weights.data(), sw, sh, source_.y.data(),
          static_cast<int>(source_.width), sx, sy, samples.data(), sw, 0, 0,
          luma_mask_row, eff_perc::csf_gain(), eff_perc::csf2_gain());
    }
    return pipe_perc::weighted_sse(
        pipe_leaf_weights.data(), sw, sh, source_.y.data(),
        static_cast<int>(source_.width), sx, sy, samples.data(), sw, 0, 0,
        luma_mask_row);
  };
  // AV2ENC_EFF_PERC: coefficient term selection.  Rate 1 prices the emitted
  // plan with the R19 interval width against the live frame-start CDF image;
  // rate 0 keeps the static §8.3 context model pipe-perc shipped with.
  // AV2ENC_PIPE_BA inherits the same front end at every tool weight, so the
  // base arm and the knob arms differ only in D.  The key picture keeps the
  // historical estimate in both gates (its probe is not seeded).
  const bool pipe_rate_on =
      eff_perc::enabled() && !keyframe_ && pipe_probe_ready_;
  const bool eff_interval =
      pipe_rate_on && (eff_perc::rate_mode() & 1) != 0;
  const bool eff_live_cdf =
      pipe_rate_on && (eff_perc::rate_mode() & 2) != 0;
  const still::qctx::CoeffCdfSlice* eff_slice =
      eff_live_cdf ? eff_cdf_slice() : nullptr;
  const auto coeff_bits = [&](const TxPlan& plan, int plane, bool is_inter) {
    if (eff_interval) {
      ++eff_perc::counters().rate_interval;
      return plan_rate_bits(plan, plane, is_inter,
                            static_cast<int>(node.size));
    }
    return pipe_plan_coefficient_bits(plan, plane, is_inter, pipe_qctx,
                                      eff_slice);
  };
  if (ibc_enabled_) {
    // §5.20.4.1 decode_block(): update_ibc_buffers() runs at the start of
    // every block, before its mode info builds the BV stack.
    ibcs::begin_ibc_block(ibc_buffer_, r, c);
  }

  syntax::BlockSyntax& entry = sink.blocks[index];

  // The CUDA pipeline evaluates inter candidates only when the luma block
  // fits inside the reconstructed reference extent.  A bottom/right edge node
  // can straddle the MI-aligned plane; both paths must make the same choice,
  // so the whole-frame CPU reference applies the same test.  Tiled coding has
  // its own clipped geometry and keeps the historical path.
  //
  // With --video-edge-inter the bottom edge is reconstructed into the
  // SB-aligned padded plane and the reference carries the same edge-extended
  // rows the decoder pads, so a leaf whose visible part is inside the picture
  // keeps the inter search.
  const bool inter_fits =
      width_ != static_cast<int>(source_.width) ||
      height_ != static_cast<int>(source_.height) ||
      (x0 + w <= mi_cols_ * 4 &&
       (y0 + h <= mi_rows_ * 4 ||
        (edge_pad_rows_ > 0 && y0 < mi_rows_ * 4)));

  InterCandidate best;
  InterCandidate same_ref_best;
  // Lowest-SSE signallable non-GLOBALMV candidate.  §5.20.7.6 makes BAWP
  // unavailable on a GLOBALMV leaf, so when the plain search picks GLOBALMV
  // (every zero-MV block of a static clip does) this carrier is the same MV
  // expressed as NEARMV/NEWMV and can carry use_bawp.
  InterCandidate bawp_carrier;
  InterCandidate global_cand;
  bool global_cand_valid = false;
#if defined(AV2ENC_VIDEO_DELTAQ)
  // SEG_LVL_SKIP / SEG_LVL_GLOBALMV force YMode = GLOBALMV and use
  // SkipSegFrame for RefFrame[0] on inter blocks.  The encoded block must
  // therefore keep the first reference and the §7.12.2.2 global MV.
  const bool seg_skip_leaf =
      !keyframe_ && seg_enabled_ && skip_seg_id_ >= 0 &&
      st.current_segment_id == skip_seg_id_;
  const bool seg_globalmv_leaf =
      !keyframe_ && seg_enabled_ && globalmv_seg_id_ >= 0 &&
      st.current_segment_id == globalmv_seg_id_;
  const bool seg_forced_inter = seg_skip_leaf || seg_globalmv_leaf;
#else
  const bool seg_skip_leaf = false;
  const bool seg_globalmv_leaf = false;
  const bool seg_forced_inter = false;
#endif
  // --video-leaf-rd prices the inter candidates with the same cost the leaf
  // decision uses, so a NEWMV that wins by a handful of SSE but costs ten
  // MVD bits no longer shadows a NEARMV/GLOBALMV that codes zero MVD.
  const double inter_lambda =
      options_.enable_leaf_rd
          ? leaf_rd_lambda(st.current_qindex, options_)
          : 0.0;
  const bool exact_on = exact_rd::enabled();
  // R09 (AV2ENC_RATE_HYBRID_S): when the model drives the decision, the
  // candidate ranking and the skip/residual bar use the exact CDF probe and
  // the Laplacian coefficient rate.  The measure switch keeps the historical
  // decisions and only records the model's prediction error.
  const bool rate_model_on = rate_hybrid::enabled() &&
                            rate_hybrid::syntax_enabled() &&
                            rate_probe_ready_;
  const bool rate_reporting = rate_hybrid::reporting() && rate_probe_ready_;
  // Minimal BlockSyntax view of one candidate, for the syntax probe.  The
  // probe only reads the mode/ref/DRL/MVD fields and the block geometry.
  const auto probe_node_for = [&](const InterCandidate& cand) {
    syntax::BlockSyntax probe{};
    probe.mi_row = static_cast<std::uint16_t>(r);
    probe.mi_col = static_cast<std::uint16_t>(c);
    probe.bsize = static_cast<std::uint8_t>(node.size);
    probe.is_inter = 1;
    probe.inter_mode = static_cast<std::uint8_t>(cand.mode);
    probe.ref_index = static_cast<std::uint8_t>(cand.ref_index);
    probe.drl_index = static_cast<std::uint8_t>(cand.drl_index);
    probe.mvd_y = cand.mvd.y;
    probe.mvd_x = cand.mvd.x;
    probe.compound = cand.compound ? 1 : 0;
    probe.compound_mode = static_cast<std::uint8_t>(cand.compound_mode);
    probe.ref_index1 = static_cast<std::uint8_t>(cand.ref_index1);
    probe.use_amvd = cand.use_amvd ? 1 : 0;
    probe.block_mv_precision = static_cast<std::int8_t>(cand.block_mv_precision);
    probe.mvd_sign_derived = cand.mvd_sign_derived ? 1 : 0;
    return probe;
  };
  // Exact mode/ref/DRL/MVD price, or the historical constant formula when the
  // probe cannot model this candidate.
  const auto candidate_syntax_bits = [&](const InterCandidate& cand) {
    if (pipe_on) {
      rate_hybrid::Terms terms;
      if (pipe_probe_.price_mode(probe_node_for(cand), &terms)) {
        return terms.mode_info();
      }
      ++(eff_on ? eff_perc::counters().fallbacks
                : pipe_perc::counters().fallbacks);
    }
    if (rate_model_on) {
      rate_hybrid::Terms terms;
      if (rate_probe_.price_mode(probe_node_for(cand), &terms)) {
        return terms.mode_info();
      }
    }
    return inter_syntax_bits(cand.mode, cand.drl_index, cand.mvd.y,
                             cand.mvd.x, cand.use_amvd);
  };
  const auto inter_cost = [&](const InterCandidate& cand) {
    return static_cast<double>(cand.sse) +
           (pipe_on ? pipe_lambda : inter_lambda) *
               candidate_syntax_bits(cand);
  };
  // AV2ENC_EXACT_RD: the candidates the cheap search considered, with the
  // exact cost key the proxy used.  `order` makes the ranking total: equal
  // costs keep the historical search order.  Gate off leaves this empty.
  struct RankedCandidate {
    InterCandidate cand;
    double cost = 0.0;
    std::size_t order = 0;
  };
  std::vector<RankedCandidate> ranked_candidates;
  // The historical ranking, kept in parallel under the R09 gate so the report
  // can count how many winners the model changed.
  InterCandidate best_hist;
  const auto consider = [&](InterCandidate cand) {
    if (!cand.valid) return;
    if (exact_on || pipe_on) {
      ranked_candidates.push_back(
          RankedCandidate{cand, inter_cost(cand), ranked_candidates.size()});
    }
    if (rate_reporting) {
      if (!best_hist.valid) {
        best_hist = cand;
      } else if (inter_lambda > 0.0
                     ? static_cast<double>(cand.sse) +
                               inter_lambda *
                                   inter_syntax_bits(cand.mode, cand.drl_index,
                                                     cand.mvd.y, cand.mvd.x,
                                                     cand.use_amvd) <
                           static_cast<double>(best_hist.sse) +
                               inter_lambda *
                                   inter_syntax_bits(
                                       best_hist.mode, best_hist.drl_index,
                                       best_hist.mvd.y, best_hist.mvd.x,
                                       best_hist.use_amvd)
                     : cand.sse < best_hist.sse) {
        best_hist = cand;
      }
    }
    if (!best.valid) {
      best = cand;
    } else if (inter_lambda > 0.0 ? inter_cost(cand) < inter_cost(best)
                                   : cand.sse < best.sse) {
      best = cand;
    }
    // §5.20.7.6 BAWP is coded only for a non-GLOBALMV single-prediction
    // leaf.  Track the best carrier so a GLOBALMV winner can still use the
    // fitted predictor when the candidate set contains the same MV.
    if (options_.bawp_carrier && !cand.compound &&
        cand.mode != syntax::kInterGlobalMv &&
        (!bawp_carrier.valid || cand.sse < bawp_carrier.sse)) {
      bawp_carrier = cand;
    }
  };

  // §5.20.3.1 sub-8 luma leaves and shared-ChromaMi owners keep the intra
  // path: inter MC derives chroma from the leaf rectangle, which is not the
  // owned chroma block.  Leaves at 8x8 and above keep the historical inter
  // search, so every inter leaf still owns a leaf-sized chroma block.
  const bool inter_allowed =
      !shared_chroma && w >= 8 && h >= 8;
  if (!keyframe_ && inter_fits && inter_allowed) {
    for (int ref_index = 0; ref_index < num_total_refs_; ++ref_index) {
      merge::StackParams stack_params;
      merge::MvStack stack;
      const bool have_stack =
          load_stack(r, c, w4, h4, ref_index, sink.state, stack_params,
                     stack);

      // GLOBALMV (§5.20.7.6 / §7.13.3.14).  With an accepted non-identity
      // model the decoder either warps the block (§7.13.3.19/.20, blocks at
      // least 8x8) or applies the §7.12.2.2 projected centre MV as ordinary
      // translational MC.  Without a model this is the historical zero MV.
      {
        bool gm_use_warp = false;
        inter::mc::Mv gm_mv{};
        global_mv_for_block(ref_index, r, c, w4, h4, gm_use_warp, gm_mv);
        std::vector<std::uint8_t> luma;
        bool predicted = false;
#if defined(AV2ENC_VIDEO_GME)
        if (gm_use_warp && gm_plan_ != nullptr &&
            ref_index < gm_plan_->num_refs) {
          std::string warp_error;
          predicted = warp_predict_plane(
              gm_plan_->params[static_cast<std::size_t>(ref_index)],
              references_[static_cast<std::size_t>(ref_index)], 0, ax0, ay0, w,
              h, r, c, w4, h4, luma, warp_error);
        }
#endif
        if (!predicted) {
          gm_use_warp = false;
          predicted = predict_luma(ref_index, gm_mv, ax0, ay0, w, h, luma);
        }
        if (predicted) {
          InterCandidate cand;
          cand.valid = true;
          cand.use_warp = gm_use_warp;
          cand.mode = syntax::kInterGlobalMv;
          cand.ref_index = ref_index;
          cand.mv = gm_mv;
          cand.sse = sse_vis(0, ax0, ay0, w, h, luma);
          if (trace_) {
            std::cerr << "  cand ref=" << ref_index << " global mv=("
                      << gm_mv.y << "," << gm_mv.x << ") warp="
                      << (gm_use_warp ? 1 : 0) << " sse=" << cand.sse
                      << "\n";
          }
          consider(cand);
          if (!global_cand_valid || cand.sse < global_cand.sse) {
            global_cand = cand;
            global_cand_valid = true;
          }
        }
      }
      if (have_stack && !options_.force_global_mv) {
        if (options_.enable_nearmv) {
          // §7.12.2.20 inserts MVP-combination candidates only when both
          // dimensions exceed 32.  merge::build_stack_checked already applies
          // that gate.
          // The historical search truncates small blocks to the first two
          // candidates.  With temporal MVs enabled the §7.12.2.7/.8 entries
          // can sit past that point (sorting by weight), so search the whole
          // stack; every index is codable and the writer validates it.
          const int near_limit =
              (w > 32 && h > 32) || options_.enable_ref_frame_mvs ||
                      options_.refmvbank
                  ? stack.count
                  : std::min(stack.count, 2);
          for (int i = 0; i < near_limit; ++i) {
            merge::ResolvedMode resolved;
            if (merge::resolve_single_mode(
                    stack, stack_params, merge::SingleMode::kNearMv, i,
                    inter::mc::Mv{0, 0}, &resolved) !=
                merge::Status::kOk) {
              continue;
            }
            std::vector<std::uint8_t> luma;
            const inter::mc::Mv near_mv = resolved.mv[0];
            if (!predict_luma(ref_index, near_mv, ax0, ay0, w, h, luma)) {
              continue;
            }
            InterCandidate cand;
            cand.valid = true;
            cand.mode = syntax::kInterNearMv;
            cand.ref_index = ref_index;
            cand.drl_index = i;
            cand.mv = near_mv;
            cand.from_temporal = stack.cand[i].from_temporal;
            cand.from_refmvbank = stack.cand[i].from_refmvbank;
            cand.sse = sse_vis(0, ax0, ay0, w, h, luma);
            if (options_.ref_mvs_stress && cand.from_temporal) {
              cand.sse = -1;
            }
            if (trace_) {
              std::cerr << "  cand ref=" << ref_index << " near drl=" << i
                        << " mv=(" << near_mv.y << "," << near_mv.x
                        << ") bank=" << (cand.from_refmvbank ? 1 : 0)
                        << " sse=" << cand.sse << "\n";
            }
            consider(cand);
          }
        }

        if (options_.enable_newmv) {
          // seq_force_integer_mv pins FrameMvPrecision to one pel for the
          // whole sequence, so the inter search codes one-pel MVDs.
          const int mv_precision =
              options_.seq_force_integer_mv ? 3 : options_.mv_precision;
          inter::mc::Mv me_mv{};
          if (run_me(ref_index, ax0, ay0, w, h, me_mv, error)) {
            me_mv = quantize_mv(me_mv, mv_precision);
            inter::mc::Mv pred = stack.cand[0].mv[0];
            // §5.20.7.13 assign_mv(): for a frame precision coarser than
            // 1/2 pel (MV_PRECISION_ONE_PEL) the decoder lowers the predictor
            // before adding DiffMv.  The quarter/half frames keep it as-is.
            if (mv_precision == 3) {
              me2::me2_lower_mv_precision(&pred.y, &pred.x,
                                          me2::kMvPrecisionOnePel);
            }
            // §5.20.7.13 only lowers the predictor when MvPrecision is
            // coarser than 1/2 pel.  Our quarter/half frames keep the
            // predictor as-is; the MVD itself is coded in precision units.
            const inter::mc::Mv delta{me_mv.y - pred.y, me_mv.x - pred.x};
            inter::mc::Mv mvd =
                mv_precision == 0
                    ? delta
                    : quantize_mv(delta, mv_precision);
            const int frame_prec =
                mv_precision == 0
                    ? me2::kMvPrecisionEighthPel
                    : (mv_precision == 1
                           ? me2::kMvPrecisionQuarterPel
                           : (mv_precision == 2
                                  ? me2::kMvPrecisionHalfPel
                                  : me2::kMvPrecisionOnePel));
            // §5.20.7.6 AMVD: the MVD magnitudes must be exactly
            // representable by Amvd_Index_To_Mvd.  When they are, the
            // index form codes fewer bits and reconstructs the same MV.
            bool use_amvd = false;
            int target = frame_prec;
            if (options_.enable_adaptive_mvd && (mvd.y != 0 || mvd.x != 0)) {
              int amvd_index = 0;
              const bool row_ok =
                  mvd.y == 0 ||
                  inter::mvprec::amvd_index_for_magnitude(mvd.y, &amvd_index);
              const bool col_ok =
                  mvd.x == 0 ||
                  inter::mvprec::amvd_index_for_magnitude(mvd.x, &amvd_index);
              use_amvd = row_ok && col_ok;
            }
            // §5.20.7.6 flexible MV resolution: pick the coarsest
            // signallable precision that represents the chosen MVD exactly.
            // From a 1/8-pel frame the reachable coarser step is 1/2 pel
            // (pb_mv_precision = 0); from a 1/4- or 1/2-pel frame it is
            // one pel.  An exact multiple of the coarser radix reconstructs
            // the identical MV with a smaller shell code.
            if (!use_amvd && options_.enable_flex_mvres) {
              const int coarse = frame_prec == me2::kMvPrecisionEighthPel
                                     ? me2::kMvPrecisionHalfPel
                                     : me2::kMvPrecisionOnePel;
              const int radix =
                  1 << (me2::kMvPrecisionEighthPel - coarse);
              if ((mvd.y % radix) == 0 && (mvd.x % radix) == 0) {
                target = coarse;
              }
            }
            // §5.20.7.13 mv_sign derivation.  The decoder derives the last
            // nonzero sign from the parity of the coded magnitudes, so the
            // encoder forces that sign before costing the candidate.
            inter::mvprec::MvdSignConfig sign_config;
            sign_config.enable_mvd_sign_derive =
                options_.enable_mvd_sign_derive;
            sign_config.frame_precision = frame_prec;
            sign_config.mv_precision = use_amvd ? frame_prec : target;
            sign_config.y_mode = inter::mvprec::kYModeNewMv;
            sign_config.is_adaptive_mvd = use_amvd;
            sign_config.is_compound = false;
            sign_config.ref_mv_idx = 0;
            bool sign_derived = false;
            if (inter::mvprec::mvd_sign_derive_allowed(sign_config)) {
              const int shift =
                  me2::kMvPrecisionEighthPel - sign_config.mv_precision;
              std::int32_t values[2] = {mvd.y, mvd.x};
              int last_sign = 0;
              int last_comp = -1;
              for (int comp = 0; comp < 2; ++comp) {
                if (values[comp] == 0) continue;
                const std::int32_t magnitude =
                    values[comp] < 0 ? -values[comp] : values[comp];
                last_sign += magnitude >> shift;
                last_comp = comp;
              }
              if (last_comp >= 0) {
                const bool negative = (last_sign & 1) != 0;
                std::int32_t& component = last_comp == 0 ? mvd.y : mvd.x;
                const std::int32_t magnitude =
                    component < 0 ? -component : component;
                component = negative ? -magnitude : magnitude;
                sign_derived = true;
              }
            }
            merge::ResolvedMode resolved;
            const bool resolved_ok =
                mv_precision == 0
                    ? merge::resolve_single_mode(
                          stack, stack_params, merge::SingleMode::kNewMv, 0,
                          mvd, &resolved) == merge::Status::kOk
                    : true;
            if (resolved_ok) {
              std::vector<std::uint8_t> luma;
              const inter::mc::Mv new_mv =
                  mv_precision == 0
                      ? resolved.mv[0]
                      : inter::mc::Mv{pred.y + mvd.y, pred.x + mvd.x};
              if (predict_luma(ref_index, new_mv, ax0, ay0, w, h, luma)) {
                InterCandidate cand;
                cand.valid = true;
                cand.mode = syntax::kInterNewMv;
                cand.ref_index = ref_index;
                cand.drl_index = 0;
                cand.mv = new_mv;
                cand.mvd = mv_precision == 0 ? resolved.mvd[0] : mvd;
                cand.use_amvd = use_amvd;
                cand.block_mv_precision =
                    target == frame_prec ? -1 : target;
                cand.mvd_sign_derived = sign_derived;
                cand.from_temporal = stack.cand[0].from_temporal;
                cand.sse =
                    sse_vis(0, ax0, ay0, w, h, luma);
                if (trace_) {
                  std::cerr << "  cand ref=" << ref_index << " new me=("
                            << me_mv.y << "," << me_mv.x << ") res=("
                            << new_mv.y << "," << new_mv.x << ") mvd=("
                            << cand.mvd.y << "," << cand.mvd.x
                            << ") amvd=" << (cand.use_amvd ? 1 : 0)
                            << " prec=" << cand.block_mv_precision
                            << " derive=" << (cand.mvd_sign_derived ? 1 : 0)
                            << " sse=" << cand.sse << "\n";
                }
                consider(cand);
              }
            }
          }
        }
      }
    }
  }

  // Compound candidate (§5.20.7.10 comp_mode + §7.13.3.18 AVERAGE blend).
  // The gated subset is non-joint GLOBAL_GLOBALMV over two references, so
  // both MVs are zero and the blend is the CWP_EQUAL average.  The blender
  // addresses frame coordinates, so it writes into per-frame scratch planes
  // and the block is copied out afterwards.
#if defined(AV2ENC_VIDEO_FRAMES)
  if (!keyframe_ && options_.compound && num_total_refs_ >= 2 &&
      w >= 8 && h >= 8 && !options_.force_intra && inter_fits) {
    const inter::mc::BlockRect rect{x0, y0, w, h};
    multiref::CompoundPrediction prediction;
    std::string comp_error;
    if (multiref::build_compound_prediction(
            references_[0], references_[1], rect,
            static_cast<int>(node.size), inter::mc::Mv{0, 0},
            inter::mc::Mv{0, 0}, leaf_interp_filter(),
            mi_cols_, mi_rows_, prediction, comp_error)) {
      std::uint8_t* dst[3] = {compound_pred_[0].samples.data(),
                              compound_pred_[1].samples.data(),
                              compound_pred_[2].samples.data()};
      int dst_stride[3] = {compound_pred_[0].stride,
                           compound_pred_[1].stride,
                           compound_pred_[2].stride};
      multiref::CompoundModeParams params;
      params.type = inter::adv::CompoundType::kAverage;
      params.cwp_weight = multiref::kCwpEqual;
      std::int64_t compound_sse = -1;
      if (options_.compound_wedge || options_.compound_diffwtd) {
        // §7.13.3.27-.30 masked compound decision over the block-local 1/16
        // predictions.  The search order (AVERAGE, wedge sign 0 then 1 over
        // signalable indices, then DIFFWTD mask_type 0/1) is the same
        // enumeration the CUDA backend uses, so both paths pick the same
        // block on an SSE tie.
        std::vector<std::uint8_t> search_scratch(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        inter::adv::CompoundSearchInput in{};
        in.bsize = static_cast<int>(node.size);
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
        in.scratch = search_scratch.data();
        in.scratch_stride = w;
        in.enable_average = true;
        in.enable_cwp = false;
        in.enable_wedge = options_.compound_wedge;
        in.enable_diffwtd = options_.compound_diffwtd;
        in.enable_imp_mask = false;
        const std::optional<inter::adv::CompoundDecision> decision =
            inter::adv::search_compound(in);
        if (!decision.has_value()) {
          error = "compound search failed";
          return false;
        }
        params.type = decision->type;
        params.cwp_weight = decision->cwp_weight;
        params.wedge_index = decision->wedge_index;
        params.wedge_sign = decision->wedge_sign;
        params.mask_type = decision->mask_type;
        compound_sse = decision->sse;
        if (trace_) {
          std::cerr << "  cand compound search type="
                    << static_cast<int>(decision->type)
                    << " wedge=" << decision->wedge_index
                    << " sign=" << decision->wedge_sign
                    << " mask_type=" << decision->mask_type
                    << " sse=" << decision->sse << "\n";
        }
      }
      if (multiref::apply_compound_prediction(prediction, params, dst,
                                              dst_stride, comp_error)) {
        sink.compound_y.resize(static_cast<std::size_t>(w) *
                               static_cast<std::size_t>(h));
        sink.compound_u.resize(static_cast<std::size_t>(cw) *
                               static_cast<std::size_t>(ch));
        sink.compound_v.resize(static_cast<std::size_t>(cw) *
                               static_cast<std::size_t>(ch));
        for (int yy = 0; yy < h; ++yy) {
          std::memcpy(sink.compound_y.data() +
                          static_cast<std::size_t>(yy) *
                              static_cast<std::size_t>(w),
                      compound_pred_[0].samples.data() +
                          static_cast<std::size_t>(y0 + yy) *
                              static_cast<std::size_t>(
                                  compound_pred_[0].stride) +
                          x0,
                      static_cast<std::size_t>(w));
        }
        for (int yy = 0; yy < ch; ++yy) {
          std::memcpy(sink.compound_u.data() +
                          static_cast<std::size_t>(yy) *
                              static_cast<std::size_t>(cw),
                      compound_pred_[1].samples.data() +
                          static_cast<std::size_t>(cy0 + yy) *
                              static_cast<std::size_t>(
                                  compound_pred_[1].stride) +
                          cx0,
                      static_cast<std::size_t>(cw));
          std::memcpy(sink.compound_v.data() +
                          static_cast<std::size_t>(yy) *
                              static_cast<std::size_t>(cw),
                      compound_pred_[2].samples.data() +
                          static_cast<std::size_t>(cy0 + yy) *
                              static_cast<std::size_t>(
                                  compound_pred_[2].stride) +
                          cx0,
                      static_cast<std::size_t>(cw));
        }
        InterCandidate cand;
        cand.valid = true;
        cand.compound = true;
        cand.mode = syntax::kInterGlobalMv;
        cand.ref_index = 0;
        cand.mv = inter::mc::Mv{0, 0};
        cand.sse = compound_sse >= 0
                       ? compound_sse
                       : sse_vis(0, x0, y0, w, h,
                                            sink.compound_y);
        cand.compound_type =
            params.type == inter::adv::CompoundType::kWedge
                ? 0
                : (params.type == inter::adv::CompoundType::kDiffWtd ? 1 : 2);
        cand.wedge_index = params.wedge_index;
        cand.wedge_sign = params.wedge_sign;
        cand.mask_type = params.mask_type;
        if (trace_) {
          std::cerr << "  cand compound global-global sse=" << cand.sse
                    << "\n";
        }
        consider(cand);
      }
    }
  }

  // §5.4.6 num_same_ref_compound > 0.  The same-reference pair (0,0) may be
  // coded as GLOBAL_GLOBALMV with zero MVs; the AVERAGE blend of the two
  // identical predictions is the §7.13.3.18 result for that pair.  This is
  // evaluated independently of the (0,1) carrier above so the flag produces
  // a real block decision (the gate counts the leaves it selects).
  if (!keyframe_ && options_.same_ref_compound && options_.compound &&
      num_total_refs_ >= 2 && w >= 8 && h >= 8 && !options_.force_intra &&
      inter_fits) {
    const inter::mc::Mv zeros[2] = {inter::mc::Mv{0, 0},
                                    inter::mc::Mv{0, 0}};
    multiref::CompoundModeParams params;
    params.type = inter::adv::CompoundType::kAverage;
    params.cwp_weight = multiref::kCwpEqual;
    std::string same_error;
    if (compound_plain(zeros, static_cast<int>(node.size), ax0, ay0, w, h,
                       sink.compound_y, sink.compound_u, sink.compound_v,
                       same_error, &params, /*ref0=*/0, /*ref1=*/0)) {
      InterCandidate cand;
      cand.valid = true;
      cand.compound = true;
      cand.same_ref = true;
      cand.mode = syntax::kInterGlobalMv;
      cand.ref_index = 0;
      cand.ref_index1 = 0;
      cand.compound_mode = 3;
      cand.compound_type = 2;  // AVERAGE (no comp_group_idx symbol)
      cand.mv = inter::mc::Mv{0, 0};
      cand.mv1 = inter::mc::Mv{0, 0};
      cand.sse = sse_vis(0, x0, y0, w, h, sink.compound_y);
      same_ref_best = cand;
    }
  }

  // §5.20.7.6 NEAR_NEARMV compound candidates (--video-opfl /
  // --video-refine-mv).  The two lists are merged independently; with
  // use_optflow the block instead codes a single RefMvIdx into the two-list
  // stack and runs §7.13.3.8-.12 on the prediction.
  if (!keyframe_ && options_.compound && options_.compound_near_near &&
      !options_.force_global_mv && num_total_refs_ >= 2 && w >= 8 && h >= 8 &&
      !options_.force_intra && inter_fits) {
    const int m = 5;  // max_drl_bits_minus_1 + 1
    merge::StackParams p0;
    merge::StackParams p1;
    merge::MvStack l0;
    merge::MvStack l1;
    const bool have0 = load_stack(r, c, w4, h4, 0, sink.state, p0, l0);
    const bool have1 = load_stack(r, c, w4, h4, 1, sink.state, p1, l1);
    const int d0 = frame_distance_[0];
    const int d1 = frame_distance_[1];
    const bool optflow_ok =
        options_.opfl_refine_type != 0 &&
        temporal::opfl_allowed_for_refs(d0, d1, false, false, false);
    // §5.20.7.17: a NEAR_NEARMV compound block is not switchable, so a
    // frame with enable_refinemv derives use_refinemv for every eligible
    // block.  The decision must match the prediction, so it is per block.
    bool block_refine = false;
    if (options_.enable_refinemv) {
      temporal::RefineMvConfig rc;
      rc.enable_refinemv = true;
      rc.opfl_refine_type =
          static_cast<temporal::RefineType>(options_.opfl_refine_type);
      rc.frame_is_switch = false;
      rc.is_compound = true;
      rc.skip_mode = false;
      rc.use_optflow = false;
      rc.is_global_global_mv = false;
      rc.motion_mode_simple = true;
      rc.has_newmv = false;
      rc.y_mode_near_near_mv = true;
      rc.y_mode_joint_new_mv = false;
      rc.block_width = w;
      rc.block_height = h;
      rc.order_hint = 64;
      rc.ref0_order_hint = 64 - d0;
      rc.ref1_order_hint = 64 - d1;
      temporal::RefineMvSyntax syntax;
      if (temporal::refine_mv_syntax(rc, &syntax) ==
          temporal::TemporalStatus::kOk) {
        block_refine = syntax.use_refinemv;
      }
    }
    std::vector<std::pair<inter::mc::Mv, inter::mc::Mv>> seen;
    const auto already = [&seen](inter::mc::Mv a, inter::mc::Mv b) {
      for (const auto& s : seen) {
        if (s.first.y == a.y && s.first.x == a.x && s.second.y == b.y &&
            s.second.x == b.x) {
          return true;
        }
      }
      seen.emplace_back(a, b);
      return false;
    };
    const auto evaluate = [&](inter::mc::Mv mv0, inter::mc::Mv mv1,
                              int drl0, int drl1, bool use_optflow) {
      if (already(mv0, mv1)) return;
      bool ok = false;
      const inter::mc::Mv mvs[2] = {mv0, mv1};
      multiref::CompoundModeParams chosen_params;
      std::int64_t chosen_sse = -1;
      if (use_optflow) {
        ok = predict_compound_optflow(sink, mvs, ax0, ay0, w, h, error);
      } else if (block_refine) {
        ok = predict_compound_refinemv(sink, mvs, ax0, ay0, w, h, error);
      } else if (options_.enable_cwp_weighted ||
                 options_.enable_imp_msk_bld) {
        ok = compound_search_average(
            mvs, static_cast<int>(node.size), ax0, ay0, w, h,
            sink.compound_y, sink.compound_u, sink.compound_v, chosen_params,
            chosen_sse, error);
      } else {
        ok = compound_plain(mvs, static_cast<int>(node.size), ax0, ay0, w, h,
                            sink.compound_y, sink.compound_u, sink.compound_v,
                            error);
      }
      if (!ok) return;
      InterCandidate cand;
      cand.valid = true;
      cand.compound = true;
      cand.mode = syntax::kInterNearMv;
      cand.ref_index = 0;
      cand.mv = mv0;
      cand.mv1 = mv1;
      cand.drl_index = drl0;
      cand.drl_index1 = drl1;
      cand.compound_mode = 0;  // NEAR_NEARMV
      cand.use_optflow = use_optflow;
      cand.use_refinemv = block_refine && !use_optflow;
      cand.cwp_index = chosen_params.cwp_index;
      cand.cwp_weight = chosen_params.cwp_weight;
      cand.use_imp_mask = chosen_params.use_imp_mask;
      cand.sse = chosen_sse >= 0
                     ? chosen_sse
                     : sse_vis(0, ax0, ay0, w, h,
                                          sink.compound_y);
      if (trace_) {
        std::cerr << "  cand compound near-near mv0=(" << mv0.y << ","
                  << mv0.x << ") mv1=(" << mv1.y << "," << mv1.x
                  << ") drl=(" << drl0 << "," << drl1 << ") optflow="
                  << (use_optflow ? 1 : 0) << " refinemv="
                  << (cand.use_refinemv ? 1 : 0) << " cwp="
                  << cand.cwp_index << " imp="
                  << (cand.use_imp_mask ? 1 : 0) << " sse=" << cand.sse
                  << "\n";
      }
      consider(cand);
    };
    if (optflow_ok && !block_refine) {
      merge::StackParams cparams =
          make_stack_params(r, c, w4, h4, 0, sink.state);
      cparams.ref_frame[1] = 1;
      cparams.is_compound = true;
      bool gm_warp = false;
      inter::mc::Mv gm_mv1{};
      global_mv_for_block(1, r, c, w4, h4, gm_warp, gm_mv1);
      cparams.global_mv[1] = gm_mv1;
      merge::MvStack compound;
      if (merge::build_stack_checked(cparams, &compound) ==
          merge::Status::kOk) {
        const int limit = std::min(compound.count, m);
        for (int i = 0; i < limit; ++i) {
          const inter::mc::Mv mv0 =
              merge::clamp_mv_to_integer(compound.cand[i].mv[0]);
          const inter::mc::Mv mv1 =
              merge::clamp_mv_to_integer(compound.cand[i].mv[1]);
          evaluate(mv0, mv1, i, 0, /*use_optflow=*/true);
        }
      }
    }
    if (have0 && have1) {
      const int limit0 =
          std::min(l0.count, block_refine ? 2 : m);
      const int limit1 =
          std::min(l1.count, block_refine ? 2 : m);
      for (int i = 0; i < limit0; ++i) {
        for (int j = 0; j < limit1; ++j) {
          merge::MvStack pair[2] = {l0, l1};
          inter::mc::Mv resolve[2];
          if (merge::resolve_compound_near_near(pair, i, j, resolve) !=
              merge::Status::kOk) {
            continue;
          }
          evaluate(resolve[0], resolve[1], i, j, /*use_optflow=*/false);
        }
      }
    }
  }
  // Prefer the same-reference compound carrier on an SSE tie with the best
  // ordinary candidate.  Both predictions are decoder-exact; the preference
  // makes --video-same-ref-compound actually exercise the §5.20.7.10
  // compound_mode_same_refs syntax instead of being shadowed by GLOBALMV.
  if (same_ref_best.valid &&
      (!best.valid || same_ref_best.sse <= best.sse)) {
    best = same_ref_best;
  }
#endif  // AV2ENC_VIDEO_FRAMES

  // Intra fallback: DC prediction + DCT residual, the M1 transform path.
  // With --video-intra-full the luma leaf runs the still product's
  // 61-candidate search (see src/pipeline/video/video_intra_search.h);
  // chroma stays DC_PRED because the M2 syntax cannot signal more.
  bool have_intra = false;
  std::int64_t intra_sse = 0;
  std::vector<std::uint8_t> intra_y_pred;
  std::vector<std::uint8_t> intra_u_pred;
  std::vector<std::uint8_t> intra_v_pred;
  TxPlan intra_y;
  TxPlan intra_u;
  TxPlan intra_v;
  int intra_cctx_type = 0;
  std::uint8_t intra_y_mode = 0;  // AV2 YMode (0 = DC_PRED)
  std::int8_t intra_y_angle_delta = 0;
  std::uint8_t intra_mrl_index = 0;
  bool intra_mrl_sec_index = false;
  bool intra_use_dip = false;
  int intra_dip_mode = 0;
  bool intra_dip_transpose = false;
  bool intra_is_cfl = false;
  int intra_cfl_index = 1;
  int intra_cfl_mh_dir = 0;
  // The reconstruction planes are tile-local, so plain bounds give the
  // decoder's §5.20.9.1 availability at every tile edge.  The bottom bound
  // is the coded MI extent, not the SB-aligned --video-edge-inter buffer.
  const int luma_max_x = recon_[0].width - 1;
  const int luma_max_y = mi_rows_ * 4 - 1;
  const bool luma_left = x0 > 0;
  const bool luma_above = y0 > 0;
  const int chroma_max_x = recon_[1].width - 1;
  const int chroma_max_y = mi_rows_ * 2 - 1;
  const bool chroma_left = cx0 > 0;
  const bool chroma_above = cy0 > 0;
  if (keyframe_ || options_.enable_intra_fallback || options_.force_intra) {
    bool luma_ok = chroma_part;
    if (!chroma_part) {
    const intra_search::Availability luma_avail = intra_search::availability(
        st.decoded[0], x0 / 4, y0 / 4, w / 4, h / 4, mi_cols_, mi_rows_,
        /*sb_mi=*/sb_size_ / 4);
    if (options_.enable_intra_full) {
      bool above_smooth = false;
      bool left_smooth = false;
      luma_neighbor_smooth(r, c, above_smooth, left_smooth);
      intra_search::LumaChoice choice;
      intra_y_pred.resize(static_cast<std::size_t>(w) *
                          static_cast<std::size_t>(h));
      luma_ok = intra_search::search_luma(
          recon_[0].samples.data(), recon_[0].stride, x0, y0, w, h,
          luma_avail,
          above_smooth, left_smooth, /*enable_edge_filter=*/true,
          source_.y.data(), static_cast<int>(source_.width),
          static_cast<int>(source_.width) - 1,
          static_cast<int>(source_.height) - 1, intra_y_pred.data(), choice,
          options_.enable_ibp);
      if (luma_ok) {
        intra_y_mode = static_cast<std::uint8_t>(choice.mode);
        intra_y_angle_delta = static_cast<std::int8_t>(choice.angle_delta);
        if (options_.enable_mrl) {
          // §5.20.5.5 MRL: only directional candidates can signal
          // mrl_index.  Re-use the still encoder's candidate order and
          // strict `<` tie-break; the baseline search already holds the
          // mrl_index 0 winner.
          const int sec_allowed = !(w == 4 && h == 4);
          std::int64_t best_sse = choice.sse;
          std::array<intra::ModeCandidate, intra::kMaxModeCandidates>
              candidates;
          const int count = intra::build_mode_candidates(candidates, true);
          for (int i = 0; i < count; ++i) {
            const intra::ModeCandidate& candidate =
                candidates[static_cast<std::size_t>(i)];
            if (!intra::is_directional(candidate.mode)) continue;
            for (int mrl_index = 1; mrl_index <= mrl::kMaxMrlIndex;
                 ++mrl_index) {
              for (int sec = 0; sec <= (sec_allowed ? 1 : 0); ++sec) {
                std::vector<std::uint8_t> mrl_pred;
                if (!predict_luma_mrl(
                        recon_[0], luma_avail, x0, y0, w, h, above_smooth,
                        left_smooth, static_cast<int>(candidate.mode),
                        candidate.angle_delta, mrl_index, sec != 0,
                        /*sb_height=*/64, mrl_pred, error)) {
                  return false;
                }
                const std::int64_t mrl_sse = sse_against_source(
                    source_, 0, x0, y0, w, h, mrl_pred);
                if (mrl_sse < best_sse) {
                  best_sse = mrl_sse;
                  intra_y_mode =
                      static_cast<std::uint8_t>(candidate.mode);
                  intra_y_angle_delta =
                      static_cast<std::int8_t>(candidate.angle_delta);
                  intra_mrl_index = static_cast<std::uint8_t>(mrl_index);
                  intra_mrl_sec_index = sec != 0;
                  intra_y_pred = std::move(mrl_pred);
                }
              }
            }
          }
          if (intra_mrl_index == 0) intra_mrl_sec_index = false;
        }
      }
    } else {
      luma_ok = intra_dc_predict(recon_[0], source_, 0, x0, y0, w, h,
                                 luma_max_x, luma_max_y, luma_left,
                                 luma_above, options_.enable_ibp,
                                 intra_y_pred);
    }
    // §5.20.7.9/§7.13.2.3: DIP is an alternative to ordinary DC at the same
    // YMode on the intra keyframe, not a new mode.  Evaluate the normative
    // candidate set and keep it only when its prediction SSE beats the
    // incumbent.
    if (options_.enable_dip && keyframe_ && luma_ok &&
        dip::dip_block_available(w, h)) {
      dip::Block dip_block;
      dip_block.recon = dip::PlaneView::from_u8(
          recon_[0].samples.data(), recon_[0].width);
      dip_block.x = x0;
      dip_block.y = y0;
      dip_block.w = w;
      dip_block.h = h;
      dip_block.max_x = luma_max_x;
      dip_block.max_y = luma_max_y;
      dip_block.have_left = luma_left;
      dip_block.have_above = luma_above;
      dip_block.num4_above_right = luma_avail.num4_above_right;
      dip_block.num4_below_left = luma_avail.num4_below_left;
      dip::Edges dip_edges;
      if (dip::prepare_edges(dip_block, 8, dip_edges) != dip::Status::kOk) {
        error = "dip::prepare_edges failed";
        return false;
      }
      std::vector<std::uint8_t> dip_src(
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
      for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
          dip_src[static_cast<std::size_t>(yy) *
                      static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(xx)] =
              static_cast<std::uint8_t>(
                  source_sample(source_, 0, ax0 + xx, ay0 + yy));
        }
      }
      dip::DecisionParams dip_params;
      dip_params.enable_dip = true;
      dip_params.y_mode = 0;  // DC_PRED
      dip_params.palette_size_y = 0;
      dip_params.tree_is_chroma_part = false;
      dip_params.metric = dip::CostMetric::kSse;
      dip::Decision dip_decision;
      if (dip::decide(dip_edges, dip_src.data(),
                      static_cast<std::size_t>(w), dip_params,
                      dip_decision) != dip::Status::kOk ||
          !dip_decision.available) {
        error = "dip::decide failed";
        return false;
      }
      const std::int64_t incumbent_sse =
          sse_vis(0, ax0, ay0, w, h, intra_y_pred);
      if (static_cast<std::int64_t>(dip_decision.cost.sse) <
          incumbent_sse) {
        dip::Params dip_pred_params;
        dip_pred_params.mode = dip_decision.mode;
        dip_pred_params.transpose = dip_decision.transpose;
        std::vector<std::uint8_t> dip_pred(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        if (dip::predict_block8(dip_block, 8, dip_pred_params,
                                dip_pred.data(),
                                static_cast<std::size_t>(w)) !=
            dip::Status::kOk) {
          error = "dip::predict_block8 failed";
          return false;
        }
        intra_y_pred = std::move(dip_pred);
        intra_y_mode = 0;
        intra_y_angle_delta = 0;
        intra_mrl_index = 0;
        intra_mrl_sec_index = false;
        intra_use_dip = true;
        intra_dip_mode = dip_decision.mode;
        intra_dip_transpose = dip_decision.transpose;
      }
    }
    // --video-key-intra-txpart: an intra luma partition needs per-part
    // prediction, so it replaces the single-predictor TX search here.  DIP is
    // a whole-block alternative prediction and stays TX_PARTITION_NONE.
    const bool intra_tx_partition_search =
        options_.enable_key_intra_tx_part && options_.enable_tx_search &&
        keyframe_ && !chroma_part && !intra_use_dip &&
        intra_y_pred.size() ==
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    bool part_above_smooth = false;
    bool part_left_smooth = false;
    if (intra_tx_partition_search) {
      luma_neighbor_smooth(r, c, part_above_smooth, part_left_smooth);
    }
    if (!luma_ok ||
        (options_.enable_tx_search
             ? (intra_tx_partition_search
                    ? !encode_intra_luma_tx_partition(
                          r, c, w, h, node.size, intra_y_pred, intra_y_mode,
                          intra_y_angle_delta, intra_mrl_index,
                          intra_mrl_sec_index, part_above_smooth,
                          part_left_smooth, luma_mask_row, sink.state,
                          &intra_y, error)
                    : !search_luma_tx(
                          source_, ax0, ay0, w, h, intra_y_pred,
                          st.current_qindex, txtsearch::Mode::kIntra,
                          key_tx_rd_lambda(options_, st.current_qindex) *
                              perceptual::tx_lambda_scale(perceptual_weights),
                          options_.enable_idtx_intra,
                          std::max(0, mrl_stream::mapped_intra_mode_for_mrl(
                                          intra_y_mode, intra_y_angle_delta,
                                          intra_mrl_index, w, h)),
                          options_.enable_tcq,
                          ::av2enc::still::qctx::slice_for_base_q_idx(
                              options_.tx_context_rate ? base_q_index()
                                                       : st.current_qindex),
                          /*enable_inter_ddt=*/false,
                          /*search_partition=*/false,
                          /*context_rate=*/options_.tx_context_rate,
                          options_.trace, /*allow_rdoq=*/false,
                          /*rdoq_qctx=*/0,
                          rdoq_sse_scale_at(options_, st.current_qindex),
                          intra_y, error, luma_mask_row))
             : !compute_tx(source_, 0, ax0, ay0, w, h, intra_y_pred,
                           st.plane_quant[0], intra_y, error,
                           qm_enabled_ ? &qm_state_ : nullptr,
                           options_.enable_tcq,
                           ::av2enc::still::qctx::slice_for_base_q_idx(
                               st.current_qindex),
                           /*is_inter=*/false, /*allow_rdoq=*/false,
                           /*rdoq_qctx=*/0,
                           rdoq_sse_scale_at(options_, st.current_qindex),
                           luma_mask_row))) {
      error = "intra prediction failed";
      return false;
    }
    // §7.15.3 luma IST refinement (--video-ist).  Applies the still
    // encoder's analysis projection to the quantised luma levels and keeps
    // the candidate that minimises SSE + lambda * symbol bits.
    if (options_.enable_ist &&
        !refine_luma_ist(source_, ax0, ay0, w, h, intra_y_pred,
                         st.quant_step, intra_y_mode, intra_y_angle_delta,
                         intra_mrl_index, intra_y, error)) {
      return false;
    }
    // §7.13.5 CfL reads the reconstructed luma of this block, so the luma
    // reconstruction must be in the plane before the chroma decision.  A
    // losing intra candidate is overwritten by the inter reconstruction.
    write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
    }  // !chroma_part
    if (owns_chroma) {
    if (!intra_dc_predict(recon_[1], source_, 1, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_u_pred) ||
        !intra_dc_predict(recon_[2], source_, 2, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_v_pred)) {
      error = "intra prediction failed";
      return false;
    }
    if (options_.enable_cfl) {
      // Opt-in CfL chroma decision (§7.13.5).  The candidate set is the M1
      // still product's default {DC, CfL} with derived alpha.
      const std::int64_t dc_sse_u =
          sse_vis(1, acx0, acy0, cw, ch, intra_u_pred);
      const int frame_chroma_width = static_cast<int>((source_.width + 1U) / 2U);
      const int frame_chroma_height =
          static_cast<int>((source_.height + 1U) / 2U);
      chroma_search::PlaneView u_plane;
      u_plane.recon = recon_[1].samples.data();
      u_plane.recon_stride = recon_[1].stride;
      u_plane.recon_width = recon_[1].width;
      u_plane.recon_height = mi_rows_ * 2;
      // The helper indexes the source with the block origin, so the pointer
      // carries the tile origin only (coded coordinates stay tile-local).
      const int tile_cx = tile_x_ / 2;
      const int tile_cy = tile_y_ / 2;
      u_plane.source = source_.u.data() +
                       static_cast<std::size_t>(tile_cy) *
                           static_cast<std::size_t>(frame_chroma_width) +
                       static_cast<std::size_t>(tile_cx);
      u_plane.source_stride = frame_chroma_width;
      u_plane.source_max_x = frame_chroma_width - 1 - tile_cx;
      u_plane.source_max_y = frame_chroma_height - 1 - tile_cy;
      chroma_search::PlaneView v_plane = u_plane;
      v_plane.recon = recon_[2].samples.data();
      v_plane.recon_stride = recon_[2].stride;
      v_plane.source = source_.v.data() +
                       static_cast<std::size_t>(tile_cy) *
                           static_cast<std::size_t>(frame_chroma_width) +
                       static_cast<std::size_t>(tile_cx);
      const intra_search::Availability chroma_avail =
          intra_search::availability(st.decoded[1], cx0 / 4, cy0 / 4, cw / 4,
                                     ch / 4, mi_cols_ / 2, mi_rows_ / 2,
                                     /*sb_mi=*/sb_size_ / 8);
      chroma_search::Options chroma_options;
      chroma_options.enable_cfl = true;
      chroma_options.ds_filter = 0;
      chroma_options.enable_mhccp = options_.enable_mhccp;
      chroma_search::Choice choice;
      if (!chroma_search::choose(
              u_plane, v_plane, recon_[0].samples.data(), recon_[0].stride,
              mi_cols_ * 4 - 1, mi_rows_ * 4 - 1, (cy0 * 2 / 64) * 64, cx0,
              cy0, cw, ch, mi_cols_ * 2, mi_rows_ * 2,
              chroma_avail.num4_above_right, chroma_avail.num4_below_left,
              intra_u_pred, intra_v_pred, dc_sse_u, chroma_options, choice,
              error)) {
        return false;
      }
      intra_is_cfl = choice.is_cfl;
      intra_cfl_index = choice.cfl_index;
      intra_cfl_mh_dir = choice.cfl_mh_dir;
      intra_u_pred = choice.u;
      intra_v_pred = choice.v;
    }
    if (options_.enable_cctx) {
      // §5.4.8 enable_cctx / §7.14.6: joint U/V search.  is_cctx_allowed()
      // is true for 4:2:0; the CLI rejects the other formats with CCTX.
      if (!compute_chroma_cctx(source_, acx0, acy0, cw, ch, intra_u_pred,
                               intra_v_pred, st.quant_step,
                               /*is_inter=*/false,
                               intra_u, intra_v, &intra_cctx_type, error,
                               chroma_mask_row)) {
        return false;
      }
    } else if (!compute_tx(source_, 1, acx0, acy0, cw, ch, intra_u_pred,
                           st.plane_quant[1], intra_u, error,
                           qm_enabled_ ? &qm_state_ : nullptr,
                           /*allow_tcq=*/false, /*qctx_slice=*/0,
                           /*is_inter=*/false, chroma_mask_row) ||
               !compute_tx(source_, 2, acx0, acy0, cw, ch, intra_v_pred,
                           st.plane_quant[2], intra_v, error,
                           qm_enabled_ ? &qm_state_ : nullptr,
                           /*allow_tcq=*/false, /*qctx_slice=*/0,
                           /*is_inter=*/false, chroma_mask_row)) {
      return false;
    }
    }  // owns_chroma
    intra_sse = chroma_part
                    ? 0
                    : sse_vis(0, ax0, ay0, w, h,
                                         intra_y.recon);
    if (trace_) {
      std::cerr << "  cand intra sse=" << intra_sse;
      if (options_.enable_cctx) {
        std::cerr << " cctx=" << intra_cctx_type;
      }
      std::cerr << "\n";
    }
    have_intra = true;
  }

  // §5.20.8 luma palette (--video-palette).  The palette codes eob=0 on all
  // three planes, so its prediction is compared with the SSE of the ordinary
  // intra *prediction*, not the residual-coded reconstruction.
  PaletteChoice palette_choice;
  bool use_palette = false;
  std::int64_t intra_pred_sse = intra_sse;
  if (keyframe_ && palette_enabled_ && have_intra) {
    intra_pred_sse =
        sse_vis(0, ax0, ay0, w, h, intra_y_pred);
    if (!choose_palette_leaf(r, c, static_cast<int>(node.size), w, h,
                             intra_pred_sse, palette_choice, error)) {
      return false;
    }
    use_palette = palette_choice.found;
    if (trace_) {
      std::cerr << "  cand palette applied=" << (use_palette ? 1 : 0)
                << " pred_sse=" << intra_pred_sse
                << " palette_sse=" << palette_choice.sse << "\n";
    }
  }

  // SEG_LVL_SKIP / SEG_LVL_GLOBALMV: the decoder does not read is_inter or
  // any mode/ref syntax and forces YMode = GLOBALMV with SkipSegFrame.  Keep
  // the first-reference global candidate whenever one exists.
  if (seg_forced_inter && global_cand_valid) best = global_cand;

  // Inter residual (feature 1).  The predictor comes from the winner of the
  // motion search; when enabled, a quantised residual is coded for Y/U/V and
  // the reconstructed block is pred + dequantised residual.  The decision is
  // an SSE + lambda * estimated-bits comparison against skip_flag = 1.
  // One fully evaluated inter candidate.  The proxy search ranks candidates by
  // prediction SSE + static mode syntax; this structure holds what the real
  // predict -> TX -> quantise -> reconstruct -> rate path produced for one of
  // them, including the measured D and R used by AV2ENC_EXACT_RD.
  struct InterLeafResult {
    InterCandidate cand;
    std::vector<std::uint8_t> y;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
    bool residual = false;
    TxPlan res_y;
    TxPlan res_u;
    TxPlan res_v;
    int cctx = 0;
    bool bawp = false;
    inter::adv::BawpFit fit{};
    bool interintra = false;
    int ii_mode = 0;
    bool ii_wedge = false;
    int ii_index = 0;
    std::int64_t ii_sse = 0;
    std::int64_t luma_sse = 0;
    // Measured objective terms.  Only filled when AV2ENC_EXACT_RD is on.
    double distortion = 0.0;
    double bits = 0.0;
    // AV2ENC_PIPE_PERC objective terms of the emitted form.  `pipe_bits`
    // includes the exact syntax (is_inter/skip/mode/ref/DRL/MVD) and the
    // context-coded residual; `pipe_distortion` is the local weighted D.
    double pipe_distortion = 0.0;
    double pipe_bits = 0.0;
    bool pipe_modeled = false;
    // §5.20.7.6 switchable interp_filter (0..2) carried by this evaluation.
    int filter = 0;
    bool filter_coded = false;
  };

  // Real leaf path for one candidate.  This is the historical winner-only
  // block, parameterised by the candidate instead of `best`: predict, apply
  // BAWP / inter-intra, run the shared luma TX search and chroma transform /
  // quantise / reconstruct, and decide skip vs residual.  With the gate off it
  // runs exactly once for the proxy winner, so every historical stream keeps
  // its bytes.
  const auto evaluate_inter_leaf =
      [&](const InterCandidate& input, InterLeafResult& out) -> bool {
    out = InterLeafResult{};
    out.cand = input;
    out.luma_sse = input.sse;
    if (out.cand.compound) {
#if defined(AV2ENC_VIDEO_FRAMES)
      // Re-run the candidate: the compound scratch planes hold the last
      // evaluated candidate, not necessarily this one.
      const inter::mc::Mv mvs[2] = {out.cand.mv, out.cand.mv1};
      bool predicted = false;
      if (out.cand.use_optflow) {
        predicted =
            predict_compound_optflow(sink, mvs, ax0, ay0, w, h, error);
      } else if (out.cand.use_refinemv) {
        predicted =
            predict_compound_refinemv(sink, mvs, ax0, ay0, w, h, error);
      } else {
        multiref::CompoundModeParams wparams;
        wparams.cwp_weight = out.cand.cwp_weight;
        wparams.cwp_index = static_cast<int>(out.cand.cwp_index);
        wparams.use_imp_mask = out.cand.use_imp_mask;
        wparams.type = out.cand.compound_type == 0
                           ? inter::adv::CompoundType::kWedge
                           : (out.cand.compound_type == 1
                                  ? inter::adv::CompoundType::kDiffWtd
                                  : inter::adv::CompoundType::kAverage);
        if (out.cand.compound_mode == 0) {
          wparams.type = inter::adv::CompoundType::kAverage;
        }
        wparams.wedge_index = out.cand.wedge_index;
        wparams.wedge_sign = out.cand.wedge_sign;
        wparams.mask_type = out.cand.mask_type;
        predicted = compound_plain(
            mvs, static_cast<int>(node.size), ax0, ay0, w, h,
            sink.compound_y, sink.compound_u, sink.compound_v, error,
            &wparams, 0, out.cand.same_ref ? 0 : 1);
      }
      if (!predicted) {
        error = "compound prediction failed";
        return false;
      }
#endif
      out.y = sink.compound_y;
      out.u = sink.compound_u;
      out.v = sink.compound_v;
    } else if (out.cand.use_warp) {
#if defined(AV2ENC_VIDEO_GME)
      std::string warp_error;
      if (gm_plan_ == nullptr || out.cand.ref_index >= gm_plan_->num_refs ||
          !warp_predict_420(
              gm_plan_->params[static_cast<std::size_t>(out.cand.ref_index)],
              references_[static_cast<std::size_t>(out.cand.ref_index)], ax0,
              ay0, w, h, r, c, w4, h4, out.y, out.u, out.v, warp_error)) {
        error = "inter warp prediction failed: " + warp_error;
        return false;
      }
#else
      error = "inter warp prediction is not linked in this build";
      return false;
#endif
    } else if (!predict_all(out.cand.ref_index, out.cand.mv, ax0, ay0, w, h,
                            out.y, out.u, out.v)) {
      error = "inter prediction failed";
      return false;
    }
    out.luma_sse = sse_vis(0, ax0, ay0, w, h, out.y);
    // §5.20.7.6 / §7.13.3.25: evaluate the fitted BAWP predictor for an
    // eligible single-prediction block (>= 8x8, not GLOBALMV, no warp).
    // The decoder applies the same fit before the residual, so a selected
    // block carries use_bawp = 1 and the adapted prediction.
    //
    // A GLOBALMV winner may still take BAWP through a NEARMV/NEWMV carrier
    // at the same MV: §5.20.7.6 does not code use_bawp on GLOBALMV, so the
    // encoder re-expresses an identical prediction in a mode that can.  The
    // swap never changes the MV or reference.  Segment-forced GLOBALMV
    // leaves keep their forced mode.
    const bool bawp_carrier_ok =
        out.cand.mode != syntax::kInterGlobalMv ||
        (options_.bawp_carrier && !seg_forced_inter &&
         bawp_carrier.valid &&
         bawp_carrier.ref_index == out.cand.ref_index &&
         bawp_carrier.mv == out.cand.mv);
    if (options_.enable_bawp && options_.tile_cols == 0 && !out.cand.compound &&
        !out.cand.use_warp && bawp_carrier_ok && w >= 8 && h >= 8 &&
        mi_cols_ * 4 == static_cast<int>(source_.width) &&
        mi_rows_ * 4 == static_cast<int>(source_.height)) {
      std::vector<std::uint8_t> adapted = out.y;
      inter::adv::BawpFit fit{};
      const inter::mc::PlaneView& ref_y =
          references_[static_cast<std::size_t>(out.cand.ref_index)].plane[0];
      if (bawp_adapt_luma(recon_[0].samples.data(), recon_[0].stride,
                          ref_y.data, ref_y.stride, mi_cols_, mi_rows_, r,
                          c, x0, y0, w, h, out.cand.mv.y, out.cand.mv.x,
                          adapted, &fit)) {
        // Keep the fit even when BAWP is not selected: the fade gate uses it
        // as the per-leaf illumination-ramp evidence.
        out.fit = fit;
        const std::int64_t adapted_sse =
            sse_vis(0, ax0, ay0, w, h, adapted);
        if (adapted_sse < out.luma_sse) {
          if (out.cand.mode == syntax::kInterGlobalMv) {
            out.cand = bawp_carrier;
          }
          out.bawp = true;
          out.fit = fit;
          out.y = adapted;
          out.luma_sse = adapted_sse;
        }
      }
      if (trace_) {
        std::cerr << "  cand bawp alpha=" << out.fit.alpha
                  << " count=" << out.fit.count
                  << " carrier=" << static_cast<int>(out.cand.mode)
                  << " applied=" << (out.bawp ? 1 : 0) << "\n";
      }
    }
    // §5.20.7.14/.15 + §7.13.3.29/.30 single-reference inter-intra.  The
    // blended block is skip-only (no residual) in this gated surface, and it
    // is evaluated after BAWP so the §5.20.7.14 use_bawp early return can
    // never be violated.
    if (options_.enable_interintra && !out.cand.compound &&
        !out.cand.use_warp && !out.bawp &&
        out.cand.mode != syntax::kInterGlobalMv && w >= 8 && h >= 8 &&
        w <= 64 && h <= 64 && !options_.force_inter_residual) {
      std::vector<std::uint8_t> ii_y;
      std::vector<std::uint8_t> ii_u;
      std::vector<std::uint8_t> ii_v;
      if (!evaluate_inter_intra(static_cast<int>(node.size), x0, y0, w, h,
                                out.y, out.u, out.v, ii_y, ii_u, ii_v,
                                out.ii_mode, out.ii_wedge, out.ii_index,
                                out.ii_sse, sink.state, error)) {
        return false;
      }
      const std::int64_t plain_sse =
          sse_vis(0, ax0, ay0, w, h, out.y) +
          sse_vis(1, acx0, acy0, cw, ch, out.u) +
          sse_vis(2, acx0, acy0, cw, ch, out.v);
      if (out.ii_sse < plain_sse) {
        out.interintra = true;
        out.y = ii_y;
        out.u = ii_u;
        out.v = ii_v;
        out.luma_sse = sse_vis(0, ax0, ay0, w, h, out.y);
      }
      if (trace_) {
        std::cerr << "  cand interintra applied="
                  << (out.interintra ? 1 : 0)
                  << " mode=" << out.ii_mode
                  << " wedge=" << (out.ii_wedge ? 1 : 0)
                  << " index=" << out.ii_index
                  << " sse=" << out.ii_sse
                  << " plain=" << plain_sse << "\n";
      }
    }
    // A compound leaf is coded skip-only in the gated subset: the writer
    // forces skip_flag = 1 below and never codes the residual.  Running the
    // residual search here would replace the prediction with
    // pred + dequantised residual in the encoder recon while the decoder
    // reconstructs the plain compound prediction, so the recon drifts from
    // the bitstream.  Leave compound leaves prediction-only, as emitted.
    if (options_.enable_inter_residual && !out.cand.compound &&
        !out.interintra && !seg_skip_leaf) {
      if (ba_on) ++pipe_ba::counters().tx_calls;
      const double pred_total =
          sse_pipe(0, ax0, ay0, w, h, out.y) +
          sse_pipe(1, acx0, acy0, cw, ch, out.u) +
          sse_pipe(2, acx0, acy0, cw, ch, out.v);
      if (((options_.enable_tx_search || options_.enable_inter_ddt)
               ? !search_luma_tx(source_, ax0, ay0, w, h, out.y,
                                 st.current_qindex,
                                 txtsearch::Mode::kInter,
                                 (pipe_on
                                      ? pipe_lambda
                                      : tx_rd_lambda(options_,
                                                     st.current_qindex) *
                                            perceptual::tx_lambda_scale(
                                                perceptual_weights)),
                                 /*allow_idtx=*/false,
                                 /*intra_mapped_mode=*/0,
                                 options_.enable_tcq,
                                 ::av2enc::still::qctx::slice_for_base_q_idx(
                                     (pipe_on || options_.tx_context_rate)
                                         ? base_q_index()
                                         : st.current_qindex),
                                 options_.enable_inter_ddt,
                                 /*search_partition=*/
                                 options_.enable_tx_search,
                                 /*context_rate=*/
                                 pipe_on || options_.tx_context_rate,
                                 options_.trace, options_.enable_rdoq,
                                 ::av2enc::still::qctx::slice_for_base_q_idx(
                                     static_cast<int>(options_.base_q_idx)),
                                 rdoq_sse_scale_at(options_,
                                                   st.current_qindex),
                                 out.res_y, error, luma_mask_row,
                                 pipe_on ? pipe_leaf_weights.data() : nullptr,
                                 eff_slice,
                                 ba_on ? pipe_ba::config().low_band_gain
                                       : (eff_on ? eff_perc::csf_gain() : 0.0),
                                 ba_on ? 0.0
                                       : (eff_on ? eff_perc::csf2_gain() : 0.0))
               : !compute_tx(source_, 0, ax0, ay0, w, h, out.y,
                             st.plane_quant[0], out.res_y, error,
                             qm_enabled_ ? &qm_state_ : nullptr,
                             options_.enable_tcq,
                             ::av2enc::still::qctx::slice_for_base_q_idx(
                                 st.current_qindex),
                             /*is_inter=*/true, options_.enable_rdoq,
                             ::av2enc::still::qctx::slice_for_base_q_idx(
                                 static_cast<int>(options_.base_q_idx)),
                             rdoq_sse_scale_at(options_, st.current_qindex),
                             luma_mask_row)) ||
          (options_.enable_cctx
               ? !compute_chroma_cctx(source_, acx0, acy0, cw, ch, out.u,
                                      out.v, st.quant_step,
                                      /*is_inter=*/true,
                                      out.res_u, out.res_v, &out.cctx, error,
                                      chroma_mask_row)
               : (!compute_tx(source_, 1, acx0, acy0, cw, ch, out.u,
                              st.plane_quant[1], out.res_u, error,
                              qm_enabled_ ? &qm_state_ : nullptr,
                              /*allow_tcq=*/false, /*qctx_slice=*/0,
                              /*is_inter=*/true, /*allow_rdoq=*/false,
                              /*rdoq_qctx=*/0, 0.0, chroma_mask_row) ||
                  !compute_tx(source_, 2, acx0, acy0, cw, ch, out.v,
                              st.plane_quant[2], out.res_v, error,
                              qm_enabled_ ? &qm_state_ : nullptr,
                              /*allow_tcq=*/false, /*qctx_slice=*/0,
                              /*is_inter=*/true, /*allow_rdoq=*/false,
                              /*rdoq_qctx=*/0, 0.0, chroma_mask_row)))) {
        return false;
      }
      if (trace_ && options_.enable_cctx) {
        std::cerr << "  cand inter cctx=" << out.cctx << "\n";
      }
      const double res_total =
          sse_pipe(0, ax0, ay0, w, h, out.res_y.recon) +
          sse_pipe(1, acx0, acy0, cw, ch, out.res_u.recon) +
          sse_pipe(2, acx0, acy0, cw, ch, out.res_v.recon);
      const bool all_zero =
          luma_plan_all_zero(out.res_y) && out.res_u.eob == 0 &&
          out.res_v.eob == 0;
      // AV2ENC_PORT_HORIZON (default off): price the error the skip and the
      // residual arm install in the next K real source pictures, not only the
      // current one.  Same decaying aggregation as the lane-2 driver
      // (w_k = 0.5^k); the prediction IS the skipped reconstruction, so the
      // two arms are measured against the same future frames.  Zero when the
      // gate is off, which leaves every decision below untouched.
      double horizon_skip = 0.0;
      double horizon_res = 0.0;
      if (horizon_port::enabled() && !horizon_future_.empty()) {
        const int horizon = horizon_port::horizon_k();
        horizon_skip = horizon_port::future_distortion(
            horizon_future_, horizon, ax0, ay0, w, h, out.y.data(),
            out.u.data(), out.v.data());
        horizon_res = horizon_port::future_distortion(
            horizon_future_, horizon, ax0, ay0, w, h, out.res_y.recon.data(),
            out.res_u.recon.data(), out.res_v.recon.data());
      }
      const double skip_distortion = pred_total + horizon_skip;
      const double res_distortion = res_total + horizon_res;
      // Reconstruction step is qlookup/8 (66 at QP 90); sse_scale 0.5 is
      // the M1 calibrated value.  --video-leaf-rd swaps in the same
      // QP-derived lambda the intra/inter decision uses.
      const double kLambda =
          options_.enable_leaf_rd
              ? leaf_rd_lambda(st.current_qindex, options_)
              : options_.inter_rd_lambda;
      // Historical bar: 3.0 bits of residual syntax vs a 2.0-bit skip.
      const double hist_res_bits =
          3.0 + luma_plan_bits(out.res_y, true) +
          estimate_tx_bits(out.res_u, false) +
          estimate_tx_bits(out.res_v, false);
      // AV2ENC_SKIP_DRIFT: a skip leaf does not just take this frame's
      // prediction error, it installs that error in the reference and the
      // following frames copy it through subpel MC.  Charge the frozen error
      // in the skip arm of every comparison below.  Off by default; the
      // zero-MV skip is an exact copy, so its factor can be held at 1.
      const bool zero_mv_skip = out.cand.mv.y == 0 && out.cand.mv.x == 0;
      double drift_mv_factor =
          zero_mv_skip ? skip_drift::zero_mv_factor()
                       : skip_drift::nz_mv_factor();
      if (drift_mv_factor == 0.0 && zero_mv_skip && skip_drift::fade_enabled()) {
        // Illumination-ramp evidence: a zero-MV skip on a fade freezes a
        // signed DC bias in the reference.  Static content and sensor noise
        // have a near-zero leaf mean error, a fade does not.
        std::int64_t signed_sum = 0;
        for (int yy = 0; yy < h; ++yy) {
          for (int xx = 0; xx < w; ++xx) {
            const int s = source_sample(source_, 0, ax0 + xx, ay0 + yy);
            signed_sum += s - static_cast<int>(
                                  out.y[static_cast<std::size_t>(yy) *
                                            static_cast<std::size_t>(w) +
                                        static_cast<std::size_t>(xx)]);
          }
        }
        const double bias =
            static_cast<double>(signed_sum) / static_cast<double>(w * h);
        if (std::abs(bias) >= skip_drift::fade_threshold()) {
          drift_mv_factor = 1.0;
        }
      }
      if (drift_mv_factor == 0.0 && zero_mv_skip && illum_active_) {
        // AV2ENC_SKIP_DRIFT_ILLUM: the whole frame is moving under an
        // illumination ramp, so a zero-MV skip freezes the previous frame's
        // illumination even where the leaf's own mean error is below the
        // per-leaf bias bar.
        drift_mv_factor = 1.0;
      }
      if (drift_mv_factor == 0.0 && zero_mv_skip && ref_deficit_active_) {
        // AV2ENC_SKIP_DRIFT_REFDEF: the primary reference has already lost
        // more than the threshold against the key anchor, so the zero-MV skip
        // would freeze chain-added error, not the key's priced error.
        drift_mv_factor = 1.0;
      }
      const double drift_factor =
          skip_drift::enabled() ? skip_drift::alpha() * drift_mv_factor : 0.0;
      const double skip_drift_cost = drift_factor * pred_total;
      const bool hist_want =
          options_.force_inter_residual ||
          (!all_zero &&
           res_distortion + kLambda * hist_res_bits <
               skip_distortion + skip_drift_cost + kLambda * 2.0);
      // R09 bar: skip_flag = 0 pays the coefficient rate, skip_flag = 1 pays
      // the exact -log2 P(skip=1); the difference replaces the fixed 2.0/3.0
      // pair.  Both candidates pay the same is_inter/mode syntax, so it
      // cancels at this comparison.
      double res_syntax = 3.0;
      double skip_syntax = 2.0;
      if (rate_model_on) {
        double delta = 0.0;
        if (rate_probe_.skip_delta(probe_node_for(out.cand), &delta)) {
          res_syntax = 0.0;
          skip_syntax = delta;
        }
      }
      const double res_bits = res_syntax +
                              decision_coeff_bits(out.res_y, true) +
                              decision_coeff_bits(out.res_u, false) +
                              decision_coeff_bits(out.res_v, false);
      bool want =
          options_.force_inter_residual ||
          (!all_zero &&
           res_distortion + kLambda * res_bits <
               skip_distortion + skip_drift_cost + kLambda * skip_syntax);
      if (pipe_on) {
        // Joint skip/residual choice with the pipeline objective.  Both forms
        // pay the exact is_inter/skip/mode syntax; only the residual form
        // adds the context-coded coefficients.  This happens inside every
        // re-scored candidate, not after the candidate has been chosen.
        syntax::BlockSyntax probe = probe_node_for(out.cand);
        probe.use_bawp = out.bawp ? 1 : 0;
        probe.motion_mode = out.interintra ? 1 : 0;
        probe.skip_flag = 1;
        rate_hybrid::Terms skip_terms;
        const bool skip_ok = pipe_probe_.price_full(probe, &skip_terms);
        probe.skip_flag = 0;
        rate_hybrid::Terms res_terms;
        const bool res_ok = pipe_probe_.price_full(probe, &res_terms);
        if (skip_ok && res_ok) {
          const double full_res_bits =
              res_terms.total() +
              coeff_bits(out.res_y, 0, true) +
              coeff_bits(out.res_u, 1, true) +
              coeff_bits(out.res_v, 2, true);
          const double full_skip_bits = skip_terms.total();
          const double skip_cost =
              skip_distortion + skip_drift_cost + pipe_lambda * full_skip_bits;
          const double exact_res_cost =
              res_distortion + pipe_lambda * full_res_bits;
          want = options_.force_inter_residual ||
                 (!all_zero && exact_res_cost < skip_cost);
          if (want != hist_want) {
            ++(eff_on ? eff_perc::counters().skip_flips
                      : pipe_perc::counters().skip_flips);
          }
          out.pipe_modeled = true;
          out.pipe_distortion = want ? res_total : pred_total;
          out.pipe_bits = want ? full_res_bits : full_skip_bits;
        } else {
          ++(eff_on ? eff_perc::counters().fallbacks
                : pipe_perc::counters().fallbacks);
        }
      }
      if (rate_hybrid::enabled() && rate_reporting && want != hist_want) {
        rate_hybrid::report::count_decision("skip");
      }
      if (trace_) {
        std::cerr << "  ileaf " << w << "x" << h << " at " << x0 << ","
                  << y0 << " mv=" << out.cand.mv.y << "," << out.cand.mv.x
                  << " ref=" << static_cast<int>(out.cand.ref_index)
                  << " pred=" << pred_total << " res=" << res_total
                  << " rbits=" << res_bits
                  << " parts=" << out.res_y.parts.size()
                  << " eob=" << out.res_y.eob
                  << " zero=" << (all_zero ? 1 : 0)
                  << " want=" << (want ? 1 : 0);
        for (const TxPlan::Part& part : out.res_y.parts) {
          std::cerr << " [" << part.coded_w << "x" << part.coded_h << "/e"
                    << part.eob << "]";
        }
        std::cerr << "\n";
      }
      if (want) {
        out.residual = true;
        out.y = out.res_y.recon;
        out.u = out.res_u.recon;
        out.v = out.res_v.recon;
        out.luma_sse = sse_vis(0, ax0, ay0, w, h, out.y);
        if (trace_ && options_.enable_inter_ddt) {
          for (const TxPlan::Part& part : out.res_y.parts) {
            const auto part_params = txv::make_tx_params(
                static_cast<txv::TxSize>(part.tx_size),
                static_cast<txv::TxType>(part.tx_type));
            av2enc::video::ddt::Context ddt_ctx;
            ddt_ctx.enable_inter_ddt = true;
            ddt_ctx.is_inter = true;
            if (part_params.has_value() &&
                av2enc::video::ddt::plan_has_ddt(*part_params, ddt_ctx)) {
              std::cerr << "m2 inter ddt residual r=" << r << " c=" << c
                        << " size=" << part.coded_w << "x" << part.coded_h
                        << " type=" << part.tx_type << "\n";
            }
          }
        }
      }
    }
    if (trace_) {
      std::cerr << "  cand inter final sse=" << out.luma_sse
                << " residual=" << (out.residual ? 1 : 0) << "\n";
    }
    // Measured objective terms for AV2ENC_EXACT_RD.  D is the reconstruction
    // distortion of the block the leaf would actually emit (all owned
    // planes); R is the same syntax + coefficient estimate the leaf decision
    // charges, with the mode the candidate ended up carrying after the BAWP
    // carrier swap and the skip-only inter-intra / compound forms.  Gate off
    // does not compute this, so the historical path is untouched.
    if (exact_rd::enabled()) {
      out.distortion = 0.0;
      if (!chroma_part) {
        out.distortion +=
            static_cast<double>(sse_vis(0, ax0, ay0, w, h, out.y));
      }
      if (owns_chroma) {
        out.distortion +=
            static_cast<double>(sse_vis(1, acx0, acy0, cw, ch, out.u));
        out.distortion +=
            static_cast<double>(sse_vis(2, acx0, acy0, cw, ch, out.v));
      }
      out.bits = 2.0 + inter_syntax_bits(out.cand.mode, out.cand.drl_index,
                                         out.cand.mvd.y, out.cand.mvd.x,
                                         out.cand.use_amvd);
      if (rate_model_on) {
        syntax::BlockSyntax probe = probe_node_for(out.cand);
        probe.skip_flag = out.residual ? 0 : 1;
        probe.use_bawp = out.bawp ? 1 : 0;
        probe.motion_mode = out.interintra ? 1 : 0;
        rate_hybrid::Terms terms;
        if (rate_probe_.price_full(probe, &terms)) out.bits = terms.total();
      }
      if (out.bawp && !rate_model_on) out.bits += 1.0;          // use_bawp
      if (out.interintra && !rate_model_on) out.bits += 2.0;    // motion_mode
      if (out.residual) {
        out.bits += decision_coeff_bits(out.res_y, true) +
                    decision_coeff_bits(out.res_u, false) +
                    decision_coeff_bits(out.res_v, false);
      }
    }
    return true;
  };

  // §5.20.7.6 switchable interp_filter selection.  One candidate is evaluated
  // under each of the three TileInterpFilterCdf symbols and the perceptual RD
  // rule picks the winner:
  //   D = plain SSE + w * (perceptual SSE - plain SSE)
  //   J = D + lambda * (syntax/coefficients + filter symbol)
  // `w` is AV2ENC_INTERP_BLOCK_CSF; w = 1 is exactly the eff-perc D the leaf
  // and TX search use, and larger values price low/mid-band error harder.
  // Gate off: one evaluation, filter 0, byte-identical to the historical path.
  constexpr double kInterpFilterSymbolBits = 1.585;  // -log2(1/3)
  const auto evaluate_inter_leaf_filtered =
      [&](const InterCandidate& input, InterLeafResult& out) -> bool {
    out = InterLeafResult{};
    const auto filter_y_mode = [](int mode) {
      return mode == syntax::kInterGlobalMv
                 ? inter::mvprec::kYModeGlobalMv
                 : (mode == syntax::kInterNearMv
                        ? inter::mvprec::kYModeNearMv
                        : inter::mvprec::kYModeNewMv);
    };
    const bool eligible =
        interp_block_enabled() && !input.compound && !input.use_warp &&
        inter::mvprec::needs_interp_filter(w >= 8 && h >= 8,
                                           filter_y_mode(input.mode),
                                           /*motion_mode=*/0);
    if (!eligible) {
      if (!evaluate_inter_leaf(input, out)) return false;
      out.filter = 0;
      return true;
    }
    const double lambda =
        pipe_on ? pipe_lambda
                : (exact_on ? leaf_rd_lambda(st.current_qindex, options_)
                            : inter_lambda);
    const double csf_weight = interp_block_csf_weight();
    const auto filter_cost = [&](const InterLeafResult& result) {
      double plain = 0.0;
      if (!chroma_part) {
        plain += static_cast<double>(sse_vis(0, ax0, ay0, w, h, result.y));
      }
      if (owns_chroma) {
        plain += static_cast<double>(sse_vis(1, acx0, acy0, cw, ch, result.u));
        plain += static_cast<double>(sse_vis(2, acx0, acy0, cw, ch, result.v));
      }
      double perceptual = plain;
      if (pipe_on) {
        perceptual = sse_pipe(0, ax0, ay0, w, h, result.y) +
                     sse_pipe(1, acx0, acy0, cw, ch, result.u) +
                     sse_pipe(2, acx0, acy0, cw, ch, result.v);
      }
      const double distortion = plain + csf_weight * (perceptual - plain);
      double bits = 0.0;
      if (pipe_on && result.pipe_modeled) {
        bits = result.pipe_bits;
      } else if (exact_on) {
        bits = result.bits;
      } else {
        bits = inter_syntax_bits(result.cand.mode, result.cand.drl_index,
                                 result.cand.mvd.y, result.cand.mvd.x,
                                 result.cand.use_amvd) +
               (result.bawp ? 1.0 : 0.0) +
               (result.interintra ? 2.0 : 0.0) +
               (result.residual
                    ? (luma_plan_bits(result.res_y, true) +
                       estimate_tx_bits(result.res_u, false) +
                       estimate_tx_bits(result.res_v, false))
                    : 2.0);
      }
      return distortion + lambda * (bits + kInterpFilterSymbolBits);
    };
    const int saved_override = interp_block_filter_override();
    InterLeafResult chosen;
    double chosen_cost = 0.0;
    bool have_chosen = false;
    for (int filter = 0; filter <= 2; ++filter) {
      interp_block_filter_override() = filter;
      InterLeafResult result;
      if (!evaluate_inter_leaf(input, result)) {
        interp_block_filter_override() = saved_override;
        return false;
      }
      const double cost = filter_cost(result);
      if (!have_chosen || cost < chosen_cost) {
        chosen = std::move(result);
        chosen.filter = filter;
        chosen_cost = cost;
        have_chosen = true;
      }
    }
    interp_block_filter_override() = saved_override;
    if (!have_chosen) {
      error = "interp filter evaluation produced no candidate";
      return false;
    }
    out = std::move(chosen);
    out.filter_coded = true;
    out.pipe_bits += kInterpFilterSymbolBits;
    out.bits += kInterpFilterSymbolBits;
    return true;
  };

  // The gate-off path evaluates exactly the proxy winner.  With the gate on,
  // the cheap ranking is re-scored: the proxy winner first (preserving the
  // historical strict-< tie-break), then the next cheapest candidates in the
  // same order.  N=all would therefore visit a deterministic total order.
  std::vector<const InterCandidate*> exact_eval_order;
  if ((exact_on || pipe_on) && !ranked_candidates.empty()) {
    std::stable_sort(
        ranked_candidates.begin(), ranked_candidates.end(),
        [](const RankedCandidate& a, const RankedCandidate& b) {
          if (a.cost != b.cost) return a.cost < b.cost;
          return a.order < b.order;
        });
    const auto same_candidate = [](const InterCandidate& a,
                                   const InterCandidate& b) {
      return a.mode == b.mode && a.ref_index == b.ref_index &&
             a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
             a.compound == b.compound && a.same_ref == b.same_ref &&
             a.compound_mode == b.compound_mode && a.drl_index == b.drl_index &&
             a.mv1.y == b.mv1.y && a.mv1.x == b.mv1.x &&
             a.use_warp == b.use_warp;
    };
    if (best.valid) exact_eval_order.push_back(&best);
    const std::size_t top_n = static_cast<std::size_t>(
        eff_on ? eff_perc::top_n()
               : (pipe_on ? pipe_perc::top_n() : exact_rd::top_n()));
    for (const RankedCandidate& ranked : ranked_candidates) {
      if (exact_eval_order.size() >= top_n) break;
      bool duplicate = false;
      for (const InterCandidate* already : exact_eval_order) {
        if (same_candidate(*already, ranked.cand)) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) exact_eval_order.push_back(&ranked.cand);
    }
  } else if (best.valid) {
    exact_eval_order.push_back(&best);
  }

  std::vector<std::uint8_t> inter_y;
  std::vector<std::uint8_t> inter_u;
  std::vector<std::uint8_t> inter_v;
  int inter_filter = 0;
  bool inter_residual = false;
  TxPlan res_y;
  TxPlan res_u;
  TxPlan res_v;
  int inter_cctx_type = 0;
  bool bawp_applied = false;
  inter::adv::BawpFit bawp_fit{};
  bool inter_intra_applied = false;
  int inter_intra_mode = 0;
  bool inter_intra_wedge = false;
  int inter_intra_index = 0;
  std::int64_t inter_luma_sse = best.sse;
  // Exact objective of the selected inter candidate.  Only read when the gate
  // is on (the intra/inter comparison below).
  double exact_inter_distortion = 0.0;
  double exact_inter_bits = 0.0;
  // Pipeline objective of the selected inter candidate (D and R are the
  // emitted form, so the intra/inter comparison below uses one J).
  double pipe_inter_distortion = 0.0;
  double pipe_inter_bits = 0.0;
  bool pipe_inter_valid = false;
  // R09 report: the model's candidate ranking vs the historical constants.
  // Both rankings are over the same candidate set; a changed winner is one of
  // the decisions the model moves.
  if (rate_model_on && rate_reporting && best.valid && best_hist.valid) {
    const auto same_candidate = [](const InterCandidate& a,
                                   const InterCandidate& b) {
      return a.mode == b.mode && a.ref_index == b.ref_index &&
             a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
             a.compound == b.compound && a.same_ref == b.same_ref &&
             a.compound_mode == b.compound_mode &&
             a.drl_index == b.drl_index && a.mv1.y == b.mv1.y &&
             a.mv1.x == b.mv1.x && a.use_warp == b.use_warp;
    };
    if (!same_candidate(best, best_hist)) {
      rate_hybrid::report::count_decision("candidate");
    }
  }
  if (!keyframe_ && best.valid && !exact_eval_order.empty()) {
    InterLeafResult chosen;
    double chosen_cost = 0.0;
    bool have_chosen = false;
    // The proxy winner after its own evaluation (it may have swapped to the
    // BAWP carrier).  This is what the historical path would have emitted, so
    // the "exact differs from proxy" counter compares against it rather than
    // against the pre-evaluation candidate.
    InterCandidate proxy_after_eval;
    const double exact_lambda =
        leaf_rd_lambda(st.current_qindex, options_);
    for (const InterCandidate* candidate : exact_eval_order) {
      InterLeafResult result;
      if (!evaluate_inter_leaf_filtered(*candidate, result)) return false;
      if (pipe_on) {
        if (!result.pipe_modeled) {
          // The probe cannot model this carrier (compound / TIP / AMVD /
          // flex precision).  Price the emitted form with the local D and the
          // historical syntax estimate, and count the fallback.
          result.pipe_distortion =
              sse_pipe(0, ax0, ay0, w, h, result.y) +
              sse_pipe(1, acx0, acy0, cw, ch, result.u) +
              sse_pipe(2, acx0, acy0, cw, ch, result.v);
          result.pipe_bits =
              inter_syntax_bits(result.cand.mode, result.cand.drl_index,
                                result.cand.mvd.y, result.cand.mvd.x,
                                result.cand.use_amvd) +
              (result.bawp ? 1.0 : 0.0) +
              (result.interintra ? 2.0 : 0.0) +
              (result.residual
                   ? (luma_plan_bits(result.res_y, true) +
                      estimate_tx_bits(result.res_u, false) +
                      estimate_tx_bits(result.res_v, false))
                   : 2.0) +
              (result.filter_coded ? kInterpFilterSymbolBits : 0.0);
        }
        const double cost =
            result.pipe_distortion + pipe_lambda * result.pipe_bits;
        ++(eff_on ? eff_perc::counters().rescored
                  : pipe_perc::counters().rescored);
        if (!have_chosen) {
          proxy_after_eval = result.cand;
        }
        if (!have_chosen || cost < chosen_cost) {
          chosen = std::move(result);
          chosen_cost = cost;
          have_chosen = true;
        }
      } else if (exact_on) {
        const double cost =
            result.distortion + exact_lambda * result.bits;
        ++exact_rd::counters().rescored;
        if (!have_chosen) {
          proxy_after_eval = result.cand;
          if (result.residual) ++exact_rd::counters().proxy_residual;
        }
        if (!have_chosen || cost < chosen_cost) {
          chosen = std::move(result);
          chosen_cost = cost;
          have_chosen = true;
        }
      } else {
        chosen = std::move(result);
        chosen_cost = 0.0;
        have_chosen = true;
      }
      if (!exact_on && !pipe_on) break;
    }
    if (!have_chosen) {
      error = "exact RD produced no inter candidate";
      return false;
    }
    if (exact_on) {
      ++exact_rd::counters().leaves;
      const auto same_candidate = [](const InterCandidate& a,
                                     const InterCandidate& b) {
        return a.mode == b.mode && a.ref_index == b.ref_index &&
               a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
               a.compound == b.compound && a.same_ref == b.same_ref &&
               a.compound_mode == b.compound_mode &&
               a.drl_index == b.drl_index && a.mv1.y == b.mv1.y &&
               a.mv1.x == b.mv1.x && a.use_warp == b.use_warp;
      };
      if (!same_candidate(chosen.cand, proxy_after_eval)) {
        ++exact_rd::counters().inter_winner_changed;
      }
      if (chosen.residual) ++exact_rd::counters().exact_residual;
      exact_inter_distortion = chosen.distortion;
      exact_inter_bits = chosen.bits;
    }
    if (pipe_on) {
      ++(eff_on ? eff_perc::counters().leaves
                : pipe_perc::counters().leaves);
      const auto same_candidate = [](const InterCandidate& a,
                                     const InterCandidate& b) {
        return a.mode == b.mode && a.ref_index == b.ref_index &&
               a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
               a.compound == b.compound && a.same_ref == b.same_ref &&
               a.compound_mode == b.compound_mode &&
               a.drl_index == b.drl_index && a.mv1.y == b.mv1.y &&
               a.mv1.x == b.mv1.x && a.use_warp == b.use_warp;
      };
      if (!same_candidate(chosen.cand, proxy_after_eval)) {
        ++(eff_on ? eff_perc::counters().inter_winner_changed
                  : pipe_perc::counters().inter_winner_changed);
      }
      if (chosen.residual) {
        ++(eff_on ? eff_perc::counters().residual_leaves
                  : pipe_perc::counters().residual_leaves);
      }
      pipe_inter_distortion = chosen.pipe_distortion;
      pipe_inter_bits = chosen.pipe_bits;
      pipe_inter_valid = true;
    }
    best = chosen.cand;
    inter_y = std::move(chosen.y);
    inter_u = std::move(chosen.u);
    inter_v = std::move(chosen.v);
    inter_filter = chosen.filter;
    inter_residual = chosen.residual;
    res_y = std::move(chosen.res_y);
    res_u = std::move(chosen.res_u);
    res_v = std::move(chosen.res_v);
    inter_cctx_type = chosen.cctx;
    bawp_applied = chosen.bawp;
    bawp_fit = chosen.fit;
    inter_intra_applied = chosen.interintra;
    inter_intra_mode = chosen.ii_mode;
    inter_intra_wedge = chosen.ii_wedge;
    inter_intra_index = chosen.ii_index;
    inter_luma_sse = chosen.luma_sse;
  }

  bool use_intra = keyframe_ || options_.force_intra || !best.valid;
  if (seg_forced_inter && best.valid && !keyframe_) use_intra = false;
  // §5.20.5.3/.4 IBC + §7.13.3.26 morph on the intra keyframe.  The copy is
  // compared with the ordinary intra leaf (or a winning palette leaf) by
  // luma SSE; a win selects the IBC skip block (no residual) and, when it
  // helps, morph_pred = 1.
  IbcChoice ibc_choice;
  bool use_ibc = false;
  if (keyframe_ && ibc_enabled_ && have_intra && !shared_chroma &&
      w >= 8 && h >= 8) {
    const std::int64_t intra_distortion =
        use_palette ? palette_choice.sse : intra_sse;
    if (!choose_ibc_leaf(r, c, static_cast<int>(node.size),
                         intra_distortion, ibc_choice, error)) {
      return false;
    }
    use_ibc = ibc_choice.found;
  }
  if (use_ibc) {
    use_intra = false;
    use_palette = false;
  }
  if (!use_intra && have_intra && !seg_forced_inter) {
    if (pipe_on && pipe_inter_valid) {
      // Joint intra/inter comparison with the pipeline objective.  The intra
      // candidate pays its own is_inter symbol, mode/MRL/DIP/CfL syntax and
      // the context-coded coefficients; the inter side's terms were recorded
      // by evaluate_inter_leaf() for the emitted skip or residual form.
      syntax::BlockSyntax intra_probe = probe_node_for(best);
      intra_probe.is_inter = 0;
      intra_probe.skip_flag = 0;
      double intra_is_inter = 2.0;
      double unused_skip = 0.0;
      if (!pipe_probe_.price_leaf_flags(intra_probe, &intra_is_inter,
                                        &unused_skip)) {
        intra_is_inter = 2.0;
        ++(eff_on ? eff_perc::counters().fallbacks
                : pipe_perc::counters().fallbacks);
      }
      const double intra_bits =
          intra_is_inter +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx) +
          coeff_bits(intra_y, 0, false) +
          coeff_bits(intra_u, 1, false) +
          coeff_bits(intra_v, 2, false);
      double intra_distortion = 0.0;
      if (!chroma_part) {
        intra_distortion += sse_pipe(0, ax0, ay0, w, h, intra_y.recon);
      }
      if (owns_chroma) {
        intra_distortion += sse_pipe(1, acx0, acy0, cw, ch, intra_u.recon);
        intra_distortion += sse_pipe(2, acx0, acy0, cw, ch, intra_v.recon);
      }
      use_intra =
          intra_distortion + pipe_lambda * intra_bits <
          pipe_inter_distortion + pipe_lambda * pipe_inter_bits;
      if (trace_) {
        std::cerr << "  pipe leafrd intra=" << intra_distortion << "+"
                  << pipe_lambda << "*" << intra_bits << " inter="
                  << pipe_inter_distortion << "+" << pipe_lambda << "*"
                  << pipe_inter_bits << " use_intra="
                  << (use_intra ? 1 : 0) << "\n";
      }
      {
        // Decision-change count against the historical constants computed on
        // exactly the same intra and inter plans.
        const double lambda = leaf_rd_lambda(st.current_qindex, options_);
        const double hist_intra_bits =
            2.0 + luma_plan_bits(intra_y, false) +
            estimate_tx_bits(intra_u, false) +
            estimate_tx_bits(intra_v, false) +
            intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                   intra_mrl_index, intra_mrl_sec_index,
                                   intra_use_dip) +
            intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
        double hist_inter_bits =
            2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                    best.mvd.x, best.use_amvd) +
            (bawp_applied ? 1.0 : 0.0) +
            (inter_intra_applied ? 2.0 : 0.0);
        if (inter_residual) {
          hist_inter_bits += luma_plan_bits(res_y, true) +
                             estimate_tx_bits(res_u, false) +
                             estimate_tx_bits(res_v, false);
        }
        const bool use_intra_hist =
            static_cast<double>(intra_sse) + lambda * hist_intra_bits <
            static_cast<double>(inter_luma_sse) +
                lambda * hist_inter_bits;
        if (use_intra != use_intra_hist) {
          ++(eff_on ? eff_perc::counters().intra_flips
                    : pipe_perc::counters().intra_flips);
        }
      }
    } else if (exact_on) {
      // Exact intra/inter comparison: both sides are priced with the same
      // QP-derived leaf lambda, the reconstruction distortion of every plane
      // the leaf owns and the same syntax + coefficient estimate the leaf
      // decision charges.  The inter side's terms were recorded by
      // evaluate_inter_leaf(), including the residual plan it chose.
      const double lambda = leaf_rd_lambda(st.current_qindex, options_);
      double intra_distortion = 0.0;
      if (!chroma_part) {
        intra_distortion +=
            static_cast<double>(sse_vis(0, ax0, ay0, w, h, intra_y.recon));
      }
      if (owns_chroma) {
        intra_distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, intra_u.recon));
        intra_distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, intra_v.recon));
      }
      const double intra_bits =
          2.0 + decision_coeff_bits(intra_y, false) +
          decision_coeff_bits(intra_u, false) +
          decision_coeff_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
      use_intra =
          intra_distortion + lambda * intra_bits <
          exact_inter_distortion + lambda * exact_inter_bits;
      if (use_intra) ++exact_rd::counters().intra_flipped;
      if (trace_) {
        std::cerr << "  leafrd exact intra=" << intra_distortion << "+"
                  << lambda << "*" << intra_bits << " inter="
                  << exact_inter_distortion << "+" << lambda << "*"
                  << exact_inter_bits << " use_intra="
                  << (use_intra ? 1 : 0) << "\n";
      }
    } else if (options_.enable_leaf_rd) {
      // One leaf cost for the intra/inter decision: coded SSE plus one
      // QP-derived lambda times the real-ish syntax rate.  Both sides pay
      // the same is_inter/skip base; each adds its own mode syntax and its
      // residual estimate, so a mode that is cheap to signal can win even
      // when its prediction SSE is slightly worse.
      const double lambda = leaf_rd_lambda(st.current_qindex, options_);
      double intra_bits =
          2.0 + decision_coeff_bits(intra_y, false) +
          decision_coeff_bits(intra_u, false) +
          decision_coeff_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
      double inter_bits =
          2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                  best.mvd.x, best.use_amvd);
      if (inter_residual) {
        inter_bits += decision_coeff_bits(res_y, true) +
                      decision_coeff_bits(res_u, false) +
                      decision_coeff_bits(res_v, false);
      }
      const double hist_intra_bits =
          2.0 + luma_plan_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
      double hist_inter_bits =
          2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                  best.mvd.x, best.use_amvd);
      if (inter_residual) {
        hist_inter_bits += luma_plan_bits(res_y, true) +
                           estimate_tx_bits(res_u, false) +
                           estimate_tx_bits(res_v, false);
      }
      const bool intra_hist =
          static_cast<double>(intra_sse) + lambda * hist_intra_bits <
          static_cast<double>(inter_luma_sse) + lambda * hist_inter_bits;
      if (rate_model_on) {
        // is_inter is coded on both sides: the inter leaf pays -log2 P(1)
        // (plus skip), the intra leaf -log2 P(0).  The mode information and
        // the residual then use the exact probe / Laplacian.
        syntax::BlockSyntax probe = probe_node_for(best);
        probe.skip_flag = inter_residual ? 0 : 1;
        probe.use_bawp = bawp_applied ? 1 : 0;
        probe.motion_mode = inter_intra_applied ? 1 : 0;
        rate_hybrid::Terms inter_terms;
        if (rate_probe_.price_full(probe, &inter_terms)) {
          inter_bits = inter_terms.total();
          if (inter_residual) {
            inter_bits += decision_coeff_bits(res_y, true) +
                          decision_coeff_bits(res_u, false) +
                          decision_coeff_bits(res_v, false);
          }
        }
        double intra_is_inter = 2.0;
        double unused_skip = 0.0;
        probe.is_inter = 0;
        if (rate_probe_.price_leaf_flags(probe, &intra_is_inter,
                                         &unused_skip)) {
          intra_bits = intra_is_inter + decision_coeff_bits(intra_y, false) +
                       decision_coeff_bits(intra_u, false) +
                       decision_coeff_bits(intra_v, false) +
                       intra_luma_syntax_bits(
                           intra_y_mode, intra_y_angle_delta, intra_mrl_index,
                           intra_mrl_sec_index, intra_use_dip) +
                       intra_chroma_syntax_bits(intra_is_cfl,
                                                options_.enable_cctx);
        }
      }
      use_intra = static_cast<double>(intra_sse) + lambda * intra_bits <
                  static_cast<double>(inter_luma_sse) + lambda * inter_bits;
      if (rate_hybrid::enabled() && rate_reporting &&
          use_intra != intra_hist) {
        rate_hybrid::report::count_decision("intra");
      }
      if (trace_) {
        std::cerr << "  leafrd intra=" << intra_sse << "+" << lambda << "*"
                  << intra_bits << " inter=" << inter_luma_sse << "+"
                  << lambda << "*" << inter_bits << " use_intra="
                  << (use_intra ? 1 : 0) << "\n";
      }
    } else if (options_.enable_inter_residual) {
      // Rate-aware decision on the opt-in residual path.  The historical
      // path compared raw SSE only, so a fine quantiser made intra win even
      // when it cost two orders of magnitude more bits.
      const double lambda = options_.inter_rd_lambda;
      const double intra_bits =
          3.0 + luma_plan_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false);
      double inter_bits = 8.0;
      if (inter_residual) {
        inter_bits += 3.0 + luma_plan_bits(res_y, true) +
                      estimate_tx_bits(res_u, false) +
                      estimate_tx_bits(res_v, false);
      }
      use_intra = static_cast<double>(intra_sse) + lambda * intra_bits <
                  static_cast<double>(inter_luma_sse) + lambda * inter_bits;
    } else {
      use_intra = intra_sse < inter_luma_sse;
    }
  }

  if (use_ibc) {
    entry.is_inter = 0;
    entry.skip_flag = 0;
    entry.y_mode = 0;  // DC_PRED (read_intrabc_info forces it)
    entry.flags |= syntax::kFlagIsIbc;
    entry.ibc_skip = 1;
    entry.ibc_mode = ibc_choice.stack_index >= 0 ? 1 : 0;
    entry.ibc_ref_mv_idx =
        ibc_choice.stack_index >= 0
            ? static_cast<std::uint8_t>(ibc_choice.stack_index)
            : 0;
    entry.ibc_bv_row = ibc_choice.bv.row;
    entry.ibc_bv_col = ibc_choice.bv.col;
    entry.morph_pred = ibc_choice.morph ? 1 : 0;
    write_plane_block(recon_[0], x0, y0, w, h, ibc_choice.luma);
    write_plane_block(recon_[1], cx0, cy0, cw, ch, ibc_choice.u);
    write_plane_block(recon_[2], cx0, cy0, cw, ch, ibc_choice.v);
    update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
    set_intra_joint(r, c, w4, h4, 0);
    ++st.intra_blocks;
    if (trace_) {
      std::cerr << "m2 ibc r=" << r << " c=" << c << " " << w << "x" << h
                << " bv=(" << ibc_choice.bv.row << "," << ibc_choice.bv.col
                << ") mode=" << static_cast<int>(entry.ibc_mode)
                << " morph=" << static_cast<int>(entry.morph_pred)
                << " sse=" << ibc_choice.sse << "\n";
    }
  } else if (use_palette) {
    if (!have_intra) {
      error = "palette fallback unavailable";
      return false;
    }
    // §5.20.8: palette is luma only and forces YMode = DC_PRED.  The block
    // codes eob=0 on all three planes, so the reconstruction is exactly the
    // palette prediction plus the ordinary (recomputed) chroma path.  CfL is
    // not signalled: its derived alpha would read the palette-predicted
    // luma, so recompute chroma DC from the palette reconstruction.
    entry.is_inter = 0;
    entry.skip_flag = 0;
    entry.y_mode = 0;
    entry.y_angle_delta = 0;
    entry.uv_mode = 0;
    entry.palette_index = static_cast<std::uint32_t>(sink.palettes.size());
    sink.palettes.push_back(std::move(palette_choice.result));
    // Update the encoder-side §5.20.8.2 cache source only now: an IBC leaf
    // also seen this block would have cleared the palette decision.
    palette::store_palette_state(palette_state_, r, c, w, h,
                                 sink.palettes.back());
    const std::optional<txv::TxParams> params_opt =
        txv::make_tx_params(w, h, txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error = "unsupported palette block transform size";
      return false;
    }
    intra_y = TxPlan{};
    intra_y.tx_size = params_opt->tx_size;
    intra_y.adj_w = params_opt->w;
    intra_y.adj_h = params_opt->h;
    intra_y.eob = 0;
    intra_y.recon = palette_choice.luma;
    write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
    if (!intra_dc_predict(recon_[1], source_, 1, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_u_pred) ||
        !intra_dc_predict(recon_[2], source_, 2, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_v_pred) ||
        !compute_tx(source_, 1, acx0, acy0, cw, ch, intra_u_pred,
                    st.plane_quant[1], intra_u, error,
                    qm_enabled_ ? &qm_state_ : nullptr) ||
        !compute_tx(source_, 2, acx0, acy0, cw, ch, intra_v_pred,
                    st.plane_quant[2], intra_v, error,
                    qm_enabled_ ? &qm_state_ : nullptr)) {
      error = "palette chroma prediction failed";
      return false;
    }
    write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_u.recon);
    write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_v.recon);
    entry.levels_offset = static_cast<std::uint32_t>(sink.levels.size());
    write_luma_plan(entry, intra_y, sink.levels);
    {
      const TxPlan* plans[2] = {&intra_u, &intra_v};
      for (int plane = 1; plane < 3; ++plane) {
        entry.tx_size[plane] =
            static_cast<std::uint8_t>(plans[plane - 1]->tx_size);
        entry.eob[plane] = static_cast<std::uint32_t>(plans[plane - 1]->eob);
        const int area = plans[plane - 1]->adj_w * plans[plane - 1]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane - 1]->adj_w, plans[plane - 1]->adj_h,
                           scan);
        for (int c2 = 0; c2 < plans[plane - 1]->eob; ++c2) {
          sink.levels.push_back(plans[plane - 1]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
    }
    update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
    set_intra_joint(r, c, w4, h4, 0);
    ++st.intra_blocks;
    if (trace_) {
      std::cerr << "m2 palette r=" << r << " c=" << c << " " << w << "x" << h
                << " size=" << sink.palettes.back().size
                << " direction=" << sink.palettes.back().direction
                << " sse=" << palette_choice.sse << "\n";
    }
  } else if (use_intra) {
    if (!have_intra) {
      error = "intra fallback unavailable";
      return false;
    }
    entry.is_inter = 0;
    entry.skip_flag = 0;
    entry.y_mode = chroma_part ? 0 : intra_y_mode;
    entry.y_angle_delta = intra_y_angle_delta;
    entry.mrl_index = intra_mrl_index;
    entry.mrl_sec_index = intra_mrl_sec_index ? 1 : 0;
    entry.use_dip = intra_use_dip ? 1 : 0;
    entry.dip_transpose = intra_dip_transpose ? 1 : 0;
    entry.dip_mode = static_cast<std::uint8_t>(intra_dip_mode);
    entry.uv_mode = 0;  // DC_PRED (also the CfL base mode)
    if (intra_is_cfl) {
      entry.flags |= syntax::kFlagIsCfl;
      entry.cfl_index = static_cast<std::uint8_t>(intra_cfl_index);
      entry.cfl_mh_dir = static_cast<std::uint8_t>(intra_cfl_mh_dir);
    }
    entry.levels_offset = static_cast<std::uint32_t>(sink.levels.size());
    if (!chroma_part) {
      write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
      write_luma_plan(entry, intra_y, sink.levels);
    }
    entry.cctx_type = static_cast<std::uint8_t>(intra_cctx_type);
    if (owns_chroma) {
      write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_u.recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_v.recon);
      const TxPlan* plans[2] = {&intra_u, &intra_v};
      for (int plane = 1; plane < 3; ++plane) {
        entry.tx_size[plane] =
            static_cast<std::uint8_t>(plans[plane - 1]->tx_size);
        entry.eob[plane] = static_cast<std::uint32_t>(plans[plane - 1]->eob);
        const int area = plans[plane - 1]->adj_w * plans[plane - 1]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane - 1]->adj_w, plans[plane - 1]->adj_h,
                           scan);
        for (int c2 = 0; c2 < plans[plane - 1]->eob; ++c2) {
          sink.levels.push_back(plans[plane - 1]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
    }
    if (!chroma_part) {
      update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
      set_intra_joint(r, c, w4, h4,
                      syntax::intra_joint_mode(intra_y_mode,
                                               intra_y_angle_delta));
    }
    ++st.intra_blocks;
    if (trace_) {
      int trace_ist_sec = intra_y.sec_tx_type;
      int trace_ist_set = intra_y.stx_set;
      if (intra_y.searched && intra_y.parts.size() == 1) {
        trace_ist_sec = intra_y.parts[0].sec_tx_type;
        trace_ist_set = intra_y.parts[0].stx_set;
      }
      std::cerr << "m2 intra r=" << r << " c=" << c << " " << w << "x" << h
                << " mode=" << static_cast<int>(intra_y_mode)
                << " delta=" << static_cast<int>(intra_y_angle_delta)
                << " mrl=" << static_cast<int>(intra_mrl_index)
                << " sec=" << (intra_mrl_sec_index ? 1 : 0)
                << (intra_use_dip
                        ? " dip=" + std::to_string(intra_dip_mode) + "/" +
                              std::to_string(intra_dip_transpose ? 1 : 0)
                        : std::string());
      if (options_.enable_ibp && !chroma_part) {
        // §5.4.5/§7.13.2: report the decision the decoder derives from the
        // signalled (mode, angle_delta).  The gate greps ibp_blend=1 to
        // prove the two-angle path is live.
        ibp::DecisionParams dp;
        dp.enable_ibp = true;
        dp.mode = static_cast<intra::Mode>(intra_y_mode);
        dp.angle_delta = intra_y_angle_delta;
        dp.w = w;
        dp.h = h;
        dp.plane = 0;
        dp.mrl_index = intra_mrl_index;
        const ibp::Decision dec = ibp::decide(dp);
        std::cerr << " ibp_apply=" << (dec.apply_ibp_edge_rules ? 1 : 0)
                  << " ibp_blend=" << (dec.use_ibp ? 1 : 0);
      }
      std::cerr << " ist=" << trace_ist_sec << "/" << trace_ist_set
                << " sse=" << intra_sse << "\n";
    }
  } else {
    entry.is_inter = 1;
    entry.skip_flag = inter_residual ? 0 : 1;
    if (best.from_temporal) ++st.temporal_selected;
    if (best.from_refmvbank) ++st.refmvbank_selected;
    if (best.same_ref) ++st.same_ref_blocks;
    if (best.use_amvd) ++st.amvd_selected;
    if (best.block_mv_precision >= 0) ++st.flex_precision_selected;
    if (best.mvd_sign_derived) ++st.derived_sign_selected;
    entry.inter_mode = static_cast<std::uint8_t>(best.mode);
    entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
    entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
    entry.compound = best.compound ? 1 : 0;
    entry.ref_index1 = static_cast<std::uint8_t>(
        best.compound ? best.ref_index1 : 1);
    entry.compound_joint = 0;
    entry.compound_mode = static_cast<std::uint8_t>(best.compound_mode);
    entry.drl_index1 = static_cast<std::uint8_t>(best.drl_index1);
    if (best.compound) {
      // A compound leaf never carries a residual in the gated subset.
      entry.skip_flag = 1;
      inter_residual = false;
      entry.compound_masked =
          best.compound_mode == 0 ? 0 : (best.compound_type != 2 ? 1 : 0);
      // §5.20.7.16: without comp_group_idx = 1 there is no compound_type
      // symbol, so the AVERAGE sentinel is 0 for the writer.
      entry.compound_type = static_cast<std::uint8_t>(
          entry.compound_masked ? best.compound_type : 0);
      entry.wedge_index = static_cast<std::uint8_t>(best.wedge_index);
      entry.wedge_sign = static_cast<std::uint8_t>(best.wedge_sign);
      entry.mask_type = static_cast<std::uint8_t>(best.mask_type);
      entry.cwp_index = static_cast<std::uint8_t>(best.cwp_index);
      entry.cwp_weight = static_cast<std::int8_t>(best.cwp_weight);
      entry.use_optflow = best.use_optflow ? 1 : 0;
      entry.use_refinemv = best.use_refinemv ? 1 : 0;
      if (options_.enable_df_sub_pu &&
          (best.use_optflow || best.use_refinemv)) {
        collect_sub_pus(sink, x0, y0, w, h, best.use_optflow,
                        best.use_refinemv);
      }
    }
    if (best.compound && best.compound_mode == 0) {
      entry.mv_y = best.mv.y;
      entry.mv_x = best.mv.x;
      entry.mv1_y = best.mv1.y;
      entry.mv1_x = best.mv1.x;
    } else {
      entry.mv_y = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.y;
      entry.mv_x = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.x;
      entry.mv1_y = 0;
      entry.mv1_x = 0;
    }
    entry.mvd_y = best.mvd.y;
    entry.mvd_x = best.mvd.x;
    entry.use_amvd = best.use_amvd ? 1 : 0;
    entry.block_mv_precision =
        static_cast<std::int8_t>(best.block_mv_precision);
    entry.mvd_sign_derived = best.mvd_sign_derived ? 1 : 0;
    // §5.20.7.6 switchable interp_filter carried on the inter leaf.  The
    // writer codes it where needs_interp_filter() is true; for every other
    // carrier (compound, warp, large GLOBALMV) it is 0 = EIGHTTAP and no
    // symbol is coded.  The intra-only mrl fields are unused here.
    entry.mrl_index = static_cast<std::uint8_t>(inter_filter);
    entry.use_bawp = bawp_applied ? 1 : 0;
    entry.bawp_alpha = static_cast<std::int16_t>(bawp_fit.alpha);
    if (inter_intra_applied) {
      // §5.20.7.14 INTERINTRA: the block is skip-only, RefFrame[1] is
      // INTRA_FRAME and wedge_sign is forced 0.
      entry.skip_flag = 1;
      inter_residual = false;
      entry.motion_mode = 1;
      entry.interintra_mode = static_cast<std::uint8_t>(inter_intra_mode);
      entry.wedge_interintra = inter_intra_wedge ? 1 : 0;
      entry.wedge_index = static_cast<std::uint8_t>(inter_intra_index);
    }
    write_plane_block(recon_[0], x0, y0, w, h, inter_y);
    write_plane_block(recon_[1], cx0, cy0, cw, ch, inter_u);
    write_plane_block(recon_[2], cx0, cy0, cw, ch, inter_v);
    if (inter_residual) {
      entry.levels_offset = static_cast<std::uint32_t>(sink.levels.size());
      write_luma_plan(entry, res_y, sink.levels);
      entry.cctx_type = static_cast<std::uint8_t>(inter_cctx_type);
      const TxPlan* plans[2] = {&res_u, &res_v};
      for (int plane = 1; plane < 3; ++plane) {
        entry.tx_size[plane] =
            static_cast<std::uint8_t>(plans[plane - 1]->tx_size);
        entry.eob[plane] =
            static_cast<std::uint32_t>(plans[plane - 1]->eob);
        const int area = plans[plane - 1]->adj_w * plans[plane - 1]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane - 1]->adj_w, plans[plane - 1]->adj_h,
                           scan);
        for (int c2 = 0; c2 < plans[plane - 1]->eob; ++c2) {
          sink.levels.push_back(plans[plane - 1]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
    }
    update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv,
                   best.compound ? best.ref_index1 : merge::kRefNone,
                   best.compound ? &best.mv1 : nullptr,
                   /*warp0=*/best.use_warp, /*warp1=*/false,
                   /*is_wedge=*/best.compound && best.compound_type == 0,
                   best.wedge_index, best.wedge_sign,
                   static_cast<int>(node.size));
    set_intra_joint(r, c, w4, h4, 0);  // inter_block_mode_info(): DC
    ++st.inter_blocks;
    if (trace_) {
      std::cerr << "m2 inter r=" << r << " c=" << c << " " << w << "x" << h
                << " mode=" << static_cast<int>(best.mode)
                << " drl=" << best.drl_index
                << " ref=" << best.ref_index << " mv=(" << best.mv.y << ","
                << best.mv.x << ") mv1=(" << best.mv1.y << ","
                << best.mv1.x << ") sse=" << inter_luma_sse
                << " skip=" << static_cast<int>(entry.skip_flag)
                << " comp=" << static_cast<int>(entry.compound)
                << " cm=" << static_cast<int>(entry.compound_mode)
                << " ifilter=" << static_cast<int>(entry.mrl_index)
                << " optflow=" << static_cast<int>(entry.use_optflow)
                << " refinemv=" << static_cast<int>(entry.use_refinemv)
                << " ct=" << static_cast<int>(entry.compound_type)
                << " wedge=" << static_cast<int>(entry.wedge_index)
                << " sign=" << static_cast<int>(entry.wedge_sign)
                << " mask=" << static_cast<int>(entry.mask_type) << "\n";
    }
  }

#if defined(AV2ENC_VIDEO_DELTAQ)
  if (qp_tools_active()) {
    // §5.20.7.2: a skipped inter leaf codes no segment id when the frame has
    // no lossless segment; the decoder infers the §5.20.7.4 neighbour
    // prediction.  Replay it here so the deblock metadata uses the same
    // segment id (and therefore the same effective qindex) the decoder will.
    int segment_id = seg_enabled_ ? st.current_segment_id : 0;
    // A SEG_LVL_GLOBALMV segment cannot hold an intra leaf: §5.20.7.3
    // forces is_inter = 1 when the feature is active.  If no inter candidate
    // was available, fall back to the ALT_Q-free baseline segment (the
    // table gives every feature segment ALT_Q 0 in that composition).
    if (seg_enabled_ && !keyframe_ && entry.is_inter == 0 &&
        ((globalmv_seg_id_ >= 0 && segment_id == globalmv_seg_id_) ||
         (skip_seg_id_ >= 0 && segment_id == skip_seg_id_))) {
      segment_id = 0;
    }
    bool inferred = false;
    if (seg_enabled_ && !keyframe_ && entry.is_inter != 0 &&
        entry.skip_flag != 0 && !enc_seg_.has_lossless_segment &&
        !enc_seg_.seg_id_pre_skip) {
      segment_id = deltaq::segment_prediction(enc_seg_grid_, r, c);
      inferred = true;
    }
    int qindex = st.current_qindex;
    if (seg_enabled_) {
      std::string qp_error;
      if (!deltaq::effective_qindex(
              enc_seg_, base_q_index(), st.current_qindex, delta_q_enabled_,
              segment_id, /*ignore_delta_q=*/false, qindex, &qp_error)) {
        error = "get_qindex: " + qp_error;
        return false;
      }
    }
    if (qindex < 0 || qindex > 255) {
      error = "get_qindex produced an out-of-range quantizer";
      return false;
    }
    entry.segment_id = static_cast<std::uint8_t>(segment_id);
    entry.qindex = static_cast<std::int16_t>(qindex);
    if (trace_ && inferred) {
      std::cerr << "segq inferred r=" << r << " c=" << c
                << " segment=" << segment_id << " qindex=" << qindex
                << "\n";
    }
    if (seg_enabled_) {
      deltaq::SegmentBlock segment_block;
      segment_block.mi_row = r;
      segment_block.mi_col = c;
      segment_block.bw4 = w4;
      segment_block.bh4 = h4;
      std::string seg_error;
      if (!deltaq::commit_segment_block(enc_seg_grid_, segment_block,
                                        segment_id, &seg_error)) {
        error = "segment map: " + seg_error;
        return false;
      }
    }
  }
#endif

  // AV2ENC_EXACT_RD partition input: record the measured (D, R) of the leaf
  // the encoder will actually emit.  D is the reconstruction SSE of every
  // plane the leaf owns; R is the same syntax + coefficient estimate the leaf
  // decision charges.  The partition refinement later prices these with the
  // node's own partition lambda and compares them with the NEWMV probe on
  // nodes that were not encoded in the first pass.  Gate off: no record, no
  // extra work.
  if (exact_rd::enabled()) {
    exact_rd::LeafCost measured;
    if (use_ibc) {
      if (!chroma_part) {
        measured.distortion += static_cast<double>(
            sse_vis(0, ax0, ay0, w, h, ibc_choice.luma));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, ibc_choice.u));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, ibc_choice.v));
      }
      // is_inter/is_ibc + mode/BV syntax: coarse but consistent with the
      // probe's constant mode/skip terms.  IBC is keyframe-only, so this never
      // enters the P-frame partition refinement.
      measured.bits = 8.0;
    } else if (use_palette) {
      if (!chroma_part) {
        measured.distortion += static_cast<double>(
            sse_vis(0, ax0, ay0, w, h, palette_choice.luma));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, intra_u.recon));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, intra_v.recon));
      }
      measured.bits =
          2.0 + 4.0 + estimate_tx_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
    } else if (use_intra) {
      if (!chroma_part) {
        measured.distortion += static_cast<double>(
            sse_vis(0, ax0, ay0, w, h, intra_y.recon));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, intra_u.recon));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, intra_v.recon));
      }
      measured.bits =
          2.0 + luma_plan_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
    } else {
      if (!chroma_part) {
        measured.distortion +=
            static_cast<double>(sse_vis(0, ax0, ay0, w, h, inter_y));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, inter_u));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, inter_v));
      }
      measured.bits =
          2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                  best.mvd.x, best.use_amvd);
      if (bawp_applied) measured.bits += 1.0;
      if (inter_intra_applied) measured.bits += 2.0;
      if (inter_residual) {
        measured.bits += luma_plan_bits(res_y, true) +
                         estimate_tx_bits(res_u, false) +
                         estimate_tx_bits(res_v, false);
      }
    }
    measured.valid = true;
    if (sink.leaf_costs.size() < sink.blocks.size()) {
      sink.leaf_costs.resize(sink.blocks.size());
    }
    sink.leaf_costs[index] = measured;
  }

  // R09 (AV2ENC_RATE_HYBRID_S): record the decision-side rate for this leaf
  // and advance the probe.  Measurement runs (gate off, measure on) also come
  // here, so the report covers the historical decisions byte-identically.
  if (rate_reporting) {
    syntax::BlockSyntax probe = entry;
    rate_hybrid::Terms terms;
    const bool priced = rate_probe_.price_full(probe, &terms);
    if (entry.is_inter) {
      syntax::BlockSyntax::RateDiag& diag = entry.rate_diag;
      double old_syntax =
          2.0 + inter_syntax_bits(entry.inter_mode, entry.drl_index,
                                  entry.mvd_y, entry.mvd_x,
                                  entry.use_amvd != 0);
      if (entry.use_bawp) old_syntax += 1.0;
      if (entry.motion_mode == 1) old_syntax += 2.0;
      diag.syntax_old = static_cast<float>(old_syntax);
      diag.syntax_new =
          static_cast<float>(priced ? terms.total() : old_syntax);
      diag.term[0] = static_cast<float>(terms.is_inter);
      diag.term[1] = static_cast<float>(terms.skip);
      diag.term[2] = static_cast<float>(terms.mode);
      diag.term[3] = static_cast<float>(terms.ref + terms.comp_mode);
      diag.term[4] = static_cast<float>(terms.drl);
      diag.term[5] = static_cast<float>(terms.mvd);
      diag.bawp = static_cast<float>(terms.bawp);
      if (entry.skip_flag == 0) {
        float eob_sum = 0.0f;
        float nz_sum = 0.0f;
        float abs_sum = 0.0f;
        const auto accumulate = [&](const TxPlan& plan) {
          if (!plan.searched) {
            eob_sum += static_cast<float>(plan.eob);
            std::vector<std::uint16_t> scan(
                static_cast<std::size_t>(plan.adj_w * plan.adj_h));
            syntax::build_scan(plan.adj_w, plan.adj_h, scan);
            for (int c2 = 0; c2 < plan.eob; ++c2) {
              const std::int32_t v =
                  plan.levels[scan[static_cast<std::size_t>(c2)]];
              if (v != 0) {
                nz_sum += 1.0f;
                abs_sum += static_cast<float>(v < 0 ? -v : v);
              }
            }
            return;
          }
          for (const TxPlan::Part& part : plan.parts) {
            eob_sum += static_cast<float>(part.eob);
            for (int c2 = 0; c2 < part.eob; ++c2) {
              const std::int32_t v = part.levels[static_cast<std::size_t>(c2)];
              if (v != 0) {
                nz_sum += 1.0f;
                abs_sum += static_cast<float>(v < 0 ? -v : v);
              }
            }
          }
        };
        accumulate(res_y);
        accumulate(res_u);
        accumulate(res_v);
        diag.eob = eob_sum;
        diag.nonzero = nz_sum;
        diag.sum_abs = abs_sum;
        diag.coeff_old =
            static_cast<float>(luma_plan_bits(res_y, true) +
                               estimate_tx_bits(res_u, false) +
                               estimate_tx_bits(res_v, false));
        diag.coeff_new =
            static_cast<float>(laplacian_plan_bits(res_y, true) +
                               laplacian_plan_bits(res_u, false) +
                               laplacian_plan_bits(res_v, false));
      }
      diag.valid = 1;
      diag.fallback = priced ? 0 : 1;
      if (!priced) rate_hybrid::report::count_fallback();
    }
    rate_probe_.commit(entry);
  }

  // AV2ENC_PIPE_PERC: advance the exact-syntax probe with the leaf the writer
  // will code, in coded order, so the next leaf's contexts match the writer.
  if (pipe_on && !chroma_part) pipe_probe_.commit(entry);

  // §6.19.2.3: this leaf's cells are now decoded for the availability of the
  // following leaves in this superblock.
  if (!chroma_part) st.decoded[0].fill(c, r, w4, h4);
  if (owns_chroma) {
    st.decoded[1].fill(cx0 / 4, cy0 / 4, cw / 4, ch / 4);
    st.decoded[2].fill(cx0 / 4, cy0 / 4, cw / 4, ch / 4);
  }
  if (ibc_enabled_) {
    ibcs::finish_ibc_block(ibc_buffer_, ibc_neighbours_, r, c,
                           static_cast<int>(node.size), use_ibc,
                           use_ibc ? ibc_choice.bv : ibc::BlockVector{});
  }
  (void)node;
  return true;
}

bool InterFrameEncoder::tip_block_ref_active() const {
  return options_.tip_block_ref && !keyframe_ && temporal_ctx_.tip_as_ref;
}

temporal::MvPair InterFrameEncoder::tip_unit_mvs(int unit_mi_row, int unit_mi_col, inter::mc::Mv block_mv,
                              const temporal::TipConfig& tip) const {
  temporal::TipCandInput in;
  in.cand_row = unit_mi_row;
  in.cand_col = unit_mi_col;
  in.base_row = unit_mi_row;
  in.base_col = unit_mi_col;
  in.is_16x16 = false;
  in.base_mv = temporal::Mv{block_mv.y, block_mv.x};
  in.field = temporal_ctx_.field;
  return temporal::get_tip_cand(in, tip);
}

bool InterFrameEncoder::predict_tip_leaf(int x0, int y0, int w, int h, int unit, inter::mc::Mv block_mv,
                      const temporal::TipConfig& tip, std::vector<temporal::MvPair>* unit_mvs,
                      std::string& error) {
  const int units_x = (w + unit - 1) / unit;
  const int units_y = (h + unit - 1) / unit;
  unit_mvs->assign(static_cast<std::size_t>(units_x) *
                       static_cast<std::size_t>(units_y),
                   temporal::MvPair{});
  std::vector<std::int16_t> scratch(2U * 16U * 16U, 0);
  std::vector<std::uint8_t> block(16U * 16U, 0);
  const int frame_w = mi_cols_ * 4;
  const int frame_h = mi_rows_ * 4;
  for (int uy = 0; uy < units_y; ++uy) {
    for (int ux = 0; ux < units_x; ++ux) {
      const int px = x0 + ux * unit;
      const int py = y0 + uy * unit;
      const int unit_mi_row = py / 4;
      const int unit_mi_col = px / 4;
      const temporal::MvPair mvs = tip_unit_mvs(unit_mi_row, unit_mi_col, block_mv, tip);
      (*unit_mvs)[static_cast<std::size_t>(uy) *
                      static_cast<std::size_t>(units_x) +
                  static_cast<std::size_t>(ux)] = mvs;
      for (int plane = 0; plane < 3; ++plane) {
        const int sub_x = plane == 0 ? 0 : 1;
        const int sub_y = plane == 0 ? 0 : 1;
        const int bw = unit >> sub_x;
        const int bh = unit >> sub_y;
        const int bx = px >> sub_x;
        const int by = py >> sub_y;
        // §7.13.3.2: the decoder skips a prediction unit whose top-left is
        // outside the frame.
        if (bx >= recon_[plane].width || by >= recon_[plane].height) continue;
        temporal::TipPredictInput in;
        in.ref[0] = temporal_plane(references_[0], plane);
        in.ref[1] = temporal_plane(references_[1], plane);
        in.num_refs = 2;
        in.x = bx;
        in.y = by;
        in.w = bw;
        in.h = bh;
        in.sub_x = sub_x;
        in.sub_y = sub_y;
        in.frame_w = frame_w;
        in.frame_h = frame_h;
        in.mv[0] = mvs.mv[0];
        in.mv[1] = mvs.mv[1];
        in.cwp_weight = temporal::kTipWeightingFactor[tip.tip_global_wtd_index];
        in.filter = tip.interp_filter;
        in.use_ref_area = false;
        in.dst = block.data();
        in.dst_stride = bw;
        in.scratch0 = scratch.data();
        in.scratch1 = scratch.data() + 16 * 16;
        in.scratch_stride = bw;
        if (temporal::tip_predict_block(in) != temporal::TemporalStatus::kOk) {
          error = "TIP block prediction failed";
          return false;
        }
        const int vis_w = std::min(bw, recon_[plane].width - bx);
        const int vis_h = std::min(bh, recon_[plane].height - by);
        for (int r = 0; r < vis_h; ++r) {
          std::copy_n(block.data() + static_cast<std::size_t>(r) *
                                          static_cast<std::size_t>(bw),
                      vis_w,
                      recon_[plane].samples.data() +
                          static_cast<std::ptrdiff_t>(by + r) * recon_[plane].stride + bx);
        }
      }
    }
  }
  return true;
}

temporal::TipConfig InterFrameEncoder::tip_block_config() const {
  temporal::TipConfig tip;
  tip.mode = temporal::TipFrameMode::kAsReference;
  tip.tip_global_wtd_index = temporal_ctx_.tip_weight_index;
  tip.tip_global_mv = temporal::Mv{0, 0};
  tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
  tip.enable_tip_refinemv = false;
  tip.enable_refinemv = false;
  tip.opfl_refine_type = temporal::RefineType::kNone;
  tip.allow_tip_hole_fill = temporal_ctx_.tip_allow_hole_fill;
  tip.num_past_refs = temporal_ctx_.num_past_refs;
  tip.num_future_refs = temporal_ctx_.num_future_refs;
  tip.order_hint = temporal_ctx_.order_hint;
  tip.closest_past_order_hint = temporal_ctx_.tip_closest_past_order_hint;
  tip.closest_future_order_hint = temporal_ctx_.tip_closest_future_order_hint;
  return tip;
}

bool InterFrameEncoder::tip_ref_stack_ok(int r, int c, int w4, int h4, std::string& error) const {
  merge::StackParams params;
  params.mi_row = r;
  params.mi_col = c;
  params.bw4 = w4;
  params.bh4 = h4;
  params.sb_h4 = 16;
  params.mi_cols = mi_cols_;
  params.mi_rows = mi_rows_;
  params.mi = mv_grid_.data();
  params.ref_frame[0] = merge::kTipFrame;
  params.ref_frame[1] = merge::kRefNone;
  params.is_compound = false;
  params.global_mv[0] = inter::mc::Mv{0, 0};
  params.global_mv[1] = inter::mc::Mv{0, 0};
  params.reorder = merge::DrlReorder::kDisabled;
  params.max_drl_bits_minus_1 = 4;
  params.num_refs = num_total_refs_;
  params.frame_distance = frame_distance_.data();
  params.closest_past = temporal_ctx_.closest_past;
  params.closest_future = temporal_ctx_.closest_future;
  merge::MvStack stack;
  if (merge::build_stack_checked(params, &stack) != merge::Status::kOk) {
    error =
        "TIP block reference: TIP-ref MV stack needs an unimplemented "
        "§7.12.2.15/.17/.18 candidate";
    return false;
  }
  if (stack.count != 1 || stack.cand[0].mv[0].y != 0 || stack.cand[0].mv[0].x != 0) {
    error =
        "TIP block reference: expected the single zero GlobalMvs "
        "candidate";
    return false;
  }
  return true;
}

bool InterFrameEncoder::locate_tip_leaf(std::string& error) {
  tip_leaf_r_ = -1;
  tip_leaf_c_ = -1;
  tip_leaf_mv_ = inter::mc::Mv{0, 0};
  const std::uint8_t leaf_partition = static_cast<std::uint8_t>(partition::PartitionType::kNone);
  int last_inter = -1;
  for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
    if (blocks_[static_cast<std::size_t>(i)].is_inter != 0) {
      last_inter = i;
      break;
    }
  }
  if (last_inter < 0) {
    error = "TIP block reference: no trailing inter leaf is TIP-eligible";
    return false;
  }
  for (int i = last_inter + 1; i < static_cast<int>(blocks_.size()); ++i) {
    if (blocks_[static_cast<std::size_t>(i)].is_inter != 0) {
      error =
          "TIP block reference: an inter leaf is decoded after the "
          "candidate TIP leaf";
      return false;
    }
  }
  const syntax::BlockSyntax& node = blocks_[static_cast<std::size_t>(last_inter)];
  if (node.partition != leaf_partition || node.tree_type != 0) {
    error = "TIP block reference: no trailing inter leaf is TIP-eligible";
    return false;
  }
  const int w = partition::block_width(static_cast<partition::BlockSize>(node.bsize));
  const int h = partition::block_height(static_cast<partition::BlockSize>(node.bsize));
  if (w < 8 || h < 8) {
    error = "TIP block reference: last leaf is smaller than 8x8";
    return false;
  }
  if (!tip_ref_stack_ok(node.mi_row, node.mi_col, w / 4, h / 4, error)) {
    return false;
  }
  inter::mc::Mv mv{0, 0};
  if (node.inter_mode != syntax::kInterGlobalMv) {
    mv = inter::mc::Mv{node.mv_y, node.mv_x};
  }
  if (mv.y <= inter::mc::kMvLow + 8 || mv.y >= inter::mc::kMvUpp - 8 ||
      mv.x <= inter::mc::kMvLow + 8 || mv.x >= inter::mc::kMvUpp - 8) {
    mv = inter::mc::Mv{0, 0};
  }
  tip_leaf_r_ = node.mi_row;
  tip_leaf_c_ = node.mi_col;
  tip_leaf_mv_ = mv;
  return true;
}

bool InterFrameEncoder::encode_tip_leaf(SbSink& sink, const PartitionNode& node,
                     std::size_t index, std::string& error) {
  SbSink::State& st = sink.state;
  const int r = node.mi_row;
  const int c = node.mi_col;
  const int w = partition::block_width(node.size);
  const int h = partition::block_height(node.size);
  const int w4 = w / 4;
  const int h4 = h / 4;
  const int x0 = c * 4;
  const int y0 = r * 4;
  if (w < 8 || h < 8) {
    error = "TIP block reference: target leaf is smaller than 8x8";
    return false;
  }
  if (!tip_ref_stack_ok(r, c, w4, h4, error)) return false;
  const int unit = (w >= 16 && h >= 16) ? 16 : 8;
  const temporal::TipConfig tip = tip_block_config();
  std::vector<temporal::MvPair> unit_mvs;
  if (!predict_tip_leaf(x0, y0, w, h, unit, tip_leaf_mv_, tip, &unit_mvs, error)) {
    return false;
  }
  std::int64_t sse = 0;
  {
    const int vis_w = std::min(w, static_cast<int>(source_.width) - x0);
    const int vis_h = std::min(h, static_cast<int>(source_.height) - y0);
    for (int yy = 0; yy < vis_h; ++yy) {
      for (int xx = 0; xx < vis_w; ++xx) {
        const int d = static_cast<int>(recon_[0].at(x0 + xx, y0 + yy)) -
                      static_cast<int>(
                          source_.y[static_cast<std::size_t>(y0 + yy) *
                                        static_cast<std::size_t>(
                                            source_.width) +
                                    static_cast<std::size_t>(x0 + xx)]);
        sse += d * d;
      }
    }
  }

  syntax::BlockSyntax& entry = sink.blocks[index];
  entry.is_inter = 1;
  entry.skip_flag = 1;
  entry.tip_mode = 1;
  entry.tip_pred_mode = 1;  // NEWMV
  entry.inter_mode = syntax::kInterNewMv;
  entry.drl_index = 0;
  entry.ref_index = 0;
  entry.ref_index1 = 1;
  entry.compound = 0;
  entry.motion_mode = 0;
  entry.use_bawp = 0;
  entry.explicit_bawp = 0;
  entry.explicit_bawp_scale = 0;
  entry.use_bawp_chroma = 0;
  entry.mv_y = tip_leaf_mv_.y;
  entry.mv_x = tip_leaf_mv_.x;
  entry.mv1_y = 0;
  entry.mv1_x = 0;
  entry.mvd_y = tip_leaf_mv_.y;
  entry.mvd_x = tip_leaf_mv_.x;
  entry.use_amvd = 0;
  entry.block_mv_precision = -1;
  entry.mvd_sign_derived = 0;
  // A TIP leaf codes no residual; keep the sink-local "no levels" marker so
  // merge_sinks() maps it to the writer's 0 rather than to a rebased offset.
  entry.levels_offset = kNoLevels;
  entry.tx_part_count = 0;
  entry.cctx_type = 0;
  for (int plane = 0; plane < 3; ++plane) {
    entry.eob[plane] = 0;
  }

  // §7.22 field: the decoder stores the per-8x8 get_tip_mv() pair with
  // RefFrame[0]/[1] = the TIP pair, so mirror it in the MI grid.
  const int units_x = (w + unit - 1) / unit;
  for (int yy = 0; yy < h4 && r + yy < mi_rows_; ++yy) {
    for (int xx = 0; xx < w4 && c + xx < mi_cols_; ++xx) {
      const int ux = (xx * 4) / unit;
      const int uy = (yy * 4) / unit;
      const temporal::MvPair& mvs =
          unit_mvs[static_cast<std::size_t>(uy) *
                       static_cast<std::size_t>(units_x) +
                   static_cast<std::size_t>(ux)];
      inter::mc::MiInfo& mi =
          mv_grid_[static_cast<std::size_t>(r + yy) * static_cast<std::size_t>(mi_cols_) +
                   static_cast<std::size_t>(c + xx)];
      mi = inter::mc::MiInfo{};
      mi.ref[0] = temporal_ctx_.closest_past >= 0
                      ? static_cast<std::int8_t>(temporal_ctx_.closest_past)
                      : merge::kRefNone;
      mi.ref[1] = temporal_ctx_.closest_future >= 0
                      ? static_cast<std::int8_t>(temporal_ctx_.closest_future)
                      : merge::kRefNone;
      mi.mv[0][0] = mvs.mv[0].row;
      mi.mv[0][1] = mvs.mv[0].col;
      mi.mv[1][0] = mvs.mv[1].row;
      mi.mv[1][1] = mvs.mv[1].col;
      mi.is_inter = 1;
      mi.w4 = static_cast<std::int8_t>(w4);
      mi.h4 = static_cast<std::int8_t>(h4);
      mi.col_base = static_cast<std::int16_t>(c);
      mi.row_base = static_cast<std::int16_t>(r);
      mi.bsize = static_cast<std::int16_t>(node.size);
    }
  }
  set_intra_joint(r, c, w4, h4, 0);  // inter_block_mode_info(): DC
  // §6.19.2.3: the leaf is now decoded for the availability of later leaves.
  st.decoded[0].fill(c, r, w4, h4);
  st.decoded[1].fill(c / 2, r / 2, w4 / 2, h4 / 2);
  st.decoded[2].fill(c / 2, r / 2, w4 / 2, h4 / 2);
  ++st.inter_blocks;
  ++st.tip_blocks;
  if (trace_) {
    const temporal::MvPair first = unit_mvs.empty() ? temporal::MvPair{} : unit_mvs.front();
    std::cerr << "m2 tip_block r=" << r << " c=" << c << " " << w << "x" << h
              << " ref=TIP pred=NEWMV drl=0 mv=(" << tip_leaf_mv_.y << "," << tip_leaf_mv_.x
              << ") unit=" << unit << " tipmv0=(" << first.mv[0].row << "," << first.mv[0].col
              << ")/(" << first.mv[1].row << "," << first.mv[1].col
              << ") weight=" << temporal::kTipWeightingFactor[tip.tip_global_wtd_index]
              << " holefill=" << (tip.allow_tip_hole_fill ? 1 : 0) << " sse=" << sse << "\n";
  }
  return true;
}

bool InterFrameEncoder::apply_tip_block_ref(std::string& error) {
  if (!tip_block_ref_active()) return true;
  if (blocks_.empty()) {
    error = "TIP block reference: frame has no coded leaves";
    return false;
  }
  if (!locate_tip_leaf(error)) return false;
  if (tip_leaf_r_ < 0 || tip_leaf_c_ < 0) {
    error = "TIP block reference: no TIP-eligible trailing inter leaf";
    return false;
  }
  tip_blocks_ = 0;
  tip_leaf_armed_ = true;
  if (!decide(error)) {
    tip_leaf_armed_ = false;
    return false;
  }
  tip_leaf_armed_ = false;
  if (tip_blocks_ != 1) {
    error =
        "TIP block reference: the second decision pass coded no TIP "
        "leaf";
    return false;
  }
  return true;
}

}  // namespace av2enc::video
