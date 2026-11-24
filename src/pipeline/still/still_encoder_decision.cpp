// Still-picture encoder: partition, mode, IBC and TX-partition
// decisions.  Split verbatim out of still_encoder.cpp; see
// still_encoder_internal.h for the shared declaration.

#include "pipeline/still/still_encoder_internal.h"

namespace av2enc::still {
namespace detail {

bool Encoder::collect_node(const partition::PartitionNode& node,
                  const ChromaInfo& inherited_chroma) {
    const ChromaInfo chroma = chroma_step(node, inherited_chroma);
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    std::uint16_t mask = 0;
    for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
      if (node.allowed.allowed[p]) mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
    entry.flags |= syntax::kFlagHasChromaInfo;
    entry.has_chroma = node.has_chroma ? 1 : 0;
    entry.chroma_offset = node.chroma_offset ? 1 : 0;
    entry.chroma_bsize = static_cast<std::uint8_t>(chroma.size);
    entry.chroma_mi_row = static_cast<std::uint16_t>(chroma.mi_row);
    entry.chroma_mi_col = static_cast<std::uint16_t>(chroma.mi_col);
    entry.segment_id = sb_segment_id(node.mi_row, node.mi_col);

    if (trace_) {
      std::cerr << "m1 node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " part=" << static_cast<int>(node.partition)
                << " forced=" << (node.forced ? 1 : 0) << " allowed="
                << node.allowed.allowed[0] << node.allowed.allowed[1]
                << node.allowed.allowed[2] << node.allowed.allowed[3] << "\n";
    }

    if (node.partition == PartitionType::kNone) {
      if (!encode_block(node.mi_row, node.mi_col, node.size, index,
                        node.has_chroma, chroma)) {
        return false;
      }
    } else {
      for (const partition::PartitionNode& child : node.children) {
        if (!collect_node(child, chroma)) return false;
      }
    }
    const std::size_t subtree = blocks_.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error_ = "partition subtree too large for the syntax stream";
      return false;
    }
    blocks_[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

bool Encoder::collect_node_ext(const pe::PartitionNode& node) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    std::uint16_t mask = 0;
    for (int p = 0; p < pe::kPartitionCount; ++p) {
      if (node.allowed.allowed[p]) mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
    entry.flags |= syntax::kFlagHasChromaInfo;
    entry.has_chroma = node.has_chroma ? 1 : 0;
    entry.chroma_offset = node.chroma_offset ? 1 : 0;
    entry.chroma_bsize = static_cast<std::uint8_t>(node.chroma.size);
    entry.chroma_mi_row = static_cast<std::uint16_t>(node.chroma.mi_row);
    entry.chroma_mi_col = static_cast<std::uint16_t>(node.chroma.mi_col);
    // §5.20.5.7 intra_segment_id(): the ext tree must code the same per-SB
    // segment as collect_node(); otherwise quant/dequant/deblock use segment 1
    // locally while the bitstream says segment 0.
    entry.segment_id = sb_segment_id(node.mi_row, node.mi_col);

    if (trace_) {
      std::cerr << "m1 ext node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " part=" << static_cast<int>(node.partition)
                << " forced=" << (node.forced ? 1 : 0) << "\n";
    }

    if (node.partition == pe::PartitionType::kNone) {
      const ChromaInfo chroma{
          node.has_chroma, node.chroma.mi_row, node.chroma.mi_col,
          node.chroma.size};
      if (!encode_block(node.mi_row, node.mi_col, node.size, index,
                        node.has_chroma, chroma)) {
        return false;
      }
    } else {
      for (const pe::PartitionNode& child : node.children) {
        if (!collect_node_ext(child)) return false;
      }
    }
    const std::size_t subtree = blocks_.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error_ = "partition subtree too large for the syntax stream";
      return false;
    }
    blocks_[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

bool Encoder::collect_sdp_node(const sdpp::Node& node) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    entry.tree_type = static_cast<std::uint8_t>(node.tree_type);
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
    // §5.20.5.7: carry the per-SB segment id into the two-tree walk too.
    entry.segment_id = sb_segment_id(node.mi_row, node.mi_col);

    if (trace_) {
      std::cerr << "m1 sdp node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " tree=" << sdpp::tree_type_name(node.tree_type)
                << " part=" << static_cast<int>(node.partition)
                << " sdp_split=" << (node.sdp_split ? 1 : 0)
                << " forced=" << (node.forced ? 1 : 0) << "\n";
    }

    if (node.sdp_split) {
      if (node.children.size() != 2) {
        error_ = "SDP split node does not have exactly two tree children";
        return false;
      }
      for (const sdpp::Node& child : node.children) {
        if (!collect_sdp_node(child)) return false;
      }
    } else if (node.partition == sdpp::PartitionType::kNone) {
      const bool has_chroma = node.codes_chroma;
      const ChromaInfo chroma{node.has_chroma, node.chroma.mi_row,
                              node.chroma.mi_col, node.chroma.size};
      if (!encode_block(node.mi_row, node.mi_col, node.size, index,
                        has_chroma, chroma,
                        static_cast<int>(node.tree_type))) {
        return false;
      }
    } else {
      for (const sdpp::Node& child : node.children) {
        if (!collect_sdp_node(child)) return false;
      }
    }
    const std::size_t subtree = blocks_.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error_ = "partition subtree too large for the syntax stream";
      return false;
    }
    blocks_[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

bool Encoder::prepare_edges(int plane, int x0, int y0, int w, int h,
                   intra::IntraEdges& edges) const {
    intra::NeighborRequest request;
    request.recon = intra::PlaneView{recon_[plane].samples.data(),
                                     recon_[plane].width};
    request.x = x0;
    request.y = y0;
    request.w = w;
    request.h = h;
    request.max_x = (plane == 0 ? mi_cols_ * 4 : mi_cols_ * 2) - 1;
    request.max_y = (plane == 0 ? mi_rows_ * 4 : mi_rows_ * 2) - 1;
    request.have_left = x0 > 0;
    request.have_above = y0 > 0;
    const int sub = plane == 0 ? 0 : 1;
    request.num4_above_right =
        count_top_right(plane, x0 >> 2, y0 >> 2, w >> 2);
    request.num4_below_left =
        count_bottom_left(plane, x0 >> 2, y0 >> 2, h >> 2);
    (void)sub;
    request.color_plane = plane;
    return intra::prepare_edges(request, 8, edges) == intra::IntraStatus::kOk;
  }

bool Encoder::neighborhood_smooth(int plane, int r, int c, int w4, int h4,
                         bool& above_smooth, bool& left_smooth) const {
    above_smooth = false;
    left_smooth = false;
    if (plane == 0) {
      // IntraJointMode stores the reordered non-directional index (DC=0,
      // SMOOTH=1, SMOOTH_V=2, SMOOTH_H=3, PAETH=4) or 5+directional index.
      const auto joint_is_smooth = [](int joint) {
        return joint >= 1 && joint <= 3;
      };
      if (r > 0 && c < mi_cols_) {
        above_smooth = joint_is_smooth(
            intra_joint_[static_cast<std::size_t>(r - 1) *
                             static_cast<std::size_t>(mi_cols_) +
                         static_cast<std::size_t>(c)]);
      }
      if (c > 0 && r < mi_rows_) {
        left_smooth = joint_is_smooth(
            intra_joint_[static_cast<std::size_t>(r) *
                             static_cast<std::size_t>(mi_cols_) +
                         static_cast<std::size_t>(c - 1)]);
      }
    } else {
      // Callers pass the chroma 4x4-grid index (luma MI / 2), which is already
      // the index used to write uv_smooth_. Do not halve it again.
      const int cr = r;
      const int cc = c;
      const int cmi_cols = mi_cols_ / 2;
      if (cr > 0 && cc < cmi_cols) {
        above_smooth =
            uv_smooth_[static_cast<std::size_t>(cr - 1) *
                           static_cast<std::size_t>(cmi_cols) +
                       static_cast<std::size_t>(cc)] != 0;
      }
      if (cc > 0 && cr < mi_rows_ / 2) {
        left_smooth =
            uv_smooth_[static_cast<std::size_t>(cr) *
                           static_cast<std::size_t>(cmi_cols) +
                       static_cast<std::size_t>(cc - 1)] != 0;
      }
    }
    (void)w4;
    (void)h4;
    return true;
  }

bool Encoder::predict_block(int plane, int x0, int y0, int w, int h,
                   intra::Mode mode, int angle_delta,
                   std::vector<std::uint8_t>& out,
                   bool uv_is_cfl, int mrl_index,
                   bool mrl_sec_index, bool use_dip,
                   int dip_mode, bool dip_transpose,
                   int block_mi_row, int block_mi_col) {
    if (mrl_index > 0) {
      // MRL is luma-only (§7.13.2.1); chroma always predicts line 0.
      if (plane != 0) {
        error_ = "MRL is not signalable for chroma";
        return false;
      }
      return predict_luma_mrl(x0, y0, w, h, mode, angle_delta, mrl_index,
                              mrl_sec_index, out, block_mi_row,
                              block_mi_col);
    }
    if (use_dip) {
      if (plane != 0 || mode != intra::Mode::kDc) {
        error_ = "DIP is luma DC only";
        return false;
      }
      return predict_luma_dip(x0, y0, w, h, dip_mode, dip_transpose, out);
    }
    intra::IntraEdges edges;
    if (!prepare_edges(plane, x0, y0, w, h, edges)) {
      error_ = "prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    // §7.13.2.15/.16: the smooth flags are read at the coding-block origin
    // (MiRow/MiCol), not at the transform-part origin.
    const int smooth_row = block_mi_row >= 0 ? block_mi_row : (y0 >> 2);
    const int smooth_col = block_mi_col >= 0 ? block_mi_col : (x0 >> 2);
    neighborhood_smooth(plane, smooth_row, smooth_col, w >> 2, h >> 2,
                        above_smooth, left_smooth);
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (options_.enable_ibp) {
      // §7.13.2.7-.12: IBP changes DC (post-filter), directional edge
      // filtering and the two-angle blend.  ibp::predict is the drop-in
      // replacement for intra::predict_intra and never mutates the edges.
      ibp::PredictParams params;
      params.mode = mode;
      params.angle_delta = angle_delta;
      params.enable_ibp = true;
      params.enable_edge_filter = edge_filter_enabled();
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      params.use_dip = false;
      params.uv_is_cfl = uv_is_cfl;
      if (ibp::predict(edges, params, out.data(),
                       static_cast<std::size_t>(w)) !=
          intra::IntraStatus::kOk) {
        error_ = "ibp::predict failed";
        return false;
      }
      return true;
    }
    intra::PredictParams params;
    params.mode = mode;
    params.angle_delta = angle_delta;
    params.enable_edge_filter = edge_filter_enabled();
    params.above_neighbor_is_smooth = above_smooth;
    params.left_neighbor_is_smooth = left_smooth;
    if (intra::predict_intra(edges, params, out.data(),
                             static_cast<std::size_t>(w)) !=
        intra::IntraStatus::kOk) {
      error_ = "predict_intra failed";
      return false;
    }
    return true;
  }

bool Encoder::evaluate_luma_dip(int x0, int y0, int w, int h, ModeChoice& best) {
    if (!options_.enable_dip || !dip::dip_block_available(w, h)) return true;
    const dip::Block block = make_dip_block(x0, y0, w, h);
    dip::Edges dip_edges;
    if (dip::prepare_edges(block, 8, dip_edges) != dip::Status::kOk) {
      error_ = "dip::prepare_edges failed";
      return false;
    }
    std::vector<std::uint8_t> dip_src(
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int yy = 0; yy < h; ++yy) {
      for (int xx = 0; xx < w; ++xx) {
        dip_src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(xx)] =
            source_sample(0, x0 + xx, y0 + yy);
      }
    }
    dip::DecisionParams dip_params;
    dip_params.enable_dip = true;
    dip_params.y_mode = kDcPred;
    dip_params.palette_size_y = 0;
    dip_params.tree_is_chroma_part = false;
    dip_params.metric = dip::CostMetric::kSse;
    dip::Decision dip_decision;
    if (dip::decide(dip_edges, dip_src.data(), static_cast<std::size_t>(w),
                    dip_params, dip_decision) != dip::Status::kOk ||
        !dip_decision.available) {
      error_ = "dip::decide failed";
      return false;
    }
    if (static_cast<std::int64_t>(dip_decision.cost.sse) < best.sse) {
      best.sse = static_cast<std::int64_t>(dip_decision.cost.sse);
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.mrl_index = 0;
      best.mrl_sec_index = false;
      best.use_dip = true;
      best.dip_mode = dip_decision.mode;
      best.dip_transpose = dip_decision.transpose;
    }
    return true;
  }

dip::Block Encoder::make_dip_block(int x0, int y0, int w, int h) {
    dip::Block block;
    block.recon =
        dip::PlaneView::from_u8(recon_[0].samples.data(), recon_[0].width);
    block.x = x0;
    block.y = y0;
    block.w = w;
    block.h = h;
    block.max_x = mi_cols_ * 4 - 1;
    block.max_y = mi_rows_ * 4 - 1;
    block.have_left = x0 > 0;
    block.have_above = y0 > 0;
    block.num4_above_right = count_top_right(0, x0 >> 2, y0 >> 2, w >> 2);
    block.num4_below_left = count_bottom_left(0, x0 >> 2, y0 >> 2, h >> 2);
    return block;
  }

bool Encoder::predict_luma_dip(int x0, int y0, int w, int h, int dip_mode,
                      bool dip_transpose, std::vector<std::uint8_t>& out) {
    const dip::Block block = make_dip_block(x0, y0, w, h);
    dip::Params params;
    params.mode = dip_mode;
    params.transpose = dip_transpose;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (dip::predict_block8(block, 8, params, out.data(),
                            static_cast<std::size_t>(w)) !=
        dip::Status::kOk) {
      error_ = "dip::predict_block8 failed";
      return false;
    }
    return true;
  }

bool Encoder::predict_luma_mrl(int x0, int y0, int w, int h, intra::Mode mode,
                      int angle_delta, int mrl_index, bool mrl_sec_index,
                      std::vector<std::uint8_t>& out, int block_mi_row,
                      int block_mi_col) {
    bool above_smooth = false;
    bool left_smooth = false;
    const int smooth_row = block_mi_row >= 0 ? block_mi_row : (y0 >> 2);
    const int smooth_col = block_mi_col >= 0 ? block_mi_col : (x0 >> 2);
    neighborhood_smooth(0, smooth_row, smooth_col, w >> 2, h >> 2,
                        above_smooth, left_smooth);
    mrl::Params params;
    if (!mrl_stream::make_predict_params(
            av2_y_mode(mode), angle_delta, mrl_index, mrl_sec_index,
            edge_filter_enabled(), above_smooth, left_smooth, &params, w, h)) {
      error_ = "MRL prediction parameters rejected";
      return false;
    }
    mrl::Block block;
    block.recon =
        intra::PlaneView{recon_[0].samples.data(), recon_[0].width};
    block.x = x0;
    block.y = y0;
    block.w = w;
    block.h = h;
    block.max_x = mi_cols_ * 4 - 1;
    block.max_y = mi_rows_ * 4 - 1;
    block.have_left = x0 > 0;
    block.have_above = y0 > 0;
    block.num4_above_right = count_top_right(0, x0 >> 2, y0 >> 2, w >> 2);
    block.num4_below_left = count_bottom_left(0, x0 >> 2, y0 >> 2, h >> 2);
    block.plane = 0;
    // §7.13.2.1: sbHeight = Block_Height[SbSize].  An intra frame uses the
    // frame SbSize (128x128 under --still-sb128/sb256, 64x64 otherwise), so
    // the MRL above-row reset must follow geometry_, not a fixed 64.
    block.sb_height = geometry_.sb_size;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (mrl::predict_block(block, 8, params, out.data(),
                           static_cast<std::size_t>(w)) !=
        mrl::MrlStatus::kOk) {
      error_ = "mrl::predict_block failed";
      return false;
    }
    return true;
  }

std::int64_t Encoder::sse_against_source(int plane, int x0, int y0, int w, int h,
                                const std::vector<std::uint8_t>& pred) const {
    std::int64_t sse = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)];
        const int d = src - p;
        sse += static_cast<std::int64_t>(d) * d;
      }
    }
    return sse;
  }

bool Encoder::choose_luma_mode(int x0, int y0, int w, int h, ModeChoice& best) {
    const char* forced_mode = std::getenv("AV2ENC_STILL_FORCE_Y");
    if (x0 > 0) {
      if (const char* second = std::getenv("AV2ENC_STILL_FORCE_Y2")) {
        forced_mode = second;
      }
    }
    if (forced_mode != nullptr) {
      int mode = 0;
      int delta = 0;
      if (std::sscanf(forced_mode, "%d:%d", &mode, &delta) == 2) {
        best.mode = static_cast<intra::Mode>(mode);
        best.angle_delta = delta;
        best.av2_mode = mode;
        best.sse = 0;
        return true;
      }
    }
    if (options_.luma_dc_only || options_.force_dc ||
        std::getenv("AV2ENC_STILL_LUMA_DC") != nullptr) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      if (options_.enable_dip || options_.palette) {
        // Both the DIP and the palette candidates are scored against the
        // ordinary DC prediction the encoder would otherwise code, so its SSE
        // must be real here (the value is unused otherwise).
        std::vector<std::uint8_t> dc_pred;
        if (!predict_block(0, x0, y0, w, h, intra::Mode::kDc, 0, dc_pred)) {
          return false;
        }
        best.sse = sse_against_source(0, x0, y0, w, h, dc_pred);
        if (!evaluate_luma_dip(x0, y0, w, h, best)) return false;
      }
      return true;
    }
    intra::IntraEdges edges;
    if (!prepare_edges(0, x0, y0, w, h, edges)) {
      error_ = "luma prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    neighborhood_smooth(0, y0 >> 2, x0 >> 2, w >> 2, h >> 2, above_smooth,
                        left_smooth);
    std::array<intra::ModeCandidate, 61> candidates;
    const int count = intra::build_mode_candidates(candidates, true);
    if (count <= 0) {
      error_ = "build_mode_candidates failed";
      return false;
    }
    std::vector<std::uint8_t> pred;
    pred.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    const bool base_only = std::getenv("AV2ENC_STILL_BASE_ONLY") != nullptr;
    const bool no_directional = std::getenv("AV2ENC_STILL_NO_DIR") != nullptr;
    const bool mrl_search = options_.enable_mrl;
    std::vector<av2enc::rdo::IntraCandidateCost> rdo_candidates;
    const av2enc::rdo::IntraRdoConfig rdo_config =
        av2enc::rdo::make_intra_rdo_config(
            static_cast<double>(quant_step_), options_.rdo_sse_scale, true);
    if (options_.rdo_intra) {
      rdo_candidates.reserve(static_cast<std::size_t>(count));
    }
    best.sse = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < count; ++i) {
      intra::PredictParams params;
      params.mode = candidates[static_cast<std::size_t>(i)].mode;
      params.angle_delta = candidates[static_cast<std::size_t>(i)].angle_delta;
      if (base_only && params.angle_delta != 0) continue;
      if (no_directional && intra::is_directional(params.mode)) continue;
      params.enable_edge_filter = edge_filter_enabled();
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      if (options_.enable_ibp) {
        ibp::PredictParams ibp_params;
        ibp_params.mode = params.mode;
        ibp_params.angle_delta = params.angle_delta;
        ibp_params.enable_ibp = true;
        ibp_params.enable_edge_filter = params.enable_edge_filter;
        ibp_params.above_neighbor_is_smooth = above_smooth;
        ibp_params.left_neighbor_is_smooth = left_smooth;
        if (ibp::predict(edges, ibp_params, pred.data(),
                         static_cast<std::size_t>(w)) !=
            intra::IntraStatus::kOk) {
          error_ = "luma ibp::predict failed";
          return false;
        }
      } else if (intra::predict_intra(edges, params, pred.data(),
                                      static_cast<std::size_t>(w)) !=
                 intra::IntraStatus::kOk) {
        error_ = "luma predict_intra failed";
        return false;
      }
      const std::int64_t sse = sse_against_source(0, x0, y0, w, h, pred);
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = params.mode;
        best.angle_delta = params.angle_delta;
        best.av2_mode = av2_y_mode(params.mode);
        best.mrl_index = 0;
        best.mrl_sec_index = false;
      }
      // §5.20.5.5 MRL: only a directional luma mode can signal mrl_index, so
      // the extra search runs on those candidates only.  mrl_index == 0 is the
      // candidate just evaluated above.
      if (mrl_search && intra::is_directional(params.mode)) {
        const bool sec_allowed = !(w == 4 && h == 4);
        for (int mrl_index = 1; mrl_index <= mrl::kMaxMrlIndex; ++mrl_index) {
          for (int sec = 0; sec <= (sec_allowed ? 1 : 0); ++sec) {
            std::vector<std::uint8_t> mrl_pred;
            if (!predict_luma_mrl(x0, y0, w, h, params.mode,
                                  params.angle_delta, mrl_index, sec != 0,
                                  mrl_pred)) {
              return false;
            }
            const std::int64_t mrl_sse =
                sse_against_source(0, x0, y0, w, h, mrl_pred);
            if (mrl_sse < best.sse) {
              best.sse = mrl_sse;
              best.mode = params.mode;
              best.angle_delta = params.angle_delta;
              best.av2_mode = av2_y_mode(params.mode);
              best.mrl_index = mrl_index;
              best.mrl_sec_index = sec != 0;
            }
          }
        }
      }
      if (options_.rdo_intra) {
        // Probe the real quantised levels for this candidate.  compute_tx()
        // reads only the source and the predictor, so it does not perturb the
        // reconstruction state.
        TxPlan probe;
        if (!compute_tx(0, x0, y0, w, h, pred, probe)) {
          error_ = "rate-aware intra: compute_tx probe failed";
          return false;
        }
        std::vector<std::uint16_t> probe_scan(
            static_cast<std::size_t>(probe.adj_w * probe.adj_h));
        syntax::build_scan(probe.adj_w, probe.adj_h, probe_scan);
        std::vector<std::int32_t> probe_levels;
        probe_levels.reserve(static_cast<std::size_t>(probe.eob));
        for (int scan_pos = 0; scan_pos < probe.eob; ++scan_pos) {
          probe_levels.push_back(
              probe.levels[probe_scan[static_cast<std::size_t>(scan_pos)]]);
        }
        av2enc::rdo::IntraCandidateCost candidate;
        candidate.mode = av2_y_mode(params.mode);
        candidate.angle_delta = params.angle_delta;
        candidate.sse = static_cast<double>(sse);
        candidate.syntax_rate_bits =
            av2enc::rdo::default_intra_syntax_rate_bits(candidate.mode,
                                                        candidate.angle_delta);
        candidate.coefficient_rate_bits =
            av2enc::rdo::intra_coefficient_rate_bits_from_levels(
                probe_levels, probe.eob, probe.adj_w * probe.adj_h);
        rdo_candidates.push_back(candidate);
      }
    }
    if (options_.rdo_intra && !rdo_candidates.empty()) {
      // The choice carries `found`; the message is not consumed here.
      std::string intra_rdo_error;
      const av2enc::rdo::IntraModeChoice choice =
          av2enc::rdo::select_intra_mode(rdo_candidates, rdo_config,
                                         intra_rdo_error);
      if (choice.found) {
        best.mode = static_cast<intra::Mode>(choice.mode);
        best.angle_delta = choice.angle_delta;
        best.av2_mode = choice.mode;
        best.sse = static_cast<std::int64_t>(choice.sse);
      }
    }
    // §5.20.7.9: DIP alternative to DC, compared against the best ordinary
    // prediction by SSE.
    if (!base_only && !evaluate_luma_dip(x0, y0, w, h, best)) return false;
    return true;
  }

std::int64_t Encoder::block_variance_cost_at(int mi_row, int mi_col,
                                    BlockSize size) const {
    const partition::BlockRect rect =
        partition::block_rect(geometry_, mi_row, mi_col, size);
    const int w = rect.visible_width;
    const int h = rect.visible_height;
    if (w <= 0 || h <= 0) return 0.0;
    std::int64_t sum = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        sum += source_sample(0, rect.x + x, rect.y + y);
      }
    }
    const double mean =
        static_cast<double>(sum) / static_cast<double>(w * h);
    double sse = 0.0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const double d = static_cast<double>(source_sample(
                            0, rect.x + x, rect.y + y)) -
                         mean;
        sse += d * d;
      }
    }
    return static_cast<std::int64_t>(sse);
  }

bool Encoder::choose_luma_tx_partition(int r, int c, int w, int h,
                              partition::BlockSize size,
                              const ModeChoice& y,
                              std::vector<TxPlan>* plans,
                              std::vector<syntax::LumaTxPart>* parts,
                              int* partition_out) {
    std::vector<tpart::TxPartition> options;
    options.push_back(tpart::TxPartition::kNone);
    const tpart::LegalSplits legal =
        tpart::legal_splits(static_cast<int>(size),
                            /*reduced_tx_part_set=*/false);
    for (int i = 0; i < legal.option_count; ++i) {
      const tpart::TxPartition option = legal.options[i];
      if (option == tpart::TxPartition::kNone) continue;
      if (option != tpart::TxPartition::kSplit &&
          option != tpart::TxPartition::kHorz &&
          option != tpart::TxPartition::kVert) {
        continue;
      }
      tpart::SplitGeometry geometry;
      std::string geometry_error;
      if (!tpart::tx_partition_geometry(static_cast<int>(size), r, c, option,
                                        &geometry, geometry_error)) {
        continue;
      }
      bool inside = true;
      for (int k = 0; k < geometry.count; ++k) {
        if (geometry.parts[k].mi_col >= mi_cols_ ||
            geometry.parts[k].mi_row >= mi_rows_) {
          inside = false;
          break;
        }
      }
      if (inside) options.push_back(option);
    }

    // Debug gate hook: force one legal partition by name.
    bool have_forced = false;
    tpart::TxPartition forced = tpart::TxPartition::kNone;
    if (const char* value = std::getenv("AV2ENC_STILL_FORCE_TX_PARTITION")) {
      if (std::strcmp(value, "none") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kNone;
      } else if (std::strcmp(value, "split") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kSplit;
      } else if (std::strcmp(value, "horz") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kHorz;
      } else if (std::strcmp(value, "vert") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kVert;
      }
    }

    const double lambda =
        0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const LumaSnapshot snapshot = take_luma_snapshot(c*4, r*4, w, h);
    double best_cost = std::numeric_limits<double>::infinity();
    int best_partition = static_cast<int>(tpart::TxPartition::kNone);
    for (const tpart::TxPartition option : options) {
      if (have_forced && option != forced) continue;
      restore_luma_snapshot(snapshot);
      std::vector<TxPlan> trial_plans;
      std::vector<syntax::LumaTxPart> trial_parts;
      double sse = 0.0;
      if (!encode_luma_parts(r, c, w, h, size, y, static_cast<int>(option),
                             &trial_plans, &trial_parts, &sse)) {
        return false;
      }
      const double syntax_bits =
          option == tpart::TxPartition::kNone ? 0.0 : 3.0;
      const double cost = sse + lambda * syntax_bits;
      if (cost < best_cost) {
        best_cost = cost;
        best_partition = static_cast<int>(option);
      }
    }
    restore_luma_snapshot(snapshot);
    double sse = 0.0;
    if (!encode_luma_parts(r, c, w, h, size, y, best_partition, plans, parts,
                           &sse)) {
      return false;
    }
    *partition_out = best_partition;
    return true;
  }

ibc::FrameView Encoder::ibc_frame_view() const {
    // The intra/palette paths read only the reconstructed planes.  Bounds are
    // the coded MiCols*4 / MiRows*4 plane; is_bv_valid owns the tile rules.
    ibc::FrameView view;
    view.y = ibc::PlaneView{recon_[0].samples.data(), recon_[0].width,
                            mi_cols_ * 4, mi_rows_ * 4};
    view.u = ibc::PlaneView{recon_[1].samples.data(), recon_[1].width,
                            mi_cols_ * 2, mi_rows_ * 2};
    view.v = ibc::PlaneView{recon_[2].samples.data(), recon_[2].width,
                            mi_cols_ * 2, mi_rows_ * 2};
    view.has_chroma = true;
    view.sub_x = 1;
    view.sub_y = 1;
    return view;
  }

bool Encoder::ibc_copy_sse(const ibc::FrameView& frame, int r, int c,
                  partition::BlockSize size, ibc::BlockVector bv,
                  int visible_w, int visible_h, std::int64_t* sse) {
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int x0 = c * 4;
    const int y0 = r * 4;
    std::vector<std::uint8_t> copy_y(static_cast<std::size_t>(w) *
                                     static_cast<std::size_t>(h));
    const int cw = (w + 1) / 2;
    const int ch = (h + 1) / 2;
    std::vector<std::uint8_t> copy_u(static_cast<std::size_t>(cw) *
                                     static_cast<std::size_t>(ch));
    std::vector<std::uint8_t> copy_v(copy_u.size());
    if (!ibc::copy_block(frame, r, c, static_cast<int>(size), bv,
                         copy_y.data(), w,
                         copy_u.data(), cw, copy_v.data(), cw, visible_w,
                         visible_h)) {
      return false;
    }
    std::int64_t total = 0;
    for (int yy = 0; yy < visible_h; ++yy) {
      for (int xx = 0; xx < visible_w; ++xx) {
        const int d = static_cast<int>(
                          copy_y[static_cast<std::size_t>(yy) *
                                     static_cast<std::size_t>(w) +
                                 static_cast<std::size_t>(xx)]) -
                      static_cast<int>(source_sample(0, x0 + xx, y0 + yy));
        total += static_cast<std::int64_t>(d) * d;
      }
    }
    *sse = total;
    if (total != 0) return true;
    const int cvw = (visible_w + 1) / 2;
    const int cvh = (visible_h + 1) / 2;
    for (int plane = 1; plane < 3; ++plane) {
      const std::vector<std::uint8_t>& copy = plane == 1 ? copy_u : copy_v;
      const int px0 = x0 / 2;
      const int py0 = y0 / 2;
      for (int yy = 0; yy < cvh; ++yy) {
        for (int xx = 0; xx < cvw; ++xx) {
          if (copy[static_cast<std::size_t>(yy) *
                       static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(xx)] !=
              source_sample(plane, px0 + xx, py0 + yy)) {
            *sse = 1;  // non-zero marks "not an exact copy"
            return true;
          }
        }
      }
    }
    return true;
  }

bool Encoder::ibc_low_color_rank(int x0, int y0, int w, int h) const {
    const int visible_w = std::min(w, mi_cols_ * 4 - x0);
    const int visible_h = std::min(h, mi_rows_ * 4 - y0);
    bool seen[256] = {};
    int distinct = 0;
    for (int yy = 0; yy < visible_h; ++yy) {
      for (int xx = 0; xx < visible_w; ++xx) {
        const int v = source_sample(0, x0 + xx, y0 + yy);
        if (!seen[v]) {
          seen[v] = true;
          if (++distinct > 8) return false;
        }
      }
    }
    return true;
  }

bool Encoder::choose_ibc(int r, int c, partition::BlockSize size, IbcChoice& out) {
    out = IbcChoice{};
    if (!options_.ibc ||
        !ibc_stream::intrabc_available(true, static_cast<int>(size))) {
      return true;
    }
    const ibc::FrameView frame = ibc_frame_view();
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int visible_w = std::min(w, mi_cols_ * 4 - x0);
    const int visible_h = std::min(h, mi_rows_ * 4 - y0);
    const ibc_stream::BvStack stack = ibc_stream::build_bv_stack(
        frame, ibc_config_, ibc_buffer_, r, c, static_cast<int>(size),
        ibc_neighbours_);
    const auto try_bv = [&](ibc::BlockVector bv, int stack_index) {
      // Integer block vectors only (force_integer_mv = 1).
      if ((bv.row & 7) != 0 || (bv.col & 7) != 0) return false;
      std::string ignored;
      if (!ibc_stream::is_bv_encodable(frame, ibc_config_, ibc_buffer_, r, c,
                                       static_cast<int>(size), bv, ignored)) {
        return false;
      }
      std::int64_t sse = 0;
      if (!ibc_copy_sse(frame, r, c, size, bv, visible_w, visible_h, &sse) ||
          sse != 0) {
        return false;
      }
      out.found = true;
      out.bv = bv;
      out.stack_index = stack_index;
      return true;
    };
    for (int i = 0; i < stack.count; ++i) {
      if (!stack.entries[static_cast<std::size_t>(i)].valid) continue;
      if (try_bv(stack.entries[static_cast<std::size_t>(i)].bv, i)) {
        return true;
      }
    }
    static constexpr int kPeriodic[] = {8, 16, 32, 64, 128};
    for (const int d : kPeriodic) {
      const ibc::BlockVector cands[2] = {ibc::BlockVector{-d * 8, 0},
                                         ibc::BlockVector{0, -d * 8}};
      for (const ibc::BlockVector& bv : cands) {
        if (try_bv(bv, -1)) return true;
      }
    }
    if (!ibc_low_color_rank(x0, y0, w, h)) return true;
    const int step = std::max(1, options_.ibc_search_step);
    for (int sy = std::max(0, y0 - 64); sy <= y0; sy += step) {
      for (int sx = std::max(0, x0 - 64); sx <= x0; sx += step) {
        const ibc::BlockVector bv{(sy - y0) * 8, (sx - x0) * 8};
        if (try_bv(bv, -1)) return true;
      }
    }
    return true;
  }

bool Encoder::encode_block(int r, int c, partition::BlockSize size,
                  std::size_t index, bool has_chroma,
                  const ChromaInfo& chroma, int tree_type) {
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    // §6.19.3.1: a CHROMA_PART leaf codes no luma mode, TX size or luma
    // residual; a LUMA_PART/SHARED leaf codes luma (and chroma only when it
    // owns a chroma block).
    const bool chroma_part = tree_type == 2;
    const bool do_luma = !chroma_part;
    if (options_.ibc) {
      ibc_stream::begin_ibc_block(ibc_buffer_, r, c);
    }

    ModeChoice y;
    palette::PaletteResult palette_result;
    bool use_palette = false;
    IbcChoice ibc_choice;
    bool use_ibc = false;
    std::vector<TxPlan> luma_plans;
    std::vector<syntax::LumaTxPart> luma_parts;
    int luma_partition = 0;
    if (do_luma) {
      if (!choose_luma_mode(x0, y0, w, h, y)) return false;

      // §5.20.5.5: the decoder sets IntraJointModes for the whole coding block
      // before residual(), so a later transform part inside the block reads the
      // current block's joint mode.  Update it before per-part prediction.  (For
      // the single-transform path this is equivalent to updating it afterwards.)
      const int joint = syntax::y_mode_joint(y.av2_mode, y.angle_delta);
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          intra_joint(rr, cc) = joint;
        }
      }

      // §5.20.8 palette candidate: luma-only and only when the luma mode is DC
      // (palette_mode_info is conditioned on YMode == DC_PRED).  A palette-only
      // block codes eob = 0 on all planes, so compare its prediction SSE with
      // the chosen intra prediction and keep it only when it is better.
      if (options_.palette && w >= 8 && h >= 8 && w <= 64 && h <= 64) {
        std::uint8_t cache[16] = {};
        const int cache_n = palette::get_palette_cache(palette_state_, r, c,
                                                       c > 0, cache);
        std::vector<std::uint8_t> pal_src(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        for (int yy = 0; yy < h; ++yy) {
          for (int xx = 0; xx < w; ++xx) {
            pal_src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(xx)] =
                source_sample(0, x0 + xx, y0 + yy);
          }
        }
        const int visible_w = std::min(w, mi_cols_ * 4 - x0);
        const int visible_h = std::min(h, mi_rows_ * 4 - y0);
        palette::PaletteParams palette_params;
        palette_params.max_colors = 8;
        palette_params.exact_when_possible = true;
        palette_params.choose_direction = true;
        std::string palette_error;
        if (!palette::analyze_palette(
                pal_src.data(), w, w, h, visible_w, visible_h, cache, cache_n,
                palette_params, palette_result, palette_error)) {
          error_ = "analyze_palette failed: " + palette_error;
          return false;
        }
        // Defensive: §5.20.8.1 codes a strictly increasing palette; a duplicate
        // would make the delta syntax illegal.  Fall back to intra prediction
        // instead of emitting a stream the decoder rejects.
        bool representable = palette_result.has_palette;
        for (int i = 1; i < palette_result.size && representable; ++i) {
          if (palette_result.colors[static_cast<std::size_t>(i)] <=
              palette_result.colors[static_cast<std::size_t>(i - 1)]) {
            representable = false;
          }
        }
        if (palette_result.has_palette && !representable) {
          palette_result.has_palette = false;
        }
        if (palette_result.has_palette && palette_result.sse < y.sse) {
          use_palette = true;
          y.mode = intra::Mode::kDc;
          y.angle_delta = 0;
          y.av2_mode = kDcPred;
          y.mrl_index = 0;
          y.mrl_sec_index = false;
          y.use_dip = false;
          y.dip_mode = 0;
          y.dip_transpose = false;
          y.sse = palette_result.sse;
        }
      }

      // §5.20.5.3/.4 intra block copy.  skip_flag = 1 codes no residual, so the
      // encoder only selects an exact copy (checked on all three planes with the
      // same `copy_block` arithmetic the decoder uses).  Blocks whose chroma
      // residual block is not the plain 4:2:0 half-size (shared-chroma
      // partitions) are skipped: §7.13.3.18's copy would not cover that block.
      if (options_.ibc) {
        const BlockSize ibc_chroma =
            partition::chroma_plane_residual_size(geometry_, size);
        if (partition::is_valid_block_size(ibc_chroma) &&
            partition::block_width(ibc_chroma) == (w / 2) &&
            partition::block_height(ibc_chroma) == (h / 2)) {
          if (!choose_ibc(r, c, size, ibc_choice)) return false;
          use_ibc = ibc_choice.found;
        }
      }
      if (use_ibc) {
        // read_intrabc_info() forces DC luma/chroma and clears palette/DIP
        // state.
        use_palette = false;
        y.mode = intra::Mode::kDc;
        y.angle_delta = 0;
        y.av2_mode = kDcPred;
        y.mrl_index = 0;
        y.mrl_sec_index = false;
        y.use_dip = false;
        y.dip_mode = 0;
        y.dip_transpose = false;
        y.sse = 0;
      }

      if (use_ibc) {
        std::vector<std::uint8_t> luma_pred;
        luma_pred.resize(static_cast<std::size_t>(w) *
                         static_cast<std::size_t>(h));
        const ibc::FrameView frame = ibc_frame_view();
        const int visible_w = std::min(w, mi_cols_ * 4 - x0);
        const int visible_h = std::min(h, mi_rows_ * 4 - y0);
        if (!ibc::copy_block(frame, r, c, static_cast<int>(size),
                             ibc_choice.bv, luma_pred.data(), w, nullptr, 0,
                             nullptr, 0, visible_w, visible_h)) {
          error_ = "IBC luma copy failed";
          return false;
        }
        const std::optional<txv::TxParams> params_opt =
            txv::make_tx_params(w, h, txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error_ = "unsupported IBC block transform size";
          return false;
        }
        TxPlan luma_plan;
        luma_plan.plane = 0;
        luma_plan.tx_size = params_opt->tx_size;
        luma_plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
        luma_plan.coded_w = w;
        luma_plan.coded_h = h;
        luma_plan.adj_w = params_opt->w;
        luma_plan.adj_h = params_opt->h;
        luma_plan.eob = 0;
        luma_plan.pred = luma_pred;
        luma_plan.recon = luma_pred;
        write_recon(0, x0, y0, w, h, luma_plan);
        luma_plans.push_back(std::move(luma_plan));
      } else if (use_palette) {
        std::vector<std::uint8_t> luma_pred;
        luma_pred.resize(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        palette::predict_palette_block(palette_result, w, h, luma_pred.data(),
                                       w);
        // Transform-skip: a palette-only block codes eob = 0 on every plane.
        const std::optional<txv::TxParams> params_opt =
            txv::make_tx_params(w, h, txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error_ = "unsupported palette block transform size";
          return false;
        }
        TxPlan luma_plan;
        luma_plan.plane = 0;
        luma_plan.tx_size = params_opt->tx_size;
        luma_plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
        luma_plan.coded_w = w;
        luma_plan.coded_h = h;
        luma_plan.adj_w = params_opt->w;
        luma_plan.adj_h = params_opt->h;
        luma_plan.eob = 0;
        luma_plan.pred = luma_pred;
        luma_plan.recon = luma_pred;
        write_recon(0, x0, y0, w, h, luma_plan);
        luma_plans.push_back(std::move(luma_plan));
      } else if (options_.tx_partition) {
        if (!choose_luma_tx_partition(r, c, w, h, size, y, &luma_plans,
                                      &luma_parts, &luma_partition)) {
          return false;
        }
      } else {
        std::vector<std::uint8_t> luma_pred;
        if (!predict_block(0, x0, y0, w, h, y.mode, y.angle_delta, luma_pred,
                           /*uv_is_cfl=*/false, y.mrl_index, y.mrl_sec_index,
                           y.use_dip, y.dip_mode, y.dip_transpose)) {
          return false;
        }
        TxPlan luma_plan;
        if (!compute_tx(0, x0, y0, w, h, luma_pred, luma_plan,
                        options_.tx_type || options_.idtx, y.av2_mode,
                        y.angle_delta, y.mrl_index)) {
          return false;
        }
        write_recon(0, x0, y0, w, h, luma_plan);
        luma_plans.push_back(std::move(luma_plan));
      }
    }

    if (do_luma) {
      // Record the final signalled luma mode over the whole coding block for
      // the CHROMA_PART trees (SDP codes luma before chroma).
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          const std::size_t cell =
              static_cast<std::size_t>(rr) *
                  static_cast<std::size_t>(mi_cols_) +
              static_cast<std::size_t>(cc);
          luma_mode_[cell] = static_cast<std::uint8_t>(y.av2_mode);
          luma_angle_[cell] = static_cast<std::int8_t>(y.angle_delta);
        }
      }
    }

    // §5.20.3.1: a sub-8 leaf that is not the chroma owner codes luma only;
    // the owner codes the whole inherited ChromaMi block.  For every leaf at
    // 8x8 or above ChromaMi is the leaf itself, so this reduces to the
    // historical behaviour.
    const BlockSize chroma_bsize = chroma.valid ? chroma.size : size;
    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, chroma_bsize);
    if (has_chroma && !partition::is_valid_block_size(chroma_size)) {
      error_ = "invalid chroma residual size";
      return false;
    }
    const int cw = partition::block_width(chroma_size);
    const int ch = partition::block_height(chroma_size);
    const int cx0 = chroma.valid ? chroma.mi_col * 2 : x0 / 2;
    const int cy0 = chroma.valid ? chroma.mi_row * 2 : y0 / 2;

    // §5.20.5.5/.6: YMode for a CHROMA_PART leaf is the co-located luma-tree
    // mode; it drives the UV CDF context, uv_mode_index and AngleDeltaUV.  For
    // SHARED/LUMA_PART leaves it is the leaf's own coded luma mode.
    const int uv_y_mode =
        do_luma ? y.av2_mode
                : luma_mode_at(chroma.valid ? chroma.mi_row : r,
                               chroma.valid ? chroma.mi_col : c);
    const int uv_y_angle_delta =
        do_luma ? y.angle_delta
                : luma_angle_at(chroma.valid ? chroma.mi_row : r,
                                chroma.valid ? chroma.mi_col : c);
    UvChoice uv;
    TxPlan u_plan;
    TxPlan v_plan;
    std::vector<std::uint8_t> u_pred;
    std::vector<std::uint8_t> v_pred;
    bool cfl = false;
    if (has_chroma) {
      if (use_ibc) {
        cctx_type_ = 0;
        // §7.13.3.17: the chroma block vector offset is the luma offset scaled
        // by the 4:2:0 subsampling.  The source region was already validated by
        // is_bv_encodable(), so this read is inside the coded plane.
        u_pred.resize(static_cast<std::size_t>(cw) *
                      static_cast<std::size_t>(ch));
        v_pred.resize(u_pred.size());
        const int cdx = ibc::plane_offset(ibc_choice.bv.col, 1);
        const int cdy = ibc::plane_offset(ibc_choice.bv.row, 1);
        const int sx = cx0 + cdx;
        const int sy = cy0 + cdy;
        for (int yy = 0; yy < ch; ++yy) {
          for (int xx = 0; xx < cw; ++xx) {
            u_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                    static_cast<std::size_t>(xx)] =
                recon_[1].at(sx + xx, sy + yy);
            v_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                    static_cast<std::size_t>(xx)] =
                recon_[2].at(sx + xx, sy + yy);
          }
        }
        for (int plane = 1; plane < 3; ++plane) {
          TxPlan& plan = plane == 1 ? u_plan : v_plan;
          const std::optional<txv::TxParams> params_opt =
              txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
          if (!params_opt.has_value()) {
            error_ = "unsupported IBC chroma transform size";
            return false;
          }
          plan.plane = plane;
          plan.tx_size = params_opt->tx_size;
          plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
          plan.coded_w = cw;
          plan.coded_h = ch;
          plan.adj_w = params_opt->w;
          plan.adj_h = params_opt->h;
          plan.eob = 0;
          plan.pred = plane == 1 ? u_pred : v_pred;
          plan.recon = plan.pred;
        }
      } else {
        if (!choose_uv_mode(cx0, cy0, cw, ch, uv_y_mode, uv_y_angle_delta,
                            uv)) {
          return false;
        }
        cfl = uv.is_cfl;
        if (cfl && (cw > 32 || ch > 32)) {
          // §7.13.2 dispatch: a CfL chroma block with w > 32 or h > 32 must use
          // the DC intra prediction subsampled process (§7.13.2.11), which this
          // encoder does not implement.  M1's largest chroma block is 32x32, so
          // this is a guard for a future format/SB configuration, not a path
          // the current search can select.
          error_ =
              "CfL chroma block exceeds 32 samples: §7.13.2.11 DC-subsampled "
              "prediction is not implemented";
          return false;
        }
        if (cfl) {
          // The CfL component expects the DC prediction in the chroma plane and
          // reconstructed above/left neighbours. Derive the implicit alpha and
          // write the CfL prediction into the planes.
          if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_pred,
                             /*uv_is_cfl=*/true)) {
            return false;
          }
          if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_pred,
                             /*uv_is_cfl=*/true)) {
            return false;
          }
          for (int yy = 0; yy < ch; ++yy) {
            for (int xx = 0; xx < cw; ++xx) {
              recon_[1].at(cx0 + xx, cy0 + yy) =
                  u_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                         static_cast<std::size_t>(xx)];
              recon_[2].at(cx0 + xx, cy0 + yy) =
                  v_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                         static_cast<std::size_t>(xx)];
            }
          }
          cfl::Scratch scratch;
          cfl::Block ub;
          cfl::Block vb;
          make_cfl_block(1, cx0, cy0, cw, ch, ub);
          make_cfl_block(2, cx0, cy0, cw, ch, vb);
          // The mode decision already chose between derived, explicit and MHCCP
          // hypotheses by SSE; apply exactly the chosen prediction so the
          // reconstruction matches the signalled syntax.
          cfl::compute_luma(ub, scratch);
          if (uv.cfl_index == 2) {
            const int num4_above_right =
                count_top_right(1, cx0 >> 2, cy0 >> 2, cw >> 2);
            const int num4_below_left =
                count_bottom_left(1, cx0 >> 2, cy0 >> 2, ch >> 2);
            av2enc::cfl::mhccp::Block mh_u = av2enc::cfl::mhccp::make_block(
                ub, num4_above_right, num4_below_left, mi_cols_ * 4,
                mi_rows_ * 4);
            av2enc::cfl::mhccp::Block mh_v = av2enc::cfl::mhccp::make_block(
                vb, num4_above_right, num4_below_left, mi_cols_ * 4,
                mi_rows_ * 4);
            av2enc::cfl::mhccp::apply_mhccp_pair(mh_u, mh_v, uv.cfl_mh_dir,
                                                 uv.mh_params_u, uv.mh_params_v);
          } else {
            cfl::apply_prediction(ub, uv.cfl_alpha_q8_u, scratch);
            cfl::apply_prediction(vb, uv.cfl_alpha_q8_v, scratch);
          }
          for (int yy = 0; yy < ch; ++yy) {
            for (int xx = 0; xx < cw; ++xx) {
              u_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                     static_cast<std::size_t>(xx)] = recon_[1].at(cx0 + xx, cy0 + yy);
              v_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                     static_cast<std::size_t>(xx)] = recon_[2].at(cx0 + xx, cy0 + yy);
            }
          }
        } else {
          if (!predict_block(1, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                             u_pred)) {
            return false;
          }
          if (!predict_block(2, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                             v_pred)) {
            return false;
          }
        }

        cctx_type_ = 0;
        if (options_.cctx) {
          if (!compute_chroma_cctx(cx0, cy0, cw, ch, u_pred, v_pred, u_plan,
                                   v_plan)) {
            return false;
          }
        } else {
          if (!compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan)) return false;
          if (!compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) return false;
        }
      }
      write_recon(1, cx0, cy0, cw, ch, u_plan);
      write_recon(2, cx0, cy0, cw, ch, v_plan);
    }

    if (trace_ && do_luma) {
      const TxPlan& luma_plan = luma_plans.front();
      std::cerr << "m1 block r=" << r << " c=" << c
                << " size=" << partition::block_size_name(size)
                << " y=" << y.av2_mode << "/" << y.angle_delta
                << " uv=" << (cfl ? -1 : uv.av2_mode) << "/"
                << uv.angle_delta << " cfl=" << (cfl ? 1 : 0)
                << " tx=" << luma_plan.coded_w << "x" << luma_plan.coded_h
                << " type=" << luma_plan.tx_type
                << " parts=" << luma_plans.size()
                << " eob=" << luma_plan.eob
                << " ist=" << luma_plan.sec_tx_type << "/"
                << luma_plan.most_probable_stx_set << "\n";
    }

    // Fill this leaf's syntax entry: modes, transform sizes, EOBs, and its
    // slice of the flat scan-order coefficient array.  No bits are written
    // here; syntax::assemble_still_frame() emits the frame after the tree is done.
    syntax::BlockSyntax& leaf = blocks_[index];
    leaf.flags |= syntax::kFlagHasChromaInfo;
    leaf.has_chroma = has_chroma ? 1 : 0;
    leaf.tree_type = static_cast<std::uint8_t>(tree_type);
    leaf.chroma_bsize = static_cast<std::uint8_t>(chroma_bsize);
    leaf.chroma_mi_row =
        static_cast<std::uint16_t>(chroma.valid ? chroma.mi_row : r);
    leaf.chroma_mi_col =
        static_cast<std::uint16_t>(chroma.valid ? chroma.mi_col : c);
    leaf.y_mode = static_cast<std::uint8_t>(uv_y_mode);
    leaf.y_angle_delta = static_cast<std::int8_t>(uv_y_angle_delta);
    leaf.mrl_index = static_cast<std::int8_t>(y.mrl_index);
    leaf.mrl_sec_index = static_cast<std::uint8_t>(y.mrl_sec_index ? 1 : 0);
    leaf.uv_mode = static_cast<std::uint8_t>(uv.av2_mode);
    if (cfl) leaf.flags |= syntax::kFlagIsCfl;
    leaf.cfl_index = static_cast<std::uint8_t>(uv.cfl_index);
    leaf.cfl_alpha_signs = static_cast<std::uint8_t>(uv.cfl_alpha_signs);
    leaf.cfl_alpha_u = static_cast<std::uint8_t>(uv.cfl_alpha_u);
    leaf.cfl_alpha_v = static_cast<std::uint8_t>(uv.cfl_alpha_v);
    leaf.cfl_mh_dir = static_cast<std::uint8_t>(uv.cfl_mh_dir);
    leaf.use_dip = static_cast<std::uint8_t>(y.use_dip ? 1 : 0);
    leaf.dip_transpose = static_cast<std::uint8_t>(y.dip_transpose ? 1 : 0);
    leaf.dip_mode = static_cast<std::uint8_t>(y.dip_mode);
    if (use_palette) {
      leaf.flags |= syntax::kFlagIsPalette;
      leaf.palette_index = static_cast<std::uint32_t>(palette_results_.size());
      palette::store_palette_state(palette_state_, r, c, w, h, palette_result);
      palette_results_.push_back(std::move(palette_result));
    }
    if (use_ibc) {
      leaf.flags |= syntax::kFlagIsIbc;
      leaf.ibc_skip = 1;
      leaf.ibc_mode = ibc_choice.stack_index >= 0 ? 1 : 0;
      leaf.ibc_ref_mv_idx =
          ibc_choice.stack_index >= 0
              ? static_cast<std::uint8_t>(ibc_choice.stack_index)
              : 0;
      leaf.ibc_bv_row = ibc_choice.bv.row;
      leaf.ibc_bv_col = ibc_choice.bv.col;
    }
    // Leaf-order screen-content decision plan for the GPU orchestrator.  One
    // entry per leaf, appended in the same preorder the GPU FramePlan uses.
    {
      std::uint32_t leaf_palette = StillResult::kNoPaletteLeaf;
      if (use_palette) {
        leaf_palette = static_cast<std::uint32_t>(palette_results_.size() - 1);
      }
      palette_index_by_leaf_.push_back(leaf_palette);
      StillResult::IbcPlan ibc_plan;
      if (use_ibc) {
        ibc_plan.used = 1;
        ibc_plan.mode = leaf.ibc_mode;
        ibc_plan.ref_mv_idx = leaf.ibc_ref_mv_idx;
        ibc_plan.bv_row = leaf.ibc_bv_row;
        ibc_plan.bv_col = leaf.ibc_bv_col;
      }
      ibc_by_leaf_.push_back(ibc_plan);
    }
    const TxPlan* plans[3] = {luma_plans.empty() ? nullptr : luma_plans.data(),
                              &u_plan, &v_plan};
    leaf.levels_offset = static_cast<std::uint32_t>(levels_.size());
    leaf.sec_tx_type =
        do_luma ? static_cast<std::uint8_t>(luma_plans.front().sec_tx_type) : 0;
    leaf.most_probable_stx_set =
        do_luma
            ? static_cast<std::uint8_t>(luma_plans.front().most_probable_stx_set)
            : 0;
    leaf.cctx_type = static_cast<std::uint8_t>(cctx_type_);
    if (options_.tx_partition && do_luma) {
      leaf.tx_part_count = static_cast<std::uint8_t>(luma_parts.size());
      leaf.tx_partition = static_cast<std::uint8_t>(luma_partition);
      for (std::size_t i = 0; i < luma_parts.size(); ++i) {
        leaf.tx_parts[i] = luma_parts[i];
      }
      for (const TxPlan& plan : luma_plans) append_levels(plan);
      leaf.tx_size[0] = static_cast<std::uint8_t>(luma_plans.front().tx_size);
      leaf.tx_type[0] = static_cast<std::uint8_t>(luma_plans.front().tx_type);
      leaf.eob[0] = static_cast<std::uint32_t>(luma_plans.front().eob);
    } else if (do_luma) {
      leaf.tx_size[0] = static_cast<std::uint8_t>(luma_plans.front().tx_size);
      leaf.tx_type[0] = static_cast<std::uint8_t>(luma_plans.front().tx_type);
      leaf.eob[0] = static_cast<std::uint32_t>(luma_plans.front().eob);
      append_levels(luma_plans.front());
    } else {
      // A CHROMA_PART leaf codes no luma transform.
      leaf.tx_size[0] = 0;
      leaf.tx_type[0] = 0;
      leaf.eob[0] = 0;
    }
    for (int plane = 1; plane < 3; ++plane) {
      const TxPlan& plan = *plans[plane];
      leaf.tx_size[plane] = static_cast<std::uint8_t>(plan.tx_size);
      leaf.tx_type[plane] = static_cast<std::uint8_t>(plan.tx_type);
      leaf.eob[plane] = static_cast<std::uint32_t>(plan.eob);
      append_levels(plan);
    }

    // intra_joint was updated before the residual so per-part prediction sees
    // this coding block's mode; only uv_smooth remains to record here.
    const int cr = chroma.valid ? chroma.mi_row / 2 : r / 2;
    const int cc0 = chroma.valid ? chroma.mi_col / 2 : c / 2;
    const int cw4 = cw / 4;
    const int ch4 = ch / 4;
    const int chroma_mi_rows = mi_rows_ / 2;
    const int chroma_mi_cols = mi_cols_ / 2;
    if (has_chroma) {
      for (int yy = 0; yy < ch4; ++yy) {
        const int rr = cr + yy;
        if (rr >= chroma_mi_rows) break;
        for (int xx = 0; xx < cw4; ++xx) {
          const int cc = cc0 + xx;
          if (cc >= chroma_mi_cols) break;
          uv_smooth(rr, cc) =
              (!cfl && y_mode_is_smooth(uv.av2_mode)) ? 1 : 0;
        }
      }
    }

    if (do_luma) {
      decoded_[0].fill(c, r, w4, h4);
    }
    if (has_chroma) {
      decoded_[1].fill(cc0, cr, cw4, ch4);
      decoded_[2].fill(cc0, cr, cw4, ch4);
    }

    if (options_.enable_deblock) {
      const int chroma_mi_row = chroma.valid ? chroma.mi_row : r;
      const int chroma_mi_col = chroma.valid ? chroma.mi_col : c;
      if (options_.tx_partition && do_luma) {
        fill_deblock_mi_parts(r, c, size, cw, ch, luma_parts, luma_partition,
                              u_plan.tx_size, has_chroma, chroma_mi_row,
                              chroma_mi_col, chroma_bsize, do_luma);
      } else {
        fill_deblock_mi(
            r, c, size, cw, ch,
            do_luma ? luma_plans.front().tx_size
                    : static_cast<int>(u_plan.tx_size),
            u_plan.tx_size, has_chroma, chroma_mi_row, chroma_mi_col,
            chroma_bsize, do_luma);
      }
    }
    if (use_ibc) ++ibc_blocks_;
    if (options_.ibc) {
      ibc_stream::finish_ibc_block(
          ibc_buffer_, ibc_neighbours_, r, c, static_cast<int>(size),
          use_ibc, use_ibc ? ibc_choice.bv : ibc::BlockVector{});
    }
    return true;
  }

void Encoder::make_cfl_block(int plane, int cx0, int cy0, int cw, int ch,
                    cfl::Block& block) {
    block.luma = recon_[0].samples.data();
    block.luma_stride = recon_[0].width;
    block.luma_max_x = mi_cols_ * 4 - 1;
    block.luma_max_y = mi_rows_ * 4 - 1;
    block.sb_top_y = (cy0 * 2 / geometry_.sb_size) * geometry_.sb_size;
    block.chroma = recon_[plane].samples.data();
    block.chroma_stride = recon_[plane].width;
    block.x = cx0;
    block.y = cy0;
    block.w = cw;
    block.h = ch;
    block.coded_width = mi_cols_ * 2;
    block.coded_height = mi_rows_ * 2;
    block.have_above = cfl::cfl_avail_above(cy0, 1) ? 1 : 0;
    block.have_left = cfl::cfl_avail_left(cx0, 1) ? 1 : 0;
    block.bit_depth = 8;
    block.ds_filter = options_.cfl_ds_filter;
    block.sub_x = 1;
    block.sub_y = 1;
  }

bool Encoder::choose_uv_mode(int cx0, int cy0, int cw, int ch, int y_mode,
                    int y_angle_delta, UvChoice& best) {
    best.is_cfl = false;
    best.sse = std::numeric_limits<std::int64_t>::max();
    const bool uv_full =
        options_.uv_full || std::getenv("AV2ENC_STILL_UV_FULL") != nullptr;
    if (std::getenv("AV2ENC_STILL_UV_DC") != nullptr) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    // Debug: force one chroma mode for decoder verification.
    if (const char* forced = std::getenv("AV2ENC_STILL_FORCE_UV")) {
      int mode = 0;
      int delta = 0;
      if (std::sscanf(forced, "%d:%d", &mode, &delta) == 2 &&
          mode >= 0 && mode <= 12 && delta >= -3 && delta <= 3) {
        // Only UVMode == YMode can carry a non-zero AngleDeltaUV, and the
        // decoder derives it from AngleDeltaY; the explicit delta here is
        // accepted only for that case.
        const int derived =
            (y_mode_is_directional(y_mode) && mode == y_mode)
                ? y_angle_delta
                : 0;
        if (delta != 0 && delta != derived) {
          error_ = "AV2ENC_STILL_FORCE_UV non-zero delta requires UVMode==YMode";
          return false;
        }
        best.mode = static_cast<intra::Mode>(mode);
        best.angle_delta = derived;
        best.av2_mode = av2_y_mode(best.mode);
        best.sse = 0;
        return true;
      }
      error_ = "invalid AV2ENC_STILL_FORCE_UV (expected mode:delta)";
      return false;
    }
    if (options_.force_dc || (!options_.enable_cfl && !uv_full)) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    std::vector<std::uint8_t> pred;

    // Try the 13 base modes. AngleDeltaUV is not signalled: §6.19.5.6
    // derives it as AngleDeltaY only when UVMode == YMode and YMode is
    // directional; every other combination uses 0. There is no separate UV
    // angle-delta syntax to search over.
    static const intra::Mode kBase[13] = {
        intra::Mode::kDc,      intra::Mode::kV,       intra::Mode::kH,
        intra::Mode::kD45,     intra::Mode::kD135,    intra::Mode::kD113,
        intra::Mode::kD157,    intra::Mode::kD203,    intra::Mode::kD67,
        intra::Mode::kSmooth,  intra::Mode::kSmoothV, intra::Mode::kSmoothH,
        intra::Mode::kPaeth};

    for (int i = 0; i < 13; ++i) {
      const intra::Mode mode = kBase[i];
      // §6.19.5.6: if UVMode == YMode and YMode is directional, the decoder
      // derives AngleDeltaUV = AngleDeltaY. AngleDeltaUV is not signalled, so
      // evaluate and store that derived delta for this candidate. All other
      // UV modes use AngleDeltaUV = 0.
      const int delta =
          (y_mode_is_directional(y_mode) &&
           static_cast<int>(mode) == y_mode)
              ? y_angle_delta
              : 0;
      // M1 default: chroma is DC or CfL. Directional chroma prediction is
      // enabled only behind uv_full (--still-uv-full) until forced-UV decoder
      // verification passes.
      if (!uv_full && mode != intra::Mode::kDc) continue;
      if (!predict_block(1, cx0, cy0, cw, ch, mode, delta, pred)) return false;
      std::int64_t sse = sse_against_source(1, cx0, cy0, cw, ch, pred);
      if (uv_full) {
        // Score both chroma planes for the full search so a U-only win cannot
        // pick a mode that is worse for V. The default DC/CfL path keeps the
        // historical U-only DC score for decision parity.
        if (!predict_block(2, cx0, cy0, cw, ch, mode, delta, pred)) return false;
        sse += sse_against_source(2, cx0, cy0, cw, ch, pred);
      }
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = mode;
        best.angle_delta = delta;
        best.av2_mode = av2_y_mode(mode);
      }
    }

    if (options_.enable_cfl) {
      std::vector<std::uint8_t> u_dc;
      std::vector<std::uint8_t> v_dc;
      // The CfL base prediction must not receive the IBP DC post-filter
      // (§7.13.2.12 excludes uv_is_cfl blocks).
      if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_dc,
                         /*uv_is_cfl=*/true)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_dc,
                         /*uv_is_cfl=*/true)) {
        return false;
      }
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          recon_[1].at(cx0 + xx, cy0 + yy) =
              u_dc[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(xx)];
          recon_[2].at(cx0 + xx, cy0 + yy) =
              v_dc[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(xx)];
        }
      }
      cfl::Scratch scratch;
      cfl::Block ub;
      cfl::Block vb;
      make_cfl_block(1, cx0, cy0, cw, ch, ub);
      make_cfl_block(2, cx0, cy0, cw, ch, vb);
      cfl::compute_luma(ub, scratch);
      int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
      cfl::compute_luma(vb, scratch);
      int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
      cfl::apply_prediction(ub, alpha_u, scratch);
      cfl::apply_prediction(vb, alpha_v, scratch);
      std::int64_t sse = 0;
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          const int su = source_sample(1, cx0 + xx, cy0 + yy);
          const int pu = recon_[1].at(cx0 + xx, cy0 + yy);
          const int du = su - pu;
          sse += static_cast<std::int64_t>(du) * du;
          const int sv = source_sample(2, cx0 + xx, cy0 + yy);
          const int pv = recon_[2].at(cx0 + xx, cy0 + yy);
          const int dv = sv - pv;
          sse += static_cast<std::int64_t>(dv) * dv;
        }
      }
      int cfl_index = 1;
      av2enc::cfl::Signalling explicit_syntax{};
      if (options_.cfl_explicit) {
        // §5.20.7.32 CFL_EXPLICIT: exhaustive sign/magnitude search on the
        // source planes.  The derived prediction above is the incumbent, so a
        // tie keeps cfl_index = 1 and the historical stream.
        cfl::SourcePlane su;
        su.plane = picture_.u.data();
        su.stride = chroma_width_;
        su.max_x = chroma_width_ - 1;
        su.max_y = chroma_height_ - 1;
        cfl::SourcePlane sv;
        sv.plane = picture_.v.data();
        sv.stride = chroma_width_;
        sv.max_x = chroma_width_ - 1;
        sv.max_y = chroma_height_ - 1;
        const std::optional<cfl::SearchResult> explicit_choice =
            cfl::search_alpha(ub, vb, su, sv,
                              cfl::SearchMode::kExhaustiveSse, scratch);
        if (explicit_choice.has_value()) {
          cfl::apply_prediction_pair(ub, vb, explicit_choice->alpha, scratch);
          std::int64_t explicit_sse = 0;
          for (int yy = 0; yy < ch; ++yy) {
            for (int xx = 0; xx < cw; ++xx) {
              const int du = source_sample(1, cx0 + xx, cy0 + yy) -
                             recon_[1].at(cx0 + xx, cy0 + yy);
              explicit_sse += static_cast<std::int64_t>(du) * du;
              const int dv = source_sample(2, cx0 + xx, cy0 + yy) -
                             recon_[2].at(cx0 + xx, cy0 + yy);
              explicit_sse += static_cast<std::int64_t>(dv) * dv;
            }
          }
          if (explicit_sse < sse) {
            sse = explicit_sse;
            cfl_index = 0;
            explicit_syntax = explicit_choice->syntax;
            alpha_u = cfl::alpha_q8(explicit_choice->alpha.u);
            alpha_v = cfl::alpha_q8(explicit_choice->alpha.v);
          } else {
            // Restore the derived prediction that won the comparison.
            cfl::compute_luma(ub, scratch);
            cfl::apply_prediction(ub, cfl::derive_implicit_alpha(ub, scratch),
                                  scratch);
            cfl::apply_prediction(vb, cfl::derive_implicit_alpha(vb, scratch),
                                  scratch);
          }
        }
      }
      int cfl_mh_dir = 0;
      av2enc::cfl::mhccp::MultiParams mh_params_u{};
      av2enc::cfl::mhccp::MultiParams mh_params_v{};
      if (options_.mhccp &&
          av2enc::cfl::mhccp::allowed(cw, ch, /*enable_mhccp=*/true,
                                      /*is_chroma_part=*/false,
                                      /*frame_is_intra=*/true,
                                      /*cfl_allowed_in_sdp=*/false,
                                      /*lossless=*/false)) {
        // §7.13.6-.8: a third hypothesis with derived Q16 weights.  Only
        // cfl_mh_dir is signalled.  Compared against the incumbent by SSE.
        const int num4_above_right =
            count_top_right(1, cx0 >> 2, cy0 >> 2, cw >> 2);
        const int num4_below_left =
            count_bottom_left(1, cx0 >> 2, cy0 >> 2, ch >> 2);
        av2enc::cfl::mhccp::Block mh_u = av2enc::cfl::mhccp::make_block(
            ub, num4_above_right, num4_below_left, mi_cols_ * 4,
            mi_rows_ * 4);
        av2enc::cfl::mhccp::Block mh_v = av2enc::cfl::mhccp::make_block(
            vb, num4_above_right, num4_below_left, mi_cols_ * 4,
            mi_rows_ * 4);
        av2enc::cfl::mhccp::SourcePlane mh_su;
        mh_su.plane = picture_.u.data();
        mh_su.stride = chroma_width_;
        mh_su.max_x = chroma_width_ - 1;
        mh_su.max_y = chroma_height_ - 1;
        av2enc::cfl::mhccp::SourcePlane mh_sv;
        mh_sv.plane = picture_.v.data();
        mh_sv.stride = chroma_width_;
        mh_sv.max_x = chroma_width_ - 1;
        mh_sv.max_y = chroma_height_ - 1;
        const std::optional<av2enc::cfl::mhccp::Candidate> multi =
            av2enc::cfl::mhccp::search_mhccp(mh_u, mh_v, mh_su, mh_sv);
        if (multi.has_value()) {
          av2enc::cfl::mhccp::apply_mhccp_pair(mh_u, mh_v, multi->mh_dir,
                                               multi->params_u,
                                               multi->params_v);
          if (multi->total_sse() < sse) {
            sse = multi->total_sse();
            cfl_index = 2;
            cfl_mh_dir = multi->mh_dir;
            mh_params_u = multi->params_u;
            mh_params_v = multi->params_v;
            alpha_u = 0;
            alpha_v = 0;
            explicit_syntax = av2enc::cfl::Signalling{};
          } else {
            // Restore the incumbent prediction.
            cfl::compute_luma(ub, scratch);
            if (cfl_index == 0) {
              cfl::apply_prediction(ub, alpha_u, scratch);
              cfl::apply_prediction(vb, alpha_v, scratch);
            } else {
              cfl::apply_prediction(ub, cfl::derive_implicit_alpha(ub, scratch),
                                    scratch);
              cfl::apply_prediction(vb, cfl::derive_implicit_alpha(vb, scratch),
                                    scratch);
            }
          }
        }
      }
      if (sse < best.sse) {
        best.sse = sse;
        best.is_cfl = true;
        best.mode = intra::Mode::kDc;
        best.angle_delta = 0;
        best.av2_mode = kDcPred;
        best.cfl_index = cfl_index;
        best.cfl_alpha_signs = explicit_syntax.alpha_signs;
        best.cfl_alpha_u = explicit_syntax.alpha_u;
        best.cfl_alpha_v = explicit_syntax.alpha_v;
        best.cfl_alpha_q8_u = alpha_u;
        best.cfl_alpha_q8_v = alpha_v;
        best.cfl_mh_dir = cfl_mh_dir;
        best.mh_params_u = mh_params_u;
        best.mh_params_v = mh_params_v;
      }
    }
    return true;
  }

}  // namespace detail
}  // namespace av2enc::still
