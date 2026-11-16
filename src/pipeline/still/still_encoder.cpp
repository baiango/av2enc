// Integrated M1 still-picture encoder: entry point and frame assembly.
//
// The shared declaration and the component definitions live in
// still_encoder_internal.h and the still_encoder_*.cpp role files; this
// translation unit keeps the constructor, the partition-tree walk and
// encode_still_picture(), which frames the picture and runs the normative
// filters and header writers.  The M0 path in
// src/encoder/basic_encoder.cpp and the §8.2 entropy coder are untouched.
#include "pipeline/still/still_encoder_internal.h"

namespace av2enc::still {
namespace detail {

Encoder::Encoder(const StillPicture& picture, const StillOptions& options,
        const dcdeltaq::QuantFrameState* delta_quant)
    : picture_(picture), options_(options), delta_quant_(delta_quant) {
    width_ = static_cast<int>(picture.width);
    height_ = static_cast<int>(picture.height);
    chroma_width_ = (width_ + 1) / 2;
    chroma_height_ = (height_ + 1) / 2;
    // §7.14.2 qlookup(base_q_idx), 8-bit.  At the historical M1 operating point
    // (90) this is exactly txv::kQuantStep, so the pinned path is unchanged.
    quant_step_ = rc::kQlookupQ0To255[options.base_q_idx];
    // §8.3.1 coefficient-CDF qctx slice for the RDOQ rate model.  The syntax
    // writer selects the same slice from the same raw base_q_idx.
    qctx_slice_ = qctx::slice_for_base_q_idx(options.base_q_idx);

    // §5.18.6.2: one picture-level QM set (pic_qm_num_minus_1 = 0) with the
    // same spec-defined default matrix level for all planes and the single
    // segment.  >8x8 blocks read qm_y/u/v[0]; <=8x8 blocks read
    // SegQMLevel[plane][0].  No quantizer_matrix_obu() is emitted, so
    // QmDataPresent[] stays 0 and only the normative default tables apply.
    if (options_.qm) {
      qm_state_.using_qmatrix = true;
      qm_state_.pic_qm_num_minus_1 = 0;
      const auto level = static_cast<std::uint8_t>(options_.qm_level);
      qm_state_.qm_y[0] = level;
      qm_state_.qm_u[0] = level;
      qm_state_.qm_v[0] = level;
      for (int plane = 0; plane < 3; ++plane) {
        qm_state_.seg_qm_level[plane][0] = level;
      }
      if (options_.user_qm) {
        // §5.4.11: a real user-defined matrix for the selected level.  All-32
        // is the identity (Round2(q * 32, 5) == q), so the §7.14.4 dequant is
        // unchanged while the stream carries genuine user QM data.
        qm_state_.qm_data_present[level] = true;
        qm::fill_user_qm_constant(&qm_state_.user_qm[level], 32);
      }
    }

    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    sb_mi_ = partition::sb_size_mi(geometry_);
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;

    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    const int coded_luma_w = mi_cols_ * 4;
    const int coded_luma_h = mi_rows_ * 4;
    // The decoder reconstructs a whole coding/transform block even when it
    // extends past the MiCols*4 x MiRows*4 coded plane. Allocate the usual
    // 64-sample superblock padding so edge blocks stay in bounds; prediction
    // still clamps to the coded plane (max_x/max_y below).
    luma_stride_ = (coded_luma_w + 63) & ~63;
    luma_height_ = (coded_luma_h + 63) & ~63;
    chroma_stride_ = luma_stride_ / 2;
    chroma_padded_height_ = luma_height_ / 2;

    // dav2d initializes CurrFrame to 0. CfL's L/lumaAvg read the reconstructed
    // luma below/right of the current block (clamped only at block edges), so
    // zero padding is part of the encoder/decoder contract; 128 shifts CfL
    // predictions by 1 LSB and then cascades.
    recon_[0].resize(luma_stride_, luma_height_, 0);
    recon_[1].resize(chroma_stride_, chroma_padded_height_, 0);
    recon_[2].resize(chroma_stride_, chroma_padded_height_, 0);

    intra_joint_.assign(
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_),
        0);
    // §5.20.5.5: a CHROMA_PART leaf reads YMode/AngleDeltaY from the co-located
    // luma tree at the ChromaMi origin.  Keep the decided luma modes so the
    // chroma tree (coded after luma in SDP) can reuse them for the UV CDF
    // context, uv_mode_index and the derived AngleDeltaUV.
    luma_mode_.assign(static_cast<std::size_t>(mi_rows_) *
                          static_cast<std::size_t>(mi_cols_),
                      kDcPred);
    luma_angle_.assign(static_cast<std::size_t>(mi_rows_) *
                           static_cast<std::size_t>(mi_cols_),
                       0);
    uv_smooth_.assign(
        static_cast<std::size_t>(mi_rows_ / 2) *
            static_cast<std::size_t>(mi_cols_ / 2),
        0);

    for (int plane = 0; plane < 3; ++plane) {
      decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                            plane == 0 ? mi_rows_ : mi_rows_ / 2);
    }

    if (options_.palette) {
      palette_state_.reset(mi_cols_, mi_rows_);
    }
    if (options_.ibc) {
      ibc_buffer_.reset(mi_cols_, mi_rows_);
      ibc_neighbours_.reset(mi_cols_, mi_rows_);
      ibc_config_ = ibc::IbcConfig{};
      ibc_config_.sb_size = geometry_.sb_size;
      ibc_config_.allow_global = true;
      // The M1 frame header signals the global region only; the four-buffer
      // local ring is not modelled by the writer.
      ibc_config_.allow_local = false;
      ibc_config_.max_bvp_drl_bits_minus_1 = 0;
    }

    if (options_.enable_deblock) {
      deblock_cols_ = mi_cols_;
      deblock_rows_ = mi_rows_;
      deblock::DeblockMi neutral;
      neutral.qindex = options_.base_q_idx;
      for (int plane = 0; plane < 3; ++plane) {
        deblock_mi_[plane].assign(
            static_cast<std::size_t>(mi_cols_) *
                static_cast<std::size_t>(mi_rows_),
            neutral);
      }
    }
  }

bool Encoder::encode(SyntaxStream* stream, std::string& error) {
    FrameGeometry geometry = geometry_;
    const double lambda = 0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;
    // Sub-8 opt-in: the 4-sample floor needs the full rect recursion depth.
    const int min_block_px = options_.sub8 ? 4 : options_.min_block_size;
    const int max_depth =
        options_.sub8
            ? std::max(options_.max_partition_depth, 8)
            : options_.max_partition_depth;

    // Shared leaf-cost policy: SSE block variance plus the writer's 64x64 leaf
    // limit (one TX_MODE_LARGEST transform block per plane).  Both solvers use
    // it so rect-only choices do not drift.
    const auto leaf_cost = [&](int mi_row, int mi_col, BlockSize size) {
      double cost = static_cast<double>(
          block_variance_cost_at(mi_row, mi_col, size));
      if (partition::block_width(size) > 64 ||
          partition::block_height(size) > 64) {
        cost += 1.0e18;
      }
      if (std::getenv("AV2ENC_STILL_SQUARE_ONLY") != nullptr &&
          partition::block_width(size) != partition::block_height(size)) {
        cost += 1.0e18;
      }
      if (std::getenv("AV2ENC_STILL_LEAF32") != nullptr &&
          (partition::block_width(size) > 32 ||
           partition::block_height(size) > 32)) {
        cost += 1.0e18;
      }
      return cost;
    };

    if (options_.sdp) {
      sdpp::Tools tools;
      tools.enable_sdp = true;
      tools.frame_is_intra = true;
      tools.single_picture_header_flag = true;
      sdpp::DecisionOptions sopt;
      // First decoder-gated configuration: every intra SDP 64x64 split carries
      // a forced PARTITION_NONE luma tree and a forced PARTITION_NONE chroma
      // tree.  The two-tree walk, PlaneStart-selected CDFs and chroma ownership
      // are exercised; searching the sub-trees is the next step.
      sopt.min_block_size_px = 64;
      sopt.max_depth = 0;
      sopt.allow_rect = false;
      sopt.allow_split = false;
      sopt.allow_3way = false;
      sopt.allow_uneven4 = false;
      sopt.allow_sdp = true;
      sopt.allow_extended_sdp = false;
      const auto sdp_cost = [&](const sdpp::NodeInfo& info) {
        sdpp::NodeCosts costs;
        if (info.tree_type == sdpp::TreeType::kChromaPart) {
          costs.leaf = 0.0;
        } else {
          costs.leaf = leaf_cost(info.mi_row, info.mi_col, info.size);
        }
        costs.none_signal = 0.0;
        costs.horz_signal = sdpp::kInfiniteCost;
        costs.vert_signal = sdpp::kInfiniteCost;
        costs.split_signal = sdpp::kInfiniteCost;
        return costs;
      };
      const sdpp::FrameDecision frame =
          sdpp::decide_frame(geometry, tools, sdp_cost, sopt);
      if (frame.status != sdpp::DecisionStatus::kOk) {
        error = "SDP partition decision failed: " + frame.error;
        return false;
      }
      if (!sdpp::validate_frame(geometry, tools, frame, error)) {
        error = "SDP partition tree invalid: " + error;
        return false;
      }
      leaves_ = frame.leaves;
      blocks_.clear();
      levels_.clear();
      for (const sdpp::Node& sb : frame.superblocks) {
        begin_superblock(sb);
        if (!collect_sdp_node(sb)) {
          error = error_;
          return false;
        }
      }
      stream->blocks = std::move(blocks_);
      stream->levels = std::move(levels_);
      return true;
    }

    if (options_.ext_partitions) {
      pe::Tools tools;
      tools.enable_ext_partitions = true;
      tools.enable_uneven_4way_partitions = true;
      pe::DecisionOptions eopt;
      // The extended families include 1/8 subsizes: HORZ_4A@32x32 is 32x4.
      // Chroma ownership for those shared-chroma cases is wired, so the ext
      // search uses the 4-sample floor.
      eopt.min_block_size_px = 4;
      eopt.max_depth = max_depth;
      eopt.allow_rect = true;
      eopt.allow_split = geometry_.sb_size == 128;
      eopt.allow_3way = true;
      eopt.allow_uneven4 = true;
      // With the extended families unavailable this order reproduces
      // partition::decide_frame's tie-breaks exactly.
      eopt.tie_break = pe::TieBreakOrder::kRectCompatOrder;
      if (options_.force_single_block) {
        eopt.max_depth = 0;
        eopt.min_block_size_px = 64;
      }
      const auto cost_fn = [&](const pe::NodeInfo& info) {
        pe::NodeCosts costs;
        costs.leaf = leaf_cost(info.mi_row, info.mi_col, info.size);
        costs.none_signal = 0.0;
        costs.split_signal = signal_cost;
        costs.horz_signal = signal_cost;
        costs.vert_signal = signal_cost;
        const int w = partition::block_width(info.size);
        const int h = partition::block_height(info.size);
        // Chroma ownership is now carried by the leaf (ChromaMi), so the
        // shared 8x16/16x8 middle block of a 3-way split and the first
        // children of a 4-way split are codable.  The spec's allowed-set
        // validation rejects every geometrically impossible candidate; these
        // thresholds only keep the unexpectedly small sizes out of the search.
        costs.horz3_signal = (h >= 32 && w >= 8) ? signal_cost
                                                  : pe::kInfiniteCost;
        costs.vert3_signal = (w >= 32 && h >= 8) ? signal_cost
                                                  : pe::kInfiniteCost;
        const double h4 = (h >= 32 && w >= 8) ? signal_cost : pe::kInfiniteCost;
        // Vertical uneven 4-way with a 4-wide 1/8 child still produces a
        // tile desync at the bottom edge (reproducer: 640x40 vertical
        // stripes); keep it at 8-wide subsizes until that is fixed.
        const double v4 = (w >= 64 && h >= 8) ? signal_cost
                                               : pe::kInfiniteCost;
        costs.horz4a_signal = h4;
        costs.horz4b_signal = h4;
        costs.vert4a_signal = v4;
        costs.vert4b_signal = v4;
        return costs;
      };
      const pe::FrameDecision frame =
          pe::decide_frame(geometry, cost_fn, tools, eopt);
      if (frame.status != pe::DecisionStatus::kOk) {
        error = "extended partition decision failed: " + frame.error;
        return false;
      }
      if (!pe::validate_frame(geometry, tools, frame, error)) {
        error = "extended partition tree invalid: " + error;
        return false;
      }
      leaves_ = frame.leaves;
      blocks_.clear();
      levels_.clear();
      for (const pe::PartitionNode& sb : frame.superblocks) {
        begin_superblock(sb);
        if (!collect_node_ext(sb)) {
          error = error_;
          return false;
        }
      }
      stream->blocks = std::move(blocks_);
      stream->levels = std::move(levels_);
      return true;
    }

    partition::DecisionOptions poptions;
    poptions.min_block_size_px = min_block_px;
    poptions.max_depth = max_depth;
    poptions.allow_rect = true;
    // PARTITION_SPLIT exists only at a 128x128 (or 256x256) node.  The 64x64
    // superblock keeps the historical rect-only search.
    poptions.allow_split = geometry_.sb_size == 128;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }

    const auto cost_fn = [&](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = leaf_cost(info.mi_row, info.mi_col, info.size);
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };

    const partition::FrameDecision frame =
        partition::decide_frame(geometry, cost_fn, poptions);
    if (frame.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame.error;
      return false;
    }
    if (!partition::validate_frame(geometry, frame, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    leaves_ = frame.leaves;

    blocks_.clear();
    levels_.clear();
    for (const partition::PartitionNode& sb : frame.superblocks) {
      begin_superblock(sb);
      if (!collect_node(sb, ChromaInfo{})) {
        error = error_;
        return false;
      }
    }

    stream->blocks = std::move(blocks_);
    stream->levels = std::move(levels_);
    return true;
  }

}  // namespace detail
  // namespace

bool encode_still_picture(const StillPicture& picture,
                             const StillOptions& options, StillResult& result,
                             std::string& error) {
  error.clear();
  if (picture.width == 0 || picture.height == 0) {
    error = "empty picture";
    return false;
  }
  if ((picture.width & 1U) != 0U || (picture.height & 1U) != 0U) {
    error = "M1 4:2:0 requires even dimensions";
    return false;
  }
  // base_q_idx 1..255 is supported: the coefficient CDFs come from the §8.3.1
  // qctx slice for the frame QP and the quantiser uses qlookup(base_q_idx).
  // 0 is lossless, which this path does not implement.
  if (options.base_q_idx == 0) {
    error = "M1 does not support lossless base_q_idx = 0";
    return false;
  }
  // CDEF's block-index context and CCSO's unit size are still derived from a
  // 64x64 superblock.  Reject them by name instead of emitting a desynchronised
  // stream.
  if (options.sb128 && options.sb256) {
    error = "sb128 and sb256 are mutually exclusive";
    return false;
  }
  if (options.ext_partitions && options.force_single_block) {
    error = "--still-ext-partitions cannot be combined with --still-single";
    return false;
  }
  if (options.sdp) {
    if (options.sb128 || options.sb256) {
      error = "--still-sdp is only wired for 64x64 superblocks";
      return false;
    }
    if (options.ext_partitions || options.sub8) {
      error = "--still-sdp is not wired with extended partitions or sub-8";
      return false;
    }
    if (options.enable_cdef || options.enable_ccso) {
      error = "--still-sdp with CDEF/CCSO is not wired yet";
      return false;
    }
    if (options.enable_segmentation || options.enable_delta_q) {
      error = "--still-sdp with segmentation/delta-Q is not wired yet";
      return false;
    }
  }
  if (options.delta_q_res < 0 || options.delta_q_res > 3) {
    error = "delta_q_res must be in [0, 3]";
    return false;
  }
  if (options.qm) {
    if (options.qm_level < 0 || options.qm_level > 14) {
      error = "--still-qm-level must be in 0..14 (15 is the no-QM sentinel)";
      return false;
    }
    if (options.rdoq) {
      error = "--still-qm is not supported with --still-rdoq yet";
      return false;
    }
  }
  if (options.cctx) {
    if (options.qm) {
      error = "--still-cctx is not supported with --still-qm yet";
      return false;
    }
    if (options.delta_q) {
      error = "--still-cctx is not supported with --still-dc-ac-delta-q yet";
      return false;
    }
  }
  if (options.tx_partition) {
    if (options.qm) {
      error = "--still-tx-partition is not supported with --still-qm yet";
      return false;
    }
    if (options.delta_q) {
      error =
          "--still-tx-partition is not supported with --still-dc-ac-delta-q "
          "yet";
      return false;
    }
    // These change the intra prediction of the coding block (reference-line
    // selection, DIP tap geometry, IBP post-filter), which is applied per
    // transform part.  Reject instead of guessing the per-part semantics.
    if (options.enable_dip) {
      error = "--still-tx-partition is not supported with --still-dip yet";
      return false;
    }
    if (options.enable_mrl) {
      error = "--still-tx-partition is not supported with --still-mrl yet";
      return false;
    }
    if (options.enable_ibp) {
      error = "--still-tx-partition is not supported with --still-ibp yet";
      return false;
    }
  }
  if (options.ddt) {
    error =
        "--still-ddt is not signallable in a still picture: §5.4.8 omits "
        "enable_inter_ddt when single_picture_header_flag = 1 and §7.15.4.1 "
        "applies DDT only to inter blocks; use the M2 inter residual path "
        "(src/codec/transform/ddt/gate)";
    return false;
  }
  if (options.tcq) {
    if (options.qm) {
      error = "--still-tcq is not supported with --still-qm yet";
      return false;
    }
    if (options.rdoq) {
      error = "--still-tcq is not supported with --still-rdoq yet";
      return false;
    }
    if (options.delta_q) {
      error = "--still-tcq is not supported with --still-dc-ac-delta-q yet";
      return false;
    }
    if (options.ist) {
      // The IST analysis projects the Quant domain; under TCQ the writer
      // carries the §5.20.7.27 base levels, so the projected block cannot be
      // mapped back to syntax levels without a second trellis.  Reject instead
      // of producing a stream whose reconstruction disagrees with the decoder.
      error = "--still-tcq is not supported with --still-ist yet";
      return false;
    }
  }
  if (options.delta_q) {
    if (options.qm) {
      error = "--still-dc-ac-delta-q is not supported with --still-qm yet";
      return false;
    }
    if (options.rdoq) {
      error = "--still-dc-ac-delta-q is not supported with --still-rdoq yet";
      return false;
    }
    if ((picture.width % 8U) != 0U || (picture.height % 8U) != 0U) {
      error = "--still-dc-ac-delta-q requires width and height multiples of 8";
      return false;
    }
  }
  // IBP's MrlIndex scope is 0 (src/codec/intra/ibp) and MRL changes the edge
  // filter rules, so the two cannot be combined in one sequence.
  if (options.enable_mrl && options.enable_ibp) {
    error = "--still-mrl and --still-ibp cannot be combined";
    return false;
  }
  if (options.enable_mrl && options.rdo_intra) {
    error = "--still-mrl and --still-rdo-intra cannot be combined";
    return false;
  }
  if (options.cfl_ds_filter < 0 || options.cfl_ds_filter > 3) {
    error = "cfl_ds_filter_index must be in 0..3";
    return false;
  }
  if (options.mhccp && !options.enable_cfl) {
    error = "--still-mhccp requires CfL (do not combine with --no-cfl)";
    return false;
  }
  // Palette intra is opt-in (`--still-palette`).  §5.20.8 is luma-only and is
  // only signalled for DC luma blocks of at least 8x8; the encoder decision in
  // encode_block() and the palette_stream adapter enforce both.
  // IntraBC is opt-in (`--still-ibc`).  The §5.20.5.1 ring/delay model is
  // defined for 64x64 superblocks, so the 128/256-SB modes are rejected rather
  // than signalled with a mismatched delay window.
  if (options.ibc && (options.sb128 || options.sb256)) {
    error = "--still-ibc requires 64x64 superblocks (drop --still-sb128/256)";
    return false;
  }
  if (options.ibc_search_step < 1 || options.ibc_search_step > 64) {
    error = "ibc_search_step must be in 1..64";
    return false;
  }
  // §5.4.7/§5.18.3.3/.4: let the screen-content module validate the flag
  // combination the frame header will carry.  Single-picture M1 infers
  // seq_force_screen_content_tools = seq_force_integer_mv = SELECT_*, so the
  // frame codes allow_screen_content_tools/force_integer_mv and, for IBC, the
  // global-local region bits.
  {
    scc_flags::SequenceSccConfig seq_cfg;
    const scc_flags::SequenceSccState seq_state =
        scc_flags::derive_sequence_scc_state(seq_cfg);
    scc_flags::FrameSccContext ctx;
    ctx.frame_is_intra = true;
    scc_flags::FrameSccConfig scc;
    scc.allow_screen_content_tools = options.palette || options.ibc;
    scc.force_integer_mv = scc.allow_screen_content_tools;
    scc.allow_intrabc = options.ibc;
    scc.allow_global_intrabc = options.ibc;
    scc.allow_local_intrabc = false;
    scc.max_bvp_drl_bits_minus_1 = 0;
    std::string scc_error;
    if (!scc_flags::validate_frame_scc_config(
            seq_state, ctx, scc, /*seq_max_bvp_drl_bits_minus_1=*/0,
            /*allow_frame_max_bvp_drl_bits=*/false, scc_error)) {
      error = "screen-content flags: " + scc_error;
      return false;
    }
    const scc_flags::FrameSccState state = scc_flags::derive_frame_scc_state(
        seq_state, ctx, scc, 0, false);
    // make_intrabc_bv() only emits integer sample offsets; the syntax would
    // otherwise need intrabc_precision + quarter-pel MVDs.
    if (state.IsIntraBC && !state.force_integer_mv) {
      error = "intra block copy requires force_integer_mv = 1";
      return false;
    }
  }
  if (options.per_segment_q &&
      (options.enable_segmentation || options.enable_delta_q || options.qm ||
       options.ist)) {
    error = "per-segment Q cannot be combined with the neutral segmentation, "
            "delta-Q, QM or IST paths in this integration";
    return false;
  }
  const std::size_t luma_size = static_cast<std::size_t>(picture.width) *
                                static_cast<std::size_t>(picture.height);
  const std::size_t chroma_size =
      static_cast<std::size_t>((picture.width + 1U) / 2U) *
      static_cast<std::size_t>((picture.height + 1U) / 2U);
  if (picture.y.size() < luma_size || picture.u.size() < chroma_size ||
      picture.v.size() < chroma_size) {
    error = "picture planes are too small";
    return false;
  }

  // Optional frame-level DC/AC delta-Q decision.  The component's search fixes
  // the sequence base deltas at 0 and returns the legal frame deltas plus the
  // derived per-plane quantizers used by forward quantisation and dequant.
  dcdeltaq::SequenceDeltaQ seq_delta_q;
  dcdeltaq::FrameDeltaQ frame_delta_q;
  dcdeltaq::QuantFrameState delta_quant;
  const dcdeltaq::QuantFrameState* delta_quant_ptr = nullptr;
  if (options.delta_q) {
    dcdeltaq::FramePlanesView view;
    view.width = static_cast<int>(picture.width);
    view.height = static_cast<int>(picture.height);
    view.y = picture.y;
    view.u = picture.u;
    view.v = picture.v;
    dcdeltaq::DeltaQSearchConfig search_config;
    search_config.base_q_idx = options.base_q_idx;
    search_config.equal_ac_dc_q = false;
    search_config.separate_uv_delta_q = false;
    search_config.max_delta_magnitude = 24;
    search_config.max_passes = 3;
    search_config.max_luma_blocks = 2048;
    dcdeltaq::DeltaQSearchResult search_result;
    std::string search_error;
    if (!dcdeltaq::search_frame_delta_q(view, search_config, &search_result,
                                        &search_error)) {
      error = "delta-q search failed: " + search_error;
      return false;
    }
    seq_delta_q = search_result.seq;
    frame_delta_q = search_result.frame;
    delta_quant = search_result.quant;
    delta_quant_ptr = &delta_quant;
    result.delta_q_enabled = true;
    result.delta_q_index = delta_quant.qindex;
    result.delta_q_dc[0] = delta_quant.delta.y_dc;
    result.delta_q_dc[1] = delta_quant.delta.u_dc;
    result.delta_q_dc[2] = delta_quant.delta.v_dc;
    result.delta_q_ac[0] = 0;  // luma AC never carries a delta
    result.delta_q_ac[1] = delta_quant.delta.u_ac;
    result.delta_q_ac[2] = delta_quant.delta.v_ac;
    if (std::getenv("AV2ENC_STILL_TRACE") != nullptr) {
      std::cerr << "delta-q: frame ydc=" << frame_delta_q.delta_q_y_dc
                << " udc=" << frame_delta_q.delta_q_u_dc
                << " vdc=" << frame_delta_q.delta_q_v_dc
                << " uac=" << frame_delta_q.delta_q_u_ac
                << " vac=" << frame_delta_q.delta_q_v_ac
                << " yq=" << delta_quant.plane[0].dc << "/"
                << delta_quant.plane[0].ac
                << " uq=" << delta_quant.plane[1].dc << "/"
                << delta_quant.plane[1].ac
                << " vq=" << delta_quant.plane[2].dc << "/"
                << delta_quant.plane[2].ac << "\n";
    }
  }

  // §5.4.9 / §5.18.7.1 real per-SB segmentation: assign each superblock to
  // segment 0 (base qindex) or segment 1 (base + per_segment_q_delta) using a
  // deterministic luma-variance split.  The encoder uses the derived qindex for
  // forward quant, dequant and the deblock grid; the syntax writer signals the
  // same segment id.
  rc::SegmentationParams segment_params;
  StillOptions encode_options = options;
  if (options.per_segment_q) {
    const int sb_px = (options.sb128 || options.sb256) ? 128 : 64;
    const int width = static_cast<int>(picture.width);
    const int height = static_cast<int>(picture.height);
    const int sb_cols = (width + sb_px - 1) / sb_px;
    const int sb_rows = (height + sb_px - 1) / sb_px;
    const std::size_t sb_count =
        static_cast<std::size_t>(sb_cols) * static_cast<std::size_t>(sb_rows);
    encode_options.sb_segment.assign(sb_count, 0);
    std::vector<std::uint64_t> variance(sb_count, 0);
    std::uint64_t total_variance = 0;
    for (int sby = 0; sby < sb_rows; ++sby) {
      for (int sbx = 0; sbx < sb_cols; ++sbx) {
        const int x0 = sbx * sb_px;
        const int y0 = sby * sb_px;
        const int w = std::min(sb_px, width - x0);
        const int h = std::min(sb_px, height - y0);
        std::uint64_t sum = 0;
        std::uint64_t sumsq = 0;
        for (int y = 0; y < h; ++y) {
          const std::uint8_t* row =
              picture.y.data() +
              static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x0);
          for (int x = 0; x < w; ++x) {
            const std::uint64_t v = row[x];
            sum += v;
            sumsq += v * v;
          }
        }
        const std::uint64_t n = static_cast<std::uint64_t>(w) * h;
        const std::uint64_t var =
            n == 0 ? 0 : (sumsq * n - sum * sum) / (n * n);
        variance[static_cast<std::size_t>(sby * sb_cols + sbx)] = var;
        total_variance += var;
      }
    }
    const std::uint64_t mean_variance =
        sb_count == 0 ? 0 : total_variance / sb_count;
    for (std::size_t i = 0; i < sb_count; ++i) {
      encode_options.sb_segment[i] =
          variance[i] <= mean_variance ? 1 : 0;
    }
    segment_params.enabled = true;
    segment_params.update_map = true;
    std::string seg_error;
    if (!rc::set_segment_alt_q(segment_params, /*segment_id=*/1,
                               options.per_segment_q_delta, &seg_error)) {
      error = "per-segment Q: " + seg_error;
      return false;
    }
  }

  Encoder encoder(picture, encode_options, delta_quant_ptr);
  result.qm_enabled = options.qm;
  result.qm_state = encoder.qm_state();
  SyntaxStream stream;
  if (!encoder.encode(&stream, error)) {
    return false;
  }

  // Deblock the reconstruction before CDEF or output.  Intra prediction already
  // used the unfiltered reconstruction, matching normative filter order.
  if (options.enable_deblock) {
    deblock::DeblockFrame frame;
    for (int plane = 0; plane < 3; ++plane) {
      frame.planes[plane].data = encoder.recon_mut(plane).samples.data();
      frame.planes[plane].stride = encoder.recon_mut(plane).width;
      frame.mi[plane] = encoder.deblock_mi(plane);
    }
    frame.mi_cols = encoder.deblock_cols();
    frame.mi_rows = encoder.deblock_rows();
    frame.subsampling_x = 1;
    frame.subsampling_y = 1;
    deblock::DeblockOptions dopts;
    for (int i = 0; i < 4; ++i) dopts.apply_deblocking_filter[i] = true;
    dopts.frame_is_intra = true;
    dopts.enable_sdp = options.sdp;
    // §7.17.6: chroma filter strength uses the effective UV AC delta.  Without
    // this the decoder and encoder deblock with different thresholds whenever
    // a UV AC delta is signalled.
    dopts.delta_q_u_ac = frame_delta_q.delta_q_u_ac;
    dopts.delta_q_v_ac = frame_delta_q.delta_q_v_ac;
    dopts.base_uv_ac_delta_q = seq_delta_q.base_uv_ac_delta_q;
    const deblock::DeblockStatus status =
        deblock::deblock_frame(frame, dopts);
    if (status != deblock::DeblockStatus::kOk) {
      error = std::string("deblock failed: ") +
              deblock::deblock_status_string(status);
      return false;
    }
  }

  // CDEF (§7.18) runs after deblock and before output.  The search needs the
  // original picture (only for the SSE metric) and the deblocked reconstruction.
  cuda::still::CdefStagePlan cdef_plan;
  cuda::still::CdefMiGrids cdef_grids;
  Plane cdef_planes[3];
  bool cdef_ready = false;
  if (options.enable_cdef) {
    if (options.cdef_num_strengths < 1 || options.cdef_num_strengths > 8) {
      error = "cdef_num_strengths outside 1..8";
      return false;
    }
    cuda::still::CdefMiInputs mi_inputs;
    mi_inputs.mi_cols = encoder.mi_cols();
    mi_inputs.mi_rows = encoder.mi_rows();
    // §7.18.1 Skips[]: IBC is the only still-intra path that codes
    // skip_flag = 1, and §5.20.10.1 read_cdef() leaves such blocks unfiltered
    // unless cdef_on_skip_txfm_frame_enable is set.  Materialise the skip grid
    // from the decided IBC leaves instead of assuming every block is intra.
    std::vector<std::uint8_t> cdef_skip_grid(
        static_cast<std::size_t>(mi_inputs.mi_cols) *
            static_cast<std::size_t>(mi_inputs.mi_rows),
        0);
    bool any_skip = false;
    for (const syntax::BlockSyntax& leaf : stream.blocks) {
      if ((leaf.flags & syntax::kFlagIsIbc) == 0 || leaf.ibc_skip == 0) {
        continue;
      }
      const BlockSize leaf_size = static_cast<BlockSize>(leaf.bsize);
      const int leaf_w4 = partition::num_4x4_blocks_wide(leaf_size);
      const int leaf_h4 = partition::num_4x4_blocks_high(leaf_size);
      for (int y = 0; y < leaf_h4; ++y) {
        const int rr = leaf.mi_row + y;
        if (rr >= mi_inputs.mi_rows) break;
        for (int x = 0; x < leaf_w4; ++x) {
          const int cc = leaf.mi_col + x;
          if (cc >= mi_inputs.mi_cols) break;
          cdef_skip_grid[static_cast<std::size_t>(rr) *
                             static_cast<std::size_t>(mi_inputs.mi_cols) +
                         static_cast<std::size_t>(cc)] = 1;
          any_skip = true;
        }
      }
    }
    if (any_skip) mi_inputs.skip = cdef_skip_grid.data();
    if (!cuda::still::still_cdef_build_mi_grids(mi_inputs, &cdef_grids, &error)) {
      return false;
    }

    // The search/filter want MI-aligned reference planes; the visible picture
    // may be narrower.  Only the visible area is scored, so zero padding is
    // sufficient.
    const int mi_luma_w = encoder.mi_cols() * 4;
    const int mi_luma_h = encoder.mi_rows() * 4;
    const int mi_chroma_w = mi_luma_w / 2;
    const int mi_chroma_h = mi_luma_h / 2;
    const int ref_w[3] = {mi_luma_w, mi_chroma_w, mi_chroma_w};
    const int ref_h[3] = {mi_luma_h, mi_chroma_h, mi_chroma_h};
    const int vis_w[3] = {static_cast<int>(picture.width),
                          static_cast<int>((picture.width + 1U) / 2U),
                          static_cast<int>((picture.width + 1U) / 2U)};
    const int vis_h[3] = {static_cast<int>(picture.height),
                          static_cast<int>((picture.height + 1U) / 2U),
                          static_cast<int>((picture.height + 1U) / 2U)};
    std::vector<std::uint8_t> ref_planes[3];
    const std::uint8_t* ref_ptr[3] = {nullptr, nullptr, nullptr};
    for (int plane = 0; plane < 3; ++plane) {
      const std::span<const std::uint8_t> src =
          plane == 0 ? picture.y : (plane == 1 ? picture.u : picture.v);
      ref_planes[plane].assign(
          static_cast<std::size_t>(ref_w[plane]) *
              static_cast<std::size_t>(ref_h[plane]),
          0);
      for (int y = 0; y < vis_h[plane]; ++y) {
        std::copy_n(src.begin() +
                        static_cast<std::ptrdiff_t>(y) * vis_w[plane],
                    vis_w[plane],
                    ref_planes[plane].begin() +
                        static_cast<std::ptrdiff_t>(y) * ref_w[plane]);
      }
      ref_ptr[plane] = ref_planes[plane].data();
    }

    cuda::still::CdefStageFrame cdef_frame{};
    for (int plane = 0; plane < 3; ++plane) {
      const Plane& recon = encoder.recon(plane);
      cdef_planes[plane].resize(recon.width, recon.height, 0);
      cdef_frame.plane[plane].src = recon.samples.data();
      cdef_frame.plane[plane].dst = cdef_planes[plane].samples.data();
      cdef_frame.plane[plane].stride = recon.width;
      cdef_frame.plane[plane].height = recon.height;
      cdef_frame.ref[plane] = ref_ptr[plane];
      cdef_frame.ref_stride[plane] = ref_w[plane];
    }
    cdef_frame.width = vis_w[0];
    cdef_frame.height = vis_h[0];
    cdef_frame.mi_cols = encoder.mi_cols();
    cdef_frame.mi_rows = encoder.mi_rows();
    cdef_frame.skip = cdef_grids.skip.data();
    cdef_frame.lossless = cdef_grids.lossless.data();
    cdef_frame.lossless_chroma = cdef_grids.lossless_chroma.data();

    cuda::still::CdefStageOptions cdef_options;
    cdef_options.damping = options.cdef_damping;
    cdef_options.num_strengths = options.cdef_num_strengths;
    cdef_options.on_skip_txfm = options.cdef_on_skip_txfm;
    cdef_options.include_off = true;
    if (!cuda::still::still_cdef_plan(cdef_frame, cdef_options, &cdef_plan,
                                &error)) {
      error = "cdef plan failed: " + error;
      return false;
    }

    // Every coding block whose 64x64 origin is already coded writes nothing;
    // fill every leaf so the syntax writer validates the value everywhere.
    const int n64_cols = cdef_plan.n64_cols;
    for (syntax::BlockSyntax& leaf : stream.blocks) {
      if (leaf.partition != static_cast<std::uint8_t>(PartitionType::kNone)) {
        continue;
      }
      const int br = leaf.mi_row >> 4;
      const int bc = leaf.mi_col >> 4;
      if (br < 0 || bc < 0 || br >= cdef_plan.n64_rows ||
          bc >= n64_cols) {
        error = "cdef index grid does not cover a coding block";
        return false;
      }
      leaf.cdef_idx = cdef_plan.cdef_idx[static_cast<std::size_t>(br) *
                                             static_cast<std::size_t>(n64_cols) +
                                         static_cast<std::size_t>(bc)];
    }
    cdef_ready = true;
  }

  // CCSO reads the *pre-CDEF* luma for its band classifier and loop
  // restoration uses it as CurrFrame, so keep a copy of the deblocked planes
  // before CDEF overwrites the reconstruction.
  //
  // Luma can only have one FrameRestorationType: PC-Wiener and Wiener NS are
  // alternatives for plane 0.  Chroma LR is a different plane and composes.
  if (options.enable_pc_wiener && options.enable_wiener) {
    error = "--still-pc-wiener and --still-wiener select two luma "
            "FrameRestorationTypes";
    return false;
  }
  // The combined luma PC-Wiener + chroma Wiener header row composes: plane 0
  // is RESTORE_PC_WIENER and planes 1/2 are RESTORE_WIENER_NONSEP.  Only two
  // luma FrameRestorationTypes conflict (checked above).
  const bool use_loop_restoration = options.enable_wiener ||
                                    options.enable_chroma_lr ||
                                    options.enable_pc_wiener;
  // §7.20: LrFrame := CdefFrame, then Wiener/PC-Wiener read CdefFrame and
  // write LrFrame, then GDF reads CdefFrame and rewrites LrFrame luma.  Keep
  // the pre-CDEF CurrFrame and the post-CCSO CdefFrame alive as separate
  // buffers so GDF no longer consumes the restoration output.
  Plane pre_cdef_planes[3];
  if (options.enable_ccso || use_loop_restoration || options.enable_gdf) {
    for (int plane = 0; plane < 3; ++plane) {
      pre_cdef_planes[plane] = encoder.recon(plane);
    }
  }

  // Apply CDEF now: CCSO consumes the post-CDEF frame, and the syntax stage
  // only needs the decisions (already in `stream`).
  if (cdef_ready) {
    cdef::CdefFrame filter_frame{};
    for (int plane = 0; plane < 3; ++plane) {
      const Plane& recon = encoder.recon(plane);
      filter_frame.planes[plane].src = recon.samples.data();
      filter_frame.planes[plane].dst = cdef_planes[plane].samples.data();
      filter_frame.planes[plane].stride = recon.width;
    }
    filter_frame.mi_cols = encoder.mi_cols();
    filter_frame.mi_rows = encoder.mi_rows();
    filter_frame.subsampling_x = 1;
    filter_frame.subsampling_y = 1;
    filter_frame.skip = cdef_grids.skip.data();
    filter_frame.lossless = cdef_grids.lossless.data();
    filter_frame.lossless_chroma = cdef_grids.lossless_chroma.data();
    filter_frame.cdef_idx = cdef_plan.cdef_idx.data();
    filter_frame.n64_cols = cdef_plan.n64_cols;
    filter_frame.n64_rows = cdef_plan.n64_rows;
    const cdef::CdefStatus status =
        cdef::cdef_frame(filter_frame, cdef_plan.search.options);
    if (status != cdef::CdefStatus::kOk) {
      error = std::string("cdef filter failed: ") +
              cdef::cdef_status_string(status);
      return false;
    }
    for (int plane = 0; plane < 3; ++plane) {
      encoder.recon_mut(plane).samples.swap(cdef_planes[plane].samples);
    }
  }

  StillConfig config;

  // CCSO (§7.19).  The default is band-offset-only with one offset per band
  // and plane; the fit minimises SSE against the original picture.  The
  // opt-in --still-ccso-edge form additionally searches
  // (scale, quant, ext_filter, edge_clf), fits a (band, cls0, cls1) offset
  // table and prunes units whose filter does not reduce SSE (ccso_blk = 0).
  restoration::CcsoFrameParams ccso_params;
  std::vector<std::uint8_t> ccso_blk_grid;
  std::vector<std::uint8_t> ccso_edge_grids[3];
  const std::uint8_t* ccso_grid_ptr[3] = {nullptr, nullptr, nullptr};
  int ccso_blk_cols[3] = {0, 0, 0};
  int ccso_blk_rows[3] = {0, 0, 0};
  bool ccso_ready = false;
  if (options.enable_ccso) {
    const int bands_log2 = options.ccso_bands_log2;
    if (bands_log2 < 0 || bands_log2 > 6) {
      error = "ccso_bands_log2 outside 0..6";
      return false;
    }
    const int vis_w[3] = {static_cast<int>(picture.width),
                          static_cast<int>((picture.width + 1U) / 2U),
                          static_cast<int>((picture.width + 1U) / 2U)};
    const int vis_h[3] = {static_cast<int>(picture.height),
                          static_cast<int>((picture.height + 1U) / 2U),
                          static_cast<int>((picture.height + 1U) / 2U)};
    const std::span<const std::uint8_t> src_planes[3] = {
        picture.y, picture.u, picture.v};
    const Plane& curr_luma = pre_cdef_planes[0];
    // §5.18.7.12: CcsoLumaSizeLog2 = Mi_Width_Log2[SbSize] + 2 (6 for 64,
    // 7 for a 128x128 frame superblock).
    const int ccso_luma_log2 =
        (options.sb128 || options.sb256) ? 7 : 6;
    ccso_params.ccso_luma_size_log2 = ccso_luma_log2;

    if (!options.ccso_edge) {
      const int bands = 1 << bands_log2;
      const int band_shift = 8 - bands_log2;
      for (int plane = 0; plane < 3; ++plane) {
        const int sub = plane == 0 ? 0 : 1;
        const Plane& rec = encoder.recon(plane);
        std::vector<double> sum_e(static_cast<std::size_t>(bands), 0.0);
        std::vector<double> sum_e2(static_cast<std::size_t>(bands), 0.0);
        std::vector<double> count(static_cast<std::size_t>(bands), 0.0);
        for (int y = 0; y < vis_h[plane]; ++y) {
          for (int x = 0; x < vis_w[plane]; ++x) {
            const int lx = x << sub;
            const int ly = y << sub;
            const int band = curr_luma.at(lx, ly) >> band_shift;
            const int src = src_planes[plane][static_cast<std::size_t>(y) *
                                                  static_cast<std::size_t>(
                                                      vis_w[plane]) +
                                              static_cast<std::size_t>(x)];
            const int d = src - rec.at(x, y);
            sum_e[static_cast<std::size_t>(band)] += d;
            sum_e2[static_cast<std::size_t>(band)] +=
                static_cast<double>(d) * d;
            count[static_cast<std::size_t>(band)] += 1.0;
          }
        }
        StillCcsoPlane& plane_config = config.ccso.plane[plane];
        plane_config.enabled = true;
        plane_config.bo_only = true;
        plane_config.scale_idx = 0;
        plane_config.max_band_log2 = bands_log2;
        int nonzero = 0;
        for (int band = 0; band < bands; ++band) {
          int best_idx = 0;
          double best_sse = sum_e2[static_cast<std::size_t>(band)];
          for (int i = 0; i < 8; ++i) {
            const double off =
                static_cast<double>(restoration::kCcsoOffset[i]);
            const double sse =
                sum_e2[static_cast<std::size_t>(band)] -
                2.0 * off * sum_e[static_cast<std::size_t>(band)] +
                off * off * count[static_cast<std::size_t>(band)];
            if (sse < best_sse - 1e-9) {
              best_sse = sse;
              best_idx = i;
            }
          }
          plane_config.offset_idx[band] =
              static_cast<std::uint8_t>(best_idx);
          if (best_idx != 0) ++nonzero;
        }
        if (nonzero == 0 && plane == 0) {
          plane_config.enabled = false;
        }
        restoration::CcsoParams& params = ccso_params.plane[plane];
        params.enabled = plane_config.enabled;
        params.bo_only = true;
        params.scale_idx = plane_config.scale_idx;
        params.quant_idx = 0;
        params.ext_filter = 0;
        params.edge_clf = 0;
        params.max_band_log2 = bands_log2;
        if (params.enabled &&
            !restoration::ccso_set_offsets(params, plane_config.offset_idx,
                                           static_cast<std::size_t>(bands))) {
          error = "ccso_set_offsets failed";
          return false;
        }
      }
      // One shared all-ones grid keeps the published --still-ccso bytes.
      int cols = 0;
      int rows = 0;
      restoration::ccso_blk_grid_size(encoder.mi_cols() * 4,
                                      encoder.mi_rows() * 4, ccso_luma_log2,
                                      0, 0, &cols, &rows);
      ccso_blk_grid.assign(static_cast<std::size_t>(cols) *
                               static_cast<std::size_t>(rows),
                           1);
      for (int plane = 0; plane < 3; ++plane) {
        ccso_grid_ptr[plane] = ccso_blk_grid.data();
        ccso_blk_cols[plane] = cols;
        ccso_blk_rows[plane] = rows;
        ccso_params.blks[plane] = ccso_grid_ptr[plane];
        ccso_params.blk_cols[plane] = cols;
        ccso_params.blk_rows[plane] = rows;
      }
    } else {
      // Edge-classified form.  §5.18.7.12 limits max_band_log2 to 3 here.
      const int edge_bands_log2 =
          std::min(bands_log2, restoration::kCcsoMaxBandLog2Normal);
      const int bands = 1 << edge_bands_log2;
      const int band_shift = 8 - edge_bands_log2;
      const int mi_luma_w = encoder.mi_cols() * 4;
      const int mi_luma_h = encoder.mi_rows() * 4;
      const int luma_stride = curr_luma.width;
      for (int plane = 0; plane < 3; ++plane) {
        const int sub = plane == 0 ? 0 : 1;
        const Plane& rec = encoder.recon(plane);
        const int shift = ccso_luma_log2 - sub;  // CCSO unit size
        int cols = 0;
        int rows = 0;
        restoration::ccso_blk_grid_size((mi_luma_w) >> sub,
                                        (mi_luma_h) >> sub, ccso_luma_log2,
                                        sub, sub, &cols, &rows);

        struct EdgeFit {
          double sse = std::numeric_limits<double>::infinity();
          int scale_idx = 0;
          int quant_idx = 0;
          int ext_filter = 0;
          int edge_clf = 0;
          std::vector<std::uint8_t> idx;
        } best_fit;
        for (int scale_idx = 0; scale_idx < 4; ++scale_idx) {
          for (int quant_idx = 0; quant_idx < 4; ++quant_idx) {
            const int quant =
                restoration::kCcsoQuantSz[scale_idx][quant_idx];
            if (quant == 0) continue;
            for (int ext_filter = 0; ext_filter < 7; ++ext_filter) {
              for (int edge_clf = 0; edge_clf < 2; ++edge_clf) {
                const int interval = 3 - edge_clf;
                const int cells = bands * interval * interval;
                std::vector<double> sum_e(
                    static_cast<std::size_t>(cells), 0.0);
                std::vector<double> sum_e2(
                    static_cast<std::size_t>(cells), 0.0);
                std::vector<double> count(
                    static_cast<std::size_t>(cells), 0.0);
                const int dy = restoration::kCcsoPos[ext_filter][0];
                const int dx = restoration::kCcsoPos[ext_filter][1];
                for (int y = 0; y < vis_h[plane]; ++y) {
                  for (int x = 0; x < vis_w[plane]; ++x) {
                    const int lx = x << sub;
                    const int ly = y << sub;
                    const int c = curr_luma.at(lx, ly);
                    const int px = std::clamp(lx + dx, 0, mi_luma_w - 1);
                    const int py = std::clamp(ly + dy, 0, mi_luma_h - 1);
                    const int mx = std::clamp(lx - dx, 0, mi_luma_w - 1);
                    const int my = std::clamp(ly - dy, 0, mi_luma_h - 1);
                    const int plus =
                        curr_luma.samples[static_cast<std::size_t>(py) *
                                              static_cast<std::size_t>(
                                                  luma_stride) +
                                          static_cast<std::size_t>(px)];
                    const int minus =
                        curr_luma.samples[static_cast<std::size_t>(my) *
                                              static_cast<std::size_t>(
                                                  luma_stride) +
                                          static_cast<std::size_t>(mx)];
                    const int cls0 =
                        restoration::ccso_score(plus - c, quant, edge_clf);
                    const int cls1 =
                        restoration::ccso_score(minus - c, quant, edge_clf);
                    const int band = c >> band_shift;
                    const std::size_t cell =
                        static_cast<std::size_t>(
                            (cls0 * interval + cls1) * bands + band);
                    const int src =
                        src_planes[plane][static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(
                                                  vis_w[plane]) +
                                          static_cast<std::size_t>(x)];
                    const int d = src - rec.at(x, y);
                    sum_e[cell] += d;
                    sum_e2[cell] += static_cast<double>(d) * d;
                    count[cell] += 1.0;
                  }
                }
                const int scale = scale_idx + 1;
                EdgeFit fit;
                fit.scale_idx = scale_idx;
                fit.quant_idx = quant_idx;
                fit.ext_filter = ext_filter;
                fit.edge_clf = edge_clf;
                fit.idx.assign(static_cast<std::size_t>(cells), 0);
                double total = 0.0;
                for (int cell = 0; cell < cells; ++cell) {
                  int best_idx = 0;
                  double best_sse = sum_e2[static_cast<std::size_t>(cell)];
                  for (int i = 0; i < 8; ++i) {
                    const double off =
                        static_cast<double>(restoration::kCcsoOffset[i]) *
                        scale;
                    const double sse =
                        sum_e2[static_cast<std::size_t>(cell)] -
                        2.0 * off * sum_e[static_cast<std::size_t>(cell)] +
                        off * off * count[static_cast<std::size_t>(cell)];
                    if (sse < best_sse - 1e-9) {
                      best_sse = sse;
                      best_idx = i;
                    }
                  }
                  fit.idx[static_cast<std::size_t>(cell)] =
                      static_cast<std::uint8_t>(best_idx);
                  total += best_sse;
                }
                fit.sse = total;
                if (fit.sse < best_fit.sse) best_fit = std::move(fit);
              }
            }
          }
        }

        StillCcsoPlane& plane_config = config.ccso.plane[plane];
        plane_config.enabled = true;
        plane_config.bo_only = false;
        plane_config.scale_idx = best_fit.scale_idx;
        plane_config.quant_idx = best_fit.quant_idx;
        plane_config.ext_filter = best_fit.ext_filter;
        plane_config.edge_clf = best_fit.edge_clf;
        plane_config.max_band_log2 = edge_bands_log2;
        const int cells = static_cast<int>(best_fit.idx.size());
        for (int k = 0; k < cells && k < 128; ++k) {
          plane_config.offset_idx[k] = best_fit.idx[static_cast<std::size_t>(k)];
        }
        restoration::CcsoParams& params = ccso_params.plane[plane];
        params.enabled = true;
        params.bo_only = false;
        params.scale_idx = plane_config.scale_idx;
        params.quant_idx = plane_config.quant_idx;
        params.ext_filter = plane_config.ext_filter;
        params.edge_clf = plane_config.edge_clf;
        params.max_band_log2 = edge_bands_log2;
        if (!restoration::ccso_set_offsets(
                params, plane_config.offset_idx,
                static_cast<std::size_t>(cells))) {
          error = "ccso_set_offsets failed";
          return false;
        }

        // Per-unit ccso_blk decision: keep a unit only when the fitted filter
        // lowers SSE on the visible samples of that unit.
        std::vector<double> unit_on(
            static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows),
            0.0);
        std::vector<double> unit_off(unit_on.size(), 0.0);
        const int interval = 3 - plane_config.edge_clf;
        const int quant = restoration::kCcsoQuantSz[plane_config.scale_idx]
                                                   [plane_config.quant_idx];
        const int dy = restoration::kCcsoPos[plane_config.ext_filter][0];
        const int dx = restoration::kCcsoPos[plane_config.ext_filter][1];
        for (int y = 0; y < vis_h[plane]; ++y) {
          for (int x = 0; x < vis_w[plane]; ++x) {
            const int lx = x << sub;
            const int ly = y << sub;
            const int c = curr_luma.at(lx, ly);
            const int px = std::clamp(lx + dx, 0, mi_luma_w - 1);
            const int py = std::clamp(ly + dy, 0, mi_luma_h - 1);
            const int mx = std::clamp(lx - dx, 0, mi_luma_w - 1);
            const int my = std::clamp(ly - dy, 0, mi_luma_h - 1);
            const int plus =
                curr_luma.samples[static_cast<std::size_t>(py) *
                                      static_cast<std::size_t>(luma_stride) +
                                  static_cast<std::size_t>(px)];
            const int minus =
                curr_luma.samples[static_cast<std::size_t>(my) *
                                      static_cast<std::size_t>(luma_stride) +
                                  static_cast<std::size_t>(mx)];
            const int cls0 =
                restoration::ccso_score(plus - c, quant, plane_config.edge_clf);
            const int cls1 =
                restoration::ccso_score(minus - c, quant, plane_config.edge_clf);
            if (cls0 >= interval || cls1 >= interval) continue;
            const int band = c >> band_shift;
            const int off =
                params.offsets[restoration::ccso_offset_index(band, cls0,
                                                              cls1)];
            const int src =
                src_planes[plane][static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(vis_w[plane]) +
                                  static_cast<std::size_t>(x)];
            const int base = rec.at(x, y);
            const int filtered =
                std::clamp(base + off, 0, 255);
            const int unit_col = x >> shift;
            const int unit_row = y >> shift;
            const std::size_t unit =
                static_cast<std::size_t>(unit_row) *
                    static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(unit_col);
            if (unit >= unit_on.size()) continue;
            const double d_off = src - base;
            const double d_on = src - filtered;
            unit_off[unit] += d_off * d_off;
            unit_on[unit] += d_on * d_on;
          }
        }
        ccso_edge_grids[plane].assign(unit_on.size(), 1);
        int enabled_units = 0;
        for (std::size_t u = 0; u < unit_on.size(); ++u) {
          if (unit_on[u] + 1e-9 < unit_off[u]) {
            ccso_edge_grids[plane][u] = 1;
            ++enabled_units;
          } else {
            ccso_edge_grids[plane][u] = 0;
          }
        }
        plane_config.enabled = enabled_units > 0;
        params.enabled = plane_config.enabled;
        ccso_grid_ptr[plane] = ccso_edge_grids[plane].data();
        ccso_blk_cols[plane] = cols;
        ccso_blk_rows[plane] = rows;
        ccso_params.blks[plane] = ccso_grid_ptr[plane];
        ccso_params.blk_cols[plane] = cols;
        ccso_params.blk_rows[plane] = rows;
      }
    }
    restoration::RestorationFrame filter_frame{};
    for (int plane = 0; plane < 3; ++plane) {
      const Plane& curr = pre_cdef_planes[plane];
      Plane& rec = encoder.recon_mut(plane);
      filter_frame.curr[plane].data =
          const_cast<std::uint8_t*>(curr.samples.data());
      filter_frame.curr[plane].stride = curr.width;
      filter_frame.cdef[plane].data = rec.samples.data();
      filter_frame.cdef[plane].stride = rec.width;
      filter_frame.lr[plane].data = rec.samples.data();
      filter_frame.lr[plane].stride = rec.width;
    }
    filter_frame.mi_cols = encoder.mi_cols();
    filter_frame.mi_rows = encoder.mi_rows();
    filter_frame.subsampling_x = 1;
    filter_frame.subsampling_y = 1;
    const restoration::RestorationStatus status =
        restoration::ccso_frame(filter_frame, ccso_params);
    if (status != restoration::RestorationStatus::kOk) {
      error = std::string("ccso failed: ") +
              restoration::restoration_status_string(status);
      return false;
    }
    bool any_plane = false;
    for (int plane = 0; plane < 3; ++plane) {
      any_plane = any_plane || config.ccso.plane[plane].enabled;
    }
    if (!any_plane) {
      config.ccso.enable = false;
      ccso_ready = false;
    } else {
      config.ccso.enable = true;
      ccso_ready = true;
    }
  }

  // §7.20.2 CdefFrame: the post-CDEF/CCSO picture before loop restoration.
  // Wiener, PC-Wiener and GDF all sample this buffer; only LrFrame receives
  // their output.  Keep it separate from the reconstruction so the stages can
  // compose in normative order.
  Plane cdef_frame_planes[3];
  if (use_loop_restoration || options.enable_gdf) {
    for (int plane = 0; plane < 3; ++plane) {
      cdef_frame_planes[plane] = encoder.recon(plane);
    }
  }

  // §7.20 loop restoration (non-separable Wiener NS), after CDEF and CCSO.
  // The search minimises SSE against the source; per-unit filters at
  // LoopRestorationSize 64 (luma) / 32 (chroma) keep the tile syntax simple.
  restoration::classes::WienerDecision lr_decision;
  if (use_loop_restoration) {
    const int mi_luma_w = encoder.mi_cols() * 4;
    const int mi_luma_h = encoder.mi_rows() * 4;
    const int mi_chroma_w = mi_luma_w / 2;
    const int mi_chroma_h = mi_luma_h / 2;
    const int target_w[3] = {mi_luma_w, mi_chroma_w, mi_chroma_w};
    const int target_h[3] = {mi_luma_h, mi_chroma_h, mi_chroma_h};
    const int vis_w[3] = {static_cast<int>(picture.width),
                          static_cast<int>((picture.width + 1U) / 2U),
                          static_cast<int>((picture.width + 1U) / 2U)};
    const int vis_h[3] = {static_cast<int>(picture.height),
                          static_cast<int>((picture.height + 1U) / 2U),
                          static_cast<int>((picture.height + 1U) / 2U)};
    std::vector<std::uint8_t> target_storage[3];
    const std::uint8_t* target[3] = {nullptr, nullptr, nullptr};
    for (int plane = 0; plane < 3; ++plane) {
      const std::span<const std::uint8_t> src =
          plane == 0 ? picture.y : (plane == 1 ? picture.u : picture.v);
      target_storage[plane].assign(
          static_cast<std::size_t>(target_w[plane]) *
              static_cast<std::size_t>(target_h[plane]),
          0);
      for (int y = 0; y < vis_h[plane]; ++y) {
        std::copy_n(src.begin() +
                        static_cast<std::ptrdiff_t>(y) * vis_w[plane],
                    vis_w[plane],
                    target_storage[plane].begin() +
                        static_cast<std::ptrdiff_t>(y) * target_w[plane]);
      }
      target[plane] = target_storage[plane].data();
    }

    restoration::classes::WienerDecisionInput lr_input;
    for (int plane = 0; plane < 3; ++plane) {
      lr_input.core.plane[plane].curr =
          pre_cdef_planes[plane].samples.data();
      lr_input.core.plane[plane].curr_stride = pre_cdef_planes[plane].width;
      lr_input.core.plane[plane].cdef =
          cdef_frame_planes[plane].samples.data();
      lr_input.core.plane[plane].cdef_stride = cdef_frame_planes[plane].width;
      lr_input.target[plane] = target[plane];
      lr_input.target_stride[plane] = target_w[plane];
    }
    lr_input.core.mi_cols = encoder.mi_cols();
    lr_input.core.mi_rows = encoder.mi_rows();
    lr_input.core.sub_x = 1;
    lr_input.core.sub_y = 1;
    lr_input.core.bit_depth = 8;
    // §7.20.1: sbShift = Mi_Width_Log2[SbSize] and unitRow uses that SbSize.
    // An intra frame uses SbSize 128 under --still-sb128/sb256 (§5.18.1).
    const int lr_sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    const int lr_sb_shift = lr_sb_size == 128 ? 5 : 4;
    lr_input.core.sb_shift = lr_sb_shift;
    lr_input.base_q_idx = options.base_q_idx;

    restoration::classes::WienerDecisionOptions lr_options;
    lr_options.allow_luma = options.enable_wiener;
    lr_options.allow_chroma = options.enable_chroma_lr;
    lr_options.allow_per_unit = true;
    lr_options.allow_frame_filters = false;
    lr_options.max_num_filter_classes = 1;
    lr_options.lambda_bits = 0.0;  // distortion-only integration gate
    // §5.18.7.11: a 128x128 SbSize caps the luma shift at 2 (unit 128) and
    // the chroma shift at 2 (unit 64 at 4:2:0); 64x64 keeps unit 64/32.
    // One unit per frame superblock keeps read_lr's per-SB visit shape.
    lr_options.forced_luma_shift = lr_sb_size == 128 ? 2 : 3;
    lr_options.forced_chroma_shift = lr_sb_size == 128 ? 2 : 3;
    const restoration::classes::WienerDecisionStatus lr_status =
        restoration::classes::wiener_decide(lr_input, lr_options, &lr_decision);
    if (lr_status != restoration::classes::WienerDecisionStatus::kOk) {
      error = std::string("wiener decision failed: ") +
              restoration::classes::wiener_decision_status_string(lr_status);
      return false;
    }

    bool lr_any = false;
    for (int plane = 0; plane < 3; ++plane) {
      lr_any = lr_any ||
               lr_decision.params.frame_type[plane] ==
                   restoration::kRestoreWienerNs;
    }
    if (lr_any) {
      restoration::LoopRestorationParams lr_params;
      restoration::classes::wiener_decision_to_loop_params(lr_decision,
                                                           &lr_params);
      Plane lr_planes[3];
      restoration::RestorationFrame lr_frame{};
      for (int plane = 0; plane < 3; ++plane) {
        const Plane& rec = encoder.recon(plane);
        lr_planes[plane].resize(rec.width, rec.height, 0);
        lr_frame.curr[plane].data =
            const_cast<std::uint8_t*>(pre_cdef_planes[plane].samples.data());
        lr_frame.curr[plane].stride = pre_cdef_planes[plane].width;
        lr_frame.cdef[plane].data = cdef_frame_planes[plane].samples.data();
        lr_frame.cdef[plane].stride = cdef_frame_planes[plane].width;
        lr_frame.lr[plane].data = lr_planes[plane].samples.data();
        lr_frame.lr[plane].stride = lr_planes[plane].width;
      }
      lr_frame.mi_cols = encoder.mi_cols();
      lr_frame.mi_rows = encoder.mi_rows();
      lr_frame.subsampling_x = 1;
      lr_frame.subsampling_y = 1;
      lr_frame.bit_depth = 8;
      lr_frame.sb_width = lr_sb_size;
      lr_frame.sb_height = lr_sb_size;
      lr_frame.sb_shift = lr_sb_shift;
      const restoration::RestorationStatus lr_filter_status =
          restoration::loop_restoration_frame(lr_frame, lr_params);
      if (lr_filter_status != restoration::RestorationStatus::kOk) {
        error = std::string("loop restoration failed: ") +
                restoration::restoration_status_string(lr_filter_status);
        return false;
      }
      for (int plane = 0; plane < 3; ++plane) {
        encoder.recon_mut(plane).samples.swap(lr_planes[plane].samples);
      }

      config.lr.enable = true;
      config.lr.uv_present = true;
      for (int plane = 0; plane < 3; ++plane) {
        StillLrPlane& dst = config.lr.plane[plane];
        dst.enabled = lr_decision.params.frame_type[plane] ==
                      restoration::kRestoreWienerNs;
        dst.shift = plane == 0 ? lr_decision.params.luma_shift
                               : lr_decision.params.chroma_shift;
        dst.unit_rows = lr_decision.unit_rows[plane];
        dst.unit_cols = lr_decision.unit_cols[plane];
        dst.unit_type = lr_decision.unit_type[plane];
        dst.unit_coeffs = lr_decision.unit_coeffs[plane];
        const std::size_t units = dst.unit_type.size();
        dst.unit_subset.assign(units, 0);
        dst.unit_uv_sym.assign(units, 0);
        dst.unit_merged.assign(units, 0);
        dst.unit_bank_k.assign(units, -1);
        if (lr_decision.plane[plane].units.size() != units) {
          error = "loop restoration unit plan size mismatch";
          return false;
        }
        for (std::size_t i = 0; i < units; ++i) {
          const restoration::classes::WienerUnitPlan& plan =
              lr_decision.plane[plane].units[i];
          dst.unit_subset[i] = static_cast<std::uint8_t>(plan.subset);
          dst.unit_uv_sym[i] = plan.uv_sym ? 1 : 0;
          dst.unit_merged[i] = plan.merged ? 1 : 0;
          dst.unit_bank_k[i] = static_cast<std::int8_t>(plan.use_bank_k);
        }
      }
    }
  }

  // §7.20.4 PC-Wiener (luma only), also after CDEF/CCSO.  The decision uses
  // the recovered classifier/filter bank and emits per-unit on/off symbols.
  if (options.enable_pc_wiener) {
    namespace pcw = av2enc::filters::pc_wiener_stream;
    const int mi_luma_w = encoder.mi_cols() * 4;
    const int mi_luma_h = encoder.mi_rows() * 4;
    const int vis_w = static_cast<int>(picture.width);
    const int vis_h = static_cast<int>(picture.height);
    std::vector<std::uint8_t> target(
        static_cast<std::size_t>(mi_luma_w) *
            static_cast<std::size_t>(mi_luma_h),
        0);
    for (int y = 0; y < vis_h; ++y) {
      const std::uint8_t* row =
          picture.y.data() + static_cast<std::ptrdiff_t>(y) * vis_w;
      std::copy_n(row, vis_w,
                  target.begin() + static_cast<std::ptrdiff_t>(y) * mi_luma_w);
      // Edge-replicate the MI padding so the per-unit SSE decision is not
      // biased by zeros outside the visible picture.
      std::fill(target.begin() + static_cast<std::ptrdiff_t>(y) * mi_luma_w +
                    vis_w,
                target.begin() + static_cast<std::ptrdiff_t>(y + 1) *
                                    mi_luma_w,
                vis_w > 0 ? row[vis_w - 1] : 0);
    }
    for (int y = vis_h; y < mi_luma_h; ++y) {
      const std::uint8_t* prev =
          target.data() + static_cast<std::ptrdiff_t>(y - 1) * mi_luma_w;
      std::copy_n(prev, mi_luma_w,
                  target.begin() + static_cast<std::ptrdiff_t>(y) * mi_luma_w);
    }

    pcw::DecisionPlanes planes;
    planes.width = mi_luma_w;
    planes.height = mi_luma_h;
    planes.bit_depth = 8;
    planes.base_q_idx = options.base_q_idx;
    planes.sub_x = 1;
    planes.sub_y = 1;
    // §5.18.7.11: SbSize 128 limits the luma unit shift to 2 (see the Wiener
    // decision above); 64x64 keeps the M1 64/32 units.
    const int pcw_sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    planes.luma_shift = pcw_sb_size == 128 ? 2 : 3;
    planes.chroma_shift = pcw_sb_size == 128 ? 2 : 3;
    for (int plane = 0; plane < 3; ++plane) {
      planes.curr[plane] = pre_cdef_planes[plane].samples.data();
      planes.curr_stride[plane] = pre_cdef_planes[plane].width;
      planes.cdef[plane] = cdef_frame_planes[plane].samples.data();
      planes.cdef_stride[plane] = cdef_frame_planes[plane].width;
    }
    planes.target[0] = target.data();
    planes.target_stride[0] = mi_luma_w;

    // §7.20.1 LrTxSkip[row][col] = skip_flag || (eob == 0) for luma.  M1 has
    // no block-skip syntax, so it is exactly the per-4x4 eob==0 map.
    std::vector<std::uint8_t> lr_tx_skip(
        static_cast<std::size_t>(encoder.mi_cols()) *
            static_cast<std::size_t>(encoder.mi_rows()),
        0);
    for (const syntax::BlockSyntax& leaf : stream.blocks) {
      if (leaf.partition != static_cast<std::uint8_t>(PartitionType::kNone)) {
        continue;
      }
      const int w4 =
          partition::block_width(static_cast<BlockSize>(leaf.bsize)) / 4;
      const int h4 =
          partition::block_height(static_cast<BlockSize>(leaf.bsize)) / 4;
      const std::uint8_t value = leaf.eob[0] == 0 ? 1 : 0;
      for (int r = 0; r < h4; ++r) {
        const int row = leaf.mi_row + r;
        if (row >= encoder.mi_rows()) break;
        for (int c = 0; c < w4; ++c) {
          const int col = leaf.mi_col + c;
          if (col >= encoder.mi_cols()) break;
          lr_tx_skip[static_cast<std::size_t>(row) *
                         static_cast<std::size_t>(encoder.mi_cols()) +
                     static_cast<std::size_t>(col)] = value;
        }
      }
    }
    planes.lr_tx_skip = lr_tx_skip.data();
    planes.lr_tx_skip_stride = encoder.mi_cols();
    // Publish the same map to the GPU orchestrator: its §7.20.4 classifier
    // reads LrTxSkip for every 4x4 block, and the still path's skip map is the
    // per-block eob==0 mask (not all zero).
    result.lr_tx_skip = lr_tx_skip;
    result.lr_tx_skip_cols = encoder.mi_cols();
    result.lr_tx_skip_rows = encoder.mi_rows();

    pcw::DecisionResult pc_result;
    if (!pcw::DecidePerUnitPcWiener(planes, 0.0, &pc_result, &error)) {
      error = "PC-Wiener decision failed: " + error;
      return false;
    }
    if (pc_result.active_units > 0) {
      const int unit_size =
          pcw::kRestorationTileSizeMax >> pc_result.params.luma_shift;
      Plane lr_luma;
      const Plane& rec = encoder.recon(0);
      lr_luma.resize(rec.width, rec.height, 0);
      const int filtered = pcw::ApplyPerUnitPcWiener(
          planes, pc_result.unit_type[0], unit_size,
          lr_luma.samples.data(), lr_luma.width);
      if (filtered <= 0) {
        error = "PC-Wiener application failed";
        return false;
      }
      encoder.recon_mut(0).samples.swap(lr_luma.samples);

      config.lr.enable = true;
      config.lr.uv_present = true;
      config.lr.pc_wiener = true;
      StillLrPlane& dst = config.lr.plane[0];
      dst.enabled = true;
      dst.shift = pc_result.params.luma_shift;
      dst.unit_rows = pcw::CountUnits(unit_size, mi_luma_h);
      dst.unit_cols = pcw::CountUnits(unit_size, mi_luma_w);
      dst.unit_type = pc_result.unit_type[0];
      // Plane 0 is PC-Wiener.  Chroma LR, when enabled above, keeps its
      // RESTORE_WIENER_NONSEP plan; the pair has no syntax conflict.
    }
  }

  // §7.20.5 GDF: last stage of loop restoration.  §7.20.1 runs the Wiener or
  // PC-Wiener filter for the block first and then conditionally applies GDF to
  // LrFrame; both sample CdefFrame (inside the stripe) and CurrFrame (outside),
  // never each other's output.  So GDF reads the post-CCSO CdefFrame snapshot
  // and rewrites the luma LrFrame produced by the restoration stage above.
  if (options.enable_gdf) {
    namespace gdf = av2enc::filters::gdf;
    const int gdf_w = encoder.mi_cols() * 4;
    const int gdf_h = encoder.mi_rows() * 4;
    const int vis_w = static_cast<int>(picture.width);
    const int vis_h = static_cast<int>(picture.height);
    std::vector<std::uint16_t> curr16(
        static_cast<std::size_t>(gdf_w) * static_cast<std::size_t>(gdf_h));
    std::vector<std::uint16_t> cdef16(curr16.size());
    std::vector<std::uint16_t> lr16(curr16.size());
    std::vector<std::uint16_t> original16(curr16.size());
    for (int y = 0; y < gdf_h; ++y) {
      const std::uint8_t* curr_row =
          pre_cdef_planes[0].samples.data() +
          static_cast<std::ptrdiff_t>(y) * pre_cdef_planes[0].width;
      const std::uint8_t* cdef_row =
          cdef_frame_planes[0].samples.data() +
          static_cast<std::ptrdiff_t>(y) * cdef_frame_planes[0].width;
      const std::uint8_t* lr_row =
          encoder.recon(0).samples.data() +
          static_cast<std::ptrdiff_t>(y) * encoder.recon(0).width;
      const std::uint8_t* src_row = nullptr;
      std::uint8_t edge = 0;
      if (y < vis_h) {
        src_row = picture.y.data() + static_cast<std::ptrdiff_t>(y) * vis_w;
        edge = vis_w > 0 ? src_row[vis_w - 1] : 0;
      } else {
        const int last = vis_h > 0 ? vis_h - 1 : 0;
        src_row = picture.y.data() + static_cast<std::ptrdiff_t>(last) * vis_w;
        edge = vis_w > 0 ? src_row[vis_w - 1] : 0;
      }
      for (int x = 0; x < gdf_w; ++x) {
        const std::size_t i =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(gdf_w) +
            static_cast<std::size_t>(x);
        curr16[i] = curr_row[x];
        cdef16[i] = cdef_row[x];
        lr16[i] = lr_row[x];
        original16[i] = x < vis_w ? src_row[x] : edge;
      }
    }

    gdf::GdfSourceFrame source;
    source.curr = curr16.data();
    source.cdef = cdef16.data();
    source.stride = gdf_w;
    gdf::GdfPlane original;
    original.data = original16.data();
    original.width = gdf_w;
    original.height = gdf_h;
    original.stride = gdf_w;
    // §5.18.7.9: SbSize is the frame superblock size (128x128 for an intra
    // frame under --still-sb128/sb256, 64x64 otherwise).  The sequence-level
    // gdf_unit_matches_sb_size is only coded when get_seq_sb_size() is 64 and
    // is inferred 0 here, so GdfBlkSize = Max(Block_Width[SbSize], 128).
    const int gdf_sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    gdf::FrameGdfContext gdf_context;
    gdf_context.enable_gdf = true;
    gdf_context.gdf_unit_matches_sb_size = gdf_sb_size == 64;
    gdf_context.coded_lossless = false;
    gdf_context.single_picture_header_flag = true;
    gdf_context.sb_size = gdf_sb_size;
    gdf_context.width = gdf_w;
    gdf_context.height = gdf_h;
    gdf_context.mi_cols = encoder.mi_cols();
    gdf_context.mi_rows = encoder.mi_rows();
    gdf_context.disable_loopfilters_across_tiles = false;
    gdf_context.tile_cols = 1;
    gdf_context.tile_rows = 1;

    gdf::GdfDecisionOptions gdf_options;
    gdf_options.filter.base_q_idx = options.base_q_idx;
    gdf_options.filter.frame_is_intra = true;
    gdf_options.filter.bit_depth = 8;
    gdf_options.disable_cdf_update = !options.adaptive_cdf;
    gdf_options.lambda_bits = 0.0;
    gdf::GdfDecision gdf_decision;
    if (!gdf::DecideGdf(source, original, gdf_context, gdf_options,
                        &gdf_decision, &error)) {
      error = "GDF decision failed: " + error;
      return false;
    }
    if (gdf_decision.params.frame_enable) {
      gdf::FrameGdfParams params = gdf_decision.params;
      params.per_block = false;  // M1 codes no use_gdf symbols
      std::vector<std::uint16_t> out16(lr16);
      if (!gdf::ApplyGdfToFrame(source, gdf_options.filter, gdf_context, params,
                                {}, out16.data(), gdf_w, &error)) {
        error = "GDF apply failed: " + error;
        return false;
      }
      Plane& rec = encoder.recon_mut(0);
      for (int y = 0; y < gdf_h; ++y) {
        for (int x = 0; x < gdf_w; ++x) {
          rec.samples[static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(rec.width) +
                      static_cast<std::size_t>(x)] =
              static_cast<std::uint8_t>(
                  out16[static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(gdf_w) +
                        static_cast<std::size_t>(x)]);
        }
      }
      config.gdf.enable = true;
      // The sequence row only carries gdf_unit_matches_sb_size for a 64x64
      // sequence SB; a 128x128 frame SbSize makes it inferred 0.  The header
      // writer keys the bit on this field, and gdf_params() re-derives the
      // same 128-sample GdfBlkSize from SbSize.
      config.gdf.unit_matches_sb_size = gdf_sb_size == 64;
      config.gdf.per_block = false;
      config.gdf.sb_size = gdf_sb_size;
      config.gdf.pic_qc_idx = params.pic_qc_idx;
      config.gdf.pic_scale_idx = params.pic_scale_idx;
      config.gdf.blk_size = params.blk_size;
      config.gdf.cols = params.cols;
      config.gdf.rows = params.rows;
      config.gdf.pix_scale = params.pix_scale;
    }
  }

  // Film grain (§5.14/§5.18.10).  The default signals a fixed two-point 4:2:0
  // model; --still-grain-adaptive estimates the model from the reconstruction
  // with the GrainStreamPlanner.  Synthesis is decoder-side, so the encoder
  // recon stays unfiltered either way.
  if (options.enable_grain) {
    namespace grain = av2enc::filters::grain;
    if (options.grain_adaptive) {
      // §5.18.10.1 infers apply_grain = 1 for a single-picture header, so the
      // estimator may not cancel.  min_activity = 0 keeps the planner's
      // validity checks but never selects "no grain" for content with a
      // non-zero scaling curve; a genuinely flat frame falls through to the
      // no-film-grain case below (no OBU, film_grain_params_present = 0).
      grain::GrainPlanOptions plan_options;
      plan_options.chroma_idc = grain::GrainChromaIdc::k420;
      plan_options.estimate.min_activity = 0.0;
      grain::GrainStreamPlanner planner(plan_options);
      grain::GrainFrameView view;
      view.y = encoder.recon(0).samples.data();
      view.u = encoder.recon(1).samples.data();
      view.v = encoder.recon(2).samples.data();
      view.width = static_cast<int>(picture.width);
      view.height = static_cast<int>(picture.height);
      view.y_stride = encoder.recon(0).width;
      view.c_stride = encoder.recon(1).width;
      grain::GrainFramePlan plan;
      if (!planner.PlanFrame(view, /*frame_index=*/0, /*tlayer_id=*/0,
                             /*mlayer_id=*/0, &plan, &error)) {
        error = "film grain planner failed: " + error;
        return false;
      }
      if (plan.apply_grain) {
        if (!plan.emit_model) {
          error = "film grain planner reused a slot without a model OBU";
          return false;
        }
        std::string grain_error;
        config.film_grain_obu =
            grain::MakeFilmGrainObu(plan.obu, &grain_error);
        if (config.film_grain_obu.empty()) {
          error = "film grain OBU failed: " + grain_error;
          return false;
        }
        config.film_grain_present = true;
        config.film_grain_id = plan.fgm_id;
        config.grain_seed = plan.grain_seed;
      }
    } else {
      grain::GrainModel model;
      model.chroma_scaling_from_luma = true;
      model.num_y_points = 2;
      model.point_y_value[0] = 0;
      model.point_y_value[1] = 128;
      model.point_y_scaling[0] = 64;
      model.point_y_scaling[1] = 88;
      model.point_y_increment_bits_minus_1 = 7;
      model.point_y_scaling_bits_minus_5 = 3;
      model.grain_scaling_minus_8 = 0;
      model.ar_coeff_lag = 2;
      model.bits_per_ar_coeff_y_minus_5 = 3;
      model.bits_per_ar_coeff_cb_minus_5 = 3;
      model.bits_per_ar_coeff_cr_minus_5 = 3;
      const int positions = 2 * model.ar_coeff_lag * (model.ar_coeff_lag + 1);
      for (int i = 0; i < positions; ++i) {
        const int tap = ((i % 3) - 1) * 8;
        model.ar_coeffs_y[i] = tap;
        model.ar_coeffs_cb[i] = tap / 2;
        model.ar_coeffs_cr[i] = tap / 2;
      }
      model.ar_coeffs_cb[positions] = 0;
      model.ar_coeffs_cr[positions] = 0;
      model.ar_coeff_shift_minus_6 = 1;
      model.overlap_flag = true;
      model.clip_to_restricted_range = false;
      model = grain::NormalizeModel(model, grain::GrainChromaIdc::k420);
      std::string grain_why;
      if (!grain::ValidateModel(model, grain::GrainChromaIdc::k420,
                                &grain_why)) {
        error = "film grain model is invalid: " + grain_why;
        return false;
      }
      grain::FilmGrainObu obu;
      obu.update_flags = 1;  // slot 0
      obu.chroma_idc = grain::GrainChromaIdc::k420;
      obu.models[0] = model;
      std::string grain_error;
      config.film_grain_obu = grain::MakeFilmGrainObu(obu, &grain_error);
      if (config.film_grain_obu.empty()) {
        error = "film grain OBU failed: " + grain_error;
        return false;
      }
      config.film_grain_present = true;
      config.film_grain_id = 0;
      config.grain_seed = 0;
    }
  }

  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_intra_edge_filter =
      options.enable_intra_edge_filter &&
      std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr;
  config.enable_cfl_intra = options.enable_cfl;
  config.enable_chroma_dctonly = true;
  config.enable_idtx_intra = options.idtx;
  config.enable_intra_ist = options.ist;
  config.enable_inter_ist = false;
  config.enable_tcq = options.tcq;
  config.enable_cctx = options.cctx;
  config.cfl_ds_filter_index = options.cfl_ds_filter;
  config.enable_dip = options.enable_dip;
  config.enable_palette = options.palette;
  config.enable_mhccp = options.mhccp;
  config.allow_screen_content_tools = options.palette || options.ibc;
  config.force_integer_mv = true;
  config.allow_intrabc = options.ibc;
  config.allow_global_intrabc = options.ibc;
  config.allow_local_intrabc = false;
  config.enable_ibp = options.enable_ibp;
  config.enable_mrls = options.enable_mrl;
  config.tx_mode_select = options.tx_partition;
  config.reduced_tx_set = 0;
  config.reduced_tx_part_set = false;
  config.use_128x128_superblock = options.sb128;
  config.use_256x256_superblock = options.sb256;
  config.enable_ext_partitions = options.ext_partitions;
  config.enable_uneven_4way_partitions = options.ext_partitions;
  config.enable_sdp = options.sdp;
  config.qm = encoder.qm_state();
  if (options.user_qm) {
    const auto level = static_cast<int>(options.qm_level);
    qm::QmObuConfig obu;
    obu.bit_map = static_cast<std::uint16_t>(1U << level);
    obu.chroma_info_present = true;
    obu.levels[level].is_default = false;
    obu.levels[level].user = config.qm.user_qm[level];
    std::string qm_error;
    config.qm_obu = qm::make_quantizer_matrix_obu(obu, &qm_error);
    if (config.qm_obu.empty()) {
      error = "quantizer_matrix_obu failed: " + qm_error;
      return false;
    }
  }
  config.dc_ac_delta_q = seq_delta_q;
  config.frame_delta_q = frame_delta_q;
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u = options.enable_deblock;
  config.apply_deblock_v = options.enable_deblock;
  config.per_sb_delta_q.segmentation_enabled =
      options.enable_segmentation || options.per_segment_q;
  config.per_sb_delta_q.delta_q_present = options.enable_delta_q;
  config.per_sb_delta_q.delta_q_res = options.enable_delta_q ? options.delta_q_res : 0;
  config.per_sb_delta_q.segmentation =
      options.per_segment_q ? &segment_params : nullptr;
  config.content_interpretation = options.content_interpretation;
  if (cdef_ready) {
    const cuda::still::CdefHeaderParams& header = cdef_plan.header;
    config.cdef.enable = header.frame_enable;
    config.cdef.damping = header.damping_minus_3 + 3;
    config.cdef.num_strengths = header.num_strengths;
    config.cdef.on_skip_txfm = header.on_skip_txfm_frame_enable;
    for (int i = 0; i < header.num_strengths; ++i) {
      config.cdef.strengths[i].y_pri = header.set[i].y_pri_strength;
      config.cdef.strengths[i].y_sec = header.set[i].y_sec_value;
      config.cdef.strengths[i].uv_pri = header.set[i].uv_pri_strength;
      config.cdef.strengths[i].uv_sec = header.set[i].uv_sec_value;
    }
  }

  // Syntax writing is a separate stage: it re-derives every CDF context from
  // the block stream, exactly like the GPU path will.
  syntax::FramePlan plan;
  plan.width = static_cast<int>(picture.width);
  plan.height = static_cast<int>(picture.height);
  plan.sb_size = config.use_128x128_superblock ||
                         config.use_256x256_superblock
                     ? 128
                     : 64;
  if (ccso_ready) {
    for (int plane = 0; plane < 3; ++plane) {
      plan.ccso_blks[plane] = ccso_grid_ptr[plane];
      plan.ccso_blk_cols[plane] = ccso_blk_cols[plane];
      plan.ccso_blk_rows[plane] = ccso_blk_rows[plane];
    }
  }
  std::size_t tile_bytes = 0;
  if (!syntax::assemble_still_frame(plan, stream.blocks, stream.levels, config,
                                 result.frame, error, &tile_bytes,
                                 encoder.palette_results(),
                                 std::span<const std::uint16_t>(options.cdf_seed),
                                 &result.cdf_final)) {
    return false;
  }
  result.tile_bytes = static_cast<std::uint32_t>(tile_bytes);
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.leaves = encoder.leaves();
  result.ibc_blocks = encoder.ibc_blocks();
  result.palette_results = encoder.palette_results();
  result.palette_index_by_leaf = encoder.palette_index_by_leaf();
  result.ibc_by_leaf = encoder.ibc_by_leaf();

  const auto crop = [](const Plane& plane, int width, int height) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
      std::copy_n(plane.samples.begin() +
                      static_cast<std::ptrdiff_t>(y) * plane.width,
                  width,
                  out.begin() + static_cast<std::ptrdiff_t>(y) * width);
    }
    return out;
  };
  result.recon_y = crop(encoder.recon(0), static_cast<int>(picture.width),
                        static_cast<int>(picture.height));
  result.recon_u =
      crop(encoder.recon(1), static_cast<int>((picture.width + 1U) / 2U),
           static_cast<int>((picture.height + 1U) / 2U));
  result.recon_v =
      crop(encoder.recon(2), static_cast<int>((picture.width + 1U) / 2U),
           static_cast<int>((picture.height + 1U) / 2U));
  // Hand the exact header/filter configuration and the CDEF index grid to the
  // GPU orchestrator (it reuses them instead of re-deriving the parameters).
  result.config = config;
  if (cdef_ready) {
    result.cdef_idx = cdef_plan.cdef_idx;
    result.cdef_n64_cols = cdef_plan.n64_cols;
    result.cdef_n64_rows = cdef_plan.n64_rows;
  }
  if (ccso_ready) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::uint8_t* grid = ccso_grid_ptr[plane];
      const int cols = ccso_blk_cols[plane];
      const int rows = ccso_blk_rows[plane];
      if (grid == nullptr || cols <= 0 || rows <= 0) {
        continue;
      }
      result.ccso_blks[plane].assign(
          grid, grid + static_cast<std::size_t>(cols) *
                         static_cast<std::size_t>(rows));
      result.ccso_blk_cols[plane] = cols;
      result.ccso_blk_rows[plane] = rows;
    }
  }
  return true;
}

}  // namespace av2enc::still
