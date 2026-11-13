// CUDA M2 P-frame orchestration.
//
// This is a decision-identical mirror of src/encoder/m2/video_frame.cpp.  The
// partition tree, MV-stack derivation, candidate ordering and tie-breaking are
// the same code paths (the CPU partition/MC decision helpers); the pixel and
// transform stages are delegated to the linked backend.
//
// The module never edits or wraps the CPU encoder: it is a separate consumer of
// the same public components, so the CPU gate keeps its own reference path.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "codec/inter/mc/mc.h"
#include "codec/inter/merge/merge.h"
#include "codec/inter/bframes/bframes.h"
#include "codec/inter/tlayers/tlayers.h"
#include "codec/entropy/avg_cdf/avg_cdf_bank.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "frame/mfh/mfh.h"
#include "frame/metadata/content_interpretation/content_interpretation.h"

#include "video_backend.h"
#include "video_gpu.h"
#include "video_sequence_internal.h"

namespace av2enc::cuda::video {
namespace {

using av2enc::video::VideoConfig;
using av2enc::video::VideoFrameHeader;
using av2enc::video::VideoInputFrame;
using av2enc::video::VideoOptions;
using av2enc::video::VideoReconFrame;
using av2enc::video::VideoSequenceResult;
using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionNode;
using partition::PartitionType;

namespace detail = av2enc::cuda::video::detail;
namespace merge = ::av2enc::video::merge;

// The CPU M2 path grows optional tools over time (inter residual, DRL
// reordering).  Detect the option members instead of hard-depending on them,
// so this tree builds against both the baseline and the extended VideoOptions.
template <typename T, typename = void>
struct HasEnableInterResidual : std::false_type {};
template <typename T>
struct HasEnableInterResidual<
    T, std::void_t<decltype(std::declval<T&>().enable_inter_residual)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasForceInterResidual : std::false_type {};
template <typename T>
struct HasForceInterResidual<
    T, std::void_t<decltype(std::declval<T&>().force_inter_residual)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasDrlReorder : std::false_type {};
template <typename T>
struct HasDrlReorder<T,
                     std::void_t<decltype(std::declval<T&>().drl_reorder)>>
    : std::true_type {};

template <typename T>
bool opt_enable_inter_residual(const T& options) {
  if constexpr (HasEnableInterResidual<T>::value) {
    return options.enable_inter_residual;
  } else {
    return false;
  }
}

template <typename T, typename = void>
struct HasInterRdLambda : std::false_type {};
template <typename T>
struct HasInterRdLambda<
    T, std::void_t<decltype(std::declval<T&>().inter_rd_lambda)>>
    : std::true_type {};

template <typename T>
double opt_inter_rd_lambda(const T& options) {
  if constexpr (HasInterRdLambda<T>::value) {
    return options.inter_rd_lambda;
  } else {
    return 4.0;
  }
}

template <typename T>
bool opt_force_inter_residual(const T& options) {
  if constexpr (HasForceInterResidual<T>::value) {
    return options.force_inter_residual;
  } else {
    return false;
  }
}

template <typename T>
int opt_drl_reorder(const T& options) {
  if constexpr (HasDrlReorder<T>::value) {
    return options.drl_reorder;
  } else {
    return 0;
  }
}

template <typename C>
void set_config_drl_reorder(C& config, int value) {
  if constexpr (HasDrlReorder<C>::value) {
    config.drl_reorder = static_cast<std::uint8_t>(value);
  }
}

// estimate_tx_bits() from the CPU M2 inter-residual decision.  Deliberately a
// heuristic, but it must be bit-for-bit the same heuristic for parity.
double estimate_tx_bits(int eob, const std::vector<std::int32_t>& levels,
                        bool is_inter_luma) {
  if (eob == 0) return 1.0;
  double bits = 8.0;
  if (is_inter_luma) bits += 4.0;
  int nonzero = 0;
  for (int c = 0; c < eob; ++c) {
    if (levels[static_cast<std::size_t>(c)] != 0) ++nonzero;
  }
  bits += static_cast<double>(nonzero) * 2.5 +
          static_cast<double>(eob) * 0.5;
  return bits;
}

// Round a 1/8-pel MV to the signalled frame precision: 1/8, 1/4 or 1/2 pel.
// Mirrors the CPU reference vp quantize_mv() exactly (half away from zero).
inter::mc::Mv quantize_mv(inter::mc::Mv mv, int precision) {
  const int quantum = precision == 1 ? 2 : (precision == 2 ? 4 : 1);
  if (quantum == 1) return mv;
  const auto round_component = [quantum](int value) {
    const int magnitude = value < 0 ? -value : value;
    const int rounded = ((magnitude + quantum / 2) / quantum) * quantum;
    return value < 0 ? -rounded : rounded;
  };
  return inter::mc::Mv{round_component(mv.y), round_component(mv.x)};
}

int eob_from_levels(const std::vector<std::int32_t>& levels, int w, int h,
                    std::vector<std::uint16_t>* scan_out) {
  const int area = w * h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::video::syntax::build_scan(w, h, scan);
  int eob = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (levels[scan[static_cast<std::size_t>(c)]] != 0) {
      eob = c + 1;
      break;
    }
  }
  if (scan_out != nullptr) *scan_out = std::move(scan);
  return eob;
}

struct Plane {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    stride = w;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                   value);
  }
  std::uint8_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x)];
  }
};

int source_sample(const VideoInputFrame& frame, int plane, int x, int y) {
  const int w = plane == 0 ? static_cast<int>(frame.width)
                           : static_cast<int>((frame.width + 1U) / 2U);
  const int h = plane == 0 ? static_cast<int>(frame.height)
                           : static_cast<int>((frame.height + 1U) / 2U);
  const int cx = std::clamp(x, 0, w - 1);
  const int cy = std::clamp(y, 0, h - 1);
  const std::span<const std::uint8_t> data =
      plane == 0 ? frame.y : (plane == 1 ? frame.u : frame.v);
  return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(w) +
              static_cast<std::size_t>(cx)];
}

std::int64_t sse_against_source(const VideoInputFrame& frame, int plane, int x0,
                                int y0, int w, int h,
                                const std::vector<std::uint8_t>& samples) {
  std::int64_t sse = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int src = source_sample(frame, plane, x0 + x, y0 + y);
      const int p = samples[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(w) +
                           static_cast<std::size_t>(x)];
      const int d = src - p;
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

void write_plane_block(Plane& plane, int x0, int y0, int w, int h,
                       const std::vector<std::uint8_t>& samples) {
  for (int y = 0; y < h; ++y) {
    const int py = y0 + y;
    if (py < 0 || py >= plane.height) continue;
    for (int x = 0; x < w; ++x) {
      const int px = x0 + x;
      if (px < 0 || px >= plane.width) continue;
      plane.at(px, py) =
          samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(x)];
    }
  }
}

std::uint64_t node_key(int mi_row, int mi_col, BlockSize size) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(mi_row)) << 32) |
         (static_cast<std::uint64_t>(static_cast<std::uint32_t>(mi_col)) << 8) |
         static_cast<std::uint64_t>(static_cast<std::uint8_t>(size));
}

// ME results are per (leaf, reference); pack the reference index in the low
// three bits so the map stays a flat uint64 key.
std::uint64_t me_key(int mi_row, int mi_col, BlockSize size, int ref_index) {
  return node_key(mi_row, mi_col, size) * 8u +
         static_cast<std::uint64_t>(ref_index);
}

struct InterCandidate {
  bool valid = false;
  bool compound = false;
  int mode = av2enc::video::syntax::kInterGlobalMv;
  int drl_index = 0;
  int ref_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  std::int64_t sse = 0;
};

class GpuFrameEncoder {
 public:
  GpuFrameEncoder(detail::Backend& backend, const VideoInputFrame& source,
                  std::span<const VideoReconFrame* const> references,
                  std::array<int, 7> frame_distance, int num_total_refs,
                  const VideoOptions& options, bool trace, bool keyframe,
                  std::string& error,
                  std::span<const std::uint16_t> cdf_seed = {},
                  std::vector<std::uint16_t>* cdf_final = nullptr)
      : backend_(backend),
        source_(source),
        options_(options),
        frame_distance_(frame_distance),
        num_total_refs_(num_total_refs),
        cdf_seed_(cdf_seed),
        cdf_final_(cdf_final),
        trace_(trace),
        keyframe_(keyframe) {
    width_ = static_cast<int>(source.width);
    height_ = static_cast<int>(source.height);

    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = 64;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);

    recon_[0].resize(mi_cols_ * 4, mi_rows_ * 4, 0);
    recon_[1].resize(mi_cols_ * 2, mi_rows_ * 2, 0);
    recon_[2].resize(mi_cols_ * 2, mi_rows_ * 2, 0);

    mv_grid_.assign(static_cast<std::size_t>(mi_rows_) *
                        static_cast<std::size_t>(mi_cols_),
                    inter::mc::MiInfo{});

    if (!backend_.begin_frame(source_, mi_cols_, mi_rows_, references, error)) {
      begin_ok_ = false;
    }
  }

  ~GpuFrameEncoder() = default;

  bool begin_ok() const { return begin_ok_; }

  bool encode(std::vector<std::uint8_t>& tile_payload,
              std::vector<av2enc::video::syntax::BlockSyntax>& blocks,
              std::vector<std::int32_t>& levels, std::string& error) {
    if (!begin_ok_) {
      error = "backend begin_frame failed";
      return false;
    }

    if (!decide_partition(error)) return false;
    if (!precompute_me(error)) return false;

    blocks_.clear();
    levels_.clear();
    for (const PartitionNode& sb : frame_.superblocks) {
      if (!collect_node(sb, error)) return false;
    }

    av2enc::video::syntax::FramePlan plan;
    plan.width = width_;
    plan.height = height_;
    plan.sb_size = 64;
    plan.inter_frame = !keyframe_;
    plan.num_total_refs = num_total_refs_;
    plan.reference_select = options_.compound && num_total_refs_ > 1;
    plan.mv_precision = options_.mv_precision;
    plan.frame_distance = frame_distance_;
    VideoConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(width_);
    config.max_frame_height = static_cast<std::uint16_t>(height_);
    config.base_q_idx = options_.base_q_idx;
    config.order_hint_bits = 4;
    config.num_ref_frames = 2;
    config.num_total_refs = 1;
    config.max_drl_bits_minus_1 = 4;
    config.disable_cdf_update = !options_.adaptive_cdf;
    config.enable_avg_cdf = options_.avg_cdf;
    config.avg_cdf_type = true;
    set_config_drl_reorder(config, opt_drl_reorder(options_));
    config.enable_cfl_intra = true;

    std::size_t tile_bytes = 0;
    if (!av2enc::video::syntax::assemble_video_frame(
            plan, blocks_, levels_, config, tile_payload, error, &tile_bytes,
            cdf_seed_, cdf_final_)) {
      return false;
    }
    blocks = std::move(blocks_);
    levels = std::move(levels_);

    if (!backend_.end_frame(error)) return false;
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  std::uint32_t inter_blocks() const { return inter_blocks_; }
  std::uint32_t intra_blocks() const { return intra_blocks_; }

 private:
  bool decide_partition(std::string& error) {
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = options_.min_block_size;
    poptions.max_depth = options_.max_partition_depth;
    poptions.allow_rect = true;
    poptions.allow_split = false;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }
    // Pass 1: record every node the CPU decision traversal can visit.  The
    // visited set is geometry-driven; the callback returns finite zero costs so
    // both policy passes are explored.  Extra nodes are harmless.
    std::vector<detail::VarianceNode> nodes;
    std::vector<std::uint64_t> keys;
    std::map<std::uint64_t, std::size_t> index;
    const auto record = [&](const partition::PartitionNodeInfo& info) {
      const partition::BlockRect rect = partition::block_rect(
          geometry_, info.mi_row, info.mi_col, info.size);
      const std::uint64_t key = node_key(info.mi_row, info.mi_col, info.size);
      if (index.find(key) == index.end()) {
        index.emplace(key, nodes.size());
        keys.push_back(key);
        nodes.push_back(detail::VarianceNode{rect.x, rect.y, rect.visible_width,
                                             rect.visible_height});
      }
      partition::NodeCosts costs;
      costs.leaf = 0.0;
      costs.none_signal = 0.0;
      costs.split_signal = 0.0;
      costs.horz_signal = 0.0;
      costs.vert_signal = 0.0;
      return costs;
    };
    const partition::FrameDecision discovery =
        partition::decide_frame(geometry_, record, poptions);
    if (discovery.status != partition::DecisionStatus::kOk) {
      error = "partition discovery failed: " + discovery.error;
      return false;
    }

    std::vector<std::int64_t> costs;
    if (!backend_.variance_costs(nodes, costs, error)) return false;
    if (costs.size() != nodes.size()) {
      error = "backend returned the wrong variance-cost count";
      return false;
    }
    std::map<std::uint64_t, std::int64_t> cost_map;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      cost_map.emplace(keys[i], costs[i]);
    }

    const double lambda =
        0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;
    const auto cost_fn = [this, &cost_map, signal_cost](
                             const partition::PartitionNodeInfo& info) {
      partition::NodeCosts node_costs;
      const auto it =
          cost_map.find(node_key(info.mi_row, info.mi_col, info.size));
      if (it == cost_map.end()) {
        missing_cost_ = true;
        node_costs.leaf = partition::kInfiniteCost;
        return node_costs;
      }
      node_costs.leaf = static_cast<double>(it->second);
      node_costs.none_signal = 0.0;
      node_costs.split_signal = signal_cost;
      node_costs.horz_signal = signal_cost;
      node_costs.vert_signal = signal_cost;
      return node_costs;
    };

    missing_cost_ = false;
    frame_ = partition::decide_frame(geometry_, cost_fn, poptions);
    if (missing_cost_) {
      error = "partition decision queried a node with no device variance cost";
      return false;
    }
    if (frame_.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame_.error;
      return false;
    }
    if (!partition::validate_frame(geometry_, frame_, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    return true;
  }

  bool precompute_me(std::string& error) {
    me_results_.clear();
    me_index_.clear();
    if (keyframe_ || options_.force_global_mv || !options_.enable_newmv) {
      return true;
    }
    std::vector<partition::LeafBlock> leaves;
    for (const PartitionNode& sb : frame_.superblocks) {
      partition::collect_leaves(sb, leaves);
    }
    std::vector<detail::MeRequest> jobs;
    std::vector<std::uint64_t> job_keys;
    jobs.reserve(leaves.size() * static_cast<std::size_t>(num_total_refs_));
    job_keys.reserve(jobs.capacity());
    for (const partition::LeafBlock& leaf : leaves) {
      const int w = partition::block_width(leaf.size);
      const int h = partition::block_height(leaf.size);
      // The CPU reference only evaluates inter when the block fits inside the
      // reference planes (mi_cols*4 x mi_rows*4).  Edge blocks that straddle
      // the bottom/right boundary therefore never get an inter candidate.  ME
      // is skipped for them as well so the backend never reads out of plane.
      if (leaf.mi_col * 4 + w > mi_cols_ * 4 ||
          leaf.mi_row * 4 + h > mi_rows_ * 4) {
        continue;
      }
      for (int ref = 0; ref < num_total_refs_; ++ref) {
        job_keys.push_back(me_key(leaf.mi_row, leaf.mi_col, leaf.size, ref));
        detail::MeRequest request;
        request.x = leaf.mi_col * 4;
        request.y = leaf.mi_row * 4;
        request.w = w;
        request.h = h;
        request.ref_index = ref;
        jobs.push_back(request);
      }
    }
    if (jobs.empty()) return true;

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

    if (!backend_.me_batch(jobs, opts, me_results_, error)) return false;
    if (me_results_.size() != jobs.size()) {
      error = "backend returned the wrong ME result count";
      return false;
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      me_index_.emplace(job_keys[i], i);
    }
    return true;
  }

  bool collect_node(const PartitionNode& node, std::string& error) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    av2enc::video::syntax::BlockSyntax& entry = blocks_.back();
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
    if (node.forced) entry.flags |= av2enc::video::syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= av2enc::video::syntax::kFlagSbRoot;

    if (node.partition == PartitionType::kNone) {
      if (!encode_leaf(node, index, error)) return false;
    } else {
      for (const PartitionNode& child : node.children) {
        if (!collect_node(child, error)) return false;
      }
    }
    const std::size_t subtree = blocks_.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error = "partition subtree too large";
      return false;
    }
    blocks_[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

  merge::StackParams make_stack_params(int mi_row, int mi_col, int w4,
                                       int h4, int ref_index) const {
    merge::StackParams params;
    params.mi_row = mi_row;
    params.mi_col = mi_col;
    params.bw4 = w4;
    params.bh4 = h4;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mv_grid_.data();
    // The merge layer uses AV2 RefFrame labels: 0..6 are reference indices.
    params.ref_frame[0] = static_cast<std::int8_t>(ref_index);
    params.ref_frame[1] = merge::kRefNone;
    params.is_compound = false;
    params.global_mv[0] = inter::mc::Mv{0, 0};
    params.global_mv[1] = inter::mc::Mv{0, 0};
    // DRL reordering is rejected by the CLI and by the entry-point validation;
    // keep the spec-exact stack with reordering disabled.
    params.reorder = merge::DrlReorder::kDisabled;
    params.max_drl_bits_minus_1 = 4;
    if (num_total_refs_ > 1) {
      params.num_refs = num_total_refs_;
      params.frame_distance = frame_distance_.data();
    }
    return params;
  }

  bool load_stack(int mi_row, int mi_col, int w4, int h4, int ref_index,
                  merge::StackParams& params, merge::MvStack& stack) const {
    params = make_stack_params(mi_row, mi_col, w4, h4, ref_index);
    return merge::build_stack_checked(params, &stack) == merge::Status::kOk;
  }

  bool encode_leaf(const PartitionNode& node, std::size_t index,
                   std::string& error) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int cw = w / 2;
    const int ch = h / 2;
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;
    // Mirror the CPU reference: inter candidates are only evaluated when the
    // luma block fits inside the reference planes.  At the bottom/right frame
    // edge a 64x64 node can straddle the plane; the CPU's predict_plane()
    // rejects it, so the GPU path must not invent an inter candidate there.
    const bool inter_fits = x0 + w <= mi_cols_ * 4 && y0 + h <= mi_rows_ * 4;

    av2enc::video::syntax::BlockSyntax& entry = blocks_[index];

    InterCandidate best;
    const auto consider = [&](const InterCandidate& cand) {
      if (!cand.valid) return;
      if (!best.valid || cand.sse < best.sse) best = cand;
    };

    // Build the inter candidate list in the CPU's consideration order, once
    // per explicit reference: GLOBALMV, then NEARMV (DRL 0..limit-1), then
    // NEWMV.  One batched MC+SSE call scores all of them.
    std::vector<detail::InterJob> score_jobs;
    std::vector<int> score_kind;  // 0 global, 1 near, 2 new
    std::vector<int> score_drl;
    std::vector<int> score_ref;
    std::vector<inter::mc::Mv> score_mv;
    std::vector<inter::mc::Mv> score_mvd;

    const auto add_score = [&](int kind, int drl, inter::mc::Mv mv,
                               inter::mc::Mv mvd, int ref_index) {
      detail::InterJob job;
      job.x = x0;
      job.y = y0;
      job.w = w;
      job.h = h;
      job.mv_y = mv.y;
      job.mv_x = mv.x;
      job.ref_index = ref_index;
      score_jobs.push_back(job);
      score_kind.push_back(kind);
      score_drl.push_back(drl);
      score_ref.push_back(ref_index);
      score_mv.push_back(mv);
      score_mvd.push_back(mvd);
    };

    if (!keyframe_) {
      for (int ref_index = 0; ref_index < num_total_refs_; ++ref_index) {
        merge::StackParams stack_params;
        merge::MvStack stack;
        const bool have_stack =
            load_stack(r, c, w4, h4, ref_index, stack_params, stack);

        if (inter_fits) {
          add_score(0, 0, inter::mc::Mv{0, 0}, inter::mc::Mv{0, 0}, ref_index);
        }
        if (trace_ && inter_fits && !have_stack) {
          std::cerr << "  empty stack r=" << r << " c=" << c << " size="
                    << static_cast<int>(node.size) << " ref=" << ref_index
                    << "\n";
        }
        if (!inter_fits || !have_stack || options_.force_global_mv) continue;

        if (options_.enable_nearmv) {
          const int near_limit =
              (w > 32 && h > 32) ? stack.count : std::min(stack.count, 2);
          for (int i = 0; i < near_limit; ++i) {
            merge::ResolvedMode resolved;
            if (merge::resolve_single_mode(stack, stack_params,
                                           merge::SingleMode::kNearMv, i,
                                           inter::mc::Mv{0, 0}, &resolved) !=
                merge::Status::kOk) {
              continue;
            }
            add_score(1, i, resolved.mv[0], resolved.mvd[0], ref_index);
          }
        }

        if (options_.enable_newmv) {
          const auto it =
              me_index_.find(me_key(r, c, node.size, ref_index));
          if (it == me_index_.end() && trace_) {
            std::cerr << "  no me job r=" << r << " c=" << c << " size="
                      << static_cast<int>(node.size) << " stack=" << stack.count
                      << " ref=" << ref_index << "\n";
          }
          if (it != me_index_.end() && it->second < me_results_.size()) {
            const detail::MeOutcome& me_out = me_results_[it->second];
            const inter::mc::Mv me_mv = inter::mc::mv_from_me_1_16(
                me_out.mv_y, me_out.mv_x);
            if (options_.mv_precision != 0 && stack.count > 0) {
              // §5.20.7.13: quarter/half-pel frames keep the predictor as-is
              // and quantise the ME MV and the signalling MVD.  This mirrors
              // the CPU reference instead of routing through resolve_mode(),
              // which is only the 1/8-pel path.
              const inter::mc::Mv me_q =
                  quantize_mv(me_mv, options_.mv_precision);
              const inter::mc::Mv pred = stack.cand[0].mv[0];
              const inter::mc::Mv delta{me_q.y - pred.y, me_q.x - pred.x};
              const inter::mc::Mv mvd =
                  quantize_mv(delta, options_.mv_precision);
              add_score(2, 0, inter::mc::Mv{pred.y + mvd.y, pred.x + mvd.x},
                        mvd, ref_index);
            } else {
              const inter::mc::Mv pred = stack.cand[0].mv[0];
              const inter::mc::Mv mvd{me_mv.y - pred.y, me_mv.x - pred.x};
              merge::ResolvedMode resolved;
              if (merge::resolve_single_mode(stack, stack_params,
                                             merge::SingleMode::kNewMv, 0, mvd,
                                             &resolved) == merge::Status::kOk) {
                add_score(2, 0, resolved.mv[0], resolved.mvd[0], ref_index);
              }
            }
          }
        }
      }
    }

    if (!score_jobs.empty()) {
      std::vector<std::int64_t> sse;
      if (!backend_.inter_sse_batch(score_jobs, sse, error)) return false;
      if (sse.size() != score_jobs.size()) {
        error = "backend returned the wrong inter-SSE count";
        return false;
      }
      for (std::size_t i = 0; i < score_jobs.size(); ++i) {
        InterCandidate cand;
        cand.valid = true;
        cand.mode = score_kind[i] == 0
                        ? av2enc::video::syntax::kInterGlobalMv
                        : (score_kind[i] == 1
                               ? av2enc::video::syntax::kInterNearMv
                               : av2enc::video::syntax::kInterNewMv);
        cand.drl_index = score_drl[i];
        cand.ref_index = score_ref[i];
        cand.mv = score_mv[i];
        cand.mvd = score_mvd[i];
        cand.sse = sse[i];
        if (trace_) {
          std::cerr << "  cand kind=" << score_kind[i]
                    << " drl=" << score_drl[i]
                    << " ref=" << score_ref[i]
                    << " mv=(" << cand.mv.y << ","
                    << cand.mv.x << ") sse=" << cand.sse << "\n";
        }
        consider(cand);
      }
    }

    // Compound candidate (§5.20.7.10 comp_mode + §7.13.3.18 AVERAGE blend).
    // Gated subset: non-joint GLOBAL_GLOBALMV over two references, both MVs
    // zero, CWP_EQUAL average, no residual.
    if (!keyframe_ && options_.compound && num_total_refs_ >= 2 && w >= 8 &&
        h >= 8 && !options_.force_intra && inter_fits) {
      detail::InterJob job;
      job.x = x0;
      job.y = y0;
      job.w = w;
      job.h = h;
      job.mv_y = 0;
      job.mv_x = 0;
      job.ref_index = 0;
      std::vector<detail::InterJob> jobs{job};
      std::vector<std::int64_t> sse;
      if (!backend_.compound_sse_batch(jobs, sse, error)) return false;
      if (sse.size() == 1) {
        InterCandidate cand;
        cand.valid = true;
        cand.compound = true;
        cand.mode = av2enc::video::syntax::kInterGlobalMv;
        cand.ref_index = 0;
        cand.mv = inter::mc::Mv{0, 0};
        cand.sse = sse[0];
        if (trace_) {
          std::cerr << "  cand compound global-global sse=" << cand.sse
                    << "\n";
        }
        consider(cand);
      }
    }

    // Intra fallback: DC prediction + DCT residual, the M1 transform path.
    bool have_intra = false;
    std::int64_t intra_sse = 0;
    std::array<detail::PlaneTx, 3> intra_tx;
    if (keyframe_ || options_.enable_intra_fallback || options_.force_intra) {
      if (!backend_.intra_candidate(x0, y0, w, h, intra_tx, error)) {
        return false;
      }
      intra_sse =
          sse_against_source(source_, 0, x0, y0, w, h, intra_tx[0].recon);
      if (trace_) {
        std::cerr << "  cand intra sse=" << intra_sse << "\n";
      }
      have_intra = true;
    }

    // Optional inter residual tool (present in the extended CPU M2 path).
    // Mirrors the CPU decision: predict with the winning MV, transform the
    // residual, and compare SSE + lambda * heuristic bits against skip.
    const bool inter_residual_enabled =
        !keyframe_ && !best.compound && opt_enable_inter_residual(options_);
    bool inter_residual = false;
    double res_bits = 0.0;
    std::vector<std::uint8_t> inter_y;
    std::vector<std::uint8_t> inter_u;
    std::vector<std::uint8_t> inter_v;
    std::array<detail::PlaneTx, 3> res_tx;
    std::int64_t inter_luma_sse = best.sse;
    detail::InterJob best_job;
    best_job.x = x0;
    best_job.y = y0;
    best_job.w = w;
    best_job.h = h;
    best_job.mv_y = best.mv.y;
    best_job.mv_x = best.mv.x;
    best_job.ref_index = best.ref_index;
    if (inter_residual_enabled && best.valid) {
      if (!backend_.mc_420(best_job, inter_y, inter_u, inter_v, error)) {
        return false;
      }
      inter_luma_sse =
          sse_against_source(source_, 0, x0, y0, w, h, inter_y);
      const std::int64_t pred_total =
          inter_luma_sse +
          sse_against_source(source_, 1, cx0, cy0, cw, ch, inter_u) +
          sse_against_source(source_, 2, cx0, cy0, cw, ch, inter_v);
      if (!backend_.inter_residual_420(best_job, res_tx, error)) return false;
      const std::int64_t res_total =
          sse_against_source(source_, 0, x0, y0, w, h,
                             res_tx[0].recon) +
          sse_against_source(source_, 1, cx0, cy0, cw, ch,
                             res_tx[1].recon) +
          sse_against_source(source_, 2, cx0, cy0, cw, ch,
                             res_tx[2].recon);
      const int rcoded[3] = {w, cw, cw};
      const int rcoded_h[3] = {h, ch, ch};
      int eobs[3] = {0, 0, 0};
      for (int plane = 0; plane < 3; ++plane) {
        const auto params_opt = txv::make_tx_params(
            rcoded[plane], rcoded_h[plane], txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error = "unsupported inter residual transform size";
          return false;
        }
        eobs[plane] = eob_from_levels(
            res_tx[static_cast<std::size_t>(plane)].levels,
            params_opt->w, params_opt->h, nullptr);
      }
      const bool all_zero = eobs[0] == 0 && eobs[1] == 0 && eobs[2] == 0;
      const double kLambda = opt_inter_rd_lambda(options_);
      res_bits =
          3.0 + estimate_tx_bits(eobs[0], res_tx[0].levels, true) +
          estimate_tx_bits(eobs[1], res_tx[1].levels, false) +
          estimate_tx_bits(eobs[2], res_tx[2].levels, false);
      const bool want =
          opt_force_inter_residual(options_) ||
          (!all_zero &&
           static_cast<double>(res_total) + kLambda * res_bits <
               static_cast<double>(pred_total) + kLambda * 2.0);
      if (want) {
        inter_residual = true;
        inter_y = res_tx[0].recon;
        inter_u = res_tx[1].recon;
        inter_v = res_tx[2].recon;
        inter_luma_sse =
            sse_against_source(source_, 0, x0, y0, w, h, inter_y);
      }
      if (trace_) {
        std::cerr << "  cand inter final sse=" << inter_luma_sse
                  << " residual=" << (inter_residual ? 1 : 0) << "\n";
      }
    }

    bool use_intra = keyframe_ || options_.force_intra || !best.valid;
    if (!use_intra && have_intra) {
      if (opt_enable_inter_residual(options_)) {
        // Rate-aware decision, mirroring the CPU residual path.
        const double lambda = opt_inter_rd_lambda(options_);
        int intra_eobs[3] = {0, 0, 0};
        double intra_bits = 3.0;
        const int coded_w[3] = {w, cw, cw};
        const int coded_h[3] = {h, ch, ch};
        for (int plane = 0; plane < 3; ++plane) {
          const auto params_opt = txv::make_tx_params(
              coded_w[plane], coded_h[plane], txv::TxType::kDctDct);
          if (!params_opt.has_value()) {
            error = "unsupported intra transform size";
            return false;
          }
          intra_eobs[plane] = eob_from_levels(
              intra_tx[static_cast<std::size_t>(plane)].levels,
              params_opt->w, params_opt->h, nullptr);
          intra_bits += estimate_tx_bits(
              intra_eobs[plane],
              intra_tx[static_cast<std::size_t>(plane)].levels,
              false);
        }
        double inter_bits = 8.0;
        if (inter_residual) inter_bits += res_bits;
        use_intra = static_cast<double>(intra_sse) + lambda * intra_bits <
                    static_cast<double>(inter_luma_sse) + lambda * inter_bits;
      } else {
        use_intra = intra_sse < inter_luma_sse;
      }
    }

    if (use_intra) {
      if (!have_intra) {
        error = "intra fallback unavailable";
        return false;
      }
      entry.is_inter = 0;
      entry.skip_flag = 0;
      entry.y_mode = 0;  // DC_PRED
      entry.y_angle_delta = 0;
      entry.uv_mode = 0;  // DC_PRED
      write_plane_block(recon_[0], x0, y0, w, h, intra_tx[0].recon);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_tx[1].recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_tx[2].recon);
      if (!backend_.write_recon_420(x0, y0, w, h, intra_tx[0].recon,
                                    intra_tx[1].recon, intra_tx[2].recon,
                                    error)) {
        return false;
      }
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      const int coded[3] = {w, cw, cw};
      const int coded_h[3] = {h, ch, ch};
      for (int plane = 0; plane < 3; ++plane) {
        const auto params_opt = txv::make_tx_params(
            coded[plane], coded_h[plane], txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error = "unsupported intra transform size";
          return false;
        }
        const txv::TxParams& params = *params_opt;
        const std::vector<std::int32_t>& plane_levels =
            intra_tx[static_cast<std::size_t>(plane)].levels;
        if (plane_levels.size() !=
            static_cast<std::size_t>(params.w) *
                static_cast<std::size_t>(params.h)) {
          error = "backend returned the wrong level count";
          return false;
        }
        int eob = 0;
        std::vector<std::uint16_t> scan;
        eob = eob_from_levels(plane_levels, params.w, params.h, &scan);
        entry.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);
        entry.eob[plane] = static_cast<std::uint32_t>(eob);
        for (int c2 = 0; c2 < eob; ++c2) {
          levels_.push_back(
              plane_levels[scan[static_cast<std::size_t>(c2)]]);
        }
      }
      update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
      ++intra_blocks_;
      if (trace_) {
        std::cerr << "m2 intra r=" << r << " c=" << c << " " << w << "x" << h
                  << " sse=" << intra_sse << "\n";
      }
    } else {
      if (!inter_residual_enabled) {
        if (best.compound) {
          if (!backend_.compound_420(best_job, inter_y, inter_u, inter_v,
                                     error)) {
            return false;
          }
        } else if (!backend_.mc_420(best_job, inter_y, inter_u, inter_v,
                                    error)) {
          return false;
        }
      }
      entry.is_inter = 1;
      entry.skip_flag = inter_residual ? 0 : 1;
      entry.inter_mode = static_cast<std::uint8_t>(best.mode);
      entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
      entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
      entry.compound = best.compound ? 1 : 0;
      entry.ref_index1 = 1;
      entry.compound_joint = 0;
      entry.compound_mode = 3;  // GLOBAL_GLOBALMV
      entry.mv_y = best.mv.y;
      entry.mv_x = best.mv.x;
      entry.mvd_y = best.mvd.y;
      entry.mvd_x = best.mvd.x;
      write_plane_block(recon_[0], x0, y0, w, h, inter_y);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, inter_u);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, inter_v);
      if (inter_residual) {
        if (!backend_.write_recon_420(x0, y0, w, h, inter_y, inter_u, inter_v,
                                      error)) {
          return false;
        }
        entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
        const int coded[3] = {w, cw, cw};
        const int coded_h[3] = {h, ch, ch};
        for (int plane = 0; plane < 3; ++plane) {
          const auto params_opt = txv::make_tx_params(
              coded[plane], coded_h[plane], txv::TxType::kDctDct);
          if (!params_opt.has_value()) {
            error = "unsupported inter residual transform size";
            return false;
          }
          const txv::TxParams& params = *params_opt;
          const std::vector<std::int32_t>& plane_levels =
              res_tx[static_cast<std::size_t>(plane)].levels;
          int eob = 0;
          std::vector<std::uint16_t> scan;
          eob = eob_from_levels(plane_levels, params.w, params.h, &scan);
          entry.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);
          entry.eob[plane] = static_cast<std::uint32_t>(eob);
          for (int c2 = 0; c2 < eob; ++c2) {
            levels_.push_back(
                plane_levels[scan[static_cast<std::size_t>(c2)]]);
          }
        }
      }
      update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv,
                     best.compound ? 1 : merge::kRefNone);
      ++inter_blocks_;
      if (trace_) {
        std::cerr << "m2 inter r=" << r << " c=" << c << " " << w << "x" << h
                  << " mode=" << static_cast<int>(best.mode)
                  << " drl=" << best.drl_index
                  << " ref=" << best.ref_index << " mv=(" << best.mv.y << ","
                  << best.mv.x << ") sse=" << inter_luma_sse
                  << " skip=" << static_cast<int>(entry.skip_flag)
                  << " comp=" << static_cast<int>(entry.compound) << "\n";
      }
    }

    (void)node;
    return true;
  }

  void update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                      int ref_index, inter::mc::Mv mv,
                      int ref_index1 = merge::kRefNone) {
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
        // AV2 RefFrame labels used by the merge layer: 0..6 are reference
        // indices, 8 = INTRA.
        mi.ref[0] = is_inter ? static_cast<std::int8_t>(ref_index)
                             : merge::kIntraFrame;
        mi.ref[1] = static_cast<std::int8_t>(ref_index1);
        mi.mv[0][0] = mv.y;
        mi.mv[0][1] = mv.x;
        mi.mv[1][0] = mv.y;
        mi.mv[1][1] = mv.x;
        mi.is_inter = static_cast<std::int8_t>(is_inter);
        mi.w4 = static_cast<std::int8_t>(w4);
        mi.h4 = static_cast<std::int8_t>(h4);
        mi.col_base = static_cast<std::int16_t>(c);
        mi.row_base = static_cast<std::int16_t>(r);
      }
    }
  }

  detail::Backend& backend_;
  const VideoInputFrame& source_;
  const VideoOptions& options_;
  std::array<int, 7> frame_distance_{};
  int num_total_refs_ = 1;
  std::span<const std::uint16_t> cdf_seed_;
  std::vector<std::uint16_t>* cdf_final_ = nullptr;
  bool trace_ = false;
  bool keyframe_ = false;
  bool begin_ok_ = true;
  int width_ = 0;
  int height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  FrameGeometry geometry_;
  partition::FrameDecision frame_;
  bool missing_cost_ = false;
  Plane recon_[3];
  std::vector<inter::mc::MiInfo> mv_grid_;
  std::vector<av2enc::video::syntax::BlockSyntax> blocks_;
  std::vector<std::int32_t> levels_;
  std::vector<detail::MeOutcome> me_results_;
  std::map<std::uint64_t, std::size_t> me_index_;
  std::uint32_t inter_blocks_ = 0;
  std::uint32_t intra_blocks_ = 0;
};

VideoReconFrame to_recon_frame(const Plane& y, const Plane& u, const Plane& v,
                            std::uint32_t width, std::uint32_t height) {
  VideoReconFrame out;
  out.width = width;
  out.height = height;
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  out.y.resize(static_cast<std::size_t>(width) * height);
  out.u.resize(static_cast<std::size_t>(cw) * ch);
  out.v.resize(static_cast<std::size_t>(cw) * ch);
  for (std::uint32_t row = 0; row < height; ++row) {
    std::memcpy(out.y.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(width),
                y.samples.data() + static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(y.stride),
                width);
  }
  for (std::uint32_t row = 0; row < ch; ++row) {
    std::memcpy(out.u.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(cw),
                u.samples.data() + static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(u.stride),
                cw);
    std::memcpy(out.v.data() + static_cast<std::size_t>(row) *
                                   static_cast<std::size_t>(cw),
                v.samples.data() + static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(v.stride),
                cw);
  }
  return out;
}

int relative_dist(std::uint32_t a, std::uint32_t b, int bits) {
  const int modulus = 1 << bits;
  int diff = static_cast<int>((a - b) & static_cast<std::uint32_t>(modulus - 1));
  if (diff >= (modulus >> 1)) diff -= modulus;
  return diff;
}

VideoConfig make_config(std::uint32_t width, std::uint32_t height,
                     const VideoOptions& options) {
  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames =
      options.num_total_refs > 1 ? 8 : 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  set_config_drl_reorder(config, opt_drl_reorder(options));
  config.enable_cfl_intra = true;
  if (options.mfh) config.cur_mfh_id = 1;
  if (options.video_timing) {
    // §5.4.1 decoder-compatible timing block.  The nested §5.4.13 block stays
    // off: pinned dav2d f9dafbc cannot parse it (docs/negative-results.md).
    config.initial_display_delay_present = true;
    config.initial_display_delay_minus_1 = 0;
    config.decoder_model_info_present = true;
    config.num_units_in_decoding_tick =
        options.timing_num_units_in_display_tick;
  }
  return config;
}

}  // namespace

namespace detail {
namespace {

using av2enc::video::VideoConfig;
using av2enc::video::VideoFrameHeader;
using av2enc::video::VideoInputFrame;
using av2enc::video::VideoOptions;
using av2enc::video::VideoReconFrame;
using av2enc::video::VideoSequenceResult;

// Fills the per-frame header fields from a B/temporal plan.  The syntax/header
// writers are the CPU reference's; only the tile payload is device-produced.
VideoFrameHeader make_plan_header(bool keyframe, std::uint32_t order_hint,
                                  int num_refs, const int* ref_slots,
                                  std::uint32_t refresh_flags,
                                  bool immediate_output,
                                  bool implicit_output, bool reference_select,
                                  const VideoOptions& options) {
  VideoFrameHeader header;
  header.keyframe = keyframe;
  header.order_hint = order_hint;
  header.num_total_refs = static_cast<std::uint8_t>(num_refs);
  for (int r = 0; r < num_refs; ++r) {
    header.ref_frame_idx[static_cast<std::size_t>(r)] =
        static_cast<std::uint8_t>(ref_slots[r]);
  }
  header.ref_slot = num_refs > 0 ? header.ref_frame_idx[0] : 0;
  header.refresh_frame_flags = refresh_flags;
  header.immediate_output_frame = immediate_output;
  header.implicit_output_frame = implicit_output;
  header.reference_select = reference_select;
  header.use_qtr_precision_mv = options.mv_precision == 1;
  header.allow_high_precision_mv = options.mv_precision != 2;
  return header;
}

bool common_gop_checks(std::span<const VideoInputFrame> frames,
                       const VideoOptions& options, std::string& error) {
  if (!options.frame_base_q_idx.empty()) {
    error = "the GPU B-frame/temporal-layer plan requires a fixed base_q_idx";
    return false;
  }
  if (frames.empty()) {
    error = "no input frames";
    return false;
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  if ((width & 7U) != 0 || (height & 7U) != 0) {
    error = "M2 requires width/height divisible by 8";
    return false;
  }
  for (const VideoInputFrame& f : frames) {
    if (f.width != width || f.height != height) {
      error = "all frames must have the same dimensions";
      return false;
    }
  }
  return true;
}

}  // namespace

bool encode_b_sequence_gpu(Backend& backend,
                           std::span<const VideoInputFrame> frames,
                           const VideoOptions& options,
                           VideoSequenceResult& result, std::string& error) {
  namespace bf = ::av2enc::inter::bframes;
  if (!common_gop_checks(frames, options, error)) return false;
  const int gop_size = options.b_frames + 2;
  if (gop_size > 16) {
    error = "--b-frames supports at most 6 B frames per GOP";
    return false;
  }
  if (frames.size() != static_cast<std::size_t>(gop_size)) {
    error = "--b-frames codes one closed GOP: pass exactly " +
            std::to_string(gop_size) + " frames (N B frames + key + P anchor)";
    return false;
  }
  bf::SequenceConfig sc;
  sc.num_gops = static_cast<int>(frames.size()) / gop_size;
  sc.gop_size = gop_size;
  sc.pattern = bf::GopPattern::kIpbb;
  sc.order_hint_bits = 4;
  sc.num_slots = 16;
  sc.base_q_idx = options.base_q_idx;
  sc.leading_pictures = 0;
  if (sc.num_gops != 1) {
    error = "--b-frames codes exactly one GOP";
    return false;
  }
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
  bf::ValidationReport report;
  if (!bf::validate_plan(plan, &report)) {
    error = std::string("B-frame plan invalid: ") +
            bf::plan_error_name(report.error) + " (coding " +
            std::to_string(report.error_coding_index) + ", display " +
            std::to_string(report.error_display_index) + ")";
    return false;
  }

  VideoConfig config;
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = static_cast<std::uint8_t>(plan.order_hint_bits);
  config.num_ref_frames = static_cast<std::uint8_t>(plan.num_slots);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.max_tlayer_id = 0;
  config.monotonic_output_order_flag = false;

  const int mi_cols = 2 * static_cast<int>((frames[0].width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((frames[0].height + 7U) >> 3U);
  std::vector<VideoReconFrame> recon_by_display(
      static_cast<std::size_t>(plan.display_count()));
  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;

  bool wrote_sequence_header = false;
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    const int display = plan.coding_order[ci];
    const bf::FramePlan* fp = plan.frame_at_display(display);
    if (fp == nullptr) {
      error = "B-frame plan is missing a display index";
      return false;
    }
    const int num_refs = fp->num_total_refs;
    std::vector<const VideoReconFrame*> references;
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < num_refs; ++r) {
      const int ref_display = fp->ref_display[r];
      if (ref_display < 0 ||
          recon_by_display[static_cast<std::size_t>(ref_display)].y.empty()) {
        error = "B-frame reference is not available";
        return false;
      }
      references.push_back(
          &recon_by_display[static_cast<std::size_t>(ref_display)]);
      frame_distance[static_cast<std::size_t>(r)] =
          relative_dist(static_cast<std::uint32_t>(fp->display_order_hint),
                        static_cast<std::uint32_t>(fp->ref_order_hint[r]),
                        plan.order_hint_bits);
    }

    GpuFrameEncoder encoder(
        backend, frames[static_cast<std::size_t>(display)], references,
        frame_distance, std::max(1, num_refs), frame_options, options.trace,
        fp->is_key, error);
    if (!encoder.begin_ok()) {
      error = "B-frame display " + std::to_string(display) +
              " backend setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::vector<av2enc::video::syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "B-frame display " + std::to_string(display) + ": " + error;
      return false;
    }

    const bf::FrameHeaderFields hf = bf::header_fields(*fp);
    int ref_slots[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int r = 0; r < hf.num_total_refs; ++r) {
      ref_slots[r] = hf.ref_frame_idx[r];
    }
    const VideoFrameHeader header = make_plan_header(
        hf.keyframe, hf.order_hint_lsbs, hf.num_total_refs, ref_slots,
        hf.refresh_frame_flags, hf.immediate_output_frame,
        hf.implicit_output_frame,
        hf.reference_select && options.compound && hf.num_total_refs > 1,
        options);

    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq =
          av2enc::video::make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const std::vector<std::uint8_t> obu =
        av2enc::video::make_frame_obu(config, header, tile, 0);
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
  }

  for (int display = 0; display < plan.display_count(); ++display) {
    result.recon.push_back(recon_by_display[static_cast<std::size_t>(display)]);
  }
  result.frame_output_index.resize(plan.coding_order.size());
  for (std::size_t ci = 0; ci < plan.coding_order.size(); ++ci) {
    result.frame_output_index[ci] = plan.coding_order[ci];
  }
  (void)mi_cols;
  (void)mi_rows;
  return true;
}

bool encode_temporal_sequence_gpu(Backend& backend,
                                  std::span<const VideoInputFrame> frames,
                                  const VideoOptions& options,
                                  VideoSequenceResult& result,
                                  std::string& error) {
  namespace tl = ::av2enc::inter::tlayers;
  if (!common_gop_checks(frames, options, error)) return false;
  const int gop = options.temporal_layers;
  if (frames.size() != static_cast<std::size_t>(gop)) {
    error = "--video-tlayers codes one closed GOP: pass exactly " +
            std::to_string(gop) + " input frames";
    return false;
  }
  tl::PlanConfig pc;
  pc.gop_size = gop;
  pc.schedule = tl::Schedule::kHierarchicalP;
  pc.dependency = tl::DependencyMode::kAllLower;
  pc.explicit_dependency_map = false;
  pc.order_hint_bits = 4;
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
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = static_cast<std::uint8_t>(pc.order_hint_bits);
  config.num_ref_frames = static_cast<std::uint8_t>(plan.num_slots);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.max_tlayer_id = static_cast<std::uint8_t>(plan.max_tlayer_id);
  config.monotonic_output_order_flag = false;
  config.tlayer_dependency_present_flag = plan.tlayer_dependency_present_flag;
  for (int curr = 0; curr <= plan.max_tlayer_id; ++curr) {
    for (int ref = 0; ref <= plan.max_tlayer_id; ++ref) {
      config.tlayer_dependency[curr][ref] =
          plan.tlayer_dependency[static_cast<std::size_t>(curr)]
                                [static_cast<std::size_t>(ref)];
    }
  }

  std::vector<VideoReconFrame> recon_by_display(
      static_cast<std::size_t>(plan.display_count()));
  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;

  bool wrote_sequence_header = false;
  for (std::size_t ci = 0; ci < coding.size(); ++ci) {
    const int display = coding[ci];
    const tl::PlannedFrame& pf =
        plan.frames[static_cast<std::size_t>(display)];
    if (pf.is_key && ci != 0) {
      error = "temporal-layer plan codes a key outside coding position 0";
      return false;
    }

    std::vector<const VideoReconFrame*> references;
    std::array<int, 7> frame_distance{};
    if (!pf.is_key) {
      const int ref_display = pf.ref_display[0];
      if (ref_display < 0 ||
          recon_by_display[static_cast<std::size_t>(ref_display)].y.empty()) {
        error = "temporal-layer reference is not available";
        return false;
      }
      references.push_back(
          &recon_by_display[static_cast<std::size_t>(ref_display)]);
      frame_distance[0] = relative_dist(
          pf.display_order_hint, pf.ref_order_hint[0],
          static_cast<int>(pc.order_hint_bits));
    }

    GpuFrameEncoder encoder(backend, frames[static_cast<std::size_t>(display)],
                            references, frame_distance, 1, frame_options,
                            options.trace, pf.is_key, error);
    if (!encoder.begin_ok()) {
      error = "temporal-layer frame " + std::to_string(display) +
              " backend setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::vector<av2enc::video::syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "temporal-layer frame " + std::to_string(display) + ": " + error;
      return false;
    }

    int ref_slots[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int r = 0; r < pf.num_refs; ++r) {
      ref_slots[r] = pf.ref_slot[r];
    }
    const VideoFrameHeader header = make_plan_header(
        pf.is_key, pf.order_hint_lsbs, pf.num_refs, ref_slots,
        pf.refresh_frame_flags, pf.immediate_output_frame,
        pf.implicit_output_frame, false, options);

    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq =
          av2enc::video::make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const std::vector<std::uint8_t> obu = av2enc::video::make_frame_obu(
        config, header, tile, static_cast<std::uint8_t>(pf.obu_tlayer_id()));
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
  }

  std::vector<int> retained(coding);
  std::sort(retained.begin(), retained.end());
  std::vector<std::int32_t> rank(static_cast<std::size_t>(plan.display_count()),
                                 -1);
  for (std::size_t i = 0; i < retained.size(); ++i) {
    const int display = retained[i];
    rank[static_cast<std::size_t>(display)] = static_cast<std::int32_t>(i);
    result.recon.push_back(recon_by_display[static_cast<std::size_t>(display)]);
  }
  result.frame_output_index.resize(coding.size());
  for (std::size_t ci = 0; ci < coding.size(); ++ci) {
    result.frame_output_index[ci] =
        rank[static_cast<std::size_t>(coding[ci])];
  }
  return true;
}

}  // namespace detail

bool encode_video_sequence_gpu(std::span<const VideoInputFrame> frames,
                            const VideoOptions& options, VideoSequenceResult& result,
                            std::string& error, VideoGpuStats* stats,
                            av2enc::cuda::Arena* arena) {
  error.clear();
  result = VideoSequenceResult{};
  if (frames.empty()) {
    error = "no input frames";
    return false;
  }
  if (detail::is_frame_type_mode(options)) {
    return detail::encode_special_sequence_gpu(frames, options, result, error,
                                               stats);
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  if (width == 0 || height == 0 || (width & 7U) != 0 || (height & 7U) != 0) {
    error = "M2 requires non-zero width/height divisible by 8";
    return false;
  }
  if (width > 0xFFFFU || height > 0xFFFFU) {
    error = "M2 dimensions exceed the sequence header field width";
    return false;
  }
  if (options.base_q_idx == 0) {
    error = "M2 does not support lossless base_q_idx = 0";
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
  const auto frame_qp = [&options](std::size_t index) -> std::uint8_t {
    return options.frame_base_q_idx.empty()
               ? options.base_q_idx
               : options.frame_base_q_idx[index];
  };
  // The device transform chain wires the shared `tx_variants` quantiser, whose
  // quant_factor and dequant are pinned to qlookup(90).  Per-frame QP would
  // need the CPU `compute_tx()` quant-step scaling ported first; reject it
  // rather than emit bytes that differ from the CPU reference.
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frame_qp(i) != 90) {
      error = "GPU M2 baseline pins base_q_idx = 90 (the device transform "
              "quantiser is qlookup(90))";
      return false;
    }
  }
  // The GPU pipeline mirrors the CPU reference model.  Every option the CPU
  // uses to build a different sequence/ref model that this backend has not
  // implemented is rejected by name here as well as in the CLI, so a caller
  // can never get bytes from a mode this backend did not implement.
  if (options.num_total_refs < 1 || options.num_total_refs > 3) {
    error = "num_total_refs must be in [1, 3]";
    return false;
  }
  if (options.long_term || options.olk || options.leading_frames != 0 ||
      options.sef || options.ra_switch || options.ras) {
    error = "GPU M2 baseline does not implement the frame-type wrappers here";
    return false;
  }
  if (options.tile_cols > 1 || options.tile_rows > 1) {
    error = "GPU M2 baseline emits a single tile";
    return false;
  }
  if (options.cdf_continuity) {
    error = "GPU M2 baseline does not implement cross-frame CDF carry";
    return false;
  }
  if (!options.frame_base_q_idx.empty()) {
    error = "GPU M2 baseline does not implement per-frame QP";
    return false;
  }
  if (opt_drl_reorder(options) != 0) {
    error = "GPU M2 baseline keeps DRL reordering off (not decoder-gated)";
    return false;
  }
  if (options.mv_precision < 0 || options.mv_precision > 2) {
    error = "mv_precision must be 0 (1/8), 1 (1/4) or 2 (1/2) pel";
    return false;
  }
  if (opt_drl_reorder(options) < 0 || opt_drl_reorder(options) > 2) {
    error = "M2 DrlReorder must be 0 (disabled), 1 (constraint) or 2 (always)";
    return false;
  }
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const VideoInputFrame& f = frames[i];
    if (f.width != width || f.height != height) {
      error = "all frames must have the same dimensions";
      return false;
    }
    const std::size_t luma = static_cast<std::size_t>(width) * height;
    const std::size_t chroma = static_cast<std::size_t>((width + 1U) / 2U) *
                               ((height + 1U) / 2U);
    if (f.y.size() < luma || f.u.size() < chroma || f.v.size() < chroma) {
      error = "input frame planes are too small";
      return false;
    }
  }

  if (options.b_frames > 0 || options.temporal_layers > 0) {
    std::unique_ptr<detail::Backend> gop_backend = detail::make_backend(arena);
    if (!gop_backend) {
      error = "no M2 backend linked";
      return false;
    }
    gop_backend->reset_stats();
    const bool ok = options.b_frames > 0
                        ? detail::encode_b_sequence_gpu(
                              *gop_backend, frames, options, result, error)
                        : detail::encode_temporal_sequence_gpu(
                              *gop_backend, frames, options, result, error);
    if (!ok) return false;
    if (stats != nullptr) {
      *stats = gop_backend->stats();
      std::uint64_t inter = 0;
      std::uint64_t intra = 0;
      for (std::size_t i = 0; i < result.inter_blocks.size(); ++i) {
        inter += result.inter_blocks[i];
        intra += result.intra_blocks[i];
      }
      stats->inter_blocks = inter;
      stats->intra_blocks = intra;
    }
    return true;
  }

  std::unique_ptr<detail::Backend> backend = detail::make_backend(arena);
  if (!backend) {
    error = "no M2 backend linked";
    return false;
  }
  backend->reset_stats();

  VideoConfig config = make_config(width, height, options);

  // §8.2.4 average-CDF / §7.5 frame-end bookkeeping.  The decision path and
  // tile writer are the CPU reference's; only the seed/final CDF images are
  // threaded through the device backend so they stay host-side serial state.
  const bool cdf_chain = options.avg_cdf || options.adaptive_cdf;
  std::vector<std::uint16_t> cdf_tile;
  std::vector<std::uint16_t> cdf_scratch;
  std::vector<std::uint16_t> cdf_frame;
  std::vector<std::uint16_t> cdf_saved;
  std::vector<std::uint16_t> cdf_defaults;
  std::vector<entropy::avg_cdf::CdfTable> cdf_bank_tables;
  entropy::avg_cdf::CdfBank cdf_bank;
  std::array<std::vector<std::uint16_t>, 8> cdf_slots;
  if (cdf_chain) {
    const std::size_t words = av2enc::video::syntax::cdf_image_words();
    cdf_frame.assign(words, 0);
    cdf_tile.assign(words, 0);
    cdf_saved.assign(words, 0);
    cdf_defaults.assign(words, 0);
    if (!av2enc::video::syntax::cdf_image_defaults_for_base_q_idx(
            static_cast<int>(options.base_q_idx), cdf_defaults, &error)) {
      return false;
    }
    if (!av2enc::video::syntax::cdf_image_tables(
            cdf_frame, cdf_tile, cdf_saved, cdf_defaults, cdf_bank_tables,
            &error)) {
      return false;
    }
    entropy::avg_cdf::CdfBankConfig bank_config;
    bank_config.enable_avg_cdf = options.avg_cdf;
    bank_config.avg_cdf_type = true;
    bank_config.context_update_tile_id = 0;
    if (!cdf_bank.init(cdf_bank_tables, bank_config, &error)) {
      error = "average-CDF bank: " + error;
      return false;
    }
    cdf_bank.begin_frame(/*reset_to_defaults=*/true);
  }

  // Frame 0: intra keyframe (DC intra + DCT), mirroring the CPU reference.
  {
    const std::array<int, 7> no_distance{};
    VideoOptions key_options = options;
    key_options.base_q_idx = frame_qp(0);
    key_options.frame_base_q_idx.clear();
    GpuFrameEncoder key_encoder(
        *backend, frames[0],
        std::span<const VideoReconFrame* const>{}, no_distance, 1, key_options,
        options.trace, /*keyframe=*/true, error,
        cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                  : std::span<const std::uint16_t>(),
        cdf_chain ? &cdf_scratch : nullptr);
    if (!key_encoder.begin_ok()) {
      error = "keyframe backend setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> key_tile;
    std::vector<av2enc::video::syntax::BlockSyntax> key_blocks;
    std::vector<std::int32_t> key_levels;
    if (!key_encoder.encode(key_tile, key_blocks, key_levels, error)) {
      error = "keyframe encode failed: " + error;
      return false;
    }
    if (key_tile.empty()) {
      error = "keyframe produced an empty tile payload";
      return false;
    }
    if (cdf_chain) {
      // exit_symbol() -> §8.2.4 Saved* average (§7.5 runs after the last tile).
      std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
      if (!cdf_bank.save_tile_cdfs(0, 1)) {
        error = "average-CDF save_tile_cdfs rejected the keyframe tile";
        return false;
      }
      cdf_bank.average_at_frame_end();
      for (std::vector<std::uint16_t>& slot : cdf_slots) {
        slot.assign(cdf_frame.begin(), cdf_frame.end());
      }
    }

    VideoFrameHeader key_header;
    key_header.keyframe = true;
    key_header.order_hint = 0;
    key_header.ref_slot = 0;
    key_header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
    key_header.use_qtr_precision_mv = options.mv_precision == 1;
    key_header.allow_high_precision_mv = options.mv_precision != 2;

    config.base_q_idx = frame_qp(0);

    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq =
        av2enc::video::make_sequence_header_obu(config);
    std::vector<std::uint8_t> ci;
    if (options.video_timing) {
      // §5.15 content interpretation OBU carrying §5.4.12 timing_info().  It is
      // the first OBU of the coded output frame unit after the sequence header.
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
    std::vector<std::uint8_t> mfh_obu;
    if (options.mfh) {
      // §5.7: one minimal MFH (sequence header 0, id 1, no frame size /
      // deblocking update / segmentation table).  Availability precedes the
      // first frame header that references it (§7.3.8.7).
      av2enc::frame::mfh::MultiFrameHeader header;
      header.seq_header_id = 0;
      header.mfh_id = 1;
      header.frame_size_present = false;
      header.deblocking_filter_update = false;
      header.seg_info_present = false;
      std::string mfh_error;
      mfh_obu =
          av2enc::frame::mfh::make_multi_frame_header_obu(header, &mfh_error);
      if (mfh_obu.empty()) {
        error = "MFH OBU: " + mfh_error;
        return false;
      }
    }
    const std::vector<std::uint8_t> obu =
        av2enc::video::make_frame_obu(config, key_header, key_tile);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + seq.size() + ci.size() + mfh_obu.size() +
                 obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq.begin(), seq.end());
    unit.insert(unit.end(), ci.begin(), ci.end());
    unit.insert(unit.end(), mfh_obu.begin(), mfh_obu.end());
    unit.insert(unit.end(), obu.begin(), obu.end());

    result.units.push_back(std::move(unit));
    result.recon.push_back(to_recon_frame(
        key_encoder.recon(0), key_encoder.recon(1), key_encoder.recon(2),
        width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(key_tile.size()));
    result.inter_blocks.push_back(0);
    result.intra_blocks.push_back(key_encoder.intra_blocks());
  }

  // Decoded picture buffer.  This mirrors the CPU reference exactly: the key
  // frame refreshes every slot, `recent_slots` is kept most-recent-first, and
  // each P frame references the first `num_total_refs` entries and refreshes
  // slot `i % NumRefFrames`.
  std::array<VideoReconFrame, 8> slot_recon;
  std::array<std::uint32_t, 8> slot_order{};
  std::vector<int> recent_slots;
  slot_recon[0] = result.recon.back();
  slot_order[0] = 0;
  recent_slots.push_back(0);

  for (std::size_t i = 1; i < frames.size(); ++i) {
    const std::uint32_t order_hint =
        static_cast<std::uint32_t>(i) & 0xFU;
    const int ref_count = std::min<int>(
        options.num_total_refs, static_cast<int>(recent_slots.size()));
    std::vector<const VideoReconFrame*> references;
    references.reserve(static_cast<std::size_t>(ref_count));
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < ref_count; ++r) {
      const int slot = recent_slots[static_cast<std::size_t>(r)];
      references.push_back(&slot_recon[static_cast<std::size_t>(slot)]);
      frame_distance[static_cast<std::size_t>(r)] =
          relative_dist(order_hint, slot_order[static_cast<std::size_t>(slot)],
                        4);
    }
    VideoOptions frame_options = options;
    frame_options.base_q_idx = frame_qp(i);
    frame_options.frame_base_q_idx.clear();
    if (cdf_chain) cdf_bank.begin_frame(/*reset_to_defaults=*/true);
    std::vector<std::uint16_t> cdf_seed;
    if (cdf_chain) cdf_seed = cdf_tile;
    GpuFrameEncoder encoder(*backend, frames[i], references, frame_distance,
                            ref_count, frame_options, options.trace,
                            /*keyframe=*/false, error,
                            cdf_chain ? std::span<const std::uint16_t>(cdf_seed)
                                      : std::span<const std::uint16_t>(),
                            cdf_chain ? &cdf_scratch : nullptr);
    if (!encoder.begin_ok()) {
      error = "P-frame " + std::to_string(i) + " setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::vector<av2enc::video::syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }
    const int write_slot = static_cast<int>(
        static_cast<unsigned>(i) % static_cast<unsigned>(config.num_ref_frames));
    if (cdf_chain) {
      std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
      if (!cdf_bank.save_tile_cdfs(0, 1)) {
        error = "average-CDF save_tile_cdfs rejected the P-frame tile";
        return false;
      }
      cdf_bank.average_at_frame_end();
      std::vector<std::uint16_t>& saved_slot =
          cdf_slots[static_cast<std::size_t>(write_slot)];
      std::copy(cdf_frame.begin(), cdf_frame.end(), saved_slot.begin());
    }

    VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = order_hint;
    if (cdf_chain) {
      header.signal_primary_ref_frame = false;
      header.disable_cross_frame_cdf_init = true;
      header.primary_ref_frame = 0;
    }
    header.num_total_refs = static_cast<std::uint8_t>(ref_count);
    for (int r = 0; r < ref_count; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(recent_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    // §7.23 rotating refresh slot: the CPU reference writes slot i modulo
    // NumRefFrames, so the refresh mask must be the same bit.
    header.refresh_frame_flags =
        1U << (static_cast<unsigned>(i) %
               static_cast<unsigned>(config.num_ref_frames));
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    header.reference_select = options.compound && ref_count > 1;
    // SEF wrapper: code the last picture hidden so the following SEF unit can
    // re-show it (the decoder output count is unchanged).
    header.immediate_output_frame =
        !(options.hide_last_picture && i + 1U == frames.size());

    config.base_q_idx = frame_qp(i);
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu =
        av2enc::video::make_frame_obu(config, header, tile);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.recon.push_back(to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2), width, height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());

    slot_recon[static_cast<std::size_t>(write_slot)] = result.recon.back();
    slot_order[static_cast<std::size_t>(write_slot)] = order_hint;
    recent_slots.erase(
        std::remove(recent_slots.begin(), recent_slots.end(), write_slot),
        recent_slots.end());
    recent_slots.insert(recent_slots.begin(), write_slot);
    if (recent_slots.size() > 7) recent_slots.resize(7);
  }

  // Default path: every coded picture is output in decode order.  A hidden
  // last picture (SEF wrapper) maps to -1; the following SEF unit carries the
  // display entry.
  result.frame_output_index.resize(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const bool hidden =
        options.hide_last_picture && i + 1U == frames.size();
    result.frame_output_index[i] =
        hidden ? -1 : static_cast<std::int32_t>(i);
  }

  if (stats != nullptr) {
    *stats = backend->stats();
    std::uint64_t inter = 0;
    std::uint64_t intra = 0;
    for (std::size_t i = 0; i < result.inter_blocks.size(); ++i) {
      inter += result.inter_blocks[i];
      intra += result.intra_blocks[i];
    }
    stats->inter_blocks = inter;
    stats->intra_blocks = intra;
  }
  return true;
}

const char* video_backend_name() {
  static const char* name = detail::make_backend_name();
  return name;
}

}  // namespace av2enc::cuda::video
