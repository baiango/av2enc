// CUDA M2 P-frame orchestration.
//
// This is a decision-identical mirror of src/encoder/m2/m2_frame.cpp.  The
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
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "encoder/inter/mc/mc.h"
#include "encoder/m2/m2_headers.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"

#include "m2_backend.h"
#include "m2_gpu.h"

namespace av2enc::cuda::m2 {
namespace {

using av2enc::m2::M2Config;
using av2enc::m2::M2FrameHeader;
using av2enc::m2::M2InputFrame;
using av2enc::m2::M2Options;
using av2enc::m2::M2ReconFrame;
using av2enc::m2::M2SequenceResult;
using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionNode;
using partition::PartitionType;

namespace detail = av2enc::cuda::m2::detail;

// The CPU M2 path grows optional tools over time (inter residual, DRL
// reordering).  Detect the option members instead of hard-depending on them,
// so this tree builds against both the baseline and the extended M2Options.
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

int eob_from_levels(const std::vector<std::int32_t>& levels, int w, int h,
                    std::vector<std::uint16_t>* scan_out) {
  const int area = w * h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::m2::syntax::build_scan(w, h, scan);
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

int source_sample(const M2InputFrame& frame, int plane, int x, int y) {
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

std::int64_t sse_against_source(const M2InputFrame& frame, int plane, int x0,
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

struct InterCandidate {
  bool valid = false;
  int mode = av2enc::m2::syntax::kInterGlobalMv;
  int drl_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  std::int64_t sse = 0;
};

class GpuFrameEncoder {
 public:
  GpuFrameEncoder(detail::Backend& backend, const M2InputFrame& source,
                  const M2ReconFrame* reference, const M2Options& options,
                  bool trace, bool keyframe, std::string& error)
      : backend_(backend),
        source_(source),
        options_(options),
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

    if (!backend_.begin_frame(source_, mi_cols_, mi_rows_, reference, error)) {
      begin_ok_ = false;
    }
  }

  ~GpuFrameEncoder() = default;

  bool begin_ok() const { return begin_ok_; }

  bool encode(std::vector<std::uint8_t>& tile_payload,
              std::vector<av2enc::m2::syntax::BlockSyntax>& blocks,
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

    av2enc::m2::syntax::FramePlan plan;
    plan.width = width_;
    plan.height = height_;
    plan.sb_size = 64;
    plan.inter_frame = !keyframe_;
    M2Config config;
    config.max_frame_width = static_cast<std::uint16_t>(width_);
    config.max_frame_height = static_cast<std::uint16_t>(height_);
    config.base_q_idx = options_.base_q_idx;
    config.order_hint_bits = 4;
    config.num_ref_frames = 2;
    config.num_total_refs = 1;
    config.max_drl_bits_minus_1 = 4;
    config.disable_cdf_update = true;
    set_config_drl_reorder(config, opt_drl_reorder(options_));
    config.enable_cfl_intra = true;

    std::size_t tile_bytes = 0;
    if (!av2enc::m2::syntax::assemble_m2_frame(plan, blocks_, levels_, config,
                                               tile_payload, error,
                                               &tile_bytes)) {
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
    jobs.reserve(leaves.size());
    for (const partition::LeafBlock& leaf : leaves) {
      const int w = partition::block_width(leaf.size);
      const int h = partition::block_height(leaf.size);
      jobs.push_back(detail::MeRequest{leaf.mi_col * 4, leaf.mi_row * 4, w, h});
    }
    if (jobs.empty()) return true;

    me::MeCoreOptions opts;
    opts.method = me::kMeDiamond;
    opts.metric = me::kMeSad;
    opts.range = options_.me_range;
    opts.subpel_stages = me::kMeSubpelHalf | me::kMeSubpelQuarter;
    opts.subpel_interp = me::kMeInterpRegular;
    opts.use_predictors = 1;
    opts.use_merge_candidates = 0;

    if (!backend_.me_batch(jobs, opts, me_results_, error)) return false;
    if (me_results_.size() != jobs.size()) {
      error = "backend returned the wrong ME result count";
      return false;
    }
    for (std::size_t i = 0; i < leaves.size(); ++i) {
      me_index_.emplace(
          node_key(leaves[i].mi_row, leaves[i].mi_col, leaves[i].size), i);
    }
    return true;
  }

  bool collect_node(const PartitionNode& node, std::string& error) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    av2enc::m2::syntax::BlockSyntax& entry = blocks_.back();
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
    if (node.forced) entry.flags |= av2enc::m2::syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= av2enc::m2::syntax::kFlagSbRoot;

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

  inter::mc::MvStackParams make_stack_params(int mi_row, int mi_col, int w4,
                                             int h4) const {
    inter::mc::MvStackParams params;
    params.mi_row = mi_row;
    params.mi_col = mi_col;
    params.bw4 = w4;
    params.bh4 = h4;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mv_grid_.data();
    params.ref_frame = inter::mc::kRefLast;
    params.global_mv = inter::mc::Mv{0, 0};
    params.reorder =
        static_cast<inter::mc::DrlReorder>(opt_drl_reorder(options_));
    return params;
  }

  bool load_stack(int mi_row, int mi_col, int w4, int h4,
                  inter::mc::MvStackParams& params,
                  inter::mc::MvStack& stack) const {
    params = make_stack_params(mi_row, mi_col, w4, h4);
    return inter::mc::build_mv_stack(params, &stack) ==
           inter::mc::McStatus::kOk;
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

    av2enc::m2::syntax::BlockSyntax& entry = blocks_[index];

    InterCandidate best;
    const auto consider = [&](const InterCandidate& cand) {
      if (!cand.valid) return;
      if (!best.valid || cand.sse < best.sse) best = cand;
    };

    inter::mc::MvStackParams stack_params;
    inter::mc::MvStack stack;
    bool have_stack = false;
    if (!keyframe_) {
      have_stack = load_stack(r, c, w4, h4, stack_params, stack);
    }

    // Build the inter candidate list in the CPU's consideration order:
    // GLOBALMV, then NEARMV (DRL 0..limit-1), then NEWMV.  One batched
    // MC+SSE call scores all of them.
    std::vector<detail::InterJob> score_jobs;
    std::vector<int> score_kind;  // 0 global, 1 near, 2 new
    std::vector<int> score_drl;
    std::vector<inter::mc::Mv> score_mv;
    std::vector<inter::mc::Mv> score_mvd;

    const auto add_score = [&](int kind, int drl, inter::mc::Mv mv,
                               inter::mc::Mv mvd) {
      detail::InterJob job;
      job.x = x0;
      job.y = y0;
      job.w = w;
      job.h = h;
      job.mv_y = mv.y;
      job.mv_x = mv.x;
      score_jobs.push_back(job);
      score_kind.push_back(kind);
      score_drl.push_back(drl);
      score_mv.push_back(mv);
      score_mvd.push_back(mvd);
    };

    if (!keyframe_) {
      add_score(0, 0, inter::mc::Mv{0, 0}, inter::mc::Mv{0, 0});
    }
    if (!keyframe_ && have_stack && !options_.force_global_mv) {
      if (options_.enable_nearmv) {
        const int near_limit =
            (w > 32 && h > 32) ? stack.count : std::min(stack.count, 2);
        for (int i = 0; i < near_limit; ++i) {
          inter::mc::ModeDecision decision;
          decision.mode = inter::mc::InterMode::kNearmv;
          decision.drl_index = i;
          decision.ref_frame = inter::mc::kRefLast;
          decision.filter = inter::mc::InterpFilter::kEighttap;
          inter::mc::ResolvedMode resolved;
          if (inter::mc::resolve_mode(stack, stack_params, decision,
                                      &resolved) !=
              inter::mc::McStatus::kOk) {
            continue;
          }
          add_score(1, i, resolved.mv, resolved.mvd);
        }
      }

      if (options_.enable_newmv) {
        const auto it = me_index_.find(node_key(r, c, node.size));
        if (it != me_index_.end() && it->second < me_results_.size()) {
          const detail::MeOutcome& me_out = me_results_[it->second];
          const inter::mc::Mv me_mv = inter::mc::mv_from_me_1_16(
              me_out.mv_y, me_out.mv_x);
          inter::mc::ModeDecision decision;
          decision.mode = inter::mc::InterMode::kNewmv;
          decision.drl_index = 0;
          decision.ref_frame = inter::mc::kRefLast;
          decision.filter = inter::mc::InterpFilter::kEighttap;
          decision.me_mv = me_mv;
          decision.me_absolute = true;
          inter::mc::ResolvedMode resolved;
          if (inter::mc::resolve_mode(stack, stack_params, decision,
                                      &resolved) ==
              inter::mc::McStatus::kOk) {
            add_score(2, 0, resolved.mv, resolved.mvd);
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
                        ? av2enc::m2::syntax::kInterGlobalMv
                        : (score_kind[i] == 1
                               ? av2enc::m2::syntax::kInterNearMv
                               : av2enc::m2::syntax::kInterNewMv);
        cand.drl_index = score_drl[i];
        cand.mv = score_mv[i];
        cand.mvd = score_mvd[i];
        cand.sse = sse[i];
        if (trace_) {
          std::cerr << "  cand kind=" << score_kind[i]
                    << " drl=" << score_drl[i] << " mv=(" << cand.mv.y << ","
                    << cand.mv.x << ") sse=" << cand.sse << "\n";
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
        !keyframe_ && opt_enable_inter_residual(options_);
    bool inter_residual = false;
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
      constexpr double kLambda = 0.5 * 66.0 * 66.0;
      const double res_bits =
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

    const bool use_intra =
        keyframe_ || options_.force_intra || !best.valid ||
        (have_intra && intra_sse < inter_luma_sse);

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
      update_mv_grid(r, c, w4, h4, 0, inter::mc::Mv{0, 0});
      ++intra_blocks_;
      if (trace_) {
        std::cerr << "m2 intra r=" << r << " c=" << c << " " << w << "x" << h
                  << " sse=" << intra_sse << "\n";
      }
    } else {
      if (!inter_residual_enabled) {
        if (!backend_.mc_420(best_job, inter_y, inter_u, inter_v, error)) {
          return false;
        }
      }
      entry.is_inter = 1;
      entry.skip_flag = inter_residual ? 0 : 1;
      entry.inter_mode = static_cast<std::uint8_t>(best.mode);
      entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
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
      update_mv_grid(r, c, w4, h4, 1, best.mv);
      ++inter_blocks_;
      if (trace_) {
        std::cerr << "m2 inter r=" << r << " c=" << c << " " << w << "x" << h
                  << " mode=" << static_cast<int>(best.mode)
                  << " drl=" << best.drl_index << " mv=(" << best.mv.y << ","
                  << best.mv.x << ") sse=" << inter_luma_sse
                  << " skip=" << static_cast<int>(entry.skip_flag) << "\n";
      }
    }

    (void)node;
    return true;
  }

  void update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                      inter::mc::Mv mv) {
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
        mi.ref[0] =
            is_inter ? inter::mc::kRefLast : inter::mc::kRefIntra;
        mi.ref[1] = inter::mc::kRefNone;
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
  const M2InputFrame& source_;
  const M2Options& options_;
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
  std::vector<av2enc::m2::syntax::BlockSyntax> blocks_;
  std::vector<std::int32_t> levels_;
  std::vector<detail::MeOutcome> me_results_;
  std::map<std::uint64_t, std::size_t> me_index_;
  std::uint32_t inter_blocks_ = 0;
  std::uint32_t intra_blocks_ = 0;
};

M2ReconFrame to_recon_frame(const Plane& y, const Plane& u, const Plane& v,
                            std::uint32_t width, std::uint32_t height) {
  M2ReconFrame out;
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

M2Config make_config(std::uint32_t width, std::uint32_t height,
                     const M2Options& options) {
  M2Config config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  set_config_drl_reorder(config, opt_drl_reorder(options));
  config.enable_cfl_intra = true;
  return config;
}

}  // namespace

bool encode_m2_sequence_gpu(std::span<const M2InputFrame> frames,
                            const M2Options& options, M2SequenceResult& result,
                            std::string& error, M2GpuStats* stats) {
  error.clear();
  result = M2SequenceResult{};
  if (frames.empty()) {
    error = "no input frames";
    return false;
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
  if (options.base_q_idx != 90) {
    error = "M2 pins base_q_idx = 90 (coefficient CDF qctx slice 0)";
    return false;
  }
  if (opt_drl_reorder(options) < 0 || opt_drl_reorder(options) > 2) {
    error = "M2 DrlReorder must be 0 (disabled), 1 (constraint) or 2 (always)";
    return false;
  }
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const M2InputFrame& f = frames[i];
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

  std::unique_ptr<detail::Backend> backend = detail::make_backend();
  if (!backend) {
    error = "no M2 backend linked";
    return false;
  }
  backend->reset_stats();

  const M2Config config = make_config(width, height, options);

  // Frame 0: intra keyframe (DC intra + DCT), mirroring the CPU reference.
  {
    GpuFrameEncoder key_encoder(*backend, frames[0], nullptr, options,
                                options.trace, /*keyframe=*/true, error);
    if (!key_encoder.begin_ok()) {
      error = "keyframe backend setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> key_tile;
    std::vector<av2enc::m2::syntax::BlockSyntax> key_blocks;
    std::vector<std::int32_t> key_levels;
    if (!key_encoder.encode(key_tile, key_blocks, key_levels, error)) {
      error = "keyframe encode failed: " + error;
      return false;
    }
    if (key_tile.empty()) {
      error = "keyframe produced an empty tile payload";
      return false;
    }

    M2FrameHeader key_header;
    key_header.keyframe = true;
    key_header.order_hint = 0;
    key_header.ref_slot = 0;
    key_header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;

    const std::vector<std::uint8_t> td =
        av2enc::m2::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq =
        av2enc::m2::make_sequence_header_obu(config);
    const std::vector<std::uint8_t> obu =
        av2enc::m2::make_frame_obu(config, key_header, key_tile);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + seq.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq.begin(), seq.end());
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

  for (std::size_t i = 1; i < frames.size(); ++i) {
    const M2ReconFrame& reference = result.recon.back();
    GpuFrameEncoder encoder(*backend, frames[i], &reference, options,
                            options.trace, /*keyframe=*/false, error);
    if (!encoder.begin_ok()) {
      error = "P-frame " + std::to_string(i) + " setup failed: " + error;
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::vector<av2enc::m2::syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }

    M2FrameHeader header;
    header.keyframe = false;
    header.order_hint = static_cast<std::uint32_t>(i);
    header.ref_slot = 0;
    header.refresh_frame_flags = 1;

    const std::vector<std::uint8_t> td =
        av2enc::m2::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu =
        av2enc::m2::make_frame_obu(config, header, tile);
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

const char* m2_backend_name() {
  static const char* name = detail::make_backend_name();
  return name;
}

}  // namespace av2enc::cuda::m2
