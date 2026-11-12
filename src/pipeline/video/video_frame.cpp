#include "pipeline/video/video_frame.h"

#include "codec/intra/modes/intra_modes.h"
#include "pipeline/video/video_frames.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "codec/partition/tree/partition.h"
#include "codec/rate/rc/rc_tables.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/inter/mc/mc.h"
#include "codec/inter/me/me.h"
#include "codec/inter/merge/merge.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::video {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionNode;
using partition::PartitionType;
namespace drl = inter::drl;

// AV2 RefFrame labels used by the merge layer's MI grid.  mc::kRefLast is an
// internal MC label (1); §6.19.7.9 numbers LAST_FRAME = 0 and
// INTRA_FRAME = 8.
constexpr std::int8_t kRefLastAv2 = 0;
constexpr std::int8_t kRefIntraAv2 = 8;

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

struct TxPlan {
  int tx_size = 0;
  int adj_w = 0;
  int adj_h = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;  // adjusted-size coefficient array
  std::vector<std::uint8_t> recon;   // final coded-size samples
};

struct InterCandidate {
  bool valid = false;
  int mode = syntax::kInterGlobalMv;
  int drl_index = 0;
  int ref_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  std::int64_t sse = 0;
};

constexpr int kMaxRefSlots = 8;

// One decoded picture buffer slot.  The planes are MI-aligned (mi_cols*4 by
// mi_rows*4 luma) because MC clips taps to the padded reference extent.
struct RefSlot {
  bool valid = false;
  std::uint32_t order_hint = 0;
  Plane plane[3];
};

int relative_dist(std::uint32_t a, std::uint32_t b, int bits) {
  const int modulus = 1 << bits;
  int diff = static_cast<int>((a - b) & static_cast<std::uint32_t>(modulus - 1));
  if (diff >= (modulus >> 1)) diff -= modulus;
  return diff;
}

// Round a 1/8-pel MV to the signalled frame precision: 1/8, 1/4 or 1/2 pel.
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

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
int clip1(int value) { return clip3(0, 255, value); }

int source_sample(const VideoInputFrame& frame, int plane, int x, int y) {
  const int w = plane == 0 ? static_cast<int>(frame.width)
                           : static_cast<int>((frame.width + 1U) / 2U);
  const int h = plane == 0 ? static_cast<int>(frame.height)
                           : static_cast<int>((frame.height + 1U) / 2U);
  const int cx = clip3(0, w - 1, x);
  const int cy = clip3(0, h - 1, y);
  const std::span<const std::uint8_t> data =
      plane == 0 ? frame.y : (plane == 1 ? frame.u : frame.v);
  const int stride = w;
  return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
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

bool intra_dc_predict(const Plane& recon, const VideoInputFrame& frame, int plane,
                      int x0, int y0, int w, int h,
                      std::vector<std::uint8_t>& out) {
  intra::NeighborRequest request;
  request.recon = intra::PlaneView{recon.samples.data(), recon.stride};
  request.x = x0;
  request.y = y0;
  request.w = w;
  request.h = h;
  request.max_x = recon.width - 1;
  request.max_y = recon.height - 1;
  request.have_left = x0 > 0;
  request.have_above = y0 > 0;
  request.num4_above_right = 0;
  request.num4_below_left = 0;
  request.color_plane = plane;
  intra::PredictParams params;
  params.mode = intra::Mode::kDc;
  params.angle_delta = 0;
  params.enable_edge_filter = false;
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  const intra::IntraStatus status =
      intra::predict_block(request, 8, params, out.data(),
                           static_cast<std::size_t>(w));
  (void)frame;
  return status == intra::IntraStatus::kOk;
}

// §7.14.2 qlookup(base_q_idx) for 8-bit streams.  528 at QP 90, the value the
// transform component bakes into TxParams.
int frame_quant_step(const VideoOptions& options) {
  const int q = options.base_q_idx;
  if (q <= 0 || q > 255) return txv::kQuantStep;
  return rc::kQlookupQ0To255[q];
}

bool compute_tx(const VideoInputFrame& frame, int plane, int x0, int y0, int cw,
                int ch, const std::vector<std::uint8_t>& pred, int quant_step,
                TxPlan& plan, std::string& error) {
  auto params_opt = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "unsupported transform size";
    return false;
  }
  if (quant_step != txv::kQuantStep) {
    params_opt->quant_factor *=
        static_cast<double>(txv::kQuantStep) / static_cast<double>(quant_step);
  }
  const txv::TxParams& params = *params_opt;
  plan.tx_size = params.tx_size;
  plan.adj_w = params.w;
  plan.adj_h = params.h;

  const int coded_pels = cw * ch;
  std::vector<std::int16_t> residual_coded(
      static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const int src = source_sample(frame, plane, x0 + x, y0 + y);
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(cw) +
                         static_cast<std::size_t>(x)];
      residual_coded[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(cw) +
                     static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(src - p);
    }
  }
  std::vector<std::int16_t> residual_adj(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  txv::downsample_residual(residual_coded.data(), params,
                           residual_adj.data());
  std::vector<double> coeff(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  plan.levels.assign(coeff.size(), 0);
  txv::forward_transform(residual_adj.data(), params, coeff.data(),
                         plan.levels.data());

  const int area = params.w * params.h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  syntax::build_scan(params.w, params.h, scan);
  plan.eob = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (plan.levels[scan[static_cast<std::size_t>(c)]] != 0) {
      plan.eob = c + 1;
      break;
    }
  }

  std::vector<std::int32_t> dequant(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  std::vector<std::int16_t> recon_residual(
      static_cast<std::size_t>(coded_pels));
  if (quant_step == txv::kQuantStep) {
    txv::inverse_transform(plan.levels.data(), params, dequant.data(),
                           recon_residual.data());
  } else {
    // §7.14.4 dequant with the frame's qlookup step, then the shared inverse.
    const int n = params.w * params.h;
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc = plan.levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant_step) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          params.dq_denom;
      const std::int32_t value = qc < 0 ? -dq : dq;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, value));
    }
    txv::inverse_block(dequant.data(), params, recon_residual.data());
  }
  plan.recon.resize(static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(cw) +
                         static_cast<std::size_t>(x)];
      const int r = recon_residual[static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(cw) +
                                   static_cast<std::size_t>(x)];
      plan.recon[static_cast<std::size_t>(y) *
                     static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(clip1(p + r));
    }
  }
  return true;
}

// Rough coefficient rate estimate used only for the skip/residual decision.
// The actual cost is measured by the entropy coder; this heuristic is not
// normative and never affects bitstream correctness.
double estimate_tx_bits(const TxPlan& plan, bool is_inter_luma) {
  if (plan.eob == 0) return 1.0;
  double bits = 8.0;  // EOB symbol + extra bits
  if (is_inter_luma) bits += 4.0;  // inter TX_TYPE syntax
  int nonzero = 0;
  for (int c = 0; c < plan.eob; ++c) {
    const int level = plan.levels[static_cast<std::size_t>(c)];
    if (level != 0) ++nonzero;
  }
  bits += static_cast<double>(nonzero) * 2.5 + plan.eob * 0.5;
  return bits;
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

class InterFrameEncoder {
 public:
  InterFrameEncoder(const VideoInputFrame& source,
                    std::vector<inter::mc::RefFrame> references,
                    std::array<int, 7> frame_distance, int num_total_refs,
                    const VideoOptions& options, bool trace, bool keyframe)
      : source_(source),
        references_(std::move(references)),
        frame_distance_(frame_distance),
        num_total_refs_(num_total_refs),
        options_(options),
        trace_(trace),
        keyframe_(keyframe) {
    width_ = static_cast<int>(source.width);
    height_ = static_cast<int>(source.height);
    chroma_width_ = width_ / 2;
    chroma_height_ = height_ / 2;

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

    quant_step_ = frame_quant_step(options_);
    mv_grid_.assign(static_cast<std::size_t>(mi_rows_) *
                        static_cast<std::size_t>(mi_cols_),
                    inter::mc::MiInfo{});
  }

  std::int64_t block_variance_cost_at(
      const partition::PartitionNodeInfo& info) const {
    const partition::BlockRect rect = partition::block_rect(
        geometry_, info.mi_row, info.mi_col, info.size);
    const int w = rect.visible_width;
    const int h = rect.visible_height;
    if (w <= 0 || h <= 0) return 0;
    std::int64_t sum = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        sum += source_sample(source_, 0, rect.x + x, rect.y + y);
      }
    }
    const double mean = static_cast<double>(sum) /
                        static_cast<double>(w * h);
    double sse = 0.0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const double d =
            static_cast<double>(source_sample(source_, 0, rect.x + x,
                                              rect.y + y)) -
            mean;
        sse += d * d;
      }
    }
    return static_cast<std::int64_t>(sse);
  }

  bool encode(std::vector<std::uint8_t>& tile_payload,
              std::vector<syntax::BlockSyntax>& blocks,
              std::vector<std::int32_t>& levels, std::string& error) {
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = options_.min_block_size;
    poptions.max_depth = options_.max_partition_depth;
    poptions.allow_rect = true;
    poptions.allow_split = false;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }

    const double lambda =
        0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;
    const auto cost_fn = [this, signal_cost](
                             const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = static_cast<double>(block_variance_cost_at(info));
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };

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

    blocks_.clear();
    levels_.clear();
    for (const PartitionNode& sb : frame.superblocks) {
      if (!collect_node(sb, error)) return false;
    }

    syntax::FramePlan plan;
    plan.width = width_;
    plan.height = height_;
    plan.sb_size = 64;
    plan.inter_frame = !keyframe_;
    plan.num_total_refs = num_total_refs_;
    plan.mv_precision = options_.mv_precision;
    plan.drl_reorder = options_.drl_reorder
                           ? (options_.drl_reorder_always ? 2 : 1)
                           : 0;
    plan.frame_distance = frame_distance_;
    VideoConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(width_);
    config.max_frame_height = static_cast<std::uint16_t>(height_);
    config.base_q_idx = options_.base_q_idx;
    config.order_hint_bits = 4;
    config.num_ref_frames = 2;
    config.num_total_refs = 1;
    config.max_drl_bits_minus_1 = 4;
    config.disable_cdf_update = true;
    config.enable_cfl_intra = true;

    std::size_t tile_bytes = 0;
    if (!syntax::assemble_video_frame(plan, blocks_, levels_, config,
                                   tile_payload, error, &tile_bytes)) {
      return false;
    }
    blocks = std::move(blocks_);
    levels = std::move(levels_);
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  int recon_mi_cols() const { return mi_cols_; }
  int recon_mi_rows() const { return mi_rows_; }
  std::uint32_t inter_blocks() const { return inter_blocks_; }
  std::uint32_t intra_blocks() const { return intra_blocks_; }

 private:
  bool collect_node(const PartitionNode& node, std::string& error) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
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
    // The merge layer uses AV2 RefFrame labels: 0..6 are reference indices,
    // INTRA = 8.
    params.ref_frame[0] = static_cast<std::int8_t>(ref_index);
    params.ref_frame[1] = merge::kRefNone;
    params.is_compound = false;
    params.global_mv[0] = inter::mc::Mv{0, 0};
    params.global_mv[1] = inter::mc::Mv{0, 0};
    // DRL reordering is not gated (see src/pipeline/video/README.md).
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
    const drl::Mode mode =
        options_.drl_reorder
            ? (options_.drl_reorder_always ? drl::Mode::kAlways
                                           : drl::Mode::kConstraint)
            : drl::Mode::kDisabled;
    if (mode == drl::Mode::kDisabled) {
      return merge::build_stack_checked(params, &stack) ==
             merge::Status::kOk;
    }
    return drl::build_reordered_stack(params, mode, /*use_temporal_first=*/false,
                                      &stack) == merge::Status::kOk;
  }

  bool predict_luma(int ref_index, inter::mc::Mv mv, int x0, int y0, int w,
                    int h, std::vector<std::uint8_t>& out) const {
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    const inter::mc::BlockRect rect{x0, y0, w, h};
    return inter::mc::predict_plane(references_[ref_index], 0, rect, mv,
                                    inter::mc::InterpFilter::kEighttap,
                                    out.data(), w) ==
           inter::mc::McStatus::kOk;
  }

  bool predict_all(int ref_index, inter::mc::Mv mv, int x0, int y0, int w,
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
               references_[ref_index], rect, mv,
               inter::mc::InterpFilter::kEighttap, dst) ==
           inter::mc::McStatus::kOk;
  }

  bool run_me(int ref_index, int x0, int y0, int w, int h,
              inter::mc::Mv& mv, std::string& error) {
    me::MeCoreFrame cur;
    cur.width = width_;
    cur.height = height_;
    cur.plane[0] = me::MeCorePlane{source_.y.data(), width_, width_, height_};
    cur.plane[1] = me::MeCorePlane{source_.u.data(), chroma_width_,
                                   chroma_width_, chroma_height_};
    cur.plane[2] = me::MeCorePlane{source_.v.data(), chroma_width_,
                                   chroma_width_, chroma_height_};
    me::MeCoreRef ref;
    ref.plane[0] =
        me::MeCorePlane{references_[ref_index].plane[0].data,
                        references_[ref_index].plane[0].stride, mi_cols_ * 4,
                        mi_rows_ * 4};
    ref.plane[1] =
        me::MeCorePlane{references_[ref_index].plane[1].data,
                        references_[ref_index].plane[1].stride, mi_cols_ * 2,
                        mi_rows_ * 2};
    ref.plane[2] =
        me::MeCorePlane{references_[ref_index].plane[2].data,
                        references_[ref_index].plane[2].stride, mi_cols_ * 2,
                        mi_rows_ * 2};

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

    syntax::BlockSyntax& entry = blocks_[index];

    InterCandidate best;
    const auto consider = [&](InterCandidate cand) {
      if (!cand.valid) return;
      if (!best.valid || cand.sse < best.sse) best = cand;
    };

    if (!keyframe_) {
      for (int ref_index = 0; ref_index < num_total_refs_; ++ref_index) {
        merge::StackParams stack_params;
        merge::MvStack stack;
        const bool have_stack =
            load_stack(r, c, w4, h4, ref_index, stack_params, stack);

        // GLOBALMV: zero MV, no stack dependency.
        {
          std::vector<std::uint8_t> luma;
          if (predict_luma(ref_index, inter::mc::Mv{0, 0}, x0, y0, w, h,
                           luma)) {
            InterCandidate cand;
            cand.valid = true;
            cand.mode = syntax::kInterGlobalMv;
            cand.ref_index = ref_index;
            cand.mv = inter::mc::Mv{0, 0};
            cand.sse = sse_against_source(source_, 0, x0, y0, w, h, luma);
            if (trace_) {
              std::cerr << "  cand ref=" << ref_index
                        << " global mv=(0,0) sse=" << cand.sse << "\n";
            }
            consider(cand);
          }
        }

        if (have_stack && !options_.force_global_mv) {
          if (options_.enable_nearmv) {
            // §7.12.2.20 inserts MVP-combination candidates only when both
            // dimensions exceed 32.  merge::build_stack_checked already applies
            // that gate.
            const int near_limit =
                (w > 32 && h > 32) ? stack.count : std::min(stack.count, 2);
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
              if (!predict_luma(ref_index, near_mv, x0, y0, w, h, luma)) {
                continue;
              }
              InterCandidate cand;
              cand.valid = true;
              cand.mode = syntax::kInterNearMv;
              cand.ref_index = ref_index;
              cand.drl_index = i;
              cand.mv = near_mv;
              cand.sse = sse_against_source(source_, 0, x0, y0, w, h, luma);
              if (trace_) {
                std::cerr << "  cand ref=" << ref_index << " near drl=" << i
                          << " mv=(" << near_mv.y << "," << near_mv.x
                          << ") sse=" << cand.sse << "\n";
              }
              consider(cand);
            }
          }

          if (options_.enable_newmv) {
            inter::mc::Mv me_mv{};
            if (run_me(ref_index, x0, y0, w, h, me_mv, error)) {
              me_mv = quantize_mv(me_mv, options_.mv_precision);
              const inter::mc::Mv pred = stack.cand[0].mv[0];
              // §5.20.7.13 only lowers the predictor when MvPrecision is
              // coarser than 1/2 pel.  Our quarter/half frames keep the
              // predictor as-is; the MVD itself is coded in precision units.
              const inter::mc::Mv delta{me_mv.y - pred.y, me_mv.x - pred.x};
              const inter::mc::Mv mvd =
                  options_.mv_precision == 0
                      ? delta
                      : quantize_mv(delta, options_.mv_precision);
              merge::ResolvedMode resolved;
              const bool resolved_ok =
                  options_.mv_precision == 0
                      ? merge::resolve_single_mode(
                            stack, stack_params, merge::SingleMode::kNewMv, 0,
                            mvd, &resolved) == merge::Status::kOk
                      : true;
              if (resolved_ok) {
                std::vector<std::uint8_t> luma;
                const inter::mc::Mv new_mv =
                    options_.mv_precision == 0
                        ? resolved.mv[0]
                        : inter::mc::Mv{pred.y + mvd.y, pred.x + mvd.x};
                if (predict_luma(ref_index, new_mv, x0, y0, w, h, luma)) {
                  InterCandidate cand;
                  cand.valid = true;
                  cand.mode = syntax::kInterNewMv;
                  cand.ref_index = ref_index;
                  cand.drl_index = 0;
                  cand.mv = new_mv;
                  cand.mvd = options_.mv_precision == 0 ? resolved.mvd[0] : mvd;
                  cand.sse =
                      sse_against_source(source_, 0, x0, y0, w, h, luma);
                  if (trace_) {
                    std::cerr << "  cand ref=" << ref_index << " new me=("
                              << me_mv.y << "," << me_mv.x << ") res=("
                              << new_mv.y << "," << new_mv.x << ") mvd=("
                              << cand.mvd.y << "," << cand.mvd.x
                              << ") sse=" << cand.sse << "\n";
                  }
                  consider(cand);
                }
              }
            }
          }
        }
      }
    }

    // Intra fallback: DC prediction + DCT residual, the M1 transform path.
    bool have_intra = false;
    std::int64_t intra_sse = 0;
    std::vector<std::uint8_t> intra_y_pred;
    std::vector<std::uint8_t> intra_u_pred;
    std::vector<std::uint8_t> intra_v_pred;
    TxPlan intra_y;
    TxPlan intra_u;
    TxPlan intra_v;
    if (keyframe_ || options_.enable_intra_fallback || options_.force_intra) {
      if (!intra_dc_predict(recon_[0], source_, 0, x0, y0, w, h,
                            intra_y_pred) ||
          !intra_dc_predict(recon_[1], source_, 1, cx0, cy0, cw, ch,
                            intra_u_pred) ||
          !intra_dc_predict(recon_[2], source_, 2, cx0, cy0, cw, ch,
                            intra_v_pred)) {
        error = "intra DC prediction failed";
        return false;
      }
      if (!compute_tx(source_, 0, x0, y0, w, h, intra_y_pred, quant_step_,
                      intra_y, error) ||
          !compute_tx(source_, 1, cx0, cy0, cw, ch, intra_u_pred, quant_step_,
                      intra_u, error) ||
          !compute_tx(source_, 2, cx0, cy0, cw, ch, intra_v_pred, quant_step_,
                      intra_v, error)) {
        return false;
      }
      intra_sse = sse_against_source(source_, 0, x0, y0, w, h, intra_y.recon);
      if (trace_) {
        std::cerr << "  cand intra sse=" << intra_sse << "\n";
      }
      have_intra = true;
    }

    // Inter residual (feature 1).  The predictor comes from the winner of the
    // motion search; when enabled, a quantised residual is coded for Y/U/V and
    // the reconstructed block is pred + dequantised residual.  The decision is
    // an SSE + lambda * estimated-bits comparison against skip_flag = 1.
    std::vector<std::uint8_t> inter_y;
    std::vector<std::uint8_t> inter_u;
    std::vector<std::uint8_t> inter_v;
    bool inter_residual = false;
    TxPlan res_y;
    TxPlan res_u;
    TxPlan res_v;
    std::int64_t inter_luma_sse = best.sse;
    if (!keyframe_ && best.valid) {
      if (!predict_all(best.ref_index, best.mv, x0, y0, w, h, inter_y,
                       inter_u, inter_v)) {
        error = "inter prediction failed";
        return false;
      }
      inter_luma_sse =
          sse_against_source(source_, 0, x0, y0, w, h, inter_y);
      if (options_.enable_inter_residual) {
        const std::int64_t pred_total =
            inter_luma_sse +
            sse_against_source(source_, 1, cx0, cy0, cw, ch, inter_u) +
            sse_against_source(source_, 2, cx0, cy0, cw, ch, inter_v);
        if (!compute_tx(source_, 0, x0, y0, w, h, inter_y, quant_step_, res_y,
                        error) ||
            !compute_tx(source_, 1, cx0, cy0, cw, ch, inter_u, quant_step_,
                        res_u, error) ||
            !compute_tx(source_, 2, cx0, cy0, cw, ch, inter_v, quant_step_,
                        res_v, error)) {
          return false;
        }
        const std::int64_t res_total =
            sse_against_source(source_, 0, x0, y0, w, h, res_y.recon) +
            sse_against_source(source_, 1, cx0, cy0, cw, ch, res_u.recon) +
            sse_against_source(source_, 2, cx0, cy0, cw, ch, res_v.recon);
        const bool all_zero =
            res_y.eob == 0 && res_u.eob == 0 && res_v.eob == 0;
        // Reconstruction step is qlookup/8 (66 at QP 90); sse_scale 0.5 is
        // the M1 calibrated value.
        const double sample_step = static_cast<double>(quant_step_) / 8.0;
        const double kLambda = 0.5 * sample_step * sample_step;
        const double res_bits = 3.0 + estimate_tx_bits(res_y, true) +
                                estimate_tx_bits(res_u, false) +
                                estimate_tx_bits(res_v, false);
        const bool want =
            options_.force_inter_residual ||
            (!all_zero &&
             static_cast<double>(res_total) + kLambda * res_bits <
                 static_cast<double>(pred_total) + kLambda * 2.0);
        if (want) {
          inter_residual = true;
          inter_y = res_y.recon;
          inter_u = res_u.recon;
          inter_v = res_v.recon;
          inter_luma_sse =
              sse_against_source(source_, 0, x0, y0, w, h, inter_y);
        }
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
      entry.y_mode = 0;   // DC_PRED
      entry.y_angle_delta = 0;
      entry.uv_mode = 0;  // DC_PRED
      write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_u.recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_v.recon);
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      const TxPlan* plans[3] = {&intra_y, &intra_u, &intra_v};
      for (int plane = 0; plane < 3; ++plane) {
        entry.tx_size[plane] = static_cast<std::uint8_t>(plans[plane]->tx_size);
        entry.eob[plane] = static_cast<std::uint32_t>(plans[plane]->eob);
        const int area = plans[plane]->adj_w * plans[plane]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane]->adj_w, plans[plane]->adj_h, scan);
        for (int c2 = 0; c2 < plans[plane]->eob; ++c2) {
          levels_.push_back(plans[plane]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
      update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
      ++intra_blocks_;
      if (trace_) {
        std::cerr << "m2 intra r=" << r << " c=" << c << " " << w << "x" << h
                  << " sse=" << intra_sse << "\n";
      }
    } else {
      entry.is_inter = 1;
      entry.skip_flag = inter_residual ? 0 : 1;
      entry.inter_mode = static_cast<std::uint8_t>(best.mode);
      entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
      entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
      entry.mv_y = best.mv.y;
      entry.mv_x = best.mv.x;
      entry.mvd_y = best.mvd.y;
      entry.mvd_x = best.mvd.x;
      write_plane_block(recon_[0], x0, y0, w, h, inter_y);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, inter_u);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, inter_v);
      if (inter_residual) {
        entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
        const TxPlan* plans[3] = {&res_y, &res_u, &res_v};
        for (int plane = 0; plane < 3; ++plane) {
          entry.tx_size[plane] =
              static_cast<std::uint8_t>(plans[plane]->tx_size);
          entry.eob[plane] = static_cast<std::uint32_t>(plans[plane]->eob);
          const int area = plans[plane]->adj_w * plans[plane]->adj_h;
          std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
          syntax::build_scan(plans[plane]->adj_w, plans[plane]->adj_h, scan);
          for (int c2 = 0; c2 < plans[plane]->eob; ++c2) {
            levels_.push_back(plans[plane]->levels[scan[
                static_cast<std::size_t>(c2)]]);
          }
        }
      }
      update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv);
      ++inter_blocks_;
      if (trace_) {
        std::cerr << "m2 inter r=" << r << " c=" << c << " " << w << "x" << h
                  << " mode=" << static_cast<int>(best.mode)
                  << " drl=" << best.drl_index
                  << " ref=" << best.ref_index << " mv=(" << best.mv.y << ","
                  << best.mv.x << ") sse=" << inter_luma_sse
                  << " skip=" << static_cast<int>(entry.skip_flag) << "\n";
      }
    }

    (void)node;
    return true;
  }

  void update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                      int ref_index, inter::mc::Mv mv) {
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
        mi.ref[1] = merge::kRefNone;
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

  const VideoInputFrame& source_;
  std::vector<inter::mc::RefFrame> references_;
  std::array<int, 7> frame_distance_{};
  int num_total_refs_ = 1;
  const VideoOptions& options_;
  bool trace_ = false;
  bool keyframe_ = false;
  int quant_step_ = txv::kQuantStep;  // §7.14.2 qlookup(base_q_idx)
  int width_ = 0;
  int height_ = 0;
  int chroma_width_ = 0;
  int chroma_height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  FrameGeometry geometry_;
  Plane recon_[3];
  inter::mc::RefFrame ref_frame_;
  std::vector<inter::mc::MiInfo> mv_grid_;
  std::vector<syntax::BlockSyntax> blocks_;
  std::vector<std::int32_t> levels_;
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
    std::memcpy(out.y.data() + static_cast<std::size_t>(row) * width,
                y.samples.data() + static_cast<std::size_t>(row) * y.stride,
                width);
  }
  for (std::uint32_t row = 0; row < ch; ++row) {
    std::memcpy(out.u.data() + static_cast<std::size_t>(row) * cw,
                u.samples.data() + static_cast<std::size_t>(row) * u.stride,
                cw);
    std::memcpy(out.v.data() + static_cast<std::size_t>(row) * cw,
                v.samples.data() + static_cast<std::size_t>(row) * v.stride,
                cw);
  }
  return out;
}

}  // namespace

bool encode_video_sequence(std::span<const VideoInputFrame> frames,
                        const VideoOptions& options, VideoSequenceResult& result,
                        std::string& error) {
  error.clear();
  result = VideoSequenceResult{};
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
  // The syntax writer selects the §8.3.1 qctx slice from base_q_idx and the
  // quantiser below uses qlookup(base_q_idx).  0 (lossless) is not implemented.
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
  // Per-frame quantizer selection: an empty vector means the historical
  // fixed-QP stream, so the default path does not construct any extra state.
  const auto frame_qp = [&options](std::size_t index) -> std::uint8_t {
    return options.frame_base_q_idx.empty() ? options.base_q_idx
                                            : options.frame_base_q_idx[index];
  };
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
  if (options.num_total_refs < 1 || options.num_total_refs > 3) {
    error = "num_total_refs must be in [1, 3]";
    return false;
  }
  if (options.mv_precision < 0 || options.mv_precision > 2) {
    error = "mv_precision must be 0 (1/8), 1 (1/4) or 2 (1/2) pel";
    return false;
  }
  if (options.drl_reorder) {
    error = "DRL reordering is not decoder-gated yet "
            "(docs/negative-results.md)";
    return false;
  }
  const int num_total_refs = options.num_total_refs;
  const int num_ref_frames = num_total_refs > 1 ? kMaxRefSlots : 2;
  const int order_hint_bits = 4;

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
  const VideoOptions key_options = options_for_frame(0);
  InterFrameEncoder key_encoder(frames[0], no_refs, no_distance, 1, key_options,
                                options.trace, /*keyframe=*/true);
  std::vector<std::uint8_t> key_tile;
  std::vector<syntax::BlockSyntax> key_blocks;
  std::vector<std::int32_t> key_levels;
  if (!key_encoder.encode(key_tile, key_blocks, key_levels, error)) {
    error = "keyframe encode failed: " + error;
    return false;
  }
  if (key_tile.empty()) {
    error = "keyframe produced an empty tile payload";
    return false;
  }

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = order_hint_bits;
  config.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.disable_drl_reorder = !options.drl_reorder;
  config.constrain_drl_reorder = !options.drl_reorder_always;
  config.enable_cfl_intra = true;

  VideoFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.ref_slot = 0;
  key_header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;

  std::vector<std::uint8_t> key_unit;
  {
    config.base_q_idx = frame_qp(0);
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
    const std::vector<std::uint8_t> obu =
        make_frame_obu(config, key_header, key_tile);
    key_unit.reserve(td.size() + seq.size() + obu.size());
    key_unit.insert(key_unit.end(), td.begin(), td.end());
    key_unit.insert(key_unit.end(), seq.begin(), seq.end());
    key_unit.insert(key_unit.end(), obu.begin(), obu.end());
  }
  result.units.push_back(std::move(key_unit));
  result.recon.push_back(to_recon_frame(
      key_encoder.recon(0), key_encoder.recon(1), key_encoder.recon(2), width,
      height));
  result.frame_bytes.push_back(
      static_cast<std::uint32_t>(result.units.back().size()));
  result.tile_bytes.push_back(static_cast<std::uint32_t>(key_tile.size()));
  result.inter_blocks.push_back(0);
  result.intra_blocks.push_back(key_encoder.intra_blocks());

  // Decoded picture buffer.  Key §7.23 keeps only the first refreshed slot
  // valid, so the first P frame has a single reference.  Later frames reference
  // the most recently decoded slots and refresh a rotating slot.
  std::array<RefSlot, kMaxRefSlots> slots;
  std::vector<int> recent_slots;
  slots[0].valid = true;
  slots[0].order_hint = 0;
  for (int plane = 0; plane < 3; ++plane) {
    slots[0].plane[plane] = key_encoder.recon(plane);
  }
  recent_slots.push_back(0);

  const auto make_reference = [&](int slot) {
    inter::mc::RefFrame ref;
    ref.plane[0] = inter::mc::PlaneView{slots[slot].plane[0].samples.data(),
                                        slots[slot].plane[0].stride};
    ref.plane[1] = inter::mc::PlaneView{slots[slot].plane[1].samples.data(),
                                        slots[slot].plane[1].stride};
    ref.plane[2] = inter::mc::PlaneView{slots[slot].plane[2].samples.data(),
                                        slots[slot].plane[2].stride};
    ref.mi_cols = key_encoder.recon_mi_cols();
    ref.mi_rows = key_encoder.recon_mi_rows();
    return ref;
  };
  (void)make_reference;

  for (std::size_t i = 1; i < frames.size(); ++i) {
    const std::uint32_t order_hint =
        static_cast<std::uint32_t>(i) &
        static_cast<std::uint32_t>((1U << order_hint_bits) - 1U);
    const int ref_count = std::min<int>(
        num_total_refs, static_cast<int>(recent_slots.size()));
    std::vector<inter::mc::RefFrame> references;
    references.reserve(static_cast<std::size_t>(ref_count));
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < ref_count; ++r) {
      const int slot = recent_slots[static_cast<std::size_t>(r)];
      references.push_back(make_reference(slot));
      frame_distance[static_cast<std::size_t>(r)] =
          relative_dist(order_hint, slots[slot].order_hint, order_hint_bits);
    }

    const VideoOptions frame_options = options_for_frame(i);
    InterFrameEncoder encoder(frames[i], std::move(references), frame_distance,
                              ref_count, frame_options, options.trace,
                              /*keyframe=*/false);
    std::vector<std::uint8_t> tile;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }

    const int write_slot = static_cast<int>(i % num_ref_frames);
    VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = order_hint;
    header.num_total_refs = static_cast<std::uint8_t>(ref_count);
    for (int r = 0; r < ref_count; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(recent_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    header.refresh_frame_flags = 1U << write_slot;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    // SEF wrapper: code the last picture hidden so a following SEF unit can
    // re-show it; the decoder output count is unchanged.
    header.immediate_output_frame =
        !(options.hide_last_picture && i + 1U == frames.size());

    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    config.base_q_idx = frame_qp(i);
    const std::vector<std::uint8_t> obu = make_frame_obu(config, header, tile);
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

    // Store the reconstruction and refresh the rotating slot.
    slots[write_slot].valid = true;
    slots[write_slot].order_hint = order_hint;
    for (int plane = 0; plane < 3; ++plane) {
      slots[write_slot].plane[plane] = encoder.recon(plane);
    }
    recent_slots.erase(
        std::remove(recent_slots.begin(), recent_slots.end(), write_slot),
        recent_slots.end());
    recent_slots.insert(recent_slots.begin(), write_slot);
    if (recent_slots.size() > static_cast<std::size_t>(kMaxRefSlots - 1)) {
      recent_slots.resize(static_cast<std::size_t>(kMaxRefSlots - 1));
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
