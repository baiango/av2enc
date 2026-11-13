#include "pipeline/video/video_frame.h"

#include "codec/intra/modes/intra_modes.h"
#include "frame/metadata/content_interpretation/content_interpretation.h"
#include "frame/mfh/mfh.h"
#include "pipeline/video/video_frames.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_intra_search.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/video_tx_search.h"
#include "codec/partition/tree/partition.h"
#include "codec/rate/rc/rc_tables.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/inter/mc/mc.h"
#include "codec/inter/me/me.h"
#include "codec/inter/merge/merge.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"
#if defined(AV2ENC_VIDEO_FRAMES)
#include "codec/inter/multiref/compound_predict.h"
#endif
#if defined(AV2ENC_VIDEO_FRAMES)
#include "codec/inter/bframes/bframes.h"
#include "codec/inter/tlayers/tlayers.h"
#endif
#if defined(AV2ENC_VIDEO_GME)
#include "codec/inter/gme/gme.h"
#include "pipeline/video/video_gme.h"
#include "pipeline/video/video_warp.h"
#endif
#include "pipeline/video/post/tiled_frame.h"
#include "pipeline/video/post/deblock_grid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
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

  // Opt-in luma TX search (--video-tx-search).  When `searched` is true the
  // per-part list below is authoritative: `parts[i].levels` are in §5.20.7.30
  // scan order and `rate_bits` is the rate the device module also reports.
  bool searched = false;
  int partition = 0;
  double rate_bits = 0.0;
  struct Part {
    int x = 0;
    int y = 0;
    int coded_w = 0;
    int coded_h = 0;
    int tx_size = 0;
    int tx_type = 0;
    int mi_row = 0;
    int mi_col = 0;
    int eob = 0;
    std::vector<std::int32_t> levels;  // scan order
  };
  std::vector<Part> parts;
};

struct InterCandidate {
  bool valid = false;
  bool compound = false;
  // GLOBALMV with an accepted non-identity model and the warped predictor.
  bool use_warp = false;
  int mode = syntax::kInterGlobalMv;
  int drl_index = 0;
  int ref_index = 0;
  inter::mc::Mv mv{};
  inter::mc::Mv mvd{};
  std::int64_t sse = 0;
};

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
                      int x0, int y0, int w, int h, int max_x, int max_y,
                      bool have_left, bool have_above,
                      std::vector<std::uint8_t>& out) {
  intra::NeighborRequest request;
  request.recon = intra::PlaneView{recon.samples.data(), recon.stride};
  request.x = x0;
  request.y = y0;
  request.w = w;
  request.h = h;
  request.max_x = max_x;
  request.max_y = max_y;
  request.have_left = have_left;
  request.have_above = have_above;
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

// estimate_tx_bits() for a searched plan: the same heuristic applied per
// transform leaf (the plan's scan-order levels are already the exact nonzero
// set, so counting the first `eob` entries is the nonzero count).
double search_plan_estimate_bits(const TxPlan& plan, bool is_inter_luma) {
  double bits = 0.0;
  for (const TxPlan::Part& part : plan.parts) {
    if (part.eob == 0) {
      bits += 1.0;
      continue;
    }
    int nonzero = 0;
    for (int c = 0; c < part.eob; ++c) {
      if (part.levels[static_cast<std::size_t>(c)] != 0) ++nonzero;
    }
    bits += 8.0 + (is_inter_luma ? 4.0 : 0.0) +
            static_cast<double>(nonzero) * 2.5 + part.eob * 0.5;
  }
  return bits;
}

// Rate used by the skip/residual and intra/inter decisions.  The TX search's
// own cost model only selects the plan; the decision keeps the historical
// estimate_tx_bits heuristic (per leaf) so enabling the search does not
// re-price every residual against skip.
double luma_plan_bits(const TxPlan& plan, bool is_inter_luma) {
  return plan.searched ? search_plan_estimate_bits(plan, is_inter_luma)
                       : estimate_tx_bits(plan, is_inter_luma);
}

bool luma_plan_all_zero(const TxPlan& plan) {
  if (!plan.searched) return plan.eob == 0;
  for (const TxPlan::Part& part : plan.parts) {
    if (part.eob != 0) return false;
  }
  return true;
}

// Copies one leaf's luma coefficient layout into the syntax node and appends
// the scan-order levels.  The no-search path is the historical single DCT; the
// search path fills the TX_MODE_SELECT per-part list.
void write_luma_plan(syntax::BlockSyntax& entry, const TxPlan& plan,
                     std::vector<std::int32_t>& levels) {
  entry.tx_type[0] = 0;
  if (!plan.searched) {
    entry.tx_size[0] = static_cast<std::uint8_t>(plan.tx_size);
    entry.eob[0] = static_cast<std::uint32_t>(plan.eob);
    entry.tx_partition = 0;
    entry.tx_part_count = 0;
    const int area = plan.adj_w * plan.adj_h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(plan.adj_w, plan.adj_h, scan);
    for (int c = 0; c < plan.eob; ++c) {
      levels.push_back(plan.levels[scan[static_cast<std::size_t>(c)]]);
    }
    return;
  }
  entry.tx_partition = static_cast<std::uint8_t>(plan.partition);
  entry.tx_part_count = static_cast<std::uint8_t>(plan.parts.size());
  const int count = static_cast<int>(plan.parts.size());
  for (int i = 0; i < count; ++i) {
    const TxPlan::Part& part = plan.parts[static_cast<std::size_t>(i)];
    entry.tx_part_tx_size[i] = static_cast<std::uint8_t>(part.tx_size);
    entry.tx_part_tx_type[i] = static_cast<std::uint8_t>(part.tx_type);
    entry.tx_part_mi_row[i] = static_cast<std::int16_t>(part.mi_row);
    entry.tx_part_mi_col[i] = static_cast<std::int16_t>(part.mi_col);
    entry.tx_part_eob[i] = static_cast<std::uint32_t>(part.eob);
    entry.tx_part_levels_offset[i] =
        static_cast<std::uint32_t>(levels.size());
    for (int c = 0; c < part.eob; ++c) {
      levels.push_back(part.levels[static_cast<std::size_t>(c)]);
    }
  }
  entry.tx_size[0] = entry.tx_part_tx_size[0];
  entry.tx_type[0] = entry.tx_part_tx_type[0];
  entry.eob[0] = entry.tx_part_eob[0];
}

// Opt-in luma TX size/type/partition search.  Builds the coded-size residual
// from the source and prediction and runs the shared CPU reference
// (`pipeline/video/video_tx_search.h`), which mirrors the device module under
// `src/cuda/pipeline/video/tx/`.
bool search_luma_tx(const VideoInputFrame& frame, int x0, int y0, int cw,
                    int ch, const std::vector<std::uint8_t>& pred, int qindex,
                    txtsearch::Mode mode, double lambda, TxPlan& plan,
                    std::string& error) {
  std::vector<std::int16_t> residual(
      static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const int src = source_sample(frame, 0, x0 + x, y0 + y);
      const int p = pred[static_cast<std::size_t>(y) * cw + x];
      residual[static_cast<std::size_t>(y) * cw + x] =
          static_cast<std::int16_t>(src - p);
    }
  }
  txtsearch::Options options;
  options.lambda = lambda;
  // §5.20.7.24 predicts per *transform block*, in decode order, and each part's
  // reconstruction feeds the next part's prediction.  A fixed block prediction
  // is only equivalent when the transform block covers the coding block, so the
  // partition search is inter-only: MC with one MV is position-independent, and
  // the decoder's per-part prediction equals the block prediction at the same
  // coordinates.  Intra keeps TX_PARTITION_NONE and searches the TX type only;
  // the full intra partition search needs the per-part prediction/recon
  // feedback the device module's fixed-predictor contract does not carry.
  options.search_partition = mode == txtsearch::Mode::kInter;
  txtsearch::Result result;
  if (!txtsearch::search(residual.data(), cw, ch, mode, qindex, 0, 0,
                         pred.data(), options, &result, &error)) {
    return false;
  }
  plan.searched = true;
  plan.partition = result.partition;
  plan.rate_bits = result.rate_bits;
  plan.recon = std::move(result.recon);
  plan.parts.clear();
  plan.parts.reserve(static_cast<std::size_t>(result.count));
  for (int i = 0; i < result.count; ++i) {
    const txtsearch::Leaf& leaf = result.leaves[i];
    TxPlan::Part part;
    part.x = leaf.x;
    part.y = leaf.y;
    part.coded_w = leaf.coded_w;
    part.coded_h = leaf.coded_h;
    part.tx_size = leaf.tx_size;
    part.tx_type = leaf.tx_type;
    part.mi_row = leaf.mi_row;
    part.mi_col = leaf.mi_col;
    part.eob = leaf.eob;
    part.levels = leaf.levels;
    plan.parts.push_back(std::move(part));
  }
  if (plan.parts.empty()) {
    error = "TX search returned no transform leaves";
    return false;
  }
  plan.tx_size = plan.parts[0].tx_size;
  plan.eob = plan.parts[0].eob;
  return true;
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
                    const VideoOptions& options, bool trace, bool keyframe,
                    int tile_x = 0, int tile_y = 0, int tile_w = 0,
                    int tile_h = 0, std::span<const std::uint16_t> cdf_seed = {},
                    std::vector<std::uint16_t>* cdf_final = nullptr,
                    const GmFramePlan* gm_plan = nullptr,
                    std::array<int, 7> ref_order_hints = {})
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
    // Coded (tile-local) geometry.  A zero tile_w/tile_h means "whole frame",
    // which keeps the historical single-tile path untouched.  Source reads and
    // motion-compensated predictions add tile_x_/tile_y_ so the reference frame
    // stays global, exactly as the decoder's tile-local walk sees it.
    width_ = tile_w > 0 ? tile_w : static_cast<int>(source.width);
    height_ = tile_h > 0 ? tile_h : static_cast<int>(source.height);
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
    frame_mi_cols_ = static_cast<int>(2U * ((source.width + 7U) >> 3U));
    frame_mi_rows_ = static_cast<int>(2U * ((source.height + 7U) >> 3U));

    recon_[0].resize(mi_cols_ * 4, mi_rows_ * 4, 0);
    recon_[1].resize(mi_cols_ * 2, mi_rows_ * 2, 0);
    recon_[2].resize(mi_cols_ * 2, mi_rows_ * 2, 0);
    compound_pred_[0].resize(mi_cols_ * 4, mi_rows_ * 4, 0);
    compound_pred_[1].resize(mi_cols_ * 2, mi_rows_ * 2, 0);
    compound_pred_[2].resize(mi_cols_ * 2, mi_rows_ * 2, 0);

    quant_step_ = frame_quant_step(options_);
    mv_grid_.assign(static_cast<std::size_t>(mi_rows_) *
                        static_cast<std::size_t>(mi_cols_),
                    inter::mc::MiInfo{});
    // §5.20.5.5 IntraJointMode per 4x4 cell, used only by the opt-in full
    // intra search to derive §7.13.2.15/.16 filterTypeAbove/Left.  Inter
    // leaves write 0, exactly like the syntax writer's inter_block_mode_info().
    intra_joint_.assign(static_cast<std::size_t>(mi_rows_) *
                            static_cast<std::size_t>(mi_cols_),
                        0);
    // §6.19.2.3 BlockDecoded maps (reset at every superblock root).
    decoded_[0].reset(mi_cols_, mi_rows_);
    decoded_[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    decoded_[2].reset(mi_cols_ / 2, mi_rows_ / 2);
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

    // Opt-in M2 loop filters (§7.18–§7.20).  They run on the completed
    // reconstruction, before it becomes the output picture and the reference
    // for later frames.  The plan is passed to the tile writer so the
    // per-superblock / per-block symbols match the encoder decisions.
    filter_plan_ = post::VideoFilterPlan{};
    if (options_.loop_filters.any()) {
      if (width_ != static_cast<int>(source_.width) ||
          height_ != static_cast<int>(source_.height)) {
        error = "video loop filters need the single-tile writer";
        return false;
      }
      post::VideoLoopFilterFrame lf;
      lf.width = width_;
      lf.height = height_;
      lf.mi_cols = mi_cols_;
      lf.mi_rows = mi_rows_;
      lf.frame_is_intra = keyframe_;
      lf.base_q_idx = options_.base_q_idx;
      lf.num_total_refs = num_total_refs_;
      lf.restricted_oh = -1;
      for (int r = 0; r < 2; ++r) {
        lf.ref_order_hints[r] = ref_order_hints_[static_cast<std::size_t>(r)];
        lf.frame_distance[r] = frame_distance_[static_cast<std::size_t>(r)];
      }
      // §7.18.1 Skips[]: derived from every coded inter leaf's skip_flag.  The
      // decoder derives the same array, and with CdefOnSkipTxfm DISABLED it
      // leaves a fully-skipped 8x8 block unfiltered.
      std::vector<post::VideoLoopFilterBlock> filter_blocks;
      if (!keyframe_) {
        filter_blocks.reserve(blocks_.size());
        for (const syntax::BlockSyntax& leaf : blocks_) {
          if (leaf.partition !=
              static_cast<std::uint8_t>(PartitionType::kNone)) {
            continue;
          }
          post::VideoLoopFilterBlock block;
          block.mi_row = leaf.mi_row;
          block.mi_col = leaf.mi_col;
          block.w4 =
              partition::block_width(
                  static_cast<partition::BlockSize>(leaf.bsize)) >>
              2;
          block.h4 =
              partition::block_height(
                  static_cast<partition::BlockSize>(leaf.bsize)) >>
              2;
          block.skip = leaf.is_inter && leaf.skip_flag != 0;
          filter_blocks.push_back(block);
        }
      }
      lf.blocks = filter_blocks;
      const std::span<const std::uint8_t> src[3] = {source_.y, source_.u,
                                                    source_.v};
      for (int plane = 0; plane < 3; ++plane) {
        lf.plane[plane] = recon_[plane].samples.data();
        lf.stride[plane] = recon_[plane].width;
        lf.source[plane] = src[plane].data();
        lf.source_stride[plane] =
            plane == 0 ? static_cast<int>(source_.width)
                       : static_cast<int>(source_.width / 2);
      }
      if (!post::run_video_loop_filters(lf, options_.loop_filters,
                                        &filter_plan_, nullptr, &error)) {
        return false;
      }
    }

    syntax::FramePlan plan;
    plan.width = width_;
    plan.height = height_;
    plan.sb_size = 64;
    plan.inter_frame = !keyframe_;
    plan.tx_mode_select = options_.enable_tx_search;
    plan.reduced_tx_part_set = false;
    plan.reduced_tx_set = 0;
    plan.num_total_refs = num_total_refs_;
    plan.reference_select = options_.compound && num_total_refs_ > 1;
    plan.mv_precision = options_.mv_precision;
    plan.drl_reorder = options_.drl_reorder
                           ? (options_.drl_reorder_always ? 2 : 1)
                           : 0;
    plan.frame_distance = frame_distance_;
#if defined(AV2ENC_VIDEO_GME)
    if (gm_plan_ != nullptr) {
      plan.gm_num_refs = gm_plan_->num_refs;
      plan.gm_params = gm_plan_->params;
    }
#endif
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
    config.enable_cfl_intra = true;
    config.enable_deblock = options_.enable_deblock;
    config.tx_mode_select = options_.enable_tx_search;

    std::size_t tile_bytes = 0;
    if (!syntax::assemble_video_frame(plan, blocks_, levels_, config,
                                   tile_payload, error, &tile_bytes,
                                   cdf_seed_, cdf_final_,
                                   filter_plan_.any() ? &filter_plan_ : nullptr)) {
      return false;
    }
    // §7.17: deblock the reconstruction before it is emitted or stored as a
    // reference.  The metadata comes from the real per-leaf decisions above.
    // Multi-tile deblocking is rejected at the sequence level (the pinned
    // decoder filters across interior tile edges; see docs/negative-results.md).
    if (options_.enable_deblock) {
      const bool tile_local =
          (tile_w_ != 0 && tile_w_ != static_cast<int>(source_.width)) ||
          (tile_h_ != 0 && tile_h_ != static_cast<int>(source_.height));
      if (tile_local) {
        error = "video deblock: per-tile deblocking is not supported";
        return false;
      }
      post::VideoDeblockGrid grid;
      if (!post::build_video_deblock_grid(blocks_, mi_cols_, mi_rows_,
                                          options_.base_q_idx, keyframe_, &grid,
                                          &error)) {
        return false;
      }
      std::uint8_t* data[3] = {recon_[0].samples.data(),
                               recon_[1].samples.data(),
                               recon_[2].samples.data()};
      const int stride[3] = {recon_[0].stride, recon_[1].stride,
                             recon_[2].stride};
      if (!post::apply_video_deblock_host(grid, data, stride, &error)) {
        return false;
      }
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
  const post::VideoFilterPlan& filter_plan() const { return filter_plan_; }

 private:
  bool collect_node(const PartitionNode& node, std::string& error) {
    if (node.depth == 0) {
      // §6.19.2.3 clear_block_decoded_flags(): the map is superblock-local.
      decoded_[0].reset(mi_cols_, mi_rows_);
      decoded_[1].reset(mi_cols_ / 2, mi_rows_ / 2);
      decoded_[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    }
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

  // §7.12.2.2 GLOBALMV projection for this block.  A non-identity model may be
  // predicted with the warped predictor (`use_warp`); otherwise the projected
  // centre MV drives ordinary translational MC.  When global motion is off (or
  // the reference has no model) this is the historical zero MV.
  void global_mv_for_block(int ref_index, int mi_row, int mi_col, int bw4,
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
    use_warp = bw4 * 4 >= 8 && bh4 * 4 >= 8;
#else
    (void)ref_index;
    (void)mi_row;
    (void)mi_col;
    (void)bw4;
    (void)bh4;
#endif
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
    ref.plane[0] =
        me::MeCorePlane{references_[ref_index].plane[0].data,
                        references_[ref_index].plane[0].stride,
                        frame_mi_cols_ * 4, frame_mi_rows_ * 4};
    ref.plane[1] =
        me::MeCorePlane{references_[ref_index].plane[1].data,
                        references_[ref_index].plane[1].stride,
                        frame_mi_cols_ * 2, frame_mi_rows_ * 2};
    ref.plane[2] =
        me::MeCorePlane{references_[ref_index].plane[2].data,
                        references_[ref_index].plane[2].stride,
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

  // §7.13.2.15/.16 neighbour smooth flags at the coding-block origin, derived
  // from the IntraJointMode grid exactly like still_encoder's
  // neighborhood_smooth().  Only the opt-in full intra search needs them.
  void luma_neighbor_smooth(int r, int c, bool& above_smooth,
                            bool& left_smooth) const {
    above_smooth = false;
    left_smooth = false;
    if (r > 0 && c < mi_cols_) {
      above_smooth = intra_search::joint_is_smooth(
          intra_joint_[static_cast<std::size_t>(r - 1) *
                           static_cast<std::size_t>(mi_cols_) +
                       static_cast<std::size_t>(c)]);
    }
    if (c > 0 && r < mi_rows_) {
      left_smooth = intra_search::joint_is_smooth(
          intra_joint_[static_cast<std::size_t>(r) *
                           static_cast<std::size_t>(mi_cols_) +
                       static_cast<std::size_t>(c - 1)]);
    }
  }

  void set_intra_joint(int r, int c, int w4, int h4, int joint) {
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = r + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = c + xx;
        if (cc >= mi_cols_) break;
        intra_joint_[static_cast<std::size_t>(rr) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(cc)] =
            static_cast<std::uint8_t>(joint);
      }
    }
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
    // Absolute luma/chroma coordinates for source reads and MC.  The coded
    // geometry and reconstruction stay tile-local.
    const int ax0 = x0 + tile_x_;
    const int ay0 = y0 + tile_y_;
    const int acx0 = cx0 + tile_x_ / 2;
    const int acy0 = cy0 + tile_y_ / 2;

    syntax::BlockSyntax& entry = blocks_[index];

    // The CUDA pipeline evaluates inter candidates only when the luma block
    // fits inside the reconstructed reference extent.  A bottom/right edge node
    // can straddle the MI-aligned plane; both paths must make the same choice,
    // so the whole-frame CPU reference applies the same test.  Tiled coding has
    // its own clipped geometry and keeps the historical path.
    const bool inter_fits =
        width_ != static_cast<int>(source_.width) ||
        height_ != static_cast<int>(source_.height) ||
        (x0 + w <= mi_cols_ * 4 && y0 + h <= mi_rows_ * 4);

    InterCandidate best;
    const auto consider = [&](InterCandidate cand) {
      if (!cand.valid) return;
      if (!best.valid || cand.sse < best.sse) best = cand;
    };

    if (!keyframe_ && inter_fits) {
      for (int ref_index = 0; ref_index < num_total_refs_; ++ref_index) {
        merge::StackParams stack_params;
        merge::MvStack stack;
        const bool have_stack =
            load_stack(r, c, w4, h4, ref_index, stack_params, stack);

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
                references_[ref_index], 0, ax0, ay0, w, h, r, c, w4, h4,
                luma, warp_error);
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
            cand.sse = sse_against_source(source_, 0, ax0, ay0, w, h, luma);
            if (trace_) {
              std::cerr << "  cand ref=" << ref_index << " global mv=("
                        << gm_mv.y << "," << gm_mv.x << ") warp="
                        << (gm_use_warp ? 1 : 0) << " sse=" << cand.sse
                        << "\n";
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
              if (!predict_luma(ref_index, near_mv, ax0, ay0, w, h, luma)) {
                continue;
              }
              InterCandidate cand;
              cand.valid = true;
              cand.mode = syntax::kInterNearMv;
              cand.ref_index = ref_index;
              cand.drl_index = i;
              cand.mv = near_mv;
              cand.sse = sse_against_source(source_, 0, ax0, ay0, w, h, luma);
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
            if (run_me(ref_index, ax0, ay0, w, h, me_mv, error)) {
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
                if (predict_luma(ref_index, new_mv, ax0, ay0, w, h, luma)) {
                  InterCandidate cand;
                  cand.valid = true;
                  cand.mode = syntax::kInterNewMv;
                  cand.ref_index = ref_index;
                  cand.drl_index = 0;
                  cand.mv = new_mv;
                  cand.mvd = options_.mv_precision == 0 ? resolved.mvd[0] : mvd;
                  cand.sse =
                      sse_against_source(source_, 0, ax0, ay0, w, h, luma);
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
              inter::mc::Mv{0, 0}, inter::mc::InterpFilter::kEighttap,
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
        if (multiref::apply_compound_prediction(prediction, params, dst,
                                                dst_stride, comp_error)) {
          compound_y_.resize(static_cast<std::size_t>(w) *
                             static_cast<std::size_t>(h));
          compound_u_.resize(static_cast<std::size_t>(cw) *
                             static_cast<std::size_t>(ch));
          compound_v_.resize(static_cast<std::size_t>(cw) *
                             static_cast<std::size_t>(ch));
          for (int yy = 0; yy < h; ++yy) {
            std::memcpy(compound_y_.data() +
                            static_cast<std::size_t>(yy) * w,
                        compound_pred_[0].samples.data() +
                            static_cast<std::size_t>(y0 + yy) *
                                compound_pred_[0].stride +
                            x0,
                        static_cast<std::size_t>(w));
          }
          for (int yy = 0; yy < ch; ++yy) {
            std::memcpy(compound_u_.data() +
                            static_cast<std::size_t>(yy) * cw,
                        compound_pred_[1].samples.data() +
                            static_cast<std::size_t>(cy0 + yy) *
                                compound_pred_[1].stride +
                            cx0,
                        static_cast<std::size_t>(cw));
            std::memcpy(compound_v_.data() +
                            static_cast<std::size_t>(yy) * cw,
                        compound_pred_[2].samples.data() +
                            static_cast<std::size_t>(cy0 + yy) *
                                compound_pred_[2].stride +
                            cx0,
                        static_cast<std::size_t>(cw));
          }
          InterCandidate cand;
          cand.valid = true;
          cand.compound = true;
          cand.mode = syntax::kInterGlobalMv;
          cand.ref_index = 0;
          cand.mv = inter::mc::Mv{0, 0};
          cand.sse = sse_against_source(source_, 0, x0, y0, w, h, compound_y_);
          if (trace_) {
            std::cerr << "  cand compound global-global sse=" << cand.sse
                      << "\n";
          }
          consider(cand);
        }
      }
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
    std::uint8_t intra_y_mode = 0;  // AV2 YMode (0 = DC_PRED)
    std::int8_t intra_y_angle_delta = 0;
    // The reconstruction planes are tile-local, so plain bounds give the
    // decoder's §5.20.9.1 availability at every tile edge.
    const int luma_max_x = recon_[0].width - 1;
    const int luma_max_y = recon_[0].height - 1;
    const bool luma_left = x0 > 0;
    const bool luma_above = y0 > 0;
    const int chroma_max_x = recon_[1].width - 1;
    const int chroma_max_y = recon_[1].height - 1;
    const bool chroma_left = cx0 > 0;
    const bool chroma_above = cy0 > 0;
    if (keyframe_ || options_.enable_intra_fallback || options_.force_intra) {
      bool luma_ok = false;
      if (options_.enable_intra_full) {
        bool above_smooth = false;
        bool left_smooth = false;
        luma_neighbor_smooth(r, c, above_smooth, left_smooth);
        const intra_search::Availability avail = intra_search::availability(
            decoded_[0], x0 / 4, y0 / 4, w / 4, h / 4, mi_cols_, mi_rows_,
            /*sb_mi=*/16);
        intra_search::LumaChoice choice;
        intra_y_pred.resize(static_cast<std::size_t>(w) *
                            static_cast<std::size_t>(h));
        luma_ok = intra_search::search_luma(
            recon_[0].samples.data(), recon_[0].stride, x0, y0, w, h, avail,
            above_smooth, left_smooth, /*enable_edge_filter=*/true,
            source_.y.data(), static_cast<int>(source_.width),
            static_cast<int>(source_.width) - 1,
            static_cast<int>(source_.height) - 1, intra_y_pred.data(), choice);
        if (luma_ok) {
          intra_y_mode = static_cast<std::uint8_t>(choice.mode);
          intra_y_angle_delta = static_cast<std::int8_t>(choice.angle_delta);
        }
      } else {
        luma_ok = intra_dc_predict(recon_[0], source_, 0, x0, y0, w, h,
                                   luma_max_x, luma_max_y, luma_left,
                                   luma_above, intra_y_pred);
      }
      if (!luma_ok ||
          !intra_dc_predict(recon_[1], source_, 1, cx0, cy0, cw, ch,
                            chroma_max_x, chroma_max_y, chroma_left,
                            chroma_above, intra_u_pred) ||
          !intra_dc_predict(recon_[2], source_, 2, cx0, cy0, cw, ch,
                            chroma_max_x, chroma_max_y, chroma_left,
                            chroma_above, intra_v_pred)) {
        error = "intra prediction failed";
        return false;
      }
      if ((options_.enable_tx_search
               ? !search_luma_tx(source_, ax0, ay0, w, h, intra_y_pred,
                                 options_.base_q_idx,
                                 txtsearch::Mode::kIntra,
                                 options_.tx_rd_lambda, intra_y, error)
               : !compute_tx(source_, 0, ax0, ay0, w, h, intra_y_pred,
                             quant_step_, intra_y, error)) ||
          !compute_tx(source_, 1, acx0, acy0, cw, ch, intra_u_pred, quant_step_,
                      intra_u, error) ||
          !compute_tx(source_, 2, acx0, acy0, cw, ch, intra_v_pred, quant_step_,
                      intra_v, error)) {
        return false;
      }
      intra_sse = sse_against_source(source_, 0, ax0, ay0, w, h, intra_y.recon);
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
      if (best.compound) {
        inter_y = compound_y_;
        inter_u = compound_u_;
        inter_v = compound_v_;
      } else if (best.use_warp) {
#if defined(AV2ENC_VIDEO_GME)
        std::string warp_error;
        if (gm_plan_ == nullptr || best.ref_index >= gm_plan_->num_refs ||
            !warp_predict_420(
                gm_plan_->params[static_cast<std::size_t>(best.ref_index)],
                references_[best.ref_index], ax0, ay0, w, h, r, c, w4, h4,
                inter_y, inter_u, inter_v, warp_error)) {
          error = "inter warp prediction failed: " + warp_error;
          return false;
        }
#else
        error = "inter warp prediction is not linked in this build";
        return false;
#endif
      } else if (!predict_all(best.ref_index, best.mv, ax0, ay0, w, h, inter_y,
                              inter_u, inter_v)) {
        error = "inter prediction failed";
        return false;
      }
      inter_luma_sse =
          sse_against_source(source_, 0, ax0, ay0, w, h, inter_y);
      if (options_.enable_inter_residual) {
        const std::int64_t pred_total =
            inter_luma_sse +
            sse_against_source(source_, 1, acx0, acy0, cw, ch, inter_u) +
            sse_against_source(source_, 2, acx0, acy0, cw, ch, inter_v);
        if ((options_.enable_tx_search
                 ? !search_luma_tx(source_, ax0, ay0, w, h, inter_y,
                                   options_.base_q_idx,
                                   txtsearch::Mode::kInter,
                                   options_.tx_rd_lambda, res_y, error)
                 : !compute_tx(source_, 0, ax0, ay0, w, h, inter_y,
                               quant_step_, res_y, error)) ||
            !compute_tx(source_, 1, acx0, acy0, cw, ch, inter_u, quant_step_,
                        res_u, error) ||
            !compute_tx(source_, 2, acx0, acy0, cw, ch, inter_v, quant_step_,
                        res_v, error)) {
          return false;
        }
        const std::int64_t res_total =
            sse_against_source(source_, 0, ax0, ay0, w, h, res_y.recon) +
            sse_against_source(source_, 1, acx0, acy0, cw, ch, res_u.recon) +
            sse_against_source(source_, 2, acx0, acy0, cw, ch, res_v.recon);
        const bool all_zero =
            luma_plan_all_zero(res_y) && res_u.eob == 0 && res_v.eob == 0;
        // Reconstruction step is qlookup/8 (66 at QP 90); sse_scale 0.5 is
        // the M1 calibrated value.
        const double kLambda = options_.inter_rd_lambda;
        const double res_bits = 3.0 + luma_plan_bits(res_y, true) +
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
              sse_against_source(source_, 0, ax0, ay0, w, h, inter_y);
        }
      }
      if (trace_) {
        std::cerr << "  cand inter final sse=" << inter_luma_sse
                  << " residual=" << (inter_residual ? 1 : 0) << "\n";
      }
    }

    bool use_intra = keyframe_ || options_.force_intra || !best.valid;
    if (!use_intra && have_intra) {
      if (options_.enable_inter_residual) {
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

    if (use_intra) {
      if (!have_intra) {
        error = "intra fallback unavailable";
        return false;
      }
      entry.is_inter = 0;
      entry.skip_flag = 0;
      entry.y_mode = intra_y_mode;
      entry.y_angle_delta = intra_y_angle_delta;
      entry.uv_mode = 0;  // DC_PRED
      write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_u.recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_v.recon);
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      write_luma_plan(entry, intra_y, levels_);
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
            levels_.push_back(plans[plane - 1]->levels[scan[
                static_cast<std::size_t>(c2)]]);
          }
        }
      }
      update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
      set_intra_joint(r, c, w4, h4,
                      syntax::intra_joint_mode(intra_y_mode,
                                               intra_y_angle_delta));
      ++intra_blocks_;
      if (trace_) {
        std::cerr << "m2 intra r=" << r << " c=" << c << " " << w << "x" << h
                  << " mode=" << static_cast<int>(intra_y_mode)
                  << " delta=" << static_cast<int>(intra_y_angle_delta)
                  << " sse=" << intra_sse << "\n";
      }
    } else {
      entry.is_inter = 1;
      entry.skip_flag = inter_residual ? 0 : 1;
      entry.inter_mode = static_cast<std::uint8_t>(best.mode);
      entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
      entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
      entry.compound = best.compound ? 1 : 0;
      entry.ref_index1 = 1;
      entry.compound_joint = 0;
      entry.compound_mode = 3;  // GLOBAL_GLOBALMV
      if (best.compound) {
        // A compound leaf never carries a residual in the gated subset.
        entry.skip_flag = 1;
        inter_residual = false;
      }
      entry.mv_y = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.y;
      entry.mv_x = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.x;
      entry.mvd_y = best.mvd.y;
      entry.mvd_x = best.mvd.x;
      write_plane_block(recon_[0], x0, y0, w, h, inter_y);
      write_plane_block(recon_[1], cx0, cy0, cw, ch, inter_u);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, inter_v);
      if (inter_residual) {
        entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
        write_luma_plan(entry, res_y, levels_);
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
            levels_.push_back(plans[plane - 1]->levels[scan[
                static_cast<std::size_t>(c2)]]);
          }
        }
      }
      update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv,
                     best.compound ? 1 : merge::kRefNone);
      set_intra_joint(r, c, w4, h4, 0);  // inter_block_mode_info(): DC
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

    // §6.19.2.3: this leaf's cells are now decoded for the availability of the
    // following leaves in this superblock.
    decoded_[0].fill(c, r, w4, h4);
    decoded_[1].fill(c / 2, r / 2, cw / 4, ch / 4);
    decoded_[2].fill(c / 2, r / 2, cw / 4, ch / 4);
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
        mi.ref[0] = is_inter ? static_cast<std::int8_t>(ref_index)
                             : kRefIntraAv2;
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

  const VideoInputFrame& source_;
  std::vector<inter::mc::RefFrame> references_;
  std::array<int, 7> frame_distance_{};
  int num_total_refs_ = 1;
  const VideoOptions& options_;
  bool trace_ = false;
  bool keyframe_ = false;
  int tile_x_ = 0;
  int tile_y_ = 0;
  int tile_w_ = 0;
  int tile_h_ = 0;
  std::span<const std::uint16_t> cdf_seed_;
  std::vector<std::uint16_t>* cdf_final_ = nullptr;
  const GmFramePlan* gm_plan_ = nullptr;
  int quant_step_ = txv::kQuantStep;  // §7.14.2 qlookup(base_q_idx)
  int width_ = 0;
  int height_ = 0;
  int frame_mi_cols_ = 0;
  int frame_mi_rows_ = 0;
  int chroma_width_ = 0;
  int chroma_height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  FrameGeometry geometry_;
  Plane recon_[3];
  Plane compound_pred_[3];
  post::VideoFilterPlan filter_plan_;
  std::vector<std::uint8_t> compound_y_;
  std::vector<std::uint8_t> compound_u_;
  std::vector<std::uint8_t> compound_v_;
  inter::mc::RefFrame ref_frame_;
  std::vector<inter::mc::MiInfo> mv_grid_;
  std::vector<std::uint8_t> intra_joint_;
  std::array<intra_search::BlockDecoded, 3> decoded_;
  std::vector<syntax::BlockSyntax> blocks_;
  std::array<int, 7> ref_order_hints_{};
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
  config.enable_masked_compound = false;
  config.enable_cwp = false;
  config.enable_imp_msk_bld = false;
  config.num_same_ref_compound = 0;
  config.disable_drl_reorder = true;
  config.constrain_drl_reorder = true;
  config.enable_cfl_intra = true;
  config.enable_deblock = options.enable_deblock;
  config.tx_mode_select = options.enable_tx_search;
  config.max_tlayer_id = 0;
  config.monotonic_output_order_flag = false;

  const int mi_cols = 2 * static_cast<int>((width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((height + 7U) >> 3U);
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
    std::vector<inter::mc::RefFrame> references;
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < num_refs; ++r) {
      const int ref_display = fp->ref_display[r];
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
      frame_distance[static_cast<std::size_t>(r)] = static_cast<int>(
          relative_dist(static_cast<std::uint32_t>(fp->display_order_hint),
                        static_cast<std::uint32_t>(fp->ref_order_hint[r]),
                        plan.order_hint_bits));
    }

    InterFrameEncoder encoder(frames[static_cast<std::size_t>(display)],
                              std::move(references), frame_distance,
                              std::max(1, num_refs), frame_options,
                              options.trace, fp->is_key);
    std::vector<std::uint8_t> tile;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "B-frame display " + std::to_string(display) + ": " + error;
      return false;
    }

    const bf::FrameHeaderFields hf = bf::header_fields(*fp);
    VideoFrameHeader header;
    header.keyframe = hf.keyframe;
    header.order_hint = hf.order_hint_lsbs;
    header.num_total_refs = hf.num_total_refs;
    for (int r = 0; r < hf.num_total_refs; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] = hf.ref_frame_idx[r];
    }
    header.ref_slot = header.ref_frame_idx[0];
    header.refresh_frame_flags = hf.refresh_frame_flags;
    header.immediate_output_frame = hf.immediate_output_frame;
    header.implicit_output_frame = hf.implicit_output_frame;
    header.reference_select = hf.reference_select && options.compound &&
                              hf.num_total_refs > 1;
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
    const std::vector<std::uint8_t> obu = make_frame_obu(config, header, tile, 0);
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2), width, height);
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
  if (!options.frame_base_q_idx.empty()) {
    error = "--video-tlayers does not compose with per-frame rate control";
    return false;
  }

  tl::PlanConfig pc;
  pc.gop_size = gop;
  pc.schedule = tl::Schedule::kHierarchicalP;
  pc.dependency = tl::DependencyMode::kAllLower;
  pc.explicit_dependency_map = false;  // pinned dav2d parses the default map
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
  config.enable_deblock = options.enable_deblock;
  config.tx_mode_select = options.enable_tx_search;
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

  const int mi_cols = 2 * static_cast<int>((width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((height + 7U) >> 3U);
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

    std::vector<inter::mc::RefFrame> references;
    std::array<int, 7> frame_distance{};
    if (!pf.is_key) {
      const int ref_display = pf.ref_display[0];
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
      frame_distance[0] = static_cast<int>(relative_dist(
          pf.display_order_hint, pf.ref_order_hint[0],
          static_cast<int>(pc.order_hint_bits)));
    }

    InterFrameEncoder encoder(frames[static_cast<std::size_t>(display)],
                              std::move(references), frame_distance, 1,
                              frame_options, options.trace, pf.is_key);
    std::vector<std::uint8_t> tile;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    if (!encoder.encode(tile, blocks, levels, error)) {
      error = "temporal-layer frame " + std::to_string(display) + ": " + error;
      return false;
    }

    VideoFrameHeader header;
    header.keyframe = pf.is_key;
    header.order_hint = pf.order_hint_lsbs;
    header.num_total_refs = static_cast<std::uint8_t>(pf.num_refs);
    for (int r = 0; r < pf.num_refs; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(pf.ref_slot[r]);
    }
    header.ref_slot = static_cast<std::uint8_t>(pf.ref_slot[0]);
    header.refresh_frame_flags = pf.refresh_frame_flags;
    header.immediate_output_frame = pf.immediate_output_frame;
    header.implicit_output_frame = pf.implicit_output_frame;
    header.reference_select = false;  // --video-tlayers is single-reference
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
        make_frame_obu(config, header, tile,
                       static_cast<std::uint8_t>(pf.obu_tlayer_id()));
    unit.insert(unit.end(), obu.begin(), obu.end());
    result.units.push_back(std::move(unit));

    result.tile_bytes.push_back(static_cast<std::uint32_t>(tile.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    recon_by_display[static_cast<std::size_t>(display)] = to_recon_frame(
        encoder.recon(0), encoder.recon(1), encoder.recon(2), width, height);
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

}  // namespace

int resolved_num_ref_frames(const VideoOptions& options) {
  if (options.num_ref_frames != 0) return options.num_ref_frames;
  return options.num_total_refs > 1 ? kMaxRefSlots : 2;
}

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
  // The video leaf coder derives 4:2:0 chroma as w/2, h/2 and the intra
  // predictor requires both dimensions >= 4. A sub-8 luma leaf (4x4, 4x8,
  // 32x4, ...) would need §5.20.3.1 shared-chroma ownership; until that is
  // implemented, reject it instead of aborting inside the keyframe with
  // "intra DC prediction failed".
  if (options.min_block_size < 8) {
    error = "min_block_size must be at least 8: the video path does not "
            "implement shared-chroma prediction for sub-8 luma leaves";
    return false;
  }
  if (options.max_partition_depth < 0 || options.max_partition_depth > 8) {
    error = "max_partition_depth must be in [0, 8]";
    return false;
  }
  if (options.enable_tx_search && options.enable_deblock) {
    error = "--video-tx-search does not compose with --video-deblock yet";
    return false;
  }
  if (options.enable_tx_search && (options.avg_cdf || options.adaptive_cdf)) {
    error = "--video-tx-search does not compose with --avg-cdf/--adaptive-cdf";
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
  // The opt-in loop filters are wired on the flat single-tile key+P chain.  The
  // frame-type wrappers, B frames and temporal layers rebuild frame headers in
  // their own loops; enabling a filter there without carrying the per-frame
  // plan would desynchronise the decoder, so reject by name instead.
  if (options.loop_filters.any()) {
    if (options.tile_cols > 1 || options.tile_rows > 1) {
      error = "video loop filters do not compose with --tiles > 1x1 yet";
      return false;
    }
    if (options.b_frames > 0) {
      error = "video loop filters do not compose with --b-frames yet";
      return false;
    }
    if (options.temporal_layers > 0) {
      error = "video loop filters do not compose with --video-tlayers yet";
      return false;
    }
    if (options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras) {
      error = "video loop filters do not compose with the random-access "
              "frame-type modes yet";
      return false;
    }
  }
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
  if (options.mv_precision < 0 || options.mv_precision > 2) {
    error = "mv_precision must be 0 (1/8), 1 (1/4) or 2 (1/2) pel";
    return false;
  }
  if (options.drl_reorder) {
    error = "DRL reordering is not decoder-gated yet "
            "(docs/negative-results.md)";
    return false;
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
  const int num_total_refs = options.num_total_refs;
  const int num_ref_frames = resolved_num_ref_frames(options);
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
  tile_sequence.filters.disable_loopfilters_across_tiles = true;
  // §5.4.8: the tiled frame writer owns its own sequence header, so it must
  // signal the average-CDF flags the per-tile bank below actually runs.
  tile_sequence.enable_avg_cdf = options.avg_cdf;
  tile_sequence.avg_cdf_type = true;
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
  Plane frame_recon[3];
  frame_recon[0].resize(recon_mi_cols * 4, recon_mi_rows * 4, 0);
  frame_recon[1].resize(recon_mi_cols * 2, recon_mi_rows * 2, 0);
  frame_recon[2].resize(recon_mi_cols * 2, recon_mi_rows * 2, 0);

  const auto paste_tile = [](const Plane& src, Plane& dst, int x0, int y0,
                             int w, int h) {
    for (int y = 0; y < h; ++y) {
      std::memcpy(dst.samples.data() +
                      static_cast<std::size_t>(y0 + y) *
                          static_cast<std::size_t>(dst.stride) +
                      static_cast<std::size_t>(x0),
                  src.samples.data() +
                      static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(src.stride),
                  static_cast<std::size_t>(w));
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
  if (cdf_chain && !options.frame_base_q_idx.empty()) {
    error = "average-CDF/continuity requires a fixed base_q_idx "
            "(no --rc crf/vbr per-frame QP)";
    return false;
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
    if (!syntax::cdf_image_defaults_for_base_q_idx(
            static_cast<int>(options.base_q_idx), cdf_defaults, &error)) {
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
    // CLK resets to the §9.3 defaults (§5.18.2: FrameIsIntra,
    // primary_ref_frame = PRIMARY_REF_NONE).
    cdf_bank.begin_frame(/*reset_to_defaults=*/true);
  }
#endif

  const VideoOptions key_options = options_for_frame(0);

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = order_hint_bits;
  config.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_masked_compound = false;
  config.enable_cwp = false;
  config.enable_imp_msk_bld = false;
  config.num_same_ref_compound = 0;
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
    // off: pinned dav2d f9dafbc cannot parse it (docs/negative-results.md).
    config.initial_display_delay_present = true;
    config.initial_display_delay_minus_1 = 0;
    config.decoder_model_info_present = true;
    config.num_units_in_decoding_tick =
        options.timing_num_units_in_display_tick;
  }
  config.disable_drl_reorder = !options.drl_reorder;
  config.constrain_drl_reorder = !options.drl_reorder_always;
  config.enable_cfl_intra = true;
  config.enable_deblock = options.enable_deblock;
  config.enable_global_motion = options.enable_gme;
  // §5.4.10 CDEF/CCSO sequence flags for the opt-in M2 loop filters.
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
  if (options.loop_filters.wiener || options.loop_filters.chroma_lr) {
    config.enable_restoration = true;
    config.lr_disable_luma_pc_wiener = true;
    config.lr_disable_luma_wiener_ns = !options.loop_filters.wiener;
    config.lr_uv_present = options.loop_filters.chroma_lr;
    config.lr_disable_chroma_wiener_ns = !options.loop_filters.chroma_lr;
  }  config.tx_mode_select = options.enable_tx_search;  config.seq_lcr_id = options.seq_lcr_id;
  config.max_tlayer_id = options.max_tlayer_id;
  config.max_mlayer_id = options.max_mlayer_id;
  config.seq_max_mlayer_cnt = options.seq_max_mlayer_cnt;

  VideoFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.ref_slot = 0;
  key_header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;

  // Key frame: intra keyframe coded once per tile.
  std::uint32_t key_payload_bytes = 0;
  std::uint32_t key_intra_blocks = 0;
  std::vector<std::uint16_t> cdf_seed;
  if (cdf_chain) cdf_seed = cdf_tile;  // §8.2.2 FrameCdf seed for every tile
  [[maybe_unused]] std::uint32_t cdf_tile_index = 0;
  for (CodedTile& tile : tile_sources) {
    if (cdf_chain) cdf_tile = cdf_seed;
    InterFrameEncoder encoder(
        frames[0], no_refs, no_distance, 1, key_options, options.trace,
        /*keyframe=*/true, tile.x, tile.y, tile.w, tile.h,
        cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                  : std::span<const std::uint16_t>(),
        cdf_chain ? &cdf_scratch : nullptr);
    if (!encoder.encode(tile.payload, tile.blocks, tile.levels, error)) {
      error = "keyframe encode failed: " + error;
      return false;
    }
    tile.filter_plan = encoder.filter_plan();
    if (tile.payload.empty()) {
      error = "keyframe produced an empty tile payload";
      return false;
    }
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
    if (cdf_chain) {
      // exit_symbol() -> §8.2.4 Saved* average (§7.5 runs after the last tile).
      std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
      if (!cdf_bank.save_tile_cdfs(
              cdf_tile_index,
              static_cast<std::uint32_t>(tile_sources.size()))) {
        error = "average-CDF save_tile_cdfs rejected tile " +
                std::to_string(cdf_tile_index);
        return false;
      }
    }
#endif
    ++cdf_tile_index;
    key_payload_bytes += static_cast<std::uint32_t>(tile.payload.size());
    key_intra_blocks += encoder.intra_blocks();
    paste_tile(encoder.recon(0), frame_recon[0], tile.x, tile.y, tile.w, tile.h);
    paste_tile(encoder.recon(1), frame_recon[1], tile.x / 2, tile.y / 2,
               tile.w / 2, tile.h / 2);
    paste_tile(encoder.recon(2), frame_recon[2], tile.x / 2, tile.y / 2,
               tile.w / 2, tile.h / 2);
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
    key_header.cdef = tile_sources[0].filter_plan.cdef;
    key_header.ccso = tile_sources[0].filter_plan.ccso;
    key_header.lr = tile_sources[0].filter_plan.lr;
    key_header.gdf = tile_sources[0].filter_plan.gdf;
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
      seq = make_sequence_header_obu(config);
      obu = make_frame_obu(config, key_header, tile_sources[0].payload);
    }
    key_unit.reserve(td.size() + seq.size() + ci.size() + mfh_obu.size() +
                     obu.size());
    key_unit.insert(key_unit.end(), td.begin(), td.end());
    key_unit.insert(key_unit.end(), seq.begin(), seq.end());
    key_unit.insert(key_unit.end(), ci.begin(), ci.end());
    key_unit.insert(key_unit.end(), mfh_obu.begin(), mfh_obu.end());
    key_unit.insert(key_unit.end(), obu.begin(), obu.end());
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
  std::vector<int> recent_slots;
  slots[0].valid = true;
  slots[0].order_hint = 0;
  for (int plane = 0; plane < 3; ++plane) {
    slots[0].plane[plane] = frame_recon[plane];
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
    ref.mi_cols = recon_mi_cols;
    ref.mi_rows = recon_mi_rows;
    return ref;
  };

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
        const int slot = recent_slots[static_cast<std::size_t>(r)];
        ref_order_hint[static_cast<std::size_t>(r)] =
            static_cast<int>(slots[slot].order_hint);
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
      if (options.cdf_continuity) {
        // §7.4.4 load_cdfs(ref_frame_idx[primary_ref_frame]); we signal
        // primary_ref_frame = 0, so the first reference's saved CDFs are
        // installed into the frame state.
        const int primary_slot = recent_slots.front();
        std::copy(cdf_slots[static_cast<std::size_t>(primary_slot)].begin(),
                  cdf_slots[static_cast<std::size_t>(primary_slot)].end(),
                  cdf_frame.begin());
        cdf_bank.begin_frame(/*reset_to_defaults=*/false);
      } else {
        cdf_bank.begin_frame(/*reset_to_defaults=*/true);
      }
    }
#endif
    std::vector<std::uint16_t> cdf_seed;
    if (cdf_chain) cdf_seed = cdf_tile;  // §8.2.2 FrameCdf seed per tile
    [[maybe_unused]] std::uint32_t cdf_tile_index = 0;
    // §7.20.1 needs the reference OrderHints for the GDF table derivation.
    std::array<int, 7> ref_order_hints{};
    for (int r = 0; r < ref_count && r < 2; ++r) {
      ref_order_hints[static_cast<std::size_t>(r)] = static_cast<int>(
          slots[recent_slots[static_cast<std::size_t>(r)]].order_hint);
    }
    for (CodedTile& tile : tile_sources) {
      if (cdf_chain) cdf_tile = cdf_seed;
      InterFrameEncoder encoder(
          frames[i], references, frame_distance, ref_count, frame_options,
          options.trace, /*keyframe=*/false, tile.x, tile.y, tile.w, tile.h,
          cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                    : std::span<const std::uint16_t>(),
          cdf_chain ? &cdf_scratch : nullptr, gm_ptr, ref_order_hints);
      if (!encoder.encode(tile.payload, tile.blocks, tile.levels, error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      tile.filter_plan = encoder.filter_plan();
#if defined(AV2ENC_VIDEO_CDF_CHAIN)
      if (cdf_chain) {
        // exit_symbol() -> §8.2.4 Saved* average (§7.5 after the last tile).
        std::copy(cdf_scratch.begin(), cdf_scratch.end(), cdf_tile.begin());
        if (!cdf_bank.save_tile_cdfs(
                cdf_tile_index,
                static_cast<std::uint32_t>(tile_sources.size()))) {
          error = "average-CDF save_tile_cdfs rejected tile " +
                  std::to_string(cdf_tile_index);
          return false;
        }
      }
#endif
      ++cdf_tile_index;
      payload_bytes += static_cast<std::uint32_t>(tile.payload.size());
      inter_blocks += encoder.inter_blocks();
      intra_blocks += encoder.intra_blocks();
      paste_tile(encoder.recon(0), frame_recon[0], tile.x, tile.y, tile.w,
                 tile.h);
      paste_tile(encoder.recon(1), frame_recon[1], tile.x / 2, tile.y / 2,
                 tile.w / 2, tile.h / 2);
      paste_tile(encoder.recon(2), frame_recon[2], tile.x / 2, tile.y / 2,
                 tile.w / 2, tile.h / 2);
    }

    const int write_slot = static_cast<int>(i % num_ref_frames);
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
    header.order_hint = order_hint;
    if (cdf_chain) {
      header.signal_primary_ref_frame = options.cdf_continuity;
      header.disable_cross_frame_cdf_init = !options.cdf_continuity;
      header.primary_ref_frame = 0;
    }
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
    header.reference_select = options.compound && ref_count > 1;
    header.gm = gm_plan;
    header.cdef = tile_sources[0].filter_plan.cdef;
    header.ccso = tile_sources[0].filter_plan.ccso;
    header.lr = tile_sources[0].filter_plan.lr;
    header.gdf = tile_sources[0].filter_plan.gdf;

    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    config.base_q_idx = frame_qp(i);
    std::vector<std::uint8_t> obu;
    if (tiled) {
      post::VideoTiledFrameHeader tiled_header;
      tiled_header.keyframe = false;
      tiled_header.order_hint = header.order_hint;
      tiled_header.ref_slot = header.ref_slot;
      tiled_header.refresh_frame_flags = header.refresh_frame_flags;
      tiled_header.base_q_idx = config.base_q_idx;
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
    result.units.push_back(std::move(unit));

    result.recon.push_back(
        to_recon_frame(frame_recon[0], frame_recon[1], frame_recon[2], width,
                       height));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(payload_bytes);
    result.inter_blocks.push_back(inter_blocks);
    result.intra_blocks.push_back(intra_blocks);

    // Store the reconstruction and refresh the rotating slot.
    slots[write_slot].valid = true;
    slots[write_slot].order_hint = order_hint;
    for (int plane = 0; plane < 3; ++plane) {
      slots[write_slot].plane[plane] = frame_recon[plane];
    }
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
