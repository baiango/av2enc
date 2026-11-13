#include "pipeline/video/video_frame.h"

#include "codec/intra/modes/intra_modes.h"
#include "codec/intra/dip/dip.h"
#include "codec/intra/mrl/core/mrl.h"
#include "codec/intra/mrl/stream/mrl_interaction.h"
#include "codec/transform/secondary/tx_secondary.h"
#include "frame/metadata/content_interpretation/content_interpretation.h"
#include "frame/mfh/mfh.h"
#include "pipeline/video/video_frames.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_chroma_search.h"
#include "pipeline/video/video_intra_search.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/video_temporal.h"
#include "pipeline/video/video_tx_search.h"
#include "codec/partition/tree/partition.h"
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
#include "codec/partition/ext/partition_ext.h"
#endif
#include "codec/partition/sdp/sdp.h"
#include "codec/rate/rc/rc_tables.h"
#if defined(AV2ENC_VIDEO_DELTAQ)
#include "codec/entropy/deltaq/still/deltaq.h"
#include "codec/rate/rc/rc.h"
#endif
#include "codec/transform/variants/tx_variants.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/transform/ddt/ddt.h"
#include "codec/transform/qm/qm.h"
#include "codec/transform/tcq/cctx.h"
#include "codec/transform/tcq/tcq.h"
#include "codec/transform/tcq/tcq_av2_model.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/rate/rdo/rdo.h"
#include "pipeline/video/video_bawp.h"
#include "codec/inter/mc/mc.h"
#include "codec/inter/me/me.h"
#include "codec/inter/merge/merge.h"
#include "codec/inter/morph/morph_common.h"
#include "codec/inter/mv_stack/ref_map.h"
#include "codec/intra/ibc/core/ibc.h"
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/intra/ibc/stream/still_ibc_stream.h"
#include "codec/intra/ibp/ibp.h"
#include "codec/intra/palette/core/palette.h"
#include "codec/intra/palette/stream/palette_stream.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/opfl.h"
#include "codec/inter/temporal/refinemv.h"
#include "codec/inter/temporal/temporal_predict.h"
#include "codec/inter/tip/tip_build.h"
#include "pipeline/grain/grain_stream.h"
#if defined(AV2ENC_VIDEO_FRAMES)
#include "codec/inter/multiref/compound_predict.h"
#endif
#include "codec/inter/advanced/compound.h"
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
#if defined(AV2ENC_VIDEO_FORMATS)
#include "pipeline/video/hbd/video_hbd.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
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
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
namespace pe = partition_ext;
#endif
namespace temporal = inter::temporal;
namespace tip = inter::tip;
namespace sdpp = av2enc::sdp;
namespace ibcs = still::ibc_stream;
namespace mrl_stream = still::mrl_stream;
namespace txsec = ::av2enc::txsec;
namespace morph = inter::morph;
namespace pals = still::palette_stream;

// AV2 RefFrame labels used by the merge layer's MI grid.  mc::kRefLast is an
// internal MC label (1); §6.19.7.9 numbers LAST_FRAME = 0 and
// INTRA_FRAME = 8.
constexpr std::int8_t kRefLastAv2 = 0;
constexpr std::int8_t kRefIntraAv2 = 8;

namespace grain = ::av2enc::filters::grain;

// §5.14 film grain: estimate/plan one picture and insert the model OBU before
// the coded-frame OBU of its temporal unit (§7.3.3/.4, §7.3.8.8).
bool plan_grain_frame(const VideoInputFrame& frame, std::uint32_t index,
                      grain::GrainStreamPlanner& planner,
                      grain::GrainFramePlan& plan, std::string& error) {
  grain::GrainFrameView view;
  view.y = frame.y.data();
  view.u = frame.u.data();
  view.v = frame.v.data();
  view.width = static_cast<int>(frame.width);
  view.height = static_cast<int>(frame.height);
  view.y_stride = static_cast<int>(frame.width);
  view.c_stride = static_cast<int>(frame.width / 2U);
  if (!planner.PlanFrame(view, index, /*tlayer_id=*/0, /*mlayer_id=*/0, &plan,
                         &error)) {
    return false;
  }
  return true;
}

bool insert_grain_obu(const grain::GrainFramePlan& plan,
                      std::vector<std::uint8_t>& unit, std::string& error) {
  if (!plan.emit_model) return true;
  std::string grain_error;
  const std::vector<std::uint8_t> obu =
      grain::MakeFilmGrainObu(plan.obu, &grain_error);
  if (obu.empty()) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  std::vector<std::uint8_t> with_grain;
  if (!grain::InsertFilmGrainObu(unit, obu, &with_grain, &grain_error)) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  unit.swap(with_grain);
  return true;
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

struct TxPlan {
  int tx_size = 0;
  int adj_w = 0;
  int adj_h = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;  // adjusted-size coefficient array
  // §5.20.7.27 TCQ: the adjusted-size Quant[] values (base levels mapped by
  // Quant = 2*base - q0) that the §7.14.4 dequant consumes.  Empty when TCQ
  // did not run on this plan.
  std::vector<std::int32_t> dequant_levels;
  std::vector<std::uint8_t> recon;   // final coded-size samples
  // §5.20.8.2 intra IST decision (--video-ist).  0 = no secondary transform.
  int tx_type = 0;       // txv::TxType of the single (non-searched) block
  int sec_tx_type = 0;
  int stx_set = 0;

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
    int sec_tx_type = 0;
    int stx_set = 0;
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
  // §5.20.7.16 masked compound parameters (ignored for AVERAGE).
  int compound_type = 0;  // 0 = WEDGE, 1 = DIFFWTD
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
  // §5.20.7.6 / §7.13.3.14 CWP + implicit-mask decision (NEAR_NEARMV only).
  int cwp_index = 0;
  int cwp_weight = 8;  // CWP_EQUAL
  bool use_imp_mask = false;
  // This candidate's MV/DRL index came from a §7.12.2.7/.8 temporal entry
  // (instrumentation for the gate; never coded).
  bool from_temporal = false;
  // NEAR_NEARMV / optflow / refine-MV compound carrier.
  int compound_mode = 3;  // 0 = NEAR_NEARMV, 3 = GLOBAL_GLOBALMV
  inter::mc::Mv mv1{};
  int drl_index1 = 0;
  bool use_optflow = false;
  bool use_refinemv = false;};

// One decoded picture buffer slot.  The planes are MI-aligned (mi_cols*4 by
// mi_rows*4 luma) because MC clips taps to the padded reference extent.
struct RefSlot {
  bool valid = false;
  bool is_key = false;
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

// §7.7 dependency maps for the implicit reference mapping.  The M2 sequence
// headers either omit TLayerDependencyMap (default `ref_tlayer <=
// obu_tlayer_id`) or code it explicitly; MLayerDependencyMap is the default
// `ref_mlayer <= obu_mlayer_id`, and the M2 writers are single-mlayer.
inter::mv_stack_full::DependencyMaps implicit_dependency_maps(
    const VideoConfig& config, int tlayer_id) {
  inter::mv_stack_full::DependencyMaps deps;
  deps.mlayer_mask = ~std::uint64_t{0};
  const int layer = clip3(0, 3, tlayer_id);
  if (!config.tlayer_dependency_present_flag) {
    deps.tlayer_mask = (std::uint64_t{1} << (layer + 1)) - 1;
  } else {
    std::uint64_t mask = 0;
    for (int ref = 0; ref <= 3; ++ref) {
      if (config.tlayer_dependency[layer][ref]) {
        mask |= std::uint64_t{1} << ref;
      }
    }
    deps.tlayer_mask = mask;
  }
  return deps;
}

// Runs the §7.7 derivation over the §7.23 bank the encoder mirrors.  `width`
// and `height` are the current frame's size, which §7.7 checks with checkRes=1
// (both passes are identical while frame_size_override_flag = 0).
bool derive_implicit_fields(const inter::mv_stack_full::RefSlotBank& bank,
                            int order_hint, int order_hint_bits, int width,
                            int height, int tlayer_id,
                            const inter::mv_stack_full::DependencyMaps& deps,
                            inter::mv_stack_full::ImplicitRefMap* out,
                            std::string* error) {
  inter::mv_stack_full::RefMapParams params;
  params.num_ref_frames = bank.num_ref_frames;
  params.active_num_ref_frames =
      std::min(inter::mv_stack_full::kRefsPerFrameConst, bank.num_ref_frames);
  params.order_hint = order_hint;
  params.order_hint_bits = order_hint_bits;
  params.obu_mlayer_id = 0;
  params.obu_tlayer_id = tlayer_id;
  params.deps = deps;
  params.allowed_frames = ~std::uint32_t{0};
  params.is_bridge = false;
  params.check_res = true;
  params.frame_width = width;
  params.frame_height = height;
  params.slots = bank.slots;
  return inter::mv_stack_full::derive_implicit_ref_map(params, out, error);
}

// --video-trace line for the derived map.  The gate greps it to prove the
// stream actually uses the §7.7 mapping (variable NumTotalRefs).
void trace_implicit_map(const char* path, std::size_t frame, int num_refs,
                        const std::vector<int>& slots) {
  std::cerr << "  implicit ref map " << path << " frame=" << frame
            << " num_total_refs=" << num_refs << " slots=";
  for (int r = 0; r < num_refs; ++r) {
    std::cerr << (r == 0 ? "" : ",") << slots[static_cast<std::size_t>(r)];
  }
  std::cerr << "\n";
}

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
                      bool have_left, bool have_above, bool enable_ibp,
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
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  intra::IntraStatus status = intra::IntraStatus::kInvalidArgument;
  if (enable_ibp) {
    // §7.13.2.12: enable_ibp changes DC via the Ibp_Weights post-filter for
    // every non-4x4 block.  The decoder applies it whenever YMode is DC_PRED,
    // so the encoder's DC predictor must use the same path.
    ibp::PredictParams params;
    params.mode = intra::Mode::kDc;
    params.angle_delta = 0;
    params.enable_ibp = true;
    params.enable_edge_filter = false;
    status = ibp::predict_block(request, 8, params, out.data(),
                                static_cast<std::size_t>(w));
  } else {
    intra::PredictParams params;
    params.mode = intra::Mode::kDc;
    params.angle_delta = 0;
    params.enable_edge_filter = false;
    status = intra::predict_block(request, 8, params, out.data(),
                                  static_cast<std::size_t>(w));
  }
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

#if defined(AV2ENC_VIDEO_DELTAQ)
// §5.18.7.1 segmentation_params() / §5.18.7.8 delta_q_params() for one frame.
// The table is frame-level and identical on every frame of the sequence; the
// per-frame map lives in the tile payload.  Validates the plan before the
// header writer runs (the writer has no error channel).
bool build_video_frame_qp(const VideoOptions& options, int base_q_idx,
                          bool primary_ref_is_none, VideoFrameHeader& header,
                          std::string& error) {
  header.qp = VideoFrameQp{};
  header.qp.primary_ref_is_none = primary_ref_is_none;
  if (!options.enable_segmentation && !options.per_segment_q &&
      !options.enable_delta_q) {
    return true;
  }
  if (options.enable_segmentation || options.per_segment_q) {
    header.qp.segmentation.enabled = true;
    header.qp.segmentation.update_map = true;
    header.qp.segmentation.temporal_update = false;
    // --video-segmentation alone keeps ALT_Q = 0 so segment id 1 stays
    // representable (§5.20.5.8) without moving the quantizer;
    // --video-per-segment-q supplies the real value.
    const int delta =
        options.per_segment_q ? options.seg_q_delta : 0;
    std::string seg_error;
    if (!rc::set_segment_alt_q(header.qp.segmentation, /*segment_id=*/1, delta,
                               &seg_error)) {
      error = "segmentation: " + seg_error;
      return false;
    }
  }
  header.qp.delta_q.present = options.enable_delta_q;
  header.qp.delta_q.res = options.enable_delta_q ? options.delta_q_res : 0;
  deltaq::FrameQpHeader qp_header;
  qp_header.primary_ref_is_none = primary_ref_is_none;
  qp_header.using_qmatrix = options.enable_qm;
  deltaq::FrameSegmentation derived;
  if (!deltaq::derive_frame_segmentation(header.qp.segmentation,
                                         header.qp.delta_q, base_q_idx,
                                         qp_header, derived, &error)) {
    error = "segmentation: " + error;
    return false;
  }
  if (derived.has_lossless_segment) {
    error = "segmentation: a segment qindex reaches 0 (lossless); the video "
            "transform path does not implement lossless";
    return false;
  }
  return true;
}
#endif  // AV2ENC_VIDEO_DELTAQ

bool compute_tx(const VideoInputFrame& frame, int plane, int x0, int y0, int cw,
                int ch, const std::vector<std::uint8_t>& pred, int quant_step,
                TxPlan& plan, std::string& error,
                const qm::QmFrameState* qm_state = nullptr,
                bool allow_tcq = false, int qctx_slice = 0,
                bool is_inter = false) {
  auto params_opt = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "unsupported transform size";
    return false;
  }
  if (quant_step != txv::kQuantStep && qm_state == nullptr) {
    // With QM the per-coefficient q2 replaces the scalar step; the forward
    // expression below uses the unscaled quant_factor.
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
  qm::QmDequantParams qp;
  std::vector<std::int32_t> q2;
  if (qm_state != nullptr) {
    // §7.14.4 QM: q2 = Round2(q * m, 5) per coefficient.  The forward target
    // scales the scalar qf by q / q2 so the returned levels reconstruct to
    // the same residual under the QM dequantizer (still_encoder.cpp).
    qp.state = qm_state;
    qp.tx_size = params.tx_size;
    qp.plane = plane;
    qp.tx_type = params.tx_type;
    qp.segment_id = 0;
    qp.dc_quant = quant_step;
    qp.ac_quant = quant_step;
    qp.bit_depth = 8;
    q2.assign(coeff.size(), 0);
    if (!qm::build_q2_matrix(qp, q2.data())) {
      error = "qm q2 matrix build failed";
      return false;
    }
    txv::forward_block(residual_adj.data(), params, coeff.data());
    for (std::size_t i = 0; i < coeff.size(); ++i) {
      const double factor = static_cast<double>(params.quant_factor) *
                            static_cast<double>(quant_step) /
                            static_cast<double>(q2[i]);
      plan.levels[i] = static_cast<std::int32_t>(std::lround(coeff[i] * factor));
    }
  } else {
    txv::forward_transform(residual_adj.data(), params, coeff.data(),
                           plan.levels.data());
  }

  const int area = params.w * params.h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  syntax::build_scan(params.w, params.h, scan);

  // §5.20.7.27 trellis-coded quantisation.  Luma-only and TX_CLASS_2D-only;
  // QM changes the per-coefficient quantiser and is rejected with TCQ before
  // this point.  The DP returns the syntax base levels (what coeff_base/br and
  // the read_quant escape code) and the mapped Quant values §7.14.4
  // dequantises after the extra dqDenom shift.
  const bool use_tcq =
      allow_tcq && plane == 0 && params.tx_type <= 9 && qm_state == nullptr;
  if (use_tcq) {
    std::vector<av2enc::tcq::Coefficient> input(
        static_cast<std::size_t>(area));
    std::vector<std::int32_t> reference(static_cast<std::size_t>(area));
    for (int c = 0; c < area; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / params.w;
      const int col = pos % params.w;
      input[static_cast<std::size_t>(c)].original =
          coeff[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].level_context = 0;
      input[static_cast<std::size_t>(c)].pre_max = (row + col) < 4 ? 8 : 6;
      // Frozen neighbour/escape reference: the scalar level at half the
      // syntax step (the same approximation the still TCQ path uses).
      const std::int32_t scalar = plan.levels[static_cast<std::size_t>(pos)];
      const std::int32_t magnitude = std::abs(scalar);
      const std::int32_t base = (magnitude + 1) / 2;
      reference[static_cast<std::size_t>(c)] = scalar < 0 ? -base : base;
    }
    av2enc::tcq::av2::BlockInfo info;
    info.plane = 0;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = 0;
    info.qctx = qctx_slice;
    info.is_inter = is_inter;
    info.tx_class_2d = true;
    av2enc::tcq::av2::Model model;
    std::string tcq_error;
    if (!model.init(info, reference, &tcq_error)) {
      error = "tcq model init failed: " + tcq_error;
      return false;
    }
    av2enc::tcq::Config tcq_config;
    tcq_config.quant_step = 1.0 / params.quant_factor;
    tcq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
        tcq_config.quant_step, /*sse_scale=*/0.125);
    tcq_config.max_coefficients = area;
    av2enc::tcq::Result tcq_result;
    if (!av2enc::tcq::quantize(input, tcq_config, model.rate_model(),
                               tcq_result, &tcq_error)) {
      error = "tcq quantize failed: " + tcq_error;
      return false;
    }
    std::fill(plan.levels.begin(), plan.levels.end(), 0);
    plan.dequant_levels.assign(static_cast<std::size_t>(area), 0);
    for (int c = 0; c < tcq_result.eob; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      plan.levels[static_cast<std::size_t>(pos)] =
          tcq_result.base_levels[static_cast<std::size_t>(c)];
      plan.dequant_levels[static_cast<std::size_t>(pos)] =
          tcq_result.quant_levels[static_cast<std::size_t>(c)];
    }
  } else {
    plan.dequant_levels.clear();
  }

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
  if (qm_state != nullptr) {
    const int n = params.w * params.h;
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc = plan.levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * q2[static_cast<std::size_t>(i)]) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          params.dq_denom;
      const std::int32_t value = qc < 0 ? -dq : dq;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, value));
    }
    txv::inverse_block(dequant.data(), params, recon_residual.data());
  } else if (!plan.dequant_levels.empty()) {
    // §7.14.4 TCQ: dequantise the mapped Quant[] values with the extra
    // dqDenom shift (`shift += 1` for luma TX_CLASS_2D), then inverse.
    const int n = params.w * params.h;
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc =
          plan.dequant_levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant_step) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          (params.dq_denom << 1);
      const std::int32_t value = qc < 0 ? -dq : dq;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, value));
    }
    txv::inverse_block(dequant.data(), params, recon_residual.data());
  } else if (quant_step == txv::kQuantStep) {
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
  entry.sec_tx_type[0] = static_cast<std::uint8_t>(plan.sec_tx_type);
  entry.most_probable_stx_set[0] =
      static_cast<std::uint8_t>(plan.stx_set);
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
    entry.tx_part_sec_tx_type[i] =
        static_cast<std::uint8_t>(part.sec_tx_type);
    entry.tx_part_stx_set[i] = static_cast<std::uint8_t>(part.stx_set);
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
                    txtsearch::Mode mode, double lambda, bool allow_idtx,
                    int intra_mapped_mode, bool allow_tcq, int qctx_slice,
                    bool enable_inter_ddt, bool search_partition, bool trace,
                    TxPlan& plan, std::string& error) {
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
  options.allow_idtx = allow_idtx;
  options.intra_mapped_mode = intra_mapped_mode;
  options.tcq = allow_tcq;
  options.tcq_qctx = qctx_slice;
  options.enable_inter_ddt = enable_inter_ddt;
  // §5.20.7.24 predicts per *transform block*, in decode order, and each part's
  // reconstruction feeds the next part's prediction.  A fixed block prediction
  // is only equivalent when the transform block covers the coding block, so the
  // partition search is inter-only: MC with one MV is position-independent, and
  // the decoder's per-part prediction equals the block prediction at the same
  // coordinates.  Intra keeps TX_PARTITION_NONE and searches the TX type only;
  // the full intra partition search needs the per-part prediction/recon
  // feedback the device module's fixed-predictor contract does not carry.
  // Under TX_MODE_LARGEST the bitstream codes no partition, so the DDT type
  // search must also restrict itself to TX_PARTITION_NONE.
  options.search_partition = search_partition;
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
    part.sec_tx_type = 0;
    part.stx_set = 0;
    if (trace) {
      std::cerr << "  video tx: leaf " << leaf.coded_w << "x" << leaf.coded_h
                << " type=" << leaf.tx_type << " at " << x0 + leaf.x << ","
                << y0 + leaf.y << "\n";
    }
    if (trace && leaf.tx_type == 9) {
      std::cerr << "  video tx: idtx leaf " << leaf.coded_w << "x"
                << leaf.coded_h << " at " << x0 + leaf.x << ","
                << y0 + leaf.y << "\n";
    }
    if (trace && enable_inter_ddt) {
      const auto tx_params = txv::make_tx_params(
          static_cast<txv::TxSize>(leaf.tx_size),
          static_cast<txv::TxType>(leaf.tx_type));
      if (tx_params.has_value()) {
        av2enc::video::ddt::Context ddt_ctx;
        ddt_ctx.enable_inter_ddt = true;
        ddt_ctx.is_inter = true;
        if (av2enc::video::ddt::plan_has_ddt(*tx_params, ddt_ctx)) {
          std::cerr << "  video tx: ddt leaf " << leaf.coded_w << "x"
                    << leaf.coded_h << " type=" << leaf.tx_type << " at "
                    << x0 + leaf.x << "," << y0 + leaf.y << "\n";
        }
      }
    }
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

// §7.13.2.1/.7/.8 MRL prediction for a directional luma candidate.  The M1
// still encoder's predict_luma_mrl() is the reference; the availability inputs
// come from the shared intra search.
bool predict_luma_mrl(const Plane& recon,
                      const intra_search::Availability& avail, int x0, int y0,
                      int w, int h, bool above_smooth, bool left_smooth,
                      int y_mode, int angle_delta, int mrl_index,
                      bool mrl_sec_index, int sb_height,
                      std::vector<std::uint8_t>& out, std::string& error) {
  mrl::Params params;
  if (!mrl_stream::make_predict_params(
          y_mode, angle_delta, mrl_index, mrl_sec_index,
          /*sequence_enable_edge_filter=*/true, above_smooth, left_smooth,
          &params, w, h)) {
    error = "MRL prediction parameters rejected";
    return false;
  }
  mrl::Block block;
  block.recon = intra::PlaneView{recon.samples.data(), recon.stride};
  block.x = x0;
  block.y = y0;
  block.w = w;
  block.h = h;
  block.max_x = avail.max_x;
  block.max_y = avail.max_y;
  block.have_left = avail.have_left != 0;
  block.have_above = avail.have_above != 0;
  block.num4_above_right = avail.num4_above_right;
  block.num4_below_left = avail.num4_below_left;
  block.plane = 0;
  block.sb_height = sb_height;
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  if (mrl::predict_block(block, 8, params, out.data(),
                         static_cast<std::size_t>(w)) !=
      mrl::MrlStatus::kOk) {
    error = "mrl::predict_block failed";
    return false;
  }
  return true;
}

// §7.15.3 luma IST decision for one intra leaf.  Mirrors the M1 still
// encoder's compute_tx() block: the projection is encoder-side analysis, and
// the cost is reconstructed SSE + lambda * §9.3 symbol bits.  Only a single
// transform block is considered; with a TX partition the search plan wins.
bool refine_luma_ist(const VideoInputFrame& frame, int ax0, int ay0, int cw,
                     int ch, const std::vector<std::uint8_t>& pred,
                     int quant_step, int y_mode, int angle_delta,
                     int mrl_index, TxPlan& plan, std::string& error) {
  int tx_size = plan.tx_size;
  int tx_type = plan.tx_type;
  int coded_w = cw;
  int coded_h = ch;
  int eob = plan.eob;
  std::vector<std::int32_t> full;
  TxPlan::Part* part = nullptr;
  if (plan.searched) {
    if (plan.parts.size() != 1) return true;
    part = &plan.parts[0];
    if (part->coded_w != cw || part->coded_h != ch) return true;
    tx_size = part->tx_size;
    tx_type = part->tx_type;
    eob = part->eob;
    const auto params_opt =
        txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                            static_cast<txv::TxType>(tx_type));
    if (!params_opt.has_value()) return true;
    const int area = params_opt->w * params_opt->h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(params_opt->w, params_opt->h, scan);
    full.assign(static_cast<std::size_t>(area), 0);
    for (int c = 0; c < eob; ++c) {
      full[scan[static_cast<std::size_t>(c)]] = part->levels[c];
    }
    plan.tx_size = tx_size;
  } else {
    full = plan.levels;
  }

  const auto params_opt =
      txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                          static_cast<txv::TxType>(tx_type));
  if (!params_opt.has_value()) return true;
  const txv::TxParams& params = *params_opt;
  if (params.coded_w != coded_w || params.coded_h != coded_h) return true;
  const int area = params.w * params.h;
  if (static_cast<int>(full.size()) < area) {
    error = "IST: coefficient array is smaller than the transform area";
    return false;
  }
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  syntax::build_scan(params.w, params.h, scan);
  const auto eob_of = [&](const std::vector<std::int32_t>& lv) {
    for (int c = area - 1; c >= 0; --c) {
      if (lv[scan[static_cast<std::size_t>(c)]] != 0) return c + 1;
    }
    return 0;
  };

  txsec::SecondaryTransformConfig ist_cfg;
  ist_cfg.enable_intra_ist = true;
  txsec::SecondarySyntax stx;
  stx.plane_tx_type = static_cast<txv::TxType>(tx_type);
  stx.coded_w = coded_w;
  stx.coded_h = coded_h;
  stx.y_mode = y_mode;
  stx.angle_delta = angle_delta;
  stx.mrl_index = mrl_index;
  stx.is_inter = false;
  stx.lossless = false;
  stx.bit_depth = 8;
  if (eob < 2 || !txsec::secondary_syntax_applicable(0, eob, stx, ist_cfg)) {
    return true;
  }
  txsec::SecondaryCandidate candidates[32];
  const int count =
      txsec::enumerate_secondary_candidates(stx, ist_cfg, eob, candidates, 32);
  if (count <= 0) return true;

  const auto reconstruct = [&](const std::vector<std::int32_t>& lv, int ev,
                               int sec, int set, std::vector<std::int16_t>* rr,
                               bool* ok) -> bool {
    txsec::SecondarySyntax cs = stx;
    cs.sec_tx_type = sec;
    cs.most_probable_stx_set = set;
    if (!txsec::secondary_syntax_valid(0, ev, cs, ist_cfg)) return false;
    std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
    for (int i = 0; i < area; ++i) {
      const std::int32_t qc = lv[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant_step) & 0xFFFFFF;
      const std::int32_t dq =
          txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
          params.dq_denom;
      dequant[static_cast<std::size_t>(i)] =
          std::max(-(1 << 15), std::min((1 << 15) - 1, qc < 0 ? -dq : dq));
    }
    const txsec::SecondaryStatus status = txsec::apply_inverse_secondary(
        params, 0, ev, cs, ist_cfg, dequant.data(), dequant.size());
    if (status != txsec::SecondaryStatus::kApplied) return false;
    rr->assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
    txv::inverse_block(dequant.data(), params, rr->data());
    *ok = true;
    return true;
  };
  const auto recon_of = [&](const std::vector<std::int16_t>& rr) {
    std::vector<std::uint8_t> recon(
        static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int p = pred[static_cast<std::size_t>(y) * cw + x];
        const int r = rr[static_cast<std::size_t>(y) * cw + x];
        recon[static_cast<std::size_t>(y) * cw + x] =
            static_cast<std::uint8_t>(clip1(p + r));
      }
    }
    return recon;
  };

  double best_cost = static_cast<double>(
      sse_against_source(frame, 0, ax0, ay0, cw, ch, plan.recon));
  std::vector<std::int32_t> best_levels;
  std::vector<std::uint8_t> best_recon;
  int best_eob = eob;
  int best_sec = 0;
  int best_set = 0;
  const double lambda =
      av2enc::rdo::lambda_sse_from_quant_step(1.0 / params.quant_factor, 0.125);
  for (int i = 0; i < count; ++i) {
    const txsec::SecondaryCandidate& candidate = candidates[i];
    std::vector<std::int32_t> lv = full;
    if (!txsec::project_ist_levels(params, stx.plane_tx_type, candidate.kernel,
                                   candidate.sec_tx_type, candidate.transpose,
                                   lv.data(), lv.size())) {
      continue;
    }
    const int ev = eob_of(lv);
    if (ev < 2) continue;
    bool ok = false;
    std::vector<std::int16_t> rr;
    if (!reconstruct(lv, ev, candidate.sec_tx_type,
                     candidate.most_probable_stx_set, &rr, &ok) ||
        !ok) {
      continue;
    }
    std::vector<std::uint8_t> recon = recon_of(rr);
    const double sse = static_cast<double>(
        sse_against_source(frame, 0, ax0, ay0, cw, ch, recon));
    const double bits =
        txsec::sec_tx_type_bits(candidate.sec_tx_type, params.tx_size, false) +
        txsec::most_probable_stx_set_bits(candidate.most_probable_stx_set,
                                          stx.plane_tx_type, coded_w, coded_h);
    const double cost = sse + lambda * bits;
    if (cost < best_cost) {
      best_cost = cost;
      best_levels = std::move(lv);
      best_recon = std::move(recon);
      best_eob = ev;
      best_sec = candidate.sec_tx_type;
      best_set = candidate.most_probable_stx_set;
    }
  }
  if (best_sec == 0) return true;

  if (part != nullptr) {
    part->eob = best_eob;
    part->sec_tx_type = best_sec;
    part->stx_set = best_set;
    part->levels.resize(static_cast<std::size_t>(best_eob));
    for (int c = 0; c < best_eob; ++c) {
      part->levels[static_cast<std::size_t>(c)] =
          best_levels[scan[static_cast<std::size_t>(c)]];
    }
    plan.eob = best_eob;
    plan.tx_type = tx_type;
  } else {
    plan.levels = std::move(best_levels);
    plan.eob = best_eob;
    plan.sec_tx_type = best_sec;
    plan.stx_set = best_set;
  }
  plan.recon = std::move(best_recon);
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

// §5.4.8 enable_cctx / §7.14.6.  Chroma U/V are quantised jointly: each of the
// seven rotation types is enumerated, and the exact decoder reconstruction
// (integer §7.14.4 dequant, §7.14.6 rotation, inverse transform) picks the
// winner by SSE + lambda * cctx_type bits.  Mirrors the still path
// (still_encoder.cpp compute_chroma_cctx) without RDOQ.
bool compute_chroma_cctx(const VideoInputFrame& frame, int x0, int y0, int cw,
                         int ch, const std::vector<std::uint8_t>& u_pred,
                         const std::vector<std::uint8_t>& v_pred,
                         int quant_step, bool is_inter, TxPlan& u_plan,
                         TxPlan& v_plan, int* cctx_type_out,
                         std::string& error) {
  auto params_opt = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "unsupported chroma transform size for CCTX";
    return false;
  }
  if (quant_step != txv::kQuantStep) {
    params_opt->quant_factor *=
        static_cast<double>(txv::kQuantStep) / static_cast<double>(quant_step);
  }
  const txv::TxParams params = *params_opt;
  const int area = params.w * params.h;
  const int coded_pels = cw * ch;

  const auto residual_adj = [&](int plane,
                                const std::vector<std::uint8_t>& pred) {
    std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int src = source_sample(frame, plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        coded[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x)] =
            static_cast<std::int16_t>(src - p);
      }
    }
    std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
    txv::downsample_residual(coded.data(), params, adj.data());
    return adj;
  };
  const std::vector<std::int16_t> u_adj = residual_adj(1, u_pred);
  const std::vector<std::int16_t> v_adj = residual_adj(2, v_pred);
  std::vector<double> u_coeff(static_cast<std::size_t>(area));
  std::vector<double> v_coeff(static_cast<std::size_t>(area));
  txv::forward_block(u_adj.data(), params, u_coeff.data());
  txv::forward_block(v_adj.data(), params, v_coeff.data());

  av2enc::tcq::CctxConfig config;
  config.enabled = true;
  config.quant_step_u = 1.0 / params.quant_factor;
  config.quant_step_v = config.quant_step_u;
  config.bitdepth = 8;
  std::vector<av2enc::tcq::CctxResult> candidates;
  std::string cctx_error;
  if (!av2enc::tcq::cctx_enumerate_scalar(u_coeff, v_coeff, params.w,
                                          params.h, config, &candidates,
                                          &cctx_error)) {
    error = "cctx enumerate failed: " + cctx_error;
    return false;
  }

  // §7.14.4 dequant for one chroma plane (no QM/delta-Q/TCQ shift).
  const auto dequant_plane = [&](const std::vector<std::int32_t>& levels,
                                 std::vector<std::int32_t>& dequant) {
    dequant.assign(static_cast<std::size_t>(area), 0);
    for (int i = 0; i < area; ++i) {
      const std::int32_t qc = levels[static_cast<std::size_t>(i)];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant_step) & 0xFFFFFF;
      const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                          txv::kQuantTableBits) /
                              params.dq_denom;
      dequant[static_cast<std::size_t>(i)] = std::max(
          -(1 << 15), std::min((1 << 15) - 1, qc < 0 ? -dq : dq));
    }
  };
  const auto reconstruct = [&](const av2enc::tcq::CctxResult& candidate,
                               std::vector<std::int16_t>& u_res,
                               std::vector<std::int16_t>& v_res) {
    std::vector<std::int32_t> du;
    std::vector<std::int32_t> dv;
    dequant_plane(candidate.u_levels, du);
    dequant_plane(candidate.v_levels, dv);
    std::vector<std::int32_t> pu(static_cast<std::size_t>(area));
    std::vector<std::int32_t> pv(static_cast<std::size_t>(area));
    for (int i = 0; i < area; ++i) {
      const av2enc::tcq::CctxPair pair = av2enc::tcq::cctx_dequant_pair(
          du[static_cast<std::size_t>(i)], dv[static_cast<std::size_t>(i)],
          candidate.type, 8);
      pu[static_cast<std::size_t>(i)] = pair.u;
      pv[static_cast<std::size_t>(i)] = pair.v;
    }
    u_res.assign(static_cast<std::size_t>(coded_pels), 0);
    v_res.assign(static_cast<std::size_t>(coded_pels), 0);
    txv::inverse_block(pu.data(), params, u_res.data());
    txv::inverse_block(pv.data(), params, v_res.data());
  };
  const auto distortion = [&](const std::vector<std::int16_t>& u_res,
                              const std::vector<std::int16_t>& v_res) {
    double sse = 0.0;
    for (int plane = 1; plane <= 2; ++plane) {
      const std::vector<std::uint8_t>& pred = plane == 1 ? u_pred : v_pred;
      const std::vector<std::int16_t>& res = plane == 1 ? u_res : v_res;
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const std::size_t index =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x);
          const int value = clip1(static_cast<int>(pred[index]) +
                                  static_cast<int>(res[index]));
          const double d =
              static_cast<double>(source_sample(frame, plane, x0 + x, y0 + y)) -
              static_cast<double>(value);
          sse += d * d;
        }
      }
    }
    return sse;
  };

  const double lambda = av2enc::rdo::lambda_sse_from_quant_step(
      1.0 / params.quant_factor, /*sse_scale=*/0.125);
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  syntax::build_scan(params.w, params.h, scan);
  const auto scan_eob = [&](const std::vector<std::int32_t>& levels) {
    for (int c = area - 1; c >= 0; --c) {
      if (levels[static_cast<std::size_t>(
              scan[static_cast<std::size_t>(c)])] != 0) {
        return c + 1;
      }
    }
    return 0;
  };

  int best = -1;
  double best_cost = std::numeric_limits<double>::infinity();
  int best_eob_u = 0;
  int best_eob_v = 0;
  std::vector<std::int16_t> best_u_res;
  std::vector<std::int16_t> best_v_res;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const av2enc::tcq::CctxResult& candidate = candidates[i];
    const int eob_u = scan_eob(candidate.u_levels);
    // The decoder takes the all_zero branch (cctx_type inferred 0, no symbol)
    // when U eob is 0; an intra block with U eob == 1 also infers 0.
    if (candidate.type != av2enc::tcq::CctxType::kNone) {
      if (eob_u == 0) continue;
      if (!is_inter && eob_u == 1) continue;
    }
    std::vector<std::int16_t> u_res;
    std::vector<std::int16_t> v_res;
    reconstruct(candidate, u_res, v_res);
    const double sse = distortion(u_res, v_res);
    // The symbol is present for every non-zero U block except an intra block
    // with eob == 1; CCTX_NONE pays it too (the decoder always reads the
    // symbol for an eligible block).
    const double bits =
        (eob_u > 0 && (is_inter || eob_u != 1))
            ? av2enc::tcq::cctx_type_bits(candidate.type)
            : 0.0;
    const double cost = sse + lambda * bits;
    if (cost < best_cost) {
      best_cost = cost;
      best = static_cast<int>(i);
      best_eob_u = eob_u;
      best_eob_v = scan_eob(candidate.v_levels);
      best_u_res = std::move(u_res);
      best_v_res = std::move(v_res);
    }
  }
  if (best < 0) {
    error = "no signallable CCTX candidate";
    return false;
  }
  const av2enc::tcq::CctxResult& chosen =
      candidates[static_cast<std::size_t>(best)];

  const auto fill_plan = [&](TxPlan& plan, const std::vector<std::int32_t>& levels,
                             int eob, const std::vector<std::uint8_t>& pred,
                             const std::vector<std::int16_t>& res) {
    plan.tx_size = params.tx_size;
    plan.adj_w = params.w;
    plan.adj_h = params.h;
    plan.eob = eob;
    plan.levels = levels;
    plan.dequant_levels.clear();
    plan.recon.resize(static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const std::size_t index =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
            static_cast<std::size_t>(x);
        plan.recon[index] = static_cast<std::uint8_t>(
            clip1(static_cast<int>(pred[index]) +
                  static_cast<int>(res[index])));
      }
    }
  };
  fill_plan(u_plan, chosen.u_levels, best_eob_u, u_pred, best_u_res);
  fill_plan(v_plan, chosen.v_levels, best_eob_v, v_pred, best_v_res);
  if (cctx_type_out != nullptr) {
    *cctx_type_out = static_cast<int>(chosen.type);
  }
  return true;
}

// §5.20.3.1 ChromaMi bookkeeping for a sub-8 luma leaf.  `size` is the
// ChromaMiSize in luma 4x4 units; `mi_row`/`mi_col` its luma MI origin.  The
// first leaf (in decode order) whose incoming HasChroma is still 1 codes the
// whole inherited chroma block; its siblings code luma only.
struct ChromaInfo {
  bool valid = false;
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
};

struct TemporalFrameContext {
  bool use_ref_frame_mvs = false;
  int order_hint = 0;
  int num_past_refs = 0;
  int num_future_refs = 0;
  int closest_past = -1;
  int closest_future = -1;
  const temporal::MotionField* field = nullptr;
};

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
                    std::array<int, 7> ref_order_hints = {},
                    bool primary_ref_is_none = true,
                    const TemporalFrameContext* temporal = nullptr)
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
#else
    (void)primary_ref_is_none;
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
    current_qindex_ = static_cast<int>(options_.base_q_idx);
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
    // §6.19.2.3 BlockDecoded maps (reset at every superblock root).
    decoded_[0].reset(mi_cols_, mi_rows_);
    decoded_[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    decoded_[2].reset(mi_cols_ / 2, mi_rows_ / 2);
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


#if defined(AV2ENC_VIDEO_DELTAQ)
  // ---------------------------------------------------------------------
  // §5.18.7.1/.8 per-superblock segmentation / delta-Q planning.
  // ---------------------------------------------------------------------
  int qp_sb_mi() const { return partition::sb_size_mi(geometry_); }
  int qp_sb_cols() const { return (mi_cols_ + qp_sb_mi() - 1) / qp_sb_mi(); }
  int qp_sb_rows() const { return (mi_rows_ + qp_sb_mi() - 1) / qp_sb_mi(); }
  int qp_sb_count() const { return qp_sb_cols() * qp_sb_rows(); }
  int base_q_index() const { return static_cast<int>(options_.base_q_idx); }
  bool qp_tools_active() const {
    return options_.enable_segmentation || options_.enable_delta_q ||
           options_.per_segment_q;
  }

  // Deterministic per-superblock split, identical to the still
  // --still-per-segment-q planner: segment 1 is every superblock whose luma
  // variance is <= the frame mean variance.  --video-delta-q instead uses a
  // raster-parity staircase for its target qindex.
  bool build_sb_maps(std::string& error) {
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
    for (int i = 0; i < count; ++i) {
      const bool low =
          variance[static_cast<std::size_t>(i)] <= mean;
      sb_segment_[static_cast<std::size_t>(i)] = low ? 1 : 0;
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

  bool build_segmentation(std::string& error) {
    seg_params_ = rc::SegmentationParams{};
    if (!seg_enabled_) return true;
    seg_params_.enabled = true;
    // The frame header codes update_map / temporal_update only when the derived
    // primary reference is not NONE; both values are the same either way.
    seg_params_.update_map = true;
    seg_params_.temporal_update = false;
    // --video-segmentation alone signals ALT_Q = 0 for segment 1 so both ids
    // are representable (§5.20.5.8 neg_interleave covers 0..LastActiveSegId)
    // while the quantizer stays at base_q_idx.  --video-per-segment-q supplies
    // the real delta.
    const int delta =
        options_.per_segment_q ? options_.seg_q_delta : 0;
    if (!rc::set_segment_alt_q(seg_params_, /*segment_id=*/1, delta,
                               &error)) {
      return false;
    }
    return true;
  }

  bool prepare_qp_plan(std::string& error) {
    seg_enabled_ = options_.enable_segmentation || options_.per_segment_q;
    delta_q_enabled_ = options_.enable_delta_q;
    if (!seg_enabled_ && !delta_q_enabled_) return true;
    if (!build_sb_maps(error)) return false;
    if (!build_segmentation(error)) return false;
    dq_params_ = rc::DeltaQParams{};
    dq_params_.present = delta_q_enabled_;
    dq_params_.res = delta_q_enabled_ ? options_.delta_q_res : 0;
    deltaq::FrameQpHeader header;
    header.primary_ref_is_none = true;
    header.enable_ext_seg = false;
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
            /*tile_mi_row_end=*/mi_rows_, /*enable_ext_seg=*/false,
            deltaq::max_segments(false), {}, &error)) {
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

  // Builds the per-superblock delta-Q steps from sb_target_q_.  A superblock
  // whose first (and only) leaf is a full-SB inter skip cannot code
  // read_delta_qindex() (§5.20.5.11: MiSize == SbSize && skip_flag), so the
  // plan must leave it at the carried qindex.  That target fix-up can shift
  // later steps, so re-run rc::plan_delta_q() until no target changes.
  bool plan_delta_q_steps(std::vector<int>& actual, std::string& error) {
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
#endif  // AV2ENC_VIDEO_DELTAQ

  // §5.4.3 enable_sdp: decide the intra keyframe as per-64x64 luma and chroma
  // partition trees.  The luma tree uses the same --video-min-block /
  // --video-max-depth controls and rect-only family as the shared tree.  The
  // chroma tree is pinned to a single PARTITION_NONE block: the reference
  // model can search it and the writer can signal it, but the pinned dav2d
  // disagrees with the model's chroma-plane partition contexts inside a
  // decoded chroma tree, so no product stream produces one.
  bool decide_sdp(std::string& error) {
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

    blocks_.clear();
    levels_.clear();
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
    for (const sdpp::Node& sb : sdp_frame_.superblocks) {
      if (!collect_sdp_node(sb, error)) return false;
    }
    return true;
  }

  // Decision + reconstruction phase (no syntax, no filters).  Kept
  // separate so a multi-tile frame can run one frame-level filter pass
  // after every tile has been reconstructed.
  bool decide(std::string& error) {
    if (keyframe_ && options_.enable_sdp) {
      return decide_sdp(error);
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

    const double lambda =
        0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;
    const auto cost_fn = [this, signal_cost](
                             const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      // The leaf coder handles coding blocks up to 64x64.  A larger node must
      // split (the 128/256 root and its rectangular children); the spec never
      // implies NONE at those sizes, so the solver always has a partition.
      const int bw = partition::block_width(info.size);
      const int bh = partition::block_height(info.size);
      costs.leaf = (bw <= 64 && bh <= 64)
                       ? static_cast<double>(block_variance_cost_at(info))
                       : partition::kInfiniteCost;
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };

    std::vector<PartitionNode> base_superblocks;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    std::vector<pe::PartitionNode> ext_superblocks;
    if (options_.ext_partitions) {
      pe::Tools tools;
      tools.enable_ext_partitions = true;
      tools.enable_uneven_4way_partitions = true;
      pe::DecisionOptions eopt;
      eopt.min_block_size_px = keyframe_ ? options_.min_block_size
                                         : std::max(8, options_.min_block_size);
      eopt.max_depth = options_.max_partition_depth;
      eopt.allow_rect = true;
      eopt.allow_split = sb_size_ > 64;
      eopt.allow_3way = true;
      eopt.allow_uneven4 = true;
      eopt.tie_break = pe::TieBreakOrder::kRectCompatOrder;
      if (options_.force_single_block) {
        eopt.max_depth = 8;
        eopt.min_block_size_px = 64;
      }
      const auto ext_cost_fn = [this, signal_cost](
                                   const pe::NodeInfo& info) {
        pe::NodeCosts costs;
        const int bw = partition::block_width(info.size);
        const int bh = partition::block_height(info.size);
        partition::PartitionNodeInfo binfo;
        binfo.mi_row = info.mi_row;
        binfo.mi_col = info.mi_col;
        binfo.size = static_cast<partition::BlockSize>(info.size);
        costs.leaf = (bw <= 64 && bh <= 64)
                         ? static_cast<double>(block_variance_cost_at(binfo))
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
    }

    const auto collect_superblocks = [&]() -> bool {
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
      if (options_.ext_partitions) {
        for (const pe::PartitionNode& sb : ext_superblocks) {
          if (!collect_node_ext(sb, error)) return false;
        }
      } else
#endif
      {
        for (const PartitionNode& sb : base_superblocks) {
          if (!collect_node(sb, error)) return false;
        }
      }
      return true;
    };

#if defined(AV2ENC_VIDEO_DELTAQ)
    if (!prepare_qp_plan(error)) return false;
    const auto collect_frame = [&]() -> bool {
      blocks_.clear();
      levels_.clear();
      palettes_.clear();
      if (palette_enabled_) palette_state_.reset(mi_cols_, mi_rows_);
      // A delta-Q fixpoint runs this walk more than once.  The merge stack
      // reads the above-right / below-left MI grid, so stale entries from the
      // previous pass would make the encoder's stack differ from the writer's.
      // Re-initialise the per-4x4 decision grids to their constructor state.
      mv_grid_.assign(mv_grid_.size(), inter::mc::MiInfo{});
      intra_joint_.assign(intra_joint_.size(), 0);
      for (int plane = 0; plane < 3; ++plane) {
        std::fill(recon_[plane].samples.begin(), recon_[plane].samples.end(),
                  static_cast<std::uint8_t>(0));
      }
      inter_blocks_ = 0;
      intra_blocks_ = 0;
      if (seg_enabled_) {
        deltaq::clear_above_context(enc_seg_grid_);
        deltaq::clear_left_context(enc_seg_grid_);
      }
      return collect_superblocks();
    };
    if (qp_tools_active()) {
      if (delta_q_enabled_) {
        std::vector<int> effective(
            static_cast<std::size_t>(qp_sb_count()), base_q_index());
        bool converged = false;
        for (int pass = 0; pass < 8; ++pass) {
          sb_qindex_ = effective;
          if (!collect_frame()) return false;
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
          if (!collect_frame()) return false;
          std::vector<int> actual;
          if (!plan_delta_q_steps(actual, error)) return false;
          if (actual != sb_qindex_) {
            error = "delta-Q plan: neutral fallback did not produce a zero "
                    "ramp";
            return false;
          }
        }
      } else if (!collect_frame()) {
        return false;
      }
      return true;
    }
    return collect_frame();
#else
    blocks_.clear();
    levels_.clear();
    palettes_.clear();
    if (palette_enabled_) palette_state_.reset(mi_cols_, mi_rows_);
    return collect_superblocks();
#endif
  }

  // Runs the §7.18-§7.20 decision + pixel stage on this encoder's
  // reconstruction, filling filter_plan_.  Single-tile path only.
  bool apply_loop_filters(std::string& error) {
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
      {
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
          // §7.18.1 Skips[]: an inter skip leaf and an IBC skip leaf both code
          // skip_flag = 1; with CdefOnSkipTxfm DISABLED the decoder leaves a
          // fully skipped 8x8 block unfiltered.
          block.skip = (leaf.is_inter && leaf.skip_flag != 0) ||
                       (leaf.flags & syntax::kFlagIsIbc) != 0;
          block.luma_eob_zero = leaf.eob[0] == 0;
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

    return true;
  }

  // §7.17 deblocking of this encoder's reconstruction.  Runs after decide()
  // and before the frame-level filter pass / reference update.
  bool deblock(std::string& error) {
    if (!options_.enable_deblock) return true;
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
                                          &error, sub_pus_.data(),
                                          static_cast<int>(sub_pus_.size()))) {
        return false;
      }
      // §5.18.2 / §7.17.2: when the frame codes allow_df_sub_pu = 1 the
      // deblocker filters the §7.13.3.15 sub-PU edges.  The grid already
      // carries SubPuSize/SubPuColBase/SubPuRowBase for every leaf.
      grid.options.allow_df_sub_pu = options_.enable_df_sub_pu;
      std::uint8_t* data[3] = {recon_[0].samples.data(),
                               recon_[1].samples.data(),
                               recon_[2].samples.data()};
      const int stride[3] = {recon_[0].stride, recon_[1].stride,
                             recon_[2].stride};
      if (!post::apply_video_deblock_host(grid, data, stride, &error)) {
        return false;
      }
    }
    return true;
  }

  // Syntax phase.  `filters` is the frame-level plan (a single-tile encoder
  // passes its own); a multi-tile caller passes the shared frame plan so the
  // per-superblock symbols match the filtered reference.
  bool write(std::vector<std::uint8_t>& tile_payload,
             std::vector<syntax::BlockSyntax>& blocks,
             std::vector<std::int32_t>& levels,
             const post::VideoFilterPlan* filters, std::string& error) {
    syntax::FramePlan plan;
    plan.width = width_;
    plan.height = height_;
    plan.sb_size = sb_size_;
    plan.frame_mi_cols = frame_mi_cols_;
    plan.frame_mi_rows = frame_mi_rows_;
    plan.tile_mi_row = tile_mi_row_;
    plan.tile_mi_col = tile_mi_col_;
    plan.inter_frame = !keyframe_;
    plan.enable_sdp = options_.enable_sdp;
    plan.frame_sdp = keyframe_ && options_.enable_sdp;
    plan.tx_mode_select = options_.enable_tx_search;
    plan.reduced_tx_part_set = false;
    plan.reduced_tx_set = 0;
    plan.allow_tcq = options_.enable_tcq;
    plan.enable_cctx = options_.enable_cctx;
    plan.num_total_refs = num_total_refs_;
    plan.reference_select = options_.compound && num_total_refs_ > 1;
    plan.enable_cwp = options_.enable_cwp_weighted;
    plan.allow_bawp = (options_.enable_bawp || options_.enable_morph) &&
                      !keyframe_;
    // §5.4.7/§5.18.3.3/.4 screen-content tools.  Only the intra keyframe and
    // only the single-tile writer: the palette above/left cache and the IBC
    // BV ring are tile-local, and a tile boundary would change both.
    const bool scc_carrier = keyframe_ && options_.tile_cols == 0 &&
                             options_.tile_rows == 0;
    plan.allow_screen_content_tools =
        scc_carrier && (options_.enable_morph || options_.enable_palette ||
                        options_.enable_ibc);
    plan.allow_intrabc =
        scc_carrier && (options_.enable_morph || options_.enable_ibc);
    plan.force_integer_mv = true;
    // §5.20.5.4 reads morph_pred inside read_intrabc_info() whenever
    // allow_screen_content_tools && enable_bawp, independent of --video-morph.
    // So any IBC carrier on a BAWP sequence codes the symbol; without the
    // morph fit it is always 0.
    plan.morph_pred_present =
        plan.allow_intrabc && (options_.enable_bawp || options_.enable_morph);
    plan.palettes = std::move(palettes_);
    plan.enable_interintra = !keyframe_ && options_.enable_interintra;
    plan.enable_masked_compound =
        plan.reference_select &&
        (options_.compound_wedge || options_.compound_diffwtd);
    plan.opfl_refine_type = plan.reference_select ? options_.opfl_refine_type : 0;
    plan.enable_refinemv = plan.reference_select && options_.enable_refinemv;
    plan.mv_precision = options_.mv_precision;
    plan.drl_reorder = options_.drl_reorder
                           ? (options_.drl_reorder_always ? 2 : 1)
                           : 0;
    plan.frame_distance = frame_distance_;
#if defined(AV2ENC_VIDEO_DELTAQ)
    plan.segmentation = seg_params_;
    plan.delta_q = dq_params_;
    plan.primary_ref_is_none = primary_ref_is_none_;
    plan.delta_steps = sb_delta_steps_;
#endif
    plan.temporal.use_ref_frame_mvs = temporal_ctx_.use_ref_frame_mvs;
    plan.temporal.order_hint = temporal_ctx_.order_hint;
    plan.temporal.num_past_refs = temporal_ctx_.num_past_refs;
    plan.temporal.num_future_refs = temporal_ctx_.num_future_refs;
    plan.temporal.closest_past = temporal_ctx_.closest_past;
    plan.temporal.closest_future = temporal_ctx_.closest_future;
    plan.temporal.ref_order_hints = ref_order_hints_;
    plan.temporal.field = temporal_ctx_.field;
    plan.temporal.scan = temporal_ctx_.use_ref_frame_mvs
                             ? &video_temporal_scan
                             : nullptr;
#if defined(AV2ENC_VIDEO_GME)
    if (gm_plan_ != nullptr) {
      plan.gm_num_refs = gm_plan_->num_refs;
      plan.gm_params = gm_plan_->params;
    }
#endif
    VideoConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(width_);
    config.max_frame_height = static_cast<std::uint16_t>(height_);
    config.use_128x128_superblock = options_.sb128;
    config.use_256x256_superblock = options_.sb256;
    config.enable_ext_partitions = options_.ext_partitions;
    config.enable_uneven_4way_partitions = options_.ext_partitions;
    config.base_q_idx = options_.base_q_idx;
    config.order_hint_bits = 4;
    config.num_ref_frames = 2;
    config.num_total_refs = 1;
    config.max_drl_bits_minus_1 = 4;
    config.disable_cdf_update = !options_.adaptive_cdf;
    config.enable_avg_cdf = options_.avg_cdf;
    config.avg_cdf_type = true;
    config.enable_cfl_intra = true;
    config.enable_mhccp = options_.enable_mhccp;
    config.enable_ibp = options_.enable_ibp;
    config.enable_idtx_intra = options_.enable_idtx_intra;
    config.enable_mrls = options_.enable_mrl;
    config.enable_dip = options_.enable_dip;
    config.enable_intra_ist = options_.enable_ist;
    config.enable_inter_ist = false;
    config.enable_tcq = options_.enable_tcq;
    config.choose_tcq_per_frame = options_.enable_tcq;
    config.enable_cctx = options_.enable_cctx;
    config.enable_sdp = options_.enable_sdp;
    config.enable_extended_sdp = false;
    config.enable_inter_ddt = options_.enable_inter_ddt;
    config.using_qmatrix = options_.enable_qm;
    config.qm_user_data = options_.user_qm;
    config.enable_deblock = options_.enable_deblock;
    config.tx_mode_select = options_.enable_tx_search;
    config.enable_bawp = options_.enable_bawp || options_.enable_morph;
    config.seq_screen_content_tools_present =
        keyframe_ && (options_.enable_morph || options_.enable_palette ||
                      options_.enable_ibc);
    config.enable_interintra = options_.enable_interintra;

    std::size_t tile_bytes = 0;
    if (!syntax::assemble_video_frame(plan, blocks_, levels_, config,
                                   tile_payload, error, &tile_bytes,
                                   cdf_seed_, cdf_final_, filters)) {
      return false;
    }
    blocks = std::move(blocks_);
    levels = std::move(levels_);
    return true;
  }

  bool encode(std::vector<std::uint8_t>& tile_payload,
              std::vector<syntax::BlockSyntax>& blocks,
              std::vector<std::int32_t>& levels, std::string& error) {
    if (!decide(error)) return false;
    if (!deblock(error)) return false;
    if (!apply_loop_filters(error)) return false;
    return write(tile_payload, blocks, levels,
                 filter_plan_.any() ? &filter_plan_ : nullptr, error);
  }

  const std::vector<syntax::BlockSyntax>& blocks() const {
    return blocks_;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  // §7.22 motion-field source: the per-4x4 coded mode/MV grid and its extent.
  // Used by the opt-in TIP path to store a reference frame's SavedMvField.
  const std::vector<inter::mc::MiInfo>& mv_grid() const { return mv_grid_; }
  int mv_mi_cols() const { return mi_cols_; }
  int mv_mi_rows() const { return mi_rows_; }
  int recon_mi_cols() const { return mi_cols_; }
  int recon_mi_rows() const { return mi_rows_; }
  std::uint32_t inter_blocks() const { return inter_blocks_; }
  std::uint32_t intra_blocks() const { return intra_blocks_; }
  int temporal_added() const { return temporal_added_; }
  int temporal_selected() const { return temporal_selected_; }
  const post::VideoFilterPlan& filter_plan() const { return filter_plan_; }

 private:
  // §5.20.3.2 symbol plan for one SDP node, derived by the reference planner
  // with the authoritative boundary/allowed set the solver stored.  The
  // writer emits exactly these symbols.
  bool plan_sdp_symbols(const sdpp::Node& node, syntax::BlockSyntax& entry,
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

  // §5.20.3.1 SDP intra tree walk.  A split node carries no partition syntax
  // and its two children are the luma root and then the chroma root.  Leaves
  // are encoded with the tree-type/chroma-ownership arguments so a CHROMA_PART
  // leaf codes chroma only and a LUMA_PART leaf codes luma only.
  bool collect_sdp_node(const sdpp::Node& node, std::string& error) {
    if (node.depth == 0) {
      // §6.19.2.3 clear_block_decoded_flags(): superblock-local.
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
        if (!collect_sdp_node(child, error)) return false;
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
      if (!encode_leaf(
              pnode, index, error, static_cast<int>(node.tree_type),
              node.codes_chroma, node.chroma.mi_row, node.chroma.mi_col,
              node.chroma.size == BlockSize::kInvalid
                  ? -1
                  : static_cast<int>(node.chroma.size))) {
        return false;
      }
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
        if (!collect_sdp_node(child, error)) return false;
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

  // §5.20.3.1: a node with HasChroma and chromaOffset == 0 captures the
  // inherited chroma block before its children are decoded.  ChromaMi* is the
  // last such node on the path, so a sub-8 split hands the same ChromaMi block
  // to the child that keeps HasChroma.
  static ChromaInfo chroma_step(const PartitionNode& node,
                                const ChromaInfo& inherited) {
    if (node.has_chroma && !node.chroma_offset) {
      return ChromaInfo{true, node.mi_row, node.mi_col, node.size};
    }
    return inherited;
  }

  bool collect_node(const PartitionNode& node, std::string& error,
                    const ChromaInfo& inherited_chroma = ChromaInfo{}) {
    const ChromaInfo chroma = chroma_step(node, inherited_chroma);
    if (node.depth == 0) {
      // §6.19.2.3 clear_block_decoded_flags(): the map is superblock-local.
      decoded_[0].reset(mi_cols_, mi_rows_);
      decoded_[1].reset(mi_cols_ / 2, mi_rows_ / 2);
      decoded_[2].reset(mi_cols_ / 2, mi_rows_ / 2);
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
        current_qindex_ = sb_qindex_[static_cast<std::size_t>(sb)];
        current_segment_id_ =
            seg_enabled_
                ? static_cast<int>(sb_segment_[static_cast<std::size_t>(sb)])
                : 0;
        quant_step_ = rc::kQlookupQ0To255[std::clamp(current_qindex_, 0, 255)];
        if (seg_enabled_ && node.mi_col == 0) {
          deltaq::clear_left_context(enc_seg_grid_);
        }
        if (trace_ && seg_enabled_) {
          std::cerr << "segq sb=" << sb
                    << " segment=" << current_segment_id_
                    << " qindex=" << current_qindex_ << "\n";
        }
      }
#endif
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
      if (!encode_leaf(node, index, error, /*tree_type=*/0,
                       node.has_chroma, chroma.mi_row, chroma.mi_col,
                       chroma.valid ? static_cast<int>(chroma.size) : -1)) {
        return false;
      }
    } else {
      for (const PartitionNode& child : node.children) {
        if (!collect_node(child, error, chroma)) return false;
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

#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  // Extended-partition variant of collect_node().  The solver already resolved
  // §5.20.3.1 ChromaMi* for every node (including the HORZ_3/VERT_3 middle
  // pair and the 1/8 subsizes), so the leaf coder receives exactly the
  // ownership arguments the decoder will derive.
  bool collect_node_ext(const pe::PartitionNode& node, std::string& error) {
    if (node.depth == 0) {
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
      if (!encode_leaf(
              leaf, index, error, /*tree_type=*/0, node.has_chroma,
              node.chroma.mi_row, node.chroma.mi_col,
              node.chroma.size == BlockSize::kInvalid
                  ? -1
                  : static_cast<int>(node.chroma.size))) {
        return false;
      }
    } else {
      for (const pe::PartitionNode& child : node.children) {
        if (!collect_node_ext(child, error)) return false;
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
#endif  // AV2ENC_VIDEO_EXT_PARTITIONS

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
    params.use_ref_frame_mvs = temporal_ctx_.use_ref_frame_mvs;
    params.motion_field = temporal_ctx_.field;
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
    params.temporal_added = &temporal_added_;
    attach_video_temporal_scan(&params);
    return params;
  }

  bool drl_constraint_sort_suppressed(int ref_index) const {
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

  // ---------------------------------------------------------------------
  // §5.20.5.3/.4 intra block copy + §7.13.3.26 morphological prediction
  // ---------------------------------------------------------------------
  ibc::FrameView ibc_frame_view() const {
    ibc::FrameView view;
    view.y = ibc::PlaneView{recon_[0].samples.data(), recon_[0].stride,
                            mi_cols_ * 4, mi_rows_ * 4};
    view.u = ibc::PlaneView{recon_[1].samples.data(), recon_[1].stride,
                            mi_cols_ * 2, mi_rows_ * 2};
    view.v = ibc::PlaneView{recon_[2].samples.data(), recon_[2].stride,
                            mi_cols_ * 2, mi_rows_ * 2};
    view.has_chroma = true;
    view.sub_x = 1;
    view.sub_y = 1;
    return view;
  }

  struct IbcChoice {
    bool found = false;
    ibc::BlockVector bv{};
    int stack_index = -1;  // >= 0 => NEARMV with that RefMvIdx
    bool morph = false;    // §7.13.3.26 adaptation selected
    std::int64_t sse = 0;
    std::vector<std::uint8_t> luma;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
    std::vector<std::uint8_t> morph_scratch;
  };

  // Chooses an IBC block vector for a keyframe leaf.  A plain exact copy and a
  // §7.13.3.26 morphological adaptation are both evaluated; the best luma SSE
  // below the caller's intra SSE wins.  Returns false only on a hard error.
  bool choose_ibc_leaf(int r, int c, int bsize, std::int64_t intra_sse,
                       IbcChoice& out, std::string& error) {
    out = IbcChoice{};
    if (!ibc_enabled_ || !ibcs::intrabc_available(true, bsize)) return true;
    const auto size = static_cast<partition::BlockSize>(bsize);
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int visible_w = std::min(w, mi_cols_ * 4 - x0);
    const int visible_h = std::min(h, mi_rows_ * 4 - y0);
    const int cw = w / 2;
    const int ch = h / 2;
    // IBC is an 8-bit 4:2:0 tool here; a leaf whose chroma block is not the
    // plain half-size (4:2:0 shared-chroma partitions) is not coded.
    if (!ibc_available_for_leaf(bsize)) return true;
    const ibc::FrameView frame = ibc_frame_view();
    const ibcs::BvStack stack = ibcs::build_bv_stack(
        frame, ibc_config_, ibc_buffer_, r, c, bsize, ibc_neighbours_);

    morph::MorphContext mctx;
    mctx.mi_col = c;
    mctx.mi_row = r;
    mctx.mi_cols = mi_cols_;
    mctx.mi_rows = mi_rows_;
    mctx.avail_u = y0 > 0;
    mctx.avail_l = x0 > 0;
    mctx.bit_depth = 8;
    const morph::MorphRect rect{x0, y0, w, h};

    const auto consider = [&](ibc::BlockVector bv, int stack_index) {
      if ((bv.row & 7) != 0 || (bv.col & 7) != 0) return;
      if (!ibcs::is_bv_encodable(frame, ibc_config_, ibc_buffer_, r, c, bsize,
                                 bv, nullptr)) {
        return;
      }
      std::vector<std::uint8_t> luma(static_cast<std::size_t>(w) *
                                     static_cast<std::size_t>(h));
      std::vector<std::uint8_t> u(static_cast<std::size_t>(cw) *
                                  static_cast<std::size_t>(ch));
      std::vector<std::uint8_t> v(u.size());
      if (!ibc::copy_block(frame, r, c, bsize, bv, luma.data(), w, u.data(),
                           cw, v.data(), cw, visible_w, visible_h)) {
        return;
      }
      const std::int64_t copy_sse = sse_against_source(
          source_, 0, x0 + tile_x_, y0 + tile_y_, w, h, luma);
      bool best_morph = false;
      std::int64_t best_sse = copy_sse;
      std::vector<std::uint8_t> adapted;
      if (plan_morph_) {
        std::string why;
        if (ibcs::is_bv_offset_valid(frame, ibc_config_, ibc_buffer_, r, c,
                                     bsize, bv, -1, -1, &why)) {
          // The fit reads the reconstructed above/left template and the source
          // block at refX/refY, so evaluate it on a full-plane snapshot where
          // only this block has been written.
          out.morph_scratch.assign(recon_[0].samples.begin(),
                                   recon_[0].samples.end());
          morph::MorphPlane plane{out.morph_scratch.data(), recon_[0].stride};
          for (int yy = 0; yy < h; ++yy) {
            std::memcpy(out.morph_scratch.data() +
                            static_cast<std::size_t>(y0 + yy) *
                                static_cast<std::size_t>(recon_[0].stride) +
                            static_cast<std::size_t>(x0),
                        luma.data() + static_cast<std::size_t>(yy) * w,
                        static_cast<std::size_t>(w));
          }
          morph::MorphFit fit{};
          if (morph::morph_predict(mctx, plane, rect, bv.row, bv.col, &fit)) {
            adapted.resize(luma.size());
            for (int yy = 0; yy < h; ++yy) {
              std::memcpy(
                  adapted.data() + static_cast<std::size_t>(yy) * w,
                  out.morph_scratch.data() +
                      static_cast<std::size_t>(y0 + yy) *
                          static_cast<std::size_t>(recon_[0].stride) +
                      static_cast<std::size_t>(x0),
                  static_cast<std::size_t>(w));
            }
            const std::int64_t morph_sse = sse_against_source(
                source_, 0, x0 + tile_x_, y0 + tile_y_, w, h, adapted);
            if (morph_sse < best_sse) {
              best_sse = morph_sse;
              best_morph = true;
            }
          }
        }
      }
      if (!out.found || best_sse < out.sse) {
        out.found = true;
        out.bv = bv;
        out.stack_index = stack_index;
        out.morph = best_morph;
        out.sse = best_sse;
        out.luma = best_morph ? adapted : luma;
        out.u = std::move(u);
        out.v = std::move(v);
      }
    };

    for (int i = 0; i < stack.count; ++i) {
      if (!stack.entries[static_cast<std::size_t>(i)].valid) continue;
      consider(stack.entries[static_cast<std::size_t>(i)].bv, i);
    }
    static constexpr int kPeriodic[] = {8, 16, 32, 64, 128};
    for (const int d : kPeriodic) {
      const ibc::BlockVector cands[2] = {ibc::BlockVector{-d * 8, 0},
                                         ibc::BlockVector{0, -d * 8}};
      for (const ibc::BlockVector& bv : cands) {
        consider(bv, -1);
      }
    }
    // A tie picks the IBC skip block: it codes no residual, so equal luma
    // distortion is a rate win over the intra leaf.
    if (!out.found || out.sse > intra_sse) {
      out.found = false;
    }
    (void)error;
    return true;
  }

  bool ibc_available_for_leaf(int bsize) const {
    const auto size = static_cast<partition::BlockSize>(bsize);
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    if (w < 8 || h < 8) return false;
    // §7.13.3.18 copies exactly w x h luma and (w/2) x (h/2) chroma.  Reject
    // the shared-chroma ownership cases where the chroma residual block is not
    // the plain half-size.
    return w % 2 == 0 && h % 2 == 0 && w / 2 >= 4 && h / 2 >= 4;
  }

  // ---------------------------------------------------------------------
  // §5.20.8 luma palette (--video-palette)
  // ---------------------------------------------------------------------
  struct PaletteChoice {
    bool found = false;
    palette::PaletteResult result;
    std::vector<std::uint8_t> luma;
    std::int64_t sse = 0;
  };

  bool palette_available_for_leaf(int bsize, int w, int h) const {
    if (!palette_enabled_) return false;
    pals::ScreenContentFlags flags;
    flags.single_picture_header = false;
    flags.allow_screen_content_tools = true;
    flags.allow_palette = true;
    flags.force_integer_mv = true;
    return pals::palette_allowed_for_block(flags, bsize, w, h);
  }

  // Builds the §5.20.8 palette for one keyframe leaf from the same above/left
  // cache the decoder's get_palette_cache() sees.  The palette prediction is
  // scored against `intra_pred_sse`, the SSE of the ordinary intra prediction
  // (the palette codes eob=0 on all planes, so a residual-coded intra leaf is
  // not the right comparison).  The caller stores a winning palette in the
  // frame PaletteState only after the IBC decision, so the following blocks'
  // cache matches the writer.
  bool choose_palette_leaf(int r, int c, int bsize, int w, int h,
                           std::int64_t intra_pred_sse, PaletteChoice& out,
                           std::string& error) {
    out = PaletteChoice{};
    if (!palette_available_for_leaf(bsize, w, h)) {
      return true;
    }
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int visible_w = std::min(w, mi_cols_ * 4 - x0);
    const int visible_h = std::min(h, mi_rows_ * 4 - y0);
    if (visible_w <= 0 || visible_h <= 0) return true;
    std::uint8_t cache[16] = {};
    const int cache_n = palette::get_palette_cache(palette_state_, r, c,
                                                   c > 0, cache);
    std::vector<std::uint8_t> src(static_cast<std::size_t>(w) *
                                  static_cast<std::size_t>(h));
    for (int yy = 0; yy < h; ++yy) {
      for (int xx = 0; xx < w; ++xx) {
        src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
            static_cast<std::size_t>(xx)] =
            static_cast<std::uint8_t>(source_sample(
                source_, 0, tile_x_ + x0 + xx, tile_y_ + y0 + yy));
      }
    }
    palette::PaletteParams params;
    params.max_colors = 8;
    params.exact_when_possible = true;
    params.choose_direction = true;
    palette::PaletteResult result;
    std::string palette_error;
    if (!palette::analyze_palette(src.data(), w, w, h, visible_w, visible_h,
                                  cache, cache_n, params, result,
                                  palette_error)) {
      error = "analyze_palette failed: " + palette_error;
      return false;
    }
    // §5.20.8.1 delta-codes a strictly increasing palette.
    bool representable = result.has_palette;
    for (int i = 1; i < result.size && representable; ++i) {
      if (result.colors[static_cast<std::size_t>(i)] <=
          result.colors[static_cast<std::size_t>(i - 1)]) {
        representable = false;
      }
    }
    if (!representable) result.has_palette = false;
    if (!result.has_palette || result.sse >= intra_pred_sse) return true;
    out.luma.resize(static_cast<std::size_t>(w) *
                    static_cast<std::size_t>(h));
    palette::predict_palette_block(result, w, h, out.luma.data(), w);
    out.sse = result.sse;
    out.result = std::move(result);
    out.found = true;
    return true;
  }

  // §7.13.2 single-mode prediction for the inter-intra candidates.  The
  // neighbour counts come from the same §6.19.2.3 replay the intra search uses;
  // chroma smooth flags are always false in this encoder (UVMode is DC/CfL).
  bool predict_intra_plane_mode(int plane, int x0, int y0, int w, int h,
                                intra::Mode mode, bool above_smooth,
                                bool left_smooth,
                                std::vector<std::uint8_t>& out) const {
    const int sb_mi = plane == 0 ? sb_size_ / 4 : sb_size_ / 8;
    const int mi_c = plane == 0 ? mi_cols_ : mi_cols_ / 2;
    const int mi_r = plane == 0 ? mi_rows_ : mi_rows_ / 2;
    const intra_search::Availability avail = intra_search::availability(
        decoded_[plane], x0 / 4, y0 / 4, w / 4, h / 4, mi_c, mi_r, sb_mi);
    intra::NeighborRequest request;
    request.recon =
        intra::PlaneView{recon_[plane].samples.data(), recon_[plane].stride};
    request.x = x0;
    request.y = y0;
    request.w = w;
    request.h = h;
    request.max_x = recon_[plane].width - 1;
    request.max_y = recon_[plane].height - 1;
    request.have_left = x0 > 0;
    request.have_above = y0 > 0;
    request.num4_above_right = avail.num4_above_right;
    request.num4_below_left = avail.num4_below_left;
    request.color_plane = plane;
    intra::PredictParams params;
    params.mode = mode;
    params.angle_delta = 0;
    params.enable_edge_filter = true;
    params.above_neighbor_is_smooth = above_smooth;
    params.left_neighbor_is_smooth = left_smooth;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    return intra::predict_block(request, 8, params, out.data(),
                                static_cast<std::size_t>(w)) ==
           intra::IntraStatus::kOk;
  }

  // §7.13.2 single-mode prediction for the inter-intra candidates.  The
  // neighbour counts come from the same §6.19.2.3 replay the intra search uses;
  // chroma smooth flags are always false in this encoder (UVMode is DC/CfL).
  // §7.13.3.29/.30 inter-intra.  Evaluates the four intra modes plus the
  // signalable wedge masks against the inter prediction and returns the best
  // luma+chroma blend.  `inter_y/u/v` are the single-reference predictions.
  bool evaluate_inter_intra(int bsize, int x0, int y0, int w, int h,
                            const std::vector<std::uint8_t>& inter_y,
                            const std::vector<std::uint8_t>& inter_u,
                            const std::vector<std::uint8_t>& inter_v,
                            std::vector<std::uint8_t>& out_y,
                            std::vector<std::uint8_t>& out_u,
                            std::vector<std::uint8_t>& out_v, int& mode_out,
                            bool& wedge_out, int& index_out,
                            std::int64_t& sse_out, std::string& error) {
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
                                    left_smooth, ii_y[m]) ||
          !predict_intra_plane_mode(1, x0 / 2, y0 / 2, w / 2, h / 2,
                                    kModes[m], false, false, ii_u[m]) ||
          !predict_intra_plane_mode(2, x0 / 2, y0 / 2, w / 2, h / 2,
                                    kModes[m], false, false, ii_v[m])) {
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
          luma_mask[static_cast<std::size_t>(i) * w + j] =
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

#if defined(AV2ENC_VIDEO_FRAMES)
  // --- §7.13.3.8-.12 / §7.13.3.3-.6 compound prediction -------------------
  //
  // The temporal tools predict from the two reference planes by absolute
  // frame coordinates and write a block-local result, so a candidate never
  // disturbs the reconstructed reference planes.
  static temporal::PlaneView temporal_plane(const inter::mc::RefFrame& ref,
                                            int plane) {
    temporal::PlaneView view;
    view.data = ref.plane[plane].data;
    view.stride = ref.plane[plane].stride;
    view.width = ref.plane_width(plane);
    view.height = ref.plane_height(plane);
    return view;
  }

  static temporal::RefArea full_area(const temporal::PlaneView& view) {
    return temporal::RefArea{0, 0, view.width - 1, view.height - 1};
  }

  // §7.13.3.4 get ref area for one reference list (the input is identical for
  // both lists except the candidate MV).
  bool temporal_ref_area(int sub_x, int sub_y, int x, int y, int w, int h,
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
      in.ref_mi_cols[list] = references_[list].mi_cols;
      in.ref_mi_rows[list] = references_[list].mi_rows;
      in.ref_luma_w[list] = in.frame_w;
      in.ref_luma_h[list] = in.frame_h;
      in.cand_mvs[list] = temporal::Mv{mvs[list].y, mvs[list].x};
    }
    return temporal::compute_ref_area(in, out) ==
           temporal::TemporalStatus::kOk;
  }

  // Builds the two 1/16-sample predictions for one compound block.  Shared by
  // the plain path and the CWP / implicit-mask decision, which evaluates
  // several blend modes over the same pair of predictions.
  bool build_compound_block(const inter::mc::Mv mvs[2], int bsize, int x0,
                            int y0, int w, int h,
                            multiref::CompoundPrediction& prediction,
                            std::string& error) {
    return multiref::build_compound_prediction(
        references_[0], references_[1], inter::mc::BlockRect{x0, y0, w, h},
        bsize, mvs[0], mvs[1], inter::mc::InterpFilter::kEighttap,
        references_[0].mi_cols, references_[0].mi_rows, prediction, error);
  }

  // Applies one blend mode and copies the block out of the frame-sized
  // scratch planes.  `params.use_imp_mask` is taken as given (the caller
  // derives it with multiref::implicit_mask_eligible()).
  bool store_compound_block(const multiref::CompoundPrediction& prediction,
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
      std::memcpy(y.data() + static_cast<std::size_t>(yy) * w,
                  compound_pred_[0].samples.data() +
                      static_cast<std::size_t>(y0 + yy) * compound_pred_[0].stride +
                      x0,
                  static_cast<std::size_t>(w));
    }
    for (int yy = 0; yy < ch; ++yy) {
      std::memcpy(u.data() + static_cast<std::size_t>(yy) * cw,
                  compound_pred_[1].samples.data() +
                      static_cast<std::size_t>(cy0 + yy) * compound_pred_[1].stride +
                      cx0,
                  static_cast<std::size_t>(cw));
      std::memcpy(v.data() + static_cast<std::size_t>(yy) * cw,
                  compound_pred_[2].samples.data() +
                      static_cast<std::size_t>(cy0 + yy) * compound_pred_[2].stride +
                      cx0,
                  static_cast<std::size_t>(cw));
    }
    return true;
  }

  // Plain COMPOUND_AVERAGE prediction at the coded MVs.
  bool compound_plain(const inter::mc::Mv mvs[2], int bsize, int x0, int y0,
                      int w, int h, std::vector<std::uint8_t>& y,
                      std::vector<std::uint8_t>& u,
                      std::vector<std::uint8_t>& v, std::string& error,
                      const multiref::CompoundModeParams* mode_params =
                          nullptr) {
    multiref::CompoundPrediction prediction;
    if (!build_compound_block(mvs, bsize, x0, y0, w, h, prediction, error)) {
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

  // §5.20.7.6 / §7.13.3.14: choose the AVERAGE blend for a NEAR_NEARMV
  // compound block.  With `enable_cwp_weighted` the non-equal
  // Cwp_Weighting_Factor row is searched; with `enable_imp_msk_bld` the
  // derived implicit mask is evaluated first.  Both are off by default and the
  // caller falls back to compound_plain(), keeping historical streams exact.
  bool compound_search_average(const inter::mc::Mv mvs[2], int bsize, int x0,
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

  // §7.13.3.15 save_subpu for the §7.13.3.8 OPFL / §7.13.3.1 refine-MV
  // subblocks.  The two tools split luma and chroma independently, so each
  // plane contributes its own rect at the same plane coordinates the predictor
  // used.  Called only when --video-df-sub-pu is on.
  void collect_sub_pus(int x0, int y0, int w, int h, bool optflow,
                       bool refine) {
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
      sub_pus_.push_back(rect);
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

  bool predict_compound_optflow(const inter::mc::Mv mvs[2], int x0, int y0,
                                int w, int h, std::string& error) {
    const int fw = static_cast<int>(source_.width);
    const int fh = static_cast<int>(source_.height);
    const temporal::PlaneView l0 = temporal_plane(references_[0], 0);
    const temporal::PlaneView l1 = temporal_plane(references_[1], 0);
    const temporal::RefArea fa0 = full_area(l0);
    const temporal::RefArea fa1 = full_area(l1);
    std::vector<std::uint8_t> ip0(static_cast<std::size_t>(w) * h);
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
    std::vector<std::int16_t> opfl(static_cast<std::size_t>(w4) * h4 * 4);
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
          ((static_cast<std::size_t>(row4) * w4 + col4) * 2 + list) * 2 + comp);
    };
    compound_y_.assign(static_cast<std::size_t>(w) * h, 0);
    const int cw = w / 2;
    const int ch = h / 2;
    compound_u_.assign(static_cast<std::size_t>(cw) * ch, 0);
    compound_v_.assign(compound_u_.size(), 0);
    std::vector<std::int16_t> s0(static_cast<std::size_t>(n) * n);
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
            compound_y_.data() + static_cast<std::size_t>(i2) * w + j2, w);
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
          plane == 1 ? compound_u_ : compound_v_;
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
              dst.data() + static_cast<std::size_t>(i2) * cw + j2, cw);
        }
      }
    }
    return true;
  }

  // §7.13.3.3-.6 refine-MV, writing compound_y_/u_/v_ with EIGHTTAP_SHARP.
  //
  // §7.13.3.1 predicts a refine-MV block in 16x16 (plane) sub-blocks:
  //   tw = Min(w, 16 >> subX), th = Min(h, 16 >> subY)
  // and invokes predict_refinemv() per sub-block.  The §7.13.3.6 search, the
  // §7.13.3.4 ref area and the prediction are therefore all sub-block sized;
  // only the luma pass searches, and chroma reuses the luma RefineMvs entry
  // for the colocated sub-block (predict_refinemv, plane > 0).
  bool predict_compound_refinemv(const inter::mc::Mv mvs[2], int x0, int y0,
                                 int w, int h, std::string& error) {
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

    std::vector<temporal::MvPair> sb_mvs(static_cast<std::size_t>(rows) * cols);
    std::vector<std::uint8_t> s0(static_cast<std::size_t>(tw + 4) * (th + 4));
    std::vector<std::uint8_t> s1(s0.size());
    std::vector<std::int16_t> r0(static_cast<std::size_t>(tw) * th);
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

    compound_y_.assign(static_cast<std::size_t>(w) * h, 0);
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
          search.ref_mi_cols[list] = references_[list].mi_cols;
          search.ref_mi_rows[list] = references_[list].mi_rows;
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
        sb_mvs[static_cast<std::size_t>(r) * cols + c] = result.offset_mvs;
        if (!predict_plane_refine(
                0, 0, l0, l1, x, y, tw, th, area, result.offset_mvs,
                compound_y_.data() +
                    static_cast<std::size_t>(r * th) * w + c * tw,
                w)) {
          error = "refine-MV luma prediction failed";
          return false;
        }
      }
    }

    compound_u_.assign(static_cast<std::size_t>(cw) * ch, 0);
    compound_v_.resize(compound_u_.size());
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
            sb_mvs[static_cast<std::size_t>(r) * cols + c];
        if (!predict_plane_refine(
                1, 1, u0, u1, x0 / 2 + x, y0 / 2 + y, twc, thc, carea, mvs_off,
                compound_u_.data() + static_cast<std::size_t>(y) * cw + x,
                cw) ||
            !predict_plane_refine(
                1, 1, v0, v1, x0 / 2 + x, y0 / 2 + y, twc, thc, carea, mvs_off,
                compound_v_.data() + static_cast<std::size_t>(y) * cw + x,
                cw)) {
          error = "refine-MV chroma prediction failed";
          return false;
        }
      }
    }
    return true;
  }

#endif  // AV2ENC_VIDEO_FRAMES
  bool encode_leaf(const PartitionNode& node, std::size_t index,
                   std::string& error, int tree_type = 0,
                   bool leaf_has_chroma = true, int chroma_mi_row = -1,
                   int chroma_mi_col = -1, int chroma_bsize = -1) {
    const int r = node.mi_row;
    const int c = node.mi_col;
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
    if (ibc_enabled_) {
      // §5.20.4.1 decode_block(): update_ibc_buffers() runs at the start of
      // every block, before its mode info builds the BV stack.
      ibcs::begin_ibc_block(ibc_buffer_, r, c);
    }

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
            // The historical search truncates small blocks to the first two
            // candidates.  With temporal MVs enabled the §7.12.2.7/.8 entries
            // can sit past that point (sorting by weight), so search the whole
            // stack; every index is codable and the writer validates it.
            const int near_limit =
                (w > 32 && h > 32) || options_.enable_ref_frame_mvs
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
              cand.sse = sse_against_source(source_, 0, ax0, ay0, w, h, luma);
              if (options_.ref_mvs_stress && cand.from_temporal) {
                cand.sse = -1;
              }
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
                  cand.from_temporal = stack.cand[0].from_temporal;
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
          cand.sse = compound_sse >= 0
                         ? compound_sse
                         : sse_against_source(source_, 0, x0, y0, w, h,
                                              compound_y_);
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
      const bool have0 = load_stack(r, c, w4, h4, 0, p0, l0);
      const bool have1 = load_stack(r, c, w4, h4, 1, p1, l1);
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
          ok = predict_compound_optflow(mvs, ax0, ay0, w, h, error);
        } else if (block_refine) {
          ok = predict_compound_refinemv(mvs, ax0, ay0, w, h, error);
        } else if (options_.enable_cwp_weighted ||
                   options_.enable_imp_msk_bld) {
          ok = compound_search_average(
              mvs, static_cast<int>(node.size), ax0, ay0, w, h, compound_y_,
              compound_u_, compound_v_, chosen_params, chosen_sse, error);
        } else {
          ok = compound_plain(mvs, static_cast<int>(node.size), ax0, ay0, w, h,
                              compound_y_, compound_u_, compound_v_, error);
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
                       : sse_against_source(source_, 0, ax0, ay0, w, h,
                                            compound_y_);
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
        merge::StackParams cparams = make_stack_params(r, c, w4, h4, 0);
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
      bool luma_ok = chroma_part;
      if (!chroma_part) {
      const intra_search::Availability luma_avail = intra_search::availability(
          decoded_[0], x0 / 4, y0 / 4, w / 4, h / 4, mi_cols_, mi_rows_,
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
            sse_against_source(source_, 0, ax0, ay0, w, h, intra_y_pred);
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
      if (!luma_ok ||
          (options_.enable_tx_search
               ? !search_luma_tx(
                     source_, ax0, ay0, w, h, intra_y_pred, current_qindex_,
                     txtsearch::Mode::kIntra, options_.tx_rd_lambda,
                     options_.enable_idtx_intra,
                     std::max(0, mrl_stream::mapped_intra_mode_for_mrl(
                                     intra_y_mode, intra_y_angle_delta,
                                     intra_mrl_index, w, h)),
                     options_.enable_tcq,
                     ::av2enc::still::qctx::slice_for_base_q_idx(
                         current_qindex_),
                     /*enable_inter_ddt=*/false, /*search_partition=*/false,
                     options_.trace, intra_y, error)
               : !compute_tx(source_, 0, ax0, ay0, w, h, intra_y_pred,
                             quant_step_, intra_y, error,
                             qm_enabled_ ? &qm_state_ : nullptr,
                             options_.enable_tcq,
                             ::av2enc::still::qctx::slice_for_base_q_idx(current_qindex_),
                             /*is_inter=*/false))) {
        error = "intra prediction failed";
        return false;
      }
      // §7.15.3 luma IST refinement (--video-ist).  Applies the still
      // encoder's analysis projection to the quantised luma levels and keeps
      // the candidate that minimises SSE + lambda * symbol bits.
      if (options_.enable_ist &&
          !refine_luma_ist(source_, ax0, ay0, w, h, intra_y_pred,
                           quant_step_, intra_y_mode, intra_y_angle_delta,
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
            sse_against_source(source_, 1, acx0, acy0, cw, ch, intra_u_pred);
        const int frame_chroma_width = static_cast<int>((source_.width + 1U) / 2U);
        const int frame_chroma_height =
            static_cast<int>((source_.height + 1U) / 2U);
        chroma_search::PlaneView u_plane;
        u_plane.recon = recon_[1].samples.data();
        u_plane.recon_stride = recon_[1].stride;
        u_plane.recon_width = recon_[1].width;
        u_plane.recon_height = recon_[1].height;
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
            intra_search::availability(decoded_[1], cx0 / 4, cy0 / 4, cw / 4,
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
                                 intra_v_pred, quant_step_, /*is_inter=*/false,
                                 intra_u, intra_v, &intra_cctx_type, error)) {
          return false;
        }
      } else if (!compute_tx(source_, 1, acx0, acy0, cw, ch, intra_u_pred,
                             quant_step_, intra_u, error,
                             qm_enabled_ ? &qm_state_ : nullptr) ||
                 !compute_tx(source_, 2, acx0, acy0, cw, ch, intra_v_pred,
                             quant_step_, intra_v, error,
                             qm_enabled_ ? &qm_state_ : nullptr)) {
        return false;
      }
      }  // owns_chroma
      intra_sse = chroma_part
                      ? 0
                      : sse_against_source(source_, 0, ax0, ay0, w, h,
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
          sse_against_source(source_, 0, ax0, ay0, w, h, intra_y_pred);
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
    int inter_cctx_type = 0;
    bool bawp_applied = false;
    inter::adv::BawpFit bawp_fit{};
    bool inter_intra_applied = false;
    int inter_intra_mode = 0;
    bool inter_intra_wedge = false;
    int inter_intra_index = 0;
    std::int64_t inter_intra_sse = 0;
    std::int64_t inter_luma_sse = best.sse;
    if (!keyframe_ && best.valid) {
      if (best.compound) {
#if defined(AV2ENC_VIDEO_FRAMES)
        // Re-run the winning candidate: the compound scratch planes hold the
        // last evaluated candidate, not necessarily this one.
        const inter::mc::Mv mvs[2] = {best.mv, best.mv1};
        bool predicted = false;
        if (best.use_optflow) {
          predicted = predict_compound_optflow(mvs, ax0, ay0, w, h, error);
        } else if (best.use_refinemv) {
          predicted = predict_compound_refinemv(mvs, ax0, ay0, w, h, error);
        } else {
          multiref::CompoundModeParams wparams;
          wparams.cwp_weight = best.cwp_weight;
          wparams.cwp_index = static_cast<int>(best.cwp_index);
          wparams.use_imp_mask = best.use_imp_mask;
          wparams.type = best.compound_type == 0
                             ? inter::adv::CompoundType::kWedge
                             : (best.compound_type == 1
                                    ? inter::adv::CompoundType::kDiffWtd
                                    : inter::adv::CompoundType::kAverage);
          if (best.compound_mode == 0) {
            wparams.type = inter::adv::CompoundType::kAverage;
          }
          wparams.wedge_index = best.wedge_index;
          wparams.wedge_sign = best.wedge_sign;
          wparams.mask_type = best.mask_type;
          predicted = compound_plain(
              mvs, static_cast<int>(node.size), ax0, ay0, w, h, compound_y_,
              compound_u_, compound_v_, error, &wparams);
        }
        if (!predicted) {
          error = "compound prediction failed";
          return false;
        }
#endif
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
      // §5.20.7.6 / §7.13.3.25: evaluate the fitted BAWP predictor for an
      // eligible single-prediction block (>= 8x8, not GLOBALMV, no warp).
      // The decoder applies the same fit before the residual, so a selected
      // block carries use_bawp = 1 and the adapted prediction.
      if (options_.enable_bawp && options_.tile_cols == 0 && !best.compound &&
          !best.use_warp &&
          best.mode != syntax::kInterGlobalMv && w >= 8 && h >= 8 &&
          mi_cols_ * 4 == static_cast<int>(source_.width) &&
          mi_rows_ * 4 == static_cast<int>(source_.height)) {
        std::vector<std::uint8_t> adapted = inter_y;
        inter::adv::BawpFit fit{};
        const inter::mc::PlaneView& ref_y =
            references_[best.ref_index].plane[0];
        if (bawp_adapt_luma(recon_[0].samples.data(), recon_[0].stride,
                            ref_y.data, ref_y.stride, mi_cols_, mi_rows_, r,
                            c, x0, y0, w, h, best.mv.y, best.mv.x, adapted,
                            &fit)) {
          const std::int64_t adapted_sse =
              sse_against_source(source_, 0, ax0, ay0, w, h, adapted);
          if (adapted_sse < inter_luma_sse) {
            bawp_applied = true;
            bawp_fit = fit;
            inter_y = adapted;
            inter_luma_sse = adapted_sse;
          }
        }
        if (trace_) {
          std::cerr << "  cand bawp alpha=" << bawp_fit.alpha
                    << " count=" << bawp_fit.count
                    << " applied=" << (bawp_applied ? 1 : 0) << "\n";
        }
      }
      // §5.20.7.14/.15 + §7.13.3.29/.30 single-reference inter-intra.  The
      // blended block is skip-only (no residual) in this gated surface, and it
      // is evaluated after BAWP so the §5.20.7.14 use_bawp early return can
      // never be violated.
      if (options_.enable_interintra && !best.compound && !best.use_warp &&
          !bawp_applied && best.mode != syntax::kInterGlobalMv && w >= 8 &&
          h >= 8 && w <= 64 && h <= 64 && !options_.force_inter_residual) {
        std::vector<std::uint8_t> ii_y;
        std::vector<std::uint8_t> ii_u;
        std::vector<std::uint8_t> ii_v;
        if (!evaluate_inter_intra(static_cast<int>(node.size), x0, y0, w, h,
                                  inter_y, inter_u, inter_v, ii_y, ii_u, ii_v,
                                  inter_intra_mode, inter_intra_wedge,
                                  inter_intra_index, inter_intra_sse, error)) {
          return false;
        }
        const std::int64_t plain_sse =
            sse_against_source(source_, 0, ax0, ay0, w, h, inter_y) +
            sse_against_source(source_, 1, acx0, acy0, cw, ch, inter_u) +
            sse_against_source(source_, 2, acx0, acy0, cw, ch, inter_v);
        if (inter_intra_sse < plain_sse) {
          inter_intra_applied = true;
          inter_y = ii_y;
          inter_u = ii_u;
          inter_v = ii_v;
          inter_luma_sse =
              sse_against_source(source_, 0, ax0, ay0, w, h, inter_y);
        }
        if (trace_) {
          std::cerr << "  cand interintra applied="
                    << (inter_intra_applied ? 1 : 0)
                    << " mode=" << inter_intra_mode
                    << " wedge=" << (inter_intra_wedge ? 1 : 0)
                    << " index=" << inter_intra_index
                    << " sse=" << inter_intra_sse
                    << " plain=" << plain_sse << "\n";
        }
      }
      if (options_.enable_inter_residual && !inter_intra_applied) {
        const std::int64_t pred_total =
            inter_luma_sse +
            sse_against_source(source_, 1, acx0, acy0, cw, ch, inter_u) +
            sse_against_source(source_, 2, acx0, acy0, cw, ch, inter_v);
        if (((options_.enable_tx_search || options_.enable_inter_ddt)
                 ? !search_luma_tx(source_, ax0, ay0, w, h, inter_y,
                                   current_qindex_,
                                   txtsearch::Mode::kInter,
                                   options_.tx_rd_lambda,
                                   /*allow_idtx=*/false,
                                   /*intra_mapped_mode=*/0,
                                   options_.enable_tcq,
                                   ::av2enc::still::qctx::slice_for_base_q_idx(
                                       current_qindex_),
                                   options_.enable_inter_ddt,
                                   /*search_partition=*/
                                   options_.enable_tx_search, options_.trace,
                                   res_y, error)
                 : !compute_tx(source_, 0, ax0, ay0, w, h, inter_y,
                               quant_step_, res_y, error,
                               qm_enabled_ ? &qm_state_ : nullptr,
                               options_.enable_tcq,
                               ::av2enc::still::qctx::slice_for_base_q_idx(current_qindex_),
                               /*is_inter=*/true)) ||
            (options_.enable_cctx
                 ? !compute_chroma_cctx(source_, acx0, acy0, cw, ch, inter_u,
                                        inter_v, quant_step_, /*is_inter=*/true,
                                        res_u, res_v, &inter_cctx_type, error)
                 : (!compute_tx(source_, 1, acx0, acy0, cw, ch, inter_u,
                                quant_step_, res_u, error,
                                qm_enabled_ ? &qm_state_ : nullptr) ||
                    !compute_tx(source_, 2, acx0, acy0, cw, ch, inter_v,
                                quant_step_, res_v, error,
                                qm_enabled_ ? &qm_state_ : nullptr)))) {
          return false;
        }
        if (trace_ && options_.enable_cctx) {
          std::cerr << "  cand inter cctx=" << inter_cctx_type << "\n";
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
          if (trace_ && options_.enable_inter_ddt) {
            for (const TxPlan::Part& part : res_y.parts) {
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
        std::cerr << "  cand inter final sse=" << inter_luma_sse
                  << " residual=" << (inter_residual ? 1 : 0) << "\n";
      }
    }

    bool use_intra = keyframe_ || options_.force_intra || !best.valid;
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
      ++intra_blocks_;
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
      entry.palette_index = static_cast<std::uint32_t>(palettes_.size());
      palettes_.push_back(std::move(palette_choice.result));
      // Update the encoder-side §5.20.8.2 cache source only now: an IBC leaf
      // also seen this block would have cleared the palette decision.
      palette::store_palette_state(palette_state_, r, c, w, h,
                                   palettes_.back());
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
                      quant_step_, intra_u, error,
                      qm_enabled_ ? &qm_state_ : nullptr) ||
          !compute_tx(source_, 2, acx0, acy0, cw, ch, intra_v_pred,
                      quant_step_, intra_v, error,
                      qm_enabled_ ? &qm_state_ : nullptr)) {
        error = "palette chroma prediction failed";
        return false;
      }
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
      set_intra_joint(r, c, w4, h4, 0);
      ++intra_blocks_;
      if (trace_) {
        std::cerr << "m2 palette r=" << r << " c=" << c << " " << w << "x" << h
                  << " size=" << palettes_.back().size
                  << " direction=" << palettes_.back().direction
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
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      if (!chroma_part) {
        write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
        write_luma_plan(entry, intra_y, levels_);
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
            levels_.push_back(plans[plane - 1]->levels[scan[
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
      ++intra_blocks_;
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
      if (best.from_temporal) ++temporal_selected_;
      entry.inter_mode = static_cast<std::uint8_t>(best.mode);
      entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
      entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
      entry.compound = best.compound ? 1 : 0;
      entry.ref_index1 = 1;
      entry.compound_joint = 0;
      entry.compound_mode = static_cast<std::uint8_t>(best.compound_mode);
      entry.drl_index1 = static_cast<std::uint8_t>(best.drl_index1);
      if (best.compound) {
        // A compound leaf never carries a residual in the gated subset.
        entry.skip_flag = 1;
        inter_residual = false;
        entry.compound_masked =
            best.compound_mode == 0 ? 0 : (best.compound_type != 2 ? 1 : 0);
        entry.compound_type = static_cast<std::uint8_t>(
            best.compound_mode == 0 ? 2 : best.compound_type);
        entry.wedge_index = static_cast<std::uint8_t>(best.wedge_index);
        entry.wedge_sign = static_cast<std::uint8_t>(best.wedge_sign);
        entry.mask_type = static_cast<std::uint8_t>(best.mask_type);
        entry.cwp_index = static_cast<std::uint8_t>(best.cwp_index);
        entry.cwp_weight = static_cast<std::int8_t>(best.cwp_weight);
        entry.use_optflow = best.use_optflow ? 1 : 0;
        entry.use_refinemv = best.use_refinemv ? 1 : 0;
        if (options_.enable_df_sub_pu &&
            (best.use_optflow || best.use_refinemv)) {
          collect_sub_pus(x0, y0, w, h, best.use_optflow, best.use_refinemv);
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
        entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
        write_luma_plan(entry, res_y, levels_);
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
            levels_.push_back(plans[plane - 1]->levels[scan[
                static_cast<std::size_t>(c2)]]);
          }
        }
      }
      update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv,
                     best.compound ? 1 : merge::kRefNone,
                     best.compound ? &best.mv1 : nullptr);
      set_intra_joint(r, c, w4, h4, 0);  // inter_block_mode_info(): DC
      ++inter_blocks_;
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
      int segment_id = seg_enabled_ ? current_segment_id_ : 0;
      bool inferred = false;
      if (seg_enabled_ && !keyframe_ && entry.is_inter != 0 &&
          entry.skip_flag != 0 && !enc_seg_.has_lossless_segment) {
        segment_id = deltaq::segment_prediction(enc_seg_grid_, r, c);
        inferred = true;
      }
      int qindex = current_qindex_;
      if (seg_enabled_) {
        std::string qp_error;
        if (!deltaq::effective_qindex(
                enc_seg_, base_q_index(), current_qindex_, delta_q_enabled_,
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

    // §6.19.2.3: this leaf's cells are now decoded for the availability of the
    // following leaves in this superblock.
    if (!chroma_part) decoded_[0].fill(c, r, w4, h4);
    if (owns_chroma) {
      decoded_[1].fill(cx0 / 4, cy0 / 4, cw / 4, ch / 4);
      decoded_[2].fill(cx0 / 4, cy0 / 4, cw / 4, ch / 4);
    }
    if (ibc_enabled_) {
      ibcs::finish_ibc_block(ibc_buffer_, ibc_neighbours_, r, c,
                             static_cast<int>(node.size), use_ibc,
                             use_ibc ? ibc_choice.bv : ibc::BlockVector{});
    }
    (void)node;
    return true;
  }

  void update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                      int ref_index, inter::mc::Mv mv,
                      int ref_index1 = merge::kRefNone,
                      const inter::mc::Mv* mv1 = nullptr) {
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
  int tile_mi_row_ = 0;
  int tile_mi_col_ = 0;
  int tile_y_ = 0;
  int tile_w_ = 0;
  int tile_h_ = 0;
  std::span<const std::uint16_t> cdf_seed_;
  std::vector<std::uint16_t>* cdf_final_ = nullptr;
  const GmFramePlan* gm_plan_ = nullptr;
  int quant_step_ = txv::kQuantStep;  // §7.14.2 qlookup(base_q_idx)
  int current_qindex_ = 0;            // §5.20.1 CurrentQIndex for this SB
  int current_segment_id_ = 0;        // §5.20.5.7 SegmentIds for this SB
#if defined(AV2ENC_VIDEO_DELTAQ)
  // §5.18.7.1/.8 segmentation + delta-Q planning for this tile.
  bool seg_enabled_ = false;
  bool delta_q_enabled_ = false;
  rc::SegmentationParams seg_params_{};
  rc::DeltaQParams dq_params_{};
  deltaq::FrameSegmentation enc_seg_{};
  deltaq::SegmentGrid enc_seg_grid_{};
  std::vector<std::uint8_t> sb_segment_;   // variance split: 1 = low variance
  std::vector<int> sb_target_q_;           // delta-Q target per superblock
  std::vector<int> sb_qindex_;             // effective qindex per superblock
  std::vector<rc::DeltaQPlanStep> sb_delta_steps_;  // raster order
  bool primary_ref_is_none_ = true;
#endif
  qm::QmFrameState qm_state_;
  bool qm_enabled_ = false;
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
  // §7.13.3.15 sub-PU rectangles from OPFL/refine-MV prediction.  Only used
  // when --video-df-sub-pu is on; ordinary whole-PU leaves take the normative
  // early return and leave the deblock grid's Max_Tx_Size_Rect values.
  std::vector<post::VideoSubPuRect> sub_pus_;
  inter::mc::RefFrame ref_frame_;
  std::vector<inter::mc::MiInfo> mv_grid_;
  std::vector<std::uint8_t> intra_joint_;
  // §5.20.5.3/.4 IBC + §7.13.3.26 morph state (intra keyframe only).
  bool ibc_enabled_ = false;
  bool plan_morph_ = false;
  ibc::IbcBufferState ibc_buffer_;
  ibcs::BvNeighbourMap ibc_neighbours_;
  ibc::IbcConfig ibc_config_{};
  // §5.20.8 palette state (intra keyframe only).  The state is the same MI
  // cache source the writer's PaletteSyntaxWriter maintains, and the results
  // are handed to the writer through FramePlan::palettes.
  bool palette_enabled_ = false;
  palette::PaletteState palette_state_;
  std::vector<palette::PaletteResult> palettes_;
  std::array<intra_search::BlockDecoded, 3> decoded_;
  std::vector<syntax::BlockSyntax> blocks_;
  // §5.4.3 enable_sdp: the decided luma/chroma trees for the intra keyframe.
  sdpp::FrameDecision sdp_frame_;
  // §8.3.2 SDP plane grids, maintained in decode order while the trees are
  // collected.  Contexts for each node's partition symbols come from these.
  sdpp::MiSizeGrid sdp_luma_sizes_;
  sdpp::MiSizeGrid sdp_chroma_sizes_;
  std::array<int, 7> ref_order_hints_{};
  TemporalFrameContext temporal_ctx_;
  std::vector<std::int32_t> levels_;
  std::uint32_t inter_blocks_ = 0;
  std::uint32_t intra_blocks_ = 0;
  int sb_size_ = 64;
  mutable int temporal_added_ = 0;
  int temporal_selected_ = 0;
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

// ---------------------------------------------------------------------------
// §7.22 saved motion field for an ordinary reference slot.
//
// The decoder's `motion_field_motion_vector_storage()` stores the 8x8 motion
// field of every inter frame when `enable_ref_frame_mvs = 1`; a later frame
// with `use_ref_frame_mvs = 1` projects those fields in §7.9 and consumes the
// result in §7.12.2.7/.8.  This is the same field the TIP path already built
// for its slots, factored out for the P / B / temporal-layer loops.
// ---------------------------------------------------------------------------
struct SlotField {
  bool valid = false;
  int order_hint = 0;
  int w8 = 0;
  int h8 = 0;
  int num_saved_refs = 0;
  int saved_order_hint[7] = {temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint,
                             temporal::kRestrictedOrderHint};
  std::vector<std::int8_t> field_ref;
  std::vector<std::int16_t> field_mv;

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
void init_slot_field(SlotField* slot, int mi_cols, int mi_rows) {
  slot->w8 = mi_cols / 2;
  slot->h8 = mi_rows / 2;
  slot->num_saved_refs = 0;
  for (int i = 0; i < 7; ++i) {
    slot->saved_order_hint[i] = temporal::kRestrictedOrderHint;
  }
  slot->field_ref.assign(static_cast<std::size_t>(slot->w8) *
                             static_cast<std::size_t>(slot->h8) * 2U,
                         static_cast<std::int8_t>(temporal::kRefNone));
  slot->field_mv.assign(static_cast<std::size_t>(slot->w8) *
                            static_cast<std::size_t>(slot->h8) * 4U,
                        0);
}

// Stores one coded frame's §7.22 field from its per-4x4 mode/MV grid.
bool store_slot_field(const InterFrameEncoder& encoder, int order_hint,
                      const int* ref_order_hint, int num_refs,
                      SlotField* slot, std::string& error) {
  const std::vector<inter::mc::MiInfo>& grid = encoder.mv_grid();
  const int mi_cols = encoder.mv_mi_cols();
  const int mi_rows = encoder.mv_mi_rows();
  init_slot_field(slot, mi_cols, mi_rows);
  temporal::RefMvField view;
  view.w8 = slot->w8;
  view.h8 = slot->h8;
  view.ref = slot->field_ref.data();
  view.mv = slot->field_mv.data();
  for (int y8 = 0; y8 < slot->h8; ++y8) {
    for (int x8 = 0; x8 < slot->w8; ++x8) {
      const std::size_t index =
          static_cast<std::size_t>(y8 * 2) *
              static_cast<std::size_t>(mi_cols) +
          static_cast<std::size_t>(x8 * 2);
      const inter::mc::MiInfo& mi = grid[index];
      temporal::MotionFieldStoreEntry entry;
      for (int list = 0; list < 2; ++list) {
        const int ref = mi.ref[list];
        entry.refs[list] =
            (ref >= 0 && ref < num_refs) ? ref : temporal::kRefNone;
        entry.ref_order_hint[list] =
            entry.refs[list] >= 0
                ? ref_order_hint[entry.refs[list]]
                : temporal::kRestrictedOrderHint;
      }
      entry.mv[0] = temporal::Mv{mi.mv[0][0], mi.mv[0][1]};
      entry.mv[1] = temporal::Mv{mi.mv[1][0], mi.mv[1][1]};
      entry.order_hint = order_hint;
      entry.i8 = y8 & 1;
      entry.j8 = x8 & 1;
      if (temporal::store_motion_field_entry(entry, &view, y8, x8) !=
          temporal::TemporalStatus::kOk) {
        error = "§7.22 motion-field store failed";
        return false;
      }
    }
  }
  return true;
}

// §7.9 estimation result plus the arrays it borrows.
struct MotionFieldFrame {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::int32_t> offset;
  std::vector<std::uint8_t> valid;
  temporal::MotionField field;
  temporal::MotionFieldConfig config;
  temporal::MotionFieldResult result;
  bool ready = false;
};

// Runs §7.9 for the current frame from its reference slots' saved fields and
// fills closest_past/closest_future from the §7.8 past/future split.
bool estimate_motion_field(const SlotField* const* ref_fields,
                           const int* ref_slots, const int* ref_order_hint,
                           const std::uint8_t* ref_is_key, bool ref_valid,
                           int num_total_refs, int order_hint, int frame_width,
                           int frame_height, int mi_cols, int mi_rows,
                           int proj_step, MotionFieldFrame* out,
                           std::string& error) {
  out->mi_cols = mi_cols;
  out->mi_rows = mi_rows;
  out->ready = false;
  const std::size_t cells =
      static_cast<std::size_t>(mi_cols / 2) *
      static_cast<std::size_t>(mi_rows / 2);
  out->mv.assign(cells * 2, 0);
  out->offset.assign(cells, 0);
  out->valid.assign(cells, 0);
  out->field.w8 = mi_cols / 2;
  out->field.h8 = mi_rows / 2;
  out->field.mv = out->mv.data();
  out->field.valid = out->valid.data();
  out->field.offset = out->offset.data();

  temporal::MotionFieldConfig& cfg = out->config;
  cfg = temporal::MotionFieldConfig{};
  cfg.mi_cols = mi_cols;
  cfg.mi_rows = mi_rows;
  cfg.proj_step = proj_step;
  cfg.sb_64x64 = proj_step == 1;
  cfg.enable_tip = false;
  cfg.enable_mv_traj = false;
  cfg.reduced_ref_frame_mvs_mode = false;
  cfg.use_bru = false;
  cfg.bru_ref = -1;
  cfg.order_hint = order_hint;
  cfg.num_total_refs = num_total_refs;
  cfg.num_past_refs = 0;
  cfg.num_future_refs = 0;
  cfg.frame_width = frame_width;
  cfg.frame_height = frame_height;

  int closest_past = -1;
  int closest_future = -1;
  int closest_past_hint = 0;
  int closest_future_hint = 0;
  for (int rf = 0; rf < num_total_refs; ++rf) {
    cfg.ref_order_hint[rf] = ref_order_hint[rf];
    cfg.refs[rf].slot = ref_slots[rf];
    cfg.refs[rf].type = ref_is_key[rf] ? temporal::FrameType::kKey
                                       : temporal::FrameType::kInter;
    cfg.refs[rf].width = frame_width;
    cfg.refs[rf].height = frame_height;
    cfg.refs[rf].order_hint = ref_order_hint[rf];
    cfg.refs[rf].num_saved_refs = ref_fields[rf]->num_saved_refs;
    for (int k = 0; k < ref_fields[rf]->num_saved_refs; ++k) {
      cfg.refs[rf].saved_order_hint[k] = ref_fields[rf]->saved_order_hint[k];
    }
    cfg.saved[rf] = ref_fields[rf]->saved();
    const int dist = order_hint - ref_order_hint[rf];
    if (dist > 0) {
      ++cfg.num_past_refs;
      if (closest_past < 0 || ref_order_hint[rf] > closest_past_hint) {
        closest_past = rf;
        closest_past_hint = ref_order_hint[rf];
      }
    } else if (dist < 0) {
      ++cfg.num_future_refs;
      if (closest_future < 0 || ref_order_hint[rf] < closest_future_hint) {
        closest_future = rf;
        closest_future_hint = ref_order_hint[rf];
      }
    }
  }
  if (!ref_valid) {
    // No valid reference slot: the decoder leaves the field empty.
    out->result = temporal::MotionFieldResult{};
    out->ready = true;
    return true;
  }
  const temporal::TemporalStatus status = temporal::motion_field_estimation(
      cfg, &out->field, nullptr, &out->result);
  if (status != temporal::TemporalStatus::kOk) {
    error = "§7.9 motion-field estimation failed";
    return false;
  }
  // §5.18.2: fill_tpl_mvs_sample_gap() runs when TIP is disabled.
  if (temporal::fill_motion_field_sample_gap(cfg, &out->field) !=
      temporal::TemporalStatus::kOk) {
    error = "§7.10.5 motion-field gap fill failed";
    return false;
  }
  out->result.closest_past = closest_past;
  out->result.closest_future = closest_future;
  out->ready = true;
  return true;
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
  config.explicit_ref_frame_map = !options.implicit_ref_map;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
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
  config.enable_ref_frame_mvs = options.enable_ref_frame_mvs;
  config.max_tlayer_id = 0;
  config.monotonic_output_order_flag = false;

  const int mi_cols = 2 * static_cast<int>((width + 7U) >> 3U);
  const int mi_rows = 2 * static_cast<int>((height + 7U) >> 3U);
  std::vector<VideoReconFrame> recon_by_display(
      static_cast<std::size_t>(plan.display_count()));
  std::vector<SlotField> b_fields(static_cast<std::size_t>(plan.num_slots));
  std::vector<bool> b_is_key(static_cast<std::size_t>(plan.num_slots), false);
  VideoOptions frame_options = options;
  frame_options.temporal_layers = 0;
  frame_options.tlayer_drop = -1;

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
      const SlotField* ref_fields[kRefsPerFrame] = {};
      int ref_slots[kRefsPerFrame] = {};
      std::uint8_t ref_is_key[kRefsPerFrame] = {};
      for (int r = 0; r < num_refs; ++r) {
        const int slot = fp->ref_slot[r];
        if (slot < 0 || slot >= plan.num_slots) {
          error = "B-frame reference slot outside the encoder DPB";
          return false;
        }
        ref_fields[r] = &b_fields[static_cast<std::size_t>(slot)];
        ref_slots[r] = slot;
        ref_is_key[r] = b_is_key[static_cast<std::size_t>(slot)] ? 1U : 0U;
      }
      if (!estimate_motion_field(
              ref_fields, ref_slots, ref_order_hints.data(), ref_is_key,
              /*ref_valid=*/true, num_refs, fp->display_order_hint,
              static_cast<int>(width), static_cast<int>(height), mi_cols,
              mi_rows, /*proj_step=*/1, &motion_field, error)) {
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
    if (options.implicit_ref_map) {
      // §7.23: the next frame's derivation reads this frame's slot state.
      if (!ref_bank.refresh(hf.refresh_frame_flags, fp->display_order_hint,
                            static_cast<int>(width), static_cast<int>(height),
                            config.base_q_idx, hf.keyframe, /*tlayer_id=*/0,
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
        if (!store_slot_field(encoder, fp->display_order_hint,
                              ref_order_hints.data(), num_refs, &field,
                              error)) {
          error = "B-frame display " + std::to_string(display) + ": " + error;
          return false;
        }
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
  if (!options.frame_base_q_idx.empty()) {
    error = "--video-tlayers does not compose with per-frame rate control";
    return false;
  }

  tl::PlanConfig pc;
  pc.gop_size = gop;
  pc.schedule = tl::Schedule::kHierarchicalP;
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
      const SlotField* ref_fields[kRefsPerFrame] = {};
      int ref_slots[kRefsPerFrame] = {};
      std::uint8_t ref_is_key[kRefsPerFrame] = {};
      for (int r = 0; r < pf.num_refs; ++r) {
        const int slot = pf.ref_slot[r];
        if (slot < 0 || slot >= plan.num_slots) {
          error = "temporal-layer reference slot outside the encoder DPB";
          return false;
        }
        ref_fields[r] = &t_fields[static_cast<std::size_t>(slot)];
        ref_slots[r] = slot;
        ref_is_key[r] = t_is_key[static_cast<std::size_t>(slot)] ? 1U : 0U;
      }
      if (!estimate_motion_field(
              ref_fields, ref_slots, ref_order_hints.data(), ref_is_key,
              /*ref_valid=*/true, pf.num_refs, pf.display_order_hint,
              static_cast<int>(width), static_cast<int>(height), mi_cols,
              mi_rows, /*proj_step=*/1, &motion_field, error)) {
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
    header.reference_select = false;  // --video-tlayers is single-reference
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    header.use_ref_frame_mvs =
        options.enable_ref_frame_mvs && !pf.is_key;
    apply_video_loop_filter_frame_header(encoder.filter_plan(), header);

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
    if (options.implicit_ref_map) {
      if (!ref_bank.refresh(pf.refresh_frame_flags, pf.display_order_hint,
                            static_cast<int>(width), static_cast<int>(height),
                            config.base_q_idx, pf.is_key, tlayer_id,
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
        if (!store_slot_field(encoder, static_cast<int>(pf.display_order_hint),
                              ref_order_hints.data(), pf.num_refs, &field,
                              error)) {
          error = "temporal-layer frame " + std::to_string(display) + ": " +
                  error;
          return false;
        }
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

}  // namespace

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

  bool wrote_sequence_header = false;
  for (std::size_t i = 0; i < frames.size(); ++i) {
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
    const std::vector<std::uint8_t> obu = make_frame_obu(config, header, tile);
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
  const std::vector<inter::mc::MiInfo>& grid = encoder.mv_grid();
  const int mi_cols = encoder.mv_mi_cols();
  const int mi_rows = encoder.mv_mi_rows();
  init_empty_field(slot, mi_cols, mi_rows);
  temporal::RefMvField view;
  view.w8 = slot->w8;
  view.h8 = slot->h8;
  view.ref = slot->field_ref.data();
  view.mv = slot->field_mv.data();
  for (int y8 = 0; y8 < slot->h8; ++y8) {
    for (int x8 = 0; x8 < slot->w8; ++x8) {
      const std::size_t index =
          static_cast<std::size_t>(y8 * 2) *
              static_cast<std::size_t>(mi_cols) +
          static_cast<std::size_t>(x8 * 2);
      const inter::mc::MiInfo& mi = grid[index];
      temporal::MotionFieldStoreEntry entry;
      for (int list = 0; list < 2; ++list) {
        const int ref = mi.ref[list];
        entry.refs[list] =
            (ref >= 0 && ref < num_refs) ? ref : temporal::kRefNone;
        entry.ref_order_hint[list] =
            entry.refs[list] >= 0
                ? ref_order_hint[entry.refs[list]]
                : temporal::kRestrictedOrderHint;
      }
      entry.mv[0] = temporal::Mv{mi.mv[0][0], mi.mv[0][1]};
      entry.mv[1] = temporal::Mv{mi.mv[1][0], mi.mv[1][1]};
      entry.order_hint = order_hint;
      entry.i8 = y8 & 1;
      entry.j8 = x8 & 1;
      if (temporal::store_motion_field_entry(entry, &view, y8, x8) !=
          temporal::TemporalStatus::kOk) {
        error = "§7.22 motion-field store failed";
        return false;
      }
    }
  }
  return true;
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

  std::array<TipSlotState, kMaxRefSlots> slots;

  // --- Key frame -----------------------------------------------------------
  VideoFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.ref_slot = 0;
  key_header.refresh_frame_flags = (1U << num_ref_frames) - 1U;

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

    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu =
        make_frame_obu(config, header, payload);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
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
  std::vector<std::int16_t> pred_scratch(2U * 16U * 16U, 0);
  std::vector<std::uint8_t> block_scratch(16U * 16U, 0);
  for (int i = 1; i < frame_count - 1; ++i) {
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
  return true;
}

bool encode_video_sequence(std::span<const VideoInputFrame> frames,
                        const VideoOptions& options, VideoSequenceResult& result,
                        std::string& error) {
  if (options.sb128 && options.sb256) {
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
    if (options.adaptive_cdf || options.avg_cdf || options.cdf_continuity) {
      error = "--video-ext-partitions does not compose with the CDF "
              "image/continuity tools yet";
      return false;
    }
    if (options.loop_filters.any()) {
      error = "--video-ext-partitions does not compose with the loop filters "
              "yet; deblocking is supported";
      return false;
    }
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
        (options.enable_segmentation || options.per_segment_q)) {
      error = "--video-ibc is not wired with "
              "--video-segmentation/--video-per-segment-q";
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
  // Non-default formats use the depth/chroma-generic core.  The historical  // 8-bit 4:2:0 path below is untouched.
  if (options.format.chroma != fmt::ChromaFormat::k420 ||
      options.format.depth != fmt::BitDepth::k8) {
    if (options.implicit_ref_map) {
      error = "the implicit reference map is wired for the 8-bit 4:2:0 video "
              "core only; the format cores keep the explicit map";
      return false;
    }
    return encode_video_sequence_format(frames, options, result, error);
  }
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
    return options.frame_base_q_idx.empty() ? options.base_q_idx
                                            : options.frame_base_q_idx[index];
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
  // plain single-tile temporal units.  The frame-type wrappers and the tiled
  // writer have their own header writers and are rejected by name.
  if (options.film_grain) {
    if (options.tile_cols > 0 || options.tile_rows > 0) {
      error = "--video-grain does not compose with --tiles";
      return false;
    }
    if (options.b_frames > 0 || options.temporal_layers > 0 ||
        options.long_term || options.olk || options.leading_frames > 0 ||
        options.sef || options.ra_switch || options.ras ||
        options.enable_tip) {
      error = "--video-grain does not compose with the B/temporal-layer/"
              "random-access/TIP frame schedules";
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
                        options.per_segment_q;
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
  const int num_ref_frames = av2enc::video::resolved_num_ref_frames(options);
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
  config.use_128x128_superblock = options.sb128;
  config.use_256x256_superblock = options.sb256;
  config.enable_ext_partitions = options.ext_partitions;
  config.enable_uneven_4way_partitions = options.ext_partitions;
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = order_hint_bits;
  config.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  config.explicit_ref_frame_map = !options.implicit_ref_map;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_masked_compound =
      options.compound && (options.compound_wedge || options.compound_diffwtd);
  config.enable_cwp = options.enable_cwp_weighted;
  config.enable_imp_msk_bld = options.enable_imp_msk_bld;
  config.enable_df_sub_pu = options.enable_df_sub_pu;
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
  // §5.4.7/§5.20.5.4: the screen-content carriers select the per-frame form
  // and code the IBC/palette mode-info path on the intra keyframe.
  config.seq_screen_content_tools_present =
      options.enable_morph || options.enable_palette || options.enable_ibc;
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

  VideoFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.ref_slot = 0;
  key_header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (!build_video_frame_qp(key_options,
                            static_cast<int>(key_options.base_q_idx),
                            /*primary_ref_is_none=*/true, key_header,
                            error)) {
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
  if (config.seq_screen_content_tools_present) {
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
  std::vector<std::unique_ptr<InterFrameEncoder>> key_encoders;
  key_encoders.reserve(tile_sources.size());
  for (CodedTile& tile : tile_sources) {
    if (cdf_chain) cdf_tile = cdf_seed;
    auto encoder = std::make_unique<InterFrameEncoder>(
        frames[0], no_refs, no_distance, 1, key_options, options.trace,
        /*keyframe=*/true, tile.x, tile.y, tile.w, tile.h,
        cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                  : std::span<const std::uint16_t>(),
        cdf_chain ? &cdf_scratch : nullptr);
    if (!encoder->decide(error) || !encoder->deblock(error)) {
      error = "keyframe encode failed: " + error;
      return false;
    }
    paste_tile(encoder->recon(0), frame_recon[0], tile.x, tile.y, tile.w,
               tile.h);
    paste_tile(encoder->recon(1), frame_recon[1], tile.x / 2, tile.y / 2,
               tile.w / 2, tile.h / 2);
    paste_tile(encoder->recon(2), frame_recon[2], tile.x / 2, tile.y / 2,
               tile.w / 2, tile.h / 2);
    key_intra_blocks += encoder->intra_blocks();
    key_encoders.push_back(std::move(encoder));
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
    for (int plane = 0; plane < 3; ++plane) {
      lf.plane[plane] = frame_recon[plane].samples.data();
      lf.stride[plane] = frame_recon[plane].stride;
      lf.source[plane] = plane == 0 ? frames[0].y.data() : (plane == 1 ? frames[0].u.data() : frames[0].v.data());
      lf.source_stride[plane] =
          plane == 0 ? static_cast<int>(width) : static_cast<int>(width / 2);
    }
    // §7.18.1 Skips[] / §7.20.1 LrTxSkip for the screen-content keyframe.  An
    // IBC skip leaf codes skip_flag = 1, and the real decoder leaves such a
    // block out of CDEF and LR; ordinary intra leaves (including palette
    // leaves, which code eob = 0) contribute nothing to Skips[].  Only build
    // the list on a screen-content carrier so every historical keyframe keeps
    // the previous all-zero grid.
    if (config.seq_screen_content_tools_present) {
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
    key_unit.reserve(td.size() + seq.size() + qm_obu.size() + ci.size() +
                     mfh_obu.size() + obu.size());
    key_unit.insert(key_unit.end(), td.begin(), td.end());
    key_unit.insert(key_unit.end(), seq.begin(), seq.end());
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
    if (!ref_bank.refresh(key_header.refresh_frame_flags,
                          key_header.order_hint, static_cast<int>(width),
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
    std::vector<inter::mc::RefFrame> references;
    references.reserve(static_cast<std::size_t>(ref_count));
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < ref_count; ++r) {
      const int slot = frame_ref_slots[static_cast<std::size_t>(r)];
      references.push_back(make_reference(slot));
      frame_distance[static_cast<std::size_t>(r)] =
          options.implicit_ref_map
              ? derived_distance[static_cast<std::size_t>(r)]
              : relative_dist(order_hint, slots[slot].order_hint,
                              order_hint_bits);
    }

    const VideoOptions frame_options = options_for_frame(i);
    // §5.14: plan this picture's grain model in coding order before the frame
    // header is written; a changed model emits one film grain OBU.
    grain::GrainFramePlan grain_plan;
    if (options.film_grain &&
        !plan_grain_frame(frames[i], static_cast<std::uint32_t>(i),
                          grain_planner, grain_plan, error)) {
      error = "P-frame " + std::to_string(i) + " film grain: " + error;
      return false;
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
        const int primary_slot = frame_ref_slots.front();
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
    for (int r = 0; r < ref_count; ++r) {
      ref_order_hints[static_cast<std::size_t>(r)] = static_cast<int>(
          slots[frame_ref_slots[static_cast<std::size_t>(r)]].order_hint);
    }
    // §7.9 + §7.12.2.7/.8: project this frame's reference motion fields into
    // the current frame's motion field for the temporal MV candidates.
    TemporalFrameContext temporal_ctx;
    MotionFieldFrame motion_field;
    if (options.enable_ref_frame_mvs) {
      const SlotField* ref_fields[kRefsPerFrame] = {};
      int ref_slots[kRefsPerFrame] = {};
      std::uint8_t ref_is_key[kRefsPerFrame] = {};
      for (int r = 0; r < ref_count; ++r) {
        const int slot = recent_slots[static_cast<std::size_t>(r)];
        ref_fields[r] = &slot_fields[static_cast<std::size_t>(slot)];
        ref_slots[r] = slot;
        ref_is_key[r] = slots[slot].is_key ? 1U : 0U;
      }
      if (!estimate_motion_field(
              ref_fields, ref_slots, ref_order_hints.data(), ref_is_key,
              /*ref_valid=*/true, ref_count, static_cast<int>(order_hint),
              static_cast<int>(width), static_cast<int>(height),
              recon_mi_cols, recon_mi_rows, /*proj_step=*/1, &motion_field,
              error)) {
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
    }
    std::vector<std::unique_ptr<InterFrameEncoder>> tile_encoders;
    tile_encoders.reserve(tile_sources.size());
    for (CodedTile& tile : tile_sources) {
      if (cdf_chain) cdf_tile = cdf_seed;
      auto encoder = std::make_unique<InterFrameEncoder>(
          frames[i], references, frame_distance, ref_count, frame_options,
          options.trace, /*keyframe=*/false, tile.x, tile.y, tile.w, tile.h,
          cdf_chain ? std::span<const std::uint16_t>(cdf_tile)
                    : std::span<const std::uint16_t>(),
          cdf_chain ? &cdf_scratch : nullptr, gm_ptr, ref_order_hints,
          /*primary_ref_is_none=*/i == 1, &temporal_ctx);      if (!encoder->decide(error) || !encoder->deblock(error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
      inter_blocks += encoder->inter_blocks();
      intra_blocks += encoder->intra_blocks();
      paste_tile(encoder->recon(0), frame_recon[0], tile.x, tile.y, tile.w,
                 tile.h);
      paste_tile(encoder->recon(1), frame_recon[1], tile.x / 2, tile.y / 2,
                 tile.w / 2, tile.h / 2);
      paste_tile(encoder->recon(2), frame_recon[2], tile.x / 2, tile.y / 2,
                 tile.w / 2, tile.h / 2);
      tile_encoders.push_back(std::move(encoder));
    }
    for (const std::unique_ptr<InterFrameEncoder>& encoder : tile_encoders) {
      result.temporal_candidates +=
          static_cast<std::uint64_t>(encoder->temporal_added());
      result.temporal_selected +=
          static_cast<std::uint64_t>(encoder->temporal_selected());
    }
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
      for (int plane = 0; plane < 3; ++plane) {
        lf.plane[plane] = frame_recon[plane].samples.data();
        lf.stride[plane] = frame_recon[plane].stride;
        lf.source[plane] =
            plane == 0 ? frames[i].y.data()
                       : (plane == 1 ? frames[i].u.data() : frames[i].v.data());
        lf.source_stride[plane] =
            plane == 0 ? static_cast<int>(width)
                       : static_cast<int>(width / 2);
      }
      if (!post::run_video_loop_filters(lf, options.loop_filters,
                                        &frame_filters, nullptr, &error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
    }
    for (std::size_t t = 0; t < tile_sources.size(); ++t) {
      CodedTile& tile = tile_sources[t];
      InterFrameEncoder& encoder = *tile_encoders[t];
      if (cdf_chain) cdf_tile = cdf_seed;
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
                               : static_cast<int>(i % num_ref_frames);
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
          static_cast<std::uint8_t>(frame_ref_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    header.frame_explicit_ref_frame_map = !options.implicit_ref_map;
    header.refresh_frame_flags = 1U << write_slot;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
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
                              /*primary_ref_is_none=*/i == 1, header, error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }
#endif

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
      tiled_header.tx_mode_select = options.enable_tx_search;
      tiled_header.cdef = post::make_tiled_cdef_params(header.cdef);
      tiled_header.ccso = post::make_tiled_ccso_params(header.ccso);
      tiled_header.lr = header.lr;
      tiled_header.gdf = header.gdf;
      tiled_header.cdef_frame_enable = header.cdef.enabled;
      tiled_header.ccso_frame_flag = header.ccso.enabled;
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

    // Store the reconstruction and refresh the rotating slot.
    slots[write_slot].valid = true;
    slots[write_slot].is_key = false;
    slots[write_slot].order_hint = order_hint;
    for (int plane = 0; plane < 3; ++plane) {
      slots[write_slot].plane[plane] = frame_recon[plane];
    }
    // §7.23: the next frame's implicit mapping must see this frame exactly as
    // the decoder stores it.
    if (options.implicit_ref_map &&
        !ref_bank.refresh(header.refresh_frame_flags,
                          static_cast<int>(order_hint),
                          static_cast<int>(width), static_cast<int>(height),
                          frame_qp(i), /*key_frame=*/false, 0, 0,
                          header.implicit_output_frame,
                          header.immediate_output_frame, -1, &error)) {
      error = "P-frame " + std::to_string(i) + " reference bank: " + error;
      return false;
    }
    // §7.22/§7.23: save this frame's motion field and reference order hints
    // for later §7.9 estimation.
    if (options.enable_ref_frame_mvs) {
      SlotField& field = slot_fields[static_cast<std::size_t>(write_slot)];
      if (!store_slot_field(*tile_encoders[0], static_cast<int>(order_hint),
                            ref_order_hints.data(), ref_count, &field,
                            error)) {
        error = "P-frame " + std::to_string(i) + ": " + error;
        return false;
      }
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
