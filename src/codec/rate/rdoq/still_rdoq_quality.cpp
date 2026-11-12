// M1 coefficient RDOQ measurement harness.
//
// Replays the M1 still-picture encoder's luma/chroma decisions *without*
// modifying src/encoder/m1/:
//
//   * the partition tree comes from the M1 GPU plan component
//     (src/cuda/m1/plan/), which is documented and tested to match
//     partition::decide_frame in still_encoder.cpp;
//   * luma uses the M1 default DC intra mode;
//   * chroma uses the M1 default DC / CfL decision;
//   * the residual, transform, quantisation and reconstruction use the same
//     tx_variants arithmetic as M1, with -ffp-contract=off;
//   * coefficient contexts (txb_skip / DC sign / cumulative levels) are tracked
//     exactly as the §8.3 writer derives them.
//
// The replay is checked against a real `av2enc::still::encode_still_picture` run when
// --check-reference is given (deblocking disabled on both sides; deblocking
// does not affect coefficients).
//
// For every block the harness compares:
//   scalar quantisation   vs   the full trellis in src/encoder/rdoq/
// using the §9.3 CDF rate model in rdoq_av2_model.h and lambda from the RDO
// framework.  It reports coefficient SSE, reconstructed-pixel SSE (with the
// prediction held fixed), rate and RD cost.  It does not write a bitstream and
// does not claim a decoder-gated BD-rate result.

#include "cuda/pipeline/still/plan/plan.h"
#include "codec/intra/cfl/cfl.h"
#include "codec/intra/modes/intra_modes.h"
#include "pipeline/still/still_headers.h"
#include "pipeline/still/still_encoder.h"
#include "pipeline/still/still_syntax.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/variants/tx_variants.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using av2enc::intra::IntraEdges;
using av2enc::intra::Mode;
using av2enc::intra::NeighborRequest;
using av2enc::intra::PlaneView;
using av2enc::intra::PredictParams;
using av2enc::rdoq::Coefficient;
using av2enc::rdoq::Config;
using av2enc::rdoq::RdoqResult;
using av2enc::txv::TxParams;

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
int clip1(int value) { return clip3(0, 255, value); }

struct Options {
  std::string input;
  int qp = 90;
  int depth = 4;
  int min_block_size = 8;
  bool enable_cfl = true;
  bool enable_edge_filter = true;
  double sse_scale = 0.125;
  std::int32_t prev_level_cap = 2;
  std::int32_t level_cap = 0;
  bool check_reference = false;
  bool per_block_csv = false;
  std::string write_ivf_prefix;  // empty = do not write streams
  int max_blocks = 0;  // 0 = all
  int plane_mask = 7;  // bit 0 Y, bit 1 U, bit 2 V
};

struct Plane {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> samples;

  void reset(int w, int h) {
    width = w;
    height = h;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  }
  std::uint8_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
};

struct PlaneRecord {
  int tx_size = 0;
  int scalar_eob = 0;
  int rdoq_eob = 0;
  std::vector<std::int32_t> scalar_levels;  // scan order, [0, scalar_eob)
  std::vector<std::int32_t> rdoq_levels;    // scan order, [0, rdoq_eob)
};

struct LeafRecord {
  bool is_cfl = false;
  PlaneRecord planes[3];
};

struct Stats {
  std::int64_t blocks = 0;
  std::int64_t coefficients = 0;
  std::int64_t scalar_nonzeros = 0;
  std::int64_t rdoq_nonzeros = 0;
  std::int64_t scalar_eob_sum = 0;
  std::int64_t rdoq_eob_sum = 0;
  std::int64_t changed_blocks = 0;
  std::int64_t samples = 0;
  double scalar_coeff_sse = 0.0;
  double rdoq_coeff_sse = 0.0;
  double scalar_pixel_sse = 0.0;
  double rdoq_pixel_sse = 0.0;
  double scalar_bits = 0.0;
  double rdoq_bits = 0.0;
  double scalar_cost = 0.0;
  double rdoq_cost = 0.0;
  double max_abs_cost_delta = 0.0;
  double max_pixel_delta = 0.0;

  void add(const Stats& other) {
    blocks += other.blocks;
    coefficients += other.coefficients;
    scalar_nonzeros += other.scalar_nonzeros;
    rdoq_nonzeros += other.rdoq_nonzeros;
    scalar_eob_sum += other.scalar_eob_sum;
    rdoq_eob_sum += other.rdoq_eob_sum;
    changed_blocks += other.changed_blocks;
    samples += other.samples;
    scalar_coeff_sse += other.scalar_coeff_sse;
    rdoq_coeff_sse += other.rdoq_coeff_sse;
    scalar_pixel_sse += other.scalar_pixel_sse;
    rdoq_pixel_sse += other.rdoq_pixel_sse;
    scalar_bits += other.scalar_bits;
    rdoq_bits += other.rdoq_bits;
    scalar_cost += other.scalar_cost;
    rdoq_cost += other.rdoq_cost;
    max_abs_cost_delta = std::max(max_abs_cost_delta, other.max_abs_cost_delta);
    max_pixel_delta = std::max(max_pixel_delta, other.max_pixel_delta);
  }
};

class Replay {
 public:
  Replay(const av2enc::still::StillPicture& picture, const Options& options,
         const av2enc::still::plan::FramePlan& plan, bool run_rdoq,
         bool use_rdoq_recon)
      : picture_(picture),
        options_(options),
        plan_(plan),
        luma_width_(static_cast<int>(picture.width)),
        luma_height_(static_cast<int>(picture.height)),
        chroma_width_(static_cast<int>((picture.width + 1U) / 2U)),
        chroma_height_(static_cast<int>((picture.height + 1U) / 2U)),
        run_rdoq_(run_rdoq),
        use_rdoq_recon_(use_rdoq_recon) {
    recon_[0].reset(plan.luma_stride, plan.luma_padded_height);
    recon_[1].reset(plan.chroma_stride, plan.chroma_padded_height);
    recon_[2].reset(plan.chroma_stride, plan.chroma_padded_height);
    for (int plane = 0; plane < 3; ++plane) {
      const int cols = plane == 0 ? plan.mi_cols : plan.mi_cols / 2;
      const int rows = plane == 0 ? plan.mi_rows : plan.mi_rows / 2;
      above_level_[plane].assign(static_cast<std::size_t>(cols) + 16, 0);
      left_level_[plane].assign(static_cast<std::size_t>(rows) + 16, 0);
      above_dc_[plane].assign(static_cast<std::size_t>(cols) + 16, 0);
      left_dc_[plane].assign(static_cast<std::size_t>(rows) + 16, 0);
    }
  }

  bool run(Stats* luma, Stats* u, Stats* v, std::string* error);

  const Plane& recon(int plane) const { return recon_[plane]; }
  const std::vector<LeafRecord>& records() const { return records_; }

 private:
  std::uint8_t source(int plane, int x, int y) const {
    const std::span<const std::uint8_t> data =
        plane == 0 ? picture_.y : (plane == 1 ? picture_.u : picture_.v);
    const int w = plane == 0 ? luma_width_ : chroma_width_;
    const int h = plane == 0 ? luma_height_ : chroma_height_;
    const int cx = clip3(0, w - 1, x);
    const int cy = clip3(0, h - 1, y);
    return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(cx)];
  }

  bool predict(int plane, int x0, int y0, int w, int h, bool have_left,
               bool have_above, int num4_above_right, int num4_below_left,
               std::vector<std::uint8_t>& out) {
    NeighborRequest request;
    request.recon = PlaneView{recon_[plane].samples.data(),
                              recon_[plane].width};
    request.x = x0;
    request.y = y0;
    request.w = w;
    request.h = h;
    request.max_x = (plane == 0 ? plan_.mi_cols * 4 : plan_.mi_cols * 2) - 1;
    request.max_y = (plane == 0 ? plan_.mi_rows * 4 : plan_.mi_rows * 2) - 1;
    request.have_left = have_left;
    request.have_above = have_above;
    request.num4_above_right = num4_above_right;
    request.num4_below_left = num4_below_left;
    request.color_plane = plane;
    IntraEdges edges;
    if (av2enc::intra::prepare_edges(request, 8, edges) !=
        av2enc::intra::IntraStatus::kOk) {
      return false;
    }
    PredictParams params;
    params.mode = Mode::kDc;
    params.angle_delta = 0;
    params.enable_edge_filter = options_.enable_edge_filter;
    params.above_neighbor_is_smooth = false;
    params.left_neighbor_is_smooth = false;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    return av2enc::intra::predict_intra(edges, params, out.data(),
                                        static_cast<std::size_t>(w)) ==
           av2enc::intra::IntraStatus::kOk;
  }

  int above_or(int plane, int x4, int w4) const {
    const int cols = plane == 0 ? plan_.mi_cols : plan_.mi_cols / 2;
    const int limit = plane == 0 ? plan_.mi_cols : plan_.mi_cols / 2;
    (void)limit;
    int value = 0;
    for (int i = 0; i < w4; ++i) {
      if (x4 + i < cols) {
        value |= above_level_[plane][static_cast<std::size_t>(x4 + i)];
        value |= above_dc_[plane][static_cast<std::size_t>(x4 + i)];
      }
    }
    return value;
  }

  int left_or(int plane, int y4, int h4) const {
    const int rows = plane == 0 ? plan_.mi_rows : plan_.mi_rows / 2;
    int value = 0;
    for (int i = 0; i < h4; ++i) {
      if (y4 + i < rows) {
        value |= left_level_[plane][static_cast<std::size_t>(y4 + i)];
        value |= left_dc_[plane][static_cast<std::size_t>(y4 + i)];
      }
    }
    return value;
  }

  int txb_skip_context(int plane, int x4, int y4, int w4, int h4,
                       int bsize_w, int bsize_h, int coded_w, int coded_h,
                       int eob_u) const {
    if (plane == 0) {
      int top = 0;
      int left = 0;
      const int cols = plan_.mi_cols;
      const int rows = plan_.mi_rows;
      for (int i = 0; i < w4; ++i) {
        if (x4 + i < cols) {
          top |= above_level_[0][static_cast<std::size_t>(x4 + i)];
        }
      }
      for (int i = 0; i < h4; ++i) {
        if (y4 + i < rows) {
          left |= left_level_[0][static_cast<std::size_t>(y4 + i)];
        }
      }
      top = std::min(top, 4);
      left = std::min(left, 4);
      if (bsize_w == coded_w && bsize_h == coded_h) {
        return 0;
      }
      return (top + left + 3) >> 1;
    }
    const int above = above_or(plane, x4, w4) != 0 ? 1 : 0;
    const int left = left_or(plane, y4, h4) != 0 ? 1 : 0;
    int ctx = above + left;
    if (plane == 2) {
      if (bsize_w * bsize_h > coded_w * coded_h) ctx += 3;
      if (eob_u != 0) ctx += 6;
    } else {
      ctx += 6;
    }
    return ctx;
  }

  int dc_sign_context(int x4, int y4, int w4, int h4) const {
    int dc_sign = 0;
    for (int i = 0; i < w4; ++i) {
      if (x4 + i < plan_.mi_cols) {
        const int sign = above_dc_[0][static_cast<std::size_t>(x4 + i)];
        if (sign == 1) --dc_sign;
        if (sign == 2) ++dc_sign;
      }
    }
    for (int i = 0; i < h4; ++i) {
      if (y4 + i < plan_.mi_rows) {
        const int sign = left_dc_[0][static_cast<std::size_t>(y4 + i)];
        if (sign == 1) --dc_sign;
        if (sign == 2) ++dc_sign;
      }
    }
    if (dc_sign < 0) return 1;
    if (dc_sign > 0) return 2;
    return 0;
  }

  void update_context(int plane, int x4, int y4, int w4, int h4, int eob,
                      std::span<const std::int32_t> levels_raster,
                      std::span<const std::int32_t> scan) {
    int cul_level = 0;
    int dc_category = 0;
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const std::int32_t level = levels_raster[static_cast<std::size_t>(pos)];
      const int abs_value = std::abs(level);
      if (pos == 0 && abs_value > 0) {
        dc_category = level < 0 ? 1 : 2;
      }
      cul_level = std::min(4, cul_level + abs_value);
    }
    const int cols = plane == 0 ? plan_.mi_cols : plan_.mi_cols / 2;
    const int rows = plane == 0 ? plan_.mi_rows : plan_.mi_rows / 2;
    for (int i = 0; i < w4; ++i) {
      if (x4 + i < cols) {
        above_level_[plane][static_cast<std::size_t>(x4 + i)] =
            static_cast<std::uint8_t>(cul_level);
        above_dc_[plane][static_cast<std::size_t>(x4 + i)] =
            static_cast<std::uint8_t>(dc_category);
      }
    }
    for (int i = 0; i < h4; ++i) {
      if (y4 + i < rows) {
        left_level_[plane][static_cast<std::size_t>(y4 + i)] =
            static_cast<std::uint8_t>(cul_level);
        left_dc_[plane][static_cast<std::size_t>(y4 + i)] =
            static_cast<std::uint8_t>(dc_category);
      }
    }
  }

  bool code_plane(int plane, std::int32_t x0, std::int32_t y0,
                  std::int32_t coded_w, std::int32_t coded_h,
                  std::int32_t bsize_w, std::int32_t bsize_h,
                  int eob_u, std::span<const std::uint8_t> pred, Stats* stats,
                  PlaneRecord* record, std::int32_t* eob_out,
                  std::string* error);

  const av2enc::still::StillPicture& picture_;
  const Options& options_;
  const av2enc::still::plan::FramePlan& plan_;
  int luma_width_;
  int luma_height_;
  int chroma_width_;
  int chroma_height_;
  Plane recon_[3];
  std::vector<std::uint8_t> above_level_[3];
  std::vector<std::uint8_t> left_level_[3];
  std::vector<std::uint8_t> above_dc_[3];
  std::vector<std::uint8_t> left_dc_[3];
  int block_counter_ = 0;
  bool run_rdoq_ = true;
  bool use_rdoq_recon_ = false;
  std::vector<LeafRecord> records_;
};

// Codes one transform block: scalar quantisation, optional RDOQ, cost
// accumulation, and reconstruction.  The prediction is fixed; reconstruction
// is used only to measure the block-local SSE, not to feed later predictions.
bool Replay::code_plane(int plane, std::int32_t x0, std::int32_t y0,
                        std::int32_t coded_w, std::int32_t coded_h,
                        std::int32_t bsize_w, std::int32_t bsize_h,
                        int eob_u, std::span<const std::uint8_t> pred,
                        Stats* stats, PlaneRecord* record,
                        std::int32_t* eob_out, std::string* error) {
  const auto params_opt =
      av2enc::txv::make_tx_params(coded_w, coded_h,
                                  av2enc::txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    *error = "unsupported transform size";
    return false;
  }
  const TxParams& params = *params_opt;
  const int area = params.w * params.h;
  const int coded_pels = coded_w * coded_h;

  std::vector<std::int16_t> residual_coded(
      static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < coded_h; ++y) {
    for (int x = 0; x < coded_w; ++x) {
      const int src = source(plane, x0 + x, y0 + y);
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(coded_w) +
                         static_cast<std::size_t>(x)];
      residual_coded[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(coded_w) +
                     static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(src - p);
    }
  }
  std::vector<std::int16_t> residual_adj(static_cast<std::size_t>(area));
  av2enc::txv::downsample_residual(residual_coded.data(), params,
                                   residual_adj.data());
  std::vector<double> coeff(static_cast<std::size_t>(area));
  std::vector<std::int32_t> scalar_raster(static_cast<std::size_t>(area));
  av2enc::txv::forward_transform(residual_adj.data(), params, coeff.data(),
                                 scalar_raster.data());
  std::vector<std::int32_t> scan(static_cast<std::size_t>(area));
  av2enc::rdoq::av2::build_scan(params.w, params.h, scan);
  std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area));
  std::vector<double> coeff_scan(static_cast<std::size_t>(area));
  for (int c = 0; c < area; ++c) {
    const std::size_t pos = static_cast<std::size_t>(scan[static_cast<std::size_t>(c)]);
    scalar_scan[static_cast<std::size_t>(c)] = scalar_raster[pos];
    coeff_scan[static_cast<std::size_t>(c)] = coeff[pos];
  }
  std::int32_t scalar_eob = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (scalar_scan[static_cast<std::size_t>(c)] != 0) {
      scalar_eob = c + 1;
      break;
    }
  }

  // Contexts must be sampled before this block updates them.
  const int x4 = plane == 0 ? static_cast<int>(x0) / 4
                            : static_cast<int>(x0) / 4;
  const int y4 = plane == 0 ? static_cast<int>(y0) / 4
                            : static_cast<int>(y0) / 4;
  const int w4 = params.coded_w / 4;
  const int h4 = params.coded_h / 4;
  const int skip_ctx =
      txb_skip_context(plane, x4, y4, w4, h4, bsize_w, bsize_h,
                       params.coded_w, params.coded_h, eob_u);
  const int dc_ctx = plane == 0 ? dc_sign_context(x4, y4, w4, h4) : 0;

  if ((options_.plane_mask & (1 << plane)) == 0) {
    // Still reconstruct and update the decoded-neighbour context so later
    // blocks see the scalar path, but do not run the RDOQ comparison.
    std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
    std::vector<std::int16_t> recon_residual(
        static_cast<std::size_t>(coded_pels));
    av2enc::txv::inverse_transform(scalar_raster.data(), params,
                                   dequant.data(), recon_residual.data());
    for (int y = 0; y < coded_h; ++y) {
      const int py = y0 + y;
      if (py < 0 || py >= recon_[plane].height) continue;
      for (int x = 0; x < coded_w; ++x) {
        const int px = x0 + x;
        if (px < 0 || px >= recon_[plane].width) continue;
        const int p = pred[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(coded_w) +
                           static_cast<std::size_t>(x)];
        const int r = recon_residual[static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(coded_w) +
                                     static_cast<std::size_t>(x)];
        recon_[plane].at(px, py) = static_cast<std::uint8_t>(clip1(p + r));
      }
    }
    update_context(plane, x4, y4, w4, h4, scalar_eob, scalar_raster, scan);
    if (record != nullptr) {
      record->tx_size = params.tx_size;
      record->scalar_eob = scalar_eob;
      record->rdoq_eob = scalar_eob;
      record->scalar_levels.assign(scalar_scan.begin(),
                                   scalar_scan.begin() + scalar_eob);
      record->rdoq_levels = record->scalar_levels;
    }
    *eob_out = scalar_eob;
    return true;
  }

  av2enc::rdoq::av2::BlockInfo info;
  info.plane = plane;
  info.params = params;
  info.txb_skip_context = skip_ctx;
  info.dc_sign_context = dc_ctx;
  av2enc::rdoq::av2::Model model;
  if (!model.init(info, scalar_scan, error)) {
    return false;
  }

  const double quant_step = 1.0 / params.quant_factor;
  const double lambda =
      av2enc::rdo::lambda_sse_from_quant_step(quant_step, options_.sse_scale);

  // Scalar baseline cost under the same model.
  double scalar_coeff_sse = 0.0;
  for (int c = 0; c < area; ++c) {
    const double reconstructed =
        static_cast<double>(scalar_scan[static_cast<std::size_t>(c)]) *
        quant_step;
    const double difference =
        coeff_scan[static_cast<std::size_t>(c)] - reconstructed;
    scalar_coeff_sse += difference * difference;
  }
  const double scalar_bits = model.scalar_bits();
  const double scalar_cost = scalar_coeff_sse + lambda * scalar_bits;

  std::vector<Coefficient> coefficients(static_cast<std::size_t>(area));
  for (int c = 0; c < area; ++c) {
    coefficients[static_cast<std::size_t>(c)].original =
        coeff_scan[static_cast<std::size_t>(c)];
    coefficients[static_cast<std::size_t>(c)].scalar_level =
        scalar_scan[static_cast<std::size_t>(c)];
    coefficients[static_cast<std::size_t>(c)].level_context = c;
  }
  Config config;
  config.lambda = lambda;
  config.quant_step = quant_step;
  config.prev_level_cap = options_.prev_level_cap;
  config.level_cap = options_.level_cap;
  RdoqResult rdoq;
  if (run_rdoq_) {
    if (!av2enc::rdoq::rdoq_quantize(coefficients, config, model.rate_model(),
                                     rdoq, error)) {
      return false;
    }
  } else {
    rdoq.levels = scalar_scan;
    rdoq.eob = scalar_eob;
    rdoq.distortion = scalar_coeff_sse;
    rdoq.rate_bits = scalar_bits;
    rdoq.cost = scalar_cost;
  }

  // Reconstruct both variants with the same prediction for block-local SSE.
  std::vector<std::int32_t> scalar_dequant(static_cast<std::size_t>(area));
  std::vector<std::int16_t> scalar_residual(
      static_cast<std::size_t>(coded_pels));
  av2enc::txv::inverse_transform(scalar_raster.data(), params,
                                 scalar_dequant.data(),
                                 scalar_residual.data());
  std::vector<std::int32_t> rdoq_raster(static_cast<std::size_t>(area), 0);
  for (int c = 0; c < rdoq.eob; ++c) {
    rdoq_raster[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] =
        rdoq.levels[static_cast<std::size_t>(c)];
  }
  std::vector<std::int32_t> rdoq_dequant(static_cast<std::size_t>(area));
  std::vector<std::int16_t> rdoq_residual(static_cast<std::size_t>(coded_pels));
  av2enc::txv::inverse_transform(rdoq_raster.data(), params,
                                 rdoq_dequant.data(), rdoq_residual.data());

  const bool use_rdoq = use_rdoq_recon_ && run_rdoq_;
  const std::vector<std::int32_t>& chosen_raster =
      use_rdoq ? rdoq_raster : scalar_raster;
  const std::vector<std::int16_t>& chosen_residual =
      use_rdoq ? rdoq_residual : scalar_residual;
  const std::int32_t chosen_eob = use_rdoq ? rdoq.eob : scalar_eob;

  double scalar_pixel_sse = 0.0;
  double rdoq_pixel_sse = 0.0;
  int max_pixel_delta = 0;
  int scalar_nonzeros = 0;
  int rdoq_nonzeros = 0;
  std::int64_t visible_samples = 0;
  const int visible_w = plane == 0 ? luma_width_ : chroma_width_;
  const int visible_h = plane == 0 ? luma_height_ : chroma_height_;
  for (int c = 0; c < scalar_eob; ++c) {
    if (scalar_scan[static_cast<std::size_t>(c)] != 0) ++scalar_nonzeros;
  }
  for (int c = 0; c < rdoq.eob; ++c) {
    if (rdoq.levels[static_cast<std::size_t>(c)] != 0) ++rdoq_nonzeros;
  }
  for (int y = 0; y < coded_h; ++y) {
    const int py = y0 + y;
    if (py < 0 || py >= recon_[plane].height) continue;
    for (int x = 0; x < coded_w; ++x) {
      const int px = x0 + x;
      if (px < 0 || px >= recon_[plane].width) continue;
      const int p = pred[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(coded_w) +
                         static_cast<std::size_t>(x)];
      const int src = source(plane, px, py);
      const std::size_t index = static_cast<std::size_t>(y) *
                                    static_cast<std::size_t>(coded_w) +
                                static_cast<std::size_t>(x);
      const int scalar_value = clip1(p + scalar_residual[index]);
      const int rdoq_value = clip1(p + rdoq_residual[index]);
      max_pixel_delta = std::max(max_pixel_delta, std::abs(scalar_value - rdoq_value));
      const int chosen_value = clip1(p + chosen_residual[index]);
      recon_[plane].at(px, py) = static_cast<std::uint8_t>(chosen_value);
      if (px < visible_w && py < visible_h) {
        const int scalar_diff = src - scalar_value;
        const int rdoq_diff = src - rdoq_value;
        scalar_pixel_sse +=
            static_cast<double>(scalar_diff) * static_cast<double>(scalar_diff);
        rdoq_pixel_sse +=
            static_cast<double>(rdoq_diff) * static_cast<double>(rdoq_diff);
        ++visible_samples;
      }
    }
  }

  stats->blocks += 1;
  stats->coefficients += area;
  stats->samples += visible_samples;
  stats->scalar_nonzeros += scalar_nonzeros;
  stats->rdoq_nonzeros += rdoq_nonzeros;
  stats->scalar_eob_sum += scalar_eob;
  stats->rdoq_eob_sum += rdoq.eob;
  stats->scalar_coeff_sse += scalar_coeff_sse;
  stats->rdoq_coeff_sse += rdoq.distortion;
  stats->scalar_pixel_sse += scalar_pixel_sse;
  stats->rdoq_pixel_sse += rdoq_pixel_sse;
  stats->scalar_bits += scalar_bits;
  stats->rdoq_bits += rdoq.rate_bits;
  stats->scalar_cost += scalar_cost;
  stats->rdoq_cost += rdoq.cost;
  stats->max_abs_cost_delta =
      std::max(stats->max_abs_cost_delta, rdoq.cost - scalar_cost);
  stats->max_pixel_delta =
      std::max(stats->max_pixel_delta, static_cast<double>(max_pixel_delta));
  if (rdoq.eob != scalar_eob) {
    ++stats->changed_blocks;
  } else {
    for (int c = 0; c < area; ++c) {
      if (rdoq.levels[static_cast<std::size_t>(c)] !=
          scalar_scan[static_cast<std::size_t>(c)]) {
        ++stats->changed_blocks;
        break;
      }
    }
  }

  if (options_.per_block_csv) {
    std::printf(
        "block,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        block_counter_, plane, x0, y0, params.coded_w, params.coded_h,
        scalar_eob, rdoq.eob, scalar_bits, rdoq.rate_bits, scalar_coeff_sse,
        rdoq.distortion, scalar_cost, rdoq.cost);
  }
  ++block_counter_;

  if (record != nullptr) {
    record->tx_size = params.tx_size;
    record->scalar_eob = scalar_eob;
    record->rdoq_eob = rdoq.eob;
    record->scalar_levels.assign(scalar_scan.begin(),
                                 scalar_scan.begin() + scalar_eob);
    record->rdoq_levels.assign(rdoq.levels.begin(),
                               rdoq.levels.begin() + rdoq.eob);
  }

  update_context(plane, x4, y4, w4, h4, chosen_eob, chosen_raster, scan);
  *eob_out = chosen_eob;
  return true;
}

bool Replay::run(Stats* luma, Stats* u, Stats* v, std::string* error) {
  records_.assign(plan_.blocks.size(), LeafRecord{});
  std::size_t leaf_index = 0;
  for (const av2enc::still::plan::BlockPlan& block : plan_.blocks) {
    LeafRecord& record = records_[leaf_index++];
    // Luma: M1 default is DC.
    if (block.mi_col == 0) {
      for (int plane = 0; plane < 3; ++plane) {
        std::fill(left_level_[plane].begin(), left_level_[plane].end(), 0);
        std::fill(left_dc_[plane].begin(), left_dc_[plane].end(), 0);
      }
    }
    const bool have_left = block.x > 0;
    const bool have_above = block.y > 0;
    std::vector<std::uint8_t> luma_pred;
    if (!predict(0, block.x, block.y, block.luma_w, block.luma_h, have_left,
                 have_above, block.luma_avail.num4_above_right,
                 block.luma_avail.num4_below_left, luma_pred)) {
      *error = "luma prediction failed";
      return false;
    }
    std::int32_t luma_eob = 0;
    if (!code_plane(0, block.x, block.y, block.luma_w, block.luma_h,
                    block.luma_w, block.luma_h, 0, luma_pred, luma,
                    &record.planes[0], &luma_eob, error)) {
      return false;
    }

    // Chroma: M1 default is DC or CfL (uv_full is off by default).
    const std::int32_t cx0 = block.chroma_x;
    const std::int32_t cy0 = block.chroma_y;
    const std::int32_t cw = block.chroma_w;
    const std::int32_t ch = block.chroma_h;
    std::vector<std::uint8_t> u_dc;
    std::vector<std::uint8_t> v_dc;
    if (!predict(1, cx0, cy0, cw, ch, cx0 > 0, cy0 > 0,
                 block.chroma_avail.num4_above_right,
                 block.chroma_avail.num4_below_left, u_dc) ||
        !predict(2, cx0, cy0, cw, ch, cx0 > 0, cy0 > 0,
                 block.chroma_avail.num4_above_right,
                 block.chroma_avail.num4_below_left, v_dc)) {
      *error = "chroma prediction failed";
      return false;
    }
    // M1's default (uv_full = false) scores the DC candidate with the U
    // plane only, then scores CfL with U + V.  Reproduce that asymmetry.
    std::int64_t best_sse = 0;
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(cw) +
                                  static_cast<std::size_t>(x);
        const int su = source(1, cx0 + x, cy0 + y);
        const int du = su - static_cast<int>(u_dc[index]);
        best_sse += static_cast<std::int64_t>(du) * du;
      }
    }
    bool use_cfl = false;
    if (options_.enable_cfl) {
      // Temporary DC prediction in the chroma planes is what M1 does before
      // deriving the implicit alpha.
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const std::size_t index = static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(cw) +
                                    static_cast<std::size_t>(x);
          recon_[1].at(cx0 + x, cy0 + y) = u_dc[index];
          recon_[2].at(cx0 + x, cy0 + y) = v_dc[index];
        }
      }
      for (int plane = 1; plane <= 2; ++plane) {
        av2enc::cfl::Block block_desc;
        block_desc.luma = recon_[0].samples.data();
        block_desc.luma_stride = recon_[0].width;
        block_desc.luma_max_x = plan_.mi_cols * 4 - 1;
        block_desc.luma_max_y = plan_.mi_rows * 4 - 1;
        block_desc.sb_top_y = (cy0 * 2 / 64) * 64;
        block_desc.chroma = recon_[plane].samples.data();
        block_desc.chroma_stride = recon_[plane].width;
        block_desc.x = cx0;
        block_desc.y = cy0;
        block_desc.w = cw;
        block_desc.h = ch;
        block_desc.coded_width = plan_.mi_cols * 2;
        block_desc.coded_height = plan_.mi_rows * 2;
        block_desc.have_above = av2enc::cfl::cfl_avail_above(cy0, 1) ? 1 : 0;
        block_desc.have_left = av2enc::cfl::cfl_avail_left(cx0, 1) ? 1 : 0;
        block_desc.bit_depth = 8;
        block_desc.ds_filter = 0;
        block_desc.sub_x = 1;
        block_desc.sub_y = 1;
        av2enc::cfl::Scratch scratch;
        av2enc::cfl::compute_luma(block_desc, scratch);
        const int alpha = av2enc::cfl::derive_implicit_alpha(block_desc, scratch);
        av2enc::cfl::apply_prediction(block_desc, alpha, scratch);
      }
      std::int64_t cfl_sse = 0;
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const int su = source(1, cx0 + x, cy0 + y);
          const int sv = source(2, cx0 + x, cy0 + y);
          const int du = su - static_cast<int>(recon_[1].at(cx0 + x, cy0 + y));
          const int dv = sv - static_cast<int>(recon_[2].at(cx0 + x, cy0 + y));
          cfl_sse += static_cast<std::int64_t>(du) * du;
          cfl_sse += static_cast<std::int64_t>(dv) * dv;
        }
      }
      use_cfl = cfl_sse < best_sse;
    }
    record.is_cfl = use_cfl;
    std::vector<std::uint8_t> u_pred = u_dc;
    std::vector<std::uint8_t> v_pred = v_dc;
    if (use_cfl) {
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const std::size_t index = static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(cw) +
                                    static_cast<std::size_t>(x);
          recon_[1].at(cx0 + x, cy0 + y) = u_dc[index];
          recon_[2].at(cx0 + x, cy0 + y) = v_dc[index];
        }
      }
      for (int plane = 1; plane <= 2; ++plane) {
        av2enc::cfl::Block block_desc;
        block_desc.luma = recon_[0].samples.data();
        block_desc.luma_stride = recon_[0].width;
        block_desc.luma_max_x = plan_.mi_cols * 4 - 1;
        block_desc.luma_max_y = plan_.mi_rows * 4 - 1;
        block_desc.sb_top_y = (cy0 * 2 / 64) * 64;
        block_desc.chroma = recon_[plane].samples.data();
        block_desc.chroma_stride = recon_[plane].width;
        block_desc.x = cx0;
        block_desc.y = cy0;
        block_desc.w = cw;
        block_desc.h = ch;
        block_desc.coded_width = plan_.mi_cols * 2;
        block_desc.coded_height = plan_.mi_rows * 2;
        block_desc.have_above = av2enc::cfl::cfl_avail_above(cy0, 1) ? 1 : 0;
        block_desc.have_left = av2enc::cfl::cfl_avail_left(cx0, 1) ? 1 : 0;
        block_desc.bit_depth = 8;
        block_desc.ds_filter = 0;
        block_desc.sub_x = 1;
        block_desc.sub_y = 1;
        av2enc::cfl::Scratch scratch;
        av2enc::cfl::compute_luma(block_desc, scratch);
        const int alpha = av2enc::cfl::derive_implicit_alpha(block_desc, scratch);
        av2enc::cfl::apply_prediction(block_desc, alpha, scratch);
        std::vector<std::uint8_t>& target = plane == 1 ? u_pred : v_pred;
        for (int y = 0; y < ch; ++y) {
          for (int x = 0; x < cw; ++x) {
            target[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)] =
                recon_[plane].at(cx0 + x, cy0 + y);
          }
        }
      }
    }

    std::int32_t u_eob = 0;
    if (!code_plane(1, cx0, cy0, cw, ch, cw, ch, 0, u_pred, u,
                    &record.planes[1], &u_eob, error)) {
      return false;
    }
    std::int32_t v_eob = 0;
    if (!code_plane(2, cx0, cy0, cw, ch, cw, ch,
                    u_eob != 0 ? 1 : 0, v_pred, v, &record.planes[2],
                    &v_eob, error)) {
      return false;
    }
    (void)v_eob;
  }
  return true;
}

// Builds a §8.2 syntax stream from the replayed partition tree and decisions.
// The partition/mode/tx decisions are identical to the scalar M1 path; only the
// coefficient levels select scalar or RDOQ.  This is a quantisation-only
// re-encode, not a full RDO search.
bool build_and_encode(const av2enc::still::plan::FramePlan& plan,
                      const std::vector<LeafRecord>& records,
                      bool use_rdoq, const Options& options,
                      std::uint32_t width, std::uint32_t height,
                      std::vector<std::uint8_t>* frame,
                      std::uint32_t* tile_bytes, std::string* error) {
  std::vector<av2enc::still::syntax::BlockSyntax> blocks;
  blocks.reserve(plan.tree.size());
  std::vector<std::int32_t> levels;

  for (const av2enc::still::plan::TreeNode& node : plan.tree) {
    av2enc::still::syntax::BlockSyntax entry;
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = node.block_size;
    entry.partition = node.partition;
    entry.allowed_mask = node.allowed_mask;
    entry.flags = 0;
    if (node.sb_root) {
      entry.flags |= av2enc::still::syntax::kFlagSbRoot;
    }
    if (node.forced) {
      entry.flags |= av2enc::still::syntax::kFlagForced;
    }
    entry.subtree = static_cast<std::uint16_t>(node.subtree);
    if (node.child_count == 0) {
      if (node.leaf_index >= records.size()) {
        *error = "plan leaf index outside the replay records";
        return false;
      }
      const LeafRecord& record = records[node.leaf_index];
      entry.y_mode = 0;  // DC_PRED
      entry.y_angle_delta = 0;
      entry.uv_mode = 0;  // DC_PRED (CfL uses is_cfl + DC UVMode)
      if (record.is_cfl) {
        entry.flags |= av2enc::still::syntax::kFlagIsCfl;
      }
      entry.levels_offset = static_cast<std::uint32_t>(levels.size());
      for (int plane = 0; plane < 3; ++plane) {
        const PlaneRecord& plane_record = record.planes[plane];
        entry.tx_size[plane] = static_cast<std::uint8_t>(plane_record.tx_size);
        const int eob = use_rdoq ? plane_record.rdoq_eob
                                 : plane_record.scalar_eob;
        entry.eob[plane] = static_cast<std::uint32_t>(eob);
        const std::vector<std::int32_t>& source =
            use_rdoq ? plane_record.rdoq_levels : plane_record.scalar_levels;
        if (static_cast<std::size_t>(eob) > source.size()) {
          *error = "plane EOB exceeds the recorded level count";
          return false;
        }
        levels.insert(levels.end(), source.begin(),
                      source.begin() + static_cast<std::ptrdiff_t>(eob));
      }
    }
    blocks.push_back(entry);
  }

  av2enc::still::StillConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = static_cast<std::uint8_t>(options.qp);
  config.disable_cdf_update = true;
  config.enable_intra_edge_filter = options.enable_edge_filter;
  config.enable_cfl_intra = options.enable_cfl;
  config.enable_chroma_dctonly = true;
  config.tx_mode_select = false;
  config.apply_deblock_y_vert = false;
  config.apply_deblock_y_horz = false;
  config.apply_deblock_u = false;
  config.apply_deblock_v = false;

  av2enc::still::syntax::FramePlan frame_plan;
  frame_plan.width = static_cast<int>(width);
  frame_plan.height = static_cast<int>(height);
  frame_plan.sb_size = 64;
  std::size_t encoded_tile_bytes = 0;
  if (!av2enc::still::syntax::assemble_still_frame(frame_plan, blocks, levels,
                                             config, *frame, *error,
                                             &encoded_tile_bytes)) {
    return false;
  }
  *tile_bytes = static_cast<std::uint32_t>(encoded_tile_bytes);
  return true;
}

bool write_ivf(const std::string& path,
               std::span<const std::uint8_t> frame, std::uint32_t width,
               std::uint32_t height, std::string* error) {
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  av2enc::IvfWriter writer;
  if (!writer.open(path, header, *error)) {
    return false;
  }
  if (!writer.write_frame(frame, 0, *error)) {
    return false;
  }
  return writer.close(*error);
}

void print_stats(const char* name, const Stats& stats) {
  const double coeff_psnr =
      stats.scalar_coeff_sse > 0.0 ? 0.0 : 0.0;  // unused, kept explicit
  (void)coeff_psnr;
  const double scalar_bits = stats.scalar_bits;
  const double rdoq_bits = stats.rdoq_bits;
  const double bit_saving =
      scalar_bits > 0.0 ? (scalar_bits - rdoq_bits) / scalar_bits * 100.0 : 0.0;
  const double cost_gain =
      stats.scalar_cost > 0.0
          ? (stats.scalar_cost - stats.rdoq_cost) / stats.scalar_cost * 100.0
          : 0.0;
  const double changed =
      stats.blocks > 0
          ? static_cast<double>(stats.changed_blocks) /
                static_cast<double>(stats.blocks) * 100.0
          : 0.0;
  std::printf(
      "%-4s blocks=%lld coeffs=%lld nonzeros %lld->%lld eob_sum %lld->%lld "
      "changed=%.1f%%\n",
      name, static_cast<long long>(stats.blocks),
      static_cast<long long>(stats.coefficients),
      static_cast<long long>(stats.scalar_nonzeros),
      static_cast<long long>(stats.rdoq_nonzeros),
      static_cast<long long>(stats.scalar_eob_sum),
      static_cast<long long>(stats.rdoq_eob_sum), changed);
  const double psnr_scalar =
      stats.scalar_pixel_sse > 0.0 && stats.samples > 0
          ? 10.0 * std::log10(255.0 * 255.0 *
                              static_cast<double>(stats.samples) /
                              stats.scalar_pixel_sse)
          : 0.0;
  const double psnr_rdoq =
      stats.rdoq_pixel_sse > 0.0 && stats.samples > 0
          ? 10.0 * std::log10(255.0 * 255.0 *
                              static_cast<double>(stats.samples) /
                              stats.rdoq_pixel_sse)
          : 0.0;
  std::printf(
      "     coeff_sse %.3f -> %.3f  pixel_sse %.3f -> %.3f  "
      "psnr %.3f -> %.3f dB  bits %.3f -> %.3f (%.2f%%)  "
      "cost %.3f -> %.3f (%.2f%%)\n",
      stats.scalar_coeff_sse, stats.rdoq_coeff_sse, stats.scalar_pixel_sse,
      stats.rdoq_pixel_sse, psnr_scalar, psnr_rdoq, scalar_bits, rdoq_bits,
      bit_saving, stats.scalar_cost, stats.rdoq_cost, cost_gain);
}

bool parse_int(std::string_view text, int* value) {
  try {
    std::size_t used = 0;
    const int parsed = std::stoi(std::string(text), &used);
    if (used != text.size()) return false;
    *value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

void usage() {
  std::printf(
      "still_rdoq_quality -i input.y4m [options]\n"
      "\n"
      "Replays M1 default decisions, then measures full-trellis RDOQ against\n"
      "scalar quantisation on the real M1 coefficients.\n"
      "\n"
      "  --qp N              base_q_idx (default 90; the rate model is qctx 0)\n"
      "  --depth N           max partition depth (default 4)\n"
      "  --min-block-size N  partition floor (default 8)\n"
      "  --no-cfl            disable CfL (chroma DC only)\n"
      "  --no-edge-filter    disable the intra edge filter\n"
      "  --sse-scale X       RDO lambda scale (default 0.125)\n"
      "  --prev-cap N        trellis previous-level state cap (default 2)\n"
      "  --level-cap N       hard level cap (0 = exact derived bound)\n"
      "  --plane y|u|v|all   which planes to measure (default all)\n"
      "  --check-reference   run encode_still_picture and compare recon\n"
      "  --per-block         print one CSV row per transform block\n"
      "  --write-ivf PREFIX  write PREFIX.scalar.ivf and PREFIX.rdoq.ivf\n"
      "  --max-blocks N      stop after N leaf blocks (0 = all)\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto next = [&](std::string* out) {
      if (i + 1 >= argc) return false;
      *out = argv[++i];
      return true;
    };
    std::string value;
    if (arg == "-i" || arg == "--input") {
      if (!next(&options.input)) return 2;
    } else if (arg == "--qp") {
      if (!next(&value) || !parse_int(value, &options.qp)) return 2;
    } else if (arg == "--depth") {
      if (!next(&value) || !parse_int(value, &options.depth)) return 2;
    } else if (arg == "--min-block-size") {
      if (!next(&value) || !parse_int(value, &options.min_block_size)) return 2;
    } else if (arg == "--no-cfl") {
      options.enable_cfl = false;
    } else if (arg == "--no-edge-filter") {
      options.enable_edge_filter = false;
    } else if (arg == "--sse-scale") {
      if (!next(&value)) return 2;
      options.sse_scale = std::stod(value);
    } else if (arg == "--prev-cap") {
      if (!next(&value) || !parse_int(value, &options.prev_level_cap)) return 2;
    } else if (arg == "--level-cap") {
      if (!next(&value) || !parse_int(value, &options.level_cap)) return 2;
    } else if (arg == "--plane") {
      if (!next(&value)) return 2;
      if (value == "y") options.plane_mask = 1;
      else if (value == "u") options.plane_mask = 2;
      else if (value == "v") options.plane_mask = 4;
      else if (value == "all") options.plane_mask = 7;
      else return 2;
    } else if (arg == "--check-reference") {
      options.check_reference = true;
    } else if (arg == "--per-block") {
      options.per_block_csv = true;
    } else if (arg == "--write-ivf") {
      if (!next(&value)) return 2;
      options.write_ivf_prefix = value;
    } else if (arg == "--max-blocks") {
      if (!next(&value) || !parse_int(value, &options.max_blocks)) return 2;
    } else if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }
  if (options.input.empty()) {
    usage();
    return 2;
  }
  if (options.qp < 0 || options.qp > 90) {
    std::fprintf(stderr, "the rate model covers base_q_idx <= 90 (qctx 0)\n");
    return 2;
  }

  av2enc::Y4mReader reader(options.input);
  std::string error;
  if (!reader.open(error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult read = reader.read_frame(frame, error);
  if (read != av2enc::Y4mReadResult::kFrame) {
    std::fprintf(stderr, "no frame: %s\n", error.c_str());
    return 1;
  }

  av2enc::still::StillPicture picture;
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();

  av2enc::still::plan::PlanOptions plan_options;
  plan_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
  plan_options.min_block_size = options.min_block_size;
  plan_options.max_partition_depth = options.depth;
  plan_options.allow_rect = true;
  plan_options.allow_split = false;
  plan_options.force_single_block = false;
  av2enc::still::plan::apply_debug_env(plan_options);

  av2enc::still::plan::FramePlan plan;
  if (!av2enc::still::plan::build_frame_plan(
          static_cast<int>(picture.width), static_cast<int>(picture.height),
          picture.y, plan_options, plan, error)) {
    std::fprintf(stderr, "plan: %s\n", error.c_str());
    return 1;
  }

  // Two replays with the same partition tree and mode decisions:
  //   scalar: scalar levels, scalar reconstruction;
  //   rdoq:   full-trellis levels, RDOQ reconstruction.  The second pass uses
  //           its own reconstruction for later predictions and CfL, so the
  //           emitted stream is internally consistent (the decoder sees the
  //           same recon trajectory).
  Stats scalar_luma;
  Stats scalar_u;
  Stats scalar_v;
  Replay scalar_replay(picture, options, plan, /*run_rdoq=*/false,
                       /*use_rdoq_recon=*/false);
  if (!scalar_replay.run(&scalar_luma, &scalar_u, &scalar_v, &error)) {
    std::fprintf(stderr, "scalar replay: %s\n", error.c_str());
    return 1;
  }
  Stats rdoq_luma;
  Stats rdoq_u;
  Stats rdoq_v;
  Replay rdoq_replay(picture, options, plan, /*run_rdoq=*/true,
                     /*use_rdoq_recon=*/true);
  if (!rdoq_replay.run(&rdoq_luma, &rdoq_u, &rdoq_v, &error)) {
    std::fprintf(stderr, "rdoq replay: %s\n", error.c_str());
    return 1;
  }

  const auto combine = [](const Stats& scalar, const Stats& rdoq) {
    Stats out = scalar;
    out.rdoq_nonzeros = rdoq.rdoq_nonzeros;
    out.rdoq_eob_sum = rdoq.rdoq_eob_sum;
    out.changed_blocks = rdoq.changed_blocks;
    out.rdoq_coeff_sse = rdoq.rdoq_coeff_sse;
    out.rdoq_pixel_sse = rdoq.rdoq_pixel_sse;
    out.rdoq_bits = rdoq.rdoq_bits;
    out.rdoq_cost = rdoq.rdoq_cost;
    out.max_abs_cost_delta = rdoq.max_abs_cost_delta;
    out.max_pixel_delta = rdoq.max_pixel_delta;
    return out;
  };
  const Stats luma = combine(scalar_luma, rdoq_luma);
  const Stats u = combine(scalar_u, rdoq_u);
  const Stats v = combine(scalar_v, rdoq_v);

  std::printf("input: %s  %ux%u  qp=%d depth=%d min_block=%d lambda_scale=%.3f\n",
              options.input.c_str(), picture.width, picture.height, options.qp,
              options.depth, options.min_block_size, options.sse_scale);
  std::printf("plan: leaves=%d nodes=%d cost=%.3f\n", plan.leaves, plan.nodes,
              plan.cost);
  print_stats("Y", luma);
  if (options.plane_mask & 2) print_stats("U", u);
  if (options.plane_mask & 4) print_stats("V", v);
  Stats total = luma;
  if (options.plane_mask & 2) total.add(u);
  if (options.plane_mask & 4) total.add(v);
  print_stats("ALL", total);

  // Quantisation-only syntax re-encode: same partition/mode/tx decisions, only
  // the coefficient levels differ.  The RDOQ stream is produced by the RDOQ
  // replay, so its predictions and CfL are consistent with the coded levels.
  std::vector<std::uint8_t> scalar_frame;
  std::vector<std::uint8_t> rdoq_frame;
  std::uint32_t scalar_tile = 0;
  std::uint32_t rdoq_tile = 0;
  if (!build_and_encode(plan, scalar_replay.records(), false, options,
                        picture.width, picture.height, &scalar_frame,
                        &scalar_tile, &error) ||
      !build_and_encode(plan, rdoq_replay.records(), true, options,
                        picture.width, picture.height, &rdoq_frame,
                        &rdoq_tile, &error)) {
    std::fprintf(stderr, "syntax re-encode: %s\n", error.c_str());
    return 1;
  }
  const double scalar_tile_bits = static_cast<double>(scalar_tile) * 8.0;
  const double rdoq_tile_bits = static_cast<double>(rdoq_tile) * 8.0;
  const double tile_saving =
      scalar_tile_bits > 0.0
          ? (scalar_tile_bits - rdoq_tile_bits) / scalar_tile_bits * 100.0
          : 0.0;
  std::printf(
      "tile (same modes/partition): scalar %u B (%.0f bits) -> rdoq %u B "
      "(%.0f bits), %.2f%% saving; model %.0f -> %.0f bits\n",
      scalar_tile, scalar_tile_bits, rdoq_tile, rdoq_tile_bits, tile_saving,
      total.scalar_bits, total.rdoq_bits);
  if (total.scalar_bits > 0.0) {
    const double lambda_eff =
        (total.scalar_cost - total.scalar_coeff_sse) / total.scalar_bits;
    const double actual_scalar =
        total.scalar_pixel_sse + lambda_eff * scalar_tile_bits;
    const double actual_rdoq =
        total.rdoq_pixel_sse + lambda_eff * rdoq_tile_bits;
    const double actual_gain =
        actual_scalar > 0.0
            ? (actual_scalar - actual_rdoq) / actual_scalar * 100.0
            : 0.0;
    std::printf(
        "actual-tile RD (single weighted lambda=%.3f): %.3f -> %.3f "
        "(%.2f%%), visible samples=%lld psnr %.3f -> %.3f dB\n",
        lambda_eff, actual_scalar, actual_rdoq, actual_gain,
        static_cast<long long>(total.samples),
        total.scalar_pixel_sse > 0.0 && total.samples > 0
            ? 10.0 * std::log10(255.0 * 255.0 *
                                static_cast<double>(total.samples) /
                                total.scalar_pixel_sse)
            : 0.0,
        total.rdoq_pixel_sse > 0.0 && total.samples > 0
            ? 10.0 * std::log10(255.0 * 255.0 *
                                static_cast<double>(total.samples) /
                                total.rdoq_pixel_sse)
            : 0.0);
  }

  if (!options.write_ivf_prefix.empty()) {
    if (!write_ivf(options.write_ivf_prefix + ".scalar.ivf", scalar_frame,
                   picture.width, picture.height, &error) ||
        !write_ivf(options.write_ivf_prefix + ".rdoq.ivf", rdoq_frame,
                   picture.width, picture.height, &error)) {
      std::fprintf(stderr, "write ivf: %s\n", error.c_str());
      return 1;
    }
    std::printf("wrote %s.scalar.ivf and %s.rdoq.ivf\n",
                options.write_ivf_prefix.c_str(),
                options.write_ivf_prefix.c_str());
  }

  if (options.check_reference) {
    av2enc::still::StillOptions still_options;
    still_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
    still_options.enable_cfl = options.enable_cfl;
    still_options.enable_deblock = false;
    still_options.enable_intra_edge_filter = options.enable_edge_filter;
    still_options.adaptive_cdf = false;
    still_options.min_block_size = options.min_block_size;
    still_options.max_partition_depth = options.depth;
    still_options.luma_dc_only = true;
    still_options.uv_full = false;
    av2enc::still::StillResult reference;
    if (!av2enc::still::encode_still_picture(picture, still_options, reference,
                                             error)) {
      std::fprintf(stderr, "reference encode: %s\n", error.c_str());
      return 1;
    }
    const auto compare_plane = [](const Plane& replay_plane,
                                  const std::vector<std::uint8_t>& reference,
                                  int width, int height, const char* name) {
      std::int64_t differences = 0;
      int max_difference = 0;
      const std::size_t count =
          static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const std::size_t index =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x);
          if (index >= reference.size()) break;
          const int difference =
              std::abs(static_cast<int>(replay_plane.at(x, y)) -
                       static_cast<int>(reference[index]));
          if (difference != 0) ++differences;
          max_difference = std::max(max_difference, difference);
        }
      }
      std::printf("reference %s: %lld/%zu differ, max |d|=%d\n", name,
                  static_cast<long long>(differences), count, max_difference);
      return differences;
    };
    const int cw = static_cast<int>((picture.width + 1U) / 2U);
    const int ch = static_cast<int>((picture.height + 1U) / 2U);
    const std::int64_t total_differences =
        compare_plane(scalar_replay.recon(0), reference.recon_y,
                      static_cast<int>(picture.width),
                      static_cast<int>(picture.height), "Y") +
        compare_plane(scalar_replay.recon(1), reference.recon_u, cw, ch, "U") +
        compare_plane(scalar_replay.recon(2), reference.recon_v, cw, ch, "V");
    std::printf("reference: tile_bytes=%u frame_bytes=%u leaves=%d\n",
                reference.tile_bytes, reference.frame_bytes, reference.leaves);
    if (reference.tile_bytes == scalar_tile) {
      std::printf("reference: scalar syntax re-encode tile bytes MATCH\n");
    } else {
      std::printf("reference: scalar tile bytes differ (%u vs %u)\n",
                  scalar_tile, reference.tile_bytes);
    }
    if (total_differences == 0) {
      std::printf("reference: EXACT replay\n");
    } else {
      std::printf("reference: replay differs (coefficients may still be exact)\n");
    }
  }

  const double scalar_pixel = total.scalar_pixel_sse;
  const double rdoq_pixel = total.rdoq_pixel_sse;
  const double samples =
      static_cast<double>(total.blocks) * 0.0;  // unused
  (void)samples;
  const double bit_saving = total.scalar_bits > 0.0
                                ? (total.scalar_bits - total.rdoq_bits) /
                                      total.scalar_bits * 100.0
                                : 0.0;
  const double cost_gain = total.scalar_cost > 0.0
                               ? (total.scalar_cost - total.rdoq_cost) /
                                     total.scalar_cost * 100.0
                               : 0.0;
  std::printf(
      "summary: bits %.3f -> %.3f (%.2f%%), cost %.3f -> %.3f (%.2f%%), "
      "pixel_sse %.3f -> %.3f\n",
      total.scalar_bits, total.rdoq_bits, bit_saving, total.scalar_cost,
      total.rdoq_cost, cost_gain, scalar_pixel, rdoq_pixel);
  return 0;
}
