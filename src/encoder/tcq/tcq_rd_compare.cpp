// Real-coefficient RD comparison: scalar vs RDOQ vs TCQ, and CCTX off vs on.
//
// This harness reads a real Y4M frame, forms DC-predicted residual blocks,
// runs the existing `tx_variants` forward transform, and then quantises the
// same coefficient blocks three ways:
//
//   * scalar  - `txv::quantize()`
//   * RDOQ    - `src/encoder/rdoq/` with its static §9.3 model
//   * TCQ     - this component with the TCQ-state-aware model
//
// All three use the same lambda and the same block geometry.  Chroma blocks
// additionally compare CCTX NONE against the seven-type CCTX search.  This
// writes no bitstream; it is a quantisation-only RD measurement on real
// coefficients.  The decode gate is covered separately (see README).

#include "cctx.h"
#include "tcq.h"
#include "tcq_av2_model.h"

#include "encoder/m1/qctx/qctx.h"
#include "encoder/rc/rc_tables.h"
#include "encoder/rdo/rdo.h"
#include "encoder/rdoq/rdoq.h"
#include "encoder/rdoq/rdoq_av2_model.h"
#include "encoder/tx_variants/tx_variants.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::tcq::CctxConfig;
using av2enc::tcq::CctxResult;
using av2enc::tcq::CctxType;
using av2enc::txv::TxParams;

int clip1(int value) {
  return value < 0 ? 0 : (value > 255 ? 255 : value);
}

struct Options {
  std::string input;
  int base_q_idx = 90;
  int block = 8;
  double sse_scale = 0.125;
  int max_blocks = 0;
  int plane_mask = 7;  // bit 0 Y, bit 1 U/V (CCTX), bit 2 unused
  bool verbose = false;
};

struct Stats {
  std::int64_t blocks = 0;
  std::int64_t nonzero = 0;
  double sse = 0.0;
  double bits = 0.0;
  double cost = 0.0;

  void add(double sse_value, double bits_value, double lambda,
           std::int64_t nz = 0) {
    sse += sse_value;
    bits += bits_value;
    cost += sse_value + lambda * bits_value;
    nonzero += nz;
  }
};

int parse_int(const char* text, int fallback) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') return fallback;
  return static_cast<int>(value);
}

double parse_double(const char* text, double fallback) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0') return fallback;
  return value;
}

void print_row(const char* name, const Stats& stats) {
  std::printf(
      "  %-8s nz=%9lld sse=%14.3f bits=%12.2f cost=%14.3f\n", name,
      static_cast<long long>(stats.nonzero), stats.sse, stats.bits,
      stats.cost);
}

double percent(double from, double to) {
  if (from == 0.0) return 0.0;
  return 100.0 * (to - from) / from;
}

std::vector<std::int32_t> make_scan(int w, int h) {
  std::vector<std::int32_t> scan(static_cast<std::size_t>(w) * h, 0);
  av2enc::tcq::av2::Model::build_scan(w, h, scan);
  return scan;
}

// Reference base-level approximation for the TCQ context model: scalar level
// magnitude at half step, rounded up.  Only contexts are frozen this way.
std::vector<std::int32_t> reference_base_levels(
    const std::vector<std::int32_t>& scalar_scan) {
  std::vector<std::int32_t> out(scalar_scan.size(), 0);
  for (std::size_t i = 0; i < scalar_scan.size(); ++i) {
    const std::int32_t value = scalar_scan[i];
    const std::int32_t magnitude = std::abs(value);
    const std::int32_t base = (magnitude + 1) / 2;
    out[i] = value < 0 ? -base : base;
  }
  return out;
}

bool run_luma(const av2enc::Y4mFrame& frame, const Options& options,
              Stats* scalar, Stats* rdoq, Stats* tcq) {
  const int width = static_cast<int>(frame.width());
  const int height = static_cast<int>(frame.height());
  const int b = options.block;
  if (b < 4 || b > 64 || (b & (b - 1)) != 0) {
    std::fprintf(stderr, "block must be a power of two in [4, 64]\n");
    return false;
  }
  std::optional<TxParams> params_opt =
      av2enc::txv::make_tx_params(b, b, av2enc::txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    std::fprintf(stderr, "unsupported block size %d\n", b);
    return false;
  }
  const int step = av2enc::rc::kQlookupQ0To255[std::clamp(options.base_q_idx, 1, 255)];
  TxParams params = *params_opt;
  params.quant_factor *=
      static_cast<double>(av2enc::txv::kQuantStep) / static_cast<double>(step);
  const double quant_step = 1.0 / params.quant_factor;
  const double lambda =
      av2enc::rdo::lambda_sse_from_quant_step(quant_step, options.sse_scale);
  const int qctx = av2enc::m1::qctx::slice_for_base_q_idx(options.base_q_idx);

  const std::span<const std::uint8_t> y = frame.y();
  const int stride = width;
  std::vector<std::int16_t> residual(
      static_cast<std::size_t>(params.w) * params.h);
  std::vector<double> coeff(static_cast<std::size_t>(params.w) * params.h);
  std::vector<std::int32_t> scalar_raster(coeff.size(), 0);
  const std::vector<std::int32_t> scan = make_scan(params.w, params.h);
  const int area = params.w * params.h;

  int blocks = 0;
  for (int y0 = 0; y0 + b <= height; y0 += b) {
    for (int x0 = 0; x0 + b <= width; x0 += b) {
      if (options.max_blocks > 0 && blocks >= options.max_blocks) break;
      std::int64_t sum = 0;
      for (int yy = 0; yy < b; ++yy) {
        for (int xx = 0; xx < b; ++xx) {
          sum += y[static_cast<std::size_t>(y0 + yy) * stride + x0 + xx];
        }
      }
      const int pred =
          static_cast<int>((sum + (b * b) / 2) / (b * b));
      av2enc::txv::make_residual(y.data(), stride, width, height, x0, y0,
                                 pred, params, residual.data());
      av2enc::txv::forward_transform(residual.data(), params, coeff.data(),
                                     scalar_raster.data());

      std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area));
      std::vector<av2enc::rdoq::Coefficient> rdoq_input(
          static_cast<std::size_t>(area));
      for (int c = 0; c < area; ++c) {
        const std::int32_t pos = scan[static_cast<std::size_t>(c)];
        scalar_scan[static_cast<std::size_t>(c)] =
            scalar_raster[static_cast<std::size_t>(pos)];
        rdoq_input[static_cast<std::size_t>(c)].original =
            coeff[static_cast<std::size_t>(pos)];
        rdoq_input[static_cast<std::size_t>(c)].scalar_level =
            scalar_raster[static_cast<std::size_t>(pos)];
        rdoq_input[static_cast<std::size_t>(c)].level_context = 0;
      }

      // Scalar baseline under the RDOQ AV2 model.
      av2enc::rdoq::av2::BlockInfo rdoq_info;
      rdoq_info.plane = 0;
      rdoq_info.params = params;
      rdoq_info.txb_skip_context = 0;
      rdoq_info.dc_sign_context = 0;
      rdoq_info.qctx = qctx;
      av2enc::rdoq::av2::Model rdoq_model;
      std::string error;
      if (!rdoq_model.init(rdoq_info, scalar_scan, &error)) {
        std::fprintf(stderr, "rdoq model init: %s\n", error.c_str());
        return false;
      }
      double scalar_sse = 0.0;
      std::int64_t scalar_nz = 0;
      for (int c = 0; c < area; ++c) {
        const std::int32_t level = scalar_scan[static_cast<std::size_t>(c)];
        if (level != 0) ++scalar_nz;
        const double recon = static_cast<double>(level) * quant_step;
        const double err =
            rdoq_input[static_cast<std::size_t>(c)].original - recon;
        scalar_sse += err * err;
      }
      scalar->add(scalar_sse, rdoq_model.scalar_bits(), lambda, scalar_nz);

      // RDOQ.
      av2enc::rdoq::Config rdoq_config;
      rdoq_config.quant_step = quant_step;
      rdoq_config.lambda = lambda;
      rdoq_config.max_coefficients = area;
      av2enc::rdoq::RdoqResult rdoq_result;
      if (!av2enc::rdoq::rdoq_quantize(rdoq_input, rdoq_config,
                                       rdoq_model.rate_model(), rdoq_result,
                                       &error)) {
        std::fprintf(stderr, "rdoq: %s\n", error.c_str());
        return false;
      }
      std::int64_t rdoq_nz = 0;
      for (std::int32_t c = 0; c < rdoq_result.eob; ++c) {
        if (rdoq_result.levels[static_cast<std::size_t>(c)] != 0) ++rdoq_nz;
      }
      rdoq->add(rdoq_result.distortion, rdoq_result.rate_bits, lambda,
                rdoq_nz);

      // TCQ.
      const std::vector<std::int32_t> ref =
          reference_base_levels(scalar_scan);
      av2enc::tcq::av2::BlockInfo tcq_info;
      tcq_info.plane = 0;
      tcq_info.params = params;
      tcq_info.txb_skip_context = 0;
      tcq_info.dc_sign_context = 0;
      tcq_info.qctx = qctx;
      tcq_info.is_inter = false;
      tcq_info.tx_class_2d = true;
      av2enc::tcq::av2::Model tcq_model;
      if (!tcq_model.init(tcq_info, ref, &error)) {
        std::fprintf(stderr, "tcq model init: %s\n", error.c_str());
        return false;
      }
      std::vector<av2enc::tcq::Coefficient> tcq_input(
          static_cast<std::size_t>(area));
      for (int c = 0; c < area; ++c) {
        const std::int32_t pos = scan[static_cast<std::size_t>(c)];
        const int row = pos / params.w;
        const int col = pos % params.w;
        tcq_input[static_cast<std::size_t>(c)].original =
            coeff[static_cast<std::size_t>(pos)];
        tcq_input[static_cast<std::size_t>(c)].level_context = 0;
        tcq_input[static_cast<std::size_t>(c)].pre_max =
            (row + col) < 4 ? 8 : 6;
      }
      av2enc::tcq::Config tcq_config;
      tcq_config.quant_step = quant_step;
      tcq_config.lambda = lambda;
      tcq_config.max_coefficients = area;
      av2enc::tcq::Result tcq_result;
      if (!av2enc::tcq::quantize(tcq_input, tcq_config,
                                 tcq_model.rate_model(), tcq_result,
                                 &error)) {
        std::fprintf(stderr, "tcq: %s\n", error.c_str());
        return false;
      }
      std::int64_t tcq_nz = 0;
      for (std::int32_t c = 0; c < tcq_result.eob; ++c) {
        if (tcq_result.base_levels[static_cast<std::size_t>(c)] != 0) ++tcq_nz;
      }
      tcq->add(tcq_result.distortion, tcq_result.rate_bits, lambda, tcq_nz);
      ++blocks;
    }
  }
  // Stash block count in the first stat.
  scalar->blocks += blocks;
  rdoq->blocks += blocks;
  tcq->blocks += blocks;
  return blocks > 0;
}

bool run_chroma(const av2enc::Y4mFrame& frame, const Options& options,
                Stats* base, Stats* cctx, std::int64_t* cctx_types,
                std::int64_t* cctx_chosen) {
  const int width = static_cast<int>(frame.chroma_width());
  const int height = static_cast<int>(frame.chroma_height());
  const int b = options.block / 2;
  if (b < 4) return true;
  std::optional<TxParams> params_opt =
      av2enc::txv::make_tx_params(b, b, av2enc::txv::TxType::kDctDct);
  if (!params_opt.has_value()) return true;
  const int step =
      av2enc::rc::kQlookupQ0To255[std::clamp(options.base_q_idx, 1, 255)];
  TxParams params = *params_opt;
  params.quant_factor *=
      static_cast<double>(av2enc::txv::kQuantStep) / static_cast<double>(step);
  const double quant_step = 1.0 / params.quant_factor;
  const double lambda =
      av2enc::rdo::lambda_sse_from_quant_step(quant_step, options.sse_scale);
  const int qctx = av2enc::m1::qctx::slice_for_base_q_idx(options.base_q_idx);
  const std::vector<std::int32_t> scan = make_scan(params.w, params.h);
  const int area = params.w * params.h;

  std::vector<std::int16_t> residual(
      static_cast<std::size_t>(params.w) * params.h);
  std::vector<double> cu(static_cast<std::size_t>(area));
  std::vector<double> cv(static_cast<std::size_t>(area));
  std::vector<std::int32_t> lu(static_cast<std::size_t>(area));
  std::vector<std::int32_t> lv(static_cast<std::size_t>(area));
  const std::span<const std::uint8_t> plane_u = frame.u();
  const std::span<const std::uint8_t> plane_v = frame.v();

  int blocks = 0;
  for (int y0 = 0; y0 + b <= height; y0 += b) {
    for (int x0 = 0; x0 + b <= width; x0 += b) {
      if (options.max_blocks > 0 && blocks >= options.max_blocks) break;
      for (int plane = 1; plane <= 2; ++plane) {
        const std::span<const std::uint8_t> src = plane == 1 ? plane_u : plane_v;
        std::int64_t sum = 0;
        for (int yy = 0; yy < b; ++yy) {
          for (int xx = 0; xx < b; ++xx) {
            sum += src[static_cast<std::size_t>(y0 + yy) * width + x0 + xx];
          }
        }
        const int pred = static_cast<int>((sum + (b * b) / 2) / (b * b));
        av2enc::txv::make_residual(src.data(), width, width, height, x0, y0,
                                   pred, params, residual.data());
        std::vector<std::int32_t> levels(static_cast<std::size_t>(area));
        std::vector<double> coeff(static_cast<std::size_t>(area));
        av2enc::txv::forward_transform(residual.data(), params, coeff.data(),
                                       levels.data());
        for (int c = 0; c < area; ++c) {
          const std::int32_t pos = scan[static_cast<std::size_t>(c)];
          if (plane == 1) {
            cu[static_cast<std::size_t>(c)] = coeff[static_cast<std::size_t>(pos)];
            lu[static_cast<std::size_t>(c)] = levels[static_cast<std::size_t>(pos)];
          } else {
            cv[static_cast<std::size_t>(c)] = coeff[static_cast<std::size_t>(pos)];
            lv[static_cast<std::size_t>(c)] = levels[static_cast<std::size_t>(pos)];
          }
        }
      }

      // Baseline: both planes scalar, no cctx_type symbol.
      auto plane_bits = [&](int plane, const std::vector<std::int32_t>& levels) {
        av2enc::rdoq::av2::BlockInfo info;
        info.plane = plane;
        info.params = params;
        info.txb_skip_context = 0;
        info.dc_sign_context = 0;
        info.qctx = qctx;
        av2enc::rdoq::av2::Model model;
        std::string error;
        if (!model.init(info, levels, &error)) return 0.0;
        return model.scalar_bits();
      };
      double base_sse = 0.0;
      std::int64_t base_nz = 0;
      for (int c = 0; c < area; ++c) {
        if (lu[static_cast<std::size_t>(c)] != 0) ++base_nz;
        if (lv[static_cast<std::size_t>(c)] != 0) ++base_nz;
        const double du =
            cu[static_cast<std::size_t>(c)] -
            static_cast<double>(lu[static_cast<std::size_t>(c)]) * quant_step;
        const double dv =
            cv[static_cast<std::size_t>(c)] -
            static_cast<double>(lv[static_cast<std::size_t>(c)]) * quant_step;
        base_sse += du * du + dv * dv;
      }
      const double base_bits =
          plane_bits(1, lu) + plane_bits(2, lv);
      base->add(base_sse, base_bits, lambda, base_nz);

      // CCTX search.  cctx_type is present for plane 1 when eob_u != 1 (and
      // the frame allows CCTX).  A 1-coefficient U block cannot signal it.
      std::int32_t eob_u = 0;
      for (int c = 0; c < area; ++c) {
        if (lu[static_cast<std::size_t>(c)] != 0) eob_u = c + 1;
      }
      const bool signaled = eob_u != 1;
      CctxConfig config;
      config.enabled = signaled;
      config.quant_step_u = quant_step;
      config.quant_step_v = quant_step;
      config.bitdepth = 8;
      std::vector<CctxResult> candidates;
      std::string error;
      if (!av2enc::tcq::cctx_enumerate_scalar(cu, cv, params.w, params.h,
                                              config, &candidates, &error)) {
        std::fprintf(stderr, "cctx: %s\n", error.c_str());
        return false;
      }
      // RD selection: coefficient costs plus the cctx_type symbol when the
      // frame signals it.
      const CctxResult* chosen = nullptr;
      double chosen_bits = 0.0;
      double chosen_cost = std::numeric_limits<double>::infinity();
      for (const CctxResult& candidate : candidates) {
        const double bits =
            plane_bits(1, candidate.u_levels) +
            plane_bits(2, candidate.v_levels) +
            (signaled ? av2enc::tcq::cctx_type_bits(candidate.type) : 0.0);
        const double cost = candidate.distortion + lambda * bits;
        if (cost < chosen_cost) {
          chosen_cost = cost;
          chosen_bits = bits;
          chosen = &candidate;
        }
      }
      if (chosen == nullptr) {
        std::fprintf(stderr, "cctx: no candidate\n");
        return false;
      }
      std::int64_t cctx_nz = 0;
      for (int c = 0; c < area; ++c) {
        if (chosen->u_levels[static_cast<std::size_t>(c)] != 0) ++cctx_nz;
        if (chosen->v_levels[static_cast<std::size_t>(c)] != 0) ++cctx_nz;
      }
      cctx->add(chosen->distortion, chosen_bits, lambda, cctx_nz);
      ++(*cctx_chosen);
      if (chosen->type != CctxType::kNone) ++(*cctx_types);
      ++blocks;
    }
  }
  base->blocks += blocks;
  cctx->blocks += blocks;
  return blocks > 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      options.input = argv[++i];
    } else if (std::strcmp(argv[i], "--qp") == 0 && i + 1 < argc) {
      options.base_q_idx = parse_int(argv[++i], 90);
    } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
      options.block = parse_int(argv[++i], 8);
    } else if (std::strcmp(argv[i], "--sse-scale") == 0 && i + 1 < argc) {
      options.sse_scale = parse_double(argv[++i], 0.125);
    } else if (std::strcmp(argv[i], "--max-blocks") == 0 && i + 1 < argc) {
      options.max_blocks = parse_int(argv[++i], 0);
    } else if (std::strcmp(argv[i], "--planes") == 0 && i + 1 < argc) {
      options.plane_mask = parse_int(argv[++i], 7);
    } else if (std::strcmp(argv[i], "--verbose") == 0) {
      options.verbose = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }
  if (options.input.empty()) {
    std::fprintf(stderr,
                 "usage: %s -i input.y4m [--qp N] [--block N] [--sse-scale X]\n"
                 "          [--max-blocks N] [--planes MASK]\n",
                 argv[0]);
    return 2;
  }

  av2enc::Y4mReader reader(options.input);
  std::string error;
  if (!reader.open(error)) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::fprintf(stderr, "read frame: %s\n", error.c_str());
    return 1;
  }

  const int step =
      av2enc::rc::kQlookupQ0To255[std::clamp(options.base_q_idx, 1, 255)];
  std::printf("file: %s  %dx%d  qp=%d  block=%d  sse_scale=%.5f\n",
              options.input.c_str(), static_cast<int>(frame.width()),
              static_cast<int>(frame.height()), options.base_q_idx,
              options.block, options.sse_scale);
  std::printf("qlookup=%d\n", step);

  Stats scalar;
  Stats rdoq;
  Stats tcq;
  if ((options.plane_mask & 1) != 0) {
    if (!run_luma(frame, options, &scalar, &rdoq, &tcq)) {
      std::fprintf(stderr, "luma run failed\n");
      return 1;
    }
    std::printf("luma (%lld blocks)\n",
                static_cast<long long>(scalar.blocks));
    print_row("scalar", scalar);
    print_row("rdoq", rdoq);
    print_row("tcq", tcq);
    std::printf("  rdoq vs scalar cost: %+7.2f%%  bits %+7.2f%%\n",
                percent(scalar.cost, rdoq.cost),
                percent(scalar.bits, rdoq.bits));
    std::printf("  tcq  vs scalar cost: %+7.2f%%  bits %+7.2f%%\n",
                percent(scalar.cost, tcq.cost),
                percent(scalar.bits, tcq.bits));
    std::printf("  tcq  vs rdoq   cost: %+7.2f%%  bits %+7.2f%%\n",
                percent(rdoq.cost, tcq.cost), percent(rdoq.bits, tcq.bits));
  }

  if ((options.plane_mask & 2) != 0) {
    Stats base;
    Stats cctx;
    std::int64_t cctx_types = 0;
    std::int64_t cctx_chosen = 0;
    if (!run_chroma(frame, options, &base, &cctx, &cctx_types,
                    &cctx_chosen)) {
      std::fprintf(stderr, "chroma run failed\n");
      return 1;
    }
    std::printf("chroma (%lld block pairs)\n",
                static_cast<long long>(base.blocks));
    print_row("none", base);
    print_row("cctx", cctx);
    std::printf("  cctx vs none cost: %+7.2f%%  bits %+7.2f%%  "
                "non-NONE=%lld/%lld\n",
                percent(base.cost, cctx.cost), percent(base.bits, cctx.bits),
                static_cast<long long>(cctx_types),
                static_cast<long long>(cctx_chosen));
  }
  return 0;
}
