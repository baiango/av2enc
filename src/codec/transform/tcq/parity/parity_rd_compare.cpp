// Real-coefficient RD comparison for AV2 parity hiding.
//
// Reads a real Y4M frame, forms DC-predicted residual blocks, runs the
// existing `tx_variants` forward transform, and compares five coefficient
// decisions on the same coefficients and the same lambda:
//
//   scalar          - `txv::quantize()`, tool off
//   RDOQ            - `src/encoder/rdoq/`, tool off
//   scalar+PH(dist) - scalar levels + `quantize(kMinDistortion)`
//   RDOQ+PH(dist)   - RDOQ levels + `quantize(kMinDistortion)`
//   RDOQ+PH(RD)     - RDOQ levels + `quantize(kRateDistortion)`
//
// Every candidate is scored with the same static §9.3 model
// (`parity::block_rate_bits`) and the same coefficient SSE, so the comparison
// is apples-to-apples.  The tool-off baselines use the standard DC syntax;
// the PH candidates use `TileCoeffBasePhCdf`, the hidden-DC `maxLevel` and
// `TileDcSignCdf[..][1][..]`.
//
// This is quantisation-only: no partition/mode re-decision and no tile
// re-encode.  See README.md for the blocked full-stream gate.

#include "parity_hiding.h"

#include "codec/entropy/qctx/qctx.h"
#include "codec/rate/rc/rc_tables.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/variants/tx_variants.h"
#include "core/io/y4m.h"

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

using av2enc::tcq::parity::Geometry;
using av2enc::tcq::parity::QuantizeConfig;
using av2enc::tcq::parity::QuantizeMode;
using av2enc::tcq::parity::QuantizeResult;
using av2enc::txv::TxParams;

struct Options {
  std::string input;
  int base_q_idx = 90;
  int block = 8;
  double sse_scale = 0.125;
  int max_blocks = 0;
};

struct Stats {
  std::int64_t blocks = 0;
  std::int64_t nonzero = 0;
  std::int64_t hidden = 0;
  std::int64_t dc_adjust = 0;
  std::int64_t ac_adjust = 0;
  std::int64_t turned_off = 0;
  double sse = 0.0;
  double bits = 0.0;
  double cost = 0.0;

  void add(double sse_value, double bits_value, double lambda,
           std::int64_t nz) {
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

double percent(double from, double to) {
  if (from == 0.0) return 0.0;
  return 100.0 * (to - from) / from;
}

std::vector<std::int32_t> make_scan(int w, int h) {
  std::vector<std::int32_t> scan(static_cast<std::size_t>(w) * h, 0);
  av2enc::tcq::parity::build_scan(w, h,
                                  av2enc::tcq::parity::TxClass::k2D, scan);
  return scan;
}

double levels_distortion(std::span<const double> original,
                         std::span<const std::int32_t> levels,
                         double quant_step) {
  double sum = 0.0;
  for (std::size_t i = 0; i < levels.size(); ++i) {
    const double err = original[i] - static_cast<double>(levels[i]) *
                                        quant_step;
    sum += err * err;
  }
  return sum;
}

std::int64_t count_nonzero(std::span<const std::int32_t> levels) {
  std::int64_t count = 0;
  for (std::int32_t level : levels) {
    if (level != 0) ++count;
  }
  return count;
}

bool baseline_bits(std::span<const std::int32_t> levels,
                   const Geometry& geometry,
                   const av2enc::tcq::parity::RateConfig& rate,
                   double* bits) {
  av2enc::tcq::parity::Eligibility eligibility;  // PH off
  av2enc::tcq::parity::BlockSyntax syntax;
  std::string error;
  if (!av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                          &syntax, &error)) {
    std::fprintf(stderr, "baseline derive: %s\n", error.c_str());
    return false;
  }
  const double value =
      av2enc::tcq::parity::block_rate_bits(syntax, geometry, rate);
  if (value < 0.0) return false;
  *bits = value;
  return true;
}

void print_row(const char* name, const Stats& stats) {
  std::printf("  %-16s nz=%9lld hidden=%7lld dc=%7lld ac=%7lld off=%6lld "
              "sse=%14.3f bits=%12.2f cost=%14.3f\n",
              name, static_cast<long long>(stats.nonzero),
              static_cast<long long>(stats.hidden),
              static_cast<long long>(stats.dc_adjust),
              static_cast<long long>(stats.ac_adjust),
              static_cast<long long>(stats.turned_off), stats.sse,
              stats.bits, stats.cost);
}

bool run(const av2enc::Y4mFrame& frame, const Options& options) {
  const int width = static_cast<int>(frame.width());
  const int height = static_cast<int>(frame.height());
  const int b = options.block;
  if (b < 4 || b > 64 || (b & (b - 1)) != 0) {
    std::fprintf(stderr, "block must be a power of two in [4, 64]\n");
    return false;
  }
  const auto params_opt =
      av2enc::txv::make_tx_params(b, b, av2enc::txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    std::fprintf(stderr, "unsupported block size %d\n", b);
    return false;
  }
  const int step = av2enc::rc::kQlookupQ0To255[std::clamp(
      options.base_q_idx, 1, 255)];
  TxParams params = *params_opt;
  params.quant_factor *= static_cast<double>(av2enc::txv::kQuantStep) /
                         static_cast<double>(step);
  const double quant_step = 1.0 / params.quant_factor;
  const double lambda =
      av2enc::rdo::lambda_sse_from_quant_step(quant_step, options.sse_scale);
  const int qctx =
      av2enc::still::qctx::slice_for_base_q_idx(options.base_q_idx);

  Geometry geometry;
  std::string error;
  if (!av2enc::tcq::parity::make_geometry(
          params.tx_size, av2enc::tcq::parity::TxClass::k2D, 0, &geometry,
          &error)) {
    std::fprintf(stderr, "geometry: %s\n", error.c_str());
    return false;
  }
  const int area = geometry.w * geometry.h;
  const std::vector<std::int32_t> scan = make_scan(geometry.w, geometry.h);

  std::vector<std::int16_t> residual(
      static_cast<std::size_t>(geometry.w) * geometry.h);
  std::vector<double> coeff(
      static_cast<std::size_t>(geometry.w) * geometry.h);
  std::vector<std::int32_t> scalar_raster(
      static_cast<std::size_t>(geometry.w) * geometry.h, 0);

  Stats scalar;
  Stats rdoq;
  Stats scalar_ph_dist;
  Stats rdoq_ph_dist;
  Stats rdoq_ph_rd;

  const std::span<const std::uint8_t> y = frame.y();
  const int stride = width;
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
      const int pred = static_cast<int>((sum + (b * b) / 2) / (b * b));
      av2enc::txv::make_residual(y.data(), stride, width, height, x0, y0,
                                 pred, params, residual.data());
      av2enc::txv::forward_transform(residual.data(), params, coeff.data(),
                                     scalar_raster.data());

      std::vector<std::int32_t> scalar_scan(
          static_cast<std::size_t>(area));
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
      }
      // For distortion we need the real-valued coefficients, not their
      // rounded scan copy.
      std::vector<double> original_scan(static_cast<std::size_t>(area));
      for (int c = 0; c < area; ++c) {
        original_scan[static_cast<std::size_t>(c)] =
            coeff[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])];
      }

      av2enc::tcq::parity::RateConfig rate;
      rate.qctx = qctx;
      rate.txb_skip_ctx = 0;
      rate.dc_sign_ctx = 0;

      // Scalar baseline (tool off).
      {
        double bits = 0.0;
        if (!baseline_bits(scalar_scan, geometry, rate, &bits)) return false;
        scalar.add(levels_distortion(original_scan, scalar_scan, quant_step),
                   bits, lambda, count_nonzero(scalar_scan));
      }

      // RDOQ baseline (tool off).
      av2enc::rdoq::av2::BlockInfo rdoq_info;
      rdoq_info.plane = 0;
      rdoq_info.params = params;
      rdoq_info.txb_skip_context = 0;
      rdoq_info.dc_sign_context = 0;
      rdoq_info.qctx = qctx;
      av2enc::rdoq::av2::Model rdoq_model;
      if (!rdoq_model.init(rdoq_info, scalar_scan, &error)) {
        std::fprintf(stderr, "rdoq model: %s\n", error.c_str());
        return false;
      }
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
      std::vector<std::int32_t> rdoq_levels(
          static_cast<std::size_t>(area), 0);
      for (std::int32_t c = 0; c < rdoq_result.eob; ++c) {
        rdoq_levels[static_cast<std::size_t>(c)] =
            rdoq_result.levels[static_cast<std::size_t>(c)];
      }
      {
        double bits = 0.0;
        if (!baseline_bits(rdoq_levels, geometry, rate, &bits)) return false;
        rdoq.add(levels_distortion(original_scan, rdoq_levels, quant_step),
                 bits, lambda, count_nonzero(rdoq_levels));
      }

      auto run_ph = [&](const std::vector<std::int32_t>& seed,
                        QuantizeMode mode, Stats* stats) -> bool {
        QuantizeConfig config;
        config.eligibility.allow_parity_hiding = true;
        config.mode = mode;
        config.quant_step = quant_step;
        config.lambda = lambda;
        config.rate = rate;
        config.exhaustive_max_area = 64;
        config.pruned_ac_candidates = 8;
        QuantizeResult result;
        if (!av2enc::tcq::parity::quantize(original_scan, seed, geometry,
                                           config, &result, &error)) {
          std::fprintf(stderr, "parity quantize: %s\n", error.c_str());
          return false;
        }
        stats->add(result.distortion, result.rate_bits, lambda,
                   count_nonzero(result.levels));
        if (result.hidden) ++stats->hidden;
        if (result.adjusted_scan_index == 0) ++stats->dc_adjust;
        if (result.adjusted_scan_index > 0) ++stats->ac_adjust;
        if (result.hiding_turned_off) ++stats->turned_off;
        ++stats->blocks;
        return true;
      };
      if (!run_ph(scalar_scan, QuantizeMode::kMinDistortion,
                  &scalar_ph_dist)) {
        return false;
      }
      if (!run_ph(rdoq_levels, QuantizeMode::kMinDistortion,
                  &rdoq_ph_dist)) {
        return false;
      }
      if (!run_ph(rdoq_levels, QuantizeMode::kRateDistortion,
                  &rdoq_ph_rd)) {
        return false;
      }

      ++scalar.blocks;
      ++rdoq.blocks;
      ++blocks;
    }
  }
  if (blocks == 0) {
    std::fprintf(stderr, "no complete blocks\n");
    return false;
  }

  std::printf("blocks=%d block=%dx%d base_q_idx=%d quant_step=%.6f "
              "lambda=%.6f\n",
              blocks, b, b, options.base_q_idx, quant_step, lambda);
  print_row("scalar", scalar);
  print_row("RDOQ", rdoq);
  print_row("scalar+PH(dist)", scalar_ph_dist);
  print_row("RDOQ+PH(dist)", rdoq_ph_dist);
  print_row("RDOQ+PH(RD)", rdoq_ph_rd);
  std::printf(
      "  vs scalar: RDOQ %+.3f%%, scalar+PH(dist) %+.3f%%, "
      "RDOQ+PH(dist) %+.3f%%, RDOQ+PH(RD) %+.3f%%\n",
      percent(scalar.cost, rdoq.cost), percent(scalar.cost, scalar_ph_dist.cost),
      percent(scalar.cost, rdoq_ph_dist.cost),
      percent(scalar.cost, rdoq_ph_rd.cost));
  std::printf(
      "  vs RDOQ: scalar+PH(dist) %+.3f%%, RDOQ+PH(dist) %+.3f%%, "
      "RDOQ+PH(RD) %+.3f%%\n",
      percent(rdoq.cost, scalar_ph_dist.cost),
      percent(rdoq.cost, rdoq_ph_dist.cost),
      percent(rdoq.cost, rdoq_ph_rd.cost));
  std::printf(
      "  bits vs RDOQ: scalar+PH(dist) %+.2f%%, RDOQ+PH(dist) %+.2f%%, "
      "RDOQ+PH(RD) %+.2f%%\n",
      percent(rdoq.bits, scalar_ph_dist.bits),
      percent(rdoq.bits, rdoq_ph_dist.bits),
      percent(rdoq.bits, rdoq_ph_rd.bits));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      options.input = argv[++i];
    } else if (std::strcmp(argv[i], "--qp") == 0 && i + 1 < argc) {
      options.base_q_idx = parse_int(argv[++i], options.base_q_idx);
    } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
      options.block = parse_int(argv[++i], options.block);
    } else if (std::strcmp(argv[i], "--max-blocks") == 0 && i + 1 < argc) {
      options.max_blocks = parse_int(argv[++i], options.max_blocks);
    } else if (std::strcmp(argv[i], "--sse-scale") == 0 && i + 1 < argc) {
      options.sse_scale = parse_double(argv[++i], options.sse_scale);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }
  if (options.input.empty()) {
    std::fprintf(stderr,
                 "usage: %s -i <input.y4m> [--qp N] [--block N] "
                 "[--max-blocks N] [--sse-scale X]\n",
                 argv[0]);
    return 2;
  }
  av2enc::Y4mReader reader(options.input);
  std::string error;
  if (!reader.open(error)) {
    std::fprintf(stderr, "open %s: %s\n", options.input.c_str(),
                 error.c_str());
    return 1;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::fprintf(stderr, "read %s: %s\n", options.input.c_str(),
                 error.c_str());
    return 1;
  }
  return run(frame, options) ? 0 : 1;
}
