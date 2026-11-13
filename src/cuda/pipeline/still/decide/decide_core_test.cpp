// Host parity test for `decide_core.cuh`.
//
// The header is compiled for the device by nvcc (the still GPU orchestrator)
// and for the host here, so the exact same code can be compared with the
// shipped CPU references:
//
//   * the §7.16 coefficient costs against `rdoq::av2::Model` and
//     `tcq::av2::Model`;
//   * the full-trellis RDOQ port against `rdoq::quantize`;
//   * the §9.3 TCQ port against `tcq::quantize`.
//
// Build: ../build_and_test.sh (no nvcc, no GPU).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/tcq/tcq.h"
#include "codec/transform/tcq/tcq_av2_model.h"
#include "codec/transform/variants/tx_variants.h"
#include "cuda/pipeline/still/decide/decide_core.cuh"
#include "cuda/pipeline/still/decide/decide_tables.h"
#include "pipeline/still/still_tables.h"

namespace dec = av2enc::cuda::still::decide;
namespace txv = av2enc::txv;

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    if (failures < 25) std::printf("FAIL: %s\n", what.c_str());
  }
}

const char* tx_size_name(int idx) {
  static const char* names[25] = {
      "4x4",   "8x8",   "16x16", "32x32", "64x64", "4x8",   "8x4",
      "8x16",  "16x8",  "16x32", "32x16", "32x64", "64x32", "4x16",
      "16x4",  "8x32",  "32x8",  "16x64", "64x16", "4x32",  "32x4",
      "8x64",  "64x8",  "4x64",  "64x4"};
  return idx >= 0 && idx < 25 ? names[idx] : "?";
}

struct Block {
  dec::DecTables tables;
  dec::DecBlockInfo info;
  std::vector<double> original;          // scan order
  std::vector<std::int32_t> scalar;      // scan order
  std::vector<std::int32_t> reference;   // scan order
  std::vector<std::int32_t> pre_max;     // scan order
  std::vector<std::int32_t> level_context;
  std::vector<dec::DecPosition> positions;
  std::vector<std::int32_t> level_pre;
};

txv::TxParams make_params(int w, int h) {
  const auto p = txv::make_tx_params(w, h, txv::TxType::kDctDct);
  return *p;
}

void fill_block(Block* b, const dec::DecTables& tables, std::mt19937& rng,
                int plane, int coded_w, int coded_h, double scale) {
  const auto params = *txv::make_tx_params(coded_w, coded_h,
                                           txv::TxType::kDctDct);
  b->tables = tables;
  b->info = dec::DecBlockInfo{};
  b->info.plane = plane;
  b->info.coded_w = params.coded_w;
  b->info.coded_h = params.coded_h;
  b->info.w = params.w;
  b->info.h = params.h;
  b->info.tx_size = params.tx_size;
  b->info.txb_skip_context = 0;
  b->info.dc_sign_context = static_cast<int>(rng() % 3);
  b->info.qctx = 0;
  b->info.is_inter = 0;
  const int area = params.w * params.h;
  std::normal_distribution<double> dist(0.0, scale);
  b->original.resize(static_cast<std::size_t>(area));
  b->scalar.resize(static_cast<std::size_t>(area));
  b->reference.resize(static_cast<std::size_t>(area));
  b->pre_max.resize(static_cast<std::size_t>(area));
  b->level_context.assign(static_cast<std::size_t>(area), 0);
  const double quant_factor = params.quant_factor;
  for (int c = 0; c < area; ++c) {
    double v = dist(rng);
    // Inject the occasional large coefficient so the escape path is covered.
    if ((rng() % 64) == 0) v *= 12.0;
    b->original[static_cast<std::size_t>(c)] = v;
    const std::int32_t scalar = static_cast<std::int32_t>(
        std::lround(v * quant_factor));
    b->scalar[static_cast<std::size_t>(c)] = scalar;
    const int pos = dec::dec_scan_pos(c, params.w, params.h);
    const int row = pos / params.w;
    const int col = pos % params.w;
    b->pre_max[static_cast<std::size_t>(c)] =
        (row + col) < 4 ? 8 : 6;
    const std::int32_t abs_scalar = scalar < 0 ? -scalar : scalar;
    const std::int32_t base = (abs_scalar + 1) / 2;
    b->reference[static_cast<std::size_t>(c)] = scalar < 0 ? -base : base;
  }
  b->positions.resize(static_cast<std::size_t>(area));
  b->level_pre.resize(static_cast<std::size_t>(area));
}

void compare_model_rates(const Block& b, int tcq, std::mt19937& rng) {
  const int area = b.info.w * b.info.h;
  const auto params = make_params(b.info.coded_w, b.info.coded_h);
  // Host model on the same inputs.
  if (tcq == 0) {
    av2enc::rdoq::av2::BlockInfo info;
    info.plane = b.info.plane;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = b.info.dc_sign_context;
    info.qctx = 0;
    av2enc::rdoq::av2::Model model;
    std::string error;
    if (!model.init(info, b.scalar, &error)) {
      check(false, "rdoq model init: " + error);
      return;
    }
    for (int c = 0; c < area; ++c) {
      for (int step = 0; step < 6; ++step) {
        const int magnitude = static_cast<int>(rng() % 40);
        const int sign = (rng() & 1) ? 1 : -1;
        const std::int32_t level = magnitude == 0 ? 0 : sign * magnitude;
        const int is_eob = static_cast<int>(rng() & 1);
        if (is_eob != 0 && level == 0) continue;
        const double host = av2enc::rdoq::av2::Model::coefficient_rate(
            &model, c, 0, level, 0, is_eob);
        const double dev = dec::dec_rdoq_coefficient_bits(
            b.tables, b.info, b.positions.data(), c, level, is_eob);
        check(host == dev || (std::isinf(host) && std::isinf(dev)),
              std::string("rdoq rate ") + tx_size_name(b.info.tx_size) +
                  " c=" + std::to_string(c) + " level=" +
                  std::to_string(level) + " eob=" + std::to_string(is_eob) +
                  " host=" + std::to_string(host) + " dev=" +
                  std::to_string(dev));
      }
    }
    for (int eob = 0; eob <= area; ++eob) {
      const double host = av2enc::rdoq::av2::Model::eob_rate(&model, eob, area);
      const double dev = dec::dec_eob_bits(b.tables, b.info, eob, 0);
      check(host == dev, "rdoq eob bits eob=" + std::to_string(eob));
    }
  } else {
    av2enc::tcq::av2::BlockInfo info;
    info.plane = 0;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = b.info.dc_sign_context;
    info.qctx = 0;
    info.is_inter = false;
    info.tx_class_2d = true;
    av2enc::tcq::av2::Model model;
    std::string error;
    if (!model.init(info, b.reference, &error)) {
      check(false, "tcq model init: " + error);
      return;
    }
    for (int c = 0; c < area; ++c) {
      for (int state = 0; state < 8; ++state) {
        for (int step = 0; step < 4; ++step) {
          const int magnitude = static_cast<int>(rng() % 40);
          const int sign = (rng() & 1) ? 1 : -1;
          const std::int32_t level = magnitude == 0 ? 0 : sign * magnitude;
          const int is_eob = static_cast<int>(rng() & 1);
          if (is_eob != 0 && level == 0) continue;
          const double host = av2enc::tcq::av2::Model::coefficient_rate(
              &model, c, state, level, 0, 0, 0, is_eob);
          const double dev = dec::dec_tcq_coefficient_bits(
              b.tables, b.info, b.positions.data(), c, level, state, is_eob);
          check(host == dev,
                std::string("tcq rate ") + tx_size_name(b.info.tx_size) +
                    " c=" + std::to_string(c) + " level=" +
                    std::to_string(level) + " state=" + std::to_string(state) +
                    " eob=" + std::to_string(is_eob) + " host=" +
                    std::to_string(host) + " dev=" + std::to_string(dev));
        }
      }
    }
    for (int eob = 0; eob <= area; ++eob) {
      const double host = av2enc::tcq::av2::Model::eob_rate(&model, eob, area);
      const double dev = dec::dec_eob_bits(b.tables, b.info, eob, 1);
      check(host == dev, "tcq eob bits eob=" + std::to_string(eob));
    }
  }
}

void compare_rdoq(const Block& b, double sse_scale, std::mt19937& rng) {
  const int area = b.info.w * b.info.h;
  const auto params = make_params(b.info.coded_w, b.info.coded_h);
  av2enc::rdoq::av2::BlockInfo info;
  info.plane = b.info.plane;
  info.params = params;
  info.txb_skip_context = 0;
  info.dc_sign_context = b.info.dc_sign_context;
  info.qctx = 0;
  av2enc::rdoq::av2::Model model;
  std::string error;
  if (!model.init(info, b.scalar, &error)) {
    check(false, "rdoq model init: " + error);
    return;
  }
  std::vector<av2enc::rdoq::Coefficient> input(static_cast<std::size_t>(area));
  for (int c = 0; c < area; ++c) {
    input[static_cast<std::size_t>(c)].original =
        b.original[static_cast<std::size_t>(c)];
    input[static_cast<std::size_t>(c)].scalar_level =
        b.scalar[static_cast<std::size_t>(c)];
    input[static_cast<std::size_t>(c)].level_context = 0;
  }
  av2enc::rdoq::Config config;
  config.quant_step = 1.0 / params.quant_factor;
  config.lambda =
      av2enc::rdo::lambda_sse_from_quant_step(config.quant_step, sse_scale);
  config.max_coefficients = area;
  av2enc::rdoq::RdoqResult host;
  if (!av2enc::rdoq::rdoq_quantize(input, config, model.rate_model(), host,
                                   &error)) {
    check(false, "rdoq host quantize: " + error);
    return;
  }
  dec::DecRdoqConfig dconfig;
  dconfig.quant_step = config.quant_step;
  dconfig.lambda = config.lambda;
  dconfig.prev_level_cap = config.prev_level_cap;
  std::vector<std::int32_t> levels(static_cast<std::size_t>(area), 0);
  std::vector<std::int32_t> chosen_level(
      static_cast<std::size_t>(area) * 3, 0);
  std::vector<std::uint8_t> chosen_mode(static_cast<std::size_t>(area) * 3, 0);
  std::vector<double> trellis_f64(static_cast<std::size_t>(area + 1) * 3);
  std::vector<std::int32_t> trellis_i32(static_cast<std::size_t>(area) * 2);
  dec::DecTrellisScratch scratch;
  scratch.eob_bits = trellis_f64.data();
  scratch.suffix_zero = trellis_f64.data() + (area + 1);
  scratch.tail = trellis_f64.data() + 2 * (area + 1);
  scratch.signs = trellis_i32.data();
  scratch.bounds = trellis_i32.data() + area;
  const dec::DecRdoqResult dev = dec::dec_rdoq_quantize(
      b.tables, b.info, b.positions.data(), b.original.data(), b.scalar.data(),
      area, dconfig, scratch, levels.data(), chosen_level.data(),
      chosen_mode.data());
  check(dev.status == 0, "rdoq device status");
  check(dev.eob == host.eob,
        "rdoq eob " + std::to_string(dev.eob) + " vs " +
            std::to_string(host.eob));
  check(dev.cost == host.cost, "rdoq cost");
  check(dev.rate_bits == host.rate_bits, "rdoq rate");
  check(dev.distortion == host.distortion, "rdoq distortion");
  bool same = true;
  for (int i = 0; i < area; ++i) {
    if (levels[static_cast<std::size_t>(i)] !=
        host.levels[static_cast<std::size_t>(i)]) {
      same = false;
    }
  }
  check(same, "rdoq levels " + std::string(tx_size_name(b.info.tx_size)));
  (void)rng;
}

void compare_tcq(const Block& b, double sse_scale) {
  const int area = b.info.w * b.info.h;
  const auto params = make_params(b.info.coded_w, b.info.coded_h);
  av2enc::tcq::av2::BlockInfo info;
  info.plane = 0;
  info.params = params;
  info.txb_skip_context = 0;
  info.dc_sign_context = b.info.dc_sign_context;
  info.qctx = 0;
  info.is_inter = false;
  info.tx_class_2d = true;
  av2enc::tcq::av2::Model model;
  std::string error;
  if (!model.init(info, b.reference, &error)) {
    check(false, "tcq model init: " + error);
    return;
  }
  std::vector<av2enc::tcq::Coefficient> input(static_cast<std::size_t>(area));
  for (int c = 0; c < area; ++c) {
    input[static_cast<std::size_t>(c)].original =
        b.original[static_cast<std::size_t>(c)];
    input[static_cast<std::size_t>(c)].pre_max =
        b.pre_max[static_cast<std::size_t>(c)];
    input[static_cast<std::size_t>(c)].level_context = 0;
  }
  av2enc::tcq::Config config;
  config.quant_step = 1.0 / params.quant_factor;
  config.lambda =
      av2enc::rdo::lambda_sse_from_quant_step(config.quant_step, sse_scale);
  config.max_coefficients = area;
  av2enc::tcq::Result host;
  if (!av2enc::tcq::quantize(input, config, model.rate_model(), host, &error)) {
    check(false, "tcq host quantize: " + error);
    return;
  }
  dec::DecTcqConfig dconfig;
  dconfig.quant_step = config.quant_step;
  dconfig.lambda = config.lambda;
  std::vector<std::int32_t> base(static_cast<std::size_t>(area), 0);
  std::vector<std::int32_t> quant(static_cast<std::size_t>(area), 0);
  std::vector<std::int32_t> states(static_cast<std::size_t>(area), 0);
  std::vector<dec::DecChoice> dp(static_cast<std::size_t>(area) * 8);
  std::vector<double> trellis_f64(static_cast<std::size_t>(area + 1) * 3);
  std::vector<std::int32_t> trellis_i32(static_cast<std::size_t>(area) * 2);
  dec::DecTrellisScratch scratch;
  scratch.eob_bits = trellis_f64.data();
  scratch.suffix_zero = trellis_f64.data() + (area + 1);
  scratch.tail = trellis_f64.data() + 2 * (area + 1);
  scratch.signs = trellis_i32.data();
  scratch.bounds = trellis_i32.data() + area;
  const dec::DecTcqResult dev = dec::dec_tcq_quantize(
      b.tables, b.info, b.positions.data(), b.original.data(),
      b.pre_max.data(), b.level_context.data(), area, dconfig, scratch,
      base.data(), quant.data(), states.data(), dp.data());
  check(dev.status == 0, "tcq device status");
  check(dev.eob == host.eob, "tcq eob " + std::to_string(dev.eob) + " vs " +
                                std::to_string(host.eob));
  check(dev.cost == host.cost, "tcq cost");
  check(dev.rate_bits == host.rate_bits, "tcq rate");
  check(dev.distortion == host.distortion, "tcq distortion");
  bool same = true;
  for (int i = 0; i < area; ++i) {
    if (base[static_cast<std::size_t>(i)] !=
            host.base_levels[static_cast<std::size_t>(i)] ||
        quant[static_cast<std::size_t>(i)] !=
            host.quant_levels[static_cast<std::size_t>(i)]) {
      same = false;
    }
  }
  check(same, "tcq levels " + std::string(tx_size_name(b.info.tx_size)));
}

}  // namespace

int main() {
  // §9.2 tables must match the shipped copies.
  for (int i = 0; i < 25; ++i) {
    check(dec::kDecTxSizeSqr[i] == av2enc::still::kTxSizeSqr[i],
          "kDecTxSizeSqr copy");
    check(dec::kDecTxSizeSqrUp[i] == av2enc::still::kTxSizeSqrUp[i],
          "kDecTxSizeSqrUp copy");
  }
  // TCQ mappings must match codec/transform/tcq.
  for (int s = 0; s < 8; ++s) {
    for (int p = 0; p < 2; ++p) {
      check(dec::dec_next_state(s, p) == av2enc::tcq::next_state(s, p),
            "next_state copy");
    }
    check(dec::dec_state_q0(s) == av2enc::tcq::state_q0(s), "state_q0 copy");
    check(dec::dec_state_cdf_index(s) == av2enc::tcq::state_cdf_index(s),
          "state_cdf_index copy");
  }
  for (int level = 0; level < 200; ++level) {
    for (int pre = 5; pre <= 8; ++pre) {
      check(dec::dec_syntax_level_for(level, pre) ==
                av2enc::tcq::syntax_level_for(level, pre),
            "syntax_level_for copy");
      check(dec::dec_syntax_escape_for(level, pre) ==
                av2enc::tcq::syntax_escape_for(level, pre),
            "syntax_escape_for copy");
    }
  }

  dec::DecHostTables host_tables;
  if (!dec::build_dec_tables(&host_tables)) {
    std::printf("FAIL: table packing\n");
    return 1;
  }
  const std::vector<double> log2p = dec::build_log2_prob_table();
  dec::DecTables tables = host_tables.view;
  tables.base = host_tables.blob.data();
  tables.log2p = log2p.data();

  std::mt19937 rng(20261113u);
  static const int sizes[][2] = {{4, 4},   {8, 8},   {16, 16}, {32, 32},
                                 {4, 8},   {8, 4},   {8, 16},  {16, 8},
                                 {16, 32}, {32, 16}, {4, 16},  {16, 4},
                                 {8, 32},  {32, 8},  {4, 32},  {32, 4}};
  for (const auto& size : sizes) {
    const int w = size[0];
    const int h = size[1];
    for (int trial = 0; trial < 3; ++trial) {
      // Luma block with the §7.16 model.
      Block block;
      fill_block(&block, tables, rng, 0, w, h,
                 trial == 0 ? 6.0 : (trial == 1 ? 40.0 : 200.0));
      dec::dec_build_positions(block.info, block.scalar.data(), 0,
                               block.level_pre.data(), block.positions.data());
      compare_model_rates(block, 0, rng);
      compare_rdoq(block, 0.125, rng);
      // TCQ is luma/2D only.
      dec::dec_build_positions(block.info, block.reference.data(), 1,
                               block.level_pre.data(), block.positions.data());
      compare_model_rates(block, 1, rng);
      compare_tcq(block, 0.125);
      // Chroma block through the RDOQ model (plane 1/2 contexts).
      Block chroma;
      fill_block(&chroma, tables, rng, 1, w, h, 20.0);
      dec::dec_build_positions(chroma.info, chroma.scalar.data(), 0,
                               chroma.level_pre.data(), chroma.positions.data());
      compare_model_rates(chroma, 0, rng);
      compare_rdoq(chroma, 0.125, rng);
    }
  }

  if (failures == 0) {
    std::printf("%d checks, 0 failures\nRESULT: PASS\n", checks);
    return 0;
  }
  std::printf("%d checks, %d failures\nRESULT: FAIL\n", checks, failures);
  return 1;
}
