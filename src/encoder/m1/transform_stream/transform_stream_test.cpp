// CPU reference tests for the M1 transform-stream adapter.
//
// Suites:
//   * scan order vs the RDOQ AV2 model's scan
//   * §5.20.8.2 symbol classification for all sizes/modes/reduced sets/eob
//   * lambda = 0 recovers scalar quantisation
//   * cost identity, reconstruction identity and determinism
//   * custom rate-model hook plumbing (build/release/type-bits)
//   * real-frame luma blocks from tests/data: validity, TX-type gain and the
//     model-bit / residual-PSNR trade
//
// Build: ./build_and_test.sh [--data DIR]
#include "encoder/m1/transform_stream/transform_stream.h"

#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/rdoq/rdoq_av2_model.h"
#include "encoder/tx_type/tx_type_tables.h"
#include "encoder/tx_variants/tx_tables.h"
#include "encoder/tx_variants/tx_variants.h"
#include "io/y4m.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace ts = av2enc::m1::ts;
using av2enc::txtype::TxType;
using av2enc::txv::TxParams;

int g_checks = 0;
int g_failures = 0;
const char* g_suite = "";

void suite(const char* name) { g_suite = name; }

void check(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL [%s:%d] %s\n", g_suite, line, what);
    if (g_failures > 60) {
      std::printf("too many failures; aborting\n");
      std::exit(1);
    }
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_EQ(a, b)                         \
  do {                                         \
    const auto va = (a);                       \
    const auto vb = (b);                       \
    check(va == vb, #a " == " #b, __LINE__);   \
  } while (0)

int size_count() { return 25; }

TxParams params_for(int index) {
  return *av2enc::txv::make_tx_params(av2enc::txv::kTxSizes[index].width,
                                      av2enc::txv::kTxSizes[index].height);
}

// The rule implemented by luma_tx_type_syntax() in m1_syntax.cpp, extended
// with INTRA_2 for reduced_tx_set 1/3.
ts::TxTypeSyntaxKind expected_kind(int tx_index, int reduced, int eob,
                                   bool dct_only) {
  if (dct_only || reduced == 2 || eob <= 1) {
    return ts::TxTypeSyntaxKind::kNone;
  }
  const int sqr = av2enc::m1::kTxSizeSqr[tx_index];
  const int sqr_up = av2enc::m1::kTxSizeSqrUp[tx_index];
  if (sqr_up > 3) {
    return sqr >= 3 ? ts::TxTypeSyntaxKind::kNone
                    : ts::TxTypeSyntaxKind::kLong64;
  }
  if (sqr_up == 3) {
    return sqr == 3 ? ts::TxTypeSyntaxKind::kNone
                    : ts::TxTypeSyntaxKind::kLong32;
  }
  if (reduced == 1 || reduced == 3) {
    return ts::TxTypeSyntaxKind::kIntraSet2;
  }
  return ts::TxTypeSyntaxKind::kIntraSet1;
}

void test_scan_order() {
  suite("scan");
  for (int index = 0; index < size_count(); ++index) {
    const TxParams params = params_for(index);
    const int area = params.w * params.h;
    std::vector<std::uint16_t> a(static_cast<std::size_t>(area));
    av2enc::txtype::diagonal_scan(params.w, params.h, a.data());
    std::vector<std::int32_t> b(static_cast<std::size_t>(area));
    av2enc::rdoq::av2::build_scan(params.w, params.h, b);
    bool equal = true;
    std::vector<int> seen(static_cast<std::size_t>(area), 0);
    for (int i = 0; i < area; ++i) {
      if (a[static_cast<std::size_t>(i)] !=
          static_cast<std::uint16_t>(b[static_cast<std::size_t>(i)])) {
        equal = false;
      }
      const int pos = a[static_cast<std::size_t>(i)];
      if (pos < 0 || pos >= area) {
        equal = false;
      } else if (seen[static_cast<std::size_t>(pos)] != 0) {
        equal = false;
      } else {
        seen[static_cast<std::size_t>(pos)] = 1;
      }
    }
    CHECK(equal);
  }
}

void test_symbol_classification() {
  suite("symbols");
  for (int index = 0; index < size_count(); ++index) {
    const TxParams params = params_for(index);
    for (int mode = 0; mode < 13; ++mode) {
      for (int reduced : {0, 1, 2, 3}) {
        av2enc::txtype::TxTypeQuery query;
        query.coded_w = params.coded_w;
        query.coded_h = params.coded_h;
        query.mapped_mode = mode;
        query.reduced_tx_set = reduced;
        av2enc::txtype::TxTypeSet set;
        CHECK(av2enc::txtype::intra_tx_candidates(query, &set));
        for (int eob : {0, 1, 2, 3, params.w * params.h}) {
          const ts::TxTypeSyntaxKind want =
              expected_kind(index, reduced, eob, false);
          for (int t = 0; t < 16; ++t) {
            const TxType type = static_cast<TxType>(t);
            const ts::TxTypeSymbol symbol =
                ts::classify_tx_type(query, type, eob);
            const bool in_set = [&]() {
              for (int k = 0; k < set.count; ++k) {
                if (set.candidates[k].type == type) return true;
              }
              return false;
            }();
            if (!in_set) {
              CHECK(!symbol.valid);
              continue;
            }
            if (want == ts::TxTypeSyntaxKind::kNone) {
              // eob <= 1, reduced_tx_set == 2 or DCTONLY: only DCT_DCT is
              // signallable.
              CHECK_EQ(symbol.valid, type == TxType::kDctDct);
              if (symbol.valid) {
                CHECK_EQ(symbol.symbol, -1);
                CHECK_EQ(symbol.bits, 0.0);
                CHECK(!symbol.has_long_flag);
                CHECK(!symbol.is_long_side_dct);
              }
              continue;
            }
            CHECK(symbol.valid);
            CHECK_EQ(static_cast<int>(symbol.kind),
                     static_cast<int>(want));
            const auto match = [&]() -> const av2enc::txtype::TxTypeCandidate* {
              for (int k = 0; k < set.count; ++k) {
                if (set.candidates[k].type == type) return &set.candidates[k];
              }
              return nullptr;
            }();
            CHECK(match != nullptr);
            if (match == nullptr) continue;
            CHECK_EQ(static_cast<int>(symbol.symbol),
                     static_cast<int>(match->symbol));
            CHECK_EQ(symbol.has_long_flag, match->codes_long_flag);
            CHECK_EQ(symbol.is_long_side_dct, match->is_long_side_dct);
            CHECK(symbol.bits >= 0.0 && std::isfinite(symbol.bits));
            const int cdf_symbols =
                want == ts::TxTypeSyntaxKind::kIntraSet1
                    ? 7
                    : (want == ts::TxTypeSyntaxKind::kIntraSet2 ? 2 : 4);
            CHECK_EQ(static_cast<int>(symbol.cdf_symbols), cdf_symbols);
            const double static_bits =
                av2enc::txtype::static_type_bits(query, *match);
            CHECK_EQ(symbol.bits, static_bits);
          }
        }
        // Chroma / forced-DCT rule.
        query.dct_only = true;
        for (int eob : {0, 1, 2, 7}) {
          const ts::TxTypeSymbol dct = ts::classify_tx_type(
              query, TxType::kDctDct, eob);
          CHECK(dct.valid);
          CHECK_EQ(static_cast<int>(dct.kind),
                   static_cast<int>(ts::TxTypeSyntaxKind::kNone));
          CHECK_EQ(dct.symbol, -1);
          CHECK_EQ(dct.bits, 0.0);
          if (eob >= 2) {
            const ts::TxTypeSymbol adst = ts::classify_tx_type(
                query, TxType::kAdstAdst, eob);
            CHECK(!adst.valid);
          }
        }
      }
    }
  }
}

std::vector<std::int16_t> make_residual(int w, int h, std::uint32_t seed) {
  std::vector<std::int16_t> residual(static_cast<std::size_t>(w * h));
  std::uint32_t state = seed * 1664525u + 1013904223u;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      state = state * 1664525u + 1013904223u;
      const int noise = static_cast<int>((state >> 24) % 91) - 45;
      const int value = (x * 5 - y * 3) + noise;
      residual[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
               static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(std::clamp(value, -255, 255));
    }
  }
  return residual;
}

ts::Request base_request(const std::vector<std::int16_t>& residual, int w,
                         int h) {
  ts::Request request;
  request.residual_coded = residual.data();
  request.coded_w = w;
  request.coded_h = h;
  request.intra_mode = 0;
  request.angle_delta = 0;
  request.reduced_tx_set = 0;
  request.plane = 0;
  request.txb_skip_context = 0;
  request.dc_sign_context = 0;
  // M1 signals enable_idtx_intra = 0; the adapter defaults to the component's
  // full set, so the replay suites opt into the M1 rule explicitly.
  request.enable_idtx_intra = false;
  return request;
}

void test_lambda_zero_is_scalar() {
  suite("lambda0");
  for (int index : {0, 1, 2, 3, 4, 5, 9, 13, 17}) {
    const TxParams params = params_for(index);
    const auto residual =
        make_residual(params.coded_w, params.coded_h,
                      static_cast<std::uint32_t>(index + 1));
    ts::Request request =
        base_request(residual, params.coded_w, params.coded_h);
    request.lambda = 0.0;
    const TxType dct_type = TxType::kDctDct;
    request.candidate_types = std::span<const TxType>(&dct_type, 1);
    ts::Decision rdoq;
    ts::Decision scalar;
    std::string error;
    CHECK(ts::quantize_block(request, &rdoq, &error));
    CHECK(ts::quantize_block_scalar(request, &scalar, &error));
    CHECK(rdoq.valid && scalar.valid);
    CHECK_EQ(rdoq.eob, scalar.eob);
    CHECK_EQ(rdoq.levels, scalar.levels);
    CHECK_EQ(rdoq.type, TxType::kDctDct);
  }
}

void test_cost_and_recon() {
  suite("cost/recon");
  for (int index : {0, 1, 2, 9, 10, 19}) {
    const TxParams params = params_for(index);
    const auto residual =
        make_residual(params.coded_w, params.coded_h,
                      static_cast<std::uint32_t>(index + 100));
    ts::Request request =
        base_request(residual, params.coded_w, params.coded_h);
    request.lambda = ts::default_lambda(params);
    CHECK(request.lambda > 0.0);
    ts::Decision decision;
    std::string error;
    CHECK(ts::quantize_block(request, &decision, &error));
    CHECK(decision.valid);
    CHECK_EQ(decision.cost,
             decision.sse + request.lambda * decision.rate_bits);
    CHECK_EQ(decision.rate_bits,
             decision.coeff_rate_bits + decision.type_bits);
    CHECK(decision.eob >= 0);
    CHECK(static_cast<int>(decision.levels.size()) ==
          params.w * params.h);
    for (int c = decision.eob; c < params.w * params.h; ++c) {
      CHECK_EQ(decision.levels[static_cast<std::size_t>(c)], 0);
    }

    // Rebuild the winner's reconstruction independently.
    std::vector<std::uint16_t> scan(
        static_cast<std::size_t>(params.w * params.h));
    av2enc::txtype::diagonal_scan(params.w, params.h, scan.data());
    std::vector<std::int32_t> raster(
        static_cast<std::size_t>(params.w * params.h), 0);
    for (int c = 0; c < decision.eob; ++c) {
      raster[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] =
          decision.levels[static_cast<std::size_t>(c)];
    }
    std::vector<std::int32_t> dequant(
        static_cast<std::size_t>(params.w * params.h));
    std::vector<std::int16_t> recon(
        static_cast<std::size_t>(params.coded_w * params.coded_h));
    av2enc::txv::inverse_transform(raster.data(), decision.params,
                                   dequant.data(), recon.data());
    CHECK(recon == decision.recon_coded);

    ts::Decision again;
    CHECK(ts::quantize_block(request, &again, &error));
    CHECK_EQ(again.eob, decision.eob);
    CHECK_EQ(again.levels, decision.levels);
    CHECK_EQ(again.cost, decision.cost);
    CHECK_EQ(static_cast<int>(again.type), static_cast<int>(decision.type));

    std::vector<ts::Decision> all;
    CHECK(ts::evaluate_candidates(request, &all, &error));
    CHECK(!all.empty());
    CHECK_EQ(static_cast<int>(all.size()), decision.candidates_considered);
    const ts::Decision* best = nullptr;
    for (const ts::Decision& candidate : all) {
      CHECK(candidate.valid);
      if (!candidate.codable) continue;
      if (best == nullptr || candidate.cost < best->cost ||
          (candidate.cost == best->cost && candidate.sse < best->sse) ||
          (candidate.cost == best->cost && candidate.sse == best->sse &&
           candidate.type < best->type)) {
        best = &candidate;
      }
    }
    CHECK(best != nullptr);
    if (best != nullptr) {
      CHECK_EQ(static_cast<int>(best->type), static_cast<int>(decision.type));
      CHECK_EQ(best->eob, decision.eob);
      CHECK_EQ(best->levels, decision.levels);
    }
  }
}

struct HookState {
  int builds = 0;
  int releases = 0;
  int type_calls = 0;
  double type_bits = 42.0;
};

double hook_coefficient_rate(void* context, std::int32_t scan_pos,
                             std::int32_t prev_abs_state, std::int32_t level,
                             std::int32_t level_context, std::int32_t is_eob) {
  (void)context;
  (void)level_context;
  const int magnitude = level < 0 ? -level : level;
  double bits = 0.35 + 0.15 * static_cast<double>(magnitude);
  if (prev_abs_state == 0) bits += 0.2;
  if (is_eob != 0) bits += 0.1;
  bits += 0.001 * static_cast<double>(scan_pos);
  return bits;
}

double hook_eob_rate(void* context, std::int32_t eob,
                     std::int32_t num_coefficients) {
  (void)context;
  (void)num_coefficients;
  return 1.0 + 0.25 * std::log2(static_cast<double>(eob) + 1.0);
}

bool hook_build(void* user, const ts::RateModelRequest& request,
                av2enc::rdoq::RateModel* model, std::string* error) {
  (void)request;
  if (error != nullptr) error->clear();
  HookState* state = static_cast<HookState*>(user);
  ++state->builds;
  model->context = state;
  model->coefficient_rate = &hook_coefficient_rate;
  model->eob_rate = &hook_eob_rate;
  return true;
}

void hook_release(void* user, av2enc::rdoq::RateModel* model) noexcept {
  (void)model;
  ++static_cast<HookState*>(user)->releases;
}

double hook_type_bits(void* user, const av2enc::txtype::TxTypeQuery& query,
                      const av2enc::txtype::TxTypeCandidate& candidate) {
  (void)query;
  (void)candidate;
  HookState* state = static_cast<HookState*>(user);
  ++state->type_calls;
  return state->type_bits;
}

void test_custom_hook() {
  suite("custom-hook");
  const TxParams params = params_for(2);  // 16x16, INTRA_1
  const auto residual = make_residual(params.coded_w, params.coded_h, 777);
  ts::Request request =
      base_request(residual, params.coded_w, params.coded_h);
  request.lambda = 10.0;
  HookState state;
  ts::RateModelHooks hooks;
  hooks.build = &hook_build;
  hooks.release = &hook_release;
  hooks.user = &state;
  hooks.type_bits = &hook_type_bits;
  request.hooks = &hooks;
  ts::Decision decision;
  std::string error;
  CHECK(ts::quantize_block(request, &decision, &error));
  CHECK(decision.valid);
  CHECK_EQ(state.builds, decision.candidates_considered);
  CHECK_EQ(state.releases, state.builds);
  CHECK(state.type_calls >= 1);
  if (decision.symbol.kind != ts::TxTypeSyntaxKind::kNone) {
    CHECK_EQ(decision.type_bits, state.type_bits);
  } else {
    CHECK_EQ(decision.type_bits, 0.0);
  }

  request.hooks = nullptr;
  ts::Decision plain;
  CHECK(ts::quantize_block(request, &plain, &error));
  CHECK(plain.valid);
  CHECK(plain.type_bits <= 8.0);  // §9.3 static cost, not the hook's 42.
}

void test_idtx_filter_and_subset() {
  suite("idtx/subset");
  const TxParams params = params_for(9);  // 16x32, WIDE_32 long set
  const auto residual = make_residual(16, 32, 4242);
  ts::Request request = base_request(residual, 16, 32);
  request.lambda = ts::default_lambda(params);
  std::vector<ts::Decision> all;
  std::vector<ts::Decision> filtered;
  std::string error;
  request.enable_idtx_intra = true;
  CHECK(ts::evaluate_candidates(request, &all, &error));
  request.enable_idtx_intra = false;
  CHECK(ts::evaluate_candidates(request, &filtered, &error));
  CHECK(filtered.size() <= all.size());
  CHECK(!filtered.empty());
  for (const ts::Decision& candidate : filtered) {
    av2enc::txv::Tx1dType row = av2enc::txv::Tx1dType::kDct;
    av2enc::txv::Tx1dType col = av2enc::txv::Tx1dType::kDct;
    av2enc::txv::tx_1d_types(candidate.type, &row, &col);
    CHECK(row != av2enc::txv::Tx1dType::kIdt);
    CHECK(col != av2enc::txv::Tx1dType::kIdt);
  }
  ts::Decision decision;
  CHECK(ts::quantize_block(request, &decision, &error));
  CHECK(decision.valid);

  // Explicit candidate subset is honoured.
  request.enable_idtx_intra = true;
  const TxType subset[] = {TxType::kDctDct, TxType::kDctAdst};
  request.candidate_types = std::span<const TxType>(subset, 2);
  CHECK(ts::evaluate_candidates(request, &all, &error));
  CHECK_EQ(all.size(), static_cast<std::size_t>(2));
  CHECK_EQ(static_cast<int>(all[0].type), static_cast<int>(TxType::kDctDct));
  CHECK_EQ(static_cast<int>(all[1].type), static_cast<int>(TxType::kDctAdst));
  CHECK(ts::quantize_block(request, &decision, &error));
  CHECK(decision.valid);

  // A filtered-to-empty list is an error.
  const TxType idtx = TxType::kIdtx;
  request.candidate_types = std::span<const TxType>(&idtx, 1);
  request.enable_idtx_intra = false;
  CHECK(!ts::quantize_block(request, &decision, &error));
}

void test_invalid() {
  suite("invalid");
  const TxParams params = params_for(1);  // 8x8
  const auto residual = make_residual(8, 8, 3);
  ts::Request request = base_request(residual, 8, 8);
  ts::Decision decision;
  std::string error;

  ts::Request bad = request;
  bad.residual_coded = nullptr;
  CHECK(!ts::quantize_block(bad, &decision, &error));
  CHECK(!decision.valid);

  bad = request;
  bad.coded_w = 5;
  bad.coded_h = 5;
  CHECK(!ts::quantize_block(bad, &decision, &error));

  bad = request;
  bad.intra_mode = 13;
  CHECK(!ts::quantize_block(bad, &decision, &error));

  bad = request;
  bad.lambda = std::numeric_limits<double>::quiet_NaN();
  CHECK(!ts::quantize_block(bad, &decision, &error));

  bad = request;
  bad.prev_level_cap = 64;
  CHECK(!ts::quantize_block(bad, &decision, &error));

  bad = request;
  bad.reduced_tx_set = 4;
  CHECK(!ts::quantize_block(bad, &decision, &error));

  bad = request;
  const TxType illegal = TxType::kIdtx;
  bad.candidate_types = std::span<const TxType>(&illegal, 1);
  // IDTX is not in every INTRA_1 mode table; use a 32x32 DCTONLY block where
  // only DCT_DCT is legal to make the expectation unconditional.
  const auto residual32 = make_residual(32, 32, 9);
  bad.residual_coded = residual32.data();
  bad.coded_w = 32;
  bad.coded_h = 32;
  CHECK(!ts::quantize_block(bad, &decision, &error));

  CHECK(ts::quantize_block(request, &decision, &error));
  (void)params;
}

struct RealStats {
  long long blocks = 0;
  long long samples = 0;
  long long nonzero = 0;
  long long eob_sum = 0;
  double sse = 0.0;
  double bits = 0.0;
  double cost = 0.0;
  long long types[16] = {};
};

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> luma;
};

bool load_frame(const std::filesystem::path& path, Frame* frame) {
  av2enc::Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::printf("  cannot open %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  av2enc::Y4mFrame y4m;
  if (reader.read_frame(y4m, error) != av2enc::Y4mReadResult::kFrame) {
    std::printf("  cannot read %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  frame->width = static_cast<int>(y4m.width());
  frame->height = static_cast<int>(y4m.height());
  const std::span<const std::uint8_t> y = y4m.y();
  frame->luma.assign(y.begin(), y.end());
  return true;
}

bool predict_dc(const Frame& frame, int x0, int y0, int w, int h,
                std::vector<std::uint8_t>* pred) {
  av2enc::intra::NeighborRequest request;
  request.recon.data = frame.luma.data();
  request.recon.stride = frame.width;
  request.x = x0;
  request.y = y0;
  request.w = w;
  request.h = h;
  request.max_x = frame.width - 1;
  request.max_y = frame.height - 1;
  request.have_above = y0 > 0;
  request.have_left = x0 > 0;
  request.color_plane = 0;
  av2enc::intra::PredictParams params;
  params.mode = av2enc::intra::Mode::kDc;
  pred->resize(static_cast<std::size_t>(w * h));
  return av2enc::intra::predict_block(request, 8, params, pred->data(),
                                      static_cast<std::size_t>(w)) ==
         av2enc::intra::IntraStatus::kOk;
}

RealStats run_real_blocks(const Frame& frame, int block_w, int block_h,
                          bool full_types, int max_blocks) {
  RealStats stats;
  std::vector<std::uint8_t> pred;
  std::vector<std::int16_t> residual(
      static_cast<std::size_t>(block_w * block_h));
  const TxParams params = *av2enc::txv::make_tx_params(block_w, block_h);
  int done = 0;
  for (int y0 = 0; y0 + block_h <= frame.height; y0 += block_h) {
    for (int x0 = 0; x0 + block_w <= frame.width; x0 += block_w) {
      if (max_blocks > 0 && done >= max_blocks) return stats;
      if (!predict_dc(frame, x0, y0, block_w, block_h, &pred)) continue;
      for (int y = 0; y < block_h; ++y) {
        for (int x = 0; x < block_w; ++x) {
          const int src = frame.luma[static_cast<std::size_t>(y0 + y) *
                                         static_cast<std::size_t>(frame.width) +
                                     static_cast<std::size_t>(x0 + x)];
          const int p = pred[static_cast<std::size_t>(y * block_w + x)];
          residual[static_cast<std::size_t>(y * block_w + x)] =
              static_cast<std::int16_t>(src - p);
        }
      }
      ts::Request request =
          base_request(residual, block_w, block_h);
      request.lambda = ts::default_lambda(params);
      std::array<TxType, 1> dct_only = {TxType::kDctDct};
      if (!full_types) request.candidate_types = dct_only;
      ts::Decision decision;
      std::string error;
      if (!ts::quantize_block(request, &decision, &error)) {
        std::printf("  real block failed (%d,%d): %s\n", x0, y0,
                    error.c_str());
        return stats;
      }
      CHECK(decision.valid);
      CHECK(decision.symbol.valid);
      CHECK(decision.eob >= 0 && decision.eob <= params.w * params.h);
      stats.blocks += 1;
      stats.samples +=
          static_cast<long long>(decision.params.coded_w) *
          decision.params.coded_h;
      stats.nonzero += decision.nonzero;
      stats.eob_sum += decision.eob;
      stats.sse += decision.sse;
      stats.bits += decision.rate_bits;
      stats.cost += decision.cost;
      if (decision.type >= TxType::kDctDct &&
          static_cast<int>(decision.type) < 16) {
        stats.types[static_cast<int>(decision.type)] += 1;
      }
      ++done;
    }
  }
  return stats;
}

void print_real(const char* label, const RealStats& stats) {
  const double psnr =
      stats.sse > 0.0 && stats.samples > 0
          ? 10.0 * std::log10(255.0 * 255.0 *
                              static_cast<double>(stats.samples) / stats.sse)
          : 0.0;
  std::printf("  %-22s %5lld blocks bits %10.1f  sse %12.1f  PSNR %6.3f dB"
              "  nz %7lld  eob %8lld\n",
              label, stats.blocks, stats.bits, stats.sse, psnr, stats.nonzero,
              stats.eob_sum);
}

void test_real_frames(const std::filesystem::path& data_dir) {
  suite("real-frames");
  struct Case {
    const char* file;
    int block_w;
    int block_h;
    int max_blocks;
  };
  const Case cases[] = {
      {"m0_64x64_1f.y4m", 64, 64, 0},
      {"m0_64x64_1f.y4m", 16, 16, 0},
      {"m0_640x360_1f.y4m", 16, 16, 0},
      {"m0_640x360_1f.y4m", 32, 32, 0},
  };
  for (const Case& item : cases) {
    const std::filesystem::path path = data_dir / item.file;
    Frame frame;
    if (!load_frame(path, &frame)) {
      std::printf("  SKIP %s (missing test media)\n", path.c_str());
      continue;
    }
    const RealStats dct =
        run_real_blocks(frame, item.block_w, item.block_h, false,
                        item.max_blocks);
    const RealStats full =
        run_real_blocks(frame, item.block_w, item.block_h, true,
                        item.max_blocks);
    if (dct.blocks == 0 || full.blocks == 0) {
      CHECK(false);
      continue;
    }
    CHECK_EQ(dct.blocks, full.blocks);
    CHECK(full.cost <= dct.cost + 1e-9);
    char label[96];
    std::snprintf(label, sizeof(label), "DCT_DCT %dx%d", item.block_w,
                  item.block_h);
    print_real(label, dct);
    std::snprintf(label, sizeof(label), "TX-type+RDOQ %dx%d", item.block_w,
                  item.block_h);
    print_real(label, full);
    long long non_dct = 0;
    long long hist[16] = {};
    for (int t = 0; t < 16; ++t) {
      hist[t] = full.types[t];
      if (t != static_cast<int>(TxType::kDctDct)) non_dct += full.types[t];
    }
    std::printf("  %-22s non-DCT wins %lld / %lld", "type histogram",
                non_dct, full.blocks);
    for (int t = 0; t < 16; ++t) {
      if (hist[t] != 0) std::printf("  %d:%lld", t, hist[t]);
    }
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "tests/data";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--data" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (std::string(argv[i]) == "--help") {
      std::printf("usage: %s [--data DIR]\n", argv[0]);
      return 0;
    } else {
      std::printf("unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  std::printf("av2enc M1 transform-stream adapter test\n");
  test_scan_order();
  test_symbol_classification();
  test_lambda_zero_is_scalar();
  test_cost_and_recon();
  test_custom_hook();
  test_idtx_filter_and_subset();
  test_invalid();
  test_real_frames(data_dir);

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
