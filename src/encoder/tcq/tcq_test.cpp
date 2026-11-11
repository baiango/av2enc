// CPU reference tests for AV2 TCQ / CCTX.
//
// The exhaustive tests enumerate every canonical base-level path for a small
// block and compare the trellis against an independent forward cost evaluator
// built on `decode_levels()` (the normative mapping).

#include "cctx.h"
#include "tcq.h"
#include "tcq_flags.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void expect(bool condition, const char* what) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

void expect_near(double a, double b, double tolerance, const char* what) {
  ++checks;
  const double diff = std::fabs(a - b);
  const double scale = 1.0 + std::max(std::fabs(a), std::fabs(b));
  if (!(diff <= tolerance * scale)) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s (%.17g vs %.17g)\n", what, a, b);
  }
}

std::uint32_t rng_state = 0x9e3779b9u;

std::uint32_t next_random() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

double uniform(double low, double high) {
  const double unit =
      static_cast<double>(next_random() & 0xffffffu) / 16777215.0;
  return low + (high - low) * unit;
}

// ---------------------------------------------------------------------------
// Rate models used by the tests
// ---------------------------------------------------------------------------

double test_coefficient_rate(void* context, std::int32_t scan_pos,
                             std::int32_t state, std::int32_t base_level,
                             std::int32_t next_state,
                             std::int32_t signed_quant,
                             std::int32_t level_context,
                             std::int32_t is_eob) {
  (void)context;
  (void)next_state;
  (void)signed_quant;
  (void)level_context;
  const std::int32_t magnitude = std::abs(base_level);
  // Deliberately non-monotone in a few places so the DP cannot assume a
  // monotone cost.
  double bits = magnitude == 0 ? 0.65 : 1.1 + 0.31 * magnitude;
  bits += 0.07 * ((state * 7 + scan_pos * 3) % 5);
  if (is_eob != 0) bits += 0.37;
  return bits;
}

double test_eob_rate(void* context, std::int32_t eob,
                     std::int32_t num_coefficients) {
  (void)context;
  (void)num_coefficients;
  return eob == 0 ? 1.9 : 0.8 + 0.17 * eob;
}

av2enc::tcq::RateModel test_model() {
  av2enc::tcq::RateModel model;
  model.context = nullptr;
  model.coefficient_rate = &test_coefficient_rate;
  model.eob_rate = &test_eob_rate;
  return model;
}

// ---------------------------------------------------------------------------
// Independent forward evaluator
// ---------------------------------------------------------------------------

struct PathCost {
  double cost = std::numeric_limits<double>::infinity();
  double distortion = 0.0;
  double rate = 0.0;
  std::vector<std::int32_t> base_levels;
  std::vector<std::int32_t> quant_levels;
  std::int32_t eob = 0;
};

PathCost evaluate(const std::vector<std::int32_t>& base,
                  const std::vector<double>& original,
                  const av2enc::tcq::Config& config,
                  const av2enc::tcq::RateModel& model) {
  PathCost out;
  out.base_levels = base;
  av2enc::tcq::DecodeResult decoded;
  if (!av2enc::tcq::decode_levels(base, /*allow_tcq=*/true,
                                  /*lossless=*/false, &decoded, nullptr)) {
    return out;
  }
  out.quant_levels = decoded.quant_levels;
  out.eob = decoded.eob;
  double distortion = 0.0;
  double rate = 0.0;
  const double half = config.quant_step * 0.5;
  for (std::size_t i = static_cast<std::size_t>(out.eob); i < original.size();
       ++i) {
    distortion += original[i] * original[i];
  }
  for (std::int32_t c = 0; c < out.eob; ++c) {
    const double error =
        original[static_cast<std::size_t>(c)] -
        static_cast<double>(decoded.quant_levels[static_cast<std::size_t>(c)]) *
            half;
    distortion += error * error;
    const bool is_eob = c == out.eob - 1;
    rate += model.coefficient_rate(
        model.context, c, decoded.states[static_cast<std::size_t>(c)],
        base[static_cast<std::size_t>(c)],
        av2enc::tcq::next_state(decoded.states[static_cast<std::size_t>(c)],
                                std::abs(base[static_cast<std::size_t>(c)]) &
                                    1),
        decoded.quant_levels[static_cast<std::size_t>(c)], 0,
        is_eob ? 1 : 0);
  }
  rate += model.eob_rate(model.context, out.eob,
                         static_cast<std::int32_t>(base.size()));
  out.distortion = distortion;
  out.rate = rate;
  out.cost = distortion + config.lambda * rate;
  return out;
}

void enumerate_paths(std::size_t index, std::int32_t max_level,
                     std::vector<std::int32_t>* magnitudes,
                     const std::vector<double>& original,
                     const av2enc::tcq::Config& config,
                     const av2enc::tcq::RateModel& model, PathCost* best) {
  if (index == magnitudes->size()) {
    std::vector<std::int32_t> signed_levels(magnitudes->size(), 0);
    for (std::size_t i = 0; i < magnitudes->size(); ++i) {
      const int sign = original[i] < 0.0 ? -1 : 1;
      signed_levels[i] = (*magnitudes)[i] == 0 ? 0 : sign * (*magnitudes)[i];
    }
    const PathCost candidate =
        evaluate(signed_levels, original, config, model);
    if (candidate.cost < best->cost) *best = candidate;
    return;
  }
  for (std::int32_t level = 0; level <= max_level; ++level) {
    (*magnitudes)[index] = level;
    enumerate_paths(index + 1, max_level, magnitudes, original, config, model,
                    best);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_state_table() {
  static constexpr int expected[8][2] = {
      {0, 4}, {4, 0}, {1, 5}, {5, 1},
      {6, 2}, {2, 6}, {7, 3}, {3, 7},
  };
  for (int s = 0; s < 8; ++s) {
    for (int p = 0; p < 2; ++p) {
      expect(av2enc::tcq::kNextState[s][p] == expected[s][p],
             "Tcq_Next_State matches §5.20.7.27");
      expect(av2enc::tcq::next_state(s, p) == expected[s][p],
             "next_state helper matches the table");
    }
  }
  // q0 is bit 1; it is also the coeff_base CDF selector.
  static constexpr int expected_q0[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  for (int s = 0; s < 8; ++s) {
    expect(av2enc::tcq::state_q0(s) == expected_q0[s],
           "state_q0 matches (state >> 1) & 1");
    expect(av2enc::tcq::state_cdf_index(s) == expected_q0[s],
           "state_cdf_index matches (state >> 1) & 1");
  }
}

void test_dequant_shift() {
  using av2enc::tcq::dequant_shift_extra;
  expect(dequant_shift_extra(true, 0, false, true, false) == 1,
         "TCQ luma 2D lossy adds one dequant shift");
  expect(dequant_shift_extra(true, 1, false, true, false) == 0,
         "TCQ never applies to chroma");
  expect(dequant_shift_extra(true, 0, true, true, false) == 0,
         "TCQ is disabled in lossless");
  expect(dequant_shift_extra(true, 0, false, false, false) == 0,
         "TCQ requires TX_CLASS_2D");
  expect(dequant_shift_extra(true, 0, false, true, true) == 0,
         "TCQ requires !useFsc");
  expect(dequant_shift_extra(false, 0, false, true, false) == 0,
         "allow_tcq=0 adds no shift");
  expect(av2enc::tcq::dequant_shift(4096, true, 0, false, true, false) == 3,
         "dequant_shift = base shift + TCQ shift");
  expect(av2enc::tcq::dequant_shift(1024, true, 0, false, true, false) == 2,
         "dequant_shift 32x32 = dq_denom 1 + TCQ 1");
}

void test_decoder_mapping() {
  using av2enc::tcq::DecodeResult;
  using av2enc::tcq::decode_levels;
  std::vector<std::int32_t> levels = {1, 1, 1};
  DecodeResult decoded;
  expect(decode_levels(levels, true, false, &decoded, nullptr),
         "decode_levels succeeds");
  // Reverse scan: c2 state 0 -> quant 2, next 4; c1 state 4 -> quant 2,
  // next 2; c0 state 2 (q0=1) -> quant 2*1-1 = 1, next 5.
  expect(decoded.quant_levels[2] == 2, "decoder c2 quant");
  expect(decoded.quant_levels[1] == 2, "decoder c1 quant");
  expect(decoded.quant_levels[0] == 1, "decoder c0 quant with q0=1");
  expect(decoded.states[2] == 0 && decoded.states[1] == 4 &&
             decoded.states[0] == 2,
         "decoder state sequence");

  // Disabled TCQ is the identity mapping.
  expect(decode_levels(levels, false, false, &decoded, nullptr),
         "decode_levels disabled succeeds");
  expect(decoded.quant_levels == levels, "allow_tcq=0 is identity");
  expect(decode_levels(levels, true, true, &decoded, nullptr),
         "decode_levels lossless succeeds");
  expect(decoded.quant_levels == levels, "lossless is identity");

  // Every full level is representable: pre_max-1 and pre_max cover both
  // parities once the read_quant escape is active.
  expect(av2enc::tcq::syntax_level_for(4, 6) == 4, "syntax level below cap");
  expect(av2enc::tcq::syntax_level_for(5, 6) == 5, "syntax level pre_max-1");
  expect(av2enc::tcq::syntax_level_for(6, 6) == 6, "syntax level pre_max");
  expect(av2enc::tcq::syntax_level_for(7, 6) == 5, "odd level above cap");
  expect(av2enc::tcq::syntax_level_for(8, 6) == 6, "even level above cap");
  expect(av2enc::tcq::syntax_level_for(9, 8) == 7, "LF odd level above cap");
  expect(av2enc::tcq::syntax_level_for(10, 8) == 8, "LF even level above cap");
  expect(av2enc::tcq::syntax_escape_for(4, 6) == 0,
         "no escape below pre_max-1");
  expect(av2enc::tcq::syntax_escape_for(5, 6) == 0,
         "escape x=0 at pre_max-1");
  expect(av2enc::tcq::syntax_escape_for(6, 6) == 0,
         "escape x=0 at pre_max");
  expect(av2enc::tcq::syntax_escape_for(7, 6) == 1, "odd above-cap x");
  expect(av2enc::tcq::syntax_escape_for(8, 6) == 1, "even above-cap x");
  expect(av2enc::tcq::syntax_escape_for(9, 6) == 2, "next odd above-cap x");

  std::vector<std::int32_t> large = {8};
  expect(decode_levels(large, true, false, &decoded, nullptr),
         "even level above pre_max decodes");
  expect(decoded.quant_levels[0] == 16, "large level maps through q0=0");
  std::vector<std::int32_t> odd_large = {7};
  expect(decode_levels(odd_large, true, false, &decoded, nullptr),
         "odd level above pre_max decodes");
  expect(decoded.quant_levels[0] == 14, "odd large level maps through q0=0");
  std::vector<std::int32_t> pre_max8 = {9};
  std::vector<std::int32_t> pre_max_scan = {8};
  expect(decode_levels(pre_max8, true, false, &decoded, nullptr, pre_max_scan),
         "LF pre_max=8 accepts an odd level above 8");
  expect(decoded.quant_levels[0] == 18, "LF odd large level maps");

  // Signed levels keep their sign.
  std::vector<std::int32_t> signed_levels = {-3, 0, 2};
  expect(decode_levels(signed_levels, true, false, &decoded, nullptr),
         "decode_levels signed succeeds");
  expect(decoded.quant_levels[2] == 4, "positive sign preserved");
  // c0 enters with state 0 (q0=0), so level 3 maps to 6 and keeps its sign.
  expect(decoded.quant_levels[0] == -6, "negative sign preserved");
}

void test_exhaustive_4x4() {
  using av2enc::tcq::Coefficient;
  using av2enc::tcq::Config;
  using av2enc::tcq::Result;
  const av2enc::tcq::RateModel model = test_model();
  for (int trial = 0; trial < 80; ++trial) {
    constexpr int n = 4;
    std::vector<Coefficient> coefficients(n);
    std::vector<double> original(n);
    for (int i = 0; i < n; ++i) {
      const std::size_t index = static_cast<std::size_t>(i);
      original[index] = uniform(-6.0, 6.0);
      coefficients[index].original = original[index];
    }
    Config config;
    config.lambda = (trial % 3 == 0) ? 0.0 : uniform(0.0, 3.0);
    config.quant_step = 2.0;
    config.level_cap = 3;
    config.max_coefficients = n;
    config.allow_zero_to_one = true;
    Result result;
    std::string error;
    if (!av2enc::tcq::quantize(coefficients, config, model, result, &error)) {
      expect(false, "quantize succeeds");
      std::fprintf(stderr, "  error: %s\n", error.c_str());
      continue;
    }
    PathCost best;
    std::vector<std::int32_t> magnitudes(n, 0);
    enumerate_paths(0, 3, &magnitudes, original, config, model, &best);
    expect_near(result.cost, best.cost, 1e-12, "exhaustive minimum cost");

    // The encoder's own base levels must decode to its quantised levels.
    av2enc::tcq::DecodeResult decoded;
    expect(av2enc::tcq::decode_levels(result.base_levels, true, false,
                                      &decoded, nullptr),
           "encoder output decodes");
    expect(decoded.quant_levels == result.quant_levels,
           "encoder quant_levels match normative mapping");
    expect(decoded.eob == result.eob, "encoder EOB matches normative mapping");
    expect_near(result.cost,
                result.distortion + config.lambda * result.rate_bits, 1e-12,
                "result cost = distortion + lambda*rate");
  }
}

void test_exhaustive_large_levels() {
  using av2enc::tcq::Coefficient;
  using av2enc::tcq::Config;
  using av2enc::tcq::Result;
  const av2enc::tcq::RateModel model = test_model();
  for (int trial = 0; trial < 40; ++trial) {
    constexpr int n = 3;
    std::vector<Coefficient> coefficients(n);
    std::vector<double> original(n);
    for (int i = 0; i < n; ++i) {
      const std::size_t index = static_cast<std::size_t>(i);
      original[index] = uniform(-35.0, 35.0);
      coefficients[index].original = original[index];
      coefficients[index].pre_max = 6;
    }
    Config config;
    config.lambda = (trial % 2 == 0) ? 0.0 : uniform(0.0, 2.0);
    config.quant_step = 2.0;
    config.level_cap = 9;
    config.max_coefficients = n;
    config.allow_zero_to_one = true;
    Result result;
    std::string error;
    if (!av2enc::tcq::quantize(coefficients, config, model, result, &error)) {
      expect(false, "large-level quantize succeeds");
      std::fprintf(stderr, "  error: %s\n", error.c_str());
      continue;
    }
    PathCost best;
    std::vector<std::int32_t> magnitudes(n, 0);
    enumerate_paths(0, 9, &magnitudes, original, config, model, &best);
    expect_near(result.cost, best.cost, 1e-12,
                "large-level exhaustive minimum cost");
    av2enc::tcq::DecodeResult decoded;
    expect(av2enc::tcq::decode_levels(result.base_levels, true, false,
                                      &decoded, nullptr),
           "large-level output decodes");
    expect(decoded.quant_levels == result.quant_levels,
           "large-level quant levels round-trip");
  }
}

void test_random_roundtrip() {
  using av2enc::tcq::Coefficient;
  using av2enc::tcq::Config;
  using av2enc::tcq::Result;
  const av2enc::tcq::RateModel model = av2enc::tcq::default_rate_model();
  for (int trial = 0; trial < 200; ++trial) {
    const int n = 1 + static_cast<int>(next_random() % 64u);
    std::vector<Coefficient> coefficients(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      coefficients[static_cast<std::size_t>(i)].original =
          uniform(-40.0, 40.0);
      coefficients[static_cast<std::size_t>(i)].level_context = i & 3;
    }
    Config config;
    config.lambda = uniform(0.0, 5.0);
    config.quant_step = uniform(0.5, 12.0);
    config.max_coefficients = n;
    config.allow_zero_to_one = (trial & 1) != 0;
    Result result;
    std::string error;
    if (!av2enc::tcq::quantize(coefficients, config, model, result, &error)) {
      expect(false, "random quantize succeeds");
      continue;
    }
    av2enc::tcq::DecodeResult decoded;
    expect(av2enc::tcq::decode_levels(result.base_levels, true, false,
                                      &decoded, nullptr),
           "random output decodes");
    expect(decoded.quant_levels == result.quant_levels,
           "random quant levels round-trip");
    expect(decoded.eob == result.eob, "random EOB round-trip");
  }
}

void test_header_flags() {
  using av2enc::tcq::HeaderFlags;
  using av2enc::tcq::HeaderInput;
  using av2enc::tcq::derive_header_flags;

  HeaderInput input;
  HeaderFlags flags;
  expect(derive_header_flags(input, &flags, nullptr), "flags derive");
  expect(!flags.enable_cctx && !flags.enable_tcq, "defaults off");
  expect(flags.sequence_bits == 3, "default sequence bits");
  expect(flags.frame_bits == 0, "default frame bits");

  input = HeaderInput{};
  input.single_picture_header_flag = true;
  input.enable_tcq = true;
  expect(derive_header_flags(input, &flags, nullptr), "single-pic derive");
  expect(flags.choose_tcq_per_frame == false,
         "single picture forces choose_tcq_per_frame=0");
  expect(flags.enable_parity_hiding == false,
         "TCQ without choose forces parity hiding off");
  expect(flags.allow_tcq == true, "allow_tcq inferred from enable_tcq");
  expect(flags.allow_parity_hiding == false,
         "allow_tcq forces allow_parity_hiding=0");
  expect(flags.sequence_bits == 2, "single-pic TCQ sequence bits");

  input = HeaderInput{};
  input.enable_tcq = true;
  input.choose_tcq_per_frame = true;
  input.enable_parity_hiding = true;
  input.frame_allow_tcq = true;
  expect(derive_header_flags(input, &flags, nullptr), "choose derive");
  expect(flags.choose_tcq_per_frame == true, "choose preserved");
  expect(flags.enable_parity_hiding == true, "sequence parity enabled");
  expect(flags.allow_tcq == true, "frame allow_tcq read");
  expect(flags.allow_parity_hiding == false,
         "frame allow_tcq forces allow_parity_hiding=0");
  expect(flags.sequence_bits == 4, "choose sequence bits");
  expect(flags.frame_bits == 1, "choose frame bits");

  input.frame_allow_tcq = false;
  input.frame_allow_parity_hiding = true;
  expect(derive_header_flags(input, &flags, nullptr), "parity derive");
  expect(flags.allow_tcq == false, "frame can disable TCQ");
  expect(flags.allow_parity_hiding == true, "parity can be allowed");
  expect(flags.frame_bits == 2, "both frame bits coded");

  input = HeaderInput{};
  input.coded_lossless = true;
  input.enable_tcq = true;
  input.enable_parity_hiding = true;
  expect(derive_header_flags(input, &flags, nullptr), "lossless derive");
  expect(flags.allow_tcq == false, "lossless forces allow_tcq=0");
  expect(flags.allow_parity_hiding == false,
         "lossless forces allow_parity_hiding=0");

  input = HeaderInput{};
  input.monochrome = true;
  input.enable_cctx = true;
  expect(derive_header_flags(input, &flags, nullptr), "mono derive");
  expect(flags.enable_cctx == false, "monochrome forces enable_cctx=0");
  expect(flags.sequence_bits == 2, "monochrome omits the cctx bit");
}

void test_cctx() {
  using av2enc::tcq::CctxConfig;
  using av2enc::tcq::CctxPair;
  using av2enc::tcq::CctxResult;
  using av2enc::tcq::CctxType;
  using av2enc::tcq::cctx_allowed;
  using av2enc::tcq::cctx_dequant_pair;
  using av2enc::tcq::cctx_forward_rotate;

  expect(cctx_allowed(true, false, true, 64, 64), "4:2:0 always allowed");
  expect(!cctx_allowed(false, false, true, 64, 64), "enable_cctx=0 blocks");
  expect(!cctx_allowed(true, true, true, 64, 64), "lossless blocks CCTX");
  expect(cctx_allowed(true, false, false, 16, 16), "small 4:4:4 allowed");
  expect(!cctx_allowed(true, false, false, 32, 32), "32x32 4:4:4 blocked");
  expect(cctx_allowed(true, false, false, 32, 16), "thin 4:4:4 allowed");

  // Cctx_Mtx type 1 rotates (256,0) to (181,181).
  const CctxPair p = cctx_dequant_pair(256, 0, CctxType::k45, 8);
  expect(p.u == 181 && p.v == 181, "CCTX_45 rotation");
  const CctxPair none = cctx_dequant_pair(123, -45, CctxType::kNone, 8);
  expect(none.u == 123 && none.v == -45, "CCTX_NONE is identity");
  const CctxPair minus = cctx_dequant_pair(256, 0, CctxType::kMinus45, 8);
  expect(minus.u == 181 && minus.v == -181, "CCTX_MINUS45 rotation");

  // Forward rotation followed by the decoder rotation is near identity.
  for (int t = 1; t < 7; ++t) {
    const CctxType type = static_cast<CctxType>(t);
    for (int k = 0; k < 16; ++k) {
      const double u = uniform(-400.0, 400.0);
      const double v = uniform(-400.0, 400.0);
      double ru = 0.0;
      double rv = 0.0;
      cctx_forward_rotate(u, v, type, &ru, &rv);
      const CctxPair back =
          cctx_dequant_pair(static_cast<std::int32_t>(std::llround(ru)),
                            static_cast<std::int32_t>(std::llround(rv)), type,
                            8);
      // The search quantises the rotated pair to integers, so one unit of
      // rounding can rotate back into up to ~sqrt(2) coefficient units.
      expect_near(static_cast<double>(back.u), u, 2.0,
                  "CCTX forward is inverse of decoder rotation (U)");
      expect_near(static_cast<double>(back.v), v, 2.0,
                  "CCTX forward is inverse of decoder rotation (V)");
    }
  }

  expect(av2enc::tcq::cctx_type_bits(CctxType::kNone) > 0.0,
         "CCTX type cost is finite");
  expect(av2enc::tcq::cctx_type_bits(CctxType::kNone) <
             av2enc::tcq::cctx_type_bits(CctxType::kMinus60),
         "CCTX type CDF is not uniform");

  // A flat block that is exactly representable: NONE should be optimal.
  constexpr int n = 16;
  std::vector<double> u(n, 10.0);
  std::vector<double> v(n, -10.0);
  CctxConfig config;
  config.enabled = true;
  config.quant_step_u = 1.0;
  config.quant_step_v = 1.0;
  config.bitdepth = 8;
  std::vector<CctxResult> candidates;
  std::string error;
  expect(av2enc::tcq::cctx_enumerate_scalar(u, v, 4, 4, config, &candidates,
                                            &error),
         "CCTX enumeration succeeds");
  expect(candidates.size() == 7, "CCTX enumerates all seven types");
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    expect(static_cast<int>(candidates[i].type) == static_cast<int>(i),
           "CCTX candidates are ordered by type");
  }
  config.enabled = false;
  expect(av2enc::tcq::cctx_enumerate_scalar(u, v, 4, 4, config, &candidates,
                                            &error),
         "CCTX disabled enumeration succeeds");
  expect(candidates.size() == 1 && candidates[0].type == CctxType::kNone,
         "CCTX disabled enumerates only NONE");
  config.enabled = true;
  CctxResult result;
  expect(av2enc::tcq::cctx_quantize_scalar(u, v, 4, 4, config, &result,
                                           &error),
         "CCTX search succeeds");
  expect(result.type == CctxType::kNone, "CCTX NONE wins an exact block");
  expect_near(result.distortion, 0.0, 1e-12, "CCTX exact block distortion");

  config.enabled = false;
  expect(av2enc::tcq::cctx_quantize_scalar(u, v, 4, 4, config, &result,
                                           &error),
         "CCTX disabled search succeeds");
  expect(result.type == CctxType::kNone, "CCTX disabled returns NONE");

  // A rotated source should never be worse than the NONE candidate.
  std::vector<double> heavy_u(n);
  std::vector<double> heavy_v(n);
  for (int i = 0; i < n; ++i) {
    heavy_u[static_cast<std::size_t>(i)] = 40.0 + 7.0 * i;
    heavy_v[static_cast<std::size_t>(i)] = -20.0 + 3.0 * i;
  }
  config = CctxConfig{};
  config.quant_step_u = 8.25;
  config.quant_step_v = 8.25;
  CctxResult chosen;
  expect(av2enc::tcq::cctx_quantize_scalar(heavy_u, heavy_v, 4, 4, config,
                                           &chosen, &error),
         "CCTX heavy search succeeds");
  CctxConfig none_config = config;
  none_config.enabled = false;
  CctxResult baseline;
  expect(av2enc::tcq::cctx_quantize_scalar(heavy_u, heavy_v, 4, 4, none_config,
                                           &baseline, &error),
         "CCTX baseline succeeds");
  expect(chosen.distortion <= baseline.distortion + 1e-9,
         "CCTX search cannot be worse than NONE");
}

void test_config_validation() {
  using av2enc::tcq::Config;
  Config config;
  expect(av2enc::tcq::config_valid(config), "default config valid");
  config.lambda = -1.0;
  expect(!av2enc::tcq::config_valid(config), "negative lambda rejected");
  config = Config{};
  config.quant_step = 0.0;
  expect(!av2enc::tcq::config_valid(config), "zero quant step rejected");
}

}  // namespace

int main() {
  test_state_table();
  test_dequant_shift();
  test_decoder_mapping();
  test_exhaustive_4x4();
  test_exhaustive_large_levels();
  test_random_roundtrip();
  test_header_flags();
  test_cctx();
  test_config_validation();

  std::printf("av2enc TCQ/CCTX test\n%d checks, %d failures\n", checks,
              failures);
  if (failures != 0) {
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
