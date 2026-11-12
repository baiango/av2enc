// Tests for the explicit CfL alpha path (AV2 v1.0.0 §5.20.7.32 / §7.13.5).
//
// Coverage:
//   * quantisation of every Q8 slope in [-1024, 1024] to 1..8 eighths;
//   * all 8 joint sign combinations x magnitudes x U/V pairs;
//   * the pinned §9.3 CDFs and §8.3 context tables;
//   * exact writer/reader round-trips, CDF lockstep, pinned byte goldens;
//   * a minimal CfL tile payload for explicit and derived alpha;
//   * exhaustive explicit search recovery for all sign/magnitude pairs;
//   * derived-vs-explicit selection by SSE and RD, deterministic tie-break;
//   * every per-plane alpha (17), clipping, frame-edge and partial blocks.
//
// Build: ./build_and_test.sh

#include "alpha.h"
#include "syntax.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#include "../../../../pipeline/still/still_cdf_tables.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using av2enc::cfl::Alpha;
using av2enc::cfl::AlphaPair;
using av2enc::cfl::Block;
using av2enc::cfl::Scratch;
using av2enc::cfl::Signalling;
using av2enc::cfl::SourcePlane;
using av2enc::cfl::derived_signalling;
using namespace av2enc::cfl::alpha;
namespace av2 = av2enc::av2;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                     \
  } while (false)

#define CHECK_EQ(actual, expected)                                          \
  do {                                                                      \
    ++g_checks;                                                             \
    const long long actual_value = static_cast<long long>(actual);          \
    const long long expected_value = static_cast<long long>(expected);      \
    if (actual_value != expected_value) {                                   \
      ++g_failures;                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n",        \
                   __FILE__, __LINE__, #actual, actual_value,               \
                   expected_value);                                         \
    }                                                                       \
  } while (false)

std::size_t ix(int row, int col, int stride) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(col);
}

// Independent §4.8 Round2Signed (the library's shared helper is under test).
int ref_round2_signed(int value) {
  constexpr int kShift = 11;
  if (value >= 0) {
    return (value + (1 << (kShift - 1))) >> kShift;
  }
  return -(((-value) + (1 << (kShift - 1))) >> kShift);
}

int ref_clip1(int value) { return std::clamp(value, 0, 255); }

// ---------------------------------------------------------------------------
// Fixture: one luma plane, two DC chroma planes and two source planes.
// ---------------------------------------------------------------------------

struct Fixture {
  static constexpr int kLumaStride = 192;
  static constexpr int kLumaSize = 192;
  static constexpr int kChromaStride = 128;
  static constexpr int kChromaSize = 128;

  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma_u;
  std::vector<std::uint8_t> chroma_v;
  std::vector<std::uint8_t> source_u;
  std::vector<std::uint8_t> source_v;
  Block u{};
  Block v{};

  Fixture(int coded_w, int coded_h, int w, int h, int x, int y)
      : luma(static_cast<std::size_t>(kLumaStride) * kLumaSize, 128),
        chroma_u(static_cast<std::size_t>(kChromaStride) * kChromaSize, 128),
        chroma_v(static_cast<std::size_t>(kChromaStride) * kChromaSize, 128),
        source_u(static_cast<std::size_t>(kChromaStride) * kChromaSize, 128),
        source_v(static_cast<std::size_t>(kChromaStride) * kChromaSize, 128) {
    u.luma = luma.data();
    u.luma_stride = kLumaStride;
    u.luma_max_x = kLumaSize - 1;
    u.luma_max_y = kLumaSize - 1;
    u.sb_top_y = ((y * 2) / 64) * 64;
    u.chroma = chroma_u.data();
    u.chroma_stride = kChromaStride;
    u.x = x;
    u.y = y;
    u.w = w;
    u.h = h;
    u.coded_width = coded_w;
    u.coded_height = coded_h;
    u.have_above = y >= 4 ? 1 : 0;
    u.have_left = x >= 4 ? 1 : 0;
    u.bit_depth = 8;
    u.ds_filter = 0;
    u.sub_x = 1;
    u.sub_y = 1;
    v = u;
    v.chroma = chroma_v.data();
  }

  std::uint8_t& luma_at(int row, int col) {
    return luma[ix(row, col, kLumaStride)];
  }
  std::uint8_t& u_at(int row, int col) {
    return chroma_u[ix(row, col, kChromaStride)];
  }
  std::uint8_t& v_at(int row, int col) {
    return chroma_v[ix(row, col, kChromaStride)];
  }
  std::uint8_t& src_u_at(int row, int col) {
    return source_u[ix(row, col, kChromaStride)];
  }
  std::uint8_t& src_v_at(int row, int col) {
    return source_v[ix(row, col, kChromaStride)];
  }
  SourcePlane source_plane_u() const {
    return SourcePlane{source_u.data(), kChromaStride, kChromaSize - 1,
                       kChromaSize - 1};
  }
  SourcePlane source_plane_v() const {
    return SourcePlane{source_v.data(), kChromaStride, kChromaSize - 1,
                       kChromaSize - 1};
  }
  SourcePlane plane_source_u() const { return source_plane_u(); }
  SourcePlane plane_source_v() const { return source_plane_v(); }
};

// A ramp with enough distinct L values that every signallable alpha produces a
// distinct prediction (used for exact search recovery).
void fill_luma_ramp(Fixture& f) {
  for (int row = 0; row < Fixture::kLumaSize; ++row) {
    for (int col = 0; col < Fixture::kLumaSize; ++col) {
      const int value = 40 + ((row * 7 + col * 11) % 161);
      f.luma_at(row, col) = static_cast<std::uint8_t>(value);
    }
  }
}

// Fill the source planes with the exact prediction for `pair` (no clipping in
// the caller's chosen range), leaving the DC planes at 128.
void fill_source_from_pair(Fixture& f, const Scratch& scratch,
                           AlphaPair pair) {
  for (int i = 0; i < f.u.h; ++i) {
    for (int j = 0; j < f.u.w; ++j) {
      const int luma_value = scratch.luma[i * f.u.w + j];
      const int du = ref_round2_signed(alpha_q8(pair.u) *
                                       (luma_value - scratch.luma_avg));
      const int dv = ref_round2_signed(alpha_q8(pair.v) *
                                       (luma_value - scratch.luma_avg));
      f.src_u_at(f.u.y + i, f.u.x + j) =
          static_cast<std::uint8_t>(std::clamp(128 + du, 4, 251));
      f.src_v_at(f.u.y + i, f.u.x + j) =
          static_cast<std::uint8_t>(std::clamp(128 + dv, 4, 251));
    }
  }
}

std::string hex_string(const std::uint8_t* data, std::size_t size) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(size * 2);
  for (std::size_t i = 0; i < size; ++i) {
    out.push_back(kHex[data[i] >> 4]);
    out.push_back(kHex[data[i] & 0xf]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// 1. Quantisation.
// ---------------------------------------------------------------------------

void test_quantisation() {
  for (int q8 = -1024; q8 <= 1024; ++q8) {
    const Alpha a = quantise_q8(q8);
    if (q8 == 0) {
      CHECK_EQ(a.sign, av2enc::cfl::kSignZero);
      CHECK_EQ(a.magnitude, 0);
      CHECK_EQ(alpha_q8(a), 0);
      continue;
    }
    const int magnitude = std::clamp((std::abs(q8) + 16) >> 5, 1,
                                     av2enc::cfl::kAlphabetSize);
    CHECK_EQ(a.sign, q8 < 0 ? av2enc::cfl::kSignNeg : av2enc::cfl::kSignPos);
    CHECK_EQ(a.magnitude, magnitude);
    CHECK_EQ(alpha_q8(a), (q8 < 0 ? -1 : 1) * magnitude * 32);
  }

  // Exact halfway ties round away from zero.
  CHECK_EQ(quantise_q8(16).magnitude, 1);
  CHECK_EQ(quantise_q8(16).sign, av2enc::cfl::kSignPos);
  CHECK_EQ(quantise_q8(-16).magnitude, 1);
  CHECK_EQ(quantise_q8(-16).sign, av2enc::cfl::kSignNeg);
  CHECK_EQ(quantise_q8(15).magnitude, 1);
  CHECK_EQ(quantise_q8(47).magnitude, 1);
  CHECK_EQ(quantise_q8(48).magnitude, 2);
  CHECK_EQ(quantise_q8(-48).magnitude, 2);
  CHECK_EQ(quantise_q8(255).magnitude, 8);
  CHECK_EQ(quantise_q8(256).magnitude, 8);
  CHECK_EQ(quantise_q8(100000).magnitude, 8);
  CHECK_EQ(quantise_q8(-100000).magnitude, 8);

  // Pair quantisation and the (zero,zero) fallback.
  for (int q = -80; q <= 80; ++q) {
    const AlphaPair p = quantise_pair(q, -q);
    CHECK(av2enc::cfl::signalable(p));
    if (q != 0) {
      CHECK_EQ(alpha_q8(p.u), alpha_q8(quantise_q8(q)));
      CHECK_EQ(alpha_q8(p.v), alpha_q8(quantise_q8(-q)));
    }
  }
  AlphaPair both_zero = quantise_pair(0, 0);
  CHECK_EQ(both_zero.u.sign, av2enc::cfl::kSignPos);
  CHECK_EQ(both_zero.u.magnitude, 1);
  CHECK_EQ(both_zero.v.sign, av2enc::cfl::kSignZero);
  CHECK_EQ(quantise_pair(0, -1).v.sign, av2enc::cfl::kSignNeg);
  CHECK_EQ(quantise_pair(-1, 0).u.sign, av2enc::cfl::kSignNeg);
}

// ---------------------------------------------------------------------------
// 2. Signalling over every joint sign and magnitude.
// ---------------------------------------------------------------------------

void test_signalling_all() {
  using namespace av2enc::cfl;
  // §6.19.7.25 Table 6.27, literal: CFL_SIGN_ZERO=0, NEG=1, POS=2.
  const int table_sign_u[8] = {0, 0, 1, 1, 1, 2, 2, 2};
  const int table_sign_v[8] = {1, 2, 0, 1, 2, 0, 1, 2};
  for (int signs = 0; signs < kJointSigns; ++signs) {
    CHECK_EQ(joint_sign_u(signs), table_sign_u[signs]);
    CHECK_EQ(joint_sign_v(signs), table_sign_v[signs]);
    CHECK_EQ(kSignZero, 0);
    CHECK_EQ(kSignNeg, 1);
    CHECK_EQ(kSignPos, 2);
  }
  for (int signs = 0; signs < kJointSigns; ++signs) {
    const int sign_u = joint_sign_u(signs);
    const int sign_v = joint_sign_v(signs);
    for (int mag_u = 1; mag_u <= kAlphabetSize; ++mag_u) {
      for (int mag_v = 1; mag_v <= kAlphabetSize; ++mag_v) {
        const AlphaPair pair = pair_from_joint(signs, mag_u, mag_v);
        CHECK_EQ(pair.u.sign, sign_u);
        CHECK_EQ(pair.v.sign, sign_v);
        CHECK(signalable(pair));
        CHECK_EQ(joint_signs(pair), signs);
        CHECK_EQ(per_plane_index(pair.u), per_plane_index(pair.u));
        const Signalling s = signalling_for(pair);
        CHECK_EQ(s.cfl_index, kIndexExplicit);
        CHECK_EQ(s.alpha_signs, signs);
        CHECK_EQ(s.alpha_u, sign_u == kSignZero ? 0 : mag_u - 1);
        CHECK_EQ(s.alpha_v, sign_v == kSignZero ? 0 : mag_v - 1);
        const std::optional<AlphaPair> back =
            pair_from_signalling_checked(s);
        CHECK(back.has_value());
        if (back.has_value()) {
          CHECK_EQ(back->u.sign, pair.u.sign);
          CHECK_EQ(back->u.magnitude, pair.u.magnitude);
          CHECK_EQ(back->v.sign, pair.v.sign);
          CHECK_EQ(back->v.magnitude, pair.v.magnitude);
        }
        // The 0..7 symbols are magnitude - 1.
        if (sign_u != kSignZero) {
          CHECK(s.alpha_u >= 0 && s.alpha_u <= 7);
        }
        if (sign_v != kSignZero) {
          CHECK(s.alpha_v >= 0 && s.alpha_v <= 7);
        }
      }
    }
  }

  // Every per-plane value appears exactly once and round-trips.
  for (int i = 0; i < kPerPlaneAlphas; ++i) {
    const Alpha a = per_plane_alpha(i);
    CHECK_EQ(per_plane_index(a), i);
    CHECK(av2enc::cfl::alpha_signalable(a));
  }
  CHECK_EQ(per_plane_index(Alpha{av2enc::cfl::kSignZero, 1}), -1);
  CHECK_EQ(per_plane_index(Alpha{av2enc::cfl::kSignPos, 9}), -1);
  CHECK_EQ(per_plane_index(Alpha{7, 1}), -1);

  // Invalid Signalling is rejected.
  Signalling bad{};
  bad.cfl_index = av2enc::cfl::kIndexExplicit;
  bad.alpha_signs = 0;  // signU zero, V positive
  bad.alpha_u = 3;      // non-canonical
  bad.alpha_v = 0;
  CHECK(!pair_from_signalling_checked(bad).has_value());
  bad = Signalling{};
  bad.cfl_index = av2enc::cfl::kIndexMulti;
  CHECK(!pair_from_signalling_checked(bad).has_value());
  bad = Signalling{};
  bad.cfl_index = av2enc::cfl::kIndexExplicit;
  bad.alpha_signs = 8;
  CHECK(!pair_from_signalling_checked(bad).has_value());
  CHECK_EQ(signalling_for(AlphaPair{Alpha{0, 0}, Alpha{0, 0}}).cfl_index, -1);

  // §6.19.7.25 Table 6.27.2: (zero, zero) has no cfl_alpha_signs.
  CHECK(!av2enc::cfl::signalable(
      AlphaPair{Alpha{av2enc::cfl::kSignZero, 0},
                Alpha{av2enc::cfl::kSignZero, 0}}));
}

// ---------------------------------------------------------------------------
// 2b. Quantisation of the derived slope.
// ---------------------------------------------------------------------------

void test_derived_quantisation() {
  using namespace av2enc::cfl;
  const auto rounded_chroma = [](int luma_value, int magnitude) {
    const int scaled = magnitude * (luma_value - 128);
    const int rounded =
        scaled >= 0 ? (scaled + 4) >> 3 : -((-scaled + 4) >> 3);
    return std::clamp(128 + rounded, 0, 255);
  };
  for (int magnitude = 1; magnitude <= kAlphabetSize; ++magnitude) {
    Fixture f(64, 64, 8, 8, 8, 8);
    for (int row = 0; row < 64; ++row) {
      for (int col = 0; col < 64; ++col) {
        f.luma_at(row, col) =
            static_cast<std::uint8_t>(40 + (row * 5 + col * 7) % 180);
      }
    }
    Scratch scratch;
    compute_luma(f.u, scratch);
    for (int i = 0; i < 8; ++i) {
      const int u_value = rounded_chroma(scratch.luma_above[i], magnitude);
      const int v_value = rounded_chroma(scratch.luma_above[i], magnitude);
      f.u_at(f.u.y - 1, f.u.x + i) = static_cast<std::uint8_t>(u_value);
      f.v_at(f.v.y - 1, f.v.x + i) = static_cast<std::uint8_t>(v_value);
    }
    for (int j = 0; j < 8; ++j) {
      const int u_value = rounded_chroma(scratch.luma_left[j], magnitude);
      const int v_value = rounded_chroma(scratch.luma_left[j], magnitude);
      f.u_at(f.u.y + j, f.u.x - 1) = static_cast<std::uint8_t>(u_value);
      f.v_at(f.v.y + j, f.v.x - 1) = static_cast<std::uint8_t>(v_value);
    }
    const AlphaPair pair = quantise_derived_pair(f.u, f.v, scratch);
    CHECK(av2enc::cfl::signalable(pair));
    CHECK_EQ(pair.u.sign, kSignPos);
    CHECK_EQ(pair.v.sign, kSignPos);
    CHECK_EQ(pair.u.magnitude, magnitude);
    CHECK_EQ(pair.v.magnitude, magnitude);
  }

  // No neighbours: both derived slopes are 0, so the pair is repaired to a
  // signalable one and never (zero,zero).
  Fixture none(64, 64, 8, 8, 0, 0);
  Scratch scratch;
  compute_luma(none.u, scratch);
  const AlphaPair pair = quantise_derived_pair(none.u, none.v, scratch);
  CHECK(av2enc::cfl::signalable(pair));
  CHECK_EQ(pair.u.sign, kSignPos);
  CHECK_EQ(pair.u.magnitude, 1);
  CHECK_EQ(pair.v.sign, kSignZero);
}

// ---------------------------------------------------------------------------
// 3. CDFs and §8.3 contexts.
// ---------------------------------------------------------------------------

void test_cdfs_pinned() {
  const CflCdfs cdfs = default_cfl_cdfs();
  CHECK(cdfs_valid(cdfs));

  const std::uint16_t is_cfl[3][3] = {
      {20441, 30, 0}, {11610, 6, 0}, {4643, 0, 0}};
  const std::uint16_t index[3] = {12507, 30, 0};
  const std::uint16_t sign[9] = {2421, 4332, 11256, 12766,
                                 21386, 28725, 32087, 62, 0};
  const std::uint16_t alpha[6][9] = {
      {21679, 25305, 30646, 31512, 32537, 32646, 32696, 62, 0},
      {8262, 16302, 24082, 29422, 31398, 32286, 32525, 62, 0},
      {17235, 26166, 30378, 31305, 32373, 32549, 32668, 62, 0},
      {17618, 25732, 27865, 30338, 31125, 31522, 32238, 62, 0},
      {17542, 23066, 27907, 28728, 30702, 31165, 31435, 62, 0},
      {17675, 24802, 30468, 30783, 31841, 32264, 32422, 62, 0},
  };
  for (int ctx = 0; ctx < 3; ++ctx) {
    for (int i = 0; i < 3; ++i) {
      CHECK_EQ(cdfs.is_cfl[ctx][i], is_cfl[ctx][i]);
    }
  }
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ(cdfs.cfl_index[i], index[i]);
  }
  for (int i = 0; i < 9; ++i) {
    CHECK_EQ(cdfs.cfl_sign[i], sign[i]);
  }
  for (int ctx = 0; ctx < 6; ++ctx) {
    for (int i = 0; i < 9; ++i) {
      CHECK_EQ(cdfs.cfl_alpha[ctx][i], alpha[ctx][i]);
    }
  }

  // The repo's generated M1 tables are the same normative arrays. This catches
  // a transcription drift on either side without editing the M1 component.
  CHECK(std::memcmp(cdfs.is_cfl, av2enc::still::kIsCflCdf,
                    sizeof(cdfs.is_cfl)) == 0);
  CHECK(std::memcmp(cdfs.cfl_index, av2enc::still::kCflIndexCdf,
                    sizeof(cdfs.cfl_index)) == 0);
  CHECK(std::memcmp(cdfs.cfl_sign, av2enc::still::kCflSignCdf,
                    sizeof(cdfs.cfl_sign)) == 0);
  CHECK(std::memcmp(cdfs.cfl_alpha, av2enc::still::kCflAlphaCdf,
                    sizeof(cdfs.cfl_alpha)) == 0);

  // §8.3 Table 8.4 (cfl_alpha_u context) and Table 8.5 (cfl_alpha_v context).
  const int table_u[8] = {-1, -1, 0, 1, 2, 3, 4, 5};
  const int table_v[8] = {0, 3, -1, 1, 4, -1, 2, 5};
  for (int signs = 0; signs < 8; ++signs) {
    CHECK_EQ(alpha_u_context(signs), table_u[signs]);
    CHECK_EQ(alpha_v_context(signs), table_v[signs]);
  }
  CHECK_EQ(alpha_u_context(-1), -1);
  CHECK_EQ(alpha_u_context(8), -1);
  CHECK_EQ(alpha_v_context(-1), -1);
  CHECK_EQ(alpha_v_context(8), -1);

  // is_cfl context is the count of CfL-flagged available neighbours.
  CHECK_EQ(is_cfl_context(false, false), 0);
  CHECK_EQ(is_cfl_context(true, false), 1);
  CHECK_EQ(is_cfl_context(false, true), 1);
  CHECK_EQ(is_cfl_context(true, true), 2);

  // Corruption is detected.
  CflCdfs bad = cdfs;
  bad.cfl_sign[3] = 40000;
  CHECK(!cdfs_valid(bad));
  bad = cdfs;
  bad.cfl_alpha[2][8] = 33;
  CHECK(!cdfs_valid(bad));
}

// ---------------------------------------------------------------------------
// 4. Model rate.
// ---------------------------------------------------------------------------

void test_model_rate() {
  // Uniform 8-symbol CDF: cdf[s] = (s + 1) * 32768 / 8, so every symbol is
  // ~3 bits.
  std::uint16_t uniform[9] = {4096, 8192, 12288, 16384, 20480, 24576,
                              28672, 0, 0};
  for (int symbol = 0; symbol < 8; ++symbol) {
    const int rate = symbol_rate_q8(uniform, 8, symbol);
    CHECK(rate >= 700 && rate <= 860);  // ~3 bits in 1/256 units
  }
  std::uint16_t uniform2[3] = {16384, 0, 0};
  for (int symbol = 0; symbol < 2; ++symbol) {
    const int rate = symbol_rate_q8(uniform2, 2, symbol);
    CHECK(rate >= 240 && rate <= 270);  // ~1 bit
  }
  CHECK_EQ(symbol_rate_q8(nullptr, 2, 0), 1 << 20);
  CHECK_EQ(symbol_rate_q8(uniform, 8, 8), 1 << 20);

  const CflCdfs cdfs = default_cfl_cdfs();
  const Signalling derived = derived_signalling();
  const int derived_rate = alpha_syntax_rate_q8(derived, cdfs);
  CHECK(derived_rate > 0);

  // Explicit is always at least as expensive as derived in the default model.
  for (int signs = 0; signs < 8; ++signs) {
    for (int mag_u = 1; mag_u <= 8; ++mag_u) {
      for (int mag_v = 1; mag_v <= 8; ++mag_v) {
        const Signalling s = signalling_for(pair_from_joint(
            signs, mag_u, mag_v));
        const int rate = alpha_syntax_rate_q8(s, cdfs);
        CHECK(rate > derived_rate);
        if (joint_sign_u(signs) == 0 || joint_sign_v(signs) == 0) {
          // One alpha symbol, not two.
          const Signalling s2 = signalling_for(pair_from_joint(
              signs, mag_u, mag_v));
          CHECK_EQ(alpha_syntax_rate_q8(s2, cdfs), rate);
        }
      }
    }
  }
  Signalling multi{};
  multi.cfl_index = av2enc::cfl::kIndexMulti;
  CHECK_EQ(alpha_syntax_rate_q8(multi, cdfs), 1 << 20);
}

// ---------------------------------------------------------------------------
// 5/6. Exact writers, readers, minimal payloads.
// ---------------------------------------------------------------------------

const char* kGoldenExplicitHex = "732c4000";  // is_cfl=1, explicit (neg,neg) 8/3
const char* kGoldenDerivedHex = "a00000";     // is_cfl=1, CFL_DERIVED_ALPHA
const char* kGoldenSequenceHex =
    "06e7550bf9bc5652780402b66f0000";  // signs 0..7 then derived

struct Payload {
  std::array<std::uint8_t, 256> bytes{};
  std::size_t size = 0;
  CflCdfs cdfs{};
};

Payload encode_sequence(int context, bool is_cfl, const Signalling& s) {
  Payload payload{};
  payload.cdfs = default_cfl_cdfs();
  av2::Encoder enc;
  CHECK(enc.init_symbol(payload.bytes.data(), payload.bytes.size()) ==
        av2::Status::kOk);
  CHECK(write_cfl_mode_and_alphas(enc, payload.cdfs, context, is_cfl, s) ==
        av2::Status::kOk);
  CHECK(enc.exit_symbol(&payload.size) == av2::Status::kOk);
  return payload;
}

void test_syntax_roundtrip_all() {
  using namespace av2enc::cfl;
  for (int context = 0; context < 3; ++context) {
    for (int is_cfl = 0; is_cfl <= 1; ++is_cfl) {
      if (!is_cfl) {
        const Payload p = encode_sequence(context, false, Signalling{});
        av2::Decoder dec;
        CflCdfs cdfs = default_cfl_cdfs();
        CHECK(dec.init_symbol(p.bytes.data(), p.bytes.size(), p.size) ==
              av2::Status::kOk);
        bool decoded_is_cfl = true;
        Signalling decoded{};
        CHECK(read_cfl_mode_and_alphas(dec, cdfs, context, &decoded_is_cfl,
                                       &decoded) == av2::Status::kOk);
        CHECK(!decoded_is_cfl);
        CHECK_EQ(decoded.cfl_index, -1);
        CHECK(dec.exit_symbol() == av2::Status::kOk);
        continue;
      }
      for (int signs = 0; signs < kJointSigns; ++signs) {
        for (int mag_u = 1; mag_u <= kAlphabetSize; ++mag_u) {
          for (int mag_v = 1; mag_v <= kAlphabetSize; ++mag_v) {
            const AlphaPair pair = pair_from_joint(signs, mag_u, mag_v);
            const Signalling s = signalling_for(pair);
            const Payload p = encode_sequence(context, true, s);

            av2::Decoder dec;
            CflCdfs cdfs = default_cfl_cdfs();
            CHECK(dec.init_symbol(p.bytes.data(), p.bytes.size(), p.size) ==
                  av2::Status::kOk);
            bool decoded_is_cfl = false;
            Signalling decoded{};
            CHECK(read_cfl_mode_and_alphas(dec, cdfs, context,
                                           &decoded_is_cfl, &decoded) ==
                  av2::Status::kOk);
            CHECK(decoded_is_cfl);
            CHECK_EQ(decoded.cfl_index, kIndexExplicit);
            CHECK_EQ(decoded.alpha_signs, s.alpha_signs);
            CHECK_EQ(decoded.alpha_u, s.alpha_u);
            CHECK_EQ(decoded.alpha_v, s.alpha_v);
            CHECK(dec.exit_symbol() == av2::Status::kOk);
            // CDF state stays in lockstep.
            CHECK(std::memcmp(&p.cdfs, &cdfs, sizeof(CflCdfs)) == 0);
          }
        }
      }
      // Derived mode.
      const Payload p =
          encode_sequence(context, true, derived_signalling());
      av2::Decoder dec;
      CflCdfs cdfs = default_cfl_cdfs();
      CHECK(dec.init_symbol(p.bytes.data(), p.bytes.size(), p.size) ==
            av2::Status::kOk);
      bool decoded_is_cfl = false;
      Signalling decoded{};
      CHECK(read_cfl_mode_and_alphas(dec, cdfs, context, &decoded_is_cfl,
                                     &decoded) == av2::Status::kOk);
      CHECK(decoded_is_cfl);
      CHECK_EQ(decoded.cfl_index, kIndexDerivedAlpha);
      CHECK(dec.exit_symbol() == av2::Status::kOk);
      CHECK(std::memcmp(&p.cdfs, &cdfs, sizeof(CflCdfs)) == 0);
    }
  }

  // CFL_MULTI and non-canonical explicit are rejected by the writer.
  Payload p{};
  p.cdfs = default_cfl_cdfs();
  av2::Encoder enc;
  CHECK(enc.init_symbol(p.bytes.data(), p.bytes.size()) ==
        av2::Status::kOk);
  Signalling multi{};
  multi.cfl_index = av2enc::cfl::kIndexMulti;
  CHECK(write_cfl_alphas(enc, p.cdfs, multi) ==
        av2::Status::kInvalidSymbol);
  CHECK_EQ(enc.symbol_count, 0);
  // Non-canonical explicit is rejected before any symbol is written.
  Signalling noncanonical{};
  noncanonical.cfl_index = av2enc::cfl::kIndexExplicit;
  noncanonical.alpha_signs = 0;
  noncanonical.alpha_u = 4;
  noncanonical.alpha_v = 0;
  CHECK(write_cfl_alphas(enc, p.cdfs, noncanonical) ==
        av2::Status::kInvalidSymbol);
  CHECK_EQ(enc.symbol_count, 0);
  // The coder is still usable and produces the canonical derived stream.
  CHECK(write_cfl_alphas(enc, p.cdfs, derived_signalling()) ==
        av2::Status::kOk);
  CHECK(enc.exit_symbol(&p.size) == av2::Status::kOk);
  CHECK_EQ(p.cdfs.cfl_index[2], 1);  // one adaptation, not two
}

void print_golden(const char* name, const Payload& p) {
  std::printf("GOLDEN %s %s\n", name,
              hex_string(p.bytes.data(), p.size).c_str());
}

void test_pinned_payloads() {
  const Signalling explicit_pair =
      signalling_for(pair_from_joint(3, 8, 3));  // signs 3 = (neg,neg)
  const Payload explicit_payload = encode_sequence(1, true, explicit_pair);
  const Payload derived_payload =
      encode_sequence(1, true, derived_signalling());

  // A single payload coding every sign symbol and then derived mode, so the
  // golden also pins CDF adaptation across symbols.
  Payload sequence{};
  sequence.cdfs = default_cfl_cdfs();
  av2::Encoder sequence_enc;
  CHECK(sequence_enc.init_symbol(sequence.bytes.data(), sequence.bytes.size()) ==
        av2::Status::kOk);
  for (int signs = 0; signs < av2enc::cfl::kJointSigns; ++signs) {
    const int mag_u = 1 + signs;
    const int mag_v = 8 - signs;
    CHECK(write_cfl_alphas(sequence_enc, sequence.cdfs,
                           signalling_for(pair_from_joint(signs, mag_u,
                                                          mag_v))) ==
          av2::Status::kOk);
  }
  CHECK(write_cfl_alphas(sequence_enc, sequence.cdfs, derived_signalling()) ==
        av2::Status::kOk);
  CHECK(sequence_enc.exit_symbol(&sequence.size) == av2::Status::kOk);

  if (std::getenv("ALPHA_PRINT_GOLDENS") != nullptr) {
    print_golden("explicit", explicit_payload);
    print_golden("derived", derived_payload);
    print_golden("sequence", sequence);
  }

  av2::Decoder sequence_dec;
  CflCdfs sequence_cdfs = default_cfl_cdfs();
  CHECK(sequence_dec.init_symbol(sequence.bytes.data(),
                                 sequence.bytes.size(), sequence.size) ==
        av2::Status::kOk);
  for (int signs = 0; signs < av2enc::cfl::kJointSigns; ++signs) {
    Signalling decoded{};
    CHECK(read_cfl_alphas(sequence_dec, sequence_cdfs, &decoded) ==
          av2::Status::kOk);
    CHECK_EQ(decoded.cfl_index, av2enc::cfl::kIndexExplicit);
    CHECK_EQ(decoded.alpha_signs, signs);
  }
  Signalling sequence_derived{};
  CHECK(read_cfl_alphas(sequence_dec, sequence_cdfs, &sequence_derived) ==
        av2::Status::kOk);
  CHECK_EQ(sequence_derived.cfl_index, av2enc::cfl::kIndexDerivedAlpha);
  CHECK(sequence_dec.exit_symbol() == av2::Status::kOk);
  CHECK(std::memcmp(&sequence.cdfs, &sequence_cdfs, sizeof(CflCdfs)) == 0);
  if (kGoldenSequenceHex[0] != '\0') {
    CHECK(hex_string(sequence.bytes.data(), sequence.size) ==
          kGoldenSequenceHex);
  }

  // The minimal payload for a CfL block: is_cfl=1, cfl_index, and (explicit)
  // cfl_alpha_signs + cfl_alpha_u/v. Decoding it reproduces the same fields.
  CHECK(explicit_payload.size > 1);
  CHECK(derived_payload.size > 0);
  av2::Decoder dec;
  CflCdfs cdfs = default_cfl_cdfs();
  CHECK(dec.init_symbol(explicit_payload.bytes.data(),
                        explicit_payload.bytes.size(),
                        explicit_payload.size) == av2::Status::kOk);
  bool is_cfl = false;
  Signalling decoded{};
  CHECK(read_cfl_mode_and_alphas(dec, cdfs, 1, &is_cfl, &decoded) ==
        av2::Status::kOk);
  CHECK(is_cfl);
  CHECK_EQ(decoded.cfl_index, av2enc::cfl::kIndexExplicit);
  CHECK_EQ(decoded.alpha_signs, 3);
  CHECK_EQ(decoded.alpha_u, 7);
  CHECK_EQ(decoded.alpha_v, 2);
  CHECK(dec.exit_symbol() == av2::Status::kOk);

  const std::string explicit_hex =
      hex_string(explicit_payload.bytes.data(), explicit_payload.size);
  const std::string derived_hex =
      hex_string(derived_payload.bytes.data(), derived_payload.size);
  if (kGoldenExplicitHex[0] != '\0') {
    CHECK(explicit_hex == kGoldenExplicitHex);
    CHECK(derived_hex == kGoldenDerivedHex);
  }

  // Exact call-site parity with the current M1 derived writer: the symbol
  // sequence in still_syntax.cpp::code_uv_mode() is is_cfl=1 then cfl_index=1.
  Payload m1{};
  m1.cdfs = default_cfl_cdfs();
  av2::Encoder still_enc;
  CHECK(still_enc.init_symbol(m1.bytes.data(), m1.bytes.size()) ==
        av2::Status::kOk);
  CHECK(still_enc.encode_symbol(m1.cdfs.is_cfl[1], 2, 1) == av2::Status::kOk);
  CHECK(still_enc.encode_symbol(m1.cdfs.cfl_index, 2, 1) == av2::Status::kOk);
  CHECK(still_enc.exit_symbol(&m1.size) == av2::Status::kOk);
  CHECK_EQ(m1.size, derived_payload.size);
  CHECK(std::memcmp(m1.bytes.data(), derived_payload.bytes.data(), m1.size) ==
        0);
  CHECK(std::memcmp(&m1.cdfs, &derived_payload.cdfs, sizeof(CflCdfs)) == 0);

  // Truncation: a declared size larger than the backing buffer is rejected at
  // init; an all-zero one-byte stream cannot satisfy the §8.2.4 trailing bit.
  av2::Decoder bad_dec;
  CflCdfs bad_cdfs = default_cfl_cdfs();
  CHECK(bad_dec.init_symbol(derived_payload.bytes.data(),
                            derived_payload.size,
                            derived_payload.size + 1) ==
        av2::Status::kTruncatedInput);
  const std::uint8_t zero_byte = 0;
  CHECK(bad_dec.init_symbol(&zero_byte, 1, 1) == av2::Status::kOk);
  bool zero_is_cfl = false;
  Signalling zero_alpha{};
  const av2::Status zero_status = read_cfl_mode_and_alphas(
      bad_dec, bad_cdfs, 1, &zero_is_cfl, &zero_alpha);
  CHECK(zero_status != av2::Status::kOk ||
        bad_dec.exit_symbol() != av2::Status::kOk);
}

// ---------------------------------------------------------------------------
// 7. Exhaustive explicit search recovery over all sign/magnitude pairs.
// ---------------------------------------------------------------------------

void test_search_recovery_all() {
  using namespace av2enc::cfl;
  const CflCdfs cdfs = default_cfl_cdfs();
  Config config{};
  config.cost.use_rate = false;

  int cases = 0;
  for (int signs = 0; signs < kJointSigns; ++signs) {
    for (int mag_u = 1; mag_u <= kAlphabetSize; ++mag_u) {
      for (int mag_v = 1; mag_v <= kAlphabetSize; ++mag_v) {
        const AlphaPair target = pair_from_joint(signs, mag_u, mag_v);
        Fixture f(64, 64, 8, 8, 16, 16);
        fill_luma_ramp(f);
        Scratch scratch;
        compute_luma(f.u, scratch);
        fill_source_from_pair(f, scratch, target);

        const auto found =
            search_explicit(f.u, f.v, f.plane_source_u(), f.plane_source_v(),
                            config, cdfs, scratch);
        CHECK(found.has_value());
        if (!found.has_value()) {
          continue;
        }
        CHECK_EQ(found->total_sse(), 0);
        CHECK(found->explicit_alpha);
        CHECK_EQ(found->syntax.cfl_index, kIndexExplicit);
        CHECK_EQ(found->syntax.alpha_signs, signs);
        CHECK_EQ(found->q8_u, alpha_q8(target.u));
        CHECK_EQ(found->q8_v, alpha_q8(target.v));
        // The chosen syntax round-trips through the exact reader.
        const Signalling s = found->syntax;
        const auto pair = pair_from_signalling_checked(s);
        CHECK(pair.has_value());
        if (pair.has_value()) {
          CHECK_EQ(alpha_q8(pair->u), alpha_q8(target.u));
          CHECK_EQ(alpha_q8(pair->v), alpha_q8(target.v));
        }
        ++cases;
      }
    }
  }
  CHECK_EQ(cases, 512);

  // apply_candidate reproduces the target source exactly.
  Fixture f(64, 64, 8, 8, 16, 16);
  fill_luma_ramp(f);
  Scratch scratch;
  compute_luma(f.u, scratch);
  const AlphaPair target = pair_from_joint(7, 4, 6);
  fill_source_from_pair(f, scratch, target);
  const auto found = search_explicit(f.u, f.v, f.plane_source_u(),
                                     f.plane_source_v(), config, cdfs, scratch);
  CHECK(found.has_value());
  if (found.has_value()) {
    const std::vector<std::uint8_t> before_u = f.chroma_u;
    const std::vector<std::uint8_t> before_v = f.chroma_v;
    // Recompute luma because search_explicit filled `scratch` already.
    apply_candidate(f.u, f.v, *found, scratch);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        CHECK_EQ(f.u_at(16 + i, 16 + j), f.src_u_at(16 + i, 16 + j));
        CHECK_EQ(f.v_at(16 + i, 16 + j), f.src_v_at(16 + i, 16 + j));
      }
    }
    CHECK(before_u != f.chroma_u);
    CHECK(before_v != f.chroma_v);
  }
}

// ---------------------------------------------------------------------------
// 8. Derived-vs-explicit selection and tie-break.
// ---------------------------------------------------------------------------

void test_selection() {
  using namespace av2enc::cfl;
  const CflCdfs cdfs = default_cfl_cdfs();

  // Explicit is clearly better: source built from a known explicit pair on a
  // ramp, while the derived neighbour regression sees a flat relation.
  Fixture f(64, 64, 8, 8, 16, 16);
  fill_luma_ramp(f);
  for (int i = 0; i < 64; ++i) {
    f.u_at(15, i) = 128;
    f.u_at(i, 15) = 128;
    f.v_at(15, i) = 128;
    f.v_at(i, 15) = 128;
  }
  Scratch scratch;
  compute_luma(f.u, scratch);
  const AlphaPair target = pair_from_joint(3, 5, 2);  // (neg,neg), magnitudes 5/2
  fill_source_from_pair(f, scratch, target);

  Config sse_config{};
  sse_config.cost.use_rate = false;
  const auto sse_decision = decide_cfl_alpha(
      f.u, f.v, f.plane_source_u(), f.plane_source_v(), sse_config, cdfs,
      scratch);
  CHECK(sse_decision.has_value());
  if (sse_decision.has_value()) {
    CHECK(sse_decision->has_derived);
    CHECK(sse_decision->has_explicit);
    CHECK(sse_decision->chosen.explicit_alpha);
    CHECK_EQ(sse_decision->chosen.syntax.cfl_index, kIndexExplicit);
    CHECK_EQ(sse_decision->chosen.total_sse(), 0);
    CHECK(sse_decision->chosen.total_sse() <
          sse_decision->derived.total_sse());
  }

  // A large lambda flips the decision to derived: derived's rate is much
  // lower and the explicit SSE advantage is small once the source is noisy.
  Fixture noisy(64, 64, 8, 8, 16, 16);
  fill_luma_ramp(noisy);
  Scratch noisy_scratch;
  compute_luma(noisy.u, noisy_scratch);
  fill_source_from_pair(noisy, noisy_scratch, target);
  std::mt19937 rng(0x51u);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const int du = static_cast<int>(rng() % 9) - 4;
      const int dv = static_cast<int>(rng() % 9) - 4;
      noisy.src_u_at(16 + i, 16 + j) = static_cast<std::uint8_t>(
          std::clamp(noisy.u_at(16 + i, 16 + j) + du, 0, 255));
      noisy.src_v_at(16 + i, 16 + j) = static_cast<std::uint8_t>(
          std::clamp(noisy.u_at(16 + i, 16 + j) + dv, 0, 255));
    }
  }
  Config rd_config{};
  rd_config.cost.use_rate = true;
  rd_config.cost.lambda_q8 = 1000000;  // very large lambda
  const auto rd_decision = decide_cfl_alpha(
      noisy.u, noisy.v, noisy.plane_source_u(), noisy.plane_source_v(),
      rd_config, cdfs, noisy_scratch);
  CHECK(rd_decision.has_value());
  if (rd_decision.has_value()) {
    CHECK(!rd_decision->chosen.explicit_alpha);
    CHECK_EQ(rd_decision->chosen.syntax.cfl_index, kIndexDerivedAlpha);
  }

  // Exact tie: constant luma means every alpha predicts the same DC. Derived
  // (0) and explicit both have SSE 0; the tie-break decides.
  Fixture tie(64, 64, 8, 8, 16, 16);
  for (auto& value : tie.luma) {
    value = 100;
  }
  for (auto& value : tie.chroma_u) {
    value = 128;
  }
  for (auto& value : tie.chroma_v) {
    value = 128;
  }
  for (auto& value : tie.source_u) {
    value = 128;
  }
  for (auto& value : tie.source_v) {
    value = 128;
  }
  Scratch tie_scratch;
  Config prefer_derived{};
  prefer_derived.cost.use_rate = false;
  prefer_derived.tie_break = TieBreak::kPreferDerived;
  const auto tie_derived = decide_cfl_alpha(
      tie.u, tie.v, tie.plane_source_u(), tie.plane_source_v(),
      prefer_derived, cdfs, tie_scratch);
  CHECK(tie_derived.has_value());
  if (tie_derived.has_value()) {
    CHECK_EQ(tie_derived->chosen.total_sse(), 0);
    CHECK_EQ(tie_derived->chosen.syntax.cfl_index, kIndexDerivedAlpha);
  }
  Config prefer_explicit = prefer_derived;
  prefer_explicit.tie_break = TieBreak::kPreferExplicit;
  const auto tie_explicit = decide_cfl_alpha(
      tie.u, tie.v, tie.plane_source_u(), tie.plane_source_v(),
      prefer_explicit, cdfs, tie_scratch);
  CHECK(tie_explicit.has_value());
  if (tie_explicit.has_value()) {
    CHECK_EQ(tie_explicit->chosen.total_sse(), 0);
    CHECK_EQ(tie_explicit->chosen.syntax.cfl_index, kIndexExplicit);
    CHECK_EQ(tie_explicit->chosen.syntax.alpha_signs, 0);
    CHECK_EQ(tie_explicit->chosen.syntax.alpha_v, 0);
  }

  // Determinism: repeated calls return the identical decision.
  for (int iteration = 0; iteration < 5; ++iteration) {
    const auto again = decide_cfl_alpha(
        f.u, f.v, f.plane_source_u(), f.plane_source_v(), sse_config, cdfs,
        scratch);
    CHECK(again.has_value());
    if (again.has_value() && sse_decision.has_value()) {
      CHECK_EQ(again->chosen.syntax.cfl_index,
               sse_decision->chosen.syntax.cfl_index);
      CHECK_EQ(again->chosen.syntax.alpha_signs,
               sse_decision->chosen.syntax.alpha_signs);
      CHECK_EQ(again->chosen.syntax.alpha_u,
               sse_decision->chosen.syntax.alpha_u);
      CHECK_EQ(again->chosen.syntax.alpha_v,
               sse_decision->chosen.syntax.alpha_v);
    }
  }

  // Explicit-only and derived-only configurations.
  Config explicit_only{};
  explicit_only.allow_derived = false;
  const auto only_explicit = decide_cfl_alpha(
      f.u, f.v, f.plane_source_u(), f.plane_source_v(), explicit_only, cdfs,
      scratch);
  CHECK(only_explicit.has_value());
  if (only_explicit.has_value()) {
    CHECK(!only_explicit->has_derived);
    CHECK(only_explicit->chosen.explicit_alpha);
  }
  Config derived_only{};
  derived_only.allow_explicit = false;
  const auto only_derived = decide_cfl_alpha(
      f.u, f.v, f.plane_source_u(), f.plane_source_v(), derived_only, cdfs,
      scratch);
  CHECK(only_derived.has_value());
  if (only_derived.has_value()) {
    CHECK(!only_derived->has_explicit);
    CHECK_EQ(only_derived->chosen.syntax.cfl_index, kIndexDerivedAlpha);
  }
}

// ---------------------------------------------------------------------------
// 9. Every per-plane alpha, clipping, edges, partial blocks.
// ---------------------------------------------------------------------------

void test_all_alphas_prediction() {
  using namespace av2enc::cfl;
  for (int index = 0; index < kPerPlaneAlphas; ++index) {
    const Alpha a = per_plane_alpha(index);
    Fixture f(64, 64, 8, 8, 8, 8);
    fill_luma_ramp(f);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        f.u_at(8 + i, 8 + j) = 70;
        f.v_at(8 + i, 8 + j) = 190;
      }
    }
    Scratch scratch;
    compute_luma(f.u, scratch);
    apply_prediction(f.u, alpha_q8(a), scratch);
    apply_prediction(f.v, alpha_q8(a), scratch);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const int luma_value = scratch.luma[i * 8 + j];
        const int delta = ref_round2_signed(
            alpha_q8(a) * (luma_value - scratch.luma_avg));
        CHECK_EQ(f.u_at(8 + i, 8 + j), ref_clip1(70 + delta));
        CHECK_EQ(f.v_at(8 + i, 8 + j), ref_clip1(190 + delta));
      }
    }
  }

  // Extreme clipping: luma 255, chroma 250/5, alpha +-8*32.
  Fixture f(64, 64, 4, 4, 0, 0);
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; ++col) {
      f.luma_at(row, col) = 255;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      f.u_at(i, j) = 250;
      f.v_at(i, j) = 5;
    }
  }
  Scratch scratch;
  compute_luma(f.u, scratch);
  apply_prediction(f.u, alpha_q8(Alpha{kSignPos, 8}), scratch);
  apply_prediction(f.v, alpha_q8(Alpha{kSignNeg, 8}), scratch);
  CHECK_EQ(f.u_at(0, 0), 255);
  CHECK_EQ(f.v_at(0, 0), 0);
}

void test_edges_and_partial() {
  using namespace av2enc::cfl;
  const CflCdfs cdfs = default_cfl_cdfs();
  Config config{};
  config.cost.use_rate = false;

  const int positions[4] = {0, 4, 28, 56};
  for (int py = 0; py < 4; ++py) {
    for (int px = 0; px < 4; ++px) {
      Fixture f(64, 64, 8, 8, positions[px], positions[py]);
      fill_luma_ramp(f);
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          f.src_u_at(f.u.y + i, f.u.x + j) =
              static_cast<std::uint8_t>(120 + ((i * 3 + j) % 15));
          f.src_v_at(f.u.y + i, f.u.x + j) =
              static_cast<std::uint8_t>(130 - ((i + j * 2) % 15));
        }
      }
      Scratch scratch;
      const auto decision = decide_cfl_alpha(
          f.u, f.v, f.plane_source_u(), f.plane_source_v(), config, cdfs,
          scratch);
      CHECK(decision.has_value());
      if (!decision.has_value()) {
        continue;
      }
      CHECK_EQ(decision->chosen.cost_q16(config.cost),
               config.cost.cost_q16(decision->chosen.total_sse(),
                                   decision->chosen.rate_q8));
      if (decision->chosen.explicit_alpha) {
        CHECK(av2enc::cfl::signalable(decision->chosen.explicit_pair));
        const Signalling s = decision->chosen.syntax;
        CHECK(pair_from_signalling_checked(s).has_value());
      }
      // Applying the candidate and recomputing SSE with the same derived Q8
      // matches the stored SSE (no hidden state changed).
      const std::vector<std::uint8_t> u_dc = f.chroma_u;
      const std::vector<std::uint8_t> v_dc = f.chroma_v;
      apply_candidate(f.u, f.v, decision->chosen, scratch);
      if (!decision->chosen.explicit_alpha) {
        CHECK_EQ(prediction_sse(f.u, decision->chosen.q8_u, scratch,
                                f.plane_source_u()),
                 decision->chosen.sse_u);
      }
      f.chroma_u = u_dc;
      f.chroma_v = v_dc;
    }
  }

  // Partial right/bottom block: coded grid smaller than the block, so the
  // luma-average replication and padded-plane reads are exercised.
  Fixture p(6, 6, 8, 8, 2, 2);
  fill_luma_ramp(p);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      p.src_u_at(2 + i, 2 + j) = static_cast<std::uint8_t>(100 + i + j);
      p.src_v_at(2 + i, 2 + j) = static_cast<std::uint8_t>(150 - i + j);
    }
  }
  Scratch scratch;
  const auto partial = decide_cfl_alpha(
      p.u, p.v, p.plane_source_u(), p.plane_source_v(), config, cdfs, scratch);
  CHECK(partial.has_value());
  if (partial.has_value()) {
    CHECK(partial->chosen.total_sse() >= 0);
    const auto reevaluated = evaluate_signalling(
        partial->chosen.syntax, p.u, p.v, p.plane_source_u(),
        p.plane_source_v(), cdfs, scratch);
    CHECK(reevaluated.has_value());
    if (reevaluated.has_value()) {
      CHECK_EQ(reevaluated->q8_u, partial->chosen.q8_u);
      CHECK_EQ(reevaluated->q8_v, partial->chosen.q8_v);
      CHECK_EQ(reevaluated->total_sse(), partial->chosen.total_sse());
    }
  }
}

// ---------------------------------------------------------------------------
// 10. Validation.
// ---------------------------------------------------------------------------

void test_validation() {
  using namespace av2enc::cfl;
  const CflCdfs cdfs = default_cfl_cdfs();
  Config config{};
  Fixture f(64, 64, 8, 8, 16, 16);
  fill_luma_ramp(f);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      f.src_u_at(16 + i, 16 + j) = 128;
      f.src_v_at(16 + i, 16 + j) = 128;
    }
  }
  Scratch scratch;
  CHECK(search_explicit(f.u, f.v, f.plane_source_u(), f.plane_source_v(),
                        config, cdfs, scratch)
            .has_value());
  CHECK(!search_explicit(f.u, f.v, SourcePlane{}, f.plane_source_v(), config,
                         cdfs, scratch)
             .has_value());
  CHECK(!decide_cfl_alpha(f.u, f.v, SourcePlane{}, f.plane_source_v(), config,
                          cdfs, scratch)
             .has_value());

  Block bad = f.u;
  bad.luma = nullptr;
  CHECK(!search_explicit(bad, f.v, f.plane_source_u(), f.plane_source_v(),
                         config, cdfs, scratch)
             .has_value());
  bad = f.u;
  bad.w = 65;
  CHECK(!decide_cfl_alpha(bad, f.v, f.plane_source_u(), f.plane_source_v(),
                          config, cdfs, scratch)
             .has_value());

  // Mismatched U/V geometry.
  Block other = f.v;
  other.x += 4;
  CHECK(!decide_cfl_alpha(f.u, other, f.plane_source_u(), f.plane_source_v(),
                          config, cdfs, scratch)
             .has_value());

  // evaluate_signalling rejects CFL_MULTI and non-canonical explicit.
  Signalling multi{};
  multi.cfl_index = kIndexMulti;
  CHECK(!evaluate_signalling(multi, f.u, f.v, f.plane_source_u(),
                             f.plane_source_v(), cdfs, scratch)
             .has_value());
  Signalling noncanonical{};
  noncanonical.cfl_index = kIndexExplicit;
  noncanonical.alpha_signs = 0;
  noncanonical.alpha_u = 3;
  noncanonical.alpha_v = 0;
  CHECK(!evaluate_signalling(noncanonical, f.u, f.v, f.plane_source_u(),
                             f.plane_source_v(), cdfs, scratch)
             .has_value());

  // Neither mode allowed.
  Config none{};
  none.allow_derived = false;
  none.allow_explicit = false;
  CHECK(!decide_cfl_alpha(f.u, f.v, f.plane_source_u(), f.plane_source_v(),
                          none, cdfs, scratch)
             .has_value());

  // Invalid lambda (negative / overflow-prone) is rejected rather than
  // producing a non-deterministic or wrapped cost.
  Config bad_lambda{};
  bad_lambda.cost.use_rate = true;
  bad_lambda.cost.lambda_q8 = -1;
  CHECK(!search_explicit(f.u, f.v, f.plane_source_u(), f.plane_source_v(),
                         bad_lambda, cdfs, scratch)
             .has_value());
  CHECK(!decide_cfl_alpha(f.u, f.v, f.plane_source_u(), f.plane_source_v(),
                          bad_lambda, cdfs, scratch)
             .has_value());
  bad_lambda.cost.lambda_q8 = std::int64_t{1} << 41;
  CHECK(!decide_cfl_alpha(f.u, f.v, f.plane_source_u(), f.plane_source_v(),
                          bad_lambda, cdfs, scratch)
             .has_value());
}

// ---------------------------------------------------------------------------
// 11. Randomised cross-check of the RD/SSE selection.
// ---------------------------------------------------------------------------

void test_randomised_selection() {
  using namespace av2enc::cfl;
  const CflCdfs cdfs = default_cfl_cdfs();
  std::mt19937 rng(0xCf1u);
  int decisions = 0;
  for (int iteration = 0; iteration < 120; ++iteration) {
    const int w = 4 + static_cast<int>(rng() % 9);
    const int h = 4 + static_cast<int>(rng() % 9);
    const int x = 4 + static_cast<int>(rng() % static_cast<unsigned>(100 - w));
    const int y = 4 + static_cast<int>(rng() % static_cast<unsigned>(100 - h));
    Fixture f(64, 64, w, h, x, y);
    for (auto& value : f.luma) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& value : f.chroma_u) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& value : f.chroma_v) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        f.src_u_at(y + i, x + j) = static_cast<std::uint8_t>(rng() & 0xFF);
        f.src_v_at(y + i, x + j) = static_cast<std::uint8_t>(rng() & 0xFF);
      }
    }
    f.u.ds_filter = static_cast<int>(rng() % 4);
    f.v.ds_filter = f.u.ds_filter;
    f.u.sb_top_y = ((y * 2) / 64) * 64;
    f.v.sb_top_y = f.u.sb_top_y;

    Config sse_config{};
    sse_config.cost.use_rate = false;
    Scratch scratch;
    const auto decision = decide_cfl_alpha(
        f.u, f.v, f.plane_source_u(), f.plane_source_v(), sse_config, cdfs,
        scratch);
    CHECK(decision.has_value());
    if (!decision.has_value()) {
      continue;
    }
    // The chosen candidate is never worse than either input candidate.
    CHECK(decision->chosen.total_sse() <= decision->derived.total_sse());
    CHECK(decision->chosen.total_sse() <= decision->explicit_alpha.total_sse());
    // Explicit candidates are always signalable and canonical.
    if (decision->explicit_alpha.explicit_alpha) {
      CHECK(av2enc::cfl::signalable(
          decision->explicit_alpha.explicit_pair));
      CHECK(pair_from_signalling_checked(
                decision->explicit_alpha.syntax)
                .has_value());
    }
    // Re-evaluating the chosen syntax reproduces its exact alphas and SSE.
    const auto again = evaluate_signalling(
        decision->chosen.syntax, f.u, f.v, f.plane_source_u(),
        f.plane_source_v(), cdfs, scratch);
    CHECK(again.has_value());
    if (again.has_value()) {
      CHECK_EQ(again->q8_u, decision->chosen.q8_u);
      CHECK_EQ(again->q8_v, decision->chosen.q8_v);
      CHECK_EQ(again->total_sse(), decision->chosen.total_sse());
      CHECK_EQ(again->rate_q8, decision->chosen.rate_q8);
    }
    ++decisions;
  }
  CHECK(decisions > 100);
}

// ---------------------------------------------------------------------------
// 12. Randomised search vs an independent brute-force enumeration.
// ---------------------------------------------------------------------------

void test_randomised_bruteforce() {
  using namespace av2enc::cfl;
  const CflCdfs cdfs = default_cfl_cdfs();
  std::mt19937 rng(0xB7u);
  int cases = 0;
  int rd_cases = 0;
  for (int iteration = 0; iteration < 30; ++iteration) {
    Fixture f(64, 64, 8, 8, 16, 16);
    for (auto& value : f.luma) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& value : f.chroma_u) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (auto& value : f.chroma_v) {
      value = static_cast<std::uint8_t>(rng() & 0xFF);
    }
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        f.src_u_at(16 + i, 16 + j) = static_cast<std::uint8_t>(rng() & 0xFF);
        f.src_v_at(16 + i, 16 + j) = static_cast<std::uint8_t>(rng() & 0xFF);
      }
    }
    f.u.ds_filter = static_cast<int>(rng() % 4);
    f.v.ds_filter = f.u.ds_filter;
    Scratch scratch;
    compute_luma(f.u, scratch);

    // Independent brute force over every signalable pair.
    std::int64_t best_sse = 0;
    bool have_best = false;
    std::int64_t best_rd[2] = {0, 0};
    bool have_rd[2] = {false, false};
    const std::int64_t lambdas[2] = {256, 20480};  // lambda 1 and 80
    Config sse_config{};
    sse_config.cost.use_rate = false;
    for (int signs = 0; signs < kJointSigns; ++signs) {
      for (int mag_u = 1; mag_u <= kAlphabetSize; ++mag_u) {
        for (int mag_v = 1; mag_v <= kAlphabetSize; ++mag_v) {
          const AlphaPair pair = pair_from_joint(signs, mag_u, mag_v);
          const std::int64_t su =
              prediction_sse(f.u, alpha_q8(pair.u), scratch, f.plane_source_u());
          const std::int64_t sv =
              prediction_sse(f.v, alpha_q8(pair.v), scratch, f.plane_source_v());
          const std::int64_t sse = su + sv;
          if (!have_best || sse < best_sse) {
            have_best = true;
            best_sse = sse;
          }
          const Signalling s = signalling_for(pair);
          const int rate = alpha_syntax_rate_q8(s, cdfs);
          for (int k = 0; k < 2; ++k) {
            const std::int64_t cost =
                sse * 65536 + lambdas[k] * rate;
            if (!have_rd[k] || cost < best_rd[k]) {
              have_rd[k] = true;
              best_rd[k] = cost;
            }
          }
        }
      }
    }

    const auto found = search_explicit(f.u, f.v, f.plane_source_u(),
                                       f.plane_source_v(), sse_config, cdfs,
                                       scratch);
    CHECK(found.has_value());
    if (!found.has_value()) {
      continue;
    }
    CHECK_EQ(found->total_sse(), best_sse);
    // The existing core oracle (per-plane exhaustive SSE + (zero,zero)
    // repair) must reach the same optimum.
    const auto core_oracle = search_alpha(
        f.u, f.v, f.plane_source_u(), f.plane_source_v(),
        SearchMode::kExhaustiveSse, scratch);
    CHECK(core_oracle.has_value());
    if (core_oracle.has_value()) {
      CHECK_EQ(core_oracle->total_sse(), found->total_sse());
    }
    for (int k = 0; k < 2; ++k) {
      Config rd_config{};
      rd_config.cost.use_rate = true;
      rd_config.cost.lambda_q8 = lambdas[k];
      rd_config.allow_derived = false;  // compare the explicit branch only
      const auto rd = decide_cfl_alpha(f.u, f.v, f.plane_source_u(),
                                       f.plane_source_v(), rd_config, cdfs,
                                       scratch);
      CHECK(rd.has_value());
      if (rd.has_value()) {
        CHECK_EQ(rd->chosen.cost_q16(rd_config.cost), best_rd[k]);
        ++rd_cases;
      }
    }
    ++cases;
  }
  CHECK(cases >= 25);
  CHECK(rd_cases == 2 * cases);
}

}  // namespace

int main() {
  test_quantisation();
  test_derived_quantisation();
  test_signalling_all();
  test_cdfs_pinned();
  test_model_rate();
  test_syntax_roundtrip_all();
  test_pinned_payloads();
  test_search_recovery_all();
  test_selection();
  test_all_alphas_prediction();
  test_edges_and_partial();
  test_validation();
  test_randomised_selection();
  test_randomised_bruteforce();

  if (g_failures == 0) {
    std::printf("cfl_alpha_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::fprintf(stderr, "cfl_alpha_test: %d/%d checks FAILED\n", g_failures,
               g_checks);
  return 1;
}
