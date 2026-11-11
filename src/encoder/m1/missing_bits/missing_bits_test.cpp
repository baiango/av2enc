// Unit tests for src/encoder/m1/missing_bits/.
//
// Covers:
//   * §7.13.2.11 subsampled DC: extents 1..64, all four edge availabilities,
//     odd sizes, 8/10/12-bit, production parity when the rule is inactive.
//   * §7.13.2.14 filter corner: exhaustive w/h/pAngle guard, 8-bit input
//     sweep, production parity when inactive and an "is it active" check.
//   * §5.20.7.27 dc_sign_horz_vert: element selection, dc_sign context,
//     writer/reader round-trips and pinned entropy bytes.
//   * §7.13.3.15 save_subPU: TX_SIZES_ALL table parity with the deblock core,
//     early termination, exhaustive plane/subsampling/size coverage and
//     pinned array goldens.
//
// Usage:
//   missing_bits_test [--dump-goldens]

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "encoder/filters/deblock/deblock_core.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m0_cdf_tables.h"
#include "encoder/m1/missing_bits/dc_intra_subsampled.h"
#include "encoder/m1/missing_bits/dc_sign_horz_vert.h"
#include "encoder/m1/missing_bits/intra_filter_corner.h"
#include "encoder/m1/missing_bits/missing_bits_goldens.h"
#include "encoder/m1/missing_bits/save_subpu.h"

namespace mb = av2enc::m1::missing_bits;
namespace intra = av2enc::intra;
namespace deblock = av2enc::deblock;

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
      if (g_failures > 40) {                                              \
        std::printf("too many failures\n");                               \
        std::exit(1);                                                     \
      }                                                                   \
    }                                                                     \
  } while (0)

template <typename T>
long long as_ll(T value) {
  if constexpr (std::is_enum_v<T>) {
    return static_cast<long long>(static_cast<std::underlying_type_t<T>>(value));
  } else {
    return static_cast<long long>(value);
  }
}

#define CHECK_EQ(a, b)                                                    \
  do {                                                                    \
    ++g_checks;                                                           \
    const auto va = (a);                                                  \
    const auto vb = (b);                                                  \
    if (!(va == vb)) {                                                    \
      ++g_failures;                                                       \
      std::printf("FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__,      \
                  __LINE__, #a, #b, as_ll(va), as_ll(vb));                \
      if (g_failures > 40) {                                              \
        std::printf("too many failures\n");                               \
        std::exit(1);                                                     \
      }                                                                   \
    }                                                                     \
  } while (0)

// --- independent §7.13.2.11 helpers for expected values --------------------

int ref_floor_log2(int v) {
  int r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

int ref_round2(long long v, int shift) {
  if (shift <= 0) return static_cast<int>(v);
  return static_cast<int>((v + (1LL << (shift - 1))) >> shift);
}

int ref_resolve_divisor(int dad) {
  const int n = ref_floor_log2(dad);
  const int e = dad - (1 << n);
  const int f = n > 7 ? ref_round2(e, n - 7) : e << (7 - n);
  return static_cast<int>(mb::kDivLut[f]);
}

int ref_approx_divide(long long num, int den) {
  return ref_round2(num * ref_resolve_divisor(den), ref_floor_log2(den) + 9);
}

struct ExpectedDc {
  int step_w;
  int step_h;
  long long sum;
  int count;
  int avg;
};

ExpectedDc expected_dc(const std::vector<int>& above, const std::vector<int>& left,
                       int w, int h, int bit_depth, bool have_above,
                       bool have_left) {
  ExpectedDc out{};
  out.step_w = have_above ? (w > 32 ? 2 : 1) : 0;
  out.step_h = have_left ? (h > 32 ? 2 : 1) : 0;
  if (have_left) {
    for (int k = 0; k < h; k += out.step_h) {
      out.sum += left[static_cast<std::size_t>(k)];
      ++out.count;
    }
  }
  if (have_above) {
    for (int k = 0; k < w; k += out.step_w) {
      out.sum += above[static_cast<std::size_t>(k)];
      ++out.count;
    }
  }
  if (out.count == 0) {
    out.avg = 1 << (bit_depth - 1);
  } else {
    const int high = (1 << bit_depth) - 1;
    const int avg = ref_approx_divide(out.sum, out.count);
    out.avg = avg < 0 ? 0 : (avg > high ? high : avg);
  }
  return out;
}

// Deterministic xorshift for reproducible edge samples.
std::uint32_t g_rng = 0x9e3779b9u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

std::vector<int> make_row(int n, int max_value) {
  std::vector<int> v(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    v[static_cast<std::size_t>(i)] =
        static_cast<int>(next_random() % static_cast<std::uint32_t>(max_value + 1));
  }
  return v;
}

std::string hex_of(const std::uint8_t* data, std::size_t n) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(digits[data[i] >> 4]);
    out.push_back(digits[data[i] & 0xf]);
  }
  return out;
}

// --- 1. §7.13.2.11 ----------------------------------------------------------

void test_dc_subsampled() {
  // stepW/stepH.
  for (int extent = 1; extent <= 64; ++extent) {
    CHECK_EQ(mb::dc_subsampled_step(extent), extent > 32 ? 2 : 1);
  }

  // approx_divide against the independent transcription.
  for (int den = 1; den <= 64; ++den) {
    for (int num = 0; num <= 4096; num += 7) {
      CHECK_EQ(mb::approx_divide(num, den), ref_approx_divide(num, den));
    }
  }

  // Exhaustive extents 1..64 x 4 availability combinations x 8/10/12-bit.
  for (int bit_depth : {8, 10, 12}) {
    const int max_value = (1 << bit_depth) - 1;
    for (int w = 1; w <= 64; ++w) {
      for (int h = 1; h <= 64; ++h) {
        const std::vector<int> above = make_row(w, max_value);
        const std::vector<int> left = make_row(h, max_value);
        std::vector<std::int16_t> above16(above.begin(), above.end());
        std::vector<std::int16_t> left16(left.begin(), left.end());
        for (int mask = 0; mask < 4; ++mask) {
          const bool have_above = (mask & 1) != 0;
          const bool have_left = (mask & 2) != 0;
          mb::DcSubsampledInput in;
          in.above = above16.data();
          in.left = left16.data();
          in.w = w;
          in.h = h;
          in.bit_depth = bit_depth;
          in.have_above = have_above;
          in.have_left = have_left;
          std::vector<std::uint16_t> dst(
              static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
          mb::DcSubsampledTrace trace;
          CHECK(mb::dc_intra_subsampled(in, dst.data(), &trace));
          const ExpectedDc expected = expected_dc(above, left, w, h, bit_depth,
                                                  have_above, have_left);
          CHECK_EQ(trace.step_w, expected.step_w);
          CHECK_EQ(trace.step_h, expected.step_h);
          CHECK_EQ(trace.sum, static_cast<int>(expected.sum));
          CHECK_EQ(trace.count, expected.count);
          CHECK_EQ(trace.avg, expected.avg);
          for (std::uint16_t value : dst) {
            CHECK_EQ(static_cast<int>(value), expected.avg);
          }
        }
      }
    }
  }

  // Invalid inputs.
  {
    mb::DcSubsampledInput in;
    in.w = 0;
    in.h = 4;
    std::uint16_t dst[16] = {};
    CHECK(!mb::dc_intra_subsampled(in, dst, 4));
    in.w = 65;
    in.h = 4;
    CHECK(!mb::dc_intra_subsampled(in, dst, 65));
    in.w = 4;
    in.h = 4;
    in.bit_depth = 7;
    CHECK(!mb::dc_intra_subsampled(in, dst, 4));
  }

  // Production parity when the rule is inactive (w <= 32 && h <= 32).
  // Production `predict_dc` implements §7.13.2.10; the two processes must be
  // byte-identical for every availability combination on power-of-two extents.
  for (int w : {4, 8, 16, 32}) {
    for (int h : {4, 8, 16, 32}) {
      const std::vector<int> above = make_row(w, 255);
      const std::vector<int> left = make_row(h, 255);
      std::vector<std::int16_t> above16(above.begin(), above.end());
      std::vector<std::int16_t> left16(left.begin(), left.end());
      for (int mask = 0; mask < 4; ++mask) {
        const bool have_above = (mask & 1) != 0;
        const bool have_left = (mask & 2) != 0;

        intra::IntraEdges edges;
        edges.w = w;
        edges.h = h;
        edges.bit_depth = 8;
        edges.have_above = have_above;
        edges.have_left = have_left;
        edges.valid_above_px = w;
        edges.valid_left_px = h;
        for (int i = 0; i < w; ++i) {
          edges.above[i] = above16[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < h; ++i) {
          edges.left[i] = left16[static_cast<std::size_t>(i)];
        }
        edges.above[-1] = edges.above[0];
        edges.left[-1] = edges.left[0];

        std::vector<std::uint8_t> production(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
        intra::PredictParams params;
        params.mode = intra::Mode::kDc;
        CHECK(intra::predict_intra(edges, params, production.data(), static_cast<std::size_t>(w)) ==
              intra::IntraStatus::kOk);

        mb::DcSubsampledInput in;
        in.above = above16.data();
        in.left = left16.data();
        in.w = w;
        in.h = h;
        in.bit_depth = 8;
        in.have_above = have_above;
        in.have_left = have_left;
        std::vector<std::uint16_t> reference(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
        CHECK(mb::dc_intra_subsampled(in, reference.data()));
        std::vector<std::uint8_t> narrowed(reference.size());
        for (std::size_t i = 0; i < reference.size(); ++i) {
          narrowed[i] = static_cast<std::uint8_t>(reference[i]);
        }
        CHECK(std::memcmp(production.data(), narrowed.data(),
                          production.size()) == 0);
      }
    }
  }

  // Active rule: the subsampled average differs from §7.13.2.10 for a
  // saw-tooth edge, and matches the pinned golden.
  {
    mb::DcSubsampledInput in;
    std::vector<std::int16_t> above(64);
    std::vector<std::int16_t> left(64);
    for (int i = 0; i < 64; ++i) {
      above[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>((i & 1) ? 255 : 0);
      left[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>((i & 1) ? 0 : 255);
    }
    in.above = above.data();
    in.left = left.data();
    in.w = 64;
    in.h = 64;
    in.bit_depth = 8;
    in.have_above = true;
    in.have_left = true;
    mb::DcSubsampledTrace trace;
    std::uint16_t dst[64 * 64];
    CHECK(mb::dc_intra_subsampled(in, dst, 64, &trace));
    CHECK_EQ(trace.step_w, 2);
    CHECK_EQ(trace.step_h, 2);
    CHECK_EQ(trace.count, 64);
    CHECK_EQ(trace.avg, 128);  // 32 even above (0) + 32 even left (255)
  }

  // Pinned goldens.
  for (const mb::goldens::DcGolden& g : mb::goldens::kDcGoldens) {
    if (g.w == 0) continue;  // placeholder until --dump-goldens is pinned
    std::vector<std::int16_t> above(static_cast<std::size_t>(g.w));
    std::vector<std::int16_t> left(static_cast<std::size_t>(g.h));
    const int max_value = (1 << g.bit_depth) - 1;
    for (int i = 0; i < g.w; ++i) {
      above[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>((i * 7 + 3) & max_value);
    }
    for (int i = 0; i < g.h; ++i) {
      left[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>((i * 5 + 11) & max_value);
    }
    mb::DcSubsampledInput in;
    in.above = above.data();
    in.left = left.data();
    in.w = g.w;
    in.h = g.h;
    in.bit_depth = g.bit_depth;
    in.have_above = g.have_above;
    in.have_left = g.have_left;
    std::vector<std::uint16_t> dst(
        static_cast<std::size_t>(g.w) * static_cast<std::size_t>(g.h));
    mb::DcSubsampledTrace trace;
    CHECK(mb::dc_intra_subsampled(in, dst.data(), &trace));
    CHECK_EQ(trace.step_w, g.step_w);
    CHECK_EQ(trace.step_h, g.step_h);
    CHECK_EQ(trace.sum, g.sum);
    CHECK_EQ(trace.count, g.count);
    CHECK_EQ(trace.avg, g.avg);
  }
}

// --- 2. §7.13.2.14 ----------------------------------------------------------

void test_filter_corner() {
  // Exhaustive guard over the legal block extents and all signalable pAngles.
  for (int w = 1; w <= 64; ++w) {
    for (int h = 1; h <= 64; ++h) {
      for (int angle = 1; angle < 270; ++angle) {
        const bool expected =
            angle > 90 && angle < 180 && (w + h) >= 24;
        CHECK_EQ(mb::filter_corner_applies(w, h, angle), expected);
      }
    }
  }

  // Literal 5/6/5 subset and Round2(...,4) for every 8-bit input.
  for (int left0 = 0; left0 < 256; ++left0) {
    for (int above_m1 = 0; above_m1 < 256; ++above_m1) {
      for (int above0 = 0; above0 < 256; ++above0) {
        const int expected =
            (left0 * 5 + above_m1 * 6 + above0 * 5 + 8) >> 4;
        CHECK_EQ(mb::filter_corner_value(left0, above_m1, above0), expected);
      }
    }
  }

  // Both corner slots are set to the same value; negative indices untouched
  // otherwise.
  {
    intra::EdgeRow above;
    intra::EdgeRow left;
    above[-2] = 7;
    left[-2] = 9;
    above[0] = 200;
    left[0] = 10;
    above[-1] = 30;
    const int expected = mb::filter_corner_value(10, 30, 200);
    CHECK(mb::filter_corner_if_applicable(16, 16, 135, above, left));
    CHECK_EQ(above[-1], expected);
    CHECK_EQ(left[-1], expected);
    CHECK_EQ(above[-2], 7);
    CHECK_EQ(left[-2], 9);
    CHECK(!mb::filter_corner_if_applicable(8, 8, 135, above, left));
    CHECK(!mb::filter_corner_if_applicable(16, 16, 90, above, left));
    CHECK(!mb::filter_corner_if_applicable(16, 16, 180, above, left));
  }

  // Production parity when inactive: pre-applying the reference to an
  // IntraEdges copy must not change predict_intra() output for any
  // directional mode / angle delta / size where the guard is false.
  int inactive_cases = 0;
  int active_changed = 0;
  int active_cases = 0;
  int total_cases = 0;
  for (int w : {4, 8, 16, 32, 64}) {
    for (int h : {4, 8, 16, 32, 64}) {
      for (int mode_index = 0; mode_index < intra::kNumModes; ++mode_index) {
        const intra::Mode mode = static_cast<intra::Mode>(mode_index);
        if (!intra::is_directional(mode)) continue;
        for (int delta = -3; delta <= 3; ++delta) {
          ++total_cases;
          intra::IntraEdges edges;
          edges.w = w;
          edges.h = h;
          edges.bit_depth = 8;
          edges.have_above = true;
          edges.have_left = true;
          edges.valid_above_px = w;
          edges.valid_left_px = h;
          for (int i = 0; i < w + h + 2; ++i) {
            edges.above[i] = static_cast<std::int16_t>((i * 37 + 11) & 255);
            edges.left[i] = static_cast<std::int16_t>((i * 53 + 29) & 255);
          }
          edges.above[-1] = 71;
          edges.left[-1] = 71;

          intra::PredictParams params;
          params.mode = mode;
          params.angle_delta = delta;
          params.enable_edge_filter = true;
          params.above_neighbor_is_smooth = true;
          params.left_neighbor_is_smooth = true;

          const intra::MappedAngle mapped =
              intra::map_angle(mode, delta, w, h);
          if (mapped.p_angle <= 0 || mapped.p_angle >= 270) continue;

          std::vector<std::uint8_t> baseline(
              static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
          CHECK(intra::predict_intra(edges, params, baseline.data(), static_cast<std::size_t>(w)) ==
                intra::IntraStatus::kOk);

          intra::IntraEdges modified = edges;
          const bool applied = mb::filter_corner_if_applicable(
              w, h, mapped.p_angle, modified.above, modified.left);
          std::vector<std::uint8_t> after(
              static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
          CHECK(intra::predict_intra(modified, params, after.data(), static_cast<std::size_t>(w)) ==
                intra::IntraStatus::kOk);

          if (applied) {
            ++active_cases;
            if (std::memcmp(baseline.data(), after.data(), baseline.size()) !=
                0) {
              ++active_changed;
            }
          } else {
            ++inactive_cases;
            CHECK(std::memcmp(baseline.data(), after.data(),
                              baseline.size()) == 0);
          }
        }
      }
    }
  }
  CHECK(inactive_cases > 0);
  CHECK_EQ(inactive_cases + active_cases, total_cases);
  // Production really consumes the corner: pre-applying it changes at least
  // one prediction, which proves the call site is live.
  CHECK(active_cases > 0);
  CHECK(active_changed > 0);

  // Pinned goldens.
  for (const auto& g : mb::goldens::kCornerGoldens) {
    if (g[0] == 0 && g[1] == 0 && g[2] == 0 && g[3] == 0) continue;
    CHECK_EQ(mb::filter_corner_value(g[0], g[1], g[2]), g[3]);
  }
}

// --- 3. §5.20.7.27 dc_sign_horz_vert ---------------------------------------

mb::SignElement expected_sign_element(int plane, int row, int col,
                                      mb::TxClass tx_class) {
  if (plane == 0) {
    if (row == 0 && col == 0) return mb::SignElement::kDcSign;
    if (tx_class == mb::TxClass::kHoriz && col == 0) {
      return mb::SignElement::kDcSignHorzVert;
    }
    if (tx_class == mb::TxClass::kVert && row == 0) {
      return mb::SignElement::kDcSignHorzVert;
    }
  }
  return mb::SignElement::kSignBit;
}

int expected_dc_sign_context(const std::vector<std::int8_t>& above,
                             const std::vector<std::int8_t>& left, int x4,
                             int y4, int w4, int h4, int mi_cols,
                             int mi_rows) {
  int dc_sign = 0;
  for (int k = 0; k < w4; ++k) {
    if (x4 + k < mi_cols) {
      const int sign = above[static_cast<std::size_t>(x4 + k)];
      if (sign == 1) --dc_sign;
      else if (sign == 2) ++dc_sign;
    }
  }
  for (int k = 0; k < h4; ++k) {
    if (y4 + k < mi_rows) {
      const int sign = left[static_cast<std::size_t>(y4 + k)];
      if (sign == 1) --dc_sign;
      else if (sign == 2) ++dc_sign;
    }
  }
  return dc_sign < 0 ? 1 : (dc_sign > 0 ? 2 : 0);
}

void test_dc_sign_horz_vert() {
  // Element selection, exhaustive over the structural inputs.
  for (int plane = 0; plane <= 2; ++plane) {
    for (int row = 0; row <= 8; ++row) {
      for (int col = 0; col <= 8; ++col) {
        for (int cls = 0; cls < 3; ++cls) {
          const auto tx_class = static_cast<mb::TxClass>(cls);
          CHECK_EQ(mb::select_sign_element(plane, row, col, tx_class),
                   expected_sign_element(plane, row, col, tx_class));
        }
      }
    }
  }
  CHECK_EQ(mb::dc_sign_horz_vert_context(), 0);

  // dc_sign context.
  for (int trial = 0; trial < 4000; ++trial) {
    const int mi_cols = 1 + static_cast<int>(next_random() % 8);
    const int mi_rows = 1 + static_cast<int>(next_random() % 8);
    std::vector<std::int8_t> above(static_cast<std::size_t>(mi_cols));
    std::vector<std::int8_t> left(static_cast<std::size_t>(mi_rows));
    for (auto& v : above) v = static_cast<std::int8_t>(next_random() % 3);
    for (auto& v : left) v = static_cast<std::int8_t>(next_random() % 3);
    const int x4 = static_cast<int>(next_random() % 8);
    const int y4 = static_cast<int>(next_random() % 8);
    const int w4 = static_cast<int>(next_random() % 5);
    const int h4 = static_cast<int>(next_random() % 5);
    CHECK_EQ(mb::dc_sign_context(above.data(), left.data(), x4, y4, w4, h4,
                                 mi_cols, mi_rows),
             expected_dc_sign_context(above, left, x4, y4, w4, h4, mi_cols,
                                      mi_rows));
  }

  // CDF slot resolution.
  {
    mb::DcSignCdfTable table = {};
    for (int p = 0; p < 2; ++p) {
      for (int h = 0; h < 2; ++h) {
        for (int c = 0; c < 3; ++c) {
          CHECK(mb::dc_sign_cdf_slot(table, p, h, c) == &table[p][h][c][0]);
        }
        CHECK(mb::dc_sign_horz_vert_cdf_slot(table, p, h) ==
              &table[p][h][0][0]);
      }
    }
  }

  // Writer + reader round-trip with CDF adaptation, both frozen and adaptive.
  const int signs[10] = {0, 1, 1, 0, 1, 0, 0, 1, 1, 1};
  for (int frozen = 0; frozen <= 1; ++frozen) {
    std::uint16_t enc_cdf[2][2][3][3];
    std::uint16_t dec_cdf[2][2][3][3];
    std::memcpy(enc_cdf, av2enc::m0::kDcSignCdf, sizeof(enc_cdf));
    std::memcpy(dec_cdf, av2enc::m0::kDcSignCdf, sizeof(dec_cdf));

    std::vector<std::uint8_t> buffer(256, 0);
    av2enc::av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
          av2enc::av2::Status::kOk);
    encoder.disable_cdf_update = static_cast<std::uint8_t>(frozen);
    for (int sign : signs) {
      CHECK(mb::write_dc_sign_horz_vert(
          encoder, mb::dc_sign_horz_vert_cdf_slot(enc_cdf, 0, 0), sign));
    }
    std::size_t bytes = 0;
    CHECK(encoder.exit_symbol(&bytes) == av2enc::av2::Status::kOk);

    std::vector<std::uint8_t> direct(256, 0);
    av2enc::av2::Encoder plain;
    CHECK(plain.init_symbol(direct.data(), direct.size()) ==
          av2enc::av2::Status::kOk);
    plain.disable_cdf_update = static_cast<std::uint8_t>(frozen);
    std::uint16_t plain_cdf[2][2][3][3];
    std::memcpy(plain_cdf, av2enc::m0::kDcSignCdf, sizeof(plain_cdf));
    for (int sign : signs) {
      CHECK(plain.encode_symbol(
                mb::dc_sign_horz_vert_cdf_slot(plain_cdf, 0, 0), 2, sign) ==
            av2enc::av2::Status::kOk);
    }
    std::size_t plain_bytes = 0;
    CHECK(plain.exit_symbol(&plain_bytes) == av2enc::av2::Status::kOk);
    CHECK_EQ(bytes, plain_bytes);
    CHECK(std::memcmp(buffer.data(), direct.data(), bytes) == 0);
    CHECK(std::memcmp(enc_cdf, plain_cdf, sizeof(enc_cdf)) == 0);

    av2enc::av2::Decoder decoder;
    CHECK(decoder.init_symbol(buffer.data(), buffer.size(), bytes) ==
          av2enc::av2::Status::kOk);
    decoder.disable_cdf_update = static_cast<std::uint8_t>(frozen);
    for (int i = 0; i < 10; ++i) {
      int sign = -1;
      CHECK(mb::read_dc_sign_horz_vert(
          decoder, mb::dc_sign_horz_vert_cdf_slot(dec_cdf, 0, 0), &sign));
      CHECK_EQ(sign, signs[i]);
    }
    CHECK(decoder.exit_symbol() == av2enc::av2::Status::kOk);
    CHECK(std::memcmp(enc_cdf, dec_cdf, sizeof(enc_cdf)) == 0);

    if (frozen == 0) {
      CHECK(hex_of(buffer.data(), bytes) == std::string(mb::goldens::kSignAdaptiveHex));
    } else {
      CHECK(hex_of(buffer.data(), bytes) == std::string(mb::goldens::kSignFrozenHex));
    }
  }

  // Dispatcher: all three elements round-trip in one stream.
  {
    std::vector<std::uint8_t> buffer(512, 0);
    av2enc::av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
          av2enc::av2::Status::kOk);
    std::uint16_t enc_cdf[2][2][3][3];
    std::uint16_t dec_cdf[2][2][3][3];
    std::memcpy(enc_cdf, av2enc::m0::kDcSignCdf, sizeof(enc_cdf));
    std::memcpy(dec_cdf, av2enc::m0::kDcSignCdf, sizeof(dec_cdf));
    struct Step {
      mb::SignElement element;
      int ctx;
      int sign;
    };
    const Step steps[] = {
        {mb::SignElement::kDcSign, 2, 1},
        {mb::SignElement::kDcSignHorzVert, 0, 0},
        {mb::SignElement::kSignBit, 0, 1},
        {mb::SignElement::kDcSign, 0, 0},
        {mb::SignElement::kDcSignHorzVert, 0, 1},
        {mb::SignElement::kSignBit, 0, 0},
    };
    for (const Step& step : steps) {
      const int ctx = step.element == mb::SignElement::kDcSign ? step.ctx : 0;
      CHECK(mb::write_sign_element(encoder, step.element,
                                   mb::dc_sign_cdf_slot(enc_cdf, 0, 0, ctx),
                                   ctx, step.sign));
    }
    std::size_t bytes = 0;
    CHECK(encoder.exit_symbol(&bytes) == av2enc::av2::Status::kOk);
    av2enc::av2::Decoder decoder;
    CHECK(decoder.init_symbol(buffer.data(), buffer.size(), bytes) ==
          av2enc::av2::Status::kOk);
    for (const Step& step : steps) {
      const int ctx = step.element == mb::SignElement::kDcSign ? step.ctx : 0;
      int sign = -1;
      CHECK(mb::read_sign_element(decoder, step.element,
                                  mb::dc_sign_cdf_slot(dec_cdf, 0, 0, ctx),
                                  ctx, &sign));
      CHECK_EQ(sign, step.sign);
    }
    CHECK(decoder.exit_symbol() == av2enc::av2::Status::kOk);
    CHECK(std::memcmp(enc_cdf, dec_cdf, sizeof(enc_cdf)) == 0);
  }

  // Invalid arguments.
  {
    std::vector<std::uint8_t> buffer(64, 0);
    av2enc::av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
          av2enc::av2::Status::kOk);
    CHECK(!mb::write_dc_sign_horz_vert(encoder, nullptr, 0));
    std::uint16_t cdf[3] = {16384, 0, 0};
    CHECK(!mb::write_dc_sign_horz_vert(encoder, cdf, 2));
    CHECK(!mb::write_dc_sign(encoder, cdf, 3, 0));
  }
}

// --- 4. §7.13.3.15 save_subPU ---------------------------------------------

void test_save_subpu() {
  // TX_SIZES_ALL tables match the production deblock model.
  for (int i = 0; i < mb::kTxSizesAll; ++i) {
    const auto size = static_cast<mb::TxSize>(i);
    const auto prod = static_cast<deblock::TxSize>(i);
    CHECK_EQ(mb::tx_width(size), deblock::tx_width(prod));
    CHECK_EQ(mb::tx_height(size), deblock::tx_height(prod));
    CHECK_EQ(mb::find_tx_size(mb::tx_width(size), mb::tx_height(size)), i);
  }
  CHECK_EQ(mb::find_tx_size(4, 12), mb::kTxInvalid);
  CHECK_EQ(mb::find_tx_size(0, 0), mb::kTxInvalid);
  CHECK_EQ(mb::find_tx_size(-4, 4), mb::kTxInvalid);
  CHECK_EQ(mb::find_tx_size(6, 6), mb::kTxInvalid);

  // Early termination: w == PuWidth && h == PuHeight leaves the grid alone.
  {
    mb::SubPuGrid grid;
    grid.reset(8, 8);
    for (auto& v : grid.size[0]) v = 0xAA;
    for (auto& v : grid.col_base[0]) v = -7;
    CHECK(mb::save_subpu(0, 0, 0, 16, 8, 0, 0, 16, 8, grid));
    for (auto v : grid.size[0]) CHECK_EQ(static_cast<int>(v), 0xAA);
    for (auto v : grid.col_base[0]) CHECK_EQ(v, -7);
  }

  // Exhaustive valid sub-PU shapes: plane and subsampling coverage.
  for (int w : {4, 8, 16, 32, 64}) {
    for (int h : {4, 8, 16, 32, 64}) {
      const int tx = mb::find_tx_size(w, h);
      for (int plane = 0; plane <= 2; ++plane) {
        for (int sub = 0; sub <= 1; ++sub) {
          for (int x : {0, 4, 8, 16}) {
            for (int y : {0, 4, 8, 16}) {
              mb::SubPuGrid grid;
              grid.reset(64, 64);
              std::string error;
              const bool ok = mb::save_subpu(plane, x, y, w, h, sub, sub, 128,
                                             128, grid, &error);
              if (tx == mb::kTxInvalid) {
                CHECK(!ok);
                continue;
              }
              CHECK(ok);
              const int sub_x = plane > 0 ? sub : 0;
              const int sub_y = plane > 0 ? sub : 0;
              const int luma_row = y >> (2 - sub_y);
              const int luma_col = x >> (2 - sub_x);
              const int rows = h >> (2 - sub_y);
              const int cols = w >> (2 - sub_x);
              const int plane_index = plane > 0 ? 1 : 0;
              for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                  CHECK_EQ(static_cast<int>(grid.size_at(plane_index,
                                                         luma_row + r,
                                                         luma_col + c)),
                           tx);
                  CHECK_EQ(grid.col_base_at(plane_index, luma_row + r,
                                            luma_col + c),
                           luma_col);
                  CHECK_EQ(grid.row_base_at(plane_index, luma_row + r,
                                            luma_col + c),
                           luma_row);
                }
              }
            }
          }
        }
      }
    }
  }

  // Odd / fractional shapes are not transform sizes.
  for (int w : {5, 7, 9, 12, 33, 63}) {
    for (int h : {5, 7, 9, 12, 33, 63}) {
      mb::SubPuGrid grid;
      grid.reset(32, 32);
      CHECK(!mb::save_subpu(0, 0, 0, w, h, 0, 0, 128, 128, grid));
    }
  }

  // Grid bounds.
  {
    mb::SubPuGrid grid;
    grid.reset(4, 4);
    CHECK(!mb::save_subpu(0, 16, 0, 8, 8, 0, 0, 128, 128, grid));
    CHECK(!mb::save_subpu(0, 0, 16, 8, 8, 0, 0, 128, 128, grid));
    CHECK(!mb::save_subpu(3, 0, 0, 8, 8, 0, 0, 128, 128, grid));
    CHECK(!mb::save_subpu(0, -4, 0, 8, 8, 0, 0, 128, 128, grid));
  }

  // Pinned goldens.
  for (const mb::goldens::SubPuGolden& g : mb::goldens::kSubPuGoldens) {
    mb::SubPuGrid grid;
    grid.reset(64, 64);
    std::string error;
    CHECK(mb::save_subpu(g.plane, g.x, g.y, g.w, g.h, g.sub, g.sub, g.pu_w,
                         g.pu_h, grid, &error));
    CHECK_EQ(static_cast<int>(grid.size_at(g.plane > 0 ? 1 : 0, g.luma_row,
                                           g.luma_col)),
             g.tx);
    CHECK_EQ(grid.col_base_at(g.plane > 0 ? 1 : 0, g.luma_row, g.luma_col),
             g.luma_col);
    CHECK_EQ(grid.row_base_at(g.plane > 0 ? 1 : 0, g.luma_row, g.luma_col),
             g.luma_row);
    CHECK_EQ(static_cast<int>(
                 grid.size_at(g.plane > 0 ? 1 : 0, g.luma_row + g.rows - 1,
                              g.luma_col + g.cols - 1)),
             g.tx);
  }

  // Pinned examples: the M2 32x16 luma PU split into 16x16 sub-PUs and its
  // 16x8 chroma PU split into 8x8 sub-PUs.
  {
    mb::SubPuGrid grid;
    grid.reset(4, 8);  // MiRows=4, MiCols=8 for a 32x16 frame
    std::string error;
    CHECK(mb::save_subpu(0, 0, 0, 16, 16, 0, 0, 32, 16, grid, &error));
    CHECK_EQ(static_cast<int>(grid.size_at(0, 0, 0)), 2);  // TX_16X16
    CHECK_EQ(grid.col_base_at(0, 0, 0), 0);
    CHECK_EQ(grid.row_base_at(0, 0, 0), 0);
    CHECK_EQ(static_cast<int>(grid.size_at(0, 3, 3)), 2);
    CHECK(mb::save_subpu(0, 16, 0, 16, 16, 0, 0, 32, 16, grid, &error));
    CHECK_EQ(grid.col_base_at(0, 0, 4), 4);
    CHECK_EQ(static_cast<int>(grid.size_at(0, 3, 7)), 2);
    mb::SubPuGrid chroma;
    chroma.reset(4, 8);
    CHECK(mb::save_subpu(1, 0, 0, 8, 8, 1, 1, 16, 8, chroma, &error));
    CHECK_EQ(static_cast<int>(chroma.size_at(1, 0, 0)), 1);  // TX_8X8
    CHECK_EQ(chroma.col_base_at(1, 0, 0), 0);
    CHECK_EQ(chroma.row_base_at(1, 0, 0), 0);
    CHECK_EQ(chroma.col_base_at(1, 3, 3), 0);
    CHECK(mb::save_subpu(1, 8, 0, 8, 8, 1, 1, 16, 8, chroma, &error));
    CHECK_EQ(chroma.col_base_at(1, 0, 4), 4);
    CHECK_EQ(static_cast<int>(chroma.size_at(1, 3, 7)), 1);
  }
}

// --- goldens dump ----------------------------------------------------------

void dump_goldens() {
  std::printf("=== DC goldens ===\n");
  struct Case {
    int w, h, bd;
    bool above, left;
  };
  const Case cases[] = {
      {64, 64, 8, true, true},   {64, 32, 8, true, true},
      {48, 40, 8, true, true},   {33, 7, 8, true, true},
      {63, 37, 8, true, false},  {37, 63, 8, false, true},
      {32, 32, 8, true, true},   {16, 8, 8, true, true},
      {4, 4, 8, false, false},   {64, 64, 10, true, true},
      {64, 64, 12, true, false}, {7, 5, 8, true, true},
  };
  for (const Case& c : cases) {
    std::vector<int> above(static_cast<std::size_t>(c.w));
    std::vector<int> left(static_cast<std::size_t>(c.h));
    const int mask = (1 << c.bd) - 1;
    for (int i = 0; i < c.w; ++i) {
      above[static_cast<std::size_t>(i)] = (i * 7 + 3) & mask;
    }
    for (int i = 0; i < c.h; ++i) {
      left[static_cast<std::size_t>(i)] = (i * 5 + 11) & mask;
    }
    const ExpectedDc e = expected_dc(above, left, c.w, c.h, c.bd, c.above,
                                     c.left);
    std::printf("    {%d, %d, %d, %s, %s, %d, %d, %lld, %d, %d},\n", c.w,
                c.h, c.bd, c.above ? "true" : "false",
                c.left ? "true" : "false", e.step_w, e.step_h, e.sum, e.count,
                e.avg);
  }

  std::printf("=== corner goldens ===\n");
  const int corner_cases[][3] = {
      {0, 0, 0},       {255, 255, 255}, {10, 30, 200},
      {127, 128, 129}, {1, 2, 3},       {64, 192, 96},
  };
  for (const auto& c : corner_cases) {
    std::printf("    {%d, %d, %d, %d},\n", c[0], c[1], c[2],
                mb::filter_corner_value(c[0], c[1], c[2]));
  }

  std::printf("=== dc_sign_horz_vert entropy ===\n");
  const int signs[10] = {0, 1, 1, 0, 1, 0, 0, 1, 1, 1};
  for (int frozen = 0; frozen <= 1; ++frozen) {
    std::uint16_t cdf[2][2][3][3];
    std::memcpy(cdf, av2enc::m0::kDcSignCdf, sizeof(cdf));
    std::vector<std::uint8_t> buffer(256, 0);
    av2enc::av2::Encoder encoder;
    encoder.init_symbol(buffer.data(), buffer.size());
    encoder.disable_cdf_update = static_cast<std::uint8_t>(frozen);
    for (int sign : signs) {
      mb::write_dc_sign_horz_vert(
          encoder, mb::dc_sign_horz_vert_cdf_slot(cdf, 0, 0), sign);
    }
    std::size_t bytes = 0;
    encoder.exit_symbol(&bytes);
    std::printf("%s: \"%s\" (%zu bytes)\n",
                frozen ? "kSignFrozenHex" : "kSignAdaptiveHex",
                hex_of(buffer.data(), bytes).c_str(), bytes);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--dump-goldens") {
    dump_goldens();
    return 0;
  }

  test_dc_subsampled();
  test_filter_corner();
  test_dc_sign_horz_vert();
  test_save_subpu();

  if (g_failures != 0) {
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return 1;
  }
  std::printf("%d checks, 0 failures\n", g_checks);
  return 0;
}
