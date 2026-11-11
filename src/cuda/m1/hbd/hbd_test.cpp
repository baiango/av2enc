// CPU-only tests for the 10-bit M1 transform path and 4:2:0 geometry.
//
// These pin the bit-depth-dependent bounds and prove the 10-bit CPU reference
// reduces exactly to the existing 8-bit `tx_variants` reference at bit_depth=8.
// The CPU-vs-GPU half lives in hbd_test.cu (requires nvcc + sm_86).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda/m1/hbd/hbd_cpu.h"
#include "encoder/format/format.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

namespace hbd = av2enc::hbd;
namespace fmt = av2enc::fmt;
namespace txv = av2enc::txv;

int g_failures = 0;
long long g_checks = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,           \
                  #condition);                                               \
    }                                                                        \
    ++g_checks;                                                              \
  } while (false)

#define CHECK_EQ(actual, expected)                                           \
  do {                                                                       \
    const auto a_ = (actual);                                                \
    const auto e_ = (expected);                                              \
    if (!(a_ == e_)) {                                                       \
      ++g_failures;                                                          \
      std::printf("%s:%d: CHECK_EQ failed: %s (%lld) != %s (%lld)\n",        \
                  __FILE__, __LINE__, #actual,                               \
                  static_cast<long long>(a_), #expected,                     \
                  static_cast<long long>(e_));                               \
    }                                                                        \
    ++g_checks;                                                              \
  } while (false)

std::mt19937 g_rng(0x10B17u);

void test_bit_depth_bounds() {
  for (const int depth : {8, 10, 12}) {
    const int high = (1 << depth) - 1;
    CHECK_EQ(hbd::dc_pred_bias(depth), 1 << (depth - 1));
    CHECK_EQ(hbd::clip1(-1, depth), 0);
    CHECK_EQ(hbd::clip1(high + 1, depth), high);
    CHECK_EQ(hbd::clip_dequant(-(1 << (7 + depth)) - 1, depth),
             -(1 << (7 + depth)));
    CHECK_EQ(hbd::clip_dequant((1 << (7 + depth)), depth),
             (1 << (7 + depth)) - 1);
    CHECK_EQ(hbd::clip_inverse_row((1 << (depth + 7)), depth),
             (1 << (depth + 7)) - 1);
    CHECK_EQ(hbd::clip_inverse_col(-(1 << depth) - 1, depth),
             -(1 << depth));
    CHECK_EQ(hbd::clip_inverse_col((1 << depth), depth), (1 << depth) - 1);
  }
  // The 10-bit values the M1 owner must use, stated explicitly.
  CHECK_EQ(hbd::dc_pred_bias(10), 512);
  CHECK_EQ(hbd::clip1(1024, 10), 1023);
  CHECK_EQ(hbd::clip_inverse_col(1024, 10), 1023);
  CHECK_EQ(hbd::clip_dequant(1 << 20, 10), (1 << 17) - 1);

  // fmt:: helpers and the CUDA-side CPU reference must agree.
  for (const fmt::BitDepth d : {fmt::BitDepth::k8, fmt::BitDepth::k10,
                                fmt::BitDepth::k12}) {
    const int depth = fmt::bit_depth_value(d);
    const fmt::BitDepthLimits limits = fmt::bit_depth_limits(d);
    CHECK_EQ(limits.bit_depth, depth);
    CHECK_EQ(limits.dc_bias, hbd::dc_pred_bias(depth));
    CHECK_EQ(limits.dequant_min, hbd::clip_dequant(-(int64_t{1} << 40), depth));
    CHECK_EQ(limits.dequant_max, hbd::clip_dequant(int64_t{1} << 40, depth));
    CHECK_EQ(limits.inverse_row_min,
             hbd::clip_inverse_row(-(int64_t{1} << 40), depth));
    CHECK_EQ(limits.inverse_col_max,
             hbd::clip_inverse_col(int64_t{1} << 40, depth));
  }
}

void test_scan() {
  for (int w = 4; w <= 32; w *= 2) {
    for (int h = 4; h <= 32; h *= 2) {
      const int area = w * h;
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      hbd::build_scan(w, h, scan.data());
      std::vector<bool> seen(static_cast<std::size_t>(area), false);
      for (int c = 0; c < area; ++c) {
        CHECK_EQ(hbd::scan_pos(c, w, h),
                 static_cast<int>(scan[static_cast<std::size_t>(c)]));
        const int pos = scan[static_cast<std::size_t>(c)];
        CHECK(pos >= 0 && pos < area);
        if (pos >= 0 && pos < area) {
          CHECK(!seen[static_cast<std::size_t>(pos)]);
          seen[static_cast<std::size_t>(pos)] = true;
        }
      }
      // Every raster position appears exactly once.
      CHECK(std::all_of(seen.begin(), seen.end(), [](bool v) { return v; }));
    }
  }

  // EOB is the 1-based scan index of the last non-zero coefficient.
  const int w = 4;
  const int h = 4;
  std::vector<std::int32_t> levels(static_cast<std::size_t>(w * h), 0);
  levels[0] = 3;
  CHECK_EQ(hbd::eob_from_levels(levels.data(), w, h), 1);
  std::vector<std::uint16_t> scan(16);
  hbd::build_scan(w, h, scan.data());
  levels[scan[9]] = -7;
  CHECK_EQ(hbd::eob_from_levels(levels.data(), w, h), 10);
}

// At bit_depth 8, hbd::dequantize/inverse_block must be byte-identical to the
// existing tx_variants reference.  This is the proof that the 10-bit code is
// an extension, not a divergent copy.
void test_eight_bit_reduces_to_txv() {
  std::uniform_int_distribution<int> level_dist(-400, 400);
  const int sizes[][2] = {{4, 4},  {8, 8},   {16, 16}, {32, 32}, {4, 16},
                          {16, 4}, {8, 32}, {32, 8},  {16, 64}, {64, 16},
                          {64, 64}, {8, 16}, {16, 8}};
  for (const auto& size : sizes) {
    const int w = size[0];
    const int h = size[1];
    for (const txv::TxType type :
         {txv::TxType::kDctDct, txv::TxType::kAdstAdst,
          txv::TxType::kIdtx, txv::TxType::kDctAdst}) {
      const auto params = txv::make_tx_params(w, h, type);
      CHECK(params.has_value());
      if (!params) {
        continue;
      }
      const int area = params->w * params->h;
      const int coded = params->coded_w * params->coded_h;
      std::vector<std::int32_t> levels(static_cast<std::size_t>(area));
      for (std::int32_t& value : levels) {
        value = level_dist(g_rng);
      }

      std::vector<std::int32_t> expected_dequant(static_cast<std::size_t>(area));
      std::vector<std::int32_t> actual_dequant(static_cast<std::size_t>(area));
      txv::dequantize(levels.data(), *params, expected_dequant.data());
      hbd::dequantize(levels.data(), *params, actual_dequant.data(), 8);
      CHECK(expected_dequant == actual_dequant);

      std::vector<std::int16_t> expected_residual(static_cast<std::size_t>(coded));
      std::vector<std::int16_t> actual_residual(static_cast<std::size_t>(coded));
      txv::inverse_block(expected_dequant.data(), *params,
                         expected_residual.data());
      hbd::inverse_block(actual_dequant.data(), *params,
                         actual_residual.data(), 8);
      CHECK(expected_residual == actual_residual);
    }
  }
}

void test_ten_bit_clips_apply() {
  // A 4x4 with a very large DC level: 10-bit must clip dequantization to
  // ±(1<<17) and the column pass to ±1023, while 8-bit clips much lower.
  const auto params = txv::make_tx_params(4, 4, txv::TxType::kDctDct);
  CHECK(params.has_value());
  if (!params) return;
  const int area = params->w * params->h;
  std::vector<std::int32_t> levels(static_cast<std::size_t>(area), 0);
  levels[0] = 1000000;
  std::vector<std::int32_t> ten_dequant(static_cast<std::size_t>(area));
  hbd::dequantize(levels.data(), *params, ten_dequant.data(), 10);
  CHECK_EQ(ten_dequant[0], (1 << 17) - 1);

  std::vector<std::int32_t> eight_dequant(static_cast<std::size_t>(area));
  hbd::dequantize(levels.data(), *params, eight_dequant.data(), 8);
  CHECK_EQ(eight_dequant[0], (1 << 15) - 1);

  std::vector<std::int16_t> ten_residual(
      static_cast<std::size_t>(params->coded_w * params->coded_h));
  hbd::inverse_block(ten_dequant.data(), *params, ten_residual.data(), 10);
  for (const std::int16_t value : ten_residual) {
    CHECK(value >= -1024 && value <= 1023);
  }
  std::vector<std::int16_t> eight_residual(ten_residual.size());
  hbd::inverse_block(eight_dequant.data(), *params, eight_residual.data(), 8);
  for (const std::int16_t value : eight_residual) {
    CHECK(value >= -256 && value <= 255);
  }
}

void test_encode_block() {
  // 64x64 10-bit block exercising the adjusted 32x32 path and edge clamping.
  const int cw = 64;
  const int ch = 64;
  const int src_w = 70;
  const int src_h = 66;
  std::vector<std::uint16_t> src(static_cast<std::size_t>(src_w) * src_h);
  std::uniform_int_distribution<int> sample_dist(0, 1023);
  for (std::uint16_t& value : src) {
    value = static_cast<std::uint16_t>(sample_dist(g_rng));
  }
  std::vector<std::uint16_t> pred(static_cast<std::size_t>(cw * ch));
  for (std::uint16_t& value : pred) {
    value = static_cast<std::uint16_t>(sample_dist(g_rng));
  }
  const auto params = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
  CHECK(params.has_value());
  if (!params) return;

  hbd::BlockResult result;
  CHECK(hbd::encode_block(src.data(), src_w, src_w, src_h, 3, 2, pred.data(),
                          *params, 10, &result));
  CHECK_EQ(result.levels.size(),
           static_cast<std::size_t>(params->w * params->h));
  CHECK_EQ(result.residual.size(), static_cast<std::size_t>(cw * ch));
  CHECK_EQ(result.recon.size(), static_cast<std::size_t>(cw * ch));
  CHECK(result.eob >= 0 && result.eob <= params->w * params->h);
  for (const std::uint16_t value : result.recon) {
    CHECK(value <= 1023);
  }

  // Zero residual: eob 0, residual 0, recon == prediction.
  std::vector<std::uint16_t> zero_pred(static_cast<std::size_t>(cw * ch), 512);
  const std::vector<std::int16_t> zero_residual(
      static_cast<std::size_t>(cw * ch), 0);
  hbd::BlockResult zero;
  CHECK(hbd::encode_block_residual(zero_residual.data(), zero_pred.data(),
                                   *params, 10, &zero));
  CHECK_EQ(zero.eob, 0);
  CHECK(zero.recon == zero_pred);
  CHECK(std::all_of(zero.residual.begin(), zero.residual.end(),
                    [](std::int16_t v) { return v == 0; }));

  // Prediction at the top of the range stays clipped, never wrap to 0.
  std::vector<std::uint16_t> high_pred(static_cast<std::size_t>(cw * ch), 1023);
  const std::vector<std::int16_t> high_residual(
      static_cast<std::size_t>(cw * ch), 400);
  hbd::BlockResult high;
  CHECK(hbd::encode_block_residual(high_residual.data(), high_pred.data(),
                                   *params, 10, &high));
  for (const std::uint16_t value : high.recon) {
    CHECK(value == 1023);
  }

  // Direct residual mode handles the smallest block (no adjusted downscale).
  const auto tiny = txv::make_tx_params(4, 4, txv::TxType::kDctDct);
  CHECK(tiny.has_value());
  if (tiny) {
    std::vector<std::int16_t> residual(16, 5);
    std::vector<std::uint16_t> tiny_pred(16, 1000);
    hbd::BlockResult tiny_result;
    CHECK(hbd::encode_block_residual(residual.data(), tiny_pred.data(), *tiny,
                                     10, &tiny_result));
    for (const std::uint16_t value : tiny_result.recon) {
      // Lossy quantize of a flat 5 reconstructs within a few LSBs of 1005;
      // the point is that 10-bit arithmetic does not wrap at 255.
      CHECK(value >= 997 && value <= 1013);
    }
  }
}

void test_420_geometry_and_conversion() {
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (const auto& dims : {std::pair<int, int>{64, 64},
                           std::pair<int, int>{640, 360},
                           std::pair<int, int>{641, 361},
                           std::pair<int, int>{67, 43}}) {
    const int width = dims.first;
    const int height = dims.second;
    const int chroma_w = (width + 1) / 2;
    const int chroma_h = (height + 1) / 2;
    std::vector<std::uint8_t> y(static_cast<std::size_t>(width) *
                               static_cast<std::size_t>(height));
    std::vector<std::uint8_t> u(static_cast<std::size_t>(chroma_w) *
                               static_cast<std::size_t>(chroma_h));
    std::vector<std::uint8_t> v(u.size());
    for (std::uint8_t& value : y) value = static_cast<std::uint8_t>(byte_dist(g_rng));
    for (std::uint8_t& value : u) value = static_cast<std::uint8_t>(byte_dist(g_rng));
    for (std::uint8_t& value : v) value = static_cast<std::uint8_t>(byte_dist(g_rng));

    fmt::Source420_8 source;
    source.width = static_cast<std::uint32_t>(width);
    source.height = static_cast<std::uint32_t>(height);
    source.y = {y.data(), static_cast<std::uint32_t>(width)};
    source.u = {u.data(), static_cast<std::uint32_t>(chroma_w)};
    source.v = {v.data(), static_cast<std::uint32_t>(chroma_w)};

    fmt::Picture picture;
    const fmt::ConvertResult converted = fmt::convert_420_8_to_picture(
        source, fmt::ChromaFormat::k420, fmt::BitDepth::k10, picture);
    CHECK(converted.ok());
    if (!converted.ok()) continue;

    CHECK_EQ(picture.format.depth, fmt::BitDepth::k10);
    const fmt::FrameGeometry& g = picture.geometry;
    CHECK_EQ(g.coded_width, 8U * static_cast<std::uint32_t>((width + 7) / 8));
    CHECK_EQ(g.coded_height, 8U * static_cast<std::uint32_t>((height + 7) / 8));
    CHECK_EQ(g.planes[1].visible_width, static_cast<std::uint32_t>(chroma_w));
    CHECK_EQ(g.planes[1].visible_height, static_cast<std::uint32_t>(chroma_h));

    // Luma is an exact left shift; padded samples replicate the last visible.
    bool luma_ok = true;
    for (int yy = 0; yy < height; ++yy) {
      for (int xx = 0; xx < width; ++xx) {
        luma_ok &= picture.planes[0].at(static_cast<std::uint32_t>(xx),
                                       static_cast<std::uint32_t>(yy)) ==
                   static_cast<std::uint16_t>(
                       y[static_cast<std::size_t>(yy) *
                             static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(xx)]
                       << 2);
      }
    }
    CHECK(luma_ok);
    CHECK(picture.planes[0].at(g.planes[0].coded_width - 1,
                               g.planes[0].coded_height - 1) ==
          static_cast<std::uint16_t>(
              y[static_cast<std::size_t>(height - 1) *
                    static_cast<std::size_t>(width) +
                static_cast<std::size_t>(width - 1)]
              << 2));
    CHECK(fmt::samples_in_range(picture));
  }
}

}  // namespace

int main() {
  test_bit_depth_bounds();
  test_scan();
  test_eight_bit_reduces_to_txv();
  test_ten_bit_clips_apply();
  test_encode_block();
  test_420_geometry_and_conversion();

  if (g_failures != 0) {
    std::printf("hbd_test: %d failure(s), %lld checks\n", g_failures, g_checks);
    return 1;
  }
  std::printf("hbd_test: PASS (%lld checks)\n", g_checks);
  return 0;
}
