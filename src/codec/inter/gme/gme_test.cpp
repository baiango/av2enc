// CPU correctness tests for AV2 global motion estimation.
//
// Coverage:
//   * §5.18.9 syntax round-trip (types, per-reference params, base model,
//     differential coding, quantisation grid).
//   * §7.12.2.2 projection / precision lowering / clamps.
//   * Deterministic affine / similarity / translation fitting from point
//     correspondences, per-block warps and block MVs.
//   * Deterministic outlier rejection, degradation and validation.
//   * Synthetic pan / zoom / rotation, both from exact correspondences and
//     through a real block matcher on synthetic frames.
//   * Real 720p pairs from testdata/shine_1280x720_60f.y4m.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gme.h"
#include "gme_syntax.h"
#include "warp_common.h"  // test-only parity check for the shear inequalities

namespace {

using namespace av2enc::inter::gme;
using av2enc::BitWriter;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) std::printf("FAIL line %d: %s\n", line, what);
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)
#define EXPECT_EQ(a, b)                                                                 \
  do {                                                                                  \
    const long long va = static_cast<long long>(a);                                     \
    const long long vb = static_cast<long long>(b);                                     \
    ++g_checks;                                                                         \
    if (va != vb) {                                                                     \
      ++g_failures;                                                                     \
      if (g_failures <= 40) {                                                           \
        std::printf("FAIL line %d: %s (%lld) != %s (%lld)\n", __LINE__, #a, va, #b, vb); \
      }                                                                                 \
    }                                                                                   \
  } while (0)

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

std::int32_t q16(double value) {
  return static_cast<std::int32_t>(std::lround(value * 65536.0));
}

GmParams make_similarity(double scale, double angle_deg, double tx, double ty) {
  const double radians = angle_deg * 3.14159265358979323846 / 180.0;
  const double c = std::cos(radians) * scale;
  const double s = std::sin(radians) * scale;
  GmParams params;
  params.type = GmType::kRotZoom;
  params.p[0] = q16(tx);
  params.p[1] = q16(ty);
  params.p[2] = q16(c);
  params.p[3] = q16(-s);
  params.p[4] = q16(s);
  params.p[5] = q16(c);
  return params;
}

GmParams make_affine(double p2, double p3, double p4, double p5, double tx, double ty) {
  GmParams params;
  params.type = GmType::kAffine;
  params.p[0] = q16(tx);
  params.p[1] = q16(ty);
  params.p[2] = q16(p2);
  params.p[3] = q16(p3);
  params.p[4] = q16(p4);
  params.p[5] = q16(p5);
  return params;
}

void project_q8(const GmParams& params, int x_q8, int y_q8, int* rx_q8, int* ry_q8) {
  std::int32_t rx = 0;
  std::int32_t ry = 0;
  project_gm_point(params, x_q8 << 13, y_q8 << 13, &rx, &ry);
  *rx_q8 = rx >> 13;
  *ry_q8 = ry >> 13;
}

std::vector<GmPoint> grid_points(const GmParams& params, int width, int height, int step_px,
                                 int noise_seed, int noise_amplitude) {
  std::vector<GmPoint> points;
  std::uint32_t seed = static_cast<std::uint32_t>(noise_seed) * 2654435761U + 1U;
  auto next = [&]() {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
  };
  for (int y = 8; y < height - 8; y += step_px) {
    for (int x = 8; x < width - 8; x += step_px) {
      const int sx = x * 8;
      const int sy = y * 8;
      int rx = 0;
      int ry = 0;
      project_q8(params, sx, sy, &rx, &ry);
      if (noise_amplitude > 0) {
        rx += static_cast<int>(next() % static_cast<std::uint32_t>(2 * noise_amplitude + 1)) -
              noise_amplitude;
        ry += static_cast<int>(next() % static_cast<std::uint32_t>(2 * noise_amplitude + 1)) -
              noise_amplitude;
      }
      GmPoint point;
      point.src_x = sx;
      point.src_y = sy;
      point.dst_x = rx;
      point.dst_y = ry;
      point.weight = 1;
      points.push_back(point);
    }
  }
  return points;
}

void check_model_close(const GmParams& got, const GmParams& want, int coeff_tol, int trans_tol,
                       const char* what) {
  const bool close = got.type == want.type &&
                     std::abs(got.p[0] - want.p[0]) <= trans_tol &&
                     std::abs(got.p[1] - want.p[1]) <= trans_tol &&
                     std::abs(got.p[2] - want.p[2]) <= coeff_tol &&
                     std::abs(got.p[3] - want.p[3]) <= coeff_tol &&
                     std::abs(got.p[4] - want.p[4]) <= coeff_tol &&
                     std::abs(got.p[5] - want.p[5]) <= coeff_tol;
  ++g_checks;
  if (!close) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf(
          "FAIL %s: got type=%s p=[%d %d %d %d %d %d] want type=%s "
          "p=[%d %d %d %d %d %d]\n",
          what, gm_type_name(got.type), got.p[0], got.p[1], got.p[2], got.p[3], got.p[4],
          got.p[5], gm_type_name(want.type), want.p[0], want.p[1], want.p[2], want.p[3],
          want.p[4], want.p[5]);
    }
  }
}

// ---------------------------------------------------------------------------
// Syntax.
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> write_bits(const std::function<void(BitWriter&)>& fn,
                                     std::size_t* bit_count = nullptr) {
  BitWriter writer;
  fn(writer);
  if (bit_count != nullptr) *bit_count = writer.bit_size();
  return std::vector<std::uint8_t>(writer.bytes().begin(), writer.bytes().end());
}

void test_sequence_bit() {
  std::size_t bits = 0;
  const std::vector<std::uint8_t> bytes = write_bits(
      [](BitWriter& writer) {
        writer.put_bit(true);   // enable_flex_mvres
        write_sequence_enable_global_motion(writer, true);
        writer.put_bit(false);  // enable_short_refresh_frame_flags
      },
      &bits);
  EXPECT_EQ(bits, 3U);
  EXPECT_EQ(bytes[0], 0xC0U);  // 110xxxxx
  GmBitReader reader(bytes);
  bool a = false;
  bool b = false;
  bool c = false;
  EXPECT(reader.read_bit(&a));
  EXPECT(reader.read_bit(&b));
  EXPECT(reader.read_bit(&c));
  EXPECT(a && b && !c);
}

void test_global_param_roundtrip() {
  GmParams params;
  params.type = GmType::kAffine;
  for (int index = 0; index < 6; ++index) {
    const int step = index < 2 ? kGmTransStep : kGmAlphaStep;
    const int max_offset = index < 2 ? kGmTransMax : kGmAlphaMax;
    const int center = (index == 2 || index == 5) ? kGmOne : 0;
    const int half_offset = (max_offset / 2) * step;
    const int values[] = {-max_offset * step, -half_offset, -step, 0, step,
                          half_offset, max_offset * step};
    for (int value : values) {
      const std::int32_t target = static_cast<std::int32_t>(center + value);
      std::int32_t prev[6] = {0, 0, kGmOne, 0, 0, kGmOne};
      std::int32_t decoded[6] = {0, 0, kGmOne, 0, 0, kGmOne};
      GmParams local = params;
      local.p[static_cast<std::size_t>(index)] = target;
      BitWriter writer;
      std::string error;
      if (!write_global_param(writer, 0, index, prev, local, &error)) {
        std::printf("write_global_param failed: %s\n", error.c_str());
        EXPECT(false);
        continue;
      }
      GmBitReader reader(writer.bytes());
      EXPECT(read_global_param(reader, 0, index, prev, decoded, &error));
      EXPECT_EQ(decoded[index], target);
      EXPECT_EQ(reader.bit_pos(), writer.bit_size());
    }
  }
}

void test_global_param_prev_sweep() {
  // Exercise the reference predictor with non-zero deltas and negative r:
  // the gate's first frame always uses r = 0 (default model), so this is the
  // unit-level coverage for the recenter/subexp paths.
  for (int index = 0; index < 6; ++index) {
    const int step = index < 2 ? kGmTransStep : kGmAlphaStep;
    const int max_offset = index < 2 ? kGmTransMax : kGmAlphaMax;
    const int center = (index == 2 || index == 5) ? kGmOne : 0;
    const int low = center - max_offset * step;
    const int high = center + max_offset * step;
    const int prev_values[] = {low, low / 3, low / 2, -step, 0, step, high / 2, high / 3,
                               high};
    for (int prev_value : prev_values) {
      std::int32_t prev[6] = {0, 0, kGmOne, 0, 0, kGmOne};
      prev[index] = prev_value;
      for (int k = 0; k <= 8; ++k) {
        const int target =
            center + (-max_offset + (2 * max_offset) * k / 8) * step;
        GmParams params;
        params.type = GmType::kAffine;
        params.p[static_cast<std::size_t>(index)] = target;
        BitWriter writer;
        std::string error;
        if (!write_global_param(writer, 0, index, prev, params, &error)) {
          std::printf("prev-sweep write failed index=%d prev=%d target=%d: %s\n",
                      index, prev_value, target, error.c_str());
          EXPECT(false);
          continue;
        }
        std::int32_t decoded[6] = {0, 0, kGmOne, 0, 0, kGmOne};
        GmBitReader reader(writer.bytes());
        EXPECT(read_global_param(reader, 0, index, prev, decoded, &error));
        EXPECT_EQ(decoded[index], target);
        EXPECT_EQ(reader.bit_pos(), writer.bit_size());
      }
    }
  }
}

GmSyntaxContext make_syntax_context() {
  GmSyntaxContext context;
  context.enable_global_motion = true;
  context.use_global_motion = true;
  context.num_total_refs = 3;
  context.order_hint = 4;
  context.ref_order_hint = {0, 2, 3};
  context.ref_frame_idx = {0, 1, 2};
  context.params[0] = make_identity_gm();
  context.params[1] = make_similarity(1.03, 1.7, 1.25, -0.75);
  context.params[2] = make_affine(1.03, 0.02, -0.03, 0.94, -2.5, 3.0);
  GmOptions options;
  context.params[1] = finalize_gm_params(context.params[1], options).params;
  context.params[2] = finalize_gm_params(context.params[2], options).params;
  return context;
}

void test_global_motion_params_roundtrip() {
  GmSyntaxContext context = make_syntax_context();
  BitWriter writer;
  std::string error;
  const std::size_t bits = write_global_motion_params(writer, context, &error);
  EXPECT(bits > 0U);
  EXPECT(error.empty());

  GmBitReader reader(writer.bytes());
  std::array<GmParams, kMaxRefsPerFrame> decoded{};
  bool use = false;
  EXPECT(read_global_motion_params(reader, context, &decoded, &error, &use));
  EXPECT(use);
  EXPECT_EQ(reader.bit_pos(), bits);
  for (int ref = 0; ref < 3; ++ref) {
    EXPECT(decoded[static_cast<std::size_t>(ref)] ==
           context.params[static_cast<std::size_t>(ref)]);
  }
  EXPECT_EQ(decoded[2].type, GmType::kAffine);

  // use_global_motion = 0 writes one bit and leaves every reference identity.
  GmSyntaxContext off = context;
  off.use_global_motion = false;
  for (auto& param : off.params) param = make_identity_gm();
  BitWriter off_writer;
  const std::size_t off_bits = write_global_motion_params(off_writer, off, &error);
  EXPECT_EQ(off_bits, 1U);
  GmBitReader off_reader(off_writer.bytes());
  std::array<GmParams, kMaxRefsPerFrame> off_decoded{};
  bool off_use = true;
  EXPECT(read_global_motion_params(off_reader, off, &off_decoded, &error, &off_use));
  EXPECT(!off_use);
  for (const auto& param : off_decoded) EXPECT(is_identity_model(param));

  // Intra frames and disabled sequences write nothing.
  GmSyntaxContext intra = context;
  intra.frame_is_intra = true;
  BitWriter intra_writer;
  EXPECT_EQ(write_global_motion_params(intra_writer, intra, &error), 0U);
  GmSyntaxContext disabled = context;
  disabled.enable_global_motion = false;
  BitWriter disabled_writer;
  EXPECT_EQ(write_global_motion_params(disabled_writer, disabled, &error), 0U);
}

void test_global_motion_base_roundtrip() {
  GmSyntaxContext context;
  context.enable_global_motion = true;
  context.use_global_motion = true;
  context.num_total_refs = 2;
  context.order_hint = 5;
  context.ref_order_hint = {2, 4};
  context.ref_frame_idx = {7, 9};
  // Base model from saved reference slot 7, their_ref 0, distance 2 (hint 2
  // saved vs current hint 4 of our_ref).
  context.our_ref = 0;
  context.their_ref = 0;
  const int ref_slot = 7;
  context.history.ref_num_total_refs[static_cast<std::size_t>(ref_slot)] = 1;
  context.history.saved_order_hints[static_cast<std::size_t>(ref_slot)][0] = 2;
  const GmParams saved = make_similarity(1.01, 0.8, 0.5, 0.25);
  for (int i = 0; i < 6; ++i) {
    context.history.saved_gm_params[static_cast<std::size_t>(ref_slot)][0]
                                  [static_cast<std::size_t>(i)] = saved.p[static_cast<std::size_t>(i)];
  }
  GmOptions options;
  context.params[0] = finalize_gm_params(make_similarity(1.02, 1.1, 0.75, 0.5), options).params;
  context.params[1] = finalize_gm_params(make_similarity(1.04, 1.5, 1.0, 0.75), options).params;

  BitWriter writer;
  std::string error;
  const std::size_t bits = write_global_motion_params(writer, context, &error);
  EXPECT(bits > 0U);
  GmBitReader reader(writer.bytes());
  std::array<GmParams, kMaxRefsPerFrame> decoded{};
  EXPECT(read_global_motion_params(reader, context, &decoded, &error));
  EXPECT_EQ(reader.bit_pos(), bits);
  EXPECT(decoded[0] == context.params[0]);
  EXPECT(decoded[1] == context.params[1]);
}

void test_scale_warp_model() {
  // Values on the syntax grids so equal distances round-trip exactly.
  std::int32_t base[6] = {8192, -16384, kGmOne + 4096, 2048, -1024, kGmOne - 4096};
  std::int32_t out[6];
  scale_warp_model(base, 2, 2, out);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], base[i]);
  scale_warp_model(base, 2, -2, out);
  for (int i = 0; i < 6; ++i) {
    const int center = (i == 2 || i == 5) ? kGmOne : 0;
    EXPECT_EQ(out[i], center - (base[i] - center));
  }
  scale_warp_model(base, 2, 4, out);
  EXPECT(out[2] > base[2]);
  EXPECT(out[5] < base[5]);
  // baseDistance 0 returns Default_Warp_Params.
  scale_warp_model(base, 0, 3, out);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(out[i], kDefaultWarpParams[i]);
}

// ---------------------------------------------------------------------------
// Projection / validation.
// ---------------------------------------------------------------------------

void test_projection() {
  GmParams identity = make_identity_gm();
  int mv[2] = {123, 456};
  get_warp_motion_vector(identity, 10, 5, 16, 16, 100, 100, MvPrecision::kEighthPel, mv);
  EXPECT_EQ(mv[0], 0);
  EXPECT_EQ(mv[1], 0);

  GmParams translation;
  translation.type = GmType::kRotZoom;
  translation.p[0] = -2 * 65536;  // 2 px left (column)
  translation.p[1] = 3 * 65536;   // 3 px down (row)
  get_warp_motion_vector(translation, 10, 5, 16, 16, 100, 100, MvPrecision::kEighthPel, mv);
  EXPECT_EQ(mv[0], 24);
  EXPECT_EQ(mv[1], -16);
  get_warp_motion_vector(translation, 10, 5, 16, 16, 100, 100, MvPrecision::kQuarterPel, mv);
  EXPECT_EQ(mv[0], 24);
  EXPECT_EQ(mv[1], -16);
  get_warp_motion_vector(translation, 10, 5, 16, 16, 100, 100, MvPrecision::kOnePel, mv);
  EXPECT_EQ(mv[0], 24);
  EXPECT_EQ(mv[1], -16);

  const int x = 10 * 4 + 8 - 1;
  const int y = 5 * 4 + 8 - 1;
  std::int32_t rx = 0;
  std::int32_t ry = 0;
  project_gm_point(translation, x << 16, y << 16, &rx, &ry);
  EXPECT_EQ(rx, (x - 2) << 16);
  EXPECT_EQ(ry, (y + 3) << 16);

  GmGlobalMvUse use = global_mv_for_block(translation, 1, 10, 5, 16, 16, 100, 100);
  EXPECT(use.usable);
  EXPECT(use.use_warp);
  EXPECT_EQ(use.mv[0], 24);
  EXPECT_EQ(use.mv[1], -16);
  EXPECT(!global_mv_for_block(identity, 1, 10, 5, 16, 16, 100, 100).usable);
}

void test_shear_parity() {
  // §7.13.3.21 transcription parity against the advanced-inter implementation.
  std::uint32_t seed = 0x9e3779b9U;
  auto next = [&]() {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
  };
  for (int n = 0; n < 20000; ++n) {
    const int p2 = kGmOne + static_cast<int>(next() % (2U * 32705U)) - 32704;
    const int p5 = kGmOne + static_cast<int>(next() % (2U * 32705U)) - 32704;
    const int p3 = static_cast<int>(next() % (2U * 32705U)) - 32704;
    const int p4 = static_cast<int>(next() % (2U * 32705U)) - 32704;
    int params[6] = {0, 0, p2, p3, p4, p5};
    av2enc::inter::adv::WarpParams warp;
    for (int i = 0; i < 6; ++i) warp.p[i] = params[i];
    const bool expected = av2enc::inter::adv::setup_shear(warp).valid;
    EXPECT_EQ(setup_shear_valid(params), expected);
  }
}

void test_validation_and_degradation() {
  GmOptions options;
  GmParams bad = make_identity_gm();
  bad.p[0] = 1;
  EXPECT(!gm_params_codable(bad, nullptr));
  bad.p[0] = kGmTransStep;
  bad.type = GmType::kRotZoom;
  EXPECT(gm_params_codable(bad, nullptr));
  GmParams out_of_range;
  out_of_range.p[0] = (kGmTransMax + 1) * kGmTransStep;
  EXPECT(!gm_params_codable(out_of_range, nullptr));
  GmParams asym = make_similarity(1.02, 1.0, 0.0, 0.0);
  asym.p[4] += kGmAlphaStep;
  EXPECT(!gm_params_codable(asym, nullptr));

  // An affine that is almost a similarity degrades to ROTZOOM.
  GmEstimate estimate = finalize_gm_params(make_affine(1.02, 0.01, -0.009, 1.0195, 0.5, 0.5),
                                           options);
  EXPECT(estimate.diag.accepted);
  EXPECT_EQ(estimate.params.type, GmType::kRotZoom);
  EXPECT(estimate.diag.degraded);
  EXPECT(gm_params_codable(estimate.params, nullptr));
  EXPECT(!estimate.diag.identity);

  // A reflection (negative determinant) is rejected to IDENTITY.
  GmOptions strict_wild = options;
  strict_wild.require_shear_valid = true;
  GmEstimate wild =
      finalize_gm_params(make_affine(1.0, 2.0, 2.0, 1.0, 0.0, 0.0), strict_wild);
  EXPECT(wild.diag.accepted);
  EXPECT(wild.diag.identity);
  EXPECT(is_identity_model(wild.params));

  // force_integer_mv snaps translation to whole pixels.
  GmOptions integer_options = options;
  integer_options.force_integer_mv = true;
  GmParams fractional = make_similarity(1.0, 0.0, 1.37, -2.62);
  GmEstimate integer_estimate = finalize_gm_params(fractional, integer_options);
  EXPECT_EQ(integer_estimate.params.p[0] % kGmOne, 0);
  EXPECT_EQ(integer_estimate.params.p[1] % kGmOne, 0);
  GmParams huge = make_similarity(1.0, 0.0, 3000.0, -3000.0);
  GmEstimate huge_estimate = finalize_gm_params(huge, integer_options);
  EXPECT(huge_estimate.diag.codable);
  EXPECT_EQ(huge_estimate.params.p[0] % kGmOne, 0);
  EXPECT_EQ(huge_estimate.params.p[1] % kGmOne, 0);

  // Shear-required profile rejects a model that the decoder would predict
  // with the extended-warp fallback.
  GmParams shear_bad = make_affine(1.0, 0.45, 0.0, 1.0, 0.0, 0.0);
  GmDiagnostics diag;
  GmOptions strict = options;
  strict.require_shear_valid = true;
  EXPECT(!validate_gm_params(shear_bad, strict, &diag));
  EXPECT(!diag.shear_valid);
}

void test_frame_plan() {
  GmOptions options;
  GmEstimate still = estimate_gm_points(grid_points(make_identity_gm(), 320, 240, 32, 1, 0),
                                        options);
  GmEstimate moving =
      estimate_gm_points(grid_points(make_similarity(1.02, 0.5, 2.0, -1.0), 320, 240, 32, 2, 0),
                         options);
  EXPECT(still.diag.accepted);
  EXPECT(moving.diag.accepted);
  std::vector<GmEstimate> estimates = {still, moving};
  GmFramePlan plan = make_gm_frame_plan(estimates, true);
  EXPECT(plan.sequence_enable_global_motion);
  EXPECT(plan.use_global_motion);
  EXPECT_EQ(plan.num_total_refs, 2);
  EXPECT(is_identity_model(plan.params[0]));
  EXPECT(!is_identity_model(plan.params[1]));

  GmFramePlan disabled = make_gm_frame_plan(estimates, false);
  EXPECT(!disabled.sequence_enable_global_motion);
  EXPECT(!disabled.use_global_motion);
  EXPECT(is_identity_model(disabled.params[0]));
  EXPECT(is_identity_model(disabled.params[1]));
}

// ---------------------------------------------------------------------------
// Fitting.
// ---------------------------------------------------------------------------

void test_fit_exact() {
  GmOptions options;
  options.frame_width = 640;
  options.frame_height = 360;

  const GmParams pan = make_similarity(1.0, 0.0, 7.5, -3.25);
  std::vector<GmPoint> pan_points = grid_points(pan, 640, 360, 40, 1, 0);
  GmEstimate pan_estimate = estimate_gm_points(pan_points, options);
  EXPECT(pan_estimate.diag.ok);
  EXPECT(pan_estimate.diag.accepted);
  EXPECT_EQ(pan_estimate.params.type, GmType::kRotZoom);
  check_model_close(pan_estimate.params, finalize_gm_params(pan, options).params, 64, 2 * kGmTransStep,
                    "pan");

  const GmParams zoom = make_similarity(1.07, 0.0, 0.0, 0.0);
  std::vector<GmPoint> zoom_points = grid_points(zoom, 640, 360, 32, 2, 0);
  GmEstimate zoom_estimate = estimate_gm_points(zoom_points, options);
  EXPECT(zoom_estimate.diag.ok);
  EXPECT(zoom_estimate.diag.accepted);
  check_model_close(zoom_estimate.params, finalize_gm_params(zoom, options).params, 256, 4 * kGmTransStep,
                    "zoom");

  const GmParams rot = make_similarity(1.0, 3.0, 0.0, 0.0);
  std::vector<GmPoint> rot_points = grid_points(rot, 640, 360, 24, 3, 0);
  GmEstimate rot_estimate = estimate_gm_points(rot_points, options);
  EXPECT(rot_estimate.diag.ok);
  EXPECT(rot_estimate.diag.accepted);
  check_model_close(rot_estimate.params, finalize_gm_params(rot, options).params, 256, 4 * kGmTransStep,
                    "rotation");

  const GmParams affine = make_affine(1.03, 0.02, -0.02, 0.94, 2.5, -1.5);
  std::vector<GmPoint> affine_points = grid_points(affine, 640, 360, 24, 4, 0);
  GmEstimate affine_estimate = estimate_gm_points(affine_points, options);
  EXPECT(affine_estimate.diag.ok);
  EXPECT(affine_estimate.diag.accepted);
  EXPECT_EQ(affine_estimate.params.type, GmType::kAffine);
  check_model_close(affine_estimate.params, finalize_gm_params(affine, options).params, 512, 8 * kGmTransStep,
                    "affine");
}

void test_fit_noise_and_outliers() {
  GmOptions options;
  options.frame_width = 640;
  options.frame_height = 360;
  options.outlier_passes = 3;
  options.outlier_threshold_q8 = 16;

  const GmParams truth = make_similarity(1.03, -1.5, 3.0, -2.0);
  std::vector<GmPoint> points = grid_points(truth, 640, 360, 24, 5, 1);
  // 10% gross outliers.
  for (std::size_t i = 0; i < points.size(); i += 10) {
    points[i].dst_x += 80;
    points[i].dst_y -= 60;
  }
  GmEstimate estimate = estimate_gm_points(points, options);
  EXPECT(estimate.diag.ok);
  EXPECT(estimate.diag.accepted);
  EXPECT(estimate.diag.outlier_passes_run > 0);
  check_model_close(estimate.params, finalize_gm_params(truth, options).params, 512, 8 * kGmTransStep,
                    "noise+outliers");
  EXPECT(estimate.diag.model_sse_q8 < estimate.diag.identity_sse_q8);
}

void test_fit_from_warps_and_mvs() {
  GmOptions options;
  options.frame_width = 640;
  options.frame_height = 360;
  const GmParams truth = make_similarity(1.02, 2.0, 1.0, -1.0);
  std::vector<GmWarpSample> warps;
  for (int y = 32; y < 360; y += 64) {
    for (int x = 32; x < 640; x += 64) {
      GmWarpSample warp;
      warp.x = x * 8;
      warp.y = y * 8;
      warp.radius = 32;
      warp.model = truth;
      warp.weight = 4;
      warps.push_back(warp);
    }
  }
  GmEstimate warp_estimate = estimate_gm_warps(warps, options);
  EXPECT(warp_estimate.diag.ok);
  EXPECT(warp_estimate.diag.accepted);
  check_model_close(warp_estimate.params, finalize_gm_params(truth, options).params, 256, 4 * kGmTransStep,
                    "warp fit");

  // Block MVs sampled from the same model.
  std::vector<GmBlockMv> blocks;
  for (int y = 32; y < 360; y += 32) {
    for (int x = 32; x < 640; x += 32) {
      int rx = 0;
      int ry = 0;
      project_q8(truth, x * 8, y * 8, &rx, &ry);
      GmBlockMv block;
      block.x = x * 8;
      block.y = y * 8;
      block.mv_x = rx - block.x;
      block.mv_y = ry - block.y;
      block.weight = 1;
      blocks.push_back(block);
    }
  }
  GmEstimate mv_estimate = estimate_gm_block_mvs(blocks, options);
  EXPECT(mv_estimate.diag.ok);
  EXPECT(mv_estimate.diag.accepted);
  check_model_close(mv_estimate.params, finalize_gm_params(truth, options).params, 256, 4 * kGmTransStep,
                    "block MV fit");
}

void test_determinism() {
  GmOptions options;
  options.frame_width = 640;
  options.frame_height = 360;
  options.outlier_passes = 3;
  const GmParams truth = make_affine(1.02, 0.01, -0.006, 0.99, 2.0, -1.0);
  std::vector<GmPoint> points = grid_points(truth, 640, 360, 24, 9, 1);
  for (std::size_t i = 0; i < points.size(); i += 13) {
    points[i].dst_x += 40;
  }
  const GmEstimate a = estimate_gm_points(points, options);
  // Reverse the input order: integer sums and the median are order-independent.
  std::reverse(points.begin(), points.end());
  const GmEstimate b = estimate_gm_points(points, options);
  EXPECT(a.params == b.params);
  EXPECT_EQ(a.diag.points_used, b.diag.points_used);
  EXPECT_EQ(a.diag.model_sse_q8, b.diag.model_sse_q8);
}

// ---------------------------------------------------------------------------
// Image helpers, synthetic frames and real 720p.
// ---------------------------------------------------------------------------

struct Plane {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> data;
  void init(int width, int height) {
    w = width;
    h = height;
    data.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  }
  std::uint8_t& at(int x, int y) {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)];
  }
};

std::uint8_t texture_sample(int x, int y) {
  // Smooth multi-frequency texture: enough 2-D structure for block matching
  // without the high-frequency aliasing that destroys scaled/rotated matches.
  const double fx = static_cast<double>(x);
  const double fy = static_cast<double>(y);
  const double value = 128.0 + 58.0 * std::sin(fx * 0.057) +
                       48.0 * std::cos(fy * 0.071) +
                       34.0 * std::sin((fx + fy) * 0.031) +
                       24.0 * std::cos((fx - fy) * 0.043) +
                       18.0 * std::sin(std::sqrt((fx - 320.0) * (fx - 320.0) +
                                                 (fy - 180.0) * (fy - 180.0)) *
                                       0.11);
  return static_cast<std::uint8_t>(std::clamp(static_cast<int>(std::lround(value)), 0, 255));
}

Plane make_texture(int w, int h) {
  Plane plane;
  plane.init(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) plane.at(x, y) = texture_sample(x, y);
  }
  return plane;
}

Plane warp_plane(const GmParams& params, const Plane& src, int w, int h) {
  Plane dst;
  dst.init(w, h);
  const bool ok = warp_plane_bilinear(params, src.data.data(), src.w, src.h, src.w,
                                      dst.data.data(), w, h, w, 8);
  EXPECT(ok);
  return dst;
}

std::int64_t sad_at(const Plane& ref, const Plane& cur, int bx, int by, int block, int dx,
                    int dy, int sample_step) {
  std::int64_t sad = 0;
  for (int y = 0; y < block; y += sample_step) {
    for (int x = 0; x < block; x += sample_step) {
      const int rx = bx + x + dx;
      const int ry = by + y + dy;
      sad += std::abs(static_cast<int>(cur.at(bx + x, by + y)) - static_cast<int>(ref.at(rx, ry)));
    }
  }
  return sad;
}

struct MatchOptions {
  int block = 16;
  int search = 16;
  int sample_step = 2;
  int min_variance = 24;
};

std::vector<GmBlockMv> match_blocks(const Plane& ref, const Plane& cur, const MatchOptions& options,
                                    std::uint64_t* blocks_considered = nullptr) {
  std::vector<GmBlockMv> out;
  std::uint64_t considered = 0;
  for (int by = 0; by + options.block <= cur.h; by += options.block) {
    for (int bx = 0; bx + options.block <= cur.w; bx += options.block) {
      // Texture gate on the current block.
      std::int64_t sum = 0;
      std::int64_t sum2 = 0;
      int count = 0;
      for (int y = 0; y < options.block; y += options.sample_step) {
        for (int x = 0; x < options.block; x += options.sample_step) {
          const int v = cur.at(bx + x, by + y);
          sum += v;
          sum2 += v * v;
          ++count;
        }
      }
      const double mean = static_cast<double>(sum) / count;
      const double variance =
          static_cast<double>(sum2) / count - mean * mean;
      if (variance < options.min_variance) continue;
      ++considered;

      std::int64_t best = std::numeric_limits<std::int64_t>::max();
      int best_dx = 0;
      int best_dy = 0;
      for (int dy = -options.search; dy <= options.search; ++dy) {
        for (int dx = -options.search; dx <= options.search; ++dx) {
          if (bx + dx < 0 || by + dy < 0 || bx + dx + options.block > ref.w ||
              by + dy + options.block > ref.h) {
            continue;
          }
          const std::int64_t sad = sad_at(ref, cur, bx, by, options.block, dx, dy,
                                          options.sample_step);
          if (sad < best) {
            best = sad;
            best_dx = dx;
            best_dy = dy;
          }
        }
      }
      if (best == std::numeric_limits<std::int64_t>::max()) continue;

      // One-dimensional parabolic refinement on each axis.
      auto subpel = [&](int delta_x, int delta_y) -> double {
        if (bx + delta_x < 0 || by + delta_y < 0 ||
            bx + delta_x + options.block > ref.w ||
            by + delta_y + options.block > ref.h) {
          return static_cast<double>(best);
        }
        return static_cast<double>(
            sad_at(ref, cur, bx, by, options.block, delta_x, delta_y, options.sample_step));
      };
      const double best_sad = static_cast<double>(best);
      const double smx = subpel(best_dx - 1, best_dy);
      const double spx = subpel(best_dx + 1, best_dy);
      const double smy = subpel(best_dx, best_dy - 1);
      const double spy = subpel(best_dx, best_dy + 1);
      const double denx = smx - 2.0 * best_sad + spx;
      const double deny = smy - 2.0 * best_sad + spy;
      double sub_x = denx > 0.0 ? 0.5 * (smx - spx) / denx : 0.0;
      double sub_y = deny > 0.0 ? 0.5 * (smy - spy) / deny : 0.0;
      sub_x = std::clamp(sub_x, -0.5, 0.5);
      sub_y = std::clamp(sub_y, -0.5, 0.5);

      GmBlockMv block;
      block.x = (bx + options.block / 2) * 8;
      block.y = (by + options.block / 2) * 8;
      block.mv_x = best_dx * 8 + static_cast<int>(std::lround(sub_x * 8.0));
      block.mv_y = best_dy * 8 + static_cast<int>(std::lround(sub_y * 8.0));
      block.weight = static_cast<int>(std::clamp(variance / 8.0, 1.0, 255.0));
      out.push_back(block);
    }
  }
  if (blocks_considered != nullptr) *blocks_considered = considered;
  return out;
}

void test_synthetic_frames() {
  const int w = 640;
  const int h = 360;
  const Plane source = make_texture(w, h);
  GmOptions options;
  options.frame_width = w;
  options.frame_height = h;
  options.outlier_passes = 3;
  options.outlier_threshold_q8 = 16;

  const std::pair<const char*, GmParams> cases[] = {
      {"pan", make_similarity(1.0, 0.0, 9.5, -5.25)},
      {"zoom", make_similarity(1.06, 0.0, 0.0, 0.0)},
      {"rotate", make_similarity(1.0, 2.0, 0.0, 0.0)},
      {"pan+zoom+rotate", make_similarity(1.04, 1.5, 4.0, -3.0)},
  };
  for (const auto& item : cases) {
    const Plane current = warp_plane(item.second, source, w, h);
    std::uint64_t considered = 0;
    MatchOptions match;
    match.block = 24;
    match.search = 24;
    match.sample_step = 2;
    const std::vector<GmBlockMv> blocks = match_blocks(source, current, match, &considered);
    EXPECT(blocks.size() > 100U);
    GmEstimate estimate = estimate_gm_block_mvs(blocks, options);
    EXPECT(estimate.diag.ok);
    EXPECT(estimate.diag.accepted);
    const GmParams expected = finalize_gm_params(item.second, options).params;
    check_model_close(estimate.params, expected, 512, 8 * kGmTransStep, item.first);
    const std::int64_t identity_sse =
        warp_plane_sse(make_identity_gm(), source.data.data(), w, h, w, current.data.data(), w, h,
                       w, 8);
    const std::int64_t model_sse =
        warp_plane_sse(estimate.params, source.data.data(), w, h, w, current.data.data(), w, h, w,
                       8);
    EXPECT(identity_sse >= 0);
    EXPECT(model_sse >= 0);
    EXPECT(model_sse <= identity_sse);
    std::printf("  synthetic %-16s points=%d used=%d type=%s gain=%.2f dB sse %lld -> %lld\n",
                item.first, estimate.diag.points_in, estimate.diag.points_used,
                gm_type_name(estimate.params.type), estimate.diag.gain_db,
                static_cast<long long>(identity_sse), static_cast<long long>(model_sse));
  }
}

bool load_y4m_frames(const std::string& path, int max_frames, std::vector<Plane>* out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::string line;
  if (!std::getline(input, line) || line.rfind("YUV4MPEG2", 0) != 0) return false;
  std::istringstream header(line);
  std::string token;
  int width = 0;
  int height = 0;
  bool c420 = false;
  while (header >> token) {
    if (token.size() > 1) {
      const char tag = token[0];
      const std::string value = token.substr(1);
      if (tag == 'W') width = std::stoi(value);
      if (tag == 'H') height = std::stoi(value);
      if (tag == 'C') c420 = value == "420" || value == "420jpeg" || value == "420mpeg2";
    }
  }
  if (width <= 0 || height <= 0 || !c420) return false;
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  std::vector<std::uint8_t> ybuf(static_cast<std::size_t>(width) * height);
  std::vector<std::uint8_t> ubuf(static_cast<std::size_t>(cw) * ch);
  std::vector<std::uint8_t> vbuf(static_cast<std::size_t>(cw) * ch);
  for (int frame = 0; frame < max_frames; ++frame) {
    if (!std::getline(input, line) || line.rfind("FRAME", 0) != 0) break;
    if (!input.read(reinterpret_cast<char*>(ybuf.data()),
                    static_cast<std::streamsize>(ybuf.size())) ||
        !input.read(reinterpret_cast<char*>(ubuf.data()),
                    static_cast<std::streamsize>(ubuf.size())) ||
        !input.read(reinterpret_cast<char*>(vbuf.data()),
                    static_cast<std::streamsize>(vbuf.size()))) {
      break;
    }
    Plane plane;
    plane.init(width, height);
    plane.data = ybuf;
    out->push_back(std::move(plane));
  }
  return !out->empty();
}

void test_real_720p(const std::string& repo_root) {
  const std::string path = repo_root + "/testdata/shine_1280x720_60f.y4m";
  std::vector<Plane> frames;
  if (!load_y4m_frames(path, 11, &frames) || frames.size() < 2) {
    std::printf("  real 720p: skipped (no testdata)\n");
    return;
  }
  GmOptions options;
  options.frame_width = frames[0].w;
  options.frame_height = frames[0].h;
  options.outlier_passes = 3;
  options.outlier_threshold_q8 = 24;

  auto analyze = [&](const Plane& reference, const Plane& current, const char* label,
                     int search) {
    MatchOptions match;
    match.block = 16;
    match.search = search;
    match.sample_step = 2;
    const std::vector<GmBlockMv> blocks = match_blocks(reference, current, match);
    GmEstimate estimate = estimate_gm_block_mvs(blocks, options);
    const std::int64_t identity_sse =
        warp_plane_sse(make_identity_gm(), reference.data.data(), reference.w, reference.h,
                       reference.w, current.data.data(), current.w, current.h, current.w, 8);
    const std::int64_t model_sse =
        warp_plane_sse(estimate.params, reference.data.data(), reference.w, reference.h,
                       reference.w, current.data.data(), current.w, current.h, current.w, 8);
    EXPECT(estimate.diag.ok);
    EXPECT(model_sse <= identity_sse);
    std::printf("  real 720p %-16s blocks=%zu used=%d type=%s gain=%.2f dB "
                "candidate=%.2f dB sse %lld -> %lld\n",
                label, blocks.size(), estimate.diag.points_used,
                gm_type_name(estimate.params.type), estimate.diag.gain_db,
                estimate.diag.candidate_gain_db,
                static_cast<long long>(identity_sse), static_cast<long long>(model_sse));
    return estimate;
  };

  analyze(frames[0], frames[1], "pair 0/1", 16);
  if (frames.size() >= 11) {
    analyze(frames[0], frames[10], "pair 0/10", 32);
  }

  // Real 720p content with a known global transform: this checks that the
  // matcher + estimator recover motion from real texture, not a synthetic
  // image.
  MatchOptions match;
  match.block = 16;
  match.search = 32;
  match.sample_step = 2;
  const GmParams known = make_similarity(1.02, 0.5, 3.0, -2.0);
  const Plane warped = warp_plane(known, frames[0], frames[0].w, frames[0].h);
  const std::vector<GmBlockMv> warped_blocks = match_blocks(frames[0], warped, match);
  GmEstimate warped_estimate = estimate_gm_block_mvs(warped_blocks, options);
  EXPECT(warped_estimate.diag.ok);
  EXPECT(warped_estimate.diag.accepted);
  check_model_close(warped_estimate.params, finalize_gm_params(known, options).params, 1024,
                    32 * kGmTransStep, "real 720p known warp");
  const std::int64_t known_identity_sse =
      warp_plane_sse(make_identity_gm(), frames[0].data.data(), frames[0].w, frames[0].h,
                     frames[0].w, warped.data.data(), warped.w, warped.h, warped.w, 8);
  const std::int64_t known_model_sse =
      warp_plane_sse(warped_estimate.params, frames[0].data.data(), frames[0].w, frames[0].h,
                     frames[0].w, warped.data.data(), warped.w, warped.h, warped.w, 8);
  EXPECT(known_model_sse <= known_identity_sse);
  std::printf("  real 720p known warp: blocks=%zu used=%d type=%s p0=%d p1=%d "
              "gain=%.2f dB sse %lld -> %lld\n",
              warped_blocks.size(), warped_estimate.diag.points_used,
              gm_type_name(warped_estimate.params.type), warped_estimate.params.p[0],
              warped_estimate.params.p[1], warped_estimate.diag.gain_db,
              static_cast<long long>(known_identity_sse),
              static_cast<long long>(known_model_sse));
}
}  // namespace

int main(int argc, char** argv) {
  const std::string repo_root = argc > 1 ? argv[1] : ".";
  std::printf("gme_test: AV2 global motion CPU reference\n");

  test_sequence_bit();
  test_global_param_roundtrip();
  test_global_param_prev_sweep();
  test_global_motion_params_roundtrip();
  test_global_motion_base_roundtrip();
  test_scale_warp_model();
  test_projection();
  test_shear_parity();
  test_validation_and_degradation();
  test_frame_plan();
  test_fit_exact();
  test_fit_noise_and_outliers();
  test_fit_from_warps_and_mvs();
  test_determinism();
  test_synthetic_frames();
  test_real_720p(repo_root);

  std::printf("gme_test: %d checks passed", g_checks);
  if (g_failures != 0) std::printf(", %d FAILED", g_failures);
  std::printf("\n");
  return g_failures == 0 ? 0 : 1;
}
