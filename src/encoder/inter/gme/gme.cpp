#include "gme.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

namespace av2enc::inter::gme {
namespace {

using i128 = __int128;

constexpr std::int64_t kI32Min = std::numeric_limits<std::int32_t>::min();
constexpr std::int64_t kI32Max = std::numeric_limits<std::int32_t>::max();

std::int64_t clamp_i64(std::int64_t value, std::int64_t low, std::int64_t high) {
  return value < low ? low : (value > high ? high : value);
}

std::int64_t to_i64_sat(i128 value) {
  if (value > static_cast<i128>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::max();
  }
  if (value < static_cast<i128>(std::numeric_limits<std::int64_t>::min())) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return static_cast<std::int64_t>(value);
}

std::int32_t saturate_i32(i128 value) {
  if (value > static_cast<i128>(kI32Max)) return static_cast<std::int32_t>(kI32Max);
  if (value < static_cast<i128>(kI32Min)) return static_cast<std::int32_t>(kI32Min);
  return static_cast<std::int32_t>(value);
}

i128 abs_i128(i128 value) { return value < 0 ? -value : value; }

// Round(num / den) to nearest, ties away from zero. den must be > 0.
std::int64_t div_round_i128(i128 num, i128 den) {
  if (den <= 0) return 0;
  const bool negative = num < 0;
  const i128 a = negative ? -num : num;
  i128 q = a / den;
  const i128 r = a % den;
  if (r * 2 >= den) ++q;
  if (q > static_cast<i128>(std::numeric_limits<std::int64_t>::max())) {
    return negative ? std::numeric_limits<std::int64_t>::min()
                    : std::numeric_limits<std::int64_t>::max();
  }
  const std::int64_t result = static_cast<std::int64_t>(q);
  return negative ? -result : result;
}

// Round(value * 2^-shift) for signed values, ties away from zero.
std::int64_t round_shift_i128(i128 value, int shift) {
  if (shift <= 0) return static_cast<std::int64_t>(value);
  return div_round_i128(value, static_cast<i128>(1) << shift);
}

// Exact Q16 ratio num/den with round-to-nearest. Uses integer long division so
// the result does not depend on the host floating-point environment.
std::int64_t ratio_q16(i128 num, i128 den) {
  if (den == 0) return 0;
  const bool negative = (num < 0) != (den < 0);
  i128 n = abs_i128(num);
  const i128 d = abs_i128(den);
  i128 integer = n / d;
  i128 rem = n % d;
  std::uint64_t frac = 0;
  for (int bit = 0; bit < 16; ++bit) {
    rem <<= 1;
    frac <<= 1;
    if (rem >= d) {
      rem -= d;
      frac |= 1U;
    }
  }
  // Round to nearest using the next fractional bit.
  rem <<= 1;
  if (rem >= d) ++frac;
  if (frac == (1U << 16)) {
    frac = 0;
    ++integer;
  }
  if (integer > static_cast<i128>(kI32Max >> 16)) {
    return negative ? kI32Min : kI32Max;
  }
  std::int64_t result = static_cast<std::int64_t>(integer) * 65536 +
                        static_cast<std::int64_t>(frac);
  if (negative) result = -result;
  return clamp_i64(result, kI32Min, kI32Max);
}

i128 project_x_q8(const GmParams& params, std::int64_t x_q8, std::int64_t y_q8) {
  // Absolute reference position in Q16 luma samples.
  const i128 x_q16 = static_cast<i128>(x_q8) * 8192;
  const i128 y_q16 = static_cast<i128>(y_q8) * 8192;
  const i128 value = (static_cast<i128>(params.p[2]) * x_q16 +
                      static_cast<i128>(params.p[3]) * y_q16) >>
                     16;
  return (value + params.p[0]) >> 13;
}

i128 project_y_q8(const GmParams& params, std::int64_t x_q8, std::int64_t y_q8) {
  const i128 x_q16 = static_cast<i128>(x_q8) * 8192;
  const i128 y_q16 = static_cast<i128>(y_q8) * 8192;
  const i128 value = (static_cast<i128>(params.p[4]) * x_q16 +
                      static_cast<i128>(params.p[5]) * y_q16) >>
                     16;
  return (value + params.p[1]) >> 13;
}

std::int64_t point_residual(const GmParams& params, const GmPoint& point) {
  const i128 px = project_x_q8(params, point.src_x, point.src_y);
  const i128 py = project_y_q8(params, point.src_x, point.src_y);
  const i128 dx = px - point.dst_x;
  const i128 dy = py - point.dst_y;
  const i128 sse = dx * dx + dy * dy;
  return static_cast<std::int64_t>(
      sse > static_cast<i128>(std::numeric_limits<std::int64_t>::max())
          ? std::numeric_limits<std::int64_t>::max()
          : sse);
}

std::int64_t median_of(std::vector<std::int64_t> values) {
  if (values.empty()) return 0;
  const std::size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                   values.end());
  return values[mid];
}

struct FitCore {
  GmParams params{};
  bool ok = false;
  std::string reason;
  int points_used = 0;
};

FitCore fit_core(std::span<const GmPoint> points, GmFitModel model, int min_points) {
  FitCore out;
  if (points.empty()) {
    out.reason = "no points";
    return out;
  }
  i128 weight_sum = 0;
  i128 sum_x = 0;
  i128 sum_y = 0;
  i128 sum_u = 0;
  i128 sum_v = 0;
  int used = 0;
  std::int64_t max_abs = 0;
  for (const GmPoint& point : points) {
    if (point.weight <= 0) continue;
    const i128 w = point.weight;
    weight_sum += w;
    sum_x += w * point.src_x;
    sum_y += w * point.src_y;
    sum_u += w * point.dst_x;
    sum_v += w * point.dst_y;
    max_abs = std::max<std::int64_t>(
        max_abs,
        std::max(std::max(std::llabs(static_cast<long long>(point.src_x)),
                          std::llabs(static_cast<long long>(point.src_y))),
                 std::max(std::llabs(static_cast<long long>(point.dst_x)),
                          std::llabs(static_cast<long long>(point.dst_y)))));
    ++used;
  }
  if (used < min_points || weight_sum <= 0) {
    out.reason = "too few weighted points";
    return out;
  }

  const std::int64_t mean_x = div_round_i128(sum_x, weight_sum);
  const std::int64_t mean_y = div_round_i128(sum_y, weight_sum);
  const std::int64_t mean_u = div_round_i128(sum_u, weight_sum);
  const std::int64_t mean_v = div_round_i128(sum_v, weight_sum);

  int shift = 0;
  while ((max_abs >> shift) > 16384) ++shift;

  i128 sxx = 0;
  i128 sxy = 0;
  i128 syy = 0;
  i128 sxu = 0;
  i128 syu = 0;
  i128 sxv = 0;
  i128 syv = 0;
  i128 den = 0;
  i128 num_a = 0;
  i128 num_b = 0;

  for (const GmPoint& point : points) {
    if (point.weight <= 0) continue;
    const i128 w = point.weight;
    const std::int64_t cx = (static_cast<std::int64_t>(point.src_x) - mean_x) >> shift;
    const std::int64_t cy = (static_cast<std::int64_t>(point.src_y) - mean_y) >> shift;
    const std::int64_t cu = (static_cast<std::int64_t>(point.dst_x) - mean_u) >> shift;
    const std::int64_t cv = (static_cast<std::int64_t>(point.dst_y) - mean_v) >> shift;
    const i128 x = cx;
    const i128 y = cy;
    const i128 u = cu;
    const i128 v = cv;
    if (model != GmFitModel::kTranslation) {
      sxx += w * x * x;
      sxy += w * x * y;
      syy += w * y * y;
      sxu += w * x * u;
      syu += w * y * u;
      sxv += w * x * v;
      syv += w * y * v;
      den += w * (x * x + y * y);
      num_a += w * (x * u + y * v);
      num_b += w * (x * v - y * u);
    }
  }

  std::int64_t a = kGmOne;
  std::int64_t b = 0;
  std::int64_t c = 0;
  std::int64_t d = kGmOne;

  if (model == GmFitModel::kRotZoom) {
    if (den == 0) {
      out.reason = "degenerate point distribution";
      return out;
    }
    a = ratio_q16(num_a, den);
    b = ratio_q16(num_b, den);
    c = -b;
    d = a;
  } else if (model == GmFitModel::kAffine) {
    const i128 det = sxx * syy - sxy * sxy;
    if (det <= 0) {
      out.reason = "collinear points (det <= 0)";
      return out;
    }
    a = ratio_q16(sxu * syy - syu * sxy, det);
    b = ratio_q16(syu * sxx - sxu * sxy, det);
    c = ratio_q16(sxv * syy - syv * sxy, det);
    d = ratio_q16(syv * sxx - sxv * sxy, det);
  }

  const std::int64_t tx =
      mean_u - round_shift_i128(static_cast<i128>(a) * mean_x +
                                    static_cast<i128>(b) * mean_y,
                                16);
  const std::int64_t ty =
      mean_v - round_shift_i128(static_cast<i128>(c) * mean_x +
                                    static_cast<i128>(d) * mean_y,
                                16);

  out.params.type = model == GmFitModel::kAffine ? GmType::kAffine : GmType::kRotZoom;
  if (model == GmFitModel::kTranslation) {
    out.params.type = GmType::kRotZoom;  // identity affine part + translation
    out.params.p = {0, 0, kGmOne, 0, 0, kGmOne};
  } else {
    out.params.p[2] = saturate_i32(a);
    out.params.p[3] = saturate_i32(b);
    out.params.p[4] = saturate_i32(c);
    out.params.p[5] = saturate_i32(d);
  }
  out.params.p[0] = saturate_i32(static_cast<i128>(tx) * 8192);
  out.params.p[1] = saturate_i32(static_cast<i128>(ty) * 8192);
  out.ok = true;
  out.points_used = used;
  return out;
}

std::int32_t quantize_step(std::int64_t value, std::int64_t step, std::int64_t min_value,
                           std::int64_t max_value) {
  std::int64_t quantised = div_round_i128(value, step) * step;
  quantised = clamp_i64(quantised, min_value, max_value);
  return static_cast<std::int32_t>(quantised);
}

GmParams quantize_model(const GmParams& input, const GmOptions& options) {
  GmParams out = input;
  const std::int64_t trans_step = options.force_integer_mv ? kGmOne : kGmTransStep;
  const std::int64_t trans_raw_limit =
      static_cast<std::int64_t>(kGmTransMax) * kGmTransStep;
  // Keep the clamp on the selected quantisation grid. This matters for
  // force_integer_mv, where the syntax limit is not a whole-pixel multiple.
  const std::int64_t trans_limit = (trans_raw_limit / trans_step) * trans_step;
  out.p[0] = quantize_step(input.p[0], trans_step, -trans_limit, trans_limit);
  out.p[1] = quantize_step(input.p[1], trans_step, -trans_limit, trans_limit);

  const std::int64_t alpha_limit =
      static_cast<std::int64_t>(kGmAlphaMax) * kGmAlphaStep;
  if (out.type == GmType::kRotZoom) {
    const std::int64_t a = (static_cast<std::int64_t>(input.p[2]) + input.p[5]) / 2;
    const std::int64_t b = (static_cast<std::int64_t>(input.p[3]) - input.p[4]) / 2;
    const std::int32_t qa = quantize_step(a - kGmOne, kGmAlphaStep, -alpha_limit,
                                          alpha_limit) +
                           kGmOne;
    const std::int32_t qb =
        quantize_step(b, kGmAlphaStep, -alpha_limit, alpha_limit);
    out.p[2] = qa;
    out.p[3] = qb;
    out.p[4] = -qb;
    out.p[5] = qa;
  } else if (out.type == GmType::kAffine) {
    out.p[2] = quantize_step(static_cast<std::int64_t>(input.p[2]) - kGmOne,
                             kGmAlphaStep, -alpha_limit, alpha_limit) +
               kGmOne;
    out.p[3] = quantize_step(input.p[3], kGmAlphaStep, -alpha_limit, alpha_limit);
    out.p[4] = quantize_step(input.p[4], kGmAlphaStep, -alpha_limit, alpha_limit);
    out.p[5] = quantize_step(static_cast<std::int64_t>(input.p[5]) - kGmOne,
                             kGmAlphaStep, -alpha_limit, alpha_limit) +
               kGmOne;
  } else {
    out.p = {0, 0, kGmOne, 0, 0, kGmOne};
  }
  if (is_identity_model(out)) out.type = GmType::kIdentity;
  return out;
}

bool model_scale_ok(const GmParams& params, const GmOptions& options) {
  if (params.type == GmType::kIdentity) return true;
  if (params.p[2] < options.min_scale_q16 || params.p[2] > options.max_scale_q16) {
    return false;
  }
  if (params.type == GmType::kAffine) {
    if (params.p[5] < options.min_scale_q16 || params.p[5] > options.max_scale_q16) {
      return false;
    }
    const std::int64_t det = static_cast<std::int64_t>(params.p[2]) * params.p[5] -
                             static_cast<std::int64_t>(params.p[3]) * params.p[4];
    if (det <= 0) return false;
  }
  return true;
}

bool corner_projection_ok(const GmParams& params, const GmOptions& options) {
  if (options.frame_width <= 0 || options.frame_height <= 0) return true;
  const std::int64_t corners[4][2] = {
      {0, 0},
      {static_cast<std::int64_t>(options.frame_width - 1) << 16, 0},
      {0, static_cast<std::int64_t>(options.frame_height - 1) << 16},
      {static_cast<std::int64_t>(options.frame_width - 1) << 16,
       static_cast<std::int64_t>(options.frame_height - 1) << 16},
  };
  for (const auto& corner : corners) {
    std::int32_t rx = 0;
    std::int32_t ry = 0;
    project_gm_point(params, saturate_i32(corner[0]), saturate_i32(corner[1]), &rx, &ry);
    const std::int64_t dx = static_cast<std::int64_t>(rx) - corner[0];
    const std::int64_t dy = static_cast<std::int64_t>(ry) - corner[1];
    const std::int64_t limit = options.max_corner_displacement_q8 * 8192;
    if (std::llabs(static_cast<long long>(dx)) > limit ||
        std::llabs(static_cast<long long>(dy)) > limit) {
      return false;
    }
  }
  return true;
}

GmEstimate finalize_once(const GmParams& candidate, const GmOptions& options, bool degraded,
                         const std::string& reason) {
  GmEstimate out;
  out.params = options.quantize ? quantize_model(candidate, options) : candidate;
  out.diag.degraded = degraded;
  out.diag.reason = reason;
  out.diag.identity = is_identity_model(out.params);
  out.diag.shear_valid = setup_shear_valid(out.params.p.data());
  out.diag.shear_mismatch_q16 =
      std::max(abs_int(out.params.p[3] + out.params.p[4]),
               abs_int(out.params.p[5] - out.params.p[2]));
  out.diag.codable = gm_params_codable(out.params, &out.diag.reason);
  out.diag.ok = true;
  if (out.diag.identity) {
    out.diag.accepted = true;
    out.diag.codable = true;
    return out;
  }
  if (!out.diag.codable) {
    out.diag.accepted = false;
    return out;
  }
  if (!model_scale_ok(out.params, options)) {
    out.diag.accepted = false;
    out.diag.reason = "model scale/determinant out of range";
    return out;
  }
  if (options.require_shear_valid && !out.diag.shear_valid) {
    out.diag.accepted = false;
    out.diag.reason = "shear model is not valid";
    return out;
  }
  if (!corner_projection_ok(out.params, options)) {
    out.diag.accepted = false;
    out.diag.reason = "corner projection out of range";
    return out;
  }
  out.diag.accepted = true;
  return out;
}

GmEstimate identity_estimate(const std::string& reason) {
  GmEstimate out;
  out.params = make_identity_gm();
  out.diag.ok = false;
  out.diag.accepted = false;
  out.diag.codable = true;
  out.diag.identity = true;
  out.diag.reason = reason;
  return out;
}

}  // namespace

const char* gm_type_name(GmType type) {
  switch (type) {
    case GmType::kIdentity:
      return "IDENTITY";
    case GmType::kRotZoom:
      return "ROTZOOM";
    case GmType::kAffine:
      return "AFFINE";
  }
  return "UNKNOWN";
}

const char* gm_fit_model_name(GmFitModel model) {
  switch (model) {
    case GmFitModel::kTranslation:
      return "translation";
    case GmFitModel::kRotZoom:
      return "rotzoom";
    case GmFitModel::kAffine:
      return "affine";
  }
  return "unknown";
}

GmParams make_identity_gm() { return GmParams{}; }

GmParams make_gm(GmType type, const std::array<std::int32_t, 6>& p) {
  GmParams out;
  out.type = type;
  out.p = p;
  return out;
}

bool is_identity_model(const GmParams& params) {
  return params.p[0] == 0 && params.p[1] == 0 && params.p[2] == kGmOne &&
         params.p[3] == 0 && params.p[4] == 0 && params.p[5] == kGmOne;
}

GmQuality evaluate_gm_quality(const GmParams& params, std::span<const GmPoint> points) {
  GmQuality quality;
  i128 identity_sse = 0;
  i128 model_sse = 0;
  i128 weight = 0;
  for (const GmPoint& point : points) {
    if (point.weight <= 0) continue;
    const i128 w = point.weight;
    const i128 idx = static_cast<i128>(point.dst_x) - point.src_x;
    const i128 idy = static_cast<i128>(point.dst_y) - point.src_y;
    identity_sse += w * (idx * idx + idy * idy);
    const i128 px = project_x_q8(params, point.src_x, point.src_y);
    const i128 py = project_y_q8(params, point.src_x, point.src_y);
    const i128 dx = px - point.dst_x;
    const i128 dy = py - point.dst_y;
    model_sse += w * (dx * dx + dy * dy);
    weight += w;
  }
  const i128 max_i64 = static_cast<i128>(std::numeric_limits<std::int64_t>::max());
  quality.identity_sse_q8 =
      static_cast<std::int64_t>(identity_sse > max_i64 ? max_i64 : identity_sse);
  quality.model_sse_q8 =
      static_cast<std::int64_t>(model_sse > max_i64 ? max_i64 : model_sse);
  quality.total_weight =
      static_cast<std::int64_t>(weight > max_i64 ? max_i64 : weight);
  const double w = quality.total_weight > 0 ? static_cast<double>(quality.total_weight) : 1.0;
  quality.identity_rmse_q8 = std::sqrt(static_cast<double>(quality.identity_sse_q8) / w);
  quality.model_rmse_q8 = std::sqrt(static_cast<double>(quality.model_sse_q8) / w);
  quality.improves = quality.model_sse_q8 < quality.identity_sse_q8;
  if (quality.model_sse_q8 > 0 && quality.identity_sse_q8 > 0) {
    quality.gain_db =
        10.0 * std::log10(static_cast<double>(quality.identity_sse_q8) /
                          static_cast<double>(quality.model_sse_q8));
  } else if (quality.model_sse_q8 == 0) {
    quality.gain_db = 99.0;
  }
  return quality;
}

GmEstimate fit_gm_points(std::span<const GmPoint> points, const GmOptions& options) {
  GmEstimate out;
  if (points.size() < static_cast<std::size_t>(std::max(1, options.min_points))) {
    out.diag.reason = "too few correspondences";
    return out;
  }
  if (points.size() > static_cast<std::size_t>(std::max(1, options.max_points))) {
    out.diag.reason = "too many correspondences";
    return out;
  }

  std::vector<GmPoint> active(points.begin(), points.end());
  FitCore fit = fit_core(active, options.model, options.min_points);
  int passes = 0;
  if (options.outlier_passes > 0 && fit.ok) {
    const i128 threshold_floor = static_cast<i128>(options.outlier_threshold_q8) *
                                 options.outlier_threshold_q8;
    for (int pass = 0; pass < options.outlier_passes; ++pass) {
      std::vector<std::int64_t> residuals;
      residuals.reserve(active.size());
      for (const GmPoint& point : active) {
        if (point.weight <= 0) continue;
        residuals.push_back(point_residual(fit.params, point));
      }
      if (residuals.size() <= static_cast<std::size_t>(options.min_points)) break;
      const std::int64_t median = median_of(residuals);
      const i128 threshold = std::max<i128>(threshold_floor, static_cast<i128>(median) * 4);
      std::vector<GmPoint> kept;
      kept.reserve(active.size());
      for (const GmPoint& point : active) {
        if (point.weight <= 0) continue;
        if (static_cast<i128>(point_residual(fit.params, point)) <= threshold) {
          kept.push_back(point);
        }
      }
      ++passes;
      if (kept.size() < static_cast<std::size_t>(options.min_points)) break;
      if (kept.size() == active.size()) break;
      active = std::move(kept);
      fit = fit_core(active, options.model, options.min_points);
      if (!fit.ok) break;
    }
  }

  out.params = fit.params;
  out.diag.ok = fit.ok;
  out.diag.points_in = static_cast<int>(points.size());
  out.diag.points_used = fit.points_used;
  out.diag.outlier_passes_run = passes;
  out.diag.reason = fit.reason;
  out.diag.identity = is_identity_model(fit.params);
  return out;
}

GmEstimate fit_gm_block_mvs(std::span<const GmBlockMv> blocks, const GmOptions& options) {
  std::vector<GmPoint> points;
  points.reserve(blocks.size());
  for (const GmBlockMv& block : blocks) {
    if (block.weight <= 0) continue;
    const std::int64_t dst_x = static_cast<std::int64_t>(block.x) + block.mv_x;
    const std::int64_t dst_y = static_cast<std::int64_t>(block.y) + block.mv_y;
    if (dst_x < kI32Min || dst_x > kI32Max || dst_y < kI32Min || dst_y > kI32Max) {
      return identity_estimate("block MV correspondence exceeds int32");
    }
    GmPoint point;
    point.src_x = block.x;
    point.src_y = block.y;
    point.dst_x = static_cast<std::int32_t>(dst_x);
    point.dst_y = static_cast<std::int32_t>(dst_y);
    point.weight = block.weight;
    points.push_back(point);
  }
  return fit_gm_points(points, options);
}

GmEstimate fit_gm_warps(std::span<const GmWarpSample> warps, const GmOptions& options) {
  std::vector<GmPoint> points;
  points.reserve(warps.size() * 5U);
  for (const GmWarpSample& warp : warps) {
    if (warp.weight <= 0) continue;
    const std::int32_t base = warp.weight / 5;
    std::int32_t remainder = warp.weight % 5;
    const std::int32_t offsets[5][2] = {
        {0, 0},
        {warp.radius, 0},
        {-warp.radius, 0},
        {0, warp.radius},
        {0, -warp.radius},
    };
    for (const auto& offset : offsets) {
      const std::int64_t sx = static_cast<std::int64_t>(warp.x) + offset[0];
      const std::int64_t sy = static_cast<std::int64_t>(warp.y) + offset[1];
      if (sx < kI32Min || sx > kI32Max || sy < kI32Min || sy > kI32Max) {
        return identity_estimate("warp sample coordinate exceeds int32");
      }
      std::int32_t rx = 0;
      std::int32_t ry = 0;
      project_gm_point(warp.model, static_cast<std::int32_t>(sx) * 8192,
                       static_cast<std::int32_t>(sy) * 8192, &rx, &ry);
      GmPoint point;
      point.src_x = static_cast<std::int32_t>(sx);
      point.src_y = static_cast<std::int32_t>(sy);
      point.dst_x = static_cast<std::int32_t>(static_cast<std::int64_t>(rx) >> 13);
      point.dst_y = static_cast<std::int32_t>(static_cast<std::int64_t>(ry) >> 13);
      point.weight = base > 0 ? base : 1;
      if (remainder > 0) {
        point.weight += 1;
        --remainder;
      }
      points.push_back(point);
    }
  }
  return fit_gm_points(points, options);
}

bool gm_params_codable(const GmParams& params, std::string* reason) {
  auto fail = [&](const char* text) {
    if (reason != nullptr) *reason = text;
    return false;
  };
  for (int i = 0; i < 6; ++i) {
    const std::int32_t value = params.p[static_cast<std::size_t>(i)];
    if (i < 2) {
      if ((value % kGmTransStep) != 0) return fail("translation not on the 1/8-pel grid");
      const std::int64_t limit = static_cast<std::int64_t>(kGmTransMax) * kGmTransStep;
      if (value < -limit || value > limit) {
        return fail("translation outside GM_TRANS range");
      }
    } else {
      if ((value % kGmAlphaStep) != 0) return fail("alpha not on the 1/1024 grid");
      const std::int64_t offset = (i == 2 || i == 5) ? value - kGmOne : value;
      const std::int64_t limit = static_cast<std::int64_t>(kGmAlphaMax) * kGmAlphaStep;
      if (offset < -limit || offset > limit) {
        return fail("alpha outside GM_ALPHA range");
      }
    }
  }
  if (params.type == GmType::kIdentity) {
    if (!is_identity_model(params)) return fail("IDENTITY with non-default parameters");
    return true;
  }
  if (params.type == GmType::kRotZoom) {
    if (params.p[4] != -params.p[3] || params.p[5] != params.p[2]) {
      return fail("ROTZOOM requires p4 = -p3 and p5 = p2");
    }
  } else if (params.type != GmType::kAffine) {
    return fail("unknown GmType");
  }
  return true;
}

bool validate_gm_params(const GmParams& params, const GmOptions& options,
                        GmDiagnostics* diag) {
  GmDiagnostics local;
  local.ok = true;
  local.identity = is_identity_model(params);
  std::string reason;
  local.codable = gm_params_codable(params, &reason);
  local.shear_valid = setup_shear_valid(params.p.data());
  local.shear_mismatch_q16 =
      std::max(abs_int(params.p[4] + params.p[3]), abs_int(params.p[5] - params.p[2]));
  if (!local.codable) {
    local.accepted = false;
    local.reason = reason;
  } else if (local.identity) {
    local.accepted = true;
  } else if (!model_scale_ok(params, options)) {
    local.accepted = false;
    local.reason = "model scale/determinant out of range";
  } else if (options.require_shear_valid && !local.shear_valid) {
    local.accepted = false;
    local.reason = "shear model is not valid";
  } else if (!corner_projection_ok(params, options)) {
    local.accepted = false;
    local.reason = "corner projection out of range";
  } else {
    local.accepted = true;
  }
  if (diag != nullptr) *diag = local;
  return local.accepted;
}

GmEstimate finalize_gm_params(const GmParams& raw, const GmOptions& options) {
  GmEstimate out;
  out.diag.points_in = 0;
  out.diag.points_used = 0;

  GmParams candidate = raw;
  bool degraded = false;
  std::string reason;
  if (candidate.type == GmType::kAffine) {
    const std::int32_t mismatch =
        std::max(abs_int(candidate.p[4] + candidate.p[3]),
                 abs_int(candidate.p[5] - candidate.p[2]));
    if (options.allow_rotzoom_degrade && mismatch <= options.rotzoom_mismatch_q16) {
      const std::int64_t a =
          (static_cast<std::int64_t>(candidate.p[2]) + candidate.p[5]) / 2;
      const std::int64_t b =
          (static_cast<std::int64_t>(candidate.p[3]) - candidate.p[4]) / 2;
      candidate.type = GmType::kRotZoom;
      candidate.p[2] = static_cast<std::int32_t>(a);
      candidate.p[3] = static_cast<std::int32_t>(b);
      candidate.p[4] = static_cast<std::int32_t>(-b);
      candidate.p[5] = static_cast<std::int32_t>(a);
      degraded = true;
      reason = "affine close to similarity; signalled ROTZOOM";
    }
  }
  if (candidate.type == GmType::kRotZoom) {
    const std::int64_t a =
        (static_cast<std::int64_t>(candidate.p[2]) + candidate.p[5]) / 2;
    const std::int64_t b =
        (static_cast<std::int64_t>(candidate.p[3]) - candidate.p[4]) / 2;
    candidate.p[2] = static_cast<std::int32_t>(a);
    candidate.p[3] = static_cast<std::int32_t>(b);
    candidate.p[4] = static_cast<std::int32_t>(-b);
    candidate.p[5] = static_cast<std::int32_t>(a);
  }

  out = finalize_once(candidate, options, degraded, reason);
  if (out.diag.accepted) return out;

  // An AFFINE model can be legal but fail the scale/shear guard. Try its
  // similarity projection once, then fall back to IDENTITY.
  if (raw.type == GmType::kAffine && options.allow_rotzoom_degrade) {
    GmParams rot = raw;
    const std::int64_t a = (static_cast<std::int64_t>(raw.p[2]) + raw.p[5]) / 2;
    const std::int64_t b = (static_cast<std::int64_t>(raw.p[3]) - raw.p[4]) / 2;
    rot.type = GmType::kRotZoom;
    rot.p[2] = static_cast<std::int32_t>(a);
    rot.p[3] = static_cast<std::int32_t>(b);
    rot.p[4] = static_cast<std::int32_t>(-b);
    rot.p[5] = static_cast<std::int32_t>(a);
    GmEstimate rot_out = finalize_once(rot, options, true,
                                       "degraded to ROTZOOM after AFFINE rejection");
    if (rot_out.diag.accepted) {
      rot_out.diag.points_in = out.diag.points_in;
      rot_out.diag.points_used = out.diag.points_used;
      return rot_out;
    }
  }

  GmEstimate identity = finalize_once(make_identity_gm(), options, true,
                                      out.diag.reason.empty() ? "model rejected"
                                                              : out.diag.reason);
  identity.diag.ok = out.diag.ok;
  identity.diag.points_in = out.diag.points_in;
  identity.diag.points_used = out.diag.points_used;
  identity.diag.degraded = true;
  return identity;
}

namespace {

GmEstimate estimate_points_internal(std::span<const GmPoint> points, const GmOptions& options) {
  GmEstimate fit = fit_gm_points(points, options);
  if (!fit.diag.ok) return fit;
  GmEstimate final = finalize_gm_params(fit.params, options);
  final.diag.points_in = fit.diag.points_in;
  final.diag.points_used = fit.diag.points_used;
  final.diag.outlier_passes_run = fit.diag.outlier_passes_run;
  final.diag.shear_mismatch_q16 =
      std::max(abs_int(final.params.p[3] + final.params.p[4]),
               abs_int(final.params.p[5] - final.params.p[2]));

  const GmQuality quality = evaluate_gm_quality(final.params, points);
  final.diag.identity_sse_q8 = quality.identity_sse_q8;
  final.diag.model_sse_q8 = quality.model_sse_q8;
  final.diag.gain_db = quality.gain_db;
  final.diag.model_rmse_q8 = quality.model_rmse_q8;
  final.diag.identity = is_identity_model(final.params);

  if (!final.diag.identity && final.diag.accepted) {
    final.diag.candidate_gain_db = quality.gain_db;
    final.diag.candidate_model_sse_q8 = quality.model_sse_q8;
    const bool enough_gain =
        quality.identity_sse_q8 <= 0 ||
        static_cast<i128>(quality.model_sse_q8) * 100 <=
            static_cast<i128>(quality.identity_sse_q8) *
                (100 - std::max(0, options.min_gain_percent));
    if (!enough_gain) {
      final.params = make_identity_gm();
      final.diag.identity = true;
      final.diag.accepted = true;
      final.diag.degraded = true;
      final.diag.gain_db = 0.0;
      final.diag.model_sse_q8 = quality.identity_sse_q8;
      final.diag.reason =
          "global motion gain below threshold (candidate " +
          std::to_string(quality.gain_db) + " dB)";
    }
  }
  return final;
}

}  // namespace

GmEstimate estimate_gm_points(std::span<const GmPoint> points, const GmOptions& options) {
  return estimate_points_internal(points, options);
}

GmEstimate estimate_gm_block_mvs(std::span<const GmBlockMv> blocks, const GmOptions& options) {
  std::vector<GmPoint> points;
  points.reserve(blocks.size());
  for (const GmBlockMv& block : blocks) {
    if (block.weight <= 0) continue;
    const std::int64_t dst_x = static_cast<std::int64_t>(block.x) + block.mv_x;
    const std::int64_t dst_y = static_cast<std::int64_t>(block.y) + block.mv_y;
    if (dst_x < kI32Min || dst_x > kI32Max || dst_y < kI32Min || dst_y > kI32Max) {
      return identity_estimate("block MV correspondence exceeds int32");
    }
    GmPoint point;
    point.src_x = block.x;
    point.src_y = block.y;
    point.dst_x = static_cast<std::int32_t>(dst_x);
    point.dst_y = static_cast<std::int32_t>(dst_y);
    point.weight = block.weight;
    points.push_back(point);
  }
  return estimate_points_internal(points, options);
}

GmEstimate estimate_gm_warps(std::span<const GmWarpSample> warps, const GmOptions& options) {
  GmEstimate fit = fit_gm_warps(warps, options);
  if (!fit.diag.ok) return fit;
  // fit_gm_warps already ran the correspondence fit; approximate the final
  // quality with the sampled correspondences.
  std::vector<GmPoint> points;
  points.reserve(warps.size() * 5U);
  for (const GmWarpSample& warp : warps) {
    if (warp.weight <= 0) continue;
    const std::int32_t base = warp.weight / 5;
    std::int32_t remainder = warp.weight % 5;
    const std::int32_t offsets[5][2] = {
        {0, 0}, {warp.radius, 0}, {-warp.radius, 0}, {0, warp.radius}, {0, -warp.radius}};
    for (const auto& offset : offsets) {
      std::int32_t rx = 0;
      std::int32_t ry = 0;
      project_gm_point(warp.model, (warp.x + offset[0]) * 8192,
                       (warp.y + offset[1]) * 8192, &rx, &ry);
      GmPoint point;
      point.src_x = warp.x + offset[0];
      point.src_y = warp.y + offset[1];
      point.dst_x = rx >> 13;
      point.dst_y = ry >> 13;
      point.weight = base > 0 ? base : 1;
      if (remainder > 0) {
        point.weight += 1;
        --remainder;
      }
      points.push_back(point);
    }
  }
  GmEstimate final = finalize_gm_params(fit.params, options);
  final.diag.points_in = fit.diag.points_in;
  final.diag.points_used = fit.diag.points_used;
  final.diag.outlier_passes_run = fit.diag.outlier_passes_run;
  final.diag.shear_mismatch_q16 =
      std::max(abs_int(final.params.p[3] + final.params.p[4]),
               abs_int(final.params.p[5] - final.params.p[2]));
  const GmQuality quality = evaluate_gm_quality(final.params, points);
  final.diag.identity_sse_q8 = quality.identity_sse_q8;
  final.diag.model_sse_q8 = quality.model_sse_q8;
  final.diag.gain_db = quality.gain_db;
  final.diag.model_rmse_q8 = quality.model_rmse_q8;
  final.diag.identity = is_identity_model(final.params);
  if (!final.diag.identity && final.diag.accepted) {
    final.diag.candidate_gain_db = quality.gain_db;
    final.diag.candidate_model_sse_q8 = quality.model_sse_q8;
    const bool enough_gain =
        quality.identity_sse_q8 <= 0 ||
        static_cast<i128>(quality.model_sse_q8) * 100 <=
            static_cast<i128>(quality.identity_sse_q8) *
                (100 - std::max(0, options.min_gain_percent));
    if (!enough_gain) {
      final.params = make_identity_gm();
      final.diag.identity = true;
      final.diag.accepted = true;
      final.diag.degraded = true;
      final.diag.gain_db = 0.0;
      final.diag.model_sse_q8 = quality.identity_sse_q8;
      final.diag.reason =
          "global motion gain below threshold (candidate " +
          std::to_string(quality.gain_db) + " dB)";
    }
  }
  return final;
}

void project_gm_point(const GmParams& params, std::int32_t x_q16, std::int32_t y_q16,
                      std::int32_t* ref_x_q16, std::int32_t* ref_y_q16) {
  const i128 x = x_q16;
  const i128 y = y_q16;
  const i128 rx = ((static_cast<i128>(params.p[2]) * x +
                    static_cast<i128>(params.p[3]) * y) >>
                   16) +
                  params.p[0];
  const i128 ry = ((static_cast<i128>(params.p[4]) * x +
                    static_cast<i128>(params.p[5]) * y) >>
                   16) +
                  params.p[1];
  if (ref_x_q16 != nullptr) *ref_x_q16 = saturate_i32(rx);
  if (ref_y_q16 != nullptr) *ref_y_q16 = saturate_i32(ry);
}

namespace {

int lower_mv_precision_value(int value, MvPrecision precision) {
  const int bits = 7 - static_cast<int>(precision);
  if (bits <= 0) return value;
  const int radix = 1 << bits;
  const int a = abs_int(value);
  const int a_int = (a + (1 << (bits - 1)) - 1) >> bits;
  int result = value >= 0 ? (a_int << bits) : (-a_int << bits);
  if ((a_int << bits) != a) {
    result = clip3(kMvLow + radix, kMvUpp - radix, result);
  }
  return result;
}

}  // namespace

void get_warp_motion_vector(const GmParams& params, int mi_col, int mi_row, int bw4, int bh4,
                            int mi_cols, int mi_rows, MvPrecision precision, int mv[2]) {
  const int bw = bw4 * kMiSize;
  const int bh = bh4 * kMiSize;
  const int x = mi_col * kMiSize + bw / 2 - 1;
  const int y = mi_row * kMiSize + bh / 2 - 1;
  const std::int64_t xc =
      static_cast<std::int64_t>(params.p[2] - kGmOne) * x +
      static_cast<std::int64_t>(params.p[3]) * y + params.p[0];
  const std::int64_t yc =
      static_cast<std::int64_t>(params.p[4]) * x +
      static_cast<std::int64_t>(params.p[5] - kGmOne) * y + params.p[1];
  std::int64_t mv_y = 0;
  std::int64_t mv_x = 0;
  if (precision == MvPrecision::kEighthPel) {
    mv_y = round2_signed64(yc, kWarpedModelPrecBits - 3);
    mv_x = round2_signed64(xc, kWarpedModelPrecBits - 3);
  } else {
    mv_y = round2_signed64(yc, kWarpedModelPrecBits - 2) * 2;
    mv_x = round2_signed64(xc, kWarpedModelPrecBits - 2) * 2;
  }
  mv_y = clip3(kMvLow + 1, kMvUpp - 1, static_cast<int>(clamp_i64(
                                          mv_y, kI32Min, kI32Max)));
  mv_x = clip3(kMvLow + 1, kMvUpp - 1, static_cast<int>(clamp_i64(
                                          mv_x, kI32Min, kI32Max)));
  const int low_row = -(mi_row + bh4) * kMiSize * 8 - kMvBorder;
  const int high_row = (mi_rows - mi_row) * kMiSize * 8 + kMvBorder;
  const int low_col = -(mi_col + bw4) * kMiSize * 8 - kMvBorder;
  const int high_col = (mi_cols - mi_col) * kMiSize * 8 + kMvBorder;
  mv_y = clip3(low_row, high_row, static_cast<int>(mv_y));
  mv_x = clip3(low_col, high_col, static_cast<int>(mv_x));
  if (precision < MvPrecision::kHalfPel) {
    mv_y = lower_mv_precision_value(static_cast<int>(mv_y), precision);
    mv_x = lower_mv_precision_value(static_cast<int>(mv_x), precision);
  }
  mv[0] = static_cast<int>(mv_y);
  mv[1] = static_cast<int>(mv_x);
}

GmGlobalMvUse global_mv_for_block(const GmParams& params, int ref, int mi_col, int mi_row,
                                  int bw4, int bh4, int mi_cols, int mi_rows) {
  GmGlobalMvUse out;
  out.ref = ref;
  out.usable = params.type != GmType::kIdentity;
  if (!out.usable) return out;
  const int bw = bw4 * kMiSize;
  const int bh = bh4 * kMiSize;
  out.use_warp = std::min(bw, bh) >= 8;
  get_warp_motion_vector(params, mi_col, mi_row, bw4, bh4, mi_cols, mi_rows,
                         MvPrecision::kEighthPel, out.mv);
  return out;
}

GmFramePlan make_gm_frame_plan(std::span<const GmEstimate> estimates,
                               bool enable_sequence_global_motion) {
  GmFramePlan plan;
  plan.sequence_enable_global_motion = enable_sequence_global_motion;
  plan.num_total_refs = static_cast<int>(
      std::min<std::size_t>(estimates.size(), static_cast<std::size_t>(kMaxRefsPerFrame)));
  for (int ref = 0; ref < plan.num_total_refs; ++ref) {
    const GmEstimate& estimate = estimates[static_cast<std::size_t>(ref)];
    plan.params[static_cast<std::size_t>(ref)] = estimate.params;
    if (estimate.diag.accepted && !is_identity_model(estimate.params)) {
      plan.use_global_motion = true;
    }
  }
  if (!enable_sequence_global_motion) {
    plan.use_global_motion = false;
    for (int ref = 0; ref < plan.num_total_refs; ++ref) {
      plan.params[static_cast<std::size_t>(ref)] = make_identity_gm();
    }
  }
  return plan;
}

// ---------------------------------------------------------------------------
// Bilinear quality proxy (NOT normative prediction).
// ---------------------------------------------------------------------------

namespace {

std::uint8_t bilinear_sample(const std::uint8_t* plane, int w, int h, int stride,
                             std::int64_t x_q16, std::int64_t y_q16, int max_value) {
  const std::int64_t fx = x_q16 & 0xFFFF;
  const std::int64_t fy = y_q16 & 0xFFFF;
  const std::int64_t ix = x_q16 >> 16;
  const std::int64_t iy = y_q16 >> 16;
  const int x0 = static_cast<int>(clamp_i64(ix, 0, w - 1));
  const int y0 = static_cast<int>(clamp_i64(iy, 0, h - 1));
  const int x1 = static_cast<int>(clamp_i64(ix + 1, 0, w - 1));
  const int y1 = static_cast<int>(clamp_i64(iy + 1, 0, h - 1));
  const int p00 = plane[static_cast<std::ptrdiff_t>(y0) * stride + x0];
  const int p01 = plane[static_cast<std::ptrdiff_t>(y0) * stride + x1];
  const int p10 = plane[static_cast<std::ptrdiff_t>(y1) * stride + x0];
  const int p11 = plane[static_cast<std::ptrdiff_t>(y1) * stride + x1];
  const std::int64_t top = (p00 * (65536 - fx) + p01 * fx) >> 16;
  const std::int64_t bottom = (p10 * (65536 - fx) + p11 * fx) >> 16;
  const std::int64_t value = (top * (65536 - fy) + bottom * fy) >> 16;
  return static_cast<std::uint8_t>(clip3(0, max_value, static_cast<int>(value)));
}

}  // namespace

bool warp_plane_bilinear(const GmParams& params, const std::uint8_t* src, int src_w, int src_h,
                         int src_stride, std::uint8_t* dst, int dst_w, int dst_h, int dst_stride,
                         int bit_depth) {
  if (src == nullptr || dst == nullptr || src_w <= 0 || src_h <= 0 || dst_w <= 0 ||
      dst_h <= 0 || src_stride < src_w || dst_stride < dst_w || bit_depth != 8) {
    return false;
  }
  const int max_value = (1 << bit_depth) - 1;
  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      const std::int64_t x_q16 = static_cast<std::int64_t>(x) << 16;
      const std::int64_t y_q16 = static_cast<std::int64_t>(y) << 16;
      const i128 rx = ((static_cast<i128>(params.p[2]) * x_q16 +
                        static_cast<i128>(params.p[3]) * y_q16) >>
                       16) +
                      params.p[0];
      const i128 ry = ((static_cast<i128>(params.p[4]) * x_q16 +
                        static_cast<i128>(params.p[5]) * y_q16) >>
                       16) +
                      params.p[1];
      dst[static_cast<std::ptrdiff_t>(y) * dst_stride + x] = bilinear_sample(
          src, src_w, src_h, src_stride, to_i64_sat(rx), to_i64_sat(ry),
          max_value);
    }
  }
  return true;
}

std::int64_t warp_plane_sse(const GmParams& params, const std::uint8_t* ref, int ref_w,
                            int ref_h, int ref_stride, const std::uint8_t* cur, int cur_w,
                            int cur_h, int cur_stride, int bit_depth) {
  if (ref == nullptr || cur == nullptr || ref_w <= 0 || ref_h <= 0 || cur_w <= 0 ||
      cur_h <= 0 || ref_stride < ref_w || cur_stride < cur_w) {
    return -1;
  }
  std::int64_t sse = 0;
  for (int y = 0; y < cur_h; ++y) {
    for (int x = 0; x < cur_w; ++x) {
      const std::int64_t x_q16 = static_cast<std::int64_t>(x) << 16;
      const std::int64_t y_q16 = static_cast<std::int64_t>(y) << 16;
      const i128 rx = ((static_cast<i128>(params.p[2]) * x_q16 +
                        static_cast<i128>(params.p[3]) * y_q16) >>
                       16) +
                      params.p[0];
      const i128 ry = ((static_cast<i128>(params.p[4]) * x_q16 +
                        static_cast<i128>(params.p[5]) * y_q16) >>
                       16) +
                      params.p[1];
      const int predicted = bilinear_sample(ref, ref_w, ref_h, ref_stride,
                                            to_i64_sat(rx), to_i64_sat(ry),
                                            (1 << bit_depth) - 1);
      const int source = cur[static_cast<std::ptrdiff_t>(y) * cur_stride + x];
      const int d = predicted - source;
      sse += static_cast<std::int64_t>(d * d);
    }
  }
  return sse;
}

}  // namespace av2enc::inter::gme
