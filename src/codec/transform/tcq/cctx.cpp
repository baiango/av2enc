#include "cctx.h"

#include "tcq_cdf_tables.h"

#include <cmath>
#include <limits>

namespace av2enc::tcq {
namespace {

std::int32_t clip3(std::int32_t low, std::int32_t high,
                   std::int32_t value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

double symbol_bits(const std::uint16_t* cdf, int n, int symbol) noexcept {
  constexpr double kCdfScale = 32768.0;
  constexpr double kImpossible = 1.0e6;
  if (n <= 1) return 0.0;
  if (cdf == nullptr || symbol < 0 || symbol >= n) return kImpossible;
  std::int32_t probability = 0;
  if (symbol == 0) {
    probability = cdf[0];
  } else if (symbol == n - 1) {
    probability = static_cast<std::int32_t>(kCdfScale) - cdf[n - 2];
  } else {
    probability = static_cast<std::int32_t>(cdf[symbol]) - cdf[symbol - 1];
  }
  if (probability <= 0) return kImpossible;
  return -std::log2(static_cast<double>(probability) / kCdfScale);
}

}  // namespace

const char* cctx_type_name(CctxType type) noexcept {
  switch (type) {
    case CctxType::kNone:
      return "CCTX_NONE";
    case CctxType::k45:
      return "CCTX_45";
    case CctxType::k30:
      return "CCTX_30";
    case CctxType::k60:
      return "CCTX_60";
    case CctxType::kMinus45:
      return "CCTX_MINUS45";
    case CctxType::kMinus30:
      return "CCTX_MINUS30";
    case CctxType::kMinus60:
      return "CCTX_MINUS60";
  }
  return "CCTX_INVALID";
}

std::int32_t round2_signed(std::int64_t value, int n) noexcept {
  if (n <= 0) return static_cast<std::int32_t>(value);
  if (value >= 0) {
    return static_cast<std::int32_t>((value + (1LL << (n - 1))) >> n);
  }
  return static_cast<std::int32_t>(
      -((-value + (1LL << (n - 1))) >> n));
}

std::int32_t cctx_dequant_component(std::int32_t u, std::int32_t v, int plane,
                                    CctxType type, int bitdepth) noexcept {
  if (type == CctxType::kNone) return plane == 1 ? u : v;
  const int angle = static_cast<int>(type) - 1;
  if (angle < 0 || angle >= kCctxTypeCount - 1) return plane == 1 ? u : v;
  std::int32_t c_u;
  std::int32_t c_v;
  if (plane == 1) {
    c_u = kCctxMtx[angle][0];
    c_v = -kCctxMtx[angle][1];
  } else {
    c_u = kCctxMtx[angle][1];
    c_v = kCctxMtx[angle][0];
  }
  const std::int64_t value =
      static_cast<std::int64_t>(u) * c_u + static_cast<std::int64_t>(v) * c_v;
  const std::int32_t rounded = round2_signed(value, kCctxPrecBits);
  const std::int32_t low = -(1 << (bitdepth + 7));
  const std::int32_t high = (1 << (bitdepth + 7)) - 1;
  return clip3(low, high, rounded);
}

CctxPair cctx_dequant_pair(std::int32_t u, std::int32_t v, CctxType type,
                           int bitdepth) noexcept {
  CctxPair out;
  out.u = cctx_dequant_component(u, v, 1, type, bitdepth);
  out.v = cctx_dequant_component(u, v, 2, type, bitdepth);
  return out;
}

bool cctx_allowed(bool enable_cctx, bool lossless, bool is420, int chroma_w,
                  int chroma_h) noexcept {
  return enable_cctx && !lossless &&
         (is420 || chroma_w < 32 || chroma_h < 32);
}

double cctx_type_bits(CctxType type) noexcept {
  return symbol_bits(cdf::kCctxTypeCdf, kCctxTypeCount,
                     static_cast<int>(type));
}

void cctx_forward_rotate(double u, double v, CctxType type, double* out_u,
                         double* out_v) noexcept {
  if (type == CctxType::kNone) {
    if (out_u != nullptr) *out_u = u;
    if (out_v != nullptr) *out_v = v;
    return;
  }
  const int angle = static_cast<int>(type) - 1;
  if (angle < 0 || angle >= kCctxTypeCount - 1) {
    if (out_u != nullptr) *out_u = u;
    if (out_v != nullptr) *out_v = v;
    return;
  }
  const double c = static_cast<double>(kCctxMtx[angle][0]);
  const double s = static_cast<double>(kCctxMtx[angle][1]);
  const double denom = c * c + s * s;
  const double scale = 256.0 / denom;
  if (out_u != nullptr) *out_u = (c * u + s * v) * scale;
  if (out_v != nullptr) *out_v = (-s * u + c * v) * scale;
}

bool cctx_enumerate_scalar(std::span<const double> u_coeff,
                           std::span<const double> v_coeff, int w, int h,
                           const CctxConfig& config,
                           std::vector<CctxResult>* candidates,
                           std::string* error) {
  if (candidates == nullptr) {
    if (error != nullptr) *error = "CCTX candidate vector must not be null";
    return false;
  }
  candidates->clear();
  if (w <= 0 || h <= 0 ||
      u_coeff.size() !=
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h) ||
      v_coeff.size() != u_coeff.size()) {
    if (error != nullptr) *error = "CCTX coefficient spans do not match w*h";
    return false;
  }
  if (!std::isfinite(config.quant_step_u) || config.quant_step_u <= 0.0 ||
      !std::isfinite(config.quant_step_v) || config.quant_step_v <= 0.0) {
    if (error != nullptr) *error = "CCTX quant steps must be finite and > 0";
    return false;
  }
  if (config.bitdepth < 8 || config.bitdepth > 16) {
    if (error != nullptr) *error = "CCTX bitdepth must be in [8, 16]";
    return false;
  }
  for (double value : u_coeff) {
    if (!std::isfinite(value)) {
      if (error != nullptr) *error = "CCTX U coefficients must be finite";
      return false;
    }
  }
  for (double value : v_coeff) {
    if (!std::isfinite(value)) {
      if (error != nullptr) *error = "CCTX V coefficients must be finite";
      return false;
    }
  }

  const int count = static_cast<int>(u_coeff.size());
  const int types = config.enabled ? kCctxTypeCount : 1;
  candidates->reserve(static_cast<std::size_t>(types));
  for (int t = 0; t < types; ++t) {
    const CctxType type = static_cast<CctxType>(t);
    CctxResult candidate;
    candidate.type = type;
    candidate.u_levels.assign(static_cast<std::size_t>(count), 0);
    candidate.v_levels.assign(static_cast<std::size_t>(count), 0);
    double distortion = 0.0;
    for (int i = 0; i < count; ++i) {
      double rot_u = 0.0;
      double rot_v = 0.0;
      cctx_forward_rotate(u_coeff[static_cast<std::size_t>(i)],
                          v_coeff[static_cast<std::size_t>(i)], type, &rot_u,
                          &rot_v);
      const double u_q = std::round(rot_u / config.quant_step_u);
      const double v_q = std::round(rot_v / config.quant_step_v);
      const std::int32_t u_level =
          static_cast<std::int32_t>(std::max(
              static_cast<double>(std::numeric_limits<std::int32_t>::min() + 1),
              std::min(
                  static_cast<double>(std::numeric_limits<std::int32_t>::max()),
                  u_q)));
      const std::int32_t v_level =
          static_cast<std::int32_t>(std::max(
              static_cast<double>(std::numeric_limits<std::int32_t>::min() + 1),
              std::min(
                  static_cast<double>(std::numeric_limits<std::int32_t>::max()),
                  v_q)));
      candidate.u_levels[static_cast<std::size_t>(i)] = u_level;
      candidate.v_levels[static_cast<std::size_t>(i)] = v_level;
      if (u_level != 0) candidate.eob_u = i + 1;
      if (v_level != 0) candidate.eob_v = i + 1;

      // Decoder arithmetic: integer dequant, then the §7.14.6 rotation.
      const std::int32_t du = static_cast<std::int32_t>(
          std::llround(static_cast<double>(u_level) * config.quant_step_u));
      const std::int32_t dv = static_cast<std::int32_t>(
          std::llround(static_cast<double>(v_level) * config.quant_step_v));
      const CctxPair out =
          cctx_dequant_pair(du, dv, type, config.bitdepth);
      const double eu =
          u_coeff[static_cast<std::size_t>(i)] - static_cast<double>(out.u);
      const double ev =
          v_coeff[static_cast<std::size_t>(i)] - static_cast<double>(out.v);
      distortion += eu * eu + ev * ev;
    }
    candidate.distortion = distortion;
    candidates->push_back(std::move(candidate));
  }
  return true;
}

bool cctx_quantize_scalar(std::span<const double> u_coeff,
                          std::span<const double> v_coeff, int w, int h,
                          const CctxConfig& config, CctxResult* result,
                          std::string* error) {
  if (result == nullptr) {
    if (error != nullptr) *error = "CCTX result must not be null";
    return false;
  }
  *result = CctxResult{};
  std::vector<CctxResult> candidates;
  if (!cctx_enumerate_scalar(u_coeff, v_coeff, w, h, config, &candidates,
                             error)) {
    return false;
  }
  if (candidates.empty()) {
    if (error != nullptr) *error = "CCTX search produced no candidate";
    return false;
  }
  std::size_t best = 0;
  for (std::size_t i = 1; i < candidates.size(); ++i) {
    if (candidates[i].distortion < candidates[best].distortion) best = i;
  }
  *result = std::move(candidates[best]);
  return true;
}

}  // namespace av2enc::tcq
