#include "cuda/tcq/tcq_host.h"

#include <vector>

#include "cuda/tcq/tcq_device.cuh"
#include "encoder/tcq/parity/parity_hiding.h"

namespace av2enc::cuda::tcq {
namespace {

namespace detail = av2enc::cuda::tcq::detail;

bool set_error(std::string* error, const char* text) {
  if (error != nullptr) *error = text;
  return false;
}

}  // namespace

bool tcq_scan_and_pre_max(int tx_size, int tx_class, int plane,
                          int max_coefficients, int* count, int* w, int* h,
                          std::int32_t* scan, std::int32_t* pre_max,
                          std::string* error) {
  if (count == nullptr || w == nullptr || h == nullptr || scan == nullptr ||
      pre_max == nullptr || max_coefficients <= 0) {
    return set_error(error, "tcq_scan_and_pre_max: null output or capacity");
  }
  if (tx_class < 0 || tx_class > 2) {
    return set_error(error, "tcq_scan_and_pre_max: tx_class must be 0..2");
  }
  av2enc::tcq::parity::Geometry geometry;
  if (!av2enc::tcq::parity::make_geometry(
          tx_size, static_cast<av2enc::tcq::parity::TxClass>(tx_class), plane,
          &geometry, error)) {
    return false;
  }
  const int area = geometry.w * geometry.h;
  if (area > max_coefficients) {
    return set_error(error,
                     "tcq_scan_and_pre_max: block exceeds max_coefficients");
  }
  std::vector<std::int32_t> host_scan(static_cast<std::size_t>(area), 0);
  if (!av2enc::tcq::parity::build_scan(
          geometry, std::span<std::int32_t>(host_scan), error)) {
    return false;
  }
  for (int c = 0; c < area; ++c) {
    const std::int32_t pos = host_scan[static_cast<std::size_t>(c)];
    const int row = pos / geometry.w;
    const int col = pos % geometry.w;
    scan[c] = pos;
    pre_max[c] =
        av2enc::tcq::parity::level_limits(row, col, geometry.tx_class, plane)
            .pre_max;
  }
  *count = area;
  *w = geometry.w;
  *h = geometry.h;
  return true;
}

bool tcq_fill_rate_tables(const TcqJob& job, const double* original,
                          const std::int32_t* pre_max,
                          const std::int32_t* level_context,
                          const av2enc::tcq::RateModel& model,
                          double* normal_rate, double* eob_coeff_rate,
                          double* block_eob_rate, std::string* error) {
  if (model.coefficient_rate == nullptr || model.eob_rate == nullptr) {
    return set_error(error, "tcq_fill_rate_tables: rate callbacks must be set");
  }
  if (original == nullptr || pre_max == nullptr || level_context == nullptr ||
      normal_rate == nullptr || eob_coeff_rate == nullptr ||
      block_eob_rate == nullptr) {
    return set_error(error, "tcq_fill_rate_tables: null buffer");
  }
  const std::int32_t count = job.num_coefficients;
  const std::int32_t levels = job.table_levels;
  if (count < 0 || levels <= 0) {
    return set_error(error, "tcq_fill_rate_tables: bad job geometry");
  }
  if (!detail::tcq_isfinite(job.quant_step) || !(job.quant_step > 0.0)) {
    return set_error(error, "tcq_fill_rate_tables: quant_step must be finite "
                            "and > 0");
  }
  for (std::int32_t c = 0; c < count; ++c) {
    if (!detail::tcq_isfinite(original[c])) {
      return set_error(error, "tcq_fill_rate_tables: non-finite coefficient");
    }
    const std::int32_t sign = original[c] < 0.0 ? -1 : 1;
    const std::int32_t coefficient_pre_max = pre_max[c] > 0 ? pre_max[c] : 6;
    std::int32_t bound = 0;
    bool clipped = false;
    if (!detail::tcq_level_bound(original[c], job.quant_step, job.level_cap,
                                 job.allow_zero_to_one != 0, &bound,
                                 &clipped)) {
      return set_error(error, "tcq_fill_rate_tables: non-finite level bound");
    }
    (void)clipped;
    if (bound >= levels) {
      return set_error(error,
                       "tcq_fill_rate_tables: level bound exceeds table_levels");
    }
    for (std::int32_t s = 0; s < detail::kStateCount; ++s) {
      for (std::int32_t q = 0; q <= bound; ++q) {
        const bool nonzero = q > 0;
        const std::int32_t syntax_level =
            q == 0 ? 0 : detail::tcq_syntax_level_for(q, coefficient_pre_max);
        const std::int32_t q0 = nonzero ? detail::tcq_state_q0(s) : 0;
        const std::int32_t magnitude = nonzero ? 2 * q - q0 : 0;
        const std::int32_t base = nonzero ? sign * q : 0;
        const std::int32_t quant = nonzero ? sign * magnitude : 0;
        const std::int32_t next =
            detail::tcq_next_state(s, syntax_level & 1);
        const std::size_t index =
            static_cast<std::size_t>((c * detail::kStateCount + s) * levels +
                                     q);
        const double normal = model.coefficient_rate(
            model.context, c, s, base, next, quant, level_context[c], 0);
        const double eob = model.coefficient_rate(
            model.context, c, s, base, next, quant, level_context[c], 1);
        if (!detail::tcq_rate_valid(normal) ||
            !detail::tcq_rate_valid(eob)) {
          return set_error(error,
                           "tcq_fill_rate_tables: non-finite or negative rate");
        }
        normal_rate[index] = normal;
        eob_coeff_rate[index] = eob;
      }
    }
  }
  for (std::int32_t e = 0; e <= count; ++e) {
    const double rate = model.eob_rate(model.context, e, count);
    if (!detail::tcq_rate_valid(rate)) {
      return set_error(error,
                       "tcq_fill_rate_tables: non-finite or negative EOB rate");
    }
    block_eob_rate[e] = rate;
  }
  return true;
}

}  // namespace av2enc::cuda::tcq
