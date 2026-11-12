#include "codec/filters/loop_restoration/classes/wiener_fit.h"

#include "codec/filters/loop_restoration/classes/wiener_tables.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace av2enc::restoration::classes {
namespace {

constexpr int kMaxGroups = kWienerNsChromaCoeffs;

int quantize(double v) {
  if (!(v == v)) {
    return 0;
  }
  if (v > 1.0e9) {
    v = 1.0e9;
  } else if (v < -1.0e9) {
    v = -1.0e9;
  }
  if (v >= 0.0) {
    return static_cast<int>(std::floor(v + 0.5));
  }
  return -static_cast<int>(std::floor(-v + 0.5));
}

// Deterministic Gauss-Jordan with partial pivoting.  Ties choose the lowest
// row index.  Returns false when the (ridged) system is numerically singular.
bool solve_system(int n, const double* a_in, const double* b_in,
                  double ridge, double* x) {
  double a[kMaxGroups * kMaxGroups];
  double b[kMaxGroups];
  double trace = 0.0;
  for (int i = 0; i < n; ++i) {
    trace += a_in[i * n + i];
    b[i] = b_in[i];
    for (int j = 0; j < n; ++j) {
      a[i * n + j] = a_in[i * n + j];
    }
  }
  const double diag = ridge * (trace / static_cast<double>(n) + 1.0);
  for (int i = 0; i < n; ++i) {
    a[i * n + i] += diag;
  }

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::fabs(a[col * n + col]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::fabs(a[row * n + col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < 1e-12) {
      return false;
    }
    if (pivot != col) {
      for (int j = col; j < n; ++j) {
        std::swap(a[pivot * n + j], a[col * n + j]);
      }
      std::swap(b[pivot], b[col]);
    }
    const double inv = 1.0 / a[col * n + col];
    for (int j = col; j < n; ++j) {
      a[col * n + j] *= inv;
    }
    b[col] *= inv;
    for (int row = 0; row < n; ++row) {
      if (row == col) {
        continue;
      }
      const double factor = a[row * n + col];
      if (factor == 0.0) {
        continue;
      }
      for (int j = col; j < n; ++j) {
        a[row * n + j] -= factor * a[col * n + j];
      }
      b[row] -= factor * b[col];
    }
  }
  for (int i = 0; i < n; ++i) {
    x[i] = b[i];
  }
  return true;
}

bool block_in_region(const LrCoreFrame& core, int plane, int row, int col,
                     int unit_row, int unit_col, int class_index,
                     const std::uint8_t* subclass_grid, LrBlock* b) {
  if (!lr_block_context(core, plane, row, col, b)) {
    return false;
  }
  if (b->type != kRestoreWienerNs) {
    return false;
  }
  if (unit_row >= 0 &&
      (b->unit_row != unit_row || b->unit_col != unit_col)) {
    return false;
  }
  if (class_index >= 0 && subclass_grid != nullptr &&
      subclass_grid[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(core.mi_cols) +
                    static_cast<std::size_t>(col)] !=
          static_cast<std::uint8_t>(class_index)) {
    return false;
  }
  return true;
}

// Accumulates the group differences for one sample.  `diff[g]` is the sum of
// (source(tap) - m) over every tap that shares coefficient group g, exactly as
// §7.20.3 accumulates them.
void sample_diffs(const LrCoreFrame& core, const LrBlock& b, int plane,
                  int sx, int sy, int m, int diff[kMaxGroups]) {
  for (int i = 0; i < kMaxGroups; ++i) {
    diff[i] = 0;
  }
  const int n_taps = (plane == 0) ? kWienerNsTapsY : kWienerNsTapsUv;
  const std::int8_t(*config)[3] =
      (plane == 0) ? kWienerNsConfigY : kWienerNsConfigUv;
  for (int i = 0; i < n_taps; ++i) {
    const int dy = config[i][0];
    const int dx = config[i][1];
    const int idx = config[i][2];
    diff[idx] +=
        lr_get_source_sample(core, b, plane, sx + dx, sy + dy) - m;
  }
  if (plane > 0) {
    const int m_luma = lr_get_luma_sample(core, b, sx, sy);
    for (int i = 0; i < n_taps; ++i) {
      const int dy = config[i][0];
      const int dx = config[i][1];
      diff[kWienerNsShortCoeffs + i] +=
          lr_get_luma_sample(core, b, sx + dx, sy + dy) - m_luma;
    }
  }
}

struct NormalEquations {
  double a[kMaxGroups * kMaxGroups] = {};
  double b[kMaxGroups] = {};
};

WienerFitStatus gather_normal_equations(
    const LrCoreFrame& core, int plane, int unit_row, int unit_col,
    int class_index, const std::uint8_t* subclass_grid,
    const std::uint8_t* target, int target_stride, NormalEquations* ne,
    double* identity_sse, int* samples) {
  const int width = lr_plane_width(core, plane);
  const int height = lr_plane_height(core, plane);
  const int n_groups = (plane == 0) ? kWienerNsLumaCoeffs : kMaxGroups;
  int count = 0;
  double sse = 0.0;
  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      LrBlock b;
      if (!block_in_region(core, plane, row, col, unit_row, unit_col,
                           class_index, subclass_grid, &b)) {
        continue;
      }
      for (int r = 0; r < b.h; ++r) {
        const int sy = b.y + r;
        if (sy < 0 || sy >= height) {
          continue;
        }
        for (int c = 0; c < b.w; ++c) {
          const int sx = b.x + c;
          if (sx < 0 || sx >= width) {
            continue;
          }
          const int m = lr_get_source_sample(core, b, plane, sx, sy);
          const int t = target[static_cast<std::ptrdiff_t>(sy) *
                                   target_stride +
                               sx];
          const std::int64_t y =
              static_cast<std::int64_t>(t - m) * (1 << kWienerNsPrecBits);
          int diff[kMaxGroups];
          sample_diffs(core, b, plane, sx, sy, m, diff);
          for (int i = 0; i < n_groups; ++i) {
            const double di = static_cast<double>(diff[i]);
            ne->b[i] += di * static_cast<double>(y);
            for (int j = 0; j < n_groups; ++j) {
              ne->a[i * n_groups + j] +=
                  di * static_cast<double>(diff[j]);
            }
          }
          const int e = m - t;
          sse += static_cast<double>(e) * static_cast<double>(e);
          ++count;
        }
      }
    }
  }
  *identity_sse = sse;
  *samples = count;
  return count > 0 ? WienerFitStatus::kOk : WienerFitStatus::kNoSamples;
}

// Builds the reduced linear system for one subset.  When `uv_sym` is set the
// six luma-derived coefficient pairs (6/7, 8/9, ..., 16/17) share one unknown,
// matching the syntax rule that only the first coefficient of each pair is
// signalled.
int build_reduced_system(const NormalEquations& ne, int plane, int subset,
                         bool uv_sym, int n_groups, double* a_out,
                         double* b_out, int col_of_group[kMaxGroups]) {
  for (int i = 0; i < kMaxGroups; ++i) {
    col_of_group[i] = -1;
  }
  int cols = 0;
  for (int j = 0; j < n_groups; ++j) {
    if (!wiener_tap_present(plane, subset, j)) {
      continue;
    }
    if (uv_sym && j >= kWienerNsShortCoeffs) {
      const int pair_base =
          kWienerNsShortCoeffs + ((j - kWienerNsShortCoeffs) / 2) * 2;
      if (col_of_group[pair_base] < 0) {
        col_of_group[pair_base] = cols++;
      }
      col_of_group[j] = col_of_group[pair_base];
    } else {
      col_of_group[j] = cols++;
    }
  }
  for (int i = 0; i < cols * cols; ++i) {
    a_out[i] = 0.0;
  }
  for (int i = 0; i < cols; ++i) {
    b_out[i] = 0.0;
  }
  for (int i = 0; i < n_groups; ++i) {
    if (col_of_group[i] < 0) {
      continue;
    }
    const int ci = col_of_group[i];
    b_out[ci] += ne.b[i];
    for (int j = 0; j < n_groups; ++j) {
      if (col_of_group[j] < 0) {
        continue;
      }
      a_out[ci * cols + col_of_group[j]] += ne.a[i * n_groups + j];
    }
  }
  return cols;
}

double evaluate_sse(const LrCoreFrame& core, int plane, int unit_row,
                    int unit_col, int class_index,
                    const std::uint8_t* subclass_grid,
                    const std::uint8_t* target, int target_stride,
                    const std::int16_t coeffs[kWienerNsCoeffStride]) {
  const int width = lr_plane_width(core, plane);
  const int height = lr_plane_height(core, plane);
  double sse = 0.0;
  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      LrBlock b;
      if (!block_in_region(core, plane, row, col, unit_row, unit_col,
                           class_index, subclass_grid, &b)) {
        continue;
      }
      for (int r = 0; r < b.h; ++r) {
        const int sy = b.y + r;
        if (sy < 0 || sy >= height) {
          continue;
        }
        for (int c = 0; c < b.w; ++c) {
          const int sx = b.x + c;
          if (sx < 0 || sx >= width) {
            continue;
          }
          const int m = lr_get_source_sample(core, b, plane, sx, sy);
          int s = m << kWienerNsPrecBits;
          int diff[kMaxGroups];
          sample_diffs(core, b, plane, sx, sy, m, diff);
          for (int j = 0; j < kMaxGroups; ++j) {
            if (coeffs[j] != 0) {
              s += diff[j] * coeffs[j];
            }
          }
          const int v = clip3(0, (1 << core.bit_depth) - 1,
                              round2(s, kWienerNsPrecBits));
          const int t = target[static_cast<std::ptrdiff_t>(sy) *
                                   target_stride +
                               sx];
          const int e = v - t;
          sse += static_cast<double>(e) * static_cast<double>(e);
        }
      }
    }
  }
  return sse;
}

}  // namespace

const char* wiener_fit_status_string(WienerFitStatus status) {
  switch (status) {
    case WienerFitStatus::kOk:
      return "ok";
    case WienerFitStatus::kInvalidArgument:
      return "invalid argument";
    case WienerFitStatus::kNoSamples:
      return "no samples in the requested region";
    case WienerFitStatus::kSingular:
      return "normal equations are singular";
    case WienerFitStatus::kUnsupportedPlane:
      return "unsupported plane";
  }
  return "unknown";
}

WienerFitStatus wiener_fit_candidates(
    const LrCoreFrame& core, int plane, int unit_row, int unit_col,
    int class_index, const std::uint8_t* subclass_grid,
    const std::uint8_t* target, int target_stride,
    const WienerFitOptions& options, WienerFitResult* result) {
  if (result == nullptr || target == nullptr || plane < 0 || plane > 2 ||
      target_stride < lr_plane_width(core, plane)) {
    return WienerFitStatus::kInvalidArgument;
  }
  if (class_index >= kWienerNsClasses) {
    return WienerFitStatus::kInvalidArgument;
  }
  if (plane != 0 && plane != 1 && plane != 2) {
    return WienerFitStatus::kUnsupportedPlane;
  }
  const int n_groups = (plane == 0) ? kWienerNsLumaCoeffs : kMaxGroups;

  NormalEquations ne;
  double identity_sse = 0.0;
  int samples = 0;
  const WienerFitStatus st = gather_normal_equations(
      core, plane, unit_row, unit_col, class_index, subclass_grid, target,
      target_stride, &ne, &identity_sse, &samples);
  if (st != WienerFitStatus::kOk) {
    return st;
  }

  result->candidates.clear();
  result->identity_sse = identity_sse;
  result->samples = samples;

  const int num_subsets = wiener_num_subsets(plane);
  for (int subset = 0; subset < num_subsets; ++subset) {
    if (options.forced_subset >= 0 && subset != options.forced_subset) {
      continue;
    }
    const bool allow_sym = plane > 0 && subset > 0 && options.try_uv_sym;
    for (int sym = 0; sym <= (allow_sym ? 1 : 0); ++sym) {
      const bool uv_sym = sym != 0;
      double a[kMaxGroups * kMaxGroups];
      double b[kMaxGroups];
      int col_of_group[kMaxGroups];
      const int cols = build_reduced_system(ne, plane, subset, uv_sym,
                                            n_groups, a, b, col_of_group);
      if (cols <= 0) {
        continue;
      }
      double x[kMaxGroups] = {};
      if (!solve_system(cols, a, b, options.ridge, x)) {
        continue;
      }
      WienerFitCandidate cand;
      cand.subset = subset;
      cand.uv_sym = uv_sym;
      for (int j = 0; j < n_groups; ++j) {
        const int c = col_of_group[j];
        if (c < 0) {
          cand.coeffs[j] = 0;
          continue;
        }
        int v = quantize(x[c]);
        v = wiener_ns_clip_coeff(plane != 0, j, v);
        cand.coeffs[j] = static_cast<std::int16_t>(v);
      }
      int present = 0;
      for (int j = 0; j < n_groups; ++j) {
        if (!wiener_tap_present(plane, subset, j)) {
          cand.coeffs[j] = 0;
          continue;
        }
        ++present;
        if (uv_sym && j >= kWienerNsShortCoeffs) {
          cand.coeffs[j + 1] = cand.coeffs[j];
          ++j;
        }
      }
      cand.present_coeffs = present;
      cand.identity_sse = identity_sse;
      cand.samples = samples;
      cand.sse = evaluate_sse(core, plane, unit_row, unit_col, class_index,
                              subclass_grid, target, target_stride,
                              cand.coeffs);
      result->candidates.push_back(cand);
    }
  }
  return result->candidates.empty() ? WienerFitStatus::kSingular
                                    : WienerFitStatus::kOk;
}

double wiener_filter_sse(const LrCoreFrame& core, int plane, int unit_row,
                         int unit_col, const std::uint8_t* target,
                         int target_stride,
                         const std::int16_t coeffs[kWienerNsCoeffStride]) {
  return evaluate_sse(core, plane, unit_row, unit_col, -1, nullptr, target,
                      target_stride, coeffs);
}

double wiener_filter_sse_class(
    const LrCoreFrame& core, int plane, int class_index,
    const std::uint8_t* subclass_grid, const std::uint8_t* target,
    int target_stride, const std::int16_t coeffs[kWienerNsCoeffStride]) {
  return evaluate_sse(core, plane, -1, -1, class_index, subclass_grid, target,
                      target_stride, coeffs);
}

double wiener_frame_filter_sse(
    const LrCoreFrame& core, int plane, int unit_row, int unit_col,
    int num_classes, const std::uint8_t* subclass_grid,
    const std::uint8_t* target, int target_stride,
    const std::int16_t frame_coeffs[kWienerNsClasses]
                                   [kWienerNsCoeffStride]) {
  if (target == nullptr || plane < 0 || plane > 2) {
    return 0.0;
  }
  const int width = lr_plane_width(core, plane);
  const int height = lr_plane_height(core, plane);
  double sse = 0.0;
  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      LrBlock b;
      if (!block_in_region(core, plane, row, col, unit_row, unit_col, -1,
                           nullptr, &b)) {
        continue;
      }
      int cls = 0;
      if (plane == 0 && num_classes > 1 && subclass_grid != nullptr) {
        cls = subclass_grid[static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(core.mi_cols) +
                            static_cast<std::size_t>(col)];
        if (cls < 0 || cls >= num_classes) {
          cls = 0;
        }
      }
      const std::int16_t* coeffs = frame_coeffs[cls];
      for (int r = 0; r < b.h; ++r) {
        const int sy = b.y + r;
        if (sy < 0 || sy >= height) {
          continue;
        }
        for (int c = 0; c < b.w; ++c) {
          const int sx = b.x + c;
          if (sx < 0 || sx >= width) {
            continue;
          }
          const int m = lr_get_source_sample(core, b, plane, sx, sy);
          int s = m << kWienerNsPrecBits;
          int diff[kMaxGroups];
          sample_diffs(core, b, plane, sx, sy, m, diff);
          for (int j = 0; j < kMaxGroups; ++j) {
            if (coeffs[j] != 0) {
              s += diff[j] * coeffs[j];
            }
          }
          const int v = clip3(0, (1 << core.bit_depth) - 1,
                              round2(s, kWienerNsPrecBits));
          const int t = target[static_cast<std::ptrdiff_t>(sy) *
                                   target_stride +
                               sx];
          const int e = v - t;
          sse += static_cast<double>(e) * static_cast<double>(e);
        }
      }
    }
  }
  return sse;
}

const WienerFitCandidate* wiener_best_fit_candidate(
    const WienerFitResult& result) {
  const WienerFitCandidate* best = nullptr;
  for (const WienerFitCandidate& c : result.candidates) {
    if (best == nullptr) {
      best = &c;
      continue;
    }
    if (c.sse < best->sse) {
      best = &c;
      continue;
    }
    if (c.sse > best->sse) {
      continue;
    }
    if (c.subset < best->subset) {
      best = &c;
      continue;
    }
    if (c.subset > best->subset) {
      continue;
    }
    if (c.uv_sym != best->uv_sym) {
      best = c.uv_sym ? best : &c;
      continue;
    }
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      if (c.coeffs[j] != best->coeffs[j]) {
        if (c.coeffs[j] < best->coeffs[j]) {
          best = &c;
        }
        break;
      }
    }
  }
  return best;
}

}  // namespace av2enc::restoration::classes
