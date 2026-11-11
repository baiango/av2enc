#include "encoder/filters/lr_chroma/chroma_lr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "encoder/filters/restoration/classes/wiener_fit.h"
#include "encoder/filters/restoration/classes/wiener_tables.h"

namespace av2enc::lr_chroma {
namespace {

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

bool plane_is_chroma(int plane) { return plane == 1 || plane == 2; }

int chroma_index(int plane) { return plane - 1; }

struct FitCandidate {
  ChromaFilter filter;
  double sse = 0.0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Unit sizing
// ---------------------------------------------------------------------------

bool chroma_select_unit_size(int desired_size, int sb_size, int sub_x,
                             int sub_y, ChromaUnitSize* out,
                             std::string* error) {
  if (out == nullptr) return fail(error, "null ChromaUnitSize");
  if (sb_size != 64 && sb_size != 128 && sb_size != 256) {
    return fail(error, "unsupported superblock size for chroma LR");
  }
  if (sub_x < 0 || sub_x > 1 || sub_y < 0 || sub_y > 1) {
    return fail(error, "unsupported chroma subsampling");
  }
  const int base = rest::kRestorationTileSizeMax >> std::max(sub_x, sub_y);
  ChromaUnitSize chosen;
  chosen.sb_size = sb_size;
  chosen.sub_x = sub_x;
  chosen.sub_y = sub_y;
  bool found = false;
  for (int shift = 0; shift <= 3; ++shift) {
    // Mirror write_lr_size_flags(): which shifts a superblock can signal.
    if (sb_size == 256 && shift > 1) continue;
    if (sb_size == 128 && shift == 3) continue;
    const int size = base >> shift;
    if (size != desired_size) continue;
    // §6.17.7.7: at least one stripe for chroma.
    if (size < (64 >> sub_y)) continue;
    chosen.shift = shift;
    chosen.size = size;
    chosen.half = shift == 1;
    chosen.max_size = shift == 0;
    chosen.quarter = shift == 2;
    found = true;
    break;
  }
  if (!found) {
    return fail(error, "chroma unit size " + std::to_string(desired_size) +
                           " is not representable for sb_size " +
                           std::to_string(sb_size));
  }
  *out = chosen;
  return true;
}

bool chroma_write_unit_size(BitWriter& writer, const ChromaUnitSize& size,
                            std::string* error) {
  if (size.shift < 0 || size.shift > 3) {
    return fail(error, "chroma shift outside 0..3");
  }
  const int sb = size.sb_size;
  if (sb == 256) {
    if (size.shift != 0 && size.shift != 1) {
      return fail(error, "256x256 SB supports chroma shifts 0 and 1");
    }
    writer.put_bit(size.shift == 1);
    return true;
  }
  writer.put_bit(size.shift == 1);  // lr_chroma_use_half_size
  if (size.shift == 1) return true;
  writer.put_bit(size.shift == 0);  // lr_chroma_use_max_size
  if (size.shift == 0) return true;
  if (sb == 128) {
    if (size.shift != 2) {
      return fail(error, "128x128 SB supports chroma shifts 0, 1 and 2");
    }
    return true;
  }
  if (sb != 64) return fail(error, "unsupported superblock size");
  if (size.shift != 2 && size.shift != 3) {
    return fail(error, "64x64 SB supports chroma shifts 0..3");
  }
  writer.put_bit(size.shift == 2);  // lr_chroma_use_quarter_size
  return true;
}

bool chroma_read_unit_size(classes::WienerBitReader& reader, int sb_size,
                           int sub_x, int sub_y, ChromaUnitSize* size,
                           std::string* error) {
  if (size == nullptr) return fail(error, "null ChromaUnitSize");
  bool half = false;
  if (!reader.read_bit(&half)) return fail(error, "truncated chroma size flag");
  int shift = -1;
  if (half) {
    shift = 1;
  } else if (sb_size == 256) {
    shift = 0;
  } else {
    bool max_size = false;
    if (!reader.read_bit(&max_size)) {
      return fail(error, "truncated chroma max size flag");
    }
    if (max_size) {
      shift = 0;
    } else if (sb_size == 128) {
      shift = 2;
    } else if (sb_size == 64) {
      bool quarter = false;
      if (!reader.read_bit(&quarter)) {
        return fail(error, "truncated chroma quarter size flag");
      }
      shift = quarter ? 2 : 3;
    } else {
      return fail(error, "unsupported superblock size");
    }
  }
  const int base = rest::kRestorationTileSizeMax >> std::max(sub_x, sub_y);
  size->sb_size = sb_size;
  size->sub_x = sub_x;
  size->sub_y = sub_y;
  size->shift = shift;
  size->size = base >> shift;
  size->half = shift == 1;
  size->max_size = shift == 0;
  size->quarter = shift == 2;
  return true;
}

int chroma_count_units(int unit_size, int frame_size) {
  return rest::restoration_count_units(unit_size, frame_size);
}

void ChromaUnitGrid::index(int unit_row, int unit_col, std::size_t* at) const {
  if (at != nullptr) {
    *at = static_cast<std::size_t>(unit_row) *
              static_cast<std::size_t>(unit_cols) +
          static_cast<std::size_t>(unit_col);
  }
}

bool chroma_make_unit_grid(int plane_width, int plane_height, int unit_size,
                           ChromaUnitGrid* grid, std::string* error) {
  if (grid == nullptr) return fail(error, "null ChromaUnitGrid");
  if (plane_width <= 0 || plane_height <= 0 || unit_size <= 0) {
    return fail(error, "invalid chroma grid geometry");
  }
  if ((unit_size & (unit_size - 1)) != 0) {
    return fail(error, "chroma unit size is not a power of two");
  }
  grid->plane_width = plane_width;
  grid->plane_height = plane_height;
  grid->unit_size = unit_size;
  grid->unit_cols = chroma_count_units(unit_size, plane_width);
  grid->unit_rows = chroma_count_units(unit_size, plane_height);
  return true;
}

// ---------------------------------------------------------------------------
// Filter model
// ---------------------------------------------------------------------------

int chroma_present_coeffs(int subset) {
  switch (subset) {
    case 0:
      return 6;
    case 1:
      return 10;
    case 2:
      return rest::kWienerNsChromaCoeffs;
    default:
      return 0;
  }
}

int chroma_coded_coeffs(int subset, bool uv_sym) {
  const int present = chroma_present_coeffs(subset);
  if (!uv_sym || subset == 0) return present;
  int coded = 0;
  for (int j = 0; j < present; ++j) {
    ++coded;
    if (j >= rest::kWienerNsShortCoeffs &&
        (j - rest::kWienerNsShortCoeffs) % 2 == 0) {
      // The next value is mirrored, so it is not coded.
      ++j;
    }
  }
  return coded;
}

bool chroma_filter_fits_subset(const ChromaFilter& filter, int subset) {
  if (subset < 0 || subset > 2) return false;
  const int present = chroma_present_coeffs(subset);
  for (int j = present; j < rest::kWienerNsChromaCoeffs; ++j) {
    if (filter.coeffs[j] != 0) return false;
  }
  if (filter.uv_sym) {
    if (subset == 0) return false;
    for (int j = rest::kWienerNsShortCoeffs;
         j + 1 < rest::kWienerNsChromaCoeffs; j += 2) {
      if (j < present && filter.coeffs[j] != filter.coeffs[j + 1]) {
        return false;
      }
    }
  }
  return true;
}

int chroma_derive_subset(const ChromaFilter& filter) {
  for (int subset = 0; subset <= 2; ++subset) {
    if (chroma_filter_fits_subset(filter, subset)) return subset;
  }
  return -1;
}

void chroma_zero_absent_taps(ChromaFilter* filter) {
  if (filter == nullptr) return;
  const int present = chroma_present_coeffs(filter->subset);
  for (int j = present; j < rest::kWienerNsChromaCoeffs; ++j) {
    filter->coeffs[j] = 0;
  }
}

bool chroma_mirror_filter(ChromaFilter* filter, std::string* error) {
  if (filter == nullptr) return fail(error, "null ChromaFilter");
  if (filter->subset <= 0 || filter->subset > 2) {
    return fail(error, "uv_sym requires chroma subset > 0");
  }
  const int present = chroma_present_coeffs(filter->subset);
  for (int j = rest::kWienerNsShortCoeffs; j + 1 < present; j += 2) {
    const int sum = static_cast<int>(filter->coeffs[j]) +
                    static_cast<int>(filter->coeffs[j + 1]);
    const int v = sum >= 0 ? (sum + 1) / 2 : -((-sum + 1) / 2);
    filter->coeffs[j] = static_cast<std::int16_t>(v);
    filter->coeffs[j + 1] = static_cast<std::int16_t>(v);
  }
  filter->uv_sym = true;
  return true;
}

bool chroma_filter_validate(const ChromaFilter& filter, std::string* error) {
  if (filter.subset < 0 || filter.subset > 2) {
    return fail(error, "chroma subset outside 0..2");
  }
  if (filter.uv_sym && filter.subset == 0) {
    return fail(error, "uv_sym is not coded for chroma subset 0");
  }
  if (!chroma_filter_fits_subset(filter, filter.subset)) {
    return fail(error,
                "chroma filter has non-zero coefficients outside its subset "
                "or non-mirrored uv_sym pairs");
  }
  for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
    if (classes::kWienerNsTapsPresent[1][filter.subset][j] == 0) {
      continue;
    }
    const int min = rest::kWienerNsTapsMin[1][j];
    const int max = rest::wiener_ns_coeff_max(true, j);
    if (filter.coeffs[j] < min || filter.coeffs[j] > max) {
      return fail(error, "chroma coefficient " + std::to_string(j) +
                             " is outside [" + std::to_string(min) + "," +
                             std::to_string(max) + "]");
    }
  }
  if (filter.use_bank_k < -1) {
    return fail(error, "use_bank_k outside the legal range");
  }
  return true;
}

namespace {

// Per-sample §7.20.3 differences.  Groups 0..5 are the chroma tap groups from
// Wiener_Ns_Config_Uv[][2]; groups 6..17 are the twelve chroma-from-luma taps
// (coefficient i + 6 uses config tap i).  This is the layout the decoder uses;
// the generic wiener_fit helper only fills the chroma groups, which is why
// this module carries its own fit.
void chroma_sample_diffs(const rest::LrCoreFrame& core,
                         const rest::LrBlock& block, int plane, int x, int y,
                         int m, int out[rest::kWienerNsChromaCoeffs]) {
  for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) out[j] = 0;
  const int luma_center = rest::lr_get_luma_sample(core, block, x, y);
  for (int i = 0; i < rest::kWienerNsTapsUv; ++i) {
    const int dy = rest::kWienerNsConfigUv[i][0];
    const int dx = rest::kWienerNsConfigUv[i][1];
    const int idx = rest::kWienerNsConfigUv[i][2];
    out[idx] += rest::lr_get_source_sample(core, block, plane, x + dx, y + dy) -
                m;
    out[rest::kWienerNsShortCoeffs + i] +=
        rest::lr_get_luma_sample(core, block, x + dx, y + dy) - luma_center;
  }
}

double eval_chroma_region(const rest::LrCoreFrame& core, int plane,
                          int unit_row, int unit_col,
                          const std::uint8_t* target, int target_stride,
                          const std::int16_t* coeffs) {
  const int width = rest::lr_plane_width(core, plane);
  const int height = rest::lr_plane_height(core, plane);
  double sse = 0.0;
  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      rest::LrBlock block;
      if (!rest::lr_block_context(core, plane, row, col, &block)) continue;
      if (block.type != rest::kRestoreWienerNs) continue;
      if (block.unit_row != unit_row || block.unit_col != unit_col) continue;
      for (int r = 0; r < block.h; ++r) {
        const int sy = block.y + r;
        if (sy < 0 || sy >= height) continue;
        for (int c = 0; c < block.w; ++c) {
          const int sx = block.x + c;
          if (sx < 0 || sx >= width) continue;
          const int m = rest::lr_get_source_sample(core, block, plane, sx, sy);
          int diff[rest::kWienerNsChromaCoeffs];
          chroma_sample_diffs(core, block, plane, sx, sy, m, diff);
          int s = m << rest::kWienerNsPrecBits;
          for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
            s += diff[j] * coeffs[j];
          }
          const int v = rest::clip3(0, 255, rest::round2(s, rest::kWienerNsPrecBits));
          const int t = target[static_cast<std::ptrdiff_t>(sy) * target_stride +
                               sx];
          const int e = v - t;
          sse += static_cast<double>(e) * static_cast<double>(e);
        }
      }
    }
  }
  return sse;
}

int quantize_coeff(double v) {
  if (!(v == v)) return 0;
  if (v > 1.0e9) v = 1.0e9;
  if (v < -1.0e9) v = -1.0e9;
  return v >= 0.0 ? static_cast<int>(std::floor(v + 0.5))
                  : -static_cast<int>(std::floor(-v + 0.5));
}

bool solve_normal_equations(int n, double ata[18][18], double atb[18],
                            double ridge, double* x) {
  double a[18][18];
  double b[18];
  double trace = 0.0;
  for (int i = 0; i < n; ++i) {
    trace += ata[i][i];
    b[i] = atb[i];
    for (int j = 0; j < n; ++j) a[i][j] = ata[i][j];
  }
  const double diag = ridge * (trace / static_cast<double>(n) + 1.0);
  for (int i = 0; i < n; ++i) a[i][i] += diag;
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double best = std::fabs(a[col][col]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::fabs(a[row][col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (best < 1e-12) return false;
    if (pivot != col) {
      for (int j = col; j < n; ++j) std::swap(a[pivot][j], a[col][j]);
      std::swap(b[pivot], b[col]);
    }
    const double inv = 1.0 / a[col][col];
    for (int j = col; j < n; ++j) a[col][j] *= inv;
    b[col] *= inv;
    for (int row = 0; row < n; ++row) {
      if (row == col) continue;
      const double f = a[row][col];
      if (f == 0.0) continue;
      for (int j = col; j < n; ++j) a[row][j] -= f * a[col][j];
      b[row] -= f * b[col];
    }
  }
  for (int i = 0; i < n; ++i) x[i] = b[i];
  return true;
}

// Fits one subset.  `groups` lists the coefficient groups present in the
// subset, in ascending order.
void fit_subset(const rest::LrCoreFrame& core, int plane, int unit_row,
                int unit_col, const std::uint8_t* target, int target_stride,
                const int* groups, int group_count, double ridge,
                std::int16_t out[rest::kWienerNsChromaCoeffs]) {
  for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) out[j] = 0;
  double ata[18][18] = {};
  double atb[18] = {};
  int n = 0;
  const int width = rest::lr_plane_width(core, plane);
  const int height = rest::lr_plane_height(core, plane);
  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      rest::LrBlock block;
      if (!rest::lr_block_context(core, plane, row, col, &block)) continue;
      if (block.type != rest::kRestoreWienerNs) continue;
      if (block.unit_row != unit_row || block.unit_col != unit_col) continue;
      for (int r = 0; r < block.h; ++r) {
        const int sy = block.y + r;
        if (sy < 0 || sy >= height) continue;
        for (int c = 0; c < block.w; ++c) {
          const int sx = block.x + c;
          if (sx < 0 || sx >= width) continue;
          const int m = rest::lr_get_source_sample(core, block, plane, sx, sy);
          int diff[rest::kWienerNsChromaCoeffs];
          chroma_sample_diffs(core, block, plane, sx, sy, m, diff);
          // The design matrix is in the <<WIENER_NS_PREC_BITS domain.
          const int t =
              (target[static_cast<std::ptrdiff_t>(sy) * target_stride + sx] -
               m)
              << rest::kWienerNsPrecBits;
          for (int i = 0; i < group_count; ++i) {
            const int gi = groups[i];
            const double di = static_cast<double>(diff[gi]);
            atb[i] += di * t;
            for (int j = 0; j < group_count; ++j) {
              ata[i][j] += di * static_cast<double>(diff[groups[j]]);
            }
          }
          ++n;
        }
      }
    }
  }
  if (n == 0) return;
  double solution[18] = {};
  if (!solve_normal_equations(group_count, ata, atb, ridge, solution)) return;
  for (int i = 0; i < group_count; ++i) {
    const int gi = groups[i];
    const int min = rest::kWienerNsTapsMin[1][gi];
    const int max = rest::wiener_ns_coeff_max(true, gi);
    int v = quantize_coeff(solution[i]);
    v = std::max(min, std::min(max, v));
    out[gi] = static_cast<std::int16_t>(v);
  }
}

}  // namespace

double chroma_filter_sse(const rest::LrCoreFrame& core, int plane, int unit_row,
                         int unit_col, const std::uint8_t* target,
                         int target_stride, const ChromaFilter& filter) {
  ChromaFilter copy = filter;
  chroma_zero_absent_taps(&copy);
  return eval_chroma_region(core, plane, unit_row, unit_col, target,
                            target_stride, copy.coeffs);
}

// ---------------------------------------------------------------------------
// Decision helpers
// ---------------------------------------------------------------------------

namespace {

void fill_core(const ChromaLrInput& input, int plane, int unit_size,
               std::uint8_t* dst, int dst_stride,
               std::vector<std::uint8_t>* unit_type,
               std::vector<std::int16_t>* unit_coeffs, rest::LrCoreFrame* core,
               std::string* error) {
  *core = rest::LrCoreFrame{};
  core->mi_cols = input.mi_cols;
  core->mi_rows = input.mi_rows;
  core->sub_x = input.sub_x;
  core->sub_y = input.sub_y;
  core->bit_depth = 8;
  core->cfl_ds_filter_index = input.cfl_ds_filter_index;
  const int chroma = chroma_index(plane);
  const int width = input.plane_width[chroma];
  const int height = input.plane_height[chroma];
  const int idx = chroma_index(plane);
  // Luma plane (§7.20.3 chroma-from-luma taps read CurrFrame/CdefFrame[0]).
  core->plane[0].curr = input.luma_curr;
  core->plane[0].curr_stride = input.luma_curr_stride;
  core->plane[0].cdef = input.luma_cdef;
  core->plane[0].cdef_stride = input.luma_cdef_stride;
  core->plane[0].dst = nullptr;
  core->plane[0].dst_stride = 0;
  core->unit_size[0] = 64;
  core->frame_type[0] = rest::kRestoreNone;
  rest::LrCorePlane& p = core->plane[plane];
  p.curr = input.curr[idx];
  p.curr_stride = input.curr_stride[idx];
  p.cdef = input.cdef[idx];
  p.cdef_stride = input.cdef_stride[idx];
  p.dst = dst;
  p.dst_stride = dst_stride;
  p.unit_type = unit_type->data();
  p.unit_coeffs = unit_coeffs->data();
  p.unit_cols = chroma_count_units(unit_size, width);
  p.unit_rows = chroma_count_units(unit_size, height);
  core->unit_size[plane] = unit_size;
  core->frame_type[plane] = rest::kRestoreWienerNs;
  (void)error;
}

std::vector<FitCandidate> fit_unit(const ChromaLrInput& input,
                                   const ChromaDecision& decision, int plane,
                                   int unit_row, int unit_col,
                                   const ChromaLrOptions& options,
                                   std::string* error) {
  std::vector<FitCandidate> out;
  const int chroma = chroma_index(plane);
  const int width = input.plane_width[chroma];
  const int height = input.plane_height[chroma];
  std::vector<std::uint8_t> unit_type(
      static_cast<std::size_t>(decision.grid[chroma].count()),
      static_cast<std::uint8_t>(rest::kRestoreWienerNs));
  std::vector<std::int16_t> unit_coeffs(
      static_cast<std::size_t>(decision.grid[chroma].count()) *
          rest::kWienerNsCoeffStride,
      0);
  rest::LrCoreFrame core;
  std::vector<std::uint8_t> dummy(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  fill_core(input, plane, decision.unit_size, dummy.data(), width, &unit_type,
            &unit_coeffs, &core, error);

  static const int kGroups0[] = {0, 1, 2, 3, 4, 5};
  static const int kGroups1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  static const int kGroups2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                 10, 11, 12, 13, 14, 15, 16, 17};
  for (int subset = 0; subset <= 2; ++subset) {
    if (options.forced_subset >= 0 && subset != options.forced_subset) continue;
    const int* groups = kGroups2;
    int count = 18;
    if (subset == 0) {
      groups = kGroups0;
      count = 6;
    } else if (subset == 1) {
      groups = kGroups1;
      count = 10;
    }
    std::int16_t coeffs[rest::kWienerNsChromaCoeffs] = {};
    fit_subset(core, plane, unit_row, unit_col, input.source[chroma],
               input.source_stride[chroma], groups, count, 1e-6, coeffs);
    ChromaFilter base;
    base.subset = subset;
    base.uv_sym = false;
    for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
      base.coeffs[j] = coeffs[j];
    }
    chroma_zero_absent_taps(&base);
    if (chroma_filter_validate(base, error)) {
      FitCandidate entry;
      entry.filter = base;
      entry.sse = chroma_filter_sse(core, plane, unit_row, unit_col,
                                    input.source[chroma],
                                    input.source_stride[chroma], base);
      out.push_back(entry);
    }
    if (options.allow_uv_sym && subset > 0) {
      ChromaFilter sym = base;
      std::string mirror_error;
      if (chroma_mirror_filter(&sym, &mirror_error)) {
        chroma_zero_absent_taps(&sym);
        if (chroma_filter_validate(sym, error)) {
          FitCandidate entry;
          entry.filter = sym;
          entry.sse = chroma_filter_sse(core, plane, unit_row, unit_col,
                                        input.source[chroma],
                                        input.source_stride[chroma], sym);
          out.push_back(entry);
        }
      }
    }
  }
  if (out.empty()) {
    fail(error, "chroma fit produced no candidate");
    return out;
  }
  std::sort(out.begin(), out.end(),
            [](const FitCandidate& a, const FitCandidate& b) {
              if (a.sse != b.sse) return a.sse < b.sse;
              const int ca =
                  chroma_coded_coeffs(a.filter.subset, a.filter.uv_sym);
              const int cb =
                  chroma_coded_coeffs(b.filter.subset, b.filter.uv_sym);
              if (ca != cb) return ca < cb;
              return a.filter.subset < b.filter.subset;
            });
  return out;
}

// Finds a bank entry that exactly reproduces `filter` (all 18 coefficients;
// merged units copy the whole bank entry, including taps that the subset
// would otherwise zero).
int find_merge_bank(const fs::LrState& state, int plane,
                    const ChromaFilter& filter) {
  const int size = state.bank_size[plane][0];
  if (size <= 0) return -1;
  const int ptr = state.bank_ptr[plane][0];
  for (int k = 0; k < size; ++k) {
    const int slot = (ptr - k + 4) & 3;
    bool same = true;
    for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
      if (state.ref_bank[plane][0][slot][j] != filter.coeffs[j]) {
        same = false;
        break;
      }
    }
    if (same) return k;
  }
  return -1;
}

bool write_chroma_unit_impl(av2::Encoder& encoder, fs::LrState& state,
                            int plane, int unit_row, int unit_col, bool on,
                            const ChromaFilter& filter, std::string* error);
bool read_chroma_unit_impl(av2::Decoder& decoder, fs::LrState& state,
                           int plane, int unit_row, int unit_col,
                           ChromaUnitPlan* plan, std::string* error);

// Applies a selected unit decision to the simulation state (updates banks and
// CDFs).  The bytes are discarded; only state matters.
bool commit_unit(fs::LrState* state, int plane, int unit_row, int unit_col,
                 const ChromaUnitPlan& plan, std::string* error) {
  std::vector<std::uint8_t> buffer(1 << 16);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return fail(error, "entropy init failed");
  }
  encoder.disable_cdf_update = 1;
  if (!write_chroma_unit_impl(encoder, *state, plane, unit_row, unit_col,
                              plan.on, plan.filter, error)) {
    return false;
  }
  std::size_t bytes = 0;
  if (encoder.exit_symbol(&bytes) != av2::Status::kOk) {
    return fail(error, "entropy exit failed");
  }
  return true;
}

struct PlaneFit {
  int plane = 1;
  int row = 0;
  int col = 0;
  std::vector<FitCandidate> candidates;
  double identity_sse = 0.0;
};

}  // namespace

// ---------------------------------------------------------------------------
// chroma_measure_unit_bytes / writers / readers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Exact §5.20.10.6 per-unit chroma writer/reader
// ---------------------------------------------------------------------------
//
// This is deliberately a chroma-specific implementation: the shared
// filters_stream writer requires every subset>0 filter to be pair-symmetric
// even when `wiener_ns_uv_sym == 0`, which is stricter than the spec.  The
// spec only mirrors pairs when the bit is 1, so an encoder that wants the
// extra freedom (asymmetric 10/18-tap chroma filters) needs this path.

namespace {

bool encode_raw_bit(av2::Encoder& encoder, bool bit, std::string* error) {
  if (encoder.encode_bool(bit) != av2::Status::kOk) {
    return fail(error, "entropy encode_bool failed");
  }
  return true;
}

bool encode_sym(av2::Encoder& encoder, std::uint16_t* cdf, int n, int symbol,
                std::string* error) {
  if (encoder.encode_symbol(cdf, n, symbol) != av2::Status::kOk) {
    return fail(error, "entropy encode_symbol failed");
  }
  return true;
}

bool decode_sym(av2::Decoder& decoder, std::uint16_t* cdf, int n, int* symbol,
                std::string* error) {
  if (decoder.decode_symbol(cdf, n, symbol) != av2::Status::kOk) {
    return fail(error, "entropy decode_symbol failed");
  }
  return true;
}

bool decode_raw_bit(av2::Decoder& decoder, bool* bit, std::string* error) {
  if (decoder.read_bool(bit) != av2::Status::kOk) {
    return fail(error, "entropy read_bool failed");
  }
  return true;
}

bool write_chroma_unit_impl(av2::Encoder& encoder, fs::LrState& state,
                            int plane, int unit_row, int unit_col, bool on,
                            const ChromaFilter& filter, std::string* error) {
  if (on) {
    ChromaFilter check = filter;
    chroma_zero_absent_taps(&check);
    if (!chroma_filter_validate(check, error)) return false;
  }
  const int frame_type = state.params.frame_type[plane];
  int type = rest::kRestoreNone;
  if (frame_type == rest::kRestoreWienerNs) {
    type = on ? rest::kRestoreWienerNs : rest::kRestoreNone;
    if (!encode_sym(encoder, &state.use_wiener_ns[0], 2, on ? 1 : 0, error)) {
      return false;
    }
  } else if (frame_type == rest::kRestoreNone) {
    if (on) return fail(error, "chroma unit on for a RESTORE_NONE plane");
    type = rest::kRestoreNone;
  } else {
    return fail(error,
                "chroma LR supports RESTORE_WIENER_NONSEP or RESTORE_NONE");
  }

  const std::size_t index =
      static_cast<std::size_t>(unit_row) *
          static_cast<std::size_t>(state.unit_cols[plane]) +
      static_cast<std::size_t>(unit_col);
  state.unit_type[plane][index] = static_cast<std::uint8_t>(type);
  if (type != rest::kRestoreWienerNs) return true;

  if (state.params.frame_filters_on[plane]) {
    const std::size_t base = index * rest::kWienerNsCoeffStride;
    for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.frame_coeffs[plane][0][j];
    }
    return true;
  }

  ChromaFilter effective = filter;
  chroma_zero_absent_taps(&effective);

  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];

  if (!encode_raw_bit(encoder, effective.merged, error)) return false;
  int k_sel = effective.use_bank_k;
  if (k_sel < 0) k_sel = old_size <= 1 ? 0 : old_size - 1;
  if (k_sel < 0 || k_sel >= std::max(1, old_size)) {
    return fail(error, "use_bank_k outside the current bank");
  }
  for (int k = 0; k < old_size - 1; ++k) {
    const bool bit = k == k_sel;
    if (!encode_raw_bit(encoder, bit, error)) return false;
    if (bit) break;
  }
  const int ref_slot = (old_ptr - k_sel + classes::kLrBankSize) % classes::kLrBankSize;

  if (effective.merged) {
    if (state.bank_size[plane][cls] == 0) state.bank_size[plane][cls] = 1;
    const std::size_t base = index * rest::kWienerNsCoeffStride;
    for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.ref_bank[plane][cls][ref_slot][j];
    }
    return true;
  }

  int new_ptr = old_ptr;
  int new_size = old_size;
  if (old_size < classes::kLrBankSize) {
    new_ptr = old_size;
    new_size = old_size + 1;
  } else {
    new_ptr = (old_ptr + 1) % classes::kLrBankSize;
  }

  for (int subset = 0; subset < 2; ++subset) {
    const bool len = subset < effective.subset;
    if (!encode_sym(encoder, &state.wiener_ns_length[1][0], 2, len ? 1 : 0,
                    error)) {
      return false;
    }
    if (!len) break;
  }
  bool uv_sym = false;
  if (effective.subset > 0) {
    uv_sym = effective.uv_sym;
    if (!encode_sym(encoder, &state.wiener_ns_uv_sym[0], 2, uv_sym ? 1 : 0,
                    error)) {
      return false;
    }
  }

  state.bank_ptr[plane][cls] = new_ptr;
  state.bank_size[plane][cls] = new_size;
  const std::size_t base = index * rest::kWienerNsCoeffStride;
  for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
    const int min = rest::kWienerNsTapsMin[1][j];
    const int k = rest::kWienerNsTapsK[1][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (classes::kWienerNsTapsPresent[1][effective.subset][j] != 0) {
      const int target = effective.coeffs[j];
      if (target < min || target >= min + (1 << k)) {
        return fail(error, "chroma coefficient outside its tap range");
      }
      if (!classes::wiener_encode_signed_4part(
              encoder, state.wiener_ns_base, min, k,
              state.ref_bank[plane][cls][ref_slot][j], target, error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    if (effective.subset > 0 && uv_sym &&
        j >= rest::kWienerNsShortCoeffs) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j) + 1] = v;
      state.ref_bank[plane][cls][new_ptr][j + 1] = v;
      ++j;
    }
  }
  return true;
}

bool read_chroma_unit_impl(av2::Decoder& decoder, fs::LrState& state,
                           int plane, int unit_row, int unit_col,
                           ChromaUnitPlan* plan, std::string* error) {
  const int frame_type = state.params.frame_type[plane];
  int type = rest::kRestoreNone;
  if (frame_type == rest::kRestoreWienerNs) {
    int on = 0;
    if (!decode_sym(decoder, &state.use_wiener_ns[0], 2, &on, error)) {
      return false;
    }
    type = on != 0 ? rest::kRestoreWienerNs : rest::kRestoreNone;
  } else if (frame_type == rest::kRestoreNone) {
    type = rest::kRestoreNone;
  } else {
    return fail(error,
                "chroma LR supports RESTORE_WIENER_NONSEP or RESTORE_NONE");
  }
  plan->plane = plane;
  plan->unit_row = unit_row;
  plan->unit_col = unit_col;
  plan->on = type == rest::kRestoreWienerNs;
  const std::size_t index =
      static_cast<std::size_t>(unit_row) *
          static_cast<std::size_t>(state.unit_cols[plane]) +
      static_cast<std::size_t>(unit_col);
  state.unit_type[plane][index] = static_cast<std::uint8_t>(type);
  if (type != rest::kRestoreWienerNs) return true;

  if (state.params.frame_filters_on[plane]) {
    const std::size_t base = index * rest::kWienerNsCoeffStride;
    for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.frame_coeffs[plane][0][j];
      plan->filter.coeffs[j] = state.frame_coeffs[plane][0][j];
    }
    return true;
  }

  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];
  bool merged = false;
  if (!decode_raw_bit(decoder, &merged, error)) return false;
  int k_sel = 0;
  for (int k = 0; k < old_size - 1; ++k) {
    bool bit = false;
    if (!decode_raw_bit(decoder, &bit, error)) return false;
    if (bit) {
      k_sel = k;
      break;
    }
    k_sel = k + 1;
  }
  const int ref_slot = (old_ptr - k_sel + classes::kLrBankSize) % classes::kLrBankSize;
  plan->filter.merged = merged;
  plan->filter.use_bank_k = k_sel;
  if (merged) {
    if (state.bank_size[plane][cls] == 0) state.bank_size[plane][cls] = 1;
    const std::size_t base = index * rest::kWienerNsCoeffStride;
    for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
      const std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
      plan->filter.coeffs[j] = v;
    }
    return true;
  }

  int new_ptr = old_ptr;
  int new_size = old_size;
  if (old_size < classes::kLrBankSize) {
    new_ptr = old_size;
    new_size = old_size + 1;
  } else {
    new_ptr = (old_ptr + 1) % classes::kLrBankSize;
  }
  int subset = 0;
  for (subset = 0; subset < 2; ++subset) {
    int len = 0;
    if (!decode_sym(decoder, &state.wiener_ns_length[1][0], 2, &len, error)) {
      return false;
    }
    if (len == 0) break;
  }
  bool uv_sym = false;
  if (subset > 0) {
    int bit = 0;
    if (!decode_sym(decoder, &state.wiener_ns_uv_sym[0], 2, &bit, error)) {
      return false;
    }
    uv_sym = bit != 0;
  }
  state.bank_ptr[plane][cls] = new_ptr;
  state.bank_size[plane][cls] = new_size;
  plan->filter.subset = subset;
  plan->filter.uv_sym = uv_sym;
  const std::size_t base = index * rest::kWienerNsCoeffStride;
  for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
    const int min = rest::kWienerNsTapsMin[1][j];
    const int k = rest::kWienerNsTapsK[1][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (classes::kWienerNsTapsPresent[1][subset][j] != 0) {
      int target = 0;
      if (!classes::wiener_decode_signed_4part(
              decoder, state.wiener_ns_base, min, k,
              state.ref_bank[plane][cls][ref_slot][j], &target, error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    plan->filter.coeffs[j] = v;
    if (subset > 0 && uv_sym && j >= rest::kWienerNsShortCoeffs) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j) + 1] = v;
      state.ref_bank[plane][cls][new_ptr][j + 1] = v;
      plan->filter.coeffs[j + 1] = v;
      ++j;
    }
  }
  return true;
}

}  // namespace

bool chroma_measure_unit_bytes(const fs::LrState& state, int plane,
                               int unit_row, int unit_col,
                               const ChromaFilter& filter, bool on,
                               std::vector<std::uint8_t>* bytes,
                               std::string* error) {
  if (bytes == nullptr) return fail(error, "null byte output");
  if (!plane_is_chroma(plane)) {
    return fail(error, "chroma_measure_unit_bytes expects U or V");
  }
  fs::LrState trial = state;
  std::vector<std::uint8_t> buffer(1 << 16);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return fail(error, "entropy init failed");
  }
  encoder.disable_cdf_update = 1;
  if (!write_chroma_unit_impl(encoder, trial, plane, unit_row, unit_col, on,
                              filter, error)) {
    return false;
  }
  std::size_t size = 0;
  if (encoder.exit_symbol(&size) != av2::Status::kOk) {
    return fail(error, "entropy exit failed");
  }
  bytes->assign(buffer.begin(),
                buffer.begin() + static_cast<std::ptrdiff_t>(size));
  return true;
}

bool chroma_measure_decision_bytes(const ChromaDecision& decision,
                                   const ChromaLrInput& input,
                                   std::vector<std::uint8_t>* bytes,
                                   std::string* error) {
  if (bytes == nullptr) return fail(error, "null byte output");
  fs::LrState state;
  if (!chroma_init_tile_state(decision, input, &state, error)) return false;
  std::vector<std::uint8_t> buffer(1 << 20);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return fail(error, "entropy init failed");
  }
  encoder.disable_cdf_update = 1;
  for (int plane = 1; plane <= 2; ++plane) {
    const int chroma = chroma_index(plane);
    const ChromaUnitGrid& grid = decision.grid[chroma];
    for (int unit_row = 0; unit_row < grid.unit_rows; ++unit_row) {
      for (int unit_col = 0; unit_col < grid.unit_cols; ++unit_col) {
        const ChromaUnitPlan* plan = nullptr;
        for (const ChromaUnitPlan& candidate : decision.units) {
          if (candidate.plane == plane && candidate.unit_row == unit_row &&
              candidate.unit_col == unit_col) {
            plan = &candidate;
            break;
          }
        }
        if (plan == nullptr) {
          return fail(error, "decision missing a chroma unit");
        }
        if (!chroma_write_unit(encoder, state, plane, unit_row, unit_col, *plan,
                               error)) {
          return false;
        }
      }
    }
  }
  std::size_t size = 0;
  if (encoder.exit_symbol(&size) != av2::Status::kOk) {
    return fail(error, "entropy exit failed");
  }
  bytes->assign(buffer.begin(),
                buffer.begin() + static_cast<std::ptrdiff_t>(size));
  return true;
}

bool chroma_write_unit(av2::Encoder& encoder, fs::LrState& state, int plane,
                       int unit_row, int unit_col, const ChromaUnitPlan& plan,
                       std::string* error) {
  if (!plane_is_chroma(plane)) {
    return fail(error, "chroma unit write expects U or V");
  }
  if (plan.plane != plane) {
    return fail(error, "unit plan plane mismatch");
  }
  return write_chroma_unit_impl(encoder, state, plane, unit_row, unit_col,
                                plan.on, plan.filter, error);
}

bool chroma_write_unit_decision(av2::Encoder& encoder, fs::LrState& state,
                                int plane, int unit_row, int unit_col,
                                const ChromaFilter& filter, bool on,
                                std::string* error) {
  ChromaUnitPlan plan;
  plan.plane = plane;
  plan.unit_row = unit_row;
  plan.unit_col = unit_col;
  plan.on = on;
  plan.filter = filter;
  return chroma_write_unit(encoder, state, plane, unit_row, unit_col, plan,
                           error);
}

bool chroma_read_unit(av2::Decoder& decoder, fs::LrState& state, int plane,
                      int unit_row, int unit_col, ChromaUnitPlan* plan,
                      std::string* error) {
  if (plan == nullptr) return fail(error, "null unit plan");
  if (!plane_is_chroma(plane)) {
    return fail(error, "chroma unit read expects U or V");
  }
  return read_chroma_unit_impl(decoder, state, plane, unit_row, unit_col, plan,
                               error);
}

// ---------------------------------------------------------------------------
// Tile-level writer/reader
// ---------------------------------------------------------------------------

namespace {

const ChromaUnitPlan* find_plan(const ChromaDecision& decision, int plane,
                                int unit_row, int unit_col) {
  for (const ChromaUnitPlan& plan : decision.units) {
    if (plan.plane == plane && plan.unit_row == unit_row &&
        plan.unit_col == unit_col) {
      return &plan;
    }
  }
  return nullptr;
}

ChromaUnitPlan* find_plan_mut(ChromaDecision* decision, int plane,
                              int unit_row, int unit_col) {
  for (ChromaUnitPlan& plan : decision->units) {
    if (plan.plane == plane && plan.unit_row == unit_row &&
        plan.unit_col == unit_col) {
      return &plan;
    }
  }
  return nullptr;
}

}  // namespace

bool chroma_write_read_lr(av2::Encoder& encoder, fs::LrState& state,
                          int mi_row, int mi_col, int w4, int h4,
                          int tile_mi_row_start, int tile_mi_col_start,
                          const ChromaDecision& decision, std::string* error) {
  for (int plane = 1; plane <= 2; ++plane) {
    fs::LrUnitRange range;
    if (!fs::lr_unit_range(state, plane, mi_row, mi_col, w4, h4,
                           tile_mi_row_start, tile_mi_col_start, &range,
                           error)) {
      return false;
    }
    for (int unit_row = range.row_begin; unit_row < range.row_end; ++unit_row) {
      for (int unit_col = range.col_begin; unit_col < range.col_end;
           ++unit_col) {
        const ChromaUnitPlan* plan =
            find_plan(decision, plane, unit_row, unit_col);
        if (plan == nullptr) {
          return fail(error, "missing chroma unit plan for read_lr order");
        }
        if (!chroma_write_unit(encoder, state, plane, unit_row, unit_col,
                               *plan, error)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool chroma_read_read_lr(av2::Decoder& decoder, fs::LrState& state, int mi_row,
                         int mi_col, int w4, int h4, int tile_mi_row_start,
                         int tile_mi_col_start, ChromaDecision* decision,
                         std::string* error) {
  if (decision == nullptr) return fail(error, "null ChromaDecision");
  for (int plane = 1; plane <= 2; ++plane) {
    fs::LrUnitRange range;
    if (!fs::lr_unit_range(state, plane, mi_row, mi_col, w4, h4,
                           tile_mi_row_start, tile_mi_col_start, &range,
                           error)) {
      return false;
    }
    for (int unit_row = range.row_begin; unit_row < range.row_end; ++unit_row) {
      for (int unit_col = range.col_begin; unit_col < range.col_end;
           ++unit_col) {
        ChromaUnitPlan plan;
        if (!chroma_read_unit(decoder, state, plane, unit_row, unit_col, &plan,
                              error)) {
          return false;
        }
        ChromaUnitPlan* slot = find_plan_mut(decision, plane, unit_row, unit_col);
        if (slot == nullptr) {
          decision->units.push_back(plan);
          slot = &decision->units.back();
        }
        *slot = plan;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Parameters / state / header cost
// ---------------------------------------------------------------------------

bool chroma_make_lr_params(const ChromaDecision& decision, fs::LrParams* params,
                           std::string* error) {
  if (params == nullptr) return fail(error, "null LrParams");
  *params = fs::LrParams{};
  params->frame_type[0] = rest::kRestoreNone;
  params->frame_type[1] = rest::kRestoreWienerNs;
  params->frame_type[2] = rest::kRestoreWienerNs;
  params->frame_filters_on[0] = false;
  params->frame_filters_on[1] = false;
  params->frame_filters_on[2] = false;
  params->num_filter_classes = 1;
  params->num_filter_classes_idx = 0;
  params->luma_shift = 3;
  params->chroma_shift = decision.shift;
  return true;
}

bool chroma_init_tile_state(const ChromaDecision& decision,
                            const ChromaLrInput& input, fs::LrState* state,
                            std::string* error) {
  fs::LrParams params;
  if (!chroma_make_lr_params(decision, &params, error)) return false;
  if (!fs::lr_state_init(state, params, input.mi_cols, input.mi_rows,
                         input.sub_x, input.sub_y, decision.sb_size, error)) {
    return false;
  }
  return true;
}

bool chroma_measure_header_bits(const ChromaDecision& decision, int sb_size,
                                int base_q_idx, int* bits, std::string* error) {
  if (bits == nullptr) return fail(error, "null bit output");
  classes::WienerLrHeaderConfig config;
  config.coded_lossless = false;
  config.enable_restoration = true;
  config.frame_is_intra = true;
  config.num_planes = 3;
  config.sb_size = sb_size;
  config.sub_x = 1;
  config.sub_y = 1;
  config.base_q_idx = base_q_idx;
  config.lr_tools_disable[0][rest::kRestoreNone] = false;
  config.lr_tools_disable[0][rest::kRestorePcWiener] = true;
  config.lr_tools_disable[0][rest::kRestoreWienerNs] = false;
  config.lr_tools_disable[0][rest::kRestoreSwitchable] = false;
  config.lr_tools_disable[1][rest::kRestoreNone] = false;
  config.lr_tools_disable[1][rest::kRestorePcWiener] = true;
  config.lr_tools_disable[1][rest::kRestoreWienerNs] = false;
  config.lr_tools_disable[1][rest::kRestoreSwitchable] = false;

  classes::WienerLrHeaderParams params;
  params.frame_type[0] = rest::kRestoreNone;
  params.frame_type[1] = rest::kRestoreWienerNs;
  params.frame_type[2] = rest::kRestoreWienerNs;
  params.chroma_shift = decision.shift;
  params.num_filter_classes = 1;
  params.num_filter_classes_idx = 0;
  classes::WienerReferenceSet refs;
  refs.frame_is_intra = true;
  BitWriter writer;
  if (!classes::wiener_write_lr_params(writer, config, &params, refs, error)) {
    return false;
  }
  *bits = static_cast<int>(writer.bit_size());
  return true;
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

namespace {

bool decide_for_shift(const ChromaLrInput& input,
                      const ChromaLrOptions& options, int shift,
                      ChromaDecision* decision, std::string* error) {
  ChromaUnitSize unit;
  const int width = std::max(input.plane_width[0], input.plane_width[1]);
  const int height = std::max(input.plane_height[0], input.plane_height[1]);
  if (width <= 0 || height <= 0) return fail(error, "empty chroma plane");
  if (!chroma_select_unit_size((rest::kRestorationTileSizeMax >>
                                std::max(input.sub_x, input.sub_y)) >>
                                   shift,
                               input.sb_size, input.sub_x, input.sub_y, &unit,
                               error)) {
    return false;
  }
  if (unit.shift != shift) {
    return fail(error, "internal chroma shift mismatch");
  }
  *decision = ChromaDecision{};
  decision->shift = shift;
  decision->unit_size = unit.size;
  decision->sb_size = input.sb_size;
  for (int chroma = 0; chroma < 2; ++chroma) {
    if (!chroma_make_unit_grid(input.plane_width[chroma],
                               input.plane_height[chroma], unit.size,
                               &decision->grid[chroma], error)) {
      return false;
    }
  }
  if (decision->grid[0].unit_cols != decision->grid[1].unit_cols ||
      decision->grid[0].unit_rows != decision->grid[1].unit_rows) {
    return fail(error, "U and V unit grids differ");
  }

  // Fit every unit once.  The fitter is the same stripe-aware least-squares
  // path the decoder's integer filter is evaluated with.
  std::vector<std::vector<PlaneFit>> fits(2);
  for (int plane = 1; plane <= 2; ++plane) {
    const int chroma = chroma_index(plane);
    const ChromaUnitGrid& grid = decision->grid[chroma];
    fits[static_cast<std::size_t>(chroma)].resize(
        static_cast<std::size_t>(grid.count()));
    for (int unit_row = 0; unit_row < grid.unit_rows; ++unit_row) {
      for (int unit_col = 0; unit_col < grid.unit_cols; ++unit_col) {
        PlaneFit fit;
        fit.plane = plane;
        fit.row = unit_row;
        fit.col = unit_col;
        fit.candidates = fit_unit(input, *decision, plane, unit_row, unit_col,
                                  options, error);
        if (fit.candidates.empty()) return false;
        const std::size_t at =
            static_cast<std::size_t>(unit_row) *
                static_cast<std::size_t>(grid.unit_cols) +
            static_cast<std::size_t>(unit_col);
        // Identity SSE over the §7.20 unit region.  Unit origins include the
        // +8 luma-line stripe offset, so the region is not the unit_size
        // rectangle in plane coordinates.
        ChromaFilter identity;
        std::vector<std::uint8_t> identity_type(
            static_cast<std::size_t>(grid.count()),
            static_cast<std::uint8_t>(rest::kRestoreWienerNs));
        std::vector<std::int16_t> identity_coeffs(
            static_cast<std::size_t>(grid.count()) *
                rest::kWienerNsCoeffStride,
            0);
        std::vector<std::uint8_t> identity_dummy(
            static_cast<std::size_t>(grid.plane_width) *
                static_cast<std::size_t>(grid.plane_height),
            0);
        rest::LrCoreFrame identity_core;
        fill_core(input, plane, unit.size, identity_dummy.data(),
                  grid.plane_width, &identity_type, &identity_coeffs,
                  &identity_core, error);
        fit.identity_sse = chroma_filter_sse(
            identity_core, plane, unit_row, unit_col, input.source[chroma],
            input.source_stride[chroma], identity);
        fits[static_cast<std::size_t>(chroma)][at] = std::move(fit);
      }
    }
  }

  // Optional shared-U/V filter: one coefficient set for both planes.  This is
  // an encoder choice (the syntax still codes each plane's unit separately);
  // it is selected only when it beats the independent fits on
  // SSE + lambda_bits * exact coded bytes.
  std::vector<ChromaFilter> shared_override[2];
  std::vector<bool> has_shared_override[2];
  for (int c = 0; c < 2; ++c) {
    shared_override[c].assign(static_cast<std::size_t>(decision->grid[c].count()),
                              ChromaFilter{});
    has_shared_override[c].assign(
        static_cast<std::size_t>(decision->grid[c].count()), false);
  }
  const bool grids_match = decision->grid[0].count() == decision->grid[1].count();
  if (options.allow_shared_uv && grids_match) {
    const auto eval = [&](int plane, int unit_row, int unit_col,
                          const ChromaFilter& filter) {
      const int chroma = chroma_index(plane);
      const ChromaUnitGrid& grid = decision->grid[chroma];
      std::vector<std::uint8_t> unit_type(
          static_cast<std::size_t>(grid.count()),
          static_cast<std::uint8_t>(rest::kRestoreWienerNs));
      std::vector<std::int16_t> unit_coeffs(
          static_cast<std::size_t>(grid.count()) *
              rest::kWienerNsCoeffStride,
          0);
      std::vector<std::uint8_t> dummy(
          static_cast<std::size_t>(grid.plane_width) *
              static_cast<std::size_t>(grid.plane_height),
          0);
      rest::LrCoreFrame core;
      fill_core(input, plane, decision->unit_size, dummy.data(),
                grid.plane_width, &unit_type, &unit_coeffs, &core, error);
      return chroma_filter_sse(core, plane, unit_row, unit_col,
                               input.source[chroma],
                               input.source_stride[chroma], filter);
    };
    fs::LrState fresh;
    if (!chroma_init_tile_state(*decision, input, &fresh, error)) return false;
    const auto bits = [&](int plane, int unit_row, int unit_col,
                          const ChromaFilter& filter) {
      std::vector<std::uint8_t> bytes;
      if (!chroma_measure_unit_bytes(fresh, plane, unit_row, unit_col, filter,
                                     true, &bytes, error)) {
        return std::numeric_limits<int>::max() / 8;
      }
      return static_cast<int>(bytes.size()) * 8;
    };
    for (int i = 0; i < decision->grid[0].count(); ++i) {
      const PlaneFit& u = fits[0][static_cast<std::size_t>(i)];
      const PlaneFit& v = fits[1][static_cast<std::size_t>(i)];
      const ChromaFilter& bu = u.candidates.front().filter;
      const ChromaFilter& bv = v.candidates.front().filter;
      double best_cost = u.candidates.front().sse + v.candidates.front().sse +
                         options.lambda_bits * (bits(1, u.row, u.col, bu) +
                                                bits(2, v.row, v.col, bv));
      ChromaFilter best = bu;
      bool have_shared = false;
      std::vector<ChromaFilter> all;
      all.reserve(u.candidates.size() + v.candidates.size());
      for (const FitCandidate& candidate : u.candidates) {
        all.push_back(candidate.filter);
      }
      for (const FitCandidate& candidate : v.candidates) {
        all.push_back(candidate.filter);
      }
      for (const ChromaFilter& candidate : all) {
        const double sse = eval(1, u.row, u.col, candidate) +
                           eval(2, v.row, v.col, candidate);
        const double cost =
            sse + options.lambda_bits *
                      (bits(1, u.row, u.col, candidate) +
                       bits(2, v.row, v.col, candidate));
        if (cost + 1e-9 < best_cost) {
          best_cost = cost;
          best = candidate;
          have_shared = true;
        }
      }
      if (have_shared) {
        shared_override[0][static_cast<std::size_t>(i)] = best;
        shared_override[1][static_cast<std::size_t>(i)] = best;
        has_shared_override[0][static_cast<std::size_t>(i)] = true;
        has_shared_override[1][static_cast<std::size_t>(i)] = true;
      }
    }
  }

  // Sequential per-unit selection with real bank state and real bit counts.
  fs::LrState sim;
  if (!chroma_init_tile_state(*decision, input, &sim, error)) return false;
  double total_sse = 0.0;
  double total_identity = 0.0;
  int total_bits = 0;
  for (int plane = 1; plane <= 2; ++plane) {
    const int chroma = chroma_index(plane);
    const ChromaUnitGrid& grid = decision->grid[chroma];
    for (int unit_row = 0; unit_row < grid.unit_rows; ++unit_row) {
      for (int unit_col = 0; unit_col < grid.unit_cols; ++unit_col) {
        const std::size_t at =
            static_cast<std::size_t>(unit_row) *
                static_cast<std::size_t>(grid.unit_cols) +
            static_cast<std::size_t>(unit_col);
        const PlaneFit& fit = fits[static_cast<std::size_t>(chroma)][at];
        std::vector<FitCandidate> candidates = fit.candidates;
        if (has_shared_override[chroma][at]) {
          // Force the shared U/V filter for this unit.
          const int chroma_plane = plane;
          const int chroma_grid = chroma_index(plane);
          const ChromaUnitGrid& g = decision->grid[chroma_grid];
          std::vector<std::uint8_t> unit_type(
              static_cast<std::size_t>(g.count()),
              static_cast<std::uint8_t>(rest::kRestoreWienerNs));
          std::vector<std::int16_t> unit_coeffs(
              static_cast<std::size_t>(g.count()) *
                  rest::kWienerNsCoeffStride,
              0);
          std::vector<std::uint8_t> dummy(
              static_cast<std::size_t>(g.plane_width) *
                  static_cast<std::size_t>(g.plane_height),
              0);
          rest::LrCoreFrame core;
          fill_core(input, chroma_plane, decision->unit_size, dummy.data(),
                    g.plane_width, &unit_type, &unit_coeffs, &core, error);
          const ChromaFilter& shared =
              shared_override[chroma][at];
          FitCandidate forced;
          forced.filter = shared;
          forced.sse = chroma_filter_sse(core, plane, unit_row, unit_col,
                                         input.source[chroma],
                                         input.source_stride[chroma], shared);
          candidates.assign(1, forced);
          decision->shared_uv = true;
        }
        ChromaUnitPlan best;
        best.plane = plane;
        best.unit_row = unit_row;
        best.unit_col = unit_col;
        best.identity_sse = fit.identity_sse;
        best.on = false;
        best.filter = ChromaFilter{};
        best.sse = fit.identity_sse;
        best.cost = best.sse;
        if (!options.allow_off) {
          const FitCandidate& candidate = candidates.front();
          best.on = true;
          best.filter = candidate.filter;
          best.sse = candidate.sse;
          best.cost = best.sse;
        }

        std::vector<std::uint8_t> bytes;
        if (!chroma_measure_unit_bytes(sim, plane, unit_row, unit_col,
                                       best.filter, best.on, &bytes, error)) {
          return false;
        }
        best.bits = static_cast<int>(bytes.size()) * 8;
        best.cost = best.sse + options.lambda_bits * best.bits;

        for (const FitCandidate& candidate : candidates) {
          ChromaFilter filter = candidate.filter;
          // Bank merge: only when a previous unit's filter is identical.
          int merge_k = -1;
          if (options.allow_bank_merge) {
            merge_k = find_merge_bank(sim, plane, filter);
          }
          for (int mode = 0; mode < 2; ++mode) {
            ChromaUnitPlan trial;
            trial.plane = plane;
            trial.unit_row = unit_row;
            trial.unit_col = unit_col;
            trial.identity_sse = fit.identity_sse;
            trial.on = true;
            trial.filter = filter;
            trial.sse = candidate.sse;
            if (mode == 1) {
              if (merge_k < 0) continue;
              trial.filter.merged = true;
              trial.filter.use_bank_k = merge_k;
            }
            if (!chroma_measure_unit_bytes(sim, plane, unit_row, unit_col,
                                           trial.filter, trial.on, &bytes,
                                           error)) {
              return false;
            }
            trial.bits = static_cast<int>(bytes.size()) * 8;
            trial.cost = trial.sse + options.lambda_bits * trial.bits;
            if (trial.cost < best.cost) best = trial;
          }
        }
        if (!commit_unit(&sim, plane, unit_row, unit_col, best, error)) {
          return false;
        }
        total_sse += best.sse;
        total_identity += best.identity_sse;
        total_bits += best.bits;
        decision->units.push_back(best);
      }
    }
  }
  // A shared filter is also "shared" when the independent U and V choices
  // happen to be identical; report that without changing the stream.
  if (!decision->shared_uv && options.allow_shared_uv) {
    bool all_shared = true;
    for (const ChromaUnitPlan& u : decision->units) {
      if (u.plane != 1) continue;
      const ChromaUnitPlan* v = nullptr;
      for (const ChromaUnitPlan& candidate : decision->units) {
        if (candidate.plane == 2 && candidate.unit_row == u.unit_row &&
            candidate.unit_col == u.unit_col) {
          v = &candidate;
          break;
        }
      }
      if (v == nullptr || u.on != v->on ||
          u.filter.subset != v->filter.subset ||
          u.filter.uv_sym != v->filter.uv_sym ||
          u.filter.merged != v->filter.merged ||
          std::memcmp(u.filter.coeffs, v->filter.coeffs,
                      sizeof(u.filter.coeffs)) != 0) {
        all_shared = false;
        break;
      }
    }
    decision->shared_uv = all_shared;
  }
  decision->sse = total_sse;
  decision->identity_sse = total_identity;
  decision->tile_bits = total_bits;
  // Report the exact whole-section payload (one coder) rather than the sum of
  // per-unit coders, each of which pays its own termination bytes.
  std::vector<std::uint8_t> payload;
  if (chroma_measure_decision_bytes(*decision, input, &payload, error)) {
    decision->tile_bits = static_cast<int>(payload.size()) * 8;
  }
  int header_bits = 0;
  if (!chroma_measure_header_bits(*decision, input.sb_size, 90, &header_bits,
                                  error)) {
    return false;
  }
  decision->header_bits = header_bits;
  decision->cost = decision->sse +
                   options.lambda_bits * (header_bits + decision->tile_bits);
  return true;
}

}  // namespace

bool chroma_decide(const ChromaLrInput& input, const ChromaLrOptions& options,
                   ChromaDecision* decision, std::string* error) {
  if (decision == nullptr) return fail(error, "null ChromaDecision");
  if (input.sub_x != 1 || input.sub_y != 1) {
    return fail(error, "this module targets 8-bit 4:2:0 chroma");
  }
  if (input.curr[0] == nullptr || input.curr[1] == nullptr ||
      input.cdef[0] == nullptr || input.cdef[1] == nullptr ||
      input.source[0] == nullptr || input.source[1] == nullptr ||
      input.luma_curr == nullptr || input.luma_cdef == nullptr) {
    return fail(error, "chroma decide needs curr/cdef/source and luma planes");
  }
  if (input.mi_cols <= 0 || input.mi_rows <= 0 ||
      input.plane_width[0] <= 0 || input.plane_height[0] <= 0) {
    return fail(error, "invalid chroma decide geometry");
  }
  int shifts[4] = {1, 2, 3, 0};
  int count = 4;
  if (options.forced_shift >= 0) {
    shifts[0] = options.forced_shift;
    count = 1;
  }
  bool have = false;
  ChromaDecision best;
  for (int i = 0; i < count; ++i) {
    ChromaDecision trial;
    if (!decide_for_shift(input, options, shifts[i], &trial, error)) {
      return false;
    }
    if (!have || trial.cost < best.cost) {
      best = std::move(trial);
      have = true;
    }
  }
  if (!have) return fail(error, "no chroma unit size candidate");
  *decision = std::move(best);
  return true;
}

// ---------------------------------------------------------------------------
// Apply + PSNR
// ---------------------------------------------------------------------------

rest::RestorationStatus chroma_apply(const ChromaLrInput& input,
                                     const ChromaDecision& decision,
                                     std::uint8_t* out[2],
                                     int out_stride[2]) {
  for (int chroma = 0; chroma < 2; ++chroma) {
    if (input.curr[chroma] == nullptr || input.source[chroma] == nullptr ||
        out[chroma] == nullptr) {
      return rest::RestorationStatus::kInvalidArgument;
    }
  }
  rest::RestorationFrame frame;
  frame.mi_cols = input.mi_cols;
  frame.mi_rows = input.mi_rows;
  frame.subsampling_x = input.sub_x;
  frame.subsampling_y = input.sub_y;
  frame.bit_depth = 8;
  frame.sb_width = input.sb_size;
  frame.sb_height = input.sb_size;
  frame.sb_shift = 4;
  frame.cfl_ds_filter_index = input.cfl_ds_filter_index;
  std::vector<std::uint8_t> utype[2];
  std::vector<std::int16_t> ucoeff[2];
  for (int chroma = 0; chroma < 2; ++chroma) {
    const int plane = chroma + 1;
    const ChromaUnitGrid& grid = decision.grid[chroma];
    utype[chroma].assign(static_cast<std::size_t>(grid.count()),
                         rest::kRestoreNone);
    ucoeff[chroma].assign(
        static_cast<std::size_t>(grid.count()) * rest::kWienerNsCoeffStride,
        0);
    for (const ChromaUnitPlan& plan : decision.units) {
      if (plan.plane != plane) continue;
      const std::size_t at = static_cast<std::size_t>(plan.unit_row) *
                                 static_cast<std::size_t>(grid.unit_cols) +
                             static_cast<std::size_t>(plan.unit_col);
      utype[chroma][at] = static_cast<std::uint8_t>(
          plan.on ? rest::kRestoreWienerNs : rest::kRestoreNone);
      if (!plan.on) continue;
      for (int j = 0; j < rest::kWienerNsChromaCoeffs; ++j) {
        ucoeff[chroma][at * rest::kWienerNsCoeffStride +
                       static_cast<std::size_t>(j)] = plan.filter.coeffs[j];
      }
    }
  }
  rest::LoopRestorationParams params;
  params.frame_type[0] = rest::kRestoreNone;
  params.frame_type[1] = rest::kRestoreWienerNs;
  params.frame_type[2] = rest::kRestoreWienerNs;
  params.unit_size[0] = decision.unit_size;  // unused: luma NONE
  params.unit_size[1] = decision.unit_size;
  params.unit_size[2] = decision.unit_size;
  for (int plane = 1; plane <= 2; ++plane) {
    const int chroma = chroma_index(plane);
    params.unit_type[plane] = utype[chroma].data();
    params.unit_rows[plane] = decision.grid[chroma].unit_rows;
    params.unit_cols[plane] = decision.grid[chroma].unit_cols;
    params.unit_coeffs[plane] = ucoeff[chroma].data();
  }
  for (int plane = 0; plane < 3; ++plane) {
    // RestorationFrame uses a mutable view; §7.20 only reads curr/cdef.
    frame.curr[plane].data = const_cast<std::uint8_t*>(
        plane == 0 ? input.curr[0] : input.curr[plane - 1]);
    frame.curr[plane].stride =
        plane == 0 ? input.curr_stride[0] : input.curr_stride[plane - 1];
    frame.cdef[plane].data = const_cast<std::uint8_t*>(
        plane == 0 ? input.cdef[0] : input.cdef[plane - 1]);
    frame.cdef[plane].stride =
        plane == 0 ? input.cdef_stride[0] : input.cdef_stride[plane - 1];
    frame.lr[plane].data = out[plane == 0 ? 0 : plane - 1];
    frame.lr[plane].stride = out_stride[plane == 0 ? 0 : plane - 1];
  }
  // Luma is not filtered; the core still walks plane 0.  The chroma-from-luma
  // taps read the real luma reconstruction, so plane 0 must carry it.
  const int luma_w = input.luma_width > 0
                        ? input.luma_width
                        : decision.grid[0].plane_width * 2;
  const int luma_h = input.luma_height > 0
                        ? input.luma_height
                        : decision.grid[0].plane_height * 2;
  if (input.luma_curr == nullptr || input.luma_cdef == nullptr) {
    return rest::RestorationStatus::kInvalidArgument;
  }
  std::vector<std::uint8_t> luma_out(
      static_cast<std::size_t>(luma_w) * static_cast<std::size_t>(luma_h), 0);
  frame.curr[0].data = const_cast<std::uint8_t*>(input.luma_curr);
  frame.curr[0].stride = input.luma_curr_stride;
  frame.cdef[0].data = const_cast<std::uint8_t*>(input.luma_cdef);
  frame.cdef[0].stride = input.luma_cdef_stride;
  frame.lr[0].data = luma_out.data();
  frame.lr[0].stride = luma_w;
  return rest::loop_restoration_frame(frame, params);
}

double chroma_psnr(const std::uint8_t* a, int a_stride, const std::uint8_t* b,
                   int b_stride, int width, int height) {
  if (a == nullptr || b == nullptr || width <= 0 || height <= 0) return 0.0;
  std::uint64_t sse = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t ai =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(a_stride) +
          static_cast<std::size_t>(x);
      const std::size_t bi =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(b_stride) +
          static_cast<std::size_t>(x);
      const int d = static_cast<int>(a[ai]) - static_cast<int>(b[bi]);
      sse += static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(d);
    }
  }
  if (sse == 0) return 99.0;
  const double mse = static_cast<double>(sse) /
                     (static_cast<double>(width) * static_cast<double>(height));
  return 10.0 * std::log10((255.0 * 255.0) / mse);
}

}  // namespace av2enc::lr_chroma
