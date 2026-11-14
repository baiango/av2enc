// §7.18 CDEF planning and application for the 10-bit product still writer.
//
// The pixel arithmetic is the shared `cdef_core.h` template; this header only
// builds the encoder-side decision (strength sets + per-64x64 `cdef_idx`) for
// the fixed M10 block plan at 10-bit and every product chroma format.
//
// The search mirrors `src/codec/filters/cdef/cdef_search.h`:
//   * one candidate set per 64x64 luma block,
//   * `include_off` inserts the all-zero set so a block can be left unfiltered,
//   * `CdefStrengths == 1` codes no `cdef_idx` symbol at all.
// It reuses the search's candidate pool and deterministic selection rule so
// the 8-bit and 10-bit decisions cannot drift.
#ifndef AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_CDEF_H_
#define AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_CDEF_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/cdef/cdef.h"
#include "codec/filters/cdef/cdef_core.h"
#include "codec/filters/cdef/cdef_search.h"
#include "formats/format.h"
#include "formats/high_bitdepth/hbd_depth.h"

namespace av2enc::hbd {

struct HbdCdefPlan {
  bool enabled = false;
  int damping = 3;
  int num_strengths = 1;
  cdef::CdefStrength strengths[cdef::kMaxStrengths] = {};
  int n64_cols = 0;
  int n64_rows = 0;
  std::vector<std::int8_t> index;
  cdef::CdefOptions options;
  std::int64_t sse_before = 0;
  std::int64_t sse_after = 0;
};

// Input plane views for the decision.  `src` is the unfiltered reconstruction
// (CurrFrame), `ref` the visible source picture used only for SSE.
struct HbdFilterFrame16 {
  const std::uint16_t* src[3] = {nullptr, nullptr, nullptr};
  const std::uint16_t* ref[3] = {nullptr, nullptr, nullptr};
  int src_stride[3] = {0, 0, 0};
  int ref_stride[3] = {0, 0, 0};
  int visible_w[3] = {0, 0, 0};
  int visible_h[3] = {0, 0, 0};
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int num_planes = 3;
  int bit_depth = 10;
};

namespace cdef_detail {

// One 8x8 luma CDEF block (plus its chroma blocks) evaluated with the shared
// §7.18 arithmetic.  Returns squared error against `ref` over the visible
// samples.  Chroma geometry follows the frame subsampling, so 4:2:2 and 4:4:4
// work without special cases.
inline cdef::CdefBlockSse eval_block16(const HbdFilterFrame16& f,
                                       const cdef::CdefStrength& strength,
                                       int damping, int r, int c) {
  cdef::CdefBlockSse sse;
  const int coeff_shift = f.bit_depth - 8;
  const cdef::CdefRegion region =
      cdef::cdef_filter_region(f.mi_cols, f.mi_rows, nullptr, 0, nullptr, 0,
                                false, r, c);
  int y_dir = 0;
  std::int64_t var = 0;
  cdef::CdefFilterParams luma{0, 0, 0, damping};
  cdef::CdefFilterParams chroma{0, 0, 0, damping - 1};
  if (!cdef::cdef_is_zero_strength(strength)) {
    cdef::cdef_direction_t(f.src[0], static_cast<std::size_t>(f.src_stride[0]),
                           c * cdef::kMiSize, r * cdef::kMiSize, f.bit_depth,
                           &y_dir, &var);
    const int y_pri = strength.y_pri << coeff_shift;
    const int y_sec = strength.y_sec << coeff_shift;
    luma = cdef::cdef_luma_params(y_pri, y_sec, y_dir, var,
                                  damping + coeff_shift);
    const int uv_pri = strength.uv_pri << coeff_shift;
    const int uv_sec = strength.uv_sec << coeff_shift;
    chroma = cdef::cdef_chroma_params(uv_pri, uv_sec, y_dir, f.sub_x, f.sub_y,
                                      damping + coeff_shift);
  }

  const int x0 = c * cdef::kMiSize;
  const int y0 = r * cdef::kMiSize;
  {
    const std::uint16_t* src = f.src[0];
    const std::size_t stride = static_cast<std::size_t>(f.src_stride[0]);
    const std::uint16_t* ref = f.ref[0];
    const std::size_t ref_stride = static_cast<std::size_t>(f.ref_stride[0]);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const std::size_t o = static_cast<std::size_t>(y0 + i) * stride +
                              static_cast<std::size_t>(x0 + j);
        int value = static_cast<int>(src[o]);
        if (!cdef::cdef_is_zero_strength(strength)) {
          value = cdef::cdef_filter_sample_t(
              src, stride, x0, y0, i, j, luma.dir, luma.pri, luma.sec,
              luma.damping, 0, 0, coeff_shift, region);
        }
        if (x0 + j < f.visible_w[0] && y0 + i < f.visible_h[0]) {
          const int diff =
              value - static_cast<int>(ref[static_cast<std::size_t>(y0 + i) *
                                               ref_stride +
                                           static_cast<std::size_t>(x0 + j)]);
          sse.luma += static_cast<std::int64_t>(diff) * diff;
        }
      }
    }
  }

  for (int plane = 1; plane < f.num_planes; ++plane) {
    const int sub_x = f.sub_x;
    const int sub_y = f.sub_y;
    const int cx0 = x0 >> sub_x;
    const int cy0 = y0 >> sub_y;
    const int cw = 8 >> sub_x;
    const int ch = 8 >> sub_y;
    const std::uint16_t* src = f.src[plane];
    const std::size_t stride = static_cast<std::size_t>(f.src_stride[plane]);
    const std::uint16_t* ref = f.ref[plane];
    const std::size_t ref_stride = static_cast<std::size_t>(f.ref_stride[plane]);
    for (int i = 0; i < ch; ++i) {
      for (int j = 0; j < cw; ++j) {
        const std::size_t o = static_cast<std::size_t>(cy0 + i) * stride +
                              static_cast<std::size_t>(cx0 + j);
        int value = static_cast<int>(src[o]);
        if (!cdef::cdef_is_zero_strength(strength)) {
          value = cdef::cdef_filter_sample_t(
              src, stride, cx0, cy0, i, j, chroma.dir, chroma.pri, chroma.sec,
              chroma.damping, sub_x, sub_y, coeff_shift, region);
        }
        if (cx0 + j < f.visible_w[plane] &&
            cy0 + i < f.visible_h[plane]) {
          const int diff =
              value - static_cast<int>(ref[static_cast<std::size_t>(cy0 + i) *
                                               ref_stride +
                                           static_cast<std::size_t>(cx0 + j)]);
          sse.chroma += static_cast<std::int64_t>(diff) * diff;
        }
      }
    }
  }
  return sse;
}

}  // namespace cdef_detail

// Builds the CDEF decision for one intra still frame.  `num_strengths` is the
// requested CdefStrengths (1..8); 1 codes the header-only form.
inline bool plan_cdef(const HbdFilterFrame16& frame, int damping,
                      int num_strengths, HbdCdefPlan* plan,
                      std::string* error) {
  auto fail = [error](const std::string& text) {
    if (error != nullptr) *error = text;
    return false;
  };
  if (plan == nullptr) return fail("cdef plan: null output");
  *plan = HbdCdefPlan{};
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0 ||
      (frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return fail("cdef plan: invalid MI grid");
  }
  if (frame.num_planes < 1 || frame.num_planes > 3) {
    return fail("cdef plan: invalid plane count");
  }
  if (damping < 3 || damping > 6) return fail("cdef plan: damping outside 3..6");
  if (num_strengths < 1 || num_strengths > cdef::kMaxStrengths) {
    return fail("cdef plan: num_strengths outside 1..8");
  }

  cdef::CdefSearchOptions options;
  options.damping = damping;
  options.num_strengths = num_strengths;
  options.on_skip_txfm = false;
  options.include_off = true;
  std::vector<cdef::CdefStrength> pool;
  if (!cdef::search_detail::build_pool(options, &pool, error)) return false;

  const int n64_cols = (frame.mi_cols + cdef::kCdefSize4 - 1) /
                       cdef::kCdefSize4;
  const int n64_rows = (frame.mi_rows + cdef::kCdefSize4 - 1) /
                       cdef::kCdefSize4;
  const int nblocks = n64_cols * n64_rows;
  const int ncand = static_cast<int>(pool.size());
  const std::size_t nb = static_cast<std::size_t>(nblocks);
  std::vector<std::int64_t> sse_table(static_cast<std::size_t>(ncand) * nb, 0);
  std::vector<std::int64_t> off_sse(nb, 0);
  int off_index = -1;
  for (int k = 0; k < ncand; ++k) {
    if (cdef::cdef_is_zero_strength(pool[static_cast<std::size_t>(k)])) {
      off_index = k;
      break;
    }
  }

  for (int r = 0; r < frame.mi_rows; r += cdef::kCdefBlock4) {
    for (int c = 0; c < frame.mi_cols; c += cdef::kCdefBlock4) {
      const int block = (r / cdef::kCdefSize4) * n64_cols +
                        (c / cdef::kCdefSize4);
      for (int k = 0; k < ncand; ++k) {
        const cdef::CdefBlockSse one = cdef_detail::eval_block16(
            frame, pool[static_cast<std::size_t>(k)], damping, r, c);
        sse_table[static_cast<std::size_t>(k) * nb +
                  static_cast<std::size_t>(block)] += one.luma + one.chroma;
      }
    }
  }
  if (off_index >= 0) {
    const std::size_t row = static_cast<std::size_t>(off_index) * nb;
    for (int b = 0; b < nblocks; ++b) {
      off_sse[static_cast<std::size_t>(b)] = sse_table[row +
                                                       static_cast<std::size_t>(b)];
    }
  }

  cdef::CdefSearchResult result;
  if (!cdef::search_detail::select(pool, n64_cols, n64_rows, sse_table,
                                   off_sse, options, &result, error)) {
    return false;
  }
  plan->enabled = true;
  plan->damping = damping;
  plan->num_strengths = result.num_strengths;
  for (int i = 0; i < result.num_strengths; ++i) {
    plan->strengths[i] = result.strengths[i];
  }
  plan->n64_cols = n64_cols;
  plan->n64_rows = n64_rows;
  plan->index = result.cdef_idx;
  plan->sse_before = result.sse_off_total;
  plan->sse_after = result.sse_total;
  plan->options = result.options;
  plan->options.bit_depth = frame.bit_depth;
  if (error != nullptr) error->clear();
  return true;
}

// Runs the §7.18 filter from `plan` on uint16 planes.  `src` is CurrFrame and
// `dst` the separate CdefFrame; both use the same stride.  The whole frame is
// written (copy or filter).
inline bool apply_cdef16(const HbdCdefPlan& plan, const fmt::Format& format,
                         const std::uint16_t* const src[3],
                         std::uint16_t* const dst[3], const int stride[3],
                         int mi_cols, int mi_rows,
                         std::string* error) {
  if (!plan.enabled) return true;
  const int num_planes = fmt::num_planes(format.chroma);
  cdef::CdefFrame16 frame{};
  for (int p = 0; p < num_planes; ++p) {
    frame.planes[p].src = src[p];
    frame.planes[p].dst = dst[p];
    frame.planes[p].stride = stride[p];
  }
  frame.mi_cols = mi_cols;
  frame.mi_rows = mi_rows;
  frame.subsampling_x = fmt::subsampling_x(format.chroma);
  frame.subsampling_y = fmt::subsampling_y(format.chroma);
  frame.num_planes = num_planes;
  // The M10 still plan has no skipped leaves and no lossless segments.
  std::vector<std::uint8_t> zeros(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows), 0);
  frame.skip = zeros.data();
  frame.lossless = zeros.data();
  frame.lossless_chroma = zeros.data();
  frame.cdef_idx = plan.index.data();
  frame.n64_cols = plan.n64_cols;
  frame.n64_rows = plan.n64_rows;
  cdef::CdefOptions options = plan.options;
  options.bit_depth = plan.options.bit_depth;
  const cdef::CdefStatus status = cdef::cdef_frame16(frame, options);
  if (status != cdef::CdefStatus::kOk) {
    if (error != nullptr) {
      *error = std::string("hbd cdef: ") + cdef::cdef_status_string(status);
    }
    return false;
  }
  return true;
}

}  // namespace av2enc::hbd

#endif  // AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_CDEF_H_
