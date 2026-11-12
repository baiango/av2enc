// Coefficient-CDF quantizer-context selection (AV2 v1.0.0 §8.3.1).
//
// The §9.3 default coefficient CDFs are declared with a leading
// `COEFF_CDF_Q_CTXS` dimension of four slices.  `init_coeff_cdfs()` picks one
// slice per frame from the *raw frame* `base_q_idx`:
//
//     base_q_idx <=  90  -> slice 0
//     base_q_idx <= 140  -> slice 1
//     base_q_idx <= 190  -> slice 2
//     otherwise          -> slice 3
//
// It is NOT derived from a tile qindex, a delta-Q qindex, or a segmentation
// qindex.  Those change the effective quantizer, not this frame-level CDF
// initialization.
//
// `qctx_cdf_tables.h` is generated from spec/tables/default_cdf.json by
// generate_qctx_tables.py; do not hand-edit either table file.
//
// Integration (M1/M2 writers):
//
//     const auto& q = av2enc::still::qctx::tables_for_base_q_idx(config.base_q_idx);
//     copy_array(cdfs.txb_skip, *q.txb_skip);
//     copy_array(cdfs.coeff_base, *q.coeff_base);
//     ...
//
// `q.<field>` is a pointer to the inner (slice-removed) array type, so
// `*q.<field>` is exactly the array reference the existing `copy_array`
// helper wants.
#pragma once

#include "codec/entropy/qctx/qctx_cdf_tables.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace av2enc::still::qctx {

inline constexpr int kSliceCount = kCoeffCdfSliceCount;

// §8.3.1 boundaries, inclusive upper bounds.
inline constexpr int kSlice0MaxBaseQIdx = 90;
inline constexpr int kSlice1MaxBaseQIdx = 140;
inline constexpr int kSlice2MaxBaseQIdx = 190;

// Slice index 0..3 for the raw frame base_q_idx.
//
// Negative inputs (invalid, but cheap to make safe) map to slice 0.  There is
// deliberately no upper clamp: 10-bit streams can code base_q_idx up to 303
// and every value above 190 selects slice 3.
constexpr int slice_for_base_q_idx(int base_q_idx) noexcept {
  if (base_q_idx <= kSlice0MaxBaseQIdx) return 0;
  if (base_q_idx <= kSlice1MaxBaseQIdx) return 1;
  if (base_q_idx <= kSlice2MaxBaseQIdx) return 2;
  return 3;
}

// Clamped array-index access, for callers that already have a slice index.
constexpr const CoeffCdfSlice& tables_for_slice(int slice) noexcept {
  return kCoeffCdfSlices[slice <= 0 ? 0
                                    : (slice >= kSliceCount ? kSliceCount - 1
                                                            : slice)];
}

// The only selection entry point frame code needs.
constexpr const CoeffCdfSlice& tables_for_base_q_idx(int base_q_idx) noexcept {
  return tables_for_slice(slice_for_base_q_idx(base_q_idx));
}

// Generic, descriptor-driven access used by tests/tools.  `info` must come
// from kCoeffCdfTables.  For core syntax code prefer the typed
// `tables_for_base_q_idx(...).field` accessors above.
constexpr const std::uint16_t* table_slice(const CoeffCdfTableInfo& info,
                                           int slice) noexcept {
  return info.slice0 +
         static_cast<std::size_t>(slice) * info.slice_stride;
}

// Compile-time-checked array copy.  Equivalent to the memcpy-based helpers in
// still_syntax.cpp/video_syntax.cpp, but usable in constexpr contexts and without
// <cstring>.  `N` is deduced, so a type/shape mismatch is a compile error.
// Recurses through multidimensional arrays.
template <typename T, std::size_t N>
constexpr void copy_cdf(T (&dst)[N], const T (&src)[N]) noexcept {
  for (std::size_t i = 0; i < N; ++i) {
    if constexpr (std::is_array_v<T>) {
      copy_cdf(dst[i], src[i]);
    } else {
      dst[i] = src[i];
    }
  }
}

}  // namespace av2enc::still::qctx
