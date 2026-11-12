// Device-side AV2 §7.20.4 pixel-classified Wiener filter.
//
// Shared by the CUDA kernels (pc_wiener.cu) and the host-emulation parity test
// (pc_wiener_emulation_test.cpp).  The arithmetic is an exact transcription of
// src/encoder/filters/spec_tables/pc_wiener.cpp:
//
//   * §7.20.2 luma sample sourcing (CurrFrame outside a 64-line stripe,
//     CdefFrame inside, two-line clamp),
//   * §7.14.2 get_q / get_qval_given_tskip with Pc_Wiener_Mode_Weights and
//     Pc_Wiener_Mode_Offsets (8, 10 and 12 bit),
//   * §7.20.4 get_features / get_box_features (Pc_Wiener_Lead/Lag box),
//   * the 4096-entry LUT index -> 64-class mapping (Pc_Wiener_LutToClass),
//   * Pc_Wiener_Sub_Classify (class -> filter index) and the 25-tap
//     Pc_Wiener_Filters bank,
//   * Pc_Wiener_Sub_Classify2 (filter-set/num-classes -> subclass).
//
// The normative tables are not duplicated here.  The host launcher uploads the
// CPU tables (src/encoder/filters/spec_tables/pc_wiener_tables.h) into the
// persistent arena once; this header only ever sees a `PcWienerTables` view of
// device memory.  The emulation harness fills the same view with host pointers,
// so both builds compile and run this single implementation.
//
// Intentionally integer-only: no float, no fast-math, no tensor cores.

#ifndef AV2ENC_CUDA_PC_WIENER_PC_WIENER_DEVICE_CUH_
#define AV2ENC_CUDA_PC_WIENER_PC_WIENER_DEVICE_CUH_

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::pc_wiener {

// §7.20.4 constants (the printed specification text).
inline constexpr int kTaps = 25;
inline constexpr int kPrecBits = 7;
inline constexpr int kPrecFeature = 14;
inline constexpr int kNumFeatures = 4;
inline constexpr int kLead = 1;
inline constexpr int kLag = 4;
inline constexpr int kMiSize = 4;

// Table extents (src/encoder/filters/spec_tables/pc_wiener_tables.h).
inline constexpr int kFilterSets = 4;
inline constexpr int kFiltersPerSet = 64;
inline constexpr int kCoeffsPerFilter = 13;  // 25 taps -> i >> 1
inline constexpr int kLutEntries = 4096;     // 8 (bits) * 4 features
inline constexpr int kClassEntries = 256;    // Pc_Wiener_Sub_Classify axis
inline constexpr int kSubClassEntries = 256;
inline constexpr int kSubClassSets = 8;      // Pc_Wiener_Sub_Classify2 axis
inline constexpr int kQlookupEntries = 352;
inline constexpr int kAcQlookupEntries = 25;

// Device view of the uploaded normative tables.  All pointers are device
// memory in the CUDA build and host memory in the emulation build.
struct PcWienerTables {
  const std::int16_t* filters;        // [4][64][13]
  const std::uint8_t* lut_to_class;   // [4096]
  const std::uint8_t* sub_classify;   // [4][256]
  const std::uint8_t* sub_classify2;  // [4][8][256]
  const std::int32_t* qlookup;        // [352]
  const std::int32_t* ac_qlookup;     // [25]
  const std::int8_t* taps;            // [25][2] = {dy, dx}
  const std::int32_t* normalizer;     // [5]
  const std::int32_t* mode_weights;   // [4][3]
  const std::int32_t* mode_offsets;   // [4]
};

// §7.20.1/§7.20.2 luma source for the current stripe.
struct PcWienerSource {
  const std::uint16_t* curr;
  const std::uint16_t* cdef;
  int curr_stride;
  int cdef_stride;
  int luma_start_x;
  int luma_end_x;
  int luma_start_y;
  int luma_end_y;
  int stripe_start_y;
  int stripe_end_y;
};

// §7.20.4 per-block configuration.
struct PcWienerParams {
  int base_q_idx;
  int bit_depth;  // 8, 10 or 12
  int mi_col_end;   // §7.20.1 MiColEnd
  int mi_row_start;  // §7.20.1 MiRowStart (tile clamp for LrTxSkip)
  int mi_row_end;    // §7.20.1 MiRowEnd
  const std::uint8_t* lr_tx_skip;  // luma MI grid
  int lr_tx_skip_stride;
};

// ---------------------------------------------------------------------------
// §4.8 integer math
// ---------------------------------------------------------------------------

__device__ __forceinline__ int pcw_clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ std::int64_t pcw_round2(std::int64_t value,
                                                   int shift) {
  return shift == 0 ? value
                    : (value + (std::int64_t{1} << (shift - 1))) >> shift;
}

__device__ __forceinline__ std::int64_t pcw_round2_signed(std::int64_t value,
                                                          int shift) {
  return value >= 0 ? pcw_round2(value, shift) : -pcw_round2(-value, shift);
}

__device__ __forceinline__ int pcw_abs(int value) {
  return value < 0 ? -value : value;
}

// §7.14.2 MaxQ (12 bit is kept internal, see spec_math.h).
__device__ __forceinline__ int pcw_max_q(int bit_depth) {
  return bit_depth == 8 ? 255 : (bit_depth == 10 ? 303 : 351);
}

// §7.14.2 get_q(qindex, delta).
__device__ __forceinline__ int pcw_get_q(const PcWienerTables& tables,
                                         int qindex, int delta,
                                         int bit_depth) {
  if (qindex == 0 && delta <= 0) {
    return tables.ac_qlookup[0];
  }
  const int clamped = pcw_clip3(1, pcw_max_q(bit_depth), qindex + delta);
  return tables.qlookup[clamped];
}

// §7.20.4 get_qval_given_tskip(qindex, tskip, i).
//
// Pc_Wiener_Mode_Weights[4][3] and Pc_Wiener_Mode_Offsets[4] come from the
// uploaded table, so there is one normative source only.
__device__ __forceinline__ std::int64_t pcw_qval_given_tskip(
    const PcWienerTables& tables, int qindex, std::int64_t tskip, int i,
    int bit_depth) {
  std::int64_t qstep = pcw_get_q(tables, qindex, 0, bit_depth);
  const int qstep_shift = 3 + 10;  // QUANT_TABLE_BITS + 10
  qstep = pcw_round2(qstep, bit_depth - 8);
  const int diff_shift = qstep_shift - 8;
  const std::int64_t prod = pcw_round2(tskip * qstep, 8);
  const std::int64_t qval =
      static_cast<std::int64_t>(tables.mode_weights[i * 3 + 0]) *
          (tskip << diff_shift) +
      static_cast<std::int64_t>(tables.mode_weights[i * 3 + 1]) * qstep +
      static_cast<std::int64_t>(tables.mode_weights[i * 3 + 2]) * prod;
  return 255 * (static_cast<std::int64_t>(tables.mode_offsets[i]) +
                pcw_round2_signed(qval, qstep_shift));
}

// ---------------------------------------------------------------------------
// §7.20.2 get_source_sample(0, x, y)
// ---------------------------------------------------------------------------

__device__ __forceinline__ int pcw_luma_sample(const PcWienerSource& src,
                                               int x, int y) {
  x = pcw_clip3(src.luma_start_x, src.luma_end_x, x);
  y = pcw_clip3(src.luma_start_y, src.luma_end_y, y);
  if (y < src.stripe_start_y) {
    y = src.stripe_start_y - 2 > y ? src.stripe_start_y - 2 : y;
    return static_cast<int>(
        src.curr[static_cast<std::ptrdiff_t>(y) * src.curr_stride + x]);
  }
  if (y > src.stripe_end_y) {
    y = src.stripe_end_y + 2 < y ? src.stripe_end_y + 2 : y;
    return static_cast<int>(
        src.curr[static_cast<std::ptrdiff_t>(y) * src.curr_stride + x]);
  }
  return static_cast<int>(
      src.cdef[static_cast<std::ptrdiff_t>(y) * src.cdef_stride + x]);
}

// ---------------------------------------------------------------------------
// §7.20.4 features and classification
// ---------------------------------------------------------------------------

// §7.20.4 get_tx_skip(x, y), including the block/stripe/tile clamps.
__device__ __forceinline__ int pcw_tx_skip(const PcWienerSource& src,
                                           const PcWienerParams& params,
                                           int block_start_x,
                                           int block_end_x, int x, int y) {
  x = block_end_x < x ? block_end_x : x;
  x = block_start_x > x ? block_start_x : x;
  y = pcw_clip3(src.stripe_start_y, src.stripe_end_y, y);
  y = pcw_clip3(params.mi_row_start * kMiSize,
                params.mi_row_end * kMiSize - 1, y);
  return params.lr_tx_skip[static_cast<std::ptrdiff_t>(y >> 2) *
                               params.lr_tx_skip_stride +
                           (x >> 2)];
}

// §7.20.4 get_features(x, y).
__device__ __forceinline__ void pcw_features(const PcWienerSource& src,
                                             const PcWienerParams& params,
                                             int block_start_x,
                                             int block_end_x, int x, int y,
                                             std::int64_t f[kNumFeatures],
                                             int* skip_out) {
  x = block_end_x + 2 < x ? block_end_x + 2 : x;
  const int m = pcw_luma_sample(src, x, y);
  const int up = pcw_luma_sample(src, x, y - 1);
  const int down = pcw_luma_sample(src, x, y + 1);
  const int upright = pcw_luma_sample(src, x + 1, y - 1);
  const int downleft = pcw_luma_sample(src, x - 1, y + 1);
  const int downright = pcw_luma_sample(src, x + 1, y + 1);
  const int upleft = pcw_luma_sample(src, x - 1, y - 1);

  f[0] = 0;
  f[1] = pcw_abs(up - 2 * m + down);
  f[2] = pcw_abs(upright - 2 * m + downleft);
  f[3] = pcw_abs(upleft - 2 * m + downright);
  *skip_out = pcw_tx_skip(src, params, block_start_x, block_end_x, x, y);
}

// §7.20.4 get_box_features(x, y).
__device__ __forceinline__ void pcw_box_features(
    const PcWienerTables& tables, const PcWienerSource& src,
    const PcWienerParams& params, int block_start_x, int block_end_x, int x,
    int y, std::int64_t f[kNumFeatures], std::int64_t* tskip_out) {
  f[0] = 0;
  f[1] = 0;
  f[2] = 0;
  f[3] = 0;
  std::int64_t skip_sum = 0;
  for (int dy = -kLead; dy <= kLag; ++dy) {
    for (int dx = -kLead; dx <= kLag; ++dx) {
      std::int64_t tf[kNumFeatures];
      int skip = 0;
      pcw_features(src, params, block_start_x, block_end_x, x + dx, y + dy, tf,
                   &skip);
      f[0] += tf[0];
      f[1] += tf[1];
      f[2] += tf[2];
      f[3] += tf[3];
      skip_sum += skip;
    }
  }
  const int shift = params.bit_depth - 8;
  f[0] = pcw_round2(f[0] * tables.normalizer[0], shift);
  f[1] = pcw_round2(f[1] * tables.normalizer[1], shift);
  f[2] = pcw_round2(f[2] * tables.normalizer[2], shift);
  f[3] = pcw_round2(f[3] * tables.normalizer[3], shift);
  *tskip_out = skip_sum * tables.normalizer[4];
}

// §7.20.4 get_filter_set_index().
__device__ __forceinline__ int pcw_filter_set_index(int base_q_idx) {
  if (base_q_idx < 130) return 0;
  if (base_q_idx < 190) return 1;
  if (base_q_idx < 220) return 2;
  return 3;
}

// §7.20.4 class derivation (the part before `if (skipFilter)`).
__device__ __forceinline__ int pcw_class(const PcWienerTables& tables,
                                         const PcWienerSource& src,
                                         const PcWienerParams& params, int x,
                                         int y) {
  const int block_start_x = (x >> 6) << 6;
  const int block_end_x =
      (params.mi_col_end * kMiSize - 1) < (block_start_x + 63)
          ? (params.mi_col_end * kMiSize - 1)
          : (block_start_x + 63);
  std::int64_t f[kNumFeatures];
  std::int64_t tskip = 0;
  pcw_box_features(tables, src, params, block_start_x, block_end_x, x, y, f,
                   &tskip);

  int lut_input = 0;
  for (int i = 0; i < kNumFeatures; ++i) {
    std::int64_t qval = pcw_round2_signed(
        f[i] + pcw_qval_given_tskip(tables, params.base_q_idx, tskip, i,
                                    params.bit_depth),
        kPrecFeature);
    qval = qval < 0 ? 0 : (qval > 255 ? 255 : qval);
    qval >>= 5;
    lut_input += static_cast<int>(qval) << (3 * (3 - i));
  }
  return tables.lut_to_class[lut_input];
}

// Pc_Wiener_Sub_Classify[filter_set][class] -> filter bank index.
__device__ __forceinline__ int pcw_filter_index(const PcWienerTables& tables,
                                                int filter_set, int cls) {
  return tables.sub_classify[filter_set * kClassEntries + cls];
}

// Pc_Wiener_Sub_Classify2[filter_set][num_classes_idx][class] -> subclass.
__device__ __forceinline__ int pcw_subclass(const PcWienerTables& tables,
                                            int filter_set, int num_classes_idx,
                                            int cls) {
  return tables
      .sub_classify2[(filter_set * kSubClassSets + num_classes_idx) *
                         kSubClassEntries +
                     cls];
}

// ---------------------------------------------------------------------------
// §7.20.4 filtering
// ---------------------------------------------------------------------------

// Filter a w x h block with an explicit class (the class is a function of the
// block origin in the specification; callers that already have the class grid
// use this form).
__device__ __forceinline__ void pcw_filter_block_class(
    const PcWienerTables& tables, const PcWienerSource& src,
    const PcWienerParams& params, int x, int y, int w, int h, int cls,
    std::uint16_t* lr, int lr_stride) {
  const int index = pcw_filter_set_index(params.base_q_idx);
  const int filt = pcw_filter_index(tables, index, cls);
  const std::int16_t* coeffs =
      tables.filters + static_cast<std::ptrdiff_t>(index * kFiltersPerSet + filt) *
                           kCoeffsPerFilter;
  const int max_value = (1 << params.bit_depth) - 1;

  for (int r2 = 0; r2 < h; ++r2) {
    for (int c2 = 0; c2 < w; ++c2) {
      const int m = pcw_luma_sample(src, x + c2, y + r2);
      std::int64_t s = static_cast<std::int64_t>(m) << kPrecBits;
#ifdef __CUDACC__
#pragma unroll
#endif
      for (int i = 0; i < kTaps; ++i) {
        const int dy = tables.taps[i * 2 + 0];
        const int dx = tables.taps[i * 2 + 1];
        const int sample =
            pcw_luma_sample(src, x + c2 + dx, y + r2 + dy);
        s += static_cast<std::int64_t>(sample) * coeffs[i >> 1];
      }
      const int v = static_cast<int>(pcw_round2(s, kPrecBits));
      lr[static_cast<std::ptrdiff_t>(y + r2) * lr_stride + (x + c2)] =
          static_cast<std::uint16_t>(
              pcw_clip3(0, max_value, v));
    }
  }
}

// Filter a block, deriving its class from (x, y) as §7.20.4 does.
__device__ __forceinline__ void pcw_filter_block(
    const PcWienerTables& tables, const PcWienerSource& src,
    const PcWienerParams& params, int x, int y, int w, int h,
    std::uint16_t* lr, int lr_stride) {
  const int cls = pcw_class(tables, src, params, x, y);
  pcw_filter_block_class(tables, src, params, x, y, w, h, cls, lr, lr_stride);
}

// §7.20.4 with skipFilter = 1: FilterClass[y >> 2][x >> 2] only.
__device__ __forceinline__ void pcw_classify_block(
    const PcWienerTables& tables, const PcWienerSource& src,
    const PcWienerParams& params, int x, int y, std::uint8_t* filter_class,
    int filter_class_stride) {
  const int cls = pcw_class(tables, src, params, x, y);
  filter_class[static_cast<std::ptrdiff_t>(y >> 2) * filter_class_stride +
               (x >> 2)] = static_cast<std::uint8_t>(cls);
}

// ---------------------------------------------------------------------------
// §7.20.1 stripe and loop-restoration-unit helpers
// ---------------------------------------------------------------------------

// StripeBounds(row, ...) from pc_wiener_stream.cpp: 64-luma-line stripes with
// the +8 offset used by §7.20.1.
__device__ __forceinline__ void pcw_stripe_bounds(int y, int height, int* start,
                                                  int* end) {
  const int stripe_num = (y + 8) / 64;
  const int unclipped_start = stripe_num * 64 - 8;
  const int unclipped_end = unclipped_start + 64;
  *start = unclipped_start > 0 ? unclipped_start : 0;
  *end = (unclipped_end - 1) < (height - 1) ? (unclipped_end - 1)
                                            : (height - 1);
}

// CountUnits(unit_size, frame_size) from pc_wiener_stream.cpp.
__device__ __forceinline__ int pcw_count_units(int unit_size, int frame_size) {
  if (unit_size <= 0) return 1;
  const int n = (frame_size + (unit_size >> 1)) / unit_size;
  return n > 1 ? n : 1;
}

// UnitIndexForBlock(row, col, ...) from pc_wiener_stream.cpp.
__device__ __forceinline__ int pcw_unit_index(int row, int col, int unit_size,
                                              int unit_rows, int unit_cols) {
  const int unit_row =
      (unit_rows - 1) < ((row * kMiSize + 8) / unit_size)
          ? (unit_rows - 1)
          : ((row * kMiSize + 8) / unit_size);
  const int unit_col = (unit_cols - 1) < ((col * kMiSize) / unit_size)
                           ? (unit_cols - 1)
                           : ((col * kMiSize) / unit_size);
  return unit_row * unit_cols + unit_col;
}

}  // namespace av2enc::cuda::pc_wiener

#endif  // AV2ENC_CUDA_PC_WIENER_PC_WIENER_DEVICE_CUH_
