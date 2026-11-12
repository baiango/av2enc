// Shared AV2 secondary-transform core for the CUDA kernel and the host
// emulation test.
//
// This file is deliberately pure integer C++: no CUDA intrinsics, no
// transcendentals, no floating point.  It is compiled
//   * by nvcc as __host__ __device__ code for the real GPU kernel, and
//   * by the host compiler (without __CUDACC__) for the macOS/Linux emulation
//     parity test.
//
// Tables are passed as pointers so the caller can supply either the host
// constexpr tables (emulation) or the __device__ const copies (GPU).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define TXSEC_HD __host__ __device__
#else
#define TXSEC_HD inline
#endif

namespace av2enc::txsec::device {

struct Tables {
  const std::int8_t* ist4;        // [14][3][8][16]
  const std::int8_t* ist8;        // [11][3][32][48]
  const std::uint8_t* scan_map;   // [11][3][48]
  const std::uint8_t* tx_width;   // [25]
  const std::uint8_t* tx_height;  // [25]
  const std::uint8_t* tx_width_log2;  // [25]
  const std::uint8_t* scan_order4;    // [16]
  const std::uint8_t* scan_order8;    // [64]
};

struct SizeInfo {
  int w;
  int h;
  int bwl;
  int n;
  int n2;
  int scan_w;
  int kernel_count;
  bool large;
};

TXSEC_HD int min_int(int a, int b) { return a < b ? a : b; }

TXSEC_HD std::int64_t round2_signed(std::int64_t value, int n) {
  const std::int64_t magnitude = value < 0 ? -value : value;
  const std::int64_t rounded =
      (magnitude + (static_cast<std::int64_t>(1) << (n - 1))) >> n;
  return value < 0 ? -rounded : rounded;
}

TXSEC_HD std::int64_t clip_bit_depth(std::int64_t value, int bit_depth) {
  const std::int64_t high =
      (static_cast<std::int64_t>(1) << (bit_depth + 7)) - 1;
  const std::int64_t low = -(static_cast<std::int64_t>(1) << (bit_depth + 7));
  return value < low ? low : (value > high ? high : value);
}

// First `n` entries of the §5.20.7.30 TX_CLASS_2D diagonal scan.
TXSEC_HD void first_scan(int w, int h, int n, std::uint16_t* scan) {
  int x = 0;
  int y = 0;
  for (int c = 0; c < n; ++c) {
    scan[c] = static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = min_int(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
}

TXSEC_HD bool size_info(const Tables& t, int tx_size, int plane_tx_type,
                        SizeInfo* out) {
  if (out == nullptr || tx_size < 0 || tx_size >= 25) return false;
  SizeInfo info;
  info.w = min_int(32, static_cast<int>(t.tx_width[tx_size]));
  info.h = min_int(32, static_cast<int>(t.tx_height[tx_size]));
  info.bwl = min_int(5, static_cast<int>(t.tx_width_log2[tx_size]));
  info.large = info.w >= 8 && info.h >= 8;
  const int adst_adst = 3;  // TxType::kAdstAdst
  if (!info.large) {
    info.n = 8;
  } else if (tx_size == 1 /* TX_8X8 */ || plane_tx_type == adst_adst) {
    info.n = 20;
  } else {
    info.n = 32;
  }
  info.n2 = info.large ? 48 : 16;
  info.scan_w = info.large ? 8 : 4;
  info.kernel_count = info.large ? 11 : 14;
  *out = info;
  return true;
}

// Normative §7.15.3 inverse.  `dequant` is in-place, w*h elements.
TXSEC_HD bool inverse(const Tables& t, const SizeInfo& info, int kernel,
                      int sec_tx_type, int transpose, int bit_depth,
                      std::int32_t* dequant) {
  if (dequant == nullptr || kernel < 0 || kernel >= info.kernel_count) {
    return false;
  }
  if (sec_tx_type < 1 || sec_tx_type > 3 || bit_depth < 8 || bit_depth > 12) {
    return false;
  }
  const int elems = info.w * info.h;
  const int kernel_index = kernel * 3 + (sec_tx_type - 1);
  const int kernel_width = info.large ? 48 : 16;
  const std::int8_t* matrix =
      info.large ? t.ist8 + kernel_index * 32 * 48
                 : t.ist4 + kernel_index * 8 * 16;

  std::uint16_t scan[32];
  std::int64_t coefs[32];
  first_scan(info.w, info.h, info.n, scan);
  for (int j = 0; j < info.n; ++j) {
    const int pos = scan[j];
    coefs[j] = dequant[pos];
    dequant[pos] = 0;
  }

  const int scan_bwl = info.large ? 3 : 2;
  const int scan_mask = (1 << scan_bwl) - 1;
  for (int i = 0; i < info.n2; ++i) {
    std::int64_t sum = 0;
    for (int j = 0; j < info.n; ++j) {
      sum += coefs[j] *
             static_cast<std::int64_t>(matrix[j * kernel_width + i]);
    }
    const std::int64_t value =
        clip_bit_depth(round2_signed(sum, 7), bit_depth);
    const int pos =
        info.large ? t.scan_order8[t.scan_map[kernel_index * 48 + i]]
                   : t.scan_order4[i];
    const int x = pos & scan_mask;
    const int y = pos >> scan_bwl;
    const int index = transpose ? x * info.w + y : y * info.w + x;
    if (index < 0 || index >= elems) return false;
    dequant[index] = static_cast<std::int32_t>(value);
  }
  return true;
}

// Encoder-side analysis projection (not normative).  In-place, w*h elements.
TXSEC_HD bool forward(const Tables& t, const SizeInfo& info, int kernel,
                      int sec_tx_type, int transpose, int bit_depth,
                      std::int32_t* coeff) {
  if (coeff == nullptr || kernel < 0 || kernel >= info.kernel_count) {
    return false;
  }
  if (sec_tx_type < 1 || sec_tx_type > 3 || bit_depth < 8 || bit_depth > 12) {
    return false;
  }
  const int elems = info.w * info.h;
  const int kernel_index = kernel * 3 + (sec_tx_type - 1);
  const int kernel_width = info.large ? 48 : 16;
  const std::int8_t* matrix =
      info.large ? t.ist8 + kernel_index * 32 * 48
                 : t.ist4 + kernel_index * 8 * 16;

  const int scan_bwl = info.large ? 3 : 2;
  const int scan_mask = (1 << scan_bwl) - 1;
  std::int64_t full[48];
  for (int i = 0; i < info.n2; ++i) {
    const int pos =
        info.large ? t.scan_order8[t.scan_map[kernel_index * 48 + i]]
                   : t.scan_order4[i];
    const int x = pos & scan_mask;
    const int y = pos >> scan_bwl;
    const int index = transpose ? x * info.w + y : y * info.w + x;
    if (index < 0 || index >= elems) return false;
    full[i] = coeff[index];
  }

  for (int i = 0; i < elems; ++i) coeff[i] = 0;

  std::uint16_t scan[32];
  first_scan(info.w, info.h, info.n, scan);
  const std::int32_t int32_high = 0x7fffffff;
  const std::int32_t int32_low = -int32_high - 1;
  for (int j = 0; j < info.n; ++j) {
    std::int64_t sum = 0;
    for (int i = 0; i < info.n2; ++i) {
      sum += full[i] *
             static_cast<std::int64_t>(matrix[j * kernel_width + i]);
    }
    std::int64_t value = round2_signed(sum, 7);
    value = value < int32_low ? int32_low
                              : (value > int32_high ? int32_high : value);
    coeff[scan[j]] = static_cast<std::int32_t>(value);
  }
  return true;
}

}  // namespace av2enc::txsec::device
