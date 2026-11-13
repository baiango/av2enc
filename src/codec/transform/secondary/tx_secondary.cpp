// CPU reference for the AV2 secondary transform (IST).  See tx_secondary.h.
#include "codec/transform/secondary/tx_secondary.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace av2enc::txsec {
namespace {

constexpr int kMaxScanElems = 32 * 32;
constexpr int kMaxSecCoeffs = 32;
constexpr int kMaxOutCoeffs = 48;
constexpr int kAngleStep = 3;
constexpr int kMaxAngleDelta = 3;
constexpr int kMaxMrlIndex = 3;
constexpr int kPaethPred = 12;

int diagonal_scan(int w, int h, std::uint16_t* scan) {
  int x = 0;
  int y = 0;
  const int count = w * h;
  for (int c = 0; c < count; ++c) {
    scan[c] = static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
  return count;
}

// §4.8 / §8.2 Round2Signed(x, n): symmetric round, ties away from zero.
std::int64_t round2_signed(std::int64_t value, int n) {
  const std::int64_t magnitude = value < 0 ? -value : value;
  const std::int64_t rounded = (magnitude + (std::int64_t{1} << (n - 1))) >> n;
  return value < 0 ? -rounded : rounded;
}

std::int64_t clip_bit_depth(std::int64_t value, int bit_depth) {
  const std::int64_t high = (std::int64_t{1} << (bit_depth + 7)) - 1;
  const std::int64_t low = -(std::int64_t{1} << (bit_depth + 7));
  return value < low ? low : (value > high ? high : value);
}

int mrl_angle_delta(int mrl_index) noexcept {
  // Mrl_Index_To_Delta[4] = { 0, 1, -1, 0 }.
  return mrl_index == 1 ? 1 : (mrl_index == 2 ? -1 : 0);
}

}  // namespace

bool secondary_size_info(const TxParams& params, TxType plane_tx_type,
                         SecondarySizeInfo* out) {
  if (out == nullptr) return false;
  const int tx_size = static_cast<int>(params.tx_size);
  if (tx_size < 0 || tx_size >= 25) return false;
  if (kTxWidth[tx_size] != params.coded_w ||
      kTxHeight[tx_size] != params.coded_h) {
    return false;
  }

  SecondarySizeInfo info;
  info.tx_size = tx_size;
  info.coded_w = params.coded_w;
  info.coded_h = params.coded_h;
  info.w = std::min(32, static_cast<int>(kTxWidth[tx_size]));
  info.h = std::min(32, static_cast<int>(kTxHeight[tx_size]));
  info.bwl = std::min(5, static_cast<int>(kTxWidthLog2[tx_size]));
  info.large = info.w >= 8 && info.h >= 8;
  if (!info.large) {
    info.n = kIst4x4Height;
  } else if (tx_size == 1 /* TX_8X8 */ ||
             plane_tx_type == TxType::kAdstAdst) {
    info.n = kIst8x8HeightRed;
  } else {
    info.n = kIst8x8Height;
  }
  info.n2 = info.large ? kIst8x8Width : kIst4x4Width;
  info.scan_w = info.large ? 8 : 4;
  info.kernel_count = info.large ? kIstSetSize8x8 : kIstSetSize4x4;
  info.eob_limit = info.n;
  *out = info;
  return true;
}

int secondary_eob_limit(const TxParams& params, TxType plane_tx_type) {
  SecondarySizeInfo info;
  if (!secondary_size_info(params, plane_tx_type, &info)) return 0;
  return info.eob_limit;
}

bool is_directional_mode(int y_mode) noexcept {
  return y_mode >= 1 && y_mode <= 8;  // V_PRED .. D67_PRED
}

int map_y_mode(int y_mode, int angle_delta, int mrl_index, int coded_w,
               int coded_h) noexcept {
  if (y_mode < 0 || y_mode > 12 || y_mode == kPaethPred) return -1;
  if (!is_directional_mode(y_mode)) return y_mode;
  if (angle_delta < -kMaxAngleDelta || angle_delta > kMaxAngleDelta) return -1;
  if (mrl_index < 0 || mrl_index > kMaxMrlIndex) return -1;
  if (coded_w <= 0 || coded_h <= 0) return -1;

  const int p_angle = static_cast<int>(kModeToAngle[y_mode]) +
                      angle_delta * kAngleStep + mrl_angle_delta(mrl_index);

  // wide_angle_mapping(), §5.20.7.29.  Ratios use coded Tx_Width/Height.
  constexpr int kThresh2 = 61;
  constexpr int kThresh4 = 73;
  constexpr int kThresh8 = 82;
  constexpr int kThresh16 = 86;
  if ((coded_h == 2 * coded_w && p_angle < kThresh2) ||
      (coded_h == 4 * coded_w && p_angle < kThresh4) ||
      (coded_h == 8 * coded_w && p_angle < kThresh8) ||
      (coded_h == 16 * coded_w && p_angle < kThresh16)) {
    return 7;  // D203_PRED
  }
  if ((coded_w == 2 * coded_h && p_angle > 270 - kThresh2) ||
      (coded_w == 4 * coded_h && p_angle > 270 - kThresh4) ||
      (coded_w == 8 * coded_h && p_angle > 270 - kThresh8) ||
      (coded_w == 16 * coded_h && p_angle > 270 - kThresh16)) {
    return 3;  // D45_PRED
  }
  return y_mode;
}

int most_probable_stx_set_count(TxType plane_tx_type, int coded_w,
                                int coded_h) noexcept {
  if (plane_tx_type == TxType::kAdstAdst && coded_w >= 8 && coded_h >= 8) {
    return kIstReduceSetSizeAdstAdst;
  }
  return kIstDirSize;
}

bool kernel_for_syntax(int mapped_mode, TxType plane_tx_type,
                       int most_probable_stx_set, bool is_inter, int coded_w,
                       int coded_h, int* kernel) noexcept {
  if (kernel == nullptr) return false;
  if (is_inter) {
    *kernel = 0;  // §7.15.3: if (is_inter) kernel = 0
    return true;
  }
  if (mapped_mode < 0 || mapped_mode > 11) return false;

  const bool large = coded_w >= 8 && coded_h >= 8;
  int index = 0;
  if (plane_tx_type == TxType::kAdstAdst && large) {
    if (most_probable_stx_set < 0 ||
        most_probable_stx_set >= kIstReduceSetSizeAdstAdst) {
      return false;
    }
    index =
        kInvMostProbableStxMappingAdst[mapped_mode][most_probable_stx_set];
  } else {
    if (most_probable_stx_set < 0 ||
        most_probable_stx_set >= kIstDirSize) {
      return false;
    }
    index = kInvMostProbableStxMapping[mapped_mode][most_probable_stx_set];
  }
  if (plane_tx_type == TxType::kAdstAdst) index += 7;
  const int count = large ? kIstSetSize8x8 : kIstSetSize4x4;
  if (index < 0 || index >= count) return false;
  *kernel = index;
  return true;
}

bool transpose_for_mode(int mapped_mode) noexcept {
  // H_PRED, D157_PRED, D67_PRED, SMOOTH_H_PRED.
  return mapped_mode == 2 || mapped_mode == 6 || mapped_mode == 8 ||
         mapped_mode == 11;
}

bool secondary_syntax_applicable(int plane, int eob,
                                 const SecondarySyntax& syntax,
                                 const SecondaryTransformConfig& config) {
  if (plane != 0 || syntax.lossless) return false;
  const TxType type = syntax.plane_tx_type;
  if (type != TxType::kDctDct && type != TxType::kAdstAdst) return false;
  if (syntax.y_mode < 0 || syntax.y_mode > 12 ||
      syntax.y_mode == kPaethPred) {
    return false;
  }
  if (eob < 1) return false;
  if (!txv::tx_size_supported(syntax.coded_w, syntax.coded_h)) return false;

  const bool large = syntax.coded_w >= 8 && syntax.coded_h >= 8;
  const int limit =
      !large ? kIst4x4Height
             : ((syntax.coded_w == 8 && syntax.coded_h == 8) ||
                type == TxType::kAdstAdst
                    ? kIst8x8HeightRed
                    : kIst8x8Height);
  if (eob > limit) return false;
  if (syntax.is_inter) {
    if (!config.enable_inter_ist || eob <= 3 || type != TxType::kDctDct) {
      return false;
    }
    if (syntax.coded_w < 16 || syntax.coded_h < 16) return false;
    return true;
  }
  return config.enable_intra_ist && eob != 1;
}

bool secondary_syntax_valid(int plane, int eob, const SecondarySyntax& syntax,
                            const SecondaryTransformConfig& config) {
  if (!secondary_syntax_applicable(plane, eob, syntax, config)) return false;
  if (syntax.sec_tx_type < 1 || syntax.sec_tx_type >= kStxTypes) return false;
  if (syntax.bit_depth < 8 || syntax.bit_depth > 12) return false;

  const int mapped = map_y_mode(syntax.y_mode, syntax.angle_delta,
                                 syntax.mrl_index, syntax.coded_w,
                                 syntax.coded_h);
  if (mapped < 0) return false;
  if (!syntax.is_inter) {
    const int max_set =
        most_probable_stx_set_count(syntax.plane_tx_type, syntax.coded_w,
                                    syntax.coded_h);
    if (syntax.most_probable_stx_set < 0 ||
        syntax.most_probable_stx_set >= max_set) {
      return false;
    }
  }
  int kernel = 0;
  return kernel_for_syntax(mapped, syntax.plane_tx_type,
                           syntax.most_probable_stx_set, syntax.is_inter,
                           syntax.coded_w, syntax.coded_h, &kernel);
}

int enumerate_secondary_candidates(const SecondarySyntax& syntax,
                                   const SecondaryTransformConfig& config,
                                   int eob, SecondaryCandidate* out,
                                   int cap) noexcept {
  if (out == nullptr || cap <= 0) return 0;
  if (!secondary_syntax_applicable(0, eob, syntax, config)) return 0;
  const int mapped = map_y_mode(syntax.y_mode, syntax.angle_delta,
                                 syntax.mrl_index, syntax.coded_w,
                                 syntax.coded_h);
  if (mapped < 0) return 0;
  const bool transpose = transpose_for_mode(mapped);
  int count = 0;
  const int set_count = syntax.is_inter
                            ? 1
                            : most_probable_stx_set_count(
                                  syntax.plane_tx_type, syntax.coded_w,
                                  syntax.coded_h);
  for (int set = 0; set < set_count; ++set) {
    for (int sec = 1; sec < kStxTypes; ++sec) {
      int kernel = 0;
      if (!kernel_for_syntax(mapped, syntax.plane_tx_type, set,
                             syntax.is_inter, syntax.coded_w, syntax.coded_h,
                             &kernel)) {
        continue;
      }
      if (count >= cap) return count;
      out[count].sec_tx_type = sec;
      out[count].most_probable_stx_set = set;
      out[count].kernel = kernel;
      out[count].transpose = transpose;
      ++count;
    }
  }
  return count;
}

bool inverse_secondary_dequant_kernel(const TxParams& params,
                                      TxType plane_tx_type, int kernel,
                                      int sec_tx_type, bool transpose,
                                      int bit_depth, std::int32_t* dequant,
                                      std::size_t dequant_len) {
  SecondarySizeInfo info;
  if (!secondary_size_info(params, plane_tx_type, &info)) return false;
  if (sec_tx_type < 1 || sec_tx_type >= kStxTypes) return false;
  if (kernel < 0 || kernel >= info.kernel_count) return false;
  if (bit_depth < 8 || bit_depth > 12 || dequant == nullptr) return false;
  const std::size_t elems =
      static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
  if (dequant_len < elems) return false;

  std::uint16_t scan_in[kMaxScanElems];
  diagonal_scan(info.w, info.h, scan_in);

  std::int64_t coefs[kMaxSecCoeffs];
  for (int j = 0; j < info.n; ++j) {
    const int pos = scan_in[j];
    coefs[j] = dequant[pos];
    dequant[pos] = 0;
  }

  const int kernel_width = info.large ? kIst8x8Width : kIst4x4Width;
  const std::int8_t* matrix =
      info.large ? &kIst8x8Kernel[kernel][sec_tx_type - 1][0][0]
                 : &kIst4x4Kernel[kernel][sec_tx_type - 1][0][0];
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
        info.large ? kStxScanOrder8x8[kStxScanMap[kernel][sec_tx_type - 1][i]]
                   : kStxScanOrder4x4[i];
    const int x = pos & scan_mask;
    const int y = pos >> scan_bwl;
    const int index = transpose ? x * info.w + y : y * info.w + x;
    if (index < 0 || static_cast<std::size_t>(index) >= elems) return false;
    dequant[index] = static_cast<std::int32_t>(value);
  }
  return true;
}

SecondaryStatus apply_inverse_secondary(const TxParams& params, int plane,
                                        int eob,
                                        const SecondarySyntax& syntax,
                                        const SecondaryTransformConfig& config,
                                        std::int32_t* dequant,
                                        std::size_t dequant_len) {
  if (!secondary_syntax_applicable(plane, eob, syntax, config)) {
    return SecondaryStatus::kNotApplicable;
  }
  if (syntax.sec_tx_type == 0) return SecondaryStatus::kNotApplicable;
  if (!secondary_syntax_valid(plane, eob, syntax, config)) {
    return SecondaryStatus::kInvalid;
  }

  const int mapped = map_y_mode(syntax.y_mode, syntax.angle_delta,
                                 syntax.mrl_index, syntax.coded_w,
                                 syntax.coded_h);
  int kernel = 0;
  if (mapped < 0 ||
      !kernel_for_syntax(mapped, syntax.plane_tx_type,
                         syntax.most_probable_stx_set, syntax.is_inter,
                         syntax.coded_w, syntax.coded_h, &kernel)) {
    return SecondaryStatus::kInvalid;
  }
  if (!inverse_secondary_dequant_kernel(
          params, syntax.plane_tx_type, kernel, syntax.sec_tx_type,
          transpose_for_mode(mapped), syntax.bit_depth, dequant,
          dequant_len)) {
    return SecondaryStatus::kInvalid;
  }
  return SecondaryStatus::kApplied;
}

bool forward_secondary_coeffs_kernel(const TxParams& params,
                                     TxType plane_tx_type, int kernel,
                                     int sec_tx_type, bool transpose,
                                     int bit_depth, std::int32_t* coeff,
                                     std::size_t coeff_len) {
  SecondarySizeInfo info;
  if (!secondary_size_info(params, plane_tx_type, &info)) return false;
  if (sec_tx_type < 1 || sec_tx_type >= kStxTypes) return false;
  if (kernel < 0 || kernel >= info.kernel_count) return false;
  if (bit_depth < 8 || bit_depth > 12 || coeff == nullptr) return false;
  const std::size_t elems =
      static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
  if (coeff_len < elems) return false;

  const int kernel_width = info.large ? kIst8x8Width : kIst4x4Width;
  const std::int8_t* matrix =
      info.large ? &kIst8x8Kernel[kernel][sec_tx_type - 1][0][0]
                 : &kIst4x4Kernel[kernel][sec_tx_type - 1][0][0];
  const int scan_bwl = info.large ? 3 : 2;
  const int scan_mask = (1 << scan_bwl) - 1;

  std::int64_t full[kMaxOutCoeffs];
  for (int i = 0; i < info.n2; ++i) {
    const int pos =
        info.large ? kStxScanOrder8x8[kStxScanMap[kernel][sec_tx_type - 1][i]]
                   : kStxScanOrder4x4[i];
    const int x = pos & scan_mask;
    const int y = pos >> scan_bwl;
    const int index = transpose ? x * info.w + y : y * info.w + x;
    if (index < 0 || static_cast<std::size_t>(index) >= elems) return false;
    full[i] = coeff[index];
  }

  std::memset(coeff, 0, elems * sizeof(coeff[0]));

  std::uint16_t scan_in[kMaxScanElems];
  diagonal_scan(info.w, info.h, scan_in);
  const std::int32_t int32_high = 0x7fffffff;
  const std::int32_t int32_low = -int32_high - 1;
  for (int j = 0; j < info.n; ++j) {
    std::int64_t sum = 0;
    for (int i = 0; i < info.n2; ++i) {
      sum += full[i] *
             static_cast<std::int64_t>(matrix[j * kernel_width + i]);
    }
    std::int64_t value = round2_signed(sum, 7);
    value = std::max<std::int64_t>(int32_low,
                                   std::min<std::int64_t>(int32_high, value));
    coeff[scan_in[j]] = static_cast<std::int32_t>(value);
  }
  return true;
}

bool project_ist_levels(const TxParams& params, TxType plane_tx_type,
                        int kernel, int sec_tx_type, bool transpose,
                        std::int32_t* levels, std::size_t levels_len) {
  SecondarySizeInfo info;
  if (!secondary_size_info(params, plane_tx_type, &info)) return false;
  if (sec_tx_type < 1 || sec_tx_type >= kStxTypes) return false;
  if (kernel < 0 || kernel >= info.kernel_count) return false;
  if (levels == nullptr) return false;
  const std::size_t elems =
      static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
  if (levels_len < elems) return false;

  const int kernel_width = info.large ? kIst8x8Width : kIst4x4Width;
  const std::int8_t* matrix =
      info.large ? &kIst8x8Kernel[kernel][sec_tx_type - 1][0][0]
                 : &kIst4x4Kernel[kernel][sec_tx_type - 1][0][0];
  const int scan_bwl = info.large ? 3 : 2;
  const int scan_mask = (1 << scan_bwl) - 1;

  std::vector<std::uint16_t> scan(elems);
  diagonal_scan(info.w, info.h, scan.data());

  std::int32_t x[kMaxOutCoeffs];
  std::size_t out_pos[kMaxOutCoeffs];
  for (int i = 0; i < info.n2; ++i) {
    const int pos =
        info.large ? kStxScanOrder8x8[kStxScanMap[kernel][sec_tx_type - 1][i]]
                   : kStxScanOrder4x4[i];
    const int sx = pos & scan_mask;
    const int sy = pos >> scan_bwl;
    const int index = transpose ? sx * info.w + sy : sy * info.w + sx;
    if (index < 0 || static_cast<std::size_t>(index) >= elems) return false;
    out_pos[i] = static_cast<std::size_t>(index);
    x[i] = levels[out_pos[i]];
  }

  std::int32_t sec[kMaxSecCoeffs];
  const std::int32_t int32_high = 0x7fffffff;
  const std::int32_t int32_low = -int32_high - 1;
  for (int j = 0; j < info.n; ++j) {
    std::int64_t sum = 0;
    for (int i = 0; i < info.n2; ++i) {
      sum += static_cast<std::int64_t>(x[i]) *
             static_cast<std::int64_t>(matrix[j * kernel_width + i]);
    }
    std::int64_t value = round2_signed(sum, 7);
    value = std::max<std::int64_t>(int32_low,
                                   std::min<std::int64_t>(int32_high, value));
    sec[j] = static_cast<std::int32_t>(value);
  }

  // The decoder reads the first n scan positions, zeroes them, and writes the
  // n2 output positions.  Positions it overwrites must not be coded again.
  const auto in_first_n = [&](std::size_t index) {
    for (int j = 0; j < info.n; ++j) {
      if (static_cast<std::size_t>(scan[static_cast<std::size_t>(j)]) == index) {
        return true;
      }
    }
    return false;
  };
  for (int i = 0; i < info.n2; ++i) {
    if (!in_first_n(out_pos[i])) {
      levels[out_pos[i]] = 0;
    }
  }
  for (int j = 0; j < info.n; ++j) {
    levels[scan[static_cast<std::size_t>(j)]] = sec[j];
  }
  return true;
}

const std::uint16_t* sec_tx_type_cdf(int tx_size, bool is_inter) noexcept {
  if (tx_size < 0 || tx_size >= 25) return nullptr;
  return &kSecTxTypeCdf[is_inter ? 1 : 0][kTxSizeSqr[tx_size]][0];
}

const std::uint16_t* most_probable_stx_set_cdf(TxType plane_tx_type,
                                               int coded_w,
                                               int coded_h) noexcept {
  if (plane_tx_type == TxType::kAdstAdst && coded_w >= 8 && coded_h >= 8) {
    return &kMostProbableStxSetAdstCdf[0];
  }
  return &kMostProbableStxSetCdf[0];
}

namespace {

double cdf_symbol_bits(const std::uint16_t* cdf, int symbols,
                       int symbol) noexcept {
  if (cdf == nullptr || symbols <= 0 || symbol < 0 || symbol >= symbols) {
    return 0.0;
  }
  const int low = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
  const int high =
      symbol == symbols - 1 ? 32768 : static_cast<int>(cdf[symbol]);
  const double probability = static_cast<double>(high - low) / 32768.0;
  if (probability <= 0.0) return 1.0e9;
  return -std::log2(probability);
}

}  // namespace

double sec_tx_type_bits(int symbol, int tx_size, bool is_inter) noexcept {
  return cdf_symbol_bits(sec_tx_type_cdf(tx_size, is_inter), kStxTypes,
                         symbol);
}

double most_probable_stx_set_bits(int symbol, TxType plane_tx_type,
                                  int coded_w, int coded_h) noexcept {
  const int symbols =
      most_probable_stx_set_count(plane_tx_type, coded_w, coded_h);
  return cdf_symbol_bits(
      most_probable_stx_set_cdf(plane_tx_type, coded_w, coded_h), symbols,
      symbol);
}

void write_secondary_transform_header_bits(BitWriter& writer,
                                           const SecondaryTransformConfig& c) {
  // §5.4.8 order: enable_intra_ist, enable_inter_ist.
  writer.put_bit(c.enable_intra_ist);
  writer.put_bit(c.enable_inter_ist);
}

}  // namespace av2enc::txsec
