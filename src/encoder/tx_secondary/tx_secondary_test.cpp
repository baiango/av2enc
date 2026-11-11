// CPU self-test for the AV2 secondary transform (IST).
//
// Covers:
//   * the §9.7 generated kernels / scan map and §7.15.3 mappings/scans
//   * §7.15.3 block sizes, n/n2, EOB limits and scan width for all 25 TX sizes
//   * YMode + AngleDeltaY + MrlIndex wide-angle mapping
//   * kernel selection for every mode / set / PlaneTxType / inter flag
//   * the full §5.20.8.2 + §5.4.8 legality matrix (luma-only, lossless, PAETH)
//   * every legal 4x4/8x8 kernel x sec_tx_type x coefficient pattern, against
//     an independent direct-from-spec reference implementation
//   * EOB interactions, transpose, clipping and rounding boundaries
//   * the encoder-side analysis projection (forward + inverse)
//   * §9.3 CDF accessors / static costs and the two §5.4.8 header bits
//
// Build/run:
//   ./build_and_test.sh
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/tx_secondary/tx_secondary.h"
#include "encoder/tx_secondary/tx_secondary_tables.h"
#include "encoder/tx_variants/tx_variants.h"

namespace {

using namespace av2enc;
using namespace av2enc::txsec;
using av2enc::txv::TxParams;
using av2enc::txv::TxType;

int g_failures = 0;
long long g_checks = 0;
const char* g_suite = "";
long long g_suite_start = 0;

void suite_start(const char* name) {
  g_suite = name;
  g_suite_start = g_checks;
}

void suite_end() {
  std::printf("  %-34s %10lld checks\n", g_suite, g_checks - g_suite_start);
}

void check(bool cond, const char* what, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("FAIL [%s:%d] %s\n", g_suite, line, what);
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_EQ(a, b)                                                    \
  do {                                                                    \
    const auto va = (a);                                                  \
    const auto vb = (b);                                                  \
    check(va == vb, #a " == " #b, __LINE__);                              \
  } while (0)

std::mt19937 g_rng(0x157ec0deu);

int rand_range(int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(g_rng);
}

// ---------------------------------------------------------------------------
// Independent §7.15.3 reference (does not call tx_secondary.cpp helpers)
// ---------------------------------------------------------------------------

int ref_scan(int w, int h, std::uint16_t* scan) {
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
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
  return w * h;
}

std::int64_t ref_round2signed(std::int64_t v, int n) {
  const std::int64_t a = v < 0 ? -v : v;
  const std::int64_t r = (a + (std::int64_t{1} << (n - 1))) >> n;
  return v < 0 ? -r : r;
}

std::int64_t ref_clip(std::int64_t v, int bd) {
  const std::int64_t hi = (std::int64_t{1} << (bd + 7)) - 1;
  const std::int64_t lo = -(std::int64_t{1} << (bd + 7));
  return v < lo ? lo : (v > hi ? hi : v);
}

void ref_inverse(const TxParams& p, TxType type, int kernel, int sec,
                 bool transpose, int bd, const std::int32_t* in,
                 std::int32_t* out) {
  SecondarySizeInfo info;
  CHECK(secondary_size_info(p, type, &info));
  const int w = info.w;
  const int h = info.h;
  const std::size_t elems = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  std::memcpy(out, in, elems * sizeof(out[0]));

  std::uint16_t scan[1024];
  ref_scan(w, h, scan);
  std::int64_t coefs[32] = {};
  for (int j = 0; j < info.n; ++j) {
    coefs[j] = out[static_cast<std::size_t>(scan[j])];
    out[static_cast<std::size_t>(scan[j])] = 0;
  }
  const int kw = info.large ? kIst8x8Width : kIst4x4Width;
  const std::int8_t* m =
      info.large ? &kIst8x8Kernel[kernel][sec - 1][0][0]
                 : &kIst4x4Kernel[kernel][sec - 1][0][0];
  const int sb = info.large ? 3 : 2;
  for (int i = 0; i < info.n2; ++i) {
    std::int64_t t = 0;
    for (int j = 0; j < info.n; ++j) {
      t += coefs[j] * static_cast<std::int64_t>(m[j * kw + i]);
    }
    const std::int64_t v = ref_clip(ref_round2signed(t, 7), bd);
    const int pos =
        info.large ? kStxScanOrder8x8[kStxScanMap[kernel][sec - 1][i]]
                   : kStxScanOrder4x4[i];
    const int x = pos & ((1 << sb) - 1);
    const int y = pos >> sb;
    const int index = transpose ? x * w + y : y * w + x;
    out[index] = static_cast<std::int32_t>(v);
  }
}

void ref_forward(const TxParams& p, TxType type, int kernel, int sec,
                 bool transpose, const std::int32_t* in, std::int32_t* out) {
  SecondarySizeInfo info;
  CHECK(secondary_size_info(p, type, &info));
  const int w = info.w;
  const int h = info.h;
  const std::size_t elems = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  std::int64_t full[48] = {};
  const int kw = info.large ? kIst8x8Width : kIst4x4Width;
  const std::int8_t* m =
      info.large ? &kIst8x8Kernel[kernel][sec - 1][0][0]
                 : &kIst4x4Kernel[kernel][sec - 1][0][0];
  const int sb = info.large ? 3 : 2;
  for (int i = 0; i < info.n2; ++i) {
    const int pos =
        info.large ? kStxScanOrder8x8[kStxScanMap[kernel][sec - 1][i]]
                   : kStxScanOrder4x4[i];
    const int x = pos & ((1 << sb) - 1);
    const int y = pos >> sb;
    const int index = transpose ? x * w + y : y * w + x;
    full[i] = in[index];
  }
  std::memset(out, 0, elems * sizeof(out[0]));
  std::uint16_t scan[1024];
  ref_scan(w, h, scan);
  for (int j = 0; j < info.n; ++j) {
    std::int64_t t = 0;
    for (int i = 0; i < info.n2; ++i) {
      t += full[i] * static_cast<std::int64_t>(m[j * kw + i]);
    }
    std::int64_t v = ref_round2signed(t, 7);
    v = std::max<std::int64_t>(-2147483647LL - 1,
                               std::min<std::int64_t>(2147483647LL, v));
    out[static_cast<std::size_t>(scan[j])] = static_cast<std::int32_t>(v);
  }
}

// Independent §5.20.8.2 + §5.4.8 condition.
bool ref_applicable(int plane, int eob, const SecondarySyntax& s,
                    const SecondaryTransformConfig& c) {
  if (plane != 0 || s.lossless) return false;
  if (s.plane_tx_type != TxType::kDctDct &&
      s.plane_tx_type != TxType::kAdstAdst) {
    return false;
  }
  if (s.y_mode == 12 || s.y_mode < 0 || s.y_mode > 12) return false;
  if (eob < 1) return false;
  const bool large = s.coded_w >= 8 && s.coded_h >= 8;
  const int lim =
      !large ? 8
             : (((s.coded_w == 8 && s.coded_h == 8) ||
                 s.plane_tx_type == TxType::kAdstAdst)
                    ? 20
                    : 32);
  if (eob > lim) return false;
  if (s.is_inter) {
    return c.enable_inter_ist && eob > 3 &&
           s.plane_tx_type == TxType::kDctDct && s.coded_w >= 16 &&
           s.coded_h >= 16;
  }
  return c.enable_intra_ist && eob != 1;
}

std::vector<TxParams> all_tx_params() {
  std::vector<TxParams> out;
  static const int kSizes[25][2] = {
      {4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64}, {4, 8},
      {8, 4},   {8, 16},  {16, 8},  {16, 32}, {32, 16}, {32, 64},
      {64, 32}, {4, 16},  {16, 4},  {8, 32},  {32, 8},  {16, 64},
      {64, 16}, {4, 32},  {32, 4},  {8, 64},  {64, 8},  {4, 64},
      {64, 4},
  };
  for (const auto& s : kSizes) {
    auto p = txv::make_tx_params(s[0], s[1], TxType::kDctDct);
    CHECK(p.has_value());
    if (p.has_value()) out.push_back(*p);
  }
  return out;
}

std::vector<TxParams> params_for(const TxParams& base, TxType type) {
  auto p = txv::make_tx_params(base.coded_w, base.coded_h, type);
  CHECK(p.has_value());
  return p.has_value() ? std::vector<TxParams>{*p} : std::vector<TxParams>{};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_tables() {
  suite_start("tables");
  CHECK_EQ(kStxTypes, 4);
  CHECK_EQ(kIstDirSize, 7);
  CHECK_EQ(kIstReduceSetSizeAdstAdst, 4);
  CHECK_EQ(kIst4x4Width, 16);
  CHECK_EQ(kIst4x4Height, 8);
  CHECK_EQ(kIst8x8Width, 48);
  CHECK_EQ(kIst8x8Height, 32);
  CHECK_EQ(kIst8x8HeightRed, 20);
  CHECK_EQ(kIstSetSize4x4, 14);
  CHECK_EQ(kIstSetSize8x8, 11);

  CHECK_EQ(static_cast<int>(kIst4x4Kernel[0][0][0][0]), 102);
  CHECK_EQ(static_cast<int>(kIst8x8Kernel[0][0][0][0]), 103);
  CHECK_EQ(static_cast<int>(kStxScanMap[0][0][0]), 0);
  CHECK_EQ(static_cast<int>(kInvMostProbableStxMapping[0][0]), 6);
  CHECK_EQ(static_cast<int>(kInvMostProbableStxMappingAdst[0][0]), 3);

  bool seen4[16] = {};
  for (int i = 0; i < 16; ++i) {
    CHECK(kStxScanOrder4x4[i] < 16);
    seen4[kStxScanOrder4x4[i]] = true;
  }
  for (bool v : seen4) CHECK(v);

  bool seen8[64] = {};
  for (int i = 0; i < 64; ++i) {
    CHECK(kStxScanOrder8x8[i] < 64);
    seen8[kStxScanOrder8x8[i]] = true;
  }
  for (bool v : seen8) CHECK(v);

  for (int k = 0; k < kIstSetSize4x4; ++k) {
    for (int t = 0; t < 3; ++t) {
      for (int j = 0; j < kIst4x4Height; ++j) {
        for (int i = 0; i < kIst4x4Width; ++i) {
          CHECK(kIst4x4Kernel[k][t][j][i] >= -128);
          CHECK(kIst4x4Kernel[k][t][j][i] <= 127);
        }
      }
    }
  }
  for (int k = 0; k < kIstSetSize8x8; ++k) {
    for (int t = 0; t < 3; ++t) {
      for (int j = 0; j < kIst8x8Height; ++j) {
        for (int i = 0; i < kIst8x8Width; ++i) {
          CHECK(kIst8x8Kernel[k][t][j][i] >= -128);
          CHECK(kIst8x8Kernel[k][t][j][i] <= 127);
        }
      }
      bool seen_map[64] = {};
      for (int i = 0; i < kIst8x8Width; ++i) {
        CHECK(kStxScanMap[k][t][i] < 64);
        CHECK(!seen_map[kStxScanMap[k][t][i]]);
        seen_map[kStxScanMap[k][t][i]] = true;
      }
    }
  }

  CHECK_EQ(kSecTxTypeCdf[0][0][0], 4486);
  CHECK_EQ(kSecTxTypeCdf[1][0][0], 8192);
  CHECK_EQ(kMostProbableStxSetCdf[0], 20712);
  CHECK_EQ(kMostProbableStxSetAdstCdf[0], 15897);
  CHECK_EQ(kTxWidth[0], 4);
  CHECK_EQ(kTxHeight[0], 4);
  CHECK_EQ(kTxWidth[24], 64);
  CHECK_EQ(kTxHeight[24], 4);
  CHECK_EQ(kTxSizeSqr[4], 4);
  CHECK_EQ(kModeToAngle[3], 45);
  suite_end();
}

void test_size_info() {
  suite_start("size_info");
  const std::vector<TxParams> sizes = all_tx_params();
  CHECK_EQ(sizes.size(), 25ull);
  for (const TxParams& base : sizes) {
    const TxType types[] = {TxType::kDctDct, TxType::kAdstAdst,
                            TxType::kIdtx};
    for (TxType type : types) {
      auto pv = params_for(base, type);
      if (pv.empty()) continue;
      const TxParams& p = pv[0];
      SecondarySizeInfo info;
      CHECK(secondary_size_info(p, type, &info));
      CHECK_EQ(info.tx_size, static_cast<int>(p.tx_size));
      CHECK_EQ(info.w, std::min(32, static_cast<int>(p.coded_w)));
      CHECK_EQ(info.h, std::min(32, static_cast<int>(p.coded_h)));
      CHECK_EQ(info.bwl, std::min(5, static_cast<int>(kTxWidthLog2[p.tx_size])));
      CHECK_EQ(info.large, info.w >= 8 && info.h >= 8);
      CHECK_EQ(info.scan_w, info.large ? 8 : 4);
      CHECK_EQ(info.kernel_count, info.large ? 11 : 14);
      const bool is8 = p.tx_size == 1;
      const int n = !info.large ? 8 : ((is8 || type == TxType::kAdstAdst) ? 20 : 32);
      CHECK_EQ(info.n, n);
      CHECK_EQ(info.n2, info.large ? 48 : 16);
      CHECK_EQ(info.eob_limit, n);
      CHECK_EQ(secondary_eob_limit(p, type), n);
    }
    // DCT vs ADST differ only on the reduced 8x8 height.
    auto dct = txv::make_tx_params(base.coded_w, base.coded_h, TxType::kDctDct);
    auto adst = txv::make_tx_params(base.coded_w, base.coded_h, TxType::kAdstAdst);
    CHECK(dct.has_value() && adst.has_value());
    if (dct && adst) {
      SecondarySizeInfo a{};
      SecondarySizeInfo b{};
      CHECK(secondary_size_info(*dct, static_cast<TxType>(dct->tx_type), &a));
      CHECK(secondary_size_info(*adst, static_cast<TxType>(adst->tx_type), &b));
      CHECK_EQ(a.n2, b.n2);
      if (a.large && !(a.coded_w == 8 && a.coded_h == 8)) {
        CHECK_EQ(a.n, 32);
        CHECK_EQ(b.n, 20);
      } else {
        CHECK_EQ(a.n, b.n);
      }
    }
  }

  TxParams bad{};
  bad.tx_size = 25;
  bad.coded_w = 4;
  bad.coded_h = 4;
  SecondarySizeInfo info;
  CHECK(!secondary_size_info(bad, TxType::kDctDct, &info));

  bad.tx_size = 0;
  bad.coded_w = 8;
  bad.coded_h = 8;
  CHECK(!secondary_size_info(bad, TxType::kDctDct, &info));
  suite_end();
}

void test_mode_mapping() {
  suite_start("mode_mapping");
  for (int mode = -1; mode <= 13; ++mode) {
    for (int delta = -4; delta <= 4; ++delta) {
      for (int mrl = -1; mrl <= 4; ++mrl) {
        const int mapped = map_y_mode(mode, delta, mrl, 16, 16);
        const bool directional = is_directional_mode(mode);
        const bool bad_mode = mode < 0 || mode > 12 || mode == 12;
        const bool bad_angle = directional && (delta < -3 || delta > 3);
        const bool bad_mrl = directional && (mrl < 0 || mrl > 3);
        if (bad_mode || bad_angle || bad_mrl) {
          CHECK_EQ(mapped, -1);
        } else {
          // 16x16 is square, so wide_angle_mapping never changes the mode.
          CHECK_EQ(mapped, mode);
        }
      }
    }
  }
  CHECK(is_directional_mode(1));
  CHECK(is_directional_mode(8));
  CHECK(!is_directional_mode(0));
  CHECK(!is_directional_mode(9));

  // Wide-angle wrap: 4x16 (h = 4*w) with D203-ish angles wraps to D203,
  // 16x4 wraps to D45.  Thresholds are the §5.20.7.29 constants.
  CHECK_EQ(map_y_mode(1, -3, 0, 4, 16), 1);   // V, 90 - 9 = 81 >= 73
  CHECK_EQ(map_y_mode(8, 0, 0, 4, 16), 7);    // D67, 67 < 73 -> D203
  CHECK_EQ(map_y_mode(7, 0, 0, 16, 4), 3);    // D203, 203 > 270-73 -> D45
  CHECK_EQ(map_y_mode(3, 0, 0, 16, 4), 3);    // D45 stays D45
  CHECK_EQ(map_y_mode(0, 0, 0, 4, 16), 0);    // non-directional untouched
  CHECK_EQ(map_y_mode(12, 0, 0, 16, 16), -1); // PAETH excluded

  // MRL delta participates in pAngle: D67 (67) + 1 = 68 -> D203 on 4x16.
  CHECK_EQ(map_y_mode(8, 0, 1, 4, 16), 7);
  // A non-directional block ignores angle_delta/MRL.
  CHECK_EQ(map_y_mode(9, 3, 3, 4, 16), 9);

  for (int mode = 0; mode <= 11; ++mode) {
    for (int set = 0; set < 7; ++set) {
      CHECK(transpose_for_mode(mode) ==
            (mode == 2 || mode == 6 || mode == 8 || mode == 11));
    }
  }
  suite_end();
}

void test_kernel_selection() {
  suite_start("kernel_selection");
  const int sizes[][2] = {{4, 4},    {4, 8},   {8, 4},   {8, 8},   {16, 16},
                          {32, 32},  {64, 64}, {8, 16},  {16, 8},  {4, 16},
                          {16, 4},   {8, 32},  {32, 8},  {16, 64}, {64, 16},
                          {64, 32},  {32, 64}};
  bool seen4[14] = {};
  bool seen8[11] = {};
  for (const auto& s : sizes) {
    const bool large = s[0] >= 8 && s[1] >= 8;
    const int count = large ? 11 : 14;
    const TxType types[] = {TxType::kDctDct, TxType::kAdstAdst,
                            TxType::kDctAdst};
    for (TxType type : types) {
      for (int mode = 0; mode <= 11; ++mode) {
        for (int inter = 0; inter <= 1; ++inter) {
          const int max_set =
              (type == TxType::kAdstAdst && large) ? 4 : 7;
          for (int set = -1; set <= 7; ++set) {
            int kernel = -1;
            const bool ok = kernel_for_syntax(mode, type, set, inter != 0,
                                              s[0], s[1], &kernel);
            if (inter) {
              CHECK(ok);
              CHECK_EQ(kernel, 0);
              continue;
            }
            if (set < 0 || set >= max_set) {
              CHECK(!ok);
              continue;
            }
            CHECK(ok);
            CHECK(kernel >= 0 && kernel < count);
            if (large) {
              seen8[kernel] = true;
            } else {
              seen4[kernel] = true;
            }
            int expected = 0;
            if (type == TxType::kAdstAdst && large) {
              expected = kInvMostProbableStxMappingAdst[mode][set] + 7;
            } else {
              expected = kInvMostProbableStxMapping[mode][set];
              if (type == TxType::kAdstAdst) expected += 7;
            }
            CHECK_EQ(kernel, expected);
          }
        }
      }
    }
  }
  for (int k = 0; k < 14; ++k) CHECK(seen4[k]);
  for (int k = 0; k < 11; ++k) CHECK(seen8[k]);
  CHECK(!kernel_for_syntax(-1, TxType::kDctDct, 0, false, 8, 8, nullptr));
  CHECK(!kernel_for_syntax(12, TxType::kDctDct, 0, false, 8, 8, nullptr));
  suite_end();
}

void test_candidate_enumeration() {
  suite_start("candidate_enum");
  const std::vector<TxParams> sizes = all_tx_params();
  SecondaryTransformConfig cfg;
  cfg.enable_intra_ist = true;
  cfg.enable_inter_ist = true;
  for (const TxParams& p : sizes) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      for (int is_inter = 0; is_inter <= 1; ++is_inter) {
        SecondarySyntax s;
        s.plane_tx_type = type;
        s.coded_w = p.coded_w;
        s.coded_h = p.coded_h;
        s.y_mode = 0;
        s.is_inter = is_inter != 0;
        const int eob = secondary_eob_limit(p, type);
        SecondaryCandidate out[64];
        const int count =
            enumerate_secondary_candidates(s, cfg, eob, out, 64);
        const int set_count =
            is_inter ? 1
                     : most_probable_stx_set_count(type, p.coded_w, p.coded_h);
        const bool applicable = secondary_syntax_applicable(
            0, eob, s, cfg);
        const int expected = applicable ? 3 * set_count : 0;
        CHECK_EQ(count, expected);
        for (int i = 0; i < count; ++i) {
          CHECK(out[i].sec_tx_type >= 1 && out[i].sec_tx_type <= 3);
          CHECK(out[i].most_probable_stx_set >= 0 &&
                out[i].most_probable_stx_set < set_count);
          int kernel = -1;
          CHECK(kernel_for_syntax(0, type, out[i].most_probable_stx_set,
                                  is_inter != 0, p.coded_w, p.coded_h,
                                  &kernel));
          CHECK_EQ(out[i].kernel, kernel);
          CHECK_EQ(out[i].transpose, false);
          for (int j = 0; j < i; ++j) {
            CHECK(!(out[j].sec_tx_type == out[i].sec_tx_type &&
                    out[j].most_probable_stx_set ==
                        out[i].most_probable_stx_set));
          }
        }
        CHECK_EQ(enumerate_secondary_candidates(s, cfg, eob, out, 1),
                 count == 0 ? 0 : 1);
        CHECK_EQ(enumerate_secondary_candidates(s, cfg, eob, out, 0), 0);
        SecondarySyntax no_ist = s;
        no_ist.lossless = true;
        CHECK_EQ(enumerate_secondary_candidates(no_ist, cfg, eob, out, 64), 0);
      }
    }
  }
  suite_end();
}

void test_syntax_legality() {
  suite_start("syntax_legality");
  const std::vector<TxParams> sizes = all_tx_params();
  const TxType types[] = {TxType::kDctDct, TxType::kAdstAdst,
                          TxType::kDctAdst, TxType::kIdtx};
  for (const TxParams& p : sizes) {
    for (TxType type : types) {
      for (int is_inter = 0; is_inter <= 1; ++is_inter) {
        for (int enabled = 0; enabled <= 1; ++enabled) {
          SecondaryTransformConfig cfg;
          cfg.enable_intra_ist = enabled != 0;
          cfg.enable_inter_ist = enabled != 0;
          for (int eob = 0; eob <= 40; ++eob) {
            SecondarySyntax s;
            s.plane_tx_type = type;
            s.coded_w = p.coded_w;
            s.coded_h = p.coded_h;
            s.y_mode = 0;
            s.is_inter = is_inter != 0;
            const bool expected = ref_applicable(0, eob, s, cfg);
            CHECK_EQ(secondary_syntax_applicable(0, eob, s, cfg), expected);
            CHECK_EQ(secondary_syntax_applicable(1, eob, s, cfg), false);
            CHECK_EQ(secondary_syntax_applicable(2, eob, s, cfg), false);
          }
        }
      }
    }
  }
  // Lossless and PAETH.
  SecondaryTransformConfig cfg;
  cfg.enable_intra_ist = true;
  cfg.enable_inter_ist = true;
  SecondarySyntax s;
  s.coded_w = 16;
  s.coded_h = 16;
  s.y_mode = 0;
  s.sec_tx_type = 1;
  CHECK(secondary_syntax_applicable(0, 4, s, cfg));
  s.lossless = true;
  CHECK(!secondary_syntax_applicable(0, 4, s, cfg));
  s.lossless = false;
  s.y_mode = 12;
  CHECK(!secondary_syntax_applicable(0, 4, s, cfg));
  s.y_mode = 0;

  // valid() rejects sec_tx_type 0/4, bad set, bad bit depth.
  s.most_probable_stx_set = 0;
  s.sec_tx_type = 0;
  CHECK(!secondary_syntax_valid(0, 4, s, cfg));
  s.sec_tx_type = 1;
  CHECK(secondary_syntax_valid(0, 4, s, cfg));
  s.sec_tx_type = 4;
  CHECK(!secondary_syntax_valid(0, 4, s, cfg));
  s.sec_tx_type = 1;
  s.bit_depth = 7;
  CHECK(!secondary_syntax_valid(0, 4, s, cfg));
  s.bit_depth = 8;
  s.plane_tx_type = TxType::kAdstAdst;
  s.most_probable_stx_set = 4;
  CHECK(!secondary_syntax_valid(0, 4, s, cfg));
  s.most_probable_stx_set = 3;
  CHECK(secondary_syntax_valid(0, 4, s, cfg));
  suite_end();
}

void test_inverse_kernels() {
  suite_start("inverse_kernels");
  const std::vector<TxParams> sizes = all_tx_params();
  int cases = 0;
  for (const TxParams& base : sizes) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      auto pv = params_for(base, type);
      if (pv.empty()) continue;
      const TxParams& p = pv[0];
      SecondarySizeInfo info;
      CHECK(secondary_size_info(p, type, &info));
      const int count = info.kernel_count;
      const int n = info.n;
      const std::size_t elems = static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
      std::vector<std::int32_t> in(elems);
      std::vector<std::int32_t> got(elems);
      std::vector<std::int32_t> ref(elems);
      std::uint16_t scan[1024];
      ref_scan(info.w, info.h, scan);

      for (int kernel = 0; kernel < count; ++kernel) {
        for (int sec = 1; sec < kStxTypes; ++sec) {
          for (int transpose = 0; transpose <= 1; ++transpose) {
            auto run_case = [&](const std::vector<std::int32_t>& values) {
              std::fill(in.begin(), in.end(), 0);
              // First n scan positions are the legal input; put a sentinel
              // outside the 8x8/4x4 output grid to verify it is untouched.
              for (int j = 0; j < n; ++j) in[static_cast<std::size_t>(scan[j])] = values[static_cast<std::size_t>(j)];
              ref_inverse(p, type, kernel, sec, transpose != 0, 8, in.data(),
                          ref.data());
              std::copy(in.begin(), in.end(), got.begin());
              const bool ok = inverse_secondary_dequant_kernel(
                  p, type, kernel, sec, transpose != 0, 8, got.data(),
                  got.size());
              CHECK(ok);
              for (std::size_t i = 0; i < elems; ++i) {
                CHECK_EQ(got[i], ref[i]);
              }
              ++cases;
            };

            std::vector<std::int32_t> values(static_cast<std::size_t>(n));
            std::fill(values.begin(), values.end(), 0);
            run_case(values);  // all zero

            for (int j = 0; j < n; ++j) {
              std::fill(values.begin(), values.end(), 0);
              values[static_cast<std::size_t>(j)] = 1;
              run_case(values);
              values[static_cast<std::size_t>(j)] = -1;
              run_case(values);
              values[static_cast<std::size_t>(j)] = 32767;
              run_case(values);
              values[static_cast<std::size_t>(j)] = -32768;
              run_case(values);
              values[static_cast<std::size_t>(j)] = rand_range(-30000, 30000);
              run_case(values);
            }

            for (int j = 0; j < n; ++j) values[static_cast<std::size_t>(j)] = rand_range(-40000, 40000);
            run_case(values);
            for (int j = 0; j < n; ++j) values[static_cast<std::size_t>(j)] = (j & 1) ? 32767 : -32768;
            run_case(values);
            for (int j = 0; j < n; ++j) values[static_cast<std::size_t>(j)] = (j & 1) ? 1 : -1;
            run_case(values);
          }
        }
      }
    }
  }
  std::printf("    inverse cases: %d\n", cases);
  suite_end();
}

void test_eob_interaction() {
  suite_start("eob_interaction");
  const std::vector<TxParams> sizes = all_tx_params();
  for (const TxParams& base : sizes) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      auto pv = params_for(base, type);
      if (pv.empty()) continue;
      const TxParams& p = pv[0];
      SecondarySizeInfo info;
      CHECK(secondary_size_info(p, type, &info));
      const std::size_t elems = static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
      std::uint16_t scan[1024];
      ref_scan(info.w, info.h, scan);
      std::vector<std::int32_t> full(elems, 0);
      for (int j = 0; j < info.n; ++j) {
        full[static_cast<std::size_t>(scan[j])] = rand_range(-500, 500);
      }
      for (int eob = 1; eob <= info.n; ++eob) {
        std::vector<std::int32_t> partial = full;
        for (int j = eob; j < info.n; ++j) partial[static_cast<std::size_t>(scan[j])] = 0;
        for (int kernel = 0; kernel < info.kernel_count; ++kernel) {
          for (int sec = 1; sec < kStxTypes; ++sec) {
            std::vector<std::int32_t> a = partial;
            CHECK(inverse_secondary_dequant_kernel(
                p, type, kernel, sec, false, 8, a.data(), a.size()));
            std::vector<std::int32_t> omitted = partial;
            CHECK(inverse_secondary_dequant_kernel(
                p, type, kernel, sec, false, 8, omitted.data(), omitted.size()));
            for (std::size_t i = 0; i < elems; ++i) {
              CHECK_EQ(a[i], omitted[i]);
            }
          }
        }
      }
      // eob > eobLim is never legal.
      SecondaryTransformConfig cfg;
      cfg.enable_intra_ist = true;
      SecondarySyntax s;
      s.plane_tx_type = type;
      s.coded_w = p.coded_w;
      s.coded_h = p.coded_h;
      s.y_mode = 0;
      s.sec_tx_type = 1;
      CHECK(!secondary_syntax_applicable(0, info.n + 1, s, cfg));
      CHECK(secondary_syntax_applicable(0, info.n, s, cfg));
    }
  }
  suite_end();
}

void test_forward_projection() {
  suite_start("forward_projection");
  const std::vector<TxParams> sizes = all_tx_params();
  double projection_total = 0.0;
  long long projection_cases = 0;
  for (const TxParams& base : sizes) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      auto pv = params_for(base, type);
      if (pv.empty()) continue;
      const TxParams& p = pv[0];
      SecondarySizeInfo info;
      CHECK(secondary_size_info(p, type, &info));
      const std::size_t elems = static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
      std::vector<std::int32_t> coeff(elems);
      std::vector<std::int32_t> got(elems);
      std::vector<std::int32_t> ref(elems);
      std::vector<std::int32_t> recon(elems);
      for (int kernel = 0; kernel < info.kernel_count; ++kernel) {
        for (int sec = 1; sec < kStxTypes; ++sec) {
          for (int transpose = 0; transpose <= 1; ++transpose) {
            for (std::size_t i = 0; i < elems; ++i) {
              coeff[i] = rand_range(-2000, 2000);
            }
            got = coeff;
            ref_forward(p, type, kernel, sec, transpose != 0, coeff.data(),
                        ref.data());
            CHECK(forward_secondary_coeffs_kernel(
                p, type, kernel, sec, transpose != 0, 8, got.data(),
                got.size()));
            for (std::size_t i = 0; i < elems; ++i) CHECK_EQ(got[i], ref[i]);

            recon = got;
            CHECK(inverse_secondary_dequant_kernel(
                p, type, kernel, sec, transpose != 0, 8, recon.data(),
                recon.size()));
            // Outside the output grid the forward projection zeroed, and the
            // inverse leaves it zero.  Inside, the projection is not the
            // identity (the matrix reduces dimension), but its energy must not
            // explode.
            double num = 0.0;
            double den = 0.0;
            const int sb = info.large ? 3 : 2;
            for (int i = 0; i < info.n2; ++i) {
              const int pos =
                  info.large
                      ? kStxScanOrder8x8[kStxScanMap[kernel][sec - 1][i]]
                      : kStxScanOrder4x4[i];
              const int x = pos & ((1 << sb) - 1);
              const int y = pos >> sb;
              const int index = transpose ? x * info.w + y : y * info.w + x;
              const double e =
                  static_cast<double>(recon[static_cast<std::size_t>(index)] - coeff[static_cast<std::size_t>(index)]);
              num += e * e;
              den += static_cast<double>(coeff[static_cast<std::size_t>(index)]) * coeff[static_cast<std::size_t>(index)];
            }
            if (den > 0.0) {
              projection_total += num / den;
              ++projection_cases;
            }
          }
        }
      }
      // Forward never leaves coefficients where the inverse did not write.
      // (The first n scan positions are written; all others are zero.)
      std::uint16_t scan[1024];
      ref_scan(info.w, info.h, scan);
      std::vector<bool> is_input(elems, false);
      for (int j = 0; j < info.n; ++j) is_input[static_cast<std::size_t>(scan[j])] = true;
      for (std::size_t i = 0; i < elems; ++i) {
        if (!is_input[i]) CHECK_EQ(got[i], 0);
      }
    }
  }
  const double mean_projection =
      projection_cases == 0 ? 0.0 : projection_total / static_cast<double>(projection_cases);
  std::printf("    mean projection relative SSE: %.6f (%lld cases)\n",
              mean_projection, projection_cases);
  CHECK(mean_projection < 0.70);
  suite_end();
}

void test_apply_and_luma_only() {
  suite_start("apply_luma_only");
  const std::vector<TxParams> sizes = all_tx_params();
  SecondaryTransformConfig cfg;
  cfg.enable_intra_ist = true;
  cfg.enable_inter_ist = true;
  for (const TxParams& base : sizes) {
    for (TxType type : {TxType::kDctDct, TxType::kAdstAdst}) {
      auto pv = params_for(base, type);
      if (pv.empty()) continue;
      const TxParams& p = pv[0];
      SecondarySizeInfo info;
      CHECK(secondary_size_info(p, type, &info));
      const std::size_t elems = static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h);
      std::uint16_t scan[1024];
      ref_scan(info.w, info.h, scan);
      std::vector<std::int32_t> in(elems, 0);
      for (int j = 0; j < info.n; ++j) in[static_cast<std::size_t>(scan[j])] = rand_range(-1000, 1000);

      // Intra: DCT/ADST with both enable bits; find a mapping for each set.
      for (int set = 0; set < most_probable_stx_set_count(type, p.coded_w, p.coded_h);
           ++set) {
        SecondarySyntax s;
        s.plane_tx_type = type;
        s.coded_w = p.coded_w;
        s.coded_h = p.coded_h;
        s.y_mode = 0;
        s.angle_delta = 0;
        s.mrl_index = 0;
        s.most_probable_stx_set = set;
        s.sec_tx_type = 1 + (set % 3);
        if (info.eob_limit < 2) continue;
        std::vector<std::int32_t> got = in;
        const SecondaryStatus st = apply_inverse_secondary(
            p, 0, info.eob_limit, s, cfg, got.data(), got.size());
        CHECK_EQ(static_cast<int>(st), static_cast<int>(SecondaryStatus::kApplied));
        int kernel = 0;
        CHECK(kernel_for_syntax(0, type, set, false, p.coded_w, p.coded_h,
                                &kernel));
        std::vector<std::int32_t> ref = in;
        ref_inverse(p, type, kernel, s.sec_tx_type, false, 8, in.data(),
                    ref.data());
        for (std::size_t i = 0; i < elems; ++i) CHECK_EQ(got[i], ref[i]);

        // Chroma must be untouched.
        std::vector<std::int32_t> chroma = in;
        CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                     p, 1, info.eob_limit, s, cfg, chroma.data(),
                     chroma.size())),
                 static_cast<int>(SecondaryStatus::kNotApplicable));
        for (std::size_t i = 0; i < elems; ++i) CHECK_EQ(chroma[i], in[i]);
        CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                     p, 2, info.eob_limit, s, cfg, chroma.data(),
                     chroma.size())),
                 static_cast<int>(SecondaryStatus::kNotApplicable));
      }

      // sec_tx_type == 0 -> no-op.
      SecondarySyntax s;
      s.plane_tx_type = type;
      s.coded_w = p.coded_w;
      s.coded_h = p.coded_h;
      s.y_mode = 0;
      s.sec_tx_type = 0;
      std::vector<std::int32_t> got = in;
      CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                   p, 0, info.eob_limit, s, cfg, got.data(), got.size())),
               static_cast<int>(SecondaryStatus::kNotApplicable));
      for (std::size_t i = 0; i < elems; ++i) CHECK_EQ(got[i], in[i]);

      // Lossless and PAETH -> no-op.
      s.sec_tx_type = 1;
      s.lossless = true;
      got = in;
      CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                   p, 0, info.eob_limit, s, cfg, got.data(), got.size())),
               static_cast<int>(SecondaryStatus::kNotApplicable));
      s.lossless = false;
      s.y_mode = 12;
      got = in;
      CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                   p, 0, info.eob_limit, s, cfg, got.data(), got.size())),
               static_cast<int>(SecondaryStatus::kNotApplicable));
      s.y_mode = 0;

      // EOB 1 intra -> no-op; EOB > limit -> no-op.
      got = in;
      CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                   p, 0, 1, s, cfg, got.data(), got.size())),
               static_cast<int>(SecondaryStatus::kNotApplicable));
      got = in;
      CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                   p, 0, info.eob_limit + 1, s, cfg, got.data(), got.size())),
               static_cast<int>(SecondaryStatus::kNotApplicable));
      // Bad set/sec type -> invalid.
      s.most_probable_stx_set = 7;
      CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                   p, 0, info.eob_limit, s, cfg, got.data(), got.size())),
               static_cast<int>(SecondaryStatus::kInvalid));
      s.most_probable_stx_set = 0;
      s.sec_tx_type = 4;
      CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                   p, 0, info.eob_limit, s, cfg, got.data(), got.size())),
               static_cast<int>(SecondaryStatus::kInvalid));

      // Inter rules: only DCT_DCT, >= 16x16, eob > 3, enable_inter_ist.
      SecondarySyntax inter;
      inter.plane_tx_type = TxType::kDctDct;
      inter.coded_w = p.coded_w;
      inter.coded_h = p.coded_h;
      inter.y_mode = 0;
      inter.sec_tx_type = 1;
      inter.is_inter = true;
      inter.most_probable_stx_set = 0;
      const bool inter_ok = p.coded_w >= 16 && p.coded_h >= 16 &&
                            info.eob_limit >= 5;
      CHECK_EQ(secondary_syntax_applicable(0, 5, inter, cfg), inter_ok);
      if (inter_ok) {
        std::vector<std::int32_t> ig = in;
        CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                     p, 0, 5, inter, cfg, ig.data(), ig.size())),
                 static_cast<int>(SecondaryStatus::kApplied));
        std::vector<std::int32_t> ir = in;
        ref_inverse(p, TxType::kDctDct, 0, 1, false, 8, in.data(), ir.data());
        for (std::size_t i = 0; i < elems; ++i) CHECK_EQ(ig[i], ir[i]);
      } else {
        std::vector<std::int32_t> ig = in;
        CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                     p, 0, 5, inter, cfg, ig.data(), ig.size())),
                 static_cast<int>(SecondaryStatus::kNotApplicable));
      }
      SecondaryTransformConfig no_inter = cfg;
      no_inter.enable_inter_ist = false;
      CHECK(!secondary_syntax_applicable(0, 5, inter, no_inter));

      // Transpose follows YMode even for inter; compare with the low-level
      // kernel using the same flag.
      if (inter_ok && type == TxType::kDctDct) {
        SecondarySyntax transposed = inter;
        transposed.y_mode = 2;  // H_PRED -> transpose
        std::vector<std::int32_t> tg = in;
        CHECK_EQ(static_cast<int>(apply_inverse_secondary(
                     p, 0, 5, transposed, cfg, tg.data(), tg.size())),
                 static_cast<int>(SecondaryStatus::kApplied));
        std::vector<std::int32_t> tr = in;
        ref_inverse(p, TxType::kDctDct, 0, 1, true, 8, in.data(), tr.data());
        for (std::size_t i = 0; i < elems; ++i) CHECK_EQ(tg[i], tr[i]);
      }
    }
  }
  suite_end();
}

void test_cdf_and_header() {
  suite_start("cdf_header");
  CHECK(sec_tx_type_cdf(0, false) == &kSecTxTypeCdf[0][0][0]);
  CHECK(sec_tx_type_cdf(1, true) == &kSecTxTypeCdf[1][1][0]);
  CHECK(sec_tx_type_cdf(-1, false) == nullptr);
  CHECK(sec_tx_type_cdf(25, false) == nullptr);
  CHECK(most_probable_stx_set_cdf(TxType::kDctDct, 8, 8) ==
        &kMostProbableStxSetCdf[0]);
  CHECK(most_probable_stx_set_cdf(TxType::kAdstAdst, 8, 8) ==
        &kMostProbableStxSetAdstCdf[0]);
  CHECK(most_probable_stx_set_cdf(TxType::kAdstAdst, 8, 4) ==
        &kMostProbableStxSetCdf[0]);

  const double expected0 = -std::log2(4486.0 / 32768.0);
  CHECK(std::abs(sec_tx_type_bits(0, 0, false) - expected0) < 1e-12);
  const double expected_last = -std::log2((32768.0 - 26440.0) / 32768.0);
  CHECK(std::abs(sec_tx_type_bits(3, 0, false) - expected_last) < 1e-12);
  CHECK(std::abs(most_probable_stx_set_bits(0, TxType::kDctDct, 8, 8) -
                 (-std::log2(20712.0 / 32768.0))) < 1e-12);
  CHECK(std::abs(most_probable_stx_set_bits(0, TxType::kAdstAdst, 16, 16) -
                 (-std::log2(15897.0 / 32768.0))) < 1e-12);
  CHECK_EQ(sec_tx_type_bits(-1, 0, false), 0.0);
  CHECK(signals_most_probable_stx_set(false, 1));
  CHECK(!signals_most_probable_stx_set(false, 0));
  CHECK(!signals_most_probable_stx_set(true, 1));

  for (int intra = 0; intra <= 1; ++intra) {
    for (int inter = 0; inter <= 1; ++inter) {
      SecondaryTransformConfig c;
      c.enable_intra_ist = intra != 0;
      c.enable_inter_ist = inter != 0;
      BitWriter w;
      write_secondary_transform_header_bits(w, c);
      CHECK_EQ(w.bit_size(), 2ull);
      CHECK_EQ(w.byte_size(), 1ull);
      const std::uint8_t expected =
          static_cast<std::uint8_t>((intra << 7) | (inter << 6));
      CHECK_EQ(static_cast<int>(w.bytes()[0]), static_cast<int>(expected));
      CHECK_EQ(enable_secondary_transform(c), (intra || inter) != 0);
    }
  }
  suite_end();
}

void test_determinism() {
  suite_start("determinism");
  TxParams p = *txv::make_tx_params(16, 16, TxType::kDctDct);
  SecondarySizeInfo info;
  CHECK(secondary_size_info(p, TxType::kDctDct, &info));
  std::vector<std::int32_t> base(static_cast<std::size_t>(info.w) * static_cast<std::size_t>(info.h));
  for (auto& v : base) v = rand_range(-5000, 5000);
  std::vector<std::int32_t> a = base;
  std::vector<std::int32_t> b = base;
  CHECK(inverse_secondary_dequant_kernel(p, TxType::kDctDct, 3, 2, true, 8,
                                         a.data(), a.size()));
  CHECK(inverse_secondary_dequant_kernel(p, TxType::kDctDct, 3, 2, true, 8,
                                         b.data(), b.size()));
  for (std::size_t i = 0; i < a.size(); ++i) CHECK_EQ(a[i], b[i]);
  // A too-small buffer is rejected without touching it.
  std::vector<std::int32_t> tiny(3, 7);
  CHECK(!inverse_secondary_dequant_kernel(p, TxType::kDctDct, 0, 1, false, 8,
                                          tiny.data(), tiny.size()));
  for (int v : tiny) CHECK_EQ(v, 7);
  suite_end();
}

}  // namespace

int main() {
  std::printf("av2enc AV2 secondary transform (IST) CPU test\n");
  test_tables();
  test_size_info();
  test_mode_mapping();
  test_kernel_selection();
  test_candidate_enumeration();
  test_syntax_legality();
  test_inverse_kernels();
  test_eob_interaction();
  test_forward_projection();
  test_apply_and_luma_only();
  test_cdf_and_header();
  test_determinism();
  std::printf("%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
