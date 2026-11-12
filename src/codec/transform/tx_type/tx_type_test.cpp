// CPU self-test for the AV2 intra transform-type selector.
//
// Covers:
//   * the §9.2/§9.3 generated tables and the §5.20.8.3 set classification
//   * candidate enumeration for every size x mode x reduced_tx_set
//   * the §5.20.8.2 symbol mapping (inverse of the candidate table)
//   * exhaustive per-type transform/quantize/inverse against tx_variants
//   * length-32 ADST/FDST -> DCT aliasing on every affected size
//   * the pluggable rate model and the deterministic tie-break
//   * selection parity with an independent brute force over all 16 TX types
//   * real 64x64 and 640x360 luma blocks from tests/data
//
// Build/run:
//   ./build_and_test.sh
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "codec/intra/modes/intra_modes.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/tx_type/tx_type_tables.h"
#include "codec/transform/variants/tx_tables.h"
#include "codec/transform/variants/tx_variants.h"
#include "core/io/y4m.h"

namespace {

using namespace av2enc;
using namespace av2enc::txtype;

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::mt19937 g_rng(0x5eed1234u);

int rand_range(int lo, int hi) {
  std::uniform_int_distribution<int> dist(lo, hi);
  return dist(g_rng);
}

// Synthetic coded-size residual: 0 noise, 1 gradient + noise, 2 split block,
// 3 sparse spikes.
void fill_residual(int w, int h, int kind, std::int16_t* out) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 0;
      switch (kind) {
        case 0:
          v = rand_range(-60, 60);
          break;
        case 1:
          v = (x * 3 - y * 2) + rand_range(-20, 20);
          break;
        case 2:
          v = (x < w / 2 ? -40 : 40) + rand_range(-10, 10);
          break;
        default:
          v = (rand_range(0, 15) == 0) ? rand_range(-120, 120) : 0;
          break;
      }
      out[static_cast<std::size_t>(y * w + x)] =
          static_cast<std::int16_t>(v);
    }
  }
}

TxTypeCandidate make_candidate(const TxTypeQuery& q, TxType type) {
  TxTypeCandidate c;
  c.type = type;
  const auto p = txv::make_tx_params(q.coded_w, q.coded_h, type);
  if (p.has_value()) {
    c.params = *p;
  }
  int symbol = -1;
  bool flag = false;
  if (tx_type_symbol(q, type, &symbol, &flag)) {
    c.symbol = static_cast<std::int8_t>(symbol);
    c.is_long_side_dct = flag;
  }
  return c;
}

// Independent default-rate/cost recomputation for the parity tests.
double expected_rate(const TxTypeEval& ev, const TxTypeCostConfig& cfg) {
  if (cfg.rate_fn != nullptr) {
    TxTypeRateInputs in;
    in.candidate = ev.candidate;
    in.eob = ev.eob;
    in.nonzero = ev.nonzero;
    in.sum_abs = ev.sum_abs;
    in.sse = ev.sse;
    in.type_bits = ev.type_bits;
    return cfg.rate_fn(in, cfg.rate_user);
  }
  double bits = cfg.coefficient_bits * static_cast<double>(ev.nonzero);
  if (cfg.use_static_type_cost) {
    bits += ev.type_bits;
  }
  return bits;
}

bool residual_equal(const std::int16_t* a, const std::int16_t* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool levels_equal(const std::int32_t* a, const std::int32_t* b, int n) {
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

double sse_of(const std::int16_t* a, const std::int16_t* b, int n) {
  double sse = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    sse += d * d;
  }
  return sse;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------
void test_tables() {
  suite_start("tables");
  CHECK_EQ(kSizeClass[0], 0);   // 4x4
  CHECK_EQ(kSizeClass[1], 1);   // 8x8
  CHECK_EQ(kSizeClass[2], 2);   // 16x16
  CHECK_EQ(kSizeClass[3], 3);   // 32x32
  CHECK_EQ(kSizeClass[4], 3);   // 64x64
  CHECK_EQ(kSizeClass[5], 0);   // 4x8
  CHECK_EQ(kSizeClass[13], 1);  // 4x16
  CHECK_EQ(kSizeClass[19], 3);  // 4x32

  CHECK_EQ(kTxSizeSqr[4], 4);     // 64x64 (Tx_Size_Sqr keeps 64x64)
  CHECK_EQ(kTxSizeSqr[11], 3);    // 32x64 -> 32x32
  CHECK_EQ(kTxSizeSqr[17], 2);    // 16x64 -> 16x16
  CHECK_EQ(kTxSizeSqr[18], 2);    // 64x16 -> 16x16
  CHECK_EQ(kTxSizeSqrUp[4], 4);   // 64x64
  CHECK_EQ(kTxSizeSqrUp[9], 3);   // 16x32
  CHECK_EQ(kTxSizeSqrUp[17], 4);  // 16x64

  // Tx_Type_Inv_Long, §5.20.8.2.
  const std::uint8_t expected_long[2][2][4] = {
      {{10, 12, 14, 9}, {11, 13, 15, 9}},
      {{0, 1, 4, 11}, {0, 2, 5, 10}},
  };
  for (int flag = 0; flag < 2; ++flag) {
    for (int orient = 0; orient < 2; ++orient) {
      for (int i = 0; i < 4; ++i) {
        CHECK_EQ(static_cast<int>(kTxTypeInvLong[flag][orient][i]),
                 static_cast<int>(expected_long[flag][orient][i]));
      }
    }
  }

  // §5.20.7.29 membership table row sizes.
  const int expected_counts[7] = {1, 4, 4, 8, 8, 16, 16};
  for (int set = 0; set < 7; ++set) {
    int count = 0;
    for (int t = 0; t < 16; ++t) {
      count += kTxTypeInSetIntra[set][t];
    }
    CHECK_EQ(count, expected_counts[set]);
  }

  // §9.3 CDF spot values.
  CHECK_EQ(static_cast<int>(kIntraTxTypeSet1Cdf[0][0]), 5026);
  CHECK_EQ(static_cast<int>(kIntraTxTypeSet1Cdf[2][6]), 35);
  CHECK_EQ(static_cast<int>(kIntraTxTypeSet2Cdf[0][0]), 16384);
  CHECK_EQ(static_cast<int>(kIntraTxTypeLongCdf[3][0]), 8192);
  CHECK_EQ(static_cast<int>(kIsLongSideDctCdf[0][0]), 36);

  // Md_Idx_To_Type: every mode starts with DCT_DCT, ADST_ADST.
  for (int sc = 0; sc < 4; ++sc) {
    for (int mode = 0; mode < kNumIntraModes; ++mode) {
      CHECK_EQ(static_cast<int>(kMdIdxToType[sc][mode][0]), 0);
      CHECK_EQ(static_cast<int>(kMdIdxToType[sc][mode][1]), 3);
      // Every entry is a valid TX_TYPE.
      for (int i = 0; i < 7; ++i) {
        CHECK(static_cast<int>(kMdIdxToType[sc][mode][i]) < 16);
      }
    }
  }

  // §6.19.8.2 set values.
  CHECK_EQ(static_cast<int>(TxSet::kDctOnly), 0);
  CHECK_EQ(static_cast<int>(TxSet::kWide64), 1);
  CHECK_EQ(static_cast<int>(TxSet::kHigh64), 2);
  CHECK_EQ(static_cast<int>(TxSet::kWide32), 3);
  CHECK_EQ(static_cast<int>(TxSet::kHigh32), 4);
  CHECK_EQ(static_cast<int>(TxSet::kIntra1), 5);
  CHECK_EQ(static_cast<int>(TxSet::kIntra2), 6);
  CHECK_EQ(std::string(tx_set_name(TxSet::kIntra1)), "TX_SET_INTRA_1");
  suite_end();
}

// ---------------------------------------------------------------------------
// get_tx_set()
// ---------------------------------------------------------------------------
void test_intra_tx_set() {
  suite_start("get_tx_set");

  // reduced_tx_set = 0, is_inter = 0.  Index is the §6.19.6.1 TX_SIZE value.
  const TxSet expected[25] = {
      TxSet::kIntra1,   // 0  4x4
      TxSet::kIntra1,   // 1  8x8
      TxSet::kIntra1,   // 2  16x16
      TxSet::kDctOnly,  // 3  32x32
      TxSet::kDctOnly,  // 4  64x64
      TxSet::kIntra1,   // 5  4x8
      TxSet::kIntra1,   // 6  8x4
      TxSet::kIntra1,   // 7  8x16
      TxSet::kIntra1,   // 8  16x8
      TxSet::kHigh32,   // 9  16x32
      TxSet::kWide32,   // 10 32x16
      TxSet::kDctOnly,  // 11 32x64
      TxSet::kDctOnly,  // 12 64x32
      TxSet::kIntra1,   // 13 4x16
      TxSet::kIntra1,   // 14 16x4
      TxSet::kHigh32,   // 15 8x32
      TxSet::kWide32,   // 16 32x8
      TxSet::kHigh64,   // 17 16x64
      TxSet::kWide64,   // 18 64x16
      TxSet::kHigh32,   // 19 4x32
      TxSet::kWide32,   // 20 32x4
      TxSet::kHigh64,   // 21 8x64
      TxSet::kWide64,   // 22 64x8
      TxSet::kHigh64,   // 23 4x64
      TxSet::kWide64,   // 24 64x4
  };
  for (int i = 0; i < 25; ++i) {
    TxTypeQuery q;
    q.coded_w = txv::kTxSizes[i].width;
    q.coded_h = txv::kTxSizes[i].height;
    q.mapped_mode = 0;
    TxSet set;
    CHECK(intra_tx_set(q, &set));
    CHECK_EQ(static_cast<int>(set), static_cast<int>(expected[i]));
  }

  // reduced_tx_set = 1 and 3: INTRA_2 where sqr_up < 32, unchanged elsewhere.
  const TxSet expect_reduced[25] = {
      TxSet::kIntra2,   TxSet::kIntra2,   TxSet::kIntra2,   TxSet::kDctOnly,
      TxSet::kDctOnly,  TxSet::kIntra2,   TxSet::kIntra2,   TxSet::kIntra2,
      TxSet::kIntra2,   TxSet::kHigh32,   TxSet::kWide32,   TxSet::kDctOnly,
      TxSet::kDctOnly,  TxSet::kIntra2,   TxSet::kIntra2,   TxSet::kHigh32,
      TxSet::kWide32,   TxSet::kHigh64,   TxSet::kWide64,   TxSet::kHigh32,
      TxSet::kWide32,   TxSet::kHigh64,   TxSet::kWide64,   TxSet::kHigh64,
      TxSet::kWide64,
  };
  for (int i = 0; i < 25; ++i) {
    for (int reduced : {1, 3}) {
      TxTypeQuery q;
      q.coded_w = txv::kTxSizes[i].width;
      q.coded_h = txv::kTxSizes[i].height;
      q.reduced_tx_set = reduced;
      TxSet set;
      CHECK(intra_tx_set(q, &set));
      CHECK_EQ(static_cast<int>(set), static_cast<int>(expect_reduced[i]));
    }
  }

  // reduced_tx_set = 2: get_tx_set() returns DCT_IDTX below 32x32, but
  // transform_type() forces DCT_DCT (checked in test_candidate_sets).
  {
    TxTypeQuery q;
    q.coded_w = 16;
    q.coded_h = 16;
    q.reduced_tx_set = 2;
    TxSet set;
    CHECK(intra_tx_set(q, &set));
    CHECK_EQ(static_cast<int>(set), static_cast<int>(TxSet::kDctIdtx));
  }

  // Invalid inputs.
  {
    TxTypeQuery q;
    q.coded_w = 2;
    q.coded_h = 8;
    TxSet set;
    CHECK(!intra_tx_set(q, &set));
    q.coded_w = 4;
    q.coded_h = 4;
    q.reduced_tx_set = 4;
    CHECK(!intra_tx_set(q, &set));
    CHECK(!intra_tx_set(q, nullptr));
  }
  suite_end();
}

// ---------------------------------------------------------------------------
// Candidate enumeration
// ---------------------------------------------------------------------------
void test_candidate_sets() {
  suite_start("candidate_sets");

  const char* names[16] = {"DCT_DCT",         "ADST_DCT",   "DCT_ADST",
                           "ADST_ADST",       "FLIPADST_DCT", "DCT_FLIPADST",
                           "FLIPADST_FLIPADST", "FLIPADST_ADST",
                           "ADST_FLIPADST",   "IDTX",       "V_DCT",
                           "H_DCT",           "V_ADST",     "H_ADST",
                           "V_FLIPADST",      "H_FLIPADST"};
  for (int t = 0; t < 16; ++t) {
    CHECK_EQ(std::string(txv::tx_type_name(static_cast<TxType>(t))), names[t]);
  }

  for (int i = 0; i < 25; ++i) {
    for (int mode = 0; mode < kNumIntraModes; ++mode) {
      for (int reduced = 0; reduced < 4; ++reduced) {
        TxTypeQuery q;
        q.coded_w = txv::kTxSizes[i].width;
        q.coded_h = txv::kTxSizes[i].height;
        q.mapped_mode = mode;
        q.reduced_tx_set = reduced;
        TxTypeSet set;
        CHECK(intra_tx_candidates(q, &set));
        const int set_id = static_cast<int>(set.set);

        int expected_count = 0;
        switch (set.set) {
          case TxSet::kDctOnly:
            expected_count = 1;
            break;
          case TxSet::kIntra1:
            expected_count = 7;
            break;
          case TxSet::kIntra2:
            expected_count = 2;
            break;
          case TxSet::kWide64:
          case TxSet::kHigh64:
            expected_count = 4;
            break;
          case TxSet::kWide32:
          case TxSet::kHigh32:
            expected_count = 8;
            break;
          default:
            expected_count = -1;
            break;
        }
        if (reduced == 2) {
          expected_count = 1;  // transform_type() override
        }
        CHECK_EQ(set.count, expected_count);
        CHECK(set.count >= 1 && set.count <= kMaxTxTypeCandidates);

        // Uniqueness + membership + parameter consistency.
        bool seen[16] = {};
        for (int j = 0; j < set.count; ++j) {
          const TxTypeCandidate& c = set.candidates[j];
          const int type = static_cast<int>(c.type);
          CHECK(type >= 0 && type < 16);
          CHECK(!seen[type]);
          seen[type] = true;
          if (set_id >= 0 && set_id < 7) {
            CHECK_EQ(static_cast<int>(kTxTypeInSetIntra[set_id][type]), 1);
          }
          CHECK_EQ(c.params.tx_type, static_cast<std::int8_t>(type));
          CHECK_EQ(c.params.coded_w, q.coded_w);
          CHECK_EQ(c.params.coded_h, q.coded_h);
          CHECK(txv::tx_size_supported(c.params.coded_w, c.params.coded_h));
          if (reduced == 2) {
            CHECK_EQ(type, static_cast<int>(TxType::kDctDct));
            CHECK_EQ(c.symbol, -1);
          }
        }
        // DCT_DCT is always available so selection can never fail.
        CHECK(seen[static_cast<int>(TxType::kDctDct)]);

        // The inverse mapping recovers every candidate.
        for (int j = 0; j < set.count; ++j) {
          int symbol = -99;
          bool flag = false;
          CHECK(tx_type_symbol(q, set.candidates[j].type, &symbol, &flag));
          CHECK_EQ(symbol, static_cast<int>(set.candidates[j].symbol));
          CHECK_EQ(flag, set.candidates[j].is_long_side_dct);
        }

        // reduced_tx_set == 2 forces DCT_DCT before any set is consulted, so
        // the set id is reported but no type symbol exists.
        const bool forced_dct = reduced == 2;

        // INTRA_1/INTRA_2 must be exactly Md_Idx_To_Type.
        if (!forced_dct &&
            (set.set == TxSet::kIntra1 || set.set == TxSet::kIntra2)) {
          const int n = set.set == TxSet::kIntra1 ? 7 : 2;
          const int sc = kSizeClass[i];
          for (int j = 0; j < set.count; ++j) {
            CHECK_EQ(static_cast<int>(set.candidates[j].type),
                     static_cast<int>(kMdIdxToType[sc][mode][j]));
            CHECK_EQ(static_cast<int>(set.candidates[j].symbol), j);
            CHECK_EQ(n, set.count);
          }
        }
        // Long sets must be Tx_Type_Inv_Long.
        if (!forced_dct &&
            (set.set == TxSet::kWide64 || set.set == TxSet::kHigh64)) {
          const int orientation = set.set == TxSet::kWide64 ? 0 : 1;
          for (int j = 0; j < 4; ++j) {
            CHECK_EQ(static_cast<int>(set.candidates[j].type),
                     static_cast<int>(kTxTypeInvLong[1][orientation][j]));
            CHECK_EQ(static_cast<int>(set.candidates[j].symbol), j);
            CHECK(set.candidates[j].is_long_side_dct);
            CHECK(!set.candidates[j].codes_long_flag);
          }
        }
        if (!forced_dct &&
            (set.set == TxSet::kWide32 || set.set == TxSet::kHigh32)) {
          const int orientation = set.set == TxSet::kWide32 ? 0 : 1;
          for (int j = 0; j < 8; ++j) {
            const int flag = j / 4;
            CHECK_EQ(static_cast<int>(set.candidates[j].type),
                     static_cast<int>(kTxTypeInvLong[flag][orientation][j % 4]));
            CHECK_EQ(static_cast<int>(set.candidates[j].symbol), j % 4);
            CHECK_EQ(set.candidates[j].is_long_side_dct, flag == 1);
            CHECK(set.candidates[j].codes_long_flag);
          }
        }
      }
    }
  }

  // Spot checks copied from the spec tables.
  {
    TxTypeQuery q;
    q.coded_w = 4;
    q.coded_h = 4;
    q.mapped_mode = 0;
    TxTypeSet set;
    CHECK(intra_tx_candidates(q, &set));
    const int expected[7] = {0, 3, 1, 2, 7, 8, 13};
    for (int j = 0; j < 7; ++j) {
      CHECK_EQ(static_cast<int>(set.candidates[j].type), expected[j]);
    }
  }
  {
    TxTypeQuery q;
    q.coded_w = 4;
    q.coded_h = 4;
    q.mapped_mode = 12;  // PAETH
    TxTypeSet set;
    CHECK(intra_tx_candidates(q, &set));
    const int expected[7] = {0, 3, 2, 10, 11, 12, 13};
    for (int j = 0; j < 7; ++j) {
      CHECK_EQ(static_cast<int>(set.candidates[j].type), expected[j]);
    }
  }
  {
    // 16x32: HIGH_32, both is_long_side_dct branches.
    TxTypeQuery q;
    q.coded_w = 16;
    q.coded_h = 32;
    q.mapped_mode = 0;
    TxTypeSet set;
    CHECK(intra_tx_candidates(q, &set));
    const int expected[8] = {11, 13, 15, 9, 0, 2, 5, 10};
    for (int j = 0; j < 8; ++j) {
      CHECK_EQ(static_cast<int>(set.candidates[j].type), expected[j]);
    }
  }
  {
    // 16x64: HIGH_64, is_long_side_dct implied 1.
    TxTypeQuery q;
    q.coded_w = 16;
    q.coded_h = 64;
    q.mapped_mode = 0;
    TxTypeSet set;
    CHECK(intra_tx_candidates(q, &set));
    const int expected[4] = {0, 2, 5, 10};
    for (int j = 0; j < 4; ++j) {
      CHECK_EQ(static_cast<int>(set.candidates[j].type), expected[j]);
    }
  }
  {
    // 32x32 has no signalled intra type at all.
    TxTypeQuery q;
    q.coded_w = 32;
    q.coded_h = 32;
    TxTypeSet set;
    CHECK(intra_tx_candidates(q, &set));
    CHECK_EQ(set.count, 1);
    CHECK_EQ(static_cast<int>(set.candidates[0].type), 0);
    CHECK_EQ(set.candidates[0].symbol, -1);
  }

  // Mode-independent union: every legal type is reachable for some mode.
  {
    TxType out[16];
    const int n = intra_tx_types_for_size(4, 4, 0, false, out, 16);
    CHECK(n > 0);
    for (int j = 1; j < n; ++j) {
      CHECK(static_cast<int>(out[j - 1]) < static_cast<int>(out[j]));
    }
    // DCT_DCT and ADST_ADST are reachable; IDTX is not for a 4x4 intra block.
    bool has_dct = false;
    bool has_adst = false;
    bool has_idtx = false;
    for (int j = 0; j < n; ++j) {
      has_dct = has_dct || out[j] == TxType::kDctDct;
      has_adst = has_adst || out[j] == TxType::kAdstAdst;
      has_idtx = has_idtx || out[j] == TxType::kIdtx;
    }
    CHECK(has_dct);
    CHECK(has_adst);
    CHECK(!has_idtx);
    CHECK_EQ(intra_tx_types_for_size(2, 8, 0, false, out, 16), -1);
    CHECK_EQ(intra_tx_types_for_size(4, 4, 0, false, out, 1), -1);
    // IDTX is reachable on a wide-32 block (long-side identity branch).
    const int n32 = intra_tx_types_for_size(16, 32, 0, false, out, 16);
    CHECK(n32 > 0);
    bool has_idtx32 = false;
    for (int j = 0; j < n32; ++j) {
      has_idtx32 = has_idtx32 || out[j] == TxType::kIdtx;
    }
    CHECK(has_idtx32);
  }

  // dct_only mirrors enable_chroma_dctonly: INTRA_2 is reported for sizes
  // below 32x32, but only DCT_DCT can actually be coded.
  {
    TxTypeQuery q;
    q.coded_w = 16;
    q.coded_h = 8;
    q.mapped_mode = 3;
    q.dct_only = true;
    TxSet set;
    CHECK(intra_tx_set(q, &set));
    CHECK_EQ(static_cast<int>(set), static_cast<int>(TxSet::kIntra2));
    TxTypeSet candidates;
    CHECK(intra_tx_candidates(q, &candidates));
    CHECK_EQ(candidates.count, 1);
    CHECK_EQ(static_cast<int>(candidates.candidates[0].type), 0);
    CHECK_EQ(candidates.candidates[0].symbol, -1);
    // Selection over the forced set also works.
    std::vector<std::int16_t> residual(16 * 8);
    fill_residual(16, 8, 0, residual.data());
    TxTypeChoice choice;
    TxTypeCostConfig cfg;
    CHECK(choose_intra_tx_type(residual.data(), q, cfg, &choice));
    CHECK_EQ(static_cast<int>(choice.candidate.type), 0);
  }

  // Explicit candidate list: invalid arguments and the no-codable-candidate
  // path (two non-DCT types on a zero residual have eob == 0 and are skipped).
  {
    TxTypeQuery q;
    q.coded_w = 8;
    q.coded_h = 8;
    q.mapped_mode = 0;
    std::vector<std::int16_t> residual(8 * 8, 0);
    TxTypeCostConfig cfg;
    TxTypeChoice choice;
    TxTypeCandidate pair[2] = {make_candidate(q, TxType::kAdstAdst),
                               make_candidate(q, TxType::kDctAdst)};
    CHECK(!choose_tx_type_from_candidates(residual.data(), q, pair, 2, cfg,
                                          &choice));
    CHECK(!choose_tx_type_from_candidates(residual.data(), q, nullptr, 2, cfg,
                                          &choice));
    CHECK(!choose_tx_type_from_candidates(residual.data(), q, pair, 0, cfg,
                                          &choice));
    CHECK(!choose_tx_type_from_candidates(nullptr, q, pair, 2, cfg, &choice));
    TxTypeCandidate dct = make_candidate(q, TxType::kDctDct);
    CHECK(choose_tx_type_from_candidates(residual.data(), q, &dct, 1, cfg,
                                         &choice));
    CHECK_EQ(static_cast<int>(choice.candidate.type), 0);
  }

  // Invalid arguments.
  {
    TxTypeSet set;
    TxTypeQuery q;
    q.coded_w = 4;
    q.coded_h = 4;
    q.mapped_mode = 13;
    CHECK(!intra_tx_candidates(q, &set));
    q.mapped_mode = -1;
    CHECK(!intra_tx_candidates(q, &set));
    q.mapped_mode = 0;
    q.coded_w = 2;
    CHECK(!intra_tx_candidates(q, &set));
    CHECK(!intra_tx_candidates(q, nullptr));
    int symbol = 0;
    bool flag = false;
    CHECK(!tx_type_symbol(q, TxType::kDctDct, &symbol, &flag));
  }
  suite_end();
}

// ---------------------------------------------------------------------------
// wide_angle_mapping
// ---------------------------------------------------------------------------
void test_mode_mapping() {
  suite_start("mode_mapping");
  CHECK_EQ(map_intra_mode(0, 0, 4, 4), 0);    // DC
  CHECK_EQ(map_intra_mode(3, 0, 4, 4), 3);    // D45
  CHECK_EQ(map_intra_mode(9, 3, 32, 4), 9);   // SMOOTH ignores delta
  CHECK_EQ(map_intra_mode(1, 2, 4, 4), 1);    // V + delta stays V at 90 deg
  for (int mode = 0; mode < kNumIntraModes; ++mode) {
    for (int delta = -3; delta <= 3; ++delta) {
      for (int i = 0; i < 25; ++i) {
        const int w = txv::kTxSizes[i].width;
        const int h = txv::kTxSizes[i].height;
        const int mapped = map_intra_mode(mode, delta, w, h);
        const intra::MappedAngle direct = intra::map_angle(
            static_cast<intra::Mode>(mode), delta, w, h);
        CHECK_EQ(mapped, static_cast<int>(direct.mode));
        CHECK(mapped >= 0 && mapped < kNumIntraModes);
        // Only the signalled mode is mapped once; wide_angle_mapping is not
        // defined to be idempotent, so it is not applied twice here.
      }
    }
  }
  CHECK_EQ(map_intra_mode(13, 0, 4, 4), -1);
  CHECK_EQ(map_intra_mode(0, 4, 4, 4), -1);
  CHECK_EQ(map_intra_mode(0, 0, 3, 4), -1);
  suite_end();
}

// ---------------------------------------------------------------------------
// §5.20.7.30 scan
// ---------------------------------------------------------------------------
void test_scan() {
  suite_start("diagonal_scan");
  const std::uint16_t* tables[4] = {kScan4, kScan8, kScan16, kScan32};
  const int sizes[4] = {4, 8, 16, 32};
  for (int k = 0; k < 4; ++k) {
    const int n = sizes[k];
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(n * n));
    diagonal_scan(n, n, scan.data());
    for (int c = 0; c < n * n; ++c) {
      CHECK_EQ(static_cast<int>(scan[static_cast<std::size_t>(c)]),
               static_cast<int>(tables[k][c]));
    }
  }
  // Every adjusted size: a permutation of 0..w*h-1.
  for (int i = 0; i < 25; ++i) {
    const auto p = txv::make_tx_params(txv::kTxSizes[i].width,
                                      txv::kTxSizes[i].height);
    CHECK(p.has_value());
    const int w = p->w;
    const int h = p->h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(w * h));
    diagonal_scan(w, h, scan.data());
    std::vector<int> hits(static_cast<std::size_t>(w * h), 0);
    for (int c = 0; c < w * h; ++c) {
      const std::size_t pos = scan[static_cast<std::size_t>(c)];
      CHECK(pos < static_cast<std::size_t>(w * h));
      ++hits[pos];
    }
    for (int j = 0; j < w * h; ++j) {
      CHECK_EQ(hits[static_cast<std::size_t>(j)], 1);
    }
  }
  // EOB.
  {
    std::int32_t levels[16] = {};
    CHECK_EQ(eob_from_levels(levels, 4, 4), 0);
    levels[0] = 3;
    CHECK_EQ(eob_from_levels(levels, 4, 4), 1);
    levels[15] = -1;
    CHECK_EQ(eob_from_levels(levels, 4, 4), 16);
    levels[15] = 0;
    levels[4] = 1;  // scan position 1
    CHECK_EQ(eob_from_levels(levels, 4, 4), 2);
  }
  suite_end();
}

// ---------------------------------------------------------------------------
// Exhaustive per-type transform/quantize/inverse
// ---------------------------------------------------------------------------
void test_exhaustive_transforms() {
  suite_start("exhaustive_transforms");
  std::vector<std::int16_t> residual(64 * 64);
  std::vector<std::int16_t> adj(txv::kMaxAdjTxElems);
  std::vector<double> coeff(txv::kMaxAdjTxElems);
  std::vector<std::int32_t> levels_ref(txv::kMaxAdjTxElems);
  std::vector<std::int32_t> dequant(txv::kMaxAdjTxElems);
  std::vector<std::int16_t> recon_ref(64 * 64);
  std::vector<std::int32_t> levels(txv::kMaxAdjTxElems);
  std::vector<std::int16_t> recon(64 * 64);
  std::vector<std::int16_t> recon_again(64 * 64);

  for (int size = 0; size < 25; ++size) {
    const int cw = txv::kTxSizes[size].width;
    const int ch = txv::kTxSizes[size].height;
    const int coded_n = cw * ch;
    for (int kind = 0; kind < 2; ++kind) {
      fill_residual(cw, ch, kind, residual.data());
      for (int t = 0; t < 16; ++t) {
        const auto type = static_cast<TxType>(t);
        TxTypeQuery q;
        q.coded_w = cw;
        q.coded_h = ch;
        q.mapped_mode = 0;
        TxTypeCandidate cand = make_candidate(q, type);
        TxTypeCostConfig cfg;
        cfg.lambda = 0.0;
        TxTypeEval ev;
        CHECK(evaluate_intra_tx_type(residual.data(), q, cand, cfg, &ev,
                                     levels.data(), recon.data()));
        CHECK_EQ(ev.candidate.params.w, cand.params.w);
        CHECK_EQ(ev.candidate.params.h, cand.params.h);

        // Independent path straight through tx_variants.
        txv::downsample_residual(residual.data(), cand.params, adj.data());
        txv::forward_transform(adj.data(), cand.params, coeff.data(),
                               levels_ref.data());
        txv::inverse_transform(levels_ref.data(), cand.params, dequant.data(),
                               recon_ref.data());
        CHECK(levels_equal(levels.data(), levels_ref.data(),
                           cand.params.w * cand.params.h));
        CHECK(residual_equal(recon.data(), recon_ref.data(), coded_n));

        // Metrics.
        std::int32_t nz = 0;
        std::int64_t sum_abs = 0;
        for (int j = 0; j < cand.params.w * cand.params.h; ++j) {
          if (levels.data()[j] != 0) {
            ++nz;
            sum_abs += std::abs(static_cast<std::int64_t>(levels.data()[j]));
          }
        }
        CHECK_EQ(ev.nonzero, nz);
        CHECK_EQ(ev.sum_abs, sum_abs);
        CHECK_EQ(ev.eob, eob_from_levels(levels.data(), cand.params.w,
                                         cand.params.h));
        CHECK(ev.eob >= 0 && ev.eob <= cand.params.w * cand.params.h);
        CHECK(std::fabs(ev.sse - sse_of(recon.data(), residual.data(),
                                        coded_n)) < 1e-9);
        CHECK_EQ(ev.codable, candidate_codable(type, ev.eob));
        CHECK(ev.type_bits >= 0.0);
        CHECK_EQ(ev.rate_bits, expected_rate(ev, cfg));
        CHECK(std::fabs(ev.cost -
                        (ev.sse + cfg.lambda * ev.rate_bits)) < 1e-9);

        // Determinism.
        CHECK(evaluate_intra_tx_type(residual.data(), q, cand, cfg, &ev,
                                     levels.data(), recon_again.data()));
        CHECK(residual_equal(recon.data(), recon_again.data(), coded_n));

        // The 32-length aliasing rule must be visible in the coefficients.
        // (Checked exhaustively in test_aliasing_32.)
      }
    }
  }
  suite_end();
}

// ---------------------------------------------------------------------------
// §7.15.2.1 length-32 ADST/FDST aliasing
// ---------------------------------------------------------------------------
void test_aliasing_32() {
  suite_start("aliasing_32");
  std::vector<std::int16_t> residual(64 * 64);
  std::vector<std::int16_t> adj(txv::kMaxAdjTxElems);
  std::vector<double> coeff_a(txv::kMaxAdjTxElems);
  std::vector<double> coeff_b(txv::kMaxAdjTxElems);
  std::vector<std::int32_t> levels_a(txv::kMaxAdjTxElems);
  std::vector<std::int32_t> levels_b(txv::kMaxAdjTxElems);
  std::vector<std::int16_t> recon_a(64 * 64);
  std::vector<std::int16_t> recon_b(64 * 64);

  int aliased_cases = 0;
  for (int size = 0; size < 25; ++size) {
    const int cw = txv::kTxSizes[size].width;
    const int ch = txv::kTxSizes[size].height;
    fill_residual(cw, ch, 1, residual.data());
    for (int t = 0; t < 16; ++t) {
      const auto type = static_cast<TxType>(t);
      const auto pa = txv::make_tx_params(cw, ch, type);
      CHECK(pa.has_value());
      if (!pa.has_value()) {
        continue;
      }
      // Replace ADST/FDST with DCT on every 32-length dimension.
      txv::Tx1dType row = static_cast<txv::Tx1dType>(pa->row_type);
      txv::Tx1dType col = static_cast<txv::Tx1dType>(pa->col_type);
      bool changed = false;
      if (pa->w == 32 && (row == txv::Tx1dType::kAdst ||
                          row == txv::Tx1dType::kFdst)) {
        row = txv::Tx1dType::kDct;
        changed = true;
      }
      if (pa->h == 32 && (col == txv::Tx1dType::kAdst ||
                          col == txv::Tx1dType::kFdst)) {
        col = txv::Tx1dType::kDct;
        changed = true;
      }
      if (!changed) {
        continue;
      }
      TxType canonical = type;
      CHECK(txv::make_tx_type(row, col, &canonical));
      const auto pb = txv::make_tx_params(cw, ch, canonical);
      CHECK(pb.has_value());
      if (!pb.has_value()) {
        continue;
      }
      ++aliased_cases;

      txv::downsample_residual(residual.data(), *pa, adj.data());
      txv::forward_block(adj.data(), *pa, coeff_a.data());
      txv::quantize(coeff_a.data(), *pa, levels_a.data());
      txv::inverse_block(levels_a.data(), *pa, recon_a.data());

      txv::forward_block(adj.data(), *pb, coeff_b.data());
      txv::quantize(coeff_b.data(), *pb, levels_b.data());
      txv::inverse_block(levels_b.data(), *pb, recon_b.data());

      const int n = pa->w * pa->h;
      for (int i = 0; i < n; ++i) {
        CHECK(coeff_a[static_cast<std::size_t>(i)] ==
              coeff_b[static_cast<std::size_t>(i)]);
        CHECK_EQ(levels_a[static_cast<std::size_t>(i)],
                 levels_b[static_cast<std::size_t>(i)]);
      }
      CHECK(residual_equal(recon_a.data(), recon_b.data(), cw * ch));
    }
  }
  CHECK(aliased_cases > 0);
  std::printf("  aliased size/type pairs compared: %d\n", aliased_cases);

  // 64x64 is coded as an adjusted 32x32 transform, so every ADST/FLIPADST
  // variant equals 64x64 DCT_DCT.
  {
    fill_residual(64, 64, 0, residual.data());
    const auto p_dct = txv::make_tx_params(64, 64, TxType::kDctDct);
    txv::downsample_residual(residual.data(), *p_dct, adj.data());
    txv::forward_block(adj.data(), *p_dct, coeff_a.data());
    for (TxType type : {TxType::kAdstAdst, TxType::kAdstDct, TxType::kDctAdst,
                        TxType::kFlipadstFlipadst, TxType::kAdstFlipadst}) {
      const auto p = txv::make_tx_params(64, 64, type);
      txv::forward_block(adj.data(), *p, coeff_b.data());
      for (int i = 0; i < 32 * 32; ++i) {
        CHECK(coeff_a[static_cast<std::size_t>(i)] ==
              coeff_b[static_cast<std::size_t>(i)]);
      }
    }
  }
  suite_end();
}

// ---------------------------------------------------------------------------
// Rate model
// ---------------------------------------------------------------------------
double counting_rate_fn(const TxTypeRateInputs& in, void* user) {
  auto* counter = static_cast<int*>(user);
  ++*counter;
  return 7.0 + 0.5 * static_cast<double>(in.nonzero);
}

void test_rate_model() {
  suite_start("rate_model");

  TxTypeQuery q;
  q.coded_w = 4;
  q.coded_h = 4;
  q.mapped_mode = 0;
  TxTypeSet set;
  CHECK(intra_tx_candidates(q, &set));
  CHECK_EQ(set.count, 7);

  // DCT is symbol 0 of INTRA_1 row 0: P = 5026/32768.
  const double expect_dct = -std::log2(5026.0 / 32768.0);
  CHECK(std::fabs(static_type_bits(q, set.candidates[0]) - expect_dct) <
        1e-12);
  // Every symbol cost is finite and positive.
  for (int j = 0; j < set.count; ++j) {
    const double bits = static_type_bits(q, set.candidates[j]);
    CHECK(bits > 0.0 && bits < 40.0);
  }
  // Static costs must agree with the default CDF probabilities.  Row 0 of
  // Default_Intra_Tx_Type_Set1_Cdf is {5026, 16816, 19974, 23404, 26845,
  // 30499, 8, 0}, so symbol 1 (ADST_ADST) is the most probable.
  for (int j = 0; j < set.count; ++j) {
    int lo = j == 0 ? 0 : static_cast<int>(kIntraTxTypeSet1Cdf[0][j - 1]);
    int hi = j == 6 ? 32768 : static_cast<int>(kIntraTxTypeSet1Cdf[0][j]);
    const double expect = -std::log2(static_cast<double>(hi - lo) / 32768.0);
    CHECK(std::fabs(static_type_bits(q, set.candidates[j]) - expect) < 1e-12);
  }
  CHECK(static_type_bits(q, set.candidates[1]) <
        static_type_bits(q, set.candidates[0]));

  // DCTONLY codes no type.
  {
    TxTypeQuery d;
    d.coded_w = 32;
    d.coded_h = 32;
    TxTypeSet s;
    CHECK(intra_tx_candidates(d, &s));
    CHECK_EQ(static_type_bits(d, s.candidates[0]), 0.0);
    CHECK_EQ(static_type_bits(d, s.candidates[0]), 0.0);
  }
  // Long-32 codes the flag: is_long_side_dct = 1 is far more likely than 0.
  {
    TxTypeQuery w;
    w.coded_w = 16;
    w.coded_h = 32;
    w.mapped_mode = 0;
    TxTypeSet s;
    CHECK(intra_tx_candidates(w, &s));
    // Default_Is_Long_Side_Dct_Cdf[0] = {36, 0, 0}: flag 0 costs ~9.8 bits,
    // flag 1 ~0.001 bits.  For every symbol the flag = 1 branch is cheaper,
    // and the cheapest flag = 1 candidate is well under one bit.
    double min_flag0 = 1e9;
    double min_flag1 = 1e9;
    for (int j = 0; j < 4; ++j) {
      const double b0 = static_type_bits(w, s.candidates[j]);
      const double b1 = static_type_bits(w, s.candidates[j + 4]);
      min_flag0 = std::min(min_flag0, b0);
      min_flag1 = std::min(min_flag1, b1);
      CHECK(b1 < b0);
    }
    CHECK(min_flag0 > 5.0);   // flag = 0 costs ~9.8 bits
    CHECK(min_flag1 < 1.0);   // flag = 1 costs ~0 bits
    CHECK_EQ(s.candidates[0].is_long_side_dct, false);
    CHECK_EQ(s.candidates[4].is_long_side_dct, true);
  }

  std::vector<std::int16_t> residual(16 * 16);
  fill_residual(16, 16, 0, residual.data());
  TxTypeQuery q16;
  q16.coded_w = 16;
  q16.coded_h = 16;
  q16.mapped_mode = 0;
  TxTypeSet s16;
  CHECK(intra_tx_candidates(q16, &s16));
  TxTypeEval ev;
  TxTypeCostConfig cfg;
  CHECK(evaluate_intra_tx_type(residual.data(), q16, s16.candidates[3], cfg,
                               &ev));
  CHECK_EQ(ev.type_bits, static_type_bits(q16, s16.candidates[3]));

  // Default model: type bits + nonzero.
  TxTypeRateInputs in;
  in.type_bits = ev.type_bits;
  in.nonzero = ev.nonzero;
  CHECK(std::fabs(default_rate_bits(in, nullptr) -
                  (ev.type_bits + ev.nonzero)) < 1e-12);

  // coefficient_bits scaling.
  cfg.coefficient_bits = 2.5;
  CHECK(std::fabs(tx_type_rate_bits(in, cfg) -
                  (ev.type_bits + 2.5 * ev.nonzero)) < 1e-12);
  // Disabling the static type cost.
  cfg.coefficient_bits = 1.0;
  cfg.use_static_type_cost = false;
  CHECK(std::fabs(tx_type_rate_bits(in, cfg) -
                  static_cast<double>(ev.nonzero)) < 1e-12);

  // Custom callback overrides everything and is invoked per candidate.
  int calls = 0;
  cfg.rate_fn = counting_rate_fn;
  cfg.rate_user = &calls;
  TxTypeEval ev2;
  CHECK(evaluate_intra_tx_type(residual.data(), q16, s16.candidates[3], cfg,
                               &ev2));
  CHECK_EQ(calls, 1);
  CHECK(std::fabs(ev2.rate_bits -
                  (7.0 + 0.5 * static_cast<double>(ev2.nonzero))) < 1e-12);
  CHECK(std::fabs(ev2.cost - (ev2.sse + cfg.lambda * ev2.rate_bits)) < 1e-9);
  suite_end();
}

// ---------------------------------------------------------------------------
// Deterministic tie-break
// ---------------------------------------------------------------------------
void test_tie_break() {
  suite_start("tie_break");

  // Zero residual: every candidate has SSE 0 and eob 0; only DCT_DCT is
  // codable, and it has the lowest static type cost.
  {
    std::vector<std::int16_t> residual(16 * 16, 0);
    TxTypeQuery q;
    q.coded_w = 16;
    q.coded_h = 16;
    q.mapped_mode = 0;
    TxTypeCostConfig cfg;
    cfg.lambda = 600.0;
    TxTypeChoice choice;
    CHECK(choose_intra_tx_type(residual.data(), q, cfg, &choice));
    CHECK(choice.valid);
    CHECK_EQ(static_cast<int>(choice.candidate.type),
             static_cast<int>(TxType::kDctDct));
    CHECK_EQ(choice.eval.eob, 0);
    CHECK_EQ(choice.eval.nonzero, 0);
    CHECK_EQ(choice.eval.sse, 0.0);
  }

  // Tie-break by TxType value, not by candidate order.  For a 16x32 block
  // ADST_DCT (1) aliases DCT_DCT (0) because its 32-length column transform is
  // §7.15.2.1 Dct_Kernel32.  Feed the pair in reverse order with a constant
  // rate model: the costs and SSEs are identical, so the rule must pick the
  // lower TxType (0) even though type 1 is first in the list.
  {
    std::vector<std::int16_t> residual(16 * 32);
    fill_residual(16, 32, 1, residual.data());
    TxTypeQuery q;
    q.coded_w = 16;
    q.coded_h = 32;
    q.mapped_mode = 0;
    TxTypeCostConfig cfg;
    cfg.lambda = 1000.0;
    cfg.rate_fn = [](const TxTypeRateInputs&, void*) { return 0.0; };
    TxTypeCandidate pair[2];
    pair[0] = make_candidate(q, TxType::kAdstDct);
    pair[1] = make_candidate(q, TxType::kDctDct);
    TxTypeChoice choice;
    CHECK(choose_tx_type_from_candidates(residual.data(), q, pair, 2, cfg,
                                        &choice));
    CHECK_EQ(static_cast<int>(choice.candidate.type), 0);
    CHECK_EQ(choice.candidate_index, 1);
    CHECK_EQ(choice.candidates_considered, 2);

    // Same for the {ADST_ADST, DCT_ADST} alias: ADST_ADST (3) and DCT_ADST
    // (2) differ only in the 32-length column, so type 2 must win.
    pair[0] = make_candidate(q, TxType::kAdstAdst);
    pair[1] = make_candidate(q, TxType::kDctAdst);
    CHECK(choose_tx_type_from_candidates(residual.data(), q, pair, 2, cfg,
                                        &choice));
    CHECK_EQ(static_cast<int>(choice.candidate.type),
             static_cast<int>(TxType::kDctAdst));
    CHECK_EQ(choice.candidate_index, 1);
  }

  // Cost tie -> lower SSE: a rate model that cancels the SSE term makes every
  // candidate cost 0, so the winner must be the minimum-SSE candidate found by
  // an independent scan of the legal set.
  {
    std::vector<std::int16_t> residual(8 * 8);
    fill_residual(8, 8, 2, residual.data());
    TxTypeQuery q;
    q.coded_w = 8;
    q.coded_h = 8;
    q.mapped_mode = 0;
    TxTypeCostConfig cfg;
    cfg.lambda = 1.0;
    cfg.rate_fn = [](const TxTypeRateInputs& in, void*) { return -in.sse; };
    TxTypeChoice choice;
    CHECK(choose_intra_tx_type(residual.data(), q, cfg, &choice));
    CHECK_EQ(choice.eval.cost, 0.0);
    TxTypeSet set;
    CHECK(intra_tx_candidates(q, &set));
    double min_sse = 1e300;
    int min_type = 999;
    for (int i = 0; i < set.count; ++i) {
      TxTypeEval ev;
      CHECK(evaluate_intra_tx_type(residual.data(), q, set.candidates[i], cfg,
                                   &ev));
      if (!ev.codable) {
        continue;
      }
      if (ev.sse < min_sse ||
          (ev.sse == min_sse &&
           static_cast<int>(ev.candidate.type) < min_type)) {
        min_sse = ev.sse;
        min_type = static_cast<int>(ev.candidate.type);
      }
    }
    CHECK_EQ(static_cast<int>(choice.candidate.type), min_type);
    CHECK_EQ(choice.eval.sse, min_sse);
  }

  // The alias case: 32x32 is DCTONLY for intra, but the transform layer still
  // maps every ADST/FDST at a 32-sample dimension onto DCT.  Confirmed here at
  // the transform level (selection cannot see it because the set has one type).
  {
    std::vector<std::int16_t> residual(32 * 32);
    fill_residual(32, 32, 1, residual.data());
    const auto p_dct = txv::make_tx_params(32, 32, TxType::kDctDct);
    const auto p_adst = txv::make_tx_params(32, 32, TxType::kAdstAdst);
    std::vector<double> coeff_a(txv::kMaxAdjTxElems);
    std::vector<double> coeff_b(txv::kMaxAdjTxElems);
    std::vector<std::int32_t> levels_a(txv::kMaxAdjTxElems);
    std::vector<std::int32_t> levels_b(txv::kMaxAdjTxElems);
    std::vector<std::int16_t> adj(txv::kMaxAdjTxElems);
    txv::downsample_residual(residual.data(), *p_dct, adj.data());
    txv::forward_block(adj.data(), *p_dct, coeff_a.data());
    txv::quantize(coeff_a.data(), *p_dct, levels_a.data());
    txv::forward_block(adj.data(), *p_adst, coeff_b.data());
    txv::quantize(coeff_b.data(), *p_adst, levels_b.data());
    for (int i = 0; i < 32 * 32; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      CHECK(coeff_a[idx] == coeff_b[idx]);
      CHECK_EQ(levels_a[idx], levels_b[idx]);
    }
  }

  // Candidate order must not affect the decision: the same zero-residual tie
  // in a mode whose symbol list is different still resolves to DCT_DCT.
  {
    std::vector<std::int16_t> residual(8 * 8, 0);
    TxTypeCostConfig cfg;
    cfg.lambda = 10.0;
    cfg.rate_fn = [](const TxTypeRateInputs&, void*) { return 0.0; };
    TxTypeQuery q0;
    q0.coded_w = 8;
    q0.coded_h = 8;
    q0.mapped_mode = 0;
    TxTypeQuery q12 = q0;
    q12.mapped_mode = 12;
    TxTypeChoice a;
    TxTypeChoice b;
    CHECK(choose_intra_tx_type(residual.data(), q0, cfg, &a));
    CHECK(choose_intra_tx_type(residual.data(), q12, cfg, &b));
    CHECK(a.valid && b.valid);
    CHECK_EQ(static_cast<int>(a.candidate.type), 0);
    CHECK_EQ(static_cast<int>(b.candidate.type), 0);
    CHECK_EQ(a.eval.cost, 0.0);
    CHECK_EQ(b.eval.cost, 0.0);
  }
  suite_end();
}

// ---------------------------------------------------------------------------
// Selection vs brute force
// ---------------------------------------------------------------------------
bool brute_force(const std::int16_t* residual, const TxTypeQuery& query,
                 const TxTypeCostConfig& config, int* type_out,
                 double* cost_out, double* sse_out, int* index_out) {
  bool have = false;
  TxTypeEval best;
  int best_index = -1;
  for (int t = 0; t < 16; ++t) {
    const auto type = static_cast<TxType>(t);
    int symbol = 0;
    bool flag = false;
    if (!tx_type_symbol(query, type, &symbol, &flag)) {
      continue;
    }
    TxTypeCandidate cand = make_candidate(query, type);
    cand.symbol = static_cast<std::int8_t>(symbol);
    cand.is_long_side_dct = flag;
    TxTypeEval ev;
    if (!evaluate_intra_tx_type(residual, query, cand, config, &ev)) {
      return false;
    }
    if (!ev.codable) {
      continue;
    }
    const bool better =
        !have || ev.cost < best.cost ||
        (ev.cost == best.cost &&
         (ev.sse < best.sse ||
          (ev.sse == best.sse && t < static_cast<int>(best.candidate.type))));
    if (better) {
      have = true;
      best = ev;
      best_index = t;
    }
  }
  if (!have) {
    return false;
  }
  *type_out = static_cast<int>(best.candidate.type);
  *cost_out = best.cost;
  *sse_out = best.sse;
  *index_out = best_index;
  return true;
}

void test_selection_bruteforce() {
  suite_start("selection_bruteforce");
  std::vector<std::int16_t> residual(64 * 64);
  const double lambdas[3] = {0.0, 544.5, 25394.7};
  const int modes[4] = {0, 3, 9, 12};
  const int reduced_sets[3] = {0, 1, 3};
  int cases = 0;

  for (int size = 0; size < 25; ++size) {
    const int cw = txv::kTxSizes[size].width;
    const int ch = txv::kTxSizes[size].height;
    for (int kind = 0; kind < 4; ++kind) {
      fill_residual(cw, ch, kind, residual.data());
      for (int mi = 0; mi < 4; ++mi) {
        for (int ri = 0; ri < 3; ++ri) {
          for (double lambda : lambdas) {
            TxTypeQuery q;
            q.coded_w = cw;
            q.coded_h = ch;
            q.mapped_mode = modes[mi];
            q.reduced_tx_set = reduced_sets[ri];
            TxTypeCostConfig cfg;
            cfg.lambda = lambda;
            TxTypeChoice choice;
            CHECK(choose_intra_tx_type(residual.data(), q, cfg, &choice));
            int bf_type = -1;
            int bf_index = -1;
            double bf_cost = 0.0;
            double bf_sse = 0.0;
            CHECK(brute_force(residual.data(), q, cfg, &bf_type, &bf_cost,
                              &bf_sse, &bf_index));
            CHECK_EQ(static_cast<int>(choice.candidate.type), bf_type);
            CHECK(std::fabs(choice.eval.cost - bf_cost) < 1e-9);
            CHECK(std::fabs(choice.eval.sse - bf_sse) < 1e-9);
            // The chosen type must be legal and codable.
            int symbol = 0;
            bool flag = false;
            CHECK(tx_type_symbol(q, choice.candidate.type, &symbol, &flag));
            CHECK(choice.eval.codable);

            // Winner levels/recon must match a fresh evaluation.
            std::vector<std::int32_t> levels(
                static_cast<std::size_t>(choice.candidate.params.w *
                                         choice.candidate.params.h));
            std::vector<std::int16_t> recon(static_cast<std::size_t>(cw * ch));
            CHECK(choose_intra_tx_type(residual.data(), q, cfg, &choice,
                                       levels.data(), recon.data()));
            TxTypeEval direct;
            std::vector<std::int32_t> levels2(levels.size());
            std::vector<std::int16_t> recon2(recon.size());
            CHECK(evaluate_intra_tx_type(residual.data(), q,
                                         choice.candidate, cfg, &direct,
                                         levels2.data(), recon2.data()));
            CHECK(levels_equal(levels.data(), levels2.data(),
                               static_cast<int>(levels.size())));
            CHECK(residual_equal(recon.data(), recon2.data(), cw * ch));
            ++cases;
          }
        }
      }
    }
  }
  std::printf("  selection/brute-force cases: %d\n", cases);

  // DCT is always in the set, so the chosen cost never exceeds DCT's.
  for (int size = 0; size < 25; ++size) {
    const int cw = txv::kTxSizes[size].width;
    const int ch = txv::kTxSizes[size].height;
    fill_residual(cw, ch, 0, residual.data());
    TxTypeQuery q;
    q.coded_w = cw;
    q.coded_h = ch;
    q.mapped_mode = 0;
    TxTypeCostConfig cfg;
    cfg.lambda = 544.5;
    TxTypeChoice choice;
    CHECK(choose_intra_tx_type(residual.data(), q, cfg, &choice));
    TxTypeCandidate dct = make_candidate(q, TxType::kDctDct);
    TxTypeEval dct_eval;
    CHECK(evaluate_intra_tx_type(residual.data(), q, dct, cfg, &dct_eval));
    CHECK(choice.eval.cost <= dct_eval.cost + 1e-9);
    if (cfg.lambda == 0.0) {
      CHECK(choice.eval.sse <= dct_eval.sse + 1e-9);
    }
  }
  suite_end();
}

// ---------------------------------------------------------------------------
// Real frames
// ---------------------------------------------------------------------------
struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> luma;
};

bool load_y4m(const std::filesystem::path& path, Frame* frame) {
  Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::printf("  cannot open %s: %s\n", path.c_str(), error.c_str());
    return false;
  }
  Y4mFrame y4m;
  const Y4mReadResult result = reader.read_frame(y4m, error);
  if (result != Y4mReadResult::kFrame) {
    std::printf("  cannot read frame from %s: %s\n", path.c_str(),
                error.c_str());
    return false;
  }
  frame->width = static_cast<int>(y4m.width());
  frame->height = static_cast<int>(y4m.height());
  const std::span<const std::uint8_t> y = y4m.y();
  frame->luma.assign(y.begin(), y.end());
  return true;
}

// DC prediction from the *original* frame (prediction isolation): the
// residual is source - DC(original neighbours), so the histogram measures only
// the transform-type decision, not reconstruction drift.
bool dc_residual(const Frame& frame, int x0, int y0, int bw, int bh,
                 std::vector<std::int16_t>* residual) {
  intra::NeighborRequest req;
  req.recon.data = frame.luma.data();
  req.recon.stride = frame.width;
  req.x = x0;
  req.y = y0;
  req.w = bw;
  req.h = bh;
  req.max_x = frame.width - 1;
  req.max_y = frame.height - 1;
  req.have_above = y0 > 0;
  req.have_left = x0 > 0;
  req.num4_above_right = 0;
  req.num4_below_left = 0;
  req.color_plane = 0;
  std::vector<std::uint8_t> pred(static_cast<std::size_t>(bw * bh));
  intra::PredictParams params;
  params.mode = intra::Mode::kDc;
  if (intra::predict_block(req, 8, params, pred.data(),
                           static_cast<std::size_t>(bw)) !=
      intra::IntraStatus::kOk) {
    return false;
  }
  residual->resize(static_cast<std::size_t>(bw * bh));
  for (int y = 0; y < bh; ++y) {
    for (int x = 0; x < bw; ++x) {
      const std::size_t src_pos =
          static_cast<std::size_t>((y0 + y) * frame.width + x0 + x);
      const std::size_t pos = static_cast<std::size_t>(y * bw + x);
      const int src = frame.luma[src_pos];
      const int p = pred[pos];
      (*residual)[pos] = static_cast<std::int16_t>(src - p);
    }
  }
  return true;
}

struct Histogram {
  int blocks = 0;
  int selected[16] = {};
  double sse_total = 0.0;
  double cost_total = 0.0;
  double dct_sse_total = 0.0;
  double dct_cost_total = 0.0;
  int dct_wins = 0;
  int non_dct_better_sse = 0;
  int non_dct_better_cost = 0;
};

void print_histogram(const char* label, const Histogram& hist) {
  std::printf("\n%s: %d blocks, lambda-cost\n", label, hist.blocks);
  std::printf("  type            selected   (sse sum)\n");
  for (int t = 0; t < 16; ++t) {
    if (hist.selected[t] != 0) {
      std::printf("  %-14s %8d\n", txv::tx_type_name(static_cast<TxType>(t)),
                  hist.selected[t]);
    }
  }
  std::printf("  dct wins: %d/%d; non-DCT lower SSE: %d; non-DCT lower cost: "
              "%d\n",
              hist.dct_wins, hist.blocks, hist.non_dct_better_sse,
              hist.non_dct_better_cost);
  std::printf("  sse total %.0f (dct %.0f), cost total %.0f (dct %.0f)\n",
              hist.sse_total, hist.dct_sse_total, hist.cost_total,
              hist.dct_cost_total);
}

bool run_blocks(const Frame& frame, int bw, int bh, int mode, int reduced,
                double lambda, Histogram* hist, const char* label,
                bool verbose) {
  std::vector<std::int16_t> residual;
  std::vector<std::int32_t> levels;
  std::vector<std::int16_t> recon;
  for (int y0 = 0; y0 + bh <= frame.height; y0 += bh) {
    for (int x0 = 0; x0 + bw <= frame.width; x0 += bw) {
      if (!dc_residual(frame, x0, y0, bw, bh, &residual)) {
        CHECK(false);
        return false;
      }
      TxTypeQuery q;
      q.coded_w = bw;
      q.coded_h = bh;
      q.mapped_mode = mode;
      q.reduced_tx_set = reduced;
      TxTypeCostConfig cfg;
      cfg.lambda = lambda;
      TxTypeChoice choice;
      CHECK(choose_intra_tx_type(residual.data(), q, cfg, &choice));
      if (!choice.valid) {
        return false;
      }
      TxTypeCandidate dct = make_candidate(q, TxType::kDctDct);
      TxTypeEval dct_eval;
      CHECK(evaluate_intra_tx_type(residual.data(), q, dct, cfg, &dct_eval));

      const int type = static_cast<int>(choice.candidate.type);
      CHECK(type >= 0 && type < 16);
      int symbol = 0;
      bool flag = false;
      CHECK(tx_type_symbol(q, choice.candidate.type, &symbol, &flag));
      CHECK(choice.eval.codable);
      CHECK(choice.eval.cost <= dct_eval.cost + 1e-9);

      ++hist->blocks;
      ++hist->selected[type];
      hist->sse_total += choice.eval.sse;
      hist->cost_total += choice.eval.cost;
      hist->dct_sse_total += dct_eval.sse;
      hist->dct_cost_total += dct_eval.cost;
      if (type == 0) {
        ++hist->dct_wins;
      } else {
        if (choice.eval.sse < dct_eval.sse) ++hist->non_dct_better_sse;
        if (choice.eval.cost < dct_eval.cost) ++hist->non_dct_better_cost;
      }

      // Winner output buffers must reproduce the evaluation exactly.
      const int adj_n = choice.candidate.params.w * choice.candidate.params.h;
      levels.assign(static_cast<std::size_t>(adj_n), 0);
      recon.assign(static_cast<std::size_t>(bw * bh), 0);
      TxTypeChoice again;
      CHECK(choose_intra_tx_type(residual.data(), q, cfg, &again,
                                 levels.data(), recon.data()));
      CHECK_EQ(static_cast<int>(again.candidate.type), type);
      CHECK(std::fabs(again.eval.sse - choice.eval.sse) < 1e-9);
      TxTypeEval direct;
      std::vector<std::int32_t> levels2(static_cast<std::size_t>(adj_n));
      std::vector<std::int16_t> recon2(static_cast<std::size_t>(bw * bh));
      CHECK(evaluate_intra_tx_type(residual.data(), q, choice.candidate, cfg,
                                   &direct, levels2.data(), recon2.data()));
      CHECK(levels_equal(levels.data(), levels2.data(), adj_n));
      CHECK(residual_equal(recon.data(), recon2.data(), bw * bh));
      CHECK_EQ(direct.eob, eob_from_levels(levels.data(),
                                           choice.candidate.params.w,
                                           choice.candidate.params.h));
    }
  }
  if (verbose) {
    print_histogram(label, *hist);
  }
  return true;
}

void test_real_frames(const std::filesystem::path& data_dir) {
  suite_start("real_frames");
  const std::filesystem::path path64 = data_dir / "basic_64x64_1f.y4m";
  const std::filesystem::path path360 = data_dir / "basic_640x360_1f.y4m";
  const bool allow_missing =
      std::getenv("AV2ENC_ALLOW_MISSING_MEDIA") != nullptr;

  Frame frame64;
  Frame frame360;
  const bool have64 = load_y4m(path64, &frame64);
  const bool have360 = load_y4m(path360, &frame360);
  if (!have64 || !have360) {
    check(allow_missing, "real-frame media present", __LINE__);
    if (!allow_missing) {
      suite_end();
      return;
    }
  }

  // QP 90: the tx_variants quantiser is fixed at q(90) = 528, and the
  // dequantiser step in coefficient units is 528 / 8 = 66.  Two lambdas:
  //   * the rdo.h placeholder QP->lambda calibration (documented as such),
  //   * lambda = sse_scale * step^2 with the real q(90) step and the default
  //     sse_scale.
  const double lambda_rdo = rdo::lambda_sse_from_qp(90);
  const double lambda_step =
      rdo::lambda_sse_from_quant_step(static_cast<double>(528 / 8));
  std::printf("\nQP 90 lambdas: rdo::lambda_sse_from_qp(90) = %.1f, "
              "lambda_sse_from_quant_step(66) = %.1f\n",
              lambda_rdo, lambda_step);

  if (have64) {
    CHECK_EQ(frame64.width, 64);
    CHECK_EQ(frame64.height, 64);
    Histogram hist;
    CHECK(run_blocks(frame64, 64, 64, 0, 0, lambda_step, &hist,
                     "64x64 frame, one 64x64 block (DCTONLY), DC mode", true));
    CHECK_EQ(hist.blocks, 1);
    CHECK_EQ(hist.selected[0], 1);

    Histogram h16;
    CHECK(run_blocks(frame64, 16, 16, 0, 0, lambda_step, &h16,
                     "64x64 frame, 16x16 blocks, DC mode", true));
    CHECK_EQ(h16.blocks, 16);

    Histogram h8;
    CHECK(run_blocks(frame64, 8, 8, 0, 0, lambda_step, &h8,
                     "64x64 frame, 8x8 blocks, DC mode", true));
    CHECK_EQ(h8.blocks, 64);

    Histogram h4;
    CHECK(run_blocks(frame64, 4, 4, 3, 0, lambda_step, &h4,
                     "64x64 frame, 4x4 blocks, D45 mode", true));
    CHECK_EQ(h4.blocks, 256);

    Histogram hrdo;
    CHECK(run_blocks(frame64, 16, 16, 0, 0, lambda_rdo, &hrdo,
                     "64x64 frame, 16x16 blocks, DC mode, rdo lambda", true));
    CHECK_EQ(hrdo.blocks, 16);
  }

  if (have360) {
    CHECK_EQ(frame360.width, 640);
    CHECK_EQ(frame360.height, 360);
    // 640x360: 40x22 full 16x16 blocks (the bottom 8 rows are edge remainder
    // and would need the M1 edge-partition path; they are out of scope here).
    Histogram h16dc;
    CHECK(run_blocks(frame360, 16, 16, 0, 0, lambda_step, &h16dc,
                     "640x360 frame, 16x16 blocks, DC mode", true));
    CHECK_EQ(h16dc.blocks, 40 * 22);

    Histogram h16d45;
    CHECK(run_blocks(frame360, 16, 16, 3, 0, lambda_step, &h16d45,
                     "640x360 frame, 16x16 blocks, D45 mode", true));
    CHECK_EQ(h16d45.blocks, 40 * 22);

    Histogram h16v;
    CHECK(run_blocks(frame360, 16, 16, 1, 0, lambda_step, &h16v,
                     "640x360 frame, 16x16 blocks, V mode", true));
    CHECK_EQ(h16v.blocks, 40 * 22);

    Histogram h32;
    CHECK(run_blocks(frame360, 32, 32, 0, 0, lambda_step, &h32,
                     "640x360 frame, 32x32 blocks, DC mode", true));
    CHECK_EQ(h32.blocks, 20 * 11);

    Histogram h16rdo;
    CHECK(run_blocks(frame360, 16, 16, 0, 0, lambda_rdo, &h16rdo,
                     "640x360 frame, 16x16 blocks, DC mode, rdo lambda", true));
    CHECK_EQ(h16rdo.blocks, 40 * 22);

    // Long-32 shapes: the only intra sets where IDTX is codable.
    Histogram h1632;
    CHECK(run_blocks(frame360, 16, 32, 0, 0, lambda_step, &h1632,
                     "640x360 frame, 16x32 blocks, DC mode (IDTX legal)",
                     true));
    CHECK_EQ(h1632.blocks, 40 * 11);
    Histogram h3216;
    CHECK(run_blocks(frame360, 32, 16, 0, 0, lambda_step, &h3216,
                     "640x360 frame, 32x16 blocks, DC mode (IDTX legal)",
                     true));
    CHECK_EQ(h3216.blocks, 20 * 22);

    // reduced_tx_set = 1 collapses 16x16 to {DCT_DCT, ADST_ADST}.
    Histogram h16red;
    CHECK(run_blocks(frame360, 16, 16, 0, 1, lambda_step, &h16red,
                     "640x360 frame, 16x16 blocks, DC, reduced=1", true));
    CHECK_EQ(h16red.blocks, 40 * 22);
    for (int t = 0; t < 16; ++t) {
      if (t != 0 && t != 3) {
        CHECK_EQ(h16red.selected[t], 0);
      }
    }
  }
  suite_end();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path data_dir = "tests/data";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--data" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (arg == "--help") {
      std::printf("usage: %s [--data DIR]\n", argv[0]);
      return 0;
    } else {
      std::printf("unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  std::printf("av2enc intra transform-type test\n");
  test_tables();
  test_intra_tx_set();
  test_candidate_sets();
  test_mode_mapping();
  test_scan();
  test_exhaustive_transforms();
  test_aliasing_32();
  test_rate_model();
  test_tie_break();
  test_selection_bruteforce();
  test_real_frames(data_dir);

  std::printf("\n%lld checks, %d failures\n", g_checks, g_failures);
  std::printf("RESULT: %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
