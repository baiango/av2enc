// Structural and cross-check tests for the recovered §9.2 / §9.3 tables.
//
// Provenance is enforced by generate_missing_tables.py --check (PDF SHA-256,
// per-table extents, per-table value digest, and generated-byte freshness).
// This test asserts the dimensions and a sample of values that consumers rely
// on, and compares the recovered arrays against the hand-written local copies
// that are still present in the tree.  A divergence fails the test instead of
// silently preferring one copy.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>

#include "codec/filters/deblock/deblock_core.h"
#include "codec/filters/tables/film_grain_tables.h"
#include "codec/intra/ibp/ibp_tables.h"
#include "codec/inter/advanced/tables/wedge_tables.h"
#include "codec/intra/palette/core/palette_core.h"
#include "spec_tables/missing/missing_tables.h"
#include "spec_tables/missing/missing_tables_provenance.h"
#include "pipeline/tiles/tile_layout.h"

namespace missing = av2enc::tables::missing;

namespace {

int g_failures = 0;

#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!(condition)) {                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                   #condition);                                             \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

// Extents are compile-time facts as well as runtime checks.
static_assert(std::size(missing::kModeToTxfm) == 14u);
static_assert(std::size(missing::kModeToTxfmNames) == 14u);
static_assert(std::size(missing::kPaletteColorContext) == 9u);
static_assert(std::size(missing::kPaletteColorHashMultipliers) == 3u);
static_assert(std::size(missing::kDrIntraDerivative) == 90u);
static_assert(std::size(missing::kSideThresholds) == 296u);
static_assert(std::size(missing::kQFirst) == 9u);
static_assert(std::size(missing::kWMult) == 8u);
static_assert(std::size(missing::kQThreshMults) == 8u);
static_assert(std::size(missing::kWedgeBits) == 29u);
static_assert(std::size(missing::kGaussianSequence) == 2048u);
static_assert(std::size(missing::kTileWidthScalingFactor) == 2u);
static_assert(std::size(missing::kTileWidthScalingFactor[0]) == 31u);
static_assert(std::size(missing::kTileAreaScalingFactor) == 2u);
static_assert(std::size(missing::kTileAreaScalingFactor[0]) == 31u);
static_assert(std::size(missing::kDefaultUseMostProbablePrecisionCdf) == 3u);
static_assert(std::size(missing::kDefaultUseMostProbablePrecisionCdf[0]) == 3u);
static_assert(std::size(missing::kMissingTableProvenance) == 13u);

void TestProvenance() {
  CHECK(missing::kMissingTableCount == 13);
  for (const auto& entry : missing::kMissingTableProvenance) {
    CHECK(entry.spec_name != nullptr);
    CHECK(entry.clause != nullptr);
    CHECK(entry.extents != nullptr);
    CHECK(entry.value_sha256 != nullptr);
    int len = 0;
    while (entry.value_sha256[len] != '\0') ++len;
    CHECK(len == 64);
    for (int i = 0; i < len; ++i) {
      const char c = entry.value_sha256[i];
      CHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }
    CHECK(entry.extents[0] == '[');
  }
  CHECK(std::strcmp(missing::kMissingTableProvenance[0].spec_name,
                    "Mode_To_Txfm") == 0);
  CHECK(std::strcmp(missing::kMissingTableProvenance[0].extents, "[14]") == 0);
  CHECK(std::strcmp(missing::kMissingTableProvenance[10].extents,
                    "[2][31]") == 0);
  CHECK(std::strlen(missing::kSpecPdfSha256) == 64);
  CHECK(missing::kSpecVersion[0] != '\0');
  CHECK(missing::kGeneratorPath[0] != '\0');
}

void TestModeToTxfm() {
  // §9.2 p.612.  Numeric TX_TYPE values (constants.json) and printed names.
  const std::uint8_t expected_values[14] = {0, 1, 2, 0, 3, 1, 2, 2,
                                            1, 3, 1, 2, 3, 0};
  const char* expected_names[14] = {
      "DCT_DCT",  "ADST_DCT", "DCT_ADST",  "DCT_DCT",  "ADST_ADST",
      "ADST_DCT", "DCT_ADST", "DCT_ADST",  "ADST_DCT", "ADST_ADST",
      "ADST_DCT", "DCT_ADST", "ADST_ADST", "DCT_DCT"};
  for (int i = 0; i < 14; ++i) {
    CHECK(missing::kModeToTxfm[i] == expected_values[i]);
    CHECK(std::strcmp(missing::kModeToTxfmNames[i], expected_names[i]) == 0);
  }
}

void TestPalette() {
  // §9.2 p.612 / p.613.
  const std::int8_t expected_context[9] = {-1, -1, 0, -1, -1, 4, 3, 2, 1};
  for (int i = 0; i < 9; ++i) {
    CHECK(missing::kPaletteColorContext[i] == expected_context[i]);
    CHECK(missing::kPaletteColorContext[i] ==
          av2enc::palette::kPaletteColorContext[i]);
  }
  CHECK(missing::kPaletteColorHashMultipliers[0] == 1);
  CHECK(missing::kPaletteColorHashMultipliers[1] == 2);
  CHECK(missing::kPaletteColorHashMultipliers[2] == 2);
}

void TestDrIntraDerivative() {
  constexpr int kIbpDr[90] = {AV2ENC_IBP_DR_INTRA_DERIVATIVE_TABLE};
  static_assert(std::size(kIbpDr) == 90u);
  for (int i = 0; i < 90; ++i) {
    CHECK(missing::kDrIntraDerivative[i] == kIbpDr[i]);
  }
  CHECK(missing::kDrIntraDerivative[0] == 0);
  CHECK(missing::kDrIntraDerivative[1] == 4096);
  CHECK(missing::kDrIntraDerivative[45] == 64);
  CHECK(missing::kDrIntraDerivative[67] == 24);
  CHECK(missing::kDrIntraDerivative[89] == 1);
}

void TestDeblock() {
  // §9.2 p.613-614.
  const int side_samples[] = {0, 55, 56, 57, 58, 100, 150, 200, 295};
  const std::int16_t side_expected[] = {-16, -16, -16, -14, -13,
                                        46,  240, 736, 1678};
  for (std::size_t i = 0; i < std::size(side_samples); ++i) {
    CHECK(missing::kSideThresholds[side_samples[i]] == side_expected[i]);
  }
  for (int i = 0; i < 296; ++i) {
    CHECK(missing::kSideThresholds[i] == av2enc::deblock::kSideThresholds[i]);
  }

  const std::int8_t q_first[9] = {45, 43, 40, 35, 32, 32, 32, 32, 32};
  const std::int8_t w_mult[8] = {85, 51, 37, 28, 23, 20, 17, 15};
  const std::int8_t q_thresh[8] = {32, 25, 19, 19, 18, 18, 17, 17};
  for (int i = 0; i < 9; ++i) {
    CHECK(missing::kQFirst[i] == q_first[i]);
    CHECK(missing::kQFirst[i] == av2enc::deblock::kQFirst[i]);
  }
  for (int i = 0; i < 8; ++i) {
    CHECK(missing::kWMult[i] == w_mult[i]);
    CHECK(missing::kWMult[i] == av2enc::deblock::kWMult[i]);
    CHECK(missing::kQThreshMults[i] == q_thresh[i]);
    CHECK(missing::kQThreshMults[i] ==
          av2enc::deblock::kQThreshMults[i]);
  }
}

void TestWedgeBits() {
  const std::int8_t expected[29] = {0, 0, 0, 4, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 0, 0, 0, 0, 0, 0, 0,
                                    0, 4, 4, 4, 4, 0, 0, 4, 4};
  for (int i = 0; i < 29; ++i) {
    CHECK(missing::kWedgeBits[i] == expected[i]);
    CHECK(missing::kWedgeBits[i] ==
          av2enc::inter::adv::tables::kWedgeBits[i]);
  }
}

void TestGaussianSequence() {
  CHECK(missing::kGaussianSequence[0] == 56);
  CHECK(missing::kGaussianSequence[1] == 568);
  CHECK(missing::kGaussianSequence[1024] == -8);
  CHECK(missing::kGaussianSequence[2047] == -484);
  for (int i = 0; i < 2048; ++i) {
    CHECK(missing::kGaussianSequence[i] >= -2048);
    CHECK(missing::kGaussianSequence[i] <= 2047);
    CHECK((missing::kGaussianSequence[i] % 4) == 0);
    CHECK(missing::kGaussianSequence[i] ==
          av2enc::filters::spec_tables::kGaussianSequence[i]);
  }
}

void TestTileScaling() {
  // §9.2 p.622.  Values for seq_level_idx 0..21; 22..30 are `reserved`.
  for (int tier = 0; tier < 2; ++tier) {
    for (int level = 0; level < 31; ++level) {
      const int width = missing::kTileWidthScalingFactor[tier][level];
      const int area = missing::kTileAreaScalingFactor[tier][level];
      if (level >= 22) {
        CHECK(width == missing::kReservedTileScalingFactor);
        CHECK(area == missing::kReservedTileScalingFactor);
        continue;
      }
      CHECK(width > 0);
      CHECK(area > 0);
      // Cross-check the algorithmic local equivalent (src/encoder/tiles/).
      const av2enc::tiles::LevelTileLimits local =
          av2enc::tiles::level_tile_limits(level, tier);
      CHECK(!local.reserved);
      CHECK(local.width_scaling_factor == width);
      CHECK(local.area_scaling_factor == area);
    }
  }

  // Row boundaries that the local threshold logic encodes.
  CHECK(missing::kTileWidthScalingFactor[0][17] == 4);
  CHECK(missing::kTileWidthScalingFactor[0][18] == 8);
  CHECK(missing::kTileWidthScalingFactor[0][21] == 8);
  CHECK(missing::kTileWidthScalingFactor[0][22] == -1);
  CHECK(missing::kTileWidthScalingFactor[1][13] == 4);
  CHECK(missing::kTileWidthScalingFactor[1][14] == 8);
  CHECK(missing::kTileWidthScalingFactor[1][18] == 16);
  CHECK(missing::kTileAreaScalingFactor[0][13] == 4);
  CHECK(missing::kTileAreaScalingFactor[0][14] == 8);
  CHECK(missing::kTileAreaScalingFactor[0][18] == 16);
  CHECK(missing::kTileAreaScalingFactor[1][13] == 4);
  CHECK(missing::kTileAreaScalingFactor[1][14] == 16);
  CHECK(missing::kTileAreaScalingFactor[1][18] == 32);
}

void TestPrecisionCdf() {
  // §9.3 p.641: three rows of N+1 = 3 entries.
  const std::uint16_t expected[3][3] = {
      {20650, 31, 0},
      {15758, 31, 0},
      {4571, 31, 0},
  };
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      CHECK(missing::kDefaultUseMostProbablePrecisionCdf[row][col] ==
            expected[row][col]);
    }
    CHECK(missing::kDefaultUseMostProbablePrecisionCdf[row][1] == 31);
    CHECK(missing::kDefaultUseMostProbablePrecisionCdf[row][2] == 0);
  }
}

}  // namespace

int main() {
  TestProvenance();
  TestModeToTxfm();
  TestPalette();
  TestDrIntraDerivative();
  TestDeblock();
  TestWedgeBits();
  TestGaussianSequence();
  TestTileScaling();
  TestPrecisionCdf();
  if (g_failures != 0) {
    std::fprintf(stderr, "missing_tables_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("missing_tables_test: ok\n");
  return 0;
}
