// CPU correctness tests for the advanced inter components.
//
// Coverage is in two layers:
//   1. Spec tables and closed forms (warp filter rows, wedge LUTs, Div_Lut).
//   2. Randomised cross-checks against literal, independent transcriptions of
//      the spec pseudocode (wedge mask table generation, compound blending,
//      BAWP, warp estimation, reference-list selection).
//
// Build with build_and_test.sh. The CUDA counterpart (advanced_test.cu) checks
// CPU vs GPU bit-exactness on the same job types.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "b_frame.h"
#include "bawp.h"
#include "compound.h"
#include "obmc.h"
#include "warp.h"

namespace {

using namespace av2enc::inter::adv;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)
#define EXPECT_EQ(a, b)                                                                  \
  do {                                                                                   \
    const long long va = static_cast<long long>(a);                                      \
    const long long vb = static_cast<long long>(b);                                      \
    ++g_checks;                                                                          \
    if (va != vb) {                                                                      \
      ++g_failures;                                                                      \
      if (g_failures <= 30) {                                                            \
        std::printf("FAIL line %d: %s (%lld) != %s (%lld)\n", __LINE__, #a, va, #b, vb); \
      }                                                                                  \
    }                                                                                    \
  } while (0)

std::uint32_t g_rng = 0x12345678u;

std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

int random_range(int low, int high) {
  return low + static_cast<int>(next_random() % static_cast<std::uint32_t>(high - low + 1));
}

// ---------------------------------------------------------------------------
// §4.8 / §7.13.3.22 helpers
// ---------------------------------------------------------------------------

void test_math_helpers() {
  for (int v = -5000; v <= 5000; v += 7) {
    for (int n = 0; n <= 12; n += 3) {
      const int expected =
          n == 0 ? v
                 : static_cast<int>(std::floor((v + (1 << (n - 1))) / static_cast<double>(1 << n)));
      EXPECT_EQ(round2(v, n), expected);
      EXPECT_EQ(round2_signed(v, n), v >= 0 ? round2(v, n) : -round2(-v, n));
    }
  }
  // Div_Lut is defined as lround(65536 / (128 + f)).
  for (int f = 0; f < kDivLutNum; ++f) {
    const int den = 128 + f;
    const int expected = static_cast<int>((65536 + den / 2) / den);
    EXPECT_EQ(div_lut_entry(f), expected);
  }
  // resolve_division() is an approximation clamped to (2 << shift) - 1.
  EXPECT_EQ(resolve_division(1 << 20, 1, 8), 511);
  for (int trial = 0; trial < 4000; ++trial) {
    const int n = random_range(1, 1 << 20);
    const int d = random_range(1, 1 << 16);
    const int exact = (n << 8) / d;
    if (exact > 480) {
      continue;
    }
    const int got = resolve_division(n, d, 8);
    // Measured worst case is < 1% of the quotient for quotients <= 480.
    EXPECT(std::abs(got - exact) <= 2 + exact / 64);
  }
  EXPECT_EQ(clip3(-3, 7, 10), 7);
  EXPECT_EQ(clip1(300, 8), 255);
  EXPECT_EQ(floor_log2(1024), 10);
}

// ---------------------------------------------------------------------------
// §9.5 warp filter tables
// ---------------------------------------------------------------------------

void test_warp_tables() {
  EXPECT_EQ(tables::kWarpedFilterRows, 449);
  EXPECT_EQ(tables::kExtWarpedFilterRows, 65);
  for (int r = 0; r < tables::kWarpedFilterRows; ++r) {
    int sum = 0;
    for (int t = 0; t < 8; ++t) {
      sum += tables::kWarpedFilters[r][t];
    }
    EXPECT_EQ(sum, 128);
  }
  for (int r = 0; r < tables::kExtWarpedFilterRows; ++r) {
    int sum = 0;
    for (int t = 0; t < 6; ++t) {
      sum += tables::kExtWarpedFilters[r][t];
    }
    EXPECT_EQ(sum, 128);
  }
  // The zero-phase rows are the identity interpolators.
  for (int t = 0; t < 8; ++t) {
    EXPECT_EQ(tables::kWarpedFilters[192][t], t == 3 ? 127 : (t == 4 ? 1 : 0));
  }
  for (int t = 0; t < 6; ++t) {
    EXPECT_EQ(tables::kExtWarpedFilters[64][t], t == 3 ? 128 : 0);
  }
}

// ---------------------------------------------------------------------------
// Wedge tables and masks
// ---------------------------------------------------------------------------

void test_wedge_tables() {
  const std::int8_t cos_expected[20] = {4,  4,  4,  2,  2,  0, -2, -2, -4, -4,
                                        -4, -4, -4, -2, -2, 0, 2,  2,  4,  4};
  const std::int8_t sin_expected[20] = {0, -1, -2, -2, -4, -4, -4, -2, -2, -1,
                                        0, 1,  2,  2,  4,  4,  4,  2,  2,  1};
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(tables::kWedgeCosLut[i], cos_expected[i]);
    EXPECT_EQ(tables::kWedgeSinLut[i], sin_expected[i]);
  }
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(tables::kPosDist2BldWeight[i] + tables::kNegDist2BldWeight[i], 16);
  }
  const std::int8_t wedge_bits_expected[29] = {0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0,
                                               0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 4, 4};
  for (int b = 0; b < 29; ++b) {
    EXPECT_EQ(tables::kWedgeBits[b], wedge_bits_expected[b]);
  }
  // Codebook: 68 entries, each maps to a signalable angle+offset.
  int seen = 0;
  for (int w = 0; w < kWedgeTypes; ++w) {
    const int angle = tables::kWedgeCodebook[w][0];
    EXPECT(angle >= 0 && angle < kWedgeAngles);
    EXPECT(tables::kWedgeCodebook[w][1] >= 1 && tables::kWedgeCodebook[w][1] <= 7);
    EXPECT(tables::kWedgeCodebook[w][2] >= 1 && tables::kWedgeCodebook[w][2] <= 7);
    ++seen;
  }
  EXPECT_EQ(seen, 68);
  // Every WedgeIndex 0..67 is reachable from (wedge_angle, wedge_dist).
  std::vector<bool> hit(68, false);
  for (int angle = 0; angle < kWedgeAngles; ++angle) {
    for (int dist = 0; dist < kNumWedgeDist; ++dist) {
      const int index = wedge_index_from_angle_dist(angle, dist);
      if (index >= 0) {
        EXPECT(index < 68);
        hit[static_cast<std::size_t>(index)] = true;
      }
    }
  }
  for (int i = 0; i < 68; ++i) {
    EXPECT(hit[static_cast<std::size_t>(i)]);
    EXPECT(wedge_index_signalable(i));
  }
  EXPECT(!wedge_index_signalable(68));
  EXPECT_EQ(wedge_index_from_angle_dist(0, 0), -1);
  EXPECT_EQ(wedge_index_from_angle_dist(19, 3), 67);

  // II weights: non-increasing, first 60, last 1.
  EXPECT_EQ(tables::kIiWeights1d[0], 60);
  EXPECT_EQ(tables::kIiWeights1d[127], 1);
  for (int i = 1; i < 128; ++i) {
    EXPECT(tables::kIiWeights1d[i] <= tables::kIiWeights1d[i - 1]);
  }
  EXPECT_EQ(intra_mode_mask_value(InterIntraMode::kDc, 8, 8, 3, 5), 32);
  EXPECT_EQ(intra_mode_mask_value(InterIntraMode::kVertical, 8, 8, 2, 5),
            tables::kIiWeights1d[2 * 16]);
  EXPECT_EQ(intra_mode_mask_value(InterIntraMode::kHorizontal, 8, 8, 2, 5),
            tables::kIiWeights1d[5 * 16]);
  EXPECT_EQ(intra_mode_mask_value(InterIntraMode::kSmooth, 8, 8, 5, 2),
            tables::kIiWeights1d[2 * 16]);
}

// Literal transcription of initialise_wedge_mask_table() (§7.13.3.27).
struct LiteralWedgeTables {
  static constexpr int kMaster = 128;
  std::vector<int> master;  // [2][20][128][128]
  LiteralWedgeTables() {
    master.assign(2 * 20 * kMaster * kMaster, 0);
    for (int boundary = 0; boundary < 2; ++boundary) {
      for (int angle = 0; angle < 20; ++angle) {
        for (int n = 0; n < kMaster; ++n) {
          const int y = ((n << 1) - kMaster + 1) * tables::kWedgeSinLut[angle];
          for (int m = 0; m < kMaster; ++m) {
            int d = ((m << 1) - kMaster + 1) * tables::kWedgeCosLut[angle] + y;
            if (boundary == 0) {
              d = d * 2;
            }
            const int clamp_d = clip3(-31, 31, d);
            const int v = clamp_d >= 0 ? tables::kPosDist2BldWeight[clamp_d]
                                       : tables::kNegDist2BldWeight[-clamp_d];
            master[((boundary * 20 + angle) * kMaster + n) * kMaster + m] = v << 2;
          }
        }
      }
    }
  }
  int at(int boundary, int angle, int n, int m) const {
    return master[((boundary * 20 + angle) * kMaster + n) * kMaster + m];
  }
};

void test_wedge_masks() {
  LiteralWedgeTables literal;
  const int sizes[] = {tables::kBlock8x8,   tables::kBlock16x16, 9 /*32x32*/,
                       tables::kBlock64x64, 21 /*8x32*/,         22 /*32x8*/};
  for (int bsize : sizes) {
    const int w = tables::kWedgeBlockWidth[bsize];
    const int h = tables::kWedgeBlockHeight[bsize];
    const int boundary = wedge_boundary_for_bsize(bsize);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(w) * h);
    for (int index = 0; index < kWedgeTypes; ++index) {
      for (int sign = 0; sign < 2; ++sign) {
        build_wedge_mask(bsize, sign, index, mask.data(), w);
        const int dir = tables::kWedgeCodebook[index][0];
        const int xoff = 128 / 2 - ((tables::kWedgeCodebook[index][1] * w) >> 3);
        const int yoff = 128 / 2 - ((tables::kWedgeCodebook[index][2] * h) >> 3);
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; ++j) {
            const int master = literal.at(boundary, dir, yoff + i, xoff + j);
            const int expected = sign ? (64 - master) : master;
            EXPECT_EQ(mask[static_cast<std::size_t>(i) * w + j], expected);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Compound prediction
// ---------------------------------------------------------------------------

struct PredBuffers {
  int w = 0;
  int h = 0;
  std::vector<std::int16_t> p0;
  std::vector<std::int16_t> p1;
  std::vector<std::uint8_t> dst;
  std::vector<std::uint8_t> mask;
  int stride = 0;
};

PredBuffers make_preds(int w, int h, int base0, int base1, int noise) {
  PredBuffers b;
  b.w = w;
  b.h = h;
  b.stride = w;
  b.p0.resize(static_cast<std::size_t>(w) * h);
  b.p1.resize(static_cast<std::size_t>(w) * h);
  b.dst.assign(static_cast<std::size_t>(w) * h, 0);
  b.mask.assign(static_cast<std::size_t>(w) * h, 0);
  for (int i = 0; i < w * h; ++i) {
    const int n = noise ? random_range(-noise, noise) : 0;
    b.p0[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(base0 + n);
    b.p1[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(base1 - n);
  }
  return b;
}

// Literal per-sample compound blend from §7.13.3.14/.30.
int literal_compound_sample(CompoundType type, int cw, bool imp_mask_eligible, int imp_m,
                            int inter_intra, int intra_pred, int inter_pred, int m, int p0,
                            int p1) {
  const int inter_post_round = 4;
  if (inter_intra) {
    return round2(m * intra_pred + (64 - m) * inter_pred, 6);
  }
  if (type == CompoundType::kAverage) {
    if (imp_mask_eligible) {
      return clip1(round2(imp_m * p0 + (2 - imp_m) * p1, 1 + inter_post_round), 8);
    }
    return clip1(round2(cw * p0 + (16 - cw) * p1, 4 + inter_post_round), 8);
  }
  return clip1(round2(m * p0 + (64 - m) * p1, 6 + inter_post_round), 8);
}

void test_compound_blend() {
  const int sizes[][2] = {{8, 8}, {16, 16}, {32, 32}, {64, 64}, {8, 32}, {32, 8}, {4, 4}};
  for (const auto& size : sizes) {
    const int w = size[0];
    const int h = size[1];
    const int bsize = w == h && w == 8    ? tables::kBlock8x8
                      : w == h && w == 16 ? tables::kBlock16x16
                      : w == h && w == 32 ? 9
                      : w == h && w == 64 ? tables::kBlock64x64
                      : w == 8            ? 21
                                          : 22;
    PredBuffers b = make_preds(w, h, 128 * 16, 96 * 16, 40 * 16);

    // Average with CWP.
    for (int cw : {8, 12, 4, 10, 6, 20, -4}) {
      CompoundJob job{};
      job.rect = {0, 0, w, h};
      job.bsize = bsize;
      job.p0 = b.p0.data();
      job.p0_stride = b.stride;
      job.p1 = b.p1.data();
      job.p1_stride = b.stride;
      job.dst = b.dst.data();
      job.dst_stride = b.stride;
      job.type = CompoundType::kAverage;
      job.cwp_weight = cw;
      run_compound_job(job);
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          const int idx = i * w + j;
          EXPECT_EQ(b.dst[static_cast<std::size_t>(idx)],
                    literal_compound_sample(CompoundType::kAverage, cw, false, 1, 0, 0, 0, 0,
                                            b.p0[idx], b.p1[idx]));
        }
      }
    }

    // Implicit mask.
    if (bsize_has_wedge(bsize)) {
      CompoundJob job{};
      job.rect = {0, 0, w, h};
      job.bsize = bsize;
      job.p0 = b.p0.data();
      job.p0_stride = b.stride;
      job.p1 = b.p1.data();
      job.p1_stride = b.stride;
      job.dst = b.dst.data();
      job.dst_stride = b.stride;
      job.type = CompoundType::kAverage;
      job.imp_mask_eligible = 1;
      job.mi_cols = w;
      job.mi_rows = h;
      job.ref_start_x[0] = -2;
      job.ref_start_y[0] = 0;
      job.ref_start_x[1] = w - 3;
      job.ref_start_y[1] = 0;
      run_compound_job(job);
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          const int idx = i * w + j;
          const int m = implicit_mask_value(job, i, j);
          EXPECT_EQ(b.dst[static_cast<std::size_t>(idx)],
                    literal_compound_sample(CompoundType::kAverage, 8, true, m, 0, 0, 0, 0,
                                            b.p0[idx], b.p1[idx]));
        }
      }
    }

    // Wedge: compare through the luma mask.
    if (bsize_has_wedge(bsize)) {
      for (int index : {0, 17, 33, 50, 67}) {
        for (int sign = 0; sign < 2; ++sign) {
          CompoundJob job{};
          job.rect = {0, 0, w, h};
          job.bsize = bsize;
          job.p0 = b.p0.data();
          job.p0_stride = b.stride;
          job.p1 = b.p1.data();
          job.p1_stride = b.stride;
          job.dst = b.dst.data();
          job.dst_stride = b.stride;
          job.mask_out = b.mask.data();
          job.mask_out_stride = w;
          job.type = CompoundType::kWedge;
          job.wedge_index = index;
          job.wedge_sign = sign;
          run_compound_job(job);
          for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
              const int idx = i * w + j;
              const int m = wedge_mask_value(bsize, sign, index, i, j);
              EXPECT_EQ(b.mask[static_cast<std::size_t>(idx)], m);
              EXPECT_EQ(b.dst[static_cast<std::size_t>(idx)],
                        literal_compound_sample(CompoundType::kWedge, 8, false, 0, 0, 0, 0, m,
                                                b.p0[idx], b.p1[idx]));
            }
          }
        }
      }
    }

    // Difference-weighted: both mask types.
    for (int mask_type = 0; mask_type < 2; ++mask_type) {
      CompoundJob job{};
      job.rect = {0, 0, w, h};
      job.bsize = bsize;
      job.p0 = b.p0.data();
      job.p0_stride = b.stride;
      job.p1 = b.p1.data();
      job.p1_stride = b.stride;
      job.dst = b.dst.data();
      job.dst_stride = b.stride;
      job.type = CompoundType::kDiffWtd;
      job.mask_type = mask_type;
      run_compound_job(job);
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          const int idx = i * w + j;
          const int m = diffwtd_mask_value(b.p0[idx], b.p1[idx], mask_type, 4, 8);
          EXPECT_EQ(b.dst[static_cast<std::size_t>(idx)],
                    literal_compound_sample(CompoundType::kDiffWtd, 8, false, 0, 0, 0, 0, m,
                                            b.p0[idx], b.p1[idx]));
        }
      }
    }
  }
}

void test_compound_chroma_mask() {
  // 16x16 luma wedge mask, 8x8 chroma plane, 4:2:0 subsampling.
  const int bsize = tables::kBlock16x16;
  PredBuffers luma = make_preds(16, 16, 100 * 16, 140 * 16, 10 * 16);
  PredBuffers chroma0 = make_preds(8, 8, 90 * 16, 150 * 16, 8 * 16);
  PredBuffers chroma1 = make_preds(8, 8, 80 * 16, 160 * 16, 8 * 16);
  std::vector<std::uint8_t> mask(16 * 16);
  build_wedge_mask(bsize, 1, 42, mask.data(), 16);

  CompoundJob job{};
  job.plane = 0;
  job.rect = {0, 0, 16, 16};
  job.bsize = bsize;
  job.p0 = luma.p0.data();
  job.p0_stride = 16;
  job.p1 = luma.p1.data();
  job.p1_stride = 16;
  job.dst = luma.dst.data();
  job.dst_stride = 16;
  job.mask_out = mask.data();
  job.mask_out_stride = 16;
  job.type = CompoundType::kWedge;
  job.wedge_index = 42;
  job.wedge_sign = 1;
  run_compound_job(job);

  CompoundJob cjob = job;
  cjob.plane = 1;
  cjob.sub_x = 1;
  cjob.sub_y = 1;
  cjob.rect = {0, 0, 8, 8};
  cjob.p0 = chroma0.p0.data();
  cjob.p0_stride = 8;
  cjob.p1 = chroma1.p1.data();
  cjob.p1_stride = 8;
  cjob.dst = chroma0.dst.data();
  cjob.dst_stride = 8;
  cjob.mask_out = nullptr;
  cjob.luma_mask = mask.data();
  cjob.luma_mask_stride = 16;
  run_compound_job(cjob);

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const int m = round2(mask[(2 * i) * 16 + 2 * j] + mask[(2 * i) * 16 + 2 * j + 1] +
                               mask[(2 * i + 1) * 16 + 2 * j] + mask[(2 * i + 1) * 16 + 2 * j + 1],
                           2);
      const int idx = i * 8 + j;
      EXPECT_EQ(chroma0.dst[static_cast<std::size_t>(idx)],
                literal_compound_sample(CompoundType::kWedge, 8, false, 0, 0, 0, 0, m,
                                        chroma0.p0[idx], chroma1.p1[idx]));
    }
  }
}

void test_inter_intra() {
  const int w = 16;
  const int h = 16;
  PredBuffers inter = make_preds(w, h, 120, 0, 30);
  std::vector<std::uint8_t> intra(static_cast<std::size_t>(w) * h);
  for (int i = 0; i < w * h; ++i) {
    intra[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  for (int mode = 0; mode < kInterIntraModes; ++mode) {
    CompoundJob job{};
    job.rect = {0, 0, w, h};
    job.bsize = tables::kBlock16x16;
    job.is_inter_intra = 1;
    job.interintra_mode = static_cast<InterIntraMode>(mode);
    job.type = CompoundType::kIntra;
    job.intra = intra.data();
    job.intra_stride = w;
    job.dst = inter.dst.data();
    job.dst_stride = w;
    for (int i = 0; i < w * h; ++i) {
      inter.dst[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(inter.p0[static_cast<std::size_t>(i)] >> 4);
    }
    run_compound_job(job);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = i * w + j;
        const int m = intra_mode_mask_value(job.interintra_mode, w, h, i, j);
        EXPECT_EQ(inter.dst[static_cast<std::size_t>(idx)],
                  literal_compound_sample(CompoundType::kIntra, 8, false, 0, 1,
                                          intra[static_cast<std::size_t>(idx)],
                                          inter.p0[static_cast<std::size_t>(idx)] >> 4, m, 0, 0));
      }
    }
  }

  // Wedge inter-intra uses the wedge mask with intra prediction as pred0.
  CompoundJob job{};
  job.rect = {0, 0, w, h};
  job.bsize = tables::kBlock16x16;
  job.is_inter_intra = 1;
  job.interintra_mode = InterIntraMode::kDc;
  job.type = CompoundType::kWedge;
  job.wedge_index = 7;
  job.wedge_sign = 0;
  job.intra = intra.data();
  job.intra_stride = w;
  job.dst = inter.dst.data();
  job.dst_stride = w;
  for (int i = 0; i < w * h; ++i) {
    inter.dst[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(inter.p0[static_cast<std::size_t>(i)] >> 4);
  }
  run_compound_job(job);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int idx = i * w + j;
      const int m = wedge_mask_value(tables::kBlock16x16, 0, 7, i, j);
      EXPECT_EQ(inter.dst[static_cast<std::size_t>(idx)],
                literal_compound_sample(CompoundType::kIntra, 8, false, 0, 1,
                                        intra[static_cast<std::size_t>(idx)],
                                        inter.p0[static_cast<std::size_t>(idx)] >> 4, m, 0, 0));
    }
  }
}

void test_compound_search() {
  const int w = 16;
  const int h = 16;
  const int bsize = tables::kBlock16x16;
  PredBuffers preds = make_preds(w, h, 120 * 16, 100 * 16, 20 * 16);
  std::vector<std::uint8_t> source(static_cast<std::size_t>(w) * h);
  for (int i = 0; i < w * h; ++i) {
    source[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(preds.p0[static_cast<std::size_t>(i)] >> 4);
  }
  std::vector<std::uint8_t> scratch(static_cast<std::size_t>(w) * h);

  CompoundSearchInput in{};
  in.bsize = bsize;
  in.rect = {0, 0, w, h};
  in.p0 = preds.p0.data();
  in.p0_stride = w;
  in.p1 = preds.p1.data();
  in.p1_stride = w;
  in.source = source.data();
  in.source_stride = w;
  in.scratch = scratch.data();
  in.scratch_stride = w;

  auto decision = search_compound(in);
  EXPECT(decision.has_value());
  if (!decision) {
    return;
  }
  // Because source == round(p0), the best candidate must beat (or tie) plain
  // average, and applying the decision must reproduce its reported SSE.
  CompoundJob job{};
  job.rect = in.rect;
  job.bsize = bsize;
  job.p0 = in.p0;
  job.p0_stride = w;
  job.p1 = in.p1;
  job.p1_stride = w;
  job.dst = scratch.data();
  job.dst_stride = w;
  job.type = decision->type;
  job.cwp_weight = decision->cwp_weight;
  job.wedge_index = decision->wedge_index;
  job.wedge_sign = decision->wedge_sign;
  job.mask_type = decision->mask_type;
  job.imp_mask_eligible = decision->use_imp_mask ? 1 : 0;
  run_compound_job(job);
  const std::int64_t sse = block_sse(scratch.data(), w, 0, 0, source.data(), w, 0, 0, w, h);
  EXPECT_EQ(sse, decision->sse);
  // The search must never return worse than the plain CWP_EQUAL average.
  CompoundJob avg{};
  avg.rect = in.rect;
  avg.bsize = bsize;
  avg.p0 = in.p0;
  avg.p0_stride = w;
  avg.p1 = in.p1;
  avg.p1_stride = w;
  avg.dst = scratch.data();
  avg.dst_stride = w;
  avg.type = CompoundType::kAverage;
  avg.cwp_weight = kCwpEqual;
  run_compound_job(avg);
  const std::int64_t average_sse = block_sse(scratch.data(), w, 0, 0, source.data(), w, 0, 0, w, h);
  EXPECT(decision->sse <= average_sse);

  // Inter-intra search: make the source exactly the II_V_PRED blend, so V must
  // win with zero SSE when no wedge is enabled.
  std::vector<std::uint8_t> modes[kInterIntraModes];
  for (auto& m : modes) {
    m.resize(static_cast<std::size_t>(w) * h);
  }
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int idx = i * w + j;
      for (int m = 0; m < kInterIntraModes; ++m) {
        modes[m][static_cast<std::size_t>(idx)] = static_cast<std::uint8_t>(
            intra_mode_mask_value(static_cast<InterIntraMode>(m), w, h, i, j) + idx % 7);
      }
    }
  }
  std::vector<std::uint8_t> inter_pred(static_cast<std::size_t>(w) * h, 200);
  InterIntraSearchInput ii{};
  ii.bsize = bsize;
  ii.rect = {0, 0, w, h};
  for (int m = 0; m < kInterIntraModes; ++m) {
    ii.intra[m] = modes[m].data();
  }
  ii.intra_stride = w;
  ii.inter_pred = inter_pred.data();
  ii.inter_pred_stride = w;
  ii.scratch = scratch.data();
  ii.scratch_stride = w;
  ii.enable_wedge_interintra = false;
  std::vector<std::uint8_t> target(static_cast<std::size_t>(w) * h, 0);
  {
    std::vector<std::uint8_t> save_scratch(scratch.begin(), scratch.end());
    InterIntraSearchInput tmp = ii;
    tmp.source = target.data();
    EXPECT(apply_inter_intra(tmp, InterIntraMode::kVertical, false, 0));
    std::copy(scratch.begin(), scratch.end(), target.begin());
    std::copy(save_scratch.begin(), save_scratch.end(), scratch.begin());
  }
  ii.source = target.data();
  ii.source_stride = w;
  auto ii_decision = search_inter_intra(ii);
  EXPECT(ii_decision.has_value());
  if (ii_decision) {
    EXPECT(ii_decision->mode == InterIntraMode::kVertical);
    EXPECT_EQ(ii_decision->sse, 0);
  }
}

// ---------------------------------------------------------------------------
// Warp
// ---------------------------------------------------------------------------

struct Plane {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> data;
  Plane(int width, int height) : w(width), h(height) {
    data.resize(static_cast<std::size_t>(w) * h);
  }
  PlaneView view() const {
    PlaneView v{};
    v.data = data.data();
    v.stride = w;
    v.max_x = w - 1;
    v.max_y = h - 1;
    return v;
  }
  MutablePlaneView mut() {
    MutablePlaneView v{};
    v.data = data.data();
    v.stride = w;
    v.max_x = w - 1;
    v.max_y = h - 1;
    return v;
  }
  std::uint8_t at(int x, int y) const { return data[static_cast<std::size_t>(y) * w + x]; }
};

WarpBlockGeometry make_geometry(int w, int h) {
  WarpBlockGeometry g{};
  g.rect = {0, 0, w, h};
  g.sub_x = 0;
  g.sub_y = 0;
  g.mi_col = 0;
  g.mi_row = 0;
  g.bw4 = w / 4;
  g.bh4 = h / 4;
  g.mi_cols = 64;
  g.mi_rows = 64;
  g.ref_w = 256;
  g.ref_h = 256;
  g.frame_w = 256;
  g.frame_h = 256;
  return g;
}

void test_warp_setup_shear() {
  WarpParams identity{};
  const ShearParams shear = setup_shear(identity);
  EXPECT(shear.valid);
  EXPECT_EQ(shear.alpha, 0);
  EXPECT_EQ(shear.beta, 0);
  EXPECT_EQ(shear.gamma, 0);
  EXPECT_EQ(shear.delta, 0);

  // A pure translation has no shear.
  WarpParams translation = identity;
  translation.p[0] = 5 << 16;
  translation.p[1] = -3 << 16;
  const ShearParams tshear = setup_shear(translation);
  EXPECT(tshear.valid);
  EXPECT_EQ(tshear.alpha, 0);
  EXPECT_EQ(tshear.delta, 0);

  // Extreme shear must be rejected (4*|alpha| + 7*|beta| >= 3<<16). p[2] clips,
  // so drive the failure through beta (p[3]).
  WarpParams extreme = identity;
  extreme.p[3] = 2 << 16;
  EXPECT(!setup_shear(extreme).valid);

  // reduce_warp_model keeps the identity.
  const WarpParams reduced = reduce_warp_model(identity);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(reduced.p[i], identity.p[i]);
  }
}

// ---------------------------------------------------------------------------
// Independent literal transcriptions of the warp processes, used only by the
// tests. They deliberately re-derive setup_shear and the two loops from the
// spec text instead of calling warp_common.h.
// ---------------------------------------------------------------------------

struct LiteralShear {
  bool valid = false;
  int alpha = 0;
  int beta = 0;
  int gamma = 0;
  int delta = 0;
};

LiteralShear literal_setup_shear(const WarpParams& params) {
  const int max_value = 32767 - (1 << 5);
  const int one = 1 << 16;
  auto clamp32 = [](long long v) {
    return static_cast<int>(v > 2147483647LL ? 2147483647LL
                                             : (v < -2147483648LL ? -2147483648LL : v));
  };
  const int alpha0 = clip3(-32768, max_value, clamp32((long long)params.p[2] - one));
  const int beta0 = clip3(-32768, max_value, params.p[3]);
  const DivScaleShift ds = resolve_divisor(params.p[2]);
  const long long v = (long long)params.p[4] << 16;
  const int gamma0 = clip3(-32768, max_value, clamp32(round2_signed64(v * ds.factor, ds.shift)));
  const long long w = (long long)params.p[3] * params.p[4];
  const int delta0 =
      clip3(-32768, max_value,
            clamp32((long long)params.p[5] - round2_signed64(w * ds.factor, ds.shift) - one));
  LiteralShear out{};
  out.alpha = round2_signed(alpha0, 6) << 6;
  out.beta = round2_signed(beta0, 6) << 6;
  out.gamma = round2_signed(gamma0, 6) << 6;
  out.delta = round2_signed(delta0, 6) << 6;
  out.valid = !(4 * abs_int(out.alpha) + 7 * abs_int(out.beta) >= (3 << 16)) &&
              !(4 * abs_int(out.gamma) + 4 * abs_int(out.delta) >= (3 << 16));
  return out;
}

void literal_block_warp(const WarpParams& params, const PlaneView& ref, int x, int y, int i8,
                        int j8, int block_w, int block_h, int sub_x, int sub_y, int inter_round1,
                        bool clip, int out_x0, int out_y0, std::int16_t* dst, int dst_stride) {
  const LiteralShear sh = literal_setup_shear(params);
  if (!sh.valid) {
    return;
  }
  const int src_x = (x + j8 * 8 + 4) << sub_x;
  const int src_y = (y + i8 * 8 + 4) << sub_y;
  const int dst_x = params.p[2] * src_x + params.p[3] * src_y + params.p[0];
  const int dst_y = params.p[4] * src_x + params.p[5] * src_y + params.p[1];
  const int x4 = dst_x >> sub_x;
  const int y4 = dst_y >> sub_y;
  const int ix4 = x4 >> 16;
  const int sx4 = x4 & 0xffff;
  const int iy4 = y4 >> 16;
  const int sy4 = y4 & 0xffff;
  int intermediate[15][8];
  for (int i1 = -7; i1 < 8; ++i1) {
    for (int i2 = -4; i2 < 4; ++i2) {
      const int sx = sx4 + sh.alpha * i2 + sh.beta * i1;
      const int offs = round2(sx, 10) + 3 * 64;
      int s = 0;
      for (int i3 = 0; i3 < 8; ++i3) {
        s +=
            tables::kWarpedFilters[offs][i3] * ref.data[clip3(0, ref.max_y, iy4 + i1) * ref.stride +
                                                        clip3(0, ref.max_x, ix4 + i2 - 3 + i3)];
      }
      intermediate[i1 + 7][i2 + 4] = round2(s, 3);
    }
  }
  for (int i1 = -4; i1 < 4; ++i1) {
    for (int i2 = -4; i2 < 4; ++i2) {
      const int sy = sy4 + sh.gamma * i2 + sh.delta * i1;
      const int offs = round2(sy, 10) + 3 * 64;
      int s = 0;
      for (int i3 = 0; i3 < 8; ++i3) {
        s += tables::kWarpedFilters[offs][i3] * intermediate[i1 + i3 + 4][i2 + 4];
      }
      const int local_y = i8 * 8 + i1 + 4;
      const int local_x = j8 * 8 + i2 + 4;
      if (local_y < 0 || local_y >= block_h || local_x < 0 || local_x >= block_w) {
        continue;
      }
      const int value = round2(s, inter_round1);
      dst[(out_y0 + local_y) * dst_stride + out_x0 + local_x] =
          static_cast<std::int16_t>(clip ? clip1(value, 8) : value);
    }
  }
}

void literal_warp_block(const WarpParams& params, const PlaneView& ref, int x, int y, int w, int h,
                        int sub_x, int sub_y, int inter_round1, bool clip, int out_x0, int out_y0,
                        std::int16_t* dst, int dst_stride) {
  const LiteralShear sh = literal_setup_shear(params);
  if (!sh.valid) {
    return;
  }
  for (int i1 = 0; i1 < h; ++i1) {
    for (int i2 = 0; i2 < w; ++i2) {
      dst[(out_y0 + i1) * dst_stride + out_x0 + i2] = 0;
    }
  }
  for (int i8 = 0; i8 <= (h - 1) >> 3; ++i8) {
    for (int j8 = 0; j8 <= (w - 1) >> 3; ++j8) {
      literal_block_warp(params, ref, x, y, i8, j8, w, h, sub_x, sub_y, inter_round1, clip, out_x0,
                         out_y0, dst, dst_stride);
    }
  }
}

// §7.13.3.20, unscaled branch only (the tests use same-size references).
void literal_ext_warp(const WarpParams& params, const PlaneView& ref, int x, int y, int w, int h,
                      int inter_round1, bool clip, int out_x0, int out_y0, std::int16_t* dst,
                      int dst_stride) {
  for (int i1 = 0; i1 < h; ++i1) {
    for (int i2 = 0; i2 < w; ++i2) {
      dst[(out_y0 + i1) * dst_stride + out_x0 + i2] = 0;
    }
  }
  for (int i4 = 0; i4 < (h >> 2); ++i4) {
    for (int j4 = 0; j4 < (w >> 2); ++j4) {
      const int i8 = i4 >> 1;
      const int j8 = j4 >> 1;
      const int bbox_w = w < 8 ? w : 8;
      const int bbox_h = h < 8 ? h : 8;
      int mv[2] = {0, 0};
      get_sub_block_warp_mv(params, 0, 0, 0, x + j8 * 8, y + i8 * 8, bbox_w, bbox_h, 0, mv);
      mv[0] = clip3(kMvLow + 1, kMvUpp - 1, mv[0]);
      mv[1] = clip3(kMvLow + 1, kMvUpp - 1, mv[1]);
      const MvScaleOutput scale =
          motion_vector_scaling(0, 0, 0, x + j8 * 8, y + i8 * 8, mv, false, ref.max_x + 1,
                                ref.max_y + 1, ref.max_x + 1, ref.max_y + 1);
      int first_x = clip3(0, ref.max_x, (scale.start_x >> 10) - 3);
      int first_y = clip3(0, ref.max_y, (scale.start_y >> 10) - 3);
      const int bbox_last_x =
          clip3(0, ref.max_x, ((scale.start_x + scale.step_x * (bbox_w - 1)) >> 10) + 4);
      const int bbox_last_y =
          clip3(0, ref.max_y, ((scale.start_y + scale.step_y * (bbox_h - 1)) >> 10) + 4);
      const int src_x = x + j4 * 4 + 2;
      const int src_y = y + i4 * 4 + 2;
      const int dst_x = params.p[2] * src_x + params.p[3] * src_y + params.p[0];
      const int dst_y = params.p[4] * src_x + params.p[5] * src_y + params.p[1];
      const int ix4 = dst_x >> 16;
      const int sx4 = dst_x & 0xffff;
      const int iy4 = dst_y >> 16;
      const int sy4 = dst_y & 0xffff;
      const int offs_x = round2(sx4, 10);
      int intermediate[9][4];
      for (int k = -4; k < 5; ++k) {
        for (int l = -2; l < 2; ++l) {
          int s = 0;
          for (int m = 0; m < 6; ++m) {
            s += tables::kExtWarpedFilters[offs_x][m] *
                 ref.data[clip3(first_y, bbox_last_y, iy4 + k) * ref.stride +
                          clip3(first_x, bbox_last_x, ix4 + l - 2 + m)];
          }
          intermediate[k + 4][l + 2] = round2(s, 3);
        }
      }
      const int offs_y = round2(sy4, 10);
      for (int k = -2; k < 2; ++k) {
        for (int l = -2; l < 2; ++l) {
          int s = 0;
          for (int m = 0; m < 6; ++m) {
            s += tables::kExtWarpedFilters[offs_y][m] * intermediate[k + m + 2][l + 2];
          }
          const int local_y = i4 * 4 + k + 2;
          const int local_x = j4 * 4 + l + 2;
          if (local_y < 0 || local_y >= h || local_x < 0 || local_x >= w) {
            continue;
          }
          const int value = round2(s, inter_round1);
          dst[(out_y0 + local_y) * dst_stride + out_x0 + local_x] =
              static_cast<std::int16_t>(clip ? clip1(value, 8) : value);
        }
      }
    }
  }
}

// A mildly non-trivial affine model: zoom + rotation-ish shear + translation.
WarpParams random_affine_params() {
  WarpParams p{};
  const int zoom = 65536 + random_range(-4000, 4000);
  p.p[2] = zoom;
  p.p[5] = zoom + random_range(-2000, 2000);
  p.p[3] = random_range(-1500, 1500);
  p.p[4] = random_range(-1500, 1500);
  p.p[0] = random_range(-6, 6) << 16;
  p.p[1] = random_range(-6, 6) << 16;
  return p;
}

void test_warp_literal_parity() {
  Plane ref(96, 96);
  for (int i = 0; i < ref.w * ref.h; ++i) {
    ref.data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  const int w = 32;
  const int h = 24;
  WarpBlockGeometry geom = make_geometry(w, h);
  geom.rect = {8, 12, w, h};
  geom.dst_x = 0;
  geom.dst_y = 0;
  std::int16_t got[32 * 24];
  std::int16_t want[32 * 24];

  for (int trial = 0; trial < 6; ++trial) {
    const WarpParams params = trial == 0 ? WarpParams{} : random_affine_params();
    EXPECT(literal_setup_shear(params).valid);
    EXPECT(warp_predict(params, ref.view(), geom, false, got, w));
    literal_warp_block(params, ref.view(), geom.rect.x, geom.rect.y, w, h, 0, 0, 11, true, 0, 0,
                       want, w);
    for (int i = 0; i < w * h; ++i) {
      EXPECT_EQ(got[i], want[i]);
    }
    // Compound precision.
    EXPECT(warp_predict(params, ref.view(), geom, true, got, w));
    literal_warp_block(params, ref.view(), geom.rect.x, geom.rect.y, w, h, 0, 0, 7, false, 0, 0,
                       want, w);
    for (int i = 0; i < w * h; ++i) {
      EXPECT_EQ(got[i], want[i]);
    }
    // Extended warp.
    EXPECT(warp_predict_ext(params, ref.view(), geom, false, got, w));
    literal_ext_warp(params, ref.view(), geom.rect.x, geom.rect.y, w, h, 11, true, 0, 0, want, w);
    for (int i = 0; i < w * h; ++i) {
      EXPECT_EQ(got[i], want[i]);
    }
  }

  // The 8-tap table's zero phase is 127/1, so an identity model is within 1 LSB
  // of a copy but not bit-exact; the extended warp table does have a delta row.
  WarpParams identity{};
  EXPECT(warp_predict(identity, ref.view(), geom, false, got, w));
  int max_delta = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int d = std::abs(int(got[i * w + j]) - int(ref.at(geom.rect.x + j, geom.rect.y + i)));
      max_delta = std::max(max_delta, d);
    }
  }
  EXPECT(max_delta <= 6);
}

void test_warp_identity_and_translation() {
  Plane ref(80, 80);
  for (int i = 0; i < ref.w * ref.h; ++i) {
    ref.data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  WarpBlockGeometry geom = make_geometry(32, 32);
  geom.dst_x = 0;
  geom.dst_y = 0;
  std::int16_t buf[64 * 64];

  WarpParams identity{};
  // The 8-tap table's zero phase is a 127/1 pair (a +1/128 sample leak) and each
  // pass rounds, so an identity model is a near-copy, not a bit-exact copy:
  // measured worst case is 4 LSB on random 8-bit data. This is a property of
  // §9.5, not of this implementation; test_warp_literal_parity checks the loop
  // itself sample by sample.
  EXPECT(warp_predict(identity, ref.view(), geom, false, buf, 32));
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      EXPECT(std::abs(int(buf[y * 32 + x]) - int(ref.at(x, y))) <= 6);
    }
  }

  // The 6-tap extended-warp table does have an exact zero-phase delta.
  EXPECT(warp_predict_ext(identity, ref.view(), geom, false, buf, 32));
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      EXPECT_EQ(buf[y * 32 + x], ref.at(x, y));
    }
  }

  // Compound output keeps four fractional bits; compare to the literal loop.
  EXPECT(warp_predict(identity, ref.view(), geom, true, buf, 32));
  {
    std::int16_t want[32 * 32];
    literal_warp_block(identity, ref.view(), 0, 0, 32, 32, 0, 0, 7, false, 0, 0, want, 32);
    for (int i = 0; i < 32 * 32; ++i) {
      EXPECT_EQ(buf[i], want[i]);
    }
  }

  // Integer translation. p[0] translates columns (dstX) and p[1] rows (dstY),
  // so dst(x,y) = ref(x + 3, y + 2).
  WarpParams translated{};
  translated.p[0] = 3 << 16;
  translated.p[1] = 2 << 16;
  EXPECT(warp_predict(translated, ref.view(), geom, false, buf, 32));
  {
    std::int16_t want[32 * 32];
    literal_warp_block(translated, ref.view(), 0, 0, 32, 32, 0, 0, 11, true, 0, 0, want, 32);
    for (int i = 0; i < 32 * 32; ++i) {
      EXPECT_EQ(buf[i], want[i]);
    }
  }

  EXPECT(warp_predict_ext(translated, ref.view(), geom, false, buf, 32));
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      EXPECT_EQ(buf[y * 32 + x], ref.at(x + 3, y + 2));
    }
  }

  // uint8 convenience path.
  Plane out(80, 80);
  WarpBlockGeometry local = geom;
  local.dst_x = 0;
  local.dst_y = 0;
  EXPECT(warp_predict_u8(identity, ref.view(), local, out.mut()));
  std::int16_t reference_pred[32 * 32];
  EXPECT(warp_predict(identity, ref.view(), local, false, reference_pred, 32));
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      EXPECT_EQ(out.at(x, y), reference_pred[y * 32 + x]);
    }
  }

  // A destination origin separate from the model origin just offsets writes.
  std::int16_t local_buf[64 * 64];
  WarpBlockGeometry shifted = geom;
  shifted.dst_x = 2;
  shifted.dst_y = 3;
  EXPECT(warp_predict(identity, ref.view(), shifted, false, local_buf, 40));
  EXPECT(warp_predict(identity, ref.view(), geom, false, buf, 32));
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      EXPECT_EQ(local_buf[(3 + i) * 40 + 2 + j], buf[i * 32 + j]);
    }
  }
}

void test_warp_chroma() {
  // 4:2:0 chroma: the model is expressed in luma coordinates and the predictor
  // samples the chroma plane at (dstX >> 1, dstY >> 1).
  Plane ref(64, 64);
  for (int i = 0; i < ref.w * ref.h; ++i) {
    ref.data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  WarpBlockGeometry geom{};
  geom.plane = 1;
  geom.sub_x = 1;
  geom.sub_y = 1;
  geom.rect = {8, 8, 16, 16};
  geom.dst_x = 0;
  geom.dst_y = 0;
  geom.mi_col = 4;
  geom.mi_row = 4;
  geom.bw4 = 8;
  geom.bh4 = 8;
  geom.mi_cols = 32;
  geom.mi_rows = 32;
  geom.ref_w = 128;
  geom.ref_h = 128;
  geom.frame_w = 128;
  geom.frame_h = 128;

  WarpParams identity{};
  std::int16_t buf[16 * 16];
  EXPECT(warp_predict(identity, ref.view(), geom, false, buf, 16));
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      EXPECT(std::abs(int(buf[y * 16 + x]) - int(ref.at(8 + x, 8 + y))) <= 6);
    }
  }
  EXPECT(warp_predict_ext(identity, ref.view(), geom, false, buf, 16));
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      EXPECT_EQ(buf[y * 16 + x], ref.at(8 + x, 8 + y));
    }
  }

  // A pure luma translation of 2 samples is a 1-sample chroma translation.
  WarpParams translated{};
  translated.p[0] = 2 << 16;
  EXPECT(warp_predict_ext(translated, ref.view(), geom, false, buf, 16));
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      EXPECT_EQ(buf[y * 16 + x], ref.at(9 + x, 8 + y));
    }
  }
}

void test_warp_estimation() {
  // Literal transcription of §7.13.3.23.
  auto literal = [](const WarpSample* s, int count, int mid_x, int mid_y, const int block_mv[2],
                    WarpParams* out) {
    long long A[2][2] = {{0, 0}, {0, 0}};
    long long Bx[2] = {0, 0};
    long long By[2] = {0, 0};
    const int suy = mid_y * 8;
    const int sux = mid_x * 8;
    const int duy = suy + block_mv[0];
    const int dux = sux + block_mv[1];
    auto ls = [](long long a, long long b) { return ((a * b) >> 2) + (a + b); };
    for (int i = 0; i < count; ++i) {
      const long long sy = s[i].src_y - suy;
      const long long sx = s[i].src_x - sux;
      const long long dy = s[i].dst_y - duy;
      const long long dx = s[i].dst_x - dux;
      if (std::llabs(sx - dx) < 256 && std::llabs(sy - dy) < 256) {
        A[0][0] += ls(sx, sx) + 8;
        A[0][1] += ls(sx, sy) + 4;
        A[1][1] += ls(sy, sy) + 8;
        Bx[0] += ls(sx, dx) + 8;
        Bx[1] += ls(sy, dx) + 4;
        By[0] += ls(sx, dy) + 4;
        By[1] += ls(sy, dy) + 8;
      }
    }
    const long long det = A[0][0] * A[1][1] - A[0][1] * A[0][1];
    WarpParams params{};
    if (det == 0) {
      params.p[2] = 1 << 16;
      params.p[5] = 1 << 16;
    } else {
      const DivScaleShift ds = resolve_divisor(static_cast<int>(det));
      long long div_shift = ds.shift - 16;
      long long div_factor = ds.factor;
      if (div_shift < 0) {
        div_factor <<= -div_shift;
        div_shift = 0;
      }
      auto diag = [&](long long v) {
        const long long prod = v * div_factor;
        const long long rounded =
            div_shift <= 0 ? prod
                           : (prod >= 0 ? ((prod + (1LL << (div_shift - 1))) >> div_shift)
                                        : -(((-prod) + (1LL << (div_shift - 1))) >> div_shift));
        return static_cast<int>(
            std::max<long long>(std::min<long long>(rounded, 2147483647), -2147483648LL));
      };
      params.p[2] = diag(A[1][1] * Bx[0] - A[0][1] * Bx[1]);
      params.p[3] = diag(-A[0][1] * Bx[0] + A[0][0] * Bx[1]);
      params.p[4] = diag(A[1][1] * By[0] - A[0][1] * By[1]);
      params.p[5] = diag(-A[0][1] * By[0] + A[0][0] * By[1]);
      params = reduce_warp_model(params);
    }
    // get_warp_translation.
    const int one = 1 << 16;
    const int vx = block_mv[1] * (1 << 13) - (mid_x * (params.p[2] - one) + mid_y * params.p[3]);
    const int vy = block_mv[0] * (1 << 13) - (mid_x * params.p[4] + mid_y * (params.p[5] - one));
    params.p[0] = clip3(-kWarpedModelTransClamp, kWarpedModelTransClamp - (1 << 6), vx);
    params.p[1] = clip3(-kWarpedModelTransClamp, kWarpedModelTransClamp - (1 << 6), vy);
    *out = params;
  };

  for (int trial = 0; trial < 50; ++trial) {
    const int count = random_range(0, kLeastSquaresSamplesMax);
    WarpSample samples[kLeastSquaresSamplesMax];
    const int mid_x = random_range(4, 60);
    const int mid_y = random_range(4, 60);
    const int block_mv[2] = {random_range(-40, 40), random_range(-40, 40)};
    for (int i = 0; i < count; ++i) {
      samples[i].src_x = random_range(0, 512);
      samples[i].src_y = random_range(0, 512);
      samples[i].dst_x = samples[i].src_x + random_range(-64, 64);
      samples[i].dst_y = samples[i].src_y + random_range(-64, 64);
    }
    WarpParams got{};
    WarpParams want{};
    got = warp_estimate(samples, count, mid_x, mid_y, block_mv);
    literal(samples, count, mid_x, mid_y, block_mv, &want);
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(got.p[i], want.p[i]);
    }
    // The model reproduces the block-centre MV (no clipping in this range).
    int mv[2] = {0, 0};
    get_sub_block_warp_mv(got, 0, 0, 0, mid_x, mid_y, 1, 1, 1, mv);
  }
}

void test_motion_vector_scaling() {
  const int mv[2] = {0, 0};
  MvScaleOutput s = motion_vector_scaling(0, 0, 0, 0, 0, mv, false, 256, 256, 256, 256);
  // startX/Y include the half-tap rounding offset off = 32.
  EXPECT_EQ(s.start_x, 32);
  EXPECT_EQ(s.start_y, 32);
  EXPECT_EQ(s.step_x, 1024);
  EXPECT_EQ(s.step_y, 1024);
  const int mv2[2] = {8, -8};  // 1 luma sample down, 1 left
  s = motion_vector_scaling(0, 0, 0, 0, 0, mv2, false, 256, 256, 256, 256);
  // One luma sample is 1024 in the 1/1024 output units.
  EXPECT_EQ(s.start_x, 32 - 1024);
  EXPECT_EQ(s.start_y, 32 + 1024);
  // Half-size reference: 2x scale.
  s = motion_vector_scaling(0, 0, 0, 4, 4, mv, false, 128, 128, 256, 256);
  EXPECT_EQ(s.step_x, 512);
  EXPECT_EQ(s.step_y, 512);
  EXPECT(is_scaled_ref(128, 128, 256, 256));
  EXPECT(!is_scaled_ref(256, 256, 256, 256));
}

// ---------------------------------------------------------------------------
// BAWP (§7.13.3.25)
// ---------------------------------------------------------------------------

struct BawpFixture {
  Plane cur;
  Plane prev;
  Plane dst;
  BawpFixture(int w, int h) : cur(w, h), prev(w, h), dst(w, h) {}
};

void test_bawp() {
  const int w = 192;
  const int h = 192;
  BawpFixture f(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t idx = static_cast<std::size_t>(y) * w + x;
      f.cur.data[idx] = static_cast<std::uint8_t>(random_range(20, 220));
      f.prev.data[idx] = static_cast<std::uint8_t>(random_range(20, 220));
      f.dst.data[idx] = f.cur.data[idx];
    }
  }
  // Make the prediction equal to the current reconstruction in the block:
  // BAWP should then fit alpha=256-ish, beta~0.
  for (int y = 40; y < 56; ++y) {
    for (int x = 48; x < 64; ++x) {
      f.cur.data[static_cast<std::size_t>(y) * w + x] =
          f.prev.data[static_cast<std::size_t>(y) * w + x];
    }
  }

  for (int trial = 0; trial < 40; ++trial) {
    BawpJob job{};
    job.ctx.plane = 0;
    job.ctx.sub_x = 0;
    job.ctx.sub_y = 0;
    job.ctx.mi_cols = w / 4;
    job.ctx.mi_rows = h / 4;
    job.ctx.mi_col = random_range(2, w / 4 - 6);
    job.ctx.mi_row = random_range(2, h / 4 - 6);
    job.ctx.avail_u = (next_random() & 1u) != 0u;
    job.ctx.avail_l = (next_random() & 1u) != 0u;
    job.rect.x = job.ctx.mi_col * 4;
    job.rect.y = job.ctx.mi_row * 4;
    job.rect.w = 4 * random_range(1, 4);
    job.rect.h = 4 * random_range(1, 4);
    if (job.rect.x + job.rect.w > w) {
      job.rect.w = w - job.rect.x;
    }
    if (job.rect.y + job.rect.h > h) {
      job.rect.h = h - job.rect.y;
    }
    job.mv[0] = random_range(-24, 24);
    job.mv[1] = random_range(-24, 24);

    // Copy the reconstruction window for comparison.
    std::vector<std::uint8_t> before(f.cur.data.begin(), f.cur.data.end());
    for (int i = 0; i < job.rect.h; ++i) {
      for (int j = 0; j < job.rect.w; ++j) {
        before[static_cast<std::size_t>(job.rect.y + i) * w + job.rect.x + j] =
            f.prev.data[static_cast<std::size_t>(job.rect.y + i) * w + job.rect.x + j];
      }
    }
    f.dst.data = before;

    job.cur = f.cur.view();
    job.cur.data = before.data();
    job.prev = f.prev.view();
    job.dst = f.dst.mut();
    job.explicit_bawp = false;
    job.bawp_alpha = 300;
    BawpFit fit{};
    EXPECT(bawp_apply_block(job, &fit));

    // Literal §7.13.3.25 from the saved window.
    const int dy = to_fullmv(job.mv[0]);
    const int dx = to_fullmv(job.mv[1]);
    const int ref_y = (job.ctx.mi_row * 4 + dy);
    const int ref_x = (job.ctx.mi_col * 4 + dx);
    const int bw = job.rect.w;
    const int bh = job.rect.h;
    int bw2 = bw < 16 ? bw : 16;
    int bh2 = bh < 16 ? bh : 16;
    const int width = bw2 == 12 ? 8 : bw2;
    const int height = bh2 == 12 ? 8 : bh2;
    int num_up = 0;
    int num_left = 0;
    if (job.ctx.avail_u && job.ctx.avail_l) {
      if (width == 16 && height == 16) {
        num_up = 16;
        num_left = 16;
      } else if (width > 4 && height > 4) {
        num_up = 8;
        num_left = 8;
      } else if (width < 16 && height < 16) {
        num_up = 4;
        num_left = 4;
      } else if (width == 16) {
        num_up = 16;
      } else {
        num_left = 16;
      }
    } else if (job.ctx.avail_u) {
      num_up = width;
    } else if (job.ctx.avail_l) {
      num_left = height;
    }
    long long sum_x = 0;
    long long sum_y = 0;
    long long sum_xx = 0;
    long long sum_xy = 0;
    int count = 0;
    if (num_up > 0) {
      const int step = width / num_up;
      for (int i = step >> 1; i < width; i += step) {
        const int recon = before[static_cast<std::size_t>(job.rect.y - 1) * w + job.rect.x + i];
        const int ref = f.prev.data[static_cast<std::size_t>(ref_y - 1) * w + ref_x + i];
        sum_x += ref;
        sum_y += recon;
        sum_xy += ref * recon;
        sum_xx += ref * ref;
      }
      count += num_up;
    }
    if (num_left > 0) {
      const int step = height / num_left;
      for (int i = step >> 1; i < height; i += step) {
        const int recon = before[static_cast<std::size_t>(job.rect.y + i) * w + job.rect.x - 1];
        const int ref = f.prev.data[static_cast<std::size_t>(ref_y + i) * w + ref_x - 1];
        sum_x += ref;
        sum_y += recon;
        sum_xy += ref * recon;
        sum_xx += ref * ref;
      }
      count += num_left;
    }
    int alpha = 256;
    int beta = -128;
    if (count > 0) {
      const long long nor = sum_xy - sum_x * sum_y / count;
      const long long der = sum_xx - sum_x * sum_x / count;
      if (der != 0 && nor != 0) {
        alpha = resolve_division(static_cast<int>(nor), static_cast<int>(der), 8);
        if (alpha == 0) {
          alpha = 256;
        }
      }
      beta = static_cast<int>(((sum_y << 8) - sum_x * alpha) / count);
    }
    EXPECT_EQ(fit.alpha, alpha);
    EXPECT_EQ(fit.beta, beta);
    EXPECT_EQ(fit.count, count);
    for (int i = 0; i < job.rect.h; ++i) {
      for (int j = 0; j < job.rect.w; ++j) {
        const int orig = before[static_cast<std::size_t>(job.rect.y + i) * w + job.rect.x + j];
        const int expected = clip1((orig * alpha + beta) >> 8, 8);
        EXPECT_EQ(f.dst.data[static_cast<std::size_t>(job.rect.y + i) * w + job.rect.x + j],
                  expected);
      }
    }
  }

  // Explicit alpha.
  EXPECT_EQ(bawp_explicit_alpha(0, 1, true), 256 + 16);
  EXPECT_EQ(bawp_explicit_alpha(1, 1, true), 256 + 32);
  EXPECT_EQ(bawp_explicit_alpha(2, 5, true), 256 + 64);
  EXPECT_EQ(bawp_explicit_alpha(2, 5, false), 256 - 64);
  EXPECT_EQ(bawp_list_index(true, false, false), 0);
  EXPECT_EQ(bawp_list_index(false, true, true), 1);
  EXPECT_EQ(bawp_list_index(false, true, false), 2);

  // Template plan matrix.
  BawpContext ctx{};
  ctx.plane = 0;
  ctx.avail_u = true;
  ctx.avail_l = true;
  BawpTemplatePlan plan = bawp_template_plan(ctx, 16, 16);
  EXPECT_EQ(plan.num_up, 16);
  EXPECT_EQ(plan.num_left, 16);
  plan = bawp_template_plan(ctx, 8, 8);
  EXPECT_EQ(plan.num_up, 8);
  EXPECT_EQ(plan.num_left, 8);
  plan = bawp_template_plan(ctx, 4, 4);
  EXPECT_EQ(plan.num_up, 4);
  EXPECT_EQ(plan.num_left, 4);
  plan = bawp_template_plan(ctx, 16, 4);
  EXPECT_EQ(plan.width, 16);
  EXPECT_EQ(plan.num_up, 16);
  EXPECT_EQ(plan.num_left, 0);
  ctx.avail_l = false;
  plan = bawp_template_plan(ctx, 8, 16);
  EXPECT_EQ(plan.num_up, 8);
  EXPECT_EQ(plan.num_left, 0);
  ctx.plane = 1;
  ctx.avail_l = true;
  plan = bawp_template_plan(ctx, 8, 8);
  EXPECT_EQ(plan.num_up, 8);
  EXPECT_EQ(plan.num_left, 8);
  ctx.sub_x = 1;
  ctx.sub_y = 1;
}

void test_bawp_zero_template() {
  // count == 0 -> alpha 256, beta -128, i.e. every sample loses one.
  Plane cur(64, 64);
  Plane prev(64, 64);
  for (int i = 0; i < 64 * 64; ++i) {
    cur.data[static_cast<std::size_t>(i)] = 100;
    prev.data[static_cast<std::size_t>(i)] = 90;
  }
  BawpJob job{};
  job.ctx.plane = 0;
  job.ctx.mi_cols = 16;
  job.ctx.mi_rows = 16;
  job.ctx.mi_col = 8;
  job.ctx.mi_row = 8;
  job.ctx.avail_u = false;
  job.ctx.avail_l = false;
  job.rect = {32, 32, 16, 16};
  job.mv[0] = 0;
  job.mv[1] = 0;
  job.cur = cur.view();
  job.prev = prev.view();
  job.dst = cur.mut();
  BawpFit fit{};
  EXPECT(bawp_apply_block(job, &fit));
  EXPECT_EQ(fit.alpha, 256);
  EXPECT_EQ(fit.beta, -128);
  EXPECT_EQ(fit.count, 0);
  EXPECT_EQ(int(cur.at(40, 40)), 99);
}

// ---------------------------------------------------------------------------
// B-frame reference bookkeeping and GOP structure
// ---------------------------------------------------------------------------

void test_relative_dist_and_order_hint() {
  EXPECT_EQ(get_relative_dist(10, 4), 6);
  EXPECT_EQ(get_relative_dist(4, 10), -6);
  EXPECT_EQ(get_relative_dist(kRestrictedOrderHint, 4), 127);
  EXPECT_EQ(get_relative_dist(4, kRestrictedOrderHint), -127);
  EXPECT_EQ(get_relative_dist(kRestrictedOrderHint, kRestrictedOrderHint), 0);
  EXPECT_EQ(get_relative_dist(300, 4), 127);  // clipped

  // get_disp_order_hint lift.
  EXPECT_EQ(display_order_hint(3, 5, 30), 35);
  EXPECT_EQ(display_order_hint(3, 5, 10), 3);
  EXPECT_EQ(display_order_hint(3, 0, 10), 3);

  RefSlotState state{};
  refresh_reference_slots(&state, 1u, 28, 640, 360, 100, true);
  EXPECT(state.slots[0].valid);
  EXPECT_EQ(state.slots[0].order_hint, 28);
  EXPECT_EQ(max_display_order_hint(state, kNumRefFrames, true), 28);
  // The reconstruction window is half a span around the largest hint in the
  // store, so the encoder must keep `absolute_hint - max_disp` small. Model that
  // with a store that is a few frames behind the frame being coded.
  for (int bits = 3; bits <= 8; ++bits) {
    for (int hint = 0; hint < 200; hint += 7) {
      RefSlotState near{};
      refresh_reference_slots(&near, 1u, hint > 3 ? hint - 3 : 0, 640, 360, 100, true);
      const int lsbs = order_hint_lsbs_for(hint, bits, near, kNumRefFrames);
      EXPECT_EQ(display_order_hint(lsbs, bits, max_display_order_hint(near, kNumRefFrames, true)),
                hint);
    }
  }
}

void test_ref_frame_list() {
  RefListInput in{};
  in.order_hint = 16;
  in.frame_width = 640;
  in.frame_height = 360;
  in.check_res = false;
  in.state.frame_counter = 5;
  // Slots: 0 = 8 (past, close), 1 = 0 (past, far), 2 = 24 (future),
  // 3 = duplicate of slot 0, 4 = invalid, 5 = 32 (future, far).
  const int hints[6] = {8, 0, 24, 8, -1, 32};
  const int qs[6] = {100, 100, 100, 100, 0, 100};
  const int counters[6] = {1, 2, 3, 1, -1, 4};
  for (int i = 0; i < 6; ++i) {
    in.state.slots[i].valid = hints[i] >= 0;
    in.state.slots[i].order_hint = hints[i];
    in.state.slots[i].width = 640;
    in.state.slots[i].height = 360;
    in.state.slots[i].base_q_idx = qs[i];
    in.state.slots[i].counter = counters[i];
  }
  const RefListResult list = build_ref_frame_list(in);
  EXPECT_EQ(list.num_total_refs, 4);
  // The duplicate (slot 3) must be dropped; the rest rank by distance decay.
  for (int i = 0; i < list.num_total_refs; ++i) {
    EXPECT(list.ref_frame_idx[i] != 3);
  }
  int past = 0;
  int future = 0;
  for (int i = 0; i < list.num_total_refs; ++i) {
    if (list.scores_distance[i] > 0) {
      ++past;
    } else if (list.scores_distance[i] < 0) {
      ++future;
    }
  }
  EXPECT_EQ(past, 2);
  EXPECT_EQ(future, 2);
  const PastFutureLists pfl =
      get_past_future_cur_ref_lists(list.ref_order_hint, list.scores_distance, list.num_total_refs);
  EXPECT_EQ(pfl.num_past, 2);
  EXPECT_EQ(pfl.num_future, 2);
  EXPECT_EQ(pfl.num_cur, 0);
  EXPECT(pfl.closest_past >= 0);
  EXPECT(pfl.closest_future >= 0);
  EXPECT(pfl.furthest_future >= 0);
  EXPECT_EQ(list.ref_order_hint[pfl.closest_past], 8);
  EXPECT_EQ(list.ref_order_hint[pfl.closest_future], 24);
  EXPECT_EQ(list.ref_order_hint[pfl.furthest_future], 32);
  EXPECT(pfl.skip_seg_frame == pfl.closest_past);

  // Current-frame references (distance 0) become curRefs.
  const int cur_order[7] = {16, -1, -1, -1, -1, -1, -1};
  const int cur_dist[7] = {0, 0, 0, 0, 0, 0, 0};
  const PastFutureLists cur_lists = get_past_future_cur_ref_lists(cur_order, cur_dist, 1);
  EXPECT_EQ(cur_lists.num_cur, 1);
  EXPECT_EQ(cur_lists.skip_seg_frame, 0);

  // Dropping: 9 valid, distinct candidates -> only REFS_PER_FRAME=7 survive and
  // the far/high-q ones are pushed to the end by the qThreshold rule.
  RefListInput big{};
  big.order_hint = 100;
  big.frame_width = 640;
  big.frame_height = 360;
  big.state.frame_counter = 20;
  for (int i = 0; i < 9; ++i) {
    big.state.slots[i].valid = true;
    big.state.slots[i].order_hint = 80 + i * 4;  // all past
    big.state.slots[i].width = 640;
    big.state.slots[i].height = 360;
    big.state.slots[i].base_q_idx = i < 4 ? 100 : 200;  // far ones high-q
    big.state.slots[i].counter = 10 + i;
  }
  const RefListResult big_list = build_ref_frame_list(big);
  EXPECT_EQ(big_list.num_total_refs, 7);
}

void test_gop_structure() {
  for (int gop_size : {2, 4, 8, 16}) {
    GopConfig config{};
    config.gop_size = gop_size;
    config.num_slots = kNumRefFrames;
    config.start_order_hint = 40;
    const std::vector<GopFrame> gop = build_hierarchical_gop(config);
    EXPECT_EQ(static_cast<int>(gop.size()), gop_size);
    EXPECT(gop[0].is_key);
    EXPECT(gop[static_cast<std::size_t>(gop_size - 1)].is_anchor);
    EXPECT_EQ(gop[0].coding_index, 0);

    // Coding indices must be a permutation of 0..n-1.
    std::vector<bool> seen(static_cast<std::size_t>(gop_size), false);
    std::vector<bool> slots_seen(static_cast<std::size_t>(kNumRefFrames), false);
    for (const GopFrame& f : gop) {
      EXPECT(f.coding_index >= 0 && f.coding_index < gop_size);
      EXPECT(!seen[static_cast<std::size_t>(f.coding_index)]);
      seen[static_cast<std::size_t>(f.coding_index)] = true;
      EXPECT_EQ(f.display_order_hint, config.start_order_hint + f.display_index);
      EXPECT_EQ(f.refresh_frame_flags, 1u << f.slot);
      EXPECT(!slots_seen[static_cast<std::size_t>(f.slot)]);
      slots_seen[static_cast<std::size_t>(f.slot)] = true;
      if (f.is_key) {
        EXPECT_EQ(f.num_total_refs, 0);
      } else {
        EXPECT(f.num_total_refs >= 1);
        for (int r = 0; r < f.num_total_refs; ++r) {
          EXPECT(gop[static_cast<std::size_t>(f.ref_display[r])].coding_index < f.coding_index);
          EXPECT_EQ(f.ref_order_hint[r], config.start_order_hint + f.ref_display[r]);
          // Reference temporal layers must not exceed the frame's.
          EXPECT(gop[static_cast<std::size_t>(f.ref_display[r])].temporal_layer <=
                 f.temporal_layer);
        }
        if (!f.is_anchor) {
          EXPECT_EQ(f.num_total_refs, 2);
          EXPECT(f.ref_display[0] < f.display_index);
          EXPECT(f.ref_display[1] > f.display_index);
        }
      }
    }

    const GopValidation validation = validate_gop(gop, config);
    EXPECT(validation.ok);
    EXPECT_EQ(validation.num_frames, gop_size);
    for (const GopFrame& f : gop) {
      if (!f.is_key) {
        EXPECT(validation.refs_were_past_future[f.coding_index]);
      }
    }
  }

  // Invalid GOP sizes / slot counts are rejected.
  GopConfig bad{};
  bad.gop_size = 3;
  EXPECT(build_hierarchical_gop(bad).empty());
  bad.gop_size = 8;
  bad.num_slots = 1;
  EXPECT(build_hierarchical_gop(bad).empty());
  bad.num_slots = 16;
  bad.key_slot = 16;
  EXPECT(build_hierarchical_gop(bad).empty());
}

void test_gop_key_reset() {
  // A key frame invalidates every refreshed slot except the first.
  RefSlotState state{};
  refresh_reference_slots(&state, 0xFu, 10, 640, 360, 100, false);
  EXPECT_EQ(state.slots[0].valid, true);
  EXPECT_EQ(state.slots[1].valid, true);
  refresh_reference_slots(&state, 0x3u, 20, 640, 360, 100, true);
  EXPECT_EQ(state.slots[0].valid, true);
  EXPECT_EQ(state.slots[1].valid, false);  // refreshed by the key, not first
  EXPECT_EQ(state.slots[2].valid, true);   // untouched, still valid
  EXPECT_EQ(state.slots[0].order_hint, 20);
  EXPECT_EQ(state.slots[2].order_hint, 10);
}

// ---------------------------------------------------------------------------
// OBMC (encoder-only experiment)
// ---------------------------------------------------------------------------

void test_obmc() {
  EXPECT_EQ(obmc_weight(0, 4), 51);
  EXPECT_EQ(obmc_weight(3, 4), 12);
  EXPECT_EQ(obmc_weight(4, 4), 0);
  EXPECT_EQ(obmc_weight(-1, 4), 0);
  for (int d = 0; d < 4; ++d) {
    EXPECT(obmc_weight(d, 4) >= obmc_weight(d + 1, 4));
  }
  EXPECT_EQ(obmc_blend_sample(100, 100, 32), 100);
  EXPECT_EQ(obmc_blend_sample(0, 255, 0), 0);
  EXPECT_EQ(obmc_blend_sample(0, 255, 64), 255);

  const int w = 16;
  const int h = 16;
  std::vector<std::uint8_t> dst(static_cast<std::size_t>(w) * h, 100);
  std::vector<std::uint8_t> above(static_cast<std::size_t>(w) * h, 200);
  std::vector<std::uint8_t> left(static_cast<std::size_t>(w) * h, 50);
  ObmcNeighbourPred a{above.data(), w};
  ObmcNeighbourPred l{left.data(), w};
  ObmcConfig config{};
  config.rect = {0, 0, w, h};
  config.overlap = 4;
  config.dst = dst.data();
  config.dst_stride = w;
  EXPECT(obmc_apply(config, &a, &l));
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int expected = 100;
      if (i < 4) {
        expected = obmc_blend_sample(expected, 200, obmc_weight(i, 4));
      }
      if (j < 4) {
        expected = obmc_blend_sample(expected, 50, obmc_weight(j, 4));
      }
      EXPECT_EQ(dst[static_cast<std::size_t>(i) * w + j], expected);
    }
  }
  const std::int64_t delta = obmc_prediction_delta(config, &a, &l, dst.data(), w);
  EXPECT_EQ(delta, 0);  // delta compares dst with itself
  // Overlap is clamped to min(w,h)-1.
  config.overlap = 1000;
  config.dst = dst.data();
  EXPECT(obmc_apply(config, &a, nullptr));
  // Unavailable neighbours are no-ops.
  EXPECT(obmc_apply(config, nullptr, nullptr));
}

}  // namespace

int main() {
  test_math_helpers();
  test_warp_tables();
  test_wedge_tables();
  test_wedge_masks();
  test_compound_blend();
  test_compound_chroma_mask();
  test_inter_intra();
  test_compound_search();
  test_warp_setup_shear();
  test_warp_literal_parity();
  test_warp_identity_and_translation();
  test_warp_chroma();
  test_warp_estimation();
  test_motion_vector_scaling();
  test_bawp();
  test_bawp_zero_template();
  test_relative_dist_and_order_hint();
  test_ref_frame_list();
  test_gop_structure();
  test_gop_key_reset();
  test_obmc();

  if (g_failures == 0) {
    std::printf("advanced_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::printf("advanced_test: %d/%d checks FAILED\n", g_failures, g_checks);
  return 1;
}
