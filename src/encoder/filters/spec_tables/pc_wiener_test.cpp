// CPU reference tests for §7.20.4 pixel classified Wiener filter.
//
// The oracle is a second, independent transcription of the pseudocode.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "encoder/filters/spec_tables/pc_wiener.h"
#include "encoder/filters/spec_tables/pc_wiener_tables.h"
#include "encoder/filters/spec_tables/spec_test_common.h"

namespace {

using namespace av2enc::filters::spec;
using namespace av2enc::filters::spec::test;
using namespace av2enc::filters::spec_tables;

struct Extents {
  int start_x, end_x, start_y, end_y, stripe_start_y, stripe_end_y;
};

// §7.20.4 constants, transcribed independently from the printed text.
constexpr int kOracleConfig[25][2] = {
    {1, 0},  {-1, 0}, {0, 1},  {0, -1}, {2, 0},  {-2, 0}, {0, 2},  {0, -2},
    {1, 1},  {-1, -1}, {-1, 1}, {1, -1}, {2, 1},  {-2, -1}, {2, -1}, {-2, 1},
    {1, 2},  {-1, -2}, {1, -2}, {-1, 2}, {3, 0},  {-3, 0}, {0, 3},  {0, -3},
    {0, 0},
};
constexpr int kOracleNormalizer[5] = {0, 3739, 3273, 3074, 7};
constexpr int kOracleWeights[4][3] = {
    {-527, 15325, 321}, {26436, -17705, 17905}, {366, -147, -194}, {202, -267, -179},
};
constexpr int kOracleOffsets[4] = {-547, -21565, -573, -680};

int OracleClip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }
std::int64_t OracleRound2(std::int64_t x, int n) {
  return n == 0 ? x : (x + (std::int64_t{1} << (n - 1))) >> n;
}
std::int64_t OracleRound2Signed(std::int64_t x, int n) {
  return x >= 0 ? OracleRound2(x, n) : -OracleRound2(-x, n);
}
int OracleMaxQ(int bd) { return bd == 8 ? 255 : 303; }

int OracleGetQ(int qindex, int delta, int bd) {
  if (qindex == 0 && delta <= 0) return kAcQlookup[0];
  const int q = OracleClip3(1, OracleMaxQ(bd), qindex + delta);
  return kQlookup[q];
}

std::int64_t OracleQval(int qindex, std::int64_t tskip, int i, int bd) {
  std::int64_t qstep = OracleGetQ(qindex, 0, bd);
  const int qstep_shift = 13;
  qstep = OracleRound2(qstep, bd - 8);
  const int diff_shift = qstep_shift - 8;
  const std::int64_t prod = OracleRound2(tskip * qstep, 8);
  const std::int64_t qval = kOracleWeights[i][0] * (tskip << diff_shift) +
                            kOracleWeights[i][1] * qstep +
                            kOracleWeights[i][2] * prod;
  return 255 * (kOracleOffsets[i] + OracleRound2Signed(qval, qstep_shift));
}

int OracleFilterSetIndex(int q) {
  if (q < 130) return 0;
  if (q < 190) return 1;
  if (q < 220) return 2;
  return 3;
}

struct OracleCtx {
  const Image* curr;
  const Image* cdef;
  Extents e;
  int mi_col_end;
  int mi_row_start;
  int mi_row_end;
  const std::uint8_t* tx_skip;
  int tx_skip_stride;
  int bit_depth;
};

int OracleSampleCtx(const OracleCtx& c, int x, int y) {
  return OracleSample(*c.curr, *c.cdef, c.e.start_x, c.e.end_x, c.e.start_y,
                      c.e.end_y, c.e.stripe_start_y, c.e.stripe_end_y, x, y);
}

int OracleTxSkip(const OracleCtx& c, int block_start_x, int block_end_x, int x,
                 int y) {
  x = std::min(block_end_x, x);
  x = std::max(block_start_x, x);
  y = OracleClip3(c.e.stripe_start_y, c.e.stripe_end_y, y);
  y = OracleClip3(c.mi_row_start * 4, c.mi_row_end * 4 - 1, y);
  return c.tx_skip[(y >> 2) * c.tx_skip_stride + (x >> 2)];
}

void OracleFeatures(const OracleCtx& c, int block_start_x, int block_end_x, int x,
                    int y, std::int64_t f[4], int* skip) {
  x = std::min(block_end_x + 2, x);
  const int m = OracleSampleCtx(c, x, y);
  const int up = OracleSampleCtx(c, x, y - 1);
  const int down = OracleSampleCtx(c, x, y + 1);
  const int upright = OracleSampleCtx(c, x + 1, y - 1);
  const int downleft = OracleSampleCtx(c, x - 1, y + 1);
  const int downright = OracleSampleCtx(c, x + 1, y + 1);
  const int upleft = OracleSampleCtx(c, x - 1, y - 1);
  f[0] = 0;
  f[1] = std::abs(up - 2 * m + down);
  f[2] = std::abs(upright - 2 * m + downleft);
  f[3] = std::abs(upleft - 2 * m + downright);
  *skip = OracleTxSkip(c, block_start_x, block_end_x, x, y);
}

int OracleClass(const OracleCtx& c, int x, int y, int base_q_idx) {
  const int block_start_x = (x >> 6) << 6;
  const int block_end_x = std::min(c.mi_col_end * 4 - 1, block_start_x + 63);
  std::int64_t f[4] = {0, 0, 0, 0};
  std::int64_t s = 0;
  for (int dy = -1; dy <= 4; ++dy) {
    for (int dx = -1; dx <= 4; ++dx) {
      std::int64_t tf[4];
      int skip = 0;
      OracleFeatures(c, block_start_x, block_end_x, x + dx, y + dy, tf, &skip);
      for (int i = 0; i < 4; ++i) f[i] += tf[i];
      s += skip;
    }
  }
  for (int i = 0; i < 4; ++i) {
    f[i] = OracleRound2(f[i] * kOracleNormalizer[i], c.bit_depth - 8);
  }
  const std::int64_t tskip = s * kOracleNormalizer[4];

  int lut = 0;
  for (int i = 0; i < 4; ++i) {
    std::int64_t qval =
        OracleRound2Signed(f[i] + OracleQval(base_q_idx, tskip, i, c.bit_depth), 14);
    qval = OracleClip3(0, 255, static_cast<int>(qval)) >> 5;
    lut += static_cast<int>(qval) << (3 * (3 - i));
  }
  return kPcWienerLutToClass[lut];
}

void OracleFilter(const OracleCtx& c, std::uint16_t* lr, int lr_stride,
                  int base_q_idx, int x, int y, int w, int h) {
  const int index = OracleFilterSetIndex(base_q_idx);
  const int cls = OracleClass(c, x, y, base_q_idx);
  const int filt = kPcWienerSubClassify[index][cls];
  const int maxv = (1 << c.bit_depth) - 1;
  for (int r2 = 0; r2 < h; ++r2) {
    for (int c2 = 0; c2 < w; ++c2) {
      const int m = OracleSampleCtx(c, x + c2, y + r2);
      std::int64_t s = static_cast<std::int64_t>(m) << 7;
      for (int i = 0; i < 25; ++i) {
        const int coeff = kPcWienerFilters[index][filt][i >> 1];
        const int dy = kOracleConfig[i][0];
        const int dx = kOracleConfig[i][1];
        s += static_cast<std::int64_t>(
                 OracleSampleCtx(c, x + c2 + dx, y + r2 + dy)) *
             coeff;
      }
      const int v = static_cast<int>(OracleRound2(s, 7));
      lr[(y + r2) * lr_stride + (x + c2)] =
          static_cast<std::uint16_t>(OracleClip3(0, maxv, v));
    }
  }
}

void TestFilterSetIndex() {
  CHECK(PcWienerFilterSetIndex(0) == 0);
  CHECK(PcWienerFilterSetIndex(129) == 0);
  CHECK(PcWienerFilterSetIndex(130) == 1);
  CHECK(PcWienerFilterSetIndex(189) == 1);
  CHECK(PcWienerFilterSetIndex(190) == 2);
  CHECK(PcWienerFilterSetIndex(219) == 2);
  CHECK(PcWienerFilterSetIndex(220) == 3);
  CHECK(PcWienerFilterSetIndex(255) == 3);
}

void TestAgainstOracle() {
  TestRng rng(0x0badf00d);
  int class_checks = 0;
  int filter_checks = 0;

  for (int bd : {8, 10}) {
    const int maxv = (1 << bd) - 1;
    const int qindices[] = {0, 1, 64, 90, 129, 130, 189, 190, 219, 220, 255};
    for (int base_q_idx : qindices) {
      for (int trial = 0; trial < 2; ++trial) {
        Image curr(32, 24);
        Image cdef(32, 24);
        Image lr(32, 24);
        for (int y = 0; y < 24; ++y) {
          for (int x = 0; x < 32; ++x) {
            curr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
            cdef.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
            lr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
          }
        }
        std::vector<std::uint8_t> tx_skip(6 * 8);
        for (auto& v : tx_skip) v = static_cast<std::uint8_t>(rng.Range(0, 1));

        OracleCtx oc;
        oc.curr = &curr;
        oc.cdef = &cdef;
        oc.e = Extents{0, 31, 0, 23, 4, 19};
        oc.mi_col_end = 8;  // 32 luma samples
        oc.mi_row_start = 0;
        oc.mi_row_end = 6;
        oc.tx_skip = tx_skip.data();
        oc.tx_skip_stride = 8;
        oc.bit_depth = bd;

        LumaSource src;
        src.curr = curr.px.data();
        src.cdef = cdef.px.data();
        src.curr_stride = curr.stride;
        src.cdef_stride = cdef.stride;
        src.luma_start_x = oc.e.start_x;
        src.luma_end_x = oc.e.end_x;
        src.luma_start_y = oc.e.start_y;
        src.luma_end_y = oc.e.end_y;
        src.luma_stripe_start_y = oc.e.stripe_start_y;
        src.luma_stripe_end_y = oc.e.stripe_end_y;

        for (int x : {0, 4, 8, 16}) {
          for (int y : {2, 8, 18}) {
            PcWienerConfig cfg;
            cfg.base_q_idx = base_q_idx;
            cfg.bit_depth = bd;
            cfg.mi_col_end = oc.mi_col_end;
            cfg.mi_row_start = oc.mi_row_start;
            cfg.mi_row_end = oc.mi_row_end;
            cfg.lr_tx_skip = tx_skip.data();
            cfg.lr_tx_skip_stride = 8;

            PcWienerFrame frame;
            frame.src = src;
            frame.lr = lr.px.data();
            frame.lr_stride = lr.stride;
            frame.filter_class = nullptr;
            frame.filter_class_stride = 0;

            const int got = PcWienerClass(frame, cfg, x, y);
            const int want = OracleClass(oc, x, y, base_q_idx);
            if (got != want) {
              std::fprintf(stderr, "pcw class bd=%d q=%d x=%d y=%d: %d != %d\n", bd,
                           base_q_idx, x, y, got, want);
            }
            CHECK(got == want);
            ++class_checks;

            // Classify-only must not touch LrFrame.
            Image untouched = lr;
            std::vector<std::uint8_t> grid(6 * 8, 0);
            frame.filter_class = grid.data();
            frame.filter_class_stride = 8;
            PcWienerClassifyBlock(frame, cfg, x, y);
            CHECK(lr.px == untouched.px);
            CHECK(grid[(y >> 2) * 8 + (x >> 2)] == static_cast<std::uint8_t>(want));

            Image impl = lr;
            Image ref = lr;
            PcWienerFilterBlock(frame, cfg, x, y, 4, 4);
            (void)impl;
            OracleFilter(oc, ref.px.data(), ref.stride, base_q_idx, x, y, 4, 4);
            // PcWienerFilterBlock wrote into frame.lr == lr.
            for (int yy = 0; yy < 24; ++yy) {
              for (int xx = 0; xx < 32; ++xx) {
                if (lr.At(xx, yy) != ref.At(xx, yy)) {
                  std::fprintf(stderr,
                               "pcw filter bd=%d q=%d x=%d y=%d at %d,%d: %d != %d\n",
                               bd, base_q_idx, x, y, xx, yy, lr.At(xx, yy),
                               ref.At(xx, yy));
                  CHECK(false);
                  return;
                }
                ++filter_checks;
              }
            }
            lr = untouched;
          }
        }
      }
    }
  }
  CHECK(class_checks > 200);
  CHECK(filter_checks > 100000);
}

void TestConstantImage() {
  // A flat image has zero curvature features, so the class depends only on the
  // qindex/tskip terms.  The reference must still be self-consistent across a
  // block split (the class is a function of x, y only).
  Image curr(16, 16);
  Image cdef(16, 16);
  Image lr(16, 16);
  curr.Fill(64);
  cdef.Fill(80);
  lr.Fill(50);
  std::vector<std::uint8_t> tx_skip(4 * 4, 1);
  LumaSource src;
  src.curr = curr.px.data();
  src.cdef = cdef.px.data();
  src.curr_stride = curr.stride;
  src.cdef_stride = cdef.stride;
  src.luma_start_x = 0;
  src.luma_end_x = 15;
  src.luma_start_y = 0;
  src.luma_end_y = 15;
  src.luma_stripe_start_y = 0;
  src.luma_stripe_end_y = 15;
  PcWienerConfig cfg;
  cfg.base_q_idx = 100;
  cfg.bit_depth = 8;
  cfg.mi_col_end = 4;
  cfg.mi_row_start = 0;
  cfg.mi_row_end = 4;
  cfg.lr_tx_skip = tx_skip.data();
  cfg.lr_tx_skip_stride = 4;
  PcWienerFrame frame;
  frame.src = src;
  frame.lr = lr.px.data();
  frame.lr_stride = lr.stride;
  PcWienerFilterBlock(frame, cfg, 0, 0, 8, 8);
  CHECK(lr.At(0, 0) == lr.At(7, 7));  // same class -> same constant offset
}

}  // namespace

int main() {
  TestFilterSetIndex();
  TestAgainstOracle();
  TestConstantImage();
  return Finish("pc_wiener_test");
}
