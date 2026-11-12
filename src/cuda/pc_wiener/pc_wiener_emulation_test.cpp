// Host-emulation parity test for the shared AV2 §7.20.4 PC-Wiener device code.
//
// Compiles pc_wiener_device.cuh with the CUDA keywords stubbed and compares it
// against the CPU reference in src/encoder/filters/spec_tables/pc_wiener.{h,
// cpp}.  Needs no GPU and no nvcc; the real kernel test is pc_wiener_test.cu.
//
// Coverage:
//   * class derivation for 8/10/12 bit, several qindices, stripe boundaries,
//     LrTxSkip tile clamping and the 64-pixel block_end clamp,
//   * skipFilter = 1 (one byte per MI block, LrFrame untouched),
//   * every class 0..63 x every filter set x every bit depth, with an
//     independent 25-tap oracle,
//   * Pc_Wiener_Sub_Classify and Pc_Wiener_Sub_Classify2 for all entries,
//   * get_q / get_qval_given_tskip across the full qlookup range,
//   * stripe bounds and CountUnits/UnitIndexForBlock for 64/128/256/512,
//   * a full-frame class map + filtered plane parity case.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

#include "pc_wiener_device.cuh"

#include "encoder/filters/spec_tables/pc_wiener.h"
#include "encoder/filters/spec_tables/pc_wiener_tables.h"
#include "encoder/filters/spec_tables/spec_test_common.h"

namespace {

using namespace av2enc::cuda::pc_wiener;
using namespace av2enc::filters::spec;
using namespace av2enc::filters::spec_tables;
using av2enc::filters::spec::test::Image;

int g_checks = 0;
int g_failures = 0;
long g_cases = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) std::printf("FAIL: %s\n", what);
  }
}

// Host mirrors of the uploaded table view.
std::int8_t g_taps[kTaps * 2];
std::int32_t g_normalizer[5];
std::int32_t g_mode_weights[4 * 3];
std::int32_t g_mode_offsets[4];

PcWienerTables make_tables() {
  for (int i = 0; i < kTaps; ++i) {
    g_taps[i * 2 + 0] = kPcWienerConfig[i].dy;
    g_taps[i * 2 + 1] = kPcWienerConfig[i].dx;
  }
  for (int i = 0; i < 5; ++i) g_normalizer[i] = kPcWienerNormalizer[i];
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      g_mode_weights[i * 3 + k] = kPcWienerModeWeights[i][k];
    }
    g_mode_offsets[i] = kPcWienerModeOffsets[i];
  }
  PcWienerTables tables{};
  tables.filters = &kPcWienerFilters[0][0][0];
  tables.lut_to_class = kPcWienerLutToClass;
  tables.sub_classify = &kPcWienerSubClassify[0][0];
  tables.sub_classify2 = &kPcWienerSubClassify2[0][0][0];
  tables.qlookup = kQlookup;
  tables.ac_qlookup = kAcQlookup;
  tables.taps = g_taps;
  tables.normalizer = g_normalizer;
  tables.mode_weights = g_mode_weights;
  tables.mode_offsets = g_mode_offsets;
  return tables;
}

// Independent §4.8 helpers (not the device header).
std::int64_t oracle_round2(std::int64_t x, int n) {
  return n == 0 ? x : (x + (std::int64_t{1} << (n - 1))) >> n;
}
std::int64_t oracle_round2_signed(std::int64_t x, int n) {
  return x >= 0 ? oracle_round2(x, n) : -oracle_round2(-x, n);
}
int oracle_max_q(int bd) { return bd == 8 ? 255 : (bd == 10 ? 303 : 351); }

int oracle_get_q(int qindex, int delta, int bd) {
  if (qindex == 0 && delta <= 0) return kAcQlookup[0];
  const int q = std::max(1, std::min(oracle_max_q(bd), qindex + delta));
  return kQlookup[q];
}

std::int64_t oracle_qval(int qindex, std::int64_t tskip, int i, int bd) {
  std::int64_t qstep = oracle_get_q(qindex, 0, bd);
  const int qstep_shift = 13;
  qstep = oracle_round2(qstep, bd - 8);
  const std::int64_t prod = oracle_round2(tskip * qstep, 8);
  const std::int64_t qval = kPcWienerModeWeights[i][0] * (tskip << 5) +
                            kPcWienerModeWeights[i][1] * qstep +
                            kPcWienerModeWeights[i][2] * prod;
  return 255 * (kPcWienerModeOffsets[i] + oracle_round2_signed(qval, qstep_shift));
}

struct Extents {
  int start_x;
  int end_x;
  int start_y;
  int end_y;
  int stripe_start_y;
  int stripe_end_y;
};

void oracle_stripe_bounds(int y, int height, int* start, int* end) {
  const int stripe_num = (y + 8) / 64;
  const int unclipped_start = stripe_num * 64 - 8;
  const int unclipped_end = unclipped_start + 64;
  *start = std::max(0, unclipped_start);
  *end = std::min(height - 1, unclipped_end - 1);
}

PcWienerSource make_source(const Image& curr, const Image& cdef,
                           const Extents& e) {
  PcWienerSource src{};
  src.curr = curr.px.data();
  src.cdef = cdef.px.data();
  src.curr_stride = curr.stride;
  src.cdef_stride = cdef.stride;
  src.luma_start_x = e.start_x;
  src.luma_end_x = e.end_x;
  src.luma_start_y = e.start_y;
  src.luma_end_y = e.end_y;
  src.stripe_start_y = e.stripe_start_y;
  src.stripe_end_y = e.stripe_end_y;
  return src;
}

LumaSource make_luma(const Image& curr, const Image& cdef, const Extents& e) {
  LumaSource src{};
  src.curr = curr.px.data();
  src.cdef = cdef.px.data();
  src.curr_stride = curr.stride;
  src.cdef_stride = cdef.stride;
  src.luma_start_x = e.start_x;
  src.luma_end_x = e.end_x;
  src.luma_start_y = e.start_y;
  src.luma_end_y = e.end_y;
  src.luma_stripe_start_y = e.stripe_start_y;
  src.luma_stripe_end_y = e.stripe_end_y;
  return src;
}

PcWienerParams make_params(int base_q_idx, int bd, int mi_col_end,
                           int mi_row_start, int mi_row_end,
                           const std::vector<std::uint8_t>& tx_skip,
                           int stride) {
  PcWienerParams params{};
  params.base_q_idx = base_q_idx;
  params.bit_depth = bd;
  params.mi_col_end = mi_col_end;
  params.mi_row_start = mi_row_start;
  params.mi_row_end = mi_row_end;
  params.lr_tx_skip = tx_skip.data();
  params.lr_tx_skip_stride = stride;
  return params;
}

// Independent 25-tap filter for an explicit class.
void oracle_filter_class(const Image& curr, const Image& cdef, const Extents& e,
                         int base_q_idx, int cls, int bit_depth, int x, int y,
                         int w, int h, Image* lr) {
  const int index = PcWienerFilterSetIndex(base_q_idx);
  const int filt = kPcWienerSubClassify[index][cls];
  const int maxv = (1 << bit_depth) - 1;
  for (int r2 = 0; r2 < h; ++r2) {
    for (int c2 = 0; c2 < w; ++c2) {
      const int m = av2enc::filters::spec::test::OracleSample(
          curr, cdef, e.start_x, e.end_x, e.start_y, e.end_y, e.stripe_start_y,
          e.stripe_end_y, x + c2, y + r2);
      std::int64_t s = static_cast<std::int64_t>(m) << kPrecBits;
      for (int i = 0; i < kTaps; ++i) {
        const std::int64_t coeff = kPcWienerFilters[index][filt][i >> 1];
        const int sample = av2enc::filters::spec::test::OracleSample(
            curr, cdef, e.start_x, e.end_x, e.start_y, e.end_y,
            e.stripe_start_y, e.stripe_end_y, x + c2 + kPcWienerConfig[i].dx,
            y + r2 + kPcWienerConfig[i].dy);
        s += static_cast<std::int64_t>(sample) * coeff;
      }
      const int v = static_cast<int>(oracle_round2(s, kPrecBits));
      lr->At(x + c2, y + r2) =
          static_cast<std::uint16_t>(std::max(0, std::min(maxv, v)));
    }
  }
}

void test_tables(const PcWienerTables& tables) {
  for (int set = 0; set < kFilterSets; ++set) {
    for (int cls = 0; cls < kClassEntries; ++cls) {
      check(pcw_filter_index(tables, set, cls) ==
                kPcWienerSubClassify[set][cls],
            "Pc_Wiener_Sub_Classify entry");
    }
    for (int idx = 0; idx < kSubClassSets; ++idx) {
      for (int cls = 0; cls < kSubClassEntries; ++cls) {
        check(pcw_subclass(tables, set, idx, cls) ==
                  kPcWienerSubClassify2[set][idx][cls],
              "Pc_Wiener_Sub_Classify2 entry");
      }
    }
  }
  check(kTaps == 25 && kLead == 1 && kLag == 4, "spec constants");
}

void test_get_q(const PcWienerTables& tables) {
  for (int bd : {8, 10, 12}) {
    const int maxq = oracle_max_q(bd);
    for (int q = 0; q <= maxq; ++q) {
      check(pcw_get_q(tables, q, 0, bd) == oracle_get_q(q, 0, bd),
            "get_q matches qlookup");
      for (std::int64_t tskip : {std::int64_t{0}, std::int64_t{1},
                                 std::int64_t{7}, std::int64_t{49},
                                 std::int64_t{252}}) {
        for (int i = 0; i < kNumFeatures; ++i) {
          check(pcw_qval_given_tskip(tables, q, tskip, i, bd) ==
                    oracle_qval(q, tskip, i, bd),
                "get_qval_given_tskip");
          ++g_cases;
        }
      }
    }
  }
}

void test_helpers() {
  for (int height : {1, 63, 64, 72, 160, 1080}) {
    for (int y = 0; y < height; y += 4) {
      int got_start = 0;
      int got_end = 0;
      int want_start = 0;
      int want_end = 0;
      pcw_stripe_bounds(y, height, &got_start, &got_end);
      oracle_stripe_bounds(y, height, &want_start, &want_end);
      check(got_start == want_start && got_end == want_end, "stripe bounds");
    }
  }
  for (int unit : {64, 128, 256, 512}) {
    for (int frame : {64, 100, 640, 1920}) {
      const int want = std::max(1, (frame + unit / 2) / unit);
      check(pcw_count_units(unit, frame) == want, "CountUnits");
      const int rows = want;
      const int cols = want;
      for (int row = 0; row < rows + 2; ++row) {
        for (int col = 0; col < cols + 2; ++col) {
          const int want_row = std::min(rows - 1, (row * 4 + 8) / unit);
          const int want_col = std::min(cols - 1, (col * 4) / unit);
          check(pcw_unit_index(row, col, unit, rows, cols) ==
                    want_row * cols + want_col,
                "UnitIndexForBlock");
          ++g_cases;
        }
      }
    }
  }
}

void test_class_and_skip(const PcWienerTables& tables) {
  constexpr int kWidth = 160;
  constexpr int kHeight = 160;
  constexpr int kMiCols = kWidth / 4;
  constexpr int kMiRows = kHeight / 4;
  av2enc::filters::spec::test::TestRng rng(0x51a7c0deu);

  for (int bd : {8, 10, 12}) {
    const int maxv = (1 << bd) - 1;
    for (int base_q_idx : {0, 1, 64, 90, 129, 130, 189, 190, 219, 220, 255,
                           303, 351}) {
      if (base_q_idx > oracle_max_q(bd)) continue;
      Image curr(kWidth, kHeight);
      Image cdef(kWidth, kHeight);
      Image lr(kWidth, kHeight);
      for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
          curr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
          cdef.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
          lr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
        }
      }
      std::vector<std::uint8_t> tx_skip(
          static_cast<std::size_t>(kMiCols) * kMiRows);
      for (auto& v : tx_skip) {
        v = static_cast<std::uint8_t>(rng.Range(0, 1));
      }

      const int xs[] = {0, 4, 56, 60, 64, 68, 120, 128, 156};
      const int ys[] = {2, 8, 30, 56, 60, 64, 70, 120, 156};
      for (int x : xs) {
        for (int y : ys) {
          Extents e{0, kWidth - 1, 0, kHeight - 1, 0, kHeight - 1};
          oracle_stripe_bounds(y, kHeight, &e.stripe_start_y,
                               &e.stripe_end_y);
          const PcWienerParams params =
              make_params(base_q_idx, bd, kMiCols, 2, kMiRows - 2, tx_skip,
                          kMiCols);
          PcWienerSource src = make_source(curr, cdef, e);

          const int got = pcw_class(tables, src, params, x, y);

          PcWienerFrame frame{};
          frame.src = make_luma(curr, cdef, e);
          frame.lr = lr.px.data();
          frame.lr_stride = lr.stride;
          PcWienerConfig cfg{};
          cfg.base_q_idx = base_q_idx;
          cfg.bit_depth = bd;
          cfg.mi_col_end = kMiCols;
          cfg.mi_row_start = 2;
          cfg.mi_row_end = kMiRows - 2;
          cfg.lr_tx_skip = tx_skip.data();
          cfg.lr_tx_skip_stride = kMiCols;
          const int want = PcWienerClass(frame, cfg, x, y);
          check(got == want, "class parity");
          ++g_cases;

          // skipFilter = 1: class map only, LrFrame untouched.
          Image untouched = lr;
          std::vector<std::uint8_t> got_grid(
              static_cast<std::size_t>(kMiCols) * kMiRows, 0);
          std::vector<std::uint8_t> want_grid(
              static_cast<std::size_t>(kMiCols) * kMiRows, 0);
          frame.filter_class = want_grid.data();
          frame.filter_class_stride = kMiCols;
          pcw_classify_block(tables, src, params, x, y, got_grid.data(),
                             kMiCols);
          PcWienerClassifyBlock(frame, cfg, x, y);
          check(got_grid == want_grid, "skipFilter=1 class map");
          check(lr.px == untouched.px, "skipFilter=1 leaves LrFrame untouched");
          ++g_cases;

          // skipFilter = 0 for a 4x4 block, class derived from (x, y).
          Image got_lr = lr;
          const int w = std::min(4, kWidth - x);
          const int h = std::min(4, kHeight - y);
          pcw_filter_block(tables, src, params, x, y, w, h, got_lr.px.data(),
                           got_lr.stride);
          PcWienerFilterBlock(frame, cfg, x, y, w, h);
          for (int r2 = 0; r2 < h; ++r2) {
            for (int c2 = 0; c2 < w; ++c2) {
              check(got_lr.At(x + c2, y + r2) == lr.At(x + c2, y + r2),
                    "filter derived-class parity");
            }
          }
          ++g_cases;
        }
      }
    }
  }
}

void test_every_class(const PcWienerTables& tables) {
  constexpr int kWidth = 40;
  constexpr int kHeight = 32;
  av2enc::filters::spec::test::TestRng rng(0x0badf00du);
  const int q_for_set[4] = {0, 130, 190, 220};

  for (int bd : {8, 10, 12}) {
    const int maxv = (1 << bd) - 1;
    Image curr(kWidth, kHeight);
    Image cdef(kWidth, kHeight);
    Image lr(kWidth, kHeight);
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        curr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
        cdef.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
        lr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
      }
    }
    std::vector<std::uint8_t> tx_skip(10 * 8, 1);
    Extents e{0, kWidth - 1, 0, kHeight - 1, 0, kHeight - 1};

    for (int set = 0; set < kFilterSets; ++set) {
      const int base_q_idx = q_for_set[set];
      const PcWienerParams params =
          make_params(base_q_idx, bd, 10, 0, 8, tx_skip, 10);
      PcWienerSource src = make_source(curr, cdef, e);
      for (int cls = 0; cls < 64; ++cls) {
        const int sizes[][4] = {{0, 0, 4, 4}, {5, 4, 7, 3}, {33, 27, 4, 4}};
        for (const auto& size : sizes) {
          Image got = lr;
          Image want = lr;
          pcw_filter_block_class(tables, src, params, size[0], size[1],
                                 size[2], size[3], cls, got.px.data(),
                                 got.stride);
          oracle_filter_class(curr, cdef, e, base_q_idx, cls, bd, size[0],
                              size[1], size[2], size[3], &want);
          for (int r2 = 0; r2 < size[3]; ++r2) {
            for (int c2 = 0; c2 < size[2]; ++c2) {
              check(got.At(size[0] + c2, size[1] + r2) ==
                        want.At(size[0] + c2, size[1] + r2),
                    "explicit-class filter parity");
            }
          }
          ++g_cases;
        }
      }
    }
  }
}

void test_full_frame(const PcWienerTables& tables) {
  constexpr int kWidth = 96;
  constexpr int kHeight = 72;
  constexpr int kMiCols = kWidth / 4;
  constexpr int kMiRows = kHeight / 4;
  av2enc::filters::spec::test::TestRng rng(0xfeedfaceu);

  for (int bd : {8, 10, 12}) {
    const int maxv = (1 << bd) - 1;
    Image curr(kWidth, kHeight);
    Image cdef(kWidth, kHeight);
    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        curr.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
        cdef.At(x, y) = static_cast<std::uint16_t>(rng.Range(0, maxv));
      }
    }
    std::vector<std::uint8_t> tx_skip(
        static_cast<std::size_t>(kMiCols) * kMiRows);
    for (auto& v : tx_skip) v = static_cast<std::uint8_t>(rng.Range(0, 1));

    for (int base_q_idx : {0, 140, 200, 240}) {
      if (base_q_idx > oracle_max_q(bd)) continue;
      Image cpu_lr(kWidth, kHeight);
      Image dev_lr(kWidth, kHeight);
      cpu_lr.Fill(7);
      dev_lr.Fill(7);
      std::vector<std::uint8_t> cpu_grid(
          static_cast<std::size_t>(kMiCols) * kMiRows, 0);
      std::vector<std::uint8_t> dev_grid(
          static_cast<std::size_t>(kMiCols) * kMiRows, 0);

      PcWienerFrame frame{};
      frame.lr = cpu_lr.px.data();
      frame.lr_stride = cpu_lr.stride;
      frame.filter_class = cpu_grid.data();
      frame.filter_class_stride = kMiCols;
      PcWienerConfig cfg{};
      cfg.base_q_idx = base_q_idx;
      cfg.bit_depth = bd;
      cfg.mi_col_end = kMiCols;
      cfg.mi_row_start = 0;
      cfg.mi_row_end = kMiRows;
      cfg.lr_tx_skip = tx_skip.data();
      cfg.lr_tx_skip_stride = kMiCols;

      for (int row = 0; row < kMiRows; ++row) {
        Extents e{0, kWidth - 1, 0, kHeight - 1, 0, kHeight - 1};
        oracle_stripe_bounds(row * 4, kHeight, &e.stripe_start_y,
                             &e.stripe_end_y);
        frame.src = make_luma(curr, cdef, e);
        PcWienerSource src = make_source(curr, cdef, e);
        const PcWienerParams params =
            make_params(base_q_idx, bd, kMiCols, 0, kMiRows, tx_skip, kMiCols);
        for (int col = 0; col < kMiCols; ++col) {
          const int x = col * 4;
          const int y = row * 4;
          const int cls = PcWienerClass(frame, cfg, x, y);
          cpu_grid[static_cast<std::size_t>(row) * kMiCols + col] =
              static_cast<std::uint8_t>(cls);
          dev_grid[static_cast<std::size_t>(row) * kMiCols + col] =
              static_cast<std::uint8_t>(pcw_class(tables, src, params, x, y));
          PcWienerFilterBlock(frame, cfg, x, y, 4, 4);
          pcw_filter_block(tables, src, params, x, y, 4, 4, dev_lr.px.data(),
                           dev_lr.stride);
          ++g_cases;
        }
      }
      check(cpu_grid == dev_grid, "full-frame class map parity");
      check(cpu_lr.px == dev_lr.px, "full-frame filtered plane parity");
      ++g_cases;
    }
  }
}

}  // namespace

int main() {
  const PcWienerTables tables = make_tables();
  test_tables(tables);
  test_get_q(tables);
  test_helpers();
  test_class_and_skip(tables);
  test_every_class(tables);
  test_full_frame(tables);

  if (g_failures == 0) {
    std::printf("pc_wiener_emulation_test: %ld cases, %d checks, 0 failures\n",
                g_cases, g_checks);
    return 0;
  }
  std::fprintf(stderr, "pc_wiener_emulation_test: %d/%d checks FAILED\n",
               g_failures, g_checks);
  return 1;
}
