// CPU reference tests for AV2 quantizer matrices (§9.4, §5.4.11, §5.13,
// §5.18.6.2, §7.14.4).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "encoder/qm/qm.h"
#include "encoder/qm/qm_tables.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(condition)) {                                                      \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                   #condition);                                              \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++g_checks;                                                              \
    const auto va = (a);                                                     \
    const auto vb = (b);                                                     \
    if (!(va == vb)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, \
                   __LINE__, #a, #b, static_cast<long long>(va),              \
                   static_cast<long long>(vb));                              \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

using av2enc::BitWriter;
using av2enc::qm::QmDequantParams;
using av2enc::qm::QmFrameState;
using av2enc::qm::UserQmLevel;

// ---------------------------------------------------------------------------
// Minimal MSB-first reader for syntax round-trip checks.
// ---------------------------------------------------------------------------
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}
  bool bit() {
    if (pos_ >= data_.size() * 8) {
      overflow_ = true;
      return false;
    }
    const std::uint8_t b = data_[pos_ / 8];
    const bool value = ((b >> (7 - (pos_ % 8))) & 1U) != 0U;
    ++pos_;
    return value;
  }
  std::uint32_t bits(unsigned n) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < n; ++i) {
      value = (value << 1U) | (bit() ? 1U : 0U);
    }
    return value;
  }
  std::uint32_t uvlc() {
    unsigned zeros = 0;
    while (!bit()) {
      if (++zeros >= 32) {
        overflow_ = true;
        return 0;
      }
    }
    return bits(zeros) + (1U << zeros) - 1U;
  }
  int svlc() {
    const std::uint32_t value = uvlc();
    const int half = static_cast<int>((value + 1U) >> 1U);
    return (value & 1U) != 0U ? half : -half;
  }
  bool overflow() const { return overflow_; }
  std::size_t position() const { return pos_; }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t pos_ = 0;
  bool overflow_ = false;
};

void diagonal_scan(int w, int h, std::uint16_t* scan) {
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
}

void fundamental_dims(int t, int* w, int* h) {
  if (t == 0) {
    *w = 8;
    *h = 8;
  } else if (t == 1) {
    *w = 8;
    *h = 4;
  } else {
    *w = 4;
    *h = 8;
  }
}

void parse_user_qm(BitReader& reader, UserQmLevel& user, int t, int plane,
                   int num_planes) {
  int w = 0;
  int h = 0;
  fundamental_dims(t, &w, &h);
  if (plane > 0) {
    const bool copy = reader.bit();
    if (copy) {
      std::memcpy(user.m[t][plane], user.m[t][plane - 1],
                  sizeof(user.m[t][plane]));
      return;
    }
  }
  bool symmetric = false;
  if (t == 0) {
    symmetric = reader.bit();
  } else if (t == 2) {
    const bool transpose = reader.bit();
    if (transpose) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          user.m[2][plane][i][j] = user.m[1][plane][j][i];
        }
      }
      return;
    }
  }
  std::uint16_t scan[64];
  diagonal_scan(w, h, scan);
  int quant = 32;
  bool repeat = false;
  for (int c = 0; c < w * h; ++c) {
    const int pos = scan[c];
    const int row = pos / w;
    const int col = pos % w;
    if (t == 0 && symmetric && col > row) {
      quant = user.m[0][plane][col][row];
      user.m[0][plane][row][col] = static_cast<std::uint8_t>(quant);
    } else if (repeat) {
      user.m[t][plane][row][col] = static_cast<std::uint8_t>(quant);
    } else {
      const int delta = reader.svlc();
      const int quant2 = (quant + delta) & 255;
      if (quant2 == 0) {
        repeat = true;
      } else {
        quant = quant2;
      }
      user.m[t][plane][row][col] = static_cast<std::uint8_t>(quant);
    }
  }
  (void)num_planes;
}

// Independent §7.14.4 expectation.  This is written from the spec text, not
// from qm_core.h: steps 3-8 of the dequantization process plus the §4.8
// Clip3 bound.
std::int64_t expected_clip(std::int64_t value, int bit_depth) {
  const std::int64_t bound =
      static_cast<std::int64_t>(1) << (7 + bit_depth);
  if (value < -bound) return -bound;
  if (value > bound - 1) return bound - 1;
  return value;
}

std::int32_t expected_dequant(std::int32_t level, std::int32_t q2, int denom,
                              int bit_depth = 8) {
  const std::int64_t magnitude =
      level < 0 ? -static_cast<std::int64_t>(level) : level;
  const std::int64_t high = magnitude * q2;
  const std::int64_t masked = high & 0xFFFFFFLL;
  const std::int64_t rounded = (masked + 4) >> 3;  // Round2(., 3)
  const std::int64_t value = rounded / denom;
  const std::int64_t signed_value = level < 0 ? -value : value;
  return static_cast<std::int32_t>(expected_clip(signed_value, bit_depth));
}

// §7.14.4 uses get_tx_class(PlaneTxType) == TX_CLASS_2D.  Only the six
// directional types are VERT/HORIZ; IDTX (9) and every 2-D type are 2D.
bool expected_tx_class_2d(int tx_type) {
  switch (tx_type) {
    case 10:  // V_DCT
    case 11:  // H_DCT
    case 12:  // V_ADST
    case 13:  // H_ADST
    case 14:  // V_FLIPADST
    case 15:  // H_FLIPADST
      return false;
    default:
      return true;
  }
}

int expected_denom(int tx_size, int plane, bool allow_tcq, bool lossless,
                   int tx_type) {
  const int w = av2enc::qm::tx_width(tx_size);
  const int h = av2enc::qm::tx_height(tx_size);
  const int pels = w * h;
  int shift = (pels > 256 ? 1 : 0) + (pels > 1024 ? 1 : 0);
  if (allow_tcq && plane == 0 && !lossless &&
      expected_tx_class_2d(tx_type)) {
    ++shift;
  }
  return 1 << shift;
}

// ---------------------------------------------------------------------------
// Table integrity
// ---------------------------------------------------------------------------
void test_tables() {
  using namespace av2enc::qm;
  const std::uint16_t expected_offsets[25] = {
      0, 16, 80, 336, 336, 1360, 1392, 1424, 1552, 1680, 2192, 336, 336,
      2704, 2768, 2832, 3088, 1680, 2192, 3344, 3472, 2832, 3088, 3344,
      3472};
  for (int i = 0; i < 25; ++i) {
    CHECK_EQ(qm_offset(i), expected_offsets[i]);
    CHECK(valid_tx_size(i));
    const int tw = adjusted_width(i);
    const int th = adjusted_height(i);
    CHECK(tw >= 4 && tw <= 32 && th >= 4 && th <= 32);
    CHECK_EQ(tw, std::min(32, tx_width(i)));
    CHECK_EQ(th, std::min(32, tx_height(i)));
  }
  CHECK(!valid_tx_size(-1));
  CHECK(!valid_tx_size(25));

  // Golden §9.4.3 entries (level 0, both plane types).
  const std::uint8_t luma0[16] = {32, 43, 73, 97, 43, 67, 94,
                                  110, 73, 94, 137, 150, 97, 110, 150, 200};
  const std::uint8_t chroma0[16] = {35, 46, 57, 66, 46, 60,
                                    69, 71, 57, 69, 90, 90,
                                    66, 71, 90, 109};
  for (int i = 0; i < 16; ++i) {
    CHECK_EQ(default_matrix_value(0, 0, 0, i / 4, i % 4), luma0[i]);
    CHECK_EQ(default_matrix_value(0, 1, 0, i / 4, i % 4), chroma0[i]);
  }
  CHECK_EQ(default_matrix_value(0, 0, 1, 0, 0), 32);
  CHECK_EQ(default_matrix_value(0, 0, 1, 7, 7), 220);
  CHECK_EQ(default_matrix_value(14, 0, 3, 0, 0), 32);
  CHECK_EQ(default_matrix_value(14, 1, 24, 0, 0),
           kQuantizerMatrix[14][1][kQmOffset[24]]);
  CHECK_EQ(default_matrix_value(15, 0, 0, 0, 0), 0);
  CHECK_EQ(default_matrix_value(0, 0, 0, -1, 0), 0);

  // Every level/plane/TX-size block must be addressable and inside 3600.
  for (int level = 0; level < kNumQmLevels; ++level) {
    for (int plane = 0; plane < 2; ++plane) {
      for (int tx = 0; tx < 25; ++tx) {
        const int tw = adjusted_width(tx);
        const int th = adjusted_height(tx);
        const std::uint16_t start = kQmOffset[tx];
        CHECK(start + tw * th <= kQmTableEntries);
        CHECK_EQ(default_matrix_value(level, plane, tx, 0, 0),
                 kQuantizerMatrix[level][plane][start]);
        CHECK_EQ(default_matrix_value(level, plane, tx, th - 1, tw - 1),
                 kQuantizerMatrix[level][plane][start + tw * th - 1]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.14.4 dequantization
// ---------------------------------------------------------------------------
void test_dequant_no_qm() {
  using namespace av2enc::qm;
  QmFrameState state;
  QmDequantParams params;
  params.state = &state;
  params.tx_size = 0;  // 4x4
  params.plane = 0;
  params.tx_type = 0;
  params.dc_quant = 64;
  params.ac_quant = 132;

  const std::int32_t levels[16] = {7, -3, 0, 1, 2, -5, 9, 0,
                                   1, 1, -1, 2, 3, 4, -6, 8};
  std::int32_t dequant[16] = {};
  CHECK(dequantize_block(levels, params, dequant));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const std::int32_t q = (i == 0 && j == 0) ? 64 : 132;
      CHECK_EQ(dequant[i * 4 + j],
               expected_dequant(levels[i * 4 + j], q, 1));
    }
  }

  std::int32_t q2[16] = {};
  CHECK(build_q2_matrix(params, q2));
  CHECK_EQ(q2[0], 64);
  CHECK_EQ(q2[1], 132);
}

void test_dequant_all_sizes_and_denoms() {
  using namespace av2enc::qm;
  QmFrameState state;
  state.using_qmatrix = true;
  for (int level = 0; level < kNumQmLevels; ++level) {
    state.seg_qm_level[0][0] = static_cast<std::uint8_t>(level);
    state.qm_y[0] = static_cast<std::uint8_t>(level);
    for (int tx = 0; tx < 25; ++tx) {
      const int tw = adjusted_width(tx);
      const int th = adjusted_height(tx);
      const int n = tw * th;
      std::vector<std::int32_t> levels(static_cast<std::size_t>(n));
      std::vector<std::int32_t> dequant(static_cast<std::size_t>(n));
      std::vector<std::int32_t> q2(static_cast<std::size_t>(n));
      for (int k = 0; k < n; ++k) {
        levels[static_cast<std::size_t>(k)] =
            (k % 7 == 0) ? 0 : ((k % 2 == 0) ? k + 1 : -(k + 1));
      }
      QmDequantParams params;
      params.state = &state;
      params.tx_size = tx;
      params.plane = 0;
      params.tx_type = 0;
      params.dc_quant = 64;
      params.ac_quant = 132;
      CHECK(build_q2_matrix(params, q2.data()));
      CHECK(dequantize_block(levels.data(), params, dequant.data()));
      const int denom = expected_denom(tx, 0, false, false, 0);
      for (int i = 0; i < th; ++i) {
        for (int j = 0; j < tw; ++j) {
          const int idx = i * tw + j;
          const int q = (i == 0 && j == 0) ? 64 : 132;
          const int m = default_matrix_value(level, 0, tx, i, j);
          const std::int32_t want_q2 =
              static_cast<std::int32_t>((q * m + 16) >> 5);
          CHECK_EQ(q2[static_cast<std::size_t>(idx)], want_q2);
          CHECK_EQ(dequant[static_cast<std::size_t>(idx)],
                   expected_dequant(levels[static_cast<std::size_t>(idx)],
                                    want_q2, denom));
        }
      }
    }
  }
}

void test_level_gating_and_planes() {
  using namespace av2enc::qm;
  QmFrameState state;
  state.using_qmatrix = true;
  state.qm_y[0] = 3;
  state.qm_u[0] = 5;
  state.qm_v[0] = 7;
  for (int plane = 0; plane < 3; ++plane) {
    QmDequantParams params;
    params.state = &state;
    params.tx_size = 1;  // 8x8: uses SegQMLevel
    params.plane = plane;
    params.tx_type = 0;
    params.segment_id = 0;
    params.dc_quant = 64;
    params.ac_quant = 132;
    state.seg_qm_level[plane][0] = 9;
    CHECK_EQ(resolve_seg_level(state, plane, 1, 0), 9);
    CHECK_EQ(resolve_seg_level(state, plane, 3, 0),
             plane == 0 ? 3 : (plane == 1 ? 5 : 7));
    std::int32_t q2[64];
    CHECK(build_q2_matrix(params, q2));
    const int m = default_matrix_value(9, plane > 0 ? 1 : 0, 1, 0, 1);
    CHECK_EQ(q2[1], (132 * m + 16) >> 5);
  }

  // Level 15 means QM off.
  state.using_qmatrix = true;
  state.seg_qm_level[0][0] = 15;
  QmDequantParams params;
  params.state = &state;
  params.tx_size = 1;
  params.plane = 0;
  params.tx_type = 0;
  params.dc_quant = 64;
  params.ac_quant = 132;
  std::int32_t q2[64];
  CHECK(build_q2_matrix(params, q2));
  CHECK_EQ(q2[0], 64);
  CHECK_EQ(q2[1], 132);

  // IDTX and 1-D TX types never use a QM.
  for (int tx_type : {9, 10, 11, 12, 13, 14, 15}) {
    params.tx_type = tx_type;
    state.seg_qm_level[0][0] = 2;
    CHECK(build_q2_matrix(params, q2));
    CHECK_EQ(q2[1], 132);
  }
  // 2-D types do.
  for (int tx_type = 0; tx_type < 9; ++tx_type) {
    params.tx_type = tx_type;
    state.seg_qm_level[0][0] = 2;
    CHECK(build_q2_matrix(params, q2));
    const int m = default_matrix_value(2, 0, 1, 0, 1);
    CHECK_EQ(q2[1], (132 * m + 16) >> 5);
  }

  // using_qmatrix = 0 disables everything.
  state.using_qmatrix = false;
  params.tx_type = 0;
  CHECK(build_q2_matrix(params, q2));
  CHECK_EQ(q2[1], 132);

  // Lossless segment: SegQMLevel = 15.
  state.using_qmatrix = true;
  state.lossless_segment[0] = true;
  CHECK_EQ(resolve_seg_level(state, 0, 1, 0), 15);
}

void test_allow_tcq_shift() {
  using namespace av2enc::qm;
  QmFrameState state;
  QmDequantParams params;
  params.state = &state;
  params.plane = 0;
  params.dc_quant = 64;
  params.ac_quant = 132;

  for (int tx : {1, 2, 3, 4}) {  // 8x8, 16x16, 32x32, 64x64
    params.tx_size = tx;
    const int n = adjusted_width(tx) * adjusted_height(tx);
    std::vector<std::int32_t> levels(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> out(static_cast<std::size_t>(n), 0);
    levels[1] = 5;

    params.tx_type = 0;  // DCT_DCT, TX_CLASS_2D
    params.allow_tcq = false;
    CHECK(dequantize_block(levels.data(), params, out.data()));
    const int denom_no = expected_denom(tx, 0, false, false, 0);
    CHECK_EQ(out[1], expected_dequant(5, 132, denom_no));

    params.allow_tcq = true;
    CHECK(dequantize_block(levels.data(), params, out.data()));
    const int denom_tcq = expected_denom(tx, 0, true, false, 0);
    CHECK_EQ(out[1], expected_dequant(5, 132, denom_tcq));
    CHECK(denom_tcq == denom_no * 2);

    // Chroma gets no TCQ shift.
    params.plane = 1;
    CHECK(dequantize_block(levels.data(), params, out.data()));
    CHECK_EQ(out[1], expected_dequant(5, 132, denom_no));
    params.plane = 0;

    // §7.14.4 says get_tx_class(PlaneTxType) == TX_CLASS_2D, and IDTX is 2D.
    // QM itself does not apply to IDTX (PlaneTxType < IDTX fails), but the
    // TCQ dqDenom shift does.
    params.tx_type = 9;  // IDTX
    CHECK(dequantize_block(levels.data(), params, out.data()));
    CHECK_EQ(out[1], expected_dequant(5, 132, denom_tcq));

    // The six V_*/H_* directional types are TX_CLASS_VERT/HORIZ: no shift.
    for (int tx_type : {10, 11, 12, 13, 14, 15}) {
      params.tx_type = tx_type;
      CHECK(dequantize_block(levels.data(), params, out.data()));
      CHECK_EQ(out[1], expected_dequant(5, 132, denom_no));
    }

    // Lossless also disables the TCQ shift.
    params.tx_type = 9;
    params.lossless = true;
    CHECK(dequantize_block(levels.data(), params, out.data()));
    CHECK_EQ(out[1], expected_dequant(5, 132, denom_no));
    params.lossless = false;

    // FSC with IDTX also disables it.
    params.use_fsc = true;
    CHECK(dequantize_block(levels.data(), params, out.data()));
    CHECK_EQ(out[1], expected_dequant(5, 132, denom_no));
    params.use_fsc = false;
  }
}

void test_dequant_clip_8_10bit() {
  using namespace av2enc::qm;
  // No QM, 4x4, dqDenom = 1, q = 528.  §7.14.4 step 6 rounds by 3 bits, so
  // the unclipped value is level * 528 / 8 = level * 66.
  const std::int32_t q = 528;
  const std::int32_t levels[16] = {
      496,  497,  -496, -497, 1985, 1986, -1985, -1986,
      1000, -1000, 0,    1,    -1,   2,    -2,    3};
  for (int bit_depth : {8, 10}) {
    QmFrameState state;
    QmDequantParams params;
    params.state = &state;
    params.tx_size = 0;
    params.plane = 0;
    params.tx_type = 0;
    params.dc_quant = q;
    params.ac_quant = q;
    params.bit_depth = bit_depth;
    std::int32_t dequant[16] = {};
    CHECK(dequantize_block(levels, params, dequant));
    for (int i = 0; i < 16; ++i) {
      CHECK_EQ(dequant[i],
               expected_dequant(levels[i], q, 1, bit_depth));
    }
    const std::int64_t bound =
        static_cast<std::int64_t>(1) << (7 + bit_depth);
    // Explicit bound checks, not just the shared reference.
    if (bit_depth == 8) {
      CHECK_EQ(dequant[0], 496 * 66);                               // 32736
      CHECK_EQ(dequant[1], static_cast<std::int32_t>(bound - 1));   // +32767
      CHECK_EQ(dequant[3], static_cast<std::int32_t>(-bound));      // -32768
      CHECK_EQ(dequant[9], static_cast<std::int32_t>(-bound));      // -66000
    } else {
      CHECK_EQ(dequant[4], 1985 * 66);                              // 131010
      CHECK_EQ(dequant[5], static_cast<std::int32_t>(bound - 1));   // +131071
      CHECK_EQ(dequant[7], static_cast<std::int32_t>(-bound));      // -131072
      CHECK_EQ(dequant[9], -1000 * 66);                             // no clip
    }
    CHECK(dequant[8] > 0);
  }
}

// ---------------------------------------------------------------------------
// User-defined QM
// ---------------------------------------------------------------------------
void test_user_qm_selection() {
  using namespace av2enc::qm;
  QmFrameState state;
  state.using_qmatrix = true;
  for (int level = 0; level < 15; ++level) {
    CHECK(fill_user_qm_constant(&state.user_qm[level],
                                static_cast<std::uint8_t>(16 + level)));
    state.qm_data_present[level] = true;
  }
  state.qm_y[0] = 2;
  state.seg_qm_level[0][0] = 2;

  QmDequantParams params;
  params.state = &state;
  params.plane = 0;
  params.tx_type = 0;
  params.dc_quant = 64;
  params.ac_quant = 132;

  // Constant 18 identity for level 2: q2 = (q*18+16)>>5.
  for (int tx : {0, 1, 5, 6}) {  // 4x4, 8x8, 4x8, 8x4
    const int tw = adjusted_width(tx);
    const int th = adjusted_height(tx);
    params.tx_size = tx;
    params.segment_id = 0;
    CHECK(uses_user_qm(state, 0, tx, 2));
    std::int32_t q2[1024];
    CHECK(build_q2_matrix(params, q2));
    for (int i = 0; i < th; ++i) {
      for (int j = 0; j < tw; ++j) {
        const int q = (i == 0 && j == 0) ? 64 : 132;
        CHECK_EQ(q2[i * tw + j], (q * 18 + 16) >> 5);
      }
    }
  }

  // All-32 is identity.
  CHECK(fill_user_qm_constant(&state.user_qm[2], 32));
  for (int tx : {0, 1, 5, 6}) {
    const int tw = adjusted_width(tx);
    const int th = adjusted_height(tx);
    params.tx_size = tx;
    std::int32_t q2[1024];
    CHECK(build_q2_matrix(params, q2));
    for (int i = 0; i < th; ++i) {
      for (int j = 0; j < tw; ++j) {
        const int q = (i == 0 && j == 0) ? 64 : 132;
        CHECK_EQ(q2[i * tw + j], q);
      }
    }
  }

  // 8x16 has th=16 > 8, so it falls back to the default 8x16 matrix.
  params.tx_size = 7;  // 8x16
  CHECK(!uses_user_qm(state, 0, 7, 2));
  std::int32_t q2[1024];
  CHECK(build_q2_matrix(params, q2));
  const int m = default_matrix_value(2, 0, 7, 0, 1);
  CHECK_EQ(q2[1], (132 * m + 16) >> 5);

  // A level without QmDataPresent uses the default table.
  state.qm_data_present[2] = false;
  CHECK(!uses_user_qm(state, 0, 1, 2));
  CHECK(fill_user_qm_constant(&state.user_qm[3], 32));
  state.qm_y[0] = 3;
  params.tx_size = 3;  // 32x32 -> >8 path uses qm_y[0] == 3
  CHECK(build_q2_matrix(params, q2));
  // User QM is defined only for <= 8x8, so 32x32 uses the default matrix.
  CHECK(!uses_user_qm(state, 0, 3, 3));
  const int m32 = default_matrix_value(3, 0, 3, 0, 1);
  CHECK_EQ(q2[1], (132 * m32 + 16) >> 5);
}

void make_random_user_level(UserQmLevel* level, std::mt19937* rng) {
  std::uniform_int_distribution<int> dist(1, 255);
  for (int t = 0; t < 3; ++t) {
    for (int plane = 0; plane < 3; ++plane) {
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          level->m[t][plane][i][j] =
              static_cast<std::uint8_t>(dist(*rng));
        }
      }
    }
  }
}

void test_user_qm_serialization() {
  using namespace av2enc::qm;
  std::mt19937 rng(12345);
  UserQmLevel user;
  make_random_user_level(&user, &rng);

  // Make t=1 and t=2 a transpose pair so the t=2 shortcut is exercised.
  for (int plane = 0; plane < 3; ++plane) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 4; ++j) {
        user.m[2][plane][i][j] = user.m[1][plane][j][i];
      }
    }
  }
  // Force 8x8 symmetric for luma.
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      user.m[0][0][i][j] = user.m[0][0][j][i];
    }
  }
  // Keep plane 1's 4x8 matrix non-transposed so the full-matrix path is
  // exercised too.
  user.m[2][1][0][0] = static_cast<std::uint8_t>(
      (user.m[1][1][0][0] + 1) & 255);
  if (user.m[2][1][0][0] == 0) user.m[2][1][0][0] = 2;

  QmObuConfig config;
  config.bit_map = 1U << 0;
  config.chroma_info_present = true;
  config.levels[0].is_default = false;
  config.levels[0].user = user;

  BitWriter writer;
  std::string error;
  CHECK(write_quantizer_matrix_obu_payload(writer, config, &error));
  CHECK(error.empty());

  BitReader reader(writer.bytes());
  CHECK_EQ(reader.bits(15), 1U);
  CHECK_EQ(reader.bit(), true);
  CHECK_EQ(reader.bit(), false);  // qm_is_default_flag
  UserQmLevel parsed;
  for (int t = 0; t < 3; ++t) {
    for (int plane = 0; plane < 3; ++plane) {
      parse_user_qm(reader, parsed, t, plane, 3);
    }
  }
  CHECK(!reader.overflow());
  for (int t = 0; t < 3; ++t) {
    int w = 0;
    int h = 0;
    fundamental_dims(t, &w, &h);
    for (int plane = 0; plane < 3; ++plane) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          CHECK_EQ(parsed.m[t][plane][i][j], user.m[t][plane][i][j]);
        }
      }
    }
  }
  CHECK_EQ(reader.bit(), true);  // trailing one bit
}

void test_qm_obu_writer() {
  using namespace av2enc::qm;
  QmObuConfig config;
  config.bit_map = 0;
  config.chroma_info_present = false;
  BitWriter writer;
  std::string error;
  CHECK(write_quantizer_matrix_obu_payload(writer, config, &error));
  BitReader reader(writer.bytes());
  CHECK_EQ(reader.bits(15), 0U);
  CHECK_EQ(reader.bit(), false);
  CHECK_EQ(reader.bit(), true);

  // Default level 4 and user level 7 with chroma info.
  QmObuConfig config2;
  config2.bit_map = (1U << 4) | (1U << 7);
  config2.chroma_info_present = true;
  config2.levels[4].is_default = true;
  config2.levels[7].is_default = false;
  CHECK(fill_user_qm_constant(&config2.levels[7].user, 32));
  BitWriter writer2;
  CHECK(write_quantizer_matrix_obu_payload(writer2, config2, &error));
  BitReader reader2(writer2.bytes());
  CHECK_EQ(reader2.bits(15), config2.bit_map);
  CHECK_EQ(reader2.bit(), true);
  CHECK_EQ(reader2.bit(), true);  // level 4 is default
  CHECK_EQ(reader2.bit(), false); // level 7 has data
  UserQmLevel parsed;
  for (int t = 0; t < 3; ++t) {
    for (int plane = 0; plane < 3; ++plane) {
      parse_user_qm(reader2, parsed, t, plane, 3);
    }
  }
  CHECK(!reader2.overflow());
  for (int t = 0; t < 3; ++t) {
    int w = 0;
    int h = 0;
    fundamental_dims(t, &w, &h);
    for (int plane = 0; plane < 3; ++plane) {
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          CHECK_EQ(parsed.m[t][plane][i][j], 32);
        }
      }
    }
  }

  // Complete OBU: leb128 length + header byte with type 22.
  std::string obu_error;
  const std::vector<std::uint8_t> obu =
      make_quantizer_matrix_obu(config2, &obu_error);
  CHECK(!obu.empty());
  // leb128 length prefix (one byte for this payload) then obu_header.
  CHECK(obu.size() > 2);
  CHECK_EQ(obu[0] & 0x80, 0);
  CHECK_EQ(static_cast<unsigned>((obu[1] >> 2) & 0x1fU), 22U);
  CHECK_EQ(static_cast<unsigned>(obu[1] & 3U), 0U);  // tlayer 0
}

void test_setup_qm_params() {
  using namespace av2enc::qm;
  QmFrameState state;
  state.using_qmatrix = true;
  state.pic_qm_num_minus_1 = 0;
  state.qm_y[0] = 2;
  state.qm_u[0] = 2;
  state.qm_v[0] = 2;
  BitWriter writer;
  std::string error;
  CHECK(write_setup_qm_params(writer, state, 3, false, &error));
  BitReader reader(writer.bytes());
  CHECK_EQ(reader.bit(), true);
  CHECK_EQ(reader.bits(4), 2U);
  CHECK_EQ(reader.bit(), true);  // qm_uv_same_as_y
  CHECK_EQ(reader.position(), 6U);

  // Distinct U/V, separate_uv_delta_q = 1.
  state.qm_y[0] = 1;
  state.qm_u[0] = 2;
  state.qm_v[0] = 3;
  state.separate_uv_delta_q = true;
  BitWriter writer2;
  CHECK(write_setup_qm_params(writer2, state, 3, false, &error));
  BitReader reader2(writer2.bytes());
  CHECK_EQ(reader2.bit(), true);
  CHECK_EQ(reader2.bits(4), 1U);
  CHECK_EQ(reader2.bit(), false);
  CHECK_EQ(reader2.bits(4), 2U);
  CHECK_EQ(reader2.bits(4), 3U);

  // Without separate_uv_delta_q, qm_v must follow qm_u.
  state.qm_v[0] = 2;
  state.separate_uv_delta_q = false;
  BitWriter writer3;
  CHECK(write_setup_qm_params(writer3, state, 3, false, &error));
  BitReader reader3(writer3.bytes());
  CHECK_EQ(reader3.bit(), true);
  CHECK_EQ(reader3.bits(4), 1U);
  CHECK_EQ(reader3.bit(), false);
  CHECK_EQ(reader3.bits(4), 2U);
  CHECK_EQ(reader3.position(), 10U);

  state.qm_v[0] = 3;
  BitWriter writer4;
  CHECK(!write_setup_qm_params(writer4, state, 3, false, &error));
  CHECK(!error.empty());

  // Segmentation enabled codes pic_qm_num_minus_1.
  state.pic_qm_num_minus_1 = 2;
  state.qm_y[1] = 4;
  state.qm_u[1] = 5;
  state.qm_v[1] = 5;
  state.qm_y[2] = 6;
  state.qm_u[2] = 6;
  state.qm_v[2] = 6;
  state.qm_v[0] = 2;
  state.separate_uv_delta_q = false;
  BitWriter writer5;
  CHECK(write_setup_qm_params(writer5, state, 3, true, &error));
  BitReader reader5(writer5.bytes());
  CHECK_EQ(reader5.bit(), true);
  CHECK_EQ(reader5.bits(2), 2U);
  CHECK_EQ(reader5.bits(4), 1U);
  CHECK_EQ(reader5.bit(), false);
  CHECK_EQ(reader5.bits(4), 2U);

  // using_qmatrix = 0 writes exactly one zero bit.
  QmFrameState off;
  BitWriter writer6;
  CHECK(write_setup_qm_params(writer6, off, 3, true, &error));
  CHECK_EQ(writer6.bit_size(), 1U);

  // Mono (num_planes = 1) omits the UV bits.
  QmFrameState mono;
  mono.using_qmatrix = true;
  mono.qm_y[0] = 4;
  BitWriter writer7;
  CHECK(write_setup_qm_params(writer7, mono, 1, false, &error));
  CHECK_EQ(writer7.bit_size(), 5U);
}


void test_8bit_420_qp90() {
  using namespace av2enc::qm;
  // qlookup(90) = 528 in the M1 8-bit path; luma 8x8, chroma 4x4.
  const std::int32_t q = 528;
  QmFrameState state;
  state.using_qmatrix = true;
  state.qm_y[0] = 0;
  state.qm_u[0] = 0;
  state.qm_v[0] = 0;
  for (int plane = 0; plane < 3; ++plane) {
    state.seg_qm_level[plane][0] = 0;
    const int tx = plane == 0 ? 1 : 0;
    const int tw = adjusted_width(tx);
    const int th = adjusted_height(tx);
    const int n = tw * th;
    std::vector<std::int32_t> levels(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> dequant(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < th; ++i) {
      for (int j = 0; j < tw; ++j) {
        levels[static_cast<std::size_t>(i * tw + j)] =
            (i + j) % 3 == 0 ? 0 : (i * tw + j + 1);
      }
    }
    QmDequantParams params;
    params.state = &state;
    params.tx_size = tx;
    params.plane = plane;
    params.tx_type = 0;
    params.dc_quant = q;
    params.ac_quant = q;
    CHECK(dequantize_block(levels.data(), params, dequant.data()));
    for (int i = 0; i < th; ++i) {
      for (int j = 0; j < tw; ++j) {
        const int idx = i * tw + j;
        const int m = default_matrix_value(0, plane, tx, i, j);
        const std::int32_t want_q2 = (q * m + 16) >> 5;
        CHECK_EQ(dequant[static_cast<std::size_t>(idx)],
                 expected_dequant(levels[static_cast<std::size_t>(idx)],
                                  want_q2, 1));
      }
    }
  }

  // Segment 1 can use a different level for <=8x8 blocks.
  state.seg_qm_level[0][1] = 4;
  CHECK_EQ(resolve_seg_level(state, 0, 1, 1), 4);
  CHECK_EQ(resolve_seg_level(state, 0, 3, 1), state.qm_y[0]);
}

void test_invalid_inputs() {
  using namespace av2enc::qm;
  QmFrameState state;
  QmDequantParams params;
  std::int32_t data[16] = {};
  CHECK(!dequantize_block(nullptr, params, data));
  CHECK(!dequantize_block(data, params, nullptr));
  params.state = &state;
  params.tx_size = 25;
  CHECK(!dequantize_block(data, params, data));
  params.tx_size = 0;
  params.plane = 3;
  CHECK(!dequantize_block(data, params, data));
  params.plane = 0;
  params.segment_id = 8;
  CHECK(!dequantize_block(data, params, data));
  params.segment_id = 0;
  params.dc_quant = 0;
  CHECK(!dequantize_block(data, params, data));
  params.dc_quant = 64;
  params.ac_quant = 132;
  CHECK(dequantize_block(data, params, data));

  UserQmLevel level;
  CHECK(!fill_user_qm_constant(&level, 0));
  CHECK(!fill_user_qm_constant(nullptr, 32));

  QmObuConfig config;
  config.bit_map = 0x8000;
  BitWriter writer;
  std::string error;
  CHECK(!write_quantizer_matrix_obu_payload(writer, config, &error));
}

}  // namespace

int main() {
  test_tables();
  test_dequant_no_qm();
  test_dequant_all_sizes_and_denoms();
  test_level_gating_and_planes();
  test_allow_tcq_shift();
  test_dequant_clip_8_10bit();
  test_user_qm_selection();
  test_user_qm_serialization();
  test_qm_obu_writer();
  test_setup_qm_params();
  test_8bit_420_qp90();
  test_invalid_inputs();
  std::printf("qm_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
