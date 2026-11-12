// Host-emulation parity test for the shared AV2 temporal motion-field core.
//
// The same `motion_field_device.cuh` that the CUDA kernels include is compiled
// here by the host compiler (with CUDA keywords stubbed) and compared with the
// CPU reference:
//
//   §7.9.4/.5  projection and clamp          temporal_math.h
//   §7.22      compression/uncompression     temporal_math.h
//   §7.9.6-.8  sampled / block positions     temporal_math.h
//   §7.12.2.8  temporal scan                 motion_field.cpp
//   §7.22      storage entry and block grid  motion_field.cpp
//   §7.9       projection job + trajectory   motion_field.cpp
//   §7.12.2.21 ref-MV bank update/reset/fill ref_mv_bank.cpp
//   §7.12.2.19 merge-stack sort              spec selection step
//
// No GPU, nvcc or CUDA runtime is needed.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/motion_field/motion_field_cpu_oracle.h"
#include "cuda/motion_field/motion_field_device.cuh"
#include "encoder/inter/temporal/motion_field.h"
#include "encoder/inter/temporal/ref_mv_bank.h"
#include "encoder/inter/temporal/temporal_math.h"

namespace {

namespace dev = av2enc::cuda::motion_field;
namespace cpu = av2enc::inter::temporal;

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) {
      std::fprintf(stderr, "FAIL: %s\n", what);
    }
  }
}

struct Rng {
  std::uint64_t state = 1;
  explicit Rng(std::uint64_t seed) : state(seed == 0 ? 1 : seed) {}
  std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<std::uint32_t>(state >> 32);
  }
  int range(int low, int high) {  // inclusive
    return low + static_cast<int>(next() %
                                  static_cast<std::uint32_t>(high - low + 1));
  }
  bool coin() { return (next() & 1u) != 0; }
};

bool SameMv(cpu::Mv a, cpu::Mv b) { return a.row == b.row && a.col == b.col; }

// ---------------------------------------------------------------------------
// Shared view helpers
// ---------------------------------------------------------------------------

struct FieldBuffers {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int32_t> offset;

  FieldBuffers(int w, int h) : w8(w), h8(h) {
    mv.assign(static_cast<std::size_t>(w) * h * 2, 0);
    valid.assign(static_cast<std::size_t>(w) * h, 0);
    offset.assign(static_cast<std::size_t>(w) * h, 0);
  }
  dev::MfField Device() {
    dev::MfField f;
    f.w8 = w8;
    f.h8 = h8;
    f.mv = mv.data();
    f.valid = valid.data();
    f.offset = offset.data();
    return f;
  }
  cpu::MotionField Cpu() {
    cpu::MotionField f;
    f.w8 = w8;
    f.h8 = h8;
    f.mv = mv.data();
    f.valid = valid.data();
    f.offset = offset.data();
    return f;
  }
};

struct TrajBuffers {
  int w8 = 0;
  int h8 = 0;
  int num_refs = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int16_t> pos;
  std::vector<std::uint8_t> pos_valid;

  TrajBuffers(int w, int h, int refs) : w8(w), h8(h), num_refs(refs) {
    const std::size_t cells = static_cast<std::size_t>(w) * h;
    mv.assign(cells * static_cast<std::size_t>(refs) * 2, 0);
    valid.assign(cells * static_cast<std::size_t>(refs), 0);
    pos.assign(cells * static_cast<std::size_t>(refs) * 3 * 2, 0);
    pos_valid.assign(cells * static_cast<std::size_t>(refs) * 3, 0);
  }
  dev::MfTraj Device() {
    dev::MfTraj t;
    t.w8 = w8;
    t.h8 = h8;
    t.num_refs = num_refs;
    t.mv = mv.data();
    t.valid = valid.data();
    t.pos = pos.data();
    t.pos_valid = pos_valid.data();
    return t;
  }
  cpu::TrajectoryField Cpu() {
    cpu::TrajectoryField t;
    t.w8 = w8;
    t.h8 = h8;
    t.num_refs = num_refs;
    t.mv = mv.data();
    t.valid = valid.data();
    t.pos = pos.data();
    t.pos_valid = pos_valid.data();
    return t;
  }
};

// ---------------------------------------------------------------------------
// §7.9.4/.5 projection, §7.22 compression, §7.9.6-.8 positions
// ---------------------------------------------------------------------------

void TestProjectionMath() {
  for (int i = 0; i < 32; ++i) {
    Check(dev::mf_div_mult(i) == cpu::kDivMult[i], "Div_Mult table");
  }
  const int values[] = {0,     1,      -1,     7,      -7,    8,
                        -8,    100,    -100,   2047,   -2047, 2048,
                        -2048, 65535,  -65535, 32768,  -32768};
  for (int numerator = -40; numerator <= 40; ++numerator) {
    for (int denominator = -2; denominator <= 40; ++denominator) {
      for (int value : values) {
        const cpu::Mv cm{value, -value / 2};
        const dev::Mv dm{value, -value / 2};
        const cpu::Mv want = cpu::get_mv_projection(cm, numerator, denominator);
        const dev::Mv got = dev::mf_get_mv_projection(dm, numerator, denominator);
        Check(SameMv(want, got), "§7.9.4 projection");
        const cpu::Mv wc =
            cpu::get_mv_projection_clamp(cm, numerator, denominator);
        const dev::Mv gc =
            dev::mf_get_mv_projection_clamp(dm, numerator, denominator);
        Check(SameMv(wc, gc), "§7.9.5 projection clamp");
      }
    }
  }

  for (int v = -70000; v <= 70000; ++v) {
    const std::int16_t want_c = cpu::compression_mv(v);
    const std::int16_t got_c = dev::mf_compression_mv(v);
    Check(want_c == got_c, "§7.22 compression_mv");
    Check(cpu::uncompression_mv(want_c) == dev::mf_uncompression_mv(got_c),
          "§7.22 uncompression_mv");
  }
}

void TestSampledPositions() {
  Rng rng(0x1234567);
  for (int i = 0; i < 120000; ++i) {
    const int x8 = rng.range(-4, 80);
    const int y8 = rng.range(-4, 80);
    const int dst_sign = rng.coin() ? 1 : -1;
    const cpu::Mv mv{rng.range(-3000, 3000), rng.range(-3000, 3000)};
    const int mi_cols = rng.range(2, 160);
    const int mi_rows = rng.range(2, 160);
    const int proj_step = rng.coin() ? 1 : 2;
    const cpu::SampledPosition want = cpu::get_sampled_position(
        x8, y8, dst_sign, mv, mi_cols, mi_rows, proj_step);
    const dev::MfSampledPosition got = dev::mf_get_sampled_position(
        x8, y8, dst_sign, dev::Mv{mv.row, mv.col}, mi_cols, mi_rows,
        proj_step);
    Check(want.valid == got.valid && want.x8 == got.x8 && want.y8 == got.y8,
          "§7.9.6 sampled position");

    const bool sb = rng.coin();
    const bool bvalid = rng.coin();
    const int bx = rng.range(0, 80);
    const int by = rng.range(0, 80);
    const int px = rng.range(-4, 96);
    const int py = rng.range(-4, 96);
    Check(cpu::check_block_position(bvalid, bx, by, px, py, sb, proj_step) ==
              dev::mf_check_block_position(bvalid, bx, by, px, py, sb ? 1 : 0,
                                           proj_step),
          "§7.9.8 block position");
  }
}

// ---------------------------------------------------------------------------
// §7.12.2.7/.8 temporal scan
// ---------------------------------------------------------------------------

void TestTemporalScan() {
  Rng rng(0x89abcdef);
  for (int iter = 0; iter < 20000; ++iter) {
    const int w8 = rng.range(1, 24);
    const int h8 = rng.range(1, 20);
    const int mi_cols = w8 * 2;
    const int mi_rows = h8 * 2;
    FieldBuffers f(w8, h8);
    for (std::size_t i = 0; i < f.valid.size(); ++i) {
      f.valid[i] = rng.range(0, 3) != 0 ? 1 : 0;
      f.mv[i * 2 + 0] = rng.range(-600, 600);
      f.mv[i * 2 + 1] = rng.range(-600, 600);
      f.offset[i] = rng.range(-10, 10);
    }
    const bool use_traj = rng.coin();
    TrajBuffers traj(w8, h8, 3);
    if (use_traj) {
      for (std::size_t i = 0; i < traj.valid.size(); ++i) {
        traj.valid[i] = rng.range(0, 3) != 0 ? 1 : 0;
        traj.mv[i * 2 + 0] = rng.range(-2047, 2047);
        traj.mv[i * 2 + 1] = rng.range(-2047, 2047);
      }
      for (std::size_t i = 0; i < traj.pos_valid.size(); ++i) {
        traj.pos_valid[i] = rng.range(0, 3) != 0 ? 1 : 0;
        traj.pos[i * 2 + 0] = static_cast<std::int16_t>(rng.range(0, h8 - 1));
        traj.pos[i * 2 + 1] = static_cast<std::int16_t>(rng.range(0, w8 - 1));
      }
    }

    cpu::TemporalScanInput ci;
    ci.mi_row = rng.range(0, mi_rows - 1);
    ci.mi_col = rng.range(0, mi_cols - 1);
    ci.mi_cols = mi_cols;
    ci.mi_rows = mi_rows;
    if (rng.coin()) {
      ci.tile_mi_row_start = rng.range(0, mi_rows - 1);
      ci.tile_mi_row_end = rng.range(ci.tile_mi_row_start, mi_rows);
      ci.tile_mi_col_start = rng.range(0, mi_cols - 1);
      ci.tile_mi_col_end = rng.range(ci.tile_mi_col_start, mi_cols);
    }
    ci.bw4 = (1 << rng.range(0, 4));
    ci.bh4 = (1 << rng.range(0, 4));
    ci.ref0 = rng.range(-1, 7);
    ci.ref1 = rng.range(-1, 7);
    ci.ref0_order_hint = rng.range(-1, 12);
    ci.ref1_order_hint = rng.range(-1, 12);
    ci.order_hint = rng.range(-1, 12);
    ci.is_compound = rng.coin();
    ci.use_traj = use_traj;
    ci.num_refs = 3;
    cpu::MotionField cf = f.Cpu();
    cpu::TrajectoryField ct = traj.Cpu();
    ci.field = &cf;
    ci.traj = use_traj ? &ct : nullptr;

    dev::MfScanInput di;
    di.mi_row = ci.mi_row;
    di.mi_col = ci.mi_col;
    di.mi_cols = ci.mi_cols;
    di.mi_rows = ci.mi_rows;
    di.tile_mi_col_start = ci.tile_mi_col_start;
    di.tile_mi_col_end = ci.tile_mi_col_end;
    di.tile_mi_row_start = ci.tile_mi_row_start;
    di.tile_mi_row_end = ci.tile_mi_row_end;
    di.bw4 = ci.bw4;
    di.bh4 = ci.bh4;
    di.ref0 = ci.ref0;
    di.ref1 = ci.ref1;
    di.ref0_order_hint = ci.ref0_order_hint;
    di.ref1_order_hint = ci.ref1_order_hint;
    di.order_hint = ci.order_hint;
    di.is_compound = ci.is_compound;
    di.use_traj = ci.use_traj;
    dev::MfField df = f.Device();
    dev::MfTraj dt = traj.Device();
    di.field = &df;
    di.traj = use_traj ? &dt : nullptr;

    cpu::StackMv cstack[cpu::kMaxRefMvStackSize]{};
    dev::StackMv dstack[dev::t::kMaxRefMvStackSize]{};
    const int init_found = rng.range(0, cpu::kMaxRefMvStackSize - 1);
    int cprune = rng.range(0, cpu::kMaxPrNum + 2);
    int dprune = cprune;
    for (int i = 0; i < cpu::kMaxRefMvStackSize; ++i) {
      cstack[i].mv[0] = cpu::Mv{rng.range(-800, 800), rng.range(-800, 800)};
      cstack[i].mv[1] = cpu::Mv{rng.range(-800, 800), rng.range(-800, 800)};
      cstack[i].weight = rng.range(1, 4);
      cstack[i].cwp = rng.range(0, 8);
      dstack[i].mv[0] = dev::Mv{cstack[i].mv[0].row, cstack[i].mv[0].col};
      dstack[i].mv[1] = dev::Mv{cstack[i].mv[1].row, cstack[i].mv[1].col};
      dstack[i].weight = cstack[i].weight;
      dstack[i].cwp = cstack[i].cwp;
    }
    int cfound = init_found;
    int dfound = init_found;
    cpu::temporal_scan(ci, cstack, &cfound, &cprune);
    dev::mf_temporal_scan(di, dstack, &dfound, &dprune);
    Check(cfound == dfound, "temporal scan num_found");
    Check(cprune == dprune, "temporal scan prune_count");
    for (int i = 0; i < cpu::kMaxRefMvStackSize; ++i) {
      Check(SameMv(cstack[i].mv[0], dstack[i].mv[0]), "temporal scan mv0");
      Check(SameMv(cstack[i].mv[1], dstack[i].mv[1]), "temporal scan mv1");
      Check(cstack[i].weight == dstack[i].weight, "temporal scan weight");
      Check(cstack[i].cwp == dstack[i].cwp, "temporal scan cwp");
    }
  }
}

// Odd MiRows has a trailing 4x4 row with no 8x8 field cell.  Sampling there
// must be rejected by the device core; the CPU oracle's linear read must see
// defined padding (the GPU test zero-pads its field planes for the same
// reason).  Without the extent guard the device reads a neighbour cell.
void TestTemporalScanFieldExtent() {
  const int w8 = 4;
  const int h8 = 3;
  const int mi_cols = w8 * 2;
  const int mi_rows = h8 * 2 + 1;  // odd
  const std::size_t padded = static_cast<std::size_t>(w8) * (h8 + 1);

  // CPU side: the trailing half-row is zero padding.
  std::vector<std::int32_t> cpu_mv(padded * 2, 1);
  std::vector<std::uint8_t> cpu_valid(padded, 0);
  std::vector<std::int32_t> cpu_offset(padded, 0);
  for (int y = 0; y < h8; ++y) {
    for (int x = 0; x < w8; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w8 + x;
      cpu_valid[i] = 1;
      cpu_mv[i * 2 + 0] = 8;
      cpu_mv[i * 2 + 1] = 0;
      cpu_offset[i] = 4;
    }
  }
  // Device side: same field, but the padding is nonzero.  The extent guard
  // must win before the valid check.
  std::vector<std::int32_t> dev_mv = cpu_mv;
  std::vector<std::uint8_t> dev_valid = cpu_valid;
  std::vector<std::int32_t> dev_offset = cpu_offset;
  for (std::size_t i = static_cast<std::size_t>(w8) * h8; i < padded; ++i) {
    dev_valid[i] = 1;
  }

  cpu::MotionField cf;
  cf.w8 = w8;
  cf.h8 = h8;
  cf.mv = cpu_mv.data();
  cf.valid = cpu_valid.data();
  cf.offset = cpu_offset.data();
  dev::MfField df;
  df.w8 = w8;
  df.h8 = h8;
  df.mv = dev_mv.data();
  df.valid = dev_valid.data();
  df.offset = dev_offset.data();

  cpu::TemporalScanInput ci;
  ci.mi_row = 4;  // + delta_row (2) = 6 = MiRows-1 -> y8 = h8
  ci.mi_col = 0;  // + delta_col (2) -> x8 = 1
  ci.mi_cols = mi_cols;
  ci.mi_rows = mi_rows;
  ci.bw4 = 4;
  ci.bh4 = 4;
  ci.ref0 = 0;
  ci.ref0_order_hint = 4;
  ci.order_hint = 5;
  ci.field = &cf;

  dev::MfScanInput di;
  di.mi_row = ci.mi_row;
  di.mi_col = ci.mi_col;
  di.mi_cols = ci.mi_cols;
  di.mi_rows = ci.mi_rows;
  di.bw4 = ci.bw4;
  di.bh4 = ci.bh4;
  di.ref0 = ci.ref0;
  di.ref0_order_hint = ci.ref0_order_hint;
  di.order_hint = ci.order_hint;
  di.field = &df;

  cpu::StackMv cstack[cpu::kMaxRefMvStackSize]{};
  dev::StackMv dstack[dev::t::kMaxRefMvStackSize]{};
  int cfound = 0;
  int dfound = 0;
  int cprune = 0;
  int dprune = 0;
  cpu::temporal_scan(ci, cstack, &cfound, &cprune);
  dev::mf_temporal_scan(di, dstack, &dfound, &dprune);
  Check(cfound == 0, "CPU scan rejects the odd trailing half-row");
  Check(dfound == 0, "GPU scan rejects the odd trailing half-row");
  Check(cprune == dprune, "odd trailing half-row prune parity");
}

void TestTipCand() {
  Rng rng(0x13579bdf);
  for (int iter = 0; iter < 12000; ++iter) {
    const int w8 = rng.range(1, 20);
    const int h8 = rng.range(1, 16);
    FieldBuffers f(w8, h8);
    for (std::size_t i = 0; i < f.valid.size(); ++i) {
      f.valid[i] = rng.range(0, 4) != 0 ? 1 : 0;
      f.mv[i * 2 + 0] = rng.range(-2047, 2047);
      f.mv[i * 2 + 1] = rng.range(-2047, 2047);
      f.offset[i] = rng.range(-16, 16);
    }
    cpu::TipConfig tip;
    tip.num_past_refs = rng.range(0, 2);
    tip.num_future_refs = rng.range(0, 2);
    tip.order_hint = rng.range(0, 12);
    tip.closest_past_order_hint = rng.range(-1, 12);
    tip.closest_future_order_hint = rng.range(-1, 12);
    cpu::TipCandInput ci;
    ci.cand_row = rng.range(0, h8 * 2 - 1);
    ci.cand_col = rng.range(0, w8 * 2 - 1);
    ci.base_row = rng.range(0, h8 * 2 - 1);
    ci.base_col = rng.range(0, w8 * 2 - 1);
    ci.is_16x16 = rng.coin();
    ci.base_mv = cpu::Mv{rng.range(-500, 500), rng.range(-500, 500)};
    cpu::MotionField cf = f.Cpu();
    ci.field = &cf;

    dev::MfTipCandInput di;
    di.cand_row = ci.cand_row;
    di.cand_col = ci.cand_col;
    di.base_row = ci.base_row;
    di.base_col = ci.base_col;
    di.is_16x16 = ci.is_16x16 ? 1 : 0;
    di.base_mv = dev::Mv{ci.base_mv.row, ci.base_mv.col};
    dev::MfField df = f.Device();
    di.field = &df;
    di.num_past_refs = tip.num_past_refs;
    di.num_future_refs = tip.num_future_refs;
    di.order_hint = tip.order_hint;
    di.closest_past_order_hint = tip.closest_past_order_hint;
    di.closest_future_order_hint = tip.closest_future_order_hint;

    const cpu::MvPair want = cpu::get_tip_cand(ci, tip);
    const dev::MvPair got = dev::mf_get_tip_cand(di);
    Check(SameMv(want.mv[0], got.mv[0]), "get_tip_cand list0");
    Check(SameMv(want.mv[1], got.mv[1]), "get_tip_cand list1");
  }
}

// ---------------------------------------------------------------------------
// §7.22 storage entry and block grid
// ---------------------------------------------------------------------------

struct SavedBuffers {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int8_t> ref;
  std::vector<std::int16_t> mv;
  SavedBuffers(int w, int h) : w8(w), h8(h) {
    ref.assign(static_cast<std::size_t>(w) * h * 2, 0);
    mv.assign(static_cast<std::size_t>(w) * h * 4, 0);
  }
  cpu::RefMvField Cpu() {
    cpu::RefMvField f;
    f.w8 = w8;
    f.h8 = h8;
    f.ref = ref.data();
    f.mv = mv.data();
    return f;
  }
  dev::MfSavedField Device() {
    dev::MfSavedField f;
    f.w8 = w8;
    f.h8 = h8;
    f.ref = ref.data();
    f.mv = mv.data();
    return f;
  }
};

void TestStoreEntry() {
  Rng rng(0x2468ace0);
  for (int iter = 0; iter < 20000; ++iter) {
    const int w8 = rng.range(1, 20);
    const int h8 = rng.range(1, 16);
    SavedBuffers cpu_buf(w8, h8);
    SavedBuffers dev_buf(w8, h8);
    for (auto& v : cpu_buf.ref) v = static_cast<std::int8_t>(rng.range(-3, 9));
    for (auto& v : cpu_buf.mv) v = static_cast<std::int16_t>(rng.range(-9000, 9000));
    dev_buf.ref = cpu_buf.ref;
    dev_buf.mv = cpu_buf.mv;

    cpu::MotionFieldStoreEntry ce;
    dev::MfStoreEntry de;
    ce.refs[0] = rng.range(-1, 7);
    ce.refs[1] = rng.range(-1, 7);
    ce.mv[0] = cpu::Mv{rng.range(-2600, 2600), rng.range(-2600, 2600)};
    ce.mv[1] = cpu::Mv{rng.range(-2600, 2600), rng.range(-2600, 2600)};
    ce.is_wedge = rng.coin();
    std::vector<std::uint8_t> mask(64 * 64);
    for (int y = 0; y < 64; ++y) {
      for (int x = 0; x < 64; ++x) {
        const int mode = rng.range(0, 2);
        mask[static_cast<std::size_t>(y) * 64 + x] =
            mode == 0 ? static_cast<std::uint8_t>(rng.range(0, 255))
            : mode == 1 ? static_cast<std::uint8_t>(x < 32 ? 0 : 255)
                        : static_cast<std::uint8_t>(
                              std::min(255, std::max(0, x * 4 - 60)));
      }
    }
    ce.mask = mask.data();
    ce.mask_stride = 64;
    ce.i8 = rng.range(0, 7);
    ce.j8 = rng.range(0, 7);
    ce.order_hint = rng.range(0, 12);
    ce.ref_order_hint[0] = rng.range(-1, 12);
    ce.ref_order_hint[1] = rng.range(-1, 12);

    de.refs[0] = ce.refs[0];
    de.refs[1] = ce.refs[1];
    de.mv[0] = dev::Mv{ce.mv[0].row, ce.mv[0].col};
    de.mv[1] = dev::Mv{ce.mv[1].row, ce.mv[1].col};
    de.is_wedge = ce.is_wedge;
    de.mask = mask.data();
    de.mask_stride = 64;
    de.i8 = ce.i8;
    de.j8 = ce.j8;
    de.order_hint = ce.order_hint;
    de.ref_order_hint[0] = ce.ref_order_hint[0];
    de.ref_order_hint[1] = ce.ref_order_hint[1];

    const int y8 = rng.range(0, h8 - 1);
    const int x8 = rng.range(0, w8 - 1);
    cpu::RefMvField cf = cpu_buf.Cpu();
    Check(cpu::store_motion_field_entry(ce, &cf, y8, x8) ==
              cpu::TemporalStatus::kOk,
          "CPU store entry ok");
    dev::mf_store_motion_field_entry(de, dev_buf.ref.data(), dev_buf.mv.data(),
                                     w8, h8, y8, x8);
    Check(cpu_buf.ref == dev_buf.ref, "§7.22 stored ref");
    Check(cpu_buf.mv == dev_buf.mv, "§7.22 stored mv");
  }
}

// Host oracle for one §7.22 block (shared with the real GPU test).
void CpuStoreBlock(const dev::MfBlockStoreInput& in, std::int8_t* out_ref,
                   std::int16_t* out_mv) {
  dev::mf_test_cpu_store_block(in, out_ref, out_mv);
}

void TestStoreBlock() {
  Rng rng(0xdeadbeef);
  const int sizes[] = {1, 2, 4, 8, 16, 32};
  for (int iter = 0; iter < 500; ++iter) {
    const int mi_cols = rng.range(1, 72);
    const int mi_rows = rng.range(1, 56);
    const int w8 = mi_cols >> 1;
    const int h8 = mi_rows >> 1;
    if (w8 <= 0 || h8 <= 0) continue;

    std::vector<std::int32_t> mvs(
        static_cast<std::size_t>(mi_rows) * mi_cols * 4, 0);
    for (auto& v : mvs) v = rng.range(-2300, 2300);

    // Tile the frame with nominal block sizes; the last row/column overhangs.
    int r = 0;
    while (r < mi_rows) {
      const int bh4 = sizes[rng.range(0, 4)];
      int c = 0;
      while (c < mi_cols) {
        const int bw4 = sizes[rng.range(0, 4)];
        dev::MfBlockStoreInput in;
        in.r = r;
        in.c = c;
        in.bw4 = bw4;
        in.bh4 = bh4;
        in.mi_cols = mi_cols;
        in.mi_rows = mi_rows;
        in.enable_ref_frame_mvs = 1;
        in.mv_method = rng.coin() ? 0 : 1;
        const bool tip = rng.range(0, 5) == 0;
        in.refs[0] = tip ? cpu::kTipFrame : rng.range(-1, 6);
        in.refs[1] = rng.range(-1, 6);
        if (rng.coin() && !tip) in.refs[1] = in.refs[0];
        in.ref_order_hint[0] = rng.range(-1, 10);
        in.ref_order_hint[1] = rng.range(-1, 10);
        in.order_hint = rng.range(0, 10);
        in.is_wedge = rng.coin();
        in.mask = nullptr;
        in.mask_stride = 0;
        in.src.mvs = mvs.data();
        in.src.mi_cols = mi_cols;
        in.src.mi_rows = mi_rows;
        in.src.delta_stride4 = 2 * bw4;
        in.src.motion_mode = rng.range(0, 1);
        in.src.force_integer_mv = rng.range(0, 3) == 0;
        in.src.is_global_mv_cand = rng.range(0, 3) == 0;
        in.src.use_refinemv = rng.coin();
        in.warp_mv[0] = dev::Mv{rng.range(-900, 900), rng.range(-900, 900)};
        in.warp_mv[1] = dev::Mv{rng.range(-900, 900), rng.range(-900, 900)};
        in.global_mv[0] = dev::Mv{rng.range(-900, 900), rng.range(-900, 900)};
        in.global_mv[1] = dev::Mv{rng.range(-900, 900), rng.range(-900, 900)};

        std::vector<std::int32_t> refine(
            static_cast<std::size_t>(2 * bh4) * (2 * bw4) * 4, 0);
        std::vector<std::int32_t> deltas(
            static_cast<std::size_t>(2 * bh4) * (2 * bw4) * 4, 0);
        for (auto& v : refine) v = rng.range(-1800, 1800);
        for (auto& v : deltas) v = rng.range(-40, 40);
        in.src.refine_mvs = refine.data();
        in.src.mv_deltas = deltas.data();

        const std::size_t cell_count =
            static_cast<std::size_t>(w8) * h8;
        std::vector<std::int8_t> cpu_ref(cell_count * 2,
                                        static_cast<std::int8_t>(rng.range(-1, 7)));
        std::vector<std::int16_t> cpu_mv(cell_count * 4,
                                        static_cast<std::int16_t>(
                                            rng.range(-8000, 8000)));
        std::vector<std::int8_t> dev_ref = cpu_ref;
        std::vector<std::int16_t> dev_mv = cpu_mv;

        in.tip_pred = in.refs[0] == cpu::kTipFrame;
        in.tip_frame_as_output = rng.coin();
        in.tip_global_wtd_index = rng.range(0, 7);
        in.closest_past = rng.range(0, 6);
        in.closest_future = rng.range(0, 6);
        in.tip_past_order_hint = rng.range(-1, 10);
        in.tip_future_order_hint = rng.range(-1, 10);
        in.num_past_refs = rng.range(0, 2);
        in.num_future_refs = rng.range(0, 2);

        std::vector<std::int32_t> field_mv(
            static_cast<std::size_t>(w8) * h8 * 2, 0);
        std::vector<std::uint8_t> field_valid(
            static_cast<std::size_t>(w8) * h8, 0);
        std::vector<std::int32_t> field_offset(
            static_cast<std::size_t>(w8) * h8, 0);
        for (std::size_t i = 0; i < field_valid.size(); ++i) {
          field_valid[i] = rng.range(0, 3) != 0 ? 1 : 0;
          field_mv[i * 2 + 0] = rng.range(-1200, 1200);
          field_mv[i * 2 + 1] = rng.range(-1200, 1200);
          field_offset[i] = rng.range(-8, 8);
        }
        dev::MfField field;
        field.w8 = w8;
        field.h8 = h8;
        field.mv = field_mv.data();
        field.valid = field_valid.data();
        field.offset = field_offset.data();
        in.field = field;

        std::vector<int> row_base(
            static_cast<std::size_t>(mi_rows) * mi_cols, 0);
        std::vector<int> col_base(
            static_cast<std::size_t>(mi_rows) * mi_cols, 0);
        std::vector<int> sizes16(
            static_cast<std::size_t>(mi_rows) * mi_cols, 0);
        for (std::size_t i = 0; i < row_base.size(); ++i) {
          row_base[i] = rng.range(0, mi_rows - 1);
          col_base[i] = rng.range(0, mi_cols - 1);
          sizes16[i] = rng.coin() ? 1 : 0;
        }
        in.mi_row_base = row_base.data();
        in.mi_col_base = col_base.data();
        in.tip_sizes16 = sizes16.data();

        // Wedge masks exercise the allow-list in half the cases.
        std::vector<std::uint8_t> mask;
        if (in.is_wedge && rng.coin()) {
          mask.resize(64 * 64);
          for (int y = 0; y < 64; ++y) {
            for (int x = 0; x < 64; ++x) {
              mask[static_cast<std::size_t>(y) * 64 + x] =
                  static_cast<std::uint8_t>(rng.range(0, 255));
            }
          }
          in.mask = mask.data();
          in.mask_stride = 64;
        }

        CpuStoreBlock(in, cpu_ref.data(), cpu_mv.data());
        dev::mf_store_block(in, dev_ref.data(), dev_mv.data());
        Check(cpu_ref == dev_ref, "§7.22 block ref grid");
        Check(cpu_mv == dev_mv, "§7.22 block mv grid");
        c += bw4;
      }
      r += bh4;
    }
  }
}

// ---------------------------------------------------------------------------
// §7.12.2.21 reference-MV bank
// ---------------------------------------------------------------------------

// §7.22 ordering: 4x4 blocks can share an 8x8 cell.  A parallel writer needs
// the last-descriptor-wins rule that this stamp map provides.  Build the map
// on the host, then apply the blocks in reverse order and check that the
// result still equals the CPU's sequential call order.
void TestStoreBlockOrdering() {
  Rng rng(0x0f0f1234);
  for (int iter = 0; iter < 40; ++iter) {
    // Odd MI extents deliberately exercise 4x4 origins that fall outside the
    // last 8x8 cell row/column.
    const int mi_cols = rng.range(3, 19);
    const int mi_rows = rng.range(3, 14);
    const int w8 = mi_cols >> 1;
    const int h8 = mi_rows >> 1;
    std::vector<std::int32_t> mvs(
        static_cast<std::size_t>(mi_rows) * mi_cols * 4, 0);
    for (auto& v : mvs) v = rng.range(-2000, 2000);
    std::vector<std::int32_t> refine(64 * 64 * 4, 0);
    std::vector<std::int32_t> deltas(64 * 64 * 4, 0);
    for (auto& v : refine) v = rng.range(-1500, 1500);
    for (auto& v : deltas) v = rng.range(-30, 30);

    // Force overlap: 1x1 MI blocks at every position.
    std::vector<dev::MfBlockStoreInput> blocks;
    for (int r = 0; r < mi_rows; ++r) {
      for (int c = 0; c < mi_cols; ++c) {
        dev::MfBlockStoreInput in;
        in.r = r;
        in.c = c;
        // Mix legal block sizes with >16 MI blocks, whose i8/j8 > 7 cells are
        // rejected by the entry predicate on both sides.
        in.bw4 = rng.coin() ? rng.range(1, 2) : rng.range(17, 32);
        in.bh4 = rng.coin() ? rng.range(1, 2) : rng.range(17, 32);
        in.mi_cols = mi_cols;
        in.mi_rows = mi_rows;
        in.mv_method = rng.coin() ? 0 : 1;
        in.refs[0] = rng.range(-1, 6);
        in.refs[1] = rng.range(-1, 6);
        in.ref_order_hint[0] = rng.range(-1, 9);
        in.ref_order_hint[1] = rng.range(-1, 9);
        in.order_hint = rng.range(0, 9);
        in.is_wedge = rng.coin();
        in.src.mvs = mvs.data();
        in.src.mi_cols = mi_cols;
        in.src.mi_rows = mi_rows;
        in.src.delta_stride4 = 2 * in.bw4;
        in.src.refine_mvs = refine.data();
        in.src.mv_deltas = deltas.data();
        in.src.motion_mode = rng.range(0, 3);
        in.src.force_integer_mv = rng.range(0, 4) == 0;
        in.src.is_global_mv_cand = rng.range(0, 4) == 0;
        in.src.use_refinemv = rng.coin();
        in.warp_mv[0] = dev::Mv{rng.range(-500, 500), rng.range(-500, 500)};
        in.warp_mv[1] = dev::Mv{rng.range(-500, 500), rng.range(-500, 500)};
        in.global_mv[0] = dev::Mv{rng.range(-500, 500), rng.range(-500, 500)};
        in.global_mv[1] = dev::Mv{rng.range(-500, 500), rng.range(-500, 500)};
        blocks.push_back(in);
      }
    }
    const std::size_t cells = static_cast<std::size_t>(w8) * h8;
    std::vector<std::int8_t> want_ref(cells * 2, 0);
    std::vector<std::int16_t> want_mv(cells * 4, 0);
    for (const auto& in : blocks) {
      CpuStoreBlock(in, want_ref.data(), want_mv.data());
    }

    std::vector<int> stamp(cells, -1);
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      dev::mf_mark_block_cells(blocks[i], stamp.data(), static_cast<int>(i));
    }
    std::vector<std::int8_t> got_ref(cells * 2, 0);
    std::vector<std::int16_t> got_mv(cells * 4, 0);
    // Reverse order: writes must still be limited to the max-index owner.
    for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
      dev::mf_store_block(blocks[static_cast<std::size_t>(i)], got_ref.data(),
                          got_mv.data(), stamp.data(), i);
    }
    Check(want_ref == got_ref, "§7.22 ordered ref grid");
    Check(want_mv == got_mv, "§7.22 ordered mv grid");
  }
}

void TestRefMvBank() {
  Rng rng(0x0badf00d);
  for (int iter = 0; iter < 3000; ++iter) {
    cpu::RefMvBank cpu_bank;
    dev::RefMvBank dev_bank;
    std::vector<cpu::RefMvBankRowSource> rows(32);
    for (int i = 0; i < 32; ++i) {
      rows[i].is_inter = rng.range(0, 3) != 0 ? 1 : 0;
      rows[i].ref[0] = static_cast<std::int8_t>(rng.range(-1, 6));
      rows[i].ref[1] = static_cast<std::int8_t>(rng.range(-1, 6));
      rows[i].mv[0][0] = rng.range(-300, 300);
      rows[i].mv[0][1] = rng.range(-300, 300);
      rows[i].mv[1][0] = rng.range(-300, 300);
      rows[i].mv[1][1] = rng.range(-300, 300);
      rows[i].cwp_idx = rng.range(0, 8);
      rows[i].motion_mode = static_cast<std::uint8_t>(rng.range(0, 3));
      rows[i].bw4 = rng.range(1, 4);
    }
    std::memcpy(&dev_bank, &cpu_bank, sizeof(dev_bank));

    const int ops = rng.range(1, 24);
    for (int op = 0; op < ops; ++op) {
      const int kind = rng.range(0, 2);
      const int mi_row = rng.range(0, 32);
      const int mi_col = rng.range(0, 32);
      const int bw4 = rng.range(1, 8);
      const int bh4 = rng.range(1, 8);
      const int sb_size4 = rng.coin() ? 16 : 32;
      const bool chroma = rng.coin();
      if (kind == 0) {
        const bool intra = rng.coin();
        const bool top = rng.coin();
        cpu::reset_ref_mv_bank(&cpu_bank, mi_row, mi_col, sb_size4, intra, top,
                               cpu::RefMvBankRow{rows.data(), 32});
        dev::mf_reset_ref_mv_bank(&dev_bank, mi_row, mi_col, sb_size4, intra,
                                  top, rows.data(), 32);
      } else if (kind == 1) {
        const int refs[2] = {rng.range(-1, 6), rng.range(-1, 6)};
        const cpu::Mv mvs[2] = {
            cpu::Mv{rng.range(-400, 400), rng.range(-400, 400)},
            cpu::Mv{rng.range(-400, 400), rng.range(-400, 400)}};
        const int cwp = rng.range(0, 8);
        const bool within = rng.coin();
        cpu::RefMvBankBlockContext ctx;
        ctx.mi_row = mi_row;
        ctx.mi_col = mi_col;
        ctx.bw4 = bw4;
        ctx.bh4 = bh4;
        ctx.sb_size4 = sb_size4;
        ctx.chroma_part = chroma;
        (void)cpu::update_ref_mv_bank(&cpu_bank, ctx, refs, mvs, cwp, within);
        (void)dev::mf_update_ref_mv_bank(
            &dev_bank, mi_row, mi_col, bw4, bh4, sb_size4, chroma ? 1 : 0,
            refs, mvs, cwp, within);
      } else {
        cpu::RefMvBankBlockContext ctx;
        ctx.mi_row = mi_row;
        ctx.mi_col = mi_col;
        ctx.bw4 = bw4;
        ctx.bh4 = bh4;
        ctx.sb_size4 = sb_size4;
        ctx.chroma_part = chroma;
        cpu::update_ref_mv_count(&cpu_bank, ctx);
        dev::mf_update_ref_mv_count(&dev_bank, mi_row, mi_col, bw4, bh4,
                                    sb_size4, chroma ? 1 : 0);
      }
      Check(std::memcmp(&cpu_bank, &dev_bank, sizeof(cpu_bank)) == 0,
            "§7.12.2.21 bank state");
    }

    // MVP appends.
    for (int fill = 0; fill < 8; ++fill) {
      const int refs[2] = {rng.range(-1, 6), rng.range(-1, 6)};
      const bool compound = rng.coin();
      const bool intrabc = rng.range(0, 3) == 0;
      const int drl = rng.range(0, 4);
      const int mi_row = rng.range(0, 30);
      const int mi_col = rng.range(0, 30);
      const int bw = rng.range(1, 4) * 4;
      const int bh = rng.range(1, 4) * 4;
      cpu::StackMv cstack[cpu::kMaxRefMvStackSize]{};
      dev::StackMv dstack[dev::t::kMaxRefMvStackSize]{};
      const int init = rng.range(0, 5);
      int cfound = init;
      int dfound = init;
      int cprune = rng.range(0, 18);
      int dprune = cprune;
      for (int i = 0; i < cpu::kMaxRefMvStackSize; ++i) {
        cstack[i].mv[0] = cpu::Mv{rng.range(-500, 500), rng.range(-500, 500)};
        cstack[i].mv[1] = cpu::Mv{rng.range(-500, 500), rng.range(-500, 500)};
        cstack[i].cwp = rng.range(0, 8);
        dstack[i].mv[0] = dev::Mv{cstack[i].mv[0].row, cstack[i].mv[0].col};
        dstack[i].mv[1] = dev::Mv{cstack[i].mv[1].row, cstack[i].mv[1].col};
        dstack[i].cwp = cstack[i].cwp;
      }
      cpu::fill_mvp_from_ref_mv_bank(cpu_bank, refs, compound, intrabc, drl,
                                     mi_row, mi_col, 32, 32, bw, bh,
                                     cpu::kMaxRefMvStackSize, cstack, &cfound,
                                     &cprune);
      dev::mf_fill_mvp_from_ref_mv_bank(dev_bank, refs, compound, intrabc, drl,
                                        mi_row, mi_col, 32, 32, bw, bh,
                                        dev::t::kMaxRefMvStackSize, dstack,
                                        &dfound, &dprune);
      Check(cfound == dfound, "bank appends num_found");
      Check(cprune == dprune, "bank appends prune_count");
      for (int i = 0; i < cpu::kMaxRefMvStackSize; ++i) {
        Check(SameMv(cstack[i].mv[0], dstack[i].mv[0]), "bank append mv0");
        Check(SameMv(cstack[i].mv[1], dstack[i].mv[1]), "bank append mv1");
        Check(cstack[i].cwp == dstack[i].cwp, "bank append cwp");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.12.2 merge stack: temporal scan + bank append + §7.12.2.19 sort
// ---------------------------------------------------------------------------

void CpuSortRange(cpu::StackMv* stack, int count, int start, int end) {
  if (start < 0) start = 0;
  if (end > count) end = count;
  if (start + 1 >= end) return;
  int max_weight = stack[start].weight;
  int max_idx = start;
  for (int idx = start + 1; idx < end; ++idx) {
    if (max_weight < stack[idx].weight) {
      max_weight = stack[idx].weight;
      max_idx = idx;
    }
  }
  if (max_idx != start) {
    const cpu::StackMv tmp = stack[start];
    stack[start] = stack[max_idx];
    stack[max_idx] = tmp;
  }
}

void TestMergeBlock() {
  Rng rng(0x5a5a5a5a);
  for (int iter = 0; iter < 8000; ++iter) {
    const int w8 = rng.range(1, 20);
    const int h8 = rng.range(1, 16);
    const int mi_cols = w8 * 2;
    const int mi_rows = h8 * 2;
    FieldBuffers f(w8, h8);
    for (std::size_t i = 0; i < f.valid.size(); ++i) {
      f.valid[i] = rng.range(0, 4) != 0 ? 1 : 0;
      f.mv[i * 2 + 0] = rng.range(-400, 400);
      f.mv[i * 2 + 1] = rng.range(-400, 400);
      f.offset[i] = rng.range(-6, 6);
    }
    TrajBuffers traj(w8, h8, 2);
    for (std::size_t i = 0; i < traj.valid.size(); ++i) {
      traj.valid[i] = rng.range(0, 3) != 0 ? 1 : 0;
      traj.mv[i * 2 + 0] = rng.range(-2047, 2047);
      traj.mv[i * 2 + 1] = rng.range(-2047, 2047);
    }

    dev::MfStackBlockInput in;
    in.mi_row = rng.range(0, mi_rows - 1);
    in.mi_col = rng.range(0, mi_cols - 1);
    in.mi_cols = mi_cols;
    in.mi_rows = mi_rows;
    in.bw4 = 1 << rng.range(0, 4);
    in.bh4 = 1 << rng.range(0, 4);
    in.ref0 = rng.range(-1, 7);
    in.ref1 = rng.range(-1, 7);
    in.ref0_order_hint = rng.range(-1, 12);
    in.ref1_order_hint = rng.range(-1, 12);
    in.order_hint = rng.range(-1, 12);
    in.is_compound = rng.coin();
    in.use_traj = rng.coin();
    in.num_refs = 2;
    in.use_temporal = rng.range(0, 4) != 0;
    in.use_refmvbank = rng.coin();
    in.max_drl_bits_minus_1 = rng.range(0, 4);
    in.reorder = rng.coin();
    in.num_nearest = rng.range(0, 5);
    const int init = rng.range(0, 4);
    in.num_found = init;
    in.prune_count = rng.range(0, 18);
    for (int i = 0; i < dev::t::kMaxRefMvStackSize; ++i) {
      in.stack[i].mv[0] = dev::Mv{rng.range(-700, 700), rng.range(-700, 700)};
      in.stack[i].mv[1] = dev::Mv{rng.range(-700, 700), rng.range(-700, 700)};
      in.stack[i].weight = rng.range(1, 4);
      in.stack[i].cwp = rng.range(0, 8);
    }

    cpu::RefMvBank bank;
    {
      Rng brng(static_cast<std::uint64_t>(iter) * 7919 + 13);
      std::vector<cpu::RefMvBankRowSource> dummy(4);
      for (int op = 0; op < 12; ++op) {
        cpu::RefMvBankBlockContext ctx;
        ctx.mi_row = brng.range(0, 31);
        ctx.mi_col = brng.range(0, 31);
        ctx.bw4 = 1 << brng.range(0, 3);
        ctx.bh4 = 1 << brng.range(0, 3);
        ctx.sb_size4 = brng.coin() ? 16 : 32;
        const int refs[2] = {brng.range(-1, 6), brng.range(-1, 6)};
        const cpu::Mv mvs[2] = {
            cpu::Mv{brng.range(-300, 300), brng.range(-300, 300)},
            cpu::Mv{brng.range(-300, 300), brng.range(-300, 300)}};
        (void)cpu::update_ref_mv_bank(&bank, ctx, refs, mvs,
                                      brng.range(0, 8), brng.coin());
      }
    }
    in.field = f.Device();
    dev::MfTraj dt = traj.Device();
    in.traj = dt;

    dev::StackMv got[dev::t::kMaxRefMvStackSize]{};
    int got_found = 0;
    int got_prune = 0;
    dev::mf_merge_block(in, &bank, got, &got_found, &got_prune);

    // CPU composition in the same order.
    cpu::MotionField cf = f.Cpu();
    cpu::TrajectoryField ct = traj.Cpu();
    cpu::StackMv want[cpu::kMaxRefMvStackSize]{};
    int want_found = in.num_found;
    int want_prune = in.prune_count;
    for (int i = 0; i < cpu::kMaxRefMvStackSize; ++i) {
      want[i].mv[0] = cpu::Mv{in.stack[i].mv[0].row, in.stack[i].mv[0].col};
      want[i].mv[1] = cpu::Mv{in.stack[i].mv[1].row, in.stack[i].mv[1].col};
      want[i].weight = in.stack[i].weight;
      want[i].cwp = in.stack[i].cwp;
    }
    if (in.use_temporal) {
      cpu::TemporalScanInput ci;
      ci.mi_row = in.mi_row;
      ci.mi_col = in.mi_col;
      ci.mi_cols = in.mi_cols;
      ci.mi_rows = in.mi_rows;
      ci.tile_mi_col_start = in.tile_mi_col_start;
      ci.tile_mi_col_end = in.tile_mi_col_end;
      ci.tile_mi_row_start = in.tile_mi_row_start;
      ci.tile_mi_row_end = in.tile_mi_row_end;
      ci.bw4 = in.bw4;
      ci.bh4 = in.bh4;
      ci.ref0 = in.ref0;
      ci.ref1 = in.ref1;
      ci.ref0_order_hint = in.ref0_order_hint;
      ci.ref1_order_hint = in.ref1_order_hint;
      ci.order_hint = in.order_hint;
      ci.is_compound = in.is_compound;
      ci.use_traj = in.use_traj;
      ci.num_refs = in.num_refs;
      ci.field = &cf;
      ci.traj = in.use_traj ? &ct : nullptr;
      cpu::temporal_scan(ci, want, &want_found, &want_prune);
    }
    if (in.use_refmvbank) {
      const int refs[2] = {in.ref0, in.ref1};
      cpu::fill_mvp_from_ref_mv_bank(bank, refs, in.is_compound, in.use_intrabc,
                                     in.max_drl_bits_minus_1, in.mi_row,
                                     in.mi_col, in.mi_cols, in.mi_rows,
                                     in.bw4 * 4, in.bh4 * 4,
                                     cpu::kMaxRefMvStackSize, want, &want_found,
                                     &want_prune);
    }
    if (in.reorder) {
      CpuSortRange(want, want_found, 0, in.num_nearest);
    }

    Check(got_found == want_found, "merge block num_found");
    Check(got_prune == want_prune, "merge block prune");
    for (int i = 0; i < dev::t::kMaxRefMvStackSize; ++i) {
      Check(SameMv(got[i].mv[0], want[i].mv[0]), "merge block mv0");
      Check(SameMv(got[i].mv[1], want[i].mv[1]), "merge block mv1");
      Check(got[i].weight == want[i].weight, "merge block weight");
      Check(got[i].cwp == want[i].cwp, "merge block cwp");
    }
  }
}

// ---------------------------------------------------------------------------
// §7.9 projection job + trajectory fill, end to end against the CPU driver
// ---------------------------------------------------------------------------

struct ProjectionFixture {
  int mi_cols = 16;
  int mi_rows = 16;
  int w8 = 8;
  int h8 = 8;
  FieldBuffers field;
  TrajBuffers traj;
  std::vector<std::int16_t> saved_mv[2];
  std::vector<std::int8_t> saved_ref[2];
  std::vector<std::int32_t> saved_order_hint[2];

  ProjectionFixture(int mi_c, int mi_r, int refs)
      : mi_cols(mi_c),
        mi_rows(mi_r),
        w8(mi_c >> 1),
        h8(mi_r >> 1),
        field(mi_c >> 1, mi_r >> 1),
        traj(mi_c >> 1, mi_r >> 1, refs) {
    for (int rf = 0; rf < 2; ++rf) {
      saved_mv[rf].assign(static_cast<std::size_t>(w8) * h8 * 4, 0);
      saved_ref[rf].assign(static_cast<std::size_t>(w8) * h8 * 2,
                           cpu::kRefNone);
      saved_order_hint[rf].resize(cpu::kRefsPerFrame);
    }
  }
};

void TestProjectionAndTrajFill() {
  Rng rng(0xabcdef01);
  int cases = 0;
  for (int iter = 0; iter < 60; ++iter) {
    const int mi_cols = 2 * rng.range(4, 40);
    const int mi_rows = 2 * rng.range(4, 32);
    const int w8 = mi_cols >> 1;
    const int h8 = mi_rows >> 1;
    const bool use_traj = rng.coin();
    const bool proj_step2 = rng.coin();
    const bool sb64 = rng.coin();

    ProjectionFixture fx(mi_cols, mi_rows, 2);
    cpu::MotionFieldConfig cfg;
    cfg.mi_cols = mi_cols;
    cfg.mi_rows = mi_rows;
    cfg.proj_step = proj_step2 ? 2 : 1;
    cfg.sb_64x64 = sb64;
    cfg.enable_tip = false;
    cfg.enable_mv_traj = use_traj;
    cfg.order_hint = 9;
    cfg.num_total_refs = 2;
    cfg.num_past_refs = 2;
    cfg.num_future_refs = 0;
    cfg.frame_width = mi_cols * 4;
    cfg.frame_height = mi_rows * 4;
    cfg.ref_order_hint[0] = 2;
    cfg.ref_order_hint[1] = 7;
    for (int rf = 0; rf < 2; ++rf) {
      cfg.refs[rf].slot = rf;
      cfg.refs[rf].type = cpu::FrameType::kInter;
      cfg.refs[rf].width = cfg.frame_width;
      cfg.refs[rf].height = cfg.frame_height;
      cfg.refs[rf].order_hint = cfg.ref_order_hint[rf];
      cfg.refs[rf].num_saved_refs = 0;
      for (int k = 0; k < cpu::kRefsPerFrame; ++k) {
        const int hint = rng.range(0, 8);
        cfg.refs[rf].saved_order_hint[k] = hint;
        fx.saved_order_hint[rf][k] = hint;
      }
      for (std::size_t i = 0; i < fx.saved_ref[rf].size(); ++i) {
        fx.saved_ref[rf][i] = static_cast<std::int8_t>(rng.range(-1, 1));
        if (rng.range(0, 9) == 0) {
          fx.saved_ref[rf][i] = cpu::kRefNone;
        }
      }
      for (std::size_t i = 0; i < fx.saved_mv[rf].size(); ++i) {
        const int v = rng.range(-700, 700);
        fx.saved_mv[rf][i] = cpu::compression_mv(v);
      }
      cfg.saved[rf].w8 = w8;
      cfg.saved[rf].h8 = h8;
      cfg.saved[rf].mv = fx.saved_mv[rf].data();
      cfg.saved[rf].ref = fx.saved_ref[rf].data();
    }

    FieldBuffers cpu_field_buf(w8, h8);
    TrajBuffers cpu_traj_buf(w8, h8, 2);
    cpu::MotionField cpu_field = cpu_field_buf.Cpu();
    cpu::TrajectoryField cpu_traj = cpu_traj_buf.Cpu();
    cpu::MotionFieldResult result;
    const cpu::TemporalStatus status = cpu::motion_field_estimation(
        cfg, &cpu_field, use_traj ? &cpu_traj : nullptr, &result);
    Check(status == cpu::TemporalStatus::kOk, "CPU motion field estimation");
    if (status != cpu::TemporalStatus::kOk) continue;
    ++cases;

    // The CPU process list for two past refs with no saved refs is:
    //   ref1 forward, ref0 forward, ref1 backward.
    dev::MfProjectionParams jobs[3];
    const int srcs[3] = {1, 0, 1};
    const int signs[3] = {1, 1, -1};
    const int bwd[3] = {0, 0, 1};
    for (int j = 0; j < 3; ++j) {
      dev::MfProjectionParams& p = jobs[j];
      p.mi_cols = mi_cols;
      p.mi_rows = mi_rows;
      p.proj_step = cfg.proj_step;
      p.sb_64x64 = sb64 ? 1 : 0;
      p.enable_mv_traj = use_traj ? 1 : 0;
      p.num_total_refs = 2;
      p.order_hint = cfg.order_hint;
      for (int rf = 0; rf < cpu::kRefsPerFrame; ++rf) {
        p.ref_order_hint[rf] = cfg.ref_order_hint[rf];
      }
      p.src = srcs[j];
      p.src_order_hint = cfg.ref_order_hint[srcs[j]];
      p.num_saved = 0;
      p.dst_sign = signs[j];
      p.is_bwd = bwd[j];
      p.target_frame = cpu::kRefNone;
      p.saved.w8 = w8;
      p.saved.h8 = h8;
      p.saved.mv = fx.saved_mv[srcs[j]].data();
      p.saved.ref = fx.saved_ref[srcs[j]].data();
      for (int k = 0; k < cpu::kRefsPerFrame; ++k) {
        p.saved_order_hint[k] = fx.saved_order_hint[srcs[j]][k];
      }
      p.field = fx.field.Device();
      p.traj = fx.traj.Device();
    }

    // Zero the GPU-side arrays exactly as the CPU driver does.
    std::fill(fx.field.mv.begin(), fx.field.mv.end(), 0);
    std::fill(fx.field.valid.begin(), fx.field.valid.end(), 0);
    std::fill(fx.field.offset.begin(), fx.field.offset.end(), 0);
    std::fill(fx.traj.mv.begin(), fx.traj.mv.end(), 0);
    std::fill(fx.traj.valid.begin(), fx.traj.valid.end(), 0);
    std::fill(fx.traj.pos.begin(), fx.traj.pos.end(), 0);
    std::fill(fx.traj.pos_valid.begin(), fx.traj.pos_valid.end(), 0);

    for (int j = 0; j < 3; ++j) {
      dev::mf_project_job(jobs[j]);
    }
    if (use_traj) {
      for (int rf = 0; rf < 2; ++rf) {
        for (int y8 = 0; y8 < h8; y8 += 2) {
          for (int x8 = 0; x8 < w8; x8 += 2) {
            dev::MfTraj traj = fx.traj.Device();
            dev::mf_traj_fill_cell(&traj, rf, y8, x8, w8, h8, sb64 ? 1 : 0,
                                   cfg.proj_step);
          }
        }
      }
    }

    Check(fx.field.mv.size() ==
              static_cast<std::size_t>(w8) * h8 * 2,
          "field size");
    for (std::size_t i = 0; i < fx.field.valid.size(); ++i) {
      Check(fx.field.valid[i] == cpu_field.valid[i], "§7.9 field valid");
      Check(fx.field.offset[i] == cpu_field.offset[i], "§7.9 field offset");
      Check(fx.field.mv[i * 2 + 0] == cpu_field.mv[i * 2 + 0],
            "§7.9 field mv row");
      Check(fx.field.mv[i * 2 + 1] == cpu_field.mv[i * 2 + 1],
            "§7.9 field mv col");
    }
    if (use_traj) {
      Check(fx.traj.mv.size() ==
                static_cast<std::size_t>(w8) * h8 * 2 * 2,
            "traj size");
      Check(fx.traj.valid == cpu_traj_buf.valid, "§7.9 traj valid");
      Check(fx.traj.pos == cpu_traj_buf.pos, "§7.9 traj pos");
      Check(fx.traj.pos_valid == cpu_traj_buf.pos_valid,
            "§7.9 traj pos_valid");
    }
  }
  Check(cases > 40, "projection cases exercised");
}

}  // namespace

int main() {
  std::printf(
      "motion_field host-emulation parity (shared device core vs CPU)\n");

  TestProjectionMath();
  TestSampledPositions();
  TestTemporalScan();
  TestTemporalScanFieldExtent();
  TestTipCand();
  TestStoreEntry();
  TestStoreBlock();
  TestStoreBlockOrdering();
  TestRefMvBank();
  TestMergeBlock();
  TestProjectionAndTrajFill();

  if (g_failures == 0) {
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
  }
  std::printf("FAIL: %d checks, %d failures\n", g_checks, g_failures);
  return 1;
}
