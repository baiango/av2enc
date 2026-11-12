// Host-emulation parity test for the shared AV2 chroma-LR device arithmetic.
//
// Runs without nvcc or a GPU.  Compares:
//   * `lrc_resolve_plane()` against an independent transcription of the
//     §5.20.10.6 bank/subset/uv_sym expansion (`ref::resolve_plane`),
//   * `lrc_process_frame()` against `lr_chroma::chroma_apply()` for 8-bit
//     4:2:0 (the CPU reference module's supported domain),
//   * `lrc_process_frame()` against the independent bit-depth/subsampling
//     reference (`ref::apply_frame`) for 10-bit and 4:2:2 / 4:4:4, where
//     `chroma_apply()` is hard-wired to 8-bit 4:2:0,
//   * the independent reference against `chroma_apply()` on the same 8-bit
//     4:2:0 cases.
//
// `lr_chroma_test.cu` includes this file with AV2ENC_LRC_TEST_NO_MAIN and runs
// the same case set through the kernels.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "codec/filters/loop_restoration_chroma/chroma_lr.h"
#include "codec/filters/loop_restoration/restoration_common.h"
#include "lr_chroma_device.cuh"
#include "lr_chroma_reference.h"

namespace {

namespace lrc = av2enc::cuda::lr_chroma;
namespace ref = av2enc::cuda::lr_chroma::ref;
namespace cpu = av2enc::lr_chroma;

int g_checks = 0;
int g_failures = 0;
long long g_samples = 0;
int g_cpu_oracle_cases = 0;
int g_resolver_units = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what);
  }
}

std::uint32_t g_rng = 0x8f31c2a7u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}
bool chance(int percent) {
  return static_cast<int>(next_random() % 100u) < percent;
}

struct PlaneHost {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> u8;
  std::vector<std::uint16_t> u16;
};

void allocate_plane(PlaneHost* p, int width, int height, int bit_depth) {
  p->width = width;
  p->height = height;
  const std::size_t n =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (bit_depth == 8) {
    p->u8.assign(n, 0);
    p->u16.clear();
  } else {
    p->u16.assign(n, 0);
    p->u8.clear();
  }
}

void* plane_data(PlaneHost* p, int bit_depth) {
  return bit_depth == 8 ? static_cast<void*>(p->u8.data())
                        : static_cast<void*>(p->u16.data());
}

const void* plane_data(const PlaneHost& p, int bit_depth) {
  return bit_depth == 8 ? static_cast<const void*>(p.u8.data())
                        : static_cast<const void*>(p.u16.data());
}

int plane_max(int bit_depth) { return (1 << bit_depth) - 1; }

// Smooth base + quantisation-like noise.  `cdef` gets a different noise
// pattern so that filtering is observable.
void fill_plane(PlaneHost* p, int seed, int bit_depth, int phase) {
  const int maxv = plane_max(bit_depth);
  for (int y = 0; y < p->height; ++y) {
    for (int x = 0; x < p->width; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * p->width + static_cast<std::size_t>(x);
      const int base =
          maxv / 2 + (maxv / 8) * ((x * 3 + y * 5 + phase) % 7 - 3) / 3 +
          (maxv / 16) * ((x * 11 + y * 13 + seed) % 5 - 2) / 2;
      int v = base + ((x * 17 + y * 19 + seed + phase) % 17) - 8;
      if (v < 0) v = 0;
      if (v > maxv) v = maxv;
      if (!p->u8.empty()) {
        p->u8[i] = static_cast<std::uint8_t>(v < 256 ? v : 255);
      } else {
        p->u16[i] = static_cast<std::uint16_t>(v);
      }
    }
  }
}


int case_width(const lrc::LrChromaConfig& c) {
  return (c.mi_cols * 4) >> c.sub_x;
}
int case_height(const lrc::LrChromaConfig& c) {
  return (c.mi_rows * 4) >> c.sub_y;
}
int case_unit_cols(const lrc::LrChromaConfig& c) {
  return ref::count_units(c.unit_size, case_width(c));
}
int case_unit_rows(const lrc::LrChromaConfig& c) {
  return ref::count_units(c.unit_size, case_height(c));
}
int case_unit_count(const lrc::LrChromaConfig& c) {
  return case_unit_cols(c) * case_unit_rows(c);
}

struct TestCase {
  lrc::LrChromaConfig config;
  PlaneHost curr[2];
  PlaneHost cdef[2];
  PlaneHost luma_curr;
  PlaneHost luma_cdef;
  std::vector<lrc::LrChromaUnit> units[2];
  std::vector<std::int16_t> ref_resolved[2];
  std::vector<std::uint8_t> ref_active[2];
  PlaneHost oracle[2];
  bool has_cpu_oracle = false;
  bool tiled = false;
  std::vector<std::uint16_t> mi_col_start;
  std::vector<std::uint16_t> mi_col_end;
  std::vector<std::uint16_t> mi_row_start;
  std::vector<std::uint16_t> mi_row_end;
};

// Splits the luma MI grid into square tiles of `tile_mi` MI units.  Tile edges
// are in MI units; the §6.17.7.7 checker cares about luma-sample widths.
void apply_tiles(TestCase* c, int tile_mi) {
  const int cols = c->config.mi_cols;
  const int rows = c->config.mi_rows;
  const std::size_t n = static_cast<std::size_t>(cols) * rows;
  c->mi_col_start.assign(n, 0);
  c->mi_col_end.assign(n, 0);
  c->mi_row_start.assign(n, 0);
  c->mi_row_end.assign(n, 0);
  for (int r = 0; r < rows; ++r) {
    for (int col = 0; col < cols; ++col) {
      const int cs = (col / tile_mi) * tile_mi;
      const int rs = (r / tile_mi) * tile_mi;
      const int ce = std::min(cs + tile_mi, cols);
      const int re = std::min(rs + tile_mi, rows);
      const std::size_t at = static_cast<std::size_t>(r) * cols + col;
      c->mi_col_start[at] = static_cast<std::uint16_t>(cs);
      c->mi_col_end[at] = static_cast<std::uint16_t>(ce);
      c->mi_row_start[at] = static_cast<std::uint16_t>(rs);
      c->mi_row_end[at] = static_cast<std::uint16_t>(re);
    }
  }
  c->tiled = true;
}

int present_coeffs(int subset) {
  return subset <= 0 ? 6 : (subset == 1 ? 10 : 18);
}

int random_coeff(int j) {
  const int lo = av2enc::restoration::wiener_ns_coeff_min(true, j);
  const int hi = av2enc::restoration::wiener_ns_coeff_max(true, j);
  return random_range(lo, hi);
}

// Generates one plane's unit stream.  `mode` selects subset/uv_sym:
//   0: subset 0; 1: subset 1 asym; 2: subset 1 sym; 3: subset 2 asym;
//   4: subset 2 sym.
void generate_units(TestCase* c, int chroma, int mode, int seed) {
  const int count = c->config.mi_cols > 0
                        ? case_unit_count(c->config)
                        : 1;
  const int subset = mode <= 0 ? 0 : (mode <= 2 ? 1 : 2);
  const bool uv_sym = (mode == 2 || mode == 4);
  c->units[chroma].assign(static_cast<std::size_t>(count),
                          lrc::LrChromaUnit{});
  g_rng = static_cast<std::uint32_t>(seed) * 2654435761u + 0x9e3779b9u;
  int bank_size = 0;
  int bank_ptr = 0;
  for (int i = 0; i < count; ++i) {
    lrc::LrChromaUnit& u = c->units[chroma][static_cast<std::size_t>(i)];
    const bool seed_unit = (i == 0);
    u.on = (seed_unit || chance(88)) ? 1 : 0;
    if (u.on == 0) {
      continue;
    }
    if (!seed_unit && bank_size > 0 && chance(32)) {
      u.merged = 1;
      u.bank_k = random_range(0, bank_size - 1);
      continue;
    }
    u.merged = 0;
    u.subset = subset;
    u.uv_sym = uv_sym ? 1 : 0;
    const int present = present_coeffs(subset);
    for (int j = 0; j < 18; ++j) {
      u.coeffs[j] = 0;
    }
    for (int j = 0; j < present; ++j) {
      int v = random_coeff(j);
      if (j >= 6) {
        // Make sure the chroma-from-luma taps are actually exercised.  Stay
        // inside the normative tap range if the random draw produced 0.
        if (v == 0) {
          v = av2enc::restoration::wiener_ns_coeff_min(true, j);
        }
      }
      u.coeffs[j] = static_cast<std::int16_t>(v);
      if (uv_sym && j >= 6 && j + 1 < present && (j % 2) == 0) {
        // The decoder ignores j+1 when uv_sym is set; make it different so
        // that a resolver which forgets to mirror is caught.
        int mirror = v + ((j + seed) % 3) + 1;
        if (mirror > 32767) mirror = 32767;
        u.coeffs[j + 1] = static_cast<std::int16_t>(mirror);
        ++j;
      }
    }
    // A real uv_sym pair would code one value; keep the coded j and a
    // deliberately different j+1.  The resolver must overwrite j+1.
    const int new_ptr = bank_size < 4 ? bank_size : ((bank_ptr + 1) % 4);
    bank_ptr = new_ptr;
    if (bank_size < 4) ++bank_size;
  }
}

lrc::LrChromaCore make_host_core(const TestCase& c, const PlaneHost out[2]) {
  lrc::LrChromaCore core{};
  for (int chroma = 0; chroma < 2; ++chroma) {
    core.chroma[chroma].curr = plane_data(c.curr[chroma], c.config.bit_depth);
    core.chroma[chroma].curr_stride = c.curr[chroma].width;
    core.chroma[chroma].cdef = plane_data(c.cdef[chroma], c.config.bit_depth);
    core.chroma[chroma].cdef_stride = c.cdef[chroma].width;
    core.chroma[chroma].dst = const_cast<void*>(
        plane_data(out[chroma], c.config.bit_depth));
    core.chroma[chroma].dst_stride = out[chroma].width;
    core.unit_size[chroma] = c.config.unit_size;
    core.unit_cols[chroma] = case_unit_cols(c.config);
    core.unit_rows[chroma] = case_unit_rows(c.config);
  }
  core.luma.curr = plane_data(c.luma_curr, c.config.bit_depth);
  core.luma.curr_stride = c.luma_curr.width;
  core.luma.cdef = plane_data(c.luma_cdef, c.config.bit_depth);
  core.luma.cdef_stride = c.luma_cdef.width;
  core.mi_cols = c.config.mi_cols;
  core.mi_rows = c.config.mi_rows;
  core.sub_x = c.config.sub_x;
  core.sub_y = c.config.sub_y;
  core.bit_depth = c.config.bit_depth;
  core.cfl_ds_filter_index = c.config.cfl_ds_filter_index;
  if (c.tiled) {
    core.mi_col_start = c.mi_col_start.data();
    core.mi_col_end = c.mi_col_end.data();
    core.mi_row_start = c.mi_row_start.data();
    core.mi_row_end = c.mi_row_end.data();
    core.disable_loopfilters_across_tiles = 1;
  }
  return core;
}

lrc::LrChromaFrame make_host_frame(const TestCase& c, PlaneHost out[2]) {
  lrc::LrChromaFrame f{};
  for (int chroma = 0; chroma < 2; ++chroma) {
    f.chroma[chroma].curr = plane_data(c.curr[chroma], c.config.bit_depth);
    f.chroma[chroma].curr_stride = c.curr[chroma].width;
    f.chroma[chroma].cdef = plane_data(c.cdef[chroma], c.config.bit_depth);
    f.chroma[chroma].cdef_stride = c.cdef[chroma].width;
    f.chroma[chroma].dst = plane_data(&out[chroma], c.config.bit_depth);
    f.chroma[chroma].dst_stride = out[chroma].width;
  }
  f.luma.curr = plane_data(c.luma_curr, c.config.bit_depth);
  f.luma.curr_stride = c.luma_curr.width;
  f.luma.cdef = plane_data(c.luma_cdef, c.config.bit_depth);
  f.luma.cdef_stride = c.luma_cdef.width;
  f.mi_cols = c.config.mi_cols;
  f.mi_rows = c.config.mi_rows;
  f.sub_x = c.config.sub_x;
  f.sub_y = c.config.sub_y;
  f.bit_depth = c.config.bit_depth;
  f.cfl_ds_filter_index = c.config.cfl_ds_filter_index;
  if (c.tiled) {
    f.mi_col_start = c.mi_col_start.data();
    f.mi_col_end = c.mi_col_end.data();
    f.mi_row_start = c.mi_row_start.data();
    f.mi_row_end = c.mi_row_end.data();
    f.disable_loopfilters_across_tiles = 1;
  }
  return f;
}

bool run_cpu_chroma_apply(TestCase* c, PlaneHost out[2]) {
  const int width = case_width(c->config);
  const int height = case_height(c->config);
  cpu::ChromaLrInput in;
  in.plane_width[0] = in.plane_width[1] = width;
  in.plane_height[0] = in.plane_height[1] = height;
  in.mi_cols = c->config.mi_cols;
  in.mi_rows = c->config.mi_rows;
  in.sub_x = 1;
  in.sub_y = 1;
  in.sb_size = 64;
  in.cfl_ds_filter_index = c->config.cfl_ds_filter_index;
  for (int chroma = 0; chroma < 2; ++chroma) {
    in.curr[chroma] = c->curr[chroma].u8.data();
    in.curr_stride[chroma] = c->curr[chroma].width;
    in.cdef[chroma] = c->cdef[chroma].u8.data();
    in.cdef_stride[chroma] = c->cdef[chroma].width;
    in.source[chroma] = c->curr[chroma].u8.data();
    in.source_stride[chroma] = c->curr[chroma].width;
  }
  in.luma_curr = c->luma_curr.u8.data();
  in.luma_curr_stride = c->luma_curr.width;
  in.luma_cdef = c->luma_cdef.u8.data();
  in.luma_cdef_stride = c->luma_cdef.width;
  in.luma_width = c->luma_curr.width;
  in.luma_height = c->luma_curr.height;

  cpu::ChromaDecision decision;
  decision.shift = 0;
  decision.unit_size = c->config.unit_size;
  decision.sb_size = 64;
  std::string error;
  for (int chroma = 0; chroma < 2; ++chroma) {
    if (!cpu::chroma_make_unit_grid(width, height, c->config.unit_size,
                                    &decision.grid[chroma], &error)) {
      std::printf("FAIL: chroma_make_unit_grid: %s\n", error.c_str());
      return false;
    }
    const int cols = decision.grid[chroma].unit_cols;
    for (int u = 0; u < case_unit_count(c->config);
         ++u) {
      cpu::ChromaUnitPlan plan;
      plan.plane = chroma + 1;
      plan.unit_row = u / cols;
      plan.unit_col = u % cols;
      plan.on = c->ref_active[chroma][static_cast<std::size_t>(u)] != 0;
      plan.filter.subset = 2;
      plan.filter.uv_sym = false;
      plan.filter.merged = false;
      for (int j = 0; j < 18; ++j) {
        plan.filter.coeffs[j] =
            c->ref_resolved[chroma][static_cast<std::size_t>(u) * 18 + j];
      }
      decision.units.push_back(plan);
    }
  }
  std::uint8_t* dst[2] = {out[0].u8.data(), out[1].u8.data()};
  int stride[2] = {out[0].width, out[1].width};
  const av2enc::restoration::RestorationStatus status =
      cpu::chroma_apply(in, decision, dst, stride);
  if (status != av2enc::restoration::RestorationStatus::kOk) {
    std::printf("FAIL: chroma_apply status %d\n", static_cast<int>(status));
    return false;
  }
  return true;
}

bool run_cpu_tiled_oracle(TestCase* c, PlaneHost out[2]) {
  using av2enc::restoration::LoopRestorationParams;
  using av2enc::restoration::RestorationFrame;
  using av2enc::restoration::RestorationPlane;
  using av2enc::restoration::RestorationStatus;
  const int width = case_width(c->config);
  const int luma_width = c->config.mi_cols * 4;
  const int luma_height = c->config.mi_rows * 4;
  std::vector<std::uint8_t> luma_out(
      static_cast<std::size_t>(luma_width) * luma_height, 0);
  RestorationFrame f{};
  f.mi_cols = c->config.mi_cols;
  f.mi_rows = c->config.mi_rows;
  f.subsampling_x = 1;
  f.subsampling_y = 1;
  f.bit_depth = 8;
  f.sb_width = 64;
  f.sb_height = 64;
  f.sb_shift = 4;
  f.cfl_ds_filter_index = c->config.cfl_ds_filter_index;
  f.mi_col_start = c->mi_col_start.data();
  f.mi_col_end = c->mi_col_end.data();
  f.mi_row_start = c->mi_row_start.data();
  f.mi_row_end = c->mi_row_end.data();
  f.disable_loopfilters_across_tiles = 1;
  f.curr[0] = {c->luma_curr.u8.data(), luma_width};
  f.cdef[0] = {c->luma_cdef.u8.data(), luma_width};
  f.lr[0] = {luma_out.data(), luma_width};
  for (int chroma = 0; chroma < 2; ++chroma) {
    f.curr[chroma + 1] = {c->curr[chroma].u8.data(), width};
    f.cdef[chroma + 1] = {c->cdef[chroma].u8.data(), width};
    f.lr[chroma + 1] = {out[chroma].u8.data(), width};
  }
  const std::size_t count =
      static_cast<std::size_t>(case_unit_count(c->config));
  std::vector<std::uint8_t> type[2];
  std::vector<std::int16_t> coeffs[2];
  for (int chroma = 0; chroma < 2; ++chroma) {
    type[chroma].assign(count, av2enc::restoration::kRestoreNone);
    for (std::size_t u = 0; u < count; ++u) {
      type[chroma][u] = c->ref_active[chroma][u] != 0
                            ? av2enc::restoration::kRestoreWienerNs
                            : av2enc::restoration::kRestoreNone;
    }
    coeffs[chroma] = c->ref_resolved[chroma];
  }
  LoopRestorationParams params{};
  params.frame_type[0] = av2enc::restoration::kRestoreNone;
  params.frame_type[1] = av2enc::restoration::kRestoreWienerNs;
  params.frame_type[2] = av2enc::restoration::kRestoreWienerNs;
  params.unit_size[0] = c->config.unit_size;
  params.unit_size[1] = c->config.unit_size;
  params.unit_size[2] = c->config.unit_size;
  for (int chroma = 0; chroma < 2; ++chroma) {
    params.unit_type[chroma + 1] = type[chroma].data();
    params.unit_rows[chroma + 1] = case_unit_rows(c->config);
    params.unit_cols[chroma + 1] = case_unit_cols(c->config);
    params.unit_coeffs[chroma + 1] = coeffs[chroma].data();
  }
  const RestorationStatus status =
      av2enc::restoration::loop_restoration_frame(f, params);
  if (status != RestorationStatus::kOk) {
    std::printf("FAIL: tiled loop_restoration_frame status %d\n",
                static_cast<int>(status));
    return false;
  }
  (void)count;
  return true;
}

TestCase make_case(int mi_cols, int mi_rows, int sub_x, int sub_y,
                   int bit_depth, int unit_size, int cfl, int mode, int seed,
                   int tile_mi = 0) {
  TestCase c;
  c.config.mi_cols = mi_cols;
  c.config.mi_rows = mi_rows;
  c.config.sub_x = sub_x;
  c.config.sub_y = sub_y;
  c.config.bit_depth = bit_depth;
  c.config.unit_size = unit_size;
  c.config.cfl_ds_filter_index = cfl;
  c.config.sb_shift = 4;
  const int width = case_width(c.config);
  const int height = case_height(c.config);
  const int luma_width = mi_cols * 4;
  const int luma_height = mi_rows * 4;
  for (int chroma = 0; chroma < 2; ++chroma) {
    allocate_plane(&c.curr[chroma], width, height, bit_depth);
    allocate_plane(&c.cdef[chroma], width, height, bit_depth);
    fill_plane(&c.curr[chroma], seed + chroma * 7, bit_depth, 0);
    fill_plane(&c.cdef[chroma], seed + chroma * 7, bit_depth, 5);
  }
  allocate_plane(&c.luma_curr, luma_width, luma_height, bit_depth);
  allocate_plane(&c.luma_cdef, luma_width, luma_height, bit_depth);
  fill_plane(&c.luma_curr, seed + 31, bit_depth, 2);
  fill_plane(&c.luma_cdef, seed + 31, bit_depth, 9);

  if (tile_mi > 0) {
    apply_tiles(&c, tile_mi);
  }
  for (int chroma = 0; chroma < 2; ++chroma) {
    generate_units(&c, chroma, mode, seed + chroma * 101 + 1);
    const int count =
        case_unit_count(c.config);
    c.ref_resolved[chroma].assign(static_cast<std::size_t>(count) * 18, 0);
    c.ref_active[chroma].assign(static_cast<std::size_t>(count), 0);
    ref::resolve_plane(c.units[chroma].data(), count,
                       c.ref_resolved[chroma].data(),
                       c.ref_active[chroma].data());
    allocate_plane(&c.oracle[chroma], width, height, bit_depth);
  }

  // Reference oracle (all bit depths and subsamplings).
  const lrc::LrChromaFrame frame = make_host_frame(c, c.oracle);
  const std::int16_t* resolved[2] = {c.ref_resolved[0].data(),
                                     c.ref_resolved[1].data()};
  const std::uint8_t* active[2] = {c.ref_active[0].data(),
                                   c.ref_active[1].data()};
  ref::apply_frame(c.config, frame, resolved, active);

  if (bit_depth == 8 && sub_x == 1 && sub_y == 1) {
    PlaneHost cpu_out[2];
    for (int chroma = 0; chroma < 2; ++chroma) {
      allocate_plane(&cpu_out[chroma], width, height, 8);
    }
    if (c.tiled) {
      c.has_cpu_oracle = run_cpu_tiled_oracle(&c, cpu_out);
    } else {
      c.has_cpu_oracle = run_cpu_chroma_apply(&c, cpu_out);
    }
    if (c.has_cpu_oracle) ++g_cpu_oracle_cases;
    check(c.has_cpu_oracle, "CPU chroma oracle returned ok");
    if (c.has_cpu_oracle) {
      for (int chroma = 0; chroma < 2; ++chroma) {
        bool same = true;
        for (std::size_t i = 0; i < cpu_out[chroma].u8.size(); ++i) {
          if (cpu_out[chroma].u8[i] != c.oracle[chroma].u8[i]) {
            same = false;
            break;
          }
        }
        check(same, "reference agrees with the CPU chroma oracle");
        // The CPU core is the primary oracle for this domain.
        c.oracle[chroma].u8 = cpu_out[chroma].u8;
      }
    }
  }
  return c;
}

bool compare_plane(const std::vector<std::uint8_t>& got,
                   const std::vector<std::uint8_t>& want, const char* what) {
  bool same = got.size() == want.size();
  std::size_t first = 0;
  if (same) {
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] != want[i]) {
        same = false;
        first = i;
        break;
      }
    }
  }
  check(same, what);
  if (!same && first < got.size() && first < want.size()) {
    std::printf("  mismatch at %zu: got %u want %u\n", first,
                static_cast<unsigned>(got[first]), static_cast<unsigned>(want[first]));
  }
  g_samples += static_cast<long long>(want.size());
  return same;
}

bool compare_plane16(const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& want, const char* what) {
  bool same = got.size() == want.size();
  std::size_t first = 0;
  if (same) {
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] != want[i]) {
        same = false;
        first = i;
        break;
      }
    }
  }
  check(same, what);
  if (!same && first < got.size() && first < want.size()) {
    std::printf("  mismatch at %zu: got %u want %u\n", first,
                static_cast<unsigned>(got[first]),
                static_cast<unsigned>(want[first]));
  }
  g_samples += static_cast<long long>(want.size());
  return same;
}

// Runs the shared device header on the host and checks resolver + pixels.
void run_emulated_case(const TestCase& c) {
  for (int chroma = 0; chroma < 2; ++chroma) {
    const int count =
        case_unit_count(c.config);
    std::vector<std::int16_t> got(static_cast<std::size_t>(count) * 18, 0);
    std::vector<std::uint8_t> got_active(static_cast<std::size_t>(count), 0);
    lrc::LrChromaBankState bank{};
    lrc::lrc_resolve_plane(c.units[chroma].data(), count, got.data(),
                           got_active.data(), &bank);
    for (int u = 0; u < count; ++u) {
      ++g_resolver_units;
      bool same = got_active[static_cast<std::size_t>(u)] ==
                  c.ref_active[chroma][static_cast<std::size_t>(u)];
      for (int j = 0; j < 18 && same; ++j) {
        same = got[static_cast<std::size_t>(u) * 18 + j] ==
               c.ref_resolved[chroma][static_cast<std::size_t>(u) * 18 + j];
      }
      check(same, "device resolver matches reference expansion");
    }
  }

  PlaneHost out[2];
  for (int chroma = 0; chroma < 2; ++chroma) {
    allocate_plane(&out[chroma], case_width(c.config),
                   case_height(c.config), c.config.bit_depth);
  }
  const lrc::LrChromaCore core = make_host_core(c, out);
  const std::int16_t* resolved[2] = {c.ref_resolved[0].data(),
                                     c.ref_resolved[1].data()};
  const std::uint8_t* active[2] = {c.ref_active[0].data(),
                                   c.ref_active[1].data()};
  if (c.config.bit_depth == 8) {
    lrc::lrc_process_frame<std::uint8_t>(core, resolved, active);
    for (int chroma = 0; chroma < 2; ++chroma) {
      compare_plane(out[chroma].u8, c.oracle[chroma].u8,
                    "emulated device core matches CPU oracle (8-bit)");
    }
  } else {
    lrc::lrc_process_frame<std::uint16_t>(core, resolved, active);
    for (int chroma = 0; chroma < 2; ++chroma) {
      compare_plane16(out[chroma].u16, c.oracle[chroma].u16,
                      "emulated device core matches reference (10-bit)");
    }
  }
}

std::vector<TestCase> build_cases() {
  std::vector<TestCase> cases;
  const int subs[3][2] = {{1, 1}, {1, 0}, {0, 0}};
  const int bits[2] = {8, 10};
  const int units[4] = {32, 64, 128, 256};
  // Small/odd edge grids plus larger frames that produce several units per
  // plane, so unit transitions and bank merges are exercised at every unit
  // size (32/64/128/256).
  const int geoms[6][2] = {{16, 16}, {7, 5}, {48, 32}, {65, 33}, {97, 49},
                           {257, 129}};
  int seed = 1;
  for (int s = 0; s < 3; ++s) {
    for (int b = 0; b < 2; ++b) {
      for (int u = 0; u < 4; ++u) {
        for (int mode = 0; mode < 5; ++mode) {
          for (int g = 0; g < 6; ++g) {
            const int cfl = (seed + u + g) % 4;
            TestCase c = make_case(geoms[g][0], geoms[g][1], subs[s][0],
                                   subs[s][1], bits[b], units[u], cfl, mode,
                                   seed);
            ++seed;
            cases.push_back(std::move(c));
          }
        }
      }
    }
  }
  // §7.20.1 tile-clamp cases: 8-bit 4:2:0 against the CPU core with real
  // MiCol/MiRow grids (chroma_apply() has no tile inputs).
  for (int u = 0; u < 4; ++u) {
    for (int mode : {0, 2, 4}) {
      // Tile width in MI units is unit_size/2 for 4:2:0 (luma units are twice
      // the chroma unit), so §6.17.7.7 checker widths are multiples.
      TestCase tiled = make_case(140, 70, 1, 1, 8, units[u], mode % 4, mode,
                                 seed, units[u] / 2);
      ++seed;
      cases.push_back(std::move(tiled));
    }
  }

  // Randomised layouts, including non-power-of-two MI grids.
  for (int i = 0; i < 80; ++i) {
    const int mi_cols = random_range(1, 40);
    const int mi_rows = random_range(1, 30);
    const int s = random_range(0, 2);
    const int bit = (i % 2 == 0) ? 8 : 10;
    const int unit = units[random_range(0, 3)];
    const int mode = random_range(0, 4);
    TestCase c = make_case(mi_cols, mi_rows, subs[s][0], subs[s][1], bit,
                           unit, i % 4, mode, 100000 + i * 37);
    cases.push_back(std::move(c));
  }
  return cases;
}

}  // namespace

#ifndef AV2ENC_LRC_TEST_NO_MAIN
int main() {
  std::printf("lr_chroma emulation parity (8/10-bit, 4:2:0/4:2:2/4:4:4, "
              "units 32/64/128/256)\n");
  const std::vector<TestCase> cases = build_cases();
  std::printf("  cases: %zu\n", cases.size());
  for (const TestCase& c : cases) {
    run_emulated_case(c);
  }
  std::printf(
      "PASS: %d checks, %d failures (%lld samples compared; "
      "chroma_apply oracle cases %d, resolved units %d)\n",
      g_checks, g_failures, g_samples, g_cpu_oracle_cases, g_resolver_units);
  return g_failures == 0 ? 0 : 1;
}
#endif
