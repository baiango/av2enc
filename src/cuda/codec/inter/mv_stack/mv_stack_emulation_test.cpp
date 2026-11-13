// Host-emulation parity test for the CUDA MV-stack core.
//
// Compiles `mv_stack_device.cuh` with the host compiler and compares it with
// the authoritative CPU reference:
//   * merge::build_mv_stack                 (§7.12.2 spatial stack + §7.12.2.19)
//   * multiref::build_pair_stacks           (two lists + compound)
//   * multiref::build_compound_merge_candidates
//
// No GPU and no nvcc are required.  The real sm_86 gate is `mv_stack_test.cu`.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "codec/inter/merge/merge.h"
#include "codec/inter/multiref/mvstack_multi.h"
#include "mv_stack_device.cuh"

namespace {

namespace vm = av2enc::video::merge;
namespace mr = av2enc::video::multiref;
namespace mvs = av2enc::cuda::mvstack;
namespace imc = av2enc::inter::mc;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 25) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

std::uint32_t g_rng = 0x2545f491u;

std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

int random_range(int low, int high) {
  return low + static_cast<int>(
                   next_random() % static_cast<std::uint32_t>(high - low + 1));
}

bool same_mv(imc::Mv a, imc::Mv b) { return a.y == b.y && a.x == b.x; }

bool same_stack(const vm::MvStack& a, const vm::MvStack& b) {
  if (a.count != b.count || a.num_nearest != b.num_nearest ||
      a.sorted != b.sorted) {
    return false;
  }
  for (int i = 0; i < a.count; ++i) {
    if (!same_mv(a.cand[i].mv[0], b.cand[i].mv[0]) ||
        !same_mv(a.cand[i].mv[1], b.cand[i].mv[1]) ||
        a.cand[i].weight != b.cand[i].weight ||
        a.cand[i].row_offset != b.cand[i].row_offset ||
        a.cand[i].col_offset != b.cand[i].col_offset ||
        a.cand[i].cwp != b.cand[i].cwp) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Random MI grids
// ---------------------------------------------------------------------------

struct GridCase {
  int mi_cols = 0;
  int mi_rows = 0;
  int num_refs = 0;
  std::vector<imc::MiInfo> mi;
  std::vector<int> frame_distance;
};

GridCase make_grid(int mi_cols, int mi_rows, int num_refs) {
  GridCase g;
  g.mi_cols = mi_cols;
  g.mi_rows = mi_rows;
  g.num_refs = num_refs;
  g.mi.assign(static_cast<std::size_t>(mi_cols) *
                  static_cast<std::size_t>(mi_rows),
              imc::MiInfo{});
  for (int row = 0; row < mi_rows; ++row) {
    for (int col = 0; col < mi_cols; ++col) {
      imc::MiInfo& cell =
          g.mi[static_cast<std::size_t>(row) * static_cast<std::size_t>(mi_cols) +
              static_cast<std::size_t>(col)];
      if ((next_random() % 4u) == 0u) {
        // Intra / not-yet-coded cell.
        cell.ref[0] = col == 0 ? vm::kRefNone : vm::kIntraFrame;
        cell.ref[1] = vm::kRefNone;
        cell.is_inter = 0;
        continue;
      }
      cell.is_inter = 1;
      const int w4 = random_range(1, 4);
      const int h4 = random_range(1, 4);
      cell.w4 = static_cast<std::int8_t>(w4);
      cell.h4 = static_cast<std::int8_t>(h4);
      cell.col_base = static_cast<std::int16_t>(col - (col % w4));
      cell.row_base = static_cast<std::int16_t>(row - (row % h4));
      for (int list = 0; list < 2; ++list) {
        if (list == 1 && (next_random() & 1u) != 0u) {
          cell.ref[1] = vm::kRefNone;
          continue;
        }
        cell.ref[list] = static_cast<std::int8_t>(random_range(0, num_refs - 1));
        cell.mv[list][0] = random_range(-96, 96) * 8;
        cell.mv[list][1] = random_range(-96, 96) * 8;
        if ((next_random() % 16u) == 0u) {
          cell.mv[list][0] = random_range(-4096, 4096);
          cell.mv[list][1] = random_range(-4096, 4096);
        }
      }
    }
  }
  g.frame_distance.resize(static_cast<std::size_t>(num_refs));
  for (int i = 0; i < num_refs; ++i) {
    g.frame_distance[static_cast<std::size_t>(i)] = random_range(-6, 6);
  }
  return g;
}

vm::StackParams make_params(const GridCase& g, bool allow_compound) {
  vm::StackParams p;
  p.mi = g.mi.data();
  p.mi_cols = g.mi_cols;
  p.mi_rows = g.mi_rows;
  p.num_refs = g.num_refs;
  p.frame_distance = g.frame_distance.data();
  p.bw4 = random_range(1, 16);
  p.bh4 = random_range(1, 16);
  p.mi_col = random_range(0, g.mi_cols - 1);
  p.mi_row = random_range(0, g.mi_rows - 1);
  const int sb_h4 = (next_random() & 1u) != 0u ? 8 : 16;
  p.sb_h4 = sb_h4;
  p.ref_frame[0] = static_cast<std::int8_t>(random_range(0, g.num_refs - 1));
  p.ref_frame[1] = vm::kRefNone;
  p.is_compound = false;
  if (allow_compound && (next_random() % 3u) == 0u) {
    p.ref_frame[1] = static_cast<std::int8_t>(random_range(0, g.num_refs - 1));
    p.is_compound = true;
  }
  p.global_mv[0] = imc::Mv{random_range(-32, 32) * 8,
                           random_range(-32, 32) * 8};
  p.global_mv[1] = p.is_compound
                       ? imc::Mv{random_range(-32, 32) * 8,
                                 random_range(-32, 32) * 8}
                       : p.global_mv[0];
  const int mode = random_range(0, 2);
  p.reorder = mode == 0 ? vm::DrlReorder::kDisabled
                        : (mode == 1 ? vm::DrlReorder::kConstraint
                                     : vm::DrlReorder::kAlways);
  p.use_temporal_first = false;
  p.max_drl_bits_minus_1 = random_range(0, 4);
  p.use_intrabc = false;
  return p;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_single_and_compound_stacks() {
  for (int trial = 0; trial < 1500; ++trial) {
    const int mi_cols = random_range(8, 24);
    const int mi_rows = random_range(8, 24);
    const int num_refs = random_range(1, 4);
    const GridCase grid = make_grid(mi_cols, mi_rows, num_refs);
    const vm::StackParams params = make_params(grid, true);

    vm::MvStack cpu;
    const vm::Status cpu_status = vm::build_mv_stack(params, &cpu);

    vm::MvStack gpu;
    const vm::Status gpu_status = mvs::mvs_build_stack(params, &gpu);

    CHECK(static_cast<int>(cpu_status) == static_cast<int>(gpu_status));
    if (cpu_status == vm::Status::kOk) {
      CHECK(same_stack(cpu, gpu));
    }
  }
}

void test_invalid_params() {
  GridCase grid = make_grid(12, 10, 2);
  vm::StackParams base = make_params(grid, false);
  base.ref_frame[0] = 0;
  base.ref_frame[1] = vm::kRefNone;
  base.is_compound = false;

  struct Variant {
    const char* name;
    vm::StackParams params;
  };
  std::vector<Variant> variants;

  {
    vm::StackParams p = base;
    p.mi = nullptr;
    variants.push_back({"null mi", p});
  }
  {
    vm::StackParams p = base;
    p.bw4 = 0;
    variants.push_back({"zero bw4", p});
  }
  {
    vm::StackParams p = base;
    p.bh4 = 17;
    variants.push_back({"large bh4", p});
  }
  {
    vm::StackParams p = base;
    p.sb_h4 = 6;
    variants.push_back({"non-power-of-two sb", p});
  }
  {
    vm::StackParams p = base;
    p.mi_row = grid.mi_rows;
    variants.push_back({"mi_row out of range", p});
  }
  {
    vm::StackParams p = base;
    p.use_intrabc = true;
    variants.push_back({"intrabc", p});
  }
  {
    vm::StackParams p = base;
    p.max_drl_bits_minus_1 = 5;
    variants.push_back({"max_drl_bits", p});
  }
  {
    vm::StackParams p = base;
    p.ref_frame[0] = vm::kTipFrame;
    variants.push_back({"tip ref", p});
  }
  {
    vm::StackParams p = base;
    p.is_compound = true;
    p.ref_frame[1] = vm::kRefNone;
    variants.push_back({"compound without ref1", p});
  }
  {
    vm::StackParams p = base;
    p.ref_frame[1] = 1;
    variants.push_back({"single with ref1", p});
  }
  for (const Variant& v : variants) {
    vm::MvStack cpu;
    vm::MvStack gpu;
    const vm::Status cpu_status = vm::build_mv_stack(v.params, &cpu);
    const vm::Status gpu_status = mvs::mvs_build_stack(v.params, &gpu);
    CHECK(static_cast<int>(cpu_status) == static_cast<int>(gpu_status));
    CHECK(cpu_status != vm::Status::kOk);
  }
  CHECK(mvs::mvs_build_stack(base, nullptr) == vm::Status::kInvalidArgument);
}

void test_pair_stacks_and_merge_set() {
  for (int trial = 0; trial < 800; ++trial) {
    const int mi_cols = random_range(8, 24);
    const int mi_rows = random_range(8, 24);
    const int num_refs = random_range(2, 4);
    const GridCase grid = make_grid(mi_cols, mi_rows, num_refs);
    vm::StackParams params = make_params(grid, false);
    params.ref_frame[0] = 0;
    params.ref_frame[1] = vm::kRefNone;
    params.is_compound = false;
    const std::int8_t ref0 = 0;
    const std::int8_t ref1 = static_cast<std::int8_t>(random_range(1, num_refs - 1));

    mr::RefStackPair cpu;
    std::string cpu_error;
    const bool cpu_ok =
        mr::build_pair_stacks(params, ref0, ref1, cpu, cpu_error);
    CHECK(cpu_ok);
    CHECK(cpu_error.empty());

    mvs::MvsPairJob job;
    job.params = params;
    job.ref0 = ref0;
    job.ref1 = ref1;
    mvs::MvsPairStacks gpu;
    mvs::mvs_build_pair_stacks(job, &gpu);

    CHECK(gpu.ok[0] == (cpu.ok[0] ? 1 : 0));
    CHECK(gpu.ok[1] == (cpu.ok[1] ? 1 : 0));
    CHECK(gpu.compound_ok == (cpu.compound_ok ? 1 : 0));
    CHECK(same_stack(cpu.list[0], gpu.list[0]));
    CHECK(same_stack(cpu.list[1], gpu.list[1]));
    CHECK(same_stack(cpu.compound, gpu.compound));
    CHECK(cpu.num_nearest[0] == gpu.num_nearest[0]);
    CHECK(cpu.num_nearest[1] == gpu.num_nearest[1]);

    const mr::CompoundMergeSet cpu_set =
        mr::build_compound_merge_candidates(cpu);
    mvs::MvsMergeSet gpu_set;
    mvs::mvs_build_merge_set(gpu, &gpu_set);
    CHECK(gpu_set.count == static_cast<int>(cpu_set.candidates.size()));
    const int common =
        gpu_set.count < static_cast<int>(cpu_set.candidates.size())
            ? gpu_set.count
            : static_cast<int>(cpu_set.candidates.size());
    bool all_equal = true;
    for (int i = 0; i < common; ++i) {
      const mr::CompoundMergeCandidate& a =
          cpu_set.candidates[static_cast<std::size_t>(i)];
      const mvs::MvsMergeCandidate& b = gpu_set.cand[i];
      if (!same_mv(a.mv[0], b.mv[0]) || !same_mv(a.mv[1], b.mv[1]) ||
          a.drl_index[0] != b.drl_index[0] ||
          a.drl_index[1] != b.drl_index[1] || a.weight != b.weight ||
          a.single_index != (b.single_index != 0)) {
        all_equal = false;
        if (g_failures < 25) {
          std::printf("merge candidate %d differs\n", i);
        }
        break;
      }
    }
    CHECK(all_equal);
  }
}

// DR L reordering must be observable: a stack with more than one out-of-order
// weight and CONSTRAINT/ALWAYS reorder must apply exactly one swap per step.
void test_drl_sort_is_exercised() {
  int swapped = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const int mi_cols = 16;
    const int mi_rows = 16;
    const GridCase grid = make_grid(mi_cols, mi_rows, 2);
    vm::StackParams params = make_params(grid, false);
    params.ref_frame[0] = 0;
    params.ref_frame[1] = vm::kRefNone;
    params.is_compound = false;
    params.reorder = trial % 2 == 0 ? vm::DrlReorder::kConstraint
                                    : vm::DrlReorder::kAlways;
    vm::MvStack cpu;
    vm::MvStack gpu;
    if (vm::build_mv_stack(params, &cpu) != vm::Status::kOk) {
      continue;
    }
    mvs::mvs_build_stack(params, &gpu);
    CHECK(same_stack(cpu, gpu));
    if (cpu.num_nearest >= 2 && cpu.cand[0].weight < cpu.cand[1].weight) {
      ++swapped;
    }
  }
  std::printf("drl: %d reordered stacks observed\n", swapped);
  CHECK(swapped > 0);
}

}  // namespace

int main() {
  test_single_and_compound_stacks();
  test_invalid_params();
  test_pair_stacks_and_merge_set();
  test_drl_sort_is_exercised();
  std::printf("mv_stack_emulation_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
