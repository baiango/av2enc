// Real GPU parity test for the CUDA MV-stack kernels (sm_86).
//
// Every case is generated on the host, the CPU reference
// (`merge::build_mv_stack`, `multiref::build_pair_stacks`,
// `multiref::build_compound_merge_candidates`) runs on the same data, and the
// kernels must reproduce it field for field.  All device memory comes from one
// `av2enc::cuda::Arena`.
//
//   ./build_gpu_test.sh                 # build + run
//   ./build_gpu_test.sh --no-run
//   ARCH=sm_120 ./build_gpu_test.sh

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "codec/inter/multiref/mvstack_multi.h"
#include "mv_stack.h"

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
    if (g_failures <= 30) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CUDA_OK(expr)                                                       \
  do {                                                                      \
    const cudaError_t err = (expr);                                         \
    check(err == cudaSuccess, #expr, __LINE__);                             \
    if (err != cudaSuccess) {                                               \
      std::printf("  CUDA %s: %s\n", cudaGetErrorName(err),                 \
                  cudaGetErrorString(err));                                 \
    }                                                                       \
  } while (0)

std::uint32_t g_rng = 0x9e3779b9u;

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
          g.mi[static_cast<std::size_t>(row) *
                   static_cast<std::size_t>(mi_cols) +
               static_cast<std::size_t>(col)];
      if ((next_random() % 4u) == 0u) {
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
  p.sb_h4 = (next_random() & 1u) != 0u ? 8 : 16;
  p.ref_frame[0] = static_cast<std::int8_t>(random_range(0, g.num_refs - 1));
  p.ref_frame[1] = vm::kRefNone;
  p.is_compound = false;
  if (allow_compound && (next_random() % 3u) == 0u) {
    p.ref_frame[1] = static_cast<std::int8_t>(random_range(0, g.num_refs - 1));
    p.is_compound = true;
  }
  p.global_mv[0] = imc::Mv{random_range(-32, 32) * 8,
                           random_range(-32, 32) * 8};
  p.global_mv[1] = p.is_compound ? imc::Mv{random_range(-32, 32) * 8,
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
// GPU drivers
// ---------------------------------------------------------------------------

struct DeviceCopy {
  imc::MiInfo* mi = nullptr;
  int* frame_distance = nullptr;
};

bool upload_grid(av2enc::cuda::Arena& arena, const GridCase& g, DeviceCopy* out) {
  out->mi = arena.allocate_array<imc::MiInfo>(g.mi.size());
  if (out->mi == nullptr) {
    return false;
  }
  if (cudaMemcpy(out->mi, g.mi.data(), g.mi.size() * sizeof(imc::MiInfo),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    return false;
  }
  out->frame_distance = arena.allocate_array<int>(g.frame_distance.size());
  if (out->frame_distance == nullptr) {
    return false;
  }
  return cudaMemcpy(out->frame_distance, g.frame_distance.data(),
                    g.frame_distance.size() * sizeof(int),
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

void test_stack_batch(av2enc::cuda::Arena& arena, int cases) {
  std::printf("stack batch: %d randomized cases\n", cases);
  for (int trial = 0; trial < cases; ++trial) {
    const std::uint64_t mark = arena.mark();
    const GridCase grid = make_grid(random_range(8, 24), random_range(8, 24),
                                    random_range(1, 4));
    DeviceCopy device;
    CHECK(upload_grid(arena, grid, &device));
    if (device.mi == nullptr) {
      arena.rewind(mark);
      continue;
    }
    vm::StackParams host_params = make_params(grid, true);
    vm::StackParams gpu_params = host_params;
    gpu_params.mi = device.mi;
    gpu_params.frame_distance = device.frame_distance;

    vm::MvStack cpu;
    const vm::Status cpu_status = vm::build_mv_stack(host_params, &cpu);

    mvs::StackJob* d_job = arena.allocate_array<mvs::StackJob>(1);
    mvs::StackResult* d_result = arena.allocate_array<mvs::StackResult>(1);
    CHECK(d_job != nullptr && d_result != nullptr);
    if (d_job != nullptr && d_result != nullptr) {
      CHECK(cudaMemcpy(d_job, &gpu_params, sizeof(gpu_params),
                       cudaMemcpyHostToDevice) == cudaSuccess);
      CHECK(cudaMemset(d_result, 0, sizeof(*d_result)) == cudaSuccess);
      CUDA_OK(mvs::stack_build_batch(d_job, d_result, 1));
      mvs::StackResult host_result{};
      CUDA_OK(cudaMemcpy(&host_result, d_result, sizeof(host_result),
                         cudaMemcpyDeviceToHost));
      CHECK(host_result.status == static_cast<int>(cpu_status));
      if (cpu_status == vm::Status::kOk) {
        CHECK(same_stack(cpu, host_result.stack));
      }
    }
    arena.rewind(mark);
  }
}

void test_workspace(av2enc::cuda::Arena& arena, int cases) {
  std::printf("workspace: %d randomized cases\n", cases);
  mvs::MvStackConfig config;
  config.max_jobs = cases;
  const std::size_t need = mvs::required_device_bytes(config);
  std::printf("required_device_bytes(%d) = %zu\n", cases, need);

  const std::uint64_t before = arena.used();
  mvs::Workspace ws;
  std::string error;
  std::size_t reported = 0;
  CHECK(ws.create(arena, config, &error, &reported) == av2enc::cuda::Status::kOk);
  CHECK(reported == need);
  CHECK(arena.used() - before == need);
  CHECK(ws.created());

  const std::uint64_t mark = arena.mark();
  std::vector<GridCase> grids;
  std::vector<mvs::StackJob> host_jobs;
  std::vector<vm::MvStack> cpu_stacks;
  std::vector<vm::Status> cpu_status;
  for (int i = 0; i < cases; ++i) {
    grids.push_back(make_grid(random_range(8, 24), random_range(8, 24),
                              random_range(1, 4)));
    imc::MiInfo* mi =
        arena.allocate_array<imc::MiInfo>(grids.back().mi.size());
    int* dist = arena.allocate_array<int>(grids.back().frame_distance.size());
    CHECK(mi != nullptr && dist != nullptr);
    if (mi == nullptr || dist == nullptr) {
      continue;
    }
    CHECK(cudaMemcpy(mi, grids.back().mi.data(),
                     grids.back().mi.size() * sizeof(imc::MiInfo),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dist, grids.back().frame_distance.data(),
                     grids.back().frame_distance.size() * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);

    vm::StackParams host_params = make_params(grids.back(), true);
    vm::StackParams gpu_params = host_params;
    gpu_params.mi = mi;
    gpu_params.frame_distance = dist;
    host_jobs.push_back(gpu_params);

    vm::MvStack cpu;
    cpu_status.push_back(vm::build_mv_stack(host_params, &cpu));
    cpu_stacks.push_back(cpu);
  }

  CHECK(ws.upload_stack_jobs(host_jobs.data(), cases) ==
        av2enc::cuda::Status::kOk);
  const std::uint64_t used_after_upload = arena.used();
  CHECK(ws.run_stacks(cases) == av2enc::cuda::Status::kOk);
  std::vector<mvs::StackResult> results(static_cast<std::size_t>(cases));
  CHECK(ws.download_stacks(results.data(), cases) ==
        av2enc::cuda::Status::kOk);
  CHECK(arena.used() == used_after_upload);
  for (int i = 0; i < cases; ++i) {
    CHECK(results[static_cast<std::size_t>(i)].status ==
          static_cast<int>(cpu_status[static_cast<std::size_t>(i)]));
    if (cpu_status[static_cast<std::size_t>(i)] == vm::Status::kOk) {
      CHECK(same_stack(cpu_stacks[static_cast<std::size_t>(i)],
                       results[static_cast<std::size_t>(i)].stack));
    }
  }

  // Two launches must leave the arena watermark and peak flat.
  const std::uint64_t peak = arena.peak_used();
  CHECK(ws.run_stacks(cases) == av2enc::cuda::Status::kOk);
  CHECK(ws.run_stacks(cases) == av2enc::cuda::Status::kOk);
  CHECK(arena.peak_used() == peak);

  arena.rewind(mark);
}

void test_pairs_and_merge_sets(av2enc::cuda::Arena& arena, int cases) {
  std::printf("pair/merge: %d randomized cases\n", cases);
  mvs::MvStackConfig config;
  config.max_jobs = cases;
  mvs::Workspace ws;
  std::string error;
  CHECK(ws.create(arena, config, &error, nullptr) == av2enc::cuda::Status::kOk);

  const std::uint64_t mark = arena.mark();
  std::vector<mvs::PairJob> host_jobs;
  std::vector<mr::RefStackPair> cpu_pairs;
  std::vector<mr::CompoundMergeSet> cpu_sets;
  for (int i = 0; i < cases; ++i) {
    GridCase grid = make_grid(random_range(8, 24), random_range(8, 24),
                              random_range(2, 4));
    imc::MiInfo* mi = arena.allocate_array<imc::MiInfo>(grid.mi.size());
    int* dist = arena.allocate_array<int>(grid.frame_distance.size());
    CHECK(mi != nullptr && dist != nullptr);
    if (mi == nullptr || dist == nullptr) {
      continue;
    }
    CHECK(cudaMemcpy(mi, grid.mi.data(), grid.mi.size() * sizeof(imc::MiInfo),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(dist, grid.frame_distance.data(),
                     grid.frame_distance.size() * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess);
    vm::StackParams host_params = make_params(grid, false);
    host_params.ref_frame[0] = 0;
    host_params.ref_frame[1] = vm::kRefNone;
    host_params.is_compound = false;
    const std::int8_t ref1 =
        static_cast<std::int8_t>(random_range(1, grid.num_refs - 1));

    mr::RefStackPair cpu;
    std::string error_string;
    CHECK(mr::build_pair_stacks(host_params, 0, ref1, cpu, error_string));
    cpu_pairs.push_back(cpu);
    cpu_sets.push_back(mr::build_compound_merge_candidates(cpu));

    mvs::PairJob job;
    job.params = host_params;
    job.params.mi = mi;
    job.params.frame_distance = dist;
    job.ref0 = 0;
    job.ref1 = ref1;
    host_jobs.push_back(job);
  }

  CHECK(ws.upload_pair_jobs(host_jobs.data(), cases) ==
        av2enc::cuda::Status::kOk);
  CHECK(ws.run_pairs(cases) == av2enc::cuda::Status::kOk);
  std::vector<mvs::PairStacks> gpu_pairs(static_cast<std::size_t>(cases));
  CHECK(ws.download_pairs(gpu_pairs.data(), cases) ==
        av2enc::cuda::Status::kOk);
  CHECK(ws.run_merge_sets(cases) == av2enc::cuda::Status::kOk);
  std::vector<mvs::MergeSet> gpu_sets(static_cast<std::size_t>(cases));
  CHECK(ws.download_merge_sets(gpu_sets.data(), cases) ==
        av2enc::cuda::Status::kOk);
  const std::uint64_t after = arena.used();
  CHECK(ws.run_pairs(cases) == av2enc::cuda::Status::kOk);
  CHECK(arena.used() == after);

  for (int i = 0; i < cases; ++i) {
    const mr::RefStackPair& cpu = cpu_pairs[static_cast<std::size_t>(i)];
    const mvs::PairStacks& gpu = gpu_pairs[static_cast<std::size_t>(i)];
    CHECK(gpu.ok[0] == (cpu.ok[0] ? 1 : 0));
    CHECK(gpu.ok[1] == (cpu.ok[1] ? 1 : 0));
    CHECK(gpu.compound_ok == (cpu.compound_ok ? 1 : 0));
    CHECK(same_stack(cpu.list[0], gpu.list[0]));
    CHECK(same_stack(cpu.list[1], gpu.list[1]));
    CHECK(same_stack(cpu.compound, gpu.compound));
    CHECK(cpu.num_nearest[0] == gpu.num_nearest[0]);
    CHECK(cpu.num_nearest[1] == gpu.num_nearest[1]);

    const mr::CompoundMergeSet& cpu_set = cpu_sets[static_cast<std::size_t>(i)];
    const mvs::MergeSet& gpu_set = gpu_sets[static_cast<std::size_t>(i)];
    CHECK(gpu_set.count == static_cast<int>(cpu_set.candidates.size()));
    const int common =
        gpu_set.count < static_cast<int>(cpu_set.candidates.size())
            ? gpu_set.count
            : static_cast<int>(cpu_set.candidates.size());
    bool equal = true;
    for (int k = 0; k < common; ++k) {
      const mr::CompoundMergeCandidate& a =
          cpu_set.candidates[static_cast<std::size_t>(k)];
      const mvs::MergeCandidate& b = gpu_set.cand[k];
      if (!same_mv(a.mv[0], b.mv[0]) || !same_mv(a.mv[1], b.mv[1]) ||
          a.drl_index[0] != b.drl_index[0] ||
          a.drl_index[1] != b.drl_index[1] || a.weight != b.weight ||
          a.single_index != (b.single_index != 0)) {
        equal = false;
        break;
      }
    }
    CHECK(equal);
  }
  arena.rewind(mark);
}

void test_invalid_jobs(av2enc::cuda::Arena& arena) {
  std::printf("invalid jobs\n");
  GridCase grid = make_grid(12, 12, 2);
  DeviceCopy device;
  const std::uint64_t mark = arena.mark();
  CHECK(upload_grid(arena, grid, &device));

  std::vector<mvs::StackJob> jobs(4);
  for (int i = 0; i < 4; ++i) {
    jobs[static_cast<std::size_t>(i)] = make_params(grid, false);
    jobs[static_cast<std::size_t>(i)].ref_frame[0] = 0;
    jobs[static_cast<std::size_t>(i)].ref_frame[1] = vm::kRefNone;
    jobs[static_cast<std::size_t>(i)].is_compound = false;
    jobs[static_cast<std::size_t>(i)].mi = device.mi;
    jobs[static_cast<std::size_t>(i)].frame_distance = device.frame_distance;
  }
  jobs[0].mi = nullptr;
  jobs[1].bw4 = 0;
  jobs[2].sb_h4 = 6;
  jobs[3].max_drl_bits_minus_1 = 5;

  mvs::StackJob* d_jobs = arena.allocate_array<mvs::StackJob>(jobs.size());
  mvs::StackResult* d_results =
      arena.allocate_array<mvs::StackResult>(jobs.size());
  CHECK(d_jobs != nullptr && d_results != nullptr);
  CHECK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(mvs::StackJob),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  CUDA_OK(mvs::stack_build_batch(d_jobs, d_results, static_cast<int>(jobs.size())));
  std::vector<mvs::StackResult> results(jobs.size());
  CHECK(cudaMemcpy(results.data(), d_results,
                   jobs.size() * sizeof(mvs::StackResult),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  for (const mvs::StackResult& r : results) {
    CHECK(r.status != static_cast<int>(vm::Status::kOk));
  }
  CHECK(mvs::stack_build_batch(nullptr, nullptr, 0) == cudaSuccess);
  CHECK(mvs::stack_build_batch(nullptr, nullptr, 1) == cudaErrorInvalidValue);
  arena.rewind(mark);
}

void test_short_arena() {
  std::printf("short arena\n");
  mvs::MvStackConfig config;
  config.max_jobs = 8;
  const std::size_t need = mvs::required_device_bytes(config);

  av2enc::cuda::Arena tiny;
  CHECK(tiny.init(need + 64ull * 1024ull) == av2enc::cuda::Status::kOk);
  const std::uint64_t usable = tiny.usable_capacity();
  CHECK(usable > need);
  const std::size_t leave_free = need - 1;
  if (usable > leave_free) {
    void* pad = nullptr;
    CHECK(tiny.allocate_checked(
              static_cast<std::size_t>(usable - leave_free), &pad) ==
          av2enc::cuda::Status::kOk);
  }
  const std::uint64_t before = tiny.used();
  mvs::Workspace ws;
  std::string error;
  std::size_t reported = 0;
  const av2enc::cuda::Status status = ws.create(tiny, config, &error, &reported);
  CHECK(status == av2enc::cuda::Status::kArenaExhausted);
  CHECK(reported == need);
  CHECK(tiny.used() == before);
  CHECK(!ws.created());
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::printf("mv_stack_test: no CUDA device available\n");
    return 77;
  }
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  mvs::MvStackConfig probe;
  probe.max_jobs = 64;
  const std::size_t required = mvs::required_device_bytes(probe);
  av2enc::cuda::Arena arena;
  std::string error;
  const std::uint64_t capacity = required + 64ull * 1024ull * 1024ull;
  if (arena.init(capacity, av2enc::cuda::Arena::kAlignment, &error) !=
      av2enc::cuda::Status::kOk) {
    std::printf("mv_stack_test: arena init failed: %s\n", error.c_str());
    return 1;
  }

  test_stack_batch(arena, 400);
  test_workspace(arena, 48);
  test_pairs_and_merge_sets(arena, 48);
  test_invalid_jobs(arena);
  test_short_arena();

  std::printf("arena: used=%llu peak=%llu capacity=%llu\n",
              static_cast<unsigned long long>(arena.used()),
              static_cast<unsigned long long>(arena.peak_used()),
              static_cast<unsigned long long>(arena.capacity()));
  std::printf("mv_stack_cuda_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
