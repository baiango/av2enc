// Real-GPU bit-exact parity test for the AV2 §7.13.3.15 sub-PU grid.
//
// Build with build_gpu_test.sh (nvcc + a CUDA device).  The component writes
// the same `subpu_device.cuh` core the host-emulation test checks; this test
// drives it through the real kernels and compares every cell with the CPU
// reference in `src/encoder/m1/missing_bits/save_subpu.cpp`.
//
// Coverage:
//   * required_device_bytes, exact allocation, one-byte-short kArenaTooSmall;
//   * two-launch no-growth (used() and peak_used() are flat);
//   * device Tx_Width/Tx_Height/find_tx_size table vs the CPU tables;
//   * every 25 TX shape x 3 planes x 4 independent (subX,subY) x 49 origins;
//   * early return and rejection cases leave the grid untouched;
//   * a disjoint multi-job batch compared with the CPU decode-order result.
//
// Every device byte comes from one av2enc::cuda::Arena.  The test calls no
// cudaMalloc/cudaFree/cudaMallocHost.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/codec/filters/deblock/subpu/subpu.h"
#include "codec/intra/supplemental/save_subpu.h"

namespace {

namespace sub = av2enc::cuda::subpu;
namespace mb = av2enc::still::missing_bits;

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

void CheckEqInt(int actual, int expected, const char* what) {
  ++g_checks;
  if (actual != expected) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s (actual %d, expected %d)\n", what, actual,
                 expected);
  }
}

bool SameGrid(const mb::SubPuGrid& cpu, const sub::SubPuHostGrid& gpu) {
  if (cpu.mi_rows != gpu.mi_rows || cpu.mi_cols != gpu.mi_cols) {
    return false;
  }
  for (int plane = 0; plane < 2; ++plane) {
    if (cpu.size[plane] != gpu.size[plane]) return false;
    if (cpu.col_base[plane] != gpu.col_base[plane]) return false;
    if (cpu.row_base[plane] != gpu.row_base[plane]) return false;
  }
  return true;
}

// Copies a sentinel pattern into all six device arrays.  The test owns the
// host staging vectors; no device allocation happens here.
bool UploadSentinel(const sub::SubPuBuffers& buffers, std::uint8_t size_fill,
                    std::int32_t base_fill, cudaStream_t stream) {
  const std::size_t cells = static_cast<std::size_t>(buffers.grid.mi_rows) *
                            static_cast<std::size_t>(buffers.grid.mi_cols);
  std::vector<std::uint8_t> size(cells, size_fill);
  std::vector<std::int32_t> base(cells, base_fill);
  for (int plane = 0; plane < sub::kSubPuPlaneCount; ++plane) {
    if (cudaMemcpyAsync(buffers.grid.size[plane], size.data(),
                        cells * sizeof(std::uint8_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return false;
    }
    if (cudaMemcpyAsync(buffers.grid.col_base[plane], base.data(),
                        cells * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return false;
    }
    if (cudaMemcpyAsync(buffers.grid.row_base[plane], base.data(),
                        cells * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return false;
    }
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

void FillSentinelCpu(mb::SubPuGrid& cpu) {
  for (int plane = 0; plane < 2; ++plane) {
    for (std::size_t i = 0; i < cpu.size[plane].size(); ++i) {
      cpu.size[plane][i] = 0xA5;
      cpu.col_base[plane][i] = -99;
      cpu.row_base[plane][i] = -99;
    }
  }
}

struct CaseResult {
  bool cpu_ok = false;
  int device_code = -1;
  bool same_grid = false;
};

CaseResult RunCase(const sub::SubPuBuffers& buffers, int plane, int x, int y,
                   int w, int h, int sub_x, int sub_y, int pu_w, int pu_h,
                   bool sentinel, int mi_rows, int mi_cols) {
  CaseResult result;
  mb::SubPuGrid cpu;
  cpu.reset(mi_rows, mi_cols);
  if (sentinel) {
    FillSentinelCpu(cpu);
  }

  const sub::SubPuStatus reset = sub::reset_grid(buffers);
  Check(reset == sub::SubPuStatus::kOk, "reset_grid");
  if (sentinel && !UploadSentinel(buffers, 0xA5, -99, nullptr)) {
    Check(false, "UploadSentinel");
    return result;
  }

  sub::SubPuJob job;
  job.plane = plane;
  job.x = x;
  job.y = y;
  job.w = w;
  job.h = h;
  job.subsampling_x = sub_x;
  job.subsampling_y = sub_y;
  job.pu_width = pu_w;
  job.pu_height = pu_h;

  sub::SubPuResult gpu_result;
  const sub::SubPuStatus saved =
      sub::save_one(buffers, job, &gpu_result);
  Check(saved == sub::SubPuStatus::kOk, "save_one status");

  std::string error;
  const bool cpu_ok =
      mb::save_subpu(plane, x, y, w, h, sub_x, sub_y, pu_w, pu_h, cpu, &error);

  sub::SubPuHostGrid gpu;
  const sub::SubPuStatus downloaded = sub::download_grid(buffers, &gpu);
  Check(downloaded == sub::SubPuStatus::kOk, "download_grid");

  result.cpu_ok = cpu_ok;
  result.device_code = gpu_result.code;
  result.same_grid = SameGrid(cpu, gpu);
  return result;
}

void TestArena(const sub::SubPuConfig& config) {
  const std::uint64_t need = sub::required_device_bytes(config);
  Check(need > 0, "required_device_bytes positive");
  Check(need % av2enc::cuda::Arena::kAlignment == 0,
        "required_device_bytes alignment");
  std::printf("config: max_jobs=%d mi=%dx%d required_device_bytes=%llu\n",
              config.max_jobs, config.mi_rows, config.mi_cols,
              static_cast<unsigned long long>(need));

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need) == av2enc::cuda::Status::kOk, "exact arena init");
    sub::SubPuBuffers buffers;
    std::uint64_t reported = 0;
    std::string error;
    Check(sub::allocate_checked(arena, config, &buffers, &reported, &error) ==
              sub::SubPuStatus::kOk,
          "exact allocate_checked");
    CheckEqInt(static_cast<int>(reported), static_cast<int>(need),
               "exact reported bytes");
    Check(arena.used() == need, "exact used == required");
    Check(arena.remaining() == 0, "exact arena fully used");
    sub::deallocate(arena, buffers);
    Check(arena.used() == 0, "deallocate restores arena");
  }

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need - 1) == av2enc::cuda::Status::kOk,
          "one-byte-short arena init");
    sub::SubPuBuffers buffers;
    std::uint64_t reported = 0;
    std::string error;
    Check(sub::allocate_checked(arena, config, &buffers, &reported, &error) ==
              sub::SubPuStatus::kArenaTooSmall,
          "one byte short -> kArenaTooSmall");
    CheckEqInt(static_cast<int>(reported), static_cast<int>(need),
               "short arena reports required bytes");
    Check(arena.used() == 0, "short arena bump pointer unchanged");
    Check(buffers.d_jobs == nullptr, "short arena leaves no buffers");
  }
}

void TestTxTable(const sub::SubPuBuffers& buffers) {
  Check(sub::build_tx_table(buffers) == sub::SubPuStatus::kOk,
        "build_tx_table");
  sub::SubPuTxEntry table[sub::kTxSizeCount];
  Check(sub::download_tx_table(buffers, table, sub::kTxSizeCount) ==
            sub::SubPuStatus::kOk,
        "download_tx_table");
  for (int index = 0; index < sub::kTxSizeCount; ++index) {
    const auto cpu_size = static_cast<mb::TxSize>(index);
    CheckEqInt(table[index].size, index, "tx table size field");
    CheckEqInt(table[index].width, mb::tx_width(cpu_size),
               "tx table width parity");
    CheckEqInt(table[index].height, mb::tx_height(cpu_size),
               "tx table height parity");
    CheckEqInt(table[index].roundtrip, index, "tx table round-trip");
  }
}

void TestResetParity(const sub::SubPuBuffers& buffers) {
  mb::SubPuGrid cpu;
  cpu.reset(buffers.grid.mi_rows, buffers.grid.mi_cols);
  FillSentinelCpu(cpu);
  cpu.reset(buffers.grid.mi_rows, buffers.grid.mi_cols);

  Check(UploadSentinel(buffers, 0xA5, -99, nullptr), "reset sentinel upload");
  Check(sub::reset_grid(buffers) == sub::SubPuStatus::kOk, "reset parity call");
  sub::SubPuHostGrid gpu;
  Check(sub::download_grid(buffers, &gpu) == sub::SubPuStatus::kOk,
        "reset parity download");
  Check(SameGrid(cpu, gpu), "grid reset parity");
}

void TestExhaustive(const sub::SubPuBuffers& buffers) {
  const int origins[] = {0, 4, 8, 16, 32, 48, 64};
  const int subs[][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  long long cases = 0;
  long long mismatches = 0;

  for (int tx = 0; tx < sub::kTxSizeCount; ++tx) {
    const int w = sub::tx_width(tx);
    const int h = sub::tx_height(tx);
    for (int plane = 0; plane <= 2; ++plane) {
      for (const auto& sub_xy : subs) {
        for (int x : origins) {
          for (int y : origins) {
            const CaseResult result = RunCase(
                buffers, plane, x, y, w, h, sub_xy[0], sub_xy[1], 128, 128,
                false, buffers.grid.mi_rows, buffers.grid.mi_cols);
            const bool accept_ok =
                result.cpu_ok == (result.device_code == sub::kSubPuOk);
            if (!accept_ok || !result.same_grid) {
              ++mismatches;
              if (mismatches <= 5) {
                std::printf(
                    "mismatch: plane=%d x=%d y=%d w=%d h=%d sub=(%d,%d) "
                    "cpu_ok=%d code=%d same=%d\n",
                    plane, x, y, w, h, sub_xy[0], sub_xy[1],
                    result.cpu_ok ? 1 : 0, result.device_code,
                    result.same_grid ? 1 : 0);
              }
            }
            Check(accept_ok, "accept/reject parity (exhaustive)");
            Check(result.same_grid, "grid parity (exhaustive)");
            ++cases;
          }
        }
      }
    }
  }
  std::printf("exhaustive cases: %lld, mismatches: %lld\n", cases, mismatches);
}

void TestEarlyReturn(const sub::SubPuBuffers& buffers) {
  const int subs[][2] = {{0, 0}, {1, 1}};
  int cases = 0;
  for (int tx = 0; tx < sub::kTxSizeCount; ++tx) {
    const int w = sub::tx_width(tx);
    const int h = sub::tx_height(tx);
    for (int plane = 0; plane <= 2; ++plane) {
      for (const auto& sub_xy : subs) {
        for (int origin : {0, 4, 8, 16}) {
          const CaseResult result = RunCase(
              buffers, plane, origin, origin, w, h, sub_xy[0], sub_xy[1], w,
              h, true, buffers.grid.mi_rows, buffers.grid.mi_cols);
          Check(result.cpu_ok, "early return accepted");
          CheckEqInt(result.device_code, sub::kSubPuOk, "early return code");
          Check(result.same_grid, "early return writes nothing");
          ++cases;
        }
      }
    }
  }
  // The early return happens before find_tx_size, so a non-TX extent equal to
  // the PU size is still a successful no-op (the CPU reference returns true).
  const int odd[][2] = {{5, 5}, {6, 6}, {3, 7}, {33, 32}, {64, 12}};
  for (const auto& shape : odd) {
    for (int plane = 0; plane <= 2; ++plane) {
      const CaseResult result =
          RunCase(buffers, plane, 0, 0, shape[0], shape[1], 1, 1, shape[0],
                  shape[1], true, buffers.grid.mi_rows, buffers.grid.mi_cols);
      Check(result.cpu_ok, "odd early return accepted by CPU");
      CheckEqInt(result.device_code, sub::kSubPuOk, "odd early return code");
      Check(result.same_grid, "odd early return writes nothing");
      ++cases;
    }
  }
  std::printf("early-return cases: %d\n", cases);
}

void TestRejections(const sub::SubPuBuffers& buffers) {
  struct Case {
    int plane, x, y, w, h, sx, sy, pw, ph, rows, cols;
    int expected_code;
  };
  const Case cases[] = {
      {0, 0, 0, 5, 5, 0, 0, 128, 128, 32, 32, sub::kSubPuNoTxSize},
      {0, 0, 0, 6, 6, 0, 0, 128, 128, 32, 32, sub::kSubPuNoTxSize},
      {1, 0, 0, 9, 8, 1, 1, 128, 128, 32, 32, sub::kSubPuNoTxSize},
      {2, 4, 4, 12, 16, 0, 0, 128, 128, 32, 32, sub::kSubPuNoTxSize},
      {0, 0, 0, 33, 32, 0, 0, 128, 128, 32, 32, sub::kSubPuNoTxSize},
      {0, 128, 0, 8, 8, 0, 0, 256, 256, 32, 32, sub::kSubPuOutOfBounds},
      {0, 0, 128, 8, 8, 0, 0, 256, 256, 32, 32, sub::kSubPuOutOfBounds},
      {0, 124, 0, 8, 8, 0, 0, 256, 256, 32, 32, sub::kSubPuOutOfBounds},
      {1, 124, 0, 8, 8, 1, 1, 256, 256, 32, 32, sub::kSubPuOutOfBounds},
      {3, 0, 0, 8, 8, 0, 0, 128, 128, 32, 32, sub::kSubPuInvalidPlane},
      {-1, 0, 0, 8, 8, 0, 0, 128, 128, 32, 32, sub::kSubPuInvalidPlane},
      {0, -4, 0, 8, 8, 0, 0, 128, 128, 32, 32, sub::kSubPuInvalidRegion},
      {0, 0, -4, 8, 8, 0, 0, 128, 128, 32, 32, sub::kSubPuInvalidRegion},
      {0, 0, 0, 0, 8, 0, 0, 128, 128, 32, 32, sub::kSubPuInvalidRegion},
      {0, 0, 0, 8, 0, 0, 0, 128, 128, 32, 32, sub::kSubPuInvalidRegion},
      {0, 0, 0, 8, 8, 0, 0, -1, 128, 32, 32, sub::kSubPuInvalidPuSize},
      {0, 0, 0, 8, 8, 0, 0, 128, -1, 32, 32, sub::kSubPuInvalidPuSize},
      // Validation precedes the early return: w == PuWidth but PuHeight < 0.
      {0, 0, 0, 8, 8, 0, 0, 8, -1, 32, 32, sub::kSubPuInvalidPuSize},
      {3, 0, 0, 8, 8, 0, 0, 8, 8, 32, 32, sub::kSubPuInvalidPlane},
  };
  int checked = 0;
  for (const Case& c : cases) {
    const CaseResult result = RunCase(buffers, c.plane, c.x, c.y, c.w, c.h,
                                      c.sx, c.sy, c.pw, c.ph, true, c.rows,
                                      c.cols);
    Check(!result.cpu_ok, "CPU rejects rejection case");
    CheckEqInt(result.device_code, c.expected_code, "device rejection code");
    Check(result.same_grid, "rejection writes nothing");
    ++checked;
  }
  std::printf("rejection cases: %d\n", checked);
}

// Bounds rejection on a grid whose MI extent differs from the main 32x32 grid.
// The CPU oracle and the device buffers must agree on the same 4x4 extent.
void TestSmallGridBounds() {
  const sub::SubPuConfig config{4, 4, 4};
  const std::uint64_t need = sub::required_device_bytes(config);
  av2enc::cuda::Arena arena;
  Check(arena.init(need) == av2enc::cuda::Status::kOk, "small arena init");
  sub::SubPuBuffers buffers;
  Check(sub::allocate_checked(arena, config, &buffers) == sub::SubPuStatus::kOk,
        "small allocation");

  struct Case {
    int plane, x, y, w, h, sx, sy, pw, ph;
    int expected_code;
  };
  const Case cases[] = {
      {0, 16, 0, 8, 8, 0, 0, 128, 128, sub::kSubPuOutOfBounds},
      {0, 0, 16, 8, 8, 0, 0, 128, 128, sub::kSubPuOutOfBounds},
      {0, 8, 0, 16, 8, 0, 0, 128, 128, sub::kSubPuOutOfBounds},
      {1, 8, 0, 8, 8, 1, 1, 128, 128, sub::kSubPuOutOfBounds},
      {1, 12, 0, 8, 8, 0, 0, 128, 128, sub::kSubPuOutOfBounds},
      // Exactly fits: lumaCol=2, cols=2 -> column 4 == MiCols.
      {0, 8, 8, 8, 8, 0, 0, 128, 128, sub::kSubPuOk},
  };
  int checked = 0;
  for (const Case& c : cases) {
    const CaseResult result =
        RunCase(buffers, c.plane, c.x, c.y, c.w, c.h, c.sx, c.sy, c.pw, c.ph,
                true, config.mi_rows, config.mi_cols);
    Check(result.cpu_ok == (c.expected_code == sub::kSubPuOk),
          "small-grid accept parity");
    CheckEqInt(result.device_code, c.expected_code, "small-grid code");
    Check(result.same_grid, "small-grid writes parity");
    ++checked;
  }
  std::printf("small-grid bounds cases: %d\n", checked);
  sub::deallocate(arena, buffers);
}

void TestBatch(const sub::SubPuBuffers& buffers, av2enc::cuda::Arena& arena) {
  const int mi_rows = 16;
  const int mi_cols = 16;
  std::vector<sub::SubPuJob> jobs;
  jobs.reserve(static_cast<std::size_t>(mi_rows * mi_cols));
  for (int row = 0; row < mi_rows; ++row) {
    for (int col = 0; col < mi_cols; ++col) {
      sub::SubPuJob job;
      job.plane = 0;
      job.x = col * sub::kMiSize;
      job.y = row * sub::kMiSize;
      job.w = 4;
      job.h = 4;
      job.subsampling_x = 0;
      job.subsampling_y = 0;
      job.pu_width = 128;
      job.pu_height = 128;
      jobs.push_back(job);
    }
  }
  Check(static_cast<int>(jobs.size()) <= buffers.max_jobs,
        "batch fits max_jobs");

  auto run = [&](std::vector<sub::SubPuResult>* results) {
    Check(sub::reset_grid(buffers) == sub::SubPuStatus::kOk, "batch reset");
    return sub::save_batch(buffers, jobs.data(),
                           static_cast<int>(jobs.size()),
                           results != nullptr ? results->data() : nullptr);
  };

  std::vector<sub::SubPuResult> results(jobs.size());
  Check(run(&results) == sub::SubPuStatus::kOk, "batch save");

  mb::SubPuGrid cpu;
  cpu.reset(buffers.grid.mi_rows, buffers.grid.mi_cols);
  for (const sub::SubPuJob& job : jobs) {
    std::string error;
    Check(mb::save_subpu(job.plane, job.x, job.y, job.w, job.h,
                         job.subsampling_x, job.subsampling_y, job.pu_width,
                         job.pu_height, cpu, &error),
          "CPU batch job accepted");
  }
  sub::SubPuHostGrid gpu;
  Check(sub::download_grid(buffers, &gpu) == sub::SubPuStatus::kOk,
        "batch download");
  Check(SameGrid(cpu, gpu), "batch grid parity");
  for (const sub::SubPuResult& result : results) {
    CheckEqInt(result.code, sub::kSubPuOk, "batch result code");
  }

  // Two back-to-back launches must not grow the arena.
  const std::uint64_t used_before = arena.used();
  const std::uint64_t peak_before = arena.peak_used();
  Check(run(&results) == sub::SubPuStatus::kOk, "two-launch first");
  Check(run(&results) == sub::SubPuStatus::kOk, "two-launch second");
  Check(arena.used() == used_before, "two launches do not grow used()");
  Check(arena.peak_used() == peak_before,
        "two launches do not grow peak_used()");
  std::printf("batch jobs: %zu, two-launch used/peak flat: %llu/%llu\n",
              jobs.size(),
              static_cast<unsigned long long>(arena.used()),
              static_cast<unsigned long long>(arena.peak_used()));
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return 2;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  }

  const sub::SubPuConfig config{1024, 32, 32};
  TestArena(config);

  const std::uint64_t need = sub::required_device_bytes(config);
  av2enc::cuda::Arena arena;
  Check(arena.init(need) == av2enc::cuda::Status::kOk, "main arena init");
  sub::SubPuBuffers buffers;
  Check(sub::allocate_checked(arena, config, &buffers) ==
            sub::SubPuStatus::kOk,
        "main allocation");

  TestTxTable(buffers);
  TestResetParity(buffers);
  TestExhaustive(buffers);
  TestEarlyReturn(buffers);
  TestRejections(buffers);
  TestSmallGridBounds();
  TestBatch(buffers, arena);

  sub::deallocate(arena, buffers);
  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
