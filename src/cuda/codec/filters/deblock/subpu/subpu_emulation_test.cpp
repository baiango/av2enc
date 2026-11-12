// Host-emulation parity test for the shared AV2 §7.13.3.15 sub-PU core.
//
// The same `subpu_device.cuh` that the CUDA kernel includes is compiled here by
// the host compiler and compared, cell for cell, with the CPU reference in
// `src/encoder/m1/missing_bits/save_subpu.cpp`.  No GPU, no nvcc and no CUDA
// runtime are needed: `subpu_plan.cpp` and `Arena` are linked against a tiny
// host CUDA runtime stub.
//
// Coverage:
//   * Tx_Width / Tx_Height / find_tx_size against the CPU tables;
//   * every one of the 25 TX shapes x 3 planes x 4 independent (subX, subY)
//     combinations x 49 origins (0,4,...,64);
//   * early return (w == PuWidth && h == PuHeight) leaves the grid untouched;
//   * odd / non-TX shapes, invalid planes/sizes and grid bounds rejection;
//   * grid reset parity;
//   * required_device_bytes / allocate_checked / mark / rewind /
//     kArenaTooSmall / deallocate / download.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/codec/filters/deblock/subpu/subpu.h"
#include "codec/intra/supplemental/save_subpu.h"

namespace {

namespace mb = av2enc::still::missing_bits;
using av2enc::cuda::subpu::SubPuGridView;
using av2enc::cuda::subpu::SubPuHostGrid;

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

SubPuGridView ViewOf(SubPuHostGrid& grid) {
  SubPuGridView view;
  for (int plane = 0; plane < av2enc::cuda::subpu::kSubPuPlaneCount; ++plane) {
    view.size[plane] = grid.size[plane].data();
    view.col_base[plane] = grid.col_base[plane].data();
    view.row_base[plane] = grid.row_base[plane].data();
  }
  view.mi_rows = grid.mi_rows;
  view.mi_cols = grid.mi_cols;
  return view;
}

void FillSentinel(mb::SubPuGrid& cpu, SubPuHostGrid& dev) {
  for (int plane = 0; plane < 2; ++plane) {
    for (std::size_t i = 0; i < cpu.size[plane].size(); ++i) {
      cpu.size[plane][i] = 0xA5;
      cpu.col_base[plane][i] = -99;
      cpu.row_base[plane][i] = -77;
    }
    for (std::size_t i = 0; i < dev.size[plane].size(); ++i) {
      dev.size[plane][i] = 0xA5;
      dev.col_base[plane][i] = -99;
      dev.row_base[plane][i] = -77;
    }
  }
}

bool SameGrid(const mb::SubPuGrid& cpu, const SubPuHostGrid& dev) {
  if (cpu.mi_rows != dev.mi_rows || cpu.mi_cols != dev.mi_cols) {
    return false;
  }
  for (int plane = 0; plane < 2; ++plane) {
    if (cpu.size[plane] != dev.size[plane]) return false;
    if (cpu.col_base[plane] != dev.col_base[plane]) return false;
    if (cpu.row_base[plane] != dev.row_base[plane]) return false;
  }
  return true;
}

// Runs one descriptor through both implementations on grids with the same
// starting contents and compares the results.
struct CaseResult {
  bool cpu_ok = false;
  int device_code = -1;
  bool same_grid = false;
};

CaseResult RunCase(int plane, int x, int y, int w, int h, int sub_x,
                   int sub_y, int pu_w, int pu_h, int mi_rows, int mi_cols,
                   bool sentinel) {
  mb::SubPuGrid cpu;
  cpu.reset(mi_rows, mi_cols);
  SubPuHostGrid dev;
  dev.reset(mi_rows, mi_cols);
  if (sentinel) {
    FillSentinel(cpu, dev);
  }
  SubPuGridView view = ViewOf(dev);

  std::string error;
  const bool cpu_ok = mb::save_subpu(plane, x, y, w, h, sub_x, sub_y, pu_w,
                                     pu_h, cpu, &error);
  const int code = av2enc::cuda::subpu::save_subpu(
      plane, x, y, w, h, sub_x, sub_y, pu_w, pu_h, view);

  CaseResult result;
  result.cpu_ok = cpu_ok;
  result.device_code = code;
  result.same_grid = SameGrid(cpu, dev);
  return result;
}

void TestTxTables() {
  using namespace av2enc::cuda::subpu;
  for (int index = 0; index < kTxSizeCount; ++index) {
    const auto cpu_size = static_cast<mb::TxSize>(index);
    CheckEqInt(tx_width(index), mb::tx_width(cpu_size), "tx_width parity");
    CheckEqInt(tx_height(index), mb::tx_height(cpu_size), "tx_height parity");
    CheckEqInt(find_tx_size(tx_width(index), tx_height(index)), index,
               "find_tx_size round-trip");
    CheckEqInt(mb::find_tx_size(mb::tx_width(cpu_size),
                                mb::tx_height(cpu_size)),
               index, "CPU find_tx_size round-trip");
  }
  const int bad[][2] = {{0, 0},   {4, 12},  {12, 4}, {6, 6},   {5, 5},
                        {64, 128}, {-4, 4}, {4, 3}, {33, 32}, {64, 4 + 1}};
  for (const auto& pair : bad) {
    CheckEqInt(find_tx_size(pair[0], pair[1]), kTxInvalid,
               "find_tx_size rejects non-TX shape");
  }
}

void TestResetParity() {
  using namespace av2enc::cuda::subpu;
  for (const int dims : {1, 3, 8, 17, 32}) {
    mb::SubPuGrid cpu;
    cpu.reset(dims, dims + 1);
    SubPuHostGrid dev;
    dev.reset(dims, dims + 1);
    FillSentinel(cpu, dev);
    cpu.reset(dims, dims + 1);
    SubPuGridView view = ViewOf(dev);
    reset_grid(view);
    Check(SameGrid(cpu, dev), "grid reset parity");
  }
}

void TestExhaustive() {
  using namespace av2enc::cuda::subpu;
  const int origins[] = {0, 4, 8, 16, 32, 48, 64};
  const int subs[][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  const int rows = 32;
  const int cols = 32;
  long long cases = 0;

  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    const int w = tx_width(tx);
    const int h = tx_height(tx);
    for (int plane = 0; plane <= 2; ++plane) {
      for (const auto& sub : subs) {
        for (int x : origins) {
          for (int y : origins) {
            const CaseResult result =
                RunCase(plane, x, y, w, h, sub[0], sub[1], 128, 128, rows,
                        cols, false);
            Check(result.cpu_ok == (result.device_code == kSubPuOk),
                  "accept/reject parity (exhaustive)");
            Check(result.same_grid, "grid parity (exhaustive)");
            ++cases;
          }
        }
      }
    }
  }
  std::printf("exhaustive cases: %lld\n", cases);
}

void TestEarlyReturn() {
  using namespace av2enc::cuda::subpu;
  const int subs[][2] = {{0, 0}, {1, 1}};
  int cases = 0;
  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    const int w = tx_width(tx);
    const int h = tx_height(tx);
    for (int plane = 0; plane <= 2; ++plane) {
      for (const auto& sub : subs) {
        for (int origin : {0, 4, 8, 16}) {
          const CaseResult result =
              RunCase(plane, origin, origin, w, h, sub[0], sub[1], w, h, 32,
                      32, true);
          Check(result.cpu_ok, "early return accepted");
          CheckEqInt(result.device_code, kSubPuOk, "early return code");
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
          RunCase(plane, 0, 0, shape[0], shape[1], 1, 1, shape[0], shape[1],
                  32, 32, true);
      Check(result.cpu_ok, "odd early return accepted by CPU");
      CheckEqInt(result.device_code, kSubPuOk, "odd early return code");
      Check(result.same_grid, "odd early return writes nothing");
      ++cases;
    }
  }
  std::printf("early-return cases: %d\n", cases);
}

void TestRejections() {
  using namespace av2enc::cuda::subpu;
  struct Case {
    int plane, x, y, w, h, sx, sy, pw, ph, rows, cols;
    int expected_code;
  };
  const Case cases[] = {
      // Odd / non-TX shapes: find_tx_size fails before any write.
      {0, 0, 0, 5, 5, 0, 0, 128, 128, 32, 32, kSubPuNoTxSize},
      {0, 0, 0, 6, 6, 0, 0, 128, 128, 32, 32, kSubPuNoTxSize},
      {1, 0, 0, 9, 8, 1, 1, 128, 128, 32, 32, kSubPuNoTxSize},
      {2, 4, 4, 12, 16, 0, 0, 128, 128, 32, 32, kSubPuNoTxSize},
      {0, 0, 0, 33, 32, 0, 0, 128, 128, 32, 32, kSubPuNoTxSize},
      // Grid bounds.
      {0, 128, 0, 8, 8, 0, 0, 256, 256, 32, 32, kSubPuOutOfBounds},
      {0, 0, 128, 8, 8, 0, 0, 256, 256, 32, 32, kSubPuOutOfBounds},
      {0, 124, 0, 8, 8, 0, 0, 256, 256, 32, 32, kSubPuOutOfBounds},
      {1, 124, 0, 8, 8, 1, 1, 256, 256, 32, 32, kSubPuOutOfBounds},
      // Bounds on a tiny grid.
      {0, 16, 0, 8, 8, 0, 0, 128, 128, 4, 4, kSubPuOutOfBounds},
      {0, 0, 16, 8, 8, 0, 0, 128, 128, 4, 4, kSubPuOutOfBounds},
      {1, 8, 0, 8, 8, 1, 1, 128, 128, 2, 4, kSubPuOutOfBounds},
      // Argument validation, in the CPU reference order.
      {3, 0, 0, 8, 8, 0, 0, 128, 128, 32, 32, kSubPuInvalidPlane},
      {-1, 0, 0, 8, 8, 0, 0, 128, 128, 32, 32, kSubPuInvalidPlane},
      {0, -4, 0, 8, 8, 0, 0, 128, 128, 32, 32, kSubPuInvalidRegion},
      {0, 0, -4, 8, 8, 0, 0, 128, 128, 32, 32, kSubPuInvalidRegion},
      {0, 0, 0, 0, 8, 0, 0, 128, 128, 32, 32, kSubPuInvalidRegion},
      {0, 0, 0, 8, 0, 0, 0, 128, 128, 32, 32, kSubPuInvalidRegion},
      {0, 0, 0, 8, 8, 0, 0, -1, 128, 32, 32, kSubPuInvalidPuSize},
      {0, 0, 0, 8, 8, 0, 0, 128, -1, 32, 32, kSubPuInvalidPuSize},
      // Validation precedes the early return: w == PuWidth but PuHeight < 0.
      {0, 0, 0, 8, 8, 0, 0, 8, -1, 32, 32, kSubPuInvalidPuSize},
      // Plane out of range wins over the early-return check.
      {3, 0, 0, 8, 8, 0, 0, 8, 8, 32, 32, kSubPuInvalidPlane},
  };
  int checked = 0;
  for (const Case& c : cases) {
    const CaseResult result = RunCase(c.plane, c.x, c.y, c.w, c.h, c.sx, c.sy,
                                      c.pw, c.ph, c.rows, c.cols, true);
    Check(!result.cpu_ok, "CPU rejects rejection case");
    CheckEqInt(result.device_code, c.expected_code, "device rejection code");
    Check(result.same_grid, "rejection writes nothing");
    ++checked;
  }
  std::printf("rejection cases: %d\n", checked);
}

void TestIndependentSubsampling() {
  using namespace av2enc::cuda::subpu;
  // Chroma origin 8 in x maps to luma column 4 with subX=1 and column 2 with
  // subX=0; subY must only affect the row.  This pins the independent shifts.
  mb::SubPuGrid cpu;
  cpu.reset(16, 16);
  SubPuHostGrid dev;
  dev.reset(16, 16);
  SubPuGridView view = ViewOf(dev);

  Check(mb::save_subpu(1, 8, 12, 8, 8, 1, 0, 128, 128, cpu),
        "independent subsampling CPU write");
  CheckEqInt(save_subpu(1, 8, 12, 8, 8, 1, 0, 128, 128, view), kSubPuOk,
             "independent subsampling accepted");
  Check(SameGrid(cpu, dev), "independent subsampling parity");
  // lumaCol = 8 >> 1 = 4, lumaRow = 12 >> 2 = 3, rows = 8 >> 2 = 2,
  // cols = 8 >> 1 = 4.
  CheckEqInt(dev.col_base[1][3 * 16 + 4], 4, "subX=1 column base");
  CheckEqInt(dev.row_base[1][3 * 16 + 4], 3, "subY=0 row base");
  CheckEqInt(dev.size[1][4 * 16 + 7], 1, "8x8 -> TX_8X8");
}

void TestArena() {
  using namespace av2enc::cuda::subpu;
  const SubPuConfig config{17, 9, 13};
  const std::uint64_t need = required_device_bytes(config);
  Check(need > 0, "required_device_bytes positive");
  // Manual recomputation of the documented layout.
  const std::uint64_t cells = 9ull * 13ull;
  const auto up = [](std::uint64_t v) {
    return (v + 255ull) & ~static_cast<std::uint64_t>(255ull);
  };
  const std::uint64_t expected =
      static_cast<std::uint64_t>(kSubPuPlaneCount) *
          (up(cells) + up(cells * 4) + up(cells * 4)) +
      up(17ull * sizeof(SubPuJob)) + up(17ull * sizeof(SubPuResult)) +
      up(static_cast<std::uint64_t>(kTxSizeCount) * sizeof(SubPuTxEntry));
  Check(need == expected, "required_device_bytes layout");
  Check(required_device_bytes(SubPuConfig{0, 4, 4}) == 0,
        "invalid max_jobs -> 0");
  Check(required_device_bytes(SubPuConfig{1, 0, 4}) == 0,
        "invalid rows -> 0");

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need) == av2enc::cuda::Status::kOk, "arena init exact");
    SubPuBuffers buffers;
    std::uint64_t reported = 0;
    std::string error;
    Check(allocate_checked(arena, config, &buffers, &reported, &error) ==
              SubPuStatus::kOk,
          "allocate_checked exact fit");
    CheckEqInt(static_cast<int>(reported), static_cast<int>(need),
               "reported required bytes");
    Check(arena.used() == need, "used == required after allocation");
    Check(buffers.max_jobs == config.max_jobs, "max_jobs stored");
    Check(buffers.grid.mi_rows == config.mi_rows &&
              buffers.grid.mi_cols == config.mi_cols,
          "grid dims stored");
    Check(buffers.persistent_mark != kNoArenaMark, "mark recorded");

    deallocate(arena, buffers);
    Check(arena.used() == 0, "deallocate restores the arena");
    Check(buffers.persistent_mark == kNoArenaMark, "deallocate clears mark");
  }

  {
    // mark()/rewind() bracket scratch without disturbing the buffers.
    av2enc::cuda::Arena arena;
    Check(arena.init(need + av2enc::cuda::Arena::kAlignment) ==
              av2enc::cuda::Status::kOk,
          "scratch arena init");
    SubPuBuffers buffers;
    Check(allocate_checked(arena, config, &buffers) == SubPuStatus::kOk,
          "scratch allocation");
    Check(arena.used() == need, "scratch used == required");

    const std::uint64_t probe_mark = mark(arena);
    void* probe = nullptr;
    Check(arena.allocate_checked(av2enc::cuda::Arena::kAlignment, &probe) ==
              av2enc::cuda::Status::kOk,
          "probe allocation");
    Check(rewind(arena, probe_mark), "rewind probe");
    Check(arena.used() == need, "used restored by rewind");

    deallocate(arena, buffers);
    Check(arena.used() == 0, "scratch deallocate restores the arena");
  }

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need - 1) == av2enc::cuda::Status::kOk,
          "short arena init");
    SubPuBuffers buffers;
    std::uint64_t reported = 0;
    std::string error;
    Check(allocate_checked(arena, config, &buffers, &reported, &error) ==
              SubPuStatus::kArenaTooSmall,
          "one byte short -> kArenaTooSmall");
    CheckEqInt(static_cast<int>(reported), static_cast<int>(need),
               "short arena reports required bytes");
    Check(arena.used() == 0, "short arena bump pointer unchanged");
    Check(buffers.d_jobs == nullptr, "short arena leaves no buffers");
  }

  {
    av2enc::cuda::Arena arena;
    Check(arena.init(need * 2) == av2enc::cuda::Status::kOk,
          "double arena init");
    SubPuBuffers first;
    SubPuBuffers second;
    Check(allocate_checked(arena, config, &first) == SubPuStatus::kOk,
          "first allocation");
    Check(allocate_checked(arena, config, &second) == SubPuStatus::kOk,
          "second allocation");
    Check(second.persistent_mark > first.persistent_mark,
          "second mark above first");
    deallocate(arena, second);
    Check(arena.used() == need, "rewind to first allocation");
    Check(first.grid.size[0] != nullptr, "first allocation still valid");
    deallocate(arena, first);
  }
}

void TestDownload() {
  using namespace av2enc::cuda::subpu;
  const SubPuConfig config{4, 8, 8};
  const std::uint64_t need = required_device_bytes(config);
  av2enc::cuda::Arena arena;
  Check(arena.init(need) == av2enc::cuda::Status::kOk, "download arena init");
  SubPuBuffers buffers;
  Check(allocate_checked(arena, config, &buffers) == SubPuStatus::kOk,
        "download allocation");

  SubPuGridView view = buffers.grid.view();
  reset_grid(view);
  Check(save_subpu(0, 0, 0, 8, 8, 0, 0, 128, 128, view) == kSubPuOk,
        "download write luma");
  Check(save_subpu(1, 0, 0, 8, 8, 1, 1, 128, 128, view) == kSubPuOk,
        "download write chroma");

  SubPuHostGrid host;
  Check(download_grid(buffers, &host) == SubPuStatus::kOk, "download_grid");
  CheckEqInt(host.mi_rows, 8, "download rows");
  CheckEqInt(host.mi_cols, 8, "download cols");
  CheckEqInt(host.size[0][0], 1, "downloaded luma TX_8X8");
  CheckEqInt(host.col_base[0][3 * 8 + 3], 0, "downloaded luma base");
  CheckEqInt(host.size[1][3 * 8 + 3], 1, "downloaded chroma TX_8X8");
  CheckEqInt(host.col_base[1][3 * 8 + 3], 0, "downloaded chroma base");

  deallocate(arena, buffers);
}

}  // namespace

int main() {
  TestTxTables();
  TestResetParity();
  TestExhaustive();
  TestEarlyReturn();
  TestRejections();
  TestIndependentSubsampling();
  TestArena();
  TestDownload();

  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
