// CPU-vs-GPU parity test for the AV2 lossless inter CUDA component.
//
// Every case builds random host data, runs the independent CPU reference
// (`src/encoder/lossless/inter/`) and the CUDA kernels on identical input, and
// requires integer/byte equality (0 LSB).
//
// Device memory comes exclusively from one persistent `av2enc::cuda::Arena`:
// the workspace is sized from `required_device_bytes()`, the data pool is bump
// allocated once, and the test proves two launches do not grow the arena and
// that mark()/rewind() scratch is net-zero.  An arena one byte short of the
// requirement must fail with `Status::kArenaExhausted` and report the size.
//
// Build/run: ./build_and_test.sh (needs nvcc + a CUDA device).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/lossless_inter/lossless_inter.h"
#include "encoder/lossless/inter/lossless_inter.h"

namespace {

namespace li = av2enc::cuda::lossless_inter;
namespace cpu = av2enc::lossless::inter;
using av2enc::cuda::Arena;
using av2enc::cuda::Status;

int g_checks = 0;
int g_failures = 0;
long long g_cases = 0;

void check(bool condition, const char* what, int a = 0, int b = 0, int c = 0,
           int d = 0) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL: %s (%d %d %d %d)\n", what, a, b, c, d);
    }
  }
}

std::uint32_t g_rng = 0x51ed270bu;
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

constexpr int kMaxW = 32;
constexpr int kMaxH = 32;
constexpr int kMaxBlockArea = kMaxW * kMaxH;
constexpr int kMaxJobs = 256;
constexpr int kPlaneStride = 64;
constexpr int kPlaneRows = 64;
constexpr int kPlaneBytes = kPlaneStride * kPlaneRows;

bool sync_stream(const char* what) {
  const cudaError_t error = cudaStreamSynchronize(nullptr);
  check(error == cudaSuccess, what);
  if (error != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(error));
    return false;
  }
  return true;
}

bool copy_up(const void* host, void* device, std::size_t bytes,
             const char* what) {
  const cudaError_t error =
      cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, nullptr);
  check(error == cudaSuccess, what);
  return error == cudaSuccess;
}

bool copy_down(const void* device, void* host, std::size_t bytes,
               const char* what) {
  const cudaError_t error =
      cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, nullptr);
  check(error == cudaSuccess, what);
  return error == cudaSuccess;
}

struct DeviceData {
  int* block_a = nullptr;  // kMaxJobs * kMaxBlockArea residual/levels
  int* block_b = nullptr;
  int* wht_a = nullptr;  // kMaxJobs * 4
  int* wht_b = nullptr;
  std::uint8_t* pred = nullptr;  // kMaxJobs * kPlaneBytes
  std::uint8_t* recon = nullptr;
};

struct Fixture {
  Arena arena;
  li::Workspace workspace;
  DeviceData data;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t pool_bytes = 0;
};

bool setup_fixture(Fixture& fixture) {
  li::LaunchConfig cfg;
  cfg.max_jobs = kMaxJobs;
  cfg.max_block_w = kMaxW;
  cfg.max_block_h = kMaxH;
  fixture.workspace_bytes = li::required_device_bytes(cfg);

  // Data pool: two int block buffers, two WHT buffers, pred and recon planes.
  const std::uint64_t block_ints =
      static_cast<std::uint64_t>(kMaxJobs) * kMaxBlockArea;
  fixture.pool_bytes =
      block_ints * sizeof(int) * 2 +
      static_cast<std::uint64_t>(kMaxJobs) * 4 * sizeof(int) * 2 +
      static_cast<std::uint64_t>(kMaxJobs) * kPlaneBytes * 2;

  const std::uint64_t capacity =
      fixture.workspace_bytes + fixture.pool_bytes + 64 * 1024;
  std::string error;
  Status status = fixture.arena.init(capacity, Arena::kAlignment, &error);
  check(status == Status::kOk, "arena init");
  if (status != Status::kOk) {
    std::printf("arena init: %s\n", error.c_str());
    return false;
  }

  std::uint64_t reported = 0;
  status = fixture.workspace.init(fixture.arena, cfg, &reported, &error);
  check(status == Status::kOk, "workspace init");
  if (status != Status::kOk) {
    std::printf("workspace init: %s\n", error.c_str());
    return false;
  }
  check(reported == fixture.workspace_bytes, "required_device_bytes stable");
  check(fixture.arena.used() == fixture.workspace_bytes,
        "workspace used exactly the reported bytes", 0,
        static_cast<int>(fixture.arena.used()),
        static_cast<int>(fixture.workspace_bytes));

  fixture.data.block_a =
      fixture.arena.allocate_array<int>(static_cast<std::size_t>(block_ints));
  fixture.data.block_b =
      fixture.arena.allocate_array<int>(static_cast<std::size_t>(block_ints));
  fixture.data.wht_a = fixture.arena.allocate_array<int>(
      static_cast<std::size_t>(kMaxJobs) * 4);
  fixture.data.wht_b = fixture.arena.allocate_array<int>(
      static_cast<std::size_t>(kMaxJobs) * 4);
  fixture.data.pred = fixture.arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(kMaxJobs) * kPlaneBytes);
  fixture.data.recon = fixture.arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(kMaxJobs) * kPlaneBytes);
  check(fixture.data.block_a != nullptr && fixture.data.block_b != nullptr &&
            fixture.data.wht_a != nullptr && fixture.data.wht_b != nullptr &&
            fixture.data.pred != nullptr && fixture.data.recon != nullptr,
        "data pool allocation");
  return fixture.data.block_a != nullptr;
}

// ---------------------------------------------------------------------------
// 4-sample WHT
// ---------------------------------------------------------------------------

void test_wht(Fixture& fixture) {
  std::printf("-- WHT 4-sample batch\n");
  const int checks_before = g_failures;
  const int batch = 256;
  std::vector<int> host_in(static_cast<std::size_t>(batch) * 4);
  std::vector<int> host_out(static_cast<std::size_t>(batch) * 4);
  std::vector<li::WhtJob> jobs(static_cast<std::size_t>(batch));

  for (int round = 0; round < 8; ++round) {
    for (int i = 0; i < batch; ++i) {
      for (int k = 0; k < 4; ++k) {
        host_in[static_cast<std::size_t>(i) * 4 + k] =
            (round & 1) ? (random_range(-3000, 3000) | 1)
                        : random_range(-3000, 3000);
      }
      jobs[static_cast<std::size_t>(i)].in =
          fixture.data.wht_a + static_cast<std::size_t>(i) * 4;
      jobs[static_cast<std::size_t>(i)].out =
          fixture.data.wht_b + static_cast<std::size_t>(i) * 4;
    }
    copy_up(host_in.data(), fixture.data.wht_a,
            host_in.size() * sizeof(int), "wht upload");
    Status status = li::wht4_forward(fixture.workspace, jobs.data(), batch);
    check(status == Status::kOk, "wht4_forward");
    copy_down(fixture.data.wht_b, host_out.data(),
              host_out.size() * sizeof(int), "wht download");
    if (!sync_stream("wht sync")) return;

    for (int i = 0; i < batch; ++i) {
      int want[4];
      cpu::forward_wht4(&host_in[static_cast<std::size_t>(i) * 4], want);
      ++g_cases;
      for (int k = 0; k < 4; ++k) {
        check(host_out[static_cast<std::size_t>(i) * 4 + k] == want[k],
              "gpu wht forward", i, k,
              host_out[static_cast<std::size_t>(i) * 4 + k], want[k]);
      }
    }

    for (int shift = 0; shift <= 5; ++shift) {
      for (int i = 0; i < batch; ++i) {
        jobs[static_cast<std::size_t>(i)].shift = shift;
      }
      status = li::wht4_inverse(fixture.workspace, jobs.data(), batch);
      check(status == Status::kOk, "wht4_inverse");
      copy_down(fixture.data.wht_b, host_out.data(),
                host_out.size() * sizeof(int), "wht inverse download");
      if (!sync_stream("wht inverse sync")) return;
      for (int i = 0; i < batch; ++i) {
        int want[4];
        cpu::inverse_wht4(&host_in[static_cast<std::size_t>(i) * 4], shift,
                          want);
        ++g_cases;
        for (int k = 0; k < 4; ++k) {
          check(host_out[static_cast<std::size_t>(i) * 4 + k] == want[k],
                "gpu wht inverse", i, k, shift,
                host_out[static_cast<std::size_t>(i) * 4 + k]);
        }
      }
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// 2D forward / inverse, all sizes
// ---------------------------------------------------------------------------

void test_blocks(Fixture& fixture) {
  std::printf("-- 2D forward/inverse (all 4/8/16/32 sizes, DCT_DCT + IDTX)\n");
  const int checks_before = g_failures;
  const int dims[4] = {4, 8, 16, 32};
  const int rect[][2] = {{4, 8},  {8, 4},  {4, 16},  {16, 4},
                         {8, 16}, {16, 8}, {4, 32},  {32, 4},
                         {16, 32}, {32, 16}};
  std::vector<int> host_in(static_cast<std::size_t>(kMaxJobs) * kMaxBlockArea);
  std::vector<int> host_out(host_in.size());
  std::vector<li::ForwardJob> forward_jobs;
  std::vector<li::InverseJob> inverse_jobs;
  struct Meta {
    int w;
    int h;
    int n;
    int offset;
    int dct;
  };
  std::vector<Meta> meta;

  int job = 0;
  auto add_case = [&](int w, int h, int dct) {
    const int n = w * h;
    const int offset = job * kMaxBlockArea;
    for (int i = 0; i < n; ++i) {
      host_in[static_cast<std::size_t>(offset) + static_cast<std::size_t>(i)] =
          random_range(-255, 255);
    }
    meta.push_back({w, h, n, offset, dct});
    li::ForwardJob forward;
    forward.residual = fixture.data.block_a + offset;
    forward.levels = fixture.data.block_b + offset;
    forward.w = w;
    forward.h = h;
    forward.dct_dct = dct;
    forward_jobs.push_back(forward);
    li::InverseJob inverse;
    inverse.levels = fixture.data.block_b + offset;
    inverse.residual = fixture.data.block_a + offset;
    inverse.w = w;
    inverse.h = h;
    inverse.dct_dct = dct;
    inverse_jobs.push_back(inverse);
    ++job;
  };

  for (int w : dims) {
    for (int h : dims) {
      for (int dct = 0; dct <= 1; ++dct) add_case(w, h, dct);
    }
  }
  for (const auto& dim : rect) {
    for (int dct = 0; dct <= 1; ++dct) add_case(dim[0], dim[1], dct);
  }
  check(job <= kMaxJobs, "block batch fits");

  copy_up(host_in.data(), fixture.data.block_a,
          static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
          "block upload");
  Status status =
      li::forward_blocks(fixture.workspace, forward_jobs.data(), job);
  check(status == Status::kOk, "forward_blocks");
  copy_down(fixture.data.block_b, host_out.data(),
            static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
            "levels download");
  if (!sync_stream("forward sync")) return;

  // Compare forward output against the CPU.
  for (std::size_t index = 0; index < meta.size(); ++index) {
    const Meta& m = meta[index];
    std::vector<int> want(static_cast<std::size_t>(m.n));
    cpu::forward_lossless_block(
        &host_in[static_cast<std::size_t>(m.offset)], m.w, m.h, m.dct != 0,
        want.data());
    ++g_cases;
    check(std::memcmp(want.data(),
                      &host_out[static_cast<std::size_t>(m.offset)],
                      sizeof(int) * static_cast<std::size_t>(m.n)) == 0,
          "gpu forward_lossless_block", m.w, m.h, m.dct);
  }

  // Inverse the GPU levels; the result must equal the original residual.
  status = li::inverse_blocks(fixture.workspace, inverse_jobs.data(), job);
  check(status == Status::kOk, "inverse_blocks");
  copy_down(fixture.data.block_a, host_out.data(),
            static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
            "residual download");
  if (!sync_stream("inverse sync")) return;
  for (const Meta& m : meta) {
    ++g_cases;
    check(std::memcmp(&host_in[static_cast<std::size_t>(m.offset)],
                      &host_out[static_cast<std::size_t>(m.offset)],
                      sizeof(int) * static_cast<std::size_t>(m.n)) == 0,
          "gpu block round trip", m.w, m.h, m.dct);
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// DPCM + forward-with-DPCM
// ---------------------------------------------------------------------------

void test_dpcm(Fixture& fixture) {
  std::printf("-- DPCM difference/cumsum + forward pipeline\n");
  const int checks_before = g_failures;
  const int dims[][2] = {{4, 4}, {8, 8}, {4, 8}, {8, 4},
                         {16, 16}, {32, 32}, {4, 32}, {32, 4}};
  std::vector<int> host_in(static_cast<std::size_t>(kMaxJobs) * kMaxBlockArea);
  std::vector<int> host_out(host_in.size());
  std::vector<li::DpcmJob> diff_jobs;
  std::vector<li::DpcmJob> cum_jobs;
  std::vector<li::ForwardJob> forward_jobs;
  struct Meta {
    int w;
    int h;
    int n;
    int offset;
    int vertical;
    int dct;
    bool forward;
  };
  std::vector<Meta> meta;

  int job = 0;
  for (const auto& dim : dims) {
    const int w = dim[0];
    const int h = dim[1];
    for (int vertical = 0; vertical <= 1; ++vertical) {
      const int n = w * h;
      const int offset = job * kMaxBlockArea;
      for (int i = 0; i < n; ++i) {
        host_in[static_cast<std::size_t>(offset) + i] =
            random_range(-255, 255);
      }
      li::DpcmJob diff;
      diff.in = fixture.data.block_a + offset;
      diff.out = fixture.data.block_b + offset;
      diff.w = w;
      diff.h = h;
      diff.vertical = vertical;
      diff_jobs.push_back(diff);
      li::DpcmJob cum;
      cum.in = fixture.data.block_b + offset;
      cum.out = fixture.data.block_a + offset;
      cum.w = w;
      cum.h = h;
      cum.vertical = vertical;
      cum_jobs.push_back(cum);
      meta.push_back({w, h, n, offset, vertical, 0, false});
      ++job;
    }
  }

  copy_up(host_in.data(), fixture.data.block_a,
          static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
          "dpcm upload");
  Status status =
      li::dpcm_difference(fixture.workspace, diff_jobs.data(), job);
  check(status == Status::kOk, "dpcm_difference");
  copy_down(fixture.data.block_b, host_out.data(),
            static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
            "dpcm diff download");
  if (!sync_stream("dpcm diff sync")) return;
  for (const Meta& m : meta) {
    std::vector<int> want(static_cast<std::size_t>(m.n));
    cpu::dpcm_difference_block(&host_in[static_cast<std::size_t>(m.offset)],
                               m.w, m.h, m.vertical != 0, want.data());
    ++g_cases;
    check(std::memcmp(want.data(),
                      &host_out[static_cast<std::size_t>(m.offset)],
                      sizeof(int) * static_cast<std::size_t>(m.n)) == 0,
          "gpu dpcm difference", m.w, m.h, m.vertical);
  }

  status = li::dpcm_cumsum(fixture.workspace, cum_jobs.data(), job);
  check(status == Status::kOk, "dpcm_cumsum");
  copy_down(fixture.data.block_a, host_out.data(),
            static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
            "dpcm cumsum download");
  if (!sync_stream("dpcm cumsum sync")) return;
  for (const Meta& m : meta) {
    ++g_cases;
    check(std::memcmp(&host_in[static_cast<std::size_t>(m.offset)],
                      &host_out[static_cast<std::size_t>(m.offset)],
                      sizeof(int) * static_cast<std::size_t>(m.n)) == 0,
          "gpu dpcm round trip", m.w, m.h, m.vertical);
  }

  // Forward with DPCM: all square and rectangular sizes, both directions and
  // both transforms (the 4:2:0/4:2:2/4:4:4 chroma and edge-block extents).
  meta.clear();
  forward_jobs.clear();
  job = 0;
  const int dpcm_dims[][2] = {{4, 4},   {8, 8},   {16, 16}, {32, 32},
                              {4, 8},   {8, 4},   {4, 16},  {16, 4},
                              {8, 16},  {16, 8},  {4, 32},  {32, 4},
                              {16, 32}, {32, 16}};
  for (const auto& dim : dpcm_dims) {
    const int w = dim[0];
    const int h = dim[1];
    for (int dct = 0; dct <= 1; ++dct) {
      for (int vertical = 0; vertical <= 1; ++vertical) {
        const int n = w * h;
        const int offset = job * kMaxBlockArea;
        for (int i = 0; i < n; ++i) {
          host_in[static_cast<std::size_t>(offset) + i] =
              random_range(-255, 255);
        }
        li::ForwardJob forward;
        forward.residual = fixture.data.block_a + offset;
        forward.levels = fixture.data.block_b + offset;
        forward.w = w;
        forward.h = h;
        forward.dct_dct = dct;
        forward.dpcm = 1;
        forward.vertical = vertical;
        forward_jobs.push_back(forward);
        meta.push_back({w, h, n, offset, vertical, dct, true});
        ++job;
      }
    }
  }
  check(job <= kMaxJobs, "forward dpcm batch fits");
  copy_up(host_in.data(), fixture.data.block_a,
          static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
          "forward dpcm upload");
  status =
      li::forward_blocks(fixture.workspace, forward_jobs.data(), job);
  check(status == Status::kOk, "forward_blocks dpcm");
  copy_down(fixture.data.block_b, host_out.data(),
            static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
            "forward dpcm download");
  if (!sync_stream("forward dpcm sync")) return;
  for (const Meta& m : meta) {
    std::vector<int> diff(static_cast<std::size_t>(m.n));
    std::vector<int> want(static_cast<std::size_t>(m.n));
    cpu::dpcm_difference_block(&host_in[static_cast<std::size_t>(m.offset)],
                               m.w, m.h, m.vertical != 0, diff.data());
    cpu::forward_lossless_block(diff.data(), m.w, m.h, m.dct != 0,
                                want.data());
    ++g_cases;
    check(std::memcmp(want.data(),
                      &host_out[static_cast<std::size_t>(m.offset)],
                      sizeof(int) * static_cast<std::size_t>(m.n)) == 0,
          "gpu forward with dpcm", m.w, m.h, m.dct, m.vertical);
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// Reconstruction (strided, plane edges) + segment-QP reconstruction
// ---------------------------------------------------------------------------

void test_reconstruction(Fixture& fixture) {
  std::printf("-- reconstruction (strided, edge blocks, segment QP)\n");
  const int checks_before = g_failures;
  const int dims[4] = {4, 8, 16, 32};
  struct Meta {
    int w;
    int h;
    int n;
    int offset;
    int dct;
    int dpcm;
    int vertical;
    int x;
    int y;
  };
  std::vector<Meta> meta;
  std::vector<li::ReconJob> jobs;
  std::vector<int> host_levels(static_cast<std::size_t>(kMaxJobs) *
                               kMaxBlockArea);
  std::vector<int> host_source(static_cast<std::size_t>(kMaxJobs) *
                               kMaxBlockArea);
  std::vector<std::uint8_t> host_pred(static_cast<std::size_t>(kMaxJobs) *
                                      kPlaneBytes);
  std::vector<std::uint8_t> host_recon(static_cast<std::size_t>(kMaxJobs) *
                                       kPlaneBytes);
  for (std::uint8_t& value : host_pred) {
    value = static_cast<std::uint8_t>(random_range(0, 255));
  }
  std::fill(host_recon.begin(), host_recon.end(),
            static_cast<std::uint8_t>(0xEE));

  int job = 0;
  for (int w : dims) {
    for (int h : dims) {
      for (int dct = 0; dct <= 1; ++dct) {
        for (int mode = 0; mode < 3; ++mode) {
          const int dpcm = (mode == 0) ? 0 : 1;
          const int vertical = (mode == 2) ? 1 : 0;
          const int n = w * h;
          const int offset = job * kMaxBlockArea;
          for (int i = 0; i < n; ++i) {
            const int value = random_range(-255, 255);
            host_source[static_cast<std::size_t>(offset) + i] = value;
          }
          std::vector<int> diff(static_cast<std::size_t>(n));
          if (dpcm != 0) {
            cpu::dpcm_difference_block(
                &host_source[static_cast<std::size_t>(offset)], w, h,
                vertical != 0, diff.data());
          } else {
            std::memcpy(diff.data(),
                        &host_source[static_cast<std::size_t>(offset)],
                        sizeof(int) * static_cast<std::size_t>(n));
          }
          cpu::forward_lossless_block(diff.data(), w, h, dct != 0,
                                      &host_levels[static_cast<std::size_t>(
                                          offset)]);
          // Interior block, then the same block flush against the plane edge.
          const int xs[2] = {2, kPlaneStride - w};
          const int ys[2] = {3, kPlaneRows - h};
          for (int variant = 0; variant < 2; ++variant) {
            const int x = xs[variant];
            const int y = ys[variant];
            li::ReconJob recon;
            recon.levels = fixture.data.block_a + offset;
            recon.pred =
                fixture.data.pred +
                static_cast<std::size_t>(job) * kPlaneBytes +
                static_cast<std::size_t>(y) * kPlaneStride + x;
            recon.recon =
                fixture.data.recon +
                static_cast<std::size_t>(job) * kPlaneBytes +
                static_cast<std::size_t>(y) * kPlaneStride + x;
            recon.pred_stride = kPlaneStride;
            recon.recon_stride = kPlaneStride;
            recon.w = w;
            recon.h = h;
            recon.dct_dct = dct;
            recon.dpcm = dpcm;
            recon.vertical = vertical;
            jobs.push_back(recon);
            meta.push_back({w, h, n, offset, dct, dpcm, vertical, x, y});
            ++job;
          }
        }
      }
    }
  }
  check(job <= kMaxJobs, "recon batch fits");

  copy_up(host_levels.data(), fixture.data.block_a,
          static_cast<std::size_t>(job) * kMaxBlockArea * sizeof(int),
          "levels upload");
  copy_up(host_pred.data(), fixture.data.pred,
          static_cast<std::size_t>(job) * kPlaneBytes, "pred upload");
  copy_up(host_recon.data(), fixture.data.recon,
          static_cast<std::size_t>(job) * kPlaneBytes, "recon upload");

  Status status =
      li::reconstruct_blocks(fixture.workspace, jobs.data(), job);
  check(status == Status::kOk, "reconstruct_blocks");
  copy_down(fixture.data.recon, host_recon.data(),
            static_cast<std::size_t>(job) * kPlaneBytes, "recon download");
  if (!sync_stream("recon sync")) return;

  for (std::size_t index = 0; index < meta.size(); ++index) {
    const Meta& m = meta[index];
    std::vector<int> inv(static_cast<std::size_t>(m.n));
    cpu::inverse_lossless_block(
        &host_levels[static_cast<std::size_t>(m.offset)], m.w, m.h,
        m.dct != 0, inv.data());
    std::vector<int> reference(static_cast<std::size_t>(m.n));
    if (m.dpcm != 0) {
      cpu::dpcm_cumsum_block(inv.data(), m.w, m.h, m.vertical != 0,
                             reference.data());
    } else {
      reference = inv;
    }
    bool ok = true;
    for (int row = 0; row < m.h && ok; ++row) {
      for (int col = 0; col < m.w; ++col) {
        const int p =
            host_pred[static_cast<std::size_t>(index) * kPlaneBytes +
                      static_cast<std::size_t>(m.y + row) * kPlaneStride +
                      m.x + col];
        int value = p + reference[static_cast<std::size_t>(row) * m.w + col];
        value = value < 0 ? 0 : (value > 255 ? 255 : value);
        const std::uint8_t got =
            host_recon[static_cast<std::size_t>(index) * kPlaneBytes +
                       static_cast<std::size_t>(m.y + row) * kPlaneStride +
                       m.x + col];
        if (got != static_cast<std::uint8_t>(value)) {
          ok = false;
          check(false, "gpu reconstruction sample", m.w, m.h, m.dct, m.dpcm);
          break;
        }
      }
    }
    ++g_cases;
    check(ok, "gpu reconstruction block", m.w, m.h, m.dct, m.dpcm);

    // No writes outside the block.
    bool outside_ok = true;
    for (int row = 0; row < kPlaneRows && outside_ok; ++row) {
      for (int col = 0; col < kPlaneStride; ++col) {
        const bool in_block = row >= m.y && row < m.y + m.h &&
                              col >= m.x && col < m.x + m.w;
        if (in_block) continue;
        const std::uint8_t got =
            host_recon[static_cast<std::size_t>(index) * kPlaneBytes +
                       static_cast<std::size_t>(row) * kPlaneStride + col];
        if (got != 0xEE) {
          outside_ok = false;
          break;
        }
      }
    }
    ++g_cases;
    check(outside_ok, "gpu reconstruction wrote outside block", m.w, m.h);
  }

  // --- Segment-QP reconstruction -----------------------------------------
  // Three frames: all lossless, mixed (seg 1 has ALT_Q +1), non-lossless.
  li::FrameQuantInputs quant[3];
  quant[0].base_q_idx = 0;
  quant[0].num_segments = 2;
  quant[0].alt_q_enabled[0] = true;
  quant[0].alt_q[0] = 0;
  quant[0].alt_q_enabled[1] = true;
  quant[0].alt_q[1] = 0;

  quant[1] = quant[0];
  quant[1].alt_q[1] = 1;

  quant[2].base_q_idx = 10;
  quant[2].num_segments = 1;

  std::vector<li::FrameLossless> host_lossless(3);
  Status derive_status =
      li::derive_lossless_batch(fixture.workspace, quant, 3,
                                host_lossless.data());
  check(derive_status == Status::kOk, "derive_lossless_batch");
  if (!sync_stream("derive sync")) return;
  for (int frame = 0; frame < 3; ++frame) {
    av2enc::lossless::inter::FrameQuantInputs cpu_quant;
    cpu_quant.base_q_idx = quant[frame].base_q_idx;
    cpu_quant.num_segments = quant[frame].num_segments;
    for (int s = 0; s < li::kMaxSegments; ++s) {
      cpu_quant.alt_q_enabled[s] = quant[frame].alt_q_enabled[s];
      cpu_quant.alt_q[s] = quant[frame].alt_q[s];
    }
    const cpu::FrameLossless want = cpu::derive_lossless(cpu_quant, true);
    ++g_cases;
    bool ok = host_lossless[static_cast<std::size_t>(frame)].coded_lossless ==
                  want.coded_lossless &&
              host_lossless[static_cast<std::size_t>(frame)]
                      .has_lossless_segment == want.has_lossless_segment &&
              host_lossless[static_cast<std::size_t>(frame)].num_segments ==
                  want.num_segments;
    for (int s = 0; ok && s < li::kMaxSegments; ++s) {
      ok = host_lossless[static_cast<std::size_t>(frame)].lossless[s] ==
               want.lossless[s] &&
           host_lossless[static_cast<std::size_t>(frame)].qindex[s] ==
               want.qindex[s];
    }
    check(ok, "gpu derive_lossless", frame);
  }

  // 8x8 DCT_DCT blocks; frame/segment pairs chosen to cover skip + recon.
  const int pairs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {1, 7}};
  constexpr int kSegJobs = 6;
  std::vector<li::ReconJob> seg_jobs(static_cast<std::size_t>(kSegJobs));
  std::vector<int> seg_levels(static_cast<std::size_t>(kSegJobs) * 64);
  for (int i = 0; i < kSegJobs; ++i) {
    for (int k = 0; k < 64; ++k) {
      seg_levels[static_cast<std::size_t>(i) * 64 + k] =
          random_range(-64, 64);
    }
    li::ReconJob recon;
    recon.levels = fixture.data.block_a + i * 64;
    recon.pred = fixture.data.pred +
                 static_cast<std::size_t>(i) * kPlaneBytes;
    recon.recon = fixture.data.recon +
                  static_cast<std::size_t>(i) * kPlaneBytes;
    recon.pred_stride = kPlaneStride;
    recon.recon_stride = kPlaneStride;
    recon.w = 8;
    recon.h = 8;
    recon.dct_dct = 1;
    recon.frame_index = pairs[i][0];
    recon.segment_id = pairs[i][1];
    seg_jobs[static_cast<std::size_t>(i)] = recon;
  }
  std::vector<std::uint8_t> seg_pred(
      static_cast<std::size_t>(kSegJobs) * kPlaneBytes);
  std::vector<std::uint8_t> seg_recon(
      static_cast<std::size_t>(kSegJobs) * kPlaneBytes);
  for (std::uint8_t& value : seg_pred) {
    value = static_cast<std::uint8_t>(random_range(0, 255));
  }
  std::fill(seg_recon.begin(), seg_recon.end(),
            static_cast<std::uint8_t>(0xEE));
  copy_up(seg_levels.data(), fixture.data.block_a,
          seg_levels.size() * sizeof(int), "seg levels upload");
  copy_up(seg_pred.data(), fixture.data.pred, seg_pred.size(),
          "seg pred upload");
  copy_up(seg_recon.data(), fixture.data.recon, seg_recon.size(),
          "seg recon upload");

  // segment_qp == nullptr selects the workspace's derived table.
  status = li::reconstruct_blocks_segmented(fixture.workspace, seg_jobs.data(),
                                            kSegJobs, nullptr);
  check(status == Status::kOk, "reconstruct_blocks_segmented");
  copy_down(fixture.data.recon, seg_recon.data(), seg_recon.size(),
            "seg recon download");
  std::vector<std::uint8_t> seg_status(kSegJobs);
  copy_down(fixture.workspace.recon_status, seg_status.data(), kSegJobs,
            "seg status download");
  if (!sync_stream("seg sync")) return;

  for (int i = 0; i < kSegJobs; ++i) {
    const int frame = pairs[i][0];
    const int segment = pairs[i][1];
    const int clamped_segment =
        segment < 0 ? 0 : (segment >= li::kMaxSegments ? li::kMaxSegments - 1
                                                       : segment);
    const bool lossless =
        host_lossless[static_cast<std::size_t>(frame)]
            .lossless[clamped_segment];
    ++g_cases;
    check(seg_status[static_cast<std::size_t>(i)] == (lossless ? 1 : 0),
          "segment reconstruction status", i, lossless ? 1 : 0,
          seg_status[static_cast<std::size_t>(i)]);
    if (!lossless) {
      bool untouched = true;
      for (int k = 0; k < 64; ++k) {
        if (seg_recon[static_cast<std::size_t>(i) * kPlaneBytes + k] != 0xEE) {
          untouched = false;
          break;
        }
      }
      ++g_cases;
      check(untouched, "non-lossless segment untouched", i);
      continue;
    }
    // Reconstruct the lossless rows against the CPU and compare.
    std::vector<int> inv(64);
    cpu::inverse_lossless_block(
        &seg_levels[static_cast<std::size_t>(i) * 64], 8, 8, true, inv.data());
    bool ok = true;
    for (int row = 0; row < 8 && ok; ++row) {
      for (int col = 0; col < 8; ++col) {
        const int p = seg_pred[static_cast<std::size_t>(i) * kPlaneBytes +
                               static_cast<std::size_t>(row) * kPlaneStride +
                               col];
        int value = p + inv[static_cast<std::size_t>(row) * 8 + col];
        value = value < 0 ? 0 : (value > 255 ? 255 : value);
        const std::uint8_t got =
            seg_recon[static_cast<std::size_t>(i) * kPlaneBytes +
                      static_cast<std::size_t>(row) * kPlaneStride + col];
        if (got != static_cast<std::uint8_t>(value)) {
          ok = false;
          break;
        }
      }
    }
    ++g_cases;
    check(ok, "segment lossless reconstruction", i);
  }

  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// Transform-size selection and arena accounting
// ---------------------------------------------------------------------------

void test_tx_selection(Fixture& fixture) {
  std::printf("-- §5.20.6.1 transform-size selection\n");
  const int checks_before = g_failures;
  const int dims[] = {4, 8, 12, 16, 24, 32, 64};
  std::vector<li::TxSizeRequest> requests;
  for (int bw : dims) {
    for (int bh : dims) {
      for (int is_inter = 0; is_inter <= 1; ++is_inter) {
        for (int fsc = 0; fsc <= 1; ++fsc) {
          for (int allow = 0; allow <= 1; ++allow) {
            for (int want = 0; want <= 1; ++want) {
              li::TxSizeRequest request;
              request.block_w = bw;
              request.block_h = bh;
              request.is_inter = is_inter;
              request.fsc_mode = fsc;
              request.allow_select = allow;
              request.want_large = want;
              requests.push_back(request);
            }
          }
        }
      }
    }
  }
  for (std::size_t start = 0; start < requests.size();
       start += kMaxJobs) {
    const int count = static_cast<int>(
        std::min<std::size_t>(kMaxJobs, requests.size() - start));
    std::vector<li::TxSizeDecision> decisions(static_cast<std::size_t>(count));
    Status status = li::select_tx_size_batch(
        fixture.workspace, &requests[start], count, decisions.data());
    check(status == Status::kOk, "select_tx_size_batch");
    if (!sync_stream("tx select sync")) return;
    for (int i = 0; i < count; ++i) {
      const li::TxSizeRequest& request =
          requests[start + static_cast<std::size_t>(i)];
      const cpu::TxSizeDecision want = cpu::select_tx_size(
          request.block_w, request.block_h, request.is_inter != 0,
          request.fsc_mode != 0, request.allow_select != 0,
          request.want_large != 0);
      const li::TxSizeDecision& got = decisions[static_cast<std::size_t>(i)];
      ++g_cases;
      check(got.coded_w == want.coded_w && got.coded_h == want.coded_h &&
                got.lossless_tx_size == want.lossless_tx_size &&
                got.signal_lossless_tx_size == want.signal_lossless_tx_size,
            "gpu select_tx_size", request.block_w, request.block_h,
            request.is_inter, request.want_large);
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

void test_arena_accounting(Fixture& fixture) {
  std::printf("-- arena accounting (persistent workspace, mark/rewind)\n");
  const int checks_before = g_failures;

  // Two launches on the same workspace must not grow the arena at all.
  const std::uint64_t used_before = fixture.arena.used();
  const std::uint64_t peak_before = fixture.arena.peak_used();
  std::vector<li::TxSizeRequest> requests(8);
  std::vector<li::TxSizeDecision> decisions(8);
  for (int round = 0; round < 2; ++round) {
    Status status = li::select_tx_size_batch(fixture.workspace,
                                             requests.data(), 8,
                                             decisions.data());
    check(status == Status::kOk, "accounting launch");
    if (!sync_stream("accounting sync")) return;
  }
  check(fixture.arena.used() == used_before, "no net arena growth",
        0, static_cast<int>(fixture.arena.used()),
        static_cast<int>(used_before));
  check(fixture.arena.peak_used() == peak_before, "peak unchanged");

  // mark()/rewind() scratch is net-zero.
  const std::uint64_t mark = fixture.arena.mark();
  int* scratch = fixture.arena.allocate_array<int>(4096);
  check(scratch != nullptr, "scratch allocation");
  check(fixture.arena.used() > used_before, "scratch raised the watermark");
  Status status = li::select_tx_size_batch(fixture.workspace, requests.data(),
                                           8, decisions.data());
  check(status == Status::kOk, "scratch launch");
  if (!sync_stream("scratch sync")) return;
  check(fixture.arena.rewind(mark), "rewind scratch");
  check(fixture.arena.used() == used_before, "rewind is net-zero",
        0, static_cast<int>(fixture.arena.used()),
        static_cast<int>(used_before));

  // One byte short: distinct status and required size.
  const li::LaunchConfig cfg = [] {
    li::LaunchConfig c;
    c.max_jobs = kMaxJobs;
    c.max_block_w = kMaxW;
    c.max_block_h = kMaxH;
    return c;
  }();
  const std::uint64_t required = li::required_device_bytes(cfg);
  check(required > 1, "required bytes is positive");
  Arena small;
  std::string error;
  Status init_status = small.init(required - 1, Arena::kAlignment, &error);
  check(init_status == Status::kOk, "small arena init");
  if (init_status == Status::kOk) {
    li::Workspace too_small;
    std::uint64_t reported = 0;
    const Status status =
        too_small.init(small, cfg, &reported, &error);
    ++g_cases;
    check(status == Status::kArenaExhausted,
          "undersized arena returns kArenaExhausted",
          static_cast<int>(status));
    check(reported == required, "undersized arena reports required size",
          0, static_cast<int>(reported), static_cast<int>(required));
    check(!too_small.valid(), "undersized workspace invalid");
    std::printf("   short arena: %s\n", av2enc::cuda::status_name(status));
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

}  // namespace

int main() {
  std::printf("lossless_inter GPU parity test\n");

  int device = 0;
  cudaDeviceProp properties{};
  const cudaError_t device_error = cudaGetDeviceProperties(&properties, device);
  if (device_error != cudaSuccess) {
    std::printf("no CUDA device: %s\n", cudaGetErrorString(device_error));
    return 1;
  }
  std::printf("device: %s (sm_%d%d)\n", properties.name, properties.major,
              properties.minor);

  Fixture fixture;
  if (!setup_fixture(fixture)) {
    std::printf("fixture setup failed\n");
    return 1;
  }
  std::printf("workspace required: %llu bytes; pool: %llu bytes\n",
              static_cast<unsigned long long>(fixture.workspace_bytes),
              static_cast<unsigned long long>(fixture.pool_bytes));

  test_wht(fixture);
  test_blocks(fixture);
  test_dpcm(fixture);
  test_reconstruction(fixture);
  test_tx_selection(fixture);
  test_arena_accounting(fixture);

  std::printf("arena used=%llu peak=%llu capacity=%llu\n",
              static_cast<unsigned long long>(fixture.arena.used()),
              static_cast<unsigned long long>(fixture.arena.peak_used()),
              static_cast<unsigned long long>(fixture.arena.capacity()));
  std::printf("lossless_inter_test: %lld cases, %d checks, %d failures\n",
              g_cases, g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
