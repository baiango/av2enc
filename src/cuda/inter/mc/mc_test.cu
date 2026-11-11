// CPU-vs-CUDA bit-exactness tests for AV2 inter motion compensation.
//
// Build/run: ./build_and_test.sh   (Linux + CUDA; sm_86 primary)
//
// Coverage mirrors the CPU test: subpel phases and filters, integer-MV copy,
// partial blocks, guard bands, 4:2:0 chroma, and a real 640x360 frame.  The
// GPU output is compared against src/encoder/inter/mc/mc.cpp.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "encoder/inter/mc/mc_test_common.h"
#include "mc.h"

namespace mc = av2enc::inter::mc;
using namespace av2enc::inter::mc::test;

namespace {

struct Test {
  int checks = 0;
  int failures = 0;
  void expect(bool cond, const std::string& what) {
    ++checks;
    if (!cond) {
      ++failures;
      if (failures <= 20) {
        std::printf("FAIL: %s\n", what.c_str());
      }
    }
  }
  void eq_int(long long got, long long want, const std::string& what) {
    ++checks;
    if (got != want) {
      ++failures;
      if (failures <= 20) {
        std::printf("FAIL: %s (got %lld, want %lld)\n", what.c_str(), got,
                    want);
      }
    }
  }
};

struct DevBuf {
  std::uint8_t* ptr = nullptr;
  std::size_t bytes = 0;
  DevBuf() = default;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
  bool alloc(std::size_t n) {
    bytes = n;
    return cudaMalloc(reinterpret_cast<void**>(&ptr), n) == cudaSuccess;
  }
};

struct DevFrame {
  av2enc::cuda::McDeviceFrame frame;
  DevBuf plane[3];

  bool upload(const TestFrame& src) {
    frame.mi_cols = src.mi_cols;
    frame.mi_rows = src.mi_rows;
    for (int p = 0; p < 3; ++p) {
      const std::size_t n = src.plane[p].data.size();
      if (!plane[p].alloc(n)) {
        return false;
      }
      if (cudaMemcpy(plane[p].ptr, src.plane[p].data.data(), n,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
      }
      frame.plane[p] = plane[p].ptr;
      frame.stride[p] = src.plane[p].stride;
    }
    return true;
  }
};

bool run_gpu_case(const DevFrame& df, int plane, const mc::BlockRect& rect,
                  mc::Mv mv, mc::InterpFilter filter, DestBuf* out,
                  cudaError_t* err_out) {
  DevBuf dst;
  const std::size_t n = static_cast<std::size_t>(out->stride) *
                        static_cast<std::size_t>(out->h + kGuardRows);
  if (!dst.alloc(n)) {
    return false;
  }
  std::vector<std::uint8_t> guard(n, kGuard);
  if (cudaMemcpy(dst.ptr, guard.data(), n, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    return false;
  }
  const cudaError_t err = av2enc::cuda::mc_predict_plane(
      df.frame, plane, rect, mv, filter, dst.ptr, out->stride);
  if (err_out != nullptr) {
    *err_out = err;
  }
  if (err != cudaSuccess) {
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return false;
  }
  if (cudaMemcpy(out->data.data(), dst.ptr, n, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }
  return true;
}

bool blocks_equal(const DestBuf& a, const DestBuf& b, int w, int h) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (a.at(y, x) != b.at(y, x)) {
        return false;
      }
    }
  }
  return true;
}

void test_tables(Test& t) {
  for (int f = 0; f < mc::kSubpelFilterTypes; ++f) {
    for (int p = 0; p < mc::kSubpelPhases; ++p) {
      int sum = 0;
      for (int k = 0; k < mc::kSubpelTaps; ++k) {
        sum += mc::kSubpelFilters[f][p][k];
      }
      t.eq_int(sum, 128, "device filter phase sum");
    }
  }
}

void test_plane_cases(Test& t) {
  TestFrame frame = TestFrame::make(24, 20);  // 96x80 luma
  frame.fill(Pattern::kNoise);
  DevFrame dev;
  t.expect(dev.upload(frame), "upload test frame");

  const mc::BlockRect rects[] = {
      {0, 0, 4, 4},   {4, 0, 4, 8},    {12, 8, 8, 4},   {8, 4, 8, 8},
      {0, 0, 16, 16}, {16, 8, 16, 32}, {8, 16, 32, 16}, {0, 0, 32, 32},
      {8, 0, 64, 16}, {0, 0, 16, 64},  {0, 0, 64, 64}};
  const mc::Mv mvs[] = {{0, 0},    {1, 0},   {0, 1},  {3, -5}, {7, 7},
                        {-8, 13}, {-1, -1}, {15, 9}, {-15, -9}};
  const mc::InterpFilter filters[] = {
      mc::InterpFilter::kEighttap,      mc::InterpFilter::kEighttapSmooth,
      mc::InterpFilter::kEighttapSharp, mc::InterpFilter::kBilinear};

  for (const mc::BlockRect& rect : rects) {
    if (rect.x + rect.w > 96 || rect.y + rect.h > 80) {
      continue;
    }
    for (mc::InterpFilter filter : filters) {
      for (mc::Mv mv : mvs) {
        DestBuf cpu;
        DestBuf gpu;
        cpu.init(rect.w, rect.h);
        gpu.init(rect.w, rect.h);
        mc::predict_plane(frame.ref(), 0, rect, mv, filter, cpu.ptr(),
                          cpu.stride);
        cudaError_t err = cudaSuccess;
        const bool ok =
            run_gpu_case(dev, 0, rect, mv, filter, &gpu, &err);
        if (!ok) {
          t.expect(false, std::string("gpu prediction failed: ") +
                              cudaGetErrorString(err));
          continue;
        }
        t.expect(blocks_equal(cpu, gpu, rect.w, rect.h),
                 "CPU == GPU block " + std::to_string(rect.w) + "x" +
                     std::to_string(rect.h) + " mv(" +
                     std::to_string(mv.y) + "," + std::to_string(mv.x) + ")");
        t.expect(gpu.guards_intact(), "GPU dst guards intact");
      }
    }
  }

  // Full 17-bit MV range incl. the 64-bit §7.13.3.17 derivation.
  const mc::BlockRect large_rect{0, 0, 16, 16};
  DestBuf cpu;
  DestBuf gpu;
  cpu.init(large_rect.w, large_rect.h);
  gpu.init(large_rect.w, large_rect.h);
  mc::predict_plane(frame.ref(), 0, large_rect, mc::Mv{65535, -65535},
                    mc::InterpFilter::kEighttap, cpu.ptr(), cpu.stride);
  t.expect(run_gpu_case(dev, 0, large_rect, mc::Mv{65535, -65535},
                        mc::InterpFilter::kEighttap, &gpu, nullptr),
           "large 17-bit MV gpu run");
  t.expect(blocks_equal(cpu, gpu, large_rect.w, large_rect.h),
           "large 17-bit MV CPU == GPU");
}

void test_integer_copy(Test& t) {
  TestFrame frame = TestFrame::make(8, 6);
  frame.fill(Pattern::kSoftGradient);
  DevFrame dev;
  t.expect(dev.upload(frame), "upload integer-copy frame");
  const mc::Mv mvs[] = {{0, 0}, {24, -16}, {-40, 32}};
  for (mc::Mv mv : mvs) {
    const mc::BlockRect rect{8, 8, 16, 8};
    DestBuf cpu;
    DestBuf gpu;
    cpu.init(rect.w, rect.h);
    gpu.init(rect.w, rect.h);
    mc::predict_plane(frame.ref(), 0, rect, mv,
                      mc::InterpFilter::kEighttap, cpu.ptr(), cpu.stride);
    t.expect(run_gpu_case(dev, 0, rect, mv, mc::InterpFilter::kEighttap, &gpu,
                          nullptr),
             "integer copy gpu run");
    t.expect(blocks_equal(cpu, gpu, rect.w, rect.h),
             "integer copy CPU == GPU");
  }
}

void test_partial_blocks(Test& t) {
  TestFrame frame = TestFrame::make(8, 6);  // 32x24
  frame.fill(Pattern::kChecker);
  DevFrame dev;
  t.expect(dev.upload(frame), "upload partial frame");
  const mc::BlockRect rects[] = {{28, 0, 4, 8},  {0, 20, 8, 4},
                                 {28, 20, 4, 4}, {24, 16, 8, 8},
                                 {16, 16, 16, 8}};
  const mc::Mv mvs[] = {{0, 0}, {40, 40}, {-40, -40}, {33, -37}};
  for (const mc::BlockRect& rect : rects) {
    for (mc::Mv mv : mvs) {
      DestBuf cpu;
      DestBuf gpu;
      cpu.init(rect.w, rect.h);
      gpu.init(rect.w, rect.h);
      mc::predict_plane(frame.ref(), 0, rect, mv,
                        mc::InterpFilter::kEighttapSharp, cpu.ptr(),
                        cpu.stride);
      t.expect(run_gpu_case(dev, 0, rect, mv,
                            mc::InterpFilter::kEighttapSharp, &gpu, nullptr),
               "partial gpu run");
      t.expect(blocks_equal(cpu, gpu, rect.w, rect.h),
               "partial CPU == GPU");
      t.expect(gpu.guards_intact(), "partial GPU guards");
    }
  }
}

void test_chroma(Test& t) {
  TestFrame frame = TestFrame::make(16, 12);
  frame.fill(Pattern::kRampX);
  DevFrame dev;
  t.expect(dev.upload(frame), "upload chroma frame");
  const mc::BlockRect rects[] = {{0, 0, 8, 8}, {8, 8, 4, 4}, {48, 32, 16, 16}};
  const mc::Mv mvs[] = {{0, 0}, {3, 5}, {-7, 11}};
  for (const mc::BlockRect& rect : rects) {
    for (mc::Mv mv : mvs) {
      const mc::BlockRect cr{rect.x >> 1, rect.y >> 1, rect.w >> 1,
                             rect.h >> 1};
      for (int plane = 0; plane < 3; ++plane) {
        const mc::BlockRect pr = (plane == 0) ? rect : cr;
        DestBuf cpu;
        DestBuf gpu;
        cpu.init(pr.w, pr.h);
        gpu.init(pr.w, pr.h);
        mc::predict_plane(frame.ref(), plane, pr, mv,
                          mc::InterpFilter::kEighttap, cpu.ptr(),
                          cpu.stride);
        t.expect(run_gpu_case(dev, plane, pr, mv,
                              mc::InterpFilter::kEighttap, &gpu, nullptr),
                 "chroma gpu run");
        t.expect(blocks_equal(cpu, gpu, pr.w, pr.h),
                 "chroma CPU == GPU");
      }
      // The 4:2:0 wrapper itself.
      DestBuf y;
      DestBuf u;
      DestBuf v;
      y.init(rect.w, rect.h);
      u.init(cr.w, cr.h);
      v.init(cr.w, cr.h);
      DevBuf dy;
      DevBuf du;
      DevBuf dv;
      const std::size_t ny = static_cast<std::size_t>(y.stride) *
                             static_cast<std::size_t>(y.h + kGuardRows);
      const std::size_t nc = static_cast<std::size_t>(u.stride) *
                             static_cast<std::size_t>(u.h + kGuardRows);
      dy.alloc(ny);
      du.alloc(nc);
      dv.alloc(nc);
      std::vector<std::uint8_t> guard(ny > nc ? ny : nc, kGuard);
      cudaMemcpy(dy.ptr, guard.data(), ny, cudaMemcpyHostToDevice);
      cudaMemcpy(du.ptr, guard.data(), nc, cudaMemcpyHostToDevice);
      cudaMemcpy(dv.ptr, guard.data(), nc, cudaMemcpyHostToDevice);
      std::uint8_t* dst[3] = {dy.ptr, du.ptr, dv.ptr};
      int stride[3] = {y.stride, u.stride, v.stride};
      t.expect(av2enc::cuda::mc_predict_block_420(
                   dev.frame, rect, mv, mc::InterpFilter::kEighttap, dst,
                   stride) == cudaSuccess,
               "420 gpu status");
      cudaDeviceSynchronize();
      cudaMemcpy(y.data.data(), dy.ptr, ny, cudaMemcpyDeviceToHost);
      cudaMemcpy(u.data.data(), du.ptr, nc, cudaMemcpyDeviceToHost);
      cudaMemcpy(v.data.data(), dv.ptr, nc, cudaMemcpyDeviceToHost);
      t.expect(y.guards_intact() && u.guards_intact() && v.guards_intact(),
               "420 GPU guards");
      DestBuf y_cpu;
      y_cpu.init(rect.w, rect.h);
      mc::predict_plane(frame.ref(), 0, rect, mv,
                        mc::InterpFilter::kEighttap, y_cpu.ptr(),
                        y_cpu.stride);
      t.expect(blocks_equal(y_cpu, y, rect.w, rect.h),
               "420 GPU luma == CPU");
    }
  }
}

void test_validation(Test& t) {
  TestFrame frame = TestFrame::make(4, 4);
  frame.fill(Pattern::kFlat);
  DevFrame dev;
  t.expect(dev.upload(frame), "upload validation frame");
  DestBuf dst;
  dst.init(8, 8);
  t.expect(av2enc::cuda::mc_predict_plane(dev.frame, 0,
                                          mc::BlockRect{12, 0, 8, 8},
                                          mc::Mv{},
                                          mc::InterpFilter::kEighttap,
                                          dst.ptr(), dst.stride) ==
               cudaErrorInvalidValue,
           "GPU rejects out-of-plane rect");
  t.expect(av2enc::cuda::mc_predict_plane(dev.frame, 0,
                                          mc::BlockRect{0, 0, 128, 128},
                                          mc::Mv{},
                                          mc::InterpFilter::kEighttap,
                                          dst.ptr(), 256) ==
               cudaErrorInvalidValue,
           "GPU rejects 128x128 block");
  t.expect(av2enc::cuda::mc_predict_plane(dev.frame, 0,
                                          mc::BlockRect{0, 0, 8, 8},
                                          mc::Mv{},
                                          mc::InterpFilter::kEighttap,
                                          nullptr, 8) ==
               cudaErrorInvalidValue,
           "GPU rejects null dst");
  std::uint8_t* planes[3] = {dst.ptr(), dst.ptr(), dst.ptr()};
  int strides[3] = {dst.stride, dst.stride, dst.stride};
  t.expect(av2enc::cuda::mc_predict_block_420(
               dev.frame, mc::BlockRect{1, 0, 4, 4}, mc::Mv{},
               mc::InterpFilter::kEighttap, planes, strides) ==
               cudaErrorInvalidValue,
           "GPU rejects odd 420 geometry");
  t.expect(av2enc::cuda::mc_predict_plane(
               dev.frame, 0, mc::BlockRect{0, 0, 8, 8}, mc::Mv{},
               static_cast<mc::InterpFilter>(9), dst.ptr(), dst.stride) ==
               cudaErrorInvalidValue,
           "GPU rejects invalid filter");
}

void test_batch_mixed(Test& t) {
  TestFrame frame = TestFrame::make(16, 12);  // 64x48 luma
  frame.fill(Pattern::kNoise);
  DevFrame dev;
  t.expect(dev.upload(frame), "upload batch frame");

  struct JobSpec {
    int plane;
    mc::BlockRect rect;
    mc::Mv mv;
    mc::InterpFilter filter;
  };
  const JobSpec specs[] = {
      {0, {0, 0, 8, 8}, mc::Mv{1, 2}, mc::InterpFilter::kEighttap},
      {0, {8, 8, 4, 4}, mc::Mv{-3, 5}, mc::InterpFilter::kEighttapSmooth},
      {0, {56, 40, 8, 8}, mc::Mv{13, -7}, mc::InterpFilter::kEighttapSharp},
      {0, {16, 16, 32, 16}, mc::Mv{7, 7}, mc::InterpFilter::kBilinear},
      {1, {0, 0, 16, 12}, mc::Mv{-1, 1}, mc::InterpFilter::kEighttap},
      {2, {8, 4, 8, 8}, mc::Mv{5, -5}, mc::InterpFilter::kEighttapSmooth},
  };
  constexpr int kCount = static_cast<int>(sizeof(specs) / sizeof(specs[0]));

  std::vector<DestBuf> dsts(kCount);
  std::vector<av2enc::cuda::McBlockJob> jobs(kCount);
  std::size_t total = 0;
  for (int i = 0; i < kCount; ++i) {
    dsts[i].init(specs[i].rect.w, specs[i].rect.h);
    total += static_cast<std::size_t>(dsts[i].stride) *
             static_cast<std::size_t>(dsts[i].h + kGuardRows);
    jobs[i].x = specs[i].rect.x;
    jobs[i].y = specs[i].rect.y;
    jobs[i].w = specs[i].rect.w;
    jobs[i].h = specs[i].rect.h;
    jobs[i].plane = specs[i].plane;
    jobs[i].mv = specs[i].mv;
    jobs[i].filter = specs[i].filter;
    jobs[i].dst_stride = dsts[i].stride;
  }
  DevBuf dev_dst;
  DevBuf dev_jobs;
  t.expect(dev_dst.alloc(total), "batch dst alloc");
  t.expect(dev_jobs.alloc(jobs.size() * sizeof(jobs[0])), "batch jobs alloc");
  std::vector<std::uint8_t> guard(total, kGuard);
  cudaMemcpy(dev_dst.ptr, guard.data(), total, cudaMemcpyHostToDevice);
  std::size_t offset = 0;
  for (int i = 0; i < kCount; ++i) {
    jobs[i].dst = dev_dst.ptr + offset;
    offset += static_cast<std::size_t>(dsts[i].stride) *
              static_cast<std::size_t>(dsts[i].h + kGuardRows);
  }
  cudaMemcpy(dev_jobs.ptr, jobs.data(), jobs.size() * sizeof(jobs[0]),
             cudaMemcpyHostToDevice);
  const auto* job_dev =
      reinterpret_cast<const av2enc::cuda::McBlockJob*>(dev_jobs.ptr);
  const cudaError_t err = av2enc::cuda::mc_predict_batch(dev.frame, job_dev,
                                                        kCount);
  t.expect(err == cudaSuccess, std::string("mixed batch launch: ") +
                                   cudaGetErrorString(err));
  t.expect(cudaDeviceSynchronize() == cudaSuccess, "mixed batch sync");

  offset = 0;
  for (int i = 0; i < kCount; ++i) {
    const std::size_t n = static_cast<std::size_t>(dsts[i].stride) *
                          static_cast<std::size_t>(dsts[i].h + kGuardRows);
    cudaMemcpy(dsts[i].data.data(), dev_dst.ptr + offset, n,
               cudaMemcpyDeviceToHost);
    offset += n;
    DestBuf cpu;
    cpu.init(specs[i].rect.w, specs[i].rect.h);
    mc::predict_plane(frame.ref(), specs[i].plane, specs[i].rect,
                      specs[i].mv, specs[i].filter, cpu.ptr(), cpu.stride);
    t.expect(blocks_equal(cpu, dsts[i], specs[i].rect.w, specs[i].rect.h),
             "mixed batch CPU == GPU job " + std::to_string(i));
    t.expect(dsts[i].guards_intact(), "mixed batch guards job " +
                                          std::to_string(i));
  }
}

void test_real_frame(Test& t, const std::string& repo_root) {
  const std::string path = repo_root + "/tests/data/m0_640x360_1f.y4m";
  std::vector<Y4mFrame> frames;
  if (!load_y4m(path, 1, &frames) || frames.empty()) {
    std::printf("note: %s unavailable, skipping real-frame GPU test\n",
                path.c_str());
    return;
  }
  TestFrame ref = frame_from_y4m(frames[0]);
  DevFrame dev;
  t.expect(dev.upload(ref), "upload real frame");

  const int dy = 2;
  const int dx = -4;
  TestFrame cur = TestFrame::make(ref.mi_cols, ref.mi_rows);
  for (int y = 0; y < cur.plane[0].height; ++y) {
    const int sy = std::min(std::max(y + dy, 0), ref.plane[0].height - 1);
    for (int x = 0; x < cur.plane[0].width; ++x) {
      const int sx = std::min(std::max(x + dx, 0), ref.plane[0].width - 1);
      cur.plane[0].at(y, x) = ref.plane[0].at(sy, sx);
    }
  }
  const mc::Mv mv{dy * 8, dx * 8};

  // Build one job per 32x32 luma block, upload all jobs, one launch.
  std::vector<av2enc::cuda::McBlockJob> jobs;
  std::vector<DestBuf> dsts;
  int mismatches = 0;
  int blocks = 0;
  for (int by = 0; by < ref.plane[0].height; by += 32) {
    for (int bx = 0; bx < ref.plane[0].width; bx += 32) {
      const mc::BlockRect rect{bx, by, std::min(32, ref.plane[0].width - bx),
                               std::min(32, ref.plane[0].height - by)};
      DestBuf d;
      d.init(rect.w, rect.h);
      av2enc::cuda::McBlockJob job;
      job.x = rect.x;
      job.y = rect.y;
      job.w = rect.w;
      job.h = rect.h;
      job.plane = 0;
      job.mv = mv;
      job.filter = mc::InterpFilter::kEighttap;
      job.dst = nullptr;  // filled in after the destination allocation
      job.dst_stride = d.stride;
      jobs.push_back(job);
      dsts.push_back(std::move(d));
      ++blocks;
    }
  }
  DevBuf dev_jobs;
  DevBuf dev_dst;
  std::vector<std::uint8_t> host_jobs(jobs.size() * sizeof(jobs[0]));
  std::memcpy(host_jobs.data(), jobs.data(), host_jobs.size());
  dev_jobs.alloc(host_jobs.size());
  cudaMemcpy(dev_jobs.ptr, host_jobs.data(), host_jobs.size(),
             cudaMemcpyHostToDevice);

  // Put every destination in one contiguous device buffer.
  std::size_t total = 0;
  for (const DestBuf& d : dsts) {
    total += static_cast<std::size_t>(d.stride) * (d.h + kGuardRows);
  }
  dev_dst.alloc(total);
  std::vector<std::uint8_t> guard(total, kGuard);
  cudaMemcpy(dev_dst.ptr, guard.data(), total, cudaMemcpyHostToDevice);
  std::size_t offset = 0;
  auto* job_dev = reinterpret_cast<av2enc::cuda::McBlockJob*>(dev_jobs.ptr);
  std::vector<av2enc::cuda::McBlockJob> patched = jobs;
  for (std::size_t i = 0; i < patched.size(); ++i) {
    patched[i].dst = dev_dst.ptr + offset;
    offset += static_cast<std::size_t>(dsts[i].stride) * (dsts[i].h + kGuardRows);
  }
  cudaMemcpy(job_dev, patched.data(), patched.size() * sizeof(patched[0]),
             cudaMemcpyHostToDevice);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaError_t err = av2enc::cuda::mc_predict_batch(
      dev.frame, job_dev, static_cast<int>(patched.size()));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  t.expect(err == cudaSuccess, std::string("batch launch: ") +
                                   cudaGetErrorString(err));
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  // Compare every block with the CPU reference.
  offset = 0;
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    DestBuf& d = dsts[i];
    const std::size_t n =
        static_cast<std::size_t>(d.stride) * (d.h + kGuardRows);
    cudaMemcpy(d.data.data(), dev_dst.ptr + offset, n, cudaMemcpyDeviceToHost);
    offset += n;
    DestBuf cpu;
    cpu.init(d.w, d.h);
    mc::predict_plane(ref.ref(), 0, mc::BlockRect{patched[i].x, patched[i].y,
                                                  patched[i].w, patched[i].h},
                      mv, mc::InterpFilter::kEighttap, cpu.ptr(), cpu.stride);
    bool same = blocks_equal(cpu, d, d.w, d.h);
    for (int y = 0; y < d.h && same; ++y) {
      for (int x = 0; x < d.w; ++x) {
        if (d.at(y, x) != cur.plane[0].at(patched[i].y + y,
                                          patched[i].x + x)) {
          same = false;
          break;
        }
      }
    }
    if (!same) {
      ++mismatches;
    }
    if (!d.guards_intact()) {
      ++mismatches;
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  t.eq_int(mismatches, 0, "real frame batch mismatches");
  const double mp =
      static_cast<double>(ref.plane[0].width) * ref.plane[0].height / 1e6;
  std::printf(
      "real frame GPU: %d luma blocks, one batch launch, %.3f ms "
      "(%.0f MP/s luma) [kernel timing incl. launch, not a QBS claim]\n",
      blocks, static_cast<double>(ms), mp / (static_cast<double>(ms) / 1000.0));

  // One launch per block, for comparison: this is the anti-pattern the batch
  // API exists to avoid.
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (std::size_t i = 0; i < patched.size(); ++i) {
    av2enc::cuda::mc_predict_plane(
        dev.frame, 0, mc::BlockRect{patched[i].x, patched[i].y, patched[i].w,
                                    patched[i].h},
        mv, mc::InterpFilter::kEighttap, patched[i].dst, patched[i].dst_stride);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::printf("real frame GPU: %d per-block launches, %.3f ms\n", blocks,
              static_cast<double>(ms));
  t.expect(blocks == 240, "real frame block count");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string repo_root =
      (argc > 1) ? std::string(argv[1]) : std::string("../../../../..");

  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::printf("mc CUDA: no device available\n");
    return 0;
  }
  cudaSetDevice(device);
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  Test t;
  test_tables(t);
  test_plane_cases(t);
  test_integer_copy(t);
  test_partial_blocks(t);
  test_chroma(t);
  test_validation(t);
  test_batch_mixed(t);
  test_real_frame(t, repo_root);

  const cudaError_t last = cudaGetLastError();
  t.expect(last == cudaSuccess,
           std::string("no sticky CUDA error: ") + cudaGetErrorString(last));

  std::printf("mc CUDA: %d checks, %d failures\n", t.checks, t.failures);
  return t.failures == 0 ? 0 : 1;
}
