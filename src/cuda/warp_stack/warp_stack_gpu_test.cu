// CPU-vs-GPU bit-exactness tests for the CUDA warp stack.
//
// The "CPU" side is `warp_stack_host_ref.cpp`: the same shared device header
// compiled by the host compiler. There is no tolerance and no second
// transcription. For 8-bit the host side is itself checked against the
// independent CPU reference (src/encoder/inter/advanced/warp.cpp) in
// warp_stack_emulation_test.cpp; this file additionally compares the GPU
// block/extended paths with that reference for a representative matrix,
// including the invalid-shear rejection.
//
// Every device buffer in this test comes from av2enc::cuda::Arena. There is no
// cudaMalloc/cudaFree/cudaMallocHost here either.
//
// Build/run: ./build_gpu_test.sh   (needs nvcc + an sm_86 device)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "warp_stack.h"
#include "warp_stack_host_ref.h"
#include "warp_stack_test_common.h"

#include "encoder/inter/advanced/warp.h"
#include "encoder/inter/warp_stack/warp_stack.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using av2enc::cuda::status_name;
using av2enc::cuda::warpstack::required_device_bytes;
using av2enc::cuda::warpstack::WarpCornerJob;
using av2enc::cuda::warpstack::WarpCornerResult;
using av2enc::cuda::warpstack::WarpEstimateJob;
using av2enc::cuda::warpstack::WarpPredictJob;
using av2enc::cuda::warpstack::WarpPredictResult;
using av2enc::cuda::warpstack::WarpStackConfig;
using av2enc::cuda::warpstack::Workspace;
using namespace av2enc::cuda::warpstack::test;
namespace ia = av2enc::inter::adv;
namespace iw = av2enc::inter::warp;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL: %s\n", what.c_str());
    }
  }
}

void check_status(Status status, const std::string& what) {
  check(status == Status::kOk,
        what + " (" + status_name(status) + ")");
}

constexpr std::int16_t kSentinel = 0x5a5a;
constexpr std::uint64_t kSlack = 128ull * 1024ull * 1024ull;

bool upload(void* dst, const void* src, std::size_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}
bool download(void* dst, const void* src, std::size_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool region_equal(const std::int16_t* a, const std::int16_t* b, std::size_t count) {
  return std::memcmp(a, b, count * sizeof(std::int16_t)) == 0;
}

// Copies a w x h prediction with stride `src_stride` into a pitch-pitch
// sentinel buffer so whole job slices can be compared byte for byte.
void pack_region(const std::int16_t* src, int src_stride, int w, int h, int pitch,
                 std::vector<std::int16_t>* out) {
  out->assign(static_cast<std::size_t>(pitch) * pitch, kSentinel);
  for (int y = 0; y < h; ++y) {
    std::memcpy(out->data() + static_cast<std::size_t>(y) * pitch,
                src + static_cast<std::size_t>(y) * src_stride,
                static_cast<std::size_t>(w) * sizeof(std::int16_t));
  }
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

struct PredictionBatch {
  std::vector<WarpPredictJob> host_jobs;
  std::vector<WarpPredictJob> device_jobs;
  std::vector<WarpPredictResult> host_results;
  std::vector<std::int16_t> host_dst;
  std::vector<std::int16_t> expected_dst;
  std::vector<std::int16_t> initial_dst;
};

PredictionBatch build_batch(const std::vector<PredictionCase>& cases,
                            const Plane8& plane8, const Plane16& plane16,
                            std::int16_t* d_dst) {
  PredictionBatch batch;
  const std::size_t n = cases.size();
  batch.host_jobs.resize(n);
  batch.device_jobs.resize(n);
  batch.host_results.resize(n);
  batch.host_dst.assign(n * kTestDstPitch * kTestDstPitch, kSentinel);
  batch.initial_dst = batch.host_dst;

  for (std::size_t i = 0; i < n; ++i) {
    const PredictionCase& c = cases[i];
    int x = c.x;
    int y = c.y;
    int w = c.w;
    int h = c.h;
    if (c.sub) {
      x >>= 1;
      y >>= 1;
      w = std::max(1, w >> 1);
      h = std::max(1, h >> 1);
    }
    const ia::WarpBlockGeometry geom = make_geometry(
        c.plane, c.sub, c.sub, x, y, w, h, kTestPlaneSize, kTestPlaneSize, c.bit_depth);

    WarpPredictJob job{};
    job.params = c.model.params;
    job.ref_stride = kTestPlaneSize;
    job.ref_max_x = kTestPlaneSize - 1;
    job.ref_max_y = kTestPlaneSize - 1;
    job.geom = geom;
    job.compound = c.compound;
    job.path = av2enc::cuda::warpstack::kWarpPathAuto;
    job.dst = d_dst + i * kTestDstPitch * kTestDstPitch;
    job.dst_stride = kTestDstPitch;
    batch.device_jobs[i] = job;

    job.ref = (c.bit_depth == 8) ? static_cast<const void*>(plane8.data.data())
                                 : static_cast<const void*>(plane16.data.data());
    job.dst = batch.host_dst.data() + i * kTestDstPitch * kTestDstPitch;
    batch.host_jobs[i] = job;
    batch.host_jobs[i].ref = (c.bit_depth == 8)
                                 ? static_cast<const void*>(plane8.data.data())
                                 : static_cast<const void*>(plane16.data.data());
    batch.host_results[i] = WarpPredictResult{};
  }
  return batch;
}

void test_prediction(Arena& arena, Workspace& workspace) {
  const std::vector<PredictionCase> cases = prediction_cases();
  const Plane8 plane8 = make_plane8(kTestPlaneSize, kTestPlaneSize);
  const Plane16 plane16 = make_plane16(kTestPlaneSize, kTestPlaneSize, 1023);
  const int count = static_cast<int>(cases.size());

  const std::uint64_t mark = arena.mark();
  auto* d_plane8 = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(kTestPlaneSize) * kTestPlaneSize);
  auto* d_plane16 = arena.allocate_array<std::uint16_t>(
      static_cast<std::size_t>(kTestPlaneSize) * kTestPlaneSize);
  auto* d_dst = arena.allocate_array<std::int16_t>(
      static_cast<std::size_t>(count) * kTestDstPitch * kTestDstPitch);
  check(d_plane8 != nullptr && d_plane16 != nullptr && d_dst != nullptr,
        "prediction: arena scratch");
  check(upload(d_plane8, plane8.data.data(), plane8.data.size()), "prediction: upload 8-bit ref");
  check(upload(d_plane16, plane16.data.data(), plane16.data.size() * sizeof(std::uint16_t)),
        "prediction: upload 10-bit ref");

  PredictionBatch batch = build_batch(cases, plane8, plane16, d_dst);

  // Point every device job at the device ref planes.
  for (std::size_t i = 0; i < cases.size(); ++i) {
    batch.device_jobs[i].ref = (cases[i].bit_depth == 8)
                                   ? static_cast<const void*>(d_plane8)
                                   : static_cast<const void*>(d_plane16);
  }
  const std::size_t dst_elems =
      static_cast<std::size_t>(count) * kTestDstPitch * kTestDstPitch;
  check(upload(d_dst, batch.initial_dst.data(), dst_elems * sizeof(std::int16_t)),
        "prediction: init device dst");

  // Host reference (same header, host compiler).
  for (std::size_t i = 0; i < cases.size(); ++i) {
    av2enc::cuda::warpstack::hostref::predict(
        batch.host_jobs[i],
        batch.host_dst.data() + i * kTestDstPitch * kTestDstPitch, kTestDstPitch,
        &batch.host_results[i]);
  }

  const std::uint64_t before_launch = arena.used();
  check_status(workspace.submit_predict(batch.device_jobs.data(), count), "prediction: submit");
  check(arena.used() == before_launch, "prediction: launcher did not allocate");
  check(cudaDeviceSynchronize() == cudaSuccess, "prediction: sync");

  std::vector<std::int16_t> got(dst_elems, kSentinel);
  std::vector<WarpPredictResult> got_results(static_cast<std::size_t>(count));
  check(download(got.data(), d_dst, dst_elems * sizeof(std::int16_t)),
        "prediction: download dst");
  check_status(workspace.download_predict(got_results.data(), count),
               "prediction: download results");
  check(cudaDeviceSynchronize() == cudaSuccess, "prediction: download sync");

  int byte_mismatches = 0;
  int result_mismatches = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const std::size_t off = i * kTestDstPitch * kTestDstPitch;
    if (!region_equal(got.data() + off, batch.host_dst.data() + off,
                      static_cast<std::size_t>(kTestDstPitch) * kTestDstPitch)) {
      ++byte_mismatches;
      if (byte_mismatches <= 5) {
        std::printf("  byte mismatch: %s\n", case_name(cases[i]).c_str());
      }
    }
    const WarpPredictResult& a = got_results[i];
    const WarpPredictResult& b = batch.host_results[i];
    if (a.path != b.path || a.sections != b.sections || a.shear_valid != b.shear_valid ||
        a.status != b.status) {
      ++result_mismatches;
    }
  }
  check(byte_mismatches == 0,
        "prediction: GPU bytes == host reference over " + std::to_string(cases.size()) +
            " cases (" + std::to_string(byte_mismatches) + " mismatches)");
  check(result_mismatches == 0, "prediction: GPU results == host reference");

  arena.rewind(mark);
}

// Block/extended path comparison against the independent CPU reference,
// including invalid-shear rejection.
void test_paths_against_adv(Arena& arena, Workspace& workspace) {
  const Plane8 plane8 = make_plane8(kTestPlaneSize, kTestPlaneSize);
  const std::vector<ModelCase> models = make_models();
  const int sizes[] = {4, 8, 16, 20, 32, 64};

  struct Job {
    WarpPredictJob device;
    ia::WarpParams params;
    int w = 0;
    int h = 0;
    int compound = 0;
    int path = 0;
    std::string name;
  };
  std::vector<Job> jobs;
  for (const ModelCase& model : models) {
    for (int s = 0; s < 6; ++s) {
      for (int path : {1, 2}) {
        Job job;
        job.params = model.params;
        job.w = sizes[s];
        job.h = sizes[s];
        job.path = path;
        job.name = model.name + " " + std::to_string(sizes[s]) + " path " +
                   std::to_string(path);
        job.device.params = model.params;
        job.device.ref_stride = kTestPlaneSize;
        job.device.ref_max_x = kTestPlaneSize - 1;
        job.device.ref_max_y = kTestPlaneSize - 1;
        job.device.geom = make_geometry(0, 0, 0, 0, 0, sizes[s], sizes[s], kTestPlaneSize,
                                        kTestPlaneSize, 8);
        job.device.path = path;
        jobs.push_back(job);
      }
    }
  }
  const int count = static_cast<int>(jobs.size());
  const std::uint64_t mark = arena.mark();
  auto* d_ref = arena.allocate_array<std::uint8_t>(
      static_cast<std::size_t>(kTestPlaneSize) * kTestPlaneSize);
  auto* d_dst = arena.allocate_array<std::int16_t>(
      static_cast<std::size_t>(count) * kTestDstPitch * kTestDstPitch);
  check(d_ref != nullptr && d_dst != nullptr, "paths: arena scratch");
  check(upload(d_ref, plane8.data.data(), plane8.data.size()), "paths: upload ref");

  std::vector<std::int16_t> initial(
      static_cast<std::size_t>(count) * kTestDstPitch * kTestDstPitch, kSentinel);
  check(upload(d_dst, initial.data(), initial.size() * sizeof(std::int16_t)),
        "paths: init dst");

  std::vector<WarpPredictJob> device_jobs(static_cast<std::size_t>(count));
  std::vector<WarpPredictResult> host_results(static_cast<std::size_t>(count));
  std::vector<std::int16_t> host_dst(initial);
  std::vector<std::int16_t> expected(initial);
  for (int i = 0; i < count; ++i) {
    WarpPredictJob& job = device_jobs[static_cast<std::size_t>(i)];
    job = jobs[static_cast<std::size_t>(i)].device;
    job.ref = d_ref;
    job.dst = d_dst + static_cast<std::size_t>(i) * kTestDstPitch * kTestDstPitch;
    job.dst_stride = kTestDstPitch;
    WarpPredictJob host = job;
    host.ref = plane8.data.data();
    host.dst = host_dst.data() + static_cast<std::size_t>(i) * kTestDstPitch * kTestDstPitch;
    av2enc::cuda::warpstack::hostref::predict(host, host.dst, kTestDstPitch,
                                              &host_results[static_cast<std::size_t>(i)]);

    std::vector<std::int16_t> cpu(jobs[static_cast<std::size_t>(i)].w *
                                      jobs[static_cast<std::size_t>(i)].h,
                                  kSentinel);
    const ia::WarpBlockGeometry& geom = jobs[static_cast<std::size_t>(i)].device.geom;
    if (jobs[static_cast<std::size_t>(i)].path == 1) {
      check(ia::warp_predict(jobs[static_cast<std::size_t>(i)].params, plane8.view, geom,
                             false, cpu.data(), jobs[static_cast<std::size_t>(i)].w),
            "paths: adv block reference");
    } else {
      check(ia::warp_predict_ext(jobs[static_cast<std::size_t>(i)].params, plane8.view, geom,
                                 false, cpu.data(), jobs[static_cast<std::size_t>(i)].w),
            "paths: adv ext reference");
    }
    std::vector<std::int16_t> packed;
    pack_region(cpu.data(), jobs[static_cast<std::size_t>(i)].w,
                jobs[static_cast<std::size_t>(i)].w, jobs[static_cast<std::size_t>(i)].h,
                kTestDstPitch, &packed);
    std::memcpy(expected.data() + static_cast<std::size_t>(i) * kTestDstPitch *
                                     kTestDstPitch,
                packed.data(), packed.size() * sizeof(std::int16_t));
  }

  check_status(workspace.submit_predict(device_jobs.data(), count), "paths: submit");
  check(cudaDeviceSynchronize() == cudaSuccess, "paths: sync");
  std::vector<std::int16_t> got(initial);
  check(download(got.data(), d_dst, got.size() * sizeof(std::int16_t)), "paths: download");
  std::vector<WarpPredictResult> gpu_results(static_cast<std::size_t>(count));
  check_status(workspace.download_predict(gpu_results.data(), count), "paths: download results");
  check(cudaDeviceSynchronize() == cudaSuccess, "paths: download sync");
  int bytes_bad = 0;
  int host_bad = 0;
  for (int i = 0; i < count; ++i) {
    const std::size_t off =
        static_cast<std::size_t>(i) * kTestDstPitch * kTestDstPitch;
    if (!region_equal(got.data() + off, expected.data() + off,
                      static_cast<std::size_t>(kTestDstPitch) * kTestDstPitch)) {
      ++bytes_bad;
    }
    if (!region_equal(got.data() + off, host_dst.data() + off,
                      static_cast<std::size_t>(kTestDstPitch) * kTestDstPitch)) {
      ++host_bad;
    }
  }
  check(bytes_bad == 0, "paths: GPU == adv block/ext (" + std::to_string(bytes_bad) + " bad)");
  check(host_bad == 0, "paths: GPU == host reference (" + std::to_string(host_bad) + " bad)");

  // Invalid shear: the block path must write nothing and report sections == 0.
  for (int i = 0; i < count; ++i) {
    if (jobs[static_cast<std::size_t>(i)].params.p[3] == (2 << 16) &&
        jobs[static_cast<std::size_t>(i)].path == 1) {
      check(gpu_results[static_cast<std::size_t>(i)].shear_valid == 0,
            "paths: invalid shear rejected");
      check(gpu_results[static_cast<std::size_t>(i)].sections == 0,
            "paths: invalid shear wrote no section");
      check(gpu_results[static_cast<std::size_t>(i)].path == 1,
            "paths: invalid shear kept the block path");
    }
  }
  arena.rewind(mark);
}

// ---------------------------------------------------------------------------
// Corner generation / sample collection / least-squares derivation
// ---------------------------------------------------------------------------

// Returns the first differing field, with the job geometry, for diagnostics.
std::string corner_diff(const WarpCornerJob& job, const WarpCornerResult& gpu,
                        const iw::CornerResult& cpu, bool cpu_valid) {
  char buffer[512];
  auto head = [&]() {
    return "job{row=" + std::to_string(job.mi_row) + ",col=" + std::to_string(job.mi_col) +
           ",bw4=" + std::to_string(job.bw4) + ",bh4=" + std::to_string(job.bh4) +
           ",ref0=" + std::to_string(static_cast<int>(job.ref_frame0)) +
           ",iter=" + std::to_string(job.iter) +
           ",grid=" + std::to_string(job.grid.mi_cols) + "x" +
           std::to_string(job.grid.mi_rows) + "} ";
  };
  if (gpu.model_valid != (cpu_valid ? 1 : 0)) {
    return head() + "model_valid gpu=" + std::to_string(gpu.model_valid) +
           " cpu=" + std::to_string(cpu_valid ? 1 : 0);
  }
  if (gpu.corner.corners_found != cpu.corners_found) {
    return head() + "corners_found gpu=" + std::to_string(gpu.corner.corners_found) +
           " cpu=" + std::to_string(cpu.corners_found);
  }
  if (gpu.corner.all_mvs_same != cpu.all_mvs_same) {
    return head() + "all_mvs_same gpu=" + std::to_string(gpu.corner.all_mvs_same) +
           " cpu=" + std::to_string(cpu.all_mvs_same);
  }
  if (gpu.corner.negative_ref_pts != cpu.negative_ref_pts) {
    return head() + "negative_ref_pts gpu=" + std::to_string(gpu.corner.negative_ref_pts) +
           " cpu=" + std::to_string(cpu.negative_ref_pts);
  }
  for (int i = 0; i < 3; ++i) {
    for (int c = 0; c < 2; ++c) {
      if (gpu.corner.corner_pts[i][c] != cpu.corner_pts[i][c]) {
        return head() + "corner_pts[" + std::to_string(i) + "][" + std::to_string(c) +
               "] gpu=" + std::to_string(gpu.corner.corner_pts[i][c]) +
               " cpu=" + std::to_string(cpu.corner_pts[i][c]);
      }
      if (gpu.corner.corner_mvs[i][c] != cpu.corner_mvs[i][c]) {
        return head() + "corner_mvs[" + std::to_string(i) + "][" + std::to_string(c) +
               "] gpu=" + std::to_string(gpu.corner.corner_mvs[i][c]) +
               " cpu=" + std::to_string(cpu.corner_mvs[i][c]);
      }
      if (gpu.corner.ref_pts[i][c] != cpu.ref_pts[i][c]) {
        return head() + "ref_pts[" + std::to_string(i) + "][" + std::to_string(c) +
               "] gpu=" + std::to_string(gpu.corner.ref_pts[i][c]) +
               " cpu=" + std::to_string(cpu.ref_pts[i][c]);
      }
    }
  }
  for (int i = 0; i < 6; ++i) {
    if (gpu.corner.wmmat.p[i] != cpu.wmmat.p[i]) {
      std::snprintf(buffer, sizeof(buffer),
                    "%swmmat.p[%d] gpu=%d cpu=%d all_gpu=[%d %d %d %d %d %d] "
                    "all_cpu=[%d %d %d %d %d %d]",
                    head().c_str(), i, gpu.corner.wmmat.p[i], cpu.wmmat.p[i],
                    gpu.corner.wmmat.p[0], gpu.corner.wmmat.p[1], gpu.corner.wmmat.p[2],
                    gpu.corner.wmmat.p[3], gpu.corner.wmmat.p[4], gpu.corner.wmmat.p[5],
                    cpu.wmmat.p[0], cpu.wmmat.p[1], cpu.wmmat.p[2], cpu.wmmat.p[3],
                    cpu.wmmat.p[4], cpu.wmmat.p[5]);
      return buffer;
    }
  }
  return head() + "no differing field (corner_fields_equal is stricter)";
}

void test_corners(Arena& arena, Workspace& workspace, int trials) {
  const int cols = 24;
  const int rows = 24;
  std::vector<iw::WarpMi> mi;
  fill_grid(&mi, cols, rows);
  const iw::WarpGrid host_grid{cols, rows, mi.data()};

  const std::uint64_t mark = arena.mark();
  // The grid must be uploaded before any job captures workspace.grid(), or the
  // job carries mi_cols == mi_rows == 0 and the kernel sees an empty grid.
  check_status(workspace.upload_mi_grid(mi.data(), cols, rows), "corners: upload grid");
  const iw::WarpGrid device_grid = workspace.grid();
  check(device_grid.mi != nullptr && device_grid.mi_cols == cols &&
            device_grid.mi_rows == rows,
        "corners: workspace grid shape");

  std::vector<WarpCornerJob> device_jobs(static_cast<std::size_t>(trials));
  std::vector<WarpCornerJob> host_jobs(static_cast<std::size_t>(trials));
  for (int t = 0; t < trials; ++t) {
    WarpCornerJob job{};
    job.sb_h4 = random_range(0, 1) ? 16 : 8;
    job.mi_row = random_range(0, rows - 1);
    job.mi_col = random_range(0, cols - 1);
    job.bw4 = std::min(1 << random_range(0, 4), cols - job.mi_col);
    job.bh4 = std::min(1 << random_range(0, 4), rows - job.mi_row);
    job.ref_frame0 = static_cast<std::int8_t>(random_range(1, 3));
    job.iter = random_range(0, 1);
    job.grid = host_grid;
    host_jobs[static_cast<std::size_t>(t)] = job;
    job.grid = device_grid;
    device_jobs[static_cast<std::size_t>(t)] = job;
  }

  check_status(workspace.submit_corner(device_jobs.data(), trials), "corners: submit");
  check(cudaDeviceSynchronize() == cudaSuccess, "corners: sync");
  std::vector<WarpCornerResult> got(static_cast<std::size_t>(trials));
  check_status(workspace.download_corner(got.data(), trials), "corners: download");
  check(cudaDeviceSynchronize() == cudaSuccess, "corners: download sync");

  int bad = 0;
  for (int t = 0; t < trials; ++t) {
    const WarpCornerJob& job = host_jobs[static_cast<std::size_t>(t)];
    iw::WarpStack cpu_stack{};
    const iw::CornerResult cpu = iw::generate_points_from_corners(
        host_grid, job.sb_h4, job.mi_row, job.mi_col, job.bw4, job.bh4, job.ref_frame0,
        job.iter, &cpu_stack);
    const bool valid = cpu_stack.num_warp_found > 0;
    const WarpCornerResult& result = got[static_cast<std::size_t>(t)];
    if (result.model_valid != (valid ? 1 : 0) ||
        !corner_fields_equal(result.corner, cpu)) {
      if (bad < 8) {
        std::printf("  corner mismatch #%d: %s\n", bad,
                    corner_diff(job, result, cpu, valid).c_str());
      }
      ++bad;
    }
  }
  check(bad == 0, "corners: GPU == CPU reference over " + std::to_string(trials) +
                     " jobs (" + std::to_string(bad) + " bad)");
  arena.rewind(mark);
}

void test_samples(Arena& arena, Workspace& workspace, int trials) {
  const int cols = 24;
  const int rows = 24;
  std::vector<iw::WarpMi> mi;
  fill_grid(&mi, cols, rows);
  const iw::WarpGrid host_grid{cols, rows, mi.data()};

  const std::uint64_t mark = arena.mark();
  // Same ordering rule as test_corners(): upload before capturing grid().
  check_status(workspace.upload_mi_grid(mi.data(), cols, rows), "samples: upload grid");
  const iw::WarpGrid device_grid = workspace.grid();
  check(device_grid.mi != nullptr && device_grid.mi_cols == cols &&
            device_grid.mi_rows == rows,
        "samples: workspace grid shape");

  std::vector<av2enc::cuda::warpstack::WarpSampleJob> device_jobs(
      static_cast<std::size_t>(trials));
  std::vector<iw::WarpSampleParams> host_params(static_cast<std::size_t>(trials));
  for (int t = 0; t < trials; ++t) {
    iw::WarpSampleParams params{};
    params.mi_row = random_range(1, rows - 1);
    params.mi_col = random_range(1, cols - 1);
    params.w4 = std::min(1 << random_range(0, 4), cols - params.mi_col);
    params.h4 = std::min(1 << random_range(0, 4), rows - params.mi_row);
    params.mi_cols = cols;
    params.mi_rows = rows;
    params.sb_h4 = random_range(0, 1) ? 16 : 8;
    params.grid = host_grid;
    params.ref_frame[0] = static_cast<std::int8_t>(random_range(1, 3));
    params.ref_frame[1] = random_range(0, 1) ? static_cast<std::int8_t>(random_range(1, 3))
                                             : iw::kRefNone;
    params.ref = random_range(0, 1);
    params.avail_u = true;
    params.avail_l = true;
    host_params[static_cast<std::size_t>(t)] = params;
    device_jobs[static_cast<std::size_t>(t)].params = params;
    device_jobs[static_cast<std::size_t>(t)].params.grid = device_grid;
  }

  check_status(workspace.submit_sample(device_jobs.data(), trials), "samples: submit");
  check(cudaDeviceSynchronize() == cudaSuccess, "samples: sync");
  std::vector<iw::WarpSampleSet> got(static_cast<std::size_t>(trials));
  check_status(workspace.download_sample(got.data(), trials), "samples: download");
  check(cudaDeviceSynchronize() == cudaSuccess, "samples: download sync");

  int bad = 0;
  for (int t = 0; t < trials; ++t) {
    iw::WarpSampleSet want{};
    const bool found = iw::find_warp_samples(host_params[static_cast<std::size_t>(t)], &want);
    const iw::WarpSampleSet& result = got[static_cast<std::size_t>(t)];
    bool same = (found == (result.num_samples > 0)) && result.num_samples == want.num_samples;
    for (int i = 0; same && i < want.num_samples; ++i) {
      for (int c = 0; c < 4; ++c) {
        if (result.cand[i][c] != want.cand[i][c]) {
          same = false;
        }
      }
    }
    if (!same) {
      ++bad;
    }
  }
  check(bad == 0, "samples: GPU == CPU reference over " + std::to_string(trials) +
                      " jobs (" + std::to_string(bad) + " bad)");
  arena.rewind(mark);
}

void test_estimates(Arena& arena, Workspace& workspace, int trials) {
  std::vector<WarpEstimateJob> jobs(static_cast<std::size_t>(trials));
  for (int t = 0; t < trials; ++t) {
    WarpEstimateJob& job = jobs[static_cast<std::size_t>(t)];
    job.samples.num_samples = random_range(0, iw::kLeastSquaresSamplesMax);
    for (int i = 0; i < job.samples.num_samples; ++i) {
      for (int c = 0; c < 4; ++c) {
        job.samples.cand[i][c] = random_range(-4000, 4000);
      }
    }
    job.mid_x = random_range(0, 400);
    job.mid_y = random_range(0, 400);
    job.block_mv[0] = random_range(-64, 64);
    job.block_mv[1] = random_range(-64, 64);
  }

  const std::uint64_t mark = arena.mark();
  check_status(workspace.submit_estimate(jobs.data(), trials), "estimates: submit");
  check(cudaDeviceSynchronize() == cudaSuccess, "estimates: sync");
  std::vector<ia::WarpParams> got(static_cast<std::size_t>(trials));
  check_status(workspace.download_estimate(got.data(), trials), "estimates: download");
  check(cudaDeviceSynchronize() == cudaSuccess, "estimates: download sync");

  int bad = 0;
  for (int t = 0; t < trials; ++t) {
    const WarpEstimateJob& job = jobs[static_cast<std::size_t>(t)];
    const ia::WarpParams want =
        iw::estimate_warp_from_samples(job.samples, job.mid_x, job.mid_y, job.block_mv);
    for (int i = 0; i < 6; ++i) {
      if (got[static_cast<std::size_t>(t)].p[i] != want.p[i]) {
        ++bad;
        break;
      }
    }
  }
  check(bad == 0, "estimates: GPU == CPU reference over " + std::to_string(trials) +
                      " jobs (" + std::to_string(bad) + " bad)");
  arena.rewind(mark);
}

// ---------------------------------------------------------------------------
// Arena lifecycle
// ---------------------------------------------------------------------------

WarpStackConfig make_config() {
  WarpStackConfig config;
  config.max_predict_jobs = static_cast<int>(prediction_cases().size());
  config.max_corner_jobs = 640;
  config.max_sample_jobs = 640;
  config.max_estimate_jobs = 512;
  config.max_mi_cells = 24 * 24;
  return config;
}

void test_arena_lifecycle(Arena& arena, Workspace& workspace, const WarpStackConfig& config) {
  const std::uint64_t required = required_device_bytes(config);
  const std::uint64_t after_create = arena.used();
  check(workspace.required_bytes() == required, "lifecycle: reported requirement");
  check(workspace.allocated_bytes() == required, "lifecycle: allocated requirement");
  check(after_create == required, "lifecycle: create consumed exactly required");

  // Two launch epochs with identical scratch; no net growth and no new peak.
  const std::uint64_t mark = arena.mark();
  std::uint64_t peak_after_first = 0;
  for (int epoch = 0; epoch < 2; ++epoch) {
    auto* scratch = arena.allocate_array<std::uint8_t>(1 << 20);
    check(scratch != nullptr, "lifecycle: scratch");
    check(arena.used() > after_create, "lifecycle: scratch grew the arena");
    if (epoch == 0) {
      peak_after_first = arena.peak_used();
    } else {
      check(arena.peak_used() == peak_after_first, "lifecycle: second epoch added no peak");
    }
    check(arena.rewind(mark), "lifecycle: rewind");
    check(arena.used() == after_create, "lifecycle: no net growth");
  }

  // One byte short: distinct status, reported requirement, nothing allocated.
  Arena small;
  check(small.init(required - 1) == Status::kOk, "lifecycle: init one byte short");
  Workspace short_workspace;
  std::string error;
  check(short_workspace.create(small, config, &error) == Status::kArenaExhausted,
        "lifecycle: one byte short is kArenaExhausted");
  check(short_workspace.required_bytes() == required,
        "lifecycle: requirement survives failure");
  check(!short_workspace.created(), "lifecycle: failed create not created");
  check(small.used() == 0, "lifecycle: failed create allocated nothing");
  check(!error.empty(), "lifecycle: failure reason");
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::printf("warp_stack_gpu_test: no CUDA device available\n");
    return 77;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::printf("device: %s (cc %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);

  const WarpStackConfig config = make_config();
  const std::uint64_t required = required_device_bytes(config);

  Arena arena;
  const Status init_status = arena.init(required + kSlack);
  if (init_status != Status::kOk) {
    std::printf("warp_stack_gpu_test: Arena::init failed: %s\n", status_name(init_status));
    return 1;
  }
  Workspace workspace;
  std::string error;
  const Status create_status = workspace.create(arena, config, &error);
  if (create_status != Status::kOk) {
    std::printf("warp_stack_gpu_test: Workspace::create failed: %s (%s)\n",
                status_name(create_status), error.c_str());
    return 1;
  }

  test_arena_lifecycle(arena, workspace, config);
  test_prediction(arena, workspace);
  test_paths_against_adv(arena, workspace);
  test_corners(arena, workspace, 600);
  test_samples(arena, workspace, 600);
  test_estimates(arena, workspace, 400);

  check(arena.used() == workspace.required_bytes(),
        "main: arena returned to the persistent footprint");
  std::printf(
      "warp_stack: required_device_bytes=%llu peak_used=%llu capacity=%llu "
      "persistent_after=%llu\n",
      static_cast<unsigned long long>(required),
      static_cast<unsigned long long>(arena.peak_used()),
      static_cast<unsigned long long>(arena.capacity()),
      static_cast<unsigned long long>(arena.used()));

  if (g_failures == 0) {
    std::printf("warp_stack_gpu_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::printf("warp_stack_gpu_test: %d/%d checks FAILED\n", g_failures, g_checks);
  return 1;
}
