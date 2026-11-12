// CPU-vs-GPU bit-exactness test for the AV2 §7.20.5 GDF CUDA filter.
//
// Every device buffer comes from one persistent av2enc::cuda::Arena; there are
// no cudaMalloc/cudaFree/cudaMallocHost calls here.  Launchers are exercised
// with mark()/rewind() around them and checked for zero net growth.
//
// Coverage:
//   * §7.20.1 derivation, 2048 random descriptors vs GdfBlockTableIndices;
//   * all 6x6 refDstIdx x qpIdx combinations x 3 bit depths x 4 GdfPixScale;
//   * unit sizes 2..128 incl. 32/64/128 (unit-sized and SB-sized grids);
//   * high-bit-depth full-range sources (10/12-bit);
//   * random blocks, random stripe/plane extents, mixed-bit-depth batches;
//   * GdfBlks use_gdf masks (all-on, random, all-off);
//   * argument validation and the arena one-byte-short status;
//   * 1080p grid timing with CUDA events.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/arena.h"
#include "cuda/codec/filters/gdf/gdf.h"
#include "cuda/codec/filters/gdf/gdf_device.cuh"
#include "codec/filters/tables/gdf.h"
#include "codec/filters/tables/luma_source.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using av2enc::cuda::gdf::GdfConfig;
using av2enc::cuda::gdf::GdfDeviceBuffers;
using av2enc::cuda::gdf::GdfFrameGrid;
using av2enc::cuda::gdf::GdfFrameSource;
using av2enc::cuda::gdf::GdfIndexParams;
using av2enc::cuda::gdf::GdfRegion;
using av2enc::cuda::gdf::GdfTableIndices;
using av2enc::cuda::gdf::GdfUnitJob;
namespace gdf = av2enc::cuda::gdf;
namespace spec = av2enc::filters::spec;

// Job-slot capacity: 3 stripe regions per unit covers a 30x17 (1080p) unit
// grid, the largest grid used here.
constexpr int kTestSlots = 2048;

struct Rng {
  std::uint64_t state = 1;
  explicit Rng(std::uint64_t seed) : state(seed == 0 ? 1 : seed) {}
  std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<std::uint32_t>(state >> 32);
  }
  int range(int low, int high) {
    return low + static_cast<int>(next() %
                                  static_cast<std::uint32_t>(high - low + 1));
  }
  bool coin() { return (next() & 1u) != 0; }
};

struct Extents {
  int start_x = 0;
  int end_x = 0;
  int start_y = 0;
  int end_y = 0;
  int stripe_start_y = 0;
  int stripe_end_y = 0;
};

struct DevicePlane {
  std::uint16_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;
};

struct TestContext {
  Arena arena;
  GdfDeviceBuffers buffers;
  std::size_t required = 0;
  std::uint64_t peak_before_probe = 0;
  long long checks = 0;
  long long samples = 0;
  long long changed = 0;
  int failures = 0;

  void Check(bool condition, const char* what) {
    ++checks;
    if (!condition) {
      ++failures;
      std::fprintf(stderr, "FAIL: %s\n", what);
    }
  }
};

void CudaCheck(TestContext* tc, cudaError_t error, const char* what) {
  tc->Check(error == cudaSuccess, what);
  if (error != cudaSuccess) {
    std::fprintf(stderr, "  cuda: %s (%s)\n", cudaGetErrorString(error),
                 cudaGetErrorName(error));
  }
}

void ExpectStatus(TestContext* tc, Status got, Status want, const char* what) {
  tc->Check(got == want, what);
  if (got != want) {
    std::fprintf(stderr, "  status: got %s, want %s\n",
                 av2enc::cuda::status_name(got),
                 av2enc::cuda::status_name(want));
  }
}

void FillRandom(std::vector<std::uint16_t>* plane, Rng* rng, int maxv) {
  for (auto& v : *plane) {
    v = static_cast<std::uint16_t>(rng->range(0, maxv));
  }
}

DevicePlane AllocPlane(TestContext* tc, int width, int height,
                       const char* what) {
  DevicePlane plane;
  plane.width = width;
  plane.height = height;
  plane.stride = width;
  plane.data = tc->arena.allocate_array<std::uint16_t>(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  tc->Check(plane.data != nullptr, what);
  return plane;
}

bool UploadPlane(TestContext* tc, const DevicePlane& plane,
                 const std::vector<std::uint16_t>& host, const char* what) {
  const std::size_t bytes = static_cast<std::size_t>(plane.width) *
                            static_cast<std::size_t>(plane.height) *
                            sizeof(std::uint16_t);
  CudaCheck(tc,
            cudaMemcpy(plane.data, host.data(), bytes, cudaMemcpyHostToDevice),
            what);
  return true;
}

bool DownloadPlane(TestContext* tc, const DevicePlane& plane,
                   std::vector<std::uint16_t>* host, const char* what) {
  const std::size_t bytes = static_cast<std::size_t>(plane.width) *
                            static_cast<std::size_t>(plane.height) *
                            sizeof(std::uint16_t);
  CudaCheck(tc,
            cudaMemcpy(host->data(), plane.data, bytes, cudaMemcpyDeviceToHost),
            what);
  return true;
}

GdfFrameSource MakeSource(const DevicePlane& curr, const DevicePlane& cdef,
                          const Extents& e) {
  GdfFrameSource src;
  src.curr = curr.data;
  src.curr_stride = curr.stride;
  src.cdef = cdef.data;
  src.cdef_stride = cdef.stride;
  src.start_x = e.start_x;
  src.end_x = e.end_x;
  src.start_y = e.start_y;
  src.end_y = e.end_y;
  src.stripe_start_y = e.stripe_start_y;
  src.stripe_end_y = e.stripe_end_y;
  return src;
}

spec::LumaSource MakeCpuSource(const std::vector<std::uint16_t>& curr,
                               const std::vector<std::uint16_t>& cdef,
                               int stride, const Extents& e) {
  spec::LumaSource src;
  src.curr = curr.data();
  src.cdef = cdef.data();
  src.curr_stride = stride;
  src.cdef_stride = stride;
  src.luma_start_x = e.start_x;
  src.luma_end_x = e.end_x;
  src.luma_start_y = e.start_y;
  src.luma_end_y = e.end_y;
  src.luma_stripe_start_y = e.stripe_start_y;
  src.luma_stripe_end_y = e.stripe_end_y;
  return src;
}

Extents FullExtents(int width, int height) {
  Extents e;
  e.start_x = 0;
  e.end_x = width - 1;
  e.start_y = 0;
  e.end_y = height - 1;
  e.stripe_start_y = 0;
  e.stripe_end_y = height - 1;
  return e;
}

Extents RandomExtents(Rng* rng, int width, int height) {
  if (rng->coin()) {
    return FullExtents(width, height);
  }
  Extents e;
  e.start_x = rng->range(0, width - 1);
  e.end_x = rng->range(e.start_x, width - 1);
  e.start_y = rng->range(0, height - 1);
  e.end_y = rng->range(e.start_y, height - 1);
  e.stripe_start_y = rng->range(0, height - 1);
  e.stripe_end_y = rng->range(e.stripe_start_y, height - 1);
  return e;
}

void CompareResult(TestContext* tc, const std::vector<std::uint16_t>& before,
                   const std::vector<std::uint16_t>& want,
                   const std::vector<std::uint16_t>& got, const char* label) {
  for (std::size_t i = 0; i < want.size(); ++i) {
    ++tc->samples;
    ++tc->checks;
    if (got[i] != want[i]) {
      ++tc->failures;
      if (tc->failures < 12) {
        std::fprintf(stderr, "FAIL %s: sample %zu gpu=%u cpu=%u\n", label, i,
                     static_cast<unsigned>(got[i]),
                     static_cast<unsigned>(want[i]));
      }
    }
    if (want[i] != before[i]) {
      ++tc->changed;
    }
  }
}

// ---------------------------------------------------------------------------
// §7.20.1 derivation
// ---------------------------------------------------------------------------
void TestDerivation(TestContext* tc) {
  const int count = 2048;
  const std::uint64_t mark = tc->arena.mark();
  GdfIndexParams* d_params = tc->arena.allocate_array<GdfIndexParams>(count);
  GdfTableIndices* d_out = tc->arena.allocate_array<GdfTableIndices>(count);
  int* d_hints =
      tc->arena.allocate_array<int>(static_cast<std::size_t>(2) * count);
  int* d_dists =
      tc->arena.allocate_array<int>(static_cast<std::size_t>(2) * count);
  tc->Check(d_params != nullptr && d_out != nullptr && d_hints != nullptr &&
                d_dists != nullptr,
            "derivation buffers");

  std::vector<GdfIndexParams> host_params(count);
  std::vector<int> host_hints(2 * count, 0);
  std::vector<int> host_dists(2 * count, 0);
  std::vector<GdfTableIndices> host_out(count);
  Rng rng(0x7a11ce);
  for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
    GdfIndexParams& p = host_params[i];
    p.frame_is_intra = rng.coin() ? 1 : 0;
    p.base_q_idx = rng.range(0, 255);
    p.bit_depth = rng.coin() ? 8 : (rng.coin() ? 10 : 12);
    p.gdf_pic_qc_idx = rng.range(0, 3);
    host_hints[2 * i] = rng.range(-4, 8);
    host_hints[2 * i + 1] = rng.range(-4, 8);
    host_dists[2 * i] = rng.range(-24, 24);
    host_dists[2 * i + 1] = rng.range(-24, 24);
    p.order_hints = d_hints + 2 * i;
    p.frame_distance = d_dists + 2 * i;
    p.num_total_refs = rng.range(0, 3);
    p.restricted_oh = rng.range(-1, 8);
    if (rng.coin()) {
      host_hints[2 * i] = p.restricted_oh;
    }
  }

  CudaCheck(tc,
            cudaMemcpy(d_params, host_params.data(),
                       sizeof(GdfIndexParams) * count, cudaMemcpyHostToDevice),
            "upload derivation params");
  CudaCheck(tc,
            cudaMemcpy(d_hints, host_hints.data(), sizeof(int) * 2 * count,
                       cudaMemcpyHostToDevice),
            "upload order hints");
  CudaCheck(tc,
            cudaMemcpy(d_dists, host_dists.data(), sizeof(int) * 2 * count,
                       cudaMemcpyHostToDevice),
            "upload frame distances");

  ExpectStatus(tc, gdf::gdf_derive_indices(d_params, d_out, count), Status::kOk,
               "gdf_derive_indices");
  CudaCheck(tc,
            cudaMemcpy(host_out.data(), d_out,
                       sizeof(GdfTableIndices) * count,
                       cudaMemcpyDeviceToHost),
            "download derivation");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync derivation");

  int seen_ref[6] = {0, 0, 0, 0, 0, 0};
  int seen_qp[6] = {0, 0, 0, 0, 0, 0};
  for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
    spec::GdfParams cpu;
    cpu.frame_is_intra = host_params[i].frame_is_intra != 0;
    cpu.base_q_idx = host_params[i].base_q_idx;
    cpu.bit_depth = host_params[i].bit_depth;
    cpu.gdf_pic_qc_idx = host_params[i].gdf_pic_qc_idx;
    const spec::GdfBlockIndices want = spec::GdfBlockTableIndices(
        cpu, &host_hints[2 * i], &host_dists[2 * i],
        host_params[i].num_total_refs, host_params[i].restricted_oh);
    tc->Check(host_out[i].qp_idx == want.qp_idx, "GPU §7.20.1 qpIdx");
    tc->Check(host_out[i].ref_dst_idx == want.ref_dst_idx,
              "GPU §7.20.1 refDstIdx");
    ++seen_ref[host_out[i].ref_dst_idx];
    ++seen_qp[host_out[i].qp_idx];
  }
  for (int i = 0; i < 6; ++i) {
    tc->Check(seen_ref[i] > 0, "GPU refDstIdx coverage");
    tc->Check(seen_qp[i] > 0, "GPU qpIdx coverage");
  }
  std::printf("  §7.20.1 GPU derivation: %d descriptors exact\n", count);
  (void)tc->arena.rewind(mark);
}

// ---------------------------------------------------------------------------
// Batched unit parity
// ---------------------------------------------------------------------------
void RunBatchParity(TestContext* tc, int width, int height,
                    std::vector<GdfUnitJob> jobs, int max_unit_size,
                    int max_sample_value, const Extents& e, std::uint64_t seed,
                    const char* label) {
  const std::uint64_t mark = tc->arena.mark();
  DevicePlane curr = AllocPlane(tc, width, height, "batch curr plane");
  DevicePlane cdef = AllocPlane(tc, width, height, "batch cdef plane");
  DevicePlane lr = AllocPlane(tc, width, height, "batch lr plane");
  if (curr.data == nullptr || cdef.data == nullptr || lr.data == nullptr) {
    (void)tc->arena.rewind(mark);
    return;
  }

  Rng rng(seed);
  const std::size_t pixels =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<std::uint16_t> host_curr(pixels);
  std::vector<std::uint16_t> host_cdef(pixels);
  std::vector<std::uint16_t> host_lr(pixels);
  FillRandom(&host_curr, &rng, max_sample_value);
  FillRandom(&host_cdef, &rng, max_sample_value);
  FillRandom(&host_lr, &rng, max_sample_value);
  UploadPlane(tc, curr, host_curr, "upload curr");
  UploadPlane(tc, cdef, host_cdef, "upload cdef");
  UploadPlane(tc, lr, host_lr, "upload lr");

  for (auto& job : jobs) {
    job.lr = lr.data;
    job.lr_stride = lr.stride;
  }
  ExpectStatus(tc,
               gdf::gdf_upload_jobs(tc->buffers, jobs.data(),
                                    static_cast<int>(jobs.size())),
               Status::kOk, "gdf_upload_jobs");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync job upload");

  std::vector<std::uint16_t> cpu_lr = host_lr;
  const spec::LumaSource cpu_src =
      MakeCpuSource(host_curr, host_cdef, width, e);
  for (const auto& job : jobs) {
    if (job.use_gdf == 0) {
      continue;
    }
    spec::ApplyGdfFilter(cpu_src, cpu_lr.data(), width, job.x, job.y, job.qp_idx,
                         job.ref_dst_idx, job.w, job.h, e.stripe_end_y + 1,
                         job.pix_scale, job.bit_depth);
  }

  const GdfFrameSource src = MakeSource(curr, cdef, e);
  ExpectStatus(tc,
               gdf::gdf_apply_units(src, tc->buffers.jobs,
                                    static_cast<int>(jobs.size()), max_unit_size,
                                    tc->buffers.tables),
               Status::kOk, "gdf_apply_units");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync batch filter");

  std::vector<std::uint16_t> gpu_lr(pixels);
  DownloadPlane(tc, lr, &gpu_lr, "download lr");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync lr download");
  CompareResult(tc, host_lr, cpu_lr, gpu_lr, label);

  (void)tc->arena.rewind(mark);
}

void TestAllCombinations(TestContext* tc) {
  // 22x20 tiles of 64x64; every job owns one tile so the batch has no
  // overlapping writes.  432 jobs cover 3 bit depths x 6 ref x 6 qp x 4 pix.
  const int tile = 64;
  const int columns = 22;
  const int rows = 20;
  const int width = columns * tile;
  const int height = rows * tile;
  std::vector<GdfUnitJob> jobs;
  std::vector<Extents> extents(2);
  Rng rng(0x5eed1000);
  extents[0] = FullExtents(width, height);
  extents[1] = RandomExtents(&rng, width, height);
  int index = 0;
  for (int bit_depth : {8, 10, 12}) {
    for (int ref = 0; ref < 6; ++ref) {
      for (int qp = 0; qp < 6; ++qp) {
        for (int pix = 1; pix <= 4; ++pix) {
          GdfUnitJob job;
          job.bit_depth = bit_depth;
          job.ref_dst_idx = ref;
          job.qp_idx = qp;
          job.pix_scale = pix;
          job.w = 2 * rng.range(1, 32);
          job.h = 2 * rng.range(1, 32);
          const int tile_x = (index % columns) * tile;
          const int tile_y = (index / columns) * tile;
          job.x = tile_x + 2 * rng.range(0, (tile - job.w) / 2);
          job.y = tile_y + 2 * rng.range(0, (tile - job.h) / 2);
          job.use_gdf = 1;
          jobs.push_back(job);
          ++index;
        }
      }
    }
  }
  tc->Check(static_cast<int>(jobs.size()) == 432,
            "all-combination batch has 432 jobs");
  for (const auto& e : extents) {
    RunBatchParity(tc, width, height, jobs, tile, 255, e, 0xabc001,
                   "all-combinations");
  }
  std::printf("  all refDstIdx x qpIdx x 8/10/12-bit x pixScale: 2 x %zu jobs\n",
              jobs.size());
}

void TestHighBitDepthRange(TestContext* tc) {
  // Full-range 10/12-bit sources so alpha clipping is exercised at the real
  // sample depth rather than at 8-bit amplitudes.
  for (int bit_depth : {10, 12}) {
    const int tile = 64;
    const int width = 6 * tile;
    const int height = 6 * tile;
    std::vector<GdfUnitJob> jobs;
    Rng rng(bit_depth == 10 ? 0x101010 : 0x121212);
    int index = 0;
    for (int ref = 0; ref < 6; ++ref) {
      for (int qp = 0; qp < 6; ++qp) {
        GdfUnitJob job;
        job.bit_depth = bit_depth;
        job.ref_dst_idx = ref;
        job.qp_idx = qp;
        job.pix_scale = 1 + ((ref + qp) % 4);
        job.w = 2 * rng.range(1, 32);
        job.h = 2 * rng.range(1, 32);
        const int tile_x = (index % 6) * tile;
        const int tile_y = (index / 6) * tile;
        job.x = tile_x + 2 * rng.range(0, (tile - job.w) / 2);
        job.y = tile_y + 2 * rng.range(0, (tile - job.h) / 2);
        job.use_gdf = 1;
        jobs.push_back(job);
        ++index;
      }
    }
    RunBatchParity(tc, width, height, jobs, tile, (1 << bit_depth) - 1,
                   RandomExtents(&rng, width, height),
                   static_cast<std::uint64_t>(0x777000 + bit_depth),
                   "high-bit-depth-range");
  }
  std::printf("  full-range 10/12-bit: 72 jobs exact\n");
}

void TestLargeUnits(TestContext* tc) {
  // 12x9 tiles of 128x128: 108 jobs, all 36 combinations at three depths.
  const int tile = 128;
  const int width = 12 * tile;
  const int height = 9 * tile;
  std::vector<GdfUnitJob> jobs;
  Rng rng(0x128128);
  int index = 0;
  for (int bit_depth : {8, 10, 12}) {
    for (int ref = 0; ref < 6; ++ref) {
      for (int qp = 0; qp < 6; ++qp) {
        GdfUnitJob job;
        job.bit_depth = bit_depth;
        job.ref_dst_idx = ref;
        job.qp_idx = qp;
        job.pix_scale = 1 + ((ref + qp) % 4);
        job.w = 128;
        job.h = ((ref + qp) & 1) ? 64 : 128;
        const int tile_x = (index % 12) * tile;
        const int tile_y = (index / 12) * tile;
        job.x = tile_x;
        job.y = tile_y + (((ref + qp) & 1) ? 64 : 0);
        job.use_gdf = 1;
        jobs.push_back(job);
        ++index;
      }
    }
  }
  RunBatchParity(tc, width, height, jobs, 128, 255,
                 RandomExtents(&rng, width, height), 0x9911, "128x128-units");
  std::printf("  128x128 / 128x64 units x 36 combos: %zu jobs exact\n",
              jobs.size());
}

void TestRandomBlocks(TestContext* tc) {
  const int tile = 64;
  const int width = 20 * tile;
  const int height = 20 * tile;
  std::vector<GdfUnitJob> jobs;
  Rng rng(0xdeadbeef);
  for (int i = 0; i < 300; ++i) {
    GdfUnitJob job;
    job.w = 2 * rng.range(1, 32);
    job.h = 2 * rng.range(1, 32);
    const int tile_index = i;  // 20x20 grid, one job per tile
    const int tile_x = (tile_index % 20) * tile;
    const int tile_y = (tile_index / 20) * tile;
    job.x = tile_x + 2 * rng.range(0, (tile - job.w) / 2);
    job.y = tile_y + 2 * rng.range(0, (tile - job.h) / 2);
    job.bit_depth = rng.coin() ? 8 : (rng.coin() ? 10 : 12);
    job.ref_dst_idx = rng.range(0, 5);
    job.qp_idx = rng.range(0, 5);
    job.pix_scale = rng.range(1, 4);
    job.use_gdf = rng.range(0, 9) == 0 ? 0 : 1;
    jobs.push_back(job);
  }
  RunBatchParity(tc, width, height, jobs, tile, 255,
                 RandomExtents(&rng, width, height), 0x515151,
                 "random-blocks");
  std::printf("  random blocks: %zu jobs (5%% disabled by use_gdf) exact\n",
              jobs.size());
}

// ---------------------------------------------------------------------------
// Unit-grid (GdfBlks) parity
// ---------------------------------------------------------------------------
void RunGridParity(TestContext* tc, int width, int height, int unit_size,
                   int bit_depth, const Extents& e,
                   const GdfIndexParams& index,
                   const std::vector<std::uint8_t>& use_gdf, std::uint64_t seed,
                   const char* label) {
  const int cols = (width + unit_size - 1) / unit_size;
  const int rows = (height + unit_size - 1) / unit_size;
  tc->Check(static_cast<int>(use_gdf.size()) == cols * rows,
            "use_gdf mask size");

  const std::uint64_t mark = tc->arena.mark();
  DevicePlane curr = AllocPlane(tc, width, height, "grid curr plane");
  DevicePlane cdef = AllocPlane(tc, width, height, "grid cdef plane");
  DevicePlane lr = AllocPlane(tc, width, height, "grid lr plane");
  int* d_hints =
      tc->arena.allocate_array<int>(static_cast<std::size_t>(2));
  int* d_dists =
      tc->arena.allocate_array<int>(static_cast<std::size_t>(2));
  if (curr.data == nullptr || cdef.data == nullptr || lr.data == nullptr ||
      d_hints == nullptr || d_dists == nullptr) {
    (void)tc->arena.rewind(mark);
    return;
  }

  Rng rng(seed);
  const std::size_t pixels =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<std::uint16_t> host_curr(pixels);
  std::vector<std::uint16_t> host_cdef(pixels);
  std::vector<std::uint16_t> host_lr(pixels);
  const int maxv = (1 << bit_depth) - 1;
  FillRandom(&host_curr, &rng, maxv);
  FillRandom(&host_cdef, &rng, maxv);
  FillRandom(&host_lr, &rng, maxv);
  UploadPlane(tc, curr, host_curr, "upload grid curr");
  UploadPlane(tc, cdef, host_cdef, "upload grid cdef");
  UploadPlane(tc, lr, host_lr, "upload grid lr");

  int hints[2] = {0, 1};
  int dists[2] = {0, 0};
  if (index.frame_is_intra == 0) {
    hints[0] = 4;
    hints[1] = 2;
    dists[0] = rows > 2 ? 7 : 1;
    dists[1] = 0;
  }
  CudaCheck(tc, cudaMemcpy(d_hints, hints, sizeof(hints), cudaMemcpyHostToDevice),
            "upload grid hints");
  CudaCheck(tc, cudaMemcpy(d_dists, dists, sizeof(dists), cudaMemcpyHostToDevice),
            "upload grid distances");

  GdfIndexParams dev_index = index;
  dev_index.order_hints = index.frame_is_intra == 0 ? d_hints : nullptr;
  dev_index.frame_distance = index.frame_is_intra == 0 ? d_dists : nullptr;

  std::vector<std::uint8_t> host_mask = use_gdf;
  ExpectStatus(tc,
               gdf::gdf_upload_use_gdf(tc->buffers, host_mask.data(),
                                       cols * rows),
               Status::kOk, "gdf_upload_use_gdf");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync mask upload");

  GdfFrameGrid frame;
  frame.src = MakeSource(curr, cdef, e);
  frame.lr = lr.data;
  frame.lr_stride = lr.stride;
  frame.bit_depth = bit_depth;
  frame.pix_scale = 1 + static_cast<int>(seed % 4);
  frame.unit_size = unit_size;
  frame.cols = cols;
  frame.rows = rows;
  frame.width = width;
  frame.height = height;
  frame.mi_rows = height / 4;
  frame.index = dev_index;

  // CPU oracle: same §7.20.1 derivation and the same mask.
  spec::GdfParams cpu_params;
  cpu_params.frame_is_intra = index.frame_is_intra != 0;
  cpu_params.base_q_idx = index.base_q_idx;
  cpu_params.bit_depth = index.bit_depth;
  cpu_params.gdf_pic_qc_idx = index.gdf_pic_qc_idx;
  const spec::GdfBlockIndices cpu_index = spec::GdfBlockTableIndices(
      cpu_params, hints, dists, index.num_total_refs, index.restricted_oh);

  std::vector<std::uint16_t> cpu_lr = host_lr;
  const spec::LumaSource base_src =
      MakeCpuSource(host_curr, host_cdef, width, e);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int unit = row * cols + col;
      if (host_mask[static_cast<std::size_t>(unit)] == 0) {
        continue;
      }
      GdfRegion regions[gdf::kGdfMaxRegionsPerUnit];
      const int region_count = gdf_unit_regions(
          unit_size, row, col, cols, rows, frame.mi_rows, width, height,
          regions);
      for (int r = 0; r < region_count; ++r) {
        const GdfRegion& region = regions[r];
        if (region.w < 2 || region.h < 2 || (region.w & 1) ||
            (region.h & 1)) {
          continue;
        }
        spec::LumaSource cpu_src = base_src;
        cpu_src.luma_stripe_start_y = region.stripe_start_y;
        cpu_src.luma_stripe_end_y = region.stripe_end_y;
        spec::ApplyGdfFilter(cpu_src, cpu_lr.data(), width, region.x, region.y,
                             cpu_index.qp_idx, cpu_index.ref_dst_idx, region.w,
                             region.h, region.stripe_end_y, frame.pix_scale,
                             bit_depth);
      }
    }
  }

  ExpectStatus(tc, gdf::gdf_apply_grid(frame, tc->buffers), Status::kOk,
               "gdf_apply_grid");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync grid filter");

  std::vector<std::uint16_t> gpu_lr(pixels);
  DownloadPlane(tc, lr, &gpu_lr, "download grid lr");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync grid download");
  CompareResult(tc, host_lr, cpu_lr, gpu_lr, label);
  (void)tc->arena.rewind(mark);
}

void TestGrids(TestContext* tc) {
  const int width = 320;
  const int height = 192;
  Rng rng(0x320192);
  int launches = 0;
  for (int unit_size : {32, 64, 128}) {
    for (int bit_depth : {8, 10, 12}) {
      for (int pattern = 0; pattern < 3; ++pattern) {
        const int cols = (width + unit_size - 1) / unit_size;
        const int rows = (height + unit_size - 1) / unit_size;
        std::vector<std::uint8_t> mask(
            static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows), 0);
        if (pattern == 0) {
          std::fill(mask.begin(), mask.end(), 1);
        } else if (pattern == 1) {
          for (auto& v : mask) {
            v = rng.coin() ? 1 : 0;
          }
          if (std::find(mask.begin(), mask.end(), 1) == mask.end()) {
            mask[0] = 1;
          }
        }  // pattern 2: all zero

        GdfIndexParams index;
        index.frame_is_intra = rng.coin() ? 1 : 0;
        index.base_q_idx = rng.range(0, 255);
        index.bit_depth = bit_depth;
        index.gdf_pic_qc_idx = rng.range(0, 3);
        index.num_total_refs = index.frame_is_intra ? 0 : 2;
        index.restricted_oh = index.frame_is_intra ? 0 : rng.range(0, 5);

        RunGridParity(tc, width, height, unit_size, bit_depth,
                      RandomExtents(&rng, width, height), index, mask,
                      static_cast<std::uint64_t>(0x1000 + unit_size + pattern), "grid");
        ++launches;
      }
    }
  }
  std::printf(
      "  GdfBlks grids: %d launches (unit 32/64/128 x 8/10/12-bit x "
      "all/random/off masks)\n",
      launches);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
void TestValidation(TestContext* tc) {
  GdfDeviceBuffers bad_buffers;

  // Initialization validation.
  {
    Arena uninit;
    GdfDeviceBuffers out;
    std::size_t required = 0;
    ExpectStatus(tc,
                 gdf::gdf_buffers_init(uninit, GdfConfig{1}, &out, &required),
                 Status::kNotInitialized, "init rejects uninitialized arena");
    ExpectStatus(tc,
                 gdf::gdf_buffers_init(tc->arena, GdfConfig{0}, &out, &required),
                 Status::kInvalidArgument, "init rejects max_units 0");
    ExpectStatus(tc,
                 gdf::gdf_buffers_init(tc->arena, GdfConfig{-3}, &out, &required),
                 Status::kInvalidArgument, "init rejects negative max_units");
    ExpectStatus(
        tc,
        gdf::gdf_buffers_init(tc->arena,
                              GdfConfig{gdf::kGdfMaxUnits + 1}, &out, &required),
        Status::kInvalidArgument, "init rejects oversize max_units");
    ExpectStatus(tc,
                 gdf::gdf_buffers_init(tc->arena, GdfConfig{1}, nullptr),
                 Status::kInvalidArgument, "init rejects null output");
  }

  const std::uint64_t mark = tc->arena.mark();
  DevicePlane plane = AllocPlane(tc, 16, 16, "validation plane");
  GdfFrameSource good_src;
  good_src.curr = plane.data;
  good_src.cdef = plane.data;
  good_src.curr_stride = plane.stride;
  good_src.cdef_stride = plane.stride;
  good_src.start_x = 0;
  good_src.end_x = plane.width - 1;
  good_src.start_y = 0;
  good_src.end_y = plane.height - 1;
  good_src.stripe_start_y = 0;
  good_src.stripe_end_y = plane.height - 1;

  GdfUnitJob* d_job =
      tc->arena.allocate_array<GdfUnitJob>(static_cast<std::size_t>(1));
  tc->Check(d_job != nullptr, "validation job buffer");
  CudaCheck(tc, cudaMemset(d_job, 0, sizeof(GdfUnitJob)), "clear job");

  ExpectStatus(tc, gdf::gdf_apply_units(good_src, nullptr, 1, 64, tc->buffers.tables),
               Status::kInvalidArgument, "units rejects null jobs");
  ExpectStatus(tc, gdf::gdf_apply_units(good_src, d_job, -1, 64, tc->buffers.tables),
               Status::kInvalidArgument, "units rejects negative count");
  ExpectStatus(tc, gdf::gdf_apply_units(good_src, d_job, 1, 0, tc->buffers.tables),
               Status::kInvalidArgument, "units rejects unit size 0");
  ExpectStatus(tc, gdf::gdf_apply_units(good_src, d_job, 1, 3, tc->buffers.tables),
               Status::kInvalidArgument, "units rejects odd unit size");
  ExpectStatus(tc, gdf::gdf_apply_units(good_src, d_job, 1, 130, tc->buffers.tables),
               Status::kInvalidArgument, "units rejects oversize unit");
  GdfFrameSource bad_src = good_src;
  bad_src.curr = nullptr;
  ExpectStatus(tc, gdf::gdf_apply_units(bad_src, d_job, 1, 64, tc->buffers.tables),
               Status::kInvalidArgument, "units rejects null curr");
  bad_src = good_src;
  bad_src.curr_stride = 0;
  ExpectStatus(tc, gdf::gdf_apply_units(bad_src, d_job, 1, 64, tc->buffers.tables),
               Status::kInvalidArgument, "units rejects zero stride");
  ExpectStatus(tc, gdf::gdf_apply_units(good_src, d_job, 0, 64, tc->buffers.tables), Status::kOk,
               "units accepts zero jobs");

  GdfFrameGrid frame;
  frame.src = good_src;
  frame.lr = plane.data;
  frame.lr_stride = plane.stride;
  frame.bit_depth = 8;
  frame.pix_scale = 1;
  frame.unit_size = 16;
  frame.cols = 1;
  frame.rows = 1;
  frame.width = 16;
  frame.height = 16;
  frame.mi_rows = 4;
  frame.index.frame_is_intra = 1;

  const std::vector<std::uint8_t> ones(1, 1);
  ExpectStatus(tc, gdf::gdf_upload_use_gdf(tc->buffers, ones.data(), 1),
               Status::kOk, "validation mask upload");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "sync validation mask");

  ExpectStatus(tc, gdf::gdf_apply_grid(frame, tc->buffers), Status::kOk,
               "grid accepts valid geometry");
  GdfFrameGrid bad = frame;
  bad.bit_depth = 9;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects bit depth 9");
  bad = frame;
  bad.pix_scale = 0;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects pix_scale 0");
  bad = frame;
  bad.pix_scale = 5;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects pix_scale 5");
  bad = frame;
  bad.unit_size = 33;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects odd unit size");
  bad = frame;
  bad.cols = 0;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects zero columns");
  bad = frame;
  bad.lr = nullptr;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects null lr");
  bad = frame;
  bad.src.curr_stride = -4;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects negative stride");
  bad = frame;
  bad.index.frame_is_intra = 0;
  bad.index.num_total_refs = 2;
  bad.index.order_hints = nullptr;
  bad.index.frame_distance = nullptr;
  ExpectStatus(tc, gdf::gdf_apply_grid(bad, tc->buffers),
               Status::kInvalidArgument, "grid rejects missing ref arrays");
  {
    GdfDeviceBuffers small = tc->buffers;
    small.max_units = 1;
    ExpectStatus(tc, gdf::gdf_apply_grid(frame, small),
                 Status::kInvalidArgument, "grid rejects undersized buffers");
    GdfDeviceBuffers no_mask = tc->buffers;
    no_mask.use_gdf = nullptr;
    ExpectStatus(tc, gdf::gdf_apply_grid(frame, no_mask),
                 Status::kInvalidArgument, "grid rejects null mask");
  }

  ExpectStatus(tc, gdf::gdf_derive_indices(nullptr, nullptr, 4),
               Status::kInvalidArgument, "derive rejects null buffers");
  ExpectStatus(tc, gdf::gdf_derive_indices(nullptr, nullptr, -1),
               Status::kInvalidArgument, "derive rejects negative count");
  ExpectStatus(tc, gdf::gdf_derive_indices(nullptr, nullptr, 0), Status::kOk,
               "derive accepts zero count");

  ExpectStatus(tc, gdf::gdf_upload_jobs(bad_buffers, nullptr, 1),
               Status::kNotInitialized, "upload rejects uninitialized buffers");
  ExpectStatus(tc, gdf::gdf_upload_jobs(tc->buffers, nullptr, 1),
               Status::kInvalidArgument, "upload rejects null host jobs");
  ExpectStatus(tc,
               gdf::gdf_upload_jobs(tc->buffers, nullptr,
                                    tc->buffers.max_units + 1),
               Status::kInvalidArgument, "upload rejects oversize count");
  ExpectStatus(tc, gdf::gdf_upload_use_gdf(tc->buffers, nullptr, 1),
               Status::kInvalidArgument, "mask upload rejects null host mask");
  ExpectStatus(tc, gdf::gdf_upload_jobs(tc->buffers, nullptr, 0), Status::kOk,
               "upload accepts zero jobs");
  ExpectStatus(tc, gdf::gdf_upload_use_gdf(tc->buffers, nullptr, 0),
               Status::kOk, "mask upload accepts zero");
  (void)tc->arena.rewind(mark);
  std::printf("  validation: argument and status checks passed\n");
}

// ---------------------------------------------------------------------------
// Arena sizing, mark/rewind and one-byte-short behaviour
// ---------------------------------------------------------------------------
void TestArena(TestContext* tc) {
  for (int units : {1, 256, 1024, 8192, 65536}) {
    const std::size_t bytes = gdf::required_device_bytes(GdfConfig{units});
    std::printf("  required_device_bytes(max_units=%d) = %zu B\n", units,
                bytes);
    tc->Check(bytes > 0, "required_device_bytes is positive");
  }
  tc->Check(gdf::required_device_bytes(GdfConfig{0}) == 0,
            "required_device_bytes(0) is 0");

  // Two launches between mark()/rewind(): the launchers allocate nothing, so
  // used() must be unchanged.
  const std::uint64_t mark = tc->arena.mark();
  DevicePlane curr = AllocPlane(tc, 128, 128, "arena curr");
  DevicePlane cdef = AllocPlane(tc, 128, 128, "arena cdef");
  DevicePlane lr = AllocPlane(tc, 128, 128, "arena lr");
  std::vector<std::uint16_t> zeros(
      static_cast<std::size_t>(128) * 128, 0);
  UploadPlane(tc, curr, zeros, "arena upload curr");
  UploadPlane(tc, cdef, zeros, "arena upload cdef");
  UploadPlane(tc, lr, zeros, "arena upload lr");
  std::vector<std::uint8_t> mask(4 * 4, 1);
  ExpectStatus(tc, gdf::gdf_upload_use_gdf(tc->buffers, mask.data(), 16),
               Status::kOk, "arena upload mask");

  GdfFrameGrid frame;
  frame.src = MakeSource(curr, cdef, FullExtents(128, 128));
  frame.lr = lr.data;
  frame.lr_stride = lr.stride;
  frame.bit_depth = 8;
  frame.pix_scale = 1;
  frame.unit_size = 32;
  frame.cols = 4;
  frame.rows = 4;
  frame.width = 128;
  frame.height = 128;
  frame.mi_rows = 32;
  frame.index.frame_is_intra = 1;

  const std::uint64_t after_setup = tc->arena.used();
  const std::uint64_t launch_mark = tc->arena.mark();
  ExpectStatus(tc, gdf::gdf_apply_grid(frame, tc->buffers), Status::kOk,
               "arena launch 1");
  ExpectStatus(tc, gdf::gdf_apply_grid(frame, tc->buffers), Status::kOk,
               "arena launch 2");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "arena sync");
  tc->Check(tc->arena.used() == after_setup,
            "launchers do not allocate (used unchanged)");
  tc->Check(tc->arena.used() == launch_mark,
            "mark/rewind around two launches has no net growth");
  ExpectStatus(tc, gdf::gdf_apply_grid(frame, tc->buffers), Status::kOk,
               "arena launch 3 (peak stable)");
  CudaCheck(tc, cudaStreamSynchronize(nullptr), "arena sync 3");

  // Re-initializing the buffers at a new mark reuses the same bump region.
  const std::uint64_t used_before = tc->arena.used();
  const std::uint64_t init_mark = tc->arena.mark();
  GdfDeviceBuffers first;
  std::size_t first_required = 0;
  ExpectStatus(tc,
               gdf::gdf_buffers_init(tc->arena, GdfConfig{kTestSlots}, &first,
                                     &first_required),
               Status::kOk, "arena re-init");
  tc->Check(tc->arena.used() == used_before + first_required,
            "re-init consumes exactly required_device_bytes");
  (void)tc->arena.rewind(init_mark);
  GdfDeviceBuffers second;
  std::size_t second_required = 0;
  ExpectStatus(tc,
               gdf::gdf_buffers_init(tc->arena, GdfConfig{kTestSlots}, &second,
                                     &second_required),
               Status::kOk, "arena second init after rewind");
  tc->Check(second_required == first_required, "required bytes are stable");
  tc->Check(second.jobs == first.jobs && second.use_gdf == first.use_gdf &&
                second.grid_jobs == first.grid_jobs,
            "rewind reuses the same persistent addresses");
  tc->Check(tc->arena.used() == used_before + second_required,
            "no net growth after rewind and re-init");
  (void)tc->arena.rewind(init_mark);

  // One byte short: gdf_buffers_init must report kArenaExhausted with the
  // exact requirement and must not consume arena space.  This sizing probe is
  // deliberately the last allocation in the test, so the reported peak below
  // excludes the probe's filler block.
  tc->peak_before_probe = tc->arena.peak_used();
  const std::uint64_t short_mark = tc->arena.mark();
  const std::size_t need = gdf::required_device_bytes(GdfConfig{kTestSlots});
  const std::uint64_t entry_remaining = tc->arena.remaining();
  tc->Check(entry_remaining >= need, "arena has room for the short test");

  // Arena allocations are rounded up to 256 bytes, so allocate_checked()
  // cannot land on an arbitrary byte offset.  Fill most of the arena first,
  // then rewind to the exact bump offset that leaves need-1 bytes.  This makes
  // the "one byte short" condition exact instead of approximate.
  if (entry_remaining > need - 1) {
    void* filler = tc->arena.allocate(
        static_cast<std::size_t>(entry_remaining - (need - 1)));
    tc->Check(filler != nullptr, "arena filler allocation");
  }
  const std::uint64_t short_target =
      tc->arena.usable_capacity() - (need - 1);
  tc->Check(short_target <= tc->arena.used(), "short setup can rewind back");
  if (short_target <= tc->arena.used()) {
    tc->Check(tc->arena.rewind(short_target), "rewind to one byte short");
  }
  tc->Check(tc->arena.remaining() == need - 1,
            "arena prepared one byte short");
  const std::uint64_t used_before_short = tc->arena.used();

  GdfDeviceBuffers out;
  std::size_t reported = 0;
  std::string error;
  const Status status = gdf::gdf_buffers_init(
      tc->arena, GdfConfig{kTestSlots}, &out, &reported, &error);
  ExpectStatus(tc, status, Status::kArenaExhausted,
               "one-byte-short returns kArenaExhausted");
  tc->Check(reported == need, "short status reports required bytes");
  tc->Check(out.jobs == nullptr, "short init leaves output empty");
  tc->Check(error.find(std::to_string(need)) != std::string::npos,
            "short status message contains required bytes");
  tc->Check(tc->arena.used() == used_before_short,
            "failed init does not consume arena space");
  (void)tc->arena.rewind(short_mark);

  (void)tc->arena.rewind(mark);
  std::printf(
      "  arena: required=%zu B, peak(before sizing probe)=%llu B, "
      "capacity=%llu B\n",
      tc->required,
      static_cast<unsigned long long>(tc->peak_before_probe),
      static_cast<unsigned long long>(tc->arena.capacity()));
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
void TestTiming(TestContext* tc) {
  const int width = 1920;
  const int height = 1088;
  const int unit_size = 64;
  const std::uint64_t mark = tc->arena.mark();
  DevicePlane curr = AllocPlane(tc, width, height, "timing curr");
  DevicePlane cdef = AllocPlane(tc, width, height, "timing cdef");
  DevicePlane lr = AllocPlane(tc, width, height, "timing lr");
  const int cols = width / unit_size;
  const int rows = height / unit_size;
  std::vector<std::uint16_t> pixels(
      static_cast<std::size_t>(width) * height, 128);
  UploadPlane(tc, curr, pixels, "timing upload curr");
  UploadPlane(tc, cdef, pixels, "timing upload cdef");
  UploadPlane(tc, lr, pixels, "timing upload lr");
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(cols) * rows, 1);
  ExpectStatus(tc, gdf::gdf_upload_use_gdf(tc->buffers, mask.data(), cols * rows),
               Status::kOk, "timing mask");

  GdfFrameGrid frame;
  frame.src = MakeSource(curr, cdef, FullExtents(width, height));
  frame.lr = lr.data;
  frame.lr_stride = lr.stride;
  frame.bit_depth = 8;
  frame.pix_scale = 1;
  frame.unit_size = unit_size;
  frame.cols = cols;
  frame.rows = rows;
  frame.width = width;
  frame.height = height;
  frame.mi_rows = height / 4;
  frame.index.frame_is_intra = 1;
  frame.index.base_q_idx = 160;
  frame.index.bit_depth = 8;
  frame.index.gdf_pic_qc_idx = 1;

  for (int i = 0; i < 3; ++i) {
    ExpectStatus(tc, gdf::gdf_apply_grid(frame, tc->buffers), Status::kOk,
                 "timing warmup");
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CudaCheck(tc, cudaEventCreate(&start), "create start event");
  CudaCheck(tc, cudaEventCreate(&stop), "create stop event");
  const int iterations = 30;
  CudaCheck(tc, cudaEventRecord(start), "record start");
  for (int i = 0; i < iterations; ++i) {
    ExpectStatus(tc, gdf::gdf_apply_grid(frame, tc->buffers), Status::kOk,
                 "timing launch");
  }
  CudaCheck(tc, cudaEventRecord(stop), "record stop");
  CudaCheck(tc, cudaEventSynchronize(stop), "sync stop");
  float ms = 0.0f;
  CudaCheck(tc, cudaEventElapsedTime(&ms, start, stop), "elapsed");
  std::printf(
      "  timing: 1920x1088 GDF (%dx%d units of %d, 2 launches) %.4f ms/frame "
      "(%d iters)\n",
      cols, rows, unit_size, ms / static_cast<float>(iterations), iterations);
  CudaCheck(tc, cudaEventDestroy(start), "destroy start event");
  CudaCheck(tc, cudaEventDestroy(stop), "destroy stop event");
  (void)tc->arena.rewind(mark);
}

}  // namespace

int main() {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDevice failed: %s\n",
                 cudaGetErrorString(error));
    return 2;
  }
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(error));
    return 2;
  }
  std::printf("gdf CUDA parity test\n");
  std::printf("device: %s (sm_%d%d), %d SMs, %.1f GiB\n", properties.name,
              properties.major, properties.minor,
              properties.multiProcessorCount,
              static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  TestContext tc;
  const GdfConfig config{kTestSlots};
  tc.required = gdf::required_device_bytes(config);
  const std::size_t plane_bytes =
      static_cast<std::size_t>(1920) * 1088 * 3 * sizeof(std::uint16_t);
  const std::uint64_t capacity =
      static_cast<std::uint64_t>(tc.required) + plane_bytes + 4ull * 1024 * 1024;
  std::printf("arena request: required=%zu B + planes=%zu B + slack=4 MiB\n",
              tc.required, plane_bytes);
  std::string error_text;
  const Status init_status =
      tc.arena.init(capacity, Arena::kAlignment, &error_text);
  if (init_status != Status::kOk) {
    std::fprintf(stderr, "Arena::init failed: %s %s\n",
                 av2enc::cuda::status_name(init_status), error_text.c_str());
    return 2;
  }
  const Status buffers_status =
      gdf::gdf_buffers_init(tc.arena, config, &tc.buffers, &tc.required,
                            &error_text);
  if (buffers_status != Status::kOk) {
    std::fprintf(stderr, "gdf_buffers_init failed: %s %s\n",
                 av2enc::cuda::status_name(buffers_status), error_text.c_str());
    return 2;
  }

  TestDerivation(&tc);
  TestAllCombinations(&tc);
  TestHighBitDepthRange(&tc);
  TestLargeUnits(&tc);
  TestRandomBlocks(&tc);
  TestGrids(&tc);
  TestValidation(&tc);
  TestTiming(&tc);
  TestArena(&tc);

  std::printf("  parity samples compared: %lld, changed samples: %lld\n",
              tc.samples, tc.changed);
  std::printf("  arena peak (before sizing probe): %llu B (capacity %llu B)\n",
              static_cast<unsigned long long>(tc.peak_before_probe),
              static_cast<unsigned long long>(tc.arena.capacity()));

  tc.arena.release();
  if (tc.failures == 0) {
    std::printf("PASS: %lld checks, 0 failures\n", tc.checks);
    return 0;
  }
  std::printf("FAIL: %lld checks, %d failures\n", tc.checks, tc.failures);
  return 1;
}
