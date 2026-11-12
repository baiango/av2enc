// CPU-vs-GPU bit-exactness tests for the advanced inter kernels.
//
// Every test builds a randomised job on the host, runs the CPU reference from
// src/encoder/inter/advanced/, runs the CUDA kernel on the same inputs, and
// requires byte-for-byte equality. There is no tolerance.
//
// Build/run: ./build_advanced_test.sh   (needs nvcc + an sm_86 device)
//           ./build_advanced_test.sh --no-run

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "advanced.h"

namespace {

using namespace av2enc::inter::adv;
namespace gpu = av2enc::cuda::advinter;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)
#define CUDA_OK(expr)                                                                \
  do {                                                                               \
    const cudaError_t err = (expr);                                                  \
    ++g_checks;                                                                      \
    if (err != cudaSuccess) {                                                        \
      ++g_failures;                                                                  \
      std::printf("CUDA error %s at line %d: %s\n", cudaGetErrorName(err), __LINE__, \
                  cudaGetErrorString(err));                                          \
    }                                                                                \
  } while (0)

std::uint32_t g_rng = 0x9e3779b9u;

std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

int random_range(int low, int high) {
  return low + static_cast<int>(next_random() % static_cast<std::uint32_t>(high - low + 1));
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  bool allocate(std::size_t n) {
    count = n;
    if (n == 0) {
      return true;
    }
    return cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)) == cudaSuccess;
  }

  bool upload(const std::vector<T>& host) {
    if (host.size() > count) {
      return false;
    }
    return cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
  }

  bool download(std::vector<T>* host) const {
    host->resize(count);
    return cudaMemcpy(host->data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
};

int bsize_for(int w, int h) {
  if (w == h) {
    switch (w) {
      case 8:
        return tables::kBlock8x8;
      case 16:
        return tables::kBlock16x16;
      case 32:
        return 9;
      case 64:
        return tables::kBlock64x64;
      default:
        return tables::kBlock8x8;
    }
  }
  if (w == 8 && h == 32) {
    return 21;
  }
  if (w == 32 && h == 8) {
    return 22;
  }
  return tables::kBlock8x8;
}

// ---------------------------------------------------------------------------
// Compound / inter-intra
// ---------------------------------------------------------------------------

void test_compound() {
  const int sizes[][2] = {{8, 8}, {16, 16}, {32, 32}, {64, 64}, {8, 32}, {32, 8}, {4, 4}, {16, 8}};
  for (const auto& size : sizes) {
    const int w = size[0];
    const int h = size[1];
    const int bsize = bsize_for(w, h);
    std::vector<std::int16_t> p0(static_cast<std::size_t>(w) * h);
    std::vector<std::int16_t> p1(static_cast<std::size_t>(w) * h);
    for (int i = 0; i < w * h; ++i) {
      p0[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>(random_range(0, 255) * 16 + random_range(-24, 24));
      p1[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>(random_range(0, 255) * 16 + random_range(-24, 24));
    }
    std::vector<std::uint8_t> host_dst(static_cast<std::size_t>(w) * h);
    std::vector<std::uint8_t> host_mask(static_cast<std::size_t>(w) * h);

    DeviceBuffer<std::int16_t> d_p0;
    DeviceBuffer<std::int16_t> d_p1;
    DeviceBuffer<std::uint8_t> d_dst;
    DeviceBuffer<std::uint8_t> d_mask;
    EXPECT(d_p0.allocate(p0.size()));
    EXPECT(d_p1.allocate(p1.size()));
    EXPECT(d_dst.allocate(host_dst.size()));
    EXPECT(d_mask.allocate(host_mask.size()));
    EXPECT(d_p0.upload(p0));
    EXPECT(d_p1.upload(p1));

    std::vector<CompoundJob> jobs;
    std::vector<std::string> names;
    auto add_job = [&](CompoundType type, int cw, int index, int sign, int mask_type, bool imp,
                       const char* name) {
      CompoundJob job{};
      job.rect = {0, 0, w, h};
      job.bsize = bsize;
      job.p0 = d_p0.ptr;
      job.p0_stride = w;
      job.p1 = d_p1.ptr;
      job.p1_stride = w;
      job.dst = d_dst.ptr;
      job.dst_stride = w;
      job.mask_out = d_mask.ptr;
      job.mask_out_stride = w;
      job.type = type;
      job.cwp_weight = cw;
      job.wedge_index = index;
      job.wedge_sign = sign;
      job.mask_type = mask_type;
      job.imp_mask_eligible = imp ? 1 : 0;
      job.mi_cols = w;
      job.mi_rows = h;
      job.ref_start_x[0] = -2;
      job.ref_start_x[1] = w - 2;
      jobs.push_back(job);
      names.emplace_back(name);
    };

    add_job(CompoundType::kAverage, 8, 0, 0, 0, false, "average8");
    add_job(CompoundType::kAverage, 12, 0, 0, 0, false, "average12");
    add_job(CompoundType::kAverage, -4, 0, 0, 0, false, "average-4");
    add_job(CompoundType::kAverage, 8, 0, 0, 0, true, "implicit");
    if (bsize_has_wedge(bsize)) {
      add_job(CompoundType::kWedge, 8, 0, 0, 0, false, "wedge0");
      add_job(CompoundType::kWedge, 8, 33, 1, 0, false, "wedge33");
      add_job(CompoundType::kWedge, 8, 67, 1, 0, false, "wedge67");
    }
    add_job(CompoundType::kDiffWtd, 8, 0, 0, 0, false, "diffwtd0");
    add_job(CompoundType::kDiffWtd, 8, 0, 0, 1, false, "diffwtd1");

    for (std::size_t k = 0; k < jobs.size(); ++k) {
      CompoundJob host_job = jobs[k];
      host_job.p0 = p0.data();
      host_job.p1 = p1.data();
      host_job.dst = host_dst.data();
      host_job.mask_out = host_mask.data();
      run_compound_job(host_job);

      std::vector<std::uint8_t> zeros(host_dst.size(), 0);
      EXPECT(d_dst.upload(zeros));
      EXPECT(d_mask.upload(zeros));
      CUDA_OK(gpu::compound_one(jobs[k]));
      CUDA_OK(cudaDeviceSynchronize());

      std::vector<std::uint8_t> gpu_dst;
      std::vector<std::uint8_t> gpu_mask;
      EXPECT(d_dst.download(&gpu_dst));
      EXPECT(d_mask.download(&gpu_mask));
      EXPECT(d_dst.count == host_dst.size());
      std::size_t diff = 0;
      for (std::size_t i = 0; i < host_dst.size(); ++i) {
        if (gpu_dst[i] != host_dst[i]) {
          ++diff;
        }
      }
      expect(diff == 0, names[k].c_str(), __LINE__);
      if (host_job.type == CompoundType::kWedge || host_job.type == CompoundType::kDiffWtd) {
        std::size_t mask_diff = 0;
        for (std::size_t i = 0; i < host_mask.size(); ++i) {
          if (gpu_mask[i] != host_mask[i]) {
            ++mask_diff;
          }
        }
        EXPECT(mask_diff == 0);
      }
    }
  }
}

void test_compound_inter_intra() {
  const int w = 16;
  const int h = 16;
  const int bsize = tables::kBlock16x16;
  std::vector<std::uint8_t> inter(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> intra(static_cast<std::size_t>(w) * h);
  for (int i = 0; i < w * h; ++i) {
    inter[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
    intra[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  DeviceBuffer<std::uint8_t> d_dst;
  DeviceBuffer<std::uint8_t> d_intra;
  EXPECT(d_dst.allocate(inter.size()));
  EXPECT(d_intra.allocate(intra.size()));
  EXPECT(d_intra.upload(intra));

  for (int wedge = 0; wedge < 2; ++wedge) {
    CompoundJob job{};
    job.rect = {0, 0, w, h};
    job.bsize = bsize;
    job.is_inter_intra = 1;
    job.interintra_mode = InterIntraMode::kSmooth;
    job.type = wedge ? CompoundType::kWedge : CompoundType::kIntra;
    job.wedge_index = 12;
    job.wedge_sign = 0;
    job.intra = d_intra.ptr;
    job.intra_stride = w;
    job.dst = d_dst.ptr;
    job.dst_stride = w;
    EXPECT(d_dst.upload(inter));

    CompoundJob host_job = job;
    host_job.intra = intra.data();
    host_job.dst = inter.data();
    run_compound_job(host_job);

    CUDA_OK(gpu::compound_one(job));
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> gpu_dst;
    EXPECT(d_dst.download(&gpu_dst));
    EXPECT(gpu_dst == inter);
  }
}

// ---------------------------------------------------------------------------
// Warp
// ---------------------------------------------------------------------------

WarpBlockGeometry make_geometry_local() {
  WarpBlockGeometry g{};
  g.sub_x = 0;
  g.sub_y = 0;
  g.mi_col = 4;
  g.mi_row = 4;
  g.bw4 = 8;
  g.bh4 = 6;
  g.mi_cols = 32;
  g.mi_rows = 32;
  g.ref_w = 128;
  g.ref_h = 128;
  g.frame_w = 128;
  g.frame_h = 128;
  return g;
}

struct RefFixture {
  int w = 128;
  int h = 128;
  std::vector<std::uint8_t> data;
  DeviceBuffer<std::uint8_t> device;
  PlaneView view{};

  RefFixture() {
    data.resize(static_cast<std::size_t>(w) * h);
    for (int i = 0; i < w * h; ++i) {
      data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
    }
    device.allocate(data.size());
    device.upload(data);
    view.data = device.ptr;
    view.stride = w;
    view.max_x = w - 1;
    view.max_y = h - 1;
  }
};

void test_warp() {
  RefFixture fixture;
  const int block_w = 32;
  const int block_h = 24;
  const std::size_t count = static_cast<std::size_t>(block_w) * block_h;

  for (int kind = 0; kind < 3; ++kind) {
    WarpParams params{};
    if (kind == 1) {
      params.p[0] = 3 << 16;
      params.p[1] = 2 << 16;
    } else if (kind == 2) {
      params.p[2] = 65536 + 1200;
      params.p[5] = 65536 - 900;
      params.p[3] = 700;
      params.p[0] = -2 << 16;
    }
    WarpBlockGeometry geom = make_geometry_local();
    geom.rect = {16, 16, block_w, block_h};
    geom.dst_x = 0;
    geom.dst_y = 0;

    for (int compound = 0; compound < 2; ++compound) {
      std::vector<std::int16_t> cpu(count, 0);
      EXPECT(warp_predict(params, fixture.view, geom, compound != 0, cpu.data(), block_w));

      DeviceBuffer<std::int16_t> d_dst;
      EXPECT(d_dst.allocate(count));
      std::vector<std::int16_t> zeros(count, 0);
      EXPECT(d_dst.upload(zeros));

      gpu::WarpJob job{};
      job.params = params;
      job.ref = fixture.view;
      job.geom = geom;
      job.compound = compound;
      job.dst = d_dst.ptr;
      job.dst_stride = block_w;
      CUDA_OK(gpu::warp_batch(&job, 1));
      std::vector<std::int16_t> gpu_dst;
      EXPECT(d_dst.download(&gpu_dst));
      EXPECT(gpu_dst == cpu);
    }
  }
}

void test_warp_ext() {
  RefFixture fixture;
  const int block_w = 20;
  const int block_h = 20;
  const std::size_t count = static_cast<std::size_t>(block_w) * block_h;

  for (int trial = 0; trial < 10; ++trial) {
    WarpParams params{};
    if (trial == 0) {
      // Identity.
    } else if (trial == 1) {
      params.p[0] = 2 << 16;
      params.p[1] = -3 << 16;
    } else if (trial == 2) {
      params.p[3] = 2 << 16;  // invalid shear: ext warp must still predict
    } else {
      params.p[2] = 65536 + random_range(-2000, 2000);
      params.p[5] = 65536 + random_range(-2000, 2000);
      params.p[3] = random_range(-1200, 1200);
      params.p[4] = random_range(-1200, 1200);
      params.p[0] = random_range(-4, 4) << 16;
      params.p[1] = random_range(-4, 4) << 16;
    }

    WarpBlockGeometry geom = make_geometry_local();
    geom.rect = {8, 8, block_w, block_h};
    geom.dst_x = 0;
    geom.dst_y = 0;

    for (int compound = 0; compound < 2; ++compound) {
      std::vector<std::int16_t> cpu(count, 0);
      EXPECT(warp_predict_ext(params, fixture.view, geom, compound != 0, cpu.data(), block_w));
      DeviceBuffer<std::int16_t> d_dst;
      EXPECT(d_dst.allocate(count));
      std::vector<std::int16_t> zeros(count, 0);
      EXPECT(d_dst.upload(zeros));
      gpu::WarpJob job{};
      job.params = params;
      job.ref = fixture.view;
      job.geom = geom;
      job.compound = compound;
      job.dst = d_dst.ptr;
      job.dst_stride = block_w;
      CUDA_OK(gpu::ext_warp_batch(&job, 1));
      std::vector<std::int16_t> gpu_dst;
      EXPECT(d_dst.download(&gpu_dst));
      EXPECT(gpu_dst == cpu);
    }
  }
}

void test_compound_chroma() {
  const int luma_w = 16;
  const int luma_h = 16;
  const int chroma_w = 8;
  const int chroma_h = 8;
  std::vector<std::int16_t> lp0(static_cast<std::size_t>(luma_w) * luma_h);
  std::vector<std::int16_t> lp1(lp0.size());
  std::vector<std::int16_t> cp0(static_cast<std::size_t>(chroma_w) * chroma_h);
  std::vector<std::int16_t> cp1(cp0.size());
  for (std::size_t i = 0; i < lp0.size(); ++i) {
    lp0[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
    lp1[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
  }
  for (std::size_t i = 0; i < cp0.size(); ++i) {
    cp0[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
    cp1[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
  }
  std::vector<std::uint8_t> luma_dst(lp0.size(), 0);
  std::vector<std::uint8_t> luma_mask(lp0.size(), 0);
  std::vector<std::uint8_t> chroma_dst(cp0.size(), 0);

  DeviceBuffer<std::int16_t> d_lp0;
  DeviceBuffer<std::int16_t> d_lp1;
  DeviceBuffer<std::int16_t> d_cp0;
  DeviceBuffer<std::int16_t> d_cp1;
  DeviceBuffer<std::uint8_t> d_ldst;
  DeviceBuffer<std::uint8_t> d_mask;
  DeviceBuffer<std::uint8_t> d_cdst;
  EXPECT(d_lp0.allocate(lp0.size()));
  EXPECT(d_lp1.allocate(lp1.size()));
  EXPECT(d_cp0.allocate(cp0.size()));
  EXPECT(d_cp1.allocate(cp1.size()));
  EXPECT(d_ldst.allocate(luma_dst.size()));
  EXPECT(d_mask.allocate(luma_mask.size()));
  EXPECT(d_cdst.allocate(chroma_dst.size()));
  EXPECT(d_lp0.upload(lp0));
  EXPECT(d_lp1.upload(lp1));
  EXPECT(d_cp0.upload(cp0));
  EXPECT(d_cp1.upload(cp1));

  CompoundJob luma{};
  luma.rect = {0, 0, luma_w, luma_h};
  luma.bsize = tables::kBlock16x16;
  luma.p0 = d_lp0.ptr;
  luma.p0_stride = luma_w;
  luma.p1 = d_lp1.ptr;
  luma.p1_stride = luma_w;
  luma.dst = d_ldst.ptr;
  luma.dst_stride = luma_w;
  luma.mask_out = d_mask.ptr;
  luma.mask_out_stride = luma_w;
  luma.type = CompoundType::kWedge;
  luma.wedge_index = 42;
  luma.wedge_sign = 1;

  CompoundJob chroma{};
  chroma.plane = 1;
  chroma.sub_x = 1;
  chroma.sub_y = 1;
  chroma.rect = {0, 0, chroma_w, chroma_h};
  chroma.bsize = tables::kBlock16x16;
  chroma.p0 = d_cp0.ptr;
  chroma.p0_stride = chroma_w;
  chroma.p1 = d_cp1.ptr;
  chroma.p1_stride = chroma_w;
  chroma.dst = d_cdst.ptr;
  chroma.dst_stride = chroma_w;
  chroma.luma_mask = d_mask.ptr;
  chroma.luma_mask_stride = luma_w;
  chroma.type = CompoundType::kWedge;
  chroma.wedge_index = 42;
  chroma.wedge_sign = 1;

  // CPU reference: luma first (produces the mask), then chroma.
  CompoundJob host_luma = luma;
  host_luma.p0 = lp0.data();
  host_luma.p1 = lp1.data();
  host_luma.dst = luma_dst.data();
  host_luma.mask_out = luma_mask.data();
  run_compound_job(host_luma);
  CompoundJob host_chroma = chroma;
  host_chroma.p0 = cp0.data();
  host_chroma.p1 = cp1.data();
  host_chroma.dst = chroma_dst.data();
  host_chroma.luma_mask = luma_mask.data();
  run_compound_job(host_chroma);

  CUDA_OK(gpu::compound_one(luma));
  CUDA_OK(gpu::compound_one(chroma));
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> gpu_ldst;
  std::vector<std::uint8_t> gpu_mask;
  std::vector<std::uint8_t> gpu_cdst;
  EXPECT(d_ldst.download(&gpu_ldst));
  EXPECT(d_mask.download(&gpu_mask));
  EXPECT(d_cdst.download(&gpu_cdst));
  EXPECT(gpu_ldst == luma_dst);
  EXPECT(gpu_mask == luma_mask);
  EXPECT(gpu_cdst == chroma_dst);
}

void test_warp_chroma() {
  const int ref_w = 64;
  const int ref_h = 64;
  std::vector<std::uint8_t> ref_data(static_cast<std::size_t>(ref_w) * ref_h);
  for (int i = 0; i < ref_w * ref_h; ++i) {
    ref_data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  DeviceBuffer<std::uint8_t> d_ref;
  EXPECT(d_ref.allocate(ref_data.size()));
  EXPECT(d_ref.upload(ref_data));
  PlaneView ref{};
  ref.data = d_ref.ptr;
  ref.stride = ref_w;
  ref.max_x = ref_w - 1;
  ref.max_y = ref_h - 1;

  WarpBlockGeometry geom = make_geometry_local();
  geom.plane = 1;
  geom.sub_x = 1;
  geom.sub_y = 1;
  geom.rect = {8, 8, 16, 16};
  geom.dst_x = 0;
  geom.dst_y = 0;
  geom.ref_w = 128;
  geom.ref_h = 128;
  geom.frame_w = 128;
  geom.frame_h = 128;

  WarpParams params{};
  params.p[2] = 65536 + 500;
  params.p[0] = 2 << 16;
  const std::size_t count = 16 * 16;
  for (int kind = 0; kind < 2; ++kind) {
    for (int compound = 0; compound < 2; ++compound) {
      std::vector<std::int16_t> cpu(count, 0);
      if (kind == 0) {
        EXPECT(warp_predict(params, ref, geom, compound != 0, cpu.data(), 16));
      } else {
        EXPECT(warp_predict_ext(params, ref, geom, compound != 0, cpu.data(), 16));
      }
      DeviceBuffer<std::int16_t> d_dst;
      EXPECT(d_dst.allocate(count));
      std::vector<std::int16_t> zeros(count, 0);
      EXPECT(d_dst.upload(zeros));
      gpu::WarpJob job{};
      job.params = params;
      job.ref = ref;
      job.geom = geom;
      job.compound = compound;
      job.dst = d_dst.ptr;
      job.dst_stride = 16;
      CUDA_OK(kind == 0 ? gpu::warp_batch(&job, 1) : gpu::ext_warp_batch(&job, 1));
      std::vector<std::int16_t> gpu_dst;
      EXPECT(d_dst.download(&gpu_dst));
      EXPECT(gpu_dst == cpu);
    }
  }
}

void test_warp_ext_scaled() {
  // Half-size reference: the extended-warp scaled branch.
  const int ref_w = 64;
  const int ref_h = 64;
  std::vector<std::uint8_t> ref_data(static_cast<std::size_t>(ref_w) * ref_h);
  for (int i = 0; i < ref_w * ref_h; ++i) {
    ref_data[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  DeviceBuffer<std::uint8_t> d_ref;
  EXPECT(d_ref.allocate(ref_data.size()));
  EXPECT(d_ref.upload(ref_data));
  PlaneView ref{};
  ref.data = d_ref.ptr;
  ref.stride = ref_w;
  ref.max_x = ref_w - 1;
  ref.max_y = ref_h - 1;

  const int block_w = 16;
  const int block_h = 16;
  const std::size_t count = static_cast<std::size_t>(block_w) * block_h;

  WarpParams params{};
  params.p[2] = 65536 + 800;
  params.p[5] = 65536 - 600;
  params.p[3] = 400;
  params.p[0] = -1 << 16;

  WarpBlockGeometry geom = make_geometry_local();
  geom.rect = {32, 32, block_w, block_h};
  geom.dst_x = 0;
  geom.dst_y = 0;
  geom.ref_w = ref_w;
  geom.ref_h = ref_h;
  geom.frame_w = 128;
  geom.frame_h = 128;
  // The MV clamp context must cover the frame, not the reference.
  geom.mi_cols = 32;
  geom.mi_rows = 32;

  for (int compound = 0; compound < 2; ++compound) {
    std::vector<std::int16_t> cpu(count, 0);
    EXPECT(warp_predict_ext(params, ref, geom, compound != 0, cpu.data(), block_w));
    DeviceBuffer<std::int16_t> d_dst;
    EXPECT(d_dst.allocate(count));
    std::vector<std::int16_t> zeros(count, 0);
    EXPECT(d_dst.upload(zeros));
    gpu::WarpJob job{};
    job.params = params;
    job.ref = ref;
    job.geom = geom;
    job.compound = compound;
    job.dst = d_dst.ptr;
    job.dst_stride = block_w;
    CUDA_OK(gpu::ext_warp_batch(&job, 1));
    std::vector<std::int16_t> gpu_dst;
    EXPECT(d_dst.download(&gpu_dst));
    EXPECT(gpu_dst == cpu);
  }
}

// ---------------------------------------------------------------------------
// BAWP
// ---------------------------------------------------------------------------

void test_bawp() {
  const int w = 128;
  const int h = 128;
  std::vector<std::uint8_t> cur(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> prev(static_cast<std::size_t>(w) * h);
  for (int i = 0; i < w * h; ++i) {
    cur[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(16, 240));
    prev[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(16, 240));
  }
  DeviceBuffer<std::uint8_t> d_cur;
  DeviceBuffer<std::uint8_t> d_prev;
  DeviceBuffer<std::uint8_t> d_dst;
  DeviceBuffer<BawpFit> d_fit;
  EXPECT(d_cur.allocate(cur.size()));
  EXPECT(d_prev.allocate(prev.size()));
  EXPECT(d_dst.allocate(cur.size()));
  EXPECT(d_fit.allocate(1));
  EXPECT(d_cur.upload(cur));
  EXPECT(d_prev.upload(prev));

  for (int trial = 0; trial < 24; ++trial) {
    BawpJob job{};
    job.ctx.plane = trial % 5 == 0 ? 1 : 0;
    job.ctx.sub_x = job.ctx.plane ? 1 : 0;
    job.ctx.sub_y = job.ctx.plane ? 1 : 0;
    job.ctx.mi_cols = w / 4;
    job.ctx.mi_rows = h / 4;
    job.ctx.mi_col = random_range(2, w / 8 - 3);
    job.ctx.mi_row = random_range(2, h / 8 - 3);
    job.ctx.avail_u = (next_random() & 1u) != 0u;
    job.ctx.avail_l = (next_random() & 1u) != 0u;
    job.rect.x = job.ctx.mi_col * 4;
    job.rect.y = job.ctx.mi_row * 4;
    job.rect.w = 4 * random_range(1, 4);
    job.rect.h = 4 * random_range(1, 4);
    job.mv[0] = random_range(-16, 16);
    job.mv[1] = random_range(-16, 16);
    job.bawp_alpha = random_range(180, 330);
    job.explicit_bawp = (next_random() % 4u) == 0u;
    job.explicit_bawp_scale = (next_random() & 1u) != 0u;
    job.list_index = random_range(0, 2);
    job.first_ref_dist = random_range(0, 8);
    job.morph_pred = false;

    // Host reference: copy the prediction into a work buffer first.
    std::vector<std::uint8_t> work = cur;
    for (int i = 0; i < job.rect.h; ++i) {
      for (int j = 0; j < job.rect.w; ++j) {
        work[static_cast<std::size_t>(job.rect.y + i) * w + job.rect.x + j] =
            prev[static_cast<std::size_t>(job.rect.y + i) * w + job.rect.x + j];
      }
    }
    BawpJob host = job;
    host.cur.data = work.data();
    host.cur.stride = w;
    host.cur.max_x = w - 1;
    host.cur.max_y = h - 1;
    host.prev.data = prev.data();
    host.prev.stride = w;
    host.prev.max_x = w - 1;
    host.prev.max_y = h - 1;
    host.dst.data = work.data();
    host.dst.stride = w;
    host.dst.max_x = w - 1;
    host.dst.max_y = h - 1;
    BawpFit host_fit{};
    const std::vector<std::uint8_t> work_before = work;
    EXPECT(bawp_apply_block(host, &host_fit));
    // The device job must see the same pre-adjustment reconstruction and
    // prediction as the CPU reference.
    EXPECT(d_cur.upload(work_before));
    EXPECT(d_dst.upload(work_before));
    gpu::BawpGpuJob gpu_job{};
    gpu_job.job = job;
    gpu_job.job.cur.data = d_cur.ptr;
    gpu_job.job.cur.stride = w;
    gpu_job.job.cur.max_x = w - 1;
    gpu_job.job.cur.max_y = h - 1;
    gpu_job.job.prev.data = d_prev.ptr;
    gpu_job.job.prev.stride = w;
    gpu_job.job.prev.max_x = w - 1;
    gpu_job.job.prev.max_y = h - 1;
    gpu_job.job.dst.data = d_dst.ptr;
    gpu_job.job.dst.stride = w;
    gpu_job.job.dst.max_x = w - 1;
    gpu_job.job.dst.max_y = h - 1;
    gpu_job.fit_out = d_fit.ptr;
    CUDA_OK(gpu::bawp_batch(&gpu_job, 1));

    std::vector<std::uint8_t> gpu_dst;
    std::vector<BawpFit> gpu_fit;
    EXPECT(d_dst.download(&gpu_dst));
    EXPECT(d_fit.download(&gpu_fit));
    EXPECT(gpu_fit.size() == 1);
    EXPECT(gpu_fit[0].alpha == host_fit.alpha);
    EXPECT(gpu_fit[0].beta == host_fit.beta);
    EXPECT(gpu_fit[0].count == host_fit.count);
    EXPECT(gpu_dst == work);
  }
}

// ---------------------------------------------------------------------------
// OBMC
// ---------------------------------------------------------------------------

void test_obmc() {
  const int w = 24;
  const int h = 20;
  std::vector<std::uint8_t> dst(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> above(static_cast<std::size_t>(w) * h);
  std::vector<std::uint8_t> left(static_cast<std::size_t>(w) * h);
  for (int i = 0; i < w * h; ++i) {
    dst[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
    above[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
    left[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  DeviceBuffer<std::uint8_t> d_dst;
  DeviceBuffer<std::uint8_t> d_above;
  DeviceBuffer<std::uint8_t> d_left;
  EXPECT(d_dst.allocate(dst.size()));
  EXPECT(d_above.allocate(above.size()));
  EXPECT(d_left.allocate(left.size()));
  EXPECT(d_above.upload(above));
  EXPECT(d_left.upload(left));

  for (int overlap : {0, 2, 4, 8}) {
    std::vector<std::uint8_t> host_dst = dst;
    ObmcConfig host_config{};
    host_config.rect = {2, 2, w - 4, h - 4};
    host_config.overlap = overlap;
    host_config.dst = host_dst.data();
    host_config.dst_stride = w;
    ObmcNeighbourPred host_above{above.data(), w};
    ObmcNeighbourPred host_left{left.data(), w};
    EXPECT(obmc_apply(host_config, &host_above, &host_left));

    EXPECT(d_dst.upload(dst));
    gpu::ObmcGpuJob job{};
    job.config = host_config;
    job.config.dst = d_dst.ptr;
    job.above = d_above.ptr;
    job.above_stride = w;
    job.left = d_left.ptr;
    job.left_stride = w;
    CUDA_OK(gpu::obmc_batch(&job, 1));
    std::vector<std::uint8_t> gpu_dst;
    EXPECT(d_dst.download(&gpu_dst));
    EXPECT(gpu_dst == host_dst);
  }
}

// ---------------------------------------------------------------------------
// Batch path + timing
// ---------------------------------------------------------------------------

void test_batch() {
  const int w = 32;
  const int h = 32;
  const int jobs_n = 8;
  std::vector<std::int16_t> p0(static_cast<std::size_t>(w) * h * jobs_n);
  std::vector<std::int16_t> p1(p0.size());
  for (std::size_t i = 0; i < p0.size(); ++i) {
    p0[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
    p1[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
  }
  DeviceBuffer<std::int16_t> d_p0;
  DeviceBuffer<std::int16_t> d_p1;
  DeviceBuffer<std::uint8_t> d_dst;
  std::vector<std::uint8_t> host_dst(static_cast<std::size_t>(w) * h * jobs_n);
  EXPECT(d_p0.allocate(p0.size()));
  EXPECT(d_p1.allocate(p1.size()));
  EXPECT(d_dst.allocate(host_dst.size()));
  EXPECT(d_p0.upload(p0));
  EXPECT(d_p1.upload(p1));

  std::vector<CompoundJob> jobs(static_cast<std::size_t>(jobs_n));
  for (int k = 0; k < jobs_n; ++k) {
    jobs[static_cast<std::size_t>(k)].rect = {0, 0, w, h};
    jobs[static_cast<std::size_t>(k)].bsize = 9;
    jobs[static_cast<std::size_t>(k)].p0 = d_p0.ptr + k * w * h;
    jobs[static_cast<std::size_t>(k)].p0_stride = w;
    jobs[static_cast<std::size_t>(k)].p1 = d_p1.ptr + k * w * h;
    jobs[static_cast<std::size_t>(k)].p1_stride = w;
    jobs[static_cast<std::size_t>(k)].dst = d_dst.ptr + k * w * h;
    jobs[static_cast<std::size_t>(k)].dst_stride = w;
    jobs[static_cast<std::size_t>(k)].type = CompoundType::kAverage;
    jobs[static_cast<std::size_t>(k)].cwp_weight = 8 + (k % 5) * 2;
  }
  DeviceBuffer<CompoundJob> d_jobs;
  EXPECT(d_jobs.allocate(jobs.size()));
  EXPECT(d_jobs.upload(jobs));
  CUDA_OK(gpu::compound_batch(d_jobs.ptr, jobs_n));
  CUDA_OK(cudaDeviceSynchronize());

  std::vector<std::uint8_t> gpu_dst(host_dst.size());
  EXPECT(cudaMemcpy(gpu_dst.data(), d_dst.ptr, host_dst.size() * sizeof(std::uint8_t),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int k = 0; k < jobs_n; ++k) {
    CompoundJob host = jobs[static_cast<std::size_t>(k)];
    host.p0 = p0.data() + k * w * h;
    host.p1 = p1.data() + k * w * h;
    host.dst = host_dst.data() + k * w * h;
    run_compound_job(host);
  }
  EXPECT(gpu_dst == host_dst);
}

void benchmark() {
  constexpr int kIters = 200;
  const int w = 64;
  const int h = 64;
  std::vector<std::int16_t> p0(static_cast<std::size_t>(w) * h);
  std::vector<std::int16_t> p1(p0.size());
  for (std::size_t i = 0; i < p0.size(); ++i) {
    p0[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
    p1[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
  }
  DeviceBuffer<std::int16_t> d_p0;
  DeviceBuffer<std::int16_t> d_p1;
  DeviceBuffer<std::uint8_t> d_dst;
  if (!d_p0.allocate(p0.size()) || !d_p1.allocate(p1.size()) || !d_dst.allocate(p0.size())) {
    return;
  }
  d_p0.upload(p0);
  d_p1.upload(p1);
  CompoundJob job{};
  job.rect = {0, 0, w, h};
  job.bsize = tables::kBlock64x64;
  job.p0 = d_p0.ptr;
  job.p0_stride = w;
  job.p1 = d_p1.ptr;
  job.p1_stride = w;
  job.dst = d_dst.ptr;
  job.dst_stride = w;
  job.type = CompoundType::kWedge;
  job.wedge_index = 33;
  job.wedge_sign = 1;

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Warm-up.
  for (int i = 0; i < 10; ++i) {
    gpu::compound_one(job);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(start, nullptr);
  for (int i = 0; i < kIters; ++i) {
    gpu::compound_one(job);
  }
  cudaEventRecord(stop, nullptr);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf(
      "timing: compound wedge 64x64 (%d blocks/sample pairs) %.3f ms total, "
      "%.3f us/block, %.2f Gsample/s\n",
      kIters, ms, ms * 1000.0 / kIters, (static_cast<double>(kIters) * w * h) / (ms * 1e6));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::printf("advanced_cuda_test: no CUDA device available\n");
    return 77;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::printf("device: %s (cc %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);

  test_compound();
  test_compound_inter_intra();
  test_compound_chroma();
  test_warp();
  test_warp_ext();
  test_warp_chroma();
  test_warp_ext_scaled();
  test_bawp();
  test_obmc();
  test_batch();

  if (g_failures == 0) {
    std::printf("advanced_cuda_test: %d checks passed\n", g_checks);
    benchmark();
    return 0;
  }
  std::printf("advanced_cuda_test: %d/%d checks FAILED\n", g_failures, g_checks);
  return 1;
}
