// CPU-vs-GPU bit-exactness tests for src/cuda/compound_full.
//
// Every fixture is allocated once from `av2enc::cuda::Arena`; the test performs
// no cudaMalloc/cudaFree/cudaMallocHost. The workspace is sized from
// `required_device_bytes(config)` plus slack, a one-byte-short arena is
// rejected with `Status::kArenaExhausted`, and every launch is bracketed by
// `mark()`/`rewind()` with a no-net-growth assertion.
//
// Oracles:
//   * 8-bit blend  -> inter::adv::run_compound_job()
//   * 10-bit blend -> compound_full_ref::blend_job() (independent literal)
//   * OPFL         -> inter::temporal::opfl_compute() (8-bit);
//                     10-bit GPU vs the shared header executed on the host
//   * refine-MV    -> inter::temporal::{search_refinemv, refine_predict_block}
//   * skip mode    -> av2enc::video::multiref::{build_compound_prediction,
//                                   apply_compound_prediction} (8-bit);
//                     compound_full_ref::skip_job() (10-bit)
//
// Build/run: ./build_gpu_test.sh          (needs nvcc + a CUDA device)
//           ./build_gpu_test.sh --no-run

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "compound_full.h"
#include "compound_full_device.cuh"
#include "compound_full_cpu_oracle.h"
#include "compound_full_reference.h"

#include "codec/inter/advanced/compound.h"
#include "codec/inter/temporal/opfl.h"
#include "codec/inter/temporal/refinemv.h"
#include "codec/inter/multiref/compound_predict.h"

#if !defined(__CUDA_ARCH__)
namespace cf = av2enc::cuda::compound_full;
namespace ref = compound_full_ref;
namespace adv = av2enc::inter::adv;
namespace tmc = av2enc::inter::mc;
namespace temporal = av2enc::inter::temporal;

namespace {

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)
#define CUDA_OK(expr)                                                          \
  do {                                                                         \
    const cudaError_t err_ = (expr);                                           \
    ++g_checks;                                                                \
    if (err_ != cudaSuccess) {                                                 \
      ++g_failures;                                                            \
      std::printf("CUDA error %s at line %d: %s\n", cudaGetErrorName(err_),    \
                  __LINE__, cudaGetErrorString(err_));                         \
    }                                                                          \
  } while (0)

std::uint32_t g_rng = 0xc0ffee11u;
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

int bsize_for(int w, int h) {
  // Function-local copy of the §9.2 block dimensions: host test code must not
  // reference a __device__ table from the device compilation pass.
  constexpr int kDims[29][2] = {
      {4, 4},     {4, 8},     {8, 4},     {8, 8},     {8, 16},
      {16, 8},    {16, 16},   {16, 32},   {32, 16},   {32, 32},
      {32, 64},   {64, 32},   {64, 64},   {64, 128},  {128, 64},
      {128, 128}, {128, 256}, {256, 128}, {256, 256}, {4, 16},
      {16, 4},    {8, 32},    {32, 8},    {16, 64},   {64, 16},
      {4, 32},    {32, 4},    {8, 64},    {64, 8}};
  for (int b = 0; b < 29; ++b) {
    if (kDims[b][0] == w && kDims[b][1] == h) {
      return b;
    }
  }
  return 0;
}

av2enc::cuda::Arena g_arena;
cf::CompoundFullWorkspace g_ws;

// Arena-backed device buffer. Destruction is a no-op: the arena owns memory.
template <typename T>
struct DBuf {
  T* ptr = nullptr;
  std::size_t cap = 0;

  bool alloc(std::size_t n) {
    cap = n;
    ptr = g_arena.allocate_array<T>(n);
    return ptr != nullptr;
  }
  bool upload(const std::vector<T>& host) {
    if (host.size() > cap) return false;
    if (host.empty()) return true;
    return cudaMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool download(std::vector<T>* host, std::size_t n) const {
    host->resize(n);
    if (n == 0) return true;
    return cudaMemcpy(host->data(), ptr, n * sizeof(T),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }
};

// Fixed-capacity fixtures (M2 maximum: 64x64 luma, 32x32 chroma).
constexpr int kMaxLuma = 64 * 64;
constexpr int kMaxChroma = 32 * 32;
constexpr int kPlaneMax = 80 * 80;

struct Fixture {
  DBuf<std::int16_t> p0, p1;
  DBuf<std::uint8_t> intra8;
  DBuf<std::uint16_t> intra16;
  DBuf<std::uint8_t> dst8, mask8;
  DBuf<std::uint16_t> dst16;
  DBuf<std::uint8_t> ref8[2][3];
  DBuf<std::uint16_t> ref16[2][3];
  DBuf<std::uint8_t> out8[3];
  DBuf<std::uint16_t> out16[3];
  DBuf<std::uint8_t> opfl_p0_8, opfl_p1_8;
  DBuf<std::uint16_t> opfl_p0_16, opfl_p1_16;
  DBuf<std::int16_t> opfl_out;
  DBuf<std::uint8_t> refine_ref8[2];
  DBuf<std::int16_t> batch_p0, batch_p1;
  DBuf<std::uint8_t> batch_dst;
  DBuf<std::uint8_t> skip_pool;
  DBuf<std::uint16_t> skip_pool16;
  DBuf<std::uint64_t> scratch64;

  bool init() {
    bool ok = true;
    ok &= p0.alloc(kMaxLuma);
    ok &= p1.alloc(kMaxLuma);
    ok &= intra8.alloc(kMaxLuma);
    ok &= intra16.alloc(kMaxLuma);
    ok &= dst8.alloc(kMaxLuma);
    ok &= dst16.alloc(kMaxLuma);
    ok &= mask8.alloc(kMaxLuma);
    for (int l = 0; l < 2; ++l) {
      for (int p = 0; p < 3; ++p) {
        const std::size_t n = p == 0 ? kMaxLuma : kMaxChroma;
        ok &= ref8[l][p].alloc(n);
        ok &= ref16[l][p].alloc(n);
      }
      ok &= refine_ref8[l].alloc(kPlaneMax);
    }
    for (int p = 0; p < 3; ++p) {
      const std::size_t n = p == 0 ? kMaxLuma : kMaxChroma;
      ok &= out8[p].alloc(n);
      ok &= out16[p].alloc(n);
    }
    ok &= opfl_p0_8.alloc(kMaxLuma);
    ok &= opfl_p1_8.alloc(kMaxLuma);
    ok &= opfl_p0_16.alloc(kMaxLuma);
    ok &= opfl_p1_16.alloc(kMaxLuma);
    ok &= opfl_out.alloc(kMaxLuma * 8);
    ok &= batch_p0.alloc(kMaxLuma * 32);
    ok &= batch_p1.alloc(kMaxLuma * 32);
    ok &= batch_dst.alloc(kMaxLuma * 32);
    ok &= skip_pool.alloc(kMaxLuma * 16);
    ok &= skip_pool16.alloc(kMaxLuma * 16);
    return ok;
  }
};

Fixture g_fx;

struct BlendMode {
  int type = cf::kCompoundAverage;
  int cwp = 8;
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
  int ii_mode = 0;
  int imp = 0;
  int ref_sel = 0;
};

adv::CompoundJob to_cpu_job(const cf::BlendJob& job, std::uint8_t* dst,
                            std::uint8_t* mask,
                            const std::uint8_t* luma_mask) {
  adv::CompoundJob cpu{};
  cpu.plane = job.plane;
  cpu.sub_x = job.sub_x;
  cpu.sub_y = job.sub_y;
  cpu.rect = adv::BlockRect{job.rect_x, job.rect_y, job.rect_w, job.rect_h};
  cpu.bsize = job.bsize;
  cpu.bit_depth = job.bit_depth;
  cpu.p0 = job.p0;
  cpu.p0_stride = job.p0_stride;
  cpu.p1 = job.p1;
  cpu.p1_stride = job.p1_stride;
  cpu.intra = job.intra ? static_cast<const std::uint8_t*>(job.intra) : nullptr;
  cpu.intra_stride = job.intra_stride;
  cpu.luma_mask = luma_mask;
  cpu.luma_mask_stride = job.luma_mask_stride;
  cpu.mask_out = mask;
  cpu.mask_out_stride = job.mask_out_stride;
  cpu.dst = dst;
  cpu.dst_stride = job.dst_stride;
  cpu.type = static_cast<adv::CompoundType>(job.type);
  cpu.cwp_weight = job.cwp_weight;
  cpu.wedge_index = job.wedge_index;
  cpu.wedge_sign = job.wedge_sign;
  cpu.mask_type = job.mask_type;
  cpu.imp_mask_eligible = job.imp_mask_eligible;
  for (int l = 0; l < 2; ++l) {
    cpu.ref_start_x[l] = job.ref_start_x[l];
    cpu.ref_start_y[l] = job.ref_start_y[l];
  }
  cpu.mi_cols = job.mi_cols;
  cpu.mi_rows = job.mi_rows;
  cpu.is_inter_intra = job.is_inter_intra;
  cpu.interintra_mode =
      static_cast<adv::InterIntraMode>(job.interintra_mode);
  return cpu;
}

// ---------------------------------------------------------------------------
// Blend
// ---------------------------------------------------------------------------

bool launch_blend(const std::vector<cf::BlendJob>& jobs, bool use_cache) {
  if (jobs.size() > static_cast<std::size_t>(g_ws.max_jobs)) return false;
  if (cudaMemcpy(g_ws.jobs, jobs.data(),
                 jobs.size() * sizeof(cf::BlendJob),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    return false;
  }
  if (use_cache) {
    return cf::blend_batch_ws(g_ws, jobs.data(),
                              static_cast<int>(jobs.size()), nullptr) ==
           cudaSuccess;
  }
  return cf::blend_batch(g_ws.jobs, static_cast<int>(jobs.size())) ==
         cudaSuccess;
}

void run_blend_case(const std::vector<std::int16_t>& p0,
                    const std::vector<std::int16_t>& p1,
                    const std::vector<std::uint8_t>& intra8,
                    const std::vector<std::uint16_t>& intra16, int w, int h,
                    const BlendMode& mode, int bit_depth, bool use_cache,
                    std::vector<std::uint8_t>* out_mask) {
  const std::size_t total = static_cast<std::size_t>(w) * h;
  const int max_sample = bit_depth == 8 ? 255 : 1023;
  std::vector<std::uint8_t> host_dst8(total, 0);
  std::vector<std::uint16_t> host_dst16(total, 0);
  std::vector<std::uint8_t> host_mask(total, 0);
  for (std::size_t i = 0; i < total; ++i) {
    const int inter = random_range(0, max_sample);
    host_dst8[i] = static_cast<std::uint8_t>(inter);
    host_dst16[i] = static_cast<std::uint16_t>(inter);
  }
  EXPECT(g_fx.p0.upload(p0));
  EXPECT(g_fx.p1.upload(p1));
  EXPECT(g_fx.intra8.upload(intra8));
  EXPECT(g_fx.intra16.upload(intra16));

  cf::BlendJob job{};
  job.plane = 0;
  job.rect_w = w;
  job.rect_h = h;
  job.bsize = bsize_for(w, h);
  job.bit_depth = bit_depth;
  job.p0 = g_fx.p0.ptr;
  job.p0_stride = w;
  job.p1 = g_fx.p1.ptr;
  job.p1_stride = w;
  job.type = mode.type;
  job.cwp_weight = mode.cwp;
  job.wedge_index = mode.wedge_index;
  job.wedge_sign = mode.wedge_sign;
  job.mask_type = mode.mask_type;
  job.imp_mask_eligible = mode.imp;
  job.interintra_mode = mode.ii_mode;
  job.mi_cols = w;
  job.mi_rows = h;
  job.ref_start_x[0] = mode.ref_sel == 1 ? -2 : 0;
  job.ref_start_x[1] = mode.ref_sel == 2 ? w + 2 : 0;
  job.ref_start_y[0] = mode.ref_sel == 3 ? h + 2 : 0;
  job.ref_start_y[1] = mode.ref_sel == 3 ? -2 : 0;
  job.mask_out = g_fx.mask8.ptr;
  job.mask_out_stride = w;
  job.wedge_cache = use_cache ? g_ws.wedge_cache : nullptr;

  std::vector<cf::BlendJob> jobs{job};
  std::vector<std::uint8_t> gpu_dst8;
  std::vector<std::uint16_t> gpu_dst16;
  std::vector<std::uint8_t> gpu_mask;
  if (bit_depth == 8) {
    job.dst = g_fx.dst8.ptr;
    job.dst_stride = w;
    job.intra = g_fx.intra8.ptr;
    job.intra_stride = w;
    if (mode.type == cf::kCompoundIntra || mode.imp == 2) {
      job.is_inter_intra = 1;
    }
    jobs[0] = job;
    EXPECT(g_fx.dst8.upload(host_dst8));
    EXPECT(launch_blend(jobs, use_cache));
    EXPECT(g_fx.dst8.download(&gpu_dst8, total));
    EXPECT(g_fx.mask8.download(&gpu_mask, total));
  } else {
    job.dst = g_fx.dst16.ptr;
    job.dst_stride = w;
    job.dst_16bit = 1;
    job.intra = g_fx.intra16.ptr;
    job.intra_stride = w;
    job.intra_16bit = 1;
    if (mode.type == cf::kCompoundIntra || mode.imp == 2) {
      job.is_inter_intra = 1;
    }
    jobs[0] = job;
    EXPECT(g_fx.dst16.upload(host_dst16));
    EXPECT(launch_blend(jobs, use_cache));
    EXPECT(g_fx.dst16.download(&gpu_dst16, total));
    EXPECT(g_fx.mask8.download(&gpu_mask, total));
  }

  if (out_mask != nullptr) {
    *out_mask = gpu_mask;
  }

  if (bit_depth == 8) {
    adv::CompoundJob cpu = to_cpu_job(job, host_dst8.data(), host_mask.data(),
                                      nullptr);
    cpu.p0 = p0.data();
    cpu.p1 = p1.data();
    cpu.intra = intra8.data();
    cf::test::run_compound_job_cpu(cpu);
    EXPECT(std::memcmp(gpu_dst8.data(), host_dst8.data(), total) == 0);
    EXPECT(std::memcmp(gpu_mask.data(), host_mask.data(), total) == 0);
  } else {
    cf::BlendJob cpu_job = job;
    cpu_job.p0 = p0.data();
    cpu_job.p1 = p1.data();
    cpu_job.intra = intra16.data();
    cpu_job.dst = host_dst16.data();
    std::vector<std::uint8_t> ref_mask(total, 0);
    cpu_job.mask_out = ref_mask.data();
    ref::blend_job(cpu_job);
    EXPECT(std::memcmp(gpu_dst16.data(), host_dst16.data(),
                       total * sizeof(std::uint16_t)) == 0);
    EXPECT(std::memcmp(gpu_mask.data(), ref_mask.data(), total) == 0);
  }
}

void test_blend_gpu() {
  const int sizes[][2] = {{4, 4},   {8, 8},   {8, 4},   {4, 8},   {16, 16},
                          {8, 16},  {16, 8},  {32, 32}, {32, 16}, {16, 32},
                          {64, 64}, {64, 32}, {32, 64}, {64, 16}, {16, 64},
                          {8, 32},  {32, 8},  {64, 8},  {8, 64}};
  int cases = 0;
  for (const auto& size : sizes) {
    const int w = size[0];
    const int h = size[1];
    const int bsize = bsize_for(w, h);
    const std::size_t total = static_cast<std::size_t>(w) * h;
    std::vector<std::int16_t> p0(total), p1(total);
    std::vector<std::uint8_t> intra8(total);
    std::vector<std::uint16_t> intra16(total);
    for (std::size_t i = 0; i < total; ++i) {
      p0[i] = static_cast<std::int16_t>(random_range(0, 255) * 16 +
                                        random_range(-15, 15));
      p1[i] = static_cast<std::int16_t>(random_range(0, 255) * 16 +
                                        random_range(-15, 15));
      intra8[i] = static_cast<std::uint8_t>(random_range(0, 255));
      intra16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    }

    std::vector<BlendMode> modes;
    for (int cwp : {8, 12, 4, 20, -4, 10, 6}) {
      BlendMode m;
      m.type = cf::kCompoundAverage;
      m.cwp = cwp;
      modes.push_back(m);
    }
    if (cf::detail::cf_bsize_has_wedge(bsize)) {
      const int indices[] = {0, 1, 17, 33, 34, 50, 67};
      for (int idx : indices) {
        for (int sign = 0; sign < 2; ++sign) {
          BlendMode m;
          m.type = cf::kCompoundWedge;
          m.wedge_index = idx;
          m.wedge_sign = sign;
          modes.push_back(m);
        }
      }
      if (w == 64 && h == 64) {
        for (int idx = 0; idx < cf::kWedgeTypes; ++idx) {
          for (int sign = 0; sign < 2; ++sign) {
            BlendMode m;
            m.type = cf::kCompoundWedge;
            m.wedge_index = idx;
            m.wedge_sign = sign;
            modes.push_back(m);
          }
        }
      }
      for (int idx : {0, 33, 67}) {
        BlendMode m;
        m.type = cf::kCompoundWedge;
        m.wedge_index = idx;
        m.wedge_sign = 1;
        m.ii_mode = 3;
        m.imp = 2;
        modes.push_back(m);
      }
    }
    for (int mt = 0; mt < 2; ++mt) {
      BlendMode m;
      m.type = cf::kCompoundDiffWtd;
      m.mask_type = mt;
      modes.push_back(m);
    }
    for (int ii = 0; ii < 4; ++ii) {
      BlendMode m;
      m.type = cf::kCompoundIntra;
      m.ii_mode = ii;
      m.imp = 2;
      modes.push_back(m);
    }

    std::vector<std::uint8_t> mask;
    for (const BlendMode& mode : modes) {
      for (int bit_depth : {8, 10}) {
        ++cases;
        run_blend_case(p0, p1, intra8, intra16, w, h, mode, bit_depth,
                       (cases % 2) != 0, &mask);
      }
    }
    for (int sel = 0; sel < 4; ++sel) {
      BlendMode m;
      m.type = cf::kCompoundAverage;
      m.cwp = 8;
      m.imp = 1;
      m.ref_sel = sel;
      run_blend_case(p0, p1, intra8, intra16, w, h, m, 8, false, &mask);
      run_blend_case(p0, p1, intra8, intra16, w, h, m, 10, false, &mask);
    }

    // Chroma 4:2:0 with the luma mask produced above, and the luma-prediction
    // derivation for DIFFWTD.
    if (w >= 2 && h >= 2) {
      const int cw = w / 2;
      const int ch = h / 2;
      BlendMode mask_src;
      mask_src.type = cf::kCompoundWedge;
      mask_src.wedge_index = 33;
      mask_src.wedge_sign = 1;
      run_blend_case(p0, p1, intra8, intra16, w, h, mask_src, 8, false, &mask);
      std::vector<std::int16_t> cp0(static_cast<std::size_t>(cw) * ch);
      std::vector<std::int16_t> cp1(cp0.size());
      std::vector<std::uint8_t> ci8(cp0.size());
      std::vector<std::uint16_t> ci16(cp0.size());
      for (std::size_t i = 0; i < cp0.size(); ++i) {
        cp0[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
        cp1[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
        ci8[i] = static_cast<std::uint8_t>(random_range(0, 255));
        ci16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
      }
      const std::size_t ctotal = static_cast<std::size_t>(cw) * ch;
      for (int bit_depth : {8, 10}) {
        for (BlendMode mode : {BlendMode{cf::kCompoundAverage, 8, 0, 0, 0, 0, 0,
                                         0},
                               BlendMode{cf::kCompoundWedge, 8, 33, 1, 0, 0,
                                         0, 0},
                               BlendMode{cf::kCompoundDiffWtd, 8, 0, 0, 0, 0,
                                         0, 0},
                               BlendMode{cf::kCompoundDiffWtd, 8, 0, 0, 1, 0,
                                         0, 0},
                               BlendMode{cf::kCompoundIntra, 8, 0, 0, 0, 2, 2,
                                         0}}) {
          ++cases;
          EXPECT(g_fx.p0.upload(cp0));
          EXPECT(g_fx.p1.upload(cp1));
          EXPECT(g_fx.intra8.upload(ci8));
          EXPECT(g_fx.intra16.upload(ci16));
          cf::BlendJob job{};
          job.plane = 1;
          job.sub_x = 1;
          job.sub_y = 1;
          job.rect_w = cw;
          job.rect_h = ch;
          job.bsize = bsize;
          job.bit_depth = bit_depth;
          job.p0 = g_fx.p0.ptr;
          job.p0_stride = cw;
          job.p1 = g_fx.p1.ptr;
          job.p1_stride = cw;
          job.type = mode.type;
          job.cwp_weight = mode.cwp;
          job.wedge_index = mode.wedge_index;
          job.wedge_sign = mode.wedge_sign;
          job.mask_type = mode.mask_type;
          job.interintra_mode = mode.ii_mode;
          job.mi_cols = w;
          job.mi_rows = h;
          job.luma_mask = g_fx.mask8.ptr;
          job.luma_mask_stride = w;
          if (mode.type == cf::kCompoundDiffWtd) {
            job.luma_p0 = g_fx.p0.ptr;
            job.luma_p0_stride = w;
            job.luma_p1 = g_fx.p1.ptr;
            job.luma_p1_stride = w;
          }
          // Inter-intra consumes an existing inter prediction. Keep the
          // pre-launch contents in `inter*` and the kernel result in
          // `dst*`; the reference must see the pre-launch contents, not the
          // GPU output.
          std::vector<std::uint8_t> inter8(ctotal, 0);
          std::vector<std::uint16_t> inter16(ctotal, 0);
          if (mode.type == cf::kCompoundIntra) {
            for (std::size_t i = 0; i < ctotal; ++i) {
              inter8[i] = static_cast<std::uint8_t>(random_range(0, 255));
              inter16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
            }
          }
          std::vector<std::uint8_t> dst8(ctotal, 0);
          std::vector<std::uint16_t> dst16(ctotal, 0);
          if (bit_depth == 8) {
            job.dst = g_fx.dst8.ptr;
            job.dst_stride = cw;
            job.intra = g_fx.intra8.ptr;
            job.intra_stride = cw;
            if (mode.type == cf::kCompoundIntra) job.is_inter_intra = 1;
            EXPECT(g_fx.dst8.upload(inter8));
            EXPECT(launch_blend({job}, false));
            EXPECT(g_fx.dst8.download(&dst8, ctotal));
            std::vector<std::uint8_t> cpu_dst = inter8;
            adv::CompoundJob cpu =
                to_cpu_job(job, cpu_dst.data(), nullptr, mask.data());
            cpu.p0 = cp0.data();
            cpu.p1 = cp1.data();
            cpu.intra = ci8.data();
            cf::test::run_compound_job_cpu(cpu);
            EXPECT(std::memcmp(dst8.data(), cpu_dst.data(), ctotal) == 0);
          } else {
            job.dst = g_fx.dst16.ptr;
            job.dst_stride = cw;
            job.dst_16bit = 1;
            job.intra = g_fx.intra16.ptr;
            job.intra_stride = cw;
            job.intra_16bit = 1;
            if (mode.type == cf::kCompoundIntra) job.is_inter_intra = 1;
            EXPECT(g_fx.dst16.upload(inter16));
            EXPECT(launch_blend({job}, false));
            EXPECT(g_fx.dst16.download(&dst16, ctotal));
            std::vector<std::uint16_t> ref_dst = inter16;
            cf::BlendJob ref_job = job;
            ref_job.p0 = cp0.data();
            ref_job.p1 = cp1.data();
            ref_job.intra = ci16.data();
            ref_job.luma_mask = mask.data();
            ref_job.luma_p0 = cp0.data();
            ref_job.luma_p1 = cp1.data();
            ref_job.dst = ref_dst.data();
            ref::blend_job(ref_job);
            bool equal = std::memcmp(dst16.data(), ref_dst.data(),
                                     ctotal * sizeof(std::uint16_t)) == 0;
            if (!equal) {
              for (std::size_t i = 0; i < ctotal; ++i) {
                if (dst16[i] != ref_dst[i]) {
                  std::printf(
                      "MISMATCH chroma %dx%d bsize=%d type=%d bd=%d cwp=%d "
                      "wi=%d ws=%d mt=%d ii=%d inter_intra=%d "
                      "idx=%zu row=%zu col=%zu gpu=%u ref=%u p0=%d p1=%d "
                      "intra=%u\n",
                      cw, ch, bsize, mode.type, bit_depth, mode.cwp,
                      mode.wedge_index, mode.wedge_sign, mode.mask_type,
                      mode.ii_mode, job.is_inter_intra, i, i / cw, i % cw,
                      dst16[i], ref_dst[i], cp0[i], cp1[i], ci16[i]);
                  break;
                }
              }
            }
            EXPECT(equal);
          }
        }
      }
    }
  }
  std::printf("blend: %d GPU cases\n", cases);
}

// ---------------------------------------------------------------------------
// Batch: all families, one launch each, descriptor arrays in the workspace.
// ---------------------------------------------------------------------------

void test_batch_jobs() {
  const int w = 32;
  const int h = 32;
  const std::size_t total = static_cast<std::size_t>(w) * h;
  const int jobs_n = 12;
  std::vector<std::int16_t> p0(total * jobs_n), p1(total * jobs_n);
  for (std::size_t i = 0; i < p0.size(); ++i) {
    p0[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
    p1[i] = static_cast<std::int16_t>(random_range(0, 255) * 16);
  }
  EXPECT(g_fx.batch_p0.cap >= p0.size());
  EXPECT(g_fx.batch_p1.cap >= p1.size());
  EXPECT(g_fx.batch_dst.cap >= p0.size());
  EXPECT(g_fx.batch_p0.upload(p0));
  EXPECT(g_fx.batch_p1.upload(p1));

  std::vector<cf::BlendJob> jobs(static_cast<std::size_t>(jobs_n));
  std::vector<std::uint8_t> cpu_dst(total * jobs_n, 0);
  for (int k = 0; k < jobs_n; ++k) {
    cf::BlendJob job{};
    job.plane = 0;
    job.rect_w = w;
    job.rect_h = h;
    job.bsize = 9;
    job.bit_depth = 8;
    job.p0 = g_fx.batch_p0.ptr + static_cast<std::size_t>(k) * total;
    job.p0_stride = w;
    job.p1 = g_fx.batch_p1.ptr + static_cast<std::size_t>(k) * total;
    job.p1_stride = w;
    job.dst = g_fx.batch_dst.ptr + static_cast<std::size_t>(k) * total;
    job.dst_stride = w;
    job.mi_cols = w;
    job.mi_rows = h;
    switch (k % 3) {
      case 1:
        job.type = cf::kCompoundWedge;
        job.wedge_index = (k * 7) % cf::kWedgeTypes;
        job.wedge_sign = k & 1;
        break;
      case 2:
        job.type = cf::kCompoundDiffWtd;
        job.mask_type = k & 1;
        break;
      default:
        job.type = cf::kCompoundAverage;
        job.cwp_weight = 8 + (k % 5) * 2;
        break;
    }
    jobs[static_cast<std::size_t>(k)] = job;
    // CPU reference uses host chunks (the device pointers must not be
    // dereferenced by the CPU oracle).
    adv::CompoundJob cpu = to_cpu_job(job, cpu_dst.data() + k * total, nullptr,
                                      nullptr);
    cpu.p0 = p0.data() + static_cast<std::size_t>(k) * total;
    cpu.p1 = p1.data() + static_cast<std::size_t>(k) * total;
    cpu.intra = nullptr;
    cpu.is_inter_intra = 0;
    cf::test::run_compound_job_cpu(cpu);
  }

  EXPECT(cf::blend_batch_ws(g_ws, jobs.data(), jobs_n) == cudaSuccess);
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> gpu_dst;
  EXPECT(g_fx.batch_dst.download(&gpu_dst, p0.size()));
  EXPECT(std::memcmp(gpu_dst.data(), cpu_dst.data(), cpu_dst.size()) == 0);
  std::printf("batch: %d blend descriptors in one launch\n", jobs_n);
}

// ---------------------------------------------------------------------------
// Skip mode
// ---------------------------------------------------------------------------

template <typename T>
bool skip_equal_report(const char* tag, const std::vector<T>& gpu,
                       const std::vector<T>& cpu, int plane, int dim, int x,
                       int y, int w, int h, const T* ref0, const T* ref1) {
  if (gpu.size() != cpu.size()) {
    std::printf("SKIP MISMATCH %s size %zu vs %zu\n", tag, gpu.size(),
                cpu.size());
    return false;
  }
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    if (gpu[i] == cpu[i]) continue;
    const int row = static_cast<int>(i) / dim;
    const int col = static_cast<int>(i) - row * dim;
    const bool in_block = row >= y && row < y + h && col >= x && col < x + w;
    std::printf(
        "SKIP MISMATCH %s plane=%d idx=%zu row=%d col=%d gpu=%u cpu=%u "
        "ref0=%u ref1=%u in_block=%d block=(%d,%d,%d,%d)\n",
        tag, plane, i, row, col, static_cast<unsigned>(gpu[i]),
        static_cast<unsigned>(cpu[i]), static_cast<unsigned>(ref0[i]),
        static_cast<unsigned>(ref1[i]), in_block ? 1 : 0, x, y, w, h);
    return false;
  }
  return true;
}

void test_skip_gpu() {
  const int luma = 64;
  const int chroma = 32;
  std::vector<std::uint8_t> y0(luma * luma), y1(luma * luma);
  std::vector<std::uint8_t> u0(chroma * chroma), u1(chroma * chroma);
  std::vector<std::uint8_t> v0(chroma * chroma), v1(chroma * chroma);
  std::vector<std::uint16_t> y0_16(luma * luma), y1_16(luma * luma);
  std::vector<std::uint16_t> u0_16(chroma * chroma), u1_16(chroma * chroma);
  std::vector<std::uint16_t> v0_16(chroma * chroma), v1_16(chroma * chroma);
  for (std::size_t i = 0; i < y0.size(); ++i) {
    y0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    y1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    y0_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    y1_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
  }
  for (std::size_t i = 0; i < u0.size(); ++i) {
    u0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    u1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    v0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    v1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    u0_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    u1_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    v0_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    v1_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
  }
  const std::uint8_t* src8[2][3] = {{y0.data(), u0.data(), v0.data()},
                                    {y1.data(), u1.data(), v1.data()}};
  const std::uint16_t* src16[2][3] = {{y0_16.data(), u0_16.data(), v0_16.data()},
                                      {y1_16.data(), u1_16.data(), v1_16.data()}};
  for (int l = 0; l < 2; ++l) {
    EXPECT(g_fx.ref8[l][0].upload(std::vector<std::uint8_t>(
        src8[l][0], src8[l][0] + luma * luma)));
    EXPECT(g_fx.ref8[l][1].upload(std::vector<std::uint8_t>(
        src8[l][1], src8[l][1] + chroma * chroma)));
    EXPECT(g_fx.ref8[l][2].upload(std::vector<std::uint8_t>(
        src8[l][2], src8[l][2] + chroma * chroma)));
    EXPECT(g_fx.ref16[l][0].upload(std::vector<std::uint16_t>(
        src16[l][0], src16[l][0] + luma * luma)));
    EXPECT(g_fx.ref16[l][1].upload(std::vector<std::uint16_t>(
        src16[l][1], src16[l][1] + chroma * chroma)));
    EXPECT(g_fx.ref16[l][2].upload(std::vector<std::uint16_t>(
        src16[l][2], src16[l][2] + chroma * chroma)));
  }

  tmc::RefFrame frames[2];
  for (int l = 0; l < 2; ++l) {
    for (int p = 0; p < 3; ++p) {
      frames[l].plane[p] =
          tmc::PlaneView{src8[l][p], p == 0 ? luma : chroma};
    }
    frames[l].mi_cols = luma / 4;
    frames[l].mi_rows = luma / 4;
  }

  const int rects[][4] = {{0, 0, 16, 16},   {16, 16, 32, 32}, {32, 32, 32, 32},
                          {48, 48, 16, 16}, {0, 32, 16, 16},  {48, 0, 16, 16},
                          {16, 0, 48, 16},  {0, 16, 16, 48},  {0, 0, 64, 64}};
  std::vector<cf::SkipJob> jobs;
  for (const auto& rect : rects) {
    for (int filter : {static_cast<int>(tmc::InterpFilter::kEighttap),
                       static_cast<int>(tmc::InterpFilter::kEighttapSharp)}) {
      cf::SkipJob job{};
      job.bit_depth = 8;
      job.frame_w = luma;
      job.frame_h = luma;
      job.mi_cols = luma / 4;
      job.mi_rows = luma / 4;
      job.x = rect[0];
      job.y = rect[1];
      job.w = rect[2];
      job.h = rect[3];
      job.bsize = bsize_for(job.w, job.h);
      job.mv_row = random_range(-24, 24) * 8;
      job.mv_col = random_range(-24, 24) * 8;
      job.filter = filter;
      for (int p = 0; p < 3; ++p) {
        job.ref[0][p] = cf::RefPlane{g_fx.ref8[0][p].ptr,
                                     p == 0 ? luma : chroma,
                                     p == 0 ? luma : chroma,
                                     p == 0 ? luma : chroma, 0};
        job.ref[1][p] = cf::RefPlane{g_fx.ref8[1][p].ptr,
                                     p == 0 ? luma : chroma,
                                     p == 0 ? luma : chroma,
                                     p == 0 ? luma : chroma, 0};
        job.dst[p] = g_fx.out8[p].ptr;
        job.dst_stride[p] = p == 0 ? luma : chroma;
      }
      jobs.push_back(job);
    }
  }
  EXPECT(jobs.size() <= static_cast<std::size_t>(g_ws.max_jobs));
  for (std::size_t k = 0; k < jobs.size(); ++k) {
    // The prediction writes only the block rectangle. Clear the full planes so
    // bytes outside the block match the zeroed CPU oracle buffers.
    for (int p = 0; p < 3; ++p) {
      const std::size_t n =
          p == 0 ? static_cast<std::size_t>(luma) * luma
                 : static_cast<std::size_t>(chroma) * chroma;
      EXPECT(g_fx.out8[p].upload(std::vector<std::uint8_t>(n, 0)));
      EXPECT(g_fx.out16[p].upload(std::vector<std::uint16_t>(n, 0)));
    }
    EXPECT(cf::skip_batch_ws(g_ws, &jobs[k], 1) == cudaSuccess);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> gy, gu, gv;
    EXPECT(g_fx.out8[0].download(&gy, luma * luma));
    EXPECT(g_fx.out8[1].download(&gu, chroma * chroma));
    EXPECT(g_fx.out8[2].download(&gv, chroma * chroma));
    // CPU oracle.
    av2enc::video::multiref::CompoundPrediction prediction;
    std::string error;
    EXPECT(av2enc::video::multiref::build_compound_prediction(
        frames[0], frames[1],
        tmc::BlockRect{jobs[k].x, jobs[k].y, jobs[k].w, jobs[k].h},
        jobs[k].bsize, tmc::Mv{jobs[k].mv_row, jobs[k].mv_col},
        tmc::Mv{jobs[k].mv_row, jobs[k].mv_col},
        static_cast<tmc::InterpFilter>(jobs[k].filter), luma / 4, luma / 4,
        prediction, error));
    std::vector<std::uint8_t> cy(luma * luma, 0), cu(chroma * chroma, 0),
        cv(chroma * chroma, 0);
    std::uint8_t* dst[3] = {cy.data(), cu.data(), cv.data()};
    int strides[3] = {luma, chroma, chroma};
    av2enc::video::multiref::CompoundModeParams params;
    params.type = adv::CompoundType::kAverage;
    params.cwp_weight = 8;
    EXPECT(av2enc::video::multiref::apply_compound_prediction(prediction, params,
                                                           dst, strides,
                                                           error));
    EXPECT(skip_equal_report("8bit-Y", gy, cy, 0, luma, jobs[k].x,
                              jobs[k].y, jobs[k].w, jobs[k].h, y0.data(),
                              y1.data()));
    EXPECT(skip_equal_report("8bit-U", gu, cu, 1, chroma, jobs[k].x >> 1,
                              jobs[k].y >> 1, jobs[k].w >> 1, jobs[k].h >> 1,
                              u0.data(), u1.data()));
    EXPECT(skip_equal_report("8bit-V", gv, cv, 2, chroma, jobs[k].x >> 1,
                              jobs[k].y >> 1, jobs[k].w >> 1, jobs[k].h >> 1,
                              v0.data(), v1.data()));

    // 10-bit vs the independent reference (through the GPU).
    cf::SkipJob job16 = jobs[k];
    job16.bit_depth = 10;
    for (int p = 0; p < 3; ++p) {
      const int dim = p == 0 ? luma : chroma;
      job16.ref[0][p] =
          cf::RefPlane{g_fx.ref16[0][p].ptr, dim, dim, dim, 1};
      job16.ref[1][p] =
          cf::RefPlane{g_fx.ref16[1][p].ptr, dim, dim, dim, 1};
      job16.dst[p] = g_fx.out16[p].ptr;
      job16.dst_stride[p] = dim;
      job16.dst_16bit[p] = 1;
    }
    EXPECT(cf::skip_batch_ws(g_ws, &job16, 1) == cudaSuccess);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> gy16, gu16, gv16;
    EXPECT(g_fx.out16[0].download(&gy16, luma * luma));
    EXPECT(g_fx.out16[1].download(&gu16, chroma * chroma));
    EXPECT(g_fx.out16[2].download(&gv16, chroma * chroma));
    cf::SkipJob ref_job = job16;
    std::vector<std::uint16_t> ry(luma * luma, 0), ru(chroma * chroma, 0),
        rv(chroma * chroma, 0);
    for (int l = 0; l < 2; ++l) {
      for (int p = 0; p < 3; ++p) {
        const int dim = p == 0 ? luma : chroma;
        ref_job.ref[l][p] =
            cf::RefPlane{src16[l][p], dim, dim, dim, 1};
      }
    }
    ref_job.dst[0] = ry.data();
    ref_job.dst[1] = ru.data();
    ref_job.dst[2] = rv.data();
    ref::skip_job(ref_job);
    EXPECT(skip_equal_report("10bit-Y", gy16, ry, 0, luma, jobs[k].x,
                              jobs[k].y, jobs[k].w, jobs[k].h, y0_16.data(),
                              y1_16.data()));
    EXPECT(skip_equal_report("10bit-U", gu16, ru, 1, chroma, jobs[k].x >> 1,
                              jobs[k].y >> 1, jobs[k].w >> 1, jobs[k].h >> 1,
                              u0_16.data(), u1_16.data()));
    EXPECT(skip_equal_report("10bit-V", gv16, rv, 2, chroma, jobs[k].x >> 1,
                              jobs[k].y >> 1, jobs[k].w >> 1, jobs[k].h >> 1,
                              v0_16.data(), v1_16.data()));
  }
  // Batch mode: four skip jobs in one launch with distinct output regions.
  const int base = 64 * 64 + 2 * 32 * 32;
  const int batch_n = 4;
  std::vector<cf::SkipJob> batch;
  std::vector<std::uint8_t> host_pool(static_cast<std::size_t>(base) * batch_n,
                                      0);
  for (int k = 0; k < batch_n; ++k) {
    cf::SkipJob job = jobs[static_cast<std::size_t>(k)];
    std::uint8_t* dst_base =
        g_fx.skip_pool.ptr + static_cast<std::size_t>(k) * base;
    job.dst[0] = dst_base;
    job.dst[1] = dst_base + 64 * 64;
    job.dst[2] = dst_base + 64 * 64 + 32 * 32;
    job.dst_stride[0] = 64;
    job.dst_stride[1] = 32;
    job.dst_stride[2] = 32;
    batch.push_back(job);

    av2enc::video::multiref::CompoundPrediction prediction;
    std::string error;
    EXPECT(av2enc::video::multiref::build_compound_prediction(
        frames[0], frames[1],
        tmc::BlockRect{job.x, job.y, job.w, job.h}, job.bsize,
        tmc::Mv{job.mv_row, job.mv_col}, tmc::Mv{job.mv_row, job.mv_col},
        static_cast<tmc::InterpFilter>(job.filter), luma / 4, luma / 4,
        prediction, error));
    std::uint8_t* dst[3] = {
        host_pool.data() + static_cast<std::size_t>(k) * base,
        host_pool.data() + static_cast<std::size_t>(k) * base + 64 * 64,
        host_pool.data() + static_cast<std::size_t>(k) * base + 64 * 64 +
            32 * 32};
    int strides[3] = {64, 32, 32};
    av2enc::video::multiref::CompoundModeParams params;
    params.type = adv::CompoundType::kAverage;
    params.cwp_weight = 8;
    EXPECT(av2enc::video::multiref::apply_compound_prediction(
        prediction, params, dst, strides, error));
  }
  EXPECT(g_fx.skip_pool.upload(
      std::vector<std::uint8_t>(static_cast<std::size_t>(base) * batch_n, 0)));
  EXPECT(cf::skip_batch_ws(g_ws, batch.data(), batch_n) == cudaSuccess);
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> gpu_pool;
  EXPECT(g_fx.skip_pool.download(&gpu_pool, host_pool.size()));
  EXPECT(std::memcmp(gpu_pool.data(), host_pool.data(), host_pool.size()) == 0);

  std::printf("skip: %zu jobs, %d in one batch\n", jobs.size(), batch_n);
}

// ---------------------------------------------------------------------------
// OPFL
// ---------------------------------------------------------------------------

void test_opfl_gpu() {
  struct Case {
    int w;
    int h;
    int n;
    int oh;
    int oh0;
    int oh1;
  };
  const Case cases[] = {{8, 8, 8, 4, 2, 6},   {16, 8, 8, 4, 2, 6},
                        {16, 16, 8, 4, 6, 2}, {32, 16, 8, 8, 0, 8},
                        {32, 32, 8, 4, 0, 8}, {64, 64, 8, 4, 1, 7},
                        {8, 8, 4, 4, 2, 6},   {16, 16, 4, 4, 3, 6},
                        {32, 32, 4, 4, 0, 8}, {64, 32, 4, 4, 2, 6}};
  for (const Case& c : cases) {
    std::vector<std::uint8_t> p0(static_cast<std::size_t>(c.w) * c.h);
    std::vector<std::uint8_t> p1(p0.size());
    for (std::size_t i = 0; i < p0.size(); ++i) {
      p0[i] = static_cast<std::uint8_t>(random_range(0, 255));
      p1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    }
    const int opt_x = random_range(0, 1) * c.n;
    const int opt_y = random_range(0, 1) * c.n;
    const int cols4 = c.w / 4;
    const int rows4 = opt_y / 4 + c.h / 4;
    const int stride4 = cols4 + 2;
    const std::size_t out_count =
        static_cast<std::size_t>(rows4) * stride4 * 4;

    EXPECT(g_fx.opfl_p0_8.upload(p0));
    EXPECT(g_fx.opfl_p1_8.upload(p1));

    // Allocate separate GPU output regions by carving the opfl_out buffer.
    static int cursor = 0;
    if (cursor + static_cast<int>(out_count) * 2 > kMaxLuma * 8) {
      cursor = 0;
    }
    std::int16_t* gpu_mvs = g_fx.opfl_out.ptr + cursor;
    cursor += static_cast<int>(out_count);
    std::int16_t* gpu_deltas = g_fx.opfl_out.ptr + cursor;
    cursor += static_cast<int>(out_count);

    cf::OpflJob job{};
    job.w = c.w;
    job.h = c.h;
    job.bit_depth = 8;
    job.p0 = g_fx.opfl_p0_8.ptr;
    job.p0_stride = c.w;
    job.p1 = g_fx.opfl_p1_8.ptr;
    job.p1_stride = c.w;
    job.n = c.n;
    job.opt_x = opt_x;
    job.opt_y = opt_y;
    job.order_hint = c.oh;
    job.ref0_order_hint = c.oh0;
    job.ref1_order_hint = c.oh1;
    job.mv[0][0] = random_range(-16, 16) * 8;
    job.mv[0][1] = random_range(-16, 16) * 8;
    job.mv[1][0] = random_range(-16, 16) * 8;
    job.mv[1][1] = random_range(-16, 16) * 8;
    job.opfl_mvs = gpu_mvs;
    job.opfl_stride4 = stride4;
    job.mv_deltas = gpu_deltas;
    job.delta_stride4 = stride4;
    EXPECT(cf::opfl_batch_ws(g_ws, &job, 1) == cudaSuccess);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<std::int16_t> mvs_gpu, deltas_gpu;
    mvs_gpu.resize(out_count);
    deltas_gpu.resize(out_count);
    EXPECT(cudaMemcpy(mvs_gpu.data(), gpu_mvs,
                      out_count * sizeof(std::int16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    EXPECT(cudaMemcpy(deltas_gpu.data(), gpu_deltas,
                      out_count * sizeof(std::int16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);

    std::vector<std::int16_t> mvs_cpu(out_count, 0), deltas_cpu(out_count, 0);
    temporal::OpflJob cpu{};
    cpu.w = c.w;
    cpu.h = c.h;
    cpu.p0 = p0.data();
    cpu.p0_stride = c.w;
    cpu.p1 = p1.data();
    cpu.p1_stride = c.w;
    cpu.mvs[0] = temporal::Mv{job.mv[0][0], job.mv[0][1]};
    cpu.mvs[1] = temporal::Mv{job.mv[1][0], job.mv[1][1]};
    cpu.order_hint = c.oh;
    cpu.ref0_order_hint = c.oh0;
    cpu.ref1_order_hint = c.oh1;
    cpu.n = c.n;
    cpu.opt_x = opt_x;
    cpu.opt_y = opt_y;
    temporal::OpflOutput cpu_out{};
    cpu_out.opfl_mvs = mvs_cpu.data();
    cpu_out.opfl_stride4 = stride4;
    cpu_out.mv_deltas = deltas_cpu.data();
    cpu_out.delta_stride4 = stride4;
    EXPECT(temporal::opfl_compute(cpu, cpu_out) ==
           temporal::TemporalStatus::kOk);
    EXPECT(std::memcmp(mvs_gpu.data(), mvs_cpu.data(),
                       out_count * sizeof(std::int16_t)) == 0);
    EXPECT(std::memcmp(deltas_gpu.data(), deltas_cpu.data(),
                       out_count * sizeof(std::int16_t)) == 0);

    // 10-bit: GPU vs the shared header executed on the host.
    std::vector<std::uint16_t> q0(p0.size()), q1(p1.size());
    for (std::size_t i = 0; i < p0.size(); ++i) {
      q0[i] = static_cast<std::uint16_t>(p0[i] * 4);
      q1[i] = static_cast<std::uint16_t>(p1[i] * 4);
    }
    EXPECT(g_fx.opfl_p0_16.upload(q0));
    EXPECT(g_fx.opfl_p1_16.upload(q1));
    cf::OpflJob job16 = job;
    job16.bit_depth = 10;
    job16.p0 = g_fx.opfl_p0_16.ptr;
    job16.p0_16 = 1;
    job16.p1 = g_fx.opfl_p1_16.ptr;
    job16.p1_16 = 1;
    job16.opfl_mvs = gpu_mvs;
    job16.mv_deltas = gpu_deltas;
    EXPECT(cf::opfl_batch_ws(g_ws, &job16, 1) == cudaSuccess);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<std::int16_t> mvs_gpu16, deltas_gpu16;
    mvs_gpu16.resize(out_count);
    deltas_gpu16.resize(out_count);
    EXPECT(cudaMemcpy(mvs_gpu16.data(), gpu_mvs,
                      out_count * sizeof(std::int16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    EXPECT(cudaMemcpy(deltas_gpu16.data(), gpu_deltas,
                      out_count * sizeof(std::int16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    std::vector<std::int16_t> mvs_host(out_count, 0), deltas_host(out_count, 0);
    std::vector<int> tmp(static_cast<std::size_t>(c.w) * c.h, 0);
    std::vector<int> pdiff(tmp.size(), 0);
    cf::OpflJob host_job = job16;
    host_job.p0 = q0.data();
    host_job.p1 = q1.data();
    host_job.opfl_mvs = mvs_host.data();
    host_job.mv_deltas = deltas_host.data();
    cf::detail::run_opfl_job_device(host_job, 0, 1, tmp.data(), pdiff.data());
    EXPECT(std::memcmp(mvs_gpu16.data(), mvs_host.data(),
                       out_count * sizeof(std::int16_t)) == 0);
    EXPECT(std::memcmp(deltas_gpu16.data(), deltas_host.data(),
                       out_count * sizeof(std::int16_t)) == 0);
  }
  std::printf("opfl: %zu cases\n", sizeof(cases) / sizeof(cases[0]));
}

// ---------------------------------------------------------------------------
// Refine-MV
// ---------------------------------------------------------------------------

void test_refine_gpu() {
  const int plane = 80;
  std::vector<std::uint8_t> r0(static_cast<std::size_t>(plane) * plane);
  std::vector<std::uint8_t> r1(r0.size());
  for (std::size_t i = 0; i < r0.size(); ++i) {
    r0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    r1[i] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  EXPECT(g_fx.refine_ref8[0].upload(r0));
  EXPECT(g_fx.refine_ref8[1].upload(r1));

  const int blocks[][4] = {{8, 8, 8, 8},     {16, 16, 16, 16},
                           {0, 0, 32, 32},   {32, 32, 32, 32},
                           {48, 16, 16, 32}, {16, 48, 32, 16},
                           {0, 64, 16, 8},   {64, 0, 8, 16},
                           {24, 24, 40, 40}};
  std::vector<cf::RefineSearchJob> jobs;
  std::vector<temporal::RefineSearchInput> cpu_inputs;
  for (const auto& b : blocks) {
    for (int use_area = 0; use_area < 2; ++use_area) {
      temporal::RefineSearchInput in{};
      in.x = b[0];
      in.y = b[1];
      in.w = b[2];
      in.h = b[3];
      in.frame_w = plane;
      in.frame_h = plane;
      in.ref[0] = temporal::PlaneView{r0.data(), plane, plane, plane};
      in.ref[1] = temporal::PlaneView{r1.data(), plane, plane, plane};
      in.ref_mi_cols[0] = plane / 4;
      in.ref_mi_cols[1] = plane / 4;
      in.ref_mi_rows[0] = plane / 4;
      in.ref_mi_rows[1] = plane / 4;
      in.ref_luma_w[0] = plane;
      in.ref_luma_w[1] = plane;
      in.ref_luma_h[0] = plane;
      in.ref_luma_h[1] = plane;
      in.cand_mvs[0] =
          temporal::Mv{random_range(-16, 16) * 8, random_range(-16, 16) * 8};
      in.cand_mvs[1] =
          temporal::Mv{random_range(-16, 16) * 8, random_range(-16, 16) * 8};
      in.num_refs = 2;
      in.tip_pred = random_range(0, 1) != 0;
      in.switchable = random_range(0, 1) != 0;
      in.use_ref_area = use_area != 0;
      in.scratch_stride = b[2] + 4;

      temporal::RefArea area[2]{};
      temporal::RefAreaInput area_in{};
      area_in.w = b[2];
      area_in.h = b[3];
      area_in.x = b[0];
      area_in.y = b[1];
      area_in.frame_w = plane;
      area_in.frame_h = plane;
      area_in.ref_mi_cols[0] = plane / 4;
      area_in.ref_mi_cols[1] = plane / 4;
      area_in.ref_mi_rows[0] = plane / 4;
      area_in.ref_mi_rows[1] = plane / 4;
      area_in.ref_luma_w[0] = plane;
      area_in.ref_luma_w[1] = plane;
      area_in.ref_luma_h[0] = plane;
      area_in.ref_luma_h[1] = plane;
      area_in.cand_mvs[0] = in.cand_mvs[0];
      area_in.cand_mvs[1] = in.cand_mvs[1];
      EXPECT(temporal::compute_ref_area(area_in, area) ==
             temporal::TemporalStatus::kOk);
      for (int l = 0; l < 2; ++l) {
        in.area[l] = area[l];
      }
      if (!in.use_ref_area) {
        for (int l = 0; l < 2; ++l) {
          in.area[l] = temporal::RefArea{0, 0, plane - 1, plane - 1};
        }
      }
      cpu_inputs.push_back(in);

      cf::RefineSearchJob job{};
      job.bit_depth = 8;
      job.x = in.x;
      job.y = in.y;
      job.w = in.w;
      job.h = in.h;
      job.frame_w = plane;
      job.frame_h = plane;
      job.ref[0] = cf::RefPlane{g_fx.refine_ref8[0].ptr, plane, plane, plane, 0};
      job.ref[1] = cf::RefPlane{g_fx.refine_ref8[1].ptr, plane, plane, plane, 0};
      job.cand_mv[0][0] = in.cand_mvs[0].row;
      job.cand_mv[0][1] = in.cand_mvs[0].col;
      job.cand_mv[1][0] = in.cand_mvs[1].row;
      job.cand_mv[1][1] = in.cand_mvs[1].col;
      job.num_refs = 2;
      job.tip_pred = in.tip_pred ? 1 : 0;
      job.switchable = in.switchable ? 1 : 0;
      job.use_ref_area = use_area;
      for (int l = 0; l < 2; ++l) {
        job.area[l][0] = in.area[l].first_x;
        job.area[l][1] = in.area[l].first_y;
        job.area[l][2] = in.area[l].last_x;
        job.area[l][3] = in.area[l].last_y;
      }
      jobs.push_back(job);
    }
  }
  // Host scratch buffers for the CPU search (pointers must stay alive).
  std::vector<std::vector<std::uint8_t>> cpu_scratch0(jobs.size());
  std::vector<std::vector<std::uint8_t>> cpu_scratch1(jobs.size());
  for (std::size_t k = 0; k < jobs.size(); ++k) {
    cpu_scratch0[k].resize(
        static_cast<std::size_t>(jobs[k].w + 4) * (jobs[k].h + 4));
    cpu_scratch1[k].resize(cpu_scratch0[k].size());
    cpu_inputs[k].scratch0 = cpu_scratch0[k].data();
    cpu_inputs[k].scratch1 = cpu_scratch1[k].data();
  }
  EXPECT(cf::refine_search_batch_ws(g_ws, jobs.data(),
                                    static_cast<int>(jobs.size())) ==
         cudaSuccess);
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<cf::RefineResult> results(jobs.size());
  EXPECT(cudaMemcpy(results.data(), g_ws.refine_results,
                    results.size() * sizeof(cf::RefineResult),
                    cudaMemcpyDeviceToHost) == cudaSuccess);

  for (std::size_t k = 0; k < jobs.size(); ++k) {
    temporal::RefineSearchResult cpu{};
    EXPECT(temporal::search_refinemv(cpu_inputs[k], &cpu) ==
           temporal::TemporalStatus::kOk);
    EXPECT(results[k].dx == cpu.dx);
    EXPECT(results[k].dy == cpu.dy);
    EXPECT(results[k].best_sad == cpu.best_sad);
    EXPECT(results[k].offset_mv[0][0] == cpu.offset_mvs.mv[0].row);
    EXPECT(results[k].offset_mv[0][1] == cpu.offset_mvs.mv[0].col);
    EXPECT(results[k].offset_mv[1][0] == cpu.offset_mvs.mv[1].row);
    EXPECT(results[k].offset_mv[1][1] == cpu.offset_mvs.mv[1].col);
  }

  // Final prediction batch.
  std::vector<cf::RefinePredictJob> pjobs;
  std::vector<std::vector<std::uint8_t>> cpu_dst(jobs.size());
  for (std::size_t k = 0; k < jobs.size(); ++k) {
    const auto& b = jobs[k];
    temporal::RefineSearchResult cpu_result{};
    EXPECT(temporal::search_refinemv(cpu_inputs[k], &cpu_result) ==
           temporal::TemporalStatus::kOk);
    std::vector<std::int16_t> s0(static_cast<std::size_t>(b.w) * b.h);
    std::vector<std::int16_t> s1(s0.size());
    temporal::RefinePredictInput pin{};
    pin.ref[0] = cpu_inputs[k].ref[0];
    pin.ref[1] = cpu_inputs[k].ref[1];
    pin.num_refs = 2;
    pin.x = b.x;
    pin.y = b.y;
    pin.w = b.w;
    pin.h = b.h;
    pin.frame_w = plane;
    pin.frame_h = plane;
    pin.offset_mvs = cpu_result.offset_mvs;
    pin.cwp_weight = 8;
    pin.use_ref_area = cpu_inputs[k].use_ref_area;
    for (int l = 0; l < 2; ++l) pin.area[l] = cpu_inputs[k].area[l];
    cpu_dst[k].assign(static_cast<std::size_t>(b.w) * b.h, 0);
    pin.dst = cpu_dst[k].data();
    pin.dst_stride = b.w;
    pin.scratch0 = s0.data();
    pin.scratch1 = s1.data();
    pin.scratch_stride = b.w;
    EXPECT(temporal::refine_predict_block(pin) ==
           temporal::TemporalStatus::kOk);

    cf::RefinePredictJob pjob{};
    pjob.bit_depth = 8;
    pjob.x = b.x;
    pjob.y = b.y;
    pjob.w = b.w;
    pjob.h = b.h;
    pjob.frame_w = plane;
    pjob.frame_h = plane;
    pjob.ref[0] = b.ref[0];
    pjob.ref[1] = b.ref[1];
    pjob.offset_mv[0][0] = cpu_result.offset_mvs.mv[0].row;
    pjob.offset_mv[0][1] = cpu_result.offset_mvs.mv[0].col;
    pjob.offset_mv[1][0] = cpu_result.offset_mvs.mv[1].row;
    pjob.offset_mv[1][1] = cpu_result.offset_mvs.mv[1].col;
    pjob.use_ref_area = b.use_ref_area;
    for (int l = 0; l < 2; ++l) {
      pjob.area[l][0] = b.area[l][0];
      pjob.area[l][1] = b.area[l][1];
      pjob.area[l][2] = b.area[l][2];
      pjob.area[l][3] = b.area[l][3];
    }
    // Reuse the 64x64 output fixture; blocks are at most 40x40.
    pjob.dst = g_fx.out8[0].ptr;
    pjob.dst_stride = b.w;
    pjobs.push_back(pjob);
  }
  EXPECT(cf::refine_predict_batch_ws(g_ws, pjobs.data(),
                                     static_cast<int>(pjobs.size())) ==
         cudaSuccess);
  CUDA_OK(cudaDeviceSynchronize());
  for (std::size_t k = 0; k < pjobs.size(); ++k) {
    // The batch shares one destination, so re-run each job individually into
    // the same buffer and compare with its CPU result.
    EXPECT(cf::refine_predict_batch_ws(g_ws, &pjobs[k], 1) == cudaSuccess);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> gpu_dst;
    EXPECT(g_fx.out8[0].download(&gpu_dst,
                                 static_cast<std::size_t>(pjobs[k].w) *
                                     pjobs[k].h));
    EXPECT(std::memcmp(gpu_dst.data(), cpu_dst[k].data(),
                       cpu_dst[k].size()) == 0);
  }
  std::printf("refine: %zu search jobs, %zu predict jobs\n", jobs.size(),
              pjobs.size());
}

// ---------------------------------------------------------------------------
// Arena contract
// ---------------------------------------------------------------------------

void test_arena_contract() {
  cf::CompoundFullConfig config;
  config.max_jobs = 64;
  const std::uint64_t required = cf::required_device_bytes(config);
  EXPECT(required > 1900000u);
  EXPECT((required % 256) == 0);

  av2enc::cuda::Arena small;
  std::string error;
  EXPECT(av2enc::cuda::is_ok(small.init(required - 1, 256, &error)));
  cf::CompoundFullWorkspace rejected;
  std::uint64_t reported = 0;
  EXPECT(cf::workspace_create(small, config, &rejected, &reported, &error) ==
         av2enc::cuda::Status::kArenaExhausted);
  EXPECT(reported == required);
  EXPECT(small.used() == 0);
  small.release();

  EXPECT(g_ws.required_bytes == required);
  EXPECT(g_arena.used() >= required);
  EXPECT(g_ws.persistent_mark >= required);
  const std::uint64_t used_before = g_arena.used();
  const std::uint64_t peak_before = g_arena.peak_used();
  const std::uint64_t mark = g_arena.mark();
  for (int pass = 0; pass < 2; ++pass) {
    EXPECT(g_arena.mark() == mark);
    EXPECT(g_arena.rewind(mark));
  }
  EXPECT(g_arena.used() == used_before);
  EXPECT(g_arena.peak_used() == peak_before);
  EXPECT(cf::wedge_cache_build(g_ws) == cudaSuccess);
  CUDA_OK(cudaDeviceSynchronize());
  std::printf(
      "arena: required=%llu, workspace_mark=%llu, used=%llu, peak=%llu\n",
      static_cast<unsigned long long>(required),
      static_cast<unsigned long long>(g_ws.persistent_mark),
      static_cast<unsigned long long>(g_arena.used()),
      static_cast<unsigned long long>(g_arena.peak_used()));
}

}  // namespace

int main() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::printf("compound_full_test: no CUDA device available\n");
    return 77;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::printf("device: %s (cc %d.%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  cf::CompoundFullConfig config;
  config.max_jobs = 64;
  const std::uint64_t required = cf::required_device_bytes(config);
  std::string error;
  if (!av2enc::cuda::is_ok(
          g_arena.init(required + 16ull * 1024 * 1024, 256, &error))) {
    std::printf("compound_full_test: arena init failed: %s\n", error.c_str());
    return 2;
  }
  if (cf::workspace_create(g_arena, config, &g_ws, nullptr, &error) !=
      av2enc::cuda::Status::kOk) {
    std::printf("compound_full_test: workspace_init failed: %s\n",
                error.c_str());
    return 2;
  }
  if (!g_fx.init()) {
    std::printf("compound_full_test: fixture allocation failed\n");
    return 2;
  }
  CUDA_OK(cf::wedge_cache_build(g_ws));
  CUDA_OK(cudaDeviceSynchronize());

  test_arena_contract();
  int before = g_checks;
  test_blend_gpu();
  std::printf("blend checks: %d\n", g_checks - before);
  before = g_checks;
  test_batch_jobs();
  std::printf("batch checks: %d\n", g_checks - before);
  before = g_checks;
  test_skip_gpu();
  std::printf("skip checks: %d\n", g_checks - before);
  before = g_checks;
  test_opfl_gpu();
  std::printf("opfl checks: %d\n", g_checks - before);
  before = g_checks;
  test_refine_gpu();
  std::printf("refine checks: %d\n", g_checks - before);

  if (g_failures == 0) {
    std::printf("compound_full_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::printf("compound_full_test: %d/%d checks FAILED\n", g_failures,
              g_checks);
  return 1;
}
#endif  // !__CUDA_ARCH__: the test body is host-only
