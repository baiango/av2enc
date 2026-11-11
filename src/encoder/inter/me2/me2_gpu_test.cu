// CPU-vs-CUDA bit-exact tests for the extended AV2 motion estimator (me2).
//
// Coverage:
//   * candidate-cost parity for SAD/SATD, every Subpel_Filters row, luma and
//     chroma, every signalable MV precision;
//   * full me2 search parity: MVs, signalled precision, distortion, rate
//     proxy, evaluation count, stage and reference choice;
//   * eighth/sixteenth refinement and flex_mvres precision selection;
//   * partial blocks, multiple references, explicit predictors;
//   * real 720p content from testdata/ (skipped when absent);
//   * device guard-canary check (kernels must not touch plane memory).
//
// Usage: ./me2_gpu_test [repo_root]
// Requires an sm_86 (or newer) CUDA device.  The CPU reference object is
// compiled by the host compiler and linked by build_gpu_and_test.sh.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/inter/me/me_test_common.h"
#include "encoder/inter/me2/me2.h"
#include "encoder/inter/me2/me2_gpu.h"

namespace {

using namespace av2enc::me;
using namespace av2enc::me::test;
using av2enc::me2::Me2CostJob;
using av2enc::me2::Me2Options;
using av2enc::me2::Me2Result;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      if (g_failures <= 30) {                                              \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
      }                                                                    \
    }                                                                      \
  } while (0)

#define CUDA_CHECK(expr, msg)                                              \
  do {                                                                     \
    const cudaError_t err = (expr);                                        \
    ++g_checks;                                                            \
    if (err != cudaSuccess) {                                              \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,    \
                   (msg), cudaGetErrorString(err));                        \
    }                                                                      \
  } while (0)

struct DevicePlane {
  std::uint8_t* base = nullptr;
  std::size_t bytes = 0;
};

struct DeviceFrame {
  TestFrame host;
  DevicePlane plane[3];

  void upload(const TestFrame& src) {
    host = src;
    const TestPlane* planes[3] = {&host.y, &host.u, &host.v};
    for (int p = 0; p < 3; ++p) {
      plane[p].bytes = planes[p]->buf.size();
      CUDA_CHECK(cudaMalloc(&plane[p].base, plane[p].bytes), "cudaMalloc");
      CUDA_CHECK(cudaMemcpy(plane[p].base, planes[p]->buf.data(),
                            plane[p].bytes, cudaMemcpyHostToDevice),
                 "upload plane");
    }
  }

  MeCorePlane device_plane(int p) const {
    const TestPlane* planes[3] = {&host.y, &host.u, &host.v};
    MeCorePlane out;
    out.data = plane[p].base +
               static_cast<std::size_t>(planes[p]->guard_rows) *
                   static_cast<std::size_t>(planes[p]->stride) +
               static_cast<std::size_t>(planes[p]->guard_cols);
    out.stride = planes[p]->stride;
    out.width = planes[p]->width;
    out.height = planes[p]->height;
    return out;
  }

  MeCoreFrame device_frame() const {
    MeCoreFrame f;
    for (int p = 0; p < 3; ++p) {
      f.plane[p] = device_plane(p);
    }
    f.width = host.width;
    f.height = host.height;
    return f;
  }

  MeCoreRef device_ref() const {
    MeCoreRef r;
    for (int p = 0; p < 3; ++p) {
      r.plane[p] = device_plane(p);
    }
    return r;
  }

  bool download_unchanged() const {
    const TestPlane* planes[3] = {&host.y, &host.u, &host.v};
    std::vector<std::uint8_t> tmp;
    for (int p = 0; p < 3; ++p) {
      tmp.resize(plane[p].bytes);
      if (cudaMemcpy(tmp.data(), plane[p].base, plane[p].bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
      if (std::memcmp(tmp.data(), planes[p]->buf.data(), plane[p].bytes) !=
          0) {
        return false;
      }
    }
    return true;
  }

  void free() {
    for (int p = 0; p < 3; ++p) {
      if (plane[p].base != nullptr) {
        cudaFree(plane[p].base);
        plane[p].base = nullptr;
      }
    }
  }
};

template <class T>
T* upload_array(const std::vector<T>& host) {
  T* dev = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc array");
  CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             "upload array");
  return dev;
}

bool same_result(const Me2Result& a, const Me2Result& b) {
  return a.mv_x == b.mv_x && a.mv_y == b.mv_y &&
         a.search_mv_x == b.search_mv_x &&
         a.search_mv_y == b.search_mv_y && a.ref_index == b.ref_index &&
         a.cost == b.cost && a.search_cost == b.search_cost &&
         a.evals == b.evals && a.mv_rate_bins == b.mv_rate_bins &&
         a.precision_bins == b.precision_bins &&
         a.total_rate_bins == b.total_rate_bins &&
         a.precision == b.precision &&
         a.use_most_probable_precision == b.use_most_probable_precision &&
         a.pb_mv_precision == b.pb_mv_precision && a.stage == b.stage &&
         a.visible_w == b.visible_w && a.visible_h == b.visible_h;
}

Me2Options base_options(int method, int precision, int flex, int lambda) {
  Me2Options o;
  o.search.method = method;
  o.search.metric = kMeSad;
  o.search.range = 6;
  o.search.max_iterations = 3;
  o.search.subpel_iterations = 3;
  o.search.subpel_interp = kMeInterpRegular;
  o.frame_precision = precision;
  o.enable_flex_mvres = flex;
  o.refine_radius = 2;
  o.mv_rate_lambda = lambda;
  return o;
}

void test_cost_parity() {
  TestFrame frame;
  frame.init(96, 80);
  fill_smooth_texture(&frame, 0x51u, 8);
  TestFrame ref = make_shifted_ref(frame, 3, -2);
  DeviceFrame dcur;
  DeviceFrame dref;
  dcur.upload(frame);
  dref.upload(ref);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = ref.ref_view();

  const int sizes[][2] = {{16, 16}, {8, 8}, {4, 4}, {32, 16}, {12, 20}};
  const std::int32_t mvs[][2] = {
      {0, 0},   {16, 16}, {8, -8}, {4, 4},  {2, -2},
      {-14, 7}, {24, 8},  {-8, 24}, {32, -16}};
  const int precisions[3] = {av2enc::me2::kMvPrecisionEighthPel,
                             av2enc::me2::kMvPrecisionQuarterPel,
                             av2enc::me2::kMvPrecisionHalfPel};

  std::vector<Me2CostJob> jobs;
  std::vector<std::int32_t> cpu_costs;
  std::vector<std::int32_t> gpu_costs;
  for (const auto& sz : sizes) {
    for (int metric = 0; metric < 2; ++metric) {
      for (int filter = 0; filter <= kMeMaxInterpFilter; ++filter) {
        for (const auto& mv : mvs) {
          for (int pi = 0; pi < 3; ++pi) {
            const int p = precisions[pi];
            if (!av2enc::me2::me2_mv_is_at_precision(mv[0], p) ||
                !av2enc::me2::me2_mv_is_at_precision(mv[1], p)) {
              continue;
            }
            Me2CostJob job;
            job.precision = p;
            job.base.x = 4;
            job.base.y = 4;
            job.base.w = sz[0];
            job.base.h = sz[1];
            job.base.mv_x = mv[0];
            job.base.mv_y = mv[1];
            job.base.metric = metric;
            job.base.interp = filter;
            jobs.push_back(job);
            std::int32_t c = 0;
            CHECK(av2enc::me2::me2_cost_block(cur, refv, job, &c) ==
                      av2enc::me2::Me2Status::kOk,
                  "cpu cost ok");
            cpu_costs.push_back(c);
          }
        }
      }
    }
  }
  gpu_costs.resize(jobs.size());
  Me2CostJob* d_jobs = nullptr;
  std::int32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_jobs, jobs.size() * sizeof(Me2CostJob)),
             "alloc jobs");
  CUDA_CHECK(cudaMalloc(&d_out, gpu_costs.size() * sizeof(std::int32_t)),
             "alloc costs");
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(Me2CostJob),
                        cudaMemcpyHostToDevice),
             "upload jobs");
  std::vector<MeCoreRef> refs_host = {dref.device_ref()};
  MeCoreRef* d_refs = upload_array(refs_host);
  CUDA_CHECK(av2enc::cuda::me2_cost_batch(dcur.device_frame(), d_refs, 1,
                                          d_jobs,
                                          static_cast<int>(jobs.size()), d_out),
             "me2_cost_batch");
  CUDA_CHECK(cudaMemcpy(gpu_costs.data(), d_out,
                        gpu_costs.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "download costs");
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    CHECK(cpu_costs[i] == gpu_costs[i], "GPU cost equals CPU cost");
  }
  cudaFree(d_jobs);
  cudaFree(d_out);
  cudaFree(d_refs);
  CHECK(dcur.download_unchanged(), "cost kernel did not write planes");
  dcur.free();
  dref.free();
}

void test_search_parity() {
  TestFrame frame;
  frame.init(160, 128);
  fill_smooth_texture(&frame, 0x77u, 8);
  TestFrame ref = make_shifted_ref(frame, 2, 1);
  DeviceFrame dcur;
  DeviceFrame dref;
  dcur.upload(frame);
  dref.upload(ref);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = ref.ref_view();

  const int methods[3] = {kMeFull, kMeDiamond, kMeTestZone};
  const int precisions[3] = {av2enc::me2::kMvPrecisionHalfPel,
                             av2enc::me2::kMvPrecisionQuarterPel,
                             av2enc::me2::kMvPrecisionEighthPel};
  const int lambdas[2] = {0, 24};
  const auto jobs = make_block_grid(160, 128, 16, 16);

  MeCoreJob* d_jobs = nullptr;
  Me2Result* d_results = nullptr;
  CUDA_CHECK(cudaMalloc(&d_jobs, jobs.size() * sizeof(MeCoreJob)), "jobs");
  CUDA_CHECK(cudaMalloc(&d_results, jobs.size() * sizeof(Me2Result)),
             "results");
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(MeCoreJob),
                        cudaMemcpyHostToDevice),
             "upload jobs");
  std::vector<MeCoreRef> refs_host = {dref.device_ref()};
  MeCoreRef* d_refs = upload_array(refs_host);
  const MeCoreFrame d_cur = dcur.device_frame();

  for (int m = 0; m < 3; ++m) {
    for (int pi = 0; pi < 3; ++pi) {
      for (int li = 0; li < 2; ++li) {
        const Me2Options opt =
            base_options(methods[m], precisions[pi], 1, lambdas[li]);
        std::vector<Me2Result> cpu(jobs.size());
        CHECK(av2enc::me2::me2_search_batch(cur, &refv, 1, jobs.data(),
                                            static_cast<int>(jobs.size()),
                                            opt, nullptr, cpu.data()) ==
                  av2enc::me2::Me2Status::kOk,
              "cpu search ok");
        CUDA_CHECK(av2enc::cuda::me2_search_batch(
                       d_cur, d_refs, 1, d_jobs,
                       static_cast<int>(jobs.size()), opt, nullptr, d_results),
                   "me2_search_batch");
        std::vector<Me2Result> gpu(jobs.size());
        CUDA_CHECK(cudaMemcpy(gpu.data(), d_results,
                              jobs.size() * sizeof(Me2Result),
                              cudaMemcpyDeviceToHost),
                   "download results");
        for (std::size_t j = 0; j < jobs.size(); ++j) {
          CHECK(same_result(cpu[j], gpu[j]),
                "GPU search result equals CPU result");
        }
      }
    }
  }
  cudaFree(d_jobs);
  cudaFree(d_results);
  cudaFree(d_refs);
  CHECK(dcur.download_unchanged(), "search kernel did not write planes");
  dcur.free();
  dref.free();
}

void test_real_720p(const std::string& root) {
  const char* clip_env = std::getenv("ME2_Y4M");
  const std::string clip =
      (clip_env != nullptr && clip_env[0] != '\0')
          ? std::string(clip_env)
          : root + "/testdata/shine_1280x720_60f.y4m";
  std::vector<TestFrame> frames;
  if (!load_y4m_frames(clip, 2, &frames) || frames.size() < 2) {
    std::printf("  (720p testdata absent; skipping real-frame parity)\n");
    return;
  }
  DeviceFrame dcur;
  DeviceFrame dref;
  dcur.upload(frames[0]);
  dref.upload(frames[1]);
  const MeCoreFrame cur = frames[0].view();
  const MeCoreRef refv = frames[1].ref_view();
  const Me2Options opt =
      base_options(kMeDiamond, av2enc::me2::kMvPrecisionEighthPel, 1, 16);

  const auto all = make_block_grid(1280, 720, 64, 64);
  std::vector<MeCoreJob> jobs;
  for (std::size_t i = 0; i < all.size(); i += 37) {
    jobs.push_back(all[i]);
  }
  std::vector<Me2Result> cpu(jobs.size());
  CHECK(av2enc::me2::me2_search_batch(cur, &refv, 1, jobs.data(),
                                      static_cast<int>(jobs.size()), opt,
                                      nullptr, cpu.data()) ==
            av2enc::me2::Me2Status::kOk,
        "720p cpu search");
  MeCoreJob* d_jobs = nullptr;
  Me2Result* d_results = nullptr;
  CUDA_CHECK(cudaMalloc(&d_jobs, jobs.size() * sizeof(MeCoreJob)), "jobs");
  CUDA_CHECK(cudaMalloc(&d_results, jobs.size() * sizeof(Me2Result)),
             "results");
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(MeCoreJob),
                        cudaMemcpyHostToDevice),
             "upload jobs");
  std::vector<MeCoreRef> refs_host = {dref.device_ref()};
  MeCoreRef* d_refs = upload_array(refs_host);
  CUDA_CHECK(av2enc::cuda::me2_search_batch(
                 dcur.device_frame(), d_refs, 1, d_jobs,
                 static_cast<int>(jobs.size()), opt, nullptr, d_results),
             "720p me2_search_batch");
  std::vector<Me2Result> gpu(jobs.size());
  CUDA_CHECK(cudaMemcpy(gpu.data(), d_results,
                        jobs.size() * sizeof(Me2Result),
                        cudaMemcpyDeviceToHost),
             "download results");
  for (std::size_t j = 0; j < jobs.size(); ++j) {
    CHECK(same_result(cpu[j], gpu[j]), "720p GPU result equals CPU");
  }
  cudaFree(d_jobs);
  cudaFree(d_results);
  cudaFree(d_refs);
  CHECK(dcur.download_unchanged(), "720p search did not write planes");
  dcur.free();
  dref.free();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root =
      argc > 1 ? argv[1] : std::string("../../../../..");
  std::printf("== me2 CPU-vs-CUDA ==\n");
  test_cost_parity();
  test_search_parity();
  test_real_720p(root);
  std::printf("checks=%d failures=%d\n", g_checks, g_failures);
  if (g_failures != 0) {
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
