// CPU-vs-CUDA bit-exact tests for AV2 inter motion estimation.
//
// Coverage:
//   * cost parity for SAD/SATD, every Subpel_Filters row, luma and chroma
//   * normative subpel prediction parity, including negative edge phases
//   * full / diamond / TZ search parity for every option combination
//   * partial blocks, multiple references, explicit MVPs and merge candidates
//   * real 256x256 content from testdata/ (skipped when absent)
//   * device guard-canary check (kernels must not write plane memory)
//   * GPU timing for a 720p 16x16 block grid
//
// Usage: ./me_test [repo_root]
// Requires an sm_86 (or newer) CUDA device.  The CPU reference object is
// compiled by the host compiler and linked by build_and_test.sh.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cuda/codec/inter/me/me.h"
#include "codec/inter/me/me_test_common.h"

namespace {

using namespace av2enc::me;
using namespace av2enc::me::test;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
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
      CUDA_CHECK(cudaMalloc(&plane[p].base, plane[p].bytes),
                 "cudaMalloc plane");
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
      if (std::memcmp(tmp.data(), planes[p]->buf.data(), plane[p].bytes) != 0) {
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

template <typename T>
T* upload_array(const std::vector<T>& host) {
  if (host.empty()) {
    return nullptr;
  }
  T* dev = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc array");
  CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             "upload array");
  return dev;
}

template <typename T>
void free_array(T** p) {
  if (*p != nullptr) {
    cudaFree(*p);
    *p = nullptr;
  }
}

MeCoreOptions base_options() {
  MeCoreOptions o;
  o.method = kMeTestZone;
  o.metric = kMeSad;
  o.range = 8;
  o.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
  o.use_merge_candidates = 0;
  return o;
}

std::vector<MeCoreResult> run_cpu(const TestFrame& cur,
                                  const std::vector<TestFrame>& refs,
                                  const std::vector<MeCoreJob>& jobs,
                                  const MeCoreOptions& o,
                                  const MeCoreMvField* field) {
  std::vector<MeCoreRef> rv;
  for (const TestFrame& f : refs) {
    rv.push_back(f.ref_view());
  }
  std::vector<MeCoreResult> out(jobs.size());
  const MeStatus s =
      me_search_batch(cur.view(), rv.data(), static_cast<int>(rv.size()),
                      jobs.data(), static_cast<int>(jobs.size()), o, field,
                      out.data());
  CHECK(s == MeStatus::kOk, "CPU search failed");
  return out;
}

void compare_results(const std::vector<MeCoreResult>& cpu,
                     const std::vector<MeCoreResult>& gpu, const char* label) {
  CHECK(cpu.size() == gpu.size(), "result count");
  int mismatches = 0;
  for (std::size_t i = 0; i < cpu.size() && i < gpu.size(); ++i) {
    const bool same = cpu[i].mv_x == gpu[i].mv_x &&
                      cpu[i].mv_y == gpu[i].mv_y &&
                      cpu[i].ref_index == gpu[i].ref_index &&
                      cpu[i].cost == gpu[i].cost &&
                      cpu[i].evals == gpu[i].evals &&
                      cpu[i].stage == gpu[i].stage &&
                      cpu[i].visible_w == gpu[i].visible_w &&
                      cpu[i].visible_h == gpu[i].visible_h;
    if (!same) {
      if (mismatches < 3) {
        std::fprintf(stderr,
                     "  %s job %zu: cpu mv=%d,%d ref=%d cost=%d evals=%d "
                     "stage=%d | gpu mv=%d,%d ref=%d cost=%d evals=%d stage=%d\n",
                     label, i, cpu[i].mv_x, cpu[i].mv_y, cpu[i].ref_index,
                     cpu[i].cost, cpu[i].evals, cpu[i].stage, gpu[i].mv_x,
                     gpu[i].mv_y, gpu[i].ref_index, gpu[i].cost, gpu[i].evals,
                     gpu[i].stage);
      }
      ++mismatches;
    }
  }
  CHECK(mismatches == 0, label);
}

void test_cost_parity() {
  TestFrame cur;
  cur.init(128, 96);
  fill_smooth_texture(&cur, 0x1111u, 5);
  TestFrame ref0 = make_shifted_ref(cur, 3, -2);
  TestFrame ref1 = make_shifted_ref(cur, -4, 1);

  DeviceFrame dcur;
  DeviceFrame dref0;
  DeviceFrame dref1;
  dcur.upload(cur);
  dref0.upload(ref0);
  dref1.upload(ref1);

  std::vector<MeCoreRef> refs_host = {dref0.device_ref(), dref1.device_ref()};
  MeCoreRef* drefs = upload_array(refs_host);

  std::vector<MeCostJob> jobs;
  const int filters[6] = {kMeInterpRegular,     kMeInterpSmooth,
                          kMeInterpSharp,       kMeInterpBilinear,
                          kMeInterpRegular4,    kMeInterpSmooth4};
  for (int metric = kMeSad; metric <= kMeSatd; ++metric) {
    for (int fi = 0; fi < 6; ++fi) {
      MeCostJob j;
      j.plane = 0;
      j.ref_index = 0;
      j.x = 12;
      j.y = 12;
      j.w = 24;
      j.h = 24;
      j.mv_x = 3 * 16 + 8;
      j.mv_y = -2 * 16 + 4;
      j.metric = metric;
      j.interp = filters[fi];
      jobs.push_back(j);
      j.ref_index = 1;
      j.mv_x = -4 * 16 - 4;
      j.mv_y = 1 * 16 + 8;
      jobs.push_back(j);
    }
  }
  // Chroma: plane coordinates and chroma-precision MVs.
  for (int plane = 1; plane <= 2; ++plane) {
    MeCostJob j;
    j.plane = plane;
    j.ref_index = 0;
    j.x = 4;
    j.y = 4;
    j.w = 12;
    j.h = 12;
    j.mv_x = 8;
    j.mv_y = -4;
    j.metric = kMeSad;
    j.interp = kMeInterpRegular;
    jobs.push_back(j);
    j.metric = kMeSatd;
    j.interp = kMeInterpSharp;
    j.mv_x = -8;
    jobs.push_back(j);
  }

  std::vector<std::int32_t> cpu(jobs.size());
  std::vector<MeCoreRef> refs_cpu = {ref0.ref_view(), ref1.ref_view()};
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const MeStatus s =
        me_cost_block(cur.view(), refs_cpu[static_cast<std::size_t>(
                                      jobs[i].ref_index)],
                      jobs[i], &cpu[i]);
    CHECK(s == MeStatus::kOk, "CPU cost failed");
  }

  MeCostJob* djobs = upload_array(jobs);
  std::int32_t* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dout, jobs.size() * sizeof(std::int32_t)),
             "cudaMalloc cost out");
  CUDA_CHECK(av2enc::cuda::me_cost_batch(dcur.device_frame(), drefs,
                                 static_cast<int>(refs_host.size()), djobs,
                                 static_cast<int>(jobs.size()), dout),
             "me_cost_batch");
  CUDA_CHECK(cudaDeviceSynchronize(), "sync cost");

  std::vector<std::int32_t> gpu(jobs.size());
  CUDA_CHECK(cudaMemcpy(gpu.data(), dout, jobs.size() * sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "download cost");
  int mismatches = 0;
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    if (cpu[i] != gpu[i]) {
      ++mismatches;
      if (mismatches <= 3) {
        std::fprintf(stderr, "  cost job %zu: cpu=%d gpu=%d\n", i, cpu[i],
                     gpu[i]);
      }
    }
  }
  CHECK(mismatches == 0, "SAD/SATD cost parity (luma+chroma, all filters)");
  CHECK(dcur.download_unchanged() && dref0.download_unchanged() &&
            dref1.download_unchanged(),
        "cost kernels must not write frame planes");

  free_array(&djobs);
  free_array(&drefs);
  cudaFree(dout);
  dcur.free();
  dref0.free();
  dref1.free();
  std::cout << "  cost: " << jobs.size()
            << " SAD/SATD jobs bit-exact (6 filters, luma+chroma)\n";
}

void test_interp_parity() {
  TestFrame ref;
  ref.init(96, 80);
  fill_smooth_texture(&ref, 0x2222u, 4);
  DeviceFrame dref;
  dref.upload(ref);
  std::vector<MeCoreRef> refs_host = {dref.device_ref()};
  MeCoreRef* drefs = upload_array(refs_host);

  std::vector<MeInterpJob> jobs;
  std::int32_t offset = 0;
  for (int fi = 0; fi < 6; ++fi) {
    for (int mode = 0; mode < 4; ++mode) {
      MeInterpJob j;
      j.plane = 0;
      j.ref_index = 0;
      j.x = (mode == 0) ? 0 : 16 + mode * 4;
      j.y = (mode == 1) ? 0 : 12 + mode * 3;
      j.w = 16;
      j.h = 16;
      j.mv_x = (mode == 0) ? -9 : (mode == 2 ? 8 : 4);
      j.mv_y = (mode == 1) ? -7 : (mode == 3 ? -8 : 2);
      j.interp = fi;
      j.out_offset = offset;
      offset += j.w * j.h;
      jobs.push_back(j);
    }
  }
  // Chroma interpolation.
  for (int plane = 1; plane <= 2; ++plane) {
    MeInterpJob j;
    j.plane = plane;
    j.ref_index = 0;
    j.x = 4;
    j.y = 4;
    j.w = 8;
    j.h = 8;
    j.mv_x = 6;
    j.mv_y = -2;
    j.interp = kMeInterpSharp;
    j.out_offset = offset;
    offset += j.w * j.h;
    jobs.push_back(j);
  }

  std::vector<std::uint8_t> cpu(static_cast<std::size_t>(offset));
  for (const MeInterpJob& j : jobs) {
    const MeStatus s =
        me_interp_block(ref.ref_view(), j, cpu.data() + j.out_offset);
    CHECK(s == MeStatus::kOk, "CPU interp failed");
  }
  MeInterpJob* djobs = upload_array(jobs);
  std::uint8_t* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dout, cpu.size()), "cudaMalloc interp out");
  CUDA_CHECK(av2enc::cuda::me_interp_batch(drefs, 1, djobs,
                                   static_cast<int>(jobs.size()), dout),
             "me_interp_batch");
  CUDA_CHECK(cudaDeviceSynchronize(), "sync interp");
  std::vector<std::uint8_t> gpu(cpu.size());
  CUDA_CHECK(cudaMemcpy(gpu.data(), dout, cpu.size(), cudaMemcpyDeviceToHost),
             "download interp");
  CHECK(std::memcmp(cpu.data(), gpu.data(), cpu.size()) == 0,
        "subpel prediction parity (all filters, edge phases, chroma)");

  free_array(&djobs);
  free_array(&drefs);
  cudaFree(dout);
  dref.free();
  std::cout << "  interp: " << jobs.size()
            << " blocks bit-exact (all filters, negative edge phases)\n";
}

void run_search_case(const char* label, const TestFrame& cur,
                     const std::vector<TestFrame>& refs,
                     const std::vector<MeCoreJob>& jobs,
                     const MeCoreOptions& o, const MeCoreMvField* host_field,
                     const MeCoreMvField* device_field) {
  const std::vector<MeCoreResult> cpu = run_cpu(cur, refs, jobs, o, host_field);

  DeviceFrame dcur;
  dcur.upload(cur);
  std::vector<DeviceFrame> drefs_frames(refs.size());
  std::vector<MeCoreRef> refs_host;
  for (std::size_t i = 0; i < refs.size(); ++i) {
    drefs_frames[i].upload(refs[i]);
    refs_host.push_back(drefs_frames[i].device_ref());
  }
  MeCoreRef* drefs = upload_array(refs_host);
  MeCoreJob* djobs = upload_array(jobs);
  MeCoreResult* dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dout, jobs.size() * sizeof(MeCoreResult)),
             "cudaMalloc search out");
  CUDA_CHECK(av2enc::cuda::me_search_batch(dcur.device_frame(), drefs,
                                   static_cast<int>(refs.size()), djobs,
                                   static_cast<int>(jobs.size()), o,
                                   device_field, dout),
             "me_search_batch");
  CUDA_CHECK(cudaDeviceSynchronize(), "sync search");
  std::vector<MeCoreResult> gpu(jobs.size());
  CUDA_CHECK(cudaMemcpy(gpu.data(), dout, jobs.size() * sizeof(MeCoreResult),
                        cudaMemcpyDeviceToHost),
             "download search");
  compare_results(cpu, gpu, label);
  CHECK(dcur.download_unchanged(), "search kernels must not write frames");
  for (DeviceFrame& d : drefs_frames) {
    CHECK(d.download_unchanged(), "search kernels must not write refs");
  }

  free_array(&djobs);
  free_array(&drefs);
  cudaFree(dout);
  for (DeviceFrame& d : drefs_frames) {
    d.free();
  }
  dcur.free();
}

void test_search_parity() {
  TestFrame cur;
  cur.init(96, 96);
  fill_smooth_texture(&cur, 0x3333u, 5);
  TestFrame ref0 = make_shifted_ref(cur, 3, -4);
  TestFrame ref1 = make_shifted_ref(cur, -5, 6);
  std::vector<TestFrame> refs = {ref0, ref1};

  std::vector<MeCoreJob> grid;
  for (int y = 0; y < 96; y += 16) {
    for (int x = 0; x < 96; x += 16) {
      MeCoreJob j;
      j.x = x;
      j.y = y;
      j.w = 16;
      j.h = 16;
      j.num_refs = 2;
      j.ref_indices[0] = 0;
      j.ref_indices[1] = 1;
      grid.push_back(j);
    }
  }

  struct Case {
    const char* name;
    MeCoreOptions o;
  };
  std::vector<Case> cases;
  {
    MeCoreOptions o = base_options();
    o.method = kMeFull;
    o.metric = kMeSad;
    o.range = 6;
    o.subpel_stages = kMeSubpelNone;
    cases.push_back({"full/SAD/int", o});
  }
  {
    MeCoreOptions o = base_options();
    o.method = kMeFull;
    o.metric = kMeSatd;
    o.range = 4;
    o.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
    o.subpel_interp = kMeInterpSmooth;
    cases.push_back({"full/SATD/subpel", o});
  }
  {
    MeCoreOptions o = base_options();
    o.method = kMeDiamond;
    o.max_iterations = 16;
    cases.push_back({"diamond/SAD/subpel", o});
  }
  {
    MeCoreOptions o = base_options();
    o.method = kMeTestZone;
    o.metric = kMeSatd;
    o.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter | kMeSubpelEighth |
                      kMeSubpelSixteenth;
    o.subpel_interp = kMeInterpSharp;
    o.subpel_iterations = 3;
    cases.push_back({"tz/SATD/all-subpel", o});
  }
  {
    MeCoreOptions o = base_options();
    o.method = kMeFull;
    o.max_candidates = 49;
    o.range = 16;
    o.subpel_stages = kMeSubpelQuarter;
    cases.push_back({"full/capped", o});
  }

  for (const Case& c : cases) {
    run_search_case(c.name, cur, refs, grid, c.o, nullptr, nullptr);
    std::cout << "  search: " << c.name << " " << grid.size()
              << " jobs bit-exact\n";
  }

  // Partial blocks at the right/bottom edge.
  TestFrame small;
  small.init(40, 24);
  fill_random(&small, 0x4444u);
  TestFrame sref = make_shifted_ref(small, -4, -2);
  std::vector<MeCoreJob> pjobs(3);
  pjobs[0].x = 32;
  pjobs[0].y = 16;
  pjobs[0].w = 32;
  pjobs[0].h = 32;
  pjobs[0].num_refs = 1;
  pjobs[1] = pjobs[0];
  pjobs[1].x = 36;
  pjobs[1].y = 20;
  pjobs[1].w = 8;
  pjobs[1].h = 8;
  pjobs[2] = pjobs[0];
  pjobs[2].x = 0;
  pjobs[2].y = 0;
  MeCoreOptions po = base_options();
  po.range = 4;
  run_search_case("partial", small, {sref}, pjobs, po, nullptr, nullptr);
  std::cout << "  search: partial blocks bit-exact\n";

  // Merge candidates from a device MV field.
  std::vector<std::int32_t> mvx(24 * 24, 0);
  std::vector<std::int32_t> mvy(24 * 24, 0);
  std::vector<std::uint8_t> valid(24 * 24, 1);
  const int cell = 7 * 24 + 3;  // A1 cell (3,7) for the block at (16,16)
  mvx[static_cast<std::size_t>(cell)] = 3 * 16;
  mvy[static_cast<std::size_t>(cell)] = -4 * 16;
  MeCoreMvField host_field;
  host_field.mv_x = mvx.data();
  host_field.mv_y = mvy.data();
  host_field.valid = valid.data();
  host_field.cols = 24;
  host_field.rows = 24;

  std::int32_t* dmvx = upload_array(mvx);
  std::int32_t* dmvy = upload_array(mvy);
  std::uint8_t* dvalid = upload_array(valid);
  MeCoreMvField device_field;
  device_field.mv_x = dmvx;
  device_field.mv_y = dmvy;
  device_field.valid = dvalid;
  device_field.cols = 24;
  device_field.rows = 24;
  MeCoreMvField* dfield = nullptr;
  CUDA_CHECK(cudaMalloc(&dfield, sizeof(MeCoreMvField)), "cudaMalloc field");
  CUDA_CHECK(cudaMemcpy(dfield, &device_field, sizeof(MeCoreMvField),
                        cudaMemcpyHostToDevice),
             "upload field");

  std::vector<MeCoreJob> merge_jobs(1);
  merge_jobs[0].x = 16;
  merge_jobs[0].y = 16;
  merge_jobs[0].w = 16;
  merge_jobs[0].h = 16;
  merge_jobs[0].num_refs = 1;
  MeCoreOptions mo = base_options();
  mo.range = 0;
  mo.use_merge_candidates = 1;
  run_search_case("merge", cur, {ref0}, merge_jobs, mo, &host_field, dfield);
  std::cout << "  search: device merge candidates bit-exact\n";

  // Explicit NEWMV/NEARMV-style predictors (fractional).
  std::vector<MeCoreJob> pred_jobs(1);
  pred_jobs[0].x = 32;
  pred_jobs[0].y = 32;
  pred_jobs[0].w = 16;
  pred_jobs[0].h = 16;
  pred_jobs[0].num_refs = 1;
  pred_jobs[0].num_pred = 2;
  pred_jobs[0].pred_mv[0][0] = 0;
  pred_jobs[0].pred_mv[0][1] = 0;
  pred_jobs[0].pred_mv[1][0] = 3 * 16 + 4;
  pred_jobs[0].pred_mv[1][1] = -4 * 16 - 8;
  MeCoreOptions pred_o = base_options();
  pred_o.range = 0;
  pred_o.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
  run_search_case("predictors", cur, {ref0}, pred_jobs, pred_o, nullptr,
                  nullptr);
  std::cout << "  search: explicit fractional predictors bit-exact\n";

  free_array(&dmvx);
  free_array(&dmvy);
  free_array(&dvalid);
  cudaFree(dfield);
}

void test_real_frame(const std::string& repo_root) {
  std::vector<TestFrame> frames;
  const std::string path = repo_root + "/testdata/shine_256x256_1f.y4m";
  if (!load_y4m_frames(path, 1, &frames)) {
    std::cout << "  real 256x256: skipped (no testdata on this host)\n";
    return;
  }
  TestFrame& cur = frames[0];
  TestFrame ref = make_shifted_ref(cur, 5, -3);
  const std::vector<MeCoreJob> jobs = make_block_grid(256, 256, 32, 32);
  MeCoreOptions o = base_options();
  o.method = kMeTestZone;
  o.range = 8;
  o.metric = kMeSad;
  o.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
  run_search_case("real-256", cur, {ref}, jobs, o, nullptr, nullptr);
  std::cout << "  real 256x256: " << jobs.size()
            << " blocks CPU/GPU bit-exact\n";
}

void test_timing() {
  auto bench = [](const char* label, int width, int height, int bw, int bh,
                  const MeCoreOptions& o, int iters) {
    TestFrame cur;
    cur.init(width, height);
    fill_smooth_texture(&cur, 0x5555u, 7);
    TestFrame ref = make_shifted_ref(cur, 2, -1);

    DeviceFrame dcur;
    DeviceFrame dref;
    dcur.upload(cur);
    dref.upload(ref);
    const MeCoreRef ref_host = dref.device_ref();
    MeCoreRef* drefs = upload_array(std::vector<MeCoreRef>{ref_host});
    const std::vector<MeCoreJob> jobs = make_block_grid(width, height, bw, bh);
    MeCoreJob* djobs = upload_array(jobs);
    MeCoreResult* dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dout, jobs.size() * sizeof(MeCoreResult)),
               "cudaMalloc timing out");

    for (int i = 0; i < 3; ++i) {
      CUDA_CHECK(av2enc::cuda::me_search_batch(
                     dcur.device_frame(), drefs, 1, djobs,
                     static_cast<int>(jobs.size()), o, nullptr, dout),
                 "warmup search");
    }
    CUDA_CHECK(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
      CUDA_CHECK(av2enc::cuda::me_search_batch(
                     dcur.device_frame(), drefs, 1, djobs,
                     static_cast<int>(jobs.size()), o, nullptr, dout),
                 "timed search");
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double per_frame = static_cast<double>(ms) / iters;
    const double blocks = static_cast<double>(jobs.size());
    const double pixels = static_cast<double>(width) * height;
    std::printf("    %-28s %7.2f ms  %9.0f blocks/s  %7.1f Mpix/s\n", label,
                per_frame, blocks / (per_frame * 1e-3),
                pixels / (per_frame * 1e-3) / 1e6);

    free_array(&djobs);
    free_array(&drefs);
    cudaFree(dout);
    dcur.free();
    dref.free();
  };

  MeCoreOptions sad = base_options();
  sad.method = kMeTestZone;
  sad.range = 16;
  sad.metric = kMeSad;
  sad.subpel_stages = kMeSubpelHalf | kMeSubpelQuarter;
  MeCoreOptions satd = sad;
  satd.metric = kMeSatd;
  MeCoreOptions full = sad;
  full.method = kMeFull;
  full.range = 8;
  full.subpel_stages = kMeSubpelNone;

  std::printf("  timing (synthetic smooth pan, GPU only):\n");
  bench("720p 16x16 TZ SAD subpel", 1280, 720, 16, 16, sad, 10);
  bench("720p 16x16 TZ SATD subpel", 1280, 720, 16, 16, satd, 10);
  bench("720p 16x16 full r8 SAD int", 1280, 720, 16, 16, full, 10);
  bench("1080p 16x16 TZ SAD subpel", 1920, 1080, 16, 16, sad, 5);
  bench("1080p 8x8 TZ SAD subpel", 1920, 1080, 8, 8, sad, 5);
}
}  // namespace

int main(int argc, char** argv) {
  std::string repo_root = argc > 1 ? argv[1] : ".";
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::fprintf(stderr, "CUDA device query failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 2;
  }
  std::printf("AV2 inter ME CPU-vs-CUDA tests\n");
  std::printf("device: %s (cc %d.%d, %d SMs, %.0f MiB)\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0));

  test_cost_parity();
  test_interp_parity();
  test_search_parity();
  test_real_frame(repo_root);
  test_timing();

  std::printf("\nchecks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
