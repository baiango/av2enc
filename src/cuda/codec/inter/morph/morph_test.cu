// CPU-vs-GPU bit-exactness test for the AV2 morphological prediction process.
//
// Every case builds a deterministic luma plane, runs the CPU reference
// (src/encoder/inter/morph/morph_common.h) and the CUDA kernel on the same
// input, and requires identical accept/reject decisions, fit parameters,
// per-sample output bytes, and per-sample displacement field.
//
// Also exercises:
//   * the single-job and batch entry points;
//   * the persistent MorphWorkspace (arena sizes exactly required_device_bytes);
//   * two mark()/rewind() staged launches with no net arena growth;
//   * a one-byte-short arena for both the workspace and the staged path.
//
// Device memory comes only from av2enc::cuda::Arena. Build/run:
//   ./build_gpu_test.sh            # sm_86, needs nvcc + a CUDA device
//
// Local host-only pre-flight: ./build_and_test.sh

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/codec/inter/morph/morph.h"
#include "cuda/codec/inter/morph/morph_test_cases.h"
#include "codec/inter/morph/morph_common.h"

namespace cuda = av2enc::cuda;
namespace gpu = av2enc::cuda::morph;
namespace ref = av2enc::inter::morph;
using gpu::testcases::Case;

namespace {

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

void cuda_check(cudaError_t error, const char* what) {
  check(error == cudaSuccess, std::string(what) + ": " + cudaGetErrorString(error));
}

ref::MorphFit sentinel_fit() {
  ref::MorphFit fit;
  fit.alpha = -12345;
  fit.beta = -12346;
  fit.count = -12347;
  fit.ref_x = -12348;
  fit.ref_y = -12349;
  fit.bw = -12350;
  fit.bh = -12351;
  fit.plan.width = -12352;
  fit.plan.height = -12353;
  fit.plan.num_up = -12354;
  fit.plan.num_left = -12355;
  fit.plan.up_step = -12356;
  fit.plan.left_step = -12357;
  return fit;
}

bool fit_equal(const ref::MorphFit& a, const ref::MorphFit& b) {
  return a.alpha == b.alpha && a.beta == b.beta && a.count == b.count &&
         a.ref_x == b.ref_x && a.ref_y == b.ref_y && a.bw == b.bw &&
         a.bh == b.bh && a.plan.width == b.plan.width &&
         a.plan.height == b.plan.height && a.plan.num_up == b.plan.num_up &&
         a.plan.num_left == b.plan.num_left &&
         a.plan.up_step == b.plan.up_step &&
         a.plan.left_step == b.plan.left_step;
}

struct BatchHost {
  std::vector<Case> cases;
  std::vector<std::vector<std::uint8_t>> input;
  std::vector<std::vector<std::uint8_t>> cpu_plane;
  std::vector<ref::MorphFit> cpu_fits;
  std::vector<bool> cpu_ok;
  std::vector<gpu::MorphJob> jobs;
  std::size_t plane_bytes = 0;
  std::vector<std::size_t> plane_offset;
};

BatchHost build_batch(int max_jobs) {
  BatchHost batch;
  const std::vector<Case> all = gpu::testcases::all_cases();
  for (const Case& c : all) {
    if (static_cast<int>(batch.jobs.size()) >= max_jobs) {
      break;
    }
    if (c.plane_w != 256 || c.plane_h != 256 || c.rect.w > 32 ||
        c.rect.h > 32 || c.rect.x < 0 || c.rect.y < 0) {
      continue;
    }
    std::vector<std::uint8_t> input = gpu::testcases::make_plane(c);
    std::vector<std::uint8_t> cpu_plane = input;
    ref::MorphFit cpu_fit = sentinel_fit();
    const bool cpu_ok = ref::morph_predict(
        c.ctx, ref::MorphPlane{cpu_plane.data(), c.stride}, c.rect, c.mv_row,
        c.mv_col, &cpu_fit);

    gpu::MorphJob job;
    job.ctx = c.ctx;
    job.rect = c.rect;
    job.mv_row = c.mv_row;
    job.mv_col = c.mv_col;
    job.stride = c.stride;

    batch.cases.push_back(c);
    batch.plane_offset.push_back(batch.plane_bytes);
    batch.plane_bytes += static_cast<std::size_t>(c.stride) *
                         static_cast<std::size_t>(c.rows);
    batch.input.push_back(std::move(input));
    batch.cpu_plane.push_back(std::move(cpu_plane));
    batch.cpu_fits.push_back(cpu_fit);
    batch.cpu_ok.push_back(cpu_ok);
    batch.jobs.push_back(job);
  }
  return batch;
}

std::size_t max_plane_bytes(const std::vector<Case>& cases) {
  std::size_t max_bytes = 0;
  for (const Case& c : cases) {
    const std::size_t bytes = static_cast<std::size_t>(c.stride) *
                              static_cast<std::size_t>(c.rows);
    max_bytes = std::max(max_bytes, bytes);
  }
  return max_bytes;
}

void compare_fit(const ref::MorphFit& got, const ref::MorphFit& want,
                 const std::string& what) {
  check(fit_equal(got, want), what);
}

// Single-job parity over every shared case.
void run_single_cases(std::uint8_t* d_plane, std::size_t d_plane_bytes,
                      ref::MorphFit* d_fit, int* d_status,
                      std::int16_t* d_field, std::size_t field_samples,
                      int* successes, int* rejections, int* ten_bit) {
  const std::vector<Case> cases = gpu::testcases::all_cases();
  for (const Case& c : cases) {
    const std::vector<std::uint8_t> input = gpu::testcases::make_plane(c);
    const std::size_t plane_bytes = static_cast<std::size_t>(c.stride) *
                                    static_cast<std::size_t>(c.rows);
    check(plane_bytes <= d_plane_bytes, "test plane buffer is large enough");
    if (plane_bytes > d_plane_bytes) {
      continue;
    }
    std::vector<std::uint8_t> cpu_plane = input;
    ref::MorphFit cpu_fit = sentinel_fit();
    const bool cpu_ok = ref::morph_predict(
        c.ctx, ref::MorphPlane{cpu_plane.data(), c.stride}, c.rect, c.mv_row,
        c.mv_col, &cpu_fit);

    const int samples = c.rect.w * c.rect.h;
    check(static_cast<std::size_t>(samples) <= field_samples,
          "field buffer is large enough");

    const ref::MorphFit poison = sentinel_fit();
    cuda_check(cudaMemcpy(d_plane, input.data(), plane_bytes,
                          cudaMemcpyHostToDevice),
               "upload plane");
    cuda_check(cudaMemcpy(d_fit, &poison, sizeof(ref::MorphFit),
                          cudaMemcpyHostToDevice),
               "poison fit");
    cuda_check(cudaMemset(d_status, 0xFF, sizeof(int)), "poison status");
    cuda_check(cudaMemset(d_field, 0x7F,
                          static_cast<std::size_t>(samples) * 2 *
                              sizeof(std::int16_t)),
               "poison field");

    gpu::MorphJob job;
    job.ctx = c.ctx;
    job.rect = c.rect;
    job.mv_row = c.mv_row;
    job.mv_col = c.mv_col;
    job.frame = d_plane;
    job.stride = c.stride;
    job.fit_out = d_fit;
    job.status_out = d_status;
    job.field_out = d_field;
    cuda_check(gpu::morph_predict(job), "morph_predict launch");
    cuda_check(cudaDeviceSynchronize(), "morph_predict sync");

    int status = -1;
    ref::MorphFit gpu_fit = sentinel_fit();
    cuda_check(cudaMemcpy(&status, d_status, sizeof(int),
                          cudaMemcpyDeviceToHost),
               "download status");
    cuda_check(cudaMemcpy(&gpu_fit, d_fit, sizeof(ref::MorphFit),
                          cudaMemcpyDeviceToHost),
               "download fit");
    std::vector<std::uint8_t> gpu_plane(plane_bytes, 0);
    cuda_check(cudaMemcpy(gpu_plane.data(), d_plane, plane_bytes,
                          cudaMemcpyDeviceToHost),
               "download plane");

    const bool gpu_ok = status != 0;
    check(gpu_ok == cpu_ok, std::string("accept/reject: ") + c.name);
    if (cpu_ok) {
      ++(*successes);
      compare_fit(gpu_fit, cpu_fit, std::string("fit: ") + c.name);
      check(gpu_plane == cpu_plane, std::string("samples: ") + c.name);
      std::vector<std::int16_t> field(
          static_cast<std::size_t>(samples) * 2, 0);
      cuda_check(cudaMemcpy(field.data(), d_field,
                            field.size() * sizeof(std::int16_t),
                            cudaMemcpyDeviceToHost),
                 "download field");
      bool field_ok = true;
      for (int s = 0; s < samples; ++s) {
        const std::int16_t dx =
            static_cast<std::int16_t>(ref::to_fullmv(c.mv_col));
        const std::int16_t dy =
            static_cast<std::int16_t>(ref::to_fullmv(c.mv_row));
        field_ok = field_ok &&
                   field[static_cast<std::size_t>(s) * 2] == dx &&
                   field[static_cast<std::size_t>(s) * 2 + 1] == dy;
      }
      check(field_ok, std::string("field: ") + c.name);
    } else {
      ++(*rejections);
      compare_fit(gpu_fit, sentinel_fit(),
                  std::string("fit written on reject: ") + c.name);
      check(gpu_plane == cpu_plane,
            std::string("plane changed on reject: ") + c.name);
      std::vector<std::int16_t> field(
          static_cast<std::size_t>(samples) * 2, 0);
      if (samples > 0) {
        cuda_check(cudaMemcpy(field.data(), d_field,
                              field.size() * sizeof(std::int16_t),
                              cudaMemcpyDeviceToHost),
                   "download rejected field");
      }
      bool touched = false;
      for (std::int16_t value : field) {
        touched = touched || value != 0x7f7f;
      }
      check(!touched, std::string("field written on reject: ") + c.name);
    }
    if (c.ctx.bit_depth != 8) {
      ++(*ten_bit);
      check(!cpu_ok, std::string("bit_depth 10 must reject: ") + c.name);
    }
  }
}

void test_null_inputs(gpu::MorphJob base, std::uint8_t* d_plane, int* d_status) {
  ref::MorphContext ctx;
  ctx.mi_cols = 64;
  ctx.mi_rows = 64;
  const ref::MorphRect rect{32, 32, 16, 16};
  check(!ref::morph_predict(ctx, ref::MorphPlane{nullptr, 0}, rect, 0, 0,
                            nullptr),
        "CPU rejects a null plane");

  gpu::MorphJob job = base;
  job.ctx = ctx;
  job.rect = rect;
  job.frame = nullptr;
  job.stride = 0;
  job.fit_out = nullptr;
  job.status_out = d_status;
  job.field_out = nullptr;
  cuda_check(gpu::morph_predict(job), "null-plane launch");
  cuda_check(cudaDeviceSynchronize(), "null-plane sync");
  int status = -1;
  cuda_check(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost),
             "null-plane status");
  check(status == 0, "null plane reports status 0");

  job.frame = d_plane;
  job.stride = 0;
  status = -1;
  cuda_check(gpu::morph_predict(job), "zero-stride launch");
  cuda_check(cudaDeviceSynchronize(), "zero-stride sync");
  cuda_check(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost),
             "zero-stride status");
  check(status == 0, "zero stride reports status 0");
}

void prepare_batch_planes(BatchHost& batch, std::uint8_t* d_batch_planes,
                          const std::vector<std::size_t>& offsets) {
  const int count = static_cast<int>(batch.jobs.size());
  for (int i = 0; i < count; ++i) {
    batch.jobs[static_cast<std::size_t>(i)].frame =
        d_batch_planes + offsets[static_cast<std::size_t>(i)];
  }
  for (int i = 0; i < count; ++i) {
    const Case& c = batch.cases[static_cast<std::size_t>(i)];
    const std::size_t bytes = static_cast<std::size_t>(c.stride) *
                              static_cast<std::size_t>(c.rows);
    cuda_check(cudaMemcpy(batch.jobs[static_cast<std::size_t>(i)].frame,
                          batch.input[static_cast<std::size_t>(i)].data(),
                          bytes, cudaMemcpyHostToDevice),
               "upload batch plane");
  }
}

void test_batches(cuda::Arena& arena, gpu::MorphWorkspace& workspace,
                  BatchHost& batch, std::uint8_t* d_batch_planes,
                  const std::vector<std::size_t>& offsets) {
  const int count = static_cast<int>(batch.jobs.size());
  check(count > 0, "batch has jobs");
  prepare_batch_planes(batch, d_batch_planes, offsets);

  // Workspace path.
  check(gpu::morph_upload_jobs(workspace, batch.jobs.data(), count, nullptr) ==
            cuda::Status::kOk,
        "workspace upload");
  check(gpu::morph_run_batch(workspace, count, nullptr) == cuda::Status::kOk,
        "workspace run");
  cuda_check(cudaDeviceSynchronize(), "workspace sync");
  std::vector<int> workspace_status(static_cast<std::size_t>(count), -1);
  std::vector<ref::MorphFit> workspace_fits(static_cast<std::size_t>(count),
                                            sentinel_fit());
  cuda_check(cudaMemcpy(workspace_status.data(), workspace.statuses(),
                        workspace_status.size() * sizeof(int),
                        cudaMemcpyDeviceToHost),
             "download workspace statuses");
  cuda_check(cudaMemcpy(workspace_fits.data(), workspace.fits(),
                        workspace_fits.size() * sizeof(ref::MorphFit),
                        cudaMemcpyDeviceToHost),
             "download workspace fits");
  for (int i = 0; i < count; ++i) {
    const Case& c = batch.cases[static_cast<std::size_t>(i)];
    const std::size_t bytes = static_cast<std::size_t>(c.stride) *
                              static_cast<std::size_t>(c.rows);
    std::vector<std::uint8_t> got(bytes, 0);
    cuda_check(cudaMemcpy(got.data(),
                          batch.jobs[static_cast<std::size_t>(i)].frame, bytes,
                          cudaMemcpyDeviceToHost),
               "download batch plane");
    check((workspace_status[static_cast<std::size_t>(i)] != 0) ==
              batch.cpu_ok[static_cast<std::size_t>(i)],
          "workspace accept/reject");
    if (batch.cpu_ok[static_cast<std::size_t>(i)]) {
      compare_fit(workspace_fits[static_cast<std::size_t>(i)],
                  batch.cpu_fits[static_cast<std::size_t>(i)],
                  "workspace fit");
      check(got == batch.cpu_plane[static_cast<std::size_t>(i)],
            "workspace samples");
    }
  }

  // Staged path: two launches with mark/rewind and no net growth.
  prepare_batch_planes(batch, d_batch_planes, offsets);
  const std::uint64_t mark = arena.mark();
  std::vector<int> host_status(static_cast<std::size_t>(count), -1);
  std::vector<ref::MorphFit> host_fits(static_cast<std::size_t>(count),
                                       sentinel_fit());
  gpu::MorphStagedResult result;
  result.host_status = host_status.data();
  result.host_fits = host_fits.data();
  gpu::MorphStagedResult no_result;
  std::size_t scratch_bytes = 0;
  check(gpu::morph_stage_and_run(arena, batch.jobs.data(), count, &result,
                                 nullptr, &scratch_bytes) ==
            cuda::Status::kOk,
        "staged run 1");
  check(scratch_bytes == gpu::required_scratch_bytes(count),
        "staged scratch size");
  check(arena.mark() == mark, "staged run 1 rewinds the arena");
  const std::uint64_t peak_after_first = arena.peak_used();
  check(gpu::morph_stage_and_run(arena, batch.jobs.data(), count, &no_result,
                                 nullptr, nullptr) == cuda::Status::kOk,
        "staged run 2");
  check(arena.mark() == mark, "staged run 2 rewinds the arena");
  check(arena.peak_used() == peak_after_first, "staged peak does not grow");
  for (int i = 0; i < count; ++i) {
    check((host_status[static_cast<std::size_t>(i)] != 0) ==
              batch.cpu_ok[static_cast<std::size_t>(i)],
          "staged accept/reject");
    if (batch.cpu_ok[static_cast<std::size_t>(i)]) {
      compare_fit(host_fits[static_cast<std::size_t>(i)],
                  batch.cpu_fits[static_cast<std::size_t>(i)], "staged fit");
    }
  }
}

void test_arena_shortfalls(const gpu::MorphArenaConfig& config,
                           const BatchHost& batch, std::size_t required,
                           std::size_t scratch) {
  std::string error;
  {
    cuda::Arena short_arena;
    check(short_arena.init(required - 1, 256, &error) == cuda::Status::kOk,
          "one-byte-short arena init: " + error);
    gpu::MorphWorkspace short_workspace;
    const cuda::Status status = short_workspace.init(short_arena, config, &error);
    check(status == cuda::Status::kArenaExhausted,
          "one-byte-short workspace reports kArenaExhausted");
    check(short_workspace.required_bytes() == required,
          "one-byte-short workspace reports required bytes");
    check(short_arena.mark() == 0, "short workspace leaves the arena untouched");
    check(!error.empty(), "short workspace sets an error string");
  }
  {
    const int count = static_cast<int>(std::min<std::size_t>(batch.jobs.size(), 4));
    const std::size_t need = gpu::required_scratch_bytes(count);
    cuda::Arena short_arena;
    check(short_arena.init(need - 1, 256, &error) == cuda::Status::kOk,
          "one-byte-short scratch arena init: " + error);
    std::size_t reported = 0;
    const cuda::Status status = gpu::morph_stage_and_run(
        short_arena, batch.jobs.data(), count, nullptr, nullptr, &reported);
    check(status == cuda::Status::kArenaExhausted,
          "one-byte-short staging reports kArenaExhausted");
    check(reported == need, "one-byte-short staging reports required bytes");
    check(short_arena.mark() == 0, "short staging leaves the arena untouched");
  }
  check(scratch > 0, "scratch size is non-zero");
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::printf("device: %s (cc %d.%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  const std::vector<Case> cases = gpu::testcases::all_cases();
  check(!cases.empty(), "case list is non-empty");

  gpu::MorphArenaConfig config;
  config.max_jobs = 16;
  config.max_field_samples = 4096;
  const std::size_t required = gpu::required_device_bytes(config);
  const std::size_t scratch = gpu::required_scratch_bytes(config.max_jobs);
  check(required > 0 && scratch > 0, "required sizes are non-zero");

  BatchHost batch = build_batch(config.max_jobs);
  const std::size_t max_plane = max_plane_bytes(cases);
  const std::size_t field_bytes =
      static_cast<std::size_t>(config.max_field_samples) * 2 *
      sizeof(std::int16_t);
  const std::size_t batch_arrays =
      static_cast<std::size_t>(config.max_jobs) *
      (sizeof(gpu::MorphJob) + sizeof(ref::MorphFit) + sizeof(int));
  const std::size_t test_buffers = max_plane + field_bytes + batch_arrays +
                                   batch.plane_bytes + sizeof(ref::MorphFit) +
                                   sizeof(int);
  const std::size_t capacity =
      required + scratch + test_buffers + (1u << 20);

  cuda::Arena arena;
  std::string error;
  check(arena.init(capacity, 256, &error) == cuda::Status::kOk,
        "arena init: " + error);

  const std::uint64_t before = arena.mark();
  gpu::MorphWorkspace workspace;
  check(workspace.init(arena, config, &error) == cuda::Status::kOk,
        "workspace init: " + error);
  check(arena.mark() - before == static_cast<std::uint64_t>(required),
        "workspace consumes exactly required_device_bytes");
  check(workspace.required_bytes() == required, "workspace required size");
  check(workspace.field() != nullptr, "workspace field buffer");
  check(workspace.field_samples() ==
            static_cast<std::size_t>(config.max_field_samples),
        "workspace field capacity");

  std::uint8_t* d_plane = arena.allocate_array<std::uint8_t>(max_plane);
  ref::MorphFit* d_fit = arena.allocate_array<ref::MorphFit>(1);
  int* d_status = arena.allocate_array<int>(1);
  std::int16_t* d_field =
      arena.allocate_array<std::int16_t>(
          static_cast<std::size_t>(config.max_field_samples) * 2);
  std::uint8_t* d_batch_planes =
      arena.allocate_array<std::uint8_t>(batch.plane_bytes);
  gpu::MorphJob* d_jobs =
      arena.allocate_array<gpu::MorphJob>(static_cast<std::size_t>(config.max_jobs));
  ref::MorphFit* d_fits =
      arena.allocate_array<ref::MorphFit>(static_cast<std::size_t>(config.max_jobs));
  int* d_statuses =
      arena.allocate_array<int>(static_cast<std::size_t>(config.max_jobs));
  check(d_plane != nullptr && d_fit != nullptr && d_status != nullptr &&
            d_field != nullptr && d_batch_planes != nullptr &&
            d_jobs != nullptr && d_fits != nullptr && d_statuses != nullptr,
        "test buffers allocated from the arena");

  int successes = 0;
  int rejections = 0;
  int ten_bit = 0;
  run_single_cases(d_plane, max_plane, d_fit, d_status, d_field,
                   static_cast<std::size_t>(config.max_field_samples),
                   &successes, &rejections, &ten_bit);

  gpu::MorphJob null_base;
  test_null_inputs(null_base, d_plane, d_status);

  prepare_batch_planes(batch, d_batch_planes, batch.plane_offset);

  // Raw batch entry point: per-job fit/status pointers into arena arrays.
  {
    std::vector<gpu::MorphJob> jobs = batch.jobs;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      jobs[i].fit_out = d_fits + i;
      jobs[i].status_out = d_statuses + i;
    }
    cuda_check(cudaMemcpy(d_jobs, jobs.data(),
                          jobs.size() * sizeof(gpu::MorphJob),
                          cudaMemcpyHostToDevice),
               "upload raw jobs");
    cuda_check(gpu::morph_predict_batch(d_jobs, static_cast<int>(jobs.size())),
               "raw batch launch");
    cuda_check(cudaDeviceSynchronize(), "raw batch sync");
    std::vector<ref::MorphFit> fits(jobs.size(), sentinel_fit());
    std::vector<int> statuses(jobs.size(), -1);
    cuda_check(cudaMemcpy(fits.data(), d_fits,
                          fits.size() * sizeof(ref::MorphFit),
                          cudaMemcpyDeviceToHost),
               "download raw fits");
    cuda_check(cudaMemcpy(statuses.data(), d_statuses,
                          statuses.size() * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "download raw statuses");
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      check((statuses[i] != 0) == batch.cpu_ok[i], "raw batch accept/reject");
      if (batch.cpu_ok[i]) {
        compare_fit(fits[i], batch.cpu_fits[i], "raw batch fit");
      }
      const Case& c = batch.cases[i];
      const std::size_t bytes = static_cast<std::size_t>(c.stride) *
                                static_cast<std::size_t>(c.rows);
      std::vector<std::uint8_t> got(bytes, 0);
      cuda_check(cudaMemcpy(got.data(), batch.jobs[i].frame, bytes,
                            cudaMemcpyDeviceToHost),
                 "download raw batch plane");
      check(got == batch.cpu_plane[i], "raw batch samples");
    }
  }

  test_batches(arena, workspace, batch, d_batch_planes, batch.plane_offset);
  test_arena_shortfalls(config, batch, required, scratch);

  std::printf(
      "morph_cuda_test: cases=%d accepted=%d rejected=%d bit_depth10=%d "
      "checks=%d failures=%d required_device_bytes=%zu scratch=%zu "
      "arena_capacity=%zu peak_used=%llu\n",
      successes + rejections, successes, rejections, ten_bit, g_checks,
      g_failures, required, scratch, capacity,
      static_cast<unsigned long long>(arena.peak_used()));
  return g_failures == 0 ? 0 : 1;
}
