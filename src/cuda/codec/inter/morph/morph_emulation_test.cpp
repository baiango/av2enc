// Host-emulation parity test for the AV2 morphological prediction component.
//
// Compiles the shared device header (morph_device.cuh) as host code with the
// CUDA keywords stubbed, links the emulated entry points and the same
// arena/workspace host code used on the GPU build, and compares every result
// against the independent CPU reference in src/encoder/inter/morph/.
//
// This is a local pre-flight check, not a GPU result. The real gate is
// build_gpu_test.sh on the sm_86 box.
//
// Run: ./build_and_test.sh   (falls back here when nvcc is absent)

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

void run_parity_cases(int* successes, int* rejections, int* ten_bit) {
  const std::vector<gpu::testcases::Case> cases = gpu::testcases::all_cases();
  for (const gpu::testcases::Case& c : cases) {
    const std::vector<std::uint8_t> input = gpu::testcases::make_plane(c);
    std::vector<std::uint8_t> cpu_plane = input;
    std::vector<std::uint8_t> emu_plane = input;

    ref::MorphFit cpu_fit = sentinel_fit();
    const bool cpu_ok = ref::morph_predict(
        c.ctx, ref::MorphPlane{cpu_plane.data(), c.stride}, c.rect, c.mv_row,
        c.mv_col, &cpu_fit);

    const int samples = c.rect.w * c.rect.h;
    std::vector<std::int16_t> field(
        static_cast<std::size_t>(samples) * 2, static_cast<std::int16_t>(0x7f7f));
    gpu::MorphJob job;
    job.ctx = c.ctx;
    job.rect = c.rect;
    job.mv_row = c.mv_row;
    job.mv_col = c.mv_col;
    job.frame = emu_plane.data();
    job.stride = c.stride;
    ref::MorphFit emu_fit = sentinel_fit();
    int status = -1;
    job.fit_out = &emu_fit;
    job.status_out = &status;
    job.field_out = field.data();

    const cudaError_t error = gpu::morph_predict(job);
    check(error == cudaSuccess, std::string("launch: ") + c.name);

    const bool emu_ok = status != 0;
    check(emu_ok == cpu_ok, std::string("accept/reject: ") + c.name);
    if (cpu_ok) {
      ++(*successes);
      check(fit_equal(emu_fit, cpu_fit),
            std::string("fit mismatch: ") + c.name);
      check(emu_plane == cpu_plane,
            std::string("plane mismatch: ") + c.name);
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
      check(field_ok, std::string("field mismatch: ") + c.name);
    } else {
      ++(*rejections);
      check(fit_equal(emu_fit, sentinel_fit()),
            std::string("fit written on reject: ") + c.name);
      check(emu_plane == cpu_plane,
            std::string("plane changed on reject: ") + c.name);
      bool field_touched = false;
      for (std::int16_t value : field) {
        field_touched = field_touched || value != 0x7f7f;
      }
      check(!field_touched, std::string("field written on reject: ") + c.name);
    }
    if (c.ctx.bit_depth != 8) {
      ++(*ten_bit);
      check(!cpu_ok, std::string("bit_depth 10 must reject: ") + c.name);
    }
  }
}

void test_null_inputs() {
  ref::MorphContext ctx;
  ctx.mi_cols = 64;
  ctx.mi_rows = 64;
  const ref::MorphRect rect{32, 32, 16, 16};
  const bool cpu_ok =
      ref::morph_predict(ctx, ref::MorphPlane{nullptr, 0}, rect, 0, 0, nullptr);
  check(!cpu_ok, "CPU rejects a null plane");

  gpu::MorphJob job;
  job.ctx = ctx;
  job.rect = rect;
  job.frame = nullptr;
  job.stride = 0;
  int status = -1;
  job.status_out = &status;
  check(gpu::morph_predict(job) == cudaSuccess, "null-plane launch is a no-op");
  check(status == 0, "null plane reports status 0");

  std::vector<std::uint8_t> plane(256 * 256, 100);
  job.frame = plane.data();
  job.stride = 0;
  status = -1;
  check(gpu::morph_predict(job) == cudaSuccess, "zero-stride launch");
  check(status == 0, "zero stride reports status 0");
  bool unchanged = true;
  for (std::uint8_t value : plane) {
    unchanged = unchanged && value == 100;
  }
  check(unchanged, "zero stride leaves the plane untouched");
}

// A compact batch mixing accepted, rejected and 10-bit jobs.
struct BatchInput {
  std::vector<std::vector<std::uint8_t>> planes;
  std::vector<ref::MorphFit> cpu_fits;
  std::vector<bool> cpu_ok;
  std::vector<gpu::MorphJob> jobs;
  std::vector<int> statuses;
  std::vector<ref::MorphFit> fits;
};

BatchInput make_batch_input(int max_jobs) {
  const std::vector<gpu::testcases::Case> all = gpu::testcases::all_cases();
  BatchInput batch;
  for (const gpu::testcases::Case& c : all) {
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
    batch.planes.push_back(std::move(input));
    batch.cpu_fits.push_back(cpu_fit);
    batch.cpu_ok.push_back(cpu_ok);

    gpu::MorphJob job;
    job.ctx = c.ctx;
    job.rect = c.rect;
    job.mv_row = c.mv_row;
    job.mv_col = c.mv_col;
    job.stride = c.stride;
    batch.jobs.push_back(job);
  }
  batch.statuses.assign(batch.jobs.size(), -1);
  batch.fits.assign(batch.jobs.size(), sentinel_fit());
  return batch;
}

void test_raw_batch(BatchInput& batch) {
  std::vector<gpu::MorphJob> jobs = batch.jobs;
  std::vector<ref::MorphFit> fits(batch.jobs.size(), sentinel_fit());
  std::vector<int> statuses(batch.jobs.size(), -1);
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    jobs[i].frame = batch.planes[i].data();
    jobs[i].fit_out = &fits[i];
    jobs[i].status_out = &statuses[i];
  }
  check(gpu::morph_predict_batch(jobs.data(), static_cast<int>(jobs.size())) ==
            cudaSuccess,
        "raw batch launches");
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    check((statuses[i] != 0) == batch.cpu_ok[i], "raw batch accept/reject");
    if (batch.cpu_ok[i]) {
      check(fit_equal(fits[i], batch.cpu_fits[i]), "raw batch fit");
    }
  }
}

void test_workspace_and_arena() {
  gpu::MorphArenaConfig config;
  config.max_jobs = 16;
  config.max_field_samples = 4096;
  const std::size_t required = gpu::required_device_bytes(config);
  const std::size_t scratch = gpu::required_scratch_bytes(config.max_jobs);
  check(required > 0, "required_device_bytes is non-zero");
  check(scratch > 0, "required_scratch_bytes is non-zero");

  cuda::Arena arena;
  std::string error;
  const std::size_t capacity = required + scratch + (1u << 20);
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
  check(workspace.field_samples() == static_cast<std::size_t>(4096),
        "workspace field capacity");

  BatchInput batch = make_batch_input(config.max_jobs);
  check(!batch.jobs.empty(), "batch cases available");
  for (std::size_t i = 0; i < batch.jobs.size(); ++i) {
    batch.jobs[i].frame = batch.planes[i].data();
  }
  check(gpu::morph_upload_jobs(workspace, batch.jobs.data(),
                               static_cast<int>(batch.jobs.size()),
                               nullptr) == cuda::Status::kOk,
        "workspace upload");
  check(gpu::morph_run_batch(workspace, static_cast<int>(batch.jobs.size()),
                             nullptr) == cuda::Status::kOk,
        "workspace run");
  for (std::size_t i = 0; i < batch.jobs.size(); ++i) {
    check((workspace.statuses()[i] != 0) == batch.cpu_ok[i],
          "workspace accept/reject");
    if (batch.cpu_ok[i]) {
      check(fit_equal(workspace.fits()[i], batch.cpu_fits[i]),
            "workspace fit");
    }
  }
  check(arena.mark() == before + static_cast<std::uint64_t>(required),
        "workspace launches do not allocate");

  // Staged path: two calls with mark/rewind and no net growth.
  std::vector<int> host_status(batch.jobs.size(), -1);
  std::vector<ref::MorphFit> host_fits(batch.jobs.size(), sentinel_fit());
  gpu::MorphStagedResult result;
  result.host_status = host_status.data();
  result.host_fits = host_fits.data();
  gpu::MorphStagedResult no_result;

  const std::uint64_t mark = arena.mark();
  std::size_t scratch_bytes = 0;
  check(gpu::morph_stage_and_run(
            arena, batch.jobs.data(), static_cast<int>(batch.jobs.size()),
            &result, nullptr, &scratch_bytes) == cuda::Status::kOk,
        "staged run 1");
  check(scratch_bytes == scratch, "staged scratch size");
  check(arena.mark() == mark, "staged run 1 rewinds");
  const std::uint64_t peak_after_first = arena.peak_used();
  check(gpu::morph_stage_and_run(
            arena, batch.jobs.data(), static_cast<int>(batch.jobs.size()),
            &no_result, nullptr, nullptr) == cuda::Status::kOk,
        "staged run 2");
  check(arena.mark() == mark, "staged run 2 rewinds");
  check(arena.peak_used() == peak_after_first, "staged peak does not grow");
  for (std::size_t i = 0; i < batch.jobs.size(); ++i) {
    check((host_status[i] != 0) == batch.cpu_ok[i], "staged accept/reject");
    if (batch.cpu_ok[i]) {
      check(fit_equal(host_fits[i], batch.cpu_fits[i]), "staged fit");
    }
  }

  // One-byte-short workspace: distinct status plus the required size.
  {
    cuda::Arena short_arena;
    check(short_arena.init(required - 1, 256, &error) == cuda::Status::kOk,
          "short arena init: " + error);
    gpu::MorphWorkspace short_workspace;
    const cuda::Status status = short_workspace.init(short_arena, config, &error);
    check(status == cuda::Status::kArenaExhausted,
          "one-byte-short workspace reports kArenaExhausted");
    check(short_workspace.required_bytes() == required,
          "one-byte-short workspace reports required bytes");
    check(short_arena.mark() == 0, "short workspace leaves the arena untouched");
    check(!error.empty(), "short workspace sets an error string");
  }

  // One-byte-short staging.
  {
    const std::size_t need = gpu::required_scratch_bytes(4);
    cuda::Arena short_arena;
    check(short_arena.init(need - 1, 256, &error) == cuda::Status::kOk,
          "short scratch arena init: " + error);
    std::size_t reported = 0;
    const cuda::Status status = gpu::morph_stage_and_run(
        short_arena, batch.jobs.data(), 4, nullptr, nullptr, &reported);
    check(status == cuda::Status::kArenaExhausted,
          "one-byte-short staging reports kArenaExhausted");
    check(reported == need, "one-byte-short staging reports required bytes");
    check(short_arena.mark() == 0, "short staging leaves the arena untouched");
  }

  std::printf(
      "morph_emulation_test: required_device_bytes=%zu scratch=%zu "
      "arena_capacity=%zu peak_used=%llu\n",
      required, scratch, capacity,
      static_cast<unsigned long long>(arena.peak_used()));
}

}  // namespace

int main() {
  int successes = 0;
  int rejections = 0;
  int ten_bit = 0;
  run_parity_cases(&successes, &rejections, &ten_bit);
  test_null_inputs();
  {
    BatchInput batch = make_batch_input(8);
    test_raw_batch(batch);
  }
  test_workspace_and_arena();

  std::printf(
      "morph_emulation_test: cases=%d accepted=%d rejected=%d "
      "bit_depth10=%d checks=%d failures=%d\n",
      successes + rejections, successes, rejections, ten_bit, g_checks,
      g_failures);
  return g_failures == 0 ? 0 : 1;
}
