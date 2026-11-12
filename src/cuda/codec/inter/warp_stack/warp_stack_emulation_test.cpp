// Host-emulation parity test for the shared CUDA warp-stack device header.
//
// Runs without nvcc. It exercises the exact header the kernels are compiled
// from against:
//   * the CPU reference predictors in src/encoder/inter/advanced/warp.cpp
//     (8-bit block + extended warp, including invalid-shear rejection);
//   * the CPU stack-builder helpers in src/encoder/inter/warp_stack/warp_stack.cpp
//     (corner generation, sample collection, least-squares derivation);
//   * the real Arena with a host cudaMalloc shim (sizing, mark/rewind,
//     one-byte-short negative case).
//
// Full 10-bit CPU-vs-GPU parity is in warp_stack_gpu_test.cu: it compares the
// kernels with `warp_stack_host_ref.cpp`, which is this same header compiled by
// the host compiler.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "warp_stack.h"  // CUDA API + Arena (host shim supplied by the build)
#include "warp_stack_host_ref.h"
#include "warp_stack_test_common.h"

#include "codec/inter/advanced/warp.h"
#include "codec/inter/warp_stack/warp_stack.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using av2enc::cuda::warpstack::hostref::needs_extended;
using av2enc::cuda::warpstack::hostref::shear_valid;
using av2enc::cuda::warpstack::plan_device_bytes;
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

constexpr std::int16_t kSentinel = 0x5a5a;

// Expected section count, derived from the CPU reference helpers.
int expected_sections(const ia::WarpParams& params, const ia::PlaneView& ref,
                      const ia::WarpBlockGeometry& geom, int path) {
  if (path == 1) {
    if (!shear_valid(params)) {
      return 0;
    }
    const int w = geom.rect.w;
    const int h = geom.rect.h;
    return ((h + 7) >> 3) * ((w + 7) >> 3);
  }
  ia::ExtWarpContext ctx{};
  ctx.plane = geom.plane;
  ctx.sub_x = geom.sub_x;
  ctx.sub_y = geom.sub_y;
  ctx.block_w = geom.rect.w;
  ctx.block_h = geom.rect.h;
  ctx.mi_col = geom.mi_col;
  ctx.mi_row = geom.mi_row;
  ctx.bw4 = geom.bw4;
  ctx.bh4 = geom.bh4;
  ctx.mi_cols = geom.mi_cols;
  ctx.mi_rows = geom.mi_rows;
  ctx.scaled = ia::warp_ref_is_scaled(geom);
  ctx.ref_w = geom.ref_w;
  ctx.ref_h = geom.ref_h;
  ctx.frame_w = geom.frame_w;
  ctx.frame_h = geom.frame_h;
  ctx.dst_x = geom.dst_x;
  ctx.dst_y = geom.dst_y;
  int count = 0;
  for (int i4 = 0; i4 < (geom.rect.h >> 2); ++i4) {
    for (int j4 = 0; j4 < (geom.rect.w >> 2); ++j4) {
      ia::ExtWarpSection sec{};
      if (ia::ext_warp_prepare(params, ref, ctx, geom.rect.x, geom.rect.y, i4, j4, &sec)) {
        ++count;
      }
    }
  }
  return count;
}

void run_8bit_case(const PredictionCase& c) {
  const Plane8 plane8 = make_plane8(kTestPlaneSize, kTestPlaneSize);
  const ia::PlaneView view = plane8.view;
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
  const ia::WarpBlockGeometry geom =
      make_geometry(c.plane, c.sub, c.sub, x, y, w, h, view.max_x + 1, view.max_y + 1, 8);
  const std::size_t count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  const std::string where = case_name(c);

  std::vector<std::int16_t> block_dst(count, kSentinel);
  std::vector<std::int16_t> ext_dst(count, kSentinel);
  check(ia::warp_predict(c.model.params, view, geom, c.compound != 0, block_dst.data(), w),
        where + ": adv block returns true");
  check(ia::warp_predict_ext(c.model.params, view, geom, c.compound != 0, ext_dst.data(), w),
        where + ": adv ext returns true");

  const bool ext = needs_extended(c.model.params, w, h, geom.ref_w, geom.ref_h, geom.frame_w,
                                  geom.frame_h);
  check(ext == ia::warp_needs_extended(c.model.params, geom),
        where + ": fallback agrees with adv::warp_needs_extended");

  const int paths[3] = {av2enc::cuda::warpstack::kWarpPathAuto,
                        av2enc::cuda::warpstack::kWarpPathBlock,
                        av2enc::cuda::warpstack::kWarpPathExtended};
  for (int path : paths) {
    WarpPredictJob job{};
    job.params = c.model.params;
    job.ref = view.data;
    job.ref_stride = view.stride;
    job.ref_max_x = view.max_x;
    job.ref_max_y = view.max_y;
    job.geom = geom;
    job.compound = c.compound;
    job.path = path;
    std::vector<std::int16_t> dst(count, kSentinel);
    WarpPredictResult result{};
    av2enc::cuda::warpstack::hostref::predict(job, dst.data(), w, &result);

    const std::vector<std::int16_t>& want =
        (path == 1) ? block_dst : (path == 2 ? ext_dst : (ext ? ext_dst : block_dst));
    const int want_path = (path == 0) ? (ext ? 2 : 1) : path;
    check(result.status == 0, where + ": status ok");
    check(result.path == want_path, where + ": path");
    check(result.shear_valid == (shear_valid(c.model.params) ? 1 : 0),
          where + ": shear_valid");
    check(result.sections == expected_sections(c.model.params, view, geom, want_path),
          where + ": section count");
    check(dst == want, where + ": bytes match adv path " + std::to_string(path));
  }
}

void run_10bit_case(const PredictionCase& c) {
  const Plane16 plane16 = make_plane16(kTestPlaneSize, kTestPlaneSize, 1023);
  const ia::WarpBlockGeometry geom = make_geometry(
      c.plane, c.sub, c.sub, c.x, c.y, c.w, c.h, kTestPlaneSize, kTestPlaneSize, 10);
  const std::size_t count = static_cast<std::size_t>(c.w) * static_cast<std::size_t>(c.h);
  const std::string where = case_name(c);

  WarpPredictJob job{};
  job.params = c.model.params;
  job.ref = plane16.data.data();
  job.ref_stride = kTestPlaneSize;
  job.ref_max_x = kTestPlaneSize - 1;
  job.ref_max_y = kTestPlaneSize - 1;
  job.geom = geom;
  job.compound = c.compound;
  job.path = av2enc::cuda::warpstack::kWarpPathAuto;
  std::vector<std::int16_t> first(count, kSentinel);
  std::vector<std::int16_t> second(count, kSentinel);
  WarpPredictResult r1{};
  WarpPredictResult r2{};
  av2enc::cuda::warpstack::hostref::predict(job, first.data(), c.w, &r1);
  av2enc::cuda::warpstack::hostref::predict(job, second.data(), c.w, &r2);
  check(r1.status == 0 && r2.status == 0, where + ": 10-bit status");
  check(first == second, where + ": 10-bit deterministic");
  check(r1.path == r2.path && r1.sections == r2.sections, where + ": 10-bit result");
  int min_value = 32767;
  int max_seen = -32768;
  for (std::int16_t v : first) {
    min_value = std::min(min_value, static_cast<int>(v));
    max_seen = std::max(max_seen, static_cast<int>(v));
  }
  if (!c.compound) {
    check(min_value >= 0 && max_seen <= 1023, where + ": 10-bit single range");
  }
}

void test_prediction() {
  const std::vector<PredictionCase> cases = prediction_cases();
  for (const PredictionCase& c : cases) {
    if (c.bit_depth == 8) {
      run_8bit_case(c);
    } else {
      run_10bit_case(c);
    }
  }
}

void test_10bit_reduction() {
  // When every 10-bit sample fits in 8 bits, the 10-bit path must produce
  // exactly the 8-bit prediction: the only difference is the clip bound, which
  // is unreachable.
  const std::vector<ModelCase> models = make_models();
  for (const ModelCase& model : models) {
    const Plane16 plane16 = make_plane16(64, 64, 255);
    ia::WarpBlockGeometry geom = make_geometry(0, 0, 0, 0, 0, 32, 24, 64, 64, 10);
    WarpPredictJob job{};
    job.params = model.params;
    job.ref = plane16.data.data();
    job.ref_stride = 64;
    job.ref_max_x = 63;
    job.ref_max_y = 63;
    job.geom = geom;
    job.compound = 1;  // compare the unclipped 1/16-sample arithmetic
    job.path = av2enc::cuda::warpstack::kWarpPathAuto;
    std::vector<std::int16_t> dst10(32 * 24, kSentinel);
    WarpPredictResult r{};
    av2enc::cuda::warpstack::hostref::predict(job, dst10.data(), 32, &r);

    std::vector<std::uint8_t> narrow(plane16.data.size());
    for (std::size_t i = 0; i < narrow.size(); ++i) {
      narrow[i] = static_cast<std::uint8_t>(plane16.data[i]);
    }
    ia::PlaneView view{};
    view.data = narrow.data();
    view.stride = 64;
    view.max_x = 63;
    view.max_y = 63;
    geom.bit_depth = 8;
    std::vector<std::int16_t> dst8(32 * 24, kSentinel);
    const bool ext = needs_extended(model.params, 32, 24, 64, 64, 64, 64);
    if (ext) {
      check(ia::warp_predict_ext(model.params, view, geom, true, dst8.data(), 32),
            model.name + ": 8-bit ext reference");
    } else {
      check(ia::warp_predict(model.params, view, geom, true, dst8.data(), 32),
            model.name + ": 8-bit block reference");
    }
    check(dst10 == dst8, model.name + ": 10-bit reduces to 8-bit");
  }
}

// ---------------------------------------------------------------------------
// Corner / sample / estimate parity against the CPU stack builder.
// ---------------------------------------------------------------------------

void test_corners(int trials) {
  const int cols = 24;
  const int rows = 24;
  std::vector<iw::WarpMi> mi;
  fill_grid(&mi, cols, rows);
  iw::WarpGrid grid{cols, rows, mi.data()};

  for (int t = 0; t < trials; ++t) {
    WarpCornerJob job{};
    job.grid = grid;
    job.sb_h4 = random_range(0, 1) ? 16 : 8;
    job.mi_row = random_range(0, rows - 1);
    job.mi_col = random_range(0, cols - 1);
    job.bw4 = std::min(1 << random_range(0, 4), cols - job.mi_col);
    job.bh4 = std::min(1 << random_range(0, 4), rows - job.mi_row);
    job.ref_frame0 = static_cast<std::int8_t>(random_range(1, 3));
    job.iter = random_range(0, 1);

    WarpCornerResult got{};
    av2enc::cuda::warpstack::hostref::corners(job, &got);

    iw::WarpStack cpu_stack{};
    const iw::CornerResult cpu = iw::generate_points_from_corners(
        grid, job.sb_h4, job.mi_row, job.mi_col, job.bw4, job.bh4, job.ref_frame0, job.iter,
        &cpu_stack);
    const bool cpu_valid = cpu_stack.num_warp_found > 0;
    check(got.model_valid == (cpu_valid ? 1 : 0), "corner: model validity");
    check(corner_fields_equal(got.corner, cpu), "corner: fields");
    if (got.model_valid) {
      bool same_model = true;
      for (int i = 0; i < 6; ++i) {
        if (got.corner.wmmat.p[i] != cpu_stack.warp_params[0].p[i]) {
          same_model = false;
        }
      }
      check(same_model, "corner: inserted model");

      // Derived model through the predictor (8-bit, auto path).
      const Plane8 plane8 = make_plane8(128, 128);
      const ia::WarpBlockGeometry geom = make_geometry(0, 0, 0, 8, 8, 32, 32, 128, 128, 8);
      WarpPredictJob pred{};
      pred.params = got.corner.wmmat;
      pred.ref = plane8.data.data();
      pred.ref_stride = 128;
      pred.ref_max_x = 127;
      pred.ref_max_y = 127;
      pred.geom = geom;
      pred.path = av2enc::cuda::warpstack::kWarpPathAuto;
      std::vector<std::int16_t> pred_got(32 * 32, kSentinel);
      std::vector<std::int16_t> pred_want(32 * 32, kSentinel);
      WarpPredictResult r{};
      av2enc::cuda::warpstack::hostref::predict(pred, pred_got.data(), 32, &r);
      const bool ext = needs_extended(pred.params, 32, 32, 128, 128, 128, 128);
      if (ext) {
        check(ia::warp_predict_ext(pred.params, plane8.view, geom, false, pred_want.data(), 32),
              "corner-derived: ext reference");
      } else {
        check(ia::warp_predict(pred.params, plane8.view, geom, false, pred_want.data(), 32),
              "corner-derived: block reference");
      }
      check(pred_got == pred_want, "corner-derived: prediction parity");
    }
  }
}

void test_samples(int trials) {
  const int cols = 24;
  const int rows = 24;
  std::vector<iw::WarpMi> mi;
  fill_grid(&mi, cols, rows);
  iw::WarpGrid grid{cols, rows, mi.data()};

  for (int t = 0; t < trials; ++t) {
    iw::WarpSampleParams params{};
    params.mi_row = random_range(1, rows - 1);
    params.mi_col = random_range(1, cols - 1);
    params.w4 = std::min(1 << random_range(0, 4), cols - params.mi_col);
    params.h4 = std::min(1 << random_range(0, 4), rows - params.mi_row);
    params.mi_cols = cols;
    params.mi_rows = rows;
    params.sb_h4 = random_range(0, 1) ? 16 : 8;
    params.grid = grid;
    params.ref_frame[0] = static_cast<std::int8_t>(random_range(1, 3));
    params.ref_frame[1] = random_range(0, 1) ? static_cast<std::int8_t>(random_range(1, 3))
                                             : iw::kRefNone;
    params.ref = random_range(0, 1);
    params.avail_u = true;
    params.avail_l = true;

    iw::WarpSampleSet got{};
    iw::WarpSampleSet want{};
    const bool got_ok = av2enc::cuda::warpstack::hostref::samples(params, &got);
    const bool want_ok = iw::find_warp_samples(params, &want);
    check(got_ok == want_ok, "samples: found flag");
    check(got.num_samples == want.num_samples, "samples: count");
    bool same = true;
    for (int i = 0; i < want.num_samples; ++i) {
      for (int c = 0; c < 4; ++c) {
        if (got.cand[i][c] != want.cand[i][c]) {
          same = false;
        }
      }
    }
    check(same, "samples: candidate values");
  }
}

void test_estimates(int trials) {
  for (int t = 0; t < trials; ++t) {
    WarpEstimateJob job{};
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
    const ia::WarpParams got = av2enc::cuda::warpstack::hostref::estimate(job);
    const ia::WarpParams want =
        iw::estimate_warp_from_samples(job.samples, job.mid_x, job.mid_y, job.block_mv);
    bool same = true;
    for (int i = 0; i < 6; ++i) {
      if (got.p[i] != want.p[i]) {
        same = false;
      }
    }
    check(same, "estimate: 6 parameters");
  }
}

// ---------------------------------------------------------------------------
// Workspace grid path (the same API route the GPU corner test takes).
//
// This is what caught the GPU harness ordering bug: workspace.grid() carries
// mi_cols/mi_rows from the last upload_mi_grid(). Building a job before the
// upload gives it a 0x0 grid, which the kernel correctly treats as empty.
// ---------------------------------------------------------------------------

void test_workspace_grid_corners(int trials) {
  const int cols = 24;
  const int rows = 24;
  std::vector<iw::WarpMi> mi;
  fill_grid(&mi, cols, rows);
  const iw::WarpGrid host_grid{cols, rows, mi.data()};

  WarpStackConfig config;
  config.max_corner_jobs = 256;
  config.max_mi_cells = cols * rows;
  const std::uint64_t required = required_device_bytes(config);
  Arena arena;
  check(arena.init(required + 4096) == Status::kOk, "workspace grid: arena");
  Workspace workspace;
  check(workspace.create(arena, config) == Status::kOk, "workspace grid: create");
  check(workspace.grid().mi_cols == 0 && workspace.grid().mi_rows == 0,
        "workspace grid: empty before upload");

  check(workspace.upload_mi_grid(mi.data(), cols, rows) == Status::kOk,
        "workspace grid: upload");
  const iw::WarpGrid device_grid = workspace.grid();
  check(device_grid.mi != nullptr && device_grid.mi_cols == cols &&
            device_grid.mi_rows == rows,
        "workspace grid: shape after upload");

  for (int t = 0; t < trials; ++t) {
    WarpCornerJob job{};
    job.sb_h4 = random_range(0, 1) ? 16 : 8;
    job.mi_row = random_range(0, rows - 1);
    job.mi_col = random_range(0, cols - 1);
    job.bw4 = std::min(1 << random_range(0, 4), cols - job.mi_col);
    job.bh4 = std::min(1 << random_range(0, 4), rows - job.mi_row);
    job.ref_frame0 = static_cast<std::int8_t>(random_range(1, 3));
    job.iter = random_range(0, 1);
    job.grid = device_grid;

    WarpCornerResult got{};
    av2enc::cuda::warpstack::hostref::corners(job, &got);
    iw::WarpStack cpu_stack{};
    const iw::CornerResult cpu = iw::generate_points_from_corners(
        host_grid, job.sb_h4, job.mi_row, job.mi_col, job.bw4, job.bh4, job.ref_frame0,
        job.iter, &cpu_stack);
    const bool valid = cpu_stack.num_warp_found > 0;
    check(got.model_valid == (valid ? 1 : 0) && corner_fields_equal(got.corner, cpu),
          "workspace grid: corner table parity");
  }
}

// ---------------------------------------------------------------------------
// Arena integration.
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

void test_arena() {
  const WarpStackConfig config = make_config();
  const auto& plan = plan_device_bytes(config);
  const std::uint64_t required = required_device_bytes(config);
  check(required > 0, "arena: required bytes positive");
  check(plan.total == required, "arena: plan total");
  check(required == plan.predict_jobs + plan.predict_results + plan.corner_jobs +
                          plan.corner_results + plan.sample_jobs + plan.sample_sets +
                          plan.estimate_jobs + plan.estimate_params + plan.mi_grid,
        "arena: plan sums");

  const std::uint64_t slack = 64 * 1024;
  Arena arena;
  check(arena.init(required + slack) == Status::kOk, "arena: init with slack");
  Workspace workspace;
  check(workspace.create(arena, config) == Status::kOk, "arena: create workspace");
  check(workspace.created(), "arena: created flag");
  check(workspace.required_bytes() == required, "arena: reported requirement");
  check(workspace.allocated_bytes() == required, "arena: allocated requirement");
  check(arena.used() == required, "arena: create consumes exactly required");
  check(workspace.create(arena, config) == Status::kAlreadyInitialized,
        "arena: double create is rejected");
  Arena uninitialized;
  Workspace unbound;
  check(unbound.create(uninitialized, config) == Status::kNotInitialized,
        "arena: create without init is rejected");

  const std::uint64_t after_create = arena.used();
  const std::uint64_t mark = arena.mark();
  for (int epoch = 0; epoch < 2; ++epoch) {
    auto* input = arena.allocate_array<std::uint8_t>(8192);
    auto* output = arena.allocate_array<std::int16_t>(4096);
    check(input != nullptr && output != nullptr, "arena: scratch allocations");
    check(arena.used() > after_create, "arena: scratch grows the bump pointer");
    check(arena.rewind(mark), "arena: rewind");
    check(arena.used() == after_create, "arena: no net growth after rewind");
  }

  std::printf(
      "warp_stack_emulation: required_device_bytes=%llu peak_used=%llu capacity=%llu\n",
      static_cast<unsigned long long>(required),
      static_cast<unsigned long long>(arena.peak_used()),
      static_cast<unsigned long long>(arena.capacity()));

  Arena small;
  check(small.init(required - 1) == Status::kOk, "arena: init one byte short");
  Workspace short_workspace;
  std::string error;
  check(short_workspace.create(small, config, &error) == Status::kArenaExhausted,
        "arena: one byte short is kArenaExhausted");
  check(short_workspace.required_bytes() == required, "arena: requirement survives failure");
  check(!short_workspace.created(), "arena: failed create not marked created");
  check(small.used() == 0, "arena: failed create allocates nothing");
  check(!error.empty(), "arena: failure reason");
}

}  // namespace

int main() {
  test_prediction();
  test_10bit_reduction();
  test_corners(800);
  test_workspace_grid_corners(400);
  test_samples(800);
  test_estimates(500);
  test_arena();

  if (g_failures == 0) {
    std::printf("warp_stack_emulation_test: %d checks passed\n", g_checks);
    return 0;
  }
  std::printf("warp_stack_emulation_test: %d/%d checks FAILED\n", g_failures, g_checks);
  return 1;
}
