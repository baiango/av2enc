// Standalone benchmark + correctness harness for the wavefront scheduler.
//
// It runs the same M0 coding-block list through three paths and compares every
// output buffer bit-exactly:
//
//   1. the per-block M0 path (detail::run_gpu_job, one component call per
//      stage per block: the 2,740-launch baseline at 640x360);
//   2. the wavefront plan executed eagerly (batched, one host call per kernel);
//   3. the wavefront plan replayed from a captured CUDA graph (one host call
//      per frame).
//
// It then runs the real M0 pipeline entry point (encode_still_picture_cuda) and
// checks that its reconstruction matches the same block list, which validates
// the test geometry against the production PartitionPlanner, and writes the IVF
// so tools/validate_dav2d.sh can check the pinned decoded MD5.
//
// Build/run: src/cuda/scheduler/build_scheduler_test.sh

#include "cuda/basic_pipeline.h"
#include "cuda/pipeline/dispatch/scheduler/scheduler.h"
#include "basic_geometry.h"

#include "pipeline/still/basic_encoder.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::cuda::sched::CodingBlock;
using av2enc::cuda::sched::FrameGeometry;
using av2enc::cuda::sched::FrameJob;
using av2enc::cuda::sched::PlanCounters;
using av2enc::cuda::sched::RunOptions;
using av2enc::cuda::sched::RunStats;
using av2enc::cuda::sched::ScheduledFrame;
using av2enc::cuda::sched::testutil::make_geometry;
using av2enc::cuda::sched::testutil::make_basic_k32x32_blocks;

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

struct Planes {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_y4m(const std::filesystem::path& path, Planes& planes,
              std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) {
    return false;
  }
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
  if (status != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) {
      error = "Y4M contains no frame";
    }
    return false;
  }
  planes.width = frame.width();
  planes.height = frame.height();
  planes.y.assign(frame.y().begin(), frame.y().end());
  planes.u.assign(frame.u().begin(), frame.u().end());
  planes.v.assign(frame.v().begin(), frame.v().end());
  return true;
}

void make_pattern(Planes& planes, std::uint32_t width, std::uint32_t height) {
  planes.width = width;
  planes.height = height;
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  planes.y.resize(static_cast<std::size_t>(width) * height);
  planes.u.resize(static_cast<std::size_t>(cw) * ch);
  planes.v.resize(static_cast<std::size_t>(cw) * ch);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t value =
          (x * 3U + y * 5U + ((x * x + y * y) >> 7U)) & 0xFFU;
      planes.y[static_cast<std::size_t>(y) * width + x] =
          static_cast<std::uint8_t>(value);
    }
  }
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t x = 0; x < cw; ++x) {
      const std::uint32_t value =
          (128U + (x * 7U + y * 11U) / 3U + ((x * y) >> 5U)) & 0xFFU;
      planes.u[static_cast<std::size_t>(y) * cw + x] =
          static_cast<std::uint8_t>(value);
      planes.v[static_cast<std::size_t>(y) * cw + x] =
          static_cast<std::uint8_t>(255U - value);
    }
  }
}

std::vector<std::uint8_t> crop_plane(const std::uint8_t* plane, int stride,
                                     int width, int height) {
  std::vector<std::uint8_t> out(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    std::memcpy(out.data() + static_cast<std::size_t>(y) * width,
                plane + static_cast<std::size_t>(y) * stride,
                static_cast<std::size_t>(width));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

struct Diff {
  long long differences = 0;
  long long first = -1;
  int max_abs = 0;
};

Diff diff_bytes(const std::uint8_t* a, const std::uint8_t* b, std::size_t n) {
  Diff d;
  for (std::size_t i = 0; i < n; ++i) {
    const int delta = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    if (delta != 0) {
      if (d.first < 0) {
        d.first = static_cast<long long>(i);
      }
      ++d.differences;
      d.max_abs = std::max(d.max_abs, delta < 0 ? -delta : delta);
    }
  }
  return d;
}

Diff diff_i32(const std::int32_t* a, const std::int32_t* b, std::size_t n) {
  Diff d;
  for (std::size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      if (d.first < 0) {
        d.first = static_cast<long long>(i);
      }
      ++d.differences;
      const long long delta =
          static_cast<long long>(a[i]) - static_cast<long long>(b[i]);
      const long long magnitude = delta < 0 ? -delta : delta;
      d.max_abs = static_cast<int>(std::min<long long>(magnitude, 1 << 20));
    }
  }
  return d;
}

bool report(const char* label, const Diff& d) {
  if (d.differences == 0) {
    std::printf("  %-42s bit-exact\n", label);
    return true;
  }
  std::printf("  %-42s DIFFERENT (%lld samples, first %lld, max |d| %d)\n",
              label, d.differences, d.first, d.max_abs);
  return false;
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

struct BenchResult {
  double kernel_ms = 0.0;
  double submit_ms = 0.0;
  double wall_ms = 0.0;
  int api_calls = 0;      // component calls (M0 counter) or host launches
  int kernel_launches = 0;
};

BenchResult bench_naive(av2enc::cuda::detail::GpuJob& job, int device,
                        int iterations, std::string& error, bool* ok) {
  BenchResult result;
  for (int i = 0; i < iterations; ++i) {
    av2enc::cuda::detail::GpuTimings timings;
    const auto start = std::chrono::steady_clock::now();
    if (!av2enc::cuda::detail::run_gpu_job(job, device, error, &timings)) {
      *ok = false;
      return result;
    }
    const auto end = std::chrono::steady_clock::now();
    result.kernel_ms += timings.kernel_ms;
    result.wall_ms += std::chrono::duration<double, std::milli>(end - start).count();
    result.api_calls = timings.launches;
    result.kernel_launches = timings.launches;  // component-call counter
  }
  const double n = static_cast<double>(iterations);
  result.kernel_ms /= n;
  result.wall_ms /= n;
  *ok = true;
  return result;
}

BenchResult bench_scheduled(ScheduledFrame& frame, const FrameJob& job,
                            bool graph, int iterations, std::string& error,
                            bool* ok) {
  BenchResult result;
  RunOptions options;
  options.use_cuda_graph = graph;
  for (int i = 0; i < iterations; ++i) {
    RunStats stats;
    const auto start = std::chrono::steady_clock::now();
    if (!frame.run(job, options, &stats, error)) {
      *ok = false;
      return result;
    }
    const auto end = std::chrono::steady_clock::now();
    result.kernel_ms += stats.kernel_ms;
    result.submit_ms += stats.submit_ms;
    result.wall_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
    result.api_calls = stats.host_launches;
    result.kernel_launches = stats.gpu_nodes;
  }
  const double n = static_cast<double>(iterations);
  result.kernel_ms /= n;
  result.submit_ms /= n;
  result.wall_ms /= n;
  *ok = true;
  return result;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Options {
  std::string y4m;
  std::string out_dir = ".";
  std::string name;
  std::uint32_t width = 640;
  std::uint32_t height = 360;
  int iterations = 5;
  int device = 0;
  bool e2e = true;
  bool verbose = false;
};

bool parse_args(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        error = std::string("missing value for ") + what;
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--y4m") {
      const char* value = next("--y4m");
      if (value == nullptr) {
        return false;
      }
      options.y4m = value;
    } else if (arg == "--out-dir") {
      const char* value = next("--out-dir");
      if (value == nullptr) {
        return false;
      }
      options.out_dir = value;
    } else if (arg == "--name") {
      const char* value = next("--name");
      if (value == nullptr) {
        return false;
      }
      options.name = value;
    } else if (arg == "--width") {
      const char* value = next("--width");
      if (value == nullptr) {
        return false;
      }
      options.width = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    } else if (arg == "--height") {
      const char* value = next("--height");
      if (value == nullptr) {
        return false;
      }
      options.height =
          static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    } else if (arg == "--iterations") {
      const char* value = next("--iterations");
      if (value == nullptr) {
        return false;
      }
      options.iterations = static_cast<int>(std::strtol(value, nullptr, 10));
    } else if (arg == "--device") {
      const char* value = next("--device");
      if (value == nullptr) {
        return false;
      }
      options.device = static_cast<int>(std::strtol(value, nullptr, 10));
    } else if (arg == "--no-e2e") {
      options.e2e = false;
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "usage: scheduler_harness [--y4m PATH | --width W --height H]\n"
          "                         [--out-dir DIR] [--name NAME]\n"
          "                         [--iterations N] [--device N]\n"
          "                         [--no-e2e] [--verbose]\n");
      std::exit(0);
    } else {
      error = "unknown argument: " + arg;
      return false;
    }
  }
  return true;
}

bool write_ivf(const std::filesystem::path& path,
               const std::vector<std::uint8_t>& frame, std::uint32_t width,
               std::uint32_t height, std::string& error) {
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  av2enc::IvfWriter writer;
  if (!writer.open(path, header, error)) {
    return false;
  }
  if (!writer.write_frame(frame, 0, error)) {
    return false;
  }
  return writer.close(error);
}

std::uint64_t fnv64(const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string error;
  if (!parse_args(argc, argv, options, error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 2;
  }
  if (options.iterations < 1) {
    options.iterations = 1;
  }

  Planes planes;
  if (!options.y4m.empty()) {
    if (!load_y4m(options.y4m, planes, error)) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      return 2;
    }
    if (options.name.empty()) {
      options.name = std::filesystem::path(options.y4m).stem().string();
    }
  } else {
    if ((options.width & 1U) != 0U || (options.height & 1U) != 0U ||
        options.width == 0 || options.height == 0) {
      std::fprintf(stderr, "error: generated input needs even dimensions\n");
      return 2;
    }
    make_pattern(planes, options.width, options.height);
    if (options.name.empty()) {
      options.name = std::to_string(options.width) + "x" +
                     std::to_string(options.height) + "_pattern";
    }
  }

  const int width = static_cast<int>(planes.width);
  const int height = static_cast<int>(planes.height);
  const FrameGeometry geometry = make_geometry(width, height);
  std::int64_t total_coeffs = 0;
  const std::vector<CodingBlock> blocks =
      make_basic_k32x32_blocks(geometry, &total_coeffs);
  const int block_count = static_cast<int>(blocks.size());

  std::printf("scheduler_harness: %s (%dx%d, %d coding blocks, %s)\n",
              options.name.c_str(), width, height, block_count,
              options.y4m.empty() ? "generated pattern" : options.y4m.c_str());

  std::vector<av2enc::cuda::detail::BlockDesc> descs(
      static_cast<std::size_t>(block_count));
  for (int i = 0; i < block_count; ++i) {
    const CodingBlock& b = blocks[static_cast<std::size_t>(i)];
    av2enc::cuda::detail::BlockDesc& d = descs[static_cast<std::size_t>(i)];
    d.mi_row = b.mi_row;
    d.mi_col = b.mi_col;
    d.luma_w = b.luma_w;
    d.luma_h = b.luma_h;
    d.tx_w = b.luma_w / b.upscale;
    d.tx_h = b.luma_h / b.upscale;
    d.upscale = b.upscale;
    d.chroma_w = b.chroma_w;
    d.chroma_h = b.chroma_h;
    d.luma_pels = b.luma_pels;
    d.chroma_pels = b.chroma_pels;
    d.coeff_offset = b.coeff_offset;
    d.real_coeffs = b.real_coeffs;
  }

  const std::size_t rec_y_bytes =
      static_cast<std::size_t>(geometry.padded_width) * geometry.padded_height;
  const std::size_t rec_c_bytes =
      static_cast<std::size_t>(geometry.chroma_padded_width) *
      geometry.chroma_padded_height;
  const std::size_t coeff_count = static_cast<std::size_t>(total_coeffs);
  const std::size_t pred_count = static_cast<std::size_t>(block_count) * 3u;

  // Reference (per-block M0 path) outputs.
  std::vector<std::uint8_t> ref_y(rec_y_bytes, 128);
  std::vector<std::uint8_t> ref_u(rec_c_bytes, 128);
  std::vector<std::uint8_t> ref_v(rec_c_bytes, 128);
  std::vector<std::int32_t> ref_coeffs(coeff_count, 0);
  std::vector<std::int32_t> ref_preds(pred_count, 0);

  // Wavefront outputs.
  std::vector<std::uint8_t> sched_y(rec_y_bytes, 128);
  std::vector<std::uint8_t> sched_u(rec_c_bytes, 128);
  std::vector<std::uint8_t> sched_v(rec_c_bytes, 128);
  std::vector<std::int32_t> sched_coeffs(coeff_count, 0);
  std::vector<std::int32_t> sched_preds(pred_count, 0);

  av2enc::cuda::detail::GpuJob ref_job;
  ref_job.y = planes.y.data();
  ref_job.u = planes.u.data();
  ref_job.v = planes.v.data();
  ref_job.width = width;
  ref_job.height = height;
  ref_job.chroma_width = geometry.chroma_width;
  ref_job.chroma_height = geometry.chroma_height;
  ref_job.padded_width = geometry.padded_width;
  ref_job.padded_height = geometry.padded_height;
  ref_job.chroma_padded_width = geometry.chroma_padded_width;
  ref_job.chroma_padded_height = geometry.chroma_padded_height;
  ref_job.blocks = descs.data();
  ref_job.block_count = block_count;
  ref_job.mi_cols = geometry.mi_cols;
  ref_job.mi_rows = geometry.mi_rows;
  ref_job.recon_y = ref_y.data();
  ref_job.recon_u = ref_u.data();
  ref_job.recon_v = ref_v.data();
  ref_job.coeffs = ref_coeffs.data();
  ref_job.preds = ref_preds.data();

  av2enc::cuda::detail::GpuTimings ref_timings;
  {
    const auto start = std::chrono::steady_clock::now();
    if (!av2enc::cuda::detail::run_gpu_job(ref_job, options.device, error,
                                            &ref_timings)) {
      std::fprintf(stderr, "per-block M0 path failed: %s\n", error.c_str());
      return 1;
    }
    const auto end = std::chrono::steady_clock::now();
    std::printf("  per-block path: %d component calls, %.3f ms kernel, %.3f ms wall\n",
                ref_timings.launches, ref_timings.kernel_ms,
                std::chrono::duration<double, std::milli>(end - start).count());
  }

  FrameJob job;
  job.y = planes.y.data();
  job.u = planes.u.data();
  job.v = planes.v.data();
  job.geometry = geometry;
  job.blocks = blocks.data();
  job.block_count = block_count;
  job.total_coeffs = total_coeffs;
  job.recon_y = sched_y.data();
  job.recon_u = sched_u.data();
  job.recon_v = sched_v.data();
  job.coeffs = sched_coeffs.data();
  job.preds = sched_preds.data();

  ScheduledFrame frame;
  if (!frame.prepare(job, options.device, error)) {
    std::fprintf(stderr, "scheduler prepare failed: %s\n", error.c_str());
    return 1;
  }
  const PlanCounters& counters = frame.plan().counters();
  std::printf(
      "  plan: %d plane jobs, %d wavefront levels, %d groups, %d classes, "
      "%lld dependency edges, %lld violations\n",
      counters.plane_jobs, counters.levels, counters.groups,
      counters.transform_classes, counters.dependency_edges,
      counters.dependency_violations);
  if (frame.graph_available()) {
    std::printf("  cuda graph: captured\n");
  } else {
    std::printf("  cuda graph: unavailable (%s)\n", frame.graph_error().c_str());
  }

  bool ok = true;
  int failures = 0;

  std::printf("\n== correctness: wavefront vs per-block M0 path ==\n");
  if (ref_timings.launches == counters.naive_component_calls) {
    std::printf("  per-block launch count matches the plan (%d component calls)\n",
                ref_timings.launches);
  } else {
    std::printf("  FAIL per-block launch count %d != planned %lld\n",
                ref_timings.launches, counters.naive_component_calls);
    ++failures;
  }
  RunStats graph_stats;
  RunOptions graph_options;
  graph_options.use_cuda_graph = true;
  if (!frame.run(job, graph_options, &graph_stats, error)) {
    std::fprintf(stderr, "scheduler run (graph) failed: %s\n", error.c_str());
    return 1;
  }
  if (!report("recon Y", diff_bytes(ref_y.data(), sched_y.data(), rec_y_bytes))) {
    ++failures;
  }
  if (!report("recon U", diff_bytes(ref_u.data(), sched_u.data(), rec_c_bytes))) {
    ++failures;
  }
  if (!report("recon V", diff_bytes(ref_v.data(), sched_v.data(), rec_c_bytes))) {
    ++failures;
  }
  if (!report("coefficients",
              diff_i32(ref_coeffs.data(), sched_coeffs.data(), coeff_count))) {
    ++failures;
  }
  if (!report("predictions",
              diff_i32(ref_preds.data(), sched_preds.data(), pred_count))) {
    ++failures;
  }
  std::printf("  graph replay: %d host launch, %d graph nodes\n",
              graph_stats.host_launches, graph_stats.gpu_nodes);
  std::printf("  device workspace: %.2f MiB\n",
              static_cast<double>(graph_stats.workspace_bytes) / (1024.0 * 1024.0));

  std::fill(sched_y.begin(), sched_y.end(), 0);
  std::fill(sched_u.begin(), sched_u.end(), 0);
  std::fill(sched_v.begin(), sched_v.end(), 0);
  std::fill(sched_coeffs.begin(), sched_coeffs.end(), 0);
  std::fill(sched_preds.begin(), sched_preds.end(), 0);
  RunStats eager_stats;
  RunOptions eager_options;
  eager_options.use_cuda_graph = false;
  if (!frame.run(job, eager_options, &eager_stats, error)) {
    std::fprintf(stderr, "scheduler run (eager) failed: %s\n", error.c_str());
    return 1;
  }
  if (!report("eager recon Y", diff_bytes(ref_y.data(), sched_y.data(), rec_y_bytes))) {
    ++failures;
  }
  if (!report("eager coefficients",
              diff_i32(ref_coeffs.data(), sched_coeffs.data(), coeff_count))) {
    ++failures;
  }
  if (!report("eager predictions",
              diff_i32(ref_preds.data(), sched_preds.data(), pred_count))) {
    ++failures;
  }
  std::printf("  eager: %d host launches, %d kernels\n", eager_stats.host_launches,
              eager_stats.gpu_nodes);

  // -------------------------------------------------------------------------
  // Benchmarks
  // -------------------------------------------------------------------------
  std::printf("\n== benchmark (%d iterations) ==\n", options.iterations);
  BenchResult naive = bench_naive(ref_job, options.device, options.iterations,
                                  error, &ok);
  if (!ok) {
    std::fprintf(stderr, "naive benchmark failed: %s\n", error.c_str());
    return 1;
  }
  // The M0 per-block counter counts component calls; the plan also predicts the
  // actual number of kernels launched internally (2 per forward/inverse call).
  naive.kernel_launches = static_cast<int>(counters.naive_kernel_launches);
  BenchResult wave_eager = bench_scheduled(frame, job, false, options.iterations,
                                           error, &ok);
  if (!ok) {
    std::fprintf(stderr, "eager benchmark failed: %s\n", error.c_str());
    return 1;
  }
  BenchResult wave_graph = bench_scheduled(frame, job, true, options.iterations,
                                           error, &ok);
  if (!ok) {
    std::fprintf(stderr, "graph benchmark failed: %s\n", error.c_str());
    return 1;
  }

  std::printf(
      "  %-22s %10s %10s %10s %10s %10s\n", "path", "host calls", "kernels",
      "kernel ms", "submit ms", "wall ms");
  std::printf("  %-22s %10d %10d %10.3f %10s %10.3f\n", "M0 per-block",
              naive.api_calls, naive.kernel_launches, naive.kernel_ms, "-",
              naive.wall_ms);  std::printf("  %-22s %10d %10d %10.3f %10.3f %10.3f\n", "wavefront eager",
              wave_eager.api_calls, wave_eager.kernel_launches,
              wave_eager.kernel_ms, wave_eager.submit_ms, wave_eager.wall_ms);
  std::printf("  %-22s %10d %10d %10.3f %10.3f %10.3f\n", "wavefront graph",
              wave_graph.api_calls, wave_graph.kernel_launches,
              wave_graph.kernel_ms, wave_graph.submit_ms, wave_graph.wall_ms);
  if (naive.wall_ms > 0.0 && wave_graph.wall_ms > 0.0) {
    std::printf("  speedup (wall, per-block / graph): %.2fx\n",
                naive.wall_ms / wave_graph.wall_ms);
  }
  if (naive.kernel_ms > 0.0 && wave_graph.kernel_ms > 0.0) {
    std::printf("  speedup (kernel, per-block / graph): %.2fx\n",
                naive.kernel_ms / wave_graph.kernel_ms);
  }

  // Multi-frame reuse: after all those replays the persistent recon planes and
  // prediction slots must still reproduce the reference exactly (nothing may
  // depend on per-frame re-initialisation).
  {
    std::fill(sched_y.begin(), sched_y.end(), 0);
    std::fill(sched_u.begin(), sched_u.end(), 0);
    std::fill(sched_v.begin(), sched_v.end(), 0);
    std::fill(sched_coeffs.begin(), sched_coeffs.end(), 0);
    std::fill(sched_preds.begin(), sched_preds.end(), 0);
    RunStats replay_stats;
    if (!frame.run(job, graph_options, &replay_stats, error)) {
      std::fprintf(stderr, "steady-state replay failed: %s\n", error.c_str());
      return 1;
    }
    if (!report("steady-state replay recon Y",
                diff_bytes(ref_y.data(), sched_y.data(), rec_y_bytes)) ||
        !report("steady-state replay coefficients",
                diff_i32(ref_coeffs.data(), sched_coeffs.data(), coeff_count)) ||
        !report("steady-state replay predictions",
                diff_i32(ref_preds.data(), sched_preds.data(), pred_count))) {
      ++failures;
    }
  }

  // -------------------------------------------------------------------------
  // End-to-end cross-check against the production pipeline entry point
  // -------------------------------------------------------------------------
  if (options.e2e) {
    std::printf("\n== end-to-end (production M0 planner) ==\n");
    av2enc::basic::StillPicture picture;
    picture.width = static_cast<std::uint32_t>(width);
    picture.height = static_cast<std::uint32_t>(height);
    picture.y = std::span<const std::uint8_t>(planes.y.data(), planes.y.size());
    picture.u = std::span<const std::uint8_t>(planes.u.data(), planes.u.size());
    picture.v = std::span<const std::uint8_t>(planes.v.data(), planes.v.size());
    av2enc::basic::StillPictureConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(width);
    config.max_frame_height = static_cast<std::uint16_t>(height);
    config.base_q_idx = 90;
    av2enc::cuda::PipelineOptions pipeline_options;
    pipeline_options.device = options.device;
    pipeline_options.verbose = options.verbose;
    av2enc::basic::StillEncodeResult result;
    av2enc::cuda::PipelineTimings timings;
    if (!av2enc::cuda::encode_still_picture_cuda(
            picture, config, result, error,
            av2enc::basic::BasicPartitionMode::k32x32, pipeline_options, &timings)) {
      std::fprintf(stderr, "encode_still_picture_cuda failed: %s\n",
                   error.c_str());
      return 1;
    }
    const std::vector<std::uint8_t> ref_crop = crop_plane(
        ref_y.data(), geometry.padded_width, width, height);
    if (!report("production recon Y vs per-block job",
                diff_bytes(ref_crop.data(), result.recon_y.data(),
                           ref_crop.size()))) {
      ++failures;
    }
    if (timings.kernel_launches != counters.naive_component_calls) {
      std::printf(
          "  note: production component calls %d != predicted %lld (block "
          "list mismatch)\n",
          timings.kernel_launches, counters.naive_component_calls);
      ++failures;
    } else {
      std::printf("  production launches: %d component calls (matches plan)\n",
                  timings.kernel_launches);
    }
    std::printf("  frame: %u bytes (tile %u), fnv64 %016llx\n",
                result.frame_bytes, result.tile_bytes,
                static_cast<unsigned long long>(
                    fnv64(result.frame.data(), result.frame.size())));

    std::error_code ec;
    const std::filesystem::path out_dir(options.out_dir);
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
      std::fprintf(stderr, "error: cannot create %s: %s\n",
                   out_dir.string().c_str(), ec.message().c_str());
      return 1;
    }
    const std::filesystem::path ivf = out_dir / ("gpu_" + options.name + ".ivf");
    if (!write_ivf(ivf, result.frame, planes.width, planes.height, error)) {
      std::fprintf(stderr, "error: IVF write failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("  ivf: %s (run tools/validate_dav2d.sh to check the pinned MD5)\n",
                ivf.string().c_str());
  }

  std::printf("\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
