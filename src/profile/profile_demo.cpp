// Standalone profiler demo.  No CMake, no encoder dependency.
//
//   src/profile/build_demo.sh
//   ./src/profile/build/profile_demo
//
// Set AV2ENC_PROFILE=<path> to choose the JSONL output path; otherwise a path
// under the system temp directory is used.  AV2ENC_PROFILE_NVTX=1 enables NVTX
// if the profiler was built with AV2ENC_HAVE_NVTX=1.

#include "profile/profiler.h"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

#if defined(AV2ENC_HAVE_CUDA) && AV2ENC_HAVE_CUDA
#include <cuda_runtime_api.h>
#define AV2ENC_DEMO_HAS_CUDA 1
#else
#define AV2ENC_DEMO_HAS_CUDA 0
#endif

namespace {

using av2enc::profile::CpuScope;
using av2enc::profile::CudaScope;
using av2enc::profile::Profiler;
using av2enc::profile::stages::kBitstream;
using av2enc::profile::stages::kEntropy;
using av2enc::profile::stages::kHostSync;
using av2enc::profile::stages::kInputDecode;
using av2enc::profile::stages::kInterSearch;
using av2enc::profile::stages::kIntraSearch;
using av2enc::profile::stages::kLoopFilters;
using av2enc::profile::stages::kRecon;
using av2enc::profile::stages::kRdo;
using av2enc::profile::stages::kTileSchedule;
using av2enc::profile::stages::kTotal;
using av2enc::profile::stages::kTransformQuant;

volatile std::uint64_t g_sink = 0;

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void burn(std::uint64_t iterations) {
  std::uint64_t value = g_sink + 0x9E3779B97F4A7C15ULL + iterations;
  for (std::uint64_t i = 0; i < iterations; ++i) {
    value ^= value << 13U;
    value ^= value >> 7U;
    value ^= value << 17U;
  }
  g_sink = value;
}

std::filesystem::path default_output_path() {
  return std::filesystem::temp_directory_path() / "av2enc_profile_demo.jsonl";
}

bool read_file(const std::filesystem::path& path, std::string& contents) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  contents.assign(std::istreambuf_iterator<char>(input),
                  std::istreambuf_iterator<char>());
  return true;
}

bool verify_jsonl(const std::filesystem::path& path, std::uint64_t min_events,
                  std::uint64_t expected_frames, std::string& error) {
  std::ifstream input(path);
  if (!input) {
    error = "cannot open JSONL file '" + path.string() + "'";
    return false;
  }

  std::string line;
  std::string first_event;
  std::uint64_t lines = 0;
  std::uint64_t events = 0;
  std::uint64_t frames = 0;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    ++lines;
    if (line.front() != '{' || line.back() != '}') {
      error = "JSONL line " + std::to_string(lines) + " is not an object";
      return false;
    }
    if (line.find("\"type\":\"event\"") != std::string::npos) {
      ++events;
      if (first_event.empty()) {
        first_event = line;
      }
      if (line.find("\"stage\"") == std::string::npos ||
          line.find("\"dur_ms\"") == std::string::npos ||
          line.find("\"frame\"") == std::string::npos ||
          line.find("\"thread\"") == std::string::npos) {
        error = "event line lacks required fields";
        return false;
      }
    } else if (line.find("\"type\":\"frame\"") != std::string::npos) {
      ++frames;
      if (line.find("\"wall_ms\"") == std::string::npos) {
        error = "frame line lacks wall_ms";
        return false;
      }
    } else {
      error = "unexpected JSONL record type";
      return false;
    }
  }

  if (lines == 0) {
    error = "JSONL file is empty";
    return false;
  }
  if (events < min_events) {
    error = "expected at least " + std::to_string(min_events) + " events, got " +
            std::to_string(events);
    return false;
  }
  if (frames != expected_frames) {
    error = "expected " + std::to_string(expected_frames) + " frame records, got " +
            std::to_string(frames);
    return false;
  }

  std::cout << "  jsonl events: " << events << ", frame records: " << frames
            << ", lines: " << lines << "\n";
  std::cout << "  first event: " << first_event << "\n";
  return true;
}

bool verify_summary(const std::filesystem::path& path, bool expect_vram,
                    std::string& error) {
  std::string contents;
  if (!read_file(path, contents)) {
    error = "cannot open summary '" + path.string() + "'";
    return false;
  }
  while (!contents.empty() &&
         std::isspace(static_cast<unsigned char>(contents.back())) != 0) {
    contents.pop_back();
  }
  if (contents.empty() || contents.front() != '{' || contents.back() != '}') {
    error = "summary is empty or not a JSON object";
    return false;
  }
  const char* required[] = {"\"profiler\": \"av2enc\"",
                            "\"precompute_profile\": \"auto\"",
                            "\"vram_limit_mb\"",
                            "\"wall_ms\"",
                            "\"frames\"",
                            "\"stages\"",
                            "\"cpu\"",
                            "\"cuda\"",
                            "\"input_decode\"",
                            "\"loop_filters\""};
  for (const char* needle : required) {
    if (contents.find(needle) == std::string::npos) {
      error = std::string("summary lacks ") + needle;
      return false;
    }
  }

  const char* measured_vram_fields[] = {"\"peak_vram_mb\"",
                                        "\"vram_capacity_mb\"",
                                        "\"vram_free_mb\""};
  for (const char* needle : measured_vram_fields) {
    const bool present = contents.find(needle) != std::string::npos;
    if (expect_vram && !present) {
      error = std::string("summary lacks ") + needle;
      return false;
    }
    if (!expect_vram && present) {
      error = std::string("summary unexpectedly contains ") + needle;
      return false;
    }
  }

  std::cout << "  summary bytes: " << contents.size()
            << ", measured vram fields: "
            << (expect_vram ? "present" : "omitted") << "\n";
  return true;
}

#if AV2ENC_DEMO_HAS_CUDA
__global__ void write_demo_value(int* out, int value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *out = value;
  }
}

class CudaDemo {
 public:
  CudaDemo() = default;
  ~CudaDemo() { cleanup(); }

  CudaDemo(const CudaDemo&) = delete;
  CudaDemo& operator=(const CudaDemo&) = delete;

  bool init(std::string& reason) {
    cudaError_t result = cudaGetDeviceCount(&device_count_);
    if (result != cudaSuccess) {
      reason = cudaGetErrorString(result);
      return false;
    }
    if (device_count_ <= 0) {
      reason = "no CUDA device";
      return false;
    }
    result = cudaSetDevice(0);
    if (result != cudaSuccess) {
      reason = cudaGetErrorString(result);
      return false;
    }
    result = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (result != cudaSuccess) {
      reason = cudaGetErrorString(result);
      return false;
    }
    void* raw = nullptr;
    result = cudaMalloc(&raw, sizeof(int));
    if (result != cudaSuccess) {
      reason = cudaGetErrorString(result);
      return false;
    }
    device_ = static_cast<int*>(raw);
    return true;
  }

  bool launch(int value) {
    write_demo_value<<<1, 1, 0, stream_>>>(device_, value);
    return cudaGetLastError() == cudaSuccess;
  }

  bool finish(int expected) {
    cudaError_t result = cudaStreamSynchronize(stream_);
    if (result != cudaSuccess) {
      return false;
    }
    int host = 0;
    result = cudaMemcpy(&host, device_, sizeof(int), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      return false;
    }
    return host == expected;
  }

  void cleanup() {
    if (device_ != nullptr) {
      (void)cudaFree(device_);
      device_ = nullptr;
    }
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  void* stream_handle() const { return static_cast<void*>(stream_); }
  int device_count() const { return device_count_; }

 private:
  int* device_ = nullptr;
  cudaStream_t stream_ = nullptr;
  int device_count_ = 0;
};
#endif  // AV2ENC_DEMO_HAS_CUDA

int run_demo() {
  Profiler profiler;
  std::string error;
  const bool configured = profiler.open_from_env(&error);
  if (!configured && !error.empty()) {
    std::cerr << "profile_demo: " << error << "\n";
    return 1;
  }

  if (profiler.jsonl_path().empty()) {
    const bool nvtx = profiler.nvtx_enabled();
    const std::filesystem::path fallback = default_output_path();
    if (!profiler.open(fallback.string(), nvtx, &error)) {
      std::cerr << "profile_demo: " << error << "\n";
      return 1;
    }
  }

  const std::filesystem::path jsonl_path = profiler.jsonl_path();
  const std::filesystem::path summary_path = profiler.summary_path();
  std::cout << "av2enc profiler demo\n"
            << "  jsonl:   " << jsonl_path.string() << "\n"
            << "  summary: " << summary_path.string() << "\n"
            << "  nvtx:    " << (profiler.nvtx_enabled() ? "on" : "off")
            << "\n";

#if AV2ENC_DEMO_HAS_CUDA
  CudaDemo cuda;
  std::string cuda_reason;
  const bool cuda_active = cuda.init(cuda_reason);
  if (!cuda_active) {
    std::cout << "  cuda:    skipped (" << cuda_reason << ")\n";
  } else {
    std::cout << "  cuda:    device 0, " << cuda.device_count() << " device(s)\n";
  }
#else
  [[maybe_unused]] const bool cuda_active = false;
  std::cout << "  cuda:    not compiled (AV2ENC_HAVE_CUDA=0)\n";
#endif

  constexpr std::uint64_t kFrames = 3;
  for (std::uint64_t frame = 0; frame < kFrames; ++frame) {
    profiler.begin_frame(frame);
    {
      CpuScope total(profiler, kTotal);
      {
        CpuScope scope(profiler, kInputDecode);
        burn(120000);
      }
      {
        CpuScope schedule(profiler, kTileSchedule);
        burn(20000);
        {
          CpuScope scope(profiler, kIntraSearch);
          burn(150000);
        }
        {
          CpuScope scope(profiler, kInterSearch);
          burn(90000);
        }
        {
          CpuScope scope(profiler, kRdo);
          burn(110000);
        }
        {
          CpuScope scope(profiler, kTransformQuant);
          burn(50000);
        }
        {
          CpuScope scope(profiler, kRecon);
          burn(50000);
        }
        {
          CpuScope scope(profiler, kLoopFilters);
          burn(60000);
        }
      }
      {
        CpuScope scope(profiler, kEntropy);
        burn(140000);
      }
      {
        // Exercise the runtime string lookup path, not just compiled-in ids.
        CpuScope scope(profiler, "bitstream");
        burn(40000);
      }
#if AV2ENC_DEMO_HAS_CUDA
      if (cuda_active) {
        CudaScope scope(profiler, kTransformQuant, cuda.stream_handle());
        if (!cuda.launch(static_cast<int>(frame))) {
          std::cerr << "profile_demo: CUDA kernel launch failed\n";
        }
      }
#endif
    }
    profiler.end_frame();
  }

#if AV2ENC_DEMO_HAS_CUDA
  if (cuda_active) {
    bool verified = false;
    {
      CpuScope sync_scope(profiler, kHostSync);
      verified = cuda.finish(static_cast<int>(kFrames - 1U));
    }
    cuda.cleanup();
    if (!verified) {
      std::cerr << "profile_demo: CUDA result verification failed\n";
      profiler.close();
      return 1;
    }
    std::cout << "  cuda:    kernel result verified\n";
  }
#endif

  profiler.close();

  std::string verify_error;
  std::cout << "verifying profile output\n";
  if (!verify_jsonl(jsonl_path, kFrames * 11U, kFrames, verify_error)) {
    std::cerr << "profile_demo: " << verify_error << "\n";
    return 1;
  }
  if (!verify_summary(summary_path, cuda_active, verify_error)) {
    std::cerr << "profile_demo: " << verify_error << "\n";
    return 1;
  }
  if (profiler.dropped_cuda_events() != 0) {
    std::cerr << "profile_demo: CUDA events were dropped\n";
    return 1;
  }

  std::cout << "\n";
  profiler.print_summary(std::cout);
  std::cout << "\nPASS: JSONL and summary verified\n";
  return 0;
}

int run_benchmark() {
  const std::filesystem::path jsonl_path =
      std::filesystem::temp_directory_path() /
      "av2enc_profile_overhead.jsonl";

  Profiler disabled;
  constexpr std::uint64_t kDisabledScopes = 1000000;
  std::uint64_t sink = 0;
  const std::uint64_t disabled_start = now_ns();
  for (std::uint64_t i = 0; i < kDisabledScopes; ++i) {
    CpuScope scope(disabled, kInputDecode);
    sink += i;
  }
  const std::uint64_t disabled_ns = now_ns() - disabled_start;
  g_sink = sink;

  Profiler enabled;
  std::string error;
  if (!enabled.open(jsonl_path.string(), false, &error)) {
    std::cerr << "profile_benchmark: " << error << "\n";
    return 1;
  }
  constexpr std::uint64_t kEnabledScopes = 20000;
  const std::uint64_t enabled_start = now_ns();
  for (std::uint64_t i = 0; i < kEnabledScopes; ++i) {
    CpuScope scope(enabled, (i % 2U == 0U) ? kIntraSearch : kRdo);
    sink += i;
  }
  const std::uint64_t enabled_ns = now_ns() - enabled_start;
  g_sink = sink;
  enabled.close();

  const double disabled_per_scope =
      static_cast<double>(disabled_ns) / static_cast<double>(kDisabledScopes);
  const double enabled_per_scope =
      static_cast<double>(enabled_ns) / static_cast<double>(kEnabledScopes);
  const double per_frame_ns = enabled_per_scope * 12.0;
  const double percent_of_60fps_frame =
      per_frame_ns / (1000.0 * 1000.0 * 1000.0 / 60.0) * 100.0;
  const double ms_per_second = per_frame_ns * 60.0 / 1000000.0;

  std::cout << "av2enc profiler overhead benchmark\n"
            << "  disabled: " << disabled_per_scope << " ns/scope\n"
            << "  enabled:  " << enabled_per_scope << " ns/scope\n"
            << "  12 CPU scopes/frame at 60 fps: " << ms_per_second
            << " ms per second (" << percent_of_60fps_frame
            << "% of a 16.67 ms frame)\n"
            << "  jsonl: " << jsonl_path << "\n";

#if AV2ENC_DEMO_HAS_CUDA
  CudaDemo cuda;
  std::string cuda_reason;
  if (cuda.init(cuda_reason)) {
    const std::filesystem::path vram_path =
        std::filesystem::temp_directory_path() /
        "av2enc_profile_vram_benchmark.jsonl";
    Profiler vram_profiler;
    if (vram_profiler.open(vram_path.string(), false, &error)) {
      constexpr std::uint64_t kVramSamples = 2000;
      const std::uint64_t vram_start = now_ns();
      for (std::uint64_t i = 0; i < kVramSamples; ++i) {
        vram_profiler.sample_vram();
      }
      const std::uint64_t vram_ns = now_ns() - vram_start;
      vram_profiler.close();
      std::cout << "  cudaMemGetInfo: "
                << static_cast<double>(vram_ns) /
                       static_cast<double>(kVramSamples)
                << " ns/sample\n";
    }
    cuda.cleanup();
  }
#endif

  return 0;
}

void print_usage(std::ostream& out, const char* program) {
  out << "Usage: " << program << " [--benchmark]\n"
      << "\n"
      << "  --benchmark  Measure disabled/enabled CpuScope overhead.\n"
      << "\n"
      << "Environment:\n"
      << "  AV2ENC_PROFILE=<path>   JSONL output path\n"
      << "  AV2ENC_PROFILE_NVTX=1   enable NVTX ranges if compiled in\n";
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (argument == "-h" || argument == "--help") {
      print_usage(std::cout, argv[0]);
      return 0;
    }
    if (argument == "--benchmark") {
      return run_benchmark();
    }
    std::cerr << "profile_demo: unknown option '" << argument << "'\n";
    print_usage(std::cerr, argv[0]);
    return 2;
  }
  return run_demo();
}
