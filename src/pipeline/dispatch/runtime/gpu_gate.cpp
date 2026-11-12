// Real CUDA gate for the multi-GPU runtime.
//
// It exercises the same runtime path as the mock test: device selection,
// per-GPU contexts/arenas, compute + copy streams, events, source upload,
// deterministic payload merge, reconstruction gather, and reference fill.
// The worker writes deterministic bytes; it does not encode video.
//
// Exit codes: 0 pass, 1 failure, 2 usage, 77 no CUDA device / runtime.
//
// Hardware disclaimer: this gate proves the runtime plumbing on the GPUs that
// are present. It is not an encoder result, not a bitstream validation, and not
// a measured scaling claim. Two GPUs exercise the same code as the mock
// dispatch gate, but with real contexts, streams, events, and cudaMemcpyAsync.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_backend.h"
#include "runtime.h"
#include "tests/test_worker.h"

namespace {

namespace rt = av2enc::multigpu::runtime;
namespace testing = av2enc::multigpu::runtime::testing;

struct Options {
  int width = 640;
  int height = 360;
  std::uint32_t stride = 1024;
  int frames = 2;
  std::vector<int> ordinals;
};

void usage(const char* program) {
  std::cout << "usage: " << program
            << " [--width W] [--height H] [--stride BYTES] [--frames N]"
               " [--ordinals 0,1]\n";
}

bool parse_ints(const char* text, std::vector<int>& out) {
  out.clear();
  std::string value(text);
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t comma = value.find(',', start);
    const std::string part =
        value.substr(start, comma == std::string::npos ? std::string::npos
                                                       : comma - start);
    if (part.empty()) {
      return false;
    }
    char* end = nullptr;
    const long ordinal = std::strtol(part.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || ordinal < 0 || ordinal > 64) {
      return false;
    }
    out.push_back(static_cast<int>(ordinal));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return !out.empty();
}

bool parse(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << name << " needs a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--width") {
      const char* value = next("--width");
      if (value == nullptr) {
        return false;
      }
      options.width = std::atoi(value);
    } else if (arg == "--height") {
      const char* value = next("--height");
      if (value == nullptr) {
        return false;
      }
      options.height = std::atoi(value);
    } else if (arg == "--stride") {
      const char* value = next("--stride");
      if (value == nullptr) {
        return false;
      }
      options.stride = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    } else if (arg == "--frames") {
      const char* value = next("--frames");
      if (value == nullptr) {
        return false;
      }
      options.frames = std::atoi(value);
    } else if (arg == "--ordinals") {
      const char* value = next("--ordinals");
      if (value == nullptr || !parse_ints(value, options.ordinals)) {
        std::cerr << "invalid --ordinals\n";
        return false;
      }
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage(argv[0]);
      return false;
    }
  }
  if (options.width <= 0 || options.height <= 0 || options.width % 2 != 0 ||
      options.height % 2 != 0 || options.stride == 0 || options.frames < 1) {
    std::cerr << "invalid geometry/stride/frames\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse(argc, argv, options)) {
    return 2;
  }

  std::string error;
  std::unique_ptr<rt::DeviceBackend> backend = rt::make_cuda_backend(&error);
  if (backend == nullptr) {
    std::cout << "SKIP: CUDA backend unavailable: " << error << "\n";
    std::cout << "DISCLAIMER: no real GPU gate ran. The mock dispatch gate is "
                 "the only runtime evidence; no multi-GPU hardware or scaling "
                 "claim is made.\n";
    return 77;
  }

  int device_count = 0;
  if (!rt::is_ok(backend->device_count(&device_count)) || device_count < 1) {
    std::cout << "SKIP: no CUDA devices visible\n";
    std::cout << "DISCLAIMER: no real GPU gate ran. The mock dispatch gate is "
                 "the only runtime evidence; no multi-GPU hardware or scaling "
                 "claim is made.\n";
    return 77;
  }

  std::vector<int> ordinals = options.ordinals;
  if (ordinals.empty()) {
    ordinals.push_back(0);
    if (device_count >= 2) {
      ordinals.push_back(1);
    }
  }
  for (int ordinal : ordinals) {
    if (ordinal >= device_count) {
      std::cout << "SKIP: requested GPU " << ordinal << " is not present ("
                << device_count << " visible)\n";
      return 77;
    }
  }

  std::cout << "CUDA gate: " << device_count << " device(s) visible, using";
  for (int ordinal : ordinals) {
    std::cout << " " << ordinal;
  }
  std::cout << "\n";
  std::cout << (ordinals.size() >= 2 ? "2-GPU gate\n" : "single-GPU gate\n");

  rt::RuntimeConfig config;
  config.gpu_ordinals = ordinals;
  config.planner.gpu_count = static_cast<int>(ordinals.size());
  config.width = options.width;
  config.height = options.height;
  config.payload_stride_bytes = options.stride;
  config.allow_drop_failed = false;  // no silent degradation in the gate
  config.require_homogeneous = true;

  rt::Runtime runtime;
  if (!rt::is_ok(runtime.initialize(*backend, config, &error))) {
    std::cerr << "FAIL: runtime initialize: " << error << "\n";
    return 1;
  }

  std::cout << "layout: " << runtime.plan().layout.cols << "x"
            << runtime.plan().layout.rows << ", "
            << runtime.plan().layout.tile_count() << " tiles\n";
  for (const rt::GpuRuntimeInfo& gpu : runtime.gpu_runtime()) {
    std::cout << "  GPU " << gpu.ordinal << ": " << gpu.tile_count
              << " tiles, arena " << (gpu.arena_capacity_bytes >> 20) << " MiB, "
              << (gpu.arena_used_bytes >> 20) << " MiB used\n";
  }

  testing::PatternWorker worker(options.stride, false);
  int failures = 0;
  for (int frame = 0; frame < options.frames; ++frame) {
    const std::vector<std::uint8_t> source =
        testing::expected_source_frame(options.width, options.height, frame);
    rt::FrameRequest request;
    request.frame_index = frame;
    request.source = source.data();
    request.source_bytes = source.size();
    request.capture_recon = true;
    rt::FrameResult result;
    if (!rt::is_ok(runtime.dispatch_frame(request, worker, result, &error))) {
      std::cerr << "FAIL frame " << frame << ": " << error << "\n";
      ++failures;
      break;
    }
    if (!result.valid || result.tile_payloads.size() !=
                             static_cast<std::size_t>(
                                 runtime.plan().layout.tile_count())) {
      std::cerr << "FAIL frame " << frame << ": incomplete merge\n";
      ++failures;
      break;
    }
    for (int tile = 0; tile < runtime.plan().layout.tile_count(); ++tile) {
      const std::vector<std::uint8_t> expected =
          testing::expected_payload(tile, frame, options.stride);
      if (result.tile_payloads[static_cast<std::size_t>(tile)] != expected) {
        std::cerr << "FAIL frame " << frame << " tile " << tile
                  << ": payload mismatch\n";
        ++failures;
      }
    }
    const std::vector<std::uint8_t> expected_recon =
        testing::expected_recon_frame(options.width, options.height, frame);
    if (result.recon != expected_recon) {
      std::cerr << "FAIL frame " << frame << ": assembled recon mismatch\n";
      ++failures;
    }
    std::cout << "  frame " << frame << ": payload " << result.payload_bytes
              << " B, host tx " << result.host_tx_bytes << " B\n";
    if (!rt::is_ok(runtime.synchronize(&error))) {
      std::cerr << "FAIL frame " << frame << ": synchronize: " << error << "\n";
      ++failures;
      break;
    }
  }

  runtime.shutdown();
  if (failures != 0) {
    std::cerr << "RESULT: FAIL (" << failures << ")\n";
    return 1;
  }
  std::cout << "RESULT: PASS\n";
  std::cout << "DISCLAIMER: runtime plumbing gate only (deterministic payload "
               "and recon bytes); no encoder kernels ran, no bitstream was "
               "produced, and no multi-GPU scaling claim is made.\n";
  return 0;
}
