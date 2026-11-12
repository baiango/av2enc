// Standalone M2 harness: CPU reference vs the GPU M2 pipeline.
//
// For every phase it:
//   1. encodes with av2enc::video::encode_video_sequence()      (CPU reference)
//   2. encodes with av2enc::cuda::video::encode_video_sequence_gpu()
//   3. compares every temporal unit byte-for-byte
//   4. compares every reconstructed sample of Y, U and V
//   5. writes the GPU stream and decodes it with dav2d, then compares the
//      decoded Y4M against the GPU reconstruction sample by sample
//   6. prints the kernel-launch / transfer / timing counters
//
// The same source builds against either backend:
//   build_and_test.sh   -> CPU-emulation backend (no GPU needed)
//   build_gpu_gate.sh   -> CUDA backend
//
// Usage:
//   video_gate [--input clip.y4m] [--frames N] [--out out.ivf]
//           [--full-tree] [--force-intra] [--no-newmv] [--no-intra-fallback]
//           [--force-global] [--single-block] [--trace] [--selftest N]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include "pipeline/video/video_frame.h"

#include "video_gpu.h"

namespace {

// Optional CPU M2 tools are detected rather than hard-depended on.
template <typename T, typename = void>
struct HasEnableInterResidual : std::false_type {};
template <typename T>
struct HasEnableInterResidual<
    T, std::void_t<decltype(std::declval<T&>().enable_inter_residual)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasForceInterResidual : std::false_type {};
template <typename T>
struct HasForceInterResidual<
    T, std::void_t<decltype(std::declval<T&>().force_inter_residual)>>
    : std::true_type {};

template <typename T, typename = void>
struct HasDrlReorder : std::false_type {};
template <typename T>
struct HasDrlReorder<T,
                     std::void_t<decltype(std::declval<T&>().drl_reorder)>>
    : std::true_type {};

template <typename T>
void set_enable_inter_residual(T& o) {
  if constexpr (HasEnableInterResidual<T>::value) {
    o.enable_inter_residual = true;
  }
}
template <typename T>
void set_force_inter_residual(T& o) {
  if constexpr (HasForceInterResidual<T>::value) {
    o.force_inter_residual = true;
  }
}
template <typename T>
void set_drl_reorder(T& o, int value) {
  if constexpr (HasDrlReorder<T>::value) {
    o.drl_reorder = value;
  }
}

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t square_sample(int x, int y, int frame, int plane) {
  const int sx = 8 + 4 * frame;
  const int sy = 8 + 2 * frame;
  const bool in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 32;
  if (plane == 0) {
    if (in_square) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(40 + ((lx * 37 + ly * 17) & 0x7F));
    }
    return static_cast<std::uint8_t>(96 + ((x + y) >> 3));
  }
  if (plane == 1) {
    return static_cast<std::uint8_t>(in_square ? 100 : 128);
  }
  return static_cast<std::uint8_t>(in_square ? 150 : 128);
}

std::vector<OwnedFrame> make_synthetic_clip(int frames, int width, int height) {
  std::vector<OwnedFrame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(static_cast<std::size_t>(width) *
                   static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) *
                   static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) *
                   static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)] =
            square_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
            static_cast<std::size_t>(x)] =
            square_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
            static_cast<std::size_t>(x)] =
            square_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

bool load_y4m(const std::string& path, int max_frames,
              std::vector<OwnedFrame>& out, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out.size()) < max_frames) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    OwnedFrame owned;
    owned.width = frame.width();
    owned.height = frame.height();
    owned.y.assign(frame.y().begin(), frame.y().end());
    owned.u.assign(frame.u().begin(), frame.u().end());
    owned.v.assign(frame.v().begin(), frame.v().end());
    out.push_back(std::move(owned));
  }
  if (out.empty()) {
    error = "Y4M input contained no frames";
    return false;
  }
  return true;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

bool write_ivf(const std::string& path,
               const av2enc::video::VideoSequenceResult& sequence,
               std::uint32_t width, std::uint32_t height, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(sequence.units.size());
  if (!writer.open(path, header, error)) return false;
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    if (!writer.write_frame(sequence.units[i], i, error)) return false;
  }
  return writer.close(error);
}

struct StreamDiff {
  bool ok = true;
  std::uint64_t unit_mismatches = 0;
  std::uint64_t recon_mismatches = 0;
  int max_diff = 0;
  std::size_t first_frame = 0;
  std::size_t first_offset = 0;
};

StreamDiff compare_streams(const av2enc::video::VideoSequenceResult& cpu,
                           const av2enc::video::VideoSequenceResult& gpu) {
  StreamDiff diff;
  if (cpu.units.size() != gpu.units.size()) {
    diff.ok = false;
    return diff;
  }
  for (std::size_t i = 0; i < cpu.units.size(); ++i) {
    if (cpu.units[i] != gpu.units[i]) {
      diff.ok = false;
      ++diff.unit_mismatches;
      if (diff.first_offset == 0 && diff.first_frame == 0) {
        diff.first_frame = i;
        const std::size_t limit =
            std::min(cpu.units[i].size(), gpu.units[i].size());
        for (std::size_t j = 0; j < limit; ++j) {
          if (cpu.units[i][j] != gpu.units[i][j]) {
            diff.first_offset = j;
            break;
          }
        }
      }
    }
    if (cpu.frame_bytes[i] != gpu.frame_bytes[i] ||
        cpu.tile_bytes[i] != gpu.tile_bytes[i] ||
        cpu.inter_blocks[i] != gpu.inter_blocks[i] ||
        cpu.intra_blocks[i] != gpu.intra_blocks[i]) {
      diff.ok = false;
    }
  }
  if (cpu.recon.size() != gpu.recon.size()) {
    diff.ok = false;
    return diff;
  }
  for (std::size_t f = 0; f < cpu.recon.size(); ++f) {
    const auto compare_plane = [&](const std::vector<std::uint8_t>& a,
                                   const std::vector<std::uint8_t>& b) {
      if (a.size() != b.size()) {
        diff.ok = false;
        diff.recon_mismatches += a.size() + b.size();
        return;
      }
      for (std::size_t i = 0; i < a.size(); ++i) {
        const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        if (d != 0) {
          ++diff.recon_mismatches;
          diff.max_diff = std::max(diff.max_diff, d);
          if (diff.first_offset == 0 && diff.first_frame == 0) {
            diff.first_frame = f;
            diff.first_offset = i;
          }
        }
      }
    };
    compare_plane(cpu.recon[f].y, gpu.recon[f].y);
    compare_plane(cpu.recon[f].u, gpu.recon[f].u);
    compare_plane(cpu.recon[f].v, gpu.recon[f].v);
  }
  diff.ok = diff.ok && diff.unit_mismatches == 0 && diff.recon_mismatches == 0;
  return diff;
}

struct DecodeDiff {
  bool ok = true;
  std::uint64_t mismatches = 0;
  int max_diff = 0;
  std::uint32_t frames = 0;
};

DecodeDiff compare_decoded(const std::string& decoded_path,
                           const av2enc::video::VideoSequenceResult& recon,
                           std::string& error) {
  DecodeDiff result;
  av2enc::Y4mReader reader(decoded_path);
  if (!reader.open(error)) {
    result.ok = false;
    return result;
  }
  av2enc::Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      result.ok = false;
      return result;
    }
    if (index >= recon.recon.size()) {
      error = "decoder produced more frames than the encoder";
      result.ok = false;
      return result;
    }
    const av2enc::video::VideoReconFrame& ref = recon.recon[index];
    const auto compare_plane = [&](std::span<const std::uint8_t> decoded,
                                   const std::vector<std::uint8_t>& expected) {
      if (decoded.size() != expected.size()) {
        result.ok = false;
        return;
      }
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        const int d = std::abs(static_cast<int>(decoded[i]) -
                               static_cast<int>(expected[i]));
        if (d != 0) {
          ++result.mismatches;
          result.max_diff = std::max(result.max_diff, d);
        }
      }
    };
    compare_plane(frame.y(), ref.y);
    compare_plane(frame.u(), ref.u);
    compare_plane(frame.v(), ref.v);
    ++index;
    ++result.frames;
  }
  if (index != recon.recon.size()) {
    error = "decoder produced " + std::to_string(index) + " of " +
            std::to_string(recon.recon.size()) + " frames";
    result.ok = false;
  }
  result.ok = result.ok && result.mismatches == 0;
  return result;
}

struct Phase {
  std::string name;
  std::vector<OwnedFrame> frames;
  av2enc::video::VideoOptions options;
};

void print_stats(const av2enc::cuda::video::VideoGpuStats& stats, double wall_ms) {
  std::cout << "  GPU backend=" << av2enc::cuda::video::video_backend_name()
            << " launches=" << stats.total_launches()
            << " [var=" << stats.variance_launches
            << " me=" << stats.me_launches << " mc=" << stats.mc_launches
            << " sse=" << stats.sse_launches
            << " intra=" << stats.intra_launches
            << " tx=" << stats.tx_launches
            << " recon=" << stats.recon_launches << "]"
            << " me_jobs=" << stats.me_jobs
            << " cand=" << stats.inter_candidates
            << " up=" << (stats.bytes_uploaded / 1024) << "KiB"
            << " down=" << (stats.bytes_downloaded / 1024) << "KiB"
            << " device=" << stats.device_ms << "ms"
            << " wall=" << wall_ms << "ms\n";
  if (!stats.frame_ms.empty()) {
    const char* kind =
        std::string(av2enc::cuda::video::video_backend_name()) == "cuda"
            ? "cuda-event"
            : "proxy";
    std::cout << "  frame ms (" << kind << "):";
    for (double ms : stats.frame_ms) {
      std::cout << " " << ms;
    }
    std::cout << "\n";
  }
}

bool run_phase(const Phase& phase, const std::string& out_path,
               bool require_decoder) {
  std::cout << "== " << phase.name << ": " << phase.frames.size()
            << " frame(s) " << phase.frames[0].width << "x"
            << phase.frames[0].height << " ==\n";

  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(phase.frames.size());
  for (const OwnedFrame& frame : phase.frames) {
    av2enc::video::VideoInputFrame input;
    input.width = frame.width;
    input.height = frame.height;
    input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input);
  }

  std::string error;
  av2enc::video::VideoSequenceResult cpu;
  const auto cpu_start = std::chrono::steady_clock::now();
  if (!av2enc::video::encode_video_sequence(inputs, phase.options, cpu, error)) {
    std::cerr << "CPU ENCODE FAIL: " << error << "\n";
    return false;
  }
  const double cpu_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - cpu_start)
                            .count();

  av2enc::video::VideoSequenceResult gpu;
  av2enc::cuda::video::VideoGpuStats stats;
  const auto gpu_start = std::chrono::steady_clock::now();
  if (!av2enc::cuda::video::encode_video_sequence_gpu(
          inputs, phase.options, gpu, error, &stats)) {
    std::cerr << "GPU ENCODE FAIL: " << error << "\n";
    return false;
  }
  const double gpu_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - gpu_start)
                            .count();

  for (std::size_t i = 0; i < gpu.units.size(); ++i) {
    std::cout << "  frame " << i << ": bytes=" << gpu.frame_bytes[i]
              << " tile=" << gpu.tile_bytes[i]
              << " inter=" << gpu.inter_blocks[i]
              << " intra=" << gpu.intra_blocks[i] << "\n";
  }
  print_stats(stats, gpu_ms);
  std::cout << "  cpu encode " << cpu_ms << "ms, gpu-path wall " << gpu_ms
            << "ms (proxy: "
            << (gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0) << "x cpu)\n";

  const StreamDiff diff = compare_streams(cpu, gpu);
  if (!diff.ok) {
    std::cerr << "CPU/GPU MISMATCH: unit_mismatches=" << diff.unit_mismatches
              << " recon_mismatches=" << diff.recon_mismatches
              << " max_diff=" << diff.max_diff << " first=f"
              << diff.first_frame << " off=" << diff.first_offset << "\n";
    return false;
  }
  std::cout << "  PASS: bitstream byte-identical, recon 0 diffs\n";

  if (!write_ivf(out_path, gpu, phase.frames[0].width, phase.frames[0].height,
                 error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return false;
  }
  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    if (require_decoder) {
      std::cerr << "DECODE SKIP: no dav2d binary found (set DAV2D)\n";
      return false;
    }
    std::cout << "  (no dav2d; decode comparison skipped)\n";
    return true;
  }
  const std::string decoded_path = out_path + ".decoded.y4m";
  const std::string command = dav2d + " -q -i " + out_path + " -o " +
                              decoded_path + " --threads 0";
  std::cout << "  decode: " << command << "\n";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return false;
  }
  std::string decode_error;
  const DecodeDiff decoded = compare_decoded(decoded_path, gpu, decode_error);
  if (!decoded.ok) {
    std::cerr << "DECODE/RECON MISMATCH: frames=" << decoded.frames
              << " mismatches=" << decoded.mismatches
              << " max_diff=" << decoded.max_diff;
    if (!decode_error.empty()) std::cerr << " (" << decode_error << ")";
    std::cerr << "\n";
    return false;
  }
  std::cout << "  PASS: dav2d decoded " << decoded.frames
            << " frame(s), recon exact\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "video_gpu_gate.ivf";
  int frames = 8;
  int selftest = 0;
  av2enc::video::VideoOptions options;
  av2enc::video::VideoOptions hd_options;
  hd_options.force_single_block = true;
  bool skip_hd = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--force-intra") {
      options.force_intra = true;
      hd_options.force_intra = true;
    } else if (arg == "--no-newmv") {
      options.enable_newmv = false;
      hd_options.enable_newmv = false;
    } else if (arg == "--no-intra-fallback") {
      options.enable_intra_fallback = false;
      hd_options.enable_intra_fallback = false;
    } else if (arg == "--force-global") {
      options.force_global_mv = true;
      hd_options.force_global_mv = true;
    } else if (arg == "--single-block") {
      options.force_single_block = true;
      hd_options.force_single_block = true;
    } else if (arg == "--full-tree") {
      options.force_single_block = false;
      hd_options.force_single_block = false;
    } else if (arg == "--inter-residual") {
      set_enable_inter_residual(options);
      set_enable_inter_residual(hd_options);
    } else if (arg == "--force-inter-residual") {
      set_enable_inter_residual(options);
      set_enable_inter_residual(hd_options);
      set_force_inter_residual(options);
      set_force_inter_residual(hd_options);
    } else if (arg == "--drl-reorder" && i + 1 < argc) {
      const int value = std::atoi(argv[++i]);
      set_drl_reorder(options, value);
      set_drl_reorder(hd_options, value);
    } else if (arg == "--no-hd") {
      skip_hd = true;
    } else if (arg == "--trace") {
      options.trace = true;
      hd_options.trace = true;
    } else if (arg == "--selftest" && i + 1 < argc) {
      selftest = std::atoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "usage: video_gate [--input clip.y4m] [--frames N] "
                   "[--out out.ivf] [--single-block|--full-tree] "
                   "[--force-intra] [--no-newmv] [--no-intra-fallback] "
                   "[--force-global] [--trace] [--no-hd] "
                   "[--inter-residual] [--force-inter-residual] "
                   "[--drl-reorder N] [--selftest N]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (frames < 2) frames = 2;

  if (selftest > 0) {
    std::string error;
    if (!av2enc::cuda::video::video_inter_residual_selftest(selftest, error)) {
      std::cerr << "INTER RESIDUAL SELFTEST FAIL: " << error << "\n";
      return 1;
    }
    return 0;
  }

  // Phase 1: 64x64 synthetic clip (2 frames, then the requested length).
  {
    const std::vector<OwnedFrame> clip = make_synthetic_clip(frames, 64, 64);
    std::vector<OwnedFrame> two(clip.begin(), clip.begin() + 2);
    if (!run_phase(Phase{"64x64 two-frame", two, options},
                   out_path + ".64.2f.ivf", true)) {
      std::cerr << "M2 GPU GATE FAIL\n";
      return 1;
    }
    if (clip.size() > 2) {
      if (!run_phase(Phase{"64x64 short clip", clip, options},
                     out_path + ".64.clip.ivf", true)) {
        std::cerr << "M2 GPU GATE FAIL\n";
        return 1;
      }
    }
  }

  // Phase 2: 720p clip when test media is present.
  const std::string hd_path =
      input.empty() ? "testdata/shine_1280x720_60f.y4m" : input;
  if (!skip_hd && std::filesystem::exists(hd_path)) {
    std::vector<OwnedFrame> clip;
    std::string error;
    if (!load_y4m(hd_path, frames, clip, error)) {
      std::cerr << "Y4M LOAD FAIL: " << error << "\n";
      return 2;
    }
    std::vector<OwnedFrame> two(clip.begin(),
                                clip.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        std::min<std::size_t>(2, clip.size())));
    if (!run_phase(Phase{"720p two-frame", two, hd_options},
                   out_path + ".hd.2f.ivf", true)) {
      std::cerr << "M2 GPU GATE FAIL\n";
      return 1;
    }
    if (clip.size() > 2) {
      if (!run_phase(Phase{"720p short clip", clip, hd_options},
                     out_path + ".hd.clip.ivf", true)) {
        std::cerr << "M2 GPU GATE FAIL\n";
        return 1;
      }
    }
  } else if (!skip_hd) {
    std::cout << "== 720p phase skipped: " << hd_path << " not found ==\n";
  }

  std::cout << "M2 GPU GATE PASS\n";
  return 0;
}
