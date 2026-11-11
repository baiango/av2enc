// Standalone M2 gate: encode a 2-frame P-frame stream (and a short clip),
// decode it with the pinned dav2d, and compare the decoder output against the
// encoder's reconstruction sample by sample.
//
// Usage:
//   m2_gate [--input clip.y4m] [--frames N] [--out out.ivf]
//           [--force-intra] [--no-newmv] [--force-global] [--trace]
//
// Without --input the gate synthesises a 64x64 moving-square clip, so the CPU
// reference can be checked on a machine with no test media.
#include "encoder/m2/m2_frame.h"

#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

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
  const int in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 32;
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
    frame.y.resize(static_cast<std::size_t>(width) * height);
    frame.u.resize(static_cast<std::size_t>(cw) * ch);
    frame.v.resize(static_cast<std::size_t>(cw) * ch);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * width + x] =
            square_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * cw + x] =
            square_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * cw + x] =
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

struct CompareResult {
  bool ok = true;
  std::uint64_t mismatches = 0;
  int max_diff = 0;
  std::uint32_t frames = 0;
  bool have_first = false;
  std::uint32_t first_frame = 0;
  int first_plane = 0;
  int first_x = 0;
  int first_y = 0;
  int first_decoded = 0;
  int first_expected = 0;
};

CompareResult compare_y4m(const std::string& decoded_path,
                          const std::vector<av2enc::m2::M2ReconFrame>& recon,
                          std::string& error) {
  CompareResult result;
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
    if (index >= recon.size()) {
      error = "decoder produced more frames than the encoder";
      result.ok = false;
      return result;
    }
    const av2enc::m2::M2ReconFrame& ref = recon[index];
    if (frame.width() != ref.width || frame.height() != ref.height) {
      error = "decoded frame dimensions differ from the encoder recon";
      result.ok = false;
      return result;
    }
    const auto compare_plane = [&](std::span<const std::uint8_t> decoded,
                                   const std::vector<std::uint8_t>& expected,
                                   int plane, int stride) {
      if (decoded.size() != expected.size()) {
        result.ok = false;
        return;
      }
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        const int diff = std::abs(static_cast<int>(decoded[i]) -
                                  static_cast<int>(expected[i]));
        if (diff != 0) {
          ++result.mismatches;
          result.max_diff = std::max(result.max_diff, diff);
          if (!result.have_first) {
            result.have_first = true;
            result.first_frame = static_cast<std::uint32_t>(index);
            result.first_plane = plane;
            result.first_x = static_cast<int>(i % static_cast<std::size_t>(stride));
            result.first_y = static_cast<int>(i / static_cast<std::size_t>(stride));
            result.first_decoded = decoded[i];
            result.first_expected = expected[i];
          }
        }
      }
    };
    compare_plane(frame.y(), ref.y, 0, static_cast<int>(frame.width()));
    compare_plane(frame.u(), ref.u, 1,
                  static_cast<int>((frame.width() + 1U) / 2U));
    compare_plane(frame.v(), ref.v, 2,
                  static_cast<int>((frame.width() + 1U) / 2U));
    ++index;
    ++result.frames;
  }
  if (!result.ok) {
    if (error.empty()) error = "plane size mismatch";
    return result;
  }
  if (index != recon.size()) {
    error = "decoder produced " + std::to_string(index) + " of " +
            std::to_string(recon.size()) + " frames";
    result.ok = false;
  }
  result.ok = result.ok && result.mismatches == 0;
  return result;
}

// PSNR of the encoder reconstruction against the source, per plane, in dB.
// Reported by every gate phase (bytes + quality); it is informational and does
// not affect pass/fail.
struct PsnrResult {
  double y = 0.0;
  double u = 0.0;
  double v = 0.0;
  double yuv = 0.0;
};

PsnrResult compute_psnr(const std::vector<OwnedFrame>& frames,
                        const std::vector<av2enc::m2::M2ReconFrame>& recon) {
  PsnrResult out;
  std::uint64_t sse[3] = {0, 0, 0};
  std::uint64_t count[3] = {0, 0, 0};
  const std::size_t n = std::min(frames.size(), recon.size());
  for (std::size_t i = 0; i < n; ++i) {
    const OwnedFrame& src = frames[i];
    const av2enc::m2::M2ReconFrame& dec = recon[i];
    const auto accum = [](std::span<const std::uint8_t> a,
                          const std::vector<std::uint8_t>& b,
                          std::uint64_t& sum, std::uint64_t& cnt) {
      for (std::size_t k = 0; k < a.size() && k < b.size(); ++k) {
        const int d = static_cast<int>(a[k]) - static_cast<int>(b[k]);
        sum += static_cast<std::uint64_t>(d * d);
        ++cnt;
      }
    };
    accum(std::span<const std::uint8_t>(src.y), dec.y, sse[0], count[0]);
    accum(std::span<const std::uint8_t>(src.u), dec.u, sse[1], count[1]);
    accum(std::span<const std::uint8_t>(src.v), dec.v, sse[2], count[2]);
  }
  const auto to_db = [](std::uint64_t sum, std::uint64_t cnt) {
    if (cnt == 0 || sum == 0) return 99.0;
    const double mse = static_cast<double>(sum) / static_cast<double>(cnt);
    return 10.0 * std::log10((255.0 * 255.0) / mse);
  };
  out.y = to_db(sse[0], count[0]);
  out.u = to_db(sse[1], count[1]);
  out.v = to_db(sse[2], count[2]);
  const std::uint64_t total_sse = sse[0] + sse[1] + sse[2];
  const std::uint64_t total_count = count[0] + count[1] + count[2];
  out.yuv = to_db(total_sse, total_count);
  return out;
}

bool write_ivf(const std::string& path,
               const av2enc::m2::M2SequenceResult& sequence,
               std::uint32_t width, std::uint32_t height, std::string& error) {  av2enc::IvfWriter writer;
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

bool run_phase(const char* name, const std::vector<OwnedFrame>& frames,
               const av2enc::m2::M2Options& options, const std::string& out_path,
               bool require_decoder) {
  std::cout << "== " << name << ": " << frames.size() << " frame(s) "
            << frames[0].width << "x" << frames[0].height << " ==\n";

  std::vector<av2enc::m2::M2InputFrame> inputs;
  inputs.reserve(frames.size());
  for (const OwnedFrame& frame : frames) {
    av2enc::m2::M2InputFrame input;
    input.width = frame.width;
    input.height = frame.height;
    input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input);
  }

  av2enc::m2::M2SequenceResult sequence;
  std::string error;
  if (!av2enc::m2::encode_m2_sequence(inputs, options, sequence, error)) {
    std::cerr << "ENCODE FAIL: " << error << "\n";
    return false;
  }
  if (!write_ivf(out_path, sequence, frames[0].width, frames[0].height,
                 error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return false;
  }
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    std::cout << "  frame " << i << ": bytes=" << sequence.frame_bytes[i]
              << " tile=" << sequence.tile_bytes[i]
              << " inter=" << sequence.inter_blocks[i]
              << " intra=" << sequence.intra_blocks[i] << "\n";
  }
  {
    std::uint64_t total_bytes = 0;
    for (const std::uint32_t bytes : sequence.frame_bytes) {
      total_bytes += bytes;
    }
    const PsnrResult psnr = compute_psnr(frames, sequence.recon);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(3);
    std::cout << "  total_bytes=" << total_bytes
              << " psnr_y=" << psnr.y << " psnr_u=" << psnr.u
              << " psnr_v=" << psnr.v << " psnr_yuv=" << psnr.yuv << "\n";
    std::cout.unsetf(std::ios::fixed);
  }

  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    if (require_decoder) {
      std::cerr << "DECODE SKIP: no dav2d binary found (set DAV2D)\n";
      return false;
    }
    std::cout << "  (no dav2d; recon comparison skipped)\n";
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
  const CompareResult compare = compare_y4m(decoded_path, sequence.recon, error);
  if (!compare.ok) {
    std::cerr << "RECON MISMATCH: frames=" << compare.frames
              << " mismatches=" << compare.mismatches
              << " max_diff=" << compare.max_diff;
    if (compare.have_first) {
      std::cerr << " first=f" << compare.first_frame << " p"
                << compare.first_plane << " (" << compare.first_x << ","
                << compare.first_y << ") dec=" << compare.first_decoded
                << " exp=" << compare.first_expected;
    }
    if (!error.empty()) std::cerr << " (" << error << ")";
    std::cerr << "\n";
    return false;
  }
  std::cout << "  PASS: " << compare.frames
            << " frame(s) decoded, recon exact\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "m2_gate.ivf";
  int frames = 8;
  av2enc::m2::M2Options options;
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
    } else if (arg == "--no-newmv") {
      options.enable_newmv = false;
    } else if (arg == "--no-intra-fallback") {
      options.enable_intra_fallback = false;
    } else if (arg == "--force-global") {
      options.force_global_mv = true;
    } else if (arg == "--single-block") {
      options.force_single_block = true;
    } else if (arg == "--inter-residual") {
      options.enable_inter_residual = true;
    } else if (arg == "--force-inter-residual") {
      options.enable_inter_residual = true;
      options.force_inter_residual = true;
    } else if (arg == "--min-block" && i + 1 < argc) {
      options.min_block_size = std::atoi(argv[++i]);
    } else if (arg == "--max-depth" && i + 1 < argc) {
      options.max_partition_depth = std::atoi(argv[++i]);
    } else if (arg == "--trace") {
      options.trace = true;
    } else if (arg == "--help") {
      std::cout << "usage: m2_gate [--input clip.y4m] [--frames N] "
                   "[--out out.ivf] [--force-intra] [--no-newmv] "
                   "[--no-intra-fallback] [--force-global] [--single-block] "
                   "[--inter-residual] [--force-inter-residual] "
                   "[--min-block N] [--max-depth N] [--trace]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (frames < 2) frames = 2;

  std::vector<OwnedFrame> frames_owned;
  if (!input.empty()) {
    std::string error;
    if (!load_y4m(input, frames, frames_owned, error)) {
      std::cerr << "Y4M LOAD FAIL: " << error << "\n";
      return 2;
    }
  } else {
    frames_owned = make_synthetic_clip(frames, 64, 64);
  }

  if (frames_owned.empty()) {
    std::cerr << "no frames\n";
    return 2;
  }

  // Phase 1: exactly two frames.
  std::vector<OwnedFrame> two(frames_owned.begin(),
                              frames_owned.begin() +
                                  std::min<std::size_t>(2, frames_owned.size()));
  const bool two_ok =
      run_phase("two-frame P-frame", two, options, out_path, true);

  // Phase 2: short clip.
  bool clip_ok = true;
  if (frames_owned.size() > 2) {
    clip_ok = run_phase("short clip", frames_owned, options,
                        out_path + ".clip.ivf", true);
  }

  std::cout << (two_ok && clip_ok ? "M2 GATE PASS" : "M2 GATE FAIL") << "\n";
  return two_ok && clip_ok ? 0 : 1;
}
