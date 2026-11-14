// Standalone M2 gate: encode a 2-frame P-frame stream (and a short clip),
// decode it with the pinned dav2d, and compare the decoder output against the
// encoder's reconstruction sample by sample.
//
// Usage:
//   video_gate [--input clip.y4m] [--frames N] [--out out.ivf]
//           [--force-intra] [--no-newmv] [--force-global] [--trace]
//
// Without --input the gate synthesises a 64x64 moving-square clip, so the CPU
// reference can be checked on a machine with no test media.
#include "pipeline/video/video_frame.h"

#include "core/io/ivf.h"
#include "core/io/y4m.h"
#include "pipeline/video/post/deblock_grid.h"
#include "pipeline/video/video_gme.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
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

std::vector<OwnedFrame> make_illumination_clip(int frames, int width,
                                              int height) {
  // Smooth source plus a constant per-frame brightness offset.  This is the
  // fixture where the --inter-residual rate decision codes residual leaves;
  // the generic square clip at QP 90 is skip-only, which made the old gate
  // vacuous.
  std::vector<OwnedFrame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(static_cast<std::size_t>(width) * height);
    frame.u.assign(static_cast<std::size_t>(cw) * ch, 128);
    frame.v.assign(static_cast<std::size_t>(cw) * ch, 128);
    const int offset = 15 * f;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const double value = 100.0 + 30.0 * std::sin(x / 9.0) +
                             20.0 * std::cos(y / 11.0) + offset;
        int sample = static_cast<int>(value);
        if (sample < 1) sample = 1;
        if (sample > 255) sample = 255;
        frame.y[static_cast<std::size_t>(y) * width + x] =
            static_cast<std::uint8_t>(sample);
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
                          const std::vector<av2enc::video::VideoReconFrame>& recon,
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
    const av2enc::video::VideoReconFrame& ref = recon[index];
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
                        const std::vector<av2enc::video::VideoReconFrame>& recon) {
  PsnrResult out;
  std::uint64_t sse[3] = {0, 0, 0};
  std::uint64_t count[3] = {0, 0, 0};
  const std::size_t n = std::min(frames.size(), recon.size());
  for (std::size_t i = 0; i < n; ++i) {
    const OwnedFrame& src = frames[i];
    const av2enc::video::VideoReconFrame& dec = recon[i];
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
               const av2enc::video::VideoSequenceResult& sequence,
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

std::size_t count_substring(const std::string& text,
                            const std::string& needle) {
  std::size_t count = 0;
  std::size_t pos = text.find(needle);
  while (pos != std::string::npos) {
    ++count;
    pos = text.find(needle, pos + needle.size());
  }
  return count;
}

bool run_phase(const char* name, const std::vector<OwnedFrame>& frames,
               const av2enc::video::VideoOptions& options, const std::string& out_path,
               bool require_decoder) {
  std::cout << "== " << name << ": " << frames.size() << " frame(s) "
            << frames[0].width << "x" << frames[0].height << " ==\n";

  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(frames.size());
  for (const OwnedFrame& frame : frames) {
    av2enc::video::VideoInputFrame input;
    input.width = frame.width;
    input.height = frame.height;
    input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input);
  }

  av2enc::video::VideoSequenceResult sequence;
  std::string error;
  av2enc::video::VideoOptions encode_options = options;
  std::ostringstream trace;
  std::streambuf* saved_cerr = nullptr;
  if (options.enable_inter_residual) {
    encode_options.trace = true;
    saved_cerr = std::cerr.rdbuf(trace.rdbuf());
  }
  const bool encoded =
      av2enc::video::encode_video_sequence(inputs, encode_options, sequence, error);
  if (saved_cerr != nullptr) std::cerr.rdbuf(saved_cerr);
  if (!encoded) {
    std::cerr << "ENCODE FAIL: " << error << "\n";
    return false;
  }
  if (options.enable_inter_residual) {
    const std::size_t residual =
        count_substring(trace.str(), "residual=1");
    const std::size_t candidates =
        count_substring(trace.str(), "cand inter final");
    std::cout << "  inter residual leaves: " << residual << " of "
              << candidates << " candidate(s)\n";
    if (residual == 0) {
      std::cerr << "INTER RESIDUAL FAIL: --inter-residual coded no residual "
                   "leaf (the gate would only exercise skip syntax)\n";
      return false;
    }
  }
  if (!write_ivf(out_path, sequence, frames[0].width, frames[0].height,
                 error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return false;
  }
  for (std::size_t i = 0; i < sequence.frame_bytes.size(); ++i) {
    std::cout << "  coded frame " << i << ": bytes=" << sequence.frame_bytes[i]
              << " tile=" << sequence.tile_bytes[i]
              << " inter=" << sequence.inter_blocks[i]
              << " intra=" << sequence.intra_blocks[i] << "\n";
  }
  std::cout << "  units=" << sequence.units.size()
            << " coded=" << sequence.frame_bytes.size()
            << " output=" << sequence.recon.size()
            << " temporal_candidates=" << sequence.temporal_candidates
            << " tip_block_refs=" << sequence.tip_block_refs << "\n";
  if (options.tip_block_ref && sequence.tip_block_refs == 0) {
    std::cerr << "TIP-BLOCK FAIL: --tip-block-ref coded no "
                 "RefFrame[0] == TIP_FRAME leaf\n";
    return false;
  }
  if (options.enable_ref_frame_mvs && options.num_total_refs > 1 &&
      sequence.frame_bytes.size() > 2 && sequence.temporal_candidates == 0) {
    std::cerr << "TEMPORAL-MV FAIL: --ref-mvs coded no §7.12.2.7/.8 candidate "
                 "with " << options.num_total_refs << " references\n";
    return false;
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
                              decoded_path + " --strict 1 --threads 1";
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
  // Output/display order: every coded picture that the header marks as output
  // must appear exactly once, in coded order; hidden pictures must be absent.
  std::int32_t expected_output = 0;
  for (std::size_t i = 0; i < sequence.frame_output_index.size(); ++i) {
    const std::int32_t index = sequence.frame_output_index[i];
    if (index < 0) continue;
    if (index != expected_output) {
      std::cerr << "OUTPUT ORDER FAIL: coded frame " << i << " maps to "
                << index << ", expected " << expected_output << "\n";
      return false;
    }
    ++expected_output;
  }
  if (expected_output != static_cast<std::int32_t>(sequence.recon.size()) ||
      static_cast<std::uint32_t>(compare.frames) !=
          static_cast<std::uint32_t>(sequence.recon.size())) {
    std::cerr << "OUTPUT COUNT FAIL: coded="
              << sequence.frame_output_index.size()
              << " expected outputs=" << expected_output
              << " recon=" << sequence.recon.size()
              << " decoded=" << compare.frames << "\n";
    return false;
  }
  std::cout << "  PASS: " << compare.frames
            << " output frame(s) decoded, recon exact, display order exact\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "video_gate.ivf";
  int frames = 8;
  av2enc::video::VideoOptions options;
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
    } else if (arg == "--qp" && i + 1 < argc) {
      const int qp = std::atoi(argv[++i]);
      if (qp < 1 || qp > 255) {
        std::cerr << "--qp must be in [1, 255]\n";
        return 2;
      }
      options.base_q_idx = static_cast<std::uint8_t>(qp);
    } else if (arg == "--refs" && i + 1 < argc) {
      const int refs = std::atoi(argv[++i]);
      if (refs < 1 || refs > av2enc::video::kRefsPerFrame) {
        std::cerr << "--refs must be in [1, "
                  << av2enc::video::kRefsPerFrame << "]\n";
        return 2;
      }
      options.num_total_refs = refs;
    } else if (arg == "--num-ref-frames" && i + 1 < argc) {
      const int value = std::atoi(argv[++i]);
      if (value < 4 || value > 8) {
        std::cerr << "--num-ref-frames must be in [4, 8]\n";
        return 2;
      }
      options.num_ref_frames = value;
    } else if (arg == "--subpel-eighth") {
      options.me_subpel_eighth = true;
    } else if (arg == "--mv-precision" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value == "eighth") {
        options.mv_precision = 0;
      } else if (value == "quarter") {
        options.mv_precision = 1;
      } else if (value == "half") {
        options.mv_precision = 2;
      } else {
        std::cerr << "--mv-precision must be eighth, quarter or half\n";
        return 2;
      }
    } else if (arg == "--drl-reorder") {
      options.drl_reorder = true;
    } else if (arg == "--long-term") {
      options.long_term = true;
    } else if (arg == "--olk") {
      options.long_term = true;
      options.olk = true;
    } else if (arg == "--leading" && i + 1 < argc) {
      const int value = std::atoi(argv[++i]);
      if (value < 1 || value > 7) {
        std::cerr << "--leading must be in [1, 7]\n";
        return 2;
      }
      options.leading_frames = value;
    } else if (arg == "--sef") {
      options.sef = true;
    } else if (arg == "--switch") {
      options.ra_switch = true;
    } else if (arg == "--ras") {
      options.ras = true;
    } else if (arg == "--tip") {
      options.enable_tip = true;
    } else if (arg == "--tip-no-hole-fill") {
      options.tip_hole_fill = false;
    } else if (arg == "--tip-block-ref") {
      options.tip_block_ref = true;
      options.force_single_block = true;
    } else if (arg == "--ref-mvs") {
      options.enable_ref_frame_mvs = true;
    } else if (arg == "--b-frames" && i + 1 < argc) {
      options.b_frames = std::atoi(argv[++i]);
    } else if (arg == "--tlayers" && i + 1 < argc) {
      options.temporal_layers = std::atoi(argv[++i]);
    } else if (arg == "--subpu-selftest") {
      std::string selftest_error;
      if (!av2enc::video::post::video_save_subpu_selftest(&selftest_error)) {
        std::cerr << "save_subpu selftest FAIL: " << selftest_error << "\n";
        return 1;
      }
      std::cout << "save_subpu selftest PASS (§7.13.3.15)\n";
      return 0;
    } else if (arg == "--ref-mvs-field-selftest") {
      std::string selftest_error;
      if (!av2enc::video::ref_mvs_field_selftest(&selftest_error)) {
        std::cerr << "ref-mvs field selftest FAIL: " << selftest_error << "\n";
        return 1;
      }
      std::cout << "ref-mvs field selftest PASS (§7.22 per-8x8 warp store)\n";
      return 0;
    } else if (arg == "--trace") {
      options.trace = true;
    } else if (arg == "--help") {
      std::cout << "usage: video_gate [--input clip.y4m] [--frames N] "
                   "[--out out.ivf] [--force-intra] [--no-newmv] "
                   "[--no-intra-fallback] [--force-global] [--single-block] "
                   "[--inter-residual] [--force-inter-residual] "
                   "[--min-block N] [--max-depth N] [--qp N] [--refs N] "
                   "[--num-ref-frames N] "
                   "[--subpel-eighth] [--mv-precision eighth|quarter|half] "
                   "[--drl-reorder] [--long-term] [--olk] [--leading N] "
                   "[--sef] [--switch] [--ras] [--tip] "
                   "[--tip-no-hole-fill] [--tip-block-ref] "
                   "[--ref-mvs-field-selftest] "
                   "[--trace]\n";
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
  } else if (options.enable_inter_residual) {
    // Use a fixture that actually exercises the feature: the generic square
    // clip stays skip-only, so the gate used to pass without coding residual.
    frames_owned = make_illumination_clip(frames, 320, 192);
  } else {
    frames_owned = make_synthetic_clip(frames, 64, 64);
  }

  if (frames_owned.empty()) {
    std::cerr << "no frames\n";
    return 2;
  }

  // TIP phase (--video-tip): key + future anchor P + TIP_FRAME_AS_OUTPUT
  // pictures.  The stream shape needs at least 3 input frames, so it replaces
  // the historical two-frame phase.
  if (options.enable_tip) {
    bool tip_ok = true;
    if (frames_owned.size() < 3) {
      std::cerr << "--tip needs at least 3 input frames\n";
      return 2;
    }
    if (options.tip_block_ref && !options.enable_ref_frame_mvs) {
      std::cerr << "--tip-block-ref needs --ref-mvs (the TIP block lives in "
                   "the TIP-referencing tail P)\n";
      return 2;
    }
    const std::size_t four = std::min<std::size_t>(4, frames_owned.size());
    std::vector<OwnedFrame> four_frames(frames_owned.begin(),
                                        frames_owned.begin() + four);
    tip_ok = run_phase("TIP 4 pics (key + anchor + 2 TIP)", four_frames,
                       options, out_path + ".tip4.ivf", true);
    if (tip_ok && frames_owned.size() >= 8) {
      std::vector<OwnedFrame> eight_frames(frames_owned.begin(),
                                           frames_owned.begin() + 8);
      tip_ok = run_phase("TIP 8 pics (key + anchor + 6 TIP)", eight_frames,
                         options, out_path + ".tip8.ivf", true);
    }
    std::cout << (tip_ok ? "M2 GATE PASS" : "M2 GATE FAIL") << "\n";
    return tip_ok ? 0 : 1;
  }

  // Frame-type phases.  These change the stream shape, so they replace the
  // historical default phases rather than being appended to them.
  if (options.olk || options.long_term || options.leading_frames > 0 ||
      options.ra_switch || options.ras) {
    bool frames_ok = true;
    if (options.ras) {
      if (frames_owned.size() < 3) {
        std::cerr << "--ras needs at least 3 input frames\n";
        return 2;
      }
      std::vector<OwnedFrame> three(frames_owned.begin(),
                                    frames_owned.begin() + 3);
      frames_ok = run_phase("CLK + SWITCH + RAS", three, options,
                            out_path + ".ras.ivf", true);
    } else if (options.ra_switch) {
      if (frames_owned.size() < 2) {
        std::cerr << "--switch needs at least 2 input frames\n";
        return 2;
      }
      std::vector<OwnedFrame> two_frames(frames_owned.begin(),
                                         frames_owned.begin() + 2);
      frames_ok = run_phase("CLK + SWITCH", two_frames, options,
                            out_path + ".switch.ivf", true);
    } else if (options.olk) {
      if (frames_owned.size() < 4) {
        std::cerr << "--olk needs at least 4 input frames\n";
        return 2;
      }
      std::vector<OwnedFrame> four(frames_owned.begin(),
                                   frames_owned.begin() + 4);
      frames_ok = run_phase("open-loop key GOP", four, options,
                            out_path + ".olk.ivf", true);
    } else if (options.leading_frames > 0) {
      if (frames_owned.size() <
          static_cast<std::size_t>(options.leading_frames + 2)) {
        std::cerr << "--leading " << options.leading_frames
                  << " needs at least " << options.leading_frames + 2
                  << " input frames\n";
        return 2;
      }
      frames_ok = run_phase("leading pictures", frames_owned, options,
                            out_path + ".leading.ivf", true);
    } else {
      frames_ok = run_phase("long-term references", frames_owned, options,
                            out_path + ".longterm.ivf", true);
    }
    std::cout << (frames_ok ? "M2 GATE PASS" : "M2 GATE FAIL") << "\n";
    return frames_ok ? 0 : 1;
  }

  // Phase 1: exactly two frames.
  bool two_ok = true;
  bool clip_ok = true;
  const bool exact_shape =
      options.b_frames > 0 || options.temporal_layers > 0;
  if (!exact_shape) {
    std::vector<OwnedFrame> two(
        frames_owned.begin(),
        frames_owned.begin() +
            std::min<std::size_t>(2, frames_owned.size()));
    two_ok = run_phase("two-frame P-frame", two, options, out_path, true);

    // Phase 2: short clip.
    if (frames_owned.size() > 2) {
      clip_ok = run_phase("short clip", frames_owned, options,
                          out_path + ".clip.ivf", true);
    }
  }

  // Phase 3: B frames and temporal layers with the temporal MV field.  These
  // stream shapes need an exact frame count, so they are separate phases.
  bool extra_ok = true;
  if (options.b_frames > 0) {
    const int gop = options.b_frames + 2;
    if (frames_owned.size() != static_cast<std::size_t>(gop)) {
      std::cerr << "--b-frames needs exactly " << gop
                << " input frames for this gate\n";
      return 2;
    }
    extra_ok = run_phase("B frames + temporal MVs", frames_owned, options,
                         out_path + ".b.ivf", true);
  } else if (options.temporal_layers > 0) {
    const int gop = options.temporal_layers;
    if (frames_owned.size() != static_cast<std::size_t>(gop)) {
      std::cerr << "--tlayers needs exactly " << gop
                << " input frames for this gate\n";
      return 2;
    }
    extra_ok = run_phase("temporal layers + temporal MVs", frames_owned,
                         options, out_path + ".tl.ivf", true);
  }

  std::cout << (two_ok && clip_ok && extra_ok ? "M2 GATE PASS" : "M2 GATE FAIL")
            << "\n";
  return two_ok && clip_ok && extra_ok ? 0 : 1;
}
