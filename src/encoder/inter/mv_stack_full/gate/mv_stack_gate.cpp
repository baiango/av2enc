// dav2d gate for the complete §7.12.2 MV stack and allow_warpmv_mode.
//
//   ./mv_stack_gate [--input clip.y4m] [--out mv_stack_gate.ivf] [--no-decode]
//
// The stream is a four-frame AV2 sequence: key (M2 core) -> P1 (M2 core) ->
// P2 (this module, single reference) -> P3 (this module, NumTotalRefs = 3,
// allow_warpmv_mode = 1, DELTAWARP enabled).  P3's 64x64 leaves exercise:
//   * the §7.12.2.20 MVP-combination candidates,
//   * the §7.12.2.20 global-MV candidate,
//   * §7.12.2.9/.11 warp candidates + WARPMV (is_warp / warp_mv / warp_idx).
//
// It then decodes with `dav2d --strict 1 --threads 1` and requires every
// decoded sample to equal the encoder reconstruction.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/mv_stack_full/gate/gate_stream.h"
#include "encoder/m2/m2_frame.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

namespace gate = av2enc::inter::mv_stack_full::gate;

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y, u, v;
};

std::uint8_t synthetic_sample(int x, int y, int frame, int plane) {
  if (plane == 0) {
    const int sx = 20 + 9 * frame;
    const int sy = 14 + 6 * frame;
    const bool square = x >= sx && x < sx + 52 && y >= sy && y < sy + 52;
    if (square) {
      return static_cast<std::uint8_t>(40 + ((x * 5 + y * 3 + frame * 11) & 0x7f));
    }
    return static_cast<std::uint8_t>(92 + ((x + 2 * y + frame * 5) >> 3));
  }
  const bool bright = (x + y + frame) & 1;
  if (plane == 1) return static_cast<std::uint8_t>(bright ? 118 : 138);
  return static_cast<std::uint8_t>(bright ? 142 : 122);
}

std::vector<OwnedFrame> make_synthetic_clip(int frames, int width, int height) {
  std::vector<OwnedFrame> out;
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)] = synthetic_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] = synthetic_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] = synthetic_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

bool load_y4m(const std::string& path, int frames, std::vector<OwnedFrame>& out,
              std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out.size()) < frames) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    const int w = static_cast<int>(frame.width()) & ~63;
    const int h = static_cast<int>(frame.height()) & ~63;
    if (w <= 0 || h <= 0) {
      error = "Y4M frame is smaller than 64x64";
      return false;
    }
    OwnedFrame owned;
    owned.width = static_cast<std::uint32_t>(w);
    owned.height = static_cast<std::uint32_t>(h);
    const int sw = static_cast<int>(frame.width());
    const int cw = w / 2;
    const int ch = h / 2;
    owned.y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    owned.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    owned.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < h; ++y) {
      std::memcpy(owned.y.data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(w),
                  frame.y().data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(sw),
                  static_cast<std::size_t>(w));
    }
    const int scw = (sw + 1) / 2;
    for (int y = 0; y < ch; ++y) {
      std::memcpy(owned.u.data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(cw),
                  frame.u().data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(scw),
                  static_cast<std::size_t>(cw));
      std::memcpy(owned.v.data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(cw),
                  frame.v().data() +
                      static_cast<std::size_t>(y) * static_cast<std::size_t>(scw),
                  static_cast<std::size_t>(cw));
    }
    out.push_back(std::move(owned));
  }
  if (out.size() < 4) {
    error = "gate needs four input frames";
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
  bool ok = false;
  std::uint64_t mismatches = 0;
  int max_diff = 0;
  std::uint32_t frames = 0;
  int first_frame = -1;
  int first_plane = 0;
  int first_x = 0;
  int first_y = 0;
};

bool compare_y4m(const std::string& path, const std::vector<OwnedFrame>& recon,
                 CompareResult& result, std::string& error) {
  result = CompareResult{};
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    if (index >= recon.size()) {
      error = "decoder produced more frames than the encoder";
      return false;
    }
    const OwnedFrame& ref = recon[index];
    if (frame.width() != ref.width || frame.height() != ref.height) {
      error = "decoded dimensions differ";
      return false;
    }
    const auto compare_plane = [&](std::span<const std::uint8_t> decoded,
                                   const std::vector<std::uint8_t>& expected,
                                   int plane, int stride) {
      for (std::size_t i = 0; i < decoded.size() && i < expected.size(); ++i) {
        const int diff = std::abs(static_cast<int>(decoded[i]) -
                                  static_cast<int>(expected[i]));
        if (diff != 0) {
          ++result.mismatches;
          result.max_diff = std::max(result.max_diff, diff);
          if (result.first_frame < 0) {
            result.first_frame = static_cast<int>(index);
            result.first_plane = plane;
            result.first_x = static_cast<int>(i % static_cast<std::size_t>(stride));
            result.first_y = static_cast<int>(i / static_cast<std::size_t>(stride));
          }
        }
      }
    };
    compare_plane(frame.y(), ref.y, 0, static_cast<int>(frame.width()));
    compare_plane(frame.u(), ref.u, 1, static_cast<int>((frame.width() + 1U) / 2U));
    compare_plane(frame.v(), ref.v, 2, static_cast<int>((frame.width() + 1U) / 2U));
    ++index;
    ++result.frames;
  }
  if (index != recon.size()) {
    error = "decoder produced " + std::to_string(index) + " of " +
            std::to_string(recon.size()) + " frames";
    return false;
  }
  result.ok = result.mismatches == 0;
  return true;
}

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units, int width,
               int height, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(path, header, error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) return false;
  }
  return writer.close(error);
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "mv_stack_gate.ivf";
  bool run_decoder = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--no-decode") {
      run_decoder = false;
    } else {
      std::cerr << "usage: mv_stack_gate [--input clip.y4m] [--out out.ivf] "
                   "[--no-decode]\n";
      return 2;
    }
  }

  std::vector<OwnedFrame> frames;
  std::string error;
  if (!input.empty()) {
    if (!load_y4m(input, 4, frames, error)) {
      std::cerr << "INPUT FAIL: " << error << "\n";
      return 1;
    }
  } else {
    frames = make_synthetic_clip(4, 192, 192);
  }
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  std::cout << "== mv_stack_full gate: " << width << "x" << height << " ==\n";

  std::vector<av2enc::m2::M2InputFrame> inputs;
  for (int i = 0; i < 2; ++i) {
    av2enc::m2::M2InputFrame in;
    in.width = frames[static_cast<std::size_t>(i)].width;
    in.height = frames[static_cast<std::size_t>(i)].height;
    in.y = std::span<const std::uint8_t>(frames[static_cast<std::size_t>(i)].y.data(),
                                        frames[static_cast<std::size_t>(i)].y.size());
    in.u = std::span<const std::uint8_t>(frames[static_cast<std::size_t>(i)].u.data(),
                                        frames[static_cast<std::size_t>(i)].u.size());
    in.v = std::span<const std::uint8_t>(frames[static_cast<std::size_t>(i)].v.data(),
                                        frames[static_cast<std::size_t>(i)].v.size());
    inputs.push_back(in);
  }
  av2enc::m2::M2Options options;
  options.force_single_block = true;
  options.enable_intra_fallback = false;
  av2enc::m2::M2SequenceResult m2_result;
  if (!av2enc::m2::encode_m2_sequence(inputs, options, m2_result, error)) {
    std::cerr << "M2 ENCODE FAIL: " << error << "\n";
    return 1;
  }
  const auto tile_slice = [&](std::size_t index) {
    const std::vector<std::uint8_t>& unit = m2_result.units[index];
    const std::size_t bytes = m2_result.tile_bytes[index];
    return std::span<const std::uint8_t>(unit.data() + unit.size() - bytes, bytes);
  };

  gate::StreamInput stream;
  stream.width = width;
  stream.height = height;
  stream.order_hint = 3;
  stream.key_tile = tile_slice(0);
  stream.p1_tile = tile_slice(1);
  stream.key_recon.y = m2_result.recon[0].y.data();
  stream.key_recon.u = m2_result.recon[0].u.data();
  stream.key_recon.v = m2_result.recon[0].v.data();
  stream.key_recon.stride_y = width;
  stream.key_recon.stride_uv = width / 2;
  stream.p1_recon.y = m2_result.recon[1].y.data();
  stream.p1_recon.u = m2_result.recon[1].u.data();
  stream.p1_recon.v = m2_result.recon[1].v.data();
  stream.p1_recon.stride_y = width;
  stream.p1_recon.stride_uv = width / 2;

  gate::StreamOutput gated;
  if (!gate::build_stream(stream, gated, error)) {
    std::cerr << "STREAM FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  p2 tile=" << gated.p2_tile_bytes
            << "  p3 tile=" << gated.p3_tile_bytes << " (multi-ref)\n";
  std::cout << "  coverage: is_warp_bits=" << gated.coverage.is_warp_bits
            << " warp-blocks=" << gated.coverage.warpmv
            << " global=" << gated.coverage.global_mv
            << " near-extra=" << gated.coverage.near_extra
            << " near-neighbour=" << gated.coverage.near_neighbour
            << " stack-grew=" << gated.coverage.mvp_combinations
            << " max-stack=" << gated.coverage.max_stack
            << " max-warp=" << gated.coverage.max_warp_found << "\n";
  if (!write_ivf(out_path, gated.units, width, height, error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }
  const bool warp_blocks = std::getenv("GATE_WARP_BLOCKS") != nullptr;
  const bool skip_coverage = std::getenv("GATE_SKIP_COVERAGE") != nullptr;
  if (!skip_coverage &&
      (gated.coverage.is_warp_bits < 1 || gated.coverage.global_mv < 1 ||
       gated.coverage.near_extra < 1 || gated.coverage.mvp_combinations < 1 ||
       (warp_blocks && gated.coverage.warpmv < 1))) {
    std::cerr << "COVERAGE FAIL: the stream does not exercise every required "
                 "§7.12.2.20 path\n";
    return 1;
  }


  std::vector<OwnedFrame> recon(4);
  recon[0].width = recon[1].width = recon[2].width = recon[3].width =
      static_cast<std::uint32_t>(width);
  recon[0].height = recon[1].height = recon[2].height = recon[3].height =
      static_cast<std::uint32_t>(height);
  recon[0].y = m2_result.recon[0].y;
  recon[0].u = m2_result.recon[0].u;
  recon[0].v = m2_result.recon[0].v;
  recon[1].y = m2_result.recon[1].y;
  recon[1].u = m2_result.recon[1].u;
  recon[1].v = m2_result.recon[1].v;
  recon[2].y = gated.recon2_y;
  recon[2].u = gated.recon2_u;
  recon[2].v = gated.recon2_v;
  recon[3].y = gated.recon_y;
  recon[3].u = gated.recon_u;
  recon[3].v = gated.recon_v;

  if (!run_decoder) return 0;

  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "DECODER MISSING: set DAV2D=/path/to/dav2d\n";
    return 127;
  }
  const std::string decoded_path = out_path + ".decoded.y4m";
  const std::string command = "\"" + dav2d + "\" -q -i \"" + out_path +
                              "\" -o \"" + decoded_path +
                              "\" --strict 1 --threads 1";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return 1;
  }
  CompareResult compare;
  if (!compare_y4m(decoded_path, recon, compare, error)) {
    std::cerr << "COMPARE FAIL: " << error << "\n";
    return 1;
  }
  if (!compare.ok) {
    std::cerr << "MISMATCH: " << compare.mismatches << " samples, max|diff|="
              << compare.max_diff << " first at frame " << compare.first_frame
              << " plane " << compare.first_plane << " (" << compare.first_x
              << "," << compare.first_y << ")\n";
    return 1;
  }
  std::cout << "  PASS: " << compare.frames
            << " frame(s) decoded with --strict 1 --threads 1, recon exact\n";
  return 0;
}
