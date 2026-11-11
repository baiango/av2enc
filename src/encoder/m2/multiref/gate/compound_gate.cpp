// Decoder gate for multi-reference / compound prediction.
//
// Encodes a three-frame stream:
//   key (M2 core) -> P1 single ref (M2 core) -> P2 compound (gate writer),
// decodes it with the pinned dav2d and compares every decoded sample with the
// encoder reconstruction.  The P2 frame exercises AVERAGE, CWP, WEDGE and
// DIFFWTD across four superblocks.
//
// Usage:
//   compound_gate [--input clip.y4m] [--out out.ivf] [--no-decode]
//
// Without --input a synthetic 128x128 clip is used.  Input dimensions are
// cropped down to a multiple of 64 (the gate writer codes one 64x64 leaf per
// superblock).
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

#include "encoder/m2/m2_frame.h"
#include "encoder/m2/multiref/compound_predict.h"
#include "encoder/m2/multiref/decide.h"
#include "encoder/m2/multiref/gate/compound_stream.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t synthetic_sample(int x, int y, int frame, int plane) {
  if (plane == 0) {
    // Moving gradient + moving square: enough structure for wedge/diffwtd.
    const int sx = 12 + 6 * frame;
    const int sy = 10 + 4 * frame;
    const bool square = x >= sx && x < sx + 44 && y >= sy && y < sy + 44;
    if (square) {
      return static_cast<std::uint8_t>(32 + ((x * 3 + y * 5) & 0x7f));
    }
    return static_cast<std::uint8_t>(90 + ((x + 2 * y) >> 3));
  }
  const bool bright = (x + y + frame * 3) & 1;
  if (plane == 1) return static_cast<std::uint8_t>(bright ? 120 : 136);
  return static_cast<std::uint8_t>(bright ? 140 : 124);
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
    frame.y.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)] =
            synthetic_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(x)] =
            synthetic_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) + static_cast<std::size_t>(x)] =
            synthetic_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

// Reads up to `frames` frames and crops to a multiple of 64.
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
    const int scw = (sw + 1) / 2;
    const int cw = w / 2;
    const int ch = h / 2;
    owned.y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    owned.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    owned.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < h; ++y) {
      std::memcpy(owned.y.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(w),
                  frame.y().data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(sw),
                  static_cast<std::size_t>(w));
    }
    for (int y = 0; y < ch; ++y) {
      std::memcpy(owned.u.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(cw),
                  frame.u().data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(scw),
                  static_cast<std::size_t>(cw));
      std::memcpy(owned.v.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(cw),
                  frame.v().data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(scw),
                  static_cast<std::size_t>(cw));
    }
    out.push_back(std::move(owned));
  }
  if (out.size() < 3) {
    error = "gate needs three input frames";
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
  bool have_first = false;
  std::uint32_t first_frame = 0;
  int first_plane = 0;
  int first_x = 0;
  int first_y = 0;
  int first_decoded = 0;
  int first_expected = 0;
};

bool compare_y4m(const std::string& path,
                 const std::vector<av2enc::m2::M2ReconFrame>& recon,
                 CompareResult& result, std::string& error,
                 std::vector<OwnedFrame>* decoded_out) {
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
    const av2enc::m2::M2ReconFrame& ref = recon[index];
    if (frame.width() != ref.width || frame.height() != ref.height) {
      error = "decoded dimensions differ from the encoder recon";
      return false;
    }
    const auto compare_plane = [&](std::span<const std::uint8_t> decoded,
                                   const std::vector<std::uint8_t>& expected,
                                   int plane, int stride) {
      if (decoded.size() != expected.size()) {
        result.mismatches += 1;
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
            result.first_x =
                static_cast<int>(i % static_cast<std::size_t>(stride));
            result.first_y =
                static_cast<int>(i / static_cast<std::size_t>(stride));
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
    if (decoded_out != nullptr) {
      OwnedFrame owned;
      owned.width = frame.width();
      owned.height = frame.height();
      owned.y.assign(frame.y().begin(), frame.y().end());
      owned.u.assign(frame.u().begin(), frame.u().end());
      owned.v.assign(frame.v().begin(), frame.v().end());
      decoded_out->push_back(std::move(owned));
    }
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

double psnr_plane(std::span<const std::uint8_t> a,
                  const std::vector<std::uint8_t>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::uint64_t sse = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    sse += static_cast<std::uint64_t>(d * d);
  }
  if (n == 0) return 0.0;
  if (sse == 0) return 99.0;
  const double mse = static_cast<double>(sse) / static_cast<double>(n);
  return 10.0 * std::log10((255.0 * 255.0) / mse);
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
  std::string out_path = "compound_gate.ivf";
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
      std::cerr << "usage: compound_gate [--input clip.y4m] [--out out.ivf] "
                   "[--no-decode]\n";
      return 2;
    }
  }

  std::vector<OwnedFrame> frames;
  std::string error;
  if (!input.empty()) {
    if (!load_y4m(input, 3, frames, error)) {
      std::cerr << "INPUT FAIL: " << error << "\n";
      return 1;
    }
  } else {
    frames = make_synthetic_clip(3, 128, 128);
  }
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  std::cout << "== compound gate: " << width << "x" << height << " ==\n";

  std::vector<av2enc::m2::M2InputFrame> inputs;
  for (const OwnedFrame& frame : frames) {
    av2enc::m2::M2InputFrame in;
    in.width = frame.width;
    in.height = frame.height;
    in.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    in.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    in.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(in);
  }

  av2enc::m2::M2Options options;
  options.force_single_block = true;
  options.enable_intra_fallback = false;
  av2enc::m2::M2SequenceResult m2_result;
  if (!av2enc::m2::encode_m2_sequence(
          std::span<const av2enc::m2::M2InputFrame>(inputs.data(), 2), options,
          m2_result, error)) {
    std::cerr << "M2 ENCODE FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  key: bytes=" << m2_result.frame_bytes[0]
            << " tile=" << m2_result.tile_bytes[0] << "\n";
  std::cout << "  p1:  bytes=" << m2_result.frame_bytes[1]
            << " tile=" << m2_result.tile_bytes[1] << "\n";

  const auto tile_slice = [&](std::size_t index) {
    const std::vector<std::uint8_t>& unit = m2_result.units[index];
    const std::size_t bytes = m2_result.tile_bytes[index];
    return std::span<const std::uint8_t>(unit.data() + unit.size() - bytes,
                                         bytes);
  };

  av2enc::m2::M2ReconFrame& key_recon = m2_result.recon[0];
  av2enc::m2::M2ReconFrame& p1_recon = m2_result.recon[1];

  // Use the decision API to pick the compound mode for every superblock from
  // the actual source frame and the two reference reconstructions.
  av2enc::inter::mc::RefFrame refs[2];
  refs[0].plane[0] = av2enc::inter::mc::PlaneView{key_recon.y.data(), width};
  refs[0].plane[1] = av2enc::inter::mc::PlaneView{key_recon.u.data(), width / 2};
  refs[0].plane[2] = av2enc::inter::mc::PlaneView{key_recon.v.data(), width / 2};
  refs[0].mi_cols = width / 4;
  refs[0].mi_rows = height / 4;
  refs[1].plane[0] = av2enc::inter::mc::PlaneView{p1_recon.y.data(), width};
  refs[1].plane[1] = av2enc::inter::mc::PlaneView{p1_recon.u.data(), width / 2};
  refs[1].plane[2] = av2enc::inter::mc::PlaneView{p1_recon.v.data(), width / 2};
  refs[1].mi_cols = width / 4;
  refs[1].mi_rows = height / 4;

  std::vector<av2enc::m2::multiref::gate::GateLeaf> leaves;
  int mode_counts[3] = {0, 0, 0};  // average, wedge, diffwtd
  for (int by = 0; by + 64 <= height; by += 64) {
    for (int bx = 0; bx + 64 <= width; bx += 64) {
      av2enc::m2::multiref::DecisionInput decision_input;
      decision_input.refs[0] = &refs[0];
      decision_input.refs[1] = &refs[1];
      decision_input.bsize = 12;
      decision_input.luma_rect = av2enc::inter::mc::BlockRect{bx, by, 64, 64};
      decision_input.mi_cols = width / 4;
      decision_input.mi_rows = height / 4;
      decision_input.source = frames[2].y.data() +
                              static_cast<std::size_t>(by) *
                                  static_cast<std::size_t>(width) +
                              static_cast<std::size_t>(bx);
      decision_input.source_stride = width;
      decision_input.options.allow_single = false;
      decision_input.options.seq_enable_imp_msk_bld = true;
      decision_input.options.lambda = 0.0;
      av2enc::m2::multiref::CompoundCandidateInput candidate;
      candidate.ref0 = 0;
      candidate.ref1 = 1;
      candidate.mv[0] = av2enc::inter::mc::Mv{0, 0};
      candidate.mv[1] = av2enc::inter::mc::Mv{0, 0};
      candidate.global_global_mv = true;
      decision_input.compound_candidates.push_back(candidate);
      av2enc::m2::multiref::DecisionResult result;
      if (!av2enc::m2::multiref::decide_compound(decision_input, result, error)) {
        std::cerr << "DECISION FAIL: " << error << "\n";
        return 1;
      }
      av2enc::m2::multiref::gate::GateLeaf leaf;
      leaf.mode = result.compound;
      leaf.drl_index[0] = 0;
      leaf.drl_index[1] = 0;
      if (leaf.mode.type == av2enc::inter::adv::CompoundType::kAverage) {
        leaf.y_mode = leaf.mode.cwp_weight == 8 ? 22 : 19;
        ++mode_counts[0];
      } else if (leaf.mode.type ==
                 av2enc::inter::adv::CompoundType::kWedge) {
        leaf.y_mode = 22;
        ++mode_counts[1];
      } else {
        leaf.y_mode = 22;
        ++mode_counts[2];
      }
      leaves.push_back(leaf);
    }
  }
  // Guarantee stream coverage of every compound family: the first four
  // superblocks use AVERAGE, CWP, WEDGE and DIFFWTD explicitly.  The rest keep
  // the decision output, which is what the quality numbers below measure.
  const std::vector<av2enc::m2::multiref::gate::GateLeaf> coverage =
      av2enc::m2::multiref::gate::make_default_gate_leaves(4);
  for (std::size_t i = 0; i < coverage.size() && i < leaves.size(); ++i) {
    leaves[i] = coverage[i];
  }
  std::cout << "  decision histogram: average=" << mode_counts[0]
            << " wedge=" << mode_counts[1] << " diffwtd=" << mode_counts[2]
            << " (first four blocks overridden for mode coverage)\n";

  av2enc::m2::multiref::gate::CompoundStreamInput stream;
  stream.width = width;
  stream.height = height;
  stream.order_hint = 2;
  stream.key_tile = tile_slice(0);
  stream.p1_tile = tile_slice(1);
  stream.key_recon.y = key_recon.y.data();
  stream.key_recon.u = key_recon.u.data();
  stream.key_recon.v = key_recon.v.data();
  stream.key_recon.stride_y = width;
  stream.key_recon.stride_uv = width / 2;
  stream.p1_recon.y = p1_recon.y.data();
  stream.p1_recon.u = p1_recon.u.data();
  stream.p1_recon.v = p1_recon.v.data();
  stream.p1_recon.stride_y = width;
  stream.p1_recon.stride_uv = width / 2;
  stream.leaves = std::move(leaves);

  av2enc::m2::multiref::gate::CompoundStreamOutput compound;
  if (!av2enc::m2::multiref::gate::build_compound_stream(stream, compound,
                                                         error)) {
    std::cerr << "COMPOUND ENCODE FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  p2:  tile=" << compound.p2_tile_bytes << " blocks="
            << stream.leaves.size() << "\n";

  if (!write_ivf(out_path, compound.units, width, height, error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }

  // Quality proxies: reconstruction vs source for each frame.
  std::vector<av2enc::m2::M2ReconFrame> all_recon;
  all_recon.push_back(key_recon);
  all_recon.push_back(p1_recon);
  av2enc::m2::M2ReconFrame p2;
  p2.width = static_cast<std::uint32_t>(width);
  p2.height = static_cast<std::uint32_t>(height);
  p2.y = compound.recon_y;
  p2.u = compound.recon_u;
  p2.v = compound.recon_v;
  all_recon.push_back(p2);
  for (std::size_t i = 0; i < 3; ++i) {
    const double py = psnr_plane(frames[i].y, all_recon[i].y);
    const double pu = psnr_plane(frames[i].u, all_recon[i].u);
    const double pv = psnr_plane(frames[i].v, all_recon[i].v);
    std::cout << "  psnr frame " << i << ": Y=" << py << " U=" << pu
              << " V=" << pv << " dB\n";
  }
  // Single-reference proxy: P1 recon used for frame 2 (zero-MV single ref).
  const double baseline_y = psnr_plane(frames[2].y, p1_recon.y);
  const double compound_y = psnr_plane(frames[2].y, compound.recon_y);
  std::cout << "  frame2 Y: single-ref(slot1)=" << baseline_y
            << " dB, compound=" << compound_y << " dB, delta="
            << (compound_y - baseline_y) << " dB\n";

  if (!run_decoder) return 0;

  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "DECODER MISSING: set DAV2D=/path/to/dav2d\n";
    return 127;
  }
  const std::string decoded_path = out_path + ".decoded.y4m";
  const std::string command =
      "\"" + dav2d + "\" -q -i \"" + out_path + "\" -o \"" + decoded_path +
      "\" --threads 0";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return 1;
  }
  CompareResult compare;
  std::vector<OwnedFrame> decoded;
  if (!compare_y4m(decoded_path, all_recon, compare, error, &decoded)) {
    std::cerr << "COMPARE FAIL: " << error << "\n";
    return 1;
  }
  if (!compare.ok) {
    std::cerr << "MISMATCH: " << compare.mismatches << " samples, max|diff|="
              << compare.max_diff << " first at frame " << compare.first_frame
              << " plane " << compare.first_plane << " (" << compare.first_x
              << "," << compare.first_y << ") decoded="
              << compare.first_decoded << " expected=" << compare.first_expected
              << "\n";
    return 1;
  }
  std::cout << "  PASS: " << compare.frames
            << " frame(s) decoded, recon exact\n";
  return 0;
}
