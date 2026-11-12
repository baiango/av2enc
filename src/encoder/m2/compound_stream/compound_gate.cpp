// Decoder gate for the M2 compound / skip-mode inter syntax.
//
// Builds a three-frame stream:
//   frame 0  key            (M2 core)
//   frame 1  P, order_hint 4 (M2 core; decoded before frame 2)
//   frame 2  inter, order_hint 2, refs {0,1}, reference_select = 1
//
// Frame 2 carries one 64x64 leaf per superblock: compound AVERAGE,
// skip_mode, same-reference compound, use_optflow, single_ref for reference 0,
// single_ref for reference 1, WEDGE and DIFFWTD.
//
// It then decodes with `dav2d --strict 1 --threads 1` and compares every
// decoded sample with the encoder reconstruction.  Finally it runs the
// "all disabled" byte-identity control against the M2 core single-reference
// writer.
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

#include "encoder/m2/compound_stream/compound_stream.h"
#include "encoder/m2/compound_stream/compound_syntax.h"
#include "encoder/m2/m2_frame.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

using av2enc::m2::compound::CompoundBlockPlan;
using av2enc::m2::compound::CompoundFrameConfig;
using av2enc::m2::compound::CompoundStreamInput;
using av2enc::m2::compound::CompoundStreamOutput;
using av2enc::m2::compound::RefPlaneView;
using av2enc::m2::multiref::RefFrameMap;

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t synthetic_sample(int x, int y, int frame, int plane) {
  if (plane == 0) {
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
    frame.y.resize(static_cast<std::size_t>(width) *
                   static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) *
                     static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)] = synthetic_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] =
            synthetic_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] =
            synthetic_sample(x * 2, y * 2, f, 2);
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
    const int scw = (sw + 1) / 2;
    const int cw = w / 2;
    const int ch = h / 2;
    owned.y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    owned.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    owned.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < h; ++y) {
      std::memcpy(owned.y.data() + static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(w),
                  frame.y().data() + static_cast<std::size_t>(y) *
                                          static_cast<std::size_t>(sw),
                  static_cast<std::size_t>(w));
    }
    for (int y = 0; y < ch; ++y) {
      std::memcpy(owned.u.data() + static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(cw),
                  frame.u().data() + static_cast<std::size_t>(y) *
                                          static_cast<std::size_t>(scw),
                  static_cast<std::size_t>(cw));
      std::memcpy(owned.v.data() + static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(cw),
                  frame.v().data() + static_cast<std::size_t>(y) *
                                          static_cast<std::size_t>(scw),
                  static_cast<std::size_t>(cw));
    }
    out.push_back(std::move(owned));
  }
  if (static_cast<int>(out.size()) < frames) {
    error = "gate needs " + std::to_string(frames) + " input frames";
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

CompoundBlockPlan base_plan(int sb, int sb_cols) {
  CompoundBlockPlan plan;
  plan.mi_row = (sb / sb_cols) * 16;
  plan.mi_col = (sb % sb_cols) * 16;
  plan.w4 = 16;
  plan.h4 = 16;
  plan.skip_flag = true;
  plan.ref_frame[0] = 0;
  plan.ref_frame[1] = av2enc::m2::compound::kNoneRef;
  plan.y_mode = av2enc::m2::compound::kGlobalMv;
  return plan;
}

std::vector<CompoundBlockPlan> make_leaves(int sb_cols, int sb_rows) {
  std::vector<CompoundBlockPlan> plans;
  const int count = sb_cols * sb_rows;
  for (int sb = 0; sb < count; ++sb) {
    plans.push_back(base_plan(sb, sb_cols));
  }
  using namespace av2enc::m2::compound;
  auto at = [&](int i) -> CompoundBlockPlan& {
    return plans[static_cast<std::size_t>(i % count)];
  };
  at(0).ref_frame[1] = 1;
  at(0).y_mode = kGlobalGlobalMv;

  at(1).skip_mode = true;
  at(1).ref_frame[1] = 1;
  at(1).y_mode = kNearNearMv;
  at(1).compound.cwp_weight = adv::kCwpEqual;

  at(2).ref_frame[0] = 0;
  at(2).ref_frame[1] = 0;
  at(2).y_mode = kGlobalGlobalMv;

  at(3).ref_frame[1] = 1;
  at(3).y_mode = kNearNearMv;
  at(3).use_optflow = true;

  at(4).ref_frame[0] = 0;
  at(4).y_mode = kGlobalMv;

  at(5).ref_frame[0] = 1;
  at(5).y_mode = kGlobalMv;

  at(6).ref_frame[1] = 1;
  at(6).y_mode = kGlobalGlobalMv;
  at(6).compound.type = adv::CompoundType::kWedge;
  at(6).compound.wedge_index = 22;
  at(6).compound.wedge_sign = 0;

  at(7).ref_frame[1] = 1;
  at(7).y_mode = kGlobalGlobalMv;
  at(7).compound.type = adv::CompoundType::kDiffWtd;
  at(7).compound.mask_type = 1;

  if (count >= 10) {
    // NEAR_NEARMV + AVERAGE codes cwp_idx (§5.20.7.6).  Index 1 is a
    // non-equal weight, index 0 is CWP_EQUAL.
    at(8).ref_frame[1] = 1;
    at(8).y_mode = kNearNearMv;
    at(8).compound.cwp_index = 1;
    at(9).ref_frame[1] = 1;
    at(9).y_mode = kNearNearMv;
    at(9).compound.cwp_index = 0;
  }
  return plans;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "compound_stream_gate.ivf";
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
      std::cerr << "usage: compound_stream_gate [--input clip.y4m] "
                   "[--out out.ivf] [--no-decode]\n";
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
    frames = make_synthetic_clip(3, 320, 128);
  }
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  std::cout << "== compound_stream gate: " << width << "x" << height << " ==\n";
  if (width < 320 || height < 128) {
    std::cerr << "gate needs at least 320x128 (10 superblocks)\n";
    return 1;
  }

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
  std::cout << "  key:     bytes=" << m2_result.frame_bytes[0]
            << " tile=" << m2_result.tile_bytes[0] << "\n";
  std::cout << "  future:  bytes=" << m2_result.frame_bytes[1]
            << " tile=" << m2_result.tile_bytes[1] << "\n";

  const auto tile_slice = [&](std::size_t index) {
    const std::vector<std::uint8_t>& unit = m2_result.units[index];
    const std::size_t bytes = m2_result.tile_bytes[index];
    return std::span<const std::uint8_t>(unit.data() + unit.size() - bytes,
                                         bytes);
  };

  CompoundStreamInput stream;
  stream.width = width;
  stream.height = height;
  stream.base_q_idx = 90;
  stream.key_tile = tile_slice(0);
  stream.future_tile = tile_slice(1);
  const av2enc::m2::M2ReconFrame& key_recon = m2_result.recon[0];
  const av2enc::m2::M2ReconFrame& future_recon = m2_result.recon[1];
  stream.key_recon = RefPlaneView{key_recon.y.data(), key_recon.u.data(),
                                  key_recon.v.data(), width, width / 2};
  stream.future_recon = RefPlaneView{future_recon.y.data(), future_recon.u.data(),
                                     future_recon.v.data(), width, width / 2};

  CompoundFrameConfig& cfg = stream.config;
  RefFrameMap& map = cfg.ref_map;
  map.num_total_refs = 2;
  map.ref_frame_idx[0] = 0;
  map.ref_frame_idx[1] = 1;
  map.order_hint = 2;
  map.order_hints[0] = 0;
  map.order_hints[1] = 4;
  map.frame_distance[0] = 2;
  map.frame_distance[1] = -2;
  map.reference_select = true;
  map.num_same_ref_compound = 1;
  cfg.skip_mode_present = true;
  av2enc::m2::compound::derive_skip_mode_frames(map, cfg.skip_mode_frame);
  cfg.opfl_refine_type = av2enc::m2::compound::kRefineSwitchable;
  cfg.enable_masked_compound = true;
  cfg.enable_cwp = true;
  cfg.enable_imp_msk_bld = false;
  stream.leaves = make_leaves(width / 64, height / 64);
  // Diagnostic switches used while bringing the gate up.  Each letter turns one
  // exercised feature into a plain GLOBALMV single-reference block.
  if (const char* drop = std::getenv("CS_DROP")) {
    for (const char* p = drop; *p != '\0'; ++p) {
      int index = -1;
      switch (*p) {
        case 's': index = 1; break;  // skip_mode
        case 'r': index = 2; break;  // same-reference compound
        case 'o': index = 3; break;  // use_optflow
        case 'i': index = 4; break;  // single_ref (first)
        case 'j': index = 5; break;  // single_ref (last)
        case 'm': index = 6; break;  // wedge
        case 'n': index = 7; break;  // diffwtd
        default: break;
      }
      if (index >= 0) {
        stream.leaves[static_cast<std::size_t>(index)] =
            base_plan(index, width / 64);
      }
    }
  }

  CompoundStreamOutput output;
  if (!av2enc::m2::compound::build_compound_stream(stream, output, error)) {
    std::cerr << "STREAM FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  frame2:  tile=" << output.tile_bytes << " bytes\n";

  // Payload-level self check: the independent reader reproduces every
  // decision that the writer emitted.
  std::vector<std::uint8_t> frame2_tile;
  if (!av2enc::m2::compound::write_compound_tile(
          stream.leaves, width, height, cfg, frame2_tile, error)) {
    std::cerr << "TILE FAIL: " << error << "\n";
    return 1;
  }
  std::vector<av2enc::m2::compound::CompoundBlockResult> decoded_blocks;
  if (!av2enc::m2::compound::read_compound_tile(frame2_tile, width, height, cfg,
                                                decoded_blocks, error)) {
    std::cerr << "READ FAIL: " << error << "\n";
    return 1;
  }
  if (decoded_blocks.size() != stream.leaves.size()) {
    std::cerr << "READ FAIL: block count mismatch\n";
    return 1;
  }
  for (std::size_t i = 0; i < decoded_blocks.size(); ++i) {
    if (decoded_blocks[i].skip_mode != stream.leaves[i].skip_mode ||
        decoded_blocks[i].use_optflow != stream.leaves[i].use_optflow ||
        decoded_blocks[i].y_mode != stream.leaves[i].y_mode) {
      std::cerr << "READ FAIL: block " << i << " decision mismatch\n";
      return 1;
    }
  }
  std::cout << "  payload round-trip: " << decoded_blocks.size()
            << " block(s) ok\n";
  if (!av2enc::m2::compound::check_disabled_byte_identity(
          width, height, 90, nullptr, nullptr, error)) {
    std::cerr << "BYTE-IDENTITY FAIL: " << error << "\n";
    return 1;
  }
  bool identical = false;
  std::size_t control_bytes = 0;
  if (!av2enc::m2::compound::check_disabled_byte_identity(
          width, height, 90, &identical, &control_bytes, error)) {
    std::cerr << "BYTE-IDENTITY FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  byte-identity control: " << control_bytes
            << " bytes, identical=" << (identical ? "yes" : "no") << "\n";
  if (!identical) return 1;

  if (!write_ivf(out_path, output.units, width, height, error)) {
    std::cerr << "IVF FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  wrote " << out_path << "\n";

  std::vector<av2enc::m2::M2ReconFrame> all_recon;
  all_recon.push_back(m2_result.recon[0]);
  all_recon.push_back(m2_result.recon[1]);
  av2enc::m2::M2ReconFrame frame2;
  frame2.width = static_cast<std::uint32_t>(width);
  frame2.height = static_cast<std::uint32_t>(height);
  frame2.y = output.recon_y;
  frame2.u = output.recon_u;
  frame2.v = output.recon_v;
  all_recon.push_back(std::move(frame2));
  std::cout << "  psnr frame 0: Y=" << psnr_plane(frames[0].y, all_recon[0].y)
            << " dB\n";
  std::cout << "  psnr frame 1: Y=" << psnr_plane(frames[1].y, all_recon[1].y)
            << " dB\n";
  std::cout << "  psnr frame 2: Y=" << psnr_plane(frames[2].y, all_recon[2].y)
            << " dB\n";

  if (!run_decoder) return 0;
  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "DECODER MISSING: set DAV2D=/path/to/dav2d\n";
    return 127;
  }
  const std::string decoded_path = out_path + ".decoded.y4m";
  std::error_code remove_ec;
  std::filesystem::remove(decoded_path, remove_ec);
  const std::string command = "\"" + dav2d + "\" -q --strict 1 --threads 1 -i \"" +
                              out_path + "\" -o \"" + decoded_path + "\"";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return 1;
  }
  CompareResult compare;
  if (!compare_y4m(decoded_path, all_recon, compare, error)) {
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
