// Decoder gate for AV2 AMVD / adaptive MV precision (§5.20.7.6/.13/.20/.21).
//
// The key and P1 frames come from the M2 core.  P2 is written by
// mv_precision_tile.cpp with 16x16 NEWMV leaves that alternate between
//
//   * AMVD (use_amvd = 1, IsAdaptiveMvd, mv_joint + amvd_index),
//   * flex precision (use_most_probable_precision = 0, pb = 0 -> ONE_PEL),
//   * frame precision (use_most_probable_precision = 1 -> QUARTER),
//
// and a switchable per-block interpolation filter.  The sequence header
// enables all three MV-res tools and codes the §5.4.6 motion-mode slice; every
// inter frame codes the §5.18.2 frame_enabled_motion_modes bit.
//
// The pinned dav2d (`--strict 1 --threads 1`) must decode the stream and every
// sample must equal the encoder reconstruction.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/mc/mc.h"
#include "encoder/inter/mv_precision/gate/mv_precision_stream.h"
#include "encoder/inter/mv_precision/gate/mv_precision_tile.h"
#include "encoder/m2/m2_frame.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace mvprec = av2enc::inter::mvprec;
namespace gate = av2enc::inter::mvprec::gate;
namespace mc = av2enc::inter::mc;

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
    const int sx = 20 + 8 * frame;
    const int sy = 16 + 5 * frame;
    const bool square = x >= sx && x < sx + 40 && y >= sy && y < sy + 40;
    if (square) return static_cast<std::uint8_t>(28 + ((x * 3 + y * 5) & 0x7f));
    return static_cast<std::uint8_t>(96 + ((x + 2 * y) >> 3));
  }
  const bool bright = (x + y + frame * 3) & 1;
  if (plane == 1) return static_cast<std::uint8_t>(bright ? 118 : 138);
  return static_cast<std::uint8_t>(bright ? 142 : 122);
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
        frame.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
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

bool load_y4m(const std::string& path, int frames, std::vector<OwnedFrame>* out,
              std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out->size()) < frames) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    const int w = static_cast<int>(frame.width()) & ~63;
    const int h = static_cast<int>(frame.height()) & ~63;
    if (w <= 0 || h <= 0) {
      *error = "Y4M frame is smaller than 64x64";
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
      std::memcpy(
          owned.y.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(w),
          frame.y().data() +
              static_cast<std::size_t>(y) * static_cast<std::size_t>(sw),
          static_cast<std::size_t>(w));
    }
    const int scw = (sw + 1) / 2;
    for (int y = 0; y < ch; ++y) {
      std::memcpy(
          owned.u.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(cw),
          frame.u().data() +
              static_cast<std::size_t>(y) * static_cast<std::size_t>(scw),
          static_cast<std::size_t>(cw));
      std::memcpy(
          owned.v.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(cw),
          frame.v().data() +
              static_cast<std::size_t>(y) * static_cast<std::size_t>(scw),
          static_cast<std::size_t>(cw));
    }
    out->push_back(std::move(owned));
  }
  return static_cast<int>(out->size()) >= frames;
}

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units, int width,
               int height, std::string* error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  if (!writer.open(path, header, *error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, *error)) return false;
  }
  return writer.close(*error);
}

int64_t compare_plane(std::span<const std::uint8_t> got,
                      std::span<const std::uint8_t> want, int* first_index) {
  const std::size_t n = std::min(got.size(), want.size());
  int64_t diffs = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      if (diffs == 0 && first_index != nullptr) {
        *first_index = static_cast<int>(i);
      }
      ++diffs;
    }
  }
  return diffs + static_cast<int64_t>(got.size() > want.size()
                                          ? got.size() - want.size()
                                          : want.size() - got.size());
}

int g_strict = 1;

bool decode_and_compare(const std::string& dav2d, const std::string& ivf,
                        const std::vector<av2enc::m2::M2ReconFrame>& recons,
                        int64_t* total_diffs, int* first_frame,
                        int* first_index, int* decoded_frames,
                        std::string* error) {
  const std::string y4m = ivf + ".decoded.y4m";
  std::filesystem::remove(y4m);
  const std::string cmd = "\"" + dav2d + "\" -q -i \"" + ivf + "\" -o \"" +
                          y4m + "\" --strict " + std::to_string(g_strict) +
                          " --threads 1";
  const int status = std::system(cmd.c_str());
  if (status != 0) {
    *error = "dav2d decode failed (status " + std::to_string(status) + ")";
    return false;
  }
  av2enc::Y4mReader reader(y4m);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  std::vector<av2enc::Y4mFrame> frames;
  while (true) {
    const av2enc::Y4mReadResult r = reader.read_frame(frame, *error);
    if (r == av2enc::Y4mReadResult::kEof) break;
    if (r == av2enc::Y4mReadResult::kError) return false;
    frames.push_back(frame);
  }
  *decoded_frames = static_cast<int>(frames.size());
  if (frames.size() < recons.size()) {
    *error = "decoded only " + std::to_string(frames.size()) + " frames, want " +
             std::to_string(recons.size());
    return false;
  }
  *total_diffs = 0;
  *first_frame = -1;
  *first_index = -1;
  for (std::size_t i = 0; i < recons.size(); ++i) {
    const av2enc::Y4mFrame& got = frames[i];
    const av2enc::m2::M2ReconFrame& want = recons[i];
    int idx = 0;
    const int64_t dy = compare_plane(got.y(), want.y, &idx);
    const int64_t du = compare_plane(got.u(), want.u, nullptr);
    const int64_t dv = compare_plane(got.v(), want.v, nullptr);
    const int64_t d = dy + du + dv;
    if (d != 0 && *first_frame < 0) {
      *first_frame = static_cast<int>(i);
      *first_index = idx;
    }
    *total_diffs += d;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  std::string out_path = "mv_precision_gate.ivf";
  std::string dav2d;
  int dim = 128;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << name << " requires an argument\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--input") {
      input_path = value("--input");
    } else if (arg == "--out") {
      out_path = value("--out");
    } else if (arg == "--dav2d") {
      dav2d = value("--dav2d");
    } else if (arg == "--dim") {
      dim = std::atoi(value("--dim").c_str());
    } else if (arg == "--strict") {
      g_strict = std::atoi(value("--strict").c_str());
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (dav2d.empty()) {
    const char* env = std::getenv("DAV2D");
    if (env != nullptr) dav2d = env;
  }
  if (dav2d.empty()) {
    std::cerr << "no dav2d: pass --dav2d or set DAV2D\n";
    return 2;
  }

  std::vector<OwnedFrame> clip;
  std::string error;
  if (input_path.empty()) {
    clip = make_synthetic_clip(2, dim, dim);
    std::cout << "== MV-precision gate: synthetic " << dim << "x" << dim
              << " ==\n";
  } else {
    if (!load_y4m(input_path, 2, &clip, &error)) {
      std::cerr << "LOAD FAIL: " << error << "\n";
      return 1;
    }
    std::cout << "== MV-precision gate: " << clip[0].width << "x"
              << clip[0].height << " ==\n";
  }
  const int width = static_cast<int>(clip[0].width);
  const int height = static_cast<int>(clip[0].height);

  std::vector<av2enc::m2::M2InputFrame> inputs;
  for (const OwnedFrame& frame : clip) {
    av2enc::m2::M2InputFrame in;
    in.width = frame.width;
    in.height = frame.height;
    in.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    in.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    in.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(in);
  }

  av2enc::m2::M2Options options;
  options.force_global_mv = true;
  options.enable_intra_fallback = false;
  options.trace = std::getenv("M2_TRACE") != nullptr;
  av2enc::m2::M2SequenceResult m2_result;
  if (!av2enc::m2::encode_m2_sequence(
          std::span<const av2enc::m2::M2InputFrame>(inputs.data(), 2), options,
          m2_result, error)) {
    std::cerr << "M2 ENCODE FAIL: " << error << "\n";
    return 1;
  }
  const auto tile_slice = [&](std::size_t index) {
    const std::vector<std::uint8_t>& unit = m2_result.units[index];
    const std::size_t bytes = m2_result.tile_bytes[index];
    return std::vector<std::uint8_t>(
        unit.begin() + static_cast<std::ptrdiff_t>(unit.size() - bytes),
        unit.end());
  };

  gate::TileConfig tile_config;
  gate::TileOutput tile_out;
  if (!gate::write_tile(width, height, tile_config, &tile_out, &error)) {
    std::cerr << "P2 TILE FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  P2: tile=" << tile_out.payload.size()
            << " leaves=" << tile_out.leaves.size()
            << " amvd=" << tile_out.amvd_blocks
            << " flex=" << tile_out.flex_blocks
            << " frame_precision=" << tile_out.frame_precision_blocks
            << " derived_sign=" << tile_out.derived_sign_blocks << "\n";
  if (tile_out.amvd_blocks == 0 || tile_out.flex_blocks == 0 ||
      tile_out.derived_sign_blocks == 0) {
    std::cerr << "  FAIL: the gate did not exercise every required tool\n";
    return 1;
  }

  gate::StreamConfig stream_config;
  stream_config.width = width;
  stream_config.height = height;
  stream_config.num_ref_frames = 2;
  stream_config.max_drl_bits_minus_1 = 4;
  stream_config.motion_modes.seq_enabled.enabled[mvprec::kMotionInterIntra] =
      true;
  stream_config.motion_modes.seq_frame_motion_modes_present_flag = true;
  stream_config.motion_modes.enable_six_param_warp_delta = false;
  stream_config.mv_res.enable_adaptive_mvd = true;
  stream_config.mv_res.enable_mvd_sign_derive = true;
  stream_config.mv_res.enable_flex_mvres = true;

  gate::StreamInput stream;
  stream.config = stream_config;
  stream.key_tile = tile_slice(0);
  stream.p1_tile = tile_slice(1);
  stream.p2_tile = tile_out.payload;
  gate::StreamOutput stream_out;
  if (!gate::build_stream(stream, &stream_out, &error)) {
    std::cerr << "STREAM FAIL: " << error << "\n";
    return 1;
  }
  if (!write_ivf(out_path, stream_out.units, width, height, &error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }

  // Reconstruct P2 exactly as the decoder will: prediction with the chosen
  // MV/filter over the key frame (reference slot 0), all-zero residual.
  const av2enc::m2::M2ReconFrame& key = m2_result.recon[0];
  const av2enc::m2::M2ReconFrame& p1 = m2_result.recon[1];
  mc::RefFrame ref;
  ref.plane[0] = mc::PlaneView{key.y.data(), width};
  ref.plane[1] = mc::PlaneView{key.u.data(), width / 2};
  ref.plane[2] = mc::PlaneView{key.v.data(), width / 2};
  ref.mi_cols = width / 4;
  ref.mi_rows = height / 4;

  av2enc::m2::M2ReconFrame recon;
  recon.width = static_cast<std::uint32_t>(width);
  recon.height = static_cast<std::uint32_t>(height);
  recon.y.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                 0);
  recon.u.assign(static_cast<std::size_t>(width / 2) *
                     static_cast<std::size_t>(height / 2),
                 0);
  recon.v = recon.u;
  for (const gate::LeafStats& s : tile_out.leaves) {
    const int r = s.mi_row;
    const int c = s.mi_col;
    const int lw = s.w4 * 4;
    const int lh = s.h4 * 4;
    const mc::BlockRect rect{c * 4, r * 4, lw, lh};
    std::vector<std::uint8_t> py(static_cast<std::size_t>(lw) *
                                static_cast<std::size_t>(lh));
    std::vector<std::uint8_t> pu(static_cast<std::size_t>(lw / 2) *
                                static_cast<std::size_t>(lh / 2));
    std::vector<std::uint8_t> pv = pu;
    mc::MutablePlane dst[3];
    dst[0] = mc::MutablePlane{py.data(), lw};
    dst[1] = mc::MutablePlane{pu.data(), lw / 2};
    dst[2] = mc::MutablePlane{pv.data(), lw / 2};
    const mc::InterpFilter filter =
        static_cast<mc::InterpFilter>(s.interp_filter);
    if (mc::predict_block_420(ref, rect, s.mv, filter, dst) !=
        mc::McStatus::kOk) {
      std::cerr << "MC FAIL at (" << r << "," << c << ")\n";
      return 1;
    }
    for (int y = 0; y < lh; ++y) {
      std::memcpy(recon.y.data() +
                      static_cast<std::size_t>(r * 4 + y) *
                          static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(c * 4),
                  py.data() + static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(lw),
                  static_cast<std::size_t>(lw));
    }
    for (int y = 0; y < lh / 2; ++y) {
      std::memcpy(recon.u.data() +
                      static_cast<std::size_t>(r * 2 + y) *
                          static_cast<std::size_t>(width / 2) +
                      static_cast<std::size_t>(c * 2),
                  pu.data() + static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(lw / 2),
                  static_cast<std::size_t>(lw / 2));
      std::memcpy(recon.v.data() +
                      static_cast<std::size_t>(r * 2 + y) *
                          static_cast<std::size_t>(width / 2) +
                      static_cast<std::size_t>(c * 2),
                  pv.data() + static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(lw / 2),
                  static_cast<std::size_t>(lw / 2));
    }
  }

  std::vector<av2enc::m2::M2ReconFrame> recons = {key, p1, recon};
  int64_t diffs = 0;
  int first_frame = -1;
  int first_index = -1;
  int decoded_frames = 0;
  if (!decode_and_compare(dav2d, out_path, recons, &diffs, &first_frame,
                          &first_index, &decoded_frames, &error)) {
    std::cerr << "DECODE FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  decoded=" << decoded_frames << " frames, recon diffs=" << diffs;
  if (diffs != 0) {
    std::cout << " (first at frame " << first_frame << " sample "
              << first_index << ")";
  }
  std::cout << "\n";
  if (diffs != 0) {
    std::cerr << "  FAIL: AMVD/precision stream does not match the "
                 "reconstruction\n";
    return 1;
  }
  std::cout << "  AMVD/adaptive precision: exact reconstruction under dav2d "
               "--strict 1 --threads 1\n";
  return 0;
}
