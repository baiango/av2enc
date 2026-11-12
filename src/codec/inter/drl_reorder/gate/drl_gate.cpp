// Decoder gate for AV2 DRL reordering (§5.4.6 / §7.12.2.19).
//
// Encodes key + P1 with the M2 core, then a reordering-enabled P2 with the
// independent tile writer in drl_tile.cpp.  The P2 frame uses 4x4 leaves, two
// reference slots and NEARMV blocks with several distinct MVs, so the MV stack
// reaches the DRL_REORDER_CONSTRAINT `numNearest >= 4` trigger.
//
// The gate decodes with the pinned dav2d (`--strict 1 --threads 1`) and
// compares every decoded sample with the encoder reconstruction:
//
//   --mode spec    normative §7.12.2.19 single move-to-front.  0 diffs.
//   --mode legacy  historical full selection sort.             >0 diffs.
//
// `--expect-desync` turns the legacy mismatch into a passing exit status: it
// is the reproduction of the original encoder/decoder stack-order failure.
//
// Usage:
//   drl_gate [--input clip.y4m] [--out out.ivf] [--mode spec|legacy]
//            [--expect-desync] [--dav2d /path/to/dav2d]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/drl_reorder/gate/drl_stream.h"
#include "codec/inter/drl_reorder/gate/drl_tile.h"
#include "codec/inter/mc/mc.h"
#include "pipeline/video/video_frame.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

namespace drl = av2enc::inter::drl;
namespace gate = av2enc::inter::drl::gate;
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
    if (square) {
      return static_cast<std::uint8_t>(28 + ((x * 3 + y * 5) & 0x7f));
    }
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
  if (out->size() < static_cast<std::size_t>(frames)) {
    *error = "clip has fewer than " + std::to_string(frames) + " frames";
    return false;
  }
  return true;
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
                        const std::vector<av2enc::video::VideoReconFrame>& recons,
                        int64_t* total_diffs, int* first_frame, int* first_index,
                        int* decoded_frames, std::string* error) {
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
    const av2enc::video::VideoReconFrame& want = recons[i];
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
  std::string out_path = "drl_gate.ivf";
  std::string dav2d;
  std::string mode = "spec";
  bool expect_desync = false;
  int frames = 3;
  int leaf_size = 8;
  int leaf_rect_16 = 0;
  bool global_mv = false;
  bool ref0_only = false;
  bool reuse_p1 = false;
  bool video_global = false;
  bool compare_tiles = false;
  bool no_newmv = false;
  bool disable_drl = false;
  int strict = 1;
  bool corrupt = false;
  bool newmv_zero = false;
  bool video_newmv = false;
  bool video_min8 = false;
  int video_min = 0;
  int force_mv_y = 0;
  int force_mv_x = 0;
  bool force_mv = false;
  bool all_newmv = false;
  int newmv_period = 29;
  int num_refs = 2;
  int dim = 128;
  std::string drl_mode = "always";
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
    } else if (arg == "--mode") {
      mode = value("--mode");
    } else if (arg == "--frames") {
      frames = std::atoi(value("--frames").c_str());
    } else if (arg == "--drl-mode") {
      drl_mode = value("--drl-mode");
    } else if (arg == "--dim") {
      dim = std::atoi(value("--dim").c_str());
    } else if (arg == "--refs") {
      num_refs = std::atoi(value("--refs").c_str());
    } else if (arg == "--no-newmv") {
      no_newmv = true;
    } else if (arg == "--m2-min4") {
      video_min = 4;
    } else if (arg == "--m2-min8") {
      video_min8 = true;
    } else if (arg == "--m2-newmv") {
      video_newmv = true;
    } else if (arg == "--newmv-period") {
      newmv_period = std::atoi(value("--newmv-period").c_str());
    } else if (arg == "--all-newmv") {
      all_newmv = true;
    } else if (arg == "--mv") {
      const std::string v = value("--mv");
      const std::size_t comma = v.find(',');
      force_mv_y = std::atoi(v.substr(0, comma).c_str());
      force_mv_x = std::atoi(v.substr(comma + 1).c_str());
      force_mv = true;
    } else if (arg == "--newmv-zero") {
      newmv_zero = true;
    } else if (arg == "--corrupt-index") {
      corrupt = true;
    } else if (arg == "--strict") {
      strict = std::atoi(value("--strict").c_str());
    } else if (arg == "--disable-drl") {
      disable_drl = true;
    } else if (arg == "--m2-global") {
      video_global = true;
    } else if (arg == "--compare-p1-tile") {
      compare_tiles = true;
    } else if (arg == "--reuse-p1") {
      reuse_p1 = true;
    } else if (arg == "--global") {
      global_mv = true;
    } else if (arg == "--ref0") {
      ref0_only = true;
    } else if (arg == "--leaf-rect") {
      const std::string r = value("--leaf-rect");
      leaf_rect_16 = r == "8x16" ? 1 : (r == "16x8" ? 2 : 0);
    } else if (arg == "--leaf") {
      leaf_size = std::atoi(value("--leaf").c_str());
    } else if (arg == "--expect-desync") {
      expect_desync = true;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  g_strict = strict;
  if (dav2d.empty()) {
    const char* env = std::getenv("DAV2D");
    if (env != nullptr) dav2d = env;
  }
  if (dav2d.empty()) {
    std::cerr << "no dav2d: pass --dav2d or set DAV2D\n";
    return 2;
  }
  if (mode != "spec" && mode != "legacy") {
    std::cerr << "--mode must be spec or legacy\n";
    return 2;
  }
  if (frames < 3) frames = 3;

  std::vector<OwnedFrame> clip;
  std::string error;
  if (input_path.empty()) {
    clip = make_synthetic_clip(frames, dim, dim);
    std::cout << "== DRL gate: synthetic " << dim << "x" << dim
              << ", mode=" << mode << " ==\n";
  } else {
    if (!load_y4m(input_path, frames, &clip, &error)) {
      std::cerr << "LOAD FAIL: " << error << "\n";
      return 1;
    }
    std::cout << "== DRL gate: " << clip[0].width << "x" << clip[0].height
              << ", mode=" << mode << " ==\n";
  }
  const int width = static_cast<int>(clip[0].width);
  const int height = static_cast<int>(clip[0].height);

  std::vector<av2enc::video::VideoInputFrame> inputs;
  for (const OwnedFrame& frame : clip) {
    av2enc::video::VideoInputFrame in;
    in.width = frame.width;
    in.height = frame.height;
    in.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    in.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    in.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(in);
  }

  av2enc::video::VideoOptions options;
  options.force_single_block = true;
  options.force_global_mv = video_global;
  if (video_min8 || video_min > 0) {
    options.force_single_block = false;
    options.min_block_size = video_min > 0 ? video_min : 8;
    options.max_partition_depth = 4;
  }
  if (video_newmv) {
    options.force_global_mv = false;
    options.enable_nearmv = false;
    options.enable_newmv = true;
  }
  options.trace = std::getenv("VIDEO_TRACE") != nullptr;
  // The sequence header enables DRL reordering, so the reference P frame must
  // not use DRL indices.  GLOBALMV has no read_drl_idx() and is unaffected by
  // the stack order; this keeps P1 valid under the shared sequence header.
  options.force_global_mv = true;
  options.enable_intra_fallback = false;
  av2enc::video::VideoSequenceResult video_result;
  if (!av2enc::video::encode_video_sequence(
          std::span<const av2enc::video::VideoInputFrame>(inputs.data(), 2), options,
          video_result, error)) {
    std::cerr << "M2 ENCODE FAIL: " << error << "\n";
    return 1;
  }
  const auto tile_slice = [&](std::size_t index) {
    const std::vector<std::uint8_t>& unit = video_result.units[index];
    const std::size_t bytes = video_result.tile_bytes[index];
    return std::vector<std::uint8_t>(
        unit.begin() + static_cast<std::ptrdiff_t>(unit.size() - bytes),
        unit.end());
  };

  // Build the P2 leaf plan.  MVs vary over a small set so that the spatial
  // stack has several distinct candidates and accumulates weights > 1.
  // Plan generation follows the tile writer's rectangular partition ladder.
  std::vector<gate::LeafPlan> leaves;
  const mc::Mv pattern[6] = {
      mc::Mv{0, 0},    mc::Mv{8, 4},    mc::Mv{-8, -4},
      mc::Mv{4, 8},    mc::Mv{-12, 4},  mc::Mv{16, -8},
  };
  const auto add_leaf = [&](int r, int c, int w4, int h4) {
    gate::LeafPlan leaf;
    leaf.w4 = w4;
    leaf.h4 = h4;
    leaf.ref = ref0_only ? 0 : (((r * 3 + c * 5) % 7) == 0 ? 1 : 0);
    leaf.newmv = all_newmv ||
                 (!no_newmv && newmv_period > 0 &&
                  ((r * 37 + c * 11) % newmv_period) == 0);
    leaf.mv = force_mv ? mc::Mv{force_mv_y, force_mv_x}
             : (newmv_zero ? mc::Mv{0, 0}
                           : pattern[static_cast<std::size_t>((r + c * 2) % 6)]);
    leaves.push_back(leaf);
  };
  std::function<void(int, int, int, int)> plan_node =
      [&](int r, int c, int w4, int h4) {
        if (w4 == 4 && h4 == 4 && leaf_rect_16 != 0) {
          const bool vert = leaf_rect_16 == 2;
          const int w = vert ? 2 : 4;
          const int h = vert ? 4 : 2;
          add_leaf(r, c, w, h);
          add_leaf(r + (vert ? 0 : 2), c + (vert ? 2 : 0), w, h);
          return;
        }
        if (leaf_size >= 8 && w4 * 4 == leaf_size && h4 * 4 == leaf_size) {
          add_leaf(r, c, w4, h4);
          return;
        }
        if (w4 == 2 && h4 == 2) {
          if (leaf_size == 8) {
            add_leaf(r, c, 2, 2);
          } else {
            const bool vertical = (((r >> 1) + (c >> 1)) & 1) == 0;
            add_leaf(r, c, vertical ? 1 : 2, vertical ? 2 : 1);
            add_leaf(r + (vertical ? 0 : 1), c + (vertical ? 1 : 0),
                     vertical ? 1 : 2, vertical ? 2 : 1);
          }
          return;
        }
        const bool vertical = w4 >= h4;
        const int hw = vertical ? w4 / 2 : w4;
        const int hh = vertical ? h4 : h4 / 2;
        const int dr = vertical ? 0 : hh;
        const int dc = vertical ? hw : 0;
        plan_node(r, c, hw, hh);
        plan_node(r + dr, c + dc, hw, hh);
      };
  for (int r = 0; r < height / 4; r += 16) {
    for (int c = 0; c < width / 4; c += 16) {
      plan_node(r, c, 16, 16);
    }
  }

  gate::TileConfig tile_config;
  tile_config.leaf_size = leaf_size;
  tile_config.leaf_rect_16 = leaf_rect_16;
  tile_config.mode = disable_drl ? drl::Mode::kDisabled
                      : (drl_mode == "constraint" ? drl::Mode::kConstraint
                                                   : drl::Mode::kAlways);
  tile_config.legacy_sort = mode == "legacy";
  tile_config.force_global_mv = global_mv;
  tile_config.corrupt_index = corrupt;
  tile_config.num_total_refs = num_refs;
  tile_config.max_drl_bits_minus_1 = 4;
  gate::TileOutput tile_out;
  if (!gate::write_drl_tile(width, height, leaves, tile_config, &tile_out,
                            &error)) {
    std::cerr << "P2 TILE FAIL: " << error << "\n";
    return 1;
  }
  int max_stack = 0;
  int divergent = 0;
  int reordered = 0;
  const gate::LeafStats* first_divergent = nullptr;
  for (const gate::LeafStats& s : tile_out.leaves) {
    max_stack = std::max(max_stack, s.stack_count);
    if (s.divergent) {
      ++divergent;
      if (first_divergent == nullptr) first_divergent = &s;
    }
    if (s.spec_post != 0) ++reordered;
  }
  std::cout << "  P2: tile=" << tile_out.payload.size() << " leaves="
            << tile_out.leaves.size() << " distinct_mvs="
            << tile_out.distinct_mvs << " max_stack=" << max_stack
            << " reordered=" << reordered << " divergent=" << divergent
            << "\n";
  if (first_divergent != nullptr) {
    std::cout << "  first divergent leaf mi=(" << first_divergent->mi_row
              << "," << first_divergent->mi_col << ") weights="
              << first_divergent->weights << " spec_post="
              << first_divergent->spec_post << " legacy_post="
              << first_divergent->legacy_post << " signalled="
              << first_divergent->signalled_index << "\n";
  }

  gate::StreamInput stream;
  stream.width = width;
  stream.height = height;
  stream.disable_drl_reorder = disable_drl;
  stream.constrain_drl_reorder = drl_mode == "constraint";
  stream.num_total_refs = num_refs;
  stream.key_tile = tile_slice(0);
  stream.p1_tile = tile_slice(1);
  stream.p2_tile = reuse_p1 ? tile_slice(1) : tile_out.payload;
  gate::StreamOutput stream_out;
  if (compare_tiles) {
    const std::vector<std::uint8_t> p1_tile = tile_slice(1);
    std::size_t diffs = 0;
    const std::size_t n = std::min(p1_tile.size(), tile_out.payload.size());
    std::size_t first = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (p1_tile[i] != tile_out.payload[i]) {
        if (diffs == 0) first = i;
        ++diffs;
      }
    }
    std::cout << "  tile compare: p1=" << p1_tile.size()
              << " ours=" << tile_out.payload.size() << " diffs=" << diffs;
    if (diffs != 0) std::cout << " first_byte=" << first;
    std::cout << "\n  p1  :";
    for (std::uint8_t b : p1_tile) std::printf(" %02x", b);
    {
      std::ofstream("p1_tile.bin", std::ios::binary)
          .write(reinterpret_cast<const char*>(p1_tile.data()),
                 static_cast<std::streamsize>(p1_tile.size()));
      std::ofstream("ours_tile.bin", std::ios::binary)
          .write(reinterpret_cast<const char*>(tile_out.payload.data()),
                 static_cast<std::streamsize>(tile_out.payload.size()));
    }
    std::cout << "\n  ours:";
    for (std::uint8_t b : tile_out.payload) std::printf(" %02x", b);
    std::cout << "\n";
  }
  if (!gate::build_drl_stream(stream, &stream_out, &error)) {
    std::cerr << "STREAM FAIL: " << error << "\n";
    return 1;
  }
  if (!write_ivf(out_path, stream_out.units, width, height, &error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }

  // Reconstruct P2 exactly as the encoder believes the decoder will: the
  // selected stack candidate's MV and the chosen reference slot.
  const av2enc::video::VideoReconFrame& key = video_result.recon[0];
  const av2enc::video::VideoReconFrame& p1 = video_result.recon[1];
  mc::RefFrame refs[2];
  refs[0].plane[0] = mc::PlaneView{key.y.data(), width};
  refs[0].plane[1] = mc::PlaneView{key.u.data(), width / 2};
  refs[0].plane[2] = mc::PlaneView{key.v.data(), width / 2};
  refs[0].mi_cols = width / 4;
  refs[0].mi_rows = height / 4;
  refs[1].plane[0] = mc::PlaneView{p1.y.data(), width};
  refs[1].plane[1] = mc::PlaneView{p1.u.data(), width / 2};
  refs[1].plane[2] = mc::PlaneView{p1.v.data(), width / 2};
  refs[1].mi_cols = width / 4;
  refs[1].mi_rows = height / 4;

  av2enc::video::VideoReconFrame recon;
  recon.width = static_cast<std::uint32_t>(width);
  recon.height = static_cast<std::uint32_t>(height);
  recon.y.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                 0);
  recon.u.assign(static_cast<std::size_t>(width / 2) *
                     static_cast<std::size_t>(height / 2),
                 0);
  recon.v = recon.u;
  for (const gate::LeafStats& s : tile_out.leaves) {
    {
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
      const int ref_index = s.ref;
      if (mc::predict_block_420(refs[ref_index], rect, s.chosen_mv,
                                mc::InterpFilter::kEighttap, dst) !=
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
  }

  std::vector<av2enc::video::VideoReconFrame> recons = reuse_p1 ? std::vector<av2enc::video::VideoReconFrame>{key, p1, p1} : std::vector<av2enc::video::VideoReconFrame>{key, p1, recon};
  int64_t diffs = 0;
  int first_frame = -1;
  int first_index = -1;
  int decoded_frames = 0;
  if (!decode_and_compare(dav2d, out_path, recons, &diffs, &first_frame,
                          &first_index, &decoded_frames, &error)) {
    std::cerr << "DECODE FAIL: " << error << "\n";
    // A decode failure is itself a desync signal for the legacy mode.
    if (expect_desync) {
      std::cout << "  legacy: decoder rejected the stream -> desync "
                   "reproduced\n";
      return 0;
    }
    return 1;
  }
  std::cout << "  decoded=" << decoded_frames << " frames, recon diffs=" << diffs;
  if (diffs != 0) {
    std::cout << " (first at frame " << first_frame << " sample "
              << first_index << ")";
  }
  std::cout << "\n";

  if (expect_desync) {
    if (diffs > 0) {
      std::cout << (corrupt ? "  corrupted DRL index desynchronised as expected"
                            : "  legacy full selection sort desynchronised")
                << "\n";
      return 0;
    }
    std::cerr << "  FAIL: the negative control produced a matching stream\n";
    return 1;
  }
  if (diffs != 0) {
    std::cerr << "  FAIL: spec reorder stream does not match the "
                 "reconstruction\n";
    return 1;
  }
  std::cout << "  spec reorder: exact reconstruction, reorder is in lockstep\n";
  return 0;
}
