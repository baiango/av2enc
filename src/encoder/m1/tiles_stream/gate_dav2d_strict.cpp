// dav2d --strict 1 gate for the reusable M1 tiled still writer.
//
// For every source and layout it:
//   1. encodes a real tiled still with tiles_stream::encode();
//   2. assembles the expected full frame from the per-tile visible
//      reconstructions;
//   3. writes an IVF and decodes it with `dav2d --strict 1 --threads 1`;
//   4. requires byte-identical Y/U/V planes (0 differences).
//
// The default matrix covers 1/2/4/8/16 tiles at 640x360 and 1920x1080. 16x2
// (32 tiles) is also run where the frame is wide enough to represent it; at
// 640x360 the uniform-spacing planner rejects 16 columns, which is reported as
// unsupported rather than silently replaced by another layout.
//
// A separate deblock diagnostic runs each source with 2x2 tiles and deblocking
// on. It classifies differences by distance to an interior tile edge so the
// known CPU-reference/decoder cross-tile deblock finding stays visible. The
// exactness matrix itself uses deblock off.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "encoder/m1/m1_still.h"
#include "encoder/m1/tiles_stream/tiles_stream.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

using av2enc::m1::StillPicture;
using av2enc::m1::tiles_stream::Config;
using av2enc::m1::tiles_stream::Frame;
using av2enc::m1::tiles_stream::Reconstruction;
using av2enc::tiles::TileLayout;
using av2enc::tiles::TileLayoutPreset;
using av2enc::tiles::TileRect;

int g_failures = 0;

void fail(const std::string& what) {
  ++g_failures;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

struct Planes {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_first_frame(const std::string& path, Planes* out, std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult result = reader.read_frame(frame, *error);
  if (result != av2enc::Y4mReadResult::kFrame) {
    if (error->empty()) *error = "no frame in " + path;
    return false;
  }
  out->width = static_cast<int>(frame.width());
  out->height = static_cast<int>(frame.height());
  out->y.assign(frame.y().begin(), frame.y().end());
  out->u.assign(frame.u().begin(), frame.u().end());
  out->v.assign(frame.v().begin(), frame.v().end());
  return true;
}

bool write_ivf(const std::string& path, int width, int height, std::span<const std::uint8_t> frame,
               std::string* error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  if (!writer.open(path, header, *error)) return false;
  if (!writer.write_frame(frame, 0, *error)) return false;
  return writer.close(*error);
}

bool run_decoder(const std::string& dav2d, const std::string& ivf, const std::string& y4m,
                 std::string* error) {
  // --strict 1 is the point of this gate: compliance warnings abort.
  const std::string command =
      "'" + dav2d + "' --strict 1 -q -i '" + ivf + "' -o '" + y4m + "' --threads 1 >/dev/null 2>&1";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    *error = "dav2d --strict 1 failed on " + ivf;
    return false;
  }
  return true;
}

struct MismatchStats {
  int total = 0;
  int interior = 0;  // farther than the filter tolerance from a tile edge
};

MismatchStats classify(const std::vector<std::uint8_t>& decoded,
                       const std::vector<std::uint8_t>& recon, int width, int height,
                       const TileLayout& layout, int chroma_shift) {
  MismatchStats stats;
  if (decoded.size() != recon.size()) {
    stats.total = -1;
    stats.interior = -1;
    return stats;
  }
  // Deblock taps reach at most 7 luma samples each side for the longest
  // transforms; chroma coordinates are doubled. Use 16 luma samples so a
  // longest-tap filter plus a one-sample chroma phase offset stays classified
  // as an edge difference.
  const int tolerance = 16;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                static_cast<std::size_t>(x);
      if (decoded[index] == recon[index]) continue;
      ++stats.total;
      const int luma_x = x << chroma_shift;
      const int luma_y = y << chroma_shift;
      bool on_edge = false;
      for (std::size_t c = 1; c < layout.col_starts_sb.size() && !on_edge; ++c) {
        const int edge = layout.col_starts_sb[c] * layout.sb_size;
        if (std::abs(luma_x - edge) <= tolerance) on_edge = true;
      }
      for (std::size_t r = 1; r < layout.row_starts_sb.size() && !on_edge; ++r) {
        const int edge = layout.row_starts_sb[r] * layout.sb_size;
        if (std::abs(luma_y - edge) <= tolerance) on_edge = true;
      }
      if (!on_edge) ++stats.interior;
    }
  }
  return stats;
}

void print_first_mismatches(const char* plane, const std::vector<std::uint8_t>& decoded,
                            const std::vector<std::uint8_t>& recon, int width, int limit) {
  if (decoded.size() != recon.size()) return;
  int printed = 0;
  for (std::size_t i = 0; i < decoded.size() && printed < limit; ++i) {
    if (decoded[i] == recon[i]) continue;
    std::printf("    %s mismatch @ (%d,%d): decoded %u recon %u\n", plane,
                static_cast<int>(i % static_cast<std::size_t>(width)),
                static_cast<int>(i / static_cast<std::size_t>(width)),
                static_cast<unsigned>(decoded[i]), static_cast<unsigned>(recon[i]));
    ++printed;
  }
}

bool encode_case(const Planes& source, const std::string& source_name, const std::string& out_dir,
                 const std::string& dav2d, TileLayoutPreset preset, std::uint8_t base_q,
                 bool deblock, bool exact, const std::string& tag) {
  const std::string layout_name = av2enc::tiles::preset_name(preset);
  std::printf("[%s] %s %dx%d: ", tag.c_str(), layout_name.c_str(), source.width, source.height);

  Config config;
  config.width = source.width;
  config.height = source.height;
  config.preset = preset;
  config.base_q_idx = base_q;
  config.enable_deblock = deblock;

  StillPicture picture;
  picture.width = static_cast<std::uint32_t>(source.width);
  picture.height = static_cast<std::uint32_t>(source.height);
  picture.y = source.y;
  picture.u = source.u;
  picture.v = source.v;

  Frame frame;
  std::string error;
  if (!av2enc::m1::tiles_stream::encode(picture, config, &frame, &error)) {
    std::printf("ENCODE FAIL\n");
    fail(layout_name + ": " + error);
    return false;
  }

  Reconstruction recon;
  if (!av2enc::m1::tiles_stream::assemble_reconstruction(frame, &recon, &error)) {
    std::printf("RECON FAIL\n");
    fail(layout_name + ": " + error);
    return false;
  }

  const std::string stem = std::filesystem::path(source_name).stem().string() + "_" + layout_name +
                           (deblock ? "_deblock" : "");
  const std::string ivf = out_dir + "/" + stem + ".ivf";
  const std::string decoded_path = out_dir + "/" + stem + ".y4m";
  if (!write_ivf(ivf, source.width, source.height, frame.frame, &error)) {
    std::printf("IVF FAIL\n");
    fail(layout_name + ": " + error);
    return false;
  }
  if (!run_decoder(dav2d, ivf, decoded_path, &error)) {
    std::printf("DECODE FAIL\n");
    fail(layout_name + ": " + error);
    return false;
  }
  Planes decoded;
  if (!load_first_frame(decoded_path, &decoded, &error)) {
    std::printf("READ FAIL\n");
    fail(layout_name + ": " + error);
    return false;
  }

  const MismatchStats y_stats =
      classify(decoded.y, recon.y, source.width, source.height, frame.layout, 0);
  const MismatchStats u_stats =
      classify(decoded.u, recon.u, source.width / 2, source.height / 2, frame.layout, 1);
  const MismatchStats v_stats =
      classify(decoded.v, recon.v, source.width / 2, source.height / 2, frame.layout, 1);

  std::printf(
      "%d tiles, size %zu B, tile_size_bytes %d, mismatches Y %d/%d "
      "U %d/%d V %d/%d (total/interior)\n",
      frame.layout.tile_count(), frame.frame.size(), frame.tile_size_bytes, y_stats.total,
      y_stats.interior, u_stats.total, u_stats.interior, v_stats.total, v_stats.interior);

  if (decoded.width != source.width || decoded.height != source.height) {
    fail(layout_name + ": decoded dimensions differ from the source");
    return false;
  }

  if (exact) {
    if (y_stats.total != 0 || u_stats.total != 0 || v_stats.total != 0 || y_stats.total < 0 ||
        u_stats.total < 0 || v_stats.total < 0) {
      print_first_mismatches("Y", decoded.y, recon.y, source.width, 8);
      fail(layout_name + ": decoded planes are not byte-identical");
      return false;
    }
  } else if (y_stats.interior != 0 || u_stats.interior != 0 || v_stats.interior != 0) {
    print_first_mismatches("Y", decoded.y, recon.y, source.width, 8);
    print_first_mismatches("U", decoded.u, recon.u, source.width / 2, 4);
    print_first_mismatches("V", decoded.v, recon.v, source.width / 2, 4);
    fail(layout_name + ": tile-interior samples differ with deblock on");
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> sources;
  std::string dav2d = "tools/bin/dav2d";
  std::string out_dir;
  int base_q = 90;
  bool deblock_diagnostic = true;
  bool deblock_only = false;
  std::string only_layout;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--base-q" && i + 1 < argc) {
      base_q = std::atoi(argv[++i]);
    } else if (arg == "--layout" && i + 1 < argc) {
      only_layout = argv[++i];
    } else if (arg == "--no-deblock-diagnostic") {
      deblock_diagnostic = false;
    } else if (arg == "--deblock-diagnostic-only") {
      deblock_only = true;
    } else if (arg.rfind("--", 0) == 0) {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    } else {
      sources.push_back(arg);
    }
  }
  if (sources.empty()) {
    std::fprintf(stderr,
                 "usage: %s <source.y4m>... [--dav2d PATH] [--out DIR] "
                 "[--base-q Q] [--layout NAME] "
                 "[--no-deblock-diagnostic] [--deblock-diagnostic-only]\n",
                 argv[0]);
    return 2;
  }
  if (out_dir.empty()) {
    out_dir = (std::filesystem::temp_directory_path() / "av2enc-m1-tiles-gate").string();
  }
  std::filesystem::create_directories(out_dir);

  std::string resolved_decoder = dav2d;
  if (!std::filesystem::exists(resolved_decoder)) {
    const std::string macos = dav2d + "-macos";
    if (std::filesystem::exists(macos)) {
      resolved_decoder = macos;
    } else {
      std::fprintf(stderr, "dav2d not found: %s\n", dav2d.c_str());
      return 127;
    }
  }

  static const TileLayoutPreset kMatrix[] = {
      TileLayoutPreset::k1x1, TileLayoutPreset::k2x1, TileLayoutPreset::k2x2,
      TileLayoutPreset::k4x2, TileLayoutPreset::k8x2, TileLayoutPreset::k16x2,
  };

  int cases = 0;
  int passed = 0;
  int unsupported = 0;
  int diag_cases = 0;
  int diag_passed = 0;

  for (const std::string& source_path : sources) {
    Planes source;
    std::string error;
    if (!load_first_frame(source_path, &source, &error)) {
      std::fprintf(stderr, "cannot read %s: %s\n", source_path.c_str(), error.c_str());
      return 1;
    }
    if ((source.width & 1) != 0 || (source.height & 1) != 0) {
      std::fprintf(stderr, "%s: source must have even 4:2:0 dimensions\n", source_path.c_str());
      return 1;
    }

    std::printf("source: %s %dx%d\n", source_path.c_str(), source.width, source.height);
    for (const TileLayoutPreset preset : kMatrix) {
      if (deblock_only) break;
      const std::string layout_name = av2enc::tiles::preset_name(preset);
      if (!only_layout.empty() && only_layout != layout_name) continue;

      TileLayout planned;
      if (!av2enc::m1::tiles_stream::build_layout(source.width, source.height, preset, 64, 7,
                                                  &planned, &error)) {
        ++unsupported;
        std::printf("[strict] %s %dx%d: UNSUPPORTED (%s)\n", layout_name.c_str(), source.width,
                    source.height, error.c_str());
        if (preset != TileLayoutPreset::k16x2) {
          fail("a required layout was rejected: " + layout_name);
        }
        continue;
      }

      ++cases;
      if (encode_case(source, source_path, out_dir, resolved_decoder, preset,
                      static_cast<std::uint8_t>(base_q), /*deblock=*/false,
                      /*exact=*/true, "strict")) {
        ++passed;
      }
    }

    if (deblock_diagnostic && (only_layout.empty() || deblock_only)) {
      std::printf(
          "deblock diagnostic: 2x2 with per-tile deblocking (not part of the "
          "exactness matrix)\n");
      ++diag_cases;
      if (encode_case(source, source_path, out_dir, resolved_decoder, TileLayoutPreset::k2x2,
                      static_cast<std::uint8_t>(base_q), /*deblock=*/true,
                      /*exact=*/false, "deblock-diag")) {
        ++diag_passed;
      }
    }
  }

  std::printf("\ntiled dav2d --strict 1 gate: %d/%d strict case(s) passed", passed, cases);
  if (diag_cases != 0) {
    std::printf(", %d/%d deblock diagnostic(s) run", diag_passed, diag_cases);
  }
  if (unsupported != 0) {
    std::printf(", %d layout(s) unsupported", unsupported);
  }
  std::printf("\n");
  if (g_failures != 0) {
    std::fprintf(stderr, "tiled gate: FAIL (%d failure(s))\n", g_failures);
    return 1;
  }
  std::printf("tiled gate: PASS\n");
  return 0;
}
