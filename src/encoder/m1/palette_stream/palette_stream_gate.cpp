// Minimal M1-compatible palette stream gate.
//
// Emits one legal AV2 still (palette luma + DC chroma + EOB=0 residual) and a
// reference Y4M holding the writer's reconstruction.  build_and_test.sh runs
// dav2d on the IVF and requires the decoded Y4M to be byte-identical.
#include "encoder/m1/palette_stream/palette_stream.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace av2enc::m1::palette_stream;

std::vector<std::uint8_t> make_pattern(int w, int h, int colours) {
  static constexpr std::uint8_t kValues[8] = {8, 48, 96, 152, 200, 226, 244,
                                               255};
  std::vector<std::uint8_t> y(
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  const int area = w * h;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      int idx = colours > 1 ? ((r * w + c) * colours) / area : 0;
      if (idx >= colours) idx = colours - 1;
      if (idx > 7) idx = 7;
      y[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
        static_cast<std::size_t>(c)] = kValues[idx];
    }
  }
  return y;
}

MinimalPaletteLeaf make_leaf(int mi_row, int mi_col, int bsize, int w, int h,
                             int colours, std::string& error) {
  MinimalPaletteLeaf leaf;
  leaf.mi_row = mi_row;
  leaf.mi_col = mi_col;
  leaf.bsize = bsize;
  const auto src = make_pattern(w, h, colours);
  if (!av2enc::palette::analyze_palette(src.data(), w, w, h, w, h, nullptr, 0,
                                         av2enc::palette::PaletteParams{},
                                         leaf.palette, error)) {
    error = "palette analysis failed: " + error;
  }
  return leaf;
}

bool build_plan(const std::string& mode, MinimalPaletteFramePlan& plan,
                std::string& error) {
  plan.flags = palette_screen_content_flags();
  if (mode == "single") {
    plan.width = 64;
    plan.height = 64;
    MinimalPaletteSuperblock sb;
    sb.partition = MinimalPaletteSuperblock::Partition::kNone;
    sb.leaves.push_back(make_leaf(0, 0, 12, 64, 64, 4, error));
    plan.sbs.push_back(sb);
  } else if (mode == "split") {
    plan.width = 64;
    plan.height = 64;
    MinimalPaletteSuperblock sb;
    sb.partition = MinimalPaletteSuperblock::Partition::kHorz;
    sb.leaves.push_back(make_leaf(0, 0, 11, 64, 32, 3, error));
    sb.leaves.push_back(make_leaf(8, 0, 11, 64, 32, 3, error));
    plan.sbs.push_back(sb);
  } else if (mode == "vert") {
    plan.width = 64;
    plan.height = 64;
    MinimalPaletteSuperblock sb;
    sb.partition = MinimalPaletteSuperblock::Partition::kVert;
    sb.leaves.push_back(make_leaf(0, 0, 10, 32, 64, 3, error));
    sb.leaves.push_back(make_leaf(0, 8, 10, 32, 64, 3, error));
    plan.sbs.push_back(sb);
  } else if (mode == "wide") {
    plan.width = 128;
    plan.height = 64;
    for (int sb_col = 0; sb_col < 2; ++sb_col) {
      MinimalPaletteSuperblock sb;
      sb.partition = MinimalPaletteSuperblock::Partition::kNone;
      sb.leaves.push_back(
          make_leaf(0, sb_col * 16, 12, 64, 64, 3, error));
      plan.sbs.push_back(sb);
    }
  } else {
    error = "unknown gate mode: " + mode;
    return false;
  }
  return error.empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "single";
  std::string out;
  std::string ref;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (arg == "--ref" && i + 1 < argc) {
      ref = argv[++i];
    } else {
      std::fprintf(stderr, "usage: %s --mode single|split|vert|wide --out IVF "
                           "--ref Y4M\n",
                   argv[0]);
      return 2;
    }
  }
  if (out.empty() || ref.empty()) {
    std::fprintf(stderr, "--out and --ref are required\n");
    return 2;
  }

  MinimalPaletteFramePlan plan;
  std::string error;
  if (!build_plan(mode, plan, error)) {
    std::fprintf(stderr, "plan: %s\n", error.c_str());
    return 1;
  }
  MinimalPaletteFrameResult result;
  if (!encode_minimal_palette_frame(plan, result, error)) {
    std::fprintf(stderr, "encode: %s\n", error.c_str());
    return 1;
  }
  if (!write_minimal_palette_ivf(result, plan, out, error)) {
    std::fprintf(stderr, "ivf: %s\n", error.c_str());
    return 1;
  }
  if (!write_minimal_palette_y4m(result, plan, ref, error)) {
    std::fprintf(stderr, "y4m: %s\n", error.c_str());
    return 1;
  }
  std::printf("%s: %dx%d tile=%zu bytes frame=%zu bytes transform-skip=1\n",
              mode.c_str(), plan.width, plan.height, result.tile_bytes,
              result.frame.size());
  return 0;
}
