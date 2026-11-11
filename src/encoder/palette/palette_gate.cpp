// Minimal palette dav2d gate.
//
// Emits a 64x64 8-bit 4:2:0 still picture split HORZ into two 64x32 leaves.
// Both leaves are palette-coded (the second one exercises the above-palette
// cache), every residual is EOB=0 and chroma is DC.  The palette analysis is
// exact (<= 8 colours), so the decoded luma must match the source exactly.
//
// This is the §5.20.8 / §7.13.4 tool in isolation: no IBC, no CfL, no filters.

#include "encoder/palette/palette.h"
#include "encoder/palette/scc_stream.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 64;

void fill_source(std::vector<std::uint8_t>& y) {
  // Two screen-content rows: flat fills, vertical stripes, and an icon square.
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      std::uint8_t v = (row < 32) ? 24 : 48;
      if ((col / 8) & 1) v = (row < 32) ? 210 : 190;
      if (row >= 8 && row < 24 && col >= 8 && col < 40) v = 255;
      if (row >= 40 && row < 56 && col >= 24 && col < 48) v = 0;
      y[static_cast<std::size_t>(row) * kWidth +
        static_cast<std::size_t>(col)] = v;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string out = "palette_gate.ivf";
  std::string ref;
  bool single = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (arg == "--ref" && i + 1 < argc) {
      ref = argv[++i];
    } else if (arg == "--single") {
      single = true;
    } else {
      std::fprintf(stderr,
                   "usage: %s [--out FILE] [--ref FILE.y4m] [--single]\n",
                   argv[0]);
      return 2;
    }
  }

  std::vector<std::uint8_t> y(static_cast<std::size_t>(kWidth) * kHeight);
  fill_source(y);

  av2enc::scc::SccFramePlan plan;
  plan.config.width = kWidth;
  plan.config.height = kHeight;
  plan.config.allow_intrabc = false;

  av2enc::scc::SbPlan sb;
  std::string error;
  if (single) {
    sb.partition = av2enc::scc::SbPlan::Partition::kNone;
    av2enc::scc::BlockPlan leaf;
    leaf.mi_row = 0;
    leaf.mi_col = 0;
    leaf.bsize = 12;  // BLOCK_64X64
    leaf.kind = av2enc::scc::BlockKind::kPalette;
    if (!av2enc::palette::analyze_palette(
            y.data(), kWidth, 64, 64, 64, 64, nullptr, 0,
            av2enc::palette::PaletteParams{}, leaf.palette, error)) {
      std::fprintf(stderr, "palette analysis failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("single 64x64: palette size=%d colours=%d bits=%d\n",
                leaf.palette.size, leaf.palette.distinct_colors,
                leaf.palette.map_bits);
    sb.leaves.push_back(leaf);
  } else {
    sb.partition = av2enc::scc::SbPlan::Partition::kHorz;
    for (int half = 0; half < 2; ++half) {
      av2enc::scc::BlockPlan leaf;
      leaf.mi_row = half * 8;
      leaf.mi_col = 0;
      leaf.bsize = 11;  // BLOCK_64X32
      leaf.kind = av2enc::scc::BlockKind::kPalette;
      const int y0 = half * 32;
      if (!av2enc::palette::analyze_palette(
              y.data() + static_cast<std::size_t>(y0) * kWidth, kWidth, 64, 32,
              64, 32, nullptr, 0, av2enc::palette::PaletteParams{},
              leaf.palette, error)) {
        std::fprintf(stderr, "palette analysis failed: %s\n", error.c_str());
        return 1;
      }
      if (!leaf.palette.has_palette) {
        std::fprintf(stderr, "source did not produce a palette\n");
        return 1;
      }
      std::printf("leaf %d: palette size=%d colours=%d bits=%d\n", half,
                  leaf.palette.size, leaf.palette.distinct_colors,
                  leaf.palette.map_bits);
      sb.leaves.push_back(leaf);
    }
  }
  plan.sbs.push_back(sb);

  av2enc::scc::SccEncodeResult result;
  if (!av2enc::scc::encode_scc_frame(plan, result, error)) {
    std::fprintf(stderr, "encode failed: %s\n", error.c_str());
    return 1;
  }
  int diff = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (y[i] != result.recon_y[i]) ++diff;
  }
  std::printf("palette gate: recon diff=%d, tile=%zu, frame=%zu bytes\n", diff,
              result.tile_bytes, result.frame.size());
  if (!av2enc::scc::write_scc_ivf(result, plan.config, out, error)) {
    std::fprintf(stderr, "ivf failed: %s\n", error.c_str());
    return 1;
  }
  if (!ref.empty() &&
      !av2enc::scc::write_scc_y4m(result, plan.config, ref, error)) {
    std::fprintf(stderr, "ref failed: %s\n", error.c_str());
    return 1;
  }
  return diff == 0 ? 0 : 1;
}
