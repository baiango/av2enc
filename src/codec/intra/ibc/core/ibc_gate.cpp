// Minimal IBC dav2d gate.
//
// Emits a 768x128 (12x2 superblocks) 8-bit 4:2:0 still picture.  Every leaf is
// a 64x32 DC block except:
//
//   * row 1, superblock 6, first leaf (x=384, y=64) is a palette block with a
//     two-colour checkerboard ("source");
//   * row 1, superblock 11, first leaf (x=704, y=64) is an intra-block-copy
//     block with BV (-320, 0), which points exactly at the source leaf.
//
// The source is deliberately interior (not on any frame edge): dav2d
// f9dafbc enforces the §6.19.7.12 intra-block-copy reference-pixel validity
// including the ±1 interpolation border, so a block touching the frame edge is
// rejected at `--strict 1`.  This stream is accepted at strict 1.
//
// The source is palette-coded so the copied pixels are not degenerate; after
// decoding, the IBC region must be byte-identical to the source region.  That
// proves the BV / stack-index / copy semantics, not just that the syntax parses.
//
// The palette block is the same tool this repository's palette module gates
// separately; it is used here only to make the IBC copy observable.

#include "codec/intra/palette/core/palette.h"
#include "codec/intra/palette/core/scc_stream.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 768;
constexpr int kHeight = 128;
constexpr int kSbCols = kWidth / 64;
constexpr int kSbRows = kHeight / 64;

void fill_checker(std::vector<std::uint8_t>& y) {
  // Two-colour 8x8 checkerboard with a solid marker block, so a wrong BV is
  // immediately visible.
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      y[static_cast<std::size_t>(row) * kWidth +
          static_cast<std::size_t>(col)] = 128;
    }
  }
  for (int row = 64; row < 96; ++row) {
    for (int col = 384; col < 448; ++col) {
      const bool on = ((row / 8) + (col / 8)) & 1;
      y[static_cast<std::size_t>(row) * kWidth +
          static_cast<std::size_t>(col)] = on ? 230 : 16;
    }
  }
  for (int row = 72; row < 88; ++row) {
    for (int col = 408; col < 424; ++col) {
      y[static_cast<std::size_t>(row) * kWidth +
          static_cast<std::size_t>(col)] = 90;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string out = "ibc_gate.ivf";
  std::string ref_y4m;
  int ref_index = 1;
  bool pattern = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (arg == "--ref" && i + 1 < argc) {
      ref_y4m = argv[++i];
    } else if (arg == "--ref-index" && i + 1 < argc) {
      ref_index = std::atoi(argv[++i]);
    } else if (arg == "--plain") {
      pattern = false;
    } else {
      std::fprintf(stderr,
                   "usage: %s [--out FILE] [--ref FILE.y4m] "
                   "[--ref-index 0|1] [--plain]\n",
                   argv[0]);
      return 2;
    }
  }
  if (ref_index != 0 && ref_index != 1) {
    std::fprintf(stderr, "ref index must be 0 or 1\n");
    return 2;
  }

  std::vector<std::uint8_t> y(static_cast<std::size_t>(kWidth) * kHeight);
  fill_checker(y);

  // Palette source leaf: analyse at its frame position.  In --plain mode the
  // source is flat DC content, so the stream contains only IBC and DC intra.
  av2enc::palette::PaletteResult source_palette;
  std::string error;
  const int sx = 384;
  const int sy = 64;
  if (pattern) {
    if (!av2enc::palette::analyze_palette(
            y.data() + static_cast<std::size_t>(sy) * kWidth + sx, kWidth,
            /*block_w=*/64, /*block_h=*/32, /*visible_w=*/64,
            /*visible_h=*/32, nullptr, 0, av2enc::palette::PaletteParams{},
            source_palette, error)) {
      std::fprintf(stderr, "palette analysis failed: %s\n", error.c_str());
      return 1;
    }
    if (!source_palette.has_palette || source_palette.size < 2) {
      std::fprintf(stderr, "source palette was not usable\n");
      return 1;
    }
  }

  av2enc::scc::SccFramePlan plan;
  plan.config.width = kWidth;
  plan.config.height = kHeight;
  plan.config.allow_intrabc = true;
  plan.config.allow_global_intrabc = true;
  plan.config.allow_local_intrabc = false;

  for (int sb = 0; sb < kSbCols * kSbRows; ++sb) {
    const int sb_row = sb / kSbCols;
    const int sb_col = sb % kSbCols;
    av2enc::scc::SbPlan s;
    s.partition = av2enc::scc::SbPlan::Partition::kHorz;
    for (int half = 0; half < 2; ++half) {
      av2enc::scc::BlockPlan leaf;
      leaf.mi_row = sb_row * 16 + half * 8;
      leaf.mi_col = sb_col * 16;
      leaf.bsize = 11;  // BLOCK_64X32
      leaf.kind = av2enc::scc::BlockKind::kDc;
      if (pattern && sb_row == 1 && sb_col == 6 && half == 0) {
        leaf.kind = av2enc::scc::BlockKind::kPalette;
        leaf.palette = source_palette;
      }
      if (sb_row == 1 && sb_col == 11 && half == 0) {
        leaf.kind = av2enc::scc::BlockKind::kIbc;
        leaf.bv_row = 0;
        leaf.bv_col = -320 * 8;  // -320 luma samples
        leaf.ref_mv_idx = ref_index;
        leaf.skip = true;
      }
      s.leaves.push_back(leaf);
    }
    plan.sbs.push_back(s);
  }

  av2enc::scc::SccEncodeResult result;
  if (!av2enc::scc::encode_scc_frame(plan, result, error)) {
    std::fprintf(stderr, "encode failed: %s\n", error.c_str());
    return 1;
  }

  // Writer-side check: the IBC leaf must reproduce the source leaf exactly.
  int diff = 0;
  if (pattern) {
    for (int row = 0; row < 32; ++row) {
      const std::size_t src = static_cast<std::size_t>(sy + row) * kWidth + sx;
      const std::size_t dst = static_cast<std::size_t>(64 + row) * kWidth + 704;
      for (int col = 0; col < 64; ++col) {
        if (y[src + static_cast<std::size_t>(col)] !=
            result.recon_y[dst + static_cast<std::size_t>(col)]) {
          ++diff;
        }
      }
    }
  }
  std::printf("mode=%s source palette=%d colours, copy diff=%d, tile=%zu, "
              "frame=%zu bytes, ref=%d\n",
              pattern ? "pattern" : "plain",
              pattern ? source_palette.size : 0, diff, result.tile_bytes,
              result.frame.size(), ref_index);
  if (!av2enc::scc::write_scc_ivf(result, plan.config, out, error)) {
    std::fprintf(stderr, "ivf failed: %s\n", error.c_str());
    return 1;
  }
  if (!ref_y4m.empty() &&
      !av2enc::scc::write_scc_y4m(result, plan.config, ref_y4m, error)) {
    std::fprintf(stderr, "ref failed: %s\n", error.c_str());
    return 1;
  }
  return diff == 0 ? 0 : 1;
}
