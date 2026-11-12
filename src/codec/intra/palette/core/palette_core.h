// AV2 palette intra — shared normative core (host + device).
//
// Spec-only implementation of the AV2 v1.0.0 palette tool.  This header is
// included by the CPU reference (`src/encoder/palette/palette.cpp`) and by the
// CUDA kernels (`src/cuda/palette/palette.cu`), so both paths use exactly the
// same index/context arithmetic.
//
// Normative reference: AV2_Spec_v1.0.0.pdf v1.0.0 (2026-05-28, commit 38f28e7)
//
//   §5.20.8.1  palette mode info syntax (Y only)
//   §5.20.8.2  get_palette_cache()
//   §5.20.8.4  palette tokens syntax (direction, identity rows, indices)
//   §5.20.8.5  get_palette_color_context()
//   §6.19.8.1  palette semantics
//   §6.19.8.3  palette token semantics
//   §7.13.4    palette prediction process
//   §8.3.2     palette CDF selection
//
// Important scope fact established from the v1.0.0 syntax: the palette tool is
// **luma only**.  `palette_mode_info()` has no U/V palette size, colour or
// token syntax.  Chroma is predicted normally (`read_intra_uv_mode`, including
// CfL) from the reconstructed palette-predicted luma; see palette.h for the
// helper that documents this rule.
//
// Deliberately not here: entropy syntax (palette.cpp), bitstream assembly
// (scc_stream.*), colour quantisation heuristics (palette.cpp).

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define AV2_PALETTE_HD __host__ __device__
#else
#define AV2_PALETTE_HD
#endif

namespace av2enc::palette {

// §3 constants (spec/tables/constants.json).
inline constexpr int kPaletteColors = 8;
inline constexpr int kPaletteColorContexts = 5;
inline constexpr int kPaletteMaxColorContextHash = 8;
inline constexpr int kPaletteNumNeighbors = 3;
inline constexpr int kPaletteRowFlagContexts = 4;
inline constexpr int kPaletteSizes = 7;  // sizes 2..8
inline constexpr int kPaletteMaxSize = kPaletteSizes + 1;

// §9.3 `Palette_Color_Context[PALETTE_MAX_COLOR_CONTEXT_HASH + 1]`.
inline constexpr std::int8_t kPaletteColorContext[9] = {
    -1, -1, 0, -1, -1, 4, 3, 2, 1};

// §9.3 `Palette_Color_Hash_Multipliers[PALETTE_NUM_NEIGHBORS]` = {1, 2, 2}.

AV2_PALETTE_HD inline int palette_clip1(int value) {
  return value < 0 ? 0 : (value > 255 ? 255 : value);
}

// CeilLog2(v): smallest n with (1 << n) >= v; CeilLog2(0) is taken as 0 here
// (the spec only calls it with range >= 1 in the palette delta loop, and the
// `range == 0` case is unreachable for a strictly increasing palette).
AV2_PALETTE_HD inline int palette_ceil_log2(int value) {
  if (value <= 1) return 0;
  int bits = 0;
  int v = value - 1;
  while (v > 0) {
    ++bits;
    v >>= 1;
  }
  return bits;
}

AV2_PALETTE_HD inline int palette_floor_log2(int value) {
  int bits = -1;
  while (value > 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

// §5.20.8.4 `NS(PaletteSizeY)` value width in bits.  The non-symmetric code
// writes `w-1` bits first and possibly one extra bit (see §4.11.8).
AV2_PALETTE_HD inline int palette_ns_value_bits(int n) {
  return palette_floor_log2(n) + 1 - 1;
}

// §5.20.8.4 sample access with palette_direction applied.  direction 0 reads
// row by row (r = outer index, c = inner index); direction 1 reads column by
// column (c = outer index, r = inner index).
AV2_PALETTE_HD inline int palette_coord(int direction, int outer, int inner,
                                        int* r, int* c) {
  if (direction) {
    *r = inner;
    *c = outer;
  } else {
    *r = outer;
    *c = inner;
  }
  return 0;
}

AV2_PALETTE_HD inline int palette_map_at(const std::uint8_t* map, int stride,
                                         int r, int c) {
  return map[static_cast<std::size_t>(r) * static_cast<std::size_t>(stride) +
             static_cast<std::size_t>(c)];
}

// ---------------------------------------------------------------------------
// §5.20.8.5 get_palette_color_context
//
// Returns ColorContextHash and fills `color_order[0..n-1]` with the decoder's
// ColorOrder (the mapping from coded index to palette index).  The ranking is
// a selection sort with the spec's exact tie-break rule, so this function is
// the single source of truth for both the encoder (invert ColorOrder to get
// the symbol) and a local decoder test.
// ---------------------------------------------------------------------------
AV2_PALETTE_HD inline int palette_color_context(const std::uint8_t* color_map,
                                                int stride, int r, int c,
                                                int n,
                                                std::uint8_t color_order[8]) {
  int scores[kPaletteColors];
  for (int i = 0; i < kPaletteColors; ++i) {
    scores[i] = 0;
    color_order[i] = static_cast<std::uint8_t>(i);
  }
  if (c > 0) {
    const int neighbor = palette_map_at(color_map, stride, r, c - 1);
    scores[neighbor] += 2;
  }
  if (r > 0 && c > 0) {
    const int neighbor = palette_map_at(color_map, stride, r - 1, c - 1);
    scores[neighbor] += 1;
  }
  if (r > 0) {
    const int neighbor = palette_map_at(color_map, stride, r - 1, c);
    scores[neighbor] += 2;
  }
  for (int i = 0; i < kPaletteNumNeighbors; ++i) {
    int max_score = scores[i];
    int max_idx = i;
    for (int j = i + 1; j < n; ++j) {
      if (scores[j] > max_score) {
        max_score = scores[j];
        max_idx = j;
      } else if (scores[j] > 0 && scores[j] == max_score && c > 0 &&
                 j == palette_map_at(color_map, stride, r, c - 1)) {
        max_score = scores[j];
        max_idx = j;
      }
    }
    if (max_idx != i) {
      max_score = scores[max_idx];
      const std::uint8_t max_color_order = color_order[max_idx];
      for (int k = max_idx; k > i; --k) {
        scores[k] = scores[k - 1];
        color_order[k] = color_order[k - 1];
      }
      scores[i] = max_score;
      color_order[i] = max_color_order;
    }
  }
  static_assert(kPaletteNumNeighbors == 3, "palette hash assumes 3 neighbours");
  const int hash = scores[0] + 2 * scores[1] + 2 * scores[2];
  return hash;
}

// §8.3.2 palette_color_idx_y CDF context: Palette_Color_Context[hash].
AV2_PALETTE_HD inline int palette_color_context_index(int hash) {
  if (hash < 0 || hash > kPaletteMaxColorContextHash) return -1;
  return kPaletteColorContext[hash];
}

// Inverse of ColorOrder: the coded index whose palette entry is `color`.
// Returns -1 if the colour is not in the order (impossible for n correct).
AV2_PALETTE_HD inline int palette_symbol_for_color(
    int color, const std::uint8_t color_order[8], int n) {
  for (int i = 0; i < n; ++i) {
    if (color_order[i] == color) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// §7.13.4 palette prediction
//
// Prediction is defined per transform block at 4x4 granularity:
//   CurrFrame[0][startY+i][startX+j] = colors[ColorMapY[y*4+i][x*4+j]]
// With TX_MODE_LARGEST there is one transform per coding block and x=y=0, so
// the map is read at full sample resolution.  The general form is used here so
// a future TX_MODE_SELECT path is correct.
// ---------------------------------------------------------------------------
AV2_PALETTE_HD inline void palette_predict_tx(const std::uint8_t* colors,
                                              const std::uint8_t* color_map,
                                              int cmap_stride, int x4, int y4,
                                              std::uint8_t* dst, int dst_stride,
                                              int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int idx = palette_map_at(color_map, cmap_stride, y4 * 4 + i,
                                     x4 * 4 + j);
      dst[static_cast<std::size_t>(i) * static_cast<std::size_t>(dst_stride) +
          static_cast<std::size_t>(j)] = colors[idx];
    }
  }
}

// ---------------------------------------------------------------------------
// Identity-row classification (§5.20.8.4/§6.19.8.3).
//
// Returns 2 when the whole line equals the previous line (only legal for
// outer index > 0), 1 when the line is constant but differs from the previous
// line, otherwise 0.  `line_length` is onscreenWidth or onscreenHeight.
// ---------------------------------------------------------------------------
AV2_PALETTE_HD inline int palette_classify_line(const std::uint8_t* color_map,
                                                int stride, int direction,
                                                int outer, int inner_count,
                                                bool have_previous,
                                                int n) {
  (void)n;
  if (inner_count <= 0) return 0;
  int r = 0;
  int c = 0;
  palette_coord(direction, outer, 0, &r, &c);
  const int first = palette_map_at(color_map, stride, r, c);
  bool constant = true;
  for (int j = 1; j < inner_count; ++j) {
    palette_coord(direction, outer, j, &r, &c);
    if (palette_map_at(color_map, stride, r, c) != first) {
      constant = false;
      break;
    }
  }
  if (constant) {
    if (have_previous) {
      bool same_as_previous = true;
      for (int j = 0; j < inner_count; ++j) {
        int pr = 0;
        int pc = 0;
        int cr = 0;
        int cc = 0;
        palette_coord(direction, outer - 1, j, &pr, &pc);
        palette_coord(direction, outer, j, &cr, &cc);
        if (palette_map_at(color_map, stride, pr, pc) !=
            palette_map_at(color_map, stride, cr, cc)) {
          same_as_previous = false;
          break;
        }
      }
      if (same_as_previous) return 2;
    }
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Encoder-side index selection.  `map` must already have been written with a
// legal index; this is the nearest-colour search used to build it.
// ---------------------------------------------------------------------------
AV2_PALETTE_HD inline int palette_nearest_color(const std::uint8_t* colors,
                                                int n, int value) {
  int best = 0;
  int best_dist = 1 << 30;
  for (int i = 0; i < n; ++i) {
    const int d = colors[i] - value;
    const int dist = d < 0 ? -d : d;
    if (dist < best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  return best;
}

}  // namespace av2enc::palette
