// Shared AV2 v1.0.0 spec-math primitives (single source of truth).
//
// This header holds the small, normative integer constants and arithmetic that
// several leaf modules used to re-declare.  It is deliberately dependency-free
// (no STL, no I/O, no CUDA runtime) so the same header can be included from
// the host entropy bank, the CUDA-portable symbol coder, and tests.
//
// Scope today: §8.2 CDF scale / alphabet constants and the §8.2.4
// `avg_cdf()` update.  Module-specific validation stays in the owning module;
// the helpers below assume validated inputs.
#pragma once

#include <cstdint>

namespace av2enc::spec {

// §8.2: inverse-CDF probability scale (1 << 15).
inline constexpr std::uint32_t kCdfScale = 1u << 15;

// §9.2 Prob_Inc rows have N-2 <= 6, so the largest §8.2 alphabet is N = 8.
inline constexpr int kMaxCdfSymbols = 8;

// §8.2.6: the adaptation use-count cdf[N] saturates at 32.
inline constexpr std::uint32_t kCdfCountCap = 32;

// §8.2.4: numLog2 = Min(3, FloorLog2(TileCols * TileRows)).
inline constexpr std::uint32_t kMaxAverageLog2 = 3;

// Exact §8.2.4 avg_cdf() update for one inverse-CDF array.
//
//   cdf   : destination, N+1 entries (N-1 boundaries, rate, use count)
//   tile  : Tile* array for the current tile, N+1 entries
//   symbols: N, the symbol count
//   num_log2: Min(3, FloorLog2(tile_count))
//   first_tile: true for TileNum == 0 (initialise the accumulator)
//
// Preconditions (checked by the callers): non-null pointers,
// 2 <= symbols <= kMaxCdfSymbols, num_log2 <= kMaxAverageLog2.
inline void average_cdf_step(std::uint16_t* cdf,
                             const std::uint16_t* tile_cdf, int symbols,
                             std::uint32_t num_log2,
                             bool first_tile) noexcept {
  if (first_tile) {
    // cdf[i] = 1 << 15 for i < N-1; the rate comes from tile 0; count = 0.
    for (int i = 0; i < symbols - 1; ++i) {
      cdf[i] = static_cast<std::uint16_t>(kCdfScale);
    }
    cdf[symbols - 1] = tile_cdf[symbols - 1];
    cdf[symbols] = 0;
  }
  for (int i = 0; i < symbols - 1; ++i) {
    const std::uint32_t value = cdf[i];
    const std::uint32_t tile_value = tile_cdf[i];
    cdf[i] = static_cast<std::uint16_t>(
        value - ((kCdfScale - tile_value) >> num_log2));
  }
  cdf[symbols] = static_cast<std::uint16_t>(
      cdf[symbols] +
      static_cast<std::uint16_t>(tile_cdf[symbols] >> num_log2));
}

}  // namespace av2enc::spec
