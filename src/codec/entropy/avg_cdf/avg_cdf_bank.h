// AV2 v1.0.0 average-CDF bank (§5.4.8, §5.18.7.2, §7.4.4, §7.5, §8.2.2,
// §8.2.4).
//
// This module models the decoder's CDF bookkeeping for a tile group.  It is
// deliberately independent of the M1/M2 syntax writers: a tile writer owns the
// actual entropy coder, calls `load_tile_cdfs()` at `init_symbol()`, leaves the
// encoded/decoded `tile` array in the bank, calls `save_tile_cdfs()` at
// `exit_symbol()`, and finally calls `average_at_frame_end()` after the last
// tile of the frame.
//
// Normative recap
// ---------------
// §8.2.2: at the start of every tile a fresh copy of each frame CDF array is
// made into its `Tile*` array.  Only the tile copy is mutated by read_symbol.
//
// §8.2.4 `exit_symbol()`:
//   numLog2 = Min(3, FloorLog2(TileCols * TileRows))
//   if (enable_avg_cdf && avg_cdf_type)
//       avgCdf = TileNum < (1 << numLog2)
//   else
//       copyCdf = (TileNum == context_update_tile_id)
//
// For `avgCdf` the destination is `Saved*` and the source is `Tile*`.  Tile 0
// initialises the accumulator to the full scale, then every averaged tile
// subtracts `(32768 - tilecdf[i]) >> numLog2` and adds
// `tilecdf[count] >> numLog2`.
//
// §7.5 `FrameEndUpdateCdf()` (called after `tg_end == NumTiles-1`) installs
// every `Saved*` array into the frame array and scales the use count:
//   frame[i] = saved[i]                       for i = 0..N-1  (N+1 entries)
//   frame[N] = (3 * saved[N]) >> 2
//
// §5.4.8: with `single_picture_header_flag = 1`, `enable_avg_cdf = 1` and
// `avg_cdf_type = 1` are inferred.  With `enable_avg_cdf && avg_cdf_type`
// §5.18.7.2 does not code `context_update_tile_id`.
//
// Reset points
// ------------
// CLK (`OBU_CLOSED_LOOP_KEY`) and OLK (`OBU_OPEN_LOOP_KEY`) are KEY_FRAMEs, so
// `FrameIsIntra = 1` and `primary_ref_frame = PRIMARY_REF_NONE`
// (§5.18.2/§6.17.2).  `set_primary_ref_frame_and_ctx(1)` then invokes
// `init_non_coeff_cdfs()` and the frame-header path invokes `init_coeff_cdfs()`.
// Every other frame with `primary_ref_frame == PRIMARY_REF_NONE` or
// `disable_cross_frame_cdf_init = 1` resets the same way.  Pass
// `begin_frame(true)` for those frames.  A frame that loads a reference frame
// context calls `begin_frame(false)` after `load_reference(ctx)`.
//
// Storage
// -------
// Each CDF array has N+1 `uint16_t` entries:
//   [0 .. N-2] inverse-CDF values
//   [N-1]      adaptation-rate index (copied from tile 0 by avg_cdf)
//   [N]        use count (summed across averaged tiles)
//
// The bank stores four copies per array: `frame`, `tile`, `saved`, and a frozen
// copy of `defaults`; optionally it also stores NUM_REF_FRAMES reference
// contexts (`save_cdfs`/`load_cdfs`).  It owns no entropy-coder state and no
// per-symbol hot path.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "spec_tables/spec_math.h"

namespace av2enc::entropy::avg_cdf {

// §8.2 inverse-CDF probability scale and the maximum CDF alphabet (§9.3).
// Shared with the symbol coder via src/spec_tables/spec_math.h.
inline constexpr std::uint32_t kCdfScale = spec::kCdfScale;
inline constexpr std::uint32_t kMaxAverageLog2 = spec::kMaxAverageLog2;
inline constexpr std::uint16_t kMaxSymbols =
    static_cast<std::uint16_t>(spec::kMaxCdfSymbols);

// One CDF array to allocate in an owning bank.
struct CdfSpec {
  std::uint16_t symbols = 0;      // N; the array has N+1 entries
  const std::uint16_t* defaults = nullptr;  // N+1 default entries (§9.3)
};

// Non-owning view of one CDF array's storage.  All pointers except `defaults`
// are mutable and must remain valid for the lifetime of the bank.
struct CdfTable {
  std::uint16_t symbols = 0;      // N
  std::uint16_t* frame = nullptr; // current frame CDF state
  std::uint16_t* tile = nullptr;  // per-tile working copy
  std::uint16_t* saved = nullptr; // frame-end accumulator
  const std::uint16_t* defaults = nullptr;
  // Optional reference frame contexts (§7.4.4).  `reference_stride` is N+1.
  std::uint16_t* reference_base = nullptr;
  std::uint32_t reference_stride = 0;
};

struct CdfBankConfig {
  bool enable_avg_cdf = true;   // §5.4.8 sequence header
  bool avg_cdf_type = true;     // 1 = average across tiles (§5.4.8)
  // §5.18.7.2.  Only used when !enable_avg_cdf || !avg_cdf_type.
  std::uint32_t context_update_tile_id = 0;
  // Number of `save_cdfs(ctx)` slots to allocate.  0 disables the API.
  std::uint32_t reference_contexts = 0;
};

// §8.2.4: Min(3, FloorLog2(tile_count)).  tile_count must be >= 1.
std::uint32_t average_log2(std::uint32_t tile_count) noexcept;

// 1 << average_log2(tile_count); tiles at or above this index are not averaged.
std::uint32_t average_tile_limit(std::uint32_t tile_count) noexcept;

// Exact §8.2.4 avg_cdf() arithmetic for one CDF array of N+1 entries.
// `first_tile` selects the TileNum == 0 initialisation.
void average_cdf(std::uint16_t* cdf, const std::uint16_t* tile_cdf,
                 std::uint16_t symbols, std::uint32_t num_log2,
                 bool first_tile) noexcept;

// Exact §7.4.4 blend_cdf() arithmetic for one CDF array of N+1 entries.
void blend_cdf(std::uint16_t* cdf, const std::uint16_t* saved_cdf,
               std::uint16_t symbols) noexcept;

class CdfBank {
 public:
  CdfBank() = default;
  CdfBank(const CdfBank&) = delete;
  CdfBank& operator=(const CdfBank&) = delete;
  CdfBank(CdfBank&&) = delete;
  CdfBank& operator=(CdfBank&&) = delete;

  // Non-owning initialisation.  Every table must have non-null frame/tile/
  // saved/defaults pointers and 1 <= symbols <= 8.  When
  // config.reference_contexts != 0 every table must also provide
  // reference_base with reference_stride == symbols + 1.
  bool init(std::span<const CdfTable> tables, const CdfBankConfig& config,
            std::string* error = nullptr);

  // Owning initialisation: allocates frame/tile/saved/defaults (and reference
  // contexts when requested) and copies the default arrays.
  bool init_owned(std::span<const CdfSpec> specs, const CdfBankConfig& config,
                  std::string* error = nullptr);

  const CdfBankConfig& config() const noexcept { return config_; }
  std::size_t table_count() const noexcept { return tables_.size(); }
  const CdfTable& table(std::size_t index) const { return tables_[index]; }
  CdfTable& table(std::size_t index) { return tables_[index]; }

  std::uint16_t* frame_cdf(std::size_t index) noexcept {
    return tables_[index].frame;
  }
  std::uint16_t* tile_cdf(std::size_t index) noexcept {
    return tables_[index].tile;
  }
  std::uint16_t* saved_cdf(std::size_t index) noexcept {
    return tables_[index].saved;
  }

  // Reset to the §9.3 defaults.  This is the CLK/OLK (and
  // primary_ref_frame == PRIMARY_REF_NONE / disable_cross_frame_cdf_init)
  // state: frame, tile and saved arrays all become `defaults`.
  void reset() noexcept;

  // Frame boundary.  `reset_to_defaults` is true for CLK/OLK and for any frame
  // whose primary reference is PRIMARY_REF_NONE or whose
  // disable_cross_frame_cdf_init bit is 1.  False keeps the current frame CDFs
  // (the caller has loaded a reference frame context, or is continuing after
  // frame_end_update_cdf()).
  void begin_frame(bool reset_to_defaults) noexcept;

  // `init_symbol()`: frame -> tile for every array.
  void load_tile_cdfs() noexcept;

  // `exit_symbol()`: copy or average this tile's final CDFs into `saved`.
  // Returns false for a tile index outside [0, tile_count).
  bool save_tile_cdfs(std::uint32_t tile_num,
                      std::uint32_t tile_count) noexcept;

  // `frame_end_update_cdf()` (§7.5): saved -> frame, count = (3*count)>>2.
  void average_at_frame_end() noexcept;

  // §7.4.4 `save_cdfs(ctx)` / `load_cdfs(ctx)`.  `load_reference` copies the
  // context exactly (no 3/4 count scaling; that is only at frame end).
  bool save_reference(std::uint32_t ctx) noexcept;
  bool load_reference(std::uint32_t ctx) noexcept;

  // §7.4.4 `blend_cdfs(ctx)` used only when enable_avg_cdf = 1 and
  // avg_cdf_type = 0: 7/8 current + 1/8 saved context, with the exact integer
  // rounding from the specification.
  bool blend_reference(std::uint32_t ctx) noexcept;

 private:
  bool validate(std::string* error) const;

  CdfBankConfig config_{};
  std::vector<CdfTable> tables_;
  // One allocation for all owning tables; pointers in `tables_` point into it.
  std::vector<std::uint16_t> storage_;
  bool owning_ = false;
};

}  // namespace av2enc::entropy::avg_cdf
