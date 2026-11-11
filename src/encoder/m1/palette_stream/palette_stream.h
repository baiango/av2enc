// AV2 palette intra — M1 syntax adapter.
//
// This module is the integration layer between the screen-content palette core
// (src/encoder/palette/) and an M1 block stream.  It owns *only* the palette
// syntax position and state; it does not own partition, intra mode selection,
// transform/quant, or the OBU writer.
//
// Spec references (AV2 v1.0.0):
//   §5.4.7   sequence screen content config (`seq_force_*`)
//   §5.18.3.3 screen_content_params() frame bits
//   §5.20.8.1 palette_mode_info()  (Y only, after read_intra_uv_mode)
//   §5.20.8.2 get_palette_cache()
//   §5.20.8.4 palette_tokens()     (immediately after mode info)
//   §6.19.8.1/.3 palette semantics (identity rows 0/1/2, sorted colours)
//   §8.3.2    palette CDF selection (Table 8.1 / Palette_Color_Context)
//
// Syntax position (the one rule a caller must get right)
// -----------------------------------------------------
//   read_intra_y_mode()            // must be DC_PRED for a palette block
//   read_intra_uv_mode()           // normal chroma mode (CfL allowed)
//   read_cfl_alphas()
//   palette_mode_info()            // has_palette_y + size + colours
//   palette_tokens()               // direction + identity rows + indices
//   read_block_tx_size()/residual  // normal 3-plane residual; eob=0 for
//                                  // palette-only (transform-skip) blocks
//
// `palette_mode_info()` is conditioned on `PlaneStart == 0 && YMode ==
// DC_PRED`.  If the luma mode is not DC there is no symbol at all, even when
// screen-content tools are enabled.  Use `write()`/`read()` after the UV mode
// and pass `y_mode`; the adapter reproduces the condition exactly.
//
// Luma-only rule
// --------------
// AV2 v1.0.0 has no chroma palette.  `PaletteModeInfo` and all helpers here are
// luma-only; chroma is reconstructed with the normal intra path after the
// palette-predicted luma.  There is deliberately no U/V palette API.
//
// The CPU palette core remains the normative implementation.  This file adds
// state/ordering/validation and reuses `palette::encode_palette_y()` /
// `decode_palette_y()` byte-for-byte, so the CDF state and colour-map syntax
// cannot drift from the gated component.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/palette/palette.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::m1::palette_stream {

namespace pal = av2enc::palette;

// BLOCK_8X8 in partition::BlockSize.  §5.20.5.5 conditions palette_mode_info on
// `MiSize >= BLOCK_8X8` (numeric enum order), not on a minimum sample width.
inline constexpr int kBlock8x8 = 3;
inline constexpr int kMaxPaletteBlockDim = 64;

// ---------------------------------------------------------------------------
// Frame / sequence screen-content flags
// ---------------------------------------------------------------------------

struct ScreenContentFlags {
  // M1 sequence headers set single_picture_header_flag=1.  §5.4.7 then infers
  // seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS and
  // seq_force_integer_mv = SELECT_INTEGER_MV, so the sequence header codes no
  // bits for this config and the frame header codes both allow_* / force_*.
  bool single_picture_header = true;
  bool allow_screen_content_tools = false;
  // This adapter's palette switch.  Requires allow_screen_content_tools.
  bool allow_palette = false;
  // Only meaningful when allow_screen_content_tools is set.
  bool force_integer_mv = false;
  // The palette stream does not require IBC; defaults keep it off.
  bool allow_intrabc = false;
  bool allow_global_intrabc = false;
  bool allow_local_intrabc = true;
};

// Rejects inconsistent combinations.  `allow_palette` without
// `allow_screen_content_tools` is an error, as is local/global IBC without
// `allow_intrabc`, and a non-single-picture sequence needs the explicit
// seq_force_* values (not modelled yet).
bool validate_screen_content_flags(const ScreenContentFlags& flags,
                                   std::string& error);

// Convenience: screen-content tools on, palette on, IBC off, integer MV forced.
ScreenContentFlags palette_screen_content_flags();

// §5.4.7 sequence_scc_config().  With single_picture_header_flag=1 this writes
// zero bits (the values are inferred); otherwise it writes the two/three bits
// shown in the spec.  Exposed so a future header writer can share the exact
// rule instead of hand-rolling it.
std::size_t write_sequence_screen_content_params(BitWriter& writer,
                                                 const ScreenContentFlags& flags);

// §5.18.3.3 screen_content_params() for an intra frame:
//   allow_screen_content_tools f(1)
//   force_integer_mv          f(1) when allow_screen_content_tools
// Returns the number of bits written.
std::size_t write_frame_screen_content_params(BitWriter& writer,
                                              const ScreenContentFlags& flags);

// §5.20.5.5 gating rule (MiSize / block dimensions / allow_screen_content_tools
// plus this adapter's allow_palette switch).  A 4x4 block (MiSize
// BLOCK_4X4) is *not* palette-signallable.
bool palette_allowed_for_block(const ScreenContentFlags& flags, int bsize,
                               int block_w, int block_h) noexcept;

// ---------------------------------------------------------------------------
// palette_mode_info metadata + §8.3.2 CDF accessors
// ---------------------------------------------------------------------------

struct PaletteModeInfo {
  bool has_palette_y = false;
  int size = 0;                 // 2..8 when has_palette_y
  bool direction_signalled = false;
  int direction = 0;            // 0 = rows, 1 = columns
  // One entry per outer line of the onscreen area, exactly as
  // §6.19.8.3 classifies it (0 = not constant, 1 = constant, 2 = same as the
  // previous line).
  std::vector<std::uint8_t> identity_rows;
  int cache_len = 0;            // cacheN from get_palette_cache()
  int cache_used = 0;           // number of reused cache colours
  int onscreen_width = 0;
  int onscreen_height = 0;
};

// §8.3.2 default/adaptive CDF pointers.  They return nullptr for an invalid
// size/context, so callers can fail loudly rather than index out of bounds.
const std::uint16_t* y_mode_cdf(const pal::PaletteCdfState& cdfs) noexcept;
const std::uint16_t* y_size_cdf(const pal::PaletteCdfState& cdfs) noexcept;
const std::uint16_t* color_cdf(const pal::PaletteCdfState& cdfs, int size,
                               int context_index) noexcept;
const std::uint16_t* identity_cdf(const pal::PaletteCdfState& cdfs,
                                  int prev_identity) noexcept;

// PaletteSizeY -> palette_size_y_minus_2 symbol (0..6).
int palette_size_symbol(int size) noexcept;

// Colour-order sample at (r,c) of a palette block: the decoder's ColorOrder and
// context hash, and the coded symbol for the actual colour map entry.  This is
// the per-sample "colour order" state §5.20.8.5/§7.13.4 uses; it is derived,
// not stored, so the encoder and decoder compute it identically.
struct ColorOrderSample {
  int r = 0;
  int c = 0;
  int hash = 0;
  int context_index = -1;
  int symbol = -1;
  std::array<std::uint8_t, pal::kPaletteMaxSize> order{};
};

bool color_order_at(const pal::PaletteResult& result, int block_w, int r, int c,
                    ColorOrderSample& out) noexcept;

// ---------------------------------------------------------------------------
// Palette writer / reader (per-frame per-MI state)
// ---------------------------------------------------------------------------

// Writer side.  Owns the per-MI PaletteState (sizes/colours for
// get_palette_cache) and applies exactly the decoder's ordering.  One instance
// per tile/frame; reset() before the first block.
class PaletteSyntaxWriter {
 public:
  explicit PaletteSyntaxWriter(const ScreenContentFlags& flags = {});

  bool reset(int mi_cols, int mi_rows, std::string& error);

  // Writes palette_mode_info() then palette_tokens() for one coding block.
  //
  //   mi_row/mi_col   block origin in 4x4 MI units
  //   bsize           partition::BlockSize numeric value
  //   block_w/height  coded block dimensions (4..64)
  //   visible_w/height onscreen area (partial edge blocks)
  //   result          palette analysis; nullptr or !has_palette writes
  //                   has_palette_y=0 (when the syntax position is reachable)
  //   y_mode          AV2 YMode already coded for this block; palette syntax is
  //                   only present when y_mode == DC_PRED (0)
  //
  // Returns false only on a plan/syntax error.  On success `*info` (optional)
  // describes the symbols that were actually written.  The caller must code the
  // normal 3-plane residual next (eob=0 for a palette-only block).
  bool write(av2::Encoder& enc, pal::PaletteCdfState& cdfs, int mi_row,
             int mi_col, int bsize, int block_w, int block_h, int visible_w,
             int visible_h, const pal::PaletteResult* result, int y_mode,
             PaletteModeInfo* info, std::string& error);

  // Non-mutating cache lookup for this block (same as the decoder's
  // get_palette_cache call in palette_mode_info()).
  int cache(int mi_row, int mi_col, bool avail_l,
            std::uint8_t out[16]) const noexcept;

  const pal::PaletteState& state() const noexcept { return state_; }
  pal::PaletteState& mutable_state() noexcept { return state_; }
  const ScreenContentFlags& flags() const noexcept { return flags_; }

 private:
  ScreenContentFlags flags_;
  pal::PaletteState state_;
};

// Reader side; mirrors PaletteSyntaxWriter and rebuilds the decoded map/state.
class PaletteSyntaxReader {
 public:
  explicit PaletteSyntaxReader(const ScreenContentFlags& flags = {});

  bool reset(int mi_cols, int mi_rows, std::string& error);

  bool read(av2::Decoder& dec, pal::PaletteCdfState& cdfs, int mi_row,
            int mi_col, int bsize, int block_w, int block_h, int visible_w,
            int visible_h, int y_mode, pal::PaletteResult& out,
            PaletteModeInfo* info, std::string& error);

  int cache(int mi_row, int mi_col, bool avail_l,
            std::uint8_t out[16]) const noexcept;

  const pal::PaletteState& state() const noexcept { return state_; }
  const ScreenContentFlags& flags() const noexcept { return flags_; }

 private:
  ScreenContentFlags flags_;
  pal::PaletteState state_;
};

// ---------------------------------------------------------------------------
// Minimal M1-compatible palette stream (gate path)
// ---------------------------------------------------------------------------
//
// The production M1 writer (src/encoder/m1/m1_syntax.cpp + m1_headers.cpp) is
// not modified by this module.  Until that integration lands, this small
// wrapper emits a legal AV2 still using the existing M1 sequence header and the
// gated SCC tile writer: PARTITION_NONE / HORZ / VERT inside 64x64 superblocks,
// luma palette on every leaf, chroma DC, and EOB=0 on all three planes
// (transform-skip).  It is intentionally narrow and exists so the palette
// syntax can be validated end-to-end with dav2d without core edits.
//
// Production call sites for the full M1 path are listed in README.md.

struct MinimalPaletteLeaf {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 0;  // partition::BlockSize numeric value
  pal::PaletteResult palette;
};

struct MinimalPaletteSuperblock {
  enum class Partition { kNone, kHorz, kVert };
  Partition partition = Partition::kNone;
  std::vector<MinimalPaletteLeaf> leaves;
};

struct MinimalPaletteFramePlan {
  ScreenContentFlags flags;
  int width = 64;
  int height = 64;
  std::uint8_t base_q_idx = 90;
  bool enable_cfl_intra = false;
  bool enable_intra_edge_filter = false;
  bool apply_deblock_y_vert = false;
  bool apply_deblock_y_horz = false;
  bool apply_deblock_u = false;
  bool apply_deblock_v = false;
  std::vector<MinimalPaletteSuperblock> sbs;  // raster superblock order
};

struct MinimalPaletteFrameResult {
  std::vector<std::uint8_t> frame;  // TD + sequence header + CLK
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::size_t tile_bytes = 0;
  // Every leaf codes eob=0 on Y/U/V, i.e. transform-skip.
  bool transform_skip_eob0 = true;
};

bool encode_minimal_palette_frame(const MinimalPaletteFramePlan& plan,
                                  MinimalPaletteFrameResult& out,
                                  std::string& error);

bool write_minimal_palette_ivf(const MinimalPaletteFrameResult& result,
                               const MinimalPaletteFramePlan& plan,
                               const std::string& path, std::string& error);

bool write_minimal_palette_y4m(const MinimalPaletteFrameResult& result,
                               const MinimalPaletteFramePlan& plan,
                               const std::string& path, std::string& error);

}  // namespace av2enc::m1::palette_stream
