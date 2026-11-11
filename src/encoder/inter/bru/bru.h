// Backwards reference update (BRU) syntax, §5.4.6 / §5.18.2 / §5.19 /
// §5.20.2.1 / §7.2 / §7.17.2 of the AV2 v1.0.0 specification.
//
// The rest of the tree only consumes a caller-provided BruModes map in the
// loop-filter components.  This module owns the missing producer and an
// independent consumer:
//
//   * sequence `enable_bru`                      (§5.4.6, p63)
//   * frame `use_bru`, `bru_ref`, `bru_inactive` (§5.18.2, p102)
//   * tile-group `bru_tile_active`               (§5.19, p145)
//   * tile-level `bru_mode S()`                  (§5.20.2.1, p147)
//   * `bru_region_valid()` / R10                 (§7.2, p377-378)
//   * deblock gating when BruModes != BRU_ACTIVE (§7.17.2, p531)
//
// The writer emits BRU frames whose active superblocks are coded as
// GLOBALMV/skip inter blocks (zero MV, no residual) and whose inactive or
// support superblocks use `bru_mode_info()` (skip, zero MV, RefFrame[0] ==
// bru_ref).  This keeps the gate payload small while exercising every BRU
// branch.  The module deliberately does not change any M2 core file.
#pragma once

#include "bitstream/bitwriter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::inter::bru {

// Table 6.21 (p351).
enum class BruMode : std::uint8_t {
  kInactive = 0,
  kSupport = 1,
  kActive = 2,
};

inline constexpr int kBruModeCount = 3;

const char* bru_mode_name(BruMode mode);

// §5.4.6 sequence inter config.  The remaining inter tools stay disabled so
// the BRU-enabled stream is bit-compatible with the M2 baseline when
// `enable_bru` is false (verified by the byte-identity control test).
struct SequenceConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t base_q_idx = 90;
  std::uint8_t order_hint_bits = 4;
  std::uint8_t num_ref_frames = 2;
  std::uint8_t max_drl_bits_minus_1 = 4;
  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;
  bool enable_bru = false;
};

// §5.18.2 frame header subset needed by the BRU writer/reader.
struct FrameConfig {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  bool immediate_output_frame = true;
  // §5.18.2 explicit ref map.  The BRU gate uses two references so an active
  // block can reference a slot other than bru_ref (required by p362).
  std::uint8_t num_total_refs = 1;
  std::array<std::uint8_t, 7> ref_frame_idx{};
  std::uint32_t refresh_frame_flags = 1;
  // §5.18.2 BRU fields.  `use_bru` is only coded when enable_bru == 1,
  // FrameType == INTER_FRAME and the frame is not TIP/bridge.
  bool use_bru = false;
  std::uint8_t bru_ref = 0;
  bool bru_inactive = false;

  // §5.18.7.2 tile_info() for uniform spacing.  `tile_cols_log2` and
  // `tile_rows_log2` are the absolute log2 tile counts, not the increment
  // syntax elements.
  std::uint8_t tile_cols_log2 = 0;
  std::uint8_t tile_rows_log2 = 0;
  std::uint8_t tile_size_bytes_minus_1 = 0;
  std::uint8_t context_update_tile_id = 0;
};

struct TilePlan {
  // Per-tile §5.19 bru_tile_active bits, row-major over the tile grid.  Only
  // used when the frame has use_bru == 1 and NumTiles > 1.
  std::vector<std::uint8_t> bru_tile_active;
  // Full-frame superblock map, row-major, one entry per 64x64 superblock.
  std::vector<BruMode> sb_modes;
};

// Number of 4x4 MI units for a frame edge, and the 64x64 superblock grid.
int mi_cols(int frame_width);
int mi_rows(int frame_height);
int sb_cols_for(int frame_width);
int sb_rows_for(int frame_height);

// §5.18.2 / §5.18.7.7.  Uniform-spacing tile starts in superblock units.
struct UniformTileLayout {
  int sb_cols = 1;
  int sb_rows = 1;
  int tile_cols = 1;
  int tile_rows = 1;
  std::vector<int> sb_col_starts;
  std::vector<int> sb_row_starts;
};

bool compute_uniform_tile_layout(const SequenceConfig& sequence,
                                 const FrameConfig& frame,
                                 UniformTileLayout& out, std::string& error);

// The full sequence_header_obu() for non-single-picture streams.  With
// `enable_bru == false` this is byte-identical to
// m2::make_sequence_header_obu() for the same width/height/order-hint config.
std::vector<std::uint8_t> make_sequence_header_obu(
    const SequenceConfig& config);

// frame_header_info() syntax bits, without the leading is_first_tile_group
// bit and without byte_alignment().
std::size_t write_frame_header(BitWriter& writer, const SequenceConfig& config,
                               const FrameConfig& frame);

// tile_group_obu() header.  When frame.bru_inactive is true this emits the
// §5.19 early-return form (header plus trailing_bits only) and appends no
// tile payloads.
std::size_t write_tile_group_header(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameConfig& frame,
                                    const TilePlan& plan);

// Encodes one tile payload for the BRU gate subset.  `tile_modes` is row-major
// over the tile's superblocks.  When `tile_active` is false no symbols are
// written at all: decode_tile() infers BRU_INACTIVE for every superblock.
bool encode_tile_payload(std::span<const BruMode> tile_modes,
                         int tile_sb_cols, int tile_sb_rows,
                         std::uint8_t bru_ref, bool tile_active,
                         std::vector<std::uint8_t>& out, std::string& error);

// Complete OBU_REGULAR_TILE_GROUP / OBU_CLOSED_LOOP_KEY payload.  For a normal
// tile group `tile_payloads` must have one entry per tile in tg_start..tg_end
// order.  For bru_inactive it must be empty.
std::vector<std::uint8_t> make_tile_group_obu(
    const SequenceConfig& config, const FrameConfig& frame,
    const TilePlan& plan,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    std::string& error);

// §7.2 p378.  `modes` is the per-superblock BruModes map (one entry per
// superblock, not per MI unit).
bool bru_region_valid(int sb_cols, int sb_rows,
                      std::span<const BruMode> modes);

// §7.17.2 p531.  An edge in the superblock containing (mi_row, mi_col) is
// filtered only when use_bru == 0 or the superblock is BRU_ACTIVE.
bool deblock_edge_allowed(bool use_bru, int sb_shift, int mi_row, int mi_col,
                          int sb_cols, std::span<const BruMode> modes);

// §5.20.2.1 p169 / update_ibc_buffers(): an IBC buffer cached at a superblock
// is invalidated when that superblock is BRU_INACTIVE.
bool bru_ibc_buffer_valid(int sb_shift, int mi_row, int mi_col, int sb_cols,
                          std::span<const BruMode> modes);

// --- Independent reader -------------------------------------------------

struct ParsedFrame {
  FrameConfig header;
  int sb_cols = 0;
  int sb_rows = 0;
  int tile_cols = 1;
  int tile_rows = 1;
  std::vector<std::uint8_t> bru_tile_active;
  std::vector<BruMode> sb_modes;
  bool modes_reconstructed = false;
  bool whole_frame_inactive = false;
  bool region_valid = true;
};

struct ParsedStream {
  SequenceConfig sequence;
  std::vector<ParsedFrame> frames;
};

// Parses one or more concatenated temporal units (as written into IVF
// frames).  The reader never shares state with the writer: it re-derives CDF
// state, tile geometry and the BruModes map from the bytes.
bool parse_stream(std::span<const std::uint8_t> bytes, ParsedStream& out,
                  std::string& error);

}  // namespace av2enc::inter::bru
