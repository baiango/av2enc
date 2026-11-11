// Tiled M2 frame headers and tile groups.
//
// This is the M2 counterpart of src/encoder/m2/m2_headers.cpp for NumTiles > 1.
// The core writer is intentionally single-tile (it writes increment_tile_* = 0
// and no context_update_tile_id/tile_size_bytes fields); it is owned by another
// agent and must not be edited here.  This module duplicates the exact same
// syntax, adds the §5.18.7.2-7 tile_info tail, and delegates the filter rows to
// the verified adapters in src/encoder/m1/filters_stream/.
//
// Byte-exactness contract: for one tile, depth 8, all loop filters off, and the
// same tile payload, make_m2_frame_obu() produces the same OBU bytes as
// m2::make_frame_obu().  The unit test pins that equality.  Any future change
// to the core header must be mirrored here or the test fails.
//
// Tile payloads are produced by re-entering m2::syntax::assemble_m2_frame()
// once per tile with a tile-local block list and tile-local frame plan.  That
// is legal because AV2 tile boundaries are superblock-aligned and every
// partition subtree is contained in its superblock:
//   * block roots are split on superblock rows/columns;
//   * mi coordinates are shifted to the tile origin;
//   * per-tile context arrays start cleared, exactly as the decoder's tile
//     start does;
//   * CDFs are initialised per tile.  This is only correct with
//     disable_cdf_update = 1 (frozen defaults); with adaptive CDFs the decoder
//     carries CDF state between tiles (and context_update_tile_id), which this
//     layer does not model, so multi-tile + adaptive CDFs is rejected.
//
// Not implemented here: the per-block read_cdef()/read_ccso() symbols that must
// be inserted into the tile payload when those tools are enabled.  The header
// writers are provided and tested; the tile hooks are integration work in the
// M2 syntax writer.  Deblocking needs no tile symbols and can be enabled with
// just the frame header flags.
#pragma once

#include "bitstream/bitwriter.h"
#include "encoder/format/format.h"
#include "encoder/m1/filters_stream/filters_stream.h"
#include "encoder/m2/m2_headers.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/post/loop_filters.h"
#include "encoder/pipeline/pipeline.h"
#include "encoder/tiles/tile_group.h"
#include "encoder/tiles/tile_layout.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m2::post {

namespace filters_stream = av2enc::m1::filters_stream;

struct M2PostSequenceOptions {
  fmt::BitDepth depth = fmt::BitDepth::k8;
  // §5.4.10.  The default matches m2::make_sequence_header_obu(): all tools
  // off, cdef_on_skip_txfm_disabled = 1.
  filters_stream::SequenceFilterConfig filters{};

  M2PostSequenceOptions() { filters.cdef_on_skip_txfm_disabled = true; }
};

struct M2TilePlan {
  bool ok = false;
  tiles::TileLayout layout{};
  std::string error;
};

struct M2TiledFrameHeader;

// Validated wrapper around tiles::plan_tile_layout() plus the M2 geometry
// constraints (dimensions divisible by 8, superblock 64/128).
M2TilePlan plan_m2_tiles(int width, int height, int cols, int rows,
                         int sb_size = 64, int seq_level_idx = 7,
                         int seq_tier = 0);

// Adoption helper for the verified pipeline planner.  Copies the already
// validated layout and frame qindex from a pipeline::FramePipelinePlan.  The
// current M2 header writer signals only a frame-uniform base_q_idx, so a plan
// with per-tile qindexes (delta-Q or segmentation) is rejected instead of
// silently dropping the tile decisions.
bool adopt_pipeline_frame_plan(const pipeline::FramePipelinePlan& plan,
                               M2TiledFrameHeader* header, M2TilePlan* tiles,
                               std::string* error = nullptr);

// Frame-header fields owned by this layer.  base_q_idx < 0 means "use
// config.base_q_idx".  The filter flags must agree with the sequence options;
// a mismatch is rejected instead of being silently dropped.
struct M2TiledFrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint8_t ref_slot = 0;
  std::uint32_t refresh_frame_flags = 1;
  int base_q_idx = -1;

  bool apply_deblocking_filter[4] = {false, false, false, false};
  bool df_delta_q_present[4] = {false, false, false, false};
  // §5.18.7.9 DfDeltaQ[i]; coded only for a present plane.  The range is
  // decided by sequence_filter_config().df_par_bits_minus_2 (written from
  // M2PostSequenceOptions::filters).
  int df_delta_q[4] = {0, 0, 0, 0};

  bool cdef_frame_enable = false;
  filters_stream::CdefParams cdef{};

  bool ccso_frame_flag = false;
  filters_stream::CcsoParams ccso{};
};

// tile_info() + the §5.18.7.2 tail.  Exposed for unit tests and for callers
// that want the exact bit count.
bool write_m2_tile_info(BitWriter& writer, const tiles::TileLayout& layout,
                        int tile_size_bytes, std::string* error);

// frame_header_info() syntax, without the leading is_first_tile_group bit and
// without byte_alignment().  The caller must know TileSizeBytes first; for a
// one-tile frame `tile_size_bytes` is ignored.
bool write_m2_frame_header_syntax(BitWriter& writer, const M2Config& config,
                                  const M2PostSequenceOptions& sequence,
                                  const M2TiledFrameHeader& header,
                                  const tiles::TileLayout& layout,
                                  int tile_size_bytes, std::size_t* bits,
                                  std::string* error);

// Tile-local block list produced by splitting the whole-frame list.
struct M2TileBlocks {
  int tile_index = 0;
  int width = 0;
  int height = 0;
  int sb_cols = 0;
  int sb_rows = 0;
  std::vector<syntax::BlockSyntax> blocks;
};

// Splits `blocks` (whole-frame, superblock-rooted, preorder) into one tile.
// Every root must fall inside the tile's superblock rectangle; coordinates are
// shifted to the tile origin.  Levels are not touched (they are indexed by the
// caller's absolute levels_offset).
bool split_m2_blocks_for_tile(const tiles::TileLayout& layout, int tile_index,
                              std::span<const syntax::BlockSyntax> blocks,
                              M2TileBlocks* out, std::string* error);

struct M2TiledFrame {
  std::vector<std::vector<std::uint8_t>> tile_payloads;  // raster order
  std::vector<std::uint32_t> tile_bytes;
  std::uint32_t largest_tile_bytes = 0;
  int tile_size_bytes = 0;
  std::size_t frame_header_bits = 0;
  std::size_t tile_group_header_bytes = 0;
  std::vector<std::uint8_t> obu;
};

// Encodes every tile and assembles one OBU_CLOSED_LOOP_KEY /
// OBU_REGULAR_TILE_GROUP payload (Annex B length-delimited OBU).
bool assemble_m2_tiled_frame(const M2Config& config,
                             const M2PostSequenceOptions& sequence,
                             const M2TiledFrameHeader& header,
                             const tiles::TileLayout& layout,
                             std::span<const syntax::BlockSyntax> blocks,
                             std::span<const std::int32_t> levels,
                             M2TiledFrame* out, std::string* error);

// Sequence header with the §5.4.10 filter row.  For the default 8-bit,
// all-filters-off options this is byte-identical to
// m2::make_sequence_header_obu(config).
std::vector<std::uint8_t> make_m2_sequence_header_obu(
    const M2Config& config, const M2PostSequenceOptions& sequence,
    std::string* error);

// Escape hatch for callers that already have tile payloads (for example a GPU
// tile coder).  `tile_payloads` must be in raster TileNum order.
bool assemble_m2_tile_group_from_payloads(
    const M2Config& config, const M2PostSequenceOptions& sequence,
    const M2TiledFrameHeader& header, const tiles::TileLayout& layout,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    M2TiledFrame* out, std::string* error);

}  // namespace av2enc::m2::post
