// Tiled M2 frame headers and tile groups.
//
// This is the M2 counterpart of src/encoder/m2/video_headers.cpp for NumTiles > 1.
// The core writer is intentionally single-tile (it writes increment_tile_* = 0
// and no context_update_tile_id/tile_size_bytes fields); it is owned by another
// agent and must not be edited here.  This module duplicates the exact same
// syntax, adds the §5.18.7.2-7 tile_info tail, and delegates the filter rows to
// the verified adapters in src/encoder/m1/filters_stream/.
//
// Byte-exactness contract: for one tile, depth 8, all loop filters off, and the
// same tile payload, make_video_frame_obu() produces the same OBU bytes as
// av2enc::video::make_frame_obu().  The unit test pins that equality.  Any future change
// to the core header must be mirrored here or the test fails.
//
// Tile payloads are produced by re-entering av2enc::video::syntax::assemble_video_frame()
// once per tile with a tile-local block list and tile-local frame plan.  That
// is legal because AV2 tile boundaries are superblock-aligned and every
// partition subtree is contained in its superblock:
//   * block roots are split on superblock rows/columns;
//   * mi coordinates are shifted to the tile origin;
//   * per-tile context arrays start cleared, exactly as the decoder's tile
//     start does;
//   * every tile starts from the frame CDFs (§8.2.2).  With adaptive CDFs the
//     caller seeds each tile from the same frame CDFs and runs the §8.2.4
//     save/§7.5 average bookkeeping once per frame; with enable_avg_cdf &&
//     avg_cdf_type §5.18.7.2 infers context_update_tile_id = 0 and codes no
//     bits for it.  With adaptation off the per-tile CDFs are frozen and the
//     tail still codes context_update_tile_id = 0.
//
// Not implemented here: the per-block read_cdef()/read_ccso() symbols that must
// be inserted into the tile payload when those tools are enabled.  The header
// writers are provided and tested; the tile hooks are integration work in the
// M2 syntax writer.  Deblocking needs no tile symbols and can be enabled with
// just the frame header flags.
#pragma once

#include "core/bitstream/bitwriter.h"
#include "formats/format.h"
#include "codec/filters/stream/filters_stream.h"
#include "pipeline/video/post/video_filter_plan.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/post/loop_filters.h"
#include "pipeline/planner/pipeline.h"
#include "pipeline/tiles/tile_group.h"
#include "pipeline/tiles/tile_layout.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::video::post {

namespace filters_stream = av2enc::still::filters_stream;

struct VideoPostSequenceOptions {
  fmt::BitDepth depth = fmt::BitDepth::k8;
  // §5.4.10.  The default matches av2enc::video::make_sequence_header_obu(): all tools
  // off, cdef_on_skip_txfm_disabled = 1.
  filters_stream::SequenceFilterConfig filters{};
  // §5.4.8 average-CDF config.  Off by default so every previously pinned
  // stream keeps the same sequence header.  When enabled, avg_cdf_type = 1
  // averages the first §8.2.4 tile set at frame end and §5.18.7.2 omits
  // context_update_tile_id from tile_info().
  bool enable_avg_cdf = false;
  bool avg_cdf_type = true;
  // §5.4.2 film_grain_params_present (--video-grain).  Off by default, so the
  // tiled sequence header keeps its historical bit.
  bool film_grain_params_present = false;

  VideoPostSequenceOptions() { filters.cdef_on_skip_txfm_disabled = true; }
};

struct VideoTilePlan {
  bool ok = false;
  tiles::TileLayout layout{};
  std::string error;
};

struct VideoTiledFrameHeader;

// Validated wrapper around tiles::plan_tile_layout() plus the M2 geometry
// constraints (dimensions divisible by 8, superblock 64/128).
VideoTilePlan plan_video_tiles(int width, int height, int cols, int rows,
                         int sb_size = 64, int seq_level_idx = 7,
                         int seq_tier = 0);

// Adoption helper for the verified pipeline planner.  Copies the already
// validated layout and frame qindex from a pipeline::FramePipelinePlan.  The
// current M2 header writer signals only a frame-uniform base_q_idx, so a plan
// with per-tile qindexes (delta-Q or segmentation) is rejected instead of
// silently dropping the tile decisions.
bool adopt_pipeline_frame_plan(const pipeline::FramePipelinePlan& plan,
                               VideoTiledFrameHeader* header, VideoTilePlan* tiles,
                               std::string* error = nullptr);

// Frame-header fields owned by this layer.  base_q_idx < 0 means "use
// config.base_q_idx".  The filter flags must agree with the sequence options;
// a mismatch is rejected instead of being silently dropped.
struct VideoTiledFrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint8_t ref_slot = 0;
  std::uint32_t refresh_frame_flags = 1;
  int base_q_idx = -1;

  // §5.18.8.1 tx_mode_select (0 = TX_MODE_LARGEST, 1 = TX_MODE_SELECT).  The
  // bit is frame-level, so every tile payload in the group must have been
  // coded under the same mode.
  bool tx_mode_select = false;

  // §5.18.10.1 film_grain_config().  Written only when the sequence sets
  // film_grain_params_present; every tiled picture is an immediate-output
  // picture, so apply_grain is always written (never inferred).
  bool apply_grain = false;
  std::uint8_t fgm_id = 0;
  std::uint16_t grain_seed = 0;

  bool apply_deblocking_filter[4] = {false, false, false, false};
  bool df_delta_q_present[4] = {false, false, false, false};
  // §5.18.7.9 DfDeltaQ[i]; coded only for a present plane.  The range is
  // decided by sequence_filter_config().df_par_bits_minus_2 (written from
  // VideoPostSequenceOptions::filters).
  int df_delta_q[4] = {0, 0, 0, 0};

  bool cdef_frame_enable = false;
  filters_stream::CdefParams cdef{};

  bool ccso_frame_flag = false;
  filters_stream::CcsoParams ccso{};

  // §5.18.7.11 lr_params() / §5.18.7.9 gdf_params().  The frame header carries
  // the frame-level parameters; the per-unit symbols stay in the tile payload.
  VideoLrPlan lr{};
  VideoGdfPlan gdf{};
};

// Converts a per-frame filter decision into the header-parameter form the
// §5.18.7.10/§5.18.7.12 writers take.
filters_stream::CdefParams make_tiled_cdef_params(const VideoCdefPlan& plan);
filters_stream::CcsoParams make_tiled_ccso_params(const VideoCcsoPlan& plan);

// tile_info() + the §5.18.7.2 tail.  Exposed for unit tests and for callers
// that want the exact bit count.  `context_update_present` is false when
// enable_avg_cdf && avg_cdf_type, in which case the spec infers
// context_update_tile_id = 0 and codes no bits for it.
bool write_video_tile_info(BitWriter& writer, const tiles::TileLayout& layout,
                        int tile_size_bytes, std::string* error,
                        bool context_update_present = true);

// frame_header_info() syntax, without the leading is_first_tile_group bit and
// without byte_alignment().  The caller must know TileSizeBytes first; for a
// one-tile frame `tile_size_bytes` is ignored.
bool write_video_frame_header_syntax(BitWriter& writer, const VideoConfig& config,
                                  const VideoPostSequenceOptions& sequence,
                                  const VideoTiledFrameHeader& header,
                                  const tiles::TileLayout& layout,
                                  int tile_size_bytes, std::size_t* bits,
                                  std::string* error);

// Tile-local block list produced by splitting the whole-frame list.
struct VideoTileBlocks {
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
bool split_video_blocks_for_tile(const tiles::TileLayout& layout, int tile_index,
                              std::span<const syntax::BlockSyntax> blocks,
                              VideoTileBlocks* out, std::string* error);

struct VideoTiledFrame {
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
bool assemble_video_tiled_frame(const VideoConfig& config,
                             const VideoPostSequenceOptions& sequence,
                             const VideoTiledFrameHeader& header,
                             const tiles::TileLayout& layout,
                             std::span<const syntax::BlockSyntax> blocks,
                             std::span<const std::int32_t> levels,
                             VideoTiledFrame* out, std::string* error);

// Sequence header with the §5.4.10 filter row.  For the default 8-bit,
// all-filters-off options this is byte-identical to
// av2enc::video::make_sequence_header_obu(config).
std::vector<std::uint8_t> make_video_sequence_header_obu(
    const VideoConfig& config, const VideoPostSequenceOptions& sequence,
    std::string* error);

// Escape hatch for callers that already have tile payloads (for example a GPU
// tile coder).  `tile_payloads` must be in raster TileNum order.
bool assemble_video_tile_group_from_payloads(
    const VideoConfig& config, const VideoPostSequenceOptions& sequence,
    const VideoTiledFrameHeader& header, const tiles::TileLayout& layout,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    VideoTiledFrame* out, std::string* error);

}  // namespace av2enc::video::post
