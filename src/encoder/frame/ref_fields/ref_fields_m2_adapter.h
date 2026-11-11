// M2 integration for the frame-header reference machinery.
//
// This is the *reference integration* for src/encoder/m2/m2_headers.cpp:
// it mirrors the M2 sequence header and frame header bit for bit and inserts
// the four §5.18 mechanisms that the M2 core does not emit:
//
//   * primary_ref_frame / disable_cross_frame_cdf_init   (§5.18.2)
//   * has_refresh_frame_flags / frame_to_refresh         (§5.18.2)
//   * max_drl_bits_minus_1 / change_drl                  (§5.18.2)
//   * change_bvp_drl / max_bvp_drl_bits_minus_1          (§5.18.3.4)
//   * frame_size_with_refs() / found_ref                 (§5.18.4.3)
//
// The sequence header mirror additionally can set the three sequence flags
// that gate those elements:
//
//   allow_frame_max_drl_bits, allow_frame_max_bvp_drl_bits,
//   enable_short_refresh_frame_flags
//
// With every new option off the emitted bytes are identical to
// m2::make_sequence_header_obu() and m2::write_frame_header(); the unit test
// pins that equality.
//
// §5.18.1 frame_header_copy() is a tile-group-level path: the first tile group
// of a frame records NumFrameHeaderBits, later tile groups repeat them.  See
// make_m2_ref_frame_obus().
#pragma once

#include "bitstream/bitwriter.h"
#include "encoder/frame/ref_fields/ref_fields.h"
#include "encoder/tiles/tile_layout.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::frame::ref_fields {

struct M2RefConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;
  std::uint8_t base_q_idx = 90;
  std::uint8_t order_hint_bits = 4;
  std::uint8_t num_ref_frames = 2;
  std::uint8_t num_total_refs = 1;
  std::uint8_t seq_max_drl_bits_minus_1 = 4;
  std::uint8_t seq_max_bvp_drl_bits_minus_1 = 0;
  bool allow_frame_max_drl_bits = false;
  bool allow_frame_max_bvp_drl_bits = false;
  bool enable_short_refresh_frame_flags = false;
  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;
};

// Uniform tile grid description needed by tile_info().  `make_m2_tile_info()`
// copies the level-derived bounds from a planned layout so the increment bits
// match what the decoder derives from the sequence level.
struct M2TileInfo {
  int cols_log2 = 0;
  int rows_log2 = 0;
  int min_cols_log2 = 0;
  int max_cols_log2 = 0;
  int min_tiles_log2 = 0;
  int max_rows_log2 = 0;
  int tile_count = 1;
  int tile_size_bytes = 1;
};

M2TileInfo make_m2_tile_info(const tiles::TileLayout& layout,
                             int tile_size_bytes);

// Scope: the M2 subset.  Every frame is an output frame
// (immediate_output_frame = 1), the reference map is explicit, segmentation /
// QM / delta-Q / loop filters stay off, and only one primary reference is
// considered.  Fields not listed here are written exactly as the M2 core does.
struct M2RefFrameHeader {
  bool keyframe = false;
  // Coded only for non-key frames.
  bool frame_is_inter = true;
  std::uint32_t order_hint = 0;

  // §5.18.2 refresh block.  For a CLK with max_mlayer_id == 0 the mask is
  // inferred to allFrames and 0 here means "infer allFrames".
  std::uint32_t refresh_frame_flags = 1;

  // §5.18.2 primary-ref block (IGNORED for key/intra-only/switch frames).
  bool signal_primary_ref_frame = false;
  std::uint32_t primary_ref_frame = kPrimaryRefChoose;
  bool disable_cross_frame_cdf_init = true;

  // §5.18.4 frame size.  frame_size_override_flag is coded for every
  // non-single-picture frame, key or inter.
  bool frame_size_override = false;
  // Inter frames only: inherit from a reference when possible.
  bool prefer_found_ref = true;
  std::uint32_t frame_width = 0;   // explicit frame_size() values
  std::uint32_t frame_height = 0;

  // §5.18.3.4 intrabc_params().
  bool allow_intrabc = false;
  bool allow_global_intrabc = false;   // intra frames only
  bool allow_local_intrabc = true;     // intra frames, when global is 1
  bool change_bvp_drl = false;
  std::uint32_t max_bvp_drl_bits_minus_1 = 0;

  // §5.18.2 max_drl_bits_minus_1 (inter frames only).
  bool change_drl = false;
  std::uint32_t max_drl_bits_minus_1 = 0;

  // §5.18.2 explicit reference map.  The M2 sequence header always sets
  // explicit_ref_frame_map = 1, so this is coded for every inter frame; the
  // implicit get_ref_frames() path is out of scope here.
  bool explicit_ref_frame_map = true;
  std::uint8_t ref_slot = 0;
};

// Parsed form of one complete M2 frame header (first tile group).
struct M2RefParsedHeader {
  M2RefFrameHeader header;
  FrameSizeResolved size;
  PrimaryRefResolved primary;
  RefreshSignal refresh;
  DrlSignal drl;
  BvDrlSignal bvp_drl;
  std::uint32_t base_q_idx = 0;
  bool segmentation_enabled = false;
  bool using_qmatrix = false;
  bool delta_q_present = false;
  bool apply_deblocking_filter[2] = {false, false};
  bool reference_select = false;
  bool skip_mode_present = false;
};

// sequence_header_obu() mirror with the three gate flags.
std::vector<std::uint8_t> make_m2_ref_sequence_header_obu(
    const M2RefConfig& config, std::string* error);

// frame_header_info() syntax without is_first_tile_group and without
// byte_alignment().  `header_bits` receives the number of bits of
// frame_header_info() (what a later frame_header_copy() must repeat).
std::size_t write_m2_ref_frame_header(BitWriter& writer,
                                      const M2RefConfig& config,
                                      const M2RefFrameHeader& header,
                                      const RefList& refs,
                                      const FrameRefState& state,
                                      const M2TileInfo& tiles,
                                      std::size_t* header_bits,
                                      std::string* error);

// Complete Annex B OBUs for a frame split into `tile_payloads` (raster order).
// One OBU is returned per tile group:
//   split_tile_groups == false : one OBU carrying every tile
//                                (tile_start_and_end_present_flag = 0)
//   split_tile_groups == true  : one OBU per tile.  The first OBU carries the
//                                frame header; every later OBU sets
//                                frame_header_present_flag = 1 and repeats the
//                                first header through frame_header_copy()
//                                (§5.18.1), the shared-header path.
// Returns an empty vector on error.  §7.3.3 requires the same-type OBUs to be
// contiguous in the coded frame unit; the caller appends them in order.
std::vector<std::vector<std::uint8_t>> make_m2_ref_frame_obus(
    const M2RefConfig& config, const M2RefFrameHeader& header,
    const RefList& refs, const FrameRefState& state, const M2TileInfo& tiles,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    bool split_tile_groups, std::string* error);

// Independent reader for the same header subset.  Validates every value
// against `refs`/`state` exactly like the writer.  `tiles` must describe the
// same grid; the reader uses it to know how many tile_info bits to consume.
bool read_m2_ref_frame_header(RefBitReader& reader, bool keyframe,
                              const M2RefConfig& config, const RefList& refs,
                              const FrameRefState& state,
                              const M2TileInfo& tiles,
                              M2RefParsedHeader* out, std::string* error);

// §5.19 non-first tile group header reader:
//   frame_header_present_flag f(1)
//   if present: frame_header_copy()  (verified against `store`)
bool read_m2_ref_nonfirst_tile_group_header(RefBitReader& reader,
                                            const FrameHeaderStore& store,
                                            bool* frame_header_present,
                                            std::string* error);

}  // namespace av2enc::frame::ref_fields
