// AV2 sequence-header writer/reader for the non-4:2:0 chroma formats.
//
// The M0/M1/M2 header writers hard-code `chroma_format_idc = 0` (4:2:0).  This
// header generalises §5.4.1 for 4:2:0, 4:2:2, 4:4:4 and 4:0:0 and, importantly,
// follows the Monochrome-conditional syntax:
//
//   sequence_partition_config():          enable_sdp is *absent* (inferred 0)
//   sequence_intra_config():              cfl_ds_filter_index is *absent*
//   sequence_transform_quant_entropy_config():
//                                         enable_chroma_dctonly, enable_cctx,
//                                         separate_uv_delta_q and the UV
//                                         delta-q fields are *absent*
//
// The rest of the sequence header matches the pinned M0 tool-disable set, so a
// stream differs from M0 only in the format fields and the Monochrome
// conditionals.
#pragma once

#include "bitstream/bitwriter.h"
#include "encoder/format/chroma/chroma_format.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace av2enc::chroma {

struct ChromaSeqConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  ChromaFormat chroma = ChromaFormat::k420;
  std::uint8_t base_q_idx = 90;
  bool disable_cdf_update = true;

  // Optional cropping window (§5.4.1).  The minimal still writer does not use
  // it; it is supported so a padded coded frame can be signalled as a smaller
  // visible frame if a future partition tree needs one.
  bool seq_cropping_window_present_flag = false;
  std::uint32_t seq_cropping_win_left_offset = 0;
  std::uint32_t seq_cropping_win_right_offset = 0;
  std::uint32_t seq_cropping_win_top_offset = 0;
  std::uint32_t seq_cropping_win_bottom_offset = 0;

  // Optional tools.  The minimal still writer keeps every one disabled.
  bool enable_dip = false;
  bool enable_intra_edge_filter = false;
  bool enable_mrls = false;
  bool enable_cfl_intra = false;
  int cfl_ds_filter_index = 0;
  bool enable_mhccp = false;
  bool enable_ibp = false;

  bool enable_fsc = false;
  bool enable_idtx_intra = false;
  bool enable_intra_ist = false;
  bool enable_inter_ist = false;
  bool enable_chroma_dctonly = false;
  bool reduce_tx_part_set = false;
  bool enable_cctx = false;
  bool enable_tcq = false;
  bool enable_parity_hiding = false;
  bool separate_uv_delta_q = false;
  bool equal_ac_dc_q = true;
  int base_uv_ac_delta_q = 23;  // DELTA_DCQUANT_MIN + 23 == 0
  bool uv_ac_delta_q_enabled = false;

  bool use_256x256_superblock = false;
  bool use_128x128_superblock = false;
  bool enable_sdp = false;
  bool enable_ext_partitions = false;
  bool reduce_pb_aspect_ratio = false;

  bool enable_ext_seg = false;
  bool seq_seg_info_present_flag = false;

  bool disable_loopfilters_across_tiles = false;
  bool enable_cdef = false;
  bool enable_gdf = false;
  bool enable_restoration = false;
  bool enable_ccso = false;
  std::uint8_t df_par_bits_minus_2 = 0;

  bool seq_tile_info_present_flag = false;
  bool film_grain_params_present = false;
};

// Validates the parts of the sequence header that are constrained by the
// chroma format (Monochrome conditionals, profiles, level, quantizer).
bool validate_chroma_seq_config(const ChromaSeqConfig& config,
                                std::string& error);

// Writes exactly the sequence_header_obu() syntax bits (§5.4.1), without the
// OBU header, obu_extension_flag or trailing_bits.  Returns the bit count.
std::size_t write_sequence_header_syntax(BitWriter& writer,
                                         const ChromaSeqConfig& config);

// Builds the length-delimited OBU_SEQUENCE_HEADER (Annex B.2) including the
// §5.2.1 obu_extension_flag and §5.2.3 trailing_bits.
std::vector<std::uint8_t> make_sequence_header_obu(
    const ChromaSeqConfig& config);

// Parsed sequence header.  `Monochrome`, `SubsamplingX/Y` and `NumPlanes` are
// the set_chroma_format_and_bit_depth() inferences, not independent fields.
struct SequenceHeaderInfo {
  int seq_profile_idc = -1;
  bool single_picture_header_flag = false;
  int seq_level_idx = -1;
  bool seq_tier = false;
  int chroma_format_idc = -1;
  ChromaFormat chroma = ChromaFormat::k420;
  int bit_depth_idc = -1;
  int bit_depth = 0;
  int max_quantizer = 0;
  bool monochrome = false;
  int subsampling_x = 0;
  int subsampling_y = 0;
  int num_planes = 0;
  std::uint32_t max_frame_width = 0;
  std::uint32_t max_frame_height = 0;
  bool cropping_window_present = false;
  std::uint32_t crop_left = 0;
  std::uint32_t crop_right = 0;
  std::uint32_t crop_top = 0;
  std::uint32_t crop_bottom = 0;
  bool enable_sdp = false;
  bool enable_ext_partitions = false;
  bool enable_cfl_intra = false;
  bool enable_mhccp = false;
  bool enable_chroma_dctonly = false;
  bool enable_cctx = false;
  bool separate_uv_delta_q = false;
  bool enable_cdef = false;
  bool enable_restoration = false;
  bool enable_ccso = false;
  bool film_grain_params_present = false;
};

// Reads the sequence_header_obu() syntax from `payload`, which starts at
// seq_header_id and may or may not include the trailing §5.2.1 extension flag
// and §5.2.3 trailing_bits.  When present they are validated.
bool read_sequence_header_syntax(std::span<const std::uint8_t> payload,
                                 SequenceHeaderInfo& info, std::string& error);

// Parses one length-delimited OBU_SEQUENCE_HEADER (as produced by
// make_sequence_header_obu()).  Returns false for a different OBU type,
// malformed leb128, or invalid syntax.
bool parse_sequence_header_obu(std::span<const std::uint8_t> obu,
                               SequenceHeaderInfo& info, std::string& error);

// Scans a temporal unit (TD + SEQ + CLK, as stored in an IVF frame) and parses
// the first OBU_SEQUENCE_HEADER.  Useful for gate/tests without depending on
// src/bitstream's OBU parser.
bool find_sequence_header(std::span<const std::uint8_t> temporal_unit,
                          SequenceHeaderInfo& info, std::string& error);

// Assembles TD -> SEQ -> CLK (§7.3.3) for one still picture.  The CLK OBU and
// its frame header come from the existing M0 writer; the frame header is
// chroma-format independent when all optional tools are disabled.
std::vector<std::uint8_t> assemble_chroma_still(
    const ChromaSeqConfig& config, std::span<const std::uint8_t> tile_payload);

}  // namespace av2enc::chroma
