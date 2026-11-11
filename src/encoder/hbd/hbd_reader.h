// Reader for the 10-bit `src/encoder/hbd/` still stream.
//
// The project's generic OBU parser (`src/bitstream/obu.*`) is intentionally
// still unimplemented, so this component parses its own Annex B OBU framing and
// the exact syntax it writes.  It is a *bit golden* checker, not a full AV2
// decoder: it validates the 10-bit sequence/frame fields, the tile-group
// header boundary, and returns the tile payload bytes.
//
// Normative clauses: §5.2.2 (OBU header), Annex B.2 (leb128 size prefix),
// §5.4.1 (sequence header), §5.18.2 (frame header), §5.18.7.3 (tile_info,
// single-tile subset), §5.19 (tile group), §5.2.4 (byte_alignment).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::hbd {

struct HbdObu {
  int type = 0;                 // §6.2.2 Table 6.1
  int tlayer_id = 0;
  bool extension_flag = false;
  std::size_t prefix_offset = 0;  // first leb128 byte
  std::size_t header_bytes = 1;   // OBU header (+ extension byte)
  std::size_t payload_offset = 0;
  std::size_t payload_size = 0;
};

struct HbdSequenceHeader {
  std::uint32_t seq_header_id = 0;
  int seq_profile_idc = 0;
  bool single_picture_header_flag = false;
  int seq_level_idx = 0;
  int chroma_format_idc = 0;
  int bit_depth_idc = 1;
  std::uint32_t max_frame_width = 0;
  std::uint32_t max_frame_height = 0;
  bool enable_intra_edge_filter = false;
  bool enable_mrls = false;
  bool enable_cfl_intra = false;
  int cfl_ds_filter_index = 0;
  bool enable_chroma_dctonly = false;
  bool enable_cdef = false;
  bool enable_ccso = false;
  bool seq_tile_info_present_flag = false;
};

struct HbdFrameHeader {
  bool is_first_tile_group = false;
  std::uint16_t base_q_idx = 0;
  bool disable_cdf_update = false;
  bool segmentation_enabled = false;
  bool using_qmatrix = false;
  bool delta_q_present = false;
  bool apply_deblocking_filter[4] = {false, false, false, false};
  bool tx_mode_select = false;
  int reduced_tx_set = 0;
};

struct HbdParsedStream {
  std::vector<HbdObu> obus;
  HbdSequenceHeader sequence;
  bool has_sequence = false;
  HbdFrameHeader frame;
  bool has_frame = false;
  std::vector<std::uint8_t> tile_payload;
  std::size_t tile_payload_offset = 0;  // within the input span
};

// Parses one in-memory still stream.  Returns false with `error` on malformed
// framing, a non-10-bit sequence header, trailing garbage, or a tile-group
// header that does not end on a byte boundary.
bool parse_stream(std::span<const std::uint8_t> bytes, HbdParsedStream* out,
                  std::string* error);

}  // namespace av2enc::hbd
