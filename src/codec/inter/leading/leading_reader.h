#pragma once

#include "codec/inter/leading/leading.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::inter::leading {

// One Annex B.2 length-delimited OBU.  The span points into the caller's input.
struct ParsedObu {
  ObuType type = ObuType::kTemporalDelimiter;
  bool header_extension = false;
  std::uint8_t tlayer_id = 0;
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
  std::size_t offset = 0;        // first byte of the leb128 prefix
  std::size_t header_bytes = 1;  // 1 or 2
  std::size_t payload_offset = 0;
  std::size_t payload_size = 0;
  std::size_t total_size = 0;  // leb128 + header + payload
  std::span<const std::uint8_t> payload;
};

// Parses one OBU.  Fails on a truncated leb128, a zero-length OBU, an
// out-of-range payload size, or a reserved OBU type this module does not know.
bool parse_obu(std::span<const std::uint8_t> bytes, ParsedObu& obu,
               std::string& error);

// Parses every OBU in `bytes`.  Trailing bytes that are not a whole OBU fail.
bool parse_obu_sequence(std::span<const std::uint8_t> bytes,
                        std::vector<ParsedObu>& obus, std::string& error);

// Fields recovered from §5.18.2 frame_header_info().  Only the fields that vary
// between leading/regular frames are stored; fixed tool-set fields are
// validated in place.
struct ParsedFrameHeader {
  FrameKind kind = FrameKind::kRegularTileGroup;
  bool key_frame = false;
  bool is_regular = false;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 0;
  bool immediate_output_frame = false;
  bool implicit_output_frame = false;
  bool disable_cdf_update = true;
  std::uint8_t num_total_refs = 0;
  std::array<std::uint8_t, 8> ref_frame_idx{};
  // Number of bytes occupied by is_first_tile_group + frame_header +
  // byte_alignment.  The tile payload starts at this offset in the OBU payload.
  std::size_t header_bytes = 0;
};

// Parses the tile_group_obu() header from a frame OBU payload.  `type` is the
// OBU type recovered by parse_obu().  SEF/TIP OBUs are rejected here (they have
// no tile payload).
bool parse_frame_header(std::span<const std::uint8_t> payload, ObuType type,
                        const SequenceConfig& config, ParsedFrameHeader& header,
                        std::string& error);

// Parses and validates sequence_header_obu() for the tool set leading.cpp
// writes.  Round-trips with make_sequence_header_obu().
bool parse_sequence_header(std::span<const std::uint8_t> payload,
                           SequenceConfig& config, std::string& error);

}  // namespace av2enc::inter::leading
