#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::obu {

// Byte-level OBU container.
//
// The AV2 OBU header bit layout has NOT been implemented here. AV2 is still
// moving and copying AV1's header layout would be an invented syntax. The
// header agent should fill this in from docs/spec-* (cite the section number)
// and replace the TODO stubs below. Until then this type only keeps already-
// serialized header bytes next to payload bytes.
struct RawObu {
  std::vector<std::uint8_t> header;
  std::vector<std::uint8_t> payload;

  // Concatenates header bytes and payload bytes verbatim.
  std::vector<std::uint8_t> serialize() const;
};

// Appends `obu.header` then `obu.payload` to `out`, unchanged.
void append_serialized(std::vector<std::uint8_t>& out, const RawObu& obu);

// TODO(spec): Parse an AV2 OBU header from `bytes` and split it into a RawObu.
// Not implemented: returns false with an error instead of guessing syntax.
bool parse_header(std::span<const std::uint8_t> bytes,
                  RawObu& out,
                  std::size_t& consumed,
                  std::string& error);

// TODO(spec): Serialize an AV2 OBU header for the given raw type/flags.
// Not implemented: returns false with an error instead of guessing syntax.
bool write_header(std::uint8_t obu_type,
                  bool has_size_field,
                  std::vector<std::uint8_t>& out,
                  std::string& error);

}  // namespace av2enc::obu
