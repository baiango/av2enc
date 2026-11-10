#include "bitstream/obu.h"

namespace av2enc::obu {

std::vector<std::uint8_t> RawObu::serialize() const {
  std::vector<std::uint8_t> result;
  result.reserve(header.size() + payload.size());
  result.insert(result.end(), header.begin(), header.end());
  result.insert(result.end(), payload.begin(), payload.end());
  return result;
}

void append_serialized(std::vector<std::uint8_t>& out, const RawObu& obu) {
  out.insert(out.end(), obu.header.begin(), obu.header.end());
  out.insert(out.end(), obu.payload.begin(), obu.payload.end());
}

bool parse_header(std::span<const std::uint8_t>,
                  RawObu&,
                  std::size_t&,
                  std::string& error) {
  error =
      "OBU parsing is not implemented: the AV2 OBU header layout is not pinned yet "
      "(see docs/spec-*)";
  return false;
}

bool write_header(std::uint8_t,
                  bool,
                  std::vector<std::uint8_t>&,
                  std::string& error) {
  error =
      "OBU header writing is not implemented: the AV2 OBU header layout is not pinned yet "
      "(see docs/spec-*)";
  return false;
}

}  // namespace av2enc::obu
