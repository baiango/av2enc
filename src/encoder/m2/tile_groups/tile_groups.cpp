#include "encoder/m2/tile_groups/tile_groups.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace av2enc::m2::tile_groups {
namespace {

constexpr std::uint32_t kMinTileBytes = 2;  // init_symbol() reads 15 bits

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

std::string group_label(std::size_t index) {
  std::ostringstream out;
  out << "tile group " << index;
  return out.str();
}

unsigned bit_of(std::uint8_t byte, unsigned index) {
  return (byte >> (7U - index)) & 1U;
}

bool header_bit_at(const FrameHeaderBits& bits, std::size_t index) {
  const std::size_t byte = index >> 3U;
  return bit_of(bits.bytes[byte], static_cast<unsigned>(index & 7U)) != 0U;
}

void put_frame_header_bits(BitWriter& writer, const FrameHeaderBits& bits) {
  for (std::size_t i = 0; i < bits.bit_count; ++i) {
    writer.put_bit(header_bit_at(bits, i));
  }
}

int ceil_log2_exact(int value) {
  if (value <= 1) return 0;
  int bits = 0;
  int v = value - 1;
  while (v != 0) {
    v >>= 1;
    ++bits;
  }
  return bits;
}

bool is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// Reads bits MSB-first out of an OBU payload. All accesses are bounds checked.
class PayloadBitReader {
 public:
  explicit PayloadBitReader(std::span<const std::uint8_t> bytes)
      : bytes_(bytes) {}

  bool get_bit(bool* out) {
    if (bit_pos_ >= total_bits()) return false;
    const std::size_t byte = bit_pos_ >> 3U;
    const unsigned index = static_cast<unsigned>(bit_pos_ & 7U);
    *out = bit_of(bytes_[byte], index) != 0U;
    ++bit_pos_;
    return true;
  }

  bool get_bits(unsigned count, std::uint64_t* out) {
    if (count > 64U) return false;
    std::uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!get_bit(&bit)) return false;
      value = (value << 1U) | (bit ? 1U : 0U);
    }
    *out = value;
    return true;
  }

  // §5.19 byte_alignment(): zero bits until the next byte boundary.
  bool align_to_byte_zero(std::string* error) {
    while ((bit_pos_ & 7U) != 0U) {
      bool bit = false;
      if (!get_bit(&bit)) {
        return fail(error, "truncated byte_alignment()");
      }
      if (bit) {
        return fail(error, "byte_alignment() padding contains a one bit");
      }
    }
    return true;
  }

  std::size_t bit_position() const { return bit_pos_; }
  std::size_t total_bits() const { return bytes_.size() * 8U; }
  bool at_end() const { return bit_pos_ == total_bits(); }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

std::uint32_t read_le(std::span<const std::uint8_t> payload,
                      std::size_t offset, int width) {
  std::uint32_t value = 0;
  for (int i = 0; i < width; ++i) {
    value |= static_cast<std::uint32_t>(payload[offset + static_cast<std::size_t>(i)])
             << (8U * static_cast<unsigned>(i));
  }
  return value;
}

bool validate_span(const TileGroupGeometry& geometry,
                   const TileGroupSpan& span, std::string* error) {
  if (span.tg_start < 0 || span.tg_end < 0) {
    return fail(error, "tile group has a negative tg_start/tg_end");
  }
  if (span.tg_end < span.tg_start) {
    return fail(error, "tg_end is less than tg_start (§6.18)");
  }
  if (span.tg_end >= geometry.num_tiles) {
    std::ostringstream out;
    out << "tg_end " << span.tg_end << " is outside NumTiles " << geometry.num_tiles;
    return fail(error, out.str());
  }
  if (geometry.use_bru && !geometry.bru_inactive &&
      geometry.num_tiles > 1) {
    if (span.bru_tile_active.size() !=
        static_cast<std::size_t>(span.tile_count())) {
      return fail(error,
                  "bru_tile_active must have one entry per tile in the group");
    }
    for (std::uint8_t active : span.bru_tile_active) {
      if (active > 1U) {
        return fail(error, "bru_tile_active must be 0 or 1");
      }
    }
  } else if (!span.bru_tile_active.empty()) {
    return fail(error,
                "bru_tile_active is only coded when use_bru = 1, "
                "bru_inactive = 0 and NumTiles > 1");
  }
  return true;
}

bool validate_payload_size(std::uint32_t size, int tile_size_bytes,
                           bool require_capacity, std::string* error) {
  if (size < kMinTileBytes) {
    return fail(error, "every tile payload must be at least 2 bytes");
  }
  if (require_capacity && tile_size_bytes >= 1 && tile_size_bytes <= 4) {
    const std::uint64_t capacity = 1ull << (8U * static_cast<unsigned>(tile_size_bytes));
    if (static_cast<std::uint64_t>(size) > capacity) {
      std::ostringstream out;
      out << "tile payload of " << size << " bytes does not fit TileSizeBytes = "
          << tile_size_bytes;
      return fail(error, out.str());
    }
  }
  return true;
}

}  // namespace

bool capture_frame_header_bits(std::span<const std::uint8_t> bytes,
                               std::size_t bit_count, FrameHeaderBits* out,
                               std::string* error) {
  if (out == nullptr) return fail(error, "null frame header output");
  const std::size_t needed = (bit_count + 7U) / 8U;
  if (bit_count == 0) {
    return fail(error, "NumFrameHeaderBits must be greater than zero");
  }
  if (bytes.size() < needed) {
    std::ostringstream text;
    text << "frame header needs " << needed << " bytes for " << bit_count
         << " bits, got " << bytes.size();
    return fail(error, text.str());
  }
  out->bytes.assign(bytes.begin(),
                    bytes.begin() + static_cast<std::ptrdiff_t>(needed));
  // Clear padding bits above bit_count so copies and goldens are canonical.
  const unsigned used = static_cast<unsigned>(bit_count & 7U);
  if (used != 0U) {
    out->bytes.back() &=
        static_cast<std::uint8_t>(0xFFU << (8U - used));
  }
  out->bit_count = bit_count;
  return true;
}

bool frame_header_bit(const FrameHeaderBits& bits, std::size_t index,
                      bool* in_range) {
  const bool valid = index < bits.bit_count;
  if (in_range != nullptr) *in_range = valid;
  return valid && header_bit_at(bits, index);
}

TileGroupGeometry geometry_from_layout(const tiles::TileLayout& layout,
                                       int tile_size_bytes, bool use_bru,
                                       bool bru_inactive) {
  TileGroupGeometry geometry;
  geometry.num_tiles = layout.tile_count();
  geometry.tile_cols_log2 = layout.cols_log2;
  geometry.tile_rows_log2 = layout.rows_log2;
  geometry.tile_size_bytes = tile_size_bytes;
  geometry.use_bru = use_bru;
  geometry.bru_inactive = bru_inactive;
  return geometry;
}

bool validate_tile_group_geometry(const TileGroupGeometry& geometry,
                                  std::string* error) {
  if (geometry.num_tiles < 1) {
    return fail(error, "NumTiles must be at least 1");
  }
  if (!is_power_of_two(geometry.num_tiles)) {
    return fail(error, "NumTiles must be a power of two (TileCols * TileRows)");
  }
  if (geometry.tile_cols_log2 < 0 || geometry.tile_rows_log2 < 0) {
    return fail(error, "TileColsLog2/TileRowsLog2 must be non-negative");
  }
  if (geometry.num_tiles > (1 << 12)) {
    return fail(error, "NumTiles exceeds the AV2 64x64 tile limit");
  }
  if (geometry.tile_bits() != ceil_log2_exact(geometry.num_tiles)) {
    std::ostringstream out;
    out << "TileColsLog2 + TileRowsLog2 (" << geometry.tile_bits()
        << ") does not describe NumTiles (" << geometry.num_tiles << ")";
    return fail(error, out.str());
  }
  if (geometry.num_tiles == 1) {
    if (geometry.tile_size_bytes != 0) {
      return fail(error,
                  "TileSizeBytes is not signalled when NumTiles == 1");
    }
  } else if (geometry.tile_size_bytes < 1 || geometry.tile_size_bytes > 4) {
    return fail(error, "TileSizeBytes must be in 1..4");
  }
  if (geometry.bru_inactive && !geometry.use_bru) {
    return fail(error,
                "bru_inactive is only defined when the frame header has "
                "use_bru = 1");
  }
  return true;
}

bool plan_contiguous_tile_groups(const TileGroupGeometry& geometry,
                                 std::span<const int> tiles_per_group,
                                 bool copy_frame_header,
                                 bool bru_active_default,
                                 std::vector<TileGroupSpan>* groups,
                                 std::string* error) {
  if (groups == nullptr) return fail(error, "null tile group output");
  if (!validate_tile_group_geometry(geometry, error)) return false;
  if (tiles_per_group.empty()) {
    return fail(error, "at least one tile group is required");
  }
  if (geometry.bru_inactive && tiles_per_group.size() != 1) {
    return fail(error,
                "a bru_inactive frame is a single header-only tile group");
  }
  std::vector<TileGroupSpan> result;
  result.reserve(tiles_per_group.size());
  int next = 0;
  for (std::size_t i = 0; i < tiles_per_group.size(); ++i) {
    const int count = tiles_per_group[i];
    if (count < 1) {
      return fail(error, "every tile group must contain at least one tile");
    }
    if (next + count > geometry.num_tiles) {
      return fail(error, "tile group sizes exceed NumTiles");
    }
    TileGroupSpan span;
    span.tg_start = next;
    span.tg_end = next + count - 1;
    span.frame_header_present = i > 0 && copy_frame_header;
    if (geometry.use_bru && !geometry.bru_inactive &&
        geometry.num_tiles > 1) {
      span.bru_tile_active.assign(
          static_cast<std::size_t>(count),
          static_cast<std::uint8_t>(bru_active_default ? 1U : 0U));
    }
    result.push_back(std::move(span));
    next += count;
  }
  if (next != geometry.num_tiles) {
    std::ostringstream out;
    out << "tile group sizes cover " << next << " tiles, NumTiles is "
        << geometry.num_tiles;
    return fail(error, out.str());
  }
  *groups = std::move(result);
  return true;
}

bool validate_frame_tile_groups(const FrameTileGroupPlan& plan,
                                std::string* error) {
  if (!validate_tile_group_geometry(plan.geometry, error)) return false;
  if (plan.groups.empty()) {
    return fail(error, "a frame needs at least one tile group");
  }

  if (plan.geometry.bru_inactive) {
    if (plan.groups.size() != 1) {
      return fail(error,
                  "a bru_inactive frame is one header-only tile group; "
                  "this writer does not split it");
    }
    if (plan.frame_header.empty()) {
      return fail(error, "the first tile group must carry the frame header");
    }
    if (!plan.groups[0].bru_tile_active.empty()) {
      return fail(error,
                  "bru_tile_active is not coded for a bru_inactive frame");
    }
    return true;
  }

  if (plan.frame_header.empty()) {
    return fail(error, "the first tile group must carry the frame header");
  }
  if (plan.groups[0].tg_start != 0) {
    return fail(error,
                "the first tile group must start at TileNum 0 (§6.18)");
  }
  int expected = 0;
  for (std::size_t i = 0; i < plan.groups.size(); ++i) {
    const TileGroupSpan& span = plan.groups[i];
    if (!validate_span(plan.geometry, span, error)) {
      if (error != nullptr) *error = group_label(i) + ": " + *error;
      return false;
    }
    if (span.tg_start != expected) {
      std::ostringstream out;
      out << group_label(i) << " starts at tile " << span.tg_start
          << ", expected " << expected;
      if (span.tg_start < expected) {
        out << " (overlap or reordering)";
      } else {
        out << " (gap)";
      }
      return fail(error, out.str());
    }
    if (i > 0 && span.frame_header_present != plan.copy_frame_header) {
      return fail(error,
                  group_label(i) +
                      ": frame_header_present_flag must be " +
                      (plan.copy_frame_header ? "1 (copy)" : "0 (shared)"));
    }
    expected = span.tg_end + 1;
  }
  if (expected != plan.geometry.num_tiles) {
    std::ostringstream out;
    out << "the last tile group ends at tile " << (expected - 1)
        << "; the last group must have tg_end == NumTiles - 1 ("
        << (plan.geometry.num_tiles - 1) << ")";
    return fail(error, out.str());
  }
  return true;
}

bool write_tile_group(const TileGroupGeometry& geometry,
                      const FrameHeaderBits& frame_header,
                      const TileGroupWriteParams& params,
                      std::span<const std::vector<std::uint8_t>> tile_payloads,
                      std::vector<std::uint8_t>* out, std::string* error) {
  if (out == nullptr) return fail(error, "null tile group output");
  if (!validate_tile_group_geometry(geometry, error)) return false;

  const bool present =
      params.is_first_tile_group || params.frame_header_present;
  if (present && frame_header.empty()) {
    return fail(error, "frame_header_present_flag = 1 with no header bits");
  }

  BitWriter writer;
  writer.put_bit(params.is_first_tile_group);  // is_first_tile_group
  if (!params.is_first_tile_group) {
    writer.put_bit(params.frame_header_present);  // frame_header_present_flag
  }
  if (present) put_frame_header_bits(writer, frame_header);

  if (geometry.bru_inactive) {
    // §5.19: trailing_bits(sz * 8 - headerBits), then return.
    writer.put_bit(true);  // trailing_one_bit
    writer.align_to_byte();
    *out = writer.take_bytes();
    return true;
  }

  if (!validate_span(geometry, params.span, error)) return false;
  if (tile_payloads.size() != static_cast<std::size_t>(geometry.num_tiles)) {
    std::ostringstream text;
    text << "expected " << geometry.num_tiles
         << " raster tile payloads, got " << tile_payloads.size();
    return fail(error, text.str());
  }

  const int tile_bits = geometry.tile_bits();
  const bool covers_frame =
      params.span.tg_start == 0 && params.span.tg_end == geometry.num_tiles - 1;
  if (geometry.num_tiles > 1) {
    const bool present_range = !covers_frame;
    writer.put_bit(present_range);  // tile_start_and_end_present_flag
    if (present_range) {
      writer.put_bits(static_cast<std::uint64_t>(params.span.tg_start),
                      static_cast<unsigned>(tile_bits));
      writer.put_bits(static_cast<std::uint64_t>(params.span.tg_end),
                      static_cast<unsigned>(tile_bits));
    }
  }
  if (geometry.use_bru && geometry.num_tiles > 1) {
    for (int tile = params.span.tg_start; tile <= params.span.tg_end; ++tile) {
      const std::size_t index = static_cast<std::size_t>(tile - params.span.tg_start);
      writer.put_bit(params.span.bru_tile_active[index] != 0U);
    }
  }
  writer.align_to_byte();

  for (int tile = params.span.tg_start; tile <= params.span.tg_end; ++tile) {
    const std::vector<std::uint8_t>& payload =
        tile_payloads[static_cast<std::size_t>(tile)];
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
      return fail(error, "tile payload exceeds 4 GiB");
    }
    const std::uint32_t size = static_cast<std::uint32_t>(payload.size());
    if (!validate_payload_size(size, geometry.tile_size_bytes,
                               /*require_capacity=*/tile != params.span.tg_end,
                               error)) {
      std::ostringstream text;
      text << "tile " << tile << ": " << (error != nullptr ? *error : "");
      return fail(error, text.str());
    }
    if (tile != params.span.tg_end) {
      const std::uint32_t minus_one = size - 1U;
      for (int b = 0; b < geometry.tile_size_bytes; ++b) {
        writer.put_bits(
            (minus_one >> (8U * static_cast<unsigned>(b))) & 0xFFU, 8U);
      }
    }
    for (std::uint8_t byte : payload) writer.put_bits(byte, 8U);
  }

  *out = writer.take_bytes();
  return true;
}

bool write_frame_tile_groups(
    const FrameTileGroupPlan& plan,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    std::vector<std::vector<std::uint8_t>>* out, std::string* error) {
  if (out == nullptr) return fail(error, "null tile group output");
  if (!validate_frame_tile_groups(plan, error)) return false;

  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(plan.groups.size());
  for (std::size_t i = 0; i < plan.groups.size(); ++i) {
    TileGroupWriteParams params;
    params.is_first_tile_group = i == 0;
    params.frame_header_present = plan.groups[i].frame_header_present;
    params.span = plan.groups[i];
    std::vector<std::uint8_t> payload;
    if (!write_tile_group(plan.geometry, plan.frame_header, params,
                          tile_payloads, &payload, error)) {
      if (error != nullptr) *error = group_label(i) + ": " + *error;
      return false;
    }
    result.push_back(std::move(payload));
  }
  *out = std::move(result);
  return true;
}

bool read_tile_group(std::span<const std::uint8_t> payload,
                     const TileGroupGeometry& geometry,
                     std::size_t num_frame_header_bits, ParsedTileGroup* out,
                     std::string* error) {
  if (out == nullptr) return fail(error, "null parsed tile group");
  if (!validate_tile_group_geometry(geometry, error)) return false;
  if (payload.empty()) return fail(error, "empty tile group OBU payload");

  ParsedTileGroup parsed;
  PayloadBitReader reader(payload);

  if (!reader.get_bit(&parsed.is_first_tile_group)) {
    return fail(error, "truncated is_first_tile_group");
  }
  if (parsed.is_first_tile_group) {
    parsed.frame_header_present = true;  // inferred by §5.19
  } else if (!reader.get_bit(&parsed.frame_header_present)) {
    return fail(error, "truncated frame_header_present_flag");
  }

  if (parsed.frame_header_present) {
    if (num_frame_header_bits == 0) {
      return fail(error,
                  "frame header is present but NumFrameHeaderBits is 0");
    }
    std::vector<std::uint8_t> header((num_frame_header_bits + 7U) / 8U, 0);
    for (std::size_t i = 0; i < num_frame_header_bits; ++i) {
      bool bit = false;
      if (!reader.get_bit(&bit)) {
        return fail(error, "truncated frame_header/header_bit[]");
      }
      if (bit) {
        header[i >> 3U] |= static_cast<std::uint8_t>(1U << (7U - (i & 7U)));
      }
    }
    parsed.frame_header.bytes = std::move(header);
    parsed.frame_header.bit_count = num_frame_header_bits;
    parsed.frame_header_is_copy = !parsed.is_first_tile_group;
  }

  if (geometry.bru_inactive) {
    parsed.bru_inactive = true;
    // headerBits is measured before trailing_bits() (§5.19).
    parsed.header_bits = reader.bit_position();
    // trailing_bits(sz * 8 - headerBits): one 1 bit, then zero bits.
    bool one = false;
    if (!reader.get_bit(&one) || !one) {
      return fail(error, "bru_inactive trailing_bits() is missing its one bit");
    }
    while (!reader.at_end()) {
      bool bit = false;
      if (!reader.get_bit(&bit) || bit) {
        return fail(error,
                    "bru_inactive trailing_bits() contains a non-zero bit");
      }
    }
    parsed.header_bytes = payload.size();
    parsed.tg_start = 0;
    parsed.tg_end = geometry.num_tiles - 1;
    *out = std::move(parsed);
    return true;
  }

  if (geometry.num_tiles > 1) {
    if (!reader.get_bit(&parsed.tile_start_and_end_present)) {
      return fail(error, "truncated tile_start_and_end_present_flag");
    }
    const unsigned tile_bits = static_cast<unsigned>(geometry.tile_bits());
    if (parsed.tile_start_and_end_present) {
      std::uint64_t start = 0;
      std::uint64_t end = 0;
      if (!reader.get_bits(tile_bits, &start) ||
          !reader.get_bits(tile_bits, &end)) {
        return fail(error, "truncated tg_start/tg_end");
      }
      parsed.tg_start = static_cast<int>(start);
      parsed.tg_end = static_cast<int>(end);
    } else {
      parsed.tg_start = 0;
      parsed.tg_end = geometry.num_tiles - 1;
    }
  } else {
    parsed.tg_start = 0;
    parsed.tg_end = 0;
  }
  if (parsed.tg_end < parsed.tg_start) {
    return fail(error, "tg_end is less than tg_start (§6.18)");
  }
  if (parsed.tg_end >= geometry.num_tiles) {
    return fail(error, "tg_end is outside NumTiles");
  }

  if (geometry.use_bru && geometry.num_tiles > 1) {
    parsed.bru_tile_active.reserve(
        static_cast<std::size_t>(parsed.tg_end - parsed.tg_start + 1));
    for (int tile = parsed.tg_start; tile <= parsed.tg_end; ++tile) {
      bool active = false;
      if (!reader.get_bit(&active)) {
        return fail(error, "truncated bru_tile_active");
      }
      parsed.bru_tile_active.push_back(active ? 1U : 0U);
    }
  }

  if (!reader.align_to_byte_zero(error)) return false;
  parsed.header_bits = reader.bit_position();
  parsed.header_bytes = reader.bit_position() / 8U;
  if (parsed.header_bytes > payload.size()) {
    return fail(error, "tile group header is larger than the payload");
  }

  std::size_t cursor = parsed.header_bytes;
  for (int tile = parsed.tg_start; tile <= parsed.tg_end; ++tile) {
    std::uint32_t size = 0;
    if (tile != parsed.tg_end) {
      if (geometry.tile_size_bytes < 1 || geometry.tile_size_bytes > 4) {
        return fail(error, "a multi-tile group needs TileSizeBytes in 1..4");
      }
      if (cursor + static_cast<std::size_t>(geometry.tile_size_bytes) >
          payload.size()) {
        return fail(error, "truncated tile_size_minus_1");
      }
      const std::uint32_t minus_one =
          read_le(payload, cursor, geometry.tile_size_bytes);
      cursor += static_cast<std::size_t>(geometry.tile_size_bytes);
      size = minus_one + 1U;
      if (!validate_payload_size(size, geometry.tile_size_bytes,
                                 /*require_capacity=*/true, error)) {
        return false;
      }
      if (cursor + size > payload.size()) {
        return fail(error, "tile_size_minus_1 runs past the OBU payload");
      }
    } else {
      // §6.19.1: the last tile's size is the remaining OBU bytes.
      const std::size_t remaining = payload.size() - cursor;
      if (remaining > std::numeric_limits<std::uint32_t>::max()) {
        return fail(error, "last tile payload exceeds 4 GiB");
      }
      size = static_cast<std::uint32_t>(remaining);
      if (!validate_payload_size(size, geometry.tile_size_bytes,
                                 /*require_capacity=*/false, error)) {
        return false;
      }
    }
    parsed.tile_bytes.push_back(size);
    parsed.tile_offsets.push_back(cursor);
    cursor += size;
  }
  if (cursor != payload.size()) {
    return fail(error, "tile payloads do not consume the whole OBU payload");
  }
  *out = std::move(parsed);
  return true;
}

bool read_tile_group_sequence(
    std::span<const std::vector<std::uint8_t>> payloads,
    const TileGroupGeometry& geometry, std::size_t num_frame_header_bits,
    TileGroupSequence* out, std::string* error) {
  if (out == nullptr) return fail(error, "null tile group sequence output");
  if (!validate_tile_group_geometry(geometry, error)) return false;
  if (payloads.empty()) return fail(error, "a frame has no tile group OBUs");

  TileGroupSequence result;
  result.num_tiles = geometry.num_tiles;
  result.bru_inactive = geometry.bru_inactive;

  if (geometry.bru_inactive) {
    if (payloads.size() != 1) {
      return fail(error,
                  "a bru_inactive frame is one header-only tile group; "
                  "multiple groups are not representable");
    }
  }

  result.groups.reserve(payloads.size());
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    ParsedTileGroup parsed;
    if (!read_tile_group(payloads[i], geometry, num_frame_header_bits, &parsed,
                         error)) {
      if (error != nullptr) *error = group_label(i) + ": " + *error;
      return false;
    }
    if (geometry.bru_inactive && !parsed.bru_inactive) {
      return fail(error, "bru_inactive frame did not take the early-return path");
    }
    result.groups.push_back(std::move(parsed));
  }

  if (geometry.bru_inactive) {
    *out = std::move(result);
    return true;
  }

  // §6.18 SeenFrameHeader != is_first_tile_group. SeenFrameHeader becomes 1
  // after the first group and is cleared by decode_frame_wrapup() after the
  // last group, so exactly one group may say it is first, and it must be the
  // first payload.
  if (!result.groups[0].is_first_tile_group) {
    return fail(error,
                "the first tile group OBU has is_first_tile_group = 0 "
                "(SeenFrameHeader mismatch)");
  }
  for (std::size_t i = 1; i < result.groups.size(); ++i) {
    if (result.groups[i].is_first_tile_group) {
      std::ostringstream text;
      text << group_label(i) << " sets is_first_tile_group = 1 after the first "
            "group (SeenFrameHeader mismatch)";
      return fail(error, text.str());
    }
  }

  if (result.groups[0].tg_start != 0) {
    std::ostringstream text;
    text << "the first tile group starts at tile " << result.groups[0].tg_start
         << "; it must start at TileNum 0 (§6.18)";
    return fail(error, text.str());
  }

  int expected = 0;
  for (std::size_t i = 0; i < result.groups.size(); ++i) {
    const ParsedTileGroup& group = result.groups[i];
    if (group.tg_start != expected) {
      std::ostringstream text;
      text << group_label(i) << " starts at tile " << group.tg_start
           << ", expected " << expected;
      if (group.tg_start < expected) {
        text << " (overlap or reordering)";
      } else {
        text << " (gap)";
      }
      return fail(error, text.str());
    }
    expected = group.tg_end + 1;
  }
  if (expected != geometry.num_tiles) {
    std::ostringstream text;
    text << "the last tile group ends at tile " << (expected - 1)
         << "; the last group must have tg_end == NumTiles - 1 ("
         << (geometry.num_tiles - 1) << ")";
    return fail(error, text.str());
  }

  const FrameHeaderBits& canonical = result.groups[0].frame_header;
  for (std::size_t i = 1; i < result.groups.size(); ++i) {
    const ParsedTileGroup& group = result.groups[i];
    if (!group.frame_header_present) {
      ++result.groups_with_shared_header;
      continue;
    }
    ++result.groups_with_header_copy;
    if (group.frame_header.bit_count != canonical.bit_count) {
      return fail(error, "frame_header_copy has a different bit count");
    }
    if (group.frame_header.bytes != canonical.bytes) {
      std::size_t first_diff = 0;
      while (first_diff < canonical.bit_count &&
             frame_header_bit(group.frame_header, first_diff,
                              nullptr) ==
                 frame_header_bit(canonical, first_diff, nullptr)) {
        ++first_diff;
      }
      std::ostringstream text;
      text << "frame_header_copy in " << group_label(i)
           << " differs from the first tile group's frame header at bit "
           << first_diff << " (§6.17.1)";
      return fail(error, text.str());
    }
  }

  *out = std::move(result);
  return true;
}

}  // namespace av2enc::m2::tile_groups
