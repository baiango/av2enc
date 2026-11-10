#include "tile_group.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace av2enc::tiles {
namespace {
constexpr std::uint32_t kMinTileBytes = 2;  // init_symbol() reads up to 15 bits
}  // namespace

int tile_size_bytes_for(std::uint32_t largest_tile_bytes) {
  if (largest_tile_bytes < kMinTileBytes) {
    return 0;
  }
  for (int bytes = 1; bytes <= 4; ++bytes) {
    const std::uint64_t capacity = 1ull << (8 * bytes);
    if (static_cast<std::uint64_t>(largest_tile_bytes) <= capacity) {
      return bytes;
    }
  }
  return 0;
}

TileGroupPlan plan_single_tile_group(
    const TileLayout& layout, std::span<const std::uint32_t> tile_sizes,
    std::uint64_t frame_header_bits, bool is_first_tile_group) {
  TileGroupPlan plan;
  plan.is_first_tile_group = is_first_tile_group;
  plan.frame_header_bits = frame_header_bits;
  plan.tile_count = layout.tile_count();
  plan.tile_cols_log2 = layout.cols_log2;
  plan.tile_rows_log2 = layout.rows_log2;
  plan.tile_bits = layout.cols_log2 + layout.rows_log2;

  if (layout.empty()) {
    plan.error = "tile layout is empty";
    return plan;
  }
  if (tile_sizes.size() != layout.tiles.size()) {
    std::ostringstream out;
    out << "expected " << layout.tiles.size() << " tile sizes, got "
        << tile_sizes.size();
    plan.error = out.str();
    return plan;
  }
  if (!is_first_tile_group && frame_header_bits != 0) {
    plan.error =
        "partial non-first tile groups with a frame header are not modelled";
    return plan;
  }

  std::uint32_t largest = 0;
  std::uint32_t smallest = std::numeric_limits<std::uint32_t>::max();
  for (std::uint32_t size : tile_sizes) {
    if (size < kMinTileBytes) {
      plan.error = "every tile payload must be at least 2 bytes";
      return plan;
    }
    if (size > largest) {
      largest = size;
    }
    if (size < smallest) {
      smallest = size;
    }
  }
  plan.largest_tile_bytes = largest;
  plan.smallest_tile_bytes = smallest;
  plan.tile_size_bytes = tile_size_bytes_for(largest);
  if (plan.tile_size_bytes == 0) {
    plan.error = "tile size cannot be represented in 1..4 bytes";
    return plan;
  }

  // §5.19 header:
  //   is_first_tile_group                          f(1)
  //   if is_first: frame_header()                  frame_header_bits
  //   else:        frame_header_present_flag = 0   f(1)
  //   if NumTiles > 1: tile_start_and_end_present_flag = 0  f(1)
  //   byte_alignment()
  std::uint64_t bits = 1;
  if (is_first_tile_group) {
    bits += frame_header_bits;
  } else {
    bits += 1;
  }
  if (plan.tile_count > 1) {
    bits += 1;
  }
  const std::uint64_t alignment = (8 - (bits % 8)) % 8;
  plan.header_bits = bits + alignment;
  plan.header_bytes = plan.header_bits / 8;

  plan.size_field_count = plan.tile_count - 1;
  plan.size_field_bytes =
      static_cast<std::uint64_t>(plan.size_field_count) *
      static_cast<std::uint64_t>(plan.tile_size_bytes);

  plan.tile_payload_offsets.resize(tile_sizes.size());
  plan.tile_size_minus_1.resize(static_cast<std::size_t>(plan.size_field_count));
  std::uint64_t offset = 0;
  for (int i = 0; i < plan.tile_count; ++i) {
    plan.tile_payload_offsets[static_cast<std::size_t>(i)] = offset;
    const std::uint32_t size = tile_sizes[static_cast<std::size_t>(i)];
    if (i < plan.size_field_count) {
      plan.tile_size_minus_1[static_cast<std::size_t>(i)] = size - 1;
    }
    offset += static_cast<std::uint64_t>(size);
  }
  plan.payload_bytes = offset;
  plan.obu_payload_bytes =
      plan.header_bytes + plan.size_field_bytes + plan.payload_bytes;
  plan.valid = true;
  return plan;
}

}  // namespace av2enc::tiles
