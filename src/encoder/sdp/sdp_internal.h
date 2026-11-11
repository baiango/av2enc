// Shared inline helpers for the SDP reference implementation.
//
// Not a public header: it only keeps the geometry predicates used by sdp.cpp,
// sdp_validate.cpp and sdp_syntax.cpp in one place.

#pragma once

#include "encoder/sdp/sdp.h"

namespace av2enc::sdp::detail {

constexpr int kSentinelLog2 = 6;  // BLOCK_256X256: never "smaller than current"

inline bool geometry_ready(const partition::FrameGeometry& geometry) {
  return geometry.width > 0 && geometry.height > 0 &&
         (geometry.sb_size == 64 || geometry.sb_size == 128) &&
         geometry.max_pb_aspect_ratio >= 1;
}

inline int num_planes(const partition::FrameGeometry& geometry) {
  return geometry.has_chroma ? 3 : 1;
}

inline bool is_three_way(PartitionType p) {
  return p == PartitionType::kHorz3 || p == PartitionType::kVert3;
}

inline bool is_uneven_4way(PartitionType p) {
  return p == PartitionType::kHorz4A || p == PartitionType::kHorz4B ||
         p == PartitionType::kVert4A || p == PartitionType::kVert4B;
}

inline bool rect_is_horizontal(PartitionType p) {
  return p == PartitionType::kHorz || p == PartitionType::kHorz3 ||
         p == PartitionType::kHorz4A || p == PartitionType::kHorz4B;
}

inline bool rect_is_vertical(PartitionType p) {
  return p == PartitionType::kVert || p == PartitionType::kVert3 ||
         p == PartitionType::kVert4A || p == PartitionType::kVert4B;
}

inline PartitionType rect_part_table(bool do_ext, bool do_uneven, int type,
                                     bool horizontal) {
  if (!do_ext) {
    return horizontal ? PartitionType::kHorz : PartitionType::kVert;
  }
  if (!do_uneven) {
    return horizontal ? PartitionType::kHorz3 : PartitionType::kVert3;
  }
  if (horizontal) {
    return type == 0 ? PartitionType::kHorz4A : PartitionType::kHorz4B;
  }
  return type == 0 ? PartitionType::kVert4A : PartitionType::kVert4B;
}

inline int log2_or_sentinel(const partition::MiSizeGrid& grid, int mi_row,
                            int mi_col, bool width) {
  const partition::BlockSize size = grid.at(mi_row, mi_col);
  if (!partition::is_valid_block_size(size)) return kSentinelLog2;
  return width ? partition::mi_width_log2(size)
               : partition::mi_height_log2(size);
}

inline bool same_set(const PartitionSet& a, const PartitionSet& b) {
  if (a.count != b.count) return false;
  for (int i = 0; i < kPartitionCount; ++i) {
    if (a.allowed[i] != b.allowed[i]) return false;
  }
  return true;
}

inline bool set_allows(const PartitionSet& set, PartitionType p) {
  const int index = static_cast<int>(p);
  return index >= 0 && index < kPartitionCount && set.allowed[index];
}

}  // namespace av2enc::sdp::detail
