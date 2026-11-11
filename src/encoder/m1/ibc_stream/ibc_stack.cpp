#include "encoder/m1/ibc_stream/ibc_stack.h"

#include <algorithm>

#include "encoder/m1/m1_tables.h"

namespace av2enc::m1::ibc_stream {
namespace {

bool stack_contains(const BvStack& stack, BlockVector bv) {
  for (int i = 0; i < stack.count; ++i) {
    if (stack.entries[static_cast<std::size_t>(i)].bv == bv) return true;
  }
  return false;
}

void push_entry(BvStack& stack, int capacity, BlockVector bv, bool neighbour,
                int mi_row, int mi_col, bool dedupe) {
  if (stack.count >= capacity || stack.count >= 8) return;
  if (dedupe && stack_contains(stack, bv)) return;
  BvStackEntry& e = stack.entries[static_cast<std::size_t>(stack.count++)];
  e.bv = bv;
  e.neighbour = neighbour;
  e.valid = false;
  e.mi_row = mi_row;
  e.mi_col = mi_col;
}

int floor_div2(int v) {
  return v >= 0 ? v / 2 : -((-v + 1) / 2);
}

}  // namespace

bool bv_in_range(const BlockVector& bv) noexcept {
  return bv.row > kMvLow && bv.row < kMvUp && bv.col > kMvLow &&
         bv.col < kMvUp;
}

void BvNeighbourMap::reset(int mi_cols, int mi_rows) {
  mi_cols_ = std::max(0, mi_cols);
  mi_rows_ = std::max(0, mi_rows);
  records_.assign(
      static_cast<std::size_t>(mi_cols_) * static_cast<std::size_t>(mi_rows_),
      BvMiRecord{});
}

void BvNeighbourMap::mark_block(int mi_row, int mi_col, int bw, int bh,
                                bool ibc, BlockVector bv) noexcept {
  if (mi_cols_ <= 0 || mi_rows_ <= 0) return;
  const int w4 = std::max(1, bw / 4);
  const int h4 = std::max(1, bh / 4);
  for (int y = 0; y < h4; ++y) {
    const int r = mi_row + y;
    if (r < 0 || r >= mi_rows_) break;
    for (int x = 0; x < w4; ++x) {
      const int c = mi_col + x;
      if (c < 0 || c >= mi_cols_) break;
      BvMiRecord& rec =
          records_[static_cast<std::size_t>(r) *
                       static_cast<std::size_t>(mi_cols_) +
                   static_cast<std::size_t>(c)];
      rec.coded = true;
      rec.ibc = ibc;
      rec.bv = bv;
      rec.base_row = mi_row;
      rec.base_col = mi_col;
    }
  }
}

bool BvNeighbourMap::inside(int mi_row, int mi_col) const noexcept {
  return mi_row >= 0 && mi_col >= 0 && mi_row < mi_rows_ && mi_col < mi_cols_;
}

const BvMiRecord* BvNeighbourMap::at(int mi_row, int mi_col) const noexcept {
  if (!inside(mi_row, mi_col)) return nullptr;
  return &records_[static_cast<std::size_t>(mi_row) *
                       static_cast<std::size_t>(mi_cols_) +
                   static_cast<std::size_t>(mi_col)];
}

int bv_stack_capacity(int max_bvp_drl_bits_minus_1) noexcept {
  return std::min(6, std::max(1, max_bvp_drl_bits_minus_1 + 2));
}

BvStack build_bv_stack(const FrameView& frame, const IbcConfig& config,
                       const IbcBufferState& state, int mi_row, int mi_col,
                       int bsize, const BvNeighbourMap& map) {
  BvStack stack;
  const int capacity = bv_stack_capacity(config.max_bvp_drl_bits_minus_1);
  const auto size = static_cast<partition::BlockSize>(bsize);
  const int bw4 = std::max(1, partition::block_width(size) / 4);
  const int bh4 = std::max(1, partition::block_height(size) / 4);
  const int sb_mi = std::max(1, config.sb_size / 4);
  const bool is_sb_border = (mi_row & (sb_mi - 1)) == 0;

  const auto try_add_point = [&](int mv_row, int mv_col) {
    const BvMiRecord* rec = map.at(mv_row, mv_col);
    if (rec == nullptr || !rec->coded || !rec->ibc) return;
    push_entry(stack, capacity, rec->bv, true, mv_row, mv_col, true);
  };

  const auto scan_point = [&](int delta_row, int delta_col) {
    int mv_row = mi_row + delta_row;
    int mv_col = mi_col + delta_col;
    if (delta_row < 0 && is_sb_border) {
      mv_col = floor_div2(mv_col) * 2;
    }
    try_add_point(mv_row, mv_col);
  };

  const auto scan_point_if_valid = [&](int delta_row, int delta_col) {
    const int mv_row = mi_row + delta_row;
    const int mv_col = mi_col + delta_col;
    const int other_col = mi_col - 1;
    const BvMiRecord* rec = map.at(mv_row, mv_col);
    const BvMiRecord* other = map.at(mv_row, other_col);
    const int other_base = other != nullptr ? other->base_col : -1;
    if (rec != nullptr && rec->coded && rec->ibc &&
        rec->base_col != other_base) {
      push_entry(stack, capacity, rec->bv, true, mv_row, mv_col, true);
    }
  };

  // §7.12.2 Find MV stack process, single-list scan order.
  scan_point(bh4 - 1, -1);
  scan_point(-1, std::max(0, bw4 - 1 - (is_sb_border ? 1 : 0)));
  if (bh4 >= 2) scan_point(0, -1);
  if (bw4 >= (is_sb_border ? 4 : 2)) scan_point(-1, 0);
  if (bh4 <= 16) scan_point(bh4, -1);
  if (bw4 <= 16) {
    scan_point(-1, is_sb_border ? std::max(2, bw4) : bw4);
  }
  scan_point(-1, -1 - (is_sb_border ? 1 : 0));

  // §7.12.2.5 Scan col process (deltaCol = -3).
  {
    int delta_col = -3;
    if (bw4 == 1) delta_col += (mi_col & 1);
    scan_point_if_valid(bh4 - 1, delta_col);
    if (bh4 > 1) scan_point_if_valid(0, delta_col);
  }

  // §7.12.2.20 Extra search process, add_to_ref_bv().  Defaults are appended
  // without pruning, in the spec's order.
  const int sb_w = config.sb_size;
  const int sb_h = config.sb_size;
  const int block_w = partition::block_width(size);
  const int block_h = partition::block_height(size);
  push_entry(stack, capacity, BlockVector{-sb_h * 8, 0}, false, -1, -1, false);
  push_entry(stack, capacity,
             BlockVector{0, -(sb_w + config.delay_pixels) * 8}, false, -1, -1,
             false);
  push_entry(stack, capacity, BlockVector{-block_h * 8, 0}, false, -1, -1,
             false);
  push_entry(stack, capacity, BlockVector{0, -block_w * 8}, false, -1, -1,
             false);

  for (int i = 0; i < stack.count; ++i) {
    BvStackEntry& e = stack.entries[static_cast<std::size_t>(i)];
    e.valid = is_bv_valid(frame, config, state, mi_row, mi_col, bsize, e.bv,
                          nullptr);
  }
  return stack;
}

int find_bv_index(const BvStack& stack, BlockVector bv) noexcept {
  for (int i = 0; i < stack.count; ++i) {
    if (stack.entries[static_cast<std::size_t>(i)].bv == bv) return i;
  }
  return -1;
}

bool is_bv_offset_valid(const FrameView& frame, const IbcConfig& config,
                        const IbcBufferState& state, int mi_row, int mi_col,
                        int bsize, BlockVector bv, int dx, int dy,
                        std::string* reason) {
  const BlockVector offset{bv.row + dy * 8, bv.col + dx * 8};
  if (!bv_in_range(offset)) {
    if (reason != nullptr) *reason = "offset block vector is outside MV range";
    return false;
  }
  return is_bv_valid(frame, config, state, mi_row, mi_col, bsize, offset,
                     reason);
}

bool is_bv_border_valid(const FrameView& frame, const IbcConfig& config,
                        const IbcBufferState& state, int mi_row, int mi_col,
                        int bsize, BlockVector bv, std::string* reason) {
  // §6.19.5.4: is_offset_mv_valid(-1, -1) guarantees the extra reference
  // pixels fetched for morph prediction / block copy are inside the tile.
  return is_bv_offset_valid(frame, config, state, mi_row, mi_col, bsize, bv,
                            -1, -1, reason);
}

bool is_bv_encodable(const FrameView& frame, const IbcConfig& config,
                     const IbcBufferState& state, int mi_row, int mi_col,
                     int bsize, BlockVector bv, std::string* reason) {
  if (!bv_in_range(bv)) {
    if (reason != nullptr) *reason = "block vector is outside MV range";
    return false;
  }
  std::string why;
  if (!is_bv_valid(frame, config, state, mi_row, mi_col, bsize, bv, &why)) {
    if (reason != nullptr) *reason = why;
    return false;
  }
  if (!is_bv_border_valid(frame, config, state, mi_row, mi_col, bsize, bv,
                          &why)) {
    if (reason != nullptr) *reason = "±1 reference border: " + why;
    return false;
  }
  return true;
}

void begin_ibc_block(IbcBufferState& state, int mi_row, int mi_col) {
  state.update_buffers(mi_row, mi_col);
}

void finish_ibc_block(IbcBufferState& state, BvNeighbourMap& map, int mi_row,
                      int mi_col, int bsize, bool ibc, BlockVector bv) {
  const auto size = static_cast<partition::BlockSize>(bsize);
  const int bw = partition::block_width(size);
  const int bh = partition::block_height(size);
  state.mark_coded(mi_row, mi_col, bw, bh);
  map.mark_block(mi_row, mi_col, bw, bh, ibc, bv);
}

}  // namespace av2enc::m1::ibc_stream
