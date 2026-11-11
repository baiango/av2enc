// Intra-block-copy block-vector stack and reference-border validity.
//
// This is the M1 integration layer around src/encoder/ibc/ibc_core.h.  It
// implements the ordered candidate construction from §7.12.2 (scan points,
// scan col, then add_to_ref_bv) for use_intrabc = 1, with the §6.19.7.12
// validity check from the IBC core.
//
// The existing ibc::build_stack() exposes only four named neighbours.  This
// adapter keeps the decoder's per-MI block map instead, so the scan order and
// the superblock-border column adjustment are exact, and a later block can use
// any already-decoded IBC neighbour as RefMvIdx 0.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "encoder/ibc/ibc.h"

namespace av2enc::m1::ibc_stream {

using ibc::BlockVector;
using ibc::FrameView;
using ibc::IbcBufferState;
using ibc::IbcConfig;

// §3 MV range, in 1/8 luma samples.  mv_clamp_to_integer() uses the +1/-
// endpoints; a coded BV is required to stay inside the open interval.
inline constexpr int kMvLow = -65536;
inline constexpr int kMvUp = 65536;

bool bv_in_range(const BlockVector& bv) noexcept;

// Per-4x4 luma record of a decoded block.  `base_*` is the top-left MI of the
// block containing the sample, used by scan_point_if_valid().
struct BvMiRecord {
  bool coded = false;
  bool ibc = false;
  BlockVector bv{};
  int base_row = -1;
  int base_col = -1;
};

class BvNeighbourMap {
 public:
  void reset(int mi_cols, int mi_rows);

  // Mark a whole decoded block.  `ibc` is IsInters && RefFrame[0]==INTRA_FRAME.
  void mark_block(int mi_row, int mi_col, int bw, int bh, bool ibc,
                  BlockVector bv) noexcept;

  bool inside(int mi_row, int mi_col) const noexcept;
  const BvMiRecord* at(int mi_row, int mi_col) const noexcept;
  int mi_cols() const noexcept { return mi_cols_; }
  int mi_rows() const noexcept { return mi_rows_; }

 private:
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  std::vector<BvMiRecord> records_;
};

struct BvStackEntry {
  BlockVector bv{};
  bool neighbour = false;  // inserted by a §7.12.2 scan, not add_to_ref_bv
  bool valid = false;      // §6.19.7.12 is_mv_valid for the current block
  int mi_row = -1;
  int mi_col = -1;
};

struct BvStack {
  std::array<BvStackEntry, 8> entries{};
  int count = 0;
};

int bv_stack_capacity(int max_bvp_drl_bits_minus_1) noexcept;

// Build RefStackMv for an IBC block in §7.12.2 order.  `map` must contain all
// previously decoded blocks; the current block must not be marked yet.
BvStack build_bv_stack(const FrameView& frame, const IbcConfig& config,
                       const IbcBufferState& state, int mi_row, int mi_col,
                       int bsize, const BvNeighbourMap& map);

int find_bv_index(const BvStack& stack, BlockVector bv) noexcept;

// §6.19.5.4 is_offset_mv_valid(dx, dy).  `dx`/`dy` are integer luma-sample
// offsets; the spec forms offsetMv = BlockMv + (dy*8, dx*8) and runs
// is_mv_valid on it.  For morph_pred the conformance call is (-1, -1).
bool is_bv_offset_valid(const FrameView& frame, const IbcConfig& config,
                        const IbcBufferState& state, int mi_row, int mi_col,
                        int bsize, BlockVector bv, int dx, int dy,
                        std::string* reason = nullptr);

// The ±1 reference border conformance rule: a source block whose extra
// top-left reference pixel lies outside the tile is rejected.  Integer BVs are
// not exempt (dav2d --strict 1 enforces this even without morph_pred).
bool is_bv_border_valid(const FrameView& frame, const IbcConfig& config,
                        const IbcBufferState& state, int mi_row, int mi_col,
                        int bsize, BlockVector bv, std::string* reason = nullptr);

// Full encoder-side gate: range, §6.19.7.12 geometry/local/global, and ±1
// reference border.
bool is_bv_encodable(const FrameView& frame, const IbcConfig& config,
                     const IbcBufferState& state, int mi_row, int mi_col,
                     int bsize, BlockVector bv, std::string* reason = nullptr);

// Maintain the IBC ring and the neighbour map from one decoded block.  Call
// update_buffers() before mode info (§5.20.5.1) and mark_coded()/mark_block()
// after reconstruction.
void begin_ibc_block(IbcBufferState& state, int mi_row, int mi_col);
void finish_ibc_block(IbcBufferState& state, BvNeighbourMap& map, int mi_row,
                      int mi_col, int bsize, bool ibc, BlockVector bv);

}  // namespace av2enc::m1::ibc_stream
