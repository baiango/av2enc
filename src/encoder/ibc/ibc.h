// AV2 intra block copy (IBC) — CPU reference and encoder-side search.
//
// Spec-only: AV2 v1.0.0 §5.20.5.3/.4, §6.19.5.4, §6.19.7.12, §7.13.3.17,
// §7.13.3.18.  No AVM/SVT/clave encoder code.
//
// What this module owns
// ---------------------
//   * block-vector representation (1/8 luma samples) and plane scaling;
//   * the §5.20.5.1 local IBC buffer (four 64x64 regions) and IBCCoded grid;
//   * the §6.19.7.12 is_mv_valid rules: tile bounds, local buffer, global
//     INTRABC_DELAY_SB64 window and wavefront, plus the chroma top-left
//     override;
//   * the IBC candidate stack defaults/neighbour insert order and RefMvIdx
//     selection used by `intrabc_mode = 1` (NEARMV, no MVD);
//   * block matching in the reconstructed frame and exact block copy for
//     luma + 4:2:0 chroma;
//   * transform-skip semantics: an IBC block with skip_flag=1 copies the
//     source exactly; there is no residual syntax.  A residual path is also
//     exposed for completeness (`copy_then_residual`).
//
// Not here: entropy syntax (the gate writer in `palette/scc_stream.*` emits
// it), NEWMV/MVD syntax, morph_pred, 128x128 superblocks, 4:2:2/4:4:4.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "encoder/ibc/ibc_core.h"
#include "encoder/partition/partition.h"

namespace av2enc::ibc {

using partition::BlockSize;

struct PlaneView {
  const std::uint8_t* data = nullptr;
  int stride = 0;
  int width = 0;
  int height = 0;
};

struct FrameView {
  PlaneView y;
  PlaneView u;
  PlaneView v;
  bool has_chroma = true;
  int sub_x = 1;
  int sub_y = 1;
};

// Block vector in 1/8 luma samples.  Positive row = down, positive col = right.
struct BlockVector {
  int row = 0;
  int col = 0;

  friend bool operator==(const BlockVector& a, const BlockVector& b) {
    return a.row == b.row && a.col == b.col;
  }
  friend bool operator!=(const BlockVector& a, const BlockVector& b) {
    return !(a == b);
  }
};

struct IbcConfig {
  int sb_size = 64;
  bool allow_global = true;
  bool allow_local = true;
  int delay_sb64 = kIntrabcDelaySb64;
  int delay_pixels = kIntrabcDelayPixels;
  bool allow_extra_ibc_range = false;  // 0 for 64x64 SBs
  // §5.20.5.4: max_bvp_drl_bits_minus_1 + 1 coded DRL bits; the stack holds at
  // most max_bvp_drl_bits_minus_1 + 2 entries.
  int max_bvp_drl_bits_minus_1 = 0;
  // Minimum SAD for an encoder match to be accepted.
  int64_t max_sad = 0;  // 0 = accept the best candidate
};

int stack_capacity(const IbcConfig& config) noexcept;

struct IbcBufferState {
  int mi_cols = 0;
  int mi_rows = 0;
  int cur_buf_row = 0;
  int cur_buf_col = 0;
  std::array<int, kIbcNumBuffers> entry_row{};
  std::array<int, kIbcNumBuffers> entry_col{};
  std::array<bool, kIbcNumBuffers> entry_valid{};
  std::vector<std::uint8_t> coded;  // mi_rows x mi_cols

  void reset(int cols, int rows);
  // §5.20.5.1 update_ibc_buffers().
  void update_buffers(int mi_row, int mi_col) noexcept;
  // Mark an entire decoded block IBCCoded (called after reconstruction).
  void mark_coded(int mi_row, int mi_col, int bw, int bh) noexcept;
};

// §6.19.7.12.  `reason` receives a short diagnostic on failure.
bool is_bv_valid(const FrameView& frame, const IbcConfig& config,
                 const IbcBufferState& state, int mi_row, int mi_col,
                 int bsize, BlockVector bv, std::string* reason = nullptr);

// ---------------------------------------------------------------------------
// Candidate stack (find_mv_stack for IBC, integer-pel NEARMV path)
// ---------------------------------------------------------------------------
struct BvCandidate {
  BlockVector bv;
  bool neighbour = false;
  bool valid = false;  // set by build_stack()
};

struct BvStack {
  std::array<BvCandidate, 8> candidates{};
  int count = 0;
};

// Defaults from §7.12.2.20 in the spec's order:
//   (0, -SbHeight), (-SbWidth - 256, 0), (0, -BlockHeight), (-BlockWidth, 0)
// followed by neighbour BVs (above, left, below-left, above-right) that are
// already IBC and match RefFrame == INTRA_FRAME.  Entries are capped at
// stack_capacity().  `above_bv` etc. are optional (nullptr = unavailable).
struct NeighbourBvs {
  const BlockVector* above = nullptr;
  const BlockVector* left = nullptr;
  const BlockVector* below_left = nullptr;
  const BlockVector* above_right = nullptr;
};

BvStack build_stack(const FrameView& frame, const IbcConfig& config,
                    const IbcBufferState& state, int mi_row, int mi_col,
                    int bsize, const NeighbourBvs& neighbours);

// Returns the stack index whose BV equals `bv`, or -1.
int find_stack_index(const BvStack& stack, BlockVector bv) noexcept;

// ---------------------------------------------------------------------------
// Copy / residual
// ---------------------------------------------------------------------------
// Copy the source block into the destination planes.  `visible_w/h` clamp the
// current block to the frame edge (<= block size).  Returns false if the
// source is outside the plane.
bool copy_block(const FrameView& frame, int mi_row, int mi_col, int bsize,
                BlockVector bv, std::uint8_t* dst_y, int dst_stride_y,
                std::uint8_t* dst_u, int dst_stride_u, std::uint8_t* dst_v,
                int dst_stride_v, int visible_w, int visible_h);

// Transform-skip residual for an IBC block: residual = source - copy, over the
// visible area.  For skip_flag=1 the caller simply does not code it.
void copy_then_residual(const FrameView& frame, int mi_row, int mi_col,
                        int bsize, BlockVector bv, std::int16_t* residual_y,
                        int res_stride_y, std::int16_t* residual_u,
                        int res_stride_u, std::int16_t* residual_v,
                        int res_stride_v, int visible_w, int visible_h);

// The decoder-side prediction after `copy_block`: returns a buffer with the
// copied luma (and chroma if requested), useful for tests/PSNR.
struct CopyResult {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};
bool predict_copy(const FrameView& frame, int mi_row, int mi_col, int bsize,
                  BlockVector bv, int visible_w, int visible_h,
                  CopyResult& out, std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Encoder search
// ---------------------------------------------------------------------------
struct Match {
  bool found = false;
  BlockVector bv;
  std::int64_t sad = 0;
  int stack_index = -1;  // >=0 when the BV is already a NEARMV candidate
};

struct SearchOptions {
  int step = 4;             // candidate grid in luma samples
  bool stack_only = false;  // NEARMV only (no MVD syntax this module)
  bool compare_chroma = false;
  std::int64_t max_sad = 0;  // 0 = accept best
};

// Brute-force search over all previously reconstructed, is_mv_valid positions.
// The source is the reconstructed frame; callers must have reconstructed all
// earlier blocks (raster order) before calling.
Match search_block(const FrameView& frame, const IbcConfig& config,
                   const IbcBufferState& state, int mi_row, int mi_col,
                   int bsize, const SearchOptions& options,
                   const NeighbourBvs* neighbours = nullptr);

// ---------------------------------------------------------------------------
// Quality helpers
// ---------------------------------------------------------------------------
std::int64_t block_sse(const std::uint8_t* a, int a_stride,
                       const std::uint8_t* b, int b_stride, int w, int h);

struct SyntheticScreen {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  FrameView view() const;
};

// Deterministic screen-content generator: flat UI regions, text-like stripes,
// an icon block and a repeated 64x32 tile row (IBC-friendly).
SyntheticScreen make_synthetic_screen(int width, int height);

}  // namespace av2enc::ibc
