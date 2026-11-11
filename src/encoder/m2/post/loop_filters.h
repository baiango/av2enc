// Post-reconstruction loop-filter chain for M2: deblock -> CDEF -> CCSO.
//
// This is the integration-ready CPU composition of the three existing filter
// components.  It does not replace them and it does not write bitstream syntax;
// the matching frame-header flag writers are in tiled_frame.h / this header.
//
// Normative order (AV2 v1.0.0 §7.17, §7.18, §7.19; docs/negative-results.md):
//
//   reconstruct -> deblock -> CDEF -> CCSO -> (loop restoration, if any)
//
// Deblock works in place on the reconstruction.  CDEF reads the *unfiltered*
// (i.e. post-deblock) samples and writes a separate destination.  CCSO reads
// the post-deblock buffer plus the post-CDEF buffer and filters the post-CDEF
// buffer in place.  Running CDEF before deblock, or CCSO before CDEF, is a
// reference-frame bug; run_m2_post_filters() enforces the order and reports it.
//
// Only 8-bit 4:2:0 is supported: all three underlying CPU filters reject other
// depths/subsampling.  Use bit_depth_checklist.h before claiming 10-bit.

#pragma once

#include "encoder/filters/cdef/cdef.h"
#include "encoder/filters/deblock/deblock.h"
#include "encoder/filters/restoration/restoration.h"
#include "encoder/m2/m2_frame.h"
#include "encoder/tiles/tile_layout.h"
#include "bitstream/bitwriter.h"

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::m2::post {

enum class M2PostFilterStage : std::uint8_t {
  kDeblock = 0,
  kCdef = 1,
  kCcso = 2,
};

const char* post_filter_stage_name(M2PostFilterStage stage) noexcept;

struct M2PostFilterConfig {
  bool enable_deblock = false;
  deblock::DeblockOptions deblock{};

  bool enable_cdef = false;
  cdef::CdefOptions cdef{};

  bool enable_ccso = false;
  restoration::CcsoFrameParams ccso{};

  // §5.4.10 disable_loopfilters_across_tiles.  When true the tile starts from
  // set_tile_layout() are used to clip every filter at interior tile edges.
  bool disable_loopfilters_across_tiles = false;
};

struct M2PostFilterReport {
  int bit_depth = 8;
  bool deblock_ran = false;
  bool cdef_ran = false;
  bool ccso_ran = false;
  std::vector<M2PostFilterStage> order;
};

// Owns the MI-aligned reconstruction, the per-filter metadata grids and the
// intermediate buffers.  The default metadata is intentionally conservative
// and must be replaced from the M2 partition/skip/CCSO decisions for a real
// encode:
//   * deblock: every 4x4 cell is its own intra block, tx 4x4, qindex =
//     base_q_idx.  This filters every 4x4 edge; use set_deblock_block() to
//     describe the real partition tree.
//   * CDEF: skip = 0, lossless = 0, cdef_idx = 0.
//   * CCSO: every CcsoBlks entry = 0 (the neutral band).
class M2PostFilterFrame {
 public:
  M2PostFilterFrame() = default;

  // Copies an M2 reconstruction into MI-aligned planes and allocates metadata.
  bool init_from_recon(const M2ReconFrame& recon, int base_q_idx = 90,
                       bool frame_is_intra = true,
                       std::string* error = nullptr);

  // Tile-aware variant: also fills deblock/CDEF tile starts and the
  // restoration MiCol/MiRow start/end grids.
  bool set_tile_layout(const tiles::TileLayout& layout, std::string* error);

  // Describe one coding block in the deblock metadata of one plane.  `mi_row`
  // and `mi_col` are luma MI units; w4/h4 are Num_4x4_Blocks_Wide/High.
  bool set_deblock_block(int plane, int mi_row, int mi_col, int w4, int h4,
                         int sub_pu_w4, int sub_pu_h4, bool skip,
                         bool region_intra, int qindex,
                         deblock::TxSize tx_size = deblock::TxSize::k4x4);

  // Convenience: one rectangle on all three planes.
  void fill_deblock_rect(int mi_row, int mi_col, int w4, int h4, bool skip,
                         bool region_intra, int qindex);

  int mi_cols() const noexcept { return mi_cols_; }
  int mi_rows() const noexcept { return mi_rows_; }
  int bit_depth() const noexcept { return bit_depth_; }
  int base_q_idx() const noexcept { return base_q_idx_; }
  bool frame_is_intra() const noexcept { return frame_is_intra_; }

  std::uint8_t* plane_data(int plane) {
    return plane_[plane].data();
  }
  const std::uint8_t* plane_data(int plane) const {
    return plane_[plane].data();
  }
  int plane_stride(int plane) const { return stride_[plane]; }
  int plane_width(int plane) const;
  int plane_height(int plane) const;

  std::vector<std::uint8_t>& skip_grid() { return skip_; }
  const std::vector<std::uint8_t>& skip_grid() const { return skip_; }
  std::vector<std::uint8_t>& lossless_grid() { return lossless_; }
  std::vector<std::uint8_t>& lossless_chroma_grid() {
    return lossless_chroma_;
  }
  std::vector<std::int8_t>& cdef_idx_grid() { return cdef_idx_; }
  const std::vector<std::int8_t>& cdef_idx_grid() const { return cdef_idx_; }
  int n64_cols() const noexcept { return n64_cols_; }
  int n64_rows() const noexcept { return n64_rows_; }

  std::vector<std::uint8_t>& ccso_blks(int plane) {
    return ccso_blk_[plane];
  }
  const std::vector<std::uint8_t>& ccso_blks(int plane) const {
    return ccso_blk_[plane];
  }
  int ccso_blk_cols(int plane) const { return ccso_blk_cols_[plane]; }
  int ccso_blk_rows(int plane) const { return ccso_blk_rows_[plane]; }

  std::vector<deblock::DeblockMi>& deblock_mi(int plane) {
    return deblock_mi_[plane];
  }
  const std::vector<deblock::DeblockMi>& deblock_mi(int plane) const {
    return deblock_mi_[plane];
  }

  const std::vector<std::uint16_t>& tile_col_starts() const {
    return tile_col_starts_;
  }
  const std::vector<std::uint16_t>& tile_row_starts() const {
    return tile_row_starts_;
  }

  // Final output after the chain.  Valid after run_m2_post_filters(); when no
  // filter runs it is a copy of the input reconstruction.
  const std::vector<std::uint8_t>& output_plane(int plane) const {
    return output_[plane];
  }
  int output_stride(int plane) const { return stride_[plane]; }

  void set_bit_depth(int depth) { bit_depth_ = depth; }

 private:
  friend bool run_m2_post_filters(M2PostFilterFrame&,
                                  const M2PostFilterConfig&,
                                  M2PostFilterReport&, std::string*);

  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int sub_x_ = 1;
  int sub_y_ = 1;
  int bit_depth_ = 8;
  int base_q_idx_ = 90;
  bool frame_is_intra_ = true;
  int sb_size_ = 64;

  std::vector<std::uint8_t> plane_[3];
  int stride_[3] = {0, 0, 0};

  std::vector<deblock::DeblockMi> deblock_mi_[3];
  std::vector<std::uint8_t> skip_;
  std::vector<std::uint8_t> lossless_;
  std::vector<std::uint8_t> lossless_chroma_;
  std::vector<std::int8_t> cdef_idx_;
  int n64_cols_ = 0;
  int n64_rows_ = 0;

  std::vector<std::uint8_t> ccso_blk_[3];
  int ccso_blk_cols_[3] = {0, 0, 0};
  int ccso_blk_rows_[3] = {0, 0, 0};

  std::vector<std::uint16_t> tile_col_starts_;
  std::vector<std::uint16_t> tile_row_starts_;
  std::vector<std::uint16_t> mi_col_start_;
  std::vector<std::uint16_t> mi_col_end_;
  std::vector<std::uint16_t> mi_row_start_;
  std::vector<std::uint16_t> mi_row_end_;

  std::vector<std::uint8_t> cdef_out_[3];
  std::vector<std::uint8_t> output_[3];
};

// Runs the configured stages in the normative order.  `config.ccso.blks` and
// `blk_cols/rows` are filled from the frame's default grids when they are null;
// a caller-supplied grid is kept.  Returns false and fills `error` on a filter
// validation failure; no output is claimed in that case.
bool run_m2_post_filters(M2PostFilterFrame& frame,
                         const M2PostFilterConfig& config,
                         M2PostFilterReport& report,
                         std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Frame/sequence header flags (syntax only)
// ---------------------------------------------------------------------------
//
// write_m2_deblock_params() writes the exact §5.18.7.9 row.  The CDEF and CCSO
// rows are written by tiled_frame.cpp through the verified adapters in
// src/encoder/m1/filters_stream/ (write_cdef_params/write_ccso_params).
// Enabling CDEF or CCSO additionally requires the per-block tile symbols
// (read_cdef()/read_ccso()), which the current M2 core syntax writer does not
// emit; see README.md.

// Writes the deblocking_filter_params() row only (apply flags + df_delta_q
// bits).  The frame header writer places it before cdef_params().
std::size_t write_m2_deblock_params(
    BitWriter& writer, const bool apply_deblocking_filter[4],
    const bool df_delta_q_present[4], std::string* error);

}  // namespace av2enc::m2::post
