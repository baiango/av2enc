// Minimal M1 still-picture stream with global intra block copy.
//
// The production M1 writer (src/encoder/m1/still_headers.cpp) hard-codes
// allow_screen_content_tools = allow_intrabc = 0.  This component emits an
// otherwise-M1 sequence header plus a frame header with:
//
//   allow_screen_content_tools = 1, force_integer_mv = 1,
//   allow_intrabc = 1, allow_global_intrabc = 1, allow_local_intrabc = 0,
//   max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1 (0 for M1).
//
// The tile supports PARTITION_NONE / HORZ / VERT per 64x64 superblock, DC
// intra blocks (y/uv DC + palette flag 0 + EOB=0 residual) and IBC blocks with
// NEARMV or NEWMV (§5.20.5.4).  It is deliberately small enough to audit; the
// production call sites are listed in README.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/intra/ibc/stream/ibc_syntax.h"

namespace av2enc::still::ibc_stream {

struct StreamConfig {
  int width = 768;
  int height = 128;
  std::uint8_t base_q_idx = 90;
  std::uint8_t seq_level_idx = 7;
  bool allow_screen_content_tools = true;
  bool force_integer_mv = true;
  bool allow_intrabc = true;
  bool allow_global_intrabc = true;
  bool allow_local_intrabc = false;
  int seq_max_bvp_drl_bits_minus_1 = 0;
  bool disable_cdf_update = true;
  bool enable_cfl_intra = false;
  bool enable_intra_edge_filter = false;
  bool apply_deblock_y_vert = false;
  bool apply_deblock_y_horz = false;
  bool apply_deblock_u = false;
  bool apply_deblock_v = false;
  // Reject BVs whose -1,-1 reference border leaves the tile (dav2d --strict 1).
  bool strict_border = true;
  // Extra sequence-header tool bits kept identical to the M1 reference stream.
  bool enable_chroma_dctonly = true;
  int reduced_tx_set = 0;
};

struct StreamBlock {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 12;  // partition::BlockSize
  bool ibc = false;
  BvSyntax syntax;
  BlockVector bv{};
  bool skip = true;  // IBC only; true = no residual symbols
};

struct StreamSb {
  enum class Partition { kNone, kHorz, kVert };
  Partition partition = Partition::kNone;
  std::vector<StreamBlock> leaves;
};

struct StreamPlan {
  StreamConfig config;
  std::vector<StreamSb> sbs;  // raster order
};

struct StreamResult {
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::size_t tile_bytes = 0;
  int ibc_blocks = 0;
  int newmv_blocks = 0;
};

// Validate the plan, write all symbols, and reconstruct.  Every IBC BV is
// checked with is_bv_encodable() before a single bit is emitted.
bool encode_ibc_stream(const StreamPlan& plan, StreamResult& out,
                       std::string& error);

bool write_ibc_ivf(const StreamResult& result, const StreamConfig& config,
                   const std::string& path, std::string& error);
bool write_ibc_y4m(const StreamResult& result, const StreamConfig& config,
                   const std::string& path, std::string& error);

// §5.20.4.1 add_neighbor() locations.  NPos (buffered=false) is the
// use_intrabc context list: at most two positions in the current superblock
// row.  NPosBuf (buffered=true) is the skip_flag context list: at most two
// positions and it may cross a superblock row.  A wide left block is
// intentionally visited twice (bottom-left and left), so the context can be 2
// from a single decoded block.
struct MiNeighbours {
  int count = 0;
  int row[2] = {0, 0};
  int col[2] = {0, 0};
};

MiNeighbours mode_info_neighbours(int mi_row, int mi_col, int bsize,
                                  int mi_cols, int mi_rows, int sb_size,
                                  bool buffered) noexcept;

// Sum the supplied MI grid over the exact NPos / NPosBuf lists.
int ibc_symbol_context(const std::vector<std::uint8_t>& grid, int mi_cols,
                       int mi_rows, int mi_row, int mi_col, int bsize,
                       int sb_size) noexcept;
int skip_symbol_context(const std::vector<std::uint8_t>& grid, int mi_cols,
                        int mi_rows, int mi_row, int mi_col, int bsize,
                        int sb_size) noexcept;

}  // namespace av2enc::still::ibc_stream
