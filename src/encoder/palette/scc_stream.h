// Minimal AV2 stream writer for the screen-content tools.
//
// This is the *gate* writer used by the palette and IBC modules: it emits a
// legal AV2 still picture whose blocks are DC intra, palette intra, or intra
// block copy, so `dav2d` can validate the new syntax in isolation.  It
// deliberately does not touch `src/encoder/m1/m1_syntax.cpp` or
// `src/main.cpp`; the production integration is a later, separate step.
//
// The sequence header is the existing M1 header (single_picture_header_flag=1,
// 8-bit 4:2:0, one tile, 64x64 SBs, no filters).  The frame header is re-emitted
// here because `m1_headers.cpp` hard-codes `allow_screen_content_tools = 0` and
// `allow_intrabc = 0`; those two bits (plus `force_integer_mv` and the
// IBC allow flags) are the only difference.
//
// Supported block plans: PARTITION_NONE (one 64x64 leaf), PARTITION_HORZ
// (two 64x32) and PARTITION_VERT (two 32x64) per superblock.  That is enough
// for the 64x64 palette gate and the 384x64 IBC gate, and it keeps the writer
// small enough to audit against the spec.
//
// Reconstruction: this writer only needs the *decoder's* reconstruction for
// IBC source blocks.  DC blocks are predicted with the normative DC process
// (flat 128 on an empty edge); palette blocks use §7.13.4; chroma is DC.  It
// does not implement transforms: every residual is EOB=0, so the
// reconstruction equals the prediction.  That is the transform-skip case and
// is exactly what palette "lossless colour" mode means in AV2.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "encoder/palette/palette.h"

namespace av2enc::scc {

struct SccConfig {
  int width = 64;
  int height = 64;
  std::uint8_t base_q_idx = 90;
  bool allow_screen_content_tools = true;
  bool allow_intrabc = false;
  bool allow_global_intrabc = false;
  bool allow_local_intrabc = true;
  // Sequence header value is 0 (one DRL bit).  Keep it fixed for the gate.
  int seq_max_bvp_drl_bits_minus_1 = 0;
  // Optional tool flags.  The default keeps the gate stream minimal; the
  // tests flip CfL/deblock to match the integrated M1 header byte-for-byte.
  bool enable_cfl_intra = false;
  bool enable_intra_edge_filter = false;
  bool apply_deblock_y_vert = false;
  bool apply_deblock_y_horz = false;
  bool apply_deblock_u = false;
  bool apply_deblock_v = false;
};

enum class BlockKind {
  kDc,      // DC intra, EOB=0
  kPalette, // luma palette (§5.20.8), chroma DC, EOB=0
  kIbc,     // intra block copy, skip_flag=1 (no residual)
};

struct BlockPlan {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 12;  // partition::BlockSize numeric value
  BlockKind kind = BlockKind::kDc;

  // Palette payload (used for kPalette).
  palette::PaletteResult palette;

  // IBC payload (used for kIbc).  Block vector in 1/8 luma samples, as in the
  // bitstream.  `ref_mv_idx` selects a RefStackMv candidate; `intrabc_mode`
  // true means NEARMV (no MVD) and matches the gate's candidate-only search.
  int bv_row = 0;
  int bv_col = 0;
  int ref_mv_idx = 0;
  bool intrabc_mode = true;
  bool skip = true;
};

struct SbPlan {
  enum class Partition { kNone, kHorz, kVert };
  Partition partition = Partition::kNone;
  std::vector<BlockPlan> leaves;  // 1 for kNone, 2 for HORZ/VERT
};

struct SccFramePlan {
  SccConfig config;
  std::vector<SbPlan> sbs;  // raster order
};

struct SccEncodeResult {
  std::vector<std::uint8_t> frame;  // complete OBU stream (TD + SEQ + CLK)
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  std::size_t tile_bytes = 0;
};

// Validate the plan against the frame geometry and the tool constraints, then
// encode.  On failure `error` explains which symbol/step failed.
bool encode_scc_frame(const SccFramePlan& plan, SccEncodeResult& out,
                      std::string& error);

// Write one complete IVF file (32-byte DKIF header + one frame) using the
// existing IvfWriter.  The frame payload is `result.frame`.
bool write_scc_ivf(const SccEncodeResult& result, const SccConfig& config,
                   const std::string& path, std::string& error);

// Write the writer's reconstruction as a one-frame 4:2:0 Y4M.  The gate
// scripts compare this with dav2d's decoded Y4M byte-for-byte.
bool write_scc_y4m(const SccEncodeResult& result, const SccConfig& config,
                   const std::string& path, std::string& error);

// §7.13.3.17 IBC plane offset: luma BV (1/8 samples) -> integer sample offset
// on `plane` with the given subsampling.  Exposed for the tests.
int ibc_plane_offset(int bv8, int subsampling);

// Global-only BV validity check used by the gate writer (§6.19.7.12).  Returns
// false and a reason if the source is outside the tile or inside the
// INTRABC_DELAY_SB64 window.
bool validate_global_bv(const SccConfig& config, int mi_row, int mi_col,
                        int bsize, int bv_row, int bv_col,
                        std::string& error);

}  // namespace av2enc::scc
