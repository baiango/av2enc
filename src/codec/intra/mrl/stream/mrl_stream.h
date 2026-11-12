// Minimal M1 still-picture stream with MRL as the only optional tool.
//
// The stock M1 header writer hard-codes enable_mrls = 0 (src/encoder/m1/
// still_headers.cpp), and the stock M1 syntax writer has no mrl_index /
// mrl_sec_index fields.  Editing either file is outside this workstream, so
// this module produces a *self-contained* minimal stream instead:
//
//   * a byte-identical copy of still_headers.cpp's sequence header, with
//     enable_mrls exposed (make_sequence_header_obu);
//   * the stock frame header (av2enc::still::write_frame_header) unchanged;
//   * a narrow tile writer that codes rect-only partition symbols through the
//     existing partition_stream adapter, then for every leaf:
//         read_intra_y_mode  (YMode + the MRL symbols above)
//         read_intra_uv_mode (DC, CfL disabled)
//         txb_skip = 1 for Y/U/V
//
// Only MRL is enabled: CfL, CDEF, CCSO, restoration, deblocking, delta-q,
// palette, DIP, IBP, FSC, and non-DCT transforms are off.  With all residuals
// zero and loop filters off, decoded luma == intra prediction, which is why
// MinimalMrlFrame::recon_y is the exact expected decoder output.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pipeline/still/still_headers.h"

namespace av2enc::still::mrl_stream {

// One leaf's luma decision.  y_mode uses AV2 YMode numbers (0 = DC_PRED,
// 1..8 = directional, 9..12 smooth/Paeth); angle_delta is -3..3.
struct MinimalMrlLeaf {
  int y_mode = 1;  // V_PRED
  int angle_delta = 0;
  int mrl_index = 0;
  bool mrl_sec_index = false;
};

struct MinimalMrlConfig {
  int width = 64;
  int height = 64;
  int sb_size = 64;              // 64 or 128
  int target_block_size = 32;    // square leaf size, 4..64 (power of two)
  std::uint8_t base_q_idx = 90;  // coefficient CDF slice + qlookup
  bool disable_cdf_update = true;
  bool enable_intra_edge_filter = true;
  bool enable_cfl_intra = false;  // must stay false: chroma is coded without CfL
  // Optional explicit per-leaf decisions in decode order.  When empty a
  // deterministic policy is used (all V_PRED, mrl_index cycling 0..3, the
  // secondary flag on every other non-4x4 MRL block).
  std::vector<MinimalMrlLeaf> leaves;
};

struct MinimalMrlLeafRecord {
  int mi_row = 0;
  int mi_col = 0;
  int bsize = 0;  // av2enc::partition::BlockSize
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  bool has_chroma = false;
  MinimalMrlLeaf leaf;
};

struct MinimalMrlFrame {
  std::vector<std::uint8_t> temporal_unit;  // TD + SEQ + CLK OBUs
  std::vector<std::uint8_t> tile_payload;
  std::vector<MinimalMrlLeafRecord> leaves;  // decode order
  // Exact expected decoder luma while all residuals are zero and loop filters
  // are off.  It is exact for leaves whose prediction only reads the adjacent
  // row/column (DC, V, H, and MRL V/H); the gate uses V and DC.  For other
  // directional modes at mrl_index == 0, the §7.13.2.1 above-right /
  // below-left extension counts are not modelled here (they do not affect MRL
  // prediction in the gate cases).
  std::vector<std::uint8_t> recon_y;
  int blocks = 0;
  int mrl_blocks = 0;        // mrl_index > 0
  int secondary_blocks = 0;  // raw mrl_sec_index == 1
  std::string error;
};

// Builds the complete temporal unit and the expected luma reconstruction.
// Returns false with a non-empty error on any inconsistency.
bool write_minimal_mrl_frame(const MinimalMrlConfig& config,
                             MinimalMrlFrame& out);

// Sequence header OBU written by this module.  With enable_mrls == false the
// bytes are identical to av2enc::still::make_sequence_header_obu(config); the
// test pins that equality, so this copy cannot drift silently.
std::vector<std::uint8_t> make_sequence_header_obu(const StillConfig& config,
                                                   bool enable_mrls);

}  // namespace av2enc::still::mrl_stream
