// In-loop filter configuration shared by the 10-bit product writers.
//
// The planner headers (`hbd_cdef.h`, `hbd_ccso.h`, ...) fill these structures
// from the reconstruction, `hbd_headers.cpp` serialises them into the
// sequence/frame header, and `high_bitdepth_entropy.cpp` codes the matching
// tile symbols.  Keeping one description is what keeps the three in sync.
#ifndef AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_FILTER_CONFIG_H_
#define AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_FILTER_CONFIG_H_

#include <cstdint>
#include <vector>

#include "codec/filters/loop_restoration/restoration_common.h"

namespace av2enc::hbd {

// §5.18.7.12 ccso_params() for one plane plus the §5.20.10.2 tile decision
// grid.  `offset_idx` holds the raw 7-bit ccso_offset_idx values in syntax
// order (d0, d1, band), which is exactly what `write_ccso_params` consumes.
struct HbdCcsoPlaneConfig {
  bool enabled = false;
  bool bo_only = true;
  int scale_idx = 0;
  int quant_idx = 0;
  int ext_filter = 0;
  int edge_clf = 0;
  int max_band_log2 = 0;
  std::vector<std::uint8_t> offset_idx;
};

struct HbdCcsoConfig {
  bool enabled = false;
  int ccso_luma_size_log2 = 6;  // 6 for a 64x64 frame superblock
  bool unit_matches_sb_size = true;
  HbdCcsoPlaneConfig plane[3];
  // CcsoBlks[plane]: one byte per luma CCSO unit, row-major.
  std::vector<std::uint8_t> blks[3];
  int blk_cols[3] = {0, 0, 0};
  int blk_rows[3] = {0, 0, 0};
};

// §5.20.10.4 LrWienerNs / §7.20.3 per-unit Wiener NS decision for one plane.
struct HbdLrPlaneConfig {
  bool enabled = false;
  // RestorationType: 1 = PC-Wiener (luma only), 2 = Wiener NS.
  int frame_type = 0;
  int unit_size = 64;   // plane samples
  int shift = 3;        // LoopRestorationSize shift
  int unit_rows = 0;
  int unit_cols = 0;
  // LrType[plane][unitRow][unitCol].
  std::vector<std::uint8_t> unit_type;
  // LrWienerNs[plane][unit][coeff], stride kWienerNsCoeffStride.
  std::vector<std::int16_t> unit_coeffs;
  // Per-unit syntax choices (raster order, same indexing as unit_type).
  std::vector<std::uint8_t> unit_subset;
  std::vector<std::uint8_t> unit_uv_sym;
};

// §7.20 loop restoration frame configuration.
struct HbdLrConfig {
  bool enabled = false;
  int luma_shift = 3;
  int chroma_shift = 3;
  HbdLrPlaneConfig plane[3];
  // §7.20.4 LrTxSkip[mi_row][mi_col] (luma MI grid), used by PC-Wiener.
  std::vector<std::uint8_t> lr_tx_skip;
  int tx_skip_cols = 0;
  int tx_skip_rows = 0;
  int base_q_idx = 90;
};

// §5.18.7.9 GDF frame parameters (header-only mode: gdf_per_block = 0).
struct HbdGdfConfig {
  bool enabled = false;
  int pic_qc_idx = 0;
  int pic_scale_idx = 0;
  bool frame_is_intra = true;
  int base_q_idx = 90;
};

}  // namespace av2enc::hbd

#endif  // AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_FILTER_CONFIG_H_
