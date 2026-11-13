// 10-bit 4:2:0 still-picture encoder on top of the verified M10 components.
//
// This is the *sample/transform/reconstruction* half of `src/formats/high_bitdepth/`.
// It deliberately reuses:
//
//   * `src/pipeline/still/high_bitdepth/high_bitdepth_still.*`   block plan, geometry guard, DC predictor
//                                     shape, entropy/OBU assembly (tile payload)
//   * `src/cuda/pipeline/still/hbd/hbd_cpu.*`     depth-aware dequant/inverse/recon oracle
//   * `src/codec/transform/variants/`      forward transform + quantisation
//
// What it changes relative to `av2enc::high_bitdepth::encode_high_bitdepth_cpu()` is the quantizer step:
// §7.14.2 `qlookup(raw_base_q_idx)` replaces the fixed QP-90 step (528).  The
// forward `TxParams::quant_factor` is scaled by 528/qlookup and the inverse
// dequant uses qlookup, exactly as the decoder derives it from `base_q_idx`.
// At `base_q_idx == 90` the output is byte-identical to M10 (pinned in tests).
//
// qindex seam (closed): `encode_still()` installs the §8.3.1 coefficient-CDF
// slice for the raw base_q_idx, so the tile payload writer handles the full
// 10-bit MaxQ range 1..303.  Slice 0 is byte-identical to the old frozen M10
// tables.  See README.md.
//
// Tool subset (identical to M10): one tile, one intra key frame, 64x64
// superblocks, DC intra only, DCT_DCT, TX_MODE_LARGEST, no CfL / deblock /
// CDEF / CCSO / GDF / restoration / film grain.  The depth helpers in
// `hbd_depth.h` define the exact sample ranges those tools must use when they
// are wired.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "formats/format.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

namespace av2enc::hbd {

struct HbdOptions {
  // Raw qindex written f(9).  The encode path supports the full 10-bit range
  // 1..303 (§8.3.1 slice selected from base_q_idx).
  std::uint16_t base_q_idx = 90;
  bool disable_cdf_update = true;
  // §7.17 deblocking at the 10-bit sample depth.  Off by default so the pinned
  // 10-bit streams are unchanged; `--still-deblock` opts in.
  bool enable_deblock = false;
  // §7.18 CDEF at 10-bit.  Off by default; the strength search runs over the
  // default candidate ladder.  `cdef_num_strengths == 1` codes the
  // header-only form (no per-block index), 2..8 codes `cdef_idx`.
  bool enable_cdef = false;
  int cdef_damping = 3;        // CdefDamping 3..6
  int cdef_num_strengths = 4;  // CdefStrengths 1..8
  // §7.19 CCSO at 10-bit (band-offset and edge-classified forms).
  bool enable_ccso = false;
  int ccso_luma_size_log2 = 6;  // 6 for the 64x64 frame superblock
  // `--still-ccso-edge`: restrict the fit to the edge-classified form so the
  // gate exercises its offset table even when the band-only form wins.
  bool ccso_edge_only = false;
  // §7.20 loop restoration.  `enable_wiener` covers luma Wiener NS,
  // `enable_chroma_lr` the chroma planes.  `enable_pc_wiener` selects
  // RESTORE_PC_WIENER for luma instead of Wiener NS.
  bool enable_wiener = false;
  bool enable_chroma_lr = false;
  bool enable_pc_wiener = false;
  // §7.20.5 GDF (luma, header-only mode).
  bool enable_gdf = false;
};

struct HbdResult {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  int quant_step = 0;
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK
  std::vector<av2enc::high_bitdepth::HighBitDepthBlock> blocks;
  std::vector<std::int32_t> levels;
  av2enc::high_bitdepth::HighBitDepthPlane recon[3];
};

// True when the *tile payload* writer can code this qindex (1..303).  Returns
// false with a precise reason for 0 or >MaxQ.
bool qindex_encodable(std::uint16_t base_q_idx, std::string* reason);

// Encodes a 10-bit 4:2:0 picture.  Visible samples must be in [0,1023] and the
// geometry must satisfy the M10 fixed partition tree.
bool encode_still(const fmt::Picture& picture, const HbdOptions& options,
                  HbdResult& result, std::string& error);

}  // namespace av2enc::hbd
