// 10-bit 4:2:0 still-picture encoder on top of the verified M10 components.
//
// This is the *sample/transform/reconstruction* half of `src/encoder/hbd/`.
// It deliberately reuses:
//
//   * `src/encoder/m10/high_bitdepth_still.*`   block plan, geometry guard, DC predictor
//                                     shape, entropy/OBU assembly (tile payload)
//   * `src/cuda/m1/hbd/hbd_cpu.*`     depth-aware dequant/inverse/recon oracle
//   * `src/encoder/tx_variants/`      forward transform + quantisation
//
// What it changes relative to `av2enc::high_bitdepth::encode_high_bitdepth_cpu()` is the quantizer step:
// §7.14.2 `qlookup(raw_base_q_idx)` replaces the fixed QP-90 step (528).  The
// forward `TxParams::quant_factor` is scaled by 528/qlookup and the inverse
// dequant uses qlookup, exactly as the decoder derives it from `base_q_idx`.
// At `base_q_idx == 90` the output is byte-identical to M10 (pinned in tests).
//
// qindex seam: the reused M10 entropy writer has only the frozen §9.3
// coefficient-CDF slice 0.  §8.3.1 selects slices 1..3 for base_q_idx > 90, so
// encoding is limited to 1..90 even though the header writer and the depth
// helpers handle the full 10-bit MaxQ range (303).  See README.md.
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
  // Raw qindex written f(9).  Encoding supports 1..90 (slice 0); the header
  // writer/reader and depth helpers support the full 10-bit range 1..303.
  std::uint16_t base_q_idx = 90;
  bool disable_cdf_update = true;
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

// True when the *tile payload* writer can code this qindex (1..90).  Returns
// false with a precise reason for 0, >MaxQ, or the slice 1..3 qindexes that
// need a qctx-aware entropy writer.
bool qindex_encodable(std::uint16_t base_q_idx, std::string* reason);

// Encodes a 10-bit 4:2:0 picture.  Visible samples must be in [0,1023] and the
// geometry must satisfy the M10 fixed partition tree.
bool encode_still(const fmt::Picture& picture, const HbdOptions& options,
                  HbdResult& result, std::string& error);

}  // namespace av2enc::hbd
