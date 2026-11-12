// Standalone 10-bit 4:2:0 still-picture encoder (M10).
//
// Tool subset (deliberately smaller than M1, so it has no dependency on the
// M1 decision/reconstruction core):
//   * one tile, one intra key frame, 64x64 superblocks
//   * full superblocks: PARTITION_VERT -> 32x64 -> PARTITION_HORZ -> 32x32
//   * partial superblocks: PARTITION_NONE at 64x64 (the M0-proven edge rule)
//   * DC intra only, no intra edge filter, no CfL, no palette / IBP / DIP
//   * DCT_DCT, TX_MODE_LARGEST, no deblock/CDEF/CCSO/GDF/restoration/film grain
//
// The reconstructed samples are the decoder's samples: 10-bit clips, DC bias
// 512, inverse transform from `hbd_cpu`.  Entropy is the normative §8.2 coder
// with CDF slice 0 selected from the raw base_q_idx (no depth normalization).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "formats/format.h"

namespace av2enc::high_bitdepth {

struct HighBitDepthOptions {
  // Raw qindex written f(9).  The quantizer is the fixed QP-90 reference, so
  // the encoder accepts only 90; other values would desynchronize dequant.
  std::uint16_t base_q_idx = 90;
  bool disable_cdf_update = true;
};

// One leaf coding block.  `luma_size` is 32 or 64 (square).  Coefficient
// levels are stored in the flat array in scan order, Y/U/V contiguous.
struct HighBitDepthBlock {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t luma_size = 32;
  std::uint8_t tx_size[3] = {3, 2, 2};  // TxSize: 32x32 / 16x16
  std::uint32_t eob[3] = {0, 0, 0};
  std::uint32_t levels_offset = 0;
};

struct HighBitDepthPlane {
  int width = 0;           // coded, 64-aligned storage width
  int height = 0;          // 64-aligned storage height
  int coded_width = 0;     // MiCols*4 / chroma shift
  int coded_height = 0;
  int visible_width = 0;
  int visible_height = 0;
  std::vector<std::uint16_t> samples;  // 10-bit, zero-initialized
};

struct HighBitDepthResult {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK
  std::vector<HighBitDepthBlock> blocks;     // preorder partition leaves
  std::vector<std::int32_t> levels; // flat scan-order coefficients
  HighBitDepthPlane recon[3];
};

// Partition leaf order (decode order) for the fixed M10 tree.  Both the CPU
// and GPU block stages consume this plan, so their block indexing cannot
// drift.
struct HighBitDepthBlockDesc {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t luma_size = 32;
};

std::vector<HighBitDepthBlockDesc> make_block_plan(std::uint32_t width,
                                          std::uint32_t height);

// The fixed 32x32/64x64 M10 tree codes PARTITION_NONE for a partial superblock
// only when the remaining size is at least 8 MI (32 luma samples) in each
// direction.  Smaller remainders force a normative implied partition and would
// need the full 32x16/32x8/16x32 tree; M10 rejects them instead of emitting a
// desynchronized stream.  Returns false with a reason for those geometries.
bool high_bitdepth_geometry_supported(std::uint32_t width, std::uint32_t height,
                            std::string* reason);

// CPU reference.  Accepts a 10-bit 4:2:0 picture from `fmt`; visible samples
// must be in [0,1023].  Produces the complete frame OBUs, the decoder-exact
// reconstructed planes, and the block/level stream used by the entropy writer.
bool encode_high_bitdepth_cpu(const fmt::Picture& picture, const HighBitDepthOptions& options,
                    HighBitDepthResult& result, std::string& error);

// Shared entropy/OBU assembly.  `blocks` must be in decode order and `levels`
// must already be scan-ordered; both the CPU and GPU front ends call this.
bool assemble_high_bitdepth_stream(const fmt::Picture& picture, const HighBitDepthOptions& options,
                         const HighBitDepthBlock* blocks, std::size_t block_count,
                         const std::int32_t* levels, std::size_t level_count,
                         std::vector<std::uint8_t>* out, std::uint32_t* tile_bytes,
                         std::string* error);

// 10-bit samples -> the same planes shifted left by 2.  The format component
// owns the conversion; this helper is for the Y4M tool's 8-bit input path.
bool convert_8bit_420_to_picture(const std::uint8_t* y, const std::uint8_t* u,
                                 const std::uint8_t* v, std::uint32_t width,
                                 std::uint32_t height, std::uint32_t y_stride,
                                 fmt::Picture* picture, std::string* error);

}  // namespace av2enc::high_bitdepth
