// M2 adapter: a §5.18 frame header writer that can reference an MFH.
//
// This is the *reference integration* for src/encoder/m2/m2_headers.cpp.  It
// mirrors write_frame_header_syntax() there bit for bit when `mfh == nullptr`;
// the decoder gate asserts that fact against streams produced by the untouched
// M2 encoder.  When an MFH is supplied, the header:
//
//   * codes cur_mfh_id = mfh->mfh_id and omits seq_header_id_in_frame_header
//     (§5.18.2: the sequence header is loaded through MfhSeqHeaderId);
//   * omits the apply_deblocking_filter bits when
//     mfh_deblocking_filter_update is 1 (§5.18.5.2);
//   * lets segmentation_params() reuse the MFH table (§5.18.7.1).
//
// Frame size is inherited implicitly: frame_size_override_flag == 0 makes
// frame_size() use mfh_frame_width/height_minus_1 (§5.18.4.1), which is where
// the MFH frame size takes effect.
//
// This file deliberately duplicates the small M2 header subset instead of
// editing the M2 core.  The exact replacement sites are listed in README.md.
#pragma once

#include "bitstream/bitwriter.h"
#include "encoder/frame/mfh/mfh.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::frame::mfh {

struct M2MfhConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t base_q_idx = 90;
  std::uint8_t order_hint_bits = 4;
  std::uint8_t num_ref_frames = 2;
  std::uint8_t num_total_refs = 1;
  bool disable_cdf_update = true;
  bool enable_ext_seg = false;
};

struct M2MfhFrameHeader {
  bool keyframe = false;
  // Only coded for non-key frames (OBU_REGULAR_TILE_GROUP and friends).
  bool frame_is_inter = true;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 1;
  std::uint8_t ref_slot = 0;
  bool allow_intrabc = false;

  // Explicit frame size.  Supported for intra frames (the frame_size() path).
  // Inter frames would need the frame_size_with_refs() found_ref bits; that is
  // intentionally rejected here to keep the adapter honest.
  bool frame_size_override = false;
  std::uint32_t frame_width = 0;
  std::uint32_t frame_height = 0;

  bool segmentation_enabled = false;
  // The coded reuse_seg_info bit, used only when the MFH/sequence allows a
  // change.  Ignored otherwise (the spec derives the value).
  bool reuse_seg_info = true;
  // New table when reuse is not selected.
  SegInfo seg_info;
};

// Writes one frame_header_info() (§5.18.2) without is_first_tile_group and
// without byte_alignment().  Returns the bit count, or 0 on error.
std::size_t write_m2_frame_header_with_mfh(BitWriter& writer,
                                           const M2MfhConfig& config,
                                           const M2MfhFrameHeader& header,
                                           const MultiFrameHeader* mfh,
                                           std::string* error);

// §5.19 tile_group_obu() header: is_first_tile_group + frame_header +
// byte_alignment().
std::size_t write_m2_tile_group_header_with_mfh(
    BitWriter& writer, const M2MfhConfig& config,
    const M2MfhFrameHeader& header, const MultiFrameHeader* mfh,
    std::string* error);

// Complete OBU_CLOSED_LOOP_KEY / OBU_REGULAR_TILE_GROUP with the supplied tile
// payload (which must end with the §8.2.4 exit_symbol bits).
std::vector<std::uint8_t> make_m2_frame_obu_with_mfh(
    const M2MfhConfig& config, const M2MfhFrameHeader& header,
    const MultiFrameHeader* mfh, std::span<const std::uint8_t> tile_payload,
    std::string* error);

}  // namespace av2enc::frame::mfh
