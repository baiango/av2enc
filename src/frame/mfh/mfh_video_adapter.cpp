#include "frame/mfh/mfh_video_adapter.h"

#include "core/bitstream/basic_headers.h"

#include <algorithm>
#include <cassert>

namespace av2enc::frame::mfh {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    assert(leading_zeros < 32U);
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  writer.put_bits(static_cast<std::uint64_t>(value) -
                      ((1ULL << leading_zeros) - 1ULL),
                  leading_zeros);
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

unsigned size_bits(std::uint32_t maximum) {
  unsigned bits = bit_width(maximum - 1U);
  return bits == 0 ? 1U : bits;
}

unsigned ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  unsigned bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

std::uint32_t mi_cols(const VideoMfhConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const VideoMfhConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, const VideoMfhConfig& c) {
  const std::uint32_t sb_cols = (mi_cols(c) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(c) + 15U) >> 4U;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);  // increment_tile_cols_log2 = 0
  if (sb_rows > 1U) writer.put_bit(false);  // increment_tile_rows_log2 = 0
}

void write_seg_info(BitWriter& writer, const SegInfo& seg) {
  const int segments = seg.segments();
  for (int i = 0; i < segments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      const SegmentFeature& feature = seg.feature[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      writer.put_bit(feature.enabled);
      if (!feature.enabled) continue;
      if (j == kSegLvlAltQ) {
        const std::uint32_t mask = (1U << (kSegAltQBits + 1U)) - 1U;
        writer.put_bits(static_cast<std::uint32_t>(feature.value) & mask,
                        kSegAltQBits + 1U);
      }
    }
  }
}

bool write_segmentation_params(BitWriter& writer, const VideoMfhConfig& config,
                               const VideoMfhFrameHeader& header,
                               const MultiFrameHeader* mfh,
                               std::string* error) {
  writer.put_bit(header.segmentation_enabled);
  if (!header.segmentation_enabled) return true;

  bool have_seg_params = false;
  bool allow_change = false;
  const SegInfo* source = nullptr;
  if (mfh != nullptr && mfh->seg_info_present) {
    have_seg_params = (mfh->seg_info.ext_seg == config.enable_ext_seg);
    allow_change = have_seg_params && mfh->allow_seg_info_change;
    source = &mfh->seg_info;
  } else if (config.enable_ext_seg) {
    // M2 does not write a sequence-level seg_info(); this branch only exists
    // to keep the mirror faithful if the sequence ever gains one.
    have_seg_params = false;
    allow_change = false;
  }

  bool reuse = false;
  if (allow_change) {
    writer.put_bit(header.reuse_seg_info);
    reuse = header.reuse_seg_info;
  } else {
    reuse = have_seg_params;
  }
  if (reuse) {
    if (source == nullptr) {
      if (error != nullptr) *error = "segmentation reuse without a source";
      return false;
    }
    // No bits: FeatureEnabled/FeatureData copy from the MFH.
    return true;
  }
  SegInfo seg = header.seg_info;
  seg.ext_seg = config.enable_ext_seg;
  write_seg_info(writer, seg);
  return true;
}

}  // namespace

std::size_t write_video_frame_header_with_mfh(BitWriter& writer,
                                           const VideoMfhConfig& config,
                                           const VideoMfhFrameHeader& header,
                                           const MultiFrameHeader* mfh,
                                           std::string* error) {
  if (error != nullptr) error->clear();
  if (config.base_q_idx == 0) {
    if (error != nullptr) *error = "M2 adapter does not support lossless q = 0";
    return 0;
  }
  if (header.frame_size_override && !header.keyframe) {
    if (error != nullptr) {
      *error =
          "inter-frame frame_size_override needs the "
          "frame_size_with_refs() found_ref bits; not implemented in the "
          "adapter";
    }
    return 0;
  }
  if (mfh != nullptr && mfh->frame_size_present &&
      (mfh->frame_width > config.max_frame_width ||
       mfh->frame_height > config.max_frame_height)) {
    if (error != nullptr) *error = "MFH frame size exceeds the sequence maximum";
    return 0;
  }

  const std::size_t start = writer.bit_size();
  const std::uint32_t cur_mfh_id = mfh != nullptr ? mfh->mfh_id : 0U;

  put_uvlc(writer, cur_mfh_id);  // cur_mfh_id
  if (cur_mfh_id == 0) {
    put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  }

  const bool key = header.keyframe;
  if (!key) writer.put_bit(header.frame_is_inter);
  writer.put_bit(true);  // immediate_output_frame

  if (key) {
    // OBU_CLOSED_LOOP_KEY + max_mlayer_id == 0: refresh_frame_flags is
    // inferred to allFrames and codes no bits.
  }

  writer.put_bit(header.frame_size_override);  // frame_size_override_flag
  writer.put_bits(header.order_hint, config.order_hint_bits);  // order_hint

  if (key) {
    // FrameIsIntra: primary_ref_frame = PRIMARY_REF_NONE, no bits.
    if (header.frame_size_override) {
      const unsigned width_bits = size_bits(config.max_frame_width);
      const unsigned height_bits = size_bits(config.max_frame_height);
      writer.put_bits(header.frame_width - 1U, width_bits);
      writer.put_bits(header.frame_height - 1U, height_bits);
    }
    writer.put_bit(header.allow_intrabc);
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(header.refresh_frame_flags, config.num_ref_frames);
    writer.put_bit(true);   // frame_explicit_ref_frame_map
    writer.put_bits(config.num_total_refs, 3U);
    writer.put_bits(header.ref_slot, ceil_log2(config.num_ref_frames));
    // frame_size_override is rejected for inter frames above.
    writer.put_bit(header.allow_intrabc);
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv (1/8 pel)
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTTAP
  }

  writer.put_bit(config.disable_cdf_update);  // disable_cdf_update
  write_tile_info(writer, config);

  writer.put_bits(config.base_q_idx, 8U);  // base_q_idx
  if (!write_segmentation_params(writer, config, header, mfh, error)) {
    return 0;
  }
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  // §5.18.5.2 deblocking_filter_params().
  if (mfh != nullptr && mfh->deblocking_filter_update) {
    std::array<bool, 4> flags{};
    flags[0] = mfh->apply_deblocking_filter[0];
    flags[1] = mfh->apply_deblocking_filter[1];
    if (flags[0] || flags[1]) {
      flags[2] = mfh->apply_deblocking_filter[2];
      flags[3] = mfh->apply_deblocking_filter[3];
    }
    for (int i = 0; i < 4; ++i) {
      if (!flags[static_cast<std::size_t>(i)]) continue;
      writer.put_bit(false);  // df_delta_q_present
    }
  } else {
    writer.put_bit(false);  // apply_deblocking_filter[0]
    writer.put_bit(false);  // apply_deblocking_filter[1]
  }

  writer.put_bit(false);  // tx_mode_select = TX_MODE_LARGEST
  if (!key) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set

  return writer.bit_size() - start;
}

std::size_t write_video_tile_group_header_with_mfh(
    BitWriter& writer, const VideoMfhConfig& config,
    const VideoMfhFrameHeader& header, const MultiFrameHeader* mfh,
    std::string* error) {
  const std::size_t start = writer.bit_size();
  writer.put_bit(true);  // is_first_tile_group
  if (write_video_frame_header_with_mfh(writer, config, header, mfh, error) == 0) {
    return 0;
  }
  writer.align_to_byte();
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_video_frame_obu_with_mfh(
    const VideoMfhConfig& config, const VideoMfhFrameHeader& header,
    const MultiFrameHeader* mfh, std::span<const std::uint8_t> tile_payload,
    std::string* error) {
  BitWriter writer;
  if (write_video_tile_group_header_with_mfh(writer, config, header, mfh, error) ==
      0) {
    return {};
  }
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const ObuType type =
      header.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  return av2enc::basic::make_obu(ObuHeader{type}, payload);
}

}  // namespace av2enc::frame::mfh
