#include "formats/high_bitdepth/hbd_reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::hbd {
namespace {

// MSB-first bit reader, matching BitWriter.
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

  bool read_bit(bool* out) {
    if (bit_pos_ >= data_.size() * 8U) return false;
    const std::size_t byte = bit_pos_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    *out = ((data_[byte] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool read_bits(unsigned count, std::uint32_t* out) {
    if (count > 32U) return false;
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      value = (value << 1U) | (bit ? 1U : 0U);
    }
    *out = value;
    return true;
  }

  // uvlc() (§4.11.3).
  bool uvlc(std::uint32_t* out) {
    unsigned leading_zeros = 0;
    bool bit = false;
    while (true) {
      if (!read_bit(&bit)) return false;
      if (bit) break;
      if (++leading_zeros >= 32U) return false;
    }
    std::uint32_t payload = 0;
    if (leading_zeros > 0 && !read_bits(leading_zeros, &payload)) {
      return false;
    }
    const std::uint64_t base =
        leading_zeros == 0 ? 0ULL : ((1ULL << leading_zeros) - 1ULL);
    *out = static_cast<std::uint32_t>(base + payload);
    return true;
  }

  // ns(n) (§4.11.8).
  bool ns(std::uint32_t n, std::uint32_t* out) {
    if (n < 2U) {
      *out = 0;
      return true;
    }
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    std::uint32_t value = 0;
    if (!read_bits(w - 1U, &value)) return false;
    if (value < m) {
      *out = value;
      return true;
    }
    bool extra = false;
    if (!read_bit(&extra)) return false;
    *out = ((value << 1U) | (extra ? 1U : 0U)) - m;
    return true;
  }

  bool align_zero() {
    while ((bit_pos_ & 7U) != 0U) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      if (bit) return false;
    }
    return true;
  }

  std::size_t bits_read() const { return bit_pos_; }
  std::size_t bits_total() const { return data_.size() * 8U; }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t bit_pos_ = 0;
};

bool read_leb128(std::span<const std::uint8_t> bytes, std::size_t* pos,
                 std::uint32_t* value) {
  std::uint64_t result = 0;
  unsigned shift = 0;
  while (true) {
    if (*pos >= bytes.size()) return false;
    const std::uint8_t byte = bytes[(*pos)++];
    result |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
    if ((byte & 0x80U) == 0U) break;
    shift += 7U;
    if (shift > 28U) return false;
  }
  if (result > 0xffffffffULL) return false;
  *value = static_cast<std::uint32_t>(result);
  return true;
}

bool parse_sequence(std::span<const std::uint8_t> payload,
                    HbdSequenceHeader* out, std::string* error) {
  const auto fail = [error](const char* text) {
    if (error != nullptr) *error = text;
    return false;
  };
  BitReader reader(payload);
  std::uint32_t value = 0;

  if (!reader.uvlc(&value)) return fail("sequence: truncated seq_header_id");
  out->seq_header_id = value;
  if (!reader.read_bits(5U, &value)) return fail("sequence: truncated profile");
  out->seq_profile_idc = static_cast<int>(value);
  bool bit = false;
  if (!reader.read_bit(&bit)) return fail("sequence: truncated single_picture");
  out->single_picture_header_flag = bit;
  if (!reader.read_bits(5U, &value)) return fail("sequence: truncated level");
  out->seq_level_idx = static_cast<int>(value);
  if (!reader.uvlc(&value)) return fail("sequence: truncated chroma_format_idc");
  out->chroma_format_idc = static_cast<int>(value);
  if (!reader.uvlc(&value)) return fail("sequence: truncated bit_depth_idc");
  out->bit_depth_idc = static_cast<int>(value);

  std::uint32_t width_bits = 0;
  std::uint32_t height_bits = 0;
  if (!reader.read_bits(4U, &width_bits)) return fail("sequence: width_bits");
  if (!reader.read_bits(4U, &height_bits)) return fail("sequence: height_bits");
  std::uint32_t width_minus_1 = 0;
  std::uint32_t height_minus_1 = 0;
  if (!reader.read_bits(width_bits + 1U, &width_minus_1)) {
    return fail("sequence: max_frame_width");
  }
  if (!reader.read_bits(height_bits + 1U, &height_minus_1)) {
    return fail("sequence: max_frame_height");
  }
  out->max_frame_width = width_minus_1 + 1U;
  out->max_frame_height = height_minus_1 + 1U;

  // seq_cropping_window_present_flag.
  if (!reader.read_bit(&bit)) return fail("sequence: cropping flag");

  // §5.4.3.
  for (int i = 0; i < 5; ++i) {
    if (!reader.read_bit(&bit)) return fail("sequence: partition config");
  }
  // §5.4.4.
  if (!reader.read_bit(&bit)) return fail("sequence: segment config 0");
  if (!reader.read_bit(&bit)) return fail("sequence: segment config 1");
  // §5.4.5.
  if (!reader.read_bit(&bit)) return fail("sequence: dip");  // enable_dip
  if (!reader.read_bit(&bit)) return fail("sequence: edge filter");
  out->enable_intra_edge_filter = bit;
  if (!reader.read_bit(&bit)) return fail("sequence: mrls");
  out->enable_mrls = bit;
  if (!reader.read_bit(&bit)) return fail("sequence: cfl");
  out->enable_cfl_intra = bit;
  if (!reader.read_bits(2U, &value)) return fail("sequence: cfl_ds");
  out->cfl_ds_filter_index = static_cast<int>(value);
  if (!reader.read_bit(&bit)) return fail("sequence: mhccp");
  if (!reader.read_bit(&bit)) return fail("sequence: ibp");
  // §5.4.6.
  if (!reader.read_bit(&bit)) return fail("sequence: refmvbank");
  if (!reader.read_bit(&bit)) return fail("sequence: drl reorder");
  if (!reader.ns(3U, &value)) return fail("sequence: bvp drl");
  if (!reader.read_bit(&bit)) return fail("sequence: frame max bvp");
  if (!reader.read_bit(&bit)) return fail("sequence: bawp");
  // §5.4.8.
  if (!reader.read_bit(&bit)) return fail("sequence: fsc");
  if (!reader.read_bit(&bit)) return fail("sequence: idtx");
  if (!reader.read_bit(&bit)) return fail("sequence: intra ist");
  if (!reader.read_bit(&bit)) return fail("sequence: inter ist");
  if (!reader.read_bit(&bit)) return fail("sequence: chroma dctonly");
  out->enable_chroma_dctonly = bit;
  if (!reader.read_bit(&bit)) return fail("sequence: reduced tx part");
  if (!reader.read_bit(&bit)) return fail("sequence: cctx");
  if (!reader.read_bit(&bit)) return fail("sequence: tcq");
  if (!reader.read_bit(&bit)) return fail("sequence: parity hiding");
  if (!reader.read_bit(&bit)) return fail("sequence: separate uv delta q");
  if (!reader.read_bit(&bit)) return fail("sequence: equal ac dc q");
  if (!reader.read_bits(5U, &value)) return fail("sequence: base uv ac delta");
  if (!reader.read_bit(&bit)) return fail("sequence: uv ac delta enabled");
  // §5.4.10.
  if (!reader.read_bit(&bit)) return fail("sequence: loopfilters across tiles");
  if (!reader.read_bit(&bit)) return fail("sequence: cdef");
  out->enable_cdef = bit;
  if (!reader.read_bit(&bit)) return fail("sequence: gdf");
  if (!reader.read_bit(&bit)) return fail("sequence: restoration");
  if (!reader.read_bit(&bit)) return fail("sequence: ccso");
  out->enable_ccso = bit;
  if (!reader.read_bits(2U, &value)) return fail("sequence: df par bits");
  // §5.4.2.
  if (!reader.read_bit(&bit)) return fail("sequence: seq tile info");
  out->seq_tile_info_present_flag = bit;
  // film_grain_params_present.
  if (!reader.read_bit(&bit)) return fail("sequence: film grain");

  // §5.2.1 obu_extension_flag, then trailing_bits().
  if (!reader.read_bit(&bit)) return fail("sequence: obu_extension_flag");
  if (bit) return fail("sequence: extension payload not supported");
  if (!reader.read_bit(&bit) || !bit) return fail("sequence: trailing one bit");
  while (reader.bits_read() < reader.bits_total()) {
    if (!reader.read_bit(&bit)) return fail("sequence: trailing bits");
    if (bit) return fail("sequence: nonzero trailing bit");
  }
  return true;
}

}  // namespace

bool parse_stream(std::span<const std::uint8_t> bytes, HbdParsedStream* out,
                  std::string* error) {
  if (out == nullptr) {
    if (error != nullptr) *error = "null output";
    return false;
  }
  *out = HbdParsedStream{};
  if (error != nullptr) error->clear();

  std::size_t pos = 0;
  while (pos < bytes.size()) {
    HbdObu obu;
    obu.prefix_offset = pos;
    std::uint32_t obu_size = 0;
    if (!read_leb128(bytes, &pos, &obu_size)) {
      if (error != nullptr) *error = "bad leb128 OBU size";
      return false;
    }
    if (obu_size == 0U || pos + obu_size > bytes.size()) {
      if (error != nullptr) *error = "OBU size out of range";
      return false;
    }
    const std::uint8_t header = bytes[pos];
    obu.extension_flag = (header & 0x80U) != 0U;
    obu.type = static_cast<int>((header >> 2U) & 0x1fU);
    obu.tlayer_id = static_cast<int>(header & 0x03U);
    obu.header_bytes = obu.extension_flag ? 2U : 1U;
    if (obu.header_bytes > obu_size) {
      if (error != nullptr) *error = "OBU header larger than OBU";
      return false;
    }
    obu.payload_offset = pos + obu.header_bytes;
    obu.payload_size = obu_size - obu.header_bytes;
    out->obus.push_back(obu);
    pos += obu_size;
  }

  // Parse the sequence header and the first tile group.
  for (const HbdObu& obu : out->obus) {
    const std::span<const std::uint8_t> payload =
        bytes.subspan(obu.payload_offset, obu.payload_size);
    if (obu.type == 1) {  // OBU_SEQUENCE_HEADER
      HbdSequenceHeader seq;
      if (!parse_sequence(payload, &seq, error)) return false;
      if (seq.bit_depth_idc != 0) {
        if (error != nullptr) *error = "sequence is not 10-bit (idc != 0)";
        return false;
      }
      if (seq.chroma_format_idc != 0) {
        if (error != nullptr) *error = "sequence is not 4:2:0";
        return false;
      }
      if (out->has_sequence) {
        if (error != nullptr) *error = "multiple sequence headers";
        return false;
      }
      out->sequence = seq;
      out->has_sequence = true;
    }
  }
  if (!out->has_sequence) {
    if (error != nullptr) *error = "missing sequence header";
    return false;
  }

  for (const HbdObu& obu : out->obus) {
    if (obu.type != 4 && obu.type != 7) continue;  // CLK / regular tile group
    const std::span<const std::uint8_t> payload =
        bytes.subspan(obu.payload_offset, obu.payload_size);
    BitReader reader(payload);
    HbdFrameHeader frame;
    std::uint32_t value = 0;
    bool bit = false;

    if (!reader.read_bit(&bit) || !bit) {
      if (error != nullptr) *error = "tile group: is_first_tile_group";
      return false;
    }
    frame.is_first_tile_group = true;
    if (!reader.uvlc(&value)) {
      if (error != nullptr) *error = "frame: cur_mfh_id";
      return false;
    }
    if (!reader.uvlc(&value)) {
      if (error != nullptr) *error = "frame: seq_header_id";
      return false;
    }
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: screen content tools";
      return false;
    }
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: allow_intrabc";
      return false;
    }
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: disable_cdf_update";
      return false;
    }
    frame.disable_cdf_update = bit;

    // tile_info(): one 0 bit per dimension with more than one SB.
    const std::uint32_t sb_cols =
        (2U * ((out->sequence.max_frame_width + 7U) >> 3U) + 15U) >> 4U;
    const std::uint32_t sb_rows =
        (2U * ((out->sequence.max_frame_height + 7U) >> 3U) + 15U) >> 4U;
    if (!reader.read_bit(&bit) || !bit) {
      if (error != nullptr) *error = "frame: uniform_tile_spacing";
      return false;
    }
    if (sb_cols > 1U) {
      if (!reader.read_bit(&bit) || bit) {
        if (error != nullptr) *error = "frame: increment_tile_cols_log2";
        return false;
      }
    }
    if (sb_rows > 1U) {
      if (!reader.read_bit(&bit) || bit) {
        if (error != nullptr) *error = "frame: increment_tile_rows_log2";
        return false;
      }
    }

    // §5.18.6.1: f(9) at 10-bit.
    if (!reader.read_bits(9U, &value)) {
      if (error != nullptr) *error = "frame: base_q_idx";
      return false;
    }
    frame.base_q_idx = static_cast<std::uint16_t>(value);
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: segmentation";
      return false;
    }
    frame.segmentation_enabled = bit;
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: using_qmatrix";
      return false;
    }
    frame.using_qmatrix = bit;
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: delta_q_present";
      return false;
    }
    frame.delta_q_present = bit;

    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: deblock y vert";
      return false;
    }
    frame.apply_deblocking_filter[0] = bit;
    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: deblock y horz";
      return false;
    }
    frame.apply_deblocking_filter[1] = bit;
    const bool have_chroma =
        frame.apply_deblocking_filter[0] || frame.apply_deblocking_filter[1];
    if (have_chroma) {
      if (!reader.read_bit(&bit)) {
        if (error != nullptr) *error = "frame: deblock u";
        return false;
      }
      frame.apply_deblocking_filter[2] = bit;
      if (!reader.read_bit(&bit)) {
        if (error != nullptr) *error = "frame: deblock v";
        return false;
      }
      frame.apply_deblocking_filter[3] = bit;
    }
    for (int plane = 0; plane < 4; ++plane) {
      if (!frame.apply_deblocking_filter[plane]) continue;
      if (!reader.read_bit(&bit) || bit) {
        if (error != nullptr) *error = "frame: df_delta_q_present";
        return false;
      }
    }

    if (!reader.read_bit(&bit)) {
      if (error != nullptr) *error = "frame: tx_mode_select";
      return false;
    }
    frame.tx_mode_select = bit;
    if (!reader.read_bits(2U, &value)) {
      if (error != nullptr) *error = "frame: reduced_tx_set";
      return false;
    }
    frame.reduced_tx_set = static_cast<int>(value);

    if (!reader.align_zero()) {
      if (error != nullptr) *error = "frame: byte_alignment not zero";
      return false;
    }
    const std::size_t header_bytes = reader.bits_read() / 8U;
    out->frame = frame;
    out->has_frame = true;
    out->tile_payload_offset = obu.payload_offset + header_bytes;
    out->tile_payload.assign(
        bytes.begin() + static_cast<std::ptrdiff_t>(out->tile_payload_offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(obu.payload_offset +
                                                    obu.payload_size));
    return true;
  }

  if (error != nullptr) *error = "missing tile group";
  return false;
}

}  // namespace av2enc::hbd
