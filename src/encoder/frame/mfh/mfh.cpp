#include "encoder/frame/mfh/mfh.h"

#include "bitstream/m0_headers.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <utility>

namespace av2enc::frame::mfh {
namespace {

// ---------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

unsigned min_frame_size_bits(std::uint32_t size) {
  // n = mfh_frame_width_bits_minus_1 + 1 carries (size - 1); n >= 1.
  const unsigned bits = bit_width(size - 1U);
  return bits == 0 ? 1U : bits;
}

// uvlc() (§4.11.3).  The largest legal value is (1 << 32) - 2.
void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    assert(leading_zeros < 32U);
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

// su(n) (§4.11.7): n-bit two's complement.
void put_su(BitWriter& writer, std::int32_t value, unsigned bits) {
  assert(bits >= 1U && bits <= 32U);
  const std::uint32_t mask =
      bits == 32U ? 0xFFFFFFFFU : ((1U << bits) - 1U);
  writer.put_bits(static_cast<std::uint32_t>(value) & mask, bits);
}

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

  bool read_bit(bool* value) {
    if (bit_pos_ >= data_.size() * 8U) {
      return fail("unexpected end of MFH payload");
    }
    const std::size_t byte = bit_pos_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    *value = ((data_[byte] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool read_bits(unsigned count, std::uint32_t* value) {
    if (count > 32U) {
      return fail("read_bits count exceeds 32");
    }
    std::uint32_t result = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      result = (result << 1U) | (bit ? 1U : 0U);
    }
    *value = result;
    return true;
  }

  bool read_uvlc(std::uint32_t* value) {
    unsigned leading_zeros = 0;
    while (true) {
      bool done = false;
      if (!read_bit(&done)) return false;
      if (done) break;
      ++leading_zeros;
      if (leading_zeros >= 32U) {
        return fail("uvlc leading zeros >= 32");
      }
    }
    std::uint32_t payload = 0;
    if (!read_bits(leading_zeros, &payload)) return false;
    const std::uint64_t result =
        static_cast<std::uint64_t>(payload) + (1ULL << leading_zeros) - 1ULL;
    if (result > std::numeric_limits<std::uint32_t>::max()) {
      return fail("uvlc value exceeds 32 bits");
    }
    *value = static_cast<std::uint32_t>(result);
    return true;
  }

  bool read_su(unsigned bits, std::int32_t* value) {
    std::uint32_t raw = 0;
    if (!read_bits(bits, &raw)) return false;
    const std::uint32_t sign_mask = 1U << (bits - 1U);
    std::int64_t signed_value = raw;
    if ((raw & sign_mask) != 0U) {
      signed_value -= static_cast<std::int64_t>(sign_mask) * 2;
    }
    *value = static_cast<std::int32_t>(signed_value);
    return true;
  }

  std::size_t bits_remaining() const {
    const std::size_t total = data_.size() * 8U;
    return bit_pos_ <= total ? total - bit_pos_ : 0U;
  }

  std::size_t bit_position() const { return bit_pos_; }

  // §5.2.3 trailing_bits(): one 1 bit followed by zero bits to the end of the
  // OBU payload.  Additional zero padding bytes are allowed.
  bool read_trailing_bits() {
    bool one = false;
    if (!read_bit(&one)) return false;
    if (!one) return fail("trailing_bits must start with a 1 bit");
    while (bits_remaining() > 0U) {
      bool zero = true;
      if (!read_bit(&zero)) return false;
      if (zero) return fail("trailing_bits contains a non-zero padding bit");
    }
    return true;
  }

  const std::string& error() const { return error_; }
  void set_error(std::string message) { error_ = std::move(message); }

 private:
  bool fail(std::string message) {
    if (error_.empty()) {
      error_ = std::move(message) + " at bit " + std::to_string(bit_pos_);
    }
    return false;
  }

  std::span<const std::uint8_t> data_;
  std::size_t bit_pos_ = 0;
  std::string error_;
};

// ---------------------------------------------------------------------------
// seg_info()
// ---------------------------------------------------------------------------

bool validate_seg_info(const SegInfo& seg, int segments, bool strict,
                       std::string* error) {
  for (int i = 0; i < kMaxSegments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      const SegmentFeature& feature = seg.feature[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (i >= segments) {
        if (strict && (feature.enabled || feature.value != 0)) {
          if (error != nullptr) {
            *error = "seg_info row " + std::to_string(i) +
                     " is outside the coded segment count " +
                     std::to_string(segments);
          }
          return false;
        }
        continue;
      }
      if (!feature.enabled && feature.value != 0) {
        if (error != nullptr) {
          *error = "seg_info feature " + std::to_string(i) + "/" +
                   std::to_string(j) + " has a value but is disabled";
        }
        return false;
      }
      if (j == kSegLvlAltQ) {
        if (feature.value < -kSegAltQMax || feature.value > kSegAltQMax) {
          if (error != nullptr) {
            *error = "seg_info ALT_Q value " + std::to_string(feature.value) +
                     " is outside [-" + std::to_string(kSegAltQMax) + ", " +
                     std::to_string(kSegAltQMax) + "]";
          }
          return false;
        }
      } else if (feature.value != 0) {
        // SEG_LVL_SKIP and SEG_LVL_GLOBALMV have zero feature bits and a
        // zero maximum, so only zero is representable.
        if (error != nullptr) {
          *error = "seg_info feature " + std::to_string(j) +
                   " must be zero (zero coded bits)";
        }
        return false;
      }
    }
  }
  return true;
}

void write_seg_info(BitWriter& writer, const SegInfo& seg) {
  const int segments = seg.segments();
  for (int i = 0; i < segments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      const SegmentFeature& feature = seg.feature[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      writer.put_bit(feature.enabled);
      if (!feature.enabled) continue;
      if (j == kSegLvlAltQ) {
        // Segmentation_Feature_Bits[0] == 9, signed; su(10).
        put_su(writer, feature.value, kSegAltQBits + 1U);
      }
      // j == SEG_LVL_SKIP / SEG_LVL_GLOBALMV code zero additional bits.
    }
  }
}

bool read_seg_info(BitReader& reader, bool ext_seg, SegInfo* seg) {
  *seg = SegInfo{};
  seg->ext_seg = ext_seg;
  const int segments = seg->segments();
  for (int i = 0; i < segments; ++i) {
    for (int j = 0; j < kSegLvlMax; ++j) {
      bool enabled = false;
      if (!reader.read_bit(&enabled)) return false;
      SegmentFeature& feature = seg->feature[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      feature.enabled = enabled;
      if (!enabled) continue;
      if (j == kSegLvlAltQ) {
        std::int32_t raw = 0;
        if (!reader.read_su(kSegAltQBits + 1U, &raw)) return false;
        feature.value = std::clamp(raw, -kSegAltQMax, kSegAltQMax);
      } else {
        feature.value = 0;
      }
    }
  }
  return true;
}

bool check_header_ranges(const MultiFrameHeader& header, std::string* error) {
  if (header.seq_header_id >= kMaxSeqNum) {
    if (error != nullptr) {
      *error = "mfh_seq_header_id " + std::to_string(header.seq_header_id) +
               " >= MAX_SEQ_NUM (" + std::to_string(kMaxSeqNum) + ")";
    }
    return false;
  }
  if (header.mfh_id == 0 || header.mfh_id >= kMaxMfhNum) {
    if (error != nullptr) {
      *error = "mfh_id " + std::to_string(header.mfh_id) +
               " is outside [1, MAX_MFH_NUM) = [1, " +
               std::to_string(kMaxMfhNum) + ")";
    }
    return false;
  }
  if (header.tlayer_id > 3U) {
    if (error != nullptr) *error = "obu_tlayer_id must be in [0, 3]";
    return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

const char* to_string(FieldSource source) {
  switch (source) {
    case FieldSource::kAbsent:
      return "absent";
    case FieldSource::kSequence:
      return "sequence";
    case FieldSource::kMfh:
      return "mfh";
    case FieldSource::kExplicit:
      return "explicit";
    case FieldSource::kForced:
      return "forced";
  }
  return "?";
}

bool validate_multi_frame_header(const MultiFrameHeader& header,
                                 std::string* error) {
  if (error != nullptr) error->clear();
  if (!check_header_ranges(header, error)) return false;

  if (header.frame_size_present) {
    if (header.frame_width == 0 || header.frame_height == 0) {
      if (error != nullptr) {
        *error = "mfh_frame_size_present_flag is set but the width/height is 0";
      }
      return false;
    }
    const unsigned width_bits =
        header.frame_width_bits == 0
            ? min_frame_size_bits(header.frame_width)
            : header.frame_width_bits;
    const unsigned height_bits =
        header.frame_height_bits == 0
            ? min_frame_size_bits(header.frame_height)
            : header.frame_height_bits;
    if (width_bits < 1U || width_bits > 16U || height_bits < 1U ||
        height_bits > 16U) {
      if (error != nullptr) {
        *error = "mfh frame size bit widths must be in [1, 16]";
      }
      return false;
    }
    if (header.frame_width - 1U >= (1ULL << width_bits)) {
      if (error != nullptr) {
        *error = "mfh_frame_width_minus_1 does not fit in " +
                 std::to_string(width_bits) + " bits";
      }
      return false;
    }
    if (header.frame_height - 1U >= (1ULL << height_bits)) {
      if (error != nullptr) {
        *error = "mfh_frame_height_minus_1 does not fit in " +
                 std::to_string(height_bits) + " bits";
      }
      return false;
    }
  } else if (header.frame_width != 0 || header.frame_height != 0 ||
             header.frame_width_bits != 0 || header.frame_height_bits != 0) {
    if (error != nullptr) {
      *error =
          "frame size fields must be zero when mfh_frame_size_present_flag "
          "is 0";
    }
    return false;
  }

  if (!header.deblocking_filter_update) {
    for (int i = 0; i < 4; ++i) {
      if (header.apply_deblocking_filter[static_cast<std::size_t>(i)]) {
        if (error != nullptr) {
          *error =
              "mfh_apply_deblocking_filter is absent when "
              "mfh_deblocking_filter_update is 0";
        }
        return false;
      }
    }
  }

  if (!header.seg_info_present) {
    if (header.allow_seg_info_change) {
      if (error != nullptr) {
        *error =
            "mfh_allow_seg_info_change is absent when "
            "mfh_seg_info_present_flag is 0";
      }
      return false;
    }
    for (int i = 0; i < kMaxSegments; ++i) {
      for (int j = 0; j < kSegLvlMax; ++j) {
        const SegmentFeature& feature =
            header.seg_info.feature[static_cast<std::size_t>(i)]
                                   [static_cast<std::size_t>(j)];
        if (feature.enabled || feature.value != 0) {
          if (error != nullptr) {
            *error =
                "segment features are absent when "
                "mfh_seg_info_present_flag is 0";
          }
          return false;
        }
      }
    }
    return true;
  }

  return validate_seg_info(header.seg_info, header.seg_info.segments(), true,
                           error);
}

std::size_t write_multi_frame_header_syntax(BitWriter& writer,
                                            const MultiFrameHeader& header,
                                            std::string* error) {
  if (error != nullptr) error->clear();
  std::string local_error;
  if (!validate_multi_frame_header(header, &local_error)) {
    if (error != nullptr) *error = local_error;
    return 0;
  }
  const std::size_t start = writer.bit_size();

  put_uvlc(writer, header.seq_header_id);       // mfh_seq_header_id
  put_uvlc(writer, header.mfh_id - 1U);         // mfh_id_minus_1
  writer.put_bit(header.frame_size_present);    // mfh_frame_size_present_flag
  if (header.frame_size_present) {
    const unsigned width_bits =
        header.frame_width_bits == 0
            ? min_frame_size_bits(header.frame_width)
            : header.frame_width_bits;
    const unsigned height_bits =
        header.frame_height_bits == 0
            ? min_frame_size_bits(header.frame_height)
            : header.frame_height_bits;
    writer.put_bits(width_bits - 1U, 4U);   // mfh_frame_width_bits_minus_1
    writer.put_bits(height_bits - 1U, 4U);  // mfh_frame_height_bits_minus_1
    writer.put_bits(header.frame_width - 1U, width_bits);
    writer.put_bits(header.frame_height - 1U, height_bits);
  }
  writer.put_bit(header.deblocking_filter_update);
  if (header.deblocking_filter_update) {
    for (int i = 0; i < 4; ++i) {
      writer.put_bit(header.apply_deblocking_filter[static_cast<std::size_t>(i)]);
    }
  }
  writer.put_bit(header.seg_info_present);
  if (header.seg_info_present) {
    writer.put_bit(header.seg_info.ext_seg);        // mfh_ext_seg_flag
    writer.put_bit(header.allow_seg_info_change);   // mfh_allow_seg_info_change
    write_seg_info(writer, header.seg_info);        // seg_info(MaxSegments)
  }
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_multi_frame_header_obu(
    const MultiFrameHeader& header, std::string* error) {
  BitWriter syntax;
  std::string local_error;
  if (write_multi_frame_header_syntax(syntax, header, &local_error) == 0) {
    if (error != nullptr) *error = local_error;
    return {};
  }
  syntax.put_bit(true);  // §5.2.3 trailing_bits
  syntax.align_to_byte();
  // OBU_MULTI_FRAME_HEADER == 3, obu_extension_flag == 0 (§6.2.1 requires it),
  // obu_tlayer_id from the MFH.
  const m0::ObuHeader obu_header{
      static_cast<m0::ObuType>(kObuTypeMultiFrameHeader), header.tlayer_id,
      false, 0, 0};
  return m0::make_obu(obu_header, syntax.bytes());
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

bool parse_obu(std::span<const std::uint8_t> bytes, std::size_t offset,
               ObuView* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) {
    if (error != nullptr) *error = "parse_obu: null output";
    return false;
  }
  *out = ObuView{};
  out->offset = offset;
  if (offset >= bytes.size()) {
    if (error != nullptr) *error = "parse_obu: offset past end of buffer";
    return false;
  }
  std::size_t pos = offset;
  std::uint64_t size = 0;
  unsigned shift = 0;
  unsigned leb_bytes = 0;
  bool done = false;
  for (unsigned i = 0; i < 8; ++i) {
    if (pos >= bytes.size()) {
      if (error != nullptr) *error = "parse_obu: truncated leb128 size";
      return false;
    }
    const std::uint8_t byte = bytes[pos++];
    ++leb_bytes;
    size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    shift += 7U;
    if ((byte & 0x80U) == 0U) {
      done = true;
      break;
    }
  }
  if (!done) {
    if (error != nullptr) *error = "parse_obu: leb128 size exceeds 8 bytes";
    return false;
  }
  if (size > 0xFFFFFFFFULL) {
    if (error != nullptr) *error = "parse_obu: OBU size exceeds 32 bits";
    return false;
  }
  if (pos >= bytes.size()) {
    if (error != nullptr) *error = "parse_obu: truncated OBU header";
    return false;
  }
  const std::uint8_t header = bytes[pos++];
  out->extension_flag = (header & 0x80U) != 0U;
  out->obu_type = static_cast<std::uint8_t>((header >> 2U) & 0x1FU);
  out->tlayer_id = static_cast<std::uint8_t>(header & 0x03U);
  std::size_t header_bytes = pos - (offset + leb_bytes);
  if (out->extension_flag) {
    if (pos >= bytes.size()) {
      if (error != nullptr) *error = "parse_obu: truncated OBU extension";
      return false;
    }
    const std::uint8_t extension = bytes[pos++];
    out->mlayer_id = static_cast<std::uint8_t>((extension >> 5U) & 0x07U);
    out->xlayer_id = static_cast<std::uint8_t>(extension & 0x1FU);
    ++header_bytes;
  }
  if (size < header_bytes) {
    if (error != nullptr) {
      *error = "parse_obu: OBU size is smaller than its header";
    }
    return false;
  }
  const std::size_t payload_size = static_cast<std::size_t>(size) - header_bytes;
  if (payload_size > bytes.size() - pos) {
    if (error != nullptr) *error = "parse_obu: truncated OBU payload";
    return false;
  }
  out->payload_offset = pos;
  out->payload_size = payload_size;
  out->total_size = leb_bytes + static_cast<std::size_t>(size);
  if (out->total_size > bytes.size() - offset) {
    if (error != nullptr) *error = "parse_obu: OBU overruns the buffer";
    return false;
  }
  return true;
}

bool parse_multi_frame_header_payload(std::span<const std::uint8_t> payload,
                                      MultiFrameHeader* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) {
    if (error != nullptr) *error = "parse_multi_frame_header_payload: null out";
    return false;
  }
  *out = MultiFrameHeader{};
  BitReader reader(payload);

  std::uint32_t seq_header_id = 0;
  std::uint32_t mfh_id_minus_1 = 0;
  if (!reader.read_uvlc(&seq_header_id) ||
      !reader.read_uvlc(&mfh_id_minus_1)) {
    if (error != nullptr) *error = reader.error();
    return false;
  }
  out->seq_header_id = seq_header_id;
  out->mfh_id = mfh_id_minus_1 + 1U;

  bool size_present = false;
  if (!reader.read_bit(&size_present)) {
    if (error != nullptr) *error = reader.error();
    return false;
  }
  out->frame_size_present = size_present;
  if (size_present) {
    std::uint32_t width_bits_minus_1 = 0;
    std::uint32_t height_bits_minus_1 = 0;
    if (!reader.read_bits(4U, &width_bits_minus_1) ||
        !reader.read_bits(4U, &height_bits_minus_1)) {
      if (error != nullptr) *error = reader.error();
      return false;
    }
    const unsigned width_bits = width_bits_minus_1 + 1U;
    const unsigned height_bits = height_bits_minus_1 + 1U;
    out->frame_width_bits = static_cast<std::uint8_t>(width_bits);
    out->frame_height_bits = static_cast<std::uint8_t>(height_bits);
    std::uint32_t width_minus_1 = 0;
    std::uint32_t height_minus_1 = 0;
    if (!reader.read_bits(width_bits, &width_minus_1) ||
        !reader.read_bits(height_bits, &height_minus_1)) {
      if (error != nullptr) *error = reader.error();
      return false;
    }
    out->frame_width = width_minus_1 + 1U;
    out->frame_height = height_minus_1 + 1U;
  }

  bool deblocking_update = false;
  if (!reader.read_bit(&deblocking_update)) {
    if (error != nullptr) *error = reader.error();
    return false;
  }
  out->deblocking_filter_update = deblocking_update;
  if (deblocking_update) {
    for (int i = 0; i < 4; ++i) {
      bool flag = false;
      if (!reader.read_bit(&flag)) {
        if (error != nullptr) *error = reader.error();
        return false;
      }
      out->apply_deblocking_filter[static_cast<std::size_t>(i)] = flag;
    }
  }

  bool seg_present = false;
  if (!reader.read_bit(&seg_present)) {
    if (error != nullptr) *error = reader.error();
    return false;
  }
  out->seg_info_present = seg_present;
  if (seg_present) {
    bool ext_seg = false;
    bool allow_change = false;
    if (!reader.read_bit(&ext_seg) || !reader.read_bit(&allow_change)) {
      if (error != nullptr) *error = reader.error();
      return false;
    }
    out->allow_seg_info_change = allow_change;
    if (!read_seg_info(reader, ext_seg, &out->seg_info)) {
      if (error != nullptr) *error = reader.error();
      return false;
    }
  }

  if (!reader.read_trailing_bits()) {
    if (error != nullptr) *error = reader.error();
    return false;
  }

  std::string range_error;
  if (!check_header_ranges(*out, &range_error)) {
    if (error != nullptr) *error = range_error;
    return false;
  }
  return true;
}

bool parse_multi_frame_header_obu(std::span<const std::uint8_t> bytes,
                                  std::size_t offset, MultiFrameHeader* out,
                                  ObuView* obu, std::string* error) {
  if (error != nullptr) error->clear();
  ObuView view;
  if (!parse_obu(bytes, offset, &view, error)) return false;
  if (view.obu_type != kObuTypeMultiFrameHeader) {
    if (error != nullptr) {
      *error = "expected OBU_MULTI_FRAME_HEADER (3), found type " +
               std::to_string(view.obu_type);
    }
    return false;
  }
  if (view.extension_flag) {
    // §6.2.1: conforming bitstreams must have obu_extension_flag == 0.
    if (error != nullptr) {
      *error = "OBU_MULTI_FRAME_HEADER has obu_extension_flag == 1";
    }
    return false;
  }
  MultiFrameHeader header;
  const auto payload =
      bytes.subspan(view.payload_offset, view.payload_size);
  if (!parse_multi_frame_header_payload(payload, &header, error)) return false;
  header.tlayer_id = view.tlayer_id;
  if (out != nullptr) *out = header;
  if (obu != nullptr) *obu = view;
  return true;
}

// ---------------------------------------------------------------------------
// Store / scan
// ---------------------------------------------------------------------------

void MfhStore::reset() {
  slots_.fill(std::nullopt);
  count_ = 0;
}

bool MfhStore::insert(const MultiFrameHeader& header, std::string* error) {
  if (error != nullptr) error->clear();
  if (!check_header_ranges(header, error)) return false;
  if (header.seg_info_present) {
    if (!validate_seg_info(header.seg_info, header.seg_info.segments(), false,
                           error)) {
      return false;
    }
  }
  const std::size_t index = static_cast<std::size_t>(header.mfh_id);
  if (!slots_[index].has_value()) ++count_;
  slots_[index] = header;
  return true;
}

const MultiFrameHeader* MfhStore::find(std::uint32_t mfh_id) const {
  if (mfh_id == 0 || mfh_id >= kMaxMfhNum) return nullptr;
  const auto& slot = slots_[static_cast<std::size_t>(mfh_id)];
  return slot.has_value() ? &slot.value() : nullptr;
}

bool MfhStore::parse_and_insert(std::span<const std::uint8_t> bytes,
                                std::size_t offset, MultiFrameHeader* parsed,
                                std::string* error) {
  MultiFrameHeader header;
  if (!parse_multi_frame_header_obu(bytes, offset, &header, nullptr, error)) {
    return false;
  }
  if (!insert(header, error)) return false;
  if (parsed != nullptr) *parsed = header;
  return true;
}

bool scan_multi_frame_headers(std::span<const std::uint8_t> bytes,
                              MfhStore* store, std::size_t* mfh_count,
                              std::string* error) {
  if (error != nullptr) error->clear();
  if (store == nullptr) {
    if (error != nullptr) *error = "scan_multi_frame_headers: null store";
    return false;
  }
  std::size_t count = 0;
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    ObuView view;
    if (!parse_obu(bytes, offset, &view, error)) return false;
    if (view.obu_type == kObuTypeMultiFrameHeader) {
      MultiFrameHeader header;
      const auto payload =
          bytes.subspan(view.payload_offset, view.payload_size);
      if (!parse_multi_frame_header_payload(payload, &header, error)) {
        return false;
      }
      header.tlayer_id = view.tlayer_id;
      if (!store->insert(header, error)) return false;
      ++count;
    }
    if (view.total_size == 0) {
      if (error != nullptr) *error = "scan_multi_frame_headers: zero-size OBU";
      return false;
    }
    offset += view.total_size;
  }
  if (mfh_count != nullptr) *mfh_count = count;
  return true;
}

// ---------------------------------------------------------------------------
// Inheritance / precedence (§5.18)
// ---------------------------------------------------------------------------

LayerDependencyMap LayerDependencyMap::identity() {
  LayerDependencyMap map;
  for (int m = 0; m < 8; ++m) {
    for (int k = 0; k < 8; ++k) map.mlayer[m][k] = (m == k);
    for (int t = 0; t < 4; ++t) {
      for (int k = 0; k < 4; ++k) map.tlayer[m][t][k] = (t == k);
    }
  }
  return map;
}

bool resolve_frame_fields(const MfhStore& mfh_store,
                          const SequenceFrameFields& sequence,
                          const FrameHeaderFields& frame,
                          const LayerDependencyMap* deps,
                          ResolvedFrameFields* out, std::string* error) {
  if (error != nullptr) error->clear();
  if (out == nullptr) {
    if (error != nullptr) *error = "resolve_frame_fields: null out";
    return false;
  }
  *out = ResolvedFrameFields{};

  const MultiFrameHeader* mfh = nullptr;
  if (frame.cur_mfh_id > 0) {
    if (frame.cur_mfh_id >= kMaxMfhNum) {
      if (error != nullptr) {
        *error = "cur_mfh_id " + std::to_string(frame.cur_mfh_id) +
                 " >= MAX_MFH_NUM";
      }
      return false;
    }
    mfh = mfh_store.find(frame.cur_mfh_id);
    if (mfh == nullptr) {
      if (error != nullptr) {
        *error = "cur_mfh_id " + std::to_string(frame.cur_mfh_id) +
                 " references an MFH that is not available";
      }
      return false;
    }
    if (mfh->seq_header_id >= kMaxSeqNum ||
        !sequence.sequence_headers[mfh->seq_header_id]) {
      if (error != nullptr) {
        *error = "MFH " + std::to_string(frame.cur_mfh_id) +
                 " references sequence header " +
                 std::to_string(mfh->seq_header_id) +
                 " which is not available";
      }
      return false;
    }
    out->seq_header_id = mfh->seq_header_id;
    if (deps != nullptr) {
      const int m = frame.obu_mlayer_id;
      const int t = frame.obu_tlayer_id;
      if (m >= 8 || t >= 4) {
        if (error != nullptr) *error = "frame layer ids are out of range";
        return false;
      }
      if (!deps->mlayer[m][0]) {
        if (error != nullptr) {
          *error =
              "MLayerDependencyMap[frame_mlayer][MfhMLayerId] is not 1";
        }
        return false;
      }
      if (!deps->tlayer[m][t][mfh->tlayer_id]) {
        if (error != nullptr) {
          *error =
              "TLayerDependencyMap[frame_mlayer][frame_tlayer][MfhTLayerId] "
              "is not 1";
        }
        return false;
      }
    } else if (frame.obu_mlayer_id != 0 ||
               mfh->tlayer_id > frame.obu_tlayer_id) {
      // Without explicit dependency maps only the single-embedded-layer
      // default (layer t may depend on temporal layers <= t) is decidable.
      if (error != nullptr) {
        *error =
            "multi-layer or forward-temporal MFH reference requires an "
            "explicit LayerDependencyMap";
      }
      return false;
    }
  } else {
    out->seq_header_id = frame.seq_header_id;
    if (frame.seq_header_id >= kMaxSeqNum ||
        !sequence.sequence_headers[frame.seq_header_id]) {
      if (error != nullptr) {
        *error = "seq_header_id_in_frame_header " +
                 std::to_string(frame.seq_header_id) +
                 " is not available";
      }
      return false;
    }
  }

  // --- Frame size (§5.18.2, §5.18.4.1) -----------------------------------
  std::uint32_t inherited_width = sequence.max_frame_width;
  std::uint32_t inherited_height = sequence.max_frame_height;
  FieldSource inherited_source = FieldSource::kSequence;
  if (mfh != nullptr && mfh->frame_size_present) {
    inherited_width = mfh->frame_width;
    inherited_height = mfh->frame_height;
    inherited_source = FieldSource::kMfh;
    if (mfh->frame_width > sequence.max_frame_width ||
        mfh->frame_height > sequence.max_frame_height) {
      if (error != nullptr) {
        *error = "MFH frame size " + std::to_string(mfh->frame_width) + "x" +
                 std::to_string(mfh->frame_height) +
                 " exceeds the sequence maximum " +
                 std::to_string(sequence.max_frame_width) + "x" +
                 std::to_string(sequence.max_frame_height);
      }
      return false;
    }
  }
  if (frame.frame_size_override) {
    if (frame.frame_width == 0 || frame.frame_height == 0) {
      if (error != nullptr) *error = "frame_size_override with zero size";
      return false;
    }
    if (frame.frame_width > sequence.max_frame_width ||
        frame.frame_height > sequence.max_frame_height) {
      if (error != nullptr) {
        *error = "explicit frame size exceeds the sequence maximum";
      }
      return false;
    }
    out->frame_width = frame.frame_width;
    out->frame_height = frame.frame_height;
    out->frame_size_source = FieldSource::kExplicit;
  } else {
    out->frame_width = inherited_width;
    out->frame_height = inherited_height;
    out->frame_size_source = inherited_source;
  }

  // --- Deblocking (§5.18.5.2) --------------------------------------------
  out->deblocking_bits_present = false;
  if (frame.coded_lossless) {
    out->apply_deblocking_filter = {false, false, false, false};
    out->deblocking_source = FieldSource::kForced;
  } else if (mfh != nullptr && mfh->deblocking_filter_update) {
    out->apply_deblocking_filter[0] = mfh->apply_deblocking_filter[0];
    out->apply_deblocking_filter[1] = mfh->apply_deblocking_filter[1];
    out->apply_deblocking_filter[2] = false;
    out->apply_deblocking_filter[3] = false;
    if (!sequence.monochrome &&
        (out->apply_deblocking_filter[0] ||
         out->apply_deblocking_filter[1])) {
      out->apply_deblocking_filter[2] = mfh->apply_deblocking_filter[2];
      out->apply_deblocking_filter[3] = mfh->apply_deblocking_filter[3];
    }
    out->deblocking_source = FieldSource::kMfh;
  } else {
    out->apply_deblocking_filter[0] = frame.apply_deblocking_filter[0];
    out->apply_deblocking_filter[1] = frame.apply_deblocking_filter[1];
    out->apply_deblocking_filter[2] = false;
    out->apply_deblocking_filter[3] = false;
    if (!sequence.monochrome &&
        (out->apply_deblocking_filter[0] ||
         out->apply_deblocking_filter[1])) {
      out->apply_deblocking_filter[2] = frame.apply_deblocking_filter[2];
      out->apply_deblocking_filter[3] = frame.apply_deblocking_filter[3];
    }
    out->deblocking_source = FieldSource::kExplicit;
    out->deblocking_bits_present = true;
  }
  for (int i = 0; i < 4; ++i) {
    out->deblocking_delta_q_bits_present[i] =
        out->apply_deblocking_filter[static_cast<std::size_t>(i)];
  }

  // --- Segmentation (§5.18.7.1) ------------------------------------------
  out->max_segments = sequence.enable_ext_seg ? 16 : 8;
  out->segmentation_enabled = frame.segmentation_enabled;
  if (!frame.segmentation_enabled) {
    out->seg_info = SegInfo{};
    out->seg_info.ext_seg = sequence.enable_ext_seg;
    out->seg_info_source = FieldSource::kAbsent;
    return true;
  }

  bool have_seg_params = false;
  bool allow_change = false;
  const MultiFrameHeader* source_mfh = nullptr;
  if (mfh != nullptr && mfh->seg_info_present) {
    have_seg_params = (mfh->seg_info.ext_seg == sequence.enable_ext_seg);
    allow_change = have_seg_params && mfh->allow_seg_info_change;
    source_mfh = mfh;
  } else if (sequence.seq_seg_info_present) {
    have_seg_params = true;
    allow_change = sequence.seq_allow_seg_info_change;
  }

  if (allow_change) {
    out->reuse_seg_info_bit_present = true;
    out->reuse_seg_info = frame.reuse_seg_info;
  } else {
    out->reuse_seg_info = have_seg_params;
  }

  if (out->reuse_seg_info) {
    if (source_mfh != nullptr && have_seg_params) {
      out->seg_info = source_mfh->seg_info;
      out->seg_info_source = FieldSource::kMfh;
    } else {
      out->seg_info = sequence.seq_seg_info;
      out->seg_info_source = FieldSource::kSequence;
    }
    out->seg_info.ext_seg = sequence.enable_ext_seg;
  } else {
    out->seg_info = frame.seg_info;
    out->seg_info.ext_seg = sequence.enable_ext_seg;
    out->seg_info_source = FieldSource::kExplicit;
    if (!validate_seg_info(out->seg_info, out->max_segments, true, error)) {
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::frame::mfh
