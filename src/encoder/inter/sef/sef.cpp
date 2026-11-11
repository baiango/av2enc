#include "encoder/inter/sef/sef.h"

#include "bitstream/m0_headers.h"

#include <cassert>
#include <limits>

namespace av2enc::inter::sef {
namespace {

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

// MSB-first reader over one OBU payload. Kept local: the shared bitstream
// directory has a writer only, and this parser is deliberately tiny.
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool read_bit(bool& bit) {
    if (bit_pos_ >= bytes_.size() * 8U) return false;
    const std::size_t byte = bit_pos_ / 8U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    bit = ((bytes_[byte] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool read_bits(unsigned num_bits, std::uint32_t& value) {
    if (num_bits > 32U) return false;
    std::uint32_t out = 0;
    for (unsigned i = 0; i < num_bits; ++i) {
      bool bit = false;
      if (!read_bit(bit)) return false;
      out = (out << 1U) | (bit ? 1U : 0U);
    }
    value = out;
    return true;
  }

  // §4.11.3 uvlc(). leadingZeros < 32 is a conformance requirement.
  bool read_uvlc(std::uint32_t& value) {
    unsigned leading_zeros = 0;
    bool done = false;
    while (true) {
      if (!read_bit(done)) return false;
      if (done) break;
      ++leading_zeros;
      if (leading_zeros >= 32U) return false;
    }
    std::uint32_t payload = 0;
    if (leading_zeros > 0U && !read_bits(leading_zeros, payload)) {
      return false;
    }
    const std::uint64_t decoded =
        static_cast<std::uint64_t>(payload) +
        ((1ULL << leading_zeros) - 1ULL);
    if (decoded > std::numeric_limits<std::uint32_t>::max()) return false;
    value = static_cast<std::uint32_t>(decoded);
    return true;
  }

  std::size_t bit_position() const { return bit_pos_; }
  std::size_t total_bits() const { return bytes_.size() * 8U; }

  // Reads from `bit_position()` to the end of the payload: one trailing 1 bit
  // followed only by zero bits (§5.2.3). Extra zero bytes are legal padding.
  bool check_trailing_bits() {
    bool one = false;
    if (!read_bit(one) || !one) return false;
    while (bit_pos_ < total_bits()) {
      bool bit = false;
      if (!read_bit(bit) || bit) return false;
    }
    return true;
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

void set_error(SefError* error, SefError value) {
  if (error != nullptr) *error = value;
}

bool fail(SefError* error, SefError value) {
  set_error(error, value);
  return false;
}

}  // namespace

const char* sef_error_name(SefError error) {
  switch (error) {
    case SefError::kOk:
      return "ok";
    case SefError::kBadNumRefFrames:
      return "bad num_ref_frames";
    case SefError::kBadOrderHintBits:
      return "bad order_hint_bits";
    case SefError::kBadMapIdx:
      return "frame_to_show_map_idx out of range";
    case SefError::kInvalidReference:
      return "reference slot is not valid";
    case SefError::kReferenceAlreadyShown:
      return "reference frame was already shown";
    case SefError::kReferenceNotHidden:
      return "reference frame is not hidden (immediate/implicit output set)";
    case SefError::kOrderHintNotSignalable:
      return "order hint is not signalable with the given OrderHintBits";
    case SefError::kOrderHintMismatch:
      return "explicit sef_order_hint does not match the reference OrderHint";
    case SefError::kHintOutOfRange:
      return "derived OrderHint is out of range";
    case SefError::kUnsupportedFilmGrain:
      return "film grain parameters are not supported by this writer";
    case SefError::kUnsupportedMfh:
      return "multi-frame-header SEF (cur_mfh_id != 0) is not supported";
    case SefError::kTruncated:
      return "truncated SEF payload";
    case SefError::kBadTrailingBits:
      return "bad SEF trailing bits";
    case SefError::kBadObuHeader:
      return "not a SEF OBU";
    case SefError::kBadObuLength:
      return "bad SEF OBU length";
  }
  return "unknown SEF error";
}

const char* sef_kind_name(SefKind kind) {
  switch (kind) {
    case SefKind::kLeading:
      return "OBU_LEADING_SEF";
    case SefKind::kRegular:
      return "OBU_REGULAR_SEF";
  }
  return "unknown SEF kind";
}

unsigned ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  unsigned bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0U) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

std::size_t sef_frame_header_bits(const SefParams& params,
                                  const SefHeader& header) {
  auto uvlc_bits = [](std::uint32_t value) {
    unsigned leading_zeros = 0;
    while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
      ++leading_zeros;
    }
    return static_cast<std::size_t>(2U * leading_zeros + 1U);
  };
  std::size_t bits = uvlc_bits(header.cur_mfh_id) +
                     uvlc_bits(header.seq_header_id) +
                     ceil_log2(params.num_ref_frames) + 1U;
  if (!header.derive_sef_order_hint) {
    bits += params.order_hint_bits;
  }
  return bits;
}

std::size_t write_sef_frame_header(BitWriter& writer, const SefParams& params,
                                   const SefHeader& header) {
  const std::size_t start = writer.bit_size();
  assert(params.num_ref_frames >= 1U &&
         params.num_ref_frames <= kNumRefFramesMax);
  assert(params.order_hint_bits >= 1U && params.order_hint_bits <= 16U);
  assert(header.frame_to_show_map_idx < params.num_ref_frames);
  assert(!params.film_grain_params_present);

  put_uvlc(writer, header.cur_mfh_id);  // cur_mfh_id
  put_uvlc(writer, header.seq_header_id);  // seq_header_id_in_frame_header
  // ShowExistingFrame = is_sef() == 1; no bit is coded.
  const unsigned index_bits = ceil_log2(params.num_ref_frames);
  writer.put_bits(header.frame_to_show_map_idx, index_bits);
  writer.put_bit(header.derive_sef_order_hint);
  if (!header.derive_sef_order_hint) {
    assert(header.sef_order_hint < (1U << params.order_hint_bits));
    writer.put_bits(header.sef_order_hint, params.order_hint_bits);
  }
  // §5.18.2 returns here: refresh_frame_flags = 0 and
  // immediate_output_frame = 1 are inferred, and film_grain_config() codes
  // nothing when film_grain_params_present == 0.
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_sef_obu(const SefParams& params,
                                       const SefHeader& header) {
  assert(!params.film_grain_params_present);
  BitWriter writer;
  write_sef_frame_header(writer, params, header);
  writer.put_bit(true);  // §5.2.3 trailing_one_bit
  writer.align_to_byte();
  const std::vector<std::uint8_t> payload = writer.take_bytes();
  const av2enc::m0::ObuHeader obu_header{
      static_cast<av2enc::m0::ObuType>(static_cast<std::uint8_t>(header.kind)),
      0, false, 0, 0};
  return av2enc::m0::make_obu(obu_header, payload);
}

std::vector<std::uint8_t> make_sef_temporal_unit(const SefParams& params,
                                                 const SefHeader& header) {
  const std::vector<std::uint8_t> td =
      av2enc::m0::make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> obu = make_sef_obu(params, header);
  std::vector<std::uint8_t> unit;
  unit.reserve(td.size() + obu.size());
  unit.insert(unit.end(), td.begin(), td.end());
  unit.insert(unit.end(), obu.begin(), obu.end());
  return unit;
}

bool parse_sef_frame_header(std::span<const std::uint8_t> payload,
                            const SefParams& params, SefHeader& header,
                            std::size_t& bits_consumed, SefError* error) {
  set_error(error, SefError::kOk);
  if (params.num_ref_frames < 1U ||
      params.num_ref_frames > kNumRefFramesMax) {
    return fail(error, SefError::kBadNumRefFrames);
  }
  if (params.order_hint_bits < 1U || params.order_hint_bits > 16U) {
    return fail(error, SefError::kBadOrderHintBits);
  }
  if (params.film_grain_params_present) {
    return fail(error, SefError::kUnsupportedFilmGrain);
  }
  if (payload.empty()) return fail(error, SefError::kTruncated);

  BitReader reader(payload);
  std::uint32_t value = 0;
  if (!reader.read_uvlc(value)) return fail(error, SefError::kTruncated);
  header.cur_mfh_id = value;
  if (!reader.read_uvlc(value)) return fail(error, SefError::kTruncated);
  header.seq_header_id = value;

  const unsigned index_bits = ceil_log2(params.num_ref_frames);
  if (!reader.read_bits(index_bits, value)) {
    return fail(error, SefError::kTruncated);
  }
  if (value >= params.num_ref_frames) {
    return fail(error, SefError::kBadMapIdx);
  }
  header.frame_to_show_map_idx = static_cast<std::uint8_t>(value);

  bool derive = false;
  if (!reader.read_bit(derive)) return fail(error, SefError::kTruncated);
  header.derive_sef_order_hint = derive;
  header.sef_order_hint = 0;
  if (!derive) {
    if (!reader.read_bits(params.order_hint_bits, value)) {
      return fail(error, SefError::kTruncated);
    }
    header.sef_order_hint = value;
  }

  bits_consumed = reader.bit_position();
  if (!reader.check_trailing_bits()) {
    return fail(error, SefError::kBadTrailingBits);
  }
  return true;
}

bool parse_sef_obu(std::span<const std::uint8_t> bytes, const SefParams& params,
                   SefHeader& header, std::size_t& bytes_consumed,
                   SefError* error) {
  set_error(error, SefError::kOk);
  bytes_consumed = 0;
  if (bytes.empty()) return fail(error, SefError::kTruncated);

  // Annex B.2 leb128(num_bytes_in_obu).
  std::uint64_t obu_size = 0;
  unsigned shift = 0;
  std::size_t pos = 0;
  while (true) {
    if (pos >= bytes.size()) return fail(error, SefError::kTruncated);
    const std::uint8_t byte = bytes[pos++];
    obu_size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) break;
    shift += 7U;
    if (shift >= 35U) return fail(error, SefError::kBadObuLength);
  }
  if (obu_size == 0U || obu_size > bytes.size() - pos) {
    return fail(error, SefError::kBadObuLength);
  }

  const std::span<const std::uint8_t> obu = bytes.subspan(pos, obu_size);
  std::size_t header_bytes = 1;
  const bool extension_flag = (obu[0] & 0x80U) != 0U;
  const std::uint8_t obu_type =
      static_cast<std::uint8_t>((obu[0] >> 2U) & 0x1FU);
  if (obu_type != kObuLeadingSef && obu_type != kObuRegularSef) {
    return fail(error, SefError::kBadObuHeader);
  }
  if (extension_flag) header_bytes = 2;
  if (obu.size() <= header_bytes) return fail(error, SefError::kTruncated);
  header.kind = obu_type == kObuLeadingSef ? SefKind::kLeading
                                           : SefKind::kRegular;

  std::size_t syntax_bits = 0;
  if (!parse_sef_frame_header(obu.subspan(header_bytes), params, header,
                              syntax_bits, error)) {
    return false;
  }
  bytes_consumed = pos + obu_size;
  return true;
}

const RefSlotState* SlotMap::slot(std::uint8_t index) const {
  if (index >= num_ref_frames || index >= kNumRefFramesMax) return nullptr;
  return &slots[index];
}

std::uint32_t SlotMap::max_disp_order_hint() const {
  std::uint32_t max_disp = 0;
  for (std::uint8_t i = 0; i < num_ref_frames; ++i) {
    const RefSlotState& state = slots[i];
    if (!state.valid) continue;
    // get_max_disp_order_hint(onlyShowable = 1): a hidden frame does not
    // contribute to the display-order window.
    if (!state.immediate_output_frame && !state.implicit_output_frame) {
      continue;
    }
    if (state.order_hint > max_disp) max_disp = state.order_hint;
  }
  return max_disp;
}

void apply_refresh(SlotMap& map, const RefreshFrame& frame) {
  const std::uint8_t count =
      map.num_ref_frames > kNumRefFramesMax ? kNumRefFramesMax
                                            : map.num_ref_frames;
  bool first = true;
  for (std::uint8_t i = 0; i < count; ++i) {
    if ((frame.refresh_frame_flags & (1U << i)) == 0U) continue;
    RefSlotState& state = map.slots[i];
    state = RefSlotState{};
    // §7.23: RefValid[i] = (KEY_FRAME || SWITCH_FRAME) ? first : 1. The frame
    // store is still updated by the reference code, but the slot is not a
    // usable reference for anything except the first refreshed slot.
    if (!first && frame.key_frame) {
      first = false;
      continue;
    }
    state.valid = true;
    state.order_hint = frame.order_hint;
    state.frame_store_id = frame.frame_store_id;
    state.immediate_output_frame = frame.immediate_output_frame;
    state.implicit_output_frame = frame.implicit_output_frame;
    state.shown = false;
    first = false;
  }
}

void mark_shown(SlotMap& map, std::uint8_t index) {
  if (index < map.num_ref_frames && index < kNumRefFramesMax) {
    map.slots[index].shown = true;
  }
}

bool validate_sef_reference(const SlotMap& map, const SefParams& params,
                            const SefHeader& header, SefError* error) {
  set_error(error, SefError::kOk);
  if (params.num_ref_frames < 1U ||
      params.num_ref_frames > kNumRefFramesMax) {
    return fail(error, SefError::kBadNumRefFrames);
  }
  if (params.order_hint_bits < 1U || params.order_hint_bits > 16U) {
    return fail(error, SefError::kBadOrderHintBits);
  }
  if (header.cur_mfh_id != 0U) {
    // Multi-frame-header SEFs would need the MFH state; this writer only
    // supports the M2 single-sequence-header form.
    return fail(error, SefError::kUnsupportedMfh);
  }
  if (header.frame_to_show_map_idx >= map.num_ref_frames) {
    return fail(error, SefError::kBadMapIdx);
  }
  const RefSlotState& state = map.slots[header.frame_to_show_map_idx];
  if (!state.valid) return fail(error, SefError::kInvalidReference);

  if (header.derive_sef_order_hint) {
    if (state.immediate_output_frame || state.implicit_output_frame) {
      return fail(error, SefError::kReferenceNotHidden);
    }
    if (state.shown) {
      return fail(error, SefError::kReferenceAlreadyShown);
    }
  } else {
    const std::uint32_t max_disp = map.max_disp_order_hint();
    const std::uint32_t hint = disp_order_hint(
        header.sef_order_hint, max_disp, params.order_hint_bits);
    if (hint >= (1U << (kDisplayOrderHintBits - 1U))) {
      return fail(error, SefError::kHintOutOfRange);
    }
  }
  return true;
}

std::uint32_t disp_order_hint(std::uint32_t order_hint_lsbs,
                              std::uint32_t max_disp_order_hint,
                              std::uint8_t order_hint_bits) {
  if (order_hint_bits == 0U) return order_hint_lsbs;
  const std::uint32_t half = 1U << (order_hint_bits - 1U);
  std::uint32_t disp = order_hint_lsbs;
  const std::int64_t offset = static_cast<std::int64_t>(max_disp_order_hint) -
                              static_cast<std::int64_t>(half) -
                              static_cast<std::int64_t>(order_hint_lsbs);
  if (offset >= 0) {
    const std::uint32_t steps =
        static_cast<std::uint32_t>(offset >> order_hint_bits);
    disp += (steps + 1U) << order_hint_bits;
  }
  return disp;
}

bool sef_order_hint_lsbs(std::uint32_t target_order_hint,
                         std::uint32_t max_disp_order_hint,
                         std::uint8_t order_hint_bits,
                         std::uint32_t& order_hint_lsbs) {
  if (order_hint_bits < 1U || order_hint_bits > 16U) return false;
  const std::uint32_t modulus = 1U << order_hint_bits;
  for (std::uint32_t candidate = 0; candidate < modulus; ++candidate) {
    if (disp_order_hint(candidate, max_disp_order_hint, order_hint_bits) ==
        target_order_hint) {
      order_hint_lsbs = candidate;
      return true;
    }
  }
  return false;
}

bool make_show_header(const SlotMap& map, const SefParams& params,
                      std::uint8_t index, SefKind kind, SefHeader& header,
                      SefError* error) {
  set_error(error, SefError::kOk);
  if (index >= map.num_ref_frames) {
    return fail(error, SefError::kBadMapIdx);
  }
  const RefSlotState& state = map.slots[index];
  if (!state.valid) return fail(error, SefError::kInvalidReference);

  header = SefHeader{};
  header.kind = kind;
  header.frame_to_show_map_idx = index;
  // Prefer derive=1: it is the only form that needs no order-hint bits and the
  // only form §6.17.2 allows for a hidden frame.
  if (!state.immediate_output_frame && !state.implicit_output_frame &&
      !state.shown) {
    header.derive_sef_order_hint = true;
    return true;
  }

  header.derive_sef_order_hint = false;
  const std::uint32_t max_disp = map.max_disp_order_hint();
  std::uint32_t lsbs = 0;
  if (!sef_order_hint_lsbs(state.order_hint, max_disp,
                           params.order_hint_bits, lsbs)) {
    return fail(error, SefError::kOrderHintNotSignalable);
  }
  header.sef_order_hint = lsbs;
  if (disp_order_hint(header.sef_order_hint, max_disp,
                      params.order_hint_bits) != state.order_hint) {
    return fail(error, SefError::kOrderHintNotSignalable);
  }
  return true;
}

bool validate_dav2d_compat(const SlotMap& map, const SefParams& params,
                           const SefHeader& header, SefError* error) {
  set_error(error, SefError::kOk);
  const RefSlotState* state = map.slot(header.frame_to_show_map_idx);
  if (state == nullptr) return fail(error, SefError::kBadMapIdx);
  if (!state->valid) return fail(error, SefError::kInvalidReference);
  if (header.derive_sef_order_hint) return true;
  const std::uint32_t max_disp = map.max_disp_order_hint();
  const std::uint32_t decoded = disp_order_hint(
      header.sef_order_hint, max_disp, params.order_hint_bits);
  // dav2d: `if (cpoc != poc) goto error` in the SEF parse path.
  if (decoded != state->order_hint) {
    return fail(error, SefError::kOrderHintMismatch);
  }
  return true;
}

bool dav2d_needs_output_invisible(const RefSlotState& slot) {
  // dav2d drops an output whose POC equals dpb_poc. A re-shown already-output
  // frame is exactly that case. The hidden/derive=1 path has no prior output
  // for that POC and therefore needs no extra flag.
  return slot.shown;
}

}  // namespace av2enc::inter::sef
