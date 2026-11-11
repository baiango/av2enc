#include "encoder/m1/scc_flags/scc_flags.h"

#include "bitstream/m0_headers.h"
#include "encoder/m1/m1_headers.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace av2enc::m1::scc_flags {
namespace {

using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;

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

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  assert(n >= 2U);
  assert(value < n);
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

// For change_bvp_drl: find the ns(2) code that produces `target`.
bool change_bvp_drl_code(int seq_value, int target, int* code) {
  if (code == nullptr || target < 0) return false;
  const int candidate = target > seq_value ? target - 1 : target;
  if (candidate < 0 || candidate > 1) return false;
  const int decoded =
      candidate >= seq_value ? candidate + 1 : candidate;
  if (decoded != target) return false;
  *code = candidate;
  return true;
}

// ---------------------------------------------------------------------------
// Bit capture/splice helpers
// ---------------------------------------------------------------------------

struct BitImage {
  std::vector<std::uint8_t> bytes;
  std::size_t bits = 0;
};

BitImage capture(const BitWriter& writer) {
  BitImage image;
  const auto span = writer.bytes();
  image.bytes.assign(span.begin(), span.end());
  image.bits = writer.bit_size();
  return image;
}

bool bit_at(const BitImage& image, std::size_t index) {
  assert(index < image.bits);
  const std::size_t byte = index >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(index & 7U);
  return ((image.bytes[byte] >> shift) & 1U) != 0U;
}

void append_bits(BitWriter& writer, const BitImage& image, std::size_t from) {
  for (std::size_t i = from; i < image.bits; ++i) {
    writer.put_bit(bit_at(image, i));
  }
}

// ---------------------------------------------------------------------------
// OBU walking for read_still_frame_scc()
// ---------------------------------------------------------------------------

bool read_leb128(std::span<const std::uint8_t> bytes, std::size_t& offset,
                 std::uint32_t& value) {
  value = 0;
  unsigned shift = 0;
  for (;;) {
    if (offset >= bytes.size() || shift >= 32U) return false;
    const std::uint8_t byte = bytes[offset++];
    value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) return true;
    shift += 7U;
  }
}

bool find_obu_payload(std::span<const std::uint8_t> bytes, ObuType type,
                      std::span<const std::uint8_t>& payload,
                      std::string& error) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    std::uint32_t obu_size = 0;
    if (!read_leb128(bytes, offset, obu_size)) {
      error = "malformed leb128 OBU size";
      return false;
    }
    if (obu_size == 0U || offset + obu_size > bytes.size()) {
      error = "OBU size runs past the temporal unit";
      return false;
    }
    const std::uint8_t header = bytes[offset];
    const unsigned obu_type = (header >> 2U) & 0x1FU;
    if (obu_type == static_cast<unsigned>(type)) {
      // obu_extension_flag == 0 for every still writer in this tree.
      if ((header & 0x80U) != 0U) {
        error = "OBU extension is not supported by the still reader";
        return false;
      }
      payload = bytes.subspan(offset + 1U, obu_size - 1U);
      return true;
    }
    offset += obu_size;
  }
  error = "required OBU not found";
  return false;
}

// ---------------------------------------------------------------------------
// Frame header assembly
// ---------------------------------------------------------------------------

// Writes is_first_tile_group + frame_header() with the SCC region authored by
// this module and the M1 tail spliced from m1::write_frame_header().
bool write_still_frame_header(BitWriter& writer, const M1Config& config,
                              const StillSccConfig& scc,
                              int seq_max_bvp_drl_bits_minus_1,
                              bool allow_frame_max_bvp_drl_bits,
                              SequenceSccState* sequence_state,
                              FrameSccState* frame_state, std::string& error) {
  error.clear();
  if (!scc.sequence.single_picture_header_flag) {
    error =
        "assemble_still_picture() needs single_picture_header_flag = 1; the "
        "M1 sequence writer is single-picture only";
    return false;
  }

  SequenceSccState seq_state;
  std::size_t seq_bits = 0;
  {
    BitWriter scratch;
    if (!write_sequence_scc_config(scratch, scc.sequence, &seq_state, &seq_bits,
                                   error)) {
      return false;
    }
    if (seq_bits != 0U) {
      error = "single-picture §5.4.7 must code zero bits";
      return false;
    }
  }

  // M1's frame header prefix: cur_mfh_id uvlc(0), seq_header_id uvlc(0) and
  // the two all-off screen-content bits.  The SCC region starts after the two
  // UVLC ids; the tail starts after those four bits.
  BitWriter m1;
  m1::write_frame_header(m1, config);
  const BitImage m1_image = capture(m1);
  if (m1_image.bits < 4U) {
    error = "M1 frame header is shorter than its fixed prefix";
    return false;
  }
  const bool prefix_ok = bit_at(m1_image, 0) && bit_at(m1_image, 1) &&
                         !bit_at(m1_image, 2) && !bit_at(m1_image, 3);
  if (!prefix_ok) {
    error =
        "M1 frame header prefix changed (expected uvlc(0), uvlc(0), "
        "allow_screen_content_tools=0, allow_intrabc=0)";
    return false;
  }

  writer.put_bit(true);  // is_first_tile_group
  put_uvlc(writer, 0);   // cur_mfh_id
  put_uvlc(writer, 0);   // seq_header_id_in_frame_header

  FrameSccContext context;
  context.frame_is_intra = true;  // M1 still
  FrameSccState state;
  std::size_t scc_bits = 0;
  if (!write_frame_scc_params(writer, seq_state, context, scc.frame,
                              seq_max_bvp_drl_bits_minus_1,
                              allow_frame_max_bvp_drl_bits, &state, &scc_bits,
                              error)) {
    return false;
  }
  (void)scc_bits;

  append_bits(writer, m1_image, 4U);

  if (sequence_state != nullptr) *sequence_state = seq_state;
  if (frame_state != nullptr) *frame_state = state;
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// §5.4.7
// ---------------------------------------------------------------------------

bool validate_sequence_scc_config(const SequenceSccConfig& config,
                                  std::string& error) {
  error.clear();
  if (config.single_picture_header_flag) {
    if (config.seq_choose_screen_content_tools ||
        config.seq_force_screen_content_tools ||
        config.seq_choose_integer_mv || config.seq_force_integer_mv) {
      error =
          "single_picture_header_flag=1 removes every seq_*_screen_content/"
          "integer_mv field; the values are inferred SELECT";
      return false;
    }
    return true;
  }
  if (config.seq_choose_screen_content_tools &&
      config.seq_force_screen_content_tools) {
    error =
        "seq_force_screen_content_tools is not present when "
        "seq_choose_screen_content_tools=1";
    return false;
  }
  const bool force_screen =
      config.seq_choose_screen_content_tools ||
      config.seq_force_screen_content_tools;
  if (!force_screen &&
      (config.seq_choose_integer_mv || config.seq_force_integer_mv)) {
    error =
        "integer-MV fields are not present when "
        "seq_force_screen_content_tools=0";
    return false;
  }
  if (config.seq_choose_integer_mv && config.seq_force_integer_mv) {
    error =
        "seq_force_integer_mv is not present when seq_choose_integer_mv=1";
    return false;
  }
  return true;
}

SequenceSccState derive_sequence_scc_state(const SequenceSccConfig& config) {
  SequenceSccState state;
  state.single_picture_header_flag = config.single_picture_header_flag;
  if (config.single_picture_header_flag) {
    state.seq_force_screen_content_tools = kSelectScreenContentTools;
    state.seq_force_integer_mv = kSelectIntegerMv;
    return state;
  }
  state.coded_seq_choose_screen_content_tools = true;
  if (config.seq_choose_screen_content_tools) {
    state.seq_force_screen_content_tools = kSelectScreenContentTools;
  } else {
    state.coded_seq_force_screen_content_tools = true;
    state.seq_force_screen_content_tools =
        config.seq_force_screen_content_tools ? 1 : 0;
  }
  if (state.seq_force_screen_content_tools > 0) {
    state.coded_seq_choose_integer_mv = true;
    if (config.seq_choose_integer_mv) {
      state.seq_force_integer_mv = kSelectIntegerMv;
    } else {
      state.coded_seq_force_integer_mv = true;
      state.seq_force_integer_mv = config.seq_force_integer_mv ? 1 : 0;
    }
  } else {
    state.seq_force_integer_mv = kSelectIntegerMv;
  }
  return state;
}

bool write_sequence_scc_config(BitWriter& writer,
                               const SequenceSccConfig& config,
                               SequenceSccState* state, std::size_t* bits,
                               std::string& error) {
  if (!validate_sequence_scc_config(config, error)) return false;
  const std::size_t start = writer.bit_size();
  const SequenceSccState derived = derive_sequence_scc_state(config);
  if (!config.single_picture_header_flag) {
    writer.put_bit(config.seq_choose_screen_content_tools);
    if (!config.seq_choose_screen_content_tools) {
      writer.put_bit(config.seq_force_screen_content_tools);
    }
    if (derived.seq_force_screen_content_tools > 0) {
      writer.put_bit(config.seq_choose_integer_mv);
      if (!config.seq_choose_integer_mv) {
        writer.put_bit(config.seq_force_integer_mv);
      }
    }
  }
  if (state != nullptr) *state = derived;
  if (bits != nullptr) *bits = writer.bit_size() - start;
  error.clear();
  return true;
}

bool read_sequence_scc_config(BitReader& reader,
                              bool single_picture_header_flag,
                              SequenceSccState& state, std::string& error) {
  error.clear();
  state = SequenceSccState{};
  state.single_picture_header_flag = single_picture_header_flag;
  if (single_picture_header_flag) {
    state.seq_force_screen_content_tools = kSelectScreenContentTools;
    state.seq_force_integer_mv = kSelectIntegerMv;
    return true;
  }
  bool bit = false;
  if (!reader.get_bit(bit)) {
    error = "truncated seq_choose_screen_content_tools";
    return false;
  }
  state.coded_seq_choose_screen_content_tools = true;
  if (bit) {
    state.seq_force_screen_content_tools = kSelectScreenContentTools;
  } else {
    if (!reader.get_bit(bit)) {
      error = "truncated seq_force_screen_content_tools";
      return false;
    }
    state.coded_seq_force_screen_content_tools = true;
    state.seq_force_screen_content_tools = bit ? 1 : 0;
  }
  if (state.seq_force_screen_content_tools > 0) {
    if (!reader.get_bit(bit)) {
      error = "truncated seq_choose_integer_mv";
      return false;
    }
    state.coded_seq_choose_integer_mv = true;
    if (bit) {
      state.seq_force_integer_mv = kSelectIntegerMv;
    } else {
      if (!reader.get_bit(bit)) {
        error = "truncated seq_force_integer_mv";
        return false;
      }
      state.coded_seq_force_integer_mv = true;
      state.seq_force_integer_mv = bit ? 1 : 0;
    }
  } else {
    state.seq_force_integer_mv = kSelectIntegerMv;
  }
  return true;
}

// ---------------------------------------------------------------------------
// §5.18.3.3 / §5.18.3.4
// ---------------------------------------------------------------------------

bool validate_frame_scc_config(const SequenceSccState& sequence,
                               const FrameSccContext& context,
                               const FrameSccConfig& config,
                               int seq_max_bvp_drl_bits_minus_1,
                               bool allow_frame_max_bvp_drl_bits,
                               std::string& error) {
  error.clear();
  const bool present = frame_scc_present(context);
  if (!present) {
    if (config.allow_screen_content_tools || config.force_integer_mv ||
        config.allow_intrabc || config.allow_global_intrabc ||
        config.allow_local_intrabc || config.change_bvp_drl) {
      error =
          "screen_content_params()/intrabc_params() are absent for this "
          "frame (TIP-as-output, BRU-inactive inter, or bridge); all flags "
          "are inferred 0";
      return false;
    }
    if (config.max_bvp_drl_bits_minus_1 != seq_max_bvp_drl_bits_minus_1) {
      error =
          "max_bvp_drl_bits_minus_1 is not coded and is inferred from the "
          "sequence header";
      return false;
    }
    return true;
  }

  bool allow_screen = false;
  if (sequence.seq_force_screen_content_tools == kSelectScreenContentTools) {
    allow_screen = config.allow_screen_content_tools;
  } else {
    allow_screen = sequence.seq_force_screen_content_tools != 0;
    if (config.allow_screen_content_tools != allow_screen) {
      error =
          "allow_screen_content_tools is forced by "
          "seq_force_screen_content_tools";
      return false;
    }
  }

  if (allow_screen) {
    if (sequence.seq_force_integer_mv == kSelectIntegerMv) {
      // force_integer_mv bit is coded; either value is legal.
    } else {
      const bool forced = sequence.seq_force_integer_mv != 0;
      if (config.force_integer_mv != forced) {
        error = "force_integer_mv is forced by seq_force_integer_mv";
        return false;
      }
    }
  } else if (config.force_integer_mv) {
    error = "force_integer_mv is inferred 0 when allow_screen_content_tools=0";
    return false;
  }

  if (!config.allow_intrabc) {
    if (config.allow_global_intrabc || config.allow_local_intrabc) {
      error =
          "allow_global_intrabc/allow_local_intrabc are absent when "
          "allow_intrabc=0";
      return false;
    }
    if (config.change_bvp_drl) {
      error = "change_bvp_drl is absent when allow_intrabc=0";
      return false;
    }
    if (config.max_bvp_drl_bits_minus_1 != seq_max_bvp_drl_bits_minus_1) {
      error =
          "max_bvp_drl_bits_minus_1 must equal "
          "seq_max_bvp_drl_bits_minus_1 when change_bvp_drl is absent";
      return false;
    }
    return true;
  }

  if (context.frame_is_intra) {
    if (!config.allow_global_intrabc && !config.allow_local_intrabc) {
      // global=0 infers local=1; asking for both off is a contradiction.
      error =
          "allow_local_intrabc is inferred 1 when allow_global_intrabc=0";
      return false;
    }
  } else {
    if (config.allow_global_intrabc) {
      error = "allow_global_intrabc is inferred 0 for an inter frame";
      return false;
    }
    if (!config.allow_local_intrabc) {
      error = "allow_local_intrabc is inferred 1 for an inter frame";
      return false;
    }
  }

  if (seq_max_bvp_drl_bits_minus_1 < 0) {
    error = "seq_max_bvp_drl_bits_minus_1 must be non-negative";
    return false;
  }
  if (!allow_frame_max_bvp_drl_bits || !config.change_bvp_drl) {
    if (config.change_bvp_drl) {
      error =
          "change_bvp_drl is absent when allow_frame_max_bvp_drl_bits=0";
      return false;
    }
    if (config.max_bvp_drl_bits_minus_1 != seq_max_bvp_drl_bits_minus_1) {
      error =
          "max_bvp_drl_bits_minus_1 must equal "
          "seq_max_bvp_drl_bits_minus_1 when change_bvp_drl=0";
      return false;
    }
  } else {
    int code = 0;
    if (!change_bvp_drl_code(seq_max_bvp_drl_bits_minus_1,
                             config.max_bvp_drl_bits_minus_1, &code)) {
      error =
          "max_bvp_drl_bits_minus_1 is not reachable by change_bvp_drl "
          "ns(2) from seq_max_bvp_drl_bits_minus_1";
      return false;
    }
    (void)code;
  }
  return true;
}

FrameSccState derive_frame_scc_state(const SequenceSccState& sequence,
                                     const FrameSccContext& context,
                                     const FrameSccConfig& config,
                                     int seq_max_bvp_drl_bits_minus_1,
                                     bool allow_frame_max_bvp_drl_bits) {
  FrameSccState state;
  state.scc_present = frame_scc_present(context);
  if (!state.scc_present) {
    state.max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1;
    state.IsIntraBC = false;
    state.codes_frame_mv_precision = false;
    state.frame_mv_precision = kMvPrecisionOnePel;
    state.intrabc_base_mv_precision = kMvPrecisionQuarterPel;
    state.intrabc_precision_coded = false;
    return state;
  }

  state.coded_allow_screen_content_tools =
      sequence.seq_force_screen_content_tools == kSelectScreenContentTools;
  if (state.coded_allow_screen_content_tools) {
    state.allow_screen_content_tools = config.allow_screen_content_tools;
  } else {
    state.allow_screen_content_tools =
        sequence.seq_force_screen_content_tools != 0;
  }

  state.coded_force_integer_mv =
      state.allow_screen_content_tools &&
      sequence.seq_force_integer_mv == kSelectIntegerMv;
  if (state.allow_screen_content_tools) {
    if (state.coded_force_integer_mv) {
      state.force_integer_mv = config.force_integer_mv;
    } else {
      state.force_integer_mv = sequence.seq_force_integer_mv != 0;
    }
  }

  state.coded_allow_intrabc = true;
  state.allow_intrabc = config.allow_intrabc;
  if (state.allow_intrabc) {
    if (context.frame_is_intra) {
      state.coded_allow_global_intrabc = true;
      state.allow_global_intrabc = config.allow_global_intrabc;
      if (state.allow_global_intrabc) {
        state.coded_allow_local_intrabc = true;
        state.allow_local_intrabc = config.allow_local_intrabc;
      } else {
        state.allow_local_intrabc = true;
      }
    } else {
      state.allow_global_intrabc = false;
      state.allow_local_intrabc = true;
    }
    state.max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1;
    state.coded_change_bvp_drl =
        allow_frame_max_bvp_drl_bits && state.allow_intrabc;
    if (config.change_bvp_drl) {
      state.max_bvp_drl_bits_minus_1 = config.max_bvp_drl_bits_minus_1;
    }
  } else {
    state.max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1;
  }

  state.IsIntraBC = state.allow_intrabc;
  state.intrabc_base_mv_precision = intrabc_mv_precision(state);
  state.intrabc_precision_coded =
      state.allow_intrabc && !state.force_integer_mv;

  if (context.frame_is_intra) {
    // §5.18.2 initialises FrameMvPrecision = MV_PRECISION_ONE_PEL before the
    // intra branch and never codes the frame-precision bits there.
    state.codes_frame_mv_precision = false;
    state.frame_mv_precision = kMvPrecisionOnePel;
  } else if (!state.scc_present) {
    state.codes_frame_mv_precision = false;
    state.frame_mv_precision = kMvPrecisionOnePel;
  } else {
    state.codes_frame_mv_precision = !state.force_integer_mv;
    state.frame_mv_precision =
        state.force_integer_mv ? kMvPrecisionOnePel : 0;
  }
  return state;
}

bool write_frame_scc_params(BitWriter& writer,
                            const SequenceSccState& sequence,
                            const FrameSccContext& context,
                            const FrameSccConfig& config,
                            int seq_max_bvp_drl_bits_minus_1,
                            bool allow_frame_max_bvp_drl_bits,
                            FrameSccState* state, std::size_t* bits,
                            std::string& error) {
  if (!validate_frame_scc_config(sequence, context, config,
                                 seq_max_bvp_drl_bits_minus_1,
                                 allow_frame_max_bvp_drl_bits, error)) {
    return false;
  }
  const std::size_t start = writer.bit_size();
  const FrameSccState derived =
      derive_frame_scc_state(sequence, context, config,
                             seq_max_bvp_drl_bits_minus_1,
                             allow_frame_max_bvp_drl_bits);
  if (!derived.scc_present) {
    if (state != nullptr) *state = derived;
    if (bits != nullptr) *bits = 0;
    return true;
  }

  // screen_content_params()
  if (sequence.seq_force_screen_content_tools == kSelectScreenContentTools) {
    writer.put_bit(derived.allow_screen_content_tools);
  }
  if (derived.allow_screen_content_tools) {
    if (sequence.seq_force_integer_mv == kSelectIntegerMv) {
      writer.put_bit(derived.force_integer_mv);
    }
  }

  // intrabc_params()
  writer.put_bit(derived.allow_intrabc);
  if (derived.allow_intrabc) {
    if (context.frame_is_intra) {
      writer.put_bit(derived.allow_global_intrabc);
      if (derived.allow_global_intrabc) {
        writer.put_bit(derived.allow_local_intrabc);
      }
    }
    if (allow_frame_max_bvp_drl_bits) {
      writer.put_bit(derived.coded_change_bvp_drl
                         ? config.change_bvp_drl
                         : false);
      if (derived.coded_change_bvp_drl && config.change_bvp_drl) {
        int code = 0;
        if (!change_bvp_drl_code(seq_max_bvp_drl_bits_minus_1,
                                 derived.max_bvp_drl_bits_minus_1, &code)) {
          error = "change_bvp_drl target is not representable";
          return false;
        }
        put_ns(writer, 2U, static_cast<std::uint32_t>(code));
      }
    }
  }

  if (state != nullptr) *state = derived;
  if (bits != nullptr) *bits = writer.bit_size() - start;
  return true;
}

bool read_frame_scc_params(BitReader& reader, const SequenceSccState& sequence,
                           const FrameSccContext& context,
                           int seq_max_bvp_drl_bits_minus_1,
                           bool allow_frame_max_bvp_drl_bits,
                           FrameSccState& state, std::string& error) {
  error.clear();
  state = FrameSccState{};
  state.scc_present = frame_scc_present(context);
  if (!state.scc_present) {
    state.max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1;
    state.IsIntraBC = false;
    state.codes_frame_mv_precision = false;
    state.frame_mv_precision = kMvPrecisionOnePel;
    state.intrabc_base_mv_precision = kMvPrecisionQuarterPel;
    return true;
  }

  bool bit = false;
  if (sequence.seq_force_screen_content_tools == kSelectScreenContentTools) {
    if (!reader.get_bit(bit)) {
      error = "truncated allow_screen_content_tools";
      return false;
    }
    state.coded_allow_screen_content_tools = true;
    state.allow_screen_content_tools = bit;
  } else {
    state.allow_screen_content_tools =
        sequence.seq_force_screen_content_tools != 0;
  }

  if (state.allow_screen_content_tools) {
    if (sequence.seq_force_integer_mv == kSelectIntegerMv) {
      if (!reader.get_bit(bit)) {
        error = "truncated force_integer_mv";
        return false;
      }
      state.coded_force_integer_mv = true;
      state.force_integer_mv = bit;
    } else {
      state.force_integer_mv = sequence.seq_force_integer_mv != 0;
    }
  }

  if (!reader.get_bit(bit)) {
    error = "truncated allow_intrabc";
    return false;
  }
  state.coded_allow_intrabc = true;
  state.allow_intrabc = bit;
  state.max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1;

  if (state.allow_intrabc) {
    if (context.frame_is_intra) {
      if (!reader.get_bit(bit)) {
        error = "truncated allow_global_intrabc";
        return false;
      }
      state.coded_allow_global_intrabc = true;
      state.allow_global_intrabc = bit;
      if (state.allow_global_intrabc) {
        if (!reader.get_bit(bit)) {
          error = "truncated allow_local_intrabc";
          return false;
        }
        state.coded_allow_local_intrabc = true;
        state.allow_local_intrabc = bit;
      } else {
        state.allow_local_intrabc = true;
      }
    } else {
      state.allow_global_intrabc = false;
      state.allow_local_intrabc = true;
    }

    if (allow_frame_max_bvp_drl_bits) {
      if (!reader.get_bit(bit)) {
        error = "truncated change_bvp_drl";
        return false;
      }
      state.coded_change_bvp_drl = true;
      if (bit) {
        std::uint32_t code = 0;
        if (!reader.get_ns(2U, code)) {
          error = "truncated max_bvp_drl_bits_minus_1";
          return false;
        }
        int value = static_cast<int>(code);
        if (value >= seq_max_bvp_drl_bits_minus_1) ++value;
        state.max_bvp_drl_bits_minus_1 = value;
      }
    }
  }

  state.IsIntraBC = state.allow_intrabc;
  state.intrabc_base_mv_precision = intrabc_mv_precision(state);
  state.intrabc_precision_coded =
      state.allow_intrabc && !state.force_integer_mv;

  if (context.frame_is_intra || !state.scc_present) {
    state.codes_frame_mv_precision = false;
    state.frame_mv_precision = kMvPrecisionOnePel;
  } else {
    state.codes_frame_mv_precision = !state.force_integer_mv;
    state.frame_mv_precision =
        state.force_integer_mv ? kMvPrecisionOnePel : 0;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Block legality
// ---------------------------------------------------------------------------

bool palette_allowed_for_block(const FrameSccState& state, int plane,
                               int y_mode, int bsize, int block_w,
                               int block_h) noexcept {
  return plane == 0 && state.allow_screen_content_tools && y_mode == 0 &&
         bsize >= kBlock8x8 && bsize < kBlockSizeCount && block_w > 0 &&
         block_h > 0 && block_w <= kMaxScreenContentBlockDim &&
         block_h <= kMaxScreenContentBlockDim;
}

bool intrabc_allowed_for_block(const FrameSccState& state, int bsize,
                               int block_w, int block_h,
                               bool tree_is_chroma_part,
                               bool region_is_mixed,
                               bool frame_is_intra) noexcept {
  if (!state.allow_intrabc || tree_is_chroma_part) return false;
  if (bsize < 0 || bsize >= kBlockSizeCount) return false;
  if (bsize == kBlock64x64) return false;
  if (block_w <= 0 || block_h <= 0 ||
      block_w > kMaxScreenContentBlockDim ||
      block_h > kMaxScreenContentBlockDim) {
    return false;
  }
  if (!frame_is_intra && !region_is_mixed) return false;
  return true;
}

bool morph_pred_allowed(const FrameSccState& state, bool frame_is_intra,
                        bool enable_bawp) noexcept {
  return frame_is_intra && state.allow_screen_content_tools && enable_bawp;
}

int intrabc_mv_precision(const FrameSccState& state) noexcept {
  return state.force_integer_mv ? kMvPrecisionOnePel
                                : kMvPrecisionQuarterPel;
}

// ---------------------------------------------------------------------------
// BitReader
// ---------------------------------------------------------------------------

bool BitReader::get_bit(bool& out) noexcept {
  out = false;
  if (!ok_ || bit_pos_ >= bytes_.size() * 8U) {
    ok_ = false;
    return false;
  }
  const std::size_t byte = bit_pos_ >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
  out = ((bytes_[byte] >> shift) & 1U) != 0U;
  ++bit_pos_;
  return true;
}

bool BitReader::get_bits(unsigned count, std::uint32_t& out) noexcept {
  out = 0;
  if (count > 32U) {
    ok_ = false;
    return false;
  }
  for (unsigned i = 0; i < count; ++i) {
    bool bit = false;
    if (!get_bit(bit)) return false;
    out = (out << 1U) | (bit ? 1U : 0U);
  }
  return true;
}

bool BitReader::get_uvlc(std::uint32_t& out) noexcept {
  out = 0;
  unsigned leading_zeros = 0;
  bool bit = false;
  for (;;) {
    if (!get_bit(bit)) return false;
    if (bit) break;
    ++leading_zeros;
    if (leading_zeros >= 32U) {
      ok_ = false;
      return false;
    }
  }
  std::uint32_t payload = 0;
  if (!get_bits(leading_zeros, payload)) return false;
  const std::uint64_t value = (1ULL << leading_zeros) - 1ULL + payload;
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    ok_ = false;
    return false;
  }
  out = static_cast<std::uint32_t>(value);
  return true;
}

bool BitReader::get_ns(std::uint32_t n, std::uint32_t& out) noexcept {
  out = 0;
  if (n < 2U) {
    ok_ = false;
    return false;
  }
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  std::uint32_t value = 0;
  if (!get_bits(w - 1U, value)) return false;
  if (value < m) {
    out = value;
    return true;
  }
  bool bit = false;
  if (!get_bit(bit)) return false;
  out = (value << 1U) + (bit ? 1U : 0U) - m;
  return true;
}

bool BitReader::skip_bits(std::size_t count) noexcept {
  if (!ok_ || bit_pos_ + count > bytes_.size() * 8U) {
    ok_ = false;
    return false;
  }
  bit_pos_ += count;
  return true;
}

// ---------------------------------------------------------------------------
// Still-picture assembly
// ---------------------------------------------------------------------------

bool still_frame_header_size(const M1Config& config,
                             const StillSccConfig& scc,
                             int seq_max_bvp_drl_bits_minus_1,
                             bool allow_frame_max_bvp_drl_bits,
                             std::size_t* bits, std::size_t* bytes,
                             FrameSccState* state, std::string& error) {
  BitWriter writer;
  FrameSccState frame_state;
  if (!write_still_frame_header(writer, config, scc,
                                seq_max_bvp_drl_bits_minus_1,
                                allow_frame_max_bvp_drl_bits, nullptr,
                                &frame_state, error)) {
    return false;
  }
  const std::size_t bit_count = writer.bit_size();
  if (bits != nullptr) *bits = bit_count;
  if (bytes != nullptr) *bytes = (bit_count + 7U) / 8U;
  if (state != nullptr) *state = frame_state;
  return true;
}

bool assemble_still_picture(const M1Config& config, const StillSccConfig& scc,
                            int seq_max_bvp_drl_bits_minus_1,
                            bool allow_frame_max_bvp_drl_bits,
                            std::span<const std::uint8_t> tile_payload,
                            StillSccResult& out, std::string& error) {
  error.clear();
  out = StillSccResult{};

  BitWriter writer;
  SequenceSccState sequence_state;
  FrameSccState frame_state;
  if (!write_still_frame_header(writer, config, scc,
                                seq_max_bvp_drl_bits_minus_1,
                                allow_frame_max_bvp_drl_bits, &sequence_state,
                                &frame_state, error)) {
    return false;
  }
  out.frame_header_bits = writer.bit_size();
  writer.align_to_byte();
  out.frame_header_bytes = writer.bit_size() / 8U;

  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());

  const std::vector<std::uint8_t> td =
      av2enc::m0::make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq =
      m1::make_sequence_header_obu(config);
  const std::vector<std::uint8_t> clk = av2enc::m0::make_obu(
      ObuHeader{ObuType::kClosedLoopKey}, payload);

  out.frame.reserve(td.size() + seq.size() + clk.size());
  out.frame.insert(out.frame.end(), td.begin(), td.end());
  out.frame.insert(out.frame.end(), seq.begin(), seq.end());
  out.frame.insert(out.frame.end(), clk.begin(), clk.end());

  out.sequence_state = sequence_state;
  out.frame_state = frame_state;
  out.tile_bytes = tile_payload.size();
  return true;
}

bool read_still_frame_scc(std::span<const std::uint8_t> frame,
                          const SequenceSccState& sequence,
                          const FrameSccContext& context,
                          int seq_max_bvp_drl_bits_minus_1,
                          bool allow_frame_max_bvp_drl_bits,
                          FrameSccState& state, std::string& error) {
  error.clear();
  std::span<const std::uint8_t> payload;
  if (!find_obu_payload(frame, ObuType::kClosedLoopKey, payload, error)) {
    return false;
  }
  BitReader reader(payload);
  bool first = false;
  if (!reader.get_bit(first)) {
    error = "truncated is_first_tile_group";
    return false;
  }
  if (!first) {
    error = "only a first tile group carries a frame header";
    return false;
  }
  std::uint32_t id = 0;
  if (!reader.get_uvlc(id) || !reader.get_uvlc(id)) {
    error = "truncated frame-header OBU id prefix";
    return false;
  }
  return read_frame_scc_params(reader, sequence, context,
                               seq_max_bvp_drl_bits_minus_1,
                               allow_frame_max_bvp_drl_bits, state, error);
}

}  // namespace scc_flags
