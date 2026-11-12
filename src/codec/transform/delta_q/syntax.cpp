// §5.4.8 sequence delta-Q syntax and §5.18.6.1 quantization_params().
#include "codec/transform/delta_q/dcdeltaq.h"

#include <algorithm>

namespace av2enc::dcdeltaq {
namespace {

void set_error(std::string* error, const char* what) {
  if (error != nullptr) *error = what;
}

bool in_range(int value, int lo, int hi) {
  return value >= lo && value <= hi;
}

int base_delta_raw(int value) { return value - kDeltaDcQuantMin; }

bool read_bit(deltaq::BitReader& reader, bool& value, std::string* error,
              const char* what) {
  if (!reader.read_bit(value)) {
    set_error(error, what);
    return false;
  }
  return true;
}

bool read_bits(deltaq::BitReader& reader, unsigned n, std::uint32_t& value,
               std::string* error, const char* what) {
  if (!reader.read_bits(n, value)) {
    set_error(error, what);
    return false;
  }
  return true;
}

}  // namespace

SequenceDeltaQ normalized_sequence_delta_q(const SequenceDeltaQ& seq) {
  SequenceDeltaQ out = seq;
  if (out.monochrome) {
    out.separate_uv_delta_q = false;
    out.uv_dc_delta_q_enabled = false;
    out.uv_ac_delta_q_enabled = false;
    out.base_uv_dc_delta_q = 0;
    out.base_uv_ac_delta_q = 0;
  }
  if (out.equal_ac_dc_q) {
    out.y_dc_delta_q_enabled = false;
    out.uv_dc_delta_q_enabled = false;
    out.base_y_dc_delta_q = 0;
    out.base_uv_dc_delta_q = out.base_uv_ac_delta_q;
  }
  return out;
}

bool validate_sequence_delta_q(const SequenceDeltaQ& seq, std::string* error) {
  if (!in_range(seq.base_y_dc_delta_q, kDeltaDcQuantMin, kDeltaDcQuantMax) ||
      !in_range(seq.base_uv_dc_delta_q, kDeltaDcQuantMin, kDeltaDcQuantMax) ||
      !in_range(seq.base_uv_ac_delta_q, kDeltaDcQuantMin, kDeltaDcQuantMax)) {
    set_error(error, "sequence delta-Q base value outside [-23, 8]");
    return false;
  }
  if (seq.monochrome) {
    if (seq.separate_uv_delta_q || seq.uv_dc_delta_q_enabled ||
        seq.uv_ac_delta_q_enabled || seq.base_uv_dc_delta_q != 0 ||
        seq.base_uv_ac_delta_q != 0) {
      set_error(error, "monochrome sequence must not code UV delta-Q fields");
      return false;
    }
  }
  if (seq.equal_ac_dc_q) {
    // §5.4.8 does not code the Y-DC or UV-DC base fields when equal_ac_dc_q
    // is 1; BaseUVDcDeltaQ is copied from BaseUVAcDeltaQ.
    if (seq.y_dc_delta_q_enabled || seq.uv_dc_delta_q_enabled) {
      set_error(error, "equal_ac_dc_q = 1 forbids the DC delta enables");
      return false;
    }
    if (seq.base_y_dc_delta_q != 0) {
      set_error(error, "equal_ac_dc_q = 1 forces BaseYDcDeltaQ = 0");
      return false;
    }
    // BaseUVDcDeltaQ is not coded when equal_ac_dc_q = 1; it is derived
    // from BaseUVAcDeltaQ by normalized_sequence_delta_q().  Accept any
    // caller value here and normalize it.
  }
  return true;
}

bool validate_frame_delta_q(const SequenceDeltaQ& seq_in,
                            const FrameDeltaQ& frame, std::string* error) {
  const SequenceDeltaQ seq = normalized_sequence_delta_q(seq_in);
  const int values[5] = {frame.delta_q_y_dc, frame.delta_q_u_dc,
                         frame.delta_q_v_dc, frame.delta_q_u_ac,
                         frame.delta_q_v_ac};
  for (int value : values) {
    if (!in_range(value, kFrameDeltaMin, kFrameDeltaMax)) {
      set_error(error, "frame delta-Q outside su(7) range [-64, 63]");
      return false;
    }
  }
  if (!seq.y_dc_delta_q_enabled && frame.delta_q_y_dc != 0) {
    set_error(error, "DeltaQYDc is not signalled (y_dc_delta_q_enabled = 0)");
    return false;
  }
  if (seq.monochrome) {
    if (frame.delta_q_u_dc != 0 || frame.delta_q_v_dc != 0 ||
        frame.delta_q_u_ac != 0 || frame.delta_q_v_ac != 0) {
      set_error(error, "monochrome frame must not signal UV deltas");
      return false;
    }
    return true;
  }
  if (!seq.equal_ac_dc_q && !seq.uv_dc_delta_q_enabled &&
      (frame.delta_q_u_dc != 0 || frame.delta_q_v_dc != 0)) {
    set_error(error, "UV DC delta is not signalled");
    return false;
  }
  if (!seq.uv_ac_delta_q_enabled &&
      (frame.delta_q_u_ac != 0 || frame.delta_q_v_ac != 0)) {
    set_error(error, "UV AC delta is not signalled");
    return false;
  }
  if (seq.equal_ac_dc_q &&
      (frame.delta_q_u_dc != frame.delta_q_u_ac ||
       frame.delta_q_v_dc != frame.delta_q_v_ac)) {
    set_error(error, "equal_ac_dc_q = 1 forces DeltaQU/VDc = DeltaQU/VAc");
    return false;
  }
  if (!seq.separate_uv_delta_q &&
      (frame.delta_q_u_dc != frame.delta_q_v_dc ||
       frame.delta_q_u_ac != frame.delta_q_v_ac)) {
    set_error(error, "separate_uv_delta_q = 0 forces DeltaQV* = DeltaQU*");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// read_delta_q() (§5.18.6.3)
// ---------------------------------------------------------------------------

std::size_t write_delta_q(BitWriter& writer, int value, std::string* error) {
  if (!in_range(value, kFrameDeltaMin, kFrameDeltaMax)) {
    set_error(error, "delta_q outside su(7) range [-64, 63]");
    return 0;
  }
  if (value == 0) {
    writer.put_bit(false);  // delta_coded = 0
    return 1;
  }
  writer.put_bit(true);  // delta_coded = 1
  // su(7): the bottom 7 bits of the signed value.
  writer.put_bits(static_cast<std::uint64_t>(value) & 0x7fU, 7U);
  return 8;
}

bool read_delta_q(deltaq::BitReader& reader, int& value, bool* coded,
                  std::string* error) {
  bool delta_coded = false;
  if (!read_bit(reader, delta_coded, error, "read_delta_q: delta_coded")) {
    return false;
  }
  if (coded != nullptr) *coded = delta_coded;
  if (!delta_coded) {
    value = 0;
    return true;
  }
  std::uint32_t raw = 0;
  if (!read_bits(reader, 7, raw, error, "read_delta_q: su(7)")) {
    return false;
  }
  value = static_cast<int>(raw);
  if ((value & 0x40) != 0) value -= 0x80;
  return true;
}

// ---------------------------------------------------------------------------
// Sequence delta fields (§5.4.8)
// ---------------------------------------------------------------------------

std::size_t write_sequence_delta_q_fields(BitWriter& writer,
                                          const SequenceDeltaQ& seq_in,
                                          std::string* error) {
  if (!validate_sequence_delta_q(seq_in, error)) return 0;
  const SequenceDeltaQ seq = normalized_sequence_delta_q(seq_in);
  const std::size_t start = writer.bit_size();

  if (!seq.monochrome) {
    writer.put_bit(seq.separate_uv_delta_q);
  }  // Monochrome infers separate_uv_delta_q = 0; no bit is coded.

  writer.put_bit(seq.equal_ac_dc_q);
  if (!seq.equal_ac_dc_q) {
    writer.put_bits(static_cast<std::uint64_t>(base_delta_raw(
                        seq.base_y_dc_delta_q)),
                    kDeltaDcQuantBits);
    writer.put_bit(seq.y_dc_delta_q_enabled);
  }
  if (!seq.monochrome) {
    if (!seq.equal_ac_dc_q) {
      writer.put_bits(static_cast<std::uint64_t>(base_delta_raw(
                          seq.base_uv_dc_delta_q)),
                      kDeltaDcQuantBits);
      writer.put_bit(seq.uv_dc_delta_q_enabled);
    }
    writer.put_bits(static_cast<std::uint64_t>(base_delta_raw(
                        seq.base_uv_ac_delta_q)),
                    kDeltaDcQuantBits);
    writer.put_bit(seq.uv_ac_delta_q_enabled);
  }
  return writer.bit_size() - start;
}

bool read_sequence_delta_q_fields(deltaq::BitReader& reader, bool monochrome,
                                  SequenceDeltaQ& seq, std::string* error) {
  seq = SequenceDeltaQ{};
  seq.monochrome = monochrome;
  bool flag = false;
  if (!monochrome) {
    if (!read_bit(reader, flag, error,
                  "read_sequence_delta_q_fields: separate_uv_delta_q")) {
      return false;
    }
    seq.separate_uv_delta_q = flag;
  }
  if (!read_bit(reader, flag, error,
                "read_sequence_delta_q_fields: equal_ac_dc_q")) {
    return false;
  }
  seq.equal_ac_dc_q = flag;
  std::uint32_t raw = 0;
  if (!seq.equal_ac_dc_q) {
    if (!read_bits(reader, kDeltaDcQuantBits, raw, error,
                   "read_sequence_delta_q_fields: base_y_dc_delta_q")) {
      return false;
    }
    seq.base_y_dc_delta_q = kDeltaDcQuantMin + static_cast<int>(raw);
    if (!read_bit(reader, flag, error,
                  "read_sequence_delta_q_fields: y_dc_delta_q_enabled")) {
      return false;
    }
    seq.y_dc_delta_q_enabled = flag;
  }
  if (!monochrome) {
    if (!seq.equal_ac_dc_q) {
      if (!read_bits(reader, kDeltaDcQuantBits, raw, error,
                     "read_sequence_delta_q_fields: base_uv_dc_delta_q")) {
        return false;
      }
      seq.base_uv_dc_delta_q = kDeltaDcQuantMin + static_cast<int>(raw);
      if (!read_bit(reader, flag, error,
                    "read_sequence_delta_q_fields: uv_dc_delta_q_enabled")) {
        return false;
      }
      seq.uv_dc_delta_q_enabled = flag;
    }
    if (!read_bits(reader, kDeltaDcQuantBits, raw, error,
                   "read_sequence_delta_q_fields: base_uv_ac_delta_q")) {
      return false;
    }
    seq.base_uv_ac_delta_q = kDeltaDcQuantMin + static_cast<int>(raw);
    if (!read_bit(reader, flag, error,
                  "read_sequence_delta_q_fields: uv_ac_delta_q_enabled")) {
      return false;
    }
    seq.uv_ac_delta_q_enabled = flag;
    if (seq.equal_ac_dc_q) seq.base_uv_dc_delta_q = seq.base_uv_ac_delta_q;
  }
  return true;
}

std::size_t write_sequence_transform_quant_entropy_config(
    BitWriter& writer, const SeqTransformQuantEntropyConfig& config,
    const SequenceDeltaQ& seq, std::string* error) {
  if (seq.monochrome != config.monochrome) {
    set_error(error, "sequence delta-Q monochrome mismatch");
    return 0;
  }
  const std::size_t start = writer.bit_size();

  writer.put_bit(config.enable_fsc);
  if (config.enable_fsc) {
    // enable_idtx_intra = 1 is inferred.
  } else {
    writer.put_bit(config.enable_idtx_intra);
  }
  writer.put_bit(config.enable_intra_ist);
  writer.put_bit(config.enable_inter_ist);
  if (!config.monochrome) writer.put_bit(config.enable_chroma_dctonly);
  if (!config.single_picture_header_flag) {
    writer.put_bit(config.enable_inter_ddt);
  }
  writer.put_bit(config.reduced_tx_part_set);
  if (!config.monochrome) writer.put_bit(config.enable_cctx);
  writer.put_bit(config.enable_tcq);
  if (config.enable_tcq && !config.single_picture_header_flag) {
    writer.put_bit(config.choose_tcq_per_frame);
  }
  if (!(config.enable_tcq && !config.choose_tcq_per_frame)) {
    writer.put_bit(config.enable_parity_hiding);
  }
  if (config.single_picture_header_flag) {
    // enable_avg_cdf = 1, avg_cdf_type = 1 are inferred.
  } else {
    writer.put_bit(config.enable_avg_cdf);
    if (config.enable_avg_cdf) writer.put_bit(config.avg_cdf_type);
  }

  const std::size_t delta_bits =
      write_sequence_delta_q_fields(writer, seq, error);
  if (delta_bits == 0) return 0;
  return writer.bit_size() - start;
}

bool read_sequence_transform_quant_entropy_config(
    deltaq::BitReader& reader, SeqTransformQuantEntropyConfig& config,
    SequenceDeltaQ& seq, std::string* error) {
  bool flag = false;
  if (!read_bit(reader, flag, error, "sequence_config: enable_fsc")) {
    return false;
  }
  config.enable_fsc = flag;
  if (config.enable_fsc) {
    config.enable_idtx_intra = true;
  } else {
    if (!read_bit(reader, flag, error,
                  "sequence_config: enable_idtx_intra")) {
      return false;
    }
    config.enable_idtx_intra = flag;
  }
  if (!read_bit(reader, flag, error, "sequence_config: enable_intra_ist")) {
    return false;
  }
  config.enable_intra_ist = flag;
  if (!read_bit(reader, flag, error, "sequence_config: enable_inter_ist")) {
    return false;
  }
  config.enable_inter_ist = flag;
  if (config.monochrome) {
    config.enable_chroma_dctonly = false;
  } else {
    if (!read_bit(reader, flag, error,
                  "sequence_config: enable_chroma_dctonly")) {
      return false;
    }
    config.enable_chroma_dctonly = flag;
  }
  if (!config.single_picture_header_flag) {
    if (!read_bit(reader, flag, error,
                  "sequence_config: enable_inter_ddt")) {
      return false;
    }
    config.enable_inter_ddt = flag;
  } else {
    config.enable_inter_ddt = false;
  }
  if (!read_bit(reader, flag, error,
                "sequence_config: reduced_tx_part_set")) {
    return false;
  }
  config.reduced_tx_part_set = flag;
  if (config.monochrome) {
    config.enable_cctx = false;
  } else {
    if (!read_bit(reader, flag, error, "sequence_config: enable_cctx")) {
      return false;
    }
    config.enable_cctx = flag;
  }
  if (!read_bit(reader, flag, error, "sequence_config: enable_tcq")) {
    return false;
  }
  config.enable_tcq = flag;
  config.choose_tcq_per_frame = false;
  if (config.enable_tcq && !config.single_picture_header_flag) {
    if (!read_bit(reader, flag, error,
                  "sequence_config: choose_tcq_per_frame")) {
      return false;
    }
    config.choose_tcq_per_frame = flag;
  }
  config.enable_parity_hiding = false;
  if (!(config.enable_tcq && !config.choose_tcq_per_frame)) {
    if (!read_bit(reader, flag, error,
                  "sequence_config: enable_parity_hiding")) {
      return false;
    }
    config.enable_parity_hiding = flag;
  }
  if (config.single_picture_header_flag) {
    config.enable_avg_cdf = true;
    config.avg_cdf_type = true;
  } else {
    if (!read_bit(reader, flag, error,
                  "sequence_config: enable_avg_cdf")) {
      return false;
    }
    config.enable_avg_cdf = flag;
    if (config.enable_avg_cdf) {
      if (!read_bit(reader, flag, error,
                    "sequence_config: avg_cdf_type")) {
        return false;
      }
      config.avg_cdf_type = flag;
    } else {
      config.avg_cdf_type = false;
    }
  }
  return read_sequence_delta_q_fields(reader, config.monochrome, seq, error);
}

// ---------------------------------------------------------------------------
// quantization_params() (§5.18.6.1)
// ---------------------------------------------------------------------------

std::size_t write_quantization_params(BitWriter& writer, int base_q_idx,
                                      const SequenceDeltaQ& seq_in,
                                      const FrameDeltaQ& frame,
                                      std::string* error) {
  if (!validate_sequence_delta_q(seq_in, error)) return 0;
  const SequenceDeltaQ seq = normalized_sequence_delta_q(seq_in);
  if (!in_range(base_q_idx, 0, kMaxQ)) {
    set_error(error, "base_q_idx outside [0, 255]");
    return 0;
  }
  if (!validate_frame_delta_q(seq, frame, error)) return 0;

  const std::size_t start = writer.bit_size();
  writer.put_bits(static_cast<std::uint64_t>(base_q_idx), 8U);  // 8-bit

  if (seq.y_dc_delta_q_enabled) {
    if (write_delta_q(writer, frame.delta_q_y_dc, error) == 0) return 0;
  }

  const bool have_uv = !seq.monochrome &&
                       (seq.uv_ac_delta_q_enabled ||
                        seq.uv_dc_delta_q_enabled);
  if (!have_uv) {
    return writer.bit_size() - start;
  }

  const bool diff_uv_delta =
      seq.separate_uv_delta_q &&
      (frame.delta_q_u_dc != frame.delta_q_v_dc ||
       frame.delta_q_u_ac != frame.delta_q_v_ac);
  if (seq.separate_uv_delta_q) writer.put_bit(diff_uv_delta);

  if (seq.uv_dc_delta_q_enabled) {
    if (write_delta_q(writer, frame.delta_q_u_dc, error) == 0) return 0;
  }
  if (seq.uv_ac_delta_q_enabled) {
    if (write_delta_q(writer, frame.delta_q_u_ac, error) == 0) return 0;
  }
  if (diff_uv_delta) {
    if (seq.uv_dc_delta_q_enabled) {
      if (write_delta_q(writer, frame.delta_q_v_dc, error) == 0) return 0;
    }
    if (seq.uv_ac_delta_q_enabled) {
      if (write_delta_q(writer, frame.delta_q_v_ac, error) == 0) return 0;
    }
  }
  return writer.bit_size() - start;
}

std::size_t write_disabled_quantization_params(BitWriter& writer,
                                               int base_q_idx,
                                               const SequenceDeltaQ& seq,
                                               std::string* error) {
  return write_quantization_params(writer, base_q_idx, seq, FrameDeltaQ{},
                                   error);
}

bool read_quantization_params(deltaq::BitReader& reader,
                              const SequenceDeltaQ& seq_in, int bit_depth,
                              QuantizationParams& out, std::string* error) {
  if (bit_depth != 8 && bit_depth != 9) {
    set_error(error, "quantization_params: only 8/9-bit qindex is defined");
    return false;
  }
  const SequenceDeltaQ seq = normalized_sequence_delta_q(seq_in);
  out = QuantizationParams{};
  std::uint32_t value = 0;
  if (!read_bits(reader, bit_depth == 8 ? 8U : 9U, value, error,
                 "quantization_params: base_q_idx")) {
    return false;
  }
  out.base_q_idx = static_cast<int>(value);
  FrameDeltaQ& frame = out.delta;

  if (seq.y_dc_delta_q_enabled) {
    if (!read_delta_q(reader, frame.delta_q_y_dc, &frame.y_dc_coded,
                      error)) {
      return false;
    }
  }
  const bool have_uv = !seq.monochrome &&
                       (seq.uv_ac_delta_q_enabled ||
                        seq.uv_dc_delta_q_enabled);
  if (!have_uv) return true;

  bool diff_uv_delta = false;
  if (seq.separate_uv_delta_q) {
    if (!read_bit(reader, diff_uv_delta, error,
                  "quantization_params: diff_uv_delta")) {
      return false;
    }
  }
  frame.diff_uv_delta = diff_uv_delta;
  if (seq.uv_dc_delta_q_enabled) {
    if (!read_delta_q(reader, frame.delta_q_u_dc, &frame.u_dc_coded,
                      error)) {
      return false;
    }
  }
  if (seq.uv_ac_delta_q_enabled) {
    if (!read_delta_q(reader, frame.delta_q_u_ac, &frame.u_ac_coded,
                      error)) {
      return false;
    }
  }
  if (seq.equal_ac_dc_q) frame.delta_q_u_dc = frame.delta_q_u_ac;

  if (diff_uv_delta) {
    if (seq.uv_dc_delta_q_enabled) {
      if (!read_delta_q(reader, frame.delta_q_v_dc, &frame.v_dc_coded,
                        error)) {
        return false;
      }
    }
    if (seq.uv_ac_delta_q_enabled) {
      if (!read_delta_q(reader, frame.delta_q_v_ac, &frame.v_ac_coded,
                        error)) {
        return false;
      }
    }
    if (seq.equal_ac_dc_q) frame.delta_q_v_dc = frame.delta_q_v_ac;
  } else {
    frame.delta_q_v_dc = frame.delta_q_u_dc;
    frame.delta_q_v_ac = frame.delta_q_u_ac;
    frame.v_dc_coded = frame.u_dc_coded;
    frame.v_ac_coded = frame.u_ac_coded;
  }
  return true;
}

}  // namespace av2enc::dcdeltaq
