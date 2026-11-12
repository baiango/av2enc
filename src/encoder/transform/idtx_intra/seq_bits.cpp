// See seq_bits.h.
#include "encoder/transform/idtx_intra/seq_bits.h"

namespace av2enc::idtx {
namespace {

class BitSink {
 public:
  void put_bit(bool bit) {
    if ((bits_ & 7u) == 0) bytes_.push_back(0);
    if (bit) bytes_.back() |= static_cast<std::uint8_t>(0x80u >> (bits_ & 7u));
    ++bits_;
  }
  void put_bits(std::uint32_t value, int count) {
    for (int i = count - 1; i >= 0; --i) put_bit(((value >> i) & 1u) != 0u);
  }
  const std::vector<std::uint8_t>& bytes() const { return bytes_; }
  std::size_t bit_size() const { return bits_; }

 private:
  std::vector<std::uint8_t> bytes_;
  std::size_t bits_ = 0;
};

class BitSource {
 public:
  BitSource(const std::uint8_t* data, std::size_t byte_count,
            std::size_t bit_offset)
      : data_(data), byte_count_(byte_count), pos_(bit_offset) {}

  bool get_bit(bool* bit) {
    if (pos_ >= byte_count_ * 8u) return false;
    *bit = ((data_[pos_ >> 3u] >> (7u - (pos_ & 7u))) & 1u) != 0u;
    ++pos_;
    return true;
  }
  bool get_bits(int count, std::uint32_t* value) {
    std::uint32_t out = 0;
    for (int i = 0; i < count; ++i) {
      bool bit = false;
      if (!get_bit(&bit)) return false;
      out = (out << 1u) | (bit ? 1u : 0u);
    }
    *value = out;
    return true;
  }
  std::size_t consumed(std::size_t start) const { return pos_ - start; }

 private:
  const std::uint8_t* data_;
  std::size_t byte_count_;
  std::size_t pos_;
};

void put_flag(BitSink& sink, bool value) { sink.put_bit(value); }

bool get_flag(BitSource& src, bool* value) {
  bool bit = false;
  if (!src.get_bit(&bit)) return false;
  *value = bit;
  return true;
}

}  // namespace

TransformSeqBits transform_seq_flags(const TransformSeqConfig& config) {
  TransformSeqBits info;
  info.idtx_bit_coded = !config.enable_fsc;
  info.enable_idtx_intra = config.enable_fsc || config.enable_idtx_intra;
  return info;
}

std::vector<std::uint8_t> write_transform_quant_entropy_config(
    const TransformSeqConfig& config, TransformSeqBits* info) {
  BitSink sink;

  put_flag(sink, config.enable_fsc);
  if (config.enable_fsc) {
    // enable_idtx_intra is inferred to 1; no bit.
  } else {
    put_flag(sink, config.enable_idtx_intra);
  }
  put_flag(sink, config.enable_intra_ist);
  put_flag(sink, config.enable_inter_ist);
  if (!config.monochrome) {
    put_flag(sink, config.enable_chroma_dctonly);
  }
  if (!config.single_picture_header) {
    put_flag(sink, config.enable_inter_ddt);
  }
  put_flag(sink, config.reduced_tx_part_set);
  if (!config.monochrome) {
    put_flag(sink, config.enable_cctx);
  }
  put_flag(sink, config.enable_tcq);
  if (config.enable_tcq && !config.single_picture_header) {
    put_flag(sink, config.choose_tcq_per_frame);
  }
  const bool tcq_fixed_off = config.enable_tcq && !config.choose_tcq_per_frame;
  if (!tcq_fixed_off) {
    put_flag(sink, config.enable_parity_hiding);
  }

  if (config.single_picture_header) {
    // enable_avg_cdf = 1, avg_cdf_type = 1 (inferred)
  } else {
    put_flag(sink, config.enable_avg_cdf);
    if (config.enable_avg_cdf) put_flag(sink, config.avg_cdf_type);
  }

  if (!config.monochrome) {
    put_flag(sink, config.separate_uv_delta_q);
  }
  put_flag(sink, config.equal_ac_dc_q);
  if (!config.equal_ac_dc_q) {
    sink.put_bits(static_cast<std::uint32_t>(config.base_y_dc_delta_q), 5);
    put_flag(sink, config.y_dc_delta_q_enabled);
  }
  if (!config.monochrome) {
    if (!config.equal_ac_dc_q) {
      sink.put_bits(static_cast<std::uint32_t>(config.base_uv_dc_delta_q), 5);
      put_flag(sink, config.uv_dc_delta_q_enabled);
    }
    sink.put_bits(static_cast<std::uint32_t>(config.base_uv_ac_delta_q), 5);
    put_flag(sink, config.uv_ac_delta_q_enabled);
  }

  if (info != nullptr) {
    *info = transform_seq_flags(config);
    info->bit_size = sink.bit_size();
  }
  return sink.bytes();
}

bool read_transform_quant_entropy_config(const std::uint8_t* data,
                                         std::size_t byte_count,
                                         std::size_t bit_offset,
                                         TransformSeqConfig* config,
                                         std::size_t* bits_consumed,
                                         TransformSeqBits* info) {
  if (data == nullptr || config == nullptr) return false;
  BitSource src(data, byte_count, bit_offset);
  // monochrome / single_picture_header are container facts, not coded here.
  TransformSeqConfig out = *config;

  if (!get_flag(src, &out.enable_fsc)) return false;
  if (out.enable_fsc) {
    out.enable_idtx_intra = false;  // inferred 1; raw value is meaningless
  } else if (!get_flag(src, &out.enable_idtx_intra)) {
    return false;
  }
  if (!get_flag(src, &out.enable_intra_ist)) return false;
  if (!get_flag(src, &out.enable_inter_ist)) return false;
  if (!out.monochrome && !get_flag(src, &out.enable_chroma_dctonly)) return false;
  if (!out.single_picture_header &&
      !get_flag(src, &out.enable_inter_ddt)) {
    return false;
  }
  if (!get_flag(src, &out.reduced_tx_part_set)) return false;
  if (!out.monochrome && !get_flag(src, &out.enable_cctx)) return false;
  if (!get_flag(src, &out.enable_tcq)) return false;
  if (out.enable_tcq && !out.single_picture_header &&
      !get_flag(src, &out.choose_tcq_per_frame)) {
    return false;
  }
  const bool tcq_fixed_off = out.enable_tcq && !out.choose_tcq_per_frame;
  if (!tcq_fixed_off && !get_flag(src, &out.enable_parity_hiding)) return false;

  if (!out.single_picture_header) {
    if (!get_flag(src, &out.enable_avg_cdf)) return false;
    if (out.enable_avg_cdf && !get_flag(src, &out.avg_cdf_type)) return false;
  }
  if (!out.monochrome && !get_flag(src, &out.separate_uv_delta_q)) return false;
  if (!get_flag(src, &out.equal_ac_dc_q)) return false;
  if (!out.equal_ac_dc_q) {
    std::uint32_t value = 0;
    if (!src.get_bits(5, &value)) return false;
    out.base_y_dc_delta_q = static_cast<int>(value);
    if (!get_flag(src, &out.y_dc_delta_q_enabled)) return false;
  }
  if (!out.monochrome) {
    if (!out.equal_ac_dc_q) {
      std::uint32_t value = 0;
      if (!src.get_bits(5, &value)) return false;
      out.base_uv_dc_delta_q = static_cast<int>(value);
      if (!get_flag(src, &out.uv_dc_delta_q_enabled)) return false;
    }
    std::uint32_t value = 0;
    if (!src.get_bits(5, &value)) return false;
    out.base_uv_ac_delta_q = static_cast<int>(value);
    if (!get_flag(src, &out.uv_ac_delta_q_enabled)) return false;
  }

  if (bits_consumed != nullptr) *bits_consumed = src.consumed(bit_offset);
  if (info != nullptr) {
    *info = transform_seq_flags(out);
    info->bit_size = src.consumed(bit_offset);
  }
  *config = out;
  return true;
}

}  // namespace av2enc::idtx
