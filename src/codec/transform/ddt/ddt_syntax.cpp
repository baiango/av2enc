#include "codec/transform/ddt/ddt_syntax.h"

namespace av2enc::video::ddt {

std::size_t write_enable_inter_ddt(BitWriter& writer,
                                   bool single_picture_header_flag,
                                   bool enable_inter_ddt) {
  if (single_picture_header_flag) return 0;
  writer.put_bit(enable_inter_ddt);
  return kEnableInterDdtBits;
}

bool read_enable_inter_ddt(std::span<const std::uint8_t> data,
                           std::size_t bit_offset,
                           bool single_picture_header_flag,
                           bool* enable_inter_ddt, std::size_t* next_bit) {
  if (enable_inter_ddt == nullptr || next_bit == nullptr) return false;
  if (single_picture_header_flag) {
    *enable_inter_ddt = false;
    *next_bit = bit_offset;
    return true;
  }
  const std::size_t byte = bit_offset >> 3;
  if (byte >= data.size()) return false;
  const int shift = 7 - static_cast<int>(bit_offset & 7U);
  *enable_inter_ddt = ((data[byte] >> shift) & 1U) != 0;
  *next_bit = bit_offset + 1;
  return true;
}

std::size_t payload_bytes_added(std::size_t sequence_syntax_bits_without_ddt,
                                bool single_picture_header_flag) {
  const auto payload_bytes = [](std::size_t syntax_bits) {
    // sequence_header_obu(): syntax bits, one obu_extension_flag bit, then
    // §5.2.3 trailing_bits (one '1' plus zero padding).
    return (syntax_bits + std::size_t{2} + 7U) / 8U;
  };
  const std::size_t delta = single_picture_header_flag ? 0U : 1U;
  return payload_bytes(sequence_syntax_bits_without_ddt + delta) -
         payload_bytes(sequence_syntax_bits_without_ddt);
}

}  // namespace av2enc::video::ddt
