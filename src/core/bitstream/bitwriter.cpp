#include "core/bitstream/bitwriter.h"

namespace av2enc {

void BitWriter::put_bit(bool bit) {
  const std::size_t byte_index = bit_size_ / 8;
  if (byte_index == bytes_.size()) {
    bytes_.push_back(0);
  }

  if (bit) {
    const unsigned bit_in_byte = static_cast<unsigned>(bit_size_ & 7U);
    bytes_[byte_index] |= static_cast<std::uint8_t>(0x80U >> bit_in_byte);
  }
  ++bit_size_;
}

void BitWriter::put_bits(std::uint64_t value, unsigned num_bits) {
  assert(num_bits <= 64U);
  if (num_bits > 64U) {
    num_bits = 64U;
  }

  for (unsigned remaining = num_bits; remaining > 0; --remaining) {
    const unsigned shift = remaining - 1U;
    put_bit(((value >> shift) & 1U) != 0U);
  }
}

void BitWriter::align_to_byte() {
  while (!byte_aligned()) {
    put_bit(false);
  }
}

void BitWriter::clear() {
  bytes_.clear();
  bit_size_ = 0;
}

std::vector<std::uint8_t> BitWriter::take_bytes() {
  std::vector<std::uint8_t> result = std::move(bytes_);
  bytes_.clear();
  bit_size_ = 0;
  return result;
}

}  // namespace av2enc
