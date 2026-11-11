#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// MSB-first bit reader for AV2 syntax descriptors.
//
// This is the exact inverse of `av2enc::BitWriter` for the raw syntax
// descriptors f(n) / L(n) / uvlc() / su(n).  It is deliberately tiny and has no
// dependency on the entropy coder: frame-header syntax is coded as plain bits,
// while tile symbols go through `av2enc::av2::Decoder` (§8.2).
//
// Bit 0 of the stream is the most significant bit of byte 0, matching
// `BitWriter::put_bits()` and the AV2 descriptor definition (§4.11.1).
namespace av2enc::deltaq {

class BitReader {
 public:
  BitReader() = default;
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}

  void reset(std::span<const std::uint8_t> data) {
    data_ = data;
    bit_position_ = 0;
    failed_ = false;
  }

  bool read_bit(bool& value) {
    if (failed_ || bit_position_ >= data_.size() * 8U) {
      failed_ = true;
      return false;
    }
    const std::size_t byte = bit_position_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_position_ & 7U);
    value = ((data_[byte] >> shift) & 1U) != 0U;
    ++bit_position_;
    return true;
  }

  // Reads n bits MSB-first into the low n bits of `value`. n <= 32.
  bool read_bits(unsigned n, std::uint32_t& value) {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < n; ++i) {
      bool bit = false;
      if (!read_bit(bit)) return false;
      out = (out << 1U) | (bit ? 1U : 0U);
    }
    value = out;
    return true;
  }

  // L(n): a literal of n bits, identical to f(n) (§4.11.3).
  bool read_literal(unsigned n, std::uint32_t& value) { return read_bits(n, value); }

  std::size_t bit_position() const { return bit_position_; }
  bool byte_aligned() const { return (bit_position_ & 7U) == 0U; }
  bool failed() const { return failed_; }

  // Advances to the next byte boundary with zero bits, like
  // `byte_alignment()` (§5.2.4).  The skipped bits must be zero.
  bool align_to_byte() {
    while ((bit_position_ & 7U) != 0U) {
      bool bit = false;
      if (!read_bit(bit)) return false;
      if (bit) {
        failed_ = true;
        return false;
      }
    }
    return true;
  }

  std::span<const std::uint8_t> data() const { return data_; }

 private:
  std::span<const std::uint8_t> data_{};
  std::size_t bit_position_ = 0;
  bool failed_ = false;
};

}  // namespace av2enc::deltaq
