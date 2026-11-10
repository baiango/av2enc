#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace av2enc {

// MSB-first bit writer. Bits are accumulated into a byte starting with the most
// significant bit; the final partially-filled byte is zero-padded by
// align_to_byte(). This matches the bit ordering used by AV2 syntax elements,
// but the writer itself is syntax-agnostic.
class BitWriter {
 public:
  // Writes one bit. Appends a zero byte when starting a new byte.
  void put_bit(bool bit);

  // Writes the low `num_bits` bits of `value`, most significant bit first.
  // `num_bits` must be in [0, 64].
  void put_bits(std::uint64_t value, unsigned num_bits);

  // Pads the current byte with zero bits. No-op when already aligned.
  void align_to_byte();

  // Clears all bytes and returns to the empty, aligned state.
  void clear();

  // Number of bits written, including padding added by align_to_byte().
  std::size_t bit_size() const { return bit_size_; }

  // Number of bytes produced. A partial byte is included.
  std::size_t byte_size() const { return bytes_.size(); }

  bool byte_aligned() const { return (bit_size_ & 7U) == 0; }

  std::span<const std::uint8_t> bytes() const {
    return std::span<const std::uint8_t>(bytes_);
  }

  // Moves the accumulated bytes out and resets the writer. The last byte is
  // whatever has been written so far; call align_to_byte() first if the caller
  // needs a fully padded byte stream.
  std::vector<std::uint8_t> take_bytes();

 private:
  std::vector<std::uint8_t> bytes_;
  std::size_t bit_size_ = 0;
};

}  // namespace av2enc
