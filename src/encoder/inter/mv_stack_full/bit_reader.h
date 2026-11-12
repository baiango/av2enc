// Minimal MSB-first bit reader for the syntax mirror.
//
// The writers in this module use `av2enc::BitWriter` (the project's pinned
// writer).  The readers that prove no desync are deliberately independent:
// they own their own cursor and reject truncation instead of returning zeros.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace av2enc::inter::mv_stack_full {

class BitReader {
 public:
  BitReader(const std::uint8_t* data, std::size_t size)
      : data_(data), size_bits_(size * 8) {}

  bool read_bit(bool* out) {
    if (out == nullptr || pos_ >= size_bits_) return false;
    const std::size_t byte = pos_ >> 3;
    const int shift = 7 - static_cast<int>(pos_ & 7);
    *out = ((data_[byte] >> shift) & 1u) != 0;
    ++pos_;
    return true;
  }

  bool read_bits(int count, std::uint32_t* out) {
    if (out == nullptr || count < 0 || count > 32) return false;
    if (pos_ + static_cast<std::size_t>(count) > size_bits_) return false;
    std::uint32_t value = 0;
    for (int i = 0; i < count; ++i) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      value = (value << 1u) | (bit ? 1u : 0u);
    }
    *out = value;
    return true;
  }

  // Exp-Golomb (uvlc) as used by AV2.
  bool read_uvlc(std::uint32_t* out) {
    if (out == nullptr) return false;
    int leading_zeros = 0;
    while (true) {
      bool bit = true;
      if (!read_bit(&bit)) return false;
      if (bit) break;
      if (++leading_zeros > 31) return false;
    }
    if (leading_zeros == 0) {
      *out = 0;
      return true;
    }
    std::uint32_t payload = 0;
    if (!read_bits(leading_zeros, &payload)) return false;
    *out = (payload + (1u << leading_zeros) - 1u);
    return true;
  }

  std::size_t bits_read() const { return pos_; }
  std::size_t bits_left() const { return size_bits_ - pos_; }
  bool align_to_byte() {
    const std::size_t rem = pos_ & 7;
    if (rem == 0) return true;
    const std::size_t need = 8 - rem;
    if (pos_ + need > size_bits_) return false;
    pos_ += need;
    return true;
  }

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t size_bits_ = 0;
  std::size_t pos_ = 0;
};

}  // namespace av2enc::inter::mv_stack_full
