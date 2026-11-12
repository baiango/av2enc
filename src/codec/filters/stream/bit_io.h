// MSB-first bit I/O helpers for the M1 filter/restoration header syntax.
//
// `BitWriter` (src/bitstream/bitwriter.h) is the encoder used by the rest of
// M1.  The AV2 frame/sequence header is read with the raw descriptors
// f(n), ns(n), tu(mx) and uvlc(); the entropy-coded tile payload is separate
// (src/entropy/av2).  This file provides the missing reader plus the exact
// inverse of the descriptors used by §5.4.10, §5.18.7.10–§5.18.7.12 and
// §5.20.10.4–§5.20.10.6.
//
// Spec-only; no entropy state and no CDFs here.
#ifndef AV2ENC_ENCODER_STILL_FILTERS_STREAM_BIT_IO_H_
#define AV2ENC_ENCODER_STILL_FILTERS_STREAM_BIT_IO_H_

#include "core/bitstream/bitwriter.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace av2enc::still::filters_stream {

// Reads the same MSB-first stream `BitWriter` produces.  `bit_size` lets a
// caller decode a syntax prefix that ends before the byte alignment; the
// default is the full span.
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes,
                     std::size_t bit_size = 0)
      : bytes_(bytes),
        bit_size_(bit_size == 0 ? bytes.size() * 8U : bit_size) {}

  std::size_t bit_pos() const { return bit_pos_; }
  std::size_t bit_size() const { return bit_size_; }
  std::size_t bits_left() const {
    return bit_pos_ <= bit_size_ ? bit_size_ - bit_pos_ : 0U;
  }

  bool read_bit(bool* bit) {
    if (bit == nullptr || bit_pos_ >= bit_size_) return false;
    const std::size_t byte = bit_pos_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    *bit = ((bytes_[byte] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool read_bits(unsigned num_bits, std::uint32_t* value) {
    if (value == nullptr || num_bits > 32U || bits_left() < num_bits) {
      return false;
    }
    std::uint32_t out = 0;
    for (unsigned i = 0; i < num_bits; ++i) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      out = (out << 1U) | (bit ? 1U : 0U);
    }
    *value = out;
    return true;
  }

  // §4.11.8 ns(n).
  bool read_ns(unsigned n, std::uint32_t* value) {
    if (n == 0U || value == nullptr) return false;
    if (n == 1U) {
      *value = 0;
      return true;
    }
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    std::uint32_t v = 0;
    if (!read_bits(w - 1U, &v)) return false;
    if (v < m) {
      *value = v;
      return true;
    }
    bool extra = false;
    if (!read_bit(&extra)) return false;
    *value = (v << 1U) - m + (extra ? 1U : 0U);
    return *value < n;
  }

  // §4.11.9 tu(mx).
  bool read_tu(unsigned mx, std::uint32_t* value) {
    if (value == nullptr) return false;
    std::uint32_t v = 0;
    while (v < mx) {
      bool bit = false;
      if (!read_bit(&bit)) return false;
      if (!bit) {
        *value = v;
        return true;
      }
      ++v;
    }
    *value = mx;
    return true;
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_size_ = 0;
  std::size_t bit_pos_ = 0;
};

// §4.11.8 ns(n) inverse.
inline void put_ns(BitWriter& writer, unsigned n, std::uint32_t value) {
  if (n <= 1U) return;
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

// §4.11.9 tu(mx) inverse: `value` ones then a zero, except at the maximum.
inline void put_tu(BitWriter& writer, unsigned mx, std::uint32_t value) {
  for (std::uint32_t i = 0; i < value && i < mx; ++i) {
    writer.put_bit(true);
  }
  if (value < mx) {
    writer.put_bit(false);
  }
}

}  // namespace av2enc::still::filters_stream

#endif  // AV2ENC_ENCODER_STILL_FILTERS_STREAM_BIT_IO_H_
