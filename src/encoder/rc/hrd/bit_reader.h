#pragma once

// MSB-first bit reader for the AV2 syntax descriptors used by the HRD/timing
// component.
//
// This is the exact inverse of `av2enc::BitWriter` for f(n), uvlc() and rg(n)
// (§4.11). It is self-contained: the HRD component parses only plain syntax
// bits (sequence timing, decoder model info, BRT and content interpretation),
// never tile symbols. Bit 0 of the stream is the most significant bit of byte
// 0, matching `BitWriter::put_bits()` and §4.11.1.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace av2enc::hrd {

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

  // f(n) / L(n): n <= 32.
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

  // uvlc() (§4.11.3): leadingZeros zero bits, a one bit, then leadingZeros
  // payload bits. Returns false for leadingZeros >= 32 and for a truncated
  // payload.
  bool read_uvlc(std::uint32_t& value, std::string* error = nullptr) {
    unsigned leading_zeros = 0;
    bool bit = false;
    while (true) {
      if (!read_bit(bit)) {
        set_error(error, "truncated uvlc()");
        return false;
      }
      if (bit) break;
      ++leading_zeros;
      if (leading_zeros >= 32U) {
        set_error(error, "uvlc() has 32 leading zero bits");
        return false;
      }
    }
    std::uint32_t payload = 0;
    if (!read_bits(leading_zeros, payload)) {
      set_error(error, "truncated uvlc() payload");
      return false;
    }
    if (leading_zeros == 0) {
      value = 0;
      return true;
    }
    const std::uint64_t base = (1ULL << leading_zeros) - 1ULL;
    value = static_cast<std::uint32_t>(base + payload);
    return true;
  }

  // rg(n) (§4.11.10): q unary one bits terminated by zero, then n remainder
  // bits, value = (q << n) + remainder. q is limited to 31, so the maximum
  // return value is (32 << n) - 1. A 32nd consecutive one bit is invalid
  // (the syntax process returns -1).
  bool read_rg(unsigned n, std::uint32_t& value, std::string* error = nullptr) {
    if (n > 24U) {
      set_error(error, "rg(n) parameter out of range");
      return false;
    }
    unsigned q = 0;
    for (; q < 32U; ++q) {
      bool bit = false;
      if (!read_bit(bit)) {
        set_error(error, "truncated rg() unary prefix");
        return false;
      }
      if (!bit) {
        std::uint32_t remainder = 0;
        if (!read_bits(n, remainder)) {
          set_error(error, "truncated rg() remainder");
          return false;
        }
        value = (q << n) | remainder;
        return true;
      }
    }
    set_error(error, "rg() prefix exceeds 31 one bits");
    return false;
  }

  std::size_t bit_position() const { return bit_position_; }
  std::size_t bits_remaining() const {
    const std::size_t total = data_.size() * 8U;
    return bit_position_ >= total ? 0U : total - bit_position_;
  }
  bool byte_aligned() const { return (bit_position_ & 7U) == 0U; }
  bool failed() const { return failed_; }
  std::span<const std::uint8_t> data() const { return data_; }

 private:
  static void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
  }

  std::span<const std::uint8_t> data_{};
  std::size_t bit_position_ = 0;
  bool failed_ = false;
};

}  // namespace av2enc::hrd
