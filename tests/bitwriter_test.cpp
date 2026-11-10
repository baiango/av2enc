#include "bitstream/bitwriter.h"

#include "test_common.h"

#include <array>
#include <cstdint>

namespace {

void test_empty_writer() {
  av2enc::BitWriter writer;
  CHECK(writer.bit_size() == 0U);
  CHECK(writer.byte_size() == 0U);
  CHECK(writer.byte_aligned());
  CHECK(writer.bytes().empty());
}

void test_single_bits_and_alignment() {
  av2enc::BitWriter writer;
  writer.put_bit(true);
  writer.put_bit(false);
  writer.put_bit(true);
  CHECK(writer.bit_size() == 3U);
  CHECK(writer.byte_size() == 1U);
  CHECK(!writer.byte_aligned());
  CHECK(writer.bytes()[0] == 0xA0U);

  writer.align_to_byte();
  CHECK(writer.bit_size() == 8U);
  CHECK(writer.byte_size() == 1U);
  CHECK(writer.byte_aligned());
  CHECK(writer.bytes()[0] == 0xA0U);
}

void test_byte_writes() {
  {
    av2enc::BitWriter writer;
    writer.put_bits(0xAFU, 8);
    CHECK(writer.byte_aligned());
    CHECK(writer.bytes().size() == 1U);
    CHECK(writer.bytes()[0] == 0xAFU);
  }
  {
    av2enc::BitWriter writer;
    writer.put_bits(0xABCDU, 16);
    CHECK(writer.bit_size() == 16U);
    CHECK(writer.bytes().size() == 2U);
    CHECK(writer.bytes()[0] == 0xABU);
    CHECK(writer.bytes()[1] == 0xCDU);
  }
}

void test_cross_byte_boundary() {
  av2enc::BitWriter writer;
  writer.put_bits(0b101U, 3);
  writer.put_bits(0xFFFFU, 16);
  CHECK(writer.bit_size() == 19U);
  CHECK(writer.byte_size() == 3U);
  CHECK(writer.bytes()[0] == 0xBFU);
  CHECK(writer.bytes()[1] == 0xFFU);
  CHECK(writer.bytes()[2] == 0xE0U);

  writer.align_to_byte();
  CHECK(writer.bit_size() == 24U);
  CHECK(writer.byte_size() == 3U);
  CHECK(writer.byte_aligned());
  CHECK(writer.bytes()[2] == 0xE0U);
}

void test_partial_value() {
  av2enc::BitWriter writer;
  writer.put_bits(0x1FFU, 9);  // 1 1111 1111
  CHECK(writer.bit_size() == 9U);
  CHECK(writer.byte_size() == 2U);
  CHECK(writer.bytes()[0] == 0xFFU);
  CHECK(writer.bytes()[1] == 0x80U);
}

void test_zero_bits() {
  av2enc::BitWriter writer;
  writer.put_bits(0xFFFFFFFFU, 0);
  CHECK(writer.bit_size() == 0U);
  CHECK(writer.byte_size() == 0U);
  CHECK(writer.byte_aligned());
}

void test_full_width_and_take() {
  av2enc::BitWriter writer;
  writer.put_bits(0x0123456789ABCDEFULL, 64);
  const std::array<std::uint8_t, 8> expected = {0x01, 0x23, 0x45, 0x67,
                                                0x89, 0xAB, 0xCD, 0xEF};
  CHECK(writer.byte_size() == expected.size());
  CHECK(writer.bytes().size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(writer.bytes()[i] == expected[i]);
  }

  const std::vector<std::uint8_t> taken = writer.take_bytes();
  CHECK(taken.size() == expected.size());
  CHECK(writer.bit_size() == 0U);
  CHECK(writer.byte_size() == 0U);
  CHECK(writer.byte_aligned());
}

void test_clear() {
  av2enc::BitWriter writer;
  writer.put_bits(0x12U, 8);
  writer.clear();
  CHECK(writer.bit_size() == 0U);
  CHECK(writer.byte_size() == 0U);
  writer.put_bits(0x34U, 8);
  CHECK(writer.bytes().size() == 1U);
  CHECK(writer.bytes()[0] == 0x34U);
}

}  // namespace

int main() {
  test_empty_writer();
  test_single_bits_and_alignment();
  test_byte_writes();
  test_cross_byte_boundary();
  test_partial_value();
  test_zero_bits();
  test_full_width_and_take();
  test_clear();
  return test::finish("bitwriter_test");
}
