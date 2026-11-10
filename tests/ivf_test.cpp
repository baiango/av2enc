#include "io/ivf.h"

#include "test_common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

std::uint16_t read_le16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  return static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[offset]) |
                                    (static_cast<std::uint16_t>(bytes[offset + 1]) << 8U));
}

std::uint32_t read_le32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

std::uint64_t read_le64(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(bytes[offset + i]) << static_cast<unsigned>(8U * i);
  }
  return value;
}

void test_header_and_frames() {
  const auto path = test::temp_path("ivf_structure.ivf");

  av2enc::IvfHeader header;
  header.width = 64;
  header.height = 64;
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 999;  // informational only; writer must patch the real count

  av2enc::IvfWriter writer;
  std::string error;
  CHECK(writer.open(path, header, error));
  CHECK(error.empty());
  CHECK(writer.is_open());

  const std::array<std::uint8_t, 4> payload0 = {0x11, 0x22, 0x33, 0x44};
  const std::array<std::uint8_t, 3> payload1 = {0xAA, 0xBB, 0xCC};
  CHECK(writer.write_frame(payload0, 0, error));
  CHECK(writer.write_frame(payload1, 0x123456789ULL, error));
  CHECK(writer.frame_count() == 2U);
  CHECK(writer.bytes_written() == 63U);

  CHECK(writer.close(error));
  CHECK(error.empty());
  CHECK(writer.close(error));  // idempotent

  const std::vector<std::uint8_t> bytes = test::read_file(path);
  CHECK(bytes.size() == 63U);

  if (bytes.size() == 63U) {
    // 32-byte file header.
    CHECK(bytes[0] == 'D');
    CHECK(bytes[1] == 'K');
    CHECK(bytes[2] == 'I');
    CHECK(bytes[3] == 'F');
    CHECK(read_le16(bytes, 4) == 0U);
    CHECK(read_le16(bytes, 6) == 32U);
    CHECK(bytes[8] == 'A');
    CHECK(bytes[9] == 'V');
    CHECK(bytes[10] == '0');
    CHECK(bytes[11] == '2');
    CHECK(read_le16(bytes, 12) == 64U);
    CHECK(read_le16(bytes, 14) == 64U);
    CHECK(read_le32(bytes, 16) == 30U);
    CHECK(read_le32(bytes, 20) == 1U);
    CHECK(read_le32(bytes, 24) == 2U);
    CHECK(read_le32(bytes, 28) == 0U);

    // First frame: 12-byte header at offset 32, 4-byte payload at 44.
    CHECK(read_le32(bytes, 32) == 4U);
    CHECK(read_le64(bytes, 36) == 0U);
    CHECK(bytes[44] == 0x11U);
    CHECK(bytes[45] == 0x22U);
    CHECK(bytes[46] == 0x33U);
    CHECK(bytes[47] == 0x44U);

    // Second frame: 12-byte header at offset 48, 3-byte payload at 60.
    CHECK(read_le32(bytes, 48) == 3U);
    CHECK(read_le64(bytes, 52) == 0x123456789ULL);
    CHECK(bytes[60] == 0xAAU);
    CHECK(bytes[61] == 0xBBU);
    CHECK(bytes[62] == 0xCCU);
  }

  test::remove_file(path);
}

void test_invalid_header() {
  const auto path = test::temp_path("ivf_invalid.ivf");

  av2enc::IvfHeader header;
  header.width = 0;
  header.height = 64;
  header.fps_num = 30;
  header.fps_den = 1;

  av2enc::IvfWriter writer;
  std::string error;
  CHECK(!writer.open(path, header, error));
  CHECK(!error.empty());

  header.width = 64;
  header.fps_den = 0;
  error.clear();
  CHECK(!writer.open(path, header, error));
  CHECK(!error.empty());

  test::remove_file(path);
}

void test_write_after_close() {
  const auto path = test::temp_path("ivf_closed.ivf");

  av2enc::IvfHeader header;
  header.width = 64;
  header.height = 64;
  header.fps_num = 30;
  header.fps_den = 1;

  av2enc::IvfWriter writer;
  std::string error;
  CHECK(writer.open(path, header, error));
  CHECK(writer.close(error));

  const std::array<std::uint8_t, 1> payload = {0x00};
  error.clear();
  CHECK(!writer.write_frame(payload, 0, error));
  CHECK(!error.empty());

  test::remove_file(path);
}

}  // namespace

int main() {
  test_header_and_frames();
  test_invalid_header();
  test_write_after_close();
  return test::finish("ivf_test");
}
