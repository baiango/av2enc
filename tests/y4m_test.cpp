#include "io/y4m.h"

#include "test_common.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kWidth = 64;
constexpr std::size_t kHeight = 64;
constexpr std::size_t kLumaSize = kWidth * kHeight;
constexpr std::size_t kChromaSize = (kWidth / 2) * (kHeight / 2);

void append_string(std::vector<std::uint8_t>& out, std::string_view text) {
  for (char character : text) {
    out.push_back(static_cast<std::uint8_t>(character));
  }
}

void append_fill(std::vector<std::uint8_t>& out, std::size_t count, std::uint8_t value) {
  out.insert(out.end(), count, value);
}

// Builds a tiny 64x64 C420 stream. Frame f has flat Y=100+f, U=60+f and
// V=200-f so the test can identify every plane without storing media files.
std::vector<std::uint8_t> make_y4m(std::string_view colorspace, char interlacing, int frames) {
  std::vector<std::uint8_t> bytes;
  append_string(bytes, "YUV4MPEG2 W64 H64 F30:1 I");
  bytes.push_back(static_cast<std::uint8_t>(interlacing));
  append_string(bytes, " A1:1");
  if (!colorspace.empty()) {
    bytes.push_back(static_cast<std::uint8_t>(' '));
    append_string(bytes, colorspace);
  }
  bytes.push_back(static_cast<std::uint8_t>('\n'));

  for (int frame = 0; frame < frames; ++frame) {
    append_string(bytes, "FRAME\n");
    append_fill(bytes, kLumaSize, static_cast<std::uint8_t>(100 + frame));
    append_fill(bytes, kChromaSize, static_cast<std::uint8_t>(60 + frame));
    append_fill(bytes, kChromaSize, static_cast<std::uint8_t>(200 - frame));
  }
  return bytes;
}

void write_or_fail(const std::filesystem::path& path,
                   const std::vector<std::uint8_t>& bytes) {
  CHECK(test::write_file(path, bytes));
}

void test_parse_two_frames() {
  const auto path = test::temp_path("y4m_two_frames.y4m");
  write_or_fail(path, make_y4m("C420", 'p', 2));

  av2enc::Y4mReader reader(path);
  std::string error = "stale";
  CHECK(reader.open(error));
  CHECK(error.empty());
  CHECK(reader.header().width == 64U);
  CHECK(reader.header().height == 64U);
  CHECK(reader.header().fps_num == 30U);
  CHECK(reader.header().fps_den == 1U);
  CHECK(reader.header().interlacing == 'p');
  CHECK(reader.header().aspect_num == 1U);
  CHECK(reader.header().aspect_den == 1U);
  CHECK(reader.header().colorspace == "C420");

  av2enc::Y4mFrame frame;
  CHECK(reader.read_frame(frame, error) == av2enc::Y4mReadResult::kFrame);
  CHECK(error.empty());
  CHECK(frame.width() == 64U);
  CHECK(frame.height() == 64U);
  CHECK(frame.chroma_width() == 32U);
  CHECK(frame.chroma_height() == 32U);
  CHECK(frame.y().size() == kLumaSize);
  CHECK(frame.u().size() == kChromaSize);
  CHECK(frame.v().size() == kChromaSize);
  CHECK(frame.y().front() == 100U);
  CHECK(frame.y().back() == 100U);
  CHECK(frame.u().front() == 60U);
  CHECK(frame.v().front() == 200U);

  CHECK(reader.read_frame(frame, error) == av2enc::Y4mReadResult::kFrame);
  CHECK(frame.y().front() == 101U);
  CHECK(frame.u().front() == 61U);
  CHECK(frame.v().front() == 199U);

  CHECK(reader.read_frame(frame, error) == av2enc::Y4mReadResult::kEof);
  CHECK(error.empty());
  test::remove_file(path);
}

void test_missing_colorspace_defaults_to_c420() {
  const auto path = test::temp_path("y4m_default_c420.y4m");
  write_or_fail(path, make_y4m("", 'p', 0));

  av2enc::Y4mReader reader(path);
  std::string error;
  CHECK(reader.open(error));
  CHECK(reader.header().colorspace == "C420");
  test::remove_file(path);
}

void test_unsupported_colorspaces() {
  const auto path = test::temp_path("y4m_unsupported.y4m");
  for (const std::string_view colorspace : {"C444", "C422", "C420p10"}) {
    write_or_fail(path, make_y4m(colorspace, 'p', 0));
    av2enc::Y4mReader reader(path);
    std::string error;
    CHECK(!reader.open(error));
    CHECK(error.find("8-bit 4:2:0") != std::string::npos);
  }
  test::remove_file(path);
}

void test_interlaced_rejected() {
  const auto path = test::temp_path("y4m_interlaced.y4m");
  write_or_fail(path, make_y4m("C420", 't', 0));

  av2enc::Y4mReader reader(path);
  std::string error;
  CHECK(!reader.open(error));
  CHECK(error.find("interlaced") != std::string::npos);
  test::remove_file(path);
}

void test_truncated_frame() {
  const auto path = test::temp_path("y4m_truncated.y4m");
  std::vector<std::uint8_t> bytes;
  append_string(bytes, "YUV4MPEG2 W64 H64 F30:1 Ip A1:1 C420\nFRAME\n");
  append_fill(bytes, 100, 0x7F);
  write_or_fail(path, bytes);

  av2enc::Y4mReader reader(path);
  std::string error;
  CHECK(reader.open(error));
  av2enc::Y4mFrame frame;
  CHECK(reader.read_frame(frame, error) == av2enc::Y4mReadResult::kError);
  CHECK(error.find("truncated") != std::string::npos);
  test::remove_file(path);
}

void test_bad_frame_marker() {
  const auto path = test::temp_path("y4m_bad_marker.y4m");
  std::vector<std::uint8_t> bytes;
  append_string(bytes, "YUV4MPEG2 W64 H64 F30:1 Ip A1:1 C420\nNOTFRAME\n");
  write_or_fail(path, bytes);

  av2enc::Y4mReader reader(path);
  std::string error;
  CHECK(reader.open(error));
  av2enc::Y4mFrame frame;
  CHECK(reader.read_frame(frame, error) == av2enc::Y4mReadResult::kError);
  CHECK(error.find("FRAME") != std::string::npos);
  test::remove_file(path);
}

void test_bad_header() {
  const auto path = test::temp_path("y4m_bad_header.y4m");
  std::vector<std::uint8_t> bytes;
  append_string(bytes, "YUV4MPEG2 H64 F30:1 Ip C420\n");
  write_or_fail(path, bytes);

  av2enc::Y4mReader reader(path);
  std::string error;
  CHECK(!reader.open(error));
  CHECK(!error.empty());
  test::remove_file(path);
}

}  // namespace

int main() {
  test_parse_two_frames();
  test_missing_colorspace_defaults_to_c420();
  test_unsupported_colorspaces();
  test_interlaced_rejected();
  test_truncated_frame();
  test_bad_frame_marker();
  test_bad_header();
  return test::finish("y4m_test");
}
