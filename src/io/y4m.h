#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace av2enc {

// Parsed YUV4MPEG2 file header.
//
// av2enc currently accepts only 8-bit 4:2:0 input. C420, C420jpeg, C420mpeg2
// and C420paldv all have the same planar byte layout and are accepted; higher
// bit depths and other subsampling modes are rejected with an explicit error.
struct Y4mHeader {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 0;
  char interlacing = '?';  // 'p', 't', 'b', 'm' or '?'
  std::uint32_t aspect_num = 0;
  std::uint32_t aspect_den = 0;
  std::string colorspace = "C420";
};

// One decoded Y4M frame. Planes are tightly packed: Y, then U, then V.
// Spans returned by the accessors remain valid until the frame is overwritten
// by the next Y4mReader::read_frame() call.
class Y4mFrame {
 public:
  std::uint32_t width() const { return width_; }
  std::uint32_t height() const { return height_; }
  std::uint32_t chroma_width() const { return (width_ + 1U) / 2U; }
  std::uint32_t chroma_height() const { return (height_ + 1U) / 2U; }

  std::size_t luma_size() const {
    return static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  }
  std::size_t chroma_size() const {
    return static_cast<std::size_t>(chroma_width()) *
           static_cast<std::size_t>(chroma_height());
  }

  std::span<const std::uint8_t> y() const;
  std::span<const std::uint8_t> u() const;
  std::span<const std::uint8_t> v() const;

 private:
  friend class Y4mReader;

  std::uint32_t width_ = 0;
  std::uint32_t height_ = 0;
  std::vector<std::uint8_t> data_;
};

enum class Y4mReadResult {
  kError = -1,
  kEof = 0,
  kFrame = 1,
};

class Y4mReader {
 public:
  explicit Y4mReader(std::filesystem::path path);

  Y4mReader(const Y4mReader&) = delete;
  Y4mReader& operator=(const Y4mReader&) = delete;

  // Opens the file and parses the stream header. Returns false and fills
  // `error` for missing files, malformed headers, or unsupported formats.
  bool open(std::string& error);

  // Reads the next frame. Returns kEof at a clean end of stream, kError on a
  // malformed/truncated stream, and kFrame on success. On success `error` is
  // cleared; `frame` is only guaranteed valid for kFrame.
  Y4mReadResult read_frame(Y4mFrame& frame, std::string& error);

  const Y4mHeader& header() const { return header_; }
  const std::filesystem::path& path() const { return path_; }
  bool is_open() const { return file_.is_open(); }

 private:
  bool parse_header_line(const std::string& line, std::string& error);

  std::filesystem::path path_;
  std::ifstream file_;
  Y4mHeader header_;
  bool open_ = false;
  std::uint64_t frame_index_ = 0;
};

}  // namespace av2enc
