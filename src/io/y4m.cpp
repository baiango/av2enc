#include "io/y4m.h"

#include <charconv>
#include <limits>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace av2enc {
namespace {

void strip_cr(std::string& line) {
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
  if (text.empty()) {
    return false;
  }
  const char* begin = text.data();
  const char* end = begin + text.size();
  std::uint32_t parsed = 0;
  const std::from_chars_result result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end) {
    return false;
  }
  value = parsed;
  return true;
}

bool parse_pair(std::string_view text, std::uint32_t& first, std::uint32_t& second) {
  const std::size_t colon = text.find(':');
  if (colon == std::string_view::npos) {
    return false;
  }
  return parse_u32(text.substr(0, colon), first) &&
         parse_u32(text.substr(colon + 1), second);
}

bool is_supported_colorspace(std::string_view colorspace) {
  // All of these are 8-bit 4:2:0 with the same plane dimensions.
  return colorspace == "C420" || colorspace == "C420jpeg" ||
         colorspace == "C420mpeg2" || colorspace == "C420paldv";
}

}  // namespace

std::span<const std::uint8_t> Y4mFrame::y() const {
  const std::span<const std::uint8_t> all(data_);
  if (all.size() < luma_size()) {
    return {};
  }
  return all.first(luma_size());
}

std::span<const std::uint8_t> Y4mFrame::u() const {
  const std::span<const std::uint8_t> all(data_);
  if (all.size() < luma_size() + chroma_size()) {
    return {};
  }
  return all.subspan(luma_size(), chroma_size());
}

std::span<const std::uint8_t> Y4mFrame::v() const {
  const std::span<const std::uint8_t> all(data_);
  if (all.size() < luma_size() + 2U * chroma_size()) {
    return {};
  }
  return all.subspan(luma_size() + chroma_size(), chroma_size());
}

Y4mReader::Y4mReader(std::filesystem::path path) : path_(std::move(path)) {}

bool Y4mReader::open(std::string& error) {
  error.clear();
  if (file_.is_open()) {
    error = "Y4M reader is already open";
    return false;
  }

  file_.open(path_, std::ios::binary);
  if (!file_) {
    error = "cannot open Y4M input '" + path_.string() + "'";
    return false;
  }

  std::string line;
  if (!std::getline(file_, line)) {
    error = "failed to read Y4M header from '" + path_.string() + "'";
    file_.close();
    return false;
  }
  strip_cr(line);

  if (!parse_header_line(line, error)) {
    file_.close();
    return false;
  }

  frame_index_ = 0;
  open_ = true;
  return true;
}

bool Y4mReader::parse_header_line(const std::string& line, std::string& error) {
  std::istringstream stream(line);
  std::string token;
  if (!(stream >> token) || token != "YUV4MPEG2") {
    error = "not a Y4M file: missing YUV4MPEG2 signature";
    return false;
  }

  Y4mHeader header;
  bool saw_width = false;
  bool saw_height = false;
  bool saw_frame_rate = false;

  while (stream >> token) {
    if (token.size() < 2) {
      error = "malformed Y4M header token '" + token + "'";
      return false;
    }

    const char field = token.front();
    const std::string_view value(token.data() + 1, token.size() - 1);
    switch (field) {
      case 'W':
        if (!parse_u32(value, header.width) || header.width == 0) {
          error = "invalid Y4M width '" + std::string(value) + "'";
          return false;
        }
        saw_width = true;
        break;
      case 'H':
        if (!parse_u32(value, header.height) || header.height == 0) {
          error = "invalid Y4M height '" + std::string(value) + "'";
          return false;
        }
        saw_height = true;
        break;
      case 'F':
        if (!parse_pair(value, header.fps_num, header.fps_den) || header.fps_num == 0 ||
            header.fps_den == 0) {
          error = "invalid Y4M frame rate '" + std::string(value) + "'";
          return false;
        }
        saw_frame_rate = true;
        break;
      case 'I':
        if (value.size() != 1) {
          error = "invalid Y4M interlacing field 'I" + std::string(value) + "'";
          return false;
        }
        header.interlacing = value[0];
        if (header.interlacing != 'p' && header.interlacing != 't' &&
            header.interlacing != 'b' && header.interlacing != 'm' &&
            header.interlacing != '?') {
          error = "invalid Y4M interlacing character";
          return false;
        }
        break;
      case 'A':
        if (!parse_pair(value, header.aspect_num, header.aspect_den)) {
          error = "invalid Y4M aspect ratio '" + std::string(value) + "'";
          return false;
        }
        break;
      case 'C':
        // Keep the full token (e.g. "C420jpeg") so callers can report exactly
        // what the input declared.
        header.colorspace = token;
        break;
      default:
        error = "unsupported Y4M header token '" + token + "'";
        return false;
    }
  }

  if (!saw_width || !saw_height || !saw_frame_rate) {
    error = "incomplete Y4M header: W, H and F fields are required";
    return false;
  }

  if (!is_supported_colorspace(header.colorspace)) {
    error = "unsupported Y4M colorspace '" + header.colorspace +
            "': av2enc supports 8-bit 4:2:0 (C420) only";
    return false;
  }

  if (header.interlacing != 'p' && header.interlacing != '?') {
    error = "unsupported interlaced Y4M input (I" +
            std::string(1, header.interlacing) +
            "): av2enc currently expects progressive (Ip) input";
    return false;
  }

  header_ = std::move(header);
  return true;
}

Y4mReadResult Y4mReader::read_frame(Y4mFrame& frame, std::string& error) {
  error.clear();
  if (!open_) {
    error = "Y4M reader is not open";
    return Y4mReadResult::kError;
  }

  std::string line;
  if (!std::getline(file_, line)) {
    if (file_.eof()) {
      return Y4mReadResult::kEof;
    }
    error = "failed to read next Y4M frame marker";
    return Y4mReadResult::kError;
  }
  strip_cr(line);

  const bool marker_ok =
      line == "FRAME" || (line.size() > 6 && line.compare(0, 6, "FRAME ") == 0);
  if (!marker_ok) {
    std::string preview = line.substr(0, 40);
    error = "expected FRAME marker before frame " + std::to_string(frame_index_) +
            ", got '" + preview + "'";
    return Y4mReadResult::kError;
  }

  frame.width_ = header_.width;
  frame.height_ = header_.height;

  const std::size_t luma_size = frame.luma_size();
  const std::size_t chroma_size = frame.chroma_size();
  const std::size_t total_size = luma_size + 2U * chroma_size;
  if (total_size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    error = "Y4M frame is too large to read";
    return Y4mReadResult::kError;
  }

  frame.data_.resize(total_size);
  file_.read(reinterpret_cast<char*>(frame.data_.data()),
             static_cast<std::streamsize>(total_size));
  const std::streamsize bytes_read = file_.gcount();
  if (bytes_read != static_cast<std::streamsize>(total_size)) {
    error = "truncated Y4M frame " + std::to_string(frame_index_) + ": expected " +
            std::to_string(total_size) + " bytes, got " +
            std::to_string(bytes_read);
    return Y4mReadResult::kError;
  }

  ++frame_index_;
  return Y4mReadResult::kFrame;
}

}  // namespace av2enc
