#include "core/io/ivf.h"

#include <array>
#include <cstddef>
#include <limits>
#include <span>

namespace av2enc {
namespace {

constexpr std::size_t kFileHeaderSize = 32;
constexpr std::size_t kFrameHeaderSize = 12;
constexpr std::size_t kFrameCountOffset = 24;

void store_le16(std::span<std::uint8_t> out, std::size_t offset, std::uint16_t value) {
  out[offset] = static_cast<std::uint8_t>(value & 0xFFU);
  out[offset + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
}

void store_le32(std::span<std::uint8_t> out, std::size_t offset, std::uint32_t value) {
  out[offset] = static_cast<std::uint8_t>(value & 0xFFU);
  out[offset + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
  out[offset + 2] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
  out[offset + 3] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
}

void store_le64(std::span<std::uint8_t> out, std::size_t offset, std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i) {
    out[offset + i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFU);
  }
}

}  // namespace

IvfWriter::~IvfWriter() {
  if (file_.is_open()) {
    std::string ignored;
    close(ignored);
  }
}

bool IvfWriter::open(const std::filesystem::path& path,
                     const IvfHeader& header,
                     std::string& error) {
  error.clear();
  if (file_.is_open()) {
    error = "IVF writer is already open";
    return false;
  }
  if (header.width == 0 || header.height == 0) {
    error = "IVF width and height must be non-zero";
    return false;
  }
  if (header.fps_num == 0 || header.fps_den == 0) {
    error = "IVF frame rate must be non-zero";
    return false;
  }

  file_.open(path, std::ios::binary | std::ios::trunc);
  if (!file_) {
    error = "cannot open IVF output '" + path.string() + "'";
    return false;
  }

  path_ = path;
  frame_count_ = 0;
  bytes_written_ = kFileHeaderSize;

  std::array<std::uint8_t, kFileHeaderSize> bytes{};
  bytes[0] = 'D';
  bytes[1] = 'K';
  bytes[2] = 'I';
  bytes[3] = 'F';
  store_le16(bytes, 4, 0);  // version
  store_le16(bytes, 6, static_cast<std::uint16_t>(kFileHeaderSize));
  bytes[8] = 'A';
  bytes[9] = 'V';
  bytes[10] = '0';
  bytes[11] = '2';
  store_le16(bytes, 12, header.width);
  store_le16(bytes, 14, header.height);
  store_le32(bytes, 16, header.fps_num);
  store_le32(bytes, 20, header.fps_den);
  store_le32(bytes, 24, 0);  // patched by close()
  store_le32(bytes, 28, 0);  // unused

  file_.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!file_) {
    error = "failed to write IVF file header to '" + path.string() + "'";
    file_.close();
    return false;
  }

  return true;
}

bool IvfWriter::write_frame(std::span<const std::uint8_t> payload,
                            std::uint64_t timestamp,
                            std::string& error) {
  error.clear();
  if (!file_.is_open()) {
    error = "IVF writer is not open";
    return false;
  }
  if (payload.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    error = "IVF frame payload exceeds the 32-bit size field";
    return false;
  }

  std::array<std::uint8_t, kFrameHeaderSize> frame_header{};
  store_le32(frame_header, 0, static_cast<std::uint32_t>(payload.size()));
  store_le64(frame_header, 4, timestamp);

  file_.write(reinterpret_cast<const char*>(frame_header.data()),
              static_cast<std::streamsize>(frame_header.size()));
  if (!payload.empty()) {
    file_.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
  }
  if (!file_) {
    error = "failed to write IVF frame " + std::to_string(frame_count_) + " to '" +
            path_.string() + "'";
    return false;
  }

  ++frame_count_;
  bytes_written_ += kFrameHeaderSize + payload.size();
  return true;
}

bool IvfWriter::close(std::string& error) {
  error.clear();
  if (!file_.is_open()) {
    return true;
  }

  file_.flush();
  if (!file_) {
    error = "failed to flush IVF output '" + path_.string() + "'";
    file_.close();
    return false;
  }

  file_.seekp(static_cast<std::streamoff>(kFrameCountOffset), std::ios::beg);
  std::array<std::uint8_t, 4> count{};
  store_le32(count, 0, frame_count_);
  file_.write(reinterpret_cast<const char*>(count.data()),
              static_cast<std::streamsize>(count.size()));
  file_.flush();
  if (!file_) {
    error = "failed to patch IVF frame count in '" + path_.string() + "'";
    file_.close();
    return false;
  }

  file_.close();
  if (file_.fail()) {
    error = "failed to close IVF output '" + path_.string() + "'";
    return false;
  }

  return true;
}

}  // namespace av2enc
