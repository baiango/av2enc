#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>

namespace av2enc {

// IVF file header fields. The writer always stores version 0, header length 32
// and codec FourCC 'AV02'. `frame_count` is informational; the writer patches
// the real count into the file when close() succeeds.
struct IvfHeader {
  std::uint16_t width = 0;
  std::uint16_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 0;
  std::uint32_t frame_count = 0;
};

// IVF muxer.
//
// File header (32 bytes, little-endian):
//   'D','K','I','F', version u16, length u16, FourCC 'AV02',
//   width u16, height u16, rate u32, scale u32, frame_count u32, unused u32
// Per-frame header (12 bytes, little-endian):
//   payload_size u32, timestamp u64
//
// Timestamps are chosen by the caller. The CLI uses the frame index, i.e. one
// tick in the file's rate/scale time base.
class IvfWriter {
 public:
  IvfWriter() = default;
  ~IvfWriter();

  IvfWriter(const IvfWriter&) = delete;
  IvfWriter& operator=(const IvfWriter&) = delete;

  bool open(const std::filesystem::path& path, const IvfHeader& header, std::string& error);

  // Appends one frame. `payload` is written verbatim; it must already be a
  // complete frame payload for the codec identified by the file FourCC.
  bool write_frame(std::span<const std::uint8_t> payload,
                   std::uint64_t timestamp,
                   std::string& error);

  // Flushes the stream and patches the frame count at offset 24. Idempotent.
  bool close(std::string& error);

  bool is_open() const { return file_.is_open(); }
  std::uint32_t frame_count() const { return frame_count_; }

  // Total bytes written, including the 32-byte file header and all 12-byte
  // per-frame headers. Valid after a successful open(); updated per frame.
  std::uint64_t bytes_written() const { return bytes_written_; }

 private:
  std::ofstream file_;
  std::filesystem::path path_;
  std::uint32_t frame_count_ = 0;
  std::uint64_t bytes_written_ = 0;
};

}  // namespace av2enc
