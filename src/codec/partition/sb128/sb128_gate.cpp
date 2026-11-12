// Emit minimal 128x128-superblock IVF streams for the dav2d gate.
//
//   sb128_gate --out DIR
//
// Writes one all-DC, no-residual intra frame per geometry.  The reference tree
// is the same one sb128_test checks: full 128 roots use SPLIT, 128x64 uses
// VERT, and 1080p's bottom row uses the boundary-implied HORZ root.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "codec/partition/sb128/sb128_stream.h"
#include "core/io/ivf.h"

namespace {

struct Case {
  int width;
  int height;
  const char* name;
};

constexpr Case kCases[] = {
    {128, 128, "sb128_128x128"},
    {256, 256, "sb128_256x256"},
    {640, 360, "sb128_640x360"},
    {1920, 1080, "sb128_1920x1080"},
};

bool write_ivf(const std::filesystem::path& path,
               const av2enc::sb128::StreamResult& stream, int width, int height,
               std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(path, header, error)) return false;
  if (!writer.write_frame(stream.frame, 0, error)) return false;
  return writer.close(error);
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path out_dir = "sb128_gate_out";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_dir = argv[++i];
    } else {
      std::cerr << "usage: " << argv[0] << " --out DIR\n";
      return 2;
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "cannot create " << out_dir << ": " << ec.message() << '\n';
    return 2;
  }

  for (const Case& c : kCases) {
    av2enc::sb128::StreamConfig config;
    config.width = static_cast<std::uint16_t>(c.width);
    config.height = static_cast<std::uint16_t>(c.height);
    config.base_q_idx = 90;
    config.disable_cdf_update = true;

    av2enc::sb128::StreamResult first;
    std::string error;
    if (!av2enc::sb128::make_sb128_stream(config, first, error)) {
      std::cerr << c.name << ": " << error << '\n';
      return 1;
    }
    av2enc::sb128::StreamResult second;
    if (!av2enc::sb128::make_sb128_stream(config, second, error)) {
      std::cerr << c.name << ": " << error << '\n';
      return 1;
    }
    if (first.frame != second.frame || first.tile != second.tile) {
      std::cerr << c.name << ": non-deterministic stream\n";
      return 1;
    }

    const std::filesystem::path path =
        out_dir / (std::string(c.name) + ".ivf");
    if (!write_ivf(path, first, c.width, c.height, error)) {
      std::cerr << c.name << ": " << error << '\n';
      return 1;
    }
    std::cout << c.name << ": " << first.frame.size() << " B, "
              << first.tile.size() << " tile B, " << first.leaves << " leaves, "
              << first.symbols << " partition symbols, " << path << '\n';
  }
  return 0;
}
