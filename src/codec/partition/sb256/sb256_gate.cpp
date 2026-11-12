// Emit minimal use_256x256_superblock IVF streams for the dav2d gate.
//
//   sb256_gate --out DIR
//
// Writes one all-DC, no-residual intra frame per geometry.  Per §5.18.1 the
// intra frame's SbSize is 128, so the tile is a 128x128-superblock tree even
// though the sequence header carries use_256x256_superblock = 1.  A second
// 128x128 stream under use_256=0/use_128=1 is written as a positive control.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "codec/partition/sb256/sb256_stream.h"
#include "core/io/ivf.h"

namespace {

struct Case {
  int width;
  int height;
  const char* name;
};

constexpr Case kCases[] = {
    {256, 256, "sb256_256x256"},
    {640, 360, "sb256_640x360"},
    {1920, 1080, "sb256_1920x1080"},
    // Partial superblocks on both axes (8 px right remainder, 48 px bottom).
    {520, 300, "sb256_520x300"},
};

bool write_ivf(const std::filesystem::path& path,
               const av2enc::sb256::StreamResult& stream, int width, int height,
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

bool build_and_write(const av2enc::sb256::StreamConfig& config,
                     const std::filesystem::path& path, bool use_256,
                     std::string& error) {
  av2enc::sb256::StreamResult first;
  if (use_256) {
    if (!av2enc::sb256::make_sb256_stream(config, first, error)) return false;
  } else {
    if (!av2enc::sb256::make_use128_control_stream(config, first, error)) {
      return false;
    }
  }
  av2enc::sb256::StreamResult second;
  if (use_256) {
    if (!av2enc::sb256::make_sb256_stream(config, second, error)) return false;
  } else {
    if (!av2enc::sb256::make_use128_control_stream(config, second, error)) {
      return false;
    }
  }
  if (first.frame != second.frame || first.tile != second.tile) {
    error = "non-deterministic stream";
    return false;
  }
  if (!write_ivf(path, first, config.width, config.height, error)) return false;
  std::cout << path.filename().string() << ": " << first.frame.size() << " B, "
            << first.tile.size() << " tile B, " << first.leaves << " leaves, "
            << first.superblocks << " superblocks, " << first.symbols
            << " partition symbols\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path out_dir = "sb256_gate_out";
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
    av2enc::sb256::StreamConfig config;
    config.width = static_cast<std::uint16_t>(c.width);
    config.height = static_cast<std::uint16_t>(c.height);
    config.base_q_idx = 90;
    config.disable_cdf_update = true;
    std::string error;
    if (!build_and_write(config, out_dir / (std::string(c.name) + ".ivf"),
                         /*use_256=*/true, error)) {
      std::cerr << c.name << ": " << error << '\n';
      return 1;
    }
  }

  // Positive control: identical intra tile under use_128x128_superblock = 1.
  av2enc::sb256::StreamConfig control;
  control.width = 128;
  control.height = 128;
  std::string error;
  if (!build_and_write(control, out_dir / "sb256_control_use128_128x128.ivf",
                       /*use_256=*/false, error)) {
    std::cerr << "control: " << error << '\n';
    return 1;
  }
  return 0;
}
