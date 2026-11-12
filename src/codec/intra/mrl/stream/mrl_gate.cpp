// Minimal M1 + MRL stream gate.
//
// Writes one IVF per case plus the exact expected luma plane (loop filters are
// off, all residuals are zero, so decoded luma == MRL prediction).  The Python
// driver decodes each IVF with the pinned dav2d and compares the Y plane.
//
// Usage: mrl_gate --out DIR
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "codec/intra/mrl/stream/mrl_stream.h"
#include "core/io/ivf.h"

namespace ms = av2enc::still::mrl_stream;

namespace {

struct Case {
  const char* name;
  int width;
  int height;
  int target;
  bool adaptive;
  std::vector<ms::MinimalMrlLeaf> leaves;  // empty => default policy
};

bool write_y(const std::filesystem::path& path,
             const std::vector<std::uint8_t>& data) {
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  file.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(file);
}

bool write_ivf(const std::filesystem::path& path, int width, int height,
               const std::vector<std::uint8_t>& payload) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  std::string error;
  if (!writer.open(path, header, error)) {
    std::cerr << path << ": " << error << '\n';
    return false;
  }
  if (!writer.write_frame(payload, 0, error)) {
    std::cerr << path << ": " << error << '\n';
    return false;
  }
  if (!writer.close(error)) {
    std::cerr << path << ": " << error << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path out_dir = "mrl_gate_out";
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

  // 32x32 leaves, 64x64 frame.  Decode order is TL, TR, BL, BR.
  std::vector<ms::MinimalMrlLeaf> dir32 = {
      {1, 0, 0, false},  // origin: no reference lines
      {1, 0, 1, true},   // left neighbour, secondary line
      {1, 0, 2, false},  // above neighbour
      {1, 0, 3, true},   // both neighbours, line 3 + secondary
  };
  // Single 64x64 leaf at the origin: fillers on primary and secondary lines.
  std::vector<ms::MinimalMrlLeaf> origin64 = {{1, 0, 3, true}};
  // DC only: enable_mrls = 1 but no MRL symbol is coded.
  std::vector<ms::MinimalMrlLeaf> dc64 = {{0, 0, 0, false}};

  const Case cases[] = {
      {"mrl_dc_64x64", 64, 64, 64, false, dc64},
      {"mrl_dir_64x64_32", 64, 64, 32, false, dir32},
      {"mrl_dir_64x64_16", 64, 64, 16, false, {}},
      {"mrl_dir_64x64_8", 64, 64, 8, false, {}},
      {"mrl_dir_64x64_4", 64, 64, 4, false, {}},
      {"mrl_dir_64x64_16_adaptive", 64, 64, 16, true, {}},
      {"mrl_dir_128x128_sb", 128, 128, 32, false, {}},
      {"mrl_origin_64x64", 64, 64, 64, false, origin64},
  };

  for (const Case& test : cases) {
    ms::MinimalMrlConfig config;
    config.width = test.width;
    config.height = test.height;
    config.target_block_size = test.target;
    config.disable_cdf_update = !test.adaptive;
    config.leaves = test.leaves;
    ms::MinimalMrlFrame frame;
    if (!ms::write_minimal_mrl_frame(config, frame)) {
      std::cerr << test.name << ": " << frame.error << '\n';
      return 1;
    }
    const std::filesystem::path ivf = out_dir / (std::string(test.name) + ".ivf");
    const std::filesystem::path y =
        out_dir / (std::string(test.name) + ".y");
    if (!write_ivf(ivf, test.width, test.height, frame.temporal_unit)) return 1;
    if (!write_y(y, frame.recon_y)) return 1;
    std::printf("%s: blocks=%d mrl=%d sec=%d tile=%zu frame=%zu\n", test.name,
                frame.blocks, frame.mrl_blocks, frame.secondary_blocks,
                frame.tile_payload.size(), frame.temporal_unit.size());
  }
  std::printf("wrote MRL gate streams to %s\n", out_dir.string().c_str());
  return 0;
}
