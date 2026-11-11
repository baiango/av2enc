// Standalone 10-bit 4:2:0 still encoder CLI for `src/encoder/hbd/`.
//
//   hbd_encode <input.y4m> <output.ivf> [recon.y4m] [--q N]
//
// This is a component tool, not the product CLI.  Build it with
// `build_and_test.sh`.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/hbd/hbd_depth.h"
#include "encoder/hbd/hbd_still.h"
#include "encoder/m10/m10_y4m.h"
#include "io/ivf.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: hbd_encode <input.y4m> <output.ivf> [recon.y4m] "
               "[--q N]\n"
               "  input: 8-bit C420 or 10-bit C420p10 Y4M (first frame)\n"
               "  --q:   raw base_q_idx (1..90 for the reused slice-0 entropy "
               "writer)\n");
}

bool write_recon_y4m(const std::string& path, const av2enc::hbd::HbdResult& r,
                     std::string* error) {
  return av2enc::m10::write_y4m_10bit_planes(
      path, static_cast<std::uint32_t>(r.recon[0].visible_width),
      static_cast<std::uint32_t>(r.recon[0].visible_height),
      r.recon[0].samples.data(), r.recon[0].width,
      r.recon[1].samples.data(), r.recon[1].width,
      r.recon[2].samples.data(), r.recon[2].width, r.recon[1].visible_width,
      r.recon[1].visible_height, error);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> positional;
  av2enc::hbd::HbdOptions options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--q") == 0 && i + 1 < argc) {
      options.base_q_idx = static_cast<std::uint16_t>(std::atoi(argv[++i]));
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      usage();
      return 0;
    } else {
      positional.push_back(argv[i]);
    }
  }
  if (positional.size() < 2 || positional.size() > 3) {
    usage();
    return 2;
  }

  av2enc::fmt::Picture picture;
  av2enc::m10::Y4mFrameInfo info;
  std::string error;
  if (!av2enc::m10::read_y4m_10bit(positional[0], &picture, &info, &error)) {
    std::fprintf(stderr, "hbd_encode: %s\n", error.c_str());
    return 1;
  }

  av2enc::hbd::HbdResult result;
  if (!av2enc::hbd::encode_still(picture, options, result, error)) {
    std::fprintf(stderr, "hbd_encode: %s\n", error.c_str());
    return 1;
  }

  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(info.width);
  header.height = static_cast<std::uint16_t>(info.height);
  header.fps_num = 1;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(positional[1], header, error) ||
      !writer.write_frame(result.frame, 0, error) ||
      !writer.close(error)) {
    std::fprintf(stderr, "hbd_encode: %s\n", error.c_str());
    return 1;
  }

  if (positional.size() == 3) {
    if (!write_recon_y4m(positional[2], result, &error)) {
      std::fprintf(stderr, "hbd_encode: %s\n", error.c_str());
      return 1;
    }
  }

  std::printf(
      "hbd: %ux%u %s, q=%u (step=%d, qctx=%d), blocks=%zu, tile=%u B, "
      "frame=%u B, recon=%ux%u\n",
      info.width, info.height, info.bit_depth == 10 ? "C420p10" : "C420",
      static_cast<unsigned>(options.base_q_idx), result.quant_step,
      av2enc::hbd::depth::qctx_slice(options.base_q_idx),
      result.blocks.size(), static_cast<unsigned>(result.tile_bytes),
      static_cast<unsigned>(result.frame_bytes),
      static_cast<unsigned>(result.recon[0].visible_width),
      static_cast<unsigned>(result.recon[0].visible_height));
  return 0;
}
