// Reproduction of the production M1 writer's sub-8 rejection.
//
// The production writer (src/encoder/m1/still_syntax.cpp) codes a UV mode and a
// chroma residual for *every* leaf.  When a 4x4/4x8/8x4 leaf appears, the
// decoder expects HasChroma=0 on all but the last leaf of the 8x8 chroma
// group, so the symbol stream desynchronises.  This program writes such a
// stream with the unmodified M1 encoder; the shell script then runs dav2d and
// reports the result (it is a diagnostic, not a pass/fail test: it starts
// passing once the call sites documented in README.md are applied).
#include "pipeline/still/still_encoder.h"

#include "core/io/ivf.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const char* output = argc > 1 ? argv[1] : "still_sub8_repro.ivf";
  const int width = argc > 2 ? std::atoi(argv[2]) : 8;
  const int height = argc > 3 ? std::atoi(argv[3]) : 8;

  av2enc::still::StillPicture picture;
  picture.width = static_cast<std::uint32_t>(width);
  picture.height = static_cast<std::uint32_t>(height);
  std::vector<std::uint8_t> y(static_cast<std::size_t>(width) * height, 128);
  std::vector<std::uint8_t> u(
      static_cast<std::size_t>(width / 2) * (height / 2), 128);
  std::vector<std::uint8_t> v(u.size(), 128);
  // A 4x4 checkerboard makes the partition solver pick sub-8 leaves.
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const bool white = (((row >> 2) + (col >> 2)) & 1) != 0;
      y[static_cast<std::size_t>(row) * width + col] = white ? 220 : 40;
    }
  }
  picture.y = y;
  picture.u = u;
  picture.v = v;

  av2enc::still::StillOptions options;
  options.base_q_idx = 90;
  options.min_block_size = 4;      // allow 4x4/4x8/8x4 leaves
  options.max_partition_depth = 8; // walk the forced boundary chain
  options.force_dc = true;

  av2enc::still::StillResult result;
  std::string error;
  if (!av2enc::still::encode_still_picture(picture, options, result, error)) {
    std::fprintf(stderr, "encode failed: %s\n", error.c_str());
    return 1;
  }

  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(output, header, error) ||
      !writer.write_frame(result.frame, 0, error) || !writer.close(error)) {
    std::fprintf(stderr, "ivf failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("production M1: leaves=%d frame=%u tile=%u -> %s\n", result.leaves,
              result.frame_bytes, result.tile_bytes, output);
  return 0;
}
