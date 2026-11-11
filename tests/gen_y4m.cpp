// Deterministic 8-bit 4:2:0 Y4M generator for encoder edge-geometry tests.
//
// Usage: gen_y4m <width> <height> <output.y4m>
//
// The pattern is pure integer arithmetic so the file (and therefore the coded
// stream) is identical on every host. It has enough texture to produce
// non-zero luma and chroma residuals in every coding block.
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <width> <height> <out.y4m>\n", argv[0]);
    return 2;
  }
  char* end = nullptr;
  const long width_long = std::strtol(argv[1], &end, 10);
  if (end == nullptr || *end != '\0' || width_long <= 0 || width_long > 65535) {
    std::fprintf(stderr, "bad width\n");
    return 2;
  }
  const int width = static_cast<int>(width_long);
  const long height_long = std::strtol(argv[2], &end, 10);
  if (end == nullptr || *end != '\0' || height_long <= 0 ||
      height_long > 65535) {
    std::fprintf(stderr, "bad height\n");
    return 2;
  }
  const int height = static_cast<int>(height_long);
  if ((width & 1) != 0 || (height & 1) != 0) {
    std::fprintf(stderr, "4:2:0 dimensions must be even\n");
    return 2;
  }

  std::FILE* out = std::fopen(argv[3], "wb");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", argv[3]);
    return 1;
  }
  std::fprintf(out, "YUV4MPEG2 W%d H%d F1:1 Ip A1:1 C420\nFRAME\n", width,
               height);
  const int cw = width / 2;
  const int ch = height / 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int value =
          (x * 37 + y * 19 + ((x >> 2) * (y >> 3)) * 11 + 7) & 255;
      const std::uint8_t sample = static_cast<std::uint8_t>(value);
      std::fwrite(&sample, 1, 1, out);
    }
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const std::uint8_t u =
          static_cast<std::uint8_t>((x * 11 + y * 7 + (x >> 3) * 29) & 255);
      std::fwrite(&u, 1, 1, out);
    }
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const std::uint8_t v =
          static_cast<std::uint8_t>((x * 5 + y * 23 + (y >> 3) * 17) & 255);
      std::fwrite(&v, 1, 1, out);
    }
  }
  std::fclose(out);
  return 0;
}
