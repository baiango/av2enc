// CPU-vs-GPU bit-exactness test for the standalone M10 10-bit block path.
//
// For each synthetic 10-bit 4:2:0 picture:
//   * encode with the CPU reference and the GPU block stage,
//   * require identical per-block EOB / scan-order levels / reconstruction,
//   * require identical final OBU bytes (entropy is shared CPU code, so a level
//     or block-order difference shows up here too).
//
// Run on the sm_86 box with build_m10_gpu_test.sh.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda/format/m10_gpu.h"
#include "encoder/format/format.h"
#include "encoder/m10/m10_still.h"

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

av2enc::fmt::Picture make_picture(std::uint32_t width, std::uint32_t height,
                                  std::uint32_t seed) {
  av2enc::fmt::Format format;
  format.width = width;
  format.height = height;
  format.chroma = av2enc::fmt::ChromaFormat::k420;
  format.depth = av2enc::fmt::BitDepth::k10;
  av2enc::fmt::Picture picture;
  std::string error;
  if (!av2enc::fmt::allocate_picture(format, picture, error)) {
    std::fprintf(stderr, "allocate_picture: %s\n", error.c_str());
    std::abort();
  }
  std::uint32_t state = seed * 2654435761U + 1U;
  const auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t gradient = (x * 5U + y * 3U + seed * 17U) & 1023U;
      const std::uint32_t noise = next() & 63U;
      picture.planes[0].at(x, y) =
          static_cast<std::uint16_t>((gradient + noise) & 1023U);
    }
  }
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t x = 0; x < cw; ++x) {
      picture.planes[1].at(x, y) =
          static_cast<std::uint16_t>((x * 7U + y * 11U + seed * 31U) & 1023U);
      picture.planes[2].at(x, y) =
          static_cast<std::uint16_t>((x * 13U + y * 2U + seed * 47U) & 1023U);
    }
  }
  av2enc::fmt::replicate_borders(picture.planes[0]);
  av2enc::fmt::replicate_borders(picture.planes[1]);
  av2enc::fmt::replicate_borders(picture.planes[2]);
  return picture;
}

void compare_case(std::uint32_t width, std::uint32_t height,
                  std::uint32_t seed) {
  const av2enc::fmt::Picture picture = make_picture(width, height, seed);
  av2enc::m10::M10Options options;
  options.base_q_idx = 90;

  av2enc::m10::M10Result cpu;
  av2enc::m10::M10Result gpu;
  std::string cpu_error;
  std::string gpu_error;
  expect(av2enc::m10::encode_m10_cpu(picture, options, cpu, cpu_error),
         cpu_error.c_str());
  expect(av2enc::cuda::m10::encode_m10_gpu(picture, options, gpu, &gpu_error),
         gpu_error.c_str());
  if (!cpu_error.empty() || !gpu_error.empty()) return;

  std::printf("case %ux%u seed=%u: cpu tile=%u blocks=%zu\n", width, height,
              seed, cpu.tile_bytes, cpu.blocks.size());
  expect(cpu.tile_bytes == gpu.tile_bytes, "tile byte count");
  expect(cpu.frame.size() == gpu.frame.size(), "frame byte count");
  expect(cpu.frame == gpu.frame, "final OBU bytes");
  expect(cpu.blocks.size() == gpu.blocks.size(), "block count");
  expect(cpu.levels.size() == gpu.levels.size(), "level count");
  if (cpu.levels.size() == gpu.levels.size()) {
    for (std::size_t i = 0; i < cpu.levels.size(); ++i) {
      if (cpu.levels[i] != gpu.levels[i]) {
        std::fprintf(stderr, "FAIL: level %zu cpu=%d gpu=%d\n", i, cpu.levels[i],
                     gpu.levels[i]);
        ++failures;
        break;
      }
    }
  }
  if (cpu.blocks.size() == gpu.blocks.size()) {
    for (std::size_t i = 0; i < cpu.blocks.size(); ++i) {
      const auto& a = cpu.blocks[i];
      const auto& b = gpu.blocks[i];
      if (a.mi_row != b.mi_row || a.mi_col != b.mi_col ||
          a.luma_size != b.luma_size || a.levels_offset != b.levels_offset ||
          a.eob[0] != b.eob[0] || a.eob[1] != b.eob[1] ||
          a.eob[2] != b.eob[2] || a.tx_size[0] != b.tx_size[0] ||
          a.tx_size[1] != b.tx_size[1] || a.tx_size[2] != b.tx_size[2]) {
        std::fprintf(stderr, "FAIL: block %zu descriptor mismatch\n", i);
        ++failures;
        break;
      }
    }
  }
  for (int plane = 0; plane < 3; ++plane) {
    const auto& a = cpu.recon[plane];
    const auto& b = gpu.recon[plane];
    expect(a.width == b.width && a.height == b.height, "recon storage size");
    expect(a.coded_width == b.coded_width &&
               a.coded_height == b.coded_height,
           "recon coded size");
    if (a.width == b.width && a.height == b.height) {
      for (int y = 0; y < a.coded_height; ++y) {
        for (int x = 0; x < a.coded_width; ++x) {
          const std::uint16_t av =
              a.samples[static_cast<std::size_t>(y) * a.width + x];
          const std::uint16_t bv =
              b.samples[static_cast<std::size_t>(y) * b.width + x];
          if (av != bv) {
            std::fprintf(stderr,
                         "FAIL: recon plane %d (%d,%d) cpu=%u gpu=%u\n", plane,
                         x, y, static_cast<unsigned>(av),
                         static_cast<unsigned>(bv));
            ++failures;
            return;
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  compare_case(64, 64, 1);
  compare_case(64, 64, 2);
  compare_case(128, 64, 3);
  compare_case(100, 100, 4);   // partial superblocks on both edges
  compare_case(640, 360, 5);
  if (failures == 0) {
    std::printf("m10_cuda_test: PASS\n");
    return 0;
  }
  std::fprintf(stderr, "m10_cuda_test: %d failure(s)\n", failures);
  return 1;
}
