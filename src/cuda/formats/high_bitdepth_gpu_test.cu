// CPU-vs-GPU bit-exactness test for the 10-bit / 4:2:2 / 4:4:4 / 4:0:0 block
// path.
//
// For each synthetic 10-bit picture:
//   * encode with the product CPU reference (`av2enc::hbd::encode_still`) and
//     the GPU block stage,
//   * require identical per-block EOB / scan-order levels / reconstruction,
//   * require identical final OBU bytes (entropy is shared CPU code, so a level
//     or block-order difference shows up here too).
//
// Coverage: every qindex in 1..303 (the §8.3.1 coefficient-CDF slices), exact
// 64x64 blocks, 100x100 odd/partial edges, and 640x360 with a partial bottom
// superblock row.
//
// Run on the sm_86 box with build_high_bitdepth_gpu_test.sh.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/formats/high_bitdepth_gpu.h"
#include "formats/format.h"
#include "formats/high_bitdepth/hbd_still.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"

namespace {

int failures = 0;
long long comparisons = 0;

void expect(bool condition, const char* what) {
  ++comparisons;
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

av2enc::fmt::Picture make_picture(std::uint32_t width, std::uint32_t height,
                                  av2enc::fmt::ChromaFormat chroma,
                                  std::uint32_t seed) {
  av2enc::fmt::Format format;
  format.width = width;
  format.height = height;
  format.chroma = chroma;
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
  const int planes = av2enc::fmt::num_planes(chroma);
  for (int plane = 0; plane < planes; ++plane) {
    const av2enc::fmt::PlaneBuffer& p = picture.planes[plane];
    for (std::uint32_t y = 0; y < p.visible_height; ++y) {
      for (std::uint32_t x = 0; x < p.visible_width; ++x) {
        const std::uint32_t gradient =
            (x * (5U + static_cast<std::uint32_t>(plane) * 2U) +
             y * (3U + static_cast<std::uint32_t>(plane)) + seed * 17U) &
            1023U;
        picture.planes[plane].at(x, y) =
            static_cast<std::uint16_t>((gradient + (next() & 63U)) & 1023U);
      }
    }
    av2enc::fmt::replicate_borders(picture.planes[plane]);
  }
  return picture;
}

void compare_result(const av2enc::hbd::HbdResult& cpu,
                    const av2enc::high_bitdepth::HighBitDepthResult& gpu,
                    const char* tag) {
  char label[256];
  std::snprintf(label, sizeof(label), "%s tile byte count", tag);
  expect(cpu.tile_bytes == gpu.tile_bytes, label);
  std::snprintf(label, sizeof(label), "%s frame byte count", tag);
  expect(cpu.frame.size() == gpu.frame.size(), label);
  std::snprintf(label, sizeof(label), "%s final OBU bytes", tag);
  expect(cpu.frame == gpu.frame, label);
  std::snprintf(label, sizeof(label), "%s block count", tag);
  expect(cpu.blocks.size() == gpu.blocks.size(), label);
  std::snprintf(label, sizeof(label), "%s level count", tag);
  expect(cpu.levels.size() == gpu.levels.size(), label);
  if (cpu.levels.size() == gpu.levels.size()) {
    for (std::size_t i = 0; i < cpu.levels.size(); ++i) {
      if (cpu.levels[i] != gpu.levels[i]) {
        std::fprintf(stderr, "FAIL: %s level %zu cpu=%d gpu=%d\n", tag, i,
                     cpu.levels[i], gpu.levels[i]);
        ++failures;
        break;
      }
    }
    ++comparisons;
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
        std::fprintf(stderr, "FAIL: %s block %zu descriptor mismatch\n", tag,
                     i);
        ++failures;
        break;
      }
    }
    comparisons += static_cast<long long>(cpu.blocks.size()) * 10;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const auto& a = cpu.recon[plane];
    const auto& b = gpu.recon[plane];
    if (a.samples.empty() && b.samples.empty()) continue;
    std::snprintf(label, sizeof(label), "%s recon plane %d size", tag, plane);
    expect(a.width == b.width && a.height == b.height, label);
    if (a.width != b.width || a.height != b.height) continue;
    std::snprintf(label, sizeof(label), "%s recon plane %d coded size", tag,
                  plane);
    expect(a.coded_width == b.coded_width &&
               a.coded_height == b.coded_height,
           label);
    for (int y = 0; y < a.height; ++y) {
      for (int x = 0; x < a.width; ++x) {
        const std::uint16_t av =
            a.samples[static_cast<std::size_t>(y) * a.width + x];
        const std::uint16_t bv =
            b.samples[static_cast<std::size_t>(y) * b.width + x];
        if (av != bv) {
          std::fprintf(stderr,
                       "FAIL: %s recon plane %d (%d,%d) cpu=%u gpu=%u\n", tag,
                       plane, x, y, static_cast<unsigned>(av),
                       static_cast<unsigned>(bv));
          ++failures;
          return;
        }
        ++comparisons;
      }
    }
  }
}

void compare_case(std::uint32_t width, std::uint32_t height,
                  av2enc::fmt::ChromaFormat chroma, int qindex,
                  std::uint32_t seed) {
  const av2enc::fmt::Picture picture =
      make_picture(width, height, chroma, seed);

  av2enc::hbd::HbdOptions cpu_options;
  cpu_options.base_q_idx = static_cast<std::uint16_t>(qindex);
  av2enc::high_bitdepth::HighBitDepthOptions gpu_options;
  gpu_options.base_q_idx = static_cast<std::uint16_t>(qindex);

  av2enc::hbd::HbdResult cpu;
  av2enc::high_bitdepth::HighBitDepthResult gpu;
  std::string cpu_error;
  std::string gpu_error;
  char tag[160];
  std::snprintf(tag, sizeof(tag), "%s %ux%u q=%d seed=%u",
                av2enc::fmt::chroma_format_name(chroma), width, height, qindex,
                seed);
  expect(av2enc::hbd::encode_still(picture, cpu_options, cpu, cpu_error),
         cpu_error.c_str());
  if (!cpu_error.empty()) return;

  av2enc::cuda::high_bitdepth::HighBitDepthGpuConfig arena_config;
  arena_config.width = width;
  arena_config.height = height;
  arena_config.mi_cols = picture.geometry.mi_cols;
  arena_config.mi_rows = picture.geometry.mi_rows;
  arena_config.format = picture.format;
  arena_config.bit_depth = 10;
  const std::uint64_t need =
      av2enc::cuda::high_bitdepth::required_device_bytes(arena_config);
  av2enc::cuda::Arena arena;
  std::string arena_error;
  expect(av2enc::cuda::is_ok(
             arena.init(need, av2enc::cuda::Arena::kAlignment, &arena_error)),
         arena_error.c_str());
  if (!arena_error.empty()) return;
  const av2enc::cuda::high_bitdepth::HighBitDepthGpuStatus status =
      av2enc::cuda::high_bitdepth::encode_high_bitdepth_gpu(
          arena, picture, gpu_options, gpu, &gpu_error);
  expect(status.ok(), gpu_error.c_str());
  if (!status.ok()) return;

  compare_result(cpu, gpu, tag);
}

void run_format(av2enc::fmt::ChromaFormat chroma) {
  // Full qindex sweep at the exact-block size.  Each case exercises a
  // different §8.3.1 coefficient-CDF slice and qlookup step.
  for (int q = 1; q <= 303; ++q) {
    compare_case(64, 64, chroma, q, static_cast<std::uint32_t>(q));
  }
  // Odd dimensions and partial superblocks (edge leaves / padding).
  for (int q : {1, 45, 90, 91, 140, 141, 190, 191, 303}) {
    compare_case(100, 100, chroma, q, static_cast<std::uint32_t>(q + 1000));
    compare_case(640, 360, chroma, q, static_cast<std::uint32_t>(q + 2000));
  }
}

}  // namespace

int main() {
  run_format(av2enc::fmt::ChromaFormat::k420);
  run_format(av2enc::fmt::ChromaFormat::k422);
  run_format(av2enc::fmt::ChromaFormat::k444);
  run_format(av2enc::fmt::ChromaFormat::k400);
  if (failures == 0) {
    std::printf("high_bitdepth_cuda_test: PASS (%lld comparisons)\n",
                comparisons);
    return 0;
  }
  std::fprintf(stderr, "high_bitdepth_cuda_test: %d failure(s)\n", failures);
  return 1;
}
