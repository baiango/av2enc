// Standalone data-flow test for the M1 GPU orchestrator.
//
// For every input frame this checks:
//   plan -> host buffers -> GPU stages (or CPU reference fallback) -> frame
// and requires the fallback frame bytes and all three recon planes to be
// byte-identical to `av2enc::still::encode_still_picture`.
//
// Built and run by `build_still_gpu_test.sh` on the CUDA host.  The file uses a
// .cc extension so the root CMake GLOB_RECURSE (which picks up *.cu/*.cpp under
// src/cuda/m1) never folds this test's main() into libav2enc_core.
#include "cuda/pipeline/still/still_gpu.h"

#include "pipeline/still/still_encoder.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// MD5 (test-local, so the report is reproducible without shelling out)
// ---------------------------------------------------------------------------
class Md5 {
 public:
  void update(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    length_ += static_cast<std::uint64_t>(size);
    for (std::size_t i = 0; i < size; ++i) {
      buffer_[buffer_len_++] = bytes[i];
      if (buffer_len_ == 64) {
        transform(buffer_);
        buffer_len_ = 0;
      }
    }
  }

  void final(std::uint8_t digest[16]) {
    const std::uint64_t bit_length = length_ * 8u;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t zero = 0;
    while (buffer_len_ != 56) {
      update(&zero, 1);
    }
    std::uint8_t length_bytes[8];
    for (int i = 0; i < 8; ++i) {
      length_bytes[i] =
          static_cast<std::uint8_t>((bit_length >> (8 * i)) & 0xffu);
    }
    update(length_bytes, 8);
    const std::uint32_t words[4] = {state_[0], state_[1], state_[2],
                                    state_[3]};
    for (int i = 0; i < 4; ++i) {
      for (int b = 0; b < 4; ++b) {
        digest[i * 4 + b] =
            static_cast<std::uint8_t>((words[i] >> (8 * b)) & 0xffu);
      }
    }
  }

 private:
  static std::uint32_t rotl(std::uint32_t x, std::uint32_t c) {
    return (x << c) | (x >> (32u - c));
  }

  void transform(const std::uint8_t block[64]) {
    static constexpr std::uint32_t kK[64] = {
        0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu,
        0x4787c62au, 0xa8304613u, 0xfd469501u, 0x698098d8u, 0x8b44f7afu,
        0xffff5bb1u, 0x895cd7beu, 0x6b901122u, 0xfd987193u, 0xa679438eu,
        0x49b40821u, 0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau,
        0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u, 0x21e1cde6u,
        0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u,
        0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u,
        0xfde5380cu, 0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
        0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u, 0xd9d4d039u,
        0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u, 0xf4292244u, 0x432aff97u,
        0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u, 0xffeff47du,
        0x85845dd1u, 0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
        0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};
    static constexpr std::uint32_t kS[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

    std::uint32_t m[16];
    for (int i = 0; i < 16; ++i) {
      m[i] = static_cast<std::uint32_t>(block[i * 4]) |
             (static_cast<std::uint32_t>(block[i * 4 + 1]) << 8) |
             (static_cast<std::uint32_t>(block[i * 4 + 2]) << 16) |
             (static_cast<std::uint32_t>(block[i * 4 + 3]) << 24);
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    for (int i = 0; i < 64; ++i) {
      std::uint32_t f = 0;
      int g = 0;
      if (i < 16) {
        f = (b & c) | (~b & d);
        g = i;
      } else if (i < 32) {
        f = (d & b) | (~d & c);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        f = b ^ c ^ d;
        g = (3 * i + 5) % 16;
      } else {
        f = c ^ (b | ~d);
        g = (7 * i) % 16;
      }
      const std::uint32_t tmp = d;
      d = c;
      c = b;
      b = b + rotl(a + f + kK[i] + m[g], kS[i]);
      a = tmp;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
  }

  std::uint32_t state_[4] = {0x67452301u, 0xefcdab89u, 0x98badcfeu,
                             0x10325476u};
  std::uint8_t buffer_[64] = {};
  std::uint64_t length_ = 0;
  std::size_t buffer_len_ = 0;
};

std::string md5_hex(std::span<const std::uint8_t> data) {
  Md5 md5;
  md5.update(data.data(), data.size());
  std::uint8_t digest[16];
  md5.final(digest);
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (std::uint8_t byte : digest) {
    out.push_back(kHex[byte >> 4]);
    out.push_back(kHex[byte & 0x0fu]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Input helpers
// ---------------------------------------------------------------------------
struct Frame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_first_frame(const std::filesystem::path& path, Frame& frame,
                      std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) {
    return false;
  }
  av2enc::Y4mFrame y4m;
  const av2enc::Y4mReadResult read = reader.read_frame(y4m, error);
  if (read != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) {
      error = "input has no frame";
    }
    return false;
  }
  frame.width = y4m.width();
  frame.height = y4m.height();
  frame.y.assign(y4m.y().begin(), y4m.y().end());
  frame.u.assign(y4m.u().begin(), y4m.u().end());
  frame.v.assign(y4m.v().begin(), y4m.v().end());
  return true;
}

Frame synthetic_frame(std::uint32_t width, std::uint32_t height) {
  Frame frame;
  frame.width = width;
  frame.height = height;
  frame.y.resize(static_cast<std::size_t>(width) * height);
  frame.u.resize(static_cast<std::size_t>((width + 1u) / 2u) *
                 ((height + 1u) / 2u));
  frame.v.resize(frame.u.size());
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const int value = static_cast<int>((x * 3u + y * 5u + (x ^ y)) & 0xffu);
      frame.y[static_cast<std::size_t>(y) * width + x] =
          static_cast<std::uint8_t>(value);
    }
  }
  for (std::size_t i = 0; i < frame.u.size(); ++i) {
    frame.u[i] = static_cast<std::uint8_t>((i * 7u + 31u) & 0xffu);
    frame.v[i] = static_cast<std::uint8_t>((i * 11u + 97u) & 0xffu);
  }
  return frame;
}

av2enc::still::StillPicture to_picture(const Frame& frame) {
  av2enc::still::StillPicture picture;
  picture.width = frame.width;
  picture.height = frame.height;
  picture.y = frame.y;
  picture.u = frame.u;
  picture.v = frame.v;
  return picture;
}

bool same_bytes(std::span<const std::uint8_t> a,
                std::span<const std::uint8_t> b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

// ---------------------------------------------------------------------------
// Plan invariants
// ---------------------------------------------------------------------------
bool check_plan(const av2enc::cuda::still_gpu::GpuResult& result,
                std::string& error) {
  using av2enc::cuda::still_gpu::BlockPlan;
  const auto& plan = result.plan;
  if (plan.block_count == 0 || plan.blocks.size() != plan.block_count ||
      result.blocks.size() != plan.block_count) {
    error = "plan block count does not match the block array";
    return false;
  }
  if (plan.blocks.data() != result.blocks.data()) {
    error = "plan.blocks was not rebound to result.blocks";
    return false;
  }
  const std::uint32_t mi_cols = 2u * ((plan.width + 7u) >> 3);
  const std::uint32_t mi_rows = 2u * ((plan.height + 7u) >> 3);
  if (plan.mi_cols != mi_cols || plan.mi_rows != mi_rows) {
    error = "plan MiCols/MiRows do not follow §5.18.4.4";
    return false;
  }
  if (plan.y_stride % 64u != 0 || plan.u_stride * 2u != plan.y_stride ||
      plan.padded_height % 64u != 0) {
    error = "plan padded plane geometry is not superblock aligned";
    return false;
  }

  std::uint64_t last_coeff_y = 0;
  std::uint64_t last_coeff_u = 0;
  std::uint64_t last_coeff_v = 0;
  // Coverage on the 4x4 MI grid: every visible cell must be coded exactly
  // once.  Coded edge blocks may extend past the visible frame, so a plain
  // area sum is not a valid check.
  const std::size_t mi_w = mi_cols;
  const std::size_t mi_h = mi_rows;
  std::vector<std::uint8_t> luma_cover(mi_w * mi_h, 0);
  std::vector<std::uint8_t> chroma_cover((mi_w / 2) * (mi_h / 2), 0);
  for (const BlockPlan& block : result.blocks) {
    if (block.tx_w == 0 || block.tx_h == 0 || block.luma_w == 0 ||
        block.luma_h == 0) {
      error = "plan has a zero-sized block or transform";
      return false;
    }
    const std::uint64_t luma_end =
        static_cast<std::uint64_t>(block.plane_offset_y) +
        static_cast<std::uint64_t>(block.luma_h - 1u) * plan.y_stride +
        block.luma_w;
    if (luma_end > static_cast<std::uint64_t>(plan.y_stride) *
                      plan.padded_height) {
      error = "plan luma block leaves the padded plane";
      return false;
    }
    if (block.has_chroma != 0) {
      const std::uint64_t chroma_end =
          static_cast<std::uint64_t>(block.plane_offset_u) +
          static_cast<std::uint64_t>(block.chroma_h - 1u) * plan.u_stride +
          block.chroma_w;
      if (chroma_end > static_cast<std::uint64_t>(plan.u_stride) *
                           (plan.padded_height / 2u)) {
        error = "plan chroma block leaves the padded plane";
        return false;
      }
    }
    if (block.coeff_offset_y < last_coeff_y ||
        block.coeff_offset_u < last_coeff_u ||
        block.coeff_offset_v < last_coeff_v) {
      error = "plan coefficient offsets are not monotonic";
      return false;
    }
    last_coeff_y = static_cast<std::uint64_t>(block.coeff_offset_y) +
                   static_cast<std::uint64_t>(block.tx_w) * block.tx_h;
    last_coeff_u = static_cast<std::uint64_t>(block.coeff_offset_u) +
                   static_cast<std::uint64_t>(block.has_chroma != 0 ? 1u : 0u) *
                       block.chroma_tx_w * block.chroma_tx_h;
    last_coeff_v = static_cast<std::uint64_t>(block.coeff_offset_v) +
                   static_cast<std::uint64_t>(block.has_chroma != 0 ? 1u : 0u) *
                       block.chroma_tx_w * block.chroma_tx_h;
    if (last_coeff_y > plan.coeff_count_y || last_coeff_u > plan.coeff_count_u ||
        last_coeff_v > plan.coeff_count_v) {
      error = "plan coefficient offsets leave the coefficient arrays";
      return false;
    }

    const std::uint32_t bw4 = (block.luma_w + 3u) / 4u;
    const std::uint32_t bh4 = (block.luma_h + 3u) / 4u;
    for (std::uint32_t r = 0; r < bh4 && block.mi_row + r < mi_h; ++r) {
      for (std::uint32_t c = 0; c < bw4 && block.mi_col + c < mi_w; ++c) {
        const std::size_t cell =
            static_cast<std::size_t>(block.mi_row + r) * mi_w +
            block.mi_col + c;
        if (luma_cover[cell] != 0) {
          error = "plan luma blocks overlap";
          return false;
        }
        luma_cover[cell] = 1;
      }
    }
    if (block.has_chroma != 0) {
      const std::uint32_t cx = block.mi_col / 2u;
      const std::uint32_t cy = block.mi_row / 2u;
      const std::uint32_t cw4 = (block.chroma_w + 3u) / 4u;
      const std::uint32_t ch4 = (block.chroma_h + 3u) / 4u;
      for (std::uint32_t r = 0; r < ch4 && cy + r < mi_h / 2; ++r) {
        for (std::uint32_t c = 0; c < cw4 && cx + c < mi_w / 2; ++c) {
          const std::size_t cell =
              static_cast<std::size_t>(cy + r) * (mi_w / 2) + cx + c;
          if (chroma_cover[cell] != 0) {
            error = "plan chroma blocks overlap";
            return false;
          }
          chroma_cover[cell] = 1;
        }
      }
    }
  }
  if (std::find(luma_cover.begin(), luma_cover.end(), 0) !=
      luma_cover.end()) {
    error = "plan luma blocks do not cover the visible frame exactly once";
    return false;
  }
  if (std::find(chroma_cover.begin(), chroma_cover.end(), 0) !=
      chroma_cover.end()) {
    error = "plan chroma blocks do not cover the visible chroma exactly once";
    return false;
  }

  // Packed prediction / inverse-residual layout (filled by the orchestrator).
  std::uint64_t pred_y = 0;
  std::uint64_t pred_u = 0;
  std::uint64_t pred_v = 0;
  std::uint64_t resid_y = 0;
  std::uint64_t resid_u = 0;
  std::uint64_t resid_v = 0;
  for (const BlockPlan& block : result.blocks) {
    const std::uint64_t luma_area =
        static_cast<std::uint64_t>(block.luma_w) * block.luma_h;
    const std::uint64_t chroma_area =
        static_cast<std::uint64_t>(block.chroma_w) * block.chroma_h;
    if (block.pred_offset_y != pred_y ||
        block.residual_offset_y != resid_y ||
        block.pred_offset_u != pred_u || block.pred_offset_v != pred_v ||
        block.residual_offset_u != resid_u ||
        block.residual_offset_v != resid_v) {
      error = "plan packed prediction/residual offsets are not prefix sums";
      return false;
    }
    pred_y += luma_area;
    resid_y += luma_area;
    if (block.has_chroma != 0) {
      pred_u += chroma_area;
      pred_v += chroma_area;
      resid_u += chroma_area;
      resid_v += chroma_area;
    }
  }
  return true;
}

std::string stage_summary(const av2enc::cuda::still_gpu::GpuStageStatus& stages) {
  using av2enc::cuda::still_gpu::Stage;
  std::string out;
  for (Stage stage : {Stage::kPlan, Stage::kSearch, Stage::kTx, Stage::kRecon,
                      Stage::kChroma, Stage::kCfl, Stage::kDeblock,
                      Stage::kSched}) {
    if (!out.empty()) {
      out += " ";
    }
    out += av2enc::cuda::still_gpu::stage_name(stage);
    out += "=";
    out += av2enc::cuda::still_gpu::stage_state_name(stages.get(stage));
  }
  return out;
}

int failures = 0;
std::uint32_t g_vram_cap_mb = 0;

void fail(const std::string& message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++failures;
}

// ---------------------------------------------------------------------------
// One encode case
// ---------------------------------------------------------------------------
bool run_case(const std::string& name, const Frame& frame, bool verbose,
              bool require_gpu, bool stage1) {
  using av2enc::cuda::still_gpu::GpuOptions;
  using av2enc::cuda::still_gpu::GpuResult;
  using av2enc::cuda::still_gpu::StageState;

  const av2enc::still::StillPicture picture = to_picture(frame);

  // The CPU oracle uses the same mode switches as the GPU entry point.
  av2enc::still::StillOptions cpu_options;
  if (stage1) {
    cpu_options.force_dc = true;       // forced DC luma + DC chroma
    cpu_options.enable_cfl = false;
    cpu_options.enable_deblock = false;
  }
  av2enc::still::StillResult cpu;
  std::string cpu_error;
  if (!av2enc::still::encode_still_picture(picture, cpu_options, cpu,
                                           cpu_error)) {
    fail(name + ": CPU reference failed: " + cpu_error);
    return false;
  }

  GpuOptions options;
  options.verbose = verbose;
  options.verify_against_cpu = true;
  options.vram_cap_mb = g_vram_cap_mb;
  if (stage1) {
    options.enable_stage1_gpu = true;
    options.force_dc = true;
    options.enable_cfl = false;
    options.enable_deblock = false;
  }
  GpuResult result;
  std::string error;
  if (!av2enc::cuda::still_gpu::encode_still_gpu(picture, options, result, error)) {
    fail(name + ": encode_still_gpu failed: " + error);
    return false;
  }

  // The fallback contract: frame bytes and all recon planes are exact.
  if (!same_bytes(result.frame, cpu.frame)) {
    fail(name + ": frame bytes differ from the CPU reference");
  }
  if (result.frame_bytes != cpu.frame_bytes ||
      result.tile_bytes != cpu.tile_bytes) {
    fail(name + ": frame/tile byte counts differ from the CPU reference");
  }
  if (!same_bytes(result.recon_y, cpu.recon_y) ||
      !same_bytes(result.recon_u, cpu.recon_u) ||
      !same_bytes(result.recon_v, cpu.recon_v)) {
    fail(name + ": recon planes differ from the CPU reference");
  }

  std::string plan_error;
  if (!check_plan(result, plan_error)) {
    fail(name + ": " + plan_error);
  }
  if (result.stats.sync_count > 1) {
    fail(name + ": more than one synchronisation in a frame");
  }
  if (result.stats.block_count !=
      static_cast<int>(result.plan.block_count)) {
    fail(name + ": stats block count is inconsistent");
  }
  if (require_gpu && !result.stats.gpu_runtime_ready) {
    fail(name + ": persistent CUDA runtime is not ready");
  }
  if (require_gpu) {
    if (result.frame_bytes_from_cpu) {
      fail(name + ": frame bytes are still CPU-sourced");
    }
    if (result.stats.sync_count != 1) {
      fail(name + ": the GPU path must use exactly one synchronisation");
    }
    if (!result.stats.gpu_frame_verified) {
      fail(name + ": GPU frame/recon was not verified against the CPU");
    }
    if (result.stages.search != StageState::kGpu ||
        result.stages.tx != StageState::kGpu ||
        result.stages.recon != StageState::kGpu ||
        result.stages.chroma != StageState::kGpu) {
      fail(name + ": GPU stages are not all reported as gpu");
    }
    if (result.stats.tx_jobs !=
        static_cast<int>(result.plan.block_count) * 3) {
      fail(name + ": TX job count is not 3 per block");
    }
    if (result.coeff_y.size() != result.plan.coeff_count_y ||
        result.coeff_u.size() != result.plan.coeff_count_u ||
        result.coeff_v.size() != result.plan.coeff_count_v) {
      fail(name + ": GPU coefficient planes are missing or truncated");
    }
    for (std::size_t i = 0; i < result.decisions.size(); ++i) {
      const auto& decision = result.decisions[i];
      const auto& block = result.blocks[i];
      if (stage1 && decision.luma_mode != 0) {
        fail(name + ": forced-DC decision is not DC");
        break;
      }
      if (static_cast<std::uint32_t>(decision.eob_y) >
              static_cast<std::uint32_t>(block.tx_w) * block.tx_h ||
          static_cast<std::uint32_t>(decision.eob_u) >
              static_cast<std::uint32_t>(block.chroma_tx_w) *
                  block.chroma_tx_h ||
          static_cast<std::uint32_t>(decision.eob_v) >
              static_cast<std::uint32_t>(block.chroma_tx_w) *
                  block.chroma_tx_h) {
        fail(name + ": EOB exceeds the transform area");
        break;
      }
    }
  } else if (!result.frame_bytes_from_cpu) {
    fail(name + ": unexpected GPU bytes without a runtime");
  }

  const std::string cpu_md5 = md5_hex(cpu.frame);
  const std::string gpu_md5 = md5_hex(result.frame);
  std::printf("case %s (%ux%u, %s)\n", name.c_str(), frame.width,
              frame.height,
              result.frame_bytes_from_cpu
                  ? "cpu-fallback"
                  : (stage1 ? "gpu-forced-dc" : "gpu-full"));
  std::printf("  cpu frame   : %u bytes, md5 %s\n", cpu.frame_bytes,
              cpu_md5.c_str());
  std::printf("  still_gpu frame: %u bytes, md5 %s %s\n", result.frame_bytes,
              gpu_md5.c_str(), cpu_md5 == gpu_md5 ? "IDENTICAL" : "MISMATCH");
  std::printf("  tile bytes  : cpu %u, still_gpu %u\n", cpu.tile_bytes,
              result.tile_bytes);
  std::printf("  recon md5   : Y %s U %s V %s\n",
              md5_hex(result.recon_y).c_str(),
              md5_hex(result.recon_u).c_str(),
              md5_hex(result.recon_v).c_str());
  std::printf("  plan        : %u blocks (%s), %u leaves (cpu), "
              "opts qp=%u min=%d depth=%d force-single=%d\n",
              result.plan.block_count,
              result.plan.is_placeholder != 0 ? "placeholder raster"
                                              : "stage tree",
              cpu.leaves, options.base_q_idx, options.min_block_size,
              options.max_partition_depth, options.force_single_block ? 1 : 0);
  std::printf("  stages      : %s\n",
              stage_summary(result.stages).c_str());
  std::printf("  runtime     : %s, arena %.1f MiB, transfers %llu/%llu B, "
              "sync %d, tx %d, verified %d, %.2f ms\n",
              result.stats.gpu_runtime_ready
                  ? (result.stats.device_name[0] != '\0'
                         ? result.stats.device_name
                         : "CUDA")
                  : "CPU fallback (no device)",
              static_cast<double>(result.stats.arena_capacity_bytes) /
                  (1024.0 * 1024.0),
              static_cast<unsigned long long>(result.stats.host_upload_bytes),
              static_cast<unsigned long long>(result.stats.host_download_bytes),
              result.stats.sync_count, result.stats.tx_jobs,
              result.stats.gpu_frame_verified ? 1 : 0, result.stats.total_ms);

  // Second call must reuse the persistent runtime and produce the same bytes.
  GpuResult second;
  std::string second_error;
  if (!av2enc::cuda::still_gpu::encode_still_gpu(picture, options, second,
                                           second_error)) {
    fail(name + ": second encode_still_gpu failed: " + second_error);
  } else {
    if (!same_bytes(second.frame, result.frame)) {
      fail(name + ": second call produced different frame bytes");
    }
    if (second.stats.arena_capacity_bytes !=
            result.stats.arena_capacity_bytes ||
        second.stats.arena_used_bytes != result.stats.arena_used_bytes) {
      fail(name + ": second call changed the persistent arena");
    }
    std::printf("  repeat      : identical bytes, arena reused\n");
  }

  if (verbose) {
    std::printf("  note        : plan.blocks=%zu decisions=%zu coeff=%zu/%zu/%zu\n",
                result.plan.blocks.size(), result.decisions.size(),
                result.coeff_y.size(), result.coeff_u.size(),
                result.coeff_v.size());
  }
  std::printf("\n");
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> inputs;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> synthetic_sizes;
  bool verbose = false;
  bool require_gpu = true;
  bool stage1_cases = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      inputs.emplace_back(argv[++i]);
    } else if (arg == "--synthetic" && i + 1 < argc) {
      unsigned width = 0;
      unsigned height = 0;
      if (std::sscanf(argv[++i], "%ux%u", &width, &height) != 2 ||
          width == 0 || height == 0) {
        std::fprintf(stderr, "bad --synthetic size (want WxH)\n");
        return 2;
      }
      synthetic_sizes.emplace_back(width, height);
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--allow-no-gpu") {
      require_gpu = false;
    } else if (arg == "--no-stage1") {
      stage1_cases = false;
    } else if (arg == "--vram-cap" && i + 1 < argc) {
      const std::string value = argv[++i];
      unsigned long parsed = 0;
      try {
        parsed = std::stoul(value);
      } catch (...) {
        parsed = 0;
      }
      if (parsed == 0) {
        std::fprintf(stderr, "bad --vram-cap value\n");
        return 2;
      }
      g_vram_cap_mb = static_cast<std::uint32_t>(parsed);
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: %s [--input FILE.y4m]... [--synthetic WxH]... "
          "[--verbose] [--allow-no-gpu] [--no-stage1] "
          "[--vram-cap MiB]\n",
          argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  // Self-check the test-local MD5 before trusting the report.
  const std::uint8_t abc[3] = {'a', 'b', 'c'};
  if (md5_hex(std::span<const std::uint8_t>(abc, 3)) !=
      "900150983cd24fb0d6963f7d28e17f72") {
    std::fprintf(stderr, "FAIL: test-local MD5 is broken\n");
    return 2;
  }

  std::printf("m1-gpu data-flow test\n");
  std::printf("linked stage hooks: %d\n\n",
              av2enc::cuda::still_gpu::linked_stage_count());

  bool ran = false;
  for (const std::string& input : inputs) {
    Frame frame;
    std::string error;
    if (!load_first_frame(input, frame, error)) {
      std::fprintf(stderr, "SKIP %s: %s\n", input.c_str(), error.c_str());
      continue;
    }
    const std::string name = std::filesystem::path(input).stem().string();
    ran = true;
    if (!run_case(name, frame, verbose, require_gpu, false)) {
      return 1;
    }
    if (stage1_cases &&
        !run_case(name, frame, verbose, require_gpu, true)) {
      return 1;
    }
  }
  if (!ran) {
    // Always exercise at least one frame, even without test media.
    const Frame frame = synthetic_frame(64, 64);
    if (!run_case("synthetic_64x64", frame, verbose, require_gpu, false)) {
      return 1;
    }
    if (stage1_cases &&
        !run_case("synthetic_64x64", frame, verbose, require_gpu, true)) {
      return 1;
    }
  }

  // Edge geometry: not a multiple of the 64x64 superblock in either direction,
  // so the plan must clip the last superblock row/column.
  {
    const Frame frame = synthetic_frame(136, 88);
    if (!run_case("synthetic_136x88", frame, verbose, require_gpu, false)) {
      return 1;
    }
    if (stage1_cases &&
        !run_case("synthetic_136x88", frame, verbose, require_gpu, true)) {
      return 1;
    }
  }

  // Explicit capacity probes (e.g. --synthetic 1920x1080): a real plan resizes
  // the block vector to the leaf count, so this checks the worst case the
  // superblock-count allocation would have missed.
  for (const auto& size : synthetic_sizes) {
    const Frame frame = synthetic_frame(size.first, size.second);
    const std::string name = "synthetic_" + std::to_string(size.first) + "x" +
                             std::to_string(size.second);
    if (!run_case(name, frame, verbose, require_gpu, false)) {
      return 1;
    }
    if (stage1_cases &&
        !run_case(name, frame, verbose, require_gpu, true)) {
      return 1;
    }
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("PASS: plan -> stages -> host buffers -> frame bytes\n");
  return 0;
}
