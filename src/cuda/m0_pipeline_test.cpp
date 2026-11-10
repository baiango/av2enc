// CPU-vs-GPU comparison harness for the CUDA M0 still-picture pipeline.
//
// For each picture it runs:
//   1. the CPU reference encoder (src/encoder/m0_still.cpp),
//   2. the CUDA pipeline (src/cuda/m0_pipeline.*),
// then compares the bitstream byte-for-byte, reports MD5, reconstructed-plane
// PSNR (GPU vs CPU and GPU vs source) and timing, and writes both IVF files so
// an external dav2d gate can validate the GPU output.
//
// Exit code 0 only when the frame payloads and reconstructed planes are
// bit-identical.  This harness never claims decoder success itself; run
// tools/validate_dav2d.sh (build_pipeline_test.sh does) for the MD5 gate.

#include "cuda/m0_pipeline.h"

#include "encoder/m0_still.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Small self-contained MD5 (RFC 1321) so the harness does not depend on a
// system crypto library.  Used for the encoded frame payload and the IVF file.
// ---------------------------------------------------------------------------

class Md5 {
 public:
  void update(const void* data, std::size_t len) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    bits_ += static_cast<std::uint64_t>(len) * 8U;
    while (len > 0) {
      const std::size_t take = std::min<std::size_t>(64U - buf_len_, len);
      std::memcpy(buf_ + buf_len_, bytes, take);
      buf_len_ += take;
      bytes += take;
      len -= take;
      if (buf_len_ == 64) {
        transform(buf_);
        buf_len_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 16> finish() {
    const std::uint64_t bit_count = bits_;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t zero = 0;
    while (buf_len_ != 56) {
      update(&zero, 1);
    }
    std::uint8_t length[8];
    for (int i = 0; i < 8; ++i) {
      length[i] = static_cast<std::uint8_t>((bit_count >> (8 * i)) & 0xFFU);
    }
    update(length, 8);
    std::array<std::uint8_t, 16> out{};
    const std::uint32_t words[4] = {a_, b_, c_, d_};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        out[static_cast<std::size_t>(i * 4 + j)] =
            static_cast<std::uint8_t>((words[i] >> (8 * j)) & 0xFFU);
      }
    }
    return out;
  }

 private:
  static std::uint32_t rotl(std::uint32_t x, std::uint32_t c) {
    return (x << c) | (x >> (32U - c));
  }

  void transform(const std::uint8_t block[64]) {
    static constexpr std::uint32_t kK[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf,
        0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af,
        0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e,
        0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6,
        0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
        0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039,
        0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97,
        0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d,
        0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static constexpr std::uint32_t kS[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

    std::uint32_t m[16];
    for (int i = 0; i < 16; ++i) {
      m[i] = static_cast<std::uint32_t>(block[i * 4]) |
             (static_cast<std::uint32_t>(block[i * 4 + 1]) << 8U) |
             (static_cast<std::uint32_t>(block[i * 4 + 2]) << 16U) |
             (static_cast<std::uint32_t>(block[i * 4 + 3]) << 24U);
    }
    std::uint32_t a = a_;
    std::uint32_t b = b_;
    std::uint32_t c = c_;
    std::uint32_t d = d_;
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
    a_ += a;
    b_ += b;
    c_ += c;
    d_ += d;
  }

  std::uint32_t a_ = 0x67452301U;
  std::uint32_t b_ = 0xefcdab89U;
  std::uint32_t c_ = 0x98badcfeU;
  std::uint32_t d_ = 0x10325476U;
  std::uint64_t bits_ = 0;
  std::uint8_t buf_[64]{};
  std::size_t buf_len_ = 0;
};

std::string md5_hex(std::span<const std::uint8_t> data) {
  Md5 md5;
  md5.update(data.data(), data.size());
  const auto digest = md5.finish();
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (std::uint8_t byte : digest) {
    out.push_back(kHex[byte >> 4U]);
    out.push_back(kHex[byte & 0x0FU]);
  }
  return out;
}

std::string md5_file(const std::filesystem::path& path) {
  std::FILE* f = std::fopen(path.string().c_str(), "rb");
  if (f == nullptr) {
    return "<open failed>";
  }
  Md5 md5;
  std::uint8_t buffer[65536];
  std::size_t got = 0;
  while ((got = std::fread(buffer, 1, sizeof(buffer), f)) > 0) {
    md5.update(buffer, got);
  }
  std::fclose(f);
  const auto digest = md5.finish();
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (std::uint8_t byte : digest) {
    out.push_back(kHex[byte >> 4U]);
    out.push_back(kHex[byte & 0x0FU]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct Psnr {
  double y = 0.0;
  double u = 0.0;
  double v = 0.0;
  double average = 0.0;
};

Psnr psnr_planes(std::span<const std::uint8_t> ay,
                 std::span<const std::uint8_t> au,
                 std::span<const std::uint8_t> av,
                 std::span<const std::uint8_t> by,
                 std::span<const std::uint8_t> bu,
                 std::span<const std::uint8_t> bv) {
  const auto plane = [](std::span<const std::uint8_t> a,
                        std::span<const std::uint8_t> b) {
    if (a.size() != b.size() || a.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::uint64_t sse = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
      sse += static_cast<std::uint64_t>(d * d);
    }
    if (sse == 0) {
      return std::numeric_limits<double>::infinity();
    }
    const double mse = static_cast<double>(sse) /
                       static_cast<double>(a.size());
    return 10.0 * std::log10((255.0 * 255.0) / mse);
  };
  Psnr out;
  out.y = plane(ay, by);
  out.u = plane(au, bu);
  out.v = plane(av, bv);
  std::uint64_t sse = 0;
  std::size_t count = 0;
  const auto add = [&](std::span<const std::uint8_t> a,
                       std::span<const std::uint8_t> b) {
    if (a.size() != b.size()) {
      return;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
      const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
      sse += static_cast<std::uint64_t>(d * d);
    }
    count += a.size();
  };
  add(ay, by);
  add(au, bu);
  add(av, bv);
  out.average = (sse == 0 || count == 0)
                    ? std::numeric_limits<double>::infinity()
                    : 10.0 * std::log10((255.0 * 255.0) /
                                        (static_cast<double>(sse) /
                                         static_cast<double>(count)));
  return out;
}

void print_psnr(const char* label, const Psnr& p) {
  std::printf("  %-18s Y %7.3f dB  U %7.3f dB  V %7.3f dB  avg %7.3f dB\n",
              label, p.y, p.u, p.v, p.average);
}

std::string format_double(double value) {
  if (std::isinf(value)) {
    return "inf";
  }
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.4f", value);
  return buffer;
}

// ---------------------------------------------------------------------------
// Input generation / loading
// ---------------------------------------------------------------------------

struct Planes {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_y4m(const std::filesystem::path& path, Planes& planes,
              std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) {
    return false;
  }
  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
  if (status != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) {
      error = "Y4M contains no frame";
    }
    return false;
  }
  planes.width = frame.width();
  planes.height = frame.height();
  planes.y.assign(frame.y().begin(), frame.y().end());
  planes.u.assign(frame.u().begin(), frame.u().end());
  planes.v.assign(frame.v().begin(), frame.v().end());
  return true;
}

void make_pattern(Planes& planes, std::uint32_t width, std::uint32_t height) {
  planes.width = width;
  planes.height = height;
  const std::uint32_t cw = (width + 1U) / 2U;
  const std::uint32_t ch = (height + 1U) / 2U;
  planes.y.resize(static_cast<std::size_t>(width) * height);
  planes.u.resize(static_cast<std::size_t>(cw) * ch);
  planes.v.resize(static_cast<std::size_t>(cw) * ch);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t value =
          (x * 3U + y * 5U + ((x * x + y * y) >> 7U)) & 0xFFU;
      planes.y[static_cast<std::size_t>(y) * width + x] =
          static_cast<std::uint8_t>(value);
    }
  }
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t x = 0; x < cw; ++x) {
      const std::uint32_t value =
          (128U + (x * 7U + y * 11U) / 3U + ((x * y) >> 5U)) & 0xFFU;
      planes.u[static_cast<std::size_t>(y) * cw + x] =
          static_cast<std::uint8_t>(value);
      planes.v[static_cast<std::size_t>(y) * cw + x] =
          static_cast<std::uint8_t>(255U - value);
    }
  }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Options {
  std::string y4m;
  std::string out_dir = ".";
  std::string name;
  std::uint32_t width = 640;
  std::uint32_t height = 360;
  av2enc::m0::M0PartitionMode partition =
      av2enc::m0::M0PartitionMode::k32x32;
  bool verbose = false;
};

bool parse_partition(const std::string& text,
                     av2enc::m0::M0PartitionMode* mode) {
  using Mode = av2enc::m0::M0PartitionMode;
  if (text == "32x32" || text == "k32x32") {
    *mode = Mode::k32x32;
  } else if (text == "64x64" || text == "64x64-single") {
    *mode = Mode::k64x64Single;
  } else if (text == "64x64-vert") {
    *mode = Mode::k64x64VertNone;
  } else if (text == "64x64-horz") {
    *mode = Mode::k64x64HorzNone;
  } else if (text == "32x32-horz-first") {
    *mode = Mode::k32x32HorzFirst;
  } else {
    return false;
  }
  return true;
}

bool parse_args(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        error = std::string("missing value for ") + what;
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--y4m") {
      const char* value = next("--y4m");
      if (value == nullptr) {
        return false;
      }
      options.y4m = value;
    } else if (arg == "--out-dir") {
      const char* value = next("--out-dir");
      if (value == nullptr) {
        return false;
      }
      options.out_dir = value;
    } else if (arg == "--name") {
      const char* value = next("--name");
      if (value == nullptr) {
        return false;
      }
      options.name = value;
    } else if (arg == "--width") {
      const char* value = next("--width");
      if (value == nullptr) {
        return false;
      }
      options.width = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    } else if (arg == "--height") {
      const char* value = next("--height");
      if (value == nullptr) {
        return false;
      }
      options.height =
          static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
    } else if (arg == "--partition") {
      const char* value = next("--partition");
      if (value == nullptr || !parse_partition(value, &options.partition)) {
        error = std::string("unknown partition mode: ") +
                (value == nullptr ? "" : value);
        return false;
      }
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "usage: m0_pipeline_test [--y4m PATH] [--width W --height H]\n"
          "                        [--out-dir DIR] [--name NAME]\n"
          "                        [--partition 32x32|64x64-single|"
          "64x64-vert|64x64-horz|32x32-horz-first]\n"
          "                        [--verbose]\n");
      std::exit(0);
    } else {
      error = "unknown argument: " + arg;
      return false;
    }
  }
  return true;
}

bool write_ivf(const std::filesystem::path& path,
               std::span<const std::uint8_t> frame, std::uint32_t width,
               std::uint32_t height, std::string& error) {
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  av2enc::IvfWriter writer;
  if (!writer.open(path, header, error)) {
    return false;
  }
  if (!writer.write_frame(frame, 0, error)) {
    return false;
  }
  return writer.close(error);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string error;
  if (!parse_args(argc, argv, options, error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 2;
  }

  Planes planes;
  if (!options.y4m.empty()) {
    if (!load_y4m(options.y4m, planes, error)) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      return 2;
    }
    if (options.name.empty()) {
      options.name = std::filesystem::path(options.y4m).stem().string();
    }
  } else {
    if ((options.width & 1U) != 0U || (options.height & 1U) != 0U) {
      std::fprintf(stderr, "error: generated input needs even dimensions\n");
      return 2;
    }
    make_pattern(planes, options.width, options.height);
    if (options.name.empty()) {
      options.name = std::to_string(options.width) + "x" +
                     std::to_string(options.height) + "_pattern";
    }
  }

  const std::uint32_t width = planes.width;
  const std::uint32_t height = planes.height;

  av2enc::m0::StillPicture picture;
  picture.width = width;
  picture.height = height;
  picture.y = std::span<const std::uint8_t>(planes.y.data(), planes.y.size());
  picture.u = std::span<const std::uint8_t>(planes.u.data(), planes.u.size());
  picture.v = std::span<const std::uint8_t>(planes.v.data(), planes.v.size());

  av2enc::m0::StillPictureConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;

  std::printf("m0_pipeline_test: %s (%ux%u, %s)\n", options.name.c_str(),
              width, height,
              options.y4m.empty() ? "generated pattern" : options.y4m.c_str());

  // CPU warm-up (not measured) so the reported time is steady-state.
  {
    av2enc::m0::StillEncodeResult warm;
    std::string warm_error;
    if (!av2enc::m0::encode_still_picture(picture, config, warm, warm_error,
                                         options.partition)) {
      std::fprintf(stderr, "CPU warm-up failed: %s\n", warm_error.c_str());
      return 1;
    }
  }

  // CPU reference.
  av2enc::m0::StillEncodeResult cpu;
  const auto cpu_start = std::chrono::steady_clock::now();
  if (!av2enc::m0::encode_still_picture(picture, config, cpu, error,
                                       options.partition)) {
    std::fprintf(stderr, "CPU reference failed: %s\n", error.c_str());
    return 1;
  }
  const auto cpu_end = std::chrono::steady_clock::now();
  const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  // CUDA pipeline.  The first call pays Context/Arena/StagingBuffer setup and
  // module load; run it once as a warm-up and report the steady-state call.
  av2enc::cuda::PipelineOptions pipeline_options;
  pipeline_options.verbose = options.verbose;
  {
    av2enc::m0::StillEncodeResult warm;
    av2enc::cuda::PipelineTimings warm_timings;
    std::string warm_error;
    if (!av2enc::cuda::encode_still_picture_cuda(
            picture, config, warm, warm_error, options.partition,
            pipeline_options, &warm_timings)) {
      std::fprintf(stderr, "CUDA warm-up failed: %s\n", warm_error.c_str());
      return 1;
    }
    if (warm.frame != cpu.frame) {
      std::fprintf(stderr,
                   "CUDA warm-up frame differs from the CPU reference\n");
      return 1;
    }
  }

  av2enc::m0::StillEncodeResult gpu;
  av2enc::cuda::PipelineTimings timings;
  const auto gpu_start = std::chrono::steady_clock::now();
  if (!av2enc::cuda::encode_still_picture_cuda(
          picture, config, gpu, error, options.partition, pipeline_options,
          &timings)) {
    std::fprintf(stderr, "CUDA pipeline failed: %s\n", error.c_str());
    return 1;
  }
  const auto gpu_end = std::chrono::steady_clock::now();
  const double gpu_wall_ms =
      std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

  // Bitstream comparison.
  const std::string cpu_md5 = md5_hex(cpu.frame);
  const std::string gpu_md5 = md5_hex(gpu.frame);
  bool frame_equal = cpu.frame.size() == gpu.frame.size() &&
                     std::memcmp(cpu.frame.data(), gpu.frame.data(),
                                 cpu.frame.size()) == 0;
  std::size_t first_diff = 0;
  if (!frame_equal) {
    const std::size_t common = std::min(cpu.frame.size(), gpu.frame.size());
    while (first_diff < common &&
           cpu.frame[first_diff] == gpu.frame[first_diff]) {
      ++first_diff;
    }
  }

  // Reconstruction comparison.
  const bool recon_equal =
      cpu.recon_y == gpu.recon_y && cpu.recon_u == gpu.recon_u &&
      cpu.recon_v == gpu.recon_v;
  std::size_t recon_diffs = 0;
  int max_recon_diff = 0;
  const auto diff_plane = [&](const std::vector<std::uint8_t>& a,
                              const std::vector<std::uint8_t>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
      const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
      if (d != 0) {
        ++recon_diffs;
        max_recon_diff = std::max(max_recon_diff, std::abs(d));
      }
    }
    recon_diffs += a.size() > b.size() ? a.size() - b.size() : b.size() - a.size();
  };
  diff_plane(cpu.recon_y, gpu.recon_y);
  diff_plane(cpu.recon_u, gpu.recon_u);
  diff_plane(cpu.recon_v, gpu.recon_v);

  const Psnr gpu_vs_cpu = psnr_planes(
      gpu.recon_y, gpu.recon_u, gpu.recon_v, cpu.recon_y, cpu.recon_u,
      cpu.recon_v);
  const Psnr gpu_vs_source = psnr_planes(gpu.recon_y, gpu.recon_u, gpu.recon_v,
                                         planes.y, planes.u, planes.v);

  std::printf("  CPU frame: %u bytes (tile %u), MD5 %s\n", cpu.frame_bytes,
              cpu.tile_bytes, cpu_md5.c_str());
  std::printf("  GPU frame: %u bytes (tile %u), MD5 %s\n", gpu.frame_bytes,
              gpu.tile_bytes, gpu_md5.c_str());
  if (frame_equal) {
    std::printf("  bitstream: IDENTICAL\n");
  } else {
    std::printf("  bitstream: DIFFERENT (first byte %zu, cpu %zu bytes, gpu "
                "%zu bytes)\n",
                first_diff, cpu.frame.size(), gpu.frame.size());
  }
  if (recon_equal) {
    std::printf("  recon:     bit-exact\n");
  } else {
    std::printf("  recon:     %zu samples differ, max |diff| %d\n", recon_diffs,
                max_recon_diff);
  }
  print_psnr("GPU vs CPU", gpu_vs_cpu);
  print_psnr("GPU vs source", gpu_vs_source);
  std::printf(
      "  timing:    cpu %s ms | gpu upload %s ms, kernel %s ms, download %s "
      "ms, entropy %s ms, total %s ms (wall %s ms, %d launches)\n",
      format_double(cpu_ms).c_str(), format_double(timings.upload_ms).c_str(),
      format_double(timings.kernel_ms).c_str(),
      format_double(timings.download_ms).c_str(),
      format_double(timings.entropy_ms).c_str(),
      format_double(timings.total_ms).c_str(),
      format_double(gpu_wall_ms).c_str(), timings.kernel_launches);
  if (cpu_ms > 0.0 && timings.total_ms > 0.0) {
    std::printf("  speedup:   %sx (CPU reference / GPU pipeline)\n",
                format_double(cpu_ms / timings.total_ms).c_str());
  }

  // IVF output for the external dav2d gate.
  std::error_code ec;
  const std::filesystem::path out_dir(options.out_dir);
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::fprintf(stderr, "error: cannot create %s: %s\n",
                 out_dir.string().c_str(), ec.message().c_str());
    return 1;
  }
  const std::filesystem::path gpu_ivf = out_dir / ("gpu_" + options.name + ".ivf");
  const std::filesystem::path cpu_ivf = out_dir / ("cpu_" + options.name + ".ivf");
  if (!write_ivf(gpu_ivf, gpu.frame, width, height, error) ||
      !write_ivf(cpu_ivf, cpu.frame, width, height, error)) {
    std::fprintf(stderr, "error: IVF write failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("  ivf:       %s (MD5 %s), %s (MD5 %s)\n",
              gpu_ivf.string().c_str(), md5_file(gpu_ivf).c_str(),
              cpu_ivf.string().c_str(), md5_file(cpu_ivf).c_str());

  if (!frame_equal || !recon_equal) {
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
