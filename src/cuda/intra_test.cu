// M0 intra component test: DC prediction, residual, reconstruction.
//
// CPU reference is a line-for-line mirror of encode_residual_plane() in
// src/encoder/basic_encoder.cpp (read-only): same clipping, same Round2 /
// approximate-divide arithmetic, same edge clamping.
//
// Build/run with build_intra_test.sh on the sm_86 box.

#include "intra.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cuda = av2enc::cuda;

// ---------------------------------------------------------------------------
// Error handling / test bookkeeping
// ---------------------------------------------------------------------------

namespace {

[[noreturn]] void fatal(const char* expr, cudaError_t err, const char* file,
                        int line) {
  std::fprintf(stderr, "CUDA error: %s (%d) from %s at %s:%d\n",
               cudaGetErrorString(err), static_cast<int>(err), expr, file,
               line);
  std::exit(1);
}

#define CUDA_CHECK(expr)                                                  \
  do {                                                                    \
    const cudaError_t err_ = (expr);                                      \
    if (err_ != cudaSuccess) {                                            \
      fatal(#expr, err_, __FILE__, __LINE__);                             \
    }                                                                     \
  } while (false)

struct Stage {
  const char* name;
  long long samples = 0;
  long long mismatches = 0;
  int max_abs_diff = 0;
};

Stage g_pred{"prediction"};
Stage g_pred_value{"pred_value"};
Stage g_residual{"residual"};
Stage g_recon{"recon_block"};
Stage g_recon_plane{"recon_plane"};
Stage g_stress{"clip_stress"};

std::string g_context;
int g_printed = 0;
constexpr int kMaxPrinted = 16;

void note_mismatch(const Stage& stage, const char* what, long long index,
                   long long expected, long long actual) {
  if (g_printed >= kMaxPrinted) {
    return;
  }
  std::printf("  MISMATCH [%s] %s %s: index %lld expected %lld got %lld\n",
              stage.name, g_context.c_str(), what, index, expected, actual);
  ++g_printed;
}

void compare_u8(Stage& stage, const std::uint8_t* expected,
                const std::uint8_t* actual, int count, const char* what) {
  long long mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < count; ++i) {
    const int e = static_cast<int>(expected[i]);
    const int a = static_cast<int>(actual[i]);
    if (e != a) {
      ++mismatches;
      max_diff = std::max(max_diff, std::abs(e - a));
      note_mismatch(stage, what, i, e, a);
    }
  }
  stage.samples += count;
  stage.mismatches += mismatches;
  stage.max_abs_diff = std::max(stage.max_abs_diff, max_diff);
}

void compare_i16(Stage& stage, const std::int16_t* expected,
                 const std::int16_t* actual, int count, const char* what) {
  long long mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < count; ++i) {
    const int e = static_cast<int>(expected[i]);
    const int a = static_cast<int>(actual[i]);
    if (e != a) {
      ++mismatches;
      max_diff = std::max(max_diff, std::abs(e - a));
      note_mismatch(stage, what, i, e, a);
    }
  }
  stage.samples += count;
  stage.mismatches += mismatches;
  stage.max_abs_diff = std::max(stage.max_abs_diff, max_diff);
}

void compare_int(Stage& stage, const int* expected, const int* actual,
                 int count, const char* what) {
  long long mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < count; ++i) {
    if (expected[i] != actual[i]) {
      ++mismatches;
      max_diff = std::max(max_diff, std::abs(expected[i] - actual[i]));
      note_mismatch(stage, what, i, expected[i], actual[i]);
    }
  }
  stage.samples += count;
  stage.mismatches += mismatches;
  stage.max_abs_diff = std::max(stage.max_abs_diff, max_diff);
}

// ---------------------------------------------------------------------------
// CPU reference (mirror of src/encoder/basic_encoder.cpp)
// ---------------------------------------------------------------------------

int cpu_clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int cpu_clip1(int value) { return cpu_clip3(0, 255, value); }

int cpu_round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

int cpu_floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

int cpu_approx_divide(int numerator, int denominator) {
  const int shift = 9 + cpu_floor_log2(denominator);
  return cpu_round2(numerator * 512, shift);
}

struct Plane {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint8_t> data;
};

constexpr int kPatternCount = 7;

const char* pattern_name(int pattern) {
  static const char* kNames[kPatternCount] = {
      "gradient", "black", "white", "checker", "noise", "stripes", "ramp"};
  return kNames[pattern];
}

std::uint8_t pattern_value(int pattern, int x, int y) {
  switch (pattern) {
    case 0:
      return static_cast<std::uint8_t>((x * 3 + y * 5 + 17) & 255);
    case 1:
      return 0;
    case 2:
      return 255;
    case 3:
      return static_cast<std::uint8_t>((((x >> 4) ^ (y >> 4)) & 1) ? 255 : 0);
    case 4: {
      std::uint32_t v = static_cast<std::uint32_t>(x) * 374761393u +
                        static_cast<std::uint32_t>(y) * 668265263u;
      v = (v ^ (v >> 13)) * 1274126177u;
      return static_cast<std::uint8_t>(v >> 24);
    }
    case 5:
      return static_cast<std::uint8_t>(((x / 7 + y / 5) & 1) ? 30 : 220);
    default:
      return static_cast<std::uint8_t>(((x * x + 3 * y) * 31 + 7) & 255);
  }
}

Plane make_plane(int width, int height, int pattern) {
  Plane plane;
  plane.width = width;
  plane.height = height;
  plane.stride = width;
  plane.data.resize(static_cast<std::size_t>(width) *
                    static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      plane.data[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)] = pattern_value(pattern, x, y);
    }
  }
  return plane;
}

int cpu_dc_predict(const Plane& recon, int x0, int y0, int width, int height,
                   bool have_left, bool have_above) {
  const int max_x = recon.width - 1;
  const int max_y = recon.height - 1;
  const std::size_t stride = static_cast<std::size_t>(recon.stride);
  int sum_left = 0;
  int sum_above = 0;
  if (have_left) {
    for (int i = 0; i < height; ++i) {
      const int yy = cpu_clip3(0, max_y, y0 + i);
      sum_left += static_cast<int>(
          recon.data[static_cast<std::size_t>(yy) * stride +
                     static_cast<std::size_t>(x0 - 1)]);
    }
  }
  if (have_above) {
    for (int j = 0; j < width; ++j) {
      const int xx = cpu_clip3(0, max_x, x0 + j);
      sum_above += static_cast<int>(
          recon.data[static_cast<std::size_t>(y0 - 1) * stride +
                     static_cast<std::size_t>(xx)]);
    }
  }
  if (have_left && have_above) {
    return cpu_clip1(cpu_approx_divide(sum_left + sum_above, width + height));
  }
  if (have_left) {
    return cpu_round2(sum_left, cpu_floor_log2(height));
  }
  if (have_above) {
    return cpu_round2(sum_above, cpu_floor_log2(width));
  }
  return 128;
}

int cpu_residual_at(const Plane& source, int x0, int y0, int width, int index,
                    int pred) {
  const int px = index % width;
  const int py = index / width;
  const int sx = cpu_clip3(0, source.width - 1, x0 + px);
  const int sy = cpu_clip3(0, source.height - 1, y0 + py);
  const int sample = static_cast<int>(
      source.data[static_cast<std::size_t>(sy) *
                      static_cast<std::size_t>(source.stride) +
                  static_cast<std::size_t>(sx)]);
  return sample - pred;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void allocate(std::size_t count) {
    release();
    if (count == 0) {
      return;
    }
    count_ = count;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_),
                          count * sizeof(T)));
  }

  void release() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

  void upload(const T* source) {
    CUDA_CHECK(cudaMemcpy(ptr_, source, count_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  void download(T* destination) const {
    CUDA_CHECK(cudaMemcpy(destination, ptr_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

struct Rng {
  std::uint32_t state;

  explicit Rng(std::uint32_t seed) : state(seed != 0 ? seed : 1u) {}

  std::uint32_t next() {
    std::uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
  }

  int range(int bound) {
    return static_cast<int>(next() % static_cast<std::uint32_t>(bound));
  }
};

// ---------------------------------------------------------------------------
// Per-block case
// ---------------------------------------------------------------------------

struct BlockCase {
  int x0 = 0;
  int y0 = 0;
  int w = 0;
  int h = 0;
  int flags = -1;  // -1: derive from position; bit0 left, bit1 above
  const char* label = "";
};

std::vector<BlockCase> make_luma_cases() {
  std::vector<BlockCase> cases = {
      {0, 0, 32, 32, -1, "tl"},
      {32, 0, 32, 32, -1, "top"},
      {0, 32, 32, 32, -1, "left"},
      {32, 32, 32, 32, -1, "interior"},
      {608, 0, 32, 32, -1, "tr"},
      {0, 352, 32, 8, -1, "bl-32x8"},
      {608, 352, 32, 8, -1, "br-32x8"},
      {0, 336, 32, 16, -1, "l-32x16"},
      {32, 336, 32, 16, -1, "b-32x16"},
      {608, 336, 32, 16, -1, "r-32x16"},
      {0, 0, 32, 16, -1, "tl-32x16"},
      {608, 0, 32, 16, -1, "tr-32x16"},
      {0, 0, 32, 8, -1, "tl-32x8"},
      {608, 0, 32, 8, -1, "tr-32x8"},
      {0, 320, 64, 64, -1, "bottom-64-extends"},
      {576, 320, 64, 64, -1, "br-64-extends"},
      {0, 0, 64, 64, -1, "tl-64"},
      {64, 64, 64, 64, -1, "i-64"},
      {0, 352, 32, 32, -1, "bottom-32-extends"},
      {608, 352, 32, 32, -1, "br-32-extends"},
      {0, 0, 64, 32, -1, "tl-64x32"},
      {576, 0, 64, 32, -1, "tr-64x32"},
      {0, 288, 32, 64, -1, "bl-32x64"},
      {608, 288, 32, 64, -1, "br-32x64"},
  };
  for (int flags = 0; flags < 4; ++flags) {
    cases.push_back({32, 32, 32, 32, flags, "flag-matrix"});
    cases.push_back({64, 64, 64, 64, flags, "flag-matrix-64"});
  }
  return cases;
}

std::vector<BlockCase> make_chroma_cases() {
  std::vector<BlockCase> cases = {
      {0, 0, 16, 16, -1, "tl"},
      {16, 0, 16, 16, -1, "top"},
      {0, 16, 16, 16, -1, "left"},
      {16, 16, 16, 16, -1, "interior"},
      {304, 0, 16, 16, -1, "tr"},
      {0, 160, 16, 16, -1, "bl"},
      {304, 160, 16, 16, -1, "br"},
      {0, 176, 16, 4, -1, "bl-16x4"},
      {16, 176, 16, 4, -1, "b-16x4"},
      {304, 176, 16, 4, -1, "br-16x4"},
      {0, 168, 16, 8, -1, "l-16x8"},
      {16, 168, 16, 8, -1, "b-16x8"},
      {304, 168, 16, 8, -1, "r-16x8"},
      {0, 0, 16, 8, -1, "tl-16x8"},
      {304, 0, 16, 8, -1, "tr-16x8"},
      {0, 0, 16, 4, -1, "tl-16x4"},
      {304, 0, 16, 4, -1, "tr-16x4"},
      {0, 160, 32, 32, -1, "bottom-32-extends"},
      {288, 160, 32, 32, -1, "br-32-extends"},
      {0, 0, 32, 32, -1, "tl-32"},
      {32, 32, 32, 32, -1, "i-32"},
      {288, 0, 32, 32, -1, "tr-32"},
      {0, 128, 32, 32, -1, "bl-32"},
      {0, 0, 32, 16, -1, "tl-32x16"},
      {288, 144, 32, 16, -1, "br-32x16"},
      {0, 0, 16, 32, -1, "tl-16x32"},
      {0, 144, 16, 32, -1, "bl-16x32"},
      {304, 144, 16, 32, -1, "br-16x32"},
  };
  for (int flags = 0; flags < 4; ++flags) {
    cases.push_back({32, 32, 16, 16, flags, "flag-matrix"});
    cases.push_back({32, 32, 32, 32, flags, "flag-matrix-32"});
  }
  return cases;
}

// Holds device copies of one plane pair plus scratch block buffers.
class PlaneRunner {
 public:
  PlaneRunner(const Plane& recon, const Plane& source)
      : recon_(recon), source_(source) {
    d_recon_.allocate(recon.data.size());
    d_source_.allocate(source.data.size());
    d_recon_.upload(recon.data.data());
    d_source_.upload(source.data.data());

    constexpr int kMaxSamples = cuda::kIntraMaxBlockDim * cuda::kIntraMaxBlockDim;
    d_pred_.allocate(kMaxSamples);
    d_res_.allocate(kMaxSamples);
    d_res_null_.allocate(kMaxSamples);
    d_recon_block_.allocate(kMaxSamples);
    d_pred_value_.allocate(1);
    d_plane_.allocate(recon.data.size());
  }

  void run_case(const BlockCase& c) {
    const int n = c.w * c.h;
    const int have_left =
        c.flags < 0 ? (c.x0 > 0 ? 1 : 0) : (c.flags & 1);
    const int have_above =
        c.flags < 0 ? (c.y0 > 0 ? 1 : 0) : ((c.flags >> 1) & 1);

    cuda::DcBlock block{};
    block.recon_plane = d_recon_.get();
    block.recon_stride = recon_.stride;
    block.x0 = c.x0;
    block.y0 = c.y0;
    block.block_w = c.w;
    block.block_h = c.h;
    block.plane_max_x = recon_.width - 1;
    block.plane_max_y = recon_.height - 1;
    block.have_left = have_left;
    block.have_above = have_above;
    block.source_plane = d_source_.get();
    block.source_stride = source_.stride;
    block.source_max_x = source_.width - 1;
    block.source_max_y = source_.height - 1;

    char context[192];
    std::snprintf(context, sizeof(context), "plane=%dx%d case=%s(%d,%d,%dx%d)",
                  recon_.width, recon_.height, c.label, c.x0, c.y0, c.w, c.h);
    g_context = context;

    const int expected_pred = cpu_dc_predict(
        recon_, c.x0, c.y0, c.w, c.h, have_left != 0, have_above != 0);

    std::vector<std::uint8_t> expected_pred_block(
        static_cast<std::size_t>(n), static_cast<std::uint8_t>(expected_pred));
    std::vector<std::int16_t> expected_residual(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> expected_recon(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const int residual =
          cpu_residual_at(source_, c.x0, c.y0, c.w, i, expected_pred);
      expected_residual[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>(residual);
      expected_recon[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(cpu_clip1(expected_pred + residual));
    }

    // 1. Normative predictor value.
    CUDA_CHECK(cuda::dc_pred_value(block, d_pred_value_.get(), nullptr));
    int gpu_pred_value = 0;
    CUDA_CHECK(cudaMemcpy(&gpu_pred_value, d_pred_value_.get(), sizeof(int),
                          cudaMemcpyDeviceToHost));
    compare_int(g_pred_value, &expected_pred, &gpu_pred_value, 1, "dc-pred");

    std::vector<std::uint8_t> gpu_pred(static_cast<std::size_t>(n));
    std::vector<std::int16_t> gpu_residual(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> gpu_recon(static_cast<std::size_t>(n));

    // 2. Prediction plane.
    CUDA_CHECK(cuda::dc_predict(block, d_pred_.get(), nullptr));
    CUDA_CHECK(cudaMemcpy(gpu_pred.data(), d_pred_.get(),
                          static_cast<std::size_t>(n) * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
    compare_u8(g_pred, expected_pred_block.data(), gpu_pred.data(), n,
               "pred-block");

    // 3. Residual from the explicit prediction buffer.
    CUDA_CHECK(cuda::dc_residual(block, d_pred_.get(), d_res_.get(), nullptr));
    CUDA_CHECK(cudaMemcpy(gpu_residual.data(), d_res_.get(),
                          static_cast<std::size_t>(n) * sizeof(std::int16_t),
                          cudaMemcpyDeviceToHost));
    compare_i16(g_residual, expected_residual.data(), gpu_residual.data(), n,
                "residual-with-pred");

    // 4. Residual recomputing the predictor internally.
    CUDA_CHECK(cuda::dc_residual(block, nullptr, d_res_null_.get(), nullptr));
    CUDA_CHECK(cudaMemcpy(gpu_residual.data(), d_res_null_.get(),
                          static_cast<std::size_t>(n) * sizeof(std::int16_t),
                          cudaMemcpyDeviceToHost));
    compare_i16(g_residual, expected_residual.data(), gpu_residual.data(), n,
                "residual-null-pred");

    // 5. Fused prediction + residual.
    CUDA_CHECK(
        cuda::dc_predict_residual(block, d_pred_.get(), d_res_.get(), nullptr));
    CUDA_CHECK(cudaMemcpy(gpu_pred.data(), d_pred_.get(),
                          static_cast<std::size_t>(n) * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_residual.data(), d_res_.get(),
                          static_cast<std::size_t>(n) * sizeof(std::int16_t),
                          cudaMemcpyDeviceToHost));
    compare_u8(g_pred, expected_pred_block.data(), gpu_pred.data(), n,
               "fused-pred");
    compare_i16(g_residual, expected_residual.data(), gpu_residual.data(), n,
                "fused-residual");

    // 6. add_residual_clip into a contiguous block buffer.
    CUDA_CHECK(cuda::add_residual_clip(d_pred_.get(), d_res_.get(),
                                       d_recon_block_.get(), c.w, c.h,
                                       nullptr));
    CUDA_CHECK(cudaMemcpy(gpu_recon.data(), d_recon_block_.get(),
                          static_cast<std::size_t>(n) * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
    compare_u8(g_recon, expected_recon.data(), gpu_recon.data(), n,
               "recon-block");

    // 7. add_residual_clip_to_plane, with out-of-bounds writes dropped.
    CUDA_CHECK(cudaMemcpy(d_plane_.get(), recon_.data.data(),
                          recon_.data.size() * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cuda::add_residual_clip_to_plane(
        d_pred_.get(), d_res_.get(), d_plane_.get(), recon_.stride, c.x0, c.y0,
        c.w, c.h, recon_.width - 1, recon_.height - 1, nullptr));
    std::vector<std::uint8_t> expected_plane = recon_.data;
    for (int i = 0; i < n; ++i) {
      const int x = c.x0 + i % c.w;
      const int y = c.y0 + i / c.w;
      if (x > recon_.width - 1 || y > recon_.height - 1) {
        continue;
      }
      expected_plane[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(recon_.stride) +
                     static_cast<std::size_t>(x)] =
          expected_recon[static_cast<std::size_t>(i)];
    }
    std::vector<std::uint8_t> gpu_plane(recon_.data.size());
    CUDA_CHECK(cudaMemcpy(gpu_plane.data(), d_plane_.get(),
                          recon_.data.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
    compare_u8(g_recon_plane, expected_plane.data(), gpu_plane.data(),
               static_cast<int>(recon_.data.size()), "to-plane");
  }

  void run_random_cases(int iterations, std::uint32_t seed) {
    static const int kDims[][2] = {
        {32, 32}, {32, 16}, {32, 8},  {64, 64}, {64, 32},
        {32, 64}, {16, 16}, {16, 8},  {16, 4},  {8, 16},
        {8, 8},   {4, 16},  {16, 32}, {32, 4},  {8, 32},
    };
    constexpr int kDimCount =
        static_cast<int>(sizeof(kDims) / sizeof(kDims[0]));
    Rng rng(seed);
    for (int iteration = 0; iteration < iterations; ++iteration) {
      BlockCase c;
      const int* dims = kDims[rng.range(kDimCount)];
      c.w = dims[0];
      c.h = dims[1];
      c.x0 = rng.range(recon_.width);
      c.y0 = rng.range(recon_.height);
      const int left = c.x0 > 0 ? rng.range(2) : 0;
      const int above = c.y0 > 0 ? rng.range(2) : 0;
      c.flags = left | (above << 1);
      c.label = "random";
      run_case(c);
    }
  }

 private:
  const Plane& recon_;
  const Plane& source_;
  DeviceBuffer<std::uint8_t> d_recon_;
  DeviceBuffer<std::uint8_t> d_source_;
  DeviceBuffer<std::uint8_t> d_pred_;
  DeviceBuffer<std::int16_t> d_res_;
  DeviceBuffer<std::int16_t> d_res_null_;
  DeviceBuffer<std::uint8_t> d_recon_block_;
  DeviceBuffer<std::uint8_t> d_plane_;
  DeviceBuffer<int> d_pred_value_;
};

// ---------------------------------------------------------------------------
// Full-plane dependency test: raster blocks, evolving reconstruction
// ---------------------------------------------------------------------------

void run_composition(const char* label, int width, int height, int block_w,
                     int block_h, int pattern) {
  const Plane source = make_plane(width, height, (pattern + 3) % kPatternCount);
  Plane cpu_plane;
  cpu_plane.width = width;
  cpu_plane.height = height;
  cpu_plane.stride = width;
  cpu_plane.data.assign(static_cast<std::size_t>(width) *
                            static_cast<std::size_t>(height),
                        128);

  DeviceBuffer<std::uint8_t> d_plane(cpu_plane.data.size());
  DeviceBuffer<std::uint8_t> d_source(source.data.size());
  DeviceBuffer<std::uint8_t> d_pred(
      static_cast<std::size_t>(block_w) * static_cast<std::size_t>(block_h));
  DeviceBuffer<std::int16_t> d_res(
      static_cast<std::size_t>(block_w) * static_cast<std::size_t>(block_h));
  d_plane.upload(cpu_plane.data.data());
  d_source.upload(source.data.data());

  std::vector<std::uint8_t> gpu_pred(
      static_cast<std::size_t>(block_w) * static_cast<std::size_t>(block_h));
  std::vector<std::int16_t> gpu_res(
      static_cast<std::size_t>(block_w) * static_cast<std::size_t>(block_h));

  const int max_x = width - 1;
  const int max_y = height - 1;

  for (int y0 = 0; y0 < height; y0 += block_h) {
    const int h = std::min(block_h, height - y0);
    for (int x0 = 0; x0 < width; x0 += block_w) {
      const int w = std::min(block_w, width - x0);
      const int n = w * h;
      const int have_left = x0 > 0 ? 1 : 0;
      const int have_above = y0 > 0 ? 1 : 0;

      cuda::DcBlock block{};
      block.recon_plane = d_plane.get();
      block.recon_stride = width;
      block.x0 = x0;
      block.y0 = y0;
      block.block_w = w;
      block.block_h = h;
      block.plane_max_x = max_x;
      block.plane_max_y = max_y;
      block.have_left = have_left;
      block.have_above = have_above;
      block.source_plane = d_source.get();
      block.source_stride = width;
      block.source_max_x = max_x;
      block.source_max_y = max_y;

      char context[192];
      std::snprintf(context, sizeof(context),
                    "composition=%s pattern=%s block(%d,%d,%dx%d)", label,
                    pattern_name(pattern), x0, y0, w, h);
      g_context = context;

      const int expected_pred = cpu_dc_predict(
          cpu_plane, x0, y0, w, h, have_left != 0, have_above != 0);
      std::vector<std::uint8_t> expected_pred_block(
          static_cast<std::size_t>(n),
          static_cast<std::uint8_t>(expected_pred));
      std::vector<std::int16_t> expected_residual(
          static_cast<std::size_t>(n));

      CUDA_CHECK(
          cuda::dc_predict_residual(block, d_pred.get(), d_res.get(), nullptr));
      CUDA_CHECK(cudaMemcpy(gpu_pred.data(), d_pred.get(),
                            static_cast<std::size_t>(n) * sizeof(std::uint8_t),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gpu_res.data(), d_res.get(),
                            static_cast<std::size_t>(n) * sizeof(std::int16_t),
                            cudaMemcpyDeviceToHost));
      compare_u8(g_pred, expected_pred_block.data(), gpu_pred.data(), n,
                 "composition-pred");

      for (int i = 0; i < n; ++i) {
        expected_residual[static_cast<std::size_t>(i)] =
            static_cast<std::int16_t>(cpu_residual_at(source, x0, y0, w, i,
                                                      expected_pred));
      }
      compare_i16(g_residual, expected_residual.data(), gpu_res.data(), n,
                  "composition-residual");

      // Replace the source residual with a deterministic "inverse transform"
      // residual so the reconstruction is not just the source. Both CPU and
      // GPU consume the same values, testing evolving neighbour dependencies.
      std::vector<std::int16_t> transform_residual(
          static_cast<std::size_t>(n));
      for (int i = 0; i < n; ++i) {
        const int value =
            ((i * 37 + x0 * 11 + y0 * 7 + pattern * 23) % 401) - 200;
        transform_residual[static_cast<std::size_t>(i)] =
            static_cast<std::int16_t>(value);
      }
      CUDA_CHECK(cudaMemcpy(d_res.get(), transform_residual.data(),
                            static_cast<std::size_t>(n) * sizeof(std::int16_t),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cuda::add_residual_clip_to_plane(
          d_pred.get(), d_res.get(), d_plane.get(), width, x0, y0, w, h, max_x,
          max_y, nullptr));

      for (int i = 0; i < n; ++i) {
        const int x = x0 + i % w;
        const int y = y0 + i / w;
        const int value = expected_pred +
                          static_cast<int>(transform_residual[
                              static_cast<std::size_t>(i)]);
        cpu_plane.data[static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(cpu_clip1(value));
      }
    }
  }

  std::vector<std::uint8_t> gpu_plane(cpu_plane.data.size());
  CUDA_CHECK(cudaMemcpy(gpu_plane.data(), d_plane.get(),
                        cpu_plane.data.size() * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost));
  g_context = std::string("composition=") + label + " final-plane";
  compare_u8(g_recon_plane, cpu_plane.data.data(), gpu_plane.data(),
             static_cast<int>(cpu_plane.data.size()), "composition-final");
}

// ---------------------------------------------------------------------------
// add_residual_clip stress: arbitrary residual values, saturation, bounds
// ---------------------------------------------------------------------------

void run_clip_stress() {
  constexpr int kBlock = cuda::kIntraMaxBlockDim;
  constexpr int kBlocks = 8;
  constexpr int kCount = kBlock * kBlock * kBlocks;

  std::vector<std::uint8_t> pred(static_cast<std::size_t>(kCount));
  std::vector<std::int16_t> residual(static_cast<std::size_t>(kCount));
  std::vector<std::uint8_t> expected(static_cast<std::size_t>(kCount));
  Rng rng(0x51ed270bu);
  for (int i = 0; i < kCount; ++i) {
    pred[static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>(rng.range(256));
    residual[static_cast<std::size_t>(i)] =
        static_cast<std::int16_t>(rng.next() & 0xffffu);
    expected[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(
        cpu_clip1(static_cast<int>(pred[static_cast<std::size_t>(i)]) +
                  static_cast<int>(residual[static_cast<std::size_t>(i)])));
  }

  DeviceBuffer<std::uint8_t> d_pred(static_cast<std::size_t>(kCount));
  DeviceBuffer<std::int16_t> d_res(static_cast<std::size_t>(kCount));
  DeviceBuffer<std::uint8_t> d_out(static_cast<std::size_t>(kCount));
  d_pred.upload(pred.data());
  d_res.upload(residual.data());

  g_context = "clip-stress 64x64 x8";
  for (int b = 0; b < kBlocks; ++b) {
    const std::size_t offset = static_cast<std::size_t>(b) * kBlock * kBlock;
    CUDA_CHECK(cuda::add_residual_clip(d_pred.get() + offset,
                                       d_res.get() + offset,
                                       d_out.get() + offset, kBlock, kBlock,
                                       nullptr));
  }
  std::vector<std::uint8_t> gpu_out(static_cast<std::size_t>(kCount));
  d_out.download(gpu_out.data());
  compare_u8(g_stress, expected.data(), gpu_out.data(), kCount, "saturate");

  // Plane variant with a block that extends past the visible bounds.
  constexpr int kPlaneW = 100;
  constexpr int kPlaneH = 100;
  Plane plane;
  plane.width = kPlaneW;
  plane.height = kPlaneH;
  plane.stride = kPlaneW;
  plane.data.assign(static_cast<std::size_t>(kPlaneW) * kPlaneH, 0xCD);
  std::vector<std::uint8_t> expected_plane = plane.data;
  for (int i = 0; i < kBlock * kBlock; ++i) {
    const int x = 50 + i % kBlock;
    const int y = 50 + i / kBlock;
    if (x >= kPlaneW || y >= kPlaneH) {
      continue;
    }
    expected_plane[static_cast<std::size_t>(y) * kPlaneW +
                   static_cast<std::size_t>(x)] =
        expected[static_cast<std::size_t>(i)];
  }
  DeviceBuffer<std::uint8_t> d_plane(
      static_cast<std::size_t>(kPlaneW) * kPlaneH);
  d_plane.upload(plane.data.data());
  g_context = "clip-stress to-plane 64x64 @ (50,50) in 100x100";
  CUDA_CHECK(cuda::add_residual_clip_to_plane(
      d_pred.get(), d_res.get(), d_plane.get(), kPlaneW, 50, 50, kBlock,
      kBlock, kPlaneW - 1, kPlaneH - 1, nullptr));
  std::vector<std::uint8_t> gpu_plane(expected_plane.size());
  d_plane.download(gpu_plane.data());
  compare_u8(g_stress, expected_plane.data(), gpu_plane.data(),
             static_cast<int>(gpu_plane.size()), "plane-saturate");
}

// ---------------------------------------------------------------------------
// Batched path correctness: 64 independent jobs, one CUDA block per job
// ---------------------------------------------------------------------------

void run_batch_correctness() {
  const Plane recon = make_plane(320, 180, 4);
  const Plane source = make_plane(320, 180, 2);
  constexpr int kJobs = 64;
  constexpr int kMaxSamples = 32 * 32;

  DeviceBuffer<std::uint8_t> d_recon(recon.data.size());
  DeviceBuffer<std::uint8_t> d_source(source.data.size());
  DeviceBuffer<std::uint8_t> d_pred(
      static_cast<std::size_t>(kJobs) * kMaxSamples);
  DeviceBuffer<std::int16_t> d_res(
      static_cast<std::size_t>(kJobs) * kMaxSamples);
  d_recon.upload(recon.data.data());
  d_source.upload(source.data.data());

  static const int kDims[][2] = {{32, 32}, {32, 16}, {32, 8},
                                 {16, 16}, {16, 8},  {16, 4}};
  constexpr int kDimCount =
      static_cast<int>(sizeof(kDims) / sizeof(kDims[0]));

  Rng rng(0xabcdef01u);
  std::vector<cuda::DcBlockJob> host_jobs(static_cast<std::size_t>(kJobs));
  std::vector<BlockCase> cases(static_cast<std::size_t>(kJobs));
  for (int i = 0; i < kJobs; ++i) {
    BlockCase& c = cases[static_cast<std::size_t>(i)];
    const int* dims = kDims[rng.range(kDimCount)];
    c.w = dims[0];
    c.h = dims[1];
    c.x0 = rng.range(recon.width);
    c.y0 = rng.range(recon.height);
    c.flags = (c.x0 > 0 ? 1 : 0) | (c.y0 > 0 ? 2 : 0);

    cuda::DcBlock block{};
    block.recon_plane = d_recon.get();
    block.recon_stride = recon.stride;
    block.x0 = c.x0;
    block.y0 = c.y0;
    block.block_w = c.w;
    block.block_h = c.h;
    block.plane_max_x = recon.width - 1;
    block.plane_max_y = recon.height - 1;
    block.have_left = c.flags & 1;
    block.have_above = (c.flags >> 1) & 1;
    block.source_plane = d_source.get();
    block.source_stride = source.stride;
    block.source_max_x = source.width - 1;
    block.source_max_y = source.height - 1;

    cuda::DcBlockJob& job = host_jobs[static_cast<std::size_t>(i)];
    job.block = block;
    job.pred_block =
        d_pred.get() + static_cast<std::size_t>(i) * kMaxSamples;
    job.residual_block =
        d_res.get() + static_cast<std::size_t>(i) * kMaxSamples;
  }

  DeviceBuffer<cuda::DcBlockJob> d_jobs(static_cast<std::size_t>(kJobs));
  d_jobs.upload(host_jobs.data());
  CUDA_CHECK(cuda::dc_predict_residual_batch(d_jobs.get(), kJobs, nullptr));

  std::vector<std::uint8_t> gpu_pred(
      static_cast<std::size_t>(kJobs) * kMaxSamples);
  std::vector<std::int16_t> gpu_res(
      static_cast<std::size_t>(kJobs) * kMaxSamples);
  d_pred.download(gpu_pred.data());
  d_res.download(gpu_res.data());

  for (int i = 0; i < kJobs; ++i) {
    const BlockCase& c = cases[static_cast<std::size_t>(i)];
    const int n = c.w * c.h;
    const std::size_t offset = static_cast<std::size_t>(i) * kMaxSamples;
    char context[160];
    std::snprintf(context, sizeof(context), "batch-job-%d (%d,%d,%dx%d)", i,
                  c.x0, c.y0, c.w, c.h);
    g_context = context;

    const int expected_pred =
        cpu_dc_predict(recon, c.x0, c.y0, c.w, c.h, (c.flags & 1) != 0,
                       (c.flags & 2) != 0);
    std::vector<std::uint8_t> expected_pred_block(
        static_cast<std::size_t>(n),
        static_cast<std::uint8_t>(expected_pred));
    std::vector<std::int16_t> expected_residual(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
      expected_residual[static_cast<std::size_t>(k)] =
          static_cast<std::int16_t>(
              cpu_residual_at(source, c.x0, c.y0, c.w, k, expected_pred));
    }
    compare_u8(g_pred, expected_pred_block.data(),
               gpu_pred.data() + offset, n, "batch-pred");
    compare_i16(g_residual, expected_residual.data(),
                gpu_res.data() + offset, n, "batch-residual");
  }
}


// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

template <typename LaunchFn>
float time_launches(LaunchFn launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, nullptr));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop, nullptr));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(iterations);
}

void print_stage_timing(const char* name, float ms_per_launch,
                        int samples_per_block) {
  const double us_per_launch = static_cast<double>(ms_per_launch) * 1000.0;
  const double mblocks_per_s = 1.0 / us_per_launch;  // 1 block/us == 1 Mblocks/s
  const double msamples_per_s =
      static_cast<double>(samples_per_block) / us_per_launch;
  std::printf("  %-24s %8.2f us/launch  (%6.2f Mblocks/s, %7.1f Msamples/s)\n",
              name, us_per_launch, mblocks_per_s, msamples_per_s);
}

void run_timing() {
  std::printf("\n-- timing (one block per launch unless noted) --\n");
  const Plane recon = make_plane(640, 360, 0);
  const Plane source = make_plane(640, 360, 3);

  // Build one valid 32x32 interior block on device planes.
  DeviceBuffer<std::uint8_t> d_recon(recon.data.size());
  DeviceBuffer<std::uint8_t> d_source(source.data.size());
  DeviceBuffer<std::uint8_t> d_pred(32 * 32);
  DeviceBuffer<std::int16_t> d_res(32 * 32);
  DeviceBuffer<std::uint8_t> d_recon_block(32 * 32);
  d_recon.upload(recon.data.data());
  d_source.upload(source.data.data());

  cuda::DcBlock block{};
  block.recon_plane = d_recon.get();
  block.recon_stride = 640;
  block.x0 = 64;
  block.y0 = 64;
  block.block_w = 32;
  block.block_h = 32;
  block.plane_max_x = 639;
  block.plane_max_y = 359;
  block.have_left = 1;
  block.have_above = 1;
  block.source_plane = d_source.get();
  block.source_stride = 640;
  block.source_max_x = 639;
  block.source_max_y = 359;

  constexpr int kWarmup = 100;
  constexpr int kIterations = 4000;

  const float predict_us = time_launches(
      [&] { CUDA_CHECK(cuda::dc_predict(block, d_pred.get(), nullptr)); },
      kWarmup, kIterations);
  print_stage_timing("dc_predict 32x32", predict_us, 1024);

  const float residual_us = time_launches(
      [&] {
        CUDA_CHECK(cuda::dc_residual(block, d_pred.get(), d_res.get(),
                                     nullptr));
      },
      kWarmup, kIterations);
  print_stage_timing("dc_residual 32x32", residual_us, 1024);

  const float fused_us = time_launches(
      [&] {
        CUDA_CHECK(cuda::dc_predict_residual(block, d_pred.get(), d_res.get(),
                                             nullptr));
      },
      kWarmup, kIterations);
  print_stage_timing("dc_predict_residual", fused_us, 1024);

  const float clip_us = time_launches(
      [&] {
        CUDA_CHECK(cuda::add_residual_clip(d_pred.get(), d_res.get(),
                                           d_recon_block.get(), 32, 32,
                                           nullptr));
      },
      kWarmup, kIterations);
  print_stage_timing("add_residual_clip", clip_us, 1024);

  // Batched, dependency-free prediction + residual: one CUDA block per job.
  constexpr int kJobs = 8192;
  constexpr int kJobSamples = 32 * 32;
  DeviceBuffer<std::uint8_t> d_batch_pred(
      static_cast<std::size_t>(kJobs) * kJobSamples);
  DeviceBuffer<std::int16_t> d_batch_res(
      static_cast<std::size_t>(kJobs) * kJobSamples);
  std::vector<cuda::DcBlockJob> host_jobs(kJobs);
  for (int i = 0; i < kJobs; ++i) {
    cuda::DcBlockJob& job = host_jobs[static_cast<std::size_t>(i)];
    job.block = block;
    job.block.x0 = 32 + (i % 16) * 32;
    job.block.y0 = 32 + ((i / 16) % 8) * 32;
    job.pred_block = d_batch_pred.get() + static_cast<std::size_t>(i) *
                                               kJobSamples;
    job.residual_block =
        d_batch_res.get() + static_cast<std::size_t>(i) * kJobSamples;
  }
  DeviceBuffer<cuda::DcBlockJob> d_jobs(static_cast<std::size_t>(kJobs));
  d_jobs.upload(host_jobs.data());
  const float batch_ms =
      time_launches([&] {
        CUDA_CHECK(cuda::dc_predict_residual_batch(d_jobs.get(), kJobs,
                                                   nullptr));
      }, 10, 200);
  const double batch_blocks_per_s =
      static_cast<double>(kJobs) / (static_cast<double>(batch_ms) * 1e-3);
  const double batch_samples_per_s =
      batch_blocks_per_s * static_cast<double>(kJobSamples);
  std::printf("  %-24s %8.3f ms/batch (%d blocks; %.2f Mblocks/s, "
              "%.1f Gsamples/s)\n",
              "batch predict+residual", static_cast<double>(batch_ms), kJobs,
              batch_blocks_per_s * 1e-6, batch_samples_per_s * 1e-9);

  // 640x360 M0-like raster: luma 32x32 (last row 32x8), two chroma 16x16
  // planes (last row 16x4). Per block: predict+residual + clip-to-plane.
  // Transform/quant are excluded (separate component).
  constexpr int kLumaW = 640;
  constexpr int kLumaH = 360;
  constexpr int kChromaW = 320;
  constexpr int kChromaH = 180;
  Plane luma_recon;
  luma_recon.width = kLumaW;
  luma_recon.height = kLumaH;
  luma_recon.stride = kLumaW;
  luma_recon.data.assign(static_cast<std::size_t>(kLumaW) * kLumaH, 128);
  Plane chroma_recon;
  chroma_recon.width = kChromaW;
  chroma_recon.height = kChromaH;
  chroma_recon.stride = kChromaW;
  chroma_recon.data.assign(static_cast<std::size_t>(kChromaW) * kChromaH, 128);

  DeviceBuffer<std::uint8_t> d_luma(luma_recon.data.size());
  DeviceBuffer<std::uint8_t> d_chroma_u(chroma_recon.data.size());
  DeviceBuffer<std::uint8_t> d_chroma_v(chroma_recon.data.size());
  DeviceBuffer<std::uint8_t> d_luma_src(luma_recon.data.size());
  DeviceBuffer<std::uint8_t> d_chroma_src(chroma_recon.data.size());
  d_luma.upload(luma_recon.data.data());
  d_chroma_u.upload(chroma_recon.data.data());
  d_chroma_v.upload(chroma_recon.data.data());
  {
    const Plane luma_source = make_plane(kLumaW, kLumaH, 0);
    const Plane chroma_source = make_plane(kChromaW, kChromaH, 0);
    d_luma_src.upload(luma_source.data.data());
    d_chroma_src.upload(chroma_source.data.data());
  }
  DeviceBuffer<std::uint8_t> d_frame_pred(32 * 32);
  DeviceBuffer<std::int16_t> d_frame_res(32 * 32);

  struct FrameBlock {
    int plane;
    int x0;
    int y0;
    int w;
    int h;
  };
  std::vector<FrameBlock> frame_blocks;
  for (int y0 = 0; y0 < kLumaH; y0 += 32) {
    const int h = std::min(32, kLumaH - y0);
    for (int x0 = 0; x0 < kLumaW; x0 += 32) {
      frame_blocks.push_back({0, x0, y0, 32, h});
    }
  }
  for (int plane = 1; plane <= 2; ++plane) {
    for (int y0 = 0; y0 < kChromaH; y0 += 16) {
      const int h = std::min(16, kChromaH - y0);
      for (int x0 = 0; x0 < kChromaW; x0 += 16) {
        frame_blocks.push_back({plane, x0, y0, 16, h});
      }
    }
  }

  const float frame_ms = time_launches(
      [&] {
        for (const FrameBlock& fb : frame_blocks) {
          const bool is_luma = fb.plane == 0;
          const int width = is_luma ? kLumaW : kChromaW;
          const int height = is_luma ? kLumaH : kChromaH;
          std::uint8_t* plane_ptr = is_luma
                                        ? d_luma.get()
                                        : (fb.plane == 1 ? d_chroma_u.get()
                                                         : d_chroma_v.get());
          const std::uint8_t* source_ptr =
              is_luma ? d_luma_src.get() : d_chroma_src.get();
          cuda::DcBlock fb_block{};
          fb_block.recon_plane = plane_ptr;
          fb_block.recon_stride = width;
          fb_block.x0 = fb.x0;
          fb_block.y0 = fb.y0;
          fb_block.block_w = fb.w;
          fb_block.block_h = fb.h;
          fb_block.plane_max_x = width - 1;
          fb_block.plane_max_y = height - 1;
          fb_block.have_left = fb.x0 > 0 ? 1 : 0;
          fb_block.have_above = fb.y0 > 0 ? 1 : 0;
          fb_block.source_plane = source_ptr;
          fb_block.source_stride = width;
          fb_block.source_max_x = width - 1;
          fb_block.source_max_y = height - 1;
          CUDA_CHECK(cuda::dc_predict_residual(fb_block, d_frame_pred.get(),
                                               d_frame_res.get(), nullptr));
          CUDA_CHECK(cuda::add_residual_clip_to_plane(
              d_frame_pred.get(), d_frame_res.get(), plane_ptr, width, fb.x0,
              fb.y0, fb.w, fb.h, width - 1, height - 1, nullptr));
        }
      },
      2, 10);

  std::printf(
      "  %-24s %8.3f ms/frame (%zu blocks, 2 launches/block, 640x360)\n",
      "raster intra pass", static_cast<double>(frame_ms),
      frame_blocks.size());
}

// ---------------------------------------------------------------------------
// Padded-stride test: the CPU reconstruction plane is 64-aligned, so the
// stride can exceed the visible width. Padding must never be read or written.
// ---------------------------------------------------------------------------

Plane make_padded_plane(int width, int height, int stride, int padded_height,
                        int pattern) {
  Plane plane;
  plane.width = width;
  plane.height = height;
  plane.stride = stride;
  plane.data.assign(static_cast<std::size_t>(stride) *
                        static_cast<std::size_t>(padded_height),
                    0xEE);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      plane.data[static_cast<std::size_t>(y) *
                     static_cast<std::size_t>(stride) +
                 static_cast<std::size_t>(x)] = pattern_value(pattern, x, y);
    }
  }
  return plane;
}

void run_padded_stride_test() {
  constexpr int kWidth = 100;
  constexpr int kHeight = 70;
  constexpr int kStride = 128;
  constexpr int kPaddedHeight = 72;

  const Plane recon =
      make_padded_plane(kWidth, kHeight, kStride, kPaddedHeight, 4);
  const Plane source = make_padded_plane(kWidth, kHeight, 112, kHeight, 0);
  PlaneRunner runner(recon, source);

  const std::vector<BlockCase> cases = {
      {0, 0, 32, 32, -1, "pad-tl"},
      {64, 0, 32, 32, -1, "pad-tr"},
      {0, 64, 32, 8, -1, "pad-bl-extends"},
      {64, 64, 32, 8, -1, "pad-br-extends"},
      {0, 0, 64, 64, -1, "pad-tl-64"},
      {32, 32, 32, 32, -1, "pad-interior"},
      {96, 0, 32, 32, -1, "pad-right-extends"},
      {0, 32, 32, 64, -1, "pad-bottom-extends"},
      {96, 64, 32, 8, -1, "pad-corner-extends"},
  };
  for (const BlockCase& c : cases) {
    runner.run_case(c);
  }
  for (int flags = 0; flags < 4; ++flags) {
    runner.run_case({32, 32, 32, 32, flags, "pad-flag-matrix"});
  }
  runner.run_random_cases(300, 0xdeadbeefu);
}


void run_api_validation() {
  std::printf("\n-- API validation --\n");
  cuda::DcBlock block{};
  block.block_w = 32;
  block.block_h = 32;
  block.x0 = 0;
  block.y0 = 0;
  block.plane_max_x = 639;
  block.plane_max_y = 359;

  int failures = 0;
  const auto expect_invalid = [&failures](cudaError_t err, const char* what) {
    if (err != cudaErrorInvalidValue) {
      std::printf("  FAIL: %s returned %s (expected cudaErrorInvalidValue)\n",
                  what, cudaGetErrorString(err));
      ++failures;
    }
  };

  DeviceBuffer<std::uint8_t> dummy(64 * 64);
  expect_invalid(cuda::dc_predict(block, nullptr, nullptr), "null pred output");
  block.block_w = 0;
  expect_invalid(cuda::dc_predict(block, dummy.get(), nullptr),
                 "zero block width");
  block.block_w = 32;
  block.have_left = 1;
  block.x0 = 0;
  expect_invalid(cuda::dc_predict(block, dummy.get(), nullptr),
                 "left without x0>0");
  block.have_left = 0;
  block.source_plane = nullptr;
  block.source_stride = 0;
  block.source_max_x = 0;
  block.source_max_y = 0;
  expect_invalid(cuda::dc_residual(block, nullptr, nullptr, nullptr),
                 "null residual output");
  if (failures == 0) {
    std::printf("  all invalid-argument checks passed\n");
  } else {
    std::printf("  %d invalid-argument checks FAILED\n", failures);
    g_pred.mismatches += failures;  // make main() fail
  }
}

void print_summary() {
  const Stage* stages[] = {&g_pred,      &g_pred_value, &g_residual,
                           &g_recon,     &g_recon_plane, &g_stress};
  std::printf("\n-- bit-exactness summary --\n");
  std::printf("  %-14s %16s %12s %10s\n", "stage", "samples", "mismatches",
              "max|diff|");
  for (const Stage* stage : stages) {
    std::printf("  %-14s %16lld %12lld %10d\n", stage->name, stage->samples,
                stage->mismatches, stage->max_abs_diff);
  }
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::printf("av2enc CUDA intra test\n");
  std::printf("GPU: %s, sm_%d%d, %d SMs, %.1f GiB\n", properties.name,
              properties.major, properties.minor,
              properties.multiProcessorCount,
              static_cast<double>(properties.totalGlobalMem) /
                  (1024.0 * 1024.0 * 1024.0));

  std::printf("\n-- correctness: synthetic planes, all block classes --\n");
  for (int pattern = 0; pattern < kPatternCount; ++pattern) {
    const Stage pred_before = g_pred;
    const Stage residual_before = g_residual;
    const Stage recon_before = g_recon;
    const Stage plane_before = g_recon_plane;

    {
      const Plane recon = make_plane(640, 360, pattern);
      const Plane source =
          make_plane(640, 360, (pattern + 3) % kPatternCount);
      PlaneRunner runner(recon, source);
      for (const BlockCase& c : make_luma_cases()) {
        runner.run_case(c);
      }
      runner.run_random_cases(400, 0x9e3779b9u ^
                                       static_cast<std::uint32_t>(pattern));
      for (int flags = 0; flags < 4; ++flags) {
        runner.run_case({32, 32, 32, 32, flags, "luma-flag-random-pos"});
      }
    }
    {
      const Plane recon = make_plane(320, 180, pattern);
      const Plane source =
          make_plane(320, 180, (pattern + 5) % kPatternCount);
      PlaneRunner runner(recon, source);
      for (const BlockCase& c : make_chroma_cases()) {
        runner.run_case(c);
      }
      runner.run_random_cases(400, 0x85ebca6bu ^
                                       static_cast<std::uint32_t>(pattern));
    }

    std::printf(
        "  pattern %d (%s): pred %lld, residual %lld, recon %lld, plane %lld "
        "mismatches\n",
        pattern, pattern_name(pattern), g_pred.mismatches - pred_before.mismatches,
        g_residual.mismatches - residual_before.mismatches,
        g_recon.mismatches - recon_before.mismatches,
        g_recon_plane.mismatches - plane_before.mismatches);
  }

  run_padded_stride_test();
  std::printf(
      "  padded stride (visible 100x70, recon stride 128, sentinel padding): "
      "done\n");

  std::printf("\n-- correctness: full-plane raster, evolving reconstruction --\n");
  for (int pattern = 0; pattern < kPatternCount; ++pattern) {
    run_composition("luma-32x32", 640, 360, 32, 32, pattern);
    run_composition("chroma-16x16", 320, 180, 16, 16, pattern);
  }
  std::printf("  luma 640x360 + chroma 320x180 raster x %d patterns: done\n",
              kPatternCount);

  run_clip_stress();
  run_batch_correctness();
  std::printf("  batch path: 64 independent jobs: done\n");
  run_api_validation();
  run_timing();
  print_summary();

  long long total = 0;
  for (const Stage* stage : {&g_pred, &g_pred_value, &g_residual, &g_recon,
                             &g_recon_plane, &g_stress}) {
    total += stage->mismatches;
  }
  if (total != 0) {
    std::printf("\nRESULT: FAIL (%lld mismatches)\n", total);
    return 1;
  }
  std::printf("\nRESULT: PASS (bit-exact with CPU reference)\n");
  return 0;
}
