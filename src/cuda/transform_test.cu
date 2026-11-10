// CUDA transform test: bit-exact comparison against the CPU reference.
//
// The `cpu` namespace below is copied from src/encoder/m0_still.cpp
// (TileEncoder::forward_dct, TileEncoder::inverse_dct, transform_shifts,
// Round2/clip3 helpers, and the transform/quantization block of
// encode_residual_plane).  It is the reference; do not "optimize" it.
//
// Build/run:  ./build_transform_test.sh && ./transform_test
#include "cuda/transform.h"
#include "encoder/m0_tables.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference (verbatim from src/encoder/m0_still.cpp)
// ---------------------------------------------------------------------------
namespace cpu {

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int clip1(int value) { return clip3(0, 255, value); }

// §4.8: Round2(x, n) = (x + (1 << (n - 1))) >> n (arithmetic shift).
int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

const std::int16_t* dct_kernel(int n) {
  switch (n) {
    case 4:
      return &av2enc::m0::kDct4[0][0];
    case 8:
      return &av2enc::m0::kDct8[0][0];
    case 16:
      return &av2enc::m0::kDct16[0][0];
    default:
      return &av2enc::m0::kDct32[0][0];
  }
}

void transform_shifts(int w, int h, int* row_shift, int* col_shift) {
  if (w == 32 && h == 32) {
    *row_shift = 6;
    *col_shift = 13;
  } else if (w == 16 && h == 16) {
    *row_shift = 6;
    *col_shift = 13;
  } else if ((w == 32 && h == 16) || (w == 16 && h == 32)) {
    *row_shift = 6;
    *col_shift = 12;
  } else if ((w == 32 && h == 8) || (w == 8 && h == 32)) {
    *row_shift = 6;
    *col_shift = 13;
  } else if ((w == 16 && h == 4) || (w == 4 && h == 16)) {
    *row_shift = 6;
    *col_shift = 12;
  } else if (w <= 16 && h <= 16) {
    *row_shift = 7;
    *col_shift = 11;
  } else {
    *row_shift = 6;
    *col_shift = 13;
  }
}

// Orthonormal 2D DCT-II through the normative integer kernel.
void forward_dct(const std::vector<int>& block, int w, int h,
                 std::vector<double>* out) {
  const std::int16_t* kw = dct_kernel(w);
  const std::int16_t* kh = dct_kernel(h);
  const double scale_w = 1.0 / (64.0 * std::sqrt(static_cast<double>(w)));
  const double scale_h = 1.0 / (64.0 * std::sqrt(static_cast<double>(h)));
  std::vector<double> tmp(static_cast<std::size_t>(w) * h, 0.0);
  out->assign(static_cast<std::size_t>(w) * h, 0.0);
  for (int i = 0; i < h; ++i) {
    for (int v = 0; v < w; ++v) {
      double sum = 0.0;
      for (int j = 0; j < w; ++j) {
        sum += static_cast<double>(kw[v * w + j]) *
               static_cast<double>(block[static_cast<std::size_t>(i) * w + j]);
      }
      tmp[static_cast<std::size_t>(i) * w + v] = sum * scale_w;
    }
  }
  for (int u = 0; u < h; ++u) {
    for (int v = 0; v < w; ++v) {
      double sum = 0.0;
      for (int i = 0; i < h; ++i) {
        sum += static_cast<double>(kh[u * h + i]) *
               tmp[static_cast<std::size_t>(i) * w + v];
      }
      (*out)[static_cast<std::size_t>(u) * w + v] = sum * scale_h;
    }
  }
}

// Normative 2D inverse matrix transform (§7.15.4.1).
void inverse_dct(const std::vector<int>& dequant, int w, int h,
                 std::vector<int>* residual) {
  int row_shift = 0;
  int col_shift = 0;
  transform_shifts(w, h, &row_shift, &col_shift);
  const std::int16_t* kw = dct_kernel(w);
  const std::int16_t* kh = dct_kernel(h);
  std::vector<int> intermediate(static_cast<std::size_t>(w) * h, 0);
  for (int i = 0; i < h; ++i) {
    for (int oj = 0; oj < w; ++oj) {
      std::int64_t sum = 0;
      for (int j = 0; j < w; ++j) {
        sum += static_cast<std::int64_t>(kw[j * w + oj]) *
               dequant[static_cast<std::size_t>(i) * w + j];
      }
      intermediate[static_cast<std::size_t>(i) * w + oj] = clip3(
          -(1 << 15), (1 << 15) - 1, round2(static_cast<int>(sum), row_shift));
    }
  }
  residual->assign(static_cast<std::size_t>(w) * h, 0);
  for (int oi = 0; oi < h; ++oi) {
    for (int j = 0; j < w; ++j) {
      std::int64_t sum = 0;
      for (int i = 0; i < h; ++i) {
        sum += static_cast<std::int64_t>(kh[i * h + oi]) *
               intermediate[static_cast<std::size_t>(i) * w + j];
      }
      (*residual)[static_cast<std::size_t>(oi) * w + j] =
          clip3(-256, 255, round2(static_cast<int>(sum), col_shift));
    }
  }
}

constexpr int kQuantStep = 528;
constexpr int kQuantTableBits = 3;  // QUANT_TABLE_BITS

struct RefTx {
  int tw;
  int th;
  int upscale_x;
  int upscale_y;
  int dq_denom;
  int row_shift;
  int col_shift;
  double inv_scale;
  double quant_factor;
};

// encode_residual_plane() lines 1251-1277.
RefTx ref_tx(int width, int height, bool luma) {
  int tw = width;
  int th = height;
  int upscale_x = 1;
  int upscale_y = 1;
  if (width == 64 && height == 64 && luma) {
    tw = 32;
    th = 32;
    upscale_x = 2;
    upscale_y = 2;
  }
  const int tx_pels = (upscale_x * tw) * (upscale_y * th);
  const int dq_denom = 1 << ((tx_pels > 256) + (tx_pels > 1024));
  int row_shift = 0;
  int col_shift = 0;
  transform_shifts(tw, th, &row_shift, &col_shift);
  const double inv_scale =
      (64.0 * std::sqrt(static_cast<double>(tw)) /
       static_cast<double>(1 << row_shift)) *
      (64.0 * std::sqrt(static_cast<double>(th)) /
       static_cast<double>(1 << col_shift));
  const double quant_factor =
      8.0 * static_cast<double>(dq_denom) /
      (inv_scale * static_cast<double>(kQuantStep));
  return {tw,     th,         upscale_x, upscale_y,  dq_denom,
          row_shift, col_shift, inv_scale, quant_factor};
}

// encode_residual_plane() lines 1306-1310.
void quantize(const std::vector<double>& coeff, double quant_factor,
              std::vector<int>* levels) {
  levels->resize(coeff.size());
  for (std::size_t pos = 0; pos < coeff.size(); ++pos) {
    (*levels)[pos] =
        static_cast<int>(std::lround(coeff[pos] * quant_factor));
  }
}

// encode_residual_plane() lines 1541-1551.
void dequantize(const std::vector<int>& levels, int dq_denom,
                std::vector<int>* dequant) {
  dequant->assign(levels.size(), 0);
  for (std::size_t pos = 0; pos < levels.size(); ++pos) {
    const int qc = levels[pos];
    if (qc == 0) {
      continue;
    }
    const int dq_high = (std::abs(qc) * kQuantStep) & 0xFFFFFF;
    const int dq = round2(dq_high, kQuantTableBits);
    const int value = dq / dq_denom;
    (*dequant)[pos] = qc < 0 ? -value : value;
  }
}

// TileEncoder::source_sample(): clip to the visible plane bounds.
std::uint8_t source_sample(const std::vector<std::uint8_t>& plane, int width,
                           int height, int x, int y) {
  const int cx = clip3(0, width - 1, x);
  const int cy = clip3(0, height - 1, y);
  return plane[static_cast<std::size_t>(cy) * width + cx];
}

// encode_residual_plane() lines 1280-1303 for the 64x64 -> 32x32 path.
std::vector<int> make_residual_64x64(const std::vector<std::uint8_t>& plane,
                                     int width, int height, int x0, int y0,
                                     int pred) {
  std::vector<int> residual(32 * 32, 0);
  for (int j = 0; j < 32; ++j) {
    for (int i = 0; i < 32; ++i) {
      int sum = 0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          sum += static_cast<int>(
                     source_sample(plane, width, height, x0 + 2 * i + dx,
                                   y0 + 2 * j + dy)) -
                 pred;
        }
      }
      residual[static_cast<std::size_t>(j) * 32 + i] = round2(sum, 2);
    }
  }
  return residual;
}

// Same 2x2 average applied to a full 64x64 residual.
std::vector<int> downsample_residual_64x64(
    const std::vector<std::int16_t>& residual64) {
  std::vector<int> residual(32 * 32, 0);
  for (int j = 0; j < 32; ++j) {
    for (int i = 0; i < 32; ++i) {
      int sum = 0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          sum += static_cast<int>(
              residual64[static_cast<std::size_t>(2 * j + dy) * 64 + 2 * i + dx]);
        }
      }
      residual[static_cast<std::size_t>(j) * 32 + i] = round2(sum, 2);
    }
  }
  return residual;
}

// encode_residual_plane() lines 1554-1566.
void reconstruct_64x64(const std::vector<int>& residual, int pred,
                       std::vector<std::uint8_t>* recon, int recon_width,
                       int x0, int y0) {
  for (int j = 0; j < 32; ++j) {
    for (int i = 0; i < 32; ++i) {
      const int value =
          clip1(pred + residual[static_cast<std::size_t>(j) * 32 + i]);
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          (*recon)[static_cast<std::size_t>(y0 + j * 2 + dy) * recon_width +
                   static_cast<std::size_t>(x0 + i * 2 + dx)] =
              static_cast<std::uint8_t>(value);
        }
      }
    }
  }
}

}  // namespace cpu

// ---------------------------------------------------------------------------
// Test utilities
// ---------------------------------------------------------------------------
namespace {

int g_failures = 0;
int g_checks = 0;

void cuda_ok(cudaError_t err, const char* what) {
  ++g_checks;
  if (err != cudaSuccess) {
    std::printf("  FAIL: %s: %s\n", what, cudaGetErrorString(err));
    ++g_failures;
    std::exit(2);
  }
}

template <typename T>
class DevBuf {
 public:
  explicit DevBuf(std::size_t count) : count_(count) {
    if (count_ != 0) {
      cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)),
              "cudaMalloc");
    }
  }
  ~DevBuf() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  void upload(const std::vector<T>& src) {
    cuda_ok(cudaMemcpy(ptr_, src.data(), src.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
  }
  void download(std::vector<T>* dst) const {
    dst->resize(count_);
    cuda_ok(cudaMemcpy(dst->data(), ptr_, count_ * sizeof(T),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H");
  }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

struct Stat {
  long long count = 0;
  long long mismatch = 0;
  double max_abs = 0.0;
  long long first = -1;
  double first_ref = 0.0;
  double first_got = 0.0;
};

double as_double(double v) { return v; }
double as_double(std::uint8_t v) { return static_cast<double>(v); }
double as_double(std::int16_t v) { return static_cast<double>(v); }
double as_double(std::int32_t v) { return static_cast<double>(v); }

std::vector<std::int32_t> to_i32(const std::vector<int>& v) {
  std::vector<std::int32_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = static_cast<std::int32_t>(v[i]);
  }
  return out;
}

std::vector<std::int16_t> to_i16(const std::vector<int>& v) {
  std::vector<std::int16_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = static_cast<std::int16_t>(v[i]);
  }
  return out;
}

template <typename T>
Stat compare_exact_typed(const std::vector<T>& ref, const std::vector<T>& got) {
  Stat s;
  s.count = static_cast<long long>(ref.size());
  if (got.size() != ref.size()) {
    s.mismatch = s.count;
    return s;
  }
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const bool equal =
        std::memcmp(&ref[i], &got[i], sizeof(T)) == 0;
    if (!equal) {
      if (s.first < 0) {
        s.first = static_cast<long long>(i);
        s.first_ref = as_double(ref[i]);
        s.first_got = as_double(got[i]);
      }
      ++s.mismatch;
    }
    const double d = std::fabs(as_double(ref[i]) - as_double(got[i]));
    if (d > s.max_abs) {
      s.max_abs = d;
    }
  }
  return s;
}

void report(const char* stage, const Stat& s) {
  ++g_checks;
  std::printf(" %s[n=%lld mismatch=%lld max_abs=%.9g]", stage, s.count,
              s.mismatch, s.max_abs);
  if (s.mismatch != 0) {
    std::printf("\n  FAIL: %s: first index %lld: cpu=%.17g gpu=%.17g", stage,
                s.first, s.first_ref, s.first_got);
    ++g_failures;
  }
}

void merge_stat(Stat* acc, const Stat& s) {
  const bool had_mismatch = acc->mismatch != 0;
  acc->count += s.count;
  acc->mismatch += s.mismatch;
  if (s.mismatch != 0 && !had_mismatch) {
    acc->first = s.first;
    acc->first_ref = s.first_ref;
    acc->first_got = s.first_got;
  }
  if (s.max_abs > acc->max_abs) {
    acc->max_abs = s.max_abs;
  }
}

void check_params(const av2enc::cuda::TxParams& p, const cpu::RefTx& ref,
                  const char* label) {
  ++g_checks;
  bool ok = p.w == ref.tw && p.h == ref.th &&
            p.dq_denom == ref.dq_denom && p.row_shift == ref.row_shift &&
            p.col_shift == ref.col_shift;
  const bool qf_bits =
      std::memcmp(&p.quant_factor, &ref.quant_factor, sizeof(double)) == 0;
  if (!ok || !qf_bits) {
    std::printf(
        "  FAIL: params %s: gpu{w=%d h=%d denom=%d shifts=%d/%d qf=%.17g} "
        "cpu{tw=%d th=%d denom=%d shifts=%d/%d qf=%.17g}\n",
        label, p.w, p.h, p.dq_denom, p.row_shift, p.col_shift, p.quant_factor,
        ref.tw, ref.th, ref.dq_denom, ref.row_shift, ref.col_shift,
        ref.quant_factor);
    ++g_failures;
  }
}

// ---------------------------------------------------------------------------
// Per-size transform test
// ---------------------------------------------------------------------------
std::vector<std::int16_t> build_blocks(int w, int h, int nrand,
                                       std::mt19937* rng, int* nblocks_out) {
  const int n = w * h;
  std::vector<std::int16_t> blocks;
  auto add = [&](const std::vector<int>& v) {
    for (int k = 0; k < n; ++k) {
      blocks.push_back(static_cast<std::int16_t>(v[static_cast<std::size_t>(k)]));
    }
  };
  auto fill = [&](int kind) {
    std::vector<int> v(static_cast<std::size_t>(n), 0);
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        int value = 0;
        switch (kind) {
          case 0: value = 0; break;
          case 1: value = 255; break;
          case 2: value = -255; break;
          case 3: value = 1; break;
          case 4: value = -1; break;
          case 5: value = ((i * 37 + j * 11) % 511) - 255; break;
          case 6: value = ((j * 37 + i * 11) % 511) - 255; break;
          default: value = ((i ^ j) & 1) ? 255 : -255; break;
        }
        v[static_cast<std::size_t>(j) * w + i] = value;
      }
    }
    add(v);
  };
  for (int kind = 0; kind < 8; ++kind) {
    fill(kind);
  }
  for (int p = 0; p < n; ++p) {
    std::vector<int> v(static_cast<std::size_t>(n), 0);
    v[static_cast<std::size_t>(p)] = ((p & 1) != 0) ? -255 : 255;
    add(v);
  }
  std::uniform_int_distribution<int> dist(-255, 255);
  for (int b = 0; b < nrand; ++b) {
    std::vector<int> v(static_cast<std::size_t>(n), 0);
    for (int k = 0; k < n; ++k) {
      v[static_cast<std::size_t>(k)] = dist(*rng);
    }
    add(v);
  }
  *nblocks_out = static_cast<int>(blocks.size() / static_cast<std::size_t>(n));
  return blocks;
}

bool test_size(int w, int h, std::mt19937* rng) {
  std::printf("[%2dx%-2d]", w, h);
  std::fflush(stdout);

  const auto opt = av2enc::cuda::make_tx_params(w, h, true);
  if (!opt) {
    std::printf(" FAIL: make_tx_params returned nullopt\n");
    ++g_failures;
    return false;
  }
  const av2enc::cuda::TxParams& p = *opt;
  const cpu::RefTx ref = cpu::ref_tx(w, h, true);
  check_params(p, ref, "size");

  int nblocks = 0;
  const std::vector<std::int16_t> residual =
      build_blocks(w, h, 256, rng, &nblocks);
  const int n = w * h;
  std::printf(" blocks=%d", nblocks);

  // CPU reference, block by block (same order and arithmetic as the encoder).
  std::vector<double> coeff_cpu(static_cast<std::size_t>(nblocks) * n, 0.0);
  std::vector<int> levels_cpu(static_cast<std::size_t>(nblocks) * n, 0);
  std::vector<int> dequant_cpu(static_cast<std::size_t>(nblocks) * n, 0);
  std::vector<int> resid_cpu(static_cast<std::size_t>(nblocks) * n, 0);
  std::vector<int> block_int(static_cast<std::size_t>(n), 0);
  std::vector<double> coeff_block;
  std::vector<int> levels_block;
  std::vector<int> dequant_block;
  std::vector<int> resid_block;
  for (int b = 0; b < nblocks; ++b) {
    for (int k = 0; k < n; ++k) {
      block_int[static_cast<std::size_t>(k)] =
          residual[static_cast<std::size_t>(b) * n + k];
    }
    cpu::forward_dct(block_int, w, h, &coeff_block);
    cpu::quantize(coeff_block, ref.quant_factor, &levels_block);
    cpu::dequantize(levels_block, ref.dq_denom, &dequant_block);
    cpu::inverse_dct(dequant_block, w, h, &resid_block);
    const std::size_t off = static_cast<std::size_t>(b) * n;
    for (int k = 0; k < n; ++k) {
      coeff_cpu[off + k] = coeff_block[static_cast<std::size_t>(k)];
      levels_cpu[off + k] = levels_block[static_cast<std::size_t>(k)];
      dequant_cpu[off + k] = dequant_block[static_cast<std::size_t>(k)];
      resid_cpu[off + k] = resid_block[static_cast<std::size_t>(k)];
    }
  }

  // GPU stages.
  DevBuf<std::int16_t> d_residual(residual.size());
  DevBuf<double> d_coeff(coeff_cpu.size());
  DevBuf<std::int32_t> d_levels(levels_cpu.size());
  DevBuf<std::int32_t> d_dequant(dequant_cpu.size());
  DevBuf<std::int16_t> d_resout(resid_cpu.size());
  DevBuf<double> d_coeff_scratch(coeff_cpu.size());
  DevBuf<std::int32_t> d_dequant_scratch(dequant_cpu.size());

  d_residual.upload(residual);
  cuda_ok(av2enc::cuda::forward_dct(d_residual.get(), d_coeff.get(), p,
                                    nblocks, nullptr),
          "forward_dct launch");
  cuda_ok(av2enc::cuda::quantize(d_coeff.get(), d_levels.get(), p, nblocks,
                                 nullptr),
          "quantize launch");
  cuda_ok(av2enc::cuda::dequantize(d_levels.get(), d_dequant.get(), p, nblocks,
                                   nullptr),
          "dequantize launch");
  cuda_ok(av2enc::cuda::inverse_dct(d_dequant.get(), d_resout.get(), p, nblocks,
                                    nullptr),
          "inverse_dct launch");

  std::vector<double> coeff_gpu;
  std::vector<std::int32_t> levels_gpu;
  std::vector<std::int32_t> dequant_gpu;
  std::vector<std::int16_t> resid_gpu;
  d_coeff.download(&coeff_gpu);
  d_levels.download(&levels_gpu);
  d_dequant.download(&dequant_gpu);
  d_resout.download(&resid_gpu);

  report(" coeff", compare_exact_typed(coeff_cpu, coeff_gpu));
  report(" levels", compare_exact_typed(to_i32(levels_cpu), levels_gpu));
  report(" dequant", compare_exact_typed(to_i32(dequant_cpu), dequant_gpu));
  report(" resid", compare_exact_typed(to_i16(resid_cpu), resid_gpu));

  // Composed entry points must produce identical results.
  cuda_ok(av2enc::cuda::forward_transform(d_residual.get(), d_coeff_scratch.get(),
                                          d_levels.get(), p, nblocks, nullptr),
          "forward_transform launch");
  cuda_ok(av2enc::cuda::inverse_transform(d_levels.get(), d_dequant_scratch.get(),
                                          d_resout.get(), p, nblocks, nullptr),
          "inverse_transform launch");
  std::vector<std::int32_t> levels_gpu2;
  std::vector<std::int16_t> resid_gpu2;
  d_levels.download(&levels_gpu2);
  d_resout.download(&resid_gpu2);
  report(" fused", compare_exact_typed(levels_gpu, levels_gpu2));
  report(" ifused", compare_exact_typed(resid_gpu, resid_gpu2));

  std::printf("\n");
  return true;
}

// ---------------------------------------------------------------------------
// Host parameter mirror
// ---------------------------------------------------------------------------
void test_params() {
  std::printf("[params]\n");
  struct Case {
    int w;
    int h;
    bool luma;
  };
  const Case cases[] = {
      {32, 32, true},  {16, 16, false}, {32, 16, true}, {16, 8, false},
      {32, 8, true},   {16, 4, false},  {16, 32, true}, {8, 16, false},
      {8, 8, false},   {4, 4, false},   {4, 16, false}, {16, 4, true},
      {64, 64, true},  {32, 32, false}, {8, 32, true},   {32, 16, false},
  };
  for (const Case& c : cases) {
    const auto opt = av2enc::cuda::make_tx_params(c.w, c.h, c.luma);
    ++g_checks;
    if (!opt) {
      std::printf("  FAIL: make_tx_params(%d,%d,%d) returned nullopt\n", c.w,
                  c.h, c.luma ? 1 : 0);
      ++g_failures;
      continue;
    }
    const cpu::RefTx ref = cpu::ref_tx(c.w, c.h, c.luma);
    char label[64];
    std::snprintf(label, sizeof(label), "%dx%d/luma=%d", c.w, c.h,
                  c.luma ? 1 : 0);
    check_params(*opt, ref, label);
  }
  // Sizes the CPU reference cannot transform (dct_kernel(64) is not a 64-point
  // kernel): the CUDA entry point must reject them instead of guessing.
  const Case bad[] = {{64, 32, true}, {32, 64, true}, {64, 16, true}};
  for (const Case& c : bad) {
    ++g_checks;
    if (av2enc::cuda::make_tx_params(c.w, c.h, c.luma).has_value()) {
      std::printf("  FAIL: make_tx_params(%d,%d) should be rejected\n", c.w,
                  c.h);
      ++g_failures;
    }
  }
  std::printf("  %zu block types checked, %zu rejected as expected\n",
              sizeof(cases) / sizeof(cases[0]), sizeof(bad) / sizeof(bad[0]));
}

// ---------------------------------------------------------------------------
// Rounding / dequantization edge values
// ---------------------------------------------------------------------------
void test_rounding(std::mt19937* rng) {
  std::printf("[lround]\n");
  av2enc::cuda::TxParams p = *av2enc::cuda::make_tx_params(32, 32, true);

  // quant_factor = 1.0 makes the quantize kernel compute lround(x) directly.
  p.quant_factor = 1.0;
  std::vector<double> values;
  for (int k = -5000; k <= 5000; ++k) {
    values.push_back(static_cast<double>(k) + 0.5);
    values.push_back(-(static_cast<double>(k) + 0.5));
  }
  const double specials[] = {0.49999999999999994, -0.49999999999999994,
                             0.50000000000000011, -0.50000000000000011,
                             0.5, -0.5, 1.5, -1.5, 0.0, -0.0};
  for (double v : specials) {
    values.push_back(v);
  }
  std::uniform_real_distribution<double> dist(-1.0e6, 1.0e6);
  for (int i = 0; i < 200000; ++i) {
    values.push_back(dist(*rng));
  }
  // Ensure the buffer is a whole number of 32x32 blocks.
  const std::size_t n = 32 * 32;
  while (values.size() % n != 0) {
    values.push_back(0.0);
  }
  const int nblocks = static_cast<int>(values.size() / n);

  std::vector<std::int32_t> levels_cpu(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    levels_cpu[i] = static_cast<std::int32_t>(std::lround(values[i]));
  }
  DevBuf<double> d_values(values.size());
  DevBuf<std::int32_t> d_levels(values.size());
  d_values.upload(values);
  cuda_ok(av2enc::cuda::quantize(d_values.get(), d_levels.get(), p, nblocks,
                                 nullptr),
          "quantize lround");
  std::vector<std::int32_t> levels_gpu;
  d_levels.download(&levels_gpu);
  report(" qf=1", compare_exact_typed(levels_cpu, levels_gpu));

  // Real quant_factor from the 32x32 luma block.
  const av2enc::cuda::TxParams rp = *av2enc::cuda::make_tx_params(32, 32, true);
  std::vector<double> coeff(values.size());
  std::uniform_real_distribution<double> cdist(-2000.0, 2000.0);
  for (std::size_t i = 0; i < coeff.size(); ++i) {
    coeff[i] = cdist(*rng);
  }
  for (std::size_t i = 0; i < coeff.size(); ++i) {
    levels_cpu[i] = static_cast<std::int32_t>(
        std::lround(coeff[i] * rp.quant_factor));
  }
  d_values.upload(coeff);
  cuda_ok(av2enc::cuda::quantize(d_values.get(), d_levels.get(), rp, nblocks,
                                 nullptr),
          "quantize real qf");
  d_levels.download(&levels_gpu);
  report(" qf=real", compare_exact_typed(levels_cpu, levels_gpu));
}

void test_dequant_extremes(std::mt19937* rng) {
  std::printf("\n[dequant]\n");
  const av2enc::cuda::TxParams p =
      *av2enc::cuda::make_tx_params(4, 4, true);  // dq_denom = 1
  std::vector<std::int32_t> levels;
  const std::int32_t specials[] = {
      -1000000, -100000, -31775, -31774, -10000, -1000, -1,     0,
      1,        1000,   10000,  31774,  31775,  100000, 1000000};
  for (std::int32_t v : specials) {
    levels.push_back(v);
  }
  std::uniform_int_distribution<std::int32_t> dist(-100000, 100000);
  for (int i = 0; i < 10000; ++i) {
    levels.push_back(dist(*rng));
  }
  const std::size_t n = 4 * 4;
  while (levels.size() % n != 0) {
    levels.push_back(0);
  }
  const int nblocks = static_cast<int>(levels.size() / n);
  std::vector<int> levels_int(levels.size());
  for (std::size_t i = 0; i < levels.size(); ++i) {
    levels_int[i] = levels[i];
  }
  std::vector<int> dequant_cpu;
  cpu::dequantize(levels_int, p.dq_denom, &dequant_cpu);

  DevBuf<std::int32_t> d_levels(levels.size());
  DevBuf<std::int32_t> d_dequant(levels.size());
  d_levels.upload(levels);
  cuda_ok(av2enc::cuda::dequantize(d_levels.get(), d_dequant.get(), p, nblocks,
                                   nullptr),
          "dequantize launch");
  std::vector<std::int32_t> dequant_gpu;
  d_dequant.download(&dequant_gpu);
  report(" levels", compare_exact_typed(to_i32(dequant_cpu), dequant_gpu));

  // dq_denom 2 and 4 (32x32 and 64x64-adjusted) as well.
  for (int denom : {2, 4}) {
    av2enc::cuda::TxParams q = p;
    q.dq_denom = denom;
    cpu::dequantize(levels_int, denom, &dequant_cpu);
    cuda_ok(av2enc::cuda::dequantize(d_levels.get(), d_dequant.get(), q,
                                     nblocks, nullptr),
            "dequantize launch denom");
    d_dequant.download(&dequant_gpu);
    char label[32];
    std::snprintf(label, sizeof(label), " denom=%d", denom);
    report(label, compare_exact_typed(to_i32(dequant_cpu), dequant_gpu));
  }
}

// ---------------------------------------------------------------------------
// 64x64 -> adjusted 32x32 edge path
// ---------------------------------------------------------------------------
void test_edge_64x64(std::mt19937* rng) {
  std::printf("\n[64x64 edge]\n");
  const av2enc::cuda::TxParams p = *av2enc::cuda::make_tx_params(64, 64, true);
  const cpu::RefTx ref = cpu::ref_tx(64, 64, true);
  check_params(p, ref, "64x64");

  struct PlaneCase {
    int w;
    int h;
  };
  const PlaneCase planes[] = {{128, 128}, {100, 70}};
  struct Block {
    int x0;
    int y0;
  };
  const Block blocks[] = {{0, 0}, {64, 0}, {0, 64}, {64, 64}, {32, 16}};
  const int preds[] = {0, 37, 128, 200, 255};

  Stat acc_build;
  Stat acc_ds;
  Stat acc_coeff;
  Stat acc_levels;
  Stat acc_dequant;
  Stat acc_inv;
  Stat acc_recon;
  Stat acc_vis;

  int cases = 0;
  for (const PlaneCase& pc : planes) {
    const int width = pc.w;
    const int height = pc.h;
    const int padded_w = (width + 63) & ~63;
    const int padded_h = (height + 63) & ~63;
    std::uniform_int_distribution<int> pdist(0, 255);
    std::vector<std::uint8_t> plane(static_cast<std::size_t>(width) * height);
    for (std::uint8_t& v : plane) {
      v = static_cast<std::uint8_t>(pdist(*rng));
    }
    DevBuf<std::uint8_t> d_src(plane.size());
    d_src.upload(plane);
    DevBuf<std::int16_t> d_residual32(32 * 32);
    DevBuf<std::int16_t> d_residual32b(32 * 32);
    DevBuf<double> d_coeff(32 * 32);
    DevBuf<std::int32_t> d_levels(32 * 32);
    DevBuf<std::int32_t> d_dequant(32 * 32);
    DevBuf<std::int16_t> d_resout(32 * 32);
    DevBuf<std::uint8_t> d_recon(static_cast<std::size_t>(padded_w) * padded_h);

    for (const Block& blk : blocks) {
      for (int pred : preds) {
        ++cases;
        // --- residual build (source + pred) ---
        const std::vector<int> r32_cpu =
            cpu::make_residual_64x64(plane, width, height, blk.x0, blk.y0,
                                     pred);
        std::vector<std::int16_t> r32_cpu16(r32_cpu.size());
        for (std::size_t i = 0; i < r32_cpu.size(); ++i) {
          r32_cpu16[i] = static_cast<std::int16_t>(r32_cpu[i]);
        }
        cuda_ok(av2enc::cuda::make_residual_64x64(
                    d_src.get(), width, width, height, blk.x0, blk.y0, pred,
                    d_residual32.get(), nullptr),
                "make_residual_64x64");
        std::vector<std::int16_t> r32_gpu;
        d_residual32.download(&r32_gpu);
        merge_stat(&acc_build, compare_exact_typed(r32_cpu16, r32_gpu));

        // --- full 64x64 residual, then the same 2x2 average ---
        std::vector<std::int16_t> r64_cpu(64 * 64, 0);
        for (int j = 0; j < 64; ++j) {
          for (int i = 0; i < 64; ++i) {
            r64_cpu[static_cast<std::size_t>(j) * 64 + i] =
                static_cast<std::int16_t>(
                    static_cast<int>(cpu::source_sample(plane, width, height,
                                                        blk.x0 + i,
                                                        blk.y0 + j)) -
                    pred);
          }
        }
        std::vector<int> r64ds_cpu = cpu::downsample_residual_64x64(r64_cpu);
        std::vector<std::int16_t> r64ds_cpu16(r64ds_cpu.size());
        for (std::size_t i = 0; i < r64ds_cpu.size(); ++i) {
          r64ds_cpu16[i] = static_cast<std::int16_t>(r64ds_cpu[i]);
        }
        DevBuf<std::int16_t> d_r64(r64_cpu.size());
        d_r64.upload(r64_cpu);
        cuda_ok(av2enc::cuda::downsample_residual_64x64(d_r64.get(),
                                                        d_residual32b.get(),
                                                        nullptr),
                "downsample_residual_64x64");
        std::vector<std::int16_t> r64ds_gpu;
        d_residual32b.download(&r64ds_gpu);
        merge_stat(&acc_ds, compare_exact_typed(r64ds_cpu16, r64ds_gpu));

        // --- transform stages with dq_denom = 4 ---
        std::vector<int> block_int(r32_cpu);
        std::vector<double> coeff_cpu;
        std::vector<int> levels_cpu;
        std::vector<int> dequant_cpu;
        std::vector<int> resid_cpu;
        cpu::forward_dct(block_int, 32, 32, &coeff_cpu);
        cpu::quantize(coeff_cpu, ref.quant_factor, &levels_cpu);
        cpu::dequantize(levels_cpu, ref.dq_denom, &dequant_cpu);
        cpu::inverse_dct(dequant_cpu, 32, 32, &resid_cpu);

        std::vector<std::int16_t> resid_cpu16(resid_cpu.size());
        for (std::size_t i = 0; i < resid_cpu.size(); ++i) {
          resid_cpu16[i] = static_cast<std::int16_t>(resid_cpu[i]);
        }

        d_residual32.upload(r32_cpu16);
        cuda_ok(av2enc::cuda::forward_dct(d_residual32.get(), d_coeff.get(), p,
                                          1, nullptr),
                "64x64 forward_dct");
        cuda_ok(av2enc::cuda::quantize(d_coeff.get(), d_levels.get(), p, 1,
                                       nullptr),
                "64x64 quantize");
        cuda_ok(av2enc::cuda::dequantize(d_levels.get(), d_dequant.get(), p, 1,
                                         nullptr),
                "64x64 dequantize");
        cuda_ok(av2enc::cuda::inverse_dct(d_dequant.get(), d_resout.get(), p, 1,
                                          nullptr),
                "64x64 inverse_dct");
        std::vector<double> coeff_gpu;
        std::vector<std::int32_t> levels_gpu;
        std::vector<std::int32_t> dequant_gpu;
        std::vector<std::int16_t> resid_gpu;
        d_coeff.download(&coeff_gpu);
        d_levels.download(&levels_gpu);
        d_dequant.download(&dequant_gpu);
        d_resout.download(&resid_gpu);
        merge_stat(&acc_coeff, compare_exact_typed(coeff_cpu, coeff_gpu));
        merge_stat(&acc_levels,
                   compare_exact_typed(to_i32(levels_cpu), levels_gpu));
        merge_stat(&acc_dequant,
                   compare_exact_typed(to_i32(dequant_cpu), dequant_gpu));
        merge_stat(&acc_inv, compare_exact_typed(resid_cpu16, resid_gpu));

        // --- reconstruction into the padded plane ---
        std::vector<std::uint8_t> recon_cpu(
            static_cast<std::size_t>(padded_w) * padded_h, 128);
        cpu::reconstruct_64x64(resid_cpu, pred, &recon_cpu, padded_w, blk.x0,
                               blk.y0);
        d_resout.upload(resid_cpu16);
        d_recon.upload(recon_cpu);
        cuda_ok(av2enc::cuda::reconstruct_64x64(
                    d_resout.get(), pred, d_recon.get(), padded_w, padded_w,
                    padded_h, blk.x0, blk.y0, nullptr),
                "reconstruct_64x64");
        std::vector<std::uint8_t> recon_gpu;
        d_recon.download(&recon_gpu);
        merge_stat(&acc_recon, compare_exact_typed(recon_cpu, recon_gpu));

        // --- clipped destination (visible bounds only) ---
        if (pred == 128) {
          std::vector<std::uint8_t> visible(
              static_cast<std::size_t>(width) * height, 128);
          for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
              visible[static_cast<std::size_t>(y) * width + x] =
                  recon_cpu[static_cast<std::size_t>(y) * padded_w + x];
            }
          }
          DevBuf<std::uint8_t> d_vis(visible.size());
          d_vis.upload(visible);
          cuda_ok(av2enc::cuda::reconstruct_64x64(
                      d_resout.get(), pred, d_vis.get(), width, width, height,
                      blk.x0, blk.y0, nullptr),
                  "reconstruct_64x64 visible");
          std::vector<std::uint8_t> vis_gpu;
          d_vis.download(&vis_gpu);
          merge_stat(&acc_vis, compare_exact_typed(visible, vis_gpu));
        }
      }
    }
  }
  std::printf(" cases=%d", cases);
  report(" build", acc_build);
  report(" ds", acc_ds);
  report(" coeff", acc_coeff);
  report(" levels", acc_levels);
  report(" dequant", acc_dequant);
  report(" inv", acc_inv);
  report(" recon", acc_recon);
  report(" vis", acc_vis);
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
enum class BenchMode { kForward, kInverse, kBoth };

double bench_pipeline(int w, int h, int nblocks, int iters, BenchMode mode) {
  const av2enc::cuda::TxParams p = *av2enc::cuda::make_tx_params(w, h, true);
  const std::size_t elems = static_cast<std::size_t>(nblocks) * w * h;
  DevBuf<std::int16_t> d_res(elems);
  DevBuf<double> d_coeff(elems);
  DevBuf<std::int32_t> d_levels(elems);
  DevBuf<std::int32_t> d_dequant(elems);
  DevBuf<std::int16_t> d_out(elems);
  std::vector<std::int16_t> zero(elems, 0);
  d_res.upload(zero);

  auto run = [&]() {
    if (mode != BenchMode::kInverse) {
      cuda_ok(av2enc::cuda::forward_transform(d_res.get(), d_coeff.get(),
                                              d_levels.get(), p, nblocks,
                                              nullptr),
              "bench forward");
    }
    if (mode != BenchMode::kForward) {
      cuda_ok(av2enc::cuda::inverse_transform(d_levels.get(), d_dequant.get(),
                                              d_out.get(), p, nblocks,
                                              nullptr),
              "bench inverse");
    }
  };
  for (int i = 0; i < 3; ++i) {
    run();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, nullptr);
  for (int i = 0; i < iters; ++i) {
    run();
  }
  cudaEventRecord(stop, nullptr);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return static_cast<double>(ms) / iters;
}

void bench() {
  std::printf("\n[timing]\n");
  const int iters = 10;
  for (int size : {32, 16}) {
    const int nblocks = size == 32 ? 8192 : 32768;
    const double ms_fwd =
        bench_pipeline(size, size, nblocks, iters, BenchMode::kForward);
    const double ms_inv =
        bench_pipeline(size, size, nblocks, iters, BenchMode::kInverse);
    const double ms_both =
        bench_pipeline(size, size, nblocks, iters, BenchMode::kBoth);
    const double pels = static_cast<double>(nblocks) * size * size;
    std::printf(
        "  %dx%d %d blocks: fwd(dct+quant)=%.3f ms (%.3f us/block) | "
        "inv(dequant+idct)=%.3f ms (%.3f us/block) | both=%.3f ms "
        "(%.3f us/block, %.1f Mblocks/s, %.2f Gpel/s)\n",
        size, size, nblocks, ms_fwd, ms_fwd * 1000.0 / nblocks, ms_inv,
        ms_inv * 1000.0 / nblocks, ms_both, ms_both * 1000.0 / nblocks,
        static_cast<double>(nblocks) / (ms_both * 1.0e-3) / 1.0e6,
        pels / (ms_both * 1.0e-3) / 1.0e9);
  }

  // Launch-bound single-block latency.
  {
    const av2enc::cuda::TxParams p = *av2enc::cuda::make_tx_params(32, 32, true);
    DevBuf<std::int16_t> d_res(32 * 32);
    DevBuf<double> d_coeff(32 * 32);
    DevBuf<std::int32_t> d_levels(32 * 32);
    DevBuf<std::int32_t> d_dequant(32 * 32);
    DevBuf<std::int16_t> d_out(32 * 32);
    std::vector<std::int16_t> zero(32 * 32, 0);
    d_res.upload(zero);
    for (int i = 0; i < 100; ++i) {
      cuda_ok(av2enc::cuda::forward_transform(d_res.get(), d_coeff.get(),
                                              d_levels.get(), p, 1, nullptr),
              "bench single");
    }
    cudaDeviceSynchronize();
    const int single_iters = 2000;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    for (int i = 0; i < single_iters; ++i) {
      cuda_ok(av2enc::cuda::forward_transform(d_res.get(), d_coeff.get(),
                                              d_levels.get(), p, 1, nullptr),
              "bench single");
      cuda_ok(av2enc::cuda::inverse_transform(d_levels.get(), d_dequant.get(),
                                              d_out.get(), p, 1, nullptr),
              "bench single");
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::printf("  1x 32x32 pipeline: %.3f us/iteration (%d iterations)\n",
                static_cast<double>(ms) * 1000.0 / single_iters, single_iters);
  }

  // 64x64 edge helper is one launch per block.
  {
    std::vector<std::uint8_t> plane(128 * 128, 128);
    DevBuf<std::uint8_t> d_src(plane.size());
    DevBuf<std::int16_t> d_res(32 * 32);
    d_src.upload(plane);
    for (int i = 0; i < 100; ++i) {
      cuda_ok(av2enc::cuda::make_residual_64x64(d_src.get(), 128, 128, 128, 0,
                                                0, 128, d_res.get(), nullptr),
              "bench edge");
    }
    cudaDeviceSynchronize();
    const int edge_iters = 2000;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    for (int i = 0; i < edge_iters; ++i) {
      cuda_ok(av2enc::cuda::make_residual_64x64(d_src.get(), 128, 128, 128, 0,
                                                0, 128, d_res.get(), nullptr),
              "bench edge");
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::printf("  64x64 edge residual: %.3f us/block (%d iterations)\n",
                static_cast<double>(ms) * 1000.0 / edge_iters, edge_iters);
  }
}

}  // namespace

int main() {
  int device = 0;
  cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  cuda_ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  std::printf("av2enc CUDA transform test\n");
  std::printf("device: %s (sm_%d%d, %d SMs, %.1f GiB)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount,
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  int driver_version = 0;
  cuda_ok(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  std::printf("CUDA runtime %d.%d, driver %d.%d\n", CUDART_VERSION / 1000,
              (CUDART_VERSION % 1000) / 10, driver_version / 1000,
              (driver_version % 1000) / 10);

  std::mt19937 rng(0x5EED1234U);

  test_params();

  struct Size {
    int w;
    int h;
  };
  const Size sizes[] = {{32, 32}, {16, 16}, {32, 16}, {16, 8}, {32, 8},
                        {16, 4},  {16, 32}, {8, 16}, {8, 8},  {4, 4},
                        {4, 16}};
  for (const Size& s : sizes) {
    test_size(s.w, s.h, &rng);
  }

  test_rounding(&rng);
  test_dequant_extremes(&rng);
  test_edge_64x64(&rng);
  bench();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures == 0) {
    std::printf("RESULT: PASS\n");
    return 0;
  }
  std::printf("RESULT: FAIL\n");
  return 1;
}
