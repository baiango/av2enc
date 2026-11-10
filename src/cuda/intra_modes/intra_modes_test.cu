// AV2 v1.0.0 full intra prediction: CPU-reference bit-exactness and timing.
//
// The CPU reference below is an independent scalar transcription of the same
// spec sections (7.13.2.1, .2, .7, .8, .10, .13, .14-.18) used by the kernels.
// It shares only the normative constant tables (mode_to_angle,
// Dr_Intra_Derivative, Dr_Interp_Filter, Intra_Edge_Kernel, Div_Lut) with the
// GPU path; every arithmetic step is written out again so a formula error in
// the kernel is caught rather than mirrored.
//
// Build/run: ./build_intra_modes_test.sh
//
// Coverage:
//   * all 13 intra modes, angle deltas -3..3, all square/rectangular block
//     sizes 4..64, filter-type combinations, luma (IDIF) and chroma
//   * interior, top-row, left-column, top-left, right-edge and bottom-edge
//     (partial) block positions
//   * synthetic 640x360 bottom edge, 1920x1080 bottom/right edges
//   * batched API vs the per-block API / CPU reference
//   * timing: launch-bound single blocks, batched 640x360 and 1080p grids

#include "intra_modes.h"

#ifdef AV2ENC_CPU_INTRA_PARITY
#include "encoder/intra_modes/intra_modes.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace im = av2enc::cuda::intra_modes;

// ---------------------------------------------------------------------------
// Error handling / bookkeeping
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

Stage g_pred{"pred"};
Stage g_resid{"residual"};
Stage g_batch_pred{"batch_pred"};
Stage g_batch_resid{"batch_residual"};

std::string g_context;
int g_printed = 0;
constexpr int kMaxPrinted = 12;

// ---------------------------------------------------------------------------
// CPU reference: scalar transcription of the normatives sections
// ---------------------------------------------------------------------------

int ref_clip3(int lo, int hi, int v) {
  return v < lo ? lo : (v > hi ? hi : v);
}
int ref_clip1(int v) { return ref_clip3(0, 255, v); }
int ref_round2(int v, int n) {
  if (n <= 0) return v;
  return (v + (1 << (n - 1))) >> n;
}
int ref_floor_log2(int v) {
  int r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

void ref_resolve_divisor(int d, int* shift, int* factor) {
  const int abs_d = d < 0 ? -d : d;
  const int n = ref_floor_log2(abs_d);
  const int e = abs_d - (1 << n);
  const int f =
      (n > 7) ? ref_round2(e, n - 7) : (e << (7 - n));
  *shift = n + 9;
  *factor = (d < 0) ? -im::div_lut(f) : im::div_lut(f);
}

int ref_approx_divide(int num, int den) {
  int shift = 0;
  int factor = 0;
  ref_resolve_divisor(den, &shift, &factor);
  return ref_round2(num * factor, shift);
}

int ref_wide_angle_map(int pAngle, int w, int h) {
  if ((h == 2 * w && pAngle < 61) || (h == 4 * w && pAngle < 73) ||
      (h == 8 * w && pAngle < 82) || (h == 16 * w && pAngle < 86)) {
    return 180 + pAngle;
  }
  if ((w == 2 * h && pAngle > 270 - 61) || (w == 4 * h && pAngle > 270 - 73) ||
      (w == 8 * h && pAngle > 270 - 82) ||
      (w == 16 * h && pAngle > 270 - 86)) {
    return pAngle - 180;
  }
  return pAngle;
}

int ref_pangle(int mode, int delta, int w, int h) {
  if (mode < 1 || mode > 8) return im::mode_to_angle(mode);
  return ref_wide_angle_map(im::mode_to_angle(mode) + delta * 3, w, h);
}

int ref_edge_strength(int w, int h, int filter_type, int delta) {
  const int d = delta < 0 ? -delta : delta;
  const int blk = w + h;
  int strength = 0;
  if (filter_type == 0) {
    if (blk <= 8) {
      if (d >= 56) strength = 1;
    } else if (blk <= 12) {
      if (d >= 40) strength = 1;
    } else if (blk <= 16) {
      if (d >= 40) strength = 1;
    } else if (blk <= 24) {
      if (d >= 8) strength = 1;
      if (d >= 16) strength = 2;
      if (d >= 32) strength = 3;
    } else if (blk <= 32) {
      strength = 1;
      if (d >= 4) strength = 2;
      if (d >= 32) strength = 3;
    } else {
      strength = 3;
    }
  } else {
    if (blk <= 8) {
      if (d >= 40) strength = 1;
      if (d >= 64) strength = 2;
    } else if (blk <= 16) {
      if (d >= 20) strength = 1;
      if (d >= 48) strength = 2;
    } else if (blk <= 24) {
      if (d >= 4) strength = 3;
    } else {
      strength = 3;
    }
  }
  return strength;
}

struct RefPred {
  std::vector<std::uint8_t> pred;
  std::vector<std::int16_t> residual;
};

RefPred ref_predict(const im::IntraBlock& b) {
  const int w = b.width;
  const int h = b.height;
  const int need = w + h;
  const int cap = need + 8;
  const int kHead = 3;
  std::vector<int> ab_storage(static_cast<std::size_t>(cap));
  std::vector<int> lf_storage(static_cast<std::size_t>(cap));
  int* A = ab_storage.data() + kHead;
  int* L = lf_storage.data() + kHead;

  const std::size_t stride = static_cast<std::size_t>(b.recon_stride);

  // §7.13.2.1 AboveRow / LeftCol.
  for (int i = 0; i < need; ++i) {
    if (b.have_above == 0) {
      if (b.have_left != 0) {
        A[i] = static_cast<int>(
            b.recon_plane[static_cast<std::size_t>(b.y0) * stride +
                          static_cast<std::size_t>(b.x0 - 1)]);
      } else {
        A[i] = 127;
      }
    } else {
      int limit = b.max_x;
      const int num4 =
          (b.plane > 0 && w > 32) ? 0 : b.num4_above_right;
      const int ext = b.x0 + w + 4 * num4 - 1;
      if (ext < limit) limit = ext;
      int x = b.x0 + i;
      if (x > limit) x = limit;
      A[i] = static_cast<int>(
          b.recon_plane[static_cast<std::size_t>(b.y0 - 1) * stride +
                        static_cast<std::size_t>(x)]);
    }
    if (b.have_left == 0) {
      if (b.have_above != 0) {
        L[i] = static_cast<int>(
            b.recon_plane[static_cast<std::size_t>(b.y0 - 1) * stride +
                          static_cast<std::size_t>(b.x0)]);
      } else {
        L[i] = 129;
      }
    } else {
      int limit = b.max_y;
      const int num4 =
          (b.plane > 0 && h > 32) ? 0 : b.num4_below_left;
      const int ext = b.y0 + h + 4 * num4 - 1;
      if (ext < limit) limit = ext;
      int y = b.y0 + i;
      if (y > limit) y = limit;
      L[i] = static_cast<int>(
          b.recon_plane[static_cast<std::size_t>(y) * stride +
                        static_cast<std::size_t>(b.x0 - 1)]);
    }
  }

  // Corner sample AboveRow[-1] / LeftCol[-1].
  int corner = 128;
  if (b.have_above != 0 && b.have_left != 0) {
    corner = static_cast<int>(
        b.recon_plane[static_cast<std::size_t>(b.y0 - 1) * stride +
                      static_cast<std::size_t>(b.x0 - 1)]);
  } else if (b.have_above != 0) {
    corner = A[0];
  } else if (b.have_left != 0) {
    corner = L[0];
  }
  A[-1] = corner;
  L[-1] = corner;

  const int pAngle = ref_pangle(static_cast<int>(b.mode), b.angle_delta, w, h);

  // §7.13.2.14-18 edge filtering.
  if ((static_cast<int>(b.mode) >= 1 && static_cast<int>(b.mode) <= 8) &&
      b.enable_edge_filter != 0 && pAngle != 90 && pAngle != 180) {
    const int filter_type =
        ((b.have_above != 0 && b.filter_type_above != 0) ||
         (b.have_left != 0 && b.filter_type_left != 0))
            ? 1
            : 0;
    const bool need_right = pAngle < 90;
    const bool need_bottom = pAngle > 180;
    if (pAngle > 90 && pAngle < 180 && (w + h) >= 24) {
      const int c = ref_round2(L[0] * 5 + A[-1] * 6 + A[0] * 5, 4);
      A[-1] = c;
      L[-1] = c;
    }
    if (b.have_above != 0) {
      const int strength =
          ref_edge_strength(w, h, filter_type, pAngle - 90);
      const int sz = std::min(w, b.max_x - b.x0 + 1) + (need_right ? h : 0) + 1;
      if (strength > 0 && sz > 1) {
        std::vector<int> edge(static_cast<std::size_t>(sz));
        for (int i = 0; i < sz; ++i) edge[static_cast<std::size_t>(i)] = A[i - 1];
        for (int i = 1; i < sz; ++i) {
          int s = 0;
          for (int j = 0; j < 5; ++j) {
            const int k = ref_clip3(0, sz - 1, i - 2 + j);
            s += im::intra_edge_kernel(strength - 1, j) *
                 edge[static_cast<std::size_t>(k)];
          }
          A[i - 1] = (s + 8) >> 4;
        }
      }
    }
    if (b.have_left != 0) {
      const int strength =
          ref_edge_strength(w, h, filter_type, pAngle - 180);
      const int sz = std::min(h, b.max_y - b.y0 + 1) + (need_bottom ? w : 0) + 1;
      if (strength > 0 && sz > 1) {
        std::vector<int> edge(static_cast<std::size_t>(sz));
        for (int i = 0; i < sz; ++i) edge[static_cast<std::size_t>(i)] = L[i - 1];
        for (int i = 1; i < sz; ++i) {
          int s = 0;
          for (int j = 0; j < 5; ++j) {
            const int k = ref_clip3(0, sz - 1, i - 2 + j);
            s += im::intra_edge_kernel(strength - 1, j) *
                 edge[static_cast<std::size_t>(k)];
          }
          L[i - 1] = (s + 8) >> 4;
        }
      }
    }
  }

  // §7.13.2.8 edge extension (MrlIndex = 0).
  const bool directional = (static_cast<int>(b.mode) >= 1 &&
                            static_cast<int>(b.mode) <= 8);
  if (directional) {
    const int max_base = w + h - 1;
    if (pAngle > 90 && pAngle < 180) {
      L[h] = L[h - 1];
      A[w] = A[w - 1];
      L[h + 1] = L[h - 1];
      A[w + 1] = A[w - 1];
    } else {
      L[max_base + 1] = L[max_base];
      A[max_base + 1] = A[max_base];
      L[max_base + 2] = L[max_base];
      A[max_base + 2] = A[max_base];
    }
    L[-2] = L[-1];
    A[-2] = A[-1];
  }

  // DC (§7.13.2.10).
  int dc = 128;
  if (b.mode == im::kDcPred) {
    if (b.have_left != 0 && b.have_above != 0) {
      int sum = 0;
      for (int k = 0; k < h; ++k) sum += L[k];
      for (int k = 0; k < w; ++k) sum += A[k];
      dc = ref_clip1(ref_approx_divide(sum, w + h));
    } else if (b.have_left != 0) {
      int sum = 0;
      for (int k = 0; k < h; ++k) sum += L[k];
      dc = ref_round2(sum, ref_floor_log2(h));
    } else if (b.have_above != 0) {
      int sum = 0;
      for (int k = 0; k < w; ++k) sum += A[k];
      dc = ref_round2(sum, ref_floor_log2(w));
    }
  }

  const int log2w = ref_floor_log2(w);
  const int log2h = ref_floor_log2(h);
  const int enable_idif = (b.plane == 0) ? 1 : 0;

  RefPred out;
  out.pred.resize(static_cast<std::size_t>(w * h));
  out.residual.resize(static_cast<std::size_t>(w * h));

  for (int py = 0; py < h; ++py) {
    for (int px = 0; px < w; ++px) {
      const int mode = static_cast<int>(b.mode);
      int value = 0;
      if (mode == im::kDcPred) {
        value = dc;
      } else if (mode == im::kPaethPred) {
        const int base = A[px] + L[py] - A[-1];
        const int p_left = std::abs(base - L[py]);
        const int p_top = std::abs(base - A[px]);
        const int p_top_left = std::abs(base - A[-1]);
        if (p_left <= p_top && p_left <= p_top_left) {
          value = L[py];
        } else if (p_top <= p_top_left) {
          value = A[px];
        } else {
          value = A[-1];
        }
      } else if (mode == im::kSmoothPred || mode == im::kSmoothVPred ||
                 mode == im::kSmoothHPred) {
        const int bl = L[h];
        const int tr = A[w];
        const int scale = ref_round2(log2w + log2h - 4, 2);
        const int s_top = std::min(6, (py << 1) >> scale);
        const int s_left = std::min(6, (px << 1) >> scale);
        const int top = A[px];
        const int lft = L[py];
        const int pred_h = tr + ref_round2((lft - tr) * (w - 1 - px), log2w);
        const int pred_v = bl + ref_round2((top - bl) * (h - 1 - py), log2h);
        const int pred_h2 = pred_h + ref_round2((lft - pred_h) * (32 >> s_left), 6);
        const int pred_v2 = pred_v + ref_round2((top - pred_v) * (32 >> s_top), 6);
        if (mode == im::kSmoothHPred) {
          value = pred_h2;
        } else if (mode == im::kSmoothVPred) {
          value = pred_v2;
        } else {
          value = ref_round2(pred_v2 + pred_h2, 1);
        }
      } else if (pAngle < 90) {
        const int dx = im::dr_intra_derivative(pAngle);
        const int idx = (py + 1) * dx;
        const int base = (idx >> 6) + px;
        const int shift = (idx >> 1) & 0x1F;
        const int max_base_x = w + h - 1;
        if (base < max_base_x + enable_idif) {
          if (enable_idif) {
            int s = 0;
            for (int t = 0; t < 4; ++t) {
              s += im::dr_interp_filter(shift, t) * A[base + t - 1];
            }
            value = ref_clip1(ref_round2(s, 7));
          } else {
            value = ref_round2(A[base] * (32 - shift) + A[base + 1] * shift, 5);
          }
        } else {
          value = A[max_base_x];
        }
      } else if (pAngle > 90 && pAngle < 180) {
        const int dx = im::dr_intra_derivative(180 - pAngle);
        const int dy = im::dr_intra_derivative(pAngle - 90);
        const int idx = (px << 6) - (py + 1) * dx;
        const int base = idx >> 6;
        if (base >= -1) {
          const int shift = (idx >> 1) & 0x1F;
          if (enable_idif) {
            int s = 0;
            for (int t = 0; t < 4; ++t) {
              s += im::dr_interp_filter(shift, t) * A[base + t - 1];
            }
            value = ref_clip1(ref_round2(s, 7));
          } else {
            value = ref_round2(A[base] * (32 - shift) + A[base + 1] * shift, 5);
          }
        } else {
          const int idx2 = (py << 6) - (px + 1) * dy;
          const int base2 = idx2 >> 6;
          const int shift2 = (idx2 >> 1) & 0x1F;
          if (enable_idif) {
            int s = 0;
            for (int t = 0; t < 4; ++t) {
              s += im::dr_interp_filter(shift2, t) * L[base2 + t - 1];
            }
            value = ref_clip1(ref_round2(s, 7));
          } else {
            value =
                ref_round2(L[base2] * (32 - shift2) + L[base2 + 1] * shift2, 5);
          }
        }
      } else if (pAngle > 180) {
        const int dy = im::dr_intra_derivative(270 - pAngle);
        const int idx = (px + 1) * dy;
        const int base = (idx >> 6) + py;
        const int shift = (idx >> 1) & 0x1F;
        const int max_base_y = w + h - 1;
        if (base < max_base_y + enable_idif) {
          if (enable_idif) {
            int s = 0;
            for (int t = 0; t < 4; ++t) {
              s += im::dr_interp_filter(shift, t) * L[base + t - 1];
            }
            value = ref_clip1(ref_round2(s, 7));
          } else {
            value = ref_round2(L[base] * (32 - shift) + L[base + 1] * shift, 5);
          }
        } else {
          value = L[max_base_y];
        }
      } else if (pAngle == 90) {
        value = A[px];
      } else {
        value = L[py];
      }

      const int i = py * w + px;
      out.pred[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(value);
      const int sx = ref_clip3(0, b.source_max_x, b.x0 + px);
      const int sy = ref_clip3(0, b.source_max_y, b.y0 + py);
      const int sample = static_cast<int>(
          b.source_plane[static_cast<std::size_t>(sy) *
                             static_cast<std::size_t>(b.source_stride) +
                         static_cast<std::size_t>(sx)]);
      out.residual[static_cast<std::size_t>(i)] =
          static_cast<std::int16_t>(sample - value);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Test planes
// ---------------------------------------------------------------------------

struct Plane {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> data;
};

std::uint32_t g_rng = 0x12345678u;
std::uint32_t next_rand() {
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng;
}

Plane make_plane(int w, int h, int pattern, std::uint32_t seed) {
  Plane p;
  p.w = w;
  p.h = h;
  p.data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  g_rng = seed;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 0;
      switch (pattern) {
        case 0:
          v = (x * 3 + y * 5) & 255;
          break;
        case 1:
          v = static_cast<int>(next_rand() >> 24);
          break;
        case 2:
          v = 128 + ((x * x / 17 + y * y / 23) % 64) - 32;
          break;
        case 3: {
          const int c = ((x >> 3) + (y >> 3)) & 1;
          v = c ? 220 : 35;
          v += static_cast<int>(next_rand() % 21) - 10;
          break;
        }
        case 4:
          v = (x * 2 + ((y * 7) & 63)) & 255;
          v = x > w / 2 ? 255 - v : v;
          break;
        default:
          v = (x ^ y) & 255;
          break;
      }
      p.data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
             static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(ref_clip1(v));
    }
  }
  return p;
}

// ---------------------------------------------------------------------------
// Device plane / runner
// ---------------------------------------------------------------------------

class DevicePlane {
 public:
  explicit DevicePlane(const Plane& p) : w_(p.w), h_(p.h) {
    const std::size_t bytes = p.data.size();
    CUDA_CHECK(cudaMalloc(&d_, bytes));
    CUDA_CHECK(cudaMemcpy(d_, p.data.data(), bytes, cudaMemcpyHostToDevice));
  }
  ~DevicePlane() {
    if (d_ != nullptr) cudaFree(d_);
  }
  DevicePlane(const DevicePlane&) = delete;
  DevicePlane& operator=(const DevicePlane&) = delete;

  std::uint8_t* get() { return d_; }
  const std::uint8_t* get() const { return d_; }
  int w() const { return w_; }
  int h() const { return h_; }
  int stride() const { return w_; }

 private:
  std::uint8_t* d_ = nullptr;
  int w_ = 0;
  int h_ = 0;
};

struct Case {
  int w = 32;
  int h = 32;
  int x0 = 0;
  int y0 = 0;
  int have_left = -1;   // -1: derive from x0
  int have_above = -1;  // -1: derive from y0
  int mode = im::kDcPred;
  int delta = 0;
  int ft_above = 0;
  int ft_left = 0;
  int plane = 0;
  int enable_edge_filter = 1;
  int num4_ar = -1;  // -1: derive from remaining plane extent
  int num4_bl = -1;
  const char* tag = "";
};

void note_mismatch(const Stage& stage, const char* what, long long index,
                   long long expected, long long actual) {
  if (g_printed >= kMaxPrinted) return;
  std::printf("  MISMATCH [%s] %s %s: index %lld expected %lld got %lld\n",
              stage.name, g_context.c_str(), what, index, expected, actual);
  ++g_printed;
}

void compare_pred(Stage& stage, const RefPred& ref, const std::uint8_t* gpu,
                  int count) {
  long long mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < count; ++i) {
    const int e = static_cast<int>(ref.pred[static_cast<std::size_t>(i)]);
    const int a = static_cast<int>(gpu[i]);
    if (e != a) {
      ++mismatches;
      max_diff = std::max(max_diff, std::abs(e - a));
      note_mismatch(stage, "pred", i, e, a);
    }
  }
  stage.samples += count;
  stage.mismatches += mismatches;
  stage.max_abs_diff = std::max(stage.max_abs_diff, max_diff);
}

void compare_resid(Stage& stage, const RefPred& ref, const std::int16_t* gpu,
                   int count) {
  long long mismatches = 0;
  int max_diff = 0;
  for (int i = 0; i < count; ++i) {
    const int e = static_cast<int>(ref.residual[static_cast<std::size_t>(i)]);
    const int a = static_cast<int>(gpu[i]);
    if (e != a) {
      ++mismatches;
      max_diff = std::max(max_diff, std::abs(e - a));
      note_mismatch(stage, "residual", i, e, a);
    }
  }
  stage.samples += count;
  stage.mismatches += mismatches;
  stage.max_abs_diff = std::max(stage.max_abs_diff, max_diff);
}

im::IntraBlock make_block(const Case& c, const std::uint8_t* recon,
                          int recon_w, int recon_h, const std::uint8_t* source,
                          int source_w, int source_h) {
  im::IntraBlock b;
  b.recon_plane = recon;
  b.recon_stride = recon_w;
  b.x0 = c.x0;
  b.y0 = c.y0;
  b.width = c.w;
  b.height = c.h;
  b.max_x = recon_w - 1;
  b.max_y = recon_h - 1;
  b.have_left = c.have_left >= 0 ? c.have_left : (c.x0 > 0 ? 1 : 0);
  b.have_above = c.have_above >= 0 ? c.have_above : (c.y0 > 0 ? 1 : 0);
  b.num4_above_right =
      c.num4_ar >= 0
          ? c.num4_ar
          : std::max(0, (b.max_x - (c.x0 + c.w) + 1) / 4);
  b.num4_below_left =
      c.num4_bl >= 0
          ? c.num4_bl
          : std::max(0, (b.max_y - (c.y0 + c.h) + 1) / 4);
  b.plane = c.plane;
  b.mode = static_cast<im::IntraMode>(c.mode);
  b.angle_delta = c.delta;
  b.filter_type_above = c.ft_above;
  b.filter_type_left = c.ft_left;
  b.enable_edge_filter = c.enable_edge_filter;
  b.source_plane = source;
  b.source_stride = source_w;
  b.source_max_x = source_w - 1;
  b.source_max_y = source_h - 1;
  return b;
}

struct RunStats {
  long long cases = 0;
  long long samples = 0;
};

class Runner {
 public:
  Runner(const Plane& host_recon, const Plane& host_src,
         const DevicePlane& recon, const DevicePlane& src)
      : host_recon_(host_recon),
        host_src_(host_src),
        recon_(recon),
        src_(src) {
    // Largest supported block is 64x64.
    CUDA_CHECK(cudaMalloc(&d_pred_, 64u * 64u));
    CUDA_CHECK(cudaMalloc(&d_resid_, 64u * 64u * sizeof(std::int16_t)));
    pred_.resize(64u * 64u);
    resid_.resize(64u * 64u);
  }
  ~Runner() {
    if (d_pred_ != nullptr) cudaFree(d_pred_);
    if (d_resid_ != nullptr) cudaFree(d_resid_);
  }
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  void run(const Case& c, bool require_residual = true) {
    const int n = c.w * c.h;
    const im::IntraBlock b = make_block(c, recon_.get(), recon_.w(),
                                       recon_.h(), src_.get(), src_.w(),
                                       src_.h());
    const im::IntraBlock host_b =
        make_block(c, host_recon_.data.data(), host_recon_.w, host_recon_.h,
                   host_src_.data.data(), host_src_.w, host_src_.h);
    const RefPred ref = ref_predict(host_b);

    g_context = std::string(c.tag) + " " + std::to_string(c.w) + "x" +
                std::to_string(c.h) + " @(" + std::to_string(c.x0) + "," +
                std::to_string(c.y0) + ") mode=" + std::to_string(c.mode) +
                " d=" + std::to_string(c.delta) + " ft=" +
                std::to_string(c.ft_above) + std::to_string(c.ft_left) +
                " plane=" + std::to_string(c.plane) +
                " ef=" + std::to_string(c.enable_edge_filter);

    if (require_residual) {
      CUDA_CHECK(im::intra_predict_residual(b, d_pred_, d_resid_));
      CUDA_CHECK(cudaMemcpy(pred_.data(), d_pred_,
                            static_cast<std::size_t>(n), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(resid_.data(), d_resid_,
                            static_cast<std::size_t>(n) * sizeof(std::int16_t),
                            cudaMemcpyDeviceToHost));
    } else {
      CUDA_CHECK(im::intra_predict(b, d_pred_));
      CUDA_CHECK(cudaMemcpy(pred_.data(), d_pred_,
                            static_cast<std::size_t>(n), cudaMemcpyDeviceToHost));
    }
    compare_pred(g_pred, ref, pred_.data(), n);
    if (require_residual) {
      compare_resid(g_resid, ref, resid_.data(), n);
    }
    stats_.cases++;
    stats_.samples += n;
  }

  // Batch path: every case becomes one device job; jobs are independent.
  void run_batch(const std::vector<Case>& cases, const char* tag) {
    if (cases.empty()) return;
    std::vector<im::IntraJob> jobs(cases.size());
    std::vector<std::uint8_t> pred_host;
    std::vector<std::int16_t> resid_host;
    std::vector<std::size_t> offsets(cases.size());
    std::size_t total = 0;
    for (std::size_t k = 0; k < cases.size(); ++k) {
      offsets[k] = total;
      total += static_cast<std::size_t>(cases[k].w) *
               static_cast<std::size_t>(cases[k].h);
    }
    pred_host.resize(total);
    resid_host.resize(total);
    std::uint8_t* d_pred = nullptr;
    std::int16_t* d_resid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pred, total * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_resid, total * sizeof(std::int16_t)));

    std::vector<RefPred> refs(cases.size());
    for (std::size_t k = 0; k < cases.size(); ++k) {
      im::IntraJob& job = jobs[k];
      job.block = make_block(cases[k], recon_.get(), recon_.w(), recon_.h(),
                             src_.get(), src_.w(), src_.h());
      job.pred = d_pred + offsets[k];
      job.residual = d_resid + offsets[k];
      refs[k] = ref_predict(make_block(cases[k], host_recon_.data.data(),
                                       host_recon_.w, host_recon_.h,
                                       host_src_.data.data(), host_src_.w,
                                       host_src_.h));
    }

    im::IntraJob* d_jobs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_jobs, jobs.size() * sizeof(im::IntraJob)));
    CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(),
                          jobs.size() * sizeof(im::IntraJob),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(im::intra_predict_batch(d_jobs, static_cast<int>(jobs.size())));
    CUDA_CHECK(cudaMemcpy(pred_host.data(), d_pred, total, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(resid_host.data(), d_resid,
                          total * sizeof(std::int16_t),
                          cudaMemcpyDeviceToHost));

    for (std::size_t k = 0; k < cases.size(); ++k) {
      const int n = cases[k].w * cases[k].h;
      g_context = std::string(tag) + " batch job " + std::to_string(k) + " " +
                  cases[k].tag + " mode=" + std::to_string(cases[k].mode) +
                  " d=" + std::to_string(cases[k].delta) + " @(" +
                  std::to_string(cases[k].x0) + "," +
                  std::to_string(cases[k].y0) + ")";
      compare_pred(g_batch_pred, refs[k], pred_host.data() + offsets[k], n);
      compare_resid(g_batch_resid, refs[k], resid_host.data() + offsets[k], n);
    }

    CUDA_CHECK(cudaFree(d_jobs));
    CUDA_CHECK(cudaFree(d_pred));
    CUDA_CHECK(cudaFree(d_resid));
    stats_.cases += static_cast<long long>(cases.size());
  }

  const RunStats& stats() const { return stats_; }

 private:
  const Plane& host_recon_;
  const Plane& host_src_;
  const DevicePlane& recon_;
  const DevicePlane& src_;
  std::uint8_t* d_pred_ = nullptr;
  std::int16_t* d_resid_ = nullptr;
  std::vector<std::uint8_t> pred_;
  std::vector<std::int16_t> resid_;
  RunStats stats_{};
};

// ---------------------------------------------------------------------------
// Case generators
// ---------------------------------------------------------------------------

struct Size {
  int w;
  int h;
};

const std::vector<Size>& all_sizes() {
  static const std::vector<Size> sizes = {
      {4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64}, {64, 32},
      {32, 64}, {64, 16}, {16, 64}, {8, 16},  {16, 8},  {32, 8},
      {8, 32},
  };
  return sizes;
}

// Positions inside a plane; entries may make the block extend past the right /
// bottom edge on purpose (partial blocks).
std::vector<std::pair<int, int>> positions_for(int plane_w, int plane_h,
                                               int bw, int bh) {
  std::vector<std::pair<int, int>> out;
  out.emplace_back(0, 0);                      // no neighbours
  out.emplace_back(0, std::min(bh, plane_h - 1));   // above only
  out.emplace_back(std::min(bw, plane_w - 1), 0);   // left only
  out.emplace_back(4, 4);                      // interior (all neighbours)
  out.emplace_back(plane_w / 2, plane_h / 2);
  out.emplace_back(std::max(0, plane_w - bw), std::max(0, plane_h - bh));
  out.emplace_back(std::max(0, plane_w - bw / 2),
                   std::max(0, plane_h - bh / 2));  // partial right/bottom
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

namespace {

void correctness_full_matrix() {
  std::printf("\n-- full mode/size/position matrix (CPU vs GPU) --\n");
  const Plane recon = make_plane(160, 144, 2, 0x1234u);
  const Plane source = make_plane(160, 144, 0, 0x9876u);
  const DevicePlane d_recon(recon);
  const DevicePlane d_source(source);
  Runner runner(recon, source, d_recon, d_source);

  const int angle_deltas[7] = {-3, -2, -1, 0, 1, 2, 3};
  for (const Size& s : all_sizes()) {
    for (const auto& pos : positions_for(160, 144, s.w, s.h)) {
      Case c;
      c.w = s.w;
      c.h = s.h;
      c.x0 = pos.first;
      c.y0 = pos.second;
      c.tag = "matrix";
      // Non-directional predictors.
      for (int mode : {im::kDcPred, im::kSmoothPred, im::kSmoothVPred,
                       im::kSmoothHPred, im::kPaethPred}) {
        c.mode = mode;
        c.delta = 0;
        c.ft_above = 0;
        c.ft_left = 0;
        c.plane = 0;
        c.enable_edge_filter = 1;
        runner.run(c);
      }
      // Directional predictors: all deltas, filter types, IDIF/chroma, and
      // edge filter on/off.
      for (int mode = im::kVpred; mode <= im::kD67Pred; ++mode) {
        for (int d : angle_deltas) {
          for (int ft = 0; ft < 4; ++ft) {
            for (int plane = 0; plane < 2; ++plane) {
              c.mode = mode;
              c.delta = d;
              c.ft_above = ft & 1;
              c.ft_left = (ft >> 1) & 1;
              c.plane = plane;
              c.enable_edge_filter = 1;
              runner.run(c);
            }
          }
          // Filter-off path once per (mode, delta).
          c.mode = mode;
          c.delta = d;
          c.ft_above = 0;
          c.ft_left = 0;
          c.plane = 0;
          c.enable_edge_filter = 0;
          runner.run(c);
        }
      }
    }
  }
  std::printf("  cases=%lld samples=%lld\n", runner.stats().cases,
              runner.stats().samples);
}

void correctness_random() {
  std::printf("\n-- randomized blocks (mixed sizes/positions/flags) --\n");
  const Plane recon = make_plane(96, 96, 1, 0x51u);
  const Plane source = make_plane(96, 96, 4, 0x52u);
  const DevicePlane d_recon(recon);
  const DevicePlane d_source(source);
  Runner runner(recon, source, d_recon, d_source);

  g_rng = 0xfeedbeefu;
  for (int iter = 0; iter < 3000; ++iter) {
    const auto& sizes = all_sizes();
    const Size& s = sizes[next_rand() % sizes.size()];
    Case c;
    c.w = s.w;
    c.h = s.h;
    c.x0 = static_cast<int>(next_rand() % 96u);
    c.y0 = static_cast<int>(next_rand() % 96u);
    c.mode = static_cast<int>(next_rand() % 13u);
    const bool dir = c.mode >= 1 && c.mode <= 8;
    c.delta = dir ? static_cast<int>(next_rand() % 7u) - 3 : 0;
    c.ft_above = static_cast<int>(next_rand() % 2u);
    c.ft_left = static_cast<int>(next_rand() % 2u);
    c.plane = static_cast<int>(next_rand() % 2u);
    c.enable_edge_filter = static_cast<int>(next_rand() % 2u);
    c.num4_ar = -1;
    c.num4_bl = -1;
    c.have_left = -1;
    c.have_above = -1;
    c.tag = "random";
    runner.run(c);
  }
  std::printf("  cases=%lld samples=%lld\n", runner.stats().cases,
              runner.stats().samples);
}

void correctness_640x360_edges() {
  std::printf("\n-- 640x360 bottom / right edge blocks --\n");
  const int W = 640;
  const int H = 360;
  const Plane recon = make_plane(W, H, 3, 0x640u);
  const Plane source = make_plane(W, H, 0, 0x641u);
  const DevicePlane d_recon(recon);
  const DevicePlane d_source(source);
  Runner runner(recon, source, d_recon, d_source);

  struct Edge {
    int w, h, x0, y0;
    const char* tag;
  };
  const Edge edges[] = {
      {64, 64, 576, 320, "bottom64x64-partial40"},
      {64, 64, 576, 296, "bottom64x64-partial56"},
      {32, 32, 608, 328, "bottom32x32-partial24"},
      {32, 16, 608, 344, "bottom32x16-partial16"},
      {32, 8, 608, 352, "bottom32x8-full"},
      {16, 16, 624, 344, "bottom16x16-partial8"},
      {8, 8, 632, 352, "bottom8x8-full"},
      {64, 32, 576, 328, "bottom-right64x32"},
      {32, 64, 608, 296, "right32x64-partial32"},
      {64, 8, 0, 352, "bottom-left-full"},
  };
  for (const Edge& e : edges) {
    for (int mode = 0; mode < 13; ++mode) {
      const int deltas[3] = {0, (mode >= 1 && mode <= 8) ? -3 : 0,
                             (mode >= 1 && mode <= 8) ? 3 : 0};
      for (int di = 0; di < 3; ++di) {
        for (int plane = 0; plane < 2; ++plane) {
          Case c;
          c.w = e.w;
          c.h = e.h;
          c.x0 = e.x0;
          c.y0 = e.y0;
          c.mode = mode;
          c.delta = deltas[di];
          c.plane = plane;
          c.ft_above = (mode % 2);
          c.ft_left = ((mode >> 1) % 2);
          c.enable_edge_filter = 1;
          c.have_left = -1;
          c.have_above = -1;
          c.tag = e.tag;
          runner.run(c);
        }
      }
    }
  }
  std::printf("  cases=%lld samples=%lld\n", runner.stats().cases,
              runner.stats().samples);
}

void correctness_1080p_blocks() {
  std::printf("\n-- 1920x1080-style blocks (interior + bottom/right edges) --\n");
  const int W = 1920;
  const int H = 1080;
  const Plane recon = make_plane(W, H, 4, 0x1080u);
  const Plane source = make_plane(W, H, 2, 0x1081u);
  const DevicePlane d_recon(recon);
  const DevicePlane d_source(source);
  Runner runner(recon, source, d_recon, d_source);

  struct Spot {
    int w, h, x0, y0;
    const char* tag;
  };
  const Spot spots[] = {
      {64, 64, 0, 0, "1080p-topleft"},
      {64, 64, 960, 512, "1080p-center"},
      {64, 64, 1856, 0, "1080p-right-full"},
      {64, 64, 1856, 1016, "1080p-bottomright-partial"},
      {64, 64, 0, 1016, "1080p-bottomleft-partial"},
      {32, 32, 1888, 1048, "1080p-bottom32x32-partial"},
      {32, 32, 1888, 0, "1080p-right32x32"},
      {16, 16, 1904, 1064, "1080p-bottom16x16-partial"},
      {8, 8, 1912, 1072, "1080p-bottom8x8-full"},
      {32, 64, 1888, 1016, "1080p-bottom-right-rect"},
      {64, 32, 1856, 1048, "1080p-bottom32-partial"},
  };
  for (const Spot& s : spots) {
    for (int mode = 0; mode < 13; ++mode) {
      for (int delta = -3; delta <= 3; delta += 3) {
        if (!(mode >= 1 && mode <= 8) && delta != 0) continue;
        Case c;
        c.w = s.w;
        c.h = s.h;
        c.x0 = s.x0;
        c.y0 = s.y0;
        c.mode = mode;
        c.delta = delta;
        c.plane = (mode % 2);
        c.ft_above = 1;
        c.ft_left = (mode % 3 == 0) ? 1 : 0;
        c.enable_edge_filter = 1;
        c.have_left = -1;
        c.have_above = -1;
        c.tag = s.tag;
        runner.run(c);
      }
    }
  }
  std::printf("  cases=%lld samples=%lld\n", runner.stats().cases,
              runner.stats().samples);
}

void correctness_batch() {
  std::printf("\n-- batched API vs CPU reference --\n");
  const Plane recon = make_plane(640, 360, 2, 0xb00u);
  const Plane source = make_plane(640, 360, 1, 0xb01u);
  const DevicePlane d_recon(recon);
  const DevicePlane d_source(source);
  Runner runner(recon, source, d_recon, d_source);

  // A non-overlapping 32x32 grid over 640x360 (the real raster shape), plus a
  // mixed-size batch to exercise heterogeneous jobs in one launch.
  std::vector<Case> grid;
  for (int y = 0; y < 360; y += 32) {
    for (int x = 0; x < 640; x += 32) {
      Case c;
      c.w = 32;
      c.h = 32;
      c.x0 = x;
      c.y0 = y;
      c.mode = ((x / 32) + (y / 32)) % 13;
      const bool dir = c.mode >= 1 && c.mode <= 8;
      c.delta = dir ? static_cast<int>(((x + y) / 32) % 7) - 3 : 0;
      c.ft_above = ((x / 32) & 1);
      c.ft_left = ((y / 32) & 1);
      c.plane = ((x / 32) & 1);
      c.enable_edge_filter = 1;
      c.tag = "grid640";
      grid.push_back(c);
    }
  }
  runner.run_batch(grid, "640x360-grid");
  std::printf("  grid jobs=%zu\n", grid.size());

  std::vector<Case> mixed;
  g_rng = 0xc0ffeeu;
  for (int iter = 0; iter < 1500; ++iter) {
    const auto& sizes = all_sizes();
    const Size& s = sizes[next_rand() % sizes.size()];
    Case c;
    c.w = s.w;
    c.h = s.h;
    c.x0 = static_cast<int>(next_rand() % 640u);
    c.y0 = static_cast<int>(next_rand() % 360u);
    c.mode = static_cast<int>(next_rand() % 13u);
    const bool dir = c.mode >= 1 && c.mode <= 8;
    c.delta = dir ? static_cast<int>(next_rand() % 7u) - 3 : 0;
    c.ft_above = static_cast<int>(next_rand() % 2u);
    c.ft_left = static_cast<int>(next_rand() % 2u);
    c.plane = static_cast<int>(next_rand() % 2u);
    c.enable_edge_filter = 1;
    c.tag = "mixed";
    mixed.push_back(c);
  }
  runner.run_batch(mixed, "mixed");
  std::printf("  mixed jobs=%zu\n", mixed.size());
  std::printf("  total cases=%lld\n", runner.stats().cases);
}

void invalid_arguments() {
  std::printf("\n-- invalid arguments --\n");
  im::IntraBlock b;
  b.width = 3;   // not a power-of-two transform size
  b.height = 8;
  std::uint8_t* p = nullptr;
  std::int16_t* r = nullptr;
  CUDA_CHECK(cudaMalloc(&p, 64));
  CUDA_CHECK(cudaMalloc(&r, 128));
  int failures = 0;
  if (im::intra_predict(b, p) != cudaErrorInvalidValue) ++failures;

  b.width = 8;
  b.height = 8;
  b.x0 = 0;
  b.y0 = 0;
  b.max_x = 7;
  b.max_y = 7;
  b.have_left = 1;  // x0 == 0 cannot have a left neighbour
  if (im::intra_predict(b, p) != cudaErrorInvalidValue) ++failures;

  b.have_left = 0;
  b.angle_delta = 4;  // MAX_ANGLE_DELTA is 3
  if (im::intra_predict(b, p) != cudaErrorInvalidValue) ++failures;

  b.angle_delta = 0;
  b.mode = static_cast<im::IntraMode>(13);
  if (im::intra_predict(b, p) != cudaErrorInvalidValue) ++failures;

  b.mode = im::kDcPred;
  b.angle_delta = 1;  // non-directional modes must use delta 0
  if (im::intra_predict(b, p) != cudaErrorInvalidValue) ++failures;
  b.angle_delta = 0;
  if (im::intra_predict(b, p) != cudaSuccess) ++failures;

  // Batch: zero jobs is a no-op, null with jobs is invalid.
  if (im::intra_predict_batch(nullptr, 0) != cudaSuccess) ++failures;
  if (im::intra_predict_batch(nullptr, 3) != cudaErrorInvalidValue) ++failures;
  if (im::intra_predict_batch(nullptr, -1) != cudaErrorInvalidValue) ++failures;

  CUDA_CHECK(cudaFree(p));
  CUDA_CHECK(cudaFree(r));
  std::printf("  invalid-argument failures=%d\n", failures);
  if (failures != 0) {
    std::printf("  FAIL\n");
    std::exit(1);
  }
}

// ---------------------------------------------------------------------------
// Timing (microbenchmark; not an end-to-end encoder measurement)
// ---------------------------------------------------------------------------

struct Timing {
  const char* name;
  double ms_per_launch = 0.0;
  double blocks_per_s = 0.0;
  double msamples_per_s = 0.0;
};

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return static_cast<double>(ms);
}

// Largest legal transform extent <= v (4, 8, 16, 32, 64). 0 if none.
int largest_supported_dim(int v) {
  const int dims[5] = {64, 32, 16, 8, 4};
  for (int d : dims) {
    if (d <= v) return d;
  }
  return 0;
}

// Builds one device plane and a device job array covering a whole frame with
// `bw` x `bh` (possibly partial at the edges) blocks.
Timing time_frame_grid(int frame_w, int frame_h, int bw, int bh, int repeats,
                       const char* name) {
  const Plane plane = make_plane(frame_w, frame_h, 3, 0x777u);
  DevicePlane d_recon(plane);
  DevicePlane d_source(plane);

  std::vector<im::IntraJob> jobs;
  jobs.reserve(static_cast<std::size_t>((frame_w + bw - 1) / bw) *
               static_cast<std::size_t>((frame_h + bh - 1) / bh));
  std::size_t total = 0;
  for (int y = 0; y < frame_h;) {
    const int h = largest_supported_dim(std::min(bh, frame_h - y));
    if (h == 0) break;
    for (int x = 0; x < frame_w;) {
      const int w = largest_supported_dim(std::min(bw, frame_w - x));
      if (w == 0) break;
      im::IntraJob job;
      job.block.recon_plane = d_recon.get();
      job.block.recon_stride = d_recon.stride();
      job.block.x0 = x;
      job.block.y0 = y;
      job.block.width = w;
      job.block.height = h;
      job.block.max_x = frame_w - 1;
      job.block.max_y = frame_h - 1;
      job.block.have_left = x > 0 ? 1 : 0;
      job.block.have_above = y > 0 ? 1 : 0;
      job.block.num4_above_right = (frame_w - (x + w)) / 4;
      job.block.num4_below_left = (frame_h - (y + h)) / 4;
      job.block.plane = 0;
      job.block.mode = im::kD45Pred;
      job.block.angle_delta = 1;
      job.block.filter_type_above = 1;
      job.block.filter_type_left = 1;
      job.block.enable_edge_filter = 1;
      job.block.source_plane = d_source.get();
      job.block.source_stride = d_source.stride();
      job.block.source_max_x = frame_w - 1;
      job.block.source_max_y = frame_h - 1;
      jobs.push_back(job);
      total += static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
      x += w;
    }
    y += h;
  }
  std::uint8_t* d_pred = nullptr;
  std::int16_t* d_resid = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pred, total));
  CUDA_CHECK(cudaMalloc(&d_resid, total * sizeof(std::int16_t)));
  std::size_t offset = 0;
  for (im::IntraJob& job : jobs) {
    job.pred = d_pred + offset;
    job.residual = d_resid + offset;
    offset += static_cast<std::size_t>(job.block.width) *
              static_cast<std::size_t>(job.block.height);
  }
  im::IntraJob* d_jobs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_jobs, jobs.size() * sizeof(im::IntraJob)));
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(im::IntraJob),
                        cudaMemcpyHostToDevice));

  // Warm-up.
  for (int k = 0; k < 3; ++k) {
    CUDA_CHECK(im::intra_predict_batch(d_jobs, static_cast<int>(jobs.size())));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int k = 0; k < repeats; ++k) {
    CUDA_CHECK(im::intra_predict_batch(d_jobs, static_cast<int>(jobs.size())));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const double ms = elapsed_ms(start, stop);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  const double per_launch = ms / static_cast<double>(repeats);
  Timing t;
  t.name = name;
  t.ms_per_launch = per_launch;
  t.blocks_per_s =
      static_cast<double>(jobs.size()) / (per_launch * 1.0e-3);
  t.msamples_per_s =
      static_cast<double>(total) / (per_launch * 1.0e-3) / 1.0e6;

  CUDA_CHECK(cudaFree(d_jobs));
  CUDA_CHECK(cudaFree(d_pred));
  CUDA_CHECK(cudaFree(d_resid));
  return t;
}

Timing time_single_block_launches(int bw, int bh, int repeats,
                                  const char* name) {
  const Plane plane = make_plane(bw + 16, bh + 16, 2, 0x888u);
  DevicePlane d_recon(plane);
  DevicePlane d_source(plane);
  std::uint8_t* d_pred = nullptr;
  std::int16_t* d_resid = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pred, static_cast<std::size_t>(bw * bh)));
  CUDA_CHECK(cudaMalloc(&d_resid,
                        static_cast<std::size_t>(bw * bh) * sizeof(std::int16_t)));

  im::IntraBlock b;
  b.recon_plane = d_recon.get();
  b.recon_stride = d_recon.stride();
  b.x0 = 8;
  b.y0 = 8;
  b.width = bw;
  b.height = bh;
  b.max_x = plane.w - 1;
  b.max_y = plane.h - 1;
  b.have_left = 1;
  b.have_above = 1;
  b.num4_above_right = 4;
  b.num4_below_left = 4;
  b.plane = 0;
  b.mode = im::kD45Pred;
  b.angle_delta = 1;
  b.filter_type_above = 1;
  b.filter_type_left = 1;
  b.enable_edge_filter = 1;
  b.source_plane = d_source.get();
  b.source_stride = d_source.stride();
  b.source_max_x = plane.w - 1;
  b.source_max_y = plane.h - 1;

  for (int k = 0; k < 10; ++k) {
    CUDA_CHECK(im::intra_predict_residual(b, d_pred, d_resid));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int k = 0; k < repeats; ++k) {
    CUDA_CHECK(im::intra_predict_residual(b, d_pred, d_resid));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const double ms = elapsed_ms(start, stop);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  const double per_launch_us = ms * 1.0e3 / static_cast<double>(repeats);
  Timing t;
  t.name = name;
  t.ms_per_launch = per_launch_us * 1.0e-3;
  t.blocks_per_s = 1.0e6 / per_launch_us;
  t.msamples_per_s = static_cast<double>(bw * bh) / per_launch_us;  // samples/us = Msamples/s

  CUDA_CHECK(cudaFree(d_pred));
  CUDA_CHECK(cudaFree(d_resid));
  return t;
}

}  // namespace

// Reduced spot-check sweep used by --quick (and memcheck runs).
void correctness_quick_single() {
  std::printf("\n-- quick spot checks (single-block API) --\n");
  const Plane recon = make_plane(64, 48, 3, 0xa1u);
  const Plane source = make_plane(64, 48, 0, 0xa2u);
  const DevicePlane d_recon(recon);
  const DevicePlane d_source(source);
  Runner runner(recon, source, d_recon, d_source);

  const Size sizes[3] = {{8, 16}, {32, 32}, {64, 64}};
  const std::pair<int, int> spots[2] = {{0, 0}, {3, 3}};
  for (const Size& s : sizes) {
    for (const auto& pos : spots) {
      for (int mode = 0; mode < 13; ++mode) {
        for (int delta = -3; delta <= 3; delta += 3) {
          if (!(mode >= 1 && mode <= 8) && delta != 0) continue;
          for (int plane = 0; plane < 2; ++plane) {
            Case c;
            c.w = s.w;
            c.h = s.h;
            c.x0 = pos.first;
            c.y0 = pos.second;
            c.mode = mode;
            c.delta = delta;
            c.plane = plane;
            c.ft_above = 1;
            c.ft_left = (mode & 1);
            c.enable_edge_filter = 1;
            c.tag = "quick";
            runner.run(c);
          }
        }
      }
    }
  }
  std::printf("  cases=%lld samples=%lld\n", runner.stats().cases,
              runner.stats().samples);
}

// ---------------------------------------------------------------------------
// Optional CPU-module parity (build with --cpu-parity)
// ---------------------------------------------------------------------------

#ifdef AV2ENC_CPU_INTRA_PARITY

int g_cpu_parity_failures = 0;

// Compares src/encoder/intra_modes (CPU reference module) with the CUDA
// kernels over a representative sweep.  With the edge filter disabled the two
// must agree exactly.  With it enabled this build reports the known CPU-side
// divergence: src/encoder/intra_modes/intra_modes.cpp filters the edge array
// in place and reads already-updated samples, while §7.13.2.18 filters a
// snapshot (`edge[i] = LeftCol[i-1]`) and reads the original values.  The CUDA
// kernel follows the spec, so the "on" pass is reported, not failed.
void cpu_parity_sweep() {
  std::printf("\n-- CPU module parity (src/encoder/intra_modes) --\n");
  const Plane host_recon = make_plane(640, 360, 2, 0xc0u);
  DevicePlane d_recon(host_recon);
  std::uint8_t* d_pred = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pred, 64u * 64u));

  const Size sizes[10] = {{4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64},
                          {64, 32}, {32, 64}, {8, 16},  {16, 8},  {64, 16}};
  for (int ef = 0; ef < 2; ++ef) {
    long long samples = 0;
    long long bad = 0;
    int max_diff = 0;
    for (const Size& sz : sizes) {
      for (int mode = 0; mode < 13; ++mode) {
        for (int delta = -3; delta <= 3; delta += 3) {
          if (!(mode >= 1 && mode <= 8) && delta != 0) continue;
          for (int plane_id = 0; plane_id < 2; ++plane_id) {
            const int w = sz.w;
            const int h = sz.h;
            const int x0 = (w > 32 && plane_id != 0) ? 8 : 24;
            const int y0 = (h > 32 && plane_id != 0) ? 8 : 24;
            av2enc::intra::NeighborRequest req;
            req.recon = {host_recon.data.data(), host_recon.w};
            req.x = x0;
            req.y = y0;
            req.w = w;
            req.h = h;
            req.max_x = host_recon.w - 1;
            req.max_y = host_recon.h - 1;
            req.have_left = x0 > 0;
            req.have_above = y0 > 0;
            req.num4_above_right = (host_recon.w - (x0 + w)) / 4;
            req.num4_below_left = (host_recon.h - (y0 + h)) / 4;
            req.color_plane = plane_id;

            av2enc::intra::PredictParams params;
            params.mode = static_cast<av2enc::intra::Mode>(mode);
            params.angle_delta = delta;
            params.enable_edge_filter = ef != 0;
            params.above_neighbor_is_smooth = (mode & 1) != 0;
            params.left_neighbor_is_smooth = (mode & 2) != 0;

            std::vector<std::uint8_t> cpu_pred(
                static_cast<std::size_t>(w * h));
            const av2enc::intra::IntraStatus status =
                av2enc::intra::predict_block(req, 8, params, cpu_pred.data(),
                                             static_cast<std::size_t>(w));
            if (status != av2enc::intra::IntraStatus::kOk) {
              std::printf("  CPU predict_block failed\n");
              ++g_cpu_parity_failures;
              continue;
            }

            im::IntraBlock b;
            b.recon_plane = d_recon.get();
            b.recon_stride = d_recon.stride();
            b.x0 = x0;
            b.y0 = y0;
            b.width = w;
            b.height = h;
            b.max_x = d_recon.w() - 1;
            b.max_y = d_recon.h() - 1;
            b.have_left = req.have_left ? 1 : 0;
            b.have_above = req.have_above ? 1 : 0;
            b.num4_above_right = req.num4_above_right;
            b.num4_below_left = req.num4_below_left;
            b.plane = plane_id;
            b.mode = static_cast<im::IntraMode>(mode);
            b.angle_delta = delta;
            b.filter_type_above = params.above_neighbor_is_smooth ? 1 : 0;
            b.filter_type_left = params.left_neighbor_is_smooth ? 1 : 0;
            b.enable_edge_filter = ef;
            CUDA_CHECK(im::intra_predict(b, d_pred));
            std::vector<std::uint8_t> gpu_pred(
                static_cast<std::size_t>(w * h));
            CUDA_CHECK(cudaMemcpy(gpu_pred.data(), d_pred, gpu_pred.size(),
                                  cudaMemcpyDeviceToHost));

            for (std::size_t i = 0; i < gpu_pred.size(); ++i) {
              const int e = static_cast<int>(cpu_pred[i]);
              const int a = static_cast<int>(gpu_pred[i]);
              ++samples;
              if (e != a) {
                ++bad;
                max_diff = std::max(max_diff, std::abs(e - a));
              }
            }
          }
        }
      }
    }
    std::printf("  edge_filter=%d samples=%lld mismatches=%lld max_abs_diff=%d\n",
                ef, samples, bad, max_diff);
    if (ef == 0) {
      if (bad != 0) {
        std::printf("  FAIL: CPU and CUDA disagree with the edge filter off\n");
        ++g_cpu_parity_failures;
      } else {
        std::printf("  PASS: exact CPU/CUDA parity with the edge filter off\n");
      }
    } else if (bad != 0) {
      std::printf("  NOTE: expected divergence. src/encoder/intra_modes/"
                  "intra_modes.cpp intra_edge_filter() filters in place and "
                  "reads updated samples; §7.13.2.18 filters the pre-filter "
                  "edge snapshot. CUDA follows the spec.\n");
    }
  }
  CUDA_CHECK(cudaFree(d_pred));
}

#endif  // AV2ENC_CPU_INTRA_PARITY

// ---------------------------------------------------------------------------
// Table coverage
// ---------------------------------------------------------------------------

// Enumerates every Dr_Intra_Derivative index reachable through the AV2 mode
// set (base angles 45/67/90/113/135/157/180/203, angle deltas -3..3, all
// supported block shapes) and reports which indices are exercised.  Indices
// 0..2 and 88..89 are unreachable by construction (3-degree angle steps).
void report_derivative_coverage() {
  bool used[im::kDrDerivativeCount] = {};
  const int bases[8] = {45, 67, 90, 113, 135, 157, 180, 203};
  for (const Size& s : all_sizes()) {
    for (int base : bases) {
      for (int d = -3; d <= 3; ++d) {
        int p = base + d * 3;
        const int mapped = ref_wide_angle_map(p, s.w, s.h);
        p = mapped;
        if (p > 0 && p < 90) {
          used[p] = true;
        } else if (p > 90 && p < 180) {
          used[180 - p] = true;
          used[p - 90] = true;
        } else if (p > 180 && p < 270) {
          used[270 - p] = true;
        }
      }
    }
  }
  int first = -1;
  int last = -1;
  int count = 0;
  for (int i = 0; i < im::kDrDerivativeCount; ++i) {
    if (used[i]) {
      if (first < 0) first = i;
      last = i;
      ++count;
    }
  }
  std::printf("-- Dr_Intra_Derivative coverage: %d/%d entries, reachable "
              "range [%d, %d] --\n",
              count, im::kDrDerivativeCount, first, last);
  std::printf("   unreachable (never indexed by AV2 angles):");
  for (int i = 0; i < im::kDrDerivativeCount; ++i) {
    if (!used[i]) std::printf(" %d", i);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--quick") {
      quick = true;
    }
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::printf("av2enc CUDA M1 intra-modes test\n");
  std::printf("GPU: %s, sm_%d%d, %d SMs, %.1f GiB\n", props.name, props.major,
              props.minor, props.multiProcessorCount,
              static_cast<double>(props.totalGlobalMem) /
                  (1024.0 * 1024.0 * 1024.0));

  if (quick) {
    correctness_quick_single();
  } else {
    report_derivative_coverage();
    correctness_full_matrix();
    correctness_random();
    correctness_640x360_edges();
    correctness_1080p_blocks();
  }
  correctness_batch();
  invalid_arguments();
#ifdef AV2ENC_CPU_INTRA_PARITY
  cpu_parity_sweep();
#endif

  if (quick) {
    std::printf("\n-- summary (quick) --\n");
    const Stage stages_quick[] = {g_pred, g_resid, g_batch_pred,
                                  g_batch_resid};
    long long total_quick = 0;
    long long bad_quick = 0;
    for (const Stage& s : stages_quick) {
      std::printf("  %-14s samples=%9lld mismatches=%9lld max_abs_diff=%d\n",
                  s.name, s.samples, s.mismatches, s.max_abs_diff);
      total_quick += s.samples;
      bad_quick += s.mismatches;
    }
    std::printf("  total samples=%lld, mismatches=%lld\n", total_quick,
                bad_quick);
#ifdef AV2ENC_CPU_INTRA_PARITY
    bad_quick += g_cpu_parity_failures;
#endif
    std::printf("%s", bad_quick == 0 ? "PASS\n" : "FAIL\n");
    return bad_quick == 0 ? 0 : 1;
  }

  std::printf("\n-- timing (microbenchmark, microbenchmark only) --\n");
  const Timing single32 =
      time_single_block_launches(32, 32, 1000, "single 32x32 launch");
  const Timing single64 =
      time_single_block_launches(64, 64, 1000, "single 64x64 launch");
  const Timing grid360_32 =
      time_frame_grid(640, 360, 32, 32, 200, "batch 640x360 32x32 grid");
  const Timing grid360_16 =
      time_frame_grid(640, 360, 16, 16, 200, "batch 640x360 16x16 grid");
  const Timing grid1080_32 =
      time_frame_grid(1920, 1080, 32, 32, 100, "batch 1920x1080 32x32 grid");
  const Timing grid1080_64 =
      time_frame_grid(1920, 1080, 64, 64, 100, "batch 1920x1080 64x64 grid");
  const Timing grid1080_8 =
      time_frame_grid(1920, 1080, 8, 8, 100, "batch 1920x1080 8x8 grid");
  const Timing timings[] = {single32, single64,      grid360_32,  grid360_16,
                            grid1080_32, grid1080_64, grid1080_8};
  std::printf("  %-32s %10s %12s %14s\n", "case", "ms/call", "blocks/s",
              "Msamples/s");
  for (const Timing& t : timings) {
    std::printf("  %-32s %10.4f %12.0f %14.1f\n", t.name, t.ms_per_launch,
                t.blocks_per_s, t.msamples_per_s);
  }

  std::printf("\n-- summary --\n");
  const Stage stages[] = {g_pred, g_resid, g_batch_pred, g_batch_resid};
  long long total = 0;
  long long bad = 0;
  for (const Stage& s : stages) {
    std::printf("  %-14s samples=%9lld mismatches=%9lld max_abs_diff=%d\n",
                s.name, s.samples, s.mismatches, s.max_abs_diff);
    total += s.samples;
    bad += s.mismatches;
  }
  std::printf("  total samples=%lld, mismatches=%lld\n", total, bad);
#ifdef AV2ENC_CPU_INTRA_PARITY
  bad += g_cpu_parity_failures;
#endif
  if (bad != 0) {
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
