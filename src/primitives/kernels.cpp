// Scalar reference implementations and start-up dispatch for the pixel-kernel
// layer.  See kernels.h for the contract.
//
// Every loop below is a transcription of an existing inlined loop; the source
// file and function are named at each implementation and tabulated in
// docs/primitives-inventory.md.  Do not "improve" the arithmetic: accumulation
// order, the (sum + 8) >> 4 SATD rounding and the double-precision variance
// sequence are part of the bit-exactness contract.

#include "primitives/kernels.h"

#include "primitives/subpel_filters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace av2enc::primitives {
namespace {

// ---------------------------------------------------------------------------
// SAD - transcribed from codec/inter/me/me_core.h:me_sad_rows (integer path),
// codec/intra/ibc/core/ibc_core.h:sad_block and codec/rate/rdo/rdo.cpp:sad_u8.
// The three existing copies agree bit-for-bit; this is their common shape.
// ---------------------------------------------------------------------------
std::uint64_t scalar_sad(const std::uint8_t* src, std::size_t src_stride,
                         const std::uint8_t* ref, std::size_t ref_stride,
                         int w, int h) noexcept {
  if (w <= 0 || h <= 0) return 0;
  std::uint64_t sad = 0;
  for (int i = 0; i < h; ++i) {
    const std::uint8_t* src_row =
        src + static_cast<std::size_t>(i) * src_stride;
    const std::uint8_t* ref_row =
        ref + static_cast<std::size_t>(i) * ref_stride;
    for (int j = 0; j < w; ++j) {
      const int d = static_cast<int>(src_row[j]) - static_cast<int>(ref_row[j]);
      sad += static_cast<std::uint64_t>(d < 0 ? -d : d);
    }
  }
  return sad;
}

// Fused SAD+SSE, transcribed from the second pass of
// codec/intra/modes/intra_modes.cpp:prediction_cost and
// codec/intra/dip/dip.cpp:prediction_cost8.  Both call sites compute the SAD
// with scalar_sad above; this is that loop with the SSE accumulator folded in.
// The `int` product is exact for 8-bit inputs (|d| <= 255, d*d <= 65025), so it
// is value-identical to the int64 form in dip.
// ---------------------------------------------------------------------------
void scalar_sad_sse(const std::uint8_t* src, std::size_t src_stride,
                    const std::uint8_t* ref, std::size_t ref_stride, int w,
                    int h, std::uint64_t* sad, std::uint64_t* sse) noexcept {
  if (sad == nullptr || sse == nullptr) return;
  *sad = 0;
  *sse = 0;
  if (w <= 0 || h <= 0) return;
  std::uint64_t sad_sum = 0;
  std::uint64_t sse_sum = 0;
  for (int i = 0; i < h; ++i) {
    const std::uint8_t* src_row =
        src + static_cast<std::size_t>(i) * src_stride;
    const std::uint8_t* ref_row =
        ref + static_cast<std::size_t>(i) * ref_stride;
    for (int j = 0; j < w; ++j) {
      const int d = static_cast<int>(src_row[j]) - static_cast<int>(ref_row[j]);
      sad_sum += static_cast<std::uint64_t>(d < 0 ? -d : d);
      sse_sum += static_cast<std::uint64_t>(d * d);
    }
  }
  *sad = sad_sum;
  *sse = sse_sum;
}

// ---------------------------------------------------------------------------
// SATD - 8x8 path transcribed from codec/inter/me/me_core.h:me_hadamard8,
// me_hadamard8_2d and me_satd_unit; 4x4 path transcribed from
// codec/rate/rdo/rdo.cpp:hadamard4 and satd_u8.
// ---------------------------------------------------------------------------
void hadamard8(int* v) noexcept {
  for (int len = 1; len < 8; len <<= 1) {
    for (int i = 0; i < 8; i += len << 1) {
      for (int j = 0; j < len; ++j) {
        const int a = v[i + j];
        const int b = v[i + j + len];
        v[i + j] = a + b;
        v[i + j + len] = a - b;
      }
    }
  }
}

void hadamard8_2d(int* v) noexcept {
  for (int r = 0; r < 8; ++r) {
    hadamard8(v + r * 8);
  }
  for (int len = 1; len < 8; len <<= 1) {
    for (int i = 0; i < 8; i += len << 1) {
      for (int j = 0; j < len; ++j) {
        for (int c = 0; c < 8; ++c) {
          const int a = v[(i + j) * 8 + c];
          const int b = v[(i + j + len) * 8 + c];
          v[(i + j) * 8 + c] = a + b;
          v[(i + j + len) * 8 + c] = a - b;
        }
      }
    }
  }
}

void hadamard4(int* values) noexcept {
  const int a0 = values[0] + values[1];
  const int a1 = values[0] - values[1];
  const int a2 = values[2] + values[3];
  const int a3 = values[2] - values[3];
  values[0] = a0 + a2;
  values[1] = a1 + a3;
  values[2] = a0 - a2;
  values[3] = a1 - a3;
}

std::uint64_t satd8x8(const std::uint8_t* src, std::size_t src_stride,
                      const std::uint8_t* ref, std::size_t ref_stride, int w,
                      int h) noexcept {
  // me_cost_plane(kMeSatd) sums one me_satd_unit per 8x8 tile in this order.
  std::uint64_t total = 0;
  const int tiles_x = (w + 7) >> 3;
  const int tiles_y = (h + 7) >> 3;
  for (int ty = 0; ty < tiles_y; ++ty) {
    for (int tx = 0; tx < tiles_x; ++tx) {
      int m[64];
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
          const int rr = (ty << 3) + r;
          const int cc = (tx << 3) + c;
          int v = 0;
          if (rr < h && cc < w) {
            const int src_v = static_cast<int>(
                src[static_cast<std::size_t>(rr) * src_stride +
                    static_cast<std::size_t>(cc)]);
            const int ref_v = static_cast<int>(
                ref[static_cast<std::size_t>(rr) * ref_stride +
                    static_cast<std::size_t>(cc)]);
            v = src_v - ref_v;
          }
          m[r * 8 + c] = v;
        }
      }
      hadamard8_2d(m);
      std::int32_t sum = 0;
      for (int i = 0; i < 64; ++i) {
        sum += m[i] < 0 ? -m[i] : m[i];
      }
      total += static_cast<std::uint64_t>((sum + 8) >> 4);
    }
  }
  return total;
}

std::uint64_t satd4x4(const std::uint8_t* src, std::size_t src_stride,
                      const std::uint8_t* ref, std::size_t ref_stride, int w,
                      int h) noexcept {
  // rdo::satd_u8: 4x4 tiles, zero-padded at the right/bottom edge, unscaled.
  std::uint64_t total = 0;
  for (int block_y = 0; block_y < h; block_y += 4) {
    for (int block_x = 0; block_x < w; block_x += 4) {
      int block[16] = {};
      for (int y = 0; y < 4; ++y) {
        const int source_y = block_y + y;
        if (source_y >= h) continue;
        const std::uint8_t* src_row =
            src + static_cast<std::size_t>(source_y) * src_stride;
        const std::uint8_t* ref_row =
            ref + static_cast<std::size_t>(source_y) * ref_stride;
        for (int x = 0; x < 4; ++x) {
          const int source_x = block_x + x;
          if (source_x >= w) continue;
          block[y * 4 + x] = static_cast<int>(src_row[source_x]) -
                             static_cast<int>(ref_row[source_x]);
        }
      }
      for (int y = 0; y < 4; ++y) {
        hadamard4(&block[y * 4]);
      }
      for (int x = 0; x < 4; ++x) {
        int column[4] = {block[x], block[4 + x], block[8 + x], block[12 + x]};
        hadamard4(column);
        block[x] = column[0];
        block[4 + x] = column[1];
        block[8 + x] = column[2];
        block[12 + x] = column[3];
      }
      for (int i = 0; i < 16; ++i) {
        const int value = block[i];
        total += static_cast<std::uint64_t>(value < 0 ? -value : value);
      }
    }
  }
  return total;
}

std::uint64_t scalar_satd(const std::uint8_t* src, std::size_t src_stride,
                          const std::uint8_t* ref, std::size_t ref_stride,
                          int w, int h, SatdTile tile) noexcept {
  if (w <= 0 || h <= 0) return 0;
  if (tile == SatdTile::k4x4) {
    return satd4x4(src, src_stride, ref, ref_stride, w, h);
  }
  if (tile == SatdTile::k8x8) {
    return satd8x8(src, src_stride, ref, ref_stride, w, h);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Variance - transcribed from pipeline/video/video_frame.cpp:
// block_variance_cost_at() (the still encoder and hbd copies use the same
// formula).  Row-major int64 sum, double mean, row-major double SSE,
// truncated to int64.
// ---------------------------------------------------------------------------
std::int64_t scalar_variance(const std::uint8_t* src, std::size_t src_stride,
                             int w, int h) noexcept {
  if (w <= 0 || h <= 0) return 0;
  std::int64_t sum = 0;
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* row =
        src + static_cast<std::size_t>(y) * src_stride;
    for (int x = 0; x < w; ++x) {
      sum += static_cast<std::int64_t>(row[x]);
    }
  }
  const double mean = static_cast<double>(sum) /
                      static_cast<double>(w * h);
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* row =
        src + static_cast<std::size_t>(y) * src_stride;
    for (int x = 0; x < w; ++x) {
      const double d = static_cast<double>(row[x]) - mean;
      sse += d * d;
    }
  }
  return static_cast<std::int64_t>(sse);
}

// ---------------------------------------------------------------------------
// 16-bit (hbd) kernels - transcribed from pipeline/video/hbd/video_hbd_ops.h.
// The container is uint16_t; the arithmetic is integer (SAD) or the same
// binary64 sequence as the 8-bit variance (variance_u16).
// ---------------------------------------------------------------------------
std::int64_t scalar_variance_u16(const std::uint16_t* src,
                                 std::size_t src_stride, int w,
                                 int h) noexcept {
  if (w <= 0 || h <= 0) return 0;
  std::int64_t sum = 0;
  for (int y = 0; y < h; ++y) {
    const std::uint16_t* row =
        src + static_cast<std::size_t>(y) * src_stride;
    for (int x = 0; x < w; ++x) {
      sum += static_cast<std::int64_t>(row[x]);
    }
  }
  const double mean = static_cast<double>(sum) /
                      static_cast<double>(w * h);
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    const std::uint16_t* row =
        src + static_cast<std::size_t>(y) * src_stride;
    for (int x = 0; x < w; ++x) {
      const double d = static_cast<double>(row[x]) - mean;
      sse += d * d;
    }
  }
  return static_cast<std::int64_t>(sse);
}

int clip3_value(int low, int high, int value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

std::int64_t scalar_sad_u16_clamped(const std::uint16_t* cur, int cur_stride,
                                    const std::uint16_t* ref, int ref_stride,
                                    int x, int y, int w, int h, int mv_y,
                                    int mv_x, int ref_w, int ref_h) noexcept {
  if (w <= 0 || h <= 0) return 0;
  std::int64_t sad = 0;
  for (int r = 0; r < h; ++r) {
    const int ry = clip3_value(0, ref_h - 1, y + r + mv_y);
    for (int c = 0; c < w; ++c) {
      const int rx = clip3_value(0, ref_w - 1, x + c + mv_x);
      const int a = static_cast<int>(
          cur[static_cast<std::size_t>(y + r) *
                  static_cast<std::size_t>(cur_stride) +
              static_cast<std::size_t>(x + c)]);
      const int b = static_cast<int>(
          ref[static_cast<std::size_t>(ry) * static_cast<std::size_t>(ref_stride) +
              static_cast<std::size_t>(rx)]);
      sad += a > b ? a - b : b - a;
    }
  }
  return sad;
}

// ---------------------------------------------------------------------------
// Subpel ME SAD - transcribed verbatim from codec/inter/me/me_core.h:
// me_sad_subpel_rows -> me_sample_at -> me_interp_sample.  The per-sample body
// is the normative two-pass filter: 8 clamped horizontal taps, Round0 (3),
// 8 clamped vertical taps, Round1 (11), clip to 8 bits.  The integer-MV
// shortcut is me_sample_at's, not me_interp_sample's: it reads the reference
// at the integer position WITHOUT clamping.  Both behaviours are preserved so
// the migration is byte-identical; see docs/primitives-inventory.md §6.1.
// ---------------------------------------------------------------------------
std::int32_t scalar_subpel_sad_rows(
    const std::uint8_t* cur, std::size_t cur_stride, const std::uint8_t* ref,
    std::size_t ref_stride, int ref_w, int ref_h, int x, int y, int w,
    std::int32_t mv_x, std::int32_t mv_y, int filter, int r0, int r1,
    int rstep) noexcept {
  if (w <= 0 || r1 <= r0 || rstep <= 0) return 0;
  const std::int16_t(*coeff)[8] =
      kSubpelFilters[clip3_value(0, kSubpelMaxFilter, filter)];
  const int max_ry = ref_h - 1;
  const int max_rx = ref_w - 1;
  std::int32_t sum = 0;
  for (int r = r0; r < r1; r += rstep) {
    const std::uint8_t* cr =
        cur + static_cast<std::size_t>(y + r) * cur_stride +
        static_cast<std::size_t>(x);
    for (int c = 0; c < w; ++c) {
      const int px = x + c;
      const int py = y + r;
      const int xq = (px << kSubpelBits) + mv_x;
      const int yq = (py << kSubpelBits) + mv_y;
      int pred;
      if (((xq | yq) & kSubpelMask) == 0) {
        // me_sample_at integer shortcut: no border clamp.
        pred = static_cast<int>(
            ref[static_cast<std::size_t>(yq >> kSubpelBits) * ref_stride +
                static_cast<std::size_t>(xq >> kSubpelBits)]);
      } else {
        const int ix = xq >> kSubpelBits;
        const int iy = yq >> kSubpelBits;
        const int fx = xq & kSubpelMask;
        const int fy = yq & kSubpelMask;
        std::int32_t h[8];
        for (int t = 0; t < 8; ++t) {
          const int ry = clip3_value(0, max_ry, iy + t - 3);
          const std::uint8_t* row =
              ref + static_cast<std::size_t>(ry) * ref_stride;
          std::int32_t s = 0;
          for (int u = 0; u < 8; ++u) {
            const int rx = clip3_value(0, max_rx, ix + u - 3);
            s += static_cast<std::int32_t>(coeff[fx][u]) *
                 static_cast<std::int32_t>(row[rx]);
          }
          h[t] = (s + (static_cast<std::int32_t>(1) << (kInterRound0 - 1))) >>
                 kInterRound0;
        }
        std::int32_t s = 0;
        for (int t = 0; t < 8; ++t) {
          s += static_cast<std::int32_t>(coeff[fy][t]) * h[t];
        }
        pred = clip3_value(
            0, 255,
            static_cast<int>((s + (static_cast<std::int32_t>(1)
                                   << (kInterRound1 - 1))) >>
                             kInterRound1));
      }
      const int d = static_cast<int>(cr[c]) - pred;
      sum += d < 0 ? -d : d;
    }
  }
  return sum;
}

// ---------------------------------------------------------------------------
// Block copy - transcribed from codec/intra/ibc/core/ibc_core.h:copy_plane.
// ---------------------------------------------------------------------------
void scalar_block_copy(const std::uint8_t* src, std::size_t src_stride,
                       std::uint8_t* dst, std::size_t dst_stride, int w,
                       int h) noexcept {
  if (w <= 0 || h <= 0) return;
  for (int i = 0; i < h; ++i) {
    const std::uint8_t* src_row =
        src + static_cast<std::size_t>(i) * src_stride;
    std::uint8_t* dst_row = dst + static_cast<std::size_t>(i) * dst_stride;
    for (int j = 0; j < w; ++j) {
      dst_row[j] = src_row[j];
    }
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.18 intra edge filter - transcribed from
// codec/intra/modes/intra_modes.cpp:intra_edge_filter.  `edge` is the spec's
// derived edge[] array (edge[i] == AboveRow[i-1]), so the snapshot is
// edge[0..size-1] and the writes are edge[1..size-1].
// ---------------------------------------------------------------------------
void scalar_edge_filter(std::int16_t* edge, int size, int strength) noexcept {
  constexpr int kMaxEdgeSize = 129;  // intra::EdgeRow::kMaxIndex
  if (edge == nullptr || size <= 1 || size > kMaxEdgeSize || strength < 1 ||
      strength > 3) {
    return;
  }
  constexpr int kKernel[3][5] = {{0, 4, 8, 4, 0},
                                 {0, 5, 6, 5, 0},
                                 {2, 4, 4, 4, 2}};
  const int* taps = kKernel[strength - 1];

  int snapshot[kMaxEdgeSize];
  for (int i = 0; i < size; ++i) {
    snapshot[i] = edge[i];
  }
  for (int i = 1; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < 5; ++j) {
      int k = i - 2 + j;
      if (k < 0) k = 0;
      if (k > size - 1) k = size - 1;
      sum += taps[j] * snapshot[k];
    }
    edge[i] = static_cast<std::int16_t>((sum + 8) >> 4);
  }
}

// ---------------------------------------------------------------------------
// §7.13.2.17 strength selection - transcribed from
// codec/intra/modes/intra_modes.cpp:edge_filter_strength.
// ---------------------------------------------------------------------------
int strength_impl(int w, int h, bool smooth_neighbors, int delta) noexcept {
  const int d = delta < 0 ? -delta : delta;
  const int blk_wh = w + h;
  int strength = 0;
  if (!smooth_neighbors) {
    if (blk_wh <= 8) {
      if (d >= 56) {
        strength = 1;
      }
    } else if (blk_wh <= 12) {
      if (d >= 40) {
        strength = 1;
      }
    } else if (blk_wh <= 16) {
      if (d >= 40) {
        strength = 1;
      }
    } else if (blk_wh <= 24) {
      if (d >= 8) {
        strength = 1;
      }
      if (d >= 16) {
        strength = 2;
      }
      if (d >= 32) {
        strength = 3;
      }
    } else if (blk_wh <= 32) {
      strength = 1;
      if (d >= 4) {
        strength = 2;
      }
      if (d >= 32) {
        strength = 3;
      }
    } else {
      strength = 3;
    }
  } else {
    if (blk_wh <= 8) {
      if (d >= 40) {
        strength = 1;
      }
      if (d >= 64) {
        strength = 2;
      }
    } else if (blk_wh <= 16) {
      if (d >= 20) {
        strength = 1;
      }
      if (d >= 48) {
        strength = 2;
      }
    } else if (blk_wh <= 24) {
      if (d >= 4) {
        strength = 3;
      }
    } else {
      strength = 3;
    }
  }
  return strength;
}

constexpr KernelTable make_scalar_table() noexcept {
  return KernelTable{"scalar",
                     scalar_sad,
                     scalar_sad_sse,
                     scalar_satd,
                     scalar_variance,
                     scalar_block_copy,
                     scalar_edge_filter,
                     scalar_sad_u16_clamped,
                     scalar_variance_u16,
                     scalar_subpel_sad_rows};
}

}  // namespace

const KernelTable& scalar_kernels() noexcept {
  static const KernelTable table = make_scalar_table();
  return table;
}

// ---------------------------------------------------------------------------
// Dispatch
//
// The NEON table lives in src/primitives/kernels_neon.cpp; kernels.h defines
// AV2ENC_PRIMITIVES_HAVE_NEON when the compiler target is AArch64 and declares
// neon_kernels().  AVX2 will follow the same pattern with
// AV2ENC_PRIMITIVES_HAVE_AVX2.  Nothing else in this file changes.
// ---------------------------------------------------------------------------
namespace {

bool is_auto(std::string_view requested) noexcept {
  return requested.empty() || requested == "auto";
}

bool is_scalar(std::string_view requested) noexcept {
  return requested == "scalar";
}

}  // namespace

const KernelTable& select_kernels(std::string_view requested) noexcept {
  if (is_scalar(requested)) return scalar_kernels();
#if defined(AV2ENC_PRIMITIVES_HAVE_NEON)
  if (requested == "neon") return neon_kernels();
#endif
#if defined(AV2ENC_PRIMITIVES_HAVE_AVX2)
  if (requested == "avx2") return avx2_kernels();
#endif
  if (is_auto(requested)) {
#if defined(AV2ENC_PRIMITIVES_HAVE_NEON)
    return neon_kernels();
#elif defined(AV2ENC_PRIMITIVES_HAVE_AVX2)
    return avx2_kernels();
#else
    return scalar_kernels();
#endif
  }
  // Unknown name or an ISA that is not compiled in: fall back rather than
  // abort.  intro() reports the request so a typo is visible.
  return scalar_kernels();
}

namespace {

struct StartupSelection {
  const KernelTable* table;
  const char* requested;
};

const StartupSelection& startup() noexcept {
  static const StartupSelection selection = [] {
    const char* env = std::getenv("AV2ENC_SIMD");
    const char* requested = (env != nullptr && env[0] != '\0') ? env : "auto";
    StartupSelection out;
    out.requested = requested;
    out.table = &select_kernels(requested);
    // A request for a target this build does not contain must keep running
    // (an inherited AV2ENC_SIMD can never break a run).  Say so once, from
    // inside the static, instead of on every kernel call.
    if (!is_auto(requested) && !is_scalar(requested) &&
        out.table == &scalar_kernels()) {
      std::fprintf(stderr,
                   "av2enc: primitives: AV2ENC_SIMD=%s ignored: target not "
                   "built; using scalar\n",
                   requested);
    }
    return out;
  }();
  return selection;
}

}  // namespace

const KernelTable& kernels() noexcept {
  return *startup().table;
}

std::string_view selected_target() noexcept {
  return startup().table->name;
}

std::string available_targets() {
  std::string out = scalar_kernels().name;
#if defined(AV2ENC_PRIMITIVES_HAVE_NEON)
  out += ", neon";
#endif
#if defined(AV2ENC_PRIMITIVES_HAVE_AVX2)
  out += ", avx2";
#endif
  return out;
}

std::string intro() {
  const StartupSelection& selection = startup();
  const std::string_view requested = selection.requested;
  std::string out = "primitives: ";
  out += selection.table->name;
  out += " (";
  if (selection.table == &scalar_kernels()) {
    out += "portable C++ reference";
  } else {
    out += "SIMD";
  }
  out += "; AV2ENC_SIMD=";
  out += requested;
  if (!is_auto(requested) && !is_scalar(requested) &&
      selection.table == &scalar_kernels()) {
    out += " ignored: target not built";
  }
  out += "; available: ";
  out += available_targets();
  out += ')';
  return out;
}

int intra_edge_filter_strength(int w, int h, bool smooth_neighbors,
                               int delta) noexcept {
  return strength_impl(w, h, smooth_neighbors, delta);
}

}  // namespace av2enc::primitives
