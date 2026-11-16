// Proof test for the pixel-kernel layer (src/primitives/).
//
// For every kernel this compares the new scalar implementation against the
// existing inline implementation(s) it was transcribed from, on deterministic
// random blocks:
//
//   sad        vs me::me_sad_rows, ibc::sad_block, rdo::sad_u8
//   sad_sse    vs the two verbatim SSE loops it replaces (intra::prediction_cost,
//              dip::prediction_cost8) plus the same three SAD oracles
//   sad_u16    vs a verbatim transcription of hbd::hbd_sad_at (clipped 16-bit)
//   variance_u16 vs a verbatim transcription of hbd::hbd_variance_cost
//   satd 8x8   vs me::me_cost_plane(kMeSatd)     (hadamard8, (sum+8)>>4)
//   satd 4x4   vs rdo::satd_u8                   (hadamard4, unscaled)
//   variance   vs a verbatim transcription of video_frame.cpp's private
//              block_variance_cost_at() and vs video::hbd::hbd_variance_cost
//              (same formula, uint16 container)
//   block_copy vs ibc::copy_plane
//   edge       vs intra::intra_edge_filter       (§7.13.2.18)
//   strength   vs intra::edge_filter_strength    (§7.13.2.17)
//
// Sizes cover every block the encoder uses (4/8/16/32/64), rectangular leaves,
// odd widths/heights, and unaligned (stride > width) source/reference planes at
// non-zero pointer offsets.  Exact equality is asserted, never a tolerance.

#include "primitives/kernels.h"

#include "primitives/subpel_filters.h"

#include "test_common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "codec/inter/me/me.h"
#include "codec/inter/me/me_core.h"
#include "codec/intra/dip/dip.h"
#include "codec/intra/ibc/core/ibc_core.h"
#include "codec/intra/modes/intra_modes.h"
#include "codec/rate/rdo/rdo.h"
#include "pipeline/video/hbd/video_hbd_ops.h"

namespace {

using av2enc::primitives::SatdTile;

struct Stats {
  std::uint64_t blocks = 0;
  std::uint64_t samples = 0;
  std::uint64_t edge_positions = 0;
  std::uint64_t sad_sse_blocks = 0;
  std::uint64_t call_site_blocks = 0;
  std::uint64_t sad16_blocks = 0;
  std::uint64_t variance16_blocks = 0;
  std::uint64_t comparisons = 0;
  // Direct scalar-vs-SIMD comparisons of both sad_sse outputs.
  std::uint64_t sad_sse_pairs = 0;
  // Subpel ME SAD: block count, visited samples, scalar-vs-SIMD pairs.
  std::uint64_t subpel_blocks = 0;
  std::uint64_t subpel_samples = 0;
  std::uint64_t subpel_pairs = 0;
};

Stats g_stats;

// Deterministic xorshift64*; no std::random_device, so the test is identical
// on every host and under every sanitizer.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed == 0 ? 1 : seed) {}
  std::uint32_t next() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return static_cast<std::uint32_t>((state_ * 2685821657736338717ULL) >> 32);
  }
  std::uint8_t byte() { return static_cast<std::uint8_t>(next() & 0xFFU); }

 private:
  std::uint64_t state_;
};

enum class Texture { kRandom, kGradient, kFlat, kBinary };

struct TestPlane {
  std::vector<std::uint8_t> bytes;
  std::size_t offset = 0;
  std::size_t stride = 0;

  const std::uint8_t* data() const { return bytes.data() + offset; }
  std::uint8_t* data() { return bytes.data() + offset; }
};

TestPlane make_plane(int w, int h, int pad, std::size_t offset, Texture texture,
                     Rng& rng) {
  TestPlane plane;
  plane.offset = offset;
  plane.stride = static_cast<std::size_t>(w + pad);
  const std::size_t size = offset + plane.stride * static_cast<std::size_t>(h) +
                           16;
  plane.bytes.resize(size);
  for (std::uint8_t& value : plane.bytes) {
    value = rng.byte();
  }
  std::uint8_t* base = plane.data();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      std::uint8_t value = 0;
      switch (texture) {
        case Texture::kRandom:
          value = rng.byte();
          break;
        case Texture::kGradient:
          value = static_cast<std::uint8_t>((x * 5 + y * 3 +
                                             static_cast<int>(rng.next() & 7U)) &
                                            0xFF);
          break;
        case Texture::kFlat:
          value = 96;
          break;
        case Texture::kBinary:
          value = ((x ^ y) & 1) != 0 ? 0 : 255;
          break;
      }
      base[static_cast<std::size_t>(y) * plane.stride +
           static_cast<std::size_t>(x)] = value;
    }
  }
  return plane;
}

struct TestPlane16 {
  std::vector<std::uint16_t> samples;
  std::size_t offset = 0;
  std::size_t stride = 0;

  const std::uint16_t* data() const { return samples.data() + offset; }
  std::uint16_t* data() { return samples.data() + offset; }
};

// 16-bit (high-bit-depth) plane.  Values are LSB-aligned to `bit_depth`, so a
// 10-bit plane uses 0..1023 and the padding/offsets make every stride case
// explicit.
TestPlane16 make_plane16(int w, int h, int pad, std::size_t offset,
                         int bit_depth, Rng& rng) {
  TestPlane16 plane;
  plane.offset = offset;
  plane.stride = static_cast<std::size_t>(w + pad);
  const int max_value = (1 << bit_depth) - 1;
  const std::size_t size =
      offset + plane.stride * static_cast<std::size_t>(h) + 16;
  plane.samples.resize(size);
  for (std::uint16_t& value : plane.samples) {
    value = static_cast<std::uint16_t>(rng.next() %
                                       static_cast<std::uint32_t>(max_value + 1));
  }
  std::uint16_t* base = plane.data();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      base[static_cast<std::size_t>(y) * plane.stride +
           static_cast<std::size_t>(x)] =
          static_cast<std::uint16_t>(rng.next() %
                                     static_cast<std::uint32_t>(max_value + 1));
    }
  }
  return plane;
}

av2enc::me::MeCorePlane me_plane(const TestPlane& plane, int w, int h) {
  av2enc::me::MeCorePlane out;
  out.data = plane.data();
  out.stride = static_cast<int>(plane.stride);
  out.width = w;
  out.height = h;
  return out;
}

// rdo::sad_u8 / rdo::satd_u8 take one stride for both operands.  Pack the
// visible region into contiguous stride-w buffers so the comparison exercises
// the metric, not the stride constraint.
std::vector<std::uint8_t> pack_block(const TestPlane& plane, int w, int h) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
          static_cast<std::size_t>(x)] =
          plane.data()[static_cast<std::size_t>(y) * plane.stride +
                       static_cast<std::size_t>(x)];
    }
  }
  return out;
}

// Verbatim transcription of the private
// VideoFrame::block_variance_cost_at() loop (pipeline/video/video_frame.cpp).
// The function cannot be called directly, so the oracle is copied exactly:
// row-major int64 sum, double mean, row-major double SSE, int64 truncation.
std::int64_t reference_variance(const std::uint8_t* src, std::size_t stride,
                                int w, int h) {
  if (w <= 0 || h <= 0) return 0;
  std::int64_t sum = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      sum += static_cast<std::int64_t>(
          src[static_cast<std::size_t>(y) * stride +
              static_cast<std::size_t>(x)]);
    }
  }
  const double mean = static_cast<double>(sum) /
                      static_cast<double>(w * h);
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double d =
          static_cast<double>(src[static_cast<std::size_t>(y) * stride +
                                  static_cast<std::size_t>(x)]) -
          mean;
      sse += d * d;
    }
  }
  return static_cast<std::int64_t>(sse);
}

// Verbatim transcription of the SSE half of intra::prediction_cost
// (codec/intra/modes/intra_modes.cpp).  The SAD half of that function is the
// `sad` kernel; `prediction_cost` now takes both from `sad_sse`.
std::uint64_t reference_sse_intra(const std::uint8_t* src,
                                  std::size_t src_stride,
                                  const std::uint8_t* pred,
                                  std::size_t pred_stride, int w, int h) {
  std::uint64_t sse = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int diff =
          static_cast<int>(src[static_cast<std::size_t>(i) * src_stride +
                               static_cast<std::size_t>(j)]) -
          static_cast<int>(pred[static_cast<std::size_t>(i) * pred_stride +
                                static_cast<std::size_t>(j)]);
      const int magnitude = diff < 0 ? -diff : diff;
      sse += static_cast<std::uint64_t>(magnitude * magnitude);
    }
  }
  return sse;
}

// Verbatim transcription of dip::prediction_cost8
// (codec/intra/dip/dip.cpp): int64 diff, uint64 magnitude, diff*diff.
std::uint64_t reference_sse_dip(const std::uint8_t* src,
                                std::size_t src_stride,
                                const std::uint8_t* pred,
                                std::size_t pred_stride, int w, int h) {
  std::uint64_t sse = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::int64_t diff =
          static_cast<std::int64_t>(
              src[static_cast<std::size_t>(y) * src_stride +
                  static_cast<std::size_t>(x)]) -
          static_cast<std::int64_t>(
              pred[static_cast<std::size_t>(y) * pred_stride +
                   static_cast<std::size_t>(x)]);
      sse += static_cast<std::uint64_t>(diff * diff);
    }
  }
  return sse;
}

// Verbatim copy of hbd::hbd_sad_at's body (pipeline/video/hbd/video_hbd_ops.h)
// before the kernel migration.  hbd_sad_at now calls `sad_u16_clamped` in the
// product build, so the test keeps the original loop as the oracle.
std::int64_t reference_sad16_clamped(const std::uint16_t* cur, int cur_stride,
                                     const std::uint16_t* ref, int ref_stride,
                                     int x, int y, int w, int h, int mv_y,
                                     int mv_x, int ref_w, int ref_h) {
  auto clip3 = [](int low, int high, int value) {
    return value < low ? low : (value > high ? high : value);
  };
  std::int64_t sad = 0;
  for (int r = 0; r < h; ++r) {
    const int ry = clip3(0, ref_h - 1, y + r + mv_y);
    for (int c = 0; c < w; ++c) {
      const int rx = clip3(0, ref_w - 1, x + c + mv_x);
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

// Verbatim copy of hbd::hbd_variance_cost's body
// (pipeline/video/hbd/video_hbd_ops.h) before the kernel migration.
std::int64_t reference_variance16(const std::uint16_t* src, int stride, int x,
                                  int y, int w, int h) {
  if (w <= 0 || h <= 0) return 0;
  std::int64_t sum = 0;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      sum += src[static_cast<std::size_t>(y + r) *
                     static_cast<std::size_t>(stride) +
                 static_cast<std::size_t>(x + c)];
    }
  }
  const double mean =
      static_cast<double>(sum) / static_cast<double>(w * h);
  double sse = 0.0;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const double d =
          static_cast<double>(src[static_cast<std::size_t>(y + r) *
                                      static_cast<std::size_t>(stride) +
                                  static_cast<std::size_t>(x + c)]) -
          mean;
      sse += d * d;
    }
  }
  return static_cast<std::int64_t>(sse);
}

void check_sad(const av2enc::primitives::KernelTable& k, const TestPlane& src,
               const TestPlane& ref, int w, int h) {
  const std::uint64_t got =
      k.sad(src.data(), src.stride, ref.data(), ref.stride, w, h);

  const auto cur = me_plane(src, w, h);
  const auto rp = me_plane(ref, w, h);
  const std::int32_t me =
      av2enc::me::me_sad_rows(cur, rp, 0, 0, w, 0, 0, 0, h, 1);
  CHECK(got == static_cast<std::uint64_t>(me));

  const std::int64_t ibc = av2enc::ibc::sad_block(
      src.data(), static_cast<int>(src.stride), ref.data(),
      static_cast<int>(ref.stride), w, h);
  CHECK(got == static_cast<std::uint64_t>(ibc));

  const std::uint64_t rdo =
      av2enc::rdo::sad_u8(pack_block(src, w, h), pack_block(ref, w, h), w,
                          h, w);
  CHECK(got == rdo);

  ++g_stats.blocks;
  g_stats.samples += static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
}

void check_sad_sse(const av2enc::primitives::KernelTable& k,
                   const TestPlane& src, const TestPlane& ref, int w, int h) {
  // Poison both outputs so a kernel that forgets one is caught.
  std::uint64_t sad = 0xDEADBEEFULL;
  std::uint64_t sse = 0xFEEDFACEULL;
  k.sad_sse(src.data(), src.stride, ref.data(), ref.stride, w, h, &sad, &sse);

  // SAD: the same three independent oracles check_sad() uses.
  const auto cur = me_plane(src, w, h);
  const auto rp = me_plane(ref, w, h);
  const std::int32_t me =
      av2enc::me::me_sad_rows(cur, rp, 0, 0, w, 0, 0, 0, h, 1);
  CHECK(sad == static_cast<std::uint64_t>(me));

  const std::int64_t ibc = av2enc::ibc::sad_block(
      src.data(), static_cast<int>(src.stride), ref.data(),
      static_cast<int>(ref.stride), w, h);
  CHECK(sad == static_cast<std::uint64_t>(ibc));

  const std::uint64_t rdo =
      av2enc::rdo::sad_u8(pack_block(src, w, h), pack_block(ref, w, h), w,
                          h, w);
  CHECK(sad == rdo);

  // SSE: both call-site loops, transcribed verbatim.
  const std::uint64_t sse_intra = reference_sse_intra(
      src.data(), src.stride, ref.data(), ref.stride, w, h);
  const std::uint64_t sse_dip =
      reference_sse_dip(src.data(), src.stride, ref.data(), ref.stride, w, h);
  CHECK(sse == sse_intra);
  CHECK(sse == sse_dip);
  // The two live loops really do agree for 8-bit inputs, which is why one
  // fused kernel can serve both call sites.
  CHECK(sse_intra == sse_dip);

  ++g_stats.blocks;
  ++g_stats.sad_sse_blocks;
  g_stats.samples += static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
  g_stats.comparisons += 6;
}

// Degenerate shapes must zero both outputs, matching the w <= 0 || h <= 0
// guards in both call sites (they leave PredictionCost{} untouched).
void check_sad_sse_degenerate(const av2enc::primitives::KernelTable& k, int w,
                              int h) {
  const std::array<std::uint8_t, 16> buf{};
  std::uint64_t sad = 7;
  std::uint64_t sse = 9;
  k.sad_sse(buf.data(), 4, buf.data(), 4, w, h, &sad, &sse);
  CHECK(sad == 0);
  CHECK(sse == 0);
  g_stats.comparisons += 2;
}

// `sad == nullptr || sse == nullptr` returns without writing *either* output.
// Check the untouched side as well as the no-crash behaviour of (null, null).
void check_sad_sse_null(const av2enc::primitives::KernelTable& k) {
  const std::array<std::uint8_t, 16> buf{};
  std::uint64_t out = 111;
  k.sad_sse(buf.data(), 4, buf.data(), 4, 4, 4, nullptr, &out);
  CHECK(out == 111);
  k.sad_sse(buf.data(), 4, buf.data(), 4, 4, 4, &out, nullptr);
  CHECK(out == 111);
  k.sad_sse(buf.data(), 4, buf.data(), 4, 4, 4, nullptr, nullptr);
  k.sad_sse(buf.data(), 4, buf.data(), 4, 0, 0, nullptr, &out);
  CHECK(out == 111);
  g_stats.comparisons += 3;
}

// The proof obligation for a SIMD slot: both outputs equal the scalar table
// on the same shape, strides and (unaligned) pointers.  Values are compared
// directly here as well as against the independent oracles above, so a
// mistake shared by kernel and oracle cannot pass.
void check_sad_sse_pair(const av2enc::primitives::KernelTable& k,
                        const TestPlane& src, const TestPlane& ref, int w,
                        int h) {
  if (&k == &av2enc::primitives::scalar_kernels()) return;

  std::uint64_t want_sad = 0xDEADBEEFULL;
  std::uint64_t want_sse = 0xFEEDFACEULL;
  av2enc::primitives::scalar_kernels().sad_sse(
      src.data(), src.stride, ref.data(), ref.stride, w, h, &want_sad,
      &want_sse);

  std::uint64_t got_sad = 0xDEADBEEFULL;
  std::uint64_t got_sse = 0xFEEDFACEULL;
  k.sad_sse(src.data(), src.stride, ref.data(), ref.stride, w, h, &got_sad,
            &got_sse);

  CHECK(got_sad == want_sad);
  CHECK(got_sse == want_sse);
  ++g_stats.sad_sse_pairs;
  g_stats.comparisons += 2;
}

// 1..17 in both dimensions: every remainder of the 16-wide vector loop
// (w % 16 = 0..15), every scalar-tail-only width (w < 16), w = h = 1, and
// every height.  Each case gets its own padding and non-zero pointer offset so
// the unaligned loads and the row stride are not accidentally covered by the
// main sweep.
void test_sad_sse_small_shapes(const av2enc::primitives::KernelTable& k) {
  const std::array<int, 5> pads = {0, 1, 3, 5, 7};
  const std::array<std::size_t, 5> offsets = {0, 1, 2, 3, 5};
  std::uint64_t seed = 0xC2B2AE3D27D4EB4FULL;
  int index = 0;
  for (int w = 1; w <= 17; ++w) {
    for (int h = 1; h <= 17; ++h, ++index) {
      const std::size_t selector = static_cast<std::size_t>(index);
      const int src_pad = pads[selector % pads.size()];
      const int ref_pad = pads[(selector + 2) % pads.size()];
      const std::size_t src_off = offsets[selector % offsets.size()];
      const std::size_t ref_off = offsets[(selector + 3) % offsets.size()];
      Rng src_rng(seed + static_cast<std::uint64_t>(index) * 19 + 1);
      Rng ref_rng(seed + static_cast<std::uint64_t>(index) * 23 + 5);
      TestPlane src = make_plane(w, h, src_pad, src_off, Texture::kRandom,
                                 src_rng);
      TestPlane ref = make_plane(w, h, ref_pad, ref_off, Texture::kRandom,
                                 ref_rng);
      check_sad(k, src, ref, w, h);
      check_sad_sse(k, src, ref, w, h);
      check_sad_sse_pair(k, src, ref, w, h);
    }
  }
}

void check_sad_u16(const av2enc::primitives::KernelTable& k,
                   const TestPlane16& cur, const TestPlane16& ref, int x,
                   int y, int w, int h, int mv_y, int mv_x, int ref_w,
                   int ref_h) {
  const std::int64_t got = k.sad_u16_clamped(
      cur.data(), static_cast<int>(cur.stride), ref.data(),
      static_cast<int>(ref.stride), x, y, w, h, mv_y, mv_x, ref_w, ref_h);
  const std::int64_t want = reference_sad16_clamped(
      cur.data(), static_cast<int>(cur.stride), ref.data(),
      static_cast<int>(ref.stride), x, y, w, h, mv_y, mv_x, ref_w, ref_h);
  CHECK(got == want);
  ++g_stats.sad16_blocks;
  ++g_stats.comparisons;
}

// The kernel takes the visible rectangle's origin; the call site adds
// y * stride + x.  Exercise that pointer arithmetic against the original
// coordinate-taking body.
void check_variance_u16(const av2enc::primitives::KernelTable& k,
                        const TestPlane16& plane, int x, int y, int w, int h) {
  const std::int64_t got = k.variance_u16(
      plane.data() + static_cast<std::size_t>(y) * plane.stride +
          static_cast<std::size_t>(x),
      plane.stride, w, h);
  const std::int64_t want = reference_variance16(
      plane.data(), static_cast<int>(plane.stride), x, y, w, h);
  CHECK(got == want);
  ++g_stats.variance16_blocks;
  ++g_stats.comparisons;
}

// End-to-end pin for the two migrated call sites.  `primitives_test` is built
// without AV2ENC_PRIMITIVES, but these two functions live in av2enc_core,
// which is built with it: calling them here executes the production
// `sad_sse` path.  Values must equal SAD-oracle + each old SSE loop.
void check_prediction_cost_sites(const TestPlane& src, const TestPlane& ref,
                                 int w, int h) {
  const std::uint64_t want_sad = static_cast<std::uint64_t>(
      av2enc::me::me_sad_rows(me_plane(src, w, h), me_plane(ref, w, h), 0, 0,
                              w, 0, 0, 0, h, 1));
  const std::uint64_t want_sse_intra =
      reference_sse_intra(src.data(), src.stride, ref.data(), ref.stride, w, h);
  const std::uint64_t want_sse_dip =
      reference_sse_dip(src.data(), src.stride, ref.data(), ref.stride, w, h);

  const auto intra = av2enc::intra::prediction_cost(
      src.data(), src.stride, ref.data(), ref.stride, w, h);
  CHECK(intra.sad == want_sad);
  CHECK(intra.sse == want_sse_intra);

  const auto dip = av2enc::dip::prediction_cost8(
      src.data(), src.stride, ref.data(), ref.stride, w, h);
  CHECK(dip.sad == want_sad);
  CHECK(dip.sse == want_sse_dip);

  ++g_stats.call_site_blocks;
  g_stats.comparisons += 4;
}

void check_satd(const av2enc::primitives::KernelTable& k, const TestPlane& src,
                const TestPlane& ref, int w, int h) {

  const std::uint64_t satd8 = k.satd(src.data(), src.stride, ref.data(),
                                     ref.stride, w, h, SatdTile::k8x8);
  const auto cur = me_plane(src, w, h);
  const auto rp = me_plane(ref, w, h);
  const std::int32_t me = av2enc::me::me_cost_plane(
      cur, rp, 0, 0, w, h, 0, 0, av2enc::me::kMeSatd, 0);
  CHECK(satd8 == static_cast<std::uint64_t>(me));

  const std::uint64_t satd4 = k.satd(src.data(), src.stride, ref.data(),
                                     ref.stride, w, h, SatdTile::k4x4);
  const std::uint64_t rdo = av2enc::rdo::satd_u8(
      pack_block(src, w, h), pack_block(ref, w, h), w, h, w);
  CHECK(satd4 == rdo);

  g_stats.blocks += 2;
  g_stats.samples +=
      2 * static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
}

void check_variance(const av2enc::primitives::KernelTable& k,
                    const TestPlane& src, int w, int h) {
  const std::int64_t got = k.variance(src.data(), src.stride, w, h);
  CHECK(got == reference_variance(src.data(), src.stride, w, h));

  // hbd_variance_cost has the same int64-sum/double-mean/double-SSE formula
  // over a uint16 container.  Samples are 0..255, so the values and the
  // double sequence are identical.
  std::vector<std::uint16_t> wide(static_cast<std::size_t>(w) *
                                  static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      wide[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
           static_cast<std::size_t>(x)] =
          src.data()[static_cast<std::size_t>(y) * src.stride +
                     static_cast<std::size_t>(x)];
    }
  }
  CHECK(got == av2enc::video::hbd::hbd_variance_cost(
                   wide.data(), w, 0, 0, w, h));

  ++g_stats.blocks;
  g_stats.samples += static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
}

void check_block_copy(const av2enc::primitives::KernelTable& k,
                      const TestPlane& src, int w, int h) {
  constexpr std::uint8_t kSentinel = 0xA5;

  TestPlane prim;
  prim.offset = 3;
  prim.stride = static_cast<std::size_t>(w + 5);
  prim.bytes.assign(prim.offset + prim.stride * static_cast<std::size_t>(h) + 8,
                    kSentinel);

  TestPlane reference;
  reference.offset = 3;
  reference.stride = prim.stride;
  reference.bytes.assign(prim.bytes.size(), kSentinel);

  k.block_copy(src.data(), src.stride, prim.data(), prim.stride, w, h);
  av2enc::ibc::copy_plane(src.data(), static_cast<int>(src.stride), reference.data(),
                          static_cast<int>(reference.stride), w, h);

  CHECK(prim.bytes == reference.bytes);
  // The kernel must not write outside the visible block.
  CHECK(prim.bytes[0] == kSentinel);
  CHECK(prim.bytes[prim.offset + prim.stride - 1] == kSentinel);

  ++g_stats.blocks;
  g_stats.samples += static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
}

void check_edge_filter(const av2enc::primitives::KernelTable& k, int size,
                       int strength, std::uint64_t seed) {
  Rng rng(seed);
  av2enc::intra::EdgeRow row;
  for (int i = av2enc::intra::EdgeRow::kMinIndex;
       i <= av2enc::intra::EdgeRow::kMaxIndex; ++i) {
    row[i] = static_cast<std::int16_t>(static_cast<int>(rng.next() & 0x3FFU) -
                                       512);
  }
  std::array<std::int16_t, 140> prim{};
  // `edge` points at the spec derived array: edge[0] == row[-1].
  std::int16_t* edge = prim.data() + 2;
  const std::int16_t before_minus2 = row[av2enc::intra::EdgeRow::kMinIndex];
  const std::int16_t before_last = row[size - 1];
  for (int i = 0; i < size; ++i) {
    edge[i] = row[i - 1];
  }
  const std::int16_t before_minus1 = edge[0];

  k.edge_filter(edge, size, strength);
  av2enc::intra::intra_edge_filter(row, size, strength);

  // edge[i] (i >= 1) is the filtered row[i-1]; edge[0] (row[-1]) is not
  // written by the spec process.
  CHECK(edge[0] == before_minus1);
  for (int i = 1; i < size; ++i) {
    CHECK(edge[i] == row[i - 1]);
  }
  CHECK(row[av2enc::intra::EdgeRow::kMinIndex] == before_minus2);
  CHECK(row[size - 1] == before_last);

  ++g_stats.blocks;
  g_stats.edge_positions += static_cast<std::uint64_t>(size);
}

void check_edge_filter_noops(const av2enc::primitives::KernelTable& k, int size,
                             int strength) {
  av2enc::intra::EdgeRow row;
  std::array<std::int16_t, 140> prim{};
  std::int16_t* edge = prim.data() + 2;
  for (int i = av2enc::intra::EdgeRow::kMinIndex;
       i <= av2enc::intra::EdgeRow::kMaxIndex; ++i) {
    row[i] = static_cast<std::int16_t>(i * 3 - 7);
  }
  for (int i = 0; i < size; ++i) {
    edge[i] = row[i - 1];
  }
  std::array<std::int16_t, 140> snapshot = prim;
  k.edge_filter(edge, size, strength);
  av2enc::intra::intra_edge_filter(row, size, strength);
  CHECK(prim == snapshot);
}

// ---------------------------------------------------------------------------
// Subpel ME SAD - the oracle is the *existing* per-sample implementation in
// codec/inter/me/me_core.h (me_sad_subpel_rows -> me_sample_at ->
// me_interp_sample), left untouched in the `#else` branch of the migration.
// The kernel is a block formulation: the horizontal pass is shared across the
// rows that need it, but every value it produces is the same integer.
// ---------------------------------------------------------------------------
struct SubpelCase {
  int w = 1;
  int h = 1;
  int cur_w = 1;
  int cur_h = 1;
  int ref_w = 1;
  int ref_h = 1;
  int x = 0;
  int y = 0;
  std::int32_t mv_x = 0;
  std::int32_t mv_y = 0;
  int filter = 0;
  int r0 = 0;
  int r1 = 1;
  int rstep = 1;
};

std::int32_t oracle_subpel(const TestPlane& cur, const TestPlane& ref,
                           const SubpelCase& tc) {
  return av2enc::me::me_sad_subpel_rows(
      me_plane(cur, tc.cur_w, tc.cur_h), me_plane(ref, tc.ref_w, tc.ref_h),
      tc.x, tc.y, tc.w, tc.mv_x, tc.mv_y, tc.filter, tc.r0, tc.r1, tc.rstep);
}

// Every 6*16*8 tap: the primitive layer's copy must be the same data as the
// codec's normative table, which is the only thing that makes the two-pass
// arithmetic comparable at all.
void test_subpel_filter_table() {
  for (int f = 0; f <= av2enc::me::kMeMaxInterpFilter; ++f) {
    for (int p = 0; p < 16; ++p) {
      for (int t = 0; t < 8; ++t) {
        CHECK(av2enc::primitives::kSubpelFilters[f][p][t] ==
              av2enc::me::kMeSubpelFilters[f][p][t]);
        ++g_stats.comparisons;
      }
    }
  }
}

void check_subpel(const av2enc::primitives::KernelTable& k,
                  const TestPlane& cur, const TestPlane& ref,
                  const SubpelCase& tc) {
  const std::int32_t want = oracle_subpel(cur, ref, tc);
  const std::int32_t got = k.subpel_sad_rows(
      cur.data(), cur.stride, ref.data(), ref.stride, tc.ref_w, tc.ref_h,
      tc.x, tc.y, tc.w, tc.mv_x, tc.mv_y, tc.filter, tc.r0, tc.r1, tc.rstep);
  CHECK(got == want);

  // Direct table-vs-table comparison as well, so a mistake shared by the
  // oracle and one table cannot hide a difference between the two tables.
  if (&k != &av2enc::primitives::scalar_kernels()) {
    const std::int32_t ref_got =
        av2enc::primitives::scalar_kernels().subpel_sad_rows(
            cur.data(), cur.stride, ref.data(), ref.stride, tc.ref_w, tc.ref_h,
            tc.x, tc.y, tc.w, tc.mv_x, tc.mv_y, tc.filter, tc.r0, tc.r1,
            tc.rstep);
    CHECK(got == ref_got);
    ++g_stats.subpel_pairs;
    ++g_stats.comparisons;
  }

  int rows = 0;
  for (int r = tc.r0; r < tc.r1; r += tc.rstep) ++rows;
  ++g_stats.subpel_blocks;
  g_stats.subpel_samples +=
      static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(tc.w);
  ++g_stats.comparisons;
}

// One case = one (plane geometry, block origin, MV, filter, row range) tuple.
// Planes are rebuilt per case so the padding and pointer offset rotate too.
void run_subpel_case(const av2enc::primitives::KernelTable& k,
                     const SubpelCase& tc, std::uint64_t seed) {
  static const std::array<int, 4> pads = {0, 1, 3, 7};
  static const std::array<std::size_t, 5> offsets = {0, 1, 2, 3, 5};
  const int sel =
      static_cast<int>((tc.w * 7 + tc.h * 5 + tc.x * 3 + tc.y) & 15);
  const int cur_pad = pads[static_cast<std::size_t>(sel) % pads.size()];
  const int ref_pad = pads[static_cast<std::size_t>(sel + 2) % pads.size()];
  const std::size_t cur_off =
      offsets[static_cast<std::size_t>(sel) % offsets.size()];
  const std::size_t ref_off =
      offsets[static_cast<std::size_t>(sel + 3) % offsets.size()];
  Rng cur_rng(seed * 17 + 1);
  Rng ref_rng(seed * 31 + 7);
  const auto texture = static_cast<Texture>(sel % 4);
  TestPlane cur =
      make_plane(tc.cur_w, tc.cur_h, cur_pad, cur_off, texture, cur_rng);
  TestPlane ref =
      make_plane(tc.ref_w, tc.ref_h, ref_pad, ref_off, Texture::kRandom,
                 ref_rng);
  check_subpel(k, cur, ref, tc);
}

// (a) Every filter, every fractional phase, and the integer shortcut, with
// integer offsets that force left/top and right/bottom tap clamping.
void test_subpel_phases(const av2enc::primitives::KernelTable& k) {
  struct Offset {
    int ox;
    int oy;
  };
  // (0,0) is inside the plane; the other two push the 8-tap window over the
  // left/top and right/bottom edges respectively.
  static const std::array<Offset, 3> offs = {{{0, 0}, {-11, 40}, {40, -6}}};
  std::uint64_t seed = 0x51ED2701C3B4A5F7ULL;
  for (int filter = 0; filter <= av2enc::me::kMeMaxInterpFilter; ++filter) {
    for (int fx = 0; fx < 16; ++fx) {
      for (int fy = 0; fy < 16; ++fy) {
        const bool integer_mv = (fx | fy) == 0;
        const std::size_t n_off = integer_mv ? 1 : offs.size();
        for (std::size_t oi = 0; oi < n_off; ++oi) {
          SubpelCase tc;
          tc.w = 16;
          tc.h = 16;
          tc.x = 11;
          tc.y = 7;
          tc.cur_w = tc.x + tc.w;
          tc.cur_h = tc.y + tc.h;
          // 64x56 keeps every fractional integer offset in a live row/column
          // while the taps still clamp; the integer shortcut must stay inside
          // the reference because it does not clamp.
          tc.ref_w = 64;
          tc.ref_h = 56;
          tc.mv_x = fx + offs[oi].ox * 16;
          tc.mv_y = fy + offs[oi].oy * 16;
          tc.filter = filter;
          tc.r0 = 0;
          tc.r1 = tc.h;
          tc.rstep = 1;
          run_subpel_case(k, tc,
                          seed + static_cast<std::uint64_t>(filter) * 4099 +
                              static_cast<std::uint64_t>(fx) * 257 +
                              static_cast<std::uint64_t>(fy) * 17 + oi);
        }
      }
    }
  }
}

// (b) Every block size the encoder uses (plus odd leaves) at the left/top
// edge, in the middle, flush right/bottom and starting on the last row/column,
// with the integer shortcut kept in bounds (it reads unclamped).
void test_subpel_sizes(const av2enc::primitives::KernelTable& k) {
  const std::array<int, 15> sizes = {
      1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 32, 33, 48, 64};
  const int ref_w = 72;
  const int ref_h = 72;
  std::uint64_t seed = 0x9F2C8B1D4E6A0375ULL;
  int shape = 0;
  for (int w : sizes) {
    for (int h : sizes) {
      const std::array<std::pair<int, int>, 6> positions = {{
          {0, 0}, {5, 3}, {36, 36}, {ref_w - w, ref_h - h},
          {ref_w - 1, ref_h - 1}, {ref_w - 1, 0}}};
      int pos_index = 0;
      for (const auto& [x, y] : positions) {
        SubpelCase base;
        base.w = w;
        base.h = h;
        base.x = x;
        base.y = y;
        base.cur_w = x + w;
        base.cur_h = y + h;
        base.ref_w = ref_w;
        base.ref_h = ref_h;
        base.filter = (shape * 7 + pos_index * 5) % 6;
        base.r0 = 0;
        base.r1 = h;
        base.rstep = 1;

        // Integer-aligned positions that stay inside the reference: top-left
        // and bottom-right, which is the "position at ref_w/ref_h - 1" case.
        for (int corner = 0; corner < 2; ++corner) {
          SubpelCase tc = base;
          tc.mv_x = corner == 0 ? -x * 16 : (ref_w - x - w) * 16;
          tc.mv_y = corner == 0 ? -y * 16 : (ref_h - y - h) * 16;
          run_subpel_case(k, tc,
                          seed + static_cast<std::uint64_t>(shape) * 101 +
                              static_cast<std::uint64_t>(pos_index) * 7 +
                              static_cast<std::uint64_t>(corner));
        }
        // Two fractional phases, positive and negative integer parts, both
        // axes clamping somewhere in the tap window.
        for (int variant = 0; variant < 2; ++variant) {
          SubpelCase tc = base;
          const int fx = variant == 0 ? 3 : 15;
          const int fy = variant == 0 ? 5 : 1;
          tc.mv_x = fx + (variant == 0 ? -3 : 2) * 16;
          tc.mv_y = fy + (variant == 0 ? 2 : -3) * 16;
          run_subpel_case(k, tc,
                          seed + static_cast<std::uint64_t>(shape) * 211 +
                              static_cast<std::uint64_t>(pos_index) * 11 +
                              static_cast<std::uint64_t>(variant) + 3);
        }
        ++pos_index;
      }
      ++shape;
    }
  }
}

// (c) Row subsets: rstep 1 (CPU), rstep 2, and the CUDA warp shape
// (r0 = lane, rstep = 32), plus empty ranges.
void test_subpel_row_ranges(const av2enc::primitives::KernelTable& k) {
  std::uint64_t seed = 0x1B0E5A3C7D9F2468ULL;
  for (int filter = 0; filter <= av2enc::me::kMeMaxInterpFilter; ++filter) {
    for (int variant = 0; variant < 3; ++variant) {
      SubpelCase base;
      base.w = 32;
      base.h = 48;
      base.x = 7;
      base.y = 5;
      base.cur_w = base.x + base.w;
      base.cur_h = base.y + base.h;
      base.ref_w = 96;
      base.ref_h = 80;
      base.filter = filter;
      base.mv_x = (variant == 0 ? 3 : 15) + (variant == 2 ? -16 : 0);
      base.mv_y = 6 + (variant == 1 ? 16 : 0);

      // Whole block, every other row, every other row offset by one.
      for (int start : {0, 1}) {
        SubpelCase tc = base;
        tc.r0 = start;
        tc.r1 = base.h;
        tc.rstep = 2;
        run_subpel_case(k, tc, seed + static_cast<std::uint64_t>(filter) * 37 +
                                   static_cast<std::uint64_t>(variant) * 13 +
                                   static_cast<std::uint64_t>(start));
      }
      SubpelCase whole = base;
      whole.r0 = 0;
      whole.r1 = base.h;
      whole.rstep = 1;
      run_subpel_case(k, whole, seed + static_cast<std::uint64_t>(filter) * 41 +
                                    static_cast<std::uint64_t>(variant));

      // Empty ranges are 0 in both implementations.
      SubpelCase empty = base;
      empty.r0 = base.h;
      empty.r1 = base.h;
      empty.rstep = 1;
      run_subpel_case(k, empty, seed + 1);

      // CUDA warp shape: one lane per r0, rows strided by 32.
      for (int lane = 0; lane < 32; ++lane) {
        SubpelCase tc = base;
        tc.r0 = lane;
        tc.r1 = base.h;
        tc.rstep = 32;
        run_subpel_case(k, tc, seed + static_cast<std::uint64_t>(filter) * 53 +
                                   static_cast<std::uint64_t>(variant) * 19 +
                                   static_cast<std::uint64_t>(lane));
      }
    }
  }

  // Degenerate widths produce 0 in both, and do not read a plane.
  for (int w : {0, -1, -8}) {
    SubpelCase tc;
    tc.w = w;
    tc.h = 4;
    tc.r1 = 4;
    run_subpel_case(k, tc, 1);
  }
}

// End-to-end: av2enc_core's me.cpp cost path (MeKernelCost -> me_cost_plane ->
// the migrated me_sad_subpel_rows) must return the oracle's cost for a
// fractional MV under every filter.
void check_me_cost_block_site(const TestPlane& cur, const TestPlane& ref,
                              const SubpelCase& tc) {
  av2enc::me::MeCoreFrame frame;
  frame.plane[0] = me_plane(cur, tc.cur_w, tc.cur_h);
  frame.width = tc.cur_w;
  frame.height = tc.cur_h;
  av2enc::me::MeCoreRef refs[1];
  refs[0].plane[0] = me_plane(ref, tc.ref_w, tc.ref_h);
  av2enc::me::MeCostJob job;
  job.x = tc.x;
  job.y = tc.y;
  job.w = tc.w;
  job.h = tc.h;
  job.mv_x = tc.mv_x;
  job.mv_y = tc.mv_y;
  job.metric = av2enc::me::kMeSad;
  job.interp = tc.filter;
  std::int32_t cost = -1;
  const auto status = av2enc::me::me_cost_block(frame, refs[0], job, &cost);
  CHECK(status == av2enc::me::MeStatus::kOk);
  CHECK(cost == oracle_subpel(cur, ref, tc));
  ++g_stats.call_site_blocks;
  g_stats.comparisons += 2;
}

void test_subpel_call_site() {
  std::uint64_t seed = 0x77C1A95E3B0D2846ULL;
  int index = 0;
  for (int filter = 0; filter <= av2enc::me::kMeMaxInterpFilter; ++filter) {
    for (int phase = 1; phase < 16; phase += 3) {
      SubpelCase tc;
      tc.w = 16;
      tc.h = 16;
      tc.x = 9;
      tc.y = 6;
      tc.cur_w = 40;
      tc.cur_h = 40;
      tc.ref_w = 48;
      tc.ref_h = 48;
      tc.mv_x = phase;            // in bounds: -9*16-15 .. (48-9-16)*16+15
      tc.mv_y = 16 - phase;
      tc.filter = filter;
      tc.r0 = 0;
      tc.r1 = tc.h;
      tc.rstep = 1;
      Rng cur_rng(seed + static_cast<std::uint64_t>(index) * 13 + 1);
      Rng ref_rng(seed + static_cast<std::uint64_t>(index) * 29 + 5);
      TestPlane cur = make_plane(tc.cur_w, tc.cur_h, 3, 1, Texture::kRandom,
                                 cur_rng);
      TestPlane ref = make_plane(tc.ref_w, tc.ref_h, 5, 2, Texture::kRandom,
                                 ref_rng);
      check_me_cost_block_site(cur, ref, tc);
      ++index;
    }
  }
}

// True when this binary contains a NEON table.  Derived from the advertised
// targets rather than a local #if so the test also proves the advertisement.
bool have_neon_table() {
  return av2enc::primitives::available_targets().find("neon") !=
         std::string::npos;
}

void test_dispatch() {
  using av2enc::primitives::KernelTable;
  using av2enc::primitives::available_targets;
  using av2enc::primitives::intro;
  using av2enc::primitives::kernels;
  using av2enc::primitives::scalar_kernels;
  using av2enc::primitives::select_kernels;
  using av2enc::primitives::selected_target;

  const bool neon_built = have_neon_table();
#if defined(AV2ENC_PRIMITIVES_HAVE_NEON)
  CHECK(neon_built);
#else
  CHECK(!neon_built);
#endif

  // Clear the override before the first kernels() call: the start-up snapshot
  // is taken once, so this asserts the documented default (auto).
  (void)::unsetenv("AV2ENC_SIMD");
  const KernelTable& scalar = scalar_kernels();
  const KernelTable& table = kernels();
  const char* default_name = neon_built ? "neon" : "scalar";
  CHECK(std::strcmp(table.name, default_name) == 0);
  CHECK(selected_target() == default_name);

  // `auto` (and the empty value, which means auto) prefers the SIMD table.
  for (const char* requested : {"", "auto"}) {
    const KernelTable& selected = select_kernels(requested);
    CHECK(std::strcmp(selected.name, default_name) == 0);
    CHECK(selected.sad_sse == table.sad_sse);
  }

  // AV2ENC_SIMD=scalar is the oracle and is never overridden - on AArch64 too,
  // where the neon table exists.  select_kernels() is the resolver behind the
  // start-up snapshot, so this is deterministic regardless of process env.
  const KernelTable& forced_scalar = select_kernels("scalar");
  CHECK(std::strcmp(forced_scalar.name, "scalar") == 0);
  CHECK(forced_scalar.sad == scalar.sad);
  CHECK(forced_scalar.sad_sse == scalar.sad_sse);
  CHECK(forced_scalar.satd == scalar.satd);
  CHECK(forced_scalar.variance == scalar.variance);
  CHECK(forced_scalar.block_copy == scalar.block_copy);
  CHECK(forced_scalar.edge_filter == scalar.edge_filter);
  CHECK(forced_scalar.sad_u16_clamped == scalar.sad_u16_clamped);
  CHECK(forced_scalar.variance_u16 == scalar.variance_u16);
  CHECK(forced_scalar.subpel_sad_rows == scalar.subpel_sad_rows);

  // An explicit SIMD request selects it when built and falls back to scalar
  // (never null, never a crash) when it is not.
  if (neon_built) {
    CHECK(std::strcmp(select_kernels("neon").name, "neon") == 0);
  } else {
    const KernelTable& fallback = select_kernels("neon");
    CHECK(std::strcmp(fallback.name, "scalar") == 0);
    CHECK(fallback.sad_sse == scalar.sad_sse);
  }
  // Unbuilt/unknown names always fall back to scalar.
  for (const char* requested : {"avx2", "bogus"}) {
    const KernelTable& selected = select_kernels(requested);
    CHECK(std::strcmp(selected.name, "scalar") == 0);
    CHECK(selected.sad == scalar.sad);
  }

  // available_targets() lists scalar always and neon only when built.
  CHECK(available_targets().find("scalar") != std::string::npos);
  CHECK((available_targets().find("neon") != std::string::npos) ==
        neon_built);

  // The default table overrides only proven slots; everything else must still
  // be the scalar function, never null.
  CHECK(table.sad == scalar.sad);
  CHECK(table.satd == scalar.satd);
  CHECK(table.variance == scalar.variance);
  CHECK(table.block_copy == scalar.block_copy);
  CHECK(table.edge_filter == scalar.edge_filter);
  CHECK(table.sad_u16_clamped == scalar.sad_u16_clamped);
  CHECK(table.variance_u16 == scalar.variance_u16);
  if (neon_built) {
    CHECK(table.subpel_sad_rows != scalar.subpel_sad_rows);
  } else {
    CHECK(table.subpel_sad_rows == scalar.subpel_sad_rows);
  }
  if (neon_built) {
    CHECK(table.sad_sse != scalar.sad_sse);
  } else {
    CHECK(table.sad_sse == scalar.sad_sse);
  }

#if defined(AV2ENC_PRIMITIVES_HAVE_NEON)
  // The NEON table is built as a copy of scalar plus overrides, and the
  // default selection is that exact object.
  CHECK(&table == &av2enc::primitives::neon_kernels());
  CHECK(av2enc::primitives::neon_kernels().sad_sse != scalar.sad_sse);
  CHECK(av2enc::primitives::neon_kernels().subpel_sad_rows !=
        scalar.subpel_sad_rows);
#endif

  CHECK(!intro().empty());
  CHECK(intro().find(default_name) != std::string::npos);
  CHECK(intro().find("scalar") != std::string::npos);
}

void test_edge_filter_strength() {
  const std::array<std::pair<int, int>, 18> sizes = {{
      {1, 1},   {2, 2},   {4, 4},   {4, 8},   {8, 4},   {8, 8},
      {8, 16},  {16, 8},  {16, 16}, {16, 32}, {32, 16}, {32, 32},
      {32, 64}, {64, 32}, {64, 64}, {4, 16},  {2, 8},   {12, 12},
  }};
  for (const auto& [w, h] : sizes) {
    for (bool smooth : {false, true}) {
      for (int delta = -90; delta <= 90; ++delta) {
        CHECK(av2enc::primitives::intra_edge_filter_strength(w, h, smooth,
                                                             delta) ==
              av2enc::intra::edge_filter_strength(w, h, smooth, delta));
      }
    }
  }
}

void test_all_kernels(const av2enc::primitives::KernelTable& k) {
  // 1, 2, 3 are the degenerate/edge cases; the rest span the encoder's
  // 4/8/16/32/64 sizes and odd rectangular leaves.
  const std::array<int, 16> sizes = {
      1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 24, 31, 32, 33, 48, 64};
  const std::array<int, 4> pads = {0, 1, 3, 7};
  const std::array<std::size_t, 5> offsets = {0, 1, 2, 3, 5};

  std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
  int case_index = 0;
  // Two passes over the full size cross-product so every shape sees two
  // different (src, ref) stride/offset combinations.
  for (int pass = 0; pass < 2; ++pass) {
    for (int w : sizes) {
      for (int h : sizes) {
        const std::size_t selector =
            static_cast<std::size_t>(case_index + 2 * pass);
        const int src_pad = pads[selector % pads.size()];
        const int ref_pad = pads[(selector + 1) % pads.size()];
        const std::size_t src_off = offsets[selector % offsets.size()];
        const std::size_t ref_off =
            offsets[(selector + 2) % offsets.size()];
        const auto texture = static_cast<Texture>(case_index % 4);

        Rng src_rng(seed + static_cast<std::uint64_t>(case_index) * 17 +
                    static_cast<std::uint64_t>(pass) * 1009 + 1);
        Rng ref_rng(seed + static_cast<std::uint64_t>(case_index) * 31 +
                    static_cast<std::uint64_t>(pass) * 1013 + 7);
        TestPlane src = make_plane(w, h, src_pad, src_off, texture, src_rng);
        TestPlane ref = make_plane(w, h, ref_pad, ref_off, Texture::kRandom,
                                   ref_rng);

        check_sad(k, src, ref, w, h);
        check_sad_sse(k, src, ref, w, h);
        check_sad_sse_pair(k, src, ref, w, h);
        // The migrated production call sites always go through kernels();
        // run them once, in the pass whose table is the default.
        if (&k == &av2enc::primitives::kernels()) {
          check_prediction_cost_sites(src, ref, w, h);
        }
        check_satd(k, src, ref, w, h);
        check_variance(k, src, w, h);
        check_block_copy(k, src, w, h);

        ++case_index;
      }
    }
  }

  // Degenerate shapes: both fused outputs are zero, and a null output pointer
  // suppresses both writes.
  for (int w : {0, -1}) {
    for (int h : {4, 0}) {
      check_sad_sse_degenerate(k, w, h);
    }
  }
  check_sad_sse_null(k);

  // --- subpel ME SAD (fractional MVs) --------------------------------------
  test_subpel_phases(k);
  test_subpel_sizes(k);
  test_subpel_row_ranges(k);

  // --- 16-bit (hbd) kernels ------------------------------------------------
  // 10-bit samples (0..1023), padded/offset planes, block at (2,1) inside a
  // wider plane, and a reference extent that is deliberately smaller than the
  // block plus MV so every clip path (top/left/right/bottom) is exercised.
  const std::array<int, 14> u16_sizes = {1,  2,  3,  4,  5,  6, 7,
                                         8,  12, 16, 17, 24, 31, 32};
  const std::array<std::pair<int, int>, 4> mvs = {
      {{0, 0}, {2, -3}, {-1, 4}, {-9, -7}}};
  int u16_case = 0;
  int var_case = 0;
  for (int pass = 0; pass < 2; ++pass) {
    for (int w : u16_sizes) {
      for (int h : u16_sizes) {
        const std::size_t selector =
            static_cast<std::size_t>(u16_case + 3 * pass);
        const int pad = pads[selector % pads.size()];
        const std::size_t off = offsets[selector % offsets.size()];
        Rng rng(seed + static_cast<std::uint64_t>(u16_case) * 41 +
                static_cast<std::uint64_t>(pass) * 1021 + 11);
        const int x = 2;
        const int y = 1;
        TestPlane16 cur = make_plane16(w + 6, h + 6, pad, off, 10, rng);
        const int ref_w = w + 2;
        const int ref_h = h + 1;
        TestPlane16 ref =
            make_plane16(ref_w, ref_h, pad, off + 1, 10, rng);
        for (const auto& [mv_y, mv_x] : mvs) {
          check_sad_u16(k, cur, ref, x, y, w, h, mv_y, mv_x, ref_w, ref_h);
        }
        // Fully outside the reference: every sample clamps to the top-left
        // corner (reference replication), the extreme clipping case.
        check_sad_u16(k, cur, ref, x, y, w, h, -(h + 2), -(w + 2), ref_w,
                      ref_h);

        // Variance over a rectangle at a non-zero origin in its plane.
        Rng var_rng(seed + static_cast<std::uint64_t>(u16_case) * 67 +
                    static_cast<std::uint64_t>(pass) * 1031 + 13);
        TestPlane16 wide = make_plane16(w + 6, h + 6, pad, off, 10, var_rng);
        check_variance_u16(k, wide, var_case % 4, (var_case / 4) % 4, w, h);

        ++u16_case;
        ++var_case;
      }
    }
  }

  // Edge filter: every strength over sizes from 2 up to the EdgeRow capacity,
  // including the 4/8/16/32/64 leaves and non-power-of-two numPx values.
  for (int size : {2, 3, 4, 5, 8, 12, 16, 17, 24, 32, 33, 64, 65, 128, 129}) {
    for (int strength = 1; strength <= 3; ++strength) {
      for (std::uint64_t rep = 0; rep < 3; ++rep) {
        check_edge_filter(k, size, strength, seed + static_cast<std::uint64_t>(size) * 7 +
                                               static_cast<std::uint64_t>(strength) * 3 +
                                               rep);
      }
    }
  }
  // Out-of-contract arguments are no-ops, matching intra_edge_filter.
  for (int size : {0, 1, 130}) {
    for (int strength : {0, 1, 2, 3, 4}) {
      check_edge_filter_noops(k, size, strength);
    }
  }
}

}  // namespace

int main() {
  test_dispatch();
  test_edge_filter_strength();
  test_subpel_filter_table();

  // Scalar first: it is the oracle.  Then every other compiled-in table over
  // the identical sweep, so a slot can only land by matching it exactly.
  const av2enc::primitives::KernelTable& scalar =
      av2enc::primitives::scalar_kernels();
  test_all_kernels(scalar);
  test_sad_sse_small_shapes(scalar);
  if (have_neon_table()) {
    const av2enc::primitives::KernelTable& neon =
        av2enc::primitives::select_kernels("neon");
    test_all_kernels(neon);
    test_sad_sse_small_shapes(neon);
  }
  // The migrated production call site (av2enc_core's me.cpp cost path), once.
  test_subpel_call_site();

  std::cout << "primitives_test: kernels compared, target="
            << av2enc::primitives::selected_target()
            << " blocks=" << g_stats.blocks
            << " samples=" << g_stats.samples
            << " edge_positions=" << g_stats.edge_positions
            << " sad_sse_blocks=" << g_stats.sad_sse_blocks
            << " sad_sse_pairs=" << g_stats.sad_sse_pairs
            << " subpel_blocks=" << g_stats.subpel_blocks
            << " subpel_samples=" << g_stats.subpel_samples
            << " subpel_pairs=" << g_stats.subpel_pairs
            << " call_site_blocks=" << g_stats.call_site_blocks
            << " sad16_blocks=" << g_stats.sad16_blocks
            << " variance16_blocks=" << g_stats.variance16_blocks
            << " comparisons=" << g_stats.comparisons << '\n';
  return test::finish("primitives_test");
}
