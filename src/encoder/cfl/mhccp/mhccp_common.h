#ifndef AV2ENC_ENCODER_CFL_MHCCP_COMMON_H_
#define AV2ENC_ENCODER_CFL_MHCCP_COMMON_H_

// MHCCP (multi-hypothesis cross-component prediction) shared core.
//
// AV2 v1.0.0:
//   §7.13.2.1  CflRef capture for cfl_index == CFL_MULTI
//   §7.13.6    MHCCP process
//   §7.13.7    Derive multi param process
//   §7.13.8    Gaussian elimination process
//   §9.2       MHCCP_BITS, DIV_PREC_BITS, DIV_PREC_BITS_POW2, DIV_SLOT_BITS
//
// This header is deliberately free of the standard library so the same POD
// descriptors and integer arithmetic compile for the CPU reference
// (mhccp.cpp), the host-emulation parity test, and the CUDA kernel
// (mhccp_gpu.cu). The CPU path is normative; the device path only changes the
// thread mapping.
//
// Supported geometry: 8-bit 4:2:0 chroma transform blocks with 4 <= w <= 32,
// 4 <= h <= 32 (lossless 4x4 included). That is exactly the is_mhccp_allowed()
// range in §5.20.5.6, and it keeps CflRefWidth/Height below their 128-sample
// cap so CflRef is never truncated for an allowed block.

#include "../cfl_common.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_MHCCP_HD __host__ __device__
#else
#define AV2ENC_MHCCP_HD
#endif

namespace av2enc::cfl::mhccp {

// ---------------------------------------------------------------------------
// §9.2 constants
// ---------------------------------------------------------------------------

inline constexpr int kBits = 16;              // MHCCP_BITS
inline constexpr int kDivPrecBits = 14;       // DIV_PREC_BITS
inline constexpr int kDivPrecBitsPow2 = 8;    // DIV_PREC_BITS_POW2
inline constexpr int kDivSlotBits = 3;        // DIV_SLOT_BITS
inline constexpr int kDirections = 3;         // cfl_mh_dir 0..2
inline constexpr int kHypotheses = 3;         // [luma, luma^2, bias]
inline constexpr int kMaxBlockDim = 32;       // is_mhccp_allowed() cap
inline constexpr int kMaxRefDim = 66;         // CflRef chroma extent cap (2*32+2)

// cfl_index values (§6.19.7.25 Table 6.26). Re-exported for callers that only
// include this header.
inline constexpr int kIndexExplicit = cfl::kIndexExplicit;
inline constexpr int kIndexDerivedAlpha = cfl::kIndexDerivedAlpha;
inline constexpr int kIndexMulti = cfl::kIndexMulti;

// Division_Pow2_W/O/B from §7.13.8. These are functions rather than
// namespace-scope constexpr arrays so nvcc can inline them in device code
// without --expt-relaxed-constexpr.
AV2ENC_MHCCP_HD inline constexpr int division_pow2_w(int index) {
  switch (index) {
    case 0: return 214;
    case 1: return 153;
    case 2: return 113;
    case 3: return 86;
    case 4: return 67;
    case 5: return 53;
    case 6: return 43;
    default: return 35;
  }
}

AV2ENC_MHCCP_HD inline constexpr int division_pow2_o(int index) {
  switch (index) {
    case 0: return 4822;
    case 1: return 5952;
    case 2: return 6624;
    case 3: return 6792;
    case 4: return 6408;
    case 5: return 5424;
    case 6: return 3792;
    default: return 1466;
  }
}

AV2ENC_MHCCP_HD inline constexpr int division_pow2_b(int index) {
  switch (index) {
    case 0: return 12784;
    case 1: return 12054;
    case 2: return 11670;
    case 3: return 11583;
    case 4: return 11764;
    case 5: return 12195;
    case 6: return 12870;
    default: return 13782;
  }
}

// ---------------------------------------------------------------------------
// §4.8 / §7.13.8 integer helpers
// ---------------------------------------------------------------------------

AV2ENC_MHCCP_HD inline constexpr int abs_int(int value) {
  return value < 0 ? -value : value;
}

AV2ENC_MHCCP_HD inline constexpr int max_int(int a, int b) {
  return a > b ? a : b;
}

AV2ENC_MHCCP_HD inline constexpr int min_int(int a, int b) {
  return a < b ? a : b;
}

// GetMsb(x) == FloorLog2(x) for x >= 1, and 0 for x == 0.
AV2ENC_MHCCP_HD inline constexpr int get_msb(int value) {
  return value == 0 ? 0 : cfl::floor_log2(value);
}

// CeilLog2(x) from §4.8; x >= 0, returns 0 for x < 2.
AV2ENC_MHCCP_HD inline constexpr int ceil_log2(int value) {
  if (value < 2) {
    return 0;
  }
  int bits = 1;
  int power = 2;
  while (power < value) {
    ++bits;
    power <<= 1;
  }
  return bits;
}

// Arithmetic right shift, defined for negative values on every host compiler.
AV2ENC_MHCCP_HD inline constexpr std::int64_t asr64(std::int64_t value,
                                                    int shift) {
  if (shift <= 0) {
    return value;
  }
  if (value >= 0) {
    return value >> shift;
  }
  const std::int64_t divisor = static_cast<std::int64_t>(1) << shift;
  return -(((-value) + divisor - 1) >> shift);
}

// Round2(x, n) for a 64-bit value. Equivalent to the §4.8 integer form.
AV2ENC_MHCCP_HD inline constexpr std::int64_t round2_64(std::int64_t value,
                                                        int n) {
  if (n <= 0) {
    return value;
  }
  const std::int64_t half = static_cast<std::int64_t>(1) << (n - 1);
  return (value + half) >> n;
}

// Round2Signed(x, n) for a 64-bit value.
AV2ENC_MHCCP_HD inline constexpr std::int64_t round2_signed_64(
    std::int64_t value, int n) {
  if (value >= 0) {
    return round2_64(value, n);
  }
  return -round2_64(-value, n);
}

// §7.13.8 mul_fixed32_adapt(). `shift` is the fixed-point scale (MHCCP_BITS
// for the model arithmetic).
AV2ENC_MHCCP_HD inline int mul_fixed32_adapt(int a, int b, int shift) {
  const int bits_a = get_msb(abs_int(a)) + 1;
  const int bits_b = get_msb(abs_int(b)) + 1;
  const int need = max_int(0, bits_a + bits_b - 29);
  const int s1 = need >> 1;
  const int s2 = need - s1;
  const int adj = shift - (s1 + s2);
  // Arithmetic shifts: C++20 defines signed right shift as arithmetic, but
  // keep the helper portable for older host compilers too.
  const int aa = static_cast<int>(asr64(static_cast<std::int64_t>(a), s1));
  const int bb = static_cast<int>(asr64(static_cast<std::int64_t>(b), s2));
  const std::int64_t product =
      static_cast<std::int64_t>(aa) * static_cast<std::int64_t>(bb);
  if (adj <= 0) {
    return static_cast<int>(product);
  }
  if (adj > 29) {
    return 0;
  }
  return static_cast<int>(round2_signed_64(product, adj));
}

struct DivisionScaleShift {
  int scale;
  int shift;
};

// §7.13.8 get_division_scale_shift(). denom >= 1.
AV2ENC_MHCCP_HD inline DivisionScaleShift get_division_scale_shift(int denom) {
  const int shift = cfl::floor_log2(denom);
  // The spec's `denom << DIV_PREC_BITS` is computed with unbounded integers;
  // use 64 bits so the intermediate cannot wrap for 8/10/12-bit pivots.
  const std::int64_t scaled =
      round2_64(static_cast<std::int64_t>(denom) << kDivPrecBits, shift);
  int norm_diff =
      cfl::clip3(1, 32767, static_cast<int>(scaled)) &
      ((1 << kDivPrecBits) - 1);
  const int index = norm_diff >> (kDivPrecBits - kDivSlotBits);
  const int norm_diff2 = norm_diff - division_pow2_o(index);
  std::int64_t term =
      (static_cast<std::int64_t>(division_pow2_w(index)) *
       (static_cast<std::int64_t>(norm_diff2) * norm_diff2 >>
        kDivPrecBits)) >>
      kDivPrecBitsPow2;
  term -= static_cast<std::int64_t>(norm_diff2) >> 1;
  term += division_pow2_b(index);
  const int scale =
      static_cast<int>(term) * (1 << (kBits - kDivPrecBits));
  return DivisionScaleShift{scale, shift};
}

// ---------------------------------------------------------------------------
// POD descriptors shared with the CUDA kernel
// ---------------------------------------------------------------------------

// One chroma transform block plus the reconstructed luma/chroma planes. The
// current chroma block holds the DC prediction on input and receives the MHCCP
// prediction on output.
//
// Luma/chroma coordinates are chroma samples; the luma origin is
// (x << sub_x, y << sub_y).
//
// `mi_col_end` / `mi_row_end` are the §7.13.2.1 tile limits in LUMA samples:
// MiColEnd * MI_SIZE and MiRowEnd * MI_SIZE. For a single-tile stream they are
// coded_width << sub_x and coded_height << sub_y of the cfl::Block.
//
// `sb_luma_y` is the top luma row of the enclosing superblock
// (sbRow * Block_Height[SbSize]); §7.13.2.1 uses it for the CflRef capture
// clamp and §7.13.6 for the luma tap limit.
struct Block {
  const std::uint8_t* luma = nullptr;
  int luma_stride = 0;
  int luma_max_x = 0;  // inclusive, reconstructed/padded plane bounds
  int luma_max_y = 0;
  std::uint8_t* chroma = nullptr;
  int chroma_stride = 0;
  int chroma_max_x = 0;  // inclusive, reconstructed/padded plane bounds
  int chroma_max_y = 0;
  int x = 0;  // chroma origin
  int y = 0;
  int w = 0;  // chroma transform block
  int h = 0;
  int bit_depth = 8;
  int ds_filter = 0;  // cfl_ds_filter_index, 0..3 (3 behaves as 0)
  int sub_x = 1;
  int sub_y = 1;
  int have_above = 0;
  int have_left = 0;
  int num4_above_right = 0;
  int num4_below_left = 0;
  int mi_col_end = 0;  // luma samples
  int mi_row_end = 0;  // luma samples
  int sb_luma_y = 0;   // top luma row of the enclosing superblock
};

// The three Q16 weights produced by §7.13.7. No default member initialisers:
// this type is used as a CUDA __shared__ object, and the derivation/apply
// functions always fill all three entries.
struct MultiParams {
  int p[kHypotheses];
};

// CflRef[0] (luma estimates) and CflRef[1] (reconstructed chroma targets) for
// one chroma plane. Only the captured rectangle is meaningful; indices outside
// it are zeroed by capture_refs().
// No default member initialisers: Refs is used as a CUDA __shared__ object.
// capture_refs() sets every field and zeroes the captured rectangle.
struct Refs {
  int above;     // CflAbove: 0 or 2
  int left;      // CflLeft: 0 or 2
  int rows;      // h + CflAbove
  int cols;      // w + CflLeft
  int target_h;  // Round2(CflRefHeight, sub_y)
  int target_w;  // Round2(CflRefWidth, sub_x)
  int ref_h;     // CflRefHeight >> sub_y (derive loop bound)
  int ref_w;     // CflRefWidth >> sub_x
  std::int16_t luma[kMaxRefDim][kMaxRefDim];
  std::int16_t chroma[kMaxRefDim][kMaxRefDim];
};

// Source chroma used by the encoder's direction/parameter decision.
struct SourcePlane {
  const std::uint8_t* plane = nullptr;
  int stride = 0;
  int max_x = 0;  // inclusive
  int max_y = 0;
};

// ---------------------------------------------------------------------------
// §5.20.5.6 is_mhccp_allowed()
// ---------------------------------------------------------------------------

AV2ENC_MHCCP_HD inline constexpr bool allowed(int chroma_w, int chroma_h,
                                              bool enable_mhccp,
                                              bool is_chroma_part,
                                              bool frame_is_intra,
                                              bool cfl_allowed_in_sdp,
                                              bool lossless) {
  if (!enable_mhccp) {
    return false;
  }
  if (is_chroma_part && frame_is_intra && !cfl_allowed_in_sdp) {
    return false;
  }
  if (lossless) {
    return chroma_w == 4 && chroma_h == 4;
  }
  return (chroma_w > 4 || chroma_h > 4) && chroma_w <= 32 &&
         chroma_h <= 32;
}

// Structural validation for the reference and the kernel. Rejects geometry
// outside the MHCCP/cfl_ref capture envelope: 4:2:0 only (4:2:2 and 4:4:4 are
// not encoder formats), 4..32 samples per side.
AV2ENC_MHCCP_HD inline bool valid(const Block& block) {
  if (block.luma == nullptr || block.chroma == nullptr) {
    return false;
  }
  if (block.w < 4 || block.h < 4 || block.w > kMaxBlockDim ||
      block.h > kMaxBlockDim) {
    return false;
  }
  if (block.luma_stride <= 0 || block.chroma_stride <= 0) {
    return false;
  }
  if (block.luma_max_x < 0 || block.luma_max_y < 0 ||
      block.chroma_max_x < 0 || block.chroma_max_y < 0) {
    return false;
  }
  if (block.x < 0 || block.y < 0) {
    return false;
  }
  if (block.chroma_max_x < block.x + block.w - 1 ||
      block.chroma_max_y < block.y + block.h - 1) {
    return false;
  }
  if (block.bit_depth < 8 || block.bit_depth > 12) {
    return false;
  }
  if (block.sub_x != 1 || block.sub_y != 1) {
    return false;
  }
  if (block.ds_filter < 0 || block.ds_filter > 3) {
    return false;
  }
  if (block.have_above != 0 && block.y <= 0) {
    return false;
  }
  if (block.have_left != 0 && block.x <= 0) {
    return false;
  }
  if (block.num4_above_right < 0 || block.num4_below_left < 0) {
    return false;
  }
  if (block.mi_col_end <= 0 || block.mi_row_end <= 0 ||
      block.sb_luma_y < 0) {
    return false;
  }
  // The current transform block must lie inside the tile limits, and the
  // enclosing superblock must start at or above the block's luma origin.
  if (block.mi_col_end < ((block.x + block.w) << block.sub_x) ||
      block.mi_row_end < ((block.y + block.h) << block.sub_y) ||
      block.sb_luma_y > (block.y << block.sub_y)) {
    return false;
  }
  return true;
}

AV2ENC_MHCCP_HD inline constexpr int normalised_filter(const Block& block) {
  return block.ds_filter == 3 ? 0 : block.ds_filter;
}

AV2ENC_MHCCP_HD inline constexpr bool ref_luma_avail(int i, int j, int above,
                                                     int left, int w, int h) {
  return (i < above || j < left + w) && (i < above + h || j < left);
}

// One filtered luma sample from §7.13.2.1 get_cfl_luma_sample().
// `clamp_x` / `clamp_y` are `j == 0` / `i == 0` at the capture site.
AV2ENC_MHCCP_HD inline int cfl_luma_sample(const Block& block, int chroma_x,
                                           int chroma_y, bool clamp_x,
                                           bool clamp_y) {
  const int sub_x = block.sub_x;
  const int sub_y = block.sub_y;
  const int luma_x = chroma_x << sub_x;
  const int luma_y = chroma_y << sub_y;
  const int filter = normalised_filter(block);
  const int limit_luma_y = block.sb_luma_y - 1;

  int t = 0;
  for (int dy = -sub_y; dy <= sub_y; ++dy) {
    const int shifted_y =
        luma_y + (clamp_y ? max_int(0, dy) : dy);
    const int row = cfl::clip3(0, block.luma_max_y,
                               max_int(limit_luma_y, shifted_y));
    for (int dx = -sub_x; dx <= sub_x; ++dx) {
      const int shifted_x = luma_x + (clamp_x ? max_int(0, dx) : dx);
      const int col = cfl::clip3(0, block.luma_max_x, shifted_x);
      const int weight = cfl::ds_weight(filter, dy, dx, sub_x, sub_y);
      if (weight != 0) {
        t += weight * static_cast<int>(
                         block.luma[static_cast<std::size_t>(row) *
                                        static_cast<std::size_t>(
                                            block.luma_stride) +
                                    static_cast<std::size_t>(col)]);
      }
    }
  }
  return t >> 3;
}

// §7.13.2.1 CflRef capture for cfl_index == CFL_MULTI.
AV2ENC_MHCCP_HD inline bool capture_refs(const Block& block, Refs& refs) {
  if (!valid(block)) {
    return false;
  }

  const int above = block.have_above != 0 ? 2 : 0;
  const int left = block.have_left != 0 ? 2 : 0;
  const int sub_x = block.sub_x;
  const int sub_y = block.sub_y;
  const int luma_w = block.w << sub_x;
  const int luma_h = block.h << sub_y;

  int px_top_right = 4 * block.num4_above_right;
  int px_bot_left = 4 * block.num4_below_left;
  if (luma_w <= 4 || block.have_above == 0) {
    px_top_right = 0;
  }
  if (luma_h <= 4 || block.have_left == 0) {
    px_bot_left = 0;
  }

  const int right_luma_x =
      min_int(block.mi_col_end, (block.x + block.w + px_top_right) << sub_x);
  const int bottom_luma_y =
      min_int(block.mi_row_end, (block.y + block.h + px_bot_left) << sub_y);
  const int cfl_width =
      min_int((left << sub_x) + (right_luma_x - (block.x << sub_x)), 128);
  const int cfl_height =
      min_int((above << sub_y) + (bottom_luma_y - (block.y << sub_y)), 128);

  refs.above = above;
  refs.left = left;
  refs.rows = block.h + above;
  refs.cols = block.w + left;
  refs.target_w = cfl::round2(max_int(0, cfl_width), sub_x);
  refs.target_h = cfl::round2(max_int(0, cfl_height), sub_y);
  refs.ref_w = max_int(0, cfl_width) >> sub_x;
  refs.ref_h = max_int(0, cfl_height) >> sub_y;

  // The spec zeroes the h+CflAbove by w+CflLeft rectangle. Zero the full
  // scratch rectangle so the CPU, host-emulation and GPU copies are
  // deterministic even when the above-right/below-left extents exceed the
  // current block.
  for (int i = 0; i < kMaxRefDim; ++i) {
    for (int j = 0; j < kMaxRefDim; ++j) {
      refs.luma[i][j] = 0;
      refs.chroma[i][j] = 0;
    }
  }

  const int fill_h = min_int(kMaxRefDim, refs.target_h);
  const int fill_w = min_int(kMaxRefDim, refs.target_w);
  const int sb_chroma_y = block.sb_luma_y >> sub_y;
  for (int i = 0; i < fill_h; ++i) {
    for (int j = 0; j < fill_w; ++j) {
      const int chroma_x = block.x + j - left;
      const int chroma_y = block.y + i - above;
      if (i < above || j < left) {
        const int row = max_int(chroma_y, sb_chroma_y - 1);
        const int clamped_row = cfl::clip3(0, block.chroma_max_y, row);
        const int clamped_col = cfl::clip3(0, block.chroma_max_x, chroma_x);
        refs.chroma[i][j] = static_cast<std::int16_t>(
            block.chroma[static_cast<std::size_t>(clamped_row) *
                             static_cast<std::size_t>(block.chroma_stride) +
                         static_cast<std::size_t>(clamped_col)]);
      }
      if (ref_luma_avail(i, j, above, left, block.w, block.h)) {
        refs.luma[i][j] = static_cast<std::int16_t>(
            cfl_luma_sample(block, chroma_x, chroma_y, j == 0, i == 0));
      }
    }
  }
  return true;
}

// §7.13.7 Derive multi param process. `cfl_mh_dir` selects the first
// hypothesis. `count_out` (optional) receives the number of reference samples.
AV2ENC_MHCCP_HD inline MultiParams derive_multi_params(const Block& block,
                                                       const Refs& refs,
                                                       int cfl_mh_dir,
                                                       int* count_out) {
  MultiParams result{};
  result.p[0] = 0;
  result.p[1] = 0;
  result.p[2] = 1 << kBits;
  if (count_out != nullptr) {
    *count_out = 0;
  }
  if (cfl_mh_dir < 0 || cfl_mh_dir >= kDirections) {
    return result;
  }

  const int bit_depth = block.bit_depth;
  std::int64_t ata[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  std::int64_t b[3] = {0, 0, 0};
  int v[3] = {0, 0, 1 << (bit_depth - 1)};
  int count = 0;

  const int ref_h = min_int(refs.ref_h, kMaxRefDim);
  const int ref_w = min_int(refs.ref_w, kMaxRefDim);
  const int above = refs.above;
  const int left = refs.left;

  for (int i = 1; i < ref_h - 1; ++i) {
    for (int j = 1; j < ref_w - 1; ++j) {
      if (i < above || j < left) {
        if (cfl_mh_dir == 0) {
          v[0] = refs.luma[i][j];
        } else if (cfl_mh_dir == 1) {
          v[0] = refs.luma[max_int(0, i - 1)][j];
        } else {
          v[0] = refs.luma[i][max_int(0, j - 1)];
        }
        const int luma_value = refs.luma[i][j];
        v[1] = cfl::round2(luma_value * luma_value, bit_depth);
        const int target = refs.chroma[i][j];
        for (int i0 = 0; i0 < 3; ++i0) {
          for (int i1 = i0; i1 < 3; ++i1) {
            ata[i0][i1] += static_cast<std::int64_t>(v[i0]) * v[i1];
          }
          b[i0] += static_cast<std::int64_t>(v[i0]) * target;
        }
        ++count;
      }
    }
  }

  if (count_out != nullptr) {
    *count_out = count;
  }
  if (count == 0) {
    return result;
  }

  const int matrix_shift =
      kBits + 6 - 2 * bit_depth - ceil_log2(count);
  int left_shift = 0;
  int right_shift = 0;
  if (matrix_shift > 0) {
    left_shift = matrix_shift;
  } else {
    right_shift = -matrix_shift;
  }

  int norm_ata[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  int norm_b[3] = {0, 0, 0};
  for (int i0 = 0; i0 < 3; ++i0) {
    for (int i1 = i0; i1 < 3; ++i1) {
      norm_ata[i0][i1] = static_cast<int>(
          asr64(ata[i0][i1] * (static_cast<std::int64_t>(1) << left_shift),
                right_shift));
    }
    norm_b[i0] = static_cast<int>(
        asr64(b[i0] * (static_cast<std::int64_t>(1) << left_shift),
              right_shift));
  }

  // §7.13.8 Gaussian elimination without pivoting.
  int c[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      c[i][j] = j >= i ? norm_ata[i][j] : norm_ata[j][i];
    }
    c[i][i] += 2 << (bit_depth - 8);
    c[i][3] = norm_b[i];
  }

  for (int i = 0; i < 3; ++i) {
    const int diag = max_int(1, abs_int(c[i][i]));
    const DivisionScaleShift ds = get_division_scale_shift(diag);
    for (int j = i + 1; j < 4; ++j) {
      c[i][j] = mul_fixed32_adapt(c[i][j], ds.scale, ds.shift);
    }
    for (int j = i + 1; j < 3; ++j) {
      const int scale_factor = c[j][i];
      for (int k = i + 1; k < 4; ++k) {
        c[j][k] -= mul_fixed32_adapt(scale_factor, c[i][k], kBits);
      }
    }
  }

  int params[3] = {0, 0, 0};
  params[2] = c[2][3];
  for (int i = 2; i >= 0; --i) {
    params[i] = c[i][3];
    for (int j = i + 1; j < 3; ++j) {
      params[i] -= mul_fixed32_adapt(c[i][j], params[j], kBits);
    }
  }

  result.p[0] = params[0];
  result.p[1] = params[1];
  result.p[2] = params[2];
  return result;
}

// §7.13.6 predicted sample for the current block's luma reference.
AV2ENC_MHCCP_HD inline int predict_sample(const Block& block, const Refs& refs,
                                          const MultiParams& params,
                                          int cfl_mh_dir, int i, int j) {
  const int above = refs.above;
  const int left = refs.left;
  const int row = above + i;
  const int col = left + j;
  const int a = refs.luma[row][col];
  int vec[3] = {0, 0, 1 << (block.bit_depth - 1)};
  if (cfl_mh_dir == 0) {
    vec[0] = a;
  } else if (cfl_mh_dir == 1) {
    vec[0] = refs.luma[max_int(0, row - 1)][col];
  } else {
    vec[0] = refs.luma[row][max_int(0, col - 1)];
  }
  vec[1] = cfl::round2(a * a, block.bit_depth);

  int t = 0;
  for (int k = 0; k < 3; ++k) {
    t += mul_fixed32_adapt(params.p[k], vec[k], kBits);
  }
  return cfl::clip1(t, block.bit_depth);
}

}  // namespace av2enc::cfl::mhccp

#endif  // AV2ENC_ENCODER_CFL_MHCCP_COMMON_H_
