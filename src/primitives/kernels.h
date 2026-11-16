// Pixel-kernel layer: the single seam between the encoder's decision/pixel
// loops and an ISA-specific implementation (NEON today, AVX2 later).
//
// Why this exists
// ---------------
// Every pixel loop in this tree started life inlined at its call site:
// motion-estimation cost (`codec/inter/me/me_core.h`), RDO distortion
// (`codec/rate/rdo/rdo.cpp`), intra mode cost
// (`codec/intra/modes/intra_modes.cpp`), IBC
// (`codec/intra/ibc/core/ibc_core.h`), partition variance
// (`pipeline/video/video_frame.cpp`), the §7.13.2.18 intra edge filter, and the
// 16-bit format-video core (`pipeline/video/hbd/video_hbd_ops.h`).  That made
// "add NEON" a cross-cutting edit.  This header is the seam those call sites
// moved to; each migrated site keeps its inlined loop as the
// `!AV2ENC_PRIMITIVES` fallback for component gates.
//
// Contract
// --------
//   * All functions are pure: no globals, no allocation, no observable state.
//   * The scalar reference implementation is a *transcription* of the existing
//     inlined loops, including accumulation order, rounding and truncation.
//     Bit-exactness with the current CPU reference is the contract; see
//     `docs/primitives-inventory.md` for the source line each kernel came from.
//   * Sizes are the encoder's: 4/8/16/32/64 and rectangular / odd leaves.
//     Partial blocks at the right/bottom edge are zero-padded exactly where the
//     source loop zero-pads them.
//   * Strides are in samples (bytes for 8-bit planes) and may exceed the block
//     width.  Pointers may be unaligned.
//
// Dispatch
// --------
// `kernels()` returns the table selected once, on first use (which is start-up
// for any consumer), from `AV2ENC_SIMD` (`auto`, `scalar` or `neon`; default
// `auto`).  `auto` picks the NEON table when the build contains one, `scalar`
// always picks the reference, and an explicit target that is not built falls
// back to scalar with a one-line notice.  `intro()` is the human-readable
// `--simd`-style description; the CLI may print it without changing encoder
// behaviour.
//
// Adding an ISA
// -------------
// Implement the same signatures in `src/primitives/kernels_neon.cpp`, advertise
// the table from `kernels.h` (guarded by `AV2ENC_PRIMITIVES_HAVE_NEON`) and
// return it from the `resolve()` hook in `kernels.cpp`.  No call site changes.
// The scalar table stays as the oracle and the fallback.

#ifndef AV2ENC_PRIMITIVES_KERNELS_H_
#define AV2ENC_PRIMITIVES_KERNELS_H_

// ISA availability is a property of the compiler, not of a build option:
// AArch64 defines `__ARM_NEON` and `__aarch64__` for every translation unit,
// and `kernels_neon.cpp` compiles its table only under the same guard.  Keep
// this in sync with that guard; a build system may also define the macro when
// it compiles the NEON TU conditionally (the repeated definition is harmless).
#if defined(__ARM_NEON) && defined(__aarch64__)
#define AV2ENC_PRIMITIVES_HAVE_NEON 1
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace av2enc::primitives {

// Sum of absolute differences between two w*h 8-bit blocks.
//
// Transcribed from `me_sad_rows` (integer-MV path), `ibc::sad_block` and
// `rdo::sad_u8`, which agree bit-for-bit.  Returns 0 for w <= 0 or h <= 0.
using SadFn = std::uint64_t (*)(const std::uint8_t* src, std::size_t src_stride,
                                const std::uint8_t* ref, std::size_t ref_stride,
                                int w, int h) noexcept;

// SAD and SSE of two w*h 8-bit blocks in one pass.  Both outputs are required
// and are always written (0 for w <= 0 or h <= 0).
//
// The SAD half is `sad()` above.  The SSE half is transcribed from the two
// live call-site loops that previously ran a second pass after `sad()`:
//   intra::prediction_cost  (int diff, |diff| * |diff| in int, uint64 sum)
//   dip::prediction_cost8   (int64 diff, diff * diff in int64, uint64 sum)
// For 8-bit samples both products are exact in int64/int (|diff| <= 255), so
// one accumulator reproduces both.  There is no rounding and no floating
// point in this kernel, so `-ffp-contract` does not affect it.
using SadSseFn = void (*)(const std::uint8_t* src, std::size_t src_stride,
                          const std::uint8_t* ref, std::size_t ref_stride,
                          int w, int h, std::uint64_t* sad,
                          std::uint64_t* sse) noexcept;

// Hadamard tile size, because the tree has two SATD definitions that are both
// live and are not interchangeable:
//   k4x4 - rdo::satd_u8: unscaled 4x4 Hadamard, sum of |coeff|.
//   k8x8 - me::me_satd_unit / me_cost_plane(kMeSatd): 8x8 Hadamard, each tile
//          rounded by (sum + 8) >> 4 to keep SATD in the SAD range.
enum class SatdTile : int { k4x4 = 4, k8x8 = 8 };

// Sum of absolute Hadamard-transformed differences over `tile`-sized tiles.
// Tiles at the right/bottom edge are zero-padded, matching both source loops.
// Returns 0 for an invalid size or tile.
using SatdFn = std::uint64_t (*)(const std::uint8_t* src, std::size_t src_stride,
                                 const std::uint8_t* ref, std::size_t ref_stride,
                                 int w, int h, SatdTile tile) noexcept;

// Truncated block variance cost used by the partition decision:
//   sum -> double mean -> double sum of squared deviations -> int64 truncation.
//
// Transcribed from `block_variance_cost_at()` in `pipeline/video/video_frame.cpp`
// and `pipeline/still/still_encoder.cpp` (and the hbd mirror, which uses the
// same formula).  The double accumulation order is part of the contract, so
// this kernel must stay compiled with `-ffp-contract=off`.  Returns 0 for
// w <= 0 or h <= 0.
using VarianceFn = std::int64_t (*)(const std::uint8_t* src,
                                    std::size_t src_stride, int w, int h) noexcept;

// The same truncated variance cost over a 16-bit (high-bit-depth) block.
//
// Transcribed from `hbd::hbd_variance_cost()` in
// `pipeline/video/hbd/video_hbd_ops.h`: identical int64-sum / binary64-mean /
// binary64-SSE sequence, only the container differs.  `src` points at the
// visible rectangle's origin (the call site adds x/y * stride to the pointer).
// The loop shape therefore matches `variance()`.  Requires
// `-ffp-contract=off`.  Returns 0 for w <= 0 or h <= 0.
using VarianceU16Fn = std::int64_t (*)(const std::uint16_t* src,
                                       std::size_t src_stride, int w,
                                       int h) noexcept;

// Integer-pel SAD of a 16-bit block against a reference, clipping every
// reference coordinate to the reference extent (`ref_w` x `ref_h`).
//
// Transcribed from `hbd::hbd_sad_at()` in
// `pipeline/video/hbd/video_hbd_ops.h`.  `cur` is the current frame at plane
// origin; the block's top-left is (x, y) in the current plane and the integer
// motion offset is (mv_y, mv_x).  The reference sample for every position is
// clamped independently:
//   ry = clip3(0, ref_h - 1, y + r + mv_y)
//   rx = clip3(0, ref_w - 1, x + c + mv_x)
// so partial blocks at the plane edge read a valid (replicated) sample rather
// than being skipped.  Returns 0 for w <= 0 or h <= 0.
using SadU16ClampedFn = std::int64_t (*)(
    const std::uint16_t* cur, int cur_stride, const std::uint16_t* ref,
    int ref_stride, int x, int y, int w, int h, int mv_y, int mv_x, int ref_w,
    int ref_h) noexcept;

// Row-wise copy of a w*h 8-bit block.  Source and destination may be the same
// buffer only if the ranges do not overlap; the scalar loop is forward, so
// overlapping copies are not supported.
using BlockCopyFn = void (*)(const std::uint8_t* src, std::size_t src_stride,
                             std::uint8_t* dst, std::size_t dst_stride, int w,
                             int h) noexcept;

// §7.13.2.18 intra edge filter ("IDIF filtered-edge generation").
//
// `edge` is the spec's derived edge[] array: edge[i] == AboveRow[i-1] (or
// LeftCol[i-1]) for i in [0, size).  The kernel reads edge[0..size-1] from a
// snapshot and writes the filtered samples back to edge[1..size-1], i.e.
// AboveRow[0..size-2].  This is exactly `intra::intra_edge_filter`'s
// "snapshot then write" rule, so filtered samples are never read by later
// taps.  `size` is 2..129 (the existing EdgeRow capacity); `strength` is 1..3.
// Out-of-range arguments are a no-op.
using EdgeFilterFn = void (*)(std::int16_t* edge, int size, int strength) noexcept;

// Subpel motion-estimation SAD: the normative two-pass 8-tap interpolation
// (§7.13.3.14) followed by |cur - pred| accumulation, over the rows
// `r0, r0+rstep, ... < r1` of a w-sample-wide block.
//
// Transcribed verbatim from `me::me_sad_subpel_rows` (codec/inter/me/me_core.h),
// which reads every sample through `me_sample_at`/`me_interp_sample`:
//   * integer MVs ((mv_x | mv_y) & 15 == 0) bypass the filter and read the
//     reference directly, with no clamp (that is the existing behaviour);
//   * otherwise each of the 8 horizontal taps clamps its x coordinate to
//     [0, ref_w-1], the 8 taps are accumulated, rounded by Round0 (3);
//     then each of the 8 vertical taps clamps its y coordinate to
//     [0, ref_h-1], the 8 horizontally filtered rows are accumulated,
//     rounded by Round1 (11) and clipped to [0, 255].
// `r0/r1/rstep` are kept because the CUDA warp cost owns a row subset
// (r0 = lane, rstep = 32); the CPU cost path uses r0 = 0, rstep = 1.
// Returns 0 for w <= 0 or an empty row range.
using SubpelSadRowsFn = std::int32_t (*)(
    const std::uint8_t* cur, std::size_t cur_stride, const std::uint8_t* ref,
    std::size_t ref_stride, int ref_w, int ref_h, int x, int y, int w,
    std::int32_t mv_x, std::int32_t mv_y, int filter, int r0, int r1,
    int rstep) noexcept;

// §7.13.2.17 edge-filter strength selection, including the mode-dependent
// filter type (true when either neighbouring block is SMOOTH / SMOOTH_V /
// SMOOTH_H) and the block-size thresholds.  `delta` is the angle difference in
// degrees; its absolute value is used.  This is a branch table, not a pixel
// loop, so it is not a dispatch slot: every ISA uses the same arithmetic.
//
// Transcribed from `intra::edge_filter_strength`.
int intra_edge_filter_strength(int w, int h, bool smooth_neighbors,
                               int delta) noexcept;

struct KernelTable {
  // Short target name: "scalar" today; "neon"/"avx2" when built.
  const char* name = "scalar";
  SadFn sad = nullptr;
  SadSseFn sad_sse = nullptr;
  SatdFn satd = nullptr;
  VarianceFn variance = nullptr;
  BlockCopyFn block_copy = nullptr;
  EdgeFilterFn edge_filter = nullptr;
  // High-bit-depth (16-bit container) slots.  Only the format video core
  // (`video_hbd_ops.h`) consumes these today; they exist so a NEON table can
  // vectorise the 10-bit path with the same seam as the 8-bit one.
  SadU16ClampedFn sad_u16_clamped = nullptr;
  VarianceU16Fn variance_u16 = nullptr;
  // Subpel ME cost (fractional MVs).  This is the hot loop of the CPU encoder;
  // it is not a plain sum kernel: the scalar reference reproduces
  // `me::me_sad_subpel_rows` including the border clamp and the integer-MV
  // bypass.  See docs/simd-plan.md.
  SubpelSadRowsFn subpel_sad_rows = nullptr;
};

// The portable C++ reference table.  Never changes; always available.
const KernelTable& scalar_kernels() noexcept;

#if defined(AV2ENC_PRIMITIVES_HAVE_NEON)
// AArch64 NEON table.  Every slot starts as the scalar implementation and is
// overridden only once it has been proven bit-identical to it; see
// docs/simd-plan.md and docs/primitives-inventory.md.
const KernelTable& neon_kernels() noexcept;
#endif

// Resolve a requested target ("scalar", "auto", or an ISA name that is not
// built).  Never throws and never reads the environment.  Unknown names and
// ISAs that are not compiled in return the scalar table, so an inherited
// `AV2ENC_SIMD` value can never break a run.  `nullptr`/empty means "auto".
const KernelTable& select_kernels(std::string_view requested) noexcept;

// The table selected at start-up from `AV2ENC_SIMD`.  Selected once; the
// environment is read on the first call.
const KernelTable& kernels() noexcept;

// Effective target name, for diagnostics and tests.
std::string_view selected_target() noexcept;

// One-line `--simd`-style report, e.g.
//   "primitives: scalar (portable C++; no SIMD target built; AV2ENC_SIMD=auto)"
// Includes any fallback reason when an unbuilt/unknown target was requested.
std::string intro();

// Targets this build can actually select, comma-separated ("scalar" plus any
// compiled-in ISA).  Used by `intro()` and available for a future `--simd`
// listing.
std::string available_targets();

}  // namespace av2enc::primitives

#endif  // AV2ENC_PRIMITIVES_KERNELS_H_
