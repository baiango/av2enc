// AV2 chroma loop restoration — POD descriptions shared by host and device.
//
// Scope: the *pixel* application of the chroma non-separable Wiener filter
// (§7.20.3 with the 18-coefficient chroma layout of §5.20.10.6).  The
// per-unit decision (on/off, bank merge, subset, uv_sym, coefficients) is made
// on the CPU by `src/encoder/filters/lr_chroma/`; this component moves the
// pixels.
//
// The structures here contain no CUDA keywords, no host-only types, and no
// storage.  They are safe to pass by value as kernel arguments.
//
// Chroma coefficient layout (identical to the decoder's LrWienerNs stride 18):
//   j =  0..11  Wiener_Ns_Config_Uv taps (12 chroma tap positions -> 6 groups)
//   j = 12..17  chroma-from-luma taps        (6 extra groups / 12 positions)
// A subset selects which of the 18 slots are present:
//   subset 0 -> 0..5 (6), subset 1 -> 0..9 (10), subset 2 -> 0..17 (18).

#ifndef AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_TYPES_H_
#define AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_TYPES_H_

#include <cstdint>

namespace av2enc::cuda::lr_chroma {

// Spec constants (see encoder/filters/restoration/restoration_common.h).
inline constexpr int kMiSize = 4;
inline constexpr int kCoeffStride = 18;
inline constexpr int kShortCoeffs = 6;
inline constexpr int kUvTaps = 12;
inline constexpr int kPrecBits = 7;
inline constexpr int kBankSize = 4;  // LR_BANK_SIZE
inline constexpr int kMaxUnitSize = 256;
inline constexpr int kMinUnitSize = 32;

// One frame's geometry and chroma filter configuration.  `mi_cols`/`mi_rows`
// are the luma 4x4 grid; chroma plane extents are derived:
//   width  = (mi_cols * 4) >> sub_x
//   height = (mi_rows * 4) >> sub_y
// `unit_size` is LoopRestorationSize[1] in *chroma plane samples*.
struct LrChromaConfig {
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;                 // 0 (4:4:4 / 4:2:2 vertical) or 1
  int sub_y = 1;                 // 0 (4:4:4 / 4:2:2 horizontal) or 1
  int bit_depth = 8;             // 8 or 10
  int unit_size = 64;            // 32, 64, 128 or 256 chroma samples
  int cfl_ds_filter_index = 0;   // 0..3 (3 aliases to 0 in §7.20.3)
  int sb_shift = 4;              // Mi_Width_Log2[SbSize]; diagnostics only
};

// One image plane.  `curr` is CurrFrame (deblocked, pre-CDEF) and `cdef` is
// CdefFrame (post-CDEF/CCSO).  All pointers refer to uint8_t samples for
// 8-bit and LSB-aligned uint16_t samples for 10-bit; strides are in samples.
struct LrChromaPlane {
  const void* curr = nullptr;
  int curr_stride = 0;
  const void* cdef = nullptr;
  int cdef_stride = 0;
  void* dst = nullptr;
  int dst_stride = 0;
};

// Frame-level inputs.  `chroma[0]` is U and `chroma[1]` is V; `luma` is plane
// 0 and is read by the chroma-from-luma taps.  The optional MiCol/MiRow start
// and end grids are on the luma MI grid and are only used when
// `disable_loopfilters_across_tiles != 0` (a null set means one tile).
struct LrChromaFrame {
  LrChromaPlane chroma[2];
  LrChromaPlane luma;
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int bit_depth = 8;
  int cfl_ds_filter_index = 0;

  const std::uint16_t* mi_col_start = nullptr;
  const std::uint16_t* mi_col_end = nullptr;
  const std::uint16_t* mi_row_start = nullptr;
  const std::uint16_t* mi_row_end = nullptr;
  int disable_loopfilters_across_tiles = 0;
};

// One per-unit chroma filter decision, in decoder order.  The resolver in
// `lr_chroma_device.cuh` turns this into 18 expanded coefficients:
//
//   * `on == 0`            -> unit is a copy of CdefFrame;
//   * `merged != 0`        -> copy bank entry `bank_k` (0 = newest; -1 also
//                             selects the newest entry; see note below);
//   * otherwise            -> zero taps outside `subset`, then mirror pairs
//                             j/j+1 for j >= 6 when `uv_sym != 0`.
//
// `bank_k` is the distance back from the newest bank entry, matching the
// decoder's `(bank_ptr - k + LR_BANK_SIZE) % LR_BANK_SIZE` lookup.  The CPU
// `ChromaFilter::use_bank_k` uses the same convention; its -1 sentinel is
// documented as "newest", which is what this resolver implements.
struct LrChromaUnit {
  std::int32_t on = 0;
  std::int32_t merged = 0;
  std::int32_t bank_k = -1;
  std::int32_t subset = 0;   // 0..2
  std::int32_t uv_sym = 0;   // only meaningful for subset > 0
  std::int16_t coeffs[kCoeffStride] = {};
};

// Per-plane §5.20.10.6 coefficient bank.  The ring is small and the resolver
// walks units in coding order; no inter-thread sharing is required.
struct LrChromaBankState {
  std::int16_t bank[kBankSize][kCoeffStride] = {};
  std::int32_t size = 0;
  std::int32_t ptr = 0;
};

// §7.20.1 block context for one 4x4 luma MI block of one chroma plane.
struct LrChromaBlock {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int unit_row = 0;
  int unit_col = 0;
  int unit_index = 0;
  int luma_start_x = 0;
  int luma_start_y = 0;
  int luma_end_x = 0;
  int luma_end_y = 0;
  int stripe_start_y = 0;  // luma samples
  int stripe_end_y = 0;
};

// Device view assembled by the launcher from `LrChromaFrame` + the workspace
// unit grids.  Consumed only by the shared device header.
struct LrChromaCore {
  LrChromaPlane chroma[2];
  LrChromaPlane luma;
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int bit_depth = 8;
  int cfl_ds_filter_index = 0;
  int unit_size[2] = {64, 64};
  int unit_cols[2] = {1, 1};
  int unit_rows[2] = {1, 1};
  const std::uint16_t* mi_col_start = nullptr;
  const std::uint16_t* mi_col_end = nullptr;
  const std::uint16_t* mi_row_start = nullptr;
  const std::uint16_t* mi_row_end = nullptr;
  int disable_loopfilters_across_tiles = 0;
};

}  // namespace av2enc::cuda::lr_chroma

#endif  // AV2ENC_CUDA_LR_CHROMA_LR_CHROMA_TYPES_H_
