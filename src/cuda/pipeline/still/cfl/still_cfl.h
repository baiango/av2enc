#ifndef AV2ENC_CUDA_STILL_CFL_H_
#define AV2ENC_CUDA_STILL_CFL_H_

// GPU CfL prediction stage for M1 blocks — AV2 v1.0.0 §7.13.5.
//
// This is a thin M1 adapter over src/cuda/cfl. It owns no arithmetic: the
// §7.13.5 luma average, array L, derived-alpha correlation and prediction all
// live in cfl_common.h / cfl.cu and are shared with the CPU reference
// (src/encoder/cfl/). The adapter adds only:
//
//   * an M1 block-pair descriptor (U + V for one luma block), so the plan can
//     upload one array and the stage runs both planes in one launch;
//   * descriptor construction from the decoder-layout plane bounds M1 uses;
//   * host-side validation; and
//   * two zero-copy entry points for the default derived alpha
//     (CFL_DERIVED_ALPHA) and for explicit alpha (CFL_EXPLICIT).
//
// Contract
// --------
//   * `block.chroma` must already contain this block's DC prediction on input;
//     the stage overwrites the w x h coded samples with the CfL prediction.
//   * `block.luma` is the reconstructed luma plane, `luma_max_x/_max_y` are
//     the inclusive bounds of the *coded* plane (MiCols*4-1 / MiRows*4-1) —
//     the same bounds the decoder's CurrFrame exposes. The plane allocation is
//     padded to a 64-sample multiple; samples outside the coded bounds are
//     decoder padding and are read (clamped) by §7.13.5, so they must hold the
//     decoder's values (0 for a fresh frame), not an encoder-only fill.
//   * `have_above`/`have_left` are AvailUChroma/AvailLChroma at chroma mi
//     granularity (`cfl_avail_above`/`cfl_avail_left`), not `y > 0`.
//   * No allocation, no host/device copy and no synchronization happens inside
//     these entry points. The orchestrator owns the arena, the uploads and the
//     stream; `d_pairs`/`d_alpha_q8` are device memory.
//
// Ownership: src/cuda/m1/cfl/ (workstream 6). Do not edit the shared CfL
// component from here.

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

#include "../../../codec/intra/cfl/cfl.h"

namespace av2enc::cuda::still {

// One M1 block: the U and V chroma descriptors share the same luma plane and
// geometry. `u.chroma` and `v.chroma` are the two reconstructed chroma planes;
// both already hold the DC prediction.
struct StillCflPair {
  cfl::Block u{};
  cfl::Block v{};
};

// The explicit-alpha form of one pair (cfl_index == CFL_EXPLICIT). Each plane
// carries the Q8 alpha `CflAlpha * 32` (sign included).
struct StillCflPairExplicit {
  CflJob u{};
  CflJob v{};
};

// Inputs shared by every block of a frame (or tile).
struct StillCflFrame {
  const std::uint8_t* luma = nullptr;
  int luma_stride = 0;
  int luma_max_x = 0;  // inclusive, coded plane
  int luma_max_y = 0;  // inclusive, coded plane
  int bit_depth = 8;
  int ds_filter = 0;  // cfl_ds_filter_index 0..3 (3 behaves as 0)
  int sub_x = 1;      // 4:2:0 for M1
  int sub_y = 1;
};

// One chroma transform block within the frame.
//
// `x`/`y` are chroma samples; the luma origin is (x << sub_x, y << sub_y).
// `coded_width`/`coded_height` are MiCols*MI_SIZE>>sub_x and
// MiRows*MI_SIZE>>sub_y. Availability defaults to the §5.20.4.1 derivation;
// pass 0/1 to override.
struct StillCflRect {
  std::uint8_t* chroma = nullptr;
  int chroma_stride = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int coded_width = 0;
  int coded_height = 0;
  int have_above = -1;  // < 0: derive
  int have_left = -1;   // < 0: derive
};

// Descriptor construction. `make_still_cfl_pair` checks nothing (it cannot fail
// without throwing); call still_cfl_valid_pair() before uploading if the plan is
// not trusted.
cfl::Block make_still_cfl_block(const StillCflFrame& frame, const StillCflRect& rect);
StillCflPair make_still_cfl_pair(const StillCflFrame& frame, const StillCflRect& u,
                           const StillCflRect& v);

// Host-only structural check: both cfl::Block descriptors are valid, U and V
// share luma geometry, and the two chroma planes are distinct. When `error`
// is non-null it receives a short reason on failure. This is plan validation,
// not a device operation.
bool still_cfl_valid_pair(const StillCflPair& pair, std::string* error = nullptr);

// Default M1 path (CFL_DERIVED_ALPHA): derives alpha per plane from that
// plane's reconstructed chroma above/left neighbours, then applies
//   pred = Clip1(dc + Round2Signed(alpha * (L - lumaAvg), 11)).
// All 2 * pair_count plane jobs (U and V, all pairs) run in one launch.
//
// `d_alpha_q8`, when non-null, receives 2 * pair_count ints with
// d_alpha_q8[2 * i] = U alpha and d_alpha_q8[2 * i + 1] = V alpha for pair i.
// `pair_count <= 0` is a no-op.
cudaError_t still_cfl_predict_pairs(const StillCflPair* d_pairs, int pair_count,
                                 int* d_alpha_q8 = nullptr,
                                 cudaStream_t stream = nullptr);

// Explicit-alpha path (CFL_EXPLICIT): applies the per-plane Q8 alpha stored in
// each job. One launch for all 2 * pair_count plane jobs.
cudaError_t still_cfl_predict_pairs_explicit(const StillCflPairExplicit* d_pairs,
                                          int pair_count,
                                          cudaStream_t stream = nullptr);

// Layout guarantees that make the forward above zero-copy.
static_assert(std::is_standard_layout_v<cfl::Block>);
static_assert(std::is_trivially_copyable_v<cfl::Block>);
static_assert(sizeof(StillCflPair) == 2 * sizeof(cfl::Block));
static_assert(offsetof(StillCflPair, v) == sizeof(cfl::Block));
static_assert(sizeof(StillCflPairExplicit) == 2 * sizeof(CflJob));
static_assert(offsetof(StillCflPairExplicit, v) == sizeof(CflJob));

}  // namespace av2enc::cuda::still

#endif  // AV2ENC_CUDA_STILL_CFL_H_
