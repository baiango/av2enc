// M1 GPU path — CDEF integration stage (AV2 §7.18 / §5.18.7.10).
//
// This is the integration-ready CDEF stage: it takes the unfiltered
// reconstructed planes plus the MI metadata, decides the per-64x64 `cdef_idx`
// by SSE, and produces
//
//   * the filtered planes (`CdefFrame`, i.e. the reference for later pictures),
//   * the per-64x64 `cdef_idx` grid, and
//   * the §5.18.7.10 header parameters the frame-header writer must emit.
//
// Placement (normative order, §7.17 / §7.18 / §7.19 / §7.20):
//
//   reconstruct -> deblock -> **CDEF** -> CCSO -> loop restoration
//
// M1 has no CCSO/LR yet, so CDEF runs immediately after `recon_deblock_frame()`
// and its output replaces the reconstruction used for output and for future
// references.
//
// Ownership / contract
// --------------------
//   * This directory only.  The stage never allocates in the frame loop, never
//     synchronises, and never touches CUDA error state.
//   * `src/cuda/m1/cdef/` is compiled into the orchestrator's OBJECT library by
//     the root CMake glob, so the host part (`still_cdef.cpp`) must not depend on
//     `src/encoder/filters/cdef/cdef.cpp` (that file is not in that build).
//     The search lives in the header-only `cdef_search.h` for this reason.
//   * The CUDA entry points are declared in `still_cdef_cuda.h` so this header
//     stays usable from a plain C++ build and from tests.
//
// What the M1 syntax owner must add (exact syntax in README.md):
//   1. sequence header: `enable_cdef = 1` (§5.4.10);
//   2. frame header: four `df_delta_q_present` f(1) bits for the enabled
//      planes (they are zero; the pinned decoder consumes them — see README),
//      then `cdef_params()` from `CdefHeaderParams`;
//   3. mode info: `read_cdef()` per coding block, i.e.
//      `cdef_encode_block_index()` from `still_cdef_index_syntax.h`, after
//      `intra_segment_id()` and before `read_intra_y_mode()`.
#ifndef AV2ENC_CUDA_STILL_CDEF_STILL_CDEF_H_
#define AV2ENC_CUDA_STILL_CDEF_STILL_CDEF_H_

#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/cdef/cdef.h"
#include "codec/filters/cdef/cdef_search.h"

namespace av2enc::cuda::still {

// ---------------------------------------------------------------------------
// §5.18.7.10 cdef_params() output
// ---------------------------------------------------------------------------

// One strength set, in *syntax* form.  `*_sec_strength` is the f(2) value
// 0..3; the decoder maps 3 to the sample-domain 4.  `*_pri_zero` selects the
// 1-bit "strength is 0" form, so `*_pri_strength` is only written when it is
// false.
struct CdefHeaderStrength {
  bool y_pri_zero = true;
  int y_pri_strength = 0;  // 0..15, written when !y_pri_zero
  int y_sec_strength = 0;  // syntax 0..3 (3 => 4)
  bool uv_pri_zero = true;
  int uv_pri_strength = 0;
  int uv_sec_strength = 0;
  // Sample-domain values, for cross-checking the decoder's §6.17.7.6 mapping.
  int y_sec_value = 0;   // 0, 1, 2 or 4
  int uv_sec_value = 0;
};

struct CdefHeaderParams {
  bool frame_enable = false;
  int damping_minus_3 = 0;      // f(2)
  int strengths_minus_1 = 0;    // f(3)
  bool on_skip_txfm_frame_enable = false;  // f(1) when CdefOnSkipTxfm==ADAPTIVE
  int num_strengths = 0;
  CdefHeaderStrength set[cdef::kMaxStrengths] = {};
  // Total bits cdef_params() writes for this configuration.  Excludes the
  // four deblocking `df_delta_q_present` bits that precede it.
  int bits = 0;
};

// Builds the header parameters from the filter options.  `single_picture_header`
// is M1's case: cdef_frame_enable is inferred to 1 and no bit is coded for it.
// Returns false for non-signallable options (strength out of range, sec not in
// {0,1,2,4}, damping outside 3..6, count outside 1..8).
bool still_cdef_build_header(const cdef::CdefOptions& options,
                          bool single_picture_header, CdefHeaderParams* out,
                          std::string* error);

// ---------------------------------------------------------------------------
// MI metadata -> §7.18.1 grids
// ---------------------------------------------------------------------------

// Optional per-MI inputs.  All grids are row-major on the luma MI grid
// (`mi_cols * mi_rows`).  Null pointers mean "all zero", which is the M1 case:
// intra blocks have skip_flag == 0, and QP 90 is not lossless.
struct CdefMiInputs {
  int mi_cols = 0;
  int mi_rows = 0;
  const std::uint8_t* skip = nullptr;             // Skips[mi]
  const std::uint8_t* segment_id = nullptr;       // SegmentIds[mi]
  const std::uint8_t* chroma_segment_id = nullptr;  // ChromaSegmentIds[mi]
  const std::uint8_t* lossless_array = nullptr;   // LosslessArray[segment]
  int lossless_array_size = 0;
};

struct CdefMiGrids {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<std::uint8_t> skip;
  std::vector<std::uint8_t> lossless;
  std::vector<std::uint8_t> lossless_chroma;
};

// Materialises the three MI grids the filter reads.  `lossless` is
// `LosslessArray[SegmentIds[mi]]` (0 when no segment/lossless data is given).
bool still_cdef_build_mi_grids(const CdefMiInputs& inputs, CdefMiGrids* out,
                            std::string* error);

// ---------------------------------------------------------------------------
// Stage input / options / plan
// ---------------------------------------------------------------------------

// One host-side plane binding.  `src` is the unfiltered (deblocked)
// reconstruction, `dst` the CDEF output; they must not alias.  `dst` may be
// null for a planning-only call (`still_cdef_plan` never writes pixels, which is
// what the M2 adapter uses).  The plane is MI-aligned: `mi_cols*4` x
// `mi_rows*4` samples, luma stride >= that width.
struct CdefPlaneBinding {
  const std::uint8_t* src = nullptr;
  std::uint8_t* dst = nullptr;
  int stride = 0;
  int height = 0;
};

struct CdefStageFrame {
  CdefPlaneBinding plane[3];
  // Original picture planes, used only by the strength search.  Visible size.
  const std::uint8_t* ref[3] = {nullptr, nullptr, nullptr};
  int ref_stride[3] = {0, 0, 0};
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  // Caller-owned metadata grids (see still_cdef_build_mi_grids).
  const std::uint8_t* skip = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint8_t* lossless_chroma = nullptr;
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

struct CdefStageOptions {
  // CdefDamping (3..6), i.e. cdef_damping_minus_3 + 3.
  int damping = 3;
  // CdefStrengths (1..8).  1 means the decoder reads no index syntax and every
  // 64x64 block is filtered with set 0.
  int num_strengths = 1;
  bool on_skip_txfm = false;
  bool disable_loopfilters_across_tiles = false;
  // Include an all-zero strength set so blocks can be left unfiltered.  Needed
  // whenever num_strengths >= 2 (there is no -1 cdef_idx syntax).
  bool include_off = true;
  std::int64_t chroma_weight = 1;
  // Optional candidate ladder (syntax values); null uses the default ladder.
  const cdef::CdefStrength* candidates = nullptr;
  int candidate_count = 0;
  // 2x2 MI decimation for the frame-level strength choice: evaluate every
  // `search_step`-th 8x8 block row/column.  Requires num_strengths == 1.
  int search_step = 1;
};

struct CdefStagePlan {
  // Full search report (per-block SSE, chosen sets, ...).
  cdef::CdefSearchResult search;
  CdefHeaderParams header;
  int num_strengths = 0;  // == header.num_strengths
  int n64_cols = 0;
  int n64_rows = 0;
  // One entry per 64x64 luma block, row-major, in [0, num_strengths).
  std::vector<std::int8_t> cdef_idx;
  int header_bits = 0;          // cdef_params() bits
  int index_bits_estimate = 0;  // estimated cdef_idx symbol cost
  std::int64_t sse_total = 0;
  std::int64_t sse_off_total = 0;
  bool forced_single_strength = false;
};

// Runs the strength search and produces the header + index buffers.  Does not
// filter pixels and does not touch device memory: on the GPU path the
// orchestrator uploads `cdef_idx` (or uses the device search) and calls
// `still_cdef_filter()` in `still_cdef_cuda.h`.
bool still_cdef_plan(const CdefStageFrame& frame, const CdefStageOptions& options,
                  CdefStagePlan* plan, std::string* error);

// ---------------------------------------------------------------------------
// Device view (POD; the header itself needs no CUDA)
// ---------------------------------------------------------------------------

// Every pointer is device memory.  Passed by value to the kernels.
struct CdefDeviceFrame {
  const std::uint8_t* src[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* dst[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  const std::uint8_t* skip = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint8_t* lossless_chroma = nullptr;
  const std::int8_t* cdef_idx = nullptr;
  int n64_cols = 0;
  int n64_rows = 0;
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Host-side validation shared by the device entries: geometry, strides,
// non-aliasing planes, metadata presence, option ranges.
bool still_cdef_validate_device_frame(const CdefDeviceFrame& frame,
                                   const cdef::CdefOptions& options,
                                   std::string* error);

}  // namespace av2enc::cuda::still

#endif  // AV2ENC_CUDA_STILL_CDEF_STILL_CDEF_H_
