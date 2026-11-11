// Chroma (U/V) loop restoration for AV2 §5.18.7.11 / §5.20.10.5–.6 / §7.20.3.
//
// Luma non-separable Wiener is owned by
// src/encoder/filters/restoration/classes/ (decision + syntax) and
// src/encoder/filters/restoration/ (filter core).  This module owns the
// chroma-specific pieces that are *not* shared with luma:
//
//   * §5.18.7.11 chroma unit sizing (`LoopRestorationSize[1]`, shift flags),
//   * per-unit type selection for U and V (RESTORE_NONE / WIENER_NONSEP),
//   * the 18-coefficient chroma filter layout (12 chroma taps + 6 luma taps),
//   * `wiener_ns_uv_sym` mirroring: coefficients j and j+1 are coded once for
//     j >= WIENER_NS_SHORT_COEFFS and copied by the decoder,
//   * shared-U/V decisions (one fitted filter used by both chroma planes) and
//     the per-plane mirroring rules,
//   * exact entropy syntax for a chroma restoration unit (writer + reader),
//   * an encoder decision path that measures real reconstruction against the
//     source picture on U and V.
//
// PC-Wiener is luma-only.  §5.4.10 forces
// `lr_tools_disable[1][RESTORE_PC_WIENER] = 1`, so a chroma restoration unit
// can only be RESTORE_NONE or RESTORE_WIENER_NONSEP.  The API never exposes a
// chroma PC-Wiener choice.
//
// Chroma restoration requires `lr_tools_uv_present` semantics to be correct:
// when `lr_tools_uv_present == 0` the decoder infers
// `lr_tools_disable[1][RESTORE_WIENER_NONSEP] =
//  lr_tools_disable[0][RESTORE_WIENER_NONSEP]`
// (§5.4.10), so a stream with luma Wiener enabled also has chroma Wiener
// enabled.  When `lr_tools_uv_present == 1` the chroma flag is coded
// explicitly.  This module writes/reads the explicit form so that the chroma
// tool set is independent of luma.

#ifndef AV2ENC_ENCODER_FILTERS_LR_CHROMA_CHROMA_LR_H_
#define AV2ENC_ENCODER_FILTERS_LR_CHROMA_CHROMA_LR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/filters/restoration/classes/wiener_syntax.h"
#include "encoder/filters/restoration/restoration.h"
#include "encoder/m1/filters_stream/filters_stream.h"

namespace av2enc::lr_chroma {

namespace rest = ::av2enc::restoration;
namespace classes = ::av2enc::restoration::classes;
namespace fs = ::av2enc::m1::filters_stream;

// ---------------------------------------------------------------------------
// §5.18.7.11 chroma unit sizing
// ---------------------------------------------------------------------------

// `LoopRestorationSize[1]` after the shift flags:
//
//   base  = RESTORATION_TILESIZE_MAX >> Max(SubsamplingX, SubsamplingY)
//   shift = 1 | 0 | 2 | 3 depending on the coded flags and SbSize
//   size  = base >> shift
//
// For 8-bit 4:2:0 (subX = subY = 1) and 64x64 superblocks the four legal
// chroma sizes are 128 (half), 256 (max), 64 (quarter) and 32 (not quarter).
// `LoopRestorationSize[2] = LoopRestorationSize[1]`.
struct ChromaUnitSize {
  int sb_size = 64;
  int sub_x = 1;
  int sub_y = 1;
  int shift = 2;  // 0..3
  int size = 64;  // plane samples
  bool half = false;
  bool max_size = false;
  bool quarter = false;
};

// Selects the shift flags for a desired chroma unit size.  Fails when the size
// is not representable for the given superblock size.
bool chroma_select_unit_size(int desired_size, int sb_size, int sub_x,
                             int sub_y, ChromaUnitSize* out,
                             std::string* error);

// Writes/reads the `lr_chroma_use_*` row exactly as §5.18.7.11 orders it.
bool chroma_write_unit_size(BitWriter& writer, const ChromaUnitSize& size,
                            std::string* error);
bool chroma_read_unit_size(classes::WienerBitReader& reader, int sb_size,
                           int sub_x, int sub_y, ChromaUnitSize* size,
                           std::string* error);

// §5.20.10.4 count_units_in_frame() and the chroma unit grid.
int chroma_count_units(int unit_size, int frame_size);
struct ChromaUnitGrid {
  int plane_width = 0;
  int plane_height = 0;
  int unit_size = 0;
  int unit_cols = 0;
  int unit_rows = 0;
  int count() const { return unit_cols * unit_rows; }
  void index(int unit_row, int unit_col, std::size_t* at) const;
};
bool chroma_make_unit_grid(int plane_width, int plane_height, int unit_size,
                           ChromaUnitGrid* grid, std::string* error);

// ---------------------------------------------------------------------------
// Chroma Wiener-NS filter model (§5.20.10.6, §7.20.3)
// ---------------------------------------------------------------------------
//
// Coefficient layout (identical to the decoder's LrWienerNs stride 18):
//   j =  0..11  Wiener_Ns_Config_Uv taps (chroma samples)
//   j = 12..17  chroma-from-luma taps (downsampled luma samples)
//
// Subsets (`wiener_ns_length`):
//   subset 0: taps 0..5 present      (6 coded coefficients)
//   subset 1: taps 0..9 present      (10 coded coefficients)
//   subset 2: taps 0..17 present     (18 coded coefficients)
//
// `uv_sym` mirrors pairs for every present j >= 6; the decoder copies
// coefficient j into j+1 and skips the second value.  It is coded only when
// plane > 0 and subset > 0.
struct ChromaFilter {
  int subset = 0;
  bool uv_sym = false;
  std::int16_t coeffs[rest::kWienerNsCoeffStride] = {};
  bool merged = false;   // reuse a bank entry (no coefficients coded)
  int use_bank_k = -1;   // -1 selects the newest entry
};

// Number of tap groups present for a chroma subset.
int chroma_present_coeffs(int subset);
// Number of coefficient values actually coded for (subset, uv_sym).
int chroma_coded_coeffs(int subset, bool uv_sym);
// True when the filter can be represented by `subset` (all non-present taps
// are zero).  `uv_sym` also requires mirrored pairs.
bool chroma_filter_fits_subset(const ChromaFilter& filter, int subset);
// Smallest subset that can represent the filter, or -1 if no subset does.
int chroma_derive_subset(const ChromaFilter& filter);
// Mirrors pairs j and j+1 for j >= 6.  Returns false for planes other than
// U/V or when a pair is not representable in the chosen subset.
bool chroma_mirror_filter(ChromaFilter* filter, std::string* error);
// Full syntax-range and layout validation.
bool chroma_filter_validate(const ChromaFilter& filter, std::string* error);
// Sets every absent tap to zero (the decoder infers 0 for those taps).
void chroma_zero_absent_taps(ChromaFilter* filter);

// Exact integer-filter SSE for one restoration unit against `target`
// (encoder source plane, plane coordinates).  Uses the same stripe-aware
// source sampler as the decoder.
double chroma_filter_sse(const rest::LrCoreFrame& core, int plane, int unit_row,
                         int unit_col, const std::uint8_t* target,
                         int target_stride, const ChromaFilter& filter);

// ---------------------------------------------------------------------------
// Encoder decision (§5.18.7.11 + §7.20.3, real recon vs source)
// ---------------------------------------------------------------------------

struct ChromaLrInput {
  // Chroma plane extents (U and V; luma is not filtered here).
  int plane_width[2] = {0, 0};
  int plane_height[2] = {0, 0};
  // Decoder-side inputs: deblocked (pre-CDEF) and post-CDEF/CCSO planes.
  // When CDEF is off both point at the same reconstruction.
  const std::uint8_t* curr[2] = {nullptr, nullptr};
  int curr_stride[2] = {0, 0};
  const std::uint8_t* cdef[2] = {nullptr, nullptr};
  int cdef_stride[2] = {0, 0};
  // Luma reconstruction used by the chroma-from-luma taps (§7.20.3).  The
  // filter is still chroma-only, but its last six taps read luma.
  const std::uint8_t* luma_curr = nullptr;
  int luma_curr_stride = 0;
  const std::uint8_t* luma_cdef = nullptr;
  int luma_cdef_stride = 0;
  int luma_width = 0;
  int luma_height = 0;
  // Encoder source picture for U and V (plane coordinates).
  const std::uint8_t* source[2] = {nullptr, nullptr};
  int source_stride[2] = {0, 0};

  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int sb_size = 64;
  int cfl_ds_filter_index = 0;
};

struct ChromaLrOptions {
  // -1 searches every legal chroma size for the superblock.
  int forced_shift = -1;
  bool allow_off = true;          // per-unit RESTORE_NONE
  bool allow_uv_sym = true;       // mirrored pairs
  bool allow_shared_uv = true;    // try one filter for both U and V
  bool allow_bank_merge = true;   // reuse the previous bank entry
  int forced_subset = -1;         // 0..2, or -1 to search
  // RD weight in SSE per bit.  0 selects pure distortion (tests).
  double lambda_bits = 0.0;
  // Number of previous units whose bank entry may be merged.  The spec's
  // LR_BANK_SIZE is 4; -1 keeps the full bank.
  int max_merge_distance = -1;
};

struct ChromaUnitPlan {
  int plane = 1;        // 1 = U, 2 = V
  int unit_row = 0;
  int unit_col = 0;
  bool on = false;      // RESTORE_WIENER_NONSEP vs RESTORE_NONE
  ChromaFilter filter;
  double sse = 0.0;
  double identity_sse = 0.0;
  int bits = 0;
  double cost = 0.0;
};

struct ChromaDecision {
  int shift = 2;
  int unit_size = 64;
  int sb_size = 64;
  ChromaUnitGrid grid[2];
  std::vector<ChromaUnitPlan> units;  // U units then V units, row-major
  bool shared_uv = false;
  double sse = 0.0;
  double identity_sse = 0.0;
  int header_bits = 0;
  int tile_bits = 0;
  double cost = 0.0;
  std::vector<std::string> notes;
};

// Chooses per-unit chroma filters by minimising
//   SSE(recon after §7.20.3, source) + lambda_bits * exact signalled bits.
// `input.curr`/`cdef` must be the real reconstruction; `source` the original
// picture.  The result maps 1:1 onto §5.20.10.4 read_lr() for planes 1 and 2.
bool chroma_decide(const ChromaLrInput& input, const ChromaLrOptions& options,
                   ChromaDecision* decision, std::string* error);

// Fills fs::LrParams for a chroma-only decision.  Luma stays RESTORE_NONE.
// `frame_filters_on` is always false: this module codes per-unit filters.
bool chroma_make_lr_params(const ChromaDecision& decision, fs::LrParams* params,
                           std::string* error);

// Builds the fs::LrState used by the writer and reader.
bool chroma_init_tile_state(const ChromaDecision& decision,
                            const ChromaLrInput& input, fs::LrState* state,
                            std::string* error);

// Exact §5.18.7.11 header cost of the chroma-only configuration.
bool chroma_measure_header_bits(const ChromaDecision& decision, int sb_size,
                                int base_q_idx, int* bits, std::string* error);

// ---------------------------------------------------------------------------
// Exact tile syntax: §5.20.10.5 read_lr_unit() for planes 1 and 2
// ---------------------------------------------------------------------------

// Writes the chroma units covered by one superblock in the exact
// §5.20.10.4 read_lr() order (plane 1 units, then plane 2 units, each in
// row-major order).  Luma units are not emitted.
bool chroma_write_read_lr(av2::Encoder& encoder, fs::LrState& state,
                          int mi_row, int mi_col, int w4, int h4,
                          int tile_mi_row_start, int tile_mi_col_start,
                          const ChromaDecision& decision, std::string* error);

bool chroma_read_read_lr(av2::Decoder& decoder, fs::LrState& state, int mi_row,
                         int mi_col, int w4, int h4, int tile_mi_row_start,
                         int tile_mi_col_start, ChromaDecision* decision,
                         std::string* error);

// Single-unit forms (used by the pinned goldens).  The caller must set
// `decoder.disable_cdf_update` to the stream's value before reading; the
// writer/reader pair in this module uses 1 (the M1/M2 default).
bool chroma_write_unit(av2::Encoder& encoder, fs::LrState& state, int plane,
                       int unit_row, int unit_col, const ChromaUnitPlan& plan,
                       std::string* error);
bool chroma_read_unit(av2::Decoder& decoder, fs::LrState& state, int plane,
                      int unit_row, int unit_col, ChromaUnitPlan* plan,
                      std::string* error);

// Direct read/write of a unit through the shared §5.20.10.6 coder without a
// ChromaUnitPlan (the plan is derived from the filter).  Exposed so a caller
// can pin bytes against the existing luma syntax.
bool chroma_write_unit_decision(av2::Encoder& encoder, fs::LrState& state,
                                int plane, int unit_row, int unit_col,
                                const ChromaFilter& filter, bool on,
                                std::string* error);

// Serialises the whole chroma LR section (all U units, then all V units, in
// §5.20.10.4 order) into one fresh coder.  This is the exact symbol payload
// for the frame's chroma restoration; it excludes other tile syntax and the
// shared §8.2 termination with the partition stream.
bool chroma_measure_decision_bytes(const ChromaDecision& decision,
                                   const ChromaLrInput& input,
                                   std::vector<std::uint8_t>* bytes,
                                   std::string* error);

// Serialises one filter's symbols into a fresh coder and returns the payload
// bytes.  Deterministic; used for RD bit counts and pinned goldens.
bool chroma_measure_unit_bytes(const fs::LrState& state, int plane,
                               int unit_row, int unit_col,
                               const ChromaFilter& filter, bool on,
                               std::vector<std::uint8_t>* bytes,
                               std::string* error);

// ---------------------------------------------------------------------------
// Quality measurement
// ---------------------------------------------------------------------------

// Integer plane PSNR of `a` against `b` (8-bit, plane coordinates).
double chroma_psnr(const std::uint8_t* a, int a_stride, const std::uint8_t* b,
                   int b_stride, int width, int height);

// Applies the decision with the reference filter core and returns the output
// planes (U, V).  `out_stride` must be >= plane width.
rest::RestorationStatus chroma_apply(const ChromaLrInput& input,
                                     const ChromaDecision& decision,
                                     std::uint8_t* out[2], int out_stride[2]);

}  // namespace av2enc::lr_chroma

#endif  // AV2ENC_ENCODER_FILTERS_LR_CHROMA_CHROMA_LR_H_
