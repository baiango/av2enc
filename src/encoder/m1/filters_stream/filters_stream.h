// M1 filter/restoration bitstream syntax: sequence flags, frame headers and
// tile symbols for CDEF, CCSO and non-separable Wiener loop restoration.
//
// This module owns *syntax only*.  It is host-side, allocation-free in the
// per-block paths (the state objects own their grids), and does not decide
// filter parameters or touch pixels.  The encoder-side call sites are listed in
// README.md; they are placed in the exact order the AV2 v1.0.0 syntax uses:
//
//   sequence header:  §5.4.10 sequence_filter_config()
//   frame header:     §5.18.7.10  cdef_params()
//                     §5.18.7.11  lr_params()          <- frame filters only
//                     §5.18.7.12  ccso_params()
//   tile:             §5.20.10.1 read_cdef()   per coding block
//                     §5.20.10.2 read_ccso()   per coding block
//                     §5.20.10.4 read_lr()     once per superblock, BEFORE
//                                              read_partition()
//                     §5.20.10.5 read_lr_unit()
//                     §5.20.10.6 read_wienerns_filter()
//
// Header syntax uses the raw descriptors f(n)/ns(n)/tu(mx) and therefore a
// BitWriter/BitReader.  Frame-level Wiener filters are raw-bit coded too
// (`f(1)`, `L(1)`, `ns()` in decode_subexp); only per-unit Wiener filters use
// the §8.2 arithmetic coder (`S()`).
//
// Spec-only.  The normative CDFs are in filter_cdfs.h and the CDEF tables in
// src/cuda/m1/cdef/m1_cdef_cdf.h.
#ifndef AV2ENC_ENCODER_M1_FILTERS_STREAM_FILTERS_STREAM_H_
#define AV2ENC_ENCODER_M1_FILTERS_STREAM_FILTERS_STREAM_H_

#include "bitstream/bitwriter.h"
#include "cuda/m1/cdef/m1_cdef_index_syntax.h"
#include "encoder/filters/restoration/restoration.h"
#include "encoder/filters/restoration/restoration_common.h"
#include "encoder/m1/filters_stream/bit_io.h"
#include "encoder/m1/filters_stream/filter_cdfs.h"
#include "entropy/av2/av2_entropy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::m1::filters_stream {

using restoration::kRestoreNone;
using restoration::kRestorePcWiener;
using restoration::kRestoreSwitchable;
using restoration::kRestoreWienerNs;

// ---------------------------------------------------------------------------
// §5.4.10 sequence_filter_config()
// ---------------------------------------------------------------------------

struct SequenceFilterConfig {
  bool disable_loopfilters_across_tiles = false;
  bool enable_cdef = false;
  bool enable_gdf = false;
  bool enable_restoration = false;
  // lr_tools_disable[0][RESTORE_PC_WIENER] and
  // lr_tools_disable[0][RESTORE_WIENER_NONSEP].
  bool lr_disable_luma_pc_wiener = true;
  bool lr_disable_luma_wiener_ns = false;
  // If true, lr_tools_disable[1][RESTORE_WIENER_NONSEP] is signalled;
  // otherwise it is inferred from the luma value.
  bool lr_uv_present = false;
  bool lr_disable_chroma_wiener_ns = false;
  bool enable_ccso = false;
  bool ccso_unit_matches_sb_size = true;
  // Only coded when single_picture_header_flag is false.
  bool cdef_on_skip_txfm_always_on = false;
  bool cdef_on_skip_txfm_disabled = false;
  std::uint8_t df_par_bits_minus_2 = 0;
};

// Writes the complete sequence_filter_config() row.  GDF is not supported by
// M1 (enable_gdf must be false); enabling it returns 0 and sets `error`.
std::size_t write_sequence_filter_config(BitWriter& writer,
                                         const SequenceFilterConfig& config,
                                         bool single_picture_header_flag,
                                         std::string* error);

// Mirror.  `sb_64x64` is only needed when a future GDF flag is present; M1
// rejects enable_gdf.
bool read_sequence_filter_config(BitReader& reader,
                                 bool single_picture_header_flag,
                                 bool sb_64x64,
                                 SequenceFilterConfig* config,
                                 std::string* error);

// ---------------------------------------------------------------------------
// §5.18.7.10 cdef_params()
// ---------------------------------------------------------------------------

// Decoded-domain strength fields: y_pri/uv_pri in 0..15, y_sec/uv_sec in
// {0,1,2,4}.  The writer derives the pri_zero shortcut and maps 4 -> 3.
struct CdefStrength {
  int y_pri = 0;
  int y_sec = 0;
  int uv_pri = 0;
  int uv_sec = 0;
};

struct CdefParams {
  bool frame_enable = false;
  int damping_minus_3 = 0;      // f(2)
  int strengths_minus_1 = 0;    // f(3)
  bool on_skip_txfm_frame_enable = false;  // f(1) when CDEF_ON_SKIP_TXFM_ADAPTIVE
  int num_strengths = 1;        // CdefStrengths
  CdefStrength set[8] = {};
};

struct CdefHeaderContext {
  bool coded_lossless = false;
  bool enable_cdef = false;
  // M1 still pictures set single_picture_header_flag, which infers
  // cdef_frame_enable = 1 and selects CDEF_ON_SKIP_TXFM_ADAPTIVE.
  bool single_picture_header_flag = true;
  // CDEF_ON_SKIP_TXFM_DISABLED (0), ALWAYS_ON (1) or ADAPTIVE (2).  A
  // single_picture_header_flag forces ADAPTIVE.
  int cdef_on_skip_txfm = 2;
  int num_planes = 3;
};

std::size_t write_cdef_params(BitWriter& writer, const CdefParams& params,
                              const CdefHeaderContext& context,
                              std::string* error);

bool read_cdef_params(BitReader& reader, const CdefHeaderContext& context,
                      CdefParams* params, std::string* error);

// §5.20.10.1 read_cdef().  The existing helper in
// src/cuda/m1/cdef/m1_cdef_index_syntax.h is the single implementation of the
// §8.3.1 context; these wrappers only bind it to the §8.2 coder.
struct CdefTileState {
  av2enc::cuda::m1::CdefIndexGrid grid;
  std::uint16_t index0[4][3] = {};
  std::uint16_t minus1[8] = {};
  int strengths = 1;
  int sb_width_log2 = 4;
};

bool cdef_tile_init(CdefTileState* state, int mi_cols, int mi_rows,
                    int strengths, int sb_width_log2, std::string* error);

// §5.20.9.6 clear_cdef() for the 64x64 block containing (mi_row, mi_col).
void cdef_clear_superblock(CdefTileState* state, int mi_row, int mi_col);

bool cdef_write_block(av2::Encoder& encoder, CdefTileState& state,
                      int tile_mi_col_start, int tile_mi_row_start,
                      int mi_row, int mi_col, int idx, std::string* error);

bool cdef_read_block(av2::Decoder& decoder, CdefTileState& state,
                     int tile_mi_col_start, int tile_mi_row_start,
                     int mi_row, int mi_col, int* idx, std::string* error);

// ---------------------------------------------------------------------------
// §5.18.7.12 ccso_params() and §5.20.10.2 read_ccso()
// ---------------------------------------------------------------------------

struct CcsoPlaneParams {
  bool enabled = false;
  bool bo_only = false;
  int scale_idx = 0;       // f(2)
  int quant_idx = 0;       // f(2)
  int ext_filter = 0;      // f(3)
  int edge_clf = 0;        // f(1)
  int max_band_log2 = 0;   // f(2 + bo_only)
  // ccso_offset_idx values in syntax order (d0, d1, band), each tu(7).
  std::vector<std::uint8_t> offset_idx;
};

struct CcsoParams {
  bool frame_flag = false;
  int ccso_luma_size_log2 = 6;
  CcsoPlaneParams plane[3];
  // Intra frames set both to false and code no bits for them.
  bool reuse_ccso[3] = {false, false, false};
  bool sb_reuse_ccso[3] = {false, false, false};
};

struct CcsoHeaderContext {
  bool coded_lossless = false;
  bool enable_ccso = false;
  bool single_picture_header_flag = true;
  bool frame_is_intra = true;
  int num_planes = 3;
  int sb_size = 64;
  bool ccso_unit_matches_sb_size = true;
  // OR of MiColStarts[] | MiRowStarts[]; only the low six bits matter.
  int tile_start_or = 0;
};

// Fills params.ccso_luma_size_log2 from the header rule and codes the row.
std::size_t write_ccso_params(BitWriter& writer, CcsoParams& params,
                              const CcsoHeaderContext& context,
                              std::string* error);

bool read_ccso_params(BitReader& reader, const CcsoHeaderContext& context,
                      CcsoParams* params, std::string* error);

// Fills restoration::CcsoParams.offsets from params.offset_idx.  Does not link
// restoration.cpp: the offset table is inline in restoration_common.h.
bool ccso_fill_offsets(const CcsoPlaneParams& params,
                       restoration::CcsoParams* out, std::string* error);

// Number of ccso_offset_idx values a plane reads.
int ccso_offset_count(const CcsoPlaneParams& params);

struct CcsoTileState {
  CcsoParams params;
  int ccso_luma_size_log2 = 6;
  int mi_cols = 0;
  int mi_rows = 0;
  int tile_mi_col_start = 0;
  int tile_mi_row_start = 0;
  std::vector<std::uint8_t> blks[3];
  int blk_cols[3] = {0, 0, 0};
  int blk_rows[3] = {0, 0, 0};
  std::uint16_t cdf[3][4][3] = {};
};

bool ccso_tile_init(CcsoTileState* state, const CcsoParams& params,
                    int mi_cols, int mi_rows, int sub_x, int sub_y,
                    std::string* error);

// `blk` is the CcsoBlks value for each plane (ignored when the plane is
// disabled).  A call at a non-unit-origin block is a no-op.
bool ccso_write_block(av2::Encoder& encoder, CcsoTileState& state, int mi_row,
                      int mi_col, const int blk[3], std::string* error);

bool ccso_read_block(av2::Decoder& decoder, CcsoTileState& state, int mi_row,
                     int mi_col, int blk[3], std::string* error);

// ---------------------------------------------------------------------------
// §5.18.7.11 lr_params() and §5.20.10.4–§5.20.10.6
// ---------------------------------------------------------------------------

// Encoder-side §7.20.4 class map.  The bitstream only signals
// num_filter_classes_idx; the decoder derives SubclassLookup from
// Pc_Wiener_Sub_Classify2 (an attachment header, not text in the spec).  A real
// encoder must therefore supply the same lookup table; `has_subclass_lookup`
// makes that requirement explicit.
struct WienerNsClassMap {
  int num_classes = 1;
  std::vector<std::uint8_t> filter_class_grid;  // luma MI grid, values 0..15
  std::array<std::uint8_t, 16> subclass_lookup{};
  bool has_subclass_lookup = false;
  // Legacy pre-looked-up grid accepted by the restoration stage.
  std::vector<std::uint8_t> subclass_grid;
};

struct WienerNsFrameFilter {
  // matchIndices[c]: 0 = all-zero filter, m > 0 = FrameLrWienerNs[m - 1].
  int match_index = 0;
  bool merged = false;   // f(1): reuse the matched filter
  int subset = 0;        // 0..3 luma, 0..2 chroma
  bool uv_sym = false;   // chroma symmetry (subset > 0 only)
  std::int16_t coeffs[restoration::kWienerNsCoeffStride] = {};
};

struct LrParams {
  int frame_type[3] = {kRestoreNone, kRestoreNone, kRestoreNone};
  bool frame_filters_on[3] = {false, false, false};
  int num_filter_classes = 1;      // luma NumFilterClasses
  int num_filter_classes_idx = 0;  // Decode_Num_Filter_Classes index
  // Final shift: luma size = 512 >> luma_shift,
  //             chroma size = (512 >> max(subX,subY)) >> chroma_shift.
  int luma_shift = 3;
  int chroma_shift = 3;
  WienerNsFrameFilter frame_filter[3][16];
  WienerNsClassMap class_map;
};

struct LrHeaderContext {
  bool coded_lossless = false;
  bool enable_restoration = false;
  bool single_picture_header_flag = true;
  // M1 is intra.  Temporal prediction of frame filters is not supported.
  bool frame_is_intra = true;
  int sb_size = 64;
  int num_planes = 3;
  int sub_x = 1;
  int sub_y = 1;
  int base_q_idx = 90;
  // lr_tools_disable[chroma?1:0][restoration type].  PC-Wiener must stay
  // disabled for M1.
  bool lr_tools_disable[2][4] = {
      {false, true, false, false},
      {false, true, false, false},
  };
};

std::size_t write_lr_params(BitWriter& writer, LrParams& params,
                            const LrHeaderContext& context,
                            std::string* error);

bool read_lr_params(BitReader& reader, const LrHeaderContext& context,
                    LrParams* params, std::string* error);

// Largest per-plane unit grid, used by the frame filters and the §7.20 stage.
int lr_count_units(int unit_size, int frame_size);

struct LrUnitRange {
  int row_begin = 0;
  int row_end = 0;
  int col_begin = 0;
  int col_end = 0;
};

struct LrUnitDecision {
  int type = kRestoreNone;
  // Per-unit filter syntax (only used when the frame type is
  // RESTORE_WIENER_NONSEP and frame_filters_on[plane] is false).
  bool merged = false;
  // k = 0..bank_size-2 selects a reference bank; -1 chooses the newest
  // previous bank automatically.
  int use_bank_k = -1;
  int subset = 0;
  bool uv_sym = false;
  std::int16_t coeffs[restoration::kWienerNsCoeffStride] = {};
};

// Full tile state: frame-filter coefficients, per-unit grids and the §8.3 CDF
// copies.  One per tile (M1 has one tile), never shared between tiles.
struct LrState {
  LrParams params;
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  int sb_size = 64;

  int unit_size[3] = {64, 64, 64};
  int unit_cols[3] = {0, 0, 0};
  int unit_rows[3] = {0, 0, 0};

  // LrType[plane][unitRow][unitCol] and LrWienerNs[...][coeff].
  std::vector<std::uint8_t> unit_type[3];
  std::vector<std::int16_t> unit_coeffs[3];

  // FrameLrWienerNs[plane][class][coeff].
  std::int16_t frame_coeffs[3][16][restoration::kWienerNsCoeffStride] = {};

  // RefLrWienerNs[plane][class][bank][coeff] + bank bookkeeping.
  std::int16_t ref_bank[3][16][4][restoration::kWienerNsCoeffStride] = {};
  int bank_ptr[3][16] = {};
  int bank_size[3][16] = {};

  // CDF copies (§9.3).
  std::uint16_t use_wiener_ns[3] = {};
  std::uint16_t flex_restoration_type[3][3][3] = {};
  std::uint16_t wiener_ns_length[2][3] = {};
  std::uint16_t wiener_ns_uv_sym[3] = {};
  std::uint16_t wiener_ns_base[5] = {};
};

// Initialises CDFs, banks (as decode_tile() does), unit grids and the frame
// filter state from `params`.
bool lr_state_init(LrState* state, const LrParams& params, int mi_cols,
                   int mi_rows, int sub_x, int sub_y, int sb_size,
                   std::string* error);

// §5.20.10.4 read_lr(): the restoration units covered by one superblock.
// `w4`/`h4` are Num_4x4_Blocks_Wide/High of the block (16 for a 64x64 SB).
bool lr_unit_range(const LrState& state, int plane, int mi_row, int mi_col,
                   int w4, int h4, int tile_mi_row_start,
                   int tile_mi_col_start, LrUnitRange* range,
                   std::string* error);

// Number of units read_lr() visits for one block, all planes.
int lr_block_unit_count(const LrState& state, int mi_row, int mi_col, int w4,
                        int h4, int tile_mi_row_start,
                        int tile_mi_col_start);

// Convenience for the integrator: builds the read_lr() decision list from
// LrState::unit_type and LrState::unit_coeffs.  It uses merged = false, picks
// the smallest Wiener_Ns_Taps_Present subset that covers every non-zero
// coefficient, and sets uv_sym when the chroma pairs are equal.  It does not
// invent coefficients or bank merges.
bool lr_build_decisions(const LrState& state, int mi_row, int mi_col, int w4,
                        int h4, int tile_mi_row_start,
                        int tile_mi_col_start,
                        std::vector<LrUnitDecision>* decisions,
                        std::string* error);

// Writes read_lr() for one superblock.  `decisions` must contain exactly
// lr_block_unit_count() entries, in the order read_lr() visits them
// (plane-major, then unitRow, then unitCol).
bool lr_write_block(av2::Encoder& encoder, LrState& state, int mi_row,
                    int mi_col, int w4, int h4, int tile_mi_row_start,
                    int tile_mi_col_start,
                    const std::vector<LrUnitDecision>& decisions,
                    std::string* error);

// Mirror.  Fills `decisions` (when non-null) and the LrType/coefficient grids.
bool lr_read_block(av2::Decoder& decoder, LrState& state, int mi_row,
                   int mi_col, int w4, int h4, int tile_mi_row_start,
                   int tile_mi_col_start,
                   std::vector<LrUnitDecision>* decisions, std::string* error);

// §5.20.10.5 read_lr_unit() / §5.20.10.6 read_wienerns_filter(), exposed for
// callers that already know the unit indices.
bool lr_write_unit(av2::Encoder& encoder, LrState& state, int plane,
                   int unit_row, int unit_col,
                   const LrUnitDecision& decision, std::string* error);
bool lr_read_unit(av2::Decoder& decoder, LrState& state, int plane,
                  int unit_row, int unit_col, LrUnitDecision* decision,
                  std::string* error);

}  // namespace av2enc::m1::filters_stream

#endif  // AV2ENC_ENCODER_M1_FILTERS_STREAM_FILTERS_STREAM_H_
