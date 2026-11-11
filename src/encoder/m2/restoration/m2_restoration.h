// AV2 M2 loop-restoration adapters: CCSO (§7.19) + Wiener NS (§7.20).
//
// This layer owns the *M2-specific* glue around the already-verified M1
// filter/restoration syntax module (`src/encoder/m1/filters_stream/`) and the
// CPU/GPU filter cores (`src/encoder/filters/restoration/`).  It does not
// reimplement either; every symbol is written and read through the single
// filters_stream implementation so an M2 tile and an M1 still tile share one
// codepath.
//
// Normative order (docs/negative-results.md, §7.17–§7.20):
//
//   reconstruct -> deblock -> CDEF -> CCSO -> loop restoration -> output
//
// The M2 core tile writer currently emits only partition/intra/inter/residual
// symbols.  It has no hook for the per-superblock `read_lr()` call or the
// per-unit `read_ccso()` call, so this module exposes the exact encoder-side
// call sites:
//
//   sequence header:  write_sequence_filter_config()  §5.4.10
//   frame header:     write_deblock_params()          §5.18.5.2
//                     write_cdef_params()             §5.18.7.10
//                     write_lr_params()               §5.18.7.11
//                     write_ccso_params()             §5.18.7.12
//   tile, per SB:     write_read_lr()                 §5.20.10.4
//                     before write_partition()
//   tile, per unit:   write_read_ccso()               §5.20.10.2
//                     at every CCSO unit origin
//
// The mirror readers (`read_lr_params`, `read_ccso_params`, `read_lr`,
// `read_ccso`, `read_lr_unit`, `read_wienerns_filter`) are provided for
// bitstream round-trip gates and for a future decoder.
//
// Blockers that stay explicit (see README.md):
//   * PC-Wiener (§7.20.4) needs the `pc_wiener_*` lookup tables that are not
//     in the spec text.  frame_type == RESTORE_PC_WIENER and switchable units
//     that select it are rejected.
//   * Adaptive frame-filter *group search* is not implemented.  The writer
//     supports the deterministic group-0 layout only (all classes are in the
//     first group; no `use_alt_group` symbol).  That is valid syntax, just not
//     rate-optimal.  `validate_frame_filter_layout()` checks it up front.
//   * Multi-class frame filters need an encoder-supplied §7.20.4 class map.
//     The single-class layout (`num_filter_classes == 1`) needs no class map
//     and is the gated path.
#ifndef AV2ENC_ENCODER_M2_RESTORATION_M2_RESTORATION_H_
#define AV2ENC_ENCODER_M2_RESTORATION_M2_RESTORATION_H_

#include "bitstream/bitwriter.h"
#include "encoder/filters/restoration/restoration.h"
#include "encoder/m1/filters_stream/filters_stream.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::m2::restoration {

namespace fs = av2enc::m1::filters_stream;
namespace rest = ::av2enc::restoration;

// ---------------------------------------------------------------------------
// Tool selection and gates
// ---------------------------------------------------------------------------

// Tools enabled for one M2 sequence.  `ccso` and `wiener_ns` may be combined;
// the frame-header and tile adapters below produce the matching rows.
struct M2RestorationTools {
  bool ccso = false;
  bool ccso_unit_matches_sb_size = true;
  bool wiener_ns = false;
  // Only meaningful with wiener_ns: signal frame-level filters
  // (§5.18.7.11 frame_filters_on) instead of per-unit filters.
  bool frame_filters = false;
  int num_filter_classes = 1;
  // Chroma loop restoration is signalled when any chroma plane uses LR.
  bool chroma_lr = false;
};

// Reasons a configuration cannot be turned into a conforming M2 stream or a
// filter call.  Every failure is reported with an actionable string because
// the missing normative data is a known blocker, not a coding error.
bool validate_tools(const M2RestorationTools& tools, std::string* error);

// §5.4.10 sequence_filter_config() for M2 (single_picture_header_flag = 0).
// The flags are intentionally identical to the M1 module so one sequence row
// serves both stills and P-frame streams.  PC-Wiener is always disabled.
fs::SequenceFilterConfig make_sequence_filter_config(
    const M2RestorationTools& tools, std::string* error);

std::size_t write_sequence_filter_config(BitWriter& writer,
                                         const M2RestorationTools& tools,
                                         std::string* error);
bool read_sequence_filter_config(fs::BitReader& reader, M2RestorationTools* tools,
                                 std::string* error);

// ---------------------------------------------------------------------------
// Frame header: §5.18.7.11 lr_params() and §5.18.7.12 ccso_params()
// ---------------------------------------------------------------------------

// Fills the zero-valued context fields from an M2 frame geometry.  Callers
// override frame_is_intra / base_q_idx / tile_start_or as needed.
fs::LrHeaderContext make_lr_header_context(const M2RestorationTools& tools,
                                            int sb_size, int base_q_idx,
                                            bool frame_is_intra);
fs::CcsoHeaderContext make_ccso_header_context(const M2RestorationTools& tools,
                                               int sb_size, bool frame_is_intra,
                                               int tile_start_or);

std::size_t write_lr_params(BitWriter& writer, fs::LrParams& params,
                            const fs::LrHeaderContext& context,
                            std::string* error);
bool read_lr_params(fs::BitReader& reader, const fs::LrHeaderContext& context,
                    fs::LrParams* params, std::string* error);

std::size_t write_ccso_params(BitWriter& writer, fs::CcsoParams& params,
                              const fs::CcsoHeaderContext& context,
                              std::string* error);
bool read_ccso_params(fs::BitReader& reader, const fs::CcsoHeaderContext& context,
                      fs::CcsoParams* params, std::string* error);

// Converts the parsed CCSO header into the CPU/GPU filter parameters.  The
// luminance block grid is derived from the plane geometry; every unit is
// marked enabled (the per-unit grid is then filled by read_ccso()).
//
// `out->blks` points into `state.blks`, so `state` must outlive the filter
// call.  `out->plane[p].offsets` is filled by ccso_fill_offsets().
bool make_ccso_frame_params(const fs::CcsoParams& params,
                            fs::CcsoTileState* state, int mi_cols, int mi_rows,
                            int sub_x, int sub_y,
                            rest::CcsoFrameParams* out,
                            std::string* error);

// ---------------------------------------------------------------------------
// Tile syntax: §5.20.10.2 read_ccso()
// ---------------------------------------------------------------------------

bool init_ccso_tile(fs::CcsoTileState* state, const fs::CcsoParams& params,
                    int mi_cols, int mi_rows, int sub_x, int sub_y,
                    std::string* error);

bool write_read_ccso(av2::Encoder& encoder, fs::CcsoTileState& state,
                     int mi_row, int mi_col, const int blk[3],
                     std::string* error);
bool read_ccso(av2::Decoder& decoder, fs::CcsoTileState& state, int mi_row,
               int mi_col, int blk[3], std::string* error);

// ---------------------------------------------------------------------------
// Tile syntax: §5.20.10.4 read_lr(), §5.20.10.5 read_lr_unit(),
// §5.20.10.6 read_wienerns_filter()
// ---------------------------------------------------------------------------

bool init_lr_tile(fs::LrState* state, const fs::LrParams& params, int mi_cols,
                  int mi_rows, int sub_x, int sub_y, int sb_size,
                  std::string* error);

int lr_block_unit_count(const fs::LrState& state, int mi_row, int mi_col,
                        int w4, int h4, int tile_mi_row_start,
                        int tile_mi_col_start);

// Builds the read_lr() decision list from the LrState grids (no invented bank
// merges or coefficients; see filters_stream::lr_build_decisions).
bool build_lr_decisions(const fs::LrState& state, int mi_row, int mi_col,
                        int w4, int h4, int tile_mi_row_start,
                        int tile_mi_col_start,
                        std::vector<fs::LrUnitDecision>* decisions,
                        std::string* error);

bool write_read_lr(av2::Encoder& encoder, fs::LrState& state, int mi_row,
                   int mi_col, int w4, int h4, int tile_mi_row_start,
                   int tile_mi_col_start,
                   const std::vector<fs::LrUnitDecision>& decisions,
                   std::string* error);
bool read_lr(av2::Decoder& decoder, fs::LrState& state, int mi_row, int mi_col,
             int w4, int h4, int tile_mi_row_start, int tile_mi_col_start,
             std::vector<fs::LrUnitDecision>* decisions, std::string* error);

bool write_read_lr_unit(av2::Encoder& encoder, fs::LrState& state, int plane,
                        int unit_row, int unit_col,
                        const fs::LrUnitDecision& decision, std::string* error);
bool read_lr_unit(av2::Decoder& decoder, fs::LrState& state, int plane,
                  int unit_row, int unit_col, fs::LrUnitDecision* decision,
                  std::string* error);

// Explicit aliases for the two §5.20.10.5/§5.20.10.6 entry points.  In this
// implementation the per-unit Wiener filter symbols are the tail of
// read_lr_unit(); the frame-level filters are coded by read_lr_params().
bool write_read_wienerns_filter(av2::Encoder& encoder, fs::LrState& state,
                                int plane, int unit_row, int unit_col,
                                const fs::LrUnitDecision& decision,
                                std::string* error);
bool read_wienerns_filter(av2::Decoder& decoder, fs::LrState& state, int plane,
                          int unit_row, int unit_col,
                          fs::LrUnitDecision* decision, std::string* error);

// ---------------------------------------------------------------------------
// §7.20.4 class map handling
// ---------------------------------------------------------------------------

// Validates the frame-filter class layout *before* any header bit is written.
//   * num_filter_classes == 1: accepted without a map (the gated single-class
//     path); all luma cells use class 0 and the §7.20.4 classifier is never
//     consulted.)
//   * num_filter_classes > 1: requires map.num_classes == num_filter_classes,
//     a mi_cols x mi_rows filter_class_grid, and a subclass_lookup whose
//     values are < num_filter_classes.  The grid values must come from a
//     §7.20.4 classifier; this module cannot derive them because the
//     Pc_Wiener_* tables are absent from the spec text.
bool validate_class_map(const fs::LrParams& params, int mi_cols, int mi_rows,
                        std::string* error);

// Fills the single-class map (class 0, subclass 0) for the gated path.
bool make_single_class_map(fs::WienerNsClassMap* map, int mi_cols, int mi_rows,
                           std::string* error);

// Rejects PC-Wiener in the frame type or in any unit of a switchable plane.
bool validate_no_pc_wiener(const fs::LrParams& params, int mi_cols, int mi_rows,
                           std::string* error);

// Validates the deterministic group-0 frame-filter layout the writer supports
// (no adaptive group search).  match_index 0 is the all-zero filter; class c
// may match 0..c.
bool validate_frame_filter_layout(const fs::LrParams& params,
                                  std::string* error);

// ---------------------------------------------------------------------------
// CPU filter application (M2 chain: deblock/CDEF/CCSO already applied)
// ---------------------------------------------------------------------------

// Applies CCSO to `frame.cdef` in place from parsed params + a tile state.
// `frame.curr` must be the pre-CCSO (post-CDEF) buffer and must not alias
// `frame.cdef` for an enabled plane.
rest::RestorationStatus apply_ccso_frame(
    const rest::RestorationFrame& frame, const fs::CcsoParams& params,
    const fs::CcsoTileState& state);

// Applies §7.20 into `frame.lr`; `frame.curr` is the post-deblock buffer used
// for samples outside the current stripe, `frame.cdef` the post-CCSO buffer.
rest::RestorationStatus apply_lr_frame(
    const rest::RestorationFrame& frame, const fs::LrState& state);

// Convenience: applies CCSO (if the state has a frame flag) and then loop
// restoration in the normative order.  Returns the first failure.
rest::RestorationStatus apply_restoration_frame(
    const rest::RestorationFrame& frame,
    const fs::CcsoTileState* ccso_state, const fs::LrState* lr_state);

// ---------------------------------------------------------------------------
// Exact syntax cost
// ---------------------------------------------------------------------------

// Header + tile-symbol cost of one frame's restoration tools, measured through
// the real writers.  The arithmetic-coder symbol bytes come from a standalone
// av2::Encoder: arithmetic coding is additive over symbols, so this is the
// cost of the CCSO/LR symbols themselves up to termination/padding bytes.  The
// surrounding M2 partition syntax shares the same coder state but does not
// change the per-symbol intervals.
struct M2RestorationCost {
  std::size_t sequence_bits = 0;      // delta versus filters off
  std::size_t ccso_header_bits = 0;
  std::size_t lr_header_bits = 0;
  std::size_t ccso_symbol_bytes = 0;
  std::size_t lr_symbol_bytes = 0;
  std::size_t total_bits() const {
    return sequence_bits + ccso_header_bits + lr_header_bits +
           8 * (ccso_symbol_bytes + lr_symbol_bytes);
  }
};

// Measures the cost for one frame.  `ccso` / `lr` may be null when the tool is
// disabled.  `ccso_blks` is the per-unit decision grid the encoder will signal
// (all ones for "filter this unit"); `lr_state` already owns the unit grids.
bool measure_restoration_cost(const M2RestorationTools& tools, int mi_cols,
                              int mi_rows, int sb_size, int base_q_idx,
                              bool frame_is_intra, int tile_start_or,
                              const fs::CcsoParams* ccso,
                              const fs::CcsoTileState* ccso_state,
                              fs::LrState* lr_state, M2RestorationCost* cost,
                              std::string* error);

}  // namespace av2enc::m2::restoration

#endif  // AV2ENC_ENCODER_M2_RESTORATION_M2_RESTORATION_H_
