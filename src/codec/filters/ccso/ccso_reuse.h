// AV2 v1.0.0 temporal CCSO reuse — syntax, slot store and encoder decisions.
//
// Normative references:
//   §5.18.7.12  ccso_params(): reuse_ccso / sb_reuse_ccso / ccso_ref_idx
//   §6.17.7.8   CCSO params semantics and the conformance requirements
//   §5.20.10.2  read_ccso(): ccso_blk is skipped when sb_reuse_ccso[plane]
//   §7.19/§7.19.1 apply CCSO filter process ("useCcso")
//   §7.23       reference frame update: save_ccso_params(i, plane) / load
//
// This module is spec-only and self-contained: it does not modify the M1/M2
// filter components.  It owns
//   * the exact §5.18.7.12 header row including the reuse bits and the
//     f(CeilLog2(NumTotalRefs)) reference selector,
//   * an independent reader for the same row (it never calls the writer),
//   * §7.23 save/load of CcsoLumaSizeLog2, per-plane parameters,
//     CcsoFilterOffset and CcsoBlks per reference slot,
//   * the encoder-side `use_ccso()` decision (which plane parameters and block
//     flags can be inherited from a reference, and which reference index),
//   * the §5.20.10.2 per-superblock `ccso_blk S()` hook that is skipped when
//     sb_reuse_ccso is set.
//
// Scope: 8-bit 4:2:0, matching the rest of src/encoder/filters.  Every
// validation below is a §6.17.7.8 requirement, not an encoder preference.
//
// Default-stream safety: when CCSO is disabled (the default in every existing
// sequence header) write_ccso_params() emits zero bits and read_ccso_params()
// consumes zero bits, so a call site can be added unconditionally and default
// streams stay byte-identical.  For CCSO-enabled streams without reuse the
// writer is byte-identical to the existing non-reuse writer in
// src/encoder/m1/filters_stream/filters_stream.cpp (pinned by the unit test).
#ifndef AV2ENC_ENCODER_FILTERS_CCSO_REUSE_CCSO_REUSE_H_
#define AV2ENC_ENCODER_FILTERS_CCSO_REUSE_CCSO_REUSE_H_

#include "core/bitstream/bitwriter.h"
#include "codec/filters/loop_restoration/restoration.h"
#include "codec/filters/stream/bit_io.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::ccso_reuse {

using BitReader = av2enc::still::filters_stream::BitReader;

// §3 constants.
inline constexpr int kNumRefFrames = 16;       // NUM_REF_FRAMES
inline constexpr int kCcsoBandNum = 64;        // CCSO_BAND_NUM
inline constexpr int kCcsoInputInterval = 3;   // CCSO_INPUT_INTERVAL
inline constexpr int kCcsoLumaSizeLog2 = 8;    // CCSO_LUMA_SIZE_LOG2
inline constexpr int kCcsoOffsetRange = 8;
inline constexpr int kRestrictedOrderHint = -1;  // RESTRICTED_OH

// One plane's §5.18.7.12 state.  `offset_idx` is in syntax order
// (d0, d1, band), each tu(7).  `enabled` is ccso_planes[plane].
struct PlaneParams {
  bool enabled = false;
  bool bo_only = false;
  int scale_idx = 0;      // f(2)
  int quant_idx = 0;      // f(2)
  int ext_filter = 0;     // f(3), must not be 7
  int edge_clf = 0;       // f(1)
  int max_band_log2 = 0;  // f(2 + bo_only)
  std::vector<std::uint8_t> offset_idx;
};

bool operator==(const PlaneParams& a, const PlaneParams& b);
inline bool operator!=(const PlaneParams& a, const PlaneParams& b) {
  return !(a == b);
}

// One frame's §5.18.7.12 row.
struct FrameParams {
  bool frame_flag = false;
  int ccso_luma_size_log2 = 6;
  PlaneParams plane[3];
  // Intra/SWITCH frames infer both to 0 and code no bits.
  bool reuse_ccso[3] = {false, false, false};
  bool sb_reuse_ccso[3] = {false, false, false};
  // Index into ref_frame_idx[], not a reference slot.  Coded only when
  // reuse_ccso || sb_reuse_ccso.  Zero-width when NumTotalRefs == 1.
  int ccso_ref_idx[3] = {0, 0, 0};
};

// Sizes of the CcsoBlks grid for a plane.  Mirrors
// restoration::ccso_blk_grid_size() but keeps the module independent.
void ccso_blk_grid_size(int plane_width, int plane_height,
                        int ccso_luma_size_log2, int sub_x, int sub_y,
                        int* cols, int* rows);

// §5.18.7.12 CcsoLumaSizeLog2 derivation.  `tile_start_or` is
// OR(MiColStarts[]) | OR(MiRowStarts[]).
int derive_ccso_luma_size_log2(int sb_size, int tile_start_or,
                               bool ccso_unit_matches_sb_size);

// §5.18.7.12 field representability (ranges, offset count, bo_only zeros).
bool validate_plane_params(const PlaneParams& p, std::string* error);

// Number of ccso_offset_idx values a plane reads.
int ccso_offset_count(const PlaneParams& p);

// §7.23 saved state for one (reference slot, plane).
struct SavedPlane {
  bool saved = false;         // SavedCcsoPlanes[i][plane]
  int luma_size_log2 = 0;     // SavedCcsoLumaSizeLog2[i][plane]
  PlaneParams params;         // scalars + offset_idx (offsets re-derive)
  std::vector<std::uint8_t> blks;  // CcsoBlks[plane]
  int blk_cols = 0;
  int blk_rows = 0;
};

// §7.23 saved state for one reference slot.
struct SavedSlot {
  bool valid = false;  // RefValid[i]
  int order_hint = 0;  // RefOrderHint[i]
  int mi_cols = 0;     // RefMiCols[i]
  int mi_rows = 0;     // RefMiRows[i]
  SavedPlane plane[3];
};

// The parameter area indexed by reference slot.  save_ccso_params() is
// invoked for every refreshed slot at the end of the frame (§7.23), including
// slots/planes where CCSO is off (SavedCcsoPlanes == 0).
class SlotStore {
 public:
  void reset() { slot_ = {}; }

  SavedSlot& slot(int i) { return slot_[static_cast<std::size_t>(i)]; }
  const SavedSlot& slot(int i) const {
    return slot_[static_cast<std::size_t>(i)];
  }
  bool in_range(int i) const { return i >= 0 && i < kNumRefFrames; }

 private:
  std::array<SavedSlot, kNumRefFrames> slot_{};
};

// Input to save_ccso_params() for one frame.  The caller supplies the
// *effective* parameters (after any reuse) and the final CcsoBlks grids.
struct SavedFrame {
  int ccso_luma_size_log2 = 6;
  int order_hint = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  const FrameParams* params = nullptr;
  const std::vector<std::uint8_t>* blks = nullptr;
  const int* blk_cols = nullptr;
  const int* blk_rows = nullptr;
};

// §7.23 save_ccso_params().  `i` must be a refreshed reference slot.
bool save_ccso_params(SlotStore& store, int i, const SavedFrame& frame,
                      std::string* error);

// Header context.  Fields not used by the syntax (mi_cols/mi_rows,
// num_total_refs, ref_frame_idx, slots) are required for the §6.17.7.8
// conformance checks; pass slots = nullptr for syntax-only parses.
struct HeaderContext {
  bool coded_lossless = false;
  bool enable_ccso = false;
  bool single_picture_header_flag = true;
  bool frame_is_intra = true;
  int num_planes = 3;
  int sb_size = 64;
  bool ccso_unit_matches_sb_size = true;
  int tile_start_or = 0;

  int mi_cols = 0;
  int mi_rows = 0;
  // 4:2:0 is the only depth/subsampling this module supports.
  int sub_x = 1;
  int sub_y = 1;
  int num_total_refs = 1;
  // Saved reference slots: ref_frame_idx[i] for i = 0..num_total_refs-1.
  const int* ref_frame_idx = nullptr;
  const SlotStore* slots = nullptr;
};

// §5.18.7.12 ccso_params().  `params` is both input (fresh values and reuse
// decisions) and output (frame_flag forced off when CCSO is unavailable).
// Returns the number of bits written.  On error returns 0 and fills `error`.
std::size_t write_ccso_params(BitWriter& writer, FrameParams& params,
                              const HeaderContext& context,
                              std::string* error);

// Independent §5.18.7.12 reader.  Reuse planes are loaded from
// context.slots (load_ccso_params); context.slots is therefore required as
// soon as a reuse bit is set.  On error returns false.
bool read_ccso_params(BitReader& reader, const HeaderContext& context,
                      FrameParams* params, std::string* error);

// The frame-level state a decoder derives from the header + saved slots.
struct EffectivePlane {
  PlaneParams params;
  std::vector<std::uint8_t> blks;  // inherited (sb_reuse) or to be coded
  int blk_cols = 0;
  int blk_rows = 0;
  bool sb_reuse_ccso = false;  // §5.20.10.2 suppresses ccso_blk S()
  int ref_slot = -1;           // saved slot used for reuse, else -1
};

struct EffectiveFrame {
  bool frame_flag = false;
  int ccso_luma_size_log2 = 6;
  EffectivePlane plane[3];
};

// Applies load_ccso_params() and the §6.17.7.8 conformance requirements.
// `syntax` holds the bits parsed/written; on return `out` holds the parameter
// values the decoder will filter with.  For sb_reuse_ccso the saved CcsoBlks
// are copied into `out`; otherwise the grid is zeroed and tile decoding fills
// it.  This is the function to call after read_ccso_params().
bool resolve_ccso_params(const HeaderContext& context,
                         const FrameParams& syntax, const SlotStore& slots,
                         EffectiveFrame* out, std::string* error);

// Encoder-side reuse decision.  `desired` is the plane's fresh parameters and
// `desired_blks` its freshly chosen CcsoBlks (may be empty for a plane that
// does not code block flags).  Returns in `out` whether the parameters and/or
// the block flags match a saved reference slot, plus the selected
// ccso_ref_idx.  This is the `use_ccso` decision; the spec's per-block `S()`
// symbol is handled by TileState::write_block().
struct ReuseChoice {
  bool reuse_ccso = false;
  bool sb_reuse_ccso = false;
  int ccso_ref_idx = 0;
  int ref_slot = -1;
  // Estimated header + tile bits avoided (informational).
  int bits_saved = 0;
};

bool use_ccso(const PlaneParams& desired,
              const std::vector<std::uint8_t>& desired_blks, int blk_cols,
              int blk_rows, int plane, const HeaderContext& context,
              const SlotStore& slots, ReuseChoice* out, std::string* error);

// ---------------------------------------------------------------------------
// §5.20.10.2 read_ccso() — per-superblock `ccso_blk S()` hook
// ---------------------------------------------------------------------------

// Entropy-coded CCSO block-enable state for one tile.  `params` must be the
// effective frame parameters (after load_ccso_params), and `blks` is
// pre-filled from the saved slot when sb_reuse_ccso[plane] is set.
class TileState {
 public:
  bool init(const EffectiveFrame& frame, int mi_cols, int mi_rows, int sub_x,
            int sub_y, std::string* error);

  void set_tile_origin(int mi_col, int mi_row) {
    tile_mi_col_start_ = mi_col;
    tile_mi_row_start_ = mi_row;
  }

  // Encoder decisions.  `data` is row-major blk_cols x blk_rows for `plane`.
  bool set_block_flags(int plane, const std::uint8_t* data, int cols,
                       int rows, std::string* error);

  // Encodes/decodes the ccso_blk S() symbols that occur at this coding block.
  // Calls at non-unit origins are no-ops (mirrors read_ccso()).
  bool write_block(av2::Encoder& encoder, int mi_row, int mi_col,
                   std::string* error);
  bool read_block(av2::Decoder& decoder, int mi_row, int mi_col,
                  int blk[3], std::string* error);

  const EffectiveFrame& frame() const { return frame_; }
  const std::vector<std::uint8_t>& blks(int plane) const {
    return frame_.plane[plane].blks;
  }
  int blk_cols(int plane) const { return frame_.plane[plane].blk_cols; }
  int blk_rows(int plane) const { return frame_.plane[plane].blk_rows; }
  int ccso_luma_size_log2() const { return frame_.ccso_luma_size_log2; }
  int mi_cols() const { return mi_cols_; }
  int mi_rows() const { return mi_rows_; }

  // True when a plane codes ccso_blk symbols in this tile.
  bool codes_block_flags(int plane) const;

 private:
  int block_origin(int mi_row, int mi_col, int* shift_row, int* shift_col,
                   int* blk_h4, int* blk_w4) const;
  int block_context(int plane, int mi_row, int mi_col, int shift_row,
                    int shift_col, int blk_w4) const;

  EffectiveFrame frame_{};
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int sub_x_ = 1;
  int sub_y_ = 1;
  int tile_mi_col_start_ = 0;
  int tile_mi_row_start_ = 0;
  std::uint16_t cdf_[3][4][3] = {};
  std::vector<std::uint8_t> desired_[3];
};

// Fills a restoration::CcsoFrameParams that aliases TileState's grids.  The
// TileState must outlive the returned struct.  `offsets` is filled from the
// plane's offset_idx values, so the filter core sees the exact §7.19 table.
bool make_restoration_params(const TileState& state,
                             restoration::CcsoFrameParams* out,
                             std::string* error);

// Small helper shared by call sites/tests.
int ceil_log2_int(int value);

}  // namespace av2enc::ccso_reuse

#endif  // AV2ENC_ENCODER_FILTERS_CCSO_REUSE_CCSO_REUSE_H_
