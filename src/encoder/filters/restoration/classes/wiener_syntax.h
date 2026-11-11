// Encoder-side AV2 loop-restoration syntax (§5.18.7.11, §5.20.10.4–.6).
//
// This module writes and reads the exact field order of
//
//   lr_params()            frame header   (§5.18.7.11)
//   read_lr_unit()         tile, per 4x4 block/unit (§5.20.10.5)
//   read_wienerns_filter() tile + frame (§5.20.10.6)
//
// It is deliberately independent of the M1 filters_stream adapters so the
// encoder-side search can be tested without the M1 tile pipeline.  The M1
// module can be switched to these writers (or kept) without changing the
// decision structures here; the integration notes in README.md list the call
// sites and the field-for-field correspondence.
//
// Raw descriptors (f(n), ns(n), L(1)) use BitWriter/WienerBitReader.
// Per-unit filter syntax (S() symbols) uses the normative §8.2 arithmetic
// coder from src/entropy/av2.
//
// Deliberately unsupported: PC-Wiener and GDF.  §9.8 does not print the
// Pc_Wiener_* or Gdf_* tables, so no encoder can produce those filters from
// the spec text alone.  Writing a PC-Wiener or GDF selection returns an error
// instead of emitting a stream that no local decoder can reproduce.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_SYNTAX_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_SYNTAX_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/filters/restoration/restoration_common.h"
#include "encoder/filters/restoration/classes/wiener_tables.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::restoration::classes {

// ---------------------------------------------------------------------------
// Bit-level mirror of BitWriter for the header descriptors.
// ---------------------------------------------------------------------------

class WienerBitReader {
 public:
  WienerBitReader(const std::uint8_t* data, std::size_t bit_size)
      : data_(data), bit_size_(bit_size) {}

  bool read_bit(bool* bit);
  bool read_bits(unsigned n, std::uint32_t* value);
  bool read_ns(unsigned n, std::uint32_t* value);

  std::size_t bit_pos() const { return bit_pos_; }
  std::size_t bits_left() const { return bit_size_ - bit_pos_; }

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t bit_size_ = 0;
  std::size_t bit_pos_ = 0;
};

bool wiener_put_ns(BitWriter& writer, unsigned n, std::uint32_t value,
                   std::string* error);

// §5.18.9.3–§5.18.9.6.
bool wiener_encode_signed_subexp_with_ref(BitWriter& writer, int low,
                                          int high, int r, int k,
                                          int target, std::string* error);
bool wiener_decode_signed_subexp_with_ref(WienerBitReader& reader, int low,
                                          int high, int r, int k,
                                          int* target, std::string* error);

// §5.20.10.6 decode_4part()/decode_signed_4part() (S()-coded).
bool wiener_encode_signed_4part(av2::Encoder& encoder,
                                std::uint16_t* base_cdf, int low, int k,
                                int r, int target, std::string* error);
bool wiener_decode_signed_4part(av2::Decoder& decoder,
                                std::uint16_t* base_cdf, int low, int k,
                                int r, int* target, std::string* error);

// ---------------------------------------------------------------------------
// Reference-frame frame filters (§5.20.10.6 search_frame_filters)
// ---------------------------------------------------------------------------

struct WienerRefFrameFilters {
  bool filters_on[3] = {false, false, false};
  int num_filter_classes = 1;  // RefNumFilterClasses[idx]
  std::int16_t coeffs[3][kWienerNsClasses][kWienerNsCoeffStride] = {};
};

struct WienerReferenceSet {
  int num_total_refs = 0;         // NumTotalRefs
  bool frame_is_intra = true;     // FrameIsIntra
  bool is_switch_frame = false;   // FrameType == SWITCH_FRAME
  // OrderHints[ref] and ref_frame_idx[ref] for ref = 0..num_total_refs-1.
  int order_hint[kMaxLrRefs] = {};
  int ref_frame_idx[kMaxLrRefs] = {};
  // Indexed by ref_frame_idx value (NUM_REF_FRAMES entries).
  WienerRefFrameFilters ref[kMaxLrRefs];
};

struct WienerRefFilterEntry {
  int ref_idx = 0;
  int ref_plane = 0;
  int ref_class = 0;
};

struct WienerSearchResult {
  int num_classes = 1;
  int num_ref_filters = 0;
  int match_idx = 0;
  int match_cls = 0;
  int match_plane = 0;
  // Ref filters addressable by match index m in
  // [num_classes, num_classes + num_ref_filters), in enumeration order.
  std::vector<WienerRefFilterEntry> ref_filters;
};

// ---------------------------------------------------------------------------
// §5.18.7.11 lr_params()
// ---------------------------------------------------------------------------

struct WienerLrHeaderConfig {
  bool coded_lossless = false;
  bool enable_restoration = false;
  bool frame_is_intra = true;
  bool is_switch_frame = false;
  int num_total_refs = 0;
  int num_planes = 3;
  int sb_size = 64;
  int sub_x = 1;
  int sub_y = 1;
  int base_q_idx = 0;
  // lr_tools_disable[chroma?1:0][restoration type]; index 1 is PC-Wiener.
  // This encoder requires lr_tools_disable[0][RESTORE_PC_WIENER] == true.
  bool lr_tools_disable[2][4] = {
      {false, true, false, false},
      {false, true, false, false},
  };
};

struct WienerFrameFilterSyntax {
  int match_index = 0;   // 0 = zero filter, m > 0 as in search_frame_filters()
  bool merged = false;   // f(1): reuse the matched filter
  int subset = 0;        // 0..3 luma, 0..2 chroma
  bool uv_sym = false;
  std::int16_t coeffs[kWienerNsCoeffStride] = {};
};

struct WienerLrHeaderParams {
  int frame_type[3] = {kRestoreNone, kRestoreNone, kRestoreNone};
  bool frame_filters_on[3] = {false, false, false};
  bool temporal_pred_flag[3] = {false, false, false};
  int rst_ref_pic_idx[3] = {0, 0, 0};
  int num_filter_classes = 1;
  int num_filter_classes_idx = 0;
  int luma_shift = 3;
  int chroma_shift = 3;

  // Final FrameLrWienerNs[plane][class][coeff].  On write, frame_filter[*]
  // carries the syntax decisions and these arrays must match what the reader
  // will reconstruct; on read they are the output.
  std::int16_t frame_coeffs[3][kWienerNsClasses][kWienerNsCoeffStride] = {};
  WienerFrameFilterSyntax frame_filter[3][kWienerNsClasses];
};

WienerSearchResult wiener_search_frame_filters(
    int plane, int target, const WienerLrHeaderParams& params,
    const WienerReferenceSet& refs, bool nopcw);

bool wiener_write_lr_params(BitWriter& writer,
                            const WienerLrHeaderConfig& config,
                            WienerLrHeaderParams* params,
                            const WienerReferenceSet& refs,
                            std::string* error);
bool wiener_read_lr_params(WienerBitReader& reader,
                           const WienerLrHeaderConfig& config,
                           WienerLrHeaderParams* params,
                           const WienerReferenceSet& refs,
                           std::string* error);

// ---------------------------------------------------------------------------
// §5.20.10.5 read_lr_unit() / §5.20.10.6 read_wienerns_filter()
// ---------------------------------------------------------------------------

struct WienerUnitSyntax {
  int type = kRestoreNone;
  // Per-unit filter syntax (used only when the plane's frame filters are off).
  bool merged = false;
  // -1 selects the newest bank entry automatically.
  int use_bank_k = -1;
  int subset = 0;
  bool uv_sym = false;
  std::int16_t coeffs[kWienerNsCoeffStride] = {};
};

struct WienerTileState {
  WienerLrHeaderParams params;
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;

  int unit_size[3] = {64, 64, 64};
  int unit_rows[3] = {0, 0, 0};
  int unit_cols[3] = {0, 0, 0};

  std::vector<std::uint8_t> unit_type[3];
  std::vector<std::int16_t> unit_coeffs[3];

  // §9.3 CDF copies, one per tile.
  std::uint16_t use_wiener_ns[3] = {};
  std::uint16_t use_pc_wiener[3] = {};
  std::uint16_t flex_restoration_type[3][3][3] = {};
  std::uint16_t wiener_ns_length[2][3] = {};
  std::uint16_t wiener_ns_uv_sym[3] = {};
  std::uint16_t wiener_ns_base[5] = {};

  // RefLrWienerNs[plane][class][bank][coeff] and bank bookkeeping.
  std::int16_t ref_bank[3][kWienerNsClasses][kLrBankSize]
                       [kWienerNsCoeffStride] = {};
  int bank_ptr[3][kWienerNsClasses] = {};
  int bank_size[3][kWienerNsClasses] = {};
};

bool wiener_tile_init(WienerTileState* state,
                      const WienerLrHeaderParams& params, int mi_cols,
                      int mi_rows, int sub_x, int sub_y,
                      std::string* error);

bool wiener_write_unit(av2::Encoder& encoder, WienerTileState& state,
                       int plane, int unit_row, int unit_col,
                       const WienerUnitSyntax& unit, std::string* error);
bool wiener_read_unit(av2::Decoder& decoder, WienerTileState& state,
                      int plane, int unit_row, int unit_col,
                      WienerUnitSyntax* unit, std::string* error);

// Writes/reads a whole tile in the exact §5.20.10.4 read_lr() order:
// superblocks in raster order, planes 0..2, then the units covered by that
// superblock.  `plans[plane]` is indexed by unit_row * unit_cols + unit_col.
bool wiener_write_tile(av2::Encoder& encoder, WienerTileState& state,
                       const std::vector<WienerUnitSyntax> plans[3],
                       int sb_size, int tile_mi_row_start,
                       int tile_mi_col_start, std::string* error);
bool wiener_read_tile(av2::Decoder& decoder, WienerTileState& state,
                      std::vector<WienerUnitSyntax> plans[3], int sb_size,
                      int tile_mi_row_start, int tile_mi_col_start,
                      std::string* error);

// §5.20.10.4 read_lr(): unit range covered by one coding block.  The tile
// origin and `w4`/`h4` (Num_4x4_Blocks_Wide/High) are the same inputs the
// spec passes from the partition walk.
struct WienerUnitRange {
  int row_begin = 0;
  int row_end = 0;
  int col_begin = 0;
  int col_end = 0;
};

bool wiener_unit_range(const WienerTileState& state, int plane, int mi_row,
                       int mi_col, int w4, int h4, int tile_mi_row_start,
                       int tile_mi_col_start, WienerUnitRange* range,
                       std::string* error);

}  // namespace av2enc::restoration::classes

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_SYNTAX_H_
