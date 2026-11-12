// AV2 v1.0.0 GDF syntax: sequence flags, frame params and the per-block
// `use_gdf` symbol.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §5.4.10 (enable_gdf /
// gdf_unit_matches_sb_size), §5.18.7.9 (gdf_params), §5.20.10.3 (read_gdf).
//
// This module owns *syntax only*.  It decides nothing and touches no pixels.
// The filter itself is the recovered reference in
// src/encoder/filters/spec_tables/gdf.cpp; the encoder decision that calls it
// is gdf_decision.{h,cpp}, and the dav2d gate is gdf_stream_gate.cpp.
//
// Call sites (an integration pass must add these; nothing outside this
// directory is edited by this component):
//   * sequence header, inside §5.4.10 sequence_filter_config():
//       filters_stream.cpp:317-319 currently rejects enable_gdf.  Replace the
//       single `writer.put_bit(false)` with WriteSequenceGdfFlags() and the
//       reader mirror with ReadSequenceGdfFlags().
//   * frame header, between deblocking_filter_params() and cdef_params():
//       m1_headers.cpp write_frame_header_syntax() and the M2 header writers
//       have no gdf_params() row.  Insert WriteFrameGdfParams() there.
//   * tile syntax, at the §5.20.10.3 point in read_gdf() (after the segment
//       id and before read_cdef/read_ccso): call GdfBlockCoder::WriteAt() /
//       ReadAt() once per coding block.  The coder no-ops unless the block is
//       the origin of a signalled GDF unit.
#ifndef AV2ENC_ENCODER_FILTERS_GDF_STREAM_GDF_SYNTAX_H_
#define AV2ENC_ENCODER_FILTERS_GDF_STREAM_GDF_SYNTAX_H_

#include "bitstream/bitwriter.h"
#include "encoder/m1/filters_stream/bit_io.h"
#include "entropy/av2/av2_entropy.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::filters::gdf {

// §3 constants.
inline constexpr int kGdfMinSize = 128;          // GDF_MIN_SIZE
inline constexpr int kGdfNumPicQcIdx = 4;        // gdf_pic_qc_idx is f(2)
inline constexpr int kGdfNumPicScaleIdx = 4;     // gdf_pic_scale_idx is f(2)

// §9.3 Default_Use_Gdf_Cdf: 2+1 entries, no context dimension.
inline constexpr std::uint16_t kDefaultUseGdfCdf[3] = {14593, 32, 0};

// ---------------------------------------------------------------------------
// §5.4.10 sequence_filter_config(): GDF rows
// ---------------------------------------------------------------------------
//
// syntax:
//   f(1) enable_gdf;
//   if ( enable_gdf && get_seq_sb_size() == BLOCK_64X64 )
//     f(1) gdf_unit_matches_sb_size;
//   else
//     gdf_unit_matches_sb_size = 0
//
// `sb_size` is 64 or 128 (get_seq_sb_size() returns BLOCK_64X64 for 64).
struct SequenceGdfFlags {
  bool enable_gdf = false;
  bool gdf_unit_matches_sb_size = false;  // inferred 0 when not coded
};

std::size_t WriteSequenceGdfFlags(BitWriter& writer,
                                  const SequenceGdfFlags& flags, int sb_size,
                                  std::string* error);

bool ReadSequenceGdfFlags(m1::filters_stream::BitReader& reader, int sb_size,
                          SequenceGdfFlags* flags, std::string* error);

// ---------------------------------------------------------------------------
// §5.18.7.9 gdf_params()
// ---------------------------------------------------------------------------

// Everything gdf_params() reads besides the frame bits.  `mi_col_starts` and
// `mi_row_starts` are the tile starts including 0; they feed the
// `a & 16` special case for BLOCK_64X64 when gdf_unit_matches_sb_size is 0.
struct FrameGdfContext {
  bool enable_gdf = false;
  bool gdf_unit_matches_sb_size = false;
  bool coded_lossless = false;
  bool single_picture_header_flag = false;
  int sb_size = 64;  // Block_Width[SbSize]: 64 or 128
  int width = 0;     // visible luma width
  int height = 0;    // visible luma height
  int mi_cols = 0;   // padded MiCols
  int mi_rows = 0;
  bool disable_loopfilters_across_tiles = false;
  int tile_cols = 1;
  int tile_rows = 1;
  std::vector<int> mi_col_starts{0};
  std::vector<int> mi_row_starts{0};
};

// Decoded / to-be-coded frame state.  `frame_enable` is inferred to 1 for a
// single-picture header; for a regular frame the writer codes it.
struct FrameGdfParams {
  bool frame_enable = false;
  bool per_block = false;
  int pic_qc_idx = 0;     // 0..3
  int pic_scale_idx = 0;  // 0..3
  // Derived from gdf_params().
  int blk_size = 0;       // GdfBlkSize in luma samples, 0 when disabled
  int cols = 0;           // GdfBlks width
  int rows = 0;           // GdfBlks height
  int pix_scale = 1;      // GdfPixScale = 1 + pic_scale_idx
  bool per_block_present = false;  // was the gdf_per_block bit coded?
};

// §5.18.7.9 derivation shared by writer, reader and filter.
struct GdfUnitLayout {
  int blk_size = 0;
  int cols = 0;
  int rows = 0;
  bool per_block_present = false;
  int mi_cols = 0;
  int mi_rows = 0;
  int width = 0;
  int height = 0;
};

GdfUnitLayout DeriveGdfUnitLayout(const FrameGdfContext& context);

// Writes exactly the gdf_params() row (including the inferred return cases).
// `params` is input for the regular-frame gdf_frame_enable bit and for the
// signalled per_block/qc/scale values; the derived fields are filled in.
std::size_t WriteFrameGdfParams(BitWriter& writer,
                                const FrameGdfContext& context,
                                FrameGdfParams* params, std::string* error);

// Independent mirror.  Re-derives GdfBlkSize and validates the field ranges.
bool ReadFrameGdfParams(m1::filters_stream::BitReader& reader,
                        const FrameGdfContext& context, FrameGdfParams* params,
                        std::string* error);

// ---------------------------------------------------------------------------
// §5.20.10.3 read_gdf(): per-block `use_gdf` S()
// ---------------------------------------------------------------------------
//
//   if ( !gdf_frame_enable || !gdf_per_block ) return
//   sbSize4 = Num_4x4_Blocks_Wide[ SbSize ]
//   if ( MiRow % sbSize4 != 0 || MiCol % sbSize4 != 0 ) return
//   sbRow = MiRow / sbSize4; sbCol = MiCol / sbSize4
//   sbPerGdf = GdfBlkSize / Block_Width[ SbSize ]
//   if ( sbCol % sbPerGdf != 0 || sbRow % sbPerGdf != 0 ) return
//   S() use_gdf
//   GdfBlks[ sbRow / sbPerGdf ][ sbCol / sbPerGdf ] = use_gdf
//
// The symbol uses TileUseGdfCdf, which has a single entry and therefore no
// neighbour context; its state is the per-tile CDF copy.
struct GdfSymbolPosition {
  int row = 0;
  int col = 0;
};

// Returns true when `read_gdf()` codes a symbol at this coding block.  `mi_row`
// and `mi_col` are the values read_gdf() sees: tile-relative for a multi-tile
// frame.  GdfBlks is frame-global in the specification, so multi-tile streams
// are not modelled here (see README).
bool GdfSymbolAt(const GdfUnitLayout& layout, const FrameGdfParams& params,
                 int sb_size, int mi_row, int mi_col,
                 GdfSymbolPosition* position);

// Per-tile `use_gdf` coder.  Holds the TileUseGdfCdf copy so the writer and the
// reader adapt identically; call WriteAt()/ReadAt() in coding order.
class GdfBlockCoder {
 public:
  GdfBlockCoder() = default;

  // `disable_cdf_update` must match the frame header value.  The entropy
  // coder owns the flag, so this only controls the local CDF copy consistency
  // assertions in the tests.
  void Reset();

  // Codes a symbol only at a signalled GDF-unit origin.  `use_gdf` is input
  // for the writer and output for the reader.  `coded` reports whether the
  // block was a GDF origin (so the caller can assert the grid was consumed).
  bool WriteAt(av2::Encoder& encoder, const GdfUnitLayout& layout,
               const FrameGdfParams& params, int sb_size, int mi_row,
               int mi_col, bool use_gdf, bool* coded);
  bool ReadAt(av2::Decoder& decoder, const GdfUnitLayout& layout,
              const FrameGdfParams& params, int sb_size, int mi_row,
              int mi_col, bool* use_gdf, bool* coded);

  // Exposed for golden/independence tests.
  const std::uint16_t* cdf() const { return cdf_; }

 private:
  std::uint16_t cdf_[3] = {kDefaultUseGdfCdf[0], kDefaultUseGdfCdf[1],
                           kDefaultUseGdfCdf[2]};
};

// ---------------------------------------------------------------------------
// Small helpers used by the encoder side and the tests
// ---------------------------------------------------------------------------

int GdfUnitCols(int mi_cols, int blk_size);
int GdfUnitRows(int mi_rows, int blk_size);

// §5.18.7.9 GdfBlkSize derivation, independent of the tile-alignment OR.
int DeriveGdfBlkSize(int sb_size, bool unit_matches_sb_size,
                     const std::vector<int>& mi_col_starts,
                     const std::vector<int>& mi_row_starts);

}  // namespace av2enc::filters::gdf

#endif  // AV2ENC_ENCODER_FILTERS_GDF_STREAM_GDF_SYNTAX_H_
