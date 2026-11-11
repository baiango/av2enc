// Minimal 128x128-SB stream writer.  See sb128_stream.h for scope.

#include "sb128_stream.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bitstream/m0_headers.h"
#include "encoder/m0_cdf_tables.h"
#include "encoder/m1/m1_cdf_tables.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::sb128 {
namespace {

using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (static_cast<std::uint64_t>(value) >=
         ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

Geometry geometry_for(const StreamConfig& config) {
  Geometry geometry;
  geometry.width = static_cast<int>(config.width);
  geometry.height = static_cast<int>(config.height);
  geometry.sb_size = config.use_128x128_superblock ? 128 : 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  geometry.max_pb_aspect_ratio = 8;
  return geometry;
}

void write_tile_info(BitWriter& writer, const StreamConfig& config) {
  const Geometry geometry = geometry_for(config);
  const int sb_mi = sb_size_mi(geometry);
  const int cols = (mi_cols(geometry) + sb_mi - 1) / sb_mi;
  const int rows = (mi_rows(geometry) + sb_mi - 1) / sb_mi;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (cols > 1) {
    writer.put_bit(false);  // increment_tile_cols_log2 = 0
  }
  if (rows > 1) {
    writer.put_bit(false);  // increment_tile_rows_log2 = 0
  }
}

std::vector<std::uint8_t> assemble_picture(
    const StreamConfig& config, std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header(writer, config);
  writer.align_to_byte();

  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const std::vector<std::uint8_t> clk = av2enc::m0::make_obu(
      ObuHeader{ObuType::kClosedLoopKey}, payload);
  const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);

  std::vector<std::uint8_t> out;
  out.reserve(td.size() + seq.size() + clk.size());
  out.insert(out.end(), td.begin(), td.end());
  out.insert(out.end(), seq.begin(), seq.end());
  out.insert(out.end(), clk.begin(), clk.end());
  return out;
}

// ---------------------------------------------------------------------------
// Tile CDFs and leaf coding
// ---------------------------------------------------------------------------

struct TileCdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t do_square_split[2][8][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t is_cfl[3][3];
  std::uint16_t uv_mode[2][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(TileCdfs& cdfs) {
  copy_array(cdfs.do_split, av2enc::m0::kDoSplitCdf);
  copy_array(cdfs.do_square_split, av2enc::m0::kDoSquareSplitCdf);
  copy_array(cdfs.rect_type, av2enc::m0::kRectTypeCdf);
  copy_array(cdfs.y_mode_set, av2enc::m1::kYModeSetCdf);
  copy_array(cdfs.y_mode_index, av2enc::m1::kYModeIndexCdf);
  copy_array(cdfs.is_cfl, av2enc::m1::kIsCflCdf);
  copy_array(cdfs.uv_mode, av2enc::m1::kUvModeCflNotAllowedCdf);
  copy_array(cdfs.txb_skip, av2enc::m0::kTxbSkipCdf);
  copy_array(cdfs.v_txb_skip, av2enc::m0::kVTxbSkipCdf);
}

int tx_size_for_square(int pixels) {
  switch (pixels) {
    case 4:
      return 0;  // TX_4X4
    case 8:
      return 1;  // TX_8X8
    case 16:
      return 2;  // TX_16X16
    case 32:
      return 3;  // TX_32X32
    case 64:
      return 4;  // TX_64X64
    default:
      return -1;
  }
}

enum class SymbolKind : std::uint8_t {
  kDoSplit = 0,
  kDoSquareSplit = 1,
  kRectType = 2,
};

struct PlanSymbol {
  SymbolKind kind = SymbolKind::kDoSplit;
  int value = 0;
  int context = 0;
};

bool plan_partition_syntax(const Node& node, const Contexts& contexts,
                           std::vector<PlanSymbol>& out, std::string& error) {
  out.clear();
  if (node.forced) return true;

  const bool allow_none = node.allowed.is_allowed(PartitionType::kNone);
  if (allow_none) {
    out.push_back(
        {SymbolKind::kDoSplit, node.partition == PartitionType::kNone ? 0 : 1,
         contexts.do_split});
    if (node.partition == PartitionType::kNone) return true;
  } else if (node.partition == PartitionType::kNone) {
    return fail(error, "partition NONE is not allowed but was selected");
  }

  const bool allow_split = node.allowed.is_allowed(PartitionType::kSplit);
  if (allow_split) {
    const int value = node.partition == PartitionType::kSplit ? 1 : 0;
    out.push_back(
        {SymbolKind::kDoSquareSplit, value, contexts.do_square_split});
    if (node.partition == PartitionType::kSplit) return true;
  } else if (node.partition == PartitionType::kSplit) {
    return fail(error, "PARTITION_SPLIT is not allowed at this node");
  }

  const RectTypeImplied implied = rect_type_implied_by_bsize(node.size);
  PartitionType rect = implied.partition;
  bool need_rect = !implied.implied;
  if (need_rect) {
    const bool allow_h = node.allowed.is_allowed(PartitionType::kHorz);
    const bool allow_v = node.allowed.is_allowed(PartitionType::kVert);
    if (!allow_h && !allow_v) {
      return fail(error, "no rectangular partition is allowed");
    }
    if (!allow_h) {
      rect = PartitionType::kVert;
      need_rect = false;
    } else if (!allow_v) {
      rect = PartitionType::kHorz;
      need_rect = false;
    }
  }
  if (need_rect) {
    if (node.partition != PartitionType::kHorz &&
        node.partition != PartitionType::kVert) {
      return fail(error, "non-rectangular partition reached rect_type syntax");
    }
    const int value = node.partition == PartitionType::kVert ? 1 : 0;
    out.push_back({SymbolKind::kRectType, value, contexts.rect_type});
    rect = node.partition;
  }

  if (node.partition == PartitionType::kHorz && rect != PartitionType::kHorz) {
    return fail(error, "HORZ partition contradicts the implied rect_type");
  }
  if (node.partition == PartitionType::kVert && rect != PartitionType::kVert) {
    return fail(error, "VERT partition contradicts the implied rect_type");
  }
  return true;
}

bool encode_symbol(av2::Encoder& encoder, std::uint16_t* cdf, int n,
                   int value, std::string& error) {
  if (encoder.encode_symbol(cdf, n, value) != av2::Status::kOk) {
    error = "entropy encode_symbol failed";
    return false;
  }
  return true;
}

bool encode_leaf(const Node& node, const Geometry& geometry, TileCdfs& cdfs,
                 av2::Encoder& encoder, std::string& error) {
  if (node.size != BlockSize::k64x64) {
    error = std::string("minimal 128-SB stream requires 64x64 leaves, got ") +
            block_size_name(node.size);
    return false;
  }
  const int luma_tx = tx_size_for_square(block_width(node.size));
  if (luma_tx < 0) return fail(error, "unsupported luma transform size");

  // DC_PRED: reordered index 0.  Y mode set symbol 0 (mode index < 13), index
  // 0 with joint context 0 (all neighbours also DC).
  if (!encode_symbol(encoder, cdfs.y_mode_set, 4, 0, error) ||
      !encode_symbol(encoder, cdfs.y_mode_index[0], 8, 0, error)) {
    return false;
  }
  if (node.has_chroma) {
    // is_cfl = 0 (CfL enabled), UV mode DC index 0 with context 0.
    if (!encode_symbol(encoder, cdfs.is_cfl[0], 2, 0, error) ||
        !encode_symbol(encoder, cdfs.uv_mode[0], 8, 0, error)) {
      return false;
    }
  }

  const int planes = node.has_chroma ? 3 : 1;
  int chroma_tx = -1;
  if (node.has_chroma) {
    const BlockSize chroma_size =
        chroma_plane_residual_size(geometry, node.chroma.size);
    if (!is_valid_block_size(chroma_size)) {
      return fail(error, "invalid chroma residual size");
    }
    chroma_tx = tx_size_for_square(block_width(chroma_size));
    if (chroma_tx < 0 || block_height(chroma_size) != block_width(chroma_size)) {
      return fail(error, "unsupported chroma transform size");
    }
  }

  for (int plane = 0; plane < planes; ++plane) {
    // TX_MODE_LARGEST, eob = 0.  Luma and chroma both use a full-block
    // transform, so the txb_skip context is 0 for luma, 6 for U and 0 for V
    // (EobU == 0 and bw*bh == w*h, §8.3.2).
    std::uint16_t* row = nullptr;
    if (plane == 0) {
      row = &cdfs.txb_skip[0][luma_tx][0][0];
    } else if (plane == 1) {
      row = &cdfs.txb_skip[0][chroma_tx][6][0];
    } else {
      row = &cdfs.v_txb_skip[0][0];
    }
    if (!encode_symbol(encoder, row, 2, 1, error)) return false;
  }
  return true;
}

bool emit_node(const Node& node, const Geometry& geometry, MiSizeGrid& grid,
               TileCdfs& cdfs, av2::Encoder& encoder, std::size_t& symbols,
               std::string& error) {
  if (!node.forced) {
    const Contexts contexts = partition_contexts(
        geometry, node.mi_row, node.mi_col, node.size, grid);
    std::vector<PlanSymbol> plan;
    if (!plan_partition_syntax(node, contexts, plan, error)) return false;
    for (const PlanSymbol& symbol : plan) {
      std::uint16_t* row = nullptr;
      switch (symbol.kind) {
        case SymbolKind::kDoSplit:
          row = &cdfs.do_split[0][symbol.context][0];
          break;
        case SymbolKind::kDoSquareSplit:
          row = &cdfs.do_square_split[0][symbol.context][0];
          break;
        case SymbolKind::kRectType:
          row = &cdfs.rect_type[0][symbol.context][0];
          break;
      }
      if (row == nullptr) return fail(error, "invalid partition context");
      if (!encode_symbol(encoder, row, 2, symbol.value, error)) return false;
      ++symbols;
    }
  }

  if (node.partition == PartitionType::kNone) {
    if (!encode_leaf(node, geometry, cdfs, encoder, error)) return false;
    grid.fill_block(node.mi_row, node.mi_col, node.size);
    return true;
  }
  for (const Node& child : node.children) {
    if (!emit_node(child, geometry, grid, cdfs, encoder, symbols, error)) {
      return false;
    }
  }
  return true;
}

bool build_tile(const FrameTree& frame, bool disable_cdf_update,
                std::vector<std::uint8_t>& tile, std::size_t& symbols,
                std::string& error) {
  TileCdfs cdfs{};
  init_cdfs(cdfs);

  constexpr std::size_t kTileCapacity = 1U << 21U;
  std::vector<std::uint8_t> buffer(kTileCapacity);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return fail(error, "entropy init_symbol failed");
  }
  encoder.disable_cdf_update = disable_cdf_update ? 1U : 0U;

  MiSizeGrid grid(frame.geometry);
  symbols = 0;
  for (const Node& root : frame.superblocks) {
    if (!emit_node(root, frame.geometry, grid, cdfs, encoder, symbols,
                   error)) {
      return false;
    }
  }

  std::size_t payload_bytes = 0;
  if (encoder.exit_symbol(&payload_bytes) != av2::Status::kOk) {
    return fail(error, "entropy exit_symbol failed");
  }
  tile.assign(buffer.begin(),
              buffer.begin() + static_cast<std::ptrdiff_t>(payload_bytes));
  return true;
}

}  // namespace

bool stream_geometry_supported(const Geometry& geometry, std::string& error) {
  if (!validate_geometry(geometry, error)) return false;
  if (geometry.sb_size != kSbSize) {
    return fail(error, "minimal stream requires a 128x128 superblock");
  }
  if (geometry.width % kSbSize != 0) {
    return fail(error,
                "minimal stream requires width to be a multiple of 128");
  }
  // Every leaf the reference policy creates must pass the 64x64 half-block
  // test (r + 8 < MiRows).  For a bottom remainder h this holds only for
  // h in [9,16] (the root is implied HORZ) or h in [25,32] (the root SPLITs
  // and its second 64x64 child still has more than 8 rows).  h in [1,8] and
  // [17,24] would force a sub-64x64 leaf.
  const int remainder = mi_rows(geometry) % kSbSizeMi;
  const bool height_ok = remainder == 0 || (remainder >= 9 && remainder <= 16) ||
                         remainder >= 25;
  if (!height_ok) {
    return fail(error,
                "minimal stream cannot produce 64x64 leaves at this height");
  }
  error.clear();
  return true;
}

bool make_sb128_stream(const StreamConfig& config, StreamResult& out,
                       std::string& error) {
  if (!config.use_128x128_superblock) {
    return fail(error, "use_128x128_superblock must be 1 for this stream");
  }
  if (config.base_q_idx == 0) {
    return fail(error, "base_q_idx 0 (lossless) is not supported");
  }
  const Geometry geometry = geometry_for(config);
  if (!stream_geometry_supported(geometry, error)) return false;

  FrameTree frame;
  if (!build_reference_frame(geometry, frame, error)) return false;

  std::vector<std::uint8_t> tile;
  std::size_t symbols = 0;
  if (!build_tile(frame, config.disable_cdf_update, tile, symbols, error)) {
    return false;
  }

  StreamResult result;
  result.frame = assemble_picture(config, tile);
  result.tile = std::move(tile);
  result.leaves = frame.leaves;
  result.symbols = symbols;
  out = std::move(result);
  return true;
}

std::size_t write_sequence_header(BitWriter& writer,
                                  const StreamConfig& config) {
  const std::size_t start = writer.bit_size();
  put_uvlc(writer, 0);                   // seq_header_id
  writer.put_bits(0U, 5U);               // seq_profile_idc
  writer.put_bit(true);                  // single_picture_header_flag
  writer.put_bits(config.seq_level_idx, 5U);
  put_uvlc(writer, 0);                   // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                   // bit_depth_idc = 1 => 8-bit

  const std::uint32_t width = config.width;
  const std::uint32_t height = config.height;
  unsigned width_bits = bit_width(width - 1U);
  unsigned height_bits = bit_width(height - 1U);
  if (width_bits == 0) width_bits = 1;
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(width - 1U, width_bits);
  writer.put_bits(height - 1U, height_bits);
  writer.put_bit(false);  // seq_cropping_window_present_flag

  // §5.4.3 sequence_partition_config().
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(config.use_128x128_superblock);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config().
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config().
  writer.put_bit(false);  // enable_dip
  writer.put_bit(false);  // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(config.enable_cfl_intra);
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp

  // §5.4.6 sequence_inter_config(): forced by single_picture_header_flag.
  writer.put_bit(false);   // enable_refmvbank
  writer.put_bit(true);    // disable_drl_reorder
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);   // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);   // enable_bawp

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(true);   // enable_chroma_dctonly
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present
  return writer.bit_size() - start;
}

std::size_t write_frame_header(BitWriter& writer,
                               const StreamConfig& config) {
  const std::size_t start = writer.bit_size();
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(config.disable_cdf_update);
  write_tile_info(writer, config);

  writer.put_bits(config.base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  // deblocking_filter_params(): all planes disabled keeps the header minimal.
  writer.put_bit(false);  // apply_deblocking_filter[0] (Y vertical)
  writer.put_bit(false);  // apply_deblocking_filter[1] (Y horizontal)

  writer.put_bit(false);    // tx_mode_select = 0 (TX_MODE_LARGEST)
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const StreamConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::m0::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return av2enc::m0::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

}  // namespace av2enc::sb128
