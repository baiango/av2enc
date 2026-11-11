// Unit tests for the chroma-format component.
//
// Covers Table 6.2 mapping, ChromaMiSize / Subsampled_Size, plane geometry,
// Monochrome tool legality, the §5.4.1 sequence-header writer/reader (including
// the Monochrome-conditional field omissions), the residual shape layer
// (Adjusted_Tx_Size, Transform_Shift, duplication), and a smoke encode for all
// four formats.  dav2d validation and reconstruct-vs-decoder comparison live in
// gate.sh.
#include "encoder/format/chroma/chroma_format.h"
#include "encoder/format/chroma/chroma_headers.h"
#include "encoder/format/chroma/chroma_residual.h"
#include "encoder/format/chroma/chroma_still.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "            \
                << #condition << "\n";                                       \
    }                                                                        \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!((a) == (b))) {                                                     \
      ++g_failures;                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #a      \
                << " == " << #b << "\n";                                     \
    }                                                                        \
  } while (0)

using av2enc::chroma::BlockSize;
using av2enc::chroma::ChromaFormat;
using av2enc::chroma::ChromaFormatInfo;
using av2enc::chroma::FrameShape;
using av2enc::chroma::TxShape;

void test_format_table() {
  struct Expected {
    ChromaFormat format;
    int idc;
    int sx;
    int sy;
    bool monochrome;
    int planes;
    int profile;
  };
  const Expected expected[] = {
      {ChromaFormat::k420, 0, 1, 1, false, 3, 0},
      {ChromaFormat::k400, 1, 1, 1, true, 1, 0},
      {ChromaFormat::k444, 2, 0, 0, false, 3, 4},
      {ChromaFormat::k422, 3, 1, 0, false, 3, 3},
  };
  for (const Expected& e : expected) {
    const ChromaFormatInfo info =
        av2enc::chroma::chroma_format_info(e.format);
    CHECK_EQ(info.idc, e.idc);
    CHECK_EQ(info.subsampling_x, e.sx);
    CHECK_EQ(info.subsampling_y, e.sy);
    CHECK_EQ(info.monochrome, e.monochrome);
    CHECK_EQ(info.num_planes, e.planes);
    CHECK_EQ(info.profile_idc, e.profile);
    CHECK_EQ(info.max_quantizer, 255);
    CHECK_EQ(info.base_q_idx_bits, 8);
    const auto roundtrip = av2enc::chroma::chroma_format_from_idc(e.idc);
    CHECK(roundtrip.has_value());
    CHECK(*roundtrip == e.format);
  }
  CHECK(!av2enc::chroma::chroma_format_from_idc(4).has_value());
  CHECK(av2enc::chroma::chroma_format_from_name("4:2:2") ==
        ChromaFormat::k422);
  CHECK(av2enc::chroma::chroma_format_from_name("mono") ==
        ChromaFormat::k400);
}

void test_chroma_mi_size() {
  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k64x64,
                                       ChromaFormat::k420) ==
        BlockSize::k32x32);
  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k64x64,
                                       ChromaFormat::k422) ==
        BlockSize::k32x64);
  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k64x64,
                                       ChromaFormat::k444) ==
        BlockSize::k64x64);
  CHECK(!av2enc::chroma::chroma_mi_size(BlockSize::k64x64,
                                        ChromaFormat::k400)
             .has_value());

  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k32x32,
                                       ChromaFormat::k422) ==
        BlockSize::k16x32);
  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k32x16,
                                       ChromaFormat::k422) ==
        BlockSize::k16x16);
  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k32x8,
                                       ChromaFormat::k422) ==
        BlockSize::k16x8);
  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k16x32,
                                       ChromaFormat::k422) ==
        BlockSize::k8x32);
  // Subsampled_Size has real BLOCK_INVALID entries: BLOCK_4X8 with
  // SubsamplingX=1, SubsamplingY=0 (4:2:2) has no legal residual.
  CHECK(!av2enc::chroma::chroma_mi_size(BlockSize::k4x8,
                                        ChromaFormat::k422)
             .has_value());
  // The same 4x8 block is legal in 4:2:0 (it maps to BLOCK_4X4).
  CHECK(av2enc::chroma::chroma_mi_size(BlockSize::k4x8,
                                       ChromaFormat::k420)
            .has_value());
  // is_chroma_offset_for_subsize triggers when a subsampled dimension would
  // shrink below 4 samples.
  CHECK_EQ(av2enc::chroma::is_chroma_offset_for_subsize(
               BlockSize::k4x8, ChromaFormat::k420),
           true);
  CHECK_EQ(av2enc::chroma::is_chroma_offset_for_subsize(
               BlockSize::k8x4, ChromaFormat::k420),
           true);
  CHECK_EQ(av2enc::chroma::is_chroma_offset_for_subsize(
               BlockSize::k4x8, ChromaFormat::k444),
           false);
}

void test_geometry() {
  const FrameShape s64_422 =
      av2enc::chroma::make_frame_shape(64, 64, ChromaFormat::k422);
  CHECK_EQ(s64_422.mi_cols, 16);
  CHECK_EQ(s64_422.mi_rows, 16);
  CHECK_EQ(s64_422.coded_width, 64);
  CHECK_EQ(s64_422.coded_height, 64);
  CHECK_EQ(s64_422.padded_width, 64);
  CHECK_EQ(s64_422.padded_height, 64);
  CHECK_EQ(s64_422.planes[1].visible_width, 32);
  CHECK_EQ(s64_422.planes[1].visible_height, 64);
  CHECK_EQ(s64_422.planes[1].coded_width, 32);
  CHECK_EQ(s64_422.planes[1].coded_height, 64);
  CHECK_EQ(s64_422.planes[1].padded_width, 32);
  CHECK_EQ(s64_422.planes[1].padded_height, 64);

  const FrameShape s360_420 =
      av2enc::chroma::make_frame_shape(640, 360, ChromaFormat::k420);
  CHECK_EQ(s360_420.mi_cols, 160);
  CHECK_EQ(s360_420.mi_rows, 90);
  CHECK_EQ(s360_420.planes[1].visible_width, 320);
  CHECK_EQ(s360_420.planes[1].visible_height, 180);
  CHECK_EQ(s360_420.planes[1].coded_height, 180);
  CHECK_EQ(s360_420.planes[1].padded_height, 192);

  const FrameShape s360_444 =
      av2enc::chroma::make_frame_shape(640, 360, ChromaFormat::k444);
  CHECK_EQ(s360_444.planes[1].coded_width, 640);
  CHECK_EQ(s360_444.planes[1].coded_height, 360);
  CHECK_EQ(s360_444.planes[1].padded_height, 384);

  const FrameShape s400 =
      av2enc::chroma::make_frame_shape(640, 360, ChromaFormat::k400);
  CHECK(s400.monochrome);
  CHECK(!s400.planes[1].present);
  CHECK(!s400.planes[2].present);
  CHECK_EQ(s400.num_planes, 1);

  std::string error;
  CHECK(av2enc::chroma::supports_64x64_none(s64_422, error));
  CHECK(av2enc::chroma::supports_64x64_none(s360_420, error));
  CHECK(av2enc::chroma::supports_64x64_none(s360_444, error));
  const FrameShape s322 =
      av2enc::chroma::make_frame_shape(640, 322, ChromaFormat::k422);
  CHECK(!av2enc::chroma::supports_64x64_none(s322, error));
  CHECK(!error.empty());
}

void test_tool_legality() {
  const auto mono = av2enc::chroma::tool_legality(ChromaFormat::k400);
  CHECK(mono.monochrome);
  CHECK(!mono.enable_sdp);
  CHECK(!mono.sdp_syntax_present);
  CHECK(!mono.enable_cfl_intra);
  CHECK(!mono.cfl_ds_filter_index_present);
  CHECK(!mono.enable_mhccp);
  CHECK(!mono.enable_chroma_dctonly);
  CHECK(!mono.chroma_dctonly_syntax_present);
  CHECK(!mono.enable_cctx);
  CHECK(!mono.cctx_syntax_present);
  CHECK(!mono.separate_uv_delta_q);
  CHECK(!mono.separate_uv_delta_q_syntax_present);
  CHECK(!mono.uv_delta_q_syntax_present);
  CHECK(!mono.ccso_chroma_planes_available);
  CHECK(!mono.cdef_uv_strengths_present);
  CHECK(!mono.restoration_uv_present);
  CHECK(!mono.deblock_chroma_planes_present);
  CHECK(!mono.uv_mode_syntax_present);
  CHECK(!mono.chroma_residual_present);

  for (ChromaFormat format :
       {ChromaFormat::k420, ChromaFormat::k422, ChromaFormat::k444}) {
    const auto tools = av2enc::chroma::tool_legality(format);
    CHECK(!tools.monochrome);
    CHECK(tools.sdp_syntax_present);
    CHECK(tools.cfl_ds_filter_index_present);
    CHECK(tools.chroma_dctonly_syntax_present);
    CHECK(tools.cctx_syntax_present);
    CHECK(tools.separate_uv_delta_q_syntax_present);
    CHECK(tools.uv_delta_q_syntax_present);
    CHECK(tools.ccso_chroma_planes_available);
    CHECK(tools.uv_mode_syntax_present);
    CHECK(tools.chroma_residual_present);
    CHECK(!tools.enable_cfl_intra);
    CHECK(!tools.enable_mhccp);
    CHECK(!tools.enable_sdp);
  }

  for (ChromaFormat format :
       {ChromaFormat::k420, ChromaFormat::k422, ChromaFormat::k444,
        ChromaFormat::k400}) {
    const auto report = av2enc::chroma::tool_legality_report(format);
    CHECK(!report.empty());
    bool has_idc = false;
    bool has_sdp = false;
    for (const std::string& line : report) {
      has_idc |= line.find("chroma_format_idc=") != std::string::npos;
      has_sdp |= line.find("enable_sdp=0") != std::string::npos;
    }
    CHECK(has_idc);
    CHECK(has_sdp);
    const auto checklist = av2enc::chroma::chroma_checklist(format);
    CHECK(checklist.size() >= 10);
    bool has_monochrome_rule = false;
    bool has_chroma_mi = false;
    for (const auto& item : checklist) {
      has_monochrome_rule |=
          item.id == "Monochrome conditionals" &&
          item.spec.find("§5.4") != std::string::npos;
      has_chroma_mi |= item.id == "ChromaMiSize";
    }
    CHECK(has_monochrome_rule);
    CHECK(has_chroma_mi);
  }
}

void test_headers() {
  const ChromaFormat formats[] = {ChromaFormat::k420, ChromaFormat::k422,
                                  ChromaFormat::k444, ChromaFormat::k400};
  std::size_t bit_sizes[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    av2enc::chroma::ChromaSeqConfig config;
    config.max_frame_width = 640;
    config.max_frame_height = 360;
    config.chroma = formats[i];
    config.seq_level_idx = 7;
    std::string error;
    CHECK(av2enc::chroma::validate_chroma_seq_config(config, error));

    av2enc::BitWriter writer;
    const std::size_t bits =
        av2enc::chroma::write_sequence_header_syntax(writer, config);
    bit_sizes[i] = bits;
    const std::vector<std::uint8_t> syntax = writer.take_bytes();

    av2enc::chroma::SequenceHeaderInfo info;
    CHECK(av2enc::chroma::read_sequence_header_syntax(syntax, info, error));
    CHECK_EQ(info.chroma_format_idc, av2enc::chroma::chroma_format_idc(formats[i]));
    CHECK_EQ(info.bit_depth_idc, 1);
    CHECK_EQ(info.bit_depth, 8);
    CHECK_EQ(info.monochrome, formats[i] == ChromaFormat::k400);
    CHECK_EQ(info.subsampling_x, av2enc::chroma::subsampling_x(formats[i]));
    CHECK_EQ(info.subsampling_y, av2enc::chroma::subsampling_y(formats[i]));
    CHECK_EQ(info.num_planes, av2enc::chroma::num_planes(formats[i]));
    CHECK_EQ(info.max_frame_width, 640u);
    CHECK_EQ(info.max_frame_height, 360u);
    CHECK(!info.cropping_window_present);
    CHECK(!info.enable_sdp);
    CHECK(!info.enable_cfl_intra);
    CHECK(!info.enable_mhccp);
    CHECK(!info.enable_chroma_dctonly);
    CHECK(!info.enable_cctx);
    CHECK(!info.separate_uv_delta_q);
    CHECK(!info.enable_cdef);
    CHECK(!info.enable_restoration);
    CHECK(!info.enable_ccso);
    CHECK_EQ(info.seq_profile_idc,
             av2enc::chroma::chroma_format_info(formats[i]).profile_idc);

    const std::vector<std::uint8_t> obu =
        av2enc::chroma::make_sequence_header_obu(config);
    av2enc::chroma::SequenceHeaderInfo obu_info;
    CHECK(av2enc::chroma::parse_sequence_header_obu(obu, obu_info, error));
    CHECK_EQ(obu_info.chroma_format_idc, info.chroma_format_idc);
    CHECK_EQ(obu_info.monochrome, info.monochrome);
    CHECK_EQ(obu_info.num_planes, info.num_planes);

    // A temporal unit with a dummy tile payload lets find_sequence_header()
    // exercise the real TD -> SEQ -> CLK scan.
    const std::uint8_t dummy_tile[] = {0x00};
    const std::vector<std::uint8_t> stream =
        av2enc::chroma::assemble_chroma_still(config, dummy_tile);
    av2enc::chroma::SequenceHeaderInfo found;
    CHECK(av2enc::chroma::find_sequence_header(stream, found, error));
    CHECK_EQ(found.chroma_format_idc, info.chroma_format_idc);

    // Cropping window syntax round-trips too.
    av2enc::chroma::ChromaSeqConfig cropped = config;
    cropped.seq_cropping_window_present_flag = true;
    cropped.seq_cropping_win_right_offset = 7;
    cropped.seq_cropping_win_bottom_offset = 3;
    av2enc::BitWriter crop_writer;
    av2enc::chroma::write_sequence_header_syntax(crop_writer, cropped);
    const std::vector<std::uint8_t> crop_bits = crop_writer.take_bytes();
    av2enc::chroma::SequenceHeaderInfo crop_info;
    CHECK(av2enc::chroma::read_sequence_header_syntax(crop_bits, crop_info,
                                                       error));
    CHECK(crop_info.cropping_window_present);
    CHECK_EQ(crop_info.crop_right, 7u);
    CHECK_EQ(crop_info.crop_bottom, 3u);
  }
  // Monochrome omits 12 bits relative to a non-monochrome header:
  // enable_sdp (1), cfl_ds_filter_index (2), enable_chroma_dctonly (1),
  // enable_cctx (1), separate_uv_delta_q (1), base_uv_ac_delta_q (5),
  // uv_ac_delta_q_enabled (1).  The chroma_format_idc uvlc() code length also
  // differs (0 -> 1 bit, 1 -> 3, 2 -> 3, 3 -> 5).
  const auto uvlc_bits = [](std::uint32_t value) -> std::size_t {
    std::size_t leading = 0;
    while (value >= ((1ULL << (leading + 1U)) - 1ULL)) {
      ++leading;
    }
    return leading * 2U + 1U;
  };
  // formats[] is {420, 422, 444, 400}; 400 is the last entry.
  CHECK_EQ(bit_sizes[0] - uvlc_bits(0),
           bit_sizes[3] - uvlc_bits(1) + 12);
  CHECK_EQ(bit_sizes[0] - uvlc_bits(0), bit_sizes[1] - uvlc_bits(3));
  CHECK_EQ(bit_sizes[0] - uvlc_bits(0), bit_sizes[2] - uvlc_bits(2));

  // Monochrome-inconsistent configs are rejected.
  av2enc::chroma::ChromaSeqConfig bad;
  bad.chroma = ChromaFormat::k400;
  bad.enable_chroma_dctonly = true;
  std::string error;
  CHECK(!av2enc::chroma::validate_chroma_seq_config(bad, error));
  bad = av2enc::chroma::ChromaSeqConfig{};
  bad.chroma = ChromaFormat::k400;
  bad.enable_cfl_intra = true;
  CHECK(!av2enc::chroma::validate_chroma_seq_config(bad, error));
  bad = av2enc::chroma::ChromaSeqConfig{};
  bad.chroma = ChromaFormat::k422;
  bad.enable_sdp = true;
  CHECK(!av2enc::chroma::validate_chroma_seq_config(bad, error));
  bad = av2enc::chroma::ChromaSeqConfig{};
  bad.chroma = ChromaFormat::k444;
  bad.use_128x128_superblock = true;
  CHECK(!av2enc::chroma::validate_chroma_seq_config(bad, error));
}

void test_tx_shapes() {
  struct Expected {
    int w;
    int h;
    int adj_w;
    int adj_h;
    int up_x;
    int up_y;
    int ctx;
    int row_shift;
    int col_shift;
    bool odd_rect;
    int dq_denom;
  };
  const Expected expected[] = {
      {64, 64, 32, 32, 2, 2, 4, 6, 13, false, 4},
      {32, 64, 32, 32, 1, 2, 4, 6, 12, true, 4},
      {64, 32, 32, 32, 2, 1, 4, 6, 12, true, 4},
      {32, 32, 32, 32, 1, 1, 3, 6, 13, false, 2},
      {16, 32, 16, 32, 1, 1, 3, 6, 12, true, 2},
      {32, 8, 32, 8, 1, 1, 2, 6, 13, false, 1},
      {16, 16, 16, 16, 1, 1, 2, 6, 13, false, 1},
      {8, 32, 8, 32, 1, 1, 2, 6, 13, false, 1},
  };
  for (const Expected& e : expected) {
    TxShape shape;
    CHECK(av2enc::chroma::make_tx_shape(e.w, e.h, shape));
    CHECK_EQ(shape.adj_w, e.adj_w);
    CHECK_EQ(shape.adj_h, e.adj_h);
    CHECK_EQ(shape.upscale_x, e.up_x);
    CHECK_EQ(shape.upscale_y, e.up_y);
    CHECK_EQ(shape.tx_size_ctx, e.ctx);
    CHECK_EQ(shape.row_shift, e.row_shift);
    CHECK_EQ(shape.col_shift, e.col_shift);
    CHECK_EQ(shape.odd_rect, e.odd_rect);
    CHECK_EQ(shape.dq_denom, e.dq_denom);
    CHECK_EQ(shape.seg_eob, e.adj_w * e.adj_h);
  }
}

void test_transform_roundtrip() {
  TxShape shape;
  CHECK(av2enc::chroma::make_tx_shape(32, 64, shape));
  const int n = shape.adj_w * shape.adj_h;
  std::vector<std::int16_t> residual(static_cast<std::size_t>(n), 12);

  std::vector<double> coeff(static_cast<std::size_t>(n), 0.0);
  av2enc::chroma::forward_dct(residual.data(), shape.adj_w, shape.adj_h,
                              shape.odd_rect, coeff.data());

  std::vector<int> levels(static_cast<std::size_t>(n), 0);
  const double factor = av2enc::chroma::quant_factor_for(shape);
  av2enc::chroma::quantize_coefficients(coeff.data(), n, factor, levels.data());
  CHECK(levels[0] != 0);

  std::vector<int> dequant(static_cast<std::size_t>(n), 0);
  av2enc::chroma::dequantize_block(levels.data(), n, shape.dq_denom,
                                   dequant.data());
  std::vector<int> back(static_cast<std::size_t>(n), 0);
  av2enc::chroma::inverse_dct(dequant.data(), shape, back.data());
  // The reconstructed DC should be close to the source residual (quantisation
  // is lossy, but a constant block should land within a few LSB).
  CHECK(std::abs(back[0] - 12) <= 3);

  // All-zero levels must give an all-zero residual.
  std::fill(levels.begin(), levels.end(), 0);
  av2enc::chroma::dequantize_block(levels.data(), n, shape.dq_denom,
                                   dequant.data());
  av2enc::chroma::inverse_dct(dequant.data(), shape, back.data());
  for (int v : back) {
    CHECK_EQ(v, 0);
  }

  // 64x64 uses the same adjusted grid but a different (nominal) shift; the
  // inverse must still be exactly zero for zero coefficients.
  TxShape square;
  CHECK(av2enc::chroma::make_tx_shape(64, 64, square));
  std::vector<int> square_back(static_cast<std::size_t>(n), 0);
  std::vector<int> square_dequant(static_cast<std::size_t>(n), 0);
  av2enc::chroma::inverse_dct(square_dequant.data(), square,
                              square_back.data());
  for (int v : square_back) {
    CHECK_EQ(v, 0);
  }
}

void test_encoder_smoke() {
  const ChromaFormat formats[] = {ChromaFormat::k420, ChromaFormat::k422,
                                  ChromaFormat::k444, ChromaFormat::k400};
  const int width = 64;
  const int height = 64;
  const std::size_t luma = static_cast<std::size_t>(width) * height;
  const std::size_t chroma = static_cast<std::size_t>(width / 2) * (height / 2);
  std::vector<std::uint8_t> y(luma, 128);
  std::vector<std::uint8_t> u(chroma, 140);
  std::vector<std::uint8_t> v(chroma, 116);
  // A diagonal ramp so the block is not perfectly flat.
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      y[static_cast<std::size_t>(row) * width + col] =
          static_cast<std::uint8_t>((col * 3 + row * 5) & 0xFF);
    }
  }

  for (ChromaFormat format : formats) {
    av2enc::chroma::ChromaPicture picture;
    std::string error;
    CHECK(av2enc::chroma::make_chroma_picture(width, height, format, y, u, v,
                                               picture, error));
    CHECK_EQ(picture.width, width);
    CHECK_EQ(picture.height, height);
    av2enc::chroma::ChromaStillOptions options;
    av2enc::chroma::ChromaStillResult result;
    CHECK(av2enc::chroma::encode_chroma_still_picture(picture, options, result,
                                                       error));
    CHECK(!result.frame.empty());
    CHECK(result.tile_bytes > 0);
    CHECK_EQ(result.leaves, 1);
    CHECK_EQ(result.shape.chroma, format);
    CHECK_EQ(result.shape.monochrome, format == ChromaFormat::k400);

    av2enc::chroma::SequenceHeaderInfo info;
    CHECK(av2enc::chroma::find_sequence_header(result.frame, info, error));
    CHECK_EQ(info.chroma_format_idc, av2enc::chroma::chroma_format_idc(format));
    CHECK_EQ(info.max_frame_width, 64u);
    CHECK_EQ(info.max_frame_height, 64u);
    CHECK_EQ(info.monochrome, format == ChromaFormat::k400);

    // Reconstructed planes must exist for every present plane.
    const int planes = format == ChromaFormat::k400 ? 1 : 3;
    for (int plane = 0; plane < planes; ++plane) {
      CHECK(!result.recon[plane].empty());
      std::vector<std::uint8_t> visible;
      CHECK(av2enc::chroma::copy_visible_plane(result, plane, visible));
      const auto& shape = result.shape.planes[static_cast<std::size_t>(plane)];
      CHECK_EQ(visible.size(),
               static_cast<std::size_t>(shape.visible_width) *
                   static_cast<std::size_t>(shape.visible_height));
    }
    if (format == ChromaFormat::k400) {
      CHECK(result.recon[1].empty());
      CHECK(result.recon[2].empty());
    }
  }
}

}  // namespace

int main() {
  test_format_table();
  test_chroma_mi_size();
  test_geometry();
  test_tool_legality();
  test_headers();
  test_tx_shapes();
  test_transform_roundtrip();
  test_encoder_smoke();

  if (g_failures == 0) {
    std::cout << "chroma_test: " << g_checks << " checks, PASS\n";
    return 0;
  }
  std::cerr << "chroma_test: " << g_checks << " checks, " << g_failures
            << " FAILURES\n";
  return 1;
}
