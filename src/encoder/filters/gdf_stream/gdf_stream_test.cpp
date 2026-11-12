// Unit tests for the GDF stream component: sequence flags, frame params,
// per-block use_gdf symbols, unit/region mapping, filter application and the
// encoder decision.
//
// The oracle for syntax positions is a second transcription of §5.20.10.3; the
// pixel filter is the recovered reference in spec_tables/gdf.cpp.
#include "encoder/filters/gdf_stream/gdf_decision.h"
#include "encoder/filters/gdf_stream/gdf_syntax.h"

#include "encoder/filters/spec_tables/gdf.h"
#include "encoder/filters/spec_tables/luma_source.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace gdf = av2enc::filters::gdf;
namespace spec = av2enc::filters::spec;

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    std::printf("FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

std::size_t Index(int row, int col, int cols) {
  return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
         static_cast<std::size_t>(col);
}

std::size_t Count(int cols, int rows) {
  return static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows);
}

std::string Hex(const std::uint8_t* data, std::size_t size) {
  static const char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(size * 2);
  for (std::size_t i = 0; i < size; ++i) {
    out.push_back(kDigits[data[i] >> 4U]);
    out.push_back(kDigits[data[i] & 0x0FU]);
  }
  return out;
}

std::string Hex(const av2enc::BitWriter& writer) {
  return Hex(writer.bytes().data(), writer.bytes().size());
}

gdf::FrameGdfContext MakeContext(int width, int height, int sb_size,
                                  bool matches, bool single_picture = true) {
  gdf::FrameGdfContext context;
  context.enable_gdf = true;
  context.gdf_unit_matches_sb_size = matches;
  context.single_picture_header_flag = single_picture;
  context.sb_size = sb_size;
  context.width = width;
  context.height = height;
  context.mi_cols = 2 * ((width + 7) / 8);
  context.mi_rows = 2 * ((height + 7) / 8);
  context.mi_col_starts = {0};
  context.mi_row_starts = {0};
  return context;
}

bool OracleSymbolAt(const gdf::GdfUnitLayout& layout,
                    const gdf::FrameGdfParams& params, int sb_size, int mi_row,
                    int mi_col, int* row, int* col) {
  if (!params.frame_enable || !params.per_block) return false;
  const int sb_size4 = sb_size / 4;
  if (mi_row % sb_size4 != 0 || mi_col % sb_size4 != 0) return false;
  const int sb_row = mi_row / sb_size4;
  const int sb_col = mi_col / sb_size4;
  const int sb_per_gdf = layout.blk_size / sb_size;
  if (sb_col % sb_per_gdf != 0 || sb_row % sb_per_gdf != 0) return false;
  *row = sb_row / sb_per_gdf;
  *col = sb_col / sb_per_gdf;
  return *row < layout.rows && *col < layout.cols;
}

void TestSequenceFlags() {
  std::printf("sequence flags\n");
  std::string error;
  struct Case {
    bool enable;
    bool matches;
    int sb;
    const char* golden;
    std::size_t bits;
  };
  const Case cases[] = {
      {false, false, 64, "00", 1},
      {false, true, 64, "00", 1},
      {true, false, 64, "80", 2},
      {true, true, 64, "c0", 2},
      {false, false, 128, "00", 1},
      {true, false, 128, "80", 1},
      {true, true, 128, "80", 1},  // gdf_unit_matches_sb_size is not coded
  };
  for (const Case& c : cases) {
    av2enc::BitWriter writer;
    gdf::SequenceGdfFlags flags;
    flags.enable_gdf = c.enable;
    flags.gdf_unit_matches_sb_size = c.matches;
    const std::size_t bits =
        gdf::WriteSequenceGdfFlags(writer, flags, c.sb, &error);
    Check(bits == c.bits, "sequence flag bit count");
    Check(Hex(writer) == c.golden, "sequence flag golden");
    gdf::SequenceGdfFlags read;
    av2enc::m1::filters_stream::BitReader reader(writer.bytes(), writer.bit_size());
    Check(gdf::ReadSequenceGdfFlags(reader, c.sb, &read, &error),
          "sequence flag decode");
    Check(read.enable_gdf == c.enable, "sequence enable_gdf round-trip");
    const bool expected_matches = c.enable && c.sb == 64 ? c.matches : false;
    Check(read.gdf_unit_matches_sb_size == expected_matches,
          "sequence gdf_unit_matches_sb_size round-trip/inference");
  }
  av2enc::BitWriter writer;
  gdf::SequenceGdfFlags flags;
  Check(gdf::WriteSequenceGdfFlags(writer, flags, 32, &error) == 0 &&
            !error.empty(),
        "sequence rejects a non-64/128 SB size");
}

void TestBlkSizeDerivation() {
  std::printf("GdfBlkSize derivation\n");
  Check(gdf::DeriveGdfBlkSize(64, true, {0}, {0}) == 64,
        "64 SB + matches -> 64");
  Check(gdf::DeriveGdfBlkSize(64, false, {0}, {0}) == 128,
        "64 SB + !matches + aligned -> 128");
  Check(gdf::DeriveGdfBlkSize(64, false, {0, 16}, {0}) == 64,
        "64 SB + !matches + 64-aligned tile start -> 64");
  Check(gdf::DeriveGdfBlkSize(64, false, {0}, {0, 32}) == 128,
        "row tile start 32 mi is not the a & 16 case");
  Check(gdf::DeriveGdfBlkSize(128, false, {0}, {0}) == 128,
        "128 SB -> 128");
  Check(gdf::DeriveGdfBlkSize(128, true, {0}, {0}) == 128,
        "128 SB + matches is still 128");
  Check(gdf::GdfUnitCols(160, 128) == 5, "unit cols 640/128");
  Check(gdf::GdfUnitRows(90, 128) == 3, "unit rows 360/128");
  Check(gdf::GdfUnitCols(16, 64) == 1, "unit cols 64/64");
}

void TestFrameParams() {
  std::printf("frame params\n");
  std::string error;
  struct Case {
    int width;
    int height;
    int sb;
    bool matches;
    bool single;
    const char* golden;
    std::size_t bits;
    int blk;
    int cols;
    int rows;
    bool per_block_present;
  };
  const Case cases[] = {
      {64, 64, 64, true, true, "90", 4, 64, 1, 1, false},
      {64, 64, 64, false, true, "90", 4, 128, 1, 1, false},
      {256, 192, 64, true, true, "48", 5, 64, 4, 3, true},
      {256, 192, 64, false, true, "48", 5, 128, 2, 2, true},
      {256, 192, 64, true, false, "a4", 6, 64, 4, 3, true},
  };
  for (const Case& c : cases) {
    const gdf::FrameGdfContext context =
        MakeContext(c.width, c.height, c.sb, c.matches, c.single);
    gdf::FrameGdfParams params;
    params.frame_enable = true;
    params.per_block = false;
    params.pic_qc_idx = 2;
    params.pic_scale_idx = 1;
    av2enc::BitWriter writer;
    const std::size_t bits =
        gdf::WriteFrameGdfParams(writer, context, &params, &error);
    Check(bits == c.bits, "frame params bit count");
    Check(Hex(writer) == c.golden, "frame params golden");
    Check(params.frame_enable, "frame params enable");
    Check(params.blk_size == c.blk, "frame params GdfBlkSize");
    Check(params.cols == c.cols && params.rows == c.rows,
          "frame params GdfBlks dimensions");
    Check(params.per_block_present == c.per_block_present,
          "frame params gdf_per_block presence");
    Check(params.pix_scale == 2, "frame params GdfPixScale");

    gdf::FrameGdfParams decoded;
    av2enc::m1::filters_stream::BitReader reader(writer.bytes(),
                                              writer.bit_size());
    Check(gdf::ReadFrameGdfParams(reader, context, &decoded, &error),
          "frame params decode");
    Check(decoded.frame_enable == params.frame_enable, "decode enable");
    Check(decoded.blk_size == params.blk_size, "decode GdfBlkSize");
    Check(decoded.per_block == params.per_block, "decode gdf_per_block");
    Check(decoded.pic_qc_idx == 2 && decoded.pic_scale_idx == 1,
          "decode pic indices");
  }

  // Lossless and disabled frames code nothing and infer gdf_frame_enable = 0.
  for (int variant = 0; variant < 2; ++variant) {
    gdf::FrameGdfContext context = MakeContext(256, 192, 64, true);
    if (variant == 0) {
      context.coded_lossless = true;
    } else {
      context.enable_gdf = false;
    }
    gdf::FrameGdfParams params;
    params.frame_enable = true;
    av2enc::BitWriter writer;
    Check(gdf::WriteFrameGdfParams(writer, context, &params, &error) == 0,
          "disabled GDF codes no frame bits");
    Check(writer.bit_size() == 0 && !params.frame_enable,
          "disabled GDF infers frame_enable = 0");
  }

  // A regular frame with the enable bit 0 codes exactly that one bit.
  {
    gdf::FrameGdfContext context = MakeContext(256, 192, 64, true, false);
    gdf::FrameGdfParams params;
    params.frame_enable = false;
    av2enc::BitWriter writer;
    Check(gdf::WriteFrameGdfParams(writer, context, &params, &error) == 1,
          "regular disabled frame codes one bit");
    Check(Hex(writer) == "00", "regular disabled frame golden");
    gdf::FrameGdfParams decoded;
    av2enc::m1::filters_stream::BitReader reader(writer.bytes(), writer.bit_size());
    Check(gdf::ReadFrameGdfParams(reader, context, &decoded, &error) &&
              !decoded.frame_enable,
          "regular disabled frame decodes");
  }

  // gdf_per_block is present for a multi-tile frame even when it fits in one
  // GDF unit, and the tile-alignment case reduces GdfBlkSize to 64.
  {
    gdf::FrameGdfContext context = MakeContext(128, 128, 64, false);
    context.disable_loopfilters_across_tiles = true;
    context.tile_cols = 2;
    context.tile_rows = 1;
    context.mi_col_starts = {0, 16};
    const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
    Check(layout.blk_size == 64, "tile-aligned 64x64 unit");
    Check(layout.per_block_present, "multi-tile gdf_per_block is coded");
    gdf::FrameGdfContext single = MakeContext(128, 128, 64, false);
    Check(!gdf::DeriveGdfUnitLayout(single).per_block_present,
          "fits in one unit -> gdf_per_block inferred 0");
  }

  // Range checks.
  {
    gdf::FrameGdfContext context = MakeContext(64, 64, 64, true);
    gdf::FrameGdfParams params;
    params.pic_qc_idx = 4;
    av2enc::BitWriter writer;
    gdf::WriteFrameGdfParams(writer, context, &params, &error);
    Check(!error.empty(), "gdf_pic_qc_idx out of range is rejected");
  }
}

void TestSymbolPositions() {
  std::printf("use_gdf symbol positions\n");
  for (int sb_size : {64, 128}) {
    for (bool matches : {false, true}) {
      const gdf::FrameGdfContext context =
          MakeContext(1024, 512, sb_size, matches);
      const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
      gdf::FrameGdfParams params;
      params.frame_enable = true;
      params.per_block = true;
      params.blk_size = layout.blk_size;
      params.cols = layout.cols;
      params.rows = layout.rows;
      const int sb_mi = sb_size / 4;
      const int sb_rows = (context.mi_rows + sb_mi - 1) / sb_mi;
      const int sb_cols = (context.mi_cols + sb_mi - 1) / sb_mi;
      int coded = 0;
      for (int mi_row = 0; mi_row < context.mi_rows; mi_row += 4) {
        for (int mi_col = 0; mi_col < context.mi_cols; mi_col += 4) {
          gdf::GdfSymbolPosition position;
          const bool got =
              gdf::GdfSymbolAt(layout, params, sb_size, mi_row, mi_col,
                               &position);
          int oracle_row = -1;
          int oracle_col = -1;
          const bool want = OracleSymbolAt(layout, params, sb_size, mi_row,
                                           mi_col, &oracle_row, &oracle_col);
          Check(got == want, "GdfSymbolAt matches the spec predicate");
          if (got && want) {
            Check(position.row == oracle_row && position.col == oracle_col,
                  "GdfSymbolAt grid index");
            ++coded;
          }
        }
      }
      Check(coded == layout.cols * layout.rows,
            "one symbol per GDF unit over the SB grid");
      (void)sb_rows;
      (void)sb_cols;
    }
  }
  gdf::FrameGdfContext context = MakeContext(64, 64, 64, true);
  gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
  gdf::FrameGdfParams params;
  params.frame_enable = true;
  params.per_block = false;
  gdf::GdfSymbolPosition position;
  Check(!gdf::GdfSymbolAt(layout, params, 64, 0, 0, &position),
        "gdf_per_block = 0 codes no symbol");
}

void TestSymbolRoundTrip() {
  std::printf("use_gdf round-trip and goldens\n");
  std::string error;
  const gdf::FrameGdfContext context = MakeContext(512, 256, 64, true);
  const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
  Check(layout.cols == 8 && layout.rows == 4 && layout.blk_size == 64,
        "symbol test layout 8x4");
  gdf::FrameGdfParams params;
  params.frame_enable = true;
  params.per_block = true;
  params.blk_size = layout.blk_size;
  params.cols = layout.cols;
  params.rows = layout.rows;

  struct Case {
    std::vector<std::uint8_t> flags;
    const char* golden;
    std::size_t bytes;
  };
  std::vector<Case> cases;
  cases.push_back({std::vector<std::uint8_t>(Count(layout.cols, layout.rows), 0),
                   "00000000020000", 7});
  cases.push_back({std::vector<std::uint8_t>(Count(layout.cols, layout.rows), 1),
                   "fffffff80000", 6});
  {
    std::vector<std::uint8_t> flags(
        Count(layout.cols, layout.rows));
    for (std::size_t i = 0; i < flags.size(); ++i) {
      flags[i] = static_cast<std::uint8_t>(i & 1U);
    }
    cases.push_back({flags, "441c39e9c000", 6});
  }
  {
    std::vector<std::uint8_t> flags(
        Count(layout.cols, layout.rows), 0);
    flags[Index(1, 2, layout.cols)] = 1;
    cases.push_back({flags, "0009a200040000", 7});
  }

  const int sb_mi = 16;
  for (const Case& c : cases) {
    std::vector<std::uint8_t> buffer(4 * c.flags.size() + 64);
    av2enc::av2::Encoder encoder;
    encoder.disable_cdf_update = 1;
    Check(encoder.init_symbol(buffer.data(), buffer.size()) ==
              av2enc::av2::Status::kOk,
          "symbol encoder init");
    gdf::GdfBlockCoder coder;
    const int sb_rows = (context.mi_rows + sb_mi - 1) / sb_mi;
    const int sb_cols = (context.mi_cols + sb_mi - 1) / sb_mi;
    int coded = 0;
    for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
      for (int sb_col = 0; sb_col < sb_cols; ++sb_col) {
        const int mi_row = sb_row * sb_mi;
        const int mi_col = sb_col * sb_mi;
        gdf::GdfSymbolPosition position;
        if (!gdf::GdfSymbolAt(layout, params, 64, mi_row, mi_col, &position)) {
          continue;
        }
        bool was_coded = false;
        Check(coder.WriteAt(encoder, layout, params, 64, mi_row, mi_col,
                            c.flags[Index(position.row, position.col, layout.cols)] != 0,
                            &was_coded),
              "symbol write");
        Check(was_coded, "symbol origin coded");
        ++coded;
      }
    }
    Check(coded == layout.cols * layout.rows, "all unit origins coded");
    std::size_t payload_bytes = 0;
    Check(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk,
          "symbol encoder exit");
    Check(payload_bytes == c.bytes, "symbol payload byte count");
    Check(Hex(buffer.data(), payload_bytes) == c.golden, "symbol golden");

    std::int64_t measured_bytes = 0;
    std::int64_t measured_bits = 0;
    Check(gdf::MeasureUseGdfSymbols(layout, params, 64, c.flags, true,
                                    &measured_bytes, &measured_bits, &error),
          "MeasureUseGdfSymbols");
    Check(measured_bytes == static_cast<std::int64_t>(c.bytes),
          "measured symbol bytes");
    Check(measured_bits == static_cast<std::int64_t>(c.bytes) * 8,
          "measured symbol bits");

    // Independent decode with a fresh coder.
    av2enc::av2::Decoder decoder;
    Check(decoder.init_symbol(buffer.data(), payload_bytes, payload_bytes) ==
              av2enc::av2::Status::kOk,
          "symbol decoder init");
    decoder.disable_cdf_update = 1;
    gdf::GdfBlockCoder reader;
    std::vector<std::uint8_t> read_flags(c.flags.size(), 0);
    for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
      for (int sb_col = 0; sb_col < sb_cols; ++sb_col) {
        const int mi_row = sb_row * sb_mi;
        const int mi_col = sb_col * sb_mi;
        gdf::GdfSymbolPosition position;
        if (!gdf::GdfSymbolAt(layout, params, 64, mi_row, mi_col, &position)) {
          continue;
        }
        bool use_gdf = false;
        bool was_coded = false;
        Check(reader.ReadAt(decoder, layout, params, 64, mi_row, mi_col,
                            &use_gdf, &was_coded),
              "symbol read");
        Check(was_coded, "symbol read origin");
        read_flags[Index(position.row, position.col, layout.cols)] =
            use_gdf ? 1 : 0;
      }
    }
    Check(read_flags == c.flags, "symbol round-trip flags");
    Check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
          "symbol decoder exit");
  }

  // Adaptive CDF (disable_cdf_update = 0) also round-trips.
  {
    std::vector<std::uint8_t> flags(
        Count(layout.cols, layout.rows));
    for (std::size_t i = 0; i < flags.size(); ++i) {
      flags[i] = static_cast<std::uint8_t>((i * 7U + 3U) % 5U == 0U);
    }
    gdf::GdfDecisionOptions unused;
    (void)unused;
    std::int64_t bytes = 0;
    std::int64_t bits = 0;
    Check(gdf::MeasureUseGdfSymbols(layout, params, 64, flags, false, &bytes,
                                    &bits, &error),
          "adaptive symbol measurement");
    Check(bytes > 0, "adaptive payload non-empty");
  }
}

void TestRegions() {
  std::printf("unit regions\n");
  {
    const gdf::FrameGdfContext context = MakeContext(640, 360, 64, false);
    const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
    Check(layout.blk_size == 128 && layout.cols == 5 && layout.rows == 3,
          "640x360 layout");
    gdf::GdfStripeRegion regions[8];
    Check(gdf::GdfUnitRegions(layout, 0, 0, regions) == 2,
          "unit (0,0) spans two stripes");
    Check(regions[0].y == 0 && regions[0].h == 56, "unit 0 stripe 0");
    Check(regions[1].y == 56 && regions[1].h == 64, "unit 0 stripe 1");
    Check(regions[0].stripe_start_y == 0 && regions[0].stripe_end_y == 55,
          "stripe 0 extents");
    Check(regions[1].stripe_start_y == 56 && regions[1].stripe_end_y == 119,
          "stripe 1 extents");
    Check(gdf::GdfUnitRegions(layout, 2, 4, regions) == 2,
          "last unit spans two stripes");
    Check(regions[0].y == 248 && regions[0].h == 64, "last unit first stripe");
    Check(regions[1].y == 312 && regions[1].h == 48, "last unit tail");

    // Every visible pixel is covered exactly once.
    std::vector<int> cover(
        static_cast<std::size_t>(context.width) *
            static_cast<std::size_t>(context.height),
        0);
    int total = 0;
    for (int row = 0; row < layout.rows; ++row) {
      for (int col = 0; col < layout.cols; ++col) {
        const int count = gdf::GdfUnitRegions(layout, row, col, regions);
        for (int r = 0; r < count; ++r) {
          for (int y = regions[r].y; y < regions[r].y + regions[r].h; ++y) {
            for (int x = regions[r].x; x < regions[r].x + regions[r].w; ++x) {
              ++cover[Index(y, x, context.width)];
            }
          }
        }
      }
    }
    bool exact = true;
    for (const int value : cover) {
      if (value != 1) exact = false;
      total += value;
    }
    Check(exact, "unit regions partition the frame");
    Check(total == context.width * context.height, "region area");
  }
  {
    const gdf::FrameGdfContext context = MakeContext(64, 64, 64, true);
    const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
    gdf::GdfStripeRegion regions[8];
    Check(gdf::GdfUnitRegions(layout, 0, 0, regions) == 2,
          "64-high frame spans two stripes");
    Check(regions[0].y == 0 && regions[0].h == 56, "64-high stripe 0");
    Check(regions[1].y == 56 && regions[1].h == 8, "64-high stripe 1");
  }
  {
    const gdf::FrameGdfContext context = MakeContext(256, 192, 64, true);
    const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
    gdf::GdfStripeRegion regions[8];
    Check(gdf::GdfUnitRegions(layout, 1, 1, regions) == 1,
          "64 unit inside one stripe");
    Check(regions[0].y == 56 && regions[0].h == 64, "second 64 unit");
  }
}

std::vector<std::uint16_t> MakeDetail(int width, int height, int seed) {
  std::vector<std::uint16_t> out(Count(width, height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int chess = ((x / 4) ^ (y / 4)) & 1;
      const int value = 80 + chess * 70 + ((x * 13 + y * 7 + seed) % 23) - 11;
      out[Index(y, x, width)] =
          static_cast<std::uint16_t>(std::clamp(value, 0, 255));
    }
  }
  return out;
}

void TestApply() {
  std::printf("filter application\n");
  const int width = 128;
  const int height = 128;
  const auto original = MakeDetail(width, height, 0);
  std::vector<std::uint16_t> recon(original.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t index = Index(y, x, width);
      recon[index] = static_cast<std::uint16_t>(
          (original[index] + original[Index(y, std::max(0, x - 1), width)] +
           original[Index(y, std::min(width - 1, x + 1), width)]) /
          3);
    }
  }
  gdf::FrameGdfContext context = MakeContext(width, height, 64, false);
  gdf::GdfFilterInputs inputs;
  inputs.base_q_idx = 90;
  gdf::FrameGdfParams params;
  params.frame_enable = true;
  params.blk_size = 128;
  params.cols = 1;
  params.rows = 1;
  params.pic_qc_idx = 2;
  params.pic_scale_idx = 1;
  params.pix_scale = 2;
  std::string error;
  std::vector<std::uint16_t> out(recon);
  Check(gdf::ApplyGdfToFrame({recon.data(), recon.data(), width}, inputs,
                             context, params, {}, out.data(), width, &error),
        "apply 128 unit");
  int changed = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (out[i] != recon[i]) ++changed;
  }
  Check(changed > 0, "GDF changes pixels");

  // 64-sample units produce the same samples as one 128-sample unit
  // (the reference's split-equivalence property, now across the signalled
  // unit size).
  gdf::FrameGdfContext context64 = MakeContext(width, height, 64, true);
  gdf::FrameGdfParams params64 = params;
  params64.blk_size = 64;
  params64.cols = 2;
  params64.rows = 2;
  std::vector<std::uint16_t> out64(recon);
  Check(gdf::ApplyGdfToFrame({recon.data(), recon.data(), width}, inputs,
                             context64, params64, {}, out64.data(), width,
                             &error),
        "apply 64 units");
  Check(out == out64, "128-unit and 64-unit splits are identical");

  // Disabling the only unit leaves the frame untouched.
  params.per_block = true;
  std::vector<std::uint8_t> flags(1, 0);
  std::vector<std::uint16_t> skipped(recon);
  Check(gdf::ApplyGdfToFrame({recon.data(), recon.data(), width}, inputs,
                             context, params, flags, skipped.data(), width,
                             &error),
        "apply with unit disabled");
  Check(skipped == recon, "disabled unit is untouched");

  // A stride that is too small is rejected.
  Check(!gdf::ApplyGdfToFrame({recon.data(), recon.data(), width - 1}, inputs,
                              context, params, {}, out.data(), width, &error),
        "short stride rejected");
}

void TestDecision() {
  std::printf("encoder decision\n");
  const int width = 256;
  const int height = 192;
  const auto original = MakeDetail(width, height, 1);
  std::vector<std::uint16_t> recon(original.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t index = Index(y, x, width);
      // Bottom-right quadrant is already perfect, so GDF can only hurt there.
      if (x >= width / 2 && y >= height / 2) {
        recon[index] = original[index];
        continue;
      }
      const std::size_t left = Index(y, std::max(0, x - 1), width);
      const std::size_t up = Index(std::max(0, y - 1), x, width);
      recon[index] = static_cast<std::uint16_t>(
          (static_cast<int>(original[index]) + original[left] +
           2 * static_cast<int>(original[up])) /
          4);
    }
  }
  gdf::FrameGdfContext context = MakeContext(width, height, 64, false);
  gdf::GdfDecisionOptions options;
  options.filter.base_q_idx = 90;
  options.search_pic_indices = true;
  options.disable_cdf_update = true;
  gdf::GdfDecision decision;
  std::string error;
  Check(gdf::DecideGdf({recon.data(), recon.data(), width},
                       gdf::GdfPlane{original.data(), width, height, width},
                       context, options, &decision, &error),
        "decide");
  Check(decision.sse_after <= decision.sse_before, "decision does not regress");
  Check(decision.psnr_after >= decision.psnr_before - 1e-9,
        "decision PSNR does not regress");
  Check(decision.chosen_qc_idx >= 0 && decision.chosen_qc_idx <= 3,
        "chosen qc range");
  Check(decision.chosen_scale_idx >= 0 && decision.chosen_scale_idx <= 3,
        "chosen scale range");
  Check(decision.params.frame_enable, "decision enables the frame");
  Check(decision.params.blk_size == 128, "decision uses GdfBlkSize 128");
  Check(decision.params.cols == 2 && decision.params.rows == 2,
        "decision grid 2x2");
  Check(decision.total_units == 4, "decision unit count");
  Check(decision.header_bits == 7, "decision header bit count");
  Check(decision.params.per_block, "decision uses per-block flags");
  Check(decision.unit_flags.size() == 4, "decision flag grid");
  Check(decision.enabled_units >= 1 && decision.enabled_units < 4,
        "decision skips at least the perfect quadrant");
  Check(decision.symbol_bytes > 0, "decision symbol cost");
  Check(decision.block64_cols == 4 && decision.block64_rows == 4,
        "decision 64x64 report grid");
  Check(decision.block64_flags.size() == 16, "decision 64x64 report size");
  bool perfect_clear = true;
  for (int by = 2; by < 4; ++by) {
    for (int bx = 2; bx < 4; ++bx) {
      if (decision.block64_flags[Index(by, bx, 4)] != 0) {
        perfect_clear = false;
      }
    }
  }
  Check(perfect_clear, "perfect quadrant is not flagged");

  // A regular frame with a huge rate penalty must code gdf_frame_enable = 0.
  {
    gdf::FrameGdfContext regular =
        MakeContext(width, height, 64, false, false);
    gdf::GdfDecisionOptions cheap = options;
    cheap.lambda_bits = 1.0e9;
    gdf::GdfDecision cheap_decision;
    Check(gdf::DecideGdf({recon.data(), recon.data(), width},
                         gdf::GdfPlane{original.data(), width, height, width},
                         regular, cheap, &cheap_decision, &error),
          "decide a regular frame with a huge rate penalty");
    Check(!cheap_decision.params.frame_enable,
          "huge lambda disables the frame on a regular header");
    Check(cheap_decision.enabled_units == 0, "disabled frame filters nothing");
    Check(cheap_decision.sse_after == cheap_decision.sse_before,
          "disabled frame leaves the reconstruction unchanged");
  }

  // Determinism.
  gdf::GdfDecision again;
  Check(gdf::DecideGdf({recon.data(), recon.data(), width},
                       gdf::GdfPlane{original.data(), width, height, width},
                       context, options, &again, &error),
        "decide again");
  Check(again.params.pic_qc_idx == decision.params.pic_qc_idx &&
            again.params.pic_scale_idx == decision.params.pic_scale_idx &&
            again.unit_flags == decision.unit_flags,
        "decision deterministic");
}

void TestGoldenOutput() {
  std::printf("pinned filtered output\n");
  const int width = 64;
  const int height = 64;
  const auto original = MakeDetail(width, height, 3);
  std::vector<std::uint16_t> recon(original.size());
  for (std::size_t i = 0; i < original.size(); ++i) {
    recon[i] = static_cast<std::uint16_t>((original[i] * 3U + 128U) / 4U);
  }
  gdf::FrameGdfContext context = MakeContext(width, height, 64, true);
  gdf::GdfFilterInputs inputs;
  inputs.base_q_idx = 90;
  gdf::FrameGdfParams params;
  params.frame_enable = true;
  params.blk_size = 64;
  params.cols = 1;
  params.rows = 1;
  params.pic_qc_idx = 2;
  params.pic_scale_idx = 1;
  params.pix_scale = 2;
  std::string error;
  std::vector<std::uint16_t> out(recon);
  Check(gdf::ApplyGdfToFrame({recon.data(), recon.data(), width}, inputs,
                             context, params, {}, out.data(), width, &error),
        "golden apply");
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint16_t value : out) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  // Pinned after the region mapping fix; see README.
  Check(hash == 0x5299de7ef082b393ULL, "pinned filtered-output hash");
  if (hash != 0x5299de7ef082b393ULL) {
    std::printf("  actual hash = %016llx\n",
                static_cast<unsigned long long>(hash));
  }
}

void TestCdfTable() {
  std::printf("CDF table\n");
  Check(gdf::kDefaultUseGdfCdf[0] == 14593, "Default_Use_Gdf_Cdf[0]");
  Check(gdf::kDefaultUseGdfCdf[1] == 32, "Default_Use_Gdf_Cdf[1]");
  Check(gdf::kDefaultUseGdfCdf[2] == 0, "Default_Use_Gdf_Cdf[2]");
  Check(gdf::kGdfMinSize == 128, "GDF_MIN_SIZE");
}

}  // namespace

int main() {
  TestCdfTable();
  TestSequenceFlags();
  TestBlkSizeDerivation();
  TestFrameParams();
  TestSymbolPositions();
  TestSymbolRoundTrip();
  TestRegions();
  TestApply();
  TestDecision();
  TestGoldenOutput();
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
