// Tests for the M1 filter/restoration syntax adapters.
//
// Header writers use raw descriptors and are checked against pinned bit
// strings built field-by-field from the spec.  Tile syntax is checked against
// the normative §8.2 decoder plus pinned payload bytes (regression pins), and
// every header has a byte/bit-level round-trip through the mirror reader.

#include "codec/filters/stream/filters_stream.h"

#include "test_common.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace av2enc::still::filters_stream;
using av2enc::BitWriter;
namespace av2 = av2enc::av2;
namespace restoration = av2enc::restoration;

std::string bit_string(const BitWriter& writer) {
  std::string out;
  const std::span<const std::uint8_t> bytes = writer.bytes();
  for (std::size_t i = 0; i < writer.bit_size(); ++i) {
    const bool bit =
        ((bytes[i / 8U] >> (7U - static_cast<unsigned>(i & 7U))) & 1U) != 0U;
    out.push_back(bit ? '1' : '0');
  }
  return out;
}

std::string hex_string(std::span<const std::uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4U]);
    out.push_back(kDigits[b & 0x0fU]);
  }
  return out;
}

BitReader reader_for(const BitWriter& writer) {
  return BitReader(writer.bytes(), writer.bit_size());
}

// ---------------------------------------------------------------------------
// §5.4.10
// ---------------------------------------------------------------------------

void test_sequence_filter_config() {
  {
    SequenceFilterConfig c;
    BitWriter w;
    std::string error;
    const std::size_t bits =
        write_sequence_filter_config(w, c, true, &error);
    CHECK(error.empty());
    CHECK(bits == 7U);
    CHECK(bit_string(w) == "0000000");
    SequenceFilterConfig back;
    BitReader r = reader_for(w);
    CHECK(read_sequence_filter_config(r, true, true, &back, &error));
    CHECK(error.empty());
    CHECK(!back.enable_cdef && !back.enable_restoration && !back.enable_ccso);
    CHECK(r.bits_left() == 0U);
  }
  {
    SequenceFilterConfig c;
    c.enable_cdef = true;
    BitWriter w;
    std::string error;
    const std::size_t bits =
        write_sequence_filter_config(w, c, true, &error);
    CHECK(bits == 7U);
    CHECK(bit_string(w) == "0100000");
  }
  {
    SequenceFilterConfig c;
    c.enable_cdef = true;
    c.enable_restoration = true;
    c.lr_disable_luma_pc_wiener = true;
    c.lr_disable_luma_wiener_ns = false;
    c.lr_uv_present = true;
    c.lr_disable_chroma_wiener_ns = false;
    c.enable_ccso = true;
    c.ccso_unit_matches_sb_size = true;
    BitWriter w;
    std::string error;
    const std::size_t bits =
        write_sequence_filter_config(w, c, true, &error);
    CHECK(error.empty());
    CHECK(bits == 12U);
    CHECK(bit_string(w) == "010110101100");
    SequenceFilterConfig back;
    BitReader r = reader_for(w);
    CHECK(read_sequence_filter_config(r, true, true, &back, &error));
    CHECK(back.enable_cdef && back.enable_restoration && back.enable_ccso);
    CHECK(back.lr_disable_luma_pc_wiener);
    CHECK(!back.lr_disable_luma_wiener_ns);
    CHECK(back.lr_uv_present);
    CHECK(back.ccso_unit_matches_sb_size);
  }
  {
    SequenceFilterConfig c;
    c.enable_gdf = true;
    BitWriter w;
    std::string error;
    CHECK(write_sequence_filter_config(w, c, true, &error) == 0U);
    CHECK(!error.empty());
  }
}

// ---------------------------------------------------------------------------
// §5.18.7.10
// ---------------------------------------------------------------------------

void test_cdef_header() {
  CdefHeaderContext ctx;
  ctx.enable_cdef = true;
  CdefParams p;
  p.damping_minus_3 = 0;    // CdefDamping = 3
  p.strengths_minus_1 = 1;  // two sets
  p.num_strengths = 2;
  p.on_skip_txfm_frame_enable = false;
  // set[0] stays all zero.
  p.set[1].y_pri = 4;
  p.set[1].y_sec = 4;  // syntax 3
  p.set[1].uv_pri = 2;
  p.set[1].uv_sec = 1;

  BitWriter w;
  std::string error;
  const std::size_t bits = write_cdef_params(w, p, ctx, &error);
  CHECK(error.empty());
  // 00 | 001 | 0 | 1 00 1 00 | 0 0100 11 0 0010 01
  const std::string expected =
      "00" "001" "0" "1" "00" "1" "00" "0" "0100" "11" "0" "0010" "01";
  CHECK(bits == expected.size());
  CHECK(bit_string(w) == expected);

  BitReader r = reader_for(w);
  CdefParams back;
  CHECK(read_cdef_params(r, ctx, &back, &error));
  CHECK(error.empty());
  CHECK(back.frame_enable);
  CHECK(back.damping_minus_3 == 0);
  CHECK(back.num_strengths == 2);
  CHECK(!back.on_skip_txfm_frame_enable);
  CHECK(back.set[0].y_pri == 0 && back.set[0].uv_pri == 0);
  CHECK(back.set[1].y_pri == 4);
  CHECK(back.set[1].y_sec == 4);
  CHECK(back.set[1].uv_pri == 2);
  CHECK(back.set[1].uv_sec == 1);
  CHECK(r.bits_left() == 0U);

  // CodedLossless suppresses the whole structure.
  CdefHeaderContext lossless = ctx;
  lossless.coded_lossless = true;
  BitWriter w2;
  CHECK(write_cdef_params(w2, p, lossless, &error) == 0U);
  CdefParams back2;
  BitReader r2(w2.bytes(), 1U);
  CHECK(read_cdef_params(r2, lossless, &back2, &error));
  CHECK(!back2.frame_enable);

  // Non-single-picture: frame_enable is coded.
  CdefHeaderContext multi = ctx;
  multi.single_picture_header_flag = false;
  CdefParams off;
  off.frame_enable = false;
  BitWriter w3;
  CHECK(write_cdef_params(w3, off, multi, &error) == 1U);
  CHECK(bit_string(w3) == "0");
}

// ---------------------------------------------------------------------------
// §5.20.10.1
// ---------------------------------------------------------------------------

void test_cdef_tile() {
  CdefTileState enc;
  std::string error;
  CHECK(cdef_tile_init(&enc, /*mi_cols=*/32, /*mi_rows=*/16, 4, 4, &error));
  std::vector<std::uint8_t> buffer(4096);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  encoder.disable_cdf_update = 1;
  CHECK(cdef_write_block(encoder, enc, 0, 0, 0, 0, 0, &error));
  CHECK(cdef_write_block(encoder, enc, 0, 0, 0, 16, 1, &error));
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk);
  CHECK(error.empty());

  CdefTileState dec;
  CHECK(cdef_tile_init(&dec, 32, 16, 4, 4, &error));
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
        av2::Status::kOk);
  decoder.disable_cdf_update = 1;
  int idx0 = -1;
  int idx1 = -1;
  CHECK(cdef_read_block(decoder, dec, 0, 0, 0, 0, &idx0, &error));
  CHECK(cdef_read_block(decoder, dec, 0, 0, 0, 16, &idx1, &error));
  CHECK(error.empty());
  CHECK(idx0 == 0);
  CHECK(idx1 == 1);
  CHECK(decoder.exit_symbol() == av2::Status::kOk);

  // Regression pin.  The symbol sequence is fully determined by the §8.3.1
  // context (left neighbour idx 0 gives ctx 2 for the second block).
  const std::string got = hex_string(
      std::span<const std::uint8_t>(buffer.data(), payload));
  CHECK(got == "e30000");

  // CdefStrengths == 1 codes no index at all.
  CdefTileState one;
  CHECK(cdef_tile_init(&one, 32, 16, 1, 4, &error));
  av2::Encoder e2;
  CHECK(e2.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  e2.disable_cdf_update = 1;
  CHECK(cdef_write_block(e2, one, 0, 0, 0, 0, 7, &error) == false);
  CHECK(cdef_write_block(e2, one, 0, 0, 0, 0, 0, &error));
  CHECK(one.grid.at(0, 0) == 0);
  CHECK(e2.symbol_count == 0U);
}

// ---------------------------------------------------------------------------
// §5.18.7.12 / §5.20.10.2
// ---------------------------------------------------------------------------

void test_ccso_header() {
  CcsoHeaderContext ctx;
  ctx.enable_ccso = true;
  CcsoParams p;
  p.plane[0].enabled = true;
  p.plane[0].bo_only = true;
  p.plane[0].scale_idx = 1;
  p.plane[0].max_band_log2 = 2;
  p.plane[0].offset_idx = {1, 2, 3, 4};

  BitWriter w;
  std::string error;
  const std::size_t bits = write_ccso_params(w, p, ctx, &error);
  CHECK(error.empty());
  CHECK(p.ccso_luma_size_log2 == 6);
  // plane0=1 | bo_only=1 | scale=01 | band=010 | tu(7) offsets | plane1=0 |
  // plane2=0
  const std::string expected = "1" "1" "01" "010" "10" "110" "1110" "11110"
                               "0" "0";
  CHECK(bits == expected.size());
  CHECK(bit_string(w) == expected);

  BitReader r = reader_for(w);
  CcsoParams back;
  CHECK(read_ccso_params(r, ctx, &back, &error));
  CHECK(error.empty());
  CHECK(back.plane[0].enabled && back.plane[0].bo_only);
  CHECK(back.plane[0].scale_idx == 1);
  CHECK(back.plane[0].max_band_log2 == 2);
  CHECK(back.plane[0].offset_idx.size() == 4U);
  CHECK(back.plane[0].offset_idx[0] == 1);
  CHECK(back.plane[0].offset_idx[3] == 4);
  CHECK(r.bits_left() == 0U);

  restoration::CcsoParams rp;
  CHECK(ccso_fill_offsets(back.plane[0], &rp, &error));
  CHECK(error.empty());
  CHECK(rp.offsets[restoration::ccso_offset_index(0, 0, 0)] == 2);   // 1*2
  CHECK(rp.offsets[restoration::ccso_offset_index(1, 0, 0)] == -2);  // -1*2
  CHECK(rp.offsets[restoration::ccso_offset_index(2, 0, 0)] == 6);   // 3*2
  CHECK(rp.offsets[restoration::ccso_offset_index(3, 0, 0)] == -6);  // -3*2
}

void test_ccso_tile() {
  CcsoHeaderContext ctx;
  ctx.enable_ccso = true;
  CcsoParams params;
  params.frame_flag = true;
  params.plane[0].enabled = true;
  params.plane[0].bo_only = true;
  params.plane[0].max_band_log2 = 2;
  params.plane[0].offset_idx = {0, 0, 0, 0};

  CcsoTileState enc;
  std::string error;
  CHECK(ccso_tile_init(&enc, params, /*mi_cols=*/32, /*mi_rows=*/16, 1, 1,
                       &error));
  CHECK(enc.blk_cols[0] == 2);
  CHECK(enc.blk_rows[0] == 1);
  std::vector<std::uint8_t> buffer(4096);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  encoder.disable_cdf_update = 1;
  const int blk0[3] = {1, 0, 0};
  const int blk1[3] = {0, 0, 0};
  CHECK(ccso_write_block(encoder, enc, 0, 0, blk0, &error));
  CHECK(ccso_write_block(encoder, enc, 0, 16, blk1, &error));
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk);
  CHECK(error.empty());

  CcsoTileState dec;
  CHECK(ccso_tile_init(&dec, params, 32, 16, 1, 1, &error));
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
        av2::Status::kOk);
  decoder.disable_cdf_update = 1;
  int d0[3] = {-1, -1, -1};
  int d1[3] = {-1, -1, -1};
  CHECK(ccso_read_block(decoder, dec, 0, 0, d0, &error));
  CHECK(ccso_read_block(decoder, dec, 0, 16, d1, &error));
  CHECK(error.empty());
  CHECK(d0[0] == 1);
  CHECK(d1[0] == 0);
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
  const std::string ccso_pin =
      hex_string(std::span<const std::uint8_t>(buffer.data(), payload));
  CHECK(ccso_pin == "980000");
}

// ---------------------------------------------------------------------------
// §5.18.7.11 / §5.20.10.4–§5.20.10.6
// ---------------------------------------------------------------------------

LrHeaderContext lr_context() {
  LrHeaderContext ctx;
  ctx.enable_restoration = true;
  ctx.sb_size = 64;
  ctx.base_q_idx = 90;
  return ctx;
}

LrParams lr_params_none() {
  LrParams p;
  p.frame_type[0] = kRestoreNone;
  p.frame_type[1] = kRestoreNone;
  p.frame_type[2] = kRestoreNone;
  return p;
}

void test_lr_header_simple() {
  std::string error;
  {
    LrParams p = lr_params_none();
    BitWriter w;
    const std::size_t bits =
        write_lr_params(w, p, lr_context(), &error);
    CHECK(error.empty());
    // ns(2) for each plane, all RESTORE_NONE.
    CHECK(bits == 3U);
    CHECK(bit_string(w) == "000");
    LrParams back;
    BitReader r = reader_for(w);
    CHECK(read_lr_params(r, lr_context(), &back, &error));
    CHECK(error.empty());
    CHECK(back.frame_type[0] == kRestoreNone);
    CHECK(r.bits_left() == 0U);
  }
  {
    LrParams p = lr_params_none();
    p.frame_type[0] = kRestoreWienerNs;
    p.luma_shift = 3;  // 64x64 units
    BitWriter w;
    const std::size_t bits =
        write_lr_params(w, p, lr_context(), &error);
    CHECK(error.empty());
    // plane0 ns(2)=1, plane1/2 ns(2)=0, luma half=0 max=0 quarter=0.
    CHECK(bits == 7U);
    CHECK(bit_string(w) == "1000000");
    LrParams back;
    BitReader r = reader_for(w);
    CHECK(read_lr_params(r, lr_context(), &back, &error));
    CHECK(back.frame_type[0] == kRestoreWienerNs);
    CHECK(back.luma_shift == 3);
    CHECK(r.bits_left() == 0U);
  }
  {
    // Frame filters with two classes, both merged.  c=0 match is forced to 0
    // (n == 1, no bits); c=1 match 1 maps to subexp value 0 (ns(2) = 0).
    LrParams p = lr_params_none();
    p.frame_type[0] = kRestoreWienerNs;
    p.frame_filters_on[0] = true;
    p.num_filter_classes = 2;
    p.luma_shift = 3;
    p.frame_filter[0][0].match_index = 0;
    p.frame_filter[0][0].merged = true;
    p.frame_filter[0][1].match_index = 1;
    p.frame_filter[0][1].merged = true;
    BitWriter w;
    const std::size_t bits =
        write_lr_params(w, p, lr_context(), &error);
    CHECK(error.empty());
    // plane bits 1 0 0 | luma size 000 | c0 merged 1 | c1 match 0 + merged 1
    CHECK(bits == 13U);
    CHECK(bit_string(w) == "1100100000011");
    LrParams back;
    BitReader r = reader_for(w);
    CHECK(read_lr_params(r, lr_context(), &back, &error));
    CHECK(error.empty());
    CHECK(back.num_filter_classes == 2);
    CHECK(back.frame_filters_on[0]);
    CHECK(back.frame_filter[0][0].merged);
    CHECK(back.frame_filter[0][1].merged);
    CHECK(back.frame_filter[0][1].match_index == 1);
    CHECK(r.bits_left() == 0U);
  }
}

LrState lr_state(int mi_cols, int mi_rows, const LrParams& params) {
  LrState state;
  std::string error;
  const bool ok =
      lr_state_init(&state, params, mi_cols, mi_rows, 1, 1, 64, &error);
  CHECK(ok);
  CHECK(error.empty());
  return state;
}

void test_lr_header_coefficients() {
  LrParams p = lr_params_none();
  p.frame_type[0] = kRestoreWienerNs;
  p.frame_filters_on[0] = true;
  p.num_filter_classes = 4;
  p.luma_shift = 3;
  p.class_map.num_classes = 4;
  p.class_map.has_subclass_lookup = true;
  p.class_map.subclass_lookup = {0, 1, 2, 3};
  p.class_map.filter_class_grid.assign(32 * 32, 0);
  constexpr std::int8_t kPresent[4][18] = {
      {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
  };
  for (int c = 0; c < 4; ++c) {
    p.frame_filter[0][c].match_index = c;
    p.frame_filter[0][c].merged = false;
    p.frame_filter[0][c].subset = c;
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      const int min = restoration::kWienerNsTapsMin[0][j];
      const int k = restoration::kWienerNsTapsK[0][j];
      p.frame_filter[0][c].coeffs[j] =
          kPresent[c][j] != 0
              ? static_cast<std::int16_t>(min + ((j * 7 + c * 5) % (1 << k)))
              : 0;
    }
  }
  std::string error;
  BitWriter w;
  const std::size_t bits = write_lr_params(w, p, lr_context(), &error);
  CHECK(error.empty());
  CHECK(bits > 9U);
  // Regression pin for the 4-class frame-filter syntax (match/subset/subexp
  // coefficient bits, 265 bits before byte padding).
  CHECK(bits == 265U);
  CHECK(hex_string(w.bytes()) ==
        "d81c77f08dbc794b2faf5bd76d2974ffce7bebde7fdbca58fbcb2e7fdf39eb9c7a00");

  LrParams back;
  BitReader r = reader_for(w);
  CHECK(read_lr_params(r, lr_context(), &back, &error));
  CHECK(error.empty());
  CHECK(back.num_filter_classes == 4);
  CHECK(back.frame_filters_on[0]);
  for (int c = 0; c < 4; ++c) {
    CHECK(back.frame_filter[0][c].match_index == c);
    CHECK(!back.frame_filter[0][c].merged);
    CHECK(back.frame_filter[0][c].subset == c);
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      CHECK(back.frame_filter[0][c].coeffs[j] ==
            p.frame_filter[0][c].coeffs[j]);
    }
  }
  CHECK(r.bits_left() == 0U);
}

void test_lr_class_map_required() {
  LrParams p = lr_params_none();
  p.frame_type[0] = kRestoreWienerNs;
  p.frame_filters_on[0] = true;
  p.num_filter_classes = 2;
  p.luma_shift = 3;
  p.class_map.num_classes = 2;
  // No subclass lookup.
  LrState state;
  std::string error;
  CHECK(!lr_state_init(&state, p, 32, 32, 1, 1, 64, &error));
  CHECK(error.find("Pc_Wiener_Sub_Classify2") != std::string::npos);
}

void test_lr_unit_range() {
  LrParams p = lr_params_none();
  p.frame_type[0] = kRestoreWienerNs;
  p.luma_shift = 3;  // 64x64 luma units
  LrState state = lr_state(32, 32, p);
  std::string error;
  LrUnitRange r;
  CHECK(lr_unit_range(state, 0, 0, 0, 16, 16, 0, 0, &r, &error));
  CHECK(r.row_begin == 0 && r.row_end == 1);
  CHECK(r.col_begin == 0 && r.col_end == 1);
  CHECK(lr_unit_range(state, 0, 0, 16, 16, 16, 0, 0, &r, &error));
  CHECK(r.row_begin == 0 && r.row_end == 1);
  CHECK(r.col_begin == 1 && r.col_end == 2);
  CHECK(lr_unit_range(state, 0, 16, 0, 16, 16, 0, 0, &r, &error));
  CHECK(r.row_begin == 1 && r.row_end == 2);
  CHECK(r.col_begin == 0 && r.col_end == 1);
  // Chroma is RESTORE_NONE.
  CHECK(lr_unit_range(state, 1, 0, 0, 16, 16, 0, 0, &r, &error));
  CHECK(r.row_begin == 0 && r.row_end == 0);
  CHECK(lr_block_unit_count(state, 0, 0, 16, 16, 0, 0) == 1);
}

void test_lr_tile() {
  LrParams p = lr_params_none();
  p.frame_type[0] = kRestoreWienerNs;
  p.luma_shift = 3;
  LrState enc = lr_state(32, 32, p);
  std::string error;

  constexpr int kPresent[4][18] = {
      {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
  };
  auto default_decision = [](int type, int subset) {
    LrUnitDecision d;
    d.type = type;
    d.subset = subset;
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      const int min = restoration::kWienerNsTapsMin[0][j];
      d.coeffs[j] = kPresent[subset][j] != 0
                        ? static_cast<std::int16_t>(min + 1)
                        : 0;
    }
    return d;
  };

  std::vector<std::uint8_t> buffer(8192);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  encoder.disable_cdf_update = 1;
  const std::vector<LrUnitDecision> sb00 = {
      default_decision(kRestoreWienerNs, 0)};
  const std::vector<LrUnitDecision> sb01 = {
      LrUnitDecision{/*type=*/kRestoreNone}};
  const std::vector<LrUnitDecision> sb10 = {
      default_decision(kRestoreWienerNs, 1)};
  const std::vector<LrUnitDecision> sb11 = {
      default_decision(kRestoreWienerNs, 0)};
  CHECK(lr_write_block(encoder, enc, 0, 0, 16, 16, 0, 0, sb00, &error));
  CHECK(lr_write_block(encoder, enc, 0, 16, 16, 16, 0, 0, sb01, &error));
  CHECK(lr_write_block(encoder, enc, 16, 0, 16, 16, 0, 0, sb10, &error));
  CHECK(lr_write_block(encoder, enc, 16, 16, 16, 16, 0, 0, sb11, &error));
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk);
  CHECK(error.empty());

  LrState dec = lr_state(32, 32, p);
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
        av2::Status::kOk);
  decoder.disable_cdf_update = 1;
  std::vector<LrUnitDecision> units;
  CHECK(lr_read_block(decoder, dec, 0, 0, 16, 16, 0, 0, &units, &error));
  CHECK(units.size() == 1U);
  CHECK(units[0].type == kRestoreWienerNs);
  units.clear();
  CHECK(lr_read_block(decoder, dec, 0, 16, 16, 16, 0, 0, &units, &error));
  CHECK(units.size() == 1U);
  CHECK(units[0].type == kRestoreNone);
  units.clear();
  CHECK(lr_read_block(decoder, dec, 16, 0, 16, 16, 0, 0, &units, &error));
  CHECK(units.size() == 1U);
  CHECK(units[0].type == kRestoreWienerNs);
  CHECK(units[0].subset == 1);
  units.clear();
  CHECK(lr_read_block(decoder, dec, 16, 16, 16, 16, 0, 0, &units, &error));
  CHECK(units.size() == 1U);
  CHECK(units[0].type == kRestoreWienerNs);
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
  CHECK(error.empty());

  // Compare the reconstructed grids with the encoder's decisions.
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      const std::size_t i =
          static_cast<std::size_t>(r) * 2U + static_cast<std::size_t>(c);
      CHECK(dec.unit_type[0][i] == enc.unit_type[0][i]);
      for (int j = 0; j < restoration::kWienerNsLumaCoeffs; ++j) {
        CHECK(dec.unit_coeffs[0][i * restoration::kWienerNsCoeffStride +
                                static_cast<std::size_t>(j)] ==
              enc.unit_coeffs[0][i * restoration::kWienerNsCoeffStride +
                                  static_cast<std::size_t>(j)]);
      }
    }
  }
  const std::string lr_pin =
      hex_string(std::span<const std::uint8_t>(buffer.data(), payload));
  CHECK(lr_pin == "549687b76ad7e5d38b49b4f2237cd0004000");

  // lr_build_decisions() reconstructs the same decisions from the unit grids.
  std::vector<LrUnitDecision> rebuilt;
  CHECK(lr_build_decisions(enc, 0, 0, 16, 16, 0, 0, &rebuilt, &error));
  CHECK(rebuilt.size() == 1U);
  CHECK(rebuilt[0].type == kRestoreWienerNs);
  CHECK(rebuilt[0].subset == 0);
  CHECK(lr_build_decisions(enc, 16, 0, 16, 16, 0, 0, &rebuilt, &error));
  CHECK(rebuilt.size() == 1U);
  CHECK(rebuilt[0].subset == 1);
}

void test_lr_tile_chroma_and_merged() {
  std::string error;
  {
    // Luma 64x64 units + chroma 32x32 units, both WIENER_NONSEP, with a
    // symmetric chroma filter (uv_sym).
    LrParams p = lr_params_none();
    p.frame_type[0] = kRestoreWienerNs;
    p.frame_type[1] = kRestoreWienerNs;
    p.frame_type[2] = kRestoreWienerNs;
    p.luma_shift = 3;
    p.chroma_shift = 3;
    LrState enc = lr_state(16, 16, p);

    LrUnitDecision luma;
    luma.type = kRestoreWienerNs;
    luma.subset = 2;
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      const int min = restoration::kWienerNsTapsMin[0][j];
      const int k = restoration::kWienerNsTapsK[0][j];
      luma.coeffs[j] = j < 12
                           ? static_cast<std::int16_t>(
                                 min + ((j * 3 + 1) % (1 << k)))
                           : 0;
    }
    LrUnitDecision chroma;
    chroma.type = kRestoreWienerNs;
    chroma.subset = 2;
    chroma.uv_sym = true;
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      const int min = restoration::kWienerNsTapsMin[1][j];
      const int k = restoration::kWienerNsTapsK[1][j];
      int v = min + ((j * 5 + 2) % (1 << k));
      if (j >= restoration::kWienerNsShortCoeffs && (j & 1) == 0) {
        const int pair = min + (((j / 2) * 7 + 3) % (1 << k));
        chroma.coeffs[j] = static_cast<std::int16_t>(pair);
        chroma.coeffs[j + 1] = static_cast<std::int16_t>(pair);
        ++j;
      } else {
        chroma.coeffs[j] = static_cast<std::int16_t>(v);
      }
    }

    std::vector<std::uint8_t> buffer(8192);
    av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
          av2::Status::kOk);
    encoder.disable_cdf_update = 1;
    LrUnitDecision chroma_v;
    chroma_v.type = kRestoreWienerNs;
    chroma_v.subset = 0;
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      const int min = restoration::kWienerNsTapsMin[1][j];
      const int k = restoration::kWienerNsTapsK[1][j];
      chroma_v.coeffs[j] = j < 6
                               ? static_cast<std::int16_t>(
                                     min + ((j * 2 + 1) % (1 << k)))
                               : 0;
    }
    const std::vector<LrUnitDecision> decisions = {luma, chroma, chroma_v};
    CHECK(lr_write_block(encoder, enc, 0, 0, 16, 16, 0, 0, decisions,
                         &error));
    std::size_t payload = 0;
    CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk);
    CHECK(error.empty());

    LrState dec = lr_state(16, 16, p);
    av2::Decoder decoder;
    CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
          av2::Status::kOk);
    decoder.disable_cdf_update = 1;
    std::vector<LrUnitDecision> back;
    CHECK(lr_read_block(decoder, dec, 0, 0, 16, 16, 0, 0, &back, &error));
    CHECK(decoder.exit_symbol() == av2::Status::kOk);
    CHECK(error.empty());
    CHECK(back.size() == 3U);
    CHECK(back[0].type == kRestoreWienerNs);
    CHECK(back[0].subset == 2);
    CHECK(back[1].type == kRestoreWienerNs);
    CHECK(back[1].subset == 2);
    CHECK(back[1].uv_sym);
    for (int j = 0; j < restoration::kWienerNsLumaCoeffs; ++j) {
      CHECK(back[0].coeffs[j] == luma.coeffs[j]);
    }
    for (int j = 0; j < restoration::kWienerNsChromaCoeffs; ++j) {
      CHECK(back[1].coeffs[j] == chroma.coeffs[j]);
      CHECK(back[2].coeffs[j] == chroma_v.coeffs[j]);
    }
  }
  {
    // Bank reuse: unit 0 writes a bank; unit 1 is merged and must reproduce
    // those coefficients without any coefficient bits.
    LrParams p = lr_params_none();
    p.frame_type[0] = kRestoreWienerNs;
    p.luma_shift = 3;
    LrState enc = lr_state(32, 32, p);
    LrUnitDecision first;
    first.type = kRestoreWienerNs;
    first.subset = 0;
    for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
      const int min = restoration::kWienerNsTapsMin[0][j];
      const int k = restoration::kWienerNsTapsK[0][j];
      first.coeffs[j] =
          static_cast<std::int16_t>(min + ((j * 11 + 5) % (1 << k)));
    }
    LrUnitDecision merged;
    merged.type = kRestoreWienerNs;
    merged.merged = true;

    std::vector<std::uint8_t> buffer(8192);
    av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
          av2::Status::kOk);
    encoder.disable_cdf_update = 1;
    CHECK(lr_write_block(encoder, enc, 0, 0, 16, 16, 0, 0, {first}, &error));
    CHECK(lr_write_block(encoder, enc, 0, 16, 16, 16, 0, 0, {merged}, &error));
    std::size_t payload = 0;
    CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk);
    CHECK(error.empty());

    LrState dec = lr_state(32, 32, p);
    av2::Decoder decoder;
    CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
          av2::Status::kOk);
    decoder.disable_cdf_update = 1;
    std::vector<LrUnitDecision> back;
    CHECK(lr_read_block(decoder, dec, 0, 0, 16, 16, 0, 0, &back, &error));
    CHECK(lr_read_block(decoder, dec, 0, 16, 16, 16, 0, 0, &back, &error));
    CHECK(decoder.exit_symbol() == av2::Status::kOk);
    CHECK(error.empty());
    const std::size_t unit1 =
        static_cast<std::size_t>(0) * 2U + 1U;  // (row 0, col 1)
    for (int j = 0; j < restoration::kWienerNsLumaCoeffs; ++j) {
      CHECK(enc.unit_coeffs[0][unit1 * restoration::kWienerNsCoeffStride +
                              static_cast<std::size_t>(j)] ==
            dec.unit_coeffs[0][unit1 * restoration::kWienerNsCoeffStride +
                                static_cast<std::size_t>(j)]);
    }
  }
}

}  // namespace

int main() {
  test_sequence_filter_config();
  test_cdef_header();
  test_cdef_tile();
  test_ccso_header();
  test_ccso_tile();
  test_lr_header_simple();
  test_lr_header_coefficients();
  test_lr_class_map_required();
  test_lr_unit_range();
  test_lr_tile();
  test_lr_tile_chroma_and_merged();
  return test::finish("filters_stream_test");
}
