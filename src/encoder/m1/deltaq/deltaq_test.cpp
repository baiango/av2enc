// Unit tests for src/encoder/m1/deltaq/ (delta-Q + segmentation bitstream).
//
// Covered:
//   * §9.3 CDF tables and the four COEFF_CDF_Q_CTXS slices from raw base_q_idx
//   * frame-header syntax write/read, bit counts, pinned bit goldens
//   * §5.20.5.11 delta_q_abs (including the DELTA_Q_SMALL escape)
//   * CurrentQIndex reset per tile, ReadDeltas per superblock, skip gating
//   * §5.20.5.7/.8 intra/inter segment-id syntax, negation mapping, contexts
//   * rc::DeltaQPlan / rc::SegmentationMap integration
//   * one pinned range-coded block-stream golden
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/m0_cdf_tables.h"
#include "encoder/m1/deltaq/deltaq.h"
#include "encoder/rc/rc.h"
#include "entropy/av2/av2_entropy.h"

namespace {

using namespace av2enc::deltaq;
using av2enc::BitWriter;
namespace rc = av2enc::rc;
namespace av2 = av2enc::av2;

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++g_checks;                                                              \
    const auto va = (a);                                                     \
    const auto vb = (b);                                                     \
    if (!(va == vb)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, \
                   __LINE__, #a, #b, static_cast<long long>(va),              \
                   static_cast<long long>(vb));                              \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> pack_bits(const std::string& bits) {
  std::vector<std::uint8_t> out((bits.size() + 7U) / 8U, 0U);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    if (bits[i] == '1') {
      out[i >> 3U] |= static_cast<std::uint8_t>(0x80U >> (i & 7U));
    }
  }
  return out;
}

std::vector<std::uint8_t> finish(BitWriter& writer) {
  writer.align_to_byte();
  const std::span<const std::uint8_t> bytes = writer.bytes();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2U);
  for (const std::uint8_t byte : bytes) {
    out.push_back(kHex[byte >> 4U]);
    out.push_back(kHex[byte & 0x0fU]);
  }
  return out;
}

rc::SegmentationParams two_segment_alt_q(int d0, int d1) {
  rc::SegmentationParams params;
  std::string error;
  CHECK(rc::set_segment_alt_q(params, 0, d0, &error));
  CHECK(rc::set_segment_alt_q(params, 1, d1, &error));
  params.update_map = true;
  return params;
}

bool same_features(const rc::SegmentFeatures& a,
                   const rc::SegmentFeatures& b) {
  return a.enabled == b.enabled && a.data == b.data;
}

bool same_segmentation(const rc::SegmentationParams& a,
                       const rc::SegmentationParams& b) {
  if (a.enabled != b.enabled || a.update_map != b.update_map ||
      a.temporal_update != b.temporal_update) {
    return false;
  }
  for (int i = 0; i < rc::kMaxSegments; ++i) {
    if (!same_features(a.segments[static_cast<std::size_t>(i)],
                       b.segments[static_cast<std::size_t>(i)])) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// CDF tables and coefficient slices
// ---------------------------------------------------------------------------

void test_block_cdfs() {
  BlockCdfs cdfs;
  cdfs.init();
  CHECK(std::memcmp(cdfs.delta_q, cdf::kDeltaQCdf, sizeof(cdfs.delta_q)) == 0);
  CHECK(std::memcmp(cdfs.segment_id, cdf::kSegmentIdCdf,
                    sizeof(cdfs.segment_id)) == 0);
  CHECK(std::memcmp(cdfs.segment_id_ext, cdf::kSegmentIdExtCdf,
                    sizeof(cdfs.segment_id_ext)) == 0);
  CHECK(std::memcmp(cdfs.seg_id_ext_flag, cdf::kSegIdExtFlagCdf,
                    sizeof(cdfs.seg_id_ext_flag)) == 0);
  CHECK(std::memcmp(cdfs.segment_id_predicted,
                    cdf::kSegmentIdPredictedCdf,
                    sizeof(cdfs.segment_id_predicted)) == 0);

  // Pinned values from spec/tables/default_cdf.json (clause 9.3).
  CHECK_EQ(cdfs.delta_q[0], 16594);
  CHECK_EQ(cdfs.delta_q[6], 30613);
  CHECK_EQ(cdfs.delta_q[7], 56);
  CHECK_EQ(cdfs.delta_q[8], 0);
  CHECK_EQ(cdfs.segment_id[0][0], 5622);
  CHECK_EQ(cdfs.segment_id[2][6], 32679);
}

void test_coeff_slices() {
  CHECK_EQ(coeff_cdf_qctx(0), 0);
  CHECK_EQ(coeff_cdf_qctx(90), 0);
  CHECK_EQ(coeff_cdf_qctx(91), 1);
  CHECK_EQ(coeff_cdf_qctx(140), 1);
  CHECK_EQ(coeff_cdf_qctx(141), 2);
  CHECK_EQ(coeff_cdf_qctx(190), 2);
  CHECK_EQ(coeff_cdf_qctx(191), 3);
  CHECK_EQ(coeff_cdf_qctx(255), 3);

  // Slice 0 must reproduce the M0 tables byte for byte.  That proves the
  // generator sliced the normative array correctly.
  const CoeffCdfSlice slice0 = coeff_cdf_slice(90);
  CHECK(std::memcmp(slice0.txb_skip, av2enc::m0::kTxbSkipCdf,
                    sizeof(av2enc::m0::kTxbSkipCdf)) == 0);
  CHECK(std::memcmp(slice0.v_txb_skip, av2enc::m0::kVTxbSkipCdf,
                    sizeof(av2enc::m0::kVTxbSkipCdf)) == 0);
  CHECK(std::memcmp(slice0.eob_pt_64, av2enc::m0::kEobPt64Cdf,
                    sizeof(av2enc::m0::kEobPt64Cdf)) == 0);
  CHECK(std::memcmp(slice0.eob_pt_128, av2enc::m0::kEobPt128Cdf,
                    sizeof(av2enc::m0::kEobPt128Cdf)) == 0);
  CHECK(std::memcmp(slice0.eob_pt_256, av2enc::m0::kEobPt256Cdf,
                    sizeof(av2enc::m0::kEobPt256Cdf)) == 0);
  CHECK(std::memcmp(slice0.eob_pt_512, av2enc::m0::kEobPt512Cdf,
                    sizeof(av2enc::m0::kEobPt512Cdf)) == 0);
  CHECK(std::memcmp(slice0.eob_pt_1024, av2enc::m0::kEobPt1024Cdf,
                    sizeof(av2enc::m0::kEobPt1024Cdf)) == 0);
  CHECK(std::memcmp(slice0.eob_extra, av2enc::m0::kEobExtraCdf,
                    sizeof(av2enc::m0::kEobExtraCdf)) == 0);
  CHECK(std::memcmp(slice0.dc_sign, av2enc::m0::kDcSignCdf,
                    sizeof(av2enc::m0::kDcSignCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base, av2enc::m0::kCoeffBaseCdf,
                    sizeof(av2enc::m0::kCoeffBaseCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base_lf, av2enc::m0::kCoeffBaseLfCdf,
                    sizeof(av2enc::m0::kCoeffBaseLfCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base_eob, av2enc::m0::kCoeffBaseEobCdf,
                    sizeof(av2enc::m0::kCoeffBaseEobCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base_lf_eob,
                    av2enc::m0::kCoeffBaseLfEobCdf,
                    sizeof(av2enc::m0::kCoeffBaseLfEobCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_br, av2enc::m0::kCoeffBrCdf,
                    sizeof(av2enc::m0::kCoeffBrCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_br_lf, av2enc::m0::kCoeffBrLfCdf,
                    sizeof(av2enc::m0::kCoeffBrLfCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base_uv, av2enc::m0::kCoeffBaseUvCdf,
                    sizeof(av2enc::m0::kCoeffBaseUvCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base_lf_uv, av2enc::m0::kCoeffBaseLfUvCdf,
                    sizeof(av2enc::m0::kCoeffBaseLfUvCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base_eob_uv, av2enc::m0::kCoeffBaseEobUvCdf,
                    sizeof(av2enc::m0::kCoeffBaseEobUvCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_base_lf_eob_uv,
                    av2enc::m0::kCoeffBaseLfEobUvCdf,
                    sizeof(av2enc::m0::kCoeffBaseLfEobUvCdf)) == 0);
  CHECK(std::memcmp(slice0.coeff_br_uv, av2enc::m0::kCoeffBrUvCdf,
                    sizeof(av2enc::m0::kCoeffBrUvCdf)) == 0);
  // Eob_Pt_16/32 are not in the M0 header (M1 declares slice 0 locally), so
  // pin the §9.3 values directly.
  CHECK_EQ(reinterpret_cast<const std::uint16_t*>(slice0.eob_pt_16)[0], 1946);
  CHECK_EQ(reinterpret_cast<const std::uint16_t*>(slice0.eob_pt_32)[0], 1203);

  const CoeffCdfSlice slice1 = coeff_cdf_slice(91);
  const CoeffCdfSlice slice3 = coeff_cdf_slice(255);
  CHECK(slice1.coeff_base != slice0.coeff_base);
  CHECK(slice3.coeff_base != slice1.coeff_base);
  CHECK(std::memcmp(slice1.coeff_base, slice3.coeff_base,
                    sizeof(av2enc::m0::kCoeffBaseCdf)) != 0);
  // Pinned first elements from the JSON tables (clause 9.3).
  CHECK_EQ(reinterpret_cast<const std::uint16_t*>(slice1.txb_skip)[0], 30296);
  CHECK_EQ(reinterpret_cast<const std::uint16_t*>(slice3.coeff_base)[0], 18405);

  // Slice selection depends on base_q_idx only: a delta-Q context cannot
  // change it.  There is no other input to the selector, and this checks the
  // effective qindex is irrelevant even when delta-Q is active.
  TileQp qp;
  std::string error;
  rc::DeltaQParams delta;
  delta.present = true;
  delta.res = 3;
  CHECK(begin_tile(qp, 90, delta, &error));
  CHECK(qp.current_q_index == 90);
  CHECK_EQ(coeff_cdf_qctx(qp.current_q_index), 0);
  CHECK_EQ(coeff_cdf_qctx(qp.base_q_idx), 0);
}

// ---------------------------------------------------------------------------
// Frame header
// ---------------------------------------------------------------------------

void test_frame_header_disabled_golden() {
  FrameQpHeader header;
  header.primary_ref_is_none = true;
  BitWriter writer;
  std::string error;
  const std::size_t bits = write_segmentation_and_delta_q(
      writer, rc::SegmentationParams{}, rc::DeltaQParams{}, 90, header, &error);
  CHECK(error.empty());
  CHECK_EQ(bits, 3U);
  CHECK_EQ(writer.bit_size(), 3U);
  const std::vector<std::uint8_t> bytes = finish(writer);
  CHECK(hex(bytes) == "00");

  BitReader reader(bytes);
  rc::SegmentationParams seg;
  rc::DeltaQParams delta;
  bool using_qmatrix = true;
  CHECK(read_segmentation_and_delta_q(reader, 90, header, seg, delta,
                                      using_qmatrix, &error));
  CHECK(error.empty());
  CHECK(!using_qmatrix);
  CHECK(!seg.enabled);
  CHECK(!delta.present);
  CHECK_EQ(delta.res, 0);
  CHECK_EQ(reader.bit_position(), 3U);
}

void test_frame_header_key_golden() {
  // seg 0: ALT_Q = +10; delta_q present, res = 2; key frame.
  //
  //   1                      segmentation_enabled
  //   1 0000001010 0 0       seg 0: ALT_Q on + su(10)=10, SKIP off, GMV off
  //   000 000 000 ...        seg 1..7 (7 * 3 zero bits)
  //   0                      using_qmatrix
  //   1                      delta_q_present
  //   10                     delta_q_res = 2
  //
  // Total: 1 + (1 + 10 + 1 + 1) + 21 + 1 + 1 + 2 = 39 bits.
  const std::string expected_bits =
      "1" "1" "0000001010" "0" "0" + std::string(21, '0') + "0" "1" "10";
  CHECK_EQ(expected_bits.size(), 39U);

  rc::SegmentationParams seg =
      two_segment_alt_q(/*d0=*/10, /*d1=*/0);
  // two_segment_alt_q enables segment 1 too; clear it so the golden shows
  // exactly one active feature.
  seg.segments[1] = rc::SegmentFeatures{};
  rc::DeltaQParams delta;
  delta.present = true;
  delta.res = 2;
  FrameQpHeader header;
  header.primary_ref_is_none = true;

  BitWriter writer;
  std::string error;
  const std::size_t bits = write_segmentation_and_delta_q(
      writer, seg, delta, 90, header, &error);
  CHECK(error.empty());
  CHECK_EQ(bits, 39U);
  const std::vector<std::uint8_t> bytes = finish(writer);
  CHECK(bytes == pack_bits(expected_bits));

  BitReader reader(bytes);
  rc::SegmentationParams read_seg;
  rc::DeltaQParams read_delta;
  bool using_qmatrix = true;
  CHECK(read_segmentation_and_delta_q(reader, 90, header, read_seg,
                                      read_delta, using_qmatrix, &error));
  CHECK(error.empty());
  CHECK(read_seg.enabled);
  CHECK(read_seg.segments[0].enabled[rc::kSegLvlAltQ]);
  CHECK_EQ(read_seg.segments[0].data[rc::kSegLvlAltQ], 10);
  CHECK(!read_seg.segments[1].enabled[rc::kSegLvlAltQ]);
  CHECK(!using_qmatrix);
  CHECK(read_delta.present);
  CHECK_EQ(read_delta.res, 2);
  CHECK_EQ(reader.bit_position(), 39U);

  // Key frame inference: update_map = 1, temporal_update = 0.
  FrameSegmentation derived;
  CHECK(derive_frame_segmentation(read_seg, read_delta, 90, header, derived,
                                  &error));
  CHECK(derived.enabled);
  CHECK(derived.update_map);
  CHECK(!derived.temporal_update);
  CHECK_EQ(derived.last_active_seg_id, 0);
  CHECK(!derived.seg_id_pre_skip);
  CHECK(!derived.has_lossless_segment);
}

void test_frame_header_inter_bits() {
  rc::SegmentationParams seg = two_segment_alt_q(-20, 15);
  rc::DeltaQParams delta;
  delta.present = false;
  FrameQpHeader header;
  header.primary_ref_is_none = false;

  // update_map = 1, temporal_update = 1.
  seg.update_map = true;
  seg.temporal_update = true;
  BitWriter writer;
  std::string error;
  const std::size_t bits = write_segmentation_and_delta_q(
      writer, seg, delta, 90, header, &error);
  CHECK(error.empty());
  // 1 + (8*3 + 2*11) + 1 + 1 + 2 + 1 + 1 = 45.
  //   seg enabled 1; seg_info 8 ids: 6 empty ids * 3 + 2 ids * (1+10+1+1);
  //   using_qmatrix 1; delta present 1; update_map 1; temporal 1.
  CHECK_EQ(bits, 1U + 44U + 1U + 1U + 1U + 1U);
  const std::vector<std::uint8_t> bytes = finish(writer);
  BitReader reader(bytes);
  rc::SegmentationParams read_seg;
  rc::DeltaQParams read_delta;
  bool using_qmatrix = true;
  CHECK(read_segmentation_and_delta_q(reader, 90, header, read_seg,
                                      read_delta, using_qmatrix, &error));
  CHECK(same_segmentation(read_seg, seg));
  CHECK(read_seg.update_map);
  CHECK(read_seg.temporal_update);
  CHECK(!read_delta.present);

  FrameSegmentation derived;
  CHECK(derive_frame_segmentation(read_seg, read_delta, 90, header, derived,
                                  &error));
  CHECK(derived.update_map);
  CHECK(derived.temporal_update);

  // temporal_update on a key frame is rejected.
  header.primary_ref_is_none = true;
  BitWriter bad;
  CHECK_EQ(write_segmentation_params(bad, seg, header, &error), 0U);
  CHECK(!error.empty());
}

void test_frame_header_ext_segments() {
  rc::SegmentationParams seg;
  std::string error;
  CHECK(rc::set_segment_alt_q(seg, 9, -5, &error));
  seg.update_map = true;
  FrameQpHeader header;
  header.primary_ref_is_none = true;
  header.enable_ext_seg = true;

  BitWriter writer;
  const std::size_t bits = write_segmentation_and_delta_q(
      writer, seg, rc::DeltaQParams{}, 90, header, &error);
  CHECK(error.empty());
  // 1 + 16*3 + (1+10) + 1 + 1 = 61.
  CHECK_EQ(bits, 1U + 48U + 10U + 1U + 1U);
  const std::vector<std::uint8_t> bytes = finish(writer);
  BitReader reader(bytes);
  rc::SegmentationParams read_seg;
  rc::DeltaQParams read_delta;
  bool using_qmatrix = true;
  CHECK(read_segmentation_and_delta_q(reader, 90, header, read_seg,
                                      read_delta, using_qmatrix, &error));
  CHECK(error.empty());
  CHECK(read_seg.enabled);
  CHECK(read_seg.segments[9].enabled[rc::kSegLvlAltQ]);
  CHECK_EQ(read_seg.segments[9].data[rc::kSegLvlAltQ], -5);
  CHECK_EQ(max_segments(true), 16);
  CHECK_EQ(max_segments(false), 8);

  FrameSegmentation derived;
  CHECK(derive_frame_segmentation(read_seg, read_delta, 90, header, derived,
                                  &error));
  CHECK_EQ(derived.num_segments, 16);
  CHECK_EQ(derived.last_active_seg_id, 9);
  CHECK_EQ(derived.features[9].data[rc::kSegLvlAltQ], -5);

  // Without enable_ext_seg, id 9 is not representable.
  header.enable_ext_seg = false;
  BitWriter bad;
  CHECK_EQ(write_segmentation_params(bad, seg, header, &error), 0U);
  CHECK(!error.empty());
}

void test_frame_header_errors() {
  std::string error;
  FrameQpHeader header;
  rc::SegmentationParams seg;
  CHECK(!rc::set_segment_alt_q(seg, 0, 352, &error));  // helper rejects
  CHECK(!error.empty());
  error.clear();

  rc::DeltaQParams delta;
  delta.res = 4;
  BitWriter writer;
  CHECK_EQ(write_delta_q_params(writer, delta, 90, &error), 0U);
  CHECK(!error.empty());
  error.clear();

  delta = rc::DeltaQParams{};
  delta.present = true;
  delta.res = 1;
  BitWriter base0;
  CHECK_EQ(write_delta_q_params(base0, delta, 0, &error), 0U);
  CHECK(!error.empty());
  error.clear();

  header.using_qmatrix = true;
  BitWriter qm;
  CHECK_EQ(write_segmentation_and_delta_q(qm, rc::SegmentationParams{},
                                          rc::DeltaQParams{}, 90, header,
                                          &error),
           0U);
  CHECK(!error.empty());
}

void test_delta_q_semantics() {
  std::string error;
  rc::SegmentationParams seg = two_segment_alt_q(-100, 20);
  seg.segments[1] = rc::SegmentFeatures{};
  rc::DeltaQParams delta;
  delta.present = true;
  delta.res = 0;
  FrameQpHeader header;
  FrameSegmentation derived;
  CHECK(derive_frame_segmentation(seg, delta, 90, header, derived, &error));

  // Segment 0: 90 - 100 = -10 -> 0 -> lossless because delta_q_present == 0
  // is required too; here delta is present, so it is not lossless.
  CHECK(!derived.has_lossless_segment);
  int qindex = -1;
  CHECK(effective_qindex(derived, 90, 120, true, 0, false, qindex, &error));
  CHECK_EQ(qindex, 120 - 100);  // CurrentQIndex + ALT_Q
  CHECK(effective_qindex(derived, 90, 120, true, 0, true, qindex, &error));
  CHECK_EQ(qindex, 0);  // base + ALT_Q, clipped
  CHECK(effective_qindex(derived, 90, 120, true, 1, false, qindex, &error));
  CHECK_EQ(qindex, 120);

  delta.present = false;
  CHECK(derive_frame_segmentation(seg, delta, 90, header, derived, &error));
  CHECK(derived.has_lossless_segment);

  // SEG_LVL_SKIP sets SegIdPreSkip.
  rc::SegmentationParams skip_seg = seg;
  skip_seg.segments[2].enabled[rc::kSegLvlSkip] = true;
  CHECK(derive_frame_segmentation(skip_seg, rc::DeltaQParams{}, 90, header,
                                  derived, &error));
  CHECK(derived.seg_id_pre_skip);
  CHECK_EQ(derived.last_active_seg_id, 2);
}

// ---------------------------------------------------------------------------
// delta_q_abs symbol
// ---------------------------------------------------------------------------

void test_delta_q_abs_roundtrip() {
  for (int disable = 0; disable <= 1; ++disable) {
    BlockCdfs cdfs;
    cdfs.init();
    std::vector<std::uint8_t> buffer(1U << 16, 0U);
    av2::Encoder encoder;
    CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
    encoder.disable_cdf_update = static_cast<std::uint8_t>(disable);

    std::vector<int> values;
    for (int v = -kDeltaQMaxAbs; v <= kDeltaQMaxAbs; ++v) values.push_back(v);
    // Interleave a second pass so the adapting CDF sees repeats.
    for (int v = -40; v <= 40; ++v) values.push_back(v);

    std::string error;
    for (const int value : values) {
      CHECK(write_delta_q_abs(encoder, cdfs.delta_q, value, &error));
      CHECK(error.empty());
    }
    std::size_t payload_bytes = 0;
    CHECK(encoder.exit_symbol(&payload_bytes) == av2::Status::kOk);

    BlockCdfs dec_cdfs;
    dec_cdfs.init();
    av2::Decoder decoder;
    CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
          av2::Status::kOk);
    decoder.disable_cdf_update = static_cast<std::uint8_t>(disable);
    for (const int value : values) {
      int decoded = 0;
      CHECK(read_delta_q_abs(decoder, dec_cdfs.delta_q, decoded, &error));
      CHECK(error.empty());
      CHECK_EQ(decoded, value);
    }
    CHECK(decoder.exit_symbol() == av2::Status::kOk);
    CHECK(std::memcmp(cdfs.delta_q, dec_cdfs.delta_q,
                      sizeof(cdfs.delta_q)) == 0);
  }

  // Escape boundaries and rejection of uncodeable values.
  BlockCdfs cdfs;
  cdfs.init();
  std::vector<std::uint8_t> buffer(4096U, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  std::string error;
  for (const int value : {6, 7, 8, 9, 12, 13, 20, 21, 260, 261, 516}) {
    CHECK(write_delta_q_abs(encoder, cdfs.delta_q, value, &error));
    CHECK(error.empty());
    CHECK(write_delta_q_abs(encoder, cdfs.delta_q, -value, &error));
    CHECK(error.empty());
  }
  CHECK(!write_delta_q_abs(encoder, cdfs.delta_q, 517, &error));
  CHECK(!error.empty());
}

// ---------------------------------------------------------------------------
// Tile QP state and read_delta_qindex ordering
// ---------------------------------------------------------------------------

void test_delta_qindex_tile_reset() {
  std::string error;
  rc::DeltaQParams delta;
  delta.present = true;
  delta.res = 2;
  TileQp encoder_state;
  CHECK(begin_tile(encoder_state, 90, delta, &error));
  CHECK_EQ(encoder_state.current_q_index, 90);

  const int reduced[3] = {3, -2, 1};
  const int expected[3] = {90 + 12, 90 + 12 - 8, 90 + 12 - 8 + 4};
  std::vector<std::uint8_t> buffer(1U << 14, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();

  // Three superblocks; the first block of each (64x64, not skipped) codes the
  // delta.  A second block in each SB codes nothing.
  for (int sb = 0; sb < 3; ++sb) {
    begin_superblock(encoder_state);
    CHECK(codes_delta_q(encoder_state, /*mi_size_is_sb=*/true, false));
    CHECK(write_delta_qindex(encoder, cdfs.delta_q, encoder_state, true, false,
                             reduced[sb], &error));
    CHECK(error.empty());
    CHECK_EQ(encoder_state.current_q_index, expected[sb]);
    CHECK(!codes_delta_q(encoder_state, true, false));
    CHECK(write_delta_qindex(encoder, cdfs.delta_q, encoder_state, true, false,
                             0, &error));
    CHECK_EQ(encoder_state.current_q_index, expected[sb]);
  }
  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2::Status::kOk);

  TileQp decoder_state;
  CHECK(begin_tile(decoder_state, 90, delta, &error));
  BlockCdfs dec_cdfs;
  dec_cdfs.init();
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2::Status::kOk);
  for (int sb = 0; sb < 3; ++sb) {
    begin_superblock(decoder_state);
    int value = 0;
    CHECK(read_delta_qindex(decoder, dec_cdfs.delta_q, decoder_state, true,
                            false, value, &error));
    CHECK_EQ(value, reduced[sb]);
    CHECK_EQ(decoder_state.current_q_index, expected[sb]);
    CHECK(read_delta_qindex(decoder, dec_cdfs.delta_q, decoder_state, true,
                            false, value, &error));
    CHECK_EQ(value, 0);
  }
  CHECK(decoder.exit_symbol() == av2::Status::kOk);

  // Per-tile reset (§5.20.1): a new tile starts at base_q_idx again.
  CHECK(begin_tile(decoder_state, 90, delta, &error));
  CHECK_EQ(decoder_state.current_q_index, 90);
}

void test_delta_qindex_skip_gating() {
  std::string error;
  rc::DeltaQParams delta;
  delta.present = true;
  delta.res = 0;
  TileQp state;
  CHECK(begin_tile(state, 90, delta, &error));
  std::vector<std::uint8_t> buffer(4096U, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();

  begin_superblock(state);
  // MiSize == SbSize && skip_flag: no symbol, no state change, but ReadDeltas
  // is cleared for the whole superblock.
  CHECK(!codes_delta_q(state, true, true));
  CHECK(write_delta_qindex(encoder, cdfs.delta_q, state, true, true, 5,
                           &error));
  CHECK(error.empty());
  CHECK_EQ(state.current_q_index, 90);
  CHECK(!state.read_deltas);

  // A skipped sub-superblock block does not clear ReadDeltas: only a full-SB
  // skip suppresses the delta, and the first coded block still carries it.
  begin_superblock(state);
  CHECK(codes_delta_q(state, false, true));
  CHECK(state.read_deltas);
  TileQp reload = state;
  CHECK(begin_tile(reload, 90, delta, &error));
  begin_superblock(reload);
  CHECK(reload.read_deltas);
  CHECK(!codes_delta_q(reload, true, true));
  CHECK(codes_delta_q(reload, true, false));
}

// ---------------------------------------------------------------------------
// Segment ids
// ---------------------------------------------------------------------------

void test_neg_deinterleave_inverse() {
  for (int max = 1; max <= kMaxSegmentsSpec; ++max) {
    for (int ref = 0; ref < max; ++ref) {
      for (int target = 0; target < max; ++target) {
        int coded = -1;
        CHECK(neg_interleave(target, ref, max, coded));
        CHECK(coded >= 0 && coded < max);
        CHECK_EQ(neg_deinterleave(coded, ref, max), target);
      }
    }
  }
  int coded = 0;
  CHECK(!neg_interleave(8, 0, 8, coded));
}

void test_segment_prediction_and_context() {
  SegmentGrid grid;
  std::string error;
  CHECK(init_segment_grid(grid, 4, 4, 0, 4, 0, 4, false, 8, {}, &error));
  // First block has no neighbours.
  CHECK_EQ(segment_prediction(grid, 0, 0), 0);
  CHECK_EQ(segment_id_context(-1, -1, -1), 0);

  SegmentBlock b{0, 0, 2, 2};
  CHECK(commit_segment_block(grid, b, 3, &error));
  CHECK_EQ(segment_prediction(grid, 0, 2), 3);       // left
  CHECK_EQ(segment_prediction(grid, 2, 0), 3);       // above
  CHECK_EQ(segment_prediction(grid, 2, 2), 0);       // U/L both 0, UL ignored
  CHECK_EQ(segment_id_context(3, 3, 3), 2);
  CHECK_EQ(segment_id_context(3, 3, 1), 1);
  CHECK_EQ(segment_id_context(3, 1, 2), 0);
}

void test_intra_segment_id_roundtrip() {
  std::string error;
  rc::SegmentationParams params = two_segment_alt_q(0, 12);
  params.segments[1] = rc::SegmentFeatures{};
  params.segments[2].enabled[rc::kSegLvlAltQ] = true;
  params.segments[2].data[rc::kSegLvlAltQ] = 25;
  params.update_map = true;
  FrameQpHeader header;
  FrameSegmentation seg;
  CHECK(derive_frame_segmentation(params, rc::DeltaQParams{}, 90, header, seg,
                                  &error));
  CHECK(seg.enabled);
  CHECK(!seg.temporal_update);

  SegmentGrid enc_grid;
  SegmentGrid dec_grid;
  CHECK(init_segment_grid(enc_grid, 4, 4, 0, 4, 0, 4, false, 8, {}, &error));
  CHECK(init_segment_grid(dec_grid, 4, 4, 0, 4, 0, 4, false, 8, {}, &error));

  std::vector<std::uint8_t> buffer(1U << 14, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();
  BlockCdfs dec_cdfs;
  dec_cdfs.init();

  const SegmentBlock blocks[4] = {{0, 0, 2, 2}, {0, 2, 2, 2},
                                  {2, 0, 2, 2}, {2, 2, 2, 2}};
  const int ids[4] = {2, 2, 1, 0};
  for (int i = 0; i < 4; ++i) {
    CHECK(write_intra_segment_id(encoder, cdfs, seg, enc_grid, blocks[i],
                                 ids[i], false, &error));
    CHECK(error.empty());
    CHECK(commit_segment_block(enc_grid, blocks[i], ids[i], &error));
  }
  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2::Status::kOk);

  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2::Status::kOk);
  for (int i = 0; i < 4; ++i) {
    int id = -1;
    CHECK(read_intra_segment_id(decoder, dec_cdfs, seg, dec_grid, blocks[i],
                                false, id, &error));
    CHECK(error.empty());
    CHECK_EQ(id, ids[i]);
    CHECK(commit_segment_block(dec_grid, blocks[i], id, &error));
  }
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
  CHECK(enc_grid.current == dec_grid.current);
}

void test_segment_id_skip_branch() {
  std::string error;
  rc::SegmentationParams params = two_segment_alt_q(0, 10);
  params.segments[0].enabled[rc::kSegLvlAltQ] = false;
  params.segments[0].data[rc::kSegLvlAltQ] = 0;
  params.update_map = true;
  FrameQpHeader header;
  FrameSegmentation seg;
  CHECK(derive_frame_segmentation(params, rc::DeltaQParams{}, 90, header, seg,
                                  &error));
  CHECK(!seg.has_lossless_segment);

  SegmentGrid grid;
  CHECK(init_segment_grid(grid, 2, 2, 0, 2, 0, 2, false, 8, {}, &error));
  std::vector<std::uint8_t> buffer(4096U, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();

  // No neighbours -> pred 0.  A skipped block must reuse it and codes nothing.
  SegmentBlock b{0, 0, 2, 2};
  CHECK(write_intra_segment_id(encoder, cdfs, seg, grid, b, 0, /*skip=*/true,
                               &error));
  CHECK(!write_intra_segment_id(encoder, cdfs, seg, grid, b, 1,
                                /*skip=*/true, &error));
  error.clear();

  // HasLosslessSegment = 1 forces even skipped blocks to code the id.
  rc::SegmentationParams lossless = params;
  lossless.segments[0].enabled[rc::kSegLvlAltQ] = true;
  lossless.segments[0].data[rc::kSegLvlAltQ] = -90;
  FrameSegmentation lossless_seg;
  CHECK(derive_frame_segmentation(lossless, rc::DeltaQParams{}, 90, header,
                                  lossless_seg, &error));
  CHECK(lossless_seg.has_lossless_segment);
}

void test_inter_segment_id_temporal() {
  std::string error;
  rc::SegmentationParams params = two_segment_alt_q(0, 8);
  params.segments[0].enabled[rc::kSegLvlAltQ] = false;
  params.segments[0].data[rc::kSegLvlAltQ] = 0;
  params.update_map = true;
  params.temporal_update = true;
  FrameQpHeader header;
  header.primary_ref_is_none = false;
  FrameSegmentation seg;
  CHECK(derive_frame_segmentation(params, rc::DeltaQParams{}, 90, header, seg,
                                  &error));
  CHECK(seg.temporal_update);

  // Previous frame map: claim segment 1 for the first 2x2 block.
  std::vector<std::uint8_t> previous(16, 1);
  previous[0] = 0;
  previous[1] = 0;
  SegmentGrid enc_grid;
  SegmentGrid dec_grid;
  CHECK(init_segment_grid(enc_grid, 4, 4, 0, 4, 0, 4, false, 8, previous,
                          &error));
  CHECK(init_segment_grid(dec_grid, 4, 4, 0, 4, 0, 4, false, 8, previous,
                          &error));

  std::vector<std::uint8_t> buffer(1U << 14, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();
  BlockCdfs dec_cdfs;
  dec_cdfs.init();

  const SegmentBlock blocks[4] = {{0, 0, 2, 2}, {0, 2, 2, 2},
                                  {2, 0, 2, 2}, {2, 2, 2, 2}};
  // Block 0 predicted id = min(prev over block) = 0; choose 1 -> not predicted.
  // Block 1 predicted id = 1; choose 1 -> predicted flag.
  // Blocks 2/3 predicted id = 1; choose 0 and 1.
  const int ids[4] = {1, 1, 0, 1};
  const bool pre_skip = true;  // SegIdPreSkip == 0, so pre-skip is the no-op call
  for (int i = 0; i < 4; ++i) {
    bool coded = true;
    CHECK(write_inter_segment_id(encoder, cdfs, seg, enc_grid, blocks[i],
                                 pre_skip, ids[i], false, coded, &error));
    CHECK(error.empty());
    CHECK(!coded);  // pre-skip call must be a no-op when SegIdPreSkip == 0
    CHECK(write_inter_segment_id(encoder, cdfs, seg, enc_grid, blocks[i],
                                 /*pre_skip=*/false, ids[i], false, coded,
                                 &error));
    CHECK(error.empty());
    CHECK(coded);
    CHECK(commit_segment_block(enc_grid, blocks[i], ids[i], &error));
  }
  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2::Status::kOk);

  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2::Status::kOk);
  for (int i = 0; i < 4; ++i) {
    int id = -1;
    bool coded = true;
    CHECK(read_inter_segment_id(decoder, dec_cdfs, seg, dec_grid, blocks[i],
                                pre_skip, false, id, coded, &error));
    CHECK(error.empty());
    CHECK(!coded);
    CHECK(read_inter_segment_id(decoder, dec_cdfs, seg, dec_grid, blocks[i],
                                false, false, id, coded, &error));
    CHECK(error.empty());
    CHECK(coded);
    CHECK_EQ(id, ids[i]);
    CHECK(commit_segment_block(dec_grid, blocks[i], id, &error));
  }
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
  CHECK(enc_grid.current == dec_grid.current);
  // Predicted contexts follow seg_id_predicted.
  CHECK(dec_grid.above_pred[0] == 0);  // block 0: not predicted
  CHECK(dec_grid.left_pred[0] == 1);   // block 1: predicted (same row)
  CHECK(dec_grid.above_pred[2] == 1);  // block 1/3: predicted
  CHECK(dec_grid.left_pred[2] == 1);   // block 3: predicted
}

void test_inter_segment_id_update_map_zero() {
  std::string error;
  rc::SegmentationParams params = two_segment_alt_q(0, 8);
  params.segments[0].enabled[rc::kSegLvlAltQ] = false;
  params.segments[0].data[rc::kSegLvlAltQ] = 0;
  params.update_map = false;
  params.temporal_update = false;
  FrameQpHeader header;
  header.primary_ref_is_none = false;
  FrameSegmentation seg;
  CHECK(derive_frame_segmentation(params, rc::DeltaQParams{}, 90, header, seg,
                                  &error));
  CHECK(!seg.update_map);

  std::vector<std::uint8_t> previous(16, 1);
  SegmentGrid grid;
  CHECK(init_segment_grid(grid, 4, 4, 0, 4, 0, 4, false, 8, previous, &error));

  std::vector<std::uint8_t> buffer(4096U, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();
  SegmentBlock b{0, 0, 2, 2};
  bool coded = true;
  CHECK(write_inter_segment_id(encoder, cdfs, seg, grid, b, true, 1, false,
                               coded, &error));
  CHECK(error.empty());
  CHECK(!coded);
  CHECK(write_inter_segment_id(encoder, cdfs, seg, grid, b, false, 1, false,
                               coded, &error));
  CHECK(error.empty());
  CHECK(!coded);
  // A mismatching id is rejected.
  CHECK(!write_inter_segment_id(encoder, cdfs, seg, grid, b, false, 0, false,
                                coded, &error));
}

void test_segment_id_ext() {
  std::string error;
  rc::SegmentationParams params;
  CHECK(rc::set_segment_alt_q(params, 9, 3, &error));
  params.update_map = true;
  FrameQpHeader header;
  header.primary_ref_is_none = false;
  header.enable_ext_seg = true;
  FrameSegmentation seg;
  CHECK(derive_frame_segmentation(params, rc::DeltaQParams{}, 90, header, seg,
                                  &error));
  CHECK_EQ(seg.num_segments, 16);
  CHECK_EQ(seg.last_active_seg_id, 9);

  SegmentGrid enc_grid;
  SegmentGrid dec_grid;
  CHECK(init_segment_grid(enc_grid, 2, 2, 0, 2, 0, 2, true, 16, {}, &error));
  CHECK(init_segment_grid(dec_grid, 2, 2, 0, 2, 0, 2, true, 16, {}, &error));
  std::vector<std::uint8_t> buffer(4096U, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();
  BlockCdfs dec_cdfs;
  dec_cdfs.init();
  SegmentBlock b{0, 0, 2, 2};
  CHECK(write_intra_segment_id(encoder, cdfs, seg, enc_grid, b, 9, false,
                               &error));
  CHECK(error.empty());
  // §8.3.2: the ext-flag path uses TileSegmentIdExtCdf, not TileSegmentIdCdf.
  CHECK(std::memcmp(cdfs.segment_id_ext[0], cdf::kSegmentIdExtCdf[0],
                    sizeof(cdfs.segment_id_ext[0])) != 0);
  CHECK(std::memcmp(cdfs.segment_id[0], cdf::kSegmentIdCdf[0],
                    sizeof(cdfs.segment_id[0])) == 0);
  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2::Status::kOk);
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2::Status::kOk);
  int id = -1;
  CHECK(read_intra_segment_id(decoder, dec_cdfs, seg, dec_grid, b, false, id,
                              &error));
  CHECK(error.empty());
  CHECK_EQ(id, 9);
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
}

// ---------------------------------------------------------------------------
// RC planner integration
// ---------------------------------------------------------------------------

void test_rc_delta_q_plan() {
  std::string error;
  rc::DeltaQPlanConfig config;
  config.base_q_idx = 100;
  config.delta_q_res = 2;
  config.cols = 4;
  config.rows = 3;
  const int targets[12] = {100, 108, 116, 124, 120, 112, 104, 96,
                           88,  92,  100, 108};
  rc::DeltaQPlan plan;
  CHECK(rc::plan_delta_q(config,
                         std::span<const int>(targets, 12), plan, &error));
  CHECK_EQ(plan.steps.size(), 12U);

  DeltaQPlanView view;
  CHECK(make_delta_q_plan_view(plan, 4, 3, view, &error));
  CHECK(error.empty());
  CHECK(delta_q_plan_step(view, 0, 0) != nullptr);
  CHECK(delta_q_plan_step(view, 4, 0) == nullptr);

  rc::DeltaQParams delta;
  delta.present = true;
  delta.res = 2;
  TileQp encoder_state;
  CHECK(begin_tile(encoder_state, 100, delta, &error));

  std::vector<std::uint8_t> buffer(1U << 14, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 4; ++x) {
      const rc::DeltaQPlanStep* step = delta_q_plan_step(view, x, y);
      CHECK(step != nullptr);
      CHECK(check_plan_step(encoder_state, *step, &error));
      CHECK(error.empty());
      begin_superblock(encoder_state);
      CHECK(write_delta_qindex(encoder, cdfs.delta_q, encoder_state, true,
                               false, step->reduced_delta, &error));
      CHECK(error.empty());
      CHECK_EQ(encoder_state.current_q_index, step->qindex_after);
    }
  }
  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2::Status::kOk);

  TileQp decoder_state;
  CHECK(begin_tile(decoder_state, 100, delta, &error));
  BlockCdfs dec_cdfs;
  dec_cdfs.init();
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2::Status::kOk);
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 4; ++x) {
      const rc::DeltaQPlanStep* step = delta_q_plan_step(view, x, y);
      begin_superblock(decoder_state);
      int reduced = 0;
      CHECK(read_delta_qindex(decoder, dec_cdfs.delta_q, decoder_state, true,
                              false, reduced, &error));
      CHECK_EQ(reduced, step->reduced_delta);
      CHECK_EQ(decoder_state.current_q_index, step->qindex_after);
    }
  }
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
}

void test_rc_segmentation_map() {
  rc::SegmentationMap map;
  map.mi_cols = 4;
  map.mi_rows = 2;
  map.segment_id.assign(8, 0);
  map.segment_id[3] = 2;
  map.segment_id[7] = 3;
  std::string error;
  CHECK(rc::segmentation_map_valid(map, &error));

  SegmentGrid grid;
  CHECK(init_segment_grid(grid, 4, 2, 0, 4, 0, 2, false, 8, {}, &error));
  CHECK(load_segmentation_map(map, grid, &error));
  CHECK(error.empty());
  CHECK_EQ(previous_segment_prediction(grid, 0, 0, 4, 2), 0);
  CHECK_EQ(previous_segment_prediction(grid, 0, 3, 1, 1), 2);
  CHECK_EQ(previous_segment_prediction(grid, 1, 3, 1, 1), 3);
}

// ---------------------------------------------------------------------------
// Pinned range-coded block stream
// ---------------------------------------------------------------------------

void test_block_stream_golden() {
  // One tile, two 64x64 superblocks, first block of each is not skipped.
  // SB0: segment id 2, delta 3 (res 1).  SB1: segment id 1, delta -4.
  rc::SegmentationParams params = two_segment_alt_q(0, 20);
  params.segments[2].enabled[rc::kSegLvlAltQ] = true;
  params.segments[2].data[rc::kSegLvlAltQ] = -30;
  params.update_map = true;
  FrameQpHeader header;
  FrameSegmentation seg;
  std::string error;
  CHECK(derive_frame_segmentation(params, rc::DeltaQParams{}, 90, header, seg,
                                  &error));
  CHECK_EQ(seg.last_active_seg_id, 2);

  rc::DeltaQParams delta;
  delta.present = true;
  delta.res = 1;
  TileQp qp;
  CHECK(begin_tile(qp, 90, delta, &error));

  SegmentGrid grid;
  CHECK(init_segment_grid(grid, 4, 4, 0, 4, 0, 4, false, 8, {}, &error));

  std::vector<std::uint8_t> buffer(1U << 14, 0U);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  BlockCdfs cdfs;
  cdfs.init();

  const SegmentBlock b0{0, 0, 2, 2};
  const SegmentBlock b1{2, 2, 2, 2};
  begin_superblock(qp);
  CHECK(write_intra_segment_id(encoder, cdfs, seg, grid, b0, 0, false,
                               &error));
  CHECK(error.empty());
  CHECK(commit_segment_block(grid, b0, 0, &error));
  CHECK(write_delta_qindex(encoder, cdfs.delta_q, qp, true, false, 3, &error));
  CHECK(error.empty());

  begin_superblock(qp);
  CHECK(write_intra_segment_id(encoder, cdfs, seg, grid, b1, 1, false,
                               &error));
  CHECK(error.empty());
  CHECK(commit_segment_block(grid, b1, 1, &error));
  CHECK(write_delta_qindex(encoder, cdfs.delta_q, qp, true, false, -4, &error));
  CHECK(error.empty());

  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2::Status::kOk);
  buffer.resize(payload_bytes);
  const std::string golden = hex(buffer);
  // Regression pin.  The decoder below proves the bytes carry the intended
  // symbols, so a deliberate syntax change must update this string.
  if (golden != "23d2500000") {
    std::fprintf(stderr, "block golden bytes: %s\n", golden.c_str());
  }
  CHECK(golden == "23d2500000");

  // Independent read-back.
  SegmentGrid dec_grid;
  CHECK(init_segment_grid(dec_grid, 4, 4, 0, 4, 0, 4, false, 8, {}, &error));
  TileQp dec_qp;
  CHECK(begin_tile(dec_qp, 90, delta, &error));
  BlockCdfs dec_cdfs;
  dec_cdfs.init();
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2::Status::kOk);

  begin_superblock(dec_qp);
  int id = -1;
  CHECK(read_intra_segment_id(decoder, dec_cdfs, seg, dec_grid, b0, false, id,
                              &error));
  CHECK_EQ(id, 0);
  CHECK(commit_segment_block(dec_grid, b0, id, &error));
  int reduced = 0;
  CHECK(read_delta_qindex(decoder, dec_cdfs.delta_q, dec_qp, true, false,
                          reduced, &error));
  CHECK_EQ(reduced, 3);
  CHECK_EQ(dec_qp.current_q_index, 96);

  begin_superblock(dec_qp);
  CHECK(read_intra_segment_id(decoder, dec_cdfs, seg, dec_grid, b1, false, id,
                              &error));
  CHECK_EQ(id, 1);
  CHECK(commit_segment_block(dec_grid, b1, id, &error));
  CHECK(read_delta_qindex(decoder, dec_cdfs.delta_q, dec_qp, true, false,
                          reduced, &error));
  CHECK_EQ(reduced, -4);
  CHECK_EQ(dec_qp.current_q_index, 88);
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
}

}  // namespace

int main() {
  test_block_cdfs();
  test_coeff_slices();
  test_frame_header_disabled_golden();
  test_frame_header_key_golden();
  test_frame_header_inter_bits();
  test_frame_header_ext_segments();
  test_frame_header_errors();
  test_delta_q_semantics();
  test_delta_q_abs_roundtrip();
  test_delta_qindex_tile_reset();
  test_delta_qindex_skip_gating();
  test_neg_deinterleave_inverse();
  test_segment_prediction_and_context();
  test_intra_segment_id_roundtrip();
  test_segment_id_skip_branch();
  test_inter_segment_id_temporal();
  test_inter_segment_id_update_map_zero();
  test_segment_id_ext();
  test_rc_delta_q_plan();
  test_rc_segmentation_map();
  test_block_stream_golden();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d/%d checks failed\n", g_failures, g_checks);
    return 1;
  }
  std::printf("ok: %d checks\n", g_checks);
  return 0;
}
