// CPU tests for the §5.13 / §6.12 quantizer-matrix OBU state machine,
// payload writer/reader, §5.18.6.2 frame-header fields, and pinned goldens.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/qm/obu/qm_obu.h"
#include "encoder/qm/obu/qm_obu_goldens.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(condition)) {                                                      \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                   #condition);                                              \
      ++g_failures;                                                          \
    }                                                                        \
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

using namespace av2enc::qm::obu;
using av2enc::qm::UserQmLevel;
using av2enc::qm::kUserQmPlanes;
using av2enc::qm::kUserQmT;

bool same_bytes(std::span<const std::uint8_t> a,
                std::span<const std::uint8_t> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size()) == 0;
}

std::string bits_string(const av2enc::BitWriter& writer) {
  std::string out;
  const auto bytes = writer.bytes();
  for (std::size_t i = 0; i < writer.bit_size(); ++i) {
    out.push_back(((bytes[i / 8] >> (7 - (i % 8))) & 1U) != 0U ? '1' : '0');
  }
  return out;
}

bool has_error(const std::string& error, const char* needle) {
  return error.find(needle) != std::string::npos;
}

// Builds a deterministic non-constant matrix.  Values are 1..255 so the
// §6.4.11 non-zero requirement holds.
void fill_varied(UserQmLevel* user, int seed) {
  for (int t = 0; t < kUserQmT; ++t) {
    for (int plane = 0; plane < kUserQmPlanes; ++plane) {
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          const int value =
              1 + ((i * 37 + j * 53 + t * 11 + plane * 19 + seed * 7) % 255);
          user->m[t][plane][i][j] = static_cast<std::uint8_t>(value);
        }
      }
    }
  }
}

// Makes t=0 symmetric, t=2 the transpose of t=1, and chroma planes copies of
// plane 0.  Exercises the three inferred-data paths of §5.4.11.
void make_inferred_friendly(UserQmLevel* user) {
  for (int plane = 0; plane < kUserQmPlanes; ++plane) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        user->m[0][plane][j][i] = user->m[0][plane][i][j];
      }
    }
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 4; ++j) {
        user->m[2][plane][i][j] = user->m[1][plane][j][i];
      }
    }
  }
  for (int t = 0; t < kUserQmT; ++t) {
    std::memcpy(user->m[t][1], user->m[t][0], sizeof(user->m[t][0]));
    std::memcpy(user->m[t][2], user->m[t][0], sizeof(user->m[t][0]));
  }
}

bool user_levels_equal(const UserQmLevel& a, const UserQmLevel& b,
                       int planes) {
  // UserQmLevel stores an 8x8 array for every t, but the fundamental sizes are
  // 8x8 (t=0), 8x4 (t=1) and 4x8 (t=2); bytes outside those extents are not
  // coded and must not participate in the comparison.
  for (int t = 0; t < kUserQmT; ++t) {
    const int rows = (t == 1) ? 4 : 8;
    const int cols = (t == 2) ? 4 : 8;
    for (int plane = 0; plane < planes; ++plane) {
      for (int i = 0; i < rows; ++i) {
        if (std::memcmp(a.m[t][plane][i], b.m[t][plane][i],
                        static_cast<std::size_t>(cols)) != 0) {
          return false;
        }
      }
    }
  }
  return true;
}

bool payload_roundtrip(const QuantizerMatrixObu& input,
                       QuantizerMatrixObu* read_back) {
  av2enc::BitWriter writer;
  std::string error;
  if (!write_quantizer_matrix_payload(writer, input, &error)) {
    std::fprintf(stderr, "write failed: %s\n", error.c_str());
    return false;
  }
  if (!read_quantizer_matrix_payload(writer.bytes(), read_back, &error)) {
    std::fprintf(stderr, "read failed: %s\n", error.c_str());
    return false;
  }
  return true;
}

bool payloads_equal(const QuantizerMatrixObu& a,
                    const QuantizerMatrixObu& b) {
  if (a.bit_map != b.bit_map ||
      a.chroma_info_present != b.chroma_info_present) {
    return false;
  }
  for (int level = 0; level < kNumCustomQms; ++level) {
    if ((a.bit_map & (1U << level)) == 0U) continue;
    const QmLevelData& la = a.levels[static_cast<std::size_t>(level)];
    const QmLevelData& lb = b.levels[static_cast<std::size_t>(level)];
    if (la.is_default != lb.is_default) return false;
    if (!la.is_default &&
        !user_levels_equal(la.user, lb.user, a.chroma_info_present ? 3 : 1)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
void test_obu_framing() {
  std::string error;
  std::vector<std::uint8_t> header;

  ObuHeader h;
  h.type = kObuTypeQuantizationMatrix;
  h.tlayer_id = 3;
  h.mlayer_id = 5;
  h.xlayer_id = 17;
  CHECK(write_obu_header(h, &header, &error));
  CHECK_EQ(header.size(), 2U);
  CHECK_EQ(static_cast<unsigned>(header[0]), 0xdbU);  // ext|22<<2|3
  CHECK_EQ(static_cast<unsigned>(header[1]), (5U << 5U) | 17U);
  ObuHeader parsed;
  std::size_t consumed = 0;
  CHECK(read_obu_header(header, &parsed, &consumed, &error));
  CHECK_EQ(consumed, 2U);
  CHECK_EQ(static_cast<int>(parsed.type), 22);
  CHECK_EQ(static_cast<int>(parsed.tlayer_id), 3);
  CHECK_EQ(static_cast<int>(parsed.mlayer_id), 5);
  CHECK_EQ(static_cast<int>(parsed.xlayer_id), 17);

  // Temporal delimiter without an extension byte infers GLOBAL_XLAYER_ID.
  h = ObuHeader{};
  h.type = kObuTypeTemporalDelimiter;
  CHECK(write_obu_header(h, &header, &error));
  CHECK_EQ(header.size(), 1U);
  CHECK(read_obu_header(header, &parsed, &consumed, &error));
  CHECK_EQ(static_cast<int>(parsed.xlayer_id), 31);

  // Bad ranges.
  h = ObuHeader{};
  h.tlayer_id = 4;
  CHECK(!write_obu_header(h, &header, &error));
  h = ObuHeader{};
  h.mlayer_id = 8;
  CHECK(!write_obu_header(h, &header, &error));
  h = ObuHeader{};
  h.type = kObuTypeTemporalDelimiter;
  h.xlayer_id = 4;
  CHECK(!write_obu_header(h, &header, &error));

  // Two-OBU sequence parse.
  std::vector<std::uint8_t> seq;
  std::vector<std::uint8_t> one;
  CHECK(make_constant_user_qm_obu(0, 32, true, &one, &error));
  seq.insert(seq.end(), one.begin(), one.end());
  CHECK(make_constant_user_qm_obu(1, 48, false, &one, &error));
  seq.insert(seq.end(), one.begin(), one.end());
  std::vector<ParsedObu> obus;
  CHECK(parse_obu_sequence(seq, &obus, &error));
  CHECK_EQ(obus.size(), 2U);
  CHECK_EQ(static_cast<int>(obus[0].header.type), 22);
  std::vector<std::uint8_t> rebuilt;
  for (const ParsedObu& obu : obus) {
    rebuilt.insert(rebuilt.end(), obu.raw.begin(), obu.raw.end());
  }
  CHECK(same_bytes(rebuilt, seq));

  // Truncated / malformed sequences.
  std::vector<std::uint8_t> bad = {0x80};
  CHECK(!parse_obu_sequence(bad, &obus, &error));
  bad = {0x00};
  CHECK(!parse_obu_sequence(bad, &obus, &error));
  bad = {0x05, 0x58};
  CHECK(!parse_obu_sequence(bad, &obus, &error));
  bad = {0x02, 0x80};  // header says extension but no byte
  CHECK(!parse_obu_sequence(bad, &obus, &error));
}

// ---------------------------------------------------------------------------
void test_payload_roundtrip() {
  std::string error;

  // All 15 levels as default matrices.
  {
    QuantizerMatrixObu input;
    input.bit_map = 0x7fff;
    input.chroma_info_present = true;
    for (int level = 0; level < kNumCustomQms; ++level) {
      input.levels[static_cast<std::size_t>(level)].is_default = true;
    }
    QuantizerMatrixObu output;
    CHECK(payload_roundtrip(input, &output));
    CHECK(payloads_equal(input, output));
  }

  // One varied user level, chroma, all inferred paths enabled.
  {
    QuantizerMatrixObu input;
    input.bit_map = static_cast<std::uint16_t>(1U << 5);
    input.chroma_info_present = true;
    input.levels[5].is_default = false;
    fill_varied(&input.levels[5].user, 3);
    make_inferred_friendly(&input.levels[5].user);
    QuantizerMatrixObu output;
    CHECK(payload_roundtrip(input, &output));
    CHECK(payloads_equal(input, output));
  }

  // Asymmetric 8x8, non-transposed 4x8, independent planes: the writer must
  // emit all coefficients rather than inferring them.
  {
    QuantizerMatrixObu input;
    input.bit_map = static_cast<std::uint16_t>(1U << 9);
    input.chroma_info_present = true;
    input.levels[9].is_default = false;
    fill_varied(&input.levels[9].user, 11);
    QuantizerMatrixObu output;
    CHECK(payload_roundtrip(input, &output));
    CHECK(payloads_equal(input, output));
  }

  // Mono: only plane 0 is coded.
  {
    QuantizerMatrixObu input;
    input.bit_map = static_cast<std::uint16_t>(1U << 14);
    input.chroma_info_present = false;
    input.levels[14].is_default = false;
    fill_varied(&input.levels[14].user, 7);
    make_inferred_friendly(&input.levels[14].user);
    QuantizerMatrixObu output;
    CHECK(payload_roundtrip(input, &output));
    CHECK(payloads_equal(input, output));
    CHECK_EQ(static_cast<unsigned>(output.levels[14].user.m[0][0][3][4]),
             static_cast<unsigned>(input.levels[14].user.m[0][0][3][4]));
  }

  // Mixed default + user levels in one bitmap.
  {
    QuantizerMatrixObu input;
    input.bit_map = static_cast<std::uint16_t>((1U << 0) | (1U << 4) |
                                               (1U << 14));
    input.chroma_info_present = true;
    input.levels[0].is_default = true;
    input.levels[4].is_default = false;
    fill_varied(&input.levels[4].user, 1);
    input.levels[14].is_default = false;
    fill_varied(&input.levels[14].user, 2);
    make_inferred_friendly(&input.levels[14].user);
    QuantizerMatrixObu output;
    CHECK(payload_roundtrip(input, &output));
    CHECK(payloads_equal(input, output));
  }

  // Constant helper + complete OBU round-trip through parse_obu_sequence.
  {
    std::vector<std::uint8_t> obu;
    CHECK(make_constant_user_qm_obu(6, 32, true, &obu, &error));
    std::vector<ParsedObu> parsed;
    CHECK(parse_obu_sequence(obu, &parsed, &error));
    CHECK_EQ(parsed.size(), 1U);
    QuantizerMatrixObu matrix;
    CHECK(read_quantizer_matrix_payload(parsed[0].payload, &matrix, &error));
    CHECK_EQ(static_cast<unsigned>(matrix.bit_map), 1U << 6);
    CHECK(!matrix.levels[6].is_default);
    CHECK_EQ(static_cast<unsigned>(matrix.levels[6].user.m[2][2][7][3]), 32U);
  }
}

// ---------------------------------------------------------------------------
void test_goldens() {
  std::string error;
  auto expect_obu = [&](const QuantizerMatrixObu& obu,
                        const ObuHeader* header,
                        std::span<const std::uint8_t> golden) {
    std::vector<std::uint8_t> out;
    CHECK(make_quantizer_matrix_obu(obu, header, &out, &error));
    CHECK(same_bytes(out, golden));
  };

  {
    QuantizerMatrixObu obu;
    obu.bit_map = 0;
    obu.chroma_info_present = true;
    expect_obu(obu, nullptr, goldens::kResetChroma);
    obu.chroma_info_present = false;
    expect_obu(obu, nullptr, goldens::kResetMono);
  }
  {
    QuantizerMatrixObu obu;
    obu.bit_map = 1;
    obu.chroma_info_present = true;
    obu.levels[0].is_default = true;
    expect_obu(obu, nullptr, goldens::kLevel0DefaultChroma);

    obu = QuantizerMatrixObu{};
    obu.bit_map = static_cast<std::uint16_t>(1U << 5);
    obu.chroma_info_present = false;
    obu.levels[5].is_default = true;
    expect_obu(obu, nullptr, goldens::kLevel5DefaultMono);

    obu = QuantizerMatrixObu{};
    obu.bit_map = 0x4001;
    obu.chroma_info_present = true;
    obu.levels[0].is_default = true;
    obu.levels[14].is_default = true;
    expect_obu(obu, nullptr, goldens::kLevels0And14Default);

    obu = QuantizerMatrixObu{};
    obu.bit_map = static_cast<std::uint16_t>(1U << 2);
    obu.chroma_info_present = false;
    obu.levels[2].is_default = true;
    expect_obu(obu, nullptr, goldens::kLevel2DefaultMonoNoChroma);

    obu = QuantizerMatrixObu{};
    obu.bit_map = static_cast<std::uint16_t>(1U << 7);
    obu.chroma_info_present = true;
    obu.levels[7].is_default = true;
    ObuHeader header;
    header.mlayer_id = 2;
    header.tlayer_id = 1;
    expect_obu(obu, &header, goldens::kLevel7DefaultMlayer2Tlayer1);
  }
  {
    std::vector<std::uint8_t> out;
    CHECK(make_constant_user_qm_obu(5, 32, true, &out, &error));
    CHECK(same_bytes(out, goldens::kLevel5User32Chroma));
    CHECK(make_constant_user_qm_obu(3, 32, false, &out, &error));
    CHECK(same_bytes(out, goldens::kLevel3User32Mono));
  }

  // The pinned frame-header bit strings.
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    CHECK(write_frame_qm_params(writer, qm, 3, false, &error));
    CHECK_EQ(bits_string(writer).size(),
             std::strlen(goldens::kFrameOffBits));
    CHECK(bits_string(writer) == goldens::kFrameOffBits);
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.qm_y[0] = 5;
    qm.qm_u[0] = 5;
    qm.qm_v[0] = 5;
    qm.qm_uv_same_as_y[0] = true;
    CHECK(write_frame_qm_params(writer, qm, 3, false, &error));
    CHECK(bits_string(writer) == goldens::kFrameY5UvSameBits);
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.qm_y[0] = 3;
    qm.qm_u[0] = 7;
    qm.qm_v[0] = 7;
    qm.qm_uv_same_as_y[0] = false;
    CHECK(write_frame_qm_params(writer, qm, 3, false, &error));
    CHECK(bits_string(writer) == goldens::kFrameY3U7Bits);
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.pic_qm_num_minus_1 = 1;
    qm.qm_y = {3, 9, 0, 0};
    qm.qm_u = {3, 4, 0, 0};
    qm.qm_v = {3, 4, 0, 0};
    qm.qm_uv_same_as_y = {true, false, false, false};
    qm.separate_uv_delta_q = true;
    CHECK(write_frame_qm_params(writer, qm, 3, true, &error));
    CHECK(bits_string(writer) == goldens::kFrameSeg2SeparateBits);
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.qm_y[0] = 14;
    qm.qm_u[0] = 14;
    qm.qm_v[0] = 14;
    qm.qm_uv_same_as_y[0] = true;
    CHECK(write_frame_qm_params(writer, qm, 1, false, &error));
    CHECK(bits_string(writer) == goldens::kFrameMonoY14Bits);
  }
}

// ---------------------------------------------------------------------------
void check_frame_roundtrip(const FrameQm& input, int num_planes,
                           bool segmentation) {
  std::string error;
  av2enc::BitWriter writer;
  CHECK(write_frame_qm_params(writer, input, num_planes, segmentation, &error));
  CHECK_EQ(writer.bit_size(),
           frame_qm_params_bits(input, num_planes, segmentation));
  FrameQm output;
  std::size_t bits = 0;
  CHECK(read_frame_qm_params(writer.bytes(), num_planes, segmentation,
                             input.separate_uv_delta_q, &output, &bits,
                             &error));
  CHECK_EQ(bits, writer.bit_size());
  CHECK_EQ(output.using_qmatrix, input.using_qmatrix);
  CHECK_EQ(output.pic_qm_num_minus_1, input.pic_qm_num_minus_1);
  CHECK_EQ(static_cast<int>(output.separate_uv_delta_q),
           static_cast<int>(input.separate_uv_delta_q));
  const int qm_num = input.pic_qm_num_minus_1 + 1;
  for (int i = 0; i < qm_num; ++i) {
    CHECK_EQ(static_cast<unsigned>(output.qm_y[static_cast<std::size_t>(i)]),
             static_cast<unsigned>(input.qm_y[static_cast<std::size_t>(i)]));
    if (num_planes == 1) continue;
    CHECK_EQ(static_cast<int>(
                 output.qm_uv_same_as_y[static_cast<std::size_t>(i)]),
             static_cast<int>(
                 input.qm_uv_same_as_y[static_cast<std::size_t>(i)]));
    CHECK_EQ(static_cast<unsigned>(output.qm_u[static_cast<std::size_t>(i)]),
             static_cast<unsigned>(input.qm_u[static_cast<std::size_t>(i)]));
    CHECK_EQ(static_cast<unsigned>(output.qm_v[static_cast<std::size_t>(i)]),
             static_cast<unsigned>(input.qm_v[static_cast<std::size_t>(i)]));
  }
}

void test_frame_header() {
  std::string error;
  FrameQm off;
  check_frame_roundtrip(off, 3, false);
  check_frame_roundtrip(off, 1, true);

  FrameQm same;
  same.using_qmatrix = true;
  same.qm_y[0] = 0;
  same.qm_u[0] = 0;
  same.qm_v[0] = 0;
  same.qm_uv_same_as_y[0] = true;
  check_frame_roundtrip(same, 3, false);
  check_frame_roundtrip(same, 2, false);
  check_frame_roundtrip(same, 1, false);

  FrameQm separate;
  separate.using_qmatrix = true;
  separate.qm_y[0] = 2;
  separate.qm_u[0] = 4;
  separate.qm_v[0] = 6;
  separate.qm_uv_same_as_y[0] = false;
  separate.separate_uv_delta_q = true;
  check_frame_roundtrip(separate, 3, false);

  FrameQm shared_uv;
  shared_uv.using_qmatrix = true;
  shared_uv.qm_y[0] = 2;
  shared_uv.qm_u[0] = 4;
  shared_uv.qm_v[0] = 4;
  shared_uv.qm_uv_same_as_y[0] = false;
  check_frame_roundtrip(shared_uv, 3, false);

  FrameQm four;
  four.using_qmatrix = true;
  four.pic_qm_num_minus_1 = 3;
  for (int i = 0; i < 4; ++i) {
    const auto index = static_cast<std::size_t>(i);
    four.qm_y[index] = static_cast<std::uint8_t>(i);
    four.qm_uv_same_as_y[index] = (i % 2) == 0;
    four.qm_u[index] = static_cast<std::uint8_t>(4 + i);
    four.qm_v[index] = static_cast<std::uint8_t>(8 + i);
  }
  four.separate_uv_delta_q = true;
  for (int i = 0; i < 4; ++i) {
    const auto index = static_cast<std::size_t>(i);
    if (four.qm_uv_same_as_y[index]) {
      four.qm_u[index] = four.qm_y[index];
      four.qm_v[index] = four.qm_y[index];
    }
  }
  check_frame_roundtrip(four, 3, true);

  // Writer rejections.
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.pic_qm_num_minus_1 = 1;
    CHECK(!write_frame_qm_params(writer, qm, 3, false, &error));
    CHECK(has_error(error, "pic_qm_num_minus_1"));
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.pic_qm_num_minus_1 = 4;
    CHECK(!write_frame_qm_params(writer, qm, 3, true, &error));
    CHECK(has_error(error, "f(2)"));
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.qm_y[0] = 4;
    qm.qm_u[0] = 5;
    qm.qm_v[0] = 5;
    qm.qm_uv_same_as_y[0] = true;
    CHECK(!write_frame_qm_params(writer, qm, 3, false, &error));
    CHECK(has_error(error, "qm_uv_same_as_y"));
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.qm_y[0] = 4;
    qm.qm_u[0] = 5;
    qm.qm_v[0] = 6;
    qm.qm_uv_same_as_y[0] = false;
    CHECK(!write_frame_qm_params(writer, qm, 3, false, &error));
    CHECK(has_error(error, "separate_uv_delta_q"));
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.qm_y[0] = 16;
    qm.qm_u[0] = 16;
    qm.qm_v[0] = 16;
    qm.qm_uv_same_as_y[0] = true;
    CHECK(!write_frame_qm_params(writer, qm, 3, false, &error));
    CHECK(has_error(error, "f(4)"));
  }
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    CHECK(!write_frame_qm_params(writer, qm, 0, false, &error));
    CHECK(!write_frame_qm_params(writer, qm, 4, false, &error));
  }

  // Truncated inputs.
  {
    FrameQm qm;
    std::size_t bits = 0;
    CHECK(!read_frame_qm_params(std::span<const std::uint8_t>{}, 3, false,
                                false, &qm, &bits, &error));
    const std::uint8_t one[] = {0x80};  // using=1, no qm_y
    CHECK(!read_frame_qm_params(one, 3, false, false, &qm, &bits, &error));
  }
}

// ---------------------------------------------------------------------------
void test_state_machine() {
  std::string error;

  QmState state;
  for (int level = 0; level < kNumCustomQms; ++level) {
    CHECK(!state.seen(level));
    CHECK(!state.protected_level(level));
    CHECK(!state.data_present(level));
    CHECK_EQ(state.level(level).qm_mlayer_id, -1);
    CHECK_EQ(state.level(level).qm_tlayer_id, -1);
    CHECK(state.user_qm(level) == nullptr);
  }
  CHECK_EQ(state.num_obus_in_window(), 0);

  // A default level still sets QmSeen and QmProtected but clears
  // QmDataPresent; the layer ids come from the OBU header.
  ObuHeader header;
  header.mlayer_id = 3;
  header.tlayer_id = 2;
  QuantizerMatrixObu obu;
  obu.bit_map = static_cast<std::uint16_t>(1U << 2);
  obu.chroma_info_present = true;
  obu.levels[2].is_default = true;
  CHECK(state.apply_obu(header, obu, &error));
  CHECK(state.seen(2));
  CHECK(state.protected_level(2));
  CHECK(!state.data_present(2));
  CHECK_EQ(state.level(2).qm_num_planes, 3);
  CHECK_EQ(state.level(2).qm_mlayer_id, 3);
  CHECK_EQ(state.level(2).qm_tlayer_id, 2);
  CHECK(state.user_qm(2) == nullptr);
  CHECK_EQ(state.num_obus_in_window(), 1);
  CHECK_EQ(static_cast<unsigned>(state.bit_map()), 1U << 2);

  // A user level exposes the matrix.
  obu = QuantizerMatrixObu{};
  obu.bit_map = static_cast<std::uint16_t>(1U << 5);
  obu.chroma_info_present = true;
  obu.levels[5].is_default = false;
  fill_varied(&obu.levels[5].user, 4);
  make_inferred_friendly(&obu.levels[5].user);
  CHECK(state.apply_obu(header, obu, &error));
  CHECK(state.data_present(5));
  CHECK(state.user_qm(5) != nullptr);
  CHECK(user_levels_equal(*state.user_qm(5), obu.levels[5].user, 3));

  // Same level twice in one window is a conformance violation.
  obu.bit_map = static_cast<std::uint16_t>(1U << 5);
  CHECK(!state.apply_obu(header, obu, &error));
  CHECK(has_error(error, "specified twice"));

  // A new frame header clears QmSeen and the window counter.
  state.on_frame_header();
  CHECK(!state.seen(2));
  CHECK(!state.seen(5));
  CHECK(state.data_present(5));
  CHECK_EQ(state.num_obus_in_window(), 0);
  obu.levels[5].is_default = true;
  CHECK(state.apply_obu(header, obu, &error));
  CHECK(state.seen(5));
  CHECK(!state.data_present(5));

  // A second bit_map == 0 OBU in the same window is rejected.
  QmState reset_state;
  QuantizerMatrixObu reset;
  reset.bit_map = 0;
  reset.chroma_info_present = true;
  CHECK(reset_state.apply_obu(header, reset, &error));
  CHECK(!reset_state.apply_obu(header, reset, &error));
  CHECK(has_error(error, "only allowed for the first"));
  CHECK_EQ(reset_state.num_obus_in_window(), 1);
  for (int level = 0; level < kNumCustomQms; ++level) {
    CHECK(reset_state.protected_level(level));
    CHECK(!reset_state.data_present(level));
    CHECK_EQ(reset_state.level(level).qm_mlayer_id, -1);
    CHECK_EQ(reset_state.level(level).qm_tlayer_id, -1);
  }

  // reset_qm() only clears unprotected levels.
  QmState key;
  QuantizerMatrixObu level7;
  level7.bit_map = static_cast<std::uint16_t>(1U << 7);
  level7.chroma_info_present = true;
  level7.levels[7].is_default = false;
  fill_varied(&level7.levels[7].user, 8);
  make_inferred_friendly(&level7.levels[7].user);
  CHECK(key.apply_obu(ObuHeader{}, level7, &error));
  CHECK_EQ(key.level(7).qm_mlayer_id, 0);
  key.reset_qm(3);
  CHECK(key.data_present(7));
  CHECK(key.user_qm(7) != nullptr);
  CHECK_EQ(key.level(7).qm_num_planes, 3);
  CHECK_EQ(key.level(0).qm_mlayer_id, -1);
  CHECK(!key.data_present(0));

  // Temporal delimiter drops protection; the next key frame then clears data.
  key.on_temporal_delimiter();
  CHECK(!key.protected_level(7));
  key.on_key_frame(true, 3);
  CHECK(!key.data_present(7));
  CHECK(key.user_qm(7) == nullptr);
  CHECK(key.seen(7) == false);

  // Non-first key frames do not call reset_qm().
  QmState not_first;
  CHECK(not_first.apply_obu(ObuHeader{}, level7, &error));
  not_first.on_key_frame(false, 3);
  CHECK(not_first.data_present(7));

  // Start-of-bitstream reset.
  not_first.reset();
  CHECK(!not_first.data_present(7));
  CHECK_EQ(not_first.num_obus_in_window(), 0);

  // End-to-end: parse an Annex B OBU then apply it.
  QmState from_bytes;
  std::vector<std::uint8_t> bytes;
  CHECK(make_constant_user_qm_obu(11, 32, true, &bytes, &error));
  CHECK(from_bytes.apply_obu_bytes(bytes, &error));
  CHECK(from_bytes.data_present(11));
  CHECK_EQ(from_bytes.level(11).qm_num_planes, 3);
  CHECK(from_bytes.user_qm(11) != nullptr);

  // apply_obu_bytes on a TD fails (not a QM OBU).
  std::vector<std::uint8_t> td;
  ObuHeader td_header;
  td_header.type = kObuTypeTemporalDelimiter;
  td_header.xlayer_id = kGlobalXLayerId;
  CHECK(frame_obu(td_header, {}, &td, &error));
  CHECK(!from_bytes.apply_obu_bytes(td, &error));
}

// ---------------------------------------------------------------------------
void test_conformance_rejections() {
  std::string error;

  // Writer: bit_map bit 15 is not representable.
  {
    QuantizerMatrixObu obu;
    obu.bit_map = 0x8000;
    av2enc::BitWriter writer;
    CHECK(!write_quantizer_matrix_payload(writer, obu, &error));
    CHECK(has_error(error, "f(15)"));
  }
  // Writer: coefficient 0 is forbidden by §6.4.11.
  {
    QuantizerMatrixObu obu;
    obu.bit_map = 1;
    obu.chroma_info_present = false;
    obu.levels[0].is_default = false;
    fill_varied(&obu.levels[0].user, 1);
    obu.levels[0].user.m[0][0][0][0] = 0;
    av2enc::BitWriter writer;
    CHECK(!write_quantizer_matrix_payload(writer, obu, &error));
    CHECK(has_error(error, "coefficient 0"));
  }
  // Writer: constant helper rejects zero.
  {
    std::vector<std::uint8_t> out;
    CHECK(!make_constant_user_qm_obu(1, 0, true, &out, &error));
    CHECK(!make_constant_user_qm_obu(15, 32, true, &out, &error));
    CHECK(!make_constant_user_qm_obu(-1, 32, true, &out, &error));
  }
  // Reader: empty payload.
  {
    QuantizerMatrixObu obu;
    CHECK(!read_quantizer_matrix_payload(std::span<const std::uint8_t>{},
                                        &obu, &error));
  }
  // Reader: missing trailing one bit (all-zero payload byte).
  {
    QuantizerMatrixObu obu;
    const std::uint8_t zeros[] = {0x00, 0x00, 0x00};
    CHECK(!read_quantizer_matrix_payload(zeros, &obu, &error));
    CHECK(has_error(error, "trailing_one_bit"));
  }
  // Reader: non-zero padding after the trailing one bit.
  {
    QuantizerMatrixObu obu;
    // reset bitmap, chroma=1, trailing one, then a stray one in padding.
    const std::uint8_t stray[] = {0x00, 0x01, 0xc1};
    CHECK(!read_quantizer_matrix_payload(stray, &obu, &error));
    CHECK(has_error(error, "non-zero trailing bit"));
  }
  // Reader: truncated user data.
  {
    // level 0, chroma 0, user data flag 0, then end before a full matrix.
    const std::uint8_t truncated[] = {0x00, 0x02, 0x00};
    QuantizerMatrixObu obu;
    CHECK(!read_quantizer_matrix_payload(truncated, &obu, &error));
  }
  // Reader: out-of-range svlc delta.  uvlc code 258 maps to delta 129.
  {
    av2enc::BitWriter writer;
    writer.put_bits(1U, 15U);  // bit_map = level 0
    writer.put_bit(false);     // chroma info absent
    writer.put_bit(false);     // qm_is_default_flag = 0
    writer.put_bit(false);     // qm_8x8_is_symmetric = 0
    // uvlc(258): 8 zeros, one, then 8 payload bits of 2.
    writer.put_bits(0U, 8U);
    writer.put_bit(true);
    writer.put_bits(2U, 8U);
    writer.put_bit(true);  // trailing bit
    writer.align_to_byte();
    QuantizerMatrixObu obu;
    CHECK(!read_quantizer_matrix_payload(writer.bytes(), &obu, &error));
    CHECK(has_error(error, "quant_delta"));
  }
  // State: applying a non-QM OBU type is rejected.
  {
    QmState state;
    QuantizerMatrixObu obu;
    ObuHeader header;
    header.type = kObuTypeClosedLoopKey;
    CHECK(!state.apply_obu(header, obu, &error));
    CHECK(has_error(error, "OBU_QUANTIZATION_MATRIX"));
  }
  // State: bit 15 impossible, but a crafted struct must still be rejected.
  {
    QmState state;
    QuantizerMatrixObu obu;
    obu.bit_map = 0x8000;
    CHECK(!state.apply_obu(ObuHeader{}, obu, &error));
  }
  // State: zero coefficient in a user matrix is rejected before mutation.
  {
    QmState state;
    QuantizerMatrixObu obu;
    obu.bit_map = 1;
    obu.chroma_info_present = false;
    obu.levels[0].is_default = false;
    obu.levels[0].user.m[1][0][2][2] = 0;
    CHECK(!state.apply_obu(ObuHeader{}, obu, &error));
    CHECK(!state.seen(0));
  }
  // State: duplicate level appears in the same bitmap twice is impossible
  // (a bitmap has one bit per level), but a second OBU is rejected.
  {
    QmState state;
    QuantizerMatrixObu obu;
    obu.bit_map = 3;
    obu.chroma_info_present = false;
    obu.levels[0].is_default = true;
    obu.levels[1].is_default = true;
    CHECK(state.apply_obu(ObuHeader{}, obu, &error));
    CHECK(!state.apply_obu(ObuHeader{}, obu, &error));
    CHECK(has_error(error, "specified twice"));
  }
}

// ---------------------------------------------------------------------------
// A miniature multi-OBU fixture used by the decoder gate: TD, reset OBU,
// default OBU for level 5, user OBU for level 6, CLK frame header.
void test_gate_fixture() {
  std::string error;
  QmState state;
  state.on_temporal_delimiter();

  QuantizerMatrixObu reset;
  reset.bit_map = 0;
  reset.chroma_info_present = true;
  CHECK(state.apply_obu(ObuHeader{}, reset, &error));

  QuantizerMatrixObu level5;
  level5.bit_map = static_cast<std::uint16_t>(1U << 5);
  level5.chroma_info_present = true;
  level5.levels[5].is_default = true;
  CHECK(state.apply_obu(ObuHeader{}, level5, &error));

  QuantizerMatrixObu level6;
  level6.bit_map = static_cast<std::uint16_t>(1U << 6);
  level6.chroma_info_present = true;
  level6.levels[6].is_default = false;
  fill_varied(&level6.levels[6].user, 12);
  make_inferred_friendly(&level6.levels[6].user);
  CHECK(state.apply_obu(ObuHeader{}, level6, &error));

  // §5.18.2: QmSeen reset, then reset_qm() because this is a key frame that
  // starts a temporal unit.  Both levels are protected, so both survive.
  state.on_key_frame(true, 3);
  CHECK(state.data_present(6));
  CHECK(state.user_qm(6) != nullptr);
  CHECK(state.protected_level(5));
  CHECK(!state.data_present(5));
  CHECK_EQ(state.num_obus_in_window(), 0);
}

}  // namespace

int main() {
  test_obu_framing();
  test_payload_roundtrip();
  test_goldens();
  test_frame_header();
  test_state_machine();
  test_conformance_rejections();
  test_gate_fixture();

  std::printf("qm obu: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
