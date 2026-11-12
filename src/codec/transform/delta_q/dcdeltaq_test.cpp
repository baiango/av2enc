// Unit tests for the AV2 v1.0.0 DC/AC delta-Q module.
//
// Coverage:
//   * §5.4.8 sequence delta-Q fields, all 32 base values per field, both
//     separate_uv_delta_q values, monochrome, and pinned bit goldens;
//   * the complete §5.4.8 clause round-trip (tool flags + delta fields);
//   * §5.18.6.1 quantization_params(): every legal su(7) value, pair and
//     triple combinations, both separate_uv_delta_q values, pinned goldens;
//   * §7.14.2 qlookup/get_q/get_qindex over a QP sweep and both UV modes;
//   * §5.18.2 QP-0 lossless interaction (CodedLossless, ALT_Q segments);
//   * §7.14.4 forward/inverse quantization with QM and TCQ;
//   * the encoder-side RD search: determinism, bounds, legality, RD
//     improvement, QP 0 preservation.
#include "codec/transform/delta_q/dcdeltaq.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)

std::string bit_string(const av2enc::BitWriter& writer, std::size_t n) {
  std::string out;
  const auto bytes = writer.bytes();
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t byte = bytes[i / 8];
    const unsigned shift = 7U - static_cast<unsigned>(i % 8U);
    out.push_back(((byte >> shift) & 1U) ? '1' : '0');
  }
  return out;
}

std::string hex_string(const av2enc::BitWriter& writer, std::size_t n) {
  std::string bits = bit_string(writer, n);
  while ((bits.size() & 3U) != 0) bits.push_back('0');
  std::string hex;
  for (std::size_t i = 0; i < bits.size(); i += 4) {
    int v = 0;
    for (std::size_t k = 0; k < 4; ++k) {
      v = (v << 1) | (bits[i + k] == '1' ? 1 : 0);
    }
    hex.push_back("0123456789abcdef"[v]);
  }
  return hex;
}

av2enc::dcdeltaq::SequenceDeltaQ make_seq(bool equal_ac_dc, bool separate,
                                          bool monochrome = false) {
  av2enc::dcdeltaq::SequenceDeltaQ seq;
  seq.monochrome = monochrome;
  seq.equal_ac_dc_q = equal_ac_dc;
  seq.separate_uv_delta_q = separate && !monochrome;
  seq.base_y_dc_delta_q = 0;
  seq.base_uv_dc_delta_q = 0;
  seq.base_uv_ac_delta_q = 0;
  if (!equal_ac_dc) {
    seq.y_dc_delta_q_enabled = true;
    seq.uv_dc_delta_q_enabled = !monochrome;
  }
  seq.uv_ac_delta_q_enabled = !monochrome;
  return seq;
}

bool seq_equal(const av2enc::dcdeltaq::SequenceDeltaQ& a,
               const av2enc::dcdeltaq::SequenceDeltaQ& b) {
  const auto na = av2enc::dcdeltaq::normalized_sequence_delta_q(a);
  const auto nb = av2enc::dcdeltaq::normalized_sequence_delta_q(b);
  return na.monochrome == nb.monochrome &&
         na.separate_uv_delta_q == nb.separate_uv_delta_q &&
         na.equal_ac_dc_q == nb.equal_ac_dc_q &&
         na.y_dc_delta_q_enabled == nb.y_dc_delta_q_enabled &&
         na.uv_dc_delta_q_enabled == nb.uv_dc_delta_q_enabled &&
         na.uv_ac_delta_q_enabled == nb.uv_ac_delta_q_enabled &&
         na.base_y_dc_delta_q == nb.base_y_dc_delta_q &&
         na.base_uv_dc_delta_q == nb.base_uv_dc_delta_q &&
         na.base_uv_ac_delta_q == nb.base_uv_ac_delta_q;
}

// ---------------------------------------------------------------------------
// §7.14.2 qlookup / get_q / get_qindex
// ---------------------------------------------------------------------------

void test_qlookup() {
  static const int kTable[25] = {64, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52,
                                 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72,
                                 74, 76, 78};
  for (int q = 0; q < 25; ++q) {
    CHECK(av2enc::dcdeltaq::qlookup(q) == kTable[q]);
  }
  for (int q = 25; q <= 255; ++q) {
    const int expected = kTable[((q - 1) % 24) + 1] << ((q - 1) / 24);
    CHECK(av2enc::dcdeltaq::qlookup(q) == expected);
  }
  // Monotone non-decreasing over the 8-bit domain.
  for (int q = 2; q <= 255; ++q) {
    CHECK(av2enc::dcdeltaq::qlookup(q) >= av2enc::dcdeltaq::qlookup(q - 1));
  }
  // §7.14.2 special case: qindex 0 with delta <= 0 returns Ac_Qlookup[0].
  for (int delta = -64; delta <= 0; ++delta) {
    CHECK(av2enc::dcdeltaq::get_q(0, delta) == 64);
  }
  for (int delta = 1; delta <= 63; ++delta) {
    CHECK(av2enc::dcdeltaq::get_q(0, delta) ==
          av2enc::dcdeltaq::qlookup(delta));
  }
  // Everything else clips qindex + delta to [1, MaxQ].
  for (int qindex = 1; qindex <= 255; qindex += 3) {
    for (int delta = -100; delta <= 100; delta += 7) {
      const int clamped = std::max(1, std::min(255, qindex + delta));
      CHECK(av2enc::dcdeltaq::get_q(qindex, delta) ==
            av2enc::dcdeltaq::qlookup(clamped));
    }
  }
}

void test_get_qindex() {
  using namespace av2enc::dcdeltaq;
  QuantFrameInputs in;
  in.base_q_idx = 90;
  in.delta_q_present = false;
  in.current_q_index = 0;
  in.num_segments = 8;
  CHECK(get_qindex(in, false, 0) == 90);
  CHECK(get_qindex(in, true, 0) == 90);

  in.delta_q_present = true;
  in.current_q_index = 77;
  CHECK(get_qindex(in, false, 0) == 77);
  CHECK(get_qindex(in, true, 0) == 90);

  in.alt_q_enabled[2] = true;
  in.alt_q[2] = -10;
  CHECK(get_qindex(in, false, 2) == 67);   // 77 - 10
  CHECK(get_qindex(in, true, 2) == 80);    // 90 - 10
  in.alt_q[2] = -255;
  CHECK(get_qindex(in, true, 2) == 0);     // lossless segment

  in.delta_q_present = false;
  in.alt_q[2] = 200;
  CHECK(get_qindex(in, false, 2) == 255);  // clipped to MaxQ
}

// ---------------------------------------------------------------------------
// §5.4.8 sequence syntax
// ---------------------------------------------------------------------------

void test_sequence_delta_fields() {
  using namespace av2enc::dcdeltaq;
  using av2enc::BitWriter;
  using av2enc::deltaq::BitReader;

  // Pinned goldens.
  {
    BitWriter w;
    SequenceDeltaQ seq = make_seq(true, false);
    seq.uv_ac_delta_q_enabled = false;
    const std::size_t n = write_sequence_delta_q_fields(w, seq);
    CHECK(n == 8);
    CHECK(bit_string(w, n) == "01101110");
    CHECK(hex_string(w, n) == "6e");
  }
  {
    BitWriter w;
    SequenceDeltaQ seq = make_seq(false, true);
    seq.base_y_dc_delta_q = -3;    // raw 20
    seq.base_uv_dc_delta_q = -5;   // raw 18
    seq.base_uv_ac_delta_q = 0;    // raw 23
    const std::size_t n = write_sequence_delta_q_fields(w, seq);
    CHECK(n == 20);
    CHECK(bit_string(w, n) == "10101001100101101111");
    CHECK(hex_string(w, n) == "a996f");
    BitReader r(w.bytes());
    SequenceDeltaQ out;
    CHECK(read_sequence_delta_q_fields(r, false, out, nullptr));
    CHECK(seq_equal(seq, out));
    CHECK(r.bit_position() == n);
  }

  // Every legal base-delta combination: 32^3 for equal_ac_dc_q = 0 and
  // separate_uv_delta_q = 1, plus the coupled forms.
  for (int separate = 0; separate < 2; ++separate) {
    for (int raw_y = 0; raw_y < kBaseDeltaCount; ++raw_y) {
      for (int raw_uvdc = 0; raw_uvdc < kBaseDeltaCount; ++raw_uvdc) {
        for (int raw_uvac = 0; raw_uvac < kBaseDeltaCount; ++raw_uvac) {
          SequenceDeltaQ seq = make_seq(false, separate != 0);
          seq.base_y_dc_delta_q = kDeltaDcQuantMin + raw_y;
          seq.base_uv_dc_delta_q = kDeltaDcQuantMin + raw_uvdc;
          seq.base_uv_ac_delta_q = kDeltaDcQuantMin + raw_uvac;
          BitWriter w;
          const std::size_t n = write_sequence_delta_q_fields(w, seq);
          CHECK(n == 20);
          BitReader r(w.bytes());
          SequenceDeltaQ out;
          CHECK(read_sequence_delta_q_fields(r, false, out, nullptr));
          CHECK(seq_equal(seq, out));
        }
      }
    }
  }
  for (int raw_ac = 0; raw_ac < kBaseDeltaCount; ++raw_ac) {
    for (int separate = 0; separate < 2; ++separate) {
      SequenceDeltaQ seq = make_seq(true, separate != 0);
      seq.base_uv_ac_delta_q = kDeltaDcQuantMin + raw_ac;
      BitWriter w;
      const std::size_t n = write_sequence_delta_q_fields(w, seq);
      CHECK(n == 8);
      BitReader r(w.bytes());
      SequenceDeltaQ out;
      CHECK(read_sequence_delta_q_fields(r, false, out, nullptr));
      CHECK(seq_equal(seq, out));
      CHECK(out.base_uv_dc_delta_q == out.base_uv_ac_delta_q);
    }
    // Monochrome: only equal_ac_dc_q and (when unequal) the Y-DC field.
    {
      SequenceDeltaQ seq = make_seq(false, false, true);
      seq.base_y_dc_delta_q = kDeltaDcQuantMin + raw_ac;
      BitWriter w;
      const std::size_t n = write_sequence_delta_q_fields(w, seq);
      CHECK(n == 7);  // equal + 5 + enable
      BitReader r(w.bytes());
      SequenceDeltaQ out;
      CHECK(read_sequence_delta_q_fields(r, true, out, nullptr));
      CHECK(seq_equal(seq, out));
    }
    // Monochrome with equal_ac_dc_q = 1: only the equal bit is coded.
    {
      SequenceDeltaQ seq = make_seq(true, false, true);
      BitWriter w;
      const std::size_t n = write_sequence_delta_q_fields(w, seq);
      CHECK(n == 1);
      CHECK(bit_string(w, n) == "1");
      BitReader r(w.bytes());
      SequenceDeltaQ out;
      CHECK(read_sequence_delta_q_fields(r, true, out, nullptr));
      CHECK(seq_equal(seq, out));
    }
  }

  // Invalid combinations are rejected.
  {
    SequenceDeltaQ seq = make_seq(true, false);
    seq.base_y_dc_delta_q = 1;
    CHECK(!validate_sequence_delta_q(seq, nullptr));
    BitWriter w;
    CHECK(write_sequence_delta_q_fields(w, seq, nullptr) == 0);
  }
}

void test_full_sequence_clause() {
  using namespace av2enc::dcdeltaq;
  using av2enc::BitWriter;
  using av2enc::deltaq::BitReader;

  // Default tools, pinned 17-bit group (9 tool bits + 8 delta bits).
  {
    BitWriter w;
    SeqTransformQuantEntropyConfig cfg;
    cfg.monochrome = false;
    cfg.single_picture_header_flag = true;
    SequenceDeltaQ seq = make_seq(true, false);
    seq.uv_ac_delta_q_enabled = false;
    const std::size_t n =
        write_sequence_transform_quant_entropy_config(w, cfg, seq, nullptr);
    CHECK(n == 17);
    CHECK(bit_string(w, n) == "00000000001101110");
    BitReader r(w.bytes());
    SeqTransformQuantEntropyConfig out_cfg;
    out_cfg.monochrome = false;
    out_cfg.single_picture_header_flag = true;
    SequenceDeltaQ out_seq;
    CHECK(read_sequence_transform_quant_entropy_config(r, out_cfg, out_seq,
                                                       nullptr));
    CHECK(out_cfg.enable_avg_cdf);
    CHECK(out_cfg.avg_cdf_type);
    CHECK(seq_equal(seq, out_seq));
  }

  // M2-style (single_picture_header_flag = 0) with tcq off and avg CDF
  // blending on/off.
  for (int avg = 0; avg < 2; ++avg) {
    SeqTransformQuantEntropyConfig cfg;
    cfg.monochrome = false;
    cfg.single_picture_header_flag = false;
    cfg.enable_chroma_dctonly = true;
    cfg.enable_inter_ddt = true;
    cfg.enable_tcq = false;
    cfg.enable_avg_cdf = avg != 0;
    cfg.avg_cdf_type = avg != 0;
    SequenceDeltaQ seq = make_seq(false, true);
    seq.base_uv_ac_delta_q = -4;
    BitWriter w;
    const std::size_t n =
        write_sequence_transform_quant_entropy_config(w, cfg, seq, nullptr);
    CHECK(n > 0);
    BitReader r(w.bytes());
    SeqTransformQuantEntropyConfig out_cfg;
    out_cfg.monochrome = false;
    out_cfg.single_picture_header_flag = false;
    SequenceDeltaQ out_seq;
    CHECK(read_sequence_transform_quant_entropy_config(r, out_cfg, out_seq,
                                                       nullptr));
    CHECK(out_cfg.enable_inter_ddt);
    CHECK(out_cfg.enable_tcq == false);
    CHECK(out_cfg.enable_avg_cdf == (avg != 0));
    CHECK(out_cfg.avg_cdf_type == (avg != 0));
    CHECK(seq_equal(seq, out_seq));
  }

  // Tool-flag combinations, including the inferred rules.
  for (int mask = 0; mask < (1 << 7); ++mask) {
    for (int single = 0; single < 2; ++single) {
      SeqTransformQuantEntropyConfig cfg;
      cfg.monochrome = false;
      cfg.single_picture_header_flag = single != 0;
      cfg.enable_fsc = (mask & 1) != 0;
      cfg.enable_idtx_intra = (mask & 2) != 0;
      cfg.enable_intra_ist = (mask & 4) != 0;
      cfg.enable_inter_ist = (mask & 8) != 0;
      cfg.enable_chroma_dctonly = (mask & 16) != 0;
      cfg.reduced_tx_part_set = (mask & 32) != 0;
      cfg.enable_cctx = (mask & 64) != 0;
      cfg.enable_tcq = true;
      cfg.choose_tcq_per_frame = !cfg.single_picture_header_flag;
      cfg.enable_parity_hiding = cfg.choose_tcq_per_frame;
      SequenceDeltaQ seq = make_seq(false, true);
      seq.base_y_dc_delta_q = -7;
      seq.base_uv_dc_delta_q = 3;
      seq.base_uv_ac_delta_q = 8;
      BitWriter w;
      const std::size_t n =
          write_sequence_transform_quant_entropy_config(w, cfg, seq, nullptr);
      CHECK(n > 0);
      BitReader r(w.bytes());
      SeqTransformQuantEntropyConfig out_cfg;
      out_cfg.monochrome = false;
      out_cfg.single_picture_header_flag = single != 0;
      SequenceDeltaQ out_seq;
      CHECK(read_sequence_transform_quant_entropy_config(r, out_cfg, out_seq,
                                                         nullptr));
      CHECK(out_cfg.enable_fsc == cfg.enable_fsc);
      CHECK(out_cfg.enable_idtx_intra ==
            (cfg.enable_fsc ? true : cfg.enable_idtx_intra));
      CHECK(out_cfg.enable_intra_ist == cfg.enable_intra_ist);
      CHECK(out_cfg.enable_inter_ist == cfg.enable_inter_ist);
      CHECK(out_cfg.enable_chroma_dctonly == cfg.enable_chroma_dctonly);
      CHECK(out_cfg.reduced_tx_part_set == cfg.reduced_tx_part_set);
      CHECK(out_cfg.enable_cctx == cfg.enable_cctx);
      CHECK(out_cfg.enable_tcq == cfg.enable_tcq);
      CHECK(out_cfg.choose_tcq_per_frame == cfg.choose_tcq_per_frame);
      CHECK(out_cfg.enable_parity_hiding == cfg.enable_parity_hiding);
      CHECK(seq_equal(seq, out_seq));
    }
  }
}

// ---------------------------------------------------------------------------
// §5.18.6.1 frame syntax
// ---------------------------------------------------------------------------

void test_frame_delta_q() {
  using namespace av2enc::dcdeltaq;
  using av2enc::BitWriter;
  using av2enc::deltaq::BitReader;

  // Pinned goldens.
  {
    SequenceDeltaQ seq = make_seq(false, true);
    FrameDeltaQ f;
    f.delta_q_y_dc = -3;
    f.delta_q_u_dc = -5;
    f.delta_q_u_ac = -1;
    f.delta_q_v_dc = -6;
    f.delta_q_v_ac = -2;
    BitWriter w;
    const std::size_t n = write_quantization_params(w, 90, seq, f, nullptr);
    CHECK(n == 49);
    CHECK(bit_string(w, n) ==
          "0101101011111101111111011111111111111101011111110");
    BitReader r(w.bytes());
    QuantizationParams out;
    CHECK(read_quantization_params(r, seq, 8, out, nullptr));
    CHECK(out.base_q_idx == 90);
    CHECK(out.delta.delta_q_y_dc == -3);
    CHECK(out.delta.delta_q_u_dc == -5);
    CHECK(out.delta.delta_q_u_ac == -1);
    CHECK(out.delta.delta_q_v_dc == -6);
    CHECK(out.delta.delta_q_v_ac == -2);
    CHECK(out.delta.diff_uv_delta);
  }
  {
    SequenceDeltaQ seq = make_seq(false, false);
    FrameDeltaQ f;
    f.delta_q_y_dc = -3;
    f.delta_q_u_dc = -5;
    f.delta_q_u_ac = -1;
    f.delta_q_v_dc = -5;
    f.delta_q_v_ac = -1;
    BitWriter w;
    const std::size_t n = write_quantization_params(w, 90, seq, f, nullptr);
    CHECK(n == 32);
    CHECK(hex_string(w, n) == "5afdfbff");
    BitReader r(w.bytes());
    QuantizationParams out;
    CHECK(read_quantization_params(r, seq, 8, out, nullptr));
    CHECK(!out.delta.diff_uv_delta);
    CHECK(out.delta.delta_q_v_dc == -5);
    CHECK(out.delta.delta_q_v_ac == -1);
  }
  {
    SequenceDeltaQ seq = make_seq(true, false);
    FrameDeltaQ f;  // equal_ac_dc_q: only the UV AC pair is signalled
    f.delta_q_u_dc = -1;
    f.delta_q_v_dc = -1;
    f.delta_q_u_ac = -1;
    f.delta_q_v_ac = -1;
    BitWriter w;
    const std::size_t n = write_quantization_params(w, 90, seq, f, nullptr);
    CHECK(n == 16);
    CHECK(hex_string(w, n) == "5aff");
    BitReader r(w.bytes());
    QuantizationParams out;
    CHECK(read_quantization_params(r, seq, 8, out, nullptr));
    CHECK(out.delta.delta_q_u_dc == -1);
    CHECK(out.delta.delta_q_v_dc == -1);
  }

  // All 128 su(7) values, one field at a time, both separate_uv_delta_q
  // values.
  for (int separate = 0; separate < 2; ++separate) {
    SequenceDeltaQ seq = make_seq(false, separate != 0);
    for (int value = kFrameDeltaMin; value <= kFrameDeltaMax; ++value) {
      FrameDeltaQ f;
      f.delta_q_y_dc = value;
      f.delta_q_u_dc = value;
      f.delta_q_v_dc = value;
      f.delta_q_u_ac = value;
      f.delta_q_v_ac = value;
      BitWriter w;
      CHECK(write_quantization_params(w, 0, seq, f, nullptr) > 0);
      BitReader r(w.bytes());
      QuantizationParams out;
      CHECK(read_quantization_params(r, seq, 8, out, nullptr));
      CHECK(out.delta.delta_q_y_dc == value);
      CHECK(out.delta.delta_q_u_dc == value);
      CHECK(out.delta.delta_q_v_dc == value);
      CHECK(out.delta.delta_q_u_ac == value);
      CHECK(out.delta.delta_q_v_ac == value);
    }
  }

  // All (Y-DC, U-DC) pairs and all (U-DC, U-AC) pairs for both UV modes.
  for (int separate = 0; separate < 2; ++separate) {
    SequenceDeltaQ seq = make_seq(false, separate != 0);
    for (int a = kFrameDeltaMin; a <= kFrameDeltaMax; ++a) {
      for (int b = kFrameDeltaMin; b <= kFrameDeltaMax; ++b) {
        FrameDeltaQ f;
        int ac = -b;
        ac = std::max(av2enc::dcdeltaq::kFrameDeltaMin,
                      std::min(av2enc::dcdeltaq::kFrameDeltaMax, ac));
        f.delta_q_y_dc = a;
        f.delta_q_u_dc = b;
        f.delta_q_v_dc = b;
        f.delta_q_u_ac = ac;
        f.delta_q_v_ac = ac;
        BitWriter w;
        CHECK(write_quantization_params(w, 255, seq, f, nullptr) > 0);
        BitReader r(w.bytes());
        QuantizationParams out;
        CHECK(read_quantization_params(r, seq, 8, out, nullptr));
        CHECK(out.delta.delta_q_y_dc == a);
        CHECK(out.delta.delta_q_u_dc == b);
        CHECK(out.delta.delta_q_u_ac == ac);
        if (!separate) {
          CHECK(out.delta.delta_q_v_dc == b);
          CHECK(out.delta.delta_q_v_ac == ac);
        }
      }
    }
  }

  // All (Y-DC, UV-DC, UV-AC) triples for separate_uv_delta_q = 0.  This is
  // the exhaustive 128^3 cross product of the three coded fields (the
  // separate_uv_delta_q = 1 form adds two more independent fields; its U and
  // V pairs are covered exhaustively above).
  {
    SequenceDeltaQ seq = make_seq(false, false);
    for (int y = kFrameDeltaMin; y <= kFrameDeltaMax; ++y) {
      for (int uvdc = kFrameDeltaMin; uvdc <= kFrameDeltaMax; ++uvdc) {
        for (int uvac = kFrameDeltaMin; uvac <= kFrameDeltaMax; ++uvac) {
          FrameDeltaQ f;
          f.delta_q_y_dc = y;
          f.delta_q_u_dc = uvdc;
          f.delta_q_v_dc = uvdc;
          f.delta_q_u_ac = uvac;
          f.delta_q_v_ac = uvac;
          BitWriter w;
          CHECK(write_quantization_params(w, 128, seq, f, nullptr) > 0);
          BitReader r(w.bytes());
          QuantizationParams out;
          CHECK(read_quantization_params(r, seq, 8, out, nullptr));
          CHECK(out.delta.delta_q_y_dc == y);
          CHECK(out.delta.delta_q_u_dc == uvdc);
          CHECK(out.delta.delta_q_u_ac == uvac);
        }
      }
    }
  }

  // equal_ac_dc_q = 1 with separate_uv_delta_q = 1: U and V each have one
  // DC/AC value, and diff_uv_delta is signalled when they differ.
  {
    SequenceDeltaQ seq = make_seq(true, true);
    for (int u = kFrameDeltaMin; u <= kFrameDeltaMax; ++u) {
      for (int v = kFrameDeltaMin; v <= kFrameDeltaMax; v += 7) {
        FrameDeltaQ f;
        f.delta_q_u_dc = u;
        f.delta_q_u_ac = u;
        f.delta_q_v_dc = v;
        f.delta_q_v_ac = v;
        BitWriter w;
        CHECK(write_quantization_params(w, 64, seq, f, nullptr) > 0);
        BitReader r(w.bytes());
        QuantizationParams out;
        CHECK(read_quantization_params(r, seq, 8, out, nullptr));
        CHECK(out.delta.delta_q_u_dc == u);
        CHECK(out.delta.delta_q_u_ac == u);
        CHECK(out.delta.delta_q_v_dc == v);
        CHECK(out.delta.delta_q_v_ac == v);
        CHECK(out.delta.diff_uv_delta == (u != v));
      }
    }
  }

  // Monochrome frame: base_q_idx + optional Y-DC delta only.
  {
    SequenceDeltaQ seq = make_seq(false, false, true);
    for (int y = kFrameDeltaMin; y <= kFrameDeltaMax; ++y) {
      FrameDeltaQ f;
      f.delta_q_y_dc = y;
      BitWriter w;
      const std::size_t n = write_quantization_params(w, 7, seq, f, nullptr);
      CHECK(n == 8 + (y == 0 ? 1 : 8));
      BitReader r(w.bytes());
      QuantizationParams out;
      CHECK(read_quantization_params(r, seq, 8, out, nullptr));
      CHECK(out.base_q_idx == 7);
      CHECK(out.delta.delta_q_y_dc == y);
      CHECK(out.delta.delta_q_u_dc == 0);
      CHECK(out.delta.delta_q_v_ac == 0);
    }
  }

  // su(7) boundaries and the zero shortcut.
  {
    BitWriter w;
    CHECK(write_delta_q(w, 0, nullptr) == 1);
    CHECK(write_delta_q(w, -64, nullptr) == 8);
    CHECK(write_delta_q(w, 63, nullptr) == 8);
    CHECK(write_delta_q(w, 64, nullptr) == 0);
    CHECK(write_delta_q(w, -65, nullptr) == 0);
  }

  // Invalid frame deltas are rejected.
  {
    SequenceDeltaQ seq = make_seq(false, false);
    FrameDeltaQ f;
    f.delta_q_u_dc = 1;
    f.delta_q_v_dc = 2;
    f.delta_q_u_ac = 1;
    f.delta_q_v_ac = 1;
    CHECK(!validate_frame_delta_q(seq, f, nullptr));
    f.delta_q_v_dc = 1;
    f.delta_q_u_ac = 3;
    f.delta_q_v_ac = 3;
    CHECK(validate_frame_delta_q(seq, f, nullptr));
  }
}

// ---------------------------------------------------------------------------
// Derived quantizers and QP 0
// ---------------------------------------------------------------------------

void test_derived_quant() {
  using namespace av2enc::dcdeltaq;
  const int qps[] = {0, 1, 2, 7, 8, 16, 32, 64, 90, 120, 140, 141, 190, 191,
                     254, 255};
  const int deltas[] = {-87, -64, -24, -8, -1, 0, 1, 8, 24, 63, 71};
  for (int qp : qps) {
    for (int separate = 0; separate < 2; ++separate) {
      for (int equal = 0; equal < 2; ++equal) {
        SequenceDeltaQ seq = make_seq(equal != 0, separate != 0);
        for (int d : deltas) {
          FrameDeltaQ f;
          f.delta_q_u_dc = d;
          f.delta_q_v_dc = d;
          f.delta_q_u_ac = d;
          f.delta_q_v_ac = d;
          if (!equal) f.delta_q_y_dc = d;
          QuantFrameInputs in;
          in.base_q_idx = qp;
          in.current_q_index = qp;
          in.seq = seq;
          in.frame = f;
          const QuantFrameState q = derive_quant(in, 0);
          const ResolvedDeltaQ r = resolve_delta_q(seq, f);
          const int qindex = qp;
          CHECK(q.plane[0].dc == get_q(qindex, r.y_dc));
          CHECK(q.plane[0].ac == get_q(qindex, 0));
          CHECK(q.plane[1].dc == get_q(qindex, r.u_dc));
          CHECK(q.plane[1].ac == get_q(qindex, r.u_ac));
          CHECK(q.plane[2].dc == get_q(qindex, r.v_dc));
          CHECK(q.plane[2].ac == get_q(qindex, r.v_ac));
          for (int p = 0; p < 3; ++p) {
            CHECK(q.plane[p].dc >= 40);
            CHECK(q.plane[p].ac >= 40);
          }
        }
      }
    }
  }

  // QP 0 lossless interaction (§5.18.2).
  {
    SequenceDeltaQ seq = make_seq(false, true);
    FrameDeltaQ f;
    QuantFrameInputs in;
    in.base_q_idx = 0;
    in.current_q_index = 0;
    in.seq = seq;
    in.frame = f;
    CHECK(derive_quant(in, 0).lossless);
    CHECK(derive_lossless_flags(in).coded_lossless);
    f.delta_q_y_dc = -1;
    f.delta_q_u_dc = -23;
    f.delta_q_u_ac = -5;
    f.delta_q_v_dc = -23;
    f.delta_q_v_ac = -5;
    in.frame = f;
    CHECK(derive_quant(in, 0).lossless);
    CHECK(derive_lossless_flags(in).coded_lossless);
    // get_q(0, non-positive) is 64, the §7.14.2 special case.
    CHECK(derive_quant(in, 0).plane[0].dc == 64);
    CHECK(derive_quant(in, 0).plane[0].ac == 64);
    // A positive total delta leaves CodedLossless.
    f.delta_q_u_ac = 1;
    in.frame = f;
    CHECK(!derive_quant(in, 0).lossless);
    CHECK(!derive_lossless_flags(in).coded_lossless);
    // delta_q_present also leaves CodedLossless (§5.18.2 requires it to be 0).
    f.delta_q_u_ac = 0;
    in.frame = f;
    in.delta_q_present = true;
    in.current_q_index = 0;
    in.alt_q_enabled[1] = true;
    in.alt_q[1] = -255;
    CHECK(get_qindex(in, true, 1) == 0);
    CHECK(!derive_quant(in, 1).lossless);
    // With delta_q_present = 0, an ALT_Q = -255 segment is lossless while
    // segment 0 (base_q_idx = 90) is not.
    in.delta_q_present = false;
    in.base_q_idx = 90;
    in.current_q_index = 90;
    CHECK(derive_quant(in, 1).lossless);
    CHECK(!derive_quant(in, 0).lossless);
    CHECK(derive_lossless_flags(in).has_lossless_segment);
    CHECK(!derive_lossless_flags(in).coded_lossless);
  }
}

// ---------------------------------------------------------------------------
// §7.14.4 quantizer / dequantizer
// ---------------------------------------------------------------------------

void test_quantizer() {
  using namespace av2enc::dcdeltaq;
  namespace qm = av2enc::qm;
  const int qps[] = {1, 8, 32, 64, 90, 141, 191, 255};
  for (int qp : qps) {
    SequenceDeltaQ seq = make_seq(false, true);
    FrameDeltaQ f;
    f.delta_q_y_dc = 2;
    f.delta_q_u_dc = -3;
    f.delta_q_v_dc = 4;
    f.delta_q_u_ac = -4;
    f.delta_q_v_ac = 5;
    QuantFrameInputs in;
    in.base_q_idx = qp;
    in.current_q_index = qp;
    in.seq = seq;
    in.frame = f;
    const QuantFrameState q = derive_quant(in, 0);

    for (int plane = 0; plane < 3; ++plane) {
      for (int tcq = 0; tcq < 2; ++tcq) {
        QuantBlock qb;
        qb.plane = plane;
        qb.coded_w = 8;
        qb.coded_h = 8;
        qb.tx_size = 1;  // 8x8
        qb.tx_type = 0;  // DCT_DCT
        qb.allow_tcq = tcq != 0;
        qb.q = q.plane[plane];
        // residual = 32, inv_scale = 2 -> coefficient = 64.
        qb.coefficient_scale = 8.0 / (2.0 * 528.0);
        const double coeff[64] = {64.0};
        std::int32_t levels[64] = {};
        std::int32_t dequant[64] = {};
        CHECK(quantize_block(coeff, qb, levels, nullptr));
        CHECK(dequantize_block(levels, qb, dequant, nullptr));
        const double step = static_cast<double>(qb.q.dc) / 8.0;
        CHECK(std::fabs(static_cast<double>(dequant[0]) - 32.0) <= step);
        // AC positions use q.ac; a zero coefficient stays zero.
        CHECK(levels[1] == 0);
        CHECK(dequant[1] == 0);
      }
    }
  }

  // QM identity and default matrix.
  {
    qm::QmFrameState state;
    state.using_qmatrix = true;
    state.seg_qm_level[0][0] = 3;
    qm::fill_user_qm_constant(&state.user_qm[3], 32);
    state.qm_data_present[3] = true;
    QuantBlock qb;
    qb.plane = 0;
    qb.coded_w = 8;
    qb.coded_h = 8;
    qb.tx_size = 1;
    qb.tx_type = 0;
    qb.using_qmatrix = true;
    qb.qm_state = &state;
    qb.q = {64, 132};
    std::int32_t q2[64];
    CHECK(build_q2_matrix(qb, q2, nullptr));
    for (int i = 0; i < 64; ++i) {
      const int q = i == 0 ? 64 : 132;
      CHECK(q2[i] == q);  // identity user matrix: Round2(q * 32, 5) == q
    }
    state.qm_data_present[3] = false;  // default matrix level 3
    CHECK(build_q2_matrix(qb, q2, nullptr));
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const int q = (i == 0 && j == 0) ? 64 : 132;
        const int m = qm::matrix_value(state, 0, 1, 3, i, j);
        CHECK(q2[i * 8 + j] == round2(q * m, 5));
      }
    }
  }

  // TCQ shift interaction.
  {
    QuantBlock qb;
    qb.plane = 0;
    qb.coded_w = 8;
    qb.coded_h = 8;
    qb.q = {64, 64};
    CHECK(dequant_shift(qb) == 0);
    qb.allow_tcq = true;
    CHECK(dequant_shift(qb) == 1);
    qb.lossless = true;
    CHECK(dequant_shift(qb) == 0);
    qb.lossless = false;
    qb.use_fsc = true;
    CHECK(dequant_shift(qb) == 0);
    qb.use_fsc = false;
    qb.plane = 1;
    CHECK(dequant_shift(qb) == 0);
    qb.plane = 0;
    qb.tx_type = 10;  // V_DCT -> TX_CLASS_VERT
    CHECK(dequant_shift(qb) == 0);
  }

  // TCQ level mapping.
  for (int state = 0; state < 8; ++state) {
    for (int q = 0; q < 20; ++q) {
      const int syntax = tcq_syntax_level(q, 6);
      const int full = tcq_full_level(syntax, state);
      // The normative mapping preserves the parity relationship and never
      // loses the ability to represent the level.
      CHECK(((full & 1) == tcq_state_q0(state)) || full == 0);
      CHECK(full >= 0);
    }
  }

  // Lossless pass-through.
  {
    QuantBlock qb;
    qb.plane = 0;
    qb.coded_w = 4;
    qb.coded_h = 4;
    qb.lossless = true;
    qb.q = {64, 64};
    const double coeff[16] = {1.0, -2.0, 3.4};
    std::int32_t levels[16] = {};
    std::int32_t dequant[16] = {};
    CHECK(quantize_block(coeff, qb, levels, nullptr));
    CHECK(levels[0] == 1);
    CHECK(levels[1] == -2);
    CHECK(levels[2] == 3);
    CHECK(dequantize_block(levels, qb, dequant, nullptr));
    CHECK(dequant[0] == 1 && dequant[1] == -2 && dequant[2] == 3);
  }
}

// ---------------------------------------------------------------------------
// Encoder-side search
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> make_gradient(int w, int h, int base, int dx, int dy) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = base + (x * dx + y * dy) / 8;
      v = std::max(0, std::min(255, v));
      out[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
          static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(v);
    }
  }
  return out;
}

void test_search() {
  using namespace av2enc::dcdeltaq;
  const int w = 64;
  const int h = 64;
  const std::vector<std::uint8_t> y = make_gradient(w, h, 40, 9, 5);
  const std::vector<std::uint8_t> u = make_gradient(w / 2, h / 2, 100, 3, 7);
  const std::vector<std::uint8_t> v = make_gradient(w / 2, h / 2, 160, -5, 4);
  FramePlanesView planes{w, h, y, u, v};

  for (int separate = 0; separate < 2; ++separate) {
    for (int equal = 0; equal < 2; ++equal) {
      DeltaQSearchConfig cfg;
      cfg.base_q_idx = 90;
      cfg.separate_uv_delta_q = separate != 0;
      cfg.equal_ac_dc_q = equal != 0;
      cfg.max_passes = 3;
      cfg.max_delta_magnitude = 16;
      DeltaQSearchResult searched;
      std::string error;
      CHECK(search_frame_delta_q(planes, cfg, &searched, &error));
      if (!error.empty()) std::fprintf(stderr, "search error: %s\n", error.c_str());
      CHECK(error.empty());
      CHECK(validate_frame_delta_q(searched.seq, searched.frame, nullptr));
      CHECK(searched.evaluations >= 1);
      CHECK(searched.evaluations <= 1 + cfg.max_passes * 5 * 17);

      // Determinism: a second run gives the same thing.
      DeltaQSearchResult again;
      CHECK(search_frame_delta_q(planes, cfg, &again, &error));
      CHECK(again.rd == searched.rd);
      CHECK(again.frame.delta_q_y_dc == searched.frame.delta_q_y_dc);
      CHECK(again.frame.delta_q_u_dc == searched.frame.delta_q_u_dc);
      CHECK(again.frame.delta_q_u_ac == searched.frame.delta_q_u_ac);
      CHECK(again.evaluations == searched.evaluations);

      // The search result is syntactically representable.
      {
        av2enc::BitWriter bw;
        CHECK(write_sequence_delta_q_fields(bw, searched.seq, nullptr) > 0);
        CHECK(write_quantization_params(bw, cfg.base_q_idx, searched.seq,
                                        searched.frame, nullptr) > 0);
        av2enc::deltaq::BitReader r(bw.bytes());
        SequenceDeltaQ out_seq;
        CHECK(read_sequence_delta_q_fields(r, false, out_seq, nullptr));
        QuantizationParams out_frame;
        CHECK(read_quantization_params(r, out_seq, 8, out_frame, nullptr));
        CHECK(seq_equal(searched.seq, out_seq));
        CHECK(out_frame.base_q_idx == cfg.base_q_idx);
        CHECK(out_frame.delta.delta_q_y_dc == searched.frame.delta_q_y_dc);
        CHECK(out_frame.delta.delta_q_u_dc == searched.frame.delta_q_u_dc);
        CHECK(out_frame.delta.delta_q_u_ac == searched.frame.delta_q_u_ac);
      }

      // RD cannot be worse than the zero-delta baseline (max_passes = 0).
      DeltaQSearchConfig baseline_cfg = cfg;
      baseline_cfg.max_passes = 0;
      DeltaQSearchResult baseline;
      CHECK(search_frame_delta_q(planes, baseline_cfg, &baseline, &error));
      CHECK(searched.rd <= baseline.rd + 1e-9);
    }
  }

  // QP sweep: legality and determinism.
  const int qps[] = {1, 8, 32, 64, 90, 120, 141, 191, 255};
  for (int qp : qps) {
    DeltaQSearchConfig cfg;
    cfg.base_q_idx = qp;
    cfg.separate_uv_delta_q = true;
    cfg.equal_ac_dc_q = false;
    cfg.max_passes = 2;
    cfg.max_delta_magnitude = 12;
    DeltaQSearchResult result;
    std::string error;
    CHECK(search_frame_delta_q(planes, cfg, &result, &error));
    CHECK(validate_frame_delta_q(result.seq, result.frame, nullptr));
    for (int value : {result.frame.delta_q_y_dc, result.frame.delta_q_u_dc,
                      result.frame.delta_q_v_dc, result.frame.delta_q_u_ac,
                      result.frame.delta_q_v_ac}) {
      CHECK(value >= -12 && value <= 12);
    }
  }

  // QP 0 keeps CodedLossless and selects delta 0.
  {
    DeltaQSearchConfig cfg;
    cfg.base_q_idx = 0;
    cfg.separate_uv_delta_q = true;
    cfg.equal_ac_dc_q = false;
    cfg.max_passes = 3;
    DeltaQSearchResult result;
    std::string error;
    CHECK(search_frame_delta_q(planes, cfg, &result, &error));
    CHECK(result.lossless);
    CHECK(result.frame.delta_q_y_dc == 0);
    CHECK(result.frame.delta_q_u_dc == 0);
    CHECK(result.frame.delta_q_v_dc == 0);
    CHECK(result.frame.delta_q_u_ac == 0);
    CHECK(result.frame.delta_q_v_ac == 0);
  }
}

// ---------------------------------------------------------------------------
// Round-trip integration
// ---------------------------------------------------------------------------

void test_header_roundtrip() {
  using namespace av2enc::dcdeltaq;
  using av2enc::BitWriter;
  using av2enc::deltaq::BitReader;
  for (int qp = 0; qp <= 255; qp += 17) {
    for (int separate = 0; separate < 2; ++separate) {
      SequenceDeltaQ seq = make_seq(false, separate != 0);
      seq.base_y_dc_delta_q = -7;
      seq.base_uv_dc_delta_q = 3;
      seq.base_uv_ac_delta_q = 8;
      FrameDeltaQ f;
      f.delta_q_y_dc = -4;
      f.delta_q_u_dc = 5;
      f.delta_q_v_dc = separate ? 6 : 5;
      f.delta_q_u_ac = -2;
      f.delta_q_v_ac = separate ? -3 : -2;
      BitWriter w;
      SeqTransformQuantEntropyConfig cfg;
      const std::size_t seq_bits =
          write_sequence_transform_quant_entropy_config(w, cfg, seq, nullptr);
      const std::size_t frame_bits =
          write_quantization_params(w, qp, seq, f, nullptr);
      CHECK(seq_bits > 0);
      CHECK(frame_bits > 0);
      BitReader r(w.bytes());
      SeqTransformQuantEntropyConfig out_cfg;
      SequenceDeltaQ out_seq;
      CHECK(read_sequence_transform_quant_entropy_config(r, out_cfg, out_seq,
                                                         nullptr));
      QuantizationParams out;
      CHECK(read_quantization_params(r, out_seq, 8, out, nullptr));
      CHECK(out.base_q_idx == qp);
      CHECK(seq_equal(seq, out_seq));
      QuantFrameInputs in;
      in.base_q_idx = qp;
      in.current_q_index = qp;
      in.seq = out_seq;
      in.frame = out.delta;
      const QuantFrameState q = derive_quant(in, 0);
      const ResolvedDeltaQ expected =
          resolve_delta_q(normalized_sequence_delta_q(seq), f);
      CHECK(q.plane[0].dc == get_q(qp, expected.y_dc));
      CHECK(q.plane[1].dc == get_q(qp, expected.u_dc));
      CHECK(q.plane[2].dc == get_q(qp, expected.v_dc));
    }
  }
}

}  // namespace

int main() {
  test_qlookup();
  test_get_qindex();
  test_sequence_delta_fields();
  test_full_sequence_clause();
  test_frame_delta_q();
  test_derived_quant();
  test_quantizer();
  test_search();
  test_header_roundtrip();

  std::printf("dcdeltaq tests: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
