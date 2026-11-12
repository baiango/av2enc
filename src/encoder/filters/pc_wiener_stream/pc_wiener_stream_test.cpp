// Unit tests for the PC-Wiener stream syntax and encoder decision.
//
// The reader is exercised independently of the writer: every case writes,
// parses the bytes back with ReadLrParams/ReadUnit, then re-writes the parsed
// structure and compares bytes.  Pinned goldens make a silent field-order
// regression fail loudly.

#include "encoder/filters/pc_wiener_stream/pc_wiener_stream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/filters/restoration/restoration_common.h"
#include "encoder/filters/spec_tables/spec_test_common.h"

namespace pcw = av2enc::filters::pc_wiener_stream;
namespace test = av2enc::filters::spec::test;

namespace {

std::string Hex(const std::vector<std::uint8_t>& bytes, std::size_t limit = 32) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  const std::size_t n = std::min(bytes.size(), limit);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t b = bytes[i];
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 15]);
  }
  if (bytes.size() > limit) out += "...";
  return out;
}

std::string PrefixHex(const std::vector<std::uint8_t>& bytes, std::size_t n) {
  return Hex(std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + n), n);
}

bool SameFrameFilter(const pcw::FrameFilter& a, const pcw::FrameFilter& b) {
  if (a.match_index != b.match_index || a.merged != b.merged ||
      a.subset != b.subset || a.uv_sym != b.uv_sym) {
    return false;
  }
  for (int j = 0; j < pcw::kWienerNsCoeffStride; ++j) {
    if (a.coeffs[j] != b.coeffs[j]) return false;
  }
  return true;
}

bool SameParams(const pcw::LrParams& a, const pcw::LrParams& b) {
  for (int p = 0; p < 3; ++p) {
    if (a.frame_type[p] != b.frame_type[p]) return false;
    if (a.frame_filters_on[p] != b.frame_filters_on[p]) return false;
    if (a.temporal_pred_flag[p] != b.temporal_pred_flag[p]) return false;
  }
  if (a.num_filter_classes != b.num_filter_classes) return false;
  if (a.num_filter_classes_idx != b.num_filter_classes_idx) return false;
  if (a.luma_shift != b.luma_shift || a.chroma_shift != b.chroma_shift) {
    return false;
  }
  if (a.subclass_lookup_count != b.subclass_lookup_count) return false;
  for (int i = 0; i < a.subclass_lookup_count; ++i) {
    if (a.subclass_lookup[i] != b.subclass_lookup[i]) return false;
  }
  for (int p = 0; p < 3; ++p) {
    const int n = p == 0 ? a.num_filter_classes : 1;
    for (int c = 0; c < n; ++c) {
      if (!SameFrameFilter(a.frame_filter[p][c], b.frame_filter[p][c])) {
        return false;
      }
      for (int j = 0; j < pcw::kWienerNsCoeffStride; ++j) {
        if (a.frame_coeffs[p][c][j] != b.frame_coeffs[p][c][j]) return false;
      }
    }
  }
  return true;
}

pcw::LrHeaderContext MakeContext() {
  pcw::LrHeaderContext context;
  context.coded_lossless = false;
  context.enable_restoration = true;
  context.frame_is_intra = true;
  context.sb_size = 64;
  context.num_planes = 3;
  context.sub_x = 1;
  context.sub_y = 1;
  context.base_q_idx = 90;
  context.lr_tools_disable[0][pcw::kRestoreNone] = false;
  context.lr_tools_disable[0][pcw::kRestorePcWiener] = false;
  context.lr_tools_disable[0][pcw::kRestoreWienerNs] = false;
  context.lr_tools_disable[0][pcw::kRestoreSwitchable] = false;
  context.lr_tools_disable[1][pcw::kRestoreNone] = false;
  context.lr_tools_disable[1][pcw::kRestorePcWiener] = true;
  context.lr_tools_disable[1][pcw::kRestoreWienerNs] = true;
  context.lr_tools_disable[1][pcw::kRestoreSwitchable] = false;
  return context;
}

pcw::LrHeaderContext MakePcOnlyContext() {
  pcw::LrHeaderContext context = MakeContext();
  context.lr_tools_disable[0][pcw::kRestorePcWiener] = false;
  context.lr_tools_disable[0][pcw::kRestoreWienerNs] = true;
  return context;
}

// ---------------------------------------------------------------------------
// §5.4.10 sequence filter config
// ---------------------------------------------------------------------------
void TestSequenceConfig() {
  pcw::SequenceFilterConfig cfg;
  cfg.disable_loopfilters_across_tiles = false;
  cfg.enable_cdef = false;
  cfg.enable_gdf = false;
  cfg.enable_restoration = true;
  cfg.lr_tools_disable[0][pcw::kRestorePcWiener] = false;
  cfg.lr_tools_disable[0][pcw::kRestoreWienerNs] = false;
  cfg.lr_tools_uv_present = false;
  cfg.enable_ccso = false;
  cfg.cdef_on_skip_txfm_always_on = false;
  cfg.cdef_on_skip_txfm_disabled = true;
  cfg.df_par_bits_minus_2 = 1;

  av2enc::BitWriter writer;
  std::string error;
  const std::size_t bits = pcw::WriteSequenceFilterConfig(
      writer, cfg, /*single_picture_header_flag=*/false, &error);
  CHECK(error.empty());
  CHECK(bits == writer.bit_size());
  // 1 (loopfilters) + 1 (cdef) + 1 (gdf) + 1 (restoration) + 1 (pc) +
  // 1 (ns) + 1 (uv_present) + 1 (ccso) + 1 (always_on) + 1 (disabled) +
  // 2 (df_par_bits) == 12.
  CHECK(bits == 12);
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  std::printf("sequence golden: %s\n", Hex(bytes).c_str());
  CHECK(Hex(bytes) == "1050");

  pcw::SequenceFilterConfig parsed;
  pcw::Reader reader(bytes.data(), bits);
  CHECK(pcw::ReadSequenceFilterConfig(reader, false, &parsed, &error));
  CHECK(error.empty());
  CHECK(reader.bit_pos() == bits);
  CHECK(parsed.enable_restoration == cfg.enable_restoration);
  CHECK(parsed.lr_tools_disable[0][pcw::kRestorePcWiener] == false);
  CHECK(parsed.lr_tools_disable[0][pcw::kRestoreWienerNs] == false);
  CHECK(parsed.lr_tools_disable[1][pcw::kRestorePcWiener] == true);
  CHECK(parsed.cdef_on_skip_txfm_disabled == true);
  CHECK(parsed.df_par_bits_minus_2 == 1);
  CHECK(parsed.lr_tools_uv_present == false);
  CHECK(parsed.lr_tools_disable[1][pcw::kRestoreWienerNs] ==
        parsed.lr_tools_disable[0][pcw::kRestoreWienerNs]);

  // uv_present adds exactly one bit and carries the chroma flag.
  cfg.lr_tools_uv_present = true;
  cfg.lr_tools_disable[1][pcw::kRestoreWienerNs] = false;
  CHECK(pcw::SequenceFilterConfigBits(cfg, false, &error) == 13);

  // Restoration off: only the base flags.
  pcw::SequenceFilterConfig off;
  off.enable_restoration = false;
  CHECK(pcw::SequenceFilterConfigBits(off, false, &error) == 9);
}

// ---------------------------------------------------------------------------
// §5.18.7.11 lr_params
// ---------------------------------------------------------------------------
void TestLrParamsPcWiener() {
  const pcw::LrHeaderContext context = MakeContext();
  pcw::LrParams params;
  params.frame_type[0] = pcw::kRestorePcWiener;
  params.frame_type[1] = pcw::kRestoreNone;
  params.frame_type[2] = pcw::kRestoreNone;
  params.luma_shift = 3;
  params.chroma_shift = 3;

  av2enc::BitWriter writer;
  std::string error;
  const std::size_t bits = pcw::WriteLrParams(writer, &params, context, &error);
  CHECK(error.empty());
  CHECK(bits == writer.bit_size());
  // ns(4) = 2 bits; luma size flags = 3 bits.
  CHECK(bits == 5);
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  std::printf("pc lr_params golden: %s\n", Hex(bytes).c_str());
  CHECK(Hex(bytes) == "40");

  pcw::LrParams parsed;
  pcw::Reader reader(bytes.data(), bits);
  CHECK(pcw::ReadLrParams(reader, context, &parsed, &error));
  CHECK(error.empty());
  CHECK(reader.bit_pos() == bits);
  CHECK(parsed.frame_type[0] == pcw::kRestorePcWiener);
  CHECK(parsed.frame_type[1] == pcw::kRestoreNone);
  CHECK(parsed.frame_filters_on[0] == false);
  CHECK(parsed.luma_shift == 3);

  // Re-write the parsed structure: byte-identical.
  av2enc::BitWriter again;
  CHECK(pcw::WriteLrParams(again, &parsed, context, &error) == bits);
  CHECK(std::vector<std::uint8_t>(again.bytes().begin(), again.bytes().end()) ==
        bytes);

  // A frame_filters_on bit is not coded for RESTORE_PC_WIENER.
  pcw::LrParams bad = params;
  bad.frame_filters_on[0] = true;
  CHECK(pcw::LrParamsBits(&bad, context, &error) == 0);
  CHECK(!error.empty());
}

void TestLrParamsFrameClass() {
  const pcw::LrHeaderContext context = MakeContext();
  pcw::LrParams params;
  params.frame_type[0] = pcw::kRestoreWienerNs;
  params.frame_type[1] = pcw::kRestoreNone;
  params.frame_type[2] = pcw::kRestoreNone;
  params.frame_filters_on[0] = true;
  params.num_filter_classes = 16;
  params.luma_shift = 3;
  params.chroma_shift = 3;
  // One translated PC-Wiener filter per class, merged so no coefficient bits.
  for (int c = 0; c < 16; ++c) {
    params.frame_filter[0][c].match_index = 16 + c;
    params.frame_filter[0][c].merged = true;
    pcw::TranslatedPcWienerSet(c, params.frame_filter[0][c].coeffs);
  }

  av2enc::BitWriter writer;
  std::string error;
  const std::size_t bits = pcw::WriteLrParams(writer, &params, context, &error);
  CHECK(error.empty());
  CHECK(bits == writer.bit_size());
  CHECK(bits > 5);
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  std::printf("frame-class lr_params golden: %s (%zu bits)\n",
              Hex(bytes).c_str(), bits);
  CHECK(Hex(bytes, 64) == "bc3f7aedcb76acd8af5aad4a74a8d0ffff80");

  pcw::LrParams parsed;
  pcw::Reader reader(bytes.data(), bits);
  CHECK(pcw::ReadLrParams(reader, context, &parsed, &error));
  CHECK(error.empty());
  CHECK(reader.bit_pos() == bits);
  CHECK(SameParams(params, parsed));
  CHECK(parsed.num_filter_classes == 16);
  CHECK(parsed.subclass_lookup_count == pcw::kNumLutClasses);
  const std::uint8_t* expected = pcw::PcWienerSubclassLookup(
      pcw::FilterSetIndex(context.base_q_idx), 7);
  CHECK(expected != nullptr);
  int lookup_diffs = 0;
  for (int i = 0; i < pcw::kNumLutClasses; ++i) {
    lookup_diffs += parsed.subclass_lookup[i] != expected[i];
  }
  CHECK(lookup_diffs == 0);
  int translated_diffs = 0;
  for (int c = 0; c < 16; ++c) {
    for (int j = 0; j < 12; ++j) {
      translated_diffs += parsed.frame_coeffs[0][c][j] !=
                          pcw::TranslatedPcWiener(c, j);
    }
  }
  CHECK(translated_diffs == 0);

  // Re-write the parsed structure: byte-identical.
  av2enc::BitWriter again;
  CHECK(pcw::WriteLrParams(again, &parsed, context, &error) == bits);
  CHECK(std::vector<std::uint8_t>(again.bytes().begin(), again.bytes().end()) ==
        bytes);
}

void TestLrParamsClassRefs() {
  const pcw::LrHeaderContext context = MakeContext();
  pcw::LrParams params;
  params.frame_type[0] = pcw::kRestoreWienerNs;
  params.frame_filters_on[0] = true;
  params.num_filter_classes = 4;
  params.luma_shift = 3;
  params.chroma_shift = 3;
  // Class 0 codes a subset; classes 1 and 2 merge from the previous class.
  params.frame_filter[0][0].match_index = 0;
  params.frame_filter[0][0].subset = 2;
  for (int j = 0; j < pcw::kWienerNsLumaCoeffs; ++j) {
    const int min = av2enc::restoration::kWienerNsTapsMin[0][j];
    const int k = av2enc::restoration::kWienerNsTapsK[0][j];
    params.frame_filter[0][0].coeffs[j] =
        static_cast<std::int16_t>(min + (1 << (k - 1)));
  }
  params.frame_filter[0][1].match_index = 1;
  params.frame_filter[0][1].merged = true;
  params.frame_filter[0][2].match_index = 2;
  params.frame_filter[0][2].merged = true;
  params.frame_filter[0][3].match_index = 0;
  params.frame_filter[0][3].subset = 0;
  for (int j = 0; j < pcw::kWienerNsLumaCoeffs; ++j) {
    const int min = av2enc::restoration::kWienerNsTapsMin[0][j];
    const int k = av2enc::restoration::kWienerNsTapsK[0][j];
    params.frame_filter[0][3].coeffs[j] =
        static_cast<std::int16_t>(min + (1 << (k - 1)));
  }

  av2enc::BitWriter writer;
  std::string error;
  const std::size_t bits = pcw::WriteLrParams(writer, &params, context, &error);
  CHECK(error.empty());
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  pcw::LrParams parsed;
  pcw::Reader reader(bytes.data(), bits);
  CHECK(pcw::ReadLrParams(reader, context, &parsed, &error));
  CHECK(error.empty());
  CHECK(reader.bit_pos() == bits);
  CHECK(SameParams(params, parsed));
  int differences = 0;
  for (int j = 0; j < pcw::kWienerNsLumaCoeffs; ++j) {
    differences += parsed.frame_coeffs[0][1][j] != parsed.frame_coeffs[0][0][j];
    differences += parsed.frame_coeffs[0][2][j] != parsed.frame_coeffs[0][1][j];
  }
  CHECK(differences == 0);
  av2enc::BitWriter again;
  CHECK(pcw::WriteLrParams(again, &parsed, context, &error) == bits);
  CHECK(std::vector<std::uint8_t>(again.bytes().begin(), again.bytes().end()) ==
        bytes);
  std::printf("class-ref lr_params: %zu bits\n", bits);
}

void TestLrParamsSwitchableAndChroma() {
  pcw::LrHeaderContext context = MakeContext();
  context.lr_tools_disable[1][pcw::kRestoreWienerNs] = false;
  context.lr_tools_disable[1][pcw::kRestorePcWiener] = true;
  pcw::LrParams params;
  params.frame_type[0] = pcw::kRestoreSwitchable;
  params.frame_type[1] = pcw::kRestoreWienerNs;
  params.frame_type[2] = pcw::kRestoreNone;
  params.frame_filters_on[1] = true;
  params.frame_filter[1][0].match_index = 0;
  params.frame_filter[1][0].merged = false;
  params.frame_filter[1][0].subset = 2;
  params.frame_filter[1][0].uv_sym = false;
  for (int j = 0; j < pcw::kWienerNsChromaCoeffs; ++j) {
    const int min = av2enc::restoration::kWienerNsTapsMin[1][j];
    const int k = av2enc::restoration::kWienerNsTapsK[1][j];
    params.frame_filter[1][0].coeffs[j] = static_cast<std::int16_t>(min + (1 << (k - 1)));
  }
  params.luma_shift = 2;
  params.chroma_shift = 3;

  av2enc::BitWriter writer;
  std::string error;
  const std::size_t bits = pcw::WriteLrParams(writer, &params, context, &error);
  CHECK(error.empty());
  CHECK(bits == writer.bit_size());
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  std::printf("switchable+chroma lr_params golden: %s (%zu bits)\n",
              Hex(bytes).c_str(), bits);
  CHECK(Hex(bytes, 64) == "d886c1822000000000");

  pcw::LrParams parsed;
  pcw::Reader reader(bytes.data(), bits);
  CHECK(pcw::ReadLrParams(reader, context, &parsed, &error));
  CHECK(error.empty());
  CHECK(reader.bit_pos() == bits);
  CHECK(SameParams(params, parsed));
  CHECK(parsed.frame_type[0] == pcw::kRestoreSwitchable);
  CHECK(parsed.frame_type[1] == pcw::kRestoreWienerNs);
  CHECK(parsed.frame_filters_on[1] == true);
  CHECK(parsed.frame_filter[1][0].subset == 2);

  av2enc::BitWriter again;
  CHECK(pcw::WriteLrParams(again, &parsed, context, &error) == bits);
  CHECK(std::vector<std::uint8_t>(again.bytes().begin(), again.bytes().end()) ==
        bytes);
}

// ---------------------------------------------------------------------------
// §5.20.10.5 / §5.20.10.6 unit syntax
// ---------------------------------------------------------------------------
struct EncoderBuffer {
  std::vector<std::uint8_t> bytes;
  av2enc::av2::Encoder encoder;
};

bool StartEncoder(EncoderBuffer* buf) {
  buf->bytes.assign(1U << 16, 0);
  return buf->encoder.init_symbol(buf->bytes.data(), buf->bytes.size()) ==
         av2enc::av2::Status::kOk;
}

void TestUnitPcWiener() {
  const int mi_cols = 16;
  const int mi_rows = 16;
  pcw::LrParams params;
  params.frame_type[0] = pcw::kRestorePcWiener;
  params.frame_type[1] = pcw::kRestoreNone;
  params.frame_type[2] = pcw::kRestoreNone;
  params.luma_shift = 3;
  params.chroma_shift = 3;
  pcw::TileState state;
  std::string error;
  CHECK(pcw::InitTileState(&state, params, mi_cols, mi_rows, 1, 1, &error));
  CHECK(error.empty());
  CHECK(state.unit_cols[0] == 1);
  CHECK(state.unit_rows[0] == 1);

  EncoderBuffer enc;
  CHECK(StartEncoder(&enc));
  pcw::UnitDecision on;
  on.type = pcw::kRestorePcWiener;
  const std::uint32_t before = enc.encoder.prefix_bits;
  CHECK(pcw::WriteUnit(enc.encoder, state, 0, 0, 0, on, &error));
  const std::uint32_t after = enc.encoder.prefix_bits;
  std::size_t payload = 0;
  CHECK(enc.encoder.exit_symbol(&payload) == av2enc::av2::Status::kOk);
  CHECK(payload > 0);
  std::printf("pc unit on: %zu bytes (%u bits before exit) hex=%s\n", payload,
              after - before, PrefixHex(enc.bytes, payload).c_str());
  CHECK(PrefixHex(enc.bytes, payload) == "c000");

  // Decode independently.
  pcw::TileState decode_state;
  CHECK(pcw::InitTileState(&decode_state, params, mi_cols, mi_rows, 1, 1,
                           &error));
  av2enc::av2::Decoder decoder;
  CHECK(decoder.init_symbol(enc.bytes.data(), enc.bytes.size(), payload) ==
        av2enc::av2::Status::kOk);
  pcw::UnitDecision decoded;
  CHECK(pcw::ReadUnit(decoder, decode_state, 0, 0, 0, &decoded, &error));
  CHECK(error.empty());
  CHECK(decoded.type == pcw::kRestorePcWiener);
  CHECK(decode_state.unit_type[0][0] == pcw::kRestorePcWiener);

  // Off unit.
  EncoderBuffer enc_off;
  CHECK(StartEncoder(&enc_off));
  pcw::TileState off_state;
  CHECK(pcw::InitTileState(&off_state, params, mi_cols, mi_rows, 1, 1, &error));
  pcw::UnitDecision off;
  off.type = pcw::kRestoreNone;
  CHECK(pcw::WriteUnit(enc_off.encoder, off_state, 0, 0, 0, off, &error));
  std::size_t off_payload = 0;
  CHECK(enc_off.encoder.exit_symbol(&off_payload) == av2enc::av2::Status::kOk);
  av2enc::av2::Decoder off_decoder;
  CHECK(off_decoder.init_symbol(enc_off.bytes.data(), enc_off.bytes.size(), off_payload) ==
        av2enc::av2::Status::kOk);
  pcw::TileState off_decode_state;
  CHECK(pcw::InitTileState(&off_decode_state, params, mi_cols, mi_rows, 1, 1,
                           &error));
  pcw::UnitDecision off_decoded;
  CHECK(pcw::ReadUnit(off_decoder, off_decode_state, 0, 0, 0, &off_decoded,
                      &error));
  CHECK(off_decoded.type == pcw::kRestoreNone);
}

void TestUnitSwitchableAndBanks() {
  const int mi_cols = 16;
  const int mi_rows = 16;
  pcw::LrHeaderContext context = MakeContext();
  context.lr_tools_disable[0][pcw::kRestorePcWiener] = false;
  context.lr_tools_disable[0][pcw::kRestoreWienerNs] = false;

  pcw::LrParams params;
  params.frame_type[0] = pcw::kRestoreSwitchable;
  params.frame_type[1] = pcw::kRestoreNone;
  params.frame_type[2] = pcw::kRestoreNone;
  params.frame_filters_on[0] = false;
  params.luma_shift = 3;
  params.chroma_shift = 3;

  pcw::TileState state;
  std::string error;
  CHECK(pcw::InitTileState(&state, params, mi_cols, mi_rows, 1, 1, &error));

  // One PC unit, one NONE unit in a different tile row isn't possible with a
  // 1x1 unit grid; use four 64x64 units in a 128x128 frame instead.
  const int mi_cols2 = 32;
  const int mi_rows2 = 32;
  CHECK(pcw::InitTileState(&state, params, mi_cols2, mi_rows2, 1, 1, &error));
  CHECK(state.unit_cols[0] == 2);
  CHECK(state.unit_rows[0] == 2);

  EncoderBuffer enc;
  CHECK(StartEncoder(&enc));
  std::vector<pcw::UnitDecision> decisions(4);
  // Unit 0: PC-Wiener.
  decisions[0].type = pcw::kRestorePcWiener;
  // Unit 1: NONE.
  decisions[1].type = pcw::kRestoreNone;
  // Unit 2: Wiener NS, codes coefficients.
  decisions[2].type = pcw::kRestoreWienerNs;
  decisions[2].subset = 3;
  for (int j = 0; j < pcw::kWienerNsLumaCoeffs; ++j) {
    const int min = av2enc::restoration::kWienerNsTapsMin[0][j];
    const int k = av2enc::restoration::kWienerNsTapsK[0][j];
    decisions[2].coeffs[j] =
        static_cast<std::int16_t>(min + ((j * 3 + 1) % (1 << k)));
  }
  // Unit 3: merged from the previous bank entry.
  decisions[3].type = pcw::kRestoreWienerNs;
  decisions[3].merged = true;

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      const int i = r * 2 + c;
      CHECK(pcw::WriteUnit(enc.encoder, state, 0, r, c, decisions[i], &error));
    }
  }
  std::size_t payload = 0;
  CHECK(enc.encoder.exit_symbol(&payload) == av2enc::av2::Status::kOk);

  pcw::TileState decode_state;
  CHECK(pcw::InitTileState(&decode_state, params, mi_cols2, mi_rows2, 1, 1,
                           &error));
  av2enc::av2::Decoder decoder;
  CHECK(decoder.init_symbol(enc.bytes.data(), enc.bytes.size(), payload) ==
        av2enc::av2::Status::kOk);
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      const int i = r * 2 + c;
      pcw::UnitDecision decoded;
      CHECK(pcw::ReadUnit(decoder, decode_state, 0, r, c, &decoded, &error));
      CHECK(error.empty());
      CHECK(decoded.type == decisions[i].type);
      if (decisions[i].type == pcw::kRestoreWienerNs) {
        int coeff_diffs = 0;
        for (int j = 0; j < pcw::kWienerNsLumaCoeffs; ++j) {
          coeff_diffs += decode_state.unit_coeffs[0]
                             [static_cast<std::size_t>(r * 2 + c) *
                                  pcw::kWienerNsCoeffStride +
                              j] != decisions[2].coeffs[j];
        }
        CHECK(coeff_diffs == 0);
      }
    }
  }
  std::printf("switchable units: %zu bytes hex=%s\n", payload,
              PrefixHex(enc.bytes, payload).c_str());
  CHECK(PrefixHex(enc.bytes, payload) == "c1ca9a951f50f26539e48f280000");
}

void TestUnitFrameFiltersOn() {
  const int mi_cols = 16;
  const int mi_rows = 16;
  pcw::LrParams params;
  params.frame_type[0] = pcw::kRestoreWienerNs;
  params.frame_type[1] = pcw::kRestoreNone;
  params.frame_type[2] = pcw::kRestoreNone;
  params.frame_filters_on[0] = true;
  params.num_filter_classes = 4;
  params.num_filter_classes_idx = 3;
  params.luma_shift = 3;
  params.chroma_shift = 3;
  for (int c = 0; c < 4; ++c) {
    params.frame_filter[0][c].match_index = c;
    params.frame_filter[0][c].merged = (c > 0);
    for (int j = 0; j < pcw::kWienerNsLumaCoeffs; ++j) {
      params.frame_filter[0][c].coeffs[j] = static_cast<std::int16_t>(c + j);
    }
  }
  // Fill frame_coeffs as a decoder would; the unit writer only copies class 0.
  std::string error;
  av2enc::BitWriter header;
  CHECK(pcw::WriteLrParams(header, &params, MakeContext(), &error) > 0);
  CHECK(error.empty());

  pcw::TileState state;
  CHECK(pcw::InitTileState(&state, params, mi_cols, mi_rows, 1, 1, &error));
  EncoderBuffer enc;
  CHECK(StartEncoder(&enc));
  pcw::UnitDecision on;
  on.type = pcw::kRestoreWienerNs;
  CHECK(pcw::WriteUnit(enc.encoder, state, 0, 0, 0, on, &error));
  CHECK(error.empty());
  const std::uint32_t bits = enc.encoder.prefix_bits;
  std::size_t payload = 0;
  CHECK(enc.encoder.exit_symbol(&payload) == av2enc::av2::Status::kOk);
  // Only the use_wiener_ns symbol is coded (no coefficients).
  CHECK(bits > 0);
  CHECK(bits < 5);

  pcw::TileState decode_state;
  CHECK(pcw::InitTileState(&decode_state, params, mi_cols, mi_rows, 1, 1,
                           &error));
  av2enc::av2::Decoder decoder;
  CHECK(decoder.init_symbol(enc.bytes.data(), enc.bytes.size(), payload) ==
        av2enc::av2::Status::kOk);
  pcw::UnitDecision decoded;
  CHECK(pcw::ReadUnit(decoder, decode_state, 0, 0, 0, &decoded, &error));
  CHECK(decoded.type == pcw::kRestoreWienerNs);
  int diffs = 0;
  for (int j = 0; j < pcw::kWienerNsLumaCoeffs; ++j) {
    diffs += decode_state.unit_coeffs[0][static_cast<std::size_t>(j)] !=
             params.frame_coeffs[0][0][j];
  }
  CHECK(diffs == 0);
}

// ---------------------------------------------------------------------------
// Encoder decision
// ---------------------------------------------------------------------------
struct SyntheticFrame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> curr;
  std::vector<std::uint8_t> cdef;
  std::vector<std::uint8_t> target;
};

SyntheticFrame MakeSynthetic(int width, int height) {
  SyntheticFrame f;
  f.width = width;
  f.height = height;
  const std::size_t n = static_cast<std::size_t>(width) * height;
  f.curr.assign(n, 0);
  f.cdef.assign(n, 0);
  f.target.assign(n, 0);
  test::TestRng rng(0x51a7u);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int smooth = 96 + (x * 48) / std::max(1, width - 1) +
                         (y * 32) / std::max(1, height - 1);
      const int base = std::max(0, std::min(255, smooth));
      const int noise = rng.Range(-9, 9);
      const int noisy = std::max(0, std::min(255, base + noise));
      const std::size_t i = static_cast<std::size_t>(y) * width + x;
      f.target[i] = static_cast<std::uint8_t>(base);
      f.cdef[i] = static_cast<std::uint8_t>(noisy);
      f.curr[i] = f.cdef[i];
    }
  }
  return f;
}

pcw::DecisionPlanes MakePlanes(const SyntheticFrame& f) {
  pcw::DecisionPlanes planes;
  planes.width = f.width;
  planes.height = f.height;
  planes.bit_depth = 8;
  planes.base_q_idx = 90;
  planes.sub_x = 1;
  planes.sub_y = 1;
  planes.luma_shift = 3;
  planes.chroma_shift = 3;
  planes.curr[0] = f.curr.data();
  planes.cdef[0] = f.cdef.data();
  planes.target[0] = f.target.data();
  planes.curr_stride[0] = f.width;
  planes.cdef_stride[0] = f.width;
  planes.target_stride[0] = f.width;
  return planes;
}

void TestDecisionPerUnit() {
  const SyntheticFrame frame = MakeSynthetic(256, 128);
  const pcw::DecisionPlanes planes = MakePlanes(frame);
  pcw::DecisionResult result;
  std::string error;
  CHECK(pcw::DecidePerUnitPcWiener(planes, 0.0, &result, &error));
  CHECK(error.empty());
  CHECK(result.total_units == 8);  // 256/64 * 128/64
  CHECK(result.header_bits == 4);
  CHECK(result.tile_bits > 0);
  CHECK(result.filter_class.size() == 64 * 32);
  int distinct = 0;
  bool seen[256] = {};
  for (std::uint8_t cls : result.filter_class) {
    if (!seen[cls]) {
      seen[cls] = true;
      ++distinct;
    }
  }
  CHECK(distinct > 1);
  CHECK(result.filtered_sse <= result.identity_sse);
  std::printf(
      "per-unit decision: units=%d active=%d class-values=%d sse %.0f -> "
      "%.0f header=%d tile=%d bits\n",
      result.total_units, result.active_units, distinct, result.identity_sse,
      result.filtered_sse, result.header_bits, result.tile_bits);

  // The parameters and units round-trip through an independent reader.
  const pcw::LrHeaderContext context = MakePcOnlyContext();
  pcw::LrParams params = result.params;
  av2enc::BitWriter writer;
  const std::size_t bits = pcw::WriteLrParams(writer, &params, context, &error);
  CHECK(error.empty());
  CHECK(bits == static_cast<std::size_t>(result.header_bits));
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  pcw::LrParams parsed;
  pcw::Reader reader(bytes.data(), bits);
  CHECK(pcw::ReadLrParams(reader, context, &parsed, &error));
  CHECK(parsed.frame_type[0] == pcw::kRestorePcWiener);

  pcw::TileState write_state;
  pcw::TileState read_state;
  CHECK(pcw::InitTileState(&write_state, params, 64, 32, 1, 1, &error));
  CHECK(pcw::InitTileState(&read_state, parsed, 64, 32, 1, 1, &error));
  EncoderBuffer enc;
  CHECK(StartEncoder(&enc));
  const int unit_cols = write_state.unit_cols[0];
  const int unit_rows = write_state.unit_rows[0];
  for (int r = 0; r < unit_rows; ++r) {
    for (int c = 0; c < unit_cols; ++c) {
      pcw::UnitDecision decision;
      decision.type =
          result.unit_type[0][static_cast<std::size_t>(r) * unit_cols + c];
      CHECK(pcw::WriteUnit(enc.encoder, write_state, 0, r, c, decision,
                           &error));
    }
  }
  std::size_t payload = 0;
  CHECK(enc.encoder.exit_symbol(&payload) == av2enc::av2::Status::kOk);
  CHECK(static_cast<int>(payload * 8) == result.tile_bits);
  av2enc::av2::Decoder decoder;
  CHECK(decoder.init_symbol(enc.bytes.data(), enc.bytes.size(), payload) ==
        av2enc::av2::Status::kOk);
  for (int r = 0; r < unit_rows; ++r) {
    for (int c = 0; c < unit_cols; ++c) {
      pcw::UnitDecision decoded;
      CHECK(pcw::ReadUnit(decoder, read_state, 0, r, c, &decoded, &error));
      CHECK(decoded.type ==
            result.unit_type[0][static_cast<std::size_t>(r) * unit_cols + c]);
    }
  }
}

void TestDecisionFrameClass() {
  const SyntheticFrame frame = MakeSynthetic(256, 128);
  const pcw::DecisionPlanes planes = MakePlanes(frame);
  pcw::DecisionResult result;
  std::string error;
  CHECK(pcw::DecideFrameClassPcWiener(planes, 16, 0.0, &result, &error));
  CHECK(error.empty());
  CHECK(result.used_frame_class_mode);
  CHECK(result.num_filter_classes == 16);
  CHECK(result.header_bits > 5);
  CHECK(result.tile_bits > 0);
  CHECK(result.subclass_lookup_count == pcw::kNumLutClasses);
  int translated_matches = 0;
  for (int c = 0; c < 16; ++c) {
    CHECK(result.params.frame_filter[0][c].merged);
    CHECK(result.params.frame_filter[0][c].match_index >= 16);
    for (int j = 0; j < pcw::kWienerNsCoeffStride; ++j) {
      translated_matches +=
          result.params.frame_coeffs[0][c][j] !=
          pcw::TranslatedPcWiener(
              result.params.frame_filter[0][c].match_index - 16, j);
    }
  }
  CHECK(translated_matches == 0);
  std::printf(
      "frame-class decision: classes=%d units=%d active=%d sse %.0f -> %.0f "
      "header=%d tile=%d bits\n",
      result.num_filter_classes, result.total_units, result.active_units,
      result.identity_sse, result.filtered_sse, result.header_bits,
      result.tile_bits);

  pcw::LrHeaderContext context = MakeContext();
  context.lr_tools_disable[1][pcw::kRestoreWienerNs] =
      context.lr_tools_disable[0][pcw::kRestoreWienerNs];
  pcw::LrParams params = result.params;
  av2enc::BitWriter writer;
  const std::size_t bits = pcw::WriteLrParams(writer, &params, context, &error);
  CHECK(error.empty());
  CHECK(bits == static_cast<std::size_t>(result.header_bits));
  const std::vector<std::uint8_t> bytes(writer.bytes().begin(),
                                        writer.bytes().end());
  pcw::LrParams parsed;
  pcw::Reader reader(bytes.data(), bits);
  CHECK(pcw::ReadLrParams(reader, context, &parsed, &error));
  CHECK(parsed.num_filter_classes == 16);
  CHECK(parsed.frame_filters_on[0]);
  CHECK(SameParams(params, parsed));
}

void TestApplyPerUnitMatchesReference() {
  const SyntheticFrame frame = MakeSynthetic(128, 64);
  const pcw::DecisionPlanes planes = MakePlanes(frame);
  const int unit_size = 64;
  const int unit_cols = 2;
  const int unit_rows = 1;
  std::vector<std::uint8_t> types(static_cast<std::size_t>(unit_cols * unit_rows),
                                  pcw::kRestoreNone);
  types[0] = pcw::kRestorePcWiener;
  std::vector<std::uint8_t> out(static_cast<std::size_t>(frame.width) *
                                    frame.height,
                                0);
  const int blocks = pcw::ApplyPerUnitPcWiener(planes, types, unit_size, out.data(),
                                               frame.width);
  CHECK(blocks > 0);
  // In the inactive unit the output is CdefFrame.
  int inactive_diffs = 0;
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 64; x < frame.width; ++x) {
      inactive_diffs +=
          out[static_cast<std::size_t>(y) * frame.width + x] !=
          frame.cdef[static_cast<std::size_t>(y) * frame.width + x];
    }
  }
  CHECK(inactive_diffs == 0);
  // In the active unit at least one sample changed (the bank is non-identity).
  int active_changed = 0;
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 0; x < 64; ++x) {
      active_changed += out[static_cast<std::size_t>(y) * frame.width + x] !=
                        frame.cdef[static_cast<std::size_t>(y) * frame.width + x];
    }
  }
  CHECK(active_changed > 0);
}

}  // namespace

int main() {
  TestSequenceConfig();
  TestLrParamsPcWiener();
  TestLrParamsFrameClass();
  TestLrParamsClassRefs();
  TestLrParamsSwitchableAndChroma();
  TestUnitPcWiener();
  TestUnitSwitchableAndBanks();
  TestUnitFrameFiltersOn();
  TestDecisionPerUnit();
  TestDecisionFrameClass();
  TestApplyPerUnitMatchesReference();
  return test::Finish("pc_wiener_stream_test");
}
