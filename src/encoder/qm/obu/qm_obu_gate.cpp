// Decoder-gate helper for the §5.13 quantizer-matrix OBU state machine.
//
// The M1/M2 frame-header writers in this tree hardcode using_qmatrix = 0, so
// this tool bit-splices a §5.18.6.2 QM frame-header section and zero or more
// §5.13 QM OBUs into an existing one-frame M1 IVF stream.  It does not edit the
// M1 core.  The tile payload is copied byte-for-byte.
//
// Every variant is also driven through av2enc::qm::obu::QmState in stream
// order (temporal delimiter -> QM OBUs -> key frame) and the resulting
// QmDataPresent/QmProtected state is checked before the stream is written.
//
// Modes:
//   identity         one user-defined QM OBU (all-32 == identity), level L
//   default          one qm_is_default_flag = 1 OBU, level L
//   no-obu           no QM OBU; the frame header selects level L and the
//                    normative default matrices apply
//   reset-identity   qm_bit_map = 0 OBU, then the identity OBU for level L
//   multi            reset OBU, a default OBU for a different level, then the
//                    identity OBU for level L
//
// The identity matrix satisfies Round2(q*32,5) == q, and an 8x8 4:2:0 picture
// only has 8x8/4x4 transforms (the only sizes where user QM data is used), so
// the decoded pixels must match the base stream exactly.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/qm/obu/qm_obu.h"

namespace {

using av2enc::qm::obu::FrameQm;
using av2enc::qm::obu::ObuHeader;
using av2enc::qm::obu::ParsedObu;
using av2enc::qm::obu::QmState;
using av2enc::qm::obu::QuantizerMatrixObu;

struct Options {
  std::string input;
  std::string output;
  int width = 8;
  int height = 8;
  int qp = 90;
  int level = 0;
  bool no_deblock = false;
  std::string mode = "identity";
  bool selftest = false;
};

void usage() {
  std::fprintf(
      stderr,
      "usage: qm_obu_gate <base.ivf> <out.ivf> [--width W] [--height H]\n"
      "                   [--qp Q] [--level 0..14] [--no-deblock]\n"
      "                   [--mode identity|default|no-obu|reset-identity|multi]\n"
      "       qm_obu_gate --selftest\n");
}

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}
  bool bit(bool* value) {
    if (position_ >= data_.size() * 8U) return false;
    *value = ((data_[position_ >> 3U] >> (7U - (position_ & 7U))) & 1U) != 0U;
    ++position_;
    return true;
  }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t position_ = 0;
};

bool build_frame_config(const Options& options, av2enc::m1::M1Config* config) {
  config->max_frame_width = static_cast<std::uint16_t>(options.width);
  config->max_frame_height = static_cast<std::uint16_t>(options.height);
  config->base_q_idx = static_cast<std::uint8_t>(options.qp);
  config->disable_cdf_update = true;
  config->enable_intra_edge_filter = true;
  config->enable_cfl_intra = true;
  config->enable_chroma_dctonly = true;
  config->enable_idtx_intra = false;
  config->tx_mode_select = false;
  config->reduced_tx_set = 0;
  const bool deblock = !options.no_deblock;
  config->apply_deblock_y_vert = deblock;
  config->apply_deblock_y_horz = deblock;
  config->apply_deblock_u = deblock;
  config->apply_deblock_v = deblock;
  return true;
}

// Bits before setup_qm_params()'s using_qmatrix bit, including the leading
// is_first_tile_group bit.  This mirrors the ordering in
// m1::write_frame_header(); it is also what src/encoder/qm/qm_gate.cpp uses.
std::size_t pre_qm_bits(const av2enc::m1::M1Config& config) {
  const std::uint32_t mi_cols =
      2U * ((static_cast<std::uint32_t>(config.max_frame_width) + 7U) >> 3U);
  const std::uint32_t mi_rows =
      2U * ((static_cast<std::uint32_t>(config.max_frame_height) + 7U) >> 3U);
  const std::uint32_t sb_cols = (mi_cols + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows + 15U) >> 4U;
  const std::size_t tile_bits =
      1U + (sb_cols > 1U ? 1U : 0U) + (sb_rows > 1U ? 1U : 0U);
  // is_first_tile_group + cur_mfh_id uvlc(0) + seq_header_id uvlc(0) +
  // allow_screen_content_tools + allow_intrabc + disable_cdf_update +
  // tile_info + base_q_idx f(8) + segmentation_enabled.
  return 1U + 5U + tile_bits + 8U + 1U;
}

QuantizerMatrixObu make_level_obu(int level, bool is_default) {
  QuantizerMatrixObu obu;
  obu.bit_map = static_cast<std::uint16_t>(1U << level);
  obu.chroma_info_present = true;
  obu.levels[static_cast<std::size_t>(level)].is_default = is_default;
  if (!is_default) {
    std::memset(&obu.levels[static_cast<std::size_t>(level)].user, 32,
                sizeof(obu.levels[static_cast<std::size_t>(level)].user));
  }
  return obu;
}

bool rewrite_clk_payload(const ParsedObu& clk, const Options& options,
                         const FrameQm& frame_qm,
                         std::vector<std::uint8_t>* out_payload,
                         std::string* error) {
  av2enc::m1::M1Config config;
  build_frame_config(options, &config);

  av2enc::BitWriter probe;
  const std::size_t syntax_bits =
      av2enc::m1::write_frame_header(probe, config);
  const std::size_t original_header_bits = 1U + syntax_bits;
  const std::size_t pre = pre_qm_bits(config);
  if (pre + 1U > original_header_bits) {
    *error = "frame-header probe is shorter than the using_qmatrix field";
    return false;
  }

  BitReader reader(clk.payload);
  av2enc::BitWriter writer;
  for (std::size_t i = 0; i < pre; ++i) {
    bool bit = false;
    if (!reader.bit(&bit)) {
      *error = "CLK payload truncated before using_qmatrix";
      return false;
    }
    writer.put_bit(bit);
  }
  bool original_using_qmatrix = true;
  if (!reader.bit(&original_using_qmatrix)) {
    *error = "CLK payload truncated at using_qmatrix";
    return false;
  }
  if (original_using_qmatrix) {
    *error = "base stream already has using_qmatrix = 1";
    return false;
  }
  if (!av2enc::qm::obu::write_frame_qm_params(writer, frame_qm, 3, false,
                                              error)) {
    return false;
  }
  for (std::size_t i = pre + 1U; i < original_header_bits; ++i) {
    bool bit = false;
    if (!reader.bit(&bit)) {
      *error = "CLK payload truncated in frame header";
      return false;
    }
    writer.put_bit(bit);
  }
  writer.align_to_byte();

  const auto header_bytes = writer.bytes();
  out_payload->assign(header_bytes.begin(), header_bytes.end());

  const std::size_t tile_start = (original_header_bits + 7U) / 8U;
  if (tile_start > clk.payload.size()) {
    *error = "CLK payload shorter than its frame header";
    return false;
  }
  out_payload->insert(
      out_payload->end(),
      clk.payload.begin() + static_cast<std::ptrdiff_t>(tile_start),
      clk.payload.end());
  return true;
}

bool build_variant(const Options& options, std::vector<std::vector<std::uint8_t>>* obus,
                   FrameQm* frame_qm, bool* expect_data_present,
                   std::string* error) {
  obus->clear();
  *frame_qm = FrameQm{};
  frame_qm->using_qmatrix = true;
  const auto level = static_cast<std::size_t>(options.level);
  frame_qm->qm_y[0] = static_cast<std::uint8_t>(options.level);
  frame_qm->qm_u[0] = static_cast<std::uint8_t>(options.level);
  frame_qm->qm_v[0] = static_cast<std::uint8_t>(options.level);
  frame_qm->qm_uv_same_as_y[0] = true;

  auto emit = [&](const QuantizerMatrixObu& obu) {
    std::vector<std::uint8_t> bytes;
    if (!av2enc::qm::obu::make_quantizer_matrix_obu(obu, nullptr, &bytes,
                                                    error)) {
      return false;
    }
    obus->push_back(std::move(bytes));
    return true;
  };

  if (options.mode == "no-obu") {
    *expect_data_present = false;
  } else if (options.mode == "identity") {
    if (!emit(make_level_obu(options.level, false))) return false;
    *expect_data_present = true;
  } else if (options.mode == "default") {
    if (!emit(make_level_obu(options.level, true))) return false;
    *expect_data_present = false;
  } else if (options.mode == "reset-identity") {
    QuantizerMatrixObu reset;
    reset.bit_map = 0;
    reset.chroma_info_present = true;
    if (!emit(reset)) return false;
    if (!emit(make_level_obu(options.level, false))) return false;
    *expect_data_present = true;
  } else if (options.mode == "multi") {
    QuantizerMatrixObu reset;
    reset.bit_map = 0;
    reset.chroma_info_present = true;
    if (!emit(reset)) return false;
    const int other = (options.level + 7) % 15;
    if (!emit(make_level_obu(other, true))) return false;
    if (!emit(make_level_obu(options.level, false))) return false;
    *expect_data_present = true;
  } else {
    *error = "unknown --mode";
    return false;
  }
  (void)level;
  return true;
}

// Drives the stream-order state machine and verifies the selected level.
bool verify_state(const std::vector<std::vector<std::uint8_t>>& obus,
                  int selected_level, bool expect_data_present,
                  std::string* error) {
  QmState state;
  state.on_temporal_delimiter();
  for (const auto& bytes : obus) {
    if (!state.apply_obu_bytes(bytes, error)) return false;
  }
  state.on_key_frame(true, 3);
  if (state.data_present(selected_level) != expect_data_present) {
    *error = "QmDataPresent for the selected level does not match the mode";
    return false;
  }
  if (expect_data_present && state.user_qm(selected_level) == nullptr) {
    *error = "user QM data is missing for a data-present level";
    return false;
  }
  if (!expect_data_present && state.user_qm(selected_level) != nullptr) {
    *error = "default/no-OBU level unexpectedly has user QM data";
    return false;
  }
  // Every level touched by an OBU is protected, so reset_qm() preserved it.
  for (const auto& bytes : obus) {
    std::vector<ParsedObu> parsed;
    if (!av2enc::qm::obu::parse_obu_sequence(bytes, &parsed, error)) {
      return false;
    }
    QuantizerMatrixObu matrix;
    if (!av2enc::qm::obu::read_quantizer_matrix_payload(parsed[0].payload,
                                                        &matrix, error)) {
      return false;
    }
    if (matrix.bit_map == 0) continue;
    for (int level = 0; level < av2enc::qm::obu::kNumCustomQms; ++level) {
      if ((matrix.bit_map & (1U << level)) == 0U) continue;
      if (!state.protected_level(level)) {
        *error = "a specified level lost QmProtected across the key frame";
        return false;
      }
    }
  }
  return true;
}

bool read_file(const std::string& path, std::vector<std::uint8_t>* out) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) return false;
  std::uint8_t buffer[65536];
  std::size_t got = 0;
  while ((got = std::fread(buffer, 1, sizeof(buffer), file)) != 0) {
    out->insert(out->end(), buffer, buffer + got);
  }
  std::fclose(file);
  return true;
}

bool write_file(const std::string& path, std::span<const std::uint8_t> data) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) return false;
  const bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size();
  std::fclose(file);
  return ok;
}

bool run(const Options& options) {
  std::vector<std::uint8_t> ivf;
  if (!read_file(options.input, &ivf)) {
    std::fprintf(stderr, "qm_obu_gate: cannot read %s\n",
                 options.input.c_str());
    return false;
  }
  if (ivf.size() < 44 || std::memcmp(ivf.data(), "DKIF", 4) != 0) {
    std::fprintf(stderr, "qm_obu_gate: %s is not an IVF file\n",
                 options.input.c_str());
    return false;
  }
  const std::uint32_t frame_count =
      static_cast<std::uint32_t>(ivf[24]) |
      (static_cast<std::uint32_t>(ivf[25]) << 8U) |
      (static_cast<std::uint32_t>(ivf[26]) << 16U) |
      (static_cast<std::uint32_t>(ivf[27]) << 24U);
  if (frame_count != 1) {
    std::fprintf(stderr, "qm_obu_gate: expected a one-frame IVF, got %u\n",
                 frame_count);
    return false;
  }
  const std::uint32_t frame_size =
      static_cast<std::uint32_t>(ivf[32]) |
      (static_cast<std::uint32_t>(ivf[33]) << 8U) |
      (static_cast<std::uint32_t>(ivf[34]) << 16U) |
      (static_cast<std::uint32_t>(ivf[35]) << 24U);
  if (44U + frame_size > ivf.size()) {
    std::fprintf(stderr, "qm_obu_gate: truncated IVF frame\n");
    return false;
  }
  std::uint8_t timestamp[8];
  std::memcpy(timestamp, ivf.data() + 36, sizeof(timestamp));
  const std::span<const std::uint8_t> frame(
      ivf.data() + 44, static_cast<std::size_t>(frame_size));

  std::vector<ParsedObu> input_obus;
  std::string error;
  if (!av2enc::qm::obu::parse_obu_sequence(frame, &input_obus, &error)) {
    std::fprintf(stderr, "qm_obu_gate: %s\n", error.c_str());
    return false;
  }

  std::vector<std::vector<std::uint8_t>> qm_obus;
  FrameQm frame_qm;
  bool expect_data_present = false;
  if (!build_variant(options, &qm_obus, &frame_qm, &expect_data_present,
                     &error)) {
    std::fprintf(stderr, "qm_obu_gate: %s\n", error.c_str());
    return false;
  }
  if (!verify_state(qm_obus, options.level, expect_data_present, &error)) {
    std::fprintf(stderr, "qm_obu_gate: state check: %s\n", error.c_str());
    return false;
  }

  std::vector<std::uint8_t> new_frame;
  bool inserted = false;
  for (const ParsedObu& obu : input_obus) {
    if (obu.header.type == av2enc::qm::obu::kObuTypeClosedLoopKey) {
      std::vector<std::uint8_t> payload;
      if (!rewrite_clk_payload(obu, options, frame_qm, &payload, &error)) {
        std::fprintf(stderr, "qm_obu_gate: %s\n", error.c_str());
        return false;
      }
      std::vector<std::uint8_t> changed;
      if (!av2enc::qm::obu::frame_obu(obu.header, payload, &changed, &error)) {
        std::fprintf(stderr, "qm_obu_gate: %s\n", error.c_str());
        return false;
      }
      new_frame.insert(new_frame.end(), changed.begin(), changed.end());
    } else {
      new_frame.insert(new_frame.end(), obu.raw.begin(), obu.raw.end());
    }
    if (obu.header.type == 1 && !inserted) {  // after the sequence header
      for (const auto& bytes : qm_obus) {
        new_frame.insert(new_frame.end(), bytes.begin(), bytes.end());
      }
      inserted = true;
    }
  }
  if (!inserted) {
    std::fprintf(stderr, "qm_obu_gate: no sequence header found\n");
    return false;
  }

  std::vector<std::uint8_t> out(ivf.begin(), ivf.begin() + 32);
  const std::uint32_t new_size = static_cast<std::uint32_t>(new_frame.size());
  out.push_back(static_cast<std::uint8_t>(new_size & 0xffU));
  out.push_back(static_cast<std::uint8_t>((new_size >> 8U) & 0xffU));
  out.push_back(static_cast<std::uint8_t>((new_size >> 16U) & 0xffU));
  out.push_back(static_cast<std::uint8_t>((new_size >> 24U) & 0xffU));
  out.insert(out.end(), timestamp, timestamp + 8);
  out.insert(out.end(), new_frame.begin(), new_frame.end());

  if (!write_file(options.output, out)) {
    std::fprintf(stderr, "qm_obu_gate: cannot write %s\n",
                 options.output.c_str());
    return false;
  }
  std::printf("qm_obu_gate: wrote %s (%zu bytes, mode %s, level %d)\n",
              options.output.c_str(), out.size(), options.mode.c_str(),
              options.level);
  return true;
}

bool selftest() {
  std::string error;
  int failures = 0;
  auto expect_reject = [&](bool accepted, const char* what) {
    if (accepted) {
      std::fprintf(stderr, "FAIL: %s was accepted\n", what);
      ++failures;
    }
  };

  // Duplicate level between coded frames.
  {
    QmState state;
    QuantizerMatrixObu obu = make_level_obu(3, true);
    if (!state.apply_obu(ObuHeader{}, obu, &error)) {
      std::fprintf(stderr, "FAIL: setup level 3: %s\n", error.c_str());
      ++failures;
    }
    expect_reject(state.apply_obu(ObuHeader{}, obu, &error),
                  "duplicate level");
    state.on_frame_header();
    if (!state.apply_obu(ObuHeader{}, obu, &error)) {
      std::fprintf(stderr, "FAIL: level reuse after a frame header: %s\n",
                   error.c_str());
      ++failures;
    }
  }
  // Two reset OBUs in one window.
  {
    QmState state;
    QuantizerMatrixObu reset;
    reset.bit_map = 0;
    reset.chroma_info_present = true;
    if (!state.apply_obu(ObuHeader{}, reset, &error)) {
      std::fprintf(stderr, "FAIL: first reset OBU: %s\n", error.c_str());
      ++failures;
    }
    expect_reject(state.apply_obu(ObuHeader{}, reset, &error),
                  "second qm_bit_map == 0 OBU");
  }
  // Writer rejections.
  {
    av2enc::BitWriter writer;
    QuantizerMatrixObu obu;
    obu.bit_map = 0x8000;
    expect_reject(av2enc::qm::obu::write_quantizer_matrix_payload(
                      writer, obu, &error),
                  "qm_bit_map bit 15");
    writer.clear();
    obu.bit_map = 1;
    obu.chroma_info_present = false;
    obu.levels[0].is_default = false;
    std::memset(&obu.levels[0].user, 32, sizeof(obu.levels[0].user));
    obu.levels[0].user.m[0][0][0][0] = 0;
    expect_reject(av2enc::qm::obu::write_quantizer_matrix_payload(
                      writer, obu, &error),
                  "zero user QM coefficient");
  }
  // Frame-header rejections.
  {
    av2enc::BitWriter writer;
    FrameQm qm;
    qm.using_qmatrix = true;
    qm.qm_y[0] = 4;
    qm.qm_u[0] = 5;
    qm.qm_v[0] = 5;
    qm.qm_uv_same_as_y[0] = true;
    expect_reject(av2enc::qm::obu::write_frame_qm_params(
                      writer, qm, 3, false, &error),
                  "qm_uv_same_as_y with differing qm_u");
  }
  if (failures != 0) return false;
  std::printf("qm_obu selftest: OK\n");
  return true;
}

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < -100000 || parsed > 100000) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    }
    if (arg == "--selftest") {
      options.selftest = true;
    } else if (arg == "--width" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.width)) return 2;
    } else if (arg == "--height" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.height)) return 2;
    } else if (arg == "--qp" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.qp)) return 2;
    } else if (arg == "--level" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.level)) return 2;
    } else if (arg == "--no-deblock") {
      options.no_deblock = true;
    } else if (arg == "--mode" && i + 1 < argc) {
      options.mode = argv[++i];
    } else if (arg.rfind("--", 0) == 0) {
      usage();
      return 2;
    } else {
      positional.push_back(arg);
    }
  }
  if (options.selftest) {
    if (!positional.empty()) {
      usage();
      return 2;
    }
    return selftest() ? 0 : 1;
  }
  if (positional.size() != 2) {
    usage();
    return 2;
  }
  options.input = positional[0];
  options.output = positional[1];
  if (options.level < 0 || options.level > 14) {
    std::fprintf(stderr, "qm_obu_gate: --level must be 0..14\n");
    return 2;
  }
  return run(options) ? 0 : 1;
}
