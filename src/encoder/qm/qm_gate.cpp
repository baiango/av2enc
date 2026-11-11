// Decoder gate helper: splice a §5.13 quantizer-matrix OBU and a §5.18.6.2
// using_qmatrix = 1 frame header into an existing M1 still-picture IVF stream.
//
// Why bit splicing: the M1 frame-header writer in src/encoder/m1/ is owned by
// another agent and hardcodes using_qmatrix = 0.  This tool does not edit it.
// It parses the one-frame IVF/Annex-B stream, rewrites only the CLK OBU's frame
// header (keeping the tile payload byte-identical), inserts the QM OBU after
// the sequence header, and writes a new IVF.
//
// The identity gate uses an all-32 user QM for level L.  Since
// q2 = Round2(q * 32, 5) == q, and an 8x8 picture only has 8x8/4x4 transforms
// (the only sizes that use user QM data), the decoded pixels must be identical
// to the base stream.  A default-matrix level is also available to prove that
// dav2d accepts the using_qmatrix path when the matrices are not identity.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <cstring>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/qm/qm.h"

namespace {

struct Options {
  std::string input;
  std::string output;
  int width = 8;
  int height = 8;
  int qp = 90;
  int level = 0;
  bool identity = true;
  bool no_deblock = false;
  bool emit_qm_obu = true;
};

void usage() {
  std::fprintf(
      stderr,
      "usage: qm_gate <base.ivf> <out.ivf> [--width W] [--height H] [--qp Q]\n"
      "               [--level 0..14] [--default] [--no-deblock]\n"
      "               [--no-qm-obu]\n"
      "\n"
      "  --default   signal qm_is_default_flag = 1 instead of user QM data\n"
      "              (decode must succeed; pixels are expected to change)\n"
      "  --no-qm-obu omit the QM OBU and rely on the normative default matrices\n");
}

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> data) : data_(data) {}
  bool bit(bool* value) {
    if (position_ >= data_.size() * 8) return false;
    *value = ((data_[position_ / 8] >> (7 - (position_ % 8))) & 1U) != 0U;
    ++position_;
    return true;
  }
  std::size_t position() const { return position_; }

 private:
  std::span<const std::uint8_t> data_;
  std::size_t position_ = 0;
};

bool read_leb128(std::span<const std::uint8_t> data, std::size_t* pos,
                 std::uint32_t* value) {
  std::uint32_t result = 0;
  for (int i = 0; i < 8; ++i) {
    if (*pos >= data.size()) return false;
    const std::uint8_t byte = data[(*pos)++];
    result |= static_cast<std::uint32_t>(byte & 0x7f) << (7 * i);
    if ((byte & 0x80) == 0) {
      *value = result;
      return true;
    }
  }
  return false;
}

void append_leb128(std::vector<std::uint8_t>* out, std::uint32_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7f);
    value >>= 7;
    if (value != 0) byte |= 0x80;
    out->push_back(byte);
  } while (value != 0);
}

struct Obu {
  std::uint8_t type = 0;
  std::vector<std::uint8_t> payload;
};

bool parse_obus(std::span<const std::uint8_t> frame, std::vector<Obu>* out,
                std::string* error) {
  std::size_t pos = 0;
  while (pos < frame.size()) {
    std::uint32_t size = 0;
    if (!read_leb128(frame, &pos, &size)) {
      *error = "bad OBU leb128 size";
      return false;
    }
    if (size == 0 || pos + size > frame.size()) {
      *error = "OBU size outside frame";
      return false;
    }
    const std::size_t start = pos;
    const std::uint8_t header = frame[pos];
    const bool extension = (header & 0x80U) != 0;
    const int type = (header >> 2) & 0x1f;
    if (extension || (header & 0x3U) != 0U) {
      *error = "gate only supports non-extension, tlayer 0 OBUs";
      return false;
    }
    const std::size_t header_bytes = 1 + (extension ? 1 : 0);
    if (start + header_bytes > start + size) {
      *error = "OBU header exceeds payload";
      return false;
    }
    Obu obu;
    obu.type = static_cast<std::uint8_t>(type);
    obu.payload.assign(frame.begin() + static_cast<std::ptrdiff_t>(start +
                                                                 header_bytes),
                       frame.begin() + static_cast<std::ptrdiff_t>(start + size));
    out->push_back(std::move(obu));
    pos = start + size;
  }
  return true;
}

std::vector<std::uint8_t> make_obu_bytes(const Obu& obu) {
  std::vector<std::uint8_t> out;
  append_leb128(&out, static_cast<std::uint32_t>(obu.payload.size() + 1));
  const std::uint8_t header =
      static_cast<std::uint8_t>(obu.type << 2);  // no extension, tlayer 0
  out.push_back(header);
  out.insert(out.end(), obu.payload.begin(), obu.payload.end());
  return out;
}

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

// Number of frame-header syntax bits before setup_qm_params()'s
// using_qmatrix bit (is_first_tile_group included).
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

bool rewrite_clk_payload(const Obu& clk, const Options& options,
                         const av2enc::qm::QmFrameState& state,
                         std::vector<std::uint8_t>* out_payload,
                         std::string* error) {
  av2enc::m1::M1Config config;
  build_frame_config(options, &config);

  av2enc::BitWriter probe;
  const std::size_t syntax_bits = av2enc::m1::write_frame_header(probe, config);
  const std::size_t original_header_bits = 1U + syntax_bits;
  const std::size_t pre = pre_qm_bits(config);
  if (pre + 1 > original_header_bits) {
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
  // setup_qm_params() writes the using_qmatrix bit itself.
  av2enc::BitWriter qm_bits;
  if (!av2enc::qm::write_setup_qm_params(qm_bits, state, 3, false, error)) {
    return false;
  }
  for (std::size_t i = 0; i < qm_bits.bit_size(); ++i) {
    writer.put_bit(((qm_bits.bytes()[i / 8] >> (7 - (i % 8))) & 1U) != 0U);
  }

  // Original bits after the using_qmatrix bit, through the end of
  // frame_header() syntax.
  for (std::size_t i = pre + 1; i < original_header_bits; ++i) {
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
  out_payload->insert(out_payload->end(),
                     clk.payload.begin() +
                         static_cast<std::ptrdiff_t>(tile_start),
                     clk.payload.end());
  return true;
}

bool run(const Options& options) {
  std::FILE* file = std::fopen(options.input.c_str(), "rb");
  if (file == nullptr) {
    std::fprintf(stderr, "qm_gate: cannot open %s\n", options.input.c_str());
    return false;
  }
  std::vector<std::uint8_t> ivf;
  std::uint8_t buffer[65536];
  std::size_t got = 0;
  while ((got = std::fread(buffer, 1, sizeof(buffer), file)) != 0) {
    ivf.insert(ivf.end(), buffer, buffer + got);
  }
  std::fclose(file);

  if (ivf.size() < 44 || std::memcmp(ivf.data(), "DKIF", 4) != 0) {
    std::fprintf(stderr, "qm_gate: %s is not an IVF file\n",
                 options.input.c_str());
    return false;
  }
  const std::uint32_t frame_count =
      static_cast<std::uint32_t>(ivf[24]) |
      (static_cast<std::uint32_t>(ivf[25]) << 8U) |
      (static_cast<std::uint32_t>(ivf[26]) << 16U) |
      (static_cast<std::uint32_t>(ivf[27]) << 24U);
  if (frame_count != 1) {
    std::fprintf(stderr, "qm_gate: expected a one-frame IVF, got %u\n",
                 frame_count);
    return false;
  }
  const std::uint32_t frame_size =
      static_cast<std::uint32_t>(ivf[32]) |
      (static_cast<std::uint32_t>(ivf[33]) << 8U) |
      (static_cast<std::uint32_t>(ivf[34]) << 16U) |
      (static_cast<std::uint32_t>(ivf[35]) << 24U);
  if (44U + frame_size > ivf.size()) {
    std::fprintf(stderr, "qm_gate: truncated IVF frame\n");
    return false;
  }
  std::uint8_t timestamp[8];
  std::memcpy(timestamp, ivf.data() + 36, sizeof(timestamp));
  const std::span<const std::uint8_t> frame(
      ivf.data() + 44, static_cast<std::size_t>(frame_size));

  std::vector<Obu> obus;
  std::string error;
  if (!parse_obus(frame, &obus, &error)) {
    std::fprintf(stderr, "qm_gate: %s\n", error.c_str());
    return false;
  }

  // QM configuration: one level, three planes.
  av2enc::qm::QmObuConfig obu_config;
  obu_config.bit_map = static_cast<std::uint16_t>(1U << options.level);
  obu_config.chroma_info_present = true;
  obu_config.levels[options.level].is_default = !options.identity;
  if (options.identity) {
    if (!av2enc::qm::fill_user_qm_constant(
            &obu_config.levels[options.level].user, 32)) {
      std::fprintf(stderr, "qm_gate: failed to build identity QM\n");
      return false;
    }
  }
  const std::vector<std::uint8_t> qm_obu =
      av2enc::qm::make_quantizer_matrix_obu(obu_config, &error);
  if (qm_obu.empty()) {
    std::fprintf(stderr, "qm_gate: %s\n", error.c_str());
    return false;
  }

  av2enc::qm::QmFrameState state;
  state.using_qmatrix = true;
  state.pic_qm_num_minus_1 = 0;
  state.qm_y[0] = static_cast<std::uint8_t>(options.level);
  state.qm_u[0] = static_cast<std::uint8_t>(options.level);
  state.qm_v[0] = static_cast<std::uint8_t>(options.level);
  state.seg_qm_level[0][0] = static_cast<std::uint8_t>(options.level);
  state.seg_qm_level[1][0] = static_cast<std::uint8_t>(options.level);
  state.seg_qm_level[2][0] = static_cast<std::uint8_t>(options.level);
  state.qm_data_present[options.level] = options.identity;

  std::vector<std::uint8_t> new_frame;
  bool inserted_qm = false;
  for (const Obu& obu : obus) {
    if (obu.type == 4) {  // OBU_CLOSED_LOOP_KEY
      std::vector<std::uint8_t> payload;
      if (!rewrite_clk_payload(obu, options, state, &payload, &error)) {
        std::fprintf(stderr, "qm_gate: %s\n", error.c_str());
        return false;
      }
      Obu changed = obu;
      changed.payload = std::move(payload);
      const auto bytes = make_obu_bytes(changed);
      new_frame.insert(new_frame.end(), bytes.begin(), bytes.end());
    } else {
      const auto bytes = make_obu_bytes(obu);
      new_frame.insert(new_frame.end(), bytes.begin(), bytes.end());
    }
    if (obu.type == 1 && !inserted_qm) {  // after the sequence header
      if (options.emit_qm_obu) {
        new_frame.insert(new_frame.end(), qm_obu.begin(), qm_obu.end());
      }
      inserted_qm = true;
    }
  }
  if (!inserted_qm) {
    std::fprintf(stderr, "qm_gate: no sequence header found\n");
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

  file = std::fopen(options.output.c_str(), "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "qm_gate: cannot write %s\n", options.output.c_str());
    return false;
  }
  const bool ok = std::fwrite(out.data(), 1, out.size(), file) == out.size();
  std::fclose(file);
  if (!ok) {
    std::fprintf(stderr, "qm_gate: short write\n");
    return false;
  }
  std::printf("qm_gate: wrote %s (%zu bytes, level %d, %s)\n",
              options.output.c_str(), out.size(), options.level,
              options.identity ? "user identity QM" : "default QM");
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
    if (arg == "--width" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.width)) {
        usage();
        return 2;
      }
    } else if (arg == "--height" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.height)) {
        usage();
        return 2;
      }
    } else if (arg == "--qp" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.qp)) {
        usage();
        return 2;
      }
    } else if (arg == "--level" && i + 1 < argc) {
      if (!parse_int(argv[++i], &options.level)) {
        usage();
        return 2;
      }
    } else if (arg == "--default") {
      options.identity = false;
    } else if (arg == "--no-deblock") {
      options.no_deblock = true;
    } else if (arg == "--no-qm-obu") {
      options.emit_qm_obu = false;
    } else if (arg.rfind("--", 0) == 0) {
      usage();
      return 2;
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2) {
    usage();
    return 2;
  }
  options.input = positional[0];
  options.output = positional[1];
  if (options.level < 0 || options.level > 14) {
    std::fprintf(stderr, "qm_gate: --level must be 0..14\n");
    return 2;
  }
  return run(options) ? 0 : 1;
}
