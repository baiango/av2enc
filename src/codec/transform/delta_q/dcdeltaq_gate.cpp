// Decoder gate harness: splice non-zero Y-DC/UV-DC/UV-AC deltas into a real M1
// still-picture IVF stream without editing the M1 headers.
//
// The base stream is an M1 still at QP Q_ref (default 8) with deblocking off.
// This tool:
//   1. rewrites the sequence header's §5.4.8 group with equal_ac_dc_q = 0 and
//      the requested base deltas (0) / enables;
//   2. rewrites the CLK frame header's §5.18.6.1 quantization_params() with a
//      new base_q_idx (default 1) and the requested frame deltas;
//   3. keeps the tile payload byte-identical.
//
// With Q_ref = 8, base_q_idx = 1 and deltas +7, every quantizer the flat
// DC-only gate frame uses is qlookup(8) = 49, exactly as in the Q_ref stream.
// The Y AC quantizer remains qlookup(1) = 40, but a flat plane has no AC
// coefficients.  dav2d's reconstruction must therefore match the reference
// stream sample for sample.
#include "core/bitstream/bitwriter.h"
#include "pipeline/still/still_headers.h"
#include "codec/transform/delta_q/dcdeltaq.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::BitWriter;
using av2enc::deltaq::BitReader;
using av2enc::dcdeltaq::FrameDeltaQ;
using av2enc::dcdeltaq::SequenceDeltaQ;

struct Options {
  std::string input;
  std::string output;
  int width = 64;
  int height = 64;
  int ref_qp = 8;
  int base_q_idx = 1;
  int y_dc = 0;
  int u_dc = 0;
  int v_dc = 0;
  int u_ac = 0;
  int v_ac = 0;
  int base_y_dc = 0;
  int base_uv_dc = 0;
  int base_uv_ac = 0;
  bool separate_uv = false;
  bool identity = false;
};

void usage() {
  std::fprintf(
      stderr,
      "usage: dcdeltaq_gate <base.ivf> <out.ivf> [options]\n"
      "  --width W --height H     frame size (default 64x64)\n"
      "  --ref-qp Q               base stream QP (default 8)\n"
      "  --base-q Q               new base_q_idx (default 1)\n"
      "  --ydc D --udc D --vdc D  frame DC deltas\n"
      "  --uac D --vac D          frame AC deltas\n"
      "  --base-ydc D --base-uvdc D --base-uvac D\n"
      "                           sequence base deltas (default 0)\n"
      "  --separate-uv            set separate_uv_delta_q = 1\n"
      "  --identity               rebuild the original headers byte-for-byte\n");
}

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
    if (!read_leb128(frame, &pos, &size) || size == 0 ||
        pos + size > frame.size()) {
      *error = "bad OBU framing";
      return false;
    }
    const std::size_t start = pos;
    const std::uint8_t header = frame[pos];
    const bool extension = (header & 0x80U) != 0;
    const int type = (header >> 2) & 0x1f;
    if (extension || (header & 0x3U) != 0U) {
      *error = "gate only supports non-extension tlayer-0 OBUs";
      return false;
    }
    Obu obu;
    obu.type = static_cast<std::uint8_t>(type);
    obu.payload.assign(
        frame.begin() + static_cast<std::ptrdiff_t>(start + 1),
        frame.begin() + static_cast<std::ptrdiff_t>(start + size));
    out->push_back(std::move(obu));
    pos = start + size;
  }
  return true;
}

std::vector<std::uint8_t> make_obu_bytes(const Obu& obu) {
  std::vector<std::uint8_t> out;
  append_leb128(&out, static_cast<std::uint32_t>(obu.payload.size() + 1));
  out.push_back(static_cast<std::uint8_t>(obu.type << 2));
  out.insert(out.end(), obu.payload.begin(), obu.payload.end());
  return out;
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1;
    ++bits;
  }
  return bits == 0 ? 1 : bits;
}

// §5.4.1-§5.4.7 prefix length before sequence_transform_quant_entropy_config()
// for the M1 tool set (4:2:0, 8-bit, single picture, fixed intra/inter tools).
std::size_t sequence_group_start_bits(int width, int height) {
  const unsigned wb = bit_width(static_cast<std::uint32_t>(width) - 1U);
  const unsigned hb = bit_width(static_cast<std::uint32_t>(height) - 1U);
  return 1U + 5U + 1U + 5U + 1U + 3U + 4U + 4U + wb + hb + 1U +
         5U + 2U + 8U + 5U;
}

bool copy_bits(BitReader& reader, BitWriter& writer, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    bool bit = false;
    if (!reader.read_bit(bit)) return false;
    writer.put_bit(bit);
  }
  return true;
}

bool skip_bits(BitReader& reader, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    bool bit = false;
    if (!reader.read_bit(bit)) return false;
  }
  return true;
}

std::size_t last_set_bit(std::span<const std::uint8_t> data) {
  for (std::size_t i = data.size(); i > 0; --i) {
    const std::uint8_t byte = data[i - 1];
    if (byte != 0) {
      for (int bit = 7; bit >= 0; --bit) {
        if (((byte >> bit) & 1U) != 0U) {
          return (i - 1) * 8 + static_cast<std::size_t>(7 - bit);
        }
      }
    }
  }
  return 0;
}

av2enc::still::StillConfig base_config(const Options& options) {
  av2enc::still::StillConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(options.width);
  config.max_frame_height = static_cast<std::uint16_t>(options.height);
  config.base_q_idx = static_cast<std::uint8_t>(options.ref_qp);
  config.disable_cdf_update = true;
  config.enable_intra_edge_filter = true;
  config.enable_cfl_intra = true;
  config.enable_chroma_dctonly = true;
  config.enable_idtx_intra = false;
  config.tx_mode_select = false;
  config.reduced_tx_set = 0;
  config.apply_deblock_y_vert = false;  // --no-deblock
  config.apply_deblock_y_horz = false;
  config.apply_deblock_u = false;
  config.apply_deblock_v = false;
  return config;
}

std::size_t pre_base_q_bits(const av2enc::still::StillConfig& config) {
  const std::uint32_t mi_cols =
      2U * ((static_cast<std::uint32_t>(config.max_frame_width) + 7U) >> 3U);
  const std::uint32_t mi_rows =
      2U * ((static_cast<std::uint32_t>(config.max_frame_height) + 7U) >> 3U);
  const std::uint32_t sb_cols = (mi_cols + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows + 15U) >> 4U;
  const std::size_t tile_bits =
      1U + (sb_cols > 1U ? 1U : 0U) + (sb_rows > 1U ? 1U : 0U);
  // is_first_tile_group + cur_mfh_id + seq_header_id_in_frame_header +
  // allow_screen_content_tools + allow_intrabc + disable_cdf_update + tile_info
  return 1U + 5U + tile_bits;
}

bool rewrite_sequence(const Obu& seq_obu, const Options& options,
                      std::vector<std::uint8_t>* out, std::string* error) {
  const std::span<const std::uint8_t> payload(seq_obu.payload);
  const std::size_t group = sequence_group_start_bits(options.width,
                                                      options.height);
  const std::size_t stop = last_set_bit(payload);
  if (group + 8 >= stop) {
    *error = "sequence payload too short";
    return false;
  }

  BitReader probe(payload);
  std::size_t skipped = 0;
  while (skipped + 32 <= group) {
    std::uint32_t ignored = 0;
    if (!probe.read_bits(32, ignored)) {
      *error = "sequence prefix truncated";
      return false;
    }
    skipped += 32;
  }
  while (skipped < group) {
    bool ignored = false;
    if (!probe.read_bit(ignored)) {
      *error = "sequence prefix truncated";
      return false;
    }
    ++skipped;
  }
  av2enc::dcdeltaq::SeqTransformQuantEntropyConfig config;
  config.monochrome = false;
  config.single_picture_header_flag = true;
  SequenceDeltaQ original_seq;
  if (!av2enc::dcdeltaq::read_sequence_transform_quant_entropy_config(
          probe, config, original_seq, error)) {
    return false;
  }
  const std::size_t group_end = probe.bit_position();

  SequenceDeltaQ seq;
  if (options.identity) {
    seq = original_seq;
  } else {
    seq.monochrome = false;
    seq.equal_ac_dc_q = false;
    seq.separate_uv_delta_q = options.separate_uv;
    seq.y_dc_delta_q_enabled = true;
    seq.uv_dc_delta_q_enabled = true;
    seq.uv_ac_delta_q_enabled = true;
    seq.base_y_dc_delta_q = options.base_y_dc;
    seq.base_uv_dc_delta_q = options.base_uv_dc;
    seq.base_uv_ac_delta_q = options.base_uv_ac;
  }

  BitWriter writer;
  BitReader prefix(payload);
  if (!copy_bits(prefix, writer, group)) {
    *error = "sequence prefix copy failed";
    return false;
  }
  const std::size_t new_group = av2enc::dcdeltaq::
      write_sequence_transform_quant_entropy_config(writer, config, seq, error);
  if (new_group == 0) return false;

  BitReader post(payload);
  if (!skip_bits(post, group_end)) {
    *error = "sequence post-header seek failed";
    return false;
  }
  if (!copy_bits(post, writer, stop - group_end)) {
    *error = "sequence post-header copy failed";
    return false;
  }
  writer.put_bit(true);  // trailing stop bit
  writer.align_to_byte();

  const auto bytes = writer.bytes();
  out->assign(bytes.begin(), bytes.end());
  return true;
}

bool rewrite_frame(const Obu& clk, const Options& options,
                   std::vector<std::uint8_t>* out, std::string* error) {
  av2enc::still::StillConfig config = base_config(options);
  BitWriter probe_writer;
  const std::size_t header_bits =
      1U + av2enc::still::write_frame_header(probe_writer, config);
  const std::size_t pre = pre_base_q_bits(config);
  if (pre + 8 > header_bits || header_bits > clk.payload.size() * 8) {
    *error = "frame header probe inconsistent with CLK payload";
    return false;
  }

  SequenceDeltaQ seq;
  FrameDeltaQ frame;
  if (options.identity) {
    seq.monochrome = false;
    seq.equal_ac_dc_q = true;
    seq.uv_ac_delta_q_enabled = false;
  } else {
    seq.monochrome = false;
    seq.equal_ac_dc_q = false;
    seq.separate_uv_delta_q = options.separate_uv;
    seq.y_dc_delta_q_enabled = true;
    seq.uv_dc_delta_q_enabled = true;
    seq.uv_ac_delta_q_enabled = true;
    frame.delta_q_y_dc = options.y_dc;
    frame.delta_q_u_dc = options.u_dc;
    frame.delta_q_u_ac = options.u_ac;
    frame.delta_q_v_dc = options.separate_uv ? options.v_dc : options.u_dc;
    frame.delta_q_v_ac = options.separate_uv ? options.v_ac : options.u_ac;
  }

  BitReader reader(clk.payload);
  BitWriter writer;
  if (!copy_bits(reader, writer, pre)) {
    *error = "CLK prefix copy failed";
    return false;
  }
  const int base_q = options.identity ? options.ref_qp : options.base_q_idx;
  if (av2enc::dcdeltaq::write_quantization_params(writer, base_q, seq, frame,
                                                  error) == 0) {
    return false;
  }
  // Skip the original 8-bit base_q_idx and keep the rest of the header.
  bool ignored = false;
  for (int i = 0; i < 8; ++i) {
    if (!reader.read_bit(ignored)) {
      *error = "CLK base_q_idx truncated";
      return false;
    }
  }
  if (!copy_bits(reader, writer, header_bits - pre - 8)) {
    *error = "CLK header copy failed";
    return false;
  }
  writer.align_to_byte();

  const auto bytes = writer.bytes();
  out->assign(bytes.begin(), bytes.end());
  const std::size_t tile_start = (header_bits + 7U) / 8U;
  if (tile_start > clk.payload.size()) {
    *error = "CLK payload shorter than its header";
    return false;
  }
  out->insert(out->end(),
              clk.payload.begin() + static_cast<std::ptrdiff_t>(tile_start),
              clk.payload.end());
  return true;
}

int coeff_cdf_slice(int q) {
  if (q <= 90) return 0;
  if (q <= 140) return 1;
  if (q <= 190) return 2;
  return 3;
}

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < -100000 || parsed > 100000) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool run(const Options& options) {
  if (!options.identity &&
      coeff_cdf_slice(options.base_q_idx) != coeff_cdf_slice(options.ref_qp)) {
    std::fprintf(stderr,
                 "dcdeltaq_gate: base_q_idx %d and ref QP %d are in different "
                 "coefficient CDF slices\n",
                 options.base_q_idx, options.ref_qp);
    return false;
  }

  std::FILE* file = std::fopen(options.input.c_str(), "rb");
  if (file == nullptr) {
    std::fprintf(stderr, "dcdeltaq_gate: cannot open %s\n",
                 options.input.c_str());
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
    std::fprintf(stderr, "dcdeltaq_gate: %s is not an IVF file\n",
                 options.input.c_str());
    return false;
  }
  const std::uint32_t frame_size =
      static_cast<std::uint32_t>(ivf[32]) |
      (static_cast<std::uint32_t>(ivf[33]) << 8U) |
      (static_cast<std::uint32_t>(ivf[34]) << 16U) |
      (static_cast<std::uint32_t>(ivf[35]) << 24U);
  if (44U + frame_size > ivf.size()) {
    std::fprintf(stderr, "dcdeltaq_gate: truncated IVF frame\n");
    return false;
  }
  const std::span<const std::uint8_t> frame(
      ivf.data() + 44, static_cast<std::size_t>(frame_size));

  std::vector<Obu> obus;
  std::string error;
  if (!parse_obus(frame, &obus, &error)) {
    std::fprintf(stderr, "dcdeltaq_gate: %s\n", error.c_str());
    return false;
  }

  std::vector<std::uint8_t> new_frame;
  bool saw_seq = false;
  bool saw_clk = false;
  for (const Obu& obu : obus) {
    Obu changed = obu;
    if (obu.type == 1) {  // OBU_SEQUENCE_HEADER
      if (!rewrite_sequence(obu, options, &changed.payload, &error)) {
        std::fprintf(stderr, "dcdeltaq_gate: %s\n", error.c_str());
        return false;
      }
      saw_seq = true;
    } else if (obu.type == 4) {  // OBU_CLOSED_LOOP_KEY
      if (!rewrite_frame(obu, options, &changed.payload, &error)) {
        std::fprintf(stderr, "dcdeltaq_gate: %s\n", error.c_str());
        return false;
      }
      saw_clk = true;
    }
    const auto bytes = make_obu_bytes(changed);
    new_frame.insert(new_frame.end(), bytes.begin(), bytes.end());
  }
  if (!saw_seq || !saw_clk) {
    std::fprintf(stderr, "dcdeltaq_gate: missing sequence header or CLK OBU\n");
    return false;
  }

  std::vector<std::uint8_t> out(ivf.begin(), ivf.begin() + 32);
  const std::uint32_t new_size = static_cast<std::uint32_t>(new_frame.size());
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::uint8_t>((new_size >> (8 * i)) & 0xffU));
  }
  out.insert(out.end(), ivf.begin() + 36, ivf.begin() + 44);
  out.insert(out.end(), new_frame.begin(), new_frame.end());

  file = std::fopen(options.output.c_str(), "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "dcdeltaq_gate: cannot write %s\n",
                 options.output.c_str());
    return false;
  }
  const bool ok = std::fwrite(out.data(), 1, out.size(), file) == out.size();
  std::fclose(file);
  if (!ok) {
    std::fprintf(stderr, "dcdeltaq_gate: short write\n");
    return false;
  }
  std::printf(
      "dcdeltaq_gate: wrote %s (%zu bytes, base_q %d, ydc %+d udc %+d vdc %+d "
      "uac %+d vac %+d, separate_uv %d)\n",
      options.output.c_str(), out.size(),
      options.identity ? options.ref_qp : options.base_q_idx, options.y_dc,
      options.u_dc, options.v_dc, options.u_ac, options.v_ac,
      options.separate_uv ? 1 : 0);
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
    } else if (arg == "--separate-uv") {
      options.separate_uv = true;
    } else if (arg == "--identity") {
      options.identity = true;
    } else if ((arg == "--width" || arg == "--height" || arg == "--ref-qp" ||
                arg == "--base-q" || arg == "--ydc" || arg == "--udc" ||
                arg == "--vdc" || arg == "--uac" || arg == "--vac" ||
                arg == "--base-ydc" || arg == "--base-uvdc" ||
                arg == "--base-uvac") &&
               i + 1 < argc) {
      int value = 0;
      if (!parse_int(argv[++i], &value)) {
        usage();
        return 2;
      }
      if (arg == "--width") options.width = value;
      else if (arg == "--height") options.height = value;
      else if (arg == "--ref-qp") options.ref_qp = value;
      else if (arg == "--base-q") options.base_q_idx = value;
      else if (arg == "--ydc") options.y_dc = value;
      else if (arg == "--udc") options.u_dc = value;
      else if (arg == "--vdc") options.v_dc = value;
      else if (arg == "--uac") options.u_ac = value;
      else if (arg == "--vac") options.v_ac = value;
      else if (arg == "--base-ydc") options.base_y_dc = value;
      else if (arg == "--base-uvdc") options.base_uv_dc = value;
      else if (arg == "--base-uvac") options.base_uv_ac = value;
    } else if (!arg.empty() && arg[0] == '-') {
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
  return run(options) ? 0 : 1;
}
