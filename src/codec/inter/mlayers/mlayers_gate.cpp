// Decoder gate for embedded layers (mlayers).
//
// Builds a two-embedded-layer temporal unit from two independent M2 intra
// encodes and decodes it with the pinned dav2d:
//
//   TD(xlayer = GLOBAL) + SEQ(max_mlayer_id = 1, SeqMaxMlayerCnt = 2)
//   + CLK(obu_mlayer_id = 0) + CLK(obu_mlayer_id = 1)
//
// Each layer is an independent intra picture (spatial coding only; no
// inter-layer prediction), so both are decodable on their own. Checks:
//
//   1. the emitted OBU headers carry the expected mlayer/xlayer ids;
//   2. the §5.4.1 layer fields and maps parse back exactly;
//   3. the §7.3.6 ordering/availability auditor accepts the stream;
//   4. dav2d --strict 1 --threads 1 decodes the two-layer unit and the
//      reconstruction is byte-exact against the two M2 recon outputs;
//   5. mlayer-0-only and mlayer-1-only streams decode to their own recon.
//
// Pinned-decoder note: dav2d f9dafbc does not parse
// seq_max_mlayer_cnt_minus_1 and uses a different dependency-block order.
// The gate therefore attempts the normative kSpecV1_0_0 stream first and, if
// the decoder rejects it, falls back to kPinnedDav2dCompat. The unit test pins
// the normative bytes; the gate proves the layer machinery decodes.

#include "codec/inter/mlayers/mlayers.h"
#include "codec/inter/mlayers/mlayers_syntax.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"

#include "core/bitstream/bitwriter.h"
#include "core/io/ivf.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::inter::mlayers::LayerId;
using av2enc::inter::mlayers::LayerObuRecord;
using av2enc::inter::mlayers::LayerStreamAuditor;
using av2enc::inter::mlayers::ObuHeaderFields;
using av2enc::inter::mlayers::ParsedObu;
using av2enc::inter::mlayers::ParsedSequenceHeader;
using av2enc::inter::mlayers::SequenceHeaderConfig;
using av2enc::inter::mlayers::SequenceSyntaxDialect;

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

struct Options {
  int width = 64;
  int height = 64;
  int qp = 90;
  std::string out_prefix;
  std::string dialect = "auto";
  bool keep = false;
};

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (env[0] != '\0' && std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

std::uint8_t square_sample(int x, int y, int frame, int plane) {
  const int sx = (8 + 4 * frame) % 96;
  const int sy = (8 + 2 * frame) % 64;
  const int in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 24;
  if (plane == 0) {
    if (in_square) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(
          40 + ((lx * 37 + ly * 17 + frame * 11) & 0x7F));
    }
    return static_cast<std::uint8_t>(96 + ((x + y + frame * 2) >> 3));
  }
  if (plane == 1) {
    return static_cast<std::uint8_t>(in_square ? 100 : 128);
  }
  return static_cast<std::uint8_t>(in_square ? 150 : 128);
}

OwnedFrame make_frame(int width, int height, int index) {
  OwnedFrame frame;
  frame.width = static_cast<std::uint32_t>(width);
  frame.height = static_cast<std::uint32_t>(height);
  const int cw = width / 2;
  const int ch = height / 2;
  frame.y.resize(static_cast<std::size_t>(width) *
                 static_cast<std::size_t>(height));
  frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      frame.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x)] = square_sample(x, y, index, 0);
    }
  }
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
          static_cast<std::size_t>(x);
      frame.u[i] = square_sample(x * 2, y * 2, index, 1);
      frame.v[i] = square_sample(x * 2, y * 2, index, 2);
    }
  }
  return frame;
}

av2enc::video::VideoInputFrame to_video_input(const OwnedFrame& frame) {
  av2enc::video::VideoInputFrame input;
  input.width = frame.width;
  input.height = frame.height;
  input.y = std::span<const std::uint8_t>(frame.y);
  input.u = std::span<const std::uint8_t>(frame.u);
  input.v = std::span<const std::uint8_t>(frame.v);
  return input;
}

bool extract_tile_payload(std::span<const std::uint8_t> unit,
                          const av2enc::video::VideoConfig& config,
                          std::vector<std::uint8_t>& tile, std::string& error) {
  std::size_t offset = 0;
  while (offset < unit.size()) {
    ParsedObu obu;
    if (!av2enc::inter::mlayers::parse_obu(unit, offset, obu, error)) {
      return false;
    }
    if (static_cast<int>(obu.header.obu_type) == 4 ||
        static_cast<int>(obu.header.obu_type) == 7) {
      av2enc::video::VideoFrameHeader dummy;
      dummy.keyframe = true;
      av2enc::BitWriter writer;
      const std::size_t bits =
          av2enc::video::write_tile_group_header(writer, config, dummy);
      const std::size_t header_bytes = writer.byte_size();
      if (bits != header_bytes * 8U || obu.payload.size() < header_bytes) {
        error = "unexpected M2 tile-group header size";
        return false;
      }
      tile.assign(obu.payload.begin() +
                      static_cast<std::ptrdiff_t>(header_bytes),
                  obu.payload.end());
      return true;
    }
    offset += obu.total_bytes;
  }
  error = "no frame OBU found in the M2 unit";
  return false;
}

std::vector<std::uint8_t> pack_frame_header(
    const SequenceHeaderConfig& config,
    const av2enc::inter::mlayers::KeyFrameHeader& header,
    std::span<const std::uint8_t> tile, std::string& error) {
  std::vector<std::uint8_t> header_bytes;
  std::size_t bit_count = 0;
  if (!av2enc::inter::mlayers::write_key_frame_header(
          config, header, header_bytes, bit_count, error)) {
    return {};
  }
  av2enc::BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group (§5.19 tile_group_obu)
  for (std::size_t i = 0; i < bit_count; ++i) {
    const std::uint8_t byte = header_bytes[i >> 3U];
    const unsigned shift = 7U - static_cast<unsigned>(i & 7U);
    writer.put_bit(((byte >> shift) & 1U) != 0U);
  }
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile.begin(), tile.end());
  return payload;
}

std::vector<std::uint8_t> make_clk_obu(
    const SequenceHeaderConfig& config, int mlayer_id,
    std::uint32_t refresh_frame_flags, std::span<const std::uint8_t> tile,
    std::string& error) {
  av2enc::inter::mlayers::KeyFrameHeader header;
  header.order_hint = 0;
  header.refresh_frame_flags = refresh_frame_flags;
  header.immediate_output_frame = true;
  std::vector<std::uint8_t> payload =
      pack_frame_header(config, header, tile, error);
  if (payload.empty()) return {};
  return av2enc::inter::mlayers::make_obu(
      4, 0, static_cast<std::uint8_t>(mlayer_id), 0, payload);
}

std::vector<std::uint8_t> make_td_obu() {
  return av2enc::inter::mlayers::make_obu(2, 0, 0, 31, {});
}

struct BuiltStream {
  std::vector<std::uint8_t> bytes;
  std::vector<std::uint8_t> seq_obu;
  std::vector<std::uint8_t> clk0;
  std::vector<std::uint8_t> clk1;
};

// present_mask bit 0 = mlayer 0, bit 1 = mlayer 1.
bool build_stream(const SequenceHeaderConfig& config,
                  SequenceSyntaxDialect dialect, int present_mask,
                  std::span<const std::uint8_t> tile0,
                  std::span<const std::uint8_t> tile1, BuiltStream& out,
                  std::string& error) {
  out = {};
  out.seq_obu.clear();
  if (!av2enc::inter::mlayers::make_sequence_header_obu(
          config, dialect, out.seq_obu, error)) {
    return false;
  }
  const std::vector<std::uint8_t> td = make_td_obu();
  out.bytes.insert(out.bytes.end(), td.begin(), td.end());
  out.bytes.insert(out.bytes.end(), out.seq_obu.begin(), out.seq_obu.end());
  if ((present_mask & 1) != 0) {
    out.clk0 = make_clk_obu(config, 0, 3, tile0, error);
    if (out.clk0.empty()) return false;
    out.bytes.insert(out.bytes.end(), out.clk0.begin(), out.clk0.end());
  }
  if ((present_mask & 2) != 0) {
    out.clk1 = make_clk_obu(config, 1, 3, tile1, error);
    if (out.clk1.empty()) return false;
    out.bytes.insert(out.bytes.end(), out.clk1.begin(), out.clk1.end());
  }
  return true;
}

bool write_ivf(const std::string& path,
               const std::vector<std::uint8_t>& unit, int width, int height,
               std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(path, header, error)) return false;
  if (!writer.write_frame(unit, 0, error)) return false;
  return writer.close(error);
}

bool run_dav2d(const std::string& dav2d, const std::string& input,
               const std::string& output, const std::string& log,
               std::string& error) {
  std::string command = "\"" + dav2d + "\" -q -i \"" + input + "\" -o \"" +
                        output + "\" --muxer yuv --alllayers 1 --strict 1 "
                        "--threads 1 >\"" +
                        log + "\" 2>&1";
  const int status = std::system(command.c_str());
  if (status != 0) {
    std::ifstream stream(log);
    std::string line;
    std::string tail;
    while (std::getline(stream, line)) {
      tail += line + "\n";
      if (tail.size() > 4000) tail.erase(0, tail.size() - 4000);
    }
    error = "dav2d exited with status " + std::to_string(status) + "\n" + tail;
    return false;
  }
  return true;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> recon_bytes(
    const av2enc::video::VideoReconFrame& frame) {
  std::vector<std::uint8_t> out;
  out.reserve(frame.y.size() + frame.u.size() + frame.v.size());
  out.insert(out.end(), frame.y.begin(), frame.y.end());
  out.insert(out.end(), frame.u.begin(), frame.u.end());
  out.insert(out.end(), frame.v.begin(), frame.v.end());
  return out;
}

bool compare_output(const std::vector<std::uint8_t>& expected,
                    const std::vector<std::uint8_t>& actual,
                    const char* label, std::string& error) {
  if (expected.size() != actual.size()) {
    error = std::string(label) + ": output size " +
            std::to_string(actual.size()) + " != expected " +
            std::to_string(expected.size());
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      error = std::string(label) + ": first mismatch at byte " +
              std::to_string(i) + " (expected " +
              std::to_string(expected[i]) + ", got " +
              std::to_string(actual[i]) + ")";
      return false;
    }
  }
  return true;
}

bool audit_stream(const BuiltStream& stream,
                  const av2enc::inter::mlayers::SequenceLayerConfig& layers,
                  bool enforce_seq_max_mlayer_cnt, std::string& error) {
  LayerStreamAuditor auditor(layers);
  auditor.set_enforce_seq_max_mlayer_cnt(enforce_seq_max_mlayer_cnt);
  std::size_t offset = 0;
  int sequence_headers = 0;
  int frames = 0;
  while (offset < stream.bytes.size()) {
    ParsedObu obu;
    if (!av2enc::inter::mlayers::parse_obu(stream.bytes, offset, obu, error)) {
      return false;
    }
    LayerObuRecord record;
    record.obu_type = static_cast<int>(obu.header.obu_type);
    record.layer = LayerId{obu.header.xlayer_id, obu.header.mlayer_id,
                           obu.header.tlayer_id};
    record.frame_unit_start = true;
    record.coded_output_frame = true;
    if (record.obu_type == 1) {
      ++sequence_headers;
      record.seq_header_id = 0;
      record.header_max_mlayer_id = layers.max_mlayer_id;
      record.header_max_tlayer_id = layers.max_tlayer_id;
    } else if (record.obu_type == 4) {
      if (frames >= 2) {
        error = "unexpected extra CLK OBU";
        return false;
      }
      ++frames;
      record.key_frame = true;
      record.order_hint = 0;
      record.seq_header_ref = 0;
    }
    if (!auditor.observe(record, error)) return false;
    offset += obu.total_bytes;
  }
  if (!auditor.finish(error)) return false;
  if (sequence_headers != 1 || frames < 1) {
    error = "expected one sequence header and at least one CLK frame";
    return false;
  }
  return true;
}

bool check_parsed_headers(const BuiltStream& stream,
                          SequenceSyntaxDialect dialect,
                          const SequenceHeaderConfig& config,
                          std::string& error) {
  ParsedSequenceHeader parsed;
  if (!av2enc::inter::mlayers::parse_sequence_header_obu(
          stream.seq_obu, dialect, parsed, error)) {
    return false;
  }
  if (dialect == SequenceSyntaxDialect::kSpecV1_0_0) {
    if (!parsed.seq_max_mlayer_cnt_coded) {
      error = "spec dialect did not code seq_max_mlayer_cnt_minus_1";
      return false;
    }
    if (parsed.seq_max_mlayer_cnt != config.layers.seq_max_mlayer_cnt) {
      error = "SeqMaxMlayerCnt does not round-trip";
      return false;
    }
  }
  if (parsed.max_mlayer_id != config.layers.max_mlayer_id ||
      parsed.max_tlayer_id != config.layers.max_tlayer_id) {
    error = "layer maxima do not round-trip";
    return false;
  }
  if (parsed.dependencies.mlayer != config.layers.dependencies.mlayer) {
    error = "MLayerDependencyMap does not round-trip";
    return false;
  }
  if (parsed.dependencies.tlayer != config.layers.dependencies.tlayer) {
    error = "TLayerDependencyMap does not round-trip";
    return false;
  }

  // OBU header ids.
  std::size_t offset = 0;
  bool saw_td = false;
  bool saw_seq = false;
  bool saw_m0 = false;
  bool saw_m1 = false;
  while (offset < stream.bytes.size()) {
    ParsedObu obu;
    if (!av2enc::inter::mlayers::parse_obu(stream.bytes, offset, obu, error)) {
      return false;
    }
    const int type = static_cast<int>(obu.header.obu_type);
    if (type == 2) {
      saw_td = true;
      if (obu.header.xlayer_id != av2enc::inter::mlayers::kGlobalXLayerId ||
          obu.header.extension_flag) {
        error = "TD OBU does not infer GLOBAL_XLAYER_ID";
        return false;
      }
    } else if (type == 1) {
      saw_seq = true;
      if (obu.header.extension_flag || obu.header.mlayer_id != 0 ||
          obu.header.xlayer_id != 0) {
        error = "sequence header OBU is not mlayer 0 / xlayer 0";
        return false;
      }
    } else if (type == 4) {
      if (obu.header.mlayer_id == 0) saw_m0 = true;
      if (obu.header.mlayer_id == 1) {
        saw_m1 = true;
        if (!obu.header.extension_flag) {
          error = "mlayer 1 CLK OBU did not set the extension flag";
          return false;
        }
      }
    }
    offset += obu.total_bytes;
  }
  if (!saw_td || !saw_seq || !saw_m0 || !saw_m1) {
    error = "stream is missing TD/SEQ/CLK(m0)/CLK(m1)";
    return false;
  }
  return true;
}

bool run_case(const Options& options, const std::string& dav2d,
              std::string& error) {
  // Two independent M2 intra encodes, one per embedded layer.
  const OwnedFrame frame0 =
      make_frame(options.width, options.height, 0);
  const OwnedFrame frame1 =
      make_frame(options.width, options.height, 1);

  av2enc::video::VideoOptions video_options;
  video_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
  video_options.force_single_block = true;
  video_options.enable_inter_residual = false;

  av2enc::video::VideoSequenceResult encoded0;
  av2enc::video::VideoSequenceResult encoded1;
  const std::array<av2enc::video::VideoInputFrame, 1> input0 = {
      to_video_input(frame0)};
  const std::array<av2enc::video::VideoInputFrame, 1> input1 = {
      to_video_input(frame1)};
  if (!av2enc::video::encode_video_sequence(input0, video_options, encoded0, error)) {
    error = "M2 layer-0 encode failed: " + error;
    return false;
  }
  if (!av2enc::video::encode_video_sequence(input1, video_options, encoded1, error)) {
    error = "M2 layer-1 encode failed: " + error;
    return false;
  }
  if (encoded0.units.size() != 1 || encoded1.units.size() != 1) {
    error = "M2 returned an unexpected unit count";
    return false;
  }

  av2enc::video::VideoConfig video_config;
  video_config.max_frame_width = static_cast<std::uint16_t>(options.width);
  video_config.max_frame_height = static_cast<std::uint16_t>(options.height);
  video_config.base_q_idx = static_cast<std::uint8_t>(options.qp);
  video_config.order_hint_bits = 4;
  video_config.num_ref_frames = 2;

  std::vector<std::uint8_t> tile0, tile1;
  if (!extract_tile_payload(encoded0.units[0], video_config, tile0, error)) {
    return false;
  }
  if (!extract_tile_payload(encoded1.units[0], video_config, tile1, error)) {
    return false;
  }

  SequenceHeaderConfig config = av2enc::inter::mlayers::
      baseline_sequence_header_config(
          static_cast<std::uint16_t>(options.width),
          static_cast<std::uint16_t>(options.height),
          static_cast<std::uint8_t>(options.qp), 2);
  config.layers =
      av2enc::inter::mlayers::default_sequence_layer_config(1, 0);
  // The default and explicit maps are identical here; keep the explicit flag
  // off so the pinned decoder uses its inferred default map too.
  config.layers.seq_max_mlayer_cnt = 2;
  config.layers.mlayer_dependency_present_flag = false;
  config.layers.tlayer_dependency_present_flag = false;
  config.layers.multi_tlayer_dependency_map_present_flag = false;

  std::vector<SequenceSyntaxDialect> dialects;
  if (options.dialect == "spec") {
    dialects.push_back(SequenceSyntaxDialect::kSpecV1_0_0);
  } else if (options.dialect == "compat") {
    dialects.push_back(SequenceSyntaxDialect::kPinnedDav2dCompat);
  } else {
    dialects.push_back(SequenceSyntaxDialect::kSpecV1_0_0);
    dialects.push_back(SequenceSyntaxDialect::kPinnedDav2dCompat);
  }

  const std::string prefix =
      options.out_prefix.empty()
          ? (std::filesystem::temp_directory_path() /
             ("mlayers_gate_" + std::to_string(options.width) + "x" +
              std::to_string(options.height)))
                .string()
          : options.out_prefix;

  std::vector<std::uint8_t> expected_full = recon_bytes(encoded0.recon[0]);
  const std::vector<std::uint8_t> expected1 = recon_bytes(encoded1.recon[0]);
  expected_full.insert(expected_full.end(), expected1.begin(), expected1.end());

  std::string last_error;
  bool decoded = false;
  SequenceSyntaxDialect used_dialect = SequenceSyntaxDialect::kSpecV1_0_0;
  for (SequenceSyntaxDialect dialect : dialects) {
    BuiltStream stream;
    if (!build_stream(config, dialect, 3, tile0, tile1, stream, error)) {
      error = "build two-layer stream: " + error;
      return false;
    }
    if (!check_parsed_headers(stream, dialect, config, error)) {
      error = "two-layer header check: " + error;
      return false;
    }
    if (!audit_stream(stream, config.layers,
                      dialect == SequenceSyntaxDialect::kSpecV1_0_0, error)) {
      error = "two-layer stream audit: " + error;
      return false;
    }
    const std::string path = prefix + ".two.ivf";
    if (!write_ivf(path, stream.bytes, options.width, options.height, error)) {
      return false;
    }
    const std::string decoded_path = prefix + ".two.yuv";
    const std::string log_path = prefix + ".two.log";
    if (!run_dav2d(dav2d, path, decoded_path, log_path, last_error)) {
      continue;
    }
    const std::vector<std::uint8_t> actual = read_file(decoded_path);
    if (!compare_output(expected_full, actual, "two-layer decode", last_error)) {
      continue;
    }
    used_dialect = dialect;
    decoded = true;
    std::cout << "decoded two-layer stream ("
              << (dialect == SequenceSyntaxDialect::kSpecV1_0_0
                      ? "spec v1.0.0"
                      : "pinned-dav2d compat")
              << ")\n";
    break;
  }
  if (!decoded) {
    error = "dav2d rejected both dialects: " + last_error;
    return false;
  }
  if (used_dialect == SequenceSyntaxDialect::kPinnedDav2dCompat &&
      options.dialect != "compat") {
    std::cout << "note: pinned dav2d f9dafbc does not parse "
                 "seq_max_mlayer_cnt_minus_1; gate used the compat dialect\n";
  }

  // Each embedded layer must decode on its own.
  const std::array<int, 2> present_masks = {1, 2};
  for (std::size_t index = 0; index < present_masks.size(); ++index) {
    const int present = static_cast<int>(index);
    BuiltStream stream;
    if (!build_stream(config, used_dialect, present_masks[index], tile0, tile1,
                      stream, error)) {
      error = "build single-layer stream: " + error;
      return false;
    }
    if (!audit_stream(stream, config.layers,
                      used_dialect == SequenceSyntaxDialect::kSpecV1_0_0,
                      error)) {
      error = "single-layer stream audit: " + error;
      return false;
    }
    const std::string path =
        prefix + ".m" + std::to_string(present) + ".ivf";
    if (!write_ivf(path, stream.bytes, options.width, options.height, error)) {
      return false;
    }
    const std::string decoded_path =
        prefix + ".m" + std::to_string(present) + ".yuv";
    const std::string log_path = prefix + ".m" + std::to_string(present) +
                                 ".log";
    if (!run_dav2d(dav2d, path, decoded_path, log_path, error)) {
      error = "mlayer-" + std::to_string(present) + ": " + error;
      return false;
    }
    const std::vector<std::uint8_t>& expected =
        present == 0 ? recon_bytes(encoded0.recon[0])
                     : recon_bytes(encoded1.recon[0]);
    const std::vector<std::uint8_t> actual = read_file(decoded_path);
    if (!compare_output(expected, actual,
                        present == 0 ? "mlayer-0 decode" : "mlayer-1 decode",
                        error)) {
      return false;
    }
    std::cout << "decoded mlayer-" << present << "-only stream\n";
  }
  return true;
}

void usage(const char* program) {
  std::cout << "usage: " << program
            << " [--width W] [--height H] [--qp Q] [--out PREFIX]"
               " [--dialect spec|compat|auto] [--keep]\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) return {};
      return argv[++i];
    };
    if (arg == "--width") {
      options.width = std::atoi(next().c_str());
    } else if (arg == "--height") {
      options.height = std::atoi(next().c_str());
    } else if (arg == "--qp") {
      options.qp = std::atoi(next().c_str());
    } else if (arg == "--out") {
      options.out_prefix = next();
    } else if (arg == "--dialect") {
      options.dialect = next();
    } else if (arg == "--keep") {
      options.keep = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (options.width <= 0 || options.height <= 0 ||
      (options.width % 8) != 0 || (options.height % 8) != 0) {
    std::cerr << "width/height must be positive multiples of 8\n";
    return 2;
  }
  if (options.qp < 1 || options.qp > 255) {
    std::cerr << "qp must be 1..255\n";
    return 2;
  }
  if (options.dialect != "spec" && options.dialect != "compat" &&
      options.dialect != "auto") {
    std::cerr << "dialect must be spec, compat or auto\n";
    return 2;
  }
  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "dav2d not found; set DAV2D=/path/to/dav2d\n";
    return 127;
  }
  std::string error;
  if (!run_case(options, dav2d, error)) {
    std::cerr << "mlayers gate FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "mlayers gate PASS\n";
  return 0;
}
