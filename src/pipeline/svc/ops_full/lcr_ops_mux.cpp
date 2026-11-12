// lcr_ops_mux -- splice a field-complete global LCR + OPS (and optionally a
// local OPS per extended layer) into a composed SVC temporal unit.
//
// Why this exists: `svc_mux` emits the field-complete stream *without* LCR/OPS
// (`--no-lcr --no-ops`) and this tool inserts the `ops_full` payloads in the
// §7.3.3 config-OBU position, so the black-box decoder gate exercises exactly
// the bytes produced by `lcr_ops_full.cpp`.
//
// The input must be one IVF frame containing, after the temporal delimiter,
// one sequence header + coded frame per extended layer in ascending xlayer
// order (the `svc_mux` output shape). Layer dimensions come from `--layer
// WxH`, in xlayer order.
//
// Usage:
//   lcr_ops_mux -o out.ivf --in composed.ivf --layer 64x64 --layer 640x360
//               [--profile N] [--level N] [--tier N] [--mlayers N]
//               [--features basic|full] [--atlas] [--local-ops]
//               [--inherit-ops] [--no-lcr] [--no-ops]
//
// Exit codes: 0 ok, 1 error, 2 usage.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "core/io/ivf.h"
#include "lcr_ops_full.h"
#include "svc_framing.h"

namespace {

using av2enc::svc::ops_full::GlobalPayload;
using av2enc::svc::ops_full::LayerConfigRecord;
using av2enc::svc::ops_full::OperatingPointSet;
using av2enc::svc::ops_full::OpsPoint;
using av2enc::svc::ops_full::StreamContext;
using av2enc::svc::ops_full::ValidationReport;
using av2enc::svc::ops_full::XLayerSequenceInfo;

struct LayerSpec {
  int width = 0;
  int height = 0;
};

bool read_ivf(const std::filesystem::path& path, std::vector<std::uint8_t>& frame,
              std::uint16_t& width, std::uint16_t& height,
              std::uint32_t& fps_num, std::uint32_t& fps_den,
              std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "cannot open '" + path.string() + "'";
    return false;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() < 32 || std::memcmp(bytes.data(), "DKIF", 4) != 0) {
    error = "'" + path.string() + "' is not an IVF file";
    return false;
  }
  auto le16 = [&](std::size_t at) {
    return static_cast<std::uint16_t>(bytes[at] | (bytes[at + 1] << 8));
  };
  auto le32 = [&](std::size_t at) {
    return static_cast<std::uint32_t>(bytes[at]) |
           (static_cast<std::uint32_t>(bytes[at + 1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[at + 2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[at + 3]) << 24U);
  };
  width = le16(12);
  height = le16(14);
  fps_num = le32(16);
  fps_den = le32(20);
  const std::uint32_t declared = le32(24);
  std::size_t offset = 32;
  std::vector<std::vector<std::uint8_t>> frames;
  while (offset + 12 <= bytes.size()) {
    const std::uint32_t size = le32(offset);
    offset += 12;
    if (size == 0 || size > bytes.size() - offset) {
      error = "IVF frame size is inconsistent";
      return false;
    }
    frames.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                        bytes.begin() +
                            static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
  }
  if (frames.size() != 1) {
    error = "'" + path.string() + "' must contain exactly one frame (got " +
            std::to_string(frames.size()) + ")";
    return false;
  }
  if (declared != 0 && declared != frames.size()) {
    error = "IVF frame count field disagrees with the frames present";
    return false;
  }
  frame = std::move(frames.front());
  return true;
}

bool parse_size(const char* text, LayerSpec& spec) {
  const std::string value(text);
  const std::size_t x = value.find('x');
  if (x == std::string::npos) return false;
  spec.width = std::atoi(value.substr(0, x).c_str());
  spec.height = std::atoi(value.substr(x + 1).c_str());
  return spec.width > 0 && spec.height > 0;
}

void usage() {
  std::cerr
      << "usage: lcr_ops_mux -o out.ivf --in composed.ivf --layer WxH ...\n"
         "                    [--profile N] [--level N] [--tier N]\n"
         "                    [--mlayers N] [--features basic|full]\n"
         "                    [--atlas] [--local-ops] [--local-lcr]\n"
         "                    [--inherit-ops] [--no-lcr] [--no-ops]\n";
}

struct Options {
  bool emit_lcr = true;
  bool emit_ops = true;
  bool full = true;
  bool atlas = false;
  bool local_ops = false;
  bool local_lcr = false;
  bool local_lcr_mismatch = false;
  bool inherit_ops = false;
  std::uint8_t profile = 0;
  std::uint8_t level = 7;
  bool tier = false;
  std::uint8_t mlayers = 1;
};

LayerConfigRecord build_lcr(const std::vector<LayerSpec>& layers,
                            const Options& options) {
  LayerConfigRecord record;
  record.is_global = true;
  auto& global = record.global;
  global.config_record_id = 1;
  global.xlayer_map = 0;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    global.xlayer_map |= 1U << static_cast<unsigned>(i);
  }
  global.aggregate_info_present = options.full;
  global.aggregate.config_idc = 0;  // Annex A.3 configuration 0
  global.aggregate.level_idx = options.level;
  global.aggregate.max_tier_flag = options.tier;
  global.aggregate.max_interop = 0;
  global.profile_tier_level_info_present = options.full;
  global.global_payload_present = true;
  global.dependent_xlayers_flag = false;
  global.global_atlas_id_present = options.atlas;
  global.global_atlas_id = 0;
  global.global_purpose_id = options.full ? 1 : 0;  // Stereoscopic viewports
  global.doh_constraint_flag = false;
  global.enforce_tile_alignment_flag = false;

  for (std::size_t i = 0; i < layers.size(); ++i) {
    auto& ptl = global.ptl[i];
    ptl.seq_profile_idc = options.profile;
    ptl.max_level_idx = options.level;
    ptl.tier_flag = options.tier;
    ptl.max_mlayer_count = options.mlayers;

    GlobalPayload payload;
    payload.xlayer_id = static_cast<std::uint8_t>(i);
    auto& info = payload.info;
    info.rep_info_present = options.full;
    info.rep.max_pic_width = static_cast<std::uint32_t>(layers[i].width);
    info.rep.max_pic_height = static_cast<std::uint32_t>(layers[i].height);
    info.rep.format_info_present = options.full;
    info.rep.bit_depth_idc = 1;   // 8-bit
    info.rep.chroma_format_idc = 0;  // 4:2:0
    info.purpose_present = options.full;
    info.purpose_id = 1;
    info.color_info_present = options.full;
    info.color.description_idc = 1;  // BT.709 SDR
    info.color.full_range = false;
    info.embedded_layer_info_present = true;
    info.mlayer_map = static_cast<std::uint8_t>((1U << options.mlayers) - 1U);
    for (int m = 0; m < options.mlayers; ++m) {
      auto embedded =
          av2enc::svc::ops_full::EmbeddedLayerInfo{};
      embedded.mlayer_id = static_cast<std::uint8_t>(m);
      embedded.tlayer_map = 0x01U;  // t0 only
      embedded.layer_type = av2enc::svc::ops_full::kLayerTexture;
      embedded.view_type = av2enc::svc::ops_full::kViewCenter;
      embedded.same_sh_max_resolution = true;
      if (options.atlas) {
        embedded.atlas_segment_id = static_cast<std::uint8_t>(i);
        embedded.priority_order = static_cast<std::uint8_t>(i);
        embedded.rendering_method = 0;
      }
      info.embedded.push_back(embedded);
    }
    global.payloads.push_back(std::move(payload));
  }
  return record;
}

OperatingPointSet build_global_ops(const std::vector<LayerSpec>& layers,
                                   const Options& options) {
  OperatingPointSet set;
  set.xlayer_id = av2enc::svc::ops_full::kGlobalXLayerId;
  set.ops_id = 0;
  set.reset_flag = true;
  set.priority = 0;
  set.intent = 1;  // OPSI_SCALABILITY
  set.intent_present = options.full;
  set.ptl_present = options.full;
  set.color_info_present = options.full;
  set.mlayer_info_idc = options.inherit_ops ? 2U : 1U;

  for (std::size_t i = 0; i < layers.size(); ++i) {
    OpsPoint point;
    point.op_intent = 1;
    point.aggregate.config_idc = 0;
    point.aggregate.level_idx = options.level;
    point.aggregate.max_tier_flag = options.tier;
    point.aggregate.max_interop = 0;
    point.color.description_idc = 1;
    point.decoder_model_info_present = options.full && i == 0;
    point.decoder_model.decoder_buffer_delay = 80000;
    point.decoder_model.encoder_buffer_delay = 10000;
    point.initial_display_delay_present = options.full;
    point.initial_display_delay_minus_1 = static_cast<std::uint8_t>(i);
    point.xlayer_map = 0;
    for (std::size_t j = 0; j <= i; ++j) {
      point.xlayer_map |= 1U << static_cast<unsigned>(j);
    }
    for (std::size_t j = 0; j <= i; ++j) {
      point.ptl[j].seq_profile_idc = options.profile;
      point.ptl[j].level_idx = options.level;
      point.ptl[j].tier_flag = options.tier;
      point.ptl[j].mlayer_count = options.mlayers;
      auto& info = point.layer_info[j];
      info.mlayer_map =
          static_cast<std::uint8_t>((1U << options.mlayers) - 1U);
      for (int m = 0; m < options.mlayers; ++m) {
        info.tlayer_map[static_cast<std::size_t>(m)] = 0x01U;
      }
      // Inherit the newest layer's maps from operating point 0 to exercise
      // ops_mlayer_explicit_info_flag / ops_embedded_ops_id/index.
      if (options.inherit_ops && j > 0 && i > 0) {
        info.explicit_info = false;
        info.embedded_ops_id = 0;
        info.embedded_op_index = 0;
        info.mlayer_map = 0;
        info.tlayer_map[0] = 0;
      }
    }
    set.points.push_back(point);
  }
  return set;
}

OperatingPointSet build_local_ops(std::uint8_t xlayer_id,
                                  const Options& options) {
  OperatingPointSet set;
  set.xlayer_id = xlayer_id;
  set.ops_id = 1;
  set.reset_flag = true;
  set.priority = 0;
  set.intent = 1;
  set.intent_present = options.full;
  set.ptl_present = options.full;
  set.color_info_present = options.full;
  OpsPoint point;
  point.op_intent = 1;
  point.color.description_idc = 1;
  point.ptl[xlayer_id].seq_profile_idc = options.profile;
  point.ptl[xlayer_id].level_idx = options.level;
  point.ptl[xlayer_id].tier_flag = options.tier;
  point.ptl[xlayer_id].mlayer_count = options.mlayers;
  point.initial_display_delay_present = options.full;
  point.initial_display_delay_minus_1 = 0;
  point.layer_info[xlayer_id].mlayer_map =
      static_cast<std::uint8_t>((1U << options.mlayers) - 1U);
  for (int m = 0; m < options.mlayers; ++m) {
    point.layer_info[xlayer_id].tlayer_map[static_cast<std::size_t>(m)] =
        0x01U;
  }
  set.points.push_back(point);
  return set;
}

LayerConfigRecord build_local_lcr(std::uint8_t xlayer_id,
                                  const LayerConfigRecord& global,
                                  const Options& options) {
  LayerConfigRecord record;
  record.is_global = false;
  auto& local = record.local;
  local.xlayer_id = xlayer_id;
  local.global_id = global.global.config_record_id;
  local.local_id = 2;
  local.profile_tier_level_info_present = options.full;
  local.ptl.seq_profile_idc = options.profile;
  local.ptl.max_level_idx = options.level;
  local.ptl.tier_flag = options.tier;
  local.ptl.max_mlayer_count = options.mlayers;
  local.local_atlas_id_present = options.atlas;
  local.local_atlas_id = 0;
  local.xlayer = global.global.payloads[xlayer_id].info;
  if (options.local_lcr_mismatch) {
    // Negative control: the local record claims a different temporal-layer
    // set than the activated global LCR; §6.8 requires the two to agree.
    if (!local.xlayer.embedded.empty()) {
      local.xlayer.embedded[0].tlayer_map = 0x03U;
    }
  }
  return record;
}

bool append_obu(av2enc::svc::ObuType type,
                std::span<const std::uint8_t> payload,
                std::vector<std::uint8_t>& out, std::string& error) {
  av2enc::svc::ObuHeader header;
  header.type = type;
  header.xlayer_id =
      static_cast<std::uint8_t>(av2enc::svc::layers::kGlobalXLayerId);
  return av2enc::svc::frame_obu(header, payload, out, error);
}

}  // namespace

int main(int argc, char** argv) {
  std::string output;
  std::string input;
  std::vector<LayerSpec> layers;
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "lcr_ops_mux: " << name << " needs a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "-o" || arg == "--output") {
      const char* v = need("--output");
      if (v == nullptr) return 2;
      output = v;
    } else if (arg == "--in") {
      const char* v = need("--in");
      if (v == nullptr) return 2;
      input = v;
    } else if (arg == "--layer") {
      const char* v = need("--layer");
      if (v == nullptr) return 2;
      LayerSpec spec;
      if (!parse_size(v, spec)) {
        std::cerr << "lcr_ops_mux: --layer needs WxH\n";
        return 2;
      }
      layers.push_back(spec);
    } else if (arg == "--profile") {
      const char* v = need("--profile");
      if (v == nullptr) return 2;
      options.profile = static_cast<std::uint8_t>(std::atoi(v));
    } else if (arg == "--level") {
      const char* v = need("--level");
      if (v == nullptr) return 2;
      options.level = static_cast<std::uint8_t>(std::atoi(v));
    } else if (arg == "--tier") {
      const char* v = need("--tier");
      if (v == nullptr) return 2;
      options.tier = std::atoi(v) != 0;
    } else if (arg == "--mlayers") {
      const char* v = need("--mlayers");
      if (v == nullptr) return 2;
      options.mlayers = static_cast<std::uint8_t>(std::atoi(v));
    } else if (arg == "--features") {
      const char* v = need("--features");
      if (v == nullptr) return 2;
      if (std::string(v) == "basic") {
        options.full = false;
      } else if (std::string(v) == "full") {
        options.full = true;
      } else {
        std::cerr << "lcr_ops_mux: --features must be basic|full\n";
        return 2;
      }
    } else if (arg == "--atlas") {
      options.atlas = true;
    } else if (arg == "--local-ops") {
      options.local_ops = true;
    } else if (arg == "--local-lcr") {
      options.local_lcr = true;
    } else if (arg == "--local-lcr-mismatch") {
      options.local_lcr = true;
      options.local_lcr_mismatch = true;
    } else if (arg == "--inherit-ops") {
      options.inherit_ops = true;
    } else if (arg == "--no-lcr") {
      options.emit_lcr = false;
    } else if (arg == "--no-ops") {
      options.emit_ops = false;
    } else if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else {
      std::cerr << "lcr_ops_mux: unknown argument '" << arg << "'\n";
      usage();
      return 2;
    }
  }

  if (output.empty() || input.empty() || layers.empty()) {
    usage();
    return 2;
  }
  if (layers.size() > 30) {
    std::cerr << "lcr_ops_mux: at most 30 extended layers\n";
    return 2;
  }
  if (options.mlayers < 1 || options.mlayers > 8) {
    std::cerr << "lcr_ops_mux: --mlayers must be 1..8\n";
    return 2;
  }

  std::vector<std::uint8_t> frame;
  std::uint16_t width = 0;
  std::uint16_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 0;
  std::string error;
  if (!read_ivf(input, frame, width, height, fps_num, fps_den, error)) {
    std::cerr << "lcr_ops_mux: " << error << '\n';
    return 1;
  }

  std::vector<av2enc::svc::ParsedObu> obus;
  if (!av2enc::svc::parse_obus(frame, obus, error)) {
    std::cerr << "lcr_ops_mux: " << error << '\n';
    return 1;
  }
  std::vector<av2enc::svc::ObuHeader> sequence_layers;
  for (const auto& obu : obus) {
    if (obu.header.type == av2enc::svc::ObuType::kSequenceHeader) {
      sequence_layers.push_back(obu.header);
    }
  }
  if (sequence_layers.size() != layers.size()) {
    std::cerr << "lcr_ops_mux: input has " << sequence_layers.size()
              << " sequence header(s) but " << layers.size()
              << " --layer argument(s)\n";
    return 1;
  }
  for (std::size_t i = 0; i < sequence_layers.size(); ++i) {
    if (sequence_layers[i].xlayer_id != i) {
      std::cerr << "lcr_ops_mux: sequence headers must be in ascending "
                   "xlayer order from 0\n";
      return 1;
    }
  }

  const LayerConfigRecord lcr = build_lcr(layers, options);
  const OperatingPointSet global_ops = build_global_ops(layers, options);

  // Validate before emitting, so a gate failure is a writer/validator bug and
  // not a silent malformed stream.
  StreamContext context;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    XLayerSequenceInfo info;
    info.xlayer_id = static_cast<std::uint8_t>(i);
    info.seq_profile_idc = options.profile;
    info.seq_level_idx = options.level;
    info.seq_tier = options.tier;
    info.seq_max_mlayer_cnt = options.mlayers;
    info.max_frame_width = static_cast<std::uint32_t>(layers[i].width);
    info.max_frame_height = static_cast<std::uint32_t>(layers[i].height);
    info.bit_depth_idc = 1;
    info.chroma_format_idc = 0;
    info.monotonic_output_order = true;
    context.sequences.push_back(info);
  }
  context.dependencies = av2enc::tiles::svc::default_dependency_maps(
      static_cast<int>(options.mlayers) - 1, 0);
  context.lcr_active = true;
  context.active_global_lcr_id = lcr.global.config_record_id;
  context.known_ops.push_back(
      av2enc::svc::ops_full::summarize_ops(global_ops));

  ValidationReport report;
  bool valid = true;
  if (options.emit_lcr && options.emit_ops) {
    valid = av2enc::svc::ops_full::validate_stream(lcr, global_ops, context,
                                                   report);
  } else {
    if (options.emit_lcr) {
      valid = av2enc::svc::ops_full::validate_lcr(lcr, context, report);
    }
    if (options.emit_ops) {
      valid = av2enc::svc::ops_full::validate_ops(global_ops, context,
                                                  report) && valid;
    }
  }
  if (!valid) {
    std::cerr << "lcr_ops_mux: validation failed:\n";
    for (const std::string& message : report.errors) {
      std::cerr << "  " << message << '\n';
    }
    return 1;
  }

  std::vector<std::uint8_t> out;
  bool inserted = false;
  for (const auto& obu : obus) {
    if (options.local_lcr &&
        obu.header.type == av2enc::svc::ObuType::kSequenceHeader) {
      const LayerConfigRecord local =
          build_local_lcr(obu.header.xlayer_id, lcr, options);
      ValidationReport local_report;
      if (!av2enc::svc::ops_full::validate_lcr(local, context, local_report) ||
          !av2enc::svc::ops_full::validate_local_lcr_matches_global(
              lcr, local, local_report)) {
        std::cerr << "lcr_ops_mux: local LCR validation failed:\n";
        for (const std::string& message : local_report.errors) {
          std::cerr << "  " << message << '\n';
        }
        return 1;
      }
      std::vector<std::uint8_t> local_payload;
      if (!av2enc::svc::ops_full::write_lcr_payload(local, local_payload,
                                                    error)) {
        std::cerr << "lcr_ops_mux: local LCR: " << error << '\n';
        return 1;
      }
      av2enc::svc::ObuHeader local_header;
      local_header.type = av2enc::svc::ObuType::kLayerConfigRecord;
      local_header.xlayer_id = obu.header.xlayer_id;
      if (!av2enc::svc::frame_obu(local_header, local_payload, out, error)) {
        std::cerr << "lcr_ops_mux: local LCR: " << error << '\n';
        return 1;
      }
    }
    out.insert(out.end(), obu.raw.begin(), obu.raw.end());
    if (!inserted &&
        obu.header.type == av2enc::svc::ObuType::kTemporalDelimiter) {
      inserted = true;
      if (options.emit_lcr) {
        std::vector<std::uint8_t> payload;
        if (!av2enc::svc::ops_full::write_lcr_payload(lcr, payload, error) ||
            !append_obu(av2enc::svc::ObuType::kLayerConfigRecord, payload, out,
                        error)) {
          std::cerr << "lcr_ops_mux: LCR: " << error << '\n';
          return 1;
        }
      }
      if (options.emit_ops) {
        std::vector<std::uint8_t> payload;
        if (!av2enc::svc::ops_full::write_ops_payload(global_ops, payload,
                                                      error) ||
            !append_obu(av2enc::svc::ObuType::kOperatingPointSet, payload, out,
                        error)) {
          std::cerr << "lcr_ops_mux: OPS: " << error << '\n';
          return 1;
        }
        if (options.local_ops) {
          for (std::size_t i = 0; i < layers.size(); ++i) {
            const OperatingPointSet local = build_local_ops(
                static_cast<std::uint8_t>(i), options);
            std::vector<std::uint8_t> local_payload;
            if (!av2enc::svc::ops_full::write_ops_payload(local, local_payload,
                                                          error)) {
              std::cerr << "lcr_ops_mux: local OPS: " << error << '\n';
              return 1;
            }
            av2enc::svc::ObuHeader header;
            header.type = av2enc::svc::ObuType::kOperatingPointSet;
            header.xlayer_id = static_cast<std::uint8_t>(i);
            if (!av2enc::svc::frame_obu(header, local_payload, out, error)) {
              std::cerr << "lcr_ops_mux: local OPS: " << error << '\n';
              return 1;
            }
          }
        }
      }
    }
  }
  if (!inserted) {
    std::cerr << "lcr_ops_mux: input has no temporal delimiter\n";
    return 1;
  }

  av2enc::IvfHeader header;
  header.width = width;
  header.height = height;
  header.fps_num = fps_num;
  header.fps_den = fps_den;
  header.frame_count = 1;
  av2enc::IvfWriter writer;
  if (!writer.open(output, header, error) || !writer.write_frame(out, 0, error) ||
      !writer.close(error)) {
    std::cerr << "lcr_ops_mux: " << error << '\n';
    return 1;
  }

  std::cout << "lcr_ops_mux: " << layers.size() << " layer(s), "
            << global_ops.points.size() << " operating point(s), "
            << out.size() << " bytes -> " << output << '\n';
  for (std::size_t i = 0; i < global_ops.points.size(); ++i) {
    std::cout << "  op" << i << ": xlayer_map 0x" << std::hex
              << global_ops.points[i].xlayer_map << std::dec << '\n';
  }
  return 0;
}
