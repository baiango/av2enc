// svc_mux -- compose/rewrap single-layer still units into a layered IVF.
//
// This is the ungated bridge used while the encoder core has no multi-layer
// output: each input is a complete single-layer IVF temporal unit from the
// existing encoder. svc_mux rewrites only OBU layer ids, adds a global LCR/OPS,
// and (optionally) runs Annex F.3.2 extraction for --oppoint/--alllayers.
//
// It deliberately does not re-encode sequence headers or tile payloads.
//
// Usage:
//   svc_mux -o out.ivf --layer base.ivf [--layer enh.ivf ...]
//           [--temporal N] [--embedded N] [--no-lcr] [--no-ops]
//           [--extract OP | --oppoint OP] [--alllayers]
//
// Exit codes: 0 ok, 1 error, 2 usage.

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "io/ivf.h"
#include "svc_config.h"
#include "svc_framing.h"

namespace {

struct IvfFrame {
  std::vector<std::uint8_t> payload;
};

struct IvfFile {
  std::uint16_t width = 0;
  std::uint16_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 0;
  std::vector<IvfFrame> frames;
};

std::uint16_t load_le16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t load_le32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8U) |
         (static_cast<std::uint32_t>(p[2]) << 16U) |
         (static_cast<std::uint32_t>(p[3]) << 24U);
}

bool read_ivf(const std::filesystem::path& path, IvfFile& file,
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
  file.width = load_le16(bytes.data() + 12);
  file.height = load_le16(bytes.data() + 14);
  file.fps_num = load_le32(bytes.data() + 16);
  file.fps_den = load_le32(bytes.data() + 20);
  const std::uint32_t declared = load_le32(bytes.data() + 24);
  if (file.width == 0 || file.height == 0) {
    error = "IVF has zero dimensions";
    return false;
  }
  if (file.fps_num == 0 || file.fps_den == 0) {
    error = "IVF has an invalid frame rate";
    return false;
  }
  std::size_t offset = 32;
  while (offset + 12 <= bytes.size()) {
    const std::uint32_t size = load_le32(bytes.data() + offset);
    offset += 12;
    if (size == 0 || size > bytes.size() - offset) {
      error = "IVF frame size is inconsistent";
      return false;
    }
    IvfFrame frame;
    frame.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                         bytes.begin() +
                             static_cast<std::ptrdiff_t>(offset + size));
    offset += size;
    file.frames.push_back(std::move(frame));
  }
  if (offset != bytes.size()) {
    error = "IVF has a truncated frame header";
    return false;
  }
  if (file.frames.empty()) {
    error = "IVF has no frames";
    return false;
  }
  if (declared != 0 && declared != file.frames.size()) {
    error = "IVF frame count field disagrees with the frames present";
    return false;
  }
  return true;
}

void usage() {
  std::cerr
      << "usage: svc_mux -o out.ivf --layer base.ivf [--layer enh.ivf ...]\n"
         "               [--temporal N] [--embedded N] [--no-lcr] [--no-ops]\n"
         "               [--extract OP | --oppoint OP] [--alllayers]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string output;
  std::vector<std::string> layer_paths;
  int temporal = 1;
  int embedded = 1;
  bool emit_lcr = true;
  bool emit_ops = true;
  int extract = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "svc_mux: " << name << " needs a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "-o" || arg == "--output") {
      const char* value = need_value("--output");
      if (value == nullptr) {
        return 2;
      }
      output = value;
    } else if (arg == "--layer") {
      const char* value = need_value("--layer");
      if (value == nullptr) {
        return 2;
      }
      layer_paths.emplace_back(value);
    } else if (arg == "--temporal") {
      const char* value = need_value("--temporal");
      if (value == nullptr) {
        return 2;
      }
      temporal = std::atoi(value);
    } else if (arg == "--embedded") {
      const char* value = need_value("--embedded");
      if (value == nullptr) {
        return 2;
      }
      embedded = std::atoi(value);
    } else if (arg == "--no-lcr") {
      emit_lcr = false;
    } else if (arg == "--no-ops") {
      emit_ops = false;
    } else if (arg == "--extract" || arg == "--oppoint") {
      const char* value = need_value(arg.c_str());
      if (value == nullptr) {
        return 2;
      }
      extract = std::atoi(value);
    } else if (arg == "--alllayers") {
      extract = -1;
    } else if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else {
      std::cerr << "svc_mux: unknown argument '" << arg << "'\n";
      usage();
      return 2;
    }
  }

  if (output.empty() || layer_paths.empty()) {
    usage();
    return 2;
  }
  if (layer_paths.size() > 30) {
    std::cerr << "svc_mux: at most 30 spatial layers\n";
    return 2;
  }
  if (temporal != 1 || embedded != 1) {
    std::cerr << "svc_mux: each input is one xlayer/m0/t0 unit; "
                 "--temporal/--embedded must be 1\n";
    return 2;
  }

  std::vector<IvfFile> files(layer_paths.size());
  for (std::size_t i = 0; i < layer_paths.size(); ++i) {
    std::string error;
    if (!read_ivf(layer_paths[i], files[i], error)) {
      std::cerr << "svc_mux: " << error << '\n';
      return 1;
    }
    if (files[i].frames.size() != 1) {
      std::cerr << "svc_mux: '" << layer_paths[i]
                << "' must contain exactly one frame (got "
                << files[i].frames.size() << ")\n";
      return 1;
    }
  }

  av2enc::svc::LayerConfig config;
  for (std::size_t i = 0; i < files.size(); ++i) {
    av2enc::svc::SpatialLayerSpec layer;
    layer.xlayer_id = static_cast<std::uint8_t>(i);
    layer.width = files[i].width;
    layer.height = files[i].height;
    config.spatial_layers.push_back(layer);
  }
  config.temporal_layers = static_cast<std::uint8_t>(temporal);
  config.embedded_layers = static_cast<std::uint8_t>(embedded);
  config.seq_lcr_id = 1;

  av2enc::svc::LayerPlan plan;
  std::string error;
  if (!av2enc::svc::build_layer_plan(config, plan, error)) {
    std::cerr << "svc_mux: layer plan: " << error << '\n';
    return 1;
  }

  std::vector<av2enc::svc::LayerInput> inputs;
  for (std::size_t i = 0; i < files.size(); ++i) {
    av2enc::svc::LayerInput input;
    input.layer.xlayer_id = static_cast<std::uint8_t>(i);
    input.layer.mlayer_id = 0;
    input.layer.tlayer_id = 0;
    input.obus = files[i].frames.front().payload;
    inputs.push_back(std::move(input));
  }

  av2enc::svc::ComposeOptions options;
  options.emit_lcr = emit_lcr;
  options.emit_ops = emit_ops;
  std::vector<std::uint8_t> stream;
  if (!av2enc::svc::compose_layered_stream(plan, inputs, options, stream,
                                           error)) {
    std::cerr << "svc_mux: compose: " << error << '\n';
    return 1;
  }

  if (extract >= 0) {
    av2enc::svc::LayerSelection selection;
    selection.mode = av2enc::svc::SelectionMode::kOperatingPoint;
    selection.op_index = extract;
    av2enc::tiles::svc::RetentionMap retention;
    if (!av2enc::svc::resolve_selection(plan, selection, retention, error)) {
      std::cerr << "svc_mux: extract: " << error << '\n';
      return 1;
    }
    std::vector<std::uint8_t> extracted;
    if (!av2enc::svc::extract_substream(stream, retention, extracted, error)) {
      std::cerr << "svc_mux: extract: " << error << '\n';
      return 1;
    }
    stream = std::move(extracted);
  }

  av2enc::IvfHeader header;
  header.width = files.front().width;
  header.height = files.front().height;
  header.fps_num = files.front().fps_num;
  header.fps_den = files.front().fps_den;
  header.frame_count = 1;

  av2enc::IvfWriter writer;
  if (!writer.open(output, header, error)) {
    std::cerr << "svc_mux: " << error << '\n';
    return 1;
  }
  if (!writer.write_frame(stream, 0, error)) {
    std::cerr << "svc_mux: " << error << '\n';
    return 1;
  }
  if (!writer.close(error)) {
    std::cerr << "svc_mux: " << error << '\n';
    return 1;
  }

  std::cout << "svc_mux: " << files.size() << " layer(s), "
            << plan.operating_points.size() << " operating point(s), "
            << stream.size() << " bytes -> " << output << '\n';
  for (const auto& info : plan.operating_points) {
    std::cout << "  " << info.label << '\n';
  }
  return 0;
}
