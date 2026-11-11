// Decoder gate for the temporal-layer planner.
//
// The M2 block layer is single-reference, so this gate uses a hierarchical-P
// carrier: the input frames are re-ordered into the plan's dyadic coding order
// and m2::encode_m2_sequence() encodes each as a P frame referencing the
// previous coding-order frame. The resulting tile payloads are re-wrapped with
// this directory's temporal-layer sequence/frame headers (max_tlayer_id,
// TLayerDependencyMap, obu_tlayer_id, order_hint, implicit output) and decoded
// with the pinned dav2d.
//
// Checks:
//   1. full hierarchical sequence: decoder output == encoder recon (0 diffs);
//   2. layer-0 drop: only obu_tlayer_id == 0 units, still decodable;
//   3. layer-1 drop: only obu_tlayer_id <= 1 units, still decodable;
//   4. every OBU's obu_tlayer_id matches the plan (TD/SEQ = 0).
//
// Usage:
//   tlayers_gate [--width W] [--height H] [--frames N] [--out PREFIX]
//                [--no-drop] [--keep]

#include "encoder/inter/tlayers/tlayers.h"
#include "encoder/inter/tlayers/tlayers_syntax.h"
#include "encoder/m2/m2_frame.h"
#include "encoder/m2/m2_headers.h"

#include "bitstream/bitwriter.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::inter::tlayers::FrameHeader;
using av2enc::inter::tlayers::ParsedObuHeader;
using av2enc::inter::tlayers::ParsedSequenceHeader;
using av2enc::inter::tlayers::Plan;
using av2enc::inter::tlayers::PlanConfig;
using av2enc::inter::tlayers::PlanError;
using av2enc::inter::tlayers::Schedule;
using av2enc::inter::tlayers::SequenceConfig;

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t square_sample(int x, int y, int frame, int plane) {
  const int sx = (8 + 4 * frame) % 96;
  const int sy = (8 + 2 * frame) % 64;
  const int in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 24;
  if (plane == 0) {
    if (in_square) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(40 + ((lx * 37 + ly * 17 + frame * 11) & 0x7F));
    }
    return static_cast<std::uint8_t>(96 + ((x + y + frame * 2) >> 3));
  }
  if (plane == 1) {
    return static_cast<std::uint8_t>(in_square ? 100 : 128);
  }
  return static_cast<std::uint8_t>(in_square ? 150 : 128);
}

std::vector<OwnedFrame> make_clip(int frames, int width, int height) {
  std::vector<OwnedFrame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)] =
            square_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const std::size_t index =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
            static_cast<std::size_t>(x);
        frame.u[index] = square_sample(x * 2, y * 2, f, 1);
        frame.v[index] = square_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

av2enc::m2::M2InputFrame to_input(const OwnedFrame& f) {
  av2enc::m2::M2InputFrame in;
  in.width = f.width;
  in.height = f.height;
  in.y = std::span<const std::uint8_t>(f.y);
  in.u = std::span<const std::uint8_t>(f.u);
  in.v = std::span<const std::uint8_t>(f.v);
  return in;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

av2enc::m2::M2Config m2_config_for(const SequenceConfig& sc,
                                   std::uint8_t num_total_refs) {
  av2enc::m2::M2Config config;
  config.max_frame_width = sc.max_frame_width;
  config.max_frame_height = sc.max_frame_height;
  config.base_q_idx = sc.base_q_idx;
  config.order_hint_bits = sc.order_hint_bits;
  config.num_ref_frames = sc.num_ref_frames;
  config.num_total_refs = num_total_refs;
  config.max_drl_bits_minus_1 = sc.max_drl_bits_minus_1;
  config.disable_cdf_update = sc.disable_cdf_update;
  config.enable_cfl_intra = sc.enable_cfl_intra;
  return config;
}

// Extracts the range-coded tile payload from an M2 frame OBU. M2's tile group
// header is fixed-width for a given key/inter flag and frame size, so its byte
// length can be measured by re-running the writer with a dummy header.
bool extract_tile_payload(std::span<const std::uint8_t> unit,
                          const av2enc::m2::M2Config& config, bool key,
                          std::vector<std::uint8_t>& tile, std::string& error) {
  std::size_t offset = 0;
  while (offset < unit.size()) {
    ParsedObuHeader header;
    if (!av2enc::inter::tlayers::parse_obu(unit, offset, header, error)) {
      return false;
    }
    if (header.obu_type == 4 || header.obu_type == 7) {
      av2enc::m2::M2FrameHeader dummy;
      dummy.keyframe = key;
      av2enc::BitWriter writer;
      const std::size_t bits =
          av2enc::m2::write_tile_group_header(writer, config, dummy);
      const std::size_t header_bytes = writer.byte_size();
      if (bits != header_bytes * 8U || header.payload.size() < header_bytes) {
        error = "unexpected M2 tile group header size";
        return false;
      }
      tile.assign(header.payload.begin() +
                      static_cast<std::ptrdiff_t>(header_bytes),
                  header.payload.end());
      return true;
    }
    offset += header.total_bytes;
  }
  error = "no frame OBU found in the temporal unit";
  return false;
}

// Rebuilds the whole temporal unit stream from the plan, optionally dropping
// frames with obu_tlayer_id > max_layer.
bool build_stream(const Plan& plan, const SequenceConfig& config,
                  const std::vector<std::vector<std::uint8_t>>& tiles,
                  int max_layer, std::vector<std::vector<std::uint8_t>>& units,
                  std::vector<int>& retained_display, std::string& error) {
  units.clear();
  retained_display.clear();
  bool wrote_sequence_header = false;
  for (const int display : plan.coding_order) {
    const int layer = plan.obu_tlayer_id(display);
    if (layer > max_layer) {
      continue;
    }
    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td =
        av2enc::inter::tlayers::make_temporal_delimiter_obu();
    unit.insert(unit.end(), td.begin(), td.end());
    if (!wrote_sequence_header) {
      const std::vector<std::uint8_t> seq =
          av2enc::inter::tlayers::make_sequence_header_obu(config);
      unit.insert(unit.end(), seq.begin(), seq.end());
      wrote_sequence_header = true;
    }
    const FrameHeader header =
        av2enc::inter::tlayers::frame_header_for(config, plan, display);
    const std::vector<std::uint8_t> obu =
        av2enc::inter::tlayers::make_frame_obu(
            config, header, static_cast<std::uint8_t>(layer),
            tiles[static_cast<std::size_t>(display)]);
    unit.insert(unit.end(), obu.begin(), obu.end());
    units.push_back(std::move(unit));
    retained_display.push_back(display);
  }
  if (!wrote_sequence_header) {
    error = "drop test removed every frame";
    return false;
  }
  return true;
}

bool write_ivf(const std::string& path,
               const std::vector<std::vector<std::uint8_t>>& units,
               std::uint32_t width, std::uint32_t height, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(path, header, error)) {
    return false;
  }
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) {
      return false;
    }
  }
  return writer.close(error);
}

struct CompareResult {
  bool ok = true;
  std::uint64_t mismatches = 0;
  int max_diff = 0;
  std::uint32_t frames = 0;
};

CompareResult compare_y4m(const std::string& path,
                          const std::vector<av2enc::m2::M2ReconFrame>& expected,
                          std::string& error) {
  CompareResult result;
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) {
    result.ok = false;
    return result;
  }
  av2enc::Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      result.ok = false;
      return result;
    }
    if (index >= expected.size()) {
      error = "decoder produced more frames than expected";
      result.ok = false;
      return result;
    }
    const av2enc::m2::M2ReconFrame& ref = expected[index];
    const auto compare = [&](std::span<const std::uint8_t> decoded,
                             const std::vector<std::uint8_t>& want, int stride) {
      if (decoded.size() != want.size()) {
        result.ok = false;
        return;
      }
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        const int diff = std::abs(static_cast<int>(decoded[i]) -
                                  static_cast<int>(want[i]));
        if (diff != 0) {
          ++result.mismatches;
          result.max_diff = std::max(result.max_diff, diff);
        }
      }
      (void)stride;
    };
    compare(frame.y(), ref.y, static_cast<int>(frame.width()));
    compare(frame.u(), ref.u, static_cast<int>((frame.width() + 1U) / 2U));
    compare(frame.v(), ref.v, static_cast<int>((frame.width() + 1U) / 2U));
    ++index;
    ++result.frames;
  }
  if (index != expected.size()) {
    error = "decoder produced " + std::to_string(index) + " of " +
            std::to_string(expected.size()) + " frames";
    result.ok = false;
  }
  result.ok = result.ok && result.mismatches == 0;
  return result;
}

bool decode_and_compare(const std::string& dav2d, const std::string& ivf_path,
                        const std::vector<av2enc::m2::M2ReconFrame>& expected,
                        std::string& error) {
  const std::string decoded = ivf_path + ".decoded.y4m";
  const std::string command =
      dav2d + " -q -i " + ivf_path + " -o " + decoded +
      " --strict 1 --threads 1";
  std::cout << "  decode: " << command << "\n";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    error = "dav2d returned " + std::to_string(rc);
    return false;
  }
  const CompareResult compare = compare_y4m(decoded, expected, error);
  if (!compare.ok) {
    if (error.empty()) {
      error = std::to_string(compare.mismatches) + " sample mismatches (max " +
              std::to_string(compare.max_diff) + ") in frame " +
              std::to_string(compare.frames);
    }
    return false;
  }
  std::cout << "  recon: " << compare.frames << " frame(s), 0 diffs\n";
  return true;
}

// Verifies obu_tlayer_id on every OBU and the parsed sequence-header fields.
bool audit_stream(const std::vector<std::vector<std::uint8_t>>& units,
                  const Plan& plan, const SequenceConfig& config,
                  std::string& error) {
  bool saw_sequence_header = false;
  for (std::size_t u = 0; u < units.size(); ++u) {
    std::size_t offset = 0;
    while (offset < units[u].size()) {
      ParsedObuHeader header;
      if (!av2enc::inter::tlayers::parse_obu(units[u], offset, header, error)) {
        return false;
      }
      if (header.obu_type == 2 && header.obu_tlayer_id != 0) {
        error = "OBU_TEMPORAL_DELIMITER with obu_tlayer_id != 0";
        return false;
      }
      if (header.obu_type == 1) {
        ParsedSequenceHeader parsed;
        std::span<const std::uint8_t> obu(
            units[u].data() + offset, header.total_bytes);
        if (!av2enc::inter::tlayers::parse_sequence_header_obu(obu, parsed,
                                                               error)) {
          return false;
        }
        if (parsed.max_tlayer_id != plan.max_tlayer_id ||
            parsed.monotonic_output_order_flag ||
            parsed.order_hint_bits != config.order_hint_bits ||
            parsed.tlayer_dependency_present_flag !=
                plan.tlayer_dependency_present_flag) {
          error = "parsed sequence header does not match the plan";
          return false;
        }
        for (int c = 0; c <= plan.max_tlayer_id; ++c) {
          for (int r = 0; r <= c; ++r) {
            if (parsed.tlayer_dependency[static_cast<std::size_t>(c)]
                                       [static_cast<std::size_t>(r)] !=
                plan.tlayer_dependency[static_cast<std::size_t>(c)]
                                      [static_cast<std::size_t>(r)]) {
              error = "parsed TLayerDependencyMap does not match the plan";
              return false;
            }
          }
        }
        saw_sequence_header = true;
      }
      if (header.obu_type == 4 && header.obu_tlayer_id != 0) {
        error = "key frame OBU with obu_tlayer_id != 0";
        return false;
      }
      if (header.obu_type == 7) {
        const int expected = plan.obu_tlayer_id(plan.coding_order[u]);
        if (header.obu_tlayer_id != expected) {
          error = "frame OBU obu_tlayer_id does not match the plan";
          return false;
        }
      }
      offset += header.total_bytes;
    }
  }
  if (!saw_sequence_header) {
    error = "no sequence header in the stream";
    return false;
  }
  return true;
}

struct Options {
  int width = 64;
  int height = 64;
  int frames = 8;
  std::string out;
  bool drop = true;
  bool keep = false;
};

bool run_case(const Options& options, const std::string& dav2d,
              std::string& error) {
  if (options.frames != 8 && options.frames != 4 && options.frames != 2) {
    error = "gate expects 2, 4 or 8 frames (power of two)";
    return false;
  }
  std::cout << "tlayers gate: " << options.width << "x" << options.height << ", "
            << options.frames << " frame(s)\n";

  const std::vector<OwnedFrame> display_frames =
      make_clip(options.frames, options.width, options.height);

  PlanConfig plan_config;
  plan_config.gop_size = options.frames;
  plan_config.schedule = Schedule::kHierarchicalP;
  plan_config.dependency = av2enc::inter::tlayers::DependencyMode::kAllLower;
  plan_config.explicit_dependency_map = false;
  plan_config.num_slots = options.frames;  // retain every implicit-output frame
  plan_config.order_hint_bits = 4;
  plan_config.base_q_idx = 90;
  Plan plan;
  PlanError plan_error = PlanError::kOk;
  if (!av2enc::inter::tlayers::build_plan(plan_config, plan, &plan_error)) {
    error = std::string("plan: ") +
            av2enc::inter::tlayers::plan_error_name(plan_error);
    return false;
  }
  // Drop layers are only meaningful for a real hierarchy.
  if (plan.max_tlayer_id < 1) {
    error = "gate requires a layered plan";
    return false;
  }

  // Feed M2 the frames in coding order; M2's P chain then matches
  // plan.frames[d].ref_display exactly (the preceding coding-order frame).
  std::vector<OwnedFrame> coding_frames;
  coding_frames.reserve(static_cast<std::size_t>(plan.display_count()));
  for (const int display : plan.coding_order) {
    coding_frames.push_back(display_frames[static_cast<std::size_t>(display)]);
  }
  std::vector<av2enc::m2::M2InputFrame> inputs;
  inputs.reserve(coding_frames.size());
  for (const OwnedFrame& f : coding_frames) {
    inputs.push_back(to_input(f));
  }

  av2enc::m2::M2Options m2_options;
  m2_options.base_q_idx = 90;
  m2_options.force_single_block = true;
  m2_options.enable_inter_residual = false;
  m2_options.me_range = 8;
  av2enc::m2::M2SequenceResult encoded;
  if (!av2enc::m2::encode_m2_sequence(inputs, m2_options, encoded, error)) {
    error = "M2 carrier encode failed: " + error;
    return false;
  }
  if (encoded.units.size() != coding_frames.size()) {
    error = "M2 returned the wrong number of units";
    return false;
  }

  SequenceConfig config = av2enc::inter::tlayers::baseline_sequence_config(
      static_cast<std::uint16_t>(options.width),
      static_cast<std::uint16_t>(options.height), 90, 2);
  config.apply_plan(plan);

  // M2's own header uses NumRefFrames = 2 (m2_frame.cpp), independent of the
  // plan's slot count. Its tile group header length must be measured with that
  // exact config to split the payload.
  SequenceConfig m2_side = av2enc::inter::tlayers::baseline_sequence_config(
      static_cast<std::uint16_t>(options.width),
      static_cast<std::uint16_t>(options.height), 90, 2);
  const av2enc::m2::M2Config m2_config = m2_config_for(m2_side, 1);

  std::vector<std::vector<std::uint8_t>> tiles(
      static_cast<std::size_t>(plan.display_count()));
  for (std::size_t coding = 0; coding < encoded.units.size(); ++coding) {
    const int display = plan.coding_order[coding];
    const bool key = plan.frames[static_cast<std::size_t>(display)].is_key;
    if (!extract_tile_payload(encoded.units[coding], m2_config, key,
                              tiles[static_cast<std::size_t>(display)], error)) {
      return false;
    }
  }

  std::vector<av2enc::m2::M2ReconFrame> display_recon(
      static_cast<std::size_t>(plan.display_count()));
  for (std::size_t coding = 0; coding < encoded.recon.size(); ++coding) {
    display_recon[static_cast<std::size_t>(plan.coding_order[coding])] =
        encoded.recon[coding];
  }

  // Display-order expectations for the full stream.
  std::vector<av2enc::m2::M2ReconFrame> full_expected;
  full_expected.reserve(display_recon.size());
  for (int d = 0; d < plan.display_count(); ++d) {
    full_expected.push_back(display_recon[static_cast<std::size_t>(d)]);
  }

  const std::string prefix = options.out.empty()
                                 ? (std::filesystem::temp_directory_path() /
                                    ("tlayers_gate_" + std::to_string(options.width) +
                                     "x" + std::to_string(options.height)))
                                       .string()
                                 : options.out;

  std::vector<std::vector<std::uint8_t>> units;
  std::vector<int> retained;
  if (!build_stream(plan, config, tiles, plan.max_tlayer_id, units, retained,
                    error)) {
    return false;
  }
  if (!audit_stream(units, plan, config, error)) {
    error = "full stream audit: " + error;
    return false;
  }
  const std::string full_path = prefix + ".full.ivf";
  if (!write_ivf(full_path, units, static_cast<std::uint32_t>(options.width),
                 static_cast<std::uint32_t>(options.height), error)) {
    return false;
  }
  if (!decode_and_compare(dav2d, full_path, full_expected, error)) {
    error = "full stream: " + error;
    return false;
  }

  if (!options.drop) {
    return true;
  }

  for (int max_layer = 0; max_layer < plan.max_tlayer_id; ++max_layer) {
    std::vector<std::vector<std::uint8_t>> drop_units;
    std::vector<int> drop_retained;
    if (!build_stream(plan, config, tiles, max_layer, drop_units,
                      drop_retained, error)) {
      return false;
    }
    // Decoder output order is display order for the retained frames.
    std::sort(drop_retained.begin(), drop_retained.end());
    std::vector<av2enc::m2::M2ReconFrame> expected;
    expected.reserve(drop_retained.size());
    for (const int display : drop_retained) {
      expected.push_back(display_recon[static_cast<std::size_t>(display)]);
    }
    const std::string drop_path =
        prefix + ".layer" + std::to_string(max_layer) + ".ivf";
    if (!write_ivf(drop_path, drop_units,
                   static_cast<std::uint32_t>(options.width),
                   static_cast<std::uint32_t>(options.height), error)) {
      return false;
    }
    std::cout << " drop to layer " << max_layer << " ("
              << drop_retained.size() << " frames)\n";
    if (!decode_and_compare(dav2d, drop_path, expected, error)) {
      error = "drop-layer-" + std::to_string(max_layer) + ": " + error;
      return false;
    }
  }
  return true;
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
    } else if (arg == "--frames") {
      options.frames = std::atoi(next().c_str());
    } else if (arg == "--out") {
      options.out = next();
    } else if (arg == "--no-drop") {
      options.drop = false;
    } else if (arg == "--keep") {
      options.keep = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "usage: " << argv[0]
                << " [--width W] [--height H] [--frames N] [--out PREFIX]"
                   " [--no-drop]\n";
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
  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "dav2d not found; set DAV2D=/path/to/dav2d\n";
    return 127;
  }
  std::string error;
  if (!run_case(options, dav2d, error)) {
    std::cerr << "TLayers gate FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "TLayers gate PASS\n";
  return 0;
}
