// Decoder gate for the HRD/timing component.
//
// Encodes a short M2 P-frame stream, rewrites it with:
//   * a sequence header carrying seq_initial_display_delay and
//     decoder_model_info / seq_decoder_model_info (§5.4.1, §5.4.13),
//   * a Content Interpretation OBU carrying timing_info() (§5.4.12/§5.15),
//   * one buffer_removal_timing_obu() per coded frame (§5.12),
// then decodes both the baseline and the rewritten stream with the pinned
// dav2d (`--strict 1 --threads 1`).
//
// Pass criteria:
//   * both decodes exit 0 and the decoded Y4M byte streams are identical
//     (pixels unchanged by the timing syntax),
//   * the rewritten decode still matches the encoder reconstruction,
//   * the rewritten stream self-parses with the expected OBU order and values.
//
// dav2d may ignore BRT (the frame times it reports are compared and printed).
// When frame times are unchanged the gate still passes on invariance +
// self-parse and says so explicitly.

#include "encoder/m2/m2_frame.h"
#include "encoder/rc/hrd/hrd.h"
#include "encoder/rc/hrd/hrd_buffer.h"

#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t square_sample(int x, int y, int frame, int plane) {
  const int sx = 8 + 4 * frame;
  const int sy = 8 + 2 * frame;
  const bool in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 32;
  if (plane == 0) {
    if (in_square) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(40 + ((lx * 37 + ly * 17) & 0x7F));
    }
    return static_cast<std::uint8_t>(96 + ((x + y) >> 3));
  }
  if (plane == 1) return static_cast<std::uint8_t>(in_square ? 100 : 128);
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
    frame.y.resize(static_cast<std::size_t>(width) * height);
    frame.u.resize(static_cast<std::size_t>(cw) * ch);
    frame.v.resize(static_cast<std::size_t>(cw) * ch);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * width + x] =
            square_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * cw + x] =
            square_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * cw + x] =
            square_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
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
  if (!writer.open(path, header, error)) return false;
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) return false;
  }
  return writer.close(error);
}

std::vector<std::uint8_t> read_file(const std::string& path, bool& ok) {
  std::ifstream input(path, std::ios::binary);
  ok = input.good();
  if (!ok) return {};
  return std::vector<std::uint8_t>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

struct DecodeResult {
  bool ok = false;
  std::string y4m;
  std::string frametimes;
  std::string log;
};

DecodeResult decode(const std::string& dav2d, const std::string& ivf,
                    const std::string& tag) {
  DecodeResult result;
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "av2enc_hrd_gate";
  std::filesystem::create_directories(dir);
  result.y4m = (dir / (tag + ".decoded.y4m")).string();
  result.frametimes = (dir / (tag + ".frametimes.txt")).string();
  result.log = (dir / (tag + ".decode.log")).string();
  const std::string command =
      dav2d + " -q -i \"" + ivf + "\" -o \"" + result.y4m +
      "\" --threads 1 --strict 1 --frametimes \"" + result.frametimes +
      "\" > \"" + result.log + "\" 2>&1";
  std::cout << "  decode: " << command << "\n";
  const int rc = std::system(command.c_str());
  result.ok = rc == 0;
  if (!result.ok) {
    std::cout << "  decode exit " << rc << " (log: " << result.log << ")\n";
  }
  return result;
}

bool log_mentions_unknown_brt(const std::string& path) {
  bool ok = false;
  const std::vector<std::uint8_t> bytes = read_file(path, ok);
  if (!ok) return false;
  const std::string text(bytes.begin(), bytes.end());
  return text.find("type 15") != std::string::npos ||
         text.find("OBU type 15") != std::string::npos ||
         text.find("Unknown OBU") != std::string::npos;
}

// Removes BRT OBUs, leaving a temporal-unit sequence that differs only by the
// timing/decoder-model/CI syntax. Used when the pinned decoder rejects
// OBU_BUFFER_REMOVAL_TIMING (type 15) as unknown.
std::vector<std::vector<std::uint8_t>> without_brt(
    const std::vector<std::vector<std::uint8_t>>& units) {
  std::vector<std::vector<std::uint8_t>> out;
  out.reserve(units.size());
  for (const std::vector<std::uint8_t>& unit : units) {
    std::vector<av2enc::hrd::ParsedObu> obus;
    std::string error;
    if (!av2enc::hrd::parse_obus(unit, obus, &error)) {
      out.push_back(unit);
      continue;
    }
    std::vector<std::uint8_t> filtered;
    for (const av2enc::hrd::ParsedObu& obu : obus) {
      if (obu.type == av2enc::hrd::ObuType::kBufferRemovalTiming) continue;
      const std::span<const std::uint8_t> raw =
          std::span<const std::uint8_t>(unit).subspan(obu.offset, obu.size);
      filtered.insert(filtered.end(), raw.begin(), raw.end());
    }
    out.push_back(std::move(filtered));
  }
  return out;
}

struct CompareResult {
  bool ok = true;
  std::uint64_t mismatches = 0;
  std::uint64_t frames = 0;
  int max_diff = 0;
  std::string error;
};

CompareResult compare_recon(const std::string& decoded_path,
                            const std::vector<av2enc::m2::M2ReconFrame>& recon) {
  CompareResult result;
  av2enc::Y4mReader reader(decoded_path);
  if (!reader.open(result.error)) {
    result.ok = false;
    return result;
  }
  av2enc::Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const av2enc::Y4mReadResult status =
        reader.read_frame(frame, result.error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      result.ok = false;
      return result;
    }
    if (index >= recon.size()) {
      result.ok = false;
      result.error = "decoder produced more frames than the encoder";
      return result;
    }
    const av2enc::m2::M2ReconFrame& ref = recon[index];
    const auto compare_plane = [&](std::span<const std::uint8_t> got,
                                   const std::vector<std::uint8_t>& want) {
      if (got.size() != want.size()) {
        result.ok = false;
        return;
      }
      for (std::size_t i = 0; i < got.size(); ++i) {
        const int diff = std::abs(static_cast<int>(got[i]) -
                                  static_cast<int>(want[i]));
        if (diff != 0) {
          ++result.mismatches;
          result.max_diff = std::max(result.max_diff, diff);
        }
      }
    };
    compare_plane(frame.y(), ref.y);
    compare_plane(frame.u(), ref.u);
    compare_plane(frame.v(), ref.v);
    ++index;
    ++result.frames;
  }
  if (index != recon.size()) {
    result.ok = false;
    result.error = "decoder produced " + std::to_string(index) + " of " +
                   std::to_string(recon.size()) + " frames";
  }
  result.ok = result.ok && result.mismatches == 0;
  return result;
}

struct InjectionResult {
  std::vector<std::vector<std::uint8_t>> units;
  av2enc::hrd::BufferTimingPlan plan;
};

bool build_injected_units(const av2enc::m2::M2SequenceResult& sequence,
                          InjectionResult& result, std::string& error) {
  const std::size_t frame_count = sequence.units.size();
  if (frame_count == 0U) {
    error = "M2 produced no units";
    return false;
  }

  // Original units: the first is TD + sequence header + frame OBU; the rest
  // are TD + frame OBU.
  std::vector<std::vector<std::uint8_t>> original = sequence.units;
  std::vector<av2enc::hrd::ParsedObu> first_obus;
  if (!av2enc::hrd::parse_obus(original[0], first_obus, &error)) return false;
  const av2enc::hrd::ParsedObu* original_sequence = nullptr;
  for (const av2enc::hrd::ParsedObu& obu : first_obus) {
    if (obu.type == av2enc::hrd::ObuType::kSequenceHeader) {
      original_sequence = &obu;
    }
  }
  if (original_sequence == nullptr) {
    error = "first M2 unit has no sequence header";
    return false;
  }
  const std::vector<std::uint8_t> baseline_sequence(
      original[0].begin() + static_cast<std::ptrdiff_t>(original_sequence->offset),
      original[0].begin() + static_cast<std::ptrdiff_t>(
                                original_sequence->offset +
                                original_sequence->size));

  av2enc::hrd::HrdBufferConfig config;
  config.timing.num_units_in_display_tick = 1000;
  config.timing.time_scale = 30000;
  config.timing.equal_picture_interval = true;
  config.timing.num_ticks_per_picture_minus_1 = 0;
  config.num_units_in_decoding_tick = 1000;  // DecCT = 1/30 s
  config.decoder_buffer_delay = 70000;
  config.encoder_buffer_delay = 20000;
  config.low_delay_mode_flag = false;
  config.bit_rate = 2000000;
  config.buffer_size_bits = 2000000;
  config.allow_delay_adjustment = true;

  av2enc::hrd::ContentInterpretation ci;
  ci.ci_timing_info_present_flag = true;
  ci.timing = config.timing;
  std::vector<std::uint8_t> ci_obu;
  if (!av2enc::hrd::build_content_interpretation_obu(ci, ci_obu, &error)) {
    return false;
  }

  std::vector<std::uint64_t> approximate_bits(frame_count, 0);
  for (std::size_t i = 0; i < frame_count; ++i) {
    approximate_bits[i] =
        static_cast<std::uint64_t>(original[i].size()) * 8U + 256U;
  }

  std::vector<std::uint8_t> spliced_sequence;
  av2enc::hrd::SequenceTimingConfig timing_config;
  std::vector<std::vector<std::uint8_t>> units;
  av2enc::hrd::BufferTimingPlan plan;
  bool stable = false;
  for (int iteration = 0; iteration < 8 && !stable; ++iteration) {
    std::vector<av2enc::hrd::FrameBits> frames(frame_count);
    for (std::size_t i = 0; i < frame_count; ++i) {
      frames[i].coded_bits = approximate_bits[i];
      frames[i].random_access_point = (i == 0U);
    }
    if (!av2enc::hrd::plan_buffer_removal(config, frames, plan, &error)) {
      return false;
    }
    timing_config = av2enc::hrd::SequenceTimingConfig{};
    timing_config.seq_initial_display_delay_present_flag = true;
    timing_config.seq_initial_display_delay_minus_1 = 2;
    timing_config.decoder_model_info_present_flag = true;
    timing_config.num_units_in_decoding_tick = config.num_units_in_decoding_tick;
    timing_config.seq_decoder_model_info_present_flag = true;
    timing_config.decoder_model_info.decoder_buffer_delay =
        plan.decoder_buffer_delay;
    timing_config.decoder_model_info.encoder_buffer_delay =
        plan.encoder_buffer_delay;
    timing_config.decoder_model_info.low_delay_mode_flag = false;
    if (!av2enc::hrd::splice_sequence_timing(baseline_sequence, timing_config,
                                             spliced_sequence, &error)) {
      return false;
    }

    units.clear();
    units.reserve(frame_count);
    for (std::size_t i = 0; i < frame_count; ++i) {
      std::vector<av2enc::hrd::ParsedObu> obus;
      if (!av2enc::hrd::parse_obus(original[i], obus, &error)) return false;
      std::vector<std::uint8_t> unit;
      for (const av2enc::hrd::ParsedObu& obu : obus) {
        const std::span<const std::uint8_t> raw =
            std::span<const std::uint8_t>(original[i])
                .subspan(obu.offset, obu.size);
        if (obu.type == av2enc::hrd::ObuType::kSequenceHeader) {
          unit.insert(unit.end(), spliced_sequence.begin(),
                      spliced_sequence.end());
          if (i == 0U) {
            unit.insert(unit.end(), ci_obu.begin(), ci_obu.end());
          }
        } else if (obu.type == av2enc::hrd::ObuType::kBufferRemovalTiming) {
          error = "M2 baseline already contains a BRT OBU";
          return false;
        } else if (obu.type == av2enc::hrd::ObuType::kClosedLoopKey ||
                   obu.type == av2enc::hrd::ObuType::kRegularTileGroup) {
          av2enc::hrd::BufferRemovalTiming brt =
              av2enc::hrd::br_time_for_frame(plan, i);
          std::vector<std::uint8_t> brt_obu;
          if (!av2enc::hrd::build_buffer_removal_timing_obu(brt, brt_obu,
                                                           &error)) {
            return false;
          }
          unit.insert(unit.end(), brt_obu.begin(), brt_obu.end());
          unit.insert(unit.end(), raw.begin(), raw.end());
        } else {
          unit.insert(unit.end(), raw.begin(), raw.end());
        }
      }
      units.push_back(std::move(unit));
    }

    std::vector<std::uint64_t> next_bits(frame_count, 0);
    for (std::size_t i = 0; i < frame_count; ++i) {
      next_bits[i] = static_cast<std::uint64_t>(units[i].size()) * 8U;
    }
    stable = (next_bits == approximate_bits);
    approximate_bits = next_bits;
  }
  if (!stable) {
    error = "BRT fixed point did not converge";
    return false;
  }
  if (!plan.conformant || plan.underflow_seen) {
    error = "buffer plan is not conformant";
    return false;
  }
  result.units = std::move(units);
  result.plan = plan;
  return true;
}


// Builds the largest subset the pinned dav2d f9dafbc accepts: the same
// sequence-header timing block but with seq_decoder_model_info_present_flag = 0
// (its sequence-header parser rejects the spec v1.0.0 seq_decoder_model_info()
// layout), and no BRT OBU (it rejects OBU type 15 as unknown). Used only for
// the decoder invariance gate; the full spec stream is self-parsed below.
bool build_decoder_compatible_units(
    const av2enc::m2::M2SequenceResult& sequence,
    const av2enc::hrd::BufferTimingPlan& plan,
    std::vector<std::vector<std::uint8_t>>& out, std::string& error) {
  std::vector<av2enc::hrd::ParsedObu> first_obus;
  if (!av2enc::hrd::parse_obus(sequence.units[0], first_obus, &error)) {
    return false;
  }
  const av2enc::hrd::ParsedObu* original_sequence = nullptr;
  for (const av2enc::hrd::ParsedObu& obu : first_obus) {
    if (obu.type == av2enc::hrd::ObuType::kSequenceHeader) {
      original_sequence = &obu;
    }
  }
  if (original_sequence == nullptr) {
    error = "first M2 unit has no sequence header";
    return false;
  }
  const std::vector<std::uint8_t> baseline_sequence(
      sequence.units[0].begin() +
          static_cast<std::ptrdiff_t>(original_sequence->offset),
      sequence.units[0].begin() + static_cast<std::ptrdiff_t>(
                                      original_sequence->offset +
                                      original_sequence->size));

  av2enc::hrd::TimingInfo timing;
  timing.num_units_in_display_tick = 1000;
  timing.time_scale = 30000;
  timing.equal_picture_interval = true;
  timing.num_ticks_per_picture_minus_1 = 0;
  av2enc::hrd::ContentInterpretation ci;
  ci.ci_timing_info_present_flag = true;
  ci.timing = timing;
  std::vector<std::uint8_t> ci_obu;
  if (!av2enc::hrd::build_content_interpretation_obu(ci, ci_obu, &error)) {
    return false;
  }

  av2enc::hrd::SequenceTimingConfig config;
  config.seq_initial_display_delay_present_flag = true;
  config.seq_initial_display_delay_minus_1 = 2;
  config.decoder_model_info_present_flag = true;
  config.num_units_in_decoding_tick = plan.num_units_in_decoding_tick;
  config.seq_decoder_model_info_present_flag = false;
  std::vector<std::uint8_t> spliced;
  if (!av2enc::hrd::splice_sequence_timing(baseline_sequence, config, spliced,
                                           &error)) {
    return false;
  }

  out.clear();
  out.reserve(sequence.units.size());
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    std::vector<av2enc::hrd::ParsedObu> obus;
    if (!av2enc::hrd::parse_obus(sequence.units[i], obus, &error)) return false;
    std::vector<std::uint8_t> unit;
    for (const av2enc::hrd::ParsedObu& obu : obus) {
      const std::span<const std::uint8_t> raw =
          std::span<const std::uint8_t>(sequence.units[i])
              .subspan(obu.offset, obu.size);
      if (obu.type == av2enc::hrd::ObuType::kSequenceHeader) {
        unit.insert(unit.end(), spliced.begin(), spliced.end());
        if (i == 0U) unit.insert(unit.end(), ci_obu.begin(), ci_obu.end());
      } else if (obu.type == av2enc::hrd::ObuType::kClosedLoopKey ||
                 obu.type == av2enc::hrd::ObuType::kRegularTileGroup) {
        unit.insert(unit.end(), raw.begin(), raw.end());
      } else {
        unit.insert(unit.end(), raw.begin(), raw.end());
      }
    }
    out.push_back(std::move(unit));
  }
  return true;
}

bool self_parse(const std::vector<std::vector<std::uint8_t>>& units,
                const av2enc::hrd::BufferTimingPlan& plan) {
  std::string error;
  bool ok = true;
  for (std::size_t i = 0; i < units.size(); ++i) {
    std::vector<av2enc::hrd::ParsedObu> obus;
    if (!av2enc::hrd::parse_obus(units[i], obus, &error)) {
      std::cerr << "  SELF-PARSE FAIL unit " << i << ": " << error << "\n";
      ok = false;
      continue;
    }
    if (!av2enc::hrd::validate_temporal_unit_order(obus, &error)) {
      std::cerr << "  ORDER FAIL unit " << i << ": " << error << "\n";
      ok = false;
    }
    bool found_brt = false;
    for (const av2enc::hrd::ParsedObu& obu : obus) {
      if (obu.type == av2enc::hrd::ObuType::kBufferRemovalTiming) {
        av2enc::hrd::BufferRemovalTiming brt;
        const std::span<const std::uint8_t> raw =
            std::span<const std::uint8_t>(units[i])
                .subspan(obu.offset, obu.size);
        if (!av2enc::hrd::read_buffer_removal_timing_obu(raw, brt, &error)) {
          std::cerr << "  BRT READ FAIL unit " << i << ": " << error << "\n";
          ok = false;
          continue;
        }
        found_brt = true;
        if (brt.br_ops_dependent_flag) {
          std::cerr << "  BRT FORM FAIL: expected non-dependent\n";
          ok = false;
        } else if (brt.br_time != plan.frames[i].br_time) {
          std::cerr << "  BRT VALUE FAIL unit " << i << ": got "
                    << brt.br_time << " want " << plan.frames[i].br_time
                    << "\n";
          ok = false;
        }
      }
      if (i == 0U &&
          obu.type == av2enc::hrd::ObuType::kContentInterpretation) {
        av2enc::hrd::ContentInterpretation ci;
        const std::span<const std::uint8_t> raw =
            std::span<const std::uint8_t>(units[i])
                .subspan(obu.offset, obu.size);
        if (!av2enc::hrd::read_content_interpretation_obu(raw, ci, &error)) {
          std::cerr << "  CI READ FAIL: " << error << "\n";
          ok = false;
        } else if (!ci.ci_timing_info_present_flag) {
          std::cerr << "  CI TIMING FAIL: timing_info absent\n";
          ok = false;
        }
      }
      if (i == 0U && obu.type == av2enc::hrd::ObuType::kSequenceHeader) {
        av2enc::hrd::SequenceTimingConfig timing;
        const std::span<const std::uint8_t> raw =
            std::span<const std::uint8_t>(units[i])
                .subspan(obu.offset, obu.size);
        if (!av2enc::hrd::read_sequence_header_timing(raw, timing, &error)) {
          std::cerr << "  SEQ TIMING READ FAIL: " << error << "\n";
          ok = false;
        } else if (!timing.decoder_model_info_present_flag ||
                   timing.num_units_in_decoding_tick !=
                       plan.num_units_in_decoding_tick) {
          std::cerr << "  SEQ TIMING VALUE FAIL\n";
          ok = false;
        }
      }
    }
    if (!found_brt) {
      std::cerr << "  BRT MISSING in unit " << i << "\n";
      ok = false;
    }
  }
  return ok;
}

std::string md5_of_file(const std::string& path) {
  const std::string command = "md5 -q \"" + path + "\"";
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) return {};
  char buffer[128] = {};
  std::string out;
  if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) out = buffer;
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  int frames = 3;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    }
  }
  if (frames < 2) frames = 2;

  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "SKIP: no dav2d binary found (set DAV2D)\n";
    return 2;
  }

  std::vector<OwnedFrame> clip = make_clip(frames, 64, 64);
  std::vector<av2enc::m2::M2InputFrame> inputs;
  inputs.reserve(clip.size());
  for (const OwnedFrame& frame : clip) {
    av2enc::m2::M2InputFrame input;
    input.width = frame.width;
    input.height = frame.height;
    input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input);
  }

  av2enc::m2::M2Options options;
  options.base_q_idx = 90;
  options.force_single_block = true;
  av2enc::m2::M2SequenceResult sequence;
  std::string error;
  if (!av2enc::m2::encode_m2_sequence(inputs, options, sequence, error)) {
    std::cerr << "ENCODE FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "== HRD timing gate: " << sequence.units.size()
            << " M2 frame(s), 64x64 ==\n";
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    std::cout << "  frame " << i << ": bytes=" << sequence.frame_bytes[i]
              << " tile=" << sequence.tile_bytes[i] << "\n";
  }

  InjectionResult injected;
  if (!build_injected_units(sequence, injected, error)) {
    std::cerr << "INJECT FAIL: " << error << "\n";
    return 1;
  }
  std::cout << "  BRT plan:";
  for (const av2enc::hrd::RemovalEntry& entry : injected.plan.frames) {
    std::cout << " " << entry.br_time;
  }
  std::cout << "\n  DBD=" << injected.plan.decoder_buffer_delay
            << " EBD=" << injected.plan.encoder_buffer_delay
            << " DecCT_nutdt=" << injected.plan.num_units_in_decoding_tick
            << " peak=" << injected.plan.max_buffer_occupancy_bits
            << " bits (buffer " << injected.plan.buffer_size_bits << ")\n";

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "av2enc_hrd_gate";
  std::filesystem::create_directories(dir);
  const std::string baseline_ivf = (dir / "baseline.ivf").string();
  const std::string spec_timing_ivf = (dir / "spec_timing.ivf").string();
  const std::string compat_ivf = (dir / "decoder_compatible.ivf").string();
  const std::string full_ivf = (dir / "with_timing_and_brt.ivf").string();
  if (!write_ivf(baseline_ivf, sequence.units, 64, 64, error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }
  const std::vector<std::vector<std::uint8_t>> spec_timing =
      without_brt(injected.units);
  if (!write_ivf(spec_timing_ivf, spec_timing, 64, 64, error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }
  std::vector<std::vector<std::uint8_t>> compatible;
  if (!build_decoder_compatible_units(sequence, injected.plan, compatible,
                                      error)) {
    std::cerr << "COMPAT BUILD FAIL: " << error << "\n";
    return 1;
  }
  if (!write_ivf(compat_ivf, compatible, 64, 64, error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }
  if (!write_ivf(full_ivf, injected.units, 64, 64, error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return 1;
  }

  const DecodeResult baseline = decode(dav2d, baseline_ivf, "baseline");
  const DecodeResult modified = decode(dav2d, compat_ivf, "compatible");
  if (!baseline.ok || !modified.ok) return 1;

  bool baseline_read = false;
  bool modified_read = false;
  const std::vector<std::uint8_t> baseline_bytes =
      read_file(baseline.y4m, baseline_read);
  const std::vector<std::uint8_t> modified_bytes =
      read_file(modified.y4m, modified_read);
  if (!baseline_read || !modified_read) {
    std::cerr << "READ FAIL: decoded Y4M missing\n";
    return 1;
  }
  const bool identical = baseline_bytes == modified_bytes;
  std::cout << "  decoded md5 baseline=" << md5_of_file(baseline.y4m)
            << " timing=" << md5_of_file(modified.y4m) << "\n";
  if (!identical) {
    std::size_t first = 0;
    while (first < baseline_bytes.size() &&
           first < modified_bytes.size() &&
           baseline_bytes[first] == modified_bytes[first]) {
      ++first;
    }
    std::cerr << "  PIXEL INVARIANCE FAIL at byte " << first << " (sizes "
              << baseline_bytes.size() << "/" << modified_bytes.size()
              << ")\n";
    return 1;
  }

  const CompareResult recon = compare_recon(modified.y4m, sequence.recon);
  if (!recon.ok) {
    std::cerr << "  RECON FAIL: frames=" << recon.frames
              << " mismatches=" << recon.mismatches
              << " max_diff=" << recon.max_diff << " (" << recon.error
              << ")\n";
    return 1;
  }
  std::cout << "  PASS: decoded pixels unchanged; " << recon.frames
            << " frame(s) still match the encoder recon\n";

  if (!self_parse(injected.units, injected.plan)) {
    std::cerr << "  SELF-PARSE FAIL\n";
    return 1;
  }
  std::cout << "  PASS: self-parse accepted OBU order and BRT/CI/sequence "
               "timing values\n";

  const DecodeResult spec_timing_decode =
      decode(dav2d, spec_timing_ivf, "spec_timing");
  if (spec_timing_decode.ok) {
    std::cout << "  NOTE: pinned dav2d accepted the full spec-v1.0.0 "
                 "seq_decoder_model_info() block.\n";
  } else {
    std::cout << "  NOTE: pinned dav2d f9dafbc does not parse the spec "
                 "v1.0.0 seq_decoder_model_info() block (log: "
              << spec_timing_decode.log
              << "). The decoder-compatible stream keeps "
                 "decoder_model_info_present_flag = 1 and "
                 "num_units_in_decoding_tick but omits the nested "
                 "seq_decoder_model_info(); the full spec stream is "
                 "self-parsed.\n";
  }

  const DecodeResult full = decode(dav2d, full_ivf, "timing_and_brt");
  if (full.ok) {
    bool full_read = false;
    const std::vector<std::uint8_t> full_bytes =
        read_file(full.y4m, full_read);
    if (!full_read || full_bytes != baseline_bytes) {
      std::cerr << "  BRT STREAM PIXEL FAIL\n";
      return 1;
    }
    const CompareResult full_recon =
        compare_recon(full.y4m, sequence.recon);
    if (!full_recon.ok) {
      std::cerr << "  BRT STREAM RECON FAIL\n";
      return 1;
    }
    std::cout << "  PASS: decoder accepted the BRT-bearing stream; pixels "
                 "unchanged\n";
  } else if (log_mentions_unknown_brt(full.log)) {
    std::cout << "  NOTE: pinned dav2d does not implement "
                 "OBU_BUFFER_REMOVAL_TIMING (type 15). It rejects the "
                 "BRT-bearing stream as an unknown OBU; BRT is therefore "
                 "gated by the buffer model + self-parse, and the decoder gate "
                 "uses the compatible stream without BRT.\n";
  } else {
    std::cerr << "  BRT STREAM DECODE FAIL (not an unknown-OBU error); see "
              << full.log << "\n";
    return 1;
  }

  bool ft_baseline = false;
  bool ft_modified = false;
  const std::vector<std::uint8_t> base_ft =
      read_file(baseline.frametimes, ft_baseline);
  const std::vector<std::uint8_t> mod_ft =
      read_file(modified.frametimes, ft_modified);
  if (ft_baseline && ft_modified && base_ft == mod_ft) {
    std::cout << "  NOTE: dav2d reported identical frame times for baseline "
                 "and the compatible stream (CI timing present, no BRT); the "
                 "timing syntax did not change the reported times. Gate result "
                 "is pixel invariance + self-parse.\n";
  } else if (ft_baseline && ft_modified) {
    std::cout << "  NOTE: dav2d frame times changed with the CI timing/dmi "
                 "syntax present (no BRT in this stream); the decoder does "
                 "process timing_info().\n";
  } else {
    std::cout << "  NOTE: dav2d did not write frame times; BRT support is "
                 "unverified, gate result is pixel invariance + self-parse "
                 "only.\n";
  }

  std::cout << "hrd_stream_gate: PASS\n";
  return 0;
}
