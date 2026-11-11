// append_content_interpretation: insert an AV2 v1.0.0 content interpretation
// OBU into an existing IVF stream, or scan a stream for CI OBUs.
//
// This is the decoder-gate helper for src/encoder/metadata/content_interpretation/.
// It does not encode. It takes a stream produced by av2enc and inserts an
// already-serialized OBU at the §7.3.3 position: the first OBU of the coded
// output frame unit, i.e. after the temporal delimiter and sequence header and
// before the first CLK/OLK/frame OBU.
//
// Only the selected IVF frame payload changes; the file header, timestamps and
// all other frames are copied verbatim.

#include "encoder/metadata/content_interpretation/content_interpretation.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using av2enc::ci::AspectRatio;
using av2enc::ci::ChromaFormat;
using av2enc::ci::ChromaSamplePosition;
using av2enc::ci::ColorDescription;
using av2enc::ci::ContentInterpretation;
using av2enc::ci::ContentInterpretationScanResult;
using av2enc::ci::ScanType;
using av2enc::ci::TimingInfo;

constexpr std::size_t kIvfHeaderBytes = 32;
constexpr std::size_t kIvfFrameHeaderBytes = 12;

struct IvfFrame {
  std::vector<std::uint8_t> payload;
  std::uint64_t timestamp = 0;
};

struct IvfFile {
  std::array<std::uint8_t, kIvfHeaderBytes> header{};
  std::vector<IvfFrame> frames;
};

std::uint32_t read_u32_le(const std::uint8_t* data) {
  return static_cast<std::uint32_t>(data[0]) |
         (static_cast<std::uint32_t>(data[1]) << 8U) |
         (static_cast<std::uint32_t>(data[2]) << 16U) |
         (static_cast<std::uint32_t>(data[3]) << 24U);
}

std::uint64_t read_u64_le(const std::uint8_t* data) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8U; ++index) {
    value |= static_cast<std::uint64_t>(data[index]) << (8U * index);
  }
  return value;
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (unsigned index = 0; index < 8U; ++index) {
    out.push_back(static_cast<std::uint8_t>((value >> (8U * index)) & 0xFFU));
  }
}

bool load_ivf(const std::string& path, IvfFile& out, std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "cannot open " + path;
    return false;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() < kIvfHeaderBytes) {
    error = "file is shorter than an IVF header";
    return false;
  }
  if (std::memcmp(bytes.data(), "DKIF", 4U) != 0) {
    error = "not an IVF file (missing DKIF)";
    return false;
  }
  std::memcpy(out.header.data(), bytes.data(), kIvfHeaderBytes);

  std::size_t offset = kIvfHeaderBytes;
  while (offset < bytes.size()) {
    if (bytes.size() - offset < kIvfFrameHeaderBytes) {
      error = "truncated IVF frame header";
      return false;
    }
    const std::uint32_t size = read_u32_le(bytes.data() + offset);
    const std::uint64_t timestamp = read_u64_le(bytes.data() + offset + 4U);
    offset += kIvfFrameHeaderBytes;
    if (size > bytes.size() - offset) {
      error = "IVF frame size extends past the end of the file";
      return false;
    }
    IvfFrame frame;
    frame.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                         bytes.begin() + static_cast<std::ptrdiff_t>(offset + size));
    frame.timestamp = timestamp;
    out.frames.push_back(std::move(frame));
    offset += size;
  }
  return true;
}

bool write_ivf(const std::string& path, const IvfFile& ivf, std::string& error) {
  std::vector<std::uint8_t> bytes;
  bytes.insert(bytes.end(), ivf.header.begin(), ivf.header.end());
  for (const IvfFrame& frame : ivf.frames) {
    append_u32_le(bytes, static_cast<std::uint32_t>(frame.payload.size()));
    append_u64_le(bytes, frame.timestamp);
    bytes.insert(bytes.end(), frame.payload.begin(), frame.payload.end());
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    error = "cannot open " + path + " for writing";
    return false;
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    error = "failed while writing " + path;
    return false;
  }
  return true;
}

const char* chroma_format_name(ChromaFormat format) {
  return format == ChromaFormat::k420 ? "4:2:0" : "4:2:2";
}

void print_params(const ContentInterpretation& params, std::ostream& out,
                  ChromaFormat format) {
  out << "scan_type=" << static_cast<unsigned>(params.scan_type) << " ("
      << av2enc::ci::scan_type_name(params.scan_type) << ")";
  if (params.has_color_description) {
    out << " colour{idc=" << static_cast<unsigned>(params.color.idc)
        << " primaries=" << static_cast<unsigned>(params.color.primaries)
        << " transfer=" << static_cast<unsigned>(params.color.transfer)
        << " matrix=" << static_cast<unsigned>(params.color.matrix)
        << " full_range=" << (params.color.full_range ? 1 : 0) << "}";
  } else {
    out << " colour{absent}";
  }
  if (params.has_chroma_sample_position) {
    out << " chroma{top="
        << av2enc::ci::chroma_sample_position_name(params.chroma_top) << "("
        << static_cast<unsigned>(params.chroma_top) << ")"
        << " bottom="
        << av2enc::ci::chroma_sample_position_name(params.chroma_bottom) << "("
        << static_cast<unsigned>(params.chroma_bottom) << ")"
        << " table=" << chroma_format_name(format) << "}";
  } else {
    out << " chroma{absent}";
  }
  if (params.has_aspect_ratio) {
    out << " sar{idc=" << static_cast<unsigned>(params.aspect.idc) << " "
        << params.aspect.resolved_width() << ":"
        << params.aspect.resolved_height() << "}";
  } else {
    out << " sar{absent}";
  }
  if (params.has_timing_info) {
    out << " timing{" << params.timing.num_units_in_display_tick << "/"
        << params.timing.time_scale
        << " equal_interval=" << (params.timing.equal_picture_interval ? 1 : 0)
        << " ticks_minus_1=" << params.timing.num_ticks_per_picture_minus_1
        << "}";
  } else {
    out << " timing{absent}";
  }
}

bool parse_u32(const std::string& text, std::uint32_t& out) {
  try {
    std::size_t consumed = 0;
    const unsigned long value = std::stoul(text, &consumed, 0);
    if (consumed != text.size()) {
      return false;
    }
    out = static_cast<std::uint32_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_ratio(const std::string& text, std::uint32_t& width,
                 std::uint32_t& height) {
  const std::size_t colon = text.find(':');
  const std::size_t x = text.find('x');
  const std::size_t split = colon != std::string::npos ? colon : x;
  if (split == std::string::npos) {
    return false;
  }
  return parse_u32(text.substr(0, split), width) &&
         parse_u32(text.substr(split + 1), height);
}

void usage(std::ostream& out) {
  out << "Usage:\n"
         "  append_content_interpretation <in.ivf> --scan [--strict] "
         "[--chroma-format 420|422]\n"
         "  append_content_interpretation <in.ivf> -o <out.ivf> [options]\n"
         "\n"
         "Insert mode options (defaults produce a BT.709 SDR progressive "
         "OBU):\n"
         "  --frame N               temporal unit to modify (default 0)\n"
         "  --scan-type 0..3        0 unspecified, 1 progressive,\n"
         "                          2 interlace field, 3 complementary pair\n"
         "  --color-idc N           ci_color_description_idc (default 1)\n"
         "  --primaries N           explicit ci_color_primaries (idc 0 only)\n"
         "  --transfer N            explicit ci_transfer_characteristics\n"
         "  --matrix N              explicit ci_matrix_coefficients\n"
         "  --full-range            ci_full_range_flag = 1\n"
         "  --no-color              ci_color_description_present_flag = 0\n"
         "  --chroma-top N          ci_chroma_sample_position_top (0..5)\n"
         "  --chroma-bottom N       ci_chroma_sample_position_bottom (0..5)\n"
         "  --no-chroma             chroma sample position absent\n"
         "  --aspect-idc N          ci_aspect_ratio_idc (0..16 or 255)\n"
         "  --sar WxH               explicit SAR, implies --aspect-idc 255\n"
         "  --no-aspect             aspect ratio absent\n"
         "  --num-units N           num_units_in_display_tick (default 1)\n"
         "  --time-scale N          time_scale (default 30)\n"
         "  --equal-interval        equal_picture_interval = 1\n"
         "  --ticks-minus-1 N       num_ticks_per_picture_minus_1\n"
         "  --no-timing             timing_info absent\n"
         "  --chroma-format 420|422 chroma sample position table for checks\n"
         "  --strict                reject reserved/inconsistent values\n"
         "  --probe-invalid-chroma  write a NON-CONFORMING CI OBU with\n"
         "                          ci_chroma_sample_position_top = 6; used only\n"
         "                          by run_ci_gate.sh to probe decoder support\n"
         "  --probe-truncated       write an OBU with an empty payload; tests\n"
         "                          whether the decoder parses the CI grammar\n";
}

struct Options {
  std::string input;
  std::string output;
  bool scan = false;
  std::size_t frame = 0;
  bool strict = false;
  bool probe_invalid_chroma = false;
  bool probe_truncated = false;
  ChromaFormat format = ChromaFormat::k420;

  ScanType scan_type = ScanType::kProgressive;
  bool has_color = true;
  std::uint8_t color_idc = 1;
  std::uint8_t primaries = 1;
  std::uint8_t transfer = 1;
  std::uint8_t matrix = 1;
  bool full_range = false;
  bool has_chroma = true;
  std::uint8_t chroma_top = 1;
  std::uint8_t chroma_bottom = 1;
  bool has_aspect = true;
  std::uint8_t aspect_idc = 1;
  std::uint32_t sar_width = 0;
  std::uint32_t sar_height = 0;
  bool has_timing = true;
  std::uint32_t num_units = 1;
  std::uint32_t time_scale = 30;
  bool equal_interval = false;
  std::uint32_t ticks_minus_1 = 0;
};

bool parse_options(int argc, char** argv, Options& options, std::string& error) {
  std::vector<std::string> positional;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto next = [&](std::string& value) {
      if (index + 1 >= argc) {
        error = "missing value for " + argument;
        return false;
      }
      value = argv[++index];
      return true;
    };
    std::string value;
    if (argument == "--scan") {
      options.scan = true;
    } else if (argument == "--strict") {
      options.strict = true;
    } else if (argument == "--probe-invalid-chroma") {
      options.probe_invalid_chroma = true;
    } else if (argument == "--probe-truncated") {
      options.probe_truncated = true;
    } else if (argument == "-o") {
      if (!next(options.output)) return false;
    } else if (argument == "-h" || argument == "--help") {
      usage(std::cout);
      std::exit(0);
    } else if (argument == "--frame") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed)) {
        error = "bad --frame value";
        return false;
      }
      options.frame = parsed;
    } else if (argument == "--scan-type") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 3U) {
        error = "bad --scan-type value";
        return false;
      }
      options.scan_type = static_cast<ScanType>(parsed);
    } else if (argument == "--color-idc") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 127U) {
        error = "bad --color-idc value";
        return false;
      }
      options.color_idc = static_cast<std::uint8_t>(parsed);
      options.has_color = true;
    } else if (argument == "--primaries") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 255U) return false;
      options.primaries = static_cast<std::uint8_t>(parsed);
      options.color_idc = 0;
      options.has_color = true;
    } else if (argument == "--transfer") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 255U) return false;
      options.transfer = static_cast<std::uint8_t>(parsed);
      options.color_idc = 0;
      options.has_color = true;
    } else if (argument == "--matrix") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 255U) return false;
      options.matrix = static_cast<std::uint8_t>(parsed);
      options.color_idc = 0;
      options.has_color = true;
    } else if (argument == "--full-range") {
      options.full_range = true;
    } else if (argument == "--no-color") {
      options.has_color = false;
    } else if (argument == "--chroma-top") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 6U) {
        error = "bad --chroma-top value";
        return false;
      }
      options.chroma_top = static_cast<std::uint8_t>(parsed);
      options.has_chroma = true;
    } else if (argument == "--chroma-bottom") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 6U) {
        error = "bad --chroma-bottom value";
        return false;
      }
      options.chroma_bottom = static_cast<std::uint8_t>(parsed);
      options.has_chroma = true;
    } else if (argument == "--no-chroma") {
      options.has_chroma = false;
    } else if (argument == "--aspect-idc") {
      if (!next(value)) return false;
      std::uint32_t parsed = 0;
      if (!parse_u32(value, parsed) || parsed > 255U) {
        error = "bad --aspect-idc value";
        return false;
      }
      options.aspect_idc = static_cast<std::uint8_t>(parsed);
      options.has_aspect = true;
    } else if (argument == "--sar") {
      if (!next(value)) return false;
      if (!parse_ratio(value, options.sar_width, options.sar_height)) {
        error = "bad --sar value (expected WxH or W:H)";
        return false;
      }
      options.aspect_idc = 255;
      options.has_aspect = true;
    } else if (argument == "--no-aspect") {
      options.has_aspect = false;
    } else if (argument == "--num-units") {
      if (!next(value) || !parse_u32(value, options.num_units)) return false;
    } else if (argument == "--time-scale") {
      if (!next(value) || !parse_u32(value, options.time_scale)) return false;
    } else if (argument == "--equal-interval") {
      options.equal_interval = true;
    } else if (argument == "--ticks-minus-1") {
      if (!next(value) || !parse_u32(value, options.ticks_minus_1)) return false;
    } else if (argument == "--no-timing") {
      options.has_timing = false;
    } else if (argument == "--chroma-format") {
      if (!next(value)) return false;
      if (value == "420") {
        options.format = ChromaFormat::k420;
      } else if (value == "422") {
        options.format = ChromaFormat::k422;
      } else {
        error = "--chroma-format must be 420 or 422";
        return false;
      }
    } else if (!argument.empty() && argument[0] == '-') {
      error = "unknown option " + argument;
      return false;
    } else {
      positional.push_back(argument);
    }
  }

  if (positional.size() == 1) {
    options.input = positional[0];
  } else if (positional.size() == 2 && options.output.empty()) {
    options.input = positional[0];
    options.output = positional[1];
  } else if (!positional.empty()) {
    error = "expected one input file";
    return false;
  }
  if (options.input.empty()) {
    error = "missing input IVF";
    return false;
  }
  if (!options.scan && options.output.empty()) {
    error = "insert mode requires -o <out.ivf>";
    return false;
  }
  return true;
}

ContentInterpretation build_params(const Options& options) {
  ContentInterpretation params;
  params.scan_type = options.scan_type;
  params.has_color_description = options.has_color;
  params.color.idc = options.color_idc;
  if (options.color_idc == 0U) {
    params.color.primaries = options.primaries;
    params.color.transfer = options.transfer;
    params.color.matrix = options.matrix;
  } else {
    params.color = av2enc::ci::derive_color_description(options.color_idc);
    params.color.idc = options.color_idc;
  }
  params.color.full_range = options.full_range;
  params.has_chroma_sample_position = options.has_chroma;
  params.chroma_top = static_cast<ChromaSamplePosition>(options.chroma_top);
  params.chroma_bottom =
      static_cast<ChromaSamplePosition>(options.chroma_bottom);
  params.has_aspect_ratio = options.has_aspect;
  params.aspect.idc = options.aspect_idc;
  if (options.aspect_idc == 255U) {
    params.aspect.sar_width = options.sar_width;
    params.aspect.sar_height = options.sar_height;
  } else if (options.aspect_idc <= 16U) {
    params.aspect.sar_width =
        av2enc::ci::kAspectRatioWidth[options.aspect_idc];
    params.aspect.sar_height =
        av2enc::ci::kAspectRatioHeight[options.aspect_idc];
  }
  params.has_timing_info = options.has_timing;
  params.timing.num_units_in_display_tick = options.num_units;
  params.timing.time_scale = options.time_scale;
  params.timing.equal_picture_interval = options.equal_interval;
  params.timing.num_ticks_per_picture_minus_1 = options.ticks_minus_1;

  // Keep absent fields at their default values so the validator does not warn
  // about values that the syntax will not code.
  if (!params.has_color_description) {
    params.color = ColorDescription{};
  }
  if (!params.has_chroma_sample_position) {
    params.chroma_top = ChromaSamplePosition::kUnspecified;
    params.chroma_bottom = ChromaSamplePosition::kUnspecified;
  }
  if (!params.has_aspect_ratio) {
    params.aspect = AspectRatio{};
  }
  if (!params.has_timing_info) {
    params.timing = TimingInfo{};
  }
  return params;
}

int run_scan(const IvfFile& ivf, const Options& options) {
  std::size_t total = 0;
  std::size_t invalid = 0;
  for (std::size_t index = 0; index < ivf.frames.size(); ++index) {
    ContentInterpretationScanResult result;
    std::string error;
    if (!av2enc::ci::scan_content_interpretation(ivf.frames[index].payload,
                                                 result, error, true,
                                                 options.format)) {
      std::cerr << "frame " << index << ": scan failed: " << error << '\n';
      return 1;
    }
    bool seen_ci = false;
    for (const auto& record : result.records) {
      ++total;
      std::cout << "frame " << index << ": CI OBU at byte "
                << record.obu_offset << " size " << record.obu_size;
      if (record.before_first_frame) {
        std::cout << " (before first frame: ok)";
      } else {
        std::cout << " (AFTER first frame: violates §7.3.3)";
      }
      if (seen_ci) {
        std::cout << " (duplicate CI OBU in one temporal unit)";
      }
      seen_ci = true;
      std::cout << '\n';
      if (!record.valid) {
        std::cout << "  invalid payload (rejected by the §5.15/§6.14 checks)\n";
        continue;
      }
      std::cout << "  syntax_bits=" << record.view.syntax_bits
                << " reserved_bits="
                << static_cast<unsigned>(record.view.reserved_bits) << '\n';
      std::cout << "  ";
      print_params(record.view.params, std::cout, options.format);
      std::cout << '\n';

      // Byte-exact self round-trip: re-serialize the parsed parameters and
      // compare with the OBU that is in the stream.
      std::string roundtrip_error;
      const std::vector<std::uint8_t> reencoded =
          av2enc::ci::make_content_interpretation_obu(
              record.view.params, roundtrip_error, nullptr, options.format,
              false);
      const auto original = std::span<const std::uint8_t>(
          ivf.frames[index].payload.data() + record.obu_offset,
          record.obu_size);
      const bool same =
          roundtrip_error.empty() &&
          original.size() == reencoded.size() &&
          std::equal(original.begin(), original.end(), reencoded.begin());
      std::cout << "  round-trip: " << (same ? "ok" : "MISMATCH") << '\n';
      if (!same) {
        ++invalid;
      }
    }
    invalid += result.invalid_ci_obus;
  }
  std::cout << "frames: " << ivf.frames.size() << ", CI OBUs: " << total
            << ", invalid: " << invalid << '\n';
  return invalid == 0U ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string error;
  if (!parse_options(argc, argv, options, error)) {
    std::cerr << "append_content_interpretation: " << error << "\n\n";
    usage(std::cerr);
    return 2;
  }

  IvfFile ivf;
  if (!load_ivf(options.input, ivf, error)) {
    std::cerr << "append_content_interpretation: " << error << '\n';
    return 1;
  }
  if (options.scan) {
    return run_scan(ivf, options);
  }

  if (options.frame >= ivf.frames.size()) {
    std::cerr << "append_content_interpretation: --frame " << options.frame
              << " is out of range (stream has " << ivf.frames.size()
              << " frame(s))\n";
    return 1;
  }

  const ContentInterpretation params = build_params(options);
  std::size_t payload_bits = 0;
  std::vector<std::uint8_t> ci_obu;
  if (options.probe_invalid_chroma && options.probe_truncated) {
    std::cerr << "append_content_interpretation: choose one probe mode\n";
    return 2;
  }
  if (options.probe_truncated) {
    // Annex B.2 leb128 size 1: the OBU header only, no payload. Any decoder
    // that runs content_interpretation_obu() must overrun the bit buffer.
    ci_obu = {0x01U, 0x60U};
    std::cerr << "append_content_interpretation: WARNING: writing a truncated "
                 "CI probe OBU (no payload); decoder gate use only\n";
  } else if (options.probe_invalid_chroma) {
    ci_obu = av2enc::ci::make_nonconforming_chroma_probe_obu(error);
    std::cerr << "append_content_interpretation: WARNING: writing a "
                 "non-conforming CI probe OBU (chroma position 6); decoder "
                 "gate use only\n";
  } else {
    ci_obu = av2enc::ci::make_content_interpretation_obu(
        params, error, &payload_bits, options.format, options.strict);
  }
  if (!error.empty()) {
    std::cerr << "append_content_interpretation: " << error << '\n';
    return 1;
  }

  std::vector<std::uint8_t> patched;
  std::size_t inserted_at = 0;
  if (!av2enc::ci::insert_content_interpretation(
          ivf.frames[options.frame].payload, ci_obu, patched, error,
          &inserted_at)) {
    std::cerr << "append_content_interpretation: " << error << '\n';
    return 1;
  }

  std::cout << "input:   " << options.input << '\n'
            << "frame:   " << options.frame << '\n'
            << "obu:     " << ci_obu.size() << " bytes (" << payload_bits
            << " payload bits) at temporal-unit offset " << inserted_at << '\n'
            << "format:  " << chroma_format_name(options.format) << '\n';
  if (options.probe_invalid_chroma) {
    std::cout << "params:  NON-CONFORMING PROBE (chroma top = 6)\n";
  } else if (options.probe_truncated) {
    std::cout << "params:  TRUNCATED PROBE (empty payload)\n";
  } else {
    std::cout << "params:  ";
    print_params(params, std::cout, options.format);
    std::cout << '\n';
  }

  ivf.frames[options.frame].payload = std::move(patched);
  if (!write_ivf(options.output, ivf, error)) {
    std::cerr << "append_content_interpretation: " << error << '\n';
    return 1;
  }
  std::cout << "output:  " << options.output << '\n';
  return 0;
}
