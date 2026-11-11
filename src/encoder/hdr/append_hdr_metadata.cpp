// append_hdr_metadata: insert AV2 v1.0.0 HDR metadata OBUs into an existing
// IVF stream, or scan a stream for CLL/MDCV OBUs.
//
// This is the decoder-gate helper for src/encoder/hdr/. It does not encode:
// it takes a stream produced by av2enc and inserts already-serialized OBUs at
// a spec-conformant position:
//
//   prefix (default): immediately before the first frame OBU, i.e. after the
//                     sequence header (§7.3.3 coded output frame unit)
//   suffix (--suffix): after the last OBU of the temporal unit (§7.3.3)
//
// Only the selected IVF frame payload changes; the file header, timestamps and
// all other frames are copied verbatim.

#include "encoder/hdr/hdr_metadata.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using av2enc::hdr::ContentLightLevel;
using av2enc::hdr::HdrMetadata;
using av2enc::hdr::HdrObuPacking;
using av2enc::hdr::HdrScanResult;
using av2enc::hdr::MasteringDisplayColorVolume;
using av2enc::hdr::MetadataObuConfig;
using av2enc::hdr::MetadataPlacement;

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
  return static_cast<std::uint64_t>(read_u32_le(data)) |
         (static_cast<std::uint64_t>(read_u32_le(data + 4)) << 32U);
}

void write_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
  out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

void write_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
  write_u32_le(out, static_cast<std::uint32_t>(value & 0xFFFFFFFFU));
  write_u32_le(out, static_cast<std::uint32_t>(value >> 32U));
}

bool read_file(const std::filesystem::path& path, std::vector<std::uint8_t>& out,
               std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open input: " + path.string();
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(in),
             std::istreambuf_iterator<char>());
  if (!in.good() && !in.eof()) {
    error = "failed to read input: " + path.string();
    return false;
  }
  return true;
}

bool write_file(const std::filesystem::path& path,
                std::span<const std::uint8_t> bytes, std::string& error) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "cannot open output: " + path.string();
    return false;
  }
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    error = "failed to write output: " + path.string();
    return false;
  }
  return true;
}

bool parse_ivf(std::span<const std::uint8_t> bytes, IvfFile& out,
               std::string& error) {
  if (bytes.size() < kIvfHeaderBytes) {
    error = "input is shorter than the 32-byte IVF header";
    return false;
  }
  if (std::memcmp(bytes.data(), "DKIF", 4) != 0) {
    error = "input is not an IVF file (missing DKIF)";
    return false;
  }
  out.header.fill(0);
  std::memcpy(out.header.data(), bytes.data(), kIvfHeaderBytes);
  out.frames.clear();

  std::size_t pos = kIvfHeaderBytes;
  while (pos < bytes.size()) {
    if (bytes.size() - pos < kIvfFrameHeaderBytes) {
      error = "truncated IVF frame header";
      return false;
    }
    const std::uint32_t size = read_u32_le(bytes.data() + pos);
    const std::uint64_t timestamp = read_u64_le(bytes.data() + pos + 4U);
    pos += kIvfFrameHeaderBytes;
    if (static_cast<std::size_t>(size) > bytes.size() - pos) {
      error = "truncated IVF frame payload";
      return false;
    }
    IvfFrame frame;
    frame.timestamp = timestamp;
    frame.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(pos),
                         bytes.begin() +
                             static_cast<std::ptrdiff_t>(pos + size));
    pos += size;
    out.frames.push_back(std::move(frame));
  }
  return true;
}

bool serialize_ivf(const IvfFile& file, std::vector<std::uint8_t>& out,
                   std::string& error) {
  out.clear();
  out.insert(out.end(), file.header.begin(), file.header.end());
  for (const auto& frame : file.frames) {
    if (frame.payload.size() > std::numeric_limits<std::uint32_t>::max()) {
      error = "IVF frame payload exceeds the 32-bit size field";
      return false;
    }
    write_u32_le(out, static_cast<std::uint32_t>(frame.payload.size()));
    write_u64_le(out, frame.timestamp);
    out.insert(out.end(), frame.payload.begin(), frame.payload.end());
  }
  return true;
}

std::optional<std::uint32_t> parse_u32(std::string_view text) {
  if (text.empty()) {
    return std::nullopt;
  }
  std::uint64_t value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9') {
      return std::nullopt;
    }
    value = value * 10U + static_cast<unsigned>(c - '0');
    if (value > std::numeric_limits<std::uint32_t>::max()) {
      return std::nullopt;
    }
  }
  return static_cast<std::uint32_t>(value);
}

std::vector<std::string_view> split(std::string_view text, char delimiter) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t next = text.find(delimiter, start);
    if (next == std::string_view::npos) {
      parts.push_back(text.substr(start));
      break;
    }
    parts.push_back(text.substr(start, next - start));
    start = next + 1U;
  }
  return parts;
}

bool preset_mdcv(std::string_view name, MasteringDisplayColorVolume& out,
                 std::string& error) {
  out = MasteringDisplayColorVolume{};
  if (name == "bt2020-1000") {
    out.primary_chromaticity_x = {8500, 6550, 35400};
    out.primary_chromaticity_y = {39850, 2300, 14600};
    out.white_point_chromaticity_x = 15635;
    out.white_point_chromaticity_y = 16450;
    out.luminance_max = 10000000;  // 1000 cd/m^2
    out.luminance_min = 50;        // 0.005 cd/m^2
    return true;
  }
  if (name == "bt2020-4000") {
    out.primary_chromaticity_x = {8500, 6550, 35400};
    out.primary_chromaticity_y = {39850, 2300, 14600};
    out.white_point_chromaticity_x = 15635;
    out.white_point_chromaticity_y = 16450;
    out.luminance_max = 40000000;  // 4000 cd/m^2
    out.luminance_min = 50;
    return true;
  }
  if (name == "p3-1000") {
    // Display P3 (D65 white, P3 primaries).
    out.primary_chromaticity_x = {13250, 7500, 34000};
    out.primary_chromaticity_y = {34500, 3000, 16000};
    out.white_point_chromaticity_x = 15635;
    out.white_point_chromaticity_y = 16450;
    out.luminance_max = 10000000;
    out.luminance_min = 50;
    return true;
  }
  if (name == "unknown") {
    return true;  // all zero = unknown/unspecified
  }
  error = "unknown MDCV preset '" + std::string(name) + "'";
  return false;
}

bool parse_mdcv_spec(std::string_view text, MasteringDisplayColorVolume& out,
                     std::string& error) {
  const std::vector<std::string_view> parts = split(text, ',');
  if (parts.size() != 10U) {
    error =
        "--mdcv expects 10 comma-separated integers: "
        "Gx,Gy,Bx,By,Rx,Ry,Wx,Wy,Lmax,Lmin";
    return false;
  }
  std::array<std::uint32_t, 10> values{};
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const auto value = parse_u32(parts[i]);
    if (!value) {
      error = "invalid integer in --mdcv";
      return false;
    }
    values[i] = *value;
  }
  for (std::size_t i = 0; i < 8U; ++i) {
    if (values[i] > 0xFFFFU) {
      error = "chromaticity values must fit in 16 bits";
      return false;
    }
  }
  out.primary_chromaticity_x = {static_cast<std::uint16_t>(values[0]),
                                static_cast<std::uint16_t>(values[2]),
                                static_cast<std::uint16_t>(values[4])};
  out.primary_chromaticity_y = {static_cast<std::uint16_t>(values[1]),
                                static_cast<std::uint16_t>(values[3]),
                                static_cast<std::uint16_t>(values[5])};
  out.white_point_chromaticity_x = static_cast<std::uint16_t>(values[6]);
  out.white_point_chromaticity_y = static_cast<std::uint16_t>(values[7]);
  out.luminance_max = values[8];
  out.luminance_min = values[9];
  return true;
}

void print_usage(std::ostream& out) {
  out <<
      "Usage:\n"
      "  append_hdr_metadata --input <in.ivf> --output <out.ivf> [options]\n"
      "  append_hdr_metadata --input <in.ivf> --scan\n"
      "\n"
      "Options:\n"
      "  --cll <max_cll>:<max_fall>  CLL values in cd/m^2 (default 1000:400)\n"
      "  --no-cll                    Do not insert a CLL OBU\n"
      "  --mdcv <preset|spec>        bt2020-1000 (default), bt2020-4000,\n"
      "                              p3-1000, unknown, none, or\n"
      "                              Gx,Gy,Bx,By,Rx,Ry,Wx,Wy,Lmax,Lmin\n"
      "  --group                     One OBU_METADATA_GROUP instead of two\n"
      "                              OBU_METADATA_SHORT OBUs\n"
      "  --suffix                    Suffix metadata after the frame instead\n"
      "                              of prefix metadata before it\n"
      "  --legacy-av1                NON-CONFORMANT AV1-style metadata OBU\n"
      "                              layout for probing the pinned dav2d build\n"
      "  --frame <n>                 IVF frame to modify (default 0)\n"
      "  --strict-ranges             Reject out-of-range MDCV values instead\n"
      "                              of encoding them as 'unknown'\n"
      "  --scan                      Print HDR metadata found in the input\n"
      "  -h, --help                  Show this help\n";
}

int run_scan(const IvfFile& file) {
  std::size_t total_obus = 0;
  std::size_t total_units = 0;
  std::size_t total_unknown = 0;
  std::size_t total_invalid = 0;
  std::size_t total_records = 0;

  std::cout << "scan: frames=" << file.frames.size() << '\n';
  for (std::size_t frame_index = 0; frame_index < file.frames.size();
       ++frame_index) {
    const auto& payload = file.frames[frame_index].payload;
    HdrScanResult scan;
    std::string error;
    if (!av2enc::hdr::scan_hdr_metadata(payload, scan, error)) {
      std::cerr << "scan: frame " << frame_index
                << ": malformed OBU stream: " << error << '\n';
      return 1;
    }
    total_obus += scan.metadata_obus;
    total_units += scan.metadata_units;
    total_unknown += scan.unknown_units;
    total_invalid += scan.invalid_units;
    total_records += scan.records.size();

    std::cout << "scan: frame=" << frame_index
              << " payload=" << payload.size()
              << " metadata_obus=" << scan.metadata_obus
              << " metadata_units=" << scan.metadata_units
              << " unknown_units=" << scan.unknown_units
              << " invalid_units=" << scan.invalid_units
              << " records=" << scan.records.size() << '\n';
    for (const auto& record : scan.records) {
      std::cout << "scan: record="
                << (record.type == av2enc::hdr::MetadataType::kHdrCll
                        ? "cll"
                        : "mdcv");
      if (record.type == av2enc::hdr::MetadataType::kHdrCll) {
        std::cout << " max_cll=" << record.cll.max_cll
                  << " max_fall=" << record.cll.max_fall;
      } else {
        const auto& m = record.mdcv;
        std::cout << " Gx=" << m.primary_chromaticity_x[0]
                  << " Gy=" << m.primary_chromaticity_y[0]
                  << " Bx=" << m.primary_chromaticity_x[1]
                  << " By=" << m.primary_chromaticity_y[1]
                  << " Rx=" << m.primary_chromaticity_x[2]
                  << " Ry=" << m.primary_chromaticity_y[2]
                  << " Wx=" << m.white_point_chromaticity_x
                  << " Wy=" << m.white_point_chromaticity_y
                  << " Lmax=" << m.luminance_max
                  << " Lmin=" << m.luminance_min;
      }
      std::cout << " suffix=" << (record.is_suffix ? 1 : 0)
                << " group=" << (record.in_group ? 1 : 0)
                << " before_frame=" << (record.before_first_frame ? 1 : 0)
                << " obu_offset=" << record.obu_offset << '\n';
    }
  }
  std::cout << "scan: total_metadata_obus=" << total_obus
            << " total_metadata_units=" << total_units
            << " total_unknown_units=" << total_unknown
            << " total_invalid_units=" << total_invalid
            << " total_records=" << total_records << '\n';
  return total_invalid == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string output;
  bool scan_only = false;
  bool group = false;
  bool suffix = false;
  bool legacy = false;
  bool strict_ranges = false;
  bool no_cll = false;
  bool mdcv_none = false;
  std::size_t frame_index = 0;
  ContentLightLevel cll{1000, 400};
  MasteringDisplayColorVolume mdcv;
  std::string error;
  if (!preset_mdcv("bt2020-1000", mdcv, error)) {
    std::cerr << "append_hdr_metadata: " << error << '\n';
    return 1;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(std::cout);
      return 0;
    }
    if (arg == "--input" || arg == "-i") {
      if (i + 1 >= argc) {
        std::cerr << "append_hdr_metadata: missing value for --input\n";
        return 2;
      }
      input = argv[++i];
      continue;
    }
    if (arg == "--output" || arg == "-o") {
      if (i + 1 >= argc) {
        std::cerr << "append_hdr_metadata: missing value for --output\n";
        return 2;
      }
      output = argv[++i];
      continue;
    }
    if (arg == "--scan") {
      scan_only = true;
      continue;
    }
    if (arg == "--group") {
      group = true;
      continue;
    }
    if (arg == "--suffix") {
      suffix = true;
      continue;
    }
    if (arg == "--legacy-av1") {
      legacy = true;
      continue;
    }
    if (arg == "--strict-ranges") {
      strict_ranges = true;
      continue;
    }
    if (arg == "--no-cll") {
      no_cll = true;
      continue;
    }
    if (arg == "--frame") {
      if (i + 1 >= argc) {
        std::cerr << "append_hdr_metadata: missing value for --frame\n";
        return 2;
      }
      const auto value = parse_u32(argv[++i]);
      if (!value) {
        std::cerr << "append_hdr_metadata: invalid --frame\n";
        return 2;
      }
      frame_index = *value;
      continue;
    }
    if (arg == "--cll") {
      if (i + 1 >= argc) {
        std::cerr << "append_hdr_metadata: missing value for --cll\n";
        return 2;
      }
      const std::vector<std::string_view> parts = split(argv[++i], ':');
      if (parts.size() != 2U) {
        std::cerr << "append_hdr_metadata: --cll expects max_cll:max_fall\n";
        return 2;
      }
      const auto max_cll = parse_u32(parts[0]);
      const auto max_fall = parse_u32(parts[1]);
      if (!max_cll || !max_fall || *max_cll > 0xFFFFU ||
          *max_fall > 0xFFFFU) {
        std::cerr << "append_hdr_metadata: --cll values must fit in 16 bits\n";
        return 2;
      }
      cll.max_cll = static_cast<std::uint16_t>(*max_cll);
      cll.max_fall = static_cast<std::uint16_t>(*max_fall);
      continue;
    }
    if (arg == "--mdcv") {
      if (i + 1 >= argc) {
        std::cerr << "append_hdr_metadata: missing value for --mdcv\n";
        return 2;
      }
      const std::string_view value = argv[++i];
      if (value == "none") {
        mdcv_none = true;
        continue;
      }
      if (value.find(',') != std::string_view::npos) {
        if (!parse_mdcv_spec(value, mdcv, error)) {
          std::cerr << "append_hdr_metadata: " << error << '\n';
          return 2;
        }
      } else if (!preset_mdcv(value, mdcv, error)) {
        std::cerr << "append_hdr_metadata: " << error << '\n';
        return 2;
      }
      mdcv_none = false;
      continue;
    }
    std::cerr << "append_hdr_metadata: unknown argument " << arg << '\n';
    return 2;
  }

  if (input.empty()) {
    std::cerr << "append_hdr_metadata: --input is required\n";
    return 2;
  }
  if (!scan_only && output.empty()) {
    std::cerr << "append_hdr_metadata: --output is required unless --scan\n";
    return 2;
  }
  if (legacy && (group || suffix)) {
    std::cerr << "append_hdr_metadata: --legacy-av1 cannot be combined with "
                 "--group/--suffix\n";
    return 2;
  }

  std::vector<std::uint8_t> bytes;
  if (!read_file(input, bytes, error)) {
    std::cerr << "append_hdr_metadata: " << error << '\n';
    return 1;
  }
  IvfFile file;
  if (!parse_ivf(bytes, file, error)) {
    std::cerr << "append_hdr_metadata: " << error << '\n';
    return 1;
  }
  if (scan_only) {
    return run_scan(file);
  }
  if (frame_index >= file.frames.size()) {
    std::cerr << "append_hdr_metadata: --frame " << frame_index
              << " is out of range (stream has " << file.frames.size()
              << " frame(s))\n";
    return 1;
  }

  HdrMetadata hdr;
  hdr.has_cll = !no_cll;
  hdr.cll = cll;
  hdr.has_mdcv = !mdcv_none;
  hdr.mdcv = mdcv;

  MetadataObuConfig config;
  config.suffix = suffix;
  config.strict_ranges = strict_ranges;

  std::vector<std::uint8_t> metadata_obus;
  if (legacy) {
    metadata_obus =
        av2enc::hdr::make_legacy_av1_hdr_metadata_obus(hdr, error);
  } else {
    metadata_obus = av2enc::hdr::make_hdr_metadata_obus(
        hdr, group ? HdrObuPacking::kMetadataGroup : HdrObuPacking::kShortObus,
        config, error);
  }
  if (metadata_obus.empty()) {
    std::cerr << "append_hdr_metadata: " << error << '\n';
    return 1;
  }

  const MetadataPlacement placement =
      suffix ? MetadataPlacement::kEndOfTemporalUnit
             : MetadataPlacement::kBeforeFirstFrame;
  std::vector<std::uint8_t> new_payload;
  std::size_t inserted_at = 0;
  if (!av2enc::hdr::insert_metadata_obus(file.frames[frame_index].payload,
                                         metadata_obus, placement, new_payload,
                                         error, &inserted_at)) {
    std::cerr << "append_hdr_metadata: " << error << '\n';
    return 1;
  }

  const std::size_t old_size = file.frames[frame_index].payload.size();
  file.frames[frame_index].payload = std::move(new_payload);

  std::vector<std::uint8_t> serialized;
  if (!serialize_ivf(file, serialized, error)) {
    std::cerr << "append_hdr_metadata: " << error << '\n';
    return 1;
  }
  if (!write_file(output, serialized, error)) {
    std::cerr << "append_hdr_metadata: " << error << '\n';
    return 1;
  }

  std::cout << "append_hdr_metadata: frame " << frame_index
            << " payload " << old_size << " -> "
            << file.frames[frame_index].payload.size()
            << " bytes (+" << metadata_obus.size() << " metadata at "
            << inserted_at << ")\n";
  std::cout << "append_hdr_metadata: wrote " << output << " ("
            << serialized.size() << " bytes)\n";
  return 0;
}
