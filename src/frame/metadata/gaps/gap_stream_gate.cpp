// Decoder-gate helper for src/encoder/metadata/gaps/.
//
// Inserts cancel / LAYER_VALUES / header-extension metadata OBUs into a valid
// M2 IVF stream using the conformance placement helper, writes one variant per
// case, and self-parses every variant with the module's own reader. The shell
// wrapper then decodes each variant with pinned dav2d and requires the decoded
// Y4M to be byte-identical to the unmodified stream.
//
// Usage: gap_stream_gate --base m2.ivf --out-dir DIR
//
// Exit: 0 all self-parse checks pass, 1 failure, 2 usage.

#include "frame/metadata/gaps/metadata_gaps.h"

#include "frame/metadata/hdr/hdr_metadata.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace av2enc;
using namespace av2enc::metadata;
using namespace av2enc::metadata::gaps;

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

int g_failures = 0;

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!(expression)) {                                                      \
      ++g_failures;                                                           \
      std::cerr << __FILE__ << ':' << __LINE__                                \
                << ": CHECK failed: " #expression "\n";                       \
    }                                                                         \
  } while (false)

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

std::vector<std::uint8_t> cll_unit() {
  hdr::ContentLightLevel cll;
  cll.max_cll = 1000;
  cll.max_fall = 400;
  return hdr::serialize_cll_unit(cll);
}

MetadataGapUnit cll_unit_with(const hdr::MetadataLayer layer,
                              const hdr::MetadataPersistence persistence,
                              const std::uint8_t priority,
                              const std::vector<std::uint8_t>& mlayer_map,
                              const std::vector<std::uint8_t>& extension) {
  MetadataGapUnit unit;
  unit.header.type = Type::kHdrCll;
  unit.header.layer = layer;
  unit.header.persistence = persistence;
  unit.header.priority = priority;
  unit.header.layer_values.mlayer_map = mlayer_map;
  unit.header.header_extension = extension;
  unit.payload = cll_unit();
  return unit;
}

MetadataGapUnit cancel_unit(const Type type,
                            const std::vector<std::uint8_t>& extension) {
  MetadataGapUnit unit;
  unit.header.type = type;
  unit.header.cancel = true;
  unit.header.header_extension = extension;
  return unit;
}

bool scan_frame(const IvfFile& file, std::size_t frame_index,
                GapScanResult& scan, std::string& error) {
  return scan_gap_metadata(file.frames[frame_index].payload, scan, error);
}

void report(const char* name, const GapScanResult& scan) {
  std::cout << "  " << name << ": metadata_obus=" << scan.metadata_obus
            << " metadata_units=" << scan.metadata_units
            << " supported=" << scan.supported_units
            << " cancelled=" << scan.cancelled_units
            << " invalid=" << scan.invalid_units << "\n";
}

bool write_variant(const IvfFile& base, std::size_t frame_index,
                   hdr::MetadataPlacement placement,
                   std::span<const std::uint8_t> obus,
                   const std::filesystem::path& path, IvfFile& variant,
                   std::string& error) {
  variant = base;
  std::vector<std::uint8_t> modified;
  if (!hdr::insert_metadata_obus(variant.frames[frame_index].payload, obus,
                                 placement, modified, error)) {
    return false;
  }
  variant.frames[frame_index].payload = std::move(modified);
  std::vector<std::uint8_t> bytes;
  if (!serialize_ivf(variant, bytes, error)) {
    return false;
  }
  return write_file(path, bytes, error);
}

void check_cancel_prefix(const IvfFile& base,
                         const std::filesystem::path& out_dir) {
  std::string error;
  GapObuConfig prefix;

  MetadataGapUnit short_cancel;
  short_cancel.header.type = Type::kTimecode;
  short_cancel.header.cancel = true;
  short_cancel.header.layer = hdr::MetadataLayer::kCurrent;
  const auto short_obu = make_short_obu(short_cancel, prefix, error);
  if (short_obu.empty()) {
    CHECK(false);
    return;
  }

  GapObuConfig group_config;
  group_config.necessity = hdr::MetadataNecessity::kAdvisory;
  const MetadataGapUnit group_cancel =
      cancel_unit(Type::kDecodedFrameHash, {0xAB});
  const auto group_obu =
      make_group_obu(std::span<const MetadataGapUnit>(&group_cancel, 1),
                     group_config, error);
  if (group_obu.empty()) {
    CHECK(false);
    return;
  }

  std::vector<std::uint8_t> obus = short_obu;
  obus.insert(obus.end(), group_obu.begin(), group_obu.end());

  IvfFile variant;
  const auto path = out_dir / "cancel_prefix.ivf";
  if (!write_variant(base, 0, hdr::MetadataPlacement::kBeforeFirstFrame, obus,
                     path, variant, error)) {
    std::cerr << "cancel_prefix: " << error << "\n";
    CHECK(false);
    return;
  }
  GapScanResult scan;
  if (!scan_frame(variant, 0, scan, error)) {
    std::cerr << "cancel_prefix scan: " << error << "\n";
    CHECK(false);
    return;
  }
  report("cancel_prefix", scan);
  CHECK(scan.metadata_obus == 2U);
  CHECK(scan.metadata_units == 2U);
  CHECK(scan.cancelled_units == 2U);
  CHECK(scan.supported_units == 0U);
  CHECK(scan.invalid_units == 0U);
  if (scan.records.size() == 2U) {
    CHECK(!scan.records[0].in_group);
    CHECK(scan.records[0].cancel);
    CHECK(scan.records[0].type == static_cast<std::uint32_t>(Type::kTimecode));
    CHECK(scan.records[0].before_first_frame);
    CHECK(scan.records[1].in_group);
    CHECK(scan.records[1].cancel);
    CHECK(scan.records[1].header_extension ==
          std::vector<std::uint8_t>{0xAB});
    CHECK(scan.records[1].before_first_frame);
  }
  std::cout << "VARIANT " << path.string() << "\n";
}

void check_layer_local(const IvfFile& base,
                       const std::filesystem::path& out_dir) {
  std::string error;
  GapObuConfig config;
  config.necessity = hdr::MetadataNecessity::kNecessary;
  config.application = hdr::MetadataApplicationId::kTv;
  config.xlayer_id = 0;
  config.mlayer_id = 0;

  const MetadataGapUnit unit =
      cll_unit_with(hdr::MetadataLayer::kValues,
                    hdr::MetadataPersistence::kBasic, 0x33, {0x02}, {});
  const auto obu =
      make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                     error);
  if (obu.empty()) {
    std::cerr << "layer_local: " << error << "\n";
    CHECK(false);
    return;
  }

  IvfFile variant;
  const auto path = out_dir / "layer_local_prefix.ivf";
  if (!write_variant(base, 0, hdr::MetadataPlacement::kBeforeFirstFrame, obu,
                     path, variant, error)) {
    std::cerr << "layer_local: " << error << "\n";
    CHECK(false);
    return;
  }
  GapScanResult scan;
  if (!scan_frame(variant, 0, scan, error)) {
    std::cerr << "layer_local scan: " << error << "\n";
    CHECK(false);
    return;
  }
  report("layer_local", scan);
  CHECK(scan.metadata_obus == 1U);
  CHECK(scan.metadata_units == 1U);
  CHECK(scan.supported_units == 1U);
  CHECK(scan.invalid_units == 0U);
  if (scan.records.size() == 1U) {
    CHECK(scan.records[0].in_group);
    CHECK(!scan.records[0].cancel);
    CHECK(scan.records[0].layer == hdr::MetadataLayer::kValues);
    CHECK(scan.records[0].mlayer_map == std::vector<std::uint8_t>{0x02});
    CHECK(scan.records[0].xlayer_map == 0U);
    CHECK(scan.records[0].priority == 0x33U);
    CHECK(scan.records[0].has_value);
  }
  std::cout << "VARIANT " << path.string() << "\n";
}

void check_layer_global(const IvfFile& base,
                        const std::filesystem::path& out_dir) {
  std::string error;
  GapObuConfig config;
  config.necessity = hdr::MetadataNecessity::kNecessary;
  config.application = hdr::MetadataApplicationId::kTv;
  config.xlayer_id = kGlobalXlayerId;
  config.mlayer_id = 0;

  MetadataGapUnit unit =
      cll_unit_with(hdr::MetadataLayer::kValues,
                    hdr::MetadataPersistence::kBasic, 0x21, {0x01, 0x04}, {});
  unit.header.layer_values.xlayer_map = 0x00000005U;

  const auto obu =
      make_group_obu(std::span<const MetadataGapUnit>(&unit, 1), config,
                     error);
  if (obu.empty()) {
    std::cerr << "layer_global: " << error << "\n";
    CHECK(false);
    return;
  }

  IvfFile variant;
  const auto path = out_dir / "layer_global_prefix.ivf";
  if (!write_variant(base, 0, hdr::MetadataPlacement::kBeforeFirstFrame, obu,
                     path, variant, error)) {
    std::cerr << "layer_global: " << error << "\n";
    CHECK(false);
    return;
  }
  GapScanResult scan;
  if (!scan_frame(variant, 0, scan, error)) {
    std::cerr << "layer_global scan: " << error << "\n";
    CHECK(false);
    return;
  }
  report("layer_global", scan);
  CHECK(scan.metadata_obus == 1U);
  CHECK(scan.metadata_units == 1U);
  CHECK(scan.supported_units == 1U);
  if (scan.records.size() == 1U) {
    CHECK(scan.records[0].layer == hdr::MetadataLayer::kValues);
    CHECK(scan.records[0].xlayer_map == 0x00000005U);
    CHECK(scan.records[0].mlayer_map ==
          (std::vector<std::uint8_t>{0x01, 0x04}));
  }
  std::cout << "VARIANT " << path.string() << "\n";
}

void check_suffix_mixed(const IvfFile& base,
                        const std::filesystem::path& out_dir) {
  if (base.frames.size() < 2U) {
    std::cerr << "suffix_mixed: baseline needs at least two frames\n";
    CHECK(false);
    return;
  }
  std::string error;
  GapObuConfig config;
  config.suffix = true;
  config.necessity = hdr::MetadataNecessity::kMixed;
  config.application = hdr::MetadataApplicationId::kUnspecified;
  config.xlayer_id = 0;

  const MetadataGapUnit cancel = cancel_unit(Type::kScanType, {0x55});
  const MetadataGapUnit cll =
      cll_unit_with(hdr::MetadataLayer::kValues,
                    hdr::MetadataPersistence::kNoPersistence, 0x66, {0x02},
                    {0x99});
  const std::array<MetadataGapUnit, 2> units = {cancel, cll};
  const auto obu = make_group_obu(units, config, error);
  if (obu.empty()) {
    std::cerr << "suffix_mixed: " << error << "\n";
    CHECK(false);
    return;
  }

  IvfFile variant;
  const auto path = out_dir / "layer_values_suffix.ivf";
  if (!write_variant(base, 1, hdr::MetadataPlacement::kEndOfTemporalUnit, obu,
                     path, variant, error)) {
    std::cerr << "suffix_mixed: " << error << "\n";
    CHECK(false);
    return;
  }
  GapScanResult scan;
  if (!scan_frame(variant, 1, scan, error)) {
    std::cerr << "suffix_mixed scan: " << error << "\n";
    CHECK(false);
    return;
  }
  report("suffix_mixed", scan);
  CHECK(scan.metadata_obus == 1U);
  CHECK(scan.metadata_units == 2U);
  CHECK(scan.cancelled_units == 1U);
  CHECK(scan.supported_units == 1U);
  if (scan.records.size() == 2U) {
    CHECK(scan.records[0].cancel);
    CHECK(scan.records[0].header_extension ==
          std::vector<std::uint8_t>{0x55});
    CHECK(scan.records[1].layer == hdr::MetadataLayer::kValues);
    CHECK(scan.records[1].mlayer_map == std::vector<std::uint8_t>{0x02});
    CHECK(scan.records[1].header_extension ==
          std::vector<std::uint8_t>{0x99});
    CHECK(scan.records[1].is_suffix);
    CHECK(!scan.records[1].before_first_frame);
  }
  std::cout << "VARIANT " << path.string() << "\n";
}

// Flip the trailing_bits byte of the last inserted OBU and require the reader
// to reject the frame.
void check_corrupt_rejected(const std::filesystem::path& source,
                            const std::filesystem::path& out_dir) {
  std::string error;
  std::vector<std::uint8_t> bytes;
  if (!read_file(source, bytes, error)) {
    std::cerr << "corrupt: " << error << "\n";
    CHECK(false);
    return;
  }
  IvfFile file;
  if (!parse_ivf(bytes, file, error)) {
    std::cerr << "corrupt: " << error << "\n";
    CHECK(false);
    return;
  }
  if (file.frames.empty()) {
    CHECK(false);
    return;
  }
  // The inserted metadata OBUs are the first OBUs after the sequence header;
  // find the last OBU whose header type is 9 and corrupt its final byte.
  std::vector<std::uint8_t>& payload = file.frames[0].payload;
  std::size_t pos = 0;
  bool corrupted = false;
  while (pos < payload.size()) {
    const std::size_t start = pos;
    std::uint32_t size = 0;
    unsigned shift = 0;
    do {
      if (pos >= payload.size()) break;
      const std::uint8_t byte = payload[pos++];
      size |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
      shift += 7U;
    } while ((payload[pos - 1U] & 0x80U) != 0U);
    if (pos > payload.size() || size == 0U) break;
    const std::uint8_t header = payload[pos];
    const std::uint8_t type = static_cast<std::uint8_t>((header >> 2U) & 0x1FU);
    const std::size_t end = pos + size;
    if (end > payload.size()) break;
    if (type == kObuTypeMetadataGroup) {
      payload[end - 1U] ^= 0x80U;  // trailing_bits must be 0x80
      corrupted = true;
      break;
    }
    if (end <= start) break;
    pos = end;
  }
  if (!corrupted) {
    std::cerr << "corrupt: no metadata group OBU found\n";
    CHECK(false);
    return;
  }
  const auto path = out_dir / "corrupt_trailing.ivf";
  std::vector<std::uint8_t> out;
  if (!serialize_ivf(file, out, error) || !write_file(path, out, error)) {
    std::cerr << "corrupt: " << error << "\n";
    CHECK(false);
    return;
  }
  GapScanResult scan;
  if (scan_gap_metadata(payload, scan, error)) {
    std::cerr << "corrupt: reader accepted invalid trailing_bits\n";
    CHECK(false);
    return;
  }
  std::cout << "  corrupt_trailing rejected: " << error << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path base_path;
  std::filesystem::path out_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--base" && i + 1 < argc) {
      base_path = argv[++i];
    } else if (arg == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: gap_stream_gate --base m2.ivf --out-dir DIR\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (base_path.empty() || out_dir.empty()) {
    std::cerr << "usage: gap_stream_gate --base m2.ivf --out-dir DIR\n";
    return 2;
  }
  std::error_code create_error;
  std::filesystem::create_directories(out_dir, create_error);
  if (create_error) {
    std::cerr << "cannot create output directory: " << create_error.message()
              << "\n";
    return 2;
  }

  std::string error;
  std::vector<std::uint8_t> bytes;
  if (!read_file(base_path, bytes, error)) {
    std::cerr << error << "\n";
    return 2;
  }
  IvfFile base;
  if (!parse_ivf(bytes, base, error)) {
    std::cerr << error << "\n";
    return 2;
  }
  if (base.frames.size() < 2U) {
    std::cerr << "baseline IVF needs at least two frames\n";
    return 2;
  }

  check_cancel_prefix(base, out_dir);
  check_layer_local(base, out_dir);
  check_layer_global(base, out_dir);
  check_suffix_mixed(base, out_dir);
  check_corrupt_rejected(out_dir / "layer_local_prefix.ivf", out_dir);

  if (g_failures != 0) {
    std::cerr << "gap_stream_gate: " << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "gap_stream_gate: all self-parse checks passed\n";
  return 0;
}
