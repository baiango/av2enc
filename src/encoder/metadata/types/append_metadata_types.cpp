// append_metadata_types: insert AV2 v1.0.0 non-HDR metadata OBUs into an
// existing IVF stream, or scan a stream for every metadata unit type.
//
// This is the decoder-gate helper for src/encoder/metadata/types/. It does not
// encode. It inserts already-serialized OBUs at a spec-conformant position:
//
//   prefix (default): immediately before the first frame OBU, i.e. after the
//                     sequence header (§7.3.3 coded output frame unit)
//   suffix (--suffix): after the last OBU of the temporal unit (§7.3.3)
//
// Only the selected IVF frame payload changes; the file header, timestamps and
// all other frames are copied verbatim.

#include "encoder/metadata/types/metadata_types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using av2enc::hdr::MetadataApplicationId;
using av2enc::hdr::MetadataLayer;
using av2enc::hdr::MetadataNecessity;
using av2enc::hdr::MetadataObuConfig;
using av2enc::hdr::MetadataPersistence;
using av2enc::metadata::BandingHints;
using av2enc::metadata::DecodedFrameHash;
using av2enc::metadata::GroupUnit;
using av2enc::metadata::IccProfile;
using av2enc::metadata::ItutT35;
using av2enc::metadata::MetadataUnitPayload;
using av2enc::metadata::ScanResult;
using av2enc::metadata::ScanType;
using av2enc::metadata::TemporalPointInfo;
using av2enc::metadata::Timecode;
using av2enc::metadata::Type;
using av2enc::metadata::UserDataUnregistered;

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

std::string hex_of(std::span<const std::uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2U);
  for (const std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Deterministic sample values used by the decoder gate.
// ---------------------------------------------------------------------------

ItutT35 sample_itut_t35() {
  ItutT35 value;
  value.country_code = 0xB5;  // United States
  value.payload = {0x00, 0x3C, 0x00, 0x01, 0x04};
  return value;
}

Timecode sample_timecode() {
  Timecode value;
  value.counting_type = 0;
  value.full_timestamp_flag = true;
  value.n_frames = 42;
  value.seconds_value = 7;
  value.minutes_value = 3;
  value.hours_value = 1;
  value.seconds_flag = true;
  value.minutes_flag = true;
  value.hours_flag = true;
  return value;
}

BandingHints sample_banding_hints() {
  BandingHints value;
  value.coding_banding_present_flag = true;
  value.banding_hints_flag = true;
  value.components[0].banding_in_component_present_flag = true;
  value.components[0].max_band_width_minus_4 = 12;
  value.components[0].max_band_step_minus_1 = 3;
  value.band_units_information_present_flag = true;
  value.num_band_units_rows_minus_1 = 1;
  value.num_band_units_cols_minus_1 = 2;
  value.varying_size_band_units_flag = true;
  value.band_block_in_luma_samples = 1;
  value.vert_size_in_band_blocks_minus_1 = {0, 1};
  value.horz_size_in_band_blocks_minus_1 = {0, 1, 2};
  value.banding_in_band_unit_present = {1, 0, 1, 0, 1, 1};
  return value;
}

IccProfile sample_icc_profile() {
  IccProfile value;
  value.data = {0x00, 0x00, 0x01, 0x20, 0x41, 0x44, 0x42, 0x45,
                0x04, 0x30, 0x00, 0x00, 0x6D, 0x6E, 0x74, 0x72};
  return value;
}

ScanType sample_scan_type() {
  ScanType value;
  value.mps_pic_struct_type = 1;
  value.mps_source_scan_type_idc = 2;
  return value;
}

TemporalPointInfo sample_temporal_point_info() {
  TemporalPointInfo value;
  value.frame_presentation_time = 1000;
  return value;
}

DecodedFrameHash sample_decoded_frame_hash() {
  DecodedFrameHash value;
  value.per_plane = true;
  for (std::size_t plane = 0; plane < 3U; ++plane) {
    for (std::size_t i = 0; i < value.plane_hash[plane].size(); ++i) {
      value.plane_hash[plane][i] =
          static_cast<std::uint8_t>(plane * 0x10U + i);
    }
  }
  return value;
}

UserDataUnregistered sample_user_data() {
  UserDataUnregistered value;
  for (std::size_t i = 0; i < value.uuid.size(); ++i) {
    value.uuid[i] = static_cast<std::uint8_t>(i);
  }
  value.payload = {0xDE, 0xAD, 0xBE, 0xEF};
  return value;
}

std::vector<MetadataUnitPayload> make_sample_units(bool strict,
                                                   std::string& error) {
  std::vector<MetadataUnitPayload> units;
  const auto append = [&units](std::optional<MetadataUnitPayload> unit) {
    if (!unit) {
      return false;
    }
    units.push_back(std::move(*unit));
    return true;
  };
  if (!append(av2enc::metadata::make_itut_t35_unit(
          sample_itut_t35(), error, strict)) ||
      !append(av2enc::metadata::make_timecode_unit(
          sample_timecode(), error, strict)) ||
      !append(av2enc::metadata::make_decoded_frame_hash_unit(
          sample_decoded_frame_hash(), error, strict)) ||
      !append(av2enc::metadata::make_banding_hints_unit(
          sample_banding_hints(), error, strict)) ||
      !append(av2enc::metadata::make_icc_profile_unit(
          sample_icc_profile(), error, strict)) ||
      !append(av2enc::metadata::make_scan_type_unit(
          sample_scan_type(), error, strict)) ||
      !append(av2enc::metadata::make_temporal_point_info_unit(
          sample_temporal_point_info(), error, strict)) ||
      !append(av2enc::metadata::make_user_data_unregistered_unit(
          sample_user_data(), error, strict))) {
    return {};
  }
  return units;
}

std::optional<Type> parse_type_name(std::string_view name) {
  if (name == "itut_t35" || name == "t35") {
    return Type::kItutT35;
  }
  if (name == "timecode") {
    return Type::kTimecode;
  }
  if (name == "frame_hash" || name == "decoded_frame_hash" || name == "hash") {
    return Type::kDecodedFrameHash;
  }
  if (name == "banding" || name == "banding_hints") {
    return Type::kBandingHints;
  }
  if (name == "icc" || name == "icc_profile") {
    return Type::kIccProfile;
  }
  if (name == "scan_type" || name == "scan") {
    return Type::kScanType;
  }
  if (name == "temporal_point" || name == "temporal_point_info" || name == "tpi") {
    return Type::kTemporalPointInfo;
  }
  if (name == "user_data" || name == "user_data_unregistered" || name == "udu") {
    return Type::kUserDataUnregistered;
  }
  return std::nullopt;
}

std::optional<MetadataLayer> parse_layer(std::string_view name) {
  if (name == "unspecified") return MetadataLayer::kUnspecified;
  if (name == "global") return MetadataLayer::kGlobal;
  if (name == "current") return MetadataLayer::kCurrent;
  return std::nullopt;
}

std::optional<MetadataPersistence> parse_persistence(std::string_view name) {
  if (name == "global") return MetadataPersistence::kGlobal;
  if (name == "basic") return MetadataPersistence::kBasic;
  if (name == "none" || name == "no") {
    return MetadataPersistence::kNoPersistence;
  }
  if (name == "enhanced") return MetadataPersistence::kEnhanced;
  return std::nullopt;
}

std::optional<MetadataNecessity> parse_necessity(std::string_view name) {
  if (name == "undefined") return MetadataNecessity::kUndefined;
  if (name == "necessary") return MetadataNecessity::kNecessary;
  if (name == "advisory") return MetadataNecessity::kAdvisory;
  if (name == "mixed") return MetadataNecessity::kMixed;
  return std::nullopt;
}

void print_usage(std::ostream& out) {
  out <<
      "Usage:\n"
      "  append_metadata_types --input <in.ivf> --output <out.ivf> [options]\n"
      "  append_metadata_types --input <in.ivf> --scan [--values]\n"
      "\n"
      "Options:\n"
      "  --type <list>          Comma-separated types; default: all 8.\n"
      "                         itut_t35, timecode, frame_hash, banding,\n"
      "                         icc, scan_type, temporal_point, user_data\n"
      "  --group                One OBU_METADATA_GROUP instead of one\n"
      "                         OBU_METADATA_SHORT per unit\n"
      "  --suffix               Suffix metadata after the frame instead of\n"
      "                         prefix metadata before it\n"
      "  --cancel               Emit cancel units (no metadata payload)\n"
      "  --frame <n>            IVF frame to modify (default 0)\n"
      "  --layer <idc>          unspecified | global | current (default)\n"
      "  --persistence <idc>    global (default) | basic | none | enhanced\n"
      "  --necessity <idc>      undefined | necessary | advisory (default) |\n"
      "                         mixed (group only)\n"
      "  --application <n>      0..5, 16..31 (group only; default 0)\n"
      "  --priority <n>         0..255, group only (default 0)\n"
      "  --strict-ranges        Reject reserved-but-parseable metadata values\n"
      "  --scan                 Print metadata found in the input\n"
      "  --values               With --scan, print decoded unit values\n"
      "  -h, --help             Show this help\n";
}

std::string layer_name(MetadataLayer layer) {
  switch (layer) {
    case MetadataLayer::kUnspecified:
      return "unspecified";
    case MetadataLayer::kGlobal:
      return "global";
    case MetadataLayer::kCurrent:
      return "current";
    case MetadataLayer::kValues:
      return "values";
  }
  return "reserved";
}

std::string persistence_name(MetadataPersistence persistence) {
  switch (persistence) {
    case MetadataPersistence::kGlobal:
      return "global";
    case MetadataPersistence::kBasic:
      return "basic";
    case MetadataPersistence::kNoPersistence:
      return "none";
    case MetadataPersistence::kEnhanced:
      return "enhanced";
  }
  return "reserved";
}


void print_value(const av2enc::metadata::MetadataRecord& record) {
  const auto* t35 = std::get_if<ItutT35>(&record.value);
  if (t35 != nullptr) {
    std::cout << " country_code=" << static_cast<unsigned>(t35->country_code)
              << " extension_byte="
              << static_cast<unsigned>(t35->country_code_extension_byte)
              << " payload_size=" << t35->payload.size()
              << " payload=" << hex_of(t35->payload);
    return;
  }
  const auto* tc = std::get_if<Timecode>(&record.value);
  if (tc != nullptr) {
    std::cout << " counting_type=" << static_cast<unsigned>(tc->counting_type)
              << " full=" << (tc->full_timestamp_flag ? 1 : 0)
              << " discontinuity=" << (tc->discontinuity_flag ? 1 : 0)
              << " cnt_dropped=" << (tc->cnt_dropped_flag ? 1 : 0)
              << " n_frames=" << tc->n_frames
              << " seconds_flag=" << (tc->seconds_flag ? 1 : 0)
              << " seconds=" << static_cast<unsigned>(tc->seconds_value)
              << " minutes_flag=" << (tc->minutes_flag ? 1 : 0)
              << " minutes=" << static_cast<unsigned>(tc->minutes_value)
              << " hours_flag=" << (tc->hours_flag ? 1 : 0)
              << " hours=" << static_cast<unsigned>(tc->hours_value)
              << " offset_length="
              << static_cast<unsigned>(tc->time_offset_length)
              << " offset=" << tc->time_offset_value;
    return;
  }
  const auto* hash = std::get_if<DecodedFrameHash>(&record.value);
  if (hash != nullptr) {
    std::cout << " hash_type=" << static_cast<unsigned>(hash->hash_type)
              << " per_plane=" << (hash->per_plane ? 1 : 0)
              << " has_grain=" << (hash->has_grain ? 1 : 0)
              << " is_monochrome=" << (hash->is_monochrome ? 1 : 0)
              << " hash=" << hex_of(hash->per_plane ? hash->plane_hash[0]
                                                    : hash->frame_hash);
    return;
  }
  const auto* banding = std::get_if<BandingHints>(&record.value);
  if (banding != nullptr) {
    std::cout << " coding=" << (banding->coding_banding_present_flag ? 1 : 0)
              << " source=" << (banding->source_banding_present_flag ? 1 : 0)
              << " hints=" << (banding->banding_hints_flag ? 1 : 0)
              << " three_color="
              << (banding->three_color_components_flag ? 1 : 0)
              << " band_units="
              << (banding->band_units_information_present_flag ? 1 : 0)
              << " rows="
              << static_cast<unsigned>(banding->num_band_units_rows_minus_1) +
                     1U
              << " cols="
              << static_cast<unsigned>(banding->num_band_units_cols_minus_1) +
                     1U
              << " varying="
              << (banding->varying_size_band_units_flag ? 1 : 0)
              << " band_block="
              << static_cast<unsigned>(banding->band_block_in_luma_samples)
              << " units=" << banding->banding_in_band_unit_present.size();
    return;
  }
  const auto* icc = std::get_if<IccProfile>(&record.value);
  if (icc != nullptr) {
    const auto version = av2enc::metadata::icc_profile_version(*icc);
    std::cout << " size=" << icc->data.size();
    if (version) {
      std::cout << " icc_major=" << static_cast<unsigned>(version->major)
                << " icc_minor=" << static_cast<unsigned>(version->minor);
    } else {
      std::cout << " icc_version=unavailable";
    }
    return;
  }
  const auto* scan = std::get_if<ScanType>(&record.value);
  if (scan != nullptr) {
    std::cout << " pic_struct_type="
              << static_cast<unsigned>(scan->mps_pic_struct_type)
              << " source_scan_type_idc="
              << static_cast<unsigned>(scan->mps_source_scan_type_idc)
              << " duplicate=" << (scan->mps_duplicate_flag ? 1 : 0);
    return;
  }
  const auto* temporal = std::get_if<TemporalPointInfo>(&record.value);
  if (temporal != nullptr) {
    std::cout << " frame_presentation_time="
              << temporal->frame_presentation_time;
    return;
  }
  const auto* user = std::get_if<UserDataUnregistered>(&record.value);
  if (user != nullptr) {
    std::cout << " uuid=" << hex_of(user->uuid)
              << " payload_size=" << user->payload.size()
              << " payload=" << hex_of(user->payload);
    return;
  }
  const auto* cll = std::get_if<av2enc::hdr::ContentLightLevel>(&record.value);
  if (cll != nullptr) {
    std::cout << " max_cll=" << cll->max_cll << " max_fall=" << cll->max_fall;
    return;
  }
  const auto* mdcv =
      std::get_if<av2enc::hdr::MasteringDisplayColorVolume>(&record.value);
  if (mdcv != nullptr) {
    std::cout << " Gx=" << mdcv->primary_chromaticity_x[0]
              << " Gy=" << mdcv->primary_chromaticity_y[0]
              << " Bx=" << mdcv->primary_chromaticity_x[1]
              << " By=" << mdcv->primary_chromaticity_y[1]
              << " Rx=" << mdcv->primary_chromaticity_x[2]
              << " Ry=" << mdcv->primary_chromaticity_y[2]
              << " Wx=" << mdcv->white_point_chromaticity_x
              << " Wy=" << mdcv->white_point_chromaticity_y
              << " Lmax=" << mdcv->luminance_max
              << " Lmin=" << mdcv->luminance_min;
  }
}

int run_scan(const IvfFile& file, bool print_values) {
  std::size_t total_obus = 0;
  std::size_t total_units = 0;
  std::size_t total_supported = 0;
  std::size_t total_unsupported = 0;
  std::size_t total_invalid = 0;
  std::size_t total_cancelled = 0;
  std::size_t total_records = 0;

  std::cout << "scan: frames=" << file.frames.size() << '\n';
  for (std::size_t frame_index = 0; frame_index < file.frames.size();
       ++frame_index) {
    ScanResult scan;
    std::string error;
    if (!av2enc::metadata::scan_metadata(file.frames[frame_index].payload,
                                         scan, error)) {
      std::cerr << "scan: frame " << frame_index
                << ": malformed OBU stream: " << error << '\n';
      return 1;
    }
    total_obus += scan.metadata_obus;
    total_units += scan.metadata_units;
    total_supported += scan.supported_units;
    total_unsupported += scan.unsupported_units;
    total_invalid += scan.invalid_units;
    total_cancelled += scan.cancelled_units;
    total_records += scan.records.size();

    std::cout << "scan: frame=" << frame_index
              << " payload=" << file.frames[frame_index].payload.size()
              << " metadata_obus=" << scan.metadata_obus
              << " metadata_units=" << scan.metadata_units
              << " supported_units=" << scan.supported_units
              << " unsupported_units=" << scan.unsupported_units
              << " invalid_units=" << scan.invalid_units
              << " cancelled_units=" << scan.cancelled_units
              << " records=" << scan.records.size() << '\n';
    for (const auto& record : scan.records) {
      std::cout << "scan: record="
                << av2enc::metadata::type_name(record.metadata_type)
                << " suffix=" << (record.is_suffix ? 1 : 0)
                << " group=" << (record.in_group ? 1 : 0)
                << " before_frame=" << (record.before_first_frame ? 1 : 0)
                << " cancel=" << (record.cancel ? 1 : 0)
                << " layer=" << layer_name(record.layer)
                << " persistence=" << persistence_name(record.persistence)
                << " priority=" << static_cast<unsigned>(record.priority)
                << " obu_offset=" << record.obu_offset;
      if (print_values && record.has_value) {
        print_value(record);
      }
      std::cout << '\n';
    }
  }
  std::cout << "scan: total_metadata_obus=" << total_obus
            << " total_metadata_units=" << total_units
            << " total_supported_units=" << total_supported
            << " total_unsupported_units=" << total_unsupported
            << " total_invalid_units=" << total_invalid
            << " total_cancelled_units=" << total_cancelled
            << " total_records=" << total_records << '\n';
  return total_invalid == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string output;
  bool scan_only = false;
  bool print_values = false;
  bool group = false;
  bool suffix = false;
  bool cancel = false;
  bool strict = false;
  std::size_t frame_index = 0;
  MetadataLayer layer = MetadataLayer::kCurrent;
  MetadataPersistence persistence = MetadataPersistence::kGlobal;
  MetadataNecessity necessity = MetadataNecessity::kAdvisory;
  std::uint8_t application = 0;
  std::uint8_t priority = 0;
  std::vector<Type> selected;
  std::string error;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(std::cout);
      return 0;
    }
    if (arg == "--input" || arg == "-i") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --input\n";
        return 2;
      }
      input = argv[++i];
      continue;
    }
    if (arg == "--output" || arg == "-o") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --output\n";
        return 2;
      }
      output = argv[++i];
      continue;
    }
    if (arg == "--type") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --type\n";
        return 2;
      }
      for (const std::string_view name : split(argv[++i], ',')) {
        if (name == "all") {
          continue;  // handled by the default below
        }
        const auto type = parse_type_name(name);
        if (!type) {
          std::cerr << "append_metadata_types: unknown type '" << name
                    << "'\n";
          return 2;
        }
        selected.push_back(*type);
      }
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
    if (arg == "--cancel") {
      cancel = true;
      continue;
    }
    if (arg == "--strict-ranges") {
      strict = true;
      continue;
    }
    if (arg == "--scan") {
      scan_only = true;
      continue;
    }
    if (arg == "--values") {
      print_values = true;
      continue;
    }
    if (arg == "--frame") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --frame\n";
        return 2;
      }
      const auto value = parse_u32(argv[++i]);
      if (!value) {
        std::cerr << "append_metadata_types: invalid --frame\n";
        return 2;
      }
      frame_index = *value;
      continue;
    }
    if (arg == "--layer") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --layer\n";
        return 2;
      }
      const auto value = parse_layer(argv[++i]);
      if (!value) {
        std::cerr << "append_metadata_types: invalid --layer\n";
        return 2;
      }
      layer = *value;
      continue;
    }
    if (arg == "--persistence") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --persistence\n";
        return 2;
      }
      const auto value = parse_persistence(argv[++i]);
      if (!value) {
        std::cerr << "append_metadata_types: invalid --persistence\n";
        return 2;
      }
      persistence = *value;
      continue;
    }
    if (arg == "--necessity") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --necessity\n";
        return 2;
      }
      const auto value = parse_necessity(argv[++i]);
      if (!value) {
        std::cerr << "append_metadata_types: invalid --necessity\n";
        return 2;
      }
      necessity = *value;
      continue;
    }
    if (arg == "--application") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --application\n";
        return 2;
      }
      const auto value = parse_u32(argv[++i]);
      if (!value || *value > 31U) {
        std::cerr << "append_metadata_types: invalid --application\n";
        return 2;
      }
      application = static_cast<std::uint8_t>(*value);
      continue;
    }
    if (arg == "--priority") {
      if (i + 1 >= argc) {
        std::cerr << "append_metadata_types: missing value for --priority\n";
        return 2;
      }
      const auto value = parse_u32(argv[++i]);
      if (!value || *value > 255U) {
        std::cerr << "append_metadata_types: invalid --priority\n";
        return 2;
      }
      priority = static_cast<std::uint8_t>(*value);
      continue;
    }
    std::cerr << "append_metadata_types: unknown argument " << arg << '\n';
    return 2;
  }

  if (input.empty()) {
    std::cerr << "append_metadata_types: --input is required\n";
    return 2;
  }
  if (!scan_only && output.empty()) {
    std::cerr << "append_metadata_types: --output is required unless --scan\n";
    return 2;
  }

  std::vector<std::uint8_t> bytes;
  if (!read_file(input, bytes, error)) {
    std::cerr << "append_metadata_types: " << error << '\n';
    return 1;
  }
  IvfFile file;
  if (!parse_ivf(bytes, file, error)) {
    std::cerr << "append_metadata_types: " << error << '\n';
    return 1;
  }
  if (scan_only) {
    return run_scan(file, print_values);
  }
  if (frame_index >= file.frames.size()) {
    std::cerr << "append_metadata_types: --frame " << frame_index
              << " is out of range (stream has " << file.frames.size()
              << " frame(s))\n";
    return 1;
  }

  const std::vector<MetadataUnitPayload> samples = make_sample_units(strict,
                                                                     error);
  if (samples.empty()) {
    std::cerr << "append_metadata_types: " << error << '\n';
    return 1;
  }

  std::vector<const MetadataUnitPayload*> chosen;
  if (selected.empty()) {
    for (const auto& unit : samples) {
      chosen.push_back(&unit);
    }
  } else {
    for (const Type type : selected) {
      const auto match = std::find_if(
          samples.begin(), samples.end(),
          [type](const MetadataUnitPayload& unit) { return unit.type == type; });
      if (match == samples.end()) {
        std::cerr << "append_metadata_types: no sample value for type "
                  << av2enc::metadata::type_name(type) << '\n';
        return 1;
      }
      chosen.push_back(&*match);
    }
  }

  MetadataObuConfig config;
  config.suffix = suffix;
  config.layer = layer;
  config.persistence = persistence;
  config.necessity = necessity;
  config.application = static_cast<MetadataApplicationId>(application);
  config.strict_ranges = strict;

  std::vector<std::uint8_t> metadata_obus;
  if (group) {
    std::vector<GroupUnit> entries;
    entries.reserve(chosen.size());
    for (const MetadataUnitPayload* unit : chosen) {
      GroupUnit entry;
      entry.payload.type = unit->type;
      if (cancel) {
        entry.cancel = true;
      } else {
        entry.payload.bytes = unit->bytes;
      }
      entry.layer = layer;
      entry.persistence = persistence;
      entry.priority = priority;
      entries.push_back(std::move(entry));
    }
    metadata_obus = av2enc::metadata::make_group_obu(entries, config, error);
  } else {
    for (const MetadataUnitPayload* unit : chosen) {
      std::vector<std::uint8_t> obu;
      if (cancel) {
        obu = av2enc::metadata::make_short_cancel_obu(unit->type, config,
                                                      error);
      } else {
        obu = av2enc::metadata::make_short_obu(*unit, config, error);
      }
      if (obu.empty()) {
        std::cerr << "append_metadata_types: "
                  << av2enc::metadata::type_name(unit->type) << ": " << error
                  << '\n';
        return 1;
      }
      metadata_obus.insert(metadata_obus.end(), obu.begin(), obu.end());
    }
  }
  if (metadata_obus.empty()) {
    std::cerr << "append_metadata_types: " << error << '\n';
    return 1;
  }

  const av2enc::hdr::MetadataPlacement placement =
      suffix ? av2enc::hdr::MetadataPlacement::kEndOfTemporalUnit
             : av2enc::hdr::MetadataPlacement::kBeforeFirstFrame;
  std::vector<std::uint8_t> new_payload;
  std::size_t inserted_at = 0;
  if (!av2enc::hdr::insert_metadata_obus(file.frames[frame_index].payload,
                                         metadata_obus, placement, new_payload,
                                         error, &inserted_at)) {
    std::cerr << "append_metadata_types: " << error << '\n';
    return 1;
  }

  const std::size_t old_size = file.frames[frame_index].payload.size();
  file.frames[frame_index].payload = std::move(new_payload);

  std::vector<std::uint8_t> serialized;
  if (!serialize_ivf(file, serialized, error)) {
    std::cerr << "append_metadata_types: " << error << '\n';
    return 1;
  }
  if (!write_file(output, serialized, error)) {
    std::cerr << "append_metadata_types: " << error << '\n';
    return 1;
  }

  std::cout << "append_metadata_types: frame " << frame_index << " payload "
            << old_size << " -> " << file.frames[frame_index].payload.size()
            << " bytes (+" << metadata_obus.size() << " metadata at "
            << inserted_at << ")\n";
  std::cout << "append_metadata_types: wrote " << output << " ("
            << serialized.size() << " bytes)\n";
  return 0;
}
