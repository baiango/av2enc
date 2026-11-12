// dav2d gate for src/encoder/syntax/small_header/.
//
// The gate takes a *real* M1 still-picture IVF produced by the existing encoder
// (`av2enc --still --no-deblock`), extracts the entropy-coded tile payload from
// the OBU_CLOSED_LOOP_KEY, and re-emits the temporal unit with the small
// sequence-header features switched on. Only headers are rewritten; the tile
// payload is byte-for-byte the one produced by the M1 core.
//
// Before emitting anything the gate proves its header writers are exact: with
// every feature disabled it reconstructs the TD/SEQ/CLK OBUs and byte-compares
// them with the input. That is what makes the feature streams meaningful - the
// only difference from a known-good M1 stream is the feature under test.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).
//
// Usage:
//   small_header_gate --baseline FILE.ivf --out-dir DIR --prefix NAME
//                     --width W --height H --qp Q --deblock 0|1
//                     [--single-block]

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "frame/small_header/small_header.h"

namespace {

using av2enc::BitWriter;
using namespace av2enc::syntax::small_header;

int g_failures = 0;

#define FAIL(...)                              \
  do {                                         \
    std::fprintf(stderr, "gate: " __VA_ARGS__); \
    std::fprintf(stderr, "\n");                 \
    ++g_failures;                               \
  } while (0)

struct StillTemplate {
  int width = 0;
  int height = 0;
  int qp = 90;
  int seq_level_idx = 7;
  bool deblock = false;
  bool cfl = true;
  bool single_block = false;
};

struct Features {
  CroppingWindow crop;
  PbAspectRatio aspect;
  DfPar dfpar;
  SequenceSegmentConfig segment;
  SequenceTileConfig tile;
};

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                   std::istreambuf_iterator<char>());
}

bool write_file(const std::string& path,
                const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

void append_le32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFU));
  }
}

std::uint32_t read_le32(const std::uint8_t* data) {
  return static_cast<std::uint32_t>(data[0]) |
         (static_cast<std::uint32_t>(data[1]) << 8U) |
         (static_cast<std::uint32_t>(data[2]) << 16U) |
         (static_cast<std::uint32_t>(data[3]) << 24U);
}

struct Obu {
  int type = 0;
  std::vector<std::uint8_t> payload;
};

std::uint32_t read_leb128(std::span<const std::uint8_t> data,
                          std::size_t* offset, bool* ok) {
  std::uint32_t value = 0;
  unsigned shift = 0;
  for (;;) {
    if (*offset >= data.size() || shift > 28U) {
      *ok = false;
      return 0;
    }
    const std::uint8_t byte = data[(*offset)++];
    value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    shift += 7;
    if ((byte & 0x80U) == 0U) break;
  }
  *ok = true;
  return value;
}

bool parse_obus(std::span<const std::uint8_t> frame, std::vector<Obu>* obus,
                std::string* error) {
  std::size_t offset = 0;
  while (offset < frame.size()) {
    bool ok = false;
    const std::uint32_t size = read_leb128(frame, &offset, &ok);
    if (!ok || size < 1U || offset + size > frame.size()) {
      *error = "truncated OBU length";
      return false;
    }
    const std::uint8_t header = frame[offset];
    Obu obu;
    obu.type = (header >> 2) & 0x1F;
    const bool extension = (header & 0x80U) != 0U;
    std::size_t payload_offset = offset + 1U;
    std::size_t payload_size = size - 1U;
    if (extension) {
      if (payload_size < 1U) {
        *error = "truncated OBU extension";
        return false;
      }
      ++payload_offset;
      --payload_size;
    }
    obu.payload.assign(frame.begin() + static_cast<std::ptrdiff_t>(payload_offset),
                       frame.begin() + static_cast<std::ptrdiff_t>(payload_offset +
                                                                   payload_size));
    obus->push_back(std::move(obu));
    offset += size;
  }
  return true;
}

std::vector<std::uint8_t> wrap_obu(int type,
                                   std::span<const std::uint8_t> payload) {
  const std::size_t size = payload.size() + 1U;
  std::vector<std::uint8_t> out;
  std::uint32_t value = static_cast<std::uint32_t>(size);
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7FU);
    value >>= 7U;
    if (value != 0U) byte |= 0x80U;
    out.push_back(byte);
  } while (value != 0U);
  out.push_back(static_cast<std::uint8_t>((static_cast<unsigned>(type) << 2U) & 0x7CU));
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  writer.put_bits(static_cast<std::uint64_t>(value) -
                      ((1ULL << leading_zeros) - 1ULL),
                  leading_zeros);
}

void put_ns(BitWriter& writer, unsigned n, std::uint32_t value) {
  if (n <= 1U) return;
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0U) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

std::vector<std::uint8_t> take_aligned(BitWriter& writer) {
  writer.align_to_byte();
  const std::span<const std::uint8_t> bytes = writer.bytes();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

// M1 sequence_header_obu() syntax from src/encoder/m1/still_headers.cpp, with the
// five feature sites routed through the small_header writers.
bool write_still_sequence_header(BitWriter& writer, const StillTemplate& config,
                              const Features& features, std::string* error) {
  put_uvlc(writer, 0);                                // seq_header_id
  writer.put_bits(0U, 5U);                            // seq_profile_idc
  writer.put_bit(true);                               // single_picture_header
  writer.put_bits(static_cast<std::uint32_t>(config.seq_level_idx), 5U);
  put_uvlc(writer, 0);  // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);  // bit_depth_idc = 1 => 8-bit

  const std::uint32_t width = static_cast<std::uint32_t>(config.width);
  const std::uint32_t height = static_cast<std::uint32_t>(config.height);
  unsigned width_bits = bit_width(width - 1U);
  unsigned height_bits = bit_width(height - 1U);
  if (width_bits == 0U) width_bits = 1U;
  if (height_bits == 0U) height_bits = 1U;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(height) - 1U, height_bits);

  if (!write_cropping_window(writer, features.crop, error)) return false;

  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  if (!write_pb_aspect_ratio(writer, features.aspect, error)) return false;

  if (!write_sequence_segment_config(writer, features.segment, error)) {
    return false;
  }

  writer.put_bit(false);        // enable_dip
  writer.put_bit(true);         // enable_intra_edge_filter
  writer.put_bit(false);        // enable_mrls
  writer.put_bit(config.cfl);   // enable_cfl_intra
  writer.put_bits(0U, 2U);      // cfl_ds_filter_index
  writer.put_bit(false);        // enable_mhccp
  writer.put_bit(false);        // enable_ibp

  writer.put_bit(false);   // enable_refmvbank
  writer.put_bit(true);    // disable_drl_reorder
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);   // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);   // enable_bawp

  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(true);   // enable_chroma_dctonly
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  if (!write_df_par(writer, features.dfpar, error)) return false;

  SequenceTileConfig tile = features.tile;
  if (!write_sequence_tile_config(writer, &tile, config.width, config.height,
                                  SbSize::k64x64, config.seq_level_idx, 0,
                                  error)) {
    return false;
  }

  writer.put_bit(false);  // film_grain_params_present
  return true;
}

// M1 frame_header() for a keyframe with the tile_info() body delegated to
// write_frame_tile_info(). Everything else mirrors still_headers.cpp exactly.
bool write_still_keyframe_header(BitWriter& writer, const StillTemplate& config,
                              const Features& features,
                              FrameTileInfo* tile_info,
                              std::string* error) {
  writer.put_bit(true);  // is_first_tile_group
  put_uvlc(writer, 0);   // cur_mfh_id
  put_uvlc(writer, 0);   // seq_header_id_in_frame_header
  writer.put_bit(false);  // allow_screen_content_tools
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(true);   // disable_cdf_update

  FrameTileRequest request;
  request.reuse_sequence = true;
  if (!write_frame_tile_info(writer, features.tile, config.width,
                             config.height, SbSize::k64x64,
                             config.seq_level_idx, 0, request, tile_info,
                             error)) {
    return false;
  }

  writer.put_bits(static_cast<std::uint64_t>(config.qp), 8U);  // base_q_idx
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present

  writer.put_bit(config.deblock);  // apply_deblocking_filter[0]
  writer.put_bit(config.deblock);  // apply_deblocking_filter[1]
  if (config.deblock) {
    writer.put_bit(true);  // apply_deblocking_filter[2]
    writer.put_bit(true);  // apply_deblocking_filter[3]
    for (int i = 0; i < 4; ++i) writer.put_bit(false);  // df_delta_q_present
  }

  writer.put_bit(false);    // tx_mode_select = TX_MODE_LARGEST
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return true;
}

bool features_valid(const StillTemplate& config, const Features& features,
                    std::string* error) {
  if (!validate_cropping_window(
          features.crop, static_cast<std::uint32_t>(config.width),
          static_cast<std::uint32_t>(config.height), error)) {
    return false;
  }
  if (!validate_pb_aspect_ratio(features.aspect, error)) return false;
  if (!validate_df_par(features.dfpar, error)) return false;
  if (!validate_sequence_segment_config(features.segment, error)) return false;
  if (features.tile.present) {
    if (!validate_sequence_tile_config(features.tile, config.width,
                                       config.height, SbSize::k64x64,
                                       config.seq_level_idx, 0, error)) {
      return false;
    }
  }
  return true;
}

bool build_temporal_unit(const StillTemplate& config, Features features,
                         std::span<const std::uint8_t> tile_payload,
                         std::vector<std::uint8_t>* temporal_unit,
                         FrameTileInfo* tile_info, std::string* error) {
  // Sequence tile configs are produced by the sequence writer; do the same
  // partial construction here so frame tile_info() sees the final state.
  if (features.tile.present) {
    BitWriter scratch;
    SequenceTileConfig tile = features.tile;
    if (!write_sequence_tile_config(scratch, &tile, config.width, config.height,
                                    SbSize::k64x64, config.seq_level_idx, 0,
                                    error)) {
      return false;
    }
    features.tile = tile;
  }
  if (!features_valid(config, features, error)) return false;

  BitWriter seq_writer;
  if (!write_still_sequence_header(seq_writer, config, features, error)) {
    return false;
  }
  seq_writer.put_bit(false);  // obu_extension_flag
  seq_writer.put_bit(true);   // trailing one bit
  seq_writer.align_to_byte();
  const std::vector<std::uint8_t> seq_payload = take_aligned(seq_writer);
  const std::vector<std::uint8_t> seq_obu =
      wrap_obu(1, seq_payload);  // OBU_SEQUENCE_HEADER

  BitWriter frame_writer;
  if (!write_still_keyframe_header(frame_writer, config, features, tile_info,
                                error)) {
    return false;
  }
  frame_writer.align_to_byte();
  std::vector<std::uint8_t> frame_payload = take_aligned(frame_writer);
  frame_payload.insert(frame_payload.end(), tile_payload.begin(),
                       tile_payload.end());
  const std::vector<std::uint8_t> clk_obu =
      wrap_obu(4, frame_payload);  // OBU_CLOSED_LOOP_KEY
  const std::vector<std::uint8_t> td_obu = wrap_obu(2, {});  // TD

  temporal_unit->clear();
  temporal_unit->insert(temporal_unit->end(), td_obu.begin(), td_obu.end());
  temporal_unit->insert(temporal_unit->end(), seq_obu.begin(), seq_obu.end());
  temporal_unit->insert(temporal_unit->end(), clk_obu.begin(), clk_obu.end());
  return true;
}

std::vector<std::uint8_t> make_ivf(const std::vector<std::uint8_t>& file_header,
                                   std::span<const std::uint8_t> frame) {
  std::vector<std::uint8_t> out(file_header.begin(), file_header.end());
  if (out.size() < 32U) out.resize(32U, 0U);
  // Frame count = 1.
  out[24] = 1;
  out[25] = 0;
  out[26] = 0;
  out[27] = 0;
  append_le32(out, static_cast<std::uint32_t>(frame.size()));
  for (int i = 0; i < 8; ++i) out.push_back(0);  // timestamp
  out.insert(out.end(), frame.begin(), frame.end());
  return out;
}

bool same_bytes(std::span<const std::uint8_t> a,
                std::span<const std::uint8_t> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size()) == 0;
}

struct GateCase {
  std::string name;
  Features features;
  DisplaySize display;
  int crop_left = 0;
  int crop_top = 0;
};

}  // namespace

int main(int argc, char** argv) {
  std::string baseline;
  std::string out_dir = ".";
  std::string prefix = "small_header";
  StillTemplate config;
  bool have_dims = false;
  bool have_qp = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "gate: %s requires an argument\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--baseline") {
      baseline = next("--baseline");
    } else if (arg == "--out-dir") {
      out_dir = next("--out-dir");
    } else if (arg == "--prefix") {
      prefix = next("--prefix");
    } else if (arg == "--width") {
      config.width = std::atoi(next("--width"));
      have_dims = true;
    } else if (arg == "--height") {
      config.height = std::atoi(next("--height"));
    } else if (arg == "--qp") {
      config.qp = std::atoi(next("--qp"));
      have_qp = true;
    } else if (arg == "--deblock") {
      config.deblock = std::atoi(next("--deblock")) != 0;
    } else if (arg == "--single-block") {
      config.single_block = true;
    } else {
      std::fprintf(stderr, "gate: unknown option %s\n", arg.c_str());
      return 2;
    }
  }
  if (baseline.empty() || config.width <= 0 || config.height <= 0 ||
      !have_dims || !have_qp) {
    std::fprintf(stderr,
                 "usage: %s --baseline FILE --out-dir DIR --prefix NAME "
                 "--width W --height H --qp Q [--deblock 0|1] "
                 "[--single-block]\n",
                 argv[0]);
    return 2;
  }

  const std::vector<std::uint8_t> ivf = read_file(baseline);
  if (ivf.size() < 32U + 12U) {
    FAIL("baseline %s is not an IVF file", baseline.c_str());
    return 1;
  }
  const std::vector<std::uint8_t> file_header(ivf.begin(), ivf.begin() + 32);
  const std::uint32_t frame_size = read_le32(ivf.data() + 32);
  if (32U + 12U + frame_size > ivf.size()) {
    FAIL("truncated IVF frame in %s", baseline.c_str());
    return 1;
  }
  const std::span<const std::uint8_t> frame(ivf.data() + 44,
                                           frame_size);

  std::vector<Obu> obus;
  std::string error;
  if (!parse_obus(frame, &obus, &error)) {
    FAIL("parse OBUs: %s", error.c_str());
    return 1;
  }
  const Obu* td = nullptr;
  const Obu* seq = nullptr;
  const Obu* clk = nullptr;
  for (const Obu& obu : obus) {
    if (obu.type == 2 && td == nullptr) td = &obu;
    if (obu.type == 1 && seq == nullptr) seq = &obu;
    if (obu.type == 4 && clk == nullptr) clk = &obu;
  }
  if (td == nullptr || seq == nullptr || clk == nullptr) {
    FAIL("baseline needs TD, SEQ and CLK OBUs");
    return 1;
  }
  if (!td->payload.empty()) {
    FAIL("temporal delimiter payload is not empty");
    return 1;
  }

  Features none;
  FrameTileInfo baseline_tile_info;
  std::vector<std::uint8_t> baseline_header;
  {
    BitWriter frame_writer;
    if (!write_still_keyframe_header(frame_writer, config, none,
                                  &baseline_tile_info, &error)) {
      FAIL("baseline frame header: %s", error.c_str());
      return 1;
    }
    baseline_header = take_aligned(frame_writer);
  }
  if (clk->payload.size() < baseline_header.size() ||
      !same_bytes(std::span<const std::uint8_t>(clk->payload.data(),
                                                baseline_header.size()),
                  baseline_header)) {
    FAIL("frame-header reconstruction does not match the M1 stream "
         "(header %zu bytes, CLK %zu bytes)",
         baseline_header.size(), clk->payload.size());
    return 1;
  }
  const std::span<const std::uint8_t> tile_payload(
      clk->payload.data() + baseline_header.size(),
      clk->payload.size() - baseline_header.size());
  if (tile_payload.empty()) {
    FAIL("no tile payload after the frame header");
    return 1;
  }

  // Full parity: features disabled must reproduce the original temporal unit.
  std::vector<std::uint8_t> clone;
  FrameTileInfo clone_tile_info;
  if (!build_temporal_unit(config, none, tile_payload, &clone,
                           &clone_tile_info, &error)) {
    FAIL("baseline clone: %s", error.c_str());
    return 1;
  }
  // Compare OBU-by-OBU instead of relying on a private OBU layout.
  {
    std::vector<Obu> clone_obus;
    if (!parse_obus(clone, &clone_obus, &error)) {
      FAIL("clone parse: %s", error.c_str());
      return 1;
    }
    if (clone_obus.size() != 3U || clone_obus[0].type != 2 ||
        clone_obus[1].type != 1 || clone_obus[2].type != 4) {
      FAIL("clone did not produce TD/SEQ/CLK");
      return 1;
    }
    if (!same_bytes(clone_obus[1].payload, seq->payload)) {
      FAIL("sequence-header reconstruction does not match the M1 stream");
      return 1;
    }
    if (!same_bytes(clone_obus[2].payload, clk->payload)) {
      FAIL("CLK reconstruction does not match the M1 stream");
      return 1;
    }
  }

  // Cropping uses 8 pixels at maximum frame size. All test inputs are at
  // least 64x64, so the derived display rectangle is non-empty.
  CroppingWindow crop;
  crop.present = true;
  crop.left = 8;
  crop.right = 8;
  crop.top = 8;
  crop.bottom = 8;
  DisplaySize crop_size;
  if (!derive_display_size(
          crop, static_cast<std::uint32_t>(config.width),
          static_cast<std::uint32_t>(config.height),
          static_cast<std::uint32_t>(config.width),
          static_cast<std::uint32_t>(config.height), &crop_size, &error)) {
    FAIL("crop derivation: %s", error.c_str());
    return 1;
  }

  std::vector<GateCase> cases;
  auto add_case = [&](const char* name, Features features,
                      bool cropped) {
    GateCase c;
    c.name = name;
    c.features = std::move(features);
    if (cropped) {
      c.display = crop_size;
      c.crop_left = 8;
      c.crop_top = 8;
    } else {
      c.display.width = static_cast<std::uint32_t>(config.width);
      c.display.height = static_cast<std::uint32_t>(config.height);
    }
    cases.push_back(std::move(c));
  };

  {
    Features f;
    f.crop = crop;
    add_case("crop", f, true);
  }
  {
    Features f;
    f.aspect.reduce = true;
    // --single-block streams contain only square 64x64 leaves; the full M1
    // tree is also within 4:1 on these clips. Keep the gate conservative.
    f.aspect.max_pb_aspect_ratio_log2_minus_1 =
        config.single_block ? 0 : 1;
    add_case("aspect", f, false);
  }
  {
    Features f;
    f.dfpar.df_par_bits_minus_2 = 3;  // 5-bit df_delta_q domain
    add_case("dfpar", f, false);
  }
  {
    Features f;
    f.segment.enable_ext_seg = true;  // MaxSegments = 16
    f.segment.present = true;
    f.segment.allow_change = false;
    f.segment.info.enabled[0][0] = true;
    f.segment.info.value[0][0] = -7;
    f.segment.info.enabled[1][0] = true;
    f.segment.info.value[1][0] = 12;
    add_case("segment", f, false);
  }
  {
    Features f;
    f.tile.present = true;
    f.tile.allow_change = false;
    add_case("tile", f, false);
  }
  {
    Features f;
    f.crop = crop;
    f.aspect.reduce = true;
    f.aspect.max_pb_aspect_ratio_log2_minus_1 =
        config.single_block ? 0 : 1;
    f.dfpar.df_par_bits_minus_2 = 3;
    f.segment.enable_ext_seg = true;
    f.segment.present = true;
    f.segment.info.enabled[0][0] = true;
    f.segment.info.value[0][0] = -7;
    f.tile.present = true;
    f.tile.allow_change = true;  // frame explicitly reuses sequence tiles
    add_case("all", f, true);
  }

  const std::string manifest_path = out_dir + "/" + prefix + "_manifest.txt";
  std::ofstream manifest(manifest_path, std::ios::trunc);
  if (!manifest) {
    FAIL("cannot write %s", manifest_path.c_str());
    return 1;
  }
  const std::string baseline_path = out_dir + "/" + prefix + "_baseline.ivf";
  if (!write_file(baseline_path, ivf)) {
    FAIL("cannot write %s", baseline_path.c_str());
    return 1;
  }
  manifest << "baseline " << prefix << "_baseline.ivf " << config.width << ' '
           << config.height << " 0 0\n";

  std::printf("gate: header parity OK (SEQ %zu bytes, CLK header %zu bytes, "
              "tile %zu bytes), %zu feature streams\n",
              seq->payload.size(), baseline_header.size(), tile_payload.size(),
              cases.size());

  for (const GateCase& c : cases) {
    std::vector<std::uint8_t> temporal_unit;
    FrameTileInfo tile_info;
    if (!build_temporal_unit(config, c.features, tile_payload, &temporal_unit,
                             &tile_info, &error)) {
      FAIL("%s: %s", c.name.c_str(), error.c_str());
      continue;
    }
    const std::vector<std::uint8_t> out = make_ivf(file_header, temporal_unit);
    const std::string filename = prefix + "_" + c.name + ".ivf";
    const std::string path = out_dir + "/" + filename;
    if (!write_file(path, out)) {
      FAIL("cannot write %s", path.c_str());
      continue;
    }
    manifest << c.name << ' ' << filename << ' ' << c.display.width << ' '
             << c.display.height << ' ' << c.crop_left << ' ' << c.crop_top
             << '\n';
    std::printf("gate: %-8s %s  display %ux%u  tile %dx%d  frame %zu bytes\n",
                c.name.c_str(), filename.c_str(), c.display.width,
                c.display.height, tile_info.params.tile_cols,
                tile_info.params.tile_rows, out.size());
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "gate: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("gate: PASS\n");
  return 0;
}
