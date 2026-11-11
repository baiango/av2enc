// Decoder gate for the M2 delta-Q / segmentation frame-header syntax.
//
// What this gate proves end-to-end without edits to the M1/M2 core writers:
//
//   * A P frame whose only superblock is a single skipped 64x64 block codes
//     no delta_q_abs and no segment_id symbols:
//       - §5.20.5.11 gates the delta on `!(MiSize == SbSize && skip_flag)`;
//       - §5.20.5.8 gates read_segment_id() on
//         `skip_flag && !HasLosslessSegment`;
//       - §5.20.7.2 makes the pre-skip inter_segment_id() call a no-op when
//         SegIdPreSkip == 0.
//     The M2 tile payload therefore stays valid while the frame header carries
//     real segmentation_params() / delta_q_params() bits.
//   * `primary_ref_is_none`: M2's first P frame references a key frame.
//     choose_primary_secondary_ref_frame() only considers INTER_FRAME
//     references, so DerivedPrimaryRefFrame is PRIMARY_REF_NONE and
//     segmentation_update_map / segmentation_temporal_update are inferred
//     (1 / 0) with no bits.  The gate exercises that path.
//
// The gate splices the fields into an otherwise unmodified M2 frame header,
// checks that the disabled splice is byte-identical to the M2 header, decodes
// the resulting IVF with the pinned dav2d, and compares every sample against
// the M2 key-frame reconstruction (the all-skip P frame copies it).
//
// Usage:
//   ./src/encoder/m2/deltaq/build_gate.sh
//   DAV2D=/path/to/dav2d ./src/encoder/m2/deltaq/build_gate.sh
//   GEOMETRY=0 ./src/encoder/m2/deltaq/build_gate.sh --no-decode
#include "encoder/m1/deltaq/deltaq.h"
#include "encoder/m2/deltaq/m2_deltaq.h"
#include "encoder/m2/m2_frame.h"
#include "encoder/m2/m2_headers.h"

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::m2::M2Config;
using av2enc::m2::M2FrameHeader;
using av2enc::m2::M2InputFrame;
using av2enc::m2::M2Options;
using av2enc::m2::M2SequenceResult;
namespace m2deltaq = av2enc::m2::deltaq;
namespace rc = av2enc::rc;

int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

OwnedFrame make_frame(int width, int height) {
  OwnedFrame frame;
  frame.width = static_cast<std::uint32_t>(width);
  frame.height = static_cast<std::uint32_t>(height);
  frame.y.resize(static_cast<std::size_t>(width) *
                 static_cast<std::size_t>(height));
  frame.u.resize(static_cast<std::size_t>(width / 2) *
                 static_cast<std::size_t>(height / 2));
  frame.v.resize(static_cast<std::size_t>(width / 2) *
                 static_cast<std::size_t>(height / 2));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      frame.y[static_cast<std::size_t>(y) *
                  static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(24 + ((x * 3 + y * 5) & 0x7f));
    }
  }
  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      frame.u[static_cast<std::size_t>(y) *
                  static_cast<std::size_t>(width / 2) +
              static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(96 + ((x + y) & 0x3f));
      frame.v[static_cast<std::size_t>(y) *
                  static_cast<std::size_t>(width / 2) +
              static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(160 - ((x * 2 + y) & 0x3f));
    }
  }
  return frame;
}

M2InputFrame view(const OwnedFrame& frame) {
  M2InputFrame out;
  out.width = frame.width;
  out.height = frame.height;
  out.y = frame.y;
  out.u = frame.u;
  out.v = frame.v;
  return out;
}

M2Config make_config(int width, int height) {
  M2Config config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;
  return config;
}

void copy_bits(av2enc::BitWriter& dst, const std::vector<std::uint8_t>& src,
               std::size_t src_bit, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t pos = src_bit + i;
    const bool bit = ((src[pos >> 3U] >> (7U - (pos & 7U))) & 1U) != 0U;
    dst.put_bit(bit);
  }
}

// Bits after the replaced fields (segmentation_enabled, using_qmatrix,
// delta_q_present) in the M2 frame header.  Verified against the unmodified
// header in build_p_obu() before it is relied on.
std::size_t suffix_bits_for(const M2FrameHeader& header) {
  return header.keyframe ? 5U : 7U;
}

bool parse_obu(std::span<const std::uint8_t> data, std::size_t& pos,
               std::span<const std::uint8_t>& payload, std::string& error) {
  std::uint64_t size = 0;
  unsigned shift = 0;
  std::size_t cursor = pos;
  bool terminated = false;
  while (cursor < data.size()) {
    const std::uint8_t byte = data[cursor++];
    size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    shift += 7U;
    if ((byte & 0x80U) == 0U) {
      terminated = true;
      break;
    }
    if (shift > 28U) break;
  }
  if (!terminated || size < 1U || cursor > data.size()) {
    error = "gate: malformed OBU size";
    return false;
  }
  const std::size_t leb_bytes = cursor - pos;
  if (leb_bytes + static_cast<std::size_t>(size) > data.size()) {
    error = "gate: truncated OBU";
    return false;
  }
  const std::uint8_t header = data[cursor];
  const std::size_t header_bytes = ((header & 0x80U) != 0U) ? 2U : 1U;
  if (size < header_bytes) {
    error = "gate: OBU smaller than its header";
    return false;
  }
  const std::size_t payload_start = cursor + header_bytes;
  const std::size_t payload_size =
      static_cast<std::size_t>(size) - header_bytes;
  if (payload_start + payload_size > data.size()) {
    error = "gate: truncated OBU payload";
    return false;
  }
  payload = data.subspan(payload_start, payload_size);
  pos += leb_bytes + static_cast<std::size_t>(size);
  return true;
}

// Splices segmentation/delta-Q fields into an M2 frame header and appends the
// tile payload.  Includes is_first_tile_group and byte_alignment.
bool build_p_obu(const M2Config& config, const M2FrameHeader& header,
                 const m2deltaq::FrameQpPlan& plan,
                 std::span<const std::uint8_t> tile,
                 std::vector<std::uint8_t>& obu, std::string& error) {
  error.clear();
  av2enc::BitWriter probe;
  av2enc::m2::write_frame_header(probe, config, header);
  const std::size_t original_bits = probe.bit_size();
  const std::vector<std::uint8_t> original = probe.take_bytes();
  const std::size_t suffix = suffix_bits_for(header);
  if (original_bits < 3U + suffix) {
    error = "gate: M2 frame header shorter than the replaced fields";
    return false;
  }
  const std::size_t prefix = original_bits - 3U - suffix;

  // Baseline check: prefix + three zero bits + suffix must reproduce the M2
  // header exactly.  If M2 changes the header, this fails loudly instead of
  // silently misplacing the syntax.
  {
    av2enc::BitWriter check;
    copy_bits(check, original, 0, prefix);
    check.put_bits(0U, 3U);
    copy_bits(check, original, prefix + 3U, suffix);
    const std::size_t checked_bits = check.bit_size();
    const std::vector<std::uint8_t> checked = check.take_bytes();
    const bool same_bytes =
        checked_bits == original_bits && checked.size() == original.size() &&
        std::memcmp(checked.data(), original.data(), original.size()) == 0;
    if (!same_bytes) {
      error = "gate: M2 frame-header layout changed; update suffix_bits_for()";
      return false;
    }
  }

  av2enc::BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  copy_bits(writer, original, 0, prefix);
  m2deltaq::write_frame_qp_syntax(writer, plan, header, &error);
  if (!error.empty()) return false;
  copy_bits(writer, original, prefix + 3U, suffix);
  writer.align_to_byte();

  const std::span<const std::uint8_t> header_bytes = writer.bytes();
  std::vector<std::uint8_t> payload(header_bytes.begin(), header_bytes.end());
  payload.insert(payload.end(), tile.begin(), tile.end());
  obu = av2enc::m0::make_obu(
      av2enc::m0::ObuHeader{av2enc::m0::ObuType::kRegularTileGroup}, payload);
  return true;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate : {"tools/bin/dav2d", "tools/bin/dav2d-macos",
                                "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

bool compare_decoded(const std::string& path,
                     const std::vector<OwnedFrame>& expected,
                     std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    if (index >= expected.size()) {
      error = "decoder produced more frames than expected";
      return false;
    }
    const OwnedFrame& ref = expected[index];
    if (frame.width() != ref.width || frame.height() != ref.height) {
      error = "decoded dimensions differ";
      return false;
    }
    const auto compare_plane = [&](std::span<const std::uint8_t> decoded,
                                   const std::vector<std::uint8_t>& want) {
      return decoded.size() == want.size() &&
             std::memcmp(decoded.data(), want.data(), want.size()) == 0;
    };
    if (!compare_plane(frame.y(), ref.y) ||
        !compare_plane(frame.u(), ref.u) ||
        !compare_plane(frame.v(), ref.v)) {
      error = "decoded frame " + std::to_string(index) +
              " differs from the M2 reconstruction";
      return false;
    }
    ++index;
  }
  if (index != expected.size()) {
    error = "decoder produced fewer frames than expected";
    return false;
  }
  return true;
}

bool run_variant(const std::string& name, const M2Config& config,
                 const M2FrameHeader& p_header,
                 const m2deltaq::FrameQpPlan& plan,
                 const std::vector<std::uint8_t>& key_unit,
                 const std::vector<std::uint8_t>& tile,
                 const std::vector<OwnedFrame>& expected, bool decode) {
  std::cout << "variant " << name << ": ";
  std::vector<std::uint8_t> p_obu;
  std::string error;
  if (!build_p_obu(config, p_header, plan, tile, p_obu, error)) {
    std::cerr << "FAIL build: " << error << "\n";
    ++g_failures;
    return false;
  }
  const std::vector<std::uint8_t> td = av2enc::m2::make_temporal_delimiter_obu();
  std::vector<std::uint8_t> p_unit = td;
  p_unit.insert(p_unit.end(), p_obu.begin(), p_obu.end());

  const std::filesystem::path out_path =
      std::filesystem::temp_directory_path() / ("m2_deltaq_" + name + ".ivf");
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = config.max_frame_width;
  header.height = config.max_frame_height;
  header.fps_num = 30;
  header.fps_den = 1;
  if (!writer.open(out_path, header, error) ||
      !writer.write_frame(key_unit, 0, error) ||
      !writer.write_frame(p_unit, 1, error) || !writer.close(error)) {
    std::cerr << "FAIL ivf: " << error << "\n";
    ++g_failures;
    return false;
  }
  std::cout << out_path.string();

  if (!decode) {
    std::cout << " (bytes only)\n";
    return true;
  }
  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cout << " (no dav2d; decode skipped)\n";
    std::cerr << "DECODE SKIP: set DAV2D to enable the decoder gate\n";
    return false;
  }
  const std::filesystem::path decoded = out_path.string() + ".y4m";
  const std::string command = dav2d + " -q -i " + out_path.string() + " -o " +
                              decoded.string() + " --threads 0";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "\nFAIL decode: " << command << " -> " << rc << "\n";
    ++g_failures;
    return false;
  }
  if (!compare_decoded(decoded.string(), expected, error)) {
    std::cerr << "\nFAIL recon: " << error << "\n";
    ++g_failures;
    return false;
  }
  std::cout << " -> decoded exact\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool decode = true;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-decode") == 0) {
      decode = false;
    } else {
      std::cerr << "usage: " << argv[0] << " [--no-decode]\n";
      return 2;
    }
  }

  constexpr int kWidth = 64;
  constexpr int kHeight = 64;
  const OwnedFrame frame = make_frame(kWidth, kHeight);
  std::vector<OwnedFrame> frames = {frame, frame};
  std::vector<M2InputFrame> views;
  for (const OwnedFrame& f : frames) views.push_back(view(f));

  M2Options options;
  options.base_q_idx = 90;
  options.force_single_block = true;
  options.force_global_mv = true;
  options.enable_intra_fallback = false;
  options.enable_newmv = false;
  options.enable_nearmv = false;
  options.enable_inter_residual = false;

  M2SequenceResult sequence;
  std::string error;
  if (!av2enc::m2::encode_m2_sequence(views, options, sequence, error)) {
    std::cerr << "FAIL m2 encode: " << error << "\n";
    return 1;
  }
  if (sequence.units.size() != 2) {
    std::cerr << "FAIL m2 encode: expected 2 temporal units\n";
    return 1;
  }
  CHECK(sequence.inter_blocks[1] == 1);
  CHECK(sequence.intra_blocks[1] == 0);

  const M2Config config = make_config(kWidth, kHeight);
  M2FrameHeader p_header;
  p_header.keyframe = false;
  p_header.order_hint = 1;
  p_header.ref_slot = 0;
  p_header.refresh_frame_flags = 1;

  // Extract the P tile from the M2 sequence.  With identical frames,
  // force_single_block + force_global_mv + no inter residual, the P frame is
  // one 64x64 skip block.
  av2enc::BitWriter probe;
  av2enc::m2::write_frame_header(probe, config, p_header);
  const std::size_t original_header_bits = probe.bit_size();
  const std::size_t tile_group_header_bytes =
      (1U + original_header_bits + 7U) / 8U;

  std::span<const std::uint8_t> p_payload;
  {
    const std::vector<std::uint8_t>& unit = sequence.units[1];
    std::size_t pos = 0;
    std::span<const std::uint8_t> td_payload;
    if (!parse_obu(unit, pos, td_payload, error) ||
        !parse_obu(unit, pos, p_payload, error)) {
      std::cerr << "FAIL p parse: " << error << "\n";
      return 1;
    }
  }
  if (p_payload.size() <= tile_group_header_bytes) {
    std::cerr << "FAIL p parse: no tile payload\n";
    return 1;
  }
  const std::vector<std::uint8_t> tile(
      p_payload.begin() + static_cast<std::ptrdiff_t>(tile_group_header_bytes),
      p_payload.end());

  // Baseline: the disabled splice must reproduce the M2 P frame OBU byte for
  // byte.  This pins the field offset and the tile extraction.
  {
    const std::vector<std::uint8_t>& sequence_p = sequence.units[1];
    const std::vector<std::uint8_t> td =
        av2enc::m2::make_temporal_delimiter_obu();
    if (sequence_p.size() <= td.size()) {
      std::cerr << "FAIL baseline: truncated M2 P unit\n";
      return 1;
    }
    const std::span<const std::uint8_t> sequence_obu(
        sequence_p.data() + td.size(), sequence_p.size() - td.size());

    m2deltaq::FrameQpPlan disabled;
    disabled.base_q_idx = 90;
    std::vector<std::uint8_t> obu;
    if (!build_p_obu(config, p_header, disabled, tile, obu, error)) {
      std::cerr << "FAIL baseline build: " << error << "\n";
      return 1;
    }
    if (obu.size() != sequence_obu.size() ||
        std::memcmp(obu.data(), sequence_obu.data(), obu.size()) != 0) {
      std::cerr
          << "FAIL baseline: spliced disabled header/tile differs from the "
             "M2 P frame OBU\n";
      ++g_failures;
    }
  }

  std::vector<OwnedFrame> expected;
  expected.push_back(OwnedFrame{});
  expected[0].width = static_cast<std::uint32_t>(kWidth);
  expected[0].height = static_cast<std::uint32_t>(kHeight);
  expected[0].y = sequence.recon[0].y;
  expected[0].u = sequence.recon[0].u;
  expected[0].v = sequence.recon[0].v;
  // The all-skip P frame copies the key frame.
  const OwnedFrame key_copy = expected[0];
  expected.push_back(key_copy);

  bool ok = true;

  // Delta-Q only: segmentation disabled, delta_q_present = 1.
  m2deltaq::FrameQpPlan delta_only;
  delta_only.base_q_idx = 90;
  delta_only.primary_ref_is_none = true;
  delta_only.delta_q.present = true;
  delta_only.delta_q.res = 3;
  ok = run_variant("deltaq", config, p_header, delta_only, sequence.units[0],
                   tile, expected, decode) &&
       ok;

  // Segmentation only: delta_q absent, update_map inferred 1 / temporal 0.
  m2deltaq::FrameQpPlan seg_only;
  seg_only.base_q_idx = 90;
  seg_only.primary_ref_is_none = true;
  seg_only.segmentation.enabled = true;
  seg_only.segmentation.update_map = true;
  ok = run_variant("segmentation", config, p_header, seg_only,
                   sequence.units[0], tile, expected, decode) &&
       ok;

  // Both tools together.
  m2deltaq::FrameQpPlan both;
  both.base_q_idx = 90;
  both.primary_ref_is_none = true;
  both.delta_q.present = true;
  both.delta_q.res = 0;
  both.segmentation.enabled = true;
  both.segmentation.update_map = true;
  ok = run_variant("both", config, p_header, both, sequence.units[0], tile,
                   expected, decode) &&
       ok;

  // An inter frame that references an INTER frame must code the update_map
  // bit.  Build that header with a plan that says so; it cannot be decoded as
  // a valid P frame against our key-only reference chain, so only the header
  // round-trip is checked here (see the M2 unit test for the bit-level test).
  {
    m2deltaq::FrameQpPlan coded_update;
    coded_update.base_q_idx = 90;
    coded_update.primary_ref_is_none = false;
    coded_update.segmentation.enabled = true;
    coded_update.segmentation.update_map = true;
    av2enc::BitWriter writer;
    const std::size_t bits = m2deltaq::write_frame_qp_syntax(
        writer, coded_update, p_header, &error);
    CHECK(error.empty());
    CHECK(bits > 0);
    av2enc::deltaq::BitReader reader(writer.bytes());
    rc::SegmentationParams seg;
    rc::DeltaQParams delta;
    bool using_qmatrix = true;
    const av2enc::deltaq::FrameQpHeader qh =
        m2deltaq::frame_qp_header(coded_update);
    CHECK(av2enc::deltaq::read_segmentation_and_delta_q(
        reader, coded_update.base_q_idx, qh, seg, delta, using_qmatrix,
        &error));
    CHECK(error.empty());
    CHECK(seg.enabled);
    CHECK(seg.update_map);
    CHECK(!seg.temporal_update);
    CHECK(!delta.present);
  }

  if (g_failures != 0 || !ok) {
    std::cerr << g_failures << " gate failure(s), variants_ok=" << ok << "\n";
    return 1;
  }
  std::cout << "ok: delta-Q/segmentation decoder gate\n";
  return 0;
}
