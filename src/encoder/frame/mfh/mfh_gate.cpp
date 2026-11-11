// Multi-frame header decoder gate.
//
// Encodes a short M2 P-frame sequence with the untouched M2 encoder, then
// re-emits the same tile payloads with:
//
//   baseline : the M2 headers (cur_mfh_id = 0), re-emitted through the
//              mfh_m2_adapter to prove the adapter is bit-exact
//   mfh      : one OBU_MULTI_FRAME_HEADER in the first temporal unit; every
//              frame header codes cur_mfh_id = 1, omits
//              seq_header_id_in_frame_header, omits the two
//              apply_deblocking_filter bits (MFH update present) and inherits
//              the frame size / segment table from the MFH
//
// The pinned dav2d must decode both streams with --strict 1 --threads 1 and
// match the encoder reconstruction sample for sample.
//
// Usage:
//   mfh_gate [--input clip.y4m] [--frames N] [--out out.ivf] [--dav2d path]
//
// Exit: 0 pass, 1 failure, 2 usage, 77 skip (no decoder / no frames).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "encoder/frame/mfh/mfh.h"
#include "encoder/frame/mfh/mfh_m2_adapter.h"
#include "encoder/m2/m2_frame.h"
#include "encoder/m2/m2_headers.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

using av2enc::IvfHeader;
using av2enc::IvfWriter;
using av2enc::Y4mFrame;
using av2enc::Y4mReadResult;
using av2enc::Y4mReader;
using av2enc::frame::mfh::make_m2_frame_obu_with_mfh;
using av2enc::frame::mfh::make_multi_frame_header_obu;
using av2enc::frame::mfh::M2MfhConfig;
using av2enc::frame::mfh::M2MfhFrameHeader;
using av2enc::frame::mfh::MultiFrameHeader;
using av2enc::frame::mfh::ObuView;
using av2enc::frame::mfh::parse_obu;
using av2enc::m2::M2InputFrame;
using av2enc::m2::M2Options;
using av2enc::m2::M2ReconFrame;
using av2enc::m2::M2SequenceResult;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

struct Frame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t synthetic_sample(int x, int y, int frame, int plane) {
  const int sx = 6 + 3 * frame;
  const int sy = 8 + 2 * frame;
  const bool in_square = x >= sx && x < sx + 28 && y >= sy && y < sy + 28;
  if (plane == 0) {
    if (in_square) {
      return static_cast<std::uint8_t>(48 + ((x * 29 + y * 17 + frame) & 0x7F));
    }
    return static_cast<std::uint8_t>(92 + ((x + 2 * y) >> 3));
  }
  if (plane == 1) return static_cast<std::uint8_t>(in_square ? 96 : 128);
  return static_cast<std::uint8_t>(in_square ? 152 : 128);
}

std::vector<Frame> make_synthetic_clip(int frames, int width, int height) {
  std::vector<Frame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    Frame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(static_cast<std::size_t>(width) * height);
    frame.u.resize(static_cast<std::size_t>(cw) * ch);
    frame.v.resize(static_cast<std::size_t>(cw) * ch);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * width + x] =
            synthetic_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * cw + x] =
            synthetic_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * cw + x] =
            synthetic_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

bool load_y4m(const std::string& path, int max_frames,
              std::vector<Frame>& out, std::string& error) {
  Y4mReader reader(path);
  if (!reader.open(error)) return false;
  Y4mFrame frame;
  while (static_cast<int>(out.size()) < max_frames) {
    const Y4mReadResult status = reader.read_frame(frame, error);
    if (status == Y4mReadResult::kEof) break;
    if (status == Y4mReadResult::kError) return false;
    Frame owned;
    owned.width = frame.width();
    owned.height = frame.height();
    owned.y.assign(frame.y().begin(), frame.y().end());
    owned.u.assign(frame.u().begin(), frame.u().end());
    owned.v.assign(frame.v().begin(), frame.v().end());
    out.push_back(std::move(owned));
  }
  if (out.empty()) {
    error = "Y4M input contained no frames";
    return false;
  }
  return true;
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

bool write_ivf(const std::string& path, const std::vector<std::vector<uint8_t>>& units,
               std::uint32_t width, std::uint32_t height, std::string& error) {
  IvfWriter writer;
  IvfHeader header;
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

struct SplitUnit {
  std::vector<std::uint8_t> td;
  std::vector<std::uint8_t> seq;    // empty for non-key units
  std::vector<std::uint8_t> frame;  // complete frame OBU
  std::uint8_t frame_type = 0;
};

bool split_unit(std::span<const std::uint8_t> unit, SplitUnit* out,
                std::string& error) {
  *out = SplitUnit{};
  std::size_t offset = 0;
  bool have_frame = false;
  while (offset < unit.size()) {
    ObuView view;
    if (!parse_obu(unit, offset, &view, &error)) return false;
    const auto bytes = unit.subspan(view.offset, view.total_size);
    if (view.obu_type == 2) {
      out->td.assign(bytes.begin(), bytes.end());
    } else if (view.obu_type == 1) {
      out->seq.assign(bytes.begin(), bytes.end());
    } else if (view.obu_type == 4 || view.obu_type == 7) {
      out->frame.assign(bytes.begin(), bytes.end());
      out->frame_type = view.obu_type;
      have_frame = true;
    }
    offset += view.total_size;
  }
  if (!have_frame) {
    error = "unit has no frame OBU";
    return false;
  }
  return true;
}

// Reads the leading frame_header_info bits of an M2 frame OBU far enough to
// recover frame_is_inter.  The unit tests prove the rest of the header is
// deterministic for the M2 encoder.
bool read_frame_is_inter(std::span<const std::uint8_t> frame_obu, bool key,
                         bool* out, std::string& error) {
  ObuView view;
  if (!parse_obu(frame_obu, 0, &view, &error)) return false;
  if (view.payload_offset + view.payload_size > frame_obu.size()) {
    error = "frame OBU payload out of range";
    return false;
  }
  const auto payload =
      frame_obu.subspan(view.payload_offset, view.payload_size);
  std::size_t bit = 0;
  auto read_bit = [&](bool* value) {
    if (bit >= payload.size() * 8U) return false;
    const std::size_t byte = bit >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit & 7U);
    *value = ((payload[byte] >> shift) & 1U) != 0U;
    ++bit;
    return true;
  };
  auto read_uvlc = [&](std::uint32_t* value) {
    unsigned zeros = 0;
    bool done = false;
    while (read_bit(&done)) {
      if (done) break;
      ++zeros;
      if (zeros >= 32U) return false;
    }
    std::uint32_t payload_bits = 0;
    for (unsigned i = 0; i < zeros; ++i) {
      bool b = false;
      if (!read_bit(&b)) return false;
      payload_bits = (payload_bits << 1U) | (b ? 1U : 0U);
    }
    *value = payload_bits + (1U << zeros) - 1U;
    return true;
  };

  bool is_first = false;
  std::uint32_t cur_mfh_id = 0;
  if (!read_bit(&is_first) || !read_uvlc(&cur_mfh_id)) {
    error = "truncated frame header";
    return false;
  }
  if (cur_mfh_id == 0) {
    std::uint32_t seq_id = 0;
    if (!read_uvlc(&seq_id)) {
      error = "truncated frame header";
      return false;
    }
  }
  if (key) {
    *out = false;
    return true;
  }
  bool inter = false;
  if (!read_bit(&inter)) {
    error = "truncated frame header";
    return false;
  }
  *out = inter;
  return true;
}

bool run_decoder(const std::string& dav2d, const std::string& ivf,
                 const std::string& y4m, std::string& error,
                 bool quiet = false) {
  const std::string command = "\"" + dav2d + "\" -q -i \"" + ivf + "\" -o \"" +
                              y4m + "\" --threads 1 --strict 1" +
                              (quiet ? " 2>/dev/null" : "");
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    error = "dav2d failed (" + std::to_string(rc) + "): " + command;
    return false;
  }
  if (!std::filesystem::exists(y4m)) {
    error = "dav2d produced no output";
    return false;
  }
  return true;
}

bool compare_y4m(const std::string& decoded_path,
                 const std::vector<M2ReconFrame>& recon, std::uint64_t* diffs,
                 std::string& error) {
  *diffs = 0;
  Y4mReader reader(decoded_path);
  if (!reader.open(error)) return false;
  Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const Y4mReadResult status = reader.read_frame(frame, error);
    if (status == Y4mReadResult::kEof) break;
    if (status == Y4mReadResult::kError) return false;
    if (index >= recon.size()) {
      error = "decoder produced more frames than the encoder";
      return false;
    }
    const M2ReconFrame& ref = recon[index];
    if (frame.width() != ref.width || frame.height() != ref.height) {
      error = "decoded dimensions differ";
      return false;
    }
    const auto compare = [&](std::span<const std::uint8_t> decoded,
                             const std::vector<std::uint8_t>& expected) {
      if (decoded.size() != expected.size()) {
        *diffs += 1;
        return;
      }
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        if (decoded[i] != expected[i]) ++*diffs;
      }
    };
    compare(frame.y(), ref.y);
    compare(frame.u(), ref.u);
    compare(frame.v(), ref.v);
    ++index;
  }
  if (index != recon.size()) {
    error = "decoder produced " + std::to_string(index) + " of " +
            std::to_string(recon.size()) + " frames";
    return false;
  }
  return true;
}

struct StreamStats {
  std::size_t payload_bytes = 0;
  std::size_t header_bytes = 0;
};

// Re-emits every unit.  `mfh == nullptr` reproduces the M2 headers exactly;
// otherwise the first unit gets the MFH OBU and every frame references it.
bool build_stream(const M2SequenceResult& sequence,
                  const std::vector<M2ReconFrame>& recon,
                  const M2MfhConfig& config, const MultiFrameHeader* mfh,
                  std::vector<std::vector<std::uint8_t>>* units,
                  StreamStats* stats, std::string& error) {
  units->clear();
  units->reserve(sequence.units.size());
  *stats = StreamStats{};
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    SplitUnit split;
    if (!split_unit(sequence.units[i], &split, error)) {
      error = "frame " + std::to_string(i) + ": " + error;
      return false;
    }
    const bool key = (i == 0);
    if (key != (split.frame_type == 4)) {
      error = "frame " + std::to_string(i) + ": unexpected frame OBU type";
      return false;
    }
    if (key && split.seq.empty()) {
      error = "frame 0: missing sequence header OBU";
      return false;
    }
    // Tile payload = tail of the original frame OBU payload.
    ObuView frame_view;
    if (!parse_obu(split.frame, 0, &frame_view, &error)) return false;
    const std::size_t tile_bytes = sequence.tile_bytes[i];
    if (tile_bytes == 0 || tile_bytes > frame_view.payload_size) {
      error = "frame " + std::to_string(i) + ": bad tile size";
      return false;
    }
    const auto payload = std::span<const std::uint8_t>(
                             split.frame.data(), split.frame.size())
                             .subspan(frame_view.payload_offset,
                                      frame_view.payload_size);
    const auto tile = payload.subspan(payload.size() - tile_bytes);

    bool frame_is_inter = false;
    if (!key && !read_frame_is_inter(split.frame, key, &frame_is_inter,
                                     error)) {
      return false;
    }

    M2MfhFrameHeader header;
    header.keyframe = key;
    header.frame_is_inter = frame_is_inter;
    header.order_hint = static_cast<std::uint32_t>(i);
    header.ref_slot = 0;
    header.refresh_frame_flags = key ? 3U : 1U;
    header.allow_intrabc = false;

    const auto frame_obu =
        make_m2_frame_obu_with_mfh(config, header, mfh, tile, &error);
    if (frame_obu.empty() && !error.empty()) return false;

    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), split.td.begin(), split.td.end());
    if (key) {
      unit.insert(unit.end(), split.seq.begin(), split.seq.end());
      if (mfh != nullptr) {
        const auto mfh_obu = make_multi_frame_header_obu(*mfh, &error);
        if (mfh_obu.empty() && !error.empty()) return false;
        unit.insert(unit.end(), mfh_obu.begin(), mfh_obu.end());
        stats->header_bytes += mfh_obu.size();
      }
    }
    unit.insert(unit.end(), frame_obu.begin(), frame_obu.end());
    stats->payload_bytes += unit.size();
    units->push_back(std::move(unit));
  }
  if (recon.size() != units->size()) {
    error = "reconstruction count differs from the unit count";
    return false;
  }
  return true;
}

bool decode_and_compare(const std::string& dav2d, const std::string& path,
                        const std::string& decoded_path,
                        const std::vector<M2ReconFrame>& recon,
                        const char* label, std::uint64_t* diffs) {
  std::string error;
  if (!run_decoder(dav2d, path, decoded_path, error)) {
    check(false, std::string(label) + ": " + error);
    return false;
  }
  if (!compare_y4m(decoded_path, recon, diffs, error)) {
    check(false, std::string(label) + ": " + error);
    return false;
  }
  check(*diffs == 0, std::string(label) + ": recon differs in " +
                         std::to_string(*diffs) + " samples");
  std::printf("  %-8s %zu frame(s), %llu sample diff(s)\n", label,
              recon.size(), static_cast<unsigned long long>(*diffs));
  return *diffs == 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "mfh_gate.ivf";
  std::string dav2d;
  int frames = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--help") {
      std::printf(
          "usage: mfh_gate [--input clip.y4m] [--frames N] [--out out.ivf] "
          "[--dav2d path]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (frames < 2) frames = 2;
  if (dav2d.empty()) dav2d = find_dav2d();
  if (dav2d.empty() || !std::filesystem::exists(dav2d)) {
    std::fprintf(stderr, "SKIP: no dav2d binary (set DAV2D=...)\n");
    return 77;
  }

  std::vector<Frame> frames_owned;
  std::string error;
  if (!input.empty()) {
    if (!load_y4m(input, frames, frames_owned, error)) {
      std::fprintf(stderr, "Y4M LOAD FAIL: %s\n", error.c_str());
      return 2;
    }
  } else {
    frames_owned = make_synthetic_clip(frames, 64, 64);
  }
  if (frames_owned.size() < 2) {
    std::fprintf(stderr, "SKIP: need at least two frames\n");
    return 77;
  }
  if (static_cast<int>(frames_owned.size()) < frames) {
    frames = static_cast<int>(frames_owned.size());
  }

  std::vector<M2InputFrame> inputs;
  inputs.reserve(frames_owned.size());
  for (const Frame& frame : frames_owned) {
    M2InputFrame input_frame;
    input_frame.width = frame.width;
    input_frame.height = frame.height;
    input_frame.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input_frame.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input_frame.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input_frame);
  }

  M2Options options;
  options.trace = false;
  M2SequenceResult sequence;
  if (!av2enc::m2::encode_m2_sequence(inputs, options, sequence, error)) {
    std::fprintf(stderr, "ENCODE FAIL: %s\n", error.c_str());
    return 1;
  }
  const std::uint32_t width = frames_owned[0].width;
  const std::uint32_t height = frames_owned[0].height;

  M2MfhConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.disable_cdf_update = true;

  // Baseline stream straight from the M2 encoder.
  if (!write_ivf(out_path, sequence.units, width, height, error)) {
    std::fprintf(stderr, "IVF WRITE FAIL: %s\n", error.c_str());
    return 1;
  }

  // Re-emitted baseline and MFH streams.
  std::vector<std::vector<std::uint8_t>> reheader_units;
  StreamStats reheader_stats;
  if (!build_stream(sequence, sequence.recon, config, nullptr, &reheader_units,
                    &reheader_stats, error)) {
    std::fprintf(stderr, "REHEADER FAIL: %s\n", error.c_str());
    return 1;
  }
  const std::string reheader_path = out_path + ".reheader.ivf";
  if (!write_ivf(reheader_path, reheader_units, width, height, error)) {
    std::fprintf(stderr, "IVF WRITE FAIL: %s\n", error.c_str());
    return 1;
  }

  MultiFrameHeader mfh;
  mfh.seq_header_id = 0;
  mfh.mfh_id = 1;
  mfh.tlayer_id = 0;
  mfh.frame_size_present = true;
  mfh.frame_width = width;
  mfh.frame_height = height;
  mfh.deblocking_filter_update = true;
  mfh.apply_deblocking_filter = {false, false, false, false};
  mfh.seg_info_present = true;
  mfh.allow_seg_info_change = false;
  mfh.seg_info.ext_seg = false;

  std::vector<std::vector<std::uint8_t>> mfh_units;
  StreamStats mfh_stats;
  if (!build_stream(sequence, sequence.recon, config, &mfh, &mfh_units,
                    &mfh_stats, error)) {
    std::fprintf(stderr, "MFH STREAM FAIL: %s\n", error.c_str());
    return 1;
  }
  const std::string mfh_path = out_path + ".mfh.ivf";
  if (!write_ivf(mfh_path, mfh_units, width, height, error)) {
    std::fprintf(stderr, "IVF WRITE FAIL: %s\n", error.c_str());
    return 1;
  }

  std::printf("mfh_gate: decoder %s\n", dav2d.c_str());
  std::printf("mfh_gate: %zu frame(s) %ux%u, base q %u\n", mfh_units.size(),
              width, height, static_cast<unsigned>(options.base_q_idx));
  std::size_t baseline_bytes = 0;
  for (const auto& unit : sequence.units) baseline_bytes += unit.size();
  std::printf("  baseline payload %zu bytes; reheader %zu; MFH %zu (+%zu MFH)\n",
              baseline_bytes, reheader_stats.payload_bytes,
              mfh_stats.payload_bytes, mfh_stats.header_bytes);

  std::uint64_t diffs = 0;
  bool ok = true;
  ok &= decode_and_compare(dav2d, out_path, out_path + ".base.decoded.y4m",
                           sequence.recon, "baseline", &diffs);
  ok &= decode_and_compare(dav2d, reheader_path,
                           out_path + ".reheader.decoded.y4m", sequence.recon,
                           "reheader", &diffs);
  ok &= decode_and_compare(dav2d, mfh_path, out_path + ".mfh.decoded.y4m",
                           sequence.recon, "mfh", &diffs);

  // Negative probe: an MFH that references a sequence header which is not
  // available must be rejected under --strict 1 (§7.3.8.6).  This proves the
  // decoder parses and enforces the MFH rather than skipping it.
  {
    MultiFrameHeader bad_mfh = mfh;
    bad_mfh.seq_header_id = 1;  // only sequence header 0 is in the stream
    const auto bad_obu = make_multi_frame_header_obu(bad_mfh, &error);
    check(!bad_obu.empty(), "bad seq MFH: writer produced an OBU");

    std::vector<std::uint8_t> first = mfh_units[0];
    std::size_t offset = 0;
    bool replaced = false;
    while (offset < first.size()) {
      ObuView view;
      if (!parse_obu(first, offset, &view, &error)) break;
      if (view.obu_type == av2enc::frame::mfh::kObuTypeMultiFrameHeader) {
        std::vector<std::uint8_t> patched;
        patched.reserve(first.size() - view.total_size + bad_obu.size());
        patched.insert(patched.end(), first.begin(),
                       first.begin() + static_cast<std::ptrdiff_t>(view.offset));
        patched.insert(patched.end(), bad_obu.begin(), bad_obu.end());
        patched.insert(
            patched.end(),
            first.begin() +
                static_cast<std::ptrdiff_t>(view.offset + view.total_size),
            first.end());
        first = std::move(patched);
        replaced = true;
        break;
      }
      offset += view.total_size;
    }
    check(replaced, "bad seq MFH: found and replaced the MFH OBU");
    if (replaced) {
      std::vector<std::vector<std::uint8_t>> bad_units = mfh_units;
      bad_units[0] = std::move(first);
      const std::string bad_path = out_path + ".badseq.ivf";
      if (write_ivf(bad_path, bad_units, width, height, error)) {
        const std::string decoded = bad_path + ".decoded.y4m";
        std::string decode_error;
        const bool decoded_ok =
            run_decoder(dav2d, bad_path, decoded, decode_error,
                        /*quiet=*/true);
        check(!decoded_ok,
              "bad seq MFH: dav2d --strict 1 rejects an unavailable MFH "
              "sequence header");
      }
    }
  }

  std::printf("mfh_gate: %d checks, %d failure(s)\n", g_checks, g_failures);
  std::printf("%s\n", (ok && g_failures == 0) ? "MFH GATE PASS"
                                              : "MFH GATE FAIL");
  return (ok && g_failures == 0) ? 0 : 1;
}
