// Show-existing-frame decoder gate.
//
// Builds a verified 2- and 3-frame M2 stream with the existing M2 encoder,
// appends SEF OBUs (no new tile data), decodes with the pinned dav2d and checks
// that the extra output frame is byte-identical to the referenced frame.
//
// Three phases:
//   A. Pure append, explicit order hint (derive_sef_order_hint = 0).
//      The referenced frame was already output, so the SEF is exactly the
//      "a frame can be shown with a specified order hint without affecting the
//      normal output of that frame" case (§6.17.2 NOTE). The pinned dav2d drops
//      duplicate POCs unless --outputinvisible 1; the gate asserts both the
//      drop (default) and the byte-identical extra output (with the flag).
//   B. Two SEFs (regular + leading) appended to the same stream; both extra
//      outputs must be byte-identical to the referenced frame.
//   C. Hidden-reference path. The last M2 frame's immediate_output_frame bit
//      is cleared in place (the tile payload is untouched), then a derive=1
//      SEF re-shows it. This is the normative §6.17.2 form and needs no
//      --outputinvisible: the referenced frame was never output.
//
// All decodes use `--strict 1 --threads 1`.
//
// Usage:
//   sef_gate [--input clip.y4m] [--frames N] [--dav2d PATH] [--work DIR]

#include "encoder/inter/sef/sef.h"

#include "encoder/m2/m2_frame.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace sef = av2enc::inter::sef;
namespace fs = std::filesystem;

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
  if (plane == 1) {
    return static_cast<std::uint8_t>(in_square ? 100 : 128);
  }
  return static_cast<std::uint8_t>(in_square ? 150 : 128);
}

std::vector<OwnedFrame> make_synthetic_clip(int frames, int width, int height) {
  std::vector<OwnedFrame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    const std::size_t width_s = static_cast<std::size_t>(width);
    const std::size_t cw_s = static_cast<std::size_t>(cw);
    frame.y.resize(width_s * static_cast<std::size_t>(height));
    frame.u.resize(cw_s * static_cast<std::size_t>(ch));
    frame.v.resize(cw_s * static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * width_s +
                static_cast<std::size_t>(x)] = square_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * cw_s +
                                  static_cast<std::size_t>(x);
        frame.u[index] = square_sample(x * 2, y * 2, f, 1);
        frame.v[index] = square_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

bool load_y4m(const std::string& path, int max_frames,
              std::vector<OwnedFrame>& out, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out.size()) < max_frames) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    OwnedFrame owned;
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
    if (fs::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (fs::exists(candidate)) return candidate;
  }
  return {};
}

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out += ch;
    }
  }
  out += "'";
  return out;
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

std::uint32_t read_le32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8U) |
         (static_cast<std::uint32_t>(p[2]) << 16U) |
         (static_cast<std::uint32_t>(p[3]) << 24U);
}

std::uint64_t read_le64(const std::uint8_t* p) {
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8U) | static_cast<std::uint64_t>(p[i]);
  }
  return value;
}

bool read_ivf(const std::string& path,
              std::vector<std::vector<std::uint8_t>>& units,
              std::uint32_t& width, std::uint32_t& height,
              std::string& error) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    error = "cannot open " + path;
    return false;
  }
  std::uint8_t header[32] = {};
  file.read(reinterpret_cast<char*>(header), 32);
  if (file.gcount() != 32) {
    error = "truncated IVF file header";
    return false;
  }
  if (!(header[0] == 'D' && header[1] == 'K' && header[2] == 'I' &&
        header[3] == 'F')) {
    error = "not an IVF file";
    return false;
  }
  width = static_cast<std::uint32_t>(header[12]) |
          (static_cast<std::uint32_t>(header[13]) << 8U);
  height = static_cast<std::uint32_t>(header[14]) |
           (static_cast<std::uint32_t>(header[15]) << 8U);
  while (true) {
    std::uint8_t frame_header[12] = {};
    file.read(reinterpret_cast<char*>(frame_header), 12);
    const std::streamsize got = file.gcount();
    if (got == 0) break;
    if (got != 12) {
      error = "truncated IVF frame header";
      return false;
    }
    const std::uint32_t size = read_le32(frame_header);
    (void)read_le64(frame_header + 4);  // timestamp
    std::vector<std::uint8_t> payload(size);
    file.read(reinterpret_cast<char*>(payload.data()),
              static_cast<std::streamsize>(size));
    if (file.gcount() != static_cast<std::streamsize>(size)) {
      error = "truncated IVF frame payload";
      return false;
    }
    units.push_back(std::move(payload));
  }
  if (units.size() < 2) {
    error = "IVF stream has fewer than 2 frames";
    return false;
  }
  return true;
}

bool compare_frame_owned(const OwnedFrame& got, const OwnedFrame& want,
                         std::string& error) {
  if (got.width != want.width || got.height != want.height) {
    error = "frame dimensions differ";
    return false;
  }
  return got.y == want.y && got.u == want.u && got.v == want.v;
}

struct DecodedSequence {
  av2enc::Y4mHeader header;
  std::vector<OwnedFrame> frames;
};

bool read_decoded(const std::string& path, DecodedSequence& out,
                  std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  out.header = reader.header();
  av2enc::Y4mFrame frame;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    OwnedFrame owned;
    owned.width = frame.width();
    owned.height = frame.height();
    owned.y.assign(frame.y().begin(), frame.y().end());
    owned.u.assign(frame.u().begin(), frame.u().end());
    owned.v.assign(frame.v().begin(), frame.v().end());
    out.frames.push_back(std::move(owned));
  }
  return true;
}

std::string hex(std::span<const std::uint8_t> bytes) {
  std::string out;
  char buffer[4];
  for (const std::uint8_t byte : bytes) {
    std::snprintf(buffer, sizeof(buffer), "%02x", static_cast<unsigned>(byte));
    if (!out.empty()) out.push_back(' ');
    out += buffer;
  }
  return out;
}

bool decode(const std::string& dav2d, const std::string& input,
            const std::string& output, bool output_invisible,
            std::string& error) {
  std::string command = shell_quote(dav2d) + " -q -i " + shell_quote(input) +
                        " -o " + shell_quote(output) +
                        " --threads 1 --strict 1";
  if (output_invisible) command += " --outputinvisible 1";
  std::cout << "  decode: " << command << "\n";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    error = "dav2d returned " + std::to_string(rc);
    return false;
  }
  return true;
}

// Mirrors the M2 reference update order: key frame refreshes allFrames (only
// the first slot stays valid), each P frame refreshes slot 0. Every immediate
// frame is marked shown.
sef::SlotMap model_slot_map(unsigned frame_count, bool hide_last_frame) {
  sef::SlotMap map;
  map.num_ref_frames = 2;
  for (unsigned i = 0; i < frame_count; ++i) {
    const bool hidden = hide_last_frame && i + 1 == frame_count;
    sef::RefreshFrame frame;
    frame.key_frame = i == 0;
    frame.order_hint = i;
    frame.frame_store_id = i;
    frame.immediate_output_frame = !hidden;
    frame.implicit_output_frame = false;
    frame.refresh_frame_flags =
        i == 0 ? ((1U << map.num_ref_frames) - 1U) : 1U;
    sef::apply_refresh(map, frame);
    if (!hidden) sef::mark_shown(map, 0);
  }
  return map;
}

// Gate-only M2 helper. The M2 tile_group header starts with
//   is_first_tile_group f(1)
//   cur_mfh_id uvlc(0) = '1'
//   seq_header_id_in_frame_header uvlc(0) = '1'
//   frame_is_inter f(1)
//   immediate_output_frame f(1)      <- bit index 4
// and monotonic_output_order_flag = 1 makes implicit_output_frame inferred 0,
// so clearing this one bit turns the frame into a coded non-output frame unit
// without touching the tile payload or shifting any later syntax.
bool clear_immediate_output_bit(std::vector<std::uint8_t>& unit,
                                std::string& error) {
  std::size_t pos = 0;
  while (pos < unit.size()) {
    std::uint64_t obu_size = 0;
    unsigned shift = 0;
    while (true) {
      if (pos >= unit.size()) {
        error = "truncated leb128";
        return false;
      }
      const std::uint8_t byte = unit[pos++];
      obu_size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0U) break;
      shift += 7U;
      if (shift >= 35U) {
        error = "bad leb128";
        return false;
      }
    }
    if (obu_size == 0U || obu_size > unit.size() - pos) {
      error = "bad OBU length";
      return false;
    }
    const std::size_t obu_start = pos;
    const std::uint8_t header = unit[obu_start];
    const unsigned type = (header >> 2U) & 0x1FU;
    const bool extension = (header & 0x80U) != 0U;
    const std::size_t header_bytes = extension ? 2U : 1U;
    const std::size_t payload_start = obu_start + header_bytes;
    const std::size_t obu_end = obu_start + static_cast<std::size_t>(obu_size);
    if (type == 7U) {  // OBU_REGULAR_TILE_GROUP
      if (payload_start >= obu_end) {
        error = "empty tile group";
        return false;
      }
      std::uint8_t& first = unit[payload_start];
      if ((first & 0x08U) == 0U) {
        error = "immediate_output_frame is already 0";
        return false;
      }
      first = static_cast<std::uint8_t>(first & ~0x08U);
      return true;
    }
    pos = obu_end;
  }
  error = "no OBU_REGULAR_TILE_GROUP found";
  return false;
}

struct PhaseResult {
  bool ok = false;
  std::string detail;
};

// Decodes `path` and checks the exact expected output sequence.
bool check_decode(const std::string& dav2d, const std::string& path,
                  const std::vector<OwnedFrame>& expected,
                  bool output_invisible, const std::string& decoded_path,
                  std::string& error) {
  if (!decode(dav2d, path, decoded_path, output_invisible, error)) {
    return false;
  }
  DecodedSequence decoded;
  if (!read_decoded(decoded_path, decoded, error)) return false;
  if (decoded.frames.size() != expected.size()) {
    error = "decoder produced " + std::to_string(decoded.frames.size()) +
            " frame(s), expected " + std::to_string(expected.size());
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!compare_frame_owned(decoded.frames[i], expected[i], error)) {
      error = "output frame " + std::to_string(i) + ": " + error;
      return false;
    }
  }
  return true;
}

// Shared phases A/B/C. `units_in` is the verified M2 stream in coding order;
// `recon` is non-null for streams encoded in this process.
bool run_phases(const std::string& name,
                const std::vector<std::vector<std::uint8_t>>& units_in,
                std::uint32_t width, std::uint32_t height,
                const std::vector<av2enc::m2::M2ReconFrame>* recon,
                const std::string& work_dir, const std::string& dav2d) {
  const std::size_t frame_count = units_in.size();
  std::cout << "== " << name << ": " << frame_count << " frame(s) " << width
            << "x" << height << " ==\n";
  std::string error;

  const std::string base_path = work_dir + "/base.ivf";
  if (!write_ivf(base_path, units_in, width, height, error)) {
    std::cerr << "IVF WRITE FAIL: " << error << "\n";
    return false;
  }
  const std::string base_y4m = work_dir + "/base.y4m";
  if (!decode(dav2d, base_path, base_y4m, false, error)) {
    std::cerr << "BASELINE DECODE FAIL: " << error << "\n";
    return false;
  }
  DecodedSequence baseline;
  if (!read_decoded(base_y4m, baseline, error)) {
    std::cerr << "BASELINE READ FAIL: " << error << "\n";
    return false;
  }
  if (baseline.frames.size() != frame_count) {
    std::cerr << "BASELINE COUNT FAIL: " << baseline.frames.size()
              << " output(s), expected " << frame_count << "\n";
    return false;
  }
  if (recon != nullptr) {
    if (recon->size() != baseline.frames.size()) {
      std::cerr << "BASELINE RECON COUNT FAIL\n";
      return false;
    }
    for (std::size_t i = 0; i < recon->size(); ++i) {
      OwnedFrame expected;
      expected.width = (*recon)[i].width;
      expected.height = (*recon)[i].height;
      expected.y = (*recon)[i].y;
      expected.u = (*recon)[i].u;
      expected.v = (*recon)[i].v;
      if (!compare_frame_owned(baseline.frames[i], expected, error)) {
        std::cerr << "BASELINE RECON MISMATCH at " << i << ": " << error
                  << "\n";
        return false;
      }
    }
    std::cout << "  baseline: " << baseline.frames.size()
              << " frame(s), recon exact\n";
  } else {
    std::cout << "  baseline: " << baseline.frames.size()
              << " frame(s), decoded\n";
  }

  const sef::SefParams params;  // M2: 2 refs, 4 order-hint bits.
  const std::uint32_t last_hint = static_cast<std::uint32_t>(frame_count - 1);

  // --- A: pure append, explicit hint -------------------------------------
  {
    sef::SlotMap map = model_slot_map(static_cast<unsigned>(frame_count),
                                      /*hide_last_frame=*/false);
    sef::SefHeader header;
    sef::SefError sef_error = sef::SefError::kOk;
    if (!sef::make_show_header(map, params, 0, sef::SefKind::kRegular, header,
                               &sef_error)) {
      std::cerr << "SEF HEADER FAIL: " << sef::sef_error_name(sef_error)
                << "\n";
      return false;
    }
    if (!sef::validate_sef_reference(map, params, header, &sef_error) ||
        !sef::validate_dav2d_compat(map, params, header, &sef_error)) {
      std::cerr << "SEF VALIDATE FAIL: " << sef::sef_error_name(sef_error)
                << "\n";
      return false;
    }
    const std::uint32_t decoded_hint = sef::disp_order_hint(
        header.sef_order_hint, map.max_disp_order_hint(),
        params.order_hint_bits);
    if (header.derive_sef_order_hint || decoded_hint != last_hint) {
      std::cerr << "SEF HEADER UNEXPECTED: derive="
                << header.derive_sef_order_hint
                << " hint=" << header.sef_order_hint
                << " decoded=" << decoded_hint << "\n";
      return false;
    }
    const std::vector<std::uint8_t> obu = sef::make_sef_obu(params, header);
    std::cout << "  A: " << sef::sef_kind_name(header.kind)
              << " idx=" << static_cast<unsigned>(header.frame_to_show_map_idx)
              << " derive=0 hint=" << header.sef_order_hint << " obu=["
              << hex(obu) << "]\n";

    std::vector<std::vector<std::uint8_t>> units = units_in;
    units.push_back(sef::make_sef_temporal_unit(params, header));
    const std::string path = work_dir + "/a_append.ivf";
    if (!write_ivf(path, units, width, height, error)) {
      std::cerr << "IVF WRITE FAIL: " << error << "\n";
      return false;
    }

    // Pinned dav2d drops the duplicate POC by default.
    std::vector<OwnedFrame> default_expected(baseline.frames);
    if (!check_decode(dav2d, path, default_expected, /*output_invisible=*/false,
                      work_dir + "/a_default.y4m", error)) {
      std::cerr << "A DEFAULT FAIL: " << error << "\n";
      return false;
    }
    std::cout << "  A default: duplicate POC dropped (outputs="
              << default_expected.size() << ")\n";

    // With --outputinvisible the extra output is emitted and must be exact.
    std::vector<OwnedFrame> sef_expected(baseline.frames);
    sef_expected.push_back(baseline.frames.back());
    if (!check_decode(dav2d, path, sef_expected, /*output_invisible=*/true,
                      work_dir + "/a_invisible.y4m", error)) {
      std::cerr << "A OUTPUT FAIL: " << error << "\n";
      return false;
    }
    std::cout << "  A pass: extra output == baseline frame "
              << (frame_count - 1) << "\n";
  }

  // --- B: two SEFs (regular + leading) -----------------------------------
  {
    sef::SlotMap map = model_slot_map(static_cast<unsigned>(frame_count),
                                      /*hide_last_frame=*/false);
    sef::SefHeader regular;
    sef::SefHeader leading;
    sef::SefError sef_error = sef::SefError::kOk;
    if (!sef::make_show_header(map, params, 0, sef::SefKind::kRegular, regular,
                               &sef_error) ||
        !sef::make_show_header(map, params, 0, sef::SefKind::kLeading, leading,
                               &sef_error)) {
      std::cerr << "B HEADER FAIL: " << sef::sef_error_name(sef_error) << "\n";
      return false;
    }
    std::vector<std::vector<std::uint8_t>> units = units_in;
    units.push_back(sef::make_sef_temporal_unit(params, regular));
    units.push_back(sef::make_sef_temporal_unit(params, leading));
    const std::string path = work_dir + "/b_two.ivf";
    if (!write_ivf(path, units, width, height, error)) {
      std::cerr << "IVF WRITE FAIL: " << error << "\n";
      return false;
    }
    std::vector<OwnedFrame> expected(baseline.frames);
    expected.push_back(baseline.frames.back());
    expected.push_back(baseline.frames.back());
    if (!check_decode(dav2d, path, expected, /*output_invisible=*/true,
                      work_dir + "/b.y4m", error)) {
      std::cerr << "B FAIL: " << error << "\n";
      return false;
    }
    std::cout << "  B pass: " << sef::sef_kind_name(regular.kind) << " + "
              << sef::sef_kind_name(leading.kind)
              << " => both extras == baseline frame "
              << (frame_count - 1) << "\n";
  }

  // --- C: hidden reference, derive=1 -------------------------------------
  {
    std::vector<std::vector<std::uint8_t>> units = units_in;
    if (!clear_immediate_output_bit(units.back(), error)) {
      std::cerr << "C HIDDEN PATCH FAIL: " << error << "\n";
      return false;
    }
    sef::SlotMap map = model_slot_map(static_cast<unsigned>(frame_count),
                                      /*hide_last_frame=*/true);
    sef::SefHeader header;
    sef::SefError sef_error = sef::SefError::kOk;
    if (!sef::make_show_header(map, params, 0, sef::SefKind::kRegular, header,
                               &sef_error)) {
      std::cerr << "C HEADER FAIL: " << sef::sef_error_name(sef_error) << "\n";
      return false;
    }
    if (!header.derive_sef_order_hint) {
      std::cerr << "C HEADER FAIL: expected derive=1\n";
      return false;
    }
    if (!sef::validate_sef_reference(map, params, header, &sef_error)) {
      std::cerr << "C VALIDATE FAIL: " << sef::sef_error_name(sef_error)
                << "\n";
      return false;
    }
    units.push_back(sef::make_sef_temporal_unit(params, header));
    const std::string path = work_dir + "/c_hidden.ivf";
    if (!write_ivf(path, units, width, height, error)) {
      std::cerr << "IVF WRITE FAIL: " << error << "\n";
      return false;
    }
    std::vector<OwnedFrame> expected;
    for (std::size_t i = 0; i + 1 < baseline.frames.size(); ++i) {
      expected.push_back(baseline.frames[i]);
    }
    expected.push_back(baseline.frames.back());
    if (!check_decode(dav2d, path, expected, /*output_invisible=*/false,
                      work_dir + "/c.y4m", error)) {
      std::cerr << "C FAIL: " << error << "\n";
      return false;
    }
    std::cout << "  C pass: hidden frame + derive=1 SEF => output "
              << (expected.size() - 1) << " == baseline frame "
              << (baseline.frames.size() - 1) << " (no --outputinvisible)\n";
  }

  return true;
}

bool run_encode_phase(const std::string& name,
                      const std::vector<OwnedFrame>& source,
                      const std::string& work_dir, const std::string& dav2d) {
  std::vector<av2enc::m2::M2InputFrame> inputs;
  inputs.reserve(source.size());
  for (const OwnedFrame& frame : source) {
    av2enc::m2::M2InputFrame input;
    input.width = frame.width;
    input.height = frame.height;
    input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input);
  }

  av2enc::m2::M2SequenceResult sequence;
  std::string error;
  av2enc::m2::M2Options options;
  if (!av2enc::m2::encode_m2_sequence(inputs, options, sequence, error)) {
    std::cerr << "ENCODE FAIL: " << error << "\n";
    return false;
  }
  return run_phases(name, sequence.units, source[0].width, source[0].height,
                    &sequence.recon, work_dir, dav2d);
}

bool run_ivf_phase(const std::string& path, const std::string& work_dir,
                   const std::string& dav2d) {
  std::vector<std::vector<std::uint8_t>> units;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::string error;
  if (!read_ivf(path, units, width, height, error)) {
    std::cerr << "IVF READ FAIL (" << path << "): " << error << "\n";
    return false;
  }
  if (units.size() < 2 || units.size() > 3) {
    std::cerr << "expected a 2- or 3-frame verified M2 stream, got "
              << units.size() << " frame(s)\n";
    return false;
  }
  return run_phases("existing IVF " + path, units, width, height, nullptr,
                    work_dir, dav2d);
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string input_ivf;
  int frames = 3;
  std::string dav2d;
  std::string work_dir;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--input-ivf" && i + 1 < argc) {
      input_ivf = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    } else if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--work" && i + 1 < argc) {
      work_dir = argv[++i];
    } else if (arg == "--help") {
      std::cout << "usage: sef_gate [--input clip.y4m | --input-ivf m2.ivf] "
                   "[--frames N] [--dav2d PATH] [--work DIR]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (frames < 2) frames = 2;

  if (dav2d.empty()) dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "no dav2d binary found (set DAV2D)\n";
    return 2;
  }

  if (work_dir.empty()) {
    std::error_code code;
    const fs::path root = fs::temp_directory_path(code);
    work_dir = (root / "sef_gate").string();
  }
  std::error_code code;
  fs::create_directories(work_dir, code);
  if (code) {
    std::cerr << "cannot create work dir " << work_dir << ": " << code.message()
              << "\n";
    return 2;
  }

  if (!input_ivf.empty()) {
    const bool ok = run_ivf_phase(input_ivf, work_dir, dav2d);
    std::cout << (ok ? "SEF GATE PASS" : "SEF GATE FAIL") << "\n";
    return ok ? 0 : 1;
  }

  std::vector<OwnedFrame> source;
  if (!input.empty()) {
    std::string error;
    if (!load_y4m(input, frames, source, error)) {
      std::cerr << "Y4M LOAD FAIL: " << error << "\n";
      return 2;
    }
  } else {
    source = make_synthetic_clip(frames, 64, 64);
  }
  if (source.size() < 2) {
    std::cerr << "need at least 2 frames\n";
    return 2;
  }

  std::cout << "dav2d: " << dav2d << "\n";
  std::cout << "work:  " << work_dir << "\n";

  bool ok = true;
  {
    std::vector<OwnedFrame> two(source.begin(), source.begin() + 2);
    ok = run_encode_phase("two-frame M2 + SEF", two, work_dir, dav2d) &&
         ok;
  }
  {
    std::vector<OwnedFrame> three(
        source.begin(), source.begin() + static_cast<std::ptrdiff_t>(
                                 std::min<std::size_t>(3, source.size())));
    if (three.size() >= 3) {
      ok = run_encode_phase("three-frame M2 + SEF", three, work_dir, dav2d) &&
           ok;
    }
  }

  std::cout << (ok ? "SEF GATE PASS" : "SEF GATE FAIL") << "\n";
  return ok ? 0 : 1;
}
