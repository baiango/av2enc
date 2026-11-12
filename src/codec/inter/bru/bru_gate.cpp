// BRU decoder gate: builds a three-frame stream whose second inter frame has a
// mixed BRU_ACTIVE / BRU_SUPPORT / BRU_INACTIVE superblock map, decodes it with
// the pinned `dav2d --strict 1 --threads 1`, and compares every output sample
// against the expected mix of two M2 reference reconstructions.
//
// Frame layout (256x128, 4x2 superblocks, 2x1 uniform tiles):
//
//   f0  CLK A                       refresh all slots (A)
//   f1  P   B (M2 payload, slot 0)  refresh slot 1 (B)
//   f2  BRU bru_ref=slot 0          tile 0 active:
//                                     (0,0) ACTIVE  -> reference index 1 = B
//                                     (0,1),(1,0),(1,1) SUPPORT -> A
//                                   tile 1 inactive -> A
//
// The active rectangle touches the top/left screen edge and is bordered by
// BRU_SUPPORT on the right/bottom, so bru_region_valid() (R10) must accept it.
// The pinned dav2d build does not parse enable_bru/use_bru yet, so the gate
// first tries dav2d --strict 1 --threads 1 and falls back to avmdec for the
// BRU stream.  The enable_bru=0 control always runs on dav2d.
#include "codec/inter/bru/bru.h"

#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::inter::bru::BruMode;
using av2enc::inter::bru::FrameConfig;
using av2enc::inter::bru::ParsedStream;
using av2enc::inter::bru::SequenceConfig;
using av2enc::inter::bru::TilePlan;

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
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t sample_value(int x, int y, int frame, int plane) {
  const int sx = 24 + 12 * frame;
  const int sy = 16 + 8 * frame;
  const bool in_square = x >= sx && x < sx + 72 && y >= sy && y < sy + 48;
  if (plane == 0) {
    if (in_square) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(32 + ((lx * 29 + ly * 13) & 0x7F));
    }
    return static_cast<std::uint8_t>(100 + ((x + y) >> 3));
  }
  if (plane == 1) return static_cast<std::uint8_t>(in_square ? 96 : 128);
  return static_cast<std::uint8_t>(in_square ? 160 : 128);
}

std::vector<Frame> make_clip(int frames, int width, int height) {
  std::vector<Frame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    Frame frame;
    frame.width = width;
    frame.height = height;
    frame.y.resize(static_cast<std::size_t>(width) * height);
    frame.u.resize(static_cast<std::size_t>(cw) * ch);
    frame.v.resize(static_cast<std::size_t>(cw) * ch);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * width + x] =
            sample_value(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * cw + x] =
            sample_value(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * cw + x] =
            sample_value(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

std::vector<std::vector<std::uint8_t>> encode_references(
    const std::vector<Frame>& frames, std::vector<av2enc::video::VideoReconFrame>& recon,
    std::string& error) {
  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(frames.size());
  for (const Frame& frame : frames) {
    av2enc::video::VideoInputFrame input;
    input.width = static_cast<std::uint32_t>(frame.width);
    input.height = static_cast<std::uint32_t>(frame.height);
    input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
    input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
    input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
    inputs.push_back(input);
  }
  av2enc::video::VideoOptions options;
  options.base_q_idx = 90;
  options.enable_inter_residual = true;
  av2enc::video::VideoSequenceResult result;
  if (!av2enc::video::encode_video_sequence(inputs, options, result, error)) {
    return {};
  }
  recon = std::move(result.recon);
  std::vector<std::vector<std::uint8_t>> tiles;
  tiles.reserve(result.units.size());
  for (std::size_t i = 0; i < result.units.size(); ++i) {
    const std::size_t count = result.tile_bytes[i];
    if (count == 0 || count > result.units[i].size()) {
      error = "M2 unit smaller than its tile payload";
      return {};
    }
    tiles.emplace_back(result.units[i].end() - static_cast<std::ptrdiff_t>(count),
                       result.units[i].end());
  }
  return tiles;
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
               const std::vector<std::vector<std::uint8_t>>& units, int width,
               int height, std::string& error) {
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

bool read_y4m(const std::string& path, std::vector<Frame>& out,
              std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    Frame copy;
    copy.width = static_cast<int>(frame.width());
    copy.height = static_cast<int>(frame.height());
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    out.push_back(std::move(copy));
  }
  return true;
}

Frame mix_references(const av2enc::video::VideoReconFrame& a,
                     const av2enc::video::VideoReconFrame& b, int sb_cols,
                     std::span<const BruMode> modes) {
  Frame out;
  out.width = static_cast<int>(a.width);
  out.height = static_cast<int>(a.height);
  out.y.resize(a.y.size());
  out.u.resize(a.u.size());
  out.v.resize(a.v.size());
  const auto mix_plane = [&](const std::vector<std::uint8_t>& pa,
                             const std::vector<std::uint8_t>& pb,
                             std::vector<std::uint8_t>& dst, int stride,
                             int sb_shift) {
    for (int y = 0; y < static_cast<int>(dst.size() / static_cast<std::size_t>(stride));
         ++y) {
      for (int x = 0; x < stride; ++x) {
        const int sb_col = x >> sb_shift;
        const int sb_row = y >> sb_shift;
        const BruMode mode =
            modes[static_cast<std::size_t>(sb_row) * sb_cols +
                  static_cast<std::size_t>(sb_col)];
        dst[static_cast<std::size_t>(y) * stride + x] =
            (mode == BruMode::kActive) ? pa[static_cast<std::size_t>(y) * stride + x]
                                       : pb[static_cast<std::size_t>(y) * stride + x];
      }
    }
  };
  const int width = out.width;
  const int height = out.height;
  mix_plane(a.y, b.y, out.y, width, 6);
  mix_plane(a.u, b.u, out.u, (width + 1) / 2, 5);
  mix_plane(a.v, b.v, out.v, (width + 1) / 2, 5);
  (void)height;
  return out;
}

bool compare_frame(const Frame& decoded, const Frame& expected,
                   const std::string& name) {
  if (decoded.width != expected.width || decoded.height != expected.height) {
    check(false, name + ": dimensions");
    return false;
  }
  const auto compare_plane = [&](std::span<const std::uint8_t> got,
                                 const std::vector<std::uint8_t>& want,
                                 const char* plane) {
    if (got.size() != want.size()) {
      check(false, name + ": " + plane + " size");
      return false;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] != want[i]) {
        check(false, name + ": " + plane + " sample " + std::to_string(i));
        return false;
      }
    }
    return true;
  };
  bool ok = compare_plane(decoded.y, expected.y, "Y");
  ok = compare_plane(decoded.u, expected.u, "U") && ok;
  ok = compare_plane(decoded.v, expected.v, "V") && ok;
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_dir = "build-bru/gate";
  std::string dav2d;
  std::string avmdec;
  bool control = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--avmdec" && i + 1 < argc) {
      avmdec = argv[++i];
    } else if (arg == "--control") {
      control = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (dav2d.empty()) dav2d = find_dav2d();
  if (avmdec.empty()) {
    for (const char* candidate : {"tools/bin/avm-macos/avmdec",
                                  "tools/bin/avmdec", "avmdec"}) {
      if (std::filesystem::exists(candidate)) {
        avmdec = candidate;
        break;
      }
    }
  }
  if (dav2d.empty()) {
    std::fprintf(stderr, "no dav2d found; set DAV2D=/path/to/dav2d\n");
    return 77;
  }
  std::filesystem::create_directories(out_dir);

  constexpr int kWidth = 256;
  constexpr int kHeight = 128;
  const std::vector<Frame> clip = make_clip(2, kWidth, kHeight);
  std::string error;
  std::vector<av2enc::video::VideoReconFrame> recon;
  const auto reference_tiles = encode_references(clip, recon, error);
  if (reference_tiles.size() != 2 || recon.size() != 2) {
    std::fprintf(stderr, "M2 reference encode failed: %s\n", error.c_str());
    return 1;
  }

  SequenceConfig sequence;
  sequence.max_frame_width = kWidth;
  sequence.max_frame_height = kHeight;
  sequence.enable_bru = !control;
  sequence.order_hint_bits = 4;
  sequence.num_ref_frames = 2;
  sequence.base_q_idx = 90;

  const auto seq = av2enc::inter::bru::make_sequence_header_obu(sequence);
  const auto td = av2enc::video::make_temporal_delimiter_obu();

  FrameConfig key;
  key.keyframe = true;
  key.order_hint = 0;
  key.refresh_frame_flags = 3;
  TilePlan key_plan;
  key_plan.bru_tile_active = {1};
  key_plan.sb_modes.assign(
      static_cast<std::size_t>(av2enc::inter::bru::sb_cols_for(kWidth) *
                               av2enc::inter::bru::sb_rows_for(kHeight)),
      BruMode::kActive);
  std::vector<std::vector<std::uint8_t>> key_tiles{reference_tiles[0]};
  const auto key_obu = av2enc::inter::bru::make_tile_group_obu(
      sequence, key, key_plan, key_tiles, error);
  check(!key_obu.empty(), "key OBU");

  FrameConfig pframe = key;
  pframe.keyframe = false;
  pframe.order_hint = 1;
  pframe.refresh_frame_flags = control ? 1 : 2;  // slot 1 (control mirrors M2)
  pframe.num_total_refs = 1;
  pframe.ref_frame_idx[0] = 0;
  pframe.use_bru = false;
  pframe.tile_cols_log2 = 0;
  pframe.tile_rows_log2 = 0;
  std::vector<std::vector<std::uint8_t>> p_tiles{reference_tiles[1]};
  const auto p_obu = av2enc::inter::bru::make_tile_group_obu(
      sequence, pframe, key_plan, p_tiles, error);
  check(!p_obu.empty(), "P OBU");

  if (control) {
    // Byte-identity control: the same logical key/P stream must be exactly
    // what the untouched M2 writer produces with enable_bru = 0.
    av2enc::video::VideoConfig video_config;
    video_config.max_frame_width = kWidth;
    video_config.max_frame_height = kHeight;
    video_config.order_hint_bits = sequence.order_hint_bits;
    video_config.num_ref_frames = sequence.num_ref_frames;
    video_config.base_q_idx = sequence.base_q_idx;
    av2enc::video::VideoFrameHeader video_key;
    video_key.keyframe = true;
    video_key.order_hint = 0;
    video_key.refresh_frame_flags = 3;
    av2enc::video::VideoFrameHeader video_p;
    video_p.keyframe = false;
    video_p.order_hint = 1;
    video_p.refresh_frame_flags = 1;
    video_p.ref_slot = 0;
    check(seq == av2enc::video::make_sequence_header_obu(video_config),
          "enable_bru=0 sequence byte identity");
    check(key_obu == av2enc::video::make_frame_obu(video_config, video_key,
                                                reference_tiles[0]),
          "enable_bru=0 key frame byte identity");
    check(p_obu == av2enc::video::make_frame_obu(video_config, video_p,
                                              reference_tiles[1]),
          "enable_bru=0 P frame byte identity");

    std::vector<std::vector<std::uint8_t>> units;
    units.push_back(td);
    units.back().insert(units.back().end(), seq.begin(), seq.end());
    units.back().insert(units.back().end(), key_obu.begin(), key_obu.end());
    units.push_back(td);
    units.back().insert(units.back().end(), p_obu.begin(), p_obu.end());
    const std::string ivf_path = out_dir + "/bru_control.ivf";
    check(write_ivf(ivf_path, units, kWidth, kHeight, error), "write control IVF");
    const std::string decoded_path = ivf_path + ".decoded.y4m";
    const std::string command = "\"" + dav2d + "\" -q -i \"" + ivf_path +
                                "\" -o \"" + decoded_path +
                                "\" --strict 1 --threads 1";
    const int rc = std::system(command.c_str());
    check(rc == 0, "enable_bru=0 control decodes");
    std::vector<Frame> decoded;
    if (!read_y4m(decoded_path, decoded, error)) {
      std::fprintf(stderr, "control Y4M read failed: %s\n", error.c_str());
      return 1;
    }
    check(decoded.size() == 2, "control produces two frames");
    if (decoded.size() == 2) {
      const Frame ref_a{static_cast<int>(recon[0].width),
                        static_cast<int>(recon[0].height), recon[0].y,
                        recon[0].u, recon[0].v};
      const Frame ref_b{static_cast<int>(recon[1].width),
                        static_cast<int>(recon[1].height), recon[1].y,
                        recon[1].u, recon[1].v};
      compare_frame(decoded[0], ref_a, "control f0 CLK");
      compare_frame(decoded[1], ref_b, "control f1 P");
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
  }

  // Mixed BRU frame: 2x1 tiles, tile 0 covers SBs (0,0)..(1,1).
  TilePlan bru_plan;
  const int sb_cols = av2enc::inter::bru::sb_cols_for(kWidth);
  const int sb_rows = av2enc::inter::bru::sb_rows_for(kHeight);
  bru_plan.sb_modes.assign(static_cast<std::size_t>(sb_cols * sb_rows),
                           BruMode::kInactive);
  bru_plan.sb_modes[0] = BruMode::kActive;
  bru_plan.sb_modes[1] = BruMode::kSupport;
  bru_plan.sb_modes[sb_cols] = BruMode::kSupport;
  bru_plan.sb_modes[sb_cols + 1] = BruMode::kSupport;
  bru_plan.bru_tile_active = {1, 0};

  std::vector<std::vector<std::uint8_t>> bru_tiles(2);
  const std::vector<BruMode> tile0_modes = {
      bru_plan.sb_modes[0], bru_plan.sb_modes[1], bru_plan.sb_modes[sb_cols],
      bru_plan.sb_modes[sb_cols + 1]};
  check(av2enc::inter::bru::encode_tile_payload(tile0_modes, 2, 2, 1, true,
                                                bru_tiles[0], error),
        "encode active tile");
  check(av2enc::inter::bru::encode_tile_payload(
            std::span<const BruMode>(), 2, 2, 1, false, bru_tiles[1], error),
        "encode inactive tile");

  FrameConfig bru_frame = pframe;
  bru_frame.order_hint = 2;
  bru_frame.refresh_frame_flags = 1;  // refresh bru_ref slot (slot 0)
  bru_frame.num_total_refs = 2;
  bru_frame.ref_frame_idx[0] = 0;
  bru_frame.ref_frame_idx[1] = 1;
  bru_frame.use_bru = true;
  bru_frame.bru_ref = 0;
  bru_frame.bru_inactive = false;
  bru_frame.tile_cols_log2 = 1;
  bru_frame.tile_rows_log2 = 0;
  bru_frame.tile_size_bytes_minus_1 = 0;
  const auto bru_obu = av2enc::inter::bru::make_tile_group_obu(
      sequence, bru_frame, bru_plan, bru_tiles, error);
  check(!bru_obu.empty(), "BRU OBU");

  std::vector<std::vector<std::uint8_t>> units;
  units.push_back(td);
  units.back().insert(units.back().end(), seq.begin(), seq.end());
  units.back().insert(units.back().end(), key_obu.begin(), key_obu.end());
  units.push_back(td);
  units.back().insert(units.back().end(), p_obu.begin(), p_obu.end());
  units.push_back(td);
  units.back().insert(units.back().end(), bru_obu.begin(), bru_obu.end());

  const std::string ivf_path = out_dir + "/bru_gate.ivf";
  check(write_ivf(ivf_path, units, kWidth, kHeight, error), "write IVF");

  std::vector<std::uint8_t> concatenated;
  for (const auto& unit : units) {
    concatenated.insert(concatenated.end(), unit.begin(), unit.end());
  }
  ParsedStream parsed;
  check(av2enc::inter::bru::parse_stream(concatenated, parsed, error),
        "independent reader parses the gate stream");
  check(parsed.frames.size() == 3, "reader sees three frames");
  if (parsed.frames.size() == 3) {
    check(parsed.sequence.enable_bru, "sequence enable_bru");
    check(parsed.frames[1].header.use_bru == false, "f1 use_bru = 0");
    check(parsed.frames[2].header.use_bru, "f2 use_bru = 1");
    check(parsed.frames[2].header.bru_ref == 0, "f2 bru_ref = 0");
    check(parsed.frames[2].header.bru_inactive == false, "f2 active frame");
    check(parsed.frames[2].bru_tile_active ==
              std::vector<std::uint8_t>({1, 0}),
          "f2 bru_tile_active map");
    check(parsed.frames[2].sb_modes == bru_plan.sb_modes,
          "f2 BruModes map reconstructed exactly");
    check(parsed.frames[2].region_valid, "f2 passes bru_region_valid (R10)");
  }

  const std::string decoded_path = ivf_path + ".decoded.y4m";
  const std::string dav2d_log = ivf_path + ".dav2d.log";
  const std::string dav2d_command =
      "\"" + dav2d + "\" -q -i \"" + ivf_path + "\" -o \"" +
      decoded_path + "\" --strict 1 --threads 1 > \"" + dav2d_log +
      "\" 2>&1";
  std::printf("decode (dav2d, BRU-aware builds): %s\n",
              dav2d_command.c_str());
  int rc = std::system(dav2d_command.c_str());
  std::vector<Frame> decoded;
  bool dav2d_ok = (rc == 0);
  if (dav2d_ok) {
    dav2d_ok = read_y4m(decoded_path, decoded, error) && decoded.size() == 3;
  }
  if (!dav2d_ok) {
    // The pinned dav2d f9dafbc does not parse enable_bru/use_bru yet.  The
    // AVM decoder in tools/bin/avm-macos is the black-box validator that does
    // implement BRU; use it for the BRU syntax gate and record the fallback.
    if (avmdec.empty()) {
      check(false, "dav2d rejects BRU and no avmdec fallback was found");
      std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
      return 1;
    }
    std::printf("note: pinned dav2d lacks BRU syntax; falling back to %s\n",
                avmdec.c_str());
    std::filesystem::remove(decoded_path);
    const std::string avm_command = "\"" + avmdec +
        "\" --codec=av2 -o \"" + decoded_path + "\" -t 1 \"" + ivf_path +
        "\"";
    std::printf("decode (avmdec): %s\n", avm_command.c_str());
    rc = std::system(avm_command.c_str());
    decoded.clear();
    if (!read_y4m(decoded_path, decoded, error)) {
      std::fprintf(stderr, "decoded Y4M read failed: %s\n", error.c_str());
      return 1;
    }
  }
  check(rc == 0, "BRU decoder exit status");

  constexpr int want_frames = 3;
  check(static_cast<int>(decoded.size()) == want_frames,
        "decoder produced the expected frames");
  if (static_cast<int>(decoded.size()) == want_frames) {
    const Frame ref_a{static_cast<int>(recon[0].width),
                      static_cast<int>(recon[0].height), recon[0].y,
                      recon[0].u, recon[0].v};
    const Frame ref_b{static_cast<int>(recon[1].width),
                      static_cast<int>(recon[1].height), recon[1].y,
                      recon[1].u, recon[1].v};
    // Active blocks reference ref index 1 (slot 1 = B); inactive/support
    // blocks use bru_mode_info() and copy bru_ref = slot 0 (A).
    const Frame mixed = mix_references(recon[1], recon[0], sb_cols,
                                       bru_plan.sb_modes);
    compare_frame(decoded[0], ref_a, "f0 CLK");
    compare_frame(decoded[1], ref_b, "f1 P");
    compare_frame(decoded[2], mixed, "f2 active/support/inactive mix");
  }

  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
