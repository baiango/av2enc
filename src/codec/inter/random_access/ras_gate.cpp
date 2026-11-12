// Decoder gate for the AV2 random-access switch (SWITCH) and RAS frames.
//
// Builds a three-frame sequence
//
//   frame 0  OBU_CLOSED_LOOP_KEY   (long_term_id_plus_1 = 1)
//   frame 1  OBU_SWITCH            (restricted_prediction_switch = 1)
//   frame 2  OBU_RAS_FRAME         (ref_long_term_id = {0})
//
// by splicing the M2 CPU P-frame tile payloads into the new frame headers
// written by this component. The decoder output is compared sample-by-sample
// with the encoder reconstruction. The pinned validator is invoked as
//
//   dav2d --strict 1 --threads 1 -q -i stream.ivf -o decoded.y4m
//
// Usage:
//   ras_gate [--input clip.y4m] [--frames 3] [--out out.ivf] [--trace]
//
// Without --input the gate synthesises a 128x128 moving texture clip, so it
// runs without test media.

#include "codec/inter/random_access/random_access.h"

#include "core/bitstream/bitwriter.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace ra = av2enc::inter::random_access;

namespace {

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

// Smooth moving texture: good integer-MV inter prediction, unlike flat
// synthetic blocks (which M2 codes as intra).
std::uint8_t texture_sample(int x, int y, int frame) {
  const int xx = x + frame * 4;
  const int yy = y + frame * 2;
  const double value = 128.0 + 50.0 * std::sin(xx * 0.39) +
                       30.0 * std::sin(yy * 0.23) +
                       15.0 * std::sin((xx + yy) * 0.71);
  return static_cast<std::uint8_t>(std::clamp(static_cast<int>(value), 0, 255));
}

std::vector<OwnedFrame> make_synthetic_clip(int frames, int width, int height) {
  std::vector<OwnedFrame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  const std::size_t luma_stride = static_cast<std::size_t>(width);
  const std::size_t chroma_stride = static_cast<std::size_t>(cw);
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(luma_stride * static_cast<std::size_t>(height));
    frame.u.resize(chroma_stride * static_cast<std::size_t>(ch));
    frame.v.resize(chroma_stride * static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * luma_stride +
                static_cast<std::size_t>(x)] = texture_sample(x, y, f);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * chroma_stride +
                static_cast<std::size_t>(x)] = 128;
        frame.v[static_cast<std::size_t>(y) * chroma_stride +
                static_cast<std::size_t>(x)] = 128;
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
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

// Finds the first OBU of the requested raw obu_type in an Annex B unit.
bool find_obu(std::span<const std::uint8_t> unit, std::uint8_t wanted,
              ra::ObuView& out, std::string& error) {
  std::size_t offset = 0;
  while (offset < unit.size()) {
    ra::ObuView view;
    std::size_t consumed = 0;
    if (!ra::parse_obu(unit.subspan(offset), view, consumed, error)) {
      return false;
    }
    if (consumed == 0) {
      error = "zero-length OBU";
      return false;
    }
    if (view.obu_type == wanted) {
      out = view;
      return true;
    }
    offset += consumed;
  }
  error = "requested OBU type " + std::to_string(wanted) + " not found";
  return false;
}

bool extract_video_tile(std::span<const std::uint8_t> unit, std::uint8_t obu_type,
                     const av2enc::video::VideoConfig& config,
                     const av2enc::video::VideoFrameHeader& header,
                     std::span<const std::uint8_t>& tile, std::string& error) {
  ra::ObuView view;
  if (!find_obu(unit, obu_type, view, error)) return false;
  av2enc::BitWriter probe;
  av2enc::video::write_tile_group_header(probe, config, header);
  const std::size_t header_bytes = probe.byte_size();
  if (view.payload.size() < header_bytes) {
    error = "M2 frame OBU is shorter than its tile-group header";
    return false;
  }
  tile = view.payload.subspan(header_bytes);
  if (tile.empty()) {
    error = "M2 frame OBU has an empty tile payload";
    return false;
  }
  return true;
}

av2enc::video::VideoInputFrame to_input(const OwnedFrame& frame) {
  av2enc::video::VideoInputFrame input;
  input.width = frame.width;
  input.height = frame.height;
  input.y = std::span<const std::uint8_t>(frame.y.data(), frame.y.size());
  input.u = std::span<const std::uint8_t>(frame.u.data(), frame.u.size());
  input.v = std::span<const std::uint8_t>(frame.v.data(), frame.v.size());
  return input;
}

void append(std::vector<std::uint8_t>& out,
            const std::vector<std::uint8_t>& bytes) {
  out.insert(out.end(), bytes.begin(), bytes.end());
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

struct CompareResult {
  bool ok = true;
  std::uint64_t mismatches = 0;
  std::uint32_t frames = 0;
};

CompareResult compare_y4m(const std::string& path,
                          const std::vector<av2enc::video::VideoReconFrame>& recon,
                          std::string& error) {
  CompareResult result;
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) {
    result.ok = false;
    return result;
  }
  av2enc::Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      result.ok = false;
      return result;
    }
    if (index >= recon.size()) {
      error = "decoder produced more frames than the encoder";
      result.ok = false;
      return result;
    }
    const av2enc::video::VideoReconFrame& expected = recon[index];
    if (frame.width() != expected.width || frame.height() != expected.height) {
      error = "decoded dimensions differ from encoder recon";
      result.ok = false;
      return result;
    }
    const auto compare = [&](std::span<const std::uint8_t> decoded,
                             const std::vector<std::uint8_t>& want) {
      if (decoded.size() != want.size()) {
        result.ok = false;
        return;
      }
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        if (decoded[i] != want[i]) ++result.mismatches;
      }
    };
    compare(frame.y(), expected.y);
    compare(frame.u(), expected.u);
    compare(frame.v(), expected.v);
    ++index;
    ++result.frames;
  }
  if (result.ok && index != recon.size()) {
    error = "decoder produced " + std::to_string(index) + " of " +
            std::to_string(recon.size()) + " frames";
    result.ok = false;
  }
  result.ok = result.ok && result.mismatches == 0;
  return result;
}

bool run(const std::vector<OwnedFrame>& frames, const std::string& out_path,
         bool trace, const std::string& decoder) {
  if (frames.size() != 3) {
    std::cerr << "gate needs exactly 3 frames, got " << frames.size() << "\n";
    return false;
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  std::string error;

  av2enc::video::VideoOptions options;
  options.trace = trace;
  if (trace) {
    std::cout << "M2 options: qp=" << static_cast<int>(options.base_q_idx)
              << "\n";
  }

  // A three-frame M2 sequence supplies the keyframe and the SWITCH payload
  // (frame 1 predicted from frame 0).
  std::vector<av2enc::video::VideoInputFrame> inputs3;
  for (const OwnedFrame& frame : frames) inputs3.push_back(to_input(frame));
  av2enc::video::VideoSequenceResult seq3;
  if (!av2enc::video::encode_video_sequence(inputs3, options, seq3, error)) {
    std::cerr << "M2 3-frame encode failed: " << error << "\n";
    return false;
  }
  if (seq3.units.size() != 3) {
    std::cerr << "M2 3-frame encode returned " << seq3.units.size()
              << " units\n";
    return false;
  }

  // A separate two-frame encode gives a P payload for frame 2 that predicts
  // from frame 0 (the long-term keyframe), not from the SWITCH frame. This is
  // the "re-encode against the long-term slot" branch of plan_ras_frame().
  std::vector<av2enc::video::VideoInputFrame> inputs2;
  inputs2.push_back(inputs3[0]);
  inputs2.push_back(inputs3[2]);
  av2enc::video::VideoSequenceResult seq2;
  if (!av2enc::video::encode_video_sequence(inputs2, options, seq2, error)) {
    std::cerr << "M2 RAS-payload encode failed: " << error << "\n";
    return false;
  }

  av2enc::video::VideoConfig video_config;
  video_config.max_frame_width = static_cast<std::uint16_t>(width);
  video_config.max_frame_height = static_cast<std::uint16_t>(height);
  video_config.base_q_idx = options.base_q_idx;
  video_config.order_hint_bits = 4;
  video_config.num_ref_frames = 2;
  video_config.num_total_refs = 1;
  video_config.max_drl_bits_minus_1 = 4;
  video_config.disable_cdf_update = true;
  video_config.enable_cfl_intra = true;

  av2enc::video::VideoFrameHeader video_key;
  video_key.keyframe = true;
  video_key.order_hint = 0;
  video_key.ref_slot = 0;
  video_key.refresh_frame_flags = (1U << video_config.num_ref_frames) - 1U;
  av2enc::video::VideoFrameHeader video_p1;
  video_p1.keyframe = false;
  video_p1.order_hint = 1;
  video_p1.ref_slot = 0;
  video_p1.refresh_frame_flags = 1;
  av2enc::video::VideoFrameHeader video_ras = video_p1;

  std::span<const std::uint8_t> key_tile;
  std::span<const std::uint8_t> switch_tile;
  std::span<const std::uint8_t> ras_tile;
  if (!extract_video_tile(seq3.units[0], 4, video_config, video_key, key_tile, error) ||
      !extract_video_tile(seq3.units[1], 7, video_config, video_p1, switch_tile, error) ||
      !extract_video_tile(seq2.units[1], 7, video_config, video_ras, ras_tile, error)) {
    std::cerr << "tile extraction failed: " << error << "\n";
    return false;
  }

  ra::SequenceConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.max_drl_bits_minus_1 = 4;
  config.long_term_frame_id_bits = 1;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;
  // Pinned dav2d f9dafbc reads explicit refresh bits for RAS (see
  // SequenceConfig::ras_codes_refresh_frame_flags).
  config.ras_codes_refresh_frame_flags = true;

  ra::RefState refs;

  ra::FrameHeader key_header;
  key_header.obu_type = ra::ObuType::kClosedLoopKey;
  key_header.order_hint = 0;
  key_header.long_term_id_plus_1 = 1;  // LongTermId = 0
  key_header.immediate_output_frame = true;
  ra::ValidationResult key_check =
      ra::validate_header(config, key_header, refs, 0);
  if (!key_check.ok) {
    std::cerr << "key header invalid: " << key_check.violations[0].message
              << "\n";
    return false;
  }
  const std::uint16_t key_refresh =
      static_cast<std::uint16_t>((1U << config.num_ref_frames) - 1U);
  ra::update_reference_state(config, key_header, key_refresh, refs);

  ra::PFrameCandidate switch_candidate;
  switch_candidate.reference_slot = 0;
  switch_candidate.refresh_frame_flags = 2;  // refresh slot 1, keep slot 0
  switch_candidate.order_hint = 1;
  switch_candidate.has_inter_blocks = seq3.inter_blocks[1] > 0;
  const ra::PayloadPlan switch_plan = ra::plan_switch_frame(
      config, refs, switch_candidate, /*restricted_prediction_switch=*/true);
  if (switch_plan.decision != ra::PayloadDecision::kReusePFrame) {
    std::cerr << "SWITCH plan rejected: " << switch_plan.reason << "\n";
    return false;
  }
  ra::FrameHeader switch_header = switch_plan.header;
  ra::ValidationResult switch_check =
      ra::validate_header(config, switch_header, refs, 0);
  if (!switch_check.ok) {
    std::cerr << "SWITCH header invalid: "
              << switch_check.violations[0].message << "\n";
    return false;
  }
  ra::update_reference_state(config, switch_header,
                             switch_header.refresh_frame_flags, refs);

  const std::uint8_t ras_ids[] = {0};
  ra::PFrameCandidate ras_candidate;
  ras_candidate.reference_slot = 0;
  ras_candidate.refresh_frame_flags = 0;  // RAS derives it
  ras_candidate.order_hint = 2;
  ras_candidate.has_inter_blocks = seq2.inter_blocks[1] > 0;
  const ra::PayloadPlan ras_plan =
      ra::plan_ras_frame(config, refs, ras_candidate, ras_ids);
  if (ras_plan.decision != ra::PayloadDecision::kReusePFrame) {
    std::cerr << "RAS plan rejected: " << ras_plan.reason << "\n";
    return false;
  }
  ra::FrameHeader ras_header = ras_plan.header;
  ra::ValidationResult ras_check =
      ra::validate_header(config, ras_header, refs, 0);
  if (!ras_check.ok) {
    std::cerr << "RAS header invalid: " << ras_check.violations[0].message
              << "\n";
    return false;
  }
  ra::update_reference_state(config, ras_header,
                             ras_header.refresh_frame_flags, refs);

  const std::vector<std::uint8_t> td = ra::make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = ra::make_sequence_header_obu(config);
  const std::vector<std::uint8_t> key_obu =
      ra::make_frame_obu(config, key_header, key_tile);
  const std::vector<std::uint8_t> switch_obu =
      ra::make_frame_obu(config, switch_header, switch_tile);
  const std::vector<std::uint8_t> ras_obu =
      ra::make_frame_obu(config, ras_header, ras_tile);

  std::vector<std::vector<std::uint8_t>> units;
  units.resize(3);
  append(units[0], td);
  append(units[0], seq);
  append(units[0], key_obu);
  append(units[1], td);
  append(units[1], switch_obu);
  append(units[2], td);
  append(units[2], ras_obu);

  if (!write_ivf(out_path, units, width, height, error)) {
    std::cerr << "IVF write failed: " << error << "\n";
    return false;
  }

  std::vector<av2enc::video::VideoReconFrame> recon;
  recon.push_back(seq3.recon[0]);
  recon.push_back(seq3.recon[1]);
  recon.push_back(seq2.recon[1]);

  std::printf("== random-access switch gate: %ux%u, 3 frames ==\n", width,
              height);
  std::printf("  frame 0: OBU_CLOSED_LOOP_KEY  order=0 long_term_id=0 "
              "tile=%zu bytes\n",
              key_tile.size());
  std::printf("  frame 1: OBU_SWITCH           order=1 restricted=1 "
              "ref_slot=0 refresh=0x%x inter_blocks=%u tile=%zu bytes\n",
              static_cast<unsigned>(switch_header.refresh_frame_flags),
              seq3.inter_blocks[1], switch_tile.size());
  std::printf("  frame 2: OBU_RAS_FRAME        order=2 num_long_term=%u "
              "num_total_refs=%u ref_long_term_id={%u} ref_slot=%u "
              "derived_refresh=0x%x inter_blocks=%u tile=%zu bytes\n",
              static_cast<unsigned>(ras_header.num_key_ref_frames),
              static_cast<unsigned>(ras_header.num_total_refs),
              ras_header.num_key_ref_frames > 0
                  ? static_cast<unsigned>(ras_header.ref_long_term_id[0])
                  : 0U,
              static_cast<unsigned>(ras_header.num_total_refs == 0
                                        ? 0
                                        : ras_header.ref_frame_idx[0]),
              static_cast<unsigned>(ras_header.refresh_frame_flags),
              seq2.inter_blocks[1], ras_tile.size());
  for (std::size_t i = 0; i < units.size(); ++i) {
    std::printf("  unit %zu: %zu bytes (frame OBU %zu bytes)\n", i,
                units[i].size(),
                i == 0 ? key_obu.size()
                       : (i == 1 ? switch_obu.size() : ras_obu.size()));
  }

  if (decoder.empty()) {
    std::cerr << "DECODE SKIP: no dav2d binary found (set DAV2D)\n";
    return false;
  }
  const std::string decoded_path = out_path + ".decoded.y4m";
  const std::string command = decoder + " --strict 1 --threads 1 -q -i " +
                              out_path + " -o " + decoded_path;
  std::printf("  decode: %s\n", command.c_str());
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "DECODE FAIL: dav2d returned " << rc << "\n";
    return false;
  }
  const CompareResult compare = compare_y4m(decoded_path, recon, error);
  if (!compare.ok) {
    std::cerr << "RECON MISMATCH: frames=" << compare.frames
              << " mismatches=" << compare.mismatches;
    if (!error.empty()) std::cerr << " (" << error << ")";
    std::cerr << "\n";
    return false;
  }
  std::printf("  PASS: %u frame(s) decoded with --strict 1 --threads 1, "
              "recon exact (0 diffs)\n",
              compare.frames);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "ras_gate.ivf";
  bool trace = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--trace") {
      trace = true;
    } else if (arg == "--help") {
      std::cout << "usage: ras_gate [--input clip.y4m] [--out out.ivf] "
                   "[--trace]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  std::vector<OwnedFrame> frames;
  if (!input.empty()) {
    std::string error;
    if (!load_y4m(input, 3, frames, error)) {
      std::cerr << "Y4M LOAD FAIL: " << error << "\n";
      return 2;
    }
  } else {
    frames = make_synthetic_clip(3, 128, 128);
  }
  if (frames.size() < 3) {
    std::cerr << "gate needs at least 3 input frames\n";
    return 2;
  }
  frames.resize(3);

  return run(frames, out_path, trace, find_dav2d()) ? 0 : 1;
}
