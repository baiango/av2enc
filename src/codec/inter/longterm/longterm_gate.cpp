// Decoder gate for the long-term-reference / open-random-access layer.
//
// Three phases, each one a complete IVF stream decoded with the pinned
// `dav2d --strict 1 --threads 1`:
//
//   closed      OBU_CLOSED_LOOP_KEY + P frames, long_term_frame_id_bits = 0.
//               §7.4.3: every reference buffer is invalidated at the CLK; the
//               P frames use only slot 0.
//
//   open-ltr    CLK A (long_term id 0)
//               TU 1: OLK K (id 1, names id 0, refreshes slot 1), P2 -> slot 2
//               TU 2: P3 references slot 0, the long-term reference 3 frames
//                     back, across the open random access point
//               TU 3: P4 references slot 2
//               §7.4.2.1/§7.4.4: the first Regular frame after the OLK
//               invalidates everything except OlkRefresh and the named
//               long-term references.
//
//   open-no-ltr Same shape with long_term_frame_id_bits = 0. The OLK names no
//               reference, the pre-OLK frames are dropped, and P3 uses the
//               OLK's own slot. §7.4.2.2.
//
// Every tile payload comes from the untouched M2 encoder. The M2 core always
// codes one reference and one intra keyframe, so each spliced frame is encoded
// against exactly the picture its rewritten header names:
//
//   [A]        -> CLK A
//   [K, P2]    -> OLK K, P2 (P2 predicted from K)
//   [A, P3, P4]-> P3 (predicted from A), P4 (predicted from P3)
//   [K, P3, P4]-> for open-no-ltr: P3 (predicted from K), P4 (from P3)
//
// The decoded pictures must match the M2 reconstruction sample for sample.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/longterm/longterm.h"
#include "codec/inter/longterm/longterm_stream.h"
#include "pipeline/video/video_frame.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

namespace {

using av2enc::inter::longterm::AssembledSequence;
using av2enc::inter::longterm::FrameHeader;
using av2enc::inter::longterm::FrameRole;
using av2enc::inter::longterm::PlannedFrame;
using av2enc::inter::longterm::SequenceConfig;
using av2enc::inter::longterm::TemporalUnitPlan;

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

std::uint8_t square_sample(int x, int y, int frame, int plane) {
  const int sx = 6 + 3 * frame;
  const int sy = 6 + 2 * frame;
  const int in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 32;
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

std::vector<Frame> make_synthetic_clip(int frames, int width, int height) {
  std::vector<Frame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    Frame frame;
    frame.width = width;
    frame.height = height;
    frame.y.resize(static_cast<std::size_t>(width) *
                   static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)] = square_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] = square_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] = square_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

bool load_y4m(const std::string& path, int want, std::vector<Frame>* out,
              std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out->size()) < want) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == av2enc::Y4mReadResult::kEof) {
      *error = "end of file after " + std::to_string(out->size()) + " frame(s)";
      break;
    }
    if (status != av2enc::Y4mReadResult::kFrame) return false;
    Frame copy;
    copy.width = static_cast<int>(frame.width());
    copy.height = static_cast<int>(frame.height());
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    out->push_back(std::move(copy));
  }
  return static_cast<int>(out->size()) == want;
}

struct SubEncode {
  std::vector<std::vector<std::uint8_t>> tiles;
  std::vector<av2enc::video::VideoReconFrame> recon;
};

bool run_m2(const std::vector<Frame>& frames,
            std::span<const int> indices, SubEncode* out, std::string* error) {
  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(indices.size());
  for (const int index : indices) {
    const Frame& frame = frames[static_cast<std::size_t>(index)];
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
  av2enc::video::VideoSequenceResult result;
  if (!av2enc::video::encode_video_sequence(inputs, options, result, *error)) {
    return false;
  }
  if (!av2enc::inter::longterm::extract_video_tiles(result.units,
                                                result.tile_bytes,
                                                &out->tiles, error)) {
    return false;
  }
  out->recon = std::move(result.recon);
  return true;
}

Frame to_frame(const av2enc::video::VideoReconFrame& recon) {
  Frame out;
  out.width = static_cast<int>(recon.width);
  out.height = static_cast<int>(recon.height);
  out.y = recon.y;
  out.u = recon.u;
  out.v = recon.v;
  return out;
}

bool write_ivf(const std::string& path, const AssembledSequence& sequence,
               int width, int height, std::string* error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = static_cast<std::uint32_t>(sequence.units.size());
  if (!writer.open(path, header, *error)) return false;
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    if (!writer.write_frame(sequence.units[i], i, *error)) return false;
  }
  return writer.close(*error);
}

std::string find_dav2d(const std::string& explicit_path) {
  if (!explicit_path.empty()) return explicit_path;
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

struct DiffResult {
  int frames = 0;
  std::uint64_t diffs = 0;
  int max_diff = 0;
};

bool compare(const std::vector<Frame>& decoded, const std::vector<Frame>& want,
             DiffResult* out, std::string* error) {
  out->frames = static_cast<int>(decoded.size());
  if (decoded.size() != want.size()) {
    *error = "decoder produced " + std::to_string(decoded.size()) + " of " +
             std::to_string(want.size()) + " expected pictures";
    return false;
  }
  for (std::size_t i = 0; i < decoded.size(); ++i) {
    const Frame& a = decoded[i];
    const Frame& b = want[i];
    if (a.width != b.width || a.height != b.height || a.y.size() != b.y.size() ||
        a.u.size() != b.u.size() || a.v.size() != b.v.size()) {
      *error = "frame " + std::to_string(i) + " geometry mismatch";
      return false;
    }
    const std::vector<std::uint8_t>* planes_a[3] = {&a.y, &a.u, &a.v};
    const std::vector<std::uint8_t>* planes_b[3] = {&b.y, &b.u, &b.v};
    for (int p = 0; p < 3; ++p) {
      for (std::size_t k = 0; k < planes_a[p]->size(); ++k) {
        const int d = std::abs(static_cast<int>((*planes_a[p])[k]) -
                               static_cast<int>((*planes_b[p])[k]));
        if (d != 0) {
          ++out->diffs;
          out->max_diff = std::max(out->max_diff, d);
        }
      }
    }
  }
  return out->diffs == 0;
}

bool load_y4m_all(const std::string& path, std::vector<Frame>* out,
                  std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status != av2enc::Y4mReadResult::kFrame) return false;
    Frame copy;
    copy.width = static_cast<int>(frame.width());
    copy.height = static_cast<int>(frame.height());
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    out->push_back(std::move(copy));
  }
  return true;
}

// Assembles, writes, decodes and compares one phase. `expected` holds the
// pictures the decoder must output, in coding order (an OLK is not output).
bool run_phase_decoder(const SequenceConfig& config,
                       const std::vector<TemporalUnitPlan>& units,
                       const std::vector<Frame>& expected,
                       const std::string& out_dir, const std::string& dav2d,
                       const std::string& name) {
  AssembledSequence assembled;
  std::string error;
  if (!av2enc::inter::longterm::assemble_sequence(config, units, &assembled,
                                                  &error)) {
    std::fprintf(stderr, "ASSEMBLE FAIL: %s\n", error.c_str());
    check(false, name + ": assemble");
    return false;
  }
  const std::string ivf = out_dir + "/longterm_" + name + ".ivf";
  if (!write_ivf(ivf, assembled, expected.front().width,
                 expected.front().height, &error)) {
    std::fprintf(stderr, "IVF FAIL: %s\n", error.c_str());
    check(false, name + ": ivf write");
    return false;
  }
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < assembled.units.size(); ++i) {
    total += assembled.units[i].size();
  }
  for (const av2enc::inter::longterm::FrameEvent& frame : assembled.frames) {
    std::printf(
        "  %-8s order %2u  obu %6zu B  refreshed 0x%x  invalidated 0x%x  "
        "valid 0x%x\n",
        frame.role.c_str(), frame.order_hint, frame.obu_bytes,
        frame.pool.refreshed_flags, frame.pool.invalidated_flags,
        frame.pool.valid_after);
  }
  std::printf("  %zu temporal unit(s), %llu bytes total\n",
              assembled.units.size(),
              static_cast<unsigned long long>(total));

  const std::string y4m = ivf + ".decoded.y4m";
  const std::string command = "\"" + dav2d + "\" -q -i \"" + ivf + "\" -o \"" +
                              y4m + "\" --strict 1 --threads 1";
  std::printf("  decode: %s\n", command.c_str());
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::fprintf(stderr, "DECODE FAIL: dav2d returned %d\n", rc);
    check(false, name + ": dav2d exit status");
    return false;
  }
  std::vector<Frame> decoded;
  if (!load_y4m_all(y4m, &decoded, &error)) {
    std::fprintf(stderr, "DECODE READ FAIL: %s\n", error.c_str());
    check(false, name + ": decoded y4m");
    return false;
  }
  DiffResult diff;
  const bool identical = compare(decoded, expected, &diff, &error);
  std::printf("  %d/%zu picture(s) decoded, %llu sample diffs, max %d\n",
              diff.frames, expected.size(),
              static_cast<unsigned long long>(diff.diffs), diff.max_diff);
  check(identical, name + ": recon exact (" + error + ")");
  if (identical) std::printf("  PASS: %s recon exact\n", name.c_str());
  return identical;
}

SequenceConfig base_config(int width, int height, std::uint8_t lt_bits) {
  SequenceConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 4;
  config.long_term_frame_id_bits = lt_bits;
  return config;
}

// ---------------------------------------------------------------------------
// Phase: closed random access only (CLK + P frames)
// ---------------------------------------------------------------------------
bool phase_closed(const std::vector<Frame>& frames, const std::string& out_dir,
                  const std::string& dav2d) {
  std::printf("== phase closed: CLK + 2 P frames, long_term_frame_id_bits = 0 ==\n");
  SubEncode sub;
  std::string error;
  if (!run_m2(frames, std::vector<int>{0, 1, 2}, &sub, &error)) {
    std::fprintf(stderr, "ENCODE FAIL: %s\n", error.c_str());
    return false;
  }
  const SequenceConfig config = base_config(frames[0].width, frames[0].height, 0);
  av2enc::inter::longterm::PoolState pool;
  av2enc::inter::longterm::reset_pool(config, &pool);

  std::vector<TemporalUnitPlan> units;
  std::vector<Frame> expected;

  units.emplace_back();
  FrameHeader clk{};
  if (!av2enc::inter::longterm::plan_closed_loop_key(config, 0, -1, &clk,
                                                     &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{clk, sub.tiles[0]});
  expected.push_back(to_frame(sub.recon[0]));
  av2enc::inter::longterm::PoolEvents events;
  if (!av2enc::inter::longterm::apply_frame(config, clk, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }

  for (int i = 1; i < 3; ++i) {
    units.emplace_back();
    FrameHeader p{};
    if (!av2enc::inter::longterm::plan_inter_frame(
            config, FrameRole::kRegular, static_cast<std::uint32_t>(i),
            std::vector<std::uint8_t>{0}, 0, true, &p, &error)) {
      std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
      return false;
    }
    units.back().frames.push_back(PlannedFrame{p, sub.tiles[static_cast<std::size_t>(i)]});
    expected.push_back(to_frame(sub.recon[static_cast<std::size_t>(i)]));
    if (!av2enc::inter::longterm::apply_frame(config, p, true, &pool, &events,
                                              &error)) {
      std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
      return false;
    }
  }
  return run_phase_decoder(config, units, expected, out_dir, dav2d, "closed");
}

// ---------------------------------------------------------------------------
// Phase: open random access with long-term references
// ---------------------------------------------------------------------------
bool phase_open_ltr(const std::vector<Frame>& frames,
                    const std::string& out_dir, const std::string& dav2d) {
  std::printf(
      "== phase open-ltr: CLK, OLK (pinned-dav2d list layout), P frames, "
      "long-term retention across the RAP ==\n");
  SubEncode key_a;
  SubEncode key_k;
  SubEncode chain_a;
  std::string error;
  if (!run_m2(frames, std::vector<int>{0}, &key_a, &error) ||
      !run_m2(frames, std::vector<int>{1, 2}, &key_k, &error) ||
      !run_m2(frames, std::vector<int>{0, 3, 4}, &chain_a, &error)) {
    std::fprintf(stderr, "ENCODE FAIL: %s\n", error.c_str());
    return false;
  }

  SequenceConfig config = base_config(frames[0].width, frames[0].height, 2);
  // The pinned dav2d build does not read the v1.0.0 OLK long-term list, so the
  // gate uses the documented compatibility layout. Retention across the RAP
  // still happens: §5.18.2 keeps every slot whose RefLongTermId != -1.
  config.olk_long_term_list =
      av2enc::inter::longterm::LongTermListSyntax::kOmitPinnedDav2d;
  av2enc::inter::longterm::PoolState pool;
  av2enc::inter::longterm::reset_pool(config, &pool);
  av2enc::inter::longterm::PoolEvents events;
  std::vector<TemporalUnitPlan> units;
  std::vector<Frame> expected;

  // TU 0: CLK A, long-term id 0, refreshes all slots.
  units.emplace_back();
  FrameHeader a{};
  if (!av2enc::inter::longterm::plan_closed_loop_key(config, 0, /*lt=*/0, &a,
                                                     &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{a, key_a.tiles[0]});
  expected.push_back(to_frame(key_a.recon[0]));
  if (!av2enc::inter::longterm::apply_frame(config, a, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }

  // TU 1: OLK K (id 1, retains id 0, slot 1) + P2 (slot 1 -> slot 2).
  units.emplace_back();
  FrameHeader k{};
  if (!av2enc::inter::longterm::plan_open_loop_key(
          config, pool, /*order_hint=*/1, /*olk_slot=*/1, /*lt=*/1, {}, &k,
          &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{k, key_k.tiles[0]});
  if (!av2enc::inter::longterm::apply_frame(config, k, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }
  FrameHeader p2{};
  if (!av2enc::inter::longterm::plan_inter_frame(
          config, FrameRole::kRegular, 2, std::vector<std::uint8_t>{1},
          /*refresh=*/2, true, &p2, &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{p2, key_k.tiles[1]});
  expected.push_back(to_frame(key_k.recon[1]));
  if (!av2enc::inter::longterm::apply_frame(config, p2, false, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }

  // TU 2: P3 references slot 0 (the CLK three frames back) across the open
  // random access point.
  units.emplace_back();
  FrameHeader p3{};
  if (!av2enc::inter::longterm::plan_inter_frame(
          config, FrameRole::kRegular, 3, std::vector<std::uint8_t>{0},
          /*refresh=*/2, true, &p3, &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{p3, chain_a.tiles[1]});
  expected.push_back(to_frame(chain_a.recon[1]));
  if (!av2enc::inter::longterm::apply_frame(config, p3, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }
  if (!events.invalidation_ran ||
      (events.valid_after & 0x1U) == 0U) {
    std::fprintf(stderr,
                 "FAIL: the long-term reference did not survive the OLK\n");
    return false;
  }

  // TU 3: P4 references slot 2 (P3).
  units.emplace_back();
  FrameHeader p4{};
  if (!av2enc::inter::longterm::plan_inter_frame(
          config, FrameRole::kRegular, 4, std::vector<std::uint8_t>{2},
          /*refresh=*/2, true, &p4, &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{p4, chain_a.tiles[2]});
  expected.push_back(to_frame(chain_a.recon[2]));
  if (!av2enc::inter::longterm::apply_frame(config, p4, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }

  return run_phase_decoder(config, units, expected, out_dir, dav2d, "open-ltr");
}

// ---------------------------------------------------------------------------
// Phase: open random access without long-term references (§7.4.2.2)
// ---------------------------------------------------------------------------
bool phase_open_no_ltr(const std::vector<Frame>& frames,
                       const std::string& out_dir, const std::string& dav2d) {
  std::printf(
      "== phase open-no-ltr: open random access, long_term_frame_id_bits = 0 "
      "==\n");
  SubEncode key_a;
  SubEncode key_k;
  SubEncode chain_k;
  std::string error;
  if (!run_m2(frames, std::vector<int>{0}, &key_a, &error) ||
      !run_m2(frames, std::vector<int>{1, 2}, &key_k, &error) ||
      !run_m2(frames, std::vector<int>{1, 3, 4}, &chain_k, &error)) {
    std::fprintf(stderr, "ENCODE FAIL: %s\n", error.c_str());
    return false;
  }
  const SequenceConfig config = base_config(frames[0].width, frames[0].height, 0);
  av2enc::inter::longterm::PoolState pool;
  av2enc::inter::longterm::reset_pool(config, &pool);
  av2enc::inter::longterm::PoolEvents events;
  std::vector<TemporalUnitPlan> units;
  std::vector<Frame> expected;

  units.emplace_back();
  FrameHeader a{};
  if (!av2enc::inter::longterm::plan_closed_loop_key(config, 0, -1, &a,
                                                     &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{a, key_a.tiles[0]});
  expected.push_back(to_frame(key_a.recon[0]));
  if (!av2enc::inter::longterm::apply_frame(config, a, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }

  units.emplace_back();
  FrameHeader k{};
  if (!av2enc::inter::longterm::plan_open_loop_key(
          config, pool, 1, /*olk_slot=*/1, /*lt=*/-1, {}, &k, &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{k, key_k.tiles[0]});
  if (!av2enc::inter::longterm::apply_frame(config, k, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }
  FrameHeader p2{};
  if (!av2enc::inter::longterm::plan_inter_frame(
          config, FrameRole::kRegular, 2, std::vector<std::uint8_t>{1}, 2, true,
          &p2, &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{p2, key_k.tiles[1]});
  expected.push_back(to_frame(key_k.recon[1]));
  if (!av2enc::inter::longterm::apply_frame(config, p2, false, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }

  units.emplace_back();
  FrameHeader p3{};
  if (!av2enc::inter::longterm::plan_inter_frame(
          config, FrameRole::kRegular, 3, std::vector<std::uint8_t>{1}, 2, true,
          &p3, &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{p3, chain_k.tiles[1]});
  expected.push_back(to_frame(chain_k.recon[1]));
  if (!av2enc::inter::longterm::apply_frame(config, p3, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }
  if (!events.invalidation_ran || (events.invalidated_flags & 0x1U) == 0U) {
    std::fprintf(stderr,
                 "FAIL: the pre-OLK slot should have been invalidated\n");
    return false;
  }

  units.emplace_back();
  FrameHeader p4{};
  if (!av2enc::inter::longterm::plan_inter_frame(
          config, FrameRole::kRegular, 4, std::vector<std::uint8_t>{2}, 2, true,
          &p4, &error)) {
    std::fprintf(stderr, "PLAN FAIL: %s\n", error.c_str());
    return false;
  }
  units.back().frames.push_back(PlannedFrame{p4, chain_k.tiles[2]});
  expected.push_back(to_frame(chain_k.recon[2]));
  if (!av2enc::inter::longterm::apply_frame(config, p4, true, &pool, &events,
                                            &error)) {
    std::fprintf(stderr, "APPLY FAIL: %s\n", error.c_str());
    return false;
  }

  return run_phase_decoder(config, units, expected, out_dir, dav2d,
                           "open-no-ltr");
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_dir = ".";
  std::string dav2d;
  int width = 64;
  int height = 64;
  int frames_wanted = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      frames_wanted = std::atoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--dav2d" && i + 1 < argc) {
      dav2d = argv[++i];
    } else if (arg == "--width" && i + 1 < argc) {
      width = std::atoi(argv[++i]);
    } else if (arg == "--height" && i + 1 < argc) {
      height = std::atoi(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: %s [--input clip.y4m] [--frames 5] [--out DIR] "
          "[--dav2d PATH] [--width N --height N]\n",
          argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (frames_wanted < 5) {
    std::fprintf(stderr, "--frames must be >= 5\n");
    return 2;
  }

  std::vector<Frame> frames;
  if (!input.empty()) {
    std::string error;
    if (load_y4m(input, frames_wanted, &frames, &error)) {
      std::printf("input: %s (%d frames)\n", input.c_str(), frames_wanted);
    } else {
      std::fprintf(stderr,
                   "note: %s did not provide %d frames (%s); using the "
                   "synthetic clip\n",
                   input.c_str(), frames_wanted, error.c_str());
      frames.clear();
    }
  }
  if (frames.empty()) {
    frames = make_synthetic_clip(frames_wanted, width, height);
    std::printf("input: synthetic %dx%d, %d frames\n", width, height,
                frames_wanted);
  }
  if (frames.front().width % 8 != 0 || frames.front().height % 8 != 0) {
    std::fprintf(stderr, "gate needs dimensions divisible by 8\n");
    return 2;
  }

  const std::string decoder = find_dav2d(dav2d);
  if (decoder.empty()) {
    std::fprintf(stderr, "no dav2d found; set DAV2D=/path/to/dav2d\n");
    return 77;
  }
  std::printf("dav2d: %s\n", decoder.c_str());

  bool ok = phase_closed(frames, out_dir, decoder);
  ok = phase_open_ltr(frames, out_dir, decoder) && ok;
  ok = phase_open_no_ltr(frames, out_dir, decoder) && ok;

  std::printf("%d checks, %d failure(s)\n", g_checks, g_failures);
  return ok && g_failures == 0 ? 0 : 1;
}
