// Real-encode validation for the two-pass / VBV / lookahead policy.
//
// Encodes the 720p Shine clip with the M2 CPU reference frame loop at a small
// QP sweep, builds a measured rate table from the real payload sizes, and then
// drives the second-pass controller against those measurements:
//
//   * 2-frame and 8-frame VBR runs with a VBV model and per-tile delta-Q;
//   * a CRF run with qcomp, segmentation, and a tight-buffer underflow test;
//   * determinism: identical plans for identical inputs and actual bits;
//   * one stream is decoded with dav2d when the validator is available.
//
// This test does not modify M2 and does not test bitstream syntax; the M2 gate
// owns decode correctness. It measures whether the RC policy tracks a real
// rate curve and keeps the model buffer feasible.

#include "twopass.h"

#include "../rc.h"
#include "pipeline/video/video_frame.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace av2enc;
using namespace av2enc::rc;
using namespace av2enc::rc::twopass;

int g_failures = 0;
int g_checks = 0;

void check_impl(bool condition, const char* expression, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "line " << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) check_impl((expression), #expression, __LINE__)

struct VideoFrame {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool load_frames(const std::filesystem::path& path,
                 std::size_t limit,
                 std::vector<VideoFrame>& frames,
                 std::uint32_t& width,
                 std::uint32_t& height,
                 std::uint32_t& fps_num,
                 std::uint32_t& fps_den,
                 std::string& error) {
  Y4mReader reader(path);
  if (!reader.open(error)) {
    return false;
  }
  width = reader.header().width;
  height = reader.header().height;
  fps_num = reader.header().fps_num;
  fps_den = reader.header().fps_den;
  if (fps_num == 0 || fps_den == 0) {
    error = "input has no frame rate";
    return false;
  }
  Y4mFrame frame;
  while (frames.size() < limit) {
    const Y4mReadResult status = reader.read_frame(frame, error);
    if (status == Y4mReadResult::kEof) {
      break;
    }
    if (status == Y4mReadResult::kError) {
      return false;
    }
    VideoFrame copy;
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    frames.push_back(std::move(copy));
  }
  if (frames.empty()) {
    error = "no frames read";
    return false;
  }
  return true;
}

bool encode_at(const std::vector<VideoFrame>& frames,
               std::uint32_t width,
               std::uint32_t height,
               int qp,
               av2enc::video::VideoSequenceResult& result,
               std::string& error) {
  std::vector<av2enc::video::VideoInputFrame> views;
  views.reserve(frames.size());
  for (const VideoFrame& frame : frames) {
    av2enc::video::VideoInputFrame view;
    view.width = width;
    view.height = height;
    view.y = std::span<const std::uint8_t>(frame.y);
    view.u = std::span<const std::uint8_t>(frame.u);
    view.v = std::span<const std::uint8_t>(frame.v);
    views.push_back(view);
  }
  av2enc::video::VideoOptions options;
  options.base_q_idx = static_cast<std::uint8_t>(qp);
  // The 720p gate uses a single 64x64 block to keep the CPU reference encode
  // fast. The rate-control test only needs real payload sizes.
  options.force_single_block = true;
  return av2enc::video::encode_video_sequence(views, options, result, error);
}

// Measured rate curve: bits[qp_index][frame] from a real M2 encode at qps
// [qp_index]. `frame` is set by the test before each begin_frame().
struct RateTable {
  std::vector<int> qps;
  std::vector<std::vector<double>> bits;
  std::size_t frame = 0;
  double full_pixels = 1.0;
};

double log_bits(double bits) {
  return std::log(std::max(bits, 1.0));
}

double curve_bits(const RateTable& table, std::size_t frame, int qindex) {
  const std::size_t count = table.qps.size();
  if (count == 0) {
    return 1.0;
  }
  if (count == 1) {
    return std::max(table.bits[0][frame], 1.0);
  }
  std::size_t hi = 1;
  while (hi < count &&
         static_cast<int>(table.qps[hi]) < qindex) {
    ++hi;
  }
  hi = std::min(hi, count - 1);
  const std::size_t lo = hi - 1;
  const int q0 = table.qps[lo];
  const int q1 = table.qps[hi];
  const double b0 = log_bits(table.bits[lo][frame]);
  const double b1 = log_bits(table.bits[hi][frame]);
  const double span = static_cast<double>(q1 - q0);
  const double slope = span > 0.0 ? (b1 - b0) / span : 0.0;
  const double value = std::exp(b0 + slope * static_cast<double>(qindex - q0));
  return std::clamp(value, 1.0, 1.0e12);
}

// RC selection assumes estimate(q) is non-increasing. Real P-frame payloads
// can rise slightly between two coarse QPs because the mode decision changes.
// Enforce a running minimum toward coarser QP so the calibrated curve obeys
// the same monotonicity as the rate model. The measured anchor points remain
// real encodes; the envelope only removes small non-monotone steps.
void make_monotone(RateTable& table) {
  for (std::size_t frame = 0; frame < table.bits.front().size(); ++frame) {
    for (std::size_t q = 1; q < table.qps.size(); ++q) {
      table.bits[q][frame] =
          std::min(table.bits[q][frame], table.bits[q - 1][frame]);
    }
  }
}

double measured_estimate(void* context,
                         const FrameComplexity& complexity,
                         int qindex) {
  const auto* table = static_cast<const RateTable*>(context);
  const double value = curve_bits(*table, table->frame, qindex);
  const double pixels = static_cast<double>(complexity.pixel_count());
  const double scale =
      table->full_pixels > 0.0 ? pixels / table->full_pixels : 1.0;
  return std::max(value * scale, 1.0);
}

bool same_plan(const FramePlan& a, const FramePlan& b) {
  if (a.frame_index != b.frame_index || a.keyframe != b.keyframe ||
      a.base_q_idx != b.base_q_idx ||
      a.coeff_cdf_qctx != b.coeff_cdf_qctx ||
      a.continuous_qp != b.continuous_qp || a.lambda_sse != b.lambda_sse ||
      a.allocation_bits != b.allocation_bits ||
      a.target_bits != b.target_bits || a.estimated_bits != b.estimated_bits ||
      a.rate_feedback != b.rate_feedback ||
      a.delta_q.present != b.delta_q.present ||
      a.delta_q.res != b.delta_q.res ||
      a.segmentation.enabled != b.segmentation.enabled ||
      a.vbv_fill_before != b.vbv_fill_before ||
      a.vbv_underflow_prevented != b.vbv_underflow_prevented ||
      a.target_reachable != b.target_reachable) {
    return false;
  }
  if (a.tiles.size() != b.tiles.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.tiles.size(); ++i) {
    if (a.tiles[i].sb_target_qindex != b.tiles[i].sb_target_qindex ||
        a.tiles[i].sb_achieved_qindex != b.tiles[i].sb_achieved_qindex ||
        a.tiles[i].sb_segment_id != b.tiles[i].sb_segment_id) {
      return false;
    }
    if (a.tiles[i].delta_q.steps.size() !=
        b.tiles[i].delta_q.steps.size()) {
      return false;
    }
    for (std::size_t k = 0; k < a.tiles[i].delta_q.steps.size(); ++k) {
      if (a.tiles[i].delta_q.steps[k].qindex_after !=
              b.tiles[i].delta_q.steps[k].qindex_after ||
          a.tiles[i].delta_q.steps[k].reduced_delta !=
              b.tiles[i].delta_q.steps[k].reduced_delta) {
        return false;
      }
    }
  }
  return true;
}

std::vector<FrameStats> first_pass(const std::vector<VideoFrame>& frames,
                                   std::uint32_t width,
                                   std::uint32_t height,
                                   std::string& error) {
  std::vector<FrameStats> stats;
  stats.reserve(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    FrameStats frame;
    if (!first_pass_luma(frames[i].y, static_cast<int>(width),
                         static_cast<int>(height), static_cast<int>(width),
                         2, 2, i == 0, frame, &error)) {
      stats.clear();
      return stats;
    }
    stats.push_back(std::move(frame));
  }
  return stats;
}

struct RunResult {
  std::vector<FramePlan> plans;
  std::vector<double> actual_bits;
  double total_target_bits = 0.0;
  double total_actual_bits = 0.0;
  bool underflow = false;
  bool overflow = false;
  int underflow_prevented = 0;
  int unreachable = 0;
};

bool run_second_pass(const std::vector<FrameStats>& stats,
                     TwoPassConfig config,
                     RateTable& table,
                     std::size_t frame_count,
                     RunResult& result,
                     std::string& error) {
  // Bind the estimator to this table instance, not a caller's stale context.
  config.rc.estimator.context = &table;
  config.rc.estimator.estimate = &measured_estimate;
  TwoPassController controller;
  if (!controller.configure(config, &error)) {
    return false;
  }
  if (!controller.analyze(stats, &error)) {
    return false;
  }
  result = RunResult{};
  for (std::size_t i = 0; i < frame_count; ++i) {
    table.frame = i;
    FramePlan plan;
    if (!controller.begin_frame(plan, &error)) {
      return false;
    }
    // Real encodes anchor the curve. The per-frame value at the chosen q is
    // the monotone log-linear interpolation of those measured payload sizes
    // (a QP index is not on the sparse measurement grid).
    const double actual = curve_bits(table, i, plan.base_q_idx);
    if (plan.vbv_underflow_prevented) {
      ++result.underflow_prevented;
    }
    if (!plan.target_reachable) {
      ++result.unreachable;
    }
    result.plans.push_back(plan);
    result.actual_bits.push_back(actual);
    result.total_actual_bits += actual;
    if (!controller.end_frame(static_cast<std::int64_t>(actual), &error)) {
      return false;
    }
  }
  result.total_target_bits = controller.total_target_bits();
  result.underflow = controller.vbv().underflow_events > 0;
  result.overflow = controller.vbv().overflow_events > 0;
  return true;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) {
      return env;
    }
  }
  for (const char* candidate : {"tools/bin/dav2d", "tools/bin/dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return {};
}

bool decode_check(const av2enc::video::VideoSequenceResult& sequence,
                  std::uint32_t width,
                  std::uint32_t height,
                  std::uint32_t fps_num,
                  std::uint32_t fps_den,
                  bool& decoded) {
  decoded = false;
  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    return true;
  }
  const std::filesystem::path out =
      std::filesystem::temp_directory_path() / "av2enc_twopass_real.ivf";
  const std::filesystem::path recon =
      std::filesystem::temp_directory_path() / "av2enc_twopass_real.y4m";
  IvfWriter writer;
  IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = fps_num;
  header.fps_den = fps_den;
  std::string error;
  if (!writer.open(out, header, error)) {
    std::cerr << "dav2d check: " << error << '\n';
    return false;
  }
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    if (!writer.write_frame(sequence.units[i], i, error)) {
      std::cerr << "dav2d check: " << error << '\n';
      return false;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "dav2d check: " << error << '\n';
    return false;
  }
  const std::string command = dav2d + " -q -i " + out.string() + " -o " +
                              recon.string() + " --threads 0";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "dav2d check failed: " << command << " -> " << rc << '\n';
    return false;
  }
  decoded = true;
  return true;
}

double mean_sigma(const std::vector<FrameStats>& stats) {
  double sum = 0.0;
  for (const FrameStats& frame : stats) {
    sum += frame_complexity_sigma(frame.complexity);
  }
  return stats.empty() ? 0.0 : sum / static_cast<double>(stats.size());
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path input =
      argc > 1 ? std::filesystem::path(argv[1])
               : std::filesystem::path("testdata/shine_1280x720_60f.y4m");
  std::string error;

  std::vector<VideoFrame> frames;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 0;
  if (!load_frames(input, 8, frames, width, height, fps_num, fps_den, error)) {
    std::cerr << "real test: " << error << '\n';
    return 1;
  }
  const double fps = static_cast<double>(fps_num) / static_cast<double>(fps_den);
  std::cout << "input: " << input << " " << width << "x" << height << " "
            << frames.size() << " frames, " << fps << " fps\n";

  const std::vector<FrameStats> stats =
      first_pass(frames, width, height, error);
  if (stats.size() != frames.size()) {
    std::cerr << "real test: first pass failed: " << error << '\n';
    return 1;
  }
  std::cout << "first pass:";
  for (const FrameStats& frame : stats) {
    std::cout << " " << frame_complexity_sigma(frame.complexity);
  }
  std::cout << "\n";

  // -----------------------------------------------------------------------
  // Measured rate tables (real M2 encodes)
  // -----------------------------------------------------------------------
  const std::vector<int> eight_qps = {60, 120, 180, 220};
  const std::vector<int> two_qps = {90, 180, 220};
  std::vector<av2enc::video::VideoSequenceResult> eight_sequences;
  RateTable eight_table;
  eight_table.qps = eight_qps;
  eight_table.full_pixels =
      static_cast<double>(width) * static_cast<double>(height);
  eight_table.bits.resize(eight_qps.size());
  for (std::size_t q = 0; q < eight_qps.size(); ++q) {
    av2enc::video::VideoSequenceResult sequence;
    if (!encode_at(frames, width, height, eight_qps[q], sequence, error)) {
      std::cerr << "real test: encode qp " << eight_qps[q] << ": " << error
                << '\n';
      return 1;
    }
    if (sequence.units.size() != frames.size()) {
      std::cerr << "real test: encode returned the wrong frame count\n";
      return 1;
    }
    for (std::size_t i = 0; i < frames.size(); ++i) {
      eight_table.bits[q].push_back(
          static_cast<double>(sequence.units[i].size()) * 8.0);
    }
    eight_sequences.push_back(std::move(sequence));
    std::cout << "encode qp " << eight_qps[q] << ":";
    for (const double bits : eight_table.bits[q]) {
      std::cout << " " << bits;
    }
    std::cout << "\n";
  }
  make_monotone(eight_table);
  for (std::size_t q = 1; q < eight_qps.size(); ++q) {
    for (std::size_t i = 0; i < frames.size(); ++i) {
      CHECK(eight_table.bits[q][i] <= eight_table.bits[q - 1][i] + 1e-9);
    }
  }
  // A real first pass uses the probe payload sizes. The QP 120 encode is the
  // probe: its frame bits drive the two-pass allocation, exactly the
  // FrameStats::first_pass_bits hook.
  std::vector<FrameStats> pass_stats = stats;
  for (std::size_t i = 0; i < pass_stats.size(); ++i) {
    pass_stats[i].first_pass_bits = eight_table.bits[1][i];
  }

  std::vector<av2enc::video::VideoSequenceResult> two_sequences;
  RateTable two_table;
  two_table.qps = two_qps;
  two_table.full_pixels =
      static_cast<double>(width) * static_cast<double>(height);
  two_table.bits.resize(two_qps.size());
  for (std::size_t q = 0; q < two_qps.size(); ++q) {
    av2enc::video::VideoSequenceResult sequence;
    std::vector<VideoFrame> two_frames(frames.begin(), frames.begin() + 2);
    if (!encode_at(two_frames, width, height, two_qps[q], sequence, error)) {
      std::cerr << "real test: 2-frame encode qp " << two_qps[q] << ": "
                << error << '\n';
      return 1;
    }
    for (std::size_t i = 0; i < 2; ++i) {
      two_table.bits[q].push_back(
          static_cast<double>(sequence.units[i].size()) * 8.0);
    }
    two_sequences.push_back(std::move(sequence));
  }
  make_monotone(two_table);

  // -----------------------------------------------------------------------
  // VBR, 8 frames, 2x2 tiles, delta-Q + segmentation, VBV
  // -----------------------------------------------------------------------
  double base_total = 0.0;
  for (const double bits : eight_table.bits[1]) {  // QP 120
    base_total += bits;
  }
  const double base_bps = base_total * fps / static_cast<double>(frames.size());

  TwoPassConfig vbr;
  vbr.rc.mode = RateControlMode::kVbr;
  vbr.rc.vbr.target_bitrate_bps = 0.55 * base_bps;
  vbr.rc.vbr.frame_rate_num = static_cast<std::int32_t>(fps_num);
  vbr.rc.vbr.frame_rate_den = static_cast<std::int32_t>(fps_den);
  vbr.vbv.enabled = true;
  vbr.vbv.buffer_frames = 6.0;
  vbr.vbv.initial_fill = 0.9;
  vbr.vbv.max_rate_bps = 2.0 * vbr.rc.vbr.target_bitrate_bps;
  vbr.vbv.underflow_margin = 0.05;
  vbr.max_qindex = 180;
  vbr.spatial_qp_strength = 8;
  vbr.enable_delta_q = true;
  vbr.delta_q_res = 1;
  vbr.enable_segmentation = true;
  vbr.segment_levels = 1;
  vbr.segment_qindex_step = 4;
  vbr.rate_feedback_weight = 0.5;

  RunResult vbr_run;
  CHECK(run_second_pass(pass_stats, vbr, eight_table, frames.size(), vbr_run,
                        error));
  CHECK(vbr_run.plans.size() == frames.size());
  CHECK(!vbr_run.underflow);
  std::cout << "vbr 8f: target=" << vbr_run.total_target_bits
            << " actual=" << vbr_run.total_actual_bits
            << " underflow=" << vbr_run.underflow
            << " prevented=" << vbr_run.underflow_prevented
            << " unreachable=" << vbr_run.unreachable << "\n";
  const double clip_target =
      vbr.rc.vbr.target_bitrate_bps * static_cast<double>(frames.size()) / fps;
  CHECK(std::fabs(vbr_run.total_actual_bits - clip_target) <=
        0.30 * clip_target);
  for (const FramePlan& plan : vbr_run.plans) {
    CHECK(plan.base_q_idx >= vbr.min_qindex);
    CHECK(plan.base_q_idx <= vbr.max_qindex);
    CHECK(plan.delta_q.present);
    CHECK(plan.segmentation.enabled);
    CHECK(plan.segmentation_map.segment_id.size() ==
          static_cast<std::size_t>((width / 4) * (height / 4)));
    for (const TilePlan& tile : plan.tiles) {
      CHECK(!tile.delta_q.steps.empty());
      CHECK(tile.delta_q.steps.front().qindex_before == plan.base_q_idx);
    }
  }
  // Per-frame adaptation actually moved the frame quantizer.
  bool varied = false;
  for (std::size_t i = 1; i < vbr_run.plans.size(); ++i) {
    if (vbr_run.plans[i].base_q_idx != vbr_run.plans[0].base_q_idx) {
      varied = true;
    }
  }
  CHECK(varied);

  // Determinism: the same measured trace must produce identical plans.
  RateTable eight_table_copy = eight_table;
  RunResult vbr_again;
  CHECK(run_second_pass(pass_stats, vbr, eight_table_copy, frames.size(),
                        vbr_again, error));
  CHECK(vbr_run.plans.size() == vbr_again.plans.size());
  for (std::size_t i = 0; i < vbr_run.plans.size(); ++i) {
    CHECK(same_plan(vbr_run.plans[i], vbr_again.plans[i]));
    CHECK(vbr_run.actual_bits[i] == vbr_again.actual_bits[i]);
  }
  CHECK(vbr_run.total_target_bits == vbr_again.total_target_bits);

  // -----------------------------------------------------------------------
  // VBR, 2 frames
  // -----------------------------------------------------------------------
  TwoPassConfig vbr2 = vbr;
  vbr2.rc.estimator.context = &two_table;
  vbr2.vbv.buffer_frames = 2.0;
  vbr2.vbv.initial_fill = 0.9;
  vbr2.max_qindex = 180;
  const double two_base = two_table.bits[0][0] + two_table.bits[0][1];
  vbr2.rc.vbr.target_bitrate_bps =
      0.90 * two_base * fps / 2.0;
  vbr2.vbv.max_rate_bps = 2.0 * vbr2.rc.vbr.target_bitrate_bps;
  std::vector<FrameStats> pass_two(pass_stats.begin(), pass_stats.begin() + 2);
  for (std::size_t i = 0; i < 2; ++i) {
    pass_two[i].first_pass_bits = two_table.bits[0][i];
  }
  RunResult two_run;
  CHECK(run_second_pass(pass_two, vbr2, two_table, 2, two_run, error));
  CHECK(two_run.plans.size() == 2);
  CHECK(!two_run.underflow);
  std::cout << "vbr 2f: target=" << two_run.total_target_bits
            << " actual=" << two_run.total_actual_bits << "\n";
  RateTable two_copy = two_table;
  RunResult two_again;
  CHECK(run_second_pass(pass_two, vbr2, two_copy, 2, two_again, error));
  CHECK(same_plan(two_run.plans[0], two_again.plans[0]));
  CHECK(same_plan(two_run.plans[1], two_again.plans[1]));

  // Decode the real 2-frame stream to prove these are not synthetic payloads.
  bool decoded = false;
  CHECK(decode_check(two_sequences[0], width, height, fps_num, fps_den,
                     decoded));
  std::cout << "dav2d: " << (decoded ? "decoded" : "not available") << "\n";

  // -----------------------------------------------------------------------
  // CRF with qcomp, 8 frames, plus a tight-buffer underflow test
  // -----------------------------------------------------------------------
  const double reference_sigma = mean_sigma(stats);
  CHECK(reference_sigma > 0.0);
  TwoPassConfig crf;
  crf.rc.mode = RateControlMode::kCrf;
  crf.rc.crf.crf = 40.0;
  crf.rc.crf.qcomp = 0.5;
  crf.rc.crf.complexity_reference = reference_sigma;
  crf.rc.vbr.target_bitrate_bps = base_bps;
  crf.rc.vbr.frame_rate_num = static_cast<std::int32_t>(fps_num);
  crf.rc.vbr.frame_rate_den = static_cast<std::int32_t>(fps_den);
  crf.max_qindex = 180;
  crf.vbv.enabled = true;
  crf.vbv.buffer_frames = 8.0;
  crf.vbv.initial_fill = 0.5;
  crf.vbv.max_rate_bps = 2.0 * base_bps;
  crf.spatial_qp_strength = 8;
  crf.enable_delta_q = true;
  crf.delta_q_res = 1;
  crf.enable_segmentation = true;
  crf.segment_levels = 1;
  crf.rate_feedback_weight = 0.5;

  RunResult crf_run;
  CHECK(run_second_pass(pass_stats, crf, eight_table, frames.size(), crf_run,
                        error));
  CHECK(crf_run.plans.size() == frames.size());
  CHECK(!crf_run.underflow);
  for (const FramePlan& plan : crf_run.plans) {
    CHECK(plan.crf_quality_target);
  }
  // This clip is spatially uniform (the first-pass sigma is nearly constant
  // across frames), so qcomp has almost no leverage. It must still produce a
  // near-constant quality ladder rather than a wild one.
  int crf_min_q = crf_run.plans.front().base_q_idx;
  int crf_max_q = crf_run.plans.front().base_q_idx;
  for (const FramePlan& plan : crf_run.plans) {
    crf_min_q = std::min(crf_min_q, plan.base_q_idx);
    crf_max_q = std::max(crf_max_q, plan.base_q_idx);
  }
  std::cout << "crf 8f: q range [" << crf_min_q << ", " << crf_max_q << "]\n";
  CHECK(crf_max_q - crf_min_q <= 8);

  // qcomp off: constant quality across the clip (no buffer override).
  TwoPassConfig crf_flat = crf;
  crf_flat.rc.crf.qcomp = 0.0;
  crf_flat.vbv.buffer_frames = 100.0;
  RunResult crf_flat_run;
  CHECK(run_second_pass(pass_stats, crf_flat, eight_table, frames.size(),
                        crf_flat_run, error));
  for (std::size_t i = 1; i < crf_flat_run.plans.size(); ++i) {
    CHECK(crf_flat_run.plans[i].base_q_idx ==
          crf_flat_run.plans[0].base_q_idx);
  }

  // Determinism of the CRF path.
  RateTable crf_table = eight_table;
  RunResult crf_again;
  CHECK(run_second_pass(pass_stats, crf, crf_table, frames.size(), crf_again,
                        error));
  for (std::size_t i = 0; i < crf_run.plans.size(); ++i) {
    CHECK(same_plan(crf_run.plans[i], crf_again.plans[i]));
  }

  // Tight buffer: a low nominal rate and small bucket must force the
  // controller to raise qindex rather than underflow.
  TwoPassConfig tight = crf;
  // Let the tight-buffer test use the full qindex range: it must raise the
  // quantizer, not underflow, and the calibrated curve extrapolates past the
  // last measured anchor.
  tight.max_qindex = 255;
  tight.rc.crf.crf = 30.0;
  tight.rc.vbr.target_bitrate_bps = 0.20 * base_bps;
  tight.vbv.buffer_frames = 0.75;
  tight.vbv.initial_fill = 0.05;
  tight.vbv.max_rate_bps = 0.4 * base_bps;
  tight.vbv.underflow_margin = 0.05;
  RunResult tight_run;
  CHECK(run_second_pass(pass_stats, tight, eight_table, frames.size(), tight_run,
                        error));
  CHECK(tight_run.underflow_prevented > 0 || tight_run.unreachable > 0);
  CHECK(!tight_run.underflow);

  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
