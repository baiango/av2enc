// Film grain decoder gate.
//
// Builds real M2 streams with the existing M2 encoder, rewrites their headers
// to carry film grain syntax, and decodes with the pinned dav2d.  Four phases:
//
//   A. Grain on every frame the planner selects (all frames here).  The
//      sequence header gets film_grain_params_present = 1, a §5.14 film grain
//      OBU is inserted before the coded frame OBU, and §5.18.10.1
//      film_grain_config() is appended to each frame header.  Decoded output
//      must equal the encoder reconstruction with the spec_tables synthesis
//      reference applied, sample for sample (8-bit 4:2:0).
//
//   B. apply_grain = 0 on every frame with film_grain_params_present = 1 and
//      no film grain OBU.  Decoded output must be byte-identical to the base
//      stream (decode-invariance of the disabled path), and the base stream
//      must be identical with --filmgrain 0 and --filmgrain 1.
//
//   C. Slot reuse/update.  Unit 0 stores model A in slot 0; frame 1 applies
//      it.  Unit 1 overwrites slot 0 with model B; frame 2 applies the new
//      model (same fgm_id).  Both decoded frames are checked against the
//      reference synthesis, and the two outputs must differ.
//
// All decodes use `--strict 1 --threads 1`.  No file outside this directory is
// written except the work directory.
//
// Usage:
//   film_grain_stream_gate [--input clip.y4m] [--frames N] [--work DIR]
//                          [--dav2d PATH] [--qp Q]
#include "pipeline/grain/grain_stream.h"

#include "core/bitstream/basic_headers.h"
#include "codec/filters/tables/film_grain.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
namespace grain = av2enc::filters::grain;
namespace spec = av2enc::filters::spec;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (condition) {
    std::printf("  ok   %s\n", what.c_str());
  } else {
    std::printf("  FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;

  std::uint32_t chroma_width() const { return (width + 1U) / 2U; }
  std::uint32_t chroma_height() const { return (height + 1U) / 2U; }
};

std::uint8_t SquareSample(int x, int y, int frame, int plane) {
  const int sx = 8 + 4 * frame;
  const int sy = 8 + 2 * frame;
  const bool inside = x >= sx && x < sx + 32 && y >= sy && y < sy + 32;
  if (plane == 0) {
    if (inside) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(40 + ((lx * 37 + ly * 17) & 0x7F));
    }
    return static_cast<std::uint8_t>(96 + ((x + y) >> 3));
  }
  if (plane == 1) return static_cast<std::uint8_t>(inside ? 100 : 128);
  return static_cast<std::uint8_t>(inside ? 150 : 128);
}

std::vector<OwnedFrame> MakeSyntheticClip(int frames, int width, int height) {
  std::vector<OwnedFrame> out;
  out.reserve(static_cast<std::size_t>(frames));
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(static_cast<std::size_t>(width) *
                   static_cast<std::size_t>(height));
    frame.u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    frame.v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        frame.y[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col)] =
            SquareSample(col, row, f, 0);
      }
    }
    for (int row = 0; row < ch; ++row) {
      for (int col = 0; col < cw; ++col) {
        frame.u[static_cast<std::size_t>(row) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(col)] =
            SquareSample(col * 2, row * 2, f, 1);
        frame.v[static_cast<std::size_t>(row) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(col)] =
            SquareSample(col * 2, row * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

bool LoadClip(const std::string& path, int max_frames,
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

bool ReadY4m(const std::string& path, std::vector<OwnedFrame>& out,
             std::string& error) {
  out.clear();
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
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
    out.push_back(std::move(owned));
  }
  return true;
}

std::string FindDav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (fs::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (fs::exists(candidate)) return candidate;
  }
  return {};
}

int Run(const std::string& command) {
  std::fflush(stdout);
  return std::system(command.c_str());
}

bool WriteIvf(const fs::path& path,
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

// Mirrors the VideoConfig at src/encoder/m2/video_frame.cpp:988-996.
av2enc::video::VideoConfig MakeM2Config(std::uint32_t width, std::uint32_t height,
                                  std::uint8_t qp) {
  av2enc::video::VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = qp;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;
  return config;
}

// Mirrors the VideoFrameHeader construction in video_frame.cpp.
av2enc::video::VideoFrameHeader MakeM2Header(const av2enc::video::VideoConfig& config,
                                       std::size_t index) {
  av2enc::video::VideoFrameHeader header;
  header.keyframe = index == 0;
  header.order_hint = static_cast<std::uint32_t>(index);
  header.ref_slot = 0;
  header.refresh_frame_flags =
      index == 0 ? ((1U << config.num_ref_frames) - 1U) : 1U;
  return header;
}

// Rewrites the coded-frame OBU in `unit` with the same tile payload and an
// appended §5.18.10.1 film_grain_config().  Returns false if the layout does
// not match the M2 writer.
bool RebuildFrameHeader(std::vector<std::uint8_t>* unit,
                        const av2enc::video::VideoConfig& config,
                        const av2enc::video::VideoFrameHeader& header,
                        const grain::FilmGrainConfig& grain_config,
                        const grain::FilmGrainConfigContext& context,
                        std::string& error) {
  std::vector<grain::GrainObuView> obus;
  if (!grain::ParseTemporalUnit(*unit, &obus, &error)) return false;
  const grain::GrainObuView* frame = nullptr;
  for (const auto& obu : obus) {
    if (grain::IsFrameObuType(obu.type)) {
      frame = &obu;
      break;
    }
  }
  if (frame == nullptr) {
    error = "unit has no coded frame";
    return false;
  }
  av2enc::BitWriter header_writer;
  const std::size_t header_bits =
      av2enc::video::write_frame_header(header_writer, config, header);
  const std::size_t header_bytes = (1U + header_bits + 7U) / 8U;
  if (frame->payload.size() < header_bytes + 1U) {
    error = "frame payload shorter than its header";
    return false;
  }
  const std::span<const std::uint8_t> tile = frame->payload.subspan(header_bytes);

  av2enc::BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  av2enc::video::write_frame_header(writer, config, header);
  if (!grain::WriteFilmGrainConfig(writer, grain_config, context, &error)) {
    return false;
  }
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile.begin(), tile.end());

  const std::uint8_t type = frame->type;
  const std::vector<std::uint8_t> replacement = av2enc::basic::make_obu(
      av2enc::basic::ObuHeader{static_cast<av2enc::basic::ObuType>(type)}, payload);

  const std::size_t offset = frame->offset;
  const std::size_t size = frame->size;
  std::vector<std::uint8_t> rebuilt;
  rebuilt.reserve(unit->size() - size + replacement.size());
  rebuilt.insert(rebuilt.end(), unit->begin(),
                 unit->begin() + static_cast<std::ptrdiff_t>(offset));
  rebuilt.insert(rebuilt.end(), replacement.begin(), replacement.end());
  rebuilt.insert(rebuilt.end(),
                 unit->begin() + static_cast<std::ptrdiff_t>(offset + size),
                 unit->end());
  *unit = std::move(rebuilt);
  return true;
}

grain::GrainFrameView ViewOf(const OwnedFrame& frame) {
  grain::GrainFrameView view;
  view.y = frame.y.data();
  view.u = frame.u.data();
  view.v = frame.v.data();
  view.width = static_cast<int>(frame.width);
  view.height = static_cast<int>(frame.height);
  view.y_stride = static_cast<int>(frame.width);
  view.c_stride = static_cast<int>(frame.chroma_width());
  return view;
}

// Applies the §7.21.7 reference synthesis to a reconstruction frame.
bool ApplyReferenceGrain(const OwnedFrame& recon, const grain::GrainModel& model,
                         std::uint16_t seed, OwnedFrame* out,
                         std::string& error) {
  *out = recon;
  spec::FilmGrainModel synth;
  if (!grain::ToSynthesisModel(model, &synth)) {
    error = "cannot convert grain model";
    return false;
  }
  if (!spec::FilmGrainModelValid(synth, 1, 1, 8, 3, &error)) return false;

  const std::size_t luma = recon.y.size();
  const std::size_t chroma = recon.u.size();
  std::vector<std::uint16_t> y(luma);
  std::vector<std::uint16_t> u(chroma);
  std::vector<std::uint16_t> v(chroma);
  for (std::size_t i = 0; i < luma; ++i) y[i] = recon.y[i];
  for (std::size_t i = 0; i < chroma; ++i) {
    u[i] = recon.u[i];
    v[i] = recon.v[i];
  }
  spec::FilmGrainFrame frame;
  frame.plane[0] = {y.data(), static_cast<int>(recon.width)};
  frame.plane[1] = {u.data(), static_cast<int>(recon.chroma_width())};
  frame.plane[2] = {v.data(), static_cast<int>(recon.chroma_width())};
  frame.w = static_cast<int>(recon.width);
  frame.h = static_cast<int>(recon.height);
  frame.sub_x = 1;
  frame.sub_y = 1;
  frame.bit_depth = 8;
  frame.num_planes = 3;
  spec::ApplyFilmGrain(synth, frame, seed);
  for (std::size_t i = 0; i < luma; ++i) {
    out->y[i] = static_cast<std::uint8_t>(y[i]);
  }
  for (std::size_t i = 0; i < chroma; ++i) {
    out->u[i] = static_cast<std::uint8_t>(u[i]);
    out->v[i] = static_cast<std::uint8_t>(v[i]);
  }
  return true;
}

bool Equals(const OwnedFrame& a, const OwnedFrame& b) {
  return a.width == b.width && a.height == b.height && a.y == b.y &&
         a.u == b.u && a.v == b.v;
}

std::uint64_t DiffCount(const OwnedFrame& a, const OwnedFrame& b) {
  std::uint64_t diff = 0;
  for (std::size_t i = 0; i < a.y.size(); ++i) diff += a.y[i] != b.y[i];
  for (std::size_t i = 0; i < a.u.size(); ++i) diff += a.u[i] != b.u[i];
  for (std::size_t i = 0; i < a.v.size(); ++i) diff += a.v[i] != b.v[i];
  return diff;
}

int MaxDiff(const OwnedFrame& a, const OwnedFrame& b) {
  int worst = 0;
  auto scan = [&](const std::vector<std::uint8_t>& p,
                  const std::vector<std::uint8_t>& q) {
    for (std::size_t i = 0; i < p.size(); ++i) {
      worst = std::max(worst, std::abs(static_cast<int>(p[i]) -
                                       static_cast<int>(q[i])));
    }
  };
  scan(a.y, b.y);
  scan(a.u, b.u);
  scan(a.v, b.v);
  return worst;
}

grain::FilmGrainConfigContext GrainContext() {
  grain::FilmGrainConfigContext context;
  context.film_grain_params_present = true;
  context.single_picture_header_flag = false;
  context.immediate_output_frame = true;
  context.implicit_output_frame = false;
  return context;
}

bool Decode(const std::string& dav2d, const fs::path& ivf, const fs::path& y4m,
            int filmgrain, std::string& error) {
  const std::string command = "\"" + dav2d + "\" -q -i \"" + ivf.string() +
                              "\" -o \"" + y4m.string() +
                              "\" --muxer yuv4mpeg2 --filmgrain " +
                              std::to_string(filmgrain) +
                              " --threads 1 --strict 1";
  if (Run(command) != 0) {
    error = "dav2d failed: " + command;
    return false;
  }
  return true;
}

// A fixed two-point 4:2:0 grain model used by the slot-update phase.
grain::GrainModel FixedModel(int scaling, int lag, int shift) {
  grain::GrainModel model;
  model.chroma_scaling_from_luma = true;
  model.num_y_points = 2;
  model.point_y_value[0] = 0;
  model.point_y_value[1] = 128;
  model.point_y_scaling[0] = scaling;
  model.point_y_scaling[1] = scaling + 24;
  model.point_y_increment_bits_minus_1 = 7;
  model.point_y_scaling_bits_minus_5 = 3;
  model.grain_scaling_minus_8 = 0;
  model.ar_coeff_lag = lag;
  model.bits_per_ar_coeff_y_minus_5 = 3;
  model.bits_per_ar_coeff_cb_minus_5 = 3;
  model.bits_per_ar_coeff_cr_minus_5 = 3;
  const int positions = 2 * lag * (lag + 1);
  for (int i = 0; i < positions; ++i) {
    const int tap = ((i % 3) - 1) * 8;
    model.ar_coeffs_y[i] = tap;
    model.ar_coeffs_cb[i] = tap / 2;
    model.ar_coeffs_cr[i] = tap / 2;
  }
  model.ar_coeffs_cb[positions] = 0;
  model.ar_coeffs_cr[positions] = 0;
  model.ar_coeff_shift_minus_6 = shift;
  model.overlap_flag = true;
  model.clip_to_restricted_range = false;
  return grain::NormalizeModel(model, grain::GrainChromaIdc::k420);
}

}  // namespace
int main(int argc, char** argv) {
  std::string input;
  std::string work = "film_grain_stream_work";
  std::string dav2d;
  int frames = 3;
  int qp = 90;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](std::string* out) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      *out = argv[++i];
    };
    if (arg == "--input") {
      next(&input);
    } else if (arg == "--work") {
      next(&work);
    } else if (arg == "--dav2d") {
      next(&dav2d);
    } else if (arg == "--frames") {
      std::string value;
      next(&value);
      frames = std::atoi(value.c_str());
    } else if (arg == "--qp") {
      std::string value;
      next(&value);
      qp = std::atoi(value.c_str());
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (dav2d.empty()) dav2d = FindDav2d();
  if (dav2d.empty()) {
    std::fprintf(stderr,
                 "film_grain_stream_gate: no dav2d binary; set DAV2D=...\n");
    return 127;
  }
  std::error_code ec;
  fs::create_directories(work, ec);
  if (ec) {
    std::fprintf(stderr, "cannot create work dir %s: %s\n", work.c_str(),
                 ec.message().c_str());
    return 2;
  }

  std::vector<OwnedFrame> source;
  std::string error;
  if (!input.empty()) {
    if (!LoadClip(input, frames, source, error)) {
      std::fprintf(stderr, "input: %s\n", error.c_str());
      return 2;
    }
  } else {
    source = MakeSyntheticClip(frames, 96, 96);
  }
  if (source.size() < 2) {
    std::fprintf(stderr, "gate needs at least 2 frames\n");
    return 2;
  }
  const std::uint32_t width = source[0].width;
  const std::uint32_t height = source[0].height;
  std::printf("film_grain_stream_gate: %zu frame(s) %ux%u, qp %d\n",
              source.size(), width, height, qp);

  // --- Encode the base M2 stream. -----------------------------------------
  std::vector<av2enc::video::VideoInputFrame> video_input;
  for (const auto& frame : source) {
    av2enc::video::VideoInputFrame in;
    in.width = frame.width;
    in.height = frame.height;
    in.y = std::span<const std::uint8_t>(frame.y);
    in.u = std::span<const std::uint8_t>(frame.u);
    in.v = std::span<const std::uint8_t>(frame.v);
    video_input.push_back(in);
  }
  av2enc::video::VideoOptions options;
  options.base_q_idx = static_cast<std::uint8_t>(qp);
  av2enc::video::VideoSequenceResult sequence;
  if (!av2enc::video::encode_video_sequence(video_input, options, sequence, error)) {
    std::fprintf(stderr, "M2 encode: %s\n", error.c_str());
    return 1;
  }
  std::vector<OwnedFrame> recon;
  for (const auto& r : sequence.recon) {
    OwnedFrame frame;
    frame.width = r.width;
    frame.height = r.height;
    frame.y = r.y;
    frame.u = r.u;
    frame.v = r.v;
    recon.push_back(std::move(frame));
  }
  const std::size_t frame_count = recon.size();
  const auto config = MakeM2Config(width, height, static_cast<std::uint8_t>(qp));

  const fs::path base_ivf = fs::path(work) / "base.ivf";
  if (!WriteIvf(base_ivf, sequence.units, width, height, error)) {
    std::fprintf(stderr, "ivf: %s\n", error.c_str());
    return 1;
  }
  std::vector<OwnedFrame> base_off;
  std::vector<OwnedFrame> base_on;
  if (!Decode(dav2d, base_ivf, fs::path(work) / "base0.y4m", 0, error) ||
      !Decode(dav2d, base_ivf, fs::path(work) / "base1.y4m", 1, error) ||
      !ReadY4m((fs::path(work) / "base0.y4m").string(), base_off, error) ||
      !ReadY4m((fs::path(work) / "base1.y4m").string(), base_on, error)) {
    std::fprintf(stderr, "dav2d base decode: %s\n", error.c_str());
    return 1;
  }
  Check(base_off.size() == frame_count, "base decode produced every frame");
  Check(base_on.size() == frame_count,
        "base decode with --filmgrain 1 produced every frame");
  bool base_matches_recon = base_off.size() == frame_count;
  for (std::size_t i = 0; i < base_off.size() && i < frame_count; ++i) {
    base_matches_recon = base_matches_recon && Equals(base_off[i], recon[i]);
    base_matches_recon = base_matches_recon && Equals(base_on[i], base_off[i]);
  }
  Check(base_matches_recon,
        "base stream: decoder == M2 reconstruction, --filmgrain 0 == 1");
  if (!base_matches_recon) {
    for (std::size_t i = 0; i < base_off.size() && i < frame_count; ++i) {
      std::printf("    frame %zu: recon diff %llu (max %d), filmgrain diff %llu\n",
                  i,
                  static_cast<unsigned long long>(DiffCount(base_off[i], recon[i])),
                  MaxDiff(base_off[i], recon[i]),
                  static_cast<unsigned long long>(
                      DiffCount(base_on[i], base_off[i])));
    }
  }

  // --- Phase A: estimator-driven grain on every frame. --------------------
  std::printf("phase A: estimator + slot planner\n");
  grain::GrainPlanOptions plan_options;
  plan_options.chroma_idc = grain::GrainChromaIdc::k420;
  grain::GrainStreamPlanner planner(plan_options);
  std::vector<grain::GrainFramePlan> plans(frame_count);
  std::vector<std::vector<std::uint8_t>> grain_units = sequence.units;
  std::size_t obu_bytes = 0;
  std::size_t config_bits = 0;
  for (std::size_t i = 0; i < frame_count; ++i) {
    if (!planner.PlanFrame(ViewOf(recon[i]), static_cast<std::uint32_t>(i), 0,
                           0, &plans[i], &error)) {
      std::fprintf(stderr, "plan frame %zu: %s\n", i, error.c_str());
      return 1;
    }
    if (plans[i].apply_grain) {
      Check(grain::ValidateConfigAvailability(
                planner.slots(),
                grain::FilmGrainConfig{true, plans[i].fgm_id,
                                       plans[i].grain_seed},
                grain::GrainChromaIdc::k420, 0, 0, &error),
            "frame " + std::to_string(i) + ": model available (§7.3.8.8)");
    }
  }
  for (std::size_t i = 0; i < frame_count; ++i) {
    if (i == 0) {
      if (!grain::PatchSequenceHeaderFilmGrainPresent(&grain_units[i], true,
                                                      &error)) {
        std::fprintf(stderr, "patch sequence header: %s\n", error.c_str());
        return 1;
      }
    }
    const grain::FilmGrainConfig cfg{plans[i].apply_grain, plans[i].fgm_id,
                                     plans[i].grain_seed};
    config_bits += grain::FilmGrainConfigBits(cfg, GrainContext());
    if (plans[i].emit_model) {
      const auto obu = grain::MakeFilmGrainObu(plans[i].obu, &error);
      if (obu.empty()) {
        std::fprintf(stderr, "make grain OBU: %s\n", error.c_str());
        return 1;
      }
      obu_bytes += obu.size();
      std::vector<std::uint8_t> with_obu;
      if (!grain::InsertFilmGrainObu(grain_units[i], obu, &with_obu, &error)) {
        std::fprintf(stderr, "insert grain OBU: %s\n", error.c_str());
        return 1;
      }
      grain_units[i] = std::move(with_obu);
    }
    if (!RebuildFrameHeader(&grain_units[i], config, MakeM2Header(config, i),
                            cfg, GrainContext(), error)) {
      std::fprintf(stderr, "rebuild frame %zu: %s\n", i, error.c_str());
      return 1;
    }
  }
  std::printf("  grain stream assembled\n");
  bool fgpp = false;
  Check(grain::ReadSequenceHeaderFilmGrainPresent(grain_units[0], &fgpp,
                                                  &error) &&
            fgpp,
        "sequence header film_grain_params_present == 1");

  const fs::path grain_ivf = fs::path(work) / "grain.ivf";
  if (!WriteIvf(grain_ivf, grain_units, width, height, error) ||
      !Decode(dav2d, grain_ivf, fs::path(work) / "grain.y4m", 1, error)) {
    std::fprintf(stderr, "dav2d grain decode: %s\n", error.c_str());
    return 1;
  }
  std::vector<OwnedFrame> decoded;
  if (!ReadY4m((fs::path(work) / "grain.y4m").string(), decoded, error)) {
    std::fprintf(stderr, "read grain y4m: %s\n", error.c_str());
    return 1;
  }
  Check(decoded.size() == frame_count, "grain decode produced every frame");
  if (decoded.size() == frame_count) {
    for (std::size_t i = 0; i < frame_count; ++i) {
      OwnedFrame expected = recon[i];
      if (plans[i].apply_grain) {
        if (!ApplyReferenceGrain(recon[i], plans[i].model, plans[i].grain_seed,
                                 &expected, error)) {
          std::fprintf(stderr, "reference grain frame %zu: %s\n", i,
                       error.c_str());
          return 1;
        }
      }
      const std::uint64_t diff = DiffCount(decoded[i], expected);
      Check(diff == 0, "frame " + std::to_string(i) +
                           ": decoder == reference synthesis (fgm_id " +
                           std::to_string(plans[i].fgm_id) + ", seed " +
                           std::to_string(plans[i].grain_seed) + ")");
      if (diff != 0) {
        std::printf("    diff %llu samples, max |delta| %d\n",
                    static_cast<unsigned long long>(diff),
                    MaxDiff(decoded[i], expected));
      }
      if (plans[i].apply_grain) {
        const std::uint64_t changed = DiffCount(expected, recon[i]);
        Check(changed > 0, "frame " + std::to_string(i) +
                               ": grain changed the output samples");
      }
    }
  }

  // --- Phase B: apply_grain = 0 decode invariance. ------------------------
  std::printf("phase B: apply_grain = 0 decode invariance\n");
  std::vector<std::vector<std::uint8_t>> disabled_units = sequence.units;
  bool patch_ok = true;
  if (!grain::PatchSequenceHeaderFilmGrainPresent(&disabled_units[0], true,
                                                  &error)) {
    patch_ok = false;
  }
  grain::FilmGrainConfig off{false, 0, 0};
  for (std::size_t i = 0; i < frame_count && patch_ok; ++i) {
    patch_ok = RebuildFrameHeader(&disabled_units[i], config,
                                  MakeM2Header(config, i), off, GrainContext(),
                                  error);
  }
  Check(patch_ok, "apply_grain=0 stream assembled");
  const fs::path disabled_ivf = fs::path(work) / "disabled.ivf";
  std::vector<OwnedFrame> disabled_decoded;
  bool disabled_ok =
      patch_ok && WriteIvf(disabled_ivf, disabled_units, width, height, error) &&
      Decode(dav2d, disabled_ivf, fs::path(work) / "disabled.y4m", 1, error) &&
      ReadY4m((fs::path(work) / "disabled.y4m").string(), disabled_decoded,
              error);
  Check(disabled_ok, "dav2d decoded the apply_grain=0 stream");
  if (disabled_ok && disabled_decoded.size() == frame_count) {
    std::uint64_t diff = 0;
    for (std::size_t i = 0; i < frame_count; ++i) {
      diff += DiffCount(disabled_decoded[i], base_off[i]);
    }
    Check(diff == 0,
          "apply_grain=0 decode is byte-identical to the base stream");
  }

  // --- Phase C: slot reuse/update across coded frame units. --------------
  std::printf("phase C: film grain slot update (same fgm_id)\n");
  if (frame_count < 3) {
    std::printf("  skipped: needs at least 3 frames\n");
  } else {
  const auto model_a = FixedModel(32, 2, 2);
  const auto model_b = FixedModel(96, 1, 2);
  const std::uint16_t seed_a = 0x1111;
  const std::uint16_t seed_b = 0x2222;
  std::vector<std::vector<std::uint8_t>> update_units = sequence.units;
  bool update_ok = grain::PatchSequenceHeaderFilmGrainPresent(&update_units[0],
                                                              true, &error);
  // Unit 0 stores model A in slot 0; frame 0 applies it.
  grain::FilmGrainObu obu_a;
  obu_a.update_flags = 0x01;
  obu_a.chroma_idc = grain::GrainChromaIdc::k420;
  obu_a.models[0] = model_a;
  const auto bytes_a = grain::MakeFilmGrainObu(obu_a, &error);
  std::vector<std::uint8_t> mixed;
  update_ok = update_ok && !bytes_a.empty() &&
              grain::InsertFilmGrainObu(update_units[0], bytes_a, &mixed,
                                        &error);
  update_units[0] = std::move(mixed);
  update_ok =
      update_ok && RebuildFrameHeader(&update_units[0], config,
                                      MakeM2Header(config, 0),
                                      grain::FilmGrainConfig{true, 0, seed_a},
                                      GrainContext(), error);
  // Unit 1 carries no model: frame 1 reuses slot 0 (model A).
  update_ok =
      update_ok && RebuildFrameHeader(&update_units[1], config,
                                      MakeM2Header(config, 1),
                                      grain::FilmGrainConfig{true, 0, seed_b},
                                      GrainContext(), error);
  // Unit 2 overwrites slot 0 with model B; frame 2 applies the new model.
  grain::FilmGrainObu obu_b;
  obu_b.update_flags = 0x01;
  obu_b.chroma_idc = grain::GrainChromaIdc::k420;
  obu_b.models[0] = model_b;
  const auto bytes_b = grain::MakeFilmGrainObu(obu_b, &error);
  update_ok = update_ok && !bytes_b.empty() &&
              grain::InsertFilmGrainObu(update_units[2], bytes_b, &mixed,
                                        &error);
  update_units[2] = std::move(mixed);
  update_ok =
      update_ok && RebuildFrameHeader(&update_units[2], config,
                                      MakeM2Header(config, 2),
                                      grain::FilmGrainConfig{true, 0, seed_b},
                                      GrainContext(), error);
  Check(update_ok, "slot-update stream assembled");

  OwnedFrame expected_a0;
  OwnedFrame expected_a1;
  OwnedFrame expected_b2;
  update_ok = update_ok &&
              ApplyReferenceGrain(recon[0], model_a, seed_a, &expected_a0,
                                  error) &&
              ApplyReferenceGrain(recon[1], model_a, seed_b, &expected_a1,
                                  error) &&
              ApplyReferenceGrain(recon[2], model_b, seed_b, &expected_b2,
                                  error);
  Check(update_ok, "reference synthesis for the slot-update stream");
  Check(DiffCount(expected_a1, expected_b2) > 0,
        "models A and B produce different grain");

  if (update_ok) {
    const fs::path update_ivf = fs::path(work) / "update.ivf";
    std::vector<OwnedFrame> update_decoded;
    const bool decoded_ok =
        WriteIvf(update_ivf, update_units, width, height, error) &&
        Decode(dav2d, update_ivf, fs::path(work) / "update.y4m", 1, error) &&
        ReadY4m((fs::path(work) / "update.y4m").string(), update_decoded,
                error);
    Check(decoded_ok, "dav2d decoded the slot-update stream");
    if (decoded_ok && update_decoded.size() == frame_count) {
      Check(DiffCount(update_decoded[0], expected_a0) == 0,
            "frame 0 applies slot 0 model A (fgm_id 0)");
      Check(DiffCount(update_decoded[1], expected_a1) == 0,
            "frame 1 reuses slot 0 model A without a new OBU");
      Check(DiffCount(update_decoded[2], expected_b2) == 0,
            "frame 2 applies the updated slot 0 model B (same fgm_id)");
    }
  }
  }  // frame_count >= 3

  // --- Rate accounting. ---------------------------------------------------
  const std::uint64_t base_bytes = [&]() {
    std::uint64_t total = 0;
    for (const auto& unit : sequence.units) total += unit.size();
    return total;
  }();
  std::printf(
      "rate: base %llu B; film grain OBUs %zu B (%.2f%%), config %zu bits "
      "(%.2f bits/frame), total grain syntax %zu B\n",
      static_cast<unsigned long long>(base_bytes), obu_bytes,
      100.0 * static_cast<double>(obu_bytes) /
          static_cast<double>(std::max<std::uint64_t>(base_bytes, 1U)),
      config_bits,
      static_cast<double>(config_bits) / static_cast<double>(frame_count),
      (obu_bytes + (config_bits + 7U) / 8U));
  for (std::size_t i = 0; i < frame_count; ++i) {
    const std::size_t model_bits =
        plans[i].emit_model
            ? grain::ModelBits(plans[i].model, grain::GrainChromaIdc::k420)
            : 0;
    std::printf(
        "  frame %zu: apply=%d fgm_id=%d seed=0x%04x emit=%d model=%zu bits, "
        "activity=%.2f\n",
        i, plans[i].apply_grain ? 1 : 0, plans[i].fgm_id, plans[i].grain_seed,
        plans[i].emit_model ? 1 : 0, model_bits, plans[i].activity);
  }

  // --- Negative control: a missing slot must be rejected. -----------------
  grain::GrainSlotState empty_state;
  const bool rejected = !grain::ValidateConfigAvailability(
      empty_state, grain::FilmGrainConfig{true, 3, 0x1234},
      grain::GrainChromaIdc::k420, 0, 0, &error);
  Check(rejected, "missing FilmGrainPresent[fgm_id] rejected (§7.3.8.8)");

  std::printf("film_grain_stream_gate: %s (%d failure(s))\n",
              g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
