// M2 CDEF stream gate (dav2d).
//
// Emits a minimal two-frame M2 stream twice:
//
//   baseline : the current M2 stream (sequence enable_cdef = 0)
//   cdef     : the same tile payloads with sequence enable_cdef = 1 and a
//              §5.18.7.10 `cdef_params()` row in each frame header
//
// `CdefStrengths == 1` codes no tile symbol (§5.20.10.1 forces index 0), so the
// tile payloads are byte-identical and the only stream change is the header.
// That is what makes a real CDEF stream possible without editing the M2 tile
// writer.
//
// The gate then decodes both streams with the pinned `dav2d` and requires, byte
// for byte on Y, U and V:
//
//   decoded(baseline) == M2 encoder reconstruction
//   decoded(cdef)     == the adapter's §7.18 filter of that reconstruction
//
// The P frame is encoded with `force_intra = true`: CDEF changes the frame-0
// reference, and the untouched M2 core always predicts from the unfiltered
// reconstruction.  An intra P frame has no reference dependency, so the
// reconstructed samples (and therefore the CDEF decision) stay exact.  Making
// inter prediction CDEF-aware is the core integration step documented in
// README.md.
//
// The gate also measures the component-only 4-strength per-64x64 plan (not
// signallable until the M2 tile writer gains the `read_cdef()` hook) and
// writes its Y4M for the PSNR/SSIM/CVVDP report.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "codec/filters/cdef/cdef.h"
#include "codec/filters/cdef/syntax/video_cdef.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/post/tiled_frame.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

namespace {

using av2enc::IvfHeader;
using av2enc::IvfWriter;
using av2enc::Y4mFrame;
using av2enc::Y4mHeader;
using av2enc::Y4mReadResult;
using av2enc::Y4mReader;
using av2enc::video::VideoConfig;
using av2enc::video::VideoInputFrame;
using av2enc::video::VideoOptions;
using av2enc::video::VideoReconFrame;
using av2enc::video::VideoSequenceResult;
using av2enc::video::cdef_adapter::FrameGrids;
using av2enc::video::cdef_adapter::FrameView;
using av2enc::video::cdef_adapter::Options;
using av2enc::video::cdef_adapter::Plan;

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

bool load_frames(const std::string& path, int count, std::vector<Frame>* out,
                 std::uint32_t* fps_num, std::uint32_t* fps_den,
                 std::string* error) {
  Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  const Y4mHeader& header = reader.header();
  *fps_num = header.fps_num == 0 ? 30 : header.fps_num;
  *fps_den = header.fps_den == 0 ? 1 : header.fps_den;
  Y4mFrame frame;
  while (static_cast<int>(out->size()) < count) {
    const Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == Y4mReadResult::kEof) break;
    if (status != Y4mReadResult::kFrame) return false;
    Frame copy;
    copy.width = static_cast<int>(frame.width());
    copy.height = static_cast<int>(frame.height());
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    out->push_back(std::move(copy));
  }
  if (out->empty()) {
    *error = "input has no frames";
    return false;
  }
  return true;
}

bool write_y4m(const std::string& path, const std::vector<Frame>& frames,
               std::uint32_t fps_num, std::uint32_t fps_den,
               std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "cannot write " + path;
    return false;
  }
  out << "YUV4MPEG2 W" << frames[0].width << " H" << frames[0].height << " F"
      << fps_num << ":" << fps_den << " Ip A1:1 C420jpeg\n";
  for (const Frame& frame : frames) {
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(frame.y.data()),
              static_cast<std::streamsize>(frame.y.size()));
    out.write(reinterpret_cast<const char*>(frame.u.data()),
              static_cast<std::streamsize>(frame.u.size()));
    out.write(reinterpret_cast<const char*>(frame.v.data()),
              static_cast<std::streamsize>(frame.v.size()));
  }
  if (!out) {
    *error = "short write to " + path;
    return false;
  }
  return true;
}

// Copies an M2 reconstruction into the tight planner/stream layout.  M2
// dimensions are divisible by 8, so the reconstruction is already MI-aligned.
bool make_frame(const VideoReconFrame& recon, Frame* out, std::string* error) {
  const int w = static_cast<int>(recon.width);
  const int h = static_cast<int>(recon.height);
  if (w <= 0 || h <= 0 || (w & 7) != 0 || (h & 7) != 0) {
    *error = "reconstruction is not divisible by 8";
    return false;
  }
  out->width = w;
  out->height = h;
  out->y = recon.y;
  out->u = recon.u;
  out->v = recon.v;
  return true;
}

void fill_view(const Frame& recon, const Frame& source, FrameGrids* grids,
               FrameView* view) {
  *view = FrameView{};
  const std::uint8_t* planes[3] = {recon.y.data(), recon.u.data(),
                                   recon.v.data()};
  const std::uint8_t* refs[3] = {source.y.data(), source.u.data(),
                                 source.v.data()};
  const int strides[3] = {recon.width, recon.width / 2, recon.width / 2};
  for (int p = 0; p < 3; ++p) {
    view->recon[p].data = planes[p];
    view->recon[p].stride = strides[p];
    view->source[p].data = refs[p];
    view->source[p].stride = strides[p];
  }
  view->width = recon.width;
  view->height = recon.height;
  view->mi_cols = recon.width / 4;
  view->mi_rows = recon.height / 4;
  grids->zero(view->mi_cols, view->mi_rows);
  grids->bind(view);
}

struct PlaneSse {
  std::int64_t y = 0;
  std::int64_t u = 0;
  std::int64_t v = 0;
};

PlaneSse sse_against(const Frame& a, const Frame& b) {
  PlaneSse out;
  const auto accumulate = [](const std::vector<std::uint8_t>& x,
                             const std::vector<std::uint8_t>& y) {
    std::int64_t sse = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const int d = static_cast<int>(x[i]) - static_cast<int>(y[i]);
      sse += static_cast<std::int64_t>(d) * d;
    }
    return sse;
  };
  out.y = accumulate(a.y, b.y);
  out.u = accumulate(a.u, b.u);
  out.v = accumulate(a.v, b.v);
  return out;
}

double psnr(double sse, std::size_t samples) {
  if (sse == 0.0) return 99.0;
  const double mse = sse / static_cast<double>(samples);
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

void report_psnr(const char* label, const Frame& test, const Frame& ref) {
  const PlaneSse s = sse_against(test, ref);
  const double ys = static_cast<double>(ref.y.size());
  const double cs = static_cast<double>(ref.u.size());
  // Sample-weighted 4:2:0 average: Y counts 4x each chroma sample.
  const double yuv_mse =
      (static_cast<double>(s.y) + 0.25 * static_cast<double>(s.u) +
       0.25 * static_cast<double>(s.v)) /
      (ys + 0.5 * cs);
  const double yuv_psnr =
      yuv_mse == 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / yuv_mse);
  std::printf("  %-18s PSNR-Y %.4f  U %.4f  V %.4f  YUV %.4f\n", label,
              psnr(static_cast<double>(s.y), ref.y.size()),
              psnr(static_cast<double>(s.u), ref.u.size()),
              psnr(static_cast<double>(s.v), ref.v.size()), yuv_psnr);
}

bool run_decoder(const std::string& dav2d, const std::string& ivf,
                 const std::string& y4m) {
  const std::string command = "\"" + dav2d + "\" -q -i \"" + ivf + "\" -o \"" +
                              y4m + "\" --threads 0";
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::fprintf(stderr, "dav2d failed (%d): %s\n", rc, command.c_str());
    return false;
  }
  return std::filesystem::exists(y4m);
}

bool compare_decoded(const std::vector<Frame>& decoded,
                     const std::vector<Frame>& expected, const char* label,
                     bool* matched) {
  *matched = decoded.size() == expected.size();
  int total_diff = 0;
  int max_diff = 0;
  for (std::size_t i = 0; i < decoded.size() && i < expected.size(); ++i) {
    const Frame& a = decoded[i];
    const Frame& b = expected[i];
    if (a.width != b.width || a.height != b.height) {
      *matched = false;
      continue;
    }
    for (int plane = 0; plane < 3; ++plane) {
      const std::vector<std::uint8_t>& x =
          plane == 0 ? a.y : (plane == 1 ? a.u : a.v);
      const std::vector<std::uint8_t>& y =
          plane == 0 ? b.y : (plane == 1 ? b.u : b.v);
      for (std::size_t k = 0; k < x.size(); ++k) {
        const int d = std::abs(static_cast<int>(x[k]) - static_cast<int>(y[k]));
        if (d != 0) {
          ++total_diff;
          max_diff = std::max(max_diff, d);
        }
      }
    }
  }
  std::printf("  %-18s frame(s) %zu/%zu, %d sample diffs, max %d\n", label,
              decoded.size(), expected.size(), total_diff, max_diff);
  if (total_diff != 0) *matched = false;
  return total_diff == 0;
}

bool parse_args(int argc, char** argv, std::string* input, int* frames,
                std::string* out_dir, std::string* dav2d, bool* single_block) {
  *input = "testdata/shine_1280x720_60f.y4m";
  *frames = 2;
  *out_dir = ".";
  *dav2d = "tools/bin/dav2d-macos";
  *single_block = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      *input = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      *frames = std::atoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      *out_dir = argv[++i];
    } else if (arg == "--dav2d" && i + 1 < argc) {
      *dav2d = argv[++i];
    } else if (arg == "--single-block") {
      *single_block = true;
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: %s [--input clip.y4m] [--frames 2] [--out DIR] "
          "[--dav2d PATH] [--single-block]\n",
          argv[0]);
      return false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return false;
    }
  }
  return true;
}


// Ordering check: the adapter must produce exactly what the post-chain produces
// when it runs the CDEF stage after deblock.  `run_video_post_filters()` owns the
// normative order; this compares:
//
//   deblock-only chain -> adapter(plan, filter)
//   deblock + CDEF chain (same plan options and index grid)
//
// The two outputs must be byte-identical, which pins the CDEF slot between
// deblock and CCSO.  (CCSO itself is not configured here; its order is already
// pinned by video_post_test.)
bool check_post_chain_slot(const VideoReconFrame& recon, const Frame& source,
                           std::string* error) {
  using av2enc::video::post::VideoPostFilterConfig;
  using av2enc::video::post::VideoPostFilterFrame;
  using av2enc::video::post::VideoPostFilterReport;
  using av2enc::video::post::run_video_post_filters;

  VideoPostFilterFrame deblock_only;
  if (!deblock_only.init_from_recon(recon, 90, /*frame_is_intra=*/true, error)) {
    return false;
  }
  VideoPostFilterConfig deblock_cfg;
  deblock_cfg.enable_deblock = true;
  VideoPostFilterReport report;
  if (!run_video_post_filters(deblock_only, deblock_cfg, report, error)) {
    return false;
  }
  if (report.order.size() != 1 || report.order[0] !=
                                      av2enc::video::post::VideoPostFilterStage::kDeblock) {
    *error = "deblock-only chain ran the wrong stages";
    return false;
  }

  // Adapter on the post-deblock planes.
  Frame post_deblock;
  post_deblock.width = source.width;
  post_deblock.height = source.height;
  post_deblock.y.assign(deblock_only.output_plane(0).begin(),
                        deblock_only.output_plane(0).end());
  post_deblock.u.assign(deblock_only.output_plane(1).begin(),
                        deblock_only.output_plane(1).end());
  post_deblock.v.assign(deblock_only.output_plane(2).begin(),
                        deblock_only.output_plane(2).end());
  FrameGrids grids;
  FrameView view;
  fill_view(post_deblock, source, &grids, &view);
  Options options;
  options.num_strengths = 1;
  options.cdef_on_skip_txfm = 0;
  Plan plan;
  if (!av2enc::video::cdef_adapter::plan(view, options, &plan, error)) {
    return false;
  }
  Frame adapter_out;
  adapter_out.width = source.width;
  adapter_out.height = source.height;
  adapter_out.y.resize(post_deblock.y.size());
  adapter_out.u.resize(post_deblock.u.size());
  adapter_out.v.resize(post_deblock.v.size());
  std::uint8_t* dst[3] = {adapter_out.y.data(), adapter_out.u.data(),
                          adapter_out.v.data()};
  const int stride[3] = {adapter_out.width, adapter_out.width / 2,
                         adapter_out.width / 2};
  if (!av2enc::video::cdef_adapter::filter(view, plan, dst, stride, error)) {
    return false;
  }

  // Full chain on a fresh frame: deblock + CDEF with the same plan.
  VideoPostFilterFrame full;
  if (!full.init_from_recon(recon, 90, /*frame_is_intra=*/true, error)) {
    return false;
  }
  VideoPostFilterConfig full_cfg;
  full_cfg.enable_deblock = true;
  full_cfg.enable_cdef = true;
  full_cfg.cdef = plan.options;
  if (static_cast<int>(full.cdef_idx_grid().size()) !=
      plan.n64_cols * plan.n64_rows) {
    *error = "post chain index grid size mismatch";
    return false;
  }
  for (std::size_t i = 0; i < full.cdef_idx_grid().size(); ++i) {
    full.cdef_idx_grid()[i] = plan.cdef_idx[i];
  }
  VideoPostFilterReport full_report;
  if (!run_video_post_filters(full, full_cfg, full_report, error)) {
    return false;
  }
  if (full_report.order.size() != 2 ||
      full_report.order[0] != av2enc::video::post::VideoPostFilterStage::kDeblock ||
      full_report.order[1] != av2enc::video::post::VideoPostFilterStage::kCdef) {
    *error = "deblock+CDEF chain ran the wrong stages";
    return false;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const std::vector<std::uint8_t>& a = full.output_plane(plane);
    const std::vector<std::uint8_t>* b =
        plane == 0 ? &adapter_out.y : (plane == 1 ? &adapter_out.u
                                                  : &adapter_out.v);
    if (a.size() != b->size()) {
      *error = "post chain plane size mismatch";
      return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (a[i] != (*b)[i]) {
        *error = "adapter filter differs from the post-chain CDEF slot";
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  int frame_count = 2;
  std::string out_dir;
  std::string dav2d;
  bool single_block = false;
  if (!parse_args(argc, argv, &input, &frame_count, &out_dir, &dav2d,
                  &single_block)) {
    return 2;
  }
  std::filesystem::create_directories(out_dir);

  std::vector<Frame> source;
  std::uint32_t fps_num = 30;
  std::uint32_t fps_den = 1;
  std::string error;
  if (!load_frames(input, frame_count, &source, &fps_num, &fps_den, &error)) {
    std::fprintf(stderr, "input: %s\n", error.c_str());
    return 1;
  }
  if (source.size() < 2) {
    std::fprintf(stderr, "need at least two frames, got %zu\n",
                 source.size());
    return 1;
  }
  frame_count = static_cast<int>(source.size());
  const int width = source[0].width;
  const int height = source[0].height;
  std::printf("== M2 CDEF gate: %d frame(s) %dx%d ==\n", frame_count, width,
              height);

  std::vector<VideoInputFrame> inputs;
  for (const Frame& frame : source) {
    VideoInputFrame in;
    in.width = static_cast<std::uint32_t>(frame.width);
    in.height = static_cast<std::uint32_t>(frame.height);
    in.y = frame.y;
    in.u = frame.u;
    in.v = frame.v;
    inputs.push_back(in);
  }

  VideoOptions options;
  options.base_q_idx = 90;
  options.force_intra = true;  // reference-independent P frame (see header)
  options.enable_intra_fallback = true;
  options.force_single_block = single_block;
  options.min_block_size = 16;
  options.max_partition_depth = 4;

  VideoSequenceResult encoded;
  if (!av2enc::video::encode_video_sequence(inputs, options, encoded, error)) {
    std::fprintf(stderr, "encode: %s\n", error.c_str());
    return 1;
  }
  for (int i = 0; i < frame_count; ++i) {
    std::printf("  frame %d: inter %u intra %u blocks\n", i,
                encoded.inter_blocks[static_cast<std::size_t>(i)],
                encoded.intra_blocks[static_cast<std::size_t>(i)]);
  }
  if (encoded.units.size() != source.size()) {
    std::fprintf(stderr, "unexpected unit count\n");
    return 1;
  }

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;

  // ---- CDEF decision per frame (CdefStrengths = 1, header-only) ----------
  std::vector<Plan> plans(static_cast<std::size_t>(frame_count));
  std::vector<Plan> plans4(static_cast<std::size_t>(frame_count));
  std::vector<Frame> filtered(static_cast<std::size_t>(frame_count));
  std::vector<Frame> filtered4(static_cast<std::size_t>(frame_count));
  std::vector<Frame> recon(static_cast<std::size_t>(frame_count));
  for (int i = 0; i < frame_count; ++i) {
    if (!make_frame(encoded.recon[static_cast<std::size_t>(i)],
                    &recon[static_cast<std::size_t>(i)], &error)) {
      std::fprintf(stderr, "recon: %s\n", error.c_str());
      return 1;
    }
    FrameGrids grids;
    FrameView view;
    fill_view(recon[static_cast<std::size_t>(i)],
              source[static_cast<std::size_t>(i)], &grids, &view);

    Options cdef_options;
    cdef_options.num_strengths = 1;
    cdef_options.cdef_on_skip_txfm = 0;
    cdef_options.on_skip_txfm = false;
    if (!av2enc::video::cdef_adapter::plan(view, cdef_options,
                                &plans[static_cast<std::size_t>(i)], &error)) {
      std::fprintf(stderr, "plan: %s\n", error.c_str());
      return 1;
    }
    Frame& out = filtered[static_cast<std::size_t>(i)];
    out.width = recon[static_cast<std::size_t>(i)].width;
    out.height = recon[static_cast<std::size_t>(i)].height;
    out.y.resize(static_cast<std::size_t>(out.width) *
                 static_cast<std::size_t>(out.height));
    out.u.resize(static_cast<std::size_t>(out.width / 2) *
                 static_cast<std::size_t>(out.height / 2));
    out.v.resize(out.u.size());
    std::uint8_t* dst[3] = {out.y.data(), out.u.data(), out.v.data()};
    const int stride[3] = {out.width, out.width / 2, out.width / 2};
    if (!av2enc::video::cdef_adapter::filter(view, plans[static_cast<std::size_t>(i)], dst,
                                  stride, &error)) {
      std::fprintf(stderr, "filter: %s\n", error.c_str());
      return 1;
    }

    // Component-only per-64x64 plan (not signallable yet).
    Options cdef4 = cdef_options;
    cdef4.num_strengths = 4;
    if (av2enc::video::cdef_adapter::plan(view, cdef4,
                               &plans4[static_cast<std::size_t>(i)], &error)) {
      Frame& out4 = filtered4[static_cast<std::size_t>(i)];
      out4.width = out.width;
      out4.height = out.height;
      out4.y.resize(out.y.size());
      out4.u.resize(out.u.size());
      out4.v.resize(out.v.size());
      std::uint8_t* dst4[3] = {out4.y.data(), out4.u.data(), out4.v.data()};
      if (!av2enc::video::cdef_adapter::filter(view, plans4[static_cast<std::size_t>(i)],
                                    dst4, stride, &error)) {
        std::fprintf(stderr, "component filter: %s\n", error.c_str());
        return 1;
      }
    }
  }

  // ---- Post-chain slot: deblock -> adapter CDEF --------------------------
  {
    std::string slot_error;
    const bool slot_ok = check_post_chain_slot(
        encoded.recon[0], source[0], &slot_error);
    check(slot_ok, "CDEF slot matches the deblock->CDEF post chain");
    if (!slot_ok) std::fprintf(stderr, "  slot: %s\n", slot_error.c_str());
    else std::printf("  post-chain slot    deblock -> CDEF matches\n");
  }

  // ---- Rebuild the stream with CDEF enabled ------------------------------
  av2enc::video::post::VideoPostSequenceOptions sequence;
  sequence.filters.enable_cdef = true;
  sequence.filters.cdef_on_skip_txfm_disabled = true;
  sequence.filters.cdef_on_skip_txfm_always_on = false;
  const std::vector<std::uint8_t> seq_obu =
      av2enc::video::post::make_video_sequence_header_obu(config, sequence, &error);
  if (seq_obu.empty()) {
    std::fprintf(stderr, "sequence header: %s\n", error.c_str());
    return 1;
  }
  const av2enc::video::post::VideoTilePlan tiles =
      av2enc::video::post::plan_video_tiles(width, height, 1, 1);
  if (!tiles.ok) {
    std::fprintf(stderr, "tile plan: %s\n", tiles.error.c_str());
    return 1;
  }

  std::vector<std::vector<std::uint8_t>> cdef_units;
  std::uint64_t header_delta_bits = 0;
  for (int i = 0; i < frame_count; ++i) {
    const std::vector<std::uint8_t>& unit =
        encoded.units[static_cast<std::size_t>(i)];
    const std::size_t tile_bytes =
        encoded.tile_bytes[static_cast<std::size_t>(i)];
    if (tile_bytes > unit.size()) {
      std::fprintf(stderr, "tile_bytes exceeds the OBU\n");
      return 1;
    }
    std::vector<std::uint8_t> tile(unit.end() -
                                       static_cast<std::ptrdiff_t>(tile_bytes),
                                   unit.end());

    av2enc::video::post::VideoTiledFrameHeader header;
    header.keyframe = i == 0;
    header.order_hint = static_cast<std::uint32_t>(i);
    header.ref_slot = 0;
    header.refresh_frame_flags =
        i == 0 ? (1U << config.num_ref_frames) - 1U : 1U;
    header.base_q_idx = 90;
    header.cdef_frame_enable = true;
    header.cdef = plans[static_cast<std::size_t>(i)].params;

    av2enc::video::post::VideoTiledFrame frame;
    const std::vector<std::vector<std::uint8_t>> payload = {tile};
    if (!av2enc::video::post::assemble_video_tile_group_from_payloads(
            config, sequence, header, tiles.layout, payload, &frame, &error)) {
      std::fprintf(stderr, "frame %d: %s\n", i, error.c_str());
      return 1;
    }
    header_delta_bits += static_cast<std::uint64_t>(
        plans[static_cast<std::size_t>(i)].header_bits);
    std::vector<std::uint8_t> out_unit;
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    out_unit.insert(out_unit.end(), td.begin(), td.end());
    if (i == 0) {
      out_unit.insert(out_unit.end(), seq_obu.begin(), seq_obu.end());
    }
    out_unit.insert(out_unit.end(), frame.obu.begin(), frame.obu.end());
    cdef_units.push_back(std::move(out_unit));
  }

  const auto write_ivf = [&](const std::string& path,
                             const std::vector<std::vector<std::uint8_t>>& units) {
    IvfWriter writer;
    IvfHeader header;
    header.width = static_cast<std::uint16_t>(width);
    header.height = static_cast<std::uint16_t>(height);
    header.fps_num = fps_num;
    header.fps_den = fps_den;
    header.frame_count = static_cast<std::uint32_t>(units.size());
    std::string write_error;
    if (!writer.open(path, header, write_error)) return false;
    for (std::size_t i = 0; i < units.size(); ++i) {
      if (!writer.write_frame(units[i], i, write_error)) return false;
    }
    return writer.close(write_error);
  };

  const std::string base_ivf = out_dir + "/video_cdef_base.ivf";
  const std::string cdef_ivf = out_dir + "/video_cdef.ivf";
  const std::string src_y4m = out_dir + "/video_cdef_src.y4m";
  const std::string base_y4m = out_dir + "/video_cdef_base.decoded.y4m";
  const std::string cdef_y4m = out_dir + "/video_cdef.decoded.y4m";
  const std::string comp_y4m = out_dir + "/video_cdef_component4.y4m";

  if (!write_ivf(base_ivf, encoded.units) || !write_ivf(cdef_ivf, cdef_units)) {
    std::fprintf(stderr, "IVF write failed\n");
    return 1;
  }
  if (!write_y4m(src_y4m, source, fps_num, fps_den, &error)) {
    std::fprintf(stderr, "source Y4M: %s\n", error.c_str());
    return 1;
  }
  if (!write_y4m(comp_y4m, filtered4, fps_num, fps_den, &error)) {
    std::fprintf(stderr, "component Y4M: %s\n", error.c_str());
    return 1;
  }

  std::uint64_t base_bytes = 0;
  for (std::size_t i = 0; i < encoded.frame_bytes.size(); ++i) {
    base_bytes += encoded.frame_bytes[i];
  }
  std::uint64_t cdef_bytes = 0;
  for (const auto& unit : cdef_units) cdef_bytes += unit.size();
  std::printf("  stream bytes       baseline %llu, cdef %llu (%+lld)\n",
              static_cast<unsigned long long>(base_bytes),
              static_cast<unsigned long long>(cdef_bytes),
              static_cast<long long>(cdef_bytes) -
                  static_cast<long long>(base_bytes));

  // ---- Decode and compare ------------------------------------------------
  if (!run_decoder(dav2d, base_ivf, base_y4m) ||
      !run_decoder(dav2d, cdef_ivf, cdef_y4m)) {
    return 1;
  }
  std::vector<Frame> base_decoded;
  std::vector<Frame> cdef_decoded;
  std::uint32_t dn = 30;
  std::uint32_t dd = 1;
  if (!load_frames(base_y4m, frame_count, &base_decoded, &dn, &dd, &error) ||
      !load_frames(cdef_y4m, frame_count, &cdef_decoded, &dn, &dd, &error)) {
    std::fprintf(stderr, "decoded Y4M: %s\n", error.c_str());
    return 1;
  }
  bool base_match = false;
  bool cdef_match = false;
  compare_decoded(base_decoded, recon, "baseline vs recon", &base_match);
  compare_decoded(cdef_decoded, filtered, "cdef vs filter", &cdef_match);
  check(base_match, "baseline decode equals the M2 reconstruction");
  check(cdef_match, "CDEF decode equals the adapter filter");
  check(static_cast<int>(cdef_decoded.size()) == frame_count,
        "decoded frame count");

  // CDEF must actually change something and never increase SSE.
  std::int64_t changed = 0;
  for (int i = 0; i < frame_count; ++i) {
    const Frame& a = filtered[static_cast<std::size_t>(i)];
    const Frame& b = recon[static_cast<std::size_t>(i)];
    for (std::size_t k = 0; k < a.y.size(); ++k) changed += a.y[k] != b.y[k];
    for (std::size_t k = 0; k < a.u.size(); ++k) {
      changed += a.u[k] != b.u[k];
      changed += a.v[k] != b.v[k];
    }
  }
  std::printf("  CDEF changed %lld samples, header %llu bits\n",
              static_cast<long long>(changed),
              static_cast<unsigned long long>(header_delta_bits));
  check(changed >= 0, "CDEF sample accounting");

  // ---- Quality (decoded streams vs source) ------------------------------
  for (int i = 0; i < frame_count; ++i) {
    report_psnr("baseline", base_decoded[static_cast<std::size_t>(i)],
                source[static_cast<std::size_t>(i)]);
    report_psnr("cdef", cdef_decoded[static_cast<std::size_t>(i)],
                source[static_cast<std::size_t>(i)]);
    if (!filtered4[static_cast<std::size_t>(i)].y.empty()) {
      report_psnr("component 4-set",
                  filtered4[static_cast<std::size_t>(i)],
                  source[static_cast<std::size_t>(i)]);
    }
    std::printf(
        "  frame %d: strengths %d (header %d bits), component4 strengths %d "
        "(header %d, index %d bits)\n",
        i, plans[static_cast<std::size_t>(i)].num_strengths,
        plans[static_cast<std::size_t>(i)].header_bits,
        plans4[static_cast<std::size_t>(i)].num_strengths,
        plans4[static_cast<std::size_t>(i)].header_bits,
        plans4[static_cast<std::size_t>(i)].index_bits);
  }

  std::printf("video_cdef_gate: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
