// Chroma loop-restoration quality probe.
//
// Reads a source Y4M and the decoder reconstruction of the same frame(s),
// chooses chroma restoration units on the real reconstruction
// (chroma_decide), applies them with the reference §7.20 core, and reports
// U/V PSNR before/after, exact signalled bits, and the filtered Y4M used for
// CVVDP.
//
// This is an encoder-side measurement (the target is the source picture); it
// is not a bitstream gate.  The decoder gate is `lr_chroma_gate`.
//
// Usage:
//   lr_chroma_quality --source src.y4m --recon recon.y4m
//                     [--out-prefix /tmp/lrq] [--shift N] [--lambda L]

#include "codec/filters/loop_restoration/chroma/chroma_lr.h"

#include "core/io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

namespace rest = av2enc::restoration;

struct Frame {
  int width = 0;
  int height = 0;
  std::uint32_t fps_num = 30;
  std::uint32_t fps_den = 1;
  std::vector<std::uint8_t> y, u, v;
  int cw() const { return (width + 1) / 2; }
  int ch() const { return (height + 1) / 2; }
};

bool read_all(const std::string& path, std::vector<Frame>* frames,
              std::string* error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(*error)) return false;
  av2enc::Y4mFrame frame;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, *error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    Frame out;
    out.width = static_cast<int>(frame.width());
    out.height = static_cast<int>(frame.height());
    out.fps_num = reader.header().fps_num;
    out.fps_den = reader.header().fps_den;
    out.y.assign(frame.y().begin(), frame.y().end());
    out.u.assign(frame.u().begin(), frame.u().end());
    out.v.assign(frame.v().begin(), frame.v().end());
    frames->push_back(std::move(out));
  }
  return !frames->empty();
}

bool write_all(const std::string& path, const std::vector<Frame>& frames,
               const std::vector<Frame>& chroma, std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "cannot open " + path;
    return false;
  }
  out << "YUV4MPEG2 W" << frames[0].width << " H" << frames[0].height
      << " F" << frames[0].fps_num << ":" << frames[0].fps_den
      << " Ip A1:1 C420\n";
  for (std::size_t i = 0; i < frames.size(); ++i) {
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(frames[i].y.data()),
              static_cast<std::streamsize>(frames[i].y.size()));
    out.write(reinterpret_cast<const char*>(chroma[i].u.data()),
              static_cast<std::streamsize>(chroma[i].u.size()));
    out.write(reinterpret_cast<const char*>(chroma[i].v.data()),
              static_cast<std::streamsize>(chroma[i].v.size()));
  }
  return static_cast<bool>(out);
}

av2enc::lr_chroma::ChromaLrInput make_input(const Frame& base,
                                            const Frame& target) {
  av2enc::lr_chroma::ChromaLrInput in;
  in.plane_width[0] = in.plane_width[1] = base.cw();
  in.plane_height[0] = in.plane_height[1] = base.ch();
  in.curr[0] = base.u.data();
  in.curr_stride[0] = base.cw();
  in.curr[1] = base.v.data();
  in.curr_stride[1] = base.cw();
  in.cdef[0] = base.u.data();
  in.cdef_stride[0] = base.cw();
  in.cdef[1] = base.v.data();
  in.cdef_stride[1] = base.cw();
  in.luma_curr = base.y.data();
  in.luma_curr_stride = base.width;
  in.luma_cdef = base.y.data();
  in.luma_cdef_stride = base.width;
  in.luma_width = base.width;
  in.luma_height = base.height;
  in.source[0] = target.u.data();
  in.source_stride[0] = target.cw();
  in.source[1] = target.v.data();
  in.source_stride[1] = target.cw();
  in.mi_cols = (base.width + 3) / 4;
  in.mi_rows = (base.height + 3) / 4;
  in.sub_x = 1;
  in.sub_y = 1;
  in.sb_size = 64;
  return in;
}

double psnr_plane(const std::vector<std::uint8_t>& a,
                  const std::vector<std::uint8_t>& b) {
  std::uint64_t sse = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    sse += static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(d);
  }
  if (sse == 0) return 99.0;
  const double mse =
      static_cast<double>(sse) / static_cast<double>(a.size());
  return 10.0 * std::log10((255.0 * 255.0) / mse);
}

}  // namespace

int main(int argc, char** argv) {
  std::string source_path;
  std::string recon_path;
  std::string prefix = "/tmp/lrq";
  int forced_shift = -1;
  double lambda = 0.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--source" && i + 1 < argc) {
      source_path = argv[++i];
    } else if (a == "--recon" && i + 1 < argc) {
      recon_path = argv[++i];
    } else if (a == "--out-prefix" && i + 1 < argc) {
      prefix = argv[++i];
    } else if (a == "--shift" && i + 1 < argc) {
      forced_shift = std::atoi(argv[++i]);
    } else if (a == "--lambda" && i + 1 < argc) {
      lambda = std::atof(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s --source src.y4m --recon recon.y4m "
                   "[--out-prefix P] [--shift 0..3] [--lambda L]\n",
                   argv[0]);
      return 2;
    }
  }
  if (source_path.empty() || recon_path.empty()) {
    std::fprintf(stderr, "need --source and --recon\n");
    return 2;
  }
  std::string error;
  std::vector<Frame> source;
  std::vector<Frame> recon;
  if (!read_all(source_path, &source, &error) ||
      !read_all(recon_path, &recon, &error)) {
    std::fprintf(stderr, "read failed: %s\n", error.c_str());
    return 1;
  }
  if (source.size() != recon.size()) {
    std::fprintf(stderr, "frame count mismatch: %zu vs %zu\n", source.size(),
                 recon.size());
    return 1;
  }
  if (source[0].width != recon[0].width ||
      source[0].height != recon[0].height) {
    std::fprintf(stderr, "dimension mismatch\n");
    return 1;
  }

  std::vector<Frame> filtered = recon;
  double before_u = 0.0;
  double before_v = 0.0;
  double after_u = 0.0;
  double after_v = 0.0;
  std::int64_t header_bits = 0;
  std::int64_t tile_bits = 0;
  double sse_before = 0.0;
  double sse_after = 0.0;
  int active = 0;
  int sym = 0;
  int units = 0;
  for (std::size_t i = 0; i < source.size(); ++i) {
    const av2enc::lr_chroma::ChromaLrInput in =
        make_input(recon[i], source[i]);
    av2enc::lr_chroma::ChromaLrOptions options;
    options.forced_shift = forced_shift;
    options.lambda_bits = lambda;
    av2enc::lr_chroma::ChromaDecision decision;
    if (!av2enc::lr_chroma::chroma_decide(in, options, &decision, &error)) {
      std::fprintf(stderr, "decide failed: %s\n", error.c_str());
      return 1;
    }
    Frame out = recon[i];
    std::uint8_t* dst[2] = {out.u.data(), out.v.data()};
    int stride[2] = {out.cw(), out.cw()};
    if (av2enc::lr_chroma::chroma_apply(in, decision, dst, stride) !=
        rest::RestorationStatus::kOk) {
      std::fprintf(stderr, "apply failed\n");
      return 1;
    }
    filtered[i] = out;
    before_u += psnr_plane(source[i].u, recon[i].u);
    before_v += psnr_plane(source[i].v, recon[i].v);
    after_u += psnr_plane(source[i].u, out.u);
    after_v += psnr_plane(source[i].v, out.v);
    header_bits += decision.header_bits;
    tile_bits += decision.tile_bits;
    sse_before += decision.identity_sse;
    sse_after += decision.sse;
    for (const av2enc::lr_chroma::ChromaUnitPlan& plan : decision.units) {
      ++units;
      if (plan.on) ++active;
      if (plan.on && plan.filter.uv_sym) ++sym;
    }
  }
  const double n = static_cast<double>(source.size());
  std::printf("source: %s (%dx%d, %zu frame%s)\n", source_path.c_str(),
              source[0].width, source[0].height, source.size(),
              source.size() == 1 ? "" : "s");
  std::printf("recon : %s\n", recon_path.c_str());
  std::printf(
      "U PSNR %.3f -> %.3f dB (%.3f)\n", before_u / n, after_u / n,
      after_u / n - before_u / n);
  std::printf(
      "V PSNR %.3f -> %.3f dB (%.3f)\n", before_v / n, after_v / n,
      after_v / n - before_v / n);
  std::printf("chroma SSE %.0f -> %.0f (%.2f%%)\n", sse_before, sse_after,
              100.0 * (sse_after - sse_before) / sse_before);
  std::printf(
      "units %d, active %d, uv_sym %d; header %lld bits, tile %lld bits, "
      "total %lld bits (%.1f bytes/frame)\n",
      units, active, sym, static_cast<long long>(header_bits),
      static_cast<long long>(tile_bits),
      static_cast<long long>(header_bits + tile_bits),
      static_cast<double>(header_bits + tile_bits) / (8.0 * n));
  const std::string filtered_path = prefix + ".filtered.y4m";
  if (!write_all(filtered_path, recon, filtered, &error)) {
    std::fprintf(stderr, "write failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("filtered Y4M: %s\n", filtered_path.c_str());
  return 0;
}
