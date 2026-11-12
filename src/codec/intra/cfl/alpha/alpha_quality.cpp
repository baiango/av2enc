// Prediction-level quality delta: CFL_DERIVED_ALPHA vs explicit alpha.
//
// This is not an end-to-end encode: no transform, quantiser, or coefficient
// rate is involved. It isolates the alpha decision, which is exactly what this
// component changes:
//
//   * read one 8-bit 4:2:0 Y4M frame;
//   * for every BxB chroma block, use the block mean of the source chroma as
//     an ideal DC prediction (both modes share it);
//   * derived alpha uses the source chroma above/left neighbours as the
//     reconstructed neighbours (optimistic, but identical for both modes);
//   * explicit alpha is the exhaustive SSE optimum from this component;
//   * report SSE/PSNR and the RD mode counts over a lambda sweep.
//
// Because luma is the original frame rather than a reconstruction, the absolute
// PSNR is optimistic. The derived-vs-explicit *difference* is the measurement.
//
// Usage: alpha_quality <file.y4m> [block=8] [lambda ...]

#include "alpha.h"

#include "../cfl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace alpha = av2enc::cfl::alpha;
namespace cfl = av2enc::cfl;

struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool read_frame(const char* path, Frame* frame) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return false;
  }
  std::string header;
  std::getline(in, header);
  if (header.rfind("YUV4MPEG2", 0) != 0) {
    std::fprintf(stderr, "not a Y4M file\n");
    return false;
  }
  int width = 0;
  int height = 0;
  bool c420 = false;
  std::size_t pos = 0;
  while (pos < header.size()) {
    while (pos < header.size() && header[pos] == ' ') {
      ++pos;
    }
    if (pos >= header.size()) {
      break;
    }
    const char tag = header[pos++];
    std::size_t end = pos;
    while (end < header.size() && header[end] != ' ') {
      ++end;
    }
    const std::string value = header.substr(pos, end - pos);
    pos = end;
    if (tag == 'W') {
      width = std::atoi(value.c_str());
    } else if (tag == 'H') {
      height = std::atoi(value.c_str());
    } else if (tag == 'C') {
      c420 = value == "420" || value == "420jpeg" || value == "420mpeg2" ||
             value == "420paldv";
    }
  }
  if (width <= 0 || height <= 0 || !c420) {
    std::fprintf(stderr, "only 8-bit 4:2:0 Y4M is supported\n");
    return false;
  }
  std::string frame_header;
  std::getline(in, frame_header);
  if (frame_header.rfind("FRAME", 0) != 0) {
    std::fprintf(stderr, "missing FRAME header\n");
    return false;
  }
  const int cw = width / 2;
  const int ch = height / 2;
  frame->width = width;
  frame->height = height;
  frame->y.resize(static_cast<std::size_t>(width) *
                  static_cast<std::size_t>(height));
  frame->u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  frame->v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  in.read(reinterpret_cast<char*>(frame->y.data()),
          static_cast<std::streamsize>(frame->y.size()));
  in.read(reinterpret_cast<char*>(frame->u.data()),
          static_cast<std::streamsize>(frame->u.size()));
  in.read(reinterpret_cast<char*>(frame->v.data()),
          static_cast<std::streamsize>(frame->v.size()));
  if (!in) {
    std::fprintf(stderr, "truncated Y4M frame\n");
    return false;
  }
  return true;
}

int block_mean(const std::vector<std::uint8_t>& plane, int stride, int x,
               int y, int size) {
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      sum += plane[static_cast<std::size_t>(y + i) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x + j)];
    }
  }
  const int count = size * size;
  return (sum + count / 2) / count;
}

double psnr(std::int64_t sse, std::int64_t samples) {
  if (sse == 0) {
    return 99.0;
  }
  return 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(samples) /
                           static_cast<double>(sse));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <file.y4m> [block=8] [lambda ...]\n", argv[0]);
    return 2;
  }
  const int block = argc >= 3 ? std::atoi(argv[2]) : 8;
  std::vector<double> lambdas;
  for (int i = 3; i < argc; ++i) {
    lambdas.push_back(std::atof(argv[i]));
  }
  if (lambdas.empty()) {
    // M1's QP-90 calibration is ~2.18 SSE per bit; 0..8 brackets it.
    lambdas = {0.0, 1.0, 2.178, 4.0, 8.0};
  }
  if (block <= 0 || block > 64) {
    std::fprintf(stderr, "invalid block size\n");
    return 2;
  }

  Frame frame;
  if (!read_frame(argv[1], &frame)) {
    return 1;
  }
  const int cw = frame.width / 2;
  const int ch = frame.height / 2;
  if (block > cw || block > ch) {
    std::fprintf(stderr, "block larger than chroma plane\n");
    return 2;
  }

  std::vector<std::uint8_t> recon_u(frame.u.size());
  std::vector<std::uint8_t> recon_v(frame.v.size());
  const alpha::CflCdfs cdfs = alpha::default_cfl_cdfs();

  std::int64_t sse_derived = 0;
  std::int64_t sse_explicit = 0;
  std::int64_t samples = 0;
  std::int64_t blocks = 0;

  struct LambdaStats {
    double lambda = 0.0;
    std::int64_t explicit_blocks = 0;
    std::int64_t derived_blocks = 0;
    std::int64_t sse = 0;
    int rate_q8 = 0;
  };
  std::vector<LambdaStats> stats;
  for (double lambda : lambdas) {
    LambdaStats s{};
    s.lambda = lambda;
    stats.push_back(s);
  }

  for (int y = 0; y + block <= ch; y += block) {
    for (int x = 0; x + block <= cw; x += block) {
      // Reconstructed chroma: source everywhere, DC inside the current block.
      std::copy(frame.u.begin(), frame.u.end(), recon_u.begin());
      std::copy(frame.v.begin(), frame.v.end(), recon_v.begin());
      const int dc_u = block_mean(frame.u, cw, x, y, block);
      const int dc_v = block_mean(frame.v, cw, x, y, block);
      for (int i = 0; i < block; ++i) {
        for (int j = 0; j < block; ++j) {
          const std::size_t index =
              static_cast<std::size_t>(y + i) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x + j);
          recon_u[index] = static_cast<std::uint8_t>(dc_u);
          recon_v[index] = static_cast<std::uint8_t>(dc_v);
        }
      }

      cfl::Block u{};
      u.luma = frame.y.data();
      u.luma_stride = frame.width;
      u.luma_max_x = frame.width - 1;
      u.luma_max_y = frame.height - 1;
      u.sb_top_y = ((y * 2) / 64) * 64;
      u.chroma = recon_u.data();
      u.chroma_stride = cw;
      u.x = x;
      u.y = y;
      u.w = block;
      u.h = block;
      u.coded_width = cw;
      u.coded_height = ch;
      u.have_above = y >= 4 ? 1 : 0;
      u.have_left = x >= 4 ? 1 : 0;
      u.bit_depth = 8;
      u.ds_filter = 0;
      u.sub_x = 1;
      u.sub_y = 1;
      cfl::Block v = u;
      v.chroma = recon_v.data();

      const cfl::SourcePlane src_u{frame.u.data(), cw, cw - 1, ch - 1};
      const cfl::SourcePlane src_v{frame.v.data(), cw, cw - 1, ch - 1};

      cfl::Scratch scratch;
      cfl::compute_luma(u, scratch);
      const int derived_u = cfl::derive_implicit_alpha(u, scratch);
      const int derived_v = cfl::derive_implicit_alpha(v, scratch);
      const std::int64_t d_u =
          cfl::prediction_sse(u, derived_u, scratch, src_u);
      const std::int64_t d_v =
          cfl::prediction_sse(v, derived_v, scratch, src_v);
      sse_derived += d_u + d_v;

      alpha::Config sse_config{};
      sse_config.cost.use_rate = false;
      const auto best = alpha::search_explicit(u, v, src_u, src_v, sse_config,
                                               cdfs, scratch);
      if (!best.has_value()) {
        std::fprintf(stderr, "search failed at %d,%d\n", x, y);
        return 1;
      }
      sse_explicit += best->total_sse();
      samples += static_cast<std::int64_t>(block) * block * 2;
      ++blocks;

      for (auto& s : stats) {
        alpha::Config config{};
        config.cost.use_rate = true;
        config.cost.lambda_q8 =
            static_cast<std::int64_t>(std::llround(s.lambda * 256.0));
        const auto decision = alpha::decide_cfl_alpha(u, v, src_u, src_v,
                                                      config, cdfs, scratch);
        if (!decision.has_value()) {
          std::fprintf(stderr, "decide failed at %d,%d\n", x, y);
          return 1;
        }
        if (decision->chosen.explicit_alpha) {
          ++s.explicit_blocks;
        } else {
          ++s.derived_blocks;
        }
        s.sse += decision->chosen.total_sse();
        s.rate_q8 += decision->chosen.rate_q8;
      }
    }
  }

  std::printf("file=%s block=%d blocks=%lld chroma_samples=%lld\n", argv[1],
              block, static_cast<long long>(blocks),
              static_cast<long long>(samples));
  std::printf("derived_sse=%lld derived_psnr=%.4f\n",
              static_cast<long long>(sse_derived),
              psnr(sse_derived, samples));
  std::printf("explicit_sse=%lld explicit_psnr=%.4f\n",
              static_cast<long long>(sse_explicit),
              psnr(sse_explicit, samples));
  const double reduction =
      sse_derived > 0
          ? 100.0 * (static_cast<double>(sse_derived - sse_explicit) /
                     static_cast<double>(sse_derived))
          : 0.0;
  std::printf("sse_reduction_pct=%.3f psnr_gain_db=%.4f\n", reduction,
              psnr(sse_explicit, samples) - psnr(sse_derived, samples));
  std::printf("lambda,derived_blocks,explicit_blocks,chosen_sse,"
              "chosen_psnr,avg_alpha_rate_bits\n");
  for (const auto& s : stats) {
    const double avg_rate =
        blocks > 0
            ? static_cast<double>(s.rate_q8) / 256.0 /
                  static_cast<double>(blocks)
            : 0.0;
    std::printf("%.1f,%lld,%lld,%lld,%.4f,%.4f\n", s.lambda,
                static_cast<long long>(s.derived_blocks),
                static_cast<long long>(s.explicit_blocks),
                static_cast<long long>(s.sse), psnr(s.sse, samples),
                avg_rate);
  }
  return 0;
}
