// Measurement harness for the DC/AC delta-Q RD search.
//
//   ./dcdeltaq_bench <file.y4m> [frame_index] [--max-passes N]
//
// Prints, for a QP sweep, the modelled SSE / rate / RD of the current
// `equal_ac_dc_q = 1`, single-quantizer baseline versus the searched DC/AC
// deltas.  The "rate" is the module's deterministic entropy proxy (see
// search.cpp); this harness measures the search's relative quality/rate
// trade-off, not a real encoder bitstream size.
#include "codec/transform/delta_q/dcdeltaq.h"
#include "core/io/y4m.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double psnr(double sse, std::size_t samples) {
  if (samples == 0) return 0.0;
  const double mse = sse / static_cast<double>(samples);
  if (mse <= 0.0) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

void print_row(int qp, bool separate, const av2enc::dcdeltaq::DeltaQSearchResult& base,
               const av2enc::dcdeltaq::DeltaQSearchResult& best,
               std::size_t samples) {
  const double d_sse = best.sse - base.sse;
  const double d_bits = best.bits - base.bits;
  const double d_psnr = psnr(best.sse, samples) - psnr(base.sse, samples);
  std::printf(
      "  qp=%3d sep=%d  base(sse=%10.1f bits=%9.1f psnr=%6.3f)  "
      "best(sse=%10.1f bits=%9.1f psnr=%6.3f)  dSSE=%+10.1f dBits=%+8.1f "
      "dPSNR=%+6.3f  y=%+d udc=%+d vdc=%+d uac=%+d vac=%+d evals=%d\n",
      qp, separate ? 1 : 0, base.sse, base.bits, psnr(base.sse, samples),
      best.sse, best.bits, psnr(best.sse, samples), d_sse, d_bits, d_psnr,
      best.frame.delta_q_y_dc, best.frame.delta_q_u_dc,
      best.frame.delta_q_v_dc, best.frame.delta_q_u_ac,
      best.frame.delta_q_v_ac, best.evaluations);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <file.y4m> [frame] [--max-passes N]\n",
                 argv[0]);
    return 2;
  }
  int frame_index = 0;
  int max_passes = 3;
  double lambda_scale =
      av2enc::dcdeltaq::DeltaQSearchConfig{}.lambda_scale;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--max-passes") == 0 && i + 1 < argc) {
      max_passes = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--lambda-scale") == 0 && i + 1 < argc) {
      lambda_scale = std::atof(argv[++i]);
    } else {
      frame_index = std::atoi(argv[i]);
    }
  }

  av2enc::Y4mReader reader(argv[1]);
  std::string error;
  if (!reader.open(error)) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mFrame frame;
  for (int i = 0; i <= frame_index; ++i) {
    const av2enc::Y4mReadResult r = reader.read_frame(frame, error);
    if (r != av2enc::Y4mReadResult::kFrame) {
      std::fprintf(stderr, "read frame %d: %s\n", i, error.c_str());
      return 1;
    }
  }
  const int w = static_cast<int>(frame.width());
  const int h = static_cast<int>(frame.height());
  av2enc::dcdeltaq::FramePlanesView planes{w, h, frame.y(), frame.u(),
                                           frame.v()};
  const std::size_t samples = frame.luma_size() + 2 * frame.chroma_size();

  const int qps[] = {32, 64, 90, 120, 141, 191, 255};
  for (int qp : qps) {
    for (int separate = 0; separate < 2; ++separate) {
      av2enc::dcdeltaq::DeltaQSearchConfig base_cfg;
      base_cfg.base_q_idx = qp;
      base_cfg.equal_ac_dc_q = true;
      base_cfg.separate_uv_delta_q = false;
      base_cfg.max_passes = 0;
      av2enc::dcdeltaq::DeltaQSearchResult base;
      if (!av2enc::dcdeltaq::search_frame_delta_q(planes, base_cfg, &base,
                                                  &error)) {
        std::fprintf(stderr, "baseline qp %d: %s\n", qp, error.c_str());
        return 1;
      }
      av2enc::dcdeltaq::DeltaQSearchConfig cfg;
      cfg.base_q_idx = qp;
      cfg.equal_ac_dc_q = false;
      cfg.separate_uv_delta_q = separate != 0;
      cfg.max_passes = max_passes;
      cfg.max_luma_blocks = 2048;
      cfg.lambda_scale = lambda_scale;
      av2enc::dcdeltaq::DeltaQSearchResult best;
      if (!av2enc::dcdeltaq::search_frame_delta_q(planes, cfg, &best, &error)) {
        std::fprintf(stderr, "search qp %d: %s\n", qp, error.c_str());
        return 1;
      }
      print_row(qp, separate != 0, base, best, samples);
    }
  }
  return 0;
}
