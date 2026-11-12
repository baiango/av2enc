// GDF quality measurement (640x360 and 720p).
//
// Encodes one frame with the shipped M1 CPU still encoder, runs the GDF
// encoder decision on the real reconstruction, and writes:
//   base.y4m        M1 reconstruction without GDF
//   gdf_frame.y4m   frame-wide GDF (gdf_per_block = 0)
//   gdf_block.y4m   per-block GDF (gdf_per_block = 1, GdfBlks flags)
// plus PSNR-Y and the exact signalling bit cost.  CVVDP is run by
// run_quality.sh on the Y4M files; this tool is pure C++ and writes nothing
// outside --out.
//
// Usage:
//   gdf_quality --input clip.y4m --out DIR [--qp Q] [--lambda L]
#include "encoder/filters/gdf_stream/gdf_decision.h"
#include "encoder/m1/m1_still.h"
#include "io/y4m.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace m1 = av2enc::m1;
namespace gdf = av2enc::filters::gdf;

namespace {

bool WriteY4m(const fs::path& path, int width, int height,
              const std::vector<std::uint8_t>& y,
              const std::vector<std::uint8_t>& u,
              const std::vector<std::uint8_t>& v, std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "cannot open " + path.string();
    return false;
  }
  out << "YUV4MPEG2 W" << width << " H" << height
      << " F30:1 Ip A1:1 C420\nFRAME\n";
  out.write(reinterpret_cast<const char*>(y.data()),
            static_cast<std::streamsize>(y.size()));
  out.write(reinterpret_cast<const char*>(u.data()),
            static_cast<std::streamsize>(u.size()));
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size()));
  if (!out) {
    *error = "write failed for " + path.string();
    return false;
  }
  return true;
}

std::vector<std::uint8_t> ToBytes(const std::vector<std::uint16_t>& samples) {
  std::vector<std::uint8_t> out(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    out[i] = static_cast<std::uint8_t>(samples[i]);
  }
  return out;
}

double Psnr(std::int64_t sse, std::int64_t samples) {
  if (sse <= 0) return 99.0;
  const double mse = static_cast<double>(sse) / static_cast<double>(samples);
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

struct ModeResult {
  std::int64_t sse_y = 0;
  double psnr_y = 0.0;
  std::int64_t header_bits = 0;
  std::int64_t symbol_bits = 0;
  int enabled_units = 0;
  int total_units = 0;
  int qc = 0;
  int scale = 0;
  std::vector<std::uint8_t> luma;
};

}  // namespace

int main(int argc, char** argv) {
  fs::path input;
  fs::path out_dir;
  int qp = 90;
  double lambda = 0.0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> const char* {
      if (i + 1 >= argc) return nullptr;
      return argv[++i];
    };
    if (arg == "--input") {
      const char* value = next();
      if (value == nullptr) return 1;
      input = value;
    } else if (arg == "--out") {
      const char* value = next();
      if (value == nullptr) return 1;
      out_dir = value;
    } else if (arg == "--qp") {
      const char* value = next();
      if (value == nullptr) return 1;
      qp = std::atoi(value);
    } else if (arg == "--lambda") {
      const char* value = next();
      if (value == nullptr) return 1;
      lambda = std::atof(value);
    } else {
      std::printf("unknown argument: %s\n", arg.c_str());
      return 1;
    }
  }
  if (input.empty() || out_dir.empty()) {
    std::printf(
        "usage: gdf_quality --input clip.y4m --out DIR [--qp Q] [--lambda L]\n");
    return 1;
  }
  std::error_code ec;
  fs::create_directories(out_dir, ec);

  std::string error;
  av2enc::Y4mReader reader(input);
  if (!reader.open(error)) {
    std::printf("input: %s\n", error.c_str());
    return 1;
  }
  av2enc::Y4mFrame frame;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    std::printf("frame: %s\n", error.c_str());
    return 1;
  }
  const int width = static_cast<int>(frame.width());
  const int height = static_cast<int>(frame.height());
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;

  m1::StillPicture picture;
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();
  m1::M1Options options;
  options.base_q_idx = static_cast<std::uint8_t>(qp);
  options.enable_deblock = true;
  m1::M1Result result;
  if (!m1::encode_m1_still_picture(picture, options, result, error)) {
    std::printf("encode: %s\n", error.c_str());
    return 1;
  }

  std::vector<std::uint16_t> recon(static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height));
  std::vector<std::uint16_t> orig(recon.size());
  for (std::size_t i = 0; i < recon.size(); ++i) {
    recon[i] = result.recon_y[i];
    orig[i] = frame.y()[i];
  }

  gdf::FrameGdfContext context;
  context.enable_gdf = true;
  context.gdf_unit_matches_sb_size = true;
  context.single_picture_header_flag = true;
  context.sb_size = 64;
  context.width = width;
  context.height = height;
  context.mi_cols = 2 * ((width + 7) / 8);
  context.mi_rows = 2 * ((height + 7) / 8);
  const gdf::GdfUnitLayout layout = gdf::DeriveGdfUnitLayout(context);
  if (layout.blk_size <= 0) {
    std::printf("layout: disabled\n");
    return 1;
  }

  gdf::GdfDecisionOptions decision_options;
  decision_options.filter.base_q_idx = qp;
  decision_options.filter.frame_is_intra = true;
  decision_options.filter.bit_depth = 8;
  decision_options.disable_cdf_update = true;
  decision_options.search_pic_indices = true;
  decision_options.lambda_bits = lambda;
  gdf::GdfDecision decision;
  gdf::GdfSourceFrame source;
  source.curr = recon.data();
  source.cdef = recon.data();
  source.stride = width;
  gdf::GdfPlane original_plane;
  original_plane.data = orig.data();
  original_plane.width = width;
  original_plane.height = height;
  original_plane.stride = width;
  if (!gdf::DecideGdf(source, original_plane, context, decision_options,
                      &decision, &error)) {
    std::printf("decision: %s\n", error.c_str());
    return 1;
  }

  // Baseline.
  {
    std::int64_t sse = 0;
    for (std::size_t i = 0; i < recon.size(); ++i) {
      const std::int64_t d = static_cast<std::int64_t>(orig[i]) - recon[i];
      sse += d * d;
    }
    const std::vector<std::uint8_t> y = ToBytes(recon);
    if (!WriteY4m(out_dir / "base.y4m", width, height, y, result.recon_u,
                  result.recon_v, &error)) {
      std::printf("%s\n", error.c_str());
      return 1;
    }
    std::printf(
        "{\"mode\":\"base\",\"psnr_y\":%.4f,\"sse_y\":%lld,\"frame_bytes\":%zu,"
        "\"header_bits\":0,\"symbol_bits\":0,\"units\":0}\n",
        Psnr(sse, static_cast<std::int64_t>(recon.size())),
        static_cast<long long>(sse), result.frame.size());
  }

  // Frame-wide GDF (gdf_per_block = 0).
  {
    gdf::FrameGdfParams params = decision.params;
    params.frame_enable = true;
    params.per_block = false;
    std::vector<std::uint16_t> out(recon);
    if (!gdf::ApplyGdfToFrame(source, decision_options.filter, context, params,
                              {}, out.data(), width, &error)) {
      std::printf("apply frame-wide: %s\n", error.c_str());
      return 1;
    }
    std::int64_t sse = 0;
    for (std::size_t i = 0; i < recon.size(); ++i) {
      const std::int64_t d = static_cast<std::int64_t>(orig[i]) - out[i];
      sse += d * d;
    }
    const std::vector<std::uint8_t> y = ToBytes(out);
    if (!WriteY4m(out_dir / "gdf_frame.y4m", width, height, y, result.recon_u,
                  result.recon_v, &error)) {
      std::printf("%s\n", error.c_str());
      return 1;
    }
    const std::int64_t seq_bits = 1 + 1;  // enable_gdf + matches (64 SB)
    const std::int64_t header_bits =
        seq_bits + 4 + (layout.per_block_present ? 1 : 0);
    std::printf(
        "{\"mode\":\"gdf_frame\",\"psnr_y\":%.4f,\"sse_y\":%lld,\"qc\":%d,"
        "\"scale\":%d,\"header_bits\":%lld,\"symbol_bits\":0,\"units\":%d,"
        "\"frame_bytes\":%zu}\n",
        Psnr(sse, static_cast<std::int64_t>(recon.size())),
        static_cast<long long>(sse), params.pic_qc_idx, params.pic_scale_idx,
        static_cast<long long>(header_bits), layout.cols * layout.rows,
        result.frame.size());
  }

  // Per-block GDF (gdf_per_block = 1) when the syntax allows it.
  if (layout.per_block_present) {
    gdf::FrameGdfParams params = decision.params;
    params.frame_enable = true;
    params.per_block = true;
    std::vector<std::uint8_t> flags = decision.unit_flags;
    const std::size_t unit_count =
        static_cast<std::size_t>(layout.cols) *
        static_cast<std::size_t>(layout.rows);
    if (flags.size() != unit_count) {
      flags.assign(unit_count, 1);
    }
    std::vector<std::uint16_t> out(recon);
    if (!gdf::ApplyGdfToFrame(source, decision_options.filter, context, params,
                              flags, out.data(), width, &error)) {
      std::printf("apply per-block: %s\n", error.c_str());
      return 1;
    }
    std::int64_t sse = 0;
    for (std::size_t i = 0; i < recon.size(); ++i) {
      const std::int64_t d = static_cast<std::int64_t>(orig[i]) - out[i];
      sse += d * d;
    }
    const std::vector<std::uint8_t> y = ToBytes(out);
    if (!WriteY4m(out_dir / "gdf_block.y4m", width, height, y, result.recon_u,
                  result.recon_v, &error)) {
      std::printf("%s\n", error.c_str());
      return 1;
    }
    int enabled = 0;
    for (const std::uint8_t flag : flags) enabled += flag != 0 ? 1 : 0;
    std::int64_t symbol_bytes = 0;
    std::int64_t symbol_bits = 0;
    gdf::MeasureUseGdfSymbols(layout, params, 64, flags, true, &symbol_bytes,
                              &symbol_bits, &error);
    const std::int64_t seq_bits = 2;
    const std::int64_t header_bits =
        seq_bits + 4 + (layout.per_block_present ? 1 : 0);
    std::printf(
        "{\"mode\":\"gdf_block\",\"psnr_y\":%.4f,\"sse_y\":%lld,\"qc\":%d,"
        "\"scale\":%d,\"header_bits\":%lld,\"symbol_bits\":%lld,\"units\":%d,"
        "\"enabled_units\":%d,\"frame_bytes\":%zu}\n",
        Psnr(sse, static_cast<std::int64_t>(recon.size())),
        static_cast<long long>(sse), params.pic_qc_idx, params.pic_scale_idx,
        static_cast<long long>(header_bits),
        static_cast<long long>(symbol_bits), layout.cols * layout.rows, enabled,
        result.frame.size());
  }

  std::printf(
      "{\"mode\":\"layout\",\"blk_size\":%d,\"cols\":%d,\"rows\":%d,"
      "\"width\":%d,\"height\":%d,\"qp\":%d}\n",
      layout.blk_size, layout.cols, layout.rows, width, height, qp);
  (void)cw;
  (void)ch;
  return 0;
}
