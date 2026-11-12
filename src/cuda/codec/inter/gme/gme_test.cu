// Real-GPU bit-exact parity test for the CUDA GME pipeline.
//
// Build with ./build_gpu_test.sh.  The test uses one av2enc::cuda::Arena for
// every device buffer and calls no cudaMalloc / cudaFree / cudaMallocHost.
//
// Covered:
//   * correspondence extraction + integer least squares for points, block MVs
//     and warps against gme::estimate_* (bit-exact GmParams and integer
//     diagnostics);
//   * synthetic pan / zoom / rotate / combined frames through a real block
//     matcher;
//   * real 720p pairs from testdata/shine_1280x720_60f.y4m when present;
//   * device projection over every block size 4x4..64x64, all four
//     precisions, full MV range and GLOBALMV usability;
//   * device bilinear warp / SSE quality proxy against the CPU helper;
//   * required_device_bytes, exact allocation, one-byte-short kArenaTooSmall,
//     mark/rewind, two consecutive launches with no arena growth;
//   * per-job buffers (max_jobs = 2).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/codec/inter/gme/gme.h"
#include "codec/inter/gme/gme.h"

namespace dev = av2enc::cuda::gme;
namespace cpu = av2enc::inter::gme;

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

void CheckEqI64(long long actual, long long expected, const char* what) {
  ++g_checks;
  if (actual != expected) {
    ++g_failures;
    if (g_failures <= 40) {
      std::fprintf(stderr, "FAIL: %s (actual %lld, expected %lld)\n", what,
                   actual, expected);
    }
  }
}

void CheckClose(double actual, double expected, double tolerance,
                const char* what) {
  ++g_checks;
  const double scale = std::max(1.0, std::fabs(expected));
  if (!(std::fabs(actual - expected) <= tolerance * scale)) {
    ++g_failures;
    if (g_failures <= 40) {
      std::fprintf(stderr, "FAIL: %s (actual %.12g, expected %.12g)\n", what,
                   actual, expected);
    }
  }
}

// ---------------------------------------------------------------------------
// Conversions and comparison.
// ---------------------------------------------------------------------------

dev::GmeParams ToDevParams(const cpu::GmParams& p) {
  dev::GmeParams out;
  out.type = static_cast<dev::GmeI32>(p.type);
  for (int i = 0; i < 6; ++i) out.p[i] = p.p[static_cast<std::size_t>(i)];
  return out;
}

dev::GmePoint ToDevPoint(const cpu::GmPoint& p) {
  dev::GmePoint out;
  out.src_x = p.src_x;
  out.src_y = p.src_y;
  out.dst_x = p.dst_x;
  out.dst_y = p.dst_y;
  out.weight = p.weight;
  return out;
}

dev::GmeBlockMv ToDevBlock(const cpu::GmBlockMv& b) {
  dev::GmeBlockMv out;
  out.x = b.x;
  out.y = b.y;
  out.mv_x = b.mv_x;
  out.mv_y = b.mv_y;
  out.weight = b.weight;
  return out;
}

dev::GmeWarpSample ToDevWarp(const cpu::GmWarpSample& w) {
  dev::GmeWarpSample out;
  out.x = w.x;
  out.y = w.y;
  out.radius = w.radius;
  out.model = ToDevParams(w.model);
  out.weight = w.weight;
  return out;
}

dev::GmeOptions ToDevOptions(const cpu::GmOptions& o) {
  dev::GmeOptions d;
  d.model = static_cast<dev::GmeI32>(o.model);
  d.min_points = o.min_points;
  d.max_points = o.max_points;
  d.outlier_passes = o.outlier_passes;
  d.outlier_threshold_q8 = o.outlier_threshold_q8;
  d.allow_rotzoom_degrade = o.allow_rotzoom_degrade ? 1 : 0;
  d.require_shear_valid = o.require_shear_valid ? 1 : 0;
  d.rotzoom_mismatch_q16 = o.rotzoom_mismatch_q16;
  d.min_scale_q16 = o.min_scale_q16;
  d.max_scale_q16 = o.max_scale_q16;
  d.max_corner_displacement_q8 = o.max_corner_displacement_q8;
  d.force_integer_mv = o.force_integer_mv ? 1 : 0;
  d.quantize = o.quantize ? 1 : 0;
  d.min_gain_percent = o.min_gain_percent;
  d.frame_width = o.frame_width;
  d.frame_height = o.frame_height;
  return d;
}

int ReasonCode(const std::string& reason) {
  if (reason.empty()) return dev::kGmeReasonNone;
  if (reason == "no points") return dev::kGmeReasonNoPoints;
  if (reason == "too few weighted points") return dev::kGmeReasonTooFewWeighted;
  if (reason == "degenerate point distribution") {
    return dev::kGmeReasonDegenerate;
  }
  if (reason == "collinear points (det <= 0)") return dev::kGmeReasonCollinear;
  if (reason == "too few correspondences") {
    return dev::kGmeReasonTooFewCorrespondences;
  }
  if (reason == "too many correspondences") {
    return dev::kGmeReasonTooManyCorrespondences;
  }
  if (reason == "block MV correspondence exceeds int32") {
    return dev::kGmeReasonBlockMvOverflow;
  }
  if (reason == "warp sample coordinate exceeds int32") {
    return dev::kGmeReasonWarpOverflow;
  }
  if (reason == "IDENTITY with non-default parameters") {
    return dev::kGmeReasonIdentityNonDefault;
  }
  if (reason == "translation not on the 1/8-pel grid") {
    return dev::kGmeReasonTransGrid;
  }
  if (reason == "translation outside GM_TRANS range") {
    return dev::kGmeReasonTransRange;
  }
  if (reason == "alpha not on the 1/1024 grid") {
    return dev::kGmeReasonAlphaGrid;
  }
  if (reason == "alpha outside GM_ALPHA range") {
    return dev::kGmeReasonAlphaRange;
  }
  if (reason == "ROTZOOM requires p4 = -p3 and p5 = p2") {
    return dev::kGmeReasonRotzoomSymmetry;
  }
  if (reason == "unknown GmType") return dev::kGmeReasonUnknownType;
  if (reason == "model scale/determinant out of range") {
    return dev::kGmeReasonScaleOutOfRange;
  }
  if (reason == "shear model is not valid") return dev::kGmeReasonShearInvalid;
  if (reason == "corner projection out of range") {
    return dev::kGmeReasonCornerOutOfRange;
  }
  if (reason == "affine close to similarity; signalled ROTZOOM") {
    return dev::kGmeReasonAffineToRotzoom;
  }
  if (reason == "degraded to ROTZOOM after AFFINE rejection") {
    return dev::kGmeReasonDegradedAfterReject;
  }
  if (reason == "model rejected") return dev::kGmeReasonModelRejected;
  if (reason.rfind("global motion gain below threshold", 0) == 0) {
    return dev::kGmeReasonGainBelowThreshold;
  }
  return -1;
}

void CompareEstimate(const cpu::GmEstimate& reference,
                     const dev::GmeEstimate& actual, const char* label) {
  ++g_checks;
  if (reference.params.type !=
      static_cast<cpu::GmType>(actual.params.type)) {
    ++g_failures;
    if (g_failures <= 40) {
      std::fprintf(stderr, "FAIL: %s type %d != %d\n", label,
                   static_cast<int>(reference.params.type),
                   actual.params.type);
    }
  }
  for (int i = 0; i < 6; ++i) {
    CheckEqI64(actual.params.p[i],
               reference.params.p[static_cast<std::size_t>(i)], label);
  }
  CheckEqI64(actual.diag.ok, reference.diag.ok ? 1 : 0, label);
  CheckEqI64(actual.diag.accepted, reference.diag.accepted ? 1 : 0, label);
  CheckEqI64(actual.diag.codable, reference.diag.codable ? 1 : 0, label);
  CheckEqI64(actual.diag.shear_valid, reference.diag.shear_valid ? 1 : 0,
             label);
  CheckEqI64(actual.diag.degraded, reference.diag.degraded ? 1 : 0, label);
  CheckEqI64(actual.diag.identity, reference.diag.identity ? 1 : 0, label);
  CheckEqI64(actual.diag.points_in, reference.diag.points_in, label);
  CheckEqI64(actual.diag.points_used, reference.diag.points_used, label);
  CheckEqI64(actual.diag.outlier_passes_run,
             reference.diag.outlier_passes_run, label);
  CheckEqI64(actual.diag.identity_sse_q8, reference.diag.identity_sse_q8,
             label);
  CheckEqI64(actual.diag.model_sse_q8, reference.diag.model_sse_q8, label);
  CheckEqI64(actual.diag.candidate_model_sse_q8,
             reference.diag.candidate_model_sse_q8, label);
  CheckEqI64(actual.diag.shear_mismatch_q16,
             reference.diag.shear_mismatch_q16, label);
  CheckClose(actual.diag.gain_db, reference.diag.gain_db, 1e-6, label);
  CheckClose(actual.diag.model_rmse_q8, reference.diag.model_rmse_q8, 1e-6,
             label);
  CheckClose(actual.diag.candidate_gain_db, reference.diag.candidate_gain_db,
             1e-6, label);
  CheckEqI64(actual.diag.reason, ReasonCode(reference.diag.reason), label);
}

// ---------------------------------------------------------------------------
// Models and image helpers.
// ---------------------------------------------------------------------------

cpu::GmParams Similarity(double scale, double angle_deg, double tx,
                         double ty) {
  const double radians = angle_deg * 3.14159265358979323846 / 180.0;
  const double c = std::cos(radians) * scale;
  const double s = std::sin(radians) * scale;
  cpu::GmParams params;
  params.type = cpu::GmType::kRotZoom;
  params.p[0] = static_cast<std::int32_t>(std::llround(tx * 65536.0));
  params.p[1] = static_cast<std::int32_t>(std::llround(ty * 65536.0));
  params.p[2] = static_cast<std::int32_t>(std::llround(c * 65536.0));
  params.p[3] = static_cast<std::int32_t>(std::llround(-s * 65536.0));
  params.p[4] = static_cast<std::int32_t>(std::llround(s * 65536.0));
  params.p[5] = static_cast<std::int32_t>(std::llround(c * 65536.0));
  return params;
}

cpu::GmParams Affine(double p2, double p3, double p4, double p5, double tx,
                     double ty) {
  cpu::GmParams params;
  params.type = cpu::GmType::kAffine;
  params.p[0] = static_cast<std::int32_t>(std::llround(tx * 65536.0));
  params.p[1] = static_cast<std::int32_t>(std::llround(ty * 65536.0));
  params.p[2] = static_cast<std::int32_t>(std::llround(p2 * 65536.0));
  params.p[3] = static_cast<std::int32_t>(std::llround(p3 * 65536.0));
  params.p[4] = static_cast<std::int32_t>(std::llround(p4 * 65536.0));
  params.p[5] = static_cast<std::int32_t>(std::llround(p5 * 65536.0));
  return params;
}

struct Plane {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> data;

  void Init(int width, int height) {
    w = width;
    h = height;
    data.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
  }
  std::uint8_t& At(int x, int y) {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)];
  }
  std::uint8_t At(int x, int y) const {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)];
  }
};

std::uint8_t TextureSample(int x, int y) {
  const double fx = static_cast<double>(x);
  const double fy = static_cast<double>(y);
  const double value = 128.0 + 58.0 * std::sin(fx * 0.057) +
                       48.0 * std::cos(fy * 0.071) +
                       34.0 * std::sin((fx + fy) * 0.031) +
                       24.0 * std::cos((fx - fy) * 0.043) +
                       18.0 * std::sin(std::sqrt((fx - 320.0) * (fx - 320.0) +
                                                 (fy - 180.0) * (fy - 180.0)) *
                                        0.11);
  return static_cast<std::uint8_t>(
      std::clamp(static_cast<int>(std::llround(value)), 0, 255));
}

Plane MakeTexture(int w, int h) {
  Plane plane;
  plane.Init(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) plane.At(x, y) = TextureSample(x, y);
  }
  return plane;
}

Plane WarpPlane(const cpu::GmParams& params, const Plane& src, int w, int h) {
  Plane dst;
  dst.Init(w, h);
  const bool ok = cpu::warp_plane_bilinear(params, src.data.data(), src.w,
                                           src.h, src.w, dst.data.data(), w, h,
                                           w, 8);
  Check(ok, "CPU warp plane");
  return dst;
}

std::int64_t SadAt(const Plane& ref, const Plane& cur, int bx, int by,
                   int block, int dx, int dy, int sample_step) {
  std::int64_t sad = 0;
  for (int y = 0; y < block; y += sample_step) {
    for (int x = 0; x < block; x += sample_step) {
      const int rx = bx + x + dx;
      const int ry = by + y + dy;
      sad += std::abs(static_cast<int>(cur.At(bx + x, by + y)) -
                      static_cast<int>(ref.At(rx, ry)));
    }
  }
  return sad;
}

struct MatchOptions {
  int block = 16;
  int search = 16;
  int sample_step = 2;
  int min_variance = 24;
};

std::vector<cpu::GmBlockMv> MatchBlocks(const Plane& ref, const Plane& cur,
                                         const MatchOptions& options) {
  std::vector<cpu::GmBlockMv> out;
  for (int by = 0; by + options.block <= cur.h; by += options.block) {
    for (int bx = 0; bx + options.block <= cur.w; bx += options.block) {
      std::int64_t sum = 0;
      std::int64_t sum2 = 0;
      int count = 0;
      for (int y = 0; y < options.block; y += options.sample_step) {
        for (int x = 0; x < options.block; x += options.sample_step) {
          const int v = cur.At(bx + x, by + y);
          sum += v;
          sum2 += v * v;
          ++count;
        }
      }
      const double mean = static_cast<double>(sum) / count;
      const double variance = static_cast<double>(sum2) / count - mean * mean;
      if (variance < options.min_variance) continue;

      std::int64_t best = std::numeric_limits<std::int64_t>::max();
      int best_dx = 0;
      int best_dy = 0;
      for (int dy = -options.search; dy <= options.search; ++dy) {
        for (int dx = -options.search; dx <= options.search; ++dx) {
          if (bx + dx < 0 || by + dy < 0 ||
              bx + dx + options.block > ref.w ||
              by + dy + options.block > ref.h) {
            continue;
          }
          const std::int64_t sad =
              SadAt(ref, cur, bx, by, options.block, dx, dy,
                    options.sample_step);
          if (sad < best) {
            best = sad;
            best_dx = dx;
            best_dy = dy;
          }
        }
      }
      if (best == std::numeric_limits<std::int64_t>::max()) continue;

      auto subpel = [&](int delta_x, int delta_y) -> double {
        if (bx + delta_x < 0 || by + delta_y < 0 ||
            bx + delta_x + options.block > ref.w ||
            by + delta_y + options.block > ref.h) {
          return static_cast<double>(best);
        }
        return static_cast<double>(SadAt(ref, cur, bx, by, options.block,
                                         delta_x, delta_y,
                                         options.sample_step));
      };
      const double best_sad = static_cast<double>(best);
      const double smx = subpel(best_dx - 1, best_dy);
      const double spx = subpel(best_dx + 1, best_dy);
      const double smy = subpel(best_dx, best_dy - 1);
      const double spy = subpel(best_dx, best_dy + 1);
      const double denx = smx - 2.0 * best_sad + spx;
      const double deny = smy - 2.0 * best_sad + spy;
      double sub_x = denx > 0.0 ? 0.5 * (smx - spx) / denx : 0.0;
      double sub_y = deny > 0.0 ? 0.5 * (smy - spy) / deny : 0.0;
      sub_x = std::clamp(sub_x, -0.5, 0.5);
      sub_y = std::clamp(sub_y, -0.5, 0.5);

      cpu::GmBlockMv block;
      block.x = (bx + options.block / 2) * 8;
      block.y = (by + options.block / 2) * 8;
      block.mv_x = best_dx * 8 + static_cast<int>(std::llround(sub_x * 8.0));
      block.mv_y = best_dy * 8 + static_cast<int>(std::llround(sub_y * 8.0));
      block.weight =
          static_cast<int>(std::clamp(variance / 8.0, 1.0, 255.0));
      out.push_back(block);
    }
  }
  return out;
}

bool LoadY4mFrames(const std::string& path, int max_frames,
                   std::vector<Plane>* out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::string line;
  if (!std::getline(input, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  std::istringstream header(line);
  std::string token;
  int width = 0;
  int height = 0;
  bool c420 = false;
  while (header >> token) {
    if (token.size() > 1) {
      const char tag = token[0];
      const std::string value = token.substr(1);
      if (tag == 'W') width = std::stoi(value);
      if (tag == 'H') height = std::stoi(value);
      if (tag == 'C') {
        c420 = value == "420" || value == "420jpeg" || value == "420mpeg2";
      }
    }
  }
  if (width <= 0 || height <= 0 || !c420) return false;
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  std::vector<std::uint8_t> ybuf(static_cast<std::size_t>(width) * height);
  std::vector<std::uint8_t> ubuf(static_cast<std::size_t>(cw) * ch);
  std::vector<std::uint8_t> vbuf(static_cast<std::size_t>(cw) * ch);
  for (int frame = 0; frame < max_frames; ++frame) {
    if (!std::getline(input, line) || line.rfind("FRAME", 0) != 0) break;
    if (!input.read(reinterpret_cast<char*>(ybuf.data()),
                    static_cast<std::streamsize>(ybuf.size())) ||
        !input.read(reinterpret_cast<char*>(ubuf.data()),
                    static_cast<std::streamsize>(ubuf.size())) ||
        !input.read(reinterpret_cast<char*>(vbuf.data()),
                    static_cast<std::streamsize>(vbuf.size()))) {
      break;
    }
    Plane plane;
    plane.Init(width, height);
    plane.data = ybuf;
    out->push_back(std::move(plane));
  }
  return !out->empty();
}

// ---------------------------------------------------------------------------
// GPU staging helpers.
// ---------------------------------------------------------------------------

dev::GmeEstimate RunPointsGpu(const dev::GmeBuffers& buffers, int job,
                              const std::vector<cpu::GmPoint>& points,
                              const cpu::GmOptions& options,
                              const char* label) {
  std::vector<dev::GmePoint> device_points;
  device_points.reserve(points.size());
  for (const cpu::GmPoint& point : points) {
    device_points.push_back(ToDevPoint(point));
  }
  std::string error;
  Check(dev::stage_points(buffers, job, device_points.data(),
                          static_cast<int>(device_points.size()), nullptr,
                          &error) == dev::GmeStatus::kOk,
        "stage points");
  Check(dev::estimate_points(buffers, job,
                             static_cast<int>(device_points.size()),
                             ToDevOptions(options), nullptr,
                             &error) == dev::GmeStatus::kOk,
        "estimate points");
  dev::GmeEstimate actual;
  Check(dev::download_estimate(buffers, job, &actual, nullptr, &error) ==
            dev::GmeStatus::kOk,
        "download points");
  const cpu::GmEstimate reference = cpu::estimate_gm_points(points, options);
  CompareEstimate(reference, actual, label);

  // A second launch must produce the same result and must not grow the arena.
  Check(dev::estimate_points(buffers, job,
                             static_cast<int>(device_points.size()),
                             ToDevOptions(options), nullptr,
                             &error) == dev::GmeStatus::kOk,
        "second estimate points");
  dev::GmeEstimate second;
  Check(dev::download_estimate(buffers, job, &second, nullptr, &error) ==
            dev::GmeStatus::kOk,
        "second download points");
  Check(second.params.type == actual.params.type, "points rerun type");
  for (int i = 0; i < 6; ++i) {
    CheckEqI64(second.params.p[i], actual.params.p[i], "points rerun param");
  }
  return actual;
}

dev::GmeEstimate RunBlocksGpu(const dev::GmeBuffers& buffers, int job,
                              const std::vector<cpu::GmBlockMv>& blocks,
                              const cpu::GmOptions& options,
                              const char* label) {
  std::vector<dev::GmeBlockMv> device_blocks;
  device_blocks.reserve(blocks.size());
  for (const cpu::GmBlockMv& block : blocks) {
    device_blocks.push_back(ToDevBlock(block));
  }
  std::string error;
  Check(dev::stage_blocks(buffers, job, device_blocks.data(),
                          static_cast<int>(device_blocks.size()), nullptr,
                          &error) == dev::GmeStatus::kOk,
        "stage blocks");
  Check(dev::estimate_blocks(buffers, job,
                             static_cast<int>(device_blocks.size()),
                             ToDevOptions(options), nullptr,
                             &error) == dev::GmeStatus::kOk,
        "estimate blocks");
  dev::GmeEstimate actual;
  Check(dev::download_estimate(buffers, job, &actual, nullptr, &error) ==
            dev::GmeStatus::kOk,
        "download blocks");
  const cpu::GmEstimate reference = cpu::estimate_gm_block_mvs(blocks, options);
  CompareEstimate(reference, actual, label);
  return actual;
}

dev::GmeEstimate RunWarpsGpu(const dev::GmeBuffers& buffers, int job,
                             const std::vector<cpu::GmWarpSample>& warps,
                             const cpu::GmOptions& options,
                             const char* label) {
  std::vector<dev::GmeWarpSample> device_warps;
  device_warps.reserve(warps.size());
  for (const cpu::GmWarpSample& warp : warps) {
    device_warps.push_back(ToDevWarp(warp));
  }
  std::string error;
  Check(dev::stage_warps(buffers, job, device_warps.data(),
                         static_cast<int>(device_warps.size()), nullptr,
                         &error) == dev::GmeStatus::kOk,
        "stage warps");
  Check(dev::estimate_warps(buffers, job,
                            static_cast<int>(device_warps.size()),
                            ToDevOptions(options), nullptr,
                            &error) == dev::GmeStatus::kOk,
        "estimate warps");
  dev::GmeEstimate actual;
  Check(dev::download_estimate(buffers, job, &actual, nullptr, &error) ==
            dev::GmeStatus::kOk,
        "download warps");
  const cpu::GmEstimate reference = cpu::estimate_gm_warps(warps, options);
  CompareEstimate(reference, actual, label);
  return actual;
}

// ---------------------------------------------------------------------------
// Device projection self-test (all block sizes, precisions and MV clamps).
// ---------------------------------------------------------------------------

struct ProjCase {
  dev::GmeParams params;
  int ref;
  int mi_col;
  int mi_row;
  int bw4;
  int bh4;
  int mi_cols;
  int mi_rows;
  int precision;
  int global;
};

struct ProjResult {
  int usable;
  int use_warp;
  int ref;
  int mv[2];
};

__global__ void ProjectionKernel(const ProjCase* cases, int count,
                                 ProjResult* out) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  const ProjCase& item = cases[index];
  ProjResult result;
  result.usable = 0;
  result.use_warp = 0;
  result.ref = item.ref;
  result.mv[0] = 0;
  result.mv[1] = 0;
  if (item.global) {
    const dev::GmeGlobalMvUse use = dev::gme_global_mv_for_block(
        item.params, item.ref, item.mi_col, item.mi_row, item.bw4, item.bh4,
        item.mi_cols, item.mi_rows);
    result.usable = use.usable;
    result.use_warp = use.use_warp;
    result.ref = use.ref;
    result.mv[0] = use.mv[0];
    result.mv[1] = use.mv[1];
  } else {
    int mv[2] = {0, 0};
    dev::gme_get_warp_motion_vector(
        item.params, item.mi_col, item.mi_row, item.bw4, item.bh4,
        item.mi_cols, item.mi_rows,
        static_cast<dev::GmeMvPrecision>(item.precision), mv);
    result.usable = item.params.type != dev::kGmeIdentity ? 1 : 0;
    result.mv[0] = mv[0];
    result.mv[1] = mv[1];
  }
  out[index] = result;
}

void TestProjectionGpu(const dev::GmeBuffers& buffers,
                       av2enc::cuda::Arena& arena) {
  (void)buffers;
  std::vector<cpu::GmParams> models;
  models.push_back(cpu::make_identity_gm());
  models.push_back(Similarity(1.0, 0.0, 9.5, -5.25));
  models.push_back(Similarity(1.0, 0.0, -2047.875, 2047.875));
  models.push_back(Similarity(1.06, 0.0, 0.0, 0.0));
  models.push_back(Similarity(0.33, 17.0, 4.0, -3.0));
  models.push_back(Similarity(3.0, -31.0, -100.0, 200.0));
  models.push_back(Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0));
  models.push_back(Affine(0.2, 0.4, 0.1, 1.4, 1000.0, -1000.0));
  models.push_back(Affine(1.0, 0.0, 0.0, -1.0, 0.0, 0.0));
  models.push_back(Affine(1.0, 0.0, 0.0, 1.0, 0.0, 0.0));
  // Raw full-range model: the MI grid is large enough that MV_LOW/MV_UPP,
  // not the border clamp, binds at mi=(500000,500000).
  {
    cpu::GmParams extreme = cpu::make_identity_gm();
    extreme.type = cpu::GmType::kRotZoom;
    extreme.p[0] = 2000000000;
    extreme.p[1] = -2000000000;
    extreme.p[2] = 200000;
    extreme.p[3] = -50000;
    extreme.p[4] = 50000;
    extreme.p[5] = 200000;
    models.push_back(extreme);
  }

  std::vector<ProjCase> cases;
  for (std::size_t model_index = 0; model_index < models.size();
       ++model_index) {
    const cpu::GmParams& model = models[model_index];
    const bool huge = model_index + 1 == models.size();
    for (int bw4 = 1; bw4 <= 16; ++bw4) {
      for (int bh4 = 1; bh4 <= 16; ++bh4) {
        for (int precision = 4; precision <= 7; ++precision) {
          for (int global = 0; global <= 1; ++global) {
            ProjCase item;
            item.params = ToDevParams(model);
            item.ref = 2;
            item.mi_col = huge ? 500000 : 3;
            item.mi_row = huge ? 500000 : 5;
            item.bw4 = bw4;
            item.bh4 = bh4;
            item.mi_cols = huge ? 1000001 : 40;
            item.mi_rows = huge ? 1000001 : 30;
            item.precision = precision;
            item.global = global;
            cases.push_back(item);
          }
        }
      }
    }
  }

  void* raw_cases = nullptr;
  void* raw_results = nullptr;
  const std::size_t case_bytes = cases.size() * sizeof(ProjCase);
  const std::size_t result_bytes = cases.size() * sizeof(ProjResult);
  Check(arena.allocate_checked(case_bytes, &raw_cases) ==
            av2enc::cuda::Status::kOk,
        "allocate projection cases");
  Check(arena.allocate_checked(result_bytes, &raw_results) ==
            av2enc::cuda::Status::kOk,
        "allocate projection results");
  ProjCase* d_cases = static_cast<ProjCase*>(raw_cases);
  ProjResult* d_results = static_cast<ProjResult*>(raw_results);

  Check(cudaMemcpyAsync(d_cases, cases.data(), case_bytes,
                        cudaMemcpyHostToDevice, nullptr) == cudaSuccess,
        "upload projection cases");
  const int threads = 128;
  const int blocks =
      (static_cast<int>(cases.size()) + threads - 1) / threads;
  ProjectionKernel<<<blocks, threads>>>(d_cases,
                                        static_cast<int>(cases.size()),
                                        d_results);
  Check(cudaGetLastError() == cudaSuccess, "projection kernel launch");
  std::vector<ProjResult> results(cases.size());
  Check(cudaMemcpyAsync(results.data(), d_results, result_bytes,
                        cudaMemcpyDeviceToHost, nullptr) == cudaSuccess,
        "download projection results");
  Check(cudaStreamSynchronize(nullptr) == cudaSuccess,
        "projection synchronisation");

  // Recompute the reference from the model table in the same order.
  std::size_t index = 0;
  for (const cpu::GmParams& model : models) {
    for (int bw4 = 1; bw4 <= 16; ++bw4) {
      for (int bh4 = 1; bh4 <= 16; ++bh4) {
        for (int precision = 4; precision <= 7; ++precision) {
          for (int global = 0; global <= 1; ++global) {
            const ProjCase& item = cases[index];
            const ProjResult& got = results[index];
            if (global) {
              const cpu::GmGlobalMvUse use = cpu::global_mv_for_block(
                  model, item.ref, item.mi_col, item.mi_row, item.bw4,
                  item.bh4, item.mi_cols, item.mi_rows);
              CheckEqI64(got.usable, use.usable ? 1 : 0, "gpu global usable");
              CheckEqI64(got.use_warp, use.use_warp ? 1 : 0,
                         "gpu global warp");
              CheckEqI64(got.ref, use.ref, "gpu global ref");
              CheckEqI64(got.mv[0], use.mv[0], "gpu global mv_y");
              CheckEqI64(got.mv[1], use.mv[1], "gpu global mv_x");
            } else {
              int mv[2] = {0, 0};
              cpu::get_warp_motion_vector(
                  model, item.mi_col, item.mi_row, item.bw4, item.bh4,
                  item.mi_cols, item.mi_rows,
                  static_cast<cpu::MvPrecision>(precision), mv);
              CheckEqI64(got.mv[0], mv[0], "gpu projection mv_y");
              CheckEqI64(got.mv[1], mv[1], "gpu projection mv_x");
            }
            ++index;
          }
        }
      }
    }
  }
  std::printf("gpu projection cases: %zu\n", cases.size());
}

// ---------------------------------------------------------------------------
// Device quality proxy.
// ---------------------------------------------------------------------------

void TestQualityProxyGpu(const dev::GmeBuffers& buffers,
                         av2enc::cuda::Arena& arena) {
  const int w = 128;
  const int h = 96;
  const Plane source = MakeTexture(w, h);
  const std::size_t plane_bytes = static_cast<std::size_t>(w) * h;
  void* raw_src = nullptr;
  void* raw_cur = nullptr;
  void* raw_dst = nullptr;
  void* raw_sse = nullptr;
  Check(arena.allocate_checked(plane_bytes, &raw_src) ==
            av2enc::cuda::Status::kOk,
        "allocate source plane");
  Check(arena.allocate_checked(plane_bytes, &raw_cur) ==
            av2enc::cuda::Status::kOk,
        "allocate current plane");
  Check(arena.allocate_checked(plane_bytes, &raw_dst) ==
            av2enc::cuda::Status::kOk,
        "allocate destination plane");
  Check(arena.allocate_checked(sizeof(dev::GmeI64), &raw_sse) ==
            av2enc::cuda::Status::kOk,
        "allocate SSE");
  dev::GmeU8* d_src = static_cast<dev::GmeU8*>(raw_src);
  dev::GmeU8* d_cur = static_cast<dev::GmeU8*>(raw_cur);
  dev::GmeU8* d_dst = static_cast<dev::GmeU8*>(raw_dst);
  dev::GmeI64* d_sse = static_cast<dev::GmeI64*>(raw_sse);
  Check(cudaMemcpyAsync(d_src, source.data.data(), plane_bytes,
                        cudaMemcpyHostToDevice, nullptr) == cudaSuccess,
        "upload source plane");

  const cpu::GmParams models[] = {
      cpu::make_identity_gm(),
      Similarity(1.0, 0.0, 9.5, -5.25),
      Similarity(1.06, 0.0, 0.0, 0.0),
      Similarity(1.0, 2.0, 0.0, 0.0),
      Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0),
  };
  for (const cpu::GmParams& model : models) {
    const Plane current = WarpPlane(model, source, w, h);
    Check(cudaMemcpyAsync(d_cur, current.data.data(), plane_bytes,
                          cudaMemcpyHostToDevice, nullptr) == cudaSuccess,
          "upload current plane");
    std::string error;
    Check(dev::warp_plane_bilinear(buffers, ToDevParams(model), d_src, w, h,
                                   w, d_dst, w, h, w, 8, nullptr,
                                   &error) == dev::GmeStatus::kOk,
          "gpu warp plane");
    std::vector<std::uint8_t> warped(plane_bytes);
    Check(cudaMemcpyAsync(warped.data(), d_dst, plane_bytes,
                          cudaMemcpyDeviceToHost, nullptr) == cudaSuccess,
          "download warped plane");
    Check(cudaStreamSynchronize(nullptr) == cudaSuccess,
          "warp synchronisation");
    Check(warped == current.data, "gpu warp plane pixel parity");

    Check(dev::warp_plane_sse(buffers, ToDevParams(model), d_src, w, h, w,
                              d_cur, w, h, w, 8, d_sse, nullptr,
                              &error) == dev::GmeStatus::kOk,
          "gpu warp sse");
    dev::GmeI64 sse = 0;
    Check(cudaMemcpyAsync(&sse, d_sse, sizeof(sse), cudaMemcpyDeviceToHost,
                          nullptr) == cudaSuccess,
          "download sse");
    Check(cudaStreamSynchronize(nullptr) == cudaSuccess, "sse synchronisation");
    const std::int64_t reference =
        cpu::warp_plane_sse(model, source.data.data(), w, h, w,
                            current.data.data(), w, h, w, 8);
    CheckEqI64(sse, reference, "gpu warp sse parity");
  }
}

// ---------------------------------------------------------------------------
// Fit cases.
// ---------------------------------------------------------------------------

void TestFitsGpu(const dev::GmeBuffers& buffers) {
  const int width = 640;
  const int height = 360;
  const cpu::GmParams models[] = {
      Similarity(1.0, 0.0, 9.5, -5.25),
      Similarity(1.06, 0.0, 0.0, 0.0),
      Similarity(1.0, 2.0, 0.0, 0.0),
      Similarity(1.04, 1.5, 4.0, -3.0),
      Affine(1.04, 0.02, -0.03, 0.98, 4.0, -3.0),
  };
  int job = 0;
  for (const cpu::GmParams& model : models) {
    std::vector<cpu::GmPoint> points;
    for (int y = 8; y < height - 8; y += 24) {
      for (int x = 8; x < width - 8; x += 24) {
        const int sx = x * 8;
        const int sy = y * 8;
        std::int32_t rx = 0;
        std::int32_t ry = 0;
        cpu::project_gm_point(model, sx << 13, sy << 13, &rx, &ry);
        cpu::GmPoint point;
        point.src_x = sx;
        point.src_y = sy;
        point.dst_x = rx >> 13;
        point.dst_y = ry >> 13;
        point.weight = 1 + (x / 24) % 7;
        points.push_back(point);
      }
    }
    for (int variant = 0; variant < 3; ++variant) {
      cpu::GmOptions options;
      options.frame_width = width;
      options.frame_height = height;
      options.outlier_passes = variant;
      options.outlier_threshold_q8 = 8 + variant * 8;
      options.min_gain_percent = variant == 0 ? 0 : 1;
      options.force_integer_mv = variant == 2;
      options.model = variant == 2 ? cpu::GmFitModel::kRotZoom
                                   : cpu::GmFitModel::kAffine;
      RunPointsGpu(buffers, job, points, options, "gpu points fit");
      job = 1 - job;  // alternate jobs so both per-job buffers are exercised.
    }
  }

  // Block-MV fits at every block size.
  const int block_sizes[] = {4, 8, 16, 24, 32, 48, 64};
  for (const cpu::GmParams& model : models) {
    for (int block : block_sizes) {
      // A 4x4 grid over the full 640x360 frame has 14,400 blocks, above the
      // 8192-descriptor staging capacity.  Use a 320x180 crop for 4x4 so the
      // small-block success path is exercised; the full-frame 4x4 case is
      // covered by the emulation test's "too many" parity path.
      const int local_w = block == 4 ? 320 : width;
      const int local_h = block == 4 ? 180 : height;
      std::vector<cpu::GmBlockMv> blocks;
      for (int y = block / 2; y + block <= local_h; y += block) {
        for (int x = block / 2; x + block <= local_w; x += block) {
          const int centre_x = x * 8;
          const int centre_y = y * 8;
          std::int32_t rx = 0;
          std::int32_t ry = 0;
          cpu::project_gm_point(model, centre_x << 13, centre_y << 13, &rx,
                                &ry);
          cpu::GmBlockMv mv;
          mv.x = centre_x;
          mv.y = centre_y;
          mv.mv_x = (rx >> 13) - centre_x;
          mv.mv_y = (ry >> 13) - centre_y;
          mv.weight = 16;
          blocks.push_back(mv);
        }
      }
      cpu::GmBlockMv skipped{};
      skipped.weight = 0;
      blocks.push_back(skipped);
      cpu::GmOptions options;
      options.frame_width = width;
      options.frame_height = height;
      options.outlier_passes = 2;
      options.outlier_threshold_q8 = 16;
      RunBlocksGpu(buffers, job, blocks, options, "gpu block fit");
      job = 1 - job;
    }
  }

  // Warp fits, including radius and weight distribution.
  const int radii[] = {8, 16, 32, 64};
  for (const cpu::GmParams& model : models) {
    for (int radius : radii) {
      std::vector<cpu::GmWarpSample> warps;
      for (int y = 32; y + 32 <= height; y += 48) {
        for (int x = 32; x + 32 <= width; x += 48) {
          cpu::GmWarpSample warp;
          warp.x = x * 8;
          warp.y = y * 8;
          warp.radius = radius;
          warp.model = model;
          warp.weight = 7 + (x / 48) % 17;
          warps.push_back(warp);
        }
      }
      cpu::GmOptions options;
      options.frame_width = width;
      options.frame_height = height;
      options.outlier_passes = 2;
      options.outlier_threshold_q8 = 16;
      RunWarpsGpu(buffers, job, warps, options, "gpu warp fit");
      job = 1 - job;
    }
  }

  // Overflow and empty-input paths.
  {
    std::vector<cpu::GmBlockMv> overflow(1);
    overflow[0].x = 2147483647;
    overflow[0].mv_x = 1;
    overflow[0].weight = 1;
    cpu::GmOptions options;
    RunBlocksGpu(buffers, job, overflow, options, "gpu block overflow");

    std::vector<cpu::GmWarpSample> warp_overflow(1);
    warp_overflow[0].x = 2147483647;
    warp_overflow[0].radius = 16;
    warp_overflow[0].model = cpu::make_identity_gm();
    warp_overflow[0].weight = 1;
    RunWarpsGpu(buffers, job, warp_overflow, options, "gpu warp overflow");

    std::vector<cpu::GmPoint> empty;
    RunPointsGpu(buffers, job, empty, options, "gpu empty points");
    std::vector<cpu::GmBlockMv> empty_blocks;
    RunBlocksGpu(buffers, job, empty_blocks, options, "gpu empty blocks");
    std::vector<cpu::GmWarpSample> empty_warps;
    RunWarpsGpu(buffers, job, empty_warps, options, "gpu empty warps");

    // Too many correspondences: options.max_points is below the point count.
    std::vector<cpu::GmPoint> points;
    for (int i = 0; i < 200; ++i) {
      cpu::GmPoint point;
      point.src_x = (i % 20) * 32;
      point.src_y = (i / 20) * 32;
      point.dst_x = point.src_x + 4;
      point.dst_y = point.src_y - 2;
      point.weight = 1;
      points.push_back(point);
    }
    cpu::GmOptions too_many;
    too_many.max_points = 100;
    RunPointsGpu(buffers, job, points, too_many, "gpu too many");
  }
}

// ---------------------------------------------------------------------------
// Synthetic and real frames.
// ---------------------------------------------------------------------------

void TestSyntheticGpu(const dev::GmeBuffers& buffers) {
  const int w = 640;
  const int h = 360;
  const Plane source = MakeTexture(w, h);
  cpu::GmOptions options;
  options.frame_width = w;
  options.frame_height = h;
  options.outlier_passes = 3;
  options.outlier_threshold_q8 = 16;

  struct Case {
    const char* name;
    cpu::GmParams model;
  };
  const Case cases[] = {
      {"pan", Similarity(1.0, 0.0, 9.5, -5.25)},
      {"zoom", Similarity(1.06, 0.0, 0.0, 0.0)},
      {"rotate", Similarity(1.0, 2.0, 0.0, 0.0)},
      {"pan+zoom+rotate", Similarity(1.04, 1.5, 4.0, -3.0)},
  };
  MatchOptions match;
  match.block = 24;
  match.search = 24;
  match.sample_step = 2;
  for (const Case& item : cases) {
    const Plane current = WarpPlane(item.model, source, w, h);
    const std::vector<cpu::GmBlockMv> blocks = MatchBlocks(source, current, match);
    const dev::GmeEstimate actual =
        RunBlocksGpu(buffers, 0, blocks, options, item.name);
    std::printf(
        "  gpu synthetic %-16s blocks=%zu used=%d type=%d gain=%.2f dB\n",
        item.name, blocks.size(), actual.diag.points_used, actual.params.type,
        actual.diag.gain_db);
  }
}

void TestReal720pGpu(const dev::GmeBuffers& buffers,
                     const std::string& repo_root, bool quick) {
  const std::string path = repo_root + "/testdata/shine_1280x720_60f.y4m";
  std::vector<Plane> frames;
  if (!LoadY4mFrames(path, 11, &frames) || frames.size() < 2) {
    std::printf("  gpu real 720p: skipped (no testdata)\n");
    return;
  }
  cpu::GmOptions options;
  options.frame_width = frames[0].w;
  options.frame_height = frames[0].h;
  options.outlier_passes = 3;
  options.outlier_threshold_q8 = 24;

  auto analyze = [&](const Plane& reference, const Plane& current,
                     const char* label, int search) {
    MatchOptions match;
    match.block = 16;
    match.search = search;
    match.sample_step = 2;
    const std::vector<cpu::GmBlockMv> blocks =
        MatchBlocks(reference, current, match);
    const dev::GmeEstimate actual =
        RunBlocksGpu(buffers, 0, blocks, options, label);
    std::printf("  gpu real 720p %-12s blocks=%zu used=%d type=%d gain=%.2f "
                "dB\n",
                label, blocks.size(), actual.diag.points_used,
                actual.params.type, actual.diag.gain_db);
  };

  analyze(frames[0], frames[1], "pair 0/1", 16);
  if (!quick && frames.size() >= 11) {
    analyze(frames[0], frames[10], "pair 0/10", 32);
  }

  MatchOptions match;
  match.block = 16;
  match.search = 32;
  match.sample_step = 2;
  const cpu::GmParams known = Similarity(1.02, 0.5, 3.0, -2.0);
  const Plane warped = WarpPlane(known, frames[0], frames[0].w, frames[0].h);
  const std::vector<cpu::GmBlockMv> blocks =
      MatchBlocks(frames[0], warped, match);
  const dev::GmeEstimate actual =
      RunBlocksGpu(buffers, 0, blocks, options, "real 720p known warp");
  std::printf("  gpu real 720p known warp: blocks=%zu p0=%d p1=%d gain=%.2f "
              "dB\n",
              blocks.size(), actual.params.p[0], actual.params.p[1],
              actual.diag.gain_db);
}

}  // namespace

int main(int argc, char** argv) {
  std::string repo_root = ".";
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") {
      quick = true;
    } else {
      repo_root = arg;
    }
  }

  std::printf("gme_test: CUDA GME real-GPU parity\n");
  int device = 0;
  Check(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice");
  cudaDeviceProp properties{};
  Check(cudaGetDeviceProperties(&properties, device) == cudaSuccess,
        "cudaGetDeviceProperties");
  std::printf("  device: %s (cc %d.%d)\n", properties.name, properties.major,
              properties.minor);

  const dev::GmeConfig config{8192, 8192, 2048, 2};
  const std::uint64_t need = dev::required_device_bytes(config);
  Check(need > 0, "required_device_bytes positive");

  // One-byte-short: a separate arena is intentional; the main pipeline below
  // uses exactly one arena.
  {
    av2enc::cuda::Arena short_arena;
    Check(short_arena.init(need - 1) == av2enc::cuda::Status::kOk,
          "short arena init");
    dev::GmeBuffers short_buffers;
    std::uint64_t reported = 0;
    std::string error;
    Check(dev::allocate_checked(short_arena, config, &short_buffers, &reported,
                                &error) == dev::GmeStatus::kArenaTooSmall,
          "one byte short -> kArenaTooSmall");
    CheckEqI64(static_cast<long long>(reported),
               static_cast<long long>(need), "short reports need");
    CheckEqI64(static_cast<long long>(short_arena.used()), 0,
               "short arena bump unchanged");
    Check(short_buffers.d_points == nullptr, "short leaves no buffers");
  }

  const std::uint64_t slack = 16ull * 1024ull * 1024ull;
  av2enc::cuda::Arena arena;
  Check(arena.init(need + slack) == av2enc::cuda::Status::kOk,
        "main arena init");
  dev::GmeBuffers buffers;
  std::uint64_t reported = 0;
  std::string error;
  Check(dev::allocate_checked(arena, config, &buffers, &reported, &error) ==
            dev::GmeStatus::kOk,
        "allocate GME buffers");
  CheckEqI64(static_cast<long long>(reported),
             static_cast<long long>(need), "reported need");
  CheckEqI64(static_cast<long long>(arena.used()),
             static_cast<long long>(need), "used == required");
  std::printf("  required_device_bytes(%dx%d pts, %d blocks, %d warps, %d "
              "jobs) = %llu\n",
              config.max_points, config.max_points, config.max_blocks,
              config.max_warps, config.max_jobs,
              static_cast<unsigned long long>(need));

  TestProjectionGpu(buffers, arena);
  TestQualityProxyGpu(buffers, arena);
  TestFitsGpu(buffers);
  TestSyntheticGpu(buffers);
  TestReal720pGpu(buffers, repo_root, quick);

  // Two consecutive launches with no growth.  The projection and quality
  // buffers are already allocated, so record the watermark now.
  {
    std::vector<cpu::GmPoint> points;
    for (int i = 0; i < 300; ++i) {
      cpu::GmPoint point;
      point.src_x = (i % 25) * 24;
      point.src_y = (i / 25) * 24;
      point.dst_x = point.src_x + 12;
      point.dst_y = point.src_y - 8;
      point.weight = 1 + i % 5;
      points.push_back(point);
    }
    std::vector<dev::GmePoint> device_points;
    for (const cpu::GmPoint& point : points) {
      device_points.push_back(ToDevPoint(point));
    }
    cpu::GmOptions options;
    options.outlier_passes = 2;
    const std::uint64_t used = arena.used();
    const std::uint64_t peak = arena.peak_used();
    Check(dev::stage_points(buffers, 0, device_points.data(),
                            static_cast<int>(device_points.size())) ==
              dev::GmeStatus::kOk,
          "no-growth stage");
    Check(dev::estimate_points(buffers, 0,
                               static_cast<int>(device_points.size()),
                               ToDevOptions(options)) == dev::GmeStatus::kOk,
          "no-growth first launch");
    Check(dev::estimate_points(buffers, 0,
                               static_cast<int>(device_points.size()),
                               ToDevOptions(options)) == dev::GmeStatus::kOk,
          "no-growth second launch");
    CheckEqI64(static_cast<long long>(arena.used()),
               static_cast<long long>(used), "no growth after two launches");
    CheckEqI64(static_cast<long long>(arena.peak_used()),
               static_cast<long long>(peak), "no peak growth after launches");
    dev::GmeEstimate estimate;
    Check(dev::download_estimate(buffers, 0, &estimate) ==
              dev::GmeStatus::kOk,
          "no-growth download");
  }

  dev::deallocate(arena, buffers);
  CheckEqI64(static_cast<long long>(arena.used()), 0, "deallocate zero");

  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
