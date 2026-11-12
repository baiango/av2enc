// M1 luma candidate search: CPU parity, gates, batching, chunks, timing.
//
// The reference is the real CPU intra module (`src/encoder/intra_modes`) driven
// through the same candidate loop and strict-less-than tie-break as
// `still_encoder.cpp::choose_luma_mode`. The test compares:
//   * best mode / angle / SSE per block
//   * SSE of every active candidate (debug array)
//   * identical results for the same blocks split into candidate chunks
//   * synthetic blocks plus real 64x64 and 640x360 frames, including blocks
//     that extend past the visible bottom/right edges and 16x64 / 32x8 rects.
//
// Build/run: ./build_luma_search_test.sh
//
// Media defaults:
//   testdata/basic_64x64_1f.y4m
//   testdata/shine_640x360_1f.y4m
// Override with --y4m64 PATH / --y4m360 PATH, or --allow-missing-media.

#include "luma_search.h"

#include "codec/intra/modes/intra_modes.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ls = av2enc::cuda::still::search;
namespace ci = av2enc::intra;

// ---------------------------------------------------------------------------
// Checks / stats
// ---------------------------------------------------------------------------

namespace {

[[noreturn]] void fatal(const char* expr, cudaError_t err, const char* file,
                        int line) {
  std::fprintf(stderr, "CUDA error: %s (%d) from %s at %s:%d\n",
               cudaGetErrorString(err), static_cast<int>(err), expr, file,
               line);
  std::exit(1);
}

#define CUDA_CHECK(expr)                                                  \
  do {                                                                    \
    const cudaError_t err_ = (expr);                                      \
    if (err_ != cudaSuccess) {                                            \
      fatal(#expr, err_, __FILE__, __LINE__);                             \
    }                                                                     \
  } while (false)

#define CPU_CHECK(expr)                                                   \
  do {                                                                    \
    if (!(expr)) {                                                        \
      std::fprintf(stderr, "CPU reference failed: %s at %s:%d\n", #expr,  \
                   __FILE__, __LINE__);                                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (false)

struct Stats {
  long long blocks = 0;
  long long samples = 0;
  long long candidates = 0;
  long long decision_mismatches = 0;
  long long sse_mismatches = 0;
  long long candidate_sse_mismatches = 0;

  void add(const Stats& other) {
    blocks += other.blocks;
    samples += other.samples;
    candidates += other.candidates;
    decision_mismatches += other.decision_mismatches;
    sse_mismatches += other.sse_mismatches;
    candidate_sse_mismatches += other.candidate_sse_mismatches;
  }
  bool ok() const {
    return decision_mismatches == 0 && sse_mismatches == 0 &&
           candidate_sse_mismatches == 0;
  }
};

void print_stats(const char* label, const Stats& s) {
  std::printf(
      "  %-34s blocks=%-6lld samples=%-10lld cand=%-8lld "
      "mode/sse/cand-sse mism=%lld/%lld/%lld  %s\n",
      label, s.blocks, s.samples, s.candidates, s.decision_mismatches,
      s.sse_mismatches, s.candidate_sse_mismatches, s.ok() ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Planes / frames
// ---------------------------------------------------------------------------

struct Plane {
  int w = 0;
  int h = 0;
  int stride = 0;  // >= w
  std::vector<std::uint8_t> data;

  void reset(int width, int height, int pad = 0) {
    w = width;
    h = height;
    stride = width + pad;
    data.assign(static_cast<std::size_t>(stride) *
                    static_cast<std::size_t>(h),
                0);
  }
  std::uint8_t& at(int x, int y) {
    return data[static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
};

struct Y4mFrame {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> y;
};

bool load_y4m_y(const std::string& path, Y4mFrame& frame) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string line;
  if (!std::getline(in, line) || line.rfind("YUV4MPEG2", 0) != 0) {
    return false;
  }
  int w = 0;
  int h = 0;
  std::size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() && line[pos] == ' ') {
      ++pos;
    }
    if (pos >= line.size()) {
      break;
    }
    const char tag = line[pos++];
    int value = 0;
    bool have_digits = false;
    while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
      value = value * 10 + (line[pos] - '0');
      ++pos;
      have_digits = true;
    }
    if (tag == 'W' && have_digits) {
      w = value;
    } else if (tag == 'H' && have_digits) {
      h = value;
    }
  }
  if (w <= 0 || h <= 0) {
    return false;
  }
  if (!std::getline(in, line) || line.rfind("FRAME", 0) != 0) {
    return false;
  }
  frame.w = w;
  frame.h = h;
  frame.y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  in.read(reinterpret_cast<char*>(frame.y.data()),
          static_cast<std::streamsize>(frame.y.size()));
  return in.gcount() == static_cast<std::streamsize>(frame.y.size());
}

std::uint8_t blur_recon(const std::uint8_t* src, int w, int h, int x, int y) {
  const int xl = x > 0 ? x - 1 : 0;
  const int xr = x + 1 < w ? x + 1 : w - 1;
  const int yu = y > 0 ? y - 1 : 0;
  const int yd = y + 1 < h ? y + 1 : h - 1;
  const auto s = [&](int xx, int yy) {
    return static_cast<int>(
        src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
            static_cast<std::size_t>(xx)]);
  };
  const int acc = s(xl, y) + 2 * s(x, y) + s(xr, y) + s(x, yu) + s(x, yd);
  return static_cast<std::uint8_t>((acc + 3) / 6);
}

Plane make_recon(const Y4mFrame& frame, int pad = 0) {
  Plane plane;
  plane.reset(frame.w, frame.h, pad);
  for (int y = 0; y < frame.h; ++y) {
    for (int x = 0; x < frame.w; ++x) {
      plane.at(x, y) = blur_recon(frame.y.data(), frame.w, frame.h, x, y);
    }
  }
  return plane;
}

Plane make_source(const Y4mFrame& frame) {
  Plane plane;
  plane.reset(frame.w, frame.h);
  for (int y = 0; y < frame.h; ++y) {
    std::memcpy(plane.data.data() + static_cast<std::size_t>(y) *
                                        static_cast<std::size_t>(plane.stride),
                frame.y.data() + static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(frame.w),
                static_cast<std::size_t>(frame.w));
  }
  return plane;
}

std::uint32_t lcg(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

// Synthetic plane content styles.
Plane make_texture(int w, int h, int style, std::uint32_t seed) {
  Plane plane;
  plane.reset(w, h);
  std::uint32_t state = seed | 1u;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int value = 0;
      switch (style) {
        case 0:  // random
          value = static_cast<int>(lcg(state) >> 24);
          break;
        case 1:  // diagonal gradient
          value = (x * 3 + y * 5 + 17) & 0xff;
          break;
        case 2:  // constant
          value = 128;
          break;
        default:  // checker + noise
          value = ((x / 4 + y / 4) & 1) != 0 ? 200 : 40;
          value += static_cast<int>(lcg(state) >> 28);
          value = value < 0 ? 0 : (value > 255 ? 255 : value);
          break;
      }
      plane.at(x, y) = static_cast<std::uint8_t>(value);
    }
  }
  return plane;
}

// ---------------------------------------------------------------------------
// Block construction
// ---------------------------------------------------------------------------

int sb_units_after(int origin, int extent, int limit, int block_end) {
  const int sb0 = (origin / 64) * 64;
  const int sb1 = std::min(sb0 + 64, limit);
  (void)extent;
  return std::max(0, (sb1 - block_end) / 4);
}

ls::LumaSearchBlock make_block_case(const Plane& recon, const Plane& source,
                                    int x0, int y0, int w, int h,
                                    int filter_type_above,
                                    int filter_type_left) {
  ls::LumaSearchBlock b;
  b.recon_plane = recon.data.data();
  b.recon_stride = recon.stride;
  b.x0 = x0;
  b.y0 = y0;
  b.width = w;
  b.height = h;
  b.max_x = recon.w - 1;
  b.max_y = recon.h - 1;
  b.have_left = x0 > 0 ? 1 : 0;
  b.have_above = y0 > 0 ? 1 : 0;
  b.num4_above_right = sb_units_after(x0, w, b.max_x, x0 + w);
  b.num4_below_left = sb_units_after(y0, h, b.max_y, y0 + h);
  b.filter_type_above = filter_type_above;
  b.filter_type_left = filter_type_left;
  b.source_plane = source.data.data();
  b.source_stride = source.stride;
  b.source_max_x = source.w - 1;
  b.source_max_y = source.h - 1;
  return b;
}

// ---------------------------------------------------------------------------
// CPU reference: the real CPU intra module + choose_luma_mode's loop
// ---------------------------------------------------------------------------

struct CpuSearch {
  int mode = 0;
  int angle_delta = 0;
  std::int64_t sse = 0;
  std::vector<std::int64_t> candidate_sse;  // active candidates, CPU order
};

CpuSearch cpu_search(const ls::LumaSearchBlock& b,
                     const ls::LumaSearchGates& gates) {
  CpuSearch result;
  if ((gates.flags & ls::kGateDcOnly) != 0u) {
    result.mode = static_cast<int>(ci::Mode::kDc);
    result.angle_delta = 0;
    result.sse = 0;
    return result;
  }

  ci::NeighborRequest request;
  request.recon = ci::PlaneView{b.recon_plane, b.recon_stride};
  request.x = b.x0;
  request.y = b.y0;
  request.w = b.width;
  request.h = b.height;
  request.max_x = b.max_x;
  request.max_y = b.max_y;
  request.have_left = b.have_left != 0;
  request.have_above = b.have_above != 0;
  request.num4_above_right = b.num4_above_right;
  request.num4_below_left = b.num4_below_left;
  request.color_plane = 0;

  ci::IntraEdges edges;
  CPU_CHECK(ci::prepare_edges(request, 8, edges) == ci::IntraStatus::kOk);

  ci::ModeCandidate candidates[ci::kMaxModeCandidates]{};
  const int count = ci::build_mode_candidates(candidates, true);
  CPU_CHECK(count == ls::kMaxLumaCandidates);

  const int w = b.width;
  const int h = b.height;
  std::vector<std::uint8_t> pred(static_cast<std::size_t>(w) *
                                 static_cast<std::size_t>(h));
  result.sse = std::numeric_limits<std::int64_t>::max();

  for (int i = 0; i < count; ++i) {
    const ls::LumaCandidate candidate{
        static_cast<int>(candidates[i].mode),
        static_cast<int>(candidates[i].angle_delta)};
    if (!ls::luma_candidate_enabled(candidate, gates.flags)) {
      continue;
    }
    ci::PredictParams params;
    params.mode = candidates[i].mode;
    params.angle_delta = candidates[i].angle_delta;
    params.enable_edge_filter = gates.enable_edge_filter != 0;
    params.above_neighbor_is_smooth = b.filter_type_above != 0;
    params.left_neighbor_is_smooth = b.filter_type_left != 0;
    CPU_CHECK(ci::predict_intra(edges, params, pred.data(),
                                static_cast<std::size_t>(w)) ==
              ci::IntraStatus::kOk);

    std::int64_t sse = 0;
    for (int y = 0; y < h; ++y) {
      const int sy =
          std::min(std::max(b.y0 + y, 0), b.source_max_y);
      for (int x = 0; x < w; ++x) {
        const int sx =
            std::min(std::max(b.x0 + x, 0), b.source_max_x);
        const int src = static_cast<int>(
            b.source_plane[static_cast<std::size_t>(sy) *
                               static_cast<std::size_t>(b.source_stride) +
                           static_cast<std::size_t>(sx)]);
        const int value = static_cast<int>(
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)]);
        const int diff = src - value;
        sse += static_cast<std::int64_t>(diff) * diff;
      }
    }
    result.candidate_sse.push_back(sse);
    if (sse < result.sse) {
      result.sse = sse;
      result.mode = candidate.mode;
      result.angle_delta = candidate.angle_delta;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// GPU runner
// ---------------------------------------------------------------------------

// Copies each distinct plane referenced by the blocks once and rewrites the
// block descriptors to device pointers.
class DevicePlanes {
 public:
  ~DevicePlanes() {
    for (const auto& entry : uploads_) {
      if (entry.device != nullptr) {
        cudaFree(entry.device);
      }
    }
  }

  const std::uint8_t* upload(const std::uint8_t* host,
                             std::size_t bytes) {
    if (host == nullptr) {
      return nullptr;
    }
    for (auto& entry : uploads_) {
      if (entry.host == host) {
        if (entry.bytes < bytes) {
          CUDA_CHECK(cudaFree(entry.device));
          CUDA_CHECK(cudaMalloc(&entry.device, bytes));
          CUDA_CHECK(cudaMemcpy(entry.device, host, bytes,
                                cudaMemcpyHostToDevice));
          entry.bytes = bytes;
        }
        return entry.device;
      }
    }
    void* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, bytes));
    CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    uploads_.push_back(
        Upload{host, static_cast<std::uint8_t*>(device), bytes});
    return uploads_.back().device;
  }

  std::vector<ls::LumaSearchBlock> to_device(
      const std::vector<ls::LumaSearchBlock>& blocks) {
    std::vector<ls::LumaSearchBlock> copy = blocks;
    for (ls::LumaSearchBlock& b : copy) {
      b.recon_plane = upload(
          b.recon_plane, static_cast<std::size_t>(b.recon_stride) *
                             static_cast<std::size_t>(b.max_y + 1));
      b.source_plane = upload(
          b.source_plane, static_cast<std::size_t>(b.source_stride) *
                              static_cast<std::size_t>(b.source_max_y + 1));
    }
    return copy;
  }

 private:
  struct Upload {
    const std::uint8_t* host;
    std::uint8_t* device;
    std::size_t bytes;
  };
  std::vector<Upload> uploads_;
};

struct RunOptions {
  bool check_candidate_sse = true;
  bool verbose_mismatch = true;
  const char* label = "";
};

Stats run_batch(const std::vector<ls::LumaSearchBlock>& blocks,
                const ls::LumaSearchGates& gates, int candidates_per_chunk,
                const RunOptions& options) {
  Stats stats;
  if (blocks.empty()) {
    return stats;
  }
  int max_area = 0;
  for (const ls::LumaSearchBlock& b : blocks) {
    max_area = std::max(max_area, b.width * b.height);
  }
  const int max_blocks = static_cast<int>(blocks.size());

  DevicePlanes uploads;
  const std::vector<ls::LumaSearchBlock> device_blocks =
      uploads.to_device(blocks);

  const ls::LumaSearchLayout layout =
      ls::luma_search_layout(max_blocks, max_area, candidates_per_chunk);
  void* arena = nullptr;
  CUDA_CHECK(cudaMalloc(&arena, layout.total_bytes));
  const ls::LumaSearchScratch scratch = ls::luma_search_make_scratch(arena,
                                                                    layout);

  ls::LumaSearchBlock* d_blocks = nullptr;
  ls::LumaSearchResult* d_results = nullptr;
  std::int64_t* d_debug = nullptr;
  CUDA_CHECK(cudaMalloc(&d_blocks, sizeof(ls::LumaSearchBlock) *
                                       static_cast<std::size_t>(max_blocks)));
  CUDA_CHECK(cudaMalloc(&d_results, sizeof(ls::LumaSearchResult) *
                                        static_cast<std::size_t>(max_blocks)));
  const int active_total = ls::luma_search_candidate_count(gates);
  const int debug_stride = ls::kMaxLumaCandidates;
  if (options.check_candidate_sse &&
      (gates.flags & ls::kGateDcOnly) == 0u) {
    CUDA_CHECK(cudaMalloc(&d_debug, sizeof(std::int64_t) *
                                        static_cast<std::size_t>(max_blocks) *
                                        static_cast<std::size_t>(debug_stride)));
    CUDA_CHECK(cudaMemset(d_debug, 0,
                          sizeof(std::int64_t) *
                              static_cast<std::size_t>(max_blocks) *
                              static_cast<std::size_t>(debug_stride)));
  }

  CUDA_CHECK(cudaMemcpy(d_blocks, device_blocks.data(),
                        sizeof(ls::LumaSearchBlock) *
                            static_cast<std::size_t>(max_blocks),
                        cudaMemcpyHostToDevice));

  ls::LumaSearchDebug debug{};
  debug.d_candidate_sse = d_debug;
  debug.stride = debug_stride;

  CUDA_CHECK(ls::luma_search(d_blocks, max_blocks, gates, scratch, d_results,
                             d_debug != nullptr ? &debug : nullptr, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ls::LumaSearchResult> results(
      static_cast<std::size_t>(max_blocks));
  CUDA_CHECK(cudaMemcpy(results.data(), d_results,
                        sizeof(ls::LumaSearchResult) *
                            static_cast<std::size_t>(max_blocks),
                        cudaMemcpyDeviceToHost));
  std::vector<std::int64_t> debug_values;
  if (d_debug != nullptr) {
    debug_values.resize(static_cast<std::size_t>(max_blocks) *
                        static_cast<std::size_t>(debug_stride));
    CUDA_CHECK(cudaMemcpy(debug_values.data(), d_debug,
                          sizeof(std::int64_t) * debug_values.size(),
                          cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < max_blocks; ++i) {
    const CpuSearch expected = cpu_search(blocks[static_cast<std::size_t>(i)],
                                          gates);
    const ls::LumaSearchResult& actual = results[static_cast<std::size_t>(i)];
    const ls::LumaSearchBlock& block = blocks[static_cast<std::size_t>(i)];
    ++stats.blocks;
    stats.samples += block.width * block.height;
    stats.candidates += static_cast<long long>(expected.candidate_sse.size());

    const bool decision_ok =
        actual.mode == expected.mode &&
        actual.angle_delta == expected.angle_delta;
    const bool sse_ok = actual.sse == expected.sse;
    if (!decision_ok) {
      ++stats.decision_mismatches;
    }
    if (!sse_ok) {
      ++stats.sse_mismatches;
    }
    bool cand_ok = true;
    if (d_debug != nullptr) {
      for (std::size_t k = 0; k < expected.candidate_sse.size(); ++k) {
        const std::int64_t got =
            debug_values[static_cast<std::size_t>(i) *
                             static_cast<std::size_t>(debug_stride) +
                         k];
        if (got != expected.candidate_sse[k]) {
          cand_ok = false;
          ++stats.candidate_sse_mismatches;
          break;
        }
      }
    }
    if (options.verbose_mismatch && (!decision_ok || !sse_ok || !cand_ok)) {
      std::printf(
          "  MISMATCH %s block=%d xy=%d,%d wh=%dx%d max=%d,%d have=%d/%d "
          "num4=%d/%d ft=%d/%d edge=%d\n"
          "    gpu mode=%d delta=%d sse=%lld | cpu mode=%d delta=%d sse=%lld"
          " cand=%zu\n",
          options.label, i, block.x0, block.y0, block.width, block.height,
          block.max_x, block.max_y, block.have_left, block.have_above,
          block.num4_above_right, block.num4_below_left,
          block.filter_type_above, block.filter_type_left,
          gates.enable_edge_filter, actual.mode, actual.angle_delta,
          static_cast<long long>(actual.sse), expected.mode,
          expected.angle_delta, static_cast<long long>(expected.sse),
          expected.candidate_sse.size());
    }
  }
  (void)active_total;

  CUDA_CHECK(cudaFree(d_debug));
  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cudaFree(d_blocks));
  CUDA_CHECK(cudaFree(arena));
  return stats;
}

// ---------------------------------------------------------------------------
// Candidate gate tests
// ---------------------------------------------------------------------------

bool gates_test() {
  struct Case {
    unsigned flags;
    int expected;
  };
  const Case cases[] = {
      {ls::kGateNone, 61},
      {ls::kGateBaseOnly, 13},
      {ls::kGateNoDirectional, 5},
      {ls::kGateBaseOnly | ls::kGateNoDirectional, 5},
      {ls::kGateDcOnly, 1},
      {ls::kGateDcOnly | ls::kGateBaseOnly | ls::kGateNoDirectional, 1},
  };
  bool ok = true;
  for (const Case& c : cases) {
    ls::LumaSearchGates gates;
    gates.flags = c.flags;
    const int count = ls::luma_search_candidate_count(gates);
    ls::LumaCandidate list[ls::kMaxLumaCandidates]{};
    const int written = ls::luma_search_candidates(gates, list, 61);
    std::printf("  gate flags=0x%x count=%d (expected %d) written=%d  %s\n",
                c.flags, count, c.expected, written,
                (count == c.expected && written == c.expected) ? "PASS"
                                                               : "FAIL");
    ok = ok && count == c.expected && written == c.expected;

    // Must be a subsequence of the full CPU candidate list.
    ci::ModeCandidate cpu[ci::kMaxModeCandidates]{};
    const int cpu_count = ci::build_mode_candidates(cpu, true);
    int cpu_index = 0;
    for (int i = 0; i < written; ++i) {
      bool found = false;
      while (cpu_index < cpu_count) {
        const ls::LumaCandidate candidate{
            static_cast<int>(cpu[cpu_index].mode),
            static_cast<int>(cpu[cpu_index].angle_delta)};
        ++cpu_index;
        if (candidate.mode == list[i].mode &&
            candidate.angle_delta == list[i].angle_delta &&
            ls::luma_candidate_enabled(candidate, c.flags)) {
          found = true;
          break;
        }
      }
      if (!found) {
        std::printf("    order/entry mismatch at %d (mode=%d delta=%d)\n", i,
                    list[i].mode, list[i].angle_delta);
        ok = false;
      }
    }
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Synthetic block matrix
// ---------------------------------------------------------------------------

struct Size {
  int w;
  int h;
};

Stats run_synthetic_matrix(const ls::LumaSearchGates& gates) {
  const Size sizes[] = {
      {4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64}, {16, 64},
      {64, 16}, {32, 8},  {8, 32},  {64, 32}, {32, 64}, {64, 8},
      {8, 64},
  };
  const int contents = 2;
  std::deque<Plane> owners;  // keeps plane storage alive for run_batch()
  std::vector<ls::LumaSearchBlock> blocks;
  // Keep each batch modest; the point is the matrix, not throughput.
  for (const Size& size : sizes) {
    for (int style = 0; style < contents; ++style) {
      const int plane_w = std::max(128, size.w + 64);
      const int plane_h = std::max(128, size.h + 64);
      owners.push_back(make_texture(plane_w, plane_h, style, 0x1234u + style));
      Plane& source = owners.back();
      owners.push_back(make_texture(plane_w, plane_h, 3 - style,
                                    0x9876u + style));
      Plane& recon = owners.back();
      // Positions: top-left, top edge, left edge, interior, bottom-right
      // overflow (reads are clamped, like partial M1 edge blocks).
      const int xs[] = {0, (plane_w - size.w) / 2, 0,
                        (plane_w - size.w) / 2, plane_w - size.w / 2};
      const int ys[] = {0, 0, (plane_h - size.h) / 2,
                        (plane_h - size.h) / 2, plane_h - size.h / 2};
      for (int p = 0; p < 5; ++p) {
        const int fta = ((xs[p] >> 2) + ys[p]) & 1;
        const int ftl = ((ys[p] >> 2) + xs[p] + 1) & 1;
        blocks.push_back(make_block_case(recon, source, xs[p], ys[p], size.w,
                                         size.h, fta, ftl));
      }
    }
  }
  // Extra filter-type combinations and extension counts on interior blocks.
  {
    owners.push_back(make_texture(160, 160, 3, 0x55u));
    Plane& source = owners.back();
    owners.push_back(make_texture(160, 160, 0, 0xaau));
    Plane& recon = owners.back();
    for (int fta = 0; fta < 2; ++fta) {
      for (int ftl = 0; ftl < 2; ++ftl) {
        ls::LumaSearchBlock b =
            make_block_case(recon, source, 32, 32, 64, 64, fta, ftl);
        b.num4_above_right = 4;
        b.num4_below_left = 8;
        blocks.push_back(b);
      }
    }
    // Flat source == reconstruction: many candidates tie at SSE 0. The CPU
    // tie-break must keep the first candidate in order (DC), so this catches
    // any `<` vs `<=` or ordering mistake.
    owners.push_back(make_texture(128, 128, 2, 0x77u));
    const Plane& flat = owners.back();
    for (int position = 0; position < 2; ++position) {
      for (int flags = 0; flags < 4; ++flags) {
        blocks.push_back(make_block_case(flat, flat, position * 32,
                                         position * 32, 32, 32,
                                         flags & 1, (flags >> 1) & 1));
      }
    }
    // Padded stride / padded plane (stride > width).
    Plane padded_recon;
    padded_recon.reset(160, 160, 11);
    for (int y = 0; y < 160; ++y) {
      for (int x = 0; x < 160; ++x) {
        padded_recon.at(x, y) = static_cast<std::uint8_t>((x * 7 + y * 3) ^ 0x5a);
      }
    }
    owners.push_back(std::move(padded_recon));
    Plane& padded = owners.back();
    blocks.push_back(make_block_case(padded, source, 16, 24, 32, 32, 1, 0));
  }
  return run_batch(blocks, gates, ls::kMaxLumaCandidates, RunOptions{});
}

// ---------------------------------------------------------------------------
// Real frames
// ---------------------------------------------------------------------------

struct RealFrameCase {
  std::string name;
  Plane source;
  Plane recon;
};

std::vector<ls::LumaSearchBlock> make_layout_blocks(
    const RealFrameCase& frame, const std::vector<Size>& sizes) {
  std::vector<ls::LumaSearchBlock> blocks;
  int index = 0;
  for (const Size& size : sizes) {
    const int step_x = std::max(1, size.w);
    const int step_y = std::max(1, size.h);
    for (int y = 0; y < frame.source.h; y += step_y) {
      for (int x = 0; x < frame.source.w; x += step_x) {
        const int fta = ((x >> 2) * 7 + (y >> 2) * 3 + index) & 1;
        const int ftl = ((x >> 2) * 5 + (y >> 2) * 11 + index) & 1;
        blocks.push_back(make_block_case(frame.recon, frame.source, x, y,
                                         size.w, size.h, fta, ftl));
        ++index;
      }
    }
  }
  return blocks;
}

Stats run_real_frame(const RealFrameCase& frame, const std::vector<Size>& sizes,
                     const ls::LumaSearchGates& gates) {
  const std::vector<ls::LumaSearchBlock> blocks =
      make_layout_blocks(frame, sizes);
  RunOptions options;
  options.label = frame.name.c_str();
  return run_batch(blocks, gates, ls::kMaxLumaCandidates, options);
}

// Wavefront levels over a 64x64 superblock grid: blocks with equal
// (sbx + sby) only read already-final reconstruction in a raster encoder, so a
// level is the legal cross-block batch.
Stats run_wavefront_levels(const RealFrameCase& frame,
                           const ls::LumaSearchGates& gates) {
  const int sb_cols = (frame.source.w + 63) / 64;
  const int sb_rows = (frame.source.h + 63) / 64;
  Stats total;
  for (int level = 0; level <= sb_cols - 1 + sb_rows - 1; ++level) {
    std::vector<ls::LumaSearchBlock> blocks;
    for (int sby = 0; sby < sb_rows; ++sby) {
      for (int sbx = 0; sbx < sb_cols; ++sbx) {
        if (sbx + sby != level) {
          continue;
        }
        const int x = sbx * 64;
        const int y = sby * 64;
        const int fta = (sbx * 3 + sby * 5) & 1;
        const int ftl = (sbx * 7 + sby) & 1;
        blocks.push_back(make_block_case(frame.recon, frame.source, x, y, 64,
                                         64, fta, ftl));
      }
    }
    RunOptions options;
    options.label = frame.name.c_str();
    total.add(run_batch(blocks, gates, ls::kMaxLumaCandidates, options));
  }
  return total;
}

// ---------------------------------------------------------------------------
// Chunk invariance
// ---------------------------------------------------------------------------

bool chunk_invariance(const std::vector<ls::LumaSearchBlock>& blocks,
                      const ls::LumaSearchGates& gates, const char* label) {
  const int max_blocks = static_cast<int>(blocks.size());
  int max_area = 0;
  for (const auto& b : blocks) {
    max_area = std::max(max_area, b.width * b.height);
  }
  DevicePlanes uploads;
  const std::vector<ls::LumaSearchBlock> device_blocks =
      uploads.to_device(blocks);
  auto run = [&](int chunk) {
    const ls::LumaSearchLayout layout =
        ls::luma_search_layout(max_blocks, max_area, chunk);
    void* arena = nullptr;
    CUDA_CHECK(cudaMalloc(&arena, layout.total_bytes));
    const ls::LumaSearchScratch scratch =
        ls::luma_search_make_scratch(arena, layout);
    ls::LumaSearchBlock* d_blocks = nullptr;
    ls::LumaSearchResult* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blocks, sizeof(ls::LumaSearchBlock) *
                                         static_cast<std::size_t>(max_blocks)));
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(ls::LumaSearchResult) *
                                          static_cast<std::size_t>(max_blocks)));
    CUDA_CHECK(cudaMemcpy(d_blocks, device_blocks.data(),
                          sizeof(ls::LumaSearchBlock) *
                              static_cast<std::size_t>(max_blocks),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(ls::luma_search(d_blocks, max_blocks, gates, scratch, d_results,
                               nullptr, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<ls::LumaSearchResult> results(
        static_cast<std::size_t>(max_blocks));
    CUDA_CHECK(cudaMemcpy(results.data(), d_results,
                          sizeof(ls::LumaSearchResult) *
                              static_cast<std::size_t>(max_blocks),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_blocks));
    CUDA_CHECK(cudaFree(arena));
    return results;
  };
  const std::vector<ls::LumaSearchResult> full = run(ls::kMaxLumaCandidates);
  const std::vector<ls::LumaSearchResult> chunked = run(7);
  bool ok = true;
  for (std::size_t i = 0; i < full.size(); ++i) {
    if (full[i].mode != chunked[i].mode ||
        full[i].angle_delta != chunked[i].angle_delta ||
        full[i].sse != chunked[i].sse) {
      std::printf("  CHUNK MISMATCH %s block=%zu mode %d/%d delta %d/%d "
                  "sse %lld/%lld\n",
                  label, i, full[i].mode, chunked[i].mode,
                  full[i].angle_delta, chunked[i].angle_delta,
                  static_cast<long long>(full[i].sse),
                  static_cast<long long>(chunked[i].sse));
      ok = false;
      break;
    }
  }
  std::printf("  chunk invariance %-22s blocks=%d chunk=61 vs 7  %s\n", label,
              max_blocks, ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------------
// Validation / edge cases
// ---------------------------------------------------------------------------

bool api_validation() {
  Plane recon = make_texture(64, 64, 1, 1);
  Plane source = make_texture(64, 64, 1, 2);
  const ls::LumaSearchBlock block =
      make_block_case(recon, source, 0, 0, 64, 64, 0, 0);
  const ls::LumaSearchLayout layout = ls::luma_search_layout(2, 64 * 64, 61);
  void* arena = nullptr;
  CUDA_CHECK(cudaMalloc(&arena, layout.total_bytes));
  const ls::LumaSearchScratch scratch = ls::luma_search_make_scratch(arena,
                                                                    layout);
  ls::LumaSearchBlock* d_blocks = nullptr;
  ls::LumaSearchResult* d_results = nullptr;
  CUDA_CHECK(cudaMalloc(&d_blocks, sizeof(ls::LumaSearchBlock) * 2));
  CUDA_CHECK(cudaMalloc(&d_results, sizeof(ls::LumaSearchResult) * 2));
  CUDA_CHECK(cudaMemcpy(d_blocks, &block, sizeof(block),
                        cudaMemcpyHostToDevice));

  ls::LumaSearchGates gates;
  bool ok = true;
  auto expect = [&](const char* what, cudaError_t got, cudaError_t want) {
    const bool pass = got == want;
    std::printf("  %-40s %s (%s)\n", what, pass ? "PASS" : "FAIL",
                cudaGetErrorString(got));
    ok = ok && pass;
  };
  expect("block_count = 0", ls::luma_search(nullptr, 0, gates, scratch,
                                            nullptr, nullptr, nullptr),
         cudaSuccess);
  expect("null blocks", ls::luma_search(nullptr, 1, gates, scratch, d_results,
                                        nullptr, nullptr),
         cudaErrorInvalidValue);
  expect("null results",
         ls::luma_search(d_blocks, 1, gates, scratch, nullptr, nullptr,
                         nullptr),
         cudaErrorInvalidValue);
  ls::LumaSearchScratch too_small = scratch;
  too_small.job_capacity = 1;  // needs 61
  expect("scratch too small",
         ls::luma_search(d_blocks, 1, gates, too_small, d_results, nullptr,
                         nullptr),
         cudaErrorInvalidValue);
  ls::LumaSearchDebug bad_debug{};
  CUDA_CHECK(cudaMalloc(&bad_debug.d_candidate_sse, sizeof(std::int64_t) * 61));
  bad_debug.stride = 60;  // needs 61
  expect("debug stride too small",
         ls::luma_search(d_blocks, 1, gates, scratch, d_results, &bad_debug,
                         nullptr),
         cudaErrorInvalidValue);
  CUDA_CHECK(cudaFree(bad_debug.d_candidate_sse));

  ls::LumaSearchGates bad_gates;
  bad_gates.flags = 1u << 7;
  expect("unknown gate flag",
         ls::luma_search(d_blocks, 1, bad_gates, scratch, d_results, nullptr,
                         nullptr),
         cudaErrorInvalidValue);

  // DC-only returns the CPU shortcut: mode 0, angle 0, sse 0.
  ls::LumaSearchGates dc_only;
  dc_only.flags = ls::kGateDcOnly;
  expect("dc-only launch",
         ls::luma_search(d_blocks, 1, dc_only, scratch, d_results, nullptr,
                         nullptr),
         cudaSuccess);
  ls::LumaSearchResult result{};
  CUDA_CHECK(cudaMemcpy(&result, d_results, sizeof(result),
                        cudaMemcpyDeviceToHost));
  const bool dc_ok = result.mode == static_cast<int>(ci::Mode::kDc) &&
                     result.angle_delta == 0 && result.sse == 0;
  std::printf("  %-40s %s\n", "dc-only result (0,0,0)",
              dc_ok ? "PASS" : "FAIL");
  ok = ok && dc_ok;

  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cudaFree(d_blocks));
  CUDA_CHECK(cudaFree(arena));
  return ok;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct Timing {
  double ms = 0.0;
  double blocks_per_s = 0.0;
  double msamples_per_s = 0.0;
};

Timing time_search(const char* label,
                   const std::vector<ls::LumaSearchBlock>& blocks,
                   const ls::LumaSearchGates& gates, int iterations) {
  int max_area = 0;
  long long samples = 0;
  for (const auto& b : blocks) {
    max_area = std::max(max_area, b.width * b.height);
    samples += static_cast<long long>(b.width) * b.height;
  }
  const int max_blocks = static_cast<int>(blocks.size());
  DevicePlanes uploads;
  const std::vector<ls::LumaSearchBlock> device_blocks =
      uploads.to_device(blocks);
  const ls::LumaSearchLayout layout =
      ls::luma_search_layout(max_blocks, max_area, ls::kMaxLumaCandidates);
  void* arena = nullptr;
  CUDA_CHECK(cudaMalloc(&arena, layout.total_bytes));
  const ls::LumaSearchScratch scratch = ls::luma_search_make_scratch(arena,
                                                                    layout);
  ls::LumaSearchBlock* d_blocks = nullptr;
  ls::LumaSearchResult* d_results = nullptr;
  CUDA_CHECK(cudaMalloc(&d_blocks, sizeof(ls::LumaSearchBlock) *
                                       static_cast<std::size_t>(max_blocks)));
  CUDA_CHECK(cudaMalloc(&d_results, sizeof(ls::LumaSearchResult) *
                                        static_cast<std::size_t>(max_blocks)));
  CUDA_CHECK(cudaMemcpy(d_blocks, device_blocks.data(),
                        sizeof(ls::LumaSearchBlock) *
                            static_cast<std::size_t>(max_blocks),
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(ls::luma_search(d_blocks, max_blocks, gates, scratch, d_results,
                               nullptr, nullptr));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(ls::luma_search(d_blocks, max_blocks, gates, scratch, d_results,
                               nullptr, nullptr));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  Timing timing;
  timing.ms = static_cast<double>(elapsed_ms) / iterations;
  timing.blocks_per_s =
      timing.ms > 0.0
          ? static_cast<double>(max_blocks) * 1000.0 / timing.ms
          : 0.0;
  timing.msamples_per_s =
      timing.ms > 0.0
          ? static_cast<double>(samples) * 61.0 * 1000.0 / timing.ms / 1e6
          : 0.0;
  std::printf(
      "  %-36s blocks=%-5d cand=61  %7.3f ms  %10.0f blk/s  "
      "%8.1f Msample-eval/s  scratch=%6.2f MiB\n",
      label, max_blocks, timing.ms, timing.blocks_per_s,
      timing.msamples_per_s,
      static_cast<double>(layout.total_bytes) / (1024.0 * 1024.0));

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cudaFree(d_blocks));
  CUDA_CHECK(cudaFree(arena));
  return timing;
}

std::vector<ls::LumaSearchBlock> make_grid(int plane_w, int plane_h, int bw,
                                           int bh, Plane& recon,
                                           Plane& source) {
  std::vector<ls::LumaSearchBlock> blocks;
  int index = 0;
  for (int y = 0; y < plane_h; y += bh) {
    for (int x = 0; x < plane_w; x += bw) {
      blocks.push_back(make_block_case(recon, source, x, y, bw, bh,
                                       index & 1, (index >> 1) & 1));
      ++index;
    }
  }
  return blocks;
}

// Simulates the real call pattern: one luma_search per 64x64 superblock
// wavefront level. Blocks in a level read only already-final reconstruction.
void time_wavefront(const char* label, int plane_w, int plane_h,
                    int iterations) {
  std::deque<Plane> owners;
  owners.push_back(make_texture(plane_w, plane_h, 1, 0x600du));
  Plane& recon = owners.back();
  owners.push_back(make_texture(plane_w, plane_h, 3, 0x600eu));
  Plane& source = owners.back();

  const int sb_cols = (plane_w + 63) / 64;
  const int sb_rows = (plane_h + 63) / 64;

  struct Level {
    std::vector<ls::LumaSearchBlock> device_blocks;
    ls::LumaSearchBlock* d_blocks = nullptr;
    ls::LumaSearchResult* d_results = nullptr;
    void* arena = nullptr;
    ls::LumaSearchScratch scratch{};
    int blocks = 0;
    int max_area = 0;
  };

  DevicePlanes uploads;
  std::vector<Level> levels;
  int total_blocks = 0;
  int max_level_blocks = 0;
  std::size_t max_scratch = 0;
  for (int level = 0; level <= sb_cols - 1 + sb_rows - 1; ++level) {
    Level lv;
    std::vector<ls::LumaSearchBlock> host_blocks;
    int index = 0;
    for (int sby = 0; sby < sb_rows; ++sby) {
      for (int sbx = 0; sbx < sb_cols; ++sbx) {
        if (sbx + sby != level) {
          continue;
        }
        host_blocks.push_back(make_block_case(
            recon, source, sbx * 64, sby * 64, 64, 64, index & 1,
            (index >> 1) & 1));
        ++index;
      }
    }
    lv.blocks = static_cast<int>(host_blocks.size());
    lv.device_blocks = uploads.to_device(host_blocks);
    lv.max_area = 64 * 64;
    const ls::LumaSearchLayout layout = ls::luma_search_layout(
        lv.blocks, lv.max_area, ls::kMaxLumaCandidates);
    max_scratch = std::max(max_scratch, layout.total_bytes);
    CUDA_CHECK(cudaMalloc(&lv.arena, layout.total_bytes));
    lv.scratch = ls::luma_search_make_scratch(lv.arena, layout);
    CUDA_CHECK(cudaMalloc(&lv.d_blocks, sizeof(ls::LumaSearchBlock) *
                                            static_cast<std::size_t>(
                                                lv.blocks)));
    CUDA_CHECK(cudaMalloc(&lv.d_results, sizeof(ls::LumaSearchResult) *
                                             static_cast<std::size_t>(
                                                 lv.blocks)));
    CUDA_CHECK(cudaMemcpy(lv.d_blocks, lv.device_blocks.data(),
                          sizeof(ls::LumaSearchBlock) *
                              static_cast<std::size_t>(lv.blocks),
                          cudaMemcpyHostToDevice));
    total_blocks += lv.blocks;
    max_level_blocks = std::max(max_level_blocks, lv.blocks);
    levels.push_back(std::move(lv));
  }

  ls::LumaSearchGates gates;
  auto run_once = [&]() {
    for (Level& lv : levels) {
      CUDA_CHECK(ls::luma_search(lv.d_blocks, lv.blocks, gates, lv.scratch,
                                 lv.d_results, nullptr, nullptr));
    }
  };
  for (int i = 0; i < 2; ++i) {
    run_once();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    run_once();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  const double ms = static_cast<double>(elapsed_ms) / iterations;
  std::printf(
      "  %-36s blocks=%-5d levels=%-3zu maxlvl=%-3d cand=61  %7.3f ms/frame"
      "  %10.0f blk/s  scratch/level=%6.2f MiB\n",
      label, total_blocks, levels.size(), max_level_blocks, ms,
      ms > 0.0 ? static_cast<double>(total_blocks) * 1000.0 / ms : 0.0,
      static_cast<double>(max_scratch) / (1024.0 * 1024.0));

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  for (Level& lv : levels) {
    CUDA_CHECK(cudaFree(lv.d_results));
    CUDA_CHECK(cudaFree(lv.d_blocks));
    CUDA_CHECK(cudaFree(lv.arena));
  }
}


// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Options {
  std::string y4m64 = "testdata/basic_64x64_1f.y4m";
  std::string y4m360 = "testdata/shine_640x360_1f.y4m";
  bool allow_missing_media = false;
  bool quick = false;
  bool timing = true;
  bool timing_only = false;
};

bool parse_args(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--y4m64" && i + 1 < argc) {
      options.y4m64 = argv[++i];
    } else if (arg == "--y4m360" && i + 1 < argc) {
      options.y4m360 = argv[++i];
    } else if (arg == "--allow-missing-media") {
      options.allow_missing_media = true;
    } else if (arg == "--quick") {
      options.quick = true;
    } else if (arg == "--no-timing") {
      options.timing = false;
    } else if (arg == "--timing-only") {
      options.timing_only = true;
    } else if (arg == "-h" || arg == "--help") {
      std::printf(
          "usage: luma_search_test [--y4m64 PATH] [--y4m360 PATH] "
          "[--allow-missing-media] [--quick] [--no-timing] [--timing-only]\n");
      return false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return false;
    }
  }
  return true;
}

bool run_parity_tests(const Options& options, const unsigned* gate_sets,
                      const char* const* gate_names, const int* edge_settings) {
  bool ok = true;
  std::printf("\n-- synthetic block matrix --\n");
  for (int edge_index = 0; edge_index < 2; ++edge_index) {
    const int edge = edge_settings[edge_index];
    for (int g = 0; g < 4; ++g) {
      ls::LumaSearchGates gates;
      gates.flags = gate_sets[g];
      gates.enable_edge_filter = edge;
      char label[96];
      std::snprintf(label, sizeof(label), "%s edge=%d", gate_names[g], edge);
      const Stats stats = run_synthetic_matrix(gates);
      print_stats(label, stats);
      ok = stats.ok() && ok;
    }
  }

  RealFrameCase frame64;
  frame64.name = "64x64";
  RealFrameCase frame360;
  frame360.name = "640x360";
  bool have64 = false;
  bool have360 = false;
  {
    Y4mFrame y4m;
    if (load_y4m_y(options.y4m64, y4m) && y4m.w == 64 && y4m.h == 64) {
      frame64.source = make_source(y4m);
      frame64.recon = make_recon(y4m);
      have64 = true;
    } else {
      std::fprintf(stderr, "warning: cannot read 64x64 media: %s\n",
                   options.y4m64.c_str());
    }
  }
  {
    Y4mFrame y4m;
    if (load_y4m_y(options.y4m360, y4m) && y4m.w == 640 && y4m.h == 360) {
      frame360.source = make_source(y4m);
      frame360.recon = make_recon(y4m);
      have360 = true;
    } else {
      std::fprintf(stderr, "warning: cannot read 640x360 media: %s\n",
                   options.y4m360.c_str());
    }
  }
  if (!have64 || !have360) {
    if (!options.allow_missing_media) {
      std::printf(
          "\nFAIL: real-frame parity requires the test media; pass "
          "--allow-missing-media to skip it\n");
      return false;
    }
    std::printf("\n-- real frames --\n  SKIP (media missing, allowed)\n");
    return ok;
  }

  std::printf("\n-- real 64x64 frame --\n");
  const std::vector<Size> layouts64 = {
      {64, 64}, {32, 32}, {16, 64}, {32, 8}, {64, 32}};
  for (int edge_index = 0; edge_index < 2; ++edge_index) {
    const int edge = edge_settings[edge_index];
    for (int g = 0; g < 4; ++g) {
      ls::LumaSearchGates gates;
      gates.flags = gate_sets[g];
      gates.enable_edge_filter = edge;
      char label[96];
      std::snprintf(label, sizeof(label), "64x64 %s edge=%d", gate_names[g],
                    edge);
      Stats stats;
      for (const Size& size : layouts64) {
        stats.add(run_real_frame(frame64, {size}, gates));
      }
      print_stats(label, stats);
      ok = stats.ok() && ok;
    }
  }

  std::printf("\n-- real 640x360 frame (SB edges, wavefront levels) --\n");
  for (int edge_index = 0; edge_index < 2; ++edge_index) {
    const int edge = edge_settings[edge_index];
    for (int g = 0; g < 4; ++g) {
      ls::LumaSearchGates gates;
      gates.flags = gate_sets[g];
      gates.enable_edge_filter = edge;
      char label[96];
      std::snprintf(label, sizeof(label), "640x360 %s edge=%d", gate_names[g],
                    edge);
      Stats stats;
      stats.add(run_wavefront_levels(frame360, gates));
      stats.add(run_real_frame(frame360, {{16, 64}}, gates));
      stats.add(run_real_frame(frame360, {{32, 8}}, gates));
      print_stats(label, stats);
      ok = stats.ok() && ok;
    }
  }

  std::printf("\n-- chunk invariance --\n");
  {
    ls::LumaSearchGates gates;
    const auto blocks360 = make_layout_blocks(frame360, {{64, 64}});
    ok = chunk_invariance(blocks360, gates, "640x360 64x64") && ok;
    const auto rects = make_layout_blocks(frame360, {{16, 64}, {32, 8}});
    ok = chunk_invariance(rects, gates, "640x360 rects") && ok;
    const auto blocks64 = make_layout_blocks(frame64, {{64, 64}, {32, 8}});
    ok = chunk_invariance(blocks64, gates, "64x64 mixed") && ok;
  }
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_args(argc, argv, options)) {
    return 2;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s  cc=%d.%d  smem/block=%zu  driver=%d\n", prop.name,
              prop.major, prop.minor,
              static_cast<std::size_t>(prop.sharedMemPerBlock),
              prop.multiProcessorCount);
  std::printf("luma search: 61 candidates, CPU tie-break = strictly smaller\n");

  bool ok = true;
  std::printf("\n-- candidate gates --\n");
  ok = gates_test() && ok;

  std::printf("\n-- API validation / shortcuts --\n");
  ok = api_validation() && ok;

  const unsigned gate_sets[] = {ls::kGateNone, ls::kGateBaseOnly,
                                ls::kGateNoDirectional, ls::kGateDcOnly};
  const char* gate_names[] = {"full", "base-only", "no-directional", "dc-only"};
  const int edge_settings[] = {0, 1};

  if (!options.timing_only) {
    ok = run_parity_tests(options, gate_sets, gate_names, edge_settings) && ok;
  }

  if (options.timing) {
    std::printf("\n-- timing (search only: expand+predict+SSE+combine) --\n");
    ls::LumaSearchGates gates;
    {
      Plane recon = make_texture(128, 128, 1, 11);
      Plane source = make_texture(128, 128, 3, 12);
      auto blocks = make_grid(64, 64, 64, 64, recon, source);
      time_search("1 x 64x64 block", blocks, gates, 200);
    }
    {
      Plane recon = make_texture(640, 360, 1, 21);
      Plane source = make_texture(640, 360, 3, 22);
      auto blocks = make_grid(640, 360, 64, 64, recon, source);
      time_search("640x360 SB grid, one batch", blocks, gates, 100);
    }
    {
      Plane recon = make_texture(1920, 1088, 1, 31);
      Plane source = make_texture(1920, 1088, 3, 32);
      auto blocks = make_grid(1920, 1088, 64, 64, recon, source);
      time_search("1080p SB grid, one batch", blocks, gates,
                  options.quick ? 10 : 30);
    }
    {
      Plane recon = make_texture(640, 360, 1, 41);
      Plane source = make_texture(640, 360, 3, 42);
      auto blocks = make_grid(640, 360, 16, 64, recon, source);
      time_search("640x360 16x64 grid, one batch", blocks, gates, 50);
    }
    {
      Plane recon = make_texture(640, 360, 1, 51);
      Plane source = make_texture(640, 360, 3, 52);
      auto blocks = make_grid(640, 360, 32, 8, recon, source);
      time_search("640x360 32x8 grid, one batch", blocks, gates, 50);
    }
    time_wavefront("640x360 SB wavefront levels", 640, 360,
                   options.quick ? 5 : 30);
    time_wavefront("1080p SB wavefront levels", 1920, 1088,
                   options.quick ? 3 : 10);
  }

  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
