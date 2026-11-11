// M1 GPU path — CDEF CPU-vs-GPU parity on real reconstruction.
//
// Input is the plane dump written by the stream gate:
//
//   ./run_stream_gate <src.y4m> <out> --dump-planes recon.bin
//
//   "CDEFPL01", u32 width, u32 height,
//   recon Y, U, V, source Y, U, V            (tightly packed, visible size)
//
// The test runs the stage twice on the same data:
//   CPU: cdef::cdef_search_plan() + cdef_frame()      (reference)
//   GPU: m1_cdef_search_sse()   + m1_cdef_filter()
// and requires, bit for bit:
//   * every candidate's per-64x64 SSE to match the host table;
//   * the selected header strengths and cdef_idx grid to match;
//   * every filtered byte on all three planes to match.
//
// It then reports the quality delta against the source and the CUDA timings.
// Build/run: ./build_and_test.sh --cuda.
//
// Note: cudaMalloc is fine here -- this is a test harness, not a hot path.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "cuda/m1/cdef/m1_cdef_cdf.h"
#include "cuda/m1/cdef/m1_cdef_cuda.h"
#include "cuda/m1/cdef/m1_cdef_index_syntax.h"
#include "encoder/filters/cdef/cdef.h"
#include "encoder/filters/cdef/cdef_search.h"
#include "encoder/m1/filters_stream/filters_stream.h"

namespace {

using av2enc::cdef::CdefOptions;
using av2enc::cdef::CdefStrength;
using av2enc::cuda::m1::CdefDeviceFrame;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
    }                                                                      \
  } while (0)

#define CUDA_TRY(expr, msg)                                                \
  do {                                                                     \
    const cudaError_t err__ = (expr);                                      \
    ++g_checks;                                                            \
    if (err__ != cudaSuccess) {                                            \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,    \
                   (msg), cudaGetErrorString(err__));                      \
    }                                                                      \
  } while (0)

struct Planes {
  int width = 0;
  int height = 0;
  int chroma_w = 0;
  int chroma_h = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

bool read_dump(const std::string& path, Planes* recon, Planes* source) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  char magic[8];
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "CDEFPL01", 8) != 0 ||
      std::fread(&width, 4, 1, f) != 1 ||
      std::fread(&height, 4, 1, f) != 1) {
    std::fclose(f);
    return false;
  }
  recon->width = static_cast<int>(width);
  recon->height = static_cast<int>(height);
  recon->chroma_w = (recon->width + 1) / 2;
  recon->chroma_h = (recon->height + 1) / 2;
  *source = *recon;
  const std::size_t y_size = static_cast<std::size_t>(recon->width) *
                             static_cast<std::size_t>(recon->height);
  const std::size_t c_size = static_cast<std::size_t>(recon->chroma_w) *
                             static_cast<std::size_t>(recon->chroma_h);
  recon->y.resize(y_size);
  recon->u.resize(c_size);
  recon->v.resize(c_size);
  source->y.resize(y_size);
  source->u.resize(c_size);
  source->v.resize(c_size);
  bool ok = std::fread(recon->y.data(), 1, y_size, f) == y_size &&
            std::fread(recon->u.data(), 1, c_size, f) == c_size &&
            std::fread(recon->v.data(), 1, c_size, f) == c_size &&
            std::fread(source->y.data(), 1, y_size, f) == y_size &&
            std::fread(source->u.data(), 1, c_size, f) == c_size &&
            std::fread(source->v.data(), 1, c_size, f) == c_size;
  std::fclose(f);
  return ok;
}

// MI-aligned host planes (stride = mi_cols*4 >> sub).
struct HostPlanes {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  int stride[3] = {0, 0, 0};
  int height_p[3] = {0, 0, 0};
  std::vector<std::uint8_t> src[3];
  std::vector<std::uint8_t> dst[3];

  void init(int w, int h) {
    width = w;
    height = h;
    mi_cols = 2 * ((w + 7) >> 3);
    mi_rows = 2 * ((h + 7) >> 3);
    for (int plane = 0; plane < 3; ++plane) {
      const int sub = (plane == 0) ? 0 : 1;
      stride[plane] = (mi_cols * 4) >> sub;
      height_p[plane] = (mi_rows * 4) >> sub;
      src[plane].assign(static_cast<std::size_t>(stride[plane]) *
                            static_cast<std::size_t>(height_p[plane]),
                        0);
      dst[plane].assign(src[plane].size(), 0);
    }
  }
  void copy_from(const Planes& p) {
    const std::vector<std::uint8_t>* in[3] = {&p.y, &p.u, &p.v};
    for (int plane = 0; plane < 3; ++plane) {
      const int sub = (plane == 0) ? 0 : 1;
      const int w = (p.width + (1 << sub) - 1) >> sub;
      const int h = (p.height + (1 << sub) - 1) >> sub;
      for (int y = 0; y < h && y < height_p[plane]; ++y) {
        std::memcpy(src[plane].data() +
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(stride[plane]),
                    in[plane]->data() + static_cast<std::size_t>(y) *
                                            static_cast<std::size_t>(w),
                    static_cast<std::size_t>(w));
      }
    }
  }
};

struct DevicePlanes {
  const std::uint8_t* src[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* dst[3] = {nullptr, nullptr, nullptr};
  const std::uint8_t* ref[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* skip = nullptr;
  std::uint8_t* lossless = nullptr;
  std::uint8_t* lossless_chroma = nullptr;
  std::int8_t* idx = nullptr;
  CdefStrength* candidates = nullptr;
  std::int64_t* sse = nullptr;
  ~DevicePlanes() {
    for (int plane = 0; plane < 3; ++plane) {
      if (src[plane] != nullptr) cudaFree(const_cast<std::uint8_t*>(src[plane]));
      if (dst[plane] != nullptr) cudaFree(dst[plane]);
      if (ref[plane] != nullptr) cudaFree(const_cast<std::uint8_t*>(ref[plane]));
    }
    if (skip != nullptr) cudaFree(skip);
    if (lossless != nullptr) cudaFree(lossless);
    if (lossless_chroma != nullptr) cudaFree(lossless_chroma);
    if (idx != nullptr) cudaFree(idx);
    if (candidates != nullptr) cudaFree(candidates);
    if (sse != nullptr) cudaFree(sse);
  }
};

double psnr(const std::vector<std::uint8_t>& a,
            const std::vector<std::uint8_t>& b) {
  std::uint64_t se = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    se += static_cast<std::uint64_t>(d * d);
  }
  if (se == 0) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 /
                           (static_cast<double>(se) / static_cast<double>(a.size())));
}

// Visible crop of a padded plane.
std::vector<std::uint8_t> crop(const std::vector<std::uint8_t>& plane, int stride,
                               int w, int h) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    std::memcpy(out.data() + static_cast<std::size_t>(y) *
                                 static_cast<std::size_t>(w),
                plane.data() + static_cast<std::size_t>(y) *
                                   static_cast<std::size_t>(stride),
                static_cast<std::size_t>(w));
  }
  return out;
}


// §5.18.7.10 header parity: the stage's packed header must produce exactly the
// bits the verified M1 header writer produces for the same plan, and the
// reader must recover the same parameters.
void check_header_parity(const av2enc::cdef::CdefSearchResult& plan,
                         const char* label) {
  namespace fs = av2enc::m1::filters_stream;
  using av2enc::cuda::m1::CdefHeaderParams;
  CdefHeaderParams header;
  std::string error;
  CHECK(av2enc::cuda::m1::m1_cdef_build_header(plan.options,
                                               /*single_picture_header=*/true,
                                               &header, &error),
        "header builds");
  fs::CdefParams params;
  params.frame_enable = header.frame_enable;
  params.damping_minus_3 = header.damping_minus_3;
  params.strengths_minus_1 = header.strengths_minus_1;
  params.num_strengths = header.num_strengths;
  params.on_skip_txfm_frame_enable = header.on_skip_txfm_frame_enable;
  for (int i = 0; i < header.num_strengths; ++i) {
    params.set[i].y_pri = header.set[i].y_pri_strength;
    params.set[i].y_sec = header.set[i].y_sec_value;
    params.set[i].uv_pri = header.set[i].uv_pri_strength;
    params.set[i].uv_sec = header.set[i].uv_sec_value;
  }
  fs::CdefHeaderContext context;
  context.coded_lossless = false;
  context.enable_cdef = true;
  context.single_picture_header_flag = true;
  context.cdef_on_skip_txfm = 2;
  context.num_planes = 3;
  av2enc::BitWriter writer;
  const std::size_t bits =
      fs::write_cdef_params(writer, params, context, &error);
  CHECK(error.empty(), "filters_stream cdef_params writes");
  CHECK(static_cast<int>(bits) == header.bits,
        "header bit count matches filters_stream");
  const std::vector<std::uint8_t> bytes = writer.take_bytes();
  fs::BitReader reader(bytes, bits);
  fs::CdefParams decoded;
  CHECK(fs::read_cdef_params(reader, context, &decoded, &error),
        "cdef_params reads back");
  bool equal = decoded.frame_enable == params.frame_enable &&
               decoded.damping_minus_3 == params.damping_minus_3 &&
               decoded.strengths_minus_1 == params.strengths_minus_1 &&
               decoded.on_skip_txfm_frame_enable ==
                   params.on_skip_txfm_frame_enable;
  for (int i = 0; equal && i < params.num_strengths; ++i) {
    equal = decoded.set[i].y_pri == params.set[i].y_pri &&
            decoded.set[i].y_sec == params.set[i].y_sec &&
            decoded.set[i].uv_pri == params.set[i].uv_pri &&
            decoded.set[i].uv_sec == params.set[i].uv_sec;
  }
  CHECK(equal, "cdef_params round-trips");
  std::printf("%s header: %d strengths, %d bits\n", label,
              header.num_strengths, header.bits);
}

// §5.20.10.1 parity for the selected grid: encode it with the normative coders
// and require the decoded grid to equal plan.cdef_idx.  The same calls are the
// M1/M2 syntax hook.
void check_index_grid_parity(const av2enc::cdef::CdefSearchResult& plan,
                             int mi_cols, int mi_rows, const char* label) {
  using av2enc::cuda::m1::CdefIndexCdfs;
  using av2enc::cuda::m1::CdefIndexGrid;
  if (plan.num_strengths == 1) {
    for (std::int8_t idx : plan.cdef_idx) {
      CHECK(idx == 0, "single strength forces index 0");
    }
    std::printf("%s index: no symbols (CdefStrengths == 1)\n", label);
    return;
  }
  std::vector<std::uint8_t> buffer(1U << 16, 0);
  std::uint16_t index0[4][3];
  std::memcpy(index0, av2enc::cuda::m1::kCdefIndex0Cdf, sizeof(index0));
  std::uint16_t minus1[8] = {0};
  const std::uint16_t* source =
      av2enc::cuda::m1::cdef_index_minus1_default(plan.num_strengths);
  if (source != nullptr) {
    std::memcpy(minus1, source,
                sizeof(std::uint16_t) *
                    static_cast<std::size_t>(plan.num_strengths));
  }
  CdefIndexCdfs cdfs;
  cdfs.index0 = &index0[0][0];
  cdfs.minus1 = minus1;
  av2enc::av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
            av2enc::av2::Status::kOk,
        "index encoder init");
  CdefIndexGrid grid;
  grid.reset(mi_cols, mi_rows);
  bool ok = true;
  for (int br = 0; br < plan.n64_rows && ok; ++br) {
    for (int bc = 0; bc < plan.n64_cols && ok; ++bc) {
      const int idx = static_cast<int>(
          plan.cdef_idx[static_cast<std::size_t>(br) *
                           static_cast<std::size_t>(plan.n64_cols) +
                       static_cast<std::size_t>(bc)]);
      ok = av2enc::cuda::m1::cdef_encode_block_index(
          &grid, plan.num_strengths, 4, 0, 0, br * 16, bc * 16, idx, cdfs,
          [&encoder](std::uint16_t* cdf, int n, int symbol) {
            return encoder.encode_symbol(cdf, n, symbol) ==
                   av2enc::av2::Status::kOk;
          });
    }
  }
  CHECK(ok, "selected index grid encodes");
  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk,
        "index encoder exit");
  std::memcpy(index0, av2enc::cuda::m1::kCdefIndex0Cdf, sizeof(index0));
  if (source != nullptr) {
    std::memcpy(minus1, source,
                sizeof(std::uint16_t) *
                    static_cast<std::size_t>(plan.num_strengths));
  }
  av2enc::av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
            av2enc::av2::Status::kOk,
        "index decoder init");
  CdefIndexGrid decoded;
  decoded.reset(mi_cols, mi_rows);
  ok = true;
  for (int br = 0; br < plan.n64_rows && ok; ++br) {
    for (int bc = 0; bc < plan.n64_cols && ok; ++bc) {
      int idx = -1;
      ok = av2enc::cuda::m1::cdef_decode_block_index(
          &decoded, plan.num_strengths, 4, 0, 0, br * 16, bc * 16, cdfs,
          [&decoder](std::uint16_t* cdf, int n) -> int {
            int symbol = -1;
            if (decoder.decode_symbol(cdf, n, &symbol) !=
                av2enc::av2::Status::kOk) {
              return -1;
            }
            return symbol;
          },
          &idx);
    }
  }
  CHECK(ok, "selected index grid decodes");
  CHECK(decoder.exit_symbol() == av2enc::av2::Status::kOk,
        "index decoder sees clean trailing bits");
  CHECK(decoded.origins(plan.n64_cols, plan.n64_rows) == plan.cdef_idx,
        "decoded index grid equals the selected grid");
  std::printf("%s index: %zu bytes, %d strengths\n", label, payload_bytes,
              plan.num_strengths);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <planes.bin> [--damping D] [--strengths N]\n",
                 argv[0]);
    return 2;
  }
  int damping = 3;
  int num_strengths = 4;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--damping" && i + 1 < argc) {
      damping = std::atoi(argv[++i]);
    } else if (arg == "--strengths" && i + 1 < argc) {
      num_strengths = std::atoi(argv[++i]);
    }
  }

  Planes recon;
  Planes source;
  if (!read_dump(argv[1], &recon, &source)) {
    std::fprintf(stderr, "cannot read plane dump %s\n", argv[1]);
    return 1;
  }
  std::printf("plane dump: %dx%d\n", recon.width, recon.height);

  HostPlanes host;
  host.init(recon.width, recon.height);
  host.copy_from(recon);
  const int n64_cols = (host.mi_cols + 15) / 16;
  const int n64_rows = (host.mi_rows + 15) / 16;
  const int nblocks = n64_cols * n64_rows;

  av2enc::cdef::CdefSearchFrame view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.src[plane].data = host.src[plane].data();
    view.src[plane].stride = host.stride[plane];
    const int sub = (plane == 0) ? 0 : 1;
    view.ref[plane].data =
        plane == 0 ? source.y.data() : (plane == 1 ? source.u.data()
                                                   : source.v.data());
    view.ref[plane].stride = (source.width + (1 << sub) - 1) >> sub;
  }
  view.width = recon.width;
  view.height = recon.height;
  view.mi_cols = host.mi_cols;
  view.mi_rows = host.mi_rows;
  view.subsampling_x = 1;
  view.subsampling_y = 1;
  std::vector<std::uint8_t> skip(static_cast<std::size_t>(host.mi_cols) *
                                     static_cast<std::size_t>(host.mi_rows),
                                 0);
  view.skip = skip.data();
  view.lossless = skip.data();
  view.lossless_chroma = skip.data();

  av2enc::cdef::CdefSearchOptions options;
  options.damping = damping;
  options.num_strengths = num_strengths;
  options.include_off = true;

  std::vector<CdefStrength> pool;
  std::string error;
  CHECK(av2enc::cdef::search_detail::build_pool(options, &pool, &error),
        "candidate pool");
  const int candidate_count = static_cast<int>(pool.size());

  // ---- CPU reference -----------------------------------------------------
  av2enc::cdef::CdefSearchResult cpu_plan;
  CHECK(av2enc::cdef::cdef_search_plan(view, options, &cpu_plan, &error),
        "CPU search");
  av2enc::cdef::CdefFrame cpu_frame{};
  for (int plane = 0; plane < 3; ++plane) {
    cpu_frame.planes[plane].src = host.src[plane].data();
    cpu_frame.planes[plane].dst = host.dst[plane].data();
    cpu_frame.planes[plane].stride = host.stride[plane];
  }
  cpu_frame.mi_cols = host.mi_cols;
  cpu_frame.mi_rows = host.mi_rows;
  cpu_frame.subsampling_x = 1;
  cpu_frame.subsampling_y = 1;
  cpu_frame.skip = skip.data();
  cpu_frame.lossless = skip.data();
  cpu_frame.lossless_chroma = skip.data();
  cpu_frame.cdef_idx = cpu_plan.cdef_idx.data();
  cpu_frame.n64_cols = cpu_plan.n64_cols;
  cpu_frame.n64_rows = cpu_plan.n64_rows;
  CHECK(av2enc::cdef::cdef_frame(cpu_frame, cpu_plan.options) ==
            av2enc::cdef::CdefStatus::kOk,
        "CPU filter");

  // ---- Device upload -----------------------------------------------------
  DevicePlanes dev;
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t bytes = host.src[plane].size();
    std::uint8_t* d_src = nullptr;
    std::uint8_t* d_dst = nullptr;
    std::uint8_t* d_ref = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_src), bytes), "malloc src");
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_dst), bytes), "malloc dst");
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_ref), bytes), "malloc ref");
    CUDA_TRY(cudaMemcpy(d_src, host.src[plane].data(), bytes,
                        cudaMemcpyHostToDevice),
             "upload src");
    CUDA_TRY(cudaMemcpy(d_dst, host.dst[plane].data(), bytes,
                        cudaMemcpyHostToDevice),
             "upload dst");
    dev.src[plane] = d_src;
    dev.dst[plane] = d_dst;
    dev.ref[plane] = d_ref;
  }
  // Reference planes: the stage's `ref` is the visible, tightly packed source.
  int ref_stride[3] = {0, 0, 0};
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int w = (source.width + (1 << sub) - 1) >> sub;
    const int h = (source.height + (1 << sub) - 1) >> sub;
    ref_stride[plane] = w;
    const std::vector<std::uint8_t>* p =
        plane == 0 ? &source.y : (plane == 1 ? &source.u : &source.v);
    CUDA_TRY(cudaMemcpy(const_cast<std::uint8_t*>(dev.ref[plane]), p->data(),
                        static_cast<std::size_t>(w) *
                            static_cast<std::size_t>(h),
                        cudaMemcpyHostToDevice),
             "upload ref");
  }
  const std::size_t mi_bytes = static_cast<std::size_t>(host.mi_cols) *
                               static_cast<std::size_t>(host.mi_rows);
  std::vector<std::uint8_t> mi_zero(mi_bytes, 0);
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.skip), mi_bytes), "malloc skip");
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.lossless), mi_bytes),
           "malloc lossless");
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.lossless_chroma), mi_bytes),
           "malloc lossless_chroma");
  CUDA_TRY(cudaMemcpy(dev.skip, mi_zero.data(), mi_bytes,
                      cudaMemcpyHostToDevice),
           "upload skip");
  CUDA_TRY(cudaMemcpy(dev.lossless, mi_zero.data(), mi_bytes,
                      cudaMemcpyHostToDevice),
           "upload lossless");
  CUDA_TRY(cudaMemcpy(dev.lossless_chroma, mi_zero.data(), mi_bytes,
                      cudaMemcpyHostToDevice),
           "upload lossless_chroma");
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.idx),
                      static_cast<std::size_t>(nblocks)),
           "malloc idx");
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.candidates),
                      sizeof(CdefStrength) * pool.size()),
           "malloc candidates");
  CUDA_TRY(cudaMemcpy(dev.candidates, pool.data(),
                      sizeof(CdefStrength) * pool.size(),
                      cudaMemcpyHostToDevice),
           "upload candidates");
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&dev.sse),
                      sizeof(std::int64_t) * static_cast<std::size_t>(nblocks) *
                          static_cast<std::size_t>(candidate_count)),
           "malloc sse");

  CdefDeviceFrame device_frame{};
  for (int plane = 0; plane < 3; ++plane) {
    device_frame.src[plane] = dev.src[plane];
    device_frame.dst[plane] = dev.dst[plane];
    device_frame.stride[plane] = host.stride[plane];
  }
  device_frame.width = recon.width;
  device_frame.height = recon.height;
  device_frame.mi_cols = host.mi_cols;
  device_frame.mi_rows = host.mi_rows;
  device_frame.skip = dev.skip;
  device_frame.lossless = dev.lossless;
  device_frame.lossless_chroma = dev.lossless_chroma;
  device_frame.cdef_idx = dev.idx;
  device_frame.n64_cols = n64_cols;
  device_frame.n64_rows = n64_rows;

  // ---- GPU search --------------------------------------------------------
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_TRY(cudaEventCreate(&start), "event start");
  CUDA_TRY(cudaEventCreate(&stop), "event stop");
  CUDA_TRY(m1_cdef_search_sse(device_frame, dev.ref, ref_stride, dev.candidates,
                              candidate_count, damping, false, false, dev.sse),
           "GPU search");
  CUDA_TRY(cudaDeviceSynchronize(), "sync search");
  CUDA_TRY(cudaEventRecord(start), "record start");
  for (int i = 0; i < 20; ++i) {
    CUDA_TRY(m1_cdef_search_sse(device_frame, dev.ref, ref_stride,
                                dev.candidates, candidate_count, damping,
                                false, false, dev.sse),
             "GPU search (timed)");
  }
  CUDA_TRY(cudaEventRecord(stop), "record stop");
  CUDA_TRY(cudaEventSynchronize(stop), "sync stop");
  float search_ms = 0.0f;
  CUDA_TRY(cudaEventElapsedTime(&search_ms, start, stop), "search elapsed");
  search_ms /= 20.0f;

  std::vector<std::int64_t> gpu_sse(
      static_cast<std::size_t>(nblocks) * static_cast<std::size_t>(candidate_count));
  CUDA_TRY(cudaMemcpy(gpu_sse.data(), dev.sse,
                      sizeof(std::int64_t) * gpu_sse.size(),
                      cudaMemcpyDeviceToHost),
           "download sse");

  // Host table for the same pool.
  std::vector<std::int64_t> host_sse(gpu_sse.size(), 0);
  std::uint8_t* const no_dst[3] = {nullptr, nullptr, nullptr};
  for (int k = 0; k < candidate_count; ++k) {
    for (int r = 0; r < host.mi_rows; r += 2) {
      for (int c = 0; c < host.mi_cols; c += 2) {
        const int block = (r / 16) * n64_cols + (c / 16);
        const av2enc::cdef::CdefBlockSse one = av2enc::cdef::cdef_eval_block(
            view, pool[static_cast<std::size_t>(k)], damping, false, false, r,
            c, no_dst);
        host_sse[static_cast<std::size_t>(k) *
                     static_cast<std::size_t>(nblocks) +
                 static_cast<std::size_t>(block)] += one.luma + one.chroma;
      }
    }
  }
  int sse_mismatch = 0;
  for (std::size_t i = 0; i < gpu_sse.size(); ++i) {
    if (gpu_sse[i] != host_sse[i]) ++sse_mismatch;
  }
  std::printf("search table: %d candidate rows x %d blocks, %d mismatches\n",
              candidate_count, nblocks, sse_mismatch);
  CHECK(sse_mismatch == 0, "GPU search SSE matches the host search");

  // ---- Selection on the GPU table ---------------------------------------
  av2enc::cdef::CdefSearchResult gpu_plan;
  CHECK(av2enc::cdef::search_detail::select(
            pool, n64_cols, n64_rows, gpu_sse, cpu_plan.block_sse_off, options,
            &gpu_plan, &error),
        "select on the GPU table");
  CHECK(gpu_plan.num_strengths == cpu_plan.num_strengths,
        "GPU table selects the same number of strengths");
  CHECK(gpu_plan.cdef_idx == cpu_plan.cdef_idx,
        "GPU table selects the same index grid");
  bool strengths_equal = true;
  for (int i = 0; i < gpu_plan.num_strengths; ++i) {
    strengths_equal = strengths_equal &&
                      gpu_plan.strengths[i].y_pri ==
                          cpu_plan.strengths[i].y_pri &&
                      gpu_plan.strengths[i].y_sec ==
                          cpu_plan.strengths[i].y_sec &&
                      gpu_plan.strengths[i].uv_pri ==
                          cpu_plan.strengths[i].uv_pri &&
                      gpu_plan.strengths[i].uv_sec ==
                          cpu_plan.strengths[i].uv_sec;
  }
  CHECK(strengths_equal, "GPU table selects the same strength sets");

  // ---- Header and index-grid parity for the selected plan ----------------
  check_header_parity(cpu_plan, "cpu-plan");
  check_header_parity(gpu_plan, "gpu-plan");
  check_index_grid_parity(cpu_plan, host.mi_cols, host.mi_rows, "cpu-plan");
  check_index_grid_parity(gpu_plan, host.mi_cols, host.mi_rows, "gpu-plan");

  // ---- GPU filter with the CPU plan's grid ------------------------------
  CUDA_TRY(cudaMemcpy(dev.idx, cpu_plan.cdef_idx.data(),
                      static_cast<std::size_t>(nblocks), cudaMemcpyHostToDevice),
           "upload idx");
  CUDA_TRY(m1_cdef_filter(device_frame, cpu_plan.options), "GPU filter");
  CUDA_TRY(cudaDeviceSynchronize(), "sync filter");
  CUDA_TRY(cudaEventRecord(start), "record start");
  for (int i = 0; i < 20; ++i) {
    CUDA_TRY(m1_cdef_filter(device_frame, cpu_plan.options), "GPU filter (timed)");
  }
  CUDA_TRY(cudaEventRecord(stop), "record stop");
  CUDA_TRY(cudaEventSynchronize(stop), "sync stop");
  float filter_ms = 0.0f;
  CUDA_TRY(cudaEventElapsedTime(&filter_ms, start, stop), "filter elapsed");
  filter_ms /= 20.0f;

  std::vector<std::uint8_t> gpu_out[3];
  int plane_mismatch[3] = {0, 0, 0};
  for (int plane = 0; plane < 3; ++plane) {
    gpu_out[plane].resize(host.src[plane].size());
    CUDA_TRY(cudaMemcpy(gpu_out[plane].data(), dev.dst[plane],
                        gpu_out[plane].size(), cudaMemcpyDeviceToHost),
             "download dst");
    for (std::size_t i = 0; i < gpu_out[plane].size(); ++i) {
      if (gpu_out[plane][i] != host.dst[plane][i]) ++plane_mismatch[plane];
    }
  }
  std::printf("filter output: Y %d U %d V %d mismatches\n", plane_mismatch[0],
              plane_mismatch[1], plane_mismatch[2]);
  CHECK(plane_mismatch[0] == 0 && plane_mismatch[1] == 0 &&
            plane_mismatch[2] == 0,
        "GPU filter is bit-exact with the CPU reference");

  // ---- Copy path: cdef_idx == -1 must copy every sample ------------------
  {
    std::vector<std::int8_t> copy_grid(static_cast<std::size_t>(nblocks), -1);
    CUDA_TRY(cudaMemcpy(dev.idx, copy_grid.data(), copy_grid.size(),
                        cudaMemcpyHostToDevice),
             "upload copy grid");
    CUDA_TRY(m1_cdef_filter(device_frame, cpu_plan.options), "GPU copy filter");
    CUDA_TRY(cudaDeviceSynchronize(), "sync copy filter");
    int copy_mismatch[3] = {0, 0, 0};
    for (int plane = 0; plane < 3; ++plane) {
      std::vector<std::uint8_t> out(host.src[plane].size());
      CUDA_TRY(cudaMemcpy(out.data(), dev.dst[plane], out.size(),
                          cudaMemcpyDeviceToHost),
               "download copy dst");
      for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] != host.src[plane][i]) ++copy_mismatch[plane];
      }
    }
    std::printf("copy path: Y %d U %d V %d mismatches\n", copy_mismatch[0],
                copy_mismatch[1], copy_mismatch[2]);
    CHECK(copy_mismatch[0] == 0 && copy_mismatch[1] == 0 &&
              copy_mismatch[2] == 0,
          "cdef_idx == -1 copies every plane byte-exactly on the GPU");
  }

  // ---- Quality vs source -------------------------------------------------
  const int cw = (source.width + 1) / 2;
  const int ch = (source.height + 1) / 2;
  const std::vector<std::uint8_t> gpu_y =
      crop(gpu_out[0], host.stride[0], source.width, source.height);
  const std::vector<std::uint8_t> gpu_u = crop(gpu_out[1], host.stride[1], cw, ch);
  const std::vector<std::uint8_t> gpu_v = crop(gpu_out[2], host.stride[2], cw, ch);
  const double base_psnr = psnr(source.y, recon.y);
  const double cdef_psnr = psnr(source.y, gpu_y);
  std::printf(
      "PSNR-Y  recon %.4f -> cdef %.4f (%+.4f); U %.4f -> %.4f; V %.4f -> "
      "%.4f\n",
      base_psnr, cdef_psnr, cdef_psnr - base_psnr, psnr(source.u, recon.u),
      psnr(source.u, gpu_u), psnr(source.v, recon.v), psnr(source.v, gpu_v));
  std::printf("timing: search %.3f ms, filter %.3f ms (20 iterations)\n",
              search_ms, filter_ms);
  std::printf("plan: %d strengths, header bits (see m1_cdef), blocks %dx%d\n",
              cpu_plan.num_strengths, n64_cols, n64_rows);

  CUDA_TRY(cudaEventDestroy(start), "destroy start");
  CUDA_TRY(cudaEventDestroy(stop), "destroy stop");

  std::printf("m1_cdef_gpu_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
