// AV2 intra block copy — CUDA kernels.  See ibc.h for the contract.
#include "cuda/codec/intra/ibc/ibc.h"

#include "codec/intra/ibc/core/ibc_core.h"

#include <algorithm>
#include <cstdint>

#include "cuda/arena.h"
#include "cuda/error.h"

namespace av2enc::cuda {
namespace {

using av2enc::ibc::BlockVector;
using av2enc::ibc::kMiSize;
using av2enc::ibc::plane_offset;

constexpr std::size_t kSearchScratchBytes = sizeof(unsigned long long);

std::uint64_t align_up_u64(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

// Rewinds the arena on every return path.  The search scratch is transient and
// must not leak into the caller's persistent region.
class ArenaMark {
 public:
  explicit ArenaMark(Arena& arena) : arena_(arena), mark_(arena.mark()) {}
  ~ArenaMark() { arena_.rewind(mark_); }
  ArenaMark(const ArenaMark&) = delete;
  ArenaMark& operator=(const ArenaMark&) = delete;

 private:
  Arena& arena_;
  std::uint64_t mark_;
};

__host__ __device__ int block_dim(int bsize, bool height) {
  // The subset of AV2 block sizes used by the gate and M1.  The M1 rect tree
  // reaches every 1:2 / 2:1 / 1:4 / 4:1 shape down to the 8-pixel floor, so
  // the full sub-64 set is required for product IBC leaves.
  switch (bsize) {
    case 0:  // 4x4
      return 4;
    case 1:  // 4x8
      return height ? 8 : 4;
    case 2:  // 8x4
      return height ? 4 : 8;
    case 3:  // 8x8
      return 8;
    case 4:  // 8x16
      return height ? 16 : 8;
    case 5:  // 16x8
      return height ? 8 : 16;
    case 6:  // 16x16
      return 16;
    case 7:  // 16x32
      return height ? 32 : 16;
    case 8:  // 32x16
      return height ? 16 : 32;
    case 9:  // 32x32
      return 32;
    case 10:  // 32x64
      return height ? 64 : 32;
    case 11:  // 64x32
      return height ? 32 : 64;
    case 12:  // 64x64
      return 64;
    case 19:  // 4x16
      return height ? 16 : 4;
    case 20:  // 16x4
      return height ? 4 : 16;
    case 21:  // 8x32
      return height ? 32 : 8;
    case 22:  // 32x8
      return height ? 8 : 32;
    case 23:  // 16x64
      return height ? 64 : 16;
    case 24:  // 64x16
      return height ? 16 : 64;
    case 25:  // 4x32
      return height ? 32 : 4;
    case 26:  // 32x4
      return height ? 4 : 32;
    case 27:  // 8x64
      return height ? 64 : 8;
    case 28:  // 64x8
      return height ? 8 : 64;
    default:
      return 0;
  }
}

__global__ void copy_kernel(IbcDeviceFrame frame, int mi_row, int mi_col,
                            BlockVector bv, std::uint8_t* dst_y,
                            int dst_stride_y, std::uint8_t* dst_u,
                            int dst_stride_u, std::uint8_t* dst_v,
                            int dst_stride_v, int bw, int bh, int visible_w,
                            int visible_h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= visible_w || y >= visible_h) return;
  const int dx = plane_offset(bv.col, 0);
  const int dy = plane_offset(bv.row, 0);
  const int sx = mi_col * kMiSize + dx;
  const int sy = mi_row * kMiSize + dy;
  dst_y[static_cast<std::size_t>(y) *
            static_cast<std::size_t>(dst_stride_y) +
        static_cast<std::size_t>(x)] =
      frame.y[static_cast<std::size_t>(sy + y) *
                  static_cast<std::size_t>(frame.stride_y) +
              static_cast<std::size_t>(sx + x)];
  (void)bw;
  (void)bh;
  if (!frame.has_chroma || dst_u == nullptr || dst_v == nullptr) return;
  const int cw = (visible_w + (1 << frame.sub_x) - 1) >> frame.sub_x;
  const int ch = (visible_h + (1 << frame.sub_y) - 1) >> frame.sub_y;
  const int cx = (mi_col * kMiSize >> frame.sub_x) +
                 plane_offset(bv.col, frame.sub_x);
  const int cy = (mi_row * kMiSize >> frame.sub_y) +
                 plane_offset(bv.row, frame.sub_y);
  if (x >= cw || y >= ch) return;
  const std::uint8_t* su =
      frame.u + static_cast<std::size_t>(cy + y) *
                    static_cast<std::size_t>(frame.stride_u) +
      static_cast<std::size_t>(cx + x);
  const std::uint8_t* sv =
      frame.v + static_cast<std::size_t>(cy + y) *
                    static_cast<std::size_t>(frame.stride_v) +
      static_cast<std::size_t>(cx + x);
  dst_u[static_cast<std::size_t>(y) *
            static_cast<std::size_t>(dst_stride_u) +
        static_cast<std::size_t>(x)] = *su;
  dst_v[static_cast<std::size_t>(y) *
            static_cast<std::size_t>(dst_stride_v) +
        static_cast<std::size_t>(x)] = *sv;
}

__global__ void sad_kernel(const std::uint8_t* recon, int stride, int x, int y,
                           int bw, int bh, const BlockVector* bvs,
                           std::int64_t* sad) {
  const BlockVector bv = bvs[blockIdx.x];
  const int dx = plane_offset(bv.col, 0);
  const int dy = plane_offset(bv.row, 0);
  const int sx = x + dx;
  const int sy = y + dy;
  std::int64_t local = 0;
  for (int i = threadIdx.y; i < bh; i += blockDim.y) {
    for (int j = threadIdx.x; j < bw; j += blockDim.x) {
      const int b = recon[static_cast<std::size_t>(sy + i) *
                              static_cast<std::size_t>(stride) +
                          static_cast<std::size_t>(sx + j)];
      const int d =
          recon[static_cast<std::size_t>(y + i) *
                    static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x + j)] -
          b;
      local += d < 0 ? -d : d;
    }
  }
  __shared__ std::int64_t partial[16][16];
  partial[threadIdx.y][threadIdx.x] = local;
  __syncthreads();
  for (int step = 8; step > 0; step >>= 1) {
    if (threadIdx.x < static_cast<unsigned>(step) &&
        threadIdx.y < static_cast<unsigned>(step)) {
      partial[threadIdx.y][threadIdx.x] +=
          partial[threadIdx.y + step][threadIdx.x] +
          partial[threadIdx.y][threadIdx.x + step] +
          partial[threadIdx.y + step][threadIdx.x + step];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) sad[blockIdx.x] = partial[0][0];
}

__device__ bool geometry_ok(const IbcDeviceGeometry& g, int mi_row, int mi_col,
                            int bw, int bh, BlockVector bv) {
  const int total_sb64 = ((g.mi_cols - 1) >> 4) + 1;
  int reason = 0;
  return av2enc::ibc::bv_geometry_valid(
      mi_row, mi_col, bw, bh, 0, 0, g.mi_rows, g.mi_cols, bv.row, bv.col,
      /*has_chroma=*/true, mi_row, mi_col, /*allow_local=*/false,
      g.allow_global, g.allow_extra_ibc_range, /*local_ok=*/true, g.sb_size,
      total_sb64, g.mi_rows, g.mi_cols, &reason);
}

__global__ void validate_kernel(IbcDeviceGeometry geometry, int mi_row,
                                int mi_col, int bw, int bh,
                                const BlockVector* bvs, int n,
                                std::uint8_t* valid) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  valid[i] = geometry_ok(geometry, mi_row, mi_col, bw, bh, bvs[i]) ? 1 : 0;
}

__global__ void search_kernel(const std::uint8_t* recon, int stride, int width,
                              int height, IbcDeviceGeometry geometry,
                              int mi_row, int mi_col, int bw, int bh, int step,
                              int x, int y,
                              unsigned long long* best_packed) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;
  const int cols = (width - bw) / step + 1;
  const int rows = (height - bh) / step + 1;
  unsigned long long local = ~0ULL;
  for (int index = tid; index < cols * rows; index += nthreads) {
    const int sx = (index % cols) * step;
    const int sy = (index / cols) * step;
    const BlockVector bv{(sy - y) * 8, (sx - x) * 8};
    if (!geometry_ok(geometry, mi_row, mi_col, bw, bh, bv)) continue;
    if (sx < 0 || sy < 0 || sx + bw > width || sy + bh > height) continue;
    std::int64_t sad = 0;
    for (int i = 0; i < bh; ++i) {
      for (int j = 0; j < bw; ++j) {
        const int d =
            recon[static_cast<std::size_t>(y + i) *
                      static_cast<std::size_t>(stride) +
                  static_cast<std::size_t>(x + j)] -
            recon[static_cast<std::size_t>(sy + i) *
                      static_cast<std::size_t>(stride) +
                  static_cast<std::size_t>(sx + j)];
        sad += d < 0 ? -d : d;
      }
    }
    const unsigned long long packed =
        (static_cast<unsigned long long>(sad) << 20) |
        static_cast<unsigned long long>(index & 0xFFFFF);
    if (packed < local) local = packed;
  }
  atomicMin(best_packed, local);
}

}  // namespace

std::uint64_t ibc_required_device_bytes(const IbcSearchConfig& config) {
  (void)config;  // The search keeps one packed word regardless of geometry.
  constexpr std::uint64_t kAlignment = Arena::kAlignment;
  // Arena::init() aligns the usable base inside the reservation, so budget one
  // alignment unit of base padding in addition to the aligned scratch.
  return align_up_u64(kSearchScratchBytes, kAlignment) + kAlignment - 1u;
}

cudaError_t ibc_copy_block(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                           int bsize, BlockVector bv, std::uint8_t* dst_y,
                           int dst_stride_y, std::uint8_t* dst_u,
                           int dst_stride_u, std::uint8_t* dst_v,
                           int dst_stride_v, int visible_w, int visible_h,
                           cudaStream_t stream) {
  const int bw = block_dim(bsize, false);
  const int bh = block_dim(bsize, true);
  if (frame.y == nullptr || dst_y == nullptr || bw <= 0 || bh <= 0 ||
      visible_w <= 0 || visible_h <= 0 || visible_w > bw || visible_h > bh ||
      mi_row < 0 || mi_col < 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(16, 16, 1);
  const dim3 grid(static_cast<unsigned>((visible_w + 15) / 16),
                  static_cast<unsigned>((visible_h + 15) / 16), 1);
  copy_kernel<<<grid, block, 0, stream>>>(frame, mi_row, mi_col, bv, dst_y,
                                          dst_stride_y, dst_u, dst_stride_u,
                                          dst_v, dst_stride_v, bw, bh, visible_w,
                                          visible_h);
  return cudaGetLastError();
}

cudaError_t ibc_sad_batch(const IbcDeviceFrame& frame, int mi_row, int mi_col,
                          int bsize, const BlockVector* bvs, int n,
                          std::int64_t* sads, cudaStream_t stream) {
  const int bw = block_dim(bsize, false);
  const int bh = block_dim(bsize, true);
  if (frame.y == nullptr || bvs == nullptr || sads == nullptr || n <= 0 ||
      bw <= 0 || bh <= 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(16, 16, 1);
  sad_kernel<<<n, block, 0, stream>>>(frame.y, frame.stride_y,
                                      mi_col * kMiSize, mi_row * kMiSize, bw,
                                      bh, bvs, sads);
  return cudaGetLastError();
}

cudaError_t ibc_validate_global(const IbcDeviceGeometry& geometry, int mi_row,
                                int mi_col, int bsize, const BlockVector* bvs,
                                int n, std::uint8_t* valid,
                                cudaStream_t stream) {
  const int bw = block_dim(bsize, false);
  const int bh = block_dim(bsize, true);
  if (bvs == nullptr || valid == nullptr || n <= 0 || bw <= 0 || bh <= 0 ||
      !geometry.allow_global) {
    return cudaErrorInvalidValue;
  }
  const int threads = 128;
  const int blocks = (n + threads - 1) / threads;
  validate_kernel<<<blocks, threads, 0, stream>>>(geometry, mi_row, mi_col, bw,
                                                  bh, bvs, n, valid);
  return cudaGetLastError();
}

IbcSearchStatus ibc_search(Arena& arena, const IbcDeviceFrame& frame,
                           const IbcDeviceGeometry& geometry, int mi_row,
                           int mi_col, int bsize, int step, BlockVector* best,
                           std::int64_t* best_sad, cudaStream_t stream,
                           std::uint64_t* required_bytes) {
  if (required_bytes != nullptr) *required_bytes = 0;
  const int bw = block_dim(bsize, false);
  const int bh = block_dim(bsize, true);
  if (frame.y == nullptr || best == nullptr || best_sad == nullptr || bw <= 0 ||
      bh <= 0 || step <= 0 || frame.width < bw || frame.height < bh) {
    return IbcSearchStatus::kInvalidArgument;
  }

  const IbcSearchConfig config{frame.width, frame.height, bsize, step};
  const std::uint64_t need = ibc_required_device_bytes(config);
  if (required_bytes != nullptr) *required_bytes = need;

  ArenaMark mark(arena);
  void* raw = nullptr;
  const Status alloc =
      arena.allocate_checked(kSearchScratchBytes, &raw, Arena::kAlignment);
  if (alloc == Status::kArenaExhausted) {
    return IbcSearchStatus::kArenaTooSmall;
  }
  if (!is_ok(alloc)) {
    return IbcSearchStatus::kInvalidArgument;
  }
  unsigned long long* d_best = static_cast<unsigned long long*>(raw);
  const unsigned long long init = ~0ULL;
  cudaMemcpyAsync(d_best, &init, sizeof(init), cudaMemcpyHostToDevice, stream);
  const int threads = 256;
  const int candidates =
      ((frame.width - bw) / step + 1) * ((frame.height - bh) / step + 1);
  const int blocks = std::min(1024, (candidates + threads - 1) / threads);
  search_kernel<<<blocks, threads, 0, stream>>>(
      frame.y, frame.stride_y, frame.width, frame.height, geometry, mi_row,
      mi_col, bw, bh, step, mi_col * kMiSize, mi_row * kMiSize, d_best);
  const cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    return IbcSearchStatus::kCudaError;
  }
  unsigned long long packed = ~0ULL;
  const cudaError_t copy =
      cudaMemcpyAsync(&packed, d_best, sizeof(packed), cudaMemcpyDeviceToHost,
                      stream);
  if (copy != cudaSuccess) {
    return IbcSearchStatus::kCudaError;
  }
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) return IbcSearchStatus::kCudaError;
  if (packed == ~0ULL) {
    return IbcSearchStatus::kInvalidArgument;
  }
  const int index = static_cast<int>(packed & 0xFFFFF);
  const int cols = (frame.width - bw) / step + 1;
  const int sx = (index % cols) * step;
  const int sy = (index / cols) * step;
  best->row = (sy - mi_row * kMiSize) * 8;
  best->col = (sx - mi_col * kMiSize) * 8;
  *best_sad = static_cast<std::int64_t>(packed >> 20);
  return IbcSearchStatus::kOk;
}

}  // namespace av2enc::cuda
