// GPU sample / geometry kernels for 10-bit 4:2:0.  See hbd_frame.h.
#include "cuda/pipeline/still/hbd/hbd_frame.h"

#include <cstddef>
#include <cstdint>

// Deliberately no dependency on `src/encoder/format/format.cpp` or
// `src/cuda/format/high_bitdepth.cu`: the root CMake target globs this
// directory but does not link those host components yet.  The geometry below
// is the §5.18.4.4 formula; `hbd_test.cu` pins it against
// `fmt::make_frame_geometry()`.
namespace av2enc::cuda::hbd {
namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

__global__ void expand_plane_8_to_16_kernel(
    const std::uint8_t* __restrict__ src, int src_stride_bytes, int src_width,
    int src_height, std::uint16_t* __restrict__ dst, int dst_stride_samples,
    int dst_width, int dst_height, int shift) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= dst_width || y >= dst_height) {
    return;
  }
  // Coded padding replicates the nearest visible sample.  This is the same
  // rule as fmt::replicate_borders() and keeps edge prediction reads defined.
  const int sx = x < src_width ? x : src_width - 1;
  const int sy = y < src_height ? y : src_height - 1;
  const std::uint32_t value = src[static_cast<std::size_t>(sy) *
                                      static_cast<std::size_t>(src_stride_bytes) +
                                  static_cast<std::size_t>(sx)];
  dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride_samples) +
      static_cast<std::size_t>(x)] =
      static_cast<std::uint16_t>(value << static_cast<unsigned>(shift));
}

__global__ void download_plane_10_to_8_kernel(
    const std::uint16_t* __restrict__ src, int src_stride_samples,
    std::uint8_t* __restrict__ dst, int dst_stride_bytes, int width, int height,
    int bit_depth) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }
  const std::uint32_t value =
      src[static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride_samples) +
          static_cast<std::size_t>(x)];
  std::uint32_t rounded = value;
  if (bit_depth > 8) {
    const int shift = bit_depth - 8;
    const std::uint32_t bias = 1U << static_cast<unsigned>(shift - 1);
    rounded = (value + bias) >> static_cast<unsigned>(shift);
  }
  dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride_bytes) +
      static_cast<std::size_t>(x)] =
      static_cast<std::uint8_t>(rounded > 255U ? 255U : rounded);
}

__global__ void replicate_plane_16_kernel(std::uint16_t* plane, int stride,
                                          int coded_width, int coded_height,
                                          int visible_width,
                                          int visible_height) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= coded_width || y >= coded_height) {
    return;
  }
  if (x < visible_width && y < visible_height) {
    return;
  }
  const int source_x = x < visible_width ? x : visible_width - 1;
  const int source_y = y < visible_height ? y : visible_height - 1;
  plane[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
        static_cast<std::size_t>(x)] =
      plane[static_cast<std::size_t>(source_y) *
                static_cast<std::size_t>(stride) +
            static_cast<std::size_t>(source_x)];
}

dim3 grid_for(int width, int height) {
  const int blocks_x = (width + kBlockX - 1) / kBlockX;
  const int blocks_y = (height + kBlockY - 1) / kBlockY;
  return dim3(static_cast<unsigned>(blocks_x > 0 ? blocks_x : 1),
              static_cast<unsigned>(blocks_y > 0 ? blocks_y : 1));
}

int chroma_extent(int value) { return (value + 1) / 2; }

// Local depth helpers: keep this translation unit free of a link dependency on
// format.cpp (the root CMake M1 target does not list it yet).
int depth_value(av2enc::fmt::BitDepth depth) {
  switch (depth) {
    case av2enc::fmt::BitDepth::k8:
      return 8;
    case av2enc::fmt::BitDepth::k10:
      return 10;
    case av2enc::fmt::BitDepth::k12:
      return 12;
  }
  return 0;
}

cudaError_t replicate_plane_16(std::uint16_t* plane, int stride,
                               int coded_width, int coded_height,
                               int visible_width, int visible_height,
                               cudaStream_t stream) {
  if (coded_width <= 0 || coded_height <= 0) {
    return cudaSuccess;
  }
  if (visible_width <= 0 || visible_height <= 0) {
    return cudaSuccess;
  }
  if (stride < coded_width || visible_width > coded_width ||
      visible_height > coded_height) {
    return cudaErrorInvalidValue;
  }
  if (visible_width == coded_width && visible_height == coded_height) {
    return cudaSuccess;
  }
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid = grid_for(coded_width, coded_height);
  replicate_plane_16_kernel<<<grid, block, 0, stream>>>(
      plane, stride, coded_width, coded_height, visible_width, visible_height);
  return cudaGetLastError();
}

}  // namespace

bool hbd_make_420_frame(const av2enc::fmt::Format& format, Hbd420Frame* out,
                        std::string* error) {
  if (out == nullptr) {
    if (error != nullptr) *error = "null frame";
    return false;
  }
  if (format.chroma != av2enc::fmt::ChromaFormat::k420) {
    if (error != nullptr) *error = "hbd_make_420_frame requires 4:2:0";
    return false;
  }
  const int depth = depth_value(format.depth);
  if (depth < 8 || depth > 12 || format.width == 0 || format.height == 0 ||
      format.width > 65535U || format.height > 65535U) {
    if (error != nullptr) *error = "unsupported frame format";
    return false;
  }

  // §5.18.4.4 compute_image_size(): MiCols = 2 * ((width + 7) >> 3).
  Hbd420Frame frame;
  frame.format = format;
  frame.mi_cols = 2U * ((format.width + 7U) >> 3U);
  frame.mi_rows = 2U * ((format.height + 7U) >> 3U);
  frame.coded_width = frame.mi_cols * 4U;
  frame.coded_height = frame.mi_rows * 4U;
  frame.num_planes = 3;
  frame.sub_x = 1;
  frame.sub_y = 1;
  std::size_t offset = 0;
  for (int plane = 0; plane < 3; ++plane) {
    HbdPlane420& dst = frame.planes[plane];
    dst.coded_width = frame.coded_width >> (plane == 0 ? 0 : 1);
    dst.coded_height = frame.coded_height >> (plane == 0 ? 0 : 1);
    dst.visible_width =
        plane == 0 ? format.width : (format.width + 1U) >> 1U;
    dst.visible_height =
        plane == 0 ? format.height : (format.height + 1U) >> 1U;
    dst.stride = dst.coded_width;
    dst.offset = offset;
    offset += static_cast<std::size_t>(dst.coded_width) * dst.coded_height;
  }
  frame.total_samples = offset;
  frame.frame_bytes = offset * sizeof(std::uint16_t);
  *out = frame;
  if (error != nullptr) error->clear();
  return true;
}

bool hbd_make_420_frame_10(std::uint32_t width, std::uint32_t height,
                           Hbd420Frame* out, std::string* error) {
  const av2enc::fmt::Format format{width, height, av2enc::fmt::ChromaFormat::k420,
                                   av2enc::fmt::BitDepth::k10};
  return hbd_make_420_frame(format, out, error);
}

cudaError_t hbd_expand_plane_8_to_16(const std::uint8_t* src,
                                     int src_stride_bytes, int src_width,
                                     int src_height, std::uint16_t* dst,
                                     int dst_stride_samples, int dst_width,
                                     int dst_height, int shift,
                                     cudaStream_t stream) {
  if (dst_width <= 0 || dst_height <= 0) {
    return cudaSuccess;
  }
  if (src == nullptr || dst == nullptr || src_width <= 0 || src_height <= 0 ||
      shift < 0 || shift > 4 || src_stride_bytes < src_width ||
      dst_stride_samples < dst_width) {
    return cudaErrorInvalidValue;
  }
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid = grid_for(dst_width, dst_height);
  expand_plane_8_to_16_kernel<<<grid, block, 0, stream>>>(
      src, src_stride_bytes, src_width, src_height, dst, dst_stride_samples,
      dst_width, dst_height, shift);
  return cudaGetLastError();
}

cudaError_t hbd_upload_420_8(const av2enc::fmt::Source420_8& source,
                             const Hbd420Frame& frame, std::uint16_t* d_frame,
                             cudaStream_t stream) {
  if (d_frame == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int depth = depth_value(frame.format.depth);
  if (depth < 8 || depth > 12) {
    return cudaErrorInvalidValue;
  }
  if (source.width != frame.format.width || source.height != frame.format.height) {
    return cudaErrorInvalidValue;
  }
  const int shift = depth - 8;
  const int chroma_w = chroma_extent(static_cast<int>(source.width));
  const int chroma_h = chroma_extent(static_cast<int>(source.height));
  const av2enc::fmt::ConstPlane8 src_planes[3] = {source.y, source.u, source.v};
  const int src_w[3] = {static_cast<int>(source.width), chroma_w, chroma_w};
  const int src_h[3] = {static_cast<int>(source.height), chroma_h, chroma_h};

  for (int plane = 0; plane < 3; ++plane) {
    const HbdPlane420& dst = frame.planes[plane];
    if (src_planes[plane].data == nullptr ||
        src_planes[plane].stride < static_cast<std::uint32_t>(src_w[plane])) {
      return cudaErrorInvalidValue;
    }
    const cudaError_t err = hbd_expand_plane_8_to_16(
        src_planes[plane].data, static_cast<int>(src_planes[plane].stride),
        src_w[plane], src_h[plane], d_frame + dst.offset,
        static_cast<int>(dst.stride), static_cast<int>(dst.coded_width),
        static_cast<int>(dst.coded_height), shift, stream);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

cudaError_t hbd_download_420_8(const Hbd420Frame& frame,
                               const std::uint16_t* d_frame,
                               const HbdPlane8View out[3],
                               cudaStream_t stream) {
  if (d_frame == nullptr || out == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int depth = depth_value(frame.format.depth);
  if (depth < 8 || depth > 12) {
    return cudaErrorInvalidValue;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const HbdPlane420& src = frame.planes[plane];
    const HbdPlane8View& dst = out[plane];
    if (dst.data == nullptr || dst.width <= 0 || dst.height <= 0 ||
        dst.stride_bytes < dst.width) {
      return cudaErrorInvalidValue;
    }
    if (dst.width < static_cast<int>(src.visible_width) ||
        dst.height < static_cast<int>(src.visible_height)) {
      return cudaErrorInvalidValue;
    }
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid = grid_for(static_cast<int>(src.visible_width),
                               static_cast<int>(src.visible_height));
    download_plane_10_to_8_kernel<<<grid, block, 0, stream>>>(
        d_frame + src.offset, static_cast<int>(src.stride), dst.data,
        dst.stride_bytes, static_cast<int>(src.visible_width),
        static_cast<int>(src.visible_height), depth);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

cudaError_t hbd_replicate_420_borders(const Hbd420Frame& frame,
                                      std::uint16_t* d_frame,
                                      cudaStream_t stream) {
  if (d_frame == nullptr) {
    return cudaErrorInvalidValue;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const HbdPlane420& p = frame.planes[plane];
    const cudaError_t err = replicate_plane_16(
        d_frame + p.offset, static_cast<int>(p.stride),
        static_cast<int>(p.coded_width), static_cast<int>(p.coded_height),
        static_cast<int>(p.visible_width), static_cast<int>(p.visible_height),
        stream);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

}  // namespace av2enc::cuda::hbd
