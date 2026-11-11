// GPU sample / geometry path for 10-bit 4:2:0 frames.
//
// The frame layout is the format component's coded geometry: MI-aligned luma
// with coded (padded) planes, `uint16_t` samples, LSB-aligned 10-bit values,
// one contiguous arena buffer.  CPU reference:
// `fmt::convert_420_8_to_picture()` for upload and `fmt::convert_to_420_8()`
// for download.  The upload kernel also fills the coded padding by replicating
// the last visible sample, so it is a single pass over each plane and matches
// `fmt::replicate_borders()`.
//
// These kernels are deliberately plane-generic: for a 4:2:0 source, luma and
// chroma both map destination (x, y) to source (x, y) with the plane's own
// width/height/stride.  Chroma geometry is already subsampled in the
// `Hbd420Frame`, so no scale factor appears in the kernel.
//
// Caller-owned device memory; no allocation, no synchronization.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "encoder/format/format.h"

namespace av2enc::cuda::hbd {

struct HbdPlane420 {
  std::uint32_t coded_width = 0;
  std::uint32_t coded_height = 0;
  std::uint32_t visible_width = 0;
  std::uint32_t visible_height = 0;
  std::uint32_t stride = 0;  // samples
  std::size_t offset = 0;    // samples from the frame base
};

struct Hbd420Frame {
  av2enc::fmt::Format format;
  std::uint32_t mi_cols = 0;
  std::uint32_t mi_rows = 0;
  std::uint32_t coded_width = 0;   // MiCols * 4
  std::uint32_t coded_height = 0;  // MiRows * 4
  int num_planes = 0;
  int sub_x = 0;
  int sub_y = 0;
  HbdPlane420 planes[3];
  std::size_t total_samples = 0;
  std::size_t frame_bytes = 0;
};

// Builds the 10-bit 4:2:0 frame layout from `fmt::make_frame_geometry()`.
// Accepts only `ChromaFormat::k420`; the depth may be 8/10/12 so the same
// plumbing can be tested, but the M1 deliverable is 10-bit.
bool hbd_make_420_frame(const av2enc::fmt::Format& format, Hbd420Frame* out,
                        std::string* error);
bool hbd_make_420_frame_10(std::uint32_t width, std::uint32_t height,
                           Hbd420Frame* out, std::string* error);

// Expands one 8-bit plane into a `uint16_t` destination and fills right/bottom
// padding by replication.  Exposed for tests; the frame upload calls it for
// each plane.
cudaError_t hbd_expand_plane_8_to_16(const std::uint8_t* src,
                                     int src_stride_bytes, int src_width,
                                     int src_height, std::uint16_t* dst,
                                     int dst_stride_samples, int dst_width,
                                     int dst_height, int shift,
                                     cudaStream_t stream = nullptr);

// 8-bit 4:2:0 source -> 10-bit coded/padded `uint16_t` frame.  `frame` must
// describe an arena buffer of at least `frame.total_samples` samples.
//
// IMPORTANT: `source.y/u/v.data` are DEVICE pointers; `stride` is in bytes.
// The caller copies the input planes to device memory first (or already has
// them there from a pinned/async upload).  There is no implicit host copy and
// this function allocates nothing.
cudaError_t hbd_upload_420_8(const av2enc::fmt::Source420_8& source,
                             const Hbd420Frame& frame, std::uint16_t* d_frame,
                             cudaStream_t stream = nullptr);

// 10-bit coded frame -> 8-bit 4:2:0 visible planes (round-to-nearest).  The
// three output views are the visible rectangles; only visible samples are
// written.
struct HbdPlane8View {
  std::uint8_t* data = nullptr;
  int stride_bytes = 0;
  int width = 0;
  int height = 0;
};

cudaError_t hbd_download_420_8(const Hbd420Frame& frame,
                               const std::uint16_t* d_frame,
                               const HbdPlane8View out[3],
                               cudaStream_t stream = nullptr);

// Fills coded padding for all planes in place (used after an in-place recon
// stage that only writes visible samples).
cudaError_t hbd_replicate_420_borders(const Hbd420Frame& frame,
                                      std::uint16_t* d_frame,
                                      cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda::hbd
