// Device-facing frame/plane geometry for every AV2 chroma format and depth.
//
// This is the shared layout the 10-bit / 4:2:2 / 4:4:4 / 4:0:0 device chain
// uses for:
//
//   * sample conversion (source upload, padded reconstruction storage and
//     download) -- `FrameLayout` gives the coded and padded extents, strides
//     and sample offsets for one contiguous `uint16_t` frame;
//   * prediction / transform / reconstruction -- the same layout is what the
//     DC predictor, `hbd_tx_batch` and the reconstruction kernel index;
//   * deblocking / loop-filter geometry -- `FilterGeometry` describes the
//     coded plane extent, the MI grid and 4x4/8x8 edge grids for both sample
//     container widths;
//   * bitstream-facing buffers -- `BitstreamLayout` is the capacity contract
//     for per-block levels/EOB plus the reconstructed planes that the CPU
//     entropy writer consumes after a device pass.
//
// The normative geometry is §5.18.4.4 compute_image_size() and §7.21.2.  This
// header is host-only: kernels receive the resulting plain integers/strides.
// It deliberately has no CUDA API dependency so the same code can be compiled
// by host tests and by the encoder's host support layer.
#pragma once

#include <cstddef>
#include <cstdint>

#include "formats/format.h"

namespace av2enc::cuda::fmtdev {

// ---------------------------------------------------------------------------
// Sample / frame layout
// ---------------------------------------------------------------------------

// `coded_*` is the decoder's MiCols*4 extent (the plane that prediction and
// reconstruction address).  `storage_*` is the 64-aligned coded extent used by
// the high-bit-depth device buffers (so a partial 64x64 edge leaf has room);
// `padded_*` is the format layer's 64-aligned *visible* extent, used by the
// filter/plane helpers and by `fmt::make_frame_geometry()`.
struct PlaneLayout {
  bool present = false;
  int sub_x = 0;
  int sub_y = 0;
  int visible_w = 0;
  int visible_h = 0;
  int coded_w = 0;
  int coded_h = 0;
  int storage_w = 0;
  int storage_h = 0;
  int padded_w = 0;
  int padded_h = 0;
  int stride = 0;  // samples per row in the storage plane

  std::size_t coded_offset = 0;    // samples from the frame base
  std::size_t storage_offset = 0;  // samples from the storage frame base
  std::size_t coded_samples = 0;
  std::size_t storage_samples = 0;

  // Container width used by filters and bitstream-facing dumps.  AV2 signals
  // 8 and 10 bit; 12 is accepted internally for tests.
  int sample_bytes = 2;
};

struct FrameLayout {
  av2enc::fmt::Format format;
  int bit_depth = 10;
  int mi_cols = 0;
  int mi_rows = 0;
  int coded_w = 0;    // luma MiCols*4
  int coded_h = 0;    // luma MiRows*4
  int storage_w = 0;  // 64-aligned coded luma
  int storage_h = 0;
  int padded_w = 0;   // 64-aligned visible luma
  int padded_h = 0;
  int num_planes = 0;
  int sub_x = 0;
  int sub_y = 0;
  PlaneLayout planes[3];

  std::size_t coded_samples = 0;    // source planes, tightly packed
  std::size_t storage_samples = 0;  // reconstructed storage planes
  std::size_t coded_bytes = 0;      // uint16_t container
  std::size_t storage_bytes = 0;

  bool valid() const { return num_planes > 0 && coded_w > 0 && coded_h > 0; }

  // Device bytes for the source and reconstruction frames.  Always `uint16_t`
  // per sample so 8/10/12-bit share one buffer type.
  std::uint64_t source_device_bytes() const {
    return static_cast<std::uint64_t>(coded_samples) * sizeof(std::uint16_t);
  }
  std::uint64_t recon_device_bytes() const {
    return static_cast<std::uint64_t>(storage_samples) * sizeof(std::uint16_t);
  }
};

// Builds the layout from the canonical format geometry.  Returns a layout with
// `valid() == false` for an invalid/unsupported format.
FrameLayout make_frame_layout(const av2enc::fmt::Format& format);

// Convenience overload for the CUDA components that use their own enum.
FrameLayout make_frame_layout(std::uint32_t width, std::uint32_t height,
                              av2enc::fmt::ChromaFormat chroma,
                              av2enc::fmt::BitDepth depth);

// ---------------------------------------------------------------------------
// Filter / deblock plane geometry
// ---------------------------------------------------------------------------

struct FilterPlaneLayout {
  bool present = false;
  int sub_x = 0;
  int sub_y = 0;
  int visible_w = 0;
  int visible_h = 0;
  int coded_w = 0;   // planeWidth for §7.17
  int coded_h = 0;   // planeHeight
  int stride = 0;    // coded_w, samples
  int sample_bytes = 1;
  int coded_mi_cols = 0;  // coded_w / 4
  int coded_mi_rows = 0;
  int luma_mi_cols = 0;   // parent luma grid
  int luma_mi_rows = 0;
  int edge4_cols = 0;     // interior vertical 4x4 edges
  int edge4_rows = 0;     // interior horizontal 4x4 edges
  int edge8_cols = 0;     // 8x8 grid (for the wide-filter classes)
  int edge8_rows = 0;
};

struct FilterGeometry {
  av2enc::fmt::Format format;
  int bit_depth = 10;
  int luma_mi_cols = 0;
  int luma_mi_rows = 0;
  int num_planes = 0;
  int sub_x = 0;
  int sub_y = 0;
  FilterPlaneLayout planes[3];

  bool valid() const { return num_planes > 0 && luma_mi_cols > 0; }
};

// Compiled from the same §5.18.4.4/§7.21.2 geometry as FrameLayout.  Deblock
// operates on the coded (MI-aligned) plane; `visible_*` is provided so callers
// do not filter synthetic edge padding.
FilterGeometry make_filter_geometry(const av2enc::fmt::Format& format);
FilterGeometry make_filter_geometry(const FrameLayout& layout);

// §7.17 plane extents from the MI grid (0 for an absent plane).
int filter_plane_width(const FilterGeometry& geometry, int plane);
int filter_plane_height(const FilterGeometry& geometry, int plane);

// ---------------------------------------------------------------------------
// Block rectangles (edge / partial-block geometry)
// ---------------------------------------------------------------------------

struct PlaneBlockRect {
  int x = 0;  // plane coordinates
  int y = 0;
  int width = 0;   // nominal residual size
  int height = 0;
  int visible_width = 0;   // clipped to the visible plane
  int visible_height = 0;
  bool inside = false;
  bool partial = false;
};

// Mirror of `fmt::block_rect()`: luma coordinates are MI-aligned, plane 0 uses
// the luma block size and planes 1/2 use §5.20.7.26 Subsampled_Size.  Returns
// false for monochrome chroma planes or BLOCK_INVALID residual shapes.
bool plane_block_rect(const FrameLayout& layout, int plane, int luma_x,
                      int luma_y, int luma_w, int luma_h,
                      PlaneBlockRect* out);

// ---------------------------------------------------------------------------
// Bitstream-facing buffer capacities
// ---------------------------------------------------------------------------

// Minimal host-side descriptor; the full `HighBitDepthBlock` adds the TX sizes
// but has the same leading fields.  Keeping this POD here avoids making the
// geometry layer depend on the still-pipeline headers.
struct BitstreamBlockDesc {
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t luma_size = 0;
  std::uint32_t eob[3] = {0, 0, 0};
  std::uint32_t levels_offset = 0;
};

// The GPU block stage writes raster-order levels and an EOB per plane.  The
// CPU entropy writer consumes scan-ordered levels grouped by block.  This
// struct is the capacity contract between the two halves.
struct BitstreamLayout {
  int max_blocks = 1;
  int max_tx_elems = 1024;  // adjusted 32x32 maximum
  std::size_t level_capacity = 0;  // samples, max_blocks * max_tx_elems
  std::size_t eob_capacity = 0;    // entries, max_blocks * 3
  std::size_t block_capacity = 0;  // descriptors, max_blocks

  std::uint64_t level_bytes() const {
    return static_cast<std::uint64_t>(level_capacity) * sizeof(std::int32_t);
  }
  std::uint64_t eob_bytes() const {
    return static_cast<std::uint64_t>(eob_capacity) * sizeof(std::int32_t);
  }
  std::uint64_t block_bytes() const {
    return static_cast<std::uint64_t>(block_capacity) *
           sizeof(BitstreamBlockDesc);
  }
};

BitstreamLayout make_bitstream_layout(int max_blocks);

}  // namespace av2enc::cuda::fmtdev
