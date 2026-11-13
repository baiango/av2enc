#include "codec/filters/deblock/deblock.h"

#include <cstddef>

namespace av2enc::deblock {
namespace {

int plane_horizontal_shift(const DeblockFrame& frame, int plane) {
  return (plane == 0) ? 0 : frame.subsampling_x;
}

int plane_vertical_shift(const DeblockFrame& frame, int plane) {
  return (plane == 0) ? 0 : frame.subsampling_y;
}

}  // namespace

const char* deblock_status_string(DeblockStatus status) {
  switch (status) {
    case DeblockStatus::kOk:
      return "ok";
    case DeblockStatus::kInvalidArgument:
      return "invalid argument";
    case DeblockStatus::kUnsupportedBitDepth:
      return "unsupported bit depth (only 8-bit is implemented)";
    case DeblockStatus::kUnsupportedSubsampling:
      return "unsupported chroma subsampling (only 4:2:0 is implemented)";
  }
  return "unknown";
}

int deblock_plane_width(const DeblockFrame& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_cols <= 0) {
    return 0;
  }
  return (frame.mi_cols * kMiSize) >> plane_horizontal_shift(frame, plane);
}

int deblock_plane_height(const DeblockFrame& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_rows <= 0) {
    return 0;
  }
  return (frame.mi_rows * kMiSize) >> plane_vertical_shift(frame, plane);
}

bool deblock_supported(const DeblockFrame& frame, const DeblockOptions& options,
                       DeblockStatus* status) {
  auto fail = [status](DeblockStatus s) {
    if (status != nullptr) {
      *status = s;
    }
    return false;
  };

  if (options.bit_depth != 8) {
    return fail(DeblockStatus::kUnsupportedBitDepth);
  }
  if (frame.subsampling_x != 1 || frame.subsampling_y != 1) {
    return fail(DeblockStatus::kUnsupportedSubsampling);
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  // 4:2:0 with an integral chroma 4x4 grid.  This is the standard encoder
  // configuration (frame dimensions padded to a multiple of 8) and guarantees
  // that the chroma filter reach stays inside the MI-aligned plane.
  if ((frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (frame.planes[plane].data == nullptr || frame.mi[plane] == nullptr) {
      return fail(DeblockStatus::kInvalidArgument);
    }
    const int width = deblock_plane_width(frame, plane);
    const int height = deblock_plane_height(frame, plane);
    if (frame.planes[plane].stride < width || height <= 0) {
      return fail(DeblockStatus::kInvalidArgument);
    }
  }
  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr)) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  if (frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  const int max_col_start = frame.mi_cols;
  const int max_row_start = frame.mi_rows;
  for (int i = 0; i < frame.num_tile_cols; ++i) {
    const int v = static_cast<int>(frame.tile_col_starts[i]);
    if (v < 0 || v >= max_col_start) {
      return fail(DeblockStatus::kInvalidArgument);
    }
  }
  for (int i = 0; i < frame.num_tile_rows; ++i) {
    const int v = static_cast<int>(frame.tile_row_starts[i]);
    if (v < 0 || v >= max_row_start) {
      return fail(DeblockStatus::kInvalidArgument);
    }
  }
  if (status != nullptr) {
    *status = DeblockStatus::kOk;
  }
  return true;
}

DeblockCoreFrame deblock_to_core(const DeblockFrame& frame) {
  DeblockCoreFrame core{};
  for (int plane = 0; plane < 3; ++plane) {
    core.plane[plane] = frame.planes[plane].data;
    core.stride[plane] = frame.planes[plane].stride;
    core.mi[plane] = frame.mi[plane];
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.subsampling_x;
  core.sub_y = frame.subsampling_y;
  core.tile_col_starts = frame.tile_col_starts;
  core.num_tile_cols = frame.num_tile_cols;
  core.tile_row_starts = frame.tile_row_starts;
  core.num_tile_rows = frame.num_tile_rows;
  return core;
}

DeblockStatus deblock_frame(DeblockFrame& frame,
                            const DeblockOptions& options) {
  DeblockStatus status = DeblockStatus::kOk;
  if (!deblock_supported(frame, options, &status)) {
    return status;
  }
  const DeblockCoreFrame core = deblock_to_core(frame);
  deblock_frame_core(core, options);
  return DeblockStatus::kOk;
}

int deblock_plane_width16(const DeblockFrame16& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_cols <= 0) {
    return 0;
  }
  const int shift = (plane == 0) ? 0 : frame.subsampling_x;
  return (frame.mi_cols * kMiSize) >> shift;
}

int deblock_plane_height16(const DeblockFrame16& frame, int plane) {
  if (plane < 0 || plane > 2 || frame.mi_rows <= 0) {
    return 0;
  }
  const int shift = (plane == 0) ? 0 : frame.subsampling_y;
  return (frame.mi_rows * kMiSize) >> shift;
}

bool deblock_supported16(const DeblockFrame16& frame,
                         const DeblockOptions& options,
                         DeblockStatus* status) {
  auto fail = [status](DeblockStatus s) {
    if (status != nullptr) {
      *status = s;
    }
    return false;
  };

  if (options.bit_depth != 10 && options.bit_depth != 12) {
    return fail(DeblockStatus::kUnsupportedBitDepth);
  }
  if (frame.num_planes < 1 || frame.num_planes > 3) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  if (frame.subsampling_x < 0 || frame.subsampling_x > 1 ||
      frame.subsampling_y < 0 || frame.subsampling_y > 1) {
    return fail(DeblockStatus::kUnsupportedSubsampling);
  }
  if (frame.num_planes > 1 && frame.num_planes < 3) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  for (int plane = 0; plane < frame.num_planes; ++plane) {
    if (frame.planes[plane].data == nullptr || frame.mi[plane] == nullptr) {
      return fail(DeblockStatus::kInvalidArgument);
    }
    const int width = deblock_plane_width16(frame, plane);
    const int height = deblock_plane_height16(frame, plane);
    if (frame.planes[plane].stride < width || height <= 0) {
      return fail(DeblockStatus::kInvalidArgument);
    }
  }
  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr)) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  if (frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return fail(DeblockStatus::kInvalidArgument);
  }
  if (status != nullptr) {
    *status = DeblockStatus::kOk;
  }
  return true;
}

DeblockCoreFrame16 deblock_to_core16(const DeblockFrame16& frame) {
  DeblockCoreFrame16 core{};
  for (int plane = 0; plane < 3; ++plane) {
    core.plane[plane] = frame.planes[plane].data;
    core.stride[plane] = frame.planes[plane].stride;
    core.mi[plane] = frame.mi[plane];
  }
  core.mi_cols = frame.mi_cols;
  core.mi_rows = frame.mi_rows;
  core.sub_x = frame.subsampling_x;
  core.sub_y = frame.subsampling_y;
  core.num_planes = frame.num_planes;
  core.tile_col_starts = frame.tile_col_starts;
  core.num_tile_cols = frame.num_tile_cols;
  core.tile_row_starts = frame.tile_row_starts;
  core.num_tile_rows = frame.num_tile_rows;
  return core;
}

DeblockStatus deblock_frame16(DeblockFrame16& frame,
                              const DeblockOptions& options) {
  DeblockStatus status = DeblockStatus::kOk;
  if (!deblock_supported16(frame, options, &status)) {
    return status;
  }
  const DeblockCoreFrame16 core = deblock_to_core16(frame);
  deblock_frame_core(core, options);
  return DeblockStatus::kOk;
}

DeblockEdgeInfo deblock_inspect_edge(const DeblockFrame& frame,
                                     const DeblockOptions& options, int plane,
                                     int pass, int row, int col) {
  DeblockEdgeInfo info{};
  DeblockStatus status = DeblockStatus::kOk;
  if (!deblock_supported(frame, options, &status)) {
    return info;
  }
  if (plane < 0 || plane > 2 || pass < 0 || pass > 1 || row < 0 ||
      col < 0 || row >= frame.mi_rows || col >= frame.mi_cols) {
    return info;
  }
  const DeblockCoreFrame core = deblock_to_core(frame);
  return deblock_edge_core<false>(core, options, plane, pass, row, col);
}

}  // namespace av2enc::deblock
