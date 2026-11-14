#include "formats/format_encode.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/io/ivf.h"
#include "formats/chroma/chroma_still.h"
#include "formats/format_y4m.h"
#include "formats/high_bitdepth/hbd_still.h"

namespace av2enc::formats {
namespace {

using av2enc::fmt::BitDepth;
using av2enc::fmt::ChromaFormat;
using av2enc::fmt::Format;
using av2enc::fmt::Picture;

bool write_10bit_recon(const Picture& picture,
                       const av2enc::high_bitdepth::HighBitDepthPlane recon_planes[3],
                       const std::filesystem::path& path, std::string* error) {
  Picture recon;
  Format format;
  format.width = picture.format.width;
  format.height = picture.format.height;
  format.chroma = picture.format.chroma;
  format.depth = BitDepth::k10;
  if (!av2enc::fmt::allocate_picture(format, recon, *error)) {
    return false;
  }
  const int planes = av2enc::fmt::num_planes(format.chroma);
  for (int plane = 0; plane < planes; ++plane) {
    const av2enc::high_bitdepth::HighBitDepthPlane& src = recon_planes[plane];
    av2enc::fmt::PlaneBuffer& dst = recon.planes[plane];
    for (int y = 0; y < src.visible_height; ++y) {
      for (int x = 0; x < src.visible_width; ++x) {
        dst.at(static_cast<std::uint32_t>(x),
               static_cast<std::uint32_t>(y)) =
            src.samples[static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(src.width) +
                        static_cast<std::size_t>(x)];
      }
    }
    av2enc::fmt::replicate_borders(dst);
  }
  return av2enc::fmt::write_y4m_picture(path, recon, error);
}

bool encode_10bit(const Picture& picture,
                  const StillFormatOptions& options,
                  std::vector<std::uint8_t>* frame,
                  std::uint32_t* tile_bytes, std::string* error) {
  av2enc::high_bitdepth::HighBitDepthResult result;
  if (options.high_bitdepth_gpu != nullptr) {
    if (options.enable_cdef) {
      *error = "--still-cdef has no GPU block stage for the 10-bit format "
               "writer; use --still (CPU)";
      return false;
    }
    if (options.enable_ccso) {
      *error = "--still-ccso has no GPU block stage for the 10-bit format "
               "writer; use --still (CPU)";
      return false;
    }
    if (options.enable_wiener || options.enable_chroma_lr ||
        options.enable_pc_wiener || options.enable_gdf) {
      *error = "loop restoration/GDF have no GPU block stage for the 10-bit "
               "format writer; use --still (CPU)";
      return false;
    }
    // Product 10-bit GPU block stage: DC prediction + forward/inverse TX +
    // reconstruction run on the device; the shared CPU §8.2 writer assembles
    // the same tile payload bytes as `av2enc::hbd::encode_still`.
    av2enc::high_bitdepth::HighBitDepthOptions gpu_options;
    gpu_options.base_q_idx = options.base_q_idx;
    gpu_options.enable_deblock = options.enable_deblock;
    if (!options.high_bitdepth_gpu(picture, gpu_options, result, *error)) {
      return false;
    }
  } else {
    av2enc::hbd::HbdOptions hbd_options;
    hbd_options.base_q_idx = options.base_q_idx;
    hbd_options.enable_deblock = options.enable_deblock;
    hbd_options.enable_cdef = options.enable_cdef;
    hbd_options.cdef_damping = options.cdef_damping;
    hbd_options.cdef_num_strengths = options.cdef_num_strengths;
    hbd_options.enable_ccso = options.enable_ccso;
    hbd_options.ccso_luma_size_log2 = options.ccso_luma_size_log2;
    hbd_options.ccso_edge_only = options.ccso_edge_only;
    hbd_options.enable_wiener = options.enable_wiener;
    hbd_options.enable_chroma_lr = options.enable_chroma_lr;
    hbd_options.enable_pc_wiener = options.enable_pc_wiener;
    hbd_options.enable_gdf = options.enable_gdf;
    av2enc::hbd::HbdResult cpu_result;
    if (!av2enc::hbd::encode_still(picture, hbd_options, cpu_result, *error)) {
      return false;
    }
    result.frame = std::move(cpu_result.frame);
    result.tile_bytes = cpu_result.tile_bytes;
    for (int plane = 0; plane < 3; ++plane) {
      result.recon[plane] = std::move(cpu_result.recon[plane]);
    }
  }
  *frame = std::move(result.frame);
  *tile_bytes = result.tile_bytes;

  if (!options.recon_out.empty()) {
    return write_10bit_recon(picture, result.recon, options.recon_out, error);
  }
  return true;
}

bool encode_8bit_chroma(const Picture& picture,
                        const StillFormatOptions& options,
                        std::vector<std::uint8_t>* frame,
                        std::uint32_t* tile_bytes, std::string* error) {
  av2enc::chroma::ChromaPicture chroma_picture;
  if (!av2enc::chroma::make_chroma_picture(picture, chroma_picture, *error)) {
    return false;
  }
  av2enc::chroma::ChromaStillOptions chroma_options;
  chroma_options.base_q_idx = static_cast<std::uint8_t>(options.base_q_idx);
  chroma_options.gpu = options.chroma_gpu;
  // §7.17-§7.20 in-loop filters (all off by default).
  chroma_options.filters.base_q_idx = options.base_q_idx;
  chroma_options.filters.enable_deblock = options.enable_deblock;
  chroma_options.filters.enable_cdef = options.enable_cdef;
  chroma_options.filters.cdef_damping = options.cdef_damping;
  chroma_options.filters.cdef_num_strengths = options.cdef_num_strengths;
  chroma_options.filters.enable_ccso = options.enable_ccso;
  chroma_options.filters.ccso_luma_size_log2 = options.ccso_luma_size_log2;
  chroma_options.filters.ccso_edge_only = options.ccso_edge_only;
  chroma_options.filters.enable_wiener = options.enable_wiener;
  chroma_options.filters.enable_chroma_lr = options.enable_chroma_lr;
  chroma_options.filters.enable_pc_wiener = options.enable_pc_wiener;
  chroma_options.filters.enable_gdf = options.enable_gdf;
  av2enc::chroma::ChromaStillResult result;
  if (!av2enc::chroma::encode_chroma_still_picture(chroma_picture,
                                                   chroma_options, result,
                                                   *error)) {
    return false;
  }
  *frame = std::move(result.frame);
  *tile_bytes = result.tile_bytes;

  if (!options.recon_out.empty()) {
    Picture recon;
    Format format;
    format.width = static_cast<std::uint32_t>(result.shape.width);
    format.height = static_cast<std::uint32_t>(result.shape.height);
    format.chroma = result.shape.chroma;
    format.depth = BitDepth::k8;
    if (!av2enc::fmt::allocate_picture(format, recon, *error)) {
      return false;
    }
    const int planes = av2enc::fmt::num_planes(format.chroma);
    for (int plane = 0; plane < planes; ++plane) {
      const av2enc::chroma::PlaneShape& shape =
          result.shape.planes[static_cast<std::size_t>(plane)];
      av2enc::fmt::PlaneBuffer& dst = recon.planes[plane];
      for (int y = 0; y < shape.visible_height; ++y) {
        for (int x = 0; x < shape.visible_width; ++x) {
          const std::size_t index =
              static_cast<std::size_t>(y) *
                  static_cast<std::size_t>(result.recon_width[plane]) +
              static_cast<std::size_t>(x);
          dst.at(static_cast<std::uint32_t>(x),
                 static_cast<std::uint32_t>(y)) =
              result.recon[plane][index];
        }
      }
      av2enc::fmt::replicate_borders(dst);
    }
    if (!av2enc::fmt::write_y4m_picture(options.recon_out, recon, error)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool format_still_supported(ChromaFormat chroma, BitDepth depth,
                            std::string* reason) {
  if (fmt::chroma_format_idc(chroma) < 0) {
    if (reason != nullptr) *reason = "invalid chroma format";
    return false;
  }
  if (depth == BitDepth::k10) {
    return true;  // 4:2:0 / 4:2:2 / 4:4:4 / 4:0:0, one still writer
  }
  if (depth == BitDepth::k8) {
    if (chroma == ChromaFormat::k420) {
      return true;  // default path; not routed through this helper
    }
    return true;
  }
  if (reason != nullptr) {
    *reason = "only 8-bit and 10-bit are signalable in AV2 v1.0.0";
  }
  return false;
}

bool encode_still_format(const std::filesystem::path& input,
                         const std::filesystem::path& output,
                         const StillFormatOptions& options,
                         StillFormatStats* stats, std::string* error) {
  std::string reason;
  if (!format_still_supported(options.chroma, options.depth, &reason)) {
    *error = reason;
    return false;
  }

  Picture picture;
  av2enc::fmt::Y4mFormatInfo info;
  if (!av2enc::fmt::read_y4m_picture(input, &picture, &info, error)) {
    return false;
  }
  if (info.chroma != options.chroma || info.depth != options.depth) {
    *error = "input is " + av2enc::fmt::y4m_colorspace_tag(info.chroma,
                                                           info.depth) +
             " but the requested format is " +
             av2enc::fmt::y4m_colorspace_tag(options.chroma, options.depth) +
             "; the format paths do not resample or convert input";
    return false;
  }

  std::vector<std::uint8_t> frame;
  std::uint32_t tile_bytes = 0;
  bool ok = false;
  if (options.depth == BitDepth::k10) {
    ok = encode_10bit(picture, options, &frame, &tile_bytes, error);
  } else {
    ok = encode_8bit_chroma(picture, options, &frame, &tile_bytes, error);
  }
  if (!ok) {
    return false;
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(info.width);
  ivf.height = static_cast<std::uint16_t>(info.height);
  ivf.fps_num = info.fps_num;
  ivf.fps_den = info.fps_den;

  av2enc::IvfWriter writer;
  if (!writer.open(output, ivf, *error)) {
    return false;
  }
  if (!writer.write_frame(frame, 0, *error)) {
    std::string close_error;
    writer.close(close_error);
    return false;
  }
  if (!writer.close(*error)) {
    return false;
  }

  if (stats != nullptr) {
    stats->width = info.width;
    stats->height = info.height;
    stats->fps_num = info.fps_num;
    stats->fps_den = info.fps_den;
    stats->chroma = info.chroma;
    stats->depth = info.depth;
    stats->tile_bytes = tile_bytes;
    stats->frame_bytes = static_cast<std::uint32_t>(frame.size());
    stats->ivf_bytes = writer.bytes_written();
  }
  return true;
}

}  // namespace av2enc::formats
