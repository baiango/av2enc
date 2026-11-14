// Minimal still-picture writer for 4:2:2, 4:4:4 and 4:0:0.
//
// This is a *format* deliverable, not a new production encoder.  It reuses the
// existing AV2 building blocks:
//
//   * `src/bitstream/`        OBU framing, temporal delimiter, M0 frame header
//   * `src/entropy/av2/`      §8.2 symbol coder and CDF tables
//   * `src/encoder/basic_*`      DCT kernels, qlookup(90) quantiser, 8-bit recon
//   * `src/encoder/format/`   420<->422/444/400 sample conversion
//   * `chroma_residual.*`     Adjusted_Tx_Size / duplication for 32x64, 64x64
//
// Stream shape: one key frame, one tile, one 64x64 PARTITION_NONE superblock at
// a time, DC_PRED intra, TX_MODE_LARGEST, DCT_DCT, no filters, frozen CDFs.
// The 64x64 leaf is legal for every superblock of the gate geometries (64x64
// and 640x360); `supports_64x64_none()` rejects anything else instead of
// writing an invalid partition.
#pragma once

#include "formats/chroma/chroma_format.h"
#include "formats/chroma/chroma_filters.h"
#include "formats/format.h"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace av2enc::chroma {

// One 8-bit input picture in the destination chroma format.  `planes[p]` is
// padded to FrameShape::planes[p].padded_width/height; the visible rectangle
// starts at (0,0) and the coded/padded remainder is edge-replicated.
struct ChromaPicture {
  int width = 0;   // visible luma
  int height = 0;
  ChromaFormat chroma = ChromaFormat::k420;
  int plane_width[3] = {0, 0, 0};
  int plane_height[3] = {0, 0, 0};
  std::vector<std::uint8_t> planes[3];
};

// Converts an 8-bit 4:2:0 source to `chroma` using
// fmt::convert_420_8_to_picture() (sample replication, no interpolation), then
// edge-replicates into the 64x64-superblock padding.
bool make_chroma_picture(int width, int height, ChromaFormat chroma,
                         std::span<const std::uint8_t> y,
                         std::span<const std::uint8_t> u,
                         std::span<const std::uint8_t> v,
                         ChromaPicture& picture, std::string& error);

// Same, from an already-converted fmt::Picture.  `picture.format.chroma` must
// match; the source is expected to be an 8-bit payload (LSB-aligned samples).
bool make_chroma_picture(const fmt::Picture& source, ChromaPicture& picture,
                         std::string& error);

struct TxShape;

// Optional device block executor for the 8-bit chroma product path.  The CPU
// writer keeps the §8.2 entropy/syntax loop; this interface replaces only the
// per-block pixel work (DC prediction, forward residual/transform/quantiser,
// dequant/inverse/reconstruction) and owns the device reconstruction planes.
// `encode_chroma_still_picture()` calls `begin()` once, `forward()` before the
// coefficient syntax, `reconstruct()` after it, and `download_recon()` at the
// end.  A null pointer keeps the CPU reference path byte-for-byte.
class ChromaGpuExecutor {
 public:
  virtual ~ChromaGpuExecutor() = default;

  virtual bool begin(const ChromaPicture& picture, std::string& error) = 0;

  // Fills `pred_value`, `levels` (tx.adj_w * tx.adj_h) and the 1-based `eob`.
  virtual bool forward(int plane, int x0, int y0, int nominal_w, int nominal_h,
                       const TxShape& tx, bool have_left, bool have_above,
                       int* pred_value, int* levels, int* eob,
                       std::string& error) = 0;

  // Dequant + inverse + `Clip1(pred + residual)` into the device recon plane.
  // `eob == 0` writes the constant prediction.
  virtual bool reconstruct(int plane, int x0, int y0, int nominal_w,
                           int nominal_h, const TxShape& tx, int pred_value,
                           const int* levels, int eob, std::string& error) = 0;

  // Copies the final padded reconstruction plane into `out`.
  virtual bool download_recon(int plane, std::uint8_t* out,
                              std::string& error) = 0;
};

struct ChromaStillOptions {
  std::uint8_t base_q_idx = 90;  // qlookup(90) = 528, §8.3.1 slice 0
  std::uint8_t seq_level_idx = 7;
  bool disable_cdf_update = true;
  ChromaGpuExecutor* gpu = nullptr;
  // §7.17-§7.20 in-loop filters.  All off by default, so every existing
  // stream stays byte-identical.  The filters require frozen CDFs (the
  // filter symbol CDFs are not adapted).
  ChromaFilterOptions filters;
};

struct ChromaStillResult {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK
  FrameShape shape;
  // Reconstructed planes, padded to shape.planes[p].padded_width/height.
  // These are the *filtered* planes when any in-loop filter is enabled.
  int recon_width[3] = {0, 0, 0};
  int recon_height[3] = {0, 0, 0};
  std::vector<std::uint8_t> recon[3];
  int leaves = 0;
};

// Encodes one still picture.  Returns false with `error` for an unsupported
// geometry, a chroma residual shape with no legal Subsampled_Size entry, an
// entropy coder failure, or invalid options.
bool encode_chroma_still_picture(const ChromaPicture& picture,
                                 const ChromaStillOptions& options,
                                 ChromaStillResult& result, std::string& error);

// Copies the visible rectangle of a reconstructed plane into `out`
// (visible_width * visible_height, tightly packed).
bool copy_visible_plane(const ChromaStillResult& result, int plane,
                        std::vector<std::uint8_t>& out);

// Writes a Y4M frame header + one frame for the reconstructed visible picture.
// Used by the gate tool; chroma tags: C420, C422, C444, Cmono.
bool write_recon_y4m(const ChromaStillResult& result, std::FILE* file,
                     std::uint32_t fps_num, std::uint32_t fps_den,
                     std::string& error);

}  // namespace av2enc::chroma
