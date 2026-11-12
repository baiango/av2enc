// GPU chroma-format support for 4:2:2, 4:4:4 and 4:0:0 (8-bit and 10-bit).
//
// This is the CUDA counterpart of `src/encoder/format/chroma/`.  It owns the
// *pixel* half of the extra chroma formats:
//
//   * plane geometry (§5.18.4.4, §7.21.2): visible / coded / padded extents
//     for luma and the two chroma planes, with SubsamplingX/Y from Table 6.2;
//   * ChromaMiSize / chroma residual shapes (§5.20.7.26 Subsampled_Size),
//     including the BLOCK_INVALID entries and Monochrome's "no chroma plane";
//   * transform shapes for the extra sizes (TX_32X64, TX_64X64, ...): nominal
//     size, Adjusted_Tx_Size, §7.15.4 duplication factors, §7.15.4.1
//     Transform_Shift from the *nominal* size and the rectangular pre-scale;
//   * forward DCT, quantisation, dequantisation and the inverse DCT, with the
//     BitDepth-dependent clips of §7.14.3/§7.15.2.1 (8 and 10 bit);
//   * DC prediction, forward residual construction and reconstruction
//     (§7.13.2.1/.10, §7.15.4) plus a depth-clipping plane copy.
//
// The shared `__device__` bodies live in `chroma_formats_device.cuh`; the same
// bytes are compiled by the host-emulation test.  The 8-bit arithmetic is
// bit-exact with `av2enc::chroma` (forward_dct / quantize_coefficients /
// dequantize_block / inverse_dct); 10-bit uses the same transforms with the
// 10-bit clips from `av2enc::hbd`.
//
// Memory contract (project-wide, mandatory)
// -----------------------------------------
// Every device byte comes from the single persistent `av2enc::cuda::Arena`.
// There is no cudaMalloc / cudaFree / cudaMallocHost in this component.
//
//   persistent : source, prediction and reconstruction planes plus the
//                per-block work arrays, allocated once by Workspace::init();
//   scratch    : one descriptor-staging array per launch, bracketed by
//                Arena::mark() / Arena::rewind().
//
// `required_device_bytes(config)` is exactly what `Workspace::init()` consumes
// plus the largest per-launch scratch.  A short arena returns
// `ChromaStatus::Code::kArenaTooSmall` with the requirement in
// `required_bytes`; the bump pointer is restored before returning.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "cuda/arena.h"
#include "cuda/error.h"

namespace av2enc::cuda::chroma {

// ---------------------------------------------------------------------------
// Table 6.2
// ---------------------------------------------------------------------------

// Numeric values match `av2enc::fmt::ChromaFormat`, deliberately NOT the
// bitstream idc order.  ChromaFormatIdc is reported by FormatInfo::idc.
enum class Format : std::uint8_t {
  k400 = 0,
  k420 = 1,
  k422 = 2,
  k444 = 3,
};

inline constexpr int kFormatCount = 4;
inline constexpr int kMiSize = 4;
inline constexpr int kMaxPlanes = 3;
inline constexpr int kMaxAdjDim = 32;
inline constexpr int kMaxAdjPels = kMaxAdjDim * kMaxAdjDim;  // 1024
inline constexpr int kMaxCodedPels = 64 * 64;                // 4096
inline constexpr int kMaxFrameDimension = 65535;

struct FormatInfo {
  Format format = Format::k420;
  int idc = 0;                 // ChromaFormatIdc (0, 2, 3 or 1)
  const char* name = "4:2:0";
  int sub_x = 1;               // SubsamplingX
  int sub_y = 1;               // SubsamplingY
  bool monochrome = false;     // Monochrome
  int num_planes = 3;          // NumPlanes
  int profile_idc = 0;         // Annex A.2 named profile
};

bool format_valid(Format format) noexcept;
FormatInfo format_info(Format format) noexcept;
const char* format_name(Format format) noexcept;

// ---------------------------------------------------------------------------
// Plane / frame geometry
// ---------------------------------------------------------------------------

struct PlaneGeom {
  bool present = false;
  int sub_x = 0;
  int sub_y = 0;
  int visible_w = 0;
  int visible_h = 0;
  int coded_w = 0;
  int coded_h = 0;
  int padded_w = 0;  // 64-superblock aligned luma, shifted for chroma
  int padded_h = 0;
};

struct FrameGeom {
  int width = 0;   // visible luma
  int height = 0;
  Format format = Format::k420;
  FormatInfo info{};
  int mi_cols = 0;
  int mi_rows = 0;
  int coded_w = 0;
  int coded_h = 0;
  int padded_w = 0;
  int padded_h = 0;
  int num_planes = 3;
  bool monochrome = false;
  PlaneGeom planes[kMaxPlanes]{};
};

// `make_frame_geometry` is the exact CUDA-side twin of
// `av2enc::chroma::make_frame_shape` (§5.18.4.4 compute_image_size and
// §7.21.2 visible chroma = ceil(width / 2^SubsamplingX)).
FrameGeom make_frame_geometry(int width, int height, Format format) noexcept;
FrameGeom make_frame_geometry(const FormatInfo& info, int width,
                              int height) noexcept;

// Sample position of a luma MI coordinate in a plane:
// (ChromaMiRow >> SubsamplingY) * 4 (same as chroma::plane_origin_*).
inline constexpr int plane_origin_x(int luma_mi_col, int sub_x) noexcept {
  return (luma_mi_col >> sub_x) * kMiSize;
}
inline constexpr int plane_origin_y(int luma_mi_row, int sub_y) noexcept {
  return (luma_mi_row >> sub_y) * kMiSize;
}

// ---------------------------------------------------------------------------
// ChromaMiSize / Subsampled_Size (§5.20.7.26)
// ---------------------------------------------------------------------------

// `block_index` is the AV2 BLOCK_SIZE index; width/height are the chroma
// residual extent.  `valid` is false for Monochrome and for the table's
// BLOCK_INVALID entries (e.g. 4x8 luma in 4:2:2).
struct ResidualSize {
  bool valid = false;
  int block_index = -1;
  int luma_w = 0;
  int luma_h = 0;
  int width = 0;
  int height = 0;
};

ResidualSize chroma_residual_size(int luma_w, int luma_h,
                                  Format format) noexcept;
ResidualSize chroma_residual_size_block(int block_index,
                                        Format format) noexcept;

// §5.20.3.2 is_chroma_offset_for_subsize(): true when a subsampled dimension
// would shrink below 4 samples.
bool is_chroma_offset_for_subsize(int luma_w, int luma_h,
                                  Format format) noexcept;

// Width/height of AV2 BLOCK_SIZE index `block_index` (0..28).
int block_width(int block_index) noexcept;
int block_height(int block_index) noexcept;
int block_size_index(int width, int height) noexcept;

// ---------------------------------------------------------------------------
// Transform shape (nominal txSz -> Adjusted_Tx_Size)
// ---------------------------------------------------------------------------

struct TxShape {
  int nominal_w = 0;    // Tx_Width[txSz]
  int nominal_h = 0;    // Tx_Height[txSz]
  int adj_w = 0;        // Tx_Width[Adjusted_Tx_Size[txSz]]
  int adj_h = 0;
  int upscale_x = 1;    // nominal_w / adj_w (1 or 2)
  int upscale_y = 1;
  int tx_size_ctx = 0;  // §5.20.7.27 txSzCtx
  int eob_multisize = 0;
  int seg_eob = 0;      // adj_w * adj_h
  int tx_pels = 0;      // nominal_w * nominal_h
  int dq_denom = 1;     // 1 << ((tx_pels > 256) + (tx_pels > 1024))
  int row_shift = 0;    // Transform_Shift from the *nominal* size
  int col_shift = 0;
  bool odd_rect = false;
  // Host-computed binary64 constants.  They are part of the descriptor so the
  // device never evaluates sqrt()/division: this is the bit-exactness
  // contract with `av2enc::chroma::forward_dct` and `quant_factor_for`.
  double scale_w = 0.0;        // 1 / (64 * sqrt(adj_w))
  double scale_h = 0.0;        // 1 / (64 * sqrt(adj_h))
  double rect_prescale = 1.0;  // 4096/2896 when odd_rect, else 1
  double quant_factor = 0.0;   // 8*dq_denom / (inv_scale*528)
};

// Same acceptance rule as `av2enc::chroma::make_tx_shape`: 4..64, powers of
// two.  Returns false for anything else.
bool make_tx_shape(int nominal_w, int nominal_h, TxShape* shape) noexcept;

// §5.20.7.27 txSzCtx from a nominal size.
int tx_size_context(int nominal_w, int nominal_h) noexcept;
// §7.15.4.1 rectangular pre-scale predicate.
bool odd_rect_transform(int nominal_w, int nominal_h) noexcept;

// ---------------------------------------------------------------------------
// Status and configuration
// ---------------------------------------------------------------------------

// Distinct outcome of every entry point.  kArenaTooSmall carries the exact
// device byte count the configuration needs.
struct ChromaStatus {
  enum class Code : int {
    kOk = 0,
    kInvalidArgument,
    kNotInitialized,
    kArenaTooSmall,
    kCudaError,
  };

  Code code = Code::kOk;
  std::uint64_t required_bytes = 0;

  bool ok() const noexcept { return code == Code::kOk; }
};

struct Config {
  int width = 64;
  int height = 64;
  Format format = Format::k420;
  int bit_depth = 8;   // 8 or 10 (AV2 signals exactly these two)
  int max_blocks = 16; // per-launch block batch
};

bool config_valid(const Config& config) noexcept;

// Persistent buffers only: planes + per-block work arrays.
std::uint64_t persistent_device_bytes(const Config& config) noexcept;
// Largest per-launch descriptor staging array.
std::uint64_t scratch_device_bytes(const Config& config) noexcept;
// persistent + scratch, aligned like Arena::allocate_checked().
std::uint64_t required_device_bytes(const Config& config) noexcept;

// ---------------------------------------------------------------------------
// Batch jobs
// ---------------------------------------------------------------------------
//
// All pointers are device pointers owned by the caller (typically the
// Workspace accessors below, but any arena-backed buffer is allowed).  Shapes
// come from `make_tx_shape`; the launchers validate them on the host.

// §7.13.2.1/.10 DC prediction of one block.  The whole nominal block receives
// the same rounded prediction; `pred` is packed at nominal_w*nominal_h.
struct PredictJob {
  const std::uint16_t* recon = nullptr;  // reconstruction plane
  int stride = 0;                        // samples per row
  int max_x = 0;                         // coded_w - 1
  int max_y = 0;                         // coded_h - 1
  int x0 = 0;
  int y0 = 0;
  int nominal_w = 0;
  int nominal_h = 0;
  int have_left = 0;
  int have_above = 0;
  int bit_depth = 8;
  std::uint16_t* pred = nullptr;  // nominal_w*nominal_h samples out
};

// residual_adj[j][i] = Round2(sum_{dx,dy}(src - pred), log2(ux*uy)).
// This is the forward side of the §7.15.4 duplication the decoder applies.
struct ResidualJob {
  const std::uint16_t* src = nullptr;  // source plane
  int src_stride = 0;
  int max_x = 0;  // padded_w - 1 (source reads are clamped like CPU)
  int max_y = 0;
  int x0 = 0;
  int y0 = 0;
  int pred = 0;
  int bit_depth = 8;
  TxShape shape{};
  std::int16_t* residual_adj = nullptr;  // adj_w*adj_h samples out
};

// residual_adj -> coeff -> levels.  `coeff` may be null when the caller only
// needs levels; `levels` is required.
struct ForwardJob {
  TxShape shape{};
  const std::int16_t* residual_adj = nullptr;
  double* coeff = nullptr;
  std::int32_t* levels = nullptr;
};

// levels -> dequant (§7.14.3, BitDepth-dependent clip).
struct DequantJob {
  TxShape shape{};
  int bit_depth = 8;
  const std::int32_t* levels = nullptr;
  std::int32_t* dequant = nullptr;
};

// dequant (int32, §7.14.3 clip) -> inverse DCT (§7.15.2.1) at the adjusted
// size.  The coded residual is int16: the column-pass clip is
// [-2^BitDepth, 2^BitDepth - 1] (8-bit [-256,255], 10-bit [-1024,1023]), so
// the transformed residual fits 1 + BitDepth bits and matches
// txv::inverse_block / hbd::inverse_block.  The row-pass intermediate stays
// int32 inside the kernel.
struct InverseJob {
  TxShape shape{};
  int bit_depth = 8;
  const std::int32_t* dequant = nullptr;
  std::int16_t* residual_adj = nullptr;
};

// recon = Clip1(pred + residual_adj), duplicated by upscale_x/upscale_y.
struct ReconstructJob {
  TxShape shape{};
  int bit_depth = 8;
  int pred = 0;
  const std::int16_t* residual_adj = nullptr;
  std::uint16_t* dst = nullptr;
  int dst_stride = 0;
  int max_x = 0;  // padded_w - 1
  int max_y = 0;
  int x0 = 0;
  int y0 = 0;
};

// Depth-clipping rectangular plane copy.
struct CopyJob {
  const std::uint16_t* src = nullptr;
  int src_stride = 0;
  std::uint16_t* dst = nullptr;
  int dst_stride = 0;
  int width = 0;
  int height = 0;
  int bit_depth = 8;
};

// ---------------------------------------------------------------------------
// Workspace: all device memory from one Arena
// ---------------------------------------------------------------------------

class Workspace {
 public:
  Workspace() = default;

  // Allocates the persistent planes and work arrays exactly once.  On failure
  // the bump pointer is restored; kArenaTooSmall reports the requirement.
  ChromaStatus init(Arena& arena, const Config& config,
                    std::string* error = nullptr,
                    std::uint64_t* required_out = nullptr);

  bool initialized() const noexcept { return arena_ != nullptr; }
  const Config& config() const noexcept { return config_; }
  const FrameGeom& geometry() const noexcept { return geometry_; }

  Arena* arena() noexcept { return arena_; }
  std::uint64_t persistent_bytes() const noexcept {
    return persistent_device_bytes(config_);
  }
  std::uint64_t scratch_bytes() const noexcept {
    return scratch_device_bytes(config_);
  }

  // Plane accessors.  `present` is the geometry's present flag; absent planes
  // return nullptr.  Strides are in samples.
  std::uint16_t* src_plane(int plane) noexcept { return d_src_[plane]; }
  const std::uint16_t* src_plane(int plane) const noexcept {
    return d_src_[plane];
  }
  std::uint16_t* recon_plane(int plane) noexcept { return d_recon_[plane]; }
  const std::uint16_t* recon_plane(int plane) const noexcept {
    return d_recon_[plane];
  }
  int plane_stride(int plane) const noexcept {
    return geometry_.planes[plane].padded_w;
  }
  std::size_t plane_samples(int plane) const noexcept;

  // Per-block work arrays.  `slot` < config().max_blocks; each slot holds
  // kMaxAdjPels (residual/coeff/levels/dequant) or kMaxCodedPels (pred).
  // `residual_adj` is int16 for both directions: the forward residual is
  // source-pred, and the inverse §7.15.2.1 column clip bounds its output to
  // [-2^BitDepth, 2^BitDepth - 1], so no int32 residual buffer is needed.
  std::uint16_t* pred_block(int slot) noexcept;
  std::int16_t* residual_adj(int slot) noexcept;
  double* coeff(int slot) noexcept;
  std::int32_t* levels(int slot) noexcept;
  std::int32_t* dequant(int slot) noexcept;

  // Batch entry points.  Each stages its descriptor array in arena scratch,
  // launches, synchronises on `stream`, then rewinds the arena.  Synchronous
  // by design: the scratch is dead before it is reused.
  ChromaStatus predict(const PredictJob* host_jobs, int count,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);
  ChromaStatus make_residuals(const ResidualJob* host_jobs, int count,
                              cudaStream_t stream = nullptr,
                              std::string* error = nullptr);
  ChromaStatus forward(const ForwardJob* host_jobs, int count,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);
  ChromaStatus dequantize(const DequantJob* host_jobs, int count,
                          cudaStream_t stream = nullptr,
                          std::string* error = nullptr);
  ChromaStatus inverse(const InverseJob* host_jobs, int count,
                       cudaStream_t stream = nullptr,
                       std::string* error = nullptr);
  ChromaStatus reconstruct(const ReconstructJob* host_jobs, int count,
                           cudaStream_t stream = nullptr,
                           std::string* error = nullptr);
  ChromaStatus copy_planes(const CopyJob* host_jobs, int count,
                           cudaStream_t stream = nullptr,
                           std::string* error = nullptr);

 private:
  ChromaStatus launch_error(const char* what, cudaError_t status,
                            std::string* error) const;

  Arena* arena_ = nullptr;
  Config config_{};
  FrameGeom geometry_{};
  std::uint16_t* d_src_[kMaxPlanes] = {nullptr, nullptr, nullptr};
  std::uint16_t* d_recon_[kMaxPlanes] = {nullptr, nullptr, nullptr};
  std::uint16_t* d_pred_ = nullptr;
  std::int16_t* d_residual_ = nullptr;
  double* d_coeff_ = nullptr;
  std::int32_t* d_levels_ = nullptr;
  std::int32_t* d_dequant_ = nullptr;
};

// ---------------------------------------------------------------------------
// Workspace::init (header-inline so the no-GPU emulation test exercises the
// same allocation plan as the CUDA translation unit)
// ---------------------------------------------------------------------------

namespace detail {

inline constexpr std::uint64_t align_up(std::uint64_t value,
                                       std::uint64_t alignment) noexcept {
  return alignment == 0 ? value
                        : ((value + alignment - 1) / alignment) * alignment;
}

inline constexpr std::uint64_t plane_bytes(const PlaneGeom& plane) noexcept {
  return plane.present
             ? align_up(static_cast<std::uint64_t>(plane.padded_w) *
                            static_cast<std::uint64_t>(plane.padded_h) *
                            sizeof(std::uint16_t),
                        Arena::kAlignment)
             : 0;
}

inline constexpr std::uint64_t batch_bytes(int max_blocks,
                                           std::uint64_t element) noexcept {
  return align_up(static_cast<std::uint64_t>(max_blocks) * element,
                  Arena::kAlignment);
}

}  // namespace detail

inline std::uint64_t persistent_device_bytes(const Config& config) noexcept {
  if (!config_valid(config)) {
    return 0;
  }
  const FrameGeom geometry =
      make_frame_geometry(config.width, config.height, config.format);
  std::uint64_t bytes = 0;
  for (int plane = 0; plane < kMaxPlanes; ++plane) {
    bytes += detail::plane_bytes(geometry.planes[plane]);  // source
    bytes += detail::plane_bytes(geometry.planes[plane]);  // recon
  }
  const std::uint64_t blocks =
      static_cast<std::uint64_t>(config.max_blocks);
  bytes += detail::align_up(blocks * kMaxCodedPels * sizeof(std::uint16_t),
                            Arena::kAlignment);  // prediction blocks
  bytes += detail::align_up(blocks * kMaxAdjPels * sizeof(std::int16_t),
                            Arena::kAlignment);  // residual_adj
  bytes += detail::align_up(blocks * kMaxAdjPels * sizeof(double),
                            Arena::kAlignment);  // coeff
  bytes += detail::align_up(blocks * kMaxAdjPels * sizeof(std::int32_t),
                            Arena::kAlignment);  // levels
  bytes += detail::align_up(blocks * kMaxAdjPels * sizeof(std::int32_t),
                            Arena::kAlignment);  // dequant
  return bytes;
}

inline std::uint64_t scratch_device_bytes(const Config& config) noexcept {
  if (!config_valid(config)) {
    return 0;
  }
  const std::uint64_t blocks =
      static_cast<std::uint64_t>(config.max_blocks);
  const std::uint64_t jobs[7] = {
      detail::align_up(blocks * sizeof(PredictJob), Arena::kAlignment),
      detail::align_up(blocks * sizeof(ResidualJob), Arena::kAlignment),
      detail::align_up(blocks * sizeof(ForwardJob), Arena::kAlignment),
      detail::align_up(blocks * sizeof(DequantJob), Arena::kAlignment),
      detail::align_up(blocks * sizeof(InverseJob), Arena::kAlignment),
      detail::align_up(blocks * sizeof(ReconstructJob), Arena::kAlignment),
      detail::align_up(blocks * sizeof(CopyJob), Arena::kAlignment),
  };
  std::uint64_t largest = 0;
  for (std::uint64_t value : jobs) {
    if (value > largest) {
      largest = value;
    }
  }
  return largest;
}

inline std::uint64_t required_device_bytes(const Config& config) noexcept {
  if (!config_valid(config)) {
    return 0;
  }
  return persistent_device_bytes(config) + scratch_device_bytes(config);
}

inline ChromaStatus Workspace::init(Arena& arena, const Config& config,
                                    std::string* error,
                                    std::uint64_t* required_out) {
  arena_ = nullptr;
  config_ = Config{};
  geometry_ = FrameGeom{};
  for (int plane = 0; plane < kMaxPlanes; ++plane) {
    d_src_[plane] = nullptr;
    d_recon_[plane] = nullptr;
  }
  d_pred_ = nullptr;
  d_residual_ = nullptr;
  d_coeff_ = nullptr;
  d_levels_ = nullptr;
  d_dequant_ = nullptr;

  if (!config_valid(config)) {
    if (required_out != nullptr) *required_out = 0;
    if (error != nullptr) {
      *error = "chroma_formats: invalid workspace configuration";
    }
    return ChromaStatus{ChromaStatus::Code::kInvalidArgument, 0};
  }

  const std::uint64_t need = required_device_bytes(config);
  if (required_out != nullptr) *required_out = need;
  if (!arena.initialized()) {
    if (error != nullptr) {
      *error = "chroma_formats: arena is not initialized";
    }
    return ChromaStatus{ChromaStatus::Code::kNotInitialized, need};
  }
  if (arena.remaining() < need) {
    if (error != nullptr) {
      *error = "chroma_formats: arena needs " + std::to_string(need) +
               " bytes, " + std::to_string(arena.remaining()) + " available";
    }
    return ChromaStatus{ChromaStatus::Code::kArenaTooSmall, need};
  }

  const FrameGeom geometry =
      make_frame_geometry(config.width, config.height, config.format);
  const std::uint64_t mark = arena.mark();
  bool ok = true;
  for (int plane = 0; plane < kMaxPlanes; ++plane) {
    if (!geometry.planes[plane].present) continue;
    const std::size_t samples =
        static_cast<std::size_t>(geometry.planes[plane].padded_w) *
        static_cast<std::size_t>(geometry.planes[plane].padded_h);
    void* raw = nullptr;
    if (arena.allocate_checked(samples * sizeof(std::uint16_t), &raw) !=
        Status::kOk) {
      ok = false;
      break;
    }
    d_src_[plane] = static_cast<std::uint16_t*>(raw);
    raw = nullptr;
    if (arena.allocate_checked(samples * sizeof(std::uint16_t), &raw) !=
        Status::kOk) {
      ok = false;
      break;
    }
    d_recon_[plane] = static_cast<std::uint16_t*>(raw);
  }

  const std::size_t blocks =
      static_cast<std::size_t>(config.max_blocks);
  void* raw = nullptr;
  if (ok && arena.allocate_checked(blocks * kMaxCodedPels *
                                       sizeof(std::uint16_t),
                                   &raw) == Status::kOk) {
    d_pred_ = static_cast<std::uint16_t*>(raw);
  } else {
    ok = false;
  }
  raw = nullptr;
  if (ok && arena.allocate_checked(
                blocks * kMaxAdjPels * sizeof(std::int16_t), &raw) ==
                Status::kOk) {
    d_residual_ = static_cast<std::int16_t*>(raw);
  } else {
    ok = false;
  }
  raw = nullptr;
  if (ok && arena.allocate_checked(blocks * kMaxAdjPels * sizeof(double),
                                   &raw) == Status::kOk) {
    d_coeff_ = static_cast<double*>(raw);
  } else {
    ok = false;
  }
  raw = nullptr;
  if (ok && arena.allocate_checked(blocks * kMaxAdjPels * sizeof(std::int32_t),
                                   &raw) == Status::kOk) {
    d_levels_ = static_cast<std::int32_t*>(raw);
  } else {
    ok = false;
  }
  raw = nullptr;
  if (ok && arena.allocate_checked(blocks * kMaxAdjPels * sizeof(std::int32_t),
                                   &raw) == Status::kOk) {
    d_dequant_ = static_cast<std::int32_t*>(raw);
  } else {
    ok = false;
  }

  if (!ok) {
    arena.rewind(mark);
    for (int plane = 0; plane < kMaxPlanes; ++plane) {
      d_src_[plane] = nullptr;
      d_recon_[plane] = nullptr;
    }
    d_pred_ = nullptr;
    d_residual_ = nullptr;
    d_coeff_ = nullptr;
    d_levels_ = nullptr;
    d_dequant_ = nullptr;
    if (error != nullptr) {
      *error = "chroma_formats: arena exhausted after " +
               std::to_string(need) + " bytes";
    }
    return ChromaStatus{ChromaStatus::Code::kArenaTooSmall, need};
  }

  config_ = config;
  geometry_ = geometry;
  arena_ = &arena;
  return ChromaStatus{};
}

inline std::size_t Workspace::plane_samples(int plane) const noexcept {
  if (plane < 0 || plane >= kMaxPlanes) return 0;
  const PlaneGeom& geom = geometry_.planes[plane];
  if (!geom.present) return 0;
  return static_cast<std::size_t>(geom.padded_w) *
         static_cast<std::size_t>(geom.padded_h);
}

inline std::uint16_t* Workspace::pred_block(int slot) noexcept {
  if (d_pred_ == nullptr || slot < 0 || slot >= config_.max_blocks) {
    return nullptr;
  }
  return d_pred_ + static_cast<std::size_t>(slot) * kMaxCodedPels;
}
inline std::int16_t* Workspace::residual_adj(int slot) noexcept {
  if (d_residual_ == nullptr || slot < 0 || slot >= config_.max_blocks) {
    return nullptr;
  }
  return d_residual_ + static_cast<std::size_t>(slot) * kMaxAdjPels;
}
inline double* Workspace::coeff(int slot) noexcept {
  if (d_coeff_ == nullptr || slot < 0 || slot >= config_.max_blocks) {
    return nullptr;
  }
  return d_coeff_ + static_cast<std::size_t>(slot) * kMaxAdjPels;
}
inline std::int32_t* Workspace::levels(int slot) noexcept {
  if (d_levels_ == nullptr || slot < 0 || slot >= config_.max_blocks) {
    return nullptr;
  }
  return d_levels_ + static_cast<std::size_t>(slot) * kMaxAdjPels;
}
inline std::int32_t* Workspace::dequant(int slot) noexcept {
  if (d_dequant_ == nullptr || slot < 0 || slot >= config_.max_blocks) {
    return nullptr;
  }
  return d_dequant_ + static_cast<std::size_t>(slot) * kMaxAdjPels;
}

}  // namespace av2enc::cuda::chroma
