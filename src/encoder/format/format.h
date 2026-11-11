// Standalone high-bit-depth / chroma format support for the AV2 encoder.
//
// This component owns the *format* layer only: sample representation, plane
// geometry, edge rules, range/clipping, and conversions from the M0 8-bit
// 4:2:0 input.  It does not write headers, entropy symbols, or a bitstream, and
// it is not wired into `src/encoder/m0_still.cpp`.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (commit 38f28e7).  Key clauses:
//
//   §4.8          Clip1(x) = Clip3(0, 2^BitDepth - 1, x)
//   §5.4.1        sequence_header_obu(): chroma_format_idc, bit_depth_idc
//   §5.18.6.1     quantization_params(): base_q_idx f(8 or 9)
//   §5.20.7.26    get_plane_residual_size() / Subsampled_Size
//   §6.4.1        Table 6.2 chroma formats, Table 6.3 bit depths
//   §7.13.2.10    DC prediction bias = 1 << (BitDepth - 1)
//   §7.14.3       dequant clip = ±(1 << (7 + BitDepth))
//   §7.15.2.1     1D inverse transform clip depends on BitDepth
//   §8.3.1        init_coeff_cdfs(): base_q_idx <= 90/140/190
//   Annex A.2     profiles (4:2:2 -> profile 3, 4:4:4 -> profile 4)
//
// Important spec finding: AV2 v1.0.0 signals only 8-bit and 10-bit.
// `bit_depth_idc` is a one-bit value (0 = 10-bit, 1 = 8-bit); values greater
// than 1 are reserved for AOMedia (§6.4.1).  This component therefore supports
// 12-bit *internal samples* for buffers/conversions/tests, but 12-bit cannot be
// represented in a conforming AV2 v1.0.0 bitstream.  `is_signalable()` and
// `bit_depth_idc()` report that distinction; no encoder path may emit 12-bit.
//
// Sample storage is planar, tightly packed, one `uint16_t` per sample, with the
// sample value in the low bits (LSB-aligned) and the high bits zero.  8-bit
// pictures use the same `uint16_t` container so all depths share one code path;
// this is a reference/utility layer, not the GPU hot path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace av2enc::fmt {

// ---------------------------------------------------------------------------
// Basic enums and format descriptors
// ---------------------------------------------------------------------------

enum class BitDepth : std::uint8_t {
  k8 = 8,
  k10 = 10,
  k12 = 12,  // internal only; reserved in AV2 v1.0.0 bitstreams
};

// Semantic chroma formats.  The numeric values are *not* chroma_format_idc;
// use `chroma_format_idc()` for the bitstream mapping (which is not monotonic:
// 4:2:0 = 0, 4:0:0 = 1, 4:4:4 = 2, 4:2:2 = 3, §6.4.1 Table 6.2).
enum class ChromaFormat : std::uint8_t {
  k400 = 0,
  k420 = 1,
  k422 = 2,
  k444 = 3,
};

// AV2 BLOCK_SIZE values, Table 6.22 / §6.19.3.1.  Kept local so this component
// stays standalone; `chroma_residual_size()` is the only consumer.
enum class BlockSize : std::uint8_t {
  k4x4 = 0,
  k4x8 = 1,
  k8x4 = 2,
  k8x8 = 3,
  k8x16 = 4,
  k16x8 = 5,
  k16x16 = 6,
  k16x32 = 7,
  k32x16 = 8,
  k32x32 = 9,
  k32x64 = 10,
  k64x32 = 11,
  k64x64 = 12,
  k64x128 = 13,
  k128x64 = 14,
  k128x128 = 15,
  k128x256 = 16,
  k256x128 = 17,
  k256x256 = 18,
  k4x16 = 19,
  k16x4 = 20,
  k8x32 = 21,
  k32x8 = 22,
  k16x64 = 23,
  k64x16 = 24,
  k4x32 = 25,
  k32x4 = 26,
  k8x64 = 27,
  k64x8 = 28,
};

inline constexpr int kMiSize = 4;
inline constexpr int kMiSizeLog2 = 2;

// Implementation limit, not a spec limit.  AV2 dimensions are uvlc-coded; this
// bound keeps size arithmetic and test allocations sane.
inline constexpr std::uint32_t kMaxDimension = 65535;

struct Format {
  std::uint32_t width = 0;   // visible/cropped luma width
  std::uint32_t height = 0;  // visible/cropped luma height
  ChromaFormat chroma = ChromaFormat::k420;
  BitDepth depth = BitDepth::k8;
};

bool operator==(const Format& a, const Format& b);
bool operator!=(const Format& a, const Format& b);

// ---------------------------------------------------------------------------
// Bit depth, chroma, and quantizer parameters
// ---------------------------------------------------------------------------

int bit_depth_value(BitDepth depth);
const char* bit_depth_name(BitDepth depth);

// Largest valid sample value: (1 << BitDepth) - 1.
int sample_max(BitDepth depth);

// §6.4.1 Table 6.3: 0 = 10-bit, 1 = 8-bit.  Returns -1 for 12-bit because
// bit_depth_idc > 1 is reserved in AV2 v1.0.0.
int bit_depth_idc(BitDepth depth);
std::optional<BitDepth> bit_depth_from_idc(int idc);

// MaxQ: 255 for 8-bit, 303 for 10-bit (MAXQ_8_BITS + 2 * MAXQ_OFFSET).
// 12-bit returns MAXQ_BITS = 351 for internal bookkeeping only; it is reserved
// in the bitstream.
int max_quantizer(BitDepth depth);

// quantization_params() codes base_q_idx with f(8) for 8-bit and f(9)
// otherwise (§5.18.6.1).
int base_q_idx_bits(BitDepth depth);

int chroma_format_idc(ChromaFormat chroma);
std::optional<ChromaFormat> chroma_format_from_idc(int idc);
int subsampling_x(ChromaFormat chroma);
int subsampling_y(ChromaFormat chroma);
int num_planes(ChromaFormat chroma);
bool has_chroma(ChromaFormat chroma);
const char* chroma_format_name(ChromaFormat chroma);

// ---------------------------------------------------------------------------
// Block and chroma residual geometry (§5.20.7.26, Table 6.22)
// ---------------------------------------------------------------------------

int block_width(BlockSize size);
int block_height(BlockSize size);
const char* block_size_name(BlockSize size);
std::optional<BlockSize> find_block_size(int width, int height);

// get_plane_residual_size() via the normative Subsampled_Size table.
// Nullopt for monochrome (no chroma residual) or a table entry that is
// BLOCK_INVALID (for example 4x8 luma in 4:2:2).
std::optional<BlockSize> chroma_residual_size(BlockSize luma, ChromaFormat chroma);
std::optional<BlockSize> chroma_residual_size(int luma_width, int luma_height,
                                              ChromaFormat chroma);

// ---------------------------------------------------------------------------
// Frame / plane geometry
// ---------------------------------------------------------------------------
//
// Two sizes matter and are easy to confuse:
//
//   * coded size: MiCols*MI_SIZE by MiRows*MI_SIZE for luma, halved (rounded
//     down) for chroma.  This is the size of the decoder's CurrFrame /
//     FrameStore planes and therefore the size of a GPU frame buffer.
//   * visible size: the cropped picture.  Luma is width x height; chroma is
//     (width + subX) >> subX by (height + subY) >> subY (§7.21.2).
//
// Coded luma extents are aligned to 8 (MiCols = 2 * ((width + 7) >> 3)).
// Chroma coded extents are the luma coded extent shifted right by the
// subsampling factors.  The padding between visible and coded size must be
// initialized (this component replicates the edge samples) so edge prediction
// reads defined values.

struct PlaneGeometry {
  std::uint32_t coded_width = 0;
  std::uint32_t coded_height = 0;
  std::uint32_t visible_width = 0;
  std::uint32_t visible_height = 0;
  std::uint32_t stride = 0;       // in samples; tightly packed
  std::size_t offset = 0;         // in samples from the frame base
  std::size_t sample_count = 0;
};

struct FrameGeometry {
  Format format;
  std::uint32_t mi_cols = 0;
  std::uint32_t mi_rows = 0;
  std::uint32_t coded_width = 0;   // MiCols * MI_SIZE
  std::uint32_t coded_height = 0;  // MiRows * MI_SIZE
  int num_planes = 0;
  int sub_x = 0;
  int sub_y = 0;
  PlaneGeometry planes[3];
  std::size_t total_samples = 0;
  std::size_t frame_bytes = 0;  // total_samples * sizeof(uint16_t)
};

// §5.18.4.4 compute_image_size().
std::uint32_t mi_cols(std::uint32_t width);
std::uint32_t mi_rows(std::uint32_t height);
std::uint32_t coded_luma_width(std::uint32_t width);
std::uint32_t coded_luma_height(std::uint32_t height);

// Nullopt when width/height are zero, contain an invalid enum value, or the
// coded size overflows the implementation.  12-bit is accepted (internal use).
std::optional<FrameGeometry> make_frame_geometry(const Format& format);

// ---------------------------------------------------------------------------
// Format validation / signalling
// ---------------------------------------------------------------------------

enum class FormatStatus {
  kOk = 0,
  kZeroDimension,
  kDimensionTooLarge,
  kInvalidBitDepth,
  kInvalidChromaFormat,
  kReservedBitDepth,  // valid internal samples, not representable in AV2 v1.0.0
};

struct FormatCheck {
  FormatStatus status = FormatStatus::kOk;
  std::string reason;

  bool ok() const { return status == FormatStatus::kOk; }
};

// Checks internal consistency and bitstream representability.  12-bit returns
// kReservedBitDepth; use `is_internally_supported()` if you only need buffers.
FormatCheck validate_format(const Format& format);

bool is_internally_supported(const Format& format);
bool is_signalable(const Format& format);

// Named profile from Annex A.2: 4:0:0/4:2:0 -> 0, 4:2:2 -> 3, 4:4:4 -> 4.
// Profile 31 (Configurable) is intentionally not used.  -1 when unavailable.
int profile_idc(const Format& format);

// ---------------------------------------------------------------------------
// Sample buffers
// ---------------------------------------------------------------------------

struct PlaneBuffer {
  std::uint32_t width = 0;          // coded width
  std::uint32_t height = 0;         // coded height
  std::uint32_t visible_width = 0;
  std::uint32_t visible_height = 0;
  std::uint32_t stride = 0;         // samples
  std::vector<std::uint16_t> samples;

  std::uint16_t& at(std::uint32_t x, std::uint32_t y);
  const std::uint16_t& at(std::uint32_t x, std::uint32_t y) const;
};

struct Picture {
  Format format;
  FrameGeometry geometry;
  PlaneBuffer planes[3];
};

// Allocates coded geometry and zero-fills.  Returns false with `error` on an
// invalid/unsupported format or allocation failure.
bool allocate_picture(const Format& format, Picture& picture, std::string& error);

// Reads a sample with coordinates clamped to the *visible* rectangle.  Returns
// 0 for a plane with no visible samples.  This is the edge-read rule used by
// prediction: never read outside the visible plane.
std::uint16_t sample_clamped(const PlaneBuffer& plane, int x, int y);

// Copies the last visible column/row into the coded padding so edge reads and
// GPU vector loads see defined samples.  Deterministic; corners stay corners.
void replicate_borders(PlaneBuffer& plane);

// ---------------------------------------------------------------------------
// Range / clipping
// ---------------------------------------------------------------------------

std::uint16_t clip_sample(std::int64_t value, BitDepth depth);

// 8-bit -> high depth by left shift (exact, no rounding).
std::uint16_t expand_8_to_depth(std::uint8_t value, BitDepth depth);

// High depth -> 8-bit with round-to-nearest and saturation to 255.
std::uint8_t reduce_to_8(std::uint16_t value, BitDepth depth);

// Up-conversion is a left shift; down-conversion rounds and saturates.
// 8->10->8 and 10->12->10 are exact; 12->10->12 is lossy as expected.
std::uint16_t convert_sample_depth(std::uint16_t value, BitDepth from, BitDepth to);

struct PlaneRange {
  std::uint32_t min_value = 0;
  std::uint32_t max_value = 0;
  std::size_t out_of_range = 0;
};

struct RangeReport {
  PlaneRange planes[3];
  std::size_t total_out_of_range = 0;
  int num_planes = 0;
};

// Checks every coded sample (including replicated padding) against
// [0, (1 << BitDepth) - 1].
RangeReport check_sample_range(const Picture& picture);

// Clips every sample and returns the number of changed samples.
std::size_t clamp_samples(Picture& picture);

bool samples_in_range(const Picture& picture);

// ---------------------------------------------------------------------------
// Conversion from / to the M0 8-bit 4:2:0 input
// ---------------------------------------------------------------------------

struct ConstPlane8 {
  const std::uint8_t* data = nullptr;
  std::uint32_t stride = 0;  // bytes
};

struct Source420_8 {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  ConstPlane8 y;
  ConstPlane8 u;
  ConstPlane8 v;
};

enum class ConvertStatus {
  kOk = 0,
  kBadSource,
  kUnsupportedFormat,
  kAllocationFailed,
};

struct ConvertResult {
  ConvertStatus status = ConvertStatus::kOk;
  std::string error;

  bool ok() const { return status == ConvertStatus::kOk; }
};

// Converts an 8-bit 4:2:0 source to 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 at 8/10/12-bit.
// Chroma upsampling replicates samples (no interpolation); luma is unchanged.
// Destination padding is filled by `replicate_borders`.
ConvertResult convert_420_8_to_picture(const Source420_8& source,
                                       ChromaFormat chroma, BitDepth depth,
                                       Picture& picture);

struct Plane8 {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t stride = 0;  // bytes
  std::vector<std::uint8_t> data;
};

struct Picture420_8 {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  Plane8 y;
  Plane8 u;
  Plane8 v;
};

// Converts any supported depth/chroma picture to 8-bit 4:2:0.
// Chroma uses nearest (top-left) sampling, not averaging.  This is the exact
// inverse of `convert_420_8_to_picture()` for a 4:2:0 source, which is what the
// round-trip tests pin.  Monochrome (4:0:0) is rejected: there is no chroma to
// synthesize.
ConvertResult convert_to_420_8(const Picture& picture, Picture420_8& out);

// ---------------------------------------------------------------------------
// Block rectangles and edge rules
// ---------------------------------------------------------------------------

struct BlockRect {
  std::uint32_t x = 0;              // in the plane's coded coordinate space
  std::uint32_t y = 0;
  std::uint32_t width = 0;          // nominal block size
  std::uint32_t height = 0;
  std::uint32_t visible_width = 0;  // clipped to the visible plane
  std::uint32_t visible_height = 0;
  bool inside = false;              // at least one visible sample
  bool partial = false;             // visible extent is smaller than nominal
};

// Plane 0 uses the luma block size; planes 1/2 use chroma_residual_size().
// Coordinates are luma coordinates (a multiple of 4).  Nullopt for a chroma
// residual size that Subsampled_Size marks invalid, or an out-of-range plane.
std::optional<BlockRect> block_rect(const FrameGeometry& geometry, int plane,
                                    std::uint32_t luma_x, std::uint32_t luma_y,
                                    BlockSize luma_size);

std::uint32_t visible_extent(std::uint32_t origin, std::uint32_t extent,
                             std::uint32_t visible_limit);

// ---------------------------------------------------------------------------
// Header / transform / entropy / filter requirements
// ---------------------------------------------------------------------------

struct FormatRequirements {
  // Header
  int chroma_format_idc = -1;
  int bit_depth_idc = -1;
  int profile_idc = -1;
  int max_q = 0;
  int base_q_idx_bits = 0;
  bool signalable = false;

  // Transform / reconstruction
  int dequant_clip_bits = 0;        // 7 + BitDepth
  int inverse_row_clip_bits = 0;    // BitDepth + 7
  int inverse_col_clip_bits = 0;    // BitDepth
  int dc_pred_bias = 0;             // 1 << (BitDepth - 1)
  int residual_clip_max = 0;        // (1 << BitDepth) - 1

  // Entropy
  int cdf_q_ctx_threshold[3] = {90, 140, 190};
  int cdf_q_ctx_count = 4;

  // Filters
  int cdef_coeff_shift = 0;         // BitDepth - 8
  int deblock_qindex_scale = 24;    // 24 * (BitDepth - 8) subtracted from qIdx
};

FormatRequirements format_requirements(const Format& format);

// ---------------------------------------------------------------------------
// Bit-depth-dependent arithmetic helpers
// ---------------------------------------------------------------------------
//
// These are the exact bounds a 10-bit reconstruction path must use.  They are
// collected here (instead of being repeated as `1 << 15` / `256` literals in
// the transform code) so the CPU reference, the CUDA kernels, and the
// integration checklist cannot drift apart.
//
//   §4.8       Clip1(x) = Clip3(0, (1 << BitDepth) - 1, x)
//   §7.14.3    Dequant = Clip3(-(1 << (7 + BitDepth)), (1 << (7 + BitDepth)) - 1, dq)
//   §7.15.2.1  row pass:    Clip3(-(1 << (BitDepth + 7)), (1 << (BitDepth + 7)) - 1, v)
//              column pass: Clip3(-(1 << BitDepth), (1 << BitDepth) - 1, v)
//   §7.13.2.10 DC prediction neutral value = 1 << (BitDepth - 1)
//
// `Transform_Shift` itself is bit-depth independent (§7.15.4.1); do not scale
// the shifts when moving from 8-bit to 10-bit.
struct BitDepthLimits {
  int bit_depth = 0;
  int sample_max = 0;        // (1 << BitDepth) - 1
  int dc_bias = 0;           // 1 << (BitDepth - 1)
  int dequant_min = 0;       // -(1 << (7 + BitDepth))
  int dequant_max = 0;       //  (1 << (7 + BitDepth)) - 1
  int inverse_row_min = 0;   // -(1 << (BitDepth + 7))
  int inverse_row_max = 0;   //  (1 << (BitDepth + 7)) - 1
  int inverse_col_min = 0;   // -(1 << BitDepth)
  int inverse_col_max = 0;   //  (1 << BitDepth) - 1
  // §7.14.4: the residual is representable as a signed integer with
  // 1 + BitDepth bits, i.e. -(1 << BitDepth) .. (1 << BitDepth) - 1.
  int residual_min = 0;
  int residual_max = 0;
};

// Zero/invalid depths return a zeroed struct with bit_depth == 0.
BitDepthLimits bit_depth_limits(BitDepth depth);

int clip1_sample(std::int64_t value, BitDepth depth);
int clip_dequant(std::int64_t value, BitDepth depth);
int clip_inverse_row(std::int64_t value, BitDepth depth);
int clip_inverse_col(std::int64_t value, BitDepth depth);

// §8.3.1 init_coeff_cdfs(): index 0 if base_q_idx <= 90, then 140, 190, else 3.
// Deliberately uses the raw base_q_idx; it is not normalized by bit depth.
int cdf_q_context(int base_q_idx);

enum class ChecklistArea { kHeader, kTransform, kEntropy, kFilter };

struct ChecklistItem {
  ChecklistArea area = ChecklistArea::kHeader;
  std::string id;
  std::string spec_section;
  std::string requirement;
};

// Human-readable per-format integration checklist.  This is documentation as
// data: tests assert the key values; the README explains the integration work.
std::vector<ChecklistItem> format_checklist(const Format& format);

}  // namespace av2enc::fmt
