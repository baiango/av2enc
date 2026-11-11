// AV2 v1.0.0 sequence-header feature syntax: cropping, PB aspect ratio,
// deblocking-filter parallel bits, sequence tile info and sequence segments.
//
// Spec-only. Normative reference:
//   AV2 Bitstream & Decoding Process Specification v1.0.0
//   (2026-05-28, commit 38f28e7)
//
// Sections owned here:
//   §5.4.1  seq_cropping_window_present_flag + four uvlc() offsets
//   §5.4.2  sequence_tile_config(); §5.18.7.3 tile_params();
//           §5.18.7.6 get_seq_sb_size(); §5.18.7.4 reuse_tile_params()
//   §5.4.3  reduce_pb_aspect_ratio / max_pb_aspect_ratio_log2_minus_1
//           (decoder rule: §6.19.4.1, §6.19.3.2 is_partition_allowed)
//   §5.4.4  sequence_segment_config()
//   §5.4.9  seg_info(MaxSegments)
//   §5.4.10 df_par_bits_minus_2 (used by §5.18.5.2)
//
// This module deliberately owns no CLI and does not edit the M0/M1/M2 core
// writers. The core writers call the small `write_*` functions at the exact
// points listed in README.md; the matching `read_*` functions are independent
// parsers and never call back into the writer.
#pragma once

#include "bitstream/bitwriter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::syntax::small_header {

// ---------------------------------------------------------------------------
// Bit reader (inverse of BitWriter for the raw descriptors used below)
// ---------------------------------------------------------------------------
//
// MSB-first, exactly like BitWriter. Only f(n), uvlc(), ns(n) and su(n) are
// needed by these syntax structures; there is no entropy state here.
class BitReader {
 public:
  BitReader() = default;
  explicit BitReader(std::span<const std::uint8_t> bytes,
                     std::size_t bit_size = 0)
      : bytes_(bytes), bit_size_(bit_size == 0 ? bytes.size() * 8U : bit_size) {}

  std::size_t bit_pos() const { return bit_pos_; }
  std::size_t bit_size() const { return bit_size_; }
  std::size_t bits_left() const {
    return bit_pos_ <= bit_size_ ? bit_size_ - bit_pos_ : 0U;
  }
  bool failed() const { return failed_; }

  bool read_bit(bool* bit);
  bool read_bits(unsigned num_bits, std::uint32_t* value);
  bool read_uvlc(std::uint32_t* value);
  bool read_ns(unsigned n, std::uint32_t* value);
  bool read_su(unsigned n, std::int32_t* value);
  bool align_to_byte();

 private:
  std::span<const std::uint8_t> bytes_{};
  std::size_t bit_size_ = 0;
  std::size_t bit_pos_ = 0;
  bool failed_ = false;
};

// ---------------------------------------------------------------------------
// §5.4.1 Cropping window
// ---------------------------------------------------------------------------

struct CroppingWindow {
  bool present = false;
  // uvlc() values, in pixels at maximum frame size (§5.18.4.4 scales them).
  std::uint32_t left = 0;
  std::uint32_t right = 0;
  std::uint32_t top = 0;
  std::uint32_t bottom = 0;
};

struct DisplaySize {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

// Offset limits are max_frame_width_minus_1 / max_frame_height_minus_1. Also
// rejects a window that derives an empty crop rectangle at maximum frame size.
bool validate_cropping_window(const CroppingWindow& window,
                              std::uint32_t max_frame_width,
                              std::uint32_t max_frame_height,
                              std::string* error);

bool write_cropping_window(BitWriter& writer, const CroppingWindow& window,
                           std::string* error);

bool read_cropping_window(BitReader& reader, CroppingWindow* window,
                          std::string* error);

// §5.18.4.4 compute_image_size(): CropLeft/CropRight/CropTop/CropBottom and
// CropWidth/CropHeight. `max_*` are max_frame_width_minus_1 + 1; `frame_*` are
// the actual FrameWidth/FrameHeight (they may be smaller than the maximum).
bool derive_display_size(const CroppingWindow& window,
                         std::uint32_t max_frame_width,
                         std::uint32_t max_frame_height,
                         std::uint32_t frame_width,
                         std::uint32_t frame_height, DisplaySize* size,
                         std::string* error);

// ---------------------------------------------------------------------------
// §5.4.3 Prediction-block aspect ratio
// ---------------------------------------------------------------------------

struct PbAspectRatio {
  bool reduce = false;  // reduce_pb_aspect_ratio
  // Only coded when reduce == true. 0 -> 2:1, 1 -> 4:1.
  std::uint8_t max_pb_aspect_ratio_log2_minus_1 = 0;
};

// MaxPbAspectRatio: 8 when not reduced, else 1 << (log2_minus_1 + 1).
int max_pb_aspect_ratio(const PbAspectRatio& config);

bool validate_pb_aspect_ratio(const PbAspectRatio& config,
                              std::string* error);

bool write_pb_aspect_ratio(BitWriter& writer, const PbAspectRatio& config,
                           std::string* error);

bool read_pb_aspect_ratio(BitReader& reader, PbAspectRatio* config,
                          std::string* error);

// §6.19.4.1 conformance: bw4 <= bh4 * MaxPbAspectRatio and
// bh4 <= bw4 * MaxPbAspectRatio for every decoded leaf block.
bool pb_leaf_aspect_allowed(int bw4, int bh4, int max_ratio);

// §6.19.3.2 is_partition_allowed() aspect-ratio fragment. `bw`/`bh` are the
// candidate subsize in 4x4 units.
//   * too thin => PARTITION_NONE forbidden;
//   * ratio >= 8:1 => no partition at all (returns false from the fragment);
//   * otherwise the non-NONE partitions remain eligible.
bool pb_partition_none_allowed_by_aspect(int bw4, int bh4, int max_ratio);
bool pb_non_none_partition_allowed_by_aspect(int bw4, int bh4, int max_ratio);

// ---------------------------------------------------------------------------
// §5.4.10 df_par_bits_minus_2
// ---------------------------------------------------------------------------

struct DfPar {
  // f(2): dfParBits = df_par_bits_minus_2 + 2, so the coded df_delta_q has
  // 2..5 bits and DfDeltaQ lies in [-(1 << (bits-1)), (1 << (bits-1)) - 1].
  std::uint8_t df_par_bits_minus_2 = 0;

  int bits() const { return static_cast<int>(df_par_bits_minus_2) + 2; }
  int min_value() const { return -(1 << (bits() - 1)); }
  int max_value() const { return (1 << (bits() - 1)) - 1; }
};

bool validate_df_par(const DfPar& config, std::string* error);

bool write_df_par(BitWriter& writer, const DfPar& config, std::string* error);

bool read_df_par(BitReader& reader, DfPar* config, std::string* error);

bool df_delta_q_representable(int value, const DfPar& config);

// The four §5.18.5.2 DfDeltaQ values. When present[i] is false, DfDeltaQ[i] is
// 0 except DfDeltaQ[1], which is inferred equal to DfDeltaQ[0]. The encoder
// rule is that every *coded* value must fit dfParBits; derived values are
// implicitly representable through the coded value they copy.
struct DfDeltaQSet {
  std::array<bool, 4> present = {false, false, false, false};
  std::array<int, 4> value = {0, 0, 0, 0};
};

bool validate_df_delta_q(const DfPar& config, const DfDeltaQSet& values,
                         std::string* error);

// ---------------------------------------------------------------------------
// §5.4.4 / §5.4.9 Sequence segments
// ---------------------------------------------------------------------------

inline constexpr int kMaxSegments = 16;  // MAX_SEGMENTS
inline constexpr int kBaseSegments = 8;  // enable_ext_seg == 0
inline constexpr int kSegLvlMax = 3;     // SEG_LVL_MAX
inline constexpr int kSegFeatureBits = 9;      // Segmentation_Feature_Bits[0]
inline constexpr int kSegFeatureMax = 351;     // MAXQ_BITS

struct SegmentInfo {
  std::array<std::array<bool, kSegLvlMax>, kMaxSegments> enabled{};
  std::array<std::array<int, kSegLvlMax>, kMaxSegments> value{};
};

bool operator==(const SegmentInfo& a, const SegmentInfo& b);

struct SequenceSegmentConfig {
  bool enable_ext_seg = false;  // 1 -> MaxSegments = 16, else 8
  bool present = false;         // seq_seg_info_present_flag
  bool allow_change = false;    // seq_allow_seg_info_change
  SegmentInfo info;             // valid when present

  int max_segments() const { return enable_ext_seg ? kMaxSegments : kBaseSegments; }
};

bool validate_segment_info(const SegmentInfo& info, int num_segments,
                           std::string* error);

// seg_info(numSegments) exactly as §5.4.9. For SEG_LVL_ALT_Q the value is
// su(1 + 9) and clipped to [-MAXQ_BITS, MAXQ_BITS] by the decoder; the writer
// rejects out-of-range values rather than relying on the clip. SEG_LVL_SKIP and
// SEG_LVL_GLOBALMV have zero value bits.
bool write_segment_info(BitWriter& writer, const SegmentInfo& info,
                        int num_segments, std::string* error);

bool read_segment_info(BitReader& reader, int num_segments, SegmentInfo* info,
                       std::string* error);

bool validate_sequence_segment_config(const SequenceSegmentConfig& config,
                                      std::string* error);

bool write_sequence_segment_config(BitWriter& writer,
                                   const SequenceSegmentConfig& config,
                                   std::string* error);

bool read_sequence_segment_config(BitReader& reader,
                                  SequenceSegmentConfig* config,
                                  std::string* error);

// §5.18.7.1 segmentation_params() persistence/change rules. The sequence-level
// state is `config.info`; a frame may override only when
// seq_allow_seg_info_change == 1. When the sequence did not provide segment
// info, the frame must code seg_info(MaxSegments) itself.
struct FrameSegmentRequest {
  bool enabled = false;
  // Consulted only when config.present && config.allow_change; ignored
  // otherwise. false means "code a fresh seg_info()".
  bool reuse_sequence = true;
  SegmentInfo info;  // used when reuse_sequence == false
  // Consulted only when the syntax actually codes map-update bits.
  bool update_map = true;
  bool temporal_update = false;
};

struct FrameSegmentState {
  bool have_sequence_params = false;
  bool reused_sequence = false;
  SegmentInfo info;
  bool update_map = true;
  bool temporal_update = false;
};

// `derived_primary_ref_none` is PRIMARY_REF_NONE for key/intra frames, which
// infers segmentation_update_map = 1 and codes no map-update bits.
// `frame_type_switch` is FrameType == SWITCH_FRAME.
bool write_frame_segmentation_params(BitWriter& writer,
                                     const SequenceSegmentConfig& config,
                                     const FrameSegmentRequest& request,
                                     bool derived_primary_ref_none,
                                     bool frame_type_switch,
                                     FrameSegmentState* state,
                                     std::string* error);

bool read_frame_segmentation_params(BitReader& reader,
                                    const SequenceSegmentConfig& config,
                                    bool derived_primary_ref_none,
                                    bool frame_type_switch,
                                    FrameSegmentState* state,
                                    std::string* error);

// ---------------------------------------------------------------------------
// §5.4.2 sequence_tile_config() and §5.18.7.2 tile_info() reuse
// ---------------------------------------------------------------------------

enum class SbSize : int {
  k64x64 = 64,
  k128x128 = 128,
  k256x256 = 256,
};

bool valid_sb_size(SbSize size);
int sb_width_4x4(SbSize size);   // Num_4x4_Blocks_Wide[BLOCK_*]
int sb_shift(SbSize size);       // Mi_Width_Log2[BLOCK_*]
const char* sb_size_name(SbSize size);

// §5.18.7.6 get_seq_sb_size().
SbSize sequence_superblock_size(bool use_256x256_superblock,
                                bool use_128x128_superblock);

// §5.18.7.7 tile_log2().
int tile_log2(int block_size, int target);

// §5.18.7.3 tile_params() result. Starts are in superblock units; the final
// tile extends to sb_cols / sb_rows.
struct TileParams {
  bool uniform_spacing = true;
  int frame_width = 0;
  int frame_height = 0;
  SbSize uniform_sb_size = SbSize::k64x64;
  SbSize sb_size = SbSize::k64x64;
  int sb_cols = 0;
  int sb_rows = 0;
  int tile_cols = 0;
  int tile_rows = 0;
  int tile_cols_log2 = 0;
  int tile_rows_log2 = 0;
  std::vector<int> sb_col_starts;
  std::vector<int> sb_row_starts;
  int sb_shift = 0;
  // Limits derived from seq_level_idx / seq_tier, exposed for diagnostics.
  int max_tile_width_sb = 0;
  int max_tile_area_sb = 0;
  int min_log2_tile_cols = 0;
  int min_log2_tiles = 0;
  int max_log2_tile_cols = 0;
  int max_log2_tile_rows = 0;
  std::size_t syntax_bits = 0;
};

// Writer-side request. `tile_cols_log2`/`tile_rows_log2` of -1 select the
// level-minimum tile grid. Non-uniform spacing is selected by non-empty
// width/height lists (each entry is in superblocks and must cover the frame).
struct TileParamsRequest {
  int tile_cols_log2 = -1;
  int tile_rows_log2 = -1;
  std::vector<int> width_in_sbs;
  std::vector<int> height_in_sbs;
};

bool write_tile_params(BitWriter& writer, int frame_width, int frame_height,
                       SbSize uniform_sb_size, SbSize sb_size,
                       int seq_level_idx, int seq_tier,
                       const TileParamsRequest& request, TileParams* params,
                       std::string* error);

bool read_tile_params(BitReader& reader, int frame_width, int frame_height,
                      SbSize uniform_sb_size, SbSize sb_size,
                      int seq_level_idx, int seq_tier, TileParams* params,
                      std::string* error);

struct SequenceTileConfig {
  bool present = false;       // seq_tile_info_present_flag
  bool allow_change = false;  // allow_tile_info_change
  TileParamsRequest request;  // writer input
  TileParams params;          // writer output / reader output
};

bool validate_sequence_tile_config(const SequenceTileConfig& config,
                                   int max_frame_width, int max_frame_height,
                                   SbSize sb_size, int seq_level_idx,
                                   int seq_tier, std::string* error);

bool write_sequence_tile_config(BitWriter& writer,
                                SequenceTileConfig* config,
                                int max_frame_width, int max_frame_height,
                                SbSize sb_size, int seq_level_idx,
                                int seq_tier, std::string* error);

bool read_sequence_tile_config(BitReader& reader,
                               SequenceTileConfig* config,
                               int max_frame_width, int max_frame_height,
                               SbSize sb_size, int seq_level_idx,
                               int seq_tier, std::string* error);

// §5.18.7.3 uniform_eligible().
bool uniform_eligible(int tile_log2_value, int sb_num);

struct FrameTileRequest {
  // Only consulted when the sequence tile params are eligible and
  // allow_tile_info_change == 1. true codes reuse_tile_info = 1.
  bool reuse_sequence = true;
  TileParamsRequest params;
};

struct FrameTileInfo {
  bool reused_sequence = false;
  TileParams params;
  std::size_t syntax_bits = 0;
};

// Writes the tile_info() body that a frame header needs *after* accounting for
// sequence-level tile parameters. When seq present + eligible + no change
// allowed, this writes nothing (the decoder infers reuse).
bool write_frame_tile_info(BitWriter& writer,
                           const SequenceTileConfig& sequence,
                           int frame_width, int frame_height, SbSize sb_size,
                           int seq_level_idx, int seq_tier,
                           const FrameTileRequest& request,
                           FrameTileInfo* info, std::string* error);

bool read_frame_tile_info(BitReader& reader,
                          const SequenceTileConfig& sequence,
                          int frame_width, int frame_height, SbSize sb_size,
                          int seq_level_idx, int seq_tier, FrameTileInfo* info,
                          std::string* error);

}  // namespace av2enc::syntax::small_header
