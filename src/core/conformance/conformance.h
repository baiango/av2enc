// Encoder-side Annex A conformance validator.
//
// Spec-only: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), Annex A.2/A.3/A.4/A.5 (p1101-1111).
//
// This module answers one question per audit row: does the sequence
// configuration and the measured stream stay inside the envelope of the
// profile/level/tier that the encoder signals?
//
//   * It emits no bits and changes no encoder behaviour.
//   * It is default-free: every constraint is evaluated against the supplied
//     Config and StreamStats, never against a compile-time target.
//   * It returns ALL violations, not the first one:
//       for (const Violation& v : validate(config, stats).violations)
//   * Every Violation carries a stable check id and the §/page that states the
//     constraint (e.g. "A.4.MaxPicSize", "A.4 (p1110)").
//   * Checks that cannot be evaluated because the required statistic is not
//     supplied are listed in Result::not_evaluated. They are not silently
//     treated as passes.
//
// Deliberate non-goals: this is not a decoder-model simulator and not a
// replacement for a dav2d decode. Annex E conformance needs BRT/timing data
// that today's writers do not emit; `E.4.4.LevelConformanceUnverifiable` makes
// that absence visible instead of pretending the stream is verified.
//
// Stage 1 (this file): sequence-level constraints that can be checked from the
// header config plus measured frame statistics. Stage 2 (stream_stats.frames)
// adds the per-frame decode budget. No encoder path calls this yet; see
// README.md for the exact call sites.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/conformance/conformance_tables.h"

namespace av2enc::conformance {

// ---------------------------------------------------------------------------
// Tile geometry
// ---------------------------------------------------------------------------

// Decoder-side tile geometry for the tile-level constraints. Widths/heights
// are §5.18.7.3 TileWidth/TileHeight:
//
//   TileWidth  = (MiColEnd - MiColStart) * MI_SIZE
//   TileHeight = (MiRowEnd - MiRowStart) * MI_SIZE
//
// For an interior edge this is a whole number of superblocks. The final
// column/row is clipped at MiCols/MiRows, exactly as the decoder derives it.
// `uniform()` reproduces §5.18.7.5 uniform_spacing(); explicit widths use the
// same representation so a future non-uniform writer is covered.
struct TileGrid {
  int cols = 1;
  int rows = 1;
  // Raster order: index = row * cols + col.
  std::vector<int> tile_widths;
  std::vector<int> tile_heights;
  // False when uniform() could not represent the requested grid (for example
  // 16 columns over a frame with only 10 superblock columns). The tile vectors
  // are then empty.
  bool representable = true;

  static TileGrid uniform(int width, int height, int cols, int rows,
                          int sb_size = 64);

  int64_t tile_count() const;
  int64_t max_tile_area() const;
  bool structurally_valid(std::string* why = nullptr) const;
};

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

// Sequence-level values that are coded in (or derived from) the sequence
// header. Field names track the spec syntax elements.
struct Config {
  // §5.18.1 max_frame_width/height and §5.4.1 frame size.
  int width = 0;
  int height = 0;
  int sb_size = 64;  // 64 or 128

  // §5.4.1 sequence header.
  int seq_profile_idc = 0;
  int seq_level_idx = 7;  // 7 = 5.1
  int seq_tier = 0;       // 0 = Main, 1 = High
  int chroma_format_idc = 0;  // 0 = 4:2:0, 1 = 4:0:0, 2 = 4:4:4, 3 = 4:2:2
  int bit_depth_idc = 1;      // 0 = 10-bit, 1 = 8-bit (§6.4.1, inverted)
  bool still_picture = false;
  // single_picture_header_flag == 1 removes seq_tier (p1104) and infers
  // NumRefFrames = 2 and SeqMaxMlayerCnt = 1.
  bool single_picture_header_flag = false;

  // §5.4.6 NumRefFrames. For the M2 writer this is num_ref_frames_minus_1+1
  // with explicit_num_ref_frames == 1. When explicit_num_ref_frames is false
  // the decoder infers 8 and `num_ref_frames` is ignored by the validator.
  int num_ref_frames = 2;
  bool explicit_num_ref_frames = false;
  int max_mlayer_id = 0;
  int seq_max_mlayer_cnt_minus_1 = 0;

  // §A.3 multi-stream structure. A single coded video sequence has
  // num_extended_layers = num_embedded_layers = 1.
  int num_extended_layers = 1;
  int num_embedded_layers = 1;
  int interop_point = -1;  // -1: derive from seq_profile_idc (Table A.1)
  bool msdo_present = false;
  bool global_lcr_present = false;
  bool local_lcr_present = false;
  int lcr_config_idc = -1;  // -1: no LCR

  // Tools that feed DecodeCount / InloopFilteringEnabled (p1108).
  bool allow_global_intrabc = false;
  bool inloop_filtering_enabled = true;

  // §A.3 signalling used only to report E.4.4 verifiability.
  bool ci_timing_info_present = false;
  bool decoder_model_info_present = false;

  // Tile plan. Defaults to one tile.
  TileGrid tiles{};
};

// One frame's measured statistics. Values already known from the header are
// left at 0 and derived from Config.
struct FrameSample {
  // 0 -> Config.width * Config.height (intra) or the sequence maximum (inter).
  int64_t luma_sample_count = 0;
  // Total OBU bytes belonging to this frame's DFG, before the spec's "-128"
  // overhead allowance. 0 means "not measured".
  int64_t frame_obu_bytes = 0;
  // FrameSymbolCount (read_literal + read_symbol accumulations). 0 = unknown.
  int64_t symbol_count = 0;
  // FrameParsingTime = (Removal[i+1] - Removal[i]) / DecodeCount. 0 -> derive
  // from StreamStats::fps and decode_count_2.
  double frame_parsing_time_seconds = 0.0;
  bool show_existing_frame = false;
  // DecodeCount == 2: allow_global_intrabc == 1 && InloopFilteringEnabled == 1,
  // for a frame that is not ShowExistingFrame.
  bool decode_count_2 = false;
  bool closed_loop_key = false;
  // 0 -> Config.tiles.max_tile_area().
  int64_t max_tile_size_luma_samples = 0;
  // immediate_output_frame == 1 || implicit_output_frame == 1 ||
  // ShowExistingFrame == 1.
  bool output_frame = true;
};

// Worst-case measurements. All "0/unknown" fields leave the corresponding
// check out of the result and add a note to Result::not_evaluated.
struct StreamStats {
  // Display/output rate of the stream, frames per second. Used for
  // TotalDisplayLumaSampleRate, NumFrameHeadersPerSec and FrameParsingTime
  // fallbacks.
  double fps = 0.0;
  // Worst temporal unit: sum of LumaSampleCount of output frames, and its
  // output duration in seconds.
  int64_t tu_display_luma_samples = 0;
  double tu_duration_seconds = 0.0;
  // Number of output frames in that temporal unit (fallback for the sum).
  int output_frames_in_tu = 1;
  // Busiest 1 s window of OBUs carrying a frame_header() with
  // CountFrameHeaderForLevelConstraint == 1.
  int max_frame_headers_per_second = 0;
  // NumRefFrames actually used; 0 -> Config::num_ref_frames (or inferred 8).
  int num_ref_frames = 0;
  // Measured over the busiest 1 s window. 0 = not measured.
  double max_bitrate_bps_over_1s = 0.0;
  // Peak smoothing-buffer occupancy. 0 = not measured.
  double peak_buffer_occupancy_bits = 0.0;
  bool timing_info_signalled = false;
  bool decoder_model_info_signalled = false;
  std::vector<FrameSample> frames;
};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

enum class Severity {
  kError,    // normative "shall" violated
  kWarning,  // conformance claim cannot be supported / needs a caveat
};

struct Violation {
  std::string check;    // stable id, "A.4.MaxPicSize"
  std::string section;  // "A.4"
  int page = 0;         // spec page
  Severity severity = Severity::kError;
  std::string detail;   // observed vs allowed, human readable
};

struct Result {
  std::vector<Violation> violations;
  std::vector<std::string> not_evaluated;

  bool conformant() const;  // no kError
  bool has_check(std::string_view check) const;
  int error_count() const;
  int warning_count() const;
  std::string summary() const;
};

Result validate(const Config& config, const StreamStats& stats = {});

// ---------------------------------------------------------------------------
// Resolved level (also useful to callers building a plan)
// ---------------------------------------------------------------------------

struct ResolvedLevel {
  bool valid = false;
  bool maximum_parameters = false;
  bool profile_scaling_known = false;
  int seq_level_idx = 0;
  int seq_tier = 0;
  const LevelInfo* info = nullptr;
  int width_scaling_factor = 0;
  int area_scaling_factor = 0;
  int64_t max_tile_width_luma = 0;  // WSF * MAX_TILE_WIDTH / 4
  int64_t max_tile_area_luma = 0;   // ASF * MAX_TILE_AREA / 4
  int64_t max_num_frame_headers_per_sec = 0;
  int64_t max_tiles = 0;
  int64_t max_tile_cols = 0;
  double min_comp_basis = 0.0;
  double max_bitrate_bps = 0.0;
  double max_buffer_size_bits = 0.0;
  int max_level_ref_frames = 0;
};

ResolvedLevel resolve_level(const Config& config);

// Individual limit helpers. They are the same code the validator uses, exposed
// so callers and tests can construct exact boundary values. Return -1 (or 0
// for the double helpers) when the constraint is not evaluable.
int64_t max_level_ref_frames(const Config& config,
                             bool decode_count_2_lowers_limit = false);
int64_t max_display_luma_samples_per_second(const Config& config);
int64_t max_luma_samples_for_parsing_time(const Config& config,
                                          double frame_parsing_time);
int64_t max_tiles_for_parsing_time(const Config& config,
                                   double frame_parsing_time);
double max_compressed_size_bytes(const Config& config, int64_t luma_samples,
                                 double frame_parsing_time);
double max_frame_symbol_count(const Config& config,
                              double frame_parsing_time);
// Largest MaxTileSizeInLumaSamples that still satisfies the parallelism
// constraint at this header rate. 0 when not evaluable.
int64_t max_tile_size_luma_samples(const Config& config,
                                   int64_t frame_headers_per_second);
// FrameParsingTime fallback when the caller has no BRT: 1/fps divided by
// DecodeCount. 0 for a non-positive fps.
double frame_parsing_time_from_fps(double fps, bool decode_count_2);

// ---------------------------------------------------------------------------
// Audit coverage
// ---------------------------------------------------------------------------

// One row for each ABSENT constraint in docs/spec-audit/14-conformance.md.
// `disposition` is one of:
//   "check:<id>"       implemented by validate()
//   "N/A: <reason>"    no encoder emission path can violate it today
//   "timing: <reason>" reachable only once Annex E timing is emitted
struct CoverageEntry {
  int audit_row = 0;
  const char* area = "";
  const char* constraint = "";
  const char* spec_ref = "";
  const char* disposition = "";
};

const std::vector<CoverageEntry>& absent_constraint_coverage();

}  // namespace av2enc::conformance
