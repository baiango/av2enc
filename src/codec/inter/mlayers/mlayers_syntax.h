// Byte-level AV2 v1.0.0 syntax for embedded layers.
//
// Writer + independent reader for:
//   §5.2.2 p54   obu_header(): obu_header_extension_flag, obu_mlayer_id,
//                obu_xlayer_id, GLOBAL_XLAYER_ID inference for TD/MSDO
//   §5.4.1 p55-58 sequence_header_obu():
//                max_mlayer_id, seq_max_mlayer_cnt_minus_1 (CeilLog2),
//                mlayer_dependency_present_flag/map,
//                tlayer_dependency_present_flag/map,
//                multi_tlayer_dependency_map_present_flag,
//                effective MLayerDependencyMap / TLayerDependencyMap,
//                derived MLayerPresenceMap
//
// The sequence writer emits the M2-compatible baseline tool set (profile 0,
// level 5.1, 8-bit 4:2:0, one tile, no advanced tools) plus the layer fields.
// For max_mlayer_id == max_tlayer_id == 0 it is byte-identical to
// src/encoder/m2/video_headers.cpp, which the unit test pins.
//
// SequenceSyntaxDialect::kPinnedDav2dCompat is an explicit, non-conforming
// escape hatch for the pinned dav2d f9dafbc decoder. That decoder does not
// parse seq_max_mlayer_cnt_minus_1 and reads the layer-dependency blocks in a
// different order/syntax. The compat dialect omits the count field and uses
// dav2d's condensed map rows. It is used only by mlayers_gate.cpp and is
// documented in README.md; kSpecV1_0_0 is the normative writer.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/inter/mlayers/mlayers.h"

namespace av2enc::inter::mlayers {

// OBU types from §6.2.2 Table 6.1 that matter to this layer.
enum class ObuType : std::uint8_t {
  kReserved = 0,
  kSequenceHeader = 1,
  kTemporalDelimiter = 2,
  kMultiFrameHeader = 3,
  kClosedLoopKey = 4,
  kOpenLoopKey = 5,
  kLeadingTileGroup = 6,
  kRegularTileGroup = 7,
  kMetadataShort = 8,
  kMetadataGroup = 9,
  kSwitch = 10,
  kLeadingSef = 11,
  kRegularSef = 12,
  kLeadingTip = 13,
  kRegularTip = 14,
  kBufferRemovalTiming = 15,
  kLayerConfigurationRecord = 16,
  kAtlasSegment = 17,
  kOperatingPointSet = 18,
  kBridgeFrame = 19,
  kMsdo = 20,
  kRasFrame = 21,
  kQuantizationMatrix = 22,
  kFilmGrain = 23,
  kContentInterpretation = 24,
  kPadding = 25,
};

struct ObuHeaderFields {
  bool extension_flag = false;
  std::uint8_t obu_type = 0;
  std::uint8_t tlayer_id = 0;
  std::uint8_t mlayer_id = 0;  // inferred 0 when extension_flag == 0
  std::uint8_t xlayer_id = 0;  // inferred (§5.2.2) when extension_flag == 0
};

// §5.2.2 inference when obu_header_extension_flag == 0:
//   obu_mlayer_id = 0
//   obu_xlayer_id = GLOBAL_XLAYER_ID for TD/MSDO, else 0
ObuHeaderFields inferred_obu_header_fields(std::uint8_t obu_type,
                                           std::uint8_t tlayer_id);

// §6.2.2 p257 layer-id constraints (GLOBAL_XLAYER_ID scope, forced zero
// mlayer/tlayer for non-layer-specific OBUs, forced tlayer 0 for key frames).
bool validate_obu_header_fields(const ObuHeaderFields& fields,
                                std::string& error);

// Writes the 1- or 2-byte obu_header(). The caller supplies extension_flag and
// the effective ids; use make_obu_header() for automatic inference.
std::vector<std::uint8_t> make_obu_header_bytes(const ObuHeaderFields& fields);

// Chooses extension_flag = 1 only when the requested ids differ from the
// §5.2.2 inference with extension_flag = 0.
bool make_obu_header(std::uint8_t obu_type, std::uint8_t tlayer_id,
                     std::uint8_t mlayer_id, std::uint8_t xlayer_id,
                     std::vector<std::uint8_t>& out, std::string& error);

// Parses one obu_header() at `offset` (after the leb128 length prefix).
bool parse_obu_header(std::span<const std::uint8_t> bytes, std::size_t offset,
                      ObuHeaderFields& fields, std::size_t& header_bytes,
                      std::string& error);

enum class SequenceSyntaxDialect {
  kSpecV1_0_0,
  kPinnedDav2dCompat,
};

struct SequenceHeaderConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // 7 = level 5.1
  std::uint8_t base_q_idx = 90;
  std::uint8_t order_hint_bits = 4;
  std::uint8_t num_ref_frames = 2;
  std::uint8_t max_drl_bits_minus_1 = 4;
  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;
  bool monotonic_output_order_flag = true;
  SequenceLayerConfig layers;
};

SequenceHeaderConfig baseline_sequence_header_config(std::uint16_t width,
                                                     std::uint16_t height,
                                                     std::uint8_t base_q_idx,
                                                     std::uint8_t num_ref_frames);

// Writes sequence_header_obu() payload bits (no OBU header, no trailing bits).
// `bit_count` is the exact number of syntax bits; the final byte is zero-padded
// but the caller must use bit_count when appending the extensible-OBU tail.
bool write_sequence_header_payload(const SequenceHeaderConfig& config,
                                   SequenceSyntaxDialect dialect,
                                   std::vector<std::uint8_t>& payload,
                                   std::size_t& bit_count,
                                   std::string& error);

// Annex B.2 OBU: leb128(size) + obu_header() + payload + obu_extension_flag(0)
// + trailing_bits(). The OBU header has extension_flag = 0 (sequence headers
// are forced to mlayer 0 / xlayer 0 by §6.2.2).
bool make_sequence_header_obu(const SequenceHeaderConfig& config,
                              SequenceSyntaxDialect dialect,
                              std::vector<std::uint8_t>& out,
                              std::string& error);

// Independent reader. Parses the OBU envelope, the layer fields and the fixed
// baseline tail, then verifies the extensible-OBU tail (flag 0 + trailing
// bits). `out.dependencies` is the effective map (defaults + coded overrides),
// and `out.presence` is the §5.4.1 derived presence map.
struct ParsedSequenceHeader {
  int seq_header_id = 0;
  int seq_profile_idc = 0;
  bool single_picture_header_flag = false;
  int seq_level_idx = 0;
  int chroma_format_idc = 0;
  int bit_depth_idc = 1;
  int seq_lcr_id = 0;
  bool still_picture = false;
  int max_tlayer_id = 0;
  int max_mlayer_id = 0;
  int seq_max_mlayer_cnt = 1;
  bool seq_max_mlayer_cnt_coded = false;
  bool monotonic_output_order_flag = true;
  int max_frame_width = 0;
  int max_frame_height = 0;
  bool mlayer_dependency_present_flag = false;
  bool tlayer_dependency_present_flag = false;
  bool multi_tlayer_dependency_map_present_flag = false;
  LayerDependencyMaps dependencies;
  std::array<std::array<bool, kMlayerCount>, kMlayerCount>
      mlayer_dependency_coded{};
  std::array<std::array<std::array<bool, kTlayerCount>, kTlayerCount>,
             kMlayerCount>
      tlayer_dependency_coded{};
  MLayerPresenceMap presence{};
  int order_hint_bits = 0;
  int num_ref_frames = 0;
  SequenceSyntaxDialect dialect = SequenceSyntaxDialect::kSpecV1_0_0;
};

bool parse_sequence_header_obu(std::span<const std::uint8_t> obu,
                               SequenceSyntaxDialect dialect,
                               ParsedSequenceHeader& out, std::string& error);

// Reads only the OBU envelope/type/layer ids of one Annex B.2 OBU.
struct ParsedObu {
  ObuHeaderFields header;
  std::size_t total_bytes = 0;  // leb128 prefix + header + payload
  std::span<const std::uint8_t> payload;
};

bool parse_obu(std::span<const std::uint8_t> bytes, std::size_t offset,
               ParsedObu& out, std::string& error);

// Annex B.2 helper: leb128(size) + header + payload.
std::vector<std::uint8_t> make_obu(std::uint8_t obu_type,
                                   std::uint8_t tlayer_id,
                                   std::uint8_t mlayer_id,
                                   std::uint8_t xlayer_id,
                                   std::span<const std::uint8_t> payload,
                                   bool* extension_flag = nullptr);

// §5.18.2 frame_header_info() writer for a CLK frame in the multi-embedded-
// layer case. It mirrors the M2 key-frame writer and inserts
// refresh_frame_flags f(NumRefFrames), which §5.18.2 codes when
// max_mlayer_id > 0. Used by the gate and by the reader below.
struct KeyFrameHeader {
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 3;
  bool immediate_output_frame = true;
};

bool write_key_frame_header(const SequenceHeaderConfig& config,
                            const KeyFrameHeader& header,
                            std::vector<std::uint8_t>& header_bits,
                            std::size_t& bit_count, std::string& error);

}  // namespace av2enc::inter::mlayers
