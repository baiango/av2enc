// OBU/layer framing and layer extraction for the SVC pipeline.
//
// Implements the byte-level pieces the encoder needs to emit a layered stream
// without touching the core OBU writer:
//
//   * §5.2.2 obu_header() with obu_tlayer_id / obu_mlayer_id / obu_xlayer_id
//   * Annex B.2 leb128 length prefix
//   * §5.8 layer_config_record_obu payload (global and local form)
//   * §5.10 / §5.11 operating_point_set_obu payload
//   * §7.3.6 / §7.3.7 layer/temporal-unit ordering at the compose step
//   * Annex F.3.2 sub-bitstream extraction for --alllayers / --oppoint
//
// This file does not write sequence headers or tile payloads; the pipeline
// composes already-encoded single-layer units, which is exactly the boundary
// the core encoder owns today.  It does rewrite the fixed-width f(3)
// seq_lcr_id field of each composed sequence header so the plan's global LCR
// association is not dangling; a single_picture_header_flag = 1 header has no
// such field (the id is inferred 0) and is rejected when an LCR is requested.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "svc_config.h"

namespace av2enc::svc {

// OBU types from §6.2.2, Table 6.1.
enum class ObuType : std::uint8_t {
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
  kLayerConfigRecord = 16,
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

struct ObuHeader {
  ObuType type = ObuType::kTemporalDelimiter;
  std::uint8_t tlayer_id = 0;
  std::uint8_t mlayer_id = 0;
  std::uint8_t xlayer_id = 0;
};

bool operator==(const ObuHeader& a, const ObuHeader& b);
bool operator!=(const ObuHeader& a, const ObuHeader& b);

bool is_temporal_delimiter(ObuType type);
bool is_global_layer_type(ObuType type);  // TD / MSDO infer GLOBAL_XLAYER_ID
bool is_frame_obu(ObuType type);
bool is_config_obu(ObuType type);  // SEQ / LCR / OPS / atlas

// §5.2.2 header. The extension byte is emitted only when an xlayer/mlayer id
// has to be signalled; TD/MSDO infer GLOBAL_XLAYER_ID with no extension.
bool encode_obu_header(const ObuHeader& header, std::vector<std::uint8_t>& out,
                       std::string& error);
bool decode_obu_header(std::span<const std::uint8_t> bytes, ObuHeader& header,
                       std::size_t& consumed, std::string& error);

void append_leb128(std::vector<std::uint8_t>& out, std::uint32_t value);
std::size_t leb128_size(std::uint32_t value);

// leb128(num_bytes_in_obu) | obu_header | payload (§5.2.1, Annex B.2).
// `payload` must already include trailing bits for extensible OBUs.
bool frame_obu(const ObuHeader& header, std::span<const std::uint8_t> payload,
               std::vector<std::uint8_t>& out, std::string& error);

struct ParsedObu {
  ObuHeader header;
  std::vector<std::uint8_t> payload;
  std::vector<std::uint8_t> raw;  // leb128 prefix + header + payload
};

bool parse_obus(std::span<const std::uint8_t> bytes,
                std::vector<ParsedObu>& obus, std::string& error);

// §5.8.2 global and §5.8.1 local LCR payloads, including the extensible-OBU
// `obu_extension_flag` and trailing bits. Dependency maps drive
// lcr_dependent_layer_map for mlayer > 0.
bool write_layer_config_record_payload(
    const layers::LayerConfigRecord& record,
    const layers::LayerDependencyMaps& dependencies,
    std::vector<std::uint8_t>& out, std::string& error);

// §5.10 / §5.11 OPS payloads, including the extensible-OBU tail. Emits
// ops_reset_flag = 1 and complete ops_mlayer_info() maps.
bool write_operating_point_set_payload(const layers::OperatingPointSet& set,
                                       std::vector<std::uint8_t>& out,
                                       std::string& error);

// One already-encoded single-layer temporal unit (TD + sequence header +
// frame OBUs from the existing encoder).
struct LayerInput {
  layers::LayerId layer;
  std::vector<std::uint8_t> obus;  // raw Annex B OBU bytes
};

struct ComposeOptions {
  bool emit_temporal_delimiter = true;
  bool emit_lcr = true;
  bool emit_ops = true;
};

// Re-frames each input with its xlayer id and emits one temporal unit:
// TD, global LCR, global OPS, then per-layer sequence/frame OBUs in ascending
// xlayer order. Deterministic for a given input list. Rejects missing,
// duplicated, out-of-order, or multi-frame inputs.
bool compose_layered_stream(const LayerPlan& plan,
                            const std::vector<LayerInput>& inputs,
                            const ComposeOptions& options,
                            std::vector<std::uint8_t>& out,
                            std::string& error);

// Annex F.3.2 retention decision for one OBU. The TD is always retained;
// global OBUs are retained through retention[GLOBAL][0][0]; SEQ/LCR/OPS/atlas
// survive at m0/t0 when their xlayer is selected.
bool obu_retained(const layers::RetentionMap& retention, const ObuHeader& header,
                  bool& retained, std::string& error);

// Annex F.3.2 sub-bitstream extraction. Copies retained OBUs byte-for-byte, so
// an all-layers extraction is identical to the input.
bool extract_substream(std::span<const std::uint8_t> input,
                       const layers::RetentionMap& retention,
                       std::vector<std::uint8_t>& output, std::string& error);

}  // namespace av2enc::svc
