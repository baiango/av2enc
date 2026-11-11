#ifndef AV2ENC_INTER_TLAYERS_TLAYERS_SYNTAX_H_
#define AV2ENC_INTER_TLAYERS_TLAYERS_SYNTAX_H_

// Exact AV2 v1.0.0 syntax for temporal layers:
//
//   §5.2.2  obu_header()            obu_tlayer_id f(2)
//   §5.4.1  sequence_header_obu()   max_tlayer_id f(2),
//                                   tlayer_dependency_present_flag f(1),
//                                   tlayer_dependency_map f(1),
//                                   multi_tlayer_dependency_map_present_flag
//   §5.18.2 frame_header_info()     immediate/implicit_output_frame,
//                                   order_hint, refresh_frame_flags,
//                                   explicit reference map
//
// The writer is a superset of the M2 sequence/frame header writers: for
// max_tlayer_id == 0 / monotonic_output_order_flag == 1 it emits byte-identical
// OBUs (checked by tlayers_test.cpp against src/encoder/m2/m2_headers.cpp).
// The reference model and tile payloads are not written here.

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/tlayers/tlayers.h"

namespace av2enc::inter::tlayers {

// §5.4.1 fields that interact with temporal layers. The remaining sequence
// fields are fixed to the M2 baseline (profile 0, level 5.1, 8-bit 4:2:0,
// one tile, no advanced tools).
struct SequenceConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // 7 = level 5.1
  std::uint8_t base_q_idx = 90;
  std::uint8_t order_hint_bits = 4;
  std::uint8_t num_ref_frames = 2;
  std::uint8_t max_drl_bits_minus_1 = 4;
  bool disable_cdf_update = true;
  bool enable_cfl_intra = true;

  // §5.4.1 temporal layers. max_tlayer_id is f(2) => 0..3.
  int max_tlayer_id = 0;
  // Must be false for any stream that reorders output with
  // implicit_output_frame (§5.18.2); true matches the M2 baseline.
  bool monotonic_output_order_flag = true;
  // Only coded when max_tlayer_id > 0.
  bool tlayer_dependency_present_flag = false;
  // TLayerDependencyMap[0][curr][ref]. Entry [0][0] is implied and never coded.
  std::array<std::array<bool, kMaxTLayers>, kMaxTLayers> tlayer_dependency{};

  // Fill max_tlayer_id, the dependency flag and the map from a plan.
  void apply_plan(const Plan& plan);
};

// SequenceConfig for the M2 baseline (max_tlayer_id = 0, monotonic = 1) with
// the same defaults M2 uses in m2_frame.cpp.
SequenceConfig baseline_sequence_config(std::uint16_t width,
                                        std::uint16_t height,
                                        std::uint8_t base_q_idx,
                                        std::uint8_t num_ref_frames);

// §5.18.2 per-frame fields. `order_hint_lsbs` is the value coded with
// f(OrderHintBits), not the absolute hint.
struct FrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint_lsbs = 0;
  std::uint32_t refresh_frame_flags = 0;
  bool frame_explicit_ref_frame_map = true;
  std::uint8_t num_total_refs = 0;
  std::uint8_t ref_frame_idx[kMaxRefs] = {0, 0};
  bool immediate_output_frame = true;
  bool implicit_output_frame = false;
  bool reference_select = false;
};

// Fills the frame header for a plan entry.
FrameHeader frame_header_for(const SequenceConfig& config, const Plan& plan,
                             int display_index);

// Annex B.2 length-delimited OBU. `obu_tlayer_id` is written in the 2 LSBs of
// the header byte (§5.2.2). TD/SEQ/key OBUs must use 0; the writer asserts it.
std::vector<std::uint8_t> make_obu(std::uint8_t obu_type,
                                   std::uint8_t obu_tlayer_id,
                                   std::span<const std::uint8_t> payload);

std::vector<std::uint8_t> make_sequence_header_obu(const SequenceConfig& config);
std::vector<std::uint8_t> make_temporal_delimiter_obu();

// §5.19 tile_group_obu() header bytes (is_first_tile_group, frame_header,
// byte_alignment) followed by the caller's tile payload.
std::vector<std::uint8_t> make_frame_obu(const SequenceConfig& config,
                                         const FrameHeader& header,
                                         std::uint8_t obu_tlayer_id,
                                         std::span<const std::uint8_t> tile_payload);

// Reads the temporal-layer-relevant fields of a sequence header OBU (including
// §5.2.2's OBU header and §5.2.3 trailing bits). Used by the golden tests and
// by the gate to prove the emitted header is what the plan asked for.
struct ParsedSequenceHeader {
  int seq_header_id = 0;
  int seq_profile_idc = 0;
  bool single_picture_header_flag = false;
  int seq_level_idx = 0;
  int max_tlayer_id = 0;
  int max_mlayer_id = 0;
  bool monotonic_output_order_flag = true;
  int max_frame_width = 0;
  int max_frame_height = 0;
  bool tlayer_dependency_present_flag = false;
  bool multi_tlayer_dependency_map_present_flag = false;
  // Effective map: initialised to (ref <= curr) then overridden by the coded
  // bits. Entry [0][0] is always true.
  std::array<std::array<bool, kMaxTLayers>, kMaxTLayers> tlayer_dependency{};
  // Which entries were actually coded (useful for golden pinning).
  std::array<std::array<bool, kMaxTLayers>, kMaxTLayers> tlayer_dependency_coded{};
  int order_hint_bits = 0;
  int num_ref_frames = 0;
  bool enable_short_refresh_frame_flags = false;
};

// OBU header fields (§5.2.2). `tlayer_id` is present for every OBU type.
struct ParsedObuHeader {
  bool extension_flag = false;
  int obu_type = 0;
  int obu_tlayer_id = 0;
  int obu_mlayer_id = 0;
  int obu_xlayer_id = 0;
  std::size_t header_bytes = 0;
  std::size_t total_bytes = 0;  // leb128 prefix + header + payload
  std::span<const std::uint8_t> payload;
};

bool parse_obu(std::span<const std::uint8_t> bytes, std::size_t offset,
               ParsedObuHeader& out, std::string& error);

bool parse_sequence_header_obu(std::span<const std::uint8_t> obu,
                               ParsedSequenceHeader& out, std::string& error);

// §5.2.2 header-byte helper: (extension_flag << 7) | (obu_type << 2) | tlayer.
std::uint8_t obu_header_byte(bool extension_flag, int obu_type,
                             int obu_tlayer_id);

const char* tlayer_role_name(int obu_type);

}  // namespace av2enc::inter::tlayers

#endif  // AV2ENC_INTER_TLAYERS_TLAYERS_SYNTAX_H_
