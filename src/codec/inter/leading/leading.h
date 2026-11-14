#pragma once

#include "core/bitstream/bitwriter.h"
#include "pipeline/grain/film_grain_syntax.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// AV2 v1.0.0 leading-picture framing.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification
// v1.0.0 (2026-05-28, commit 38f28e7).
//
//   * §6.2.2 Table 6.1  OBU types 6 (OBU_LEADING_TILE_GROUP),
//                       11 (OBU_LEADING_SEF), 13 (OBU_LEADING_TIP).
//   * §2              "Leading frame" and "Regular frame" definitions:
//                       IsRegular = 0 for leading OBUs.
//   * §5.18.2         frame_header_info() syntax, including the
//                       immediate_output_frame / implicit_output_frame bits.
//   * §7.3.3/§7.3.4   coded output frame unit vs coded non-output frame unit.
//   * §7.3.6/§7.3.7   coded extended layer unit and temporal unit ordering.
//   * §7.4.4          open random access: leading frames are discarded.
//   * §6.17.2         output semantics and the leading refresh restriction.
//
// This module owns only the leading picture framing (OBU headers, frame header
// fields, output suppression, reference restrictions and plan validation).  The
// picture payload itself is coded by the existing M1 intra / M2 inter encoder;
// see VIDEO_INTEGRATION.md for the exact call sites.
namespace av2enc::inter::leading {

// OBU types this module can emit or recognise.  Values are Table 6.1.
enum class ObuType : std::uint8_t {
  kSequenceHeader = 1,
  kTemporalDelimiter = 2,
  kClosedLoopKey = 4,
  kOpenLoopKey = 5,
  kLeadingTileGroup = 6,
  kRegularTileGroup = 7,
  kLeadingSef = 11,
  kRegularSef = 12,
  kLeadingTip = 13,
  kRegularTip = 14,
};

// The four frame kinds used by a leading-picture stream.  SEF/TIP are parsed
// and classified but this module only writes tile-group frames.
enum class FrameKind : std::uint8_t {
  kClosedLoopKey,
  kOpenLoopKey,
  kLeadingTileGroup,
  kRegularTileGroup,
};

ObuType obu_type_for(FrameKind kind) noexcept;
bool is_leading_obu(ObuType type) noexcept;
bool is_regular_obu(ObuType type) noexcept;
bool is_key_obu(ObuType type) noexcept;
bool is_tile_group_obu(ObuType type) noexcept;

// §5.4.1 sequence header configuration.  The defaults match VideoConfig so that an
// M2-coded tile payload can be rewrapped without changing sequence-level tools.
struct SequenceConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t base_q_idx = 90;
  std::uint8_t order_hint_bits = 4;
  std::uint8_t num_ref_frames = 8;
  std::uint8_t max_drl_bits_minus_1 = 4;
  // §5.4.1 monotonic_output_order_flag.  Leading pictures need 0: an OLK has
  // immediate_output_frame inferred 0, so implicit_output_frame is the only way
  // to output it without a second output frame in the same temporal unit.
  bool monotonic_output_order_flag = false;
  bool disable_cdf_update = true;
  // §5.4.8 average-CDF.  Opt-in (--avg-cdf); avg_cdf_type = 1 averages the
  // tile CDFs at frame end, which is the single-tile form for this writer.
  bool enable_avg_cdf = false;
  bool avg_cdf_type = true;
  // §5.4.2 film_grain_params_present. When set, every frame header codes
  // §5.18.10.1 film_grain_config() after reduced_tx_set.
  bool film_grain_params_present = false;
};

// §5.18.2 output flags actually written/derived for a frame.
struct OutputFlags {
  bool immediate = false;
  bool implicit = false;
};

// One frame's header fields.  Fields not listed here are fixed by the tool set
// this encoder enables (same set as M2) and are written by write_frame_header().
struct FrameHeader {
  FrameKind kind = FrameKind::kRegularTileGroup;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 0;
  std::uint8_t num_total_refs = 0;
  std::array<std::uint8_t, 8> ref_frame_idx{};

  // Only meaningful for FrameKind::kLeadingTileGroup.
  //   false: coded non-output frame unit (immediate=0, implicit=0).
  //   true : leading frame queued immediately when decoded (immediate=1).
  bool leading_output_frame = false;

  // §5.18.10.1 film_grain_config(), coded last when the sequence sets
  // film_grain_params_present. Non-output frames code nothing.
  filters::grain::FilmGrainConfig grain;
};

// Derives immediate_output_frame / implicit_output_frame exactly as §5.18.2
// does for the supported OBU kinds.
OutputFlags output_flags(const SequenceConfig& config,
                         const FrameHeader& header) noexcept;

// A whole decode-order plan.  frames[0] must be the CLK, then an OLK, then zero
// or more leading frames, then regular frames.
struct Plan {
  std::vector<FrameHeader> frames;
};

// Rejects plans that would violate the §7.4.4 leading-frame restrictions:
//   * leading frames only after an OLK and before the first regular frame;
//   * leading refresh_frame_flags disjoint from OlkRefresh (§6.17.2);
//   * regular frames after the OLK may only reference OlkRefresh slots
//     (no pre-OLK frame, no leading frame);
//   * non-output frames must refresh at least one slot;
//   * output-leading frames require monotonic_output_order_flag = 0;
//   * an OLK needs monotonic_output_order_flag = 0 (its immediate flag is
//     inferred 0 and this plan has no second output frame in the OLK TU).
bool validate_plan(const SequenceConfig& config, const Plan& plan,
                   std::string& error);

// §5.18.2 frame_header_info() bits (without is_first_tile_group and without
// byte_alignment).  Returns the number of bits written.
std::size_t write_frame_header(BitWriter& writer, const SequenceConfig& config,
                               const FrameHeader& header);

// §5.19 tile_group_obu() header:
//   is_first_tile_group f(1)
//   frame_header(isFirst = 1)
//   byte_alignment()
std::size_t write_tile_group_header(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameHeader& header);

// Complete tile-group OBU (header + tile payload) with the OBU type selected by
// header.kind.  `tile_payload` must already contain §8.2.4 exit_symbol().
std::vector<std::uint8_t> make_frame_obu(const SequenceConfig& config,
                                         const FrameHeader& header,
                                         std::span<const std::uint8_t> tile_payload);

// sequence_header_obu() wrapped as an Annex B.2 length-delimited OBU.
std::vector<std::uint8_t> make_sequence_header_obu(const SequenceConfig& config);

// Empty temporal delimiter OBU (§5.5).
std::vector<std::uint8_t> make_temporal_delimiter_obu();

// Assembles one temporal unit per plan frame in decode order:
//   TD + (SEQ at the CLK and at every OLK random access point) + frame OBU.
// The sequence header is repeated at the OLK because §7.3.8.7 requires it to be
// available when decoding starts at that random access point.
std::vector<std::vector<std::uint8_t>> assemble_temporal_units(
    const SequenceConfig& config, const Plan& plan,
    std::span<const std::vector<std::uint8_t>> tile_payloads);

}  // namespace av2enc::inter::leading
