// CLI carrier for the AV2 v1.0.0 metadata units (§5.17.2 short OBUs).
//
// The unit codecs live in `frame/metadata/hdr` (CLL/MDCV) and
// `frame/metadata/types` (everything else).  This module is the small
// product-facing adapter that turns the opt-in `--metadata-*` flags into
// §5.17.2 suffix short OBUs of one temporal unit.
//
// All options default to "not signalled", so `--still`/`--video` bitstreams
// are byte-identical while no flag is set.  The payload defaults are the
// neutral spec-valid forms of each unit, not invented content policy.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf
//   §5.17.2 metadata_short_obu()
//   §6.16.5 metadata_hdr_cll
//   §7.3.3  prefix/suffix placement

#pragma once

#include "frame/metadata/hdr/hdr_metadata.h"
#include "frame/metadata/types/metadata_types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::metadata::cli {

struct Options {
  // §6.16.13 decoded frame hash.  The hash is computed from the encoder
  // reconstruction (8-bit 4:2:0 plane data, the CLI paths that support it).
  // `frame_hash_per_plane` selects three 16-byte plane digests; otherwise one
  // digest covers Y+U+V in plane order.
  struct FramePlanes {
    std::span<const std::uint8_t> y;
    std::span<const std::uint8_t> u;
    std::span<const std::uint8_t> v;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
  };
  bool frame_hash = false;
  bool frame_hash_per_plane = false;

  // §6.16.8 banding hints.  The neutral form signals neither coding nor source
  // banding (the one-byte unit 0x00).
  bool banding = false;
  BandingHints banding_value{};

  // §6.16.12 user data unregistered.  The default is a zero UUID with an
  // empty payload (a 16-byte unit).
  bool user_data = false;
  UserDataUnregistered user_data_value{};

  // §6.16.5 content light level.  `{0, 0}` is the "not signalled" neutral
  // form; the flag only emits the OBU when `cll` is true.
  bool cll = false;
  hdr::ContentLightLevel cll_value{};

  // §6.16.6 mastering display colour volume.  The default is the all-zero
  // "unknown / unspecified" form (§6.16.6 defines out-of-range values that
  // way); `--metadata-mdcv bt2020` selects the ST 2086 BT.2020/D65 preset.
  bool mdcv = false;
  hdr::MasteringDisplayColorVolume mdcv_value{};

  // §6.16.9 ICC profile.  The no-argument form carries the built-in 132-byte
  // header-only v4.3 profile; a file argument is passed through byte-for-byte.
  bool icc = false;
  std::vector<std::uint8_t> icc_data;

  // §6.16.11 temporal point info (short OBU only).  Default 0.
  bool temporal_point = false;
  std::uint32_t frame_presentation_time = 0;

  bool any() const {
    return frame_hash || banding || user_data || cll || mdcv || icc ||
           temporal_point;
  }
};

// The neutral ICC payload: a structurally complete, header-only ICC v4.3
// profile (132 bytes: 128-byte header + zero tags).  §6.16.9 only interprets
// bytes 8/9; avmdec additionally requires a >=128-byte payload, which this
// satisfies.  The profile is not a colour-managed transform on its own.
std::vector<std::uint8_t> default_icc_profile();

// Appends every enabled unit as a §5.17.2 suffix short OBU (current layer,
// global persistence) to `unit`.  Returns false with `error` on a payload or
// framing failure; `unit` is left untouched on failure.
bool append_suffix_metadata(std::vector<std::uint8_t>& unit,
                            const Options& options,
                            const Options::FramePlanes* planes,
                            std::string& error);

}  // namespace av2enc::metadata::cli
