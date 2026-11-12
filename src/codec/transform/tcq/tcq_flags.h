// Header/flag derivation for AV2 TCQ and CCTX.
//
// Sequence: §5.4.8 sequence_transform_quant_entropy_config()
// Frame:    §5.18.2 frame header, `allow_tcq` / `allow_parity_hiding`
//
// The important interaction is normative: when `enable_tcq` is 1 and
// `choose_tcq_per_frame` is 0, `enable_parity_hiding` is inferred to 0.  For a
// single-picture stream `choose_tcq_per_frame` is always 0, so enabling TCQ
// also disables parity hiding.  Independently, `allow_parity_hiding` is forced
// to 0 whenever `CodedLossless` or `allow_tcq` is 1.
//
// This file derives the flags and the exact number of header bits a writer
// must emit.  It writes nothing itself.

#pragma once

#include <cstdint>
#include <string>

namespace av2enc::tcq {

struct HeaderInput {
  bool single_picture_header_flag = false;
  bool monochrome = false;
  bool coded_lossless = false;

  // Sequence-level requested values.  `enable_cctx` is ignored when
  // `monochrome` is true; `choose_tcq_per_frame` is ignored when
  // `single_picture_header_flag` is true or `enable_tcq` is false.
  bool enable_cctx = false;
  bool enable_tcq = false;
  bool choose_tcq_per_frame = false;
  bool enable_parity_hiding = false;

  // Frame-level requested values.  Only used when the syntax actually codes
  // the corresponding bit.
  bool frame_allow_tcq = false;
  bool frame_allow_parity_hiding = false;
};

struct HeaderFlags {
  bool enable_cctx = false;
  bool enable_tcq = false;
  bool choose_tcq_per_frame = false;
  bool enable_parity_hiding = false;
  bool allow_tcq = false;
  bool allow_parity_hiding = false;

  // Bits in sequence_transform_quant_entropy_config() contributed by the
  // TCQ/CCTX/parity-hiding fields (not the whole sequence header).
  int sequence_bits = 0;
  // Bits in the frame header contributed by allow_tcq / allow_parity_hiding.
  int frame_bits = 0;
};

// Returns false only for contradictory input (currently: lossless with a
// requested non-zero frame allow flag when that flag is not codable is not an
// error; it is normalised below).  The function is total and deterministic.
bool derive_header_flags(const HeaderInput& input, HeaderFlags* flags,
                         std::string* error = nullptr);

}  // namespace av2enc::tcq
