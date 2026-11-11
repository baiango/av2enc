#include "tcq_flags.h"

namespace av2enc::tcq {

bool derive_header_flags(const HeaderInput& input, HeaderFlags* flags,
                         std::string* error) {
  (void)error;
  if (flags == nullptr) return false;
  HeaderFlags out;

  // §5.4.8 sequence_transform_quant_entropy_config().
  out.enable_cctx = input.monochrome ? false : input.enable_cctx;
  out.enable_tcq = input.enable_tcq;
  out.choose_tcq_per_frame =
      input.enable_tcq && !input.single_picture_header_flag
          ? input.choose_tcq_per_frame
          : false;
  out.enable_parity_hiding =
      (input.enable_tcq && !out.choose_tcq_per_frame)
          ? false
          : input.enable_parity_hiding;

  if (!input.monochrome) ++out.sequence_bits;  // enable_cctx
  ++out.sequence_bits;                         // enable_tcq
  if (input.enable_tcq && !input.single_picture_header_flag) {
    ++out.sequence_bits;  // choose_tcq_per_frame
  }
  if (!(input.enable_tcq && !out.choose_tcq_per_frame)) {
    ++out.sequence_bits;  // enable_parity_hiding
  }

  // Frame header, after the lossless flags are known.
  if (input.coded_lossless) {
    out.allow_tcq = false;
  } else if (out.choose_tcq_per_frame) {
    out.allow_tcq = input.frame_allow_tcq;
    ++out.frame_bits;
  } else {
    out.allow_tcq = out.enable_tcq;
  }

  if (input.coded_lossless || !out.enable_parity_hiding || out.allow_tcq) {
    out.allow_parity_hiding = false;
  } else {
    out.allow_parity_hiding = input.frame_allow_parity_hiding;
    ++out.frame_bits;
  }

  *flags = out;
  return true;
}

}  // namespace av2enc::tcq
