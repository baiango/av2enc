// AV2 v1.0.0 DIP mode info syntax codec.  See dip_syntax.h.

#include "encoder/dip/dip_syntax.h"

#include <cstddef>
#include <string>

namespace av2enc::dip {

void DipCdfState::reset() noexcept {
  for (int ctx = 0; ctx < kUseDipContexts; ++ctx) {
    for (int i = 0; i < 3; ++i) {
      use_dip[static_cast<std::size_t>(ctx)][static_cast<std::size_t>(i)] =
          kDefaultUseDipCdf[ctx][i];
    }
  }
  for (int i = 0; i < kNumModes + 1; ++i) {
    dip_mode[static_cast<std::size_t>(i)] = kDefaultDipModeCdf[i];
  }
}

namespace {

bool entropy_ok(av2::Status status, std::string* error) noexcept {
  if (status == av2::Status::kOk) return true;
  if (error != nullptr) *error = "DIP syntax entropy coder error";
  return false;
}

}  // namespace

bool write_dip_mode_info(const SyntaxContext& context,
                         const DipModeInfo& info, av2::Encoder& encoder,
                         DipCdfState& cdfs, std::string* error) noexcept {
  if (!validate_dip_mode_info(context, info, error)) return false;
  if (!dip_mode_info_present(context)) return true;
  if (context.use_dip_ctx < 0 || context.use_dip_ctx >= kUseDipContexts) {
    if (error != nullptr) *error = "use_dip context outside 0..2";
    return false;
  }

  auto* use_dip_cdf =
      cdfs.use_dip[static_cast<std::size_t>(context.use_dip_ctx)].data();
  if (!entropy_ok(encoder.encode_symbol(use_dip_cdf, 2,
                                        info.use_dip ? 1 : 0),
                  error)) {
    return false;
  }
  if (!info.use_dip) return true;

  // dip_transpose L(1): §8.2.5 read_literal(1) / its inverse.
  if (!entropy_ok(encoder.encode_bool(info.dip_transpose), error)) return false;
  if (!entropy_ok(encoder.encode_symbol(cdfs.dip_mode.data(), kNumModes,
                                        info.dip_mode),
                  error)) {
    return false;
  }
  return true;
}

bool read_dip_mode_info(const SyntaxContext& context, av2::Decoder& decoder,
                        DipCdfState& cdfs, DipModeInfo& info,
                        std::string* error) noexcept {
  info = DipModeInfo{};
  if (!dip_mode_info_present(context)) {
    return validate_dip_mode_info(context, info, error);
  }
  if (context.use_dip_ctx < 0 || context.use_dip_ctx >= kUseDipContexts) {
    if (error != nullptr) *error = "use_dip context outside 0..2";
    return false;
  }

  auto* use_dip_cdf =
      cdfs.use_dip[static_cast<std::size_t>(context.use_dip_ctx)].data();
  int use_dip = 0;
  if (!entropy_ok(decoder.decode_symbol(use_dip_cdf, 2, &use_dip), error)) {
    return false;
  }
  info.use_dip = use_dip != 0;
  if (!info.use_dip) return true;

  bool transpose = false;
  if (!entropy_ok(decoder.read_bool(&transpose), error)) return false;
  info.dip_transpose = transpose;

  int mode = 0;
  if (!entropy_ok(decoder.decode_symbol(cdfs.dip_mode.data(), kNumModes, &mode),
                  error)) {
    return false;
  }
  info.dip_mode = mode;
  return validate_dip_mode_info(context, info, error);
}

}  // namespace av2enc::dip
