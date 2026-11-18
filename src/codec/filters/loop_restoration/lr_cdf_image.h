// AV2 v1.0.0 loop-restoration CDF image (§6.17.2, §7.4.4, §7.5, §8.2.2).
//
// init_non_coeff_cdfs() names six loop-restoration CDF arrays:
//
//   UseWienerNsCdf, WienerNsLengthCdf, WienerNsUvSymCdf, WienerNsBaseCdf,
//   UsePcWienerCdf, FlexRestorationTypeCdf
//
// §7.4.4 load_cdfs()/blend_cdfs() and §7.5 save_cdfs()/frame_end_update_cdf()
// copy *every* array named by init_coeff_cdfs/init_non_coeff_cdfs, so these
// arrays belong to the §8.2 frame CDF image: a frame with
// `disable_cross_frame_cdf_init = 0` starts its tile LR CDFs from the
// reference frame's adapted values, not from the §9.3 defaults.
//
// The tile-syntax modules (`filters_stream::LrState`,
// `pc_wiener_stream::TileState`) own a working copy.  A writer that carries
// CDFs across frames must seed that copy from the frame image and store the
// adapted values back before exporting the image; the helpers here and in
// those two modules do exactly that.
//
// Defaults come from the generated §9.3 tables in `classes/wiener_cdfs.h` so
// there is exactly one copy of each default row.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_LR_CDF_IMAGE_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_LR_CDF_IMAGE_H_

#include <cstdint>
#include <cstring>

#include "codec/filters/loop_restoration/classes/wiener_cdfs.h"

namespace av2enc::filters::lr_cdf {

// One uint16_t entry per CDF symbol count + 1 (§8.2 inverse-CDF layout).
struct Image {
  std::uint16_t use_wiener_ns[3];
  std::uint16_t wiener_ns_length[2][3];
  std::uint16_t wiener_ns_uv_sym[3];
  std::uint16_t wiener_ns_base[5];
  std::uint16_t use_pc_wiener[3];
  std::uint16_t flex_restoration_type[3][3][3];
};

// §6.17.2 init_non_coeff_cdfs() defaults.
inline void SetDefaults(Image* image) {
  namespace rcls = ::av2enc::restoration::classes;
  std::memcpy(image->use_wiener_ns, rcls::kUseWienerNsCdf,
              sizeof(image->use_wiener_ns));
  std::memcpy(image->wiener_ns_length, rcls::kWienerNsLengthCdf,
              sizeof(image->wiener_ns_length));
  std::memcpy(image->wiener_ns_uv_sym, rcls::kWienerNsUvSymCdf,
              sizeof(image->wiener_ns_uv_sym));
  std::memcpy(image->wiener_ns_base, rcls::kWienerNsBaseCdf,
              sizeof(image->wiener_ns_base));
  std::memcpy(image->use_pc_wiener, rcls::kUsePcWienerCdf,
              sizeof(image->use_pc_wiener));
  std::memcpy(image->flex_restoration_type, rcls::kFlexRestorationTypeCdf,
              sizeof(image->flex_restoration_type));
}

}  // namespace av2enc::filters::lr_cdf

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_LR_CDF_IMAGE_H_
