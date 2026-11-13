// §5.18.7.11 lr_params() glue shared by the 10-bit header and tile writers.
#ifndef AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_LR_SYNTAX_H_
#define AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_LR_SYNTAX_H_

#include <algorithm>
#include <cstdint>

#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "formats/high_bitdepth/hbd_filter_config.h"

namespace av2enc::hbd {

namespace lr_syntax = av2enc::filters::pc_wiener_stream;

// §5.18.7.11 derives LoopRestorationSize[chroma] from
// (512 >> max(subX, subY)) >> chroma_shift.
inline int hbd_chroma_lr_shift(int sub_x, int sub_y) {
  return 3 - std::max(sub_x, sub_y);
}

inline int hbd_lr_chroma_unit_size(int sub_x, int sub_y) {
  return 512 >> std::max(sub_x, sub_y) >> hbd_chroma_lr_shift(sub_x, sub_y);
}

// Frame-level LR state for the writer/reader pair.
inline void fill_lr_params(const HbdLrConfig& config, int num_planes,
                           lr_syntax::LrParams* params) {
  *params = lr_syntax::LrParams{};
  for (int p = 0; p < 3; ++p) {
    params->frame_type[p] =
        p < num_planes ? config.plane[p].frame_type : lr_syntax::kRestoreNone;
    params->frame_filters_on[p] = false;
  }
  params->num_filter_classes = 1;
  params->num_filter_classes_idx = 0;
  params->luma_shift = config.luma_shift;
  params->chroma_shift = config.chroma_shift;
}

// The sequence and frame tool lists must agree with the per-plane frame types.
inline void fill_lr_header_context(const HbdLrConfig& config, int num_planes,
                                   int sub_x, int sub_y, int base_q_idx,
                                   lr_syntax::LrHeaderContext* context) {
  *context = lr_syntax::LrHeaderContext{};
  context->enable_restoration = true;
  context->single_picture_header_flag = true;
  context->frame_is_intra = true;
  context->sb_size = 64;
  context->num_planes = num_planes;
  context->sub_x = sub_x;
  context->sub_y = sub_y;
  context->base_q_idx = base_q_idx;
  const bool luma_pc =
      num_planes > 0 && config.plane[0].frame_type == lr_syntax::kRestorePcWiener;
  const bool luma_ns =
      num_planes > 0 && config.plane[0].frame_type == lr_syntax::kRestoreWienerNs;
  const bool chroma_ns =
      num_planes > 2 && (config.plane[1].frame_type ==
                             lr_syntax::kRestoreWienerNs ||
                         config.plane[2].frame_type ==
                             lr_syntax::kRestoreWienerNs);
  context->lr_tools_disable[0][lr_syntax::kRestorePcWiener] = !luma_pc;
  context->lr_tools_disable[0][lr_syntax::kRestoreWienerNs] = !luma_ns;
  context->lr_tools_disable[1][lr_syntax::kRestorePcWiener] = true;
  context->lr_tools_disable[1][lr_syntax::kRestoreWienerNs] = !chroma_ns;
}

}  // namespace av2enc::hbd

#endif  // AV2ENC_FORMATS_HIGH_BITDEPTH_HBD_LR_SYNTAX_H_
