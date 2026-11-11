#include "encoder/m2/rdo/context.h"

namespace av2enc::m2::rdo {

int skip_context_from_neighbours(const merge::NeighborModes& neighbours,
                                 bool skip_mode) {
  return merge::skip_flag_context(neighbours, skip_mode);
}

bool new_mv_context_from_neighbours(
    const merge::NewMvContextParams& params, int* ctx) {
  if (ctx == nullptr) return false;
  return merge::derive_new_mv_context(params, ctx) == merge::Status::kOk;
}

RateContext make_rate_context(
    int is_inter_ctx, const merge::NeighborModes& neighbours,
    const merge::NewMvContextParams& new_mv,
    int max_drl_bits_minus_1) {
  RateContext context;
  context.is_inter_ctx = is_inter_ctx;
  context.skip_ctx = skip_context_from_neighbours(neighbours, false);
  int new_mv_ctx = 0;
  if (new_mv_context_from_neighbours(new_mv, &new_mv_ctx)) {
    context.new_mv_ctx = new_mv_ctx;
  }
  context.max_drl_bits_minus_1 = max_drl_bits_minus_1;
  return context;
}

}  // namespace av2enc::m2::rdo
