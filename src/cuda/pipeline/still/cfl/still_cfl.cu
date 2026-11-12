// M1 CfL stage — plan adapter over src/cuda/cfl.
//
// Everything normative (luma average, array L, derived alpha, prediction) is
// in cfl_common.h / src/cuda/cfl/cfl.cu. This file only builds descriptors,
// validates them and forwards whole batches, so the M1 GPU path cannot drift
// from the CPU reference or from the standalone CfL component test.

#include "still_cfl.h"

// Declares cfl::valid(), the host-side descriptor check used below. The stage
// header intentionally depends only on the CUDA CfL component (cfl_common.h);
// the CPU reference header is included by the translation unit that needs the
// oracle declaration, exactly like src/cuda/cfl/cfl_test.cu does.
#include "codec/intra/cfl/cfl.h"

namespace av2enc::cuda::still {
namespace {

// Fields that must match between the U and V descriptors of one M1 block.
// `chroma`, `chroma_stride` and the availability flags are per plane; CfL's
// availability derivation is geometry-only, so those match too.
bool same_luma_geometry(const cfl::Block& a, const cfl::Block& b) {
  return a.luma == b.luma && a.luma_stride == b.luma_stride &&
         a.luma_max_x == b.luma_max_x && a.luma_max_y == b.luma_max_y &&
         a.sb_top_y == b.sb_top_y && a.x == b.x && a.y == b.y && a.w == b.w &&
         a.h == b.h && a.coded_width == b.coded_width &&
         a.coded_height == b.coded_height && a.have_above == b.have_above &&
         a.have_left == b.have_left && a.bit_depth == b.bit_depth &&
         a.ds_filter == b.ds_filter && a.sub_x == b.sub_x &&
         a.sub_y == b.sub_y;
}

bool fail(std::string* error, const char* what) {
  if (error != nullptr) {
    *error = what;
  }
  return false;
}

}  // namespace

cfl::Block make_still_cfl_block(const StillCflFrame& frame, const StillCflRect& rect) {
  cfl::Block block;
  block.luma = frame.luma;
  block.luma_stride = frame.luma_stride;
  block.luma_max_x = frame.luma_max_x;
  block.luma_max_y = frame.luma_max_y;
  block.sb_top_y = cfl::cfl_sb_top_y(rect.y, frame.sub_y);
  block.chroma = rect.chroma;
  block.chroma_stride = rect.chroma_stride;
  block.x = rect.x;
  block.y = rect.y;
  block.w = rect.w;
  block.h = rect.h;
  block.coded_width = rect.coded_width;
  block.coded_height = rect.coded_height;
  block.have_above = rect.have_above >= 0
                         ? (rect.have_above != 0 ? 1 : 0)
                         : (cfl::cfl_avail_above(rect.y, frame.sub_y) ? 1 : 0);
  block.have_left = rect.have_left >= 0
                        ? (rect.have_left != 0 ? 1 : 0)
                        : (cfl::cfl_avail_left(rect.x, frame.sub_x) ? 1 : 0);
  block.bit_depth = frame.bit_depth;
  block.ds_filter = frame.ds_filter;
  block.sub_x = frame.sub_x;
  block.sub_y = frame.sub_y;
  return block;
}

StillCflPair make_still_cfl_pair(const StillCflFrame& frame, const StillCflRect& u, const StillCflRect& v) {
  StillCflPair pair;
  pair.u = make_still_cfl_block(frame, u);
  pair.v = make_still_cfl_block(frame, v);
  return pair;
}

bool still_cfl_valid_pair(const StillCflPair& pair, std::string* error) {
  if (!cfl::valid(pair.u)) {
    return fail(error, "U block descriptor is not valid");
  }
  if (!cfl::valid(pair.v)) {
    return fail(error, "V block descriptor is not valid");
  }
  if (!same_luma_geometry(pair.u, pair.v)) {
    return fail(error, "U and V do not share luma geometry");
  }
  if (pair.u.chroma == pair.v.chroma) {
    return fail(error, "U and V must not alias the same chroma plane");
  }
  return true;
}

cudaError_t still_cfl_predict_pairs(const StillCflPair* d_pairs, int pair_count,
                          int* d_alpha_q8, cudaStream_t stream) {
  if (pair_count <= 0) {
    return cudaSuccess;
  }
  if (d_pairs == nullptr) {
    return cudaErrorInvalidValue;
  }
  // StillCflPair is exactly two cfl::Block, so the pair array is already a planar
  // Block array for cfl_predict_derived_batch: U belongs to even jobs, V to
  // odd jobs.
  const cfl::Block* d_blocks = reinterpret_cast<const cfl::Block*>(d_pairs);
  return cfl_predict_derived_batch(d_blocks, d_alpha_q8, pair_count * 2,
                                   stream);
}

cudaError_t still_cfl_predict_pairs_explicit(const StillCflPairExplicit* d_pairs,
                                   int pair_count, cudaStream_t stream) {
  if (pair_count <= 0) {
    return cudaSuccess;
  }
  if (d_pairs == nullptr) {
    return cudaErrorInvalidValue;
  }
  const CflJob* d_jobs = reinterpret_cast<const CflJob*>(d_pairs);
  return cfl_predict_batch(d_jobs, pair_count * 2, stream);
}

}  // namespace av2enc::cuda::still
