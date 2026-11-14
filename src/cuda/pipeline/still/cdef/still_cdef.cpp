// M1 GPU path — CDEF stage host implementation.
//
// Header packing, MI-grid materialisation, stage validation and the CPU-side
// strength search.  No CUDA here: this file is part of the orchestrator's
// host OBJECT library and of the CPU gate.

#include "cuda/pipeline/still/cdef/still_cdef.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/filters/cdef/cdef_core.h"
#include "codec/filters/cdef/cdef_search.h"

namespace av2enc::cuda::still {
namespace {

using cdef::CdefStrength;
using cdef::kMaxStrengths;
using cdef::kMiSize;

// ceil(log2(values)): the fixed-width cost of a symbol with `values` states.
int bits_for_range(int values) {
  int bits = 0;
  int span = values > 0 ? values - 1 : 0;
  while (span > 0) {
    ++bits;
    span >>= 1;
  }
  return bits;
}

}  // namespace

bool still_cdef_build_header(const cdef::CdefOptions& options,
                          bool single_picture_header, CdefHeaderParams* out,
                          std::string* error) {
  auto fail = [error](const char* text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  if (out == nullptr) {
    return fail("null header output");
  }
  *out = CdefHeaderParams{};
  if (options.damping < 3 || options.damping > 6) {
    return fail("CdefDamping outside 3..6");
  }
  if (options.num_strengths < 1 || options.num_strengths > kMaxStrengths) {
    return fail("CdefStrengths outside 1..8");
  }
  for (int i = 0; i < options.num_strengths; ++i) {
    const CdefStrength& s = options.strengths[i];
    if (s.y_pri < 0 || s.y_pri > 15 || s.uv_pri < 0 || s.uv_pri > 15) {
      return fail("primary strength outside 0..15");
    }
    if ((s.y_sec != 0 && s.y_sec != 1 && s.y_sec != 2 && s.y_sec != 4) ||
        (s.uv_sec != 0 && s.uv_sec != 1 && s.uv_sec != 2 && s.uv_sec != 4)) {
      return fail("secondary strength not in {0,1,2,4}");
    }
  }

  out->frame_enable = true;  // the stage only runs for an enabled frame
  out->damping_minus_3 = options.damping - 3;
  out->strengths_minus_1 = options.num_strengths - 1;
  out->on_skip_txfm_frame_enable = options.on_skip_txfm;
  out->num_strengths = options.num_strengths;

  int bits = 2 + 3 + 1;  // damping, strengths-1, skip flag (ADAPTIVE)
  if (!single_picture_header) {
    bits += 1;  // cdef_frame_enable
  }
  for (int i = 0; i < options.num_strengths; ++i) {
    const CdefStrength& s = options.strengths[i];
    CdefHeaderStrength& h = out->set[i];
    h.y_pri_zero = s.y_pri == 0;
    h.y_pri_strength = s.y_pri;
    h.y_sec_value = s.y_sec;
    h.y_sec_strength = s.y_sec == 4 ? 3 : s.y_sec;
    h.uv_pri_zero = s.uv_pri == 0;
    h.uv_pri_strength = s.uv_pri;
    h.uv_sec_value = s.uv_sec;
    h.uv_sec_strength = s.uv_sec == 4 ? 3 : s.uv_sec;
    bits += 1 + (h.y_pri_zero ? 0 : 4) + 2;
    bits += 1 + (h.uv_pri_zero ? 0 : 4) + 2;
  }
  out->bits = bits;
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool still_cdef_build_mi_grids(const CdefMiInputs& inputs, CdefMiGrids* out,
                            std::string* error) {
  auto fail = [error](const char* text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  if (out == nullptr) {
    return fail("null grid output");
  }
  *out = CdefMiGrids{};
  if (inputs.mi_cols <= 0 || inputs.mi_rows <= 0) {
    return fail("invalid MI geometry");
  }
  if ((inputs.segment_id != nullptr || inputs.chroma_segment_id != nullptr) &&
      (inputs.lossless_array == nullptr || inputs.lossless_array_size <= 0)) {
    return fail("segment ids require a lossless array");
  }
  const std::size_t count =
      static_cast<std::size_t>(inputs.mi_cols) *
      static_cast<std::size_t>(inputs.mi_rows);
  out->mi_cols = inputs.mi_cols;
  out->mi_rows = inputs.mi_rows;
  out->skip.assign(count, 0);
  out->lossless.assign(count, 0);
  out->lossless_chroma.assign(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    if (inputs.skip != nullptr) {
      out->skip[i] = inputs.skip[i] != 0 ? 1 : 0;
    }
    if (inputs.segment_id != nullptr) {
      const int seg = static_cast<int>(inputs.segment_id[i]);
      if (seg < 0 || seg >= inputs.lossless_array_size) {
        return fail("segment id outside the lossless array");
      }
      out->lossless[i] = inputs.lossless_array[seg] != 0 ? 1 : 0;
    }
    if (inputs.chroma_segment_id != nullptr) {
      const int seg = static_cast<int>(inputs.chroma_segment_id[i]);
      if (seg < 0 || seg >= inputs.lossless_array_size) {
        return fail("chroma segment id outside the lossless array");
      }
      out->lossless_chroma[i] = inputs.lossless_array[seg] != 0 ? 1 : 0;
    }
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool still_cdef_plan(const CdefStageFrame& frame, const CdefStageOptions& options,
                  CdefStagePlan* plan, std::string* error) {
  auto fail = [error](const char* text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  if (plan == nullptr) {
    return fail("null plan output");
  }
  *plan = CdefStagePlan{};
  if (frame.width <= 0 || frame.height <= 0 || frame.mi_cols <= 0 ||
      frame.mi_rows <= 0) {
    return fail("invalid frame geometry");
  }
  if (frame.width > frame.mi_cols * kMiSize ||
      frame.height > frame.mi_rows * kMiSize) {
    return fail("visible area exceeds the MI-aligned planes");
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int width = (frame.mi_cols * kMiSize) >> sub;
    const int height = (frame.mi_rows * kMiSize) >> sub;
    if (frame.plane[plane].src == nullptr || frame.plane[plane].stride < width) {
      return fail("missing or short source plane");
    }
    // Planning does not write pixels, so `dst` may be null.  When it is
    // supplied it must not alias the source (the filter is a pure function of
    // the unfiltered frame).
    if (frame.plane[plane].dst != nullptr &&
        frame.plane[plane].src == frame.plane[plane].dst) {
      return fail("CDEF source and destination must not alias");
    }
    if (frame.plane[plane].height != 0 &&
        frame.plane[plane].height < height) {
      return fail("source plane height shorter than the MI-aligned plane");
    }
    if (frame.ref[plane] == nullptr || frame.ref_stride[plane] < width) {
      return fail("missing or short reference plane");
    }
  }

  cdef::CdefSearchFrame view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.src[plane].data = frame.plane[plane].src;
    view.src[plane].stride = frame.plane[plane].stride;
    view.ref[plane].data = frame.ref[plane];
    view.ref[plane].stride = frame.ref_stride[plane];
  }
  view.width = frame.width;
  view.height = frame.height;
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.subsampling_x = 1;
  view.subsampling_y = 1;
  view.skip = frame.skip;
  view.lossless = frame.lossless;
  view.lossless_chroma = frame.lossless_chroma;
  view.tile_col_starts = frame.tile_col_starts;
  view.num_tile_cols = frame.num_tile_cols;
  view.tile_row_starts = frame.tile_row_starts;
  view.num_tile_rows = frame.num_tile_rows;

  cdef::CdefSearchOptions search;
  search.damping = options.damping;
  search.num_strengths = options.num_strengths;
  search.on_skip_txfm = options.on_skip_txfm;
  search.disable_loopfilters_across_tiles =
      options.disable_loopfilters_across_tiles;
  search.include_off = options.include_off;
  search.chroma_weight = options.chroma_weight;
  search.candidates = options.candidates;
  search.candidate_count = options.candidate_count;
  search.eval_step = options.search_step;

  std::string search_error;
  if (!cdef::cdef_search_plan(view, search, &plan->search, &search_error)) {
    return fail(search_error.empty() ? "strength search failed"
                                     : search_error.c_str());
  }
  if (!still_cdef_build_header(plan->search.options, /*single_picture_header=*/true,
                            &plan->header, error)) {
    return false;
  }
  plan->num_strengths = plan->search.num_strengths;
  plan->n64_cols = plan->search.n64_cols;
  plan->n64_rows = plan->search.n64_rows;
  plan->cdef_idx = plan->search.cdef_idx;
  plan->header_bits = plan->header.bits;
  plan->sse_total = plan->search.sse_total;
  plan->sse_off_total = plan->search.sse_off_total;
  plan->forced_single_strength = plan->search.forced_single_strength;

  // Signalability: a stream carries an index in [0, CdefStrengths) for every
  // 64x64 block, and no index at all when CdefStrengths == 1.
  for (std::int8_t idx : plan->cdef_idx) {
    if (idx < 0 || idx >= plan->num_strengths) {
      return fail("search produced an unsignalable cdef_idx");
    }
  }
  if (plan->num_strengths == 1) {
    for (std::int8_t idx : plan->cdef_idx) {
      if (idx != 0) {
        return fail("single strength requires cdef_idx == 0 everywhere");
      }
    }
    plan->index_bits_estimate = 0;
  } else {
    const int blocks = plan->n64_cols * plan->n64_rows;
    const int minus_bits = bits_for_range(plan->num_strengths - 1);
    plan->index_bits_estimate = blocks * (1 + minus_bits);
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool still_cdef_validate_device_frame(const CdefDeviceFrame& frame,
                                   const cdef::CdefOptions& options,
                                   std::string* error) {
  auto fail = [error](const char* text) {
    if (error != nullptr) {
      *error = text;
    }
    return false;
  };
  if (options.bit_depth != 8) {
    return fail("only 8-bit is supported");
  }
  if (options.damping < 3 || options.damping > 6) {
    return fail("damping outside 3..6");
  }
  if (options.num_strengths < 1 || options.num_strengths > kMaxStrengths) {
    return fail("num_strengths outside 1..8");
  }
  for (int i = 0; i < options.num_strengths; ++i) {
    const CdefStrength& s = options.strengths[i];
    if (!cdef::cdef_valid_strength(s)) {
      return fail("illegal strength set");
    }
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0 ||
      (frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return fail("invalid or odd MI grid");
  }
  if (frame.width <= 0 || frame.height <= 0 ||
      frame.width > frame.mi_cols * kMiSize ||
      frame.height > frame.mi_rows * kMiSize) {
    return fail("invalid visible geometry");
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = (plane == 0) ? 0 : 1;
    const int width = (frame.mi_cols * kMiSize) >> sub;
    if (frame.src[plane] == nullptr || frame.dst[plane] == nullptr ||
        frame.src[plane] == frame.dst[plane]) {
      return fail("null or aliasing plane");
    }
    if (frame.stride[plane] < width) {
      return fail("short plane stride");
    }
  }
  if (frame.skip == nullptr || frame.lossless == nullptr) {
    return fail("missing metadata grid");
  }
  const int need_cols = (frame.mi_cols + cdef::kCdefSize4 - 1) /
                        cdef::kCdefSize4;
  const int need_rows = (frame.mi_rows + cdef::kCdefSize4 - 1) /
                        cdef::kCdefSize4;
  if (frame.n64_cols < need_cols || frame.n64_rows < need_rows) {
    return fail("cdef_idx grid smaller than the frame");
  }
  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr) ||
      frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return fail("invalid tile arrays");
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

}  // namespace av2enc::cuda::still
