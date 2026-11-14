// M2 CDEF adapter — implementation.
//
// The pixel core, the SSE search and the §5.20.10.1 context are reused from the
// M1 CDEF workstream.  This file only adapts the M2 frame geometry, the header
// context (single_picture_header_flag = 0) and the tile-symbol call order.

#include "codec/filters/cdef/syntax/video_cdef.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "cuda/pipeline/still/cdef/still_cdef.h"

namespace av2enc::video::cdef_adapter {
namespace {

using av2enc::cdef::CdefOptions;
using av2enc::cdef::CdefStatus;
using av2enc::cdef::CdefStrength;
using av2enc::cuda::still::CdefStageFrame;
using av2enc::cuda::still::CdefStageOptions;
using av2enc::cuda::still::CdefStagePlan;

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

int plane_shift(int plane) { return plane == 0 ? 0 : 1; }
int cdef_size4() { return av2enc::cdef::kCdefSize4; }
int mx4() { return av2enc::cdef::kMiSize; }

bool validate_view(const FrameView& frame, bool need_source,
                   std::string* error) {
  if (frame.width <= 0 || frame.height <= 0 || frame.mi_cols <= 0 ||
      frame.mi_rows <= 0) {
    return fail(error, "invalid frame geometry");
  }
  if ((frame.mi_cols & 1) != 0 || (frame.mi_rows & 1) != 0) {
    return fail(error, "odd MI grid");
  }
  if (frame.width > frame.mi_cols * mx4() ||
      frame.height > frame.mi_rows * mx4()) {
    return fail(error, "visible area exceeds the MI-aligned planes");
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int width = (frame.mi_cols * mx4()) >> plane_shift(plane);
    if (frame.recon[plane].data == nullptr ||
        (need_source && frame.source[plane].data == nullptr)) {
      return fail(error, need_source ? "null reconstruction or source plane"
                                     : "null reconstruction plane");
    }
    if (frame.recon[plane].stride < width ||
        (need_source && frame.source[plane].stride < width)) {
      return fail(error, "short plane stride");
    }
  }
  if (frame.skip == nullptr || frame.lossless == nullptr) {
    return fail(error,
                "missing MI skip/lossless grids; use FrameGrids::zero() for "
                "the intra case");
  }
  if ((frame.num_tile_cols > 0 && frame.tile_col_starts == nullptr) ||
      (frame.num_tile_rows > 0 && frame.tile_row_starts == nullptr) ||
      frame.num_tile_cols < 0 || frame.num_tile_rows < 0) {
    return fail(error, "invalid tile start arrays");
  }
  return true;
}

filters_stream::CdefParams make_params(const CdefOptions& options) {
  filters_stream::CdefParams out;
  out.frame_enable = true;
  out.damping_minus_3 = options.damping - 3;
  out.strengths_minus_1 = options.num_strengths - 1;
  out.num_strengths = options.num_strengths;
  out.on_skip_txfm_frame_enable = options.on_skip_txfm;
  for (int i = 0; i < options.num_strengths; ++i) {
    const CdefStrength& s = options.strengths[i];
    filters_stream::CdefStrength& d = out.set[i];
    d.y_pri = s.y_pri;
    d.y_sec = s.y_sec;
    d.uv_pri = s.uv_pri;
    d.uv_sec = s.uv_sec;
  }
  return out;
}

}  // namespace

void FrameGrids::zero(int mi_cols, int mi_rows) {
  const std::size_t count = static_cast<std::size_t>(std::max(mi_cols, 0)) *
                            static_cast<std::size_t>(std::max(mi_rows, 0));
  skip.assign(count, 0);
  lossless.assign(count, 0);
  lossless_chroma.assign(count, 0);
}

void FrameGrids::bind(FrameView* view) const {
  if (view == nullptr) return;
  view->skip = skip.empty() ? nullptr : skip.data();
  view->lossless = lossless.empty() ? nullptr : lossless.data();
  view->lossless_chroma =
      lossless_chroma.empty() ? nullptr : lossless_chroma.data();
}

bool plan(const FrameView& frame, const Options& options, Plan* out,
          std::string* error) {
  if (out == nullptr) return fail(error, "null plan output");
  *out = Plan{};
  if (!validate_view(frame, /*need_source=*/true, error)) return false;
  if (options.num_strengths < 1 ||
      options.num_strengths > av2enc::cdef::kMaxStrengths) {
    return fail(error, "CdefStrengths outside 1..8");
  }
  if (options.cdef_on_skip_txfm < 0 || options.cdef_on_skip_txfm > 2) {
    return fail(error, "CdefOnSkipTxfm must be DISABLED/ALWAYS_ON/ADAPTIVE");
  }
  if (options.cdef_on_skip_txfm == 0 && options.on_skip_txfm) {
    return fail(error,
                "CdefOnSkipTxfm DISABLED cannot enable on_skip_txfm");
  }
  if (options.cdef_on_skip_txfm == 1 && !options.on_skip_txfm) {
    return fail(error,
                "CdefOnSkipTxfm ALWAYS_ON requires on_skip_txfm");
  }

  CdefStageFrame view{};
  for (int plane = 0; plane < 3; ++plane) {
    view.plane[plane].src = frame.recon[plane].data;
    view.plane[plane].dst = nullptr;  // planning only
    view.plane[plane].stride = frame.recon[plane].stride;
    view.plane[plane].height =
        (frame.mi_rows * mx4()) >> plane_shift(plane);
    view.ref[plane] = frame.source[plane].data;
    view.ref_stride[plane] = frame.source[plane].stride;
  }
  view.width = frame.width;
  view.height = frame.height;
  view.mi_cols = frame.mi_cols;
  view.mi_rows = frame.mi_rows;
  view.skip = frame.skip;
  view.lossless = frame.lossless;
  view.lossless_chroma = frame.lossless_chroma;
  view.tile_col_starts = frame.tile_col_starts;
  view.num_tile_cols = frame.num_tile_cols;
  view.tile_row_starts = frame.tile_row_starts;
  view.num_tile_rows = frame.num_tile_rows;

  CdefStageOptions stage_options;
  stage_options.damping = options.damping;
  stage_options.num_strengths = options.num_strengths;
  stage_options.on_skip_txfm = options.on_skip_txfm;
  stage_options.disable_loopfilters_across_tiles =
      options.disable_loopfilters_across_tiles;
  stage_options.include_off = options.include_off;
  stage_options.chroma_weight = options.chroma_weight;
  stage_options.candidates = options.candidates;
  stage_options.candidate_count = options.candidate_count;

  CdefStagePlan stage_plan;
  if (!av2enc::cuda::still::still_cdef_plan(view, stage_options, &stage_plan,
                                      error)) {
    return false;
  }

  out->options = stage_plan.search.options;
  out->num_strengths = stage_plan.num_strengths;
  out->mi_cols = frame.mi_cols;
  out->mi_rows = frame.mi_rows;
  out->n64_cols = stage_plan.n64_cols;
  out->n64_rows = stage_plan.n64_rows;
  out->cdef_idx = stage_plan.cdef_idx;
  out->sse_before = stage_plan.sse_off_total;
  out->sse_after = stage_plan.sse_total;
  out->blocks_off = stage_plan.search.blocks_off;
  out->forced_single_strength = stage_plan.forced_single_strength;
  out->header_only_emittable = stage_plan.num_strengths == 1;
  out->params = make_params(out->options);

  // Header bits: measured by the shared §5.18.7.10 writer in the M2 context.
  {
    BitWriter counter;
    std::string header_error;
    const std::size_t bits = write_frame_header_cdef(
        counter, *out, /*single_picture_header=*/false,
        options.cdef_on_skip_txfm, &header_error);
    if (!header_error.empty()) {
      return fail(error, header_error);
    }
    out->header_bits = static_cast<int>(bits);
  }

  if (out->num_strengths == 1) {
    out->index_bits = 0;
    for (std::int8_t idx : out->cdef_idx) {
      if (idx != 0) {
        return fail(error, "single strength requires cdef_idx == 0");
      }
    }
  } else {
    std::vector<std::uint8_t> payload;
    std::string index_error;
    if (!write_index_symbols(*out, &payload, &index_error)) {
      return fail(error, index_error);
    }
    out->index_bits = static_cast<int>(payload.size() * 8U);
  }
  if (error != nullptr) error->clear();
  return true;
}

bool filter(const FrameView& frame, const Plan& plan,
            std::uint8_t* const dst[3], const int dst_stride[3],
            std::string* error) {
  if (!validate_view(frame, /*need_source=*/false, error)) return false;
  if (dst == nullptr || dst_stride == nullptr) {
    return fail(error, "null CDEF destination");
  }
  if (plan.num_strengths != plan.options.num_strengths ||
      plan.num_strengths < 1) {
    return fail(error, "plan is not initialised");
  }
  if (static_cast<int>(plan.cdef_idx.size()) !=
      plan.n64_cols * plan.n64_rows) {
    return fail(error, "plan index grid size mismatch");
  }

  av2enc::cdef::CdefFrame cf{};
  for (int plane = 0; plane < 3; ++plane) {
    if (dst[plane] == nullptr) return fail(error, "null destination plane");
    if (frame.recon[plane].data == dst[plane]) {
      return fail(error, "CDEF source and destination must not alias");
    }
    if (dst_stride[plane] <
        (frame.mi_cols * mx4()) >> plane_shift(plane)) {
      return fail(error, "short destination stride");
    }
    // cdef::CdefPlane has a single stride for source and destination.
    if (dst_stride[plane] != frame.recon[plane].stride) {
      return fail(error,
                  "CDEF needs one stride for source and destination");
    }
    cf.planes[plane].src = frame.recon[plane].data;
    cf.planes[plane].dst = dst[plane];
    cf.planes[plane].stride = dst_stride[plane];
  }
  cf.mi_cols = frame.mi_cols;
  cf.mi_rows = frame.mi_rows;
  cf.subsampling_x = 1;
  cf.subsampling_y = 1;
  cf.skip = frame.skip;
  cf.lossless = frame.lossless;
  cf.lossless_chroma = frame.lossless_chroma;
  cf.cdef_idx = plan.cdef_idx.data();
  cf.n64_cols = plan.n64_cols;
  cf.n64_rows = plan.n64_rows;
  cf.tile_col_starts = frame.tile_col_starts;
  cf.num_tile_cols = frame.num_tile_cols;
  cf.tile_row_starts = frame.tile_row_starts;
  cf.num_tile_rows = frame.num_tile_rows;

  CdefOptions options = plan.options;
  options.bit_depth = 8;
  const CdefStatus status = av2enc::cdef::cdef_frame(cf, options);
  if (status != CdefStatus::kOk) {
    return fail(error, std::string("CDEF failed: ") +
                           av2enc::cdef::cdef_status_string(status));
  }
  return true;
}

std::size_t write_frame_header_cdef(BitWriter& writer, const Plan& plan,
                                    bool single_picture_header,
                                    int cdef_on_skip_txfm,
                                    std::string* error) {
  if (plan.num_strengths != plan.params.num_strengths) {
    fail(error, "plan params disagree with the decoded strength count");
    return 0;
  }
  filters_stream::CdefHeaderContext context;
  context.coded_lossless = false;
  context.enable_cdef = true;
  context.single_picture_header_flag = single_picture_header;
  context.cdef_on_skip_txfm = single_picture_header ? 2 : cdef_on_skip_txfm;
  context.num_planes = 3;
  filters_stream::CdefParams params = plan.params;
  params.frame_enable = true;
  return filters_stream::write_cdef_params(writer, params, context, error);
}

std::size_t write_sequence_filter_config(BitWriter& writer, bool enable_cdef,
                                         bool on_skip_always_on,
                                         bool on_skip_disabled,
                                         std::string* error) {
  filters_stream::SequenceFilterConfig config;
  config.disable_loopfilters_across_tiles = false;
  config.enable_cdef = enable_cdef;
  config.enable_gdf = false;
  config.enable_restoration = false;
  config.enable_ccso = false;
  config.ccso_unit_matches_sb_size = true;
  config.cdef_on_skip_txfm_always_on = on_skip_always_on;
  config.cdef_on_skip_txfm_disabled = on_skip_disabled;
  config.df_par_bits_minus_2 = 0;
  return filters_stream::write_sequence_filter_config(
      writer, config, /*single_picture_header_flag=*/false, error);
}

bool TileIndexWriter::init(int mi_cols, int mi_rows, int strengths,
                           int sb_width_log2, std::string* error) {
  if (strengths < 1 || strengths > 8) {
    return fail(error, "CdefStrengths outside 1..8");
  }
  if (sb_width_log2 != 4 && sb_width_log2 != 5 && sb_width_log2 != 6) {
    return fail(error, "unsupported superblock size");
  }
  if (!filters_stream::cdef_tile_init(&state, mi_cols, mi_rows, strengths,
                                      sb_width_log2, error)) {
    return false;
  }
  return true;
}

void TileIndexWriter::clear_superblock(int mi_row, int mi_col) {
  filters_stream::cdef_clear_superblock(&state, mi_row, mi_col);
}

bool TileIndexWriter::write(av2::Encoder& encoder, int mi_row, int mi_col,
                            int idx, std::string* error) {
  return filters_stream::cdef_write_block(
      encoder, state, tile_mi_col_start, tile_mi_row_start, mi_row, mi_col,
      idx, error);
}

bool read_tile_index(av2::Decoder& decoder, TileIndexWriter* tile, int mi_row,
                     int mi_col, int* idx, std::string* error) {
  if (tile == nullptr || idx == nullptr) {
    return fail(error, "null tile reader");
  }
  return filters_stream::cdef_read_block(decoder, tile->state,
                                         tile->tile_mi_col_start,
                                         tile->tile_mi_row_start, mi_row, mi_col,
                                         idx, error);
}

bool write_index_symbols(const Plan& plan, std::vector<std::uint8_t>* payload,
                         std::string* error) {
  if (payload == nullptr) return fail(error, "null payload output");
  payload->clear();
  const int blocks = plan.n64_cols * plan.n64_rows;
  if (blocks <= 0) return fail(error, "empty index grid");
  if (plan.num_strengths == 1) {
    // §5.20.10.1 codes nothing with CdefStrengths == 1.  Return an empty
    // payload rather than a bare exit marker.
    return true;
  }

  const std::size_t capacity =
      256U + static_cast<std::size_t>(blocks) * 16U;
  std::vector<std::uint8_t> buffer(capacity, 0);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return fail(error, "index encoder init failed");
  }
  TileIndexWriter tile;
  if (!tile.init(plan.mi_cols, plan.mi_rows, plan.num_strengths, 4, error)) {
    return false;
  }
  for (int br = 0; br < plan.n64_rows; ++br) {
    for (int bc = 0; bc < plan.n64_cols; ++bc) {
      const int mi_row = br * cdef_size4();
      const int mi_col = bc * cdef_size4();
      // One clear per 64x64 superblock; M2's superblock is 64x64.
      tile.clear_superblock(mi_row, mi_col);
      const std::size_t at = static_cast<std::size_t>(br) *
                                 static_cast<std::size_t>(plan.n64_cols) +
                             static_cast<std::size_t>(bc);
      if (!tile.write(encoder, mi_row, mi_col,
                      static_cast<int>(plan.cdef_idx[at]), error)) {
        return false;
      }
    }
  }
  std::size_t bytes = 0;
  if (encoder.exit_symbol(&bytes) != av2::Status::kOk || bytes == 0) {
    return fail(error, "index encoder exit failed");
  }
  payload->assign(buffer.begin(), buffer.begin() +
                  static_cast<std::ptrdiff_t>(bytes));
  return true;
}

bool read_index_symbols(std::span<const std::uint8_t> payload,
                        const Plan& plan, std::vector<std::int8_t>* grid,
                        std::string* error) {
  if (grid == nullptr) return fail(error, "null grid output");
  grid->assign(static_cast<std::size_t>(plan.n64_cols) *
                   static_cast<std::size_t>(plan.n64_rows),
               static_cast<std::int8_t>(-1));
  if (plan.num_strengths == 1) {
    grid->assign(grid->size(), 0);
    return payload.empty() ? true : fail(error, "unexpected index payload");
  }
  if (payload.size() < 2) return fail(error, "index payload too short");
  av2::Decoder decoder;
  if (decoder.init_symbol(payload.data(), payload.size(), payload.size()) !=
      av2::Status::kOk) {
    return fail(error, "index decoder init failed");
  }
  TileIndexWriter tile;
  if (!tile.init(plan.mi_cols, plan.mi_rows, plan.num_strengths, 4, error)) {
    return false;
  }
  for (int br = 0; br < plan.n64_rows; ++br) {
    for (int bc = 0; bc < plan.n64_cols; ++bc) {
      const int mi_row = br * cdef_size4();
      const int mi_col = bc * cdef_size4();
      tile.clear_superblock(mi_row, mi_col);
      int idx = -1;
      if (!read_tile_index(decoder, &tile, mi_row, mi_col, &idx, error)) {
        return false;
      }
      (*grid)[static_cast<std::size_t>(br) *
                  static_cast<std::size_t>(plan.n64_cols) +
              static_cast<std::size_t>(bc)] =
          static_cast<std::int8_t>(idx);
    }
  }
  if (decoder.exit_symbol() != av2::Status::kOk) {
    return fail(error, "index decoder saw trailing bits");
  }
  return true;
}

bool build_mi_grids(int mi_cols, int mi_rows, const std::uint8_t* skip,
                    const std::uint8_t* segment_id,
                    const std::uint8_t* chroma_segment_id,
                    const std::uint8_t* lossless_array,
                    int lossless_array_size, FrameGrids* out,
                    std::string* error) {
  if (out == nullptr) return fail(error, "null grid output");
  if (mi_cols <= 0 || mi_rows <= 0) return fail(error, "invalid MI geometry");
  if ((segment_id != nullptr || chroma_segment_id != nullptr) &&
      (lossless_array == nullptr || lossless_array_size <= 0)) {
    return fail(error, "segment ids require a lossless array");
  }
  out->zero(mi_cols, mi_rows);
  const std::size_t count = static_cast<std::size_t>(mi_cols) *
                            static_cast<std::size_t>(mi_rows);
  for (std::size_t i = 0; i < count; ++i) {
    if (skip != nullptr) out->skip[i] = skip[i] != 0 ? 1 : 0;
    if (segment_id != nullptr) {
      const int seg = static_cast<int>(segment_id[i]);
      if (seg < 0 || seg >= lossless_array_size) {
        return fail(error, "segment id outside the lossless array");
      }
      out->lossless[i] = lossless_array[seg] != 0 ? 1 : 0;
    }
    if (chroma_segment_id != nullptr) {
      const int seg = static_cast<int>(chroma_segment_id[i]);
      if (seg < 0 || seg >= lossless_array_size) {
        return fail(error, "chroma segment id outside the lossless array");
      }
      out->lossless_chroma[i] = lossless_array[seg] != 0 ? 1 : 0;
    }
  }
  return true;
}

}  // namespace av2enc::video::cdef_adapter
