// §7.17-§7.20 in-loop filter plan/application for the 8-bit format still
// writer (4:2:2, 4:4:4, 4:0:0).
//
// The pixel arithmetic is the shared, depth-parameterised filter core:
//
//   §7.17 deblock          codec/filters/deblock/deblock_core.h
//   §7.18 CDEF             codec/filters/cdef/cdef_core.h
//   §7.19 CCSO             codec/filters/loop_restoration/ccso_core.h
//   §7.20 Wiener NS/PCW    codec/filters/loop_restoration/wiener_core.h
//   §7.20.5 GDF            codec/filters/tables/gdf.h
//
// The 8-bit samples are carried in the uint16_t container (BitDepth = 8).  The
// cores clip against `1 << BitDepth` and use `BitDepth - 8` for the CDEF tap
// parity, so the arithmetic is identical to an 8-bit container; the wider
// container is what lets the non-4:2:0 geometry through the generic entries
// (the uint8_t deblock/CDEF/CCSO entries are scoped to 4:2:0).
//
// The decisions reuse the 10-bit product writer's planners (`hbd_cdef.h`,
// `hbd_ccso.h`, `hbd_lr.h`, `hbd_gdf.h`), which take a BitDepth field.  Those
// headers only ever hold the plan structures; pixel buffers live here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "formats/format.h"
#include "formats/high_bitdepth/hbd_ccso.h"
#include "formats/high_bitdepth/hbd_cdef.h"
#include "formats/high_bitdepth/hbd_filter_config.h"
#include "formats/high_bitdepth/hbd_gdf.h"
#include "formats/high_bitdepth/hbd_lr.h"

namespace av2enc::chroma {

// One coded leaf of this writer: a 64x64 PARTITION_NONE block.  `tx_size[p]`
// is the §9.2 transform size index of the plane's coded transform (the
// `txv::TxSize` / `deblock::TxSize` numbering).  `luma_eob_zero` feeds
// §7.20.1 LrTxSkip (there are no skipped blocks in an intra still).
struct ChromaFilterBlock {
  int mi_row = 0;
  int mi_col = 0;
  int luma_size = 64;
  int tx_size[3] = {0, 0, 0};
  bool luma_eob_zero = false;
};

struct ChromaFilterOptions {
  int base_q_idx = 90;
  bool enable_deblock = false;
  bool enable_cdef = false;
  int cdef_damping = 3;
  int cdef_num_strengths = 4;
  bool enable_ccso = false;
  int ccso_luma_size_log2 = 6;
  bool ccso_edge_only = false;
  bool enable_wiener = false;
  bool enable_chroma_lr = false;
  bool enable_pc_wiener = false;
  bool enable_gdf = false;
};

// Syntax-visible decisions.  The planners and the apply stage share these so
// the header, the tile symbols and the filtered reconstruction cannot drift.
struct ChromaFilterPlan {
  bool deblock = false;
  av2enc::hbd::HbdCdefPlan cdef;
  av2enc::hbd::HbdCcsoPlan ccso;
  av2enc::hbd::HbdLrPlan lr;
  av2enc::hbd::HbdGdfPlan gdf;

  bool any() const {
    return deblock || cdef.enabled || ccso.config.enabled ||
           lr.config.enabled || gdf.config.enabled;
  }
};

// In-place filter chain over the padded 8-bit reconstruction.
//
//   plane[p]        padded reconstruction (stride `stride[p]`); replaced by the
//                   filtered output
//   source[p]       source plane for the SSE-driven decisions (`source_stride`
//                   is the padded plane stride); must cover visible_w/h
struct ChromaFilterPlanes {
  std::uint8_t* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  const std::uint8_t* source[3] = {nullptr, nullptr, nullptr};
  int source_stride[3] = {0, 0, 0};
  int visible_w[3] = {0, 0, 0};
  int visible_h[3] = {0, 0, 0};
};

// Validates the request against the format, plans deblock -> CDEF -> CCSO ->
// LR -> GDF and applies every enabled stage, in the normative order, to
// `planes->plane`.  Returns false with a precise reason (including the
// decoder-exactness constraints inherited from the 10-bit writer) and leaves
// the planes unspecified on failure.
bool run_chroma_filters(const fmt::Format& format,
                        const ChromaFilterPlanes& planes,
                        const std::vector<ChromaFilterBlock>& blocks,
                        const ChromaFilterOptions& options,
                        ChromaFilterPlan* plan, std::string& error);

}  // namespace av2enc::chroma
