// Host-only wavefront plan construction for the M1 block pipeline.
//
// Nothing here touches CUDA.  The dependency model, level assignment, TX class
// spans and launch accounting are unit-tested in sched_plan_test.cpp and are
// usable on a machine without a GPU.
//
// The two non-obvious pieces are:
//
//   1. availability replay.  The CPU computes num4AboveRight/num4BelowLeft by
//      walking `decoded_` with clear_block_decoded_flags() semantics.  We run
//      the same bookkeeping over the block list (4-sample cells for every
//      plane, SB boundary presets) so the graph contains the reads the kernels
//      will really perform instead of a blanket "one block up and left" edge;
//
//   2. the chroma cross-plane edge.  CfL reads reconstructed luma over the
//      co-located leaf plus one filtered line above/left (§7.13.5), and the UV
//      mode decision consumes the luma decision.  The plan adds an edge from
//      every luma block that owns a sample in that region.  It is conservative
//      for a DC-only chroma block (never selects a wrong level) and exact for
//      CfL, where getting it wrong desyncs every following chroma prediction.

#include "cuda/m1/sched/sched.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::cuda::m1::sched {
namespace {

// All planes use 4-sample coverage cells.  Luma MI cells are 4 luma samples;
// the encoder's chroma MI accounting (count_top_right / count_bottom_left in
// m1_still.cpp) indexes 4 chroma samples, i.e. 2 luma MI cells, which is why
// the chroma map is mi_cols/2 wide rather than mi_cols wide.
constexpr int kCell = 4;
constexpr int kSbCells = 16;       // 64 luma samples / 4
constexpr int kSbCellsChroma = 8;  // 64 luma samples / 4 chroma samples

std::uint64_t fnv1a(std::uint64_t hash, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

bool is_pow2(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// Small per-plane coverage grid.  -1 = unowned.
struct Cover {
  int cols = 0;
  int rows = 0;
  std::vector<int> cells;

  void reset(int c, int r) {
    cols = c;
    rows = r;
    cells.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(r), -1);
  }
  bool inside(int x, int y) const {
    return x >= 0 && y >= 0 && x < cols && y < rows;
  }
  int at(int x, int y) const {
    if (!inside(x, y)) {
      return -1;
    }
    return cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(x)];
  }
  void put(int x, int y, int value) {
    cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
          static_cast<std::size_t>(x)] = value;
  }
};

// Per-SB decoded flags, replicated from the CPU encoder.  V shares the chroma
// map with U because the encoder marks both planes for every chroma block.
struct DecodedMaps {
  std::vector<std::uint8_t> luma;
  std::vector<std::uint8_t> chroma;
  int luma_cols = 0;
  int luma_rows = 0;
  int chroma_cols = 0;
  int chroma_rows = 0;

  void reset(int lc, int lr, int cc, int cr) {
    luma_cols = lc;
    luma_rows = lr;
    chroma_cols = cc;
    chroma_rows = cr;
    luma.assign(static_cast<std::size_t>(lc) * static_cast<std::size_t>(lr), 0);
    chroma.assign(static_cast<std::size_t>(cc) * static_cast<std::size_t>(cr), 0);
  }
  void clear() {
    std::fill(luma.begin(), luma.end(), 0);
    std::fill(chroma.begin(), chroma.end(), 0);
  }
  bool luma_at(int x, int y) const {
    if (x < 0 || y < 0 || x >= luma_cols || y >= luma_rows) {
      return false;
    }
    return luma[static_cast<std::size_t>(y) * static_cast<std::size_t>(luma_cols) +
                static_cast<std::size_t>(x)] != 0;
  }
  bool chroma_at(int x, int y) const {
    if (x < 0 || y < 0 || x >= chroma_cols || y >= chroma_rows) {
      return false;
    }
    return chroma[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(chroma_cols) +
                  static_cast<std::size_t>(x)] != 0;
  }
  void mark_luma(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < luma_cols &&
            y + j < luma_rows) {
          luma[static_cast<std::size_t>(y + j) *
                   static_cast<std::size_t>(luma_cols) +
               static_cast<std::size_t>(x + i)] = 1;
        }
      }
    }
  }
  void mark_chroma(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < chroma_cols &&
            y + j < chroma_rows) {
          chroma[static_cast<std::size_t>(y + j) *
                     static_cast<std::size_t>(chroma_cols) +
                 static_cast<std::size_t>(x + i)] = 1;
        }
      }
    }
  }
};

// §7.13.2.1 count_top_right, exact CPU form (m1_still.cpp).
int count_above_right(const DecodedMaps& maps, int plane, int cell_x,
                      int cell_y, int cells_w, int sb_cx, int sb_cy,
                      int plane_cols) {
  const bool luma = (plane == 0);
  const int sb_w = luma ? std::min(kSbCells, plane_cols - sb_cx)
                        : std::min(kSbCellsChroma, plane_cols - sb_cx);
  const int border_w =
      luma ? std::min(2 * kSbCells, plane_cols - sb_cx)
           : std::min(2 * kSbCellsChroma, plane_cols - sb_cx);
  const int rel_x = cell_x - sb_cx;
  const int rel_y = cell_y - sb_cy;
  const int limit = (rel_y == 0) ? border_w : sb_w;
  int count = 0;
  for (int i = 0; i < cells_w; ++i) {
    const int x = rel_x + cells_w + i;
    if (x >= limit) {
      break;
    }
    if (rel_y - 1 >= 0) {
      const bool decoded = luma ? maps.luma_at(sb_cx + x, sb_cy + rel_y - 1)
                                : maps.chroma_at(sb_cx + x, sb_cy + rel_y - 1);
      if (!decoded) {
        break;
      }
    }
    ++count;
  }
  return count;
}

// §7.13.2.1 count_bottom_left, exact CPU form.
int count_below_left(const DecodedMaps& maps, int plane, int cell_x,
                     int cell_y, int cells_h, int sb_cx, int sb_cy,
                     int plane_rows) {
  const bool luma = (plane == 0);
  const int sb_h = luma ? std::min(kSbCells, plane_rows - sb_cy)
                        : std::min(kSbCellsChroma, plane_rows - sb_cy);
  const int rel_x = cell_x - sb_cx;
  const int rel_y = cell_y - sb_cy;
  int count = 0;
  for (int i = 0; i < cells_h; ++i) {
    const int y = rel_y + cells_h + i;
    if (y >= sb_h) {
      break;
    }
    if (rel_x - 1 >= 0) {
      const bool decoded = luma ? maps.luma_at(sb_cx + rel_x - 1, sb_cy + y)
                                : maps.chroma_at(sb_cx + rel_x - 1, sb_cy + y);
      if (!decoded) {
        break;
      }
    }
    ++count;
  }
  return count;
}

bool validate(const FramePlan& plan, std::string& error) {
  if (plan.width <= 0 || plan.height <= 0) {
    error = "frame has an empty picture";
    return false;
  }
  if ((plan.width & 1) != 0 || (plan.height & 1) != 0) {
    error = "M1 4:2:0 requires even dimensions";
    return false;
  }
  const int expect_cw = (plan.width + 1) / 2;
  const int expect_ch = (plan.height + 1) / 2;
  if (plan.chroma_width != expect_cw || plan.chroma_height != expect_ch) {
    error = "chroma geometry is not 4:2:0";
    return false;
  }
  if (plan.mi_cols != 2 * ((plan.width + 7) >> 3) ||
      plan.mi_rows != 2 * ((plan.height + 7) >> 3)) {
    error = "mi_cols/mi_rows do not match the M1 derivation";
    return false;
  }
  const int coded_w = plan.mi_cols * 4;
  const int coded_h = plan.mi_rows * 4;
  if (plan.padded_luma_width < coded_w || plan.padded_luma_height < coded_h ||
      plan.padded_luma_width % 64 != 0 || plan.padded_luma_height % 64 != 0) {
    error = "padded luma geometry must be 64-aligned and cover MiCols*4";
    return false;
  }
  if (plan.padded_chroma_width != plan.padded_luma_width / 2 ||
      plan.padded_chroma_height != plan.padded_luma_height / 2) {
    error = "padded chroma geometry must be half the padded luma geometry";
    return false;
  }
  if (plan.block_count < 0 ||
      (plan.block_count > 0 && plan.blocks == nullptr)) {
    error = "invalid block list";
    return false;
  }
  for (int i = 0; i < plan.block_count; ++i) {
    const BlockPlan& b = plan.blocks[i];
    const std::string where = "block " + std::to_string(i);
    if (b.mi_row < 0 || b.mi_col < 0) {
      error = where + " has a negative position";
      return false;
    }
    const int x = b.mi_col * 4;
    const int y = b.mi_row * 4;
    if (b.luma_w < 4 || b.luma_h < 4 || !is_pow2(b.luma_w) ||
        !is_pow2(b.luma_h) || b.luma_w > 64 || b.luma_h > 64) {
      error = where + " has an unsupported luma size";
      return false;
    }
    // The decoder reconstructs whole blocks into the 64-aligned padded plane
    // even when a bottom/right-edge leaf extends past MiCols*4 / MiRows*4.
    if (x + b.luma_w > plan.padded_luma_width ||
        y + b.luma_h > plan.padded_luma_height) {
      error = where + " extends past the padded luma plane";
      return false;
    }
    if (b.has_chroma) {
      if (b.chroma_w != b.luma_w / 2 || b.chroma_h != b.luma_h / 2) {
        error = where + " chroma size is not half the luma size";
        return false;
      }
      if (b.chroma_w < 4 || b.chroma_h < 4 || b.chroma_w % 4 != 0 ||
          b.chroma_h % 4 != 0) {
        error = where + " has an unsupported chroma size";
        return false;
      }
      if ((b.chroma_w & (b.chroma_w - 1)) != 0 ||
          (b.chroma_h & (b.chroma_h - 1)) != 0) {
        error = where + " chroma size is not a power of two";
        return false;
      }
      if (b.mi_col * 2 + b.chroma_w > plan.padded_chroma_width ||
          b.mi_row * 2 + b.chroma_h > plan.padded_chroma_height) {
        error = where + " extends past the padded chroma plane";
        return false;
      }
    }
    if (b.luma_mode < 0 || b.luma_mode > 12 || b.chroma_mode < 0 ||
        b.chroma_mode > 12 || b.luma_angle < -3 || b.luma_angle > 3 ||
        b.chroma_angle < -3 || b.chroma_angle > 3) {
      error = where + " has an out-of-range mode/angle";
      return false;
    }
  }
  return true;
}

}  // namespace

const char* plane_name(int plane) {
  switch (plane) {
    case kPlaneY:
      return "Y";
    case kPlaneU:
      return "U";
    case kPlaneV:
      return "V";
    default:
      return "?";
  }
}

void Plan::clear() {
  input_ = {};
  owned_blocks_.clear();
  jobs_.clear();
  spans_.clear();
  level_begin_.clear();
  span_begin_.clear();
  dep_begin_.clear();
  dep_indices_.clear();
  levels_ = 0;
  total_coeffs_ = 0;
  signature_ = 0;
  counters_ = {};
  error_.clear();
}

bool Plan::build(const FramePlan& plan, std::string& error) {
  clear();
  if (!validate(plan, error)) {
    error_ = error;
    return false;
  }
  if (plan.block_count > 0) {
    owned_blocks_.assign(plan.blocks, plan.blocks + plan.block_count);
  }
  input_ = plan;
  if (!owned_blocks_.empty()) {
    input_.blocks = owned_blocks_.data();
    input_.block_count = static_cast<int>(owned_blocks_.size());
  }
  const FramePlan& p = input_;

  // -------------------------------------------------------------------------
  // 1. Expand leaves into Y/U/V jobs in CPU syntax order and assign the
  //    canonical coefficient layout (Y, U, V per leaf).
  // -------------------------------------------------------------------------
  std::vector<int> block_job_begin(static_cast<std::size_t>(p.block_count) + 1,
                                   0);
  for (int i = 0; i < p.block_count; ++i) {
    const BlockPlan& b = p.blocks[i];
    block_job_begin[static_cast<std::size_t>(i)] =
        static_cast<int>(jobs_.size());
    const auto push = [&](int plane, int w, int h) {
      PlaneJob j;
      j.plane = plane;
      j.block = i;
      j.syntax = i;
      j.x = (plane == 0) ? b.mi_col * 4 : b.mi_col * 2;
      j.y = (plane == 0) ? b.mi_row * 4 : b.mi_row * 2;
      j.w = w;
      j.h = h;
      // §7.15.4: the adjusted transform size caps each coded dimension at 32
      // (64x64 is coded as an adjusted 32x32 transform).
      j.tx_w = std::min(w, 32);
      j.tx_h = std::min(h, 32);
      j.max_x = (plane == 0) ? p.mi_cols * 4 - 1 : p.mi_cols * 2 - 1;
      j.max_y = (plane == 0) ? p.mi_rows * 4 - 1 : p.mi_rows * 2 - 1;
      j.pred_index = 3 * i + plane;
      j.is_cfl = (plane == 0) ? 0 : b.is_cfl;
      j.mode = (plane == 0) ? b.luma_mode
                            : (b.is_cfl ? 0 : b.chroma_mode);
      j.angle = (plane == 0) ? b.luma_angle
                             : (b.is_cfl ? 0 : b.chroma_angle);
      j.luma_mode = b.luma_mode;
      j.luma_angle = b.luma_angle;
      j.coeff_offset = static_cast<int>(total_coeffs_);
      total_coeffs_ += static_cast<std::int64_t>(w) * static_cast<std::int64_t>(h);
      jobs_.push_back(j);
    };
    push(kPlaneY, b.luma_w, b.luma_h);
    if (b.has_chroma) {
      push(kPlaneU, b.chroma_w, b.chroma_h);
      push(kPlaneV, b.chroma_w, b.chroma_h);
    }
  }
  block_job_begin[static_cast<std::size_t>(p.block_count)] =
      static_cast<int>(jobs_.size());

  const int n = static_cast<int>(jobs_.size());
  counters_.coding_blocks = p.block_count;
  counters_.plane_jobs = n;
  if (n == 0) {
    levels_ = 0;
    level_begin_.assign(1, 0);
    span_begin_.assign(1, 0);
    dep_begin_.assign(1, 0);
    return true;
  }

  for (const PlaneJob& j : jobs_) {
    if ((j.plane == 0 ? j.w : j.w) % kCell != 0 ||
        (j.plane == 0 ? j.h : j.h) % kCell != 0 || j.x % kCell != 0 ||
        j.y % kCell != 0) {
      error = "plane block is not aligned to the 4-sample coverage grid";
      clear();
      error_ = error;
      return false;
    }
    if (j.plane == 0) {
      ++counters_.luma_jobs;
    } else {
      ++counters_.chroma_jobs;
      if (j.is_cfl != 0) {
        ++counters_.cfl_jobs;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 2. Coverage maps and exact availability replay for the directional
  //    above-right / below-left extensions.
  // -------------------------------------------------------------------------
  // Availability/decoded accounting is in the coded MiCols x MiRows grid (the
  // encoder's arrays), while coverage must also hold edge blocks that write
  // into the 64-aligned padded plane.
  const int luma_cols = p.mi_cols;
  const int luma_rows = p.mi_rows;
  const int chroma_cols = p.mi_cols / 2;
  const int chroma_rows = p.mi_rows / 2;
  const int cover_luma_cols = p.padded_luma_width / kCell;
  const int cover_luma_rows = p.padded_luma_height / kCell;
  const int cover_chroma_cols = p.padded_chroma_width / kCell;
  const int cover_chroma_rows = p.padded_chroma_height / kCell;
  Cover covers[3];
  covers[0].reset(cover_luma_cols, cover_luma_rows);
  covers[1].reset(cover_chroma_cols, cover_chroma_rows);
  covers[2].reset(cover_chroma_cols, cover_chroma_rows);

  for (int i = 0; i < n; ++i) {
    PlaneJob& j = jobs_[static_cast<std::size_t>(i)];
    Cover& cover = covers[j.plane];
    const int cx = j.x / kCell;
    const int cy = j.y / kCell;
    const int cw = j.w / kCell;
    const int ch = j.h / kCell;
    if (cx + cw > cover.cols || cy + ch > cover.rows) {
      error = std::string("plane ") + plane_name(j.plane) +
              " block extends past the padded plane";
      clear();
      error_ = error;
      return false;
    }
    for (int yy = 0; yy < ch; ++yy) {
      for (int xx = 0; xx < cw; ++xx) {
        if (cover.at(cx + xx, cy + yy) >= 0) {
          error = std::string("plane ") + plane_name(j.plane) +
                  " blocks overlap at cell (" + std::to_string(cx + xx) + "," +
                  std::to_string(cy + yy) + ")";
          clear();
          error_ = error;
          return false;
        }
        cover.put(cx + xx, cy + yy, i);
      }
    }
  }
  // The leaf list must tile the whole coded plane; otherwise a later
  // prediction could read a cell that no job reconstructs.
  const auto check_tiled = [&](const Cover& cover, int coded_cells_x,
                               int coded_cells_y, const char* name) {
    for (int y = 0; y < coded_cells_y; ++y) {
      for (int x = 0; x < coded_cells_x; ++x) {
        if (cover.at(x, y) < 0) {
          error = std::string(name) +
                  " coded plane is not fully covered by the leaf list";
          return false;
        }
      }
    }
    return true;
  };
  for (int plane = 0; plane < 3; ++plane) {
    bool used = false;
    for (const PlaneJob& j : jobs_) {
      if (j.plane == plane) {
        used = true;
        break;
      }
    }
    if (used && !check_tiled(covers[plane],
                             plane == 0 ? luma_cols : chroma_cols,
                             plane == 0 ? luma_rows : chroma_rows,
                             plane == 0 ? "luma" : plane_name(plane))) {
      clear();
      error_ = error;
      return false;
    }
  }

  DecodedMaps maps;
  maps.reset(luma_cols, luma_rows, chroma_cols, chroma_rows);
  int cur_sb_row = -1;
  int cur_sb_col = -1;

  for (int bi = 0; bi < p.block_count; ++bi) {
    const BlockPlan& b = p.blocks[bi];
    const int sb_row = (b.mi_row / kSbCells) * kSbCells;
    const int sb_col = (b.mi_col / kSbCells) * kSbCells;
    if (sb_row != cur_sb_row || sb_col != cur_sb_col) {
      maps.clear();
      cur_sb_row = sb_row;
      cur_sb_col = sb_col;
    }
    const int begin = block_job_begin[static_cast<std::size_t>(bi)];
    const int end = block_job_begin[static_cast<std::size_t>(bi) + 1];
    // Y job is always present; chroma jobs follow when has_chroma.
    for (int ji = begin; ji < end; ++ji) {
      PlaneJob& j = jobs_[static_cast<std::size_t>(ji)];
      if (j.plane == 0) {
        const int cell_x = b.mi_col;
        const int cell_y = b.mi_row;
        j.n4_above_right = count_above_right(
            maps, 0, cell_x, cell_y, j.w / kCell, sb_col, sb_row, luma_cols);
        j.n4_below_left = count_below_left(
            maps, 0, cell_x, cell_y, j.h / kCell, sb_col, sb_row, luma_rows);
      } else {
        const int cell_x = b.mi_col / 2;
        const int cell_y = b.mi_row / 2;
        const int sb_c = sb_col / 2;
        const int sb_r = sb_row / 2;
        j.n4_above_right =
            count_above_right(maps, 1, cell_x, cell_y, j.w / kCell, sb_c, sb_r,
                              chroma_cols);
        j.n4_below_left =
            count_below_left(maps, 1, cell_x, cell_y, j.h / kCell, sb_c, sb_r,
                             chroma_rows);
      }
    }
    // Mark the block's cells exactly like the CPU after coding it.
    const int lx = b.mi_col;
    const int ly = b.mi_row;
    maps.mark_luma(lx, ly, b.luma_w / kCell, b.luma_h / kCell);
    if (b.has_chroma) {
      maps.mark_chroma(b.mi_col / 2, b.mi_row / 2, b.chroma_w / kCell,
                       b.chroma_h / kCell);
    }
  }

  // -------------------------------------------------------------------------
  // 3. Dependency edges.
  // -------------------------------------------------------------------------
  std::vector<std::vector<int>> deps(static_cast<std::size_t>(n));
  const auto add_edge = [](std::vector<int>& out, int id) {
    if (id < 0) {
      return;
    }
    if (std::find(out.begin(), out.end(), id) == out.end()) {
      out.push_back(id);
    }
  };
  const auto add_cover = [&](std::vector<int>& out, const Cover& cover,
                             int cx, int cy) {
    add_edge(out, cover.at(cx, cy));
  };

  for (int i = 0; i < n; ++i) {
    const PlaneJob& j = jobs_[static_cast<std::size_t>(i)];
    const Cover& cover = covers[j.plane];
    const int cx = j.x / kCell;
    const int cy = j.y / kCell;
    const int cw = j.w / kCell;
    const int ch = j.h / kCell;
    std::vector<int>& out = deps[static_cast<std::size_t>(i)];
    if (j.x > 0) {
      for (int yy = cy; yy < cy + ch; ++yy) {
        add_cover(out, cover, cx - 1, yy);
      }
    }
    if (j.y > 0) {
      for (int xx = cx; xx < cx + cw; ++xx) {
        add_cover(out, cover, xx, cy - 1);
      }
    }
    if (j.x > 0 && j.y > 0) {
      add_cover(out, cover, cx - 1, cy - 1);  // above/left corner
    }
    if (j.y > 0) {
      for (int e = 0; e < j.n4_above_right; ++e) {
        add_cover(out, cover, cx + cw + e, cy - 1);
      }
    }
    if (j.x > 0) {
      for (int e = 0; e < j.n4_below_left; ++e) {
        add_cover(out, cover, cx - 1, cy + ch + e);
      }
    }
  }

  // Chroma cross-plane dependencies: everything CfL / the UV decision reads
  // from reconstructed luma.  sample_luma() covers [2x, 2x+2w) x [2y, 2y+2h);
  // the derived-alpha lines add one filtered row/column above-left, clamped to
  // the enclosing superblock's top row.  The region is expanded by the filter
  // taps (3 samples) to stay conservative under any ds_filter.
  for (int i = 0; i < n; ++i) {
    const PlaneJob& j = jobs_[static_cast<std::size_t>(i)];
    if (j.plane == 0) {
      continue;
    }
    const int sb_top = ((j.y * 2) / 64) * 64;  // luma row of the SB top
    int x_lo = 2 * j.x - 3;
    int x_hi = 2 * (j.x + j.w) - 1;
    int y_lo = std::max(sb_top - 1, 2 * j.y - 3);
    int y_hi = 2 * (j.y + j.h) - 1;
    x_lo = std::max(0, x_lo);
    y_lo = std::max(0, y_lo);
    x_hi = std::min(luma_cols * kCell - 1, x_hi);
    y_hi = std::min(luma_rows * kCell - 1, y_hi);
    std::vector<int>& out = deps[static_cast<std::size_t>(i)];
    for (int yy = y_lo / kCell; yy <= y_hi / kCell; ++yy) {
      for (int xx = x_lo / kCell; xx <= x_hi / kCell; ++xx) {
        add_cover(out, covers[0], xx, yy);
      }
    }
  }

  // -------------------------------------------------------------------------
  // 4. Wavefront levels (longest path over the DAG).
  // -------------------------------------------------------------------------
  std::vector<int> indegree(static_cast<std::size_t>(n), 0);
  std::vector<std::vector<int>> consumers(static_cast<std::size_t>(n));
  long long edge_count = 0;
  for (int i = 0; i < n; ++i) {
    for (int d : deps[static_cast<std::size_t>(i)]) {
      ++edge_count;
      consumers[static_cast<std::size_t>(d)].push_back(i);
      ++indegree[static_cast<std::size_t>(i)];
    }
  }
  counters_.dependency_edges = edge_count;

  // CPU order is a topological order: every edge must go from an earlier
  // syntax position (same leaf: from Y to U/V).
  for (int i = 0; i < n; ++i) {
    const PlaneJob& ji = jobs_[static_cast<std::size_t>(i)];
    for (int d : deps[static_cast<std::size_t>(i)]) {
      const PlaneJob& jd = jobs_[static_cast<std::size_t>(d)];
      const bool ordered = (jd.syntax < ji.syntax) ||
                           (jd.syntax == ji.syntax && jd.plane < ji.plane);
      if (!ordered) {
        ++counters_.syntax_order_violations;
      }
    }
  }

  std::vector<int> level(static_cast<std::size_t>(n), 0);
  std::vector<int> ready;
  ready.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (indegree[static_cast<std::size_t>(i)] == 0) {
      ready.push_back(i);
    }
  }
  int processed = 0;
  while (!ready.empty()) {
    const int u = ready.back();
    ready.pop_back();
    ++processed;
    for (int v : consumers[static_cast<std::size_t>(u)]) {
      int& lv = level[static_cast<std::size_t>(v)];
      lv = std::max(lv, level[static_cast<std::size_t>(u)] + 1);
      if (--indegree[static_cast<std::size_t>(v)] == 0) {
        ready.push_back(v);
      }
    }
  }
  if (processed != n) {
    counters_.cyclic = true;
    error = "M1 block dependency graph contains a cycle";
    clear();
    error_ = error;
    return false;
  }
  int max_level = 0;
  for (int i = 0; i < n; ++i) {
    jobs_[static_cast<std::size_t>(i)].level = level[static_cast<std::size_t>(i)];
    max_level = std::max(max_level, level[static_cast<std::size_t>(i)]);
  }
  levels_ = max_level + 1;
  counters_.levels = levels_;

  // Defensive: every edge must cross a level boundary.
  for (int i = 0; i < n; ++i) {
    for (int d : deps[static_cast<std::size_t>(i)]) {
      if (jobs_[static_cast<std::size_t>(d)].level >=
          jobs_[static_cast<std::size_t>(i)].level) {
        ++counters_.dependency_violations;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 5. TX classes: (tx_w, tx_h) selects make_tx_params() inputs.  M1 keeps
  //    tx_type = DCT_DCT, so coded size identifies the class.
  // -------------------------------------------------------------------------
  std::vector<std::pair<int, int>> class_size;
  for (PlaneJob& j : jobs_) {
    int cls = -1;
    for (int c = 0; c < static_cast<int>(class_size.size()); ++c) {
      if (class_size[static_cast<std::size_t>(c)].first == j.w &&
          class_size[static_cast<std::size_t>(c)].second == j.h) {
        cls = c;
        break;
      }
    }
    if (cls < 0) {
      cls = static_cast<int>(class_size.size());
      class_size.emplace_back(j.w, j.h);
    }
    j.cls = cls;
  }
  counters_.transform_classes = static_cast<int>(class_size.size());

  // -------------------------------------------------------------------------
  // 6. Order: level major, then TX class, then CPU syntax order.  Edges map
  //    to strictly smaller array indices, so this order is topological too.
  // -------------------------------------------------------------------------
  std::vector<int> order(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    order[static_cast<std::size_t>(i)] = i;
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const PlaneJob& ja = jobs_[static_cast<std::size_t>(a)];
    const PlaneJob& jb = jobs_[static_cast<std::size_t>(b)];
    if (ja.level != jb.level) {
      return ja.level < jb.level;
    }
    if (ja.cls != jb.cls) {
      return ja.cls < jb.cls;
    }
    if (ja.syntax != jb.syntax) {
      return ja.syntax < jb.syntax;
    }
    return ja.plane < jb.plane;
  });

  std::vector<PlaneJob> sorted;
  sorted.reserve(static_cast<std::size_t>(n));
  std::vector<int> new_index(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    new_index[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] = i;
    sorted.push_back(jobs_[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]);
  }
  jobs_ = std::move(sorted);

  // Rebuild the CSR dependencies in the new order.
  std::vector<int> counts(static_cast<std::size_t>(n), 0);
  long long total_deps = 0;
  for (int i = 0; i < n; ++i) {
    for (int d : deps[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]) {
      (void)d;
      ++counts[static_cast<std::size_t>(i)];
      ++total_deps;
    }
  }
  dep_begin_.assign(static_cast<std::size_t>(n) + 1u, 0);
  dep_indices_.clear();
  dep_indices_.reserve(static_cast<std::size_t>(total_deps));
  for (int i = 0; i < n; ++i) {
    dep_begin_[static_cast<std::size_t>(i)] = static_cast<int>(dep_indices_.size());
    const std::vector<int>& src =
        deps[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    std::vector<int> mapped;
    mapped.reserve(src.size());
    for (int d : src) {
      mapped.push_back(new_index[static_cast<std::size_t>(d)]);
    }
    std::sort(mapped.begin(), mapped.end());
    for (int d : mapped) {
      dep_indices_.push_back(d);
    }
  }
  dep_begin_[static_cast<std::size_t>(n)] = static_cast<int>(dep_indices_.size());
  counters_.dependency_violations = 0;
  for (int i = 0; i < n; ++i) {
    for (int e = dep_begin_[static_cast<std::size_t>(i)];
         e < dep_begin_[static_cast<std::size_t>(i) + 1]; ++e) {
      const int d = dep_indices_[static_cast<std::size_t>(e)];
      if (jobs_[static_cast<std::size_t>(d)].level >=
          jobs_[static_cast<std::size_t>(i)].level) {
        ++counters_.dependency_violations;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 7. Offsets, class spans, level ranges.
  // -------------------------------------------------------------------------
  long long buf_elems = 0;
  long long adj_elems = 0;
  for (PlaneJob& j : jobs_) {
    j.buf_offset = static_cast<int>(buf_elems);
    j.adj_offset = static_cast<int>(adj_elems);
    buf_elems += static_cast<long long>(j.w) * j.h;
    adj_elems += static_cast<long long>(j.tx_w) * j.tx_h;
  }

  spans_.clear();
  for (int i = 0; i < n; ++i) {
    const PlaneJob& j = jobs_[static_cast<std::size_t>(i)];
    if (spans_.empty() || spans_.back().level != j.level ||
        spans_.back().cls != j.cls) {
      ClassSpan span;
      span.level = j.level;
      span.cls = j.cls;
      span.job_begin = i;
      span.job_end = i + 1;
      spans_.push_back(span);
    } else {
      spans_.back().job_end = i + 1;
    }
  }
  counters_.groups = static_cast<int>(spans_.size());

  level_begin_.assign(static_cast<std::size_t>(levels_) + 1u, n);
  span_begin_.assign(static_cast<std::size_t>(levels_) + 1u, 0);
  {
    int cursor = 0;
    for (int lv = 0; lv < levels_; ++lv) {
      level_begin_[static_cast<std::size_t>(lv)] = cursor;
      while (cursor < n && jobs_[static_cast<std::size_t>(cursor)].level == lv) {
        ++cursor;
      }
    }
    level_begin_[static_cast<std::size_t>(levels_)] = cursor;
    int scursor = 0;
    for (int lv = 0; lv < levels_; ++lv) {
      span_begin_[static_cast<std::size_t>(lv)] = scursor;
      while (scursor < static_cast<int>(spans_.size()) &&
             spans_[static_cast<std::size_t>(scursor)].level == lv) {
        ++scursor;
      }
    }
    span_begin_[static_cast<std::size_t>(levels_)] = scursor;
  }

  for (int lv = 0; lv < levels_; ++lv) {
    counters_.max_jobs_per_level = std::max(
        counters_.max_jobs_per_level, level_end(lv) - level_begin(lv));
  }

  // -------------------------------------------------------------------------
  // 8. Launch accounting.
  // -------------------------------------------------------------------------
  long long scheduled = 0;
  long long naive = 0;
  for (const PlaneJob& j : jobs_) {
    naive += 4;  // predict, forward, inverse, reconstruct
    if (j.w != j.tx_w || j.h != j.tx_h) {
      ++naive;  // downsample the coded residual to the adjusted size
    }
    if (j.is_cfl != 0) {
      naive += 4;  // DC predict, stage into plane, CfL, stage back
    }
  }
  for (int lv = 0; lv < levels_; ++lv) {
    bool predict = false;  // luma and/or non-CfL chroma share one batch
    bool cfl = false;
    for (int i = level_begin(lv); i < level_end(lv); ++i) {
      const PlaneJob& j = jobs_[static_cast<std::size_t>(i)];
      if (j.plane == 0 || j.is_cfl == 0) {
        predict = true;
      } else {
        cfl = true;
      }
    }
    if (!predict && !cfl) {
      continue;
    }
    scheduled += (predict ? 1 : 0);
    if (cfl) {
      scheduled += 5;  // DC predict + stage + CfL + stage back + residual
    }
    for (int s = span_begin_[static_cast<std::size_t>(lv)];
         s < span_begin_[static_cast<std::size_t>(lv) + 1]; ++s) {
      const ClassSpan& span = spans_[static_cast<std::size_t>(s)];
      const PlaneJob& j = jobs_[static_cast<std::size_t>(span.job_begin)];
      if (j.w != j.tx_w || j.h != j.tx_h) {
        ++scheduled;  // one downsample per scaled (level, class) span
      }
    }
    scheduled += 3;  // forward multi, inverse multi, batched reconstruct
  }
  counters_.scheduled_kernel_launches = scheduled;
  counters_.naive_block_calls = naive;

  // -------------------------------------------------------------------------
  // 9. Signature.
  // -------------------------------------------------------------------------
  std::uint64_t hash = 1469598103934665603ull;
  hash = fnv1a(hash, &p.width, sizeof(p.width));
  hash = fnv1a(hash, &p.height, sizeof(p.height));
  hash = fnv1a(hash, &p.mi_cols, sizeof(p.mi_cols));
  hash = fnv1a(hash, &p.mi_rows, sizeof(p.mi_rows));
  hash = fnv1a(hash, &p.block_count, sizeof(p.block_count));
  for (int i = 0; i < p.block_count; ++i) {
    const BlockPlan& b = p.blocks[i];
    hash = fnv1a(hash, &b.mi_row, sizeof(b.mi_row));
    hash = fnv1a(hash, &b.mi_col, sizeof(b.mi_col));
    hash = fnv1a(hash, &b.luma_w, sizeof(b.luma_w));
    hash = fnv1a(hash, &b.luma_h, sizeof(b.luma_h));
    hash = fnv1a(hash, &b.chroma_w, sizeof(b.chroma_w));
    hash = fnv1a(hash, &b.chroma_h, sizeof(b.chroma_h));
    hash = fnv1a(hash, &b.has_chroma, sizeof(b.has_chroma));
    hash = fnv1a(hash, &b.luma_mode, sizeof(b.luma_mode));
    hash = fnv1a(hash, &b.luma_angle, sizeof(b.luma_angle));
    hash = fnv1a(hash, &b.chroma_mode, sizeof(b.chroma_mode));
    hash = fnv1a(hash, &b.chroma_angle, sizeof(b.chroma_angle));
    hash = fnv1a(hash, &b.is_cfl, sizeof(b.is_cfl));
  }
  signature_ = hash;
  return true;
}

Plan build_m1_plan(const FramePlan& plan) {
  Plan out;
  std::string error;
  out.build(plan, error);
  return out;
}

int Plan::level_begin(int level) const {
  if (level < 0 || level > levels_) {
    return 0;
  }
  return level_begin_[static_cast<std::size_t>(level)];
}

int Plan::level_end(int level) const { return level_begin(level + 1); }

LevelBatch Plan::level_batch(int level) const {
  LevelBatch batch;
  batch.level = level;
  if (level < 0 || level >= levels_) {
    return batch;
  }
  batch.job_begin = level_begin_[static_cast<std::size_t>(level)];
  batch.job_end = level_begin_[static_cast<std::size_t>(level) + 1];
  batch.class_begin = span_begin_[static_cast<std::size_t>(level)];
  batch.class_end = span_begin_[static_cast<std::size_t>(level) + 1];
  for (int i = batch.job_begin; i < batch.job_end; ++i) {
    const PlaneJob& j = jobs_[static_cast<std::size_t>(i)];
    if (j.plane == 0) {
      ++batch.luma_jobs;
    } else {
      ++batch.chroma_jobs;
      if (j.is_cfl != 0) {
        ++batch.cfl_jobs;
      }
    }
    if (batch.first_syntax < 0) {
      batch.first_syntax = j.syntax;
    }
    batch.last_syntax = std::max(batch.last_syntax, j.syntax);
  }
  return batch;
}

int Plan::dependency_begin(int job) const {
  if (job < 0 || job >= static_cast<int>(jobs_.size())) {
    return 0;
  }
  return dep_begin_[static_cast<std::size_t>(job)];
}

int Plan::dependency_end(int job) const {
  if (job < 0 || job >= static_cast<int>(jobs_.size())) {
    return 0;
  }
  return dep_begin_[static_cast<std::size_t>(job) + 1];
}

}  // namespace av2enc::cuda::m1::sched
