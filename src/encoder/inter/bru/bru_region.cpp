#include "encoder/inter/bru/bru.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace av2enc::inter::bru {
namespace {

struct Rect {
  int left = 0;
  int right = 0;
  int top = 0;
  int bottom = 0;
  bool active = true;
};

}  // namespace

// §7.2 p378, transcribed.  The map here is one entry per superblock; the
// spec indexes BruModes on the MI grid, but every check samples the top-left
// MI unit of a superblock, so the reduction is exact.
bool bru_region_valid(int sb_cols, int sb_rows,
                      std::span<const BruMode> modes) {
  if (sb_cols <= 0 || sb_rows <= 0) return false;
  if (modes.size() !=
      static_cast<std::size_t>(sb_cols) * static_cast<std::size_t>(sb_rows)) {
    return false;
  }

  std::vector<Rect> rects;
  for (int r = 0; r < sb_rows; ++r) {
    for (int c = 0; c < sb_cols; ++c) {
      if (modes[static_cast<std::size_t>(r) * sb_cols + c] ==
          BruMode::kActive) {
        rects.push_back(Rect{c - 1, c + 1, r - 1, r + 1, true});
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t a = 0; a < rects.size(); ++a) {
      if (!rects[a].active) continue;
      for (std::size_t b = a + 1; b < rects.size(); ++b) {
        if (!rects[b].active) continue;
        const bool disjoint = rects[a].right < rects[b].left ||
                              rects[b].right < rects[a].left ||
                              rects[a].bottom < rects[b].top ||
                              rects[b].bottom < rects[a].top;
        if (disjoint) continue;
        rects[a].left = std::min(rects[a].left, rects[b].left);
        rects[a].right = std::max(rects[a].right, rects[b].right);
        rects[a].top = std::min(rects[a].top, rects[b].top);
        rects[a].bottom = std::max(rects[a].bottom, rects[b].bottom);
        rects[b].active = false;
        changed = true;
      }
    }
  }

  for (const Rect& rect : rects) {
    if (!rect.active) continue;
    for (int r = rect.top; r <= rect.bottom; ++r) {
      for (int c = rect.left; c <= rect.right; ++c) {
        const int row = r;
        const int col = c;
        if (row < 0 || row >= sb_rows || col < 0 || col >= sb_cols) continue;
        const BruMode mode =
            modes[static_cast<std::size_t>(row) * sb_cols + col];
        if (mode == BruMode::kInactive) return false;
        if (r == rect.top || r == rect.bottom || c == rect.left ||
            c == rect.right) {
          if (mode != BruMode::kSupport) return false;
        }
      }
    }
  }
  return true;
}

bool deblock_edge_allowed(bool use_bru, int sb_shift, int mi_row, int mi_col,
                          int sb_cols, std::span<const BruMode> modes) {
  if (!use_bru) return true;
  if (sb_shift < 0 || sb_cols <= 0) return true;
  const int sb_row = mi_row >> sb_shift;
  const int sb_col = mi_col >> sb_shift;
  if (sb_row < 0 || sb_col < 0) return true;
  const std::size_t index =
      static_cast<std::size_t>(sb_row) * static_cast<std::size_t>(sb_cols) +
      static_cast<std::size_t>(sb_col);
  if (index >= modes.size()) return true;
  return modes[index] == BruMode::kActive;
}

bool bru_ibc_buffer_valid(int sb_shift, int mi_row, int mi_col, int sb_cols,
                          std::span<const BruMode> modes) {
  if (sb_shift < 0 || sb_cols <= 0) return true;
  const int sb_row = mi_row >> sb_shift;
  const int sb_col = mi_col >> sb_shift;
  if (sb_row < 0 || sb_col < 0) return true;
  const std::size_t index =
      static_cast<std::size_t>(sb_row) * static_cast<std::size_t>(sb_cols) +
      static_cast<std::size_t>(sb_col);
  if (index >= modes.size()) return true;
  return modes[index] != BruMode::kInactive;
}

}  // namespace av2enc::inter::bru
