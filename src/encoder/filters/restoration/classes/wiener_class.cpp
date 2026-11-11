#include "encoder/filters/restoration/classes/wiener_class.h"

#include "encoder/filters/restoration/classes/wiener_tables.h"

#include <algorithm>
#include <cstdlib>

namespace av2enc::restoration::classes {
namespace {

std::int64_t round2_signed64(std::int64_t x, int n) {
  if (n <= 0) {
    return x;
  }
  if (x >= 0) {
    return (x + (static_cast<std::int64_t>(1) << (n - 1))) >> n;
  }
  return -((-x + (static_cast<std::int64_t>(1) << (n - 1))) >> n);
}

int get_tx_skip(const WienerClassifierContext& ctx, const LrBlock& b, int x,
                int y, int block_end_x, int block_start_x) {
  if (ctx.lr_tx_skip == nullptr || ctx.core == nullptr) {
    return 0;
  }
  x = std::min(block_end_x, x);
  x = std::max(block_start_x, x);
  y = clip3(b.stripe_start_y, b.stripe_end_y, y);
  const int tile_start_y = ctx.mi_row_start * kMiSize;
  const int tile_end_y = ctx.mi_row_end * kMiSize - 1;
  y = clip3(tile_start_y, tile_end_y, y);
  const int mi_cols = ctx.core->mi_cols;
  const int row = y >> kMiSizeLog2;
  const int col = x >> kMiSizeLog2;
  if (row < 0 || col < 0 || row >= ctx.core->mi_rows || col >= mi_cols) {
    return 0;
  }
  return ctx.lr_tx_skip[row * mi_cols + col] != 0 ? 1 : 0;
}

// §7.20.4 get_features().  Returns nf[] and tskip.
void get_features(const WienerClassifierContext& ctx, const LrCoreFrame& core,
                  const LrBlock& b, int x, int y, int block_start_x,
                  int block_end_x, std::int64_t f[4], int* tskip) {
  x = std::min(block_end_x + 2, x);
  const int m = lr_get_source_sample(core, b, 0, x, y);
  const int up = lr_get_source_sample(core, b, 0, x, y - 1);
  const int down = lr_get_source_sample(core, b, 0, x, y + 1);
  const int vert = up - 2 * m + down;
  const int upright = lr_get_source_sample(core, b, 0, x + 1, y - 1);
  const int downleft = lr_get_source_sample(core, b, 0, x - 1, y + 1);
  const int anti_diag = upright - 2 * m + downleft;
  const int downright = lr_get_source_sample(core, b, 0, x + 1, y + 1);
  const int upleft = lr_get_source_sample(core, b, 0, x - 1, y - 1);
  const int diag = upleft - 2 * m + downright;

  f[0] = 0;
  f[1] = std::abs(vert);
  f[2] = std::abs(anti_diag);
  f[3] = std::abs(diag);
  *tskip = get_tx_skip(ctx, b, x, y, block_end_x, block_start_x);
}

// §7.20.4 get_box_features().  Returns the normalised feature vector and
// tskip (ns).
void get_box_features(const WienerClassifierContext& ctx,
                      const LrCoreFrame& core, const LrBlock& b, int x, int y,
                      std::int64_t f[4], int* tskip) {
  const int block_start_x = (x >> 6) << 6;
  const int block_end_x =
      std::min(ctx.mi_col_end * kMiSize - 1, block_start_x + 63);
  std::int64_t sum[4] = {0, 0, 0, 0};
  int s = 0;
  for (int dy = -kPcWienerLead; dy <= kPcWienerLag; ++dy) {
    for (int dx = -kPcWienerLead; dx <= kPcWienerLag; ++dx) {
      std::int64_t tf[4];
      int skip = 0;
      get_features(ctx, core, b, x + dx, y + dy, block_start_x, block_end_x,
                   tf, &skip);
      for (int i = 0; i < 4; ++i) {
        sum[i] += tf[i];
      }
      s += skip;
    }
  }
  for (int i = 0; i < 4; ++i) {
    f[i] = round2_signed64(sum[i] * kPcWienerNormalizer[i],
                           ctx.bit_depth - 8);
  }
  *tskip = s * kPcWienerNormalizer[4];
}

}  // namespace

const char* wiener_table_status_string(WienerTableStatus status) {
  switch (status) {
    case WienerTableStatus::kOk:
      return "ok";
    case WienerTableStatus::kMissingNormativeTable:
      return "missing normative Pc_Wiener_* table (not in the spec text)";
    case WienerTableStatus::kInvalidTable:
      return "invalid Pc_Wiener_* table contents";
    case WienerTableStatus::kInvalidArgument:
      return "invalid argument";
  }
  return "unknown";
}

WienerTableStatus wiener_derive_filter_class(const LrCoreFrame& core,
                                             const LrBlock& block,
                                             const PcWienerTables& tables,
                                             const WienerClassifierContext& ctx,
                                             std::uint8_t* filter_class) {
  if (filter_class == nullptr || ctx.core == nullptr) {
    return WienerTableStatus::kInvalidArgument;
  }
  if (!tables.has_lut_to_class()) {
    return WienerTableStatus::kMissingNormativeTable;
  }
  if (ctx.bit_depth != 8) {
    // The normaliser and get_qval specialisations here are written for 8-bit;
    // other depths must be added with their own arithmetic rather than
    // approximated.
    return WienerTableStatus::kInvalidArgument;
  }

  std::int64_t f[4];
  int tskip = 0;
  get_box_features(ctx, core, block, block.x, block.y, f, &tskip);

  int lut_input = 0;
  for (int i = 0; i < 4; ++i) {
    const std::int64_t qval_raw =
        round2_signed64(f[i] + wiener_qval_given_tskip(ctx.base_q_idx, tskip,
                                                       i),
                        kPcWienerPrecFeature);
    const int qval = clip3(0, 255, static_cast<int>(qval_raw)) >> 5;
    lut_input += qval << (3 * (3 - i));
  }
  *filter_class = tables.lut_class(lut_input);
  return WienerTableStatus::kOk;
}

WienerTableStatus wiener_derive_filter_class_grid(
    const LrCoreFrame& core, const PcWienerTables& tables,
    const WienerClassifierContext& ctx,
    std::vector<std::uint8_t>* filter_class_grid) {
  if (filter_class_grid == nullptr || ctx.core == nullptr) {
    return WienerTableStatus::kInvalidArgument;
  }
  if (!tables.has_lut_to_class()) {
    return WienerTableStatus::kMissingNormativeTable;
  }
  if (ctx.mi_col_end <= 0) {
    return WienerTableStatus::kInvalidArgument;
  }
  if (ctx.mi_row_end <= 0) {
    return WienerTableStatus::kInvalidArgument;
  }
  const std::size_t n = static_cast<std::size_t>(core.mi_cols) *
                        static_cast<std::size_t>(core.mi_rows);
  filter_class_grid->assign(n, 0);
  for (int row = 0; row < core.mi_rows; ++row) {
    for (int col = 0; col < core.mi_cols; ++col) {
      LrBlock b;
      if (!lr_block_context(core, 0, row, col, &b)) {
        continue;
      }
      std::uint8_t cls = 0;
      const WienerTableStatus st =
          wiener_derive_filter_class(core, b, tables, ctx, &cls);
      if (st != WienerTableStatus::kOk) {
        return st;
      }
      (*filter_class_grid)[static_cast<std::size_t>(row) *
                           static_cast<std::size_t>(core.mi_cols) +
                           static_cast<std::size_t>(col)] = cls;
    }
  }
  return WienerTableStatus::kOk;
}

WienerTableStatus wiener_build_subclass_grid(
    const std::uint8_t* filter_class_grid, int mi_cols, int mi_rows,
    int filter_set, int num_classes, int num_classes_idx,
    const PcWienerTables& tables, std::vector<std::uint8_t>* subclass_grid) {
  if (filter_class_grid == nullptr || subclass_grid == nullptr ||
      mi_cols <= 0 || mi_rows <= 0) {
    return WienerTableStatus::kInvalidArgument;
  }
  if (filter_set < 0 || filter_set > 3 || num_classes_idx < 0 ||
      num_classes_idx > 7 || num_classes < 1 ||
      num_classes > kWienerNsClasses) {
    return WienerTableStatus::kInvalidArgument;
  }
  if (decode_num_filter_classes(num_classes_idx) != num_classes) {
    return WienerTableStatus::kInvalidArgument;
  }
  const std::size_t n = static_cast<std::size_t>(mi_cols) *
                        static_cast<std::size_t>(mi_rows);
  subclass_grid->assign(n, 0);
  if (num_classes == 1) {
    return WienerTableStatus::kOk;  // SubclassLookup is not consulted.
  }
  if (!tables.has_sub_classify2()) {
    return WienerTableStatus::kMissingNormativeTable;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const int cls = filter_class_grid[i];
    const int subcls = tables.subclass(filter_set, num_classes_idx, cls);
    if (subcls < 0 || subcls >= num_classes) {
      return WienerTableStatus::kInvalidTable;
    }
    (*subclass_grid)[i] = static_cast<std::uint8_t>(subcls);
  }
  return WienerTableStatus::kOk;
}

}  // namespace av2enc::restoration::classes
