#include "codec/inter/morph/morph_syntax.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "codec/inter/morph/morph_common.h"

namespace av2enc::inter::morph {
namespace {

bool enc_symbol(av2::Encoder& enc, std::uint16_t* cdf, int n, int symbol,
                std::string& error) {
  const av2::Status status = enc.encode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    error = std::string("encode_symbol failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

bool dec_symbol(av2::Decoder& dec, std::uint16_t* cdf, int n, int& symbol,
                std::string& error) {
  const av2::Status status = dec.decode_symbol(cdf, n, &symbol);
  if (status != av2::Status::kOk) {
    error = std::string("decode_symbol failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

}  // namespace

void init_morph_cdfs(MorphCdfs& cdfs) noexcept {
  static constexpr std::uint16_t kDefault[3][3] = {
      {31715, 55, 0},
      {19667, 1, 0},
      {10555, 91, 0},
  };
  std::memcpy(cdfs.morph_pred, kDefault, sizeof(kDefault));
}

MorphNeighbours morph_neighbours(int mi_row, int mi_col, int bw4, int bh4,
                                 int mi_cols, int mi_rows, int sb_size) noexcept {
  MorphNeighbours out;
  int sb_mi = std::max(1, sb_size / kMiSize);
  int sb_log2 = 0;
  while (sb_mi > 1) {
    sb_mi >>= 1;
    ++sb_log2;
  }
  const auto inside = [&](int r, int c) {
    return r >= 0 && c >= 0 && r < mi_rows && c < mi_cols;
  };
  const auto add = [&](int r, int c) {
    if (out.count >= 2 || !inside(r, c)) return;
    if ((mi_row >> sb_log2) != (r >> sb_log2)) return;
    out.row[out.count] = r;
    out.col[out.count] = c;
    ++out.count;
  };
  // §5.20.4.1 decode_block(): exact add_neighbor() order.
  add(mi_row + bh4 - 1, mi_col - 1);
  add(mi_row - 1, mi_col + bw4 - 1);
  add(mi_row, mi_col - 1);
  add(mi_row - 1, mi_col);
  return out;
}

int morph_symbol_context(const std::uint8_t* morph_grid, int mi_cols,
                         int mi_rows, int mi_row, int mi_col, int bw4,
                         int bh4, int sb_size) noexcept {
  if (morph_grid == nullptr || mi_cols <= 0 || mi_rows <= 0) {
    return 0;
  }
  const MorphNeighbours n =
      morph_neighbours(mi_row, mi_col, bw4, bh4, mi_cols, mi_rows, sb_size);
  int ctx = 0;
  for (int i = 0; i < n.count; ++i) {
    ctx += morph_grid[static_cast<std::size_t>(n.row[i]) *
                          static_cast<std::size_t>(mi_cols) +
                      static_cast<std::size_t>(n.col[i])] != 0
               ? 1
               : 0;
  }
  return ctx;
}

bool write_morph_pred(av2::Encoder& enc, MorphCdfs& cdfs, int ctx, bool value,
                      std::string& error) {
  if (ctx < 0 || ctx > 2) {
    error = "morph_pred context is outside 0..2";
    return false;
  }
  return enc_symbol(enc, &cdfs.morph_pred[ctx][0], 2, value ? 1 : 0, error);
}

bool read_morph_pred(av2::Decoder& dec, MorphCdfs& cdfs, int ctx, bool& value,
                     std::string& error) {
  if (ctx < 0 || ctx > 2) {
    error = "morph_pred context is outside 0..2";
    return false;
  }
  int symbol = 0;
  if (!dec_symbol(dec, &cdfs.morph_pred[ctx][0], 2, symbol, error)) {
    return false;
  }
  if (symbol != 0 && symbol != 1) {
    error = "morph_pred decoded an out-of-range symbol";
    return false;
  }
  value = symbol == 1;
  return true;
}

bool morph_pred_present(const MorphPredGate& gate) noexcept {
  return gate.frame_is_intra && gate.allow_screen_content_tools &&
         gate.enable_bawp;
}

bool morph_pred_allowed(const MorphPredGate& gate, bool offset_mv_valid,
                        std::string* reason) {
  if (!morph_pred_present(gate)) {
    if (reason != nullptr) {
      *reason = "morph_pred is not signalled (FrameIsIntra && "
                "allow_screen_content_tools && enable_bawp is false)";
    }
    return false;
  }
  if (!gate.use_intrabc) {
    if (reason != nullptr) {
      *reason = "morph_pred is only read by read_intrabc_info()";
    }
    return false;
  }
  if (!offset_mv_valid) {
    if (reason != nullptr) {
      *reason = "morph_pred requires is_offset_mv_valid(-1, -1) == 1";
    }
    return false;
  }
  return true;
}

}  // namespace av2enc::inter::morph
