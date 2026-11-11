// AV2 M1 MRL per-block syntax.  See mrl_syntax.h for the normative map.

#include "encoder/m1/mrl_stream/mrl_syntax.h"

#include <algorithm>
#include <cstring>

namespace av2enc::m1::mrl_stream {
namespace {

void copy_cdf(std::uint16_t (&dst)[kMrlIndexSymbols + 1],
              const std::uint16_t (&src)[kMrlIndexSymbols + 1]) noexcept {
  std::memcpy(dst, src, sizeof(dst));
}

bool set_error(std::string* error, const char* text) {
  if (error != nullptr) *error = text;
  return false;
}

// §8.3.2 context clamp: NNum <= 2, so the sum is already 0..2.  The clamp keeps
// a future caller that feeds a malformed block descriptor from indexing past
// MRL_INDEX_CONTEXTS.
int clamp_context(int ctx) noexcept {
  return ctx < 0 ? 0 : (ctx >= kMrlIndexContexts ? kMrlIndexContexts - 1 : ctx);
}

}  // namespace

void init_mrl_cdfs(MrlCdfs& cdfs) noexcept {
  for (int ctx = 0; ctx < kMrlIndexContexts; ++ctx) {
    copy_cdf(cdfs.index[ctx], kDefaultMrlIndexCdf[ctx]);
    std::memcpy(cdfs.sec[ctx], kDefaultMrlSecIndexCdf[ctx],
                sizeof(cdfs.sec[ctx]));
  }
}

int collect_npos(const MrlBlockSyntax& block, int mi_rows, int mi_cols,
                 int sb_size_mi, NeighborPos out[2]) noexcept {
  const int r = block.mi_row;
  const int c = block.mi_col;
  const int bw4 = block.bw4;
  const int bh4 = block.bh4;
  if (sb_size_mi <= 0) sb_size_mi = 16;

  int count = 0;
  const int probes[4][2] = {
      {r + bh4 - 1, c - 1},  // below-left
      {r - 1, c + bw4 - 1},  // above-right
      {r, c - 1},            // left
      {r - 1, c},            // above
  };
  for (const auto& probe : probes) {
    if (count >= 2) break;
    const int n_row = probe[0];
    const int n_col = probe[1];
    if (n_row < 0 || n_col < 0 || n_row >= mi_rows || n_col >= mi_cols) {
      continue;
    }
    // add_neighbor(): exclude a neighbour in a different superblock row.
    if ((r / sb_size_mi) != (n_row / sb_size_mi)) continue;
    out[count].row = n_row;
    out[count].col = n_col;
    ++count;
  }
  return count;
}

MrlContext::MrlContext(int mi_rows, int mi_cols, int sb_size_mi) {
  reset(mi_rows, mi_cols, sb_size_mi);
}

void MrlContext::reset(int mi_rows, int mi_cols, int sb_size_mi) {
  mi_rows_ = std::max(0, mi_rows);
  mi_cols_ = std::max(0, mi_cols);
  sb_size_mi_ = sb_size_mi > 0 ? sb_size_mi : 16;
  uses_.assign(static_cast<std::size_t>(mi_rows_) *
                   static_cast<std::size_t>(mi_cols_),
               0);
}

int MrlContext::uses_at(int mi_row, int mi_col) const noexcept {
  if (mi_row < 0 || mi_col < 0 || mi_row >= mi_rows_ || mi_col >= mi_cols_) {
    return 0;
  }
  return uses_[static_cast<std::size_t>(mi_row) *
                   static_cast<std::size_t>(mi_cols_) +
               static_cast<std::size_t>(mi_col)];
}

int MrlContext::index_context(const MrlBlockSyntax& block) const noexcept {
  NeighborPos npos[2];
  const int num = collect_npos(block, mi_rows_, mi_cols_, sb_size_mi_, npos);
  int ctx = 0;
  for (int i = 0; i < num; ++i) {
    if (uses_at(npos[i].row, npos[i].col) > 0) ++ctx;
  }
  return clamp_context(ctx);
}

int MrlContext::sec_context(const MrlBlockSyntax& block) const noexcept {
  NeighborPos npos[2];
  const int num = collect_npos(block, mi_rows_, mi_cols_, sb_size_mi_, npos);
  int ctx = 0;
  for (int i = 0; i < num; ++i) {
    if (uses_at(npos[i].row, npos[i].col) == 2) ++ctx;
  }
  return clamp_context(ctx);
}

void MrlContext::update(const MrlBlockSyntax& block, int mrl_index,
                        bool mrl_sec_index) noexcept {
  const std::uint8_t value = static_cast<std::uint8_t>(
      mrl_index > 0 ? (mrl_sec_index ? 2 : 1) : 0);
  const int r0 = std::max(0, block.mi_row);
  const int c0 = std::max(0, block.mi_col);
  const int r1 = std::min(mi_rows_, block.mi_row + block.bh4);
  const int c1 = std::min(mi_cols_, block.mi_col + block.bw4);
  for (int r = r0; r < r1; ++r) {
    for (int c = c0; c < c1; ++c) {
      uses_[static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
            static_cast<std::size_t>(c)] = value;
    }
  }
}

const char* mrl_syntax_status_name(MrlSyntaxStatus status) noexcept {
  switch (status) {
    case MrlSyntaxStatus::kOk:
      return "ok";
    case MrlSyntaxStatus::kInvalidMode:
      return "non-directional mode cannot signal MRL";
    case MrlSyntaxStatus::kInvalidIndex:
      return "mrl_index outside 0..3";
    case MrlSyntaxStatus::kInvalidSecondary:
      return "mrl_sec_index is not signalable for this block";
    case MrlSyntaxStatus::kEntropyError:
      return "entropy coder error";
  }
  return "unknown";
}

bool is_directional_y_mode(int y_mode) noexcept {
  return y_mode >= 1 && y_mode <= 8;
}

bool mrl_secondary_applies(int block_w, int block_h, int mrl_index,
                           bool mrl_sec_index) noexcept {
  return mrl_index > 0 && mrl_sec_index && !(block_w == 4 && block_h == 4);
}

MrlSyntaxStatus write_mrl_syntax(av2::Encoder& encoder, MrlCdfs& cdfs,
                                 MrlContext& context,
                                 const MrlBlockSyntax& block, int y_mode,
                                 const MrlDecision& decision,
                                 std::string* error) {
  if (error != nullptr) error->clear();

  if (!is_directional_y_mode(y_mode)) {
    if (decision.mrl_index != 0 || decision.mrl_sec_index) {
      set_error(error, "non-directional YMode must have mrl_index = 0");
      return MrlSyntaxStatus::kInvalidMode;
    }
    context.update(block, 0, false);
    return MrlSyntaxStatus::kOk;
  }

  if (decision.mrl_index < 0 || decision.mrl_index > kMaxMrlIndex) {
    set_error(error, "mrl_index must be in 0..3");
    return MrlSyntaxStatus::kInvalidIndex;
  }
  if (decision.mrl_index == 0 && decision.mrl_sec_index) {
    set_error(error, "mrl_sec_index requires mrl_index > 0");
    return MrlSyntaxStatus::kInvalidSecondary;
  }
  if (decision.mrl_sec_index && block.width() == 4 && block.height() == 4) {
    set_error(error, "the decoder ignores mrl_sec_index at 4x4");
    return MrlSyntaxStatus::kInvalidSecondary;
  }

  const int index_ctx = context.index_context(block);
  const av2::Status status = encoder.encode_symbol(
      cdfs.index[index_ctx], kMrlIndexSymbols, decision.mrl_index);
  if (status != av2::Status::kOk) {
    set_error(error, av2::status_name(status));
    return MrlSyntaxStatus::kEntropyError;
  }

  if (decision.mrl_index > 0) {
    const int sec_ctx = context.sec_context(block);
    const av2::Status sec_status = encoder.encode_symbol(
        cdfs.sec[sec_ctx], 2, decision.mrl_sec_index ? 1 : 0);
    if (sec_status != av2::Status::kOk) {
      set_error(error, av2::status_name(sec_status));
      return MrlSyntaxStatus::kEntropyError;
    }
  }

  context.update(block, decision.mrl_index, decision.mrl_sec_index);
  return MrlSyntaxStatus::kOk;
}

MrlSyntaxStatus read_mrl_syntax(av2::Decoder& decoder, MrlCdfs& cdfs,
                                MrlContext& context,
                                const MrlBlockSyntax& block, int y_mode,
                                MrlDecision* decision, std::string* error) {
  if (error != nullptr) error->clear();
  if (decision == nullptr) {
    set_error(error, "null decision output");
    return MrlSyntaxStatus::kInvalidMode;
  }
  *decision = MrlDecision{};

  if (!is_directional_y_mode(y_mode)) {
    // §5.20.5.5: the else branch sets mrl_index = 0 and codes nothing.
    context.update(block, 0, false);
    return MrlSyntaxStatus::kOk;
  }

  const int index_ctx = context.index_context(block);
  int index = 0;
  const av2::Status status =
      decoder.decode_symbol(cdfs.index[index_ctx], kMrlIndexSymbols, &index);
  if (status != av2::Status::kOk) {
    set_error(error, av2::status_name(status));
    return MrlSyntaxStatus::kEntropyError;
  }
  if (index < 0 || index > kMaxMrlIndex) {
    set_error(error, "decoded mrl_index outside 0..3");
    return MrlSyntaxStatus::kInvalidIndex;
  }
  decision->mrl_index = index;

  if (index > 0) {
    const int sec_ctx = context.sec_context(block);
    int sec = 0;
    const av2::Status sec_status =
        decoder.decode_symbol(cdfs.sec[sec_ctx], 2, &sec);
    if (sec_status != av2::Status::kOk) {
      set_error(error, av2::status_name(sec_status));
      return MrlSyntaxStatus::kEntropyError;
    }
    decision->mrl_sec_index = sec != 0;
  }

  // UsesMrls stores the raw syntax bit, including a 4x4 secondary that the
  // prediction process ignores.
  context.update(block, decision->mrl_index, decision->mrl_sec_index);
  return MrlSyntaxStatus::kOk;
}

}  // namespace av2enc::m1::mrl_stream
