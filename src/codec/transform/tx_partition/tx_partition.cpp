// §5.18.8.1 / §5.20.6 transform mode and transform partition signalling.
#include "codec/transform/tx_partition/tx_partition.h"

#include <algorithm>
#include <cstring>

#include "codec/transform/tx_partition/tx_partition_tables.h"

namespace av2enc::video::txpartition {
namespace {

constexpr int kMiSize = 4;
constexpr int kMaxChunkShift = 4;  // 64x64 == 16x16 4x4 units

bool valid_tx_size(int tx_size) {
  return tx_size >= 0 && tx_size < kTxSizeCount;
}

bool valid_mi_size(int mi_size) {
  return mi_size >= 0 && mi_size < kBlockCount;
}

bool emit(SymbolSink& sink, std::uint16_t* cdf, int n, int value,
          std::string& error) {
  if (cdf == nullptr) {
    error = "null CDF";
    return false;
  }
  if (!sink.symbol(cdf, n, value, error)) return false;
  return true;
}

bool read_symbol(SymbolSource& source, std::uint16_t* cdf, int n, int* value,
                 std::string& error) {
  if (cdf == nullptr) {
    error = "null CDF";
    return false;
  }
  if (!source.symbol(cdf, n, value, error)) return false;
  if (*value < 0 || *value >= n) {
    error = "decoded symbol " + std::to_string(*value) + " outside [0," +
            std::to_string(n) + ")";
    return false;
  }
  return true;
}

bool uses_two_or_three(int mi_size) {
  return tx_type_group_vert_or_horz(mi_size) > 0;
}

bool sorted_option_contains(const LegalSplits& legal, TxPartition part) {
  for (int i = 0; i < legal.option_count; ++i) {
    if (legal.options[i] == part) return true;
  }
  return false;
}

// set_tx_size() from §5.20.6.3.  h4/w4 are in 4x4 units.
void set_tx_size(LumaTxGrid& grid, int row, int col, int h4, int w4, int mid,
                 int scan_order) {
  const int sub = find_tx_size(w4 * kMiSize, h4 * kMiSize);
  if (sub < 0) return;
  grid.set(row, col, h4, w4, sub, mid, scan_order);
}

bool write_one_partition(SymbolSink& sink, TxPartitionCdfs& cdfs,
                         const BlockTxContext& ctx, int mi_row, int mi_col,
                         TxPartition decision, SplitGeometry* geometry,
                         std::string& error) {
  if (!valid_mi_size(ctx.mi_size)) {
    error = "invalid MiSize " + std::to_string(ctx.mi_size);
    return false;
  }
  const LegalSplits legal = legal_splits(ctx.mi_size, ctx.reduced_tx_part_set);
  if (!legal.applicable) {
    // §5.20.6.3: the syntax block is guarded by
    // Block_Width[MiSize] <= 64 && Block_Height[MiSize] <= 64.  Larger blocks
    // arrive here only from the per-64x64 chunk loops of
    // read_tx_size()/read_block_tx_size(), and code no partition symbols.
    if (decision != TxPartition::kNone) {
      error = std::string("no tx partition syntax for MiSize ") +
              std::to_string(ctx.mi_size) + " (64x64 chunking)";
      return false;
    }
    if (geometry != nullptr) {
      *geometry = SplitGeometry{};
      const int tx = max_tx_size_rect(ctx.mi_size);
      if (tx >= 0) {
        geometry->parts[0] = SplitPart{mi_row, mi_col, tx};
        geometry->count = 1;
      }
    }
    return true;
  }
  if (!sorted_option_contains(legal, decision)) {
    error = std::string("illegal tx partition ") + tx_partition_name(decision) +
            " for MiSize " + std::to_string(ctx.mi_size) +
            (ctx.reduced_tx_part_set ? " with reduced_tx_part_set=1" : "");
    return false;
  }

  const bool fsc = ctx.fsc_mode;
  const bool inter = ctx.is_inter;
  std::uint16_t* do_cdf =
      cdfs.do_partition_cdf(fsc, inter, ctx.mi_size);
  if (do_cdf == nullptr) {
    error = "no tx_do_partition CDF";
    return false;
  }
  const bool do_split = decision != TxPartition::kNone;
  if (!emit(sink, do_cdf, 2, do_split ? 1 : 0, error)) return false;

  if (do_split) {
    if (legal.has_partition_type) {
      std::uint16_t* type_cdf =
          cdfs.partition_type_cdf(fsc, inter, ctx.mi_size);
      if (type_cdf == nullptr) {
        error = "no tx_partition_type CDF";
        return false;
      }
      // tx_partition_type is coded with TX_PARTITION_TYPE_NUM (7) symbols;
      // only 0..2 (SPLIT/HORZ/VERT) are reachable when both directions are
      // legal.
      int symbol = 0;
      switch (decision) {
        case TxPartition::kSplit:
          symbol = 0;
          break;
        case TxPartition::kHorz:
          symbol = 1;
          break;
        case TxPartition::kVert:
          symbol = 2;
          break;
        default:
          error = "tx_partition_type cannot signal " +
                  std::string(tx_partition_name(decision));
          return false;
      }
      if (!emit(sink, type_cdf, kTxPartitionTypeNum, symbol, error)) {
        return false;
      }
    } else if (legal.has_2or3) {
      const bool horizontal = legal.allow_horz;
      const int symbol = (horizontal ? decision == TxPartition::kHorz4
                                     : decision == TxPartition::kVert4)
                             ? 1
                             : 0;
      if (!legal.reduced_tx_part_set) {
        std::uint16_t* cdf =
            cdfs.partition_2or3_cdf(fsc, inter, ctx.mi_size);
        if (cdf == nullptr) {
          error = "no tx_2or3_partition_type CDF";
          return false;
        }
        if (!emit(sink, cdf, 2, symbol, error)) return false;
      } else if (symbol != 0) {
        error = "reduced_tx_part_set=1 cannot signal a 4-way tx partition";
        return false;
      }
    } else {
      // One direction is legal and the block size has no 2-or-3-way group, so
      // the two-way partition is implied by tx_do_partition alone.
      const TxPartition implied =
          legal.allow_horz ? TxPartition::kHorz : TxPartition::kVert;
      if (decision != implied) {
        error = "tx partition is implied; no symbol can select " +
                std::string(tx_partition_name(decision));
        return false;
      }
    }
  }

  if (geometry != nullptr) {
    if (!tx_partition_geometry(ctx.mi_size, mi_row, mi_col, decision, geometry,
                               &error)) {
      return false;
    }
  }
  return true;
}

}  // namespace

const char* tx_partition_name(TxPartition part) {
  switch (part) {
    case TxPartition::kNone:
      return "TX_PARTITION_NONE";
    case TxPartition::kSplit:
      return "TX_PARTITION_SPLIT";
    case TxPartition::kHorz:
      return "TX_PARTITION_HORZ";
    case TxPartition::kVert:
      return "TX_PARTITION_VERT";
    case TxPartition::kHorz4:
      return "TX_PARTITION_HORZ4";
    case TxPartition::kVert4:
      return "TX_PARTITION_VERT4";
    case TxPartition::kHorz5:
      return "TX_PARTITION_HORZ5";
    case TxPartition::kVert5:
      return "TX_PARTITION_VERT5";
  }
  return "unknown";
}

int max_tx_size_rect(int mi_size) {
  return valid_mi_size(mi_size) ? kTxPartitionMaxTxSizeRect[mi_size] : -1;
}
int tx_part_group(int mi_size) {
  return valid_mi_size(mi_size) ? kTxPartitionBlockGroup[mi_size] : -1;
}
int tx_type_group_vert_and_horz(int mi_size) {
  return valid_mi_size(mi_size) ? kTxPartitionTypeGroupVertAndHorz[mi_size]
                                : 255;
}
int tx_type_group_vert_or_horz(int mi_size) {
  return valid_mi_size(mi_size) ? kTxPartitionTypeGroupVertOrHorz[mi_size]
                                : 255;
}
int block_width_mi(int mi_size) {
  return valid_mi_size(mi_size) ? kTxPartitionBlockWidth[mi_size] : 0;
}
int block_height_mi(int mi_size) {
  return valid_mi_size(mi_size) ? kTxPartitionBlockHeight[mi_size] : 0;
}
int block_width_px(int mi_size) { return block_width_mi(mi_size) * kMiSize; }
int block_height_px(int mi_size) { return block_height_mi(mi_size) * kMiSize; }
int tx_width(int tx_size) {
  return valid_tx_size(tx_size) ? kTxPartitionTxWidth[tx_size] : 0;
}
int tx_height(int tx_size) {
  return valid_tx_size(tx_size) ? kTxPartitionTxHeight[tx_size] : 0;
}
int tx_width_log2(int tx_size) {
  return valid_tx_size(tx_size) ? kTxPartitionTxWidthLog2[tx_size] : 0;
}
int tx_height_log2(int tx_size) {
  return valid_tx_size(tx_size) ? kTxPartitionTxHeightLog2[tx_size] : 0;
}
int adjusted_tx_size(int tx_size) {
  return valid_tx_size(tx_size) ? kTxPartitionAdjustedTxSize[tx_size] : -1;
}

int find_tx_size(int w, int h) {
  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    if (tx_width(tx) == w && tx_height(tx) == h) return tx;
  }
  return -1;
}

int tx_chunk_cols(int mi_size) {
  const int chunks = block_width_px(mi_size) >> 6;
  return chunks > 1 ? chunks : 1;
}
int tx_chunk_rows(int mi_size) {
  const int chunks = block_height_px(mi_size) >> 6;
  return chunks > 1 ? chunks : 1;
}

LegalSplits legal_splits(int mi_size, bool reduced_tx_part_set) {
  LegalSplits out;
  out.reduced_tx_part_set = reduced_tx_part_set;
  if (!valid_mi_size(mi_size) || mi_size == kBlock4x4) return out;
  const int bw = block_width_px(mi_size);
  const int bh = block_height_px(mi_size);
  out.chunked = bw > 64 || bh > 64;
  if (out.chunked) return out;

  const int tx = max_tx_size_rect(mi_size);
  if (tx < 0) return out;
  const int tw = tx_width(tx);
  const int th = tx_height(tx);
  out.applicable = true;
  out.allow_horz = find_tx_size(tw, th / 2) >= 0;
  out.allow_vert = find_tx_size(tw / 2, th) >= 0;
  out.has_partition_type = out.allow_horz && out.allow_vert;

  const auto add = [&out](TxPartition part) {
    if (out.option_count < 8) out.options[out.option_count++] = part;
  };
  add(TxPartition::kNone);
  if (out.has_partition_type) {
    add(TxPartition::kSplit);
    add(TxPartition::kHorz);
    add(TxPartition::kVert);
  } else if (uses_two_or_three(mi_size)) {
    out.has_2or3 = true;
    if (out.allow_horz) {
      add(TxPartition::kHorz);
      if (!reduced_tx_part_set) add(TxPartition::kHorz4);
    } else {
      add(TxPartition::kVert);
      if (!reduced_tx_part_set) add(TxPartition::kVert4);
    }
  } else if (out.allow_horz || out.allow_vert) {
    add(out.allow_horz ? TxPartition::kHorz : TxPartition::kVert);
  }
  return out;
}

bool tx_partition_signallable(int mi_size, TxPartition part) {
  const LegalSplits legal = legal_splits(mi_size, /*reduced=*/false);
  return legal.applicable && sorted_option_contains(legal, part);
}

bool tx_partition_geometry(int mi_size, int mi_row, int mi_col,
                           TxPartition part, SplitGeometry* out,
                           std::string* error) {
  if (out == nullptr) {
    if (error) *error = "null geometry output";
    return false;
  }
  *out = SplitGeometry{};
  if (!valid_mi_size(mi_size)) {
    if (error) *error = "invalid MiSize";
    return false;
  }
  const int tx = max_tx_size_rect(mi_size);
  if (tx < 0) {
    if (error) *error = "invalid max transform size";
    return false;
  }
  if (!tx_partition_signallable(mi_size, part)) {
    if (error) {
      *error = std::string("tx partition ") + tx_partition_name(part) +
               " is not signallable for MiSize " + std::to_string(mi_size);
    }
    return false;
  }
  const int w4 = tx_width(tx) / kMiSize;
  const int h4 = tx_height(tx) / kMiSize;
  const auto push = [out](int row, int col, int size) {
    if (out->count < 4) {
      out->parts[out->count++] = SplitPart{row, col, size};
    }
  };

  switch (part) {
    case TxPartition::kNone:
      push(mi_row, mi_col, tx);
      break;
    case TxPartition::kSplit: {
      const int sub = find_tx_size(tx_width(tx) / 2, tx_height(tx) / 2);
      if (sub < 0) {
        if (error) *error = "SPLIT produces an invalid transform size";
        return false;
      }
      const int hw = w4 / 2;
      const int hh = h4 / 2;
      // residual() visits transform blocks row-major.
      push(mi_row, mi_col, sub);
      push(mi_row, mi_col + hw, sub);
      push(mi_row + hh, mi_col, sub);
      push(mi_row + hh, mi_col + hw, sub);
      break;
    }
    case TxPartition::kHorz: {
      const int sub = find_tx_size(tx_width(tx), tx_height(tx) / 2);
      if (sub < 0) {
        if (error) *error = "HORZ produces an invalid transform size";
        return false;
      }
      const int hh = h4 / 2;
      push(mi_row, mi_col, sub);
      push(mi_row + hh, mi_col, sub);
      break;
    }
    case TxPartition::kVert: {
      const int sub = find_tx_size(tx_width(tx) / 2, tx_height(tx));
      if (sub < 0) {
        if (error) *error = "VERT produces an invalid transform size";
        return false;
      }
      const int hw = w4 / 2;
      push(mi_row, mi_col, sub);
      push(mi_row, mi_col + hw, sub);
      break;
    }
    case TxPartition::kHorz4: {
      const int sub = find_tx_size(tx_width(tx), tx_height(tx) / 4);
      if (sub < 0) {
        if (error) *error = "HORZ4 produces an invalid transform size";
        return false;
      }
      const int q = h4 / 4;
      for (int i = 0; i < 4; ++i) push(mi_row + i * q, mi_col, sub);
      break;
    }
    case TxPartition::kVert4: {
      const int sub = find_tx_size(tx_width(tx) / 4, tx_height(tx));
      if (sub < 0) {
        if (error) *error = "VERT4 produces an invalid transform size";
        return false;
      }
      const int q = w4 / 4;
      for (int i = 0; i < 4; ++i) push(mi_row, mi_col + i * q, sub);
      break;
    }
    case TxPartition::kHorz5:
    case TxPartition::kVert5:
      if (error) {
        *error = std::string(tx_partition_name(part)) +
                 " cannot be signalled by tx_do_partition / "
                 "tx_partition_type / tx_2or3_partition_type";
      }
      return false;
  }
  return out->count > 0;
}

void TxPartitionCdfs::reset(bool reduced_tx_part_set) {
  std::memcpy(tx_do_partition, kDefaultTxDoPartitionCdf,
              sizeof(tx_do_partition));
  std::memcpy(tx_partition_type, kDefaultTxPartitionTypeCdf,
              sizeof(tx_partition_type));
  std::memcpy(tx_partition_type_reduced,
              reduced_tx_part_set ? kDefaultTxPartitionTypeReducedCdf
                                  : kDefaultTxPartitionTypeCdf,
              sizeof(tx_partition_type_reduced));
  std::memcpy(tx_2or3_partition_type, kDefaultTx2or3PartitionTypeCdf,
              sizeof(tx_2or3_partition_type));
  reduced = reduced_tx_part_set;
}

std::uint16_t* TxPartitionCdfs::do_partition_cdf(bool fsc_mode, bool is_inter,
                                                 int mi_size) {
  const int group = tx_part_group(mi_size);
  if (group < 0 || group >= kTxFmSplitGroup) return nullptr;
  return &tx_do_partition[fsc_mode ? 1 : 0][is_inter ? 1 : 0][group][0];
}

std::uint16_t* TxPartitionCdfs::partition_type_cdf(bool fsc_mode,
                                                   bool is_inter,
                                                   int mi_size) {
  const int group = tx_type_group_vert_and_horz(mi_size);
  if (group < 0 || group >= kTxPartitionTypeNumVertAndHorz) return nullptr;
  // reset() stored the §9.3-selected variant (full or reduced) here.
  return &tx_partition_type_reduced[fsc_mode ? 1 : 0][is_inter ? 1 : 0][group]
                                   [0];
}

std::uint16_t* TxPartitionCdfs::partition_2or3_cdf(bool fsc_mode,
                                                   bool is_inter,
                                                   int mi_size) {
  const int group = tx_type_group_vert_or_horz(mi_size);
  if (group <= 0 || group > kTxPartitionTypeNumVertOrHorz - 1) return nullptr;
  return &tx_2or3_partition_type[fsc_mode ? 1 : 0][is_inter ? 1 : 0]
                                [group - 1][0];
}

std::uint64_t cdf_fingerprint(const TxPartitionCdfs& cdfs) {
  // FNV-1a 64-bit over the serialised CDF state.
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&hash](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ULL;
    }
  };
  mix(cdfs.tx_do_partition, sizeof(cdfs.tx_do_partition));
  mix(cdfs.tx_partition_type, sizeof(cdfs.tx_partition_type));
  mix(cdfs.tx_partition_type_reduced, sizeof(cdfs.tx_partition_type_reduced));
  mix(cdfs.tx_2or3_partition_type, sizeof(cdfs.tx_2or3_partition_type));
  return hash;
}

void LumaTxGrid::reset(int rows, int cols) {
  mi_rows = rows > 0 ? rows : 0;
  mi_cols = cols > 0 ? cols : 0;
  const std::size_t n =
      static_cast<std::size_t>(mi_rows) * static_cast<std::size_t>(mi_cols);
  tx_size.assign(n, 0);
  middle.assign(n, 0);
  scan_order.assign(n, 0);
}

bool LumaTxGrid::contains(int row, int col) const {
  return row >= 0 && col >= 0 && row < mi_rows && col < mi_cols;
}

std::uint8_t LumaTxGrid::at(int row, int col) const {
  if (!contains(row, col)) return 0xFF;
  return tx_size[static_cast<std::size_t>(row) *
                     static_cast<std::size_t>(mi_cols) +
                 static_cast<std::size_t>(col)];
}

std::uint8_t LumaTxGrid::middle_at(int row, int col) const {
  if (!contains(row, col)) return 0xFF;
  return middle[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(mi_cols) +
                static_cast<std::size_t>(col)];
}

std::uint8_t LumaTxGrid::scan_order_at(int row, int col) const {
  if (!contains(row, col)) return 0xFF;
  return scan_order[static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(mi_cols) +
                    static_cast<std::size_t>(col)];
}

void LumaTxGrid::set(int row, int col, int h4, int w4, int tx_size_value,
                     int mid, int order) {
  for (int i = 0; i < h4; ++i) {
    for (int j = 0; j < w4; ++j) {
      const int r = row + i;
      const int c = col + j;
      if (!contains(r, c)) continue;
      const std::size_t idx =
          static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols) +
          static_cast<std::size_t>(c);
      tx_size[idx] = static_cast<std::uint8_t>(tx_size_value);
      middle[idx] = static_cast<std::uint8_t>(mid);
      scan_order[idx] = static_cast<std::uint8_t>(order);
    }
  }
}

bool write_tx_partition(SymbolSink& sink, TxPartitionCdfs& cdfs,
                        const BlockTxContext& ctx, TxPartition decision,
                        SplitGeometry* geometry, std::string& error) {
  return write_one_partition(sink, cdfs, ctx, 0, 0, decision, geometry,
                             error);
}

bool read_tx_partition(SymbolSource& source, TxPartitionCdfs& cdfs,
                       const BlockTxContext& ctx, int mi_row, int mi_col,
                       TxPartition* decision, SplitGeometry* geometry,
                       std::string& error) {
  if (decision == nullptr) {
    error = "null decision output";
    return false;
  }
  if (geometry != nullptr) *geometry = SplitGeometry{};
  *decision = TxPartition::kNone;
  if (!valid_mi_size(ctx.mi_size)) {
    error = "invalid MiSize " + std::to_string(ctx.mi_size);
    return false;
  }
  const LegalSplits legal = legal_splits(ctx.mi_size, ctx.reduced_tx_part_set);
  if (!legal.applicable) {
    // Mirrors the §5.20.6.3 guard: no symbols, the chunk stays at the block's
    // maximum transform size.
    if (geometry != nullptr) {
      const int tx = max_tx_size_rect(ctx.mi_size);
      if (tx >= 0) {
        geometry->parts[0] = SplitPart{mi_row, mi_col, tx};
        geometry->count = 1;
      }
    }
    return true;
  }

  const bool fsc = ctx.fsc_mode;
  const bool inter = ctx.is_inter;
  int do_split = 0;
  if (!read_symbol(source, cdfs.do_partition_cdf(fsc, inter, ctx.mi_size), 2,
                   &do_split, error)) {
    return false;
  }
  TxPartition part = TxPartition::kNone;
  if (do_split != 0) {
    if (legal.has_partition_type) {
      int symbol = 0;
      if (!read_symbol(source,
                       cdfs.partition_type_cdf(fsc, inter, ctx.mi_size),
                       kTxPartitionTypeNum, &symbol, error)) {
        return false;
      }
      if (symbol > 2) {
        error = "tx_partition_type symbol " + std::to_string(symbol) +
                " is reserved (only 0..2 are produced)";
        return false;
      }
      part = static_cast<TxPartition>(symbol + 1);
    } else if (legal.has_2or3) {
      int symbol = 0;
      if (!legal.reduced_tx_part_set) {
        if (!read_symbol(source,
                         cdfs.partition_2or3_cdf(fsc, inter, ctx.mi_size), 2,
                         &symbol, error)) {
          return false;
        }
      }
      if (legal.allow_horz) {
        part = symbol ? TxPartition::kHorz4 : TxPartition::kHorz;
      } else {
        part = symbol ? TxPartition::kVert4 : TxPartition::kVert;
      }
    } else {
      part = legal.allow_horz ? TxPartition::kHorz : TxPartition::kVert;
    }
  }

  SplitGeometry geom;
  if (!tx_partition_geometry(ctx.mi_size, mi_row, mi_col, part, &geom,
                             &error)) {
    return false;
  }
  *decision = part;
  if (geometry != nullptr) *geometry = geom;
  return true;
}

bool write_tx_size(SymbolSink& sink, TxPartitionCdfs& cdfs,
                   const TxSizeContext& ctx, bool allow_select,
                   const PartitionDecider& decide, LumaTxGrid& grid,
                   bool* partitioned, std::string& error) {
  if (partitioned != nullptr) *partitioned = false;
  const int mi = ctx.mi_size;
  if (!valid_mi_size(mi)) {
    error = "invalid MiSize";
    return false;
  }
  if (ctx.lossless) {
    // §5.20.6.1: the only cases reachable without the lossless_tx_size symbol.
    if (mi == kBlock4x4 || (!ctx.is_inter && !ctx.fsc_mode) || !allow_select) {
      grid.set(ctx.mi_row, ctx.mi_col, block_height_mi(mi), block_width_mi(mi),
               kTxSize4x4);
      return true;
    }
    error =
        "lossless_tx_size signalling (Lossless, inter/FSC, allowSelect) is "
        "outside this component";
    return false;
  }

  const int max_rect = max_tx_size_rect(mi);
  if (max_rect < 0) {
    error = "invalid max transform size";
    return false;
  }
  if (mi != kBlock4x4 && allow_select && ctx.tx_mode == TxMode::kSelect) {
    const int width_chunks = block_width_px(mi) >> 6;
    const int height_chunks = block_height_px(mi) >> 6;
    if (width_chunks > 1 || height_chunks > 1) {
      for (int chunk_y = 0; chunk_y < height_chunks; ++chunk_y) {
        for (int chunk_x = 0; chunk_x < width_chunks; ++chunk_x) {
          const int row = ctx.mi_row + (chunk_y << kMaxChunkShift);
          const int col = ctx.mi_col + (chunk_x << kMaxChunkShift);
          set_tx_size(grid, row, col, 16, 16, 0, 0);
        }
      }
      if (partitioned != nullptr) *partitioned = true;
      return true;
    }
    if (!decide) {
      error = "no transform partition decision callback";
      return false;
    }
    TxPartition part = TxPartition::kNone;
    if (!decide(0, ctx.mi_row, ctx.mi_col, &part, error)) return false;
    const BlockTxContext bctx{mi, ctx.is_inter, ctx.fsc_mode,
                              ctx.reduced_tx_part_set};
    SplitGeometry geom;
    if (!write_one_partition(sink, cdfs, bctx, ctx.mi_row, ctx.mi_col, part,
                             &geom, error)) {
      return false;
    }
    for (int i = 0; i < geom.count; ++i) {
      const SplitPart& p = geom.parts[i];
      set_tx_size(grid, p.mi_row, p.mi_col, tx_height(p.tx_size) / kMiSize,
                  tx_width(p.tx_size) / kMiSize, 0, 0);
    }
    if (partitioned != nullptr) *partitioned = true;
    return true;
  }

  grid.set(ctx.mi_row, ctx.mi_col, block_height_mi(mi), block_width_mi(mi),
           max_rect);
  return true;
}

bool write_block_tx_size(SymbolSink& sink, TxPartitionCdfs& cdfs,
                         const TxSizeContext& ctx,
                         const PartitionDecider& decide, LumaTxGrid& grid,
                         std::string& error) {
  const int mi = ctx.mi_size;
  if (!valid_mi_size(mi)) {
    error = "invalid MiSize";
    return false;
  }
  const bool inter_path =
      ctx.tx_mode == TxMode::kSelect && mi != kBlock4x4 && ctx.is_inter &&
      !ctx.skip_flag && !ctx.lossless;
  if (!inter_path) {
    const bool allow_select = !ctx.skip_flag || !ctx.is_inter;
    return write_tx_size(sink, cdfs, ctx, allow_select, decide, grid, nullptr,
                         error);
  }
  const int max_rect = max_tx_size_rect(mi);
  const int tx_w4 = tx_width(max_rect) / kMiSize;
  const int tx_h4 = tx_height(max_rect) / kMiSize;
  if (tx_w4 <= 0 || tx_h4 <= 0) {
    error = "invalid maximum transform size";
    return false;
  }
  const int bw4 = block_width_mi(mi);
  const int bh4 = block_height_mi(mi);
  int chunk = 0;
  for (int row = ctx.mi_row; row < ctx.mi_row + bh4; row += tx_h4) {
    for (int col = ctx.mi_col; col < ctx.mi_col + bw4; col += tx_w4) {
      if (!decide) {
        error = "no transform partition decision callback";
        return false;
      }
      TxPartition part = TxPartition::kNone;
      if (!decide(chunk, row, col, &part, error)) return false;
      const BlockTxContext bctx{mi, ctx.is_inter, ctx.fsc_mode,
                                ctx.reduced_tx_part_set};
      SplitGeometry geom;
      if (!write_one_partition(sink, cdfs, bctx, row, col, part, &geom,
                               error)) {
        return false;
      }
      for (int i = 0; i < geom.count; ++i) {
        const SplitPart& p = geom.parts[i];
        set_tx_size(grid, p.mi_row, p.mi_col, tx_height(p.tx_size) / kMiSize,
                    tx_width(p.tx_size) / kMiSize, 0, 0);
      }
      ++chunk;
    }
  }
  return true;
}

bool read_tx_size(SymbolSource& source, TxPartitionCdfs& cdfs,
                  const TxSizeContext& ctx, bool allow_select,
                  LumaTxGrid& grid, bool* partitioned, std::string& error) {
  if (partitioned != nullptr) *partitioned = false;
  const int mi = ctx.mi_size;
  if (!valid_mi_size(mi)) {
    error = "invalid MiSize";
    return false;
  }
  if (ctx.lossless) {
    if (mi == kBlock4x4 || (!ctx.is_inter && !ctx.fsc_mode) || !allow_select) {
      grid.set(ctx.mi_row, ctx.mi_col, block_height_mi(mi), block_width_mi(mi),
               kTxSize4x4);
      return true;
    }
    error =
        "lossless_tx_size signalling (Lossless, inter/FSC, allowSelect) is "
        "outside this component";
    return false;
  }

  const int max_rect = max_tx_size_rect(mi);
  if (max_rect < 0) {
    error = "invalid max transform size";
    return false;
  }
  if (mi != kBlock4x4 && allow_select && ctx.tx_mode == TxMode::kSelect) {
    const int width_chunks = block_width_px(mi) >> 6;
    const int height_chunks = block_height_px(mi) >> 6;
    if (width_chunks > 1 || height_chunks > 1) {
      for (int chunk_y = 0; chunk_y < height_chunks; ++chunk_y) {
        for (int chunk_x = 0; chunk_x < width_chunks; ++chunk_x) {
          const int row = ctx.mi_row + (chunk_y << kMaxChunkShift);
          const int col = ctx.mi_col + (chunk_x << kMaxChunkShift);
          set_tx_size(grid, row, col, 16, 16, 0, 0);
        }
      }
      if (partitioned != nullptr) *partitioned = true;
      return true;
    }
    const BlockTxContext bctx{mi, ctx.is_inter, ctx.fsc_mode,
                              ctx.reduced_tx_part_set};
    TxPartition part = TxPartition::kNone;
    SplitGeometry geom;
    if (!read_tx_partition(source, cdfs, bctx, ctx.mi_row, ctx.mi_col, &part,
                           &geom, error)) {
      return false;
    }
    for (int i = 0; i < geom.count; ++i) {
      const SplitPart& p = geom.parts[i];
      set_tx_size(grid, p.mi_row, p.mi_col, tx_height(p.tx_size) / kMiSize,
                  tx_width(p.tx_size) / kMiSize, 0, 0);
    }
    if (partitioned != nullptr) *partitioned = true;
    return true;
  }

  grid.set(ctx.mi_row, ctx.mi_col, block_height_mi(mi), block_width_mi(mi),
           max_rect);
  return true;
}

bool read_block_tx_size(SymbolSource& source, TxPartitionCdfs& cdfs,
                        const TxSizeContext& ctx, LumaTxGrid& grid,
                        std::string& error) {
  const int mi = ctx.mi_size;
  if (!valid_mi_size(mi)) {
    error = "invalid MiSize";
    return false;
  }
  const bool inter_path =
      ctx.tx_mode == TxMode::kSelect && mi != kBlock4x4 && ctx.is_inter &&
      !ctx.skip_flag && !ctx.lossless;
  if (!inter_path) {
    const bool allow_select = !ctx.skip_flag || !ctx.is_inter;
    return read_tx_size(source, cdfs, ctx, allow_select, grid, nullptr, error);
  }
  const int max_rect = max_tx_size_rect(mi);
  const int tx_w4 = tx_width(max_rect) / kMiSize;
  const int tx_h4 = tx_height(max_rect) / kMiSize;
  if (tx_w4 <= 0 || tx_h4 <= 0) {
    error = "invalid maximum transform size";
    return false;
  }
  const int bw4 = block_width_mi(mi);
  const int bh4 = block_height_mi(mi);
  for (int row = ctx.mi_row; row < ctx.mi_row + bh4; row += tx_h4) {
    for (int col = ctx.mi_col; col < ctx.mi_col + bw4; col += tx_w4) {
      const BlockTxContext bctx{mi, ctx.is_inter, ctx.fsc_mode,
                                ctx.reduced_tx_part_set};
      TxPartition part = TxPartition::kNone;
      SplitGeometry geom;
      if (!read_tx_partition(source, cdfs, bctx, row, col, &part, &geom,
                             error)) {
        return false;
      }
      for (int i = 0; i < geom.count; ++i) {
        const SplitPart& p = geom.parts[i];
        set_tx_size(grid, p.mi_row, p.mi_col, tx_height(p.tx_size) / kMiSize,
                    tx_width(p.tx_size) / kMiSize, 0, 0);
      }
    }
  }
  return true;
}

TxMode tx_mode_from_syntax(bool coded_lossless, bool tx_mode_select) {
  if (coded_lossless) return TxMode::kOnly4x4;
  return tx_mode_select ? TxMode::kSelect : TxMode::kLargest;
}

}  // namespace av2enc::video::txpartition
