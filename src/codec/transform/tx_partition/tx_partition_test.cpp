// AV2 v1.0.0 transform-mode / transform-partition component test.
//
// Covers:
//   * §9.2/§9.3 tables and a pinned CDF fingerprint
//   * legal split enumeration for every BLOCK_SIZES entry, full and reduced set
//   * exact set_tx_size() geometry: coverage, ordering, bounds, TX sizes
//   * exhaustive (block size x partition x fsc_mode x is_inter x reduced-set)
//     writer -> independent reader round-trips
//   * read_tx_size()/read_block_tx_size() driver round-trips for every block
//     size and TX-mode/skip combination, including 64x64 chunking and the
//     4x64/64x4 transforms only reachable through TX partitioning
//   * §5.20.7.30 scans for all 25 sizes and all three TX_CLASS values,
//     including rectangular and HORIZ/VERT scans missing from scan.json
//   * pinned bitstream goldens and a per-decision bit-cost report
//
// Build/run: ./build_and_test.sh
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "pipeline/still/basic_tables.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_partition/tx_partition_entropy.h"
#include "codec/transform/tx_partition/tx_partition_tables.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/tx_type/tx_type_tables.h"
#include "codec/transform/variants/tx_variants.h"

namespace {

using namespace av2enc;
using namespace av2enc::video::txpartition;

int g_failures = 0;
long long g_checks = 0;
const char* g_suite = "";
long long g_suite_start = 0;
bool g_print_goldens = false;

void suite_start(const char* name) {
  g_suite = name;
  g_suite_start = g_checks;
}

void suite_end() {
  std::printf("  %-32s %12lld checks\n", g_suite, g_checks - g_suite_start);
}

void check(bool cond, const char* what, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("FAIL [%s:%d] %s\n", g_suite, line, what);
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

std::string hex(std::span<const std::uint8_t> bytes) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::uint8_t b : bytes) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 0xF]);
  }
  return out;
}

struct Encoded {
  std::vector<std::uint8_t> bytes;
  bool ok = false;
  std::string error;
};

Encoded encode_partition(const BlockTxContext& ctx, TxPartition part,
                         bool reduced) {
  Encoded result;
  TxPartitionCdfs cdfs;
  cdfs.reset(reduced);
  TileBitWriter writer;
  SplitGeometry geom;
  if (!write_tx_partition(writer.sink(), cdfs, ctx, part, &geom,
                          result.error)) {
    return result;
  }
  std::span<const std::uint8_t> bytes;
  if (!writer.finish(&bytes, result.error)) return result;
  result.bytes.assign(bytes.begin(), bytes.end());
  result.ok = true;
  return result;
}

// Decodes a partition and compares decision + geometry.
bool round_trip(const BlockTxContext& ctx, TxPartition part, bool reduced,
                std::string& error) {
  const Encoded encoded = encode_partition(ctx, part, reduced);
  if (!encoded.ok) {
    error = encoded.error;
    return false;
  }
  TxPartitionCdfs cdfs;
  cdfs.reset(reduced);
  TileBitReader reader;
  if (!reader.init(encoded.bytes, error)) return false;
  TxPartition decoded = TxPartition::kNone;
  SplitGeometry geom;
  if (!read_tx_partition(reader.source(), cdfs, ctx, 0, 0, &decoded, &geom,
                         error)) {
    return false;
  }
  if (decoded != part) {
    error = std::string("decision mismatch: wrote ") +
            tx_partition_name(part) + ", read " +
            tx_partition_name(decoded);
    return false;
  }
  SplitGeometry expected;
  if (!tx_partition_geometry(ctx.mi_size, 0, 0, part, &expected, &error)) {
    return false;
  }
  if (geom.count != expected.count) {
    error = "geometry count mismatch";
    return false;
  }
  for (int i = 0; i < geom.count; ++i) {
    if (geom.parts[i].mi_row != expected.parts[i].mi_row ||
        geom.parts[i].mi_col != expected.parts[i].mi_col ||
        geom.parts[i].tx_size != expected.parts[i].tx_size) {
      error = "geometry part mismatch";
      return false;
    }
  }
  if (!reader.finish(error)) return false;
  return true;
}

bool grid_equal(const LumaTxGrid& a, const LumaTxGrid& b) {
  if (a.mi_rows != b.mi_rows || a.mi_cols != b.mi_cols) return false;
  for (int r = 0; r < a.mi_rows; ++r) {
    for (int c = 0; c < a.mi_cols; ++c) {
      if (a.at(r, c) != b.at(r, c)) return false;
      if (a.middle_at(r, c) != b.middle_at(r, c)) return false;
      if (a.scan_order_at(r, c) != b.scan_order_at(r, c)) return false;
    }
  }
  return true;
}

std::string grid_string(const LumaTxGrid& grid) {
  std::string out;
  for (int r = 0; r < grid.mi_rows; ++r) {
    for (int c = 0; c < grid.mi_cols; ++c) {
      out += std::to_string(static_cast<int>(grid.at(r, c)));
      out += ',';
    }
  }
  return out;
}

void suite_tables() {
  suite_start("tables");
  CHECK(kTxPartitionMaxTxSizeRect[0] == kTxSize4x4);
  CHECK(kTxPartitionMaxTxSizeRect[12] == kTxSize64x64);   // BLOCK_64X64
  CHECK(kTxPartitionMaxTxSizeRect[13] == kTxSize64x64);   // BLOCK_64X128
  CHECK(kTxPartitionMaxTxSizeRect[18] == kTxSize64x64);   // BLOCK_256X256
  CHECK(kTxPartitionMaxTxSizeRect[19] == 13);             // 4x16 -> TX_4X16
  CHECK(kTxPartitionMaxTxSizeRect[27] == 21);             // 8x64 -> TX_8X64
  CHECK(kTxPartitionMaxTxSizeRect[28] == 22);             // 64x8 -> TX_64X8

  CHECK(kTxPartitionBlockGroup[0] == 0);
  CHECK(kTxPartitionBlockGroup[3] == 1);
  CHECK(kTxPartitionBlockGroup[12] == 7);
  CHECK(kTxPartitionBlockGroup[18] == 7);
  CHECK(kTxPartitionBlockGroup[19] == 8);
  CHECK(kTxPartitionBlockGroup[28] == 8);

  CHECK(kTxPartitionTypeGroupVertAndHorz[3] == 0);   // 8x8
  CHECK(kTxPartitionTypeGroupVertAndHorz[12] == 9);  // 64x64
  CHECK(kTxPartitionTypeGroupVertAndHorz[19] == 255);
  CHECK(kTxPartitionTypeGroupVertOrHorz[19] == 1);   // 4x16
  CHECK(kTxPartitionTypeGroupVertOrHorz[20] == 2);   // 16x4
  CHECK(kTxPartitionTypeGroupVertOrHorz[25] == 1);   // 4x32
  CHECK(kTxPartitionTypeGroupVertOrHorz[26] == 2);   // 32x4
  CHECK(kTxPartitionTypeGroupVertOrHorz[3] == 255);

  TxPartitionCdfs cdfs;
  cdfs.reset(false);
  // §9.3 spot values: first and last tx_do_partition rows, first
  // tx_partition_type row, and the uniform reduced table.
  CHECK(cdfs.tx_do_partition[0][0][0][0] == 26330);
  CHECK(cdfs.tx_do_partition[0][1][8][0] == 24696);
  CHECK(cdfs.tx_do_partition[1][1][0][0] == 16384);
  CHECK(cdfs.tx_partition_type[0][0][0][0] == 32744);
  CHECK(cdfs.tx_partition_type[0][0][0][6] == 60);
  CHECK(cdfs.tx_partition_type[1][1][13][0] == 4681);
  CHECK(cdfs.tx_partition_type_reduced[0][0][0][0] == 32744);
  CHECK(cdfs.tx_2or3_partition_type[0][0][0][0] == 25131);
  CHECK(cdfs.tx_2or3_partition_type[1][1][1][0] == 16384);

  TxPartitionCdfs reduced;
  reduced.reset(true);
  CHECK(reduced.tx_partition_type_reduced[0][0][0][0] == 4681);
  CHECK(reduced.tx_partition_type_reduced[1][1][13][0] == 4681);

  // Pinned table fingerprint (full set); regenerate with --print-goldens if
  // the §9.3 source tables change.
  const std::uint64_t fingerprint = cdf_fingerprint(cdfs);
  if (g_print_goldens) {
    std::printf("GOLDEN cdf_fingerprint %016llx\n",
                static_cast<unsigned long long>(fingerprint));
  }
  constexpr std::uint64_t kPinnedFingerprint = 0x59b1af837dc95e25ULL;
  if (kPinnedFingerprint != 0) {
    CHECK(fingerprint == kPinnedFingerprint);
  }
  suite_end();
}

void suite_planner() {
  suite_start("planner");
  for (int mi = 0; mi < kBlockCount; ++mi) {
    for (int reduced = 0; reduced < 2; ++reduced) {
      const LegalSplits legal = legal_splits(mi, reduced != 0);
      if (mi == kBlock4x4) {
        CHECK(!legal.applicable);
        CHECK(legal.option_count == 0);
        continue;
      }
      const int bw = block_width_px(mi);
      const int bh = block_height_px(mi);
      if (bw > 64 || bh > 64) {
        CHECK(legal.chunked);
        CHECK(!legal.applicable);
        CHECK(legal.option_count == 0);
        CHECK(tx_chunk_cols(mi) >= 1);
        CHECK(tx_chunk_rows(mi) >= 1);
        continue;
      }
      CHECK(legal.applicable);
      CHECK(!legal.chunked);
      CHECK(legal.option_count >= 1);
      CHECK(legal.options[0] == TxPartition::kNone);

      bool seen[8] = {};
      for (int i = 0; i < legal.option_count; ++i) {
        const int index = static_cast<int>(legal.options[i]);
        CHECK(index >= 0 && index < 8);
        CHECK(!seen[index]);
        seen[index] = true;
        CHECK(legal.options[i] != TxPartition::kHorz5);
        CHECK(legal.options[i] != TxPartition::kVert5);
      }
      if (legal.has_partition_type) {
        CHECK(legal.allow_horz && legal.allow_vert);
        CHECK(legal.option_count == 4);
        CHECK(seen[static_cast<int>(TxPartition::kSplit)]);
        CHECK(seen[static_cast<int>(TxPartition::kHorz)]);
        CHECK(seen[static_cast<int>(TxPartition::kVert)]);
      }
      if (legal.has_2or3) {
        CHECK(!legal.has_partition_type);
        CHECK(legal.option_count == (reduced != 0 ? 2 : 3));
        if (legal.allow_horz) {
          CHECK(seen[static_cast<int>(TxPartition::kHorz)]);
          CHECK(seen[static_cast<int>(TxPartition::kHorz4)] == (reduced == 0));
        } else {
          CHECK(seen[static_cast<int>(TxPartition::kVert)]);
          CHECK(seen[static_cast<int>(TxPartition::kVert4)] == (reduced == 0));
        }
      }
      if (!legal.has_partition_type && !legal.has_2or3) {
        CHECK(legal.option_count == 2);
      }
    }
  }
  suite_end();
}

void suite_geometry() {
  suite_start("geometry");
  for (int mi = 0; mi < kBlockCount; ++mi) {
    const LegalSplits legal = legal_splits(mi, false);
    if (!legal.applicable) continue;
    const int bw4 = block_width_mi(mi);
    const int bh4 = block_height_mi(mi);
    const int area = bw4 * bh4;
    for (int i = 0; i < legal.option_count; ++i) {
      std::vector<int> covered(static_cast<std::size_t>(area), 0);
      SplitGeometry geom;
      std::string error;
      CHECK(tx_partition_geometry(mi, 0, 0, legal.options[i], &geom, &error));
      CHECK(geom.count >= 1 && geom.count <= 4);
      int total = 0;
      int last_row = -1;
      int last_col = -1;
      for (int p = 0; p < geom.count; ++p) {
        const SplitPart& part = geom.parts[p];
        CHECK(part.tx_size >= 0 && part.tx_size < 25);
        CHECK(part.tx_size == find_tx_size(tx_width(part.tx_size),
                                           tx_height(part.tx_size)));
        const int w4 = tx_width(part.tx_size) / 4;
        const int h4 = tx_height(part.tx_size) / 4;
        CHECK(w4 > 0 && h4 > 0);
        CHECK(part.mi_col >= 0 && part.mi_row >= 0);
        CHECK(part.mi_col + w4 <= bw4);
        CHECK(part.mi_row + h4 <= bh4);
        // Decode order is row-major (mi_row then mi_col).
        CHECK(part.mi_row > last_row ||
              (part.mi_row == last_row && part.mi_col > last_col));
        last_row = part.mi_row;
        last_col = part.mi_col;
        for (int y = 0; y < h4; ++y) {
          for (int x = 0; x < w4; ++x) {
            covered[static_cast<std::size_t>((part.mi_row + y) * bw4 +
                                             part.mi_col + x)] += 1;
          }
        }
        total += w4 * h4;
      }
      CHECK(total == area);
      for (int c = 0; c < area; ++c)
        CHECK(covered[static_cast<std::size_t>(c)] == 1);
    }
  }
  // Tiles with zero width/height must be rejected.
  SplitGeometry geom;
  std::string error;
  CHECK(!tx_partition_geometry(kBlock4x4, 0, 0, TxPartition::kNone, &geom,
                               &error));
  CHECK(!tx_partition_geometry(27, 0, 0, TxPartition::kHorz5, &geom, &error));
  suite_end();
}

void suite_round_trip() {
  suite_start("partition round-trip");
  for (int mi = 0; mi < kBlockCount; ++mi) {
    for (int reduced = 0; reduced < 2; ++reduced) {
      const LegalSplits legal = legal_splits(mi, reduced != 0);
      if (!legal.applicable) continue;
      for (int opt = 0; opt < legal.option_count; ++opt) {
        for (int fsc = 0; fsc < 2; ++fsc) {
          for (int inter = 0; inter < 2; ++inter) {
            BlockTxContext ctx;
            ctx.mi_size = mi;
            ctx.fsc_mode = fsc != 0;
            ctx.is_inter = inter != 0;
            ctx.reduced_tx_part_set = reduced != 0;
            std::string error;
            if (!round_trip(ctx, legal.options[opt], reduced != 0, error)) {
              ++g_failures;
              std::printf("FAIL round-trip mi=%d part=%s fsc=%d inter=%d "
                          "reduced=%d: %s\n",
                          mi, tx_partition_name(legal.options[opt]), fsc,
                          inter, reduced, error.c_str());
            }
            ++g_checks;
          }
        }
      }
    }
  }
  suite_end();
}

// Drives write_tx_size/read_tx_size or write_block_tx_size/read_block_tx_size
// with a deterministic decision sequence and compares the resulting grids.
void suite_drivers() {
  suite_start("tx-size drivers");
  for (int mi = 0; mi < kBlockCount; ++mi) {
    if (mi == kBlock4x4) continue;
    const int bw4 = block_width_mi(mi);
    const int bh4 = block_height_mi(mi);
    const int cols = bw4 + 2;
    const int rows = bh4 + 2;
    const LegalSplits legal = legal_splits(mi, false);
    for (int tx_mode = 0; tx_mode < 2; ++tx_mode) {
      for (int is_inter = 0; is_inter < 2; ++is_inter) {
        for (int skip = 0; skip < 2; ++skip) {
          TxSizeContext ctx;
          ctx.mi_size = mi;
          ctx.mi_row = 1;
          ctx.mi_col = 1;
          ctx.mi_rows = rows;
          ctx.mi_cols = cols;
          ctx.is_inter = is_inter != 0;
          ctx.skip_flag = skip != 0;
          ctx.tx_mode = tx_mode ? TxMode::kSelect : TxMode::kLargest;
          const bool inter_path = ctx.tx_mode == TxMode::kSelect &&
                                  mi != kBlock4x4 && ctx.is_inter &&
                                  !ctx.skip_flag;
          const bool can_partition =
              legal.applicable && (inter_path ||
                                   (!skip || !is_inter));
          // Alternate NONE / last legal option per chunk to exercise both.
          const auto decider = [&](int chunk, int, int, TxPartition* out,
                                   std::string&) {
            if (!can_partition || chunk % 2 == 0) {
              *out = TxPartition::kNone;
            } else {
              *out = legal.options[legal.option_count - 1];
            }
            return true;
          };
          TileBitWriter writer;
          TxPartitionCdfs cdfs;
          cdfs.reset(false);
          LumaTxGrid written;
          written.reset(rows, cols);
          std::string error;
          bool partitioned = false;
          const bool ok =
              inter_path
                  ? write_block_tx_size(writer.sink(), cdfs, ctx, decider,
                                        written, error)
                  : write_tx_size(writer.sink(), cdfs, ctx,
                                  !ctx.skip_flag || !ctx.is_inter, decider,
                                  written, &partitioned, error);
          CHECK(ok);

          TxPartitionCdfs read_cdfs;
          read_cdfs.reset(false);
          LumaTxGrid read_grid;
          read_grid.reset(rows, cols);
          std::span<const std::uint8_t> bytes;
          if (ok) CHECK(writer.finish(&bytes, error));
          TileBitReader reader;
          if (ok) CHECK(reader.init(bytes, error));
          bool read_partitioned = false;
          const bool read_ok =
              ok && (inter_path
                         ? read_block_tx_size(reader.source(), read_cdfs, ctx,
                                              read_grid, error)
                         : read_tx_size(reader.source(), read_cdfs, ctx,
                                        !ctx.skip_flag || !ctx.is_inter,
                                        read_grid, &read_partitioned, error));
          CHECK(read_ok);
          if (read_ok) {
            if (!grid_equal(written, read_grid)) {
              ++g_failures;
              std::printf("FAIL driver mi=%d mode=%d inter=%d skip=%d "
                          "grid mismatch\n%s\n%s\n",
                          mi, tx_mode, is_inter, skip,
                          grid_string(written).c_str(),
                          grid_string(read_grid).c_str());
            }
            ++g_checks;
            CHECK(partitioned == read_partitioned);
          }
        }
      }
    }
  }
  suite_end();
}

void suite_reachability() {
  suite_start("4x64 reachability");
  // Audit §12.1 says TX_4X64/TX_64X4 are never signallable.  With TX
  // partitioning they are: BLOCK_8X64 VERT -> TX_4X64, BLOCK_64X8 HORZ ->
  // TX_64X4.
  SplitGeometry geom;
  std::string error;
  CHECK(tx_partition_geometry(27, 0, 0, TxPartition::kVert, &geom, &error));
  CHECK(geom.count == 2);
  CHECK(geom.parts[0].tx_size == kTxSize4x64);
  CHECK(tx_partition_geometry(28, 0, 0, TxPartition::kHorz, &geom, &error));
  CHECK(geom.count == 2);
  CHECK(geom.parts[0].tx_size == kTxSize64x4);
  // And BLOCK_8X32 SPLIT -> TX_4X16, BLOCK_16X16 SPLIT -> TX_8X8.
  CHECK(tx_partition_geometry(21, 0, 0, TxPartition::kSplit, &geom, &error));
  CHECK(geom.parts[0].tx_size == 13);  // TX_4X16
  CHECK(tx_partition_geometry(6, 0, 0, TxPartition::kSplit, &geom, &error));
  CHECK(geom.count == 4);
  for (int i = 0; i < 4; ++i) CHECK(geom.parts[i].tx_size == kTxSize8x8);

  // Every one of the 25 TX sizes is reachable either as a block's
  // Max_Tx_Size_Rect or as a transform-partition part.
  std::vector<int> reachable(25, 0);
  for (int mi = 0; mi < kBlockCount; ++mi) {
    const int max_rect = max_tx_size_rect(mi);
    if (max_rect >= 0) reachable[static_cast<std::size_t>(max_rect)] = 1;
    const LegalSplits legal = legal_splits(mi, false);
    for (int i = 0; i < legal.option_count; ++i) {
      if (!tx_partition_geometry(mi, 0, 0, legal.options[i], &geom, &error)) {
        continue;
      }
      for (int p = 0; p < geom.count; ++p) {
        reachable[static_cast<std::size_t>(geom.parts[p].tx_size)] = 1;
      }
    }
  }
  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    if (!reachable[static_cast<std::size_t>(tx)]) {
      ++g_failures;
      std::printf("FAIL [reachability] TX size %d is unreachable\n", tx);
    }
    ++g_checks;
  }
  suite_end();
}

void suite_scan() {
  suite_start("scans");
  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    const int w = tx_scan_width(tx);
    const int h = tx_scan_height(tx);
    const int area = w * h;
    CHECK(w >= 4 && w <= 32);
    CHECK(h >= 4 && h <= 32);
    CHECK(area > 0);
    for (int cls = 0; cls < 3; ++cls) {
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      CHECK(get_scan(tx, static_cast<TxScanClass>(cls), scan));
      std::vector<int> seen(static_cast<std::size_t>(area), 0);
      for (int i = 0; i < area; ++i) {
        CHECK(scan[static_cast<std::size_t>(i)] < area);
        seen[scan[static_cast<std::size_t>(i)]] += 1;
      }
      for (int i = 0; i < area; ++i)
        CHECK(seen[static_cast<std::size_t>(i)] == 1);
      // HORIZ is column-major in §5.20.7.30 (outer loop over x).
      if (cls == static_cast<int>(TxScanClass::kHoriz)) {
        for (int i = 0; i < area; ++i) {
          const int x = i / h;
          const int y = i % h;
          CHECK(scan[static_cast<std::size_t>(i)] ==
                static_cast<std::uint16_t>(y * w + x));
        }
      }
      // VERT is row-major (outer loop over y).
      if (cls == static_cast<int>(TxScanClass::kVert)) {
        for (int i = 0; i < area; ++i) {
          CHECK(scan[static_cast<std::size_t>(i)] ==
                static_cast<std::uint16_t>(i));
        }
      }
    }
  }

  // Square TX_CLASS_2D scans must equal the committed m0/tx_type tables.
  {
    std::vector<std::uint16_t> scan(1024);
    CHECK(get_scan(kTxSize4x4, TxScanClass::k2D, scan));
    for (int i = 0; i < 16; ++i)
      CHECK(scan[static_cast<std::size_t>(i)] == txtype::kScan4[i]);
    CHECK(get_scan(kTxSize8x8, TxScanClass::k2D, scan));
    for (int i = 0; i < 64; ++i)
      CHECK(scan[static_cast<std::size_t>(i)] == txtype::kScan8[i]);
    CHECK(get_scan(kTxSize16x16, TxScanClass::k2D, scan));
    for (int i = 0; i < 256; ++i)
      CHECK(scan[static_cast<std::size_t>(i)] == av2enc::basic::kScan16[i]);
    CHECK(get_scan(kTxSize32x32, TxScanClass::k2D, scan));
    for (int i = 0; i < 1024; ++i)
      CHECK(scan[static_cast<std::size_t>(i)] == av2enc::basic::kScan32[i]);
  }

  // Rectangular 2D scans: independent anti-diagonal traversal.
  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    const int w = tx_scan_width(tx);
    const int h = tx_scan_height(tx);
    if (w == h) continue;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(w * h));
    CHECK(get_scan(tx, TxScanClass::k2D, scan));
    int index = 0;
    for (int sum = 0; sum <= w + h - 2; ++sum) {
      // Within one anti-diagonal the spec walks from the bottom-left sample
      // (largest y) to the top-right sample (largest x).
      for (int y = h - 1; y >= 0; --y) {
        const int x = sum - y;
        if (x < 0 || x >= w) continue;
        CHECK(scan[static_cast<std::size_t>(index++)] ==
              static_cast<std::uint16_t>(y * w + x));
      }
    }
    CHECK(index == w * h);
  }

  // get_tx_row_col is the inverse of the adjusted raster position.
  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    const int w = tx_scan_width(tx);
    const int h = tx_scan_height(tx);
    for (int pos = 0; pos < w * h; ++pos) {
      int row = 0;
      int col = 0;
      CHECK(get_tx_row_col(pos, tx, &row, &col));
      CHECK(row >= 0 && row < h && col >= 0 && col < w);
      CHECK(row * w + col == pos);
    }
  }

  // get_lf_limits spot values.
  CHECK(get_lf_limits(0, 3, TxScanClass::k2D, 0));
  CHECK(!get_lf_limits(2, 2, TxScanClass::k2D, 0));
  CHECK(get_lf_limits(0, 0, TxScanClass::k2D, 1));
  CHECK(get_lf_limits(0, 1, TxScanClass::kHoriz, 0));
  CHECK(!get_lf_limits(0, 2, TxScanClass::kHoriz, 0));
  CHECK(get_lf_limits(1, 0, TxScanClass::kVert, 0));
  CHECK(!get_lf_limits(2, 0, TxScanClass::kVert, 0));

  // tx_class_of() matches §5.20.7.30.
  CHECK(tx_class_of(0) == TxScanClass::k2D);
  CHECK(tx_class_of(9) == TxScanClass::k2D);   // IDTX is 2D
  CHECK(tx_class_of(10) == TxScanClass::kVert);
  CHECK(tx_class_of(13) == TxScanClass::kHoriz);
  CHECK(tx_class_of(15) == TxScanClass::kHoriz);
  suite_end();
}

void suite_adaptation() {
  suite_start("cdf adaptation");
  // 16x16, inter, full set: alternate SPLIT/HORZ/NONE so all three CDFs used
  // (do_partition and tx_partition_type) adapt.
  BlockTxContext ctx;
  ctx.mi_size = 6;
  ctx.is_inter = true;
  const TxPartition sequence[] = {
      TxPartition::kSplit, TxPartition::kHorz, TxPartition::kNone,
      TxPartition::kVert,  TxPartition::kSplit, TxPartition::kNone,
      TxPartition::kHorz,  TxPartition::kSplit};
  TxPartitionCdfs write_cdfs;
  write_cdfs.reset(false);
  const std::uint64_t before = cdf_fingerprint(write_cdfs);
  TileBitWriter writer;
  std::string error;
  for (TxPartition part : sequence) {
    SplitGeometry geom;
    CHECK(write_tx_partition(writer.sink(), write_cdfs, ctx, part, &geom,
                             error));
  }
  const std::uint64_t after = cdf_fingerprint(write_cdfs);
  CHECK(after != before);
  std::span<const std::uint8_t> bytes;
  CHECK(writer.finish(&bytes, error));

  TxPartitionCdfs read_cdfs;
  read_cdfs.reset(false);
  TileBitReader reader;
  CHECK(reader.init(bytes, error));
  for (TxPartition part : sequence) {
    TxPartition decoded = TxPartition::kNone;
    SplitGeometry geom;
    CHECK(read_tx_partition(reader.source(), read_cdfs, ctx, 0, 0, &decoded,
                            &geom, error));
    CHECK(decoded == part);
  }
  CHECK(cdf_fingerprint(read_cdfs) == after);
  CHECK(reader.finish(error));
  suite_end();
}

// Pinned bitstream goldens.  Regenerate with `--print-goldens`.
struct Golden {
  int mi_size;
  bool fsc;
  bool inter;
  bool reduced;
  TxPartition part;
  const char* hex;
};

constexpr Golden kGoldens[] = {
    {6, false, false, false, TxPartition::kNone, "4000"},
    {6, false, false, false, TxPartition::kSplit, "ac0000"},
    {6, false, true, false, TxPartition::kHorz, "f70000"},
    {6, true, false, false, TxPartition::kVert, "f40000"},
    {19, false, false, false, TxPartition::kHorz4, "f00000"},
    {19, false, false, true, TxPartition::kHorz, "c000"},
    {12, false, false, false, TxPartition::kVert, "c80000"},
    {27, true, true, false, TxPartition::kVert, "a80000"},
};

void suite_goldens() {
  suite_start("bitstream goldens");
  for (const Golden& golden : kGoldens) {
    BlockTxContext ctx;
    ctx.mi_size = golden.mi_size;
    ctx.fsc_mode = golden.fsc;
    ctx.is_inter = golden.inter;
    ctx.reduced_tx_part_set = golden.reduced;
    const Encoded encoded =
        encode_partition(ctx, golden.part, golden.reduced);
    CHECK(encoded.ok);
    if (!encoded.ok) continue;
    const std::string got = hex(encoded.bytes);
    if (g_print_goldens) {
      std::printf("GOLDEN mi=%d fsc=%d inter=%d reduced=%d part=%s bytes=%s\n",
                  golden.mi_size, golden.fsc ? 1 : 0, golden.inter ? 1 : 0,
                  golden.reduced ? 1 : 0,
                  tx_partition_name(golden.part), got.c_str());
    }
    if (golden.hex[0] == '\0') {
      // Not pinned yet: fail so a placeholder cannot be mistaken for a pass.
      ++g_failures;
      std::printf("FAIL [goldens] unpinned golden mi=%d part=%s\n",
                  golden.mi_size, tx_partition_name(golden.part));
    } else {
      CHECK(got == golden.hex);
      if (got != golden.hex) {
        std::printf("  mi=%d part=%s got=%s want=%s\n", golden.mi_size,
                    tx_partition_name(golden.part), got.c_str(), golden.hex);
      }
    }
  }
  suite_end();
}

void report_bit_costs() {
  std::printf("\nTransform-partition syntax bit cost, CDF frozen "
              "(exact range-coder cost, flush amortised over 4096 calls)\n");
  struct Case {
    int mi_size;
    const char* name;
  };
  constexpr Case kCases[] = {
      {3, "BLOCK_8X8"},    {6, "BLOCK_16X16"},  {9, "BLOCK_32X32"},
      {12, "BLOCK_64X64"}, {19, "BLOCK_4X16"},  {27, "BLOCK_8X64"},
      {4, "BLOCK_8X16"},   {7, "BLOCK_16X32"},  {28, "BLOCK_64X8"},
  };
  constexpr int kRepeats = 4096;
  for (const Case& c : kCases) {
    const LegalSplits legal = legal_splits(c.mi_size, false);
    if (!legal.applicable) continue;
    std::printf("  %-12s", c.name);
    for (int i = 0; i < legal.option_count; ++i) {
      BlockTxContext ctx;
      ctx.mi_size = c.mi_size;
      TxPartitionCdfs cdfs;
      cdfs.reset(false);
      TileBitWriter writer(1U << 20);
      writer.disable_cdf_update(true);
      std::string error;
      double bits = 0.0;
      bool ok = true;
      for (int r = 0; r < kRepeats && ok; ++r) {
        SplitGeometry geom;
        ok = write_tx_partition(writer.sink(), cdfs, ctx, legal.options[i],
                                &geom, error);
      }
      std::span<const std::uint8_t> bytes;
      if (ok) ok = writer.finish(&bytes, error);
      if (ok) {
        bits = static_cast<double>(bytes.size()) * 8.0 / kRepeats;
        std::printf("  %s=%.2fb", tx_partition_name(legal.options[i]), bits);
      }
    }
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--print-goldens") == 0) {
      g_print_goldens = true;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("usage: %s [--print-goldens]\n", argv[0]);
      return 0;
    }
  }

  std::printf("AV2 transform-mode / transform-partition component test\n");
  suite_tables();
  suite_planner();
  suite_geometry();
  suite_round_trip();
  suite_drivers();
  suite_reachability();
  suite_scan();
  suite_adaptation();
  suite_goldens();
  report_bit_costs();

  std::printf("\n%lld checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
