// Per-worker scratch for the TX-type RDO search (S1 of
// docs/data-oriented.md).
//
// `evaluate_candidate()` used to build ten `std::vector`s per TX-type
// candidate (coded, adj, coeff/coeff_i, levels, scan, dequant, dequant_levels,
// recon, plus the §8.3 model inputs and the TCQ/RDOQ trellis inputs) and to
// regenerate the §5.20.7.30 scan even though it depends only on
// `(tx_size, class)`.  On the 60-frame 240p `--nightly` encode that is
// 1,886,163 allocations / 6.6 GB.
//
// The buffers below replace those vectors with fixed spans.  The transform
// work buffers are per *thread*: candidate evaluation is the unit of work
// handed to the worker pool, and no two candidates on the same thread are
// live at the same time.  The payload ring is per *caller*: it holds the
// levels/residual of every candidate of one leaf while the (sequential)
// selection pass decides which one wins, so losing candidates never own heap
// storage.
//
// Size: the static scratch is 100,704 bytes (98 KiB) per thread.  The 64-coded
// sizes are coded 64x64 -> 4096 samples but transform an adjusted 32x32 grid,
// so the coefficient-domain buffers hold 1024 entries and only `coded` /
// `recon` need the 4096-sample coded size.  The bulk is the two coefficient
// input arrays (16 KiB each); they are mutually exclusive at run time (TCQ vs
// RDOQ) but kept separate so the code stays type-safe.  Eight workers hold
// under 800 KiB in total, allocated once and reused for the whole encode.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/tcq/tcq_av2_model.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::video::txtsearch {

// 64x64 coded size; the transform itself is adjusted to 32x32.
inline constexpr int kScratchMaxCoded =
    txv::kMaxCodedTxDim * txv::kMaxCodedTxDim;
inline constexpr int kScratchMaxArea = txv::kMaxAdjTxElems;
inline constexpr int kScanCacheSlots = 4;

// One generated §5.20.7.30 scan.  `size` is `tx_scan_area(tx_size)`.
struct ScanCacheEntry {
  int tx_size = -1;
  txpartition::TxScanClass cls = txpartition::TxScanClass::k2D;
  int size = 0;
  bool valid = false;
  std::uint16_t data[kScratchMaxArea] = {};
};

struct TxScratch {
  // Coded-domain residual and the inverse transform's reconstruction.
  std::int16_t coded[kScratchMaxCoded];
  std::int16_t recon[kScratchMaxCoded];
  // Adjusted-domain work buffers (w * h <= 1024).
  std::int16_t adj[kScratchMaxArea];
  double coeff[kScratchMaxArea];             // txv forward coefficients
  std::int32_t coeff_i[kScratchMaxArea];     // DDT forward coefficients
  std::int32_t levels[kScratchMaxArea];      // raster, scalar or trellis
  std::int32_t dequant[kScratchMaxArea];     // inverse transform input
  std::int32_t dequant_levels[kScratchMaxArea];  // TCQ mapped Quant[]
  std::int32_t aux[kScratchMaxArea];         // TCQ reference / RDOQ scalar seed
  std::int32_t syntax_scan[kScratchMaxArea]; // block_syntax_bits scratch
  tcq::Coefficient tcq_input[kScratchMaxArea];
  rdoq::Coefficient rdoq_input[kScratchMaxArea];
  ScanCacheEntry scans[kScanCacheSlots];
  // The §8.3 models are rebuilt per block; keeping the objects alive reuses
  // their position tables' capacity instead of reallocating them per symbol
  // pass.
  tcq::av2::Model tcq_model;
  rdoq::av2::Model rdoq_model;

  // Returns the cached scan for `(tx_size, cls)`, generating it on the first
  // call.  The table is a pure function of those two values, so the result is
  // byte-for-byte what a fresh `get_scan()` produced.
  std::span<const std::uint16_t> scan_table(int tx_size,
                                            txpartition::TxScanClass cls,
                                            bool* ok) {
    const std::size_t slot = static_cast<std::size_t>(
        (tx_size * 3 + static_cast<int>(cls)) & (kScanCacheSlots - 1));
    ScanCacheEntry& entry = scans[slot];
    if (entry.valid && entry.tx_size == tx_size && entry.cls == cls) {
      *ok = true;
      return std::span<const std::uint16_t>(
          entry.data, static_cast<std::size_t>(entry.size));
    }
    const int area = txpartition::tx_scan_area(tx_size);
    if (area <= 0 || area > kScratchMaxArea) {
      *ok = false;
      entry.valid = false;
      return {};
    }
    const std::span<std::uint16_t> out(entry.data,
                                       static_cast<std::size_t>(area));
    if (!txpartition::get_scan(tx_size, cls, out)) {
      *ok = false;
      entry.valid = false;
      return {};
    }
    entry.tx_size = tx_size;
    entry.cls = cls;
    entry.size = area;
    entry.valid = true;
    *ok = true;
    return std::span<const std::uint16_t>(entry.data,
                                          static_cast<std::size_t>(area));
  }
};

inline TxScratch& tx_scratch() {
  static thread_local TxScratch scratch{};
  return scratch;
}

// Payload slots for the candidates of one leaf.  Sized from the leaf's own
// `tx_size`, so every slot span is exactly `area` / `coded_n` long.
struct CandidatePayloadRing {
  std::vector<std::int32_t> level_store;
  std::vector<std::int16_t> residual_store;
  std::size_t slots = 0;
  int area = 0;
  int coded_n = 0;

  void prepare(std::size_t count, int leaf_area, int leaf_coded_n) {
    slots = count;
    area = leaf_area;
    coded_n = leaf_coded_n;
    const std::size_t need_levels = count * static_cast<std::size_t>(leaf_area);
    const std::size_t need_residual =
        count * static_cast<std::size_t>(leaf_coded_n);
    if (level_store.size() < need_levels) level_store.resize(need_levels);
    if (residual_store.size() < need_residual) {
      residual_store.resize(need_residual);
    }
  }

  std::span<std::int32_t> levels(std::size_t slot) {
    return std::span<std::int32_t>(
        level_store.data() + slot * static_cast<std::size_t>(area),
        static_cast<std::size_t>(area));
  }

  std::span<std::int16_t> residual(std::size_t slot) {
    return std::span<std::int16_t>(
        residual_store.data() + slot * static_cast<std::size_t>(coded_n),
        static_cast<std::size_t>(coded_n));
  }
};

inline CandidatePayloadRing& candidate_ring() {
  static thread_local CandidatePayloadRing ring;
  return ring;
}

}  // namespace av2enc::video::txtsearch
