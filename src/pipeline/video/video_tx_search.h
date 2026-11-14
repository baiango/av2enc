// CPU reference for the M2 luma TX size / TX type / TX partition search.
//
// This is the decision half of the device module under
// `src/cuda/pipeline/video/tx/` (`av2enc::cuda::videotx`).  The CUDA product
// path runs that module on the device; this file is the CPU reference it must
// match bit-for-bit, so the two implementations deliberately use the same
// candidate sets, the same arithmetic and the same tie-breaks:
//
//   * candidate plans: `TX_PARTITION_NONE` (own max-rect TX size) plus every
//     `txpartition::legal_splits()` option the caller's mask allows, in
//     legal_splits() order;
//   * candidate types: the legal §5.20.8.3 set of every leaf (intra through
//     `txtype::intra_tx_candidates()`, inter through the §5.20.8.2 inverse
//     tables), with the §7.15.4.1 DDT guard and the 1-D-class opt-in;
//   * arithmetic: `txv::downsample_residual` -> `txv::forward_block` ->
//     the product `lround(coeff * (qfactor*528) / q)` quantiser -> §7.14.4
//     dequantise -> `txv::inverse_block`, SSE in the coded domain;
//   * rate: `eob == 0 ? skip_bits : type_bits[ctx] + coefficient_bits*nonzero`
//     plus the plan's estimated §5.20.6.3 partition bits by default; with
//     `use_context_rate` (`--video-tx-context-rate`, the RD preset behavior)
//     the §8.3.2 coefficient syntax cost from the static §9.3 CDF slice
//     (`txb_skip`, EOB, coeff_base/br, escape and sign) replaces the flat
//     per-nonzero heuristic.  Cost is `sse + lambda * rate`.
//   * tie-breaks: leaf `cost`, then `sse`, then lower TX_TYPE; plan `cost`,
//     then `sse`.
//
// The module owns the bitstream contract: this decision only produces the
// winning partition, per-leaf TX sizes/types/EOBs and scan-order levels.  The
// syntax writer emits them (`tx_mode_select`, `tx_do_partition`,
// `tx_partition_type`, `TX_TYPE`); nothing here writes bits.
//
// Chroma is deliberately out of scope: `enable_chroma_dctonly = 1` in the M2
// sequence header and `get_tx_size(plane > 0, ...)` ignores the luma partition,
// so chroma keeps one DCT per plane (see `src/codec/transform/tx_partition/`).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::video::txtsearch {

// §5.20.6.3 legal_splits() returns at most 8 options; every coding block in the
// M2 tree (<= 64x64) resolves to at most 4 transform leaves.
inline constexpr int kMaxLeaves = 4;

enum class Mode : std::int32_t {
  kIntra = 0,
  kInter = 1,
};

// Same meaning as `av2enc::cuda::videotx::VideoTxSearchOptions`.
struct Options {
  bool search_none = true;
  bool search_partition = true;
  bool search_type = true;
  bool include_1d_types = false;
  // §5.4.8 enable_idtx_intra.  IDTX (type 9) is a TX_CLASS_2D identity type;
  // when the sequence bit is 0 the decoder rejects it, so the candidate is
  // filtered unless the caller signals the flag.
  bool allow_idtx = false;
  // Partition bit mask, same values as VideoTxPartitionBit.
  std::int32_t partition_mask = 0x3f;
  std::int32_t reduced_tx_set = 0;
  bool reduced_tx_part_set = false;
  bool enable_inter_ddt = false;
  bool dct_only = false;
  std::int32_t intra_mapped_mode = 0;
  std::int32_t max_tx_size = 64;
  // RD weights: cost = sse + lambda * rate_bits.
  double lambda = 4.0;
  double coefficient_bits = 1.0;
  double skip_bits = 1.2;
  bool use_type_bits = true;
  // Price every candidate leaf with the real §8.3.2 coefficient syntax
  // (txb_skip, EOB prefix/extra, coeff_base/coeff_base_eob, coeff_br, the
  // read_quant escape and the sign) from the static §9.3 CDF slice, instead
  // of `coefficient_bits * nonzero`.  The measured RD preset
  // (`--video-tx-context-rate`) turns this on; standalone `--video-tx-search`
  // keeps the historical heuristic so the CUDA device TX module still matches
  // the CPU reference.  `tcq_qctx` selects the §8.3.1 slice.
  bool use_context_rate = false;
  // §5.20.7.27 TCQ (--video-tcq).  The luma candidate quantiser runs the TCQ
  // trellis at half the scalar step and the reconstruction uses the mapped
  // Quant[] values with the extra dqDenom shift.  `tcq_qctx` is the §8.3.1
  // COEFF_CDF_Q_CTXS slice from the frame base_q_idx.
  bool tcq = false;
  std::int32_t tcq_qctx = 0;
};

// One winning transform block.
struct Leaf {
  std::int32_t x = 0;  // coded-sample origin inside the coding block
  std::int32_t y = 0;
  std::int32_t coded_w = 0;
  std::int32_t coded_h = 0;
  std::int32_t tx_size = 0;
  std::int32_t tx_type = 0;
  std::int32_t mi_row = 0;  // 4x4 origin, relative to the coding block
  std::int32_t mi_col = 0;
  std::int32_t eob = 0;
  // Scan-order levels at the adjusted size (`params.w * params.h`).
  std::vector<std::int32_t> levels;
  // Reconstructed coded-size residual (prediction not added).
  std::vector<std::int16_t> residual;
};

struct Result {
  bool ok = false;
  std::int32_t partition = 0;  // av2enc::video::txpartition::TxPartition
  std::int32_t count = 0;
  Leaf leaves[kMaxLeaves];
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  // prediction + winning residual at the coded size (stride coded_w); filled
  // when `prediction` is non-null.
  std::vector<std::uint8_t> recon;
};

// Runs the whole search for one luma coding block.
//
// `residual_coded` is `coded_w * coded_h` int16, row-major, tightly packed.
// `prediction`, when non-null, is the same shape as uint8 and is used only to
// produce `Result::recon`.  `qindex` is the effective §5.20.1 CurrentQIndex and
// `delta_dc`/`delta_ac` the resolved §7.14.2 per-plane deltas.
//
// Returns false and fills `*error` for an invalid block size or an internal
// geometry failure.
bool search(const std::int16_t* residual_coded, std::int32_t coded_w,
            std::int32_t coded_h, Mode mode, std::int32_t qindex,
            std::int32_t delta_dc, std::int32_t delta_ac,
            const std::uint8_t* prediction, const Options& options,
            Result* out, std::string* error);

}  // namespace av2enc::video::txtsearch
