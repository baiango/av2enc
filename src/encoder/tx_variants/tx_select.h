// TX split / partition selection hooks for the M1 transform-variants
// component.  These are encoder-side decisions only: nothing here writes
// bitstream syntax.  RDO can either call select_tx_partition() (greedy,
// SSE + lambda*nonzero) or use the lower-level geometry/cost helpers with its
// own cost function.
#pragma once

#include <cstdint>

#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::txv {

// A selected transform leaf, positioned relative to the root block.
struct TxLeaf {
  std::int16_t x = 0;
  std::int16_t y = 0;
  TxParams params;
};

struct TxSelection {
  int count = 0;
  TxLeaf leaves[16];       // max_depth 2 over a 64x64 block: 4^2 = 16
  double cost = 0.0;       // sse + lambda * nonzero
  double sse = 0.0;        // reconstruction SSE in the coded domain
  std::int32_t nonzero = 0;
  TxPartition root_partition = TxPartition::kNone;
};

struct TxSelectOptions {
  // Baseline transform type tried for every leaf.
  TxType tx_type = TxType::kDctDct;
  // Also try DCT_DCT, ADST_ADST, DCT_ADST, ADST_DCT, FLIPADST_FLIPADST and
  // IDTX per leaf (quantized-domain SSE decides).  ADST/FDST at a length-32
  // dimension alias DCT per §7.15.2.1.
  bool search_tx_type = false;
  // Number of partition levels below the root.  0 = NONE only; 1 = evaluate
  // NONE/SPLIT/HORZ/VERT at the root; 2 = recurse one more level.
  int max_depth = 1;
  // Rate proxy weight: cost = sse + lambda * nonzero_coefficients.
  double lambda = 0.0;
};

// Cost of one candidate transform block.  `residual_coded` is the root block
// in the coded (unadjusted) domain with row stride `block_w`.  (x0, y0) and
// (block_w, block_h) describe the candidate sub-block.  Returns false when the
// sub-block size is not an AV2 TX size.
bool tx_block_cost(const std::int16_t* residual_coded, int block_w,
                   int block_h, int x0, int y0, const TxParams& params,
                   double* cost, double* sse, std::int32_t* nonzero);

// Greedy partition/type selection.  `residual_coded` is `block_w * block_h`
// int16 residuals in raster order.  Returns false for an unsupported root
// size or when out is null.
bool select_tx_partition(const std::int16_t* residual_coded, int block_w,
                         int block_h, const TxSelectOptions& options,
                         TxSelection* out);

}  // namespace av2enc::txv
