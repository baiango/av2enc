// M2 inter-residual decision path: inter prediction + source -> levels/EOB/
// reconstruction, with an explicit skip-vs-residual cost.
//
// This is the piece the M2 core is missing: every P-frame leaf is currently
// `skip_flag = 1`, so inter prediction is never refined by a transform.  The
// component owns *decisions*, not syntax: `inter_coeff_syntax.*` writes the
// bits and `inter_tx_type.*` owns the inter TX-type set/symbol rules.  All
// transform/quantiser/RDOQ arithmetic comes from the shared components:
//
//   * `txv::forward_transform` / `txv::inverse_transform` (src/encoder/
//     tx_variants) for forward, quantise, dequantise and inverse,
//   * `rdoq::rdoq_quantize` for the optional trellis,
//   * `rdo::lambda_sse_from_quant_step` for lambda,
//   * `txtype::diagonal_scan`-compatible `av2enc::video::syntax::build_scan` for the scan.
//
// TX sizes.  M2 signals `tx_mode_select = 0` (TX_MODE_LARGEST) and
// `read_block_tx_size()` emits no bits for a non-skip inter block, so each
// plane uses exactly one transform block equal to its residual block size:
// luma = the coding block size, chroma = the 4:2:0 chroma residual size.  Both
// are `Max_Tx_Size_Rect`, which is what `inter_tx_sizes()` returns.  If a
// future frame enables TX_MODE_SELECT, `read_block_tx_size()` starts coding
// `tx_do_partition` / `tx_partition_type` and the luma plane may be split into
// several TX blocks (LumaTxSizes / LumaTxMiddle / LumaTxScanOrder); that path
// is deliberately not modelled here.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "codec/transform/inter_residual/inter_coeff_syntax.h"
#include "codec/transform/inter_residual/inter_tx_type.h"

namespace av2enc::video::residual {

struct PlaneView {
  std::span<const std::uint8_t> data;  // tightly packed rows
  int stride = 0;
};

// Geometry of one inter leaf.  Chroma is the 4:2:0 residual block size
// (`partition::chroma_plane_residual_size`), i.e. the luma block halved and
// clamped to the minimum legal block.
struct BlockGeometry {
  int mi_row = 0;
  int mi_col = 0;
  int luma_w = 0;
  int luma_h = 0;
  int chroma_w = 0;
  int chroma_h = 0;
};

// §5.20.7.25/26 TX sizes for TX_MODE_LARGEST.  Returns false when the coded
// size is not one of the 25 AV2 transforms.
bool inter_tx_sizes(const BlockGeometry& geometry, int* luma_tx_size,
                    int* chroma_tx_size);

struct DecisionOptions {
  // Quantiser: `base_q_idx` selects the §7.14.2 step used by txv.
  int base_q_idx = 90;
  // SSE-domain lambda.  <= 0 derives it with rdo::lambda_sse_from_quant_step().
  double lambda = 0.0;
  double lambda_sse_scale = 0.125;
  // Emit a residual at all.  false forces the skip decision (used by the gate's
  // zero-residual parity mode).
  bool allow_residual = true;
  // Search inter TX types for luma (chroma is DCT_DCT under
  // enable_chroma_dctonly).  With enable_inter_ddt = false the candidate set
  // excludes 1D-identity combinations.
  bool try_inter_tx_types = true;
  double tx_type_sse_scale = 0.125;
  // Full-trellis RDOQ via src/encoder/rdoq (same levels semantics as M1).
  bool rdoq = false;
  double rdoq_sse_scale = 0.125;
  bool enable_inter_ddt = false;
  int reduced_tx_set = 0;
  bool enable_chroma_dctonly = true;
};

// One plane's residual decision.  `levels` is the adjusted-size coefficient
// array in raster order, the layout `txv::*` produces and consumes;
// `scan_levels()` converts to the scan order the syntax writer wants.
struct PlaneResidual {
  int tx_size = 0;  // txv::TxSize
  int tx_type = 0;  // txv::TxType
  int coded_w = 0;
  int coded_h = 0;
  int adj_w = 0;
  int adj_h = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;
  std::vector<std::uint8_t> recon;  // coded-size reconstruction samples
};

struct BlockDecision {
  bool skip = true;  // true => no residual syntax at all
  bool residual_considered = false;
  PlaneResidual plane[kResidualPlaneCount];
  // Costs are SSE + lambda * bits over the three planes.
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  double skip_distortion = 0.0;
  double skip_rate_bits = 0.0;
  double skip_cost = 0.0;
  bool zero_residual = false;  // residual existed but every plane quantised to 0
};

struct BlockInput {
  BlockGeometry geometry;
  // Coded-size source and inter-prediction samples per plane (tightly packed,
  // `stride` rows of luma_w / chroma_w samples).
  PlaneView source[kResidualPlaneCount];
  PlaneView prediction[kResidualPlaneCount];
};

// Decides skip vs inter residual for one block and returns the quantised
// levels plus the reconstruction the decoder will produce.
bool decide_inter_block(const BlockInput& input, const DecisionOptions& options,
                        BlockDecision& decision, std::string& error);

// Fills the syntax-side description of a non-skip decision and appends the
// block's scan-order levels (Y, U, V) to `levels_out` (which is *appended*).
// `decision.skip` must be false.
bool make_syntax_block(const BlockGeometry& geometry,
                       const BlockDecision& decision,
                       InterResidualBlock* block,
                       std::vector<std::int32_t>* levels_out,
                       std::string& error);

// Appends one plane's adjusted-raster levels to `out` in diagonal scan order,
// truncated at the EOB (the syntax writer's contract).
void append_scan_levels(const PlaneResidual& plane,
                        std::vector<std::int32_t>* out);

// ---------------------------------------------------------------------------
// GPU-ready batch interface
// ---------------------------------------------------------------------------
//
// POD descriptors with element offsets, so the same batch can be handed to a
// device kernel: residual gather -> forward/quant (src/cuda/tx_variants) ->
// EOB/context derivation (inter_residual_device.cuh) -> inverse -> recon.
// The rate term is a closed-form model (no bitstream state), so the batch is
// order-independent and can run one block per thread.
struct BatchBlock {
  int32_t mi_row = 0;
  int32_t mi_col = 0;
  int32_t luma_w = 0;
  int32_t luma_h = 0;
  int32_t chroma_w = 0;
  int32_t chroma_h = 0;
  // Element offsets into BatchInput::source / ::prediction (per plane).
  int32_t src_offset[kResidualPlaneCount] = {0, 0, 0};
  int32_t pred_offset[kResidualPlaneCount] = {0, 0, 0};
  // Row strides of the source/prediction planes; 0 means "tightly packed at
  // the plane block width".
  int32_t src_stride[kResidualPlaneCount] = {0, 0, 0};
  int32_t pred_stride[kResidualPlaneCount] = {0, 0, 0};
  // Element offsets into BatchOutput::levels (adjusted raster, `adj_w*adj_h`)
  // and ::recon (coded size).
  int32_t levels_offset[kResidualPlaneCount] = {0, 0, 0};
  int32_t recon_offset[kResidualPlaneCount] = {0, 0, 0};
  // BatchBlockFlags bits.
  int32_t flags = 0;
};

enum BatchBlockFlags : int32_t {
  kBatchAllowResidual = 1 << 0,
  kBatchTryTxTypes = 1 << 1,
  kBatchRdoq = 1 << 2,
};

struct BatchInput {
  std::span<const std::uint8_t> source[kResidualPlaneCount];
  std::span<const std::uint8_t> prediction[kResidualPlaneCount];
  std::span<const BatchBlock> blocks;
  DecisionOptions options;
};

struct BatchOutput {
  std::span<std::int32_t> levels;
  std::span<std::uint8_t> recon;
  std::span<std::int32_t> eob;      // one per block per plane
  std::span<std::int32_t> tx_size;  // one per block per plane
  std::span<std::int32_t> tx_type;  // one per block per plane
  std::span<std::int32_t> skip;     // one per block
  std::span<double> distortion;     // one per block
  std::span<double> cost;           // one per block
  std::span<std::int32_t> status;   // one per block: 0 = ok, -1 = failed
};

// Runs the batch on the CPU reference path.  Output spans must be large enough
// for every block's offsets; `status` receives one entry per block and the
// function returns false if any block failed (per-block status is still set).
bool decide_inter_batch(const BatchInput& input, BatchOutput& output,
                        std::string* error);

// Rate estimate in bits for one plane's syntax (TX type symbol, EOB, base/br
// codes, signs, escapes).  This is the documented bring-up model: it is a
// probability-based estimate from the §9.3 CDFs, not the §8.2 coder.
double estimate_plane_rate_bits(const PlaneResidual& plane,
                                const DecisionOptions& options);

}  // namespace av2enc::video::residual
