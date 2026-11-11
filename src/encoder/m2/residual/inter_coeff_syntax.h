// Inter-residual coefficient syntax writer (§5.20.7.23/.24/.27, §5.20.8.2).
//
// The M2 core writer (`src/encoder/m2/m2_syntax.cpp`) owns the intra path and
// rejects `skip_flag = 0` on inter blocks.  This module is the drop-in
// replacement for that branch: given the per-plane transform sizes/types/EOBs
// and the scan-order levels a decision path produced, it emits exactly the
// symbols a decoder reads for a non-skip inter block.
//
// Inter-specific choices, all taken from the spec and *different* from the
// intra path the M2 writer implements:
//
//   * `all_zero` (TXB skip) uses `TileTxbSkipCdf[is_inter][txSzCtx][ctx]`
//     (§8.3.2 `all_zero`), i.e. CDF plane 1 instead of 0.
//   * The luma EOB CDF context is `eobCtx = is_inter` (§5.20.7.27), i.e. 1
//     instead of 0.  Chroma keeps context 2.
//   * `transform_type()` codes `inter_tx_type` (+ `inter_tx_type_offset` for
//     TX_SET_INTER_1/2) instead of `intra_tx_type`, and there is *no*
//     `eob == 1 => DCT_DCT` override for inter (§5.20.8.2).
//   * `txb_skip` / `eob_pt` are the only coefficient-CDF selections that change;
//     `coeff_base`, `coeff_base_eob`, `coeff_br`, `dc_sign` and `read_quant`
//     keep the M1/M2 derivations (they do not depend on is_inter).  DC sign is
//     coded with the `dc_sign` CDF for the luma DC only; every other DC and all
//     AC signs are raw `sign_bit` literals (§5.20.7.27).
//
// M2 signals `enable_parity_hiding = 0`, `allow_tcq = 0`, `enable_cctx = 0`,
// `enable_fsc = 0` and `enable_inter_ddt = 0`, so no parity hiding, TCQ state,
// CC TX, FSC or 1D-transform (`txClass != TX_CLASS_2D`) coefficient contexts
// are emitted.  The writer still derives every context from the coded stream,
// exactly like the decoder: the caller supplies no CDF or neighbour state.
//
// TX sizes: M2 writes `tx_mode_select = 0` (TX_MODE_LARGEST), so
// `read_block_tx_size()` codes no bits for a non-skip inter block and every
// plane uses `Max_Tx_Size_Rect` (luma = block size, chroma = chroma residual
// block size).  `inter_tx_sizes()` in `inter_residual.h` derives that pair.
//
// CDF state: this module keeps its own copy of the §9.3 coefficient CDFs
// because the M2 writer's CdfState is private.  M2 sets `disable_cdf_update
// = 1` in every frame header, so both copies stay at the defaults and the
// duplication is exact.  Before enabling adaptive CDFs the two states must be
// merged (see docs/INTEGRATION.md).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "encoder/m2/residual/inter_coeff_plan_types.h"
#include "encoder/m2/residual/m2_residual_cdf.h"

namespace av2enc::m2::residual {

inline constexpr int kResidualPlaneCount = 3;

// Symbol sink supplied by the caller.  The M2 writer binds these to
// `av2::Encoder::encode_symbol` / `encode_bool`; tests bind them to an
// in-memory recorder.  The sink must not reorder calls.
struct SymbolSink {
  void* context = nullptr;
  bool (*encode_symbol)(void* context, std::uint16_t* cdf, int n,
                        int symbol) = nullptr;
  bool (*encode_bool)(void* context, bool bit) = nullptr;
};

// §9.3 CDF state for the inter residual syntax (qctx slice 0, base_q_idx 90).
struct InterResidualCdfs {
  // Inter transform types (§9.3 Default_Inter_Tx_Type_*).
  std::uint16_t inter_tx_type_set1[3][2][3];
  std::uint16_t inter_tx_type_index_set1[3][9];
  std::uint16_t inter_tx_type_offset_set1[3][9];
  std::uint16_t inter_tx_type_set2[3][3];
  std::uint16_t inter_tx_type_index_set2[3][9];
  std::uint16_t inter_tx_type_offset_set2[3][5];
  std::uint16_t inter_tx_type_set3[3][4][3];
  std::uint16_t inter_tx_type_set4[3][4][5];
  std::uint16_t inter_tx_type_long[3][4][5];
  std::uint16_t is_long_side_dct[2][3];

  // Coefficient CDFs (§9.3).  txb_skip keeps both is_inter planes.
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  std::uint16_t eob_pt_16[3][6];
  std::uint16_t eob_pt_32[3][7];
  std::uint16_t eob_pt_64[3][8];
  std::uint16_t eob_pt_128[3][9];
  std::uint16_t eob_pt_256[3][9];
  std::uint16_t eob_pt_512[3][9];
  std::uint16_t eob_pt_1024[3][9];
  std::uint16_t eob_extra[3];
  std::uint16_t coeff_base[5][20][2][5];
  std::uint16_t coeff_base_lf[5][33][2][7];
  std::uint16_t coeff_base_eob[5][4][4];
  std::uint16_t coeff_base_lf_eob[5][4][6];
  std::uint16_t coeff_br[7][5];
  std::uint16_t coeff_br_lf[14][5];
  std::uint16_t coeff_base_uv[12][5];
  std::uint16_t coeff_base_lf_uv[12][7];
  std::uint16_t coeff_base_eob_uv[4][4];
  std::uint16_t coeff_base_lf_eob_uv[4][6];
  std::uint16_t coeff_br_uv[4][5];
  std::uint16_t dc_sign[2][2][3][3];
};

void init_inter_residual_cdfs(InterResidualCdfs& cdfs);

// AboveLevel/LeftLevel/AboveDc/LeftDc grids (§8.3.2), shared with the M2
// writer's intra path.  Arrays are indexed by 4x4 position and must have at
// least `mi_cols + kCoeffContextPad` / `mi_rows + kCoeffContextPad` entries.
inline constexpr int kCoeffContextPad = 16;

struct CoeffContextView {
  int mi_cols = 0;  // luma MiCols
  int mi_rows = 0;  // luma MiRows
  std::uint8_t* above_level[kResidualPlaneCount] = {};
  std::uint8_t* left_level[kResidualPlaneCount] = {};
  std::uint8_t* above_dc[kResidualPlaneCount] = {};
  std::uint8_t* left_dc[kResidualPlaneCount] = {};
  // §5.20.7.27 `EobU`: frame-level, set by the U plane and read by V.  Shared
  // with the M2 writer's intra path through this pointer.
  int* eob_u = nullptr;
};

// One plane's residual for one block.
struct InterPlaneResidual {
  std::uint8_t tx_size = 0;  // txv::TxSize / AV2 TX_SIZE 0..24
  std::uint8_t tx_type = 0;  // txv::TxType; luma only, chroma is DCT_DCT
  std::uint32_t eob = 0;
  // First entry of this plane's levels in the caller's scan-order array.
  std::uint32_t levels_offset = 0;
};

struct InterResidualBlock {
  int mi_row = 0;
  int mi_col = 0;
  int luma_w = 0;  // coded luma block size
  int luma_h = 0;
  int chroma_w = 0;  // coded chroma residual block size
  int chroma_h = 0;
  InterPlaneResidual plane[kResidualPlaneCount];
};

// Emits the residual syntax for one non-skip inter block: per plane the
// TXB-skip flag, EOB, (luma) inter transform type, base/br magnitudes, signs
// and escape codes, and updates `contexts` afterwards.  `levels` holds the
// block's scan-order levels (plane after plane, exactly as indexed by
// `InterPlaneResidual::levels_offset`).
//
// Returns false with `error` set on any inconsistency (invalid TX size, EOB
// outside the transform area, level array overflow, sink failure).
bool write_inter_residual(const SymbolSink& sink, CoeffContextView& contexts,
                          InterResidualCdfs& cdfs,
                          const InterResidualBlock& block,
                          std::span<const std::int32_t> levels,
                          std::string& error);

// Emits only the transform-type syntax for one luma TX block (§5.20.8.2).
// Exposed separately so a caller that codes coefficients elsewhere can still
// reuse the set/context/symbol logic.
bool write_inter_tx_type(const SymbolSink& sink, InterResidualCdfs& cdfs,
                         int tx_size, int tx_type, int eob,
                         std::string& error);

// ---------------------------------------------------------------------------
// Coefficient plan (the integer work a device coefficient writer repeats)
// ---------------------------------------------------------------------------
//
// CoeffPlanEntry (the per-position coding plan) lives in
// inter_coeff_plan_types.h so the device coefficient writer can share the POD
// layout.

// Builds the per-scan-position plan for one plane.  `levels_scan[c]` is the
// signed quantised level at scan position c (c < eob).  `dc_sign_ctx` is the
// pre-derived luma DC sign context (see dc_sign_context()).
bool build_coeff_plan(int plane, int eob, int adj_w, int adj_h,
                      std::span<const std::int32_t> levels_scan,
                      int dc_sign_ctx, std::vector<CoeffPlanEntry>* plan,
                      std::string* error);

// §8.3.2 dc_sign context from the AboveDc/LeftDc grids.
int dc_sign_context(const CoeffContextView& contexts, int plane, int x4, int y4,
                    int w4, int h4);

}  // namespace av2enc::m2::residual
