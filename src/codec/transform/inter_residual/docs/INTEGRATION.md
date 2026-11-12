# Integration notes: M2 inter residual

Everything below is either (a) an edit to the M2 core, which this component may
not make, or (b) the GPU launch plan.  The gate proves the edits work by
applying them to a build-dir copy (see `gate/patch_video_syntax.py` and
`docs/video_syntax_inter.patch`).

## 1. Encoder decision call site

Owners: M2 frame encoder (`src/encoder/m2/video_frame.cpp`, `encode_leaf`).

Today the M2 core has its own DCT-only inter-residual decision.  To use this
component instead, replace the residual block of `encode_leaf` with:

```cpp
#include "encoder/m2/residual/inter_residual.h"

// after the winning inter candidate's prediction has been produced
residual::BlockGeometry bg{node.mi_row, node.mi_col, w, h, cw, ch};
residual::BlockInput input;
input.geometry = bg;
input.source[0] = {/* source block, w*h */};
input.source[1] = {/* ... cw*ch */};
input.source[2] = {/* ... cw*ch */};
input.prediction[0] = {/* inter prediction */};
input.prediction[1] = {/* ... */};
input.prediction[2] = {/* ... */};

residual::DecisionOptions options;
options.base_q_idx = options_.base_q_idx;
options.try_inter_tx_types = true;   // false reproduces DCT-only output
options.rdoq = options_.rdoq;

residual::BlockDecision decision;
if (!residual::decide_inter_block(input, options, decision, error)) return false;

entry.skip_flag = decision.skip ? 1 : 0;
if (!decision.skip) {
  residual::InterResidualBlock syntax;
  if (!residual::make_syntax_block(bg, decision, &syntax, &levels_, error)) {
    return false;
  }
  for (int p = 0; p < 3; ++p) {
    entry.tx_size[p] = syntax.plane[p].tx_size;
    entry.eob[p] = syntax.plane[p].eob;
  }
  entry.levels_offset = syntax.plane[0].levels_offset;
  entry.tx_type[0] = syntax.plane[0].tx_type;   // see §2
}
// write decision.plane[p].recon into the reference frame in both cases
```

`decision.skip == true` also fills `decision.plane[p].recon` (it equals the
prediction), so the reconstruction update does not need a branch.

`BlockSyntax` needs one new field for the luma TX type:

```cpp
// src/encoder/m2/video_syntax.h
std::uint8_t tx_type[kPlaneCount] = {0, 0, 0};  // txv::TxType; luma is coded
```

(It has to be somewhere; the gate cannot patch the public header, so it passes
the per-leaf types to the writer out of band — see `video_residual_gate_set_inter_tx_types`
in the patch.)

## 2. Syntax writer call sites

Owner: `src/encoder/m2/video_syntax.cpp`.  Three edits, all in that file; the
generated diff is `docs/video_syntax_inter.patch`.

1. **State**: add the inter CDF state next to `CdfState cdfs_{};`

   ```cpp
   av2enc::video::residual::InterResidualCdfs residual_cdfs_{};
   ```

   and initialise it in the constructor next to `init_cdfs(cdfs_);`

   ```cpp
   av2enc::video::residual::init_inter_residual_cdfs(residual_cdfs_);
   ```

2. **Adapter**: `code_tx()` already selects the inter `txb_skip` plane and the
   inter luma EOB context; keep that.  Replace the body of the existing
   DCT-only `write_inter_tx_type(int tx_size, int eob, int coded_w, int coded_h)`
   with a call to this component, passing the leaf's signalled type:

   ```cpp
   bool write_inter_tx_type(int tx_size, int eob) {
     av2enc::video::residual::SymbolSink sink;
     sink.context = &enc_;
     sink.encode_symbol = [](void* ctx, std::uint16_t* cdf, int n, int s) {
       return static_cast<av2::Encoder*>(ctx)->encode_symbol(cdf, n, s) ==
              av2::Status::kOk;
     };
     sink.encode_bool = [](void* ctx, bool bit) {
       return static_cast<av2::Encoder*>(ctx)->encode_bool(bit) ==
              av2::Status::kOk;
     };
     return av2enc::video::residual::write_inter_tx_type(
         sink, residual_cdfs_, tx_size, node.tx_type[0], eob, error_);
   }
   ```

   (`node` is in scope in `write_block`; threading it into `code_tx` is the
   cleanest change.  The patch instead reads a coding-order type list because it
   cannot change the header.)

   `write_inter_tx_type` validates the type against `get_tx_set()` +
   `Tx_Type_In_Set_Inter` + `enable_inter_ddt`, so a bad decision fails the
   encode instead of producing an undecodable stream.

3. **Full-writer alternative**: if you prefer to own the coefficient loop here
   too, `residual::write_inter_residual(sink, contexts, cdfs, block, levels,
   error)` emits the whole non-skip inter block (TXB skip, EOB, TX type,
   base/BR, signs, escapes) and updates the caller's above/left level/DC grids.
   The writer's grids must then be exposed as a `residual::CoeffContextView`
   (`above_level[3]`, `left_level[3]`, `above_dc[3]`, `left_dc[3]`,
   `mi_cols`, `mi_rows`, `eob_u`).  The component's `write_inter_residual` is
   byte-identical to the M2 writer for DCT_DCT (gate-checked), so either route
   keeps the stream unchanged for DCT-only decisions.

## 3. Adaptive CDFs

`InterResidualCdfs` is a second §9.3 state next to the writer's `CdfState`.
M2 frames set `disable_cdf_update = 1`, so both stay at the defaults and the
duplication is exact.  If adaptive CDFs are enabled, merge the two: point
`InterResidualCdfs`' coefficient arrays at the writer's arrays (or move the
coefficient CDFs into one struct shared by both paths) before enabling
`disable_cdf_update = 0`.

## 4. GPU launch plan

Per P frame, once the inter candidate per leaf is fixed (prediction is frozen):

1. **Upload** `BatchInput`: source and prediction planes (the prediction is
   produced by the existing `src/cuda/inter/mc` kernels), plus the
   `BatchBlock[]` descriptor array (POD, offsets only).
2. **Residual gather** — `ir_residual_coded()` from
   `inter_residual_device.cuh` (one thread per sample or one block per
   warp), then `txv_downsample_residual` for adjusted sizes.
3. **Forward + quantise** — `txv_forward_transform_multi` /
   `txv_quantize` (`src/cuda/tx_variants/`), one `TxBlockJob` per plane.
   Optional RDOQ trellis runs on the host reference only.
4. **EOB + context plan** — `ir_eob_scan()` and `ir_build_coeff_plan()` per
   plane, per block.  The context plan is pure integer work and matches the
   CPU reference bit-for-bit (gated).
5. **Inverse + reconstruct** — `txv_inverse_transform` then `ir_reconstruct()`
   (clip + SSE).
6. **Decision** — `ir_cost()` / `ir_prefer_residual()` on the host or in a
   small kernel; the rate term is the closed-form estimate, so the batch is
   order-independent.
7. **Download** `BatchOutput` (levels, recon, eob, tx_size, tx_type, skip,
   costs).  The entropy writer stays on the host or in the existing single
   entropy pass: it is strictly sequential because of the above/left contexts.

The transforms are not duplicated in this component; `src/cuda/tx_variants/` is
the device implementation and already has its own CPU/GPU parity gate.  What is
validated here is the component's own integer work: residual gather, EOB, the
whole §8.3.2 coefficient plan, the DC-sign context and the skip rule
(`inter_residual_emulation_test.cpp`, 469 checks).
