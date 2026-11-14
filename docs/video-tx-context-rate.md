# TX-search coefficient-context rate model

The measured RD preset `--video-rd-preset quality` (or its explicit
`--video-tx-context-rate` spelling) prices every luma TX candidate with the
real §8.3.2 coefficient syntax instead of `1.0 bit per nonzero coefficient`.
Candidate sets, candidate order, the `cost = sse + lambda * rate` form,
tie-breaks and the bitstream writer are unchanged; only the rate estimate that
ranks existing candidates moves.

This is off for every flag-less stream, for standalone `--video-tx-search`
(with or without `--video-gpu`), and for the inter-DDT-only type search
(`--video-ddt` without the preset), so the pinned defaults and the CUDA parity
contract stay byte-identical.

## Mechanism

`src/pipeline/video/video_tx_search.cpp` previously charged each leaf

```
eob == 0 ? 1.2 : type_bits + 1.0 * nonzero
```

which omits `txb_skip`, the EOB prefix/extra bits, `coeff_base`,
`coeff_base_eob`, `coeff_br`, the `read_quant` escape and the sign, and is
~2.5–3× low per coefficient. With `Options::use_context_rate` (set by
`--video-tx-context-rate`) `block_syntax_bits()`
builds
`rdoq::av2::Model` / `tcq::av2::Model` for the candidate's own quantised levels
in §5.20.7.30 scan order and takes `scalar_bits()` / `reference_bits()`:

* static §9.3 CDF slice from the §8.3.1 slice of `base_q_idx`
  (`tcq_qctx`, set from `base_q_index()` when the context model is active; this
  also fixes the old use of the per-segment `current_qindex_` under
  `--video-per-segment-q`/`--video-delta-q` for the preset.  Standalone
  `--video-tx-search` keeps its historical `current_qindex_` slice);
* `txb_skip[is_inter][tsc][ctx]` and EOB context `is_inter ? 1 : 0` for luma —
  `rdoq::av2::BlockInfo` gained `is_inter` (default false, so every still
  caller is unchanged);
* EOB symbol + extra bits, `coeff_base`/`coeff_base_eob`, `coeff_br`,
  `read_quant` escape and sign per coefficient, from the candidate's own
  levels (the neighbour contexts are therefore exact for the chosen block);
* TCQ candidates use the TCQ escape threshold (`maxLevel - 1`) via
  `tcq::av2::Model`.

Cross-*block* contexts are the only unavailable inputs: `txb_skip_context`
and `dc_sign_context` are fixed at 0 (the writer's "transform block equals
coding block" context). Non-2D TX classes keep the historical heuristic; the
product path does not select them.

The outer skip/residual and intra/inter decisions still use
`estimate_tx_bits()` (`video_frame.cpp:937`). Feeding the search's own rate
into those decisions is the documented failure in
`docs/negative-results.md` (line 239) and was not revisited.

## Measured RD (480x240, 30 fps, 60 frames, CPU_FAST, 8 threads)

`tools/quality.py` with CVVDP 0.5.7 (`standard_fhd`, MPS), pinned
`dav2d-macos`. All four QPs decode with `dav2d -q --strict 1` and the 60-frame
encoder recon dump equals the decoder output byte for byte.

| QP | old B | old kbps | old JOD | context B | context kbps | context JOD | context PSNR-Y |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 187 | — | — | — | 26,467 | 105.868 | 6.3929 | 27.6455 |
| 188 | — | — | — | 25,682 | 102.728 | 6.3502 | 27.5782 |
| 189 | 25,926 | 103.704 | 6.3657 | 24,706 | 98.824 | 6.3025 | 27.4077 |
| 190 | 25,141 | 100.564 | 6.3021 | 24,116 | 96.464 | 6.2391 | 27.3070 |
| 191 | 25,351 | 101.404 | 6.2466 | 23,852 | 95.408 | 6.1600 | 27.0664 |
| 192 | 24,606 | 98.424 | 6.1832 | 23,081 | 92.324 | 6.1097 | 26.9940 |

"old" is the stable `quality` preset, "context" is `quality` with
`--video-tx-context-rate` with it (both from the same 8-thread CPU build). Base preset PSNR-Y at QP 189/190/191/192: 27.6984 / 27.5816 / 27.4351 /
27.3075 dB. The preset QP 191 row is a rate-control outlier (higher rate than
QP 190 at lower quality), so only 189/190/192 are useful brackets.

Iso-rate (linear interpolation of the new curve at each preset rate):

| Base point | Rate | preset JOD | new JOD | Δ JOD |
|---|---:|---:|---:|---:|
| QP 189 | 103.704 | 6.3657 | 6.3635 | −0.002 |
| QP 190 | 100.564 | 6.3021 | 6.3238 | +0.022 |
| QP 192 | 98.424 | 6.1832 | 6.2918 | **+0.109** |
| operating 100.04¹ | 100.04 | 6.2730 | 6.3174 | +0.044 |

¹ interpolated between preset QP 190/192 and new QP 189/188.

Iso-CVVDP (linear interpolation of the new curve at each preset JOD):

| Base point | JOD | preset kbps | new kbps | rate Δ |
|---|---:|---:|---:|---:|
| QP 189 | 6.3657 | 103.704 | 103.868 | +0.16% |
| QP 190 | 6.3021 | 100.564 | 98.809 | **−1.75%** |
| QP 192 | 6.1832 | 98.424 | 95.718 | **−2.75%** |

At the QP 190 operating point (the preset's own nearest measured point) the
model saves 1.75% rate at the same CVVDP, which meets the ≥1% rate criterion.
The ≥+0.10 JOD-at-iso-rate target is met at the QP 192 rate (98.4 kbps,
+0.109 JOD) but not at 100–104 kbps (+0.02 to 0.00 JOD). This is a modest
gain, not a new operating point.

Encode time (60 frames, best of 3, interleaved): preset 4.13 s, new 4.55 s
(+10%). Full per-candidate pricing is cheap because the transform/inverse
already dominates; no top-K re-rank was needed.

### Cross-clip check (1440x720, 12 frames)

| Case | kbps | JOD | PSNR-Y |
|---|---:|---:|---:|
| preset QP 190 | 897.40 | 6.3726 | 30.1334 |
| new QP 188 | 847.24 | 6.4123 | 30.1532 |
| new QP 190 | 795.72 | 6.3507 | 30.0026 |

At iso-CVVDP 6.3726 the new curve needs ~814 kbps vs 897 kbps (−9.3%); at
iso-rate 897 kbps the new curve extrapolates to +0.10 JOD. The 720p preset's
QP 190/192 rates are also nearly equal, so this is a direction check, not a
second operating-point claim.

## Validation

* flag-less 480x240 QP 189 stream: 25,009 B,
  MD5 `a1d0dba1259b20a4b628d4cd1e14c33c` (unchanged);
* standalone `--video-tx-search` (no `--video-tx-context-rate`): byte-identical
  to the base binary on 1-frame 360p (`516d3aae…`) and 12-frame 480p
  (`9ec227c0…`) inter-residual+deblock streams;
* `--video-ddt` without `--video-tx-search`: byte-identical to the base binary;
* the four preset streams are byte-identical to the pre-gate measurements
  (`baf4fa24…` / `32b640eb…` / `71f2cd61…` / `8aec02ed…`), so the table and
  JODs reproduce exactly;
* `tools/gate_video_rd_preset.sh`: preset expansion (including the new flag),
  default identity, `dav2d --strict 1`, 4-frame recon == decoder output;
* `tools/test.sh --build build-b`: 10/10;
* `tools/refactor/verify_baseline.sh --check`: 19 streams, ctest 25/25,
  feature gate 30/30;
* `gate_video_deltaq.sh`: 53/53 with every pin unchanged (standalone
  `--video-tx-search` keeps the historical rate and qctx, so the pinned
  `b_psq_ir` row is byte-identical again);
* re-encoding a given QP twice gives the same IVF MD5 (deterministic).

## Follow-ups and rejected ideas

* **CUDA device port.** `src/cuda/pipeline/video/tx/video_tx.cu` still prices
  `type_bits + coefficient_bits*nonzero`, which is why the context model is
  gated to the CPU-only preset: standalone `--video-tx-search` on CPU and the
  CUDA backend still make identical decisions.  Porting the model to the device
  (or a device cost table) is what would let the standalone tool and the preset
  use it on both backends.
* **Constant rescale of the context rate** to restore the historical
  `lambda = 256` balance (tested with `--video-tx-lambda 102`, i.e. the same
  reparametrisation): iso-rate JOD at 100.04 kbps stayed +0.042 vs +0.044.
  Lambda is a reparametrisation of the same hull here, so the absolute model
  scale was left at true bits.
* **Top-K re-rank** (score with the heuristic, re-price the best two): not
  needed; full pricing costs ~10% wall time, and it keeps the comparison
  exact for every candidate in the existing order.
* **Feed the search rate into `luma_plan_bits()`**: rejected for the documented
  reason; that 202,530 → 359,846 B blow-up came from mixing the search rate
  with the outer skip/residual decision.
* **Reuse `tcq::Result::rate_bits`**: TCQ is not in the preset and costs ~60×
  encode time; the TCQ syntax branch is wired, but no TCQ measurement is
  claimed.
* **Turning the model on for every `--video-tx-search`** (the first version):
  it changed standalone CPU tx-search decisions while the CUDA device module
  still used the heuristic, silently breaking `gate_video_tx_search`.  The
  preset-only gate keeps that contract; the explicit
  `--video-tx-context-rate --video-gpu` is rejected by name. This model is an
explicit CPU opt-in; it is not in `--nightly` because it measured worse in
combination with the bottom-edge path (`docs/video-nightly.md`).
