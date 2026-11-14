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

### TX lambda calibration on the nightly edge path

The first nightly measurement (`quality + edge + tx-context`, lambda 256)
lost ground because of the weight, not the model: the context rate charges
real bits, ~2.5–3× the historical heuristic, while the outer decisions and
the preset still use `--video-tx-lambda 256`. Sweeping the weight at the
100 kbps point (480x240x60, `--nightly --video-tx-context-rate
--video-tx-lambda L`, CVVDP measured at every point) moves the optimum to
40–56:

| `--video-tx-lambda` | QP | bytes | kbps | CVVDP | PSNR-Y | iso-rate JOD at 100.248 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 137 | 30,842 | 123.368 | — | — | (off the 100 kbps hull) |
| 32 | 140 | 25,706 | 102.824 | 8.7695 | 35.5127 | 8.7435 |
| 40 | 139 | 25,342 | 101.368 | 8.7906 | 35.5273 | 8.7795 |
| 44 | 139 | 25,190 | 100.760 | 8.7710 | 35.4576 | 8.7679 |
| **48** | **138** | **24,940** | **99.760** | **8.7884** | **35.5306** | **8.7901** |
| 52 | 138 | 25,071 | 100.284 | 8.7799 | 35.4273 | 8.7799 |
| 56 | 137 | 25,634 | 102.536 | 8.7795 | 35.4908 | 8.7732 |
| 64 | 137 | 25,529 | 102.116 | 8.7754 | 35.4603 | 8.7679 |
| 96 | 137 | 25,236 | 100.944 | 8.7709 | 35.4455 | 8.7685 |
| 128 | 137 | 25,168 | 100.672 | 8.7663 | 35.3680 | 8.7635 |
| 256 | 139 | 25,009 | 100.036 | 8.5696 | 33.8140 | 8.5696 |

Baseline `--nightly` without the context model at 100.248 kbps is 8.7465 JOD /
35.3754 dB. The plateau (40–56) is +0.02 to +0.05 JOD and the interpolated
values wiggle by ±0.01 with the QP-to-QP mode changes, so 48 is picked as the
plateau centre, not as a uniquely better point: at QP 138 it is a direct
same-rate win (99.760 kbps vs the baseline's 99.760 kbps interpolation,
+0.062 JOD / +0.21 dB), and lambda 52 independently wins at 100.284 kbps
(+0.034 JOD).

Control: `--nightly --video-tx-lambda 48` **without** the context model is
disastrous (QP 146: 103.160 kbps, 8.6581 JOD; iso-rate ≈8.637), because the
heuristic under-charges every leaf and the low weight removes the rate
penalty. The pairing of the true-bit model with a matching weight is what
wins.

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

After the `--video-tx-lambda 48` calibration (this tree):

* `--nightly` expands to `--video-rd-preset quality --video-ccso-rd
  --video-edge-inter --video-tx-context-rate --video-tx-lambda 48` and its
  480x240 QP 138 stream is pinned at 24,940 B /
  `c4797f3bef07da3ed608caf5340875c5`;
* an explicit `--video-tx-lambda` overrides the nightly default and reproduces
  the sweep rows exactly (e.g. lambda 64 QP 138 → 24,662 B, lambda 256 QP 139 →
  25,009 B);
* `tools/gate_video_nightly.sh`: expansion identity, default/quality/nightly
  pins and the gate's 4-frame strict-decode/recon checks all pass; the
  60-frame QP 138 stream separately decodes under `dav2d --strict 1` and its
  `--video-recon` payload equals the decoded payload byte for byte
  (PSNR-Y 35.5306);
* `tools/gate_video_rd_preset.sh`, `tools/gate_video_edge_inter.sh`,
  `gate_video_deltaq.sh` (53/53) and `tools/test.sh` pass; the
  `verify_baseline.sh --check` ctest/feature-gate/19-stream set is unchanged.

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
  scale was left at true bits. The rescaled weight is what composes with the
  nightly edge path; see the lambda calibration above.
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
  `--video-tx-context-rate --video-gpu` is rejected by name. The model stays
  an explicit CPU opt-in; `--nightly` turns it on with
  `--video-tx-lambda 48` (`docs/video-nightly.md`). With the historical
  weight (256) it still measures worse than edge + CCSO, so it is calibrated
  only inside the nightly aggregate.
