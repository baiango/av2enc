# Unified M1 RDO loop (`src/encoder/m1/rdo_loop/`)

Status: **component + CPU experiment.** It is not wired into `CMakeLists.txt`,
`src/main.cpp`, `src/encoder/m1/m1_still.cpp`, the entropy coder, or the
bitstream writer. It owns no encoder state, writes no bits and changes no
reference stream. The integration pass owns enabling it by default.

This directory adds the one rate-aware decision loop the M1 reference still
lacks. Instead of four independent SSE-only or proxy-based decisions, one
Lagrangian comparison covers:

| Decision | What the loop compares |
|---|---|
| partition | `NONE` / `HORZ` / `VERT` / `SPLIT` where §9.2 and the encoder policy allow it |
| luma intra | mode + angle, with SSE pruning before the transform/rate probe |
| chroma intra | DC / CfL / full UV on `u_sse + v_sse` + rate |
| TX | TX size (through the partition block size), transform type and RDOQ levels, measured through the real forward/quant/inverse path |
| lambda | one pixel-domain lambda, optionally scaled per CTU by source activity |

The decision *policy* lives here. The *measurement* is supplied by the encoder
adapter: it trial-encodes a node with the real M1 prediction, transform and
RDOQ path and returns a `CostTerms` breakdown.

TX size is not a separate tree in M1: `compute_tx()` uses the coded block size
(capped to 32) as the transform size, so the partition node decides it. The loop
selects TX *type* and RDOQ levels for that size; a future sub-transform size
search can be added without changing the policy API. That split is deliberate: the
same policy can run on the CPU reference and on a CUDA kernel that fills the
same arrays.

## Files

| File | Purpose |
|---|---|
| `config.h/.cpp` | one lambda convention, per-CTU activity scale, calibrated QP-90 profile |
| `cost.h/.cpp` | cost breakdown, exact ordering, fixed-capacity per-node trace |
| `search.h/.cpp` | node legality, candidate order, pruning, exact branch-and-bound, node selector |
| `batch.h/.cpp` | GPU-ready candidate batch and deterministic reduction |
| `activity.h/.cpp` | integer-only CTU activity for the lambda hint |
| `rdo_loop.h` | umbrella include |
| `rdo_loop_test.cpp` | standalone self-tests |
| `build_and_test.sh` | host-only build + test |
| `experiment/` | patch-neutral M1 harness, results and reproduction |

It reuses rather than reimplements:

- `src/encoder/m1/rdo_decide/` for lambda arithmetic, the transform-stream rate
  model, and the luma/chroma/TX/RDOQ selectors;
- `src/encoder/rdo/` for `lambda_sse_from_quant_step()`;
- `src/encoder/partition/` for geometry, `Partition_Subsize`, implied/forced
  partitions and the base partition set;
- `src/encoder/partition_ext/` for `enumerate_children()`; the base partitions
  delegate to `partition::` inside that module, so the loop and the entropy
  writer share one child-geometry authority.

## Cost model and exact lambda calibration

Every decision uses:

```text
cost   = SSE + lambda * rate_bits
lambda = sse_scale * quant_step^2          (8-bit SSE units per bit)
```

At M1's pinned `base_q_idx = 90` the *reconstruction* step is 66. `528` is the
forward quantiser-table entry and must not be used here; feeding it in inflates
lambda by 64x. The derivation is in `rdo_decide/config.h`.

**Calibrated joint-loop scale: `sse_scale = 0.0005`, i.e. lambda = 2.178.**
This is not the `rdo_decide` mode-layer constant (0.5). The joint loop compares
the *real coded coefficient bits of a whole partition node* (hundreds to
thousands of bits), while the older per-decision partition search used a
variance/SSE bit proxy. One lambda now governs splits, modes, TX type and RDOQ,
so it is calibrated on the joint node cost. The sweep is in
`experiment/README.md`. `0.0005` is the best common point: it maximizes CVVDP on
64x64 and 1080p, and 640x360's peak at `0.0002` is only +0.0014 JOD for +144
bytes. Larger values collapse 640x360/1080p toward low-rate blocks at the fixed
QP.

Per-CTU adaptation is implemented and defaulted **off**:

```text
scale  = clamp(1 + strength * (ctu_activity / frame_mean_activity - 1), 0.75, 1.5)
lambda = base_lambda * scale
```

`strength = 0.25/0.5/1.0` was measured on the two small fixtures and was
neutral-to-slightly-negative (see the experiment README). The default is
therefore `0`, but the hook and the exact integer activity measure are shipped
so a target-corpus calibration can enable it without changing the policy.

RDOQ uses the **same** lambda, converted to the forward-coefficient domain:

```text
lambda_pixel       = 2.178
lambda_coefficient = rdoq_lambda_from_pixel_lambda(2.178) = 2.178 / 64
                   = 0.03403125
```

The older experiment used a separate fixed `0.125` coefficient-domain scale;
that is not used by the unified loop. TX type and RDOQ both receive the CTU
lambda, so a future per-CTU calibration moves every decision together.

## Decision order, pruning and branch-and-bound

- Candidate order is `NONE, HORZ, VERT, SPLIT` (the task order, not the numeric
  `PartitionType` order). Ties prefer NONE, then HORZ, then VERT, then SPLIT.
- The total order is: cost, distortion SSE, total bits, fewer leaves, partition
  order, insertion index. Comparisons are exact; there is no epsilon.
- Spec-forced partitions are honoured; policy depth/min-block limits are
  relaxed only at a boundary, mirroring `partition::decide_superblock()`.
- Branch-and-bound is exact: when the accumulated child cost already exceeds
  the best complete candidate, the remaining children cannot have negative
  distortion or rate, so the candidate is abandoned. Equality is not pruned.
- Candidate pruning happens before the transform/rate probe: top-K by SSE plus
  a relative SSE margin, always keeping DC. Defaults are 6 luma, 6 chroma,
  margin 1.35. On the fixtures, 6/6 and 61/13 produce the same CVVDP to four
  decimals, so pruning removes probes without changing the decision.
- `ProbeBudget` is an optional per-superblock cap. Probe order is fixed, so a
  capped run is deterministic. The default is unlimited.

## CPU reference decisions and per-node breakdown

`NodeDecisionLoop` is the reference selector. It stores at most four complete
candidates (one per partition type) and exposes:

```cpp
NodeDecisionLoop loop;
loop.reset(info, options);
loop.set_lambda(lambda);
loop.add_terms(PARTITION_NONE, leaf_terms, leaves);
loop.add_partition(PARTITION_HORZ, child_costs);
const NodeDecision& decision = loop.decision();
```

Each candidate carries the full split:

```text
distortion_sse, mode_bits, coefficient_bits, tx_type_bits, signal_bits
```

`CostTrace` records one row per **committed** node (the experiment adapter walks
the final tree after encoding, so trial nodes are not mixed in). The CSV has
`mi_row,mi_col,size,depth,partition,forced,lambda,sse,mode_bits,coeff_bits,
tx_bits,signal_bits,cost,leaves`. The trace is fixed-capacity (32k entries,
~2.6 MiB) and never allocates. That capacity is deliberately bounded because
the CPU reference `Encoder` is stack-allocated.

The loop's own decision paths (`NodeDecisionLoop`, `prune_candidates`,
`CandidateBatch`, `CostTrace`) allocate nothing. The experiment adapter's
snapshot buffers and the existing M1 encode vectors are encoder/experiment
scratch, not part of the policy layer.

## GPU-ready candidate batch

`batch.h` defines one POD decision group:

```cpp
CandidateBatch batch;          // 96 records, no pointers
append_candidate(batch, record);
finalize_batch(batch);
// batch.records[i].cost, batch.best_index
```

A CUDA kernel fills `records[i].distortion_sse` and the four rate components;
the same `batch_record_better()` comparator can run as a device reduction. The
layout is `static_assert`-trivially-copyable. The comparator is the same total
order as the CPU selector: cost, distortion, bits, caller index, array slot.
Invalid records are rejected and counted, never selected.

This is the API the GPU planner should use for luma modes, chroma modes, TX
types and the four partition candidates. The CPU reference does not require the
GPU to run the policy; it only requires the same fields and order.

## CPU experiment (QP 90, one frame, one tile)

Harness: `experiment/run_m1_rdo_loop.sh`. It builds and runs three binaries:
the unpatched current reference, the patched experiment copy, and the
reconstruction dump tool. It decodes every stream with `dav2d` via
`tools/quality.py`, verifies the default `loop_part` and `loop_full`
reconstructions per sample with `tools/m1_verify` (sweep points use the same
path), and checks that the patched binary with the loop off is byte-identical
to the reference for `--m1` and `--m1-full-intra`.

At the calibrated scale (`0.0005`):

| Fixture | Case | Bytes | PSNR-Y | PSNR-avg | CVVDP |
|---|---|---:|---:|---:|---:|
| 64x64 | heuristic | 936 | 42.5016 | 43.2809 | 9.9009 |
| 64x64 | heuristic + tools | 808 | 41.5656 | 42.3509 | 9.8959 |
| 64x64 | **loop full** | **918** | **42.6732** | **43.5671** | **9.9191** |
| 640x360 | heuristic | 13,355 | 45.0892 | 46.3683 | 9.8867 |
| 640x360 | heuristic + tools | 11,317 | 44.6354 | 45.8299 | 9.8543 |
| 640x360 | **loop full** | **13,693** | **48.1219** | **49.0604** | **9.9505** |
| 1080p | heuristic | 61,574 | 46.3573 | 47.7273 | 9.8976 |
| 1080p | heuristic + tools | 49,797 | 45.9875 | 47.3233 | 9.8599 |
| 1080p | **loop full** | **61,894** | **50.0339** | **51.0151** | **9.9253** |

"heuristic + tools" is the current heuristic with the same opt-in full intra,
TX type, RDOQ and full-UV flags. It is worse than the baseline, which isolates
the win to the loop rather than to enabling features.

Delta versus the current heuristic:

| Fixture | Bytes | PSNR-Y | CVVDP |
|---|---:|---:|---:|
| 64x64 | −18 (−1.9%) | +0.172 dB | **+0.0182 JOD** |
| 640x360 | +338 (+2.5%) | **+3.033 dB** | **+0.0638 JOD** |
| 1080p | +320 (+0.5%) | **+3.677 dB** | **+0.0277 JOD** |

At **matched bytes** the loop still wins, so this is a net RD gain, not just a
byte/quality trade:

- 64x64: `loop @0.0002` = 937 B, 42.8243 dB, 9.9166 JOD versus 936 B,
  42.5016 dB, 9.9009 JOD.
- 640x360: `loop @0.005` = 13,249 B, 47.6650 dB, 9.9387 JOD and
  `loop @0.002` = 13,436 B, 47.9385 dB, 9.9449 JOD; interpolating to 13,355 B
  is ≈47.82 dB / 9.942 JOD versus 45.0892 dB / 9.8867 JOD.
- 1080p: `loop @0.001` = 60,902 B, 49.9636 dB, 9.9235 JOD, i.e. fewer bytes
  than the 61,574 B baseline and +3.61 dB / +0.026 JOD.

The full sweep is in `experiment/README.md`.

## Integration call sites

The experiment patch is `experiment/patch_m1_rdo_loop.py`; it is a generated
copy, not production code. The production integration should make the same
calls in `src/encoder/m1/m1_still.cpp`:

1. **`Encoder::encode()`** after `poptions` is filled: replace the
   `partition::decide_frame()` + `collect_node()` pair with a unified search
   that uses `rdo_loop::describe_node()`, `policy_allows()`,
   `enumerate_node_children()` and `NodeDecisionLoop`, then commits the winning
   `partition::PartitionNode` through the existing `collect_node()` path. The
   syntax stream and decoder contract do not change.
2. **`Encoder::choose_luma_mode()`**: after predicting candidates, prune with
   `rdo_loop::prune_candidates()` and replace the SSE-only winner with
   `rdo_decide::decide_luma_intra()` using the CTU lambda.
3. **`Encoder::choose_uv_mode()`**: score `u_sse + v_sse`, prune, and call
   `rdo_decide::decide_chroma()`. Keep the CfL block computation unchanged.
4. **`Encoder::compute_tx()` TX type branch**: call
   `rdo_decide::decide_tx_type()` with the CTU lambda instead of the
   per-nonzero selector. Keep `enable_idtx_intra = 0`.
5. **`Encoder::compute_tx()` RDOQ branch**: call `rdo_decide::decide_rdoq()`
   with `rdoq_lambda_from_pixel_lambda(ctu_lambda)` so RDOQ shares the one
   lambda.
6. **`M1Options` / CLI**: add an explicit `rdo_loop` switch and a lambda-scale
   option. Default off until the integration owner accepts the policy change;
   the loop-off path is byte-identical to today.
7. **Build**: add `config.cpp`, `cost.cpp`, `search.cpp`, `batch.cpp` and
   `activity.cpp` to `av2enc_core` (build owner). `partition_ext.cpp` is already
   required for `enumerate_children()`.
8. **CUDA**: fill `CandidateBatch` in the existing
   `src/cuda/m1/search/` planner and run the same comparator. No CUDA code is
   added by this directory; the CPU path remains the reference.

Do not add `PARTITION_SPLIT` below 128x128: AV2 only defines it for 128/256
superblocks. The loop preserves that through `partition::init_allowed_partitions`.

After wiring, re-run `tools/validate_dav2d.sh`, `tools/m1_verify/run_matrix.sh`,
the experiment harness, and the pinned MD5s.

## Non-claims

- CPU-only, one-frame, correctness/experiment results. **No speed claim.**
  The trial encode is deliberately expensive; the product path is the GPU
  planner, not this CPU reference.
- The transform-stream rate model is a static §9.3 CDF estimate, not the
  byte-exact §8.2 entropy coder. It ranks candidates; the cost-breakdown bit
  totals are model bits, not stream bytes.
- The partition split signal costs (NONE 1.0, HORZ/VERT 2.0, SPLIT 1.5 bits)
  are deterministic estimates. Replace them through the same `CostTerms` path
  when exact §8.3.2 contexts are available.
- The experiment is three still fixtures, not a corpus or a BD-rate claim.
  Calibrate `sse_scale` on the target corpus before changing the default.
- The loop-off path is byte-identical; the loop-on path emits no new syntax and
  every shown stream decodes with `dav2d` and is verified per sample.

## Build and test

```sh
cd src/encoder/m1/rdo_loop
./build_and_test.sh
SANITIZE=1 ./build_and_test.sh
WERROR=1 ./build_and_test.sh
```

The experiment is described in `experiment/README.md` and runs from
`experiment/run_m1_rdo_loop.sh`.
