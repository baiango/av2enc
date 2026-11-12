# Unified M1 RDO loop experiment

This is a **patch-neutral** CPU experiment for `src/encoder/m1/rdo_loop/`. It
never edits `src/encoder/m1/still_encoder.cpp`: `patch_still_rdo_loop.py` writes a
patched copy under `build-rdo-loop/`, and every hook is environment-gated. With
`AV2ENC_RDO_LOOP` unset, the patched binary must be byte-identical to the
unpatched CLI for `--still` and `--still-full-intra`. The runner checks that.

## Run

```sh
cd src/encoder/m1/rdo_loop/experiment
./run_still_rdo_loop.sh
./run_still_rdo_loop.sh --no-1080p
./run_still_rdo_loop.sh --scales 0.0002,0.0005,0.001,0.002,0.005
./run_still_rdo_loop.sh --input path.y4m --work /tmp/rdo-loop
```

Defaults:

- inputs: `testdata/basic_64x64_1f.y4m`, `tests/data/basic_640x360_1f.y4m`,
  `testdata/shine_1920x1080_10f.y4m` (first frame)
- scale: `0.0005`
- core library: `build/libav2enc_core.a`
- build/work: `build-rdo-loop/`

The script rebuilds the current unpatched M1 as `av2enc-ref`, the patched copy
as `av2enc-rdo-loop`, and `still_dump_rdo_loop` for reconstruction checks. If the
core library is stale relative to the current source tree, rebuild
`av2enc_core` first (`cmake --build build --target av2enc_core`).

## Cases

| Case | Command | Meaning |
|---|---|---|
| `heuristic` | `av2enc-ref --still` | current default: DC luma, variance partition |
| `heuristic + tools` | `--still --still-full-intra --still-tx-type --still-rdoq --still-uv-full` | same opt-in tools without the loop |
| `loop part+modes` | patched, `AV2ENC_RDO_LOOP=1`, `--still` | rate-aware partition + luma/chroma |
| `loop full` | patched, `AV2ENC_RDO_LOOP=1`, `--still --still-tx-type --still-rdoq --still-uv-full` | full unified loop at the calibrated lambda |
| `loop@S` | `AV2ENC_RDO_LOOP_SCALE=S` | sweep point |

Every stream is decoded/scored by `tools/quality.py` (dav2d + PSNR + CVVDP).
The default `loop_part` and `loop_full` streams are verified per sample with
`tools/still_verify/verify.py`; sweep points use the same encoder path and are
decoded/scored by the same gate.
Every loop stream also writes a committed-tree cost trace
(`loop_full.trace.csv`); the runner prints the summed leaf SSE, mode bits,
coefficient bits, TX-type bits and partition signal bits.

## Calibration result

**Joint-loop `sse_scale = 0.0005` → lambda = 2.178 at QP 90.** The sweep below
is the evidence. It is not the `rdo_decide` mode-layer 0.5: the joint loop
compares real coded coefficient bits for a whole partition node, whereas the
older partition search used a variance/SSE bit proxy. Shipping 0.5 in the joint
loop collapses 640x360 to 34 dB and 1080p toward low-rate blocks.

### Loop-full sweep, PSNR-Y / CVVDP

| Fixture | 0.0002 | 0.0005 | 0.001 | 0.002 | 0.005 |
|---|---:|---:|---:|---:|---:|
| 64x64 bytes | 937 | 918 | 913 | 911 | 902 |
| 64x64 PSNR-Y | 42.8243 | 42.6732 | 42.6792 | 42.6112 | 42.4738 |
| 64x64 CVVDP | 9.9166 | **9.9191** | 9.9125 | 9.9130 | 9.9056 |
| 640x360 bytes | 13,837 | 13,693 | 13,579 | 13,436 | 13,249 |
| 640x360 PSNR-Y | 48.1596 | 48.1219 | 48.0440 | 47.9385 | 47.6650 |
| 640x360 CVVDP | **9.9519** | 9.9505 | 9.9445 | 9.9449 | 9.9387 |
| 1080p bytes | — | 61,894 | 60,902 | — | — |
| 1080p PSNR-Y | — | 50.0339 | 49.9636 | — | — |
| 1080p CVVDP | — | **9.9253** | 9.9235 | — | — |

`0.0005` is the best common point: it maximizes CVVDP on 64x64 and 1080p.
`0.0002` is marginally better on 640x360 (+0.0014 JOD) but costs 144 more bytes
and loses on 64x64; `0.001` saves 712 bytes on 1080p at −0.0018 JOD. `0.005` is the right point if the target is
strictly lower rate: at 640x360 it is 106 bytes below the heuristic and still
+2.58 dB / +0.052 JOD.

## Results at 0.0005

Measured 2026-11-12 with `dav2d-macos` and CVVDP `standard_fhd` on the Apple
M4 host. Rebuild the harness after touching any M1/rdo_decide source; the tree
moves quickly.

| Fixture | Case | Bytes | PSNR-Y | PSNR-avg | CVVDP |
|---|---|---:|---:|---:|---:|
| 64x64 | heuristic | 936 | 42.5016 | 43.2809 | 9.9009 |
| 64x64 | heuristic + tools | 808 | 41.5656 | 42.3509 | 9.8959 |
| 64x64 | loop part+modes | 916 | 42.6198 | 43.5040 | 9.8998 |
| 64x64 | **loop full** | **918** | **42.6732** | **43.5671** | **9.9191** |
| 640x360 | heuristic | 13,355 | 45.0892 | 46.3683 | 9.8867 |
| 640x360 | heuristic + tools | 11,317 | 44.6354 | 45.8299 | 9.8543 |
| 640x360 | loop part+modes | 13,795 | 47.9701 | 48.9505 | 9.9472 |
| 640x360 | **loop full** | **13,693** | **48.1219** | **49.0604** | **9.9505** |
| 1080p | heuristic | 61,574 | 46.3573 | 47.7273 | 9.8976 |
| 1080p | heuristic + tools | 49,797 | 45.9875 | 47.3233 | 9.8599 |
| 1080p | loop part+modes | 62,037 | 49.9253 | 50.9148 | 9.9217 |
| 1080p | **loop full** | **61,894** | **50.0339** | **51.0151** | **9.9253** |

Delta versus the current heuristic:

| Fixture | Bytes | PSNR-Y | CVVDP |
|---|---:|---:|---:|
| 64x64 | −18 (−1.9%) | +0.172 dB | +0.0182 JOD |
| 640x360 | +338 (+2.5%) | +3.033 dB | +0.0638 JOD |
| 1080p | +320 (+0.5%) | +3.677 dB | +0.0277 JOD |

### Is it a net win?

Yes on these three fixtures, in the RD sense:

- **64x64** is a strict Pareto improvement: fewer bytes and higher PSNR/CVVDP.
- **640x360** spends 2.5% more bytes for +3.03 dB / +0.064 JOD. At matched
  bytes (13,355), interpolating the sweep gives ≈47.82 dB / 9.942 JOD, still
  +2.73 dB / +0.055 JOD over the heuristic.
- **1080p** spends 0.5% more bytes for +3.68 dB / +0.028 JOD. At matched
  bytes, `loop @0.001` is already *below* the baseline rate (60,902 B) at
  +3.61 dB / +0.026 JOD.

The "heuristic + tools" arm is worse than the baseline on all three. The win is
the joint loop, not merely enabling full intra, TX type, RDOQ or full UV.

## Ablations

### CTU-adaptive lambda

At `sse_scale = 0.0005`, `AV2ENC_RDO_LOOP_CTU` was measured at 0.25/0.5/1.0:

| Fixture | strength 0 (default) | 0.25 | 0.5 / 1.0 |
|---|---:|---:|---:|
| 64x64 bytes / CVVDP | 918 / 9.9191 | 918 / 9.9191 | 918 / 9.9191 |
| 640x360 bytes / CVVDP | 13,693 / 9.9505 | 13,667 / 9.9496 | 13,675 / 9.9491 |

Adaptation is neutral on 64x64 and slightly negative on 640x360. It stays
implemented and defaulted off.

### Candidate pruning

`AV2ENC_RDO_LOOP_MAX_LUMA=61 AV2ENC_RDO_LOOP_MAX_CHROMA=13` (no pruning) versus
the default 6/6 at 0.0005:

| Fixture | pruned 6/6 | no pruning |
|---|---|---|
| 64x64 | 918 B, 42.6732 dB, 9.9191 JOD | 918 B, 42.6732 dB, 9.9191 JOD |
| 640x360 | 13,693 B, 48.1219 dB, 9.9505 JOD | 13,696 B, 48.1232 dB, 9.9505 JOD |

Pruning is decision-neutral here to four decimals; it removes transform/RDOQ
probes, not cost accuracy.

### RDOQ lambda

The first experiment iteration used a separate fixed coefficient-domain RDOQ
scale (0.125). That dominated the joint lambda: at small `sse_scale` it kept
zeroing coefficients and made the result insensitive to the loop calibration.
The shipped loop converts the **same** pixel lambda
(`rdoq_lambda_from_pixel_lambda(ctu_lambda)`), which is why 0.0005 is a valid
single knob.

## Cost breakdown per node

`loop_full.trace.csv` contains one row per committed tree node (not trial
probes). Example aggregate at 0.0005:

| Fixture | Nodes | Leaves | SSE | Mode bits | Coeff bits | TX bits | Signal bits |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64x64 | 25 | 13 | 17,602 | 102.0 | 6,674.1 | 21.5 | 136.0 |
| 640x360 | 504 | 282 | 280,034 | 2,082.0 | 106,274.6 | 556.1 | 2,495.0 |
| 1080p | 3,356 | 1,933 | 1,608,180 | 13,466.0 | 476,656.3 | 3,249.6 | 15,991.0 |

The SSE column reconstructs PSNR-avg exactly (it is Y+U+V visible SSE). The bit
columns are the static §9.3 transform-stream model, not stream bytes; compare
them only against other loop candidates, not against the IVF size.

## Reproduce

```sh
# default: 64x64, 640x360, 1080p at the calibrated 0.0005
./run_still_rdo_loop.sh

# small-fixture sweep (fast)
./run_still_rdo_loop.sh --no-1080p --scales 0.0002,0.0005,0.001,0.002,0.005

# 1080p second point
RDO_LOOP_SCALES=0.0005,0.001 ./run_still_rdo_loop.sh \
  --input ../../../testdata/shine_1920x1080_10f.y4m
```

The default run also writes the committed-tree trace and the JSON reports under
`build-rdo-loop/results/<fixture>/`.

## Determinism and verification

- `loop part+modes` and `loop full` are verified **bit-exactly** on all three
  fixtures with `tools/still_verify/verify.py`; every sweep stream is decoded by
  `dav2d` and scored by the same `quality.py` gate.
- The loop-off path is byte-identical to the current reference for `--still` and
  `--still-full-intra`.
- Candidate order, pruning, branch-and-bound and the final selection are
  deterministic under exact floating-point comparison. Build both CPU and GPU
  with `-ffp-contract=off`.

## Why this is not a final BD-rate claim

- Three still fixtures, one frame each, one tile, QP 90 only.
- The rate model is a static §9.3 CDF estimate, not byte-exact §8.2 entropy.
- Partition signal costs are deterministic estimates, not §8.3.2 context
  costs.
- The CPU trial search is not the product path; no speed claim is made. The
  product path is the GPU candidate batch in `batch.h`.
- The `0.0005` calibration is measured on these fixtures. Re-sweep on the target
  corpus before changing the default; the cliff between 0.005 and 0.01 is real.

The streams do decode with `dav2d`, reconstruct bit-exactly, and show a
consistent RD improvement on all three sizes. That is the property this
experiment establishes.
