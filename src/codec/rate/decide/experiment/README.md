# M1 RDO decision experiment

This directory contains a **patch-neutral** CPU experiment for
`src/encoder/m1/rdo_decide/`. It never edits `src/encoder/m1/still_encoder.cpp`;
`patch_still_rdo_decide.py` writes a patched copy under `build-rdo-decide/`, and
all hooks are env-gated. With `AV2ENC_RDO_DECIDE` unset, the patched binary must
be byte-identical to the unpatched CLI for the `dc` and `sse` cases. The runner
checks that.

## Run

```sh
cd src/encoder/m1/rdo_decide/experiment
./run_still_rdo_decide.sh
./run_still_rdo_decide.sh --scales 0.25,0.5,1.0
./run_still_rdo_decide.sh --input path.y4m --work /tmp/rdo-decide
```

Defaults:

- inputs: `testdata/basic_64x64_1f.y4m`, `tests/data/basic_640x360_1f.y4m`
- scale: `0.5` (pass `--scales` for a sweep)
- core library: `build/libav2enc_core.a`
- reference CLI: `build/av2enc`

The script builds the layer sources, `tx_type.cpp`, the current `rdo`,
`rdoq` and `rdoq_av2` sources, and the patched M1 copy. If `build/av2enc` is
stale relative to the current M1 sources, rebuild `av2enc_core` first
(`cmake --build build --target av2enc_core`).

## Cases

| Case | Command | Meaning |
|---|---|---|
| `dc` | `--still` | current default, DC luma |
| `sse` | `--still --still-full-intra` | SSE-only full intra, legacy partition |
| `rdo_part` | `--still --still-full-intra` + `NO_INTRA` | rate-aware partition only |
| `rdo_chroma` | `--still --still-full-intra` + `NO_INTRA NO_PART` | rate-aware chroma only |
| `rdo_nopart` | `--still --still-full-intra` + `NO_PART` | rate-aware intra+chroma |
| `rdo_single` | `--still-single --still-full-intra` | no partition splits |
| `tx_builtin` | `--still-tx-type` | existing per-nonzero TX selector |
| `tx_rdo_decide` | `--still-tx-type` + isolated layer | stream-rate TX selector |
| `rdoq_builtin` | `--still-rdoq` | repository full-trellis RDOQ |
| `rdoq_rdo_decide` | `--still-rdoq` + isolated layer | allocation-free RDOQ |
| `rdo@S` | full stack at `sse_scale = S` | main comparison |

`NO_INTRA`, `NO_PART`, `NO_CHROMA`, `TXTYPE`, `RDOQ` are layer switches; the
`*_builtin` cases leave them unset. Every RDO stream is verified per sample
with `tools/still_verify/verify.py`, and every stream is decoded/scored by
`tools/quality.py` (dav2d + PSNR + CVVDP).

## Results

The measured tables and interpretation are in the parent
[`README.md`](../README.md). Summary at `sse_scale = 0.5`:

| Fixture | Bytes | PSNR-Y | CVVDP | vs DC |
|---|---:|---:|---:|---|
| 64x64 RDO full | 955 | 42.4526 | 9.9151 | +19 B, -0.049 dB, +0.0142 JOD |
| 640x360 RDO full | 13,770 | 47.6242 | 9.9415 | +415 B, +2.535 dB, +0.0548 JOD |

Full intra rate-aware is a rate fix versus SSE-only full intra; partition
splits are a net RD win on 640x360 and neutral on 64x64. RDOQ matches the
repository path byte-for-byte.

## Why this is not a final BD-rate claim

- One frame per fixture, CPU reference path, no speed claim.
- The TX/RDOQ stream rate is the static §9.3/CDF model, not the byte-exact
  §8.2 entropy coder.
- `estimate_leaf_rate_bits()` is a variance-based proxy: the partition tree is
  chosen before prediction/transform, so exact leaf costs are not available
  without a second reconstruction pass.
- The lambda sweep is non-monotonic; `0.5` is a measured compromise on these
  two fixtures, not a target-corpus default.

The streams do decode and reconstruct bit-exactly, which is the property this
experiment is meant to establish.
