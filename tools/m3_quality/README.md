# M3 quality / BD-rate harness

Measurement only. This directory scores the M1 baseline and the M3 tools on
**decoded** output; it does not change encoder code, `src/main.cpp`, or
`CMakeLists.txt`.

The harness was built while the encoder was still pinning `base_q_idx = 90`.
During the run the coefficient-CDF qctx integration landed, so the current
report exercises the full QP range. If you run it on a build that still rejects
other QPs, it automatically records the pin, reports a single operating point,
and refuses to compute BD-rate. It never turns a one-point rate delta into a
BD-rate.

## Run

```sh
cd $HOME/work/av2enc

python3 tools/m3_quality/run_m3_quality.py \
  --encoder build-m1/av2enc \
  --out tools/m3_quality/results
```

Useful options:

| Option | Meaning |
|---|---|
| `--qps 60,90,110,135,160,185,210,235` | QP probe list (default) |
| `--bd-qps 90,110,135,160,185,210` | primary BD-rate subset (default: core set) |
| `--tools m1_baseline,cdef,ccso,rdoq,tx_type,rdo_intra` | run a subset |
| `--frame-index N` | extract frame N from each source (default 0) |
| `--source-360p PATH` / `--source-1080p PATH` | override benchmark clips |
| `--bd-method pchip\|cubic` | BD-rate interpolation (default `pchip`) |
| `--reuse` | reuse an IVF + quality JSON if source/size/QP/encoder match |
| `--no-plot` | skip PNG generation |
| `--allow-mixed-hardware` | write a report but mark mixed-host comparisons invalid |

The defaults use the two repository benchmark clips:

- 640x360: `bench/avm_360p/shine_360p_2s.y4m`
- native 1920x1080: `bench/avm_1080p/shine_1080p_2s.y4m`

Both are the same LG QNED “Shine” 4K source at 10 s; the 640x360 fixture is a
downscale. The harness extracts exactly one frame from each and hashes it. It
then scales the 1080p frame to 640x360 and checks PSNR against the 640x360
fixture (currently 65.7 dB) to prove the two resolutions are the same source
frame, not merely similarly named files.

## What is measured

| Tool | CLI | Status in this run |
|---|---|---|
| M1 baseline | `--m1` | measured |
| M1 full intra (control) | `--m1 --m1-full-intra` | measured; anchors the RDO incremental view |
| CDEF | `--m1 --m1-cdef` | measured |
| CCSO (band-offset) | `--m1 --m1-ccso` | measured |
| RDOQ | `--m1 --m1-rdoq` | measured |
| Intra TX type | `--m1 --m1-tx-type` | measured |
| RDO (intra) | `--m1 --m1-rdo-intra` | measured |
| Wiener NS | — | **unmeasured**: no M1 bitstream path / coefficient fit |
| Delta-Q | — | **unmeasured**: M1 header codes `delta_q_present = 0` |
| QM | — | **unmeasured**: no QM OBU/setup path in the still CLI |

A tool with no verified stream gets a row with status `not_wired` and no metric
or verdict. Do not read the absence of a row as a zero effect.

RDO here is **intra-mode rate-aware decision**, not partition/inter Lagrangian
RDO. Its pointwise deltas and BD-rate are anchored to the full-intra control so
the full-intra search is not misattributed to the RDO rate term. The report also
keeps the baseline-relative delta for transparency.

## Gates

Every stream must pass both:

1. `dav2d --strict 1` decode, and
2. the repository `tools/validate_dav2d.sh` wrapper.

Only then does the harness call `tools/quality.py` (PSNR, SSIM, CVVDP). Each
per-point quality report also verifies that `frames == 1` and that the IVF size
matches the stream on disk.

The harness enforces:

- **same source**: one canonical extracted Y4M hash per resolution;
- **same frame**: frame 0 by default, recorded in the manifest;
- **same host**: every quality report must share hostname/OS/machine/CPU/GPU;
- **same decoder**: one `dav2d` binary/version;
- **same CVVDP**: one version, display model, and device.

A mismatch marks the comparison invalid and suppresses net-win/loss claims.
A non-target host (`cpu-reference`, a 12 GB dev GPU) is flagged as a proxy for
speed; quality and bitrate are still host-independent.

## BD-rate

BD-rate uses the standard integral over the common quality range:

```text
BD-rate = 10 ** mean(log10(R_test) - log10(R_anchor)) - 1
```

- Primary fit: core QPs `90,110,135,160,185,210`.
- Sensitivity: all supported QPs, including QP60 (near CVVDP saturation) and
  QP235 (near-collapse). Both are reported; a sign flip is called out.
- `pchip` is the default interpolation (monotone, no overshoot); `cubic` is
  available for comparison.
- BD-rate needs at least three QPs on both curves and a non-empty common
  quality range. Otherwise the result is `available: false` with a reason.

### Verdict rules

A metric is:

- **win** at BD-rate ≤ −1.0%,
- **loss** at BD-rate ≥ +1.0%,
- **neutral** in between.

A metric whose per-QP deltas never exceed its noise floor is treated as neutral
and cannot flip the verdict:

- CVVDP noise floor: **0.02 JOD**
- PSNR-Y noise floor: **0.02 dB**

If PSNR-Y and CVVDP disagree beyond those floors, the verdict is
`metric_dependent` — not a net win. If the two resolutions disagree, the overall
verdict is `mixed`; read the per-resolution cells.

Single-QP fallback (older builds without qctx) uses pointwise thresholds:
rate change material at ±0.5%, quality gain at ≥0.02 JOD or ≥0.05 dB, quality
loss at ≤−0.02 JOD, ≤−0.05 dB, or ≤−0.001 SSIM-All. Rate down **and** quality
down is `tradeoff`, never a win.

## Current result (2026-11-11)

Full report: [`results/summary.md`](results/summary.md) and
[`results/bd_rate.md`](results/bd_rate.md). R-D plots:
`results/rd_curve_640x360.png`, `results/rd_curve_1920x1080.png`.

Snapshot, core-QP BD-rate on decoded output:

| Tool | 640x360 PSNR-Y | 640x360 CVVDP | 1920x1080 PSNR-Y | 1920x1080 CVVDP | Verdict |
|---|---:|---:|---:|---:|---|
| CDEF | −0.63% | +1.49% (noise) | −2.32% | +1.81% (noise) | neutral @360, win @1080 → mixed |
| CCSO | −0.32% | +2.59% | −1.58% | +1.92% | loss @360, metric-dependent @1080 |
| RDOQ | −15.27% | +7.24% | −16.48% | +10.58% | metric-dependent |
| TX type | +0.15% | +0.16% (noise) | +0.01% | +0.01% (noise) | neutral |
| RDO (intra) | −0.94% | −0.62% | −3.69% | −2.35% | neutral @360, win @1080 → mixed |
| Wiener NS | — | — | — | — | unmeasured (no stream) |
| Delta-Q | — | — | — | — | unmeasured (no stream) |
| QM | — | — | — | — | unmeasured (no stream) |

Negative BD-rate is less rate at equal quality. The full-intra control is a loss
at both resolutions; that is why RDO is anchored to it.

**Bottom line:** no tool is a clean net win on both PSNR-Y and CVVDP at both
resolutions in this run.

- **Closest to a win:** RDO (intra), especially at 1080p; CDEF is a PSNR win at
  1080p but its CVVDP change is inside the noise floor and it is neutral at
  640x360.
- **Neutral:** TX type is effectively a no-op above QP110 on this content.
- **Metric-dependent:** RDOQ buys 15–16% rate on PSNR but loses 7–11% on CVVDP;
  CCSO is a PSNR loss at 640x360 and its CVVDP direction disagrees with PSNR at
  1080p. Neither can be called a net win without a perceptual review.
- **Unmeasured:** Wiener NS, delta-Q, QM. No stream, no claim.

## Caveats

1. **One frame per QP.** This is a still-picture path; it is not a video
   sequence result.
2. **One content type.** Smooth TV-demo footage. Absolute rates are easy-content
   numbers.
3. **CPU reference host.** Quality/bitrate are host-independent; encode speed is
   not target validation. The report records the actual host and flags
   `cpu-reference`/`proxy-gpu`.
4. **640x360 is a downscale**, not a native capture. The cross-resolution check
   proves it is the same source frame, but it remains a low-resolution proxy.
5. **CVVDP saturation.** At QP60–90 CVVDP is near 10 JOD. Tiny JOD changes are
   inside the noise floor; the report does not let them flip a verdict.
6. **QP range sensitivity.** QP60 and QP235 are included in the sweep but
   excluded from the primary BD-rate fit. The all-QP sensitivity is reported.
7. **Filter decisions are isolated.** Each M3 tool is enabled alone on top of
   M1. No joint optimization, filter-order ablation, or rate-aware filter
   parameter search is implied.
8. **Decoder gate.** Every reported metric comes from a stream that passed
   `dav2d --strict 1` and `tools/validate_dav2d.sh`. A failed stream has no
   metric.

## Files

```
tools/m3_quality/
  run_m3_quality.py   orchestration, gates, verdicts, reports
  bd_rate.py          BD-rate/BD-PSNR + self-test
  README.md           this file
  results/            generated (git-ignored)
    summary.md
    summary.json
    bd_rate.md
    bd_rate.json
    manifest.json
    rd_curve_640x360.png
    rd_curve_1920x1080.png
    encodes/          verified IVFs
    quality/          raw tools/quality.py JSON per point
    logs/             encode, strict-decode, validate, quality logs
    sources/          canonical extracted one-frame Y4M fixtures
```

Run the BD-rate math self-test:

```sh
python3 tools/m3_quality/bd_rate.py --self-test
```
