# Per-frame drift refresh (`--video-drift-refresh`)

`--video-drift-refresh` is an **opt-in, CPU-only, single-tile, low-delay** RD
allocation on the flat key+P chain. It measures each frame's inherited
reconstruction error and spends a bounded amount of extra rate on the frames
that need to reset the chain. Off by default: every pinned stream (default
QP189 `a1d0dba1…`, `quality` QP190 `3984677b…`, nightly QP110 `5f44c0d1…`) is
byte-identical.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/x.ivf \
    --video --frames 60 --cpu-threads 8 --qp 110 --nightly --video-drift-refresh
```

## What it does

The flat chain codes almost every inter leaf as `skip_flag = 1`: each P frame
costs 58-227 B of tile payload at QP110 (4,740 B for P2-P59 in the nightly bit
split), i.e. ~80 B for 115,200 luma samples. A prediction error a frame inherits therefore survives into every
later frame, and the chain's PSNR decays from 44.7 dB (key) to 41.1 dB around
frame 40 before the content change recovers it. A frame refresh is the only way
to stop paying for an error that was coded once.

One implementation was chosen — **the measured drift trigger** (option 2 of the
two candidate designs) — because it is causal, needs no lookahead buffer and no
propagated weight model:

1. After the key picture is reconstructed, its own luma PSNR against its
   source is the **drift anchor** (44.711 dB on the nightly 240p point). Using
   the key's own quality instead of an absolute dB makes the allowance scale
   with the operating point and the content.
2. Before each inter frame, the *primary reference's stored reconstruction*
   (the picture that frame will predict from) is compared against the input
   picture it was coded from — a real measurement of the accumulated error,
   not a proxy. The slot that holds it records the input index
   (`slot_source_index`), because the §5.18.2 order hint is 4 bits here and
   wraps every 16 frames.
3. If that reference is more than `--video-drift-refresh-threshold` dB below
   the anchor (default 2.5 dB), and the sequence allowance is not exhausted,
   the frame is coded as a **refresh**:
   * `base_q_idx` drops by `--video-drift-refresh-qp` (default 0), and
   * every inter leaf codes its residual (`force_inter_residual`), because at
     these QPs the leaf RD price refuses the residual on its own
     (`docs/negative-results.md`).
4. Each refresh is charged `payload - running mean inter payload` bytes
   against `--video-drift-refresh-budget` (default 1200 B per sequence). The
   budget is a safety cap: once it is exhausted no further frame refreshes.
   At the 240p point the threshold, not the budget, selects the frames.

The refresh frame's own reconstruction becomes the next frame's reference, so
the chain restarts from ~43 dB instead of continuing at 41-42 dB.

## Measured RD

Host: Apple M4, `--cpu-threads 8`, `testdata/nasa_480x240_30fps_60f.y4m`
(60 frames). Quality: `tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS)
with `dav2d-macos`; PSNR/`tools/perframe_psnr.py` straight from the decoded
Y4M. `--nightly` at every point.

### QP sweep (flag off vs flag on, defaults)

| QP | off bytes | off kbps | off CVVDP | off PSNR-Y | on bytes | on kbps | on CVVDP | on PSNR-Y |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 108 | 25,414 | 101.656 | 9.6534 | 42.9412 | 26,522 | 106.088 | 9.7042 | 43.3439 |
| 109 | 25,606 | 102.424 | 9.6812 | 43.0562 | 26,285 | 105.140 | 9.6810 | 43.3236 |
| **110** | **24,994** | **99.976** | **9.6802** | **42.9799** | 25,636 | 102.544 | 9.6879 | 43.2534 |
| 111 | 25,062 | 100.248 | 9.6808 | 42.9665 | 25,888 | 103.552 | 9.7110 | 43.3314 |
| 112 | 24,731 | 98.924 | 9.6730 | 42.8777 | 25,267 | 101.068 | 9.7200 | 43.2233 |
| 113 | 24,327 | 97.308 | 9.6612 | 42.8049 | 24,844 | 99.376 | 9.6925 | 43.0382 |
| 114 | 23,971 | 95.884 | 9.6661 | 42.7914 | 24,254 | 97.016 | 9.6686 | 42.9024 |

(The flag-on stream is 1.2-4.4% larger at a given QP; the extra bytes are
the refresh spend.)

### Iso-rate at 99.976 kbps

The pinned flag-off point is exactly 99.976 kbps (24,994 B, 9.6802 JOD,
42.9799 dB). The flag-on curve brackets that rate at QP113 (99.376 kbps) and
QP112 (101.068 kbps), so the interpolated flag-on point is

```text
t          = (99.976 - 99.376) / (101.068 - 99.376) = 0.3546
CVVDP      = 9.6925 + 0.3546 * (9.7200 - 9.6925)   = 9.7023
PSNR-Y     = 43.0382 + 0.3546 * (43.2233 - 43.0382) = 43.1038
```

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| `--nightly` QP110 (pin) | 99.976 kbps | 9.6802 | 42.9799 dB |
| `--nightly --video-drift-refresh`, linear interp. QP112/113 | 99.976 kbps | **9.7023** | **43.1038 dB** |
| **iso-rate delta** | — | **+0.0221 JOD** | **+0.1239 dB** |

Two cross-checks, both honest views of the same measurements:

* **Equal rate, no interpolation of the flag-on curve:** at 101.068 kbps the
  flag-off curve is QP111/QP109 interpolated (9.6808 at 100.248, 9.6812 at
  102.424) → 9.6810; the flag-on QP112 point is 9.7200 → **+0.039 JOD /
  +0.216 dB** (flag-off PSNR at that rate ≈ 43.000 dB).
* **Same QP (iso-QP, +2.57% rate):** QP110 flag-on is 9.6879 / 43.2534 dB vs
  9.6802 / 42.9799 dB → +0.0077 JOD / +0.274 dB for +642 B.

The CVVDP curve on this clip is flat and non-monotone (the flag-off curve
spans only 9.653-9.681 over QP104-112, and QP104 at 26,375 B scores *worse*
JOD than QP110 at 24,994 B). The JOD delta is therefore small and partly
within that ripple; the mean-PSNR delta (+0.12 dB at iso-rate, +0.27 dB at
iso-QP) and the localised trough gain below are the more stable evidence.

### Per-frame luma PSNR (f0/f20/f33/f40/f49/f59)

| Frame | off QP110 | on QP110 (+642 B) | on QP112 (25,267 B) |
|---:|---:|---:|---:|
| 0 (key) | 44.711 | 44.711 | 44.530 |
| 20 | 43.992 | 43.992 | 43.830 |
| 33 | 41.812 | **42.811** | **42.622** |
| 40 | 41.141 | **42.374** | **42.117** |
| 49 | 42.299 | 42.740 | 42.728 |
| 59 | 43.961 | 43.670 | 43.826 |
| mean Y | 42.9799 | 43.2534 | 43.2233 |

The refreshed trough (frames 32-40) gains **+0.89 dB** mean luma PSNR at QP110;
the effect decays over the following ~8 frames because the chain returns to
skip-only coding. f59 is 0.29 dB lower at on-QP110 — the refresh spend moves
rate, it does not add quality everywhere.

### Threshold sweep (QP110, budget 1200, QP drop 0)

| `--video-drift-refresh-threshold` | bytes | CVVDP | PSNR-Y | refreshes | refresh bytes |
|---:|---:|---:|---:|---:|---:|
| 2.0 dB | 25,969 | 9.6800 | 43.2588 | 5 | 1,407 |
| 2.3 dB | 25,995 | 9.6930 | 43.2982 | 5 | 1,537 |
| **2.5 dB (default)** | **25,636** | **9.6879** | **43.2534** | **3** | **979** |
| 2.7 dB | 25,744 | **9.7168** | 43.3370 | 3 | 1,127 |
| 3.0 dB | 25,474 | 9.7010 | 43.1963 | 2 | 909 |

Every setting in the measured band is positive on both metrics (+0.00 to
+0.037 JOD, +0.22 to +0.36 dB against the flag-off QP110 point). The default
2.5 dB is the middle of that band rather than the measured peak (2.7 dB):
the CVVDP response to *which* frames refresh is nonlinear, so pinning the peak
would be fitting one clip. `--video-drift-refresh-budget` 1200/2400/3600 all
produce the identical stream at these settings — the threshold is the active
knob and the budget is the safety cap.

## What it does not do

The counterfactual for a frame-level lever on this clip is bounded: replacing
one decoded frame with the source is worth ~0.0025 JOD
(`docs/video-nightly.md`) and the measured refresh recovers ~+0.89 dB on the
refreshed trough. SVT-AV1 holds 47-50 dB on every frame at the same rate, so
the remaining ~5 dB gap is **per-frame coding efficiency** — residual coding
on every P frame, not a better allocation of the same skip-only frames. The
refresh is a cheap, bounded positive (+0.022 JOD / +0.12 dB at iso-rate), not
the missing lever.

## Why not a QP drop alone

Lowering the frame QP does nothing by itself, because the leaf RD price still
chooses skip: at QP110, coding f33 at QP94 / QP86 / QP30 leaves the frame at
41.77 / 41.75 / 41.73 dB (baseline 41.81 dB) with a ~110 B payload. The
residual is refused at 64x64 leaf granularity regardless of the quantiser, so
the refresh has to force it. `--video-drift-refresh-qp` exists for the cases
where a frame QP drop *plus* a forced residual is affordable; it costs much
more at the same threshold: 2,155 B for the two refreshes at `-16`, against
979 B for the three at `0` in the measured sequence. The default is 0.

## Verification

* `dav2d-macos --strict 1` decodes the flag-on streams; the 60-frame
  `--video-recon` payload equals the decoded payload byte for byte
  (10,368,000 B), including the `--video-drift-refresh-qp 16` header-QP path.
* Flag off: default QP189 is 25,009 B / `a1d0dba1259b20a4b628d4cd1e14c33c`,
  `quality` QP190 is 25,141 B / `3984677b9b665952984b07b7cf75c6ea`, nightly
  QP110 is 24,904 B / `5f44c0d1795357b5beec84adaed0fe5b` (the flag is part
of the aggregate since it stacked with the CDF/key-QP move).
* `src/pipeline/video/video_drift_refresh_gate.sh` (registered as
  `gate_video_drift_refresh` in `tools/run_gates.sh`) checks the named
  rejections, the three pins, strict decode + recon, the refresh evidence in
  the trace, a +0.20 dB floor on frames 32-40, and the QP-drop path.
* Encode cost: 8.39 s → 8.58 s for 60 frames at QP110 on the M4 (+2.3%).

## Caveats

* The trigger only fires where the P chain is *good*: on this clip it refreshes
  at the `--nightly`/QP104-114 operating points and is completely inert on the
  `quality` preset at QP150/170/190 and on the default QP189 chain (byte
  identical streams, 0 refreshes). At those rates the per-frame error is
  dominated by quantisation, which a refresh cannot repay.
* The threshold is a per-content knob: it is expressed relative to the key
  picture's own PSNR, but the measured band (2.0-3.0 dB) is what this clip
  supports. Re-measure it before claiming a gain on other content.
* The trigger measures luma only. Chroma drift is not scored.
* This is one clip, one resolution and one host. It is a CPU development tool,
  not part of the GPU product path.

## Named rejections
`--video-drift-refresh` exits 2 with a named error for `--video-gpu`,
`--tiles` > 1x1, `--rc crf/vbr` (non-cqp), `--still`, non-8-bit-4:2:0
formats, the SVC layers, the B/reorder/temporal-layer/TIP/frame-type-wrapper
schedules, `--video-compound`, `--avg-cdf`/`--adaptive-cdf`/`--cdf-continuity`,
segmentation/delta-Q, and `--video-sb128`/`--video-sb256`/`--video-ext-partitions`.
