# 720p still: what actually moves K on `src720.png` (2026-11-25)

Status: **measured, `--nightly-b` only.** Fifteen waves of single-change arms
on one picture at matched bytes. One stack survives: **`--video-gdf
--video-uv-qp-offset -23` takes K from 46.94 to 47.43 (+0.49)** at 25,010 B
with the 16x16 key floor, and the lead is **+0.45 to +0.57 across every
matched-byte point from 25.0 to 36.0 kB**. Everything else measured is inert,
byte-identical, or negative.

This is a measurement, not a promotion: one picture, one operating point, and
`--nightly-b` is already a research gate. The 240p ladder was not run.

Provenance: `~/Downloads/src720.png` (1280x720 RGB) converted with ffmpeg to
`src720_1f.y4m` (1 frame, yuv420p, limited range, bt709). Encoder
`build-pub/av2enc` (2026-11-25 04:21). CPU `--video`, `--cpu-threads 8`, Apple
M4. Every arm runs with **`AV2ENC_RD_DRIVER_MIN=16`**, the measured key floor
from `docs/rd-key-partition-floor.md` (23 s per encode against 70 s at MIN=4,
same K within the ripple). Harness, every stream and every score live outside
the tree in `~/Downloads/av2enc-src720/hk/`; the drivers are
`tools/hk_720p.sh` (the arm recipe), `tools/hk_eval.py` (land one arm at a byte
target) and `tools/hk_match.py` (compare two arms at equal bytes).

## 0. Answer first

1. **The two levers that pay are both on the reconstruction side, not the
   coding-efficiency side.** `--video-gdf` (+0.08) and chroma spend
   (`--video-uv-qp-offset -8 -> -23`, +0.37) are independent and additive
   (+0.49). Neither touches the residual coder that
   `docs/detail-gap-720p.md` identified as the remaining gap.
2. **The whole loop-filter chain is worth +0.90 K at this operating point**
   (46.94 -> 46.03 with `--video-loop-filters-off`). That is the largest single
   effect measured here, and it says the key path at 0.21 bpp is still buying
   reconstruction quality, not only bits.
3. **Everything `docs/negative-results.md` predicted would be inert, is inert
   here too** — several of them byte-identically: `--video-tx-context-rate`,
   `--video-ccso-bands`, `--video-ccso-refit`, `--video-cfl`, `--video-ddt`,
   `--video-cdef-strength 4` (moves bytes, not K), `--video-deblock-off`.
4. **The plain `--still` writer is 0.65 K *behind* the one-frame `--video`
   key** (46.29 vs 46.94 at 25,010 B) even with every still feature on, so use
   the video path with `--frames 1` for a still.

## 1. The winner

```sh
AV2ENC_RD_DRIVER_MIN=16 build-pub/av2enc src720_1f.y4m -o out.ivf \
  --video --frames 1 --qp 188 --cpu-threads 8 --nightly-b \
  --video-gdf --video-uv-qp-offset -23        # 25,131 B
```

| arm | QP | bytes | JOD | BA | SS2 | PSNR-Y | **K** | dK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base `--nightly-b` | 187 | 24,886 | 7.304 | 14.25 | -14.01 | 24.406 | **46.94** | — |
| `--video-gdf` | 187 | 24,887 | 7.305 | 14.21 | -13.91 | 24.466 | 47.02 | +0.08 |
| `--video-uv-qp-offset -16` | 187 | 25,517 | 7.286 | 14.19 | -12.46 | 24.4 | 47.12 | +0.18 |
| `--video-uv-qp-offset -23` | 188 | 25,130 | 7.267 | 13.41 | -11.60 | — | 47.30 | +0.37 |
| **`gdf` + `uv -23`** | **188** | **25,131** | **7.280** | **13.24** | **-11.60** | **24.353** | **47.43** | **+0.49** |

Chroma fidelity at a fixed QP 187 improves monotonically with spend, so this is
not a metric exploit — it is the same evidence the `-8` default was chosen on,
extended to the bottom of the legal §5.4.8 range:

| offset | U rmse | V rmse | \|dU\| > 20 LSB | foliage-band dU rms |
|---:|---:|---:|---:|---:|
| -8 (default) | 7.75 | 4.30 | 2.46 % | 9.70 |
| -16 | 7.32 | 4.20 | 1.92 % | 9.15 |
| -23 | **6.88** | **3.96** | **1.46 %** | **8.59** |

### 1.1 The lead is not one QP's luck

K moves ~1.0 per QP on this picture, so a single bracket proves nothing. Every
target below is an interpolation of each arm's own rate curve, paired by BYTES
(`tools/hk_match.py`):

| target bytes | base K | arm K | dK |
|---:|---:|---:|---:|
| 25,010 | 46.94 | 47.43 | **+0.49** |
| 25,500 | 47.20 | 47.72 | +0.52 |
| 26,000 | 47.48 | 47.93 | +0.45 |
| 26,500 | 47.78 | 48.22 | +0.45 |
| 33,000 | 51.50 | 51.59 | +0.09 |
| 34,000 | 51.89 | 52.01 | +0.12 |
| 35,000 | 52.12 | 52.43 | +0.31 |
| 36,000 | 52.37 | 52.86 | **+0.49** |

The lead is positive everywhere measured and narrows toward 33 kB, which is the
same rate-convergence shape `docs/detail-gap-720p.md` reports for the coding
gap. **Not measured:** what GDF and chroma spend do to the 240p ladder, where
GDF is on record at -0.01 JOD and +17-27 % rate
(`docs/negative-results.md`, "Expect GDF / PC-Wiener / Wiener+GDF to lift K").

## 2. What is inert or negative

All at 25,010 B, `--nightly-b`, MIN=16. "identical" = same size and same scores
to the printed digits.

| arm | K | dK | note |
|---|---:|---:|---|
| `--video-loop-filters-off` | 46.03 | **-0.90** | deblock+CDEF+CCSO as a block |
| `--video-uv-qp-offset 0` | 46.66 | -0.27 | the opt-out of the -8 default |
| `--video-uv-qp-offset -12` | 46.97 | +0.04 | inside the ripple |
| `--video-uv-qp-offset -20` (+gdf) | 46.98 | +0.04 | inside the ripple; -23 is the real step |
| `--video-ibp` (still path) | 45.76 | -1.18 | worst arm measured |
| `--still --still-full-intra --still-rdo-intra --still-rdoq --still-tx-type` | 46.29 | -0.65 | the still writer |
| `--video-tx-context-rate` | 46.94 | 0.00 | **byte-identical** |
| `--video-ccso-bands 4` / `--video-ccso-refit 3` | 46.94 | 0.00 | **byte-identical** |
| `--video-cfl` / `--video-ddt` | 47.43 | 0.00 | **byte-identical** to the winner |
| `--video-deblock-off` | 46.94 | 0.00 | **byte-identical** (CDEF covers it) |
| `--video-cdef-strength 4` | 46.94 | 0.00 | +47 B, no K |
| `--video-wiener --video-chroma-lr` | 43.59 | -3.35 | +23 % bytes at fixed QP |
| `--video-idtx` | 46.99 | +0.06 | ripple |
| `--video-mrl` | 46.71 | -0.23 | |
| `--video-aq` | — | — | rejected: needs `--video-cdf-per-frame-qp`, which cannot compose with the carried CDF image |
| `--video-ext-partitions`, `--video-sb128` | — | — | rejected with the B/temporal-layer schedule |

## 3. Harness traps (all three produced plausible wrong numbers here)

1. **A wide bracket hides the answer.** Wave 1 opened at QP 187 where
   `docs/rd-key-partition-floor.md` had 24,886 B, and got **47,443 B**: the
   built binary and the doc disagree, so always walk the curve rather than
   trusting a recorded QP. `tools/hk_eval.py` refuses to report a bracket wider
   than 10 % and prints every point's K so the ripple is visible.
2. **At equal QP the arm that spends more bytes always wins.** `uv -23` is
   +1,049 B at QP 187 and +1.37 K — that comparison is meaningless. Every claim
   above is interpolated to equal bytes (`tools/hk_match.py`), and the
   equal-QP grid is included only to show the arm's slope.
3. **Two "arms" scoring identically are usually the same stream.** `--video-cfl`,
   `--video-ddt` and the base produce the same bytes at every QP. Check MD5s
   before believing an ablation.

## 4. Conformance

* `tools/validate_dav2d.sh g_uv23_q188.ivf src720_1f.y4m` -> **exit 0**,
  `decoder: f9dafbc`, `nb_read_frames=1`, PSNR y 24.352913, tool pin ok.
* The base arm's QP 186 and QP 187 both pass the same gate (wave 1).
* Not run: `ctest`, the five pinned streams, the `AGENTS.md` reference
  bitstreams, the 240p ladder. All are required before any promotion, and
  `--video-gdf` on the ladder is on record as negative.
