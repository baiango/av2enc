# K 52 on `src720.png` with `--nightly-b` — reached

Status: **target met.** The one-frame `--nightly-b` key reaches **K 52.63** at
**35,123 B** (QP 177) and **K 52.08** at **34,161 B** (QP 178). K=52 is first
crossed at **32,867 B**. Both streams decode clean under the pinned `dav2d` and
are byte-for-byte reproducible.

This supersedes the earlier attempt that carried a 25 KB byte budget
(`docs/k52-at-25k.md`): at 25 kB the same arm measures K 47.82, so K 52 and a
25 KB cap are not compatible on this content — but K 52 itself is reached
comfortably once the cap is removed.

Provenance: `build-pub/av2enc` (2026-11-25, `2ebca1e2`), `src720_1f.y4m`
(1280x720, 1 frame, yuv420p limited-range bt709 C420jpeg), CPU `--video
--frames 1 --cpu-threads 8`, Apple M4. Streams in
`~/Downloads/av2enc-src720/hk/k52/`.

## 1. The recipe

```sh
AV2ENC_RD_DRIVER_MIN=16 AV2ENC_RD_DRIVER_LAMBDA=256 \
build-pub/av2enc src720_1f.y4m -o k52.ivf --video --frames 1 --qp 177 \
  --cpu-threads 8 --nightly-b --video-gdf --video-uv-qp-offset -23
# 35,123 B, K 52.63
```

| QP | bytes | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|
| 181 | 30,357 | 7.564 | 12.76 | -1.84 | 50.61 |
| 179 | 32,756 | 7.658 | 11.68 | +1.56 | **51.99** |
| **178** | **34,161** | **7.706** | **13.18** | **+3.49** | **52.08** |
| **177** | **35,123** | **7.734** | **12.33** | **+4.39** | **52.63** |

**K=52 crossing: 32,867 B** (interpolated between QP 179 and QP 178).

Which pieces are load-bearing, **measured at the same 34,161 B**:

| arm | K @34,161 B | dK |
|---|---:|---:|
| **`MIN=16` + `LAMBDA=256` + gdf + uv -23** | **52.08** | — |
| `MIN=16` + gdf + uv -23 (no `LAMBDA=256`) | 52.18 | +0.10 (inside the ripple; the tree price is not load-bearing here) |
| plain `--nightly-b` (no floor, no gdf, uv -8) | **40.95** | **-11.13** |

The plain arm was measured for this document (QP 204/205 bracketing 34,161 B,
`dav2d`-scored). So the **16x16 key floor plus the gdf/chroma pair is worth
~11 K at this rate** — the floor does the work, exactly as
`docs/rd-key-partition-floor.md` found on this picture (+15.1 K at 25 kB, where
the syntax share is far worse).

`docs/rd-key-partition-floor.md` established `AV2ENC_RD_DRIVER_MIN=16` as the
key lever (+15.1 K at a fixed rate on this same picture);
`docs/k-720p-levers.md` added `--video-gdf` and `--video-uv-qp-offset -23`.
This document only confirms that the combination reaches K 52 at 35 kB.

## 2. Conformance and reproducibility

| check | result |
|---|---|
| `tools/validate_dav2d.sh` QP 177 | **exit 0**, `dav2d f9dafbc`, `nb_read_frames=1` |
| `tools/validate_dav2d.sh` QP 178 | **exit 0** |
| QP 177 stream, two fresh runs | 35,123 B `ea24a72d…` both, and identical to the scored stream |
| `tools/kpi.py` QP 177 | `cvvdp=7.7337 butteraugli=12.3256 ssimulacra2=4.390 K=52.63` |
| `tools/kpi.py` QP 178 | `cvvdp=7.7063 butteraugli=13.1773 ssimulacra2=3.491 K=52.08` |

## 3. Context: what this costs in rate

| encoder | bytes for K 52 | note |
|---|---:|---|
| AVM `e6f99d3` | ~25,879 | its measured curve (QP171 K 50.27 @24,204 -> QP170 K 52.15 @26,024) |
| ours | **32,867** | +27 % rate vs AVM |

So the K-52 objective is met, but not at AVM's efficiency: AVM reaches the same
K ~7 kB earlier. That gap is the same coefficient-accuracy deficit recorded in
`docs/detail-gap-720p.md` and `docs/k52-at-25k.md` — we keep ~1.8x AVM's
high-frequency energy while correlating worse (0.51 vs 0.55), i.e. detail is
present but misplaced, which SSIMULACRA2 charges for heavily.

## 4. Reproduce

```sh
tools/hk_eval.py --in src720_1f.y4m --target 35000 \
  --out ~/Downloads/av2enc-src720/hk/k52 --label k52 --qp 181 --validate \
  --env AV2ENC_RD_DRIVER_MIN=16 --env AV2ENC_RD_DRIVER_LAMBDA=256 \
  -- --nightly-b --video-gdf --video-uv-qp-offset -23
```
