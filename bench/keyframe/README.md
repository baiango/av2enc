# Keyframe contest (single picture, no temporal structure)

One 480x240 frame (`testdata/nasa_480x240_30fps_60f.y4m`, frame 0) encoded by
each encoder in isolation, decoded and scored with the same tools. No GOP, no
alt-refs, no rate control interactions: this isolates the intra coder.

## Commands

```sh
# av2enc (1 picture; --video-key-tx-lambda 0 inherits --video-tx-lambda)
build/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/a.ivf \
  --video --frames 1 --qp <Q> --nightly [--video-key-tx-lambda 16]

# SVT-AV1 preset 0
SvtAv1EncApp -i testdata/nasa_480x240_30fps_60f.y4m -b /tmp/s.ivf \
  --preset 0 --crf <C> -n 1 --progress 0
dav1d -q -i /tmp/s.ivf -o /tmp/s.y4m

# libaom (reference)
aomenc --cpu-used=<0|4> --end-usage=q --cq-level=<C> --limit=1 \
  --passes=1 -o /tmp/o.ivf testdata/nasa_480x240_30fps_60f.y4m
```

Scored with `tools/quality.py --test` (CVVDP 0.5.7, `standard_fhd`, MPS) on the
decoded picture; bytes are the IVF file size.

## Measured

| Encoder | Setting | Bytes | CVVDP | PSNR-Y |
|---|---|---:|---:|---:|
| av2enc | QP120, lambda 136 | 16,156 | 9.8837 | 44.90 |
| av2enc | QP105, lambda 136 | 20,266 | 9.9083 | 46.22 |
| av2enc | QP90, lambda 136 | 25,387 | 9.9203 | 47.74 |
| av2enc | QP60, lambda 136 | 37,065 | 9.9261 | 49.30 |
| av2enc | QP120, lambda 16 | 16,322 | 9.9117 | 45.50 |
| av2enc | QP105, lambda 16 | 20,647 | 9.9558 | 48.32 |
| av2enc | QP90, lambda 16 | 25,988 | 9.9840 | 51.25 |
| av2enc | QP60, lambda 16 | 37,940 | 9.9973 | 55.82 |
| SVT p0 | CRF8 | 17,052 | 9.9784 | 51.15 |
| SVT p0 | CRF6 | 19,247 | 9.9873 | 52.80 |
| SVT p0 | CRF3 | 24,640 | 9.9972 | 56.33 |
| SVT p0 | CRF10 | 15,376 | 9.9644 | 49.72 |
| libaom cu0 | cq20 | 9,989 | 9.8636 | 44.87 |
| libaom cu4 | cq20 | 10,213 | 9.8669 | 44.67 |
| libaom cu4 | cq30 | 7,359 | 9.7016 | 40.80 |

## Corrections after the CDEF and lambda fixes

Two defects were found by this harness after the table above was taken, so the
rows above are superseded:

1. **CDEF could pick a filter that raises SSE.** On a QP20 keyframe: 53,140 B /
   57.34 dB with CDEF vs 53,138 B / 58.36 dB without (-1.02 dB for +2 B). It also
   capped the whole curve near 55.5 dB (QP10 scored worse than QP20). Fixed by
   rejecting a plan whose `sse_after` exceeds `sse_before`; the curve now reaches
   58.36 dB at QP20 and 60.44 dB at QP10.
2. **`--video-tx-lambda 136` caps the intra RD curve** (see above). Use
   `--video-key-tx-lambda 16`.

Post-fix keyframe RD (lambda 16, CDEF fix, key offset 0 at low QP):

| QP | Bytes | CVVDP | PSNR-Y |
|---:|---:|---:|---:|
| 120 | 16,322 | 9.9117 | 45.50 |
| 105 | 20,647 | 9.9558 | 48.32 |
| 90 | 25,988 | 9.9840 | 51.25 |
| 60 | 37,940 | 9.9973 | 55.82 |
| 20 | 53,138 | 9.9980 | 58.36 |
| 10 | 59,174 | 9.9990 | 60.44 |

SVT at the same bytes is still ahead: CRF6 (19,247 B) is 52.80 dB vs our 48.32
at 20,647 B, CRF3 (24,640 B) is 56.33 vs our 51.25 at 25,988 B. The gap is
~4.5-5 dB in PSNR and ~0.03 JOD, so it is coding efficiency, not the two defects.
Note the earlier "2.4x rate" framing is not reproducible as a rate ratio on a
single frame (1.4x-3.4x depending on the criterion used).

## Findings

- **`--video-tx-lambda 136` caps the intra RD curve.** It is the inter weight
  applied to intra. At QP60 the same frame reaches 49.30 dB at 37,065 B, while
  `--video-key-tx-lambda 16` reaches 55.82 dB for +2.4% rate; 136 asymptotes at
  49.67 dB even at 45.5 kB. The mechanism is measured in
  `docs/keyframe-bit-accounting.md`: identical `pred_sse`, `recon_sse`
  86,990 -> 17,946, 53% of the SSE in 6 of 450 16x16 regions - an over-large
  weight abandons whole leaves.
- **Fixing lambda does not close the gap to AV1.** SVT scales just as steeply:
  ~5 dB ahead at 17, 20 and 25 kB. Our deficit is a genuine coding-efficiency
  gap at every rate, not a tuning artifact.
- In CVVDP the gap shrinks with rate (0.067 at 17 kB, 0.032 at 20 kB, 0.013 at
  26 kB) because the metric saturates near 10 JOD while PSNR keeps separating.
- libaom at cpu-used 0 and 4 is behind SVT on intra, so SVT is the reference to
  beat here.
- 98.1% of our keyframe tile is coefficient syntax, and intra has no trellis
  quantisation (`allow_rdoq = false` at every intra site); the coder itself is
  only 1.5% above its own entropy bound.
