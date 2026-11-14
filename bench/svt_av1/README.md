# SVT-AV1 reference point

Cross-encoder calibration on the NASA 480x240/30 60-frame clip, not an AV2
parity claim. Encoder: **SVT-AV1 v4.1.0 (release)**, macOS arm64.

## Command

```sh
SvtAv1EncApp \
  -i testdata/nasa_480x240_30fps_60f.y4m \
  -b /tmp/svt_crf42.ivf \
  --preset 4 --crf 42 \
  --fps-num 30 --fps-denom 1 --progress 0

dav1d -q -i /tmp/svt_crf42.ivf -o /tmp/svt_crf42.y4m
DAV2D=tools/bin/dav2d-macos tools/quality.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m \
  --test /tmp/svt_crf42.y4m --json /tmp/svt_crf42.json
```

## Result

| Stream | IVF bytes | kbps | CVVDP | PSNR-Y | Encode |
|---|---:|---:|---:|---:|---:|
| SVT-AV1 CRF 42, preset 4 | 25,039 | 100.16 | **9.7560 JOD** | **45.9248 dB** | 306 fps (0.196 s/60 f) |

CRF sweep: CRF 40 = 106.2 kbps, 41 = 101.7, **42 = 100.16**, 43 = 95.7.
`dav1d` decoded all streams; the same `tools/quality.py` CVVDP 0.5.7 /
`standard_fhd` / MPS model used for AV2 scored the decoded Y4M.

At the same 100 kbps point, the current `--nightly` AV2 stream is 9.5868 JOD /
41.6953 dB, so SVT-AV1 is +0.169 JOD / +4.22 dB on this clip. Encoder speed is
not comparable: SVT-AV1 preset 4 is 306 fps, nightly AV2 is 7.6 fps on the M4.
