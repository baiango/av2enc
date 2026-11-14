# Nightly CPU RD mode (`--nightly`)

`--nightly` is the latest measured CPU-only algorithmic RD aggregate. It is
**opt-in and CPU-only**: the default stream, the stable `quality` preset and
every GPU path keep their historical decisions and byte-identical streams.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/nightly.ivf \
    --video --frames 60 --qp 142 --cpu-threads 8 --nightly
```

Expansion (the gate asserts this is byte-identical to typing the flags out):

```text
--video-rd-preset quality
--video-ccso-rd
--video-edge-inter
```

The two algorithmic additions are:

| Tool | Change | Measured effect |
|---|---|---|
| CCSO RDO | prices the unary offset table, sweeps `scale_idx`, disables units below their symbol cost | ≈−3% BD-rate on the old tree; small on top of the edge path |
| Bottom-edge inter | pads the reference/recon to the SB extent so the bottom 64x64 row can use inter on its visible rows | removes the forced-intra bottom band |

`--nightly` rejects `--video-gpu` and `--still`, and the bottom-edge path
requires the single-tile low-delay P chain (no B/reorder/temporal-layer modes,
no `--video-ref-mvs`). Those restrictions are validated by the normal
combination checks.

## Measured RD

Host: Apple M4, `AV2ENC_CPU_FAST=ON`, `--cpu-threads 8`. Clip:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Quality: `tools/quality.py`
(CVVDP 0.5.7, `standard_fhd`, MPS) with `dav2d-macos`.

| QP | kbps | CVVDP | PSNR-Y |
|---:|---:|---:|---:|
| 141 | 100.884 | 8.7539 | 35.4699 |
| **142** | **100.248** | **8.7465** | **35.3754** |
| 143 | 99.044 | 8.6982 | 35.2364 |
| 144 | 96.840 | 8.6811 | 35.1690 |

At the established 100 kbps point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| default QP189 | 100.036 kbps | 4.8021 | 24.4667 dB |
| `quality` QP190 | 100.564 kbps | 6.3021 | 27.5816 dB |
| **`--nightly` QP142** | **100.248 kbps** | **8.7465** | **35.3754 dB** |

At 100.248 kbps the nightly stream is **+3.94 JOD** over the default and
**+2.44 JOD** over the stable `quality` preset, for +7.79 dB PSNR-Y over the
default. Interpolated to 100.036 kbps the nightly CVVDP is about **8.735**.

Encode time is ~4 s for 60 frames on the M4, close to the edge-only path.

## Why not stack the other algorithms?

Two other algorithmic changes were measured separately and are **not** in
`--nightly`, because they do not compose with the bottom-edge path on this
clip:

| Composition | Point | CVVDP |
|---|---|---:|
| `quality + edge` | 100.892 kbps | 8.6892 |
| `quality + edge + ccso-rd` | 100.248 kbps | **8.7465** |
| `quality + edge + tx-context` | 88.544 kbps | 8.4875 (iso-rate ≈8.57) |
| `quality + edge + partition-rd` | 101.372 kbps | 8.3811 |
| `quality + edge + partition-rd + tx-context + ccso-rd` | 96.612 kbps | 8.2522 |

`--video-partition-rd` alone is a large win over the old tree (+2.05 JOD at
iso-rate, −25% rate at iso-CVVDP; see `docs/video-partition-rd.md`), and
`--video-tx-context-rate` alone is ≈−1.75% rate at iso-CVVDP; both remain
available as explicit opt-ins. They are not in the nightly aggregate because
the measured combination is worse than edge + CCSO. This is an algorithmic
selection result, not a claim that either model is wrong on other content or
on a tree without the edge path.

## Verification

- `dav2d-macos --strict 1` decodes the nightly stream.
- 60-frame `--video-recon` payload equals the decoded payload byte for byte;
  the hidden padded rows are outside the Y4M payload.
- Default 480x240 QP189 stays **25,009 B / `a1d0dba1...`**.
- `quality` QP190 stays **25,141 B / `3984677b...`**.
- `tools/gate_video_nightly.sh` checks the expansion, both pins, strict decode,
  visible recon equality and the GPU/`--still` rejections.

## Caveats

This is one clip at one quality point on one host. The bottom-edge gain is
content-dependent (it is largest when the bottom band cannot be predicted
otherwise). The nightly aggregate is a CPU development mode, not the GPU
product path, and it is deliberately not the default.
