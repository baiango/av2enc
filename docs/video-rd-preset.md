# CPU video RD preset (`--video-rd-preset quality`)

`--video-rd-preset quality` expands to the opt-in CPU tool set that measured
best on the project's CPU quality gate (480x240, 30 fps, 60 frames, ~100 kbps).
It is **off by default**: without the flag no field is touched and every pinned
stream keeps its bytes. The preset is a pure CLI expansion, so each tool stays
individually available and the normal combination validation still runs.

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/x.ivf \
    --video --frames 60 --qp 190 --cpu-threads 8 --video-rd-preset quality
```

Expansion (exactly this list; `tools/gate_video_rd_preset.sh` asserts that the
preset stream is byte-identical to typing the flags out):

| Flag | What it adds |
|---|---|
| `--video-inter-residual` | codes the quantised inter residual instead of skip-only leaves |
| `--video-deblock` | §7.17 deblocking on every reconstructed frame (output + reference) |
| `--video-ccso` | §7.19 CCSO after CDEF (band offsets) |
| `--video-tx-search` | §5.18.8 TX_MODE_SELECT luma TX size/type/partition search |
| `--video-intra-full` | 61-candidate luma intra search for the keyframe and intra leaves |

It is CPU `--video` only (single tile, P-only chain); `--still` and
`--video-gpu` reject it. It does not change `--qp`, the TX lambda (256) or any
default.

## Measured RD

Host: Apple M4 (10 cores), `AV2ENC_CPU_FAST=ON`, `--cpu-threads 8`. Clip:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames. Quality: `tools/quality.py`
(CVVDP 0.5.7, `standard_fhd`, MPS) against the pinned `dav2d-macos` decoder.

Baseline (`a1d0dba1...`, 25,009 B) and the preset:

| QP | default kbps | default JOD | default PSNR-Y | preset kbps | preset JOD | preset PSNR-Y |
|---:|---:|---:|---:|---:|---:|---:|
| 185 | 116.40 | 4.9989 | 24.72 | 118.88 | 6.5770 | 28.28 |
| 187 | 108.16 | 4.8875 | 24.58 | 111.06 | 6.4665 | 27.95 |
| 189 | **100.04** | **4.8021** | **24.47** | 103.70 | 6.3657 | 27.70 |
| 190 | — | — | — | **100.56** | **6.3021** | **27.58** |
| 191 | 95.04 | 4.7050 | 24.35 | 101.40 | 6.2466 | 27.44 |
| 192 | — | — | — | 98.42 | 6.1832 | 27.31 |
| 193 | 89.88 | 4.5918 | 24.24 | 96.16 | 6.1116 | 27.18 |
| 195 | 82.59 | 4.4809 | 24.10 | 86.91 | 5.9878 | 26.85 |

At the operating point:

| Stream | Rate | CVVDP | PSNR-Y |
|---|---:|---:|---:|
| default, QP 189 | 100.04 kbps | 4.8021 | 24.47 dB |
| preset, QP 190 (nearest measured) | 100.56 kbps (+0.5%) | 6.3021 | 27.58 dB |
| preset, interpolated to 100.04 kbps (QP 190/192) | 100.04 kbps | 6.273 | 27.51 dB |
| **delta (interpolated)** | matched | **+1.47 JOD** | **+3.04 dB** |

Stage-by-stage at ~100 kbps (linear interpolation between the two bracketing
measured QPs, same clip):

| Tool set (incremental) | kbps | JOD | PSNR-Y | encode time |
|---|---:|---:|---:|---:|
| default | 100.04 | 4.802 | 24.47 | 1.0 s |
| `--video-inter-residual` (measured at 98.9) | 98.91 | 4.887 | 24.54 | 1.1 s |
| + `--video-deblock --video-ccso` | 100.04 | 5.909 | 25.37 | 1.5 s |
| + `--video-cdef --video-gdf` | 100.04 | 5.963 | 25.40 | 14.4 s |
| + `--video-tx-search` (on deblock+ccso) | 100.04 | 6.163 | 27.38 | 1.9 s |
| + `--video-intra-full` (= preset) | 100.04 | 6.273¹ | 27.51¹ | 2.6 s |

¹ measured QP 190 (100.56 kbps) / interpolated, see the table above.

Encode time (interleaved, best of 3, 60 frames): default 0.97 s, preset 2.57 s
(2.6x, ~23 fps vs ~62 fps). Almost all of the extra time is the TX search.

## Verification

```sh
BIN=build-rd/av2enc
DAV2D=/Users/arin/work/av2enc/tools/bin/dav2d-macos
$BIN testdata/nasa_480x240_30fps_60f.y4m -o /tmp/p.ivf --video --frames 60 \
     --qp 190 --cpu-threads 8 --video-rd-preset quality --video-recon /tmp/p.y4m
tools/validate_dav2d.sh /tmp/p.ivf                 # OK, 60 frames
$DAV2D -q --strict 1 -i /tmp/p.ivf -o /tmp/p_dec.y4m --threads 1
# encoder recon payload == decoder payload, 10,368,000 B, byte for byte
tools/quality.py --ref testdata/nasa_480x240_30fps_60f.y4m --ivf /tmp/p.ivf \
     --json /tmp/p.json                            # 6.3021 JOD, 27.5816 dB
```

`tools/gate_video_rd_preset.sh` re-checks the expansion, the flag-less
480x240 default stream (25,009 B / `a1d0dba1...`), `--strict 1` decode, and the
whole-frame recon comparison.

## Rejected tools (same clip, ~100 kbps)

| Tool | Result | Why it is not in the preset |
|---|---|---|
| `--video-cdef --video-gdf` | +0.05 JOD | 10x encode time (14.4 s vs 1.5 s) for a small filter gain |
| `--video-tcq` | 6.223 JOD at 98.1 kbps | 145 s encode (60x) for ~+0.01 JOD |
| `--video-tx-lambda 128/512/1024` | 6.18/6.17/6.18 | default 256 is already on the plateau |
| `--video-subpel-eighth` | +0.02–0.03 JOD | real but marginal; leave it as an explicit add-on |
| `--video-per-segment-q` | ~+0.02 JOD | marginal, adds a rate knob (`--video-seg-delta`) |
| `--video-delta-q` | 6.161 | worse at matched rate and 13.7 s encode |
| `--video-cctx` | 6.216 | neutral |
| `--video-uv-delta-q {1,2,4}` | 6.187 at 2 | worse at matched rate |
| `--video-refs 2 --video-compound` | 6.138 | worse at matched rate |
| `--video-adaptive-mvd` | 6.160 | neutral |
| `--video-ccso-bands 5/6` | neutral / worse | no gain |
| `--video-pc-wiener`, `--video-wiener`, `--video-chroma-lr` | small / neutral / rate explosion | not worth the symbol cost at 100 kbps |
| `--video-gme/--video-warp`, `--video-bawp` | rejected by the CLI | gated on the plain skip-only P chain, incompatible with `--video-inter-residual` |
| `--video-opfl`, `--video-compound` | rejected / needs B frames | requires `--b-frames`, which this 60-frame P chain does not use |
| `--video-qm` | rejected | gated at QP 90 |

## Caveats

This is one clip at one quality point (240p30, ~100 kbps), measured on one
host. The preset is an RD preset, not a speed preset, and the per-tool gains do
not transfer as a fixed JOD number to other content or rates. The default
stream, the GPU calibration stream and every pinned baseline remain
byte-identical; the flag is the only way to opt in.
