# Modal RD encode, local decode

`rd_encode.py` runs the CPU encoder in a Modal container and scores the returned
IVF on this Mac with `tools/quality.py` (local `dav2d-macos` + local CVVDP/MPS).
The Modal image never needs a decoder.

## Why the split

- **Develop on the M4.** The RD loop edits decisions and re-encodes short clips;
  local LTO builds plus `--cpu-threads 8` are faster than a remote round trip.
- **Encode on Modal.** Use it for the confirmation run (all QPs, the 300-frame
  clip, or several candidate presets) when local CPU time would serialize the
  work.
- **Decode and score on the Mac.** `dav2d-macos` and CVVDP are pinned here. This
  keeps the decoded-pixel gate and the JOD numbers on the same host as every
  other benchmark.

## Usage

Baseline at the 100 kbps point:

```sh
python3 tools/modal/rd_encode.py --qps 189
```

A candidate from a worktree:

```sh
AV2ENC_MODAL_ROOT=/Users/arin/work/av2enc-wt/rd-quality \
python3 tools/modal/rd_encode.py \
  --qps 187,189,191 \
  --extra "--video-rd-preset quality" \
  --out-dir /tmp/rd-quality
```

The 10 s / 300-frame NASA clip:

```sh
AV2ENC_MODAL_CLIP=/tmp/nasa_480x240_30fps_300f.y4m \
python3 tools/modal/rd_encode.py --qps 189 --frames 300 --threads 8
```

Outputs go to `/tmp/av2enc-modal-rd` by default:

- `<tag>_q<QP>.ivf` — returned bitstream
- `<tag>_q<QP>.json` — full local quality report (CVVDP, PSNR, SSIM, decode time)
- `<tag>_summary.json` — all rows plus the source/clip/flags used

`tag` is `baseline` with no `--extra` flags and `candidate` otherwise.

## What runs where

| Step | Host |
|---|---|
| Source snapshot (`src/`, `CMakeLists`, `tests`, ...) | Mac |
| CMake configure/build (`AV2ENC_LTO=ON` by default) | Modal builder |
| 300-frame / multi-QP encode | Modal (`cpu=8`, `memory=2048`) |
| IVF transfer | Modal RPC result (bytes) |
| `dav2d-macos --strict 1` decode | Mac |
| CVVDP / PSNR / SSIM | Mac (`mps`) |

The default codegen is portable LTO, not `AV2ENC_CPU_FAST=ON`: Modal's builder
and runtime host can differ, and `-march=native` would make the binary
host-specific. Set `AV2ENC_MODAL_CPU_FAST=1` only after confirming both hosts
are the same CPU class.

## Cost

The image build is cached after the first run. A single 60-frame 480x240 encode
is a few seconds of `cpu=8`; the 300-frame clip measured 14.7 s. The dominant
first-run cost is the ~1 minute image build and source/clip upload, not the
encode.

## First-run check

The container request is `cpu=8`, `memory=2048`. The earlier smoke test proved
the encoder itself runs at Modal's minimum (`cpu=0.125`, `memory=128`) for a
64x64/1-frame still, but that is not a useful RD container. Use 8 vCPUs for the
video path.
