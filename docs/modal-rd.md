# RD fan-out on Modal

`tools/modal_rd.py` runs a matrix of av2enc encodes and scores each one with
CVVDP, Butteraugli and SSIMULACRA2 on Modal. Every `(config, QP)` is an
independent container, so a leave-one-out ablation that serialises for an hour
on one Mac finishes in one job plus scheduling.

This is an **RD tool, not a speed tool**. The image builds the `dev` preset on
shared cloud CPUs; use the local `release-timing` build for any fps number.

## Split workflow (default)

Encoding is the embarrassingly parallel half; CVVDP on this Mac is not. The
normal flow is therefore **encode on Modal, score here**:

```sh
# 1. Fan out the encodes. Each returns its IVF bytes + MD5; nothing is scored.
python3 -m modal run tools/modal_rd.py --mode ablate --base drift \
    --encode-only --out-dir /tmp/modal-ablate

# 2. Decode and score locally with the pinned dav2d + MPS CVVDP + local libjxl.
python3 tools/score_ivf.py --manifest /tmp/modal-ablate/manifest.json \
    --parallel 3 --out /tmp/modal-ablate/results.json
```

No GPU is used anywhere. `tools/score_ivf.py` checks each file's MD5 against
the manifest before decoding, so a truncated download cannot silently become a
data point. This also keeps the cloud image small: encode-only needs build
tools and `dav2d`, not torch/CVVDP/libjxl.

The harness never scores. Every non-check run writes IVF files plus
`manifest.json`; `tools/score_ivf.py` is the only scorer. This keeps the cloud
image to build tools + `dav2d` + the source tree, and keeps the metric
implementation identical to every previously recorded number.

## Parity contract

The bitstream must not depend on the host. On x86 the pixel-kernel layer falls
back to the scalar reference, and the tree has been byte-identical between
clang/macOS and GCC/Linux since the portability work. The harness therefore
refuses to be trusted until it reproduces the five pins:

```sh
python3 -m modal run tools/modal_rd.py --mode check
```

It encodes `default`, `quality`, `nightly`, `explambda` and `720p` and compares
the repeated-stream MD5s (`a1d0dba1`, `3984677b`, `1b99127a`, `7f03eb6e`,
`c4ba2813`). A matrix run that does not follow a passing check is not a
measurement.

## Modes

```sh
# 1. Parity gate (do this first; the image build is cached afterwards).
python3 -m modal run tools/modal_rd.py --mode check

# 2. Leave-one-out at the tuned-drift point: baseline + one arm per
#    --video-*-off switch, QP 109/110/111, K interpolated at 100 kbps.
python3 -m modal run tools/modal_rd.py --mode ablate --base drift

# 3. Add-on arms at the plain nightly point.
python3 -m modal run tools/modal_rd.py --mode add --base nightly \
    --qps 105,106,107 --add --video-wiener,--video-chroma-lr \
    --out-dir /tmp/modal-add

# 4. Arbitrary matrix from JSON: [{"name", "flags", "qps", "clip"}].
python3 -m modal run tools/modal_rd.py --mode matrix --matrix /tmp/jobs.json \
    --out-dir /tmp/modal-matrix

# Print the job list without spending anything.
python3 -m modal run tools/modal_rd.py --mode ablate --dry-run
```

`--base drift` is `--nightly --video-drift-refresh-threshold 1.5
--video-drift-refresh-budget 3250`, the current 240p/100 kbps best
(**K 83.80**). Every table should carry its own baseline row; the interpolated
ΔK is computed against it, not against a number from another host.

## Image

| Component | Pin |
|---|---|
| encoder | the source tree at the current commit, `dev` preset |
| decoder | `dav2d` `f9dafbc`, static build (the same commit as `tools/bin/dav2d-macos`) |
| CVVDP | `cvvdp==0.5.7`, **CPU device** |
| torch | `2.5.1` CPU (`whl/cpu`) |
| Butteraugli / SSIMULACRA2 | libjxl `v0.11.1` tools, source-built if Debian's `libjxl-tools` lacks them |
| clips | the two NASA public-domain Y4Ms, baked in (no external media) |

The image rebuilds when the source tree changes; the build is cached by content
hash, so repeated runs with the same commit start workers immediately.

## Reading the output

The worker returns bytes, kbps, MD5, JOD, Butteraugli, SSIMULACRA2, K and the
encode wall time for every point. The entrypoint groups by config, prints the
QP bracket and the interpolated K at 100 kbps, and shows ΔK against the
baseline arm. `--out results.json` writes the raw rows.

Two rules carried over from the local protocol:

- **Bracket, don't single-point.** The 240p K-vs-QP curve ripples by ±1.6 K.
  Every arm needs the points on both sides of 100 kbps.
- **Metrics can differ across libjxl versions.** The image is pinned, but if
  the local Homebrew libjxl is a different release, absolute BA/SS2 can shift.
  Deltas measured entirely inside one Modal matrix are comparable; a Modal
  absolute number is not a replacement for the local pin numbers.

## Cost

CPU-only: 4 cores per job, **512 MiB**, at most **4** containers. The
reservation comes from measured peak RSS on the release build at 4 threads:
**50.5 MiB** for the 240p 60-frame nightly encode and **245.5 MiB** for the
720p pin recipe. 512 MiB is 2x the largest fixture; revisit it before adding a
larger clip. The harness is for
small matrices: a baseline plus a few arms, not a full 13-arm sweep. A matrix
over 24 jobs refuses to run unless `--allow-large` is passed. Encode-only jobs
are a few seconds each; the cost driver is container count, not job count. CVVDP is pinned to the CPU device, so every arm sees the same
implementation.

## Why the generator lives here

The ablation vocabulary is a single list at the top of the file
(`ABLATIONS`). Adding a switch, a new base preset or a new clip is one edit,
and every future matrix shares it. `--mode matrix` is the escape hatch for
experiments that do not fit the generator.
