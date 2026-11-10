# Quality metrics

The encoder is judged on bitrate and perceptual quality at matched settings.
The reference decoder is the gate; the metrics are the scoreboard.

## Tools

| Metric | Tool | Settings |
|---|---|---|
| PSNR | ffmpeg `psnr` filter | per-plane + average |
| SSIM | ffmpeg `ssim` filter | Y + All |
| CVVDP | ColorVideoVDP 0.5.7 | JOD, `standard_fhd`, MPS on Apple Silicon |
| Decode gate | dav2d f9dafbc | must decode every stream |

The Mac is used for AVM encoding and CVVDP scoring only. It is **not** an encoder
build target.

## One command

```sh
python3 tools/quality.py \
  --ref testdata/shine_1280x720_60f.y4m \
  --ivf bench/avm_baseline/qp160_720p_60f.ivf \
  --json bench/avm_baseline/qp160_720p_60f.json
```

`tools/quality.py`:

1. Decodes the IVF with `dav2d`.
2. Reads Y4M frames directly and normalizes the frame-rate header to the
   reference.
3. Truncates both to the common frame count.
4. Runs ffmpeg PSNR and SSIM.
5. Runs CVVDP and parses the `cvvdp=... [JOD]` line.
6. Reports IVF bytes, bitrate, and all metrics as JSON.

## The frame-rate trap

`dav2d` writes Y4M with `F32:1` regardless of the IVF frame rate. If you feed a
32 fps test and a 60 fps reference directly to ffmpeg, it duplicates frames to
match durations and the PSNR collapses (e.g. 35.98 dB instead of 49.62 dB).

Do not use `ffmpeg -r 60` to fix it: that also duplicates frames (60 → 113).
`tools/quality.py` rewrites only the header and keeps the original frame bytes.

## Reading CVVDP

- CVVDP reports **JOD** (Just-Objectionable-Difference).
- 10 = no visible difference; lower is worse; values can go below 0.
- A 1 JOD difference is meaningful across a population; do not compare JOD
  values computed with different display models or viewing conditions.
- Always report the version/display string:
  `ColorVideoVDP v0.5.7, 37.84 [pix/deg], Lpeak=200, ..., (standard_fhd)`.

## CVVDP-Speed score (CSS)

A raw `JOD/s` ratio is misleading: JOD is bounded and nonlinear. The useful
question is: **at the same CVVDP, how fast does each encoder run?**

For an encoder, collect operating points `(C_i, S_i, B_i)` where:

- `C_i` = decoded CVVDP in JOD
- `S_i` = encode speed in **realtime×** = `video_seconds / encode_seconds`
- `B_i` = bitrate in kbps

Interpolate `S(C)` and `B(C)` over CVVDP, then report:

- **CSS(C)** — encode speed at a target CVVDP, in realtime×.
- **Speedup(C)** — `S_encoder(C) / S_reference(C)`.
- **Rate ratio(C)** — `B_encoder(C) / B_reference(C)`.
- **CSS-AUC** — mean of `S(C)` over the common CVVDP range
  `[C_lo, C_hi]`, in realtime×. This is the single summary score.

CSS-AUC must be read together with the iso-quality bitrate. A fast encoder that
spends much more rate is not better. Use the Pareto view: at the same CVVDP, an
encoder is better only if it is both faster and no larger.

```sh
python3 tools/score_speed.py \
  --reference AVM \
  --targets 8.0,8.5,9.0,9.5 \
  bench/avm_baseline/*.json bench/ours/*.json \
  --json bench/speed_score.json --plot bench/speed_score.png
```

The script prints a markdown table and the CSS-AUC per encoder. It refuses to
score a target outside the common CVVDP range, so comparisons stay honest.

CSS is **hardware- and content-specific**. Compare encoders only on the same
machine, same source clip, same resolution/frame rate, and same decoder. It is a
benchmark score, not an intrinsic codec property. For cross-machine comparisons,
run a fixed reference encoder on each machine and report the ratio.

Every `quality.py` report records `host.cpu_model`, physical/logical cores, RAM,
OS, and GPU. `score_speed.py` and `score_tradeoff.py` refuse to compare reports
from different CPU models unless `--allow-mixed-hardware` is passed. Use
`docs/hardware.md` as the reference for the Mac and Vast.ai machines.

## QBS: Quality-Bitrate-Speed

CSS answers “how fast at a given quality,” but it still shows bitrate only as
context. The encoder decision is three-dimensional:

```text
maximize CVVDP
minimize bitrate
maximize encode speed
```

Use the Pareto front for the full trade-off, and a budgeted score for the
headline number:

```text
QBS = max CVVDP  subject to  bitrate <= R_max  and  speed >= S_min
```

This is deliberately constrained: speed cannot buy its way out of the rate
ceiling, and quality cannot buy its way out of the speed floor. The tool also
reports:

```text
min bitrate  subject to CVVDP >= Q_min and speed >= S_min
max speed    subject to CVVDP >= Q_min and bitrate <= R_max
```

```sh
python3 tools/score_tradeoff.py \
  --reference AVM \
  --q-min 9.0 --r-max 500 --s-min 0.02 \
  bench/avm_baseline/qp*.json bench/ours/*.json \
  --json bench/qbs.json
```

Current AVM examples on the M4:

| Budget | Feasible | QBS | Point |
|---|---:|---:|---|
| Q ≥ 9.0, R ≤ 500 kbps, S ≥ 0.02× | yes | **9.7221 JOD** | QP 185, 498.5 kbps, 0.0236× |
| Q ≥ 9.0, R ≤ 400 kbps, S ≥ 0.02× | yes | **9.3623 JOD** | QP 200, 368.6 kbps, 0.0291× |

The 400 kbps point is the initial bar for our encoder at 720p60. Speed is
compared against AVM rerun on the same machine; quality and bitrate are
machine-independent.

## QBS@1fps (1080p)

The product target is **≥1 encoded frame per second at 1080p**, not maximum
throughput. At that fixed speed, the score becomes:

```text
QBS@1fps = max CVVDP
           subject to  encode_fps >= 1.0 at 1080p
                       bitrate    <= R_max
```

Speed above 1 fps is not a win. It must be spent on quality: RDOQ, deeper
partitions, more modes, larger search, more references/lookahead within the
7.5 GB cap. If the encoder drops below 1 fps, remove those extras in reverse
order — never the core tools (partitions, full intra, transforms, deblock, CfL).

Report `encode_fps`, `quality_level`, `frame_time_ms`, and `peak_vram_mb` with
the QBS@1fps result. A 1080p60 clip at 1 fps is 1/60 realtime; a 1080p30 clip is
1/30 realtime.

```sh
python3 tools/score_tradeoff.py --reference AVM \
  --q-min 9.0 --r-max 500 --min-encode-fps 1.0 --source-fps 60 \
  bench/avm_baseline/qp*.json bench/ours/*.json
```




## Rules

- Compare at similar bitrate, not similar QP.
- Report PSNR, SSIM, and CVVDP together; PSNR alone rewards blur.
- Every result must come from a stream that `dav2d` decoded successfully.
- Pin the AVM version and `--cpu-used` setting when producing the baseline.
