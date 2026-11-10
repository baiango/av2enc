# AVM 1080p baseline

Exact **1920x1080**, 60 fps, 8-bit 4:2:0. Main sweep: 10 frames from
`testdata/shine_1920x1080_10f.y4m`. Optional steady-state check: 2 seconds
(120 frames) cut from the original 4K source at 10 s.

> **Host: Apple M4 Mac mini, 10 cores, 16 GiB, macOS 26.6.2.**
> AVM here is a **CPU encoder on the Mac**.
> **These are not RTX 3060 target-speed numbers.** The product target host is an
> RTX 3060 enumerating 8192 MiB (`sm_86`, `compat8g`); validate speed there with
> `tools/check_target_host.py`. Use this page only as a quality/bitrate anchor
> and as a same-machine AVM speed reference.

## Setup

| Item | Value |
|---|---|
| Encoder | AVM macOS arm64 nightly 2026-11-10, commit `cd7d8bc3db` |
| Encoder binary | `tools/bin/avm-macos/avmenc` (SHA-256 `74021f98…b5ce`) |
| AVM settings | `--end-usage=q --cpu-used=6 --threads=8`, fixed QP |
| Decoder | `dav2d` `f9dafbc` (`tools/bin/dav2d-macos`) |
| Metrics | `python3 tools/quality.py`, CVVDP 0.5.7, `standard_fhd`, MPS |
| Encode time | wall clock from `/usr/bin/time -p` (includes startup) |
| Media | IVF/Y4M/logs are ignored by `bench/avm_1080p/.gitignore`; JSON reports are kept |

Every stream decoded successfully with `dav2d` before it was scored.

## 10-frame sweep

Source: `testdata/shine_1920x1080_10f.y4m` — 10 frames at 60 fps (1/6 s).
Bitrate is normalized over 1/6 s, so the single keyframe dominates these rows.

| QP | IVF bytes | Bitrate (kbps) | PSNR-Y (dB) | PSNR avg (dB) | SSIM All | CVVDP (JOD) | Encode time | Encode fps | ms/frame | Decode fps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 110 | 68,879 | 3,306.2 | 51.265 | 52.188 | 0.997707 | 9.9084 | 28.69 s | 0.349 | 2,869 | 92.8 |
| 135 | 37,771 | 1,813.0 | 48.382 | 49.515 | 0.996070 | 9.8190 | 22.72 s | 0.440 | 2,272 | 98.5 |
| 160 | 20,743 | 995.7 | 45.497 | 46.751 | 0.992938 | 9.5485 | 18.02 s | 0.555 | 1,802 | 126.3 |
| 200 | 6,936 | 332.9 | 40.428 | 41.715 | 0.978825 | 8.3386 | 15.06 s | 0.664 | 1,506 | 106.2 |
| 235 | 706 | 33.9 | 30.605 | 31.915 | 0.924833 | 5.5824 | 14.19 s | 0.705 | 1,419 | 103.9 |

## 2-second check (120 frames)

`shine_1080p_2s.y4m` is a fresh 120-frame cut from the original 4K VP9 source at
10 s (same start as the 10-frame file), not a loop of the 10 frames.
QP 160 only.

| Case | IVF bytes | Bitrate (kbps) | PSNR-Y (dB) | PSNR avg (dB) | SSIM All | CVVDP (JOD) | Encode time | Encode fps | ms/frame | Decode fps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QP 160, 2 s / 120 f | 258,186 | 1,032.7 | 50.039 | 51.021 | 0.996547 | 9.7657 | 147.70 s | 0.812 | 1,231 | 172.2 |

Notes:

- The 10-frame QP 160 row is keyframe-dominated: the keyframe payload is
  20,315 B of the 20,743 B IVF; the other nine frames total 324 B (23–237 B
  each) because this shot is very static. The 2-second row is the better
  steady-state bitrate and CVVDP reference.
- AVM's internal bitrate summary for the 10-frame QP 160 run is 990.7 kbps;
  `quality.py` reports 995.7 kbps including IVF container overhead.
- QP 235 collapses to 706 B / 33.9 kbps / 5.58 JOD. Nearly everything is skipped;
  it is not a useful operating point for this content.
- This is easy, smooth TV-demo content. Do not generalize the absolute rates to
  hard 1080p footage.
- `shine_1080p_2s.y4m` is 356 MB. It is local-only, ignored by Git, and can be
  regenerated with the command below; delete it if disk space is tight.

## Best QP for the 1 fps-at-1080p discussion

**Recommendation: QP 160 as the primary 1 fps anchor, with QP 200 as the
low-rate fallback.**

Reasons (deltas use the 2-second QP 160 point as the steady-state reference):

- **QP 160 is the knee.** The 2-second point is 9.77 JOD at 1,033 kbps. QP 135
  buys only +0.05 JOD for about +76% bitrate; QP 110 buys +0.14 JOD for 3.2×
  the rate. The 10-frame QP 160 point is lower (9.55 JOD at 996 kbps) because
  it is keyframe-dominated.
- **QP 200 is the low-rate comparator.** 8.34 JOD at 333 kbps; it gives up
  ~1.4 JOD versus the 2-second QP 160 point for 3.1× lower rate. Use it only if
  a ~350 kbps ceiling matters more than quality.
- **QP 235 is out.** 5.58 JOD is a visible collapse for ~30× lower rate than
  QP 160, and it buys almost no extra speed.

Strictly maximizing CVVDP with no rate ceiling would pick QP 110 (9.91 JOD), but
it costs 3.2× the rate of QP 160 for +0.14 JOD. QP 160 is the better anchor for
a first 1 fps target.

Speed caveat: **no measured AVM point reaches 1 encode fps at 1080p on this M4.**
The 10-frame sweep spans 0.349–0.705 fps; the 2-second QP 160 clip reaches
0.812 fps. `score_tradeoff.py` with `--min-encode-fps 1.0 --source-fps 60`
therefore reports the budget infeasible (see `qbs_1fps_10f.json`). These Mac
timings cannot validate the ≥1 fps floor — only the 8192 MiB RTX 3060 can.
AVM is CPU-only on the Mac, so `peak_vram_mb` is not applicable here; it must be
reported from our encoder's 3060 run.

For the target discussion, aim for **QP 160-level quality (≈9.5–9.8 JOD) at
≥1 fps on the 3060**, with QP 200 (≈8.3 JOD / ≈333 kbps) as the fallback if rate
is the binding constraint. Keep the core tools; spend surplus speed on quality,
not on raising fps.

## Exact commands

### 10-frame sweep

```sh
cd $HOME/work/av2enc
mkdir -p bench/avm_1080p

for qp in 110 135 160 200 235; do
  /usr/bin/time -p tools/bin/avm-macos/avmenc --ivf \
    --end-usage=q --qp=$qp --cpu-used=6 --threads=8 --limit=10 \
    -o bench/avm_1080p/qp${qp}_1080p_10f.ivf \
    testdata/shine_1920x1080_10f.y4m \
    > bench/avm_1080p/qp${qp}_encode.log 2>&1
done
```

### Metrics

```sh
cd $HOME/work/av2enc
for qp in 110 135 160 200 235; do
  sec=$(awk '/^real /{print $2}' bench/avm_1080p/qp${qp}_encode.log)
  python3 tools/quality.py \
    --ref testdata/shine_1920x1080_10f.y4m \
    --ivf bench/avm_1080p/qp${qp}_1080p_10f.ivf \
    --json bench/avm_1080p/qp${qp}_1080p_10f.json \
    --encoder AVM --qp "$qp" --encode-seconds "$sec"
done
```

### 2-second clip

```sh
cd $HOME/work/av2enc
SRC="<media>/<source.webm>"

ffmpeg -hide_banner -v error -ss 10 -i "$SRC" -frames:v 120 \
  -vf scale=1920:1080 -pix_fmt yuv420p -an -f yuv4mpegpipe \
  bench/avm_1080p/shine_1080p_2s.y4m -y

/usr/bin/time -p tools/bin/avm-macos/avmenc --ivf \
  --end-usage=q --qp=160 --cpu-used=6 --threads=8 --limit=120 \
  -o bench/avm_1080p/qp160_1080p_2s.ivf \
  bench/avm_1080p/shine_1080p_2s.y4m \
  > bench/avm_1080p/qp160_2s_encode.log 2>&1

sec=$(awk '/^real /{print $2}' bench/avm_1080p/qp160_2s_encode.log)
python3 tools/quality.py \
  --ref bench/avm_1080p/shine_1080p_2s.y4m \
  --ivf bench/avm_1080p/qp160_1080p_2s.ivf \
  --json bench/avm_1080p/qp160_1080p_2s.json \
  --encoder AVM --qp 160 --encode-seconds "$sec"
```

### 1 fps budget check

```sh
cd $HOME/work/av2enc
python3 tools/score_tradeoff.py --reference AVM \
  --q-min 5.0 --r-max 4000 --min-encode-fps 1.0 --source-fps 60 \
  bench/avm_1080p/qp110_1080p_10f.json \
  bench/avm_1080p/qp135_1080p_10f.json \
  bench/avm_1080p/qp160_1080p_10f.json \
  bench/avm_1080p/qp200_1080p_10f.json \
  bench/avm_1080p/qp235_1080p_10f.json \
  --json bench/avm_1080p/qbs_1fps_10f.json
```

Result: `feasible: false` even with all five points inside the quality/rate
budget; infeasibility is purely the speed floor. The fastest measured point is
QP 235 at 0.705 fps, and its quality is only 5.58 JOD.

## Files

- `qp{110,135,160,200,235}_1080p_10f.json` — quality reports, 10-frame sweep.
- `qp160_1080p_2s.json` — quality report, 120-frame QP 160 check.
- `qbs_1fps_10f.json` — infeasible 1 fps budget check on the M4.
- `*.ivf`, `*.y4m`, `*.log` — local only, ignored by Git.
