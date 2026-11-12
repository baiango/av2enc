# rate-control-cli bug scout

Scope: CLI/validation + rate control (`--rc`/`--crf`/`--bitrate`, `--effort`,
`--cpu-used`, `--gpu-plan`, `--vram-cap`). Read-only; `build/av2enc` only.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-RC-01 | med | confirmed | `--rc vbr` discards its per-frame QP plan and codes every frame at one uniform QP; `--rc crf` is a byte-identical `--qp` alias | `av2enc t360.y4m -o a.ivf --video --frames 4 --rc vbr --bitrate 600` | frame QP `115 115 115 115`; frame bits `13528 22600 23016 22728`; `--rc crf --crf 32` == `--qp 130` (md5 `dfffd3de…`) | src/main.cpp:2305-2357, 2611-2650 | new |
| SCOUT-RC-02 | med | confirmed | `--rc` usage errors are raised after the IVF writer opens, so an exit-2 typo truncates an existing `-o` file | `echo SENTINEL > o.ivf; av2enc $T64 -o o.ivf --video --rc vbr` | exit 2; `o.ivf` becomes the 32-byte `DKIF…AV02` header (parse-level `--qp 200` leaves it intact) | src/main.cpp:3439-3475 | new |
| SCOUT-RC-03 | med | confirmed | `--rc crf/vbr` passes the parser with `--avg-cdf --adaptive-cdf` or `--video-leading/--video-olk/--video-ras`, then exits 1 mid-encode | `av2enc t360.y4m -o a.ivf --video --frames 6 --video-leading 2 --rc crf --crf 32` | exit 1 `frame_base_q_idx must be empty or have one entry per frame`; CDF case exit 1 `average-CDF/continuity requires a fixed base_q_idx` | src/main.cpp:1743-1938, 3423-3475; video_frame.cpp:1589-1592, 1808-1812; video_frames.cpp:45-51 | new (cf. BUG-INTER-001) |
| SCOUT-RC-04 | low | confirmed | `--crf` and `--bitrate` are accepted and silently ignored on the `--svc-layers` path | `av2enc $T64 -o s.ivf --video --svc-layers x0=64x64 --crf 20` | exit 0; byte-identical to the run without `--crf` (and with `--bitrate 9000`): md5 `0586c856…` | src/main.cpp:2064-2076, 3423-3429 | new |
| SCOUT-RC-05 | low | confirmed | `--video-sef` + `--rc vbr` divides payload by temporal units (5) instead of displayed frames (4), mis-reporting achieved bitrate | `av2enc t360.y4m -o a.ivf --video --frames 4 --video-sef --rc vbr --bitrate 600` | reports `491.232 kbps (-18.128%)`; 10234 B over 4/30 s is 614.0 kbps (+2.3%), the loop's own target | src/main.cpp:2709-2726 | new |
| SCOUT-RC-06 | low | confirmed | `--vram-cap 0` is documented as valid ("0 takes the largest … default 0") but rejected | `av2enc in.y4m --gpu-plan 1 --vram-cap 0` | exit 2 `--vram-cap requires a positive MiB value` | src/main.cpp:350-352, 1679-1686 | new |
| SCOUT-RC-07 | low | confirmed | `--effort-plan` is documented to print and exit 0 without input/output, but parse requires both | `av2enc --effort 100 --effort-plan`; `av2enc in.y4m --effort 100 --effort-plan` | exit 2 `no input file specified` / `no output file specified`, no plan printed | src/main.cpp:1706-1713, 2947-2955; docs/effort.md:187 | new |
| SCOUT-RC-08 | low | confirmed | `--gpu-plan` ignores an explicit `--tiles 1x1` (alias `1`) and plans the GPU-count default | `av2enc t360.y4m --gpu-plan 1 --tiles 1x1` | prints `layout: 4x2 (8 tiles)`; on `$T64` exits 2 `layout 4x2 rejected` although 1 tile fits | src/main.cpp:2399-2412 | new |
| SCOUT-RC-09 | low | confirmed | `--gpu-plan 3..8` has no layout fallback: the 16x2 default is refused on 640x360 although the same 16 tiles fit as 8x2 | `av2enc t360.y4m --gpu-plan 3` | exit 2 `layout 16x2 rejected: … yields 10x2 tiles`; `--gpu-plan 3 --tiles 8x2` exits 0 with 3 GPUs | src/main.cpp:2399-2412; dispatch/planner.cpp:294-302 | new |
| SCOUT-RC-10 | low | confirmed | `--still-sub8` silently raises `--still-depth 3` to 8 instead of rejecting the documented "needs a depth of at least 8" | `av2enc shine_640x360_1f.y4m -o a.ivf --still --still-sub8 --still-depth 3` | depth 3 and 8 with sub8 md5-identical (`8545d318…`); without sub8 depth 3 vs 8 differ (`bdfa39f4…`/`c9fdcabd…`) | src/main.cpp:393, 2119-2136; still_encoder.cpp:449-453 | new |

## Details

Fixtures: `B=/Users/arin/work/av2enc/build/av2enc`, `T64=/Users/arin/work/av2enc/testdata/m0_64x64_1f.y4m`,
`S=/Users/arin/work/av2enc/testdata/shine_640x360_1f.y4m`, `/tmp/t360.y4m`=generated 640x360 8f 30fps clip
(`YUV4MPEG2 W640 H360 F30:1 Ip A1:1 C420jpeg`, per frame `FRAME` + varying luma + neutral chroma).

### SCOUT-RC-01 — per-frame rate control is one uniform QP
Repro:
```sh
$B /tmp/t360.y4m -o /tmp/a.ivf --video --frames 4 --rc vbr --bitrate 600 | tail -3
$B /tmp/t360.y4m -o /tmp/c.ivf --video --frames 3 --rc crf --crf 32 >/dev/null
$B /tmp/t360.y4m -o /tmp/q.ivf --video --frames 3 --qp 130 >/dev/null; cmp -s /tmp/c.ivf /tmp/q.ivf && echo CRF_IS_QP
```
Observed: VBR prints `frame QP: 115 115 115 115` although per-frame payloads span
13528..23016 bits (1.7x); `CRF_IS_QP` prints (`--crf 32` == `--qp 130`). Expected:
`--help` calls `--rc cqp|crf|vbr` "Per-frame rate control" and says VBR "plans one
base_q_idx per frame".
Likely cause: `plan_video_rate_control()` builds a per-frame vector, but the VBR
loop reduces it to one seed (`main.cpp:2605-2611`) and overwrites it with
`frame_qp.assign(inputs.size(), probe)` (`2617`, `2650`); CRF's complexity term
`CrfParams::qcomp` is fixed at 0 with no CLI knob (`codec/rate/rc/rc.h:345-348`),
so `--qp` is also silently ignored under `--rc crf`.

### SCOUT-RC-02 — usage error truncates the output file
Repro:
```sh
cd /tmp; echo SENTINEL > o.ivf
$B $T64 -o o.ivf --video --rc vbr; echo "exit=$?"; xxd -l 8 o.ivf
```
Observed: exit 2 and `o.ivf` becomes the 32-byte `DKIF….AV02` header (same for
`--rc cqp --crf 20` / `--bitrate 500`); a Basic `--qp 200` parse error leaves the
sentinel intact. Expected: exit 2 without touching `-o` (`main.cpp:3402-3403`).
Likely cause: the three rc combination checks sit at `main.cpp:3462-3474`, after
`writer.open()` at 3439.

### SCOUT-RC-03 — missing guards: rc + CDF chain / frame-type modes
Repro:
```sh
$B /tmp/t360.y4m -o a.ivf --video --frames 6 --video-leading 2 --rc crf --crf 32; echo $?
$B /tmp/t360.y4m -o b.ivf --video --frames 3 --video-olk --rc vbr --bitrate 600; echo $?
$B $T64 -o c.ivf --video --avg-cdf --adaptive-cdf --rc crf --crf 32; echo $?
```
Observed: all exit 1 with internal messages (`frame_base_q_idx must be empty or
have one entry per frame`; `average-CDF/continuity requires a fixed base_q_idx`);
VBR burns the statistics pass first; `--video-ras` fails the same way, while
`--video-switch`/`--video-sef` work only because their plain encode gets the full
span. Expected: option-combination errors exit 2 naming both flags (README:135),
the class BUG-INTER-001 fixed for tiles+avg-CDF.
Likely cause: no `--rc` term in the CDF-chain / frame-type validation
(`main.cpp:1743-1938`); `encode_plain(frames.subspan(...))` forwards a
`frame_base_q_idx` sized for the parent sequence, tripping the size check in
`video_frame.cpp:1589-1592`.

### SCOUT-RC-04 — `--crf`/`--bitrate` ignored on `--svc-layers`
Repro:
```sh
for f in "" "--crf 20" "--bitrate 9000"; do
  $B $T64 -o /tmp/s.ivf --video --svc-layers x0=64x64 $f >/dev/null; md5 -q /tmp/s.ivf
done   # bash; each line prints the same md5
```
Observed: exit 0 and the same md5 `0586c85676d07e470a2907aaf659d4f6` all three
times, no warning (`--rc crf` itself is rejected by name). Expected: reject by
name; the CLI contract is "rejected by name, never silently dropped".
Likely cause: svc returns at `main.cpp:3429` before the rc checks at 3462-3474, and
the svc validator only inspects `rc_mode`, not `rc_crf_given`/`rc_bitrate_given`
(`2064-2076`).

### SCOUT-RC-05 — SEF + VBR achieved-kbps denominator
Repro:
```sh
$B /tmp/t360.y4m -o a.ivf --video --frames 4 --video-sef --rc vbr --bitrate 600 | tail -3
```
Observed: `5 temporal unit(s), 5 coded frame(s), 4 output frame(s)`, `achieved
491.232 kbps (-18.128% error)`; 81872 payload bits over the 4-frame display
duration is 614.0 kbps (+2.3%), which is what the loop targeted. Expected: the
achieved/error duration match `target_bits`.
Likely cause: `duration` uses `result.units.size()` (5, includes the
show-existing-frame unit) at `main.cpp:2709-2712`, while the loop uses
`inputs.size()` (4) at 2598-2600.

### SCOUT-RC-06 — `--vram-cap 0` rejected
Repro:
```sh
$B /tmp/t360.y4m --gpu-plan 1 --vram-cap 0; echo $?
```
Observed: exit 2 `--vram-cap requires a positive MiB value`. Expected: `--help`
documents `0 takes the largest reservation the device can serve (default 0)`.
Likely cause: `main.cpp:1683` rejects `vram_cap_mib == 0`, so the documented value
is unreachable (runtime only tests `!= 0`).

### SCOUT-RC-07 — `--effort-plan` requires input and output
Repro:
```sh
$B --effort 100 --effort-plan; echo $?                 # no input
$B $T64 --effort 100 --effort-plan; echo $?          # no -o
```
Observed: exit 2 `no input file specified` / `no output file specified`, usage
banner, no plan; with both it exits 0 and writes nothing. Expected:
docs/effort.md:187 "prints the report and exits 0 without opening the input or
output".
Likely cause: generic required-input/output checks at `main.cpp:1706-1713` run
before the `effort_plan` early return at 2953.

### SCOUT-RC-08 — `--gpu-plan` ignores explicit `--tiles 1x1`
Repro:
```sh
$B /tmp/t360.y4m --gpu-plan 1 --tiles 1x1 | head -3
$B $T64 --gpu-plan 1 --tiles 1x1; echo $?
```
Observed: `layout: 4x2 (8 tiles)` for the first (1x1 ignored); the 64x64 case
exits 2 `layout 4x2 rejected: tile column count exceeds the frame/level maximum`,
though one tile fits. Expected: the plan reads `--tiles`; 1x1 must not be
silently replaced.
Likely cause: `run_gpu_plan` cannot distinguish the default 1x1 from an explicit
`--tiles 1x1` and falls through to `default_layout_for_gpus()`
(`main.cpp:2399-2402`), ignoring `options.specified`.

### SCOUT-RC-09 — `--gpu-plan 3..8` cannot plan 640x360
Repro:
```sh
for n in 2 3 8; do $B /tmp/t360.y4m --gpu-plan $n >/dev/null; echo "$n -> $?"; done
$B /tmp/t360.y4m --gpu-plan 3 --tiles 8x2 >/dev/null; echo "8x2 -> $?"
```
Observed: 2 → 0; 3 and 8 → exit 2 `layout 16x2 rejected: … yields 10x2 tiles`;
with `--tiles 8x2` the 3-GPU plan exits 0 (16 tiles split 5/6/5). Expected:
`--gpu-plan <N>` should plan whenever an allowed layout (8x2) fits.
Likely cause: `main.cpp:2404-2412` pre-validates only the preferred preset;
`dispatch/planner.cpp:294-302` hard-codes 16x2 for N>2 instead of falling back.

### SCOUT-RC-10 — `--still-sub8` overrides `--still-depth`
Repro:
```sh
$B $S -o d3.ivf --still --still-depth 3 >/dev/null
$B $S -o d8.ivf --still --still-depth 8 >/dev/null
$B $S -o s3.ivf --still --still-sub8 --still-depth 3 >/dev/null
$B $S -o s8.ivf --still --still-sub8 --still-depth 8 >/dev/null
md5 -q d3.ivf d8.ivf s3.ivf s8.ivf
```
Observed: `bdfa39f4… c9fdcabd… 8545d318… 8545d318…` — with sub8, depth 3 and 8 are
identical, so `--still-depth 3` is discarded without a warning. Expected: help:393
says sub8 "needs a depth of at least 8"; tool dependencies are otherwise rejected
by name (exit 2), not silently promoted.
Likely cause: `still_encoder.cpp:449-453` forces
`max_depth = sub8 ? max(depth, 8) : depth`; `main.cpp:2119-2136` validates every
other sub8 constraint but not the depth. (Class doc/gap: either enforce or document
the promotion.)
