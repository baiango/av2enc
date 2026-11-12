# quant bug scout

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-QUANT-01 | med | confirmed | Still tool-combination errors exit 1 and leave a truncated IVF instead of exit-2 usage errors | `./build/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/rj.ivf --still --qp 90 --still-qm --still-tcq` | rc=1, `/tmp/rj.ivf` is 32 bytes; CLI reject list only covers qm+rdoq, dcaq+qm, dcaq+rdoq | `src/main.cpp:2014-2032`, `src/pipeline/still/still_encoder.cpp:3720-3802` | new |
| SCOUT-QUANT-02 | low | confirmed | `--rc crf` silently ignores `--qp` | `--video --frames 3 --rc crf --crf 40` vs `... --qp 200` | byte-identical IVF, both print `frame QP: 162 162 162` | `src/main.cpp:2330-2360`, `src/main.cpp:2516` | new |
| SCOUT-QUANT-03 | med | doc | `--rc vbr` discards its per-frame base_q_idx plan; final stream is frame-uniform | 4-frame flat/noise/flat/noise fixture, `--rc vbr --bitrate 300` | frame bits `64 20208 144 20208` but `frame QP: 184 184 184 184`; help says "plans one base_q_idx per frame" | `src/main.cpp:2588-2656`, `src/main.cpp:489` | new |
| SCOUT-QUANT-04 | low | confirmed | `still_recon_dump --qp` has no range check; wraps modulo 256 | `build/still_recon_dump -i tests/data/basic_640x360_1f.y4m -o /tmp/r300.y4m --still --qp 300` | succeeds; recon md5 identical to `--qp 44`; `--qp -1` equals `--qp 255` | `src/pipeline/still/stream/still_recon_dump.cpp:81-88,283` | new |
| SCOUT-QUANT-05 | low | doc | QM/TCQ still gates resolve a stale `build-m1/av2enc` first and fail 6/13 cases | `src/codec/transform/qm/still_qm_gate.sh` | 6 failed with `unknown option '--still'`; passes 6/6 with `AV2ENC_BIN=build/av2enc` | `src/codec/transform/qm/still_qm_gate.sh:34-38`, `src/codec/transform/tcq/still_tcq_gate.sh:33-37` | new |
| SCOUT-QUANT-06 | low | doc | 10-bit `--qp 0` rejection advises `--lossless`, which is itself rejected | `build/av2enc p10.y4m -o /tmp/x.ivf --still --bit-depth 10 --qp 0` | `--qp 0 requires --lossless`; that path is unreachable (`--lossless` needs 8-bit/no `--still`) | `src/main.cpp:1971-1975,3324`, `src/main.cpp:3360` | new |

## Details

### SCOUT-QUANT-01 — still combination validation gap leaves a 32-byte IVF
Repro:
```sh
cd /Users/arin/work/av2enc
for c in "--still-qm --still-tcq" "--still-qm --still-cctx" "--still-qm --still-tx-partition" \
         "--still-qm --still-per-segment-q" "--still-tcq --still-ist" \
         "--still-dc-ac-delta-q --still-cctx" "--still-sdp --still-cdef" \
         "--still-sdp --still-segmentation"; do
  rm -f /tmp/rj.ivf
  ./build/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/rj.ivf --still --qp 90 ${=c}
  echo "$c rc=$? size=$(stat -f%z /tmp/rj.ivf)"
done
```
Observed: every pair above exits **1** and leaves `/tmp/rj.ivf` at **32 bytes**
(an IVF header with 0 frames). Examples:
`av2enc: error: --still-tcq is not supported with --still-qm yet`,
`av2enc: error: --still-qm is not supported with --still-tx-partition yet`.
By contrast `--still-qm --still-rdoq` exits 2 with no file.

Expected: the CLI hand-rejects the still writer's unsupported pairs so they are
usage errors (exit 2, no output). The block that does this says exactly that:
"The CPU Still writer has its own tool-combination rules. Reject them at the
CLI so they exit 2 ... instead of surfacing as an encode failure after the
output file has been opened" (`src/main.cpp:2014`).

Likely cause: `main.cpp`'s reject list covers only
qm+rdoq / dcaq+qm / dcaq+rdoq / mrl+ibp / mrl+rdo_intra / pc-wiener+wiener,
while `still_encoder.cpp` rejects many more pairs (qm+cctx, qm+tcq,
qm+tx_partition, qm+per_segment_q, tcq+ist, dcaq+cctx, tx_partition+dip/ibp/mrl,
sdp+cdef/ccso/segmentation). The encoder-side errors therefore run after
`IvfWriter::open`, and `fail_encode` closes a header-only IVF instead of
removing it. A downstream gate that only checks file existence/size sees a
"written" output.

### SCOUT-QUANT-02 — `--rc crf` accepts and ignores `--qp`
Repro:
```sh
cd /Users/arin/work/av2enc
S=testdata/shine_1280x720_60f.y4m
./build/av2enc $S -o /tmp/crf_a.ivf --video --frames 3 --rc crf --crf 40
./build/av2enc $S -o /tmp/crf_b.ivf --video --frames 3 --rc crf --crf 40 --qp 200
md5 -q /tmp/crf_a.ivf /tmp/crf_b.ivf
```
Observed: identical md5 (`c7ecb05a…`) and both report `frame QP: 162 162 162`.
Expected: `--qp` is documented as base_q_idx; a value that cannot take effect
under CRF should be rejected (`--crf owns the quantizer`) or honoured as the
pass-1 anchor. Only VBR reads `options.qp` (`main.cpp:2350`); the CRF branch
never does.

### SCOUT-QUANT-03 — VBR's per-frame QP plan is thrown away
Repro (4-frame 128x128: flat / noise / flat / noise, so complexity differs 300x):
```sh
cd /Users/arin/work/av2enc
python3 - <<'PY'
import random; random.seed(7); W=H=128
def pl(n): return bytes(random.randrange(256) for _ in range(n))
flat=lambda v: bytes([v])*(W*H) + bytes([v])*(W//2*H//2)*2
noise=pl(W*H)+pl(W//2*H//2)+pl(W//2*H//2)
open('/tmp/varseq.y4m','wb').write(
  b'YUV4MPEG2 W128 H128 F30:1 Ip A1:1 C420\n' +
  b'FRAME\n'+flat(128)+b'FRAME\n'+noise+b'FRAME\n'+flat(80)+b'FRAME\n'+noise)
PY
./build/av2enc /tmp/varseq.y4m -o /tmp/vbr.ivf --video --frames 4 \
  --rc vbr --bitrate 300
```
Observed:
```
rc:      vbr target 300 kbps (closed-loop 7 pass(es)), achieved 304.68 kbps
frame QP: 184 184 184 184
frame bits: 64 20208 144 20208
```
`--rc crf --crf 40` on the same fixture prints `frame QP: 162 162 162 162`.
Expected per the CLI help (`main.cpp:489`, "VBR ... then plans one base_q_idx
per frame"): per-frame base_q_idx. The plan produced by
`plan_video_rate_control()` is only averaged into the first probe
(`main.cpp:2606-2611`); the closed loop then does
`frame_qp.assign(inputs.size(), probe)` and finally
`frame_qp.assign(inputs.size(), best_qp)` (`main.cpp:2617,2651`), so every frame
is coded at one qindex. (CRF is constant because `CrfParams::qcomp` defaults to
0 and is never set from the CLI — documented in `rc.h:345-349` but not in the
user-facing rate-control help, which calls CRF "per-frame rate control".)

### SCOUT-QUANT-04 — `still_recon_dump --qp` wraps instead of rejecting
Repro:
```sh
cd /Users/arin/work/av2enc
R=build/still_recon_dump; S=tests/data/basic_640x360_1f.y4m
$R -i $S -o /tmp/r300.y4m --still --qp 300; echo rc=$?
$R -i $S -o /tmp/r44.y4m  --still --qp 44
$R -i $S -o /tmp/rm1.y4m  --still --qp -1
$R -i $S -o /tmp/r255.y4m --still --qp 255
md5 -q /tmp/r300.y4m /tmp/r44.y4m /tmp/rm1.y4m /tmp/r255.y4m
```
Observed: all four commands exit 0; `r300 == r44` and `rm1 == r255`
(`2949fc02…` and `4e72b284…`).
Expected: the tool says it "mirrors the src/main.cpp option mapping"
(`still_recon_dump.cpp:4-5`); `main.cpp` rejects QP>255 with exit 2. Silent
`static_cast<std::uint8_t>(qp)` at line 283 turns QP 256..303 (the 10-bit range)
into 0..47, so a future 10-bit gate would compare recon from the wrong
quantizer instead of failing.

### SCOUT-QUANT-05 — quant gates pick a pre-rename binary
Repro:
```sh
cd /Users/arin/work/av2enc
src/codec/transform/qm/still_qm_gate.sh        # 6 failed, 0 passed
src/codec/transform/tcq/still_tcq_gate.sh      # 13 failed, 0 passed
AV2ENC_BIN=build/av2enc STILL_DUMP=build/still_dump src/codec/transform/qm/still_qm_gate.sh
```
Observed: the default run picks `build-m1/av2enc` (Nov 11, pre `--m1`→`--still`
rename) and every case fails with `unknown option '--still'`. Pointing
`AV2ENC_BIN` at `build/av2enc` gives 6/6 and 14/14 passes. Candidate order is
`build-int/av2enc`, `build-m1/av2enc`, `build/av2enc`
(`still_qm_gate.sh:34-38`, `still_tcq_gate.sh:33-37`), so any checkout with a
stale alternate build dir silently invalidates the QM/TCQ coverage.

### SCOUT-QUANT-06 — 10-bit QP 0 diagnostic is a dead end
Repro:
```sh
cd /Users/arin/work/av2enc
ffmpeg -v error -strict -1 -f yuv4mpegpipe -i testdata/shine_256x256_1f.y4m \
  -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe /tmp/p10.y4m
./build/av2enc /tmp/p10.y4m -o /tmp/x.ivf --still --bit-depth 10 --qp 0
./build/av2enc /tmp/p10.y4m -o /tmp/x.ivf --lossless --bit-depth 10
./build/av2enc /tmp/p10.y4m -o /tmp/x.ivf --still --bit-depth 10 --lossless
```
Observed: `--qp 0 requires --lossless`; then `--lossless --bit-depth 10` says
`--bit-depth/--chroma require --still`; and adding `--still` says
`--lossless ... do not combine it with --still`. Expected: a 10-bit QP-0
rejection that states lossless 10-bit is not implemented, since no CLI path
reaches it.

## Not reported (verified clean)
QP edges 1/2/90/91/140/141/190/191/255/303 across still (8-bit and 10-bit
4:2:0/4:2:2/4:4:4/4:0:0), video (cqp/crf/vbr, tiles, avg/adaptive CDF,
inter-residual), and multi-tile stills: encoder recon == dav2d in every case.
QM levels 0..14 at QPs 1/40/90/141/255, per-segment ALT_Q 0..64, neutral
segmentation/delta-Q, DC/AC delta-Q, RDOQ/TCQ/IST/TX-type crosses: all matched.
`quant.json`, `rc_tables.h`, deblock/wiener/pc-wiener qlookup, `qm_tables.h` vs
`qm_tables.json`, and the 90/140/190 qctx boundaries agree with the spec tables.
