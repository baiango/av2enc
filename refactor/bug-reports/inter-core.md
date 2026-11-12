# inter-core bug scout

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-INTER-CORE-01 | med | confirmed | `--video-min-block 4` + `--video-max-depth >=5` aborts in the keyframe (sub-8 / 4-tall leaves) | `build/av2enc testdata/shine_640x360_1f.y4m --video --frames 1 --video-max-depth 5 --video-min-block 4` | exit 1, `keyframe encode failed: intra DC prediction failed`; depth 4 passes | src/pipeline/video/video_frame.cpp, src/pipeline/video/video_syntax.cpp | new |
| SCOUT-INTER-CORE-02 | low | doc/gap | `--video-long-term` silently caps at 16 frames; 17+ fails after reading the input | `build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 17 --video-long-term` | exit 1, `order_hint does not fit OrderHintBits`; 16 frames exact, plain `--video` 24 frames exact | src/pipeline/video/video_frames.cpp, src/codec/inter/longterm/longterm_pool.cpp | new |
| SCOUT-INTER-CORE-03 | low | doc/gap | `--rc crf/vbr` + `--video-leading/--video-olk/--video-ras` accepted, then fails on `frame_base_q_idx` length | `build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 4 --video-leading 1 --rc crf --crf 30` | exit 1, `frame_base_q_idx must be empty or have one entry per frame`; long-term/SEF/SWITCH with the same RC pass | src/pipeline/video/video_frames.cpp, src/pipeline/video/video_frame.cpp | new |
| SCOUT-INTER-CORE-04 | low | doc/gap | inter-residual gate is vacuous on the product fixture: `--video-inter-residual` emits a byte-identical stream | `build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 4 --video-inter-residual` vs without | identical MD5 `e4cf72dd13ddc2c32c2f8683153b3c81`; 0 `residual=1` of 1847 inter candidates; gate still PASSes | src/pipeline/video/video_gate.cpp, src/pipeline/video/video_cli_gate.sh, tools/run_gates.sh | new |

## Details

### SCOUT-INTER-CORE-01 — sub-8 / 4-tall leaves abort the video keyframe
Repro:
```sh
cd /Users/arin/work/av2enc
build/av2enc testdata/shine_640x360_1f.y4m --video --frames 1 \
  --video-max-depth 5 --video-min-block 4 -o /tmp/sub8.ivf
```
Observed:
`av2enc: error: keyframe encode failed: intra DC prediction failed` (exit 1).
The same input with `--video-max-depth 4 --video-min-block 4` passes and decodes
exactly. The failure also reproduces on a generated 320x192 clip (multiple
of 64 in both dimensions), i.e. it is not a boundary geometry case.

A read-only probe of the same partition solver
(`partition::decide_frame`, depth 5, min block 4, 640x360) returns leaves
`32x4` and `4x32` (plus `8x16`, `16x8`, ...). `encode_leaf()` derives chroma as
`cw = w / 2; ch = h / 2` (video_frame.cpp:671-672), so `32x4` becomes a 16x2
chroma request; `intra::predict_block()` rejects any dimension < 4
(`kMinBlockDim = 4`), so `intra_dc_predict()` fails at video_frame.cpp:901/904.

Expected: reject `--video-min-block 4` with `--video-max-depth >= 5` up front,
or implement the shared-chroma/sub-8 leaf rules the syntax writer already uses.
Note the latent hazard: `write_block()` takes `cw/ch` from
`partition::chroma_plane_residual_size()` (video_syntax.cpp:2328) while the
encoder used `w/2, h/2` (e.g. `32x4` -> 16x4 in the writer vs 16x2 in the
producer; sub-8 luma -> 4x4 vs `w/2`). If the intra predictor were relaxed, the
per-leaf recon/levels would no longer match the decoder's shared-chroma walk
(the same class of desync fixed for `--still-sub8` in
docs/negative-results.md). Today the keyframe aborts first, so no desynced
stream is emitted.

Likely cause: the video path never implemented the §5.20.3.1 shared-chroma
ownership for sub-8 / extreme-aspect leaves; it assumes `w/2, h/2` chroma for
every leaf.

### SCOUT-INTER-CORE-02 — long-term mode order-hint limit (16 frames)
Repro:
```sh
cd /Users/arin/work/av2enc
# 60-frame 720p fixture; keep --frames <= 16 for long-term
build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 16 \
  --video-long-term -o /tmp/lt16.ivf   # passes, recon exact
build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 17 \
  --video-long-term -o /tmp/lt17.ivf   # fails
```
Observed: frame 17 fails with
`av2enc: error: order_hint does not fit OrderHintBits` (exit 1, after encoding
the earlier frames).
`encode_long_term_closed()` passes the raw frame index as `order_hint`
(video_frames.cpp:140) and `long_term_config()` pins `order_hint_bits = 4`
(16 hints, 0..15). The pool validates `order_hint < (1 << bits)`
(longterm_pool.cpp:141). The plain `--video` loop masks with `& 15`
(video_frame.cpp, `order_hint = i & ((1u << 4) - 1)`), so a 24-frame plain
stream passes.

Expected: either document/validate `--video-long-term` as at most 16 coded
frames, or mask/wrap the long-term order hints like the plain path (16-frame
key-frame refresh means hints repeat after 16).

### SCOUT-INTER-CORE-03 — RC + open-GOP frame-type modes rejected late
Repro:
```sh
cd /Users/arin/work/av2enc
build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 4 \
  --video-leading 1 --rc crf --crf 30 -o /tmp/leadcrf.ivf
build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 4 \
  --video-olk --rc crf --crf 30 -o /tmp/olkcrf.ivf
build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 3 \
  --video-ras --rc crf --crf 30 -o /tmp/rascrf.ivf
```
Observed: all exit 1 with
`frame_base_q_idx must be empty or have one entry per frame` (crf) or
`--rc vbr pass 1: frame_base_q_idx must be empty or have one entry per frame`
(vbr). `--video-long-term`, `--video-sef` and `--video-switch` with the same
RC flags pass and decode exactly.

Cause: `encode_leading_sequence()` / `encode_olk_sequence()` /
`encode_ras_sequence()` call `encode_plain()` on `frames.subspan(...)` of size
1 or 2 (video_frames.cpp:182-184, 305-306, 556) while the CLI installed one
`frame_base_q_idx` entry per *input* frame; `encode_video_sequence()` then
rejects the size mismatch (video_frame.cpp:1591). The frame-type header writers
also freeze `config.base_q_idx` to the first frame's value, so a future
non-uniform per-frame QP would desync even after the length check is relaxed
(today CRF/VBR happen to be frame-uniform).

Expected: reject `--rc crf|vbr` with these modes up front, or slice the QP
vector for each sub-encode and propagate the per-frame QP into the synthetic
frame headers.

### SCOUT-INTER-CORE-04 — inter-residual gate does not exercise a residual
Repro:
```sh
cd /Users/arin/work/av2enc
build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 4 \
  --video-inter-residual -o /tmp/720res.ivf
build/av2enc testdata/shine_1280x720_60f.y4m --video --frames 4 \
  -o /tmp/720plain.ivf
md5 -q /tmp/720res.ivf /tmp/720plain.ivf
```
Observed: both streams are byte-identical
(`e4cf72dd13ddc2c32c2f8683153b3c81`); a `--video-trace` encode reports
`residual=1` on 0 of 1847 inter-candidate evaluations. The documented feature-1
gates (`build/video_gate --inter-residual` and
`src/pipeline/video/video_cli_gate.sh ... -- --video-inter-residual`) still
report PASS because they only compare recon and never assert that a
`skip_flag = 0` residual leaf was coded. On an illumination-shift clip the
same flag does code residual (39 leaves) and decodes exactly, so the syntax
path is fine; the gate is vacuous on the shipped fixtures.

Also note `tools/run_gates.sh:225` runs the video CLI gate with no inter
feature flags at all (default P chain only), so multi-reference, subpel,
precision, compound and inter-residual have no default CI coverage.

Expected: extend the gate to assert a minimum number of non-skip inter leaves
(or add a fixture that forces residual on), and include at least one
`--video-refs 2`/`--video-inter-residual` case in the default gate row.

## Dedup
`refactor/bugs.csv` only has `BUG-INTER-001` (multi-tile avg-CDF, unrelated);
`bug-reports/` had no other scout files for these issues at write time.
