# headers-metadata bug scout

Lane: OBU/frame headers and metadata (segmentation/delta-Q, MFH, timing/decoder
model, metadata OBUs, content interpretation, sequence/frame field ordering,
SEF/OLK fields, SVC/OPS headers). Read-only; `build/av2enc` and the pinned
`tools/bin/dav2d-macos`.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-HM-01 | med | confirmed | `--video --tiles` silently drops the `--video-timing` sequence `decoder_model_info` block | `av2enc /tmp/hm256_8f.y4m -o t.ivf --video --p-frames 2 --tiles 2x2 --no-deblock --video-timing` | tiled sequence header is byte-identical with and without `--video-timing` (`8075002effffe0028006e23800000101ba0820`); only a CI OBU (type 24) is added. Plain `--video --video-timing` changes it to `8075002effffe8400000004014003711c…` | src/pipeline/video/post/tiled_frame.cpp:136-137; src/main.cpp:3149-3175 | new |
| SCOUT-HM-02 | med | confirmed | `--video --tiles --video-mfh` emits an MFH but every frame header still references MFH 0, so the MFH is dead weight | same fixture, add `--video-mfh` | MFH OBU type 3 body `c2` is inserted, but the CLK body is byte-identical to the no-MFH tiled stream (`f03a15a0…`); its first bits after `is_first_tile_group` are `1,1` = `uvlc(0)` for `cur_mfh_id` and `uvlc(0)` for `seq_header_id`. Plain `--video --video-mfh` writes `a818…` (`uvlc(cur_mfh_id=1)`) instead | src/pipeline/video/post/tiled_frame.cpp:413-414; src/main.cpp:3149-3175 | new |
| SCOUT-HM-03 | med | confirmed | Temporal-layer and B-frame modes accept `--video-timing`/`--video-mfh` (and `--tiles`) and silently ignore them | `av2enc /tmp/hm256_8f.y4m -o a.ivf --video --video-tlayers 8 --video-timing` | exit 0; md5 `8d15ac4cf1388097614f3297f4baad52`, identical to plain `--video-tlayers 8`, to `--video-mfh` and to `--tiles 2x2 --no-deblock`. Same for `--b-frames 2` on 4 frames (all four variants `1f82cd5f373a80ad962989e925dd9910`) | src/main.cpp:1762-1768 (guard only names long-term/OLK/leading/SEF/SWITCH/RAS); src/pipeline/video/video_frame.cpp:1643-1672 (special modes return before the tiled branch) | new |
| SCOUT-HM-04 | low | confirmed | SVC path accepts `--video-timing` and `--video-mfh` but emits neither timing info nor an MFH | `av2enc /tmp/hm64_8f.y4m -o s.ivf --video --p-frames 2 --svc-layers 0=64x64,1=64x64 --spatial-layers 1 --video-timing` | exit 0; md5 `6d26ae13b93de34b775e95beafdb6b1f`, identical to the same SVC encode without `--video-timing` and with `--video-mfh` | src/main.cpp:2060-2110 (SVC guard list omits both flags); src/main.cpp:2826-2895 sets no `svc.video` timing/MFH | new |

## Details

Fixtures used below (all from repo testdata; short, local):
```sh
cd /Users/arin/work/av2enc
ffmpeg -v error -f yuv4mpegpipe -i testdata/shine_1280x720_60f.y4m \
  -frames:v 8 -vf scale=256:256 -pix_fmt yuv420p -f yuv4mpegpipe /tmp/hm256_8f.y4m -y
ffmpeg -v error -f yuv4mpegpipe -i testdata/shine_1280x720_60f.y4m \
  -frames:v 8 -vf scale=64:64 -pix_fmt yuv420p -f yuv4mpegpipe /tmp/hm64_8f.y4m -y
ffmpeg -v error -f yuv4mpegpipe -i /tmp/hm64_8f.y4m -frames:v 4 \
  -f yuv4mpegpipe /tmp/hm64_4f.y4m -y
B=build/av2enc
```

### SCOUT-HM-01 — tiled video drops `decoder_model_info`
Repro:
```sh
$B /tmp/hm256_8f.y4m -o /tmp/t_base.ivf --video --p-frames 2 --tiles 2x2 --no-deblock
$B /tmp/hm256_8f.y4m -o /tmp/t_timing.ivf --video --p-frames 2 --tiles 2x2 --no-deblock --video-timing
$B /tmp/hm256_8f.y4m -o /tmp/p_timing.ivf --video --p-frames 2 --video-timing
```
Observed: both tiled first temporal units contain the same sequence-header OBU
(`8075002effffe0028006e23800000101ba0820`); `t_timing.ivf` only gains a type-24
content-interpretation OBU (`04 00000001 0000003c d0`). The plain
`p_timing.ivf` sequence header differs (`8075002effffe840…`), proving the flag
is normally coded. Both streams decode with `dav2d --strict 1` and pixels match.
Expected: the tiled sequence header carries `seq_initial_display_delay_present_flag`
and `decoder_model_info_present_flag = 1` + `num_units_in_decoding_tick`, as
`--help` states and as the non-tiled writer does.
Likely cause: `write_sequence_header()` in `post/tiled_frame.cpp` hardcodes both
flags to 0 and ignores `VideoConfig::initial_display_delay_present` /
`decoder_model_info_present`; the multi-tile guard in `main.cpp` never rejects
`--video-timing` with `--tiles`.

### SCOUT-HM-02 — tiled `--video-mfh` emits an unreferenced MFH
Repro:
```sh
$B /tmp/hm256_8f.y4m -o /tmp/t_mfh.ivf --video --p-frames 2 --tiles 2x2 --no-deblock --video-mfh
$B /tmp/hm256_8f.y4m -o /tmp/t_plain.ivf --video --p-frames 2 --tiles 2x2 --no-deblock
$B /tmp/hm256_8f.y4m -o /tmp/p_mfh.ivf --video --p-frames 2 --video-mfh
```
Observed: `t_mfh.ivf` adds one type-3 OBU (payload `c2`, i.e. `mfh_id=1`) but
all CLK/REGULAR_TILE_GROUP bodies are byte-identical to `t_plain.ivf`
(`f03a15a0…`). Their first bits after `is_first_tile_group` are `1,1`:
`cur_mfh_id = uvlc(0)` followed by `seq_header_id = uvlc(0)`. The non-tiled
`p_mfh.ivf` body starts `a818…` = `is_first=1, uvlc(1), immediate_output=1`,
i.e. it really references the MFH.
Expected: with `--video-mfh`, every frame header should code `cur_mfh_id = 1`
and omit `seq_header_id_in_frame_header`; the MFH should be referenced.
Likely cause: `write_frame_header()` in `post/tiled_frame.cpp` hardcodes
`put_uvlc(0); put_uvlc(0);` and does not consume `VideoConfig::cur_mfh_id`
(`main.cpp` sets it for `--video-mfh` in the non-tiled path only).

### SCOUT-HM-03 — special video modes silently ignore lane header flags
Repro:
```sh
for extra in "" "--video-timing" "--video-mfh" "--tiles 2x2 --no-deblock"; do
  $B /tmp/hm256_8f.y4m -o /tmp/tl.ivf --video --video-tlayers 8 $extra
  md5 -q /tmp/tl.ivf
done
for extra in "" "--video-timing" "--video-mfh" "--tiles 2x2 --no-deblock"; do
  $B /tmp/hm64_4f.y4m -o /tmp/bf.ivf --video --b-frames 2 $extra
  md5 -q /tmp/bf.ivf
done
```
Observed: all four temporal-layer outputs are
`8d15ac4cf1388097614f3297f4baad52`; all four B-frame outputs are
`1f82cd5f373a80ad962989e925dd9910`. Exit status is 0 in every case; the streams
decode (`--strict 1`). `--video-tlayers`/`--b-frames` build their own headers in
`encode_temporal_sequence()`/`encode_b_sequence()` before the tiled branch in
`encode_video_sequence()`, and neither sets `VideoConfig::cur_mfh_id` nor the
timing block.
Expected: these are unsupported combinations and should exit 2 naming both
flags (the project rule is to reject un-implemented flags instead of ignoring
them), or the selected modes should honor them.
Likely cause: the `--video-timing/--video-mfh` guard at `main.cpp:1762` only
lists the frame-type modes (long-term/OLK/leading/SEF/SWITCH/RAS); `--tiles`
has no guard for `--video-tlayers`/`--b-frames`.

### SCOUT-HM-04 — SVC accepts timing/MFH flags but emits neither
Repro:
```sh
$B /tmp/hm64_8f.y4m -o /tmp/s_base.ivf --video --p-frames 2 \
  --svc-layers 0=64x64,1=64x64 --spatial-layers 1
$B /tmp/hm64_8f.y4m -o /tmp/s_t.ivf --video --p-frames 2 \
  --svc-layers 0=64x64,1=64x64 --spatial-layers 1 --video-timing
$B /tmp/hm64_8f.y4m -o /tmp/s_m.ivf --video --p-frames 2 \
  --svc-layers 0=64x64,1=64x64 --spatial-layers 1 --video-mfh
```
Observed: all three files are byte-identical (md5
`6d26ae13b93de34b775e95beafdb6b1f`); no CI/timing OBU and no type-3 MFH OBU
appear. Exit 0.
Expected: reject both flags on `--svc-layers` (the SVC encoder emits its own
per-layer sequence headers/LCR/OPS and has no timing/MFH support) rather than
accepting them silently.
Likely cause: the `svc_given` guard block (`main.cpp:2060-2110`) checks
`--tiles`, `--rc`, frame-type modes, `--svc-temporal/embedded`, but not
`--video-timing` or `--video-mfh`; `run_svc_encode()` never reads them.
