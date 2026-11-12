# inter-advanced bug scout

Lane: advanced inter — compound/CWP/masks, warp/affine, GME, inter-intra,
BAWP, TIP, OPFL/refine-MV, temporal MV/motion field, temporal layers,
B-frames/output order.

Method: `build/av2enc --video` on 640x360/1280x720 2–12 frame clips with
`--video-recon`, decoded by `tools/bin/dav2d-macos --strict 1` and compared
sample-by-sample. Canonical controls (`--video`, `--video-refs 2/3`,
`--video-compound`, `--b-frames 1..6`, `--video-tlayers 4/8`,
`--video-tlayer-drop 0..2`, `--video-long-term`, `--video-olk`,
`--video-leading 1..7`, `--video-sef`, `--video-switch`, `--video-ras`, plus
`--tiles`, RC, MV-precision, `--video-inter-residual`, `--video-subpel-eighth`,
`--video-min-block 8`, `--video-max-depth 5..8`) all decoded **recon-exact**,
so the failures below are not baseline breakage. `--avg-cdf` alone is a
silent no-op on the long-term/OLK/leading/SWITCH/RAS/B-frame/tlayer wrappers
(byte-identical; `--video-sef` uses the plain base path and does carry it),
while `--avg-cdf --adaptive-cdf` makes those wrappers undecodable.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-INTER-ADV-01 | high | confirmed | `--avg-cdf --adaptive-cdf` with any wave-3/4 mode writes CDF-adapted tiles under headers that signal `disable_cdf_update = 1`; dav2d rejects even frame 0 | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/a.ivf --video --frames 3 --video-max-depth 3 --video-long-term --avg-cdf --adaptive-cdf` | encode exit 0; dav2d exit 1 `Error parsing frame header` / `Error decoding frame: Invalid argument`, `No data decoded` (also `--limit 1`). Reproduces for `--video-olk` (4 f), `--video-leading 1` (3 f), `--video-switch` (2 f), `--video-ras` (3 f), `--video-tlayers 4/8`, `--b-frames 2`; control `--video --avg-cdf --adaptive-cdf` decodes exact | `src/pipeline/video/video_frames.cpp:33-43,72-91,327-335,421-440,1195-1243,1426-1447`; `src/pipeline/video/video_frame.cpp:470-472 vs 1235/1434/1864,1869` | new |
| SCOUT-INTER-ADV-02 | med | confirmed | Wave-3/4 frame-count and order-hint checks run after the IVF header is written: exit 1 (docs say usage errors are 2) and the `-o` file is truncated to 32 bytes | `echo keep > /tmp/o.ivf; build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/o.ivf --video --frames 3 --video-olk` | exit 1; `/tmp/o.ivf` = 32-byte `DKIF…AV02`, previous content lost. Same for `--video-switch` ≠2 f, `--video-ras` ≠3 f, `--video-sef` <2 f, `--video-leading` <N+2 or `S+R>7` (`--video-leading 4 --frames 10`), `--video-tlayers` ≠4/8, `--b-frames` ≠N+2, `--video-tlayer-drop` > plan max (`--video-tlayers 4 --video-tlayer-drop 2`), and `--video-long-term --frames 17` (`order_hint does not fit OrderHintBits`, an undocumented 16-frame cap) | `src/main.cpp:3439,3475`; `src/pipeline/video/video_frames.cpp:171-175,279-300,468-471,548-551,660-663,1190-1193,1368-1372,1408-1418`; `src/codec/inter/longterm/longterm_pool.cpp:142` | new (class cf. SCOUT-RC-02 / BUG-INTER-001) |
| SCOUT-INTER-ADV-03 | med | confirmed | `--video-min-block 4` is accepted but the keyframe intra path cannot code 4-wide leaves: encodes abort with an internal error and a 32-byte output | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/m.ivf --video --frames 2 --video-min-block 4 --video-max-depth 8` | exit 1 `keyframe encode failed: intra DC prediction failed`, file 32 B; depths 4 OK, 5–8 fail; `--video-min-block 8 --video-max-depth 8` is recon-exact; parser accepts min-block `[4,64]` with no depth coupling | `src/main.cpp` (`--video-min-block` parser); `src/pipeline/video/video_frame.cpp:162-188,898-908` | new |
| SCOUT-INTER-ADV-04 | med | confirmed | `--rc crf/vbr` + `--video-olk/--video-leading/--video-ras`: accepted, then exits 1 mid-encode (`frame_base_q_idx` length vs sub-encode frame count) | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/a.ivf --video --frames 4 --video-max-depth 3 --video-olk --rc crf` | exit 1 `frame_base_q_idx must be empty or have one entry per frame`; 32-byte output | `src/main.cpp:2561-2589`; `src/pipeline/video/video_frames.cpp:182-189,309-325,552-559`; `src/pipeline/video/video_frame.cpp:1589-1592` | duplicate of SCOUT-RC-03 |
| SCOUT-INTER-ADV-05 | low | confirmed | `--b-frames`/`--video-tlayers` silently ignore `--video-timing`, `--video-mfh` and `--tiles` | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/a.ivf --video --frames 4 --video-tlayers 4 --video-timing` | exit 0; stream md5 identical to plain `--video-tlayers 4` (and to `--video-mfh`, `--tiles 2x2 --no-deblock`); same for `--b-frames 2` | `src/main.cpp:1762-1768`; `src/pipeline/video/video_frame.cpp:1643-1672` | duplicate of SCOUT-HM-03 (avg-cdf no-op folded into ADV-01) |
| SCOUT-INTER-ADV-06 | low | doc | Help says `--video-tlayer-drop` “implies --video-tlayers”, but it is rejected unless `--video-tlayers` is given | `build/av2enc testdata/shine_640x360_1f.y4m -o /tmp/a.ivf --video --frames 8 --video-tlayer-drop 2` | exit 2 `--video-tlayer-drop requires --video-tlayers` | `src/main.cpp:456-457` vs `1871-1897` | new |
| SCOUT-INTER-ADV-07 | low | doc | Help says `--video-compound` “requires --video-refs >= 2”, but refs 3 is rejected (exactly 2) | `build/av2enc testdata/shine_640x360_1f.y4m -o /tmp/a.ivf --video --frames 4 --video-compound --video-refs 3` | exit 2 `--video-compound needs --video-refs 2` | `src/main.cpp:458-459` vs `1836-1841` | new |
| SCOUT-INTER-ADV-08 | low | doc | Help says `--video-tlayers <N>` selects “N = 4 or 8 temporal layers”; N is the GOP/input-frame count (2 or 3 layer ids), and the video README's N=8 layer list `0 1 1 2 2 2 2` disagrees with the planner's `0 0 1 1 2 2 2 2` | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/a.ivf --video --frames 4 --video-tlayers 4` | encoder errors say “4 or 8 input frames”; `build_plan` sets `max_tlayer_id` 1 (N=4) / 2 (N=8) | `src/main.cpp:454-455`; `src/pipeline/video/README.md` tlayers section; `src/codec/inter/tlayers/tlayers.cpp:113-147` | new |

## Details

### SCOUT-INTER-ADV-01 — adaptive CDF flags are lost when the wave-3/4 wrapper rewrites the headers
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/a.ivf --video \
  --frames 3 --video-max-depth 3 --video-long-term --avg-cdf --adaptive-cdf
tools/bin/dav2d-macos -q -i /tmp/a.ivf -o /tmp/a.y4m --strict 1 --threads 1
```
Observed: encoder exits 0 and reports three frames; dav2d exits 1 with
`Error parsing frame header`, `Error parsing OBU data`,
`Error decoding frame: Invalid argument`, `No data decoded` (even with
`--limit 1`, so the keyframe itself is invalid). Reproduced identically for
`--video-olk`, `--video-leading 1`, `--video-switch`, `--video-ras`,
`--video-tlayers 4/8` and `--b-frames 2`. The same flags on the plain path
(`--video --avg-cdf --adaptive-cdf`) decode recon-exact.
Expected: either propagate `avg_cdf`/`adaptive_cdf` into the wave-3/4 sequence
and frame headers (the plain path proves dav2d handles them), or reject the
combination by name before encoding.
Likely cause: `encode_plain()` encodes the tile payloads through the normal
path, whose `InterFrameEncoder` uses `options_.adaptive_cdf` for the tile CDFs
(`video_frame.cpp:470-472`), but the emitted headers come from the
long-term/leading/random-access configs and the B/tlayer `VideoConfig`s, which
all hard-code `disable_cdf_update = true` (`video_frames.cpp:1234,1434`; the
component configs default it) and never set `enable_avg_cdf`
(`video_frames.cpp:72-91,327-335,421-440,1195-1243,1426-1447`). The plain path
sets both from the options (`video_frame.cpp:1864,1869`). Consequences: the
lt/leading/RA headers do not even have an `enable_avg_cdf` field, so
`--avg-cdf` alone is byte-identical to the base stream there; the B/tlayer
configs have the field but leave it false, and `--adaptive-cdf` still changes
the tile payload while the header says no adaptation.

### SCOUT-INTER-ADV-02 — mode/frame-count validation happens after `writer.open`
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/keep.ivf --video --frames 2
ls -l /tmp/keep.ivf                                  # valid stream
./build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/keep.ivf --video --frames 3 --video-olk
echo $?                                               # 1
ls -l /tmp/keep.ivf                                   # 32 bytes
```
Observed: `--video-olk` needs exactly 4 frames but the encoder only discovers
this inside `encode_olk_sequence`; `run_video_encode` is called after
`main.cpp:3439` opened (and truncated) the IVF. The same pattern covers
SWITCH/RAS/SEF/leading/tlayers/b-frames input counts, `--video-tlayer-drop`
above the plan maximum, and `--video-long-term` beyond 16 frames (the
`OrderHintBits = 4` limit is not documented). Exit is 1, while help says usage
and option-combination errors exit 2.
Expected: preflight these counts (and the long-term order-hint cap) next to the
other “usage errors reported before the output file is opened” checks
(`main.cpp:3402-3416`), returning 2 without touching the output.

### SCOUT-INTER-ADV-03 — `--video-min-block 4` cannot code 4-wide leaves
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/m.ivf --video \
  --frames 2 --video-min-block 4 --video-max-depth 8
```
Observed: exit 1 `keyframe encode failed: intra DC prediction failed`; `/tmp/m.ivf`
is 32 bytes. Depth 4 succeeds (its leaves never reach 4-wide/4-tall); depths
5–8 fail. `--video-min-block 8 --video-max-depth 8` is recon-exact. `--video-trace`
shows the last successful leaf (e.g. `32x16`) then the abort.
Expected: reject `--video-min-block < 8` (or couple it to the depth cap) before
encoding, or implement the sub-8 intra path; the still path uses
`--still-sub8` for this and the video CLI advertises no such limitation.
Likely cause: the keyframe intra predictor is called with coding-block size 8
(`video_frame.cpp:162-188`, `intra::predict_block(request, 8, ...)`), and the
4:2:0 chroma of a 4-wide/4-tall leaf is 2-wide.

### SCOUT-INTER-ADV-04 — duplicate
Same trigger and error as `SCOUT-RC-03` in `rate-control-cli.md`
(`--rc crf/vbr` + `--video-olk/--video-leading/--video-ras` → exit 1
`frame_base_q_idx must be empty or have one entry per frame`). Not re-reported.

### SCOUT-INTER-ADV-05 — duplicate
The b-frame/temporal-layer silent ignore of `--video-timing`, `--video-mfh`
and `--tiles` is `SCOUT-HM-03` in `headers-metadata.md`. Additional data point
for ADV-01: `--avg-cdf` alone is also byte-identical on these paths.

### SCOUT-INTER-ADV-06/07/08 — help text vs parser
All three are reachable with the command in the table; the parser errors are
shown there. ADV-08 additionally notes that the video README's layer audit
string for N=8 (`0 1 1 2 2 2 2`) is short one entry and disagrees with
`Plan.frames[].tlayer_id` (`0 0 1 1 2 2 2 2`).

## Not reachable from the product CLI
CWP, wedge/diffwtd masks, implicit mask, global motion/GME, warp/affine,
inter-intra, BAWP, TIP, MV trajectory, OPFL/refine-MV and the temporal motion
field are hard-disabled in the sequence header
(`src/pipeline/video/video_headers.cpp:203-232`,
`src/pipeline/video/post/tiled_frame.cpp:162-187`) and have no `av2enc` flag,
so no product-path desync can be reproduced for them. That matches
`docs/negative-results.md`; only `--video-compound` (GLOBAL_GLOBALMV +
COMPOUND_AVERAGE/CWP_EQUAL) is wired, and it is recon-exact with
`--video-refs 2`, quarter/half precision and `--video-inter-residual`.
