# intra-tools bug scout

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-INTRA-01 | high | confirmed | MRL hardcodes `sb_height=64`, so `--still-mrl --still-sb128/256` desyncs below y=64 | `./build/av2enc testdata/src_100x70.y4m --still -o /tmp/a.ivf --recon /tmp/a.y4m --qp 90 --still-mrl --still-sb128` | recon/decoder differ in 88 Y samples, all rows 64..69; no-sb128 control is exact | `src/pipeline/still/still_encoder.cpp:1253` | new |
| SCOUT-INTRA-02 | high | confirmed | MRL prediction is wrong on 4-wide/4-tall luma leaves (`--still-sub8`, `--still-ext-partitions`) | `./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/b.ivf --recon /tmp/b.y4m --qp 90 --still-mrl --still-sub8 --still-depth 8` | first diff is in the 32x4 leaf at (32,8); Y/U/V differ 2341/501/504 samples. `--still-mrl --still-ext-partitions` first diff is the 4x16 leaf at (28,16) | `src/pipeline/still/still_encoder.cpp:1225-1260`, `src/codec/intra/mrl/core/mrl.cpp` | new |
| SCOUT-INTRA-03 | high | confirmed | MHCCP + shared-chroma partitions (`--still-sub8` / `--still-ext-partitions`) emits a stream dav2d rejects | `./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/c.ivf --qp 90 --still-mhccp --still-sub8 --still-depth 8 && tools/bin/dav2d-macos -q -i /tmp/c.ivf -o /tmp/c.y4m --threads 1` | `Error decoding frame: Invalid argument`; both flags alone are exact. Same with `--still-ext-partitions` | `src/pipeline/still/still_syntax.cpp:1382-1400`, `src/codec/intra/cfl/mhccp/mhccp_cdf_tables.h:68` | new |
| SCOUT-INTRA-04 | low | doc | `--no-cfl` does not validate CfL-only flags; `--still-cfl-explicit` is silently ignored | `./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/d.ivf --qp 90 --no-cfl --still-cfl-explicit` | byte-identical to `--no-cfl` alone (MD5 `e2dd563e...` both). `--still-mhccp --no-cfl` is rejected, so the class is expected to be validated | `src/main.cpp:2033-2036`, `src/pipeline/still/still_encoder.cpp:3824-3827` | new |
| SCOUT-INTRA-05 | low | doc | `--still-dc --still-mrl`: force-DC path bypasses the MRL search but still sets `enable_mrls=1` | `./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/e1.ivf --qp 90 --still-dc --still-mrl` | no `mrl_index>0` block is ever coded; the stream still differs from `--still-dc` (same 1061-byte size). Conflicting combination is accepted | `src/pipeline/still/still_encoder.cpp:1296-1316`, `src/pipeline/still/still_headers.cpp:144-150` | new |

## Details

### SCOUT-INTRA-01 — MRL `sb_height` is hardcoded to 64
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/src_100x70.y4m --still -o /tmp/a.ivf --recon /tmp/a.y4m --qp 90 --still-mrl --still-sb128
tools/bin/dav2d-macos -q -i /tmp/a.ivf -o /tmp/d.y4m --threads 1
python3 - <<'PY'
# per-plane diff counts; the Y4M headers differ (A1:1 vs A0:0), so skip them
def rd(p):
    d=open(p,'rb').read(); e=d.index(b'\n')
    h=d[:e].decode(); w=int(h.split('W')[1].split()[0]); g=int(h.split('H')[1].split()[0])
    return w,g,d[e+1:]
w,h,a=rd('/tmp/a.y4m'); _,_,b=rd('/tmp/d.y4m')
ys=w*h; cs=((w+1)//2)*((h+1)//2)
for n,x,y in (('Y',a[:ys],b[:ys]),('U',a[ys:ys+cs],b[ys:ys+cs]),('V',a[ys+cs:],b[ys+cs:])):
    print(n,sum(p!=q for p,q in zip(x,y)))
PY
```
Observed: dav2d decodes the stream but the encoder reconstruction differs from the decoder output. The per-plane counter above prints 88 Y and 1 U differing samples; every differing Y sample is on rows 64..69 (the first 64-row band below the false SB boundary). `--still-sb128` alone is sample-exact; 640x360 fails much more (41810 Y diffs).
Expected: §7.13.2.1 uses `sbHeight = Block_Height[SbSize]`, i.e. 128 (or 256 for the header) under `--still-sb128/256`; the encoder must use that SB height for `sbBoundary`/`aboveMrlIndex`.
Likely cause: `predict_luma_mrl()` sets `block.sb_height = 64` unconditionally (`still_encoder.cpp:1253`). `mrl::prepare_edges()` then treats every 64-row boundary as an SB boundary, resets the above reference line to line 0, while dav2d keeps using the MRL line until the 128-row boundary. `docs/m1-features.md` says `sb_top_y` was fixed for CfL, but the MRL descriptor was not.

### SCOUT-INTRA-02 — MRL is wrong on 4-wide/4-tall leaves
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/b.ivf --recon /tmp/b.y4m --qp 90 --still-mrl --still-sub8 --still-depth 8
tools/bin/dav2d-macos -q -i /tmp/b.ivf -o /tmp/bd.y4m --threads 1
# compare planes (ignore the Y4M header A1:1 vs A0:0): cmp reports Y 2341, U 501, V 504
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/b2.ivf --recon /tmp/b2.y4m --qp 90 --still-mrl --still-ext-partitions
```
Observed: both combinations encode and decode successfully, but the samples diverge. With `AV2ENC_STILL_TRACE=1`, the first differing sample for sub8 is inside the `size=32x4` leaf at x=32,y=8 (row 8 is entirely wrong from x=38 on), and for ext inside the `size=4x16` leaf at x=28,y=16. Adding `AV2ENC_STILL_SQUARE_ONLY=1` to the ext case makes luma exact (only chroma still differs), consistent with a 4-sample-axis prediction bug. `--still-sub8`/`--still-ext-partitions` alone are exact; `--no-deblock` does not change the result.
Expected: MRL must match dav2d on every legal leaf size it accepts (`mrl::valid_dim()` explicitly allows 4).
Likely cause: the MRL integration path (`predict_luma_mrl()` → `mrl::prepare_edges`) does not reproduce dav2d's edge/availability handling for leaves with a 4-sample axis. The combination is ungated: `still_encoder.cpp` only rejects MRL+IBP and MRL+rdo-intra, and `docs/m1-features.md` gates MRL only on the 8x8+ default leaves.

### SCOUT-INTRA-03 — MHCCP + sub8/ext produces an undecodable stream
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/c.ivf --qp 90 --still-mhccp --still-sub8 --still-depth 8
tools/bin/dav2d-macos -q -i /tmp/c.ivf -o /tmp/c.y4m --threads 1   # rc=1
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/c2.ivf --qp 90 --still-mhccp --still-ext-partitions
tools/bin/dav2d-macos -q -i /tmp/c2.ivf -o /tmp/c2.y4m --threads 1  # rc=1
```
Observed: `Error decoding frame: Invalid argument` / `No data decoded` for both. `--still-mhccp` alone, `--still-sub8` alone and `--still-ext-partitions` alone decode exactly at 64x64 and 640x360. `AV2ENC_STILL_UV_DC=1` (no CfL/MHCCP chroma) makes the sub8 stream decode, so the failure is in the CfL/MHCCP chroma syntax/decision, not the luma tree.
Expected: either the combinations are rejected by name (like `--still-mhccp --no-cfl`), or the emitted stream decodes and matches.
Likely cause: at shared-chroma leaves the chroma owner can be a 4-wide/tall leaf whose `MiSize != ChromaMiSize`. The writer passes `size_group_for_chroma_block(cw, ch)` to `write_cfl_alphas()` (`still_syntax.cpp:1382-1400`), but §8.3 selects `TileCflMhDirCdf[Size_Group[MiSize]]` (the *coding block* MiSize). A second issue in the same path is chroma ownership moving between leaves when the MHCCP chroma decision changes the partition search; the trace shows the 4x32 leaf at r=8,c=5 changing from chroma owner to non-owner when MHCCP is enabled, while the surrounding leaves are unchanged. Both need a gate before `--still-mhccp` is combined with ext/sub8.

### SCOUT-INTRA-04 — CfL-only flags are not validated against `--no-cfl`
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/d1.ivf --qp 90 --no-cfl
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/d2.ivf --qp 90 --no-cfl --still-cfl-explicit
md5 -q /tmp/d1.ivf /tmp/d2.ivf   # identical
```
Observed: byte-identical stream, exit 0. `--still-cfl-ds N --no-cfl` similarly only changes the sequence-level `cfl_ds_filter_index` field (legal on the wire but unused); `--still-mhccp --no-cfl` is rejected, so the inconsistency is specific to these two flags.
Expected: `--still-cfl-explicit`/`--still-cfl-ds` require CfL, like `--still-mhccp`; or they should be documented as no-ops with `--no-cfl`.
Likely cause: only the mhccp dependency is checked (`main.cpp:2033`, `still_encoder.cpp:3824`); `cfl_explicit` and `cfl_ds_filter` reach `StillOptions` and are never consumed when `enable_cfl == false` (`choose_uv_mode()` returns DC before the CfL block).

### SCOUT-INTRA-05 — `--still-dc --still-mrl` silently drops MRL
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/e1.ivf --qp 90 --still-dc
./build/av2enc testdata/m0_64x64_1f.y4m --still -o /tmp/e2.ivf --qp 90 --still-dc --still-mrl
md5 -q /tmp/e1.ivf /tmp/e2.ivf   # differ, same 1061-byte payload size
```
Observed: both encode and decode exactly and have the same payload size, but the streams differ; `enable_mrls` is set in the sequence header while no `mrl_index > 0` is ever coded. The force-DC branch returns before `choose_luma_mode()` runs the MRL candidate loop.
Expected: `--still-mrl` is documented as "implies the directional search"; an explicit `--still-dc` contradicts it. Reject the pair (like `--still-mrl --still-ibp`) or make `--still-dc` override MRL without setting `enable_mrls`.
Likely cause: `still_dc`/`force_dc` is checked before the `luma_dc_only`/MRL search in `choose_luma_mode()`; `enable_mrls` is still derived from `options.enable_mrl` in the header writer (`still_headers.cpp:144-150`).

## Dedup notes
- `refactor/bugs.csv` and `git log --grep` have no MRL/MHCCP/sub8/sb128 entries.
- `docs/negative-results.md:52` documents a *different* ext-partition failure (VERT_4A 4-wide 1/8 child root cause unknown); SCOUT-INTRA-02 is reported because MRL is the added tool and the first diff is in a 4x16/32x4 MRL leaf, not because of the documented VERT_4A trace alone.
- DC-subsampled intra and `dc_sign_horz_vert` are already `BLOCKED` in `docs/negative-results.md:47-48`; not re-reported.
