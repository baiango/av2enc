# tiles-svc bug scout

Lane: still/video tiles, tile_info, per-tile CDF/context reset, tile-group
assembly, avg-CDF with tiles, SVC layer headers/LCR/OPS and emission order.
Read-only; `build/av2enc`, `tools/bin/dav2d-macos`.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-TILES-SVC-01 | med | confirmed | SVC path silently ignores `--video-refs`, `--video-tlayers`, `--video-tlayer-drop` | see 01 | exit 0, IVF md5 identical to the baseline for every flag; plain `--video` honours `--video-refs` and rejects a 3-frame `--video-tlayers 4` | `src/main.cpp:2060-2100`, `src/main.cpp:2820-2840` | new |
| SCOUT-TILES-SVC-02 | med | confirmed | `level_tile_limits()` uses width scale 4 for levels 7.0-7.3 (tier 0) instead of the table's 8, so `minLog2TileCols` is one too high | see 02 | conformance table says `(14,0)={width 8, area 8}`; planner at `(8192x512, lvl 14, tier 0)` rejects 1x1 and accepts 2x1 with `min_log2_tile_cols=1`, while tier 1 (same level) gives 0 | `src/pipeline/tiles/tile_layout.cpp:117`, `src/core/conformance/conformance_tables.cpp:156-181` | new |
| SCOUT-TILES-SVC-03 | med | doc | tile planner never enforces level `MaxTiles`/`MaxTileCols`; the still writer emits a 16x2 level-5.1 stream the repo's own conformance validator rejects | see 03 | `plan_tile_layout(1024,128,16,2,64,7,0).ok=1`; conformance `validate()` returns `A.4.MaxTileCols: TileCols = 16 > MaxTileCols 8 (level 5.1, Table A.9)`; CLI encodes `t16.ivf` (16 columns, seq_level_idx 7) and dav2d decodes it | `src/pipeline/tiles/tile_layout.cpp:288-296`, `src/core/conformance/conformance_tables.cpp:156-181`, `src/pipeline/tiles/stream/gate_cli_tiles.sh:200`, `src/core/conformance/conformance_test.cpp:304-322` | new |

Covered elsewhere, not re-reported: SCOUT-HM-01 (tiled `--video-timing` drops
`decoder_model_info`), SCOUT-HM-02 (tiled `--video-mfh` emits an unreferenced
MFH), SCOUT-HM-04 (SVC accepts `--video-timing`/`--video-mfh`), SCOUT-RC-04
(SVC accepts `--crf`/`--bitrate`).

## Details

### SCOUT-TILES-SVC-01 — SVC silently ignores refs/temporal-layer flags

Repro (`in3.y4m` = three copies of `testdata/shine_640x360_1f.y4m`; any
multi-frame 640x360 input works):
```sh
python3 - <<'PY'
s=open('/Users/arin/work/av2enc/testdata/shine_640x360_1f.y4m','rb').read()
n=s.index(b"\n"); p=s.index(b"\n",n+1)+1
open('/tmp/in3.y4m','wb').write(s[:n+1]+(s[n+1:p]+s[p:])*3)
PY
B=/Users/arin/work/av2enc/build/av2enc
$B /tmp/in3.y4m -o /tmp/a.ivf --video --frames 3 --svc-layers x0=320x120,x1=640x360
$B /tmp/in3.y4m -o /tmp/b.ivf --video --frames 3 --svc-layers x0=320x120,x1=640x360 --video-refs 2
$B /tmp/in3.y4m -o /tmp/c.ivf --video --frames 3 --svc-layers x0=320x120,x1=640x360 --video-tlayers 4
$B /tmp/in3.y4m -o /tmp/d.ivf --video --frames 3 --svc-layers x0=320x120,x1=640x360 --video-tlayers 4 --video-tlayer-drop 0
md5 -q /tmp/a.ivf /tmp/b.ivf /tmp/c.ivf /tmp/d.ivf
```
Observed: all four are `def960bef854b38b70c468246cdc243e`; every command exits 0.
Plain `--video --video-refs 2` changes the stream (`845a2916…`), and plain
`--video --video-tlayers 4` with 3 frames exits 1 ("pass exactly 4 input
frames"). So on the SVC path the three flags are accepted and have no effect.
Expected: reject with a named error (as is already done for frame-type modes),
or apply them.
Likely cause: `run_svc_encode()` rebuilds `svc.video` from a hand-picked field
list that hard-codes `num_total_refs = 1` and never copies
`temporal_layers`/`tlayer_drop` (`main.cpp:2820-2840`), while the `svc_given`
guard does not list these flags (`main.cpp:2060-2100`). `--video-timing` and
`--video-mfh` are the same class of bug but are already covered by SCOUT-HM-04.

### SCOUT-TILES-SVC-02 — wrong Tile_Width_Scaling_Factor at levels 7.0-7.3

Repro (probe links the repo source directly; writes only to /tmp):
```sh
cat > /tmp/p.cpp <<'EOF'
#include <cstdio>
#include "tile_layout.h"
#include "core/conformance/conformance_tables.h"
using namespace av2enc;
int main(){
  auto s=conformance::tile_scale_factor(14,0);
  printf("table 7.0 tier0 width=%d area=%d\n",s.width,s.area);
  for(int c=1;c<=2;c*=2)
    for(int tier=0;tier<=1;tier++){
      auto a=tiles::plan_tile_layout(8192,512,c,1,64,14,tier);
      printf("8192x512 %dx1 tier%d ok=%d minlog2=%d maxwidth_sb=%d\n",c,tier,
             (int)a.ok(),a.layout.min_log2_tile_cols,a.layout.max_tile_width_sb);
    }
}
EOF
cd /Users/arin/work/av2enc
clang++ -std=c++17 -I src -I src/pipeline/tiles /tmp/p.cpp \
  src/pipeline/tiles/tile_layout.cpp src/core/conformance/conformance_tables.cpp -o /tmp/p
/tmp/p
```
Observed:
```
table 7.0 tier0 width=8 area=8
8192x512 1x1 tier0 ok=0 minlog2=0 maxwidth_sb=0
8192x512 1x1 tier1 ok=1 minlog2=0 maxwidth_sb=128
8192x512 2x1 tier0 ok=1 minlog2=1 maxwidth_sb=64
8192x512 2x1 tier1 ok=1 minlog2=0 maxwidth_sb=128
```
Expected: `level_tile_limits()` must match the §9.2 table: tier-0 width scaling
is 4 for levels 2.0-6.3 (idx 0-13) and 8 for 7.0-7.3 (idx 14-17), not
`(idx <= 17) ? 4 : 8`. Today levels 7.0-7.3 tier 0 use the level-≤6.3 bound:
`maxTileWidthSb=64` instead of 128, so `minLog2TileCols` is one too high.
Impact: a legal single-column layout is rejected; worse, an accepted layout
(2x1+) writes `tile_info()` increments starting at minLog2=1 while a conforming
decoder derives minLog2=0, so the increment bits and then
`tile_start_and_end_present_flag`/size fields are parsed at the wrong offsets —
a full frame desync. The CLI cannot currently signal these levels (still pins
`seq_level_idx=7`, video 7), so no stock CLI command reproduces the stream
desync; the planner/API behaviour above is reproduced.

### SCOUT-TILES-SVC-03 — level MaxTiles/MaxTileCols not enforced by the tile planner

Repro:
```sh
cat > /tmp/c.cpp <<'EOF'
#include <cstdio>
#include "core/conformance/conformance.h"
using namespace av2enc::conformance;
int main(){
  for(int cols : {8,16}){
    Config c; c.width=1024; c.height=128; c.sb_size=64;
    c.seq_level_idx=7; c.single_picture_header_flag=true; c.still_picture=true;
    c.tiles=TileGrid::uniform(1024,128,cols,2,64);
    Result r=validate(c);
    printf("%dx2 conformant=%d\n",cols,(int)r.conformant());
    for(auto& v:r.violations) printf("  %s: %s\n",v.check.c_str(),v.detail.c_str());
  }
}
EOF
cd /Users/arin/work/av2enc
clang++ -std=c++17 -I src /tmp/c.cpp src/core/conformance/conformance.cpp \
  src/core/conformance/conformance_tables.cpp src/core/conformance/known_plans.cpp -o /tmp/c
/tmp/c
# end-to-end (/tmp/in1024.y4m = any 1024x128 4:2:0 y4m):
./build/av2enc /tmp/in1024.y4m -o /tmp/t16.ivf --still --no-deblock --tiles 16x2
```
Observed:
```
8x2 conformant=1
16x2 conformant=0
  A.4.MaxTileCols: TileCols = 16 > MaxTileCols 8 (level 5.1, Table A.9)
```
and `t16.ivf` is emitted (seq header `seq_level_idx=7`, 16 tile columns) and
decodes with dav2d. The still gate even requires the same layout at 1080p
(`gate_cli_tiles.sh:200`).
Expected: `plan_tile_layout()` should reject a grid exceeding the level's
`MaxTiles`/`MaxTileCols` (Tables A.8/A.9), or the still tiled writer should
signal a high enough level. `validate_tile_layout()`'s header comment claims it
checks "level limits", but it only re-checks geometry.
Likely cause: `tile_layout.cpp` derives `max_log2_tile_cols/rows` only from
`sbCols/sbRows` and `kMaxTileColumns` (lines 288-296) and only models the §9.2
tile width/area scaling factors; it never consults
`conformance::LevelInfo::max_tiles/max_tile_cols`. The conformance test already
labels the 16x2 level-5.1 plan "the known violation"
(`conformance_test.cpp:304-322`), so this is a known but ungated gap.
