# intra-modes bug scout

Repo: `/Users/arin/work/av2enc` @ `527738a` (read-only). Lane: intra mode
decisions (full intra search, `--still-rdo-intra`, `--still-tx-type`, directional
modes, mode-context/CDF derivation, `--no-cfl`/`--still-uv-full`).

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-INTRA-01 | high | confirmed | `--still-sdp` + any directional luma mode (`--still-full-intra`/`--still-rdo-intra`/`--still-mrl`) produces a stream dav2d rejects | `av2enc f.y4m -o s.ivf --frames 1 --still --still-sdp --still-full-intra && dav2d -q -i s.ivf -o d.y4m --strict 1` | `Error decoding frame: Invalid argument` / `No data decoded`; SDP DC and forced SMOOTH/PAETH decode, forced V/H/D\* fail | still_encoder.cpp:2947,3069; still_syntax.cpp:1449-1460 | new |
| SCOUT-INTRA-02 | high | confirmed | Directional luma search + `--still-ext-partitions` or `--still-sub8` decodes but reconstruction is wrong (up to 210 LSB, 131k samples at 640x360) | see Details | sub8 64x64: 3367/6144 luma samples differ, max 77; ext 64x64: 3320 diff, max 93; 640x360 ext: 131412 diff, max 210; DC/smooth/paeth exact | still_encoder.cpp:947-1025; intra_modes.cpp:142-280 | duplicate (intra-tools.md SCOUT-INTRA-02) |
| SCOUT-INTRA-03 | low | doc/gap | `--still-depth` is accepted and silently ignored with `--still-sdp` | `av2enc f.y4m -o o.ivf --still --still-sdp --still-depth 5` | IVF md5 identical for depth 0/2/5/default (`e46f910f...`); no CLI rejection | main.cpp:2138-2165; still_encoder.cpp:487-489 | new |
| SCOUT-INTRA-04 | low | doc/gap | Partition/intra product gates only exercise DC luma, hiding 01/02 (gate hole) | `grep -n still-dc src/codec/partition/sub8/product_gate.sh` | sub8 gate uses `--still-sub8 --still-dc`; ext and SDP gates use default DC-only; docs mark all three PASS | sub8/product_gate.sh:111; ext/product_gate.sh:74; sdp/product_gate.sh; docs/m1-features.md:553-555 | new |

## Details

### SCOUT-INTRA-01 — SDP CHROMA_PART uses YMode=0 for UV mode syntax/CDF

Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc tests/data/basic_64x64_1f.y4m -o /tmp/sdp.ivf --frames 1 \
    --still --still-sdp --still-full-intra
./tools/bin/dav2d-macos -q -i /tmp/sdp.ivf -o /tmp/sdp.y4m --threads 1 --strict 1
```
Observed: `Error decoding frame: Invalid argument` / `No data decoded` (exit 1),
even without `--strict 1`. `AV2ENC_STILL_TRACE=1` shows the full search picks
`y=1` (V) for the 64x64 leaf, `uv=0` (DC); the byte count (763 B) equals the
forced `AV2ENC_STILL_FORCE_Y=1:0` stream and is rejected the same way.
`--still --still-sdp` alone and `FORCE_Y=9:0`/`12:0` (SMOOTH/PAETH) decode fine.
Only directional luma modes (V, H, D45..D203 and any angle delta) break, on both
the 64x64 and 640x360 fixtures. `--still-sdp --still-mrl` hits the same path.
Expected: decodes with exact recon (the SDP product gate requires that).
Likely cause: for a CHROMA_PART leaf the encoder has no luma mode and writes 0:
`still_encoder.cpp:3069` (`leaf.y_mode = do_luma ? y.av2_mode : 0`, and
`leaf.y_angle_delta` likewise) and `still_encoder.cpp:2947` passes the
default-constructed `y.av2_mode`/`y.angle_delta` (0/0) to `choose_uv_mode`.
`still_syntax.cpp:1449-1460` then derives both `ctx = is_directional(YMode)` for
`TileUVModeCflNotAllowedCdf` and `uv_mode_index(uv_mode, y_mode)` from that zero.
Spec §5.20.5.5 (`read_intra_y_mode`) says for `TreeType == CHROMA_PART`:
`YMode = YModes[MiRow][MiCol]; AngleDeltaY = AngleDeltaYs[MiRow][MiCol]`, i.e. the
decoder uses the co-located luma-tree mode. With a directional luma mode both the
UV CDF context and the mode index (and chroma `AngleDeltaUV`) differ, so the range
coder desynchronises. A correct leaf needs the luma mode/angle delta of the
`ChromaMi` origin (SDP forces both trees to 64x64 NONE, so it is the luma root).

### SCOUT-INTRA-02 — directional luma + ext/sub8 thin leaves: wrong recon

Duplicate note: `intra-tools.md` SCOUT-INTRA-02 reports the same symptom for
`--still-mrl` and the same first-bad leaf (32x4 at x=32,y=8). This entry adds
that the **non-MRL** path (`--still-full-intra`/`--still-rdo-intra`) is equally
broken, so fixing only `mrl::prepare_edges` will not close it. First diff here is
also sample (32,8), inside the `size=32x4 y=8/-2` leaf emitted by `--still-sub8`.

Repro (64x64 shown; same on `tests/data/basic_640x360_1f.y4m`, 131412 diff,
max 210):
```sh
cd /Users/arin/work/av2enc
./build/av2enc tests/data/basic_64x64_1f.y4m -o /tmp/s.ivf --frames 1 \
    --still --still-sub8 --still-full-intra --recon /tmp/s.recon.y4m
./tools/bin/dav2d-macos -q -i /tmp/s.ivf -o /tmp/s.dec.y4m --threads 1 --strict 1
python3 - /tmp/s.dec.y4m /tmp/s.recon.y4m <<'PY'
import sys
def p(f):
    d=open(f,'rb').read(); i=d.index(b'\n')+1; i=d.index(b'\n',i)+1; return d[i:]
a,b=p(sys.argv[1]),p(sys.argv[2]); d=[abs(x-y) for x,y in zip(a,b) if x!=y]
print(len(d), max(d) if d else 0)
PY
```
Observed: dav2d accepts the stream; recon differs (sub8 64x64: 3367 diffs,
max 77; ext 64x64: 3320 diffs, max 93). `--still-sub8`/`--still-ext-partitions`
with default (DC) luma are exact, as is `--still-full-intra`/`--still-rdo-intra`
without a partition tool. `--no-deblock` and `AV2ENC_STILL_NO_EDGE=1` do not
change the mismatch. Forced-mode sweep (`AV2ENC_STILL_FORCE_Y=m:0`): DC, SMOOTH,
SMOOTH_V, SMOOTH_H, PAETH are exact; all directional modes differ (sub8: V and
D45..D203; ext: H and D45..D203). `--still-tx-type` alone is exact (luma stays
DC); `--still-ext-partitions --still-full-intra --still-tx-type` fails the same
way. Forcing a directional *chroma* mode on the same thin chroma blocks
(`--still-uv-full` + `AV2ENC_STILL_FORCE_UV=3:0`) is exact, so the shared edge
sampler looks right and the mismatch is on the luma directional path.
Expected: decoder recon == encoder recon, as for the plain `--still` cases.
Likely cause (suspected): the luma directional path (§7.13.2.8 IDIF is luma-only;
`intra_modes.cpp:142-280`, edges from `still_encoder.cpp:947-1025`) diverges for
4-wide / 4:1 leaves that only `--still-sub8` and `--still-ext-partitions` emit
(plain `--still` reaches at best 8x8/rect 2:1). The magnitude is consistent with
a wrong extended edge/IDIF sample (max 77..210), not a mode parse failure, since
dav2d accepts the stream. Note: with `AV2ENC_STILL_SQUARE_ONLY=1` and ext +
`FORCE_Y=3:0` luma becomes exact but U/V still differ (312/354 samples, max 3),
which may be a second, ext-shared-chroma-side issue.

### SCOUT-INTRA-03 — `--still-depth` accepted-and-ignored under `--still-sdp`

Repro:
```sh
cd /Users/arin/work/av2enc
for d in 0 2 5; do ./build/av2enc tests/data/basic_64x64_1f.y4m \
    -o /tmp/d$d.ivf --frames 1 --still --still-sdp --still-depth $d; done
md5 -q /tmp/d0.ivf /tmp/d2.ivf /tmp/d5.ivf
```
Observed: all three (and the no-flag default) are the same stream
`e46f910f8b9fcd32877987376b3b53e7`; no warning. With plain `--still` the same
depths give three different streams. Expected: reject `--still-depth` with
`--still-sdp` by name (like the other SDP-incompatible flags at
`main.cpp:2138-2165`) or honour it. The SDP solver hardcodes
`sopt.max_depth = 0` / `min_block_size_px = 64` (`still_encoder.cpp:487-489`),
so the flag cannot take effect.

### SCOUT-INTRA-04 — gates keep luma on DC, so 01/02 are invisible

`--still-sub8` product gate forces `--still-dc`
(`src/codec/partition/sub8/product_gate.sh:111`), the ext gate uses default
DC-only luma (`src/codec/partition/ext/product_gate.sh:74`), and the SDP gate
also uses default DC. `tools/still_verify/intra_gate.sh` has no ext/sub8/SDP
case and never combines `--still-full-intra`/`--still-rdo-intra` with them;
`feature_gate.sh:128-131` covers each flag only on plain `--still`. Docs still
list all three as PASS (`docs/m1-features.md:553-555`). That is why a
non-DC directional luma mode is only now surfacing as an undecodable (01) or
mis-reconstructed (02) stream.

## Dedup notes
- `refactor/bugs.csv` and `git log --all --grep` have no SDP/full-intra/sub8/ext
  mode-decision entries.
- `bug-reports/intra-tools.md` SCOUT-INTRA-02 covers the same 4-sample-axis leaf
  symptom for MRL; 02 here is marked duplicate but documents that the base
  directional predictor (no MRL) is also wrong.
- `bug-reports/formats-hbd.md` SCOUT-FORMATS-HBD-01 covers `--still-sdp`/
  `--still-sub8` being ignored on the 10-bit/chroma format paths; this report's
  01/03 are the 8-bit `--still` SDP path and are distinct.
- `docs/negative-results.md:52` documents a different ext failure (VERT_4A with a
  4-wide 1/8 child); the repros here use the default ext tree and `--still-sub8`,
  so they are not that trace.

