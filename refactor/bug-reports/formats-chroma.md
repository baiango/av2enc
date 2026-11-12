# formats-chroma bug scout (8-bit 4:2:2 / 4:4:4 / 4:0:0)

Lane: `--still --chroma 422|444|400` input parsing, geometry/edges, headers,
`--chroma` validation, and the 8-bit halves of the CUDA chroma-format component.
Read-only; `build/av2enc`, the prebuilt `chroma_test`/format unit binaries,
`tools/bin/dav2d-macos`, `compare_y4m.py`.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-CHROMA-01 | med | confirmed | `--still-sub8` is accepted and silently ignored on the 8-bit `--chroma` path (same for `--still-tcq-scale`, which also lacks its `--still-tcq` guard) | `$B /tmp/c422.y4m -o b.ivf --still --chroma 422 --still-sub8` | byte-identical IVF with/without the flag, while the same flag changes the plain 8-bit `--still` stream | `src/main.cpp:3300-3324` (rejection table), `src/main.cpp:1994-1996` | duplicate of SCOUT-FORMATS-HBD-01; tcq-scale fix exists only on unmerged `merge/trial-a` (a4b72e4) |
| SCOUT-CHROMA-02 | med | confirmed | Y4M frame body is never size-checked: a 4:4:4 body tagged `C422`/`C420` is silently re-encoded as the smaller format on the 8-bit `--chroma` path | see Details | `64x64 C422` reported although the file body is 4:4:4 (12,288 B, 4,096 B used) | `src/formats/format_y4m.cpp:273-287`, `format_y4m.h:28-31` | duplicate of SCOUT-FORMATS-HBD-02 (8-bit instance) |
| SCOUT-CHROMA-03 | low | doc/gap | Chroma-format component still points at the pre-`option-a` tree (`src/encoder/format/...`, `src/cuda/chroma_formats/...`, `src/cuda/m1/hbd/...`); the runtime high-bit-depth error and the goldens regeneration command name files that do not exist | see Details | `ls src/encoder` -> no such file; `./src/encoder/format/chroma/gate.sh` -> no such file; 46 stale refs in the README alone | `src/formats/chroma/README.md`, `chroma_format.cpp:106-109`, `chroma_still.h:8-9`, `chroma/goldens/chroma_goldens.txt:2`, `src/formats/README.md`, `format_tables.h:1`, `src/cuda/formats/chroma/README.md` | new |
| SCOUT-CHROMA-04 | low | doc/gap | 8-bit chroma gates pin only 64x64 and 640x360: no odd/partial-edge geometry case (although the README claims 100x100/101x99 manual checks) and no negative case for the documented input-colorspace rejection | see Details | both gates pass with only even, edge-light sizes; `grep -c '100x100\|101x99'` in the gates is 0 | `src/formats/format_gate.sh:186-233`, `src/formats/chroma/gate.sh:115-118` | new (HBD-05 covers only high-QP coverage) |

## Details

### SCOUT-CHROMA-01 — no-effect Still tree flags accepted on the format path
Repro:
```sh
python3 -c "open('/tmp/c422.y4m','wb').write(b'YUV4MPEG2 W64 H64 F1:1 Ip C422\nFRAME\n'+bytes(4096+2048+2048))"
B=/Users/arin/work/av2enc/build/av2enc
$B /tmp/c422.y4m -o /tmp/a.ivf --still --chroma 422
$B /tmp/c422.y4m -o /tmp/b.ivf --still --chroma 422 --still-sub8
md5 -q /tmp/a.ivf /tmp/b.ivf
$B /tmp/c422.y4m -o /tmp/c.ivf --still --chroma 422 --still-tcq-scale 0.5
md5 -q /tmp/a.ivf /tmp/c.ivf
```
Observed: all three MD5s are `4544901142729730a7356a53acf5a5ea`; both flags
exit 0. On the plain 8-bit `--still` (M1) path `--still-sub8` changes the
stream, so the flag is real but not applied here. `--still-tcq-scale` is
accepted even without `--still-tcq` (the analogous
`--still-tx-type-scale` has that guard at `src/main.cpp:1994`).
Expected: exit 2 from the format-path rejection table
(`the --bit-depth/--chroma format paths use their own minimal still writer`).
Status: `src/main.cpp:3310-3324` omits `options.still_sub8` and the
`--still-tcq-scale` token. Already reported for the 10-bit/SDP case in
`bug-reports/formats-hbd.md` (SCOUT-FORMATS-HBD-01), so this row is a
duplicate; the tcq-scale guard was fixed on `merge/trial-a` (a4b72e4) but not
on `main`.

### SCOUT-CHROMA-02 — frame-body length is never validated
Repro:
```sh
python3 -c "open('/tmp/c444.y4m','wb').write(b'YUV4MPEG2 W64 H64 F1:1 Ip C444\nFRAME\n'+bytes(4096*3))"
python3 -c "d=open('/tmp/c444.y4m','rb').read().replace(b'C444',b'C422',1);open('/tmp/c444_as422.y4m','wb').write(d)"
/Users/arin/work/av2enc/build/av2enc /tmp/c444_as422.y4m -o /tmp/c.ivf --still --chroma 422
```
Observed: `Format still-picture encode complete`, `input: ... (64x64, C422)`;
the reader consumes 4,096 B of the 12,288 B body and encodes it as 4:2:2.
Expected: reject (the header promises a "truncated/oversized frame" error).
Root cause and 10-bit repro are already in SCOUT-FORMATS-HBD-02
(`format_y4m.cpp` only checks `gcount() != raw.size()`); this row records that
the 8-bit `--chroma` path is affected identically. Duplicate.

### SCOUT-CHROMA-03 — pre-refactor paths in code, docs and goldens
Repro:
```sh
cd /Users/arin/work/av2enc
ls src/encoder                       # No such file or directory
ls src/cuda/chroma_formats           # No such file or directory
ls src/cuda/m1/hbd                   # No such file or directory
sed -n 2p src/formats/chroma/goldens/chroma_goldens.txt
grep -n 'src/encoder' src/formats/chroma/chroma_format.cpp
```
Observed: `chroma_goldens.txt` says to regenerate with
`CHROMA_UPDATE_GOLDENS=1 ./src/encoder/format/chroma/gate.sh`; the gate lives
at `src/formats/chroma/gate.sh`. `chroma_format.cpp:108` tells callers to use
`src/encoder/format/ + src/cuda/m1/hbd/`; both are gone (current
`src/formats/` and `src/cuda/pipeline/still/hbd/`). The chroma README's whole
"Exact M1/M2 core call sites" table points at `src/encoder/m1/...` (now
`src/pipeline/still/`), and its build commands at
`src/encoder/format/chroma/...`. The CUDA chroma README's build commands use
`src/cuda/chroma_formats/` (actual: `src/cuda/formats/chroma/`). Matching
lines with stale paths: 46 in `src/formats/chroma/README.md`, 12 in
`src/formats/README.md`, 8 in the CUDA README, plus `format_tables.h:1`
(generator comment) and `chroma_still.h:8-9`.
Expected: paths that resolve, or no path guidance. Since the component exists
to hand integrators exact change sites, the stale table is actively
misleading. Not covered by SCOUT-FORMATS-HBD-05 (which is hbd doc drift).

### SCOUT-CHROMA-04 — gate coverage for 8-bit chroma geometry / rejection
Repro:
```sh
cd /Users/arin/work/av2enc
grep -n 'run_case\|gen_8bit_source' src/formats/format_gate.sh | tail -20
grep -n 'run_case' src/formats/chroma/gate.sh
./src/formats/format_gate.sh --bin build/av2enc   # PASS in ~5 s
```
Observed: `format_gate.sh` runs only 64x64 and 640x360 for 4:2:2/4:4:4/4:0:0;
`src/formats/chroma/gate.sh` likewise. `src/formats/chroma/README.md` claims
100x100 and 101x99 were checked manually ("partial right edge", "odd width and
height") and that edge prediction clamps to the coded plane, but no gate pins
it — a regression in `supports_64x64_none`, `plane_max_x/y`, or the chroma
`ceil` geometry would pass. There is also no negative case for the
"input must already be in the requested colorspace" error, although
`docs/negative-results.md:107` calls that a fixed constraint.
Expected: add one odd/partial-edge case per format and one mismatch-rejection
case. (I ran 100x100, 101x99, 97x97, 100x225, 225x100 and ~40 other geometries
by hand for 422/444/400 — all decoded `--strict 1` with 0 recon diffs — so this
is a coverage gap, not a current stream failure.)

## Lanes checked with no defect found
The minimal 8-bit chroma writer/reader itself appears correct; these were
adversarially exercised and all produced `dav2d --strict 1` streams whose
recon was bit-exact (`compare_y4m.py`, 0 diffs):

* 4:2:2 / 4:4:4 / 4:0:0 at 33x33, 40x40, 64x64, 97x97, 100x100, 101x99,
  104x104, 121x225, 128x127, 161x64, 192x128, 225x225, 256x256, 320x180,
  384x64, 100x225, 225x100, 225x121, 640x360, 1920x1080 (4:4:4 random noise),
  plus the 4:2:0 legs of the same writer at odd sizes (101x99, 100x100, 33x33).
* Constant planes (0 and 255) at 101x99 for all three formats (DC/clip edges).
* Full-range random noise at 225x225/320x180/640x360 (escape/`hr_level_avg`
  path) and 1920x1080 4:4:4.
* Headers via `chroma_still --inspect`: `chroma_format_idc` 0/3/2/1,
  profile 0/3/4/0, `SubsamplingX/Y` and `Monochrome`/`NumPlanes` all correct.
* Inputs that should reject do: C420->422/444/400, C444->422, C444->420 (M1),
  Cmono10 at 8-bit, missing/invalid dimensions, truncated body, `C429`.
* `format_test` (143,399 checks) and `chroma_test` (2,541 checks) pass;
  `chroma_formats` CUDA emulation passes (1,514,866 checks). Note
  `src/formats/chroma/README.md` still says "2,531-check CPU unit test".
