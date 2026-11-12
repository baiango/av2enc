# loop restoration bug scout

Dedup: `refactor/bugs.csv` has no loop-restoration rows and no open fix branch;
`bug-reports/partition.md` already reports the SB128/256 LR/GDF/partial-unit
findings (SCOUT-PARTITION-04/05/09), and `bug-reports/cdef-ccso.md` repeats two
of them.  Those are marked duplicate below; the new finding is the qp=1
chroma-LR tile rejection, plus two documentation/gate gaps.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-LR-01 | high | confirmed | `--still-sb128/256` + Wiener/chroma/PC-Wiener writes a 64-SB `LoopRestorationSize` group; decoder infers 128-sample LR units | `av2enc tests/data/basic_640x360_1f.y4m -o /tmp/a.ivf --frames 1 --still --still-sb128 --still-wiener` + `still_dump --sb128 --wiener` + `verify.py` | Y 25319/230400 differ (max 107), U/V exact; chroma-LR U 6047 + V 5442 differ; PC-Wiener Y 5836 differ; sb256 identical; adding CDEF aborts the encoder with `use_bank_k outside the current bank` | `src/pipeline/still/still_headers.cpp:75-82,422-423,382`; `src/pipeline/still/still_encoder.cpp:4642,4652-4653,4690-4692`; `src/codec/filters/loop_restoration/classes/wiener_decision.cpp:42-50` | duplicate (SCOUT-PARTITION-05; also SCOUT-CDEF-02) |
| SCOUT-LR-02 | high | confirmed | GDF with 128/256 superblocks codes `gdf_unit_matches_sb_size`, which is absent at SbSize != 64; decoder rejects the sequence header | `av2enc tests/data/basic_640x360_1f.y4m -o /tmp/g.ivf --frames 1 --still --still-sb128 --still-gdf` then `dav2d -i /tmp/g.ivf --strict 1` | `Error parsing sequence header` / `Error parsing OBU data`, no frames; also at 64x64 and with `--still-sb256` | `src/pipeline/still/still_encoder.cpp:4912,4955-4957`; `src/pipeline/still/still_headers.cpp:185-188`; `src/codec/filters/gdf/gdf_syntax.cpp:82-84` | duplicate (SCOUT-PARTITION-04; also SCOUT-CDEF-03) |
| SCOUT-LR-03 | high | confirmed | Restoration-unit writer rejects legal frame sizes whose last partial unit starts at a unit multiple (Wiener/chroma LR/PC-Wiener) | `av2enc testdata/src_100x70.y4m -o /tmp/o.ivf --frames 1 --still --still-wiener`; also 640x80 crop | exit 1, `LR unit grid does not cover a coding block`; `--still-chroma-lr` same; `--still-pc-wiener --qp 200` says `PC-Wiener unit grid...`; GDF/CCSO encode the same frame | `src/pipeline/still/still_syntax.cpp:2216-2225,2254-2268`; spec §5.20.10.4 unit ranges | duplicate (SCOUT-PARTITION-09) |
| SCOUT-LR-04 | med | confirmed | `--qp 1` + chroma LR at 640x360 produces a stream `dav2d` rejects at tile decode | `av2enc tests/data/basic_640x360_1f.y4m -o /tmp/q1.ivf --frames 1 --still --qp 1 --still-chroma-lr` then `dav2d -i /tmp/q1.ivf --strict 1` | `Error decoding frame: Invalid argument` / `No data decoded`; qp 2/3/5/10/20/40/90 pass; depths 1-3/5-7, `--still-single`, `--still-rdoq`, `--still-cfl-ds 1/2` pass; no header error | `src/pipeline/still/still_encoder.cpp:4594-4735`; `src/codec/filters/loop_restoration/classes/wiener_syntax.cpp:1446-1472` | new |
| SCOUT-LR-05 | low | doc | Loop-restoration README still says PC-Wiener/GDF/grain are unimplemented and that nothing has been decoded with dav2d | `grep -n "Nothing here has been decoded\|remains unimplemented\|Blockers.*TODO" src/codec/filters/loop_restoration/README.md` | lines 30, 173, 231+ contradict the integrated status and the passing loop-restoration gate | `src/codec/filters/loop_restoration/README.md` | new |
| SCOUT-LR-06 | low | doc | Loop-restoration product gate has no SB128/256 or partial-unit case, so the duplicate findings passed CI | `grep -n "sb128\|sb256\|100x70\|640x80" tools/still_verify/loop_restoration_gate.sh` | no matches; all LR cases are 64x64/640x360 (plus 1080p) | `tools/still_verify/loop_restoration_gate.sh` | new |

## Duplicate cross-references

### SCOUT-LR-01 — SB128/256 LR header/unit grid
Cross-ref `bug-reports/partition.md:32` (SCOUT-PARTITION-05) and
`bug-reports/cdef-ccso.md:6` (SCOUT-CDEF-02).  Extra confirmation here: at
640x360 the exact counts are Y 25319 (max 107), chroma U 6047 + V 5442, and
PC-Wiener Y 5836; `--still-sb128 --still-cdef --still-wiener` aborts the encoder
with `use_bank_k outside the current bank` (a 32-byte partial IVF is left
behind).  Root cause: §5.18.7.11 has no shift-3 code at SbSize 128; the still
path hard-codes 64-SB geometry and `write_lr_shift()` always writes the
three-bit 64-SB form, so the encoder filters 64x64 units while the decoder
infers 128-sample units.

### SCOUT-LR-02 — SB128/256 GDF sequence-header desync
Cross-ref `bug-reports/partition.md:31` (SCOUT-PARTITION-04) and
`bug-reports/cdef-ccso.md:7` (SCOUT-CDEF-03).  Extra confirmation: the failure
is in the sequence header at both 64x64 and 640x360, and `--still-sb256`
behaves the same.  `still_encoder.cpp` always builds the GDF context with
`sb_size=64`; `still_headers.cpp` then writes `gdf_unit_matches_sb_size`, which
§5.4.10 omits for a non-64 sequence SB.

### SCOUT-LR-03 — partial restoration unit at the frame edge
Cross-ref `bug-reports/partition.md:36` (SCOUT-PARTITION-09).  Extra
confirmation: `testdata/src_100x70.y4m` fails for Wiener and chroma LR, and
`--still-pc-wiener --qp 200` fails with the PC-Wiener message; a 640x80 crop of
the tracked fixture fails identically.  `count_units_in_frame` rounds down for
the last partial unit, so §5.20.10.4 reads no symbol at the boundary, but
`write_lr_plane_units()` tests `origin % unit_size == 0` and errors instead of
using the clamped unit range.

## Details

### SCOUT-LR-04 — qp=1 chroma-LR stream rejected by the decoder
Repro:
```sh
cd /Users/arin/work/av2enc
./build/av2enc tests/data/basic_640x360_1f.y4m -o /tmp/q1.ivf \
  --frames 1 --still --qp 1 --still-chroma-lr
tools/bin/dav2d-macos -q -i /tmp/q1.ivf -o /tmp/q1.y4m --threads 1 --strict 1
```
Observed: `Error decoding frame: Invalid argument` / `No data decoded` (both
`--strict 0` and `1`).  qp 2, 3, 5, 10, 20, 40 and 90 decode; default depth 4
fails while depths 1-3/5-7, `--still-single`, `--still-rdoq` and
`--still-cfl-ds 1/2` pass.  `shine_640x360_1f.y4m` at qp1 decodes.  The encoder
recon dump is produced successfully, so this is not an encoder-side recon
assertion.

Expected: a legal qp1 chroma-LR picture decodes; the header is conformant
(the failure is at frame decode, with no `Error parsing sequence header`).

Likely cause: a rare per-unit Wiener plan (subset/`merged`/bank or coefficient
encoding) selected only by this depth-4 partition plan emits tile syntax the
pinned decoder rejects.  qp1 changes only the reconstruction/plan, not the LR
header, which is why qp2+ and the plan-changing flags pass; a tile-symbol trace
is needed to name the exact unit.

### SCOUT-LR-05 — stale loop-restoration docs
Repro:
```sh
grep -n "Nothing here has been decoded\|remains unimplemented\|Blockers.*TODO" \
  src/codec/filters/loop_restoration/README.md
```
Observed: the file says the filters have no syntax path, PC-Wiener is
unimplemented, and PC-Wiener/GDF/film grain are TODO, while the same file's
status table and `loop_restoration_gate.sh` show all three integrated.  (The
separate `docs/negative-results.md:121` CDEF/CCSO claim is already
SCOUT-CDEF-04.)

Expected: docs match the code/gates (or carry an explicit date/limitation).

### SCOUT-LR-06 — gate does not cover the failing configurations
Repro:
```sh
grep -n "sb128\|sb256\|100x70\|640x80" tools/still_verify/loop_restoration_gate.sh
```
Observed: no matches.  The gate's LR cases are only 64x64, 640x360 and 1080p,
so SCOUT-LR-01/02/03 (duplicates above) are outside CI.

Expected: add at least one SB128 LR case, one SB128 GDF case, and one
partial-unit frame-size case.

Note: film grain (fixed and adaptive) was exercised at 64x64, 640x360, odd
crops, qp 1/90/255, with and without LR/GDF; no product defect was found.  The
adaptive path intentionally emits no grain for flat content
(`apply_grain=false`), which is documented in the encoder comment.
