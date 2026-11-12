# partition bug scout

Read-only scout for partition/geometry (SB128/256, SDP, sub-8, ext/shared-chroma,
frame edges, `--still-depth`/`--still-single`). All repros use `build/av2enc`,
`tools/bin/dav2d-macos -q -i X.ivf -o X.dec.y4m --threads 1`, and compare the
`--recon` Y4M body to the decoded Y4M body (header `A0:0`/`A1:1` ignored).
Dedup: `refactor/bugs.csv` has no partition rows; no other file in `bug-reports/`.
Inputs are generated in /tmp (any 4:2:0 Y4M with the stated size works):
```sh
python3 - <<'PY'
for w,h in [(128,128),(192,128),(100,64),(100,70)]:
    open(f'/tmp/in_{w}x{h}.y4m','wb').write(
      f'YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420\nFRAME\n'.encode()
      + bytes((x*5+y*2)%256 for y in range(h) for x in range(w))
      + bytes([128])*((w//2)*(h//2)*2))
PY
```
Recon comparison used throughout (pixel body only):
```py
def body(p):
    d=open(p,'rb').read(); return d[d.index(b'\n')+1:]
a,b=body('/tmp/x.y4m'),body('/tmp/x.dec.y4m')
print('OK' if a==b else f'{sum(x!=y for x,y in zip(a,b))} diffs')
```

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-PARTITION-01 | high | confirmed | Extended partitions break on partially covered superblocks (encode failure or decoder-rejected stream) | `build/av2enc testdata/src_100x70.y4m -o /tmp/e.ivf --still --still-ext-partitions`; and 100x64 synthetic below | `extended partition decision failed: superblock (16,16): spec-forced partition conflicts with DecisionOptions`; 100x64 encodes but dav2d: `Error decoding frame: Invalid argument No data decoded` | `src/codec/partition/ext/partition_ext.cpp`, `src/pipeline/still/still_encoder.cpp` | new |
| SCOUT-PARTITION-02 | high | confirmed | MRL is not composable with any non-default partition geometry: silent recon mismatch (SB128/256, sub-8, ext) or decoder rejection (SDP) | `build/av2enc testdata/shine_640x360_1f.y4m -o /tmp/m.ivf --still --still-sb128 --still-mrl --recon /tmp/m.y4m` | decoder OK, recon differs: luma 48,235/230,400 samples (max 27); sub-8: 116,583/230,400 (max 165); ext: 148,328/230,400 (max 161); `--still-dc` makes it pass | `src/pipeline/still/still_encoder.cpp:1223-1404`, `src/pipeline/still/still_syntax.cpp:2606` | new |
| SCOUT-PARTITION-03 | high | confirmed | MHCCP + shared-chroma/ext or sub-8 leaves: decoder rejects (chroma owner mismatch) | `build/av2enc testdata/shine_640x360_1f.y4m -o /tmp/em.ivf --still --still-ext-partitions --still-mhccp` (also `--still-sub8 --still-depth 8`) | dav2d: `Error decoding frame: Invalid argument`; flat input (no ext/4x4 rows selected) decodes fine; MHCCP alone passes | `src/pipeline/still/still_encoder.cpp`, `src/pipeline/still/still_syntax.cpp` | new |
| SCOUT-PARTITION-04 | high | confirmed | SB128/256 + GDF emits a decoder-rejected stream | `build/av2enc in_128x128.y4m -o /tmp/g.ivf --still --still-sb128 --still-gdf` | dav2d: `Error decoding frame: Invalid argument`; GDF alone and SB128 alone pass; same at 192x128/320x192/640x360, sb256 too | `src/pipeline/still/still_encoder.cpp` (GDF unit sizing), `src/pipeline/still/still_headers.cpp` | new |
| SCOUT-PARTITION-05 | high | confirmed | Loop restoration (Wiener/chroma-LR) with SB128/256: silent recon mismatch, or `use_bank_k` failure | `build/av2enc in_128x128.y4m -o /tmp/w.ivf --still --still-sb128 --still-wiener --recon /tmp/w.y4m` | luma 1,625/16,384 samples differ (max 31); chroma-LR 383 bytes; Wiener+chroma-LR at 256x256 fails `use_bank_k outside the current bank`, at 640x360 it decodes but recon differs (86,217 bytes) | `src/pipeline/still/still_encoder.cpp`, `src/codec/filters/loop_restoration/` | new |
| SCOUT-PARTITION-06 | high | confirmed | SDP + chroma-LR: decoder rejects the stream (chroma LR unit written from the SDP tree) | `build/av2enc in_192x128.y4m -o /tmp/s.ivf --still --still-sdp --still-chroma-lr` | dav2d: `Error decoding frame: Invalid argument`; SDP alone and SDP+Wiener pass; `--still-dc` still fails | `src/pipeline/still/still_syntax.cpp:2204-2231`, `src/codec/partition/sdp/` | new |
| SCOUT-PARTITION-07 | med | doc | SDP only encodes partial superblocks with a strip >= 40 px; 720p/480p stills fail mid-encode | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/s.ivf --still --still-sdp` | `SDP partition decision failed: superblock (176,0): no partition candidate has a finite cost`; 640x480: `sub-tree partitions are not wired`; 128x72/88/96 fail, 128x104 passes | `src/codec/partition/sdp/sdp.cpp:539-680`, `src/main.cpp:2145` | new |
| SCOUT-PARTITION-08 | med | confirmed | TX partitioning with 4x4 leaves (sub-8 or ext) aborts on `MiSize 0` | `build/av2enc in_192x128.y4m -o /tmp/t.ivf --still --still-sub8 --still-depth 8 --still-tx-partition` | `tx partition geometry failed: tx partition TX_PARTITION_NONE is not signallable for MiSize 0`; flat input (no 4x4 leaf) passes; `--still-tx-partition` alone passes | `src/codec/transform/tx_partition/tx_partition.cpp:307-330`, `src/pipeline/still/still_encoder.cpp:2110` | new |
| SCOUT-PARTITION-09 | med | confirmed | LR/PC-Wiener unit grid undercounts partial superblocks: 720p stills are rejected | `build/av2enc testdata/shine_1280x720_60f.y4m -o /tmp/l.ivf --still --still-wiener` | `LR unit grid does not cover a coding block` (chroma-LR and PC-Wiener too); 640x360 works; 100x70 fails | `src/codec/filters/pc_wiener/pc_wiener_stream.cpp:895-898`, `src/pipeline/still/still_syntax.cpp:2212` | new |

## Details

### SCOUT-PARTITION-01 — ext partitions and partial superblocks
The 64-SB ext decision/solver lets extended rows cross the right/bottom coded
boundary instead of forcing a fit. At 100x64 the trace shows
`HORZ_3@64x64` and `VERT_3@64x16` at `mi_col=16` (x=64) although only 36 columns
are visible; a `VERT_3` child starts beyond `MiCols` and dav2d rejects the whole
stream. Encode-side failures are geometry dependent (`src_100x70.y4m` fails at
default depth; `--still-sb128 --still-ext-partitions` also fails `prepare_edges`
on 320x200 and `spec-forced partition conflicts` whenever a dimension is not a
multiple of 64, e.g. 136x88, 192x130, 200x200, 320x200).
This is the same family as negative-results.md row 52 (vertical uneven rows at
the bottom edge) but at the right edge with `HORZ_3`/`VERT_3`, and the documented
"keep vertical uneven at w>=64" guard does not prevent it.

Repro:
```sh
python3 - <<'PY'
w,h=100,64
open('/tmp/e100.y4m','wb').write(b'YUV4MPEG2 W100 H64 F30:1 Ip A1:1 C420\nFRAME\n'
 + bytes((x*5+y*2)%256 for y in range(h) for x in range(w)) + bytes([128])*((w//2)*(h//2)*2))
PY
build/av2enc /tmp/e100.y4m -o /tmp/e100.ivf --still --still-ext-partitions --recon /tmp/e100.rec.y4m
tools/bin/dav2d-macos -q -i /tmp/e100.ivf -o /tmp/e100.dec.y4m --threads 1
```
Observed: `Error decoding frame: Invalid argument / No data decoded` (also with
`--still-depth 4`/`6`, `--no-deblock`; base `--still` on the same file is exact).
Expected: either a valid ext layout within the frame or a clean rejection.

### SCOUT-PARTITION-02 — MRL with SB128/256, sub-8, ext, SDP
MRL prediction/syntax is only correct for 64x64 SBs with >=8x8 leaves.
SB128/256 (`--still-sb128/256 --still-mrl`), sub-8 (`--still-sub8
--still-depth 8 --still-mrl`) and ext (`--still-ext-partitions --still-mrl`)
decode but the decoded planes differ from the encoder reconstruction; SDP+MRL
is rejected by dav2d. `--still-dc` (MRL never selected) passes, so the
disagreement is in the MRL prediction/edges, not the container.
```sh
build/av2enc testdata/shine_640x360_1f.y4m -o /tmp/m.ivf --still --still-sb128 --still-mrl --recon /tmp/m.y4m
tools/bin/dav2d-macos -q -i /tmp/m.ivf -o /tmp/m.dec.y4m --threads 1
# body compare from the preamble: 48,235 luma diffs, max 27
```

### SCOUT-PARTITION-03 — MHCCP + shared-chroma/sub-8 ownership
`--still-ext-partitions --still-mhccp` and `--still-sub8 --still-depth 8
--still-mhccp` encode but dav2d rejects. A flat input that selects no ext/4x4
rows decodes, so MHCCP emits chroma for leaves that do not own the shared
ChromaMi (or misses the owner) at ext/sub-8 leaves. MHCCP alone, CfL,
`--still-cctx`, and `--still-uv-full` are exact.

### SCOUT-PARTITION-04 — SB128/256 + GDF
GDF's unit grid is still derived for a 64 SB while the header signals
`gdf_unit_matches_sb_size=1` for a 128 SB, so dav2d rejects every
`--still-sb128/256 --still-gdf` stream (128x128, 192x128, 320x192, 640x360).
`--still-gdf` alone and SB128/256 alone are exact. Same class as the fixed
CDEF/CCSO SB128 row in negative-results.md:118, not yet applied to GDF.

### SCOUT-PARTITION-05 — SB128/256 + loop restoration
`--still-sb128 --still-wiener` decodes but luma differs (1,625/16,384 samples at
128x128, max 31; 64,201/230,400 at 640x360). `--still-chroma-lr` differs on
both chroma planes. At 256x256 Wiener+chroma-LR aborts with `use_bank_k outside
the current bank` (640x360 decodes but differs), i.e. the LR bank/unit index is
still 64-SB based. `--still-pc-wiener` with SB128 is exact, so this is
Wiener-unit bookkeeping.

### SCOUT-PARTITION-06 — SDP + chroma LR
SDP writes the chroma LR unit symbol from the chroma-partition tree; dav2d
rejects (`Invalid argument`). `--still-sdp --still-wiener` is exact and
`--still-dc` does not help, so the break is chroma-plane LR syntax/ownership,
not luma.

### SCOUT-PARTITION-07 — SDP partial-SB coverage
SDP forces NONE/NONE trees; a boundary that forces SPLIT or needs a sub-tree
partition aborts. The supported envelope is only "partial strip >=40 px":
`128x72`, `128x88`, `128x96`, `640x328`, `640x480`, `1280x720` fail, while
`128x104`, `640x360`, `1920x1080` pass. The 640x360 product gate hides this
because its strip is 40. The CLI accepts `--still-sdp` for any size, so a
standard 720p input fails with an internal error instead of a usage rejection.

### SCOUT-PARTITION-08 — TX partition + 4x4 leaves
The still writer calls `tx_partition_geometry()` for every luma block with
`TX_PARTITION_NONE`. `tx_partition_signallable(kBlock4x4, NONE)` is false, so
any sub-8 (4x4) or ext (4x4/4xN) leaf makes the whole encode fail. The
combination is accepted by `src/main.cpp`; it should either force
`TX_PARTITION_NONE` with no syntax for non-signallable sizes or be rejected by
name.

### SCOUT-PARTITION-09 — LR/PC-Wiener unit count rounds down
`CountUnits()` uses `(frame_size + unit/2) / unit` (round-to-nearest) instead of
ceil, so a dimension whose partial superblock is smaller than 32 px yields too
few units; any coding block at the last unit origin is outside the grid and the
encode aborts. `1280x720` (720 = 11*64+16) is rejected for `--still-wiener`,
`--still-chroma-lr` and `--still-pc-wiener`; 640x360 (360 = 5*64+40) works, and 100x70 rejects
Wiener/chroma-LR (PC-Wiener at 100x70 happens to pass, but fails at 128x72).
