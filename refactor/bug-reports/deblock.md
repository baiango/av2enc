# deblock bug scout

Lane: deblock / edge selection / strengths / TX-subPU grids / chroma ownership /
per-segment qindex / CDEF interaction / `--no-deblock` / tile boundaries /
lossless-BRU.

Fault note: deblock core, tile-edge clamping and lossless skipping were exercised
with recon-vs-`dav2d` checks (see "Coverage" below) and came back exact. The two
real faults below are per-segment qindex plumbing bugs: the encoder quantises and
deblocks with `base + SEG_LVL_ALT_Q`, but two partition collectors never emit the
segment id, so the decoder dequantises *and deblocks* those blocks with segment 0.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-DEBLOCK-01 | high | confirmed | `--still-ext-partitions` drops per-SB segment id: quant + deblock qindex desync | see Details 01 | enc rc 0, `dav2d --strict 1` rc 0; Y 65532/65536, U 9298/16384, V 8518/16384 samples differ, max 105 | `src/pipeline/still/still_encoder.cpp:764-772`, `:807-850` | new |
| SCOUT-DEBLOCK-02 | high | confirmed | `--still-sdp` drops per-SB segment id: same desync | see Details 02 | enc rc 0, decode rc 0; Y 65536/65536, U 12105/16384, V 12096/16384 differ, max 89 | `src/pipeline/still/still_encoder.cpp:764-772`, `:860-910` | new |
| SCOUT-DEBLOCK-03 | low | doc | `--no-deblock` accepted-and-ignored by the format still writers (10-bit/422/444/400) | see Details 03 | identical IVF MD5 with and without the flag; no error | `src/main.cpp:3300-3345`, `src/pipeline/still/high_bitdepth/high_bitdepth_headers.cpp:194-196` | new |
| SCOUT-DEBLOCK-04 | low | doc | stale `still_recon_dump` hides the ext/SDP paths; `gate_per_segment_q.sh` cannot see SCOUT-DEBLOCK-01/02 | see Details 04 | `still_recon_dump` exits 2: unknown option `--still-ext-partitions`/`--still-sdp`/`--still-ccso-edge`/`--still-wiener`; gate only runs plain `--still-per-segment-q` | `src/pipeline/still/stream/still_recon_dump.cpp`, `src/pipeline/still/stream/gate_per_segment_q.sh:82-91` | new |

## Details

### SCOUT-DEBLOCK-01 — extended partitions drop the per-SB segment id
Repro:
```sh
B=/Users/arin/work/av2enc
$B/build/av2enc $B/testdata/shine_256x256_1f.y4m -o /tmp/ext.ivf --frames 1 \
  --still --still-ext-partitions --still-per-segment-q --still-seg-delta 40 \
  --recon /tmp/ext.rec.y4m
$B/tools/bin/dav2d-macos -q -i /tmp/ext.ivf -o /tmp/ext.dec.y4m --threads 1 --strict 1
# compare /tmp/ext.rec.y4m vs /tmp/ext.dec.y4m per plane
```
Observed: encode rc 0, decoder rc 0 (stream is accepted), but recon != decoded:
`Y 65532/65536 diff (max 105), U 9298/16384 (max 32), V 8518/16384 (max 36)`.
Adding `--no-deblock` gives the same Y 65532 mismatch, so this is not a deblock
artefact. `--still-seg-delta 0` matches exactly. Each option alone (`--still-per-segment-q`,
`--still-ext-partitions`) matches exactly.
Expected: recon-vs-decoder sample equality, as with the plain partition tree.
Likely cause: `collect_node_ext()` builds each `BlockSyntax` but never copies
`options_.sb_segment[sb_index]` into `entry.segment_id`; only `collect_node()`
does that. `BlockSyntax::segment_id` defaults to 0, so the bitstream says segment 0
(`write_intra_segment_id` in `still_syntax.cpp:2543`) while `begin_superblock()`
already used `base_q_idx + per_segment_q_delta` for forward quant, dequant and the
deblock qindex (`fill_deblock_mi`, `:719-735`). The decoder therefore dequantises
and deblocks every extended-partition leaf as segment 0.

### SCOUT-DEBLOCK-02 — SDP two-tree walk drops the per-SB segment id
Repro:
```sh
B=/Users/arin/work/av2enc
$B/build/av2enc $B/testdata/shine_256x256_1f.y4m -o /tmp/sdp.ivf --frames 1 \
  --still --still-sdp --still-per-segment-q --still-seg-delta 40 \
  --recon /tmp/sdp.rec.y4m
$B/tools/bin/dav2d-macos -q -i /tmp/sdp.ivf -o /tmp/sdp.dec.y4m --threads 1 --strict 1
```
Observed: both rc 0, but `Y 65536/65536 diff (max 89), U 12105/16384 (max 27),
V 12096/16384 (max 29)`. `--still-sdp` and `--still-per-segment-q` are each exact
alone. `--still-sdp --still-per-segment-q --still-seg-delta 0` is exact.
Expected: recon == decoded.
Likely cause: same omission in `collect_sdp_node()` (`still_encoder.cpp:860-910`);
the function predates the per-segment-Q integration commit `ba7c53d`, which only
added the `entry.segment_id` block to `collect_node()`.

### SCOUT-DEBLOCK-03 — `--no-deblock` is silently ignored by the format still writers
Repro:
```sh
B=/Users/arin/work/av2enc
python3 $B/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py 192 128 /tmp/g10.y4m
$B/build/av2enc /tmp/g10.y4m -o /tmp/a.ivf  --frames 1 --still --bit-depth 10
$B/build/av2enc /tmp/g10.y4m -o /tmp/b.ivf  --frames 1 --still --bit-depth 10 --no-deblock
md5 /tmp/a.ivf /tmp/b.ivf          # identical: 3d00181844d475c3495ee08b46488471
```
Observed: both commands succeed (`rc 0`) and produce byte-identical streams.
The same holds for `--still --chroma 444` (identical MD5 with/without the flag);
422/400 follow the same branch.
Expected: like every other still-only modifier, `--no-deblock` should be rejected
by the format writer with "not applied by the format still writer" (the format
branch's rejection list covers the `--still-*` tools but omits `--no-deblock`),
or documented as a no-op for these paths.
Likely cause: the `format_10bit || format_chroma` validation in `src/main.cpp`
enumerates the unsupported still modifiers but does not include `--no-deblock`;
the writer hardcodes `apply_deblocking_filter[] = 0`
(`high_bitdepth_headers.cpp:194-196`) and all format paths have no deblock pass.

### SCOUT-DEBLOCK-04 — stale recon harness + gate gap around per-segment Q
Repro:
```sh
/Users/arin/work/av2enc/build/still_recon_dump -i /Users/arin/work/av2enc/testdata/shine_256x256_1f.y4m \
  -o /tmp/r.y4m --still --still-ext-partitions
# -> still_recon_dump: unknown option '--still-ext-partitions'
/Users/arin/work/av2enc/build/still_recon_dump ... --still --still-sdp   # same
```
Observed: `still_recon_dump` advertises that it "mirrors the src/main.cpp option
mapping for the CPU still path", but it rejects `--still-ext-partitions`,
`--still-sdp`, `--still-sub8`, `--still-ccso-edge`, `--still-wiener`,
`--still-pc-wiener`, `--still-chroma-lr`, `--still-gdf`, `--still-grain`,
`--still-delta-q`, `--still-dc-ac-delta-q` and `--still-cdef-on-skip-txfm`.
`gate_per_segment_q.sh` only runs plain `--still --still-per-segment-q` through
this harness, which is why SCOUT-DEBLOCK-01/02 are ungated. (`av2enc --recon`
does support all of these and can be used to restore the coverage.)
Likely cause: the harness's hand-written option parser was not extended when the
new still tools were added; the gate matrix pins only the default partition paths.

## Coverage (no fault found)
Exact recon-vs-`dav2d-macos` checks at 64x64, 100x70, 128x96, 256x256, 320x64,
640x360, 640x8, 8x640, 66x66 and 2x2..650x10, with: default; `--no-deblock`;
`--still-sub8`; `--still-tx-partition`; `--still-per-segment-q` (incl. qp 1/255,
delta 0/64); `--still-sb128/256`; `--still-cdef` (strengths 1/4/8, on-skip-txfm);
`--still-ccso[-edge]`; `--still-wiener` / `--still-chroma-lr` / `--still-pc-wiener`
/ `--still-gdf`; `--still-ibc`/`--still-palette`/`--still-mrl`/`--still-dip`/`--still-mhccp`/
`--still-ibp`; `--still-dc-ac-delta-q` (nonzero UV AC deltas);
`--still-segmentation`; `--still-delta-q`; and `--lossless` (decodes equal to
source). The §7.17 CPU core self-test also passes 562/562 checks (tile edges,
lossless sides, partial blocks, frame edges).
