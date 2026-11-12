# screen-content bug scout

Lane: palette (§5.20.8), IntraBC (§5.20.5.3/.4), SCC flags, `--still-palette` /
`--still-ibc` validation, and palette/IBC + transform/partition interactions.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-SCREEN-01 | med | doc | `--still-ibc` can never pick a block at width <= 128; the two narrow IBC gate cases pass with zero exact copies | see 01 | 128x384 flat-cell: `ibc: 0`; 130x384 same content: `ibc: 36`; `--still-ibc-search 1/4/64` still 0 | `src/pipeline/still/still_encoder.cpp:2609-2680`, `tools/still_verify/intra_gate.sh:322-323` | new |
| SCOUT-SCREEN-02 | low | confirmed | `PaletteModeInfo::cache_used` double-counts duplicate `PaletteCache` colours | see 02 | cache `[32,32,96,96]`, palette `[64,96]`: writer reports `cache_used=2`; analysis/bitstream reuse 1 | `src/codec/intra/palette/stream/palette_stream.cpp:16-32,312-314` | new |
| SCOUT-SCREEN-03 | low | doc | `--still-palette --still-ext-partitions` cannot palette the 4x16/16x4 leaves it creates, although §5.20.5.5 allows them | see 03 | trace shows `size=4x16 y=0` leaves; encoder predicate is `w>=8 && h>=8` | `src/pipeline/still/still_encoder.cpp:2718`, `src/codec/intra/palette/stream/palette_stream.cpp:106-116`, `src/codec/intra/screen_content/scc_flags.cpp:747-754` | new |
| SCOUT-SCREEN-04 | low | doc | `--still-ibc` rejects `--still-sb128/256` citing a "64x64-only" delay/ring model that the spec does not define that way | see 04 | spec §5.20.5.1 has `ibc_buffer_index` else-branch and `gradient = 4 + SbSize/64`; core implements `sb_size` | `src/pipeline/still/still_encoder.cpp:3831-3835`, `src/codec/intra/ibc/core/ibc_core.h:57-62,104-127`, `docs/m1-features.md:626-640` | new |

## Details

### SCOUT-SCREEN-01 — narrow frames never get an IBC block (and the gate does not notice)
Repro:
```sh
cd /Users/arin/work/av2enc
# 128x384, 32x32 constant cells (exact copies are available 5 SB64 rows above)
python3 - <<'PY'
def mk(p,w,h,bs=32,pal=(20,80,140,220)):
    y=bytearray(w*h)
    for r in range(h):
        for c in range(w):
            y[r*w+c]=pal[((r//bs)%2)*2+((c//bs)%2)]
    cw,ch=(w+1)//2,(h+1)//2
    open(p,'wb').write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n"%(w,h)+b"FRAME\n"+bytes(y)+bytes([128])*cw*ch*2)
mk('/tmp/n128.y4m',128,384); mk('/tmp/n130.y4m',130,384)
PY
./build/av2enc /tmp/n128.y4m --still --still-ibc -o /tmp/n128.ivf --still-ibc-search 64
./build/av2enc /tmp/n130.y4m --still --still-ibc -o /tmp/n130.ivf --still-ibc-search 64
```
Observed: 128-wide stream prints no `ibc:` line (0 exact-copy blocks); the
130-wide stream prints `ibc: 36 exact-copy block(s)` for identical content.
An independent call to `ibc::is_bv_valid(...sb_size=64...)` at (y=320,x=0),
32x32 block, BV `(-320,0)` returns 1, so a spec-legal source exists. The
encoder's only source candidates are the §7.12.2.20 defaults (vertical `-64`,
horizontal `-320`), the periodic list `{-8,-16,-32,-64,-128}`, and the
`--still-ibc-search` window `[-64,0]`; at 64/128-wide geometry
`INTRABC_DELAY_SB64` needs 5/3 SB64 rows, i.e. `-320` / `-192`, and the
vertical `-320` default is absent.

Gate hole: `ibc_scc64` and `ibc_scc128` are the same two narrow fixtures and
are run without `EXPECT_IBC=1`; re-running them reproduces the pinned MD5s
(`8df4b105...`, `63be14b4...`) and produces no `ibc:` count. Only
`ibc_384x128` sets `EXPECT_IBC=1`. Two of the five IBC gate cases therefore
exercise `use_intrabc = 0` syntax only.

Expected: either search the valid minimum source distance for narrow frames, or
document that `--still-ibc` is a no-op below 129 samples wide and assert
`EXPECT_IBC=1` only where IBC can occur.

### SCOUT-SCREEN-02 — `cache_used` disagrees with the coded cache bits
Repro (small harness; `libav2enc_core.a`):
```cpp
// build a writer at MI (8,2) whose above MI (7,2) and left MI (8,1) both hold
// the palette [32,96]; get_palette_cache then returns [32,32,96,96].
PaletteResult above; above.has_palette = true; above.size = 2;
above.colors[0] = 32; above.colors[1] = 96;
store_palette_state(writer.mutable_state(), 7, 2, 4, 4, above);
store_palette_state(writer.mutable_state(), 8, 1, 4, 4, above);
// 8x8 source: left half 64, right half 96; analyze -> size 2, colors [64,96]
writer.write(enc, cdfs, 8, 2, 3, 8, 8, 8, 8, &res, 0, &info, err);
```
Observed:
```
cache_n=4 [32,32,96,96]
res size=2 colors=[64,96] cache_used=1
info.cache_used=2 (analysis result says 1)
```
Expected: `info.cache_used` equals the number of cache entries actually coded
as used. `PaletteModeInfo::cache_used` is documented as "number of reused cache
colours" (`palette_stream.h:129`).

Likely cause: `count_cache_reuse()` advances `idx` on every match, but
`split_cache_and_run()` consumes palette colours *by value* and never reuses a
cache entry twice. With `cache=[96,64]` it also under-counts. The bitstream is
unaffected; only the reported metadata (and any future user of it) is wrong.

### SCOUT-SCREEN-03 — palette eligibility is stricter than §5.20.5.5 for uneven leaves
Repro:
```sh
cd /Users/arin/work/av2enc
AV2ENC_STILL_TRACE=1 ./build/av2enc testdata/m0_64x64_1f.y4m --still \
  --still-ext-partitions --still-palette -o /tmp/p.ivf 2>&1 |
  grep -E 'size=(4x16|16x4)'
```
Observed: `--still-ext-partitions` emits DC luma leaves such as `size=4x16
y=0` / `size=16x4 y=0`. The palette decision skips them because
`still_encoder.cpp:2718` requires `w >= 8 && h >= 8`. The adapter's own rule
(`palette_stream.cpp:106-116`) and the SCC legality helper
(`scc_flags.cpp:747-754`) accept `bsize >= BLOCK_8X8` and dimensions <= 64,
which is the §5.20.5.5 predicate using numeric `enum_block_size` order
(`4x16` = 19, `16x4` = 20). The same applies to `4x32`/`32x4`.

Expected: use one predicate. As written, `--still-ext-partitions --still-palette`
is accepted but cannot use palette on a subset of leaves that the spec and the
adapter call palette-signallable. This is a missed-tool/consistency gap, not a
conformance failure (the encoder may always decline palette).

### SCOUT-SCREEN-04 — the 64x64-only IBC rejection overstates the spec
Repro:
```sh
./build/av2enc tests/data/basic_64x64_1f.y4m --still --still-ibc --still-sb128 -o /tmp/i.ivf
```
Observed: `av2enc: error: --still-ibc requires 64x64 superblocks (drop
--still-sb128/--still-sb256)`, with source comment "§5.20.5.1 ring/delay model
is defined for 64x64 superblocks" (`still_encoder.cpp:3831-3835`) and
`docs/m1-features.md` saying the model is 64x64-only.

Expected: the spec defines `ibc_buffer_index(buf_row,buf_col,sb_size)` for
`SbSize != BLOCK_64X64` and uses `gradient = INTRABC_DELAY_SB64 +
(Block_Width[SbSize]/64)` in `is_mv_valid`; `ibc_core.h:57-62,104-127` already
implements both. IBC at 128x128 SBs is either supportable or an explicit
unsupported-feature limitation; the current text claims a spec restriction that
does not exist. (The global-only production config makes the ring argument
moot.)

## Coverage / dedup
- Exact recon vs `dav2d --strict 1` over palette/IBC on 64x64..1024x384,
  odd widths 66..194, partial edges, depth 0..8, QP 1..255, sb128/256 (palette),
  tx-partition, TCQ, IST, CCTX, IDTX, QM, RDOQ, tx-type, RDO-intra, ext/sub8,
  per-segment-Q, delta-Q, GDF/Wiener/CDEF/CCSO: no stream or recon mismatch
  found. 20k random palette-core and 5k random IBC MV-syntax round-trips also
  passed.
- IBC + CDEF is **duplicate**: `bug-reports/cdef-ccso.md` SCOUT-CDEF-01 already
  covers the `skip_flag`/on-skip-txfm failure. IBC + CCSO was exact there too.
- `refactor/bugs.csv` and `git log --all --grep='palette|ibc|screen'` have no
  entries for the four items above.
