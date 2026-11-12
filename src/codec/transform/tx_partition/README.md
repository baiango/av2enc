# Transform partition / `TX_MODE_SELECT` (`src/codec/transform/tx_partition/`)

Status: **component + integrated still path.** The CPU `--still-tx-partition`
path (`src/pipeline/still/still_encoder.cpp` + `still_syntax.cpp`) now selects
NONE/SPLIT/HORZ/VERT per luma coding block, writes §5.20.6.3
`tx_do_partition` / `tx_partition_type` before the coefficients, and runs
prediction, transform, quantisation and reconstruction per transform part in
decode order. `BlockSyntax` carries a per-part luma TX list
(`tx_part_count`/`tx_partition`/`tx_parts[]`); chroma keeps one transform per
plane. The decoder gate is `still_tx_partition_gate.sh`: `--still-tx-partition`
is default-off, the pinned 64x64/640x360 `--still` streams are byte-identical,
and the natural plus forced NONE/SPLIT/HORZ/VERT streams decode sample-exactly
with `dav2d --strict 1`. TX partitioning composes with RDOQ, TX_TYPE, IST, TCQ,
CCTX and RDO-intra; QM, DC/AC delta-Q, DIP, MRL and IBP are rejected by name.

The rest of this document describes the standalone component.

Implements the AV2 v1.0.0 transform-mode and transform-partition signalled by
§5.18.8.1, §5.20.6.1, §5.20.6.2 and §5.20.6.3 — the syntax family the
`08-transform.md` audit lists as **entirely ABSENT** (row X2, §12.1 items 1–2)
and that `04-tile-block.md` ranks as the **#1 missing block-level element**:

* `read_tx_mode()` — `tx_mode_select`, `TxMode` `LARGEST` / `SELECT` / `ONLY_4X4`
* `read_tx_size(allowSelect)` — `maxRectTxSize`, 64x64 chunking, partition
* `read_block_tx_size()` — the inter non-skip driver, per-64x64-chunk
* `read_tx_partition()` — `tx_do_partition`, `tx_partition_type`,
  `tx_2or3_partition_type`, and the normative `find_tx_size()` /
  `set_tx_size()` geometry

It also adds the §5.20.7.30 scan generation the audit found missing from
`spec/tables/scan.json`: all 25 transform sizes and all three `TX_CLASS`
values, including rectangular sizes and `HORIZ`/`VERT` scans.  A scan is not
optional for this feature — a split transform block needs the right scan to
code its coefficients.

Everything here is a **component**: no shipped encoder path calls it.  The gate
(§“dav2d gate”) proves the syntax is decodable and reconstructs bit-exactly
using build-directory patched copies of `video_syntax.cpp` / `video_headers.cpp`;
the repository sources are untouched.  The core edits needed for a production
integration are listed in §“M1/M2 call sites”.

## Status vocabulary used below

| Word | Meaning |
|---|---|
| **WIRED** | A shipped encoder path emits the syntax and a decoder gate exists. |
| **COMPONENT** | Tested implementation, no shipped caller. |
| **GATE** | Exercised only through the patched build in `gate/`. |

This component is **COMPONENT + GATE**, never WIRED.

## What the spec allows

`read_tx_partition()` is a **single-level** split of `Max_Tx_Size_Rect[MiSize]`
(confirmed against the PDF, page 184: no recursive call).  Four signalling
shapes exist:

| Shape | Symbol(s) | Parts | Legal when |
|---|---|---|---|
| `NONE` | `tx_do_partition = 0` | 1 | always, if the syntax block runs |
| `SPLIT` | `tx_do_partition=1`, `tx_partition_type=0` | 4 (`w/2 × h/2`) | both halvings are TX sizes |
| `HORZ` / `VERT` | `tx_do_partition=1`, `tx_partition_type=1/2` | 2 | respective halving is a TX size |
| `HORZ4` / `VERT4` | `tx_do_partition=1`, `tx_2or3_partition_type=1` | 4 | exactly one direction halves **and** `Size_To_Tx_Type_Group_Vert_Or_Horz[MiSize] > 0` |

`TX_PARTITION_HORZ5` / `VERT5` are in Table 6.23 and in `set_tx_size()`, but no
syntax path can select them; `tx_partition_geometry()` rejects them explicitly.

The syntax block itself is skipped when `MiSize == BLOCK_4X4`, when
`TxMode != TX_MODE_SELECT`, when `allowSelect == 0`, or when the coding block
exceeds 64 in either dimension.  Blocks >64 are chunked into 64x64 transforms
with **no partition bits** (`read_tx_size` / `read_block_tx_size` chunk loops);
`legal_splits()` reports `chunked = true, applicable = false` for them.

### `tx_partition_type` vs `tx_2or3_partition_type`

The branch choice is mechanical (§5.20.6.3):

```
allowHorz = find_tx_size(Tx_W, Tx_H/2) >= 0
allowVert = find_tx_size(Tx_W/2, Tx_H) >= 0
if (allowHorz && allowVert)        -> tx_partition_type (7 symbols, 0..2 used)
else if (group_vert_or_horz > 0)   -> tx_2or3_partition_type (2 symbols)
else                               -> direction is implied by tx_do_partition
```

Only `BLOCK_4X16`, `BLOCK_16X4`, `BLOCK_4X32`, `BLOCK_32X4` have
`group_vert_or_horz > 0`, so `HORZ4`/`VERT4` are only reachable on 4-wide or
4-high coding blocks (library-only `min_block_size = 4`).

### CDFs and contexts (§9.3, §8.3.2)

| Symbol | CDF | Context |
|---|---|---|
| `tx_do_partition` | `TileTxDoPartitionCdf[fsc_mode][is_inter][group]` | `Size_To_Tx_Part_Group_Lookup[MiSize]` (0..8) |
| `tx_partition_type` | `TileTxPartitionTypeCdf[fsc_mode][is_inter][group]` | `Size_To_Tx_Type_Group_Vert_And_Horz[MiSize]` (0..13) |
| `tx_2or3_partition_type` | `TileTx2or3PartitionTypeCdf[fsc_mode][is_inter][group-1]` | `Size_To_Tx_Type_Group_Vert_Or_Horz[MiSize] - 1` (0..1) |

`TxPartitionCdfs::reset(reduced_tx_part_set)` copies the §9.3 defaults and
selects `Default_Tx_Partition_Type_Reduced_Cdf` (uniform 4681… boundaries)
when `reduced_tx_part_set = 1`, exactly like `init_tile()` (§9.3 p329).  The
four generated tables are checked into `tx_partition_tables.h`; regenerate
with `generate_tx_partition_tables.py`, which re-validates shape, rate indices
and use counts against `spec/tables/default_cdf.json`.

## API

```cpp
#include "encoder/m2/tx_partition/tx_partition.h"

// geometry (§9.2)
int max_tx_size_rect(int mi_size);
int find_tx_size(int w, int h);              // -1 == TX_INVALID
LegalSplits legal_splits(int mi_size, bool reduced_tx_part_set);
bool tx_partition_geometry(int mi_size, int mi_row, int mi_col,
                           TxPartition part, SplitGeometry* out);

// one read_tx_partition() call
bool write_tx_partition(SymbolSink&, TxPartitionCdfs&, const BlockTxContext&,
                        TxPartition decision, SplitGeometry*, std::string&);
bool read_tx_partition(SymbolSource&, TxPartitionCdfs&, const BlockTxContext&,
                       int mi_row, int mi_col, TxPartition*,
                       SplitGeometry*, std::string&);

// whole-block drivers (read_tx_size / read_block_tx_size and mirrors)
bool write_tx_size(...);   bool read_tx_size(...);
bool write_block_tx_size(...); bool read_block_tx_size(...);

TxMode tx_mode_from_syntax(bool coded_lossless, bool tx_mode_select);
```

`SymbolSink` / `SymbolSource` keep the syntax independent of the range coder;
`tx_partition_entropy.h` supplies `av2::Encoder`/`Decoder` adapters plus
`TileBitWriter`/`TileBitReader` for tests and the gate.  `LumaTxGrid` models
`LumaTxSizes` / `LumaTxMiddle` / `LumaTxScanOrder` and is what the drivers
fill; `middle` and `scan_order` stay 0 for every partition the syntax can
select (only the unreachable HORZ5/VERT5 would set them).

Parts in `SplitGeometry` are in **decode order** (the order `residual()` visits
them): row-major for `SPLIT`/`HORZ`/`VERT`, top-to-bottom for `HORZ4`,
left-to-right for `VERT4`.  A coefficient writer can therefore walk
`geometry.parts[]` directly.

### Scans (§5.20.7.30)

```cpp
#include "encoder/m2/tx_partition/tx_scan.h"
bool get_scan(int tx_size, TxScanClass cls, std::span<uint16_t> out);
bool get_tx_row_col(int pos, int tx_size, int* row, int* col);
bool get_lf_limits(int row, int col, TxScanClass cls, int plane);
TxScanClass tx_class_of(int tx_type);   // IDTX is TX_CLASS_2D
```

The scan grid is `Min(Tx_Width,32) × Min(Tx_Height,32)`, matching
`get_tx_row_col()` (which uses `Tx_Width_Log2[Adjusted_Tx_Size[txSz]]`).
`get_scan()` for `TX_CLASS_VERT` is row-major and for `TX_CLASS_HORIZ`
column-major — that is the literal §5.20.7.30 loop nesting, not a naming
intuition.

## Tests

```
./src/encoder/m2/tx_partition/build_and_test.sh
```

**107011 checks, 0 failures** (`-O2`, `WERROR=1` and `SANITIZE=1` clean).
Suites:

* **tables** — §9.2 values, §9.3 spot values, full/reduced variant, pinned
  FNV-1a fingerprint `0x59b1af837dc95e25`.
* **planner** — all 29 `BLOCK_SIZES` × `reduced_tx_part_set` 0/1; option
  uniqueness, applicability, chunking, and the exact option set per branch.
* **geometry** — every legal `(MiSize, partition)`: exact coverage (each 4x4
  cell once), in-block bounds, decode order, and `part.tx_size ==
  find_tx_size(w,h)`.
* **partition round-trip** — 624 exhaustive writer→independent-reader cases:
  every legal option × 29 block sizes × `fsc_mode` × `is_inter` × reduced set.
* **tx-size drivers** — `read/write_tx_size` and `read/write_block_tx_size`
  for every block size × TX_MODE_LARGEST/SELECT × inter/intra × skip/non-skip,
  comparing the full `LumaTxGrid`.
* **reachability** — all 25 TX sizes are reachable; in particular
  `BLOCK_8X64 VERT → TX_4X64` and `BLOCK_64X8 HORZ → TX_64X4` (see
  “Audit correction”).
* **scans** — 25 sizes × 3 classes: permutation, HORIZ/VERT structure, square
  2D equality with the committed `kScan4/8/16/32`, rectangular 2D equality
  with an independent anti-diagonal traversal, `get_tx_row_col` inverse, LF
  limits, class mapping.
* **cdf adaptation** — adaptive write/read with a mixed symbol sequence; the
  decoder must reproduce the final CDF fingerprint.
* **bitstream goldens** — 8 pinned hex payloads covering `NONE`, `SPLIT`,
  `HORZ`, `VERT`, `HORZ4`, reduced set, `fsc_mode`, inter/intra, and a
  `BLOCK_8X64` `VERT` (TX_4X64).

### Bit cost (first symbol, CDF frozen, exact range-coder cost)

| Block | NONE | SPLIT | HORZ | VERT | 4-way |
|---|---:|---:|---:|---:|---:|
| 8×8 | 0.15 | 3.40 | 14.50 | 14.50 | — |
| 16×16 | 0.60 | 4.23 | 2.78 | 3.49 | — |
| 32×32 | 1.09 | 3.38 | 2.11 | 3.57 | — |
| 64×64 | 1.15 | 4.19 | 2.19 | 3.52 | — |
| 4×16 | 1.25 | — | 1.18 | — | 2.90 (`HORZ4`) |
| 8×16 | 0.69 | 4.84 | 2.14 | 4.37 | — |
| 8×64 | 1.25 | 4.50 | 1.96 | 4.47 | — |
| 64×8 | 1.25 | 4.63 | 4.70 | 1.55 | — |

These are syntax costs only; each extra transform additionally pays its own
TX_TYPE/EOB/coefficient bits.  The 8×8 `HORZ/VERT` cost is high because
`Default_Tx_Do_Partition_Cdf[0][0][1]` strongly prefers splitting.

## dav2d gate

```
./src/encoder/m2/tx_partition/gate/build_gate.sh
```

**TX PARTITION GATE PASS: 7/7** with `dav2d --strict 1 --threads 1`, every
luma/chroma sample identical to the encoder reconstruction:

| Case | Coding block | Partition | Transform blocks | tile1 |
|---|---|---|---|---:|
| A | 64×64 | SPLIT | 4× `TX_32X32` | 916 B |
| B | 16×16 | SPLIT | 4× `TX_8X8` | 115 B |
| C | 8×8 | SPLIT | 4× `TX_4X4` | 103 B |
| D | 16×16 | HORZ | 2× `TX_16X8` | 109 B |
| E | 16×16 | VERT | 2× `TX_8X16` | 112 B |
| F | 16×16 | NONE (control) | 1× `TX_16X16` | 20 B |
| G | 16×16 | SPLIT + `tx_mode_select=0` (negative control) | — | decoder rejects the P frame |

The gate is a deterministic 2-frame 64×64 4:2:0 stream: frame 0 is an
all-zero DC intra keyframe (recon 128), frame 1 is a P frame with
`tx_mode_select = 1` whose one non-skip inter block (zero `GLOBALMV`) signals
the partition and codes real non-zero coefficients in every resulting
transform block.  See `gate/README.md`.

## M1/M2 call sites

These are the edits a production integration needs.  They are *not* made here
(ownership split).  The gate implements them in build-directory copies.

**M2 (P frames, the primary target)**

1. `src/encoder/m2/video_headers.cpp:289` — write `tx_mode_select` from the frame
   decision instead of hard-coded `false`.  It is a **per-frame** field; a
   keyframe can stay `TX_MODE_LARGEST`.
2. `src/encoder/m2/video_syntax.cpp` — `write_inter_residual()` currently calls
   `code_tx(node, 0, …)` once for the whole luma block.  It must first emit
   `write_tx_partition()` and then call `code_tx()` once per
   `SplitGeometry::parts[]`, with `sub.mi_row/mi_col`, `tx_size`, `eob` and
   `levels_offset` (the gate’s patched `txgate_code_luma()` is the reference).
   For skip blocks nothing changes: `read_block_tx_size()` takes the
   `read_tx_size(false)` path and codes no bits.
3. `src/encoder/m2/video_syntax.cpp` — add `TxPartitionCdfs` to the tile `CdfState`
   and initialise it in `init_cdfs()` with `reset(reduced_tx_part_set)`.
   The gate keeps one static copy because its tile has exactly one split block.
4. `src/encoder/m2/residual/inter_residual.cpp` — the decision and
   reconstruction must be per transform block: RDO over `legal_splits()`
   options, forward/quant/inverse per part, and `txv::reconstruct_block` per
   part.  This is the same code path DDT needed (`inter_residual.cpp:280/338`).
5. `src/encoder/m2/rdo/residual_decision.cpp` — include the syntax cost from
   `write_tx_partition()` in the rate term; otherwise the decision will
   over-split.
6. `src/encoder/m2/video_frame.cpp:~790/825` — a leaf currently stores one
   `tx_size/eob` per plane.  It needs a per-part list (the gate uses
   `GateSplitPlan` as a stand-in).

**M1 (intra stills)**

1. `src/encoder/m1/still_headers.cpp:258` — `tx_mode_select`.
2. `src/encoder/m1/still_encoder.cpp:1817` and `still_syntax.cpp:1389-1400` — replace
   the “TX size == block size” assertion with `write_tx_size()` per leaf
   (intra uses `read_tx_size(!skip_flag || !is_inter)`).
3. `src/encoder/m1/still_encoder.cpp:1124-1129` — chroma TX stays the block’s chroma
   residual size (`get_tx_size(plane, …)` ignores the luma partition), so only
   the luma residual loop changes.
4. Reconstruction (`still_encoder.cpp:765-787`) must run once per split part.  The
   transform component already has the right per-size `Transform_Shift`,
   `Adjusted_Tx_Size`, `dqDenom` and inverse; nothing in `tx_variants` changes.

**Interaction notes**

* **`reduced_tx_set` vs `reduced_tx_part_set`** — different sequence bits.
  `reduced_tx_set` (always 0 in M1/M2) selects TX_TYPE sets; it does not affect
  TX partitioning.  `reduced_tx_part_set` (sequence bit at
  `video_headers.cpp:196`, currently 0) removes `HORZ4`/`VERT4` from
  `tx_2or3_partition_type` and switches `tx_partition_type` to the uniform
  §9.3 table.  Both are modelled and tested; the gate uses the full set.
* **IST (`sec_tx_type`)** — IST is a per-transform-block tool.  With TX
  partitioning each part has its own `eob` and its own `sec_tx_type`
  applicability test (`eobLim` depends on that part’s TX size).  The encoder
  cannot enable IST today (`enable_intra_ist=enabled_inter_ist=0`); when it
  does, the IST decision must live inside the per-part coefficient loop that
  this component creates, not around the coding block.  The TX partition
  syntax itself is orthogonal: `sec_tx_type` is coded after the part’s
  `transform_type()` and before its coefficients.
* **TCQ** — two per-part effects.  (a) The §7.14.4 extra `dqDenom` shift is
  keyed on `get_tx_class(PlaneTxType) == TX_CLASS_2D` and applies per
  transform block; smaller parts change `pels` and therefore the base
  `(pels>256)+(pels>1024)` shift, so TCQ must be applied part-by-part.
  (b) TCQ levels are per part; the level mapping and parity hiding do not
  cross part boundaries.  `enable_tcq=0` in every writer today, and the audit’s
  `qm_core.h` IDTX/TCQ predicate bug was fixed in commit `e7bec71`.
* **Chroma** — `get_tx_size(plane, txSz)` for `plane > 0` returns
  `Max_Tx_Size_Rect[get_plane_residual_size(ChromaMiSize, plane)]` and ignores
  the luma TX size.  TX partitioning is **luma-only**; chroma keeps one
  transform per plane over the chroma residual area.  The gate codes chroma
  with all-zero levels to keep the comparison focused, but the syntax is
  unchanged either way.
* **Lossless** — `read_tx_size()`’s `Lossless` branch needs the
  `lossless_tx_size` CDF, which is outside this component.  The inference
  cases (`MiSize == BLOCK_4X4`, intra without FSC, `allowSelect == 0`) are
  modelled; a stream that would need the symbol is rejected with a clear error
  rather than mis-signalled.

## Audit correction: TX_4X64 / TX_64X4 *are* signallable

`08-transform.md` §12.1 says `TX_4X64`/`TX_64X4` have no `BlockSize` and are
never signallable.  That is true for `TX_MODE_LARGEST`, but false once
`TX_MODE_SELECT` exists:

* `BLOCK_8X64` (max `TX_8X64`): `allowVert` because `find_tx_size(8/2, 64) =
  TX_4X64`; `VERT` produces 2× `TX_4X64`.
* `BLOCK_64X8`: `HORZ` produces 2× `TX_64X4`.

Likewise `TX_4X32`, `TX_32X4`, `TX_4X16`, `TX_16X4` become reachable as **luma**
transforms from `BLOCK_8X64`/`BLOCK_64X8`/`BLOCK_8X32`/`BLOCK_32X8`, not only as
chroma.  `tx_partition_test` asserts all 25 sizes are reachable.  The `BLOCK_8X64 VERT`
golden pins the syntax, but the 64-wide/high adjusted
transform is reconstructed by `tx_variants` and is covered by **G-TXV**; the
dav2d gate uses the 16×16 and 64×64 families to keep the stream small.

## Files

```text
tx_partition.h/.cpp          syntax, contexts, CDFs, geometry, drivers
tx_partition_tables.h        generated §9.2/§9.3 tables (checked in)
generate_tx_partition_tables.py
tx_scan.h/.cpp               §5.20.7.30 scans, row/col, LF limits
tx_partition_entropy.h       av2::Encoder/Decoder adapters + tile helpers
tx_partition_test.cpp        component test (107011 checks)
build_and_test.sh
gate/README.md               gate design + blocked core edits
gate/build_gate.sh
gate/tx_partition_gate.cpp   stream builder, recon, dav2d comparison
gate/tx_partition_gate_hook.h/.cpp
gate/patch_video_syntax.py      build-dir copy patch (3 anchored edits)
gate/patch_video_headers.py     build-dir copy patch (2 anchored edits)
```
