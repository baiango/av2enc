# encoder/m1/sub8 — sub-8 coding blocks (4xN / Nx4)

Reference rules, exact minimal syntax, and `dav2d` gate for M1 luma coding
blocks smaller than 8x8. This directory is **additive**: `src/encoder/m1/still_*`,
`src/encoder/partition/`, `src/main.cpp` and `CMakeLists.txt` are not modified.
The M1 writer changes that are still required are listed under
[M1 writer contract](#m1-writer-contract).

- Spec: AV2 Bitstream & Decoding Process Specification v1.0.0
  (§5.20.3.1, §5.20.3.2, §5.20.5.3, §5.20.5.6, §5.20.7.23–27, §8.3.2, §9.2).
- No AVM, SVT-AV1/AV2, clave or av2-turbo code was consulted.
- Status: rules + writer/reader tests pass; the minimal sub-8 streams **decode**
  with the pinned `dav2d` `f9dafbc`. The production M1 writer still rejects
  sub-8 trees; root cause and call sites below.

## Root cause of the decoder rejection

`dav2d` reports

```
Error decoding frame: Invalid argument
No data decoded
```

and `avmdec` reports `Corrupt frame detected / Failed to decode tile data`.
This is a **tile symbol desynchronisation**, not a header problem.

`src/encoder/m1/still_syntax.cpp::write_block()` derives chroma from the *luma
leaf*:

```cpp
const BlockSize chroma_size =
    partition::chroma_plane_residual_size(geometry_, size);  // leaf size
...
if (!code_uv_mode(...)) return false;
level_base += node.eob[0];
code_tx(node, 1, ...);   // U
code_tx(node, 2, ...);   // V
```

That is correct at 8x8 and above, but wrong for every sub-8 leaf:

1. **Chroma is coded by the wrong leaf.** `decode_partition()` (§5.20.3.1)
   passes `hasChroma && !chromaOffset` to the first child and `hasChroma` to
   the second. For a 4:2:0 sub-8 split `chromaOffset` is already 1, so the
   first *k−1* leaves of an 8x8 chroma group have `HasChroma = 0`: they code
   **no** `is_cfl`, **no** `uv_mode`, and **no** chroma `all_zero`. Only the
   last leaf in decode order (bottom-right 4x4, or the surviving 4x8/8x4)
   codes the chroma mode and residual for the whole inherited chroma block.
   The current writer codes UV on every leaf, so the first sub-8 leaf is
   already desynchronised (the decoder reads `is_cfl` where it expects the
   luma `txb_skip`).
2. **Chroma is coded at the wrong origin and size source.** The owning leaf is
   not the chroma block origin. The chroma block is
   `ChromaMiRow/ChromaMiCol/ChromaMiSize` captured at the last ancestor whose
   `chromaOffset` was 0 (normally the 8x8 parent). Chroma residual size is
   `get_plane_residual_size(ChromaMiSize, plane)` and the coefficient/CfL
   neighbour contexts are indexed at `ChromaMi >> 1`. Indexing at
   `leaf.mi_row >> 1`/`leaf.mi_col >> 1` reads the wrong neighbour rows/columns
   even when the owner is correct. The production writer does this in
   `cfl_context()` and `code_tx()`.

The chroma *residual size* happens to coincide (4x4) for 4x4/4x8/8x4 leaves,
which is why the size check inside `write_block` does not catch it: the bug is
ownership + origin, not size. (It does diverge for other sub-8 shapes, e.g. a
4x16 leaf needs `get_plane_residual_size(ChromaMiSize)`, not the leaf size.)

The reproduction is `repro_still_sub8.cpp` (`REPRO=1 build_and_test.sh`): an 8x8
checkerboard with `StillOptions::min_block_size = 4` produces the tree

```
64x64 HORZ -> 64x32 VERT -> 32x32 HORZ -> 32x16 VERT
      -> 16x16 HORZ -> 16x8 VERT -> 8x8 HORZ -> 8x4 VERT (x2)
      -> four 4x4 leaves
```

Every one of the four 4x4 leaves is written with `uv=0/0`; the decoder wants
chroma only on the last one. `dav2d` rejects the stream.

## What the spec actually does for sub-8 blocks

### Partition tree (§5.20.3.2)

- `rect_type_implied_by_bsize()` is the controlling rule: a **4x8 / 4x16** block
  only allows `PARTITION_HORZ`, an **8x4 / 16x4** only `PARTITION_VERT`, and
  `rect_type` is not coded at those nodes. 4x4 has no partition at all.
- `partition_implied()` returns `PARTITION_NONE` for 4x4 and for
  `BLOCK_4X32`/`32X4`/`8X64`/`64X8`; those nodes code no partition syntax.
- `PARTITION_SPLIT` does not exist below a 128x128 root, so the M1 64x64
  superblock reaches sub-8 sizes only through HORZ/VERT.
- Aspect: `MaxPbAspectRatio` is 8; `reduce_pb_aspect_ratio = 0` in the M1
  sequence header. A sub-8 partition whose subsize would exceed the aspect is
  rejected unless the fallback forces `PARTITION_NONE`.
- `PARTITION_NONE` at a sub-8 node still has to pass the half-block test
  (`(r + numHigh4x4/2) < MiRows`, same for columns). At the bottom/right coded
  edge this can reject both NONE and the only rect direction; the spec then
  falls back to `allowed[PARTITION_NONE] = 1`, so the node is a forced leaf.
  `plan_partition()` reports that as `forced`.
- Chroma-aware rejection: when `hasChroma && chromaOffset`, HORZ/VERT are only
  allowed if `block_coded()` of the chroma-carrying child is true. This is why
  a 4x8 at the bottom coded row is a forced leaf.

### Mode info (§5.20.5.3 / §5.20.5.6)

- `read_intra_y_mode()` is called for **every** leaf, sub-8 included. There is
  no 8x8 mode grouping in AV2.
- `read_intra_uv_mode()` is called **iff `HasChroma`**, i.e. only on the leaf
  that owns the inherited 8x8 chroma block.
- `cflAllowed` uses `get_plane_residual_size(ChromaMiSize, 1)`, not the luma
  leaf. For 4:2:0 sub-8 that is 4x4, so `is_cfl` is present on the owner.
- `is_cfl` context uses `UVCfls[ChromaMiRow-1][ChromaMiCol]` and
  `[ChromaMiRow][ChromaMiCol-1]` (`ChromaMi` origin).
- Palette is only allowed when `MiSize >= BLOCK_8X8`; M1 does not enable
  screen-content tools. DIP/DPCM/FSC/MRL are disabled by the M1 header.
- The UV-mode candidate list omits the neighbour-derived directional modes
  when `MiSize < BLOCK_8X8`, so `uv_mode_index()` must use the plain
  `Default_Mode_List_Uv` path on sub-8 leaves. M1's existing UV index code is
  already correct for this; the DC-only gate pins it.

### Chroma ownership (§5.20.3.1)

At every node, before the partition branch:

```text
if (!chromaOffset && hasChroma) {
  chromaOffset = is_chroma_offset_for_partition(partition, bSize)
  ChromaMiRow = r; ChromaMiCol = c; ChromaMiSize = bSize
}
```

Then HORZ/VERT pass `(chromaOffset, hasChroma && !chromaOffset)` to the first
child and `(chromaOffset, hasChroma)` to the second; SPLIT passes
`chromaOffset = 0` but does not reset `ChromaMi*`. Consequently:

| tree | chroma owner | ChromaMiSize | chroma residual |
|---|---|---|---|
| 8x8 NONE | the 8x8 | 8x8 | 4x4 |
| 8x8 VERT -> 4x8,4x8 | second 4x8 | 8x8 | 4x4 |
| 8x8 HORZ -> 8x4,8x4 | second 8x4 | 8x8 | 4x4 |
| 8x8 VERT, first 4x8 -> HORZ | the 4x8 at (0,1) | 8x8 | 4x4 |
| 8x8 VERT, second 4x8 -> HORZ | bottom-right 4x4 | 8x8 | 4x4 |
| 8x8 VERT -> HORZ on both 4x8 | bottom-right 4x4 | 8x8 | 4x4 |
| 16x8 VERT -> two 8x8 -> HORZ | second 8x4 of each 8x8 | 8x8 per group | 4x4 each |

`sub8.h::assign_chroma()` / `build_superblock()` implement this recursion and
`sub8_test.cpp` cross-checks it against `partition::validate_tree()`.

### Coefficient contexts and transform selection

- `all_zero` (the `txb_skip` symbol) uses
  `bsize = get_plane_residual_size(plane > 0 ? ChromaMiSize : MiSize, plane)`.
  Luma: `ctx = 0` when `bsize` equals the transform size (TX_MODE_LARGEST), else
  `(min(top,4) + min(left,4) + 3) >> 1`.
  U: `ctx = 6 + (above != 0) + (left != 0)`.
  V: `ctx = 3 * (bsize > tx) + 6 * (EobU != 0) + (above != 0) + (left != 0)`.
  The `EobU` term is the one that differs from U and is easy to get wrong.
- Neighbour level/DC contexts are indexed at the transform origin: luma leaf
  origin for Y, `ChromaMi >> 1` for U/V. They are cleared over the transform
  footprint after every leaf, including all-skip leaves.
- TX_MODE_LARGEST: luma `Max_Tx_Size_Rect[MiSize]`, chroma
  `Max_Tx_Size_Rect[get_plane_residual_size(ChromaMiSize, 1)]` (TX_4X4 for
  sub-8 4:2:0). 4x4 uses eobMultisize 0 (`eob_pt_16`), 4x8/8x4 use 1
  (`eob_pt_32`).
- `transform_type()` is DCT_DCT when `eob <= 1`; for larger EOBs 4x4/4x8/8x4
  use `TX_SET_INTRA_1` (7 symbols). `enable_chroma_dctonly = 1` keeps chroma
  DCT_DCT.
- TX_MODE_SELECT would add `tx_do_partition` on 4x8/8x4; M1 uses
  TX_MODE_LARGEST, so no transform-tree syntax is present.

## Files

| File | Purpose |
|---|---|
| `sub8.h` / `sub8.cpp` | predicates, partition plan, chroma ownership, mode-info plan, `txb_skip` context, TX selection, tree builder |
| `sub8_syntax.h` / `sub8_syntax.cpp` | exact tile writer + independent reader for all-DC / zero-residual frames |
| `sub8_minimal.h` / `sub8_minimal.cpp` | complete AV2 temporal units (reuses `still_headers.cpp`) |
| `gate_minimal.cpp` | writes ten IVF streams with sub-8 leaves |
| `gate_dav2d.py` | runs the pinned decoder and checks all-128 output |
| `repro_still_sub8.cpp` | production-M1 probe (diagnostic, `REPRO=1`) |
| `sub8_test.cpp` | 215k checks: sweep, chroma, mode info, contexts, TX, goldens |
| `build_and_test.sh` | standalone build, tests, gate |

```sh
./src/encoder/m1/sub8/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/m1/sub8/build_and_test.sh
SKIP_GATE=1 ./src/encoder/m1/sub8/build_and_test.sh
REPRO=1 ./src/encoder/m1/sub8/build_and_test.sh
```

Self-contained: the script compiles `partition.cpp`, `still_headers.cpp`,
`bitwriter.cpp`, `obu.cpp`, `basic_headers.cpp`, `tx_variants.cpp` and
`av2_entropy.cpp` directly. It does not touch the root CMake project.

## API

```cpp
#include "encoder/m1/sub8/sub8.h"
using namespace av2enc::still::sub8;

bool is_sub8(BlockSize);          // 4x4/4x8/8x4/4x16/16x4/4x32/32x4
int  min_dim_px(BlockSize);       // 4 / 8 / ...

PartitionPlan plan_partition(geometry, mi_row, mi_col, size,
                             has_chroma, chroma_offset);
//   allowed, implied, rect_type, forced/forced_partition,
//   codes_do_split / codes_do_square_split / codes_rect_type,
//   none_allowed, allow_horz/allow_vert

struct Node {                     // in-frame children only, decode order
  int mi_row, mi_col; BlockSize size; PartitionType partition;
  bool forced, has_chroma, chroma_offset;   // *incoming* recursion args
  ChromaInfo chroma;                        // active ChromaMi*
  std::vector<Node> children;
};

bool build_superblock(geometry, sb_mi_row, sb_mi_col,
                      PartitionType(*choose)(const Node&, const PartitionPlan&),
                      Node& out, std::string& error);
void assign_chroma(geometry, has_chroma, chroma_offset, inherited, Node&);
void collect_leaves(const Node&, std::vector<const Node*>&);

ModeInfoPlan mode_info_plan(geometry, leaf, enable_cfl_intra);
int  txb_skip_context(plane, plane_bsize, tx_lw, tx_lh,
                      above, left, u_has_cf);
TxPlan tx_plan(geometry, leaf, eob_luma);
int  eob_multisize(int tx_size);
bool tx_type_set1(int tx_size);
```

Minimal syntax (all-DC, EOB 0):

```cpp
SyntaxOptions options;  // base_q_idx, enable_cfl, disable_cdf_update
std::vector<std::uint8_t> frame;
std::vector<DecodedLeaf> parsed;
make_minimal_frame(geometry, superblocks, options, frame, &parsed, error);
```

`read_tile()` is an independent parser: it derives the tree from the symbols,
not from the plant, and re-derives every context. It refuses non-DC modes,
non-zero EOBs, SPLIT and 128x128 superblocks.

## Tests and pinned goldens

`sub8_test.cpp` covers:

- `is_sub8` / `min_dim_px` for all 29 block sizes.
- Exhaustive sweep: every valid size at every mi position of 8x8, 12x8, 8x12,
  16x16, 20x12 and 12x20, with all four `has_chroma`/`chroma_offset`
  combinations, compared byte-for-byte against
  `partition::init_allowed_partitions`, `partition::partition_implied`,
  `rect_type_implied_by_bsize` and `is_partition_allowed`.
- Sub-8 cases: 4x4 terminal, 4x8/8x4 implied direction, 4x16/16x4, the
  `>= 4x32` terminal sizes, the bottom-edge 4x8 forced leaf, the right-edge
  4x8, and 8x8 as the first rect_type-coded size.
- Chroma ownership for ten trees, cross-checked with
  `partition::validate_tree()`; exact owner positions; two chroma groups in a
  16x8 row; rejection of an illegal partition choice.
- Mode info: UV only on the owner, CfL availability from `ChromaMiSize`,
  no palette on sub-8.
- `txb_skip` contexts for Y/U/V including `EobU` and the block-larger-than-TX
  term.
- TX selection and `eobMultisize` for 4x4..16x8, DCT-forced `eob <= 1`,
  `TX_SET_INTRA_1`.
- Writer/reader round trips in both CDF modes (frozen and adaptive) and the
  complete frame wrapper; the reader rejects garbage.
- A two-superblock 128x64 frame fully decomposed to 4x4: 512 leaves, tile
  pinned, round-tripped, and gated with `dav2d`.
- Pinned tile goldens (bytes produced by the writer; they are independent of
  header changes):

| stream | tile bytes |
|---|---|
| 8x8 NONE | `0f0000` |
| 8x8 VERT -> 4x8,4x8 | `f1460000` |
| 8x8 VERT -> 4x4,4x4,4x8 | `fe228c0000` |
| 8x8 VERT -> 4x8,4x4,4x4 | `f2b87c0000` |
| 8x8 VERT -> 4x4 x4 | `fe26baa00000` |
| 8x8 HORZ -> 8x4,8x4 | `dfe30000` |
| 8x8 HORZ -> 4x4,4x4,8x4 | `ea523c0000` |
| 8x8 HORZ -> 8x4,4x4,4x4 | `e111140000` |
| 8x8 HORZ -> 4x4 x4 | `ea55a1a00000` |
| 16x16 edge tree (16x8 VERT, 8x8 HORZ, 8x4 VERT) | `de32c59ee6e30710c4698c33890000` |
| 128x64, two full SBs decomposed to 4x4 | 381-byte tile pinned in `sub8_test.cpp` |

The first nine streams share the 8x8 all-128 decoded frame; the 16x16 stream
is the same tree repeated over the frame and exercises right/bottom edge
leaves; the 128x64 stream has two superblocks and 512 sub-8 leaves. Gate
results (`dav2d f9dafbc`):

```
baseline_none    OK  8x8  md5=817dbf600caced6f1556c6690595f08f
vert_leaf4x8     OK  8x8  (same)
vert_first4x4    OK  8x8  (same)
vert_last4x4     OK  8x8  (same)
vert_all4x4      OK  8x8  (same)
horz_leaf8x4     OK  8x8  (same)
horz_first4x4    OK  8x8  (same)
horz_last4x4     OK  8x8  (same)
horz_all4x4      OK  8x8  (same)
edge16_all4x4    OK  16x16 md5=02b5d5d5ba2a5de00017b31c40c527bc
multi_sb_128x64  OK  128x64 md5=1df78db99c089b5262bed91613ea71ce
```

The gate also verifies every decoded sample is 128, so a desynchronised
chroma mode or residual changes the frame even when the decoder does not
abort.

## Integrated product path (`--still-sub8`)

The M1 writer contract below has been applied: `still::syntax::BlockSyntax`
carries `has_chroma` / `chroma_bsize` / `chroma_mi_row` / `chroma_mi_col`
(behind `kFlagHasChromaInfo`, so older producers keep the leaf-derived path),
`still_encoder.cpp` derives ChromaMi from `partition::PartitionNode`'s
`has_chroma`/`chroma_offset`, and `still_syntax.cpp::write_block()` codes UV and
the chroma residual only on the owner at the ChromaMi origin.  The non-owner
leaf keeps only its luma payload, and the deblock/CfL grids use the owned
chroma block.

The flag is opt-in, CPU-only and default off:

```sh
av2enc in.y4m -o out.ivf --still --still-sub8 --still-dc \
       [--no-deblock] --recon out.rec.y4m
```

`--still-sub8` sets the partition floor to 4 and the rect recursion depth to at
least 8.  It is rejected with `--still-gpu` (the CUDA planner codes 8x8 leaves
only) and with `--still-single`; combining it with
`--still-ext-partitions` is rejected until that family carries chroma ownership.

Product gate (builds a 4x4 checkerboard at 8x8, 16x16, 34x30, 136x88 and
640x360, encodes with deblock on/off, decodes with the pinned `dav2d` and
requires byte-identical Y/U/V reconstruction):

```sh
src/codec/partition/sub8/product_gate.sh build-int
```

Local result: PASS for all ten cases (`dav2d` f9dafbc, `--strict 1`), including
right/bottom edge leaves and the 640x360 product geometry.  Default `--still`
streams are unchanged (pinned baseline still passes).

## M1 writer contract

These are the exact call sites in the current tree (function names are stable;
line numbers drift). Only the M1 owner should apply them.

### 1. `BlockSyntax` — `src/encoder/m1/still_syntax.h` (`struct BlockSyntax`, ~L73)

Add:

```cpp
std::uint8_t has_chroma;      // node.has_chroma
std::uint16_t chroma_mi_row;  // ChromaMiRow (luma 4x4 units)
std::uint16_t chroma_mi_col;
std::uint8_t chroma_bsize;    // ChromaMiSize (partition::BlockSize)
```

`allowed_mask` already exists. No 10-bit widening is needed for sub-8 (that is
the extended-partition adapter's concern).

### 2. `Writer::write_block()` — `still_syntax.cpp` (~L1372)

Replace the leaf-derived chroma size and unconditional UV coding:

```cpp
// node.chroma_bsize is valid whenever node.has_chroma is set.
const BlockSize chroma_size =
    static_cast<BlockSize>(node.chroma_bsize);   // ChromaMiSize
const BlockSize chroma_residual =
    partition::chroma_plane_residual_size(geometry_, chroma_size);
const int cw = partition::block_width(chroma_residual);
const int ch = partition::block_height(chroma_residual);

cur_chroma_row_ = node.chroma_mi_row;
cur_chroma_col_ = node.chroma_mi_col;

if (node.has_chroma) {
  if (!code_uv_mode(...)) return false;
}
// luma code_tx unchanged
if (node.has_chroma) {
  code_tx(node, 1, ...);   // code_tx indexes node.chroma_mi_row/col
  code_tx(node, 2, ...);
}
```

A non-owning leaf must still contribute the luma payload only: set
`eob[1] = eob[2] = 0` (no U/V levels), keep `levels_offset` for Y, and skip
`choose_uv_mode` / chroma prediction / chroma reconstruction in
`still_encoder.cpp`. The owner codes the whole inherited chroma block, so the U/V
levels in the flat level array stay contiguous (`Y, U, V` per owner).

The transform-size validation at the top of `write_block` must validate luma
against the luma leaf and chroma against `chroma_residual`, not both against
`chroma_plane_residual_size(leaf.size)`.

Update the CfL grid (`uvcfl`) over `chroma_residual` at
`chroma_mi_row/2`, `chroma_mi_col/2` (not `r/2`, `c/2`).

### 3. `Writer::code_tx()` — `still_syntax.cpp` (~L960)

Use the leaf's new fields for the chroma origin (no extra parameters are
needed because `BlockSyntax` is per leaf):

- `x4 = plane ? node.chroma_mi_col >> 1 : node.mi_col`
- `y4 = plane ? node.chroma_mi_row >> 1 : node.mi_row`
- `coded_w/coded_h` from the plane's transform (luma leaf size, chroma
  residual size)
- the existing `txb_skip` context code is already exact for the V plane's
  `eob_u_` term; keep it.

### 4. `Writer::cfl_context()` — `still_syntax.cpp` (~L770)

Index `uvcfl_` at `cur_chroma_row_/2`, `cur_chroma_col_/2` (the ChromaMi
origin), not at `cur_r_/2`, `cur_c_/2`.

### 5. Decision stream — `still_encoder.cpp` (`collect_node` ~L361, `encode_block` ~L1033)

`partition::PartitionNode` already carries `has_chroma`/`chroma_offset` but not
`ChromaMi*`. Either:

- call `sub8::assign_chroma(geometry_, true, false, {}, node)` on a local copy
  of the tree before `collect_node()` and copy `has_chroma`, `chroma.mi_row`,
  `chroma.mi_col`, `chroma.size` into `BlockSyntax`; or
- reproduce the 8-line `chroma_step()` recursion from `sub8.cpp`.

`encode_block()` must use the leaf's `has_chroma`/`ChromaMi*` for
`choose_uv_mode`, prediction, transform, reconstruction, deblock metadata and
the CfL grid. For a non-owning leaf it must **not** predict or reconstruct
chroma (the owning leaf does the whole inherited chroma block).

The decision solver itself needs no change: `StillOptions::min_block_size = 4` and
`max_partition_depth` already reach these trees. After the writer is fixed,
any RDO/golden test that uses a sub-8 floor must be re-pinned.

### 6. GPU bridge — `src/encoder/m1/stream/still_stream_bridge.*`

`TreeNode` must carry the same four fields; the packer must not derive chroma
from the luma bsize.

## Limitations

- Partition trees are limited to the four M1 partitions and a 64x64
  superblock. 128x128 (where `PARTITION_SPLIT` exists) is rejected rather than
  modelled. Extended 3/4-way partitions are out of scope.
- The minimal syntax writer/reader is DC / zero-residual only. It is a gate
  oracle, not a replacement for the M1 writer; non-zero residual coding
  (EOB, quantised levels, TX types) is exercised by the production writer and
  is already correct once chroma ownership/origin are plumbed.
- 8-bit 4:2:0 only. 4:4:4/monochrome geometry is accepted by the partition
  module but not exercised here.
- No quality claim. These streams prove syntax correctness, not compression.
