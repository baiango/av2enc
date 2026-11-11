# encoder/m1/partition_stream — extended partitions for the M1 writer

Integration-ready adapter between the AV2 extended-partition solver
(`src/encoder/partition_ext/`) and the M1 partition writer
(`src/encoder/m1/m1_syntax.*`).

This directory is **additive**. `src/encoder/partition_ext/`, `src/encoder/m1/m1_*`,
`src/cuda/m1/`, `src/main.cpp`, and `CMakeLists.txt` are not modified here.
Everything the M1 agent needs to change is listed under
[M1 writer contract](#m1-writer-contract) with the exact call sites.

- Spec: AV2 Bitstream & Decoding Process Specification v1.0.0, §5.4.3,
  §5.20.3.1/2, §6.19.3, §8.2, §8.3.2, §9.2, §9.3.
- No AVM, SVT-AV1/AV2, clave, or av2-turbo encoder code was consulted.
- Status: adapter tests pass, block-level `dav2d` gate **passes** for HORZ_3 and
  both uneven 4-way variants, in both CDF modes (see [Results](#results)).

## Files

| File | Purpose |
|---|---|
| `partition_stream.h` | public API: `TilePartitionCdfs`, `M1Node`, `BlockSyntaxV2`, `FrameStream`, contexts, `plan_node()`, `emit_symbols()`, frame codec, cost lift |
| `partition_stream.cpp` | implementation |
| `partition_stream_test.cpp` | adapter tests (CDFs, masks, contexts, symbols, stream shape, chroma, round trips, cost lift) |
| `gate_minimal.cpp` | writes the four/six minimal all-DC/all-skip IVF streams for the gate |
| `gate_dav2d.py` | patches the §5.4.3 tool flags into the sequence header and gates the streams with `dav2d` |
| `build_and_test.sh` | standalone C++20 build (warnings, ASan/UBSan, `-Werror`), tests, then the gate |

```sh
./src/encoder/m1/partition_stream/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/m1/partition_stream/build_and_test.sh
SKIP_GATE=1 ./src/encoder/m1/partition_stream/build_and_test.sh   # no dav2d
```

Self-contained: the script compiles `partition.cpp`, `partition_ext.cpp`,
`ext_syntax.cpp`, and `av2_entropy.cpp` directly; it does not touch the root
CMake build.

## API

All types are in `av2enc::m1::partition_stream`; `pe` below is
`av2enc::partition_ext`.

### CDFs

```cpp
struct TilePartitionCdfs {
  pe::PartitionCdfs ext;  // five §9.3 arrays, layout unchanged
};
void init_tile_partition_cdfs(TilePartitionCdfs&);
std::uint16_t* cdf_row(TilePartitionCdfs&, pe::SymbolKind, int context,
                       std::string& error);
```

`ext` holds exactly the arrays the M1 writer needs, including the two it does
not have today:

| adapter field | M1 name | shape |
|---|---|---|
| `ext.do_split` | `TileDoSplitCdf` | `[2][64][3]` |
| `ext.do_square_split` | `TileDoSquareSplitCdf` | `[2][8][3]` |
| `ext.rect_type` | `TileRectTypeCdf` | `[2][64][3]` |
| `ext.do_ext_partition` | **`TileDoExtPartitionCdf`** | `[2][64][3]` |
| `ext.do_uneven_4way_partition` | **`TileDoUneven4wayPartitionCdf`** | `[2][64][3]` |

Plane 0 is used (`SHARED_PART`). The default values come from
`ext_tables_generated.h`; tests pin `do_split` / `do_square_split` /
`rect_type` byte-for-byte against `src/encoder/m0_cdf_tables.h`.

### Node shape and masks

```cpp
struct M1Node {
  std::uint16_t mi_row, mi_col;
  std::uint8_t bsize;          // partition::BlockSize
  std::uint8_t partition;      // pe::PartitionType, spec value 0..9
  std::uint8_t flags;          // syntax::kFlagForced | kFlagSbRoot
  std::uint16_t subtree;       // preorder entries below this node
  std::uint16_t allowed_mask;  // bit p = allowed[PartitionType p], 10 bits
  bool has_chroma, chroma_offset;
  pe::ChromaBlock chroma;      // ChromaMi* the leaf codes
};

std::uint16_t to_allowed_mask(const pe::PartitionSet&);
bool mask_allows(std::uint16_t, PartitionType);
int mask_count(std::uint16_t);
```

`BlockSyntaxV2` is the proposed replacement for `syntax::BlockSyntax`'s tree
part; `to_block_syntax_v2(M1Node)` fills it. It adds `has_chroma`,
`chroma_bsize`, `chroma_mi_row`, `chroma_mi_col` and widens `allowed_mask`.

Two mask views:

- `M1Node::allowed_mask` is the normative 10-bit set. Use it for
  `partition_ext` planning.
- `to_writer_mask8(full_mask)` is the base-rect view the *current* writer
  understands: `NONE`, `SPLIT`, `HORZ` = whole horizontal family, `VERT` = whole
  vertical family. This reproduces the spec's `allowHorz`/`allowVert` test that
  decides whether `rect_type` is coded, so an unmodified writer's base symbols
  stay correct while the adapter emits the extended rows.
  `mask_fits_writer_u8()` reports whether the full mask fits the current
  `uint8_t` field. With uneven 4-way on it does **not** (VERT_4B is bit 8), and
  a 128x128 root with SPLIT is bit 9, so the field must be widened.

### Contexts (§8.3.2)

```cpp
struct NeighbourContext {
  int left_log2 = 6, left_mid_log2 = 6;    // LeftMiSizes[r], [r + h4/2]
  int above_log2 = 6, above_mid_log2 = 6;  // AboveMiSizes[c], [c + w4/2]
  bool left_available = false, above_available = false;
};
struct NodeContexts { int do_split, do_square_split, rect_type, do_ext; };

NodeContexts contexts_from_neighbours(BlockSize, PartitionType,
                                      const NeighbourContext&);
int ext_partition_context(BlockSize, bool horizontal,
                          const NeighbourContext&);

NeighbourContext neighbours_from_grid(const FrameGeometry&, int r, int c,
                                      BlockSize, const MiSizeGrid&);
NeighbourContext neighbours_from_arrays(std::span<const int> left_mi,
                                        std::span<const int> above_mi,
                                        int r, int c, BlockSize,
                                        bool left_available,
                                        bool above_available);
```

`neighbours_from_arrays` is the exact bridge for the writer's
`left_mi()`/`above_mi()` arrays (Mi log2 values, 6 sentinel, +16 padding).
The second probes at `r + h4/2` / `c + w4/2` are mandatory; tests compare the
derivation against `pe::node_contexts()` on a grid where the start and midpoint
probes deliberately differ.

### Planning and emission

```cpp
struct NodePlan {
  bool ok, forced;
  pe::Boundary boundary;
  pe::PartitionSet allowed;
  pe::SymbolPlan symbols;   // from plan_partition_symbols()
  std::string error;
};

NodePlan plan_node(const FrameGeometry&, const pe::Tools&, const M1Node&,
                   const NodeContexts& | const NeighbourContext& |
                   const MiSizeGrid&);
bool emit_symbols(const pe::SymbolPlan&, TilePartitionCdfs&, av2::Encoder&,
                  pe::SyntaxStats*, std::string& error);
```

`plan_node` re-derives boundary/allowed from the node, rejects a stale
`allowed_mask` or a stale `kFlagForced`, then calls
`pe::plan_partition_symbols()`. The emitted sequence is exactly
`read_partition()`:

```
do_split?  do_square_split?  rect_type?  do_ext_partition?
do_uneven_4way_partition?  L(1) uneven_4way_partition_type?
```

The type bit is a raw `encode_bool()`; it adapts no CDF. `emit_symbols` picks
`do_ext_partition` / `do_uneven_4way_partition` rows for the two new arrays and
the other rows for the existing ones.

### Preorder stream and chroma

```cpp
struct LeafChroma {
  std::uint32_t node_index;
  bool has_chroma, chroma_offset;
  pe::ChromaBlock chroma;
};
struct FrameStream {
  std::vector<M1Node> nodes;      // preorder, superblocks in raster order
  std::vector<LeafChroma> leaves; // decode order == pe::collect_leaves()
  int node_count, leaf_count;
};
bool build_superblock_stream(...);
bool build_frame_stream(const FrameGeometry&, const pe::Tools&,
                        const pe::FrameDecision&, FrameStream&, std::string&);
```

Both call `pe::validate_tree()` / `pe::validate_frame()`, enforce raster
superblock order, and fail on any stale field. `leaves[i]` indexes the preorder
node vector, so a writer can copy the leaf chroma fields without re-walking.

`chroma_plane_block(geometry, ChromaBlock)` converts the adapter's luma-unit
`ChromaBlock` into the coordinate system the M1 chroma/CfL grid uses:

```cpp
struct ChromaPlaneBlock {
  int mi_row, mi_col;            // 4x4 units of the chroma plane
  BlockSize residual_size;       // chroma_plane_residual_size(luma block)
};
```

Examples pinned by tests:

| leaf | ChromaBlock (luma units) | plane block | residual |
|---|---|---|---|
| HORZ_3@8x32 top 8x8 | `(0,0,8x8)` | `(0,0)` | `4x4` |
| HORZ_3@8x32 mid-right 4x16 | `(2,0,8x16)` | `(1,0)` | `4x8` |
| HORZ_4A@32x32 last 32x4 | `(0,0,32x32)` | `(0,0)` | `16x16` |
| VERT_3@32x8 mid-bottom 16x4 | `(0,2,16x8)` | `(0,1)` | `8x4` |

The first three HORZ_4A/4B children have `has_chroma == false`; the last child
codes the whole parent chroma block. In HORZ_3@8x32 / VERT_3@32x8 the middle
pair shares one chroma block; the mid-right (or bottom) child owns it.

### Whole-frame reference codec

```cpp
bool encode_superblock_partitions(...);  // one SB, own entropy coder
bool encode_frame_partitions(...);       // one tile payload, all SBs
bool decode_frame_partitions(...);       // independent §8.2 decoder
```

`encode_frame_partitions` produces the partition-symbol prefix of a tile
payload; `decode_frame_partitions` reads it back with an independent decoder.
Byte equality with `pe::encode_tree_symbols()` is a test invariant. Both CDF
modes are supported (`disable_cdf_update` true/false), matching the frame
header bit the M1 writer already emits.

### Cost model

```cpp
struct ExtSignalCosts { double three_way, uneven_4way; };
pe::CostFunction lift_rect_costs(const partition::CostFunction&,
                                 const ExtSignalCosts& = {});
```

Reuses the caller's six rect costs and adds the extended signal costs. The
base `PartitionSet` passed to the callback is mapped from the 10-value spec
enum (NONE/SPLIT exact, HORZ/VERT as the whole family), not by index shift.
With `ExtSignalCosts` left infinite, the lifted model reproduces the rect-only
solver.

## Chroma mapping (read before touching `write_block`)

`pe::ChromaBlock` stores **luma** mi coordinates and a **luma** `BlockSize`
representing the chroma block's area. The M1 CfL/uvcfl grid is in 4x4 units of
the chroma plane, so:

```cpp
const ps::ChromaPlaneBlock cb = ps::chroma_plane_block(geometry_, node.chroma_block);
const p::BlockSize luma_chroma_size = node.chroma_bsize;  // ChromaMiSize
```

- `cb.mi_row/mi_col` are the chroma-grid origin (divide luma mi by 2 for 4:2:0).
- `cb.residual_size` is what `partition::chroma_plane_residual_size` would
  return for the chroma block; use it for TX size, EOB and coefficient coding.
- If `has_chroma == false`, code **no** UV mode and **no** UV transform for the
  leaf version of `pe::PartitionType::kNone`.
- Update the uvcfl grid over `cb.residual_size` at `cb.mi_row/mi_col`, not over
  the luma leaf.

`m1_syntax.cpp::write_block()` currently derives the chroma size from the luma
leaf and always codes UV, which is wrong for HORZ_4A/4B first children and for
the shared HORZ_3/VERT_3 middle block.

## M1 writer contract

Exact call sites in the current tree (line numbers drift; function names are
stable). Only the M1 owner should apply these.

### 1. Sequence header flags

`src/encoder/m1/m1_headers.h` — add to `M1Config`:

```cpp
bool enable_ext_partitions = false;
bool enable_uneven_4way_partitions = false;  // only read when ext is 1
```

`src/encoder/m1/m1_headers.cpp::write_sequence_header()` (~line 102) — replace

```cpp
writer.put_bit(false);  // enable_ext_partitions
writer.put_bit(false);  // reduce_pb_aspect_ratio
```

with

```cpp
writer.put_bit(c.enable_ext_partitions);
if (c.enable_ext_partitions) {
  writer.put_bit(c.enable_uneven_4way_partitions);  // §5.4.3
}
writer.put_bit(false);  // reduce_pb_aspect_ratio
```

The dependent bit is present **only** when `enable_ext_partitions == 1`; the
minimal gate proves the pinned `dav2d` accepts both values.

### 2. `BlockSyntax` (or adopt `BlockSyntaxV2`)

`src/encoder/m1/m1_syntax.h::BlockSyntax` (~line 73):

```cpp
std::uint8_t partition;      // -> spec enum 0..9 (pe::PartitionType)
std::uint8_t allowed_mask;   // -> std::uint16_t (10 bits)
// add for leaves:
std::uint8_t has_chroma;
std::uint8_t chroma_bsize;   // pe::ChromaBlock::size
std::uint16_t chroma_mi_row, chroma_mi_col;
```

`BlockSyntaxV2` in `partition_stream.h` has exactly this layout and
`to_block_syntax_v2()` fills it. Keeping the current 4-value
`partition::PartitionType` anywhere in the path loses HORZ_3 and friends.

### 3. CDF state

`src/encoder/m1/m1_syntax.cpp::CdfState` (~line 175) and `init_cdfs()`
(~line 218): add the two arrays and copy them from the adapter defaults.

```cpp
struct CdfState {
  ...
  std::uint16_t do_ext_partition[2][64][3];
  std::uint16_t do_uneven_4way_partition[2][64][3];
};
// init_cdfs:
ps::TilePartitionCdfs partition_defaults;
ps::init_tile_partition_cdfs(partition_defaults);
copy_array(c.do_ext_partition, partition_defaults.ext.do_ext_partition);
copy_array(c.do_uneven_4way_partition,
           partition_defaults.ext.do_uneven_4way_partition);
```

`do_square_split` is not needed while `use_128x128_superblock == 0`, but the
array is already in `TilePartitionCdfs` for the 128 SB path.

### 4. `Writer::write_node()` — the actual symbol emission

`src/encoder/m1/m1_syntax.cpp::write_node()` (~line 537). Replace the
`allowed(...)` / `code_do_split()` / `need_rect` / `code_rect_type()` block with
the adapter plan. The writer must keep a `pe::Tools` member built from the
config, and a `TilePartitionCdfs` member.

```cpp
const ps::M1Node m1 = to_m1_node(node);  // or construct inline from BlockSyntax
const bool left_ok  = base::is_inside(geometry_, r, c - 1);
const bool above_ok = base::is_inside(geometry_, r - 1, c);
const ps::NeighbourContext nc = ps::neighbours_from_arrays(
    left_mi_, above_mi_, r, c, size, left_ok, above_ok);
const ps::NodePlan plan = ps::plan_node(geometry_, tools_, m1, nc);
if (!plan.ok) { error = plan.error; return false; }
if (!ps::emit_symbols(plan.symbols, partition_cdfs_, enc_, &stats_, error)) {
  error = error_;  // or the adapter's error
  return false;
}
```

Then delete `code_do_split()` / `code_rect_type()` from `write_node` (the
adapter emits all four base symbols as well, so leaving them in double-codes
symbols). `plan_node` rejects a stale mask/forced flag, so the writer must copy
the node from `ps::FrameStream`, not re-derive a 4-bit mask.

### 5. Child enumeration

`src/encoder/m1/m1_syntax.cpp::enumerate_slots()` (~line 263) only knows
`NONE/SPLIT/HORZ/VERT`. Replace it with:

```cpp
pe::ChildSlot slots[4];
const int n = pe::enumerate_children(geometry_, r, c, size, partition,
                                     node.has_chroma, node.chroma_offset,
                                     inherited_chroma, slots);
for (int i = 0; i < n; ++i) {
  if (!pe::child_is_coded(geometry_, slots[i])) continue;
  // match the next preorder child against slots[i] and recurse
}
```

`enumerate_children` also returns the per-child `has_chroma`,
`chroma_offset` and `ChromaBlock`, so the writer can validate the stream's
leaf chroma instead of guessing it. `syntax::partition_child_slots()`
(~line 1528) must either be widened or left for the rect-only bridge; do not
route extended partitions through it.

### 6. `Writer::write_block()`

`src/encoder/m1/m1_syntax.cpp::write_block()` (~line 1321), around the chroma
size derivation (~line 1335) and `code_uv_mode()` (~line 1380):

```cpp
const p::BlockSize chroma_size =
    node.chroma_bsize == 0 ? p::chroma_plane_residual_size(geometry_, size)
                           : static_cast<p::BlockSize>(node.chroma_bsize);
const ps::ChromaPlaneBlock cb = ps::chroma_plane_block(
    geometry_, pe::ChromaBlock{node.chroma_mi_row, node.chroma_mi_col,
                               static_cast<p::BlockSize>(node.chroma_bsize)});
if (node.has_chroma) {
  if (!code_uv_mode(...)) return false;
  // code U and V with cb.residual_size
}  // else: no UV mode, no UV txb_skip / eob
```

The CfL/uvcfl grid update must use `cb.mi_row/mi_col` and `cb.residual_size`.
Everything else (`code_tx`, CDF context, levels) is unchanged because it is
parameterised by the plane's coded size.

### 7. Decision path

`src/encoder/m1/m1_still.cpp`:

- `encode()` cost lambda (~line 266) and `partition::decide_frame()` (~line 288):
  switch to `partition_ext::Tools` + `partition_ext::decide_frame()` with
  `ps::lift_rect_costs(rect_cost_fn, ext_signals)`. With both tool flags off
  this reproduces the current rect-only choices.
- `collect_node()` (~line 357): the preorder walk is the same, but the node type
  becomes `partition_ext::PartitionNode`; the mask must be 10-bit and the leaf
  must copy `has_chroma` / `chroma` into `BlockSyntax` (or feed
  `ps::FrameStream`). `build_frame_stream()` already produces that order and
  validates the tree.
- `encode_block()` (~line 961): it takes `const partition::PartitionNode&` and
  derives chroma from `node.size` (~line 986). It must take the leaf's
  `has_chroma` / `ChromaBlock` (or `M1Node`) and use
  `ps::chroma_plane_block()` as in item 6.

### 8. GPU bridge (separate owner)

`src/encoder/m1/stream/m1_stream_bridge.h::TreeNode` (~line 51) has the same
`uint8_t partition` / `uint8_t allowed_mask` limit and no chroma block;
`pack_gpu_frame()` / the raster packer in `m1_stream_bridge.cpp` derive chroma
from the luma bsize. They need the same widening + chroma fields. The CUDA
`BlockPlan` (`src/cuda/m1/`) has no partition fields at all, so the GPU path
cannot supply an extended tree until that POD grows them; this adapter is the
CPU-side contract for that work.

## Results

`build_and_test.sh` (last run: macOS, `clang++`, ASan/UBSan + `-Werror`, pinned
`dav2d` `f9dafbc`):

### Adapter tests

- §9.3 CDF copies, including `TileDoExtPartitionCdf` and
  `TileDoUneven4wayPartitionCdf`; base rows byte-identical to M0.
- 10-bit masks, writer family view, uint8 overflow detection.
- Context equality against `pe::node_contexts()` on 64x64, 640x360,
  128x128 and 192x128 frames, tool flags off/ext/uneven, including a synthetic
  grid where `r` and `r + h4/2` (and `c`, `c + w4/2`) disagree.
- Symbol plans: HORZ_3, VERT_3, HORZ_4A, VERT_4B, 128x128 SPLIT, tools off,
  forced nodes; literal bit 0/1; stale-mask rejection.
- Preorder streams: 64x64, 128x128 (SPLIT root then HORZ_3), multi-SB (96x64)
  with edge superblocks; subtree counts and `kFlagSbRoot`.
- Chroma carry: HORZ_4A@32x32 (three chroma-less children), HORZ_3@8x32
  (shared 8x16) and VERT_3@32x8 (shared 16x8).
- Round trips (`encode_frame_partitions` -> `decode_frame_partitions`) for
  64 SB and 128 SB geometries, tool flags off / ext / ext+uneven, CDF update
  frozen and adaptive; tree signatures and leaf chroma match, and the bytes
  are identical to `pe::encode_tree_symbols()` for one-SB frames.
- `lift_rect_costs()` keeps rect costs and the compact base allowed set.

### dav2d gate (`gate_dav2d.py`)

Six streams, all 64x64, all-DC, all-skip, both CDF modes:

| stream | partitions | symbols | literal bits | result |
|---|---|---:|---:|---|
| `baseline` | NONE | 1 | 0 | decodes to 128 |
| `ext` | HORZ_3 root | 8 | 0 | **PASS** |
| `ext4` | HORZ -> VERT -> HORZ_4A | 14 | 1 | **PASS** |
| `ext4b` | HORZ -> VERT -> HORZ_4B + VERT_4B | 19 | 2 | **PASS** |

Baseline pixels are compared byte-for-byte with each extended stream; every
sample must be 128 (the DC prediction of an all-skip frame). The gate patches
only the two sequence-header tool bits, exactly like the header-level gate in
`partition_ext/gate_dav2d.py`. This is a **block-level** gate: `dav2d` parses
`do_ext_partition`, `do_uneven_4way_partition`, the L(1) type bit, and the
HORZ_3/4-way child geometry.

The gate is deliberately narrow (DC, zero residual). It does not replace a
full `tools/validate_dav2d.sh` run on a real M1 stream; the first stream that
the integrated writer produces must still be gated on the normal corpus.

## Limitations

- Partition symbols only. Mode info, residual coding, reconstruction, RDO rate
  modelling and OBU assembly stay in the owning modules.
- `SHARED_PART` only, one tile, 8-bit 4:2:0 geometry (4:4:4/monochrome geometry
  is accepted by `partition_ext` but not exercised by M1).
- 128x128 superblock streams are tested at the adapter level. Emitting them
  also needs the header `use_128x128_superblock` bit, 128 SB tile info, and the
  writer's block geometry; the current M1 header hardcodes 64.
- No quality claim. `docs/negative-results.md` applies: rate-aware costs and a
  full-corpus gate must land before any BD-rate number is quoted.
