# M1 MRL syntax adapter (`src/encoder/m1/mrl_stream/`)

Integration layer for AV2 v1.0.0 multiple reference line selection on top of
the tested prediction core in `src/encoder/mrl/`.

This directory **only adds files**. It does not modify `src/encoder/m1/still_*`,
`src/encoder/mrl/`, `src/cuda/m1/`, `src/main.cpp`, or `CMakeLists.txt`.

```
src/encoder/m1/mrl_stream/
  mrl_syntax.h/.cpp       per-block mrl_index / mrl_sec_index writer + reader
  mrl_interaction.h/.cpp  MrlIndex ↔ edge filter / TX-type derivation
  mrl_stream.h/.cpp       minimal M1 stream with MRL as the only optional tool
  mrl_stream_test.cpp     unit tests (58,515 checks)
  mrl_gate.cpp            writes IVF + expected luma per gate case
  gate_dav2d_mrl.py       pinned-dav2d decode + exact luma comparison
  build_and_test.sh       host build, tests, and dav2d gate
  README.md
```

## 1. Normative syntax

`§5.20.5.5 read_intra_y_mode()`:

```text
if (enable_mrls && is_directional_mode(YMode)) {
    mrl_index
    if (mrl_index > 0) mrl_sec_index
} else {
    mrl_index = 0
}
```

It is coded **after** the Y-mode symbols and **before** `read_intra_uv_mode()`.
MRL is luma-only: `§7.13.2.1` forces `MrlIndex = 0` for chroma, and
`mrl_sec_index` is ignored at exactly 4x4 (`not4x4 == 0`).

`§5.20.4.1 decode_block()` stores the context grid:

```text
UsesMrls[r][c] = mrl_index > 0 ? (mrl_sec_index ? 2 : 1) : 0
```

Contexts come from the first two `add_neighbor()` probes among below-left,
above-right, left, above, excluding neighbours outside the frame or in a
different superblock row:

```text
mrl_index ctx     = Σ (UsesMrls[NPos] > 0)      // 0..2
mrl_sec_index ctx = Σ (UsesMrls[NPos] == 2)     // 0..2
```

`§9.3` CDFs, copied byte-for-byte:

```text
Default_Mrl_Index_Cdf[3][5]     = {29573,31193,32023,78,0},
                                  {21812,27066,30279,75,0},
                                  {16076,23806,28762,1,0}
Default_Mrl_Sec_Index_Cdf[3][3] = {19678,6,0}, {12287,6,0}, {9574,6,0}
```

## 2. API

```cpp
namespace av2enc::still::mrl_stream {

inline constexpr int kMrlIndexContexts = 3;
inline constexpr int kMaxMrlIndex = 3;

struct MrlCdfs { uint16_t index[3][5]; uint16_t sec[3][3]; };
void init_mrl_cdfs(MrlCdfs&);

struct MrlBlockSyntax { int mi_row, mi_col, bw4, bh4; };
struct NeighborPos    { int row, col; };
int collect_npos(const MrlBlockSyntax&, int mi_rows, int mi_cols,
                 int sb_size_mi, NeighborPos out[2]);   // §5.20.4.1

class MrlContext {                                      // UsesMrls grid
  MrlContext(int mi_rows, int mi_cols, int sb_size_mi);
  int uses_at(int r, int c) const;
  int index_context(const MrlBlockSyntax&) const;       // 0..2
  int sec_context(const MrlBlockSyntax&) const;         // 0..2
  void update(const MrlBlockSyntax&, int mrl_index, bool sec);
};

struct MrlDecision { int mrl_index = 0; bool mrl_sec_index = false; };
enum class MrlSyntaxStatus { kOk, kInvalidMode, kInvalidIndex,
                             kInvalidSecondary, kEntropyError };

MrlSyntaxStatus write_mrl_syntax(av2::Encoder&, MrlCdfs&, MrlContext&,
                                 const MrlBlockSyntax&, int y_mode,
                                 const MrlDecision&, std::string* error);
MrlSyntaxStatus read_mrl_syntax(av2::Decoder&, MrlCdfs&, MrlContext&,
                                const MrlBlockSyntax&, int y_mode,
                                MrlDecision*, std::string* error);
}
```

Validity (checked before any bit is written):

| YMode | mrl_index | mrl_sec_index |
|---|---|---|
| non-directional (DC/SMOOTH/SMOOTH_V/SMOOTH_H/PAETH) | must be 0 | must be 0 |
| directional (V/H/D45/D135/D113/D157/D203/D67) | 0..3 | only when `mrl_index > 0` |
| any 4x4 | 0..3 | writer rejects; reader accepts (decoder ignores) |

`mrl_secondary_applies(w, h, mrl_index, sec)` is the single source of the
`not4x4` rule for callers.

## 3. Edge filter / transform-type interaction

`mrl_interaction.h` derives the inputs the core (`src/encoder/mrl/`) and
`src/encoder/tx_type/` require:

* `mrl_angle_delta(mrl)` = `Mrl_Index_To_Delta[4] = {0, 1, -1, 0}`.
* `mrl_p_angle(mode, delta, mrl)` = `Mode_To_Angle + delta*3 + Mrl_Index_To_Delta`.
* `mapped_intra_mode_for_mrl(...)` applies `§5.20.7.29 wide_angle_mapping()`
  after the MRL delta.  At `mrl == 0` it equals `txtype::map_intra_mode()`
  exactly (tested for all modes, deltas, and 4..64 sizes).
* `make_predict_params(...)` sets
  `enable_edge_filter = sequence_enable_intra_edge_filter && mrl_index == 0`
  (`§7.13.2.7`: edge filtering is applied only to the adjacent line).
* `make_tx_type_query(...)` / `tx_type_symbol_for_mrl(...)` fill the
  `§5.20.8.2` query with the MRL-aware mapped mode.

## 4. Minimal M1 stream

`write_minimal_mrl_frame()` builds a complete temporal unit
(TD + SEQ + CLK) with:

* `enable_mrls = 1`, everything else at the stock M1 defaults except the
  optional tools listed below;
* rect-only partitions (`NONE`/`SPLIT`/`HORZ`/`VERT`) to square leaves of
  `target_block_size`;
* per leaf: Y mode + MRL syntax, DC chroma with CfL disabled, `eob = 0` for
  Y/U/V;
* deblocking/CDEF/CCSO/restoration off, so decoded luma is exactly the intra
  prediction.

The sequence-header writer is a copy of `still_headers.cpp` with `enable_mrls`
exposed.  The test pins that with `enable_mrls = false` its bytes are
**identical** to `av2enc::still::make_sequence_header_obu()`, and that with `true`
exactly one bit changes.  This is why the minimal stream is possible without
editing `still_headers.cpp`.

`MinimalMrlFrame::recon_y` is the exact expected decoder luma (MRL core plus
`intra` for non-directional leaves).

## 5. Tests and gate

```sh
./build_and_test.sh                 # 58,515 checks + 8-case dav2d gate
SKIP_GATE=1 ./build_and_test.sh
SANITIZE=1 WERROR=1 ./build_and_test.sh
./build_and_test.sh --dump-goldens  # print the pinned writer bytes
```

Unit coverage:

| Area | What is checked |
|---|---|
| CDFs | both §9.3 arrays compared with independent literals |
| NPos | all frame sizes 64/128/192, 64/128 SB rows, 4..64 blocks, literal §5.20.4.1 model |
| contexts | `index_context`/`sec_context` vs a literal NPos sum; left/below-left vs above/above-right ordering |
| SB boundary | above and above-right are excluded across an SB row |
| directional-only | all 13 YModes; non-directional rejects MRL |
| writer/reader | frozen + adaptive CDF round-trip for every mode × mrl × secondary, 4x4..64x64 |
| validity | mrl_index 4, secondary with mrl 0, 4x4 secondary, non-directional MRL |
| goldens | two pinned byte strings (frozen/adaptive), see test file |
| tx interaction | mrl 0 == `txtype::map_intra_mode`; mrl 1/2 literal ±1° remap; mrl 3 == mrl 0 || edge filter | forced off for mrl > 0 regardless of the sequence flag |
| headers | byte equality with `still_headers.cpp` at 64/128/640, one-bit diff when enabled |
| stream | 4/8/16/32/64 leaves, SB-row use, DC-only with `enable_mrls = 1`, origin MRL + secondary |

Gate (`gate_dav2d_mrl.py`, pinned `tools/bin/dav2d-macos`) decodes:

| stream | leaves | MRL | secondary | result |
|---|---:|---:|---:|---|
| `mrl_dc_64x64` | 1 | 0 | 0 | luma matches |
| `mrl_dir_64x64_32` | 4 | 3 | 2 | luma matches |
| `mrl_dir_64x64_16` | 16 | 12 | 8 | luma matches |
| `mrl_dir_64x64_8` | 64 | 48 | 32 | luma matches |
| `mrl_dir_64x64_4` | 256 | 192 | 0 | luma matches |
| `mrl_dir_64x64_16_adaptive` | 16 | 12 | 8 | luma matches (adaptive CDF) |
| `mrl_dir_128x128_sb` | 16 | 12 | 8 | luma matches (SB row 1 uses MRL) |
| `mrl_origin_64x64` | 1 | 1 | 1 | luma matches (fillers + secondary) |

The gate also parses the sequence header and fails unless `enable_mrls == 1`.
Every stream passes; total compared luma samples: 45,056.

## 6. Integration into the stock M1 writer (not applied)

These are the exact edits needed to replace the minimal writer. None were made.

### `src/encoder/m1/still_headers.h:64` — config

```cpp
bool enable_intra_edge_filter = true;
bool enable_mrls = false;        // add
```

### `src/encoder/m1/still_headers.cpp:116` — sequence intra config

```cpp
writer.put_bit(c.enable_mrls);   // was: writer.put_bit(false)
```

### `src/encoder/m1/still_syntax.h:84-87` — leaf payload

```cpp
std::uint8_t y_mode = 0;
std::int8_t  y_angle_delta = 0;
std::int8_t  mrl_index = 0;      // add, 0..3
bool         mrl_sec_index = false;  // add
std::uint8_t uv_mode = 0;
```

### `src/encoder/m1/still_syntax.cpp` — writer

1. Include the adapter:

```cpp
#include "encoder/m1/mrl_stream/mrl_interaction.h"
#include "encoder/m1/mrl_stream/mrl_syntax.h"
```

2. `CdfState` (line 169) gains:

```cpp
mrl_stream::MrlCdfs mrl;
mrl_stream::MrlContext mrl_context;  // constructed with mi_rows/mi_cols/sb_mi
```

Initialize `mrl_stream::init_mrl_cdfs(cdfs_.mrl)` next to `init_cdfs(...)`
(line 316), and `cdfs_.mrl_context.reset(mi_rows_, mi_cols_,
partition::sb_size_mi(geometry_))` after the geometry is known (16 for 64x64
superblocks, 32 for 128x128 — the API takes the SB size in 4x4 units).

3. In `write_block()`, between `code_y_mode(node)` (line 1428) and
   `code_uv_mode(...)` (line 1432):

```cpp
const mrl_stream::MrlBlockSyntax mrl_block{
    r, c, w4, h4};
const mrl_stream::MrlDecision decision{node.mrl_index, node.mrl_sec_index};
const auto st = mrl_stream::write_mrl_syntax(
    enc_, cdfs_.mrl, cdfs_.mrl_context, mrl_block, node.y_mode, decision,
    &error);
if (st != mrl_stream::MrlSyntaxStatus::kOk) return false;
```

4. In `code_tx()` the luma TX-type derivation (lines 1064-1077) must use the
   MRL-aware mapping:

```cpp
int mapped_mode = mrl_stream::mapped_intra_mode_for_mrl(
    node.y_mode, node.y_angle_delta, node.mrl_index, bsize_w, bsize_h);
```

   The chroma `dct_only` path is unchanged.

5. Prediction (encoder reconstruction) must call:

```cpp
mrl::Params params;
mrl_stream::make_predict_params(
    y_mode, angle_delta, mrl_index, mrl_sec_index,
    options_.enable_intra_edge_filter, above_smooth, left_smooth,
    &params, w, h);
mrl::predict_block(block, 8, params, dst, stride);
```

   `make_predict_params` is what enforces §7.13.2.7 (no edge filter when
   `mrl_index > 0`).

### `src/encoder/m1/still_encoder.h:18` + `still_encoder.cpp` — decisions

Add an MRL search flag to `StillOptions`, try `mrl_index 0..3` and (when legal)
the secondary line in the luma mode search, then store the winner at
`still_encoder.cpp:1146`:

```cpp
leaf.mrl_index = static_cast<std::int8_t>(best_mrl_index);
leaf.mrl_sec_index = best_mrl_sec_index;
```

and use the same `mrl_stream::make_predict_params()` for the reconstruction
pass.  The syntax writer then derives every context itself, exactly as it does
for partitioning and Y-mode contexts.

### Build

When integrating, add to the `av2enc_core` source list in `CMakeLists.txt`:

```cmake
  src/encoder/m1/mrl_stream/mrl_syntax.cpp
  src/encoder/m1/mrl_stream/mrl_interaction.cpp
  src/encoder/m1/mrl_stream/mrl_stream.cpp   # optional: minimal gate writer
```

The gate and unit test are standalone (`build_and_test.sh`) and need no CMake
target.  `src/encoder/mrl/mrl.cpp` is already in the core build.
