# AV2 MRL (multiple reference line selection) — CPU reference

Spec-only CPU implementation of AV2 v1.0.0 luma MRL:

* §7.13.2.1 `AboveRow` / `LeftCol` / `AboveSecRow` / `LeftSecCol` with
  `MrlIndex > 0`;
* §7.13.2.7 `pAngle` including `Mrl_Index_To_Delta[4] = {0, 1, -1, 0}` and the
  optional secondary-line blend;
* §7.13.2.8 single directional prediction with `mrlIndex`, including the
  `minBase = -(1+mrlIndex)` / `maxBase = w+h-1+2*mrlIndex` extension.

This is a correctness oracle, not an encoder integration. It is luma-only:
§7.13.2.1 forces `MrlIndex = 0` for chroma, and the syntax only signals
`mrl_index` after `is_directional_mode(YMode)`.

## Availability / mode dependence

| Condition | Rule |
|---|---|
| mode | directional YMode only (V, H, D45, D135, D113, D157, D203, D67) |
| plane | luma only; chroma predictions use `intra_modes` directly |
| `mrl_index` | 0..3 (`Default_Mrl_Index_Cdf` has `4+1` entries) |
| `mrl_sec_index` | only when `mrl_index > 0`, and ignored at 4x4 |
| SB boundary | `aboveMrlIndex = ((y & (sbHeight-1)) == 0) ? 0 : mrl_index` |
| edge filter | only when `MrlIndex == 0` (§7.13.2.7) |

`mrl_index = 3` uses the 3-line-away primary reference and the adjacent line
as the secondary reference; its angle delta entry is 0. `mrl_index = 1/2`
add +1/-1 degree to `pAngle` before wide-angle mapping.

## Files

```
src/encoder/mrl/
  mrl.h          public API
  mrl.cpp        reference implementation
  mrl_test.cpp   standalone tests
  build_test.sh
  README.md
```

## API

```cpp
namespace av2enc::mrl {

struct Block {
  intra::PlaneView recon;
  int x, y, w, h;          // w,h 4..64, power of two
  int max_x, max_y;        // inclusive visible bounds
  bool have_left, have_above;
  int num4_above_right, num4_below_left;
  int plane;               // must be 0 (luma)
  int sb_height;           // 64 for M1
};

struct Params {
  Mode mode;               // directional
  int angle_delta;         // -3..3
  int mrl_index;           // 0..3
  bool mrl_sec_index;
  bool enable_edge_filter;
  bool above_neighbor_is_smooth, left_neighbor_is_smooth;
};

MrlStatus prepare_edges(const Block&, int bit_depth, int mrl_index, Edges&);
MrlStatus predict(const Edges&, const Params&, uint8_t* dst, size_t stride);
MrlStatus predict_block(const Block&, int bit_depth, const Params&,
                        uint8_t* dst, size_t stride);
}
```

`prepare_edges()` fills the four line arrays exactly as §7.13.2.1, including
unavailable-neighbour fillers (127/129/128 in 8-bit) and the negative indices
`-1 .. -1-mrl_index`. `predict()` never mutates the prepared edges.

## Tests

```sh
./build_test.sh              # build + run
SANITIZE=1 ./build_test.sh   # ASan + UBSan
```

`mrl_test.cpp` has two independent oracles:

* `mrl_index = 0` is compared byte-for-byte against the verified
  `src/encoder/intra_modes/` predictor for every directional mode/delta,
  including edge filtering and smooth-neighbour flags;
* `mrl_index = 1..3` and the secondary blend are compared against a literal
  nested-loop transcription of §7.13.2.1/.7/.8 written in the test file.

Coverage: all 4..64 power-of-two sizes, frame edges (`x=0`, `y=0`), SB
boundaries, 4x4 (secondary ignored/rejected), all 8 directional modes, all 7
angle deltas, all four MRL indices, secondary on/off, and random
smooth-neighbour flags. Expected output:

```
mrl_test: PASS (15417 checks)
```

## Integration notes (M1)

The M1 bitstream cannot signal MRL yet:

* `src/encoder/m1/still_headers.cpp` writes `enable_mrls = 0`;
* `still_syntax.cpp` has no `mrl_index` / `mrl_sec_index` writer;
* `StillOptions` has no MRL search flag;
* the transform-type derivation (§7.15.2 / secondary transform) also consumes
  `Mrl_Index_To_Delta`, so enabling it is not prediction-only.

Those files are outside this workstream. A decoder gate needs, at minimum:

1. `enable_mrls = 1` in the sequence header;
2. per-block `mrl_index` (+ `mrl_sec_index` when > 0) after `read_intra_y_mode`,
   with the §5.20.5.5 context (`UsesMrls` neighbours);
3. `StillOptions`/RDO wiring and reconstruction using this predictor;
4. transform-type derivation updated for `MrlIndex`.

No minimal stream was produced, so there is no decoder gate result. The module
is prediction-complete and spec-tested, not integrated.

## Quality context (prediction-only, not a net-win claim)

A directional-mode-only probe over the QP-90 reconstructed frame (8x8 blocks on
a 16x16 grid, source as distortion target, decoded plane as neighbour state,
`num4AboveRight = num4BelowLeft = 0`) measured:

| Frame | blocks | MRL beats line 0 | directional SSE, mrl 0 → 0..3+sec | PSNR |
|---|---:|---:|---:|---:|
| 640x360 | 920 | 193 (727 ties) | 8.45M → 7.46M (-11.7%) | 26.56 → 27.10 dB |
| 1920x1080 | 8160 | 1247 (6913 ties) | 14.45M → 11.80M (-18.3%) | 33.71 → 34.59 dB |

This is prediction headroom on directional blocks, not a bitstream result: the
MRL index and secondary flag cost bits, and the actual partition/mode search is
different. It is not safe to call MRL a QP-90 net win from this table.
