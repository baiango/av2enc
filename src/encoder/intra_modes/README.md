# AV2 intra prediction — CPU reference

M1 wave 1, workstream 2. Spec-only implementation of the AV2 intra prediction
process for the encoder's mode-decision and reconstruction path.

Normative source: **AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`), §7.13.2. This module is a CPU correctness
oracle; it is not wired into the bitstream and is not a performance target.

## Files

```text
src/encoder/intra_modes/
  intra_modes.h          public API + normative contract
  intra_modes.cpp        CPU reference implementation
  intra_modes_test.cpp   standalone unit tests
  build_test.sh          standalone build/run (no root CMake edits)
  README.md              this file
```

Nothing outside this directory is modified.

## Implemented

| Spec | Process |
|---|---|
| §7.13.2.1 | `AboveRow` / `LeftCol` preparation, `MrlIndex = 0`, unavailable-neighbour fillers, top-left corner, `num4AboveRight` / `num4BelowLeft`, chroma >32 extension rule, edge clamping |
| §7.13.2.2 | Basic intra prediction (Paeth) |
| §7.13.2.7 | Directional prediction: 8 base modes, `AngleDeltaY/UV` = -3..+3 in 3-degree steps, wide-angle mapping for rectangular blocks |
| §7.13.2.8 | Single directional prediction, including luma IDIF (`Dr_Interp_Filter`) and the chroma linear path |
| §7.13.2.10 | DC prediction (`approx_divide` / `Div_Lut`) |
| §7.13.2.13 | SMOOTH, SMOOTH_V, SMOOTH_H |
| §7.13.2.14 | Filter corner |
| §7.13.2.17 | Intra edge filter strength selection |
| §7.13.2.18 | Intra edge filter (five-tap on a snapshot of the original edge) |

Modes: `DC, V, H, D45, D135, D113, D157, D203, D67, SMOOTH, SMOOTH_V,
SMOOTH_H, PAETH`. The 8 directional modes are V/H plus the six named angles.

Deliberately **not** implemented here (other workstreams or later milestones):

- DIP (§7.13.2.3), MRL `MrlIndex > 0`, IBP (§7.13.2.9/.12), DC-subsampled
  (§7.13.2.11, only used by large chroma with CfL)
- CfL (§7.13.5), MHCCP, palette, IntraBC
- Bitstream syntax, entropy, transform, reconstruction, deblock
- SIMD/threading and CUDA (wave 1 workstream 3 owns `src/cuda/intra_modes/`)

Samples are 8-bit `uint8_t` planes (M1 is 8-bit 4:2:0). `bit_depth` 8..12 is
accepted for clipping and filler values, but a future >8-bit path needs a
16-bit `PlaneView`; the API will change at that point.

## Build and run

```sh
cd src/encoder/intra_modes
./build_test.sh                 # -O2, build + run
./build_test.sh --no-run        # build only
SANITIZE=1 ./build_test.sh      # ASan + UBSan
WERROR=1 ./build_test.sh        # warnings promoted to errors
```

Expected output:

```text
intra_modes_test: PASS (48058 checks)
```

The build uses `-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow`.
It does not touch the root CMake build. The test has no external dependencies.

## API

```cpp
namespace av2enc::intra {

enum class Mode : uint8_t { kDc, kV, kH, kD45, kD135, kD113, kD157,
                            kD203, kD67, kSmooth, kSmoothV, kSmoothH, kPaeth };

struct PlaneView { const uint8_t* data; int stride; };

struct NeighborRequest {
  PlaneView recon;          // reconstructed plane, possibly padded stride
  int x, y;                 // block origin in samples
  int w, h;                 // 4..64, power of two
  int max_x, max_y;         // largest visible coordinates (inclusive)
  bool have_left, have_above;
  int num4_above_right, num4_below_left;
  int color_plane;          // 0 = luma, >0 = chroma
};

IntraStatus prepare_edges(const NeighborRequest&, int bit_depth,
                          IntraEdges&);

struct PredictParams {
  Mode mode;
  int angle_delta;                 // -3..3, directional only
  bool enable_edge_filter;         // sequence enable_intra_edge_filter
  bool above_neighbor_is_smooth;   // filterTypeAbove
  bool left_neighbor_is_smooth;    // filterTypeLeft
};

IntraStatus predict_intra(const IntraEdges&, const PredictParams&,
                          uint8_t* dst, size_t dst_stride);
IntraStatus predict_block(const NeighborRequest&, int bit_depth,
                          const PredictParams&, uint8_t* dst,
                          size_t dst_stride);
}
```

`prepare_edges()` implements §7.13.2.1: it copies the above row (including
`w+h` samples to the right), the left column (including `w+h` samples below),
the `-1` corner, and the unavailable-neighbour fillers. It stores
`valid_above_px = min(w, max_x - x + 1)` / `valid_left_px` so the prediction
step can compute the exact edge-filter extent without repeating geometry.

`predict_intra()` never mutates the `IntraEdges` argument. Internally it copies
the two edge rows, then applies the corner filter, edge filtering and
directional padding on the copy. The same `IntraEdges` can therefore be scored
against every RDO candidate.

`predict_block()` is the convenience path (`prepare_edges` + `predict_intra`)
for callers that do not cache edges.

### RDO / mode-decision hooks

```cpp
struct PredictionCost { uint64_t sad; uint64_t sse; };

PredictionCost prediction_cost(const uint8_t* src, size_t src_stride,
                               const uint8_t* pred, size_t pred_stride,
                               int w, int h);

IntraStatus predict_and_cost(const IntraEdges&, const PredictParams&,
                             std::span<uint8_t> pred_scratch,
                             const uint8_t* src, size_t src_stride,
                             PredictionCost& cost);

struct ModeCandidate { Mode mode; int8_t angle_delta; };
int build_mode_candidates(std::span<ModeCandidate> out,
                          bool include_angle_deltas);
```

`build_mode_candidates()` returns 13 base candidates, or 61 with all angle
deltas (13 + 8 directional modes × 6 non-zero deltas). It returns `-1` rather
than truncating when the output span is too small. A greedy/SATD pass can call
`predict_and_cost()` for every candidate; a later Lagrangian RDO pass adds its
own rate term on top of `sad`/`sse`. The module deliberately does not guess at
bit costs.

### Exposed normative helpers

`edge_filter_strength(w, h, smooth_neighbors, delta)` and
`intra_edge_filter(EdgeRow&, size, strength)` are public so the CUDA port and
tests can verify §7.13.2.17/.18 independently of a full prediction.

## Exact arithmetic contract (for the CUDA port)

The CUDA implementation must reproduce, in this order:

- `Round2(x, n) = (x + (1 << (n-1))) >> n`, arithmetic shift, `n = 0` returns
  `x`. Negative values occur in smooth prediction and PAETH differences.
- `Clip1` to `[0, (1 << bit_depth) - 1]`.
- `approx_divide(num, den) = Round2(num * Div_Lut[f], n + DIV_LUT_PREC_BITS)`
  with the §7.13.3.22 `resolve_divisor` derivation; do not substitute a hardware
  divide.
- Edge filtering derives `edge[i] = LeftCol[i-1]` (or `AboveRow[i-1]`) from a
  snapshot of the original samples and writes the filtered values back to
  `LeftCol`/`AboveRow`. Later taps never read filtered samples.
- Luma directional prediction always uses the 4-tap IDIF; chroma directional
  prediction uses the 2-tap linear form. `enableIdif = (plane == 0)`, not a
  sequence option.
- Wide-angle mapping (§5.20.7.29) is applied before edge filtering and
  prediction for non-square blocks. The mapped angle can exceed 180 or drop
  below 90.
- `Dr_Intra_Derivative`, `Dr_Interp_Filter`, `Div_Lut`, `Mode_To_Angle`, and
  `ANGLE_STEP` are transcribed from the spec; the CPU tables in
  `intra_modes.cpp` are the reference copy.

No floating point is used anywhere in the prediction path.

## Test coverage

| Test | What it checks |
|---|---|
| mode/tables/angles | `Mode_To_Angle`, directional-mode set, ±3×3° deltas, wide-angle mapping for 4:1 and 1:4 blocks |
| edge filter strength | Both filter types across every `w+h` bucket and threshold boundary |
| edge filter golden | Hand-computed snapshot impulse response (must not feed filtered samples into later taps); explicit snapshot convolution for all three strengths; strength-0/size-1 no-ops |
| edge fillers | No-neighbour 127/129/128 fillers; above-only/left-only replication; real corner |
| edge clamping | Right/bottom frame edges; `num4` extension; chroma >32 rule; padded stride with sentinel bytes |
| DC | Both/left-only/above-only/none; hand-computed `approx_divide`; 64×64 power-of-two average |
| flat blocks | All 61 candidates × 9 block shapes (4×4 … 64×64) predict the constant exactly, including edge filtering and smooth corner blending |
| V/H exactness | V copies the above row, H copies the left column; 90°/180° are unaffected by edge filtering |
| 45°/135° golden | Projection indices derived by hand from `Dr_Intra_Derivative[45] = 64` and `shift = 0`; verifies above/left branch selection and the -1 corner |
| wide-angle prediction | Tall 4×16 D45 maps to 225° and predicts `LeftCol[i+j+1]` exactly, exercising the mapping and p>180 path |
| IDIF vs chroma | Chroma linear 2-tap result on a step edge; luma 4-tap result with row-16 taps; the two differ |
| smooth | Hand-computed 4×4 values (113/105/120) plus full-block formula check |
| PAETH | Left/top/corner wins and the `pLeft` tie-break |
| boundary blocks | Top-left, top edge, left edge, blocks crossing right/bottom, extension counts |
| edge filter effect | Sharp step changes the prediction; smooth-neighbour filter type changes the selected strength |
| RDO hooks | 13/61 candidate lists, uniqueness, capacity error; SAD/SSE vs manual sums; scratch output; `IntraEdges` immutability |
| invalid arguments | Null plane/destination, bad stride, non-power-of-two/oversized blocks, bad bit depth, angle delta out of range, non-directional delta, bad origins, unavailable-neighbour contradiction |
| randomized sweep | 120 random 80×80 blocks up to 64×64, including blocks past frame edges, all 61 candidates, repeat-call determinism, `predict_block` equivalence, edge immutability |

`SANITIZE=1 ./build_test.sh` is clean (ASan + UBSan). This is not a `dav2d`
gate: the component produces no stream. Decoder/reconstruction/quality gates
apply only after integration.

## Integration notes

1. **Edges come from the reconstruction plane.** Pass the actual reconstructed
   samples (`CurrFrame[plane]`) with the padded stride and the visible
   `max_x`/`max_y`. Do not pass allocation bounds. Edge preparation performs all
   clamping; callers never read outside the visible plane.
2. **Availability flags are contextual.** For one tile, `have_left = (x > 0)`
   and `have_above = (y > 0)` at frame scope. The API also accepts the
   false-at-a-tile-boundary case, so a later tiled scheduler can reuse it
   without changing the predictor. `have_left` with `x == 0` (or `have_above`
   with `y == 0`) is rejected as contradictory.
3. **Extension counts are partition geometry.** Compute
   `num4_above_right` / `num4_below_left` from the block position and partition
   tree in 4×4 units. For chroma, halve the plane coordinates first; the module
   applies the §7.13.2.1 rule that chroma blocks wider/taller than 32 samples
   disable the corresponding extension.
4. **Filter types come from neighbour modes.** Set
   `above_neighbor_is_smooth` / `left_neighbor_is_smooth` from
   `YModes`/`UVSmooth`: true when the neighbour uses SMOOTH, SMOOTH_V or
   SMOOTH_H. For chroma with `TreeType == SHARED_PART`, the caller must use
   `ChromaMiRow/ChromaMiCol` for the lookup; this module only consumes the
   resulting booleans.
5. **Predict, then transform.** The expected encode loop is:
   `prepare_edges` → `predict_intra` into a block buffer → residual
   `int16_t = source - prediction` → transform/quant → inverse transform →
   `recon = clip(prediction + residual)` → write into the reconstruction plane.
   The predictor is independent of the transform and residual code.
6. **Mode decision.** For each block, prepare edges once, then enumerate
   `build_mode_candidates()`. A first pass can use `predict_and_cost` (SAD/SSE)
   on a scratch block; the later RDO pass should add rate/λ terms externally and
   re-use the same prediction entry point. `predict_intra` is pure, so scoring
   order and candidate pruning do not change results.
7. **IDIF is mandatory for luma.** The 4-tap luma directional filter is part of
   the normative prediction, not an encoder option. Disabling it will make the
   encoder reconstruction diverge from the decoder. `enable_edge_filter`
   mirrors the sequence-level `enable_intra_edge_filter` and must match the
   stream header.
8. **No bitstream code here.** `Mode`, `angle_delta`, MRL, and edge-filter flags
   are syntax owned by the integration/entropy path. This module does not
   choose a mode or write symbols; it only predicts and costs candidates.
9. **CUDA parity.** `src/cuda/intra_modes/` should port these functions
   one-for-one and assert CPU/GPU bit-exactness on randomized blocks before the
   kernel is used. Keep integer-only arithmetic and the table copies above. The
   CPU implementation is intentionally simple (no SIMD), per the project rules.

## Known gaps / deferred

- No MRL, DIP, IBP, CfL, MHCCP, palette, or IntraBC.
- No 4:2:2/4:4:4 subsampling logic beyond caller-supplied `max_x`/`max_y` and
  the chroma >32 rule.
- No syntax-level mode signalling or RD rate model.
- No CUDA kernel (separate workstream), no SIMD, no threading.
- The module is not wired into `src/encoder/m0_still.cpp`, `src/main.cpp`, the
  root `CMakeLists.txt`, or any encoder pipeline.
