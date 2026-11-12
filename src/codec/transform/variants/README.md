# CPU transform variants (`src/encoder/tx_variants/`)

M1 component: AV2 transform types and sizes beyond the M0 32x32 DCT.

- coded sizes: all 25 AV2 `TX_*` sizes, 4x4 through 64x64, square and
  rectangular (`enum_tx_size`, §6.19.6.1)
- transform types: DCT, ADST, IDTX, FLIPADST in either dimension, i.e. all 16
  `TX_TYPE` values (`DCT_DCT` … `H_FLIPADST`)
- TX split selection hooks: `NONE` / `SPLIT` / `HORZ` / `VERT`, legal geometry,
  and a greedy SSE + rate-proxy selector
- CPU reference + CUDA kernel pair; the CUDA side is in
  `src/cuda/tx_variants/`

This directory is self-contained. It does not edit `src/cuda/transform.*`,
`src/encoder/basic_encoder.cpp`, the entropy/bitstream paths, or the root
`CMakeLists.txt`, and it is not wired into the encode path.

## Files

| File | Contents |
|---|---|
| `tx_variants.h` | public API: `TxType`, `TxSize`, `TxParams`, transforms, residual helpers, partition geometry |
| `tx_variants.cpp` | CPU reference implementation |
| `tx_select.h` / `tx_select.cpp` | TX split / type selection hooks (greedy, CPU) |
| `tx_tables.h` | generated §9.6 DCT/ADST/FDST kernels and §9.2 size table (host) |
| `tx_tables_dev.cuh` | generated device copies of the same tables |
| `generate_tx_tables.py` | regenerates both headers from the spec PDF + `spec/tables/block_tables.json` |
| `tx_variants_test.cpp` | CPU self-test (21.5M checks) |
| `build_and_test.sh` | standalone build/run; also invokes the CUDA test when `nvcc` exists |

## Numeric contract

The CPU path is the reference for the CUDA kernels. It follows the existing M0
transform style (double forward, normative integer inverse) with these rules:

| Stage | Arithmetic |
|---|---|
| forward 1D | binary64 accumulation, same loop order as `TileEncoder::forward_dct()`; kernel `K[v][j] / (64*sqrt(n))`; `-ffp-contract=off` |
| IDTX forward | residual passes through unchanged (times the odd-log2 compensation) |
| odd-log2 rectangles | forward multiplies coefficients by `4096/2896` to cancel §7.15.4.1's `Round2(Dequant*2896,12)` |
| quantize | `(int32)lround(coeff * quant_factor)`, `quant_factor = 8*dq_denom/(inv_scale*528)` |
| dequantize | `(abs(q)*528) & 0xFFFFFF`, `Round2(.,3)`, `/ dq_denom`, sign |
| inverse 1D | `sum K[j][i] * src[j]`, int64, `Round2`, row clip `±(1<<15)`, column clip `[-256,255]` |
| IDTX inverse | `Round2(src*identity_scale(log2 coded dim), shift)` (§7.15.2.3) |
| adjusted sizes | §7.15.4 sample duplication; forward residual built by `Round2(sum, log2(area))` averaging |

`scale_w`, `scale_h`, `rect_prescale` and `quant_factor` are computed once in
`make_tx_params()` and carried in `TxParams`; do not recompute them at a call
site or bit-exactness with the CUDA layer is lost.

### Deliberate differences from `src/cuda/transform.*`

The M0 layer is unchanged (another agent owns it). This component is
spec-correct where the M0 path is simplified:

1. **2896 pre-scale.** `src/cuda/transform.*` omits §7.15.4.1's pre-scale for
   `abs(log2W-log2H)` odd. It is applied here, so odd-log2 rectangular DCT
   coefficients differ from the M0 layer by `4096/2896`. Square sizes are
   bit-identical to the M0 arithmetic.
2. **Shifts come from the coded size.** `Transform_Shift[txSz]` is used, not a
   shift inferred from the adjusted size. They coincide for square sizes and
   for 64x64; they differ for e.g. 32x64 (`{6,12}` coded vs 32x32 `{6,13}`).
3. **Generic adjusted sizes.** The 64x64 2x2 rule is generalized to 64x32,
   32x64, 16x64, 64x16, 8x64, 64x8, 4x64, 64x4 (1x2, 2x1 or 4x1 sample
   averaging/duplication). This fixes the M0 blocker where 64x32/32x64 hit an
   out-of-bounds `dct_kernel(64)`.
4. **Length-32 ADST/FDST alias DCT.** §7.15.2.1 defines `sz == 32` with
   `Dct_Kernel32` only. 64x64 is coded as an adjusted 32x32 transform, so
   64x64 ADST/FLIPADST are bit-identical to 64x64 DCT. The API accepts them and
   the tests assert the alias.

Lossless IDTX (§7.15.4 special shift) and DPCM cumulative sums are out of scope
(this is an 8-bit lossy component).

## API

```cpp
namespace av2enc::txv {

enum class TxType : int8_t;     // AV2 TX_TYPE 0..15
enum class Tx1dType : int8_t;   // DCT=0, IDT=1, ADST=2, FDST=3
enum class TxSize : int8_t;     // AV2 TX_SIZE 0..24
enum class TxPartition : int8_t { kNone, kSplit, kHorz, kVert };

struct TxParams {
  int8_t tx_size, tx_type, row_type, col_type;
  int16_t coded_w, coded_h;   // 4..64
  int16_t w, h;               // adjusted (<=32)
  int8_t log2_w, log2_h;      // coded
  int8_t row_shift, col_shift;
  uint8_t dq_denom;
  double scale_w, scale_h, rect_prescale, quant_factor;
};

std::optional<TxParams> make_tx_params(int coded_w, int coded_h, TxType = kDctDct);
std::optional<TxParams> make_tx_params(TxSize, TxType = kDctDct);
bool tx_size_supported(int coded_w, int coded_h);
void tx_1d_types(TxType, Tx1dType* row, Tx1dType* col);
bool make_tx_type(Tx1dType row, Tx1dType col, TxType* out);
const int16_t* tx_kernel(Tx1dType type, int n);  // n=4/8/16/32

void forward_block(const int16_t* residual_adj, const TxParams&, double* coeff);
void quantize(const double* coeff, const TxParams&, int32_t* levels);
void dequantize(const int32_t* levels, const TxParams&, int32_t* dequant);
void inverse_block(const int32_t* dequant, const TxParams&, int16_t* residual_coded);
void forward_transform(...);   // forward + quantize
void inverse_transform(...);   // dequantize + inverse

void downsample_residual(const int16_t* coded, const TxParams&, int16_t* adj);
void upsample_residual(const int16_t* adj, const TxParams&, int16_t* coded);
void make_residual(const uint8_t* src, int stride, int w, int h,
                   int x0, int y0, int pred, const TxParams&, int16_t* adj);
void reconstruct_block(const int16_t* adj, int pred, const TxParams&,
                       uint8_t* dst, int stride, int dw, int dh, int x0, int y0);

// Partition hooks.
bool tx_partition_legal(int w, int h, TxPartition);
int tx_partition_candidates(int w, int h, TxPartition* out, int cap);
int tx_partition_parts(TxPartition);
int tx_partition_geometry(int w, int h, TxPartition, int* xs, int* ys,
                          int* ws, int* hs);

}
```

Buffers are tightly packed, row-major. `forward_block` / `inverse_block` use
the adjusted size for coefficients and emit/accept the coded-size residual
(the inverse expands automatically). `TxParams` is a POD so the same value can
be copied into a CUDA kernel argument.

## Partition selection hooks

`tx_select.h` provides encoder-only decisions; no bitstream syntax is produced.

```cpp
struct TxSelectOptions {
  TxType tx_type = TxType::kDctDct;
  bool search_tx_type = false;  // also try ADST/FLIPADST/IDTX per leaf
  int max_depth = 1;            // 0 = NONE only; 1 = one split level; 2 = two
  double lambda = 0.0;          // cost = sse + lambda * nonzero
};
bool select_tx_partition(const int16_t* residual_coded, int w, int h,
                         const TxSelectOptions&, TxSelection* out);
```

`TxSelection` returns a flat list of leaves (position + `TxParams`), the total
cost, coded-domain SSE and nonzero count. The default cost is
`SSE(forward→quant→dequant→inverse) + lambda * nonzero`, so it is a usable
greedy decision, not just a geometry helper. RDO can instead call
`tx_block_cost()` directly.

Legal partitions produce sub-blocks with dimensions >= 4 that are always one of
the 25 `TX_*` sizes. `SPLIT` is 4 quadrants, `HORZ`/`VERT` are 2 halves.

## Tests

`./build_and_test.sh` (macOS/clang or Linux/g++ works for the CPU half):

```
av2enc CPU transform-variants test
21497851 checks, 0 failures
RESULT: PASS
```

Coverage:

- §9.6 kernel spot values and per-row normalization (all 10 kernel tables,
  every row)
- all 25 sizes: geometry, shifts, `dq_denom`, `rect_prescale`, `quant_factor`,
  size-index round-trip
- all 16 `TX_TYPE` values: `Transform_1d_Type` row/col mapping and inverse
- IDTX forward pass-through and exact inverse scaling for every size, including
  odd-log2 pre-scale
- length-32 ADST/FDST ≡ DCT, including 64x64 adjusted
- impulse coefficients at every position for all sizes (inverse clips and
  adjusted-size duplication)
- quantize `lround` ties and dequantize formula for every size
- `make_residual == downsample(source-pred)`, upsample duplication,
  `reconstruct_block` clipping
- quantization round-trip bounds for DCT/ADST/IDTX/FLIPADST at every size
- 64x64 inverse == 2x2 duplication of the 32x32 inverse (DCT/IDTX/ADST)
- partition geometry: every candidate covers the block exactly, no gaps or
  overlaps, every leaf is a valid TX size
- selector: zero block → NONE/cost 0, split reduces cost on a discontinuity,
  leaves stay inside the block, unsupported roots rejected

The CPU test is also clean under `-fsanitize=address,undefined` (21,497,851
checks, 0 failures).

The CUDA test (`src/cuda/tx_variants/`) links this actual CPU implementation
(not a copy) and compares every stage bit-for-bit.

## Build

```sh
cd src/encoder/tx_variants
./build_and_test.sh
```

The script builds into `${TMPDIR:-/tmp}/av2enc_tx_variants_build`; no build
artifacts are written into the repo. If `nvcc` is present it also builds and
runs the CUDA component test; otherwise it prints a note that the CUDA half
must run on the `sm_86` Linux box.

## End-to-end bit-exactness notes (dav2d)

The transform arithmetic was checked against `dav2d` f9dafbc with forced-DC
streams and a reconstruction harness using **matching encoder options**:

| Stream | Luma samples | Differing samples |
|---|---:|---:|
| 512x64 crop, `--still-depth 0`, DC | 32,768 | **0** |
| 640x360, default depth, DC | 230,400 | **0** |

The 640x360 DC stream exercises 13 TX sizes end-to-end, including odd-log2
rectangles (32x64, 64x32, 64x8, 8x64) that use the §7.15.4.1 2896 pre-scale,
plus 16x16/32x32/64x64 and the rectangular adjusted paths. All match.

**Harness trap:** the `/tmp/dump_m1` helper defaults to
`StillOptions::max_partition_depth = 4`. If the bitstream was encoded with
`--still-depth 0`, the harness reconstructs a different partition tree and reports
spurious differences (e.g. 24,828/230,400 samples on the 640x360 DC stream,
first at a block boundary, range a few LSB). Set
`options.max_partition_depth` to the same value as the encoder (and rebuild the
harness against the current `libav2enc_core.a`) before comparing. With matching
options the same stream is 0/230,400.

### D45 is not a transform bug

Forced D45 on 640x360 (`base_q_idx 90`, no deblock) decodes to 13.901 dB while
the encoder reconstruction is 34.682 dB. The first mismatch is at **(318, 64)**,
inside a **32x8 D45 block at (288, 64)** (trace: `r=16 c=72 size=32x8 eob=51`).
Using the dumped levels/inverse residual for that block:

- the inverse residual `R` is identical when added to both reconstructions;
- encoder prediction and decoder prediction differ in **44/256 samples**, all at
  the block's right edge (row 0 last four: encoder `28 28 28 28`, decoder
  `28 28 27 26`).

That is a directional-prediction / intra-edge-availability difference
(`prepare_edges` / top-right or edge-filter behaviour), not
`txv::inverse_transform` or dequantization. The same 32x8 size is bit-exact in
the forced-DC stream, where the transform residual is the only variable.

A throwaway (uncommitted, in `/tmp`) patch to `still_encoder.cpp` that replaced only
the current-superblock bounds in `count_top_right` / `count_bottom_left`
(`x >= sb_w` / `y >= sb_h`) with frame bounds raised the same D45 stream from
**13.901 dB to 30.410 dB** (diff samples 93,090 → 39,555; still not exact). That
confirms the failure is in intra edge availability, not in this component, and
points the remaining work at the M1/intra predictor.

### Only normative change made here

`dequantize()` now applies the missing §7.14.4 output clip
`Clip3(-(1 << (7 + BitDepth)), ...)` for 8-bit (±32767). The M1 streams in the
tests never exceed it, so all before/after sample counts above are unchanged;
it removes a latent spec deviation for high coefficient levels.


## Integration notes (not done here)

- The encode path still uses `src/cuda/transform.*`. Wiring these variants in is
  an integration-pass job and needs its own `dav2d` gate + MD5 re-pin.
- Odd-log2 rectangular DCT changes coefficients relative to the M0 layer (the
  spec pre-scale). Do not mix the two implementations for the same block.
- The CPU selector is a correctness/reference hook. GPU RDO should batch
  candidates with `txv_forward_transform_multi()` and cost them on-device; the
  CPU variant is not a speed target.
- No `cudaMalloc` is used by the library. The CUDA batch API takes caller-owned
  device buffers; long-lived buffers belong in the arena.
