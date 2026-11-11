# AV2 intra-inter secondary transform (IST)

Normative component: `sec_tx_type` syntax legality, §7.15.3 kernel/coefficient
remapping, and the CPU reference for the CUDA kernel.

- CPU: `src/encoder/tx_secondary/`
- CUDA: `src/cuda/tx_secondary/`
- Spec: AV2 v1.0.0 (commit `38f28e7`), §5.4.8, §5.20.8.2, §6.19.8.2,
  §7.15.3, §9.3, §9.7.

The transform is **luma-only** (`plane == 0`) and **lossy-only**.  It runs on the
dequantized coefficients, before the primary 2D inverse transform:

```
levels -> dequantize -> (IST inverse, plane 0, sec_tx_type != 0) -> 2D inverse
```

The encoder-side projection in this component is explicitly **not normative**:
AV2 v1.0.0 specifies only the decoder inverse.  See “Forward projection” below.

## Files

| File | Contents |
|---|---|
| `tx_secondary.h` | public CPU API |
| `tx_secondary.cpp` | CPU reference (inverse, legality, kernel selection, CDF costs) |
| `tx_secondary_tables.h` | generated §9.2/§9.3/§9.7 host tables |
| `generate_tx_secondary_tables.py` | regenerates the table headers |
| `tx_secondary_test.cpp` | CPU self-test |
| `build_and_test.sh` | standalone build/run; no root CMake edit |
| `../cuda/tx_secondary/tx_secondary_device.cuh` | host+device arithmetic core |
| `../cuda/tx_secondary/tx_secondary_gpu.cu` | one-thread-per-block CUDA kernel |
| `../cuda/tx_secondary/tx_secondary_gpu.h` | CUDA batch API |
| `../cuda/tx_secondary/tx_secondary_test.cu` | real-GPU parity test |
| `../cuda/tx_secondary/tx_secondary_emulation_test.cpp` | host parity test of the device core |

## Tables: what was available, what was generated

The §9.7 kernels are **not** in `spec/tables/*.json` or the PDF body; the spec
references three attachment headers.  They were fetched from the normative
source and parsed:

| Source | SHA-256 | Contents |
|---|---|---|
| `https://av2.aomedia.org/v1.0.0/attachments/ist_4x4_kernel.h` | `9fafc6bc5086c992dabc1b26816713f9a56e5ccdba386130034165d6df985404` | `Ist_4x4_Kernel[14][3][8][16]` |
| `.../ist_8x8_kernel.h` | `65d17c0e9a34843da3f5fe48262c7170383b59c15e477ab55733796de871fcc4` | `Ist_8x8_Kernel[11][3][32][48]` |
| `.../stx_scan_map.h` | `8825c77db8b394c9274df8cb368c14271d12eed218eb228aae133956c4ec7146` | `Stx_Scan_Map[11][3][48]` |

The generator also reads:

- `spec/tables/constants.json`: `STX_TYPES`, `IST_*`, `TX_SIZES`
- `spec/tables/block_tables.json`: `Tx_Width/Height/Width_Log2`, `Tx_Size_Sqr`,
  `Mode_To_Angle`
- `spec/tables/default_cdf.json`: `Default_Sec_Tx_Type_Cdf`,
  `Default_Most_Probable_Stx_Set_Cdf`,
  `Default_Most_Probable_Stx_Set_Adst_Cdf`
- `spec/tables/_cache/spec.html`: §7.15.3 mapping tables and `Stx_Scan_Order_*`

The generated headers are committed.  Normal builds need neither the PDF nor
network access.  Regenerate with:

```sh
cd src/encoder/tx_secondary
python3 generate_tx_secondary_tables.py --attachments /path/to/attachments
python3 generate_tx_secondary_tables.py --check --attachments /path/to/attachments
```

No table was guessed or substituted.

## Spec facts pinned by this component

### When `sec_tx_type` is coded (§5.20.8.2)

```
plane == 0 && !Lossless
TxType in {DCT_DCT, ADST_ADST}
YMode != PAETH_PRED
eob <= eobLim
if inter:  enable_inter_ist && eob > 3 && TxType == DCT_DCT
           && Tx_Width >= 16 && Tx_Height >= 16
else:      enable_intra_ist && eob != 1
```

`eobLim` is exactly `n`:

| Block | `large` | `n` |
|---|---|---|
| `w = Min(32,W) < 8 || h = Min(32,H) < 8` | false | 8 (`IST_4X4_HEIGHT`) |
| `txSz == TX_8X8` or `PlaneTxType == ADST_ADST` | true | 20 (`IST_8X8_HEIGHT_RED`) |
| otherwise | true | 32 (`IST_8X8_HEIGHT`) |

Chroma blocks never code `sec_tx_type`, and the reconstruction step invokes the
secondary transform only for `plane == 0`.  `apply_inverse_secondary()` therefore
returns `kNotApplicable` for planes 1/2.

### Kernel selection (§7.15.3)

```
mode = YMode
if directional: pAngle = Mode_To_Angle[mode] + AngleDeltaY*3
                        + Mrl_Index_To_Delta[MrlIndex]
                mode = wide_angle_mapping(mode, Tx_Width, Tx_Height, pAngle)
if is_inter: kernel = 0
else if PlaneTxType == ADST_ADST && Tx_Width >= 8 && Tx_Height >= 8:
    kernel = Inv_Most_Probable_Stx_Mapping_Adst[mode][set]
else:
    kernel = Inv_Most_Probable_Stx_Mapping[mode][set]
if PlaneTxType == ADST_ADST: kernel += 7
transpose = mode in {H_PRED, D157_PRED, D67_PRED, SMOOTH_H_PRED}
```

`most_probable_stx_set` uses 4 symbols (`0..3`) for a large ADST_ADST block and
7 symbols (`0..6`) otherwise.  Inter blocks do not code it.  This is why the
candidate enumeration returns `3 * 4` or `3 * 7` intra entries but only `3` inter
entries.

### Inverse remap (§7.15.3)

`Dequant` is the adjusted `w*h` raster block (`w = Min(32,W)`,
`h = Min(32,H)`, stride `w`):

1. `scanIn = get_scan(txSz, TX_CLASS_2D)`; copy `coefs[j] = Dequant[scanIn[j]]`
   for `j < n`, then zero those positions.
2. For `i < n2` (`n2 = 16` for 4x4, `48` for 8x8):
   `t = sum_j coefs[j] * Ist_kernel[j][i]`;
   `v = Clip3(-(1<<(BitDepth+7)), (1<<(BitDepth+7))-1, Round2Signed(t,7))`.
3. `pos = (large ? Stx_Scan_Order_8x8[Stx_Scan_Map[kernel][sec-1][i]]
                  : Stx_Scan_Order_4x4[i])`;
   `x = pos & (scanW-1)`, `y = pos >> scanBwl`, `scanW = large ? 8 : 4`;
   `Dequant[y][x] = v` or, if transposed, `Dequant[x][y] = v`.

Non-transposed linear index is `y*w+x`; transposed is `x*w+y`.

## API

```cpp
namespace av2enc::txsec {

struct SecondaryTransformConfig {
  bool enable_intra_ist = false;   // §5.4.8
  bool enable_inter_ist = false;   // §5.4.8
};
void write_secondary_transform_header_bits(BitWriter&, const SecondaryTransformConfig&);

struct SecondarySizeInfo {
  int tx_size, coded_w, coded_h, w, h, bwl;
  bool large;
  int n, n2, scan_w, kernel_count, eob_limit;
};
bool secondary_size_info(const TxParams&, TxType plane_tx_type,
                         SecondarySizeInfo*);
int  secondary_eob_limit(const TxParams&, TxType plane_tx_type);

int  map_y_mode(int y_mode, int angle_delta, int mrl_index,
                int coded_w, int coded_h);
int  most_probable_stx_set_count(TxType, int coded_w, int coded_h);
bool kernel_for_syntax(int mapped_mode, TxType, int set, bool is_inter,
                       int coded_w, int coded_h, int* kernel);
bool transpose_for_mode(int mapped_mode);

struct SecondarySyntax {
  TxType plane_tx_type = TxType::kDctDct;
  int coded_w, coded_h;
  int y_mode, angle_delta, mrl_index;
  int most_probable_stx_set, sec_tx_type;
  bool is_inter, lossless;
  int bit_depth;
};
bool secondary_syntax_applicable(int plane, int eob,
                                 const SecondarySyntax&,
                                 const SecondaryTransformConfig&);
bool secondary_syntax_valid(int plane, int eob, const SecondarySyntax&,
                            const SecondaryTransformConfig&);

struct SecondaryCandidate { int sec_tx_type, most_probable_stx_set,
                             kernel; bool transpose; };
int enumerate_secondary_candidates(const SecondarySyntax&,
                                   const SecondaryTransformConfig&, int eob,
                                   SecondaryCandidate*, int cap);

bool inverse_secondary_dequant_kernel(const TxParams&, TxType, int kernel,
                                      int sec_tx_type, bool transpose,
                                      int bit_depth, std::int32_t* dequant,
                                      std::size_t len);
SecondaryStatus apply_inverse_secondary(const TxParams&, int plane, int eob,
                                        const SecondarySyntax&,
                                        const SecondaryTransformConfig&,
                                        std::int32_t* dequant,
                                        std::size_t len);
bool forward_secondary_coeffs_kernel(const TxParams&, TxType, int kernel,
                                     int sec_tx_type, bool transpose,
                                     int bit_depth, std::int32_t* coeff,
                                     std::size_t len);

const std::uint16_t* sec_tx_type_cdf(int tx_size, bool is_inter);
const std::uint16_t* most_probable_stx_set_cdf(TxType, int coded_w, int coded_h);
double sec_tx_type_bits(int symbol, int tx_size, bool is_inter);
double most_probable_stx_set_bits(int symbol, TxType, int coded_w, int coded_h);
}
```

`apply_inverse_secondary()` returns:

| Status | Meaning |
|---|---|
| `kNotApplicable` | no `sec_tx_type`, or the §5.20.8.2 rules disallow IST; `dequant` untouched |
| `kApplied` | inverse remap applied |
| `kInvalid` | malformed kernel/`sec_tx_type`/set/bit depth; `dequant` may have been modified only after validation passes |

The low-level `*_kernel()` entry points do not re-derive the kernel, so tests and
future RDO can exercise every §9.7 set directly.

## Forward projection (encoder side, not normative)

`forward_secondary_coeffs_kernel()` reads the `n2` output positions from a full
primary-coefficient block, computes

```
sec[j] = Round2Signed(sum_i coeff[out_pos(i)] * Ist_kernel[j][i], 7)
```

at the first `n` scan positions, and zeros the rest.  This is the transpose of
the inverse matrix with the same shift, i.e. the least-squares projection when
the kernel rows are orthonormal.  The rows are orthonormal to within rounding
(`Ist M M^T / 128^2` is the identity within ~1.3%), so this is a usable encoder
analysis tool, but **AV2 v1.0.0 does not define it and no bitstream claim is made
for it**.  Decoder correctness depends only on the normative inverse.

## Tests

CPU:

```sh
cd src/encoder/tx_secondary
./build_and_test.sh
SANITIZE=1 WERROR=1 ./build_and_test.sh
```

Result on macOS/Apple clang 21:

```
158,153,173 checks, 0 failures
RESULT: PASS
```

(and the same count under ASan+UBSan with `-Werror`).

Coverage:

| Suite | Checks | What it pins |
|---|---:|---|
| `tables` | 115,481 | §9.7 shapes/values, scan permutations, mappings, CDF values |
| `size_info` | 1,143 | all 25 sizes × DCT/ADST/IDTX: `w/h/bwl/large/n/n2/scan_w/kernel_count/eob_limit` |
| `mode_mapping` | 906 | YMode × AngleDeltaY × MrlIndex × wide-angle thresholds, PAETH rejection |
| `kernel_selection` | 24,255 | every mode/set/type/inter combination, ADST +7, 4x4/8x8 ranges, all kernels reachable |
| `candidate_enum` | 13,313 | all legal `(sec_tx_type, set)` pairs, dedup, cap handling |
| `syntax_legality` | 49,234 | full EOB × plane × type × enable-bit matrix vs independent condition |
| `inverse_kernels` | 142,735,949 | every kernel × sec_tx_type × transpose × all 25 sizes × zero/DC/impulse/extreme/random patterns vs direct-from-spec reference |
| `eob_interaction` | 13,813,641 | eob 1..n, zero tail, eob > eobLim rejection |
| `forward_projection` | 1,170,130 | forward parity vs reference, inverse stability, mean projection error |
| `apply_luma_only` | 228,827 | high-level apply, chroma/lossless/PAETH/eob/inter rules |
| `cdf_header` | 31 | CDF pointers/costs, the two §5.4.8 bits |
| `determinism` | 263 | repeatability, too-small buffer rejection |

CUDA:

```sh
cd src/cuda/tx_secondary
./build_and_test.sh
CUDA_ARCH=sm_86 ./build_and_test.sh
```

- `tx_secondary_emulation_test.cpp` compiles the **exact device core**
  (`tx_secondary_device.cuh`) with the host compiler and compares it with the
  CPU reference: **11,504,810 checks, 0 failures** (36,240 inverse+forward
  kernel cases).
- `tx_secondary_test.cu` is the real one-thread-per-block GPU test: all 25
  sizes × every kernel × sec 1..3 × transpose 0/1, batch/in-place, CPU-vs-GPU
  bit-exact.  This host has no `nvcc`, so the real GPU test was not run here;
  the `.cu` and kernel translation unit were syntax-checked with clang’s CUDA
  frontend and a stub runtime.  Run it on the sm_86 Linux box before claiming
  GPU parity.

## Decoder gate

**No end-to-end decoder gate is possible without core edits.**  The current M1
writer hardcodes both header bits to 0 and has no `sec_tx_type` syntax path, so a
minimal IST stream cannot be emitted from this component alone.  The exact call
sites are:

### 1. Sequence header bits (§5.4.8)

- `src/encoder/m1/m1_headers.h` `struct M1Config` (line ~58): add
  `bool enable_intra_ist = false; bool enable_inter_ist = false;` (or embed
  `txsec::SecondaryTransformConfig`).
- `src/encoder/m1/m1_headers.cpp` lines 131–133: replace the two hardcoded
  `writer.put_bit(false)` calls with
  `txsec::write_secondary_transform_header_bits(writer, config.secondary)` after
  the existing `enable_idtx_intra` bit.
- Same two bits in `src/bitstream/m0_headers.cpp` lines 238–240 and
  `src/encoder/pipeline/tiled_still_gate.cpp` lines 179–181 if those writers are
  used for the gate.
- `src/encoder/m1/transform_stream/transform_stream.h` `Request` (line ~156):
  add the two flags so the GPU stream path can carry them.

Order is exactly `enable_intra_ist`, then `enable_inter_ist`.

### 2. `sec_tx_type` / `most_probable_stx_set` syntax (§5.20.8.2)

- `src/encoder/m1/m1_syntax.h` `struct BlockSyntax` (line ~73): add
  `std::uint8_t sec_tx_type[kPlaneCount]` and
  `std::uint8_t most_probable_stx_set[kPlaneCount]` (chroma entries stay 0), or
  a single luma-only pair plus `mrl_index`.
- `src/encoder/m1/m1_syntax.cpp` `CdfState` (line ~200): add
  `sec_tx_type[2][5][5]`, `most_probable_stx_set[8]`,
  `most_probable_stx_set_adst[5]`; initialize them from
  `txsec::kSecTxTypeCdf`, `kMostProbableStxSetCdf`, `kMostProbableStxSetAdstCdf`
  in `init_cdfs()` (~line 250).
- `src/encoder/m1/m1_syntax.cpp` `Writer::write_leaf()` after
  `write_eob(...)` and the existing luma `intra_tx_type` block
  (lines ~1059–1115): build `txsec::SecondarySyntax` from `node.tx_type[0]`,
  `node.y_mode`, `node.y_angle_delta`, `MrlIndex` (currently absent; use 0 until
  MRL is wired), and `eob`; call
  `txsec::secondary_syntax_applicable(0, eob, syntax, config)`.  If true and
  `sec_tx_type != 0`, code
  `sym(&TileSecTxTypeCdf[is_inter][kTxSizeSqr[tx_size]][0], 4, sec_tx_type)`;
  for intra non-inter and nonzero, code
  `most_probable_stx_set` with the 7- or 5-entry CDF selected by
  `txsec::most_probable_stx_set_cdf()`.
- Chroma (`plane > 0`) must not enter this block.

### 3. Reconstruction / encoder decisions

- `src/encoder/m1/m1_still.cpp` `TxPlan` (line ~120): add
  `sec_tx_type`, `most_probable_stx_set`, `mrl_index`.
- `src/encoder/m1/m1_still.cpp` `dequant_inverse_qindex()` (line ~765): after
  building the `dequant` array and before `txv::inverse_block()` (line ~784),
  call `txsec::apply_inverse_secondary()` for plane 0.  The fast path at line
  ~770 (`txv::inverse_transform`) must be replaced by explicit
  dequantize → IST → `txv::inverse_block` whenever IST applies.
- `src/encoder/m1/m1_still.cpp` `compute_tx()` (line ~787): after the TX_TYPE
  decision, choose `sec_tx_type`/`most_probable_stx_set` (e.g. with
  `enumerate_secondary_candidates()` + the forward projection), enforce
  `eob <= txsec::secondary_eob_limit()`, and write the first `n` scan positions
  as the coded levels.  `BlockSyntax::levels_offset`/`eob` already carry
  scan-order levels, so no core layout change is needed beyond the new fields.
- `src/encoder/m1/m1_syntax.h` `assemble_m1_frame()` already receives
  scan-order levels; the writer must not code the synthesized `n2` positions
  because the decoder creates them from the first `n` levels.
- GPU stream/transform pipelines (`src/encoder/m1/transform_stream/`,
  `src/cuda/m1/`, `src/encoder/m1/transform_stream/transform_stream_gpu.cu`)
  need the same fields and a call to `txsec::txsec_inverse_dequant()` between
  dequant and primary inverse.  The CUDA batch API is:

```cpp
av2enc::cuda::TxSecondaryJob job;
job.tx_size = tx_size;
job.plane_tx_type = static_cast<int>(plane_tx_type);
job.kernel = resolved_kernel;
job.sec_tx_type = sec_tx_type;
job.transpose = transpose ? 1 : 0;
job.bit_depth = 8;
job.in_offset  = dequant_element_offset;
job.out_offset = dequant_element_offset;
av2enc::cuda::txsec_inverse_dequant(d_dequant_in, d_dequant_out, d_jobs,
                                    njobs, stream);
```

No `cudaMalloc` is used by the library; the job array and buffers are
caller-owned, and the launcher does not synchronize.

### 4. `main.cpp`

Add `--enable-intra-ist` / `--enable-inter-ist` (or `--enable-secondary-transform`)
and thread them into `M1Options`/`M1Config`.  `src/main.cpp` was not edited.

Once 1–3 are wired, gate a stream with `tools/validate_dav2d.sh` and re-pin the
M1 MD5s; until then the M1 reference bitstreams are unchanged.

## Limitations

- The M1 `BlockSyntax` has no `MrlIndex` field yet; use `mrl_index = 0` until the
  MRL component is wired.  The API already implements the full
  `Mrl_Index_To_Delta` term.
- The CPU transform layer (`tx_variants`) is an 8-bit lossy reference.  The IST
  API accepts `bit_depth` 8..12 for the normative clip, but the surrounding
  dequant/inverse path is only tested at 8-bit.
- No encoder rate-distortion decision is included (only forward projection,
  candidate enumeration, and §9.3 static symbol costs).  Choosing the best
  `sec_tx_type`/set is an integration decision; the spec does not prescribe it.
- No stream/MD5/PSNR claim is made here.
