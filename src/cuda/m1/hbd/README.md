# 10-bit M1 path (`src/cuda/m1/hbd/`)

Status: **integration-ready component, not integrated.** This directory plus
`src/encoder/format/` and `src/cuda/format/` implements the 10-bit 4:2:0
sample, geometry, transform/quant, dequant and inverse-reconstruction path. It
does not write a header or entropy payload, and **no 10-bit stream has been
encoded or decoded in this tree**. The M1 header writer still hard-codes
`bit_depth_idc = 1` (8-bit) and an 8-bit `base_q_idx`, so a 10-bit still
requires the integration steps below. Do not claim 10-bit before
`gate_10bit_still.sh` passes.

12-bit is not signalable in AV2 v1.0.0 (`bit_depth_idc` is one bit:
`0 = 10-bit`, `1 = 8-bit`; greater values are reserved, §6.4.1 Table 6.3).
The 12-bit arithmetic in this component exists only so the tests can prove the
bounds generalize. **Do not emit a 12-bit sequence header.**

## Files

| File | Purpose |
|---|---|
| `hbd_cpu.h/.cpp` | CPU oracle: depth-aware dequant/inverse clips, DC bias, EOB, whole-block reference |
| `hbd_tx.h/.cu` | batched GPU forward/inverse TX; `uint16_t` source, depth-parameterized clips |
| `hbd_frame.h/.cu` | 10-bit 4:2:0 geometry, 8→10 upload + padding, 10→8 download |
| `hbd_test.cpp` | CPU-only tests; proves `bit_depth=8` is bit-identical to `tx_variants` |
| `hbd_test.cu` | CPU-vs-GPU element-for-element tests (upload/download, mixed TX batch) |
| `build_hbd_test.sh` | standalone CPU test, plus CUDA test when `nvcc` exists |
| `gate_10bit_still.sh` | `dav2d` decode + `C420p10` header/frame-size gate for an IVF |

## Spec rules used (AV2 v1.0.0, commit `38f28e7`)

| Rule | 8-bit | 10-bit | Clause |
|---|---:|---:|---|
| `bit_depth_idc` | 1 | **0** | §6.4.1 Table 6.3 |
| `MaxQ` (`MAXQ_8_BITS + 2·MAXQ_OFFSET`, offset 24) | 255 | **303** | §6.4.1 Table 6.3 |
| `base_q_idx` field width | `f(8)` | **`f(9)`** | §5.18.6.1 |
| `Clip1` high | 255 | **1023** | §4.8 |
| DC prediction neutral value | 128 | **512** | §7.13.2.10 |
| Dequant clip | ±(1<<15) | **±(1<<17)** | §7.14.3 |
| Inverse row clip | ±(1<<15) | **±(1<<17)** | §7.15.2.1 |
| Inverse column clip | ±(1<<8) | **±(1<<10)** | §7.15.2.1 |
| Residual representable range | ±2^8 | **±2^10** | §7.14.4 |
| `Transform_Shift` | unchanged | **unchanged** | §7.15.4.1 |
| Coefficient CDF slice | raw `base_q_idx` ≤90 → 0 | same, no depth normalization | §8.3.1 |
| `chroma_format_idc` 4:2:0 | 0 | 0 | §6.4.1 Table 6.2 |
| `seq_profile_idc` 4:2:0 | 0 | 0 | Annex A.2 |
| Deblock q index offset | 0 | **24·(10−8) = 48** | §7.17 |
| CDEF `coeffShift` | 0 | **2** | §7.18 |

`fmt::format_requirements()` and `fmt::bit_depth_limits()` expose these values;
`fmt::format_checklist()` returns the machine-readable integration checklist.

## API

### Format layer (`src/encoder/format/`)

Already present and tested; the 10-bit additions are the depth-arithmetic
helpers:

```c++
namespace av2enc::fmt {
struct BitDepthLimits {
  int bit_depth, sample_max, dc_bias;
  int dequant_min, dequant_max;
  int inverse_row_min, inverse_row_max;
  int inverse_col_min, inverse_col_max;
  int residual_min, residual_max;
};
BitDepthLimits bit_depth_limits(BitDepth depth);
int clip1_sample(std::int64_t, BitDepth);
int clip_dequant(std::int64_t, BitDepth);
int clip_inverse_row(std::int64_t, BitDepth);
int clip_inverse_col(std::int64_t, BitDepth);

// 10-bit 4:2:0 picture from an 8-bit 4:2:0 source (visible + padded geometry)
ConvertResult convert_420_8_to_picture(const Source420_8&, ChromaFormat::k420,
                                       BitDepth::k10, Picture&);
}
```

`Picture` stores `uint16_t` samples at coded (MI-aligned) size; chroma is
`(width+1)>>1` by `(height+1)>>1` visible, padded by edge replication.

### CPU oracle (`hbd_cpu.h`)

```c++
namespace av2enc::hbd {
int dc_pred_bias(int bit_depth);            // 512 at 10-bit
int clip1(int value, int bit_depth);
int clip_dequant(std::int64_t, int bit_depth);
int clip_inverse_row(std::int64_t, int bit_depth);
int clip_inverse_col(std::int64_t, int bit_depth);

int scan_pos(int c, int w, int h);
void build_scan(int w, int h, std::uint16_t* scan);
int eob_from_levels(const std::int32_t* levels, int w, int h);

void downsample(const std::int16_t* coded, const TxParams&, std::int16_t* adj);
void forward(const std::int16_t* adj, const TxParams&, double* coeff,
             std::int32_t* levels);
void dequantize(const std::int32_t* levels, const TxParams&,
                std::int32_t* dequant, int bit_depth);
void inverse_block(const std::int32_t* dequant, const TxParams&,
                   std::int16_t* residual_coded, int bit_depth);

struct BlockResult { int eob; std::vector<std::int32_t> levels, dequant;
                     std::vector<std::int16_t> residual;
                     std::vector<std::uint16_t> recon; };
bool encode_block(const std::uint16_t* src, int stride, int src_w, int src_h,
                  int x0, int y0, const std::uint16_t* pred,
                  const TxParams&, int bit_depth, BlockResult*);
bool encode_block_residual(const std::int16_t* coded_residual,
                           const std::uint16_t* pred, const TxParams&,
                           int bit_depth, BlockResult*);
}
```

The forward transform and quantizer delegate to `txv::forward_transform()` so
the 10-bit path cannot drift from the 8-bit reference. Only dequant and the
inverse clips are depth-parameterized.

### GPU (`hbd_tx.h`, `hbd_frame.h`)

```c++
namespace av2enc::cuda::hbd {
bool hbd_tx_init_job(HbdTxJob*, int coded_w, int coded_h, TxType);
cudaError_t hbd_tx_batch(const HbdTxJob* d_jobs, int nblocks,
                         const HbdTxBuffers&, int bit_depth,
                         int smem_elems, cudaStream_t = nullptr);

bool hbd_make_420_frame_10(uint32_t w, uint32_t h, Hbd420Frame*,
                           std::string* error);
cudaError_t hbd_upload_420_8(const fmt::Source420_8&, const Hbd420Frame&,
                             std::uint16_t* d_frame, cudaStream_t = nullptr);
cudaError_t hbd_download_420_8(const Hbd420Frame&, const std::uint16_t*,
                               const HbdPlane8View out[3],
                               cudaStream_t = nullptr);
cudaError_t hbd_replicate_420_borders(const Hbd420Frame&, std::uint16_t*,
                                      cudaStream_t = nullptr);
}
```

`HbdTxJob`/`HbdTxBuffers` mirror `M1TxJob`/`M1TxBuffers`; the only signature
difference is `uint16_t` source/prediction and the `bit_depth` argument.
Buffers are caller-owned, no allocation, no synchronization, safe for CUDA
graphs. `smem_elems` is the largest adjusted area in the batch (0 = 32×32 max).

## Build integration

There is no `CMakeLists.txt` change to make. The root target globs
`src/cuda/m1/**/*.{cu,cpp}`, so `hbd_cpu.cpp` and `hbd_tx.cu` are compiled into
`av2enc_m1_gpu` automatically; `*_test.*` files are filtered out. Those two
translation units only need `tx_variants.cpp` (already in `av2enc_core`) and
the CUDA runtime. `hbd_frame.cu` deliberately computes the §5.18.4.4 geometry
itself instead of calling `fmt::make_frame_geometry()`, because `format.cpp`
is not part of the root M1 target; `hbd_test.cu` pins the two geometries
against each other. Run `build_hbd_test.sh` for the standalone tests.

## Parity and test status

Run:

```sh
./build_hbd_test.sh
```

On this macOS host (no `nvcc`): `hbd_test: PASS (19312 checks)`. The CUDA half
was not run here; run it on the `sm_86` box where `nvcc` is present. It compares
every element (levels, EOB, dequant, inverse residual, every uploaded sample,
every downloaded sample), not a checksum.

What the CPU test pins:

| Group | Assertion |
|---|---|
| bounds | 8/10/12-bit `Clip1`, dequant, row/column clips, DC bias; `fmt` and `hbd` agree |
| 8-bit parity | `hbd::dequantize` + `hbd::inverse_block` are **identical** to `txv::dequantize` + `txv::inverse_block` for 13 sizes × 4 TX types |
| 10-bit clips | huge DC level clips dequant to `(1<<17)-1`, 8-bit to `(1<<15)-1`; inverse columns stay in ±1023 (10-bit) / ±256 (8-bit) |
| scan/EOB | `scan_pos` == `build_scan` for all square 4..32 sizes; EOB is the 1-based last non-zero scan index |
| block ref | zero residual → EOB 0, recon == pred; 10-bit source/pred recon stays in [0,1023] |
| 4:2:0 | even/odd/640x360/641x361 geometry, luma shift, replicated padding, range check |

The CUDA test adds:

* upload of 64x64, 67x43, 640x360 and 641x361 against
  `fmt::convert_420_8_to_picture()`, every coded sample including padding;
* download against `fmt::convert_to_420_8()`, every visible sample;
* one mixed batch of 23 sizes × 5 TX types per depth (8/10/12), alternating
  source/pred and direct-residual input, packed and non-tight plane addressing.

## Integration checklist for the M1 owner

Everything below is outside this component's ownership. Do them in order; the
first stream should be 64x64, QP 90, no deblock, DC/CfL chroma.

### 1. Header writer — `src/encoder/m1/m1_headers.{h,cpp}`

- Widen `M1Config::base_q_idx` from `uint8_t` to `uint16_t` (MaxQ is 303 at
  10-bit; 303 does not fit).
- Add the bit depth to `M1Config` (e.g. `fmt::BitDepth depth` or
  `std::uint8_t bit_depth_idc`) and validate it: 8 → idc 1, 10 → idc 0,
  12 → **reject** (reserved).
- In `write_sequence_header()`, replace the hard-coded `put_uvlc(writer, 1)`
  for `bit_depth_idc` with `put_uvlc(writer, idc)`; keep `chroma_format_idc = 0`
  and `seq_profile_idc = 0` for 4:2:0.
- In `write_frame_header()`, replace `writer.put_bits(c.base_q_idx, 8U)` with
  `n = (BitDepth == 8) ? 8 : 9`; write `f(n)`. Assert `1 <= base_q_idx <= MaxQ`.
- Keep `single_picture_header_flag` and all M0/M1 tool disables unchanged.
  No colour description syntax moves into the sequence header.

### 2. Options / CLI — `src/encoder/m1/m1_still.h`, `src/main.cpp` (owner: agent 9)

- `M1Options::base_q_idx` must widen to `uint16_t`; validate against
  `fmt::max_quantizer(depth)`.
- Add a depth option (`--bit-depth 10`) and an input path that accepts 8-bit
  4:2:0 today, converting with `fmt::convert_420_8_to_picture()`.
- For the first 10-bit stream keep `base_q_idx = 90`: the M1 coefficient
  entropy tables in `m1_syntax.cpp` are a single slice (qctx 0). Higher
  qindexes require CDF slices 1–3 first (see §4 below).

### 3. Reconstruction / transform — `m1_still.cpp` and the M1 GPU stages

- Sample containers become `uint16_t`: `Predictor` samples, `source_sample`,
  `TxPlan::pred/recon`, `Leaf` recon, `M1Result::recon_y/u/v`, and the
  `uint8_t` PODs in `src/cuda/m1/m1_gpu.h`.
- Replace the hard-coded `clip1(v) { clip3(0, 255, v); }` with
  `fmt::clip1_sample(v, depth)` (1023 at 10-bit).
- Replace the `128` DC neutral value with `1 << (BitDepth - 1)` = **512**.
  This includes the encoder's DC predictor, the intra-mode reference fill, and
  the CfL luma reference initialization. Initializing recon to 0 is still
  correct (the decoder's value); do not use 128.
- Swap the 10-bit TX stage to `cuda::hbd::hbd_tx_batch(..., bit_depth=10, ...)`
  or fold the `bit_depth` argument into `src/cuda/m1/tx/m1_tx.*`. Do **not**
  call `txv::dequantize`/`txv::inverse_block` for 10-bit: they clip at the
  8-bit bounds (`±(1<<15)`, `±256`). `hbd_cpu` is the 10-bit oracle.
- Residuals stay `int16_t` (10-bit range ±1024). Dequant is `int32_t`;
  10-bit inverse-row intermediates still fit `int32_t`.
- `Transform_Shift`, `dq_denom`, scan and EOB are unchanged; only the clips
  change. The forward quantizer (`quant_factor`) is depth independent.
- Keep `-ffp-contract=off` on the CPU reference and `-fmad=false` on CUDA (the
  existing M1 rule); contract changes flip double-transform coefficients.
- Widen `src/cuda/m1/recon/*`, `src/cuda/m1/cfl/*` and
  `src/cuda/m1/chroma/*` from `uint8_t` to `uint16_t` (or add parallel HBD
  entry points) before wiring stage 1. CfL reads reconstructed luma below/right
  of the block, so it must see 10-bit luma, not a truncated copy.
- Deblock is still 8-bit. For the first 10-bit gate run with deblock disabled
  (`--no-deblock` / `enable_deblock = false`). When enabled, subtract
  `24 * (BitDepth - 8) = 48` from the q index and use `(1<<(12-BitDepth))` /
  `>> (13-BitDepth)` for `side` (§7.17). CDEF/CCSO/LR are out of scope here.

### 4. Entropy — `src/encoder/m1/m1_syntax.cpp`, `src/entropy/av2/`

- **No CDF table changes are required for the first stream.** Coefficient
  alphabets and default CDFs are bit-depth independent. `init_coeff_cdfs()`
  selects the slice from the raw `base_q_idx` (`≤90 → 0`, `≤140 → 1`,
  `≤190 → 2`, else 3, §8.3.1); do not normalize by bit depth.
- The current M1 writer is effectively qctx 0 only. Keep `base_q_idx = 90`
  until slices 1–3 are generated and tested; 10-bit may legitimately use up to
  303, but that is a separate entropy task.
- 10-bit coefficient magnitudes are up to ~4× larger. Add a unit test that
  round-trips levels above 255 through the existing base/br/escape path; the
  syntax itself does not change.
- Header parse only: `base_q_idx` is 9 bits on the wire; the entropy coder sees
  the same integer. CDF reset/adaptation remains per tile.

### 5. Gate

1. Build and run the component tests above.
2. After integration, encode a 64x64 10-bit 4:2:0 still (QP 90, DC/CfL,
   no deblock). Run:

   ```sh
   ./src/cuda/m1/hbd/gate_10bit_still.sh out.ivf [source.y4m]
   ```

   It requires `dav2d` exit 0, a `C420p10` decoded header, and a body that is a
   whole number of 10-bit 4:2:0 frames.
3. Compare all three decoded/recon planes: **0 LSB mismatch**. The recon dump
   must be `uint16_t` at the same visible geometry.
4. Pin the IVF MD5 and record the `dav2d` commit. Repeat for 640x360, then
   641x361 (odd edges) before claiming 10-bit support.

## What this component intentionally does not do

- No sequence/frame header serialization, entropy symbols, or bitstream.
- No 10-bit recon-plane writer, CfL, deblock, CDEF, CCSO or LR.
- No 4:2:2/4:4:4 path in this directory (the format layer supports the
  geometry, but the M1 10-bit deliverable is 4:2:0 only).
- No 12-bit bitstream. `bit_depth_idc > 1` is reserved.
