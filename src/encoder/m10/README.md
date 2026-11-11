# M10 — standalone 10-bit 4:2:0 still picture

Status (2026-11-11): **the CPU path produces a 10-bit stream that `dav2d`
decodes, and every decoded sample equals the encoder reconstruction (0 LSB)**
on 64x64 and 640x360.  The GPU block stage and its CPU-vs-GPU test are written
but have **not been executed on this macOS host** (no `nvcc`); run
`src/cuda/format/build_m10_gpu_test.sh` on the Linux `sm_86` box before claiming
GPU parity.

M10 is deliberately independent of the M1 decision/reconstruction core
(`src/encoder/m1/m1_still.*`, `m1_syntax.*`, `m1_gpu.*`).  It reuses only:

* `src/encoder/format/` — 10-bit sample model, geometry, conversions, bounds;
* `src/cuda/m1/hbd/hbd_cpu.*` — depth-aware dequant/inverse/recon CPU oracle;
* `src/cuda/m1/hbd/hbd_tx.*` — GPU transform/quant/dequant/inverse kernel;
* `src/encoder/tx_variants/` (forward transform/quant, scan-order levels);
* `src/entropy/av2/` (normative §8.2 symbol coder) and the existing CDF tables.

No 12-bit stream is produced or implied.  AV2 v1.0.0 `bit_depth_idc` is one
bit (`0 = 10-bit`, `1 = 8-bit`); values above 1 are reserved.  The 12-bit
arithmetic in `format.h` / `hbd_cpu.h` exists only so the bounds tests
generalize.

## Tool subset

| Item | M10 |
|---|---|
| Picture | one intra key frame, one tile, 4:2:0, 10-bit |
| Superblock | 64x64 |
| Full SB | `PARTITION_VERT` -> two 32x64 -> `PARTITION_HORZ` -> four 32x32 |
| Partial SB | `PARTITION_NONE` at 64x64 |
| Intra | DC only; `enable_intra_edge_filter = 0`, no CfL/MRL/DIP/IBP |
| Transform | DCT_DCT, `TX_MODE_LARGEST` (32x32 + 16x16 chroma; 64x64 edge) |
| Quant | raw `base_q_idx` = 90; the forward quantizer is the fixed QP-90 reference |
| Filters | none (no deblock/CDEF/CCSO/GDF/restoration/film grain) |
| Entropy | normative §8.2, frozen CDFs by default |

## 10-bit header/bitstream fields

| Field | Value | Clause |
|---|---|---|
| `bit_depth_idc` | **0** (M0 hard-codes 1) | §6.4.1 Table 6.3 |
| `MaxQ` | **303** (derived; validated against `base_q_idx`) | §6.4.1 Table 6.3 |
| `base_q_idx` | `f(9)` bits, value 90 | §5.18.6.1 |
| `chroma_format_idc` | 0 (4:2:0) | §6.4.1 Table 6.2 |
| `seq_profile_idc` | 0 | Annex A.2 |
| coefficient CDF slice | chosen from **raw** `base_q_idx` (<=90 -> 0) | §8.3.1 |
| `Clip1` | 0..1023 | §4.8 |
| DC bias | 512 | §7.13.2.10 |
| dequant clip | `+/-(1 << 17)` | §7.14.3 |
| inverse row clip | `+/-(1 << 17)` | §7.15.2.1 |
| inverse column clip | `+/-(1 << 10)` | §7.15.2.1 |

`m10_headers.cpp` writes the whole sequence/frame/tile-group OBU itself; it does
not call `m1_headers.*`.  It reuses the generic OBU framing helpers in
`bitstream/m0_headers.h` (`make_obu`, `ObuType`).

Bit counts produced by `header_bit_counts()`:

| Size | seq syntax | frame syntax (no is_first) | tile group |
|---|---:|---:|---:|
| 64x64 | 81 | 23 | 24 |
| 640x360 | 88 | 25 | 32 |

(M0 64x64 is 83 sequence bits and 22 frame bits: 10-bit `uvlc(0)` is two bits
shorter than `uvlc(1)`, and `base_q_idx` gains one bit.)

## Geometry limit (explicit)

The fixed tree codes `PARTITION_NONE` for a partial superblock only when the
remaining size is **more than 8 MI (32 luma pixels)** in each direction.
Smaller remainders force a normative *implied* partition, so M10 rejects the
geometry instead of emitting a desynchronized stream:

* supported: 64x64, 100x100, 640x360, 1920x1080, 3840x2160 (and any size whose
  `MiCols % 16` / `MiRows % 16` is 0 or > 8);
* rejected with an explicit error: 641x361, 129x100, 650x100, etc.

Supporting those needs the M0-style implied tree (32x16/32x8/16x32 leaves).
That is integration work, not a bit-depth issue.

## API

```c++
namespace av2enc::m10 {

struct M10Options { std::uint16_t base_q_idx = 90; bool disable_cdf_update = true; };

struct M10Block {              // one decode-order leaf
  std::uint16_t mi_row, mi_col;
  std::uint8_t luma_size;       // 32 or 64
  std::uint8_t tx_size[3];      // Y/U/V AV2 TX_SIZE
  std::uint32_t eob[3];
  std::uint32_t levels_offset;  // into result.levels (scan order, Y/U/V)
};

std::vector<M10BlockDesc> make_block_plan(uint32_t w, uint32_t h);
bool m10_geometry_supported(uint32_t w, uint32_t h, std::string* reason);
bool encode_m10_cpu(const fmt::Picture&, const M10Options&, M10Result&,
                    std::string& error);
bool assemble_m10_stream(...);   // shared entropy/OBU writer
bool convert_8bit_420_to_picture(...);
}
```

`M10Result` carries `frame` (TD + SEQ + CLK), `tile_bytes`, the block list, the
flat scan-order levels, and three `M10Plane` reconstructions.  `M10Plane`
stores 64-aligned `uint16_t` samples plus coded/visible extents; the visible
rectangle is what the gate compares.

GPU (`src/cuda/format/m10_gpu.{h,cu}`):

```c++
bool av2enc::cuda::m10::encode_m10_gpu(
    const av2enc::fmt::Picture&, const av2enc::m10::M10Options&,
    av2enc::m10::M10Result&, std::string* error);
```

It runs DC prediction, `hbd_tx_batch(..., bit_depth=10)`, and reconstruction on
the device and shares the CPU entropy writer.  It is a raster/one-block-at-a-time
correctness path, not the product scheduler; it uses `cudaMalloc` and per-block
copies, which the arena/wavefront integration must replace.

## Build and test

```sh
# CPU test + encoder tool (+ CUDA test if nvcc is present)
./src/encoder/m10/build_and_test.sh

# CUDA-only CPU-vs-GPU bit-exact test on the sm_86 box
./src/cuda/format/build_m10_gpu_test.sh
```

The CPU test (`m10_test.cpp`) covers the 10-bit header bits, rejected 8/12-bit
and q>90 configs, the geometry guard, the block order, an end-to-end encode, the
level range, levels above 255 (escape path), reconstruction range and OBU order.

The GPU test compares every EOB, every scan-order level, every reconstructed
coded sample and the final OBU bytes against the CPU reference for 64x64,
128x64, 100x100 (partial edges) and 640x360.

## `dav2d` gate

```sh
python3 src/encoder/m10/gen_y4m_10bit.py 64 64 /tmp/g64.y4m
./src/encoder/m10/build_and_test.sh
B=${TMPDIR:-/tmp}/av2enc_m10_build
$B/m10_encode /tmp/g64.y4m /tmp/g64.ivf /tmp/g64.recon.y4m
EXPECTED_IVF_MD5=1b5018610a86b8d79aba161e95770ae6 \
  ./src/encoder/m10/gate_10bit_still.sh /tmp/g64.ivf /tmp/g64.y4m /tmp/g64.recon.y4m
```

Pinned results (deterministic generator, `dav2d f9dafbc`, `base_q_idx = 90`):

| Input | Blocks | Tile bytes | IVF MD5 | PSNR-Y | Recon |
|---|---:|---:|---|---:|---|
| generator 64x64 | 4 | 2,713 | `1b5018610a86b8d79aba161e95770ae6` | 52.45 dB | 0 LSB |
| generator 640x360 | 210 | 196,498 | `d5cd53b7e6f18d181b83f11a08d45142` | 31.04 dB | 0 LSB |
| `testdata/m0_64x64_1f.y4m` (8-bit -> 10-bit) | 4 | 2,429 | — | 52.20 dB | 0 LSB |
| shine 640x360, 10-bit | 210 | 42,170 | — | 57.16 dB | 0 LSB |
| shine 1920x1080, 10-bit | 1,950 | 213,338 | — | — | 0 LSB |
| shine 3840x2160, 10-bit | 7,980 | 687,228 | — | — | 0 LSB |

Gate output shape:

```text
decoder: f9dafbc
format:  64x64 420p10, 1 frame, 12288 bytes
ivf:     64x64, 1 frame
recon:   0-LSB mismatch (decoded plane == encoder recon, all planes)
OK: 10-bit 4:2:0 M10 still passed the dav2d gate
```

`M10_ZERO_RESIDUAL=1` is a debug hook that codes an all-skip frame (DC
prediction as reconstruction); it is not a production mode.

## Exact changes the M1 core must make for 10-bit

M10 does not modify any M1 file.  To move the M1 core to 10-bit, apply these
changes in this order; the first stream should be 64x64, QP 90, DC/CfL chroma,
deblock disabled.

1. **Header — `src/encoder/m1/m1_headers.{h,cpp}`**
   * widen `M1Config::base_q_idx` to `uint16_t` and add
     `fmt::BitDepth depth` (or `bit_depth_idc`);
   * validate `1 <= base_q_idx <= fmt::max_quantizer(depth)` (303 at 10-bit)
     and reject 12-bit (`fmt::bit_depth_idc(k12) == -1`);
   * replace `put_uvlc(writer, 1)` with
     `put_uvlc(writer, fmt::bit_depth_idc(depth))` (10-bit = 0);
   * replace `put_bits(base_q_idx, 8)` with
     `put_bits(base_q_idx, fmt::base_q_idx_bits(depth))` (9 bits);
   * keep `single_picture_header_flag`, `chroma_format_idc = 0` and the tool
     disables unchanged.

2. **Options/CLI — `m1_still.h`, `src/main.cpp` (not owned here)**
   * widen `M1Options::base_q_idx` to `uint16_t`, validate against MaxQ;
   * add `--bit-depth 10`; for an 8-bit 4:2:0 input convert with
     `fmt::convert_420_8_to_picture(..., BitDepth::k10, ...)`;
   * keep `base_q_idx = 90` until CDF slices 1–3 exist.

3. **Samples/reconstruction — `m1_still.cpp`, `src/cuda/m1/{recon,cfl,chroma}`,
   `src/cuda/m1/m1_gpu.h`**
   * change sample containers/POD `uint8_t` -> `uint16_t`;
     `Predictor`, `source_sample`, `TxPlan::pred/recon`, `Leaf`,
     `M1Result::recon_*`, GPU PODs;
   * `clip1(v)` -> `fmt::clip1_sample(v, depth)` (1023);
   * DC neutral value `128` -> `1 << (BitDepth - 1)` = 512 in the encoder
     predictor, intra reference fill and CfL luma reference; keep the initial
     reconstructed plane at 0;
   * 10-bit TX must use `hbd::dequantize`/`hbd::inverse_block` or
     `cuda::hbd::hbd_tx_batch(..., bit_depth=10)`.  Do **not** call
     `txv::dequantize`/`txv::inverse_block` for 10-bit: they clip at 8-bit
     bounds;
   * residuals stay `int16_t`, dequant `int32_t`; `Transform_Shift`,
     `dq_denom`, scan and EOB are unchanged;
   * keep `-ffp-contract=off` (CPU) / `-fmad=false` (CUDA).

4. **Entropy — `m1_syntax.cpp` / `src/entropy/av2/`**
   * no CDF alphabet change; select the coefficient CDF slice from the **raw**
     `base_q_idx` (<=90 -> 0), never from bit depth;
   * the header now writes 9 bits; the coder sees the same integer;
   * add a test that round-trips levels above 255 through the escape path
     (M10's 10-bit pictures produce `max|level|` 700–2500);
   * keep frozen CDFs by default so pinned bytes do not move.

5. **Filters**
   * first 10-bit gate: `--no-deblock` (`enable_deblock = false`);
   * when enabling deblock, subtract `24 * (BitDepth - 8) = 48` from the q
     index and use `(1 << (12 - BitDepth))` / `>> (13 - BitDepth)` for `side`
     (§7.17); CDEF/CCSO/LR are separate work.

6. **Do not emit 12-bit**
   * `bit_depth_idc > 1` is reserved; reject at the config boundary.

## Files

| File | Purpose |
|---|---|
| `m10_headers.{h,cpp}` | 10-bit sequence/frame/tile-group + OBU writer |
| `m10_entropy.cpp` | partition/mode/residual syntax, §8.2 entropy, OBU assembly |
| `m10_still.{h,cpp}` | CPU block encode (DC + hbd transform + recon) and plan |
| `m10_y4m.{h,cpp}` | 8-bit C420 / 10-bit C420p10 input, C420p10 recon output |
| `m10_encode.cpp` | component CLI (IVF + recon Y4M) |
| `m10_test.cpp` | CPU header/geometry/encode tests |
| `build_and_test.sh` | CPU build/test, optional CUDA test |
| `gate_10bit_still.sh` | `dav2d` + C420p10 + 0-LSB recon gate |
| `gen_y4m_10bit.py` | deterministic 10-bit test source |
| `../cuda/format/m10_gpu.{h,cu}` | GPU DC/TX/recon block stage |
| `../cuda/format/m10_gpu_test.cu` | CPU-vs-GPU bit-exact test |
| `../cuda/format/build_m10_gpu_test.sh` | CUDA test build |
