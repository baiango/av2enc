# 10-bit encoder component (`src/encoder/hbd/`)

Status (2026-11-12): **standalone 10-bit 4:2:0 still path that reuses the
verified M10 sample/entropy core.**  `dav2d --strict 1` decodes 64x64, 640x360
and 3840x2160 streams with **every visible sample equal to the encoder
reconstruction (0 LSB)**.  The M1/M2 cores are still 8-bit; the exact call sites
to flip them, and the risks, are in [`INTEGRATION.md`](INTEGRATION.md).

AV2 v1.0.0 signals exactly two depths: `bit_depth_idc = 0` is 10-bit and
`bit_depth_idc = 1` is 8-bit (§6.4.1 Table 6.3).  Values above 1 are reserved,
so this component never emits 12-bit.  12-bit remains internal-only in
`src/encoder/format/`.

## What this owns

| File | Purpose |
|---|---|
| `hbd_depth.{h,cpp}` | depth-dependent sample ranges: Clip1, DC bias, intra edge defaults, dequant/inverse clips, residual range, deblock q offset/side/qThr, CfL luma average/scaled luma, `qlookup`, MaxQ, §8.3.1 CDF slice |
| `hbd_headers.{h,cpp}` | 10-bit sequence/frame/tile-group writer, `bit_depth_idc = 0`, `base_q_idx` f(9), qindex up to 303 |
| `hbd_reader.{h,cpp}` | Annex B OBU + sequence/frame reader used for bit goldens and round-trips |
| `hbd_still.{h,cpp}` | 10-bit still encoder: qlookup-aware quant/dequant + M10 block plan/DC/entropy |
| `hbd_encode.cpp` | component CLI (`hbd_encode in.y4m out.ivf [recon.y4m] [--q N]`) |
| `hbd_test.cpp` | 193 checks: ranges, qlookup/qctx, deblock/CfL, header bits, reader, encode goldens |
| `build_and_test.sh` | standalone build/test (no CMake changes) |
| `gate_hbd_still.sh` | `dav2d --strict 1` + `tools/validate_dav2d.sh` gate for 64x64/640x360/4K |
| `INTEGRATION.md` | exact M1/M2 call sites and risks |

The component does **not** edit `src/encoder/m10/`, the M1/M2 core,
`src/main.cpp` or `CMakeLists.txt`.

## Reuse boundary

```text
hbd_still  ── block plan / geometry guard / DC predictor shape ──► src/encoder/m10/
           ── forward transform + quantisation ──────────────────► src/encoder/tx_variants/
           ── depth-aware dequant / inverse / recon oracle ──────► src/cuda/m1/hbd/hbd_cpu.*
           ── tile entropy + OBU assembly (payload bytes only) ──► src/encoder/m10/high_bitdepth_entropy
hbd_headers ─ OBU framing helpers ──────────────────────────────► src/bitstream/basic_headers.*
```

`av2enc::high_bitdepth::assemble_high_bitdepth_stream()` is called with `base_q_idx = 90` only to obtain the
tile payload; its header is discarded and replaced by `hbd::assemble_still_picture()`
with the real qindex.  The payload does not depend on the header qindex: the
M10 entropy writer's coefficient CDFs are frozen at slice 0, and the encode path
is restricted to `q <= 90`.

## 10-bit fields

| Field | 8-bit | **10-bit** | Clause |
|---|---:|---:|---|
| `bit_depth_idc` | 1 | **0** | §6.4.1 Table 6.3 |
| `MaxQ` | 255 | **303** | §6.4.1 Table 6.3 |
| `base_q_idx` width | f(8) | **f(9)** | §5.18.6.1 |
| `Clip1` high | 255 | **1023** | §4.8 |
| DC neutral value | 128 | **512** | §7.13.2.10 |
| `AboveRow` unavailable fill | 127 | **511** | §7.13.2.1 |
| `LeftCol` unavailable fill | 129 | **513** | §7.13.2.1 |
| corner fill (`AboveRow[-1]`) | 128 | **512** | §7.13.2.1 |
| dequant clip | ±(1<<15) | **±(1<<17)** | §7.14.4 step 8 |
| inverse row clip | ±(1<<15) | **±(1<<17)** | §7.15.2.1 |
| inverse column clip | ±(1<<8) | **±(1<<10)** | §7.15.2.1 |
| residual conformance range | ±(1<<8) | **±(1<<10)** | §7.14.3 |
| deblock q index offset | 0 | **48** | §7.17.5 |
| deblock `side` | `(t+16)>>5` | **`(t+4)>>3`** | §7.17.5 |
| CfL lumaAvg bias | 2048 | **4096** | §7.13.5 |
| CfL lumaAvg clamp | 2047 | **8191** | §7.13.5 |
| coefficient CDF slice | raw `base_q_idx` | **same, never depth-derived** | §8.3.1 |

Pinned header bit counts (`header_bit_counts()`):

| Size | sequence | frame (no is_first) | tile group |
|---|---:|---:|---:|
| 64x64 | 81 | 23 | 24 |
| 640x360 | 88 | 25 | 32 |
| 3840x2160 | 93 | 25 | 32 |

Pinned header bytes (tests):

```text
seq 64x64   0c04827d57ffc00010006e0020
seq 640x360 0d04827e627fb380002000dc0040
seq 3840x2160 0d04827eefbfe1bc00010006e002
tile group 64x64 q90    e65a00
tile group 640x360 q90  e6168000
```

## API

```c++
namespace av2enc::hbd {

struct HbdOptions {
  std::uint16_t base_q_idx = 90;   // encode: 1..90 (see qindex seam)
  bool disable_cdf_update = true;
};

struct HbdResult {
  std::uint32_t tile_bytes, frame_bytes;
  int quant_step;
  std::vector<std::uint8_t> frame;      // TD + SEQ + CLK
  std::vector<av2enc::high_bitdepth::HighBitDepthBlock> blocks;
  std::vector<std::int32_t> levels;
  av2enc::high_bitdepth::HighBitDepthPlane recon[3];
};

bool qindex_encodable(std::uint16_t base_q_idx, std::string* reason);
bool encode_still(const fmt::Picture&, const HbdOptions&, HbdResult&,
                  std::string& error);
}
```

Header writer/reader:

```c++
namespace av2enc::hbd {

struct HbdConfig {
  std::uint16_t max_frame_width, max_frame_height;
  std::uint8_t seq_level_idx = 7;
  fmt::BitDepth depth = fmt::BitDepth::k10;  // 10-bit only
  std::uint16_t base_q_idx = 90;             // 1..303
  bool disable_cdf_update = true;
  bool enable_intra_edge_filter = false;
  bool enable_cfl_intra = false;
  bool enable_chroma_dctonly = false;
  bool tx_mode_select = false;
  int reduced_tx_set = 0;
  bool apply_deblock_y_vert = false, apply_deblock_y_horz = false;
  bool apply_deblock_u = false, apply_deblock_v = false;
};

bool validate_config(const HbdConfig&, std::string* error);
std::vector<std::uint8_t> make_sequence_header_obu(const HbdConfig&);
std::vector<std::uint8_t> assemble_still_picture(
    const HbdConfig&, std::span<const std::uint8_t> tile_payload);
HeaderBitCounts header_bit_counts(const HbdConfig&);

bool parse_stream(std::span<const std::uint8_t>, HbdParsedStream*,
                  std::string* error);
}
```

Depth helpers (`namespace av2enc::hbd::depth`):

```c++
struct SampleRanges { /* 1023, 512, 511, 513, 512, +-1<<17, +-1024, 48, 4096, 8191 */ };
SampleRanges sample_ranges();
int clip1(std::int64_t);
int clip_dequant(std::int64_t);
int clip_inverse_row(std::int64_t);
int clip_inverse_col(std::int64_t);
bool residual_representable(std::int64_t);

int max_q(int bit_depth);          // 255 / 303 / -1 (12-bit reserved)
int qlookup(int qindex);           // §7.14.2 up to q=303
int get_q(int qindex, int delta);
int qctx_slice(int base_q_idx);    // §8.3.1
int base_q_idx_bits(int bit_depth);

DeblockStrength deblock_strength(int qindex, int bit_depth,
                                 int plane_delta, int df_delta_q);

int cfl_luma_average(std::int64_t luma_sum, int luma_count);
int cfl_scaled_luma(int alpha_q8, int luma, int luma_avg);
int cfl_apply(int dc, int alpha_q8, int luma, int luma_avg);
```

## qindex / qlookup seam (read this before claiming 1..303 encoding)

* The **header writer and reader** handle the full 10-bit range
  `1 <= base_q_idx <= 303`, and `depth::qlookup` / `qctx_slice` are tested at
  q = 1, 64, 90, 140, 191, 303.
* The **encode path** supports `1 <= base_q_idx <= 90`.  `base_q_idx > 90`
  selects coefficient-CDF slice 1..3 (§8.3.1), but the reused M10 entropy
  writer has only the frozen slice-0 tables (`src/encoder/basic_cdf_tables.h`).
  Emitting q > 90 through it would desynchronise the decoder's coefficient
  CDFs.  `qindex_encodable()` rejects those values with a precise message.
* Closing the seam is a contained job: give `high_bitdepth_entropy`'s `init_cdfs()` a
  qctx slice argument and install `av2enc::still::qctx::tables_for_base_q_idx(q)` (the
  four slices already exist in `src/encoder/m1/qctx/`).  It is not done here
  because `src/encoder/m10/` is owned by another agent.
* At `q == 90` the output is byte-identical to M10.  The 64x64 test picture is
  the M10 generator source, so the gate also pins the M10 IVF MD5
  `1b5018610a86b8d79aba161e95770ae6`.

## Build, test, gate

```sh
./src/encoder/hbd/build_and_test.sh          # 193 checks + tool
WERROR=1 ./src/encoder/hbd/build_and_test.sh
SANITIZE=1 ./src/encoder/hbd/build_and_test.sh
./src/encoder/hbd/gate_hbd_still.sh          # 64x64, 640x360, 4K
./src/encoder/hbd/gate_hbd_still.sh 64x64    # one case
GATE=1 ./src/encoder/hbd/build_and_test.sh   # test + gate
```

Gate output (macOS, `dav2d f9dafbc`, `base_q_idx = 90`, `--strict 1`; the
4K source is NumPy PCG64 with seed `0x9E3779B9`):

| Input | Blocks | Tile bytes | IVF MD5 | PSNR-Y | Recon |
|---|---:|---:|---|---:|---|
| deterministic 64x64 | 4 | 2,713 | `1b5018610a86b8d79aba161e95770ae6` (pinned) | 52.45 dB | 0 LSB |
| deterministic 640x360 | 210 | 196,498 | `d5cd53b7e6f18d181b83f11a08d45142` | 31.04 dB | 0 LSB |
| NumPy 10-bit 4K | 7,980 | 7,435,161 | `e531e784adc6a4f837afe17e72ba0d23` | 37.85 dB | 0 LSB |
| 64x64, q=64 (qlookup 248) | 4 | 3,647 | `1d82350c3315d159fa89005bdc203b9a` | — | 0 LSB |

```text
OK: dav2d decoded .../4k.ivf
recon:   3840x2160 420p10, 0-LSB mismatch (all planes)
OK: 10-bit hbd stills passed dav2d --strict 1 with 0-LSB reconstruction
```

The default gate includes a 64x64 q=64 case: the reconstructed `C420p10` Y4M
equals the `dav2d --strict 1` output byte-for-byte (0 LSB) with
`qlookup(64) = 248`, confirming forward quant and inverse dequant use the same
step.  `--q 1` and `--q 64` were also checked by hand with 0 LSB.

## Sample ranges that the 8-bit core must replace

| Area | 8-bit value in tree | 10-bit use |
|---|---|---|
| intra prediction | `intra_modes` already takes `bit_depth`; fillers are `mid±1` / `mid` | set `bit_depth = 10`; only the plane pointer is `uint8_t*` |
| DC predictor in M1/M2 | literal neutral `128` / `1 << (bd-1)` | 512 |
| M1/M2 recon | `Clip1(...,255)`, `uint8_t` planes | `depth::clip1`, `uint16_t` planes |
| transform | `txv::dequantize` / `txv::inverse_block` (8-bit clips) | `hbd::dequantize` / `hbd::inverse_block(...10)` |
| CfL | `Block::luma/chroma` are `uint8_t*`; clip1 already depth-aware | `uint16_t*`, `bit_depth=10` |
| deblock | `deblock.cpp` rejects `bit_depth != 8`; `kMaxQ=255` | `uint16_t` planes, MaxQ 303, q offset 48 |
| CDEF | `coeffShift = bit_depth - 8` (0 at 8-bit) | `coeffShift = 2` at 10-bit |
| CCSO / restoration | `uint8_t` cores | not 10-bit ready; keep flags off |

## Non-goals

* No 12-bit stream.  `bit_depth_idc > 1` is reserved.
* No CfL/deblock/CDEF/CCSO/GDF/restoration in the emitted still; only their
  depth-dependent *ranges* are implemented and tested.
* No GPU path here.  The verified GPU block stage remains
  `src/cuda/format/high_bitdepth_gpu.*`; a 10-bit GPU port must match this CPU oracle.
* No CMake/product CLI changes.
