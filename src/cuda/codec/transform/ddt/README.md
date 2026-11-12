# CUDA inter DDT (`src/cuda/ddt/`)

Bit-exact GPU implementation of the M2 inter DDT (Data Dependent Transform)
transform pair:

- the decoder-normative inverse of AV2 v1.0.0 §7.15.2.1 + §7.15.4.1 with the
  §9.6 `Ddtx_Kernel8` / `Ddtx_Kernel16` tables, and
- the encoder forward from `src/encoder/m2/ddt/ddt_transform.cpp`.

The CPU reference is unchanged and is the oracle. This component is standalone:
it is not wired into the root CMake build or the encoder path, and makes no
bitstream claim.

## Files

| File | Contents |
|---|---|
| `ddt_device.cuh` | shared device body: §9.6 tables, `get_transform_1d_type()`, 1D inverse/forward, 2D row/column element functions, clips, rounding, level mapping |
| `ddt_gpu.h` | arena-backed batch API, `required_device_bytes()`, job/status types |
| `ddt_gpu.cu` | kernels + host entry points (all device memory from `Arena`) |
| `ddt_test_pairs.h` | pinned §5.20.8.3 / `Tx_Type_In_Set_Inter` masks shared by the tests (400 pairs, 100 legal DDT pairs) |
| `ddt_device_emulation_test.cpp` | no-GPU parity test of `ddt_device.cuh` against the CPU reference |
| `ddt_gpu_test.cu` | real-GPU bit-exact test, 8/10-bit, full batch + arena tests |
| `build_and_test.sh` | host emulation parity (no nvcc, no GPU) + arena guard |
| `build_gpu_test.sh` | `nvcc -arch=sm_86` real-GPU test |

## Implemented rules

- **DDT derivation.** `enable_inter_ddt && !use_intrabc && is_inter` and a
  signalled ADST/FDST at adjusted length `!= 4` becomes DDTX (ADST) or FDDT
  (FDST). The adjusted length is the post-§7.15.4 size, so 64-coded dimensions
  use length 32.
- **8/16 1D-length rule.** Only length 8 and 16 select `Ddtx_Kernel8` /
  `Ddtx_Kernel16`. Length 4 falls back to `Dct_Kernel4` (the predicate excludes
  it, but the kernel selection matches the CPU fallback anyway).
- **32 alias-to-DCT.** `get_transform_1d_type()` still derives DDTX/FDDT at
  length 32, but §7.15.2.1's `sz == 32` branch uses `Dct_Kernel32` for every
  type, so the reconstruction is identical to the non-DDT ADST/FDST alias.
  `plan_has_ddt()` reports only the 8/16 cases, exactly like the CPU.
- **FDDT reversal.** FDDT is `Ddtx_KernelN` with the sample (second) index
  mirrored; the emulation test checks `FDDT[i] == DDTX[n-1-i]` for 8 and 16.
- **8/10-bit clipping.** Row clip is `±(1 << (BitDepth + 7))`, column clip is
  `±(1 << BitDepth) - 1`, applied after `Round2` and before any narrowing cast.
  Both bit depths are tested against the CPU with normal and saturating input.
- **Integer rounding.** `Round2(x, n) = (x + 2^(n-1)) >> n` with int64
  accumulation, the §7.15.4.1 `Round2(Dequant * 2896, 12)` odd-log2 pre-scale,
  and the encoder's symmetric `round_div` / Q8 row-to-column hand-off. The
  encoder level mapping is `round(coeff * dq_denom / 66)`.

## API

```cpp
namespace av2enc::cuda::ddt {

struct DdtJob {
  txv::TxParams params;          // from txv::make_tx_params()
  std::int32_t in_offset;        // element offset in the input buffer
  std::int32_t out_offset;       // element offset in the output buffer
  std::uint8_t enable_inter_ddt, is_inter, use_intrabc, bit_depth;
};

struct BatchConfig {
  Direction direction;           // kForward | kInverse
  int max_blocks;
  bool with_levels;              // forward: also emit levels
  bool inverse_from_levels;      // inverse: input is levels, not dequant
};

std::uint64_t required_device_bytes(const BatchConfig&);

BatchStatus forward_batch(Arena&, const DdtJob* jobs, int job_count,
                          const std::int16_t* host_residual_adj,
                          std::int32_t* host_coeff,
                          std::int32_t* host_levels = nullptr,
                          cudaStream_t = nullptr,
                          std::uint64_t* required_bytes = nullptr,
                          std::string* error = nullptr);

BatchStatus inverse_batch(Arena&, const DdtJob* jobs, int job_count,
                          const std::int32_t* host_input,
                          std::int16_t* host_residual_coded,
                          bool input_is_levels = false,
                          cudaStream_t = nullptr,
                          std::uint64_t* required_bytes = nullptr,
                          std::string* error = nullptr);

// Zero-allocation, one-launch variants over caller-owned device buffers.
cudaError_t forward_batch_device(...);
cudaError_t inverse_batch_device(...);
}
```

- A batch is one `DdtJob` per transform block; `in_offset`/`out_offset` are
  element offsets into shared buffers, so mixed sizes are packed with host
  prefix sums (the same convention as `cuda/tx_variants/`).
- `forward_batch` writes `w*h` coefficients (and optionally levels) at
  `out_offset`; `inverse_batch` writes `coded_w*coded_h` samples at
  `out_offset`.
- The host-facing entry points upload, launch one kernel, download and rewind;
  they synchronize the stream. Use the `_device` launchers for a fully
  device-resident integration.

## Arena contract

Every device byte comes from one `av2enc::cuda::Arena`:

- `required_device_bytes(config)` includes the 256-byte alignment the bump
  allocator applies to every array.
- Each call takes `arena.mark()`, allocates with `allocate_checked()`, and
  rewinds on every exit path (RAII).
- A short arena returns `BatchStatus::kArenaTooSmall` and writes the exact
  requirement to `*required_bytes`; the bump pointer is left unchanged.
- No `cudaMalloc`/`cudaFree` anywhere in the module (`tools/check_arena.sh`
  passes).

Worst-case sizes (every block 32x32 adjusted / 64x64 coded), `sizeof(DdtJob)`
is 72 bytes on this ABI:

| batch | forward | forward + levels | inverse |
|---:|---:|---:|---:|
| 1 | 6,400 | 10,496 | 12,544 |
| 100 | 621,824 | 1,031,424 | 1,236,224 |
| 400 | 2,486,528 | 4,124,928 | 4,944,128 |

The GPU test prints the split (jobs / input / output) for the 100-pair batch at
run time.

## Device-symbol audit

Real nvcc rejects host-only symbols referenced from kernels; host emulation
cannot catch that. This module was audited for the three failure classes seen
in other components:

1. **Every function called from a kernel is `__device__ __forceinline__`.**
   The shared header has no unannotated free functions. Host-only helpers
   (`valid_job`, `compute_spans`, `allocate`, `check_cuda`, logging) live only
   in `ddt_gpu.cu` and are never referenced from device code.
2. **No namespace-scope host `constexpr`/`const` variable is used in device
   code.** Type and size constants are unscoped enumerators (`enum : int`),
   which have no host storage. The two §9.6 tables are
   `static __device__ const`; the standard §9.6 kernels come from
   `encoder/tx_variants/tx_tables_dev.cuh` (`__device__ const`). The only
   `constexpr` left in device code is a function-local `kLimit`.
3. **No `std::` containers or algorithms in device code**; only
   `<cstdint>` typedefs and the POD `TxParams`.
4. **No mutable static/global state** shared between host and device; the only
   file-scope objects are immutable device tables with `static` linkage.

`ddt_gpu_test.cu` intentionally does not include `ddt_device.cuh`: the device
body is compiled once in `ddt_gpu.cu`, so the non-`static` `__device__` tables
from `tx_tables_dev.cuh` cannot collide at device link time.

## Tests

### Host emulation (no GPU, no nvcc)

```sh
cd src/cuda/ddt
./build_and_test.sh
```

Observed on the development Mac (Apple clang 21 and g++ 13, ASan+UBSan also
clean):

```text
ddt device emulation: 828114 checks, 2322688 element comparisons, 0 failures
RESULT: PASS
```

Coverage: 320 table entries; 128 `get_transform_1d_type()` flag/length cases;
kernel selection for types 0..5 at lengths 4/8/16/32 (including the 32 alias);
1D inverse/forward across types, shifts, 8/10-bit and column/row clips; the
FDDT reversal; 2D inverse and forward for all 25 sizes x 16 types x 8/10-bit;
the 100 legal DDT pairs (count asserted); `level_from_coeff`; and §7.14.4
dequant rounding.

### Real GPU

```sh
cd src/cuda/ddt
./build_gpu_test.sh
```

The test compares every size/type/depth against the CPU reference, checks the
mixed-size batch offsets, the inverse-from-levels path, the 32 alias
behaviour, and the arena contract (exact size, two launches with no growth,
one-byte-short `kArenaTooSmall`, invalid-argument rejection).

Remote command (fill in the Vast.ai host/port; see `docs/remote-env.md`):

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc/src/cuda/ddt
./build_gpu_test.sh
```

Local rsync-free sync first, if needed:

```sh
AV2ENC_HOST=<user>@<host> AV2ENC_PORT=<port> tools/sync_remote.sh
```

## Integration notes

- Batch mixed-size candidates in one launch (`_multi`-style job array). One
  launch per transform is dominated by dispatch.
- `forward_batch`/`inverse_batch` exist for tests and host-side decision loops;
  the product path should use `forward_batch_device`/`inverse_batch_device`
  with arena-resident buffers.
- DDT changes only the ADST/FDST kernels at adjusted length 8/16. `DCT`/`IDT`
  components are unaffected, and length 32 is DCT-aliased.
- A stream-level claim still requires wiring this into the M2 path and passing
  `tools/validate_dav2d.sh`; nothing here claims that.
