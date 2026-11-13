# `src/cuda/intra_full/` — CUDA for the newly covered intra features

GPU counterparts for the intra rules that landed in
`src/encoder/m1/missing_bits/` and `src/encoder/transform/idtx_intra/`:

| # | Rule | CPU reference | What the GPU adds |
|---|---|---|---|
| 1 | §7.13.2.11 DC intra **subsampled** (step 2 when an extent > 32), 8/10/12-bit | `m1/missing_bits/dc_intra_subsampled.*` | batch kernel + device process + trace |
| 2 | §7.13.2.14 filter corner (5/6/5, `Round2(.,4)`, guard `pAngle in (90,180) && w+h >= 24`) | `m1/missing_bits/intra_filter_corner.*` | device value/guard/apply + batch apply kernel |
| 3 | All intra identity transform types: IDTX, V/H_DCT, V/H_ADST, V/H_FLIPADST | `tx_variants/` (`forward_block` / `inverse_block`) | new separable kernels for the seven TX_TYPEs, bit-exact including §7.15.4 duplication |
| 4 | §5.20.7.29 chroma PlaneTxType derivation when `enable_chroma_dctonly == 0` | `transform/idtx_intra/` (`derive_chroma_type`) | device derivation over wide-angle-mapped UVMode + set filtering |
| 5 | §5.20.7.27 / §8.3 `dc_sign` + `dc_sign_horz_vert` element selection and the neighbour-derived `dc_sign` context | `m1/missing_bits/dc_sign_horz_vert.*` | batch selection kernel + batch context kernel |

Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`).

```
intra_full.h                 public API, Config, arena sizing, Workspace
intra_full_device.cuh        shared device bodies (the single implementation)
intra_full.cu                kernels + arena-backed launchers
intra_full_emulation_test.cpp host emulation: same .cuh, CUDA keywords stubbed
intra_full_test.cu           real-GPU parity test
build_and_test.sh            emulation build/run on this host (no nvcc)
build_gpu_test.sh            sm_86 CUDA build/run
```

Nothing outside this directory is edited; the component is not wired into
`src/main.cpp`, the root `CMakeLists.txt`, or an encoder call site.

## API

```cpp
namespace av2enc::cuda::intra_full {

struct Config {                 // bounds the largest single launch
  int max_dc_jobs = 64;
  int max_corner_jobs = 256;
  int max_chroma_jobs = 256;
  int max_tx_blocks = 64;
  int max_sign_jobs = 256;
};

std::uint64_t required_device_bytes(const Config&);   // exact init requirement
std::uint64_t persistent_device_bytes(const Config&); // init growth
std::uint64_t scratch_device_bytes(const Config&);    // largest launch scratch

class Workspace {
  Status init(Arena&, const Config&,
              std::string* error = nullptr,
              std::uint64_t* required_out = nullptr);

  Status dc_subsampled_predict(const DcJob*, int count,
                               DcResult* host_results = nullptr,
                               cudaStream_t = nullptr);
  Status filter_corner(const CornerJob*, int count,
                       CornerResult* host_results = nullptr,
                       cudaStream_t = nullptr);
  Status derive_chroma_type(const ChromaQuery*, int count,
                            ChromaResult* host_results,
                            cudaStream_t = nullptr);
  Status select_sign_elements(const SignQuery*, int count,
                              SignResult* host_results,
                              cudaStream_t = nullptr);
  Status dc_sign_contexts(const SignContextJob*, int count, int* host_ctx,
                          cudaStream_t = nullptr);
  Status identity_forward(const TxJob*, int count,
                          const std::int16_t* residual, double* coeff,
                          cudaStream_t = nullptr);
  Status identity_inverse(const TxJob*, int count,
                          const std::int32_t* dequant, std::int16_t* coded,
                          cudaStream_t = nullptr);
};

}  // namespace
```

Semantics:

* `DcJob` carries `AboveRow[0]` / `LeftCol[0]` device pointers, `w,h`,
  `bit_depth`, availability, and a strided 16-bit destination.  The output is
  `uint16_t` so 10/12-bit predictions are not truncated.
* `CornerJob.above` / `.left` point at `AboveRow[0]` / `LeftCol[0]`; the
  `[-1]` slots must be addressable.  `CornerResult.applied` is the return value
  of the CPU `filter_corner_if_applicable()`.
* `TxJob` is a fully derived `av2enc::txv::TxParams` plus element offsets, the
  same shape as `cuda::TxBlockJob`.  Validation accepts only the seven identity
  TX_TYPEs and checks that `row_type`/`col_type` match
  `Transform_1d_Type[tx_type]`:

  | TX_TYPE | row (width) | col (height) |
  |---|---|---|
  | `IDTX` (9) | IDT | IDT |
  | `V_DCT` (10) | IDT | DCT |
  | `H_DCT` (11) | DCT | IDT |
  | `V_ADST` (12) | IDT | ADST |
  | `H_ADST` (13) | ADST | IDT |
  | `V_FLIPADST` (14) | IDT | FDST |
  | `H_FLIPADST` (15) | FDST | IDT |

  The device kernel is generic over the four 1D types, so it runs the existing
  §7.15.2 kernels directly: IDT is the identity path, DCT/ADST/FDST select
  `Dct_Kernel*/Adst_Kernel*/Fdst_Kernel*` from the generated
  `encoder/tx_variants/tx_tables_dev.cuh` tables.  Length 32 uses
  `Dct_Kernel32` for every type, matching §7.15.2.1.
* `ChromaQuery` is the §5.20.7.29 tuple (`uv_mode`, `uv_angle_delta`, coded
  size, plane, `enable_chroma_dctonly`).  `ChromaResult` returns the derived
  TX_TYPE and the §6.19.8.2 intra set value.
* `SignQuery` is one §5.20.7.27 coefficient position (`plane`, `row`, `col`,
  `tx_class`) plus optional `AboveDcContext`/`LeftDcContext` device pointers for
  the §8.3 `dc_sign` context.  `SignResult.element` is the selected sign
  element; `ctx` is the neighbour context for `kDcSign` and 0 otherwise.
  `SignContextJob` computes just the §8.3 context (both arrays may be null).
  `dc_sign_horz_vert` has no neighbour context, so its ctx is always 0.
* Batch sizes above the configured maximum, null pointers, or out-of-range
  geometry return `Status::kInvalidArgument` before any launch.

The batch entry points are synchronous: they stage host descriptors in arena
scratch, enqueue the kernel, copy requested results back, wait on the stream,
and `rewind()` the scratch.  That keeps `arena.used()` flat across repeated
launches and makes the rewind safe; per-frame scheduling belongs to the
wavefront/scheduler layer.

## Memory contract (no direct device allocation)

Every device byte comes from the caller's single `av2enc::cuda::Arena`:

* `Workspace::init()` allocates the persistent result arrays exactly once
  (`DcResult[max_dc_jobs]`, `CornerResult[max_corner_jobs]`,
  `ChromaResult[max_chroma_jobs]`) with `Arena::allocate_array`.
* Each launch records `arena.mark()`, allocates one descriptor staging array
  with `Arena::allocate_array`, and `rewind(mark)` after the stream completes.
* `required_device_bytes(config) = persistent + max(scratch)` and is exact:
  every allocation matches `Arena`'s 256-byte rounding.
* A short arena returns `Status::kArenaExhausted` (a distinct status, never a
  CUDA error) with the required byte count in `*required_out` / `error`; there
  is no `cudaMalloc` fallback and a failed init leaves `used()` untouched.

Sizing (`sizeof(DcJob)=56`, `CornerJob=32`, `ChromaQuery=24`, `TxJob=64`,
`SignQuery=56`, `SignResult=2`, `SignContextJob=40`):

| Config (`dc/corner/chroma/tx/sign`) | persistent | scratch | required |
|---|---:|---:|---:|
| defaults `64/256/256/64/256` | 6,912 B | 14,336 B | **21,248 B** |
| emulation test `8/512/512/3/256` | 9,984 B | 16,384 B | **26,368 B** |
| GPU test `4/256/256/3/256` | 5,888 B | 14,336 B | **20,224 B** |

Measured arena high-water marks:

* emulation test: peak `26,368 B` (persistent + one scratch cycle).
* GPU test: `required=20,224`, `capacity=20,008,704`, **peak `18,045,696 B`**
  (17.2 MiB).  Almost all of that is the test's own exhaustive 16.7M-entry
  corner-value buffer (16.0 MiB) plus the 8-bit guard buffer; the component's
  workspace is only 20 KiB.  The test prints the live values.

`grep -n "cudaMalloc\|cudaFree\|cudaMallocHost" src/cuda/intra_full/*.{h,cuh,cu,cpp}`
returns comments only.  The one `cudaMalloc` in `build_and_test.sh` is the
host-emulation stub for `Arena::init()` (backed by 4096-aligned `posix_memalign`
so alignment matches a device allocation); it is not compiled into the module.

## Bit-exactness contract

* DC subsampled: integer only; `resolve_divisor`/`Div_Lut`, `Round2`,
  `Clip1(1 << bit_depth)` transcribed from §7.13.2.11 / §7.13.3.22.
* Corner: `(LeftCol[0]*5 + AboveRow[-1]*6 + AboveRow[0]*5 + 8) >> 4` and the
  non-IBP guard `pAngle > 90 && pAngle < 180 && w+h >= 24`.
* Identity transforms: the device body is a transcription of
  `tx_variants.cpp`; binary64 accumulation uses `__dadd_rn`/`__dmul_rn` in the
  same loop order (`-fmad=false` also set), `TxParams.scale_w/scale_h/
  rect_prescale/quant_factor` are host-computed and never re-derived, and the
  inverse uses the normative `int64` accumulation, row clip `±(1<<15)`,
  column clip `[-256,255]`, `get_identity_scale()`, the odd-log2 `*2896`
  pre-scale, and §7.15.4 coded-size duplication.
* Chroma derivation: `Mode_To_Txfm` + wide-angle mapping (WAIP thresholds
  61/73/82/86) + `Tx_Type_In_Set_Intra[set]`, falling back to `DCT_DCT`
  exactly like `derive_chroma_type()`.

The emulation test cross-checks the device copies of
`Tx_Type_In_Set_Intra`, `Mode_To_Txfm`, `Mode_To_Angle`, the 25 TX sizes, and
the identity `Transform_1d_Type` mapping against the CPU tables/planner.

## Tests

### Emulation (no GPU, no nvcc) — `build_and_test.sh`

Compiles `intra_full_device.cuh` with CUDA keywords stubbed and one emulated
thread, against the real CPU components, plus the real `Arena` against a host
`cuda_runtime` stub.

```
intra_full_emulation_test: PASS (70926951 checks, 0 failures)
```

Identical on macOS/Apple clang 21 and on the Ubuntu 24.04 / GCC 13.3 box.

Coverage:

* **DC subsampled** — every `w,h` in `1..64`, all 4 availability combinations,
  8/10/12-bit edges: `51,916,800` output samples and all
  `step_w/step_h/sum/count/avg` trace fields.  This includes the step-2 region
  (`w>32` and/or `h>32`) and odd extents.
* **Corner** — guard over every `w,h in 1..64` × `pAngle 0..270` (1,110,016
  cases); the 5/6/5 value over all `16,777,216` 8-bit triples; the apply path
  (both corner slots and the `applied` flag) over sampled extents × 11 angles.
* **Identity transforms** — 7 TX_TYPEs × 25 sizes × 4 residual patterns;
  every forward coefficient compared bit-for-bit (`memcmp` on `double`) and
  every coded-size inverse sample.  `667,520` coefficient/residual checks.
* **Chroma derivation** — 2 planes × 2 `enable_chroma_dctonly` values ×
  14 UV modes × 7 angle deltas × 25 sizes = `9,800` cases, comparing TX_TYPE
  and set with `idtx::derive_chroma_type`.
* **dc_sign / dc_sign_horz_vert** — the full 3 planes × 3 TX classes × 8×8
  `(row,col)` grid = `576` element decisions against
  `missing_bits::select_sign_element`, including the DC-position precedence;
  `4,096` random §8.3 contexts against `missing_bits::dc_sign_context` with
  frame-edge `MiCols`/`MiRows` limits and null-array cases.
* **Arena** — sizing identity `required = persistent + scratch`, exact init
  growth, two mark/rewind scratch cycles with no net growth, stable peak, and
  a one-byte-short arena returning `kArenaExhausted` with the required size.

### Real GPU — `build_gpu_test.sh` (`sm_86`)

```
device: NVIDIA GeForce RTX 3060 (sm_86, 28 SMs)
  dc subsampled: samples=51916800 trace_fields=245760
  corner: values=16777216 guard=1110016 apply_checks=225280
  identity tx: cases=175 coeffs=177744 coded=322896
  chroma derivation: cases=9800 checks=29400
  dc_sign elements: cases=576 checks=1152
  dc_sign contexts: cases=4096 checks=4097
  arena: required=20224 persistent=5888 scratch=14336 capacity=20008704 peak=18045696
  direct device allocation: none (Arena only); reserved=20008704 bytes
intra_full_test: PASS (70920528 checks, 0 failures)
```

* `compute-sanitizer --tool memcheck`: **0 errors**.
* `ptxas` sm_86: forward kernel 40 registers, inverse 40, DC 40, corner 16,
  chroma 19; **0 stack frame, 0 spill stores/loads**, ≤5,792 B global tables.
* The GPU test repeats the full sweep above and adds argument-validation
  checks, a two-back-to-back-launch `used()`/peak check, and the one-byte-short
  arena test.  It allocates its device data from the same arena and uses two
  test-only kernels to evaluate the corner value/guard over their exhaustive
  domains through the shared device functions.

## Build / run

Local (this host, no GPU):

```sh
cd src/cuda/intra_full
./build_and_test.sh                 # emulation build + run
SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
```

On the RTX 3060 box (`sm_86`, CUDA 12.8):

```sh
cd /workspace/av2enc/src/cuda/intra_full
./build_gpu_test.sh                 # sm_86
CUDA_ARCH=sm_120 ./build_gpu_test.sh --no-run
```

Remote command actually used for the numbers above:

```sh
tar czf - -C src/cuda intra_full | \
  ssh -p 1531 root@142.127.68.223 \
  'tar xzf - -C /workspace/av2enc/src/cuda && chmod +x /workspace/av2enc/src/cuda/intra_full/*.sh'

ssh -p 1531 root@142.127.68.223 \
  'cd /workspace/av2enc/src/cuda/intra_full && ./build_gpu_test.sh'

ssh -p 1531 root@142.127.68.223 \
  'compute-sanitizer --tool memcheck /tmp/av2enc_intra_full_build/intra_full_test'
```

`tools/sync_remote.sh` (with `AV2ENC_HOST` / `AV2ENC_PORT`) syncs the whole
tree if the remote copy is older.

## Limitations / integration notes

1. **Not wired.** No encoder call site, no CMake change, no stream.  These are
   component entry points; the M1 scheduler still has to call them.
2. **DC output is 16-bit.** The shipped 8-bit pipeline narrows after the
   process; 10/12-bit edges are supported end-to-end in the API/tests.
3. **Corner is the non-IBP form.** `applyIbp == 1` is out of scope, as in the
   CPU reference.  The DC subsampled path is the >32-extent chroma case; M1
   streams still cannot signal it (audit I3/I19), so this is coverage for the
   decoder-side rule and future M4b blocks.
4. **Identity types only.** Non-identity TX_TYPEs remain in
   `src/cuda/tx_variants/`; this component deliberately rejects them so the
   "identity coverage" claim stays checkable.
5. **Batches are synchronous.** They copy descriptors/results and rewind
   scratch per call.  A future wavefront integration should keep the
   persistent result arrays and move the `mark()/rewind()` to the frame level;
   the kernel entry points themselves allocate and synchronize nothing.
6. **Chroma derivation is a decision, not syntax.** It returns the effective
   PlaneTxType/set; the §5.20.8.2 writer/reader stays in
   `src/encoder/transform/idtx_intra/type_syntax.*`.
7. **No end-to-end bitstream.** A component test cannot pass
   `tools/validate_dav2d.sh` by itself; no MD5 claim is made here.
