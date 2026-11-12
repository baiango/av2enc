# Chroma loop restoration — CUDA pixel application

CUDA implementation of the per-unit chroma non-separable Wiener filter
(§7.20.3) with the 18-coefficient chroma layout of §5.20.10.6.  The per-unit
decision stays on the CPU (`src/encoder/filters/lr_chroma/`); this component
expands the coded unit descriptors and filters pixels.

Scope:

- 18 coefficient slots: 12 chroma tap positions (6 coefficient groups) + 12
  chroma-from-luma positions (12 coefficient slots, paired 6+6 by `uv_sym`);
- subsets 0/1/2 (`Wiener_Ns_Taps_Present[1]`), `uv_sym` pair mirroring, and
  bank/merged filter selection (`LR_BANK_SIZE == 4`, `bank_k` back from the
  newest entry);
- 8-bit and 10-bit samples (`uint8_t` / LSB-aligned `uint16_t`);
- 4:2:0, 4:2:2 and 4:4:4 plane geometry;
- `LoopRestorationSize` 32/64/128/256 chroma samples;
- §7.20.1 stripe sourcing (64-line stripes offset up by 8 luma samples) and
  the optional `MiCol/MiRowStart/End` tile clamp with
  `disable_loopfilters_across_tiles`.

No bitstream integration; `src/main.cpp`, the root `CMakeLists.txt`, core
files and other components are untouched.

## Files

```text
src/cuda/lr_chroma/
  lr_chroma_types.h              POD config/frame/unit/core/block descriptions
  lr_chroma_device.cuh           shared __host__ __device__ filter + resolver
  lr_chroma.h                    host API, Arena contract, kArenaTooSmall
  lr_chroma.cu                   sizing, workspace, kernels and launchers
  lr_chroma_reference.h          independent scalar test reference
  lr_chroma_emulation_test.cpp   host parity (no GPU, no nvcc)
  lr_chroma_test.cu              real CUDA parity + arena + audit test
  build_and_test.sh              dispatch: GPU if nvcc, else emulation
  build_emulation_test.sh        host-compiler build/run
  build_gpu_test.sh              nvcc build/run
  audit_device_visibility.sh     textual + nvcc device-visibility probe
  README.md                      this file
```

## API

```cpp
namespace av2enc::cuda::lr_chroma {

struct LrChromaConfig;   // MI grid, subsampling, bit depth, unit size, CfL index
struct LrChromaFrame;    // U/V planes, luma plane, optional tile grids
struct LrChromaUnit;     // on, merged, bank_k, subset, uv_sym, coeffs[18]

inline constexpr Status kArenaTooSmall = Status::kArenaExhausted;

std::size_t required_device_bytes(const LrChromaConfig&);
bool config_is_valid(const LrChromaConfig&);

Status allocate_checked(Arena&, const LrChromaConfig&, LrChromaWorkspace*,
                        std::size_t* required_bytes = nullptr,
                        std::string* error = nullptr);

class LrChromaWorkspace {
 public:
  Status create(Arena&, const LrChromaConfig&, std::string* error = nullptr);
  void reset() noexcept;
  // unit descriptors (caller/upload), expanded coefficients, active flags
  LrChromaUnit* units(int chroma);
  const std::int16_t* resolved(int chroma) const;
  const std::uint8_t* active(int chroma) const;
  // int unit_cols(int), unit_rows(int), unit_count(int), ...
};

Status upload_units(LrChromaWorkspace&, int chroma, const LrChromaUnit*,
                    int count, cudaStream_t = nullptr, std::string* = nullptr);

Status resolve_units(LrChromaWorkspace&, cudaStream_t = nullptr,
                     std::string* = nullptr);

Status apply(const LrChromaWorkspace&, const LrChromaFrame&,
             cudaStream_t = nullptr, std::string* = nullptr);

std::uint64_t frame_mark(const Arena&);              // Arena::mark() passthrough
bool frame_rewind(Arena&, std::uint64_t mark);       // Arena::rewind() passthrough

}
```

`LrChromaUnit` is the decoder-order decision for one unit:

| field | meaning |
|---|---|
| `on` | 1 = `RESTORE_WIENER_NONSEP`, 0 = copy of CdefFrame |
| `merged` | reuse a bank entry (coefficients are not recoded) |
| `bank_k` | bank distance back from newest (`0` newest, `-1` = newest) |
| `subset` | 0 → taps 0..5, 1 → 0..9, 2 → 0..17 |
| `uv_sym` | mirror pairs `j`/`j+1` for `j >= 6` (subset > 0) |
| `coeffs[18]` | coded/expanded coefficients; absent taps ignored |

`resolve_units` reproduces the §5.20.10.6 decoder expansion on the GPU: off
units zero their coefficients and do not touch the bank; merged units copy the
whole bank row (including taps outside the subset); non-merged units zero
absent taps, mirror `uv_sym` pairs, and push a new bank entry.

`apply` performs the §7.20 copy step (`LrFrame := CdefFrame`) and the filter in
one launch.  One thread owns one 4x4 luma MI block of one chroma plane, writes
a disjoint rectangle, and reads only CurrFrame/CdefFrame.  `dst` must not
alias `curr`/`cdef` (enforced with `kInvalidArgument`).

Call order for a frame:

```cpp
lrc::upload_units(ws, 0, units_u, count_u);
lrc::upload_units(ws, 1, units_v, count_v);
lrc::resolve_units(ws, stream);
lrc::apply(ws, frame, stream);
// one sync at the frame boundary is the integration owner's job
```

## Arena contract

Every device byte comes from the caller's single persistent
`av2enc::cuda::Arena` (`src/cuda/arena.h`).  `allocate_checked()` is the only
allocating entry point; it calls `Arena::allocate_checked` for each buffer,
takes one `Arena::mark()`, and on any failure rewinds to that mark, leaves the
workspace empty, and returns `kArenaTooSmall` with the exact
`required_device_bytes(config)`.  Launchers allocate nothing, copy nothing and
never synchronise.

Layout (all buffers 256-byte aligned, per chroma plane):

```text
units[plane]     unit_count * sizeof(LrChromaUnit)   (56 B on host and device)
resolved[plane]  unit_count * 18 * sizeof(int16_t)   (36 B per unit)
active[plane]    unit_count * 1
banks            sizeof(LrChromaBankState) * 2       (152 B each)
```

`required_device_bytes` does not depend on bit depth.  `unit_count =
count_units(unit_size, plane_width) * count_units(unit_size, plane_height)`.

| frame | 4:2:0 u32 | u64 | u128 | u256 |
|---|---:|---:|---:|---:|
| 720p (1280x720) | 41,984 | 12,800 | 4,608 | 2,048 |
| 1080p (1920x1080) | 95,744 | 23,552 | 7,168 | 3,072 |
| 2160p (3840x2160) | 380,416 | 95,744 | 23,552 | 7,168 |

| frame | 4:2:2 u32 | u64 | u128 | u256 |
|---|---:|---:|---:|---:|
| 1080p | 190,976 | 48,128 | 12,800 | 4,608 |
| 2160p | 759,808 | 190,976 | 48,128 | 12,800 |

| frame | 4:4:4 u32 | u64 | u128 | u256 |
|---|---:|---:|---:|---:|
| 1080p | 380,416 | 95,744 | 23,552 | 7,168 |
| 2160p | 1,518,592 | 380,416 | 95,744 | 23,552 |

The largest measured run (all 812 test cases, 8.58 MB peak) is far below a
frame buffer; the test's one-byte-short probe is the only reason the arena
peak in the raw log looks larger, and the final line reports the peak *before*
that probe.

## Kernels

- **`resolve_kernel`** — one block, two threads (one per chroma plane).  Each
  thread walks that plane's units in coding order with a 4-entry ring bank.
  Unit counts are a few thousand at 4K, so this is a few microseconds.
- **`apply_kernel<Sample>`** — 256 threads/block, `2 * mi_cols * mi_rows`
  threads.  Each thread computes the §7.20.1 block context, copies its
  rectangle from CdefFrame to LrFrame, and — when the unit is active — applies
  the expanded 18-tap filter.  Both `Sample = uint8_t` and `uint16_t` are
  instantiated.
- No `__syncthreads`, no cross-block ordering, no hidden device sync.

## Bit-exactness

`lr_chroma_device.cuh` is the only implementation of the arithmetic; the
kernels and the host-emulation test compile the same functions.

- **8-bit 4:2:0** is compared against `lr_chroma::chroma_apply()` — the CPU
  module's actual reference — for every subset (`0/1/2`), `uv_sym` on/off,
  unit size (`32/64/128/256`), six geometries (including 1-unit and odd MI
  grids) and 80 randomised layouts: 138 cases, both planes.
- **Tiled 8-bit 4:2:0** is compared against
  `restoration::loop_restoration_frame()` with real
  `MiColStart/End`/`MiRowStart/End` grids, because `chroma_apply()` has no
  tile inputs.  12 cases (unit sizes 32/64/128/256 × three element modes).
  Together the host run has 150 CPU-oracle cases; the GPU run additionally
  builds two tiny arena/validation frames (152).
- **10-bit and 4:2:2 / 4:4:4** are compared against the independent scalar
  transcription in `lr_chroma_reference.h`.  This is necessary: the CPU
  `chroma_apply()` hard-codes `bit_depth = 8`, and
  `loop_restoration_validate()` rejects subsampling other than 4:2:0.  The
  reference is also cross-checked against `chroma_apply()` on every 8-bit
  4:2:0 case, so the two oracles agree wherever both are defined.

Measured on the RTX 3060 box (CUDA 12.8, `sm_86`):

```text
host emulation:
  cases: 812
  PASS: 33270 checks, 0 failures
        (95209920 samples compared; chroma_apply oracle cases 150,
         resolved units 31196)

real CUDA:
  cases: 812
  PASS: 57009 checks, 0 failures
        (95209920 samples compared; chroma_apply oracle cases 152,
         resolved units 31196)
  arena peak 8584704 B (before sizing probe; capacity 100663296 B)
```

The higher GPU check count includes per-case arena/allocation/upload/download
checks and the arena/validation suites.

## Device-visibility audit

`build_and_test.sh` always runs `audit_device_visibility.sh` first:

- all 15 `lrc_*` definitions in `lr_chroma_device.cuh` are preceded by
  `AV2ENC_LRC_HD` (`__host__ __device__ __forceinline__` under nvcc);
- every template definition is marked;
- no `__constant__`, `__shared__`, `cudaMalloc`/`cudaFree`/`cudaMemcpy`, host
  `std::` containers, or namespace-scope objects in the shared header;
- the only tables are `kWienerNsConfigUv` / `kWienerFilters420` from
  `restoration_common.h`, whose `AV2ENC_LR_TABLE` macro gives them device
  storage in the nvcc pass (the same mechanism the luma Wiener kernel uses);
- when nvcc is available, a probe `.cu` calls every device entry point from a
  `__global__` function.  Result on `sm_86`: `nvcc device-visibility probe:
  PASS`.

The GPU test also launches a runtime visibility probe with both sample types,
and the apply kernel itself instantiates `lrc_process_block<uint8_t>` and
`<uint16_t>`.

## Tests

`build_and_test.sh` (or `build_emulation_test.sh` on a host without nvcc)
covers:

- resolver parity for every case: device expansion vs `ref::resolve_plane`,
  including bank/merged references and deliberately different `j+1` values
  under `uv_sym`;
- pixel parity for all 812 cases and both chroma planes, sample-for-sample;
- all 8/10-bit × 4:2:0/4:2:2/4:4:4 × unit-size combinations;
- edge grids 7x5, 1-unit grids, odd 65x33 / 97x49 / 257x129 MI grids,
  80 random MI layouts, and tiled 140x70 MI frames;
- one-byte-short `kArenaTooSmall` (exact reported size, empty output,
  unchanged `used()`);
- two `apply` launches plus mark/rewind with no net growth and stable peak;
- workspace re-init after rewind reuses the same addresses;
- argument validation: wrong upload count, null units, non-chroma plane,
  bit-depth mismatch, `dst == cdef`, short strides, missing tile grids,
  invalid configs (12-bit, unit 512, `sub_x == 2`).

`tools/check_arena.sh` reports `PASS: no raw CUDA allocations outside
arena.cpp/context.cpp`.

## Build and run

```sh
src/cuda/lr_chroma/build_and_test.sh                    # auto: GPU if nvcc
src/cuda/lr_chroma/build_gpu_test.sh                    # nvcc + device
ARCH=sm_120 src/cuda/lr_chroma/build_gpu_test.sh --no-run
src/cuda/lr_chroma/build_emulation_test.sh              # no nvcc/GPU
```

Neither script edits the root CMake build or any other component.

## Remote command

The numbers above were measured on the RTX 3060 box (Vast.ai, port 1531)
with nvcc 12.8.93:

```sh
# only this new directory needs to be copied; all dependencies already match
tar czf - src/cuda/lr_chroma \
  | ssh -p 1531 root@142.127.68.223 'tar xzf - -C /workspace/av2enc'

ssh -p 1531 root@142.127.68.223 \
  'cd /workspace/av2enc/src/cuda/lr_chroma && ./build_and_test.sh'
```

For a generic box use `tools/sync_remote.sh` with `AV2ENC_HOST`/`AV2ENC_PORT`
and then the same `ssh ... ./build_and_test.sh` command.  The secondary build
target is `ARCH=sm_120 ./build_gpu_test.sh`.

## Limitations

- Chroma plane extents are derived from the MI grid:
  `width = mi_cols*4 >> sub_x`, `height = mi_rows*4 >> sub_y`.  Callers must
  provide allocation covering those (MI-padded) extents, exactly like the CPU
  `chroma_apply()`.
- `dst` must be distinct from both `curr` and `cdef`; the launcher rejects
  aliasing.
- Unit size 32 with 4:4:4 is outside §6.17.7.7 (`>= 64 >> SubsamplingY`) but
  is accepted by this pixel core and exercised as a geometry stress case.  A
  real decision layer must not signal it.
- There is no `frame_filters_on`/pixel-class path here: a frame-level filter
  must be expanded by the decision layer into the same per-unit descriptors.
  Chroma frame filters always use class 0 (§7.20.3), so this is a direct
  mapping.
- Lossless-segment and BRU skips are not modelled, matching
  `lr_chroma::ChromaLrInput`, which has no such inputs.  Tile clamping is.
- No `dav2d` gate: this is a pixel filter with no bitstream output; the
  integration owner wires it into the M1/M2 path.
