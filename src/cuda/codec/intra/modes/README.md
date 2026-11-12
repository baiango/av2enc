# CUDA M1 full intra prediction

AV2 v1.0.0 intra prediction, encoder side: DC / V / H / directional with angle
deltas / smooth / smooth-V / smooth-H / Paeth, plus the normative intra edge
filter (corner filter, filter-type selection, 5-tap edge filtering, WAIP).

Component-owned files (intra-modes CUDA agent):

```text
src/cuda/intra_modes/intra_modes.h                 API + normative tables + contract
src/cuda/intra_modes/intra_predict_device.cuh      shared device prediction body
src/cuda/intra_modes/intra_modes.cu                kernels (include the .cuh)
src/cuda/intra_modes/intra_modes_test.cu           independent CPU reference + tests
src/cuda/intra_modes/intra_device_emulation_test.cpp  host-emulation parity test
src/cuda/intra_modes/build_intra_modes_test.sh     standalone build/run script
src/cuda/intra_modes/build_emulation_test.sh       no-nvcc host emulation test
src/cuda/intra_modes/README.md                     this file
```

`intra_predict_device.cuh` is the single device implementation. The standalone
kernels and the M1 GPU chroma full-UV search
(`src/cuda/m1/chroma/still_chroma.cu`) both call
`intra_run_block_device()`, so the two callers cannot drift. The header is
also compiled by `intra_device_emulation_test.cpp` on machines without nvcc:
CUDA keywords are stubbed, one emulated thread runs the exact integer code, and
every prediction byte is checked against the CPU module.

```sh
./build_emulation_test.sh
# intra_device_emulation_test: PASS (2782 checks)
```

Not wired into `src/main.cpp` or the root `CMakeLists.txt`. Nothing outside
`src/cuda/intra_modes/` is modified.

## Normative scope

AV2 Bitstream & Decoding Process Specification v1.0.0 (commit `38f28e7`):

| Section | Process |
|---|---|
| §7.13.2.1 | `AboveRow` / `LeftCol` construction, MAX_X/MAX_Y clamping |
| §7.13.2.2 | Basic intra (Paeth) |
| §7.13.2.7 | Directional intra + edge-filter orchestration |
| §7.13.2.8 | Single directional prediction (luma IDIF / chroma bilinear) |
| §7.13.2.10 | DC intra |
| §7.13.2.13 | Smooth / smooth-V / smooth-H |
| §7.13.2.14 | Filter corner |
| §7.13.2.15/16 | Filter type above / left (neighbour smooth flags) |
| §7.13.2.17 | Intra edge filter strength selection |
| §7.13.2.18 | Intra edge filter (5-tap, strengths 1..3) |
| §5.20.7.30 | `wide_angle_mapping` (WAIP) |
| §7.13.3.22 | `resolve_divisor` / `approx_divide`, `Div_Lut` |

In scope: 8-bit, luma (plane 0, `enableIdif = 1`) and chroma (plane > 0,
bilinear), block extents 4/8/16/32/64 per axis (square or rectangular),
`angle_delta` in `[-3, +3]` (`ANGLE_STEP = 3`), WAIP, and the edge filter.

Out of scope for M1 (M4b): MRL (`MrlIndex != 0`), DIP, IBP, CfL, MHCCP,
palette/IntraBC, secondary MRL prediction, 10-bit. `MrlIndex` is fixed at 0;
the API has no field for it, so an M4b extension must add one and extent the
edge arrays.

### Normative constant tables

`Mode_To_Angle`, `Dr_Intra_Derivative`, `Dr_Interp_Filter`,
`Intra_Edge_Kernel`, and `Div_Lut` are defined once as macros in
`intra_modes.h`. The host `constexpr` accessors and the device tables in
`intra_modes.cu` both expand those macros, so the two copies cannot drift.
Values are transcribed verbatim from the v1.0.0 attachments
(`mode_to_angle.h`, `dr_intra_derivative.h`) and §7.13.2.8 / §7.13.2.18 /
§7.13.3.22. Do not recompute `Dr_Intra_Derivative` from `tan()`: the published
table is the normative one and is not a clean analytic sequence.

The test reports which derivative indices are reachable through the AV2 mode
set: **27/90**, range `[3, 87]` (indices 0..2 and 88..89 cannot be produced by
3-degree steps from the eight base angles). Those 27 entries are exercised by
the correctness sweep; MRL in M4b would make the adjacent ±1 entries reachable
and they are already present.

## API

All entry points are pure: device pointers + `cudaStream_t`, no global mutable
state, no allocations, no device synchronisation. Argument errors return
`cudaErrorInvalidValue`; launch failures come from `cudaGetLastError()`.

```cpp
namespace av2enc::cuda::intra_modes {

struct IntraBlock {
  const std::uint8_t* recon_plane;  // reconstructed plane (prediction input)
  int recon_stride;
  int x0, y0;                       // block origin in plane samples
  int width, height;                // 4/8/16/32/64 per axis
  int max_x, max_y;                 // last visible column/row, inclusive
  int have_left, have_above;        // availability (0/1)
  int num4_above_right;             // valid 4x4 units above-right
  int num4_below_left;              // valid 4x4 units below-left
  int plane;                        // 0 = luma (IDIF), >0 = chroma
  IntraMode mode;                   // kDcPred .. kPaethPred
  int angle_delta;                  // [-3, 3], directional modes only
  int filter_type_above;            // neighbour-smooth flag (§7.13.2.15)
  int filter_type_left;             // neighbour-smooth flag (§7.13.2.16)
  int enable_edge_filter;           // normative M1 setting = 1
  const std::uint8_t* source_plane; // residual entry points only
  int source_stride;
  int source_max_x, source_max_y;
};

cudaError_t intra_predict(const IntraBlock&, std::uint8_t* pred,
                          cudaStream_t = nullptr);

cudaError_t intra_predict_residual(const IntraBlock&, std::uint8_t* pred,
                                   std::int16_t* residual,
                                   cudaStream_t = nullptr);

struct IntraJob { IntraBlock block; std::uint8_t* pred; std::int16_t* residual; };

cudaError_t intra_predict_batch(const IntraJob* d_jobs, int job_count,
                                cudaStream_t = nullptr);

}  // namespace av2enc::cuda::intra_modes
```

Semantics:

- `pred` is `width * height` `uint8_t`, row-major, tightly packed.
- `residual = source(clamped) - prediction` (`int16`, range `[-255, 255]`).
- Prediction reads **reconstructed** samples at `x0-1` (left column) and
  `y0-1` (above row). All edge reads are clamped to `[0, max_x] x [0, max_y]`,
  so partial right/bottom blocks reproduce the spec's `MAX_X`/`MAX_Y`.
- `num4_above_right` / `num4_below_left` bound the above-right and below-left
  extension used by directional prediction (spec: `4 * num4` samples, clamped
  to `max_x` / `max_y`). Callers that do not track extension can pass 0.
- `filter_type_above/left` are the `get_filter_type_above/left` results (1 if
  the neighbouring block used a smooth mode). The kernel ORs them into one
  filter type, matching the IBP-disabled branch.
- `have_left`/`have_above` are caller-supplied context. When 0, the spec's
  constant padding is used (`AboveRow = 127`, `LeftCol = 128`, corner = 128).
- `enable_edge_filter = 1` is the normative M1 behaviour. Passing 0 gives the
  unfiltered directional predictor and is only used by tests.
- `plane = 0` enables luma IDIF (4-tap `Dr_Interp_Filter`); `plane > 0` uses the
  chroma bilinear interpolation. Note the spec form:
  `enableIdif = (plane == 0)`.
- `intra_predict_batch` launches one CUDA block per job. Jobs must be mutually
  independent: a job may read plane regions that earlier blocks *already
  reconstructed*, but no job may read what another job in the same launch
  writes. This is the "frozen prediction inputs" restriction documented in
  `docs/negative-results.md`; the actual dependency-ordered scheduling is the
  wavefront/CUDA-graph scheduler's job.

Helper functions exposed for tests and table checks: `mode_to_angle`,
`dr_intra_derivative`, `dr_interp_filter`, `intra_edge_kernel`, `div_lut`,
`wide_angle_map`, `intra_edge_strength`, `intra_pangle`, `is_directional_mode`.

## Bit-exactness contract

Integer only. No FP, no `-fmad` sensitivity.

```text
Round2(v, n)   = (v + (1 << (n-1))) >> n        (n <= 0 -> v)
Clip1(v)       = Clip3(0, 255, v)
approx_divide(n, d): (shift, factor) = resolve_divisor(d)
                     Round2(n * factor, shift)
```

`resolve_divisor` and `Div_Lut` follow §7.13.3.22 exactly. Negative values use
arithmetic right shift, matching the spec's two's-complement definition and
the CPU reference (`gcc`/`nvcc` on Linux both implement `>>` as arithmetic).

The test contains an **independent scalar CPU reference** for the same
sections (`ref_predict`). It shares only the normative constant tables; every
formula is written out again, so a formula error in the kernel is caught
rather than mirrored.

`src/encoder/intra_modes/` landed during development. The optional
`--cpu-parity` build links it and compares it with the kernels over a sweep of
sizes, modes, deltas, planes and both edge-filter settings:

```
edge_filter=0  samples=628256  mismatches=0  max_abs_diff=0   PASS
edge_filter=1  samples=628256  mismatches=244858             (see below)
```

With the edge filter disabled, the CUDA kernels are **exactly equal** to the
CPU module (edge construction, WAIP, DC/smooth/Paeth, IDIF and bilinear,
partial blocks). With it enabled the CPU module diverges because
`intra_edge_filter()` in `src/encoder/intra_modes/intra_modes.cpp` filters the
row in place and reads already-filtered samples, while §7.13.2.18 first builds
a snapshot (`edge[i] = LeftCol[i-1]`) and reads the original values. CUDA
follows the spec. The CPU test even documents the in-place interpretation
("later taps see the already filtered samples"), so this is a deliberate
cross-agent disagreement, not a typo. Until it is resolved, the encoder's
intra reconstruction cannot match a spec-conforming decoder whenever the edge
filter is active. `--cpu-parity` reports this, and fails only if the
edge-filter-off pass mismatches.

Unavailable-neighbour fillers used here (§7.13.2.1): `AboveRow = 127`,
`LeftCol = 129`, corner = 128 for 8-bit.

### Test result (RTX 3060 12 GB, sm_86, CUDA 12.8)

```
-- Dr_Intra_Derivative coverage: 27/90 entries, reachable range [3, 87] --
-- full mode/size/position matrix --   cases=46319 samples=44067184
-- randomized blocks --                cases=3000  samples=2861184
-- 640x360 bottom / right edge --      cases=780   samples=1163136
-- 1920x1080-style blocks --           cases=319   samples=781376
-- batched API vs CPU reference --     jobs=1740   (240 grid + 1500 mixed)
-- summary --
   pred     samples=48872880 mismatches=0 max_abs_diff=0
   residual samples=48872880 mismatches=0 max_abs_diff=0
   batch    samples= 3314464 mismatches=0 max_abs_diff=0
   total    samples=101060224 mismatches=0
PASS
```

`compute-sanitizer --tool memcheck` on the `--quick` profile: **0 errors**.

Coverage details:

- all 13 modes; directional modes with all 7 angle deltas; filter-type
  combinations `00/01/10/11`; luma and chroma; edge filter on/off
- optional `--cpu-parity` cross-check against `src/encoder/intra_modes/`
- all sizes 4x4 … 64x64 plus 64x32, 32x64, 64x16, 16x64, 8x16, 16x8, 32x8, 8x32
- positions with no neighbours, above-only, left-only, interior, right edge,
  bottom edge (partial blocks: 64x64 with 40 visible rows, 32x32 with 24, etc.)
- synthetic 640x360 bottom/right edges and 1920x1080 bottom/right edges
- batch path compared sample-for-sample with the same reference

## Timing (microbenchmark only)

One launch, predictor only (no transform, no RDO, no mode search), recon plane
pre-filled, D45 with `angle_delta = 1` (IDIF + edge filter). `ms/call` is the
mean batch launch; `Msamples/s` is predicted luma samples per second. The box
is a shared Vast.ai instance: expect roughly ±10–15% run-to-run variation.

| Case | ms/call | blocks/s | Msamples/s |
|---|---:|---:|---:|
| single 32x32 launch (1 block) | 0.005 | ~190 k | ~195 |
| single 64x64 launch (1 block) | 0.010 | ~99 k | ~405 |
| batch 640x360, 32x32 grid (240 blocks) | 0.010 | ~23 M | ~22,000 |
| batch 640x360, 16x16 grid (920 blocks) | 0.015–0.017 | 55–61 M | 14,000–15,200 |
| batch 1920x1080, 32x32 grid (2,100 blocks) | 0.069–0.079 | 27–30 M | 26,000–30,000 |
| batch 1920x1080, 64x64 grid (570 blocks) | 0.062–0.066 | 8.7–9.2 M | 31,500–33,500 |
| batch 1920x1080, 8x8 grid (32,400 blocks) | 0.34–0.43 | 75–94 M | 4,800–6,000 |

Interpretation:

- A single block per launch costs 5–10 µs: launch-latency bound, exactly the
  failure mode already recorded in `docs/negative-results.md`. Do not use the
  per-block entry point for video-scale rasters; use `intra_predict_batch`
  (frozen inputs) or the wavefront scheduler.
- A 1080p luma predictor pass over 32x32 blocks is ~70–80 µs; 64x64 blocks
  ~62–66 µs. That is the predictor only, not intra mode search.
- 8x8 blocks are occupancy-bound: the batch kernel uses 128 threads per CUDA
  block regardless of block area, so an 8x8 block runs 64 pixels with 4 warps.
  A warp-per-block (or size-class) batch is future work; it belongs with the
  scheduler agent, not this component. For M1, 16–64 blocks are the useful
  range.

These are component numbers, not encoder throughput. They do not include the
residual transform, reconstruction write-back, or ED/rate decisions.

## Build / run

```sh
cd src/cuda/intra_modes
./build_intra_modes_test.sh                  # sm_86 build + full run
./build_intra_modes_test.sh --no-run         # build only
./build_intra_modes_test.sh --cpu-parity     # + CPU-module parity sweep
ARCH=sm_120 ./build_intra_modes_test.sh      # secondary target
./build-intra-modes/intra_modes_test --quick # reduced sweep (memcheck)
```

Validated on the RTX 3060 box (`ssh -p <port> <user>@<host>`), nvcc 12.8:

- `sm_86`: 38–54 registers, 0 stack frame, 0 spills, 2,156 B shared memory
- `sm_120`: compiles cleanly with the same sources
- `-lineinfo`, `-Wall -Wextra -Wconversion -Wshadow`: no warnings

## Blockers / notes for integration

1. **CPU module edge-filter divergence (blocking recon parity).** The CPU
   module's `intra_edge_filter()` filters in place; §7.13.2.18 filters a
   snapshot. Everything else is bit-exact (`--cpu-parity`, 628,256 samples
   with the filter off, 0 mismatches). The CPU side needs the snapshot fix
   before the M1 reconstruction gate can pass. CUDA intentionally does not
   reproduce the CPU behaviour; match-to-CPU here would mean matching a
   decoder mismatch.
2. **`have_left` / `have_above` / `num4_*` / filter types are caller inputs.**
   The integration layer must derive them from the partition and neighbour
   modes exactly as the spec does (`AvailU`/`AvailL`, `MiColEnd`, `MiRowEnd`).
   Passing 0 for `num4_*` is safe (shorter extended edges, still conformant)
   but loses some prediction accuracy at block edges.
3. **One block per descriptor.** When the transform size is smaller than the
   coding block, the caller must invoke the predictor per transform block in
   raster order so later blocks see earlier reconstruction.
4. **Batch independence.** Do not batch dependent blocks; the current kernels
   read the plane they are handed and do not synchronise between jobs.
5. **8x8 batch efficiency.** See timing; a warp-per-block variant would help
   small-block-heavy rasters.
6. **MRL/DIP/IBP/CfL not implemented.** M4b must extend the descriptor (MRL
   index, DIP features) and the edge storage; the current arrays are sized for
   `MrlIndex = 0` (`w+h+2` plus two head slots).
7. **No end-to-end stream.** This component produces predictions and residuals
   only; it cannot pass `tools/validate_dav2d.sh` by itself and makes no
   bitstream claim.
