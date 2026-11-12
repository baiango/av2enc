# AV2 §7.16 TIP deblocking filter — CUDA

GPU implementation of the TIP deblocking filter, bit-exact with the CPU
reference in `src/encoder/filters/tip_deblock/`.

Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §7.16 (pages 528–529), invoking §7.17.3 filter
maximum width, §7.17.7.1 sample filtering and §7.17.7.2 filter choice.

Scope: 8-bit and 10-bit, 4:2:0 planar, even luma MI grid (frame padded to a
multiple of 8).  No bitstream integration, no hidden synchronisation, no global
state, no direct device allocation.

## Ordering (same contract as the CPU reference)

§7.16 replaces §7.17 for `TipFrameMode == TIP_FRAME_AS_OUTPUT`; it is followed
by no CDEF/CCSO/LR.  A `TIP_FRAME_AS_REF` frame never runs it.  The caller
resolves the §5.18.2 flags and quantizer state and passes the same
`TipDeblockOptions` POD the CPU API uses.

## Files

```text
src/cuda/tip_deblock/
  tip_deblock_device.h          shared §7.16 arithmetic + arena sizing plan
  tip_deblock.h                 CUDA API (device frame, arena frame, launchers)
  tip_deblock.cu                validation, edge→thread mapping, arena plumbing
  tip_deblock_test.cu           CPU-vs-GPU, arena-contract, validation, timing
  tip_deblock_emulation_test.cpp  host parity (no GPU / no nvcc)
  build_and_test.sh             dispatches to GPU when nvcc is present
  build_gpu_test.sh             nvcc build + run (sm_86 default)
  build_emulation_test.sh       host-only parity build/run
  README.md                     this file
```

`tip_deblock_device.h` is compiled as ordinary C++ by the emulation test
(`AV2ENC_TIP_HD` becomes `inline`) and as device code by nvcc.  The 8-bit
instantiation performs exactly the §7.17 kernel arithmetic; the 10-bit
instantiation differs only in `Clip1` (0..1023), because `side` already carries
the §7.16 bit-depth offsets.

## API

```cpp
namespace av2enc::cuda::tip_deblock {

using Options = av2enc::tip_deblock::TipDeblockOptions;

struct TipDeblockDeviceFrame {
  void* plane[3];                  // device pointers (uint8_t or uint16_t)
  int stride[3];                   // samples
  int mi_cols, mi_rows;            // even, > 0
  int subsampling_x, subsampling_y;   // 1, 1
  const uint16_t* tile_col_starts; int num_tile_cols;
  const uint16_t* tile_row_starts; int num_tile_rows;
};

cudaError_t tip_deblock_frame(const TipDeblockDeviceFrame&, const Options&,
                              cudaStream_t = nullptr);
cudaError_t tip_deblock_plane_pass(const TipDeblockDeviceFrame&, const Options&,
                                   int plane, int pass,
                                   cudaStream_t = nullptr);

// Arena-backed frame handle + bracket.
struct TipDeblockArenaFrame { /* frame, config, sizes, tile upload aliases */ };
Status tip_deblock_arena_frame_init(Arena&, const TipDeblockConfig&,
                                    TipDeblockArenaFrame*,
                                    std::size_t* required_bytes = nullptr);
Status tip_deblock_frame_arena(TipDeblockArenaFrame&, const Options&, Arena&,
                               cudaStream_t = nullptr);
Status tip_deblock_plane_pass_arena(TipDeblockArenaFrame&, const Options&,
                                    int plane, int pass, Arena&,
                                    cudaStream_t = nullptr);
}
```

`tip_deblock_frame()` is the integration path: it takes **caller-owned device
buffers** and allocates nothing.  The arena entry points exist for standalone
use and tests.

Validation mirrors the CPU subset: 8/10-bit, 4:2:0, even MI grid, non-null
planes, `stride >= plane width`, present tile arrays, a valid tip size, and
`apply_deblocking_filter_tip` only when §5.18.2 makes it codable.  Invalid input
returns `cudaErrorInvalidValue`.  A frame for which §7.16 is not active is a
successful no-op (the CPU's `kNotApplicable`).

## Device-memory contract (persistent Arena only)

All device memory comes from the single `av2enc::cuda::Arena`
(`src/cuda/arena.h`).  The component contains no `cudaMalloc`, `cudaFree`,
`cudaMallocHost`, or per-launch allocation; the test contains none either.

```cpp
std::size_t tip_deblock_required_device_bytes(const TipDeblockConfig&);
std::size_t tip_deblock_scratch_bytes(const TipDeblockConfig&);   // 0
```

The plan is

```text
align256(stride0*height0)                    // luma plane
+ align256(stride1*height1)                  // U
+ align256(stride2*height2)                  // V
+ align256(num_tile_cols * 2)
+ align256(num_tile_rows * 2)
+ scratch                                    // 0
```

`align256` is `Arena::kAlignment`.  §7.16 is strictly in-place, so the largest
launch needs the plane buffers and nothing else; `tip_deblock_scratch_bytes()`
is 0 and is included in the sum so the plan cannot drift silently.
`tip_deblock_arena_frame_init()` still brackets with `Arena::mark()` and
allocates any future scratch with `allocate_checked()`; the launchers themselves
never allocate.

Sizing examples (no tiles):

| frame | bit depth | required bytes | peak/persistent |
|---|---|---|---|
| 80×48 (MI 20×12, 2+2 tile starts) | 8 | 6,400 | 6,400 |
| 80×48 (MI 20×12, 2+2 tile starts) | 10 | 12,288 | 12,288 |
| 1920×1080 (MI 480×270, no tiles) | 8 | 3,110,400 (2.97 MiB) | 3,110,400 |
| 1920×1080 (MI 480×270, no tiles) | 10 | 6,220,800 (5.93 MiB) | 6,220,800 |

`required_bytes` is always written, including on failure.  A short arena returns
`Status::kArenaExhausted` and rewinds every partial allocation; the caller grows
the arena and retries.  Size the arena with `required_device_bytes(config) +
Arena::kAlignment` of slack to cover the arena's base-pointer alignment.

## Kernel design

Two launches per frame, matching the normative pass order:

```text
launch 1: pass 0 (vertical boundaries), planes Y/U/V
launch 2: pass 1 (horizontal boundaries), planes Y/U/V
```

One thread handles one §7.16 edge (one 4-sample segment).  A linear index is
split across the planes by per-plane edge counts:

```text
vertical:   x = (k+1)*sw,  y = 4*m      count = floor((w-1)/sw) * ceil(h/4)
horizontal: x = 4*m,       y = (k+1)*sh  count = ceil(w/4) * floor((h-1)/sh)
```

where `sw = tipSize >> subX` and `sh = tipSize >> subY`.  The `4` steps are
literal (§7.16 does not scale them by chroma subsampling).

Inter-thread safety needs no shared memory: the reach is `maxWidthPos` / `Neg`,
and the edge spacing (`sw`/`sh`) is strictly greater than that reach for every
reachable 4:2:0 class (luma 8→3, 16→6; chroma 8→3/2, 4→1).  Pass 0 and pass 1
are separated by a kernel launch, so vertical edges complete before horizontal
edges per plane; the three planes are independent, so interleaving them across
the two launches is equivalent to the CPU's plane-major order.

Measured on the RTX 3060 box, CUDA 12.8, `sm_86`:

```text
48 registers, 88-byte stack frame, 0 spill stores/loads, 496 bytes cmem  (8- and 10-bit kernels)
1080p 8-bit: 0.1235 ms/frame (50 iterations, CUDA events)
```

`sm_120` build: 55 registers, 224-byte stack, 0 spills (build-only; no sm_120
device in the container).

## Tests

`./build_and_test.sh` dispatches to the GPU test when `nvcc` is present and to
the host emulation test otherwise.

### GPU test — CPU-vs-GPU bit-exactness and arena contract

```text
checks: 17303  failures: 0  samples compared: 13107152  filtered samples: 57280
compute-sanitizer --tool memcheck: 0 errors
```

1. **Arena contract.**  `required_device_bytes` equals the persistent footprint
   for the handle (7,424 bytes at 80×48 with 88/48 sample strides and 2+2 tile
   starts); two `tip_deblock_frame_arena()` launches around `mark()` leave
   `used()` unchanged; an arena of `required - 1` returns
   `Status::kArenaExhausted` and reports the required size.
2. **All 256 `base_q_idx` × both tip sizes**, full frame, CPU vs GPU: exact,
   including guard columns.  20,640 filtered samples.
3. **All 256 `base_q_idx` × both tips × 3 planes × 2 passes**, CPU vs GPU after
   every single plane/pass call.
4. **60 random 8-bit frames** (random geometry, content, tiles, deltas,
   tip size, tile-disable): exact, 5,848 filtered samples.
5. **30 random 10-bit frames + all 256 `base_q_idx` × 2 tips at 10-bit**:
   GPU vs the shared-core host emulation, which the emulation test validates
   against an independent §7.16/§7.17.7 transcription.  30,792 filtered samples.
6. **Real reconstructed frame** `decoded.y4m` 640×360: exact; 734 (16×16) /
   1,245 (8×8) samples changed — the same counts the CPU reference reports.
7. **Validation**: null plane, short stride, odd MI grid, 4:4:4, 12-bit,
   non-signallable apply flag, bad plane/pass index; inactive frames are a
   no-op with unchanged pixels.
8. Guard columns are never written; 10-bit output stays in `[0, 1023]`.

### Host emulation test — no GPU needed

```text
checks: 3927146  failures: 0  samples compared: 47972256  filtered samples: 471035
```

Runs the same `tip_deblock_device.h` as plain C++ and compares against the CPU
reference for 8-bit and against an independent 10-bit transcription:

1. tip-size selection for every `enable_tip_refinemv × filter` combination.
2. strength derivation over all 256 `base_q_idx` values × 3 planes × 3 delta
   configurations × {8,10}-bit, plus a direct CPU comparison at 8-bit.
3. 3,000 randomized sample-kernel windows: 8-bit `filter_choice` /
   `sample_filtering` against the §7.17 kernels, 10-bit against the reference.
4. all 256 `base_q_idx` × 2 tips × 3 planes × 2 passes interleaved with the CPU
   (121,160 filtered samples).
5. full-frame parity on a 48×128 luma frame that reaches the 64×64-grid
   `sbEdge` clamp, with tile gating on and off (284,430 filtered samples).
6. 120 random 8-bit frames vs CPU; 60 random 10-bit + all-q 10-bit frames vs
   the independent reference.
7. the real 640×360 frame at 8 q/tip combinations.
8. arena plan arithmetic (alignment, stride override, 10-bit doubling).

## Build / run

Locally (no GPU):

```sh
./src/cuda/tip_deblock/build_and_test.sh          # emulation fallback
./src/cuda/tip_deblock/build_emulation_test.sh
```

On the RTX 3060 box:

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc/src/cuda/tip_deblock
./build_gpu_test.sh                 # sm_86
ARCH=sm_120 ./build_gpu_test.sh --no-run
compute-sanitizer --tool memcheck ../../build-tip-deblock-gpu/tip_deblock_test /workspace/av2enc
```

Concrete run used for the numbers above:

```sh
ssh -p 1531 root@142.127.68.223 'cd /workspace/av2enc/src/cuda/tip_deblock && ./build_gpu_test.sh'
```

Sync the tree with:

```sh
AV2ENC_HOST=root@142.127.68.223 AV2ENC_PORT=1531 tools/sync_remote.sh
```

## Limitations / integration notes

- 8-bit and 10-bit only; 4:2:0 only; even MI grid.  4:2:2 / 4:4:4 / 12-bit are
  rejected with `cudaErrorInvalidValue`.
- Tile-start array **contents** are device memory and are not read on the host;
  the caller must keep them in `[0, mi_cols)` / `[0, mi_rows)`.
- The CPU reference is 8-bit only.  10-bit correctness is established by the
  host-emulation/reference parity above and by GPU-vs-emulation parity; there is
  no 10-bit CPU oracle in this tree.
- Integration: for a `TIP_FRAME_AS_OUTPUT` frame, call
  `tip_deblock_frame()` in place of §7.17 and skip CDEF/CCSO/LR.  Pass the
  reconstructed planes (arena-resident in the encoder) and the TIP frame's
  `MiColStarts`/`MiRowStarts`; `tip_deblock_device_options()` maps the CPU
  options (including `select_tip_size()`) without re-deriving anything.
- No root `CMakeLists.txt`, `src/main.cpp`, core, or §7.17 edits; this component
  is standalone.
