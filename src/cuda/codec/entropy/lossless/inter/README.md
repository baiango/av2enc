# CUDA lossless inter — reversible WHT, DPCM, FSC recon, segment QP

GPU port of the reconstructible part of `src/encoder/lossless/inter/`.  Every
integer operation matches the CPU reference bit for bit (0 LSB); the CPU module
is the oracle and is linked into both tests as an independent implementation.

Normative references (AV2 Bitstream & Decoding Process Specification v1.0.0,
2026-05-28, commit `38f28e7`):

| Section | Implemented |
|---|---|
| §7.15.2.2 | reversible 4x4 WHT forward/inverse, including the `(a - d) >> 1` odd-value lifting |
| §7.15.4 | lossless 2D inverse (rows with `Dequant = level * 8`, shift 3; then columns, shift 0) and `cumsum` |
| §7.14.4 | IDTX/FSC identity residual: `Dequant >> (3 - shift) == level` at `q = 0` |
| §5.20.6.1 | `find_tx_size` / `select_tx_size` lossless branch, 4x4..32x32 |
| §5.20.8.2 | `lossless_inter_tx_type` inputs (DCT_DCT WHT vs IDTX) are carried in every job |
| §5.18.2 | `LosslessArray`, `CodedLossless`, per-segment `get_qindex` |

## Files

| File | Purpose |
|---|---|
| `lossless_inter.h` | public API: jobs, `Workspace`, `required_device_bytes`, launchers |
| `lossless_inter_math.h` | pure `__host__ __device__` integer math and §5.18.2/§5.20.6.1 derivations (single source of truth) |
| `lossless_inter_device.cuh` | shared cooperative device header: forward/inverse/recon/DPCM routines with shared-memory scratch + barriers |
| `lossless_inter.cu` | kernels and launchers; descriptor upload only |
| `lossless_inter_host.cpp` | host mirrors of the derivations and `Workspace::init` / `required_device_bytes` |
| `lossless_inter_test.cu` | CPU-vs-GPU parity + arena accounting (needs nvcc + GPU) |
| `lossless_inter_emulation_test.cpp` | compiles the shared device header with CUDA keywords stubbed; no GPU needed |
| `build_and_test.sh` | `nvcc` present ? GPU test : emulation test |
| `build_gpu_test.sh` | CUDA build/run (`ARCH=sm_86` by default) |
| `build_emulation_test.sh` | host-only parity build/run |

## API sketch

```cpp
namespace av2enc::cuda::lossless_inter {

struct Workspace { /* persistent device staging arrays */ };
std::uint64_t required_device_bytes(const LaunchConfig&);
Status Workspace::init(Arena&, const LaunchConfig&,
                       std::uint64_t* required_bytes = nullptr,
                       std::string* error = nullptr);

// residual -> levels: WHT(difference(residual)) or difference(residual) [IDTX]
Status forward_blocks (Workspace&, const ForwardJob*, int count, stream);
// levels -> residual: inverse(levels) then optional cumsum
Status inverse_blocks (Workspace&, const InverseJob*, int count, stream);
// levels -> Clip1(pred + residual) with independent pred/recon strides
Status reconstruct_blocks(Workspace&, const ReconJob*, int count, stream);
// segment-QP gated reconstruction: skips non-lossless segments
Status reconstruct_blocks_segmented(Workspace&, const ReconJob*, int count,
                                    const FrameLossless* segment_qp, stream);

Status dpcm_difference / dpcm_cumsum (Workspace&, const DpcmJob*, int, stream);
Status wht4_forward / wht4_inverse   (Workspace&, const WhtJob*, int, stream);

Status select_tx_size_batch  (Workspace&, const TxSizeRequest*, int,
                              TxSizeDecision* host_out, stream);
Status derive_lossless_batch (Workspace&, const FrameQuantInputs*, int,
                              FrameLossless* host_out, stream);
}
```

`ForwardJob` / `InverseJob` / `ReconJob` carry `w`, `h`, `dct_dct`, `dpcm`,
`vertical`; `ReconJob` additionally carries `pred_stride`, `recon_stride`,
`frame_index` and `segment_id`.  A batch may mix sizes and transform types.

## Arena contract (mandatory)

All device memory is bump-allocated from the single persistent
`av2enc::cuda::Arena`; this component **never** calls `cudaMalloc`, `cudaFree`,
`cudaMallocHost` or any per-launch allocation.

* `required_device_bytes(config)` is exact: after `Workspace::init()` the arena
  watermark has advanced by precisely that many bytes (the test asserts it).
* `Workspace::init()` takes the arena by reference and allocates every staging
  array once with `allocate_array<T>()`.
* Launchers never allocate; they copy host descriptors into the persistent
  staging arrays and launch kernels.  Kernel scratch is shared memory.
* An undersized arena returns `Status::kArenaExhausted`, leaves the workspace
  invalid and reports the required size through `required_bytes`.
* Caller data buffers (residual, levels, pred, recon) must also be
  arena-resident; they are passed as device pointers inside jobs.
* Scratch reuse uses `arena.mark()` / `arena.rewind(mark)`; the test verifies a
  scratch allocation plus launch followed by `rewind` is net-zero, and that two
  launches leave `used()` and `peak_used()` unchanged.

`required_device_bytes` for the test configuration (`max_jobs` staging slots,
32x32 max block) — each array is rounded to Arena's 256-byte alignment:

| `max_jobs` | bytes |
|---:|---:|
| 1 | 2,560 |
| 64 | 28,928 |
| 128 | 57,600 |
| 256 | 114,944 |
| 1024 | 459,776 |

(9 arrays: `WhtJob`, `ForwardJob`, `InverseJob`, `DpcmJob`, `ReconJob`,
`TxSizeRequest`, `TxSizeDecision`, `FrameQuantInputs`, `FrameLossless`, plus
the per-job recon status byte.)

## Parity

`lossless_inter_emulation_test` (run locally on macOS, no GPU):

```
lossless_inter_emulation_test: 44704 cases, 140704 checks, 0 failures
```

Coverage: 4-sample WHT forward/inverse for every shift with forced-odd lifting
values (128,000 checks); 2D forward/inverse for all 16 combinations of
4/8/16/32 plus rectangular DCT_DCT and IDTX sizes (4x8 .. 32x16), both
`dct_dct` values; DPCM difference and cumsum at 4x4, 8x8 and rectangular sizes
in V and H; the difference-before-WHT encoder order over the same size set;
reconstruction with strided planes, interior offsets and blocks flush against
the plane edge (plus a no-out-of-block-writes check); exhaustive §5.20.6.1
selection over 4..64 sizes and every flag combination; and 6,000 randomised
§5.18.2 derivations.

`lossless_inter_test.cu` repeats the same comparisons through the real kernels
in batched launches (all sizes, both transforms, both DPCM modes, edge blocks),
adds device-side `derive_lossless_batch` / `select_tx_size_batch`, segment-QP
reconstruction status, and the arena accounting / one-byte-short negative test.

## Build and run

Local (emulation only, no nvcc):

```sh
cd src/cuda/lossless_inter && ./build_and_test.sh
```

Remote CUDA box (sync first; `/workspace` is not persistent):

```sh
AV2ENC_HOST=<user>@<host> AV2ENC_PORT=<port> tools/sync_remote.sh
ssh -p <port> <user>@<host> \
  'cd /workspace/av2enc/src/cuda/lossless_inter && ARCH=sm_86 ./build_and_test.sh'
```

`build_gpu_test.sh` compiles the CPU reference separately with function
sections and links with `--gc-sections`, so only the reversible helpers and the
two derivations are pulled in (the bitstream writer/reader is discarded).
