# CUDA runtime layer (`src/cuda`)

Status: implemented and tested on the Vast.ai RTX 3060 12 GB dev box
(Ubuntu 24.04, driver 570.181, CUDA 12.8.93, `sm_86`), 2026-11-11.

This layer is host-side CUDA plumbing shared by the transform, intra and M0
pipeline kernels. It contains no encoder kernels and does not link the encoder
target itself. The integration agent owns the root `CMakeLists.txt` change.

## Files

| File | Purpose |
|---|---|
| `error.h` | `Status` enum, `status_name()`, `cuda_error_name()`, `status_from_cuda()`, `last_error()`, `format_status()`. Header-only. |
| `context.h/.cpp` | Device enumeration/query, one non-blocking stream, explicit sync, pinned `StagingBuffer`, raw async upload/download helpers. |
| `arena.h/.cpp` | One persistent `cudaMalloc` block, 256-byte bump allocator, device-derived capacity with optional `--vram-cap`, peak tracking, `cudaMemGetInfo` reporting. |
| `context_test.cu` | Device query, stream lifecycle, limits/alignment/peak, pinned round-trip, kernel round-trip, no-malloc, VRAM reporting tests. |
| `build_runtime_test.sh` | Standalone `nvcc` build and run for the test (no CMake). |
| `README-runtime.md` | This file. |

Not owned by this layer: `transform.*`, `intra.*`, `entropy.*`, `m0_pipeline.*`,
the root `CMakeLists.txt`, `src/main.cpp`, `src/encoder/`, `src/entropy/`,
`src/bitstream/`, `tests/`.

## Integration order

```cpp
#include "cuda/context.h"
#include "cuda/arena.h"

av2enc::cuda::Context context;
std::string error;
av2enc::cuda::Status status = context.init(/*device_index=*/0, &error);
if (!av2enc::cuda::is_ok(status)) { /* log error */ }

av2enc::cuda::Arena arena;
status = arena.init(/*capacity_bytes=*/device_capacity, Arena::kAlignment, &error);
if (!av2enc::cuda::is_ok(status)) { /* log error; do not silently shrink */ }

av2enc::cuda::StagingBuffer upload;
status = upload.init(max_frame_bytes);    // cudaMallocHost, no device VRAM
```

Order matters: `Context::init()` selects the device and creates the primary
context before `Arena::init()` samples VRAM. The arena and staging buffers must
be destroyed before `Context::shutdown()` (normal reverse-declaration order
handles this).

Per frame, use the one stream and sync once at the boundary:

```cpp
upload.upload(context.stream(), device_frame, host_frame, frame_bytes);
// ... enqueue kernels on context.stream() ...
download.download(context.stream(), device_payload, payload_bytes);
context.synchronize();                    // the only blocking call
// read download.data()
```

The layer never calls `cudaDeviceSynchronize()`. `StagingBuffer::upload()` and
`download()` are asynchronous; they do not wait. `Context::synchronize()` calls
`cudaStreamSynchronize()` once and then picks up any sticky launch error.

## API summary

### Errors (`error.h`)

```cpp
enum class Status { kOk, kInvalidArgument, kNotInitialized, kAlreadyInitialized,
                    kDeviceUnavailable, kOutOfMemory, kCapExceeded,
                    kArenaExhausted, kBusy, kCudaError };
```

- `status_from_cuda(cudaError_t)` maps and clears CUDA's sticky per-thread
  error; `cudaErrorNotReady` maps to `kBusy`.
- `last_error()` reads and clears the pending runtime error.
- Setup calls accept `std::string* error` for a human-readable reason.

### Device / stream (`context.h`)

```cpp
Status device_count(int* count);
Status query_device(int index, DeviceInfo* info);

struct DeviceInfo {
  int index; char name[256];
  int cc_major, cc_minor, sm_count;
  int max_threads_per_sm, max_threads_per_block;
  std::size_t shared_mem_per_block, shared_mem_per_block_optin, shared_mem_per_sm;
  int regs_per_sm;
  std::uint64_t total_vram_bytes, free_vram_bytes;
  int memory_clock_khz, memory_bus_width_bits;
  double memory_bandwidth_gbps() const;
};

class Context {
  Status init(int device_index = 0, std::string* error = nullptr);
  void shutdown();
  bool initialized() const;
  int device_index() const;
  const DeviceInfo& device() const;
  cudaStream_t stream() const;
  Status refresh_vram();                  // cudaMemGetInfo
  std::uint64_t total_vram_bytes() const;
  std::uint64_t free_vram_bytes() const;
  std::uint64_t used_vram_bytes() const;
  Status synchronize();                   // one stream sync; frame boundary only
  Status query_stream();                  // kOk or kBusy, non-blocking
};
```

`query_device()` may create the primary context on the queried device. It
switches to that device, reads `cudaMemGetInfo()`, and restores the previous
device.

### Pinned staging (`context.h`)

```cpp
class StagingBuffer {
  Status init(std::size_t capacity_bytes, std::string* error = nullptr);  // pinned
  void release();
  void* data(); const void* data() const;
  std::size_t capacity() const;
  std::size_t downloaded_bytes() const;
  Status upload(cudaStream_t, void* device_dst, const void* host_src, std::size_t bytes);
  Status download(cudaStream_t, const void* device_src, std::size_t bytes);
};

Status upload_async(cudaStream_t, void* device_dst, const void* host_src, std::size_t);
Status download_async(cudaStream_t, void* host_dst, const void* device_src, std::size_t);
```

`upload()` copies host bytes into the pinned block, then enqueues H2D.
`download()` enqueues D2H into the pinned block; the data is readable from
`data()` only after the stream completes. Both return `kInvalidArgument` for
zero bytes, null pointers, or a size above `capacity()`. There is no hidden
sync and no implicit reuse guard: do not touch the staging buffer until the
stream has finished.

### Persistent arena (`arena.h`)

```cpp
class Arena {
  static constexpr std::uint64_t kDeviceHeadroomBytes = 512 MiB;
  static constexpr std::size_t   kAlignment            = 256;

  Status init(std::uint64_t capacity_bytes,
              std::size_t alignment = kAlignment, std::string* error = nullptr);
  Status init_mb(std::uint32_t capacity_mb, ...);
  void release();

  static void set_capacity_cap_bytes(std::uint64_t bytes);  // 0 = no bound
  static std::uint64_t capacity_cap_bytes();

  void* allocate(std::size_t bytes, std::size_t alignment = kAlignment);
  Status allocate_checked(std::size_t bytes, void** out,
                          std::size_t alignment = kAlignment);
  template <typename T> T* allocate_array(std::size_t count, ...);

  std::uint64_t reserved_bytes() const;   // cudaMalloc size
  std::uint64_t capacity() const;         // reserved size
  std::uint64_t usable_capacity() const;  // after base-alignment padding
  std::uint64_t used() const;
  std::uint64_t peak_used() const;        // high-water mark, not reduced by rewind
  std::uint64_t remaining() const;
  std::size_t alignment() const;
  void* base();

  std::uint64_t mark() const;
  bool rewind(std::uint64_t mark);
  void reset();

  Status sample_vram();
  std::uint64_t device_total_bytes() const;
  std::uint64_t device_free_bytes() const;
  std::uint64_t device_used_bytes() const;       // total - free
  std::uint64_t device_peak_used_bytes() const;  // max over samples
  std::uint64_t device_free_at_init_bytes() const;
  static std::uint64_t process_reserved_bytes();
};
```

## VRAM sizing policy

There is no fixed preset and no validation cap. The runtime queries the device
at first use and reserves:

```text
capacity = min(total_vram, free_vram) - 512 MiB
```

An optional `--vram-cap <MiB>` bounds that value; `0` means the maximum the
device can serve. If the reservation fails because another process grew between
the query and `cudaMalloc`, the pipeline halves the request and retries.

`Arena::set_capacity_cap_bytes()` only records the request bound; it does not
reject allocations. `cudaMalloc` is the hard limit.

`Arena::init()` is the only device allocation. `allocate()` only advances a
bump pointer; `restart`/`rewind` reuse a scratch region. There is no
`cudaMalloc` in `allocate()` and no per-allocation `cudaFree`.

Every allocation is at least 256-byte aligned; allocation sizes are rounded up
so the next bump offset stays aligned. Larger per-allocation alignments are
supported (power of two, >= 256); the allocator pads relative to the actual
base address, so the returned pointer is correctly aligned even if the arena
base is only 256-byte aligned.

`peak_used()` is the arena high-water mark. `sample_vram()` updates the
device-wide `cudaMemGetInfo()` numbers and the device peak; it is not called
automatically per allocation. Call it after large allocations or at frame
boundaries. Device statistics remain readable after `release()` for reporting.

`cudaMemGetInfo()` is device-wide, so on a shared GPU it includes other
processes. The full-cap test therefore checks the arena delta rather than
requiring the device-wide peak to match the reservation when another container
is resident.

## Build and test

On the 3060 box:

```sh
cd /workspace/av2enc/src/cuda
./build_runtime_test.sh                    # normal suite, 231 checks
./build_runtime_test.sh --full-cap         # adds the 7244 MiB arena + VRAM report
```

The script uses `nvcc -std=c++20 -arch=sm_86 -lineinfo -O2 -g` and
`-Wall -Wextra -Wconversion -Wshadow`. `-Wpedantic` is omitted for nvcc because
nvcc's generated line directives trigger a warning on every source line; a
host-only g++ build of `context.cpp`/`arena.cpp` with the full warning set
(`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`) is clean.

`CUDA_ARCH=sm_120` and `BUILD_DIR=...` are supported for the secondary target.

CMake integration (owned by the integration agent; the root file must not be
edited by component agents): add `src/cuda/context.cpp` and
`src/cuda/arena.cpp` to the CUDA-enabled target. The root project already
provides `${CMAKE_CURRENT_SOURCE_DIR}/src` as an include directory and links
`CUDA::cudart`, so `#include "cuda/context.h"` works.

## Test coverage

| Test | What it proves |
|---|---|
| `device_query` | Device count/name, CC 8.6, 28 SMs, shared memory, total/free VRAM, bandwidth, invalid-index handling. |
| `context_stream` | Init/shutdown/re-init, one non-blocking stream, refresh, idle query, sync. |
| `arena_cap` | Request-bound API, 512 MiB device headroom, zero-cap rejection, process reservation accounting. |
| `arena_alignment` | 256-byte and 4096-byte alignment, size rounding, invalid alignments, mark/rewind/peak behavior. |
| `arena_over_cap` | `SIZE_MAX` rejection, exact-fill and rewind (no fixed cap). |
| `arena_peak` | Peak high-water tracking across alloc/reset/rewind, failed allocations do not raise the peak. |
| `staging_roundtrip` | Pinned allocation, 1 MiB H2D/D2H round-trip twice, error cases, release/re-init. |
| `kernel_roundtrip` | Arena memory is writable by a kernel on the context stream; D2H result matches. |
| `no_device_alloc` | 4096 arena allocations (16 MiB) do not change `cudaMemGetInfo()` free VRAM. |
| `vram_reporting` | `cudaMemGetInfo()` total/free/used/peak consistency with `Context`. |
| `full_cap_vram` (`--full-cap`) | Real 7244 MiB allocation, touch, head/tail readback, over-cap rejection at the top, device peak, release recovery. |

## Measured results

Exclusive run on the 3060 12 GB dev box (2026-11-11, CUDA 12.8.93):

```
=== av2enc CUDA runtime layer tests ===
CUDA runtime 12.8
device 0: NVIDIA GeForce RTX 3060 (sm_86), 28 SMs
VRAM: total=11920 MiB free=11795 MiB used=125 MiB
...
[       OK ] full_cap_vram (25 checks)

=== summary ===
checks: 256, failures: 0
RESULT: PASS
```

VRAM numbers from that run:

| Metric | Value |
|---|---:|
| Device total | 11,920 MiB |
| Free before arena | 11,795 MiB |
| Baseline used (context) | 125 MiB |
| Arena reserved | 7,244 MiB |
| Free after arena | 4,551 MiB |
| Device used after arena | 7,369 MiB |
| Device peak | **7,369 MiB** |
| Spare under 7,500 MiB cap | **130 MiB** |
| Free after release | 11,795 MiB |
| Recovery delta | 0 MiB |
| Over-cap request (7500 MiB + 1 B) | rejected, `kCapExceeded` |

The 12 GB dev box is a compatibility proxy: an actual 8 GB card (8192 MiB)
would have ~823 MiB left after the same 7369 MiB peak. Actual 8 GB hardware
remains pending.

A later run on the same box while other Vast.ai containers held 395 MiB
measured a device-wide peak of 7639 MiB; with 679 MiB resident it measured
7923 MiB. In both cases the arena delta was exactly 7244 MiB and all 25
`full_cap_vram` checks passed. That is why the test asserts the arena delta,
not the device-wide peak, when the GPU is shared.

## Notes and blockers

- No blockers for the integration agent. The API is deliberately small; the
  stream is available as `context.stream()` and converts implicitly to the
  `void*` argument used by `profile::CudaScope`.
- The box can be shared with other Vast.ai containers. Device-wide VRAM
  numbers include their contexts; the arena's own reservation and peak are
  deterministic (7244 MiB default).
- `ncu` counters remain blocked on this box (`ERR_NVGPUCTRPERM`); this layer
  uses no counters and is unaffected.
- Pinned host memory is allocated with `cudaMallocHost`; it is host RAM and does
  not consume device VRAM, but it must be released after all copies complete.
