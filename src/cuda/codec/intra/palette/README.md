# AV2 palette intra — CUDA

GPU per-sample half of AV2 v1.0.0 §5.20.8 / §7.13.4 palette prediction,
mirroring `src/codec/intra/palette/`.  The colour *selection* (histogram median
cut / exact search) stays on the CPU; the device owns the nearest-colour map,
edge replication, prediction and §5.20.8.5 colour contexts.

## API

```cpp
#include "cuda/codec/intra/palette/palette.h"
namespace cuda = av2enc::cuda;

cuda::PaletteDeviceBlock block;   // src/colors/map/dst + geometry
cuda::palette_build_map(block, stream);      // nearest colour + replication
cuda::palette_predict(block, stream);        // dst = colors[map]
cuda::palette_histogram(src, stride, w, h, hist, stream);
cuda::palette_color_contexts(ctx, stream);
```

Batched forms (`palette_build_map_batch`, `palette_predict_batch`,
`palette_color_contexts_batch`, `palette_apply_batch`) take device descriptor
arrays.  They launch a fixed 4x4x16x16 grid with `grid.z = count`; palette
blocks are at most 64x64 and each kernel returns for coordinates outside its own
block.  `palette_apply_batch` runs build-map → predict → optional contexts.
The arena workspace (`../workspace/`) stages host descriptors and calls the
three batch stages.

## Scope / contract

* 8-bit luma palette, block extents up to 64x64, palette size 2..8.
* `visible_w/h <= block_w/h`; the map is edge-replicated into the padding.
* Descriptors are POD and trivially copyable; device pointers must stay valid
  until the caller synchronizes.
* No allocations, no global state, no hidden device synchronization.

## Tests

```sh
cd src/cuda/codec/intra/palette
./build_and_test.sh                 # sm_86, needs nvcc + GPU
GPU_ARCH=sm_120 ./build_and_test.sh
```

Verified on the RTX 3060 (`sm_86`, CUDA 13.0):

```
palette_gpu_test: 200 checks, 0 failures
```

The workspace test adds a batch-vs-single-block equality check (map, prediction,
colour hash/order) over six block geometries, including a partial visible area
and all palette sizes exercised above.

## Integration

Palette is a screen-content tool: `allow_screen_content_tools` must be on and
`palette_mode_info()` must be written for every DC block that satisfies
§5.20.8.1 (see `docs/negative-results.md`).  `--still-gpu` applies
`palette_build_map`/`palette_predict` for a CPU-planned palette leaf (the colour
analysis and direction cost stay on the host, exactly as this component's scope
states); the syntax path stays on the CPU writer.
