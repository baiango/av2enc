# VRAM budget and precompute planning

## Focus

| Item | Status |
|---|---|
| **Development target** | **RTX 3060 12 GB, sm_86** |
| **Production target** | **RTX 3080 Ti 12 GB, sm_86** |
| **Maximum-speed target** | **RTX 5090 32 GB, sm_120** |
| **Dev box** | 12 GB RTX 3060, arena sized from the device |
| 12 GB behaviour | **Not an 8 GB emulation; actual 8 GB hardware pending** |
| RTX 4070 Super / RTX 5070 | Not targets |
| 4 / 6 / 16 / 24 GB capacity profiles | Estimate-only; the encoder does not select one |
| Turing sm_75 | Out of scope; not supported by the current sm_86;sm_120 fatbin |

There is no fixed VRAM ceiling. At first GPU use the encoder reserves
`min(device_total, device_free) - 512 MiB`, optionally bounded by
`--vram-cap <MiB>`. On an 8 GB card this yields a sub-7,500 MiB arena
automatically. The 12 GB dev box therefore runs at its own capacity; 8 GB
compatibility must be validated on an 8192 MiB enumerated card.

## RTX 3060 8 GB vs 12 GB

Both are GA106, sm_86, 28 SMs. Only capacity and bandwidth differ:

| Property | RTX 3060 8 GB (target) | RTX 3060 12 GB (dev box) |
|---|---|---|
| VRAM | 8 GB class | 11920 MiB reported by CUDA |
| Bus / bandwidth | 128-bit, ≈240 GB/s | 192-bit, ≈360 GB/s (measured 360.0) |
| Compute | 28 SMs, sm_86 | 28 SMs, sm_86 |
| Encoder arena | device-sized, ≤ physical | device-sized, ≤ physical |

The 8 GB card is not physically attached. Every "8 GB" result in this document
is a model or a 12 GB proxy, **actual 8 GB hardware pending**, unless explicitly
labelled otherwise.

## Frame sizes (4:2:0, uncompressed)

`frame_bytes = width × height × 1.5 × (bit_depth > 8 ? 2 : 1)`.

| Format | 4:2:0 samples | 8-bit frame | 10-bit frame |
|---|---:|---:|---:|
| 1080p (1920×1080) | 3,110,400 | 2.97 MiB | 5.93 MiB |
| 1440p (2560×1440) | 5,529,600 | 5.27 MiB | 10.55 MiB |
| 4K (3840×2160) | 12,441,600 | 11.87 MiB | **23.73 MiB** (4K10) |
| 8K (7680×4320) | 49,766,400 | 47.46 MiB | **94.92 MiB** (8K10) |

## Arena split

```text
persistent_bytes  = (refs + lookahead + 2) * frame_bytes + state_bytes
                    (+2 = current source and reconstruction)
state_bytes       = max(1 MiB, 64 B per 16x16 block)
scratch_per_tile  = tile_pixels * 24 B + 2 MiB
scratch_bytes     = tiles * scratch_per_tile
candidate_bytes   = frame_bytes * rdo_multiplier
headroom_bytes    = 512 MiB (planning reserve)
total             = persistent + scratch + candidate + headroom
```

- **Persistent** — allocated once at startup: reference frames, lookahead
  source frames, current source/reconstruction, and per-block state. Held for
  the session.
- **Reusable scratch** — one workspace reused across tiles and frames:
  transform/quantization, intra search, SAD/SATD, motion-search windows. Sized
  for the worst-case tile batch, not for every frame. The 24 B/pixel and 2
  MiB/tile constants are conservative planning values.
- **RDO candidate cache** — `1×` a raw frame for `compat8g`.
- **Headroom** — 512 MiB in the estimator. `vram_budget` uses 256 MiB direct
  arena headroom because it measures the allocation itself.

## Planning example: 8 GB class

| Parameter | Value |
|---|---:|
| Arena | device-sized (≈7.4 GiB on an 8 GB card) |
| Lookahead | 4 |
| Reference frames | 4 |
| Tiles | 4×2 (8) |
| RDO candidate cache | 1× |
| Motion-search radius | ±64 (tunable starting point) |
| Recommendation | 1080p / 4K light |

This is a sizing example for an 8 GB card, not a runtime preset. The search
radius is a quality/latency starting point and is not part of the byte estimate.

Example estimates:

| Input | Persistent | Scratch | Candidate | Headroom | Total |
|---|---:|---:|---:|---:|---:|
| 1080p8 (1920×1080) | 15.83 MiB | 55.46 MiB | 2.97 MiB | 512 MiB | **586.26 MiB** |
| 4K10 (3840×2160) | 239.28 MiB | 205.84 MiB | 23.73 MiB | 512 MiB | **980.86 MiB** |

These are model estimates, not measured allocations.

## Measured: historical `compat8g` benchmark

This `bench/vram_budget` run predates the device-derived arena. It is kept as a
capacity-planning data point, not as the current encoder policy. It was run on
the 12 GB RTX 3060 with `VRAM_LIMIT_MB=7500`:

```text
=== av2enc VRAM budget check ===
device: NVIDIA GeForce RTX 3060 (sm_86)
total: 11920 MiB (12499812352 B)
before:          total=11920 MiB  free=11795 MiB  used=125 MiB
config: VRAM_LIMIT_MB=7500, headroom=256 MiB, safety=64 MiB
arena plan: 7244 MiB (limit - headroom)
after alloc:    free=4551 MiB  used=7369 MiB
after touch:    free=4551 MiB  used=7369 MiB
budget check (cap=7500 MiB):
  used including arena: 7369 MiB, remaining under cap: 130 MiB
  reserve 512 MiB -> REJECTED (cap)
  cudaMalloc 4807 MiB (> free 4551 MiB) -> cudaErrorMemoryAllocation (clean)
after free:     free=11795 MiB  used=125 MiB
summary:
  arena: 7244 MiB
  peak used: 7369 MiB (cap 7500 MiB, spare 130 MiB)
  touch verify: PASS
  cap rejection: PASS
  cuda over-allocation: PASS
  VRAM recovery: PASS (0 MiB delta)
  peak within cap: PASS
RESULT: PASS
```

This is the **only** allocation result being reported as measured:

- Total: **11920 MiB**
- Free before: **11795 MiB** (used 125 MiB baseline)
- Arena: **7244 MiB**, fully touched
- Free after allocation/touch: **4551 MiB**
- Peak used: **7369 MiB**, spare under cap: **130 MiB**
- Free after release: **11795 MiB** (0 MiB delta)
- 512 MiB over-cap request: **rejected**
- Actual 4807 MiB `cudaMalloc` beyond free: **clean `cudaErrorMemoryAllocation`**
- 12 GB acting like an 8 GB card: **12 GB emulation, actual 8 GB hardware
  pending.** Process peak is 7369 MiB, so ~4551 MiB of the 12 GB card is left
  unused. This is a process-level budget, not a hardware partition; other
  processes can still use that VRAM. On a real 8 GB card, the same peak would
  leave ~823 MiB for the driver, context, and small allocations.

The selector with the required cap on the real 12 GB dev box:

```text
$ ./build/vram_config --width 3840 --height 2160 --bit-depth 10 --cap-mb 7500
using:    total=11920 MiB, free=11795 MiB, bandwidth=360.0 GB/s
  compat8g SELECTED (cap=7500 MiB)
profile: compat8g (cap=7500 MiB, refs=4, lookahead=4,
         tiles=4x2 (8 total), rdo=1x, search=+/-64)
  estimate total: 980.86 MiB (1028502528 B)
  cap: 7500 MiB -> fits (6519.14 MiB spare)
  physical total: 11920 MiB -> fits
  available: 11795 MiB -> fits
RESULT: OK - profile 'compat8g' with cap 7500 MiB
```

## Non-target profile notes (estimates only, out of scope)

The estimator still contains these for reference, but they are not tuned or
validated and must not be presented as hardware results:

| Profile | Cap | Lookahead | Refs | Tiles | RDO | Search | Note |
|---|---:|---:|---:|---|---:|---:|---|
| `compat4g` | 3500 | 1 | 2 | 2×2 | 1× | ±32 | 4 GB estimate, out of scope |
| `compat6g` | 5500 | 2 | 3 | 2×2/4×2 | 1× | ±48 | 6 GB estimate, out of scope |
| `auto12g` | 11000 | 12 | 8 | 8×2 | 2× | ±96 | 12 GB estimate, out of scope |
| `auto16g` | 15000 | 16 | 12 | 8×2/16×2 | 3× | ±112 | 16 GB estimate, out of scope |
| `auto24g` | 20000 | 24 | 16 | 8×2/16×2 | 4× | ±128 | 24 GB estimate, out of scope |

`auto`/`compat*` names are accepted by the offline estimator for planning.
They are not runtime presets; the encoder always sizes from the device.

## sm_75 caveat (out of scope)

The 4/6 GB capacity notes are not an sm_75 promise. Turing cards in that class
(GTX 1650/1660, RTX 2060) are sm_75; the current fatbin is **sm_86;sm_120**, and
`cp.async` needs sm_80+. An sm_75 target would require a separate build, a
no-`cp.async` fallback, and a real sm_75 host. The estimator refuses sm_75:

```text
$ ./build/vram_config ... --cc 7.5
compute capability: 7.5 (SIMULATED); current fatbin: sm_86;sm_120
RESULT: FAIL - compute capability 7.5 (sm_75) is not supported ...
sm_75 (GTX 1650/1660, RTX 2060-class) requires a separate sm_75 build and a
no-cp.async fallback; the estimator refuses to select a precompute profile.
```

## Configuration rules

1. Estimate first: query `cudaMemGetInfo`, device total, bandwidth, and compute
   capability.
2. Reserve `min(total, free) - 512 MiB`; an optional `--vram-cap` may lower it.
   Do not hard-code a card-specific ceiling.
3. If `persistent + scratch + candidate + headroom` exceeds the available
   capacity, reduce tile batch → lookahead → refs → RDO cache, then fail.
4. Never silently over-allocate. Fail with a non-zero exit and a clear message.
5. Allocate the persistent arena once; reuse one scratch workspace. No
   `cudaMalloc` in per-frame or per-block paths.
6. `ncu` counters may be unavailable; `nsys` tracing and the budget checks still
   work. This is a process-level cap, not MIG (the RTX 3060 does not support MIG).

## Exact commands

```sh
# Build
cd /workspace/av2enc/bench/vram_budget
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

cd /workspace/av2enc/bench/vram_config
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Active configuration: 7500 MiB cap on the 12 GB dev box
cd /workspace/av2enc/bench/vram_budget
VRAM_LIMIT_MB=7500 ./build/vram_budget

cd /workspace/av2enc/bench/vram_config
./build/vram_config --width 3840 --height 2160 --bit-depth 10 --cap-mb 7500
# expect: compat8g SELECTED, estimate 980.86 MiB, RESULT: OK

# Smoke checks only
./build/vram_config --width 1920 --height 1080 --bit-depth 8 --cap-mb 7500
./build/vram_config --width 1920 --height 1080 --bit-depth 8 --cap-mb 7500 --cc 7.5
# last command exits 2: sm_75 unsupported
```

## Measured vs estimated

| Claim | Basis |
|---|---|
| 12 GB 3060 total/free/arena/peak | **Measured** by `vram_budget` |
| 7500 MiB cap rejects over-cap requests | **Measured** on the 12 GB 3060 |
| 12 GB card acts like an 8 GB card | **12 GB emulation, actual 8 GB hardware pending** |
| 8 GB SKU, 4/6/16/24 GB profiles | **Estimates only**; no such hardware tested |
| Byte breakdowns / selector estimates | **Model estimates**, not allocations |
| RTX 4070 Super / RTX 5070 | Not targets |
| sm_75 Turing | Out of scope; unsupported |
