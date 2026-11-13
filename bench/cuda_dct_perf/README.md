# CUDA DCT precision/performance benchmark (M0, RTX 3060)

Standalone benchmark for the 32x32 and 16x16 M0 forward DCT and quantization.
It lives entirely under `bench/cuda_dct_perf/` and is not wired into the root
CMake project. `src/`, the root `CMakeLists.txt`, and `tests/` are unchanged.

## Answers

**Is double-precision DCT viable on the 3060 at 1 fps? Yes, with large
headroom.** A 1920x1080 frame needs 2040 luma 32x32 transforms + 4080 chroma
16x16 transforms (two planes); M0's actual `k32x32` partition needs 1950 + 3840
+ 60 chroma 32x32. Batched over those blocks, the bit-exact double path takes
**6.79 ms/frame** (6.66 ms for the actual M0 mix): **147–150 fps maximum,
0.68 % of the 1 fps budget**. The transform is not the bottleneck; 993 ms of
the 1 s budget remain for intra prediction, entropy coding, and I/O.

The one caveat is launch granularity. One 32x32 transform per kernel launch
costs **65.5 µs** (low occupancy on one SM), so a serial 6120-launch path would
spend **~0.40 s/frame** — still under 1 s, but 40 % of the budget. Batch the
transforms into per-plane launches (the frame projection below does this).

**Would float need a different coefficient policy? Yes.** Pure float is not
bit-exact: it flips **3 quantized coefficients in a 1080p luma frame** (1 DC,
2 HF) and 0 in chroma on this content; 22/2,097,152 coefficients on uniform
random 32x32 blocks. A single flip changes the bitstream/MD5. Float is fine for
RD/mode search, but not for the normative coefficient path unless it gets a
guarded double fallback. The measured guard rate is small: only 0.022 % of real
1080p 32x32 coefficients are within 1e-3 of a rounding boundary (0.0023 % within
1e-4), so a float fast path plus a narrow double fallback is practical. Since
double is already fast enough at 1 fps, the simplest policy is to keep double
(or the bit-exact hybrid) for the coefficients.

**Integer/fixed point?** A Q20 row multiplier plus a Q60 quant multiplier is
bit-exact at the quantized-coefficient level on every 16x16 case tested and
nearly bit-exact on 32x32 (2 quantized flips in a 1080p luma frame, max
|Δq| = 1), at **19x the double throughput** (114.7 ns vs 2194 ns per 32x32
block). It is viable if FP64 throughput becomes a concern, but
it needs a guard for guaranteed bitstream equality. The 32x32 row multiplier
must be δ-corrected; without the correction the mismatch count rises from 37 to
3084 per 2.1 M random coefficients.

**Bit-exact hybrid.** An exact int32 row dot product followed by the CPU's
double scale and double column pass is bit-exact everywhere and takes
**3.52 ms/frame** (284 fps, 0.35 % of budget). This is the recommended
compromise if the integration wants to halve FP64 work without any guard.

## Hardware / software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB (development target) |
| Compute capability | 8.6 (`sm_86`), 28 SMs |
| Max SM / memory clock | 2100 MHz / 7501 MHz |
| Power limit | 110 W |
| Driver | 570.181, CUDA 12.8 |
| nvcc | 12.8.93 |
| Build | `-O3 -arch=sm_86 -lineinfo -fmad=false -Xptxas=-v` |
| Loaded SASS | `binaryVersion: 86` (sm_86 cubin executed) |
| During run | 100 % util, 1695–1942 MHz SM (mostly 1935–1942), 62–102 W, 50–55 °C |

Targets are the **RTX 3060 12 GB (dev)**, **RTX 3080 Ti (production)** and **RTX 5090 (max speed)**. This benchmark is
compute-bound and allocates <250 MB, so the 12 GB dev box is a valid compute
proxy for this kernel; no 12 GB-specific tuning is used. The 8 GB 3060 has the
same 28 SMs and FP64 rate (its memory bus is narrower, but the DCT is not
memory-bound). Allocations happen once at setup, outside the timed region; the
product path uses `src/cuda/arena.*`.

`nvidia-smi` query:

```text
NVIDIA GeForce RTX 3060, 8.6, 12288 MiB, 570.181, 2100 MHz, 7501 MHz, 110.00 W
```

```text
Tue Nov 10 21:17:04 2026
|   0  NVIDIA GeForce RTX 3060        On  |   00000000:08:00.0 Off |                  N/A |
|  0%   44C    P8             19W /  110W |      16MiB / 12288MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

## What is compared

The CPU reference is a direct transcription of
`src/pipeline/still/basic_encoder.cpp:941-969` (`forward_dct`) and `:1304-1310`
(quantization), using the normative `kDct32`/`kDct16` integer tables from
`src/pipeline/still/basic_tables.h`. The M0 forward DCT is **not** a floating cosine
basis: it is an integer kernel scaled by `1/(64*sqrt(n))`, accumulated in
double, then quantized with `lround(coeff * quant_factor)`.

| Variant | Arithmetic | Expected exactness |
|---|---|---|
| `double` | full FP64 row + column passes, `__dmul_rn`/`__dadd_rn`, `-fmad=false` | bit-exact |
| `float` | full FP32 row + column passes, `__fmul_rn`/`__fadd_rn` | approximate |
| `hybrid` | exact int32 row dot product, then the CPU's double scale + double column pass | bit-exact |
| `fixed` | Q20 integer row scale, int64 column, Q60 quant multiplier with δ correction | near-exact |

The double and hybrid kernels accumulate in the same order as the CPU loops
(`j = 0..n-1` for the row pass, `i = 0..n-1` for the column pass) and use
`llround`, matching `std::lround`. One CUDA block handles one transform; a warp
owns 1–8 output frequencies and lane `i`/`v` is the row/column, with a shared
memory transpose between passes.

### Fixed-point details

`row_mul = round(scale_w * 2^20)`; the row pass stores `sum * row_mul`, i.e.
`tmp * 2^20 * (1+δ)`. The final quant multiplier is

```text
quant_mul = ceil( qf * scale_h / (2^20 * (1+δ)) * 2^60 )
q         = round( column_int64 * quant_mul / 2^60 )
```

using a 64x64→128-bit multiply. The `ceil` (rather than round) matters for
16x16: coefficients there are dyadic rationals that hit exact `lround` ties
often, and `ceil` matches the away-from-zero tie direction. The δ correction
removes the systematic gain error of `row_mul`; for 16x16 `scale_w = 1/256` is
exact, so δ = 0.

| Size | scale_w | quant_factor | dq_denom | row_mul | δ | quant_mul |
|---|---:|---:|---:|---:|---:|---:|
| 32x32 | 0.002762135864009951 | 0.12121212121212119 | 2 | 2896 | -1.06817e-4 | 368160599 |
| 16x16 | 0.00390625 | 0.12121212121212122 | 1 | 4096 | 0 | 520602097 |

The int32 row store is sized for M0 residuals in `[-255, 255]`: the worst-case
row sum is 255 × 2048 = 522,240 and `522240 × 2896 = 1.512e9` fits int32. It is
not safe for arbitrary `int16` input without widening `s_t`.

## Accuracy vs CPU

"coeff bit-exact" means the coefficient value equals the CPU double reference
exactly (bitwise `memcmp` for double/hybrid; for float, the float value
converted to double equals the reference). "q mismatch (DC/LF/HF)" counts
quantized coefficients that differ; LF is `1 <= row+col < 4`, HF is
`row+col >= 4`. "near boundary" is float-only: coefficients within
1e-3/1e-4/1e-5 of a `k+0.5` rounding boundary, i.e. the double-fallback rate a
guarded float path would need. "recon PSNR" compares the reconstructed
residual against the CPU reconstruction over all pixels.

### 32x32, uniform random residual (2048 blocks)

| variant | coeff bit-exact | max \|Δcoeff\| | RMS Δcoeff | max rel | q mismatch (DC/LF/HF) | blocks w/ q mismatch | max \|Δq\| | near boundary ≤1e-3/≤1e-4/≤1e-5 | recon PSNR | max recon Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| double | 2097152/2097152 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| float | 68193/2097152 | 0.000191 | 1.76e-05 | 6.75e-05 | 22 (2/1/19) | 22 | 1 | 4107/415/83 | 76.97 | 1 |
| hybrid | 2097152/2097152 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| fixed | n/a | n/a | n/a | n/a | 37 (2/1/34) | 36 | 1 | n/a | 74.84 | 1 |
| fixed (no δ fix) | n/a | n/a | n/a | n/a | 3084 (1/33/3050) | 1580 | 1 | n/a | 57.72 | 2 |

### 16x16, uniform random residual (4096 blocks)

| variant | coeff bit-exact | max \|Δcoeff\| | RMS Δcoeff | max rel | q mismatch (DC/LF/HF) | blocks w/ q mismatch | max \|Δq\| | near boundary ≤1e-3/≤1e-4/≤1e-5 | recon PSNR | max recon Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| double | 1048576/1048576 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| float | 989194/1048576 | 0.000168 | 5.2e-06 | 1.53e-05 | 0 (0/0/0) | 0 | 0 | 2184/319/176 | inf | 0 |
| hybrid | 1048576/1048576 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| fixed | n/a | n/a | n/a | n/a | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| fixed (no δ fix) | n/a | n/a | n/a | n/a | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |

### 32x32, shine 1080p luma, block-mean predictor (2040 blocks)

| variant | coeff bit-exact | max \|Δcoeff\| | RMS Δcoeff | max rel | q mismatch (DC/LF/HF) | blocks w/ q mismatch | max \|Δq\| | near boundary ≤1e-3/≤1e-4/≤1e-5 | recon PSNR | max recon Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| double | 2088960/2088960 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| float | 1530495/2088960 | 0.000488 | 1.64e-06 | 8.2e-05 | 3 (1/0/2) | 3 | 1 | 461/48/12 | 82.08 | 1 |
| hybrid | 2088960/2088960 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| fixed | n/a | n/a | n/a | n/a | 2 (0/0/2) | 2 | 1 | n/a | 83.97 | 1 |
| fixed (no δ fix) | n/a | n/a | n/a | n/a | 44 (1/6/37) | 40 | 1 | n/a | 71.63 | 1 |

### 16x16, shine 1080p chroma, block-mean predictor (4080 blocks)

| variant | coeff bit-exact | max \|Δcoeff\| | RMS Δcoeff | max rel | q mismatch (DC/LF/HF) | blocks w/ q mismatch | max \|Δq\| | near boundary ≤1e-3/≤1e-4/≤1e-5 | recon PSNR | max recon Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| double | 1044480/1044480 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| float | 1044480/1044480 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | 154/27/19 | inf | 0 |
| hybrid | 1044480/1044480 | 0 | 0 | 0 | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| fixed | n/a | n/a | n/a | n/a | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |
| fixed (no δ fix) | n/a | n/a | n/a | n/a | 0 (0/0/0) | 0 | 0 | n/a | inf | 0 |

Double and hybrid are bit-exact on all four data sets (2,097,152 + 1,048,576
+ 2,088,960 + 1,044,480 coefficients, plus every quantized coefficient). The
real-frame residuals use a block-mean DC predictor, not M0's reconstructed
neighbour predictor; the exact mismatch counts are therefore content- and
predictor-dependent, but the double/hybrid exactness result does not depend on
the residual distribution.

## Throughput

One CUDA block per transform, q output only. Each 32x32 launch processes 16,384
blocks; each 16x16 launch processes 65,536 blocks. `q` is what the entropy
coder needs; storing the unquantized coefficients too (`coeff+q`) changed the
best-config time by <1 % for double/hybrid and <10 % for float (within
run-to-run noise), so the q-only column is the relevant encoder path.

The box is shared. These are the **best observed** values across repeated runs
(each row is the minimum of 7 event-timed samples, then the minimum across 5
full runs); transient contention from another tenant produced occasional 2x
outliers that the minimum rejects. The frame projection below is stable to
<0.5 % across runs.

| size | variant | threads | ms/launch | ns/block | Mblocks/s | GFLOP/s* |
|---|---|---:|---:|---:|---:|---:|
| 32x32 | double | 128 | 35.9502 | 2194.2 | 0.46 | 59.7 |
| 32x32 | double | 256 | 36.8774 | 2250.8 | 0.44 | 58.2 |
| 32x32 | double | 1024 | 41.6009 | 2539.1 | 0.39 | 51.6 |
| 32x32 | float | 128 | 1.4469 | 88.3 | 11.32 | 1484.2 |
| 32x32 | float | 256 | 1.5237 | 93.0 | 10.75 | 1409.4 |
| 32x32 | float | 1024 | 3.4353 | 209.7 | 4.77 | 625.1 |
| 32x32 | hybrid | 128 | 18.3468 | 1119.8 | 0.89 | 117.0 |
| 32x32 | hybrid | 256 | 18.3768 | 1121.6 | 0.89 | 116.9 |
| 32x32 | hybrid | 1024 | 24.0488 | 1467.8 | 0.68 | 89.3 |
| 32x32 | fixed | 128 | 1.8793 | 114.7 | 8.72 | 1142.7 |
| 32x32 | fixed | 256 | 2.5572 | 156.1 | 6.41 | 839.8 |
| 32x32 | fixed | 1024 | 7.3816 | 450.5 | 2.22 | 290.9 |
| 16x16 | double | 64 | 36.7347 | 560.5 | 1.78 | 29.2 |
| 16x16 | double | 128 | 37.3083 | 569.3 | 1.76 | 28.8 |
| 16x16 | double | 256 | 38.8067 | 592.1 | 1.69 | 27.7 |
| 16x16 | float | 64 | 1.4877 | 22.7 | 44.05 | 721.7 |
| 16x16 | float | 128 | 1.6076 | 24.5 | 40.77 | 667.9 |
| 16x16 | float | 256 | 1.7529 | 26.7 | 37.39 | 612.6 |
| 16x16 | hybrid | 64 | 19.4426 | 296.7 | 3.37 | 55.2 |
| 16x16 | hybrid | 128 | 19.4157 | 296.3 | 3.38 | 55.3 |
| 16x16 | hybrid | 256 | 19.4515 | 296.8 | 3.37 | 55.2 |
| 16x16 | fixed | 64 | 1.6737 | 25.5 | 39.16 | 641.5 |
| 16x16 | fixed | 128 | 1.8440 | 28.1 | 35.54 | 582.3 |
| 16x16 | fixed | 256 | 2.2513 | 34.4 | 29.11 | 476.9 |

\* GFLOP/s counts 2 passes × n² outputs × n mul-add × 2 FLOP. The fixed variant
uses integer ALU and its value is nominal. The RTX 3060's FP64 peak is
199 GFLOP/s (FMA) at the 1.78 GHz base clock; because bit-exactness forbids
contraction, the double path executes separate DMUL and DADD and its effective
peak is ~100–109 GFLOP/s (base–boost). The 32x32 double kernel reaches
~60 GFLOP/s, 55–60 % of that effective peak.

Resource usage of the winning kernels (`cuobjdump -res-usage`):

| kernel | threads | registers | shared | stack |
|---|---:|---:|---:|---:|
| double 32x32 | 128 | 57 | 12,288 B | 0 |
| double 16x16 | 64 | 57 | 3,072 B | 0 |
| float 32x32 | 128 | 40 | 8,192 B | 0 |
| hybrid 32x32 | 128 | 64 | 12,288 B | 0 |
| fixed 32x32 | 128 | 40 | 8,192 B | 0 |

SASS audit of the winning 32x32 double kernel (`cuobjdump -sass`): 71 DMUL,
48 DADD, **0 DFMA, 0 FMUL, 0 FADD, 0 FFMA**. The accumulation is FP64 and
uncontracted, as required.

## 1080p M0 projection

Block counts for 1920x1080, 4:2:0:

| Count | Nominal flat grid | M0 `k32x32` actual |
|---|---:|---:|
| Luma 32x32 | 60 × 34 = 2040 | 480 SB × 4 + 30 edge = 1950 |
| Chroma 16x16 | 60 × 34 × 2 = 4080 | 480 SB × 4 luma × 2 planes = 3840 |
| Chroma 32x32 | 0 | 30 edge SB × 2 planes = 60 |

The M0 edge row (1080 = 16×64 + 56) codes each 64x64 superblock with
`PARTITION_NONE`; luma is the adjusted 32x32 TX and chroma is one 32x32 TX per
plane. The nominal flat grid is the figure the task asks for; the actual M0 mix
is shown for completeness. Kernel time only, q output, one launch per size
(three launches for the actual M0 mix):

| variant | nominal frame (ms) | max fps | % of 1 fps budget | M0 actual frame (ms) | max fps | % of 1 fps budget |
|---|---:|---:|---:|---:|---:|---:|
| double | 6.7942 | 147 | 0.679 % | 6.6640 | 150 | 0.666 % |
| float | 0.2673 | 3742 | 0.027 % | 0.2662 | 3756 | 0.027 % |
| hybrid | 3.5195 | 284 | 0.352 % | 3.4519 | 290 | 0.345 % |
| fixed | 0.3358 | 2978 | 0.034 % | 0.3328 | 3005 | 0.033 % |

Mixed block throughput (nominal 6120 blocks): double 0.90 Mblocks/s, hybrid
1.74 Mblocks/s, float 22.9 Mblocks/s, fixed 18.2 Mblocks/s.

Single 32x32 double block, one kernel launch (includes launch overhead and
runs at one-SM occupancy): **65.5 µs**. A serial one-block-per-launch path over
6120 blocks is ~0.40 s/frame (~2.5 fps), still under 1 fps but 40 % of the
budget. This is an argument for batching, not against double.

CPU reference on one thread (`-O3 -ffp-contract=off`, preallocated buffers):
62–76 µs per 32x32 block and 8.3–9.8 µs per 16x16 block, i.e. **~160–196 ms**
for the 1080p transform mix. The host is shared, so the range reflects host
contention. GPU double is ~24–29x faster than this scalar reference.

## Relationship to the product transform path

`src/cuda/transform.cu` already contains a bit-exact double forward DCT
(thread-per-coefficient, one CUDA block per transform, `__dadd_rn`/`__dmul_rn`,
`-fmad=false`). This benchmark is an independent cross-check and performance
study, not a replacement. Its `scale_w`/`scale_h`/`quant_factor`/`dq_denom`
values use the same expression as `make_tx_params(32,32,true)` and
`make_tx_params(16,16,false)`:

| Size | scale_w | quant_factor | dq_denom |
|---|---:|---:|---:|
| 32x32 | 0.002762135864009951 | 0.12121212121212119 | 2 |
| 16x16 | 0.00390625 | 0.12121212121212122 | 1 |

The product path must keep using `make_tx_params()`, not a duplicated host
expression. The benchmark duplicates it only to remain standalone. The
benchmark's double kernel uses a warp-per-frequency layout (128 threads for
32x32), while the product kernel uses one thread per coefficient (1024 threads
for 32x32); the 1024-thread variant measured here is 2.54 µs/block vs 2.19 µs
for the 128-thread variant, i.e. ~7.6 ms/frame instead of 6.79 ms. The
feasibility conclusion is unchanged.

The product kernel's own measured timing (`src/cuda/README-transform.md`) is
5.506 µs per 32x32 forward DCT+quant and 0.688 µs per 16x16, so the nominal
1080p mix is `2040×5.506 + 4080×0.688 = 14.0 ms/frame` (~71 fps, 1.4 % of the
1 fps budget). Even the current unoptimized product kernel is comfortably
viable when batched. Its `1x 32x32` pipeline timing of 265.7 µs/iteration is a
warning about serial launches, not about the arithmetic: the API already takes
`nblocks` and must be used in batched form. A serial forward-only path at two
launches per block (~131 µs) would be ~0.80 s/frame, and the full
forward+inverse pipeline would exceed 1 s/frame.

The `docs/negative-results.md` entry *"Use `__fadd_rn` / `__fmul_rn` for
`double` DCT math"* was independently reproduced here: the first double build
matched **0/2,097,152** coefficients (max error 1.99e-4). After switching the
double and hybrid kernels to `__dadd_rn` (keeping `__dmul_rn`), the same test
matched **2,097,152/2,097,152** with zero quantized-coefficient differences.
This confirms the documented pitfall and the fix.

## Scope and constraints

- This directory only; `src/`, the root `CMakeLists.txt`, and `tests/` are
  untouched.
- No streams are produced, so the end-to-end `dav2d` / MD5 gate remains with the
  integration agent. The claim here is coefficient/quantization exactness
  against the CPU arithmetic.
- The fixed-point path assumes M0 residuals in `[-255, 255]`. Its int32 row
  store is sized for that range; arbitrary `int16` input can overflow it.
- Timings are best-of-N minima on a shared box (another tenant occasionally
  caused 2x outliers). The frame projection is stable to <0.5 % across runs.
- `cudaMalloc` is used only for benchmark setup, outside the timed region. The
  encoder must use the 7,500 MB arena.

## Exact commands

Local repo to the CUDA box (scoped tar pipe; the project standard is
`tools/sync_remote.sh`, but a scoped copy avoids touching other agents' files):

```sh
tar czf - -C bench cuda_dct_perf \
  | ssh -p <port> <user>@<host> \
      'tar xzf - -C /workspace/av2enc/bench'
```

Build and run on the box:

```sh
ssh -p <port> <user>@<host>
cd /workspace/av2enc/bench/cuda_dct_perf

cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build

./build/cuda_dct_perf \
  --y4m /workspace/av2enc/testdata/shine_1920x1080_10f.y4m
```

Or `sh run_remote.sh` from the same directory; it builds, prints `nvidia-smi`
and `nvcc --version`, then runs the 1080p case. `--quick` runs a shorter
accuracy set. The binary exits non-zero if double or hybrid is not bit-exact.
A `CUDA_DISABLE_PTX_JIT=1 ./build/cuda_dct_perf --quick` run also passes,
confirming the sm_86 cubin (not PTX JIT) is the executed path.

Clock/power logging during a run:

```sh
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv,noheader -l 1 > /tmp/dct_smi.csv &
./build/cuda_dct_perf --y4m /workspace/av2enc/testdata/shine_1920x1080_10f.y4m
kill %1
```

## Files

| Path | Purpose |
|---|---|
| `src/common.cuh` | block parameters, variant enum, launch API |
| `src/cpu_reference.{h,cpp}` | CPU transcription of M0 forward DCT/quant and inverse DCT |
| `src/kernels.cu` | double, float, hybrid, fixed kernels + dispatch |
| `src/main.cu` | accuracy harness, timing, 1080p projection, y4m loader |
| `CMakeLists.txt` | standalone sm_86 build, `-fmad=false -lineinfo` |
| `run_remote.sh` | build/run helper for the 3060 box |
