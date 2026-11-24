# `--nightly-b` profile — where the work goes

One encode, Apple M4 dev host, sampled with `sample(1)` at 1 ms.
24 frames, QP119, `--cpu-threads 8`, ~16.2k samples per thread.

The capture was parsed with the column-of-count rule (see the recipe below);
the run reported **0 over-counted nodes**, which is the check that the tree was
read correctly.

## Totals

| | samples | note |
|---|---:|---|
| all threads | 116,336 | |
| **work only** | **13,954** | idle pool waits removed |
| idle (condvar wait + `__psynch_cvwait`/signal/mutex) | 102,382 | **88.0 %** of all samples |

88.0 % of raw samples are pool workers parked in
`condition_variable::wait`: the joint-plan walk is serial and only the inner
candidate loops use the pool. That is wall-clock with 7 of 8 cores idle, not
work — which is why every measurement below is on the **work-only** samples.

## Slowest functions by self time

| # | function | samples | share of work |
|---:|---|---:|---:|
| 1 | `txv::inverse_block` | 1,500 | 10.7 % |
| 2 | `txv::forward_block` | 1,449 | 10.4 % |
| 3 | `rdoq::Model::init` | 1,303 | 9.3 % |
| 4 | `rdoq::Model::coefficient_bits` | 1,285 | 9.2 % |
| 5 | `txsearch::` | 1,174 | 8.4 % |
| 6 | `rdoq::rdoq_quantize` | 793 | 5.7 % |
| 7 | `log2` | 734 | 5.3 % |
| 8 | `_platform_memset` | 463 | 3.3 % |
| 9 | `rdoq::Model::cost_symbol` | 402 | 2.9 % |
| 10 | `_platform_memmove` | 401 | 2.9 % |
| 11 | `txsearch::TxScratch::TxScratch` | 349 | 2.5 % |
| 12 | `rdoq::Model::eob_bits` | 318 | 2.3 % |
| 13 | `intra::` | 307 | 2.2 % |
| 14 | `???` | 282 | 2.0 % |
| 15 | `_xzm_xzone_malloc_tiny` | 255 | 1.8 % |
| 16 | `txv::downsample_residual` | 253 | 1.8 % |
| 17 | `eff_perc::weighted_sse` | 203 | 1.5 % |
| 18 | `txsearch::search` | 179 | 1.3 % |
| 19 | `mach_absolute_time` | 141 | 1.0 % |
| 20 | `rdoq::av2::build_scan` | 129 | 0.9 % |

**The single slowest function is `txv::inverse_block`** (10.7 %), with
`txv::forward_block` a close second. No function exceeds 11 % — the cost is
spread across the transform, the RDOQ trellis and the TX search.

## Grouped by subsystem

| group | samples | share |
| --- | ---: | ---: |
| RDOQ model + trellis | 4,266 | 30.6 % |
| transform pair | 2,949 | 21.1 % |
| other / unattributed | 1,830 | 13.1 % |
| tx search driver | 1,798 | 12.9 % |
| libc alloc / mem | 1,425 | 10.2 % |
| libm log2 | 734 | 5.3 % |
| intra prediction | 473 | 3.4 % |
| perceptual metric | 208 | 1.5 % |
| motion / MC | 194 | 1.4 % |
| exact rate interval | 64 | 0.5 % |
| syntax writer / entropy | 13 | 0.1 % |

Two readings worth keeping:

- **Entropy coding is not a cost.** The syntax writer is 0.1 % of work. Any
  effort aimed at the bit writer is aimed at nothing.
- **Half the encode is one call.** `txsearch::search` is **~51 % inclusive**,
  and inside it the cost splits roughly evenly between the RDOQ trellis (30.6 %)
  and the transform pair (21.1 %).

## Already mined from this

`rdoq::Model::init` (9.3 %) and `log2` (5.9 %) are both *cost of computing a
price*, not encoding, so removing them cannot change a decision. That produced
`b7da2505`: a table for the CDF symbol cost (bit-exact — CDF probabilities are
integers in [1, 32768]) and a cache for the diagonal scan by geometry. Net
**~1.7 %** on 16 f QP116 `--nightly-b` single-threaded CPU (median 105.85 ->
104.05 s, faster in 5/5 reps), streams byte-identical.

The remaining 51 % is real arithmetic (`txv::*` and the trellis DP). Cutting
that means computing less — i.e. changing decisions.

## Reproducing

```sh
# A -g build: `sample` resolves stacks to functions only with debug info.
cmake -S . -B /tmp/fg/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON \
      -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" \
      -DCMAKE_EXE_LINKER_FLAGS="-g"
cmake --build /tmp/fg/build -j8 --target av2enc

/tmp/fg/build/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/fg/out.ivf \
    --video --frames 24 --qp 119 --cpu-threads 8 --video-gops 3 --nightly-b &
PID=$!; sleep 3
sample $PID 18 1 -mayDie -file /tmp/fg/sample.txt
wait $PID

python3 docs/profiles/sample_to_table.py /tmp/fg/sample.txt
```

## Caveats

- Statistical at 1 ms; frames thinner than ~0.1 % may be absent.
- `--frames 24` rather than the pinned 60 keeps the capture short; the stage
  shares match the 16 f profile, so the shape is representative.
- This host is the CPU correctness oracle, not the product target. A CUDA
  build would look completely different.
- The 8-thread wall time is dominated by pool contention and host noise; quote
  single-threaded CPU numbers for any speed claim.
