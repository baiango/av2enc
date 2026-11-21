# RD joint plan — two-pass CPU batch (parallel plan search)

Status: **measured experiment.** Wall time 2.6-3.0×; the stream moves by
−0.057 K at one round and +0.030 K at two rounds. Not a default; the serial
raster walk stays the reference.
The batch is a research gate (`AV2ENC_RD_JOINT_BATCH`, default off); the serial
raster walk stays the reference. Branch `exp/joint-batch`, worktree
`../joint-batch`, from `exp/joint-plan-b` `3c033dfe` (both joint-plan phases).
No push, no merge.

## 0. Answer first

1. **The exact-rate serial head does not dominate.** On the 240p30 / 60 f
   fixture, `--video-rd-preset best`, 8 threads, the P-frame joint walk spends
   **0.17 %** of its wall time in `rate_interval::estimate` +
   `build_from_syntax`, **0.08 %** in the distortion measure, and **3.2 %** in
   the raster commit. **95.4 % is the per-candidate leaf encode**, and **76.9 %
   of the whole walk is the luma TX search** (`search_luma_tx`: TX
   size/type/partition plan + RDOQ + per-plan reconstruction). ME is
   **0.3 %**. The batchable fraction is **≈95 %**, so Amdahl caps it at
   `1/(0.05 + 0.95/N)`.
2. **A two-pass batch does cut the wall time**: 794.5 s → **266.3 s (2.98×)**
   on the flat chain at QP112, and 495.7 s → **191.1 s (2.59×)** on the
   chained-B carrier at QP122.
3. **It does not keep the stream for free.** At matched 100 kbps (flat chain):

| arm | speedup | K @ 100 | delta |
|---|---:|---:|---:|
| serial joint | 1.00× | 87.196 | — |
| batch, 1 round | 2.98× | 87.139 | **−0.057** |
| batch, 2 rounds | 1.62× | 87.226 | **+0.030** |

   One round is outside the ±0.05 gate; two rounds are inside it (and
   marginally positive) at a little over half the speedup. The only
   byte-identical configuration is a single pass-1 chunk, i.e. no parallelism
   (`--cpu-threads 1`), which is how the delta was localised to the chunk
   boundary rather than to a different code path.

## 1. Profile of the serial walk

Instrumented with `AV2ENC_RD_JOINT_P_PROFILE=1` (pure timers; the stream is
unchanged). 8 frames = 1 key + 7 P, QP112, `--cpu-threads 8`.

| stage | wall | share |
|---|---:|---:|
| `precompute_leaf_me` | 0.00 s | 0.0 % |
| `joint_p_decide_sb` (the search) | 89.38 s | 96.7 % |
| — reference-layout walk (`measure_ref`) | 2.99 s | 3.2 % |
| — prune (candidate layouts) | 86.37 s | 93.5 % |
| committed `collect_node` (raster commit) | 3.00 s | 3.2 % |
| **total** | **92.39 s** | |

Inside the 37,504 leaf trials:

| term | wall | share |
|---|---:|---:|
| leaf encode (`collect_node`) | 88.23 s | 95.4 % |
| — `search_luma_tx` (luma TX plan + RDOQ + recon) | **71.04 s** | **76.9 %** |
| — `compute_tx` (chroma / plain) | 1.16 s | 1.3 % |
| — `search_me` inside a trial | 0.30 s | 0.3 % |
| — remainder (MC, mode loop, chroma pred, recon writes) | 15.7 s | 17.0 % |
| exact interval rate (`estimate` + `build_from_syntax`) | 0.16 s | 0.17 % |
| distortion (`seq_rd::leaf_distortion_gains` + chroma SSE) | 0.07 s | 0.08 % |

Trial mix: **37,480 inter / 24 intra / 0 committed intra**, 21,455 of them
`skip=1`. 37,504 trials for 255 committed leaves is **147 candidates per
committed leaf** (167 per superblock). That amplification is why the walk is
slow: it runs the full TX search on every candidate.

ME is negligible because `precompute_leaf_me()` only covers the reference
tree's leaves (~36 per frame); inside the trials `search_me` runs 37,759 times
with a 2.8 % cache hit rate and still costs 0.3 %. `txtsearch::search` already
parallelises the TX *type* candidates, but only above 4 candidates and only
with a pool installed on the calling thread, so the walk stays on one core
(`sample` shows the pool workers idle for the whole frame).

**Batching the exact rate would parallelise 0.2 % of the wall.** The batch has
to target the leaf encode.

## 2. The solve

```text
pass 0 (serial, optional)   commit the reference layouts; record the frame
                            state every superblock starts from
pass 1 (parallel)           search each superblock's joint tree on a private
                            encoder clone, seeded with the frame state at the
                            superblock's chunk start, chained inside the chunk
pass 2 (serial, raster)     re-price the candidate and the reference layout
                            with the live exact objective, keep the better one,
                            commit through collect_node()
```

Nothing in pass 1 is committed: a clone's only output is a `PartitionNode`.
Every coded symbol still comes from the serial raster commit, so the writer's
reference-MV bank, the §6.19.2.3 maps and the exact-syntax probe advance in
coded order, and pass 2 evaluates the serial guard (`reference` kept unless the
exact objective prefers the candidate) on live prices. The batch can therefore
never commit a layout the exact objective ranks below the gate-off reference.

Engineering points that were needed to make it work:

* **Trial clone.** `joint_p_trial_clone()` is a second `InterFrameEncoder` over
  the same source/references whose frame-start decision state is copied in
  (`joint_p_copy_frame_state`: reconstruction planes, `intra_joint_`,
  `uv_smooth_`, `mv_grid_`, ME cache, QM state, quantiser). `plan_pass()`
  rebuilds the deterministic per-frame surfaces (eff-perc weights, BA mask,
  exact-syntax probe, §8.2 CDF image) from the source and the options.
* **Chunked chaining.** A worker takes a *contiguous* superblock range and
  commits each chosen layout into its own state before the next one
  (`AV2ENC_RD_JOINT_BATCH_CHAIN=1`). Only the first superblock of a chunk sees
  a state that is not the serial one. The chained commit must be seeded the way
  `collect_frame()` seeds its sink (`seed_unit_state`) — the first version
  omitted it, which silently priced every chained commit at qindex 0.
* **`--cpu-threads` owns the worker count.** `AV2ENC_RD_JOINT_BATCH=N` selects
  the batch; the workers are `min(N, --cpu-threads)`.
* **Counters.** Pass-1 workers accumulate into a thread-local throwaway
  `Counters`/`Profile` so the process-global totals stay race-free and describe
  only what the raster walk did.

## 3. Gates

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_JOINT_P` | off | the serial joint solve (unchanged) |
| `AV2ENC_RD_JOINT_BATCH` | 0 | 0 = serial walk; N > 0 = batch (capped by `--cpu-threads`) |
| `AV2ENC_RD_JOINT_BATCH_SEED` | 0 | 1 = pass 0 commits the reference layouts and seeds pass 1 from them |
| `AV2ENC_RD_JOINT_BATCH_CHAIN` | 0 | 1 = chain the committed state inside a contiguous chunk |
| `AV2ENC_RD_JOINT_BATCH_ROUNDS` | 1 | pass 1 / pass 2 iterations, re-seeded from the committed state |
| `AV2ENC_RD_JOINT_P_PROFILE` | off | stderr wall-clock split (§1) |

No CLI change; the serial path is `AV2ENC_RD_JOINT_BATCH=0`.

## 4. Flat P chain, `--video-rd-preset best`, 60 f

| arm | bytes | kbps | wall | speedup |
|---|---:|---:|---:|---:|
| serial joint (reference) | 25,007 | 100.028 | 794.5 s | 1.00× |
| batch chain, 1 round | 25,515 | 102.060 | **266.3 s** | **2.98×** |
| batch chain, 2 rounds | 25,047 | 100.188 | 490.7 s | 1.62× |
| batch chain, 1 chunk (`--cpu-threads 1`) | **25,007** | — | 805.1 s | 0.99× |

Oracle at matched 100 kbps (`tools/oracle.py --target-kbps 100`):

| arm | points (QP: kbps / K) | K @ 100 | delta |
|---|---|---:|---:|
| serial joint | 112: 100.028 / 87.196 | **87.196** | — |
| batch chain 1 round | 112: 102.060 / 87.275, 114: 98.796 / 87.060 | **87.139** | **−0.057** |
| batch chain 2 rounds | 112: 100.188 / 87.251, 114: 97.184 / 86.849 | **87.226** | **+0.030** |

The fully-chained arm (`--cpu-threads 1`, one chunk) is **byte-identical** to
the serial walk (`9a36aeb7…`, 25,007 B), which localises the whole delta to the
chunk-boundary state, not to a code-path difference.

## 5. Chained-B carrier (`--nightly-b` + joint envs), 60 f

| QP | arm | bytes | wall | speedup |
|---:|---|---:|---:|---:|
| 122 | serial joint | 23,686 | 495.7 s | 1.00× |
| 122 | batch chain 1 round | **23,631** (−0.23 %) | **191.1 s** | **2.59×** |
| 122 | batch no-chain 2 rounds | 23,696 | 320.3 s | 1.55× |
| 119 | serial joint | 24,845 | 462.0 s | 1.00× |
| 119 | batch chain 1 round | 24,848 (+0.01 %) | 195.0 s | 2.37× |

Oracle at QP122: serial PSNR-Y 47.141 / SS2 85.881 / K 87.322; batch 47.167 /
85.865 / K 87.218. The batch point is at a 0.19 % *lower* rate, so the fixed-QP
K column is not the matched-rate reading; the fixed-QP bytes are neutral to
slightly smaller on this carrier.

## 6. Scaling and the pass split

16 frames (1 key + 15 P), flat QP112, `--video-rd-preset best`. `--cpu-threads`
also caps the batch workers, so `t` is the real thread count.

| threads | serial joint | batch chain 1 round | speedup |
|---:|---:|---:|---:|
| 1 | 161.3 s | 206.9 s | 0.78× |
| 2 | — | 128.0 s | 1.63× |
| 4 | — | 76.4 s | 2.11× |
| 8 | 185.8 s | 57.6 s | 2.80× |

The batch at one thread is *slower* than the serial walk (0.78×): pass 0, the
pass-2 re-price and the clone construction are pure overhead when there is
nothing to overlap them with. Fitting `W(N) = S + P/N` over the four batch
points gives **P = 170.6 s (82.5 %) / S = 36.3 s (17.5 %)**, i.e. the batch
parallelises 82.5 % of its own wall and captures essentially all of it (the
host has 4 performance + 6 efficiency cores, ~4.2 cores of real throughput).

Instrumented split at 8 threads, 16 f:

| stage | batch | serial |
|---|---:|---:|
| pass 1 (parallel search) | 38.6 s | — |
| pass 2 (exact re-price + raster commit) | 10.8 s | — |
| pass 0 (reference commit) + bookkeeping | 6.3 s | — |
| joint walk total | **55.7 s** | **170.0 s** |
| — reference-layout walk (`measure_ref`) | — | 6.2 s |
| — prune (candidate layouts) | — | 163.8 s |
| raster commit (`collect_node`) | (inside pass 2) | 6.2 s |
| whole encode | **65.7 s** | **185.8 s** |

Pass 1 alone is 4.4× on 8 threads. Pass 2 is 6.3 % of the serial walk (three
layouts per superblock — candidate, reference, commit — against 113 measured
layouts), which is the measured floor the batched design pays.

The 16 f serial profile reproduces the 8 f shares: TX search 123.5 s of
176.3 s (**70 %**), exact rate 0.46 s (**0.26 %**), commit 3.5 %.

## 7. Why the stream moves

The design note called inter leaves "reference-independent". The *prediction*
is; the *price* is not:

* `bawp_adapt_luma()` fits its alpha from `recon_` neighbours, and BAWP is on
  in both recipes;
* an intra or inter-intra candidate inside a P leaf reads the reconstruction,
  the `intra_joint_` smooth flags and the §6.19.2.3 `BlockDecoded` map;
* the merge/DRV stack and DRL read `mv_grid_`;
* the exact inter-syntax group (`pipe_probe_.price_full`) reads the probe's
  skip/mode/compound/interp-filter neighbour grids, which advance in coded
  order.

The residual rate really is position-independent (`rate_interval_cdfs_` is a
per-frame snapshot), which is why the exact-rate term is 0.17 % — but that is
also why batching it would buy nothing. What is position-dependent is the leaf
*encode*, and that is where the 95 % is.

Measured consequence: the batch agrees with the serial walk for the first 32
pictures of the flat chain (per-frame `frame bits` identical through display
ordinal 31) and diverges from ordinal 32, and the divergence is
chunk-granular — one chunk is byte-identical, eight chunks are not. On the
chained-B carrier the same effect is inside ±0.25 %.

## 8. Conformance

* Gate-off pins are byte-identical: QP106 **25,008 / `0639b43630abb454f7be5ac494d2efb4`**,
  QP110 **23,690 / `1b99127a4cd1d386fa5600fb63c57a83`** (`--fast-decode`, 60 f).
* Every flat arm decodes under `tools/validate_dav2d.sh` and `dav2d --strict 1`
  (serial, batch chain 1 round, batch 2 rounds, single-chunk).
* Every chained-B arm (`tools/validate_avmdec.py`) reports **60/60 OK**: QP122
  and QP119, serial and batch.
* The single-chunk batch stream is byte-identical to the serial one
  (`9a36aeb7…`), which is the control that the batch changes the search, not
  the coder.
* The serial joint walk is unchanged: `AV2ENC_RD_JOINT_BATCH=0` reproduces
  25,007 B / `9a36aeb7c02733342e4ec5db300e4d6d` and 17,797 B /
  `ff44611d6d8c9595421a49f0a31dc596` (8 f smoke), matching the lane's published
  joint QP112 reading.
* No CLI, `CMakeLists.txt`, `main.cpp`, options/parse, preset or pin edit.

## 9. Verdict

* **The profile is the transferable result**: the joint solve is a TX-search
  problem (77 %), not an entropy problem (0.2 %). Any future speed work should
  start there — a GPU/vectorised TX candidate batch, or a candidate-pruning
  search that does not run the full TX plan 147 times per committed leaf.
* **The two-pass CPU batch is worth 2.6-3.0×** and captures essentially all of
  the available parallelism on this host (see §6). It is the right shape for
  exploration runs (lambda sweeps, tree studies) where a 2 % rate excursion is
  affordable.
* **It is not a stream-exact replacement**: −0.057 K on the flat chain at
  matched 100 kbps, neutral-to-better on the chained-B carrier. Two rounds get
  the flat chain to +0.16 % bytes at 1.62×. Shipping it as the default solve
  would need the user's call on the K trade.
* Next step if exactness plus speed is required: keep the raster recurrence but
  make the chunk starts exact — e.g. a sequential fix-up that re-runs only the
  chunks after the first committed change, which turns the measured
  chunk-granular drift into a bounded retry instead of a whole extra round.

## 10. Files

| file | role |
|---|---|
| `src/pipeline/video/video_joint_plan.h` | batch/seed/chain/rounds config, `Profile`, thread-local worker counters |
| `src/pipeline/video/video_frame_decision.cpp` | `joint_p_batch_frame`, `joint_p_measure_tree`, `joint_p_capture`/`apply`, `joint_p_same_layout`, `joint_p_trial_clone`, the pass hook, profile timers |
| `src/pipeline/video/video_frame_internal.h` | declarations, timers in `search_luma_tx`/`compute_tx` |
| `src/pipeline/video/video_frame_me.cpp` | ME call/cache counters |

## 11. Thread safety of the pass-1 workers

Pass 1 builds one `InterFrameEncoder` clone per chunk and measures leaves on
`parallel_for` workers. The leaf coder was written for the serial walk and
updates process-global diagnostic statics, so the batch is also a TSan audit.
Four races were real (not just noise):

| # | object | writer | fix |
|---|---|---|---|
| 1 | `joint_plan::profile()::value` | `ScopedTimer` in `compute_tx`, called from the partition probe's own parallel workers | `profile_mutex()` + `profile_bump()`; the destructor returns early when profiling is off, so the default path never touches the static |
| 2 | `rd_driver::cli_flag()`, `eff_perc::cli_flag()`, `tail_tx::cli_mode()` | every `InterFrameEncoder` constructor; clone construction runs concurrently | `std::atomic` (`store`, `fetch_or`) |
| 3 | `tail_tx::counters()` | leaf trials (`encode_leaf`, `video_frame_decision.cpp:4020`) | atomic counters, `.load()` in the exit print |
| 4 | `rd_driver::measuring_flag()` | `joint_p_decide_sb` on every worker | `thread_local`: the flag describes the measuring thread, not the process |

`joint_plan::Counters` was already isolated: pass 1 sets
`joint_plan::trial_worker() = true` before the first leaf, and `counters()`
returns a `thread_local` silent copy on those threads. That copy is now
constructed silent, so it cannot print an empty duplicate stats line at exit.

Race 1 predates the batch: it fires in the default `--fast-decode` partition
probe whenever a joint-plan build runs under TSan. The CI failure was hidden
because `tools/ci_encode_smoke.sh` printed the encoder's stderr with
`tail -10 <file1> <file2>`, which GNU `tail` rejects, so the log was empty; it
now uses `tail -n 10`.

Evidence, `clang++ -fsanitize=thread`, `halt_on_error=1`, 8 threads, 320x240:

| run | result |
|---|---|
| trace smoke (`tools/ci_encode_smoke.sh`, default preset) | ok, 0 warnings |
| serial joint, 3 frames, `AV2ENC_RD_JOINT_P=1` | 0 warnings |
| batch joint, 3 frames, `BATCH=8 ROUNDS=1` | 0 warnings |
| batch joint, 8 frames, `BATCH=8 ROUNDS=2` | 0 warnings |

The fixes are stats/flag plumbing only. Gate-off pins (QP106 25,008 /
`0639b436`, QP110 23,690 / `1b99127a`) and the 16-frame chained-B stream
(19,723 / `505430de`) are byte-identical after them.
