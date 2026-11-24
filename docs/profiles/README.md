# Flame graph — `--nightly-b`, one encode

Generated 2026-11-24 on the Apple M4 dev host from a real `sample` capture of
one encode, by `sample_to_flamegraph.py`.

| artifact | what |
|---|---|
| `nightly-b-24f-qp119-8t-work.html` | **work only** — idle pool waits removed |
| `nightly-b-24f-qp119-8t-all.html` | every thread, including the 84.5 % parked in `__psynch_cvwait` |

## Why HTML

The first version was a bare `.svg`. Opened as an image — which is how a
preview pane renders it — an SVG runs no script and shows no `<title>`
tooltips, so **nothing was clickable**. These are self-contained `.html`
files with inline SVG plus a small script:

- **hover** a frame: name, inclusive %, samples, self time
- **click** a frame: zoom to it; **click the background**: zoom out
- **search box**: highlights every frame whose name matches

No external references, so they work offline and from a `file://` URL.

## Recipe

```sh
# A -g build: `sample` resolves stacks to functions only with debug info.
cmake -S . -B /tmp/fg/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON \
      -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" \
      -DCMAKE_EXE_LINKER_FLAGS="-g"
cmake --build /tmp/fg/build -j8 --target av2enc

/tmp/fg/build/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/fg/out.ivf \
    --video --frames 24 --qp 119 --cpu-threads 8 --video-gops 3 --nightly-b &
PID=$!
sleep 3
sample $PID 18 1 -mayDie -file /tmp/fg/sample.txt
wait $PID

python3 docs/profiles/sample_to_flamegraph.py /tmp/fg/sample.txt out.html \
    --title "av2enc --nightly-b 24 f QP119 --cpu-threads 8 (all threads)"
```

The work-only variant drops stacks whose *self* frame is an idle wait
(`__psynch_cvwait` / `__psynch_cvsignal` / `__psynch_mutexwait`); see the note
at the end for the snippet.

## Reading it

`sample` counts are **inclusive**, and its call tree is a line drawing whose
characters the man page calls arbitrary — so the converter takes the *column
of the count* as the level, then derives self time as
`inclusive − Σchildren`. It also rejects an attachment that would over-count
its parent, because `sample` appends a flat "leaf subtree" summary after the
last thread that otherwise nests under it. A correct parse reports
**0 over-counted nodes**, which the converter prints so a bad parse is visible
instead of shipped. The committed capture reports 0.

## What it shows

**All threads** — 116,336 samples of self time. **84.5 %** is
`__psynch_cvwait`: pool workers parked in `condition_variable::wait` because
the joint-plan walk is serial and only the inner candidate loops use the pool.
That is wall-clock with 7 of 8 cores idle, not work.

**Work only** (14,046 samples): `IFE::joint_p_decide_sb` is the dominant frame
at 44 % of all work, and most of that is
`joint_p_measure_leaf → collect_node → encode_leaf → search_luma_tx`. The
entropy writer (`write()`) does not appear at all. That independently confirms
the built-in stage timers (`AV2ENC_RD_JOINT_P_PROFILE`) and shows why every
parallelism attempt failed: one deep serial chain with workers parked beside it.

## Caveats

- Sampling is statistical at 1 ms; frames thinner than ~0.1 % may be absent.
- `--frames 24` rather than the pinned 60 keeps the capture short; the stage
  shares match the 16 f profile, so the shape is representative.
- This host is the CPU correctness oracle, not the product target. A CUDA
  build would look completely different.
- The capture must come from a `-g` build; a release build resolves stacks to
  addresses only.
