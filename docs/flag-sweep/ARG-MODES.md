# The INERT argument options, re-judged per mode + the pipe-budget arming chain

## A. The 24 INERT argument-option entries (16 distinct options)

Re-run on the plain video baseline (`--frames 2`, 5,475 B) and the still
baseline (4,198 B / `5798ed59`):

| verdict | entries |
|---|---:|
| **LIVE** | **4** |
| inert in both | 20 |

Newly LIVE: `--video-cdef-strength 1`, `--video-ccso-unit-bits 2`,
`--video-key-qp-offset 1`, `--video-key-qp-offset 3`.

### `--p-frames` was shadowed by `--frames`, not inert

`p_frames` does not decide anything in the coder -- it sets the **frame limit**,
and it sits behind an `else if` in `main.cpp` (1046/1442/1636/2478):

```cpp
} else if (options.p_frames != 0) {
  limit = static_cast<std::size_t>(options.p_frames) + 1U;
```

Every sweep passes `--frames 12`, so `--p-frames` was never consulted. Without
`--frames`:

```
default (12 frames)    89,473 B
--p-frames 1  (2 f)     5,475 B
--p-frames 3  (4 f)     8,022 B
```

EFFECTIVE, and it is an alias for `--frames N-1`, not a coder switch.

### The 20 that stay inert, each with its reason

| option | reason |
|---|---|
| `--video-gops 1` | default is 1 (`options.h:379`) |
| `--video-refs 1` | default is 1 (`options.h:149`) |
| `--video-me-range 16` | default is 16 (`options.h:387`) |
| `--video-max-depth 4` | default is 4 (`options.h:386`) |
| `--bit-depth 8` | default is 8 (`options.h:483`) |
| `--tiles 1` | default is `"1x1"` (`options.h:488`) |
| `--video-me-range 32` | live, non-binding at 2 frames (EFFECTIVE at QP100 in the ladder) |
| `--video-min-block 32` | live, non-binding here (EFFECTIVE at QP100: 102,887 B) |
| `--video-max-depth 8` | above the tree's natural depth (depth 0 -> 63,027 B, depth 2 -> 76,851 B) |
| `--video-recon 1/3` | a reconstruction dump; the stream is byte-identical by design (5,475 B) |
| `--video-lossless-intra-period 1/3` | only applies to the lossless writer |
| `--video-drift-refresh-{threshold,budget,qp}` | drift refresh never triggers on a 2-frame clip |
| `--p-frames 1/3` | frame-limit alias, shadowed by `--frames` (see above) |

So none of the 24 is a silent disable: 4 are live, 6 equal their own default, 3
are live-but-not-binding, 2 are output side-effects, 2 need the lossless writer,
3 need a longer clip, and 2 are an alias the sweep always shadowed.

## B. The pipe-budget arming chain: `PIPE_BUDGET_SCALE` was the wrong knob

`pb::Config` (`video_pipe_budget.h:85-122`) is built from:

```cpp
out.on = enabled();                                  // AV2ENC_PIPE_BUDGET, default ON
out.q_lo/q_hi/q_step = AV2ENC_PIPE_BUDGET_Q_*
out.budget_bytes    = env_int("AV2ENC_PIPE_BUDGET_BYTES", 0)
out.luma_weight     = AV2ENC_PIPE_BUDGET_LUMA
out.chroma_weight   = AV2ENC_PIPE_BUDGET_CHROMA
out.gamma           = AV2ENC_PIPE_BUDGET_GAMMA
```

`enabled()` is `!parse_off(getenv("AV2ENC_PIPE_BUDGET")) && !suspend_flag()` --
**on by default**. The block that consumes it
(`video_frame_decision.cpp:7071-7080`) additionally requires `keyframe_`,
non-empty `base_superblocks`, `qp_tools_active()`, and
`sb_qindex_.size() == sb_total` with matching `sb_target_q_`.

Seven carriers were tried with `AV2ENC_PIPE_BUDGET=1` forced:

```
video-aq                6183 B  PB1 6183 B  -> INERT
video-delta-q           5544 B  PB1 5544 B  -> INERT
video-segmentation      5496 B  PB1 5496 B  -> INERT
video-per-segment-q     5243 B  PB1 5243 B  -> INERT
still-delta-q           4203 B  PB1 4203 B  -> INERT
still-segmentation      4205 B  PB1 4205 B  -> INERT
still-per-segment-q     4071 B  PB1 4071 B  -> INERT
```

Every one of those sets `qp_tools_active()` true, and the master gate is already
on -- yet the stream never moves. So the remaining gate is
`sb_qindex_.size() == sb_total`, which is filled by `build_sb_maps()` /
`plan_delta_q_steps()` (`video_frame_decision.cpp:1150-1175`), and that runs on a
longer frame sequence with a converged delta-Q pass.

**Conclusion, stated as it is:** the 5 `PIPE_BUDGET_*` scalar gates remain
**unjudged**. They are not shown dead -- the allocation block is simply not
entered by any carrier I could arm on this clip, including with the master gate
explicitly on. Naming the wrong knob (`PIPE_BUDGET_SCALE` in the previous pass)
was my error, not the encoder's.

## C. One more void result, recorded

Checking `--p-frames` I ran `--p-frames=1`; this CLI accepts only the positional
form (`unknown option '--p-frames=1'`), no file was written, and the empty md5
compared against a real one → printed EFFECTIVE. Re-run positionally: also
EFFECTIVE, but for the right reason (frame count 2 vs 12). Ninth harness error of
the sweep, same shape: **a verdict from a missing output file is not a
measurement.**

## Correction: `--video-recon` takes a path, and it dumped files into the repo root

`--video-recon 1` and `--video-recon 3` left a stream identical to the baseline
(5,475 B), which the table above reads as "output side-effect". That is right,
but the semantics are sharper: the option takes an **output path**, and the
argument is used as a filename. Two files named `1` and `3` appeared in the
repository root, each 345,651 B -- exactly two 480x240 YUV frames
(480*240*1.5*2 = 345,600), i.e. the reconstruction dump for the 2-frame encode.

Both were untracked and have been deleted; nothing was committed. This is the
same class of artefact that put two 2 MB Y4M dumps into the history earlier:
an option that writes a file where you did not expect one.
