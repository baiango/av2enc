# AV2ENC_RD_JOINT_P: the "cannot fire" verdict was wrong, and retracted

## What I claimed, and on what evidence

Earlier this session I recorded `AV2ENC_RD_JOINT_P` as the single **silent
disable** in av2enc: a gate `--nightly` sets, whose own comment credits it with
+0.68 K, that could never fire. The evidence was this trace:

```
joint-trace: result=0 keyframe=0 off=0 armed=0 sdp=0 force_single=0 palette=0
             ibc=0 qp_tools=0 tcq=0 tip=0 tile_w=480 tile_h=240 probe_ready=0
```

The run that produced it was `AV2ENC_JOINT_TRACE=1 ... --nightly` -- **without
`AV2ENC_RD_JOINT_P` set**. `armed=0` was therefore trivial: the gate was off.
The conclusion "armed=0 AND probe_ready=0" was an artefact of my own harness, in
exactly the way the pipe-budget verdict had been.

## The same trace, with the gate actually set

`AV2ENC_RD_JOINT_P=1 AV2ENC_JOINT_TRACE=1 ... --nightly`:

```
joint-trace: result=0 keyframe=1 off=0 armed=0 ... probe_ready=0   (key frame)
joint-trace: result=0 keyframe=0 off=0 armed=1 ... probe_ready=0   (inter)
```

**`armed=1` on every inter frame.** `joint_plan::armed(ordinal)` is
`cfg.on && first <= ordinal <= last` (`video_joint_plan.h:172-176`) with
`first = 1`, `last = 4096` -- so arming is not the problem at all. Only
`pipe_probe_ready_` is false.

## The missing companion

`pipe_probe_ready_` is set in `plan_pass()` under one guard
(`video_frame_decision.cpp:1592`):

```cpp
if ((pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) && ...) {
  ...
  pipe_probe_.reset(config, cdf_seed_);
  pipe_probe_ready_ = true;
}
```

i.e. `AV2ENC_PIPE_PERC`, `AV2ENC_EFF_PERC` or `AV2ENC_PIPE_BA` -- all off by
default, like `AV2ENC_RD_JOINT_P` itself. With one of them enabled the solve
fires:

```
AV2ENC_EFF_PERC=1                     10,026 B  2e9dcec4574c4c8282cb2b5d355ed689
AV2ENC_EFF_PERC=1 AV2ENC_RD_JOINT_P=1 10,074 B  6887b0cb5aa4ad5e8028252add81352c
joint-trace: result=1 ... armed=1 ...   (inter frames)
```

**EFFECTIVE.** The joint solve runs, and changes the stream, exactly as its
comment says it does.

## What the flag now does instead of failing quietly

The warning added in `4dbe3af0` said "the joint solve cannot fire here". That
was a wrong verdict printed as a fact. It now names the actual cause, in three
cases:

```
AV2ENC_RD_JOINT_P=1 alone:
  av2enc: warning: AV2ENC_RD_JOINT_P is set but the joint solve needs the
  shared pipe probe, which is only built when AV2ENC_EFF_PERC,
  AV2ENC_PIPE_PERC or AV2ENC_PIPE_BA is enabled; set one of those or the gate
  does nothing.
```

and, for an ordinal outside `FIRST..LAST` or for a feature exclusion, a message
naming that instead.

Verified:

| configuration | warning | stream |
|---|---|---|
| `AV2ENC_RD_JOINT_P=1` alone | printed, names the companion | unchanged |
| `AV2ENC_RD_JOINT_P=1` + `AV2ENC_EFF_PERC=1` | **none** (the solve fires) | **10,074 B** |
| `--nightly` alone | none | 2f4f4b39 (unchanged) |

## Consequence for the whole sweep

**There is no silent disable in av2enc.** The last candidate was a
companion-gated research gate whose companion is itself off by default, and the
diagnosis of it was my harness error, not the encoder's.

The tally of gates that looked disabled and were not:

| what looked dead | what it actually needed |
|---|---|
| 21 INERT booleans (22 of 30) | the `--nightly` baseline already enabled them |
| 36 still-path flags | `--still` (they were run under `--video`) |
| 9 of 21 scalar gates | `--nightly-b`, the carrier their code serves |
| 5 `PIPE_BUDGET_*` weights | `AV2ENC_RD_DRIVER` + delta-Q + a binding SCALE |
| `AV2ENC_RD_JOINT_P` | `AV2ENC_EFF_PERC` (the shared pipe probe) |

The repeated shape of every false verdict in this sweep is one sentence: **a
switch whose companion is unset looks identical to a switch that does nothing.**
That is a documentation and diagnostics problem -- which is precisely what the
warning now addresses -- and it is not a correctness bug in the encoder.
