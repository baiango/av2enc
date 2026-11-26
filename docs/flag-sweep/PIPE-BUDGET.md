# Pipe-budget gates: armed, judged, and all live

The previous pass recorded these as "unjudged, not dead" and named
`AV2ENC_PIPE_BUDGET_SCALE` as the arming knob. Both halves were wrong in an
instructive way.

## The arming chain, read from the source

```cpp
// video_pipe_budget.h:47-64  -- parse_off(NULL) is TRUE
inline bool parse_off(const char* value) {
  if (value == nullptr || *value == '\0') return true;
  ...
inline bool enabled() { ... return !parse_off(std::getenv("AV2ENC_PIPE_BUDGET")); }
```

So `AV2ENC_PIPE_BUDGET` is **off when unset** (an earlier note claimed the
opposite). The call site (`video_frame_decision.cpp:996`) then needs

```cpp
bool sequential_key = keyframe_ && rd_driver::enabled() && !sdp &&
                      !force_single_block && !palette && !ibc && !seg_enabled_;
if (keyframe_ && pipe_budget::enabled()) return pipe_budget_decide_frame(...);
```

`rd_driver::enabled()` is `AV2ENC_RD_DRIVER` (also off when unset), so a plain
run never reaches the walk at all -- which is why every earlier carrier was
byte-identical. And inside `pipe_budget_decide_frame()` the guard is

```cpp
if (!qp_tools_active() || sb_qindex_.size() != sb_total || sb_target_q_.size() != sb_total)
```

i.e. a key frame with delta-Q/AQ/per-segment-Q.

## Armed: `--nightly`-free carrier, 12 frames, 240p

`--video --frames 12 --qp 150 --cpu-threads 1 --video-delta-q` plus
`AV2ENC_RD_DRIVER=1`:

| `AV2ENC_PIPE_BUDGET` | bytes | verdict |
|---|---:|---|
| unset | 17,907 | -- |
| `=1` | **18,257** | **ARMED** |

Same with `--video-aq`: 19,222 -> 19,259 (armed). `--video-per-segment-q`:
17,579 -> 17,579 (segmentation excludes the sequential driver, as the source
says).

## The gates, on the armed carrier (base 18,257 B / `05b0d009`)

| gate | 0.5 | 4.0 |
|---|---|---|
| `AV2ENC_PIPE_BUDGET_STRENGTH` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_DRIVER_ACT` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_ENGINE_CAL_BR` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_ENGINE_CAL_TXB` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_PIPE_BUDGET_LUMA` | inert | inert |
| `AV2ENC_PIPE_BUDGET_CHROMA` | inert | inert |
| `AV2ENC_PIPE_BUDGET_GAMMA` | inert | inert |
| `AV2ENC_PIPE_BUDGET_ACT_MAX` | inert | inert |
| `AV2ENC_RD_DRIVER_SBALLOC_MAX` | inert | inert |

Four more gates are live. The five inert ones are **weights**, and a weight can
only matter where the allocation has to trade -- which needs the budget to bind.

## `PIPE_BUDGET_BYTES` is not the gauge; `PIPE_BUDGET_SCALE` is

`AV2ENC_PIPE_BUDGET_BYTES` at 16,000 / 12,000 / 8,000 left the 18,257 B stream
byte-identical. Reading the code explains it: the allowance is

```cpp
const long long scale = uniform_control ? 1000
    : std::max<long long>(1, pb::env_int("AV2ENC_PIPE_BUDGET_SCALE", 1000));
```

and `uniform_control` is only true when `budget_bytes < 0`, so a positive byte
count is not the arm -- the thousandths SCALE is.

On the armed carrier the SCALE does bind:

```
SCALE 1000 (default)   18,257 B
SCALE  500             18,111 B   EFFECTIVE
SCALE 2000             18,267 B   EFFECTIVE
SCALE 3000             18,273 B   EFFECTIVE
```

## With the budget binding (SCALE 500), every weight is live

| gate | 0.5 | 4.0 |
|---|---|---|
| `AV2ENC_PIPE_BUDGET_LUMA` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_PIPE_BUDGET_CHROMA` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_PIPE_BUDGET_GAMMA` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_PIPE_BUDGET_ACT_MAX` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_DRIVER_SBALLOC_MAX` | **EFFECTIVE** | inert |

## Result

Every one of the gates this pass set out to judge is now **measured live**:

- 5/5 `AV2ENC_PIPE_BUDGET_*` (STRENGTH, LUMA, CHROMA, GAMMA, ACT_MAX)
- `AV2ENC_RD_DRIVER_ACT`, `AV2ENC_RD_DRIVER_SBALLOC_MAX` (at 0.5)
- `AV2ENC_RD_ENGINE_CAL_BR`, `AV2ENC_RD_ENGINE_CAL_TXB`

Nine gates that had been carried as "inert" or "unjudged" across several
passes. All nine were live; none was ever a silent disable.

## The pattern this confirms

Nothing in av2enc was disabled. What kept looking disabled was always the
**precondition set** -- a mode, a carrier, a companion, or a value that had to
bind before the parameter had any effect. The single genuine silent no-op found
anywhere in the sweep remains `AV2ENC_RD_JOINT_P` (`armed=0`,
`probe_ready=0`), now warned about at `video_frame_decision.cpp:889`.
