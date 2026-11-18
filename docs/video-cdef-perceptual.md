# Perceptual CDEF ranking (`AV2ENC_PERCEPTUAL_FILTER`) — measured negative

Status: **opt-in, default off, not recommended**.  The knob is implemented and
gated off; the default stream and every pin are byte-identical.  The 240p30
measurement below says loop-filter parameter choices are **not** where the
Butteraugli term is lost at the 100 kbps nightly point.

```sh
# off (default): historical SSE ranking, byte-identical stream
av2enc testdata/nasa_480x240_30fps_60f.y4m -o out.ivf \
    --video --frames 60 --qp 106 --cpu-threads 8 --nightly

# conservative form: keep the SSE frame-level set, veto it per block
AV2ENC_PERCEPTUAL_FILTER=veto av2enc ... --nightly

# full perceptual ranking, 4 CdefStrengths, CdefDamping 4
AV2ENC_PERCEPTUAL_FILTER=cdef:4 av2enc ... --nightly --video-cdef-damping 4
```

## What the encoder actually decides

Butteraugli penalises ringing and blockiness, and the nightly loop-filter chain
was the last stage whose parameter choice was purely SSE-driven:

| Tool | Decision at QP106 nightly | Encoding |
|---|---|---|
| deblock §7.17 | **always on, no search** | filter size/threshold/width are *derived* from `base_q_idx`, TX sizes, edge classes and the UV AC deltas |
| CDEF §7.18 | one global strength set per frame, chosen by full-frame SSE | `CdefStrengths == 1`: no `cdef_idx` symbol is ever coded |

`AV2ENC_TRACE_CDEF=1` on the 60-frame QP106 nightly stream (all 60 frames):

```text
cdef frame=0  enabled=1 strengths=1 ... gain=27905 set=4/2/4/2
cdef frame=1  enabled=1 strengths=1 ... gain=0     set=1/0/0/0
...
cdef frame=59 enabled=1 strengths=1 ... gain=4622  set=12/4/8/4
```

* `strengths=1` on **60/60** frames: the per-64x64 index grid is never coded
  (`index_bits=0`), so the only encoder decision is one set for the whole
  frame.
* 28 frames pick `1/0/0/0`, which is an exact no-op (§7.18.3: `pri=1` with
  damping 3 constrains every tap to a zero-weight sum).
* 10 further frames have a negative-SSE winner; `--video-cdef-rd` codes them
  as `cdef_frame_enable = 0`.
* **38/60 frames therefore filter nothing**; 22 frames filter, mostly weakly
  (`2/0/0/0`, `4/0/0/0`), and only the keyframe and a few tail frames reach
  `4/2/4/2` or stronger.

`AV2ENC_TRACE_DEBLOCK=1` (same stream) inspects 21,240 on-screen edges per
frame and filters 0 (keyframe/P1) to 2,588 edges per P frame, mean 2,022.  The
chosen widths are only `1, 2, 3, 4, 6, 8` and are fully derived state; there is
no deblock strength decision to tune.  A deblock-side experiment would have to
change the normative derivation (or the frame enable bit), which is a bitstream
change, not a parameter search.

## Implementation (opt-in)

`src/codec/filters/cdef/cdef_search.h`

* `cdef_structure_window()`: integer-exact single-scale SSIM distance on 4x4
  windows (`(1 - SSIM) * 1024`), evaluated with unnormalised moments so the
  ranking is deterministic and needs no floating point.  An optional one-sided
  variance-loss term (`perceptual_detail_weight`) penalises smoothing fine
  texture, which contrast-normalised SSIM alone can reward.
* `cdef_block_structure_cost()`: four luma windows plus one 4x4 window per
  chroma plane for the filtered 8x8 block; invisible windows are skipped.
* `cdef_block_rank()`: `structure * 256 + SSE`.  Structure dominates; SSE
  breaks ties and orders partially visible blocks.
* `CdefSearchOptions::perceptual` / `perceptual_edge_weight` (0..8) scale each
  window by the *source* window's variance (emphasise structured areas);
  `perceptual_detail_weight` (0..8) adds the anti-blur term.
* `CdefSearchResult` reports `cost_total` / `cost_off_total` next to the exact
  SSE totals, and `search_detail::select()` ranks by the cost table while
  pricing/reporting SSE.  The SSE path passes no cost table and is unchanged
  (existing callers, including the CUDA search and the 10-bit planner, keep
  their signatures via defaulted arguments).

`src/codec/filters/cdef/syntax/video_cdef.{h,cpp}` adds
`Options::perceptual`, `perceptual_veto`, `perceptual_edge_weight` and
`perceptual_detail_weight`, plus `Plan::cost_{before,after}`.  When the flag is
on, `plan()` builds a `cdef::CdefSearchFrame` and calls
`cdef::cdef_search_plan()` directly; otherwise it keeps the historical
`still_cdef_plan()` wrapper.  The header, the index syntax and the filter are
the same code.  The **veto** form first runs the SSE search with
`CdefStrengths == 1` (the historical decision), then searches the two-entry
pool `{identity, SSE winner}` perceptually, so the header carries at most two
sets and no strength the SSE arm would not have used.

`src/pipeline/video/post/video_loop_filters.cpp` parses
`AV2ENC_PERCEPTUAL_FILTER`:

| Value | Meaning |
|---|---|
| unset / `0` / `off` | historical SSE ranking (byte-identical) |
| `cdef` | rank the default ladder perceptually, CLI `CdefStrengths` |
| `cdef:N[:W[:D]]` | also request `N` strengths (1..8), edge weight `W`, detail weight `D` |
| `veto[:W[:D]]` | SSE frame-level set, perceptual per-block veto |

Malformed values are an encode error, not a silent fallback, so a typo cannot
be measured as the baseline.  The "filter made the frame worse" enable check
uses the ranking cost in the opt-in arm.

`src/pipeline/video/post/deblock_grid.h` adds the `AV2ENC_TRACE_DEBLOCK`
diagnostic used above (it never changes pixels; the traced stream keeps the
pinned MD5).

## Measurement

Clip `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, `--nightly`,
`--cpu-threads 8`; metrics exactly the acceptance chain (`tools/quality.py`,
`tools/extra_metrics.py`, `tools/kpi.py`).

Baseline at QP106: 25,008 B / 100.032 kbps / JOD 9.7334 / BA 3.4202 /
SS2 80.837 / K 82.15.

### Per-frame BA is concentrated in the drifted P tail

From the per-frame `tools/extra_metrics.py` run on the baseline decode:

| Frames | Mean BA | Mean SS2 |
|---|---:|---:|
| 0-32 | 2.075 | 87.79 |
| 33-43 | 3.919 | 80.52 |
| 44-59 | 5.852 | 66.72 |

The tail carries two thirds (67%) of the total BA.  BA rises from 2.1 (f33) to
4.5-7.0 across f38-59 at a fixed QP, with no filter parameter changing.

### Where CDEF sits on those frames

Per-frame BA delta against the baseline (QP106, same rate):

| Variant | f0-32 | f33-43 | f44-59 | whole clip |
|---|---:|---:|---:|---:|
| CDEF off | +0.315 | -0.611 | **+0.061** | +0.078 |
| SSE per-block, `--video-cdef-strength 4` | +0.096 | -0.295 | **+0.032** | -0.067 |
| perceptual veto (`veto`) | -0.077 | -0.631 | **+0.305** | -0.077 |
| perceptual ranking + detail 1 (`cdef:4:0:1`, damping 4) | +0.097 | -0.639 | **-0.110** | -0.093 |

Two things follow:

1. **CDEF already helps the hot band.**  Turning it off is +0.061 BA worse on
   f44-59 (9/16 frames better off, 7 worse); vetoing it far more selectively
   is +0.305 worse (5/16 better).  On those frames CDEF is already the right
   direction; weakening it is not the lever.
2. **The large per-frame swings are chain effects, not filter strength.**  f37
   improves by 1.7-2.0 under every variant, including *turning CDEF off*; the
   f33-43 band moves by ~-0.6 for all four variants.  A filter change shifts
   the reference path, and the drift onset lands on different frames.

### The conservative veto (best iso-rate point)

`AV2ENC_PERCEPTUAL_FILTER=veto`, damping 3.  The trace shows the SSE set kept
on the header but vetoed on 24-31 of the 32 blocks per filtered frame; 30/60
frames carry only the identity set.  Rate is unchanged at QP106 (24,997 B,
99.988 kbps).

| QP | kbps | ΔJOD | ΔBA | ΔSS2 | ΔK | Δrate |
|---:|---:|---:|---:|---:|---:|---:|
| 104 | 104.648 | +0.0004 | -0.0517 | +0.384 | +0.14 | +0.02% |
| 105 | 102.372 | -0.0269 | **+1.2225** | -0.528 | **-2.03** | +0.99% |
| 106 | 99.988 | -0.0087 | -0.0769 | +0.073 | +0.11 | -0.04% |
| 107 | 98.300 | +0.0177 | -0.8799 | +0.234 | +1.53 | -1.25% |
| 108 | 96.868 | +0.0097 | -0.0928 | +0.259 | +0.25 | -0.30% |
| 109 | 96.088 | +0.0013 | -0.0490 | -0.162 | +0.07 | +0.10% |

Iso-rate at 100.0 kbps (the arm is already there):

| Arm | QP | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| off | 106.07 | 9.7320 | 3.4789 | 80.799 | 82.05 |
| veto | 105.99 | 9.7245 | 3.3504 | 80.904 | 82.25 |
| **Δ** | | **-0.0074** | **-0.1285** | **+0.106** | **+0.20** |

That is a real, rate-neutral BA/K gain at the target point, and five of the
six QPs improve.  QP105 is the exception: the tail collapses by +2.6 BA mean
(up to +5.4 on one frame), i.e. the same filter decision can send the P chain
down a much worse path.  A selector whose worst case is -2 K is not a product
default.

### Full perceptual ranking

Same-QP deltas for `cdef:4` + damping 4 (the QP106 point that looks best):

| QP | ΔJOD | ΔBA | ΔSS2 | ΔK | Δrate |
|---:|---:|---:|---:|---:|---:|
| 104 | -0.0046 | -0.3367 | +0.437 | +0.56 | -0.52% |
| 105 | -0.0119 | +0.3925 | -0.264 | -0.75 | +1.13% |
| 106 | +0.0030 | -0.1353 | -0.034 | +0.26 | +0.99% |
| 107 | -0.0075 | +0.1838 | -0.117 | -0.31 | -0.96% |
| 108 | -0.0084 | +0.6069 | -0.458 | -1.05 | +0.61% |
| 109 | -0.0114 | +0.4487 | -0.128 | -0.86 | +0.57% |

The BA sign flips between adjacent QPs; JOD is worse at five of six points.
Adding the anti-blur detail term (`cdef:4:0:1`, damping 4) makes JOD positive
at five of six QPs (+0.010 at QP106, +0.024 at QP104) but costs +0.5-1.3% rate;
at iso-rate it is ΔJOD +0.002 / ΔBA +0.254 / ΔSS2 -0.220 / ΔK -0.43.

The whole CDEF decision space -- on/off, one versus four strength sets, SSE
versus perceptual ranking, damping 3-5 -- spans BA 3.28-3.98 at ~100 kbps,
which is the same magnitude as the unmodified stream's own QP-to-QP ripple
(BA 3.35-4.32 over QP104-109).

## Verdict

* The change is correct and safe (opt-in, default off, `dav2d --strict 1`
  exact, encoder recon == decoded output, pins unchanged).
* Best iso-rate result: `AV2ENC_PERCEPTUAL_FILTER=veto`, ΔJOD -0.007,
  ΔBA **-0.129**, ΔSS2 +0.106, ΔK **+0.20** at 99.99 kbps.  It is not
  recommended as a default: QP105 regresses by -2.03 K through a tail chain
  shift, and the gain is outside the hot band.
* Deblocking has no encoder-side strength search to tune; its choice is
  normative derivation state.
* The Butteraugli term is set by the drifted P tail's coding error, and the
  existing CDEF already helps that band.  The per-frame evidence says a
  conservative selector hurts the hot frames; the lever is the
  reference/allocator chain (`docs/video-inter-residual-rd.md`), not loop-filter
  parameters.  Keep this experiment off.

## Gates

`tools/gate_video_nightly.sh` (default/quality/nightly pins, strict decode,
visible recon, BAWP), `src/pipeline/video/gate_video_qp_lambda.sh` (24,113 B
legacy pin), `src/pipeline/video/gate_video_filters.sh`,
`tools/refactor/check_wavefront_pins.sh` (720p `c4ba2813`), and
`tools/test.sh` 10/10 unit tests all pass with the flag unset.
