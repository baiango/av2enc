# AV2 intra bi-prediction (IBP) — CPU reference

Spec-only implementation of AV2 intra bi-prediction for the encoder's mode
decision and reconstruction path.

Normative source: **AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`, SHA-256 `e9916f09…b533f`).

## Files

```text
src/codec/intra/ibp/
  ibp.h                 public API, decision API, §5.4.5 syntax helpers
  ibp.cpp               CPU reference implementation
  ibp_tables.h          normative tables, shared with the CUDA device code
  ibp_test.cpp          independent reference + synthetic/real-block tests
  build_test.sh         standalone build/run (no root CMake edits)
  README.md             this file
```

Nothing outside this directory is modified. The module reads
`src/codec/intra/modes/` for §7.13.2.1 edge preparation and the ordinary
(non-IBP) predictors; it never writes there.

## Important: which "IBP weights"?

The task text mentions `Ii_Weights_1d` / §7.13.3.29. In v1.0.0 those belong to
**inter-intra prediction**, not IBP:

| Table / process | Section | Used by |
|---|---|---|
| `ibp_weights()` (16×16, 0..128) | §7.13.2.9 | **IBP blending** of two directional predictions |
| `Ibp_Weights[5][16]` | §7.13.2.12 | **IBP DC** post-filter |
| `Ii_Weights_1d[128]` | §7.13.3.29 | inter-intra luma mask |

All three are implemented (the last one via `ii_weights_1d()` and
`build_inter_intra_mask()`), but the IBP predictor uses only the first two.
There is no normative IBP table named `Ii_Weights_1d`.

## Normative scope

| Section | Process |
|---|---|
| §7.13.2.7 | IBP edge-filter implications: per-edge filter types, extra right/bottom extent, adjusted angles, corner rule |
| §7.13.2.8 | Single directional prediction, called twice (with IDIF for luma) |
| §7.13.2.9 | `ibp_weights()` |
| §7.13.2.10 | DC prediction (delegated to `intra::predict_intra`) |
| §7.13.2.12 | IBP DC process (`Ibp_Weights`) |
| §7.13.2.14 | Filter corner |
| §7.13.2.17/.18 | Edge-filter strength and 5-tap filter |
| §5.4.5 | `sequence_intra_config` `enable_ibp` f(1) |
| §7.13.3.29 | `Ii_Weights_1d` (provided for completeness, not used by IBP) |

In scope: 8-bit, luma and chroma, block extents 4..64 per axis, all 13 intra
modes, `angle_delta` in −3..+3, WAIP, `MrlIndex = 0`.

Out of scope: MRL (`MrlIndex > 0`), DIP, MHCCP, CfL, palette/IntraBC, 10/12-bit,
and bitstream emission. IBP itself has **no per-block syntax bit** and no new
CDF: it is derived from `enable_ibp`, `mode`, and `angle_delta`.

## Decision rules

```text
applyIbp = enable_ibp && !(w == 4 && h == 4)
useIBP   = applyIbp && (angle_delta is even) && plane == 0 &&
           (pAngle < 90 || pAngle > 180) && MrlIndex == 0
```

`pAngle` is formed from `Mode_To_Angle[mode] + angle_delta * ANGLE_STEP` and,
for intra (non-inter-intra) blocks, passed through `wide_angle_mapping`.

Consequences, all exercised by the tests:

* 4×4 blocks never use IBP.
* Chroma never blends (but its edge filtering and DC are still IBP-modified).
* Odd angle deltas never blend.
* Only 11 luma directional pairs are possible on any non-4×4 shape:
  `D45`, `D67`, `D203` with `angle_delta ∈ {−2, 0, +2}`, plus `V` with `−2`
  and `H` with `+2`.
* `applyIbp` is true for **every** non-4×4 directional block when
  `enable_ibp = 1`, even if `useIBP` is false (for example 135°). This changes
  edge filtering and is just as normative as the blend.
* IBP is forced by the decoder for an eligible pair. The encoder may choose a
  different `(mode, angle_delta)`, but it cannot choose to skip the blend for
  the candidate it signals.

### Edge-filter implications (§7.13.2.7)

When `applyIbp` is true and `enable_intra_edge_filter` is 1 with `MrlIndex = 0`:

* `filterTypeAbove` and `filterTypeLeft` stay separate; they are not OR-ed.
* `needRight |= pAngle > 180`, `needBottom |= pAngle < 90`.
* `angleAbove -= 180` when `> 90`; `angleLeft += 180` when `< −90`.
* The corner filter applies when `applyIbp` or `90 < pAngle < 180`, and
  `w + h >= 24`.
* `numPx` for the above edge is `min(w, maxX−x+1) + (needRight ? h : 0) + 1`;
  for the left edge it is `min(h, maxY−y+1) + (needBottom ? w : 0) + 1`.

The module reproduces these steps, then runs §7.13.2.8 twice with the same
filtered edges.

## API

```cpp
namespace av2enc::ibp {

struct WeightMatrix { /* 16x16 */ int at(r,c) const; };
WeightMatrix ibp_weights(int p_angle);          // §7.13.2.9
int ibp_dc_weight(int log2_extent, int index);  // §7.13.2.12

struct DecisionParams {
  bool enable_ibp; bool is_inter;
  Mode mode; int angle_delta;
  int w, h, plane, mrl_index;
};
struct Decision {
  bool valid;
  bool apply_ibp_edge_rules;  // applyIbp
  bool use_ibp;               // two-angle blend
  int p_angle, second_angle;
};
Decision decide(const DecisionParams&);

bool ibp_dc_applicable(bool enable_ibp, bool use_dip, Mode mode,
                       int w, int h, int plane, bool uv_is_cfl);

struct ModeCandidate { Mode mode; int8_t angle_delta; };
int build_ibp_candidates(std::span<ModeCandidate> out, int w, int h,
                         int plane = 0, int mrl_index = 0);

struct PredictParams {
  Mode mode; int angle_delta;
  bool enable_ibp, enable_edge_filter, is_inter;
  bool above_neighbor_is_smooth, left_neighbor_is_smooth;
  bool use_dip, uv_is_cfl;
};
struct PredictResult { bool blended, dc_modified; int p_angle, second_angle; };
IntraStatus predict(const IntraEdges&, const PredictParams&,
                    uint8_t* dst, size_t stride, PredictResult* = nullptr);
IntraStatus predict_block(const NeighborRequest&, int bit_depth, ...);
}
```

`predict()` is a drop-in replacement for `intra::predict_intra()` when
`enable_ibp = 1`:

* DC → ordinary DC plus §7.13.2.12 when applicable.
* directional → IBP-aware edge filtering, then one or two predictions.
* SMOOTH/SMOOTH_V/SMOOTH_H/PAETH → delegated unchanged.

It never mutates the caller's `IntraEdges`, so one prepared edge set is reusable
for every RDO candidate.

### Syntax helpers

`SequenceIntraConfig`, `BitWriter`, `BitReader`,
`write_sequence_intra_config()` and `read_sequence_intra_config()` implement the
exact §5.4.5 order:

```text
enable_dip f(1)
enable_intra_edge_filter f(1)
enable_mrls f(1)
enable_cfl_intra f(1)
if (!Monochrome) cfl_ds_filter_index f(2)
enable_mhccp f(1)
enable_ibp f(1)
```

For a non-monochrome stream `enable_ibp` is the last of 8 bits; for monochrome
it is the last of 6. There is no block-level IBP element.

## Build and test

```sh
cd src/codec/intra/ibp
./build_test.sh                 # -O2, build + run
SANITIZE=1 ./build_test.sh      # ASan + UBSan
WERROR=1 ./build_test.sh        # warnings as errors
```

Current result (Apple clang, 8-bit test media in `tests/data/`):

```text
ibp_test: PASS (56234221 checks)
decision sweep: 396 pairs, 198 enumerated luma candidates
prediction sweep: 111264 blocks, 4752 IBP blends
standard-path parity: 6954 blocks
DC sweep: 912 blocks
```

Coverage:

* `ibp_weights()` for every reachable 39..89 angle plus Max(39,·) clamping.
* Decision sweep: 19 shapes (including 2:1, 4:1, 8:1, 16:1), luma/chroma,
  `enable_ibp` 0/1, `is_inter` 0/1, all modes/deltas.
* Independent reference parity for every prediction candidate, both planes,
  smooth-neighbour filter types, edge-filter on/off, `is_inter` on/off.
* Standard-path parity against `intra::predict_intra()` when `enable_ibp = 0`,
  and for 4×4 with `enable_ibp = 1`.
* IBP DC: every shape, `haveLeft`/`haveAbove` combinations, luma/chroma,
  CfL exemption, 4×4 exemption.
* Syntax bit round-trip and exact bit positions.
* `Ii_Weights_1d` values and mask generation.
* Real luma blocks from `tests/data/basic_64x64_1f.y4m` and
  `basic_640x360_1f.y4m`, compared against the independent reference.

## Quality proxy (not RDO, not bitrate)

On real luma blocks (16×16, stride 32/64, source pixels used as neighbour
values, SAD/SSE only):

| Clip | Blocks | DC SAD/px | Best same-set, IBP off | Best same-set, IBP on | IBP wins | PSNR proxy |
|---|---:|---:|---:|---:|---:|---:|
| 64×64 | 4 | 40.24 | 35.54 | 37.03 | 2 (50.0%) | 12.989 → 12.878 dB |
| 640×360 | 220 | 5.44 | 3.16 | 3.11 | 30 (13.6%) | 25.220 → 25.431 dB |

Method: the same candidate set (DC + eight directional modes with deltas
−2/0/+2) is searched twice, once with `enable_ibp = 0` and once with
`enable_ibp = 1`; the difference therefore includes the normative edge-filter
change, not only the blend. 64×64 has too few blocks to be meaningful. This is
a predictor-side proxy only: no transform, quantisation, rate term, or
reconstructed-neighbour feedback is modelled.

## Bitstream / decoder gate

IBP has no per-block syntax: the only new bit is §5.4.5 `enable_ibp` (the last
`f(1)` of `sequence_intra_config`).  Both integrated writers emit it and the
pinned decoder (`dav2d f9dafbc --strict 1`) is the gate.

### Still (`--still --still-ibp`, M1)

`src/pipeline/still/still_headers.cpp` writes `c.enable_ibp`; the luma search
and the reconstruction path use `ibp::predict`/`ibp::predict_block` when the
option is set.  The gate is `tools/still_verify/intra_gate.sh --only ibp`
(`--bin build-int/av2enc --dump build-int/still_recon_dump`):

```text
PASS ibp_64 (938 B, recon exact)
PASS ibp_360 (13294 B, recon exact)
PASS ibp_fullintra_64 (974 B, recon exact)
PASS ibp_fullintra_360 (13216 B, recon exact)
PASS ibp_rdoq_64 (819 B, recon exact)
PASS ibp_cdef_360 (13314 B, recon exact)
PASS dip_ibp_360 (13205 B, recon exact)
```

The bytes prove the bit is live: at `tests/data/basic_64x64_1f.y4m` the
`--still` stream is `0ef98250…` and `--still --still-ibp` is `b57b3e4b…`; at
`tests/data/basic_640x360_1f.y4m` they are `e61de628…` and `36e07364…`.
Both decoded Y4Ms equal `still_recon_dump` sample for sample.

The CPU test's syntax round-trip remains the component-level gate; this stream
gate is the decoder-level proof.

### Video (`--video --video-ibp`, CPU keyframe)

`src/pipeline/video/` signals `config.enable_ibp`, routes the §7.13.2 luma
search (`video_intra_search.h::search_luma`) and the DC/chroma predictor
(`intra_dc_predict`) through this component, and reports the derived
`ibp_apply`/`ibp_blend` decision on `--video-trace`.  The gate is
`src/pipeline/video/video_ibp_gate.sh`:

```text
PASS default_64 (1080 B, exact)          PASS ibp_64 (1083 B, exact)
PASS default_360 (52363 B, exact)        PASS ibp_360 (50742 B, exact)
PASS ibp_full_64 (1083 B, blend=1)       PASS ibp_full_360 (51003 B, blend=108)
PASS dc_live_64 (3251 samples differ)    PASS dc_live_360 (267989 differ)
+ 5 named rejections
```

`--video-intra-full` off exercises the §7.13.2.12 DC post-filter; on, the
61-candidate search selects the two-angle blend (`ibp_blend=1`).  Both are
recon-exact at 64x64 and 640x360.  The default `--video` MD5s are unchanged
(the feature is default-off).  The writer rejects, by name, combinations that
predict through a different path: CfL/MHCCP chroma, MRL (`MrlIndex` is 0),
palette/IBC/morph screen content, inter-intra, tiles, SVC, the CUDA backend
and the B/temporal/TIP/random-access schedules.

The CPU test's syntax round-trip remains the component-level gate; the two
stream gates above are the decoder-level proof.
