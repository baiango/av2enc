
## Composition result (do not re-run)

Matched K does NOT stack across the merged KEEPs. Dense 6-point ladders
(QP113-118, docs/rd-compose.md): base driver 86.041, +MC 86.184, +tail 86.266,
+both 86.160 (robust fit, R2 0.012). MC's early +0.341 was a 2-point bracket
artefact; the dense-ladder value is +0.143. They compose on rate (both is 2-4%
smaller at every QP), not on matched K.

Gap decomposition vs AVM: CVVDP 9.836 vs 9.862 (equal); BA -2.82 K (70%), SS2
-1.06 K (26%). Split 48% head f1-32 / 48% tail f33-59 / 1% key. The head
spends 45-65 B/frame vs AVM ~197.

## Closed by lane 9 (per-block MC)

- Per-block MC filter (`AV2ENC_INTERP_BLOCK`, CSF 2): KEEP +0.341, merged,
  default off; agent ceiling f0-32 91.513. Do not duplicate (lane 3 picks the
  next queue item).

## Closed by the tail lane

- Tail exact-rate TX (`AV2ENC_RD_TAIL_TX`): KEEP for the f0-32 ceiling
  (+0.113 fit, +0.145 rate-normalised; K bracket +0.225, fit +0.126). Merged,
  default off.
- Tail 4x4/8x4 route: rejected (+1.2% rate, -0.28/-0.47 dB); the P-frame
  coding-block floor blocks it.

## Closed by the single loop (do not repeat)

- Head residual/QP budget: the QP plan was already negative; the new skip
  surcharge measured -0.26 (beta 0.25) and -0.55 (beta 0.5).
- DDT / IST at the KPI point: FAIL +0.010; the IST reachability fix is in but
  does not move K.

## Lane 9 focus

Start with: **GME / warp per-block and global motion**. If blocked, continue down the queue. Do not duplicate another lane's focus.

## Lane index

- lane 1: Tail: BA-aware 4x4/8x4 residual (distinct from the tail exact-rate subagent)
- lane 2: Head f1-32: inter prediction/residual efficiency (not QP allocation)
- lane 3: Per-block MC filter: EIGHTTAP/SHARP/SMOOTH by RD
- lane 4: Perceptual loop-filter RD: CDEF/deblock by BA/SS2/CSF
- lane 5: Inter partition/TX exact-rate with the driver's BA-aware D
- lane 6: Chroma TX/levels with live contexts
- lane 7: Live-context rate model in the driver/inter decisions
- lane 8: DDT / IST at the 100 kbps point
- lane 9: GME / warp per-block and global motion
- lane 10: Sequential tree edge-D / partition with the driver
# RD-engine continuous loop — lane 9

Status: **active**. One agent iteration at a time on the RD-engine queue. The
runner (`tools/agent_loop.py`) measures each iteration with `tools/envelope.py`
against the 86.013 base and keeps or reverts `src/`.

## Base

K = **86.013** (bracket) / **85.580** (fit) at 240p/100 kbps, L3+L5 merged.

Runner base env (merged into every arm; do not re-specify):

```
AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50
AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1
```

The live lane-9 runner (2026-11-20) also exports `AV2ENC_RD_DRIVER=1`; with it
the base arm measures **K 85.942** at QP114/116 (iteration 1 below).

Base args:

```
--video --frames 60 --cpu-threads 8 --fast-decode
--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

Pins (gate off): QP106 25,008 B / `0639b436…`; QP110 23,690 B / `1b99127a…`.

## Oracle ceiling target (f0-32 > 91.068)

The f0-32 source-replacement ceiling is limited by the tail f33-59. The
tail-TX threshold is saturated. The next lever is the exact-rate TX/level RD
(the L3 class) applied to the tail frames: fidelity per bit, not rate cuts.
Report the f0-32 and f44-59 ceilings with every result, not only matched K.


## Fidelity-preserving tools (priority)

Rule: a tool that cuts rate must be fidelity-neutral at fixed QP. The closed
rate-cut tools (RDOQ, TCQ, dead zone, TX band) lost because they traded
high-frequency fidelity for bytes. These replace that trade with explicit
fidelity pricing.

1. Perceptual loop-filter RD: decide CDEF/deblock with BA/SS2/CSF, not SSE;
   skip filtering where it removes detail.
2. Per-part D in the key TX/level trellis: use the TX search's per-part
   reconstruction, not a block-level CSF re-weighting (lane 2's D fix).
3. Per-coefficient perceptual quantisation: coefficient-level CSF cost instead
   of a global AC dead zone.
4. Per-block MC filter: EIGHTTAP/SHARP/SMOOTH per block by RD (the frame-global
   forms were negative; per-block unmeasured).
5. Reconstruction-domain quantisation: account for the decoder's integer
   inverse/rounding in the trellis.

## Queue (pick the first unblocked item, one per iteration)

1. **Head residual/QP budget** — f1–32 is 45 B/frame vs SVT's 82 and is the
   f0–32 ceiling limiter. Try a head-frame QP/residual offset funded by the key
   or tail; measure the f0–32/f44–59 ceilings as well as K.
2. **Per-block MC filter / GME/warp** — the frame-global forms were negative;
   the per-block switchable form is unmeasured at 100 kbps.
3. **DDT / IST / parity hiding at the KPI point** — implemented and opt-in,
   never scored in the 100 kbps aggregate.
4. **Key partition with sequential recon** — only if `rd-seq` misses; the
   candidate must be measured in its own reconstruction.
5. **Live-context rate model** — only if `rd-rate` misses.
6. Next target from the census; record why.

## In flight (do not duplicate)

- `rd-rate` live-context rate model
- `rd-seq` sequential-recon tree
- `rd-chroma-ctx` chroma live contexts
- `rd-repair` net-of-repair propagation

## Closed (do not repeat)

Key rate cuts (RDOQ/TX band/bypass/EOB/TCQ/CCTX/exact cost/TX plan/mode RD),
inter λ rescaling, reference structure (B/ARF/hierarchical), tail-TX knob
sweeps, filters/AQ/dead zones/QM, horizon/skip propagation in the current
units.

- Per-block MC filter (queue 2): measured **+0.341 K** on the live
  L3+L5+`RD_DRIVER` base (iteration 1); the EIGHTTAP skip/residual guard is
  K-neutral. Do not repeat the filter selection or the guard.

## Iteration 1 — per-block MC filter (queue 2)

**Target.** Queue item 2, per-block MC filter. The GME/warp half is blocked on
this point: the 240p global-motion fit is IDENTITY (bit-level no-op) and the
escape-hatch warp predictor is conformance-unproven and cannot compose with
`--fast-decode` (`--video-inter-residual`). The per-block switchable filter is
already in the tree (`AV2ENC_INTERP_BLOCK`); it was measured at +0.131 on the
old 85.099 base but had never been scored on the current L3+L5+driver base.

**Patch.** `AV2ENC_INTERP_BLOCK_SKIP` (default 0) in
`src/pipeline/video/video_interp_filter.h` and the filter loop in
`src/pipeline/video/video_frame_decision.cpp`: EIGHTTAP keeps its own
skip/residual decision, and a non-zero filter may refine only a leaf that
already codes residual (1 = all skips, 2 = non-zero-MV skips only). A filtered
skip installs its prediction into the reference and the following frames copy
it, which is the drift mechanism in `docs/luma-audit.md` §4.5. Gate off
(`AV2ENC_INTERP_BLOCK` unset) is byte-identical: QP106 **25,008 B /
`0639b436…`**, QP110 **23,690 B / `1b99127a…`**.

**Bracket.** The live lane-9 runner base env is the six 86.013 variables plus
`AV2ENC_RD_DRIVER=1`. Its QP118 stream is already 95.8 kbps, so the spec starts
at QP114 (the envelope's secant step stalls on a seen QP when both points are
below target). All arms used `--video --frames 60 --cpu-threads 8 --fast-decode
--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250`; base
env also `AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50 AV2ENC_RD_KEY_TX=1
AV2ENC_RD_UVLEVEL=1 AV2ENC_RD_DRIVER=1`.

| arm | env | QP114 | QP116 | K@100 |
|---|---|---|---:|---:|
| base | — | 26,160 B / 104.640 kbps / K 86.254 | 24,843 B / 99.372 kbps / K 85.900 | 85.942 |
| a1 | `AV2ENC_INTERP_BLOCK=1 AV2ENC_INTERP_BLOCK_CSF=2` | 25,826 B / 103.304 / 86.281 | 24,525 B / 98.100 / 86.284 | **86.283 (+0.341)** |
| a2 | a1 + `AV2ENC_INTERP_BLOCK_SKIP=1` | 25,845 B / 103.380 / 86.303 | 24,615 B / 98.460 / 86.254 | 86.269 (+0.327) |
| a3 | a1 + `AV2ENC_INTERP_BLOCK_SKIP=2` | 25,845 B / 103.380 / 86.303 | 24,615 B / 98.460 / 86.254 | 86.269 (+0.327) |

Envelope **86.283 vs 85.942 (+0.341), PASS** (three knob arms; a1 wins).
The guard is K-neutral on this clip (a2/a3 -0.014 against a1), so the gain is
the per-block switchable filter itself on the current base: at the same QP the
stream is 334 B (1.3 %) smaller and the K curve is flat (86.28 at both 103.3
and 98.1 kbps).

Source-replacement ceilings at the same bracket (base QP114/116, a1
QP114/116):

| segment | base | a1 | delta |
|---|---:|---:|---:|
| key f0 | 86.020 | **86.409** | +0.388 |
| f0-32 | 91.248 | **91.513** | +0.266 |
| f33-43 | 87.747 | **88.153** | +0.405 |
| f44-59 | 89.311 | **89.602** | +0.291 |

The f0-32 ceiling is above the 91.068 target on both arms, and the per-block
filter raises it another +0.266 rather than borrowing from the tail: f44-59 is
also +0.291. Every gate-on stream decodes `dav2d --strict 1` 60/60 via
`tools/validate_dav2d.sh` (a1 QP114/116, a2 and a3 QP114 checked). The a1
QP114 stream is 25,826 B / `55e5ef16…`; its `--video-recon` payload equals the
decoded payload byte for byte (10,368,399 B). `tools/test.sh --build
build-release` is 10/10 and the release build is clean with `-Werror`.

**Result.** KEEP by the runner gate. The guard stays default off and is not
the gain; the lane result is the per-block MC filter composition.

## Ledger

| # | target | patch | probe | K@100 | verdict |
|---:|---|---|---|---:|---|
| 1 | per-block MC filter (queue 2; GME/warp blocked) | `AV2ENC_INTERP_BLOCK` per-leaf switchable filter + `AV2ENC_INTERP_BLOCK_SKIP` EIGHTTAP skip/residual guard (default off) | QP114/116, live runner base + `AV2ENC_RD_DRIVER=1`; a1 86.283 vs base 85.942 | **86.283** | **KEEP** (+0.341) |
