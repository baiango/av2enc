
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

## Lane 10 focus

Start with: **Sequential tree edge-D / partition with the driver**. If blocked, continue down the queue. Do not duplicate another lane's focus.

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
# RD-engine continuous loop — lane 10

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

## Ledger

| # | target | patch | probe | K@100 | verdict |
|---:|---|---|---|---:|---|
