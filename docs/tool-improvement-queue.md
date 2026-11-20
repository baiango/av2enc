# Tool-improvement queue: 10 measured attempts at the 240p / 100 kbps point

Status: **measured ledger, no patch kept.** Branch `exp/tool-loop`, base
`97257df5`. Every attempt below is either a contained env-gated code patch
(reverted when it lost) or a measurement of an already-gated, previously
unmeasured tool. Nothing was enabled in a preset, no default changed, no new
syntax was added, and no third-party code or asset was read. External encoders
and decoders are black-box binaries only.

K is the composite from `tools/kpi.py`, interpolated linearly in rate to
100.000 kbps across the candidate's own QP bracket. Decoding and CVVDP/BA/SS2
come from `tools/oracle.py` (AVM `avmdec` only, 8-bit 4:2:0). All rows are
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 30 fps.

## 0. Ledger

| # | target | attempt | K @100 | ΔK | verdict |
|---:|---|---|---:|---:|---|
| 1 | residual/TX/entropy | intra luma RDOQ (`AV2ENC_RDOQ_INTRA`) | 81.389 | **−0.760** | negative |
| 2 | filters | CDEF 4 strength sets (`--video-cdef-strength 4`) | 82.213 | +0.065 | negative (not sign-consistent) |
| 3 | allocation | low-band drift-refresh trigger (`AV2ENC_DRIFT_BAND`) | 83.218 | **−0.563** | negative |
| 4 | prediction | CfL (`--video-cfl`) | 81.652 | −0.497 | negative |
| 5 | transform | IST (`--video-ist`) | 82.140 | −0.008 | inert / negative |
| 6 | prediction | IBP (`--video-ibp`) | 81.329 | −0.820 | negative |
| 7 | transform | IDTX (`--video-idtx`) | 81.930 | −0.219 | negative |
| 8 | prediction | DIP (`--video-dip`) | 81.758 | −0.391 | negative |
| 9 | prediction/entropy | adaptive-MVD trio + quarter pel + BAWP off | 79.663 | −2.486 | negative |
| 10 | ME/skip (drift) | skip-non-zero-MV penalty (`AV2ENC_SKIP_MV_PENALTY=8`) | 81.977 | −0.172 | negative |

Baseline for every row is the local `--fast-decode` bracket QP104/106/108:
26,158 / 25,008 / 24,291 B, K 80.734 / 82.155 / 81.604, **matched 82.149**.
The tuned drift recipe (`--video-drift-refresh-threshold 1.5
--video-drift-refresh-budget 3250`, QP108/110/112) is the baseline for row 3
only: 26,196 / 25,146 / 24,668 B, **matched 83.781**.

## 1. Queue order and what was already closed

The brief's order is ME/subpel/skip → residual/TX/entropy → filters/allocation
→ eff-perc realisation. `impl-me` (`exp/impl-me`) is at the base commit with an
empty worktree, so it has produced no result; ME/subpel is therefore open, and
row 10 is the one attempt made in it.

Two of the luma-audit's three ranked mechanisms were already measured negative
on sibling branches before this loop started, and were not repeated:

* **Rank 1, frequency-weighted RDOQ/trellis** — `exp/luma-rdoq` (commits
  `402ca0bb`, `87089f29`): g = 0.2 gives 82.006, g = 0.1 gives 81.277 at
  iso-100 against 82.149; the eff-perc drift point is byte-identical for every
  g. `docs/video-rdoq-hf.md`.
* **Rank 2, frequency-aware TX size/type** — `exp/luma-tx` (commit `6f958ced`):
  `AV2ENC_TX_DETAIL_BIAS` buys +0.7 dB full-res on the key at matched 8x8
  low-pass but loses 0.39-0.92 K at matched 60-frame rate. `docs/tx-detail-bias.md`.
* The R09 exact-syntax/analytic-coefficient rate model is negative as well
  (`docs/rate-hybrid-s.md`: −2.06 K full, −0.56 K syntax-only).

So the residual/TX lane entered this loop with its top two mechanisms dead, and
row 1 tested the one coefficient layer still priced by plain SSE in a way the
luma-audit had not: **intra**.

## 2. Attempt 1 — intra luma coefficient trellis (negative)

**Patch.** `AV2ENC_RDOQ_INTRA` (default off) passes `allow_rdoq` at the four
intra luma call sites (`video_frame_intra.cpp` 437/510,
`video_frame_decision.cpp` 2324/2334) and drops the `is_inter` requirement in
the winner-only trellis (`video_tx_search.cpp`, `video_frame_internal.h`), so
the same `rdoq_quantize()` the inter path runs is applied to the winning intra
plan with the static §8.3 intra contexts. `AV2ENC_RDOQ_INTRA_SCALE` sets the
trellis SSE calibration for intra (the inter constant 0.1 is not the intra
optimum). Gate off is byte-identical.

**Keyframe probe** (`--frames 1 --qp Q`, bytes / PSNR-Y):

| scale | QP106 | QP100 |
|---|---|---|
| off | 19,752 / 48.484 | 21,739 / 49.398 |
| 0.1 | 19,365 / 48.232 | 20,967 / 48.980 |
| 0.05 | 19,405 / 48.274 | 21,261 / 49.231 |
| 0.02 | 19,588 / 48.384 | 21,497 / 49.365 |
| 0.005 | 19,704 / 48.472 | 21,620 / 49.419 |

The key RD curve is at best a tie: at 0.005/QP106 the key saves 0.24% for
−0.012 dB, and at QP100 the iso-rate reading is +0.05..0.08 dB. **60-frame
result:** scale 0.02 → 81.389 (−0.760), scale 0.005 → 81.338 (−0.811). Streams
24,821/25,882 B at QP106/104 against 25,008/26,158.

**Why it loses.** The key is the predictor for the static head, so its
distortion is not free: whole-stream PSNR drops 0.27 dB at QP106 while the key
saves 187 B, and BA rises 3.42 → 3.69. The trellis spends key distortion at a
rate the P chain cannot recover, exactly the allocation asymmetry
`docs/luma-audit.md` §4.3 describes.

**Verdict: negative.** Reverted. Do not wire intra RDOQ without a key-QP
re-plan that spends the saved bytes (out of scope: preset change).

## 3. Attempt 2 — CDEF multi-strength (negative, inside ripple)

`--video-cdef-strength 4` at QP104/106/108: 26,146 / 25,029 / 24,523 B, K
80.164 / 82.230 / 81.936, matched 82.213 (**+0.065**). Per point: QP104 −0.570,
QP106 +0.075, QP108 +0.332. The sign flips across the bracket, so it fails the
"sign-consistent" rule; the 4-strength index symbols (~2.9 B/frame) buy a BA
change that the CVVDP term gives back at QP104. **Negative**, no gate.

## 4. Attempt 3 — low-band drift-refresh trigger (negative)

**Patch.** `AV2ENC_DRIFT_BAND` (default off) replaces the full-frame luma PSNR
trigger with the PSNR of 4x4 block means (`reference_luma_band_psnr()`,
`video_frame_internal.h`) for both the key anchor and the per-frame reference,
keeping the same 1.5 dB threshold and 3,250 B budget. Trace prints both
metrics.

**Probe.** Full-frame anchor 47.73 dB, band anchor 58.25 dB. The band deficit
already exceeds 1.5 dB at f5 (55.99) while the full-frame deficit is 1.07 dB,
so the band arm refreshes one frame earlier and then keeps firing: the
QP-neutral repair cannot restore the band to within 1.5 dB (f6 is 54.49 after
the f5 refresh), so it spends the budget on the earliest frames.

**60-frame result** (drift recipe, QP108/110/112): 26,107 / 25,280 / 24,804 B,
matched **83.218** against base **83.781** → −0.563 K. **Negative**, reverted.

## 5. Attempts 4-9 — previously unmeasured legal tools (all negative)

These compose with plain `--fast-decode` and had no K measurement
(`docs/tool-matrix.md` §9). Each was screened with the same QP104/106/108
bracket and AVM oracle.

| tool | streams (B, QP104/106/108) | matched K | ΔK |
|---|---|---:|---:|
| `--video-cfl` | 25,944 / 24,794 / 24,322 | 81.652 | −0.497 |
| `--video-ist` | 26,165 / 25,019 / 24,297 | 82.140 | −0.008 |
| `--video-ibp` | 25,998 / 24,647 / 24,177 | 81.329 | −0.820 |
| `--video-idtx` | 26,127 / 24,918 / 24,220 | 81.930 | −0.219 |
| `--video-dip` | 26,404 / 25,624 / 24,751 | 81.758 | −0.391 |
| adaptive-MVD trio¹ | 26,113 / 25,436 / 24,747 | 79.663 | −2.486 |

¹ `--video-bawp-off --video-adaptive-mvd --video-flex-mvres
--video-mvd-sign-derive --video-mv-precision quarter`.

IST is a genuine no-op on this clip (+11 B at QP106, K flat to 0.01), so its
`--fast-decode` composition is inert rather than harmful. IDTX and DIP move
decisions but lose. The adaptive-MVD trio is dominated by its rescue strip:
removing BAWP costs 0.36 K and quarter-pel precision costs the rest (PSNR-Y
drops 2.0 dB at QP106), which confirms the `--video-bawp-off` rescue as the
blocker rather than the MVD tools themselves.

## 6. Attempt 10 — skip-non-zero-MV penalty (negative)

**Patch.** `AV2ENC_SKIP_MV_PENALTY=<bits>` (default 0) adds a fixed bit price to
the skip form of a leaf whose chosen MV is non-zero, in all three skip/residual
comparisons (historical, R09, pipe/eff-perc). The target is the static-head
mechanism in `docs/luma-audit.md` §4.5: 193 of 2,356 skip leaves at QP106 carry
a non-zero MV, and those resample the reference through subpel MC with no
residual.

**Probe and result.** At QP106 the penalty costs +7 B (8 bits), +250 B (32) and
+927 B (128). The 8-bit arm at QP104/106/108: 26,092 / 25,015 / 24,316 B, K
81.665 / 81.991 / 81.348, matched **81.977 (−0.172)**. Per-point: QP104 +0.931,
QP106 −0.164, QP108 −0.256 — the sign is not consistent, and the QP104 gain is
the same reference-chain lottery that produced the negative CDEF/AQ rows.

**Verdict: negative**, reverted.

## 7. What this closes, and what remains

* **Residual/TX/entropy is exhausted at this point.** Coefficient entropy is
  bounded at ≤0.3 K (`docs/keyframe-bit-accounting.md`), the HF trellis, TX
  detail bias, R09 rate model and now intra trellis are all measured negative.
  The remaining headroom is not a coefficient decision.
* **Filters/allocation**: the best allocation arm realises 6.0 K below its own
  ceiling (`docs/target-95.md` §4.1). The band-aware trigger does not close it
  because the repair is QP-neutral and the budget is fixed; closing it needs a
  *cost-aware* refresh planner (choose frames by measured refresh gain per
  byte), which is the `lpl_refresh` design already in the tree and measured
  flat at 240p (`AV2ENC_PIPE_LPL_REFRESH`, 83.9 K).
* **eff-perc realisation**: not attempted here because its knob space (λ, CSF,
  D-shape, rate term, chroma, top-N) is already swept to 85.10 and the guessed
  mechanisms (coefficient trellis, band trigger, skip-MV penalty) all measured
  negative in rows 1/3/10.
* **Blocked targets** (exact rejections, unchanged):
  `--video-adaptive-mvd` etc. reject with `does not compose with
  --video-bawp ...` (row 9 shows the rescue costs more than the tool);
  `--video-tcq` rejects with `--video-rdoq replaces the scalar quantiser ...`
  and costs ~60x for ~+0.01 JOD; `--video-tip` is `--video-edge-inter is
  measured on the low-delay P chain ...`; `--b-frames` is 45 K down
  (`docs/bframes.md`); `--video-sdp` is hard-blocked by `--video-intra-full`.

## 8. Verification

* Gate off, the tree is the base commit (all ten patches reverted); build is
  warning-free at the `release` preset (`-Werror` on).
* Pins byte-identical: 25,009 `a1d0dba1`; 25,141 `3984677b`; 23,690 `1b99127a`;
  24,113 `7f03eb6e`; 720p 36,572 `c4ba2813`; plus the allocation-run stream
  25,008 `0639b436`.
* `tools/test.sh --build build-release`: 10/10.

## 9. Reproduce

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
for q in 104 106 108; do
  build-release/av2enc $REF -o /tmp/b$q.ivf --video --frames 60 --qp $q --fast-decode
done
python3 tools/oracle.py --ref $REF \
  --ivf base@104=/tmp/b104.ivf --ivf base@106=/tmp/b106.ivf \
  --ivf base@108=/tmp/b108.ivf --target-kbps 100

# attempt 1 (keyframe probe)
AV2ENC_RDOQ_INTRA=1 AV2ENC_RDOQ_INTRA_SCALE=0.005 \
  build-release/av2enc $REF -o /tmp/k.ivf --video --frames 1 --qp 106 --fast-decode

# attempt 3
AV2ENC_DRIFT_BAND=1 build-release/av2enc $REF -o /tmp/d.ivf --video \
  --frames 60 --qp 110 --fast-decode --video-drift-refresh-threshold 1.5 \
  --video-drift-refresh-budget 3250 --video-trace
```
