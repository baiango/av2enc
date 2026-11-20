# Lane 5 — chroma under one J

Status: **LEAD** (+0.064 matched-rate K over the M4 UV search at 240p30 /
100 kbps, inside the ±0.1–0.2 ripple).  Gate `AV2ENC_RD_UVLEVEL`, env only,
default off; every gate-off pin is byte-identical and the M4 `AV2ENC_RD_UVFULL`
streams are byte-identical with the new gate unset.

Worktree `exp/rd-lane5`, base `315a7fbb`.  Reference recipe:

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1 \
  AV2ENC_TX_DETAIL_MIN=0.50 build-release/av2enc "$REF" -o /tmp/x.ivf \
  --video --frames 60 --cpu-threads 8 --qp <Q> --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

The base arm is that command.  The candidate arm adds `AV2ENC_RD_UVLEVEL=1`
(all other knobs at their measured defaults).

## 1. What is legal

The brief asked for four chroma levers.  Three of them are settled by the
syntax before any RD runs:

| lever | finding |
|---|---|
| **chroma TX size** | **not selectable.**  §5.20.7.25 `get_tx_size(plane>0, txSz)` is `Max_Tx_Size_Rect[get_plane_residual_size(ChromaMiSize, plane)]`; it ignores the luma TX size.  The M2 tree codes at most 64x64, so the chroma residual is at most 32x32 and resolves to **one** transform per plane.  `src/codec/transform/tx_partition/README.md` ("TX partitioning is luma-only"). |
| **chroma TX type** | **not selectable.**  §5.4.8 signals `enable_chroma_dctonly = 1` (`src/pipeline/video/video_headers.cpp:470`), which forces DCT_DCT for every chroma block. |
| **chroma TX partition** | **not selectable.**  Partition syntax/splits are luma-only (same README section); chroma keeps one transform over the chroma residual area. |
| **coefficient levels** | **selectable and legal.**  Any quantised level array is a valid chroma residual.  This is the lever this lane implements. |
| **CfL explicit alpha** | **unreachable in this writer.**  `code_uv_mode()` codes `cfl_alpha_signs/u/v` only under `Enable_MHCCP` (`src/pipeline/video/video_syntax_block.cpp:617`); without it the two-symbol `cfl_index` is the derived alpha only.  The nightly recipe has MHCCP off, so the derived-alpha CfL candidate already in the M4 search is the whole reachable set. |

So the milestone's chroma RD work is: *coefficient-level RD on every chroma
residual (intra and inter) under the same perceptual J, with the chroma lambda
calibrated by measurement.*

## 2. Design

New header `src/pipeline/video/video_uv_rd.h` plus a level stage in
`video_uv_search.h`.  The shared lane-1 exact-cost interface does not exist in
this worktree (`exp/rd-lane1` points at the same base commit), so the level
site uses the local documented interface below: `LevelParams` (geometry,
§7.14.2 plane step, §8.3 qctx, mask row) in, writer-ready `TxPlan` out.

```text
J = D_perc + lambda_site * R_exact

D_perc = eff_perc::weighted_sse(neutral Q8 weights, CSF gain, chroma weight)
         over the quantised reconstruction (both planes)
R_exact = uv_mode/is_cfl/cfl_index symbols + static §8.3 EOB, txb_skip,
          coeff_base, coeff_br, read_quant escape and sign information bits
          (`txtsearch::scan_coefficient_bits`, -log2 p of the §9.3 rows)
```

On the intra site R is exactly that static §8.2 information cost.  On the
inter site the level stage uses the leaf's own coefficient price (`coeff_bits`,
which under the nightly recipe is the R19 interval model over the same symbol
families), so the level decision and the inter skip/residual decision cannot
disagree about what a level costs.

Two sites, two calibrated weights (one objective, per-site lambda):

| site | value | how measured |
|---|---|---|
| mode / candidate selection | `leaf_rd_lambda(qindex) * eff_perc::lambda_scale()` (M4 value) | level-off sweep §5.1: lowering it only spends rate, K falls |
| level selection | that value × **0.125** | sweep §5.2; the static coefficient model overprices chroma levels |
| inter residual levels | the inter leaf lambda (`pipe_lambda`) × **0.125** | same calibration |

Candidate set per chroma plane, all scored on the same J form with the
level-site lambda:

1. the M4 scalar plan from `compute_tx` (round-to-nearest level);
2. three RDOQ trellis proposals (`codec/rate/rdoq`, the AV2 §8.3 model) at
   SSE weights `{0.5, 1, 2} × 0.10` of `step²`.

The trellis only *proposes*: the exact J accepts or rejects each proposal, so a
trellis model error can never win a block.  The winner replaces the plan the
leaf emits, so the coded coefficients are exactly the scored ones (previously
the search scored `compute_tx` and then the leaf recomputed the same scalar
plan).

Search shape (unchanged two-stage screen, level stage on the survivors):

1. stage 1 ranks the 13 §5.20.5.6 modes + derived-alpha CfL by prediction
   `D_perc + lambda*R_mode` (cheap, no transform);
2. stage 2 re-scores the top `AV2ENC_RD_UVFULL_TOPN` (3) on the real transform
   and, for each plane, keeps the cheapest of {scalar, trellis ×3} under the
   level-site lambda.  Only re-scored candidates can win.

The inter path applies the same per-plane candidate set to `res_u`/`res_v`
after its `compute_tx`, with the inter leaf's own distortion and lambda, so
P-frame chroma is covered too.

### Env surface (no CLI, no CMake, no options/parse/main/gate/pin edits)

| var | default | meaning |
|---|---|---|
| `AV2ENC_RD_UVLEVEL` | unset (off) | master gate; requires `AV2ENC_RD_UVFULL=1` |
| `AV2ENC_RD_UVLEVEL_LAMBDA` | 0.125 | level-site lambda multiplier (measured) |
| `AV2ENC_RD_UVLAMBDA` | 1.0 | mode-site lambda multiplier (calibration knob) |
| `AV2ENC_RD_UVLEVEL_INTER` | on when master on | also run the level stage on the inter chroma residual; explicit `0` is the key/intra-only ablation |
| `AV2ENC_RD_UVLEVEL_TRIES` | 3 | trellis proposals per plane (clamped 0..6) |
| `AV2ENC_RD_UVLEVEL_SCALE` | 0.10 | RDOQ SSE scale; only shapes proposals |
| `AV2ENC_RD_UVLEVEL_TOPN` | 3 | stage-2 mode count when the level stage is on |
| `AV2ENC_RD_UVLEVEL_STATS` | unset | one-line exit summary (blocks, proposals, flips) |

No CLI, CMake, `main.cpp`, options/parse or gate/pin file is touched.

## 3. Correctness

* Gate off: pins byte-identical — QP106 **25,008 B `0639b43630abb454f7be5ac494d2efb4`**,
  QP110 **23,690 B `1b99127a4cd1d386fa5600fb63c57a83`** (plain `--fast-decode`).
* `AV2ENC_RD_UVLEVEL` unset with `AV2ENC_RD_UVFULL=1` reproduces every M4
  stream byte for byte (the level stage is the only consumer of the new
  header; the `have_plans` reuse is gated on `level_variants > 0`).
* Gate on: every ladder stream decodes 60/60 with `tools/validate_dav2d.sh`
  (`--strict`), and the encoder `--video-recon` Y4M equals the decoded payload
  byte for byte at QP120.
* Release build (`-Werror`, `-O3`, PCH) and `dev-cache` produce byte-identical
  streams.

## 4. UV mode distribution and chroma bits (key picture, QP120)

`AV2ENC_RD_UVFULL_STATS` + `AV2ENC_DECISION_DUMP` (CfL counted from the
`is_cfl` counter):

| UVMode | base (M4 UVFULL) | candidate |
|---|---:|---:|
| DC_PRED (0) | 22 | 21 |
| V (1) | 7 | 5 |
| H (2) | 31 | 35 |
| D45 (3) | 10 | 9 |
| D135 (4) | 3 | 3 |
| D113 (5) | 6 | 12 |
| D157 (6) | 9 | 6 |
| D203 (7) | 11 | 7 |
| D67 (8) | 5 | 6 |
| SMOOTH (9) | 64 | 65 |
| SMOOTH_V (10) | 12 | 16 |
| SMOOTH_H (11) | 14 | 14 |
| PAETH (12) | 8 | 5 |
| **CfL** | 53 | 51 |

Key chroma coefficient bits (static §8.3, both planes):

| QP | base U / V | candidate U / V | Δ |
|---:|---:|---:|---:|
| 118 | 5,743.4 / 4,659.4 | 5,017.6 / 4,086.7 | −1,298 |
| 119 | 5,598.9 / 4,453.6 | 4,919.6 / 3,931.1 | −1,202 |
| 120 | 5,460.0 / 4,356.9 | 4,700.2 / 3,842.7 | −1,274 |
| 121 | 5,286.0 / 4,232.1 | 4,736.2 / 3,754.7 | −1,027 |
| 122 | 5,121.7 / 4,116.9 | 4,440.2 / 3,678.2 | −1,120 |

Stable **−12..−13%** of key chroma coefficient bits.  (The M4 UV search itself
accounts for −17%; this lane is the incremental step on top.)

Total-stream movement at QP120 (`AV2ENC_BIT_CENSUS`, same QP, streams
byte-identical to the measured ones):

| family | base bits | candidate | Δ |
|---|---:|---:|---:|
| residual (all) | 231,672 | 233,371 | +1,699 (+0.73%) |
| coeff_base | 164,017 | 164,406 | +389 |
| coeff_mag | 84,983 | 85,083 | +100 |
| eob | 30,939 | 31,293 | +354 |
| coeff_br | 26,341 | 26,327 | −14 |
| txb_skip | 7,882 | 8,203 | +322 |
| sign | 3,782 | 3,849 | +67 |
| tx_type / tx_partition | 11,885 | 12,319 | +434 |
| payload | 45,712 B | 45,907 B | +195 B |

The key chroma cut is real, but at a fixed QP the level stage *spends* that
saving on the P-frame residual and on the extra drift-refresh frame at QP120;
the gain is the matched-rate trade, not a free byte reduction.  (The per-frame
top delta is +138 B at frame 44, a refresh frame.)

## 5. Lambda calibration

### 5.1 Mode site

Level stage off, 2-QP brackets at 118..122, matched K @ 100 kbps:

| mode-lambda multiplier | matched K | Δ vs base |
|---:|---:|---:|
| 1.0 (M4) | **85.634** | — |
| 0.5 | 85.568 | −0.066 |
| 0.25 | 85.529 | −0.105 |
| 0.125 | 85.570 | −0.064 |

Lowering the mode-site lambda only spends rate on better prediction without a
K return, so the M4 value is the calibrated one.

### 5.2 Level site

Mode site 1.0, 2-QP brackets, inter level stage on:

| level-lambda multiplier | matched K | Δ |
|---:|---:|---:|
| 1.0 | 85.563 | −0.071 |
| 0.25 | 85.612 | −0.022 |
| 0.125 | **85.704** | +0.070 |
| 0.09 | 85.515 | −0.119 |
| 0.0625 | 85.684 | +0.050 |

The 2-QP bracket overstates the final 5-QP point (85.698, §6) by 0.006.  λ =
0.125 is the measured optimum; at 1.0 the static coefficient model over-zeroes
chroma (clip PSNR-U/V fall 1.22/0.88 dB for a 0.9% rate cut) and K drops.
That is the chroma analogue of the key's luma finding that the exact-cost
lambda needed ~0.25x; the chroma level site needs ~0.125x of the leaf lambda.

## 6. Matched-rate K

5 interleaved QPs (118..122), `tools/oracle.py --target-kbps 100`:

| QP | base B | kbps | K | cand B | kbps | K |
|---:|---:|---:|---:|---:|---:|---:|
| 118 | 26,165 | 104.660 | 85.933 | 25,732 | 102.928 | 86.012 |
| 119 | 25,330 | 101.320 | 85.837 | 25,205 | 100.820 | 85.789 |
| 120 | 24,760 | 99.040 | 85.487 | 25,005 | 100.020 | 85.704 |
| 121 | 24,471 | 97.884 | 85.115 | 24,537 | 98.148 | 85.103 |
| 122 | 23,997 | 95.988 | 85.024 | 23,641 | 94.564 | 84.956 |

| arm | matched kbps | K @ 100 | method |
|---|---:|---:|---|
| base | 100.000 | 85.634 | linear (base@120..base@119) |
| candidate | 100.000 | **85.698** | linear (cand@121..cand@120) |

**ΔK = +0.064.**  The base number reproduces the M4 reference exactly
(85.634), so the harness matches.

Iso-rate components (linear in rate, both at 100 kbps):

| | base | candidate | Δ |
|---|---:|---:|---:|
| PSNR-Y | 44.828 | 44.777 | −0.051 dB |
| PSNR-U | 48.133 | 48.076 | −0.056 dB |
| PSNR-V | 48.802 | 48.652 | −0.150 dB |
| CVVDP JOD | 9.7959 | 9.7994 | +0.0035 |
| Butteraugli | 2.1589 | 2.1364 | −0.0225 |
| SSIMULACRA2 | 82.699 | 82.595 | −0.10 |
| K | 85.634 | 85.698 | **+0.064** |

The candidate is slightly worse on every pixel metric except JOD and
Butteraugli, which is where the K composite earns the +0.064.  This is a
perceptual rate/quality trade, not a new prediction gain; it is also the
weakest kind of K evidence, which is why the verdict is LEAD rather than KEEP.

### 6.1 Fixed-QP behaviour

At QP118 the candidate is 433 B smaller and K is +0.079 at its own rate; at
QP120 it is 245 B larger and K is +0.217 at its own rate (100.02 vs 99.04
kbps); at QP121/122 it is within ±0.07 K.  The sign change with QP is the
drift-refresh schedule (the same mechanism that made the M4 UV search
estimator-sensitive, §6.1 of `docs/rd-chroma.md`): the candidate flips one
refresh frame on at QP120 and one off at QP122 relative to base.  A
refresh-off control ladder (QP116..120, both arms on the same fixed schedule)
is the cleaner read; not run in this time box.

### 6.2 Ablations

| arm | 5-QP matched K | Δ | note |
|---|---:|---:|---|
| base | 85.634 | — | M4 UVFULL |
| level stage, inter on | **85.698** | +0.064 | committed default |
| level stage, inter off | 85.699 | +0.065 | P-frame chroma level RD is K-neutral |
| stage-2 TOPN=8 (2-QP) | 85.728 | +0.094 | 4x encode time, q118 worse; kept at 3 |

The inter stage changes bytes at fixed QP but not K: it applies the same
calibrated J to P-frame chroma, which is the completeness half of "chroma
under one J".  It is on by default (the brief's scope) and can be ablated.

## 7. Quality detail

Key picture (frame 0) at QP120: PSNR-Y 45.862 both arms (unchanged);
PSNR-U 48.416 → 48.243, PSNR-V 49.071 → 48.749.  The key spends 13% fewer
chroma coefficient bits for −0.17/−0.32 dB of key chroma PSNR.

Per-frame PSNR-Y over the 60 frames: mean 44.833 → 44.825; worst frame 32
43.494 → 43.550 (better); best/key unchanged at 45.862.  Clip PSNR at QP120:
Y 44.783 → 44.779, U 48.031 → 48.078, V 48.711 → 48.651.  At matched rate the
whole-clip numbers are the iso-rate row in §6 (slightly lower PSNR, better
JOD/BA).

## 8. Verdict

**LEAD.**  The coefficient-level chroma RD is a real, spec-legal, decoder-exact
mechanism: −12..−13% key chroma coefficient bits for −0.05 dB clip chroma PSNR
(key chroma −0.17..−0.32 dB), and +0.064 matched-rate K through JOD and
Butteraugli.  Base 85.634, bar 85.784, candidate **85.698**: inside the
±0.1–0.2 drift-refresh ripple and short of the +0.15 KEEP bar, so it is a
LEAD, not a KEEP.

The measured ceiling is informative: key chroma coefficient bits are only ~5%
of the stream, so even a 13% cut is a ~0.6% rate saving (≈0.07 K).  Chroma
levels alone cannot close the remaining ~4.5 K to AVM; the chroma **prediction**
gain (M4 UVFULL, +0.337) was the large lever, and the remaining gap is luma
fine-detail error (`docs/rd-engine.md`).  A KEEP from this mechanism would need
either a better coefficient rate model (live neighbourhood contexts) or a
legal chroma TX/partition degree of freedom; neither exists today.

## 9. Blockers and follow-ups

* **CUDA backend not ported.**  The device path still codes DC + derived CfL
  and the scalar chroma transform; this CPU decision (and the smooth map) is
  not mirrored on device.
* **Explicit-alpha CfL** is unreachable without MHCCP; a MHCCP arm would need
  the whole sequence flag plus the explicit-alpha search and syntax plumbing.
* **Coefficient rate model.**  The stage uses static §9.3 rows with
  `txb_skip_context = 0` / `dc_sign_context = 0`; the writer derives both from
  the neighbour grids.  A live-context slice is the obvious next step and is
  also the likely reason the level site needs 0.125x.
* **Estimator sensitivity.**  The +0.064 point estimate comes from the
  specified bracket; the per-QP K sign changes with the refresh schedule.  A
  refresh-off control ladder would settle the point estimate.
* **Speed.**  The level stage is ~1.7–2x the M4 encode time at QP120 on the M4
  Mac (25 s vs 15 s, release, 8 threads).  TOPN=8 is ~4x and was not kept.
* Only 240p30 was measured for K; 720p was not scored in this time box.

## 10. Files and reproduction

| file | role |
|---|---|
| `src/pipeline/video/video_uv_rd.h` | new: env surface, `LevelParams`, `trellis_plan()` (RDOQ proposal + spec-exact dequant/inverse), counters |
| `src/pipeline/video/video_uv_search.h` | level stage inside stage 2; plans returned in `Result`; `search()` wrapper keeps the M4 call surface |
| `src/pipeline/video/video_frame_decision.cpp` | gate wiring for intra and inter chroma; emits the scored plans |
| `tools/lane-5/ladder.sh` | interleaved base/candidate 5-QP ladder |
| `docs/rd-lane5.md` | this document |

```sh
# final ladder (5 QPs, interleaved, stats on)
tools/lane-5/ladder.sh /tmp/l5/final

# matched K
python3 tools/oracle.py --ref testdata/nasa_480x240_30fps_60f.y4m \
  --target-kbps 100 --no-segments \
  --ivf base@118=/tmp/l5/final/base_q118.ivf ... \
  --ivf cand@122=/tmp/l5/final/cand_q122.ivf

# per-stream decode (60/60) and recon parity
tools/validate_dav2d.sh /tmp/l5/final/cand_q120.ivf
# encode with --video-recon, decode with dav2d, cmp the two Y4Ms
```
