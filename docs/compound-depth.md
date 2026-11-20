# Compound depth on the chained B (2026-11-21)

**Verdict: the CWP depth arm (87.023) plus the exact compound-rate probe
(87.284) is +0.479 K over the TIP 86.805 pin at 100 kbps, decoder-exact, and
promoted into `--nightly-b`.** The first win is the CWP-weighted NEAR_NEARMV
pair search, not more MV pairs: the smallest pair set (one stack pair) is the
best point.  The second win (§7) is pricing the compound symbol set exactly
(+0.261); the signallable compound NEWMV modes are +0.054 and stay off.  The
env knobs stay for ablation, and the controls below restore the 87.023 depth,
86.805 TIP, 86.607 compound-only and 85.472 single-prediction arms byte for
byte.

This doc starts from `exp/tip-b 8d694853` (chained B + AVERAGE compound + TIP
dead ends). The baseline is the TIP arm:

```sh
env AV2ENC_TIP_LEAF=1 build-dev/av2enc "$REF" -o base.ivf \
  --video --frames 60 --qp 121 --cpu-threads 8 --video-gops 7 --nightly-b
```

## 1. Candidate set

The existing compound candidate is `GLOBAL_GLOBALMV` with zero MVs (AVERAGE).
The depth arm adds one `NEAR_NEARMV` candidate pair from the two §7.12.2
per-list stacks:

* list 0 takes `RefStack0Mvs[0]`, list 1 takes `RefStack1Mvs[0]`; this is the
  nearest spatial / previous-frame / zero-MV entry for each reference.
* the pair is priced by the existing exact-rate / BA-shaped leaf RD (EFF_PERC
  in `--nightly-b`), not by a separate motion search.
* AVERAGE and the five CWP weights (`{8,12,4,10,6}` opposite-side row,
  `{8,12,4,20,-4}` same-side row) are searched per pair.  The implicit mask
  (`--video-imp-mask`) is selectable but **not** in the promoted arm; see §2.
* the pair set is capped by `AV2ENC_COMPOUND_PAIR_LIMIT`. N=1 (one pair) is
  the measured point; N=2..5 and the full 5x5 set are neutral or worse.

Arbitrary per-reference MVs (the plan ME MV and its ±1/8..1-pel refinements)
are **not signallable** in `NEAR_NEARMV`: the decoder derives both MVs from the
per-list stacks (`RefMvIdx0/1` only). A compound `NEAR_NEWMV` / `NEW_NEARMV` /
`JOINT_NEWMV` syntax would be needed to code an arbitrary pair; that is not
wired in this tree. The task's "plan's MV + refinement" candidate therefore
becomes "the first stack pair per list" here. The measured result says the
weight search, not additional pairs, is what moves K.

> **2026-11-21 update:** §7 wires the three modes and measures them. They are
> decoder-exact but below the +0.15 gate (87.077 vs the 87.023 depth arm).
> Pricing the compound syntax exactly, which the modes need, is the win
> (87.284, +0.261). The paragraph above is the depth-arm record.

The historical proxy also priced a compound block with the single-prediction
mode/DRL formula, leaving `comp_mode`/`is_joint`/`compound_mode`/
`comp_group_idx`, the second DRL index and the compound interp-filter symbol
free. `AV2ENC_COMPOUND_RATE=1` charges them (a ~10-bit compound syntax term);
`AV2ENC_COMPOUND_RATE_SCALE` scales the term. It is worth about +0.01 K at the
matched point, i.e. inside the QP ripple, but it is kept because it makes the
pair/weight search self-consistent.

## 2. Measured result

60 pictures, first 60 frames of `nasa_480x240_30fps_60f.y4m`,
`tools/oracle.py --target-kbps 100`, AVM decode:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| TIP base | 121 | 25,299 | 101.196 | 9.8679 | 2.0013 | 85.391 | 86.901 |
| TIP base | 122 | 24,918 | 99.672 | 9.8562 | 2.0237 | 85.407 | 86.778 |
| **depth** | **123** | **25,105** | **100.420** | **9.8666** | **1.9673** | **85.694** | **87.047** |
| depth | 124 | 24,653 | 98.612 | 9.8660 | 1.9768 | 85.256 | 86.942 |
| **depth, matched** | — | — | **100.000** | — | — | — | **87.023** |

Matched K **87.023**, **+0.218** over 86.805. The gain is BA/SS2
(1.967/85.69 vs 2.001/85.39 at the rate-matched point) with JOD flat.

Per-segment source ceilings (`oracle.py` without `--no-segments`): the arm
improves the key (+0.081 vs +0.082), f0-32 (+4.442 vs +4.257) and f33-43
(+1.862 vs +1.779), and is slightly lower on f44-59 (+3.538 vs +3.936). This
is a pooling result, not an encodable stream.

Pair-limit sweep at QP123 (CWP + guard + rate):

| pair limit | bytes | K @ matched |
|---:|---:|---:|
| 1 | 25,105 | **87.023** |
| 2 | 25,139 | 87.019 |
| full (5x5) | 25,139 | 87.019 |

The implicit mask (`--video-imp-mask`) selects no different coding block on
this clip, but it does change the §7.13.3.14 blend: the 16-picture stream and
the 60-picture QP123/124 streams re-key at the same byte lengths (16 f
`88082f06…` vs the measured `d6a55a0b…`).  It is therefore **not** in the
promoted arm; the 87.023 point and its pins are the arm without it.  Adding
`--video-imp-mask` explicitly is still byte-count-neutral but needs its own
pins/K if it is ever promoted.  OPFL/refine-MV are not measured on the
winning carrier because §5.20.7.6 skips `cwp_idx` for `use_optflow`/
`use_refinemv` and the code rejects the pairing; on the unweighted
NEAR_NEARMV carrier they were +0.079 / +0.002 without TIP, below the gate
(`docs/tlayers-chain.md`). A 2-4 candidate wedge/diffwtd mask search was not
run: the existing masked search on the global-global carrier was -0.074 on
the compound-legal base.

## 3. TIP/compound field defect (workaround)

The arm needs `AV2ENC_TIP_LEAF_FIELD_GUARD=3`. Without it, a 60-picture
stream whose reference §7.22 fields contain compound blocks with motion or
CWP is **not** encoder/decoder-exact even though every coded reference picture
matches:

* unguarded TIP + `--video-compound-near-near`, QP121: display 20 differs in
  35 luma samples, max 2 LSB; every other picture is exact. The 16-picture
  proof is exact.
* unguarded TIP + `--video-compound-near-near --video-cwp-weighted`, QP121:
  displays 30 and 42 differ, max 31 LSB; QP122: displays 4, 30 and 42.
* the pinned `avmdec` is the oracle; `--video-recon` is the encoder compare.

The defect is in the TIP field path, not the coded block syntax: the
reference pictures are exact, and the mismatch is only at TIP output pictures
that consume a field containing compound cells. Forcing
`TipInterpFilter = EIGHTTAP_SMOOTH` made the unguarded NEAR_NEARMV stream exact
at 60 pictures; the field-store list-content probes (drop list 0/1, mirror,
average, drop the block) all failed, so this is not a single §7.22 list rule.
The guard is therefore a workaround, not a root-cause fix.

`AV2ENC_TIP_LEAF_FIELD_GUARD=1` rejects a TIP whose reference slot has any
compound field cell with a nonzero MV. Mode 3 also rejects a slot with a
CWP-weighted compound block. The rejected picture falls through to the normal
tile-coded B; every TIP leaf is a dead end and the CDF chain resets per
picture, so the fallback is byte-identical to the non-TIP chain. Mode 3 is the
measured arm.

This is a **negative result for the soft field**: do not remove the guard
without re-running the 60-picture avmdec recon check, and do not treat the
unguarded CWP K (87.363) as a valid point.

## 4. Proof

16 pictures (one key + two chained 8-frame B blocks), QP116.  `--nightly-b`
now expands this arm (mode 1, 0.30 dB floor, field guard 3, pair limit 1,
compound syntax pricing, NEAR_NEARMV + CWP); the explicit form below is the
same stream:

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
# promoted preset
build-release/av2enc "$REF" -o /tmp/d16.ivf --video --frames 16 --qp 116 \
  --cpu-threads 8 --video-gops 2 --nightly-b --video-recon /tmp/d16.y4m

# equivalent explicit arm
env AV2ENC_TIP_LEAF=1 AV2ENC_TIP_LEAF_FIELD_GUARD=3 \
    AV2ENC_COMPOUND_RATE=1 AV2ENC_COMPOUND_PAIR_LIMIT=1 \
  build-release/av2enc "$REF" -o /tmp/d16.ivf --video --frames 16 --qp 116 \
  --cpu-threads 8 --video-gops 2 --nightly-b \
  --video-compound-near-near --video-cwp-weighted \
  --video-recon /tmp/d16.y4m
python3 tools/validate_avmdec.py /tmp/d16.ivf --ref "$REF" \
  --recon /tmp/d16.y4m
tools/bin/dav2d-macos --strict 1 --threads 1 -i /tmp/d16.ivf \
  -o /tmp/d16.dec.y4m
cmp /tmp/d16.y4m /tmp/d16.dec.y4m
```

Result: 19,825 B, MD5 `d6a55a0bd7cf11c5c4686cfe73b77330`; `avmdec` 16/16
recon-equal; pinned `dav2d f9dafbc` 16/16 sample-exact. The checked-in
`src/pipeline/video/gate_video_compound_depth.sh` runs this proof, the
60-picture avmdec check and the control pins; it passes 8/8 on the hardened
release build.

60 pictures QP123/124:

```sh
build-release/av2enc "$REF" -o /tmp/d123.ivf --video --frames 60 --qp 123 \
  --cpu-threads 8 --video-gops 7 --nightly-b --video-recon /tmp/d123.y4m
python3 tools/validate_avmdec.py /tmp/d123.ivf --ref "$REF" \
  --recon /tmp/d123.y4m
```

QP123 is 25,105 B `291f361a…`; QP124 is 24,653 B `d0e374fa…`. Both are
`avmdec` 60/60 recon-equal. The pinned `dav2d` still emits 17/60 on the
60-picture chains (the known §7.21.6 drain defect); it is unchanged by this
arm.

## 5. Controls

The preset is the measured exact-rate arm; every earlier point stays
byte-exact:

| control | bytes | MD5 |
|---|---:|---|
| `--nightly-b` (87.284, QP121) | 25,163 | `dfa6b1aadd565a2adf47c7e186c1a973` |
| `AV2ENC_COMPOUND_EXACT_RATE=0` (87.023 depth, QP123) | 25,105 | `291f361abdc8583ada9e71c769a7de53` |
| `--nightly-b --video-compound-depth-off` (86.805 TIP, QP121) | 25,299 | `ca314db24a71fc1c28d7a1ba8eb1515f` |
| depth off + `AV2ENC_TIP_LEAF=0` (86.607 compound, QP125) | 24,988 | `3dbbad9847ab661c6c154903990ade75` |
| `--nightly-b --video-compound-off` (85.472, QP131) | 24,911 | `a4c8853e3abd752503db17be769b80a4` |
| `--fast-decode` QP106 | 25,008 | `0639b43630abb454f7be5ac494d2efb4` |
| `--fast-decode` QP110 | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |

An explicit env override still wins, including `0`, so the arm stays
inspectable: `AV2ENC_COMPOUND_PAIR_LIMIT=2` restores the neutral larger pair
set, `AV2ENC_TIP_LEAF_FIELD_GUARD=0` is the unguarded (not recon-exact) probe,
`AV2ENC_COMPOUND_RATE=0` drops the heuristic compound term, and
`AV2ENC_COMPOUND_NEWMV=1` enables the three sub-gate NEWMV modes.

## 6. What would change the answer

1. **Root-cause the TIP/compound field divergence.** The guard costs about
   600 B at QP123 (the unguarded CWP stream is 25,148 B at the same QP and
   scores 87.363, but it is not recon-exact). If the field/TIP path is fixed,
   re-measure without the guard; that is the next +0.3 K, not more pairs.
2. **A compound NEWMV mode.** It is the only way to code the plan ME pair and
   ±1/8..1-pel refinements. The current stack pair is enough here, so measure
   this only after the field fix.
3. **Mask search.** A 2-4 candidate wedge/diffwtd search on the winning pair
   is unmeasured; the global-global masked arm was negative, so it needs the
   exact-rate pricing above to be worth the code.

## 7. Compound NEWMV modes and exact compound pricing (2026-11-21)

This section is the follow-up to item 2. The three §5.20.7.6 compound NEWMV
modes are now signallable and decoder-exact, but the mode set is below the
gate. The exact *price* of the compound syntax, which the modes need and the
depth arm never had, is +0.261 K on the same candidate set and is the part
worth keeping.

### 7.1 Mode design

The candidate set is unchanged: the two per-list §7.12.2 stacks and the
two-list (skip-mode form) stack, first entry each, `AV2ENC_COMPOUND_PAIR_LIMIT=1`.
On top of it the arm adds, per block:

* `NEAR_NEWMV`: list 0 = first per-list stack entry (near), list 1 = plan ME MV
  or its ±1/8..1-pel refinement (new). §5.20.7.6 `has_second_drl()` codes
  `RefMvIdx0` and `RefMvIdx1`; the single MVD is list 1's.
* `NEW_NEARMV`: mirror. Both predictors are list 0/1 of the two-list stack at
  one `RefMvIdx`; the single MVD is list 0's.
* `JOINT_NEWMV` (`is_joint = 1`): both predictors come from the two-list stack
  entry; `jmvd_scale_mode = 0`; one MVD is coded on the §5.20.7.13 base list
  (`joint_mvd_plan` from the two FrameDistance values) and `projMv` derives the
  other list.
* The search seeds are the better of the plan ME MV (`run_me` per reference,
  the same cached search the single-reference NEWMV candidate uses) and the
  zero-delta stack predictor. The refinement is an on-axis
  `{±1/8, ±1/4, ±1/2, ±1}` search around that seed, scored by the **compound**
  blend SSE, not each reference's single-prediction SSE.
* `NEAR_NEWMV` / `NEW_NEARMV` code no `cwp_idx` (§5.20.7.6: only
  `NEAR_NEARMV` and `JOINT_NEWMV` with `jmvd_scale_mode == 0` are eligible), so
  their prediction is the equal-weight AVERAGE (or its derived implicit mask).
  `JOINT_NEWMV` runs the existing CWP weight search.
* MVD, both DRL indices, `jmvd_scale_mode`, `compound_mode_non_joint`,
  `is_joint`, `comp_group_idx`, `cwp_idx` and the compound interp-filter symbol
  are coded in the writer and priced by the eff-perc `SyntaxProbe`
  (`AV2ENC_COMPOUND_EXACT_RATE`).
* **CWP index alphabet fix.** The probe capped `cwp_idx` at 3, but
  §5.20.7.6 codes 0..4: four binary symbols in unary form (a `1` continues,
  `0` ends), so index 4 is legal and was being rejected as a probe degrade.
  A degraded probe silently falls back to the heuristic price on exactly the
  leaves that chose the fifth weight.  The exact-rate arm accepts 0..4, which
  is part of why the same candidate set moves +0.261.

The carrier is env-gated (`AV2ENC_COMPOUND_NEWMV=1`, with
`AV2ENC_COMPOUND_NEWMV_MODES` bit 1/2/4 and `AV2ENC_COMPOUND_NEWMV_RADIUS` /
`..._REFINE` for ablations). The TIP field guard stays at
`AV2ENC_TIP_LEAF_FIELD_GUARD=3`; the unguarded stream is still invalid.

### 7.2 Proof

16 pictures, QP116, `--video-gops 2 --nightly-b --video-compound-near-near
--video-cwp-weighted`:

| arm | bytes | MD5 | avmdec | pinned dav2d |
|---|---:|---|---|---|
| exact-rate (no new modes) | 19,774 | `51b664515a897a3b19be4131790867fa` | 16/16 equal | 16/16 exact |
| NEWMV modes + exact rate | 19,800 | `424abbe40d8040fcac2eedf78bf8046a` | 16/16 equal | 16/16 exact |

The NEWMV trace selects all three modes on the 16-picture proof
(NEAR_NEWMV 4, NEW_NEARMV 5, JOINT_NEWMV 32 leaves). The 60-picture streams are
`avmdec` 60/60 at every measured QP for both arms. The pinned `dav2d` is still
17/60 on the 60-picture chains (the known §7.21.6 drain defect).

### 7.3 Measured result

60 pictures, first 60 frames of `nasa_480x240_30fps_60f.y4m`,
`tools/oracle.py --target-kbps 100`, AVM decode. The depth pin is the 87.023
matched point from §2.

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| depth (heuristic price) | 123 | 25,105 | 100.420 | 9.8666 | 1.9673 | 85.694 | 87.047 |
| **exact rate, stack pair** | **121** | **25,163** | **100.652** | **9.8815** | **1.8613** | **85.817** | **87.472** |
| exact rate, stack pair | 122 | 24,859 | 99.436 | 9.8719 | 1.9610 | 85.869 | 87.122 |
| exact rate, stack pair | 123 | 24,598 | 98.392 | 9.8655 | 1.9730 | 85.753 | 87.034 |
| NEWMV modes + exact rate | 123 | 25,235 | 100.940 | 9.8704 | 1.9660 | 85.874 | 87.100 |
| NEWMV modes + exact rate | 124 | 24,904 | 99.616 | 9.8710 | 1.9509 | 85.387 | 87.068 |
| NEWMV modes + exact rate | 125 | 24,628 | 98.512 | 9.8748 | 1.9516 | 85.207 | 87.055 |

Matched at 100.000 kbps:

| arm | matched K | Δ vs 87.023 | verdict |
|---|---:|---:|---|
| exact rate, stack pair | **87.284** | **+0.261** | **KEEP** |
| NEWMV modes + exact rate | 87.077 | +0.054 | drop (< +0.15) |
| depth (heuristic price, stack pair) | 87.023 | 0 | control |

Interpolated at 100.000 kbps the exact-rate arm is JOD 9.8764 / BA 1.9148 /
SS2 85.845; the NEWMV arm is JOD 9.8708 / BA 1.9553 / SS2 85.528. The NEWMV
modes spend +624 B at QP125 (24,628 vs 24,004) and lose 0.207 K at matched
rate: the plan ME MVs do not pay for their MVDs once the price is exact.

Per-segment source ceilings for the exact-rate arm at the QP122 point
(99.436 kbps): key +0.083, f0-32 +4.294, f33-43 +1.798, f44-59 +3.691. They
are a pooling bound, not an encodable stream. The NEWMV arm at QP123:
+0.088 / +4.461 / +1.865 / +3.526.

### 7.4 Why the modes are negative

The exact probe shows the new candidates lose on rate: a compound NEWMV leaf
pays `comp_mode + is_joint + compound_mode_non_joint + jmvd_scale_mode + DRL +
shell MVD` where `NEAR_NEARMV` pays two DRL indices and no MVD. The first-stack
pair is already close to the plan ME MV on this clip, so the MVD term is pure
overhead. The depth arm's win was the CWP weight search; `NEAR_NEWMV` /
`NEW_NEARMV` cannot code CWP at all, and the CWP-capable `JOINT_NEWMV` still
loses the rate trade. This is the item-2 prediction ("the current stack pair
is enough here") measured.

Do not enable `AV2ENC_COMPOUND_NEWMV` in a measured aggregate. The useful
change is `AV2ENC_COMPOUND_EXACT_RATE=1`, which improves the depth arm's
pair/CWP decisions without adding a mode; it is promoted into `--nightly-b`
(87.284), with `AV2ENC_COMPOUND_EXACT_RATE=0` as the 87.023 control.
`gate_video_compound_exact.sh` pins the exact-rate 16/60 proof, the
signallable NEWMV proof and the controls; it passes 21/21 on the hardened
release build.
