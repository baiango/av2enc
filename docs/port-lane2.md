# Lane-2 horizon term on the nightly / eff-perc base

Status: **ported and measured. KEEP at K=4 on the non-perceptual bases
(`--fast-decode` and the tuned drift recipe); do not combine it with
`AV2ENC_EFF_PERC` (it loses 0.26–0.50 K there).** The gate is default off, so
the pinned streams and the current best are untouched.

Branch `exp/port-lane2`, worktree `port-lane2`, built from the nightly/eff-perc
base (merge-base `996cc2f5`). All numbers on
`testdata/nasa_480x240_30fps_60f.y4m` (60 frames, 480x240, 30 fps), CPU
`build-dev-cache` (`-O1`, `--cpu-threads 8`), K interpolated with
`tools/oracle.py --target-kbps 100` (AVM decoder, CVVDP + Butteraugli +
SSIMULACRA2). `kbps = IVF bytes * 8 / 2 s / 1000`.

## 1. What was ported

The lane's tool is the frame-horizon cost term, not a new pipeline. Two pieces
came over:

- **Prediction primitives** (exact hunks, 93 lines, unchanged):
  `horizon_predict_zero_mv()` and `horizon_predict_mv()` with
  `HorizonRefPlanes`, in `src/pipeline/video/video_frame_me.cpp` and
  `src/pipeline/video/video_interp_filter.h`. `horizon_predict_mv` is the
  normative §7.13.3 `inter::mc::predict_block_420` call; the cache-copy
  primitive is decoder-exact at a zero-MV leaf. No lane pipeline type is
  needed for the port.
- **The term**, in the host's own leaf decision. The lane's
  `video_pipeline_horizon.cpp` prices every inter leaf as

  ```
  J_skip = sum_k w_k * SSE(source[i+k], prediction) + lambda * R_skip
  J_res  = sum_k w_k * SSE(source[i+k], recon)      + lambda * R_res
  w_0 = 1, w_k = 0.5^k, k = 0..K
  ```

  A skipped leaf copies its prediction into the reference, and every later
  picture that keeps skipping copies it again, so the k>0 terms are the error
  the choice *installs* in the next K pictures. The host already prices k=0
  (`pred_total` / `res_total`); the port adds the real future-source terms to
  both arms and leaves the rest of the decision order alone.

### Plumbing (all default-off)

- `src/pipeline/video/video_horizon_port.h` (new): the env gate, the K knob,
  `future_plane_sse()` (the source-extent clamp `source_sample()` uses) and
  `future_distortion()` (three planes, 0.5^k weights).
- `src/pipeline/video/video_frame_internal.h` / `video_frame.cpp`: the
  `InterFrameEncoder` constructor takes an optional
  `std::span<const VideoInputFrame> horizon_future`. The only call site that
  fills it is the flat P chain in `encode_video_sequence_impl()`
  (`src/pipeline/video/video_frame.cpp`), which passes
  `frames.subspan(i + 1)` — the actual source pictures after the current one.
  Every other construction site keeps the default empty span.
- `src/pipeline/video/video_frame_decision.cpp`: the inter-leaf skip/residual
  block computes `horizon_skip` / `horizon_res` and uses
  `pred_total + horizon_skip` and `res_total + horizon_res` in all three
  comparisons (the historical bar, the `want` bar, and the pipe-perc rate
  objective). `horizon_skip`/`horizon_res` are exactly 0 when the gate is off.
- No CLI, CMake, `main.cpp`, options/parse, gate-tool or pin edit. 8-bit 4:2:0
  only by construction (the term reads `VideoInputFrame::y/u/v` in the 4:2:0
  decision TU).

```
AV2ENC_PORT_HORIZON=1          # gate, default off
AV2ENC_PORT_HORIZON_K=2|4|8    # horizon depth, default 2
```

### Gate-off identity

| Stream | Bytes | MD5 |
|---|---:|---|
| `--fast-decode --qp 106` pin | 25,008 | `0639b43630abb454f7be5ac494d2efb4` |
| `--fast-decode --qp 110` pin | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` |
| `AV2ENC_PORT_HORIZON=0 --fast-decode --qp 106` | 25,008 | `0639b436…` (identical to unset) |
| flag-less `--qp 189` pin | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` |

`ctest --test-dir build-dev-cache`: 12/12. `tools/gate_video_nightly.sh` and
`src/pipeline/video/gate_video_eff_perc.sh`: OK.

## 2. Matched-rate result

Each arm is its own QP ladder around 100 kbps (the tool spends ~7–10% more bits
at a fixed QP on `--fast-decode`, 0.4–1.7% on eff-perc, so the matched point
moves).
`K @ 100` is the oracle's linear interpolation over the bracketing pair.

| base | arm | QPs | matched kbps (points) | K @ 100 | Δ vs base |
|---|---|---|---:|---:|---:|
| `--fast-decode` | base | 104/106/108 | 100.032 / 97.164 | **82.149** | — |
| `--fast-decode` | `=1`, K=2 | 110/111 | 100.504 / 98.520 | 81.968 | **−0.181** |
| `--fast-decode` | `=1`, K=4 | 111/112 | 100.268 / 98.400 | **83.286** | **+1.137** |
| `--fast-decode` | `=1`, K=8 | 111/112 | 100.324 / 99.264 | 83.005 | +0.856 |
| eff-perc (`AV2ENC_EFF_PERC=1` + drift 1.5/3250) | base | 118/119/120 | 100.920 / 98.696 | **85.099** | — |
| eff-perc | `=1`, K=2 | 120/121 | 100.760 / 97.120 | 84.634 | −0.465 |
| eff-perc | `=1`, K=4 | 119/120 | 101.868 / 99.824 | 84.837 | −0.262 |
| eff-perc | `=1`, K=8 | 120/121 | 101.112 / 96.996 | 84.595 | −0.504 |
| tuned drift (no eff-perc, drift 1.5/3250) | base | 109/110/111 | 100.584 / 99.800 | **83.795** | — |
| tuned drift | `=1`, K=4 | 113/114 | 102.292 / 99.884 | 84.078 | +0.283 |
| tuned drift | `=1`, K=8 | 114/115 | 100.596 / 98.116 | 84.065 | +0.270 |

The tuned-drift base was added because the eff-perc recipe contains it; it
reproduces the documented 25,146 B / K 83.795 point exactly.

### The nightly gain is not QP ripple

The nightly K curve is jagged (the drift-refresh decisions are a cliff):

| QP | 104 | 106 | 107 | 108 |
|---|---:|---:|---:|---:|
| base kbps | 104.632 | 100.032 | 99.540 | 97.164 |
| base K | 80.734 | 82.155 | 80.508 | 81.604 |

K=4 at the same rates:

| QP | 110 | 111 | 112 | 113 |
|---|---:|---:|---:|---:|
| kbps | 101.932 | 100.268 | 98.400 | 97.828 |
| K | 82.541 | 83.298 | 83.216 | 82.546 |

The candidate sits +1.1 to +1.8 K above the base curve at *every* nearby rate,
not just at one lucky point (at 100.3 kbps: 83.298 vs 82.155, +1.14; at
101.9 kbps: 82.541 vs ≈81.3; at 98.0 kbps: 83.2 vs ≈81.4). The base's own best
point (QP 106) is the comparison, so the gain is conservative.

On the tuned drift base the delta (+0.28) is at the edge of the same ripple:
the candidate's four bracket points are 84.13/83.98/84.08/83.97 versus the base
83.21/83.94/83.75, so the sign is consistent but the size is small.

## 3. Where the bits and decisions move

`AV2ENC_BIT_CENSUS` / `AV2ENC_DECISION_DUMP` + `tools/bit_census.py` (the dumps
do not change the bitstream; the dumped streams keep the same MD5 as the
undumped ones).

### Nightly, same QP 106 (mechanism)

The term charges the persistent skip error, so leaves stop skipping:

| | base | K=4, same QP |
|---|---:|---:|
| inter leaves | 2,906 | 2,905 |
| skipped inter leaves | 2,356 (81.1%) | 2,104 (72.4%) |
| skip flips (matched leaves) | — | 252 × `1→0`, 41 × `0→1` |
| residual info bits | 110,512 | 124,178 (+13,667) |
| payload (tile) | 23,022 B | 25,288 B (+2,266 B) |

### Nightly, matched rate (base QP 106 vs K=4 QP 111)

Information bits: residual +3,944 (+3.57%), `raw_bits` −3,422 (−5.52%),
`inter_mode` −344, `partition` −89, `intra_mode` +115, `skip` symbol +298.
The key is coded ~1,040 B cheaper (the matched point is a coarser picture) and
the tail frames each buy 130–225 B of repairs: the top deltas are frames 35,
38, 37, 53 (+160…+224 B), while frames 25 and 0 lose 172 B and 1,340 B. Skip
share at matched rate: 81.1% → 78.3%.

### Tuned drift, matched rate (base QP 110 vs K=4 QP 114)

Residual +1,598 bits, `raw_bits` −2,970, `inter_mode` −286; key −1,039 B,
frames 43/45/44/49/52 each +130…+190 B. Skip share 47.6% → 49.9% (this base
already codes many residuals through its refresh frames, so the tool's effect
here is redistribution, not a skip collapse).

### Per-frame PSNR-Y (decoded with the pinned `dav2d`)

Nightly, matched rate (base 106 @100.03 vs K=4 111 @100.27):

| segment | base | K=4 | Δ |
|---|---:|---:|---:|
| key f0 | 48.48 | 47.62 | −0.86 |
| f1–32 | 46.54 | 45.93 | −0.61 |
| f33–43 | 43.26 | 43.49 | +0.23 |
| f44–59 | 43.12 | 43.67 | +0.55 |
| all | 45.06 | 44.91 | −0.15 |

The base over-spends on the head (f1–32 is already 46.5 dB while the tail is
43.1) and the tool moves the rate to the tail. The composite oracle prefers
that even though mean PSNR is flat-to-slightly-down: at the matched point
Butteraugli 3.420 → 3.016 and SSIMULACRA2 80.84 → 82.18.

eff-perc, base 119 @100.92 vs K=4 120 @99.82:

| segment | base | K=4 | Δ |
|---|---:|---:|---:|
| key f0 | 46.05 | 45.86 | −0.19 |
| f1–32 | 44.92 | 44.81 | −0.11 |
| f33–43 | 43.57 | 43.44 | −0.13 |
| f44–59 | 44.28 | 43.95 | −0.33 |

Every segment is worse, and skip share *rises* (54.3% → 59.0%): the plain-SSE
future term is priced against a perceptually weighted current distortion, so
on blocks whose weighted current error is already high the added unweighted
tail term can favour the skip. This is why the arm loses on eff-perc despite
winning on the plain path.

Tuned drift, base 110 @100.58 vs K=4 114 @99.88: key −0.72, f1–32 −0.49,
f33–43 −0.25, f44–59 −0.09 dB; skip share 47.6% → 49.9%. The gain is SS2-driven
(81.25 → 81.92) at equal BA/JOD.

## 4. Decode validation

Every candidate bracket stream (22 in total: nightly K=2/4/8, eff-perc
K=2/4/8, tuned drift K=4/8) passes `tools/validate_dav2d.sh` with exit 0 and
the IVF header's 60/60 pictures decoded. The base pins decode 60/60 as before.

## 5. Verdict

**KEEP (default off), K=4.**

- The mechanism transfers to the non-perceptual skip decision: +1.14 K over the
  `--fast-decode` base, +0.28 K over the tuned drift recipe — both clear the
  +0.15 K bar, and the nightly bracket is consistent across four rates.
- It does **not** transfer onto `AV2ENC_EFF_PERC`: −0.26 K at K=4, −0.47/−0.50
  at K=2/K=8, with every segment PSNR lower and skip share rising. The
  perceptual base already prices the skip correctly; the added plain-SSE
  future term double-counts in the wrong units.
- K=2 is the unstable arm on `--fast-decode` (81.97, −0.18): the decaying sum with
  only one future picture does not price the tail. K=4 is the best arm on
  every non-perceptual base; K=8 is within noise of K=4 and more expensive.
- This does not move the current best (85.297 tail-TX 3-point, eff-perc-based):
  applied to eff-perc it loses. It is a real +0.28 K on the best non-perceptual
  recipe, not a new ceiling.

## 6. Limits / blockers

- Not wired for the perceptual rate objective in matching units. The honest
  next step would be to price the future term with the same leaf weights the
  eff-perc D uses; the plain-SSE port is what was measured and it loses.
- 8-bit 4:2:0 flat-P single-tile only (the term lives in the 4:2:0 decision
  TU and the future span is passed only by `encode_video_sequence_impl`).
  B/temporal/TIP/refinement paths keep the default empty span.
- The horizon assumes the decoder copies this block forward under skip. On a
  non-zero MV or a refresh frame the future error is not exactly this block's
  prediction error, so the term is an estimate there (the lane made the same
  approximation; its MV guard is in the removed pipeline, not in the term).
- Cost: one extra K-frame luma+chroma SSE per inter leaf. Measured wall time at
  QP 111, 8 threads: 4.83 s base vs 4.72 s at K=8 (the extra SSE is inside the
  noise of the encode).
