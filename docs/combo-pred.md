# Prediction/ME combination search (`exp/combo-pred`)

Status: **measured; no novel prediction/ME combination wins.** Twenty-two
60-frame flag sets plus a three-arm TIP lane were built, hashed and screened at
QP106/108 against `--nightly`. Nothing outside the known BAWP-off marginal is
positive; the best source-replacement ceiling is **88.555**, below the 88.8
gate, so no arm was promoted to a QP104/106/108 ladder. No encoder code, tool or
preset was changed.

- Branch: `exp/combo-pred`, base `0bf1d001` (main).
- Binary: `build-release/av2enc` (release preset built
  from `0bf1d001`). The `combo-pred` worktree has no build directory, so every
  stream came from that binary; no other binary or worktree was used.
- Oracle: `tools/oracle.py` with the pinned AVM decoder (`avmdec`); K is the
  `tools/kpi.py` composite. Rate is `ivf_bytes * 8 / 2.0 s` (60 frames at 30 fps).
- Fixture: `testdata/nasa_480x240_30fps_60f.y4m`.
- Gate-off pins are unaffected: this branch adds documentation only.

## 1. Method

Baseline is the current nightly pin:

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m -o nightly.ivf \
    --video --frames 60 --cpu-threads 8 --qp 106 --nightly
# 25,008 B / 100.032 kbps / JOD 9.7334 / BA 3.4202 / SS2 80.837 / K 82.155
```

Every candidate holds `--video --frames 60 --cpu-threads 8 --qp Q --nightly`
and changes **only prediction/ME flags**. Transform, quantisation, entropy,
loop filters and rate control stay at the nightly expansion; the flags that had
to be turned off to reach a legal carrier are the nightly-implied prediction
ablation switches (`--video-bawp-off`, `--video-cdf-chain-off`,
`--video-edge-inter-off`, `--video-edge-mask-off`, `--video-loop-filters-off`,
`--video-tx-search-off`, ...); their cost is part of the measured set and is
reported as such.

Screen: QP106 and QP108, compared with `--nightly` at the **same QP**. A gain is
reported only when both QPs have the same sign; the QP106→108 baseline itself
moves 0.551 K (nightly K 82.155 → 81.604). Matched-rate K at 100 kbps is a
linear interpolation over the two points where they bracket; otherwise the
oracle marks it `nearest` and the table says so.

Ceiling: the oracle's source-replacement segment rows (the rate-free ceiling).
Per the task gate, an arm whose ceiling cannot clear 88.8 gets no full ladder.

The 8-frame TIP lane is separate: `--video-tip` only accepts 3..8 pictures and
rejects the nightly filter/TX/probe/RDOQ/CDF surface by name. It is scored
against an 8-frame cut of the fixture (`ffmpeg -frames:v 8`, same 480x240
4:2:0) and is **not** comparable to the 60-frame wall.

## 2. Registry

`F` is the frame count; the canonical hash is SHA-1 of
`av2enc --video --frames F --cpu-threads 8 <flags>` (input, output and QP are
point parameters, not architecture). A duplicate flag set would show a repeated
hash; there are none.

| id | F | SHA-1 | exact flags beyond `--video --frames F --cpu-threads 8 --qp Q` | status |
|---|---:|---|---|---|
| `nightly` | 60 | `67adea12d5d382b2caee38ef56cf6bbf314b707d` | `--nightly` | baseline |
| `ctl-bawpoff` | 60 | `f89d61c29d4e223c117817da2bb98b9008db5cfc` | `--nightly --video-bawp-off` | control (known ablation) |
| `amvd` | 60 | `781e0b8cc0f16f98f3f6ccccf47e10ac0183f51f` | `--nightly --video-bawp-off --video-adaptive-mvd` | measured |
| `amvd-q` | 60 | `33305299929652e5f764c478a71d641ae9ecd116` | `--nightly --video-bawp-off --video-adaptive-mvd --video-mv-precision quarter` | measured |
| `amvd-drl` | 60 | `fff92a3e303fb3f458dd2cc9419f70336f28fbbb` | `--nightly --video-bawp-off --video-adaptive-mvd --video-drl-reorder-always` | measured (combined best) |
| `amvd-q-drl` | 60 | `7aa355e0c86f852ecae0fff0b0017bccefb2fcf0` | `--nightly --video-bawp-off --video-adaptive-mvd --video-mv-precision quarter --video-drl-reorder-always` | measured |
| `amvd-flex` | 60 | `43514f6d6aebd8fc670f1cfa97daaa04ff27cd2e` | `--nightly --video-bawp-off --video-adaptive-mvd --video-flex-mvres --video-mv-precision quarter` | measured |
| `amvd-flex-sign` | 60 | `5df7adad052ad78516fbe58d989bb3263c5c9688` | `--nightly --video-bawp-off --video-adaptive-mvd --video-flex-mvres --video-mv-precision quarter --video-mvd-sign-derive` | measured |
| `mvd-sign` | 60 | `befa570c428ca04e4df5891e320acf3df9045241` | `--nightly --video-bawp-off --video-flex-mvres --video-mv-precision quarter --video-mvd-sign-derive` | measured |
| `flex-cdfoff` | 60 | `4bfd5b91f5c7864269770daaaf329984aaa9bb74` | `--nightly --video-bawp-off --video-flex-mvres --video-cdf-chain-off` | measured |
| `flex` | 60 | `e77458ed159195836d2565231a5ae7dfbcb977ad` | `--nightly --video-bawp-off --video-flex-mvres` | **decoder-invalid** |
| `flex-q` | 60 | `d1d31390e529c997505cba5dfb2790a324a5cbc4` | `--nightly --video-bawp-off --video-flex-mvres --video-mv-precision quarter` | measured |
| `drl-always` | 60 | `8b1cbe70f2dd6a3fc59f85a67f757faaa663d825` | `--nightly --video-drl-reorder-always` | measured |
| `interintra` | 60 | `bcf7575b06d5cdf2af6d06c04a2bc76ba7d4f9b9` | `--nightly --video-bawp-off --video-cdf-chain-off --video-compound-interintra` | measured |
| `interintra-q` | 60 | `d56fb73cba32f9e4474e5c319653a4cab29483ec` | `--nightly --video-bawp-off --video-cdf-chain-off --video-compound-interintra --video-mv-precision quarter` | measured |
| `bank` | 60 | `850d188f108bee0e2ead116c8f4cf92df250b638` | `--nightly --video-cdf-chain-off --video-ref-signal-off --video-refs 2 --video-refmvbank` | measured |
| `refmvs-bank` | 60 | `f929c5ef884d69303cda3ba0a49207d563e5b03e` | `--nightly --video-edge-inter-off --video-edge-mask-off --video-cdf-chain-off --video-ref-signal-off --video-refs 2 --video-ref-mvs --video-refmvbank` | measured |
| `refmvs-traj` | 60 | `269d5a961b6871c0b69388e6c0ffe4909917f6e6` | `--nightly --video-edge-inter-off --video-edge-mask-off --video-cdf-chain-off --video-ref-signal-off --video-refs 3 --video-bawp-off --video-ref-mvs --video-mv-traj` | measured |
| `ibc` | 60 | `5f534b5320eaa13c178556b3548a5716353fc8d1` | `--nightly --video-cdf-chain-off --video-ibc` | measured |
| `palette` | 60 | `bc15cf41b405a60fb58d634f56ea610e978e1dcd` | `--nightly --video-cdf-chain-off --video-tx-search-off --video-palette` | measured |
| `morph` | 60 | `45ac73b6c285ba79db68a6ff4abfb59cf9f7ffc9` | `--nightly --video-loop-filters-off --video-cdf-chain-off --video-ref-signal-off --video-tile-group-off --video-morph` | measured |
| `implicit` | 60 | `ac8bbcb3838fa253b084446cf3d9e79247c72101` | `--nightly --video-implicit-ref-map --video-ref-signal-off --video-cdf-chain-off` | measured |
| `tip` | 8 | `7a4df352635f4d6c4eb4517bb3b4a107fe644b87` | `--video-tip` | TIP lane |
| `tipref` | 8 | `beb574f9194b6966da689c74c5469567ff0b34d3` | `--video-tip --video-ref-mvs --video-tip-block-ref --video-force-single` | TIP lane |
| `ctl-tip` | 8 | `c8e380edd0928a98ce43a6c79ff8964ce37bfc1d` | (plain P chain, no `--video-tip`) | TIP control |

## 3. Two-QP screen (60 frames)

`ΔK` is against `--nightly` at the same QP. `sign` requires both QPs to agree.
Points that do not bracket 100 kbps are still valid screen points; they are not
rate-matched.

| id | QP106 kbps | K106 | ΔK106 | QP108 kbps | K108 | ΔK108 | sign |
|---|---:|---:|---:|---:|---:|---:|---|
| `nightly` | 100.032 | 82.155 | — | 97.164 | 81.604 | — | — |
| `ctl-bawpoff` | 100.204 | 82.357 | **+0.202** | 97.684 | 82.034 | **+0.430** | ++ |
| `amvd` | 100.268 | 82.356 | **+0.201** | 97.472 | 81.992 | **+0.388** | ++ |
| `amvd-drl` | 100.120 | 82.259 | +0.104 | 98.268 | 81.802 | +0.198 | ++ |
| `drl-always` | 99.876 | 81.892 | −0.263 | 97.052 | 81.316 | −0.288 | −− |
| `amvd-flex` | 101.772 | 79.414 | −2.741 | 99.276 | 79.808 | −1.796 | −− |
| `amvd-flex-sign` | 101.744 | 79.413 | −2.742 | 98.988 | 79.808 | −1.796 | −− |
| `amvd-q` | 101.728 | 80.719 | −1.436 | 99.244 | 80.680 | −0.924 | −− |
| `amvd-q-drl` | 101.740 | 80.719 | −1.436 | 98.516 | 80.275 | −1.329 | −− |
| `mvd-sign` | 102.108 | 79.343 | −2.812 | 97.656 | 79.619 | −1.985 | −− |
| `flex-cdfoff` | 112.864 | 80.413 | −1.742 | 110.868 | 79.872 | −1.732 | −− |
| `flex-q` | 102.188 | 79.258 | −2.897 | 97.656 | 79.619 | −1.985 | −− |
| `interintra` | 113.320 | 80.317 | −1.838 | 110.476 | 79.533 | −2.071 | −− |
| `interintra-q` | 113.692 | 79.513 | −2.642 | 111.256 | 79.229 | −2.374 | −− |
| `bank` | 111.332 | 80.127 | −2.028 | 109.328 | 79.549 | −2.055 | −− |
| `refmvs-bank` | 160.760 | 76.743 | −5.412 | 156.164 | 69.979 | −11.625 | −− |
| `refmvs-traj` | 163.548 | 79.457 | −2.698 | 157.172 | 79.001 | −2.603 | −− |
| `ibc` | 111.484 | 80.046 | −2.109 | 108.568 | 79.719 | −1.885 | −− |
| `palette` | 90.300 | 75.170 | −6.985 | 90.988 | 75.120 | −6.484 | −− |
| `morph` | 128.880 | 75.972 | −6.183 | 123.944 | 75.790 | −5.814 | −− |
| `implicit` | 115.348 | 64.698 | −17.457 | 112.184 | 62.712 | −18.892 | −− |
| `flex` | 100.244 | — | — | 97.712 | — | — | decoder-invalid |

`flex` (`--video-flex-mvres` at the nightly eighth-pel frame precision) encodes
but both decoders reject frame 6:

```text
avmdec: Warning: Failed to decode frame 6: Corrupt frame detected
        Failed to decode tile data
dav2d:  Error decoding frame: Invalid argument
```

With `--video-cdf-chain-off` the same reconstruction set decodes 60/60
(`flex-cdfoff`, −1.74/−1.73) but the CDF carry is worth ~4.5 K, so the legal
form is not competitive. `flex-q`, `mvd-sign` and the `amvd-flex*` arms (quarter
frame precision) decode 60/60, so the trigger is the **eighth-pel frame
precision + per-block coarser precision + CDF carry** combination, not
`--video-flex-mvres` alone.

### Component detail (all 60-frame arms)

| id | JOD106 | BA106 | SS2-106 | JOD108 | BA108 | SS2-108 |
|---|---:|---:|---:|---:|---:|---:|
| `ctl-bawpoff` | 9.7082 | 3.2397 | 80.776 | 9.7107 | 3.3993 | 80.563 |
| `amvd` | 9.7081 | 3.2393 | 80.769 | 9.7048 | 3.4199 | 80.713 |
| `amvd-drl` | 9.7283 | 3.3701 | 81.063 | 9.7236 | 3.5905 | 80.825 |
| `drl-always` | 9.7181 | 3.4754 | 80.313 | 9.7248 | 3.8357 | 80.322 |
| `amvd-flex` | 9.5674 | 4.3819 | 78.499 | 9.5816 | 4.1292 | 78.278 |
| `amvd-flex-sign` | 9.5674 | 4.3819 | 78.498 | 9.5809 | 4.1203 | 78.220 |
| `amvd-q` | 9.6734 | 3.9284 | 79.165 | 9.6623 | 3.8879 | 78.878 |
| `amvd-q-drl` | 9.6733 | 3.9284 | 79.165 | 9.6544 | 4.1141 | 78.761 |
| `mvd-sign` | 9.5615 | 4.3832 | 78.266 | 9.5897 | 4.2407 | 77.876 |
| `flex-cdfoff` | 9.6895 | 4.3123 | 80.254 | 9.6838 | 4.6143 | 79.619 |
| `flex-q` | 9.5591 | 4.4284 | 78.200 | 9.5897 | 4.2407 | 77.876 |
| `interintra` | 9.6919 | 4.3468 | 79.895 | 9.6617 | 4.7351 | 79.169 |
| `interintra-q` | 9.6576 | 4.6989 | 78.901 | 9.6426 | 4.8143 | 78.512 |
| `bank` | 9.7013 | 4.5474 | 80.100 | 9.6790 | 4.8608 | 79.677 |
| `refmvs-bank` | 9.5354 | 5.8065 | 73.296 | 9.1600 | 11.2477 | 63.487 |
| `refmvs-traj` | 9.6787 | 4.6903 | 77.863 | 9.6746 | 4.9598 | 77.236 |
| `ibc` | 9.7053 | 4.6157 | 80.026 | 9.6963 | 4.7984 | 79.721 |
| `palette` | 9.4148 | 6.7425 | 72.388 | 9.4085 | 6.7456 | 72.290 |
| `morph` | 9.4382 | 6.1769 | 73.652 | 9.4157 | 6.2265 | 73.497 |
| `implicit` | 8.4649 | 11.8823 | 54.003 | 8.4215 | 17.3165 | 51.478 |

`flex` is absent: the stream has no valid decoded pictures (see the note above).
PSNR-Y and the full per-frame components are in the oracle JSONs; the K column
of the screen table is computed from these three components.

### Matched-rate K

Only arms whose measured points bracket 100 kbps get a linear matched-rate
oracle. `nearest` rows are not matched-rate candidates.

| id | method | matched kbps | K | vs nightly (same 2-point method) |
|---|---|---|---:|---:|
| `nightly` | linear 108→106 | 100.000 | 82.149 | — |
| `ctl-bawpoff` | linear 108→106 | 100.000 | 82.331 | +0.182 |
| `amvd` | linear 108→106 | 100.000 | 82.321 | +0.172 |
| `amvd-drl` | linear 108→106 | 100.000 | 82.229 | +0.080 |
| `drl-always` | nearest (QP106, 99.876) | 99.876 | 81.892 | −0.257 (same QP) |
| `amvd-flex` | linear | 100.000 | 79.693 | −2.456 |
| `amvd-flex-sign` | linear | 100.000 | 79.663 | −2.486 |
| `mvd-sign` | linear | 100.000 | 79.474 | −2.675 |
| `flex-q` | linear | 100.000 | 79.433 | −2.716 |
| `amvd-q` | linear | 100.000 | 80.692 | −1.457 |
| `amvd-q-drl` | linear | 100.000 | 80.479 | −1.670 |

The other arms sit above 100 kbps at both screen QPs (`bank` 109+, `interintra`
110+, `morph` 124+, `palette` 91 with a non-monotone point) and are reported as
same-QP deltas only; they are already 1.7–18.9 K down at the better QP, so a
matched-rate ladder cannot rescue them.

### Best-arm detail

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nightly` | 106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | 82.155 |
| `nightly` | 108 | 24,291 | 97.164 | 9.7216 | 3.6824 | 80.633 | 81.604 |
| `ctl-bawpoff` | 106 | 25,051 | 100.204 | 9.7082 | 3.2397 | 80.776 | 82.357 |
| `ctl-bawpoff` | 108 | 24,421 | 97.684 | 9.7107 | 3.3993 | 80.563 | 82.034 |
| `amvd` | 106 | 25,067 | 100.268 | 9.7081 | 3.2393 | 80.769 | 82.356 |
| `amvd` | 108 | 24,368 | 97.472 | 9.7048 | 3.4199 | 80.713 | 81.992 |
| `amvd-drl` | 106 | 25,030 | 100.120 | 9.7283 | 3.3701 | 81.063 | 82.259 |
| `amvd-drl` | 108 | 24,567 | 98.268 | 9.7236 | 3.5905 | 80.825 | 81.802 |
| `drl-always` | 106 | 24,969 | 99.876 | 9.7181 | 3.4754 | 80.313 | 81.892 |
| `drl-always` | 108 | 24,263 | 97.052 | 9.7248 | 3.8357 | 80.322 | 81.316 |

`amvd` and `ctl-bawpoff` are the same experiment twice: QP106 differs by 0.001 K
and QP108 by −0.042 K. Adaptive MVD is exactly neutral; the whole +0.2/+0.4 is
the known BAWP-off marginal. `amvd-drl` adds the DRL_ALWAYS sort and is
**−0.10/−0.23 against its own BAWP-off control**, so its +0.08 matched over
nightly is also the BAWP-off effect minus the reorder cost.

## 4. Ceilings and the 88.8 gate

Source replacement of one segment is the oracle's rate-free ceiling. Best row
per arm (all four segments are in the JSON):

| arm | best source segment | ceiling K | ΔK |
|---|---|---:|---:|
| `nightly` | f44-59 | 88.066 | +5.911 |
| `amvd` | f44-59 | 88.457 | +6.101 |
| `ctl-bawpoff` | f44-59 | 88.457 | +6.100 |
| `amvd-drl` | f44-59 | 88.295 | +6.036 |
| `drl-always` | f44-59 | **88.555** | +6.664 |

**Gate result: no arm clears 88.8. The best ceiling in the lane is 88.555
(`drl-always`, f44-59), 0.245 short.** Per the task gate, none of the consistent
two-QP positives (`ctl-bawpoff`, `amvd`, `amvd-drl`) was promoted to a full
QP104/106/108 ladder; QP104 was deliberately not encoded. The matched-rate
numbers above are the two-point (106/108) interpolation and are labelled as
such. A full ladder would not change the verdict: the same-mechanism ceiling is
already below the wall at the operating point.

These ceilings are pooling bounds, not streams. They do not say a better
allocator could not spend the same bits differently; they say this prediction
flag set cannot. They are also sufficient for the gate: the largest
source-replacement boost measured here is +6.664 K, so an arm needs a base K of
at least 82.14 to have any chance at 88.8; the only arms at or above that are
the five in the table (nightly 82.155, `ctl-bawpoff` 82.357, `amvd` 82.356,
`amvd-drl` 82.259, and `drl-always`, measured at 81.892 but included). Every
other arm is 1.7–18.9 K lower and cannot reach the gate from the same segment
boost.

## 5. Legal / illegal combinations

Exact CLI first line for one-factor additions to `--nightly` (2-frame usage
run, `--video --frames 2 --qp 106 --cpu-threads 8`). Rejections are by design;
nothing was patched.

| addition | result |
|---|---|
| `--video-gme` / `--video-warp` | reject: `--video-gme/--video-warp are gated on the plain skip-only P chain: they cannot be combined with --b-frames, --video-tlayers, a frame-type wrapper, --video-compound, --video-inter-residual or --video-drl-reorder` (nightly's `quality` preset supplies `--video-inter-residual`) |
| `--video-ref-mvs` | reject: `--video-edge-inter cannot carry the temporal MV field: the bottom SVB rows have no coded MI grid` |
| `--video-refmvbank` | reject: `reference-management tools are not wired with the average-CDF/CDF-continuity chain` |
| `--video-mv-traj` | same CDF-chain rejection; also needs `--video-ref-mvs` |
| `--video-same-ref-compound` | same CDF-chain rejection on the nightly carrier; on the compound carrier it is legal only for AVERAGE (see below) |
| `--video-compound` | reject: `--video-bawp is gated on the plain single-prediction P chain: ... --video-compound ...` |
| `--video-compound-wedge` / `--video-compound-diffwtd` | reject: `--video-compound-wedge/--video-compound-diffwtd need --video-compound` |
| `--video-compound-near-near` | reject: `--video-compound-near-near needs --video-compound` (and then needs `--b-frames`) |
| `--video-opfl` / `--video-refine-mv` | reject: `--video-opfl/--video-refine-mv need --video-compound-near-near: the temporal refinement tools only run on a NEAR_NEARMV compound block` |
| `--video-morph` | reject: `--video-morph does not compose with the loop filters: the IBC copy/morph template must read the unfiltered reconstruction` (nightly supplies CDEF/CCSO/deblock) |
| `--video-palette` | reject: `--video-palette uses per-frame §9.3 palette/IBC CDFs; it is not wired with --avg-cdf/--cdf-continuity` |
| `--video-implicit-ref-map` | reject: `--video-ref-signal-compact already selects the derived §7.7 map; drop --video-implicit-ref-map` |
| `--video-adaptive-mvd` | reject: `--video-adaptive-mvd does not compose with --video-compound, --video-compound-interintra, --video-bawp, ... (§5.20.7.6 symbol order)` |
| `--video-flex-mvres` | same §5.20.7.6 rejection (BAWP) |
| `--video-mvd-sign-derive` | same §5.20.7.6 rejection (BAWP) |
| `--video-tip` | reject: `--video-edge-inter is measured on the low-delay P chain; drop the B-frame / reorder / temporal-layer mode`; after removing every named blocker the stream still fails decode (below) |

Decoder-level rejections found on otherwise CLI-legal sets:

| set | rejection |
|---|---|
| `flex` = nightly + `--video-bawp-off --video-flex-mvres` | `avmdec`: `Failed to decode frame 6: Corrupt frame detected / Failed to decode tile data`; `dav2d`: `Error decoding frame: Invalid argument`. Legal with `--video-cdf-chain-off`, 60/60 (`flex-cdfoff`). |
| `implicit` before `--video-cdf-chain-off` | `avmdec`: `Failed to decode frame 18: Corrupt frame detected / Failed to decode tile data`. With the chain off it decodes 60/60 but scores K 64.70/62.71. |
| `refmvs-traj` before `--video-bawp-off` | `avmdec`: `Failed to decode frame 35: ... Inter BAWP template cannot outside the valid reference range`. With BAWP off it decodes 60/60. |
| nightly + TIP (all named blockers removed) | `avmdec`: `Failed to decode frame 2: ... Invalid value for EOB position token`; `dav2d` likewise rejects. Plain `--video-tip` (no nightly prediction tools) decodes 8/8, so TIP is only valid on its plain carrier. |
| `--video-same-ref-compound` + masked compound | `--video-same-ref-compound codes the AVERAGE blend only; masked compound is not wired for a same-reference pair` |
| `--video-mv-precision half` | CLI-legal; the AVM decoder produced 36/60 pictures in the earlier audit (`docs/luma-audit.md`), so it was not screened |

Legal carriers that were measured (all decode 60/60 unless marked):

| set | result |
|---|---|
| `--video-bawp-off --video-compound --video-refs 2 --video-ref-signal-off --video-drift-refresh-off --video-cdf-chain-off` | legal, decodes; prior control/tool row in `docs/reference-structure.md` (−0.17 K vs its own control) |
| `--video-compound --video-same-ref-compound` (AVERAGE only) | legal; prior −7.2 K (`docs/reference-structure.md`), not repeated |
| `--video-ibp` | legal on nightly; prior K 81.25 at 98.59 kbps (`docs/k-sweep.md`), not repeated |
| `--video-mv-precision quarter` | legal; prior K 79.73 (`docs/luma-audit.md`), not repeated |
| `--video-ref-mvs` with the edge path off | legal; prior +0.09 K over its blocker control but the blocker costs 4.71 K (`docs/reference-structure.md`), not repeated |
| `--video-tip --video-ref-mvs --video-tip-block-ref --video-force-single` | legal 8-frame carrier; measured here (`tipref`) |

Tool-matrix targets asked for by the parent, in one place:

- `--video-bawp-off --video-adaptive-mvd --video-flex-mvres
  --video-mvd-sign-derive --video-mv-precision quarter` is legal
  (`amvd-flex-sign`) and scores 79.663 matched (−2.49 vs nightly). Dropping
  `--video-mvd-sign-derive` (`amvd-flex`) changes 0.030 K; the whole flex
  family is negative.
- `--video-same-ref-compound`: legal only on the plain AVERAGE compound
  carrier, not with wedge/diff masks (exact rejection above); prior measurement
  −7.2 K (`docs/reference-structure.md`), not repeated.
- `--video-ibp`: legal on `--nightly`; prior K 81.25 at 98.59 kbps
  (`docs/k-sweep.md`), not repeated.
- `--video-mv-traj` / `--video-ref-mvs` / `--video-refmvbank` with the
  compact-map blocker handled: `bank` −2.03/−2.06, `refmvs-bank`
  −5.41/−11.63, `refmvs-traj` −2.70/−2.60. The compact-map switch
  (`--video-ref-signal-off`) is legal but costs the CDF chain
  (`--video-cdf-chain-off`).
- The matrix itself is `docs/tool-matrix.md` on main (`exp/luma-matrix`); it is
  not present in this branch's checkout, so the flags were audited against
  `src/cli/validate.cpp` and the live binary.

## 6. TIP lane (8 frames, out of lane)

`--video-tip` is the only untested prediction class in the matrix, but it only
accepts 3..8 pictures and rejects the nightly loop filters, CDF chain, TX
search, partition probe, RDOQ, edge path, drift refresh, ref-signal compact,
tile-group compact and key-QP offset by name. The legal carrier is therefore
the **plain** P chain, not the nightly lane. Scored against an 8-frame cut with
the same oracle:

| arm | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tip` | 106 | 23,869 | 716.070 | 8.5598 | 24.4599 | 46.473 | 61.609 |
| `tip` | 108 | 22,731 | 681.930 | 8.4629 | 24.4199 | 44.948 | 60.874 |
| `tipref` | 106 | 21,213 | 636.390 | 7.6277 | 9.6479 | 41.436 | 59.308 |
| `tipref` | 108 | 20,958 | 628.740 | 7.6114 | 9.7909 | 40.304 | 58.971 |
| `ctl-tip` (plain P, no TIP) | 106 | 49,922 | 1497.660 | 8.6003 | 8.8682 | 59.002 | 67.488 |
| `ctl-tip` | 108 | 46,920 | 1407.600 | 8.4984 | 8.6723 | 56.933 | 66.740 |

These are not rate-matched to the control (TIP is 2.1x smaller at the same QP),
so the comparison only says TIP at 8 frames is far below the 60-frame wall and
its prediction is much worse at low rate (BA 24.46). `tipref` saves another 11%
rate and costs 2.3 K at the same QP. A 60-frame TIP stream is not expressible
by the CLI (3..8 pictures), so this lane cannot be a wall candidate at this
operating point.

## 7. Verdict

- **No novel prediction/ME combination beats its own legal control.** Once the
  mandatory blocker is the control, adaptive MVD contributes **exactly zero**
  (82.356 vs 82.357 at QP106, 81.992 vs 82.034 at QP108). Every prediction
  addition that needs no blocker removed is negative at both screen QPs
  (`drl-always` −0.26/−0.29 is the best of them). `amvd-drl` is +0.08 matched
  over nightly but −0.10/−0.23 against its own BAWP-off control, i.e. the
  reorder is a cost and the rest is the blocker.
- The only consistent positives are `--video-bawp-off` (+0.20/+0.43) and
  `--video-adaptive-mvd` on top of it. BAWP-off is the known ablation, already
  recorded as +0.29 (unresolvable) at the plain nightly point and −0.36 at the
  tuned-drift point; it is not a new architecture and does not clear the wall.
- **Best internal combination:** `--nightly --video-bawp-off` at 2-point
  matched K **82.331** (100.0 kbps), i.e. +0.18 over the nightly 2-point
  interpolation. It is inside the documented QP ripple and below the 88.8
  gate. **Best novel combination:**
  `--nightly --video-bawp-off --video-adaptive-mvd --video-drl-reorder-always`
  at 2-point matched K **82.229**, which is its own control minus the reorder
  cost; plainly, no novel architecture won. The best novel one-factor is
  adaptive MVD, and its contribution over the BAWP-off control is zero.
- **Oracle vs 88.8:** matched K 79.4–82.3 for the 60-frame arms and an 88.555
  best source ceiling. Nothing comes within 6.5 K of the wall at matched rate.
- No full QP104/106/108 ladder was run: every arm's ceiling is below 88.8, and
  the gate forbids promoting it. The raw oracle JSONs live in the experiment's
  scratch directory; all numbers above come from them.

## 8. Reproduction

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=/path/to/build-release/av2enc

# baseline
"$BIN" "$REF" -o /tmp/n106.ivf --video --frames 60 --cpu-threads 8 \
  --qp 106 --nightly
# best internal arm (known BAWP marginal, not a novel architecture)
"$BIN" "$REF" -o /tmp/b106.ivf --video --frames 60 --cpu-threads 8 \
  --qp 106 --nightly --video-bawp-off
# combined best (adaptive MVD is neutral; DRL_ALWAYS costs 0.10-0.23)
"$BIN" "$REF" -o /tmp/c106.ivf --video --frames 60 --cpu-threads 8 \
  --qp 106 --nightly --video-bawp-off --video-adaptive-mvd \
  --video-drl-reorder-always
# screen
python3 tools/oracle.py --ref "$REF" \
  --ivf nightly@106=/tmp/n106.ivf --ivf amvd@106=/tmp/c106.ivf \
  --target-kbps 100 --json /tmp/screen.json
```

The TIP lane uses the same fixture cut to 8 frames:

```sh
ffmpeg -v error -i "$REF" -frames:v 8 -pix_fmt yuv420p -f yuv4mpegpipe /tmp/ref8.y4m
"$BIN" "$REF" -o /tmp/tip8.ivf --video --frames 8 --qp 106 --cpu-threads 8 --video-tip
python3 tools/oracle.py --ref /tmp/ref8.y4m --ivf tip@106=/tmp/tip8.ivf \
  --target-kbps 100 --json /tmp/tip8.json
```
