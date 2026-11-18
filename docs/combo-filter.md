# Combo filter / rate-allocation search at 240p30 / 100 kbps

Status: **measured; no architecture clears SVT 88.8.** Branch `exp/combo-filter`,
base `0bf1d001`; no encoder code changed. Every candidate is a distinct legal
flag set formed from the existing filter/allocation CLI tools on top of the
`--nightly` aggregate (prediction/transform/entropy stay at the nightly
defaults). The one exception is `--video-cfl`, the prediction probe the
tool-matrix handoff explicitly requested; it is reported separately and was
not promoted. 22 legal combinations were screened, ten named rejections and
three legal-but-out-of-lane probes were re-run, and one arm was promoted to a
full QP ladder.

The oracle is `tools/oracle.py` (AVM `avmdec`, CVVDP 0.5.7 `standard_fhd`, MPS,
Butteraugli, SSIMULACRA2, `tools/kpi.py`). Fixture:
`testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit 4:2:0, single tile.

## Verdict

| quantity | value |
|---|---:|
| best internal matched-rate K | **82.957** at 100.0 kbps |
| nightly pin (QP106) | 82.155 at 100.032 kbps |
| local nightly K at 100.0 (linear 108/106) | 82.149 |
| delta vs pin | **+0.80 K** |
| SVT-AV1 wall | 88.8 |
| gap to SVT | **5.84 K** |
| AVM wall | 90.2 |
| gap to AVM | 7.24 K |
| best rate-free segment ceiling | **88.964** (f44-59 source replacement, pooling bound only) |
| best combined filter+allocation | 82.454 (`x_k26_ccso4`; sign flips, not a gain) |
| best pure filter stack | 81.474 (`f_pc_cdef4`; below nightly) |

**Best exact flags (only promotion):**

```sh
--nightly --video-key-qp-offset -26 \
  --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500
```

At QP108 this is 25,028 B / 100.112 kbps / JOD 9.7315 / BA 3.0051 /
SS2 81.097 / K 82.996. Linear interpolation to exactly 100.0 kbps (QP110..108
bracket) gives **K 82.957**. It passes the two-QP same-sign screen
(QP106 +0.29, QP108 +1.39 against the nightly at the same QP) and its
rate-free ceiling (88.964) is the only ceiling above 88.8, so it is the only
arm admitted to the full ladder. It is still **below 88.8** at matched rate;
the +0.80 is not an architecture claim and sits inside the +/-1.5-1.9 K QP
ripple documented in `docs/k-sweep.md`.

The ceiling is a pooling bound, not an encodable stream: replacing all of
f44-59 with source at the same rate scores 88.964. It does not make the
encodable point 88.9.

## Method

- Encoder: `build-release/av2enc` (main `0bf1d001`,
  current main build). The worktree has no code changes; `testdata` and
  `tools/bin` are the main-worktree symlinks.
- Screen: two QPs per arm. All arms are `--video --frames 60 --cpu-threads 8
  --qp <Q> --nightly <overrides>`. Filter/CCSO/allocation arms use QP106/108.
  The Wiener+GDF stack uses QP119/121 because luma Wiener costs ~15 QP at this
  point. The promoted arm adds QP104/110 to bracket 100 kbps.
- Rate matching: `tools/oracle.py --target-kbps 100`; a two-point linear
  interpolation is a matched-rate oracle only when the ladder brackets 100. A
  nearest point within 0.5% is accepted as a measured point (the oracle's
  own rule); a point further away is reported as "nearest, not matched".
- Same-sign rule: an arm is a screen winner only when the K difference against
  the nightly baseline has the same sign at QP106 and QP108. That is necessary,
  not sufficient: the promoted arm must also clear the ceiling gate.
- Ceiling gate: the maximum source-replacement K (the rate-free ceiling) at
  the arm's nearest-to-100 point must clear 88.8 before a full ladder is run.
  Only `d_k26_th20_b2500` passes (88.964). `d_k24_th20_b2000` (89.014) and
  `x_k26_ccso4` (88.923) also clear numerically but are not screen winners, so
  they get no ladder.
- Baseline: QP104 26,158 B / 104.632 kbps / K 80.734; QP106 25,008 B /
  100.032 kbps / K 82.155 (the documented pin); QP108 24,291 B / 97.164 kbps /
  K 81.604. Local K at exactly 100.0 is 82.149.

### Canonical `--nightly` expansion

Every row is this expansion plus the listed overrides; explicit CLI values win
(the key offset, drift threshold/budget, CDEF/CCSO parameters and leaf lambda
are all caller-wins knobs). `--video-rd-preset quality` expands to
`--video-inter-residual --video-deblock --video-ccso --video-tx-search
--video-intra-full`.

```text
--video-rd-preset quality
--video-ccso-rd --video-ccso-bands 6 --video-ccso-lambda 22
  --video-ccso-refit 2 --video-ccso-unit-bits 2
--video-edge-inter --video-edge-mask --video-partition-rd
--video-key-qp-offset -27 --video-rdoq --video-cdef --video-cdef-rd
--video-bawp --video-bawp-carrier --video-leaf-rd --video-subpel-eighth
--video-key-intra-txpart --video-qp-lambda --avg-cdf --adaptive-cdf
--cdf-continuity --video-drift-refresh
--video-tile-flush-min --video-ref-signal-compact
--video-tile-group-compact --video-ccso-header-rd
```

## Registry

The hash preimage is exactly

```text
av2enc --video --frames 60 --cpu-threads 8 --qp {ladder} --nightly {overrides}
```

where `{ladder}` is the literal comma-separated screen QP list in the row.
Input/output paths are excluded because they do not change decisions. All 22
hashes are distinct.

### Candidate registry

| id | category | overrides on `--nightly` | screen QPs | canonical sha256 |
|---|---|---|---|---|
| `f_pc_gdf` | filter stack | `--video-pc-wiener --video-gdf` | 106,108 | `01d0fb84000f135efbbb47c07ce6d1d1d65e4a91bd1570f1616bf6bd850015fa` |
| `f_pc_cdef4` | filter stack | `--video-pc-wiener --video-cdef-strength 4` | 106,108 | `8da63932c3bdbcce938ad73e1b72b20a0993856cc311af65f12f1daaca781d56` |
| `f_gdf_cdef4` | filter stack | `--video-gdf --video-cdef-strength 4` | 106,108 | `f2459ff3569151cdd0dba91d6a92c2ec42c9b0ef751d4921bcfa4161b11ba759` |
| `f_pc_gdf_cdef6` | filter stack | `--video-pc-wiener --video-gdf --video-cdef-strength 6 --video-cdef-damping 5` | 106,108 | `0fddd1f44aa629c755740c7977e8bac3a056476b00e056a150d2a5d14e631090` |
| `c_cdef2_plain` | CDEF param | `--video-cdef-strength 2` | 106,108 | `1cde31b0c1d93bbf04d1ae06f92ce01b5d79cfca5c20d75c58d49ba778165495` |
| `c_cdef4_plain` | CDEF param | `--video-cdef-strength 4` | 106,108 | `f770ebc58c57114972a4caead6618f0b9b25cd54fef0b108a557dff7f5b5ff48` |
| `c_cdef8_plain` | CDEF param | `--video-cdef-strength 8` | 106,108 | `d214e6be296fbc05828bee3224151dcacc51a78202b7c1fcd563b9c39f9a2325` |
| `c_cdef35` | CDEF param | `--video-cdef-strength 3 --video-cdef-damping 5` | 106,108 | `5568c86e8d4514ba258ea342a86a5f9a751ec0eef63e119df833d5f4a02b97c4` |
| `c_cdef64` | CDEF param | `--video-cdef-strength 6 --video-cdef-damping 4` | 106,108 | `06e1e205c268247f01ed08688dd741864bd787b0876f13d834e1a9bfc889b5e9` |
| `c_ccso_low` | CCSO param | `--video-ccso-bands 3 --video-ccso-lambda 12 --video-ccso-refit 3 --video-ccso-unit-bits 1` | 106,108 | `1f9ff15d01e7a405cca74e34def2d022f19bc7c627d93830eecc25f3298e9d3c` |
| `c_ccso_high` | CCSO param | `--video-ccso-bands 4 --video-ccso-lambda 44 --video-ccso-refit 4 --video-ccso-unit-bits 4` | 106,108 | `1f9fd8f09999ed868701ae24811a23bc831ef2a4432d2e825e800d7a181e7820` |
| `c_mix` | CDEF+CCSO param | `--video-cdef-strength 2 --video-ccso-bands 5 --video-ccso-lambda 32 --video-ccso-refit 3 --video-ccso-unit-bits 3` | 106,108 | `6f8b87f195981fec48dc68174931977d00c24c1e1b9654219312216501a1fff5` |
| `cfl` | prediction (matrix target) | `--video-cfl` | 106,108 | `d43e7cb3d8a80ea3715535ee23f866bd80b3d8909ca0cafa7f37545bfbc09680` |
| `d_k22_th20_b2000` | drift/key | `--video-key-qp-offset -22 --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2000` | 106,108 | `b8d5ed33c8c7fa895db183f021816649a27b623294e9b0dad21a85e65b9e860d` |
| `d_k23_th225_b2500` | drift/key | `--video-key-qp-offset -23 --video-drift-refresh-threshold 2.25 --video-drift-refresh-budget 2500` | 106,108 | `9493dc8bd51e7990dc796f61e9c53b985360c9061c3360f245fa96f996a11ade` |
| `d_k24_th20_b2000` | drift/key | `--video-key-qp-offset -24 --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2000` | 106,108 | `93b306c60441eee5a0d7399080808b0f54e94f4f112560e3f181c007c21a69be` |
| `d_k25_th20_b2500` | drift/key | `--video-key-qp-offset -25 --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500` | 106,108 | `03a6a3296ba7f20ff0e06e9ea516a14a8c5b58faac0362190067da40c051060c` |
| `d_k26_th20_b2500` | drift/key | `--video-key-qp-offset -26 --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500` | 104,106,108,110 | `7f0eee65a41eda8776648791e2620295a1b484eb72ad9a16a1ba101928756020` |
| `l_leaf250_k24` | allocation (leaf lambda) | `--video-key-qp-offset -24 --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2000 --video-leaf-rd-lambda 250` | 106,108 | `1858bd8e42f6ed84608f1debf3dce72cd1ed606cd474f6fffc6317aaf7eaaaba` |
| `w_gdf` | filter stack (rate-shifted) | `--video-wiener --video-gdf` | 119,121 | `cbcd411f25df50c1928b8883d4f1ddac55fff7a89dae9cc0b40d6af4c17340cb` |
| `x_k26_ccso4` | combined | `--video-key-qp-offset -26 --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500 --video-ccso-bands 4 --video-ccso-lambda 44 --video-ccso-refit 4 --video-ccso-unit-bits 4` | 106,108 | `359eba3d0e2cb48d197196ff6d61b8f101314c66f1109af4e3cab0e869b6d20b` |
| `x_k26_cdef4` | combined | `--video-key-qp-offset -26 --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500 --video-cdef-strength 4` | 106,108 | `ac0059cc2422c593d51768a4925cb5ffea48430a924d776cad2d51521bdad8b6` |



## Per-point results

The two/three/four points per arm are the raw oracle streams. The screen QP
pair is the evidence for the same-sign rule; JOD/BA/SS2 are point metrics, not
interpolated quantities.

### Per-point metrics

| id | QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f_pc_gdf` | 106 | 25,495 | 101.980 | 9.6786 | 4.0511 | 79.221 | 80.567 |
| `f_pc_gdf` | 108 | 24,790 | 99.160 | 9.6449 | 4.3775 | 77.598 | 79.657 |
| `f_pc_cdef4` | 106 | 25,337 | 101.348 | 9.7117 | 3.7357 | 80.345 | 81.417 |
| `f_pc_cdef4` | 108 | 24,589 | 98.356 | 9.7249 | 3.7316 | 80.661 | 81.543 |
| `f_gdf_cdef4` | 106 | 25,535 | 102.140 | 9.7010 | 4.2341 | 79.608 | 80.474 |
| `f_gdf_cdef4` | 108 | 25,021 | 100.084 | 9.6877 | 3.5836 | 79.393 | 81.396 |
| `f_pc_gdf_cdef6` | 106 | 25,516 | 102.064 | 9.6878 | 4.3415 | 79.284 | 80.202 |
| `f_pc_gdf_cdef6` | 108 | 25,134 | 100.536 | 9.1548 | 8.8357 | 62.982 | 70.941 |
| `c_cdef2_plain` | 106 | 24,976 | 99.904 | 9.7069 | 3.6907 | 80.748 | 81.535 |
| `c_cdef2_plain` | 108 | 24,472 | 97.888 | 9.7200 | 3.9653 | 80.384 | 81.098 |
| `c_cdef4_plain` | 106 | 25,029 | 100.116 | 9.7319 | 3.3535 | 80.597 | 82.230 |
| `c_cdef4_plain` | 108 | 24,523 | 98.092 | 9.7536 | 3.5988 | 80.812 | 81.936 |
| `c_cdef8_plain` | 106 | 25,213 | 100.852 | 9.7234 | 3.2510 | 80.586 | 82.379 |
| `c_cdef8_plain` | 108 | 24,482 | 97.928 | 9.7244 | 3.8060 | 80.648 | 81.416 |
| `c_cdef35` | 106 | 25,217 | 100.868 | 9.7005 | 4.5381 | 80.192 | 80.151 |
| `c_cdef35` | 108 | 24,244 | 96.976 | 9.7320 | 4.0452 | 80.514 | 81.058 |
| `c_cdef64` | 106 | 25,268 | 101.072 | 9.7300 | 3.7006 | 80.468 | 81.588 |
| `c_cdef64` | 108 | 24,412 | 97.648 | 9.7350 | 4.1652 | 80.368 | 80.870 |
| `c_ccso_low` | 106 | 24,870 | 99.480 | 9.6971 | 4.0104 | 79.992 | 80.849 |
| `c_ccso_low` | 108 | 24,024 | 96.096 | 9.7289 | 3.9087 | 80.185 | 81.198 |
| `c_ccso_high` | 106 | 24,773 | 99.092 | 9.7303 | 3.5708 | 80.520 | 81.819 |
| `c_ccso_high` | 108 | 24,025 | 96.100 | 9.7357 | 3.6060 | 80.363 | 81.759 |
| `c_mix` | 106 | 24,729 | 98.916 | 9.7255 | 3.3451 | 80.506 | 82.199 |
| `c_mix` | 108 | 24,142 | 96.568 | 9.7315 | 4.0570 | 80.364 | 81.013 |
| `cfl` | 106 | 24,794 | 99.176 | 9.7031 | 3.4629 | 80.235 | 81.826 |
| `cfl` | 108 | 24,322 | 97.288 | 9.7175 | 3.7707 | 80.392 | 81.397 |
| `d_k22_th20_b2000` | 106 | 24,565 | 98.260 | 9.7184 | 3.7311 | 79.634 | 81.340 |
| `d_k22_th20_b2000` | 108 | 23,743 | 94.972 | 9.7333 | 3.3217 | 80.190 | 82.229 |
| `d_k23_th225_b2500` | 106 | 24,716 | 98.864 | 9.7747 | 3.2218 | 81.865 | 82.905 |
| `d_k23_th225_b2500` | 108 | 24,024 | 96.096 | 9.7455 | 3.7596 | 80.900 | 81.640 |
| `d_k24_th20_b2000` | 106 | 24,947 | 99.788 | 9.7133 | 3.3245 | 80.161 | 82.119 |
| `d_k24_th20_b2000` | 108 | 24,396 | 97.584 | 9.7194 | 3.7660 | 79.981 | 81.345 |
| `d_k25_th20_b2500` | 106 | 25,507 | 102.028 | 9.7427 | 3.2704 | 80.852 | 82.483 |
| `d_k25_th20_b2500` | 108 | 24,961 | 99.844 | 9.7094 | 3.7393 | 80.003 | 81.343 |
| `d_k26_th20_b2500` | 104 | 26,297 | 105.188 | 9.7130 | 3.4323 | 80.545 | 81.982 |
| `d_k26_th20_b2500` | 106 | 25,945 | 103.780 | 9.7416 | 3.2722 | 80.649 | 82.440 |
| `d_k26_th20_b2500` | 108 | 25,028 | 100.112 | 9.7315 | 3.0051 | 81.097 | 82.996 |
| `d_k26_th20_b2500` | 110 | 24,457 | 97.828 | 9.7263 | 3.3761 | 80.781 | 82.191 |
| `l_leaf250_k24` | 106 | 24,181 | 96.724 | 9.7159 | 3.2553 | 79.999 | 82.236 |
| `l_leaf250_k24` | 108 | 23,728 | 94.912 | 9.6880 | 3.1523 | 79.495 | 82.213 |
| `w_gdf` | 119 | 26,147 | 104.588 | 9.7078 | 3.8217 | 79.224 | 81.071 |
| `w_gdf` | 121 | 25,474 | 101.896 | 9.7135 | 3.4086 | 79.037 | 81.777 |
| `x_k26_ccso4` | 106 | 25,326 | 101.304 | 9.7142 | 3.6569 | 79.824 | 81.475 |
| `x_k26_ccso4` | 108 | 24,963 | 99.852 | 9.7187 | 3.1043 | 80.114 | 82.566 |
| `x_k26_cdef4` | 106 | 25,991 | 103.964 | 9.7172 | 3.4773 | 80.381 | 81.895 |
| `x_k26_cdef4` | 108 | 25,530 | 102.120 | 9.7281 | 3.6719 | 80.594 | 81.647 |



## Matched-rate results and rate-free ceilings

`method` is the oracle's own: `linear` brackets 100.0; `nearest` is a single
measured point within 0.5% (accepted) or farther away (not a matched-rate
oracle). A `*` marks a nearest point further than 0.5% from 100 kbps: its K is
the measured point, not a matched-rate value, and it is not used for a gain
claim. `best source ceiling` is the maximum source-replacement K over the
four oracle segments; it is a pooling bound, not an encodable stream.

### Matched-rate and ceiling

| id | matched K | matched kbps | method | best source ceiling | segment | same-sign 106/108 |
|---|---:|---:|---|---:|---|---|
| `f_pc_gdf` | 79.928 | 100.000 | linear | 87.263 | f44-59 | -- |
| `f_pc_cdef4` | 81.474 | 100.000 | linear | 88.229 | f44-59 | -- |
| `f_gdf_cdef4` | 81.396 | 100.084 | nearest | 87.768 | f44-59 | -- |
| `f_pc_gdf_cdef6` | 70.941* | 100.536 | nearest | 80.641 | f44-59 | -- |
| `c_cdef2_plain` | 81.535 | 99.904 | nearest | 88.381 | f44-59 | -- |
| `c_cdef4_plain` | 82.213 | 100.000 | linear | 88.313 | f44-59 | ++ |
| `c_cdef8_plain` | 82.099 | 100.000 | linear | 88.348 | f44-59 | +- |
| `c_cdef35` | 80.353 | 100.000 | linear | 87.369 | f44-59 | -- |
| `c_cdef64` | 81.363 | 100.000 | linear | 87.492 | f44-59 | -- |
| `c_ccso_low` | 80.849* | 99.480 | nearest | 88.225 | f44-59 | -- |
| `c_ccso_high` | 81.819* | 99.092 | nearest | 88.311 | f44-59 | -+ |
| `c_mix` | 82.199* | 98.916 | nearest | 88.493 | f44-59 | +- |
| `cfl` | 81.826* | 99.176 | nearest | 88.097 | f44-59 | -- |
| `d_k22_th20_b2000` | 81.340* | 98.260 | nearest | 88.106 | f44-59 | -+ |
| `d_k23_th225_b2500` | 82.905* | 98.864 | nearest | 88.038 | f0-32 | ++ |
| `d_k24_th20_b2000` | 82.119 | 99.788 | nearest | 89.014 | f44-59 | -- |
| `d_k25_th20_b2500` | 81.424 | 100.000 | linear | 88.093 | f44-59 | +- |
| `d_k26_th20_b2500` | 82.957 | 100.000 | linear | 88.964 | f44-59 | ++ |
| `l_leaf250_k24` | 82.236* | 96.724 | nearest | 88.519 | f44-59 | ++ |
| `w_gdf` | 81.777* | 101.896 | nearest | 87.169 | f44-59 | n/a |
| `x_k26_ccso4` | 82.454 | 100.000 | linear | 88.923 | f44-59 | -+ |
| `x_k26_cdef4` | 81.647* | 102.120 | nearest | 88.643 | f44-59 | -+ |



### Screen winners blocked by the gate

Three other arms have the same positive sign at QP106 and QP108, but their
rate-free ceiling is below 88.8, so they get no full ladder and their K is not
a matched-rate claim:

| arm | QP106 / QP108 K | nearest K (rate) | ceiling | why blocked |
|---|---|---:|---:|---|
| `c_cdef4_plain` | 82.230 / 81.936 | 82.213 (100.000, linear) | 88.313 | ceiling < 88.8; +0.06 is inside the QP ripple |
| `d_k23_th225_b2500` | 82.905 / 81.640 | 82.905* (98.864) | 88.038 | ceiling < 88.8; nearest point is 1.14% below target |
| `l_leaf250_k24` | 82.236 / 82.213 | 82.236* (96.724) | 88.519 | ceiling < 88.8; the leaf-lambda rate saving puts both screen QPs below target |

`d_k24_th20_b2000` (ceiling 89.014) and `x_k26_ccso4` (ceiling 88.923) clear the
ceiling numerically but are not screen winners (negative at QP106 for
`d_k24`, sign flip for `x_k26_ccso4`), so they also get no ladder.

## Promoted ladder: `d_k26_th20_b2500`

Exact flags: `--nightly --video-key-qp-offset -26
--video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500`.

| QP | bytes | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|---:|
| 104 | 26,297 | 105.188 | 9.7130 | 3.4323 | 80.545 | 81.982 |
| 106 | 25,945 | 103.780 | 9.7416 | 3.2722 | 80.649 | 82.440 |
| 108 | 25,028 | 100.112 | 9.7315 | 3.0051 | 81.097 | 82.996 |
| 110 | 24,457 | 97.828 | 9.7263 | 3.3761 | 80.781 | 82.191 |
| **100.0 matched** | — | **100.000** | — | — | — | **82.957** |

The 100.0 value is linear in rate between QP110 (97.828) and QP108 (100.112).
QP104/106/108 are the requested ladder; QP110 is the extra point needed to
bracket 100 because QP108 is already at 100.11 kbps.

Matched-rate components for the promoted arm and the combined best (linear in
rate over the bracketing points, same method as K; components are interpolated,
not re-scored):

| arm | kbps | JOD | BA | SS2 | K | dK vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| nightly baseline | 100.000 | 9.7333 | 3.4231 | 80.835 | 82.149 | — |
| `d_k26_th20_b2500` | 100.000 | 9.7313 | 3.0233 | 81.081 | **82.957** | **+0.808** |
| `x_k26_ccso4` (combined best) | 100.000 | 9.7182 | 3.1608 | 80.084 | 82.455 | +0.306 |

### Segment / rate-free ceilings at the segment base (QP108, 100.112 kbps)

| segment | K | dK vs 82.996 |
|---|---:|---:|
| key f0 | 83.079 | +0.082 |
| f0-32 | 86.945 | +3.949 |
| f33-43 | 84.884 | +1.888 |
| f44-59 | **88.964** | +5.967 |

The f44-59 row is the rate-free ceiling; it clears 88.8 by 0.164 K. It is the
only reason this arm was allowed a full ladder. Pooling bound only.

### Baseline ladder (for the same rates)

| QP | bytes | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|---:|
| 104 | 26,158 | 104.632 | 9.7185 | 4.1720 | 80.108 | 80.734 |
| 106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | 82.155 |
| 108 | 24,291 | 97.164 | 9.7216 | 3.6824 | 80.633 | 81.604 |
| **100.0 matched** | — | **100.000** | — | — | — | **82.149** |

## Legal and illegal combinations

All 22 registry rows are legal (`rc 0`) on the nightly aggregate and decoded
60/60 with `avmdec`. The following probes were rejected by name, or are legal
but out of lane.

| combination | status | exact result |
|---|---|---|
| `--nightly --video-aq` | **rejected** | `segmentation/delta-Q composes with the CDF chain only under --video-cdf-per-frame-qp: the per-tile delta/segment CDFs are not part of the §8.2.4 image, so the image is re-derived from the §9.3 defaults per frame rather than carried` |
| `--nightly --video-aq --video-cdf-per-frame-qp` | legal (rc 0), out of lane | the only legal AQ composition; it replaces the carried CDF chain with the per-frame reset. `docs/video-aq.md` already measures it at -0.47 K under nightly, so it is not re-measured here |
| `--nightly --video-aq --video-delta-q` | **rejected** | `--video-aq replaces the --video-delta-q staircase with an activity-driven plan; pass only one of them` |
| `--nightly --video-pc-wiener --video-wiener` | **rejected** | `--video-pc-wiener replaces the luma Wiener NS decision; do not combine the two luma restoration tools` |
| `--nightly --video-wiener --tiles 2x1` (with `--video-edge-inter-off --video-edge-mask-off --video-deblock-off --video-drift-refresh-off --video-tile-flush-off --video-ref-signal-off --video-tile-group-off`) | **rejected** | `--video-wiener/--video-chroma-lr do not support --tiles > 1x1: the pinned decoder's loop-restoration unit and window handling at shared tile edges does not match a frame-level filter (docs/negative-results.md)` |
| `--nightly --tiles 2x1` | **rejected** | `--video-edge-inter needs the single-tile writer; the tiled path keeps its tile-local reconstruction` (first blocker; tiles also require deblock off, which the nightly filter stack owns) |
| `--nightly --video-cdef-strength 9` | **rejected** | `--video-cdef-strength must be an integer in [1, 8]` |
| `--nightly --video-cdef-strength 0` | **rejected** | same range message |
| `--nightly --video-ccso-bands 7` | **rejected** | `--video-ccso-bands must be an integer in [0, 6]` |
| `--nightly --video-ccso-unit-bits -1` | **rejected** | `--video-ccso-unit-bits needs a non-negative bit count` |
| `--nightly --video-ibp --video-cfl` | **rejected** | `--video-ibp requires the plain DC chroma path: --video-cfl/--video-mhccp do not apply the §7.13.2.12 IBP DC post-filter` |
| `--nightly --video-ibp` | legal but left out of lane | IBP is prediction, not filter/allocation; the matrix says untested with nightly and `docs/k-sweep.md` measures K 81.25. Not re-measured here |
| `--nightly --video-drift-refresh-qp 8 --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250` | legal, rate probe only | QP106 27,487 B / 109.9 kbps; it needs a QP112+ bracket, not the 104/106/108 ladder. Not registered |

Already-measured paths deliberately not repeated: the exact `--nightly`
expansion (82.155), eff-perc (85.10), plain `--video-wiener`/`--video-chroma-lr`/
`--video-pc-wiener` (lr-quality.md), plain `--video-gdf` (k-sweep), plain CDEF
strength 2/4 single points (k-sweep), the tuned 1.5/3250 drift recipe
(83.795), and the `--video-tx-lambda` / `--video-partition-rd-lambda` knobs.

## Tool-matrix handoff (`docs/tool-matrix.md` in main)

These are the rows the matrix asked this lane to fill. All values are
matched-rate K at 240p/100 kbps unless marked as a point.

| tool / combination | legality on `--nightly` | measured result |
|---|---|---|
| `--video-cdef-strength 4` | legal, FP single tile | QP106 25,029 B / 100.116 kbps / JOD 9.7319 / BA 3.3535 / SS2 80.597 / K 82.230; QP108 24,523 B / 98.092 / JOD 9.7536 / BA 3.5988 / SS2 80.812 / K 81.936; matched **82.213** at 100.0. Ceiling 88.313. Same sign +, but +0.06 is inside the QP ripple; no ladder |
| `--video-cdef-strength 2` | legal | QP106 24,976 B / 99.904 kbps / JOD 9.7069 / BA 3.6907 / SS2 80.748 / K 81.535; QP108 K 81.098. Both below baseline; ceiling 88.381 |
| `--video-cdef-strength 8` | legal | QP106 25,213 B / 100.852 kbps / JOD 9.7234 / BA 3.2510 / SS2 80.586 / K 82.379; QP108 K 81.416; matched 82.099 at 100.0. Sign flips; ceiling 88.348 |
| `--video-cdef-strength 3 --video-cdef-damping 5` | legal | matched 80.353; below baseline both QPs |
| `--video-cdef-strength 6 --video-cdef-damping 4` | legal | matched 81.363; below baseline both QPs |
| `--video-pc-wiener --video-gdf` | legal | 79.928 at 100.0; below baseline |
| `--video-pc-wiener --video-cdef-strength 4` | legal | 81.474 at 100.0; below baseline |
| `--video-gdf --video-cdef-strength 4` | legal | 81.396 at 100.084; below baseline |
| `--video-wiener --video-gdf` | legal; needs QP119/121 | 81.777 at 101.896 (nearest); ceiling 87.169; below baseline |
| `--video-pc-wiener --video-gdf --video-cdef-strength 6 --video-cdef-damping 5` | legal, but the QP108 point collapses | QP106 K 80.202; QP108 JOD 9.1548 / BA 8.8357 / SS2 62.982 / K 70.941. 60/60 decoded. Excluded; see anomaly below |
| `--video-ccso-bands 3 --video-ccso-lambda 12 --video-ccso-refit 3 --video-ccso-unit-bits 1` | legal | 80.849 at 99.480 (nearest, not matched); below baseline |
| `--video-ccso-bands 4 --video-ccso-lambda 44 --video-ccso-refit 4 --video-ccso-unit-bits 4` | legal | 81.819 at 99.092 (nearest, not matched); below baseline |
| `--video-cdef-strength 2 --video-ccso-bands 5 --video-ccso-lambda 32 --video-ccso-refit 3 --video-ccso-unit-bits 3` | legal | 82.199 at 98.916 (nearest, not matched); QP106 +0.04, QP108 -0.59; sign flips; ceiling 88.493 |
| `--video-cfl` | legal, FP single tile | QP106 24,794 B / 99.176 kbps / JOD 9.7031 / BA 3.4629 / SS2 80.235 / K 81.826; QP108 24,322 B / 97.288 / JOD 9.7175 / BA 3.7707 / SS2 80.392 / K 81.397. Below baseline at both QPs; ceiling 88.097. Matches the k-sweep 81.83 |
| `--video-ibp` | legal but not filter-adjacent | left out (addendum: otherwise leave); k-sweep 81.25; also rejects `--video-cfl` by name |
| `--video-aq` | rejected with the nightly CDF chain; legal only with `--video-cdf-per-frame-qp` | out of lane (entropy reset) and already measured dead (-0.47 K, `docs/video-aq.md`) |
| `--video-key-qp-offset -26` + drift `2.0/2500` | legal | **82.957 matched**, the best internal arm; ceiling 88.964; full ladder above |
| `--video-leaf-rd-lambda 250` + key `-24` + drift `2.0/2000` | legal | 82.236 at 96.724 (nearest, not matched); ceiling 88.519; same sign + but the ceiling gate blocks a ladder |

## Anomaly: `f_pc_gdf_cdef6` at QP108

`--video-pc-wiener --video-gdf --video-cdef-strength 6
--video-cdef-damping 5` is legal and decodes 60/60, but its QP108 point
collapses (25,134 B / 100.536 kbps / JOD 9.1548 / BA 8.8357 / SS2 62.982 /
K 70.941) while QP106 is 80.202. This is not a decode failure (the oracle
decoded every frame and scored it). It is recorded as a legal but non-viable
stack; no patch was made and it is excluded from promotion. It is a warning
against CDEF strength 6 with damping 5 on this carrier, not a measured RD
result.

## Reproduction

```sh
BIN=build-release/av2enc
REF=testdata/nasa_480x240_30fps_60f.y4m
"$BIN" "$REF" -o /tmp/d26_108.ivf --video --frames 60 --cpu-threads 8 --qp 108 \
  --nightly --video-key-qp-offset -26 \
  --video-drift-refresh-threshold 2.0 --video-drift-refresh-budget 2500
python3 tools/oracle.py --ref "$REF" --ivf d26@108=/tmp/d26_108.ivf \
  --target-kbps 100 --json /tmp/d26.json
```

The full nine-point oracle command is the same `--ivf name@qp=path` form for
the registry QP lists; each run decodes with `tools/bin/avmdec` and scores with
CVVDP/Butteraugli/SSIMULACRA2.

## Bottom line

The filter/rate-allocation composition space on this carrier is exhausted for
the 88.8 wall: no stack of the existing loop filters (CDEF/CCSO/Wiener/
PC-Wiener/GDF), no CDEF/CCSO parameter pair, and no drift/key allocation pair
reaches SVT. The best internal arm is `d_k26_th20_b2500` at **82.957**
(+0.80 vs the 82.155 pin, 5.84 below 88.8). The only rate-free ceiling above
88.8 is 88.964, and it is a pooling bound, not an encodable stream. The
combined best `x_k26_ccso4` is 82.454 and sign-flips, so it is not a gain. The
lane is closed without a ship candidate.