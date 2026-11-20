# Combo-TX: legal tool-composition architectures at the 240p/100 kbps point

Status: **closed, negative on the wall.** Branch `exp/combo-tx`, base
`0bf1d001` (current `main`). No encoder source, tool or test was changed; this
document is the deliverable. All encodes used
`build-release/av2enc` (the `release` preset built from
`0bf1d001`), the AVM decoder (`tools/bin/avmdec`) and `tools/oracle.py`.
`dav2d --strict 1` independently decoded every scored stream 60/60 before the
oracle ran.

The search asked for novel architectures formed only by combining existing
transform/quantisation/entropy/rate CLI flags on top of the measured
`--fast-decode` aggregate. The prediction and filter tools stay at their nightly
values for the primary registry (the tool-matrix section is the only exception,
and says so).

Result in one line: **the best internal combination is `--fast-decode
--video-rdoq-scale 0.2 --video-tx-lambda 24` at K 82.490 / 99.176 kbps
(QP 106); its rate-free (source-replacement) ceiling is 88.459, still 0.341
below the SVT 88.8 wall, so it is not promoted to a full ladder and no
matched-rate gain is claimed.** The current internal best remains eff-perc
85.10; SVT-AV1 p0 is 88.83.

## 1. Protocol

* Fixture: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit 4:2:0,
  single tile, `--cpu-threads 8`, `--rc` default `cqp`.
* Base flag template for every registry row:

  ```text
  --video --frames 60 --cpu-threads 8 --qp <QP> --fast-decode <extra flags>
  ```

  `--fast-decode` expands deterministically at `0bf1d001` (see
  `src/cli/nightly.cpp`): quality preset (inter residual, deblock, CCSO, TX
  search, intra full), rate-aware CCSO + header RD, bottom-edge inter + mask,
  partition probe at `2.75e-6` depth 1, key QP offset −27, RDOQ at 0.1,
  keyframe intra TX partition, avg/adaptive/carried CDF, drift refresh,
  eighth-pel ME, TX lambda anchor 136, CDEF + CDEF RD, QP-scaled lambdas,
  minimum tile flush, compact ref signalling, compact tile group, BAWP carrier
  and leaf RD. The registry does not re-type that expansion; it hashes the CLI
  flag set.
* **Full-flag-set hash**: SHA-256 of the template above with `<QP>` left as the
  QP placeholder. The QP is the ladder point, not part of the architecture, so
  every point of a ladder shares one hash. The literal command for a point is
  the template with `<QP>` replaced by 106 or 108.
* Screen: every candidate encoded at QP 106 and QP 108. A gain is only
  *sign-consistent* when the same-QP K delta against the control has the same
  sign at both points. The documented QP ripple on this clip is ±0.2–0.5 K, so
  a sign-consistent delta that is inside that band is a direction, not a
  resolved gain.
* Matched rate: `tools/oracle.py` linearly interpolates each candidate's own
  two-point ladder to 100.000 kbps. A ladder that does not bracket 100 kbps is
  labelled `nearest` by the harness and is **not** a matched-rate oracle.
* Ceiling: `tools/oracle.py` source replacement of one audit segment
  (key f0 / f0-32 / f33-43 / f44-59) is the rate-free ceiling. It is a pooling
  bound, not an encodable stream. The gate is applied to the **best** source
  segment of the candidate.
* No new tools, no encoder code, no third-party encoder, no push, no merge.
  Every command below is a legal existing flag set or a named rejection.

Control:

```text
--video --frames 60 --cpu-threads 8 --qp 106 --fast-decode
25,008 B / 100.032 kbps / JOD 9.7334 / BA 3.4202 / SS2 80.837 / K 82.155
(--qp 108: 24,291 B / 97.164 kbps / 81.604; two-point matched K 82.149)
```

Hash: `4b3608db1b8c9a19467e9801763937cc152db2e4d5d39f1549c13ebfd7d8c457`.

## 2. Registry (27 distinct full flag sets, 27 unique hashes)

`ctl` is the control. `legacyexplicit` is the legal constant-lambda control for
the explicit-versus-QP-lambda exclusion (§3). `cdefstrength4` is legal but is a
filter-lane knob and was intentionally not measured here. All other rows are
legal and measured at both screening QPs.

### 2.1 Primary transform / quant / entropy / rate registry

| ID | family | extra flags on top of the base template | SHA-256 of full flag set |
|---|---|---|---|
| `ctl` | control | `(none)` | `4b3608db1b8c9a19467e9801763937cc152db2e4d5d39f1549c13ebfd7d8c457` |
| `legacyexplicit` | lambda control | `--video-tx-lambda 136 --video-key-tx-lambda 136 --video-leaf-rd-lambda 125 --video-rdoq-scale 0.1 --video-partition-rd-lambda 2.75e-6` | `f9ee446a77b1784934aac475bc4ba738a519b42f4cb6dae9d87c8dcacb56f1b1` |
| `rdoq200` | trellis | `--video-rdoq-scale 0.2` | `8956addafcc23296aacffbf249184c5daeb916e122cfc0129a48bb5788fa83e6` |
| `rdoq050` | trellis | `--video-rdoq-scale 0.05` | `f09b24265694bae1a525ff96b14b775b1256aa30da514cc8f9fa582792339f61` |
| `tx272` | rate | `--video-tx-lambda 272` | `c072254ea0ee95e0634cdcd1d041ee1b1bf0f1ba416eaf829c6990cd60b8aa1e` |
| `tx024` | rate | `--video-tx-lambda 24` | `a51f7243a56df8ea1dbd32356166b5658d4844a0f80eb01c768bebac308e1dae` |
| `keytx32` | rate | `--video-key-tx-lambda 32` | `135168e3e0eabdb12e7c217a564599388964b72114d34021f0026e625dea716d` |
| `keytx04` | rate | `--video-key-tx-lambda 4` | `f23e49c7a0431a49bfb6220b90b640a69fc2e28163e47fbceb961bc5409cf820` |
| `leaf250` | rate | `--video-leaf-rd-lambda 250` | `85ca2a651bb418dac8d0b000e8511345142df50a31b2eacd671ba6b4633e92a2` |
| `part55` | rate | `--video-partition-rd-lambda 5.5e-6` | `768515590de964dcad9c16b51a201478e1d69b69f1850a336d4bbea4c0f519cd` |
| `cdfreset_rdoq` | entropy + trellis | `--video-cdf-per-frame-qp --video-rdoq-scale 0.05` | `54217b2bcc4839e23feb28abb06f3ae474f462d55cc58cb610ce52f93432cb1b` |
| `cdfreset_ctx` | entropy + rate | `--video-cdf-per-frame-qp --video-tx-context-rate` | `1e536b47d0ada58e35304c10efdb0a8ea2d907aeb525046be8bb54a6c2275da7` |
| `cdfreset_dq` | entropy + quant | `--video-cdf-per-frame-qp --video-delta-q` | `0e62cfaa3fb9d7c3e022a2460ffbd8f70c70ebe2bf9e7badc59cdc44011ecf06` |
| `rdoq_keytx` | trellis + rate | `--video-rdoq-scale 0.05 --video-key-tx-lambda 4` | `824472b9749d792307be8733407a1004016f7b042ff8b5e4d607844686744e8e` |
| `rdoq_tx` | trellis + rate | `--video-rdoq-scale 0.2 --video-tx-lambda 24` | `32b74fd01cd797fd7c694f6a5a9b1be070fe3fbb79e27ce83eabb69419bb649c` |
| `overheaddrop` | entropy (3 levers) | `--video-tile-group-off --video-ref-signal-off --video-ccso-header-off` | `333fb7f9433ca28e57490ffc7a0d1109dd601ba02a4ff0e5c22b230ffa1368b6` |
| `rdoq050_tx24` | trellis + rate | `--video-rdoq-scale 0.05 --video-tx-lambda 24` | `b43c99f75d7fb95440476e33cc55799ea0a03388b68ce5d2418f9d85dc7cdfdd` |
| `combined_best` | combined best | `--video-rdoq-scale 0.2 --video-tx-lambda 24 --video-tile-group-off --video-ref-signal-off --video-ccso-header-off` | `46ff5225fa0cc209f96226ca0e719e72e4d5ea4e71fe6ba8a384c7443cceaa60` |
| `rdoq_tx_overhead` | trellis + entropy | `--video-rdoq-scale 0.2 --video-tx-lambda 24 --video-tile-group-off --video-ref-signal-off` | `3afc9538fba3c429d7b4d320ef963b154c610a843b9349d466f309aef9d6dc21` |

### 2.2 Tool-matrix additions requested by the parent

These are the untapped compositions named in `docs/tool-matrix.md` (parent
tree, `f91115ae`). They are outside the strict primary lane; measured here
because the matrix asks for legality and matched-rate K.

| ID | family | extra flags on top of the base template | SHA-256 of full flag set |
|---|---|---|---|
| `cfl` | matrix prediction | `--video-cfl` | `99ebacb9e4229a830d06d9b505252b347291252d2e1de899435e213d7be01ccb` |
| `ibp` | matrix prediction | `--video-ibp` | `1b96ad41cb2460b0e5d7e9e72883e763b3b3bdbfe3e296156292fe7c158501c1` |
| `ist` | matrix transform | `--video-ist` | `dec7e39e88671289ea13009fa1345b4be768022561fbca0cfe28b5e6e4ad9a41` |
| `dip` | matrix prediction | `--video-dip` | `086209f3edaee3eb20bacab7413c3554d2a70f33cd32a2361bf961f37d8b11c1` |
| `idtx` | matrix transform | `--video-idtx` | `d2d9f74054cee7b4bc592f464dc8cfda22efd74178fde6e9462da3302c29ec70` |
| `tcq_cctx` | matrix quant/entropy | `--video-rdoq-off --video-tcq --video-cctx` | `b9a297dcf97e74b6032138d699d7861445eb006cf26e08c2ce90414c5db33cc1` |
| `cdfoff_mhccp` | matrix entropy/prediction | `--video-cdf-chain-off --video-mhccp` | `5391800f8ecfb6ba10de02ce1f24780f69b2814af5789e34f1f3e5e52dd3717d` |
| `cdefstrength4` | filter lane, not measured | `--video-cdef-strength 4` | `f005b28f261915f24bbbf9408cf38eb7643b43063c139236d4fa087efba9436f` |

Hash uniqueness check: 27 rows, 27 distinct hashes. No flag set repeats
another row.

## 3. Legal control for the explicit-lambda vs QP-lambda exclusion

`--fast-decode` forces `--video-qp-lambda` and therefore the five per-site power
laws (`src/pipeline/video/qp_lambda.h`). An explicit numeric lambda flag is
legal **with** `--video-qp-lambda`: it pins exactly that one site and keeps the
legacy form; it is not multiplied by the fit a second time, and the other four
sites keep their fits. There is no CLI switch that turns `--video-qp-lambda`
off under `--fast-decode`.

The legal constant-lambda control is therefore the all-explicit set
`legacyexplicit`:

```text
--video --frames 60 --cpu-threads 8 --qp 106 --fast-decode \
  --video-tx-lambda 136 --video-key-tx-lambda 136 \
  --video-leaf-rd-lambda 125 --video-rdoq-scale 0.1 \
  --video-partition-rd-lambda 2.75e-6
```

It is legal, encodes and decodes 60/60, and reproduces the historical constant
values rather than the QP fits. It measures 25,172 B / 100.688 kbps at QP 106,
K 80.097 (−2.058 vs control), and 24,567 B / 98.268 kbps / K 80.778 at QP 108.
Two-point matched K 80.291, i.e. **−1.858** versus the control's 82.149. This
is the clean statement that the QP-scaled set is worth ~1.9 K at this point and
that the individual explicit flags used elsewhere in the registry are
per-site escape hatches, not a hidden second scaling.

`--video-qp-lambda` with a rate-controlled stream is illegal:

```text
av2enc: --video-qp-lambda needs a fixed --qp stream: the rate-controlled paths own base_q_idx and would re-scale every frame
```

## 4. Illegal / blocked compositions (exact first line, exit 2 unless noted)

| Flags tried (on top of the base template) | Exact rejection |
|---|---|
| `--video-delta-q` | `av2enc: segmentation/delta-Q composes with the CDF chain only under --video-cdf-per-frame-qp: the per-tile delta/segment CDFs are not part of the §8.2.4 image, so the image is re-derived from the §9.3 defaults per frame rather than carried` |
| `--video-cdf-per-frame-qp --video-uv-delta-q 4` | `av2enc: --video-rdoq does not compose with segmentation/--video-uv-delta-q: the trellis does not replay the §7.14.2 ALT_Q or per-plane step` |
| `--video-qm --qp 90` | `av2enc: --video-qm does not compose with --video-tx-search: the device TX search module has its own scalar quantizer` |
| `--video-qp-lambda --rc crf --crf 32` | `av2enc: --video-qp-lambda needs a fixed --qp stream: the rate-controlled paths own base_q_idx and would re-scale every frame` |
| `--video-tcq` | `av2enc: --video-rdoq replaces the scalar quantiser, it does not compose with --video-tcq (its half-step trellis owns the levels)` |
| `--video-tcq --video-cctx` | same `--video-rdoq ... does not compose with --video-tcq` line (RDOQ is on in `--fast-decode`) |
| `--video-mhccp` | `av2enc: --video-mhccp does not compose with --avg-cdf/--adaptive-cdf: the §8.2.4 bank does not carry the §5.20.7.32 cfl_mhccp/cfl_mh_dir CDFs` |
| `--video-cfl --video-ibp` | `av2enc: --video-ibp requires the plain DC chroma path: --video-cfl/--video-mhccp do not apply the §7.13.2.12 IBP DC post-filter` |
| `--video-idtx --video-min-block 4` | exit 1: `av2enc: error: --video-min-block 4 does not compose with --video-idtx yet (the key+P chain is not decoder-exact at the sub-8 search depths)` |

Legal controls for those blocked surfaces, all encoded and decoded 60/60:

* `--video-cdf-per-frame-qp --video-delta-q` (`cdfreset_dq`, measured).
* `--video-cdf-per-frame-qp --video-rdoq-off --video-uv-delta-q 4` (legal,
  not measured; the UV-DC/AC step is separable only without the trellis).
* `--video-rdoq-off --video-tcq --video-cctx` (`tcq_cctx`, measured).
* `--video-cdf-chain-off --video-mhccp` (`cdfoff_mhccp`, measured).

No combination was patched, retried with modified semantics, or silently
accepted.

## 5. Two-QP screen (QP 106 / QP 108)

`dK` is the same-QP K difference against the control at that QP. `matched` is
the `tools/oracle.py` two-point interpolation to 100.000 kbps; `nearest` means
the ladder does not bracket 100 kbps and the harness does not certify it.
`sign` is the two-QP sign test.

### 5.1 Primary registry

| ID | K 106 | dK 106 | K 108 | dK 108 | kbps 106 / 108 | matched K (method) | sign |
|---|---:|---:|---:|---:|---|---:|---|
| `ctl` | 82.155 | 0 | 81.604 | 0 | 100.032 / 97.164 | 82.149 (linear) | control |
| `legacyexplicit` | 80.097 | −2.058 | 80.778 | −0.825 | 100.688 / 98.268 | 80.291 (linear) | − − |
| `rdoq200` | 81.173 | −0.982 | 80.760 | −0.844 | 100.232 / 97.164 | 81.142 (linear) | − − |
| `rdoq050` | 82.327 | **+0.172** | 81.659 | **+0.056** | 100.524 / 95.700 | **82.255 (linear)** | + + |
| `tx272` | 81.607 | −0.548 | 81.210 | −0.394 | 101.108 / 97.772 | 81.475 (linear) | − − |
| `tx024` | 80.978 | −1.177 | 81.686 | +0.082 | 99.872 / 96.548 | 80.978 (nearest) | − + |
| `keytx32` | 80.141 | −2.014 | 81.447 | −0.156 | 99.404 / 96.808 | 80.141 (nearest) | − − |
| `keytx04` | 81.658 | −0.497 | 81.435 | −0.169 | 100.620 / 97.664 | 81.611 (linear) | − − |
| `leaf250` | 81.577 | −0.578 | 81.683 | +0.079 | 97.428 / 93.776 | 81.577 (nearest) | − + |
| `part55` | 82.007 | −0.148 | 80.791 | −0.813 | 102.732 / 99.484 | 80.984 (linear) | − − |
| `cdfreset_rdoq` | 82.327 | +0.172 | 81.659 | +0.056 | 103.256 / 98.320 | 81.887 (linear) | + + (same recon, +2.7% rate) |
| `cdfreset_ctx` | 81.013 | −1.142 | 81.169 | −0.434 | 101.676 / 97.848 | 81.081 (linear) | − − |
| `cdfreset_dq` | 81.497 | −0.658 | 81.181 | −0.422 | 100.676 / 98.164 | 81.412 (linear) | − − |
| `rdoq_keytx` | 81.268 | −0.887 | 81.131 | −0.473 | 100.764 / 97.728 | 81.234 (linear) | − − |
| `rdoq_tx` | **82.490** | **+0.335** | 81.765 | **+0.161** | 99.176 / 96.204 | 82.490 (nearest) | + + |
| `overheaddrop` | 82.435 | **+0.280** | 81.908 | **+0.305** | 102.608 / 98.864 | 82.068 (linear) | + + (rate cost) |
| `rdoq050_tx24` | 81.231 | −0.924 | 81.419 | −0.185 | 99.864 / 97.988 | 81.231 (nearest) | − − |
| `combined_best` | 82.157 | +0.002 | 81.214 | −0.390 | 102.152 / 99.108 | 81.490 (linear) | + − |
| `rdoq_tx_overhead` | 82.490 | +0.335 | 81.765 | +0.161 | 99.584 / 96.624 | 82.490 (nearest) | + + (dominated) |

Detail: full per-stream bytes, kbps, PSNR-Y, SSIM, JOD, BA, SS2 and K are in
the oracle JSON (`screen_all.json`, `screen_extra.json`; hashes in §9). The
central rows:

| stream | bytes | kbps | PSNR-Y | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctl QP106 | 25,008 | 100.032 | 44.668 | 9.7334 | 3.4202 | 80.837 | 82.155 |
| ctl QP108 | 24,291 | 97.164 | 44.598 | 9.7216 | 3.6824 | 80.633 | 81.604 |
| `rdoq_tx` QP106 | 24,794 | 99.176 | 44.644 | 9.7263 | 3.2211 | 80.814 | 82.490 |
| `rdoq_tx` QP108 | 24,051 | 96.204 | 44.517 | 9.7235 | 3.5630 | 80.317 | 81.765 |
| `rdoq050` QP106 | 25,131 | 100.524 | 44.616 | 9.7235 | 3.2706 | 80.497 | 82.327 |
| `rdoq050` QP108 | 23,925 | 95.700 | 44.618 | 9.7266 | 3.6579 | 80.569 | 81.659 |

### 5.2 Tool-matrix additions

| ID | exact extra flags | legal | K 106 | dK 106 | K 108 | dK 108 | kbps 106 / 108 | matched K (method) | verdict |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| `cfl` | `--video-cfl` | yes | 81.826 | −0.329 | 81.397 | −0.207 | 99.176 / 97.288 | 81.826 (nearest) | negative both |
| `ibp` | `--video-ibp` | yes | 81.246 | −0.909 | 80.638 | −0.966 | 98.588 / 96.708 | 81.246 (nearest) | negative both |
| `ist` | `--video-ist` | yes | 82.155 | 0.000 | 81.604 | 0.000 | 100.076 / 97.188 | 82.140 (linear) | inert: decoded pictures identical, +11 B sequence bit |
| `dip` | `--video-dip` | yes | 81.985 | −0.170 | 81.667 | +0.063 | 102.496 / 99.004 | 81.758 (linear) | sign flip, matched −0.391 |
| `idtx` | `--video-idtx` | yes | 81.956 | −0.199 | 81.861 | +0.257 | 99.672 / 96.880 | 81.956 (nearest) | sign flip, below control |
| `tcq_cctx` | `--video-rdoq-off --video-tcq --video-cctx` | yes | 81.150 | −1.005 | 80.930 | −0.674 | 95.316 / 92.852 | 81.150 (nearest) | negative both, 4.7% below target, 288/241 s encode |
| `cdfoff_mhccp` | `--video-cdf-chain-off --video-mhccp` | yes | 79.914 | −2.241 | 79.837 | −1.767 | 110.308 / 108.056 | 79.837 (nearest) | negative both, rate explosion |
| `cdefstrength4` | `--video-cdef-strength 4` | yes | — | — | — | — | — | — | filter-lane knob; outside this lane, not measured (left to filter) |

Notes:

* `cfl`, `ibp`, `idtx` and `tcq_cctx` ladders do not bracket 100 kbps. Their
  harness output is `nearest`, not a matched oracle. A monotone-slope
  extrapolation to 100 kbps is 82.01 (cfl), 81.70 (ibp), 81.97 (idtx) and 81.57
  (tcq_cctx), all below the control's 82.149; these are diagnostics only, not
  reported as matched-rate K.
* `ist` is a true no-op at this point: the decoded Y4M is byte-identical to the
  control at both QPs (the stream differs by the sequence `enable_intra_ist`
  bit only). No IST leaf is selected on this clip.
* `ibp` is the largest prediction loss. `cfl` is also negative at both QPs.
  `dip` and `idtx` flip sign between the two QPs, so they fail the two-QP rule
  even before the matched-rate check.
* `tcq_cctx` is legal only with `--video-rdoq-off`; with the nightly RDOQ on,
  the named rejection in §4 fires. It is 60–70× the encode cost for a stream
  that is 4.7% below the target rate at QP 106.
* `cdfoff_mhccp` is legal only because `--video-cdf-chain-off` removes the
  `--avg-cdf/--adaptive-cdf` that MHCCP rejects. Disabling the adaptive chain
  costs 10.3% rate at QP 106 (27,577 B vs 25,008 B) and MHCCP then loses
  another ~2.2 K. Negative.
* `--video-cdef-strength 4` parses and encodes (2-frame probe exit 0) but is a
  CDEF filter knob, not a transform/quant/entropy/rate tool. It is flagged in
  `docs/tool-matrix.md` as untapped and is left to the filter lane.

## 6. Rate-free (source-replacement) ceilings

The oracle replaces one segment of the **decoded** stream with source frames
and re-scores the 60-frame sequence. This is a pooling bound, not a stream.
The base point is the candidate's ladder point closest to 100 kbps.

| candidate | base | base kbps | key f0 | f0-32 | f33-43 | f44-59 | best ceiling |
|---|---|---:|---:|---:|---:|---:|---:|
| `ctl` | QP106 | 100.032 | 82.239 (+0.084) | 85.855 (+3.700) | 84.367 (+2.212) | 88.066 (+5.911) | **88.066** |
| `rdoq_tx` | QP106 | 99.176 | 82.577 (+0.088) | 85.993 (+3.504) | 84.822 (+2.332) | 88.459 (+5.969) | **88.459** |
| `rdoq050` | QP106 | 100.524 | 82.414 (+0.086) | 86.083 (+3.756) | 84.380 (+2.053) | 88.462 (+6.134) | **88.462** |

The best single-segment ceiling of the best internal combination is **88.459**
(`rdoq_tx`, f44-59 source). The best of any measured combination is **88.462**
(`rdoq050`), still **0.338 below 88.8**. The key and first-third segment
ceilings are far lower. No candidate can clear the wall even with a perfect
16-frame fade tail, so the gate in the task blocks a full QP 104/106/108
ladder for every candidate.

## 7. Combined-best attempt

The two surviving directions (trellis: `rdoq 0.2`; rate: `tx 24`) were stacked
with the entropy/header levers that had a positive same-QP screen
(`--video-tile-group-off --video-ref-signal-off --video-ccso-header-off`):

```text
--video --frames 60 --cpu-threads 8 --qp 106 --fast-decode \
  --video-rdoq-scale 0.2 --video-tx-lambda 24 \
  --video-tile-group-off --video-ref-signal-off --video-ccso-header-off
```

QP106: 25,538 B / 102.152 kbps / JOD 9.7154 / BA 3.3316 / SS2 80.406 / K
82.157; QP108: 24,777 B / 99.108 kbps / K 81.214. Two-point matched K 81.490,
**−0.659** versus control. The stacked overhead switches spend 2.98 kbps at
QP106 and lose the interaction. The two-lever subset `rdoq_tx_overhead`
(`--video-tile-group-off --video-ref-signal-off`) is recon-identical to
`rdoq_tx` but 0.41% larger (24,896 B vs 24,794 B), so it is dominated by
`rdoq_tx`; the third lever (`ccso-header-off`) is what changes the
reconstruction and loses.

The other obvious stack, `rdoq 0.05 + tx 24` (`rdoq050_tx24`), is also
negative: 81.231 / 99.864 kbps, matched 81.231, −0.918. The trellis and TX
weights must move together (0.2 with 24), not independently.

**Best internal combination: `rdoq_tx`.** It is not the combined preset; the
combined preset is worse.

## 8. Best internal combination, plainly

Exact flags at the measured point:

```sh
BIN=build-release/av2enc
REF=testdata/nasa_480x240_30fps_60f.y4m
"$BIN" "$REF" -o /tmp/rdoq_tx_106.ivf \
  --video --frames 60 --cpu-threads 8 --qp 106 --fast-decode \
  --video-rdoq-scale 0.2 --video-tx-lambda 24
```

| | control QP106 | `rdoq_tx` QP106 | Δ |
|---|---:|---:|---:|
| bytes | 25,008 | 24,794 | −214 (−0.86%) |
| kbps | 100.032 | 99.176 | −0.856 |
| JOD | 9.7334 | 9.7263 | −0.0071 |
| Butteraugli | 3.4202 | 3.2211 | **−0.1991** |
| SSIMULACRA2 | 80.837 | 80.814 | −0.023 |
| K | 82.155 | 82.490 | **+0.335** |

At QP108 the pair is 24,291 B / 81.604 (control) versus 24,051 B / 81.765
(`rdoq_tx`): +0.161 K at 1.0% lower rate. Both signs are positive, but the
QP108 delta is inside the documented ±0.2–0.5 QP ripple; only the QP106 delta
is at the edge of it. **No resolved gain is claimed.** The matched-rate
harness labels the two-point ladder `nearest` at 99.176 kbps / K 82.490; since
that is already a lower rate than the control's 100.000 kbps point, a
monotone-rate lower bound is **82.490 ≥ 82.149** (+0.341), but that is a bound,
not an oracle, and it does not justify a ladder under the gate.

The mechanism is a trellis/rate interaction: a more aggressive coefficient
trellis (`rdoq-scale 0.2`) combined with a much cheaper TX-search rate weight
(`tx-lambda 24`, half the 240p fit) removes 0.86% rate and improves
Butteraugli by 0.20 at essentially flat JOD/SS2. Alone, `rdoq 0.2` is −0.98 K
and `tx 24` flips sign (−1.18 / +0.08); the pair is the only positive
interaction found. `rdoq 0.05` alone is +0.172/+0.056 and matched +0.106, also
inside the ripple.

## 9. Gate decision and verdict

* Best internal combination: `rdoq_tx` (flags above), measured K **82.490** at
  **99.176 kbps**, QP106. Best matched-rate control: **82.149**. Current
  internal best: eff-perc **85.10**. SVT-AV1 p0: **88.83**.
* Best rate-free ceiling of any measured combination: **88.462**. Best of the
  best internal combination: **88.459**. Both are **below 88.8** by 0.34 K.
* Therefore the task's gate applies: **no full QP 104/106/108 ladder is run**
  for `rdoq_tx`, `rdoq050`, or any other row. No matched-rate K is claimed for
  them. The best internal combination is reported plainly above.
* No combination in this search beats eff-perc 85.10, and none is within 6.3 K
  of the 88.8 wall at its measured point. The lane is closed.
* The only positive mechanism is the `rdoq 0.2` + `tx 24` interaction
  (+0.335/+0.161 same-QP, −0.86%/−1.0% rate). It is inside the QP ripple and
  its ceiling is below the wall, so it is a candidate for a future allocation
  experiment, not a shippable preset.
* Entropy reset (`--video-cdf-per-frame-qp`) is quality-neutral at fixed QP:
  `cdfreset_rdoq` decodes to the same pictures as `rdoq050` at both QPs and
  costs 2.7% rate (matched −0.262 K). The carried CDF is the right default.
* The tool-matrix additions are all negative or inert: CfL −0.33/−0.21, IBP
  −0.91/−0.97, IST no-op, DIP/IDTX sign flips, TCQ+CCTX −1.0/−0.67 at 4.7%
  below target and 60–70× encode cost, CDF-off+MHCCP −2.24/−1.77. None merits
  a ladder.

## 10. Artifacts and reproduce

Oracle JSON (outside the repo, retained on this host, SHA-256):

| report | SHA-256 |
|---|---|
| `<cache>/combo-tx/json/screen_all.json` | `cf4570795d0f0f00a01614f703d23f7e9a46e1847e6adf79e931f6fff0788d7a` |
| `<cache>/combo-tx/json/screen_extra.json` | `5dc2a20008caf4b2f5557a6a952ccf88d897fe582d677516ff44e6776cb1139a` |
| `<cache>/combo-tx/json/screen_matrix.json` | `fa5fd6bff1e9a9b84e4544f78182fbc5d5e9f04c099ec03c5db377a2402f01c0` |
| `<cache>/combo-tx/json/segments_best.json` | `edcf20bd24b590b1cc68262f01a66b656d735c35cc0c463c9f97872517369817` |

Reproduce one candidate (the rest are the same with the registry's extra
flags):

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
BIN=build-release/av2enc
for q in 106 108; do
  "$BIN" "$REF" -o /tmp/rdoq_tx_$q.ivf --video --frames 60 \
    --cpu-threads 8 --qp $q --fast-decode \
    --video-rdoq-scale 0.2 --video-tx-lambda 24
  tools/bin/dav2d-macos -q --strict 1 -i /tmp/rdoq_tx_$q.ivf \
    -o /tmp/rdoq_tx_$q.y4m --threads 1
done
python3 tools/oracle.py --ref "$REF" \
  --ivf rdoq_tx@106=/tmp/rdoq_tx_106.ivf \
  --ivf rdoq_tx@108=/tmp/rdoq_tx_108.ivf \
  --target-kbps 100 --json /tmp/rdoq_tx.json
```

No source file outside this document was modified. No stream was pushed or
merged.
