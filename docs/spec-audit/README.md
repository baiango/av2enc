# AV2 v1.0.0 spec audit — results

Fifteen independent audits of `spec/AV2_Spec_v1.0.0.pdf` (1,169 pages) against
this repository. Each area was read directly from the PDF, then every syntax
element was traced into the actual writers/readers. Documentation and status
lines were **not** trusted; code and tests were.

Reports: `01-obus.md` … `15-cli-reachability.md`.

## Status vocabulary

| Status | Meaning |
|---|---|
| WIRED | reachable from the shipped `av2enc` binary |
| COMPONENT | implemented, tested and usually gated, but no production caller |
| PARTIAL | some fields/branches missing, or emit-only/read-only |
| ABSENT | no implementation |
| DECODER-ONLY | no encoder obligation; noted whether the encoder can exercise it |
| OUT-OF-SCOPE | atlas, MSDO, bridge, padding, inter-layer prediction, 12-bit |

## Global totals

Syntax-row audits (01–11): **1,098 items**.

| Status | Rows | Share |
|---|---:|---:|
| WIRED | 259 | 23.6% |
| COMPONENT | 425 | 38.7% |
| PARTIAL | 230 | 20.9% |
| ABSENT | 149 | 13.6% |
| DECODER-ONLY | 20 | 1.8% |
| OUT-OF-SCOPE | 14 | 1.3% |
| N/A | 1 | 0.1% |

Only **23.6%** of the spec surface is reachable today. **73%** is
component-only, partial, or absent — the encoder is a verified still-image
encoder with a large, gated library behind it.

## Per-area results

| # | Area | Items | Wired | Component | Partial | Absent | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| 01 | OBUs | 267 | 29 | 144 | 51 | 26 | +6 decoder-only, 11 out-of-scope |
| 02 | Sequence header | 173 | 55 | 32 | 67 | 19 | all §5.4.1–5.4.13 |
| 03 | Frame header | 102 | 6 | 45 | 22 | 26 | +3 out-of-scope |
| 04 | Tile/block syntax | 167 | 51 | 74 | 11 | 29 | +2 decoder-only |
| 05 | Decode/ordering | 79 | 8 | 24 | 21 | 19 | +7 decoder-only |
| 06 | Motion | 44 | 11 | 17 | 6 | 10 | |
| 07 | Prediction | 73 | 23 | 32 | 15 | 2 | +1 N/A |
| 08 | Transform/recon | 73 | 26 | 18 | 22 | 7 | |
| 09 | Entropy | 30 | 20 | 8 | 2 | 0 | descriptors exact |
| 10 | Filters | 69 | 30 | 23 | 9 | 7 | |
| 11 | Output/grain | 21 | 0 | 8 | 4 | 4 | +5 decoder-only |

Table and constraint audits:

| # | Area | Result |
|---|---|---|
| 12 | §9.1–9.3 tables | 220 audited: 208 present, **12 absent**, 26 unused, **0 mismatches** |
| 13 | §9.4–9.8 tables | 29 audited: **29 present**, 0 absent, 10 wired, 19 unwired, **0 mismatches** |
| 14 | Conformance | 54 constraints: 3 wired, 18 partial, **33 absent** |
| 15 | CLI reachability | 82 sequence flags: **12 reachable**, 64 implemented-unreachable, 6 absent; 131 component dirs, 91 gate/test-only |

## The five findings that change the plan

1. **The product is a still-image encoder.** `src/main.cpp` writes
   `single_picture_header_flag = 1`, and the M2 encoder is **not in
   `CMakeLists.txt`** — all video/inter code is gate-only.
2. **Table data is correct; wiring is not.** 260,227 values in §9.4–§9.8 and
   208/220 in §9.1–§9.3 match the PDF. 19 transform/LR tables and 21 CDFs are
   recovered but unreachable.
3. **Two-thirds of block syntax is gate-only.** Transform partition
   (`TX_MODE_SELECT`) is entirely absent; M1/M2 are locked to one block-sized
   transform. Multi-tile-group framing and `dc_sign_horz_vert` are also missing.
4. **Level conformance is not met.** No `MaxPicSize`/`MaxHSize`/`MaxVSize`
   check; the default ≥3-GPU layout (`16x2`) exceeds level 5.1 `MaxTileCols`;
   `MaxLevelRefFrames` is only an `assert`.
5. **CDF state never crosses a frame.** `disable_cdf_update` is hardcoded and
   the avg-CDF bank is uncalled, so §7.5 is untested in production.

## Conformance bugs found (not just gaps)

| # | Bug | Where | Severity |
|---|---|---|---|
| B1 | `df_delta_q` not written when `df_delta_q_present = 1` | `m2/post/loop_filters.cpp` | **decoder desync** |
| B2 | MFH omitted `obu_extension_flag` | `frame/mfh/mfh.cpp` | **fixed** `bf2c63d` |
| B3 | QM `dequant_denom` used `tx_type < 9` instead of `TX_CLASS_2D`; missing clip | `qm/` | fix in review |
| B4 | `seq_tier` written unconditionally even when `seq_level_idx <= 3` | M2/leading/tlayers/longterm | latent |
| B5 | Chroma writer can set `seq_tile_info_present_flag=1` without `allow_tile_info_change`/`tile_params` | `format/chroma` | latent |
| B6 | Long-term writer omits mlayer dependency syntax when maxima > 0 | `inter/longterm` | latent |
| B7 | SVC plans `seq_lcr_id = 1` but re-wraps sequence headers unchanged | `svc` | dangling LCR |
| B8 | RAS can write `refresh_frame_flags` where §5.18.2 derives it | `inter/random_access` | gate-only mode |
| B9 | OLK long-term list can be omitted (pinned-decoder compat mode) | `inter/longterm` | gate-only mode |
| B10 | `film_grain_params_present = 1` reachable with no grain config | `format/chroma` | **would desync** |

B8/B9 are explicit, switchable compatibility modes with the normative path as
default; they are acceptable for gates but must not ship enabled.

## Audit coverage vs the 45-item feature list

The audits confirmed the 45-item list and added these items that were missing:

- transform partition / `TX_MODE_SELECT`;
- multi-tile-group framing (`frame_header_present_flag`, `tg_start`/`tg_end`,
  `bru_tile_active`);
- `dc_sign_horz_vert`;
- the 12 absent §9.2 tables (`Mode_To_Txfm`, `Wedge_Bits`, tile scaling
  factors, …);
- four conformance defects above (B1, B4–B7);
- embedded-layer / LCR / OPS signalling being skeletal rather than complete.

## What this means for ordering

The refactor still comes before integration (see `docs/refactor-plan.md`).
These audits add three preconditions to the integration pass:

1. add M2 to the build and a CLI path, or the video features remain unreachable;
2. fix B1 and B3 before any stream claims conformance;
3. keep B8/B9/B10 unreachable by default.

Table provenance and duplication defects are folded into the refactor's
"generated files" and "shared math" phases.
