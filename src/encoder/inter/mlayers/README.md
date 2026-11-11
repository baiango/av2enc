# Embedded layers (`mlayers`)

Encoder-side AV2 v1.0.0 syntax and validation for **embedded layers**
(`obu_mlayer_id`) within one extended layer (`obu_xlayer_id`). This is the
layer machinery that was marked **ABSENT** in
`docs/spec-audit/01-obus.md` §3.1 and `docs/spec-audit/02-sequence-header.md`
§5.4.1.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §5.2.2, §5.4.1, §6.2.2, §6.4.1, §7.3.6,
§7.3.7, §7.3.8.6.

**Component only.** Nothing outside `src/encoder/inter/mlayers/` is edited:
no `CMakeLists.txt`, `main.cpp`, core writer or gate is touched. There is no
production CLI path yet; the gate is the integration proof.

## What this implements

| Syntax / rule | Spec | Writer | Reader |
|---|---|---|---|
| `obu_header_extension_flag`, `obu_mlayer_id f(3)`, `obu_xlayer_id f(5)` | §5.2.2 p54 | `make_obu_header`, `make_obu`, `make_obu_header_bytes` | `parse_obu_header`, `parse_obu` |
| `GLOBAL_XLAYER_ID` inference for TD/MSDO when the extension flag is 0 | §5.2.2 p54 | `inferred_obu_header_fields` | same |
| `seq_max_mlayer_cnt_minus_1 f(n)`, `n = CeilLog2(max_mlayer_id+1)` | §5.4.1 p56 | `write_sequence_header_payload` | `parse_sequence_header_obu` |
| `mlayer_dependency_present_flag f(1)` + `mlayer_dependency_map f(1)` per current/ref layer | §5.4.1 p57 | same | same |
| `tlayer_dependency_present_flag f(1)` + `tlayer_dependency_map f(1)` | §5.4.1 p57-58 | same | same |
| `multi_tlayer_dependency_map_present_flag f(1)` + per-layer tlayer maps | §5.4.1 p57-58 | same | same |
| Effective `MLayerDependencyMap`, `TLayerDependencyMap` (defaults + overrides) | §5.4.1 p57 | `default_dependency_maps` | parsed into `ParsedSequenceHeader::dependencies` |
| Derived `MLayerPresenceMap` | §5.4.1 p58 | `derive_mlayer_presence_map` | same, in `ParsedSequenceHeader::presence` |
| Reflexivity / transitivity / range / lower-triangle validation | §5.4.1, §6.4.1 | `validate_*_dependency_map(s)` | reader re-validates the spec dialect |
| `SeqMaxMlayerCnt` range rule (`<= max_mlayer_id + 1`) | §6.4.1 p261 | `validate_sequence_layer_config` | reader rejects out-of-range |
| OBU layer-id constraints, GLOBAL scope, forced zero mlayer/tlayer | §6.2.2 p257 | `validate_obu_header_fields` | reader |
| Coded extended layer unit ordering, mlayer ascending, CLK/OLK rules, equal OrderHint | §7.3.6 p383 | `LayerStreamAuditor` | — |
| Extended-layer ascending order per temporal unit | §7.3.7 p385 | `LayerStreamAuditor` | — |
| Sequence-header availability before a referencing frame | §7.3.8.6 p387 | `LayerStreamAuditor` | — |
| Inter-layer availability (presence map + earlier output frame) | §5.4.1 p58, §6.4.1 | `LayerStreamAuditor` | — |

### Dependency-map validation

For `MLayerDependencyMap[c][r]` and every `TLayerDependencyMap[m][c][r]`:

* lower-triangular: `r > c` must be 0;
* in range: no dependency on a layer above `max_mlayer_id` /
  `max_tlayer_id`, and no bits for undeclared embedded layers;
* reflexive: every declared layer depends on itself;
* transitive: `c -> k` and `k -> r` implies `c -> r`;
* the spec default is `r <= c` (and `m <= max_mlayer_id`), used when the
  corresponding `*_dependency_present_flag` is 0;
* `multi_tlayer_dependency_map_present_flag = 0` requires every embedded
  layer's tlayer map to equal `TLayerDependencyMap[0]`.

`MLayerPresenceMap` is derived with the exact §5.4.1 p58 loop (identity for
all 8 `mlayerId` values, then the transitive propagation from the dependency
row). It is not a hand-rolled closure.

### OBU-order / availability audit

`LayerStreamAuditor` consumes `LayerObuRecord`s in emission order. It rejects:

* invalid OBU/header combinations (global xlayer on a frame, nonzero
  mlayer/tlayer on sequence headers/LCR/OPS/atlas, nonzero tlayer on
  CLK/OLK/SWITCH/RAS);
* a frame whose `seq_header_id_in_frame_header` has not been observed for the
  same extended layer, or whose layer ids exceed that header's maxima;
* descending `obu_mlayer_id` or `obu_xlayer_id` order;
* more than one output frame unit per embedded layer, or a non-output unit
  without an output unit in the same layer;
* mismatched `OrderHint` between output frame units of one coded extended
  layer unit;
* a coded extended layer unit with no output frame unit;
* a lowest present embedded layer whose first frame is not CLK/OLK when the
  unit contains a CLK/OLK, or a CLK/OLK that is not the first frame of its
  layer;
* an inter-layer reference outside `MLayerPresenceMap` or without an earlier
  output frame unit in the same temporal unit;
* SWITCH/RAS frames that declare a dependency on another embedded layer.

## API

Header: `src/encoder/inter/mlayers/mlayers.h`

| Symbol | Purpose |
|---|---|
| `MLayerDependencyMap`, `MLayerPresenceMap`, `TLayerDependencyMap`, `TLayerDependencyMaps` | 8×8 / 4×4 boolean maps |
| `LayerDependencyMaps` | both maps in one struct |
| `default_dependency_maps(max_m, max_t)` | §5.4.1 p57 defaults |
| `derive_mlayer_presence_map(map, max_m)` | §5.4.1 p58 derivation |
| `validate_mlayer_dependency_map`, `validate_tlayer_dependency_maps`, `validate_dependency_maps` | closure/reflexivity/transitivity/range |
| `mlayer_dependencies_satisfied`, `tlayer_dependencies_satisfied` | selected-layer closure |
| `SequenceLayerConfig` | `max_mlayer_id`, `max_tlayer_id`, `seq_max_mlayer_cnt`, three flags, maps |
| `default_sequence_layer_config(max_m, max_t)` | defaults with inferred maps |
| `validate_sequence_layer_config` | field and map consistency |
| `LayerObuRecord`, `LayerStreamAuditor` | §7.3.6/§7.3.7/§7.3.8.6 audit |

Header: `src/encoder/inter/mlayers/mlayers_syntax.h`

| Symbol | Purpose |
|---|---|
| `ObuType` | §6.2.2 Table 6.1 values |
| `ObuHeaderFields`, `inferred_obu_header_fields`, `validate_obu_header_fields` | §5.2.2 header model/inference/checks |
| `make_obu_header_bytes`, `make_obu_header`, `parse_obu_header` | 1/2-byte OBU header writer and reader |
| `SequenceSyntaxDialect` | `kSpecV1_0_0` (normative), `kPinnedDav2dCompat` (gate escape hatch) |
| `SequenceHeaderConfig`, `baseline_sequence_header_config` | M2-compatible baseline plus layer fields |
| `write_sequence_header_payload`, `make_sequence_header_obu` | §5.4.1 writer |
| `ParsedSequenceHeader`, `parse_sequence_header_obu` | independent reader (fields, effective maps, presence map, trailing bits) |
| `make_obu`, `parse_obu` | Annex B.2 framing |
| `KeyFrameHeader`, `write_key_frame_header` | §5.18.2 CLK frame-header bits; adds `refresh_frame_flags f(NumRefFrames)` when `max_mlayer_id > 0` |

`write_sequence_header_payload` reports the exact syntax bit count so the
caller can append `obu_extension_flag = 0` + `trailing_bits()` without
accidentally byte-aligning the sequence header early.

## Pinned goldens

`golden_reference.py` is an independent Python bit writer for §5.2.2 and
§5.4.1. It prints the byte strings embedded in `mlayers_test.cpp`:

| Golden | Configuration |
|---|---|
| `kGoldenBaseline64` | 64×64, `max_mlayer_id = max_tlayer_id = 0`, spec |
| `kGoldenM1Inferred64` | 64×64, `max_mlayer_id = 1`, `SeqMaxMlayerCnt = 2`, inferred mlayer map |
| `kGoldenM1Explicit64` | same, explicit mlayer map |
| `kGoldenM2T2Multi320` | 320×180, `max_mlayer_id = 2`, `max_tlayer_id = 2`, multi tlayer maps |
| `kGoldenM2T2Single320` | same, layer-0 tlayer map only |
| `kGoldenM7Inferred64` | 64×64, `max_mlayer_id = 7`, `n = 3` |
| `kGoldenCompatM1_64` | 64×64, pinned-dav2d compat dialect |
| `kGoldenObuSeq/Td/ClkM1/ClkX5/Msdo` | §5.2.2 header bytes, including TD/MSDO GLOBAL inference |

`kGoldenBaseline64` is also checked against
`av2enc::m2::make_sequence_header_obu()`, so the baseline remains
byte-identical to the M2 writer.

## Tests

```sh
./src/encoder/inter/mlayers/build_and_test.sh
SANITIZE=1 ./src/encoder/inter/mlayers/build_and_test.sh --no-gate
WERROR=1   ./src/encoder/inter/mlayers/build_and_test.sh --no-gate
```

`mlayers_test` (35,223 checks) covers:

* all pinned goldens and independent-reader round-trips;
* **exhaustive** all lower-triangular `MLayerDependencyMap`s for
  `max_mlayer_id = 3` (2⁹ = 512 maps), with presence-map equality against a
  brute-force reflexive-transitive closure;
* **exhaustive** all per-layer tlayer maps for `max_tlayer_id = 2` (2⁵ = 32),
  all 32² pairs with `max_mlayer_id = 1`, and all 32³ triples with
  `max_mlayer_id = 2`;
* reflexivity, upper-triangle, undeclared-layer and transitivity negatives;
* `SequenceLayerConfig` acceptance/rejection rules;
* OBU header inference/validation and the stream auditor’s rejection cases.

## Decoder gate

```sh
DAV2D=$PWD/tools/bin/dav2d-macos ./src/encoder/inter/mlayers/build_and_test.sh
```

`mlayers_gate` builds one temporal unit:

```
TD(xlayer = GLOBAL) + SEQ(max_mlayer_id = 1, SeqMaxMlayerCnt = 2)
+ CLK(obu_mlayer_id = 0) + CLK(obu_mlayer_id = 1)
```

Each CLK is an independent M2 intra encode (one picture per embedded layer;
no inter-layer prediction). The gate checks the OBU layer ids, parses the
sequence header back, runs `LayerStreamAuditor`, and invokes:

```sh
dav2d -q -i stream.ivf -o out.yuv --muxer yuv --alllayers 1 --strict 1 --threads 1
```

Checks:

* the two-layer unit decodes to exactly `recon(m0) || recon(m1)` (0 diffs);
* an mlayer-0-only unit decodes to `recon(m0)`;
* an mlayer-1-only unit decodes to `recon(m1)`.

Both 64×64 and 640×360 pass (640×360 is exercised by the build script).

### Pinned `dav2d` caveat

`dav2d f9dafbc` (2026-11-10) does **not** parse the v1.0.0
`seq_max_mlayer_cnt_minus_1` field. Its parser reads `max_mlayer_id` and then
`monotonic_output_order_flag` immediately, and it reads the dependency blocks
in a different order (`tlayer` before `mlayer`) with `n = 1..max-1` and an
`n`-bit mask per row. A normative `kSpecV1_0_0` stream is therefore rejected
with `Error parsing sequence header / Overrun in OBU bit buffer`.

The gate therefore tries `kSpecV1_0_0` first and falls back to
`kPinnedDav2dCompat`, which omits the count field and (for nonzero flags) uses
the decoder’s row syntax. The gate prints:

```
note: pinned dav2d f9dafbc does not parse seq_max_mlayer_cnt_minus_1;
      gate used the compat dialect
```

The normative writer and its bytes are pinned by `mlayers_test`; only the
black-box decoder check uses the compat form. This is the same class of
decoder/spec mismatch already documented for `tlayer_dependency_map` in
`src/encoder/inter/tlayers/README.md`.

The compat stream is a decoder-compatibility vehicle, **not** a conforming
bitstream: it cannot signal `SeqMaxMlayerCnt`, so a decoder infers 1 while the
stream contains two distinct embedded layers. The gate therefore runs the full
`LayerStreamAuditor` (including the §6.4.1 p261 count rule) on the normative
stream, and disables only that one rule for the compat stream via
`set_enforce_seq_max_mlayer_cnt(false)`. OBU ordering and availability are
still checked on the compat stream.

## Sites that currently reject `max_mlayer_id > 0`

These are intentionally **not edited** (ownership boundary). To wire a
production multi-embedded-layer stream they need to be relaxed or routed to
this component:

1. `src/encoder/inter/longterm/longterm_headers.cpp:193-197` —
   `validate_sequence_header_config()` returns an error for
   `max_mlayer_id != 0` (“needs seq_max_mlayer_cnt_minus_1 and
   mlayer_dependency_present_flag/map … not implemented”). The writer at
   `:254` would write the field, but the validator runs first.
2. `src/encoder/svc/svc_config.cpp:79-86` —
   `single_picture_header_flag` is rejected unless
   `spatial_layers.size() == 1 && temporal_layers == 1 &&
   embedded_layers == 1`. Every current SVC input is a single-picture still,
   so `embedded_layers > 1` is rejected here.
3. `src/encoder/tiles/svc_layers.cpp:280-290` —
   `validate_sequence_layer_config()` forces `max_mlayer_id = 0`,
   `max_tlayer_id = 0` and `SeqMaxMlayerCnt = 1` when
   `single_picture_header = true`, even if the plan set
   `embedded_layers > 1`.
4. `src/encoder/svc/svc_framing.cpp:862-881` —
   `compose_layered_stream()` rewrites only `seq_lcr_id` in the input
   sequence header; it does not emit `max_mlayer_id`,
   `seq_max_mlayer_cnt_minus_1` or the dependency blocks. Non-single SVC
   composition needs this writer (or a splice of
   `write_sequence_header_payload`).
5. `src/encoder/svc/svc_framing.cpp:766-773` —
   composed inputs must be `m0/t0`; the plan can still assign mlayer ids to
   OBUs, but the sequence header stays at `max_mlayer_id = 0`.
6. Readers that reject nonzero maxima (useful as a list of places that assume
   one embedded layer): `src/encoder/inter/random_access/random_access.cpp:432-433`
   (`max_mlayer_id must be 0`), and
   `src/encoder/inter/longterm/longterm_headers.cpp:830-831` (expects the
   sequence header to match a `max_mlayer_id == 0` config).

Writers that hardcode 0 (not rejections, but no path): 
`src/encoder/m2/m2_headers.cpp:108`,
`src/encoder/inter/tlayers/tlayers_syntax.cpp:122`,
`src/encoder/inter/random_access/random_access.cpp:271`,
`src/encoder/entropy/avg_cdf/gate_dav2d_avg_cdf.cpp:122`,
`src/encoder/inter/gme/gme_gate.cpp:195`,
`src/encoder/inter/drl_reorder/gate/drl_stream.cpp:102`.

## Call sites in this component

No production call site exists yet. The intended integration points are:

* a non-single-picture sequence writer calls
  `mlayers::write_sequence_header_payload()` (or splices its layer bits) after
  selecting `default_sequence_layer_config(embedded_layers - 1,
  temporal_layers - 1)`;
* the OBU framer calls `mlayers::make_obu()` with the per-OBU
  `(xlayer, mlayer, tlayer)` triple so the extension flag and inferred global
  id are correct;
* a layer scheduler runs `mlayers::LayerStreamAuditor` over the planned
  emission order before writing.

`mlayers_gate.cpp` and `mlayers_test.cpp` are the current call sites.
