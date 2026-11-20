# Bit census and decision diff

The matched-rate gate answers one question: did K move? It cannot say *where*
the bits or the decisions moved. This diagnostic does, without touching the
bitstream.

Two env vars, both default off:

```sh
AV2ENC_BIT_CENSUS=/tmp/base.jsonl \
AV2ENC_DECISION_DUMP=/tmp/base.dec \
  build-release/av2enc clip.y4m -o /tmp/base.ivf --video ... --qp Q
```

- `AV2ENC_BIT_CENSUS` appends one JSON object per tile: the information bits
  the §8.2 coder charged, split by symbol family (`classes`) and by residual
  sub-family (`fine`), plus `payload_bytes` and the frame ordinal.
- `AV2ENC_DECISION_DUMP` appends one JSON object per coded leaf: mode, MV,
  TX, EOB, segment and qindex.

Both are diagnostics only. A run with them produces the same IVF as a run
without them (verified against the `--fast-decode` QP110 pin, 23,690 B /
`1b99127a...`).

## Comparing two runs

At the **same QP**:

```sh
python3 tools/bit_census.py --base base.jsonl --cand cand.jsonl \
    --decisions base.dec cand.dec --top 10
```

The tool prints:

- payload bytes and information bits (so a QP mismatch is visible);
- every coarse class and every residual sub-family, base/cand/delta;
- the per-frame bit deltas, largest first;
- leaf counts, removed/added leaves, and flip counts for every decision field.

## What it shows

Control 1: the carried CDF chain vs `--video-cdf-per-frame-qp` at QP 110. The
decisions are identical (3,152/3,152 leaves, 0 flips), so this isolates the
entropy layer:

| family | base bits | cand bits | delta |
|---|---:|---:|---:|
| inter_mode | 3,698 | 7,887 | **+4,189** |
| partition | 3,066 | 3,719 | +652 |
| tx_partition | 1,159 | 1,453 | +294 |
| skip | 587 | 757 | +170 |
| coeff_base | 80,038 | 80,144 | +106 |
| **total** | 173,226 | 178,953 | **+5,727 (+716 B)** |

That is the whole explanation of the +703 payload bytes: the per-frame reset
prices the inter/partition/DRL symbols from the §9.3 defaults instead of the
carried image. No coefficient-coding decision changed.

Control 2: key QP `-26` instead of `-27`. Decision flips and bits move
together:

| field | flips |
|---|---:|
| inter_mode | 434 |
| eob_y | 405 |
| bsize | 357 |
| tx_size | 309 |
| tx_partition | 248 |
| mv_y | 224 |
| tx_type | 198 |
| skip | 96 |

| family | base bits | cand bits | delta |
|---|---:|---:|---:|
| coeff_base | 80,038 | 78,839 | **-1,199** |
| coeff_mag | 51,687 | 50,710 | -977 |
| eob | 8,989 | 8,759 | -230 |
| inter_mode | 3,698 | 3,476 | -222 |
| **total** | 173,226 | 170,491 | **-2,735 (-342 B)** |

244 leaves disappeared and 190 appeared, i.e. the key QP change re-partitioned
the key and shifted the P-chain decisions. The bit census attributes the saving
to coefficient base/magnitude, not to a header row.

## Reading it

- `coeff_base`, `coeff_br`, `coeff_mag`, `eob`, `sign`, `tx_type`,
  `tx_partition`, `txb_skip` are the residual families. `coeff_mag` is the
  raw/bypass magnitude tail inside the coefficient coder; the other families
  are the CDF-coded symbols.
- `raw_bits` is every literal/bypass bit outside the residual coder (MV shell
  offsets and signs, filter rows, header-ish literals). If a change moves
  `raw_bits` and not a coefficient family, it changed motion or filter syntax.
- The census covers tile payloads only. Sequence/frame/OBU headers are not in
  it; a pure header change (e.g. compact reference signalling) is invisible
  here by design.
- `info_bits` is within 422 bits (0.88 B/tile) of the payload bits on the
  60-frame nightly stream; the remainder is the §8.2 marker and byte padding.
- Compare at one QP. At matched rate the QP differs and the census mixes the
  QP change with the tool change; use the matched-rate K for the rate decision
  and the census for the mechanism.

## Caveats

- The frame ordinal is set by the key/P loops in the flat chain. Frame-type
  wrappers (long-term/OLK/leading/SEF/SWITCH/RAS, B and temporal-layer
  schedules) currently report `-1`, so their leaves collapse into one bucket;
  the KPI point and the flat P chain are exact.
- One process appends; parallel tile writers append separate lines and the
  order within a frame is not guaranteed. The frame/tile origin fields are the
  identity, not the line order.
- This is analysis tooling. It must never feed a decision or a rate model; the
  symbols it measures are already committed.
