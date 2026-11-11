# Transform-partition dav2d gate

`build_gate.sh` proves §5.20.6 TX partitioning against the pinned `dav2d`
without editing a repository source file.  It follows the same pattern as
`src/encoder/m2/ddt/gate/`: copy the two M2 writer files into the build
directory, patch the copies, compile them instead of the originals, and fail
loudly if an anchor moved.

## Why a header splice is not enough

Flipping `tx_mode_select` in an existing `TX_MODE_LARGEST` stream would make
the decoder read `tx_do_partition` from what are actually coefficient bits and
desynchronise.  The encoder must emit the partition syntax *and* must code,
dequantise and reconstruct one transform per `set_tx_size()` part.  The gate
therefore builds a stream where both sides agree by construction.

## Patched copies

| Copy | Patch | Edits |
|---|---|---|
| `m2_syntax.cpp` | `patch_m2_syntax.py` | 1 include; one injected `TxGateSink` + `txgate_code_luma()` member; one call-site swap in `write_inter_residual()` |
| `m2_headers.cpp` | `patch_m2_headers.py` | 1 include; `tx_mode_select` reads `!key && gate::tx_mode_select_enabled()` |

Both scripts assert each anchor occurs exactly once.  The production files are
byte-identical to the repository.

`txgate_code_luma()`:

1. calls `txpartition::write_tx_partition()` with the plan supplied by the
   gate (partition type + per-part `mi_row/mi_col/tx_size/eob/levels_offset`);
2. calls the stock `code_tx()` once per part with a `BlockSyntax` copy whose
   origin, transform size, EOB and level offset describe that part.

Everything else — intra modes, inter mode info, chroma, entropy contexts — is
the stock M2 writer.  Because `code_tx()` derives the `txb_skip`,
`coeff_base`, `dc_sign` and TX_TYPE contexts from the part origin and size,
context derivation stays decoder-exact without any extra plumbing.

## Blocked core edits (production integration)

These are the edits the component cannot make under the ownership split:

1. `src/encoder/m2/m2_headers.cpp:289` — write `tx_mode_select` from the frame
   decision.
2. `src/encoder/m2/m2_syntax.cpp` `write_inter_residual()` — emit
   `read_tx_partition()` and code one transform per part (the gate's
   `txgate_code_luma()`).
3. `src/encoder/m2/m2_syntax.cpp` `CdfState` / `init_cdfs()` — own a
   `TxPartitionCdfs` per tile.  The gate uses one static copy because its tile
   has one split block and `disable_cdf_update = 1`.
4. `src/encoder/m2/residual/inter_residual.cpp:280,338` — per-part forward
   transform, quantiser and inverse reconstruction; the current code has one
   transform per leaf.
5. `src/encoder/m2/rdo/residual_decision.cpp` — add the partition syntax cost
   to the rate term.
6. `src/encoder/m2/m2_frame.cpp:793,825` — carry a per-part transform list in
   the leaf decision instead of one `tx_size/eob` per plane.

## What the gate runs

Deterministic 64×64 4:2:0, `base_q_idx = 90` (so `qlookup == txv::kQuantStep`),
two frames:

* **frame 0** — intra keyframe, `TX_MODE_LARGEST`, DC prediction, all-zero
  residual.  Reconstruction is 128 everywhere, so the P-frame prediction is
  exactly known.
* **frame 1** — P frame, `tx_mode_select = 1`.  Every block is inter
  `GLOBALMV` with MV 0; all but one are `skip_flag = 1`.  The first leaf is
  non-skip: it signals the partition and codes a deterministic non-zero
  residual in every part.  Chroma is all-zero.

The encoder-side reconstruction is built from the same `txv::forward_transform`
/ `inverse_transform` calls the writer used, so the comparison is a true
encoder-vs-decoder check, not a byte comparison.  Cases D/E additionally
cross-check the §5.20.7.30 rectangular scan: the gate orders each part's
levels with `txpartition::get_scan()` while the stock M2 writer reads them with
its own `build_scan()`; a scan mismatch would shift coefficients and fail the
sample comparison.  Every part has `eob > 0`, so each transform is
reconstructed from real coefficient data.

```
./build_gate.sh
...
TX PARTITION GATE PASS: 7/7 case(s) passed
```

| Case | Coding block | Partition | Parts | tile1 |
|---|---|---|---|---:|
| A | 64×64 | SPLIT | 4× `TX_32X32` | 916 B |
| B | 16×16 | SPLIT | 4× `TX_8X8` | 115 B |
| C | 8×8 | SPLIT | 4× `TX_4X4` | 103 B |
| D | 16×16 | HORZ | 2× `TX_16X8` | 109 B |
| E | 16×16 | VERT | 2× `TX_8X16` | 112 B |
| F | 16×16 | NONE | 1× `TX_16X16` | 20 B |
| G | 16×16 | SPLIT, header `tx_mode_select = 0` | — | negative control |

Cases A–F require `dav2d --strict 1 --threads 1` to reproduce both frames
sample-for-sample.  Case F is the control that shows the gate is not merely
reconstructing “whatever the decoder did”: with `NONE` the same coefficients
are coded as a single 16×16 transform.  Case G keeps the split syntax but tells
the decoder `tx_mode_select = 0`; the P frame then fails to decode (or decodes
to different samples), which proves the gate is sensitive to the field.

`--case A..G` runs one case; `--no-run` builds only.
