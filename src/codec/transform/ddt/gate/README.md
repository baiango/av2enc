# DDT decoder gate

`build_gate.sh` proves the normative DDT inverse against the pinned dav2d
decoder without editing any core file.

## Why a bare header splice is not enough

The obvious construction — take an existing M2 stream, flip the
`enable_inter_ddt` bit in its sequence header, decode — is a valid *decoder*
test only if the encoder reconstruction used the same DDT kernels.  The stock
M1/M2 encoder uses `txv::forward_block()` / `txv::inverse_block()`, which know
only DCT/ADST/FDST/IDT.  Flipping the bit makes dav2d reconstruct ADST/FDST
blocks with DDTX/FDDT while the encoder's own reconstruction still used
ADST/FDST, so an encoder-vs-decoder comparison fails by construction (we use
that failure as the negative control).  A splice over a purely DCT stream
would be valid but would not exercise DDT at all.

The gate therefore uses build-directory copies of three M2 files:

| Copy | Patch | Why |
|---|---|---|
| `video_syntax.cpp` | `patch_video_syntax.py` | signal the decided non-DCT `inter_tx_type` instead of hard-coded DCT_DCT |
| `inter_residual.cpp` | `patch_inter_residual.py` | one call site: reconstruction inverse → `ddt::inverse_block()` |
| `video_headers.cpp` | `patch_video_headers.py` | one bit: `enable_inter_ddt` reads `DDT_GATE_ENABLE` |

The forward transform, quantiser, partition search and coefficient writer stay
stock.  The patch scripts fail loudly if their anchors move.

## Blocked core edits (for a production integration)

These are the edits the component cannot make under the ownership split, and
which the gate works around with build-dir copies:

1. `src/encoder/m2/video_headers.cpp:195` — write `enable_inter_ddt` from config.
2. `src/encoder/m2/residual/inter_residual.cpp:338` — call
   `ddt::inverse_block()` for reconstruction.
3. `src/encoder/m2/residual/inter_residual.cpp:280` — call
   `ddt::forward_block()` (and `level_from_coeff()`/`quant_factor()`) so the
   encoder analyses with the kernel the decoder will use.
4. `src/encoder/m2/residual/inter_residual.cpp:473` — RDOQ quant step must use
   `66/dq_denom` with the DDT forward.
5. `src/encoder/m2/residual/inter_tx_type.cpp:228,284,309,334,351` — the
   `inter_tx_type_is_ddt()` filter has the wrong meaning (identity ≠ DDT) and
   `kInSetInter[6]` is missing types 8..11 for INTER_2.  See the component
   README §7.
6. `src/encoder/m2/residual/inter_coeff_syntax.cpp:266` and
   `src/encoder/m2/rdo/residual_decision.cpp:19` — stop hard-coding `false`.

Until (1)–(4) land, the production encoder cannot emit a DDT stream whose
reconstruction matches a decoder; the gate demonstrates the decoder side is
correct and that the full path works when those call sites are DDT-aware.

## What the gate runs

* deterministic 3-frame 256x256 4:2:0 clip (`make_motion_clip.py`) built from
  `testdata/shine_256x256_1f.y4m`;
* DDTX column: `AV2ENC_VIDEO_RESIDUAL_TX_TYPE=1` (type 1 `ADST_DCT`);
* DDTX row: `AV2ENC_VIDEO_RESIDUAL_TX_TYPE=2` (type 2 `DCT_ADST`, row ADST);
* FDDT column: `AV2ENC_VIDEO_RESIDUAL_TX_TYPE=4` (type 4 `FLIPADST_DCT`, col FDST);
* FDDT row: `AV2ENC_VIDEO_RESIDUAL_TX_TYPE=5` (type 5 `DCT_FLIPADST`, row FDST);
* `ddt_splice.cpp` asserts exactly one differing bit, inside the
  sequence-header OBU, and writes the spliced stream;
* the spliced stream must equal the flag-1 stream byte-for-byte;
* dav2d must decode the flag-1 stream to the encoder's DDT reconstruction
  sample-for-sample;
* the flag-0 stream must *not* match (negative control).

The four passes report 62 / 45 / 62 / 45 DDT block directions.  Example:

```
DDT splice: bit=504 seq_payload=[384,536) syntax_bit=120 reader=0->1
DDT splice: enable_inter_ddt=1, stream bytes=19057, payload bytes=19
DDT splice: PASS (spliced == DDT stream, one bit)
DDT_GATE blocks=62 directions=62 ddtx=62 fddt=0
DDT GATE PASS
```

Run `./src/encoder/m2/ddt/build_and_test.sh --gate`.
