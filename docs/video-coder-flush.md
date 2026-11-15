# Tile termination: the §8.2.4 trailer

Every coded frame in the video path is one tile, so every frame pays one §8.2
range-coder termination. This note measures that trailer on the nightly pin,
derives what the specification actually requires, and describes
`--video-tile-flush-min`, the opt-in minimum form.

Measured on `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, QP110 `--nightly`
(24,885 B, MD5 `5e316100ce7fd709b4298ccfebfffccd`), Apple M4. Every number below
comes from the writer's own trace (`AV2ENC_TRACE_TILE_FLUSH=1`), not from an
estimate.

## What the writer produced before this change

`Encoder::exit_symbol()` (`src/codec/entropy/av2/av2_entropy.h`) chose the final
code value `C` in the last arithmetic interval and wrote

```text
prefix bits [0, R)     R = renormalisation bits the symbol stream asked for
marker 1 at bit R      §8.2.4 trailing bit
14 zeros after it
0..7 alignment zeros   payload rounded up to a whole byte
```

so a tile is `ceil((R + 15) / 8)` bytes and the trailer is exactly
`15 + pad` bits, `pad` being the alignment. Over the 60-frame pin:

| Quantity | Bits | Bytes |
|---|---:|---:|
| Coded content (`R`) | 176,633 | 22,079.1 |
| Marker `1` | 60 | 7.5 |
| Marker's 14 zeros | 840 | 105.0 |
| Byte-align pad (`pad`, histogram below) | 219 | 27.4 |
| **Tile payload** | **177,752** | **22,219** |

Per-frame pad distribution (`pad` in bits): 0→3 frames, 1→9, 2→9, 3→9, 4→8,
5→7, 6→7, 7→8. Mean 3.65 bits.

The trailer is 2.33 B per frame on average, i.e. **139.9 B = 0.56 % of the
stream** (0.63 % of the tile payload). For the small P tiles it is a large
share: frame 3's tile is 7 bytes, so 2.33 B of trailer is a third of it.

## What §8.2.4 requires

The decoder derives the trailing position from its own counters:

```text
trailing    = get_position() - Min(15, SymbolMaxBits + 15)
padding_end = get_position() + Max(0, SymbolMaxBits)
require SymbolMaxBits >= -14
require bit[trailing] == 1 and zeros up to padding_end
```

`SymbolMaxBits = 8*sz - 15 - (renormalisation bits the stream requests)`, and
it goes negative when the payload ends before the requested bits do (`reads are
clipped`; the missing bits become ones in `SymbolValue`, i.e. zeros in the code
point). Substituting the counters:

```text
trailing = R          for every payload of 16 bits or more
forced bits after it = 8*sz - R
```

Two consequences, both verified against the pinned decoder:

1. **The marker never moves.** A shorter tile does not relocate the trailing
   `1`; the decoder simply stops reading earlier. So the shortest conformant
   tile is the one whose payload ends just after that `1`:
   `sz = ceil((R + 1) / 8)` bytes, and `SymbolMaxBits = -(8*sz - R)`, which the
   specification permits down to `-14`.
2. **The fourteen zeros and the padding carry no information.** They exist only
   because the historical encoder wrote the full §8.2.4 pattern; the decoder
   never needs them.

That is 1-2 bytes per frame: the difference between `ceil((R+15)/8)` and
`ceil((R+1)/8)` is one byte, or two when `R` is small enough in its last byte.

The 1-byte case (`R <= 7`) is deliberately not used: the tile-group layer
requires `sz >= 2` (`kMinTileBytes`, "init_symbol() reads up to 15 bits"), so
`exit_minimal_bits()` floors the payload at 16 bits.

## The change

`Encoder::minimal_trailer` (off by default) makes `exit_symbol()` report the
smaller payload. The written bytes are unchanged -- the trimmed tile is a
*strict prefix* of the historical tile, the marker stays at bit `R`, and the
code value is the same -- so no decision, CDF or reconstruction moves. The
writer arms it from `--video-tile-flush-min` (CPU video writer only).

Measured effect (all streams decode identically, 60/60 pictures under
`dav2d --strict 1`, encoder recon equal to the decoded payload):

| Stream | Historical | Minimum | Saved | % |
|---|---:|---:|---:|---:|
| default `--qp 189` | 25,009 | 24,906 | 103 | 0.41 |
| `quality` QP190 | 25,141 | 25,031 | 110 | 0.44 |
| **nightly QP110** | **24,885** | **24,777** | **108** | **0.43** |

Per-frame savings on the nightly pin: 48 frames save 2 B, 12 save 1 B, none save
0. The remaining trailer after the change is

| Quantity | Bits | Bytes |
|---|---:|---:|
| Marker `1` | 60 | 7.5 |
| Byte alignment after the marker (0..7 bits) | 195 | 24.4 |
| **Trailer** | **255** | **31.9** |

i.e. 0.53 B per frame, and that is the floor: the `1` and the byte alignment
are both mandatory, and no legal shorter payload exists (the module test
asserts that one byte less never decodes).

The earlier hypothesis in the task was a 2-3 B flush overshoot worth
120-180 B. That is not what the coder does: 2.33 B per frame *is* the
historical trailer, and only 1.8 B of it was avoidable. The savings are 108 B,
not ~150 B, because the trailer zeros sit in bytes that also carry coded bits
for about a fifth of the tiles.

## Verification

* `src/codec/entropy/av2/build_and_test.sh`: **120,069 checks pass**, including
  the new `test_minimal_trailer()`. It pins a byte vector for both forms,
  checks the empty-stream floor, and sweeps 400 random CDF/symbol streams
  asserting (a) the shortened stream decodes to the same symbols and passes
  `exit_symbol()`, (b) `sz == max(2, ceil((R+1)/8))`, (c) it is a prefix of the
  historical stream, and (d) one byte less never decodes. A separate 40,000-case
  sweep during development agreed on all four properties with zero failures.
* `src/pipeline/video/gate_tile_flush.sh` (`gate_tile_flush` in the product
  gates): pins unchanged with the flag off, per-tile trailer accounting from
  the trace, `dav2d --strict 1` on the shortened 2-, 6- and 60-frame streams
  with identical decoded payloads and 60/60 pictures, encoder recon equality,
  and the named rejections (`--video` required, CPU-only, single tile, lossy
  only).
* `tools/gate_video_nightly.sh` and `tools/test.sh --build build-t` stay green;
  the pins in `docs/video-nightly.md` are untouched because the flag is off by
  default and is not part of the `--nightly` expansion.

## Trace

`AV2ENC_TRACE_TILE_FLUSH=1` prints one line per tile from
`src/pipeline/video/video_syntax.cpp`:

```text
tile_flush R=155 bits=160 bytes=20 nat=176 pad=6 min=20 save=2 low=3832 range=49000
```

| Field | Meaning |
|---|---|
| `R` | renormalisation bits; the §8.2.4 marker position |
| `bits` | payload bits actually written |
| `bytes` | payload bytes written |
| `nat` | bits the historical form writes (`ceil((R+15)/8)` bytes) |
| `pad` | historical trailer bits after the marker (14 + 0..7) |
| `min` | shortest conformant payload in bytes |
| `save` | bytes the minimum termination saves (`nat/8 - min`) |

`save` is reported whether or not the flag is set, so the trace can measure the
headroom of a stream that keeps the historical form.

## Where the per-frame bytes go

For completeness, the 24,885 B nightly stream is 22,219 B of tile payload plus
2,666 B of framing: 752 B IVF + sequence header, and ~1,914 B of frame-OBU
headers over 60 frames. The frame headers are not constant -- the first P frames
spend ~10 B while the drift-refresh frames (32-59) spend 40-57 B, more than
their tile in several cases -- so the *frame header*, not the coder trailer, is
the next byte-count lever on this stream.

## Caveats

* The flag changes the bitstream, so a stream produced with it needs a decoder
  that implements §8.2.4's `SymbolMaxBits >= -14` clipping. The pinned `dav2d`
  does (that is where the `-14` limit comes from).
* The saving is per tile. Multi-tile stills pay one trailer per tile, so the
  same flag would be worth ~1.8 B per tile there; it is deliberately wired to
  the CPU video writer only, and the flag rejects `--tiles`.
* The CUDA video writer keeps the historical termination as its parity
  reference; the flag rejects `--video-gpu`.
* Savings depend on how `R` lands modulo 8, which is content and QP dependent:
  expect ~0.4 % of the tile payload, not more.
