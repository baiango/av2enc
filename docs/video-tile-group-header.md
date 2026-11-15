# Per-frame tile-group header: alignment and the tile_info() duplication

`write_tile_group_header()` (`src/pipeline/video/video_headers.cpp`) emits, for
every frame:

```
tile_group_obu(sz) {            # §5.19
  is_first_tile_group  f(1)     # always 1: one group per frame
  frame_header(1)               # §5.18.2
  # tile_start_and_end_present_flag is absent (NumTiles == 1)
  byte_alignment()              # §5.19, zero bits to the next byte
}
```

This document measures what that costs on the nightly pin, separates the
irreducible part from the one removable field, and documents
`--video-tile-group-compact`, the opt-in that removes it.

Measurement host: Apple M4, CPU-only build (`build-t`), `--cpu-threads 8`.
Pin: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, QP110, `--nightly`,
24,885 B / MD5 `5e316100ce7fd709b4298ccfebfffccd`.

## 1. What the header costs

Exact split of the 24,885-byte pin (parsed from the IVF, cross-checked against
the `AV2ENC_TRACE_FRAME_OBU` trace):

| Layer | Bytes | Per frame |
|---|---:|---:|
| IVF (32-byte header + 60×12) | 752 | 12.5 |
| Temporal delimiter OBUs (2 B × 60) | 120 | 2.00 |
| Sequence header OBU (leb128 + header + 19 B payload) | 21 | 0.35 |
| Tile-group OBU framing (leb128 79 B + 1 header byte × 60) | 139 | 2.32 |
| **`tile_group_obu()` header** | **1,634** | **27.2** |
| Tile payloads (all 60 tiles) | 22,219 | 370.3 |
| **Total** | **24,885** | **414.75** |

So the per-frame tile-group header path is 1,634 + 139 = **1,773 B (7.1%)**
(tile-group header payload plus that frame's leb128 + OBU header byte), not a
uniform 31 B/frame: the header body ranges from **74 bits** (13 frames, all in
the first 32) to **449 bits** (frames 32-59: every later P frame codes more
loop-filter parameters), mean **214.2 bits**.

## 2. `byte_alignment()`: 220 bits, and they are not recoverable

Bit-level body size and alignment padding for the 60 frames
(`AV2ENC_TRACE_FRAME_OBU=1`):

| body bits % 8 | padding bits | frames |
|---:|---:|---:|
| 0 | 0 | 7 |
| 1 | 7 | 2 |
| 2 | 6 | 18 |
| 3 | 5 | 3 |
| 4 | 4 | 11 |
| 5 | 3 | 8 |
| 6 | 2 | 4 |
| 7 | 1 | 7 |

- frame-header body total: **12,852 bits** (1,606.5 B)
- alignment padding: **220 bits = 27.5 B**, mean **3.67 bits/frame**, max 7
- padded header total: **1,634 B**

### Why it cannot be removed

§5.19 `tile_group_obu()` (the pinned v1.0.0 text numbers it 5.19; OBU framing
semantics are §6.2) calls `byte_alignment()` unconditionally before
`tile_group_payload()`, and §5.2.4 defines it as `zero_bit f(1)` until the
position is byte-aligned. There is no sequence or frame flag that moves the tile
payload off the byte grid, and the tile payload is byte-aligned at its own end
too (`exit_symbol()`, §8.2.4; see `docs/tiles.md`), so the tile cannot absorb a
partial header byte either.

**The padding also costs zero bytes.** The OBU is a whole number of bytes, so
splicing the first tile bit immediately after the header's last bit yields
`ceil((body + 8·tile)/8) = ceil(body/8) + tile` bytes - the same size. The floor
for a frame is `ceil(body/8) + tile_bytes`, and the 220 bits are simply the
unused tail of the last byte the frame header needs anyway.

Decoder evidence (pinned `dav2d`, `--strict 1`): a build variant that skipped
`align_to_byte()` *and* spliced the tile payload bits directly after the header
bits (so the tile really starts mid-byte) produces the same 24,885 B but fails
all 60 pictures - exit 1, `Error parsing frame header` × 59, `No data decoded`,
no output Y4M. In that variant the padding bits are tile data, and the pinned
decoder rejects the stream.

Reproduce the measurement:

```sh
AV2ENC_TRACE_FRAME_OBU=1 av2enc testdata/nasa_480x240_30fps_60f.y4m \
    -o /tmp/pin.ivf --video --frames 60 --qp 110 --cpu-threads 8 --nightly 2>&1 |
    grep tile_group_header          # body_bits= / pad_bits= per frame
```

## 3. The other candidates in this path, and why they stay

| Candidate | Bits/frame | Verdict |
|---|---:|---|
| `is_first_tile_group` | 1 | **Required.** §5.19 codes it unconditionally for every tile group, and §6.18 requires `SeenFrameHeader != is_first_tile_group`. It is not implied by the OBU type: §5.2.1 `is_tile_group()` routes OBU_CLOSED_LOOP_KEY, OBU_REGULAR_TILE_GROUP and the rest through the same `tile_group_obu()` syntax. |
| `tile_start_and_end_present_flag` | 0 | Already absent (NumTiles == 1). |
| `tile_size_minus_1` | 0 | Already absent: with `tg_end == NumTiles-1` the last tile's size is the rest of the OBU (§5.20.1). No redundant size field exists in this framing. |
| `context_update_tile_id` | 0 | Already absent (only coded for multi-tile without `avg_cdf_type`). |
| leb128 OBU size | 1-3 B | Already minimal (`append_leb128`). Measured: 1 byte on 42 frames, 2 bytes on 17, 3 bytes on the keyframe (18,875-byte OBU). |
| `tile_params()` (`uniform_tile_spacing_flag` + 2 increment bits) | **3** | **Redundant, and removable - see §4.** |

The remaining body bits are real per-frame syntax: `cur_mfh_id`/`seq_header_id`
(2), `frame_is_inter`/`immediate_output_frame` (2), `order_hint` (6),
`primary_ref`/CDF/refresh/map (3-5), `tile_info()` (3, see below),
`base_q_idx` (8), segmentation/QM/delta-Q (3), deblocking (8),
CDEF/CCSO/LR parameters (0-375, content), `tx_mode_select`/`reduced_tx_set`
(3), inter flags (3-5). Filter parameters are the only large, varying part, and
they are per-frame data rather than waste.

## 4. The removable part: `tile_params()` is a stream constant

Every frame codes the same three bits:

```
1         uniform_tile_spacing_flag          # §5.18.7.3
0         increment_tile_cols_log2 = 0       # one tile
0         increment_tile_rows_log2 = 0
```

§5.4.2 lets the sequence header own that grid:

```
sequence_tile_config() {                     # §5.4.2
  seq_tile_info_present_flag f(1) = 1
  allow_tile_info_change     f(1) = 0
  tile_params(seqSbSize)     -> 1 0 0
}
```

and §5.18.7.2 then infers, with no bits at all,
`reuse_tile_info = 1` whenever the sequence grid is eligible for the frame
(`uniform_eligible(SeqTileRowsLog2, sbRows) && uniform_eligible(SeqTileColsLog2,
sbCols)`). Our sequence grid is one uniformly spaced tile, so
`SeqTileColsLog2 = SeqTileRowsLog2 = 0` and `uniform_eligible(0, n)` holds for
every `n >= 1`: **every frame and every SbSize reuses it**, including the §5.18.1
clamp where a 256-superblock sequence codes an intra frame at 128 (the derived
grid is still `{0, MiCols}` × `{0, MiRows}`). The non-uniform branch
(`SeqSbCols == sbCols`) never applies.

`--video-tile-group-compact` writes exactly that form. Measured on the pin:

| Stream | Tile-group header | Sequence header OBU | Total |
|---|---:|---:|---:|
| nightly pin | 1,634 B | 21 B | 24,885 B |
| `+ --video-tile-group-compact` | **1,611 B** (−23) | 22 B (+1) | **24,863 B** |

- body bits: 12,852 → 12,672 (**−3 bits × 60 = −180 bits**)
- alignment padding: 220 → 216 bits (the frames where the padding was ≥ 3 give
  the byte back; the rest wrap)
- realized: 23 B out of the frame headers, 1 B in for the sequence header
  (its 4 new bits cross a byte boundary) → **net −22 B, −0.088 kbps** at the
  60-frame/2 s operating point (99.540 → 99.452 kbps)
- MD5 `1e46f26e30c3fdbbb758bc7a2d539280`

### Decoder evidence

- `dav2d --strict 1`: exit 0, **60/60 pictures**, no strict-mode diagnostics.
- Decoded payload is **byte-identical** to the pin's decode
  (`6b134b1c42ee1fe655868a03cff69f04`, 10,368,000 B) and to the encoder
  reconstruction, i.e. only header bits moved; the tile payloads are unchanged.
  CVVDP/PSNR therefore cannot move and were not re-run.
- Geometry sweep (4 frames each, flag off vs on, both decoded and compared):
  100x70 (MI edge), 102x72 with `--video-sb256` (intra SB clamp), 640x362 with
  `--video-sb128` (partial bottom superblock row), plus 480x240 sb128/sb256 and
  64x64. All decode identically. With 4 frames the sequence-header byte is not
  paid back yet, so those streams are equal or +1 B; the lever needs the P chain.

### Cost model

The change trades a one-time +4 bits in the sequence header for −3 bits in every
frame. The break-even is `ceil((seq_body+4)/8) - ceil(seq_body/8)` bytes against
`3·N` bits of frame savings: it pays once the P chain is long enough (4 frames
can still round to +1 B), which is why a 1-frame stream is +1 B (64x64: 734 →
735 B) and the 60-frame pin is −22 B. This is a
header-only rate lever, not a quality lever - the sibling of the
`--video-key-qp-offset`/CDF moves in `docs/video-nightly.md`, but two orders of
magnitude smaller.

## 5. Scope

- Opt-in and default off: `--nightly` does not include it, so the three pins stay
  byte-identical (25,009 `a1d0dba1...`, 25,141 `3984677b...`, 24,885
  `5e316100...`).
- Single tile, plain key+P chain. `--tiles > 1x1` and the B/TIP/frame-type
  wrappers are rejected by name: the tiled sequence-header copy carries no
  sequence `tile_params()` and the wrappers build their own sequence header.
- The sequence header is shared by the whole stream, so the flag cannot be
  mixed within one sequence - it is one flag per encode, like `--video-tier`.
- `--still` ignores the flag (still headers are a different writer).

## 6. Reproduce

```sh
build-t/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tg.ivf \
    --video --frames 60 --qp 110 --cpu-threads 8 --nightly \
    --video-tile-group-compact --video-recon /tmp/tg-recon.y4m
tools/bin/dav2d-macos -q --strict 1 -i /tmp/tg.ivf -o /tmp/tg.y4m --threads 1
tools/gate_video_tile_group.sh --bin build-t/av2enc
```

`tools/gate_video_tile_group.sh` checks the flag-off pins, the per-frame
−3-bit trace, the 22-byte net saving, 60/60 strict decoding, decoded/recon
equality, the rejections and the geometry sweep.
