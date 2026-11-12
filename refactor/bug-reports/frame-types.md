# frame-types bug scout

Scope: frame types and ordering — long-term refs/OLK, leading pictures, SEF,
SWITCH/RAS, hidden pictures, reference-slot refresh/invalidation, RAP
boundaries. Read-only; `build/av2enc`, `build/video_gate`,
`tools/bin/dav2d-macos`; 360p only.

Fixtures: `$B=/Users/arin/work/av2enc/build/av2enc`, `$IN=/tmp/f8.y4m` —
640x360 C420jpeg, 8 frames (4-px horizontal pan derived from
`testdata/src_640x360.y4m`; `/tmp/f9.y4m` and `/tmp/f17.y4m` extend the pan to
9 and 17 frames). All commands below were run against these.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-FRAME-01 | high | confirmed | `--avg-cdf --adaptive-cdf` + any frame-type mode that rebuilds the sequence header produces an undecodable stream | `$B $IN -o /tmp/a.ivf --video --frames 6 --video-max-depth 3 --video-leading 3 --avg-cdf --adaptive-cdf; tools/validate_dav2d.sh /tmp/a.ivf` | plain `--video` decodes 8/8 and `--video-sef` 6/6 with the same CDF flags; long-term/OLK/leading/SWITCH/RAS emit 0 decoded frames with `Error parsing frame header` / `Error parsing OBU data` | `src/pipeline/video/video_frames.cpp:84,335,435`; `src/pipeline/video/video_frame.cpp:1234,1434`; `src/codec/inter/longterm/longterm_headers.cpp:344`; `src/codec/inter/leading/leading.cpp:189`; `src/codec/inter/random_access/random_access.cpp:357` | new |
| SCOUT-FRAME-02 | med | confirmed | `--video-olk` is advertised as an open random-access point but the OLK temporal unit cannot be decoded as one (§7.4.4/E.7.1) | see details: drop IVF units before the OLK, decode | drop TU0 → `Error parsing OBU data`, 0 frames; re-inject the unit-0 `OBU_SEQUENCE_HEADER` into the OLK TU → `Error parsing frame header`, 0 frames. Same for `build-longterm/longterm_open-ltr.ivf` and `longterm_open-no-ltr.ivf`. `--video-leading` OLK (slot 0 + repeated SEQ) decodes from the same truncation | `src/pipeline/video/video_frames.cpp:208-210,236-245`; `src/codec/inter/longterm/longterm_stream.cpp:84-93`; `docs/spec-audit/05-decode-ordering.md` R52/R59 | new (gap noted in spec-audit R52/R59) |
| SCOUT-FRAME-03 | med | doc | `--video-olk` unconditionally emits the pinned-dav2d OLK header (no `num_key_ref_frames`/`ref_long_term_id` although `long_term_frame_id_bits = 1`) | `$B $IN -o /tmp/o.ivf --video --frames 4 --video-max-depth 3 --video-olk` | source sets `kOmitPinnedDav2d` and passes an empty retained list; §5.18.2 makes the list present when the id width is non-zero, so only the pinned non-conforming decoder parses it. spec-audit V2 says the integration "must remove it" | `src/pipeline/video/video_frames.cpp:89,208-210`; `src/codec/inter/longterm/longterm_headers.cpp:406-412`; `docs/negative-results.md:69`; `docs/spec-audit/03-frame-header.md` V2 | new (documented as gate-only) |
| SCOUT-FRAME-04 | med | duplicate | `--rc crf/vbr` passes the parser with `--video-leading/--video-olk/--video-ras` and then exits 1 with an internal `frame_base_q_idx` error | `$B $IN -o /tmp/r.ivf --video --frames 6 --video-max-depth 3 --video-leading 3 --rc crf --crf 20` | `av2enc: error: frame_base_q_idx must be empty or have one entry per frame`, exit 1 | see `rate-control-cli.md` SCOUT-RC-03 | duplicate of SCOUT-RC-03 |

## Details

### SCOUT-FRAME-01 — adaptive-CDF mode is accepted but the rebuilt sequence headers say it is off
Repro:
```sh
B=/Users/arin/work/av2enc/build/av2enc
# controls: both decode
$B /tmp/f8.y4m -o /tmp/ok1.ivf --video --frames 8 --video-max-depth 3 --avg-cdf --adaptive-cdf
$B /tmp/f8.y4m -o /tmp/ok2.ivf --video --frames 6 --video-max-depth 3 --video-sef --avg-cdf --adaptive-cdf
tools/validate_dav2d.sh /tmp/ok1.ivf   # 8 frames
tools/validate_dav2d.sh /tmp/ok2.ivf   # 6 frames
# broken: every mode that re-wraps the tiles under its own sequence header
$B /tmp/f8.y4m -o /tmp/bad.ivf --video --frames 6 --video-max-depth 3 --video-leading 3 --avg-cdf --adaptive-cdf
tools/validate_dav2d.sh /tmp/bad.ivf
```
Observed: the leading (and long-term 6, `--video-olk` 4, `--video-switch` 2,
`--video-ras` 3, `--video-tlayers 8`, `--b-frames 1` on 3 frames) streams make
dav2d abort with `Error decoding frame: Invalid argument` / `Error parsing
frame header` / `Error parsing OBU data`; 0 decoded frames. `--avg-cdf` alone
is harmless but silently ignored (byte-identical payloads/headers to the run
without it).

Expected: decode like the plain path (the gated form `--avg-cdf --adaptive-cdf`
is documented to decode exactly, `docs/negative-results.md:78`), or reject the
combination at parse time with exit 2.

Likely cause: the plain M2 writer threads the opt-ins
(`video_frame.cpp:1864,1869`: `disable_cdf_update = !adaptive_cdf`,
`enable_avg_cdf = avg_cdf`), while the frame-type writers hard-code
`disable_cdf_update = true` (`video_frames.cpp:84,335,435`;
`video_frame.cpp:1234` for B frames and `:1434` for t-layers) and
`enable_avg_cdf = 0` (`longterm_headers.cpp:344`, `leading.cpp:189`,
`random_access.cpp:357`). The reused tiles were range-coded with CDF updates
enabled, so the decoder's symbol CDFs diverge on the first frame. `--video-sef`
works because it keeps the M2 sequence header instead of rebuilding one.

### SCOUT-FRAME-02 — the OLK random-access point is not decodable from the OLK
Repro:
```sh
B=/Users/arin/work/av2enc/build/av2enc
$B /tmp/f8.y4m -o /tmp/olk.ivf --video --frames 4 --video-max-depth 3 --video-olk
python3 - <<'PY'
import struct
def rd(p):
    d=open(p,'rb').read(); h=bytearray(d[:32]); o=32; f=[]
    while o+12<=len(d):
        n=struct.unpack_from('<I',d,o)[0]; f.append(d[o+12:o+12+n]); o+=12+n
    return h,f
def wr(p,h,f):
    h=bytearray(h); struct.pack_into('<I',h,24,len(f)); b=bytearray(h)
    for i,x in enumerate(f): b+=struct.pack('<IQ',len(x),i)+x
    open(p,'wb').write(b)
def obus(b):
    o=0; r=[]
    while o<len(b):
        v=s=0
        while True:
            x=b[o]; o+=1; v|=(x&0x7f)<<s; s+=7
            if not x&0x80: break
        r.append(bytes(b[o:o+v])); o+=v
    return r
h,f=rd('/tmp/olk.ivf')
wr('/tmp/olk_rap.ivf',h,f[1:])                 # remove everything before the OLK
o0=obus(f[0]); pre=bytes([len(o0[0])])+o0[0]+bytes([len(o0[1])])+o0[1]
wr('/tmp/olk_rap_seq.ivf',h,[pre+f[1][2:]]+f[2:])   # + TD/SEQ in-band
PY
tools/bin/dav2d-macos -i /tmp/olk_rap.ivf     -o /tmp/a.y4m --strict 1 --threads 1
tools/bin/dav2d-macos -i /tmp/olk_rap_seq.ivf -o /tmp/b.y4m --strict 1 --threads 1
```
Observed: the truncated stream fails with `Error parsing OBU data` (no
sequence header at the OLK TU); after injecting the original `OBU_SEQUENCE_HEADER`
it still fails at the OLK TU with `Error parsing frame header` /
`Error parsing OBU data` and writes no frames. The canonical component stream
`build-longterm/longterm_open-ltr.ivf` behaves the same (0 frames when the
pre-OLK part is removed and a SEQ is supplied), as does the
`long_term_frame_id_bits = 0` variant `longterm_open-no-ltr.ivf`.

Control: the leading-picture stream's OLK is at slot 0, repeats the sequence
header in its TU and is output implicitly; dropping TU0 of
`--video-leading 3` (`/tmp/ft_lead6.ivf`) decodes 2 frames (OLK + 1 regular).
So the truncation method is valid; the long-term OLK framing is the
differentiator.

Expected: E.7.1 requires a conformant bitstream to be decodable from any random
access point ("a bitstream produced ... by removing the part of the bitstream
preceding a random access point associated with an OBU_OPEN_LOOP_KEY ... shall
also be a conformant bitstream"), and §7.4.4 permits regular frames after the
OLK to use pre-OLK data only via OLK-refreshed slots or LTRs named in
`ref_long_term_id`.

Likely cause: `longterm_stream.cpp:84-93` writes the sequence header only in
unit 0 (the OLK TU has none), and `video_frames.cpp:208-210` creates the OLK at
slot 1 with an **empty** retained-LTR list while P3 (`units[2]`) references
slot 0, the pre-OLK CLK. With `kOmitPinnedDav2d` the §7.4.4 check that would
catch this (`validate_open_random_access` with `required = {0}`) is never
called. Related documentation rows: spec-audit R52/R59, 03 V2,
`docs/negative-results.md:69`.

### SCOUT-FRAME-03 — product OLK uses the pinned-decoder syntax workaround
Repro:
```sh
$B /tmp/f8.y4m -o /tmp/olk.ivf --video --frames 4 --video-max-depth 3 --video-olk
```
Observed: `video_frames.cpp:89` forces
`LongTermListSyntax::kOmitPinnedDav2d` for every product OLK. The sequence
header signals `long_term_frame_id_bits = 1`, for which §5.18.2 codes
`num_key_ref_frames f(3)` + `ref_long_term_id[]` in the OLK header; the writer
omits both (`longterm_headers.cpp:406-412`). A conforming decoder therefore
desyncs on the OLK header (the pinned dav2d tolerates only this layout;
`docs/negative-results.md:69`). `docs/spec-audit/03-frame-header.md` V2 says
"Any future integration must remove it", but commit 13432d7 wired the mode
into the CLI with the workaround still on.

Expected: emit the §5.18.2 list (pass the retained id 0 to
`plan_open_loop_key`) or document that `--video-olk` is pinned-decoder-only and
reject it behind a flag.

### SCOUT-FRAME-04 — duplicate
`--video-leading/--video-olk/--video-ras` + `--rc crf|vbr` is accepted by the
parser and dies during the encode with `frame_base_q_idx must be empty or have
one entry per frame` (VBR prefixes `--rc vbr pass 1:`). Already reported as
SCOUT-RC-03 in `rate-control-cli.md`; not re-reported here.

## Coverage notes (no defect found)
Sequential decode/output order was correct and recon-exact (encoder `--video-recon`
vs dav2d, sample-exact) for: `--video-long-term` 1/2/3/6/16 frames;
`--video-olk` 4; `--video-leading N` for N=1..7 over 3..9-frame inputs (documented
`2 + regular` outputs); `--video-sef` 2..24 frames with `--video-refs` 1..3
(hidden last picture + SEF unit, `N` outputs, order 0..N-1); `--video-switch` 2;
`--video-ras` 3; `--video-tlayers` 4/8 plus `--video-tlayer-drop` 0..2;
`--b-frames` 1/3/6. Boundary checks (`--video-leading` slot/order-hint limits,
OLK exactly 4 inputs, SWITCH 2 / RAS 3, SEF >=2, long-term order-hint overflow
at 17 frames) reject as documented.
