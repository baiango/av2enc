# formats-hbd bug scout (10-bit / high bit depth)

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-FORMATS-HBD-01 | med | confirmed | `--still-sdp` / `--still-sub8` accepted and silently ignored on the 10-bit & chroma format paths | `av2enc y64_420.y4m -o a.ivf --still --bit-depth 10 --qp 90 --still-sdp` | same IVF MD5 as without the flag; the 8-bit `--still` path changes MD5 for both flags | `src/main.cpp:3220-3221, 3310-3324` | new |
| SCOUT-FORMATS-HBD-02 | med | confirmed | strict Y4M reader accepts an oversized frame body: a 4:4:4/10-bit body tagged `C420p10` is silently encoded as 4:2:0 | `python3 -c "d=open('y64_444.y4m','rb').read().replace(b'C444p10',b'C420p10',1);open('mm.y4m','wb').write(d)"; av2enc mm.y4m -o mm.ivf --still --bit-depth 10 --qp 90` | encode succeeds (`64x64 10-bit, 4:2:0`) instead of rejecting the length/chroma mismatch | `src/formats/format_y4m.cpp:273-282`, `src/formats/format_y4m.h:30` | new |
| SCOUT-FORMATS-HBD-03 | low | confirmed | `hbd_encode --q` silently truncates out-of-range/negative/malformed QP into 1..303 | `hbd_encode y64_420.y4m q.ivf --q 65537` | prints `q=1`; `--q -65535` also q=1; `--q 90abc` encodes as q=90 instead of a usage error | `src/formats/high_bitdepth/hbd_encode.cpp:47` | new |
| SCOUT-FORMATS-HBD-04 | low | confirmed | `hbd::parse_stream` cannot parse the component's own 4:2:2/4:4:4 streams and misparses 4:0:0 | harness: `encode_still()` then `parse_stream()` on `y64_422.y4m`/`y64_444.y4m`/`y64_400.y4m` | `sequence is not 4:2:0` for 422/444, `sequence: df par bits` for 400 | `src/formats/high_bitdepth/hbd_reader.cpp:154-188, 269-273` | new |
| SCOUT-FORMATS-HBD-05 | low | doc | docs still say 10-bit encoding is 4:2:0-only and limited to q<=90; the format gate also has no high-QP 4:4:4/4:0:0 case | see Details | code and goldens do support q=303 and all four chroma formats | `hbd_still.h:7-26,39,55`; `hbd_headers.h:15`; `README.md:44,66,114`; `docs/spec-audit/03-frame-header.md:118`; `format_gate.sh:220-233` | new |

## Details

### SCOUT-FORMATS-HBD-01 — no-effect Still tree flags accepted on the format path
Repro:
```sh
cd /tmp/hbdscout
BIN=/Users/arin/work/av2enc/build/av2enc
python3 /Users/arin/work/av2enc/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py 64 64 y64_420.y4m
$BIN y64_420.y4m -o base.ivf --still --bit-depth 10 --qp 90
$BIN y64_420.y4m -o sdp.ivf  --still --bit-depth 10 --qp 90 --still-sdp
$BIN y64_420.y4m -o sub8.ivf --still --bit-depth 10 --qp 90 --still-sub8
md5 -q base.ivf sdp.ivf sub8.ivf
```
Observed:
```text
1b5018610a86b8d79aba161e95770ae6
1b5018610a86b8d79aba161e95770ae6
1b5018610a86b8d79aba161e95770ae6
```
Both flags exit 0 and produce byte-identical output. On the ordinary 8-bit
`--still` path the same flags do change the stream
(`e46f910f…` for `--still-sdp`, `50c404e3…` for `--still-sub8`). The 8-bit
`--chroma 444` format path also accepts and ignores both flags.

Expected: usage error (exit 2), as for every other Still tool in this branch:
`the --bit-depth/--chroma format paths use their own minimal still writer`.
The multi-tile rejection table already lists `still_sub8`/`still_sdp`
(`src/main.cpp:3220-3221`), but the format-path table at
`src/main.cpp:3310-3324` omits them.

Likely cause: the format-path list of "tool flags that are not applied" was
copied from the tile list and missed these two fields.

### SCOUT-FORMATS-HBD-02 — frame-body length is never checked for trailing bytes
Repro:
```sh
cd /tmp/hbdscout
BIN=/Users/arin/work/av2enc/build/av2enc
python3 /Users/arin/work/av2enc/src/pipeline/still/high_bitdepth/gen_y4m_10bit.py 64 64 y64_444.y4m 444
python3 - <<'PY'
d = open('y64_444.y4m','rb').read().replace(b'C444p10', b'C420p10', 1)
open('mm.y4m','wb').write(d)
PY
$BIN mm.y4m -o mm.ivf --still --bit-depth 10 --qp 90
```
Observed: `av2enc: Format still-picture encode complete` with
`64x64 10-bit, 4:2:0`; the 4:4:4 body is truncated to the first 12,288 bytes
and encoded as 4:2:0. Appending arbitrary trailing bytes to a valid frame is
also accepted. Only `gcount() != raw.size()` (truncation) is checked
(`format_y4m.cpp:281-282`); there is no "no trailing bytes" check, although
`format_y4m.h:30` promises rejection of an "oversized frame". Any
depth/chroma tag that under-declares the real body (444→420, 422→420,
10-bit→8-bit) is therefore silently accepted.

Expected: reject with a precise frame-size error. Because the reader is
documented as a *strict* parser and the product routes 10-bit input through
it, this bypasses the "input must already match the requested format" check
in `format_encode.cpp`.

Likely cause: after the per-plane read the file position is not compared with
the end of the frame; the remaining bytes are simply left unread.

### SCOUT-FORMATS-HBD-03 — component encoder QP truncation
Repro (component tool built by `src/formats/high_bitdepth/build_and_test.sh`):
```sh
hbd_encode y64_420.y4m q.ivf --q 65537
hbd_encode y64_420.y4m q.ivf --q -65535
hbd_encode y64_420.y4m q.ivf --q 90abc
```
Observed:
```text
hbd: 64x64 C420p10, q=1 (step=40, qctx=0), ...
hbd: 64x64 C420p10, q=1 (step=40, qctx=0), ...
hbd: 64x64 C420p10, q=90 (step=528, qctx=0), ...
```
`std::atoi` has no range or full-token check, and the result is narrowed to
`std::uint16_t` before `qindex_encodable()` sees it, so 65537 and -65535 both
become 1, and `90abc` becomes 90. Expected: usage error, or at minimum a
rejection from `qindex_encodable`; the tool's own contract is
`1 <= base_q_idx <= 303`.

Likely cause: `hbd_encode.cpp:47` (`static_cast<std::uint16_t>(std::atoi(...))`)
instead of a checked range conversion (as `src/main.cpp:545` does for `--qp`).

### SCOUT-FORMATS-HBD-04 — reader rejects non-4:2:0 streams it just encoded
Build a tiny harness linked against `build/libav2enc_core.a` that calls
`fmt::read_y4m_picture()` → `hbd::encode_still()` → `hbd::parse_stream()` for
the four 64x64 generators (source kept at `/tmp/hbdscout/parse_test.cpp`,
compiled with `c++ -I/Users/arin/work/av2enc/src ... build/libav2enc_core.a`):
```text
y64_420.y4m: ok idc=0 chroma=0 depth_idc=0 64x64
y64_422.y4m: PARSE FAILED: sequence is not 4:2:0
y64_444.y4m: PARSE FAILED: sequence is not 4:2:0
y64_400.y4m: PARSE FAILED: sequence: df par bits
```
`hbd_reader.h` documents the reader as the round-trip/bit-golden checker for
the component, and the product matrix includes 4:2:2/4:4:4/4:0:0. The parser
hard-codes 4:2:0 (`hbd_reader.cpp:269-273`) and unconditionally reads the
chroma-only sequence fields (`cfl_ds_filter_index`, chroma dctonly, cctx,
separate_uv_delta_q, base_uv_ac_delta_q), so a monochrome header desyncs at
`df par bits`. `hbd_test.cpp` only exercises `parse_stream` on 4:2:0, which is
why the unit suite stays green.

Likely cause: no `chroma_format_idc`/`Monochrome` conditionals in
`parse_sequence()`, plus an explicit `!= 0` rejection.

### SCOUT-FORMATS-HBD-05 — stale constraints and gate-coverage claims
* `hbd_still.h:19,39,55` and `hbd_headers.h:15` still say encoding supports
  only `1..90` / slices 0, but `qindex_encodable()` accepts `1..303`,
  `encode_still()` passes `qctx_slice()`, and the product goldens include
  q=91/141/191/303.
* `README.md:66,114` repeat `q <= 90`, while the same file (qindex seam
  section) says the encode path supports `1..303` and `format_goldens.txt`
  pins q=91/141/191/303.
* `README.md:44,225` says the component test has 193 checks; the current
  `hbd_test` prints `PASS (275 checks)`.
* `docs/spec-audit/03-frame-header.md:118` says 10-bit `f(9)` is "not
  reachable from the CLI"; `--still --bit-depth 10 --qp 303` reaches it.
* `src/formats/format_gate.sh:229-233` gates high-QP only on 4:2:0 plus one
  4:2:2 q=191 case. There is no 4:4:4 or 4:0:0 case with base_q_idx 91..303,
  so a qctx/chroma regression there would pass `gate_format`. I verified the
  paths currently work (640x360 4:4:4 q=303 and 4:0:0 q=303 decoded with
  0-LSB recon), so this is a coverage gap, not a stream failure.

Expected: docs should describe the shipped `1..303` range and 4-format matrix,
and `format_gate.sh` should include 4:4:4/4:0:0 high-QP slices if the README
claim ("every format and high-QP slice") is to hold.
