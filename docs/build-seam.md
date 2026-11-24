# The primitives build seam

`src/primitives/kernels.h` is the single seam between the encoder's host pixel
loops and an ISA-specific implementation.  Migrating a call site is a **compile
-time** contract, not just a code change:

1. **Product build.**  `av2enc_core` defines `AV2ENC_PRIMITIVES` (PRIVATE) and
   compiles `src/primitives/kernels.cpp`, so the call site takes
   `primitives::kernels()`.
2. **Component build.**  Every `build_and_test.sh` / `build_test.sh` gate in the
   tree compiles one or a few translation units directly with the host compiler
   and **does not** compile or link `kernels.cpp`.  Those TUs must therefore
   select an inlined `#else` fallback, which must be a transcription of the
   kernel (bit-identical by construction).

A migrated TU that only satisfies (1) is *unsafe*: it builds in the product and
fails to link in its own component gate.  `tools/refactor/build_seam_check.sh`
checks both configurations for every migrated TU and is the authoritative test
for this document.

## Audit

Collected with `grep -rn 'primitives::kernels()' src/` on `d5aeafe` plus the
`refine_sad` site added by this branch.  One row per call site.

Legend:
- **Guarded** — the call is inside `#if defined(AV2ENC_PRIMITIVES)` with an
  `#else` fallback.
- **Fallback** — compiling the TU with `-c` and *without* the define succeeds
  and leaves no reference to `av2enc::primitives::kernels()`.
- **Gate** — the component gate that compiles this TU without the define and
  without `kernels.cpp`.

| # | Site | Kernel | Guarded | Fallback | Gate | Status |
|---|------|--------|---------|----------|------|--------|
| 1 | `src/codec/inter/me/me.cpp:194` (`me_search`) | `sad`/`satd` table | yes (`:193`) | yes | `src/codec/inter/me/build_and_test.sh` | **safe** |
| 2 | `src/codec/inter/me/me.cpp:226` (`me_cost_block`) | `sad`/`satd` table | yes (`:225`) | yes | same | **safe** |
| 3 | `src/codec/inter/mc/mc.cpp:527` (`sad_block`) | `sad` | yes (`:522`) | yes | `src/codec/inter/mc/build_and_test.sh` | **safe** |
| 4 | `src/codec/rate/rdo/rdo.cpp:194` (`sad_u8`) | `sad` | yes (`:191`) | yes | `src/codec/rate/rdo/build_and_test.sh` | **safe** |
| 5 | `src/pipeline/video/video_frame.cpp:2164` (`block_variance_cost_at`) | `variance` | yes (`:2154`) | yes (with the product `AV2ENC_VIDEO_*` defines; see note) | `src/pipeline/video/build_and_test.sh` | **safe** |
| 6 | `src/codec/inter/temporal/temporal_predict.h:207` (`refine_sad`) | `sad` (doubled strides) | yes (`:200`) | yes (via `refinemv.cpp`) | `src/codec/inter/temporal/build_and_test.sh` | **safe** |
| 7 | `src/codec/intra/modes/intra_modes.cpp:347` | `edge_filter` | **no** | **no** | `src/codec/intra/modes/build_test.sh` | **unsafe — rf4-gaps** |
| 8 | `src/codec/intra/modes/intra_modes.cpp:689` | `sad` | **no** | **no** | same | **unsafe — rf4-gaps** |
| 9 | `src/codec/intra/dip/dip.cpp:389` | `sad` | **no** | **no** | `src/codec/intra/dip/build_and_test.sh` | **unsafe — rf4-gaps** |
| 10 | `src/codec/intra/ibc/core/ibc.cpp:230` (block copy) | `block_copy` | **no** | **no** | `src/codec/intra/ibc/core/build_and_test.sh` | **unsafe — rf4-gaps** |
| 11 | `src/codec/intra/ibc/core/ibc.cpp:253` (chroma copy) | `block_copy` | **no** | **no** | same | **unsafe — rf4-gaps** |
| 12 | `src/codec/intra/ibc/core/ibc.cpp:379` | `sad` | **no** | **no** | same | **unsafe — rf4-gaps** |
| 13 | `src/codec/intra/ibc/core/ibc.cpp:393` | `sad` | **no** | **no** | same | **unsafe — rf4-gaps** |
| 14 | `src/codec/transform/variants/tx_variants.cpp:260` (`forward_block` row pass) | `forward_tx_row` | yes (`row_kernels()`, `:239`) | yes | `src/codec/transform/variants/build_and_test.sh` | **safe** |
| 15 | `src/codec/transform/variants/tx_variants.cpp:363` (`inverse_block` row pass) | `inverse_tx_row` | yes (`row_kernels()`, `:239`) | yes | same | **safe** |

Summary: **15 call sites in 9 TUs; 7 sites in 3 TUs are unsafe.**  All seven
unsafe sites are under `src/codec/intra/**` and are owned by the parallel
`rf4-gaps` branch; this branch does not edit them.  The other eight sites are
guarded and their fallbacks compile with no `kernels.cpp`.

Sites 14-15 were the NEON transform row passes.  They landed unguarded (every
gate that links `tx_variants.cpp` alone failed at link with an undefined
`primitives::kernels()`) and were fixed by routing both through `row_kernels()`,
which returns null pointers without `AV2ENC_PRIMITIVES` so the inlined loops
run.  `tx_variants.cpp` was also missing from `build_seam_check.sh`'s `TUS`
list, which is why the check did not catch them; it is now listed there.

What "unsafe" means concretely, on `d5aeafe`:

- `ibc.cpp` and `dip.cpp` compile (`-c`) but their objects reference
  `av2enc::primitives::kernels()`, so `ibc_test` / `dip_test` fail at link with
  an undefined symbol: neither gate compiles `kernels.cpp`.
- `intra_modes.cpp` has the same undefined reference *and* its gate
  (`src/codec/intra/modes/build_test.sh`) uses `-I../..`, which resolves to
  `src/codec` rather than the repository's `src` root, so that gate currently
  fails at the include before it reaches the link.  Both are pre-existing on
  `main` and outside this branch's file ownership.
- `src/pipeline/video/build_and_test.sh` also compiles `video_frame.cpp` +
  `intra_modes.cpp`/`ibc.cpp` without `kernels.cpp`, so it hits the same
  undefined reference.  Separately, that gate passes no `AV2ENC_VIDEO_*`
  defines and `video_frame.cpp` now uses `base_q_index()` etc. outside the
  matching guards, so the TU does not compile there at all — an unrelated
  pre-existing breakage, noted for the integrator.  `build_seam_check.sh` holds
  those switches on in both columns so the only variable is
  `AV2ENC_PRIMITIVES`; the fallback branch itself is clean.

`tools/refactor/build_seam_check.sh` reported 5/8 TUs green and the three intra
TUs red on the rf5 branch alone.  After `rf4-gaps` landed its guards it is
green — 9 TUs, both configurations, with the two transform row passes added —
and is registered in `tools/run_gates.sh` as `gate_build_seam`.

## Build-system intent

`AV2ENC_PRIMITIVES` is `PRIVATE` on `av2enc_core` (`CMakeLists.txt`), which is
the only target that compiles these TUs, and it is the only target that compiles
`src/primitives/kernels.cpp`.  No other target receives the define
transitively, so there is currently **no** target that gets the macro without
the archive.  Any future target that compiles one of these TUs must either add
`AV2ENC_PRIMITIVES` *and* link `kernels.cpp`, or leave the define off and take
the fallback.  The comment at the definition site states this; adding the
define without the source is a guaranteed link error.

## refine_sad

`refine_sad` (refine-MV search) sums even rows only.  That is the `sad` kernel
with both strides doubled and `ceil(h/2)` rows:

```cpp
rows = h > 0 ? h / 2 + (h & 1) : 0;   // the brief's literal h/2 is not enough
sad(p0, 2*p0_stride, p1, 2*p1_stride, w, rows);
```

Equivalence, for `h > 0`: the loop visits `r = 0, 2, 4, ... < h`, i.e.
`r in [0, ceil(h/2))`, and kernel row `i` at stride `2*s` addresses input row
`2i` at stride `s`.  Hence the row sets are identical.

The row count must be **ceil**, not the truncated `h/2` written in the brief:
for odd `h` the last row `h-1` is even and `h/2` drops it.  A sweep of 1904
`(w, h, stride)` cases (`w in 0..13`, `h in 0..33`, four strides) shows the
ceil form matches the original loop in every case, and the literal `h/2` form
differs in all 884 odd-`h` cases.  Zero and negative `h`, and `w <= 0`, return
0 in both forms.  Strides must be non-negative, as for every other kernel in
the layer; doubling a negative stride in `size_t` is not the same as indexing
with it.  The 8-bit refine blocks are far below the `uint64` -> `int64` limit.

The inlined loop is kept verbatim under `#else`.
