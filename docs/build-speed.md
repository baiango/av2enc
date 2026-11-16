# Build speed

Measured on 2026-11-17, M4 (10 cores), Ninja, CPU-only Release. The numbers
exist so future changes can be judged against something real.

## The problem

The full build is not the pain; the **header-triggered rebuild** is.

| Operation | Time | Work |
|---|---:|---|
| clean build | 38.2 s | 184 compile units, 214 s CPU |
| `touch bitwriter.h` (content edit) | **23.6 s** | 82 TUs |
| link `av2enc` | 0.1 s | irrelevant |

Where the compile time goes, from `clang -ftime-trace`:

| TU | Frontend (parse) | Backend (codegen) |
|---|---:|---:|
| `partition.cpp` (mid-size) | **0.92 s / 71%** | 0.37 s |
| `video_frame.cpp` (12.6 k lines) | 1.21 s / 25% | **3.64 s / 75%** |

So mid-size TUs are parse-bound (PCH helps), the one giant TU is codegen-bound
(splitting it helps), and `-O3` is a real cost that only the measurement build
needs.

## Presets

`CMakePresets.json` provides three configurations. The default (no preset) is
unchanged; presets are opt-in.

| Preset | Flags | Clean | Fresh-dir rebuild | Header edit (82 TUs) |
|---|---|---:|---:|---:|
| `release` | `-O3`, strict warnings, **hardening**, no PCH, no cache | 38.2 s | 38.2 s | 23.6 s |
| `dev` | `-O1`, no warnings, **PCH** | **18–20 s** | ~20 s | **13.5 s** |
| `dev-cache` | `-O1`, no warnings, **ccache** | 44 s | **1.15 s** | 22.6 s |

The `release` preset enables `AV2ENC_HARDEN` (`-fstack-protector-strong`,
`_FORTIFY_SOURCE=3`, `-fPIE`; `-pie -Wl,-z,relro,-z,now` on ELF CPU links).
The dev presets instead enable libc++ hardening
(`_LIBCPP_HARDENING_MODE=fast`, `_GLIBCXX_ASSERTIONS`), which turns bounds
violations into aborts during development. Neither changes code generation
semantics: every pin is byte-identical under both.

```sh
# Editing: fastest cold build and header edits.
cmake --preset dev && cmake --build --preset dev -j10

# One worktree per agent / frequent branch switches: cold is slower, but a
# rebuild of the same tree is ~1 s.
cmake --preset dev-cache && cmake --build --preset dev-cache -j10

# Anything whose bytes or timings are quoted: gates, KPI, fps.
cmake --preset release && cmake --build --preset release -j10
tools/run_gates.sh --bin build-release/av2enc
```

## Bitstreams are identical under `-O1`

Verified, not assumed: `-O1` with `-ffp-contract=off` produces the same streams
as `-O3` on every pin — default `25,009 a1d0dba1`, quality `25,141 3984677b`,
nightly `23,690 1b99127a`, and the 720p recipe `36,572 B c4ba2813…`, plus the
Stage-0 nightly dump `d96b3a53…`.

That means a dev build is safe for **correctness** work (gates, RD experiments
that compare streams). It is **not** safe for quoting encode time: `-O1` runs
slower, so fps and speedup numbers must come from `release`.

## PCH and ccache are alternatives, not a stack

Measured: with PCH enabled, **342 of 368** compile calls are uncacheable (a
PCH-including compile bypasses ccache). Combining them gives neither benefit —
that is why `dev` has no ccache and `dev-cache` has no PCH.

## Why not the obvious options

- **Unity build** — fewer TUs would cut duplicated parsing, but it destroys
  incremental builds, collides anonymous namespaces across 184 files, and can
  raise `-O3` backend time. PCH gets the same front-end saving safely.
- **LTO** — already off; it multiplies the link step, which is currently free.
- **`-O0`** — saves backend time but makes every encode 2–4× slower, which
  negates the NEON work.
- **PCH containing project headers** — would bake `AV2ENC_*` macro state into
  the wrong configuration (`av2enc_core` defines `AV2ENC_PRIMITIVES`, the
  component gates do not), so `src/pch.h` is std-only by design.

## Next lever

Splitting `video_frame.cpp` (Phase B) turns its 8.3 s TU into ~3 s TU(s) and
shortens the 82-TU header rebuilds that involve it. It is the only remaining
structural win; everything else here is already at the measurement floor.

## Full LTO: measured, rejected

`AV2ENC_LTO=ON` (CPU-only) was built and timed on the M4 with PCH and hardening:

| Workload | no LTO | LTO | Change |
|---|---:|---:|---:|
| 240p QP106, 1 thread | 6.28 s | 6.18 s | -1.6% |
| 240p QP106, 8 threads | 3.37 s | 3.55 s | **+5.3%** |
| 720p QP217, 8 threads | 27.68 s | 28.97 s | **+4.7%** |

Streams are byte-identical, so this is pure codegen. The build cost is large:
24-35 s becomes **85.6 s**, and the link step alone is 47 s for `av2enc`, 34 s
for `video_gate`, 18 s and 17 s for the two dump harnesses. Full LTO is
therefore off; `AV2ENC_LTO` remains as an experiment only.
