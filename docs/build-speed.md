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
| `release` | `-O3`, strict warnings, hardening, **PCH** | **24–35 s** | 24–35 s | ~14 s |
| `release-timing` | same as `release`, `AV2ENC_HARDEN=OFF` | 24–35 s | 24–35 s | ~14 s |
| `dev` | `-O1`, no warnings, **PCH** | **~20 s** | ~20 s | **13.5 s** |
| `dev-cache` | `-O1`, no warnings, **ccache** | 44 s | **1.15 s** | 22.6 s |

`release` enables `AV2ENC_HARDEN`: `-fstack-protector-strong`,
`_FORTIFY_SOURCE=3`, `-fPIE`, bounds-checked standard containers
(`_LIBCPP_HARDENING_MODE=fast`, `_GLIBCXX_ASSERTIONS`), and
`-pie -Wl,-z,relro,-z,now` on ELF CPU links. `release-timing` drops only
`AV2ENC_HARDEN`; it exists so fps/throughput rows are not depressed by the
container checks, and it produces byte-identical streams. Neither changes code
generation semantics: every pin is byte-identical under both.

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

## Warning set

A clean build now emits **zero** warnings with
`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`. The route there matters:

| Group | Count | Treatment |
|---|---:|---|
| `-Wmissing-braces` in generated CDF tables | 1,864 | pushed/ignored/popped **inside those five data headers**; not disabled for code, and the files are marked do-not-edit |
| `-Wmissing-braces`, hand-written | 8 | fixed |
| Other real warnings | 23 | fixed (shadowing, narrowing, dead code, unused, dead sentinel, lambda capture) |
| `-Wsign-conversion` | **1,040** | **all fixed**, file by file with per-file pin and gate checks |

The sign-conversion count was initially reported as 385 from an incremental
build; a clean build showed 1,040. Worth remembering: incremental warning
counts only cover recompiled TUs.

`AV2ENC_WERROR` (on in the release preset) makes this a build failure. It is
applied for **Clang only**: the tree has been observed warning-free under clang,
and GCC's set has not yet been seen on a runner. Enable it for GCC once an
ubuntu job has been seen to pass with it.

## GCC

Tested with Homebrew GCC 16.1.0 (aarch64 macOS). The build needed real fixes,
not just casts:

- **NEON ACLE strictness**: `uint32x4_t` and `int32x4_t` are not implicitly
  convertible under GCC; `kernels_neon.cpp` declared the SAD accumulator as
  `int32x4_t` while only ever storing unsigned results. Fixed at the declaration.
- **GCC-only warnings**: a dead `int < 0` test on `uint16_t` mi coordinates, a
  set-but-unused array, `__int128` under `-Wpedantic` (now an `__extension__
  typedef`), and a `uint64 -> uint32` mask.
- **`std::sort` false positives** on fixed 8-byte palette arrays: GCC inlines
  introsort's `__first + 16` probe and reports an array-bounds overflow for a
  range that is provably `[0, n)` with `n <= 8`. The two tiny sorts are now a
  local insertion sort (and one no-op sort on a 0/1-element array was removed),
  which is clearer than suppressing `-Warray-bounds`.

Result: GCC builds with **zero warnings**, and its encoder output is
byte-identical to clang on the nightly and 720p pins (`1b99127a`, `c4ba2813`).

The ubuntu CI job already compiles with the system GCC (the workflow does not
set `CXX`), so it exercises this; `AV2ENC_WERROR` stays Clang-only until that
job has been observed passing with it, since the runner's GCC version differs.

## Further measurements (2026-11-17)

| Change | Clean build (median) | Verdict |
|---|---:|---|
| std-only PCH | 32.2 s | baseline after the split |
| PCH + `bitwriter.h`, `spec_math.h`, `av2_entropy.h` | **29.4 s** | kept; ~9% faster, pins identical |
| same, `-Wconversion`/`-Wshadow` off | 33.8 s | no gain; not worth a preset |
| `--target av2enc` only | 24.3–29.7 s | inside host noise; tests overlap the core compile |
| `-j12` / `-j16` instead of `-j10` | 25.4 / 27.2 s | within noise; no default change |

The three added headers are macro-independent (no `AV2ENC_*` branches), which is
the condition for PCH inclusion; the no-PCH sanitizer jobs remain the check that
no source file was relying on them.
