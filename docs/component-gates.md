# Component `build_and_test.sh` scripts: known rot

Status: **red, pre-existing, not caused by the refactor.** Recorded here so the
next person does not rediscover it by bisecting.

## What they are

31 `src/**/build_and_test.sh` scripts compile a hand-maintained `CORE` source
list into a standalone test binary, without CMake and without `kernels.cpp`.
They predate the product gate runner and overlap it.

## What the syntax split changed

`video_syntax.cpp` became four TUs plus a private header. The scripts now list:

```
  "$ROOT/src/pipeline/video/video_syntax.cpp"
  "$ROOT/src/pipeline/video/video_syntax_block.cpp"
  "$ROOT/src/pipeline/video/video_syntax_inter.cpp"
  "$ROOT/src/pipeline/video/video_syntax_coeff.cpp"
  "$ROOT/src/codec/transform/delta_q/syntax.cpp"
```

The last line was already missing before the split: `video_headers.cpp` calls
`dcdeltaq::write_sequence_delta_q_fields`, which lives there.

## What is still missing (why they stay red)

The scripts pass **no `AV2ENC_VIDEO_*` defines**, while the product target
defines `AV2ENC_VIDEO_FRAMES`, `AV2ENC_VIDEO_CDF_CHAIN`, `AV2ENC_VIDEO_DELTAQ`,
`AV2ENC_VIDEO_EXT_PARTITIONS`, `AV2ENC_VIDEO_FORMATS`, `AV2ENC_VIDEO_GME` and
`AV2ENC_PRIMITIVES`. Under the product define set, whole definitions are
compiled in; without it, declarations remain and definitions vanish, so the
link fails on symbols such as `video::syntax::detail::Writer::Writer(...)`.

`src/pipeline/video/build_and_test.sh` has a second, older breakage:
`video_frame.cpp` references `base_q_index()` and `temporal_plane` outside the
guards that the product defines enable, so it does not compile at all there.

## Options, in order of preference

1. **Delete the duplicates.** Any script whose coverage is already in
   `tools/run_gates.sh` is redundant; removing it is the honest fix and shrinks
   the surface that goes stale. This is the Phase D "gates-docs" item.
2. **Give them the product define set** plus `src/primitives/kernels.cpp` if
   `AV2ENC_PRIMITIVES` is included, and keep `kernels.cpp` in the list from then
   on. Then `tools/refactor/build_seam_check.sh` covers the fallback path and
   the scripts can define the macro.
3. **Leave them red** and stop treating them as gates — worst option, because a
   red script that nobody runs is indistinguishable from a passing one.

Do not "fix" one by adding sources only: that moves the failure to the define
gap and looks like progress.
