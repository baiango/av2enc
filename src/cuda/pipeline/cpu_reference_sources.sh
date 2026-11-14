#!/usr/bin/env bash
# Canonical CPU-reference source list for the CUDA component suites.
#
# The CPU encoder is the reference every CUDA component test compares against.
# Component scripts used to carry their own hand-maintained subset of the core;
# each integration wave added modules to `still_encoder` / `still_syntax` /
# `still_headers` and the subsets silently went stale, so the suite built with
# undefined references.  This file is the one place the CUDA suites take that
# list from.
#
# It mirrors the host `.cpp` translation units of the `av2enc_core` target in
# the root `CMakeLists.txt`, minus `src/cuda/**` (each suite adds the CUDA
# runtime pieces it needs itself, because the host-emulation suites must not
# link cudart).  Regenerate after `av2enc_core` changes:
#
#   cmake -S . -B build -G Ninja -DAV2ENC_ENABLE_CUDA=ON
#   grep -o 'CMakeFiles/av2enc_core\.dir/src/[A-Za-z0-9_/.-]*\.cpp\.o' \
#       build/build.ninja |
#     sed 's|CMakeFiles/av2enc_core.dir/||; s|\.o$||' |
#     grep -v '^src/cuda/' | sort -u
#
# Usage in a build script:
#
#   source "$ROOT/src/cuda/pipeline/cpu_reference_sources.sh"
#   av2enc_cpu_core_append CPU_SOURCES "" "${CPU_SOURCES[@]+"${CPU_SOURCES[@]}"}"
#
# The second argument is the path prefix to prepend ("" for repo-relative
# entries, "$ROOT/" for absolute ones); the rest are the entries the caller
# already lists, which are not appended twice.

AV2ENC_CPU_CORE_SOURCES=(
  # Host-only 10-bit arithmetic that `formats/.../hbd_still.cpp` calls.  It
  # lives under src/cuda/ because the still-GPU target also links it, but it
  # has no CUDA dependency (the CPU-only build adds the same file to
  # av2enc_core).
  src/cuda/pipeline/still/hbd/hbd_cpu.cpp
  # Host-side CDEF stage planner the M1 CPU encoder calls
  # (`still_encoder.cpp` -> cuda::still::still_cdef_plan).  Host-only: no CUDA
  # API, compiled by the host compiler in every suite.
  src/cuda/pipeline/still/cdef/still_cdef.cpp
  src/codec/entropy/av2/av2_entropy.cpp
  src/codec/entropy/avg_cdf/avg_cdf_bank.cpp
  src/codec/entropy/deltaq/still/deltaq.cpp
  src/codec/entropy/lossless/lossless.cpp
  src/codec/filters/cdef/cdef.cpp
  src/codec/filters/cdef/syntax/video_cdef.cpp
  src/codec/filters/deblock/deblock.cpp
  src/codec/filters/gdf/gdf_decision.cpp
  src/codec/filters/gdf/gdf_syntax.cpp
  src/codec/filters/loop_restoration/classes/wiener_class.cpp
  src/codec/filters/loop_restoration/classes/wiener_decision.cpp
  src/codec/filters/loop_restoration/classes/wiener_fit.cpp
  src/codec/filters/loop_restoration/classes/wiener_syntax.cpp
  src/codec/filters/loop_restoration/restoration.cpp
  src/codec/filters/pc_wiener/pc_wiener_stream.cpp
  src/codec/filters/stream/filters_stream.cpp
  src/codec/filters/tables/film_grain.cpp
  src/codec/filters/tables/gdf.cpp
  src/codec/filters/tables/luma_source.cpp
  src/codec/filters/tables/pc_wiener.cpp
  src/codec/transform/ddt/ddt_constraints.cpp
  src/codec/transform/ddt/ddt_syntax.cpp
  src/codec/transform/ddt/ddt_transform.cpp
  src/codec/inter/advanced/b_frame.cpp
  src/codec/inter/advanced/compound.cpp
  src/codec/inter/advanced/warp.cpp
  # §7.7/§7.8 implicit reference map and the §5.18.2 frame-map syntax.
  src/codec/inter/mv_stack/ref_map.cpp
  src/codec/inter/advanced/video/advanced_syntax.cpp
  src/codec/inter/bframes/bframes.cpp
  src/codec/inter/drl_reorder/drl_reorder.cpp
  src/codec/inter/drl_reorder/video_stack_adapter.cpp
  src/codec/inter/gme/gme.cpp
  src/codec/inter/gme/gme_syntax.cpp
  src/codec/inter/leading/leading.cpp
  src/codec/inter/longterm/longterm_headers.cpp
  src/codec/inter/longterm/longterm_pool.cpp
  src/codec/inter/longterm/longterm_stream.cpp
  src/codec/inter/mc/mc.cpp
  src/codec/inter/me/me.cpp
  src/codec/inter/merge/merge.cpp
  src/codec/inter/temporal/motion_field.cpp
  src/codec/inter/temporal/temporal_common.cpp
  src/codec/inter/temporal/tip.cpp
  src/codec/inter/temporal/temporal_syntax.cpp
  src/codec/inter/temporal/refinemv.cpp
  src/codec/inter/temporal/opfl.cpp
  src/codec/inter/tip/tip_signalling.cpp
  src/codec/inter/tip/tip_candidates.cpp
  src/codec/inter/tip/tip_build.cpp
  src/codec/inter/tip/tip_output.cpp
  src/codec/inter/multiref/compound_predict.cpp
  src/codec/inter/mv_precision/mv_precision.cpp
  src/codec/inter/random_access/random_access.cpp
  src/codec/inter/sef/sef.cpp
  src/codec/inter/tlayers/tlayers.cpp
  src/codec/inter/tlayers/tlayers_syntax.cpp
  src/codec/intra/cfl/cfl.cpp
  src/codec/intra/cfl/mhccp/mhccp.cpp
  src/codec/intra/cfl/mhccp/mhccp_syntax.cpp
  src/codec/intra/dip/dip.cpp
  src/codec/intra/dip/dip_syntax.cpp
  src/codec/intra/ibc/core/ibc.cpp
  src/codec/intra/ibc/stream/ibc_stack.cpp
  src/codec/intra/ibc/stream/ibc_syntax.cpp
  src/codec/intra/ibc/stream/still_ibc_stream.cpp
  src/codec/intra/ibp/ibp.cpp
  src/codec/intra/modes/intra_modes.cpp
  src/codec/intra/mrl/core/mrl.cpp
  src/codec/intra/mrl/stream/mrl_interaction.cpp
  src/codec/intra/mrl/stream/mrl_syntax.cpp
  src/codec/intra/palette/core/palette.cpp
  src/codec/intra/palette/core/scc_stream.cpp
  src/codec/intra/palette/stream/palette_stream.cpp
  src/codec/intra/screen_content/scc_flags.cpp
  src/codec/partition/ext/ext_syntax.cpp
  src/codec/partition/ext/partition_ext.cpp
  src/codec/partition/sdp/sdp.cpp
  src/codec/partition/sdp/sdp_syntax.cpp
  src/codec/partition/sdp/sdp_validate.cpp
  src/codec/partition/stream/partition_stream.cpp
  src/codec/partition/tree/partition.cpp
  src/codec/rate/rc/rc.cpp
  src/codec/rate/rdo/intra_decision.cpp
  src/codec/rate/rdo/rdo.cpp
  src/codec/rate/rdoq/rdoq.cpp
  src/codec/rate/rdoq/rdoq_av2_model.cpp
  src/codec/transform/delta_q/quant.cpp
  src/codec/transform/delta_q/search.cpp
  src/codec/transform/delta_q/syntax.cpp
  src/codec/transform/qm/qm.cpp
  src/codec/transform/secondary/tx_secondary.cpp
  src/codec/transform/tcq/cctx.cpp
  src/codec/transform/tcq/tcq.cpp
  src/codec/transform/tcq/tcq_av2_model.cpp
  src/codec/transform/inter_residual/inter_tx_type.cpp
  src/codec/transform/tx_partition/tx_partition.cpp
  src/codec/transform/tx_partition/tx_scan.cpp
  src/codec/transform/tx_type/tx_type.cpp
  src/codec/transform/variants/tx_select.cpp
  src/codec/transform/variants/tx_variants.cpp
  src/core/bitstream/basic_headers.cpp
  src/core/bitstream/bitwriter.cpp
  src/core/bitstream/obu.cpp
  src/core/io/ivf.cpp
  src/core/io/y4m.cpp
  src/core/telemetry/profiler.cpp
  src/formats/chroma/chroma_format.cpp
  src/formats/chroma/chroma_headers.cpp
  src/formats/chroma/chroma_residual.cpp
  src/formats/chroma/chroma_still.cpp
  src/formats/format.cpp
  src/formats/format_encode.cpp
  src/formats/format_y4m.cpp
  src/formats/high_bitdepth/hbd_depth.cpp
  src/formats/high_bitdepth/hbd_headers.cpp
  src/formats/high_bitdepth/hbd_reader.cpp
  src/formats/high_bitdepth/hbd_still.cpp
  src/frame/metadata/content_interpretation/content_interpretation.cpp
  src/frame/metadata/hdr/hdr_metadata.cpp
  src/frame/metadata/types/metadata_types.cpp
  src/frame/mfh/mfh.cpp
  src/pipeline/dispatch/planner.cpp
  src/pipeline/grain/film_grain_syntax.cpp
  src/pipeline/grain/grain_stream.cpp
  src/pipeline/planner/effort.cpp
  src/pipeline/still/basic_encoder.cpp
  src/pipeline/still/high_bitdepth/high_bitdepth_entropy.cpp
  src/pipeline/still/high_bitdepth/high_bitdepth_headers.cpp
  src/pipeline/still/high_bitdepth/high_bitdepth_still.cpp
  src/pipeline/still/still_encoder.cpp
  src/pipeline/still/still_headers.cpp
  src/pipeline/still/still_syntax.cpp
  src/pipeline/still/stream/still_stream_bridge.cpp
  src/pipeline/svc/svc_config.cpp
  src/pipeline/svc/svc_framing.cpp
  src/pipeline/svc/svc_pipeline.cpp
  src/pipeline/svc/svc_stream.cpp
  src/pipeline/svc/ops_full/lcr_ops_full.cpp
  src/pipeline/tiles/stream/tiles_stream.cpp
  src/pipeline/tiles/svc_layers.cpp
  src/pipeline/tiles/tile_group.cpp
  src/pipeline/tiles/tile_layout.cpp
  src/pipeline/video/post/bit_depth_checklist.cpp
  src/pipeline/video/post/loop_filters.cpp
  src/pipeline/video/post/video_loop_filters.cpp
  src/pipeline/video/post/rc_adapter.cpp
  src/pipeline/video/post/tiled_frame.cpp
  src/pipeline/video/hbd/video_hbd.cpp
  src/pipeline/video/video_chroma_search.cpp
  src/codec/inter/advanced/bawp.cpp
  src/codec/inter/morph/morph_syntax.cpp
  src/pipeline/video/video_frame.cpp
  src/pipeline/video/video_frames.cpp
  src/pipeline/video/video_gme.cpp
  src/pipeline/video/video_headers.cpp
  src/pipeline/video/video_lcr_ops.cpp
  src/pipeline/video/video_syntax.cpp
  src/pipeline/video/video_warp.cpp
  src/pipeline/video/video_tx_search.cpp)

# Append to the array named by $1 every core source that is not already there.
#   $2 = path prefix ("" or "$ROOT/")
#   $3.. = entries the caller already lists
av2enc_cpu_core_append() {
  local target="$1"
  local prefix="$2"
  shift 2
  local seen=" " entry source
  for entry in "$@"; do
    seen+="${entry#"$prefix"} "
  done
  for source in "${AV2ENC_CPU_CORE_SOURCES[@]}"; do
    case "$seen" in
      *" $source "*) continue ;;
    esac
    eval "$target+=(\"$prefix\$source\")"
  done
}
