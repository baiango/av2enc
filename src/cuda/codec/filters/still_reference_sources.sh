#!/usr/bin/env bash
# CPU reference sources for the standalone M1 still-filter device tests.
#
# `still_encoder.cpp` and `still_syntax.cpp` pull in the whole integrated still
# tool set (IBC, palette/SCC, SDP, TCQ, DQ, QM, IST, MRL, DIP, IBP, CDEF,
# CCSO/LR and GDF).  A device component test links the same CPU reference, so
# the dependency list lives here once instead of drifting in every
# `build_and_test.sh`.
#
# Usage (from a component directory):
#   source "$ROOT/src/cuda/codec/filters/still_reference_sources.sh"
#   HOST_SOURCES=("src/pipeline/still/still_encoder.cpp"
#                 "${STILL_REFERENCE_SOURCES[@]}")
#
# Paths are relative to the repository root.  `still_encoder.cpp` is not in
# this list: the chroma harness includes it textually to reach private state.

STILL_REFERENCE_SOURCES=(
  "src/pipeline/still/still_syntax.cpp"
  "src/pipeline/still/still_headers.cpp"
  "src/core/bitstream/bitwriter.cpp"
  "src/core/bitstream/obu.cpp"
  "src/core/bitstream/basic_headers.cpp"
  "src/codec/partition/tree/partition.cpp"
  "src/codec/partition/ext/partition_ext.cpp"
  "src/codec/partition/ext/ext_syntax.cpp"
  "src/codec/partition/stream/partition_stream.cpp"
  "src/codec/partition/sdp/sdp.cpp"
  "src/codec/partition/sdp/sdp_validate.cpp"
  "src/codec/partition/sdp/sdp_syntax.cpp"
  "src/codec/intra/modes/intra_modes.cpp"
  "src/codec/intra/cfl/cfl.cpp"
  "src/codec/intra/cfl/mhccp/mhccp.cpp"
  "src/codec/intra/cfl/mhccp/mhccp_syntax.cpp"
  "src/codec/intra/dip/dip.cpp"
  "src/codec/intra/dip/dip_syntax.cpp"
  "src/codec/intra/ibp/ibp.cpp"
  "src/codec/intra/ibc/core/ibc.cpp"
  "src/codec/intra/ibc/stream/ibc_stack.cpp"
  "src/codec/intra/ibc/stream/ibc_syntax.cpp"
  "src/codec/intra/ibc/stream/still_ibc_stream.cpp"
  "src/codec/intra/mrl/core/mrl.cpp"
  "src/codec/intra/mrl/stream/mrl_syntax.cpp"
  "src/codec/intra/mrl/stream/mrl_interaction.cpp"
  "src/codec/intra/palette/core/palette.cpp"
  "src/codec/intra/palette/core/scc_stream.cpp"
  "src/codec/intra/palette/stream/palette_stream.cpp"
  "src/codec/intra/screen_content/scc_flags.cpp"
  "src/codec/intra/screen_content/scc_flags_bridge.cpp"
  "src/codec/rate/rdoq/rdoq.cpp"
  "src/codec/rate/rdoq/rdoq_av2_model.cpp"
  "src/codec/rate/rdo/rdo.cpp"
  "src/codec/rate/rdo/intra_decision.cpp"
  "src/codec/rate/rc/rc.cpp"
  "src/codec/transform/tx_type/tx_type.cpp"
  "src/codec/transform/tx_partition/tx_partition.cpp"
  "src/codec/transform/tx_partition/tx_scan.cpp"
  "src/codec/transform/variants/tx_variants.cpp"
  "src/codec/transform/variants/tx_select.cpp"
  "src/codec/transform/qm/qm.cpp"
  "src/codec/transform/secondary/tx_secondary.cpp"
  "src/codec/transform/delta_q/syntax.cpp"
  "src/codec/transform/delta_q/quant.cpp"
  "src/codec/transform/delta_q/search.cpp"
  "src/codec/transform/tcq/tcq.cpp"
  "src/codec/transform/tcq/tcq_av2_model.cpp"
  "src/codec/transform/tcq/cctx.cpp"
  "src/codec/entropy/deltaq/still/deltaq.cpp"
  "src/codec/entropy/av2/av2_entropy.cpp"
  "src/codec/filters/deblock/deblock.cpp"
  "src/cuda/pipeline/still/cdef/still_cdef.cpp"
  "src/codec/filters/cdef/cdef.cpp"
  "src/codec/filters/loop_restoration/restoration.cpp"
  "src/codec/filters/loop_restoration/classes/wiener_class.cpp"
  "src/codec/filters/loop_restoration/classes/wiener_fit.cpp"
  "src/codec/filters/loop_restoration/classes/wiener_syntax.cpp"
  "src/codec/filters/loop_restoration/classes/wiener_decision.cpp"
  "src/codec/filters/pc_wiener/pc_wiener_stream.cpp"
  "src/codec/filters/tables/pc_wiener.cpp"
  "src/codec/filters/tables/luma_source.cpp"
  "src/codec/filters/tables/gdf.cpp"
  "src/codec/filters/gdf/gdf_syntax.cpp"
  "src/codec/filters/gdf/gdf_decision.cpp"
  "src/core/io/y4m.cpp"
  "src/core/io/ivf.cpp"
  "src/pipeline/grain/film_grain_syntax.cpp"
  "src/pipeline/grain/grain_stream.cpp"
  "src/frame/metadata/content_interpretation/content_interpretation.cpp"
)
