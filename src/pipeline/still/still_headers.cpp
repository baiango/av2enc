#include "pipeline/still/still_headers.h"

#include "codec/entropy/deltaq/still/deltaq.h"
#include "codec/intra/screen_content/scc_flags.h"
#include "core/bitstream/basic_headers.h"
#include "frame/metadata/content_interpretation/content_interpretation.h"

#include "codec/filters/gdf/gdf_syntax.h"
#include "codec/filters/loop_restoration/restoration_common.h"
#include "pipeline/grain/film_grain_syntax.h"
#include "pipeline/grain/grain_stream.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"

#include <cassert>

namespace av2enc::still {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    assert(leading_zeros < 32U);
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  assert(n >= 2U);
  assert(value < n);
  unsigned w = 0;
  while ((1U << w) < n) {
    ++w;
  }
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols(const StillConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const StillConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

// §5.18.7.11 lr_luma_use_{half,max,quarter}_size / chroma equivalent for a
// 64x64 superblock.  shift 0..3 selects size = 512 >> shift (luma).
void write_lr_shift(BitWriter& writer, int shift) {
  assert(shift >= 0 && shift <= 3);
  writer.put_bit(shift == 1);  // half
  if (shift == 1) return;
  writer.put_bit(shift == 0);  // max
  if (shift == 0) return;
  writer.put_bit(shift == 2);  // quarter (0 => eighth)
}

void write_tile_info(BitWriter& writer, const StillConfig& c) {
  const std::uint32_t sb_mi =
      static_cast<std::uint32_t>(superblock_size_px(c) / 4);
  const std::uint32_t sb_cols = (mi_cols(c) + sb_mi - 1U) / sb_mi;
  const std::uint32_t sb_rows = (mi_rows(c) + sb_mi - 1U) / sb_mi;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) {
    writer.put_bit(false);  // increment_tile_cols_log2 = 0
  }
  if (sb_rows > 1U) {
    writer.put_bit(false);  // increment_tile_rows_log2 = 0
  }
}

std::size_t write_sequence_header(BitWriter& writer, const StillConfig& c) {
  const std::size_t start = writer.bit_size();

  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(0U, 5U);                    // seq_profile_idc
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(c.seq_level_idx, 5U);       // seq_level_idx
  put_uvlc(writer, 0);                        // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                        // bit_depth_idc = 1 => 8-bit

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U,
                  height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag

  // §5.4.3 sequence_partition_config(): 64x64 SB, no SDP / extended parts by
  // default.  The two superblock flags are mutually exclusive: when
  // use_256x256_superblock is 1 the use_128x128_superblock bit is not present
  // at all, so the default (both 0) keeps the historical byte sequence.
  writer.put_bit(c.use_256x256_superblock);
  if (!c.use_256x256_superblock) {
    writer.put_bit(c.use_128x128_superblock);
  }
  writer.put_bit(c.enable_sdp);  // enable_sdp
  writer.put_bit(c.enable_ext_partitions);
  if (c.enable_ext_partitions) {
    writer.put_bit(c.enable_uneven_4way_partitions);
  }
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config().
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config().
  writer.put_bit(c.enable_dip);          // enable_dip
  writer.put_bit(c.enable_intra_edge_filter);
  writer.put_bit(c.enable_mrls);        // enable_mrls
  writer.put_bit(c.enable_cfl_intra);
  writer.put_bits(static_cast<std::uint32_t>(c.cfl_ds_filter_index), 2U);
  writer.put_bit(c.enable_mhccp);  // enable_mhccp
  writer.put_bit(c.enable_ibp);  // enable_ibp

  // §5.4.6 sequence_inter_config(): forced by single_picture_header_flag.
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  put_ns(writer, 3U, 0U); // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);  // enable_bawp

  // §5.4.8 sequence_transform_quant_entropy_config().
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(c.enable_idtx_intra);  // enable_idtx_intra
  writer.put_bit(c.enable_intra_ist);   // enable_intra_ist
  writer.put_bit(c.enable_inter_ist);   // enable_inter_ist
  writer.put_bit(c.enable_chroma_dctonly);
  writer.put_bit(c.reduced_tx_part_set);  // reduced_tx_part_set
  writer.put_bit(c.enable_cctx);  // enable_cctx
  writer.put_bit(c.enable_tcq);   // enable_tcq
  // §5.4.8: choose_tcq_per_frame is absent for a single-picture header, so
  // enable_tcq = 1 infers choose_tcq_per_frame = 0.  In that case
  // enable_parity_hiding is inferred 0 and its bit is absent; otherwise the
  // historical zero bit is written.
  if (!c.enable_tcq) {
    writer.put_bit(false);  // enable_parity_hiding
  }
  // §5.4.8 delta-Q fields.  The default SequenceDeltaQ writes exactly the
  // historical 8 bits (separate_uv_delta_q=0, equal_ac_dc_q=1,
  // base_uv_ac_delta_q=0, uv_ac_delta_q_enabled=0).
  const bool dq_ok = write_sequence_delta_q_fields(writer, c.dc_ac_delta_q, nullptr);
  assert(dq_ok);
  (void)dq_ok;

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(c.cdef.enable);  // enable_cdef
  writer.put_bit(c.gdf.enable);   // enable_gdf
  if (c.gdf.enable && c.gdf.sb_size == 64) {
    writer.put_bit(c.gdf.unit_matches_sb_size);  // gdf_unit_matches_sb_size
  }
  // §5.4.10 sequence_filter_config() restoration row.  PC-Wiener stays
  // disabled (1); the luma/chroma Wiener-NS disables follow the plane flags.
  writer.put_bit(c.lr.enable);  // enable_restoration
  if (c.lr.enable) {
    const bool pc = c.lr.pc_wiener;
    writer.put_bit(!pc);  // lr_tools_disable[0][RESTORE_PC_WIENER]
    writer.put_bit(!(!pc && c.lr.plane[0].enabled));  // [0][WIENER_NONSEP]
    writer.put_bit(c.lr.uv_present);                  // lr_tools_uv_present
    if (c.lr.uv_present) {
      writer.put_bit(!c.lr.plane[1].enabled);  // [1][WIENER_NONSEP]
    }
  }
  // §5.4.10 sequence_filter_config() CCSO row.
  writer.put_bit(c.ccso.enable);  // enable_ccso
  if (c.ccso.enable) {
    writer.put_bit(true);  // ccso_unit_matches_sb_size (6/7 from SbSize)
  }
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(c.film_grain_present);  // film_grain_params_present

  return writer.bit_size() - start;
}

std::size_t write_frame_header_syntax(BitWriter& writer, const StillConfig& c) {
  const std::size_t start = writer.bit_size();
  assert(c.base_q_idx >= 1U);

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  // §5.18.3.3 screen_content_params() + §5.18.3.4 intrabc_params().  The
  // screen-content module owns the exact bits and rejects an inconsistent
  // combination; the single-picture sequence header infers both force values.
  {
    scc_flags::SequenceSccConfig seq_cfg;
    seq_cfg.single_picture_header_flag = true;
    const scc_flags::SequenceSccState seq_state =
        scc_flags::derive_sequence_scc_state(seq_cfg);
    scc_flags::FrameSccContext ctx;
    ctx.frame_is_intra = true;
    scc_flags::FrameSccConfig scc;
    scc.allow_screen_content_tools = c.allow_screen_content_tools;
    // force_integer_mv is only coded/present when screen tools are allowed.
    scc.force_integer_mv =
        c.allow_screen_content_tools ? c.force_integer_mv : false;
    scc.allow_intrabc = c.allow_intrabc;
    // The region bits are only present when allow_intrabc is set.
    scc.allow_global_intrabc = c.allow_intrabc && c.allow_global_intrabc;
    scc.allow_local_intrabc = c.allow_intrabc && c.allow_local_intrabc;
    scc.max_bvp_drl_bits_minus_1 = 0;
    std::string scc_error;
    scc_flags::FrameSccState frame_state;
    std::size_t scc_bits = 0;
    const bool scc_ok = scc_flags::write_frame_scc_params(
        writer, seq_state, ctx, scc, /*seq_max_bvp_drl_bits_minus_1=*/0,
        /*allow_frame_max_bvp_drl_bits=*/false, &frame_state, &scc_bits,
        scc_error);
    assert(scc_ok);
    assert(scc_error.empty());
    (void)scc_bits;
    assert(frame_state.allow_screen_content_tools ==
           c.allow_screen_content_tools);
    assert(!c.allow_intrabc || frame_state.IsIntraBC);
  }
  writer.put_bit(c.disable_cdf_update);
  write_tile_info(writer, c);

  // §5.18.6.1 quantization_params(): base_q_idx plus any frame DC/AC delta-Q
  // fields.  The default (disabled) state writes exactly the base_q_idx byte.
  const bool qp_ok = dcdeltaq::write_quantization_params(
      writer, c.base_q_idx, c.dc_ac_delta_q, c.frame_delta_q, nullptr);
  assert(qp_ok);
  (void)qp_ok;

  // §5.18.7.1 segmentation_params(), §5.18.7.2 setup_qm_params(), and
  // §5.18.7.8 delta_q_params().  Each writer emits exactly its historical
  // zero/disabled bit(s) when the corresponding tool is off.
  deltaq::FrameQpHeader qp_header;
  qp_header.primary_ref_is_none = true;  // OBU_CLOSED_LOOP_KEY
  rc::SegmentationParams segmentation;
  rc::DeltaQParams frame_delta_q_params;
  std::string qp_error;
  const bool seg_ok = make_delta_q_params(c.per_sb_delta_q, segmentation,
                                          frame_delta_q_params, &qp_error);
  assert(seg_ok);
  (void)seg_ok;
  const std::size_t seg_bits = deltaq::write_segmentation_params(
      writer, segmentation, qp_header, &qp_error);
  assert(seg_bits != 0);
  (void)seg_bits;
  const bool qm_ok = qm::write_setup_qm_params(
      writer, c.qm, /*num_planes=*/3, segmentation.enabled,
      &qp_error);
  assert(qm_ok);
  (void)qm_ok;
  const std::size_t dq_bits = deltaq::write_delta_q_params(
      writer, frame_delta_q_params, c.base_q_idx, &qp_error);
  assert(dq_bits != 0);
  (void)dq_bits;

  // deblocking_filter_params() (§5.18.5.2). Chroma flags are only coded when
  // at least one luma direction is enabled.
  writer.put_bit(c.apply_deblock_y_vert);    // apply_deblocking_filter[0]
  writer.put_bit(c.apply_deblock_y_horz);    // apply_deblocking_filter[1]
  const bool have_chroma_deblock =
      c.apply_deblock_y_vert || c.apply_deblock_y_horz;
  if (have_chroma_deblock) {
    writer.put_bit(c.apply_deblock_u);       // apply_deblocking_filter[2]
    writer.put_bit(c.apply_deblock_v);       // apply_deblocking_filter[3]
  }

  // §5.18.5.2: one df_delta_q_present f(1) per enabled plane, in plane order,
  // after all apply_deblocking_filter[] bits and before tx_mode_select. M1
  // sends DfDeltaQ = 0, so the bit is 0. A disabled plane has no bit.
  if (c.apply_deblock_y_vert) writer.put_bit(false);
  if (c.apply_deblock_y_horz) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_u) writer.put_bit(false);
  if (have_chroma_deblock && c.apply_deblock_v) writer.put_bit(false);

  // §5.18.7.9 gdf_params(), between deblocking_filter_params() and
  // cdef_params().  single_picture_header_flag infers gdf_frame_enable = 1.
  if (c.gdf.enable) {
    namespace gdf = av2enc::filters::gdf;
    gdf::FrameGdfContext ctx;
    ctx.enable_gdf = true;
    ctx.gdf_unit_matches_sb_size = c.gdf.unit_matches_sb_size;
    ctx.coded_lossless = false;
    ctx.single_picture_header_flag = true;
    ctx.sb_size = c.gdf.sb_size;
    ctx.width = c.max_frame_width;
    ctx.height = c.max_frame_height;
    ctx.mi_cols = static_cast<int>(mi_cols(c));
    ctx.mi_rows = static_cast<int>(mi_rows(c));
    ctx.disable_loopfilters_across_tiles = false;
    ctx.tile_cols = 1;
    ctx.tile_rows = 1;
    gdf::FrameGdfParams params;
    params.frame_enable = true;
    params.per_block = c.gdf.per_block;
    params.pic_qc_idx = c.gdf.pic_qc_idx;
    params.pic_scale_idx = c.gdf.pic_scale_idx;
    std::string gdf_error;
    const std::size_t bits =
        gdf::WriteFrameGdfParams(writer, ctx, &params, &gdf_error);
    assert(gdf_error.empty());
    assert(!params.per_block);
    assert(params.blk_size == c.gdf.blk_size);
    (void)bits;
  }

  // §5.18.7.10 cdef_params().  single_picture_header_flag implies
  // cdef_frame_enable = 1, so no bit is coded for it here.
  if (c.cdef.enable) {
    assert(c.cdef.damping >= 3 && c.cdef.damping <= 6);
    assert(c.cdef.num_strengths >= 1 && c.cdef.num_strengths <= 8);
    writer.put_bits(static_cast<std::uint32_t>(c.cdef.damping - 3), 2U);
    writer.put_bits(static_cast<std::uint32_t>(c.cdef.num_strengths - 1), 3U);
    writer.put_bit(c.cdef.on_skip_txfm);
    for (int i = 0; i < c.cdef.num_strengths; ++i) {
      const StillCdefStrength& s = c.cdef.strengths[i];
      assert(s.y_pri >= 0 && s.y_pri <= 15);
      assert(s.uv_pri >= 0 && s.uv_pri <= 15);
      assert(s.y_sec == 0 || s.y_sec == 1 || s.y_sec == 2 || s.y_sec == 4);
      assert(s.uv_sec == 0 || s.uv_sec == 1 || s.uv_sec == 2 || s.uv_sec == 4);
      writer.put_bit(s.y_pri == 0);
      if (s.y_pri != 0) {
        writer.put_bits(static_cast<std::uint32_t>(s.y_pri), 4U);
      }
      // The syntax value 3 decodes to 4; a decoded 4 is written as 3.
      writer.put_bits(static_cast<std::uint32_t>(s.y_sec == 4 ? 3 : s.y_sec),
                      2U);
      writer.put_bit(s.uv_pri == 0);
      if (s.uv_pri != 0) {
        writer.put_bits(static_cast<std::uint32_t>(s.uv_pri), 4U);
      }
      writer.put_bits(static_cast<std::uint32_t>(s.uv_sec == 4 ? 3 : s.uv_sec),
                      2U);
    }
  }

  // §5.18.7.11 lr_params().  M1 signals per-unit WIENER_NONSEP only: the tool
  // list is {RESTORE_NONE, RESTORE_WIENER_NONSEP} because PC-Wiener is
  // disabled in the sequence row and switchable needs a third tool.
  if (c.lr.enable && c.lr.pc_wiener) {
    // RESTORE_PC_WIENER codes no frame_filters_on bit and no coefficients.
    namespace pcw = av2enc::filters::pc_wiener_stream;
    pcw::LrHeaderContext ctx;
    ctx.enable_restoration = true;
    ctx.single_picture_header_flag = true;
    ctx.frame_is_intra = true;
    ctx.sb_size = 64;
    ctx.num_planes = 3;
    ctx.sub_x = 1;
    ctx.sub_y = 1;
    ctx.base_q_idx = c.base_q_idx;
    ctx.lr_tools_disable[0][pcw::kRestorePcWiener] = false;
    ctx.lr_tools_disable[0][pcw::kRestoreWienerNs] = true;
    ctx.lr_tools_disable[1][pcw::kRestorePcWiener] = true;
    // Plane 0 is RESTORE_PC_WIENER; chroma may independently be
    // RESTORE_WIENER_NONSEP, which composes because PC-Wiener is luma-only.
    const bool chroma_lr = c.lr.plane[1].enabled;
    assert(chroma_lr == c.lr.plane[2].enabled);
    ctx.lr_tools_disable[1][pcw::kRestoreWienerNs] = !chroma_lr;
    pcw::LrParams params;
    params.frame_type[0] = pcw::kRestorePcWiener;
    params.frame_type[1] =
        chroma_lr ? pcw::kRestoreWienerNs : pcw::kRestoreNone;
    params.frame_type[2] =
        chroma_lr ? pcw::kRestoreWienerNs : pcw::kRestoreNone;
    params.luma_shift = c.lr.plane[0].shift;
    params.chroma_shift = c.lr.plane[1].shift;
    std::string pc_error;
    const std::size_t bits =
        pcw::WriteLrParams(writer, &params, ctx, &pc_error);
    assert(pc_error.empty());
    assert(bits > 0);
    (void)bits;
  } else if (c.lr.enable) {
    const bool luma_lr = c.lr.plane[0].enabled;
    const bool chroma_lr = c.lr.plane[1].enabled;
    assert(chroma_lr == c.lr.plane[2].enabled);
    assert(luma_lr || chroma_lr);
    for (int plane = 0; plane < 3; ++plane) {
      const StillLrPlane& p = c.lr.plane[plane];
      // The tool list is {RESTORE_NONE} when this plane's Wiener NS is
      // disabled, so ns(1) codes no bits and RESTORE_NONE is inferred.
      if (!p.enabled) continue;
      put_ns(writer, 2U, 1U);  // FrameRestorationType = RESTORE_WIENER_NONSEP
      writer.put_bit(false);   // frame_filters_on = 0 (per-unit filters)
    }
    if (luma_lr) write_lr_shift(writer, c.lr.plane[0].shift);
    if (chroma_lr) write_lr_shift(writer, c.lr.plane[1].shift);
  }

  // §5.18.7.12 ccso_params().  single_picture_header_flag implies
  // ccso_frame_flag = 1, so no bit is coded for it.  M1 only signals the
  // band-offset-only form.
  if (c.ccso.enable) {
    for (int plane = 0; plane < 3; ++plane) {
      const StillCcsoPlane& p = c.ccso.plane[plane];
      writer.put_bit(p.enabled);
      if (!p.enabled) continue;
      assert(p.scale_idx >= 0 && p.scale_idx <= 3);
      writer.put_bit(p.bo_only);  // ccso_bo_only
      writer.put_bits(static_cast<std::uint32_t>(p.scale_idx), 2U);
      if (p.bo_only) {
        // bo_only => quant/ext/edge are inferred to 0.
        assert(p.quant_idx == 0 && p.ext_filter == 0 && p.edge_clf == 0);
      } else {
        assert(p.quant_idx >= 0 && p.quant_idx <= 3);
        assert(p.ext_filter >= 0 && p.ext_filter <= 6);
        assert(p.edge_clf == 0 || p.edge_clf == 1);
        writer.put_bits(static_cast<std::uint32_t>(p.quant_idx), 2U);
        writer.put_bits(static_cast<std::uint32_t>(p.ext_filter), 3U);
        const int quant = av2enc::restoration::kCcsoQuantSz[p.scale_idx]
                                                           [p.quant_idx];
        if (quant == 0) {
          assert(p.edge_clf == 0);
        } else {
          writer.put_bit(p.edge_clf != 0);
        }
      }
      const int band_bits = 2 + (p.bo_only ? 1 : 0);
      writer.put_bits(static_cast<std::uint32_t>(p.max_band_log2),
                      static_cast<unsigned>(band_bits));
      const int bands = 1 << p.max_band_log2;
      const int interval = p.bo_only ? 1 : (3 - p.edge_clf);
      std::size_t k = 0;
      for (int d0 = 0; d0 < interval; ++d0) {
        for (int d1 = 0; d1 < interval; ++d1) {
          for (int band = 0; band < bands; ++band) {
            // ccso_offset_idx tu(7): ones until the value, then a terminating
            // 0 unless the maximum (7) is reached.
            const int idx = p.offset_idx[k++];
            assert(idx >= 0 && idx <= 7);
            for (int i = 0; i < idx; ++i) writer.put_bit(true);
            if (idx < 7) writer.put_bit(false);
          }
        }
      }
    }
  }

  writer.put_bit(c.tx_mode_select);  // 0 = TX_MODE_LARGEST
  writer.put_bits(static_cast<std::uint32_t>(c.reduced_tx_set), 2U);

  // §5.18.10.1 film_grain_config(): last field of the frame header, after
  // global_motion_params().  A single-picture header infers apply_grain = 1.
  if (c.film_grain_present) {
    namespace grain = av2enc::filters::grain;
    grain::FilmGrainConfig config;
    config.apply_grain = true;
    config.fgm_id = c.film_grain_id;
    config.grain_seed = c.grain_seed;
    grain::FilmGrainConfigContext ctx;
    ctx.film_grain_params_present = true;
    ctx.single_picture_header_flag = true;
    ctx.immediate_output_frame = true;
    ctx.implicit_output_frame = false;
    std::string grain_error;
    const bool ok =
        grain::WriteFilmGrainConfig(writer, config, ctx, &grain_error);
    assert(ok);
    assert(grain_error.empty());
  }

  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

}  // namespace

std::size_t write_frame_header(BitWriter& writer, const StillConfig& config) {
  return write_frame_header_syntax(writer, config);
}

bool make_delta_q_params(const StillDeltaQConfig& config,
                         rc::SegmentationParams& segmentation,
                         rc::DeltaQParams& delta_q, std::string* error) {
  segmentation = rc::SegmentationParams{};
  delta_q = rc::DeltaQParams{};
  segmentation.enabled = config.segmentation_enabled;
  if (segmentation.enabled) {
    // DerivedPrimaryRefFrame == PRIMARY_REF_NONE on a CLK frame: update_map is
    // inferred 1 and temporal_update 0.
    segmentation.update_map = true;
    if (config.segmentation != nullptr) {
      // Per-segment Q planner: keep the caller's table (ALT_Q features) but
      // still force the key-frame map state.
      segmentation = *config.segmentation;
      segmentation.enabled = true;
      segmentation.update_map = true;
      segmentation.temporal_update = false;
    }
    // Otherwise every segment feature stays disabled and every segment uses
    // the frame's base_q_idx (the neutral table).
  }
  delta_q.present = config.delta_q_present;
  delta_q.res = config.delta_q_present ? config.delta_q_res : 0;
  if (!rc::segmentation_params_valid(segmentation, error)) return false;
  if (!rc::delta_q_params_valid(delta_q, error)) return false;
  return true;
}

std::vector<std::uint8_t> make_sequence_header_obu(const StillConfig& config) {
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::vector<std::uint8_t> assemble_still_picture(
    const StillConfig& config, std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, config);
  writer.align_to_byte();

  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  const std::vector<std::uint8_t> clk =
      av2enc::basic::make_obu(ObuHeader{ObuType::kClosedLoopKey}, payload);
  const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);

  std::vector<std::uint8_t> out;
  out.reserve(td.size() + seq.size() + clk.size() +
              config.film_grain_obu.size());
  out.insert(out.end(), td.begin(), td.end());
  out.insert(out.end(), seq.begin(), seq.end());
  if (!config.qm_obu.empty()) {
    // §7.3.8.9: the quantizer matrix OBU is available before the frame header
    // that references its level.
    out.insert(out.end(), config.qm_obu.begin(), config.qm_obu.end());
  }
  if (config.content_interpretation) {
    // §7.3.3: the content interpretation OBU is the first OBU of the coded
    // output frame unit, i.e. after the sequence header and before the frame.
    std::string ci_error;
    const av2enc::ci::ContentInterpretation params =
        av2enc::ci::default_content_interpretation();
    const std::vector<std::uint8_t> ci =
        av2enc::ci::make_content_interpretation_obu(params, ci_error);
    assert(ci_error.empty() && !ci.empty());
    out.insert(out.end(), ci.begin(), ci.end());
  }
  out.insert(out.end(), clk.begin(), clk.end());
  if (!config.film_grain_obu.empty()) {
    // §7.3.3/.4: the film grain OBU precedes the coded-frame OBU in the
    // temporal unit.
    std::vector<std::uint8_t> with_grain;
    std::string grain_error;
    if (!av2enc::filters::grain::InsertFilmGrainObu(
            out, config.film_grain_obu, &with_grain, &grain_error)) {
      return {};
    }
    out.swap(with_grain);
  }
  return out;
}

}  // namespace av2enc::still
