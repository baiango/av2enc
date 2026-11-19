// AV2 M2 P-frame tile syntax writer: entry point, per-frame/tile assembly and
// state setup.  The syntax helpers live in the sibling video_syntax_*.cpp TUs;
// see video_syntax_writer.h for the class and video_syntax.h for the contract.
#include "pipeline/video/video_syntax_writer.h"

#include <fstream>
#include <mutex>

namespace av2enc::video::syntax {
namespace detail {
namespace {
// Parallel tile writers append to the same diagnostic file; one lock keeps
// every JSON object on its own line.  Uncontended when the diagnostics are off.
std::mutex g_census_mutex;
}  // namespace

bool Writer::run(std::size_t* tile_bytes, std::string& error) {
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (!dqp_init_error_.empty()) {
    error = dqp_init_error_;
    return false;
  }
#endif
  if (blocks_.empty()) {
    error = "syntax stream is empty";
    return false;
  }
  const int sb_cols = plan_.sb_cols();
  const int sb_rows = plan_.sb_rows();
  const int sb_mi = partition::sb_size_mi(geometry_);
  const bool cdef_active = filters_ != nullptr && filters_->cdef.enabled;
  if (cdef_active &&
      !filters_stream::cdef_tile_init(
          &cdef_state_, mi_cols_, mi_rows_, filters_->cdef.num_strengths,
          /*sb_width_log2=*/plan_.sb_size == 64
              ? 4
              : (plan_.sb_size == 128 ? 5 : 6),
          &error)) {
    return false;
  }
  if (cdef_active) {
    // §8.2.2: the tile's CDEF index CDFs come from the frame image, not
    // from the §9.3 defaults cdef_tile_init() just installed.
    const int strengths = filters_->cdef.num_strengths;
    std::memcpy(cdef_state_.index0, cdfs_.cdef_index0,
                sizeof(cdef_state_.index0));
    if (std::uint16_t* slot = cdef_minus1_slot(cdfs_, strengths)) {
      std::memcpy(cdef_state_.minus1, slot,
                  sizeof(std::uint16_t) *
                      static_cast<std::size_t>(strengths));
    }
  }
  const bool lr_active = filters_ != nullptr && filters_->lr.enabled;
  if (lr_active && filters_->lr.pc_wiener) {
    // §5.20.10.4 with FrameRestorationType[0] = RESTORE_PC_WIENER: the luma
    // units code only use_pc_wiener; chroma Wiener NS keeps the shared
    // filters_stream writer (its plane-0 state is RESTORE_NONE so it codes
    // no luma bits).
    namespace pcw = ::av2enc::filters::pc_wiener_stream;
    pcw::LrParams pc_params;
    pc_params.frame_type[0] = pcw::kRestorePcWiener;
    pc_params.frame_type[1] = pcw::kRestoreNone;
    pc_params.frame_type[2] = pcw::kRestoreNone;
    pc_params.luma_shift = filters_->lr.plane[0].shift;
    pc_params.chroma_shift = filters_->lr.chroma_shift;
    pc_ready_ = pcw::InitTileState(&pc_state_, pc_params, mi_cols_, mi_rows_,
                                   /*sub_x=*/1, /*sub_y=*/1, &error);
    if (!pc_ready_) return false;
    // §8.2.2: the tile LR CDFs come from the frame image, which §7.4.4/
    // §7.5 carries across frames under --cdf-continuity.
    pcw::TileStateSeedCdfs(&pc_state_, cdfs_.lr_cdfs);
    pc_state_.unit_type[0] = filters_->lr.plane[0].unit_type;
    const bool chroma_lr = filters_->lr.plane[1].enabled ||
                           filters_->lr.plane[2].enabled;
    if (chroma_lr) {
      filters_stream::LrParams params;
      params.frame_type[0] = filters_stream::kRestoreNone;
      for (int plane = 1; plane < 3; ++plane) {
        params.frame_type[plane] = filters_->lr.plane[plane].enabled
                                       ? filters_stream::kRestoreWienerNs
                                       : filters_stream::kRestoreNone;
        params.frame_filters_on[plane] = false;
      }
      params.frame_filters_on[0] = false;
      params.num_filter_classes = 1;
      params.num_filter_classes_idx = 0;
      params.luma_shift = filters_->lr.luma_shift;
      params.chroma_shift = filters_->lr.chroma_shift;
      if (!filters_stream::lr_state_init(&lr_state_, params, mi_cols_,
                                         mi_rows_, 1, 1, plan_.sb_size,
                                         &error)) {
        return false;
      }
      filters_stream::lr_state_seed_cdfs(&lr_state_, cdfs_.lr_cdfs);
      for (int plane = 1; plane < 3; ++plane) {
        const post::VideoLrPlane& src = filters_->lr.plane[plane];
        if (!src.enabled) continue;
        if (static_cast<int>(lr_state_.unit_type[plane].size()) !=
                src.unit_rows * src.unit_cols ||
            static_cast<int>(lr_state_.unit_coeffs[plane].size()) !=
                src.unit_rows * src.unit_cols *
                    static_cast<int>(
                        ::av2enc::restoration::kWienerNsCoeffStride)) {
          error = "video LR: unit grid does not match the plan";
          return false;
        }
        lr_state_.unit_type[plane] = src.unit_type;
        lr_state_.unit_coeffs[plane] = src.unit_coeffs;
      }
      lr_chroma_ready_ = true;
    }
  } else if (lr_active) {
    filters_stream::LrParams params;
    for (int plane = 0; plane < 3; ++plane) {
      params.frame_type[plane] =
          filters_->lr.plane[plane].enabled ? filters_stream::kRestoreWienerNs
                                            : filters_stream::kRestoreNone;
      params.frame_filters_on[plane] = false;
    }
    params.num_filter_classes = 1;
    params.num_filter_classes_idx = 0;
    params.luma_shift = filters_->lr.luma_shift;
    params.chroma_shift = filters_->lr.chroma_shift;
    if (!filters_stream::lr_state_init(&lr_state_, params, mi_cols_,
                                       mi_rows_, 1, 1, plan_.sb_size,
                                       &error)) {
      return false;
    }
    filters_stream::lr_state_seed_cdfs(&lr_state_, cdfs_.lr_cdfs);
    // Seed the per-unit decision grids from the encoder plan: read_lr() must
    // signal the same LrType/coefficients that were applied to the pixels.
    for (int plane = 0; plane < 3; ++plane) {
      const post::VideoLrPlane& src = filters_->lr.plane[plane];
      if (!src.enabled) continue;
      if (static_cast<int>(lr_state_.unit_type[plane].size()) !=
              lr_state_.unit_rows[plane] * lr_state_.unit_cols[plane] ||
          static_cast<int>(lr_state_.unit_coeffs[plane].size()) !=
              lr_state_.unit_rows[plane] * lr_state_.unit_cols[plane] *
                  static_cast<int>(
                      ::av2enc::restoration::kWienerNsCoeffStride)) {
        error = "video LR: unit grid does not match the tile";
        return false;
      }
      // The plan is frame-level; copy the units that overlap this tile into
      // the tile-local state, translated by the tile's unit origin.
      const int sub_x = plane == 0 ? 0 : 1;
      const int sub_y = plane == 0 ? 0 : 1;
      const int unit_size = lr_state_.unit_size[plane];
      const int row_off =
          (tile_mi_row_ * 4 >> sub_y) / std::max(1, unit_size);
      const int col_off =
          (tile_mi_col_ * 4 >> sub_x) / std::max(1, unit_size);
      const int coeff_stride =
          static_cast<int>(::av2enc::restoration::kWienerNsCoeffStride);
      for (int r = 0; r < lr_state_.unit_rows[plane]; ++r) {
        const int fr = r + row_off;
        if (fr < 0 || fr >= src.unit_rows) continue;
        for (int c = 0; c < lr_state_.unit_cols[plane]; ++c) {
          const int fc = c + col_off;
          if (fc < 0 || fc >= src.unit_cols) continue;
          const std::size_t dst =
              static_cast<std::size_t>(r) *
                  static_cast<std::size_t>(lr_state_.unit_cols[plane]) +
              static_cast<std::size_t>(c);
          const std::size_t from =
              static_cast<std::size_t>(fr) *
                  static_cast<std::size_t>(src.unit_cols) +
              static_cast<std::size_t>(fc);
          lr_state_.unit_type[plane][dst] = src.unit_type[from];
          std::copy_n(src.unit_coeffs.begin() +
                          static_cast<std::ptrdiff_t>(from * coeff_stride),
                      coeff_stride,
                      lr_state_.unit_coeffs[plane].begin() +
                          static_cast<std::ptrdiff_t>(dst * coeff_stride));
        }
      }
    }
    lr_chroma_ready_ = true;
  }
  const bool ccso_active = filters_ != nullptr && filters_->ccso.enabled;
  if (ccso_active) {
    filters_stream::CcsoParams params;
    params.frame_flag = true;
    params.ccso_luma_size_log2 = filters_->ccso.ccso_luma_size_log2;
    for (int plane = 0; plane < 3; ++plane) {
      filters_stream::CcsoPlaneParams& dst = params.plane[plane];
      const post::VideoCcsoPlane& src = filters_->ccso.plane[plane];
      dst.enabled = src.enabled;
      dst.bo_only = src.bo_only;
      dst.scale_idx = src.scale_idx;
      dst.quant_idx = src.quant_idx;
      dst.ext_filter = src.ext_filter;
      dst.edge_clf = src.edge_clf;
      dst.max_band_log2 = src.max_band_log2;
      dst.offset_idx = src.offset_idx;
    }
    if (!filters_stream::ccso_tile_init(&ccso_state_, params, mi_cols_,
                                        mi_rows_, 1, 1, &error)) {
      return false;
    }
    // §8.2.2: seed the CCSO symbol CDFs from the frame image.
    std::memcpy(ccso_state_.cdf, cdfs_.ccso_index,
                sizeof(ccso_state_.cdf));
  }
  std::size_t index = 0;
  int sb_index = 0;
  while (index < blocks_.size()) {
    const BlockSyntax& node = blocks_[index];
    if ((node.flags & kFlagSbRoot) == 0) {
      error = "node " + std::to_string(index) +
              " is not rooted in a superblock (missing kFlagSbRoot)";
      return false;
    }
    if (sb_index >= sb_cols * sb_rows) {
      error = "more superblocks than the frame geometry allows";
      return false;
    }
    const int expect_row = (sb_index / sb_cols) * sb_mi;
    const int expect_col = (sb_index % sb_cols) * sb_mi;
    if (node.mi_row != expect_row || node.mi_col != expect_col) {
      error = "superblock " + std::to_string(sb_index) + " is at mi (" +
              std::to_string(node.mi_row) + "," +
              std::to_string(node.mi_col) + "), expected (" +
              std::to_string(expect_row) + "," +
              std::to_string(expect_col) + ")";
      return false;
    }
    if (node.mi_col == 0) {
      // New superblock row: the left partition context restarts.  Mirrors
      // `if (sb.mi_col == 0) clear_left_context()` in still_encoder.
      clear_left_context();
    }
#if defined(AV2ENC_VIDEO_DELTAQ)
    if (dqp_active_) {
      // §5.20.2.1: ReadDeltas = delta_q_present at every superblock start;
      // the segment-prediction left context restarts at a superblock row.
      video::deltaq::begin_superblock(dqp_, node.mi_col);
      sb_step_ = nullptr;
      if (plan_.delta_q.present) {
        if (sb_index < 0 ||
            static_cast<std::size_t>(sb_index) >= plan_.delta_steps.size()) {
          error = "delta-Q plan is missing superblock " +
                  std::to_string(sb_index);
          return false;
        }
        sb_step_ = &plan_.delta_steps[static_cast<std::size_t>(sb_index)];
      }
      sb_first_leaf_ = true;
    }
#endif
    if (cdef_active) {
      // §5.20.9.6 clear_cdef(): once per superblock, before read_partition.
      filters_stream::cdef_clear_superblock(&cdef_state_, node.mi_row,
                                            node.mi_col);
    }
    // §5.20.10.4 read_lr(): before read_partition(), once per superblock.
    if (!write_lr_superblock(node, error)) return false;
    std::size_t next = 0;
    if (!write_node(index, &next, error)) return false;
    index = next;
    ++sb_index;
  }
  if (sb_index != sb_cols * sb_rows) {
    error = "syntax stream covers " + std::to_string(sb_index) + " of " +
            std::to_string(sb_cols * sb_rows) + " superblocks";
    return false;
  }
  std::size_t payload_bytes = 0;
  const av2::Status status = enc_.exit_symbol(&payload_bytes);
  if (status != av2::Status::kOk) {
    error = std::string("entropy exit_symbol failed: ") +
            av2::status_name(status);
    return false;
  }
  payload_.assign(tile_buffer_.begin(),
                  tile_buffer_.begin() +
                      static_cast<std::ptrdiff_t>(payload_bytes));
  if (tile_bytes != nullptr) *tile_bytes = payload_bytes;
  // AV2ENC_TRACE_TILE_FLUSH: §8.2.4 termination accounting for this tile.
  //   R    renormalisation bits the symbol stream asked for (marker pos)
  //   bits payload bits written (8 * payload_bytes)
  //   nat  bits the historical "marker + 14 zeros + padding" form writes
  //   pad  trailer bits after the marker in that historical form
  //        (always the fourteen marker zeros plus 0..7 alignment bits)
  //   min  shortest conformant payload (Encoder::min_payload_bits)
  //   save bytes the minimum termination saves for this tile
  //        (nat/8 - min/8, independent of --video-tile-flush-min)
  if (std::getenv("AV2ENC_TRACE_TILE_FLUSH") != nullptr) {
    const std::uint32_t r = enc_.marker_bit_pos;
    const std::uint32_t bits = static_cast<std::uint32_t>(payload_bytes) * 8u;
    const std::uint32_t nat_bits = ((r + 15u + 7u) / 8u) * 8u;
    const std::uint32_t min_bits = enc_.min_payload_bits;
    std::fprintf(stderr,
                 "tile_flush R=%u bits=%u bytes=%zu nat=%u pad=%u min=%u "
                 "save=%u low=%u range=%u\n",
                 r, bits, payload_bytes, nat_bits, nat_bits - (r + 15u),
                 min_bits / 8u, nat_bits / 8u - min_bits / 8u, enc_.low,
                 enc_.range);
  }
  // §8.2.4 exit_symbol(): the filter tile CDFs this tile adapted belong to
  // the tile image too.
  if (cdef_active) {
    const int strengths = filters_->cdef.num_strengths;
    std::memcpy(cdfs_.cdef_index0, cdef_state_.index0,
                sizeof(cdfs_.cdef_index0));
    if (std::uint16_t* slot = cdef_minus1_slot(cdfs_, strengths)) {
      std::memcpy(slot, cdef_state_.minus1,
                  sizeof(std::uint16_t) *
                      static_cast<std::size_t>(strengths));
    }
  }
  if (ccso_active) {
    std::memcpy(cdfs_.ccso_index, ccso_state_.cdf,
                sizeof(cdfs_.ccso_index));
  }
  // §7.5/§8.2.4: the LR CDFs adapted by this tile belong to the tile image.
  if (lr_active && filters_->lr.pc_wiener) {
    ::av2enc::filters::pc_wiener_stream::TileStateStoreCdfs(
        pc_state_, &cdfs_.lr_cdfs);
    if (lr_chroma_ready_) {
      filters_stream::lr_state_store_cdfs(lr_state_, &cdfs_.lr_cdfs);
    }
  } else if (lr_active) {
    filters_stream::lr_state_store_cdfs(lr_state_, &cdfs_.lr_cdfs);
  }
  if (cdf_final_ != nullptr) {
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    // The ext writer owns its own copy of the five partition rows; fold the
    // adapted values back into the frame image before exporting it.  The
    // rect writer adapts cdfs_ itself, so it must not be overwritten.
    if (tools_.enable_ext_partitions) export_ext_cdfs_to_image();
#endif
    cdf_final_->resize(cdf_image_words_impl());
    export_cdf_image(cdfs_, cdf_final_->data());
  }
  if (cost_meter_.active) {
    // AV2ENC_SYMBOL_COST=1: one row per symbol family, in information bits.
    // "total" is the information cost the tile actually spent; the payload
    // length also carries the §8.2.4 marker and padding.
    std::cerr << "symbol_cost tile_bytes=" << payload_.size()
              << " info_bits=" << cost_meter_.total_bits();
    for (int tag = 0; tag < kCostTagCount; ++tag) {
      if (cost_meter_.count[tag] == 0) continue;
      std::cerr << " " << cost_tag_name(tag) << "="
                << cost_meter_.count[tag] << "/"
                << cost_meter_.bits[tag];
    }
    std::cerr << "\n";
  }
  if (census_active_ && !append_census(payload_bytes)) {
    std::cerr << "bit_census: cannot append to " << census_path_ << "\n";
  }
  return true;
}

// Attribute a symbol to the CDF array family that owns it.
int Writer::cost_tag_for(const std::uint16_t* p) const {
  const auto in = [p](const auto& arr) {
    const auto* b = reinterpret_cast<const std::uint16_t*>(&arr);
    return p >= b && p < b + sizeof(arr) / sizeof(std::uint16_t);
  };
  if (in(cdfs_.do_split) || in(cdfs_.do_square_split) ||
      in(cdfs_.rect_type)) {
    return kCostPartition;
  }
  if (in(cdfs_.is_inter)) return kCostIsInter;
  if (in(cdfs_.skip)) return kCostSkip;
  if (in(cdfs_.single_mode) || in(cdfs_.single_ref) ||
      in(cdfs_.drl_mode) || in(cdfs_.comp_mode) || in(cdfs_.comp_ref0) ||
      in(cdfs_.comp_ref1) || in(cdfs_.is_joint) ||
      in(cdfs_.compound_mode_non_joint) ||
      in(cdfs_.compound_mode_same_refs) || in(cdfs_.use_optflow) ||
      in(cdfs_.comp_group_idx) || in(cdfs_.compound_type) ||
      in(cdfs_.cwp_idx) || in(cdfs_.wedge_quad) || in(cdfs_.wedge_angle) ||
      in(cdfs_.wedge_dist1) || in(cdfs_.wedge_dist2) ||
      in(cdfs_.tip_mode) || in(cdfs_.tip_drl_mode) ||
      in(cdfs_.tip_pred_mode) || in(cdfs_.use_bawp) ||
      in(cdfs_.use_bawp_chroma) || in(cdfs_.explicit_bawp) ||
      in(cdfs_.explicit_bawp_scale) || in(cdfs_.skip_mode) ||
      in(cdfs_.mv_precision) || in(cdfs_.shell_set) ||
      in(cdfs_.shell6_class0) || in(cdfs_.shell6_class1) ||
      in(cdfs_.shell_last_two) || in(cdfs_.offset_low) ||
      in(cdfs_.offset_class2) || in(cdfs_.offset_other) ||
      in(cdfs_.col_mv_greater) || in(cdfs_.col_mv_index) ||
      in(adv_cdfs_) || in(cdfs_.fsc_mode)) {
    return kCostInterMode;
  }
  if (in(cdfs_.txb_skip) || in(cdfs_.v_txb_skip) || in(cdfs_.eob_pt_1024) ||
      in(cdfs_.eob_pt_512) || in(cdfs_.eob_pt_256) ||
      in(cdfs_.eob_pt_128) || in(cdfs_.eob_pt_64) || in(cdfs_.eob_pt_32) ||
      in(cdfs_.eob_pt_16) || in(cdfs_.eob_extra) || in(cdfs_.coeff_base) ||
      in(cdfs_.coeff_base_lf) || in(cdfs_.coeff_base_eob) ||
      in(cdfs_.coeff_base_lf_eob) || in(cdfs_.coeff_br) ||
      in(cdfs_.coeff_br_lf) || in(cdfs_.coeff_base_uv) ||
      in(cdfs_.coeff_base_lf_uv) || in(cdfs_.coeff_base_eob_uv) ||
      in(cdfs_.coeff_base_lf_eob_uv) || in(cdfs_.coeff_br_uv) ||
      in(cdfs_.dc_sign) || in(cdfs_.cctx_type) ||
      in(cdfs_.inter_tx_type_set1) || in(cdfs_.inter_tx_type_set2) ||
      in(cdfs_.inter_tx_type_set3) || in(cdfs_.inter_tx_type_long) ||
      in(cdfs_.inter_tx_type_index_set1) ||
      in(cdfs_.inter_tx_type_offset_set1) ||
      in(cdfs_.inter_tx_type_index_set2) ||
      in(cdfs_.inter_tx_type_offset_set2) || in(cdfs_.tx_partition)) {
    return kCostResidual;
  }
  if (in(cdfs_.y_mode_set) || in(cdfs_.y_mode_index) ||
      in(cdfs_.y_mode_offset) || in(cdfs_.uv_mode) || in(cdfs_.is_cfl) ||
      in(cdfs_.cfl_index) || in(cdfs_.cfl_sign) || in(cdfs_.cfl_alpha) ||
      in(cdfs_.mrl) || in(cdfs_.dip_cdfs) ||
      in(cdfs_.intra_tx_type_set1) || in(cdfs_.intra_tx_type_long) ||
      in(cdfs_.is_long_side_dct) || in(cdfs_.sec_tx_type) ||
      in(cdfs_.most_probable_stx_set) ||
      in(cdfs_.most_probable_stx_set_adst)) {
    return kCostIntra;
  }
  return kCostOther;
}

// Finer split for the residual bucket (AV2ENC_BIT_CENSUS).  Only called when
// the coarse tag is kCostResidual, so the CDF identity checks below cover the
// coefficient/TX families that the coarse meter folds together.
int Writer::cost_fine_tag_for(const std::uint16_t* p, int coarse) const {
  if (coarse != kCostResidual) return kFineNone;
  const auto in = [p](const auto& arr) {
    const auto* b = reinterpret_cast<const std::uint16_t*>(&arr);
    return p >= b && p < b + sizeof(arr) / sizeof(std::uint16_t);
  };
  if (in(cdfs_.txb_skip) || in(cdfs_.v_txb_skip)) return kFineTxbSkip;
  if (in(cdfs_.eob_pt_1024) || in(cdfs_.eob_pt_512) ||
      in(cdfs_.eob_pt_256) || in(cdfs_.eob_pt_128) ||
      in(cdfs_.eob_pt_64) || in(cdfs_.eob_pt_32) ||
      in(cdfs_.eob_pt_16) || in(cdfs_.eob_extra)) {
    return kFineEob;
  }
  if (in(cdfs_.coeff_base) || in(cdfs_.coeff_base_lf) ||
      in(cdfs_.coeff_base_eob) || in(cdfs_.coeff_base_lf_eob) ||
      in(cdfs_.coeff_base_uv) || in(cdfs_.coeff_base_lf_uv) ||
      in(cdfs_.coeff_base_eob_uv) || in(cdfs_.coeff_base_lf_eob_uv)) {
    return kFineCoeffBase;
  }
  if (in(cdfs_.coeff_br) || in(cdfs_.coeff_br_lf) || in(cdfs_.coeff_br_uv)) {
    return kFineCoeffBr;
  }
  if (in(cdfs_.dc_sign)) return kFineSign;
  if (in(cdfs_.tx_partition)) return kFineTxPartition;
  if (in(cdfs_.cctx_type) || in(cdfs_.inter_tx_type_set1) ||
      in(cdfs_.inter_tx_type_set2) || in(cdfs_.inter_tx_type_set3) ||
      in(cdfs_.inter_tx_type_long) ||
      in(cdfs_.inter_tx_type_index_set1) ||
      in(cdfs_.inter_tx_type_offset_set1) ||
      in(cdfs_.inter_tx_type_index_set2) ||
      in(cdfs_.inter_tx_type_offset_set2)) {
    return kFineTxType;
  }
  return kFineOtherResidual;
}

bool Writer::append_census(std::size_t payload_bytes) {
  if (!census_active_) return true;
  std::lock_guard<std::mutex> lock(g_census_mutex);
  std::ofstream out(census_path_, std::ios::app);
  if (!out) return false;
  out.setf(std::ios::fixed);
  out.precision(3);
  out << "{\"order_hint\":" << plan_.census_frame
      << ",\"tile_mi_row\":" << plan_.tile_mi_row
      << ",\"tile_mi_col\":" << plan_.tile_mi_col
      << ",\"payload_bytes\":" << payload_bytes
      << ",\"info_bits\":" << cost_meter_.total_bits() << ",\"classes\":{";
  bool first = true;
  for (int tag = 0; tag < kCostTagCount; ++tag) {
    if (cost_meter_.count[tag] == 0) continue;
    if (!first) out << ",";
    first = false;
    out << "\"" << cost_tag_name(tag) << "\":["
        << cost_meter_.count[tag] << "," << cost_meter_.bits[tag] << "]";
  }
  out << "},\"fine\":{";
  first = true;
  for (int tag = 1; tag < kFineCount; ++tag) {
    if (cost_meter_.fine_count[tag] == 0) continue;
    if (!first) out << ",";
    first = false;
    out << "\"" << cost_fine_tag_name(tag) << "\":["
        << cost_meter_.fine_count[tag] << "," << cost_meter_.fine_bits[tag]
        << "]";
  }
  out << "}}\n";
  return out.good();
}

void Writer::dump_decision(const BlockSyntax& node) const {
  if (!decision_dump_active_) return;
  std::lock_guard<std::mutex> lock(g_census_mutex);
  std::ofstream out(decision_dump_path_, std::ios::app);
  if (!out) return;
  out << "{\"order_hint\":" << plan_.census_frame
      << ",\"tile_mi_row\":" << plan_.tile_mi_row
      << ",\"tile_mi_col\":" << plan_.tile_mi_col
      << ",\"mi_row\":" << node.mi_row << ",\"mi_col\":" << node.mi_col
      << ",\"bsize\":" << static_cast<int>(node.bsize)
      << ",\"partition\":" << static_cast<int>(node.partition)
      << ",\"is_inter\":" << static_cast<int>(node.is_inter)
      << ",\"skip\":" << static_cast<int>(node.skip_flag)
      << ",\"inter_mode\":" << static_cast<int>(node.inter_mode)
      << ",\"ref\":" << static_cast<int>(node.ref_index)
      << ",\"compound\":" << static_cast<int>(node.compound)
      << ",\"mv_y\":" << node.mv_y << ",\"mv_x\":" << node.mv_x
      << ",\"mvd_y\":" << node.mvd_y << ",\"mvd_x\":" << node.mvd_x
      << ",\"y_mode\":" << static_cast<int>(node.y_mode)
      << ",\"uv_mode\":" << static_cast<int>(node.uv_mode)
      << ",\"tx_size\":" << static_cast<int>(node.tx_size[0])
      << ",\"tx_type\":" << static_cast<int>(node.tx_type[0])
      << ",\"tx_partition\":" << static_cast<int>(node.tx_partition)
      << ",\"eob_y\":" << node.eob[0] << ",\"eob_u\":" << node.eob[1]
      << ",\"eob_v\":" << node.eob[2]
      << ",\"segment\":" << static_cast<int>(node.segment_id)
      << ",\"qindex\":" << static_cast<int>(node.qindex) << "}\n";
}

}  // namespace detail

// The public entry points below keep their unqualified spelling of the
// writer helpers, exactly as they had inside the pre-split TU, where
// they shared one anonymous namespace with them.
using namespace detail;

int intra_joint_mode(int y_mode, int angle_delta) {
  return y_mode_code(y_mode, angle_delta).joint;
}

int FramePlan::sb_cols() const {
  FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  const int sb_mi = partition::sb_size_mi(geometry);
  return (partition::mi_cols(geometry) + sb_mi - 1) / sb_mi;
}

int FramePlan::sb_rows() const {
  FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  const int sb_mi = partition::sb_size_mi(geometry);
  return (partition::mi_rows(geometry) + sb_mi - 1) / sb_mi;
}

int FramePlan::superblock_count() const { return sb_cols() * sb_rows(); }

bool assemble_video_frame(const FramePlan& plan,
                             std::span<const BlockSyntax> blocks,
                             std::span<const std::int32_t> levels,
                             const VideoConfig& config,
                             std::vector<std::uint8_t>& tile_payload,
                             std::string& error, std::size_t* tile_bytes,
                             std::span<const std::uint16_t> cdf_seed,
                             std::vector<std::uint16_t>* cdf_final,
                             const post::VideoFilterPlan* filters) {
  error.clear();
  tile_payload.clear();
  if (!cdf_seed.empty() && cdf_seed.size() != cdf_image_words_impl()) {
    error = "CDF seed has the wrong number of entries";
    return false;
  }
  if (plan.width <= 0 || plan.height <= 0) {
    error = "empty frame plan";
    return false;
  }
#if !defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  if (config.enable_ext_partitions) {
    error = "extended partitions need the full build (partition_ext adapter)";
    return false;
  }
#endif
  // The writer derives MiCols/MiRows from plan.width/height with
  // §5.18.4.4, so any non-zero size is codable; the caller reconstructs the
  // MI-aligned padded frame and crops the output picture.
  if (plan.sb_size != 64 && plan.sb_size != 128 && plan.sb_size != 256) {
    error = "M2 supports 64x64, 128x128 or 256x256 superblocks";
    return false;
  }
  if (config.max_frame_width != static_cast<std::uint16_t>(plan.width) ||
      config.max_frame_height != static_cast<std::uint16_t>(plan.height)) {
    error = "frame plan size does not match the sequence/frame header config";
    return false;
  }

  Writer writer(plan, blocks, levels, config, cdf_seed, cdf_final, filters);
  if (!writer.run(tile_bytes, error)) return false;
  tile_payload = writer.payload();
  return true;
}

std::size_t cdf_image_words() { return cdf_image_words_impl(); }

bool cdf_image_defaults_for_base_q_idx(int base_q_idx,
                                       std::vector<std::uint16_t>& out,
                                       std::string* error) {
  if (base_q_idx < 1 || base_q_idx > 255) {
    if (error != nullptr) {
      *error = "base_q_idx must be in [1, 255] for a CDF default image";
    }
    return false;
  }
  CdfState state{};
  init_cdfs(state, qctx::slice_for_base_q_idx(base_q_idx));
  out.resize(cdf_image_words_impl());
  export_cdf_image(state, out.data());
  return true;
}

bool cdf_image_tables(std::vector<std::uint16_t>& frame,
                      std::vector<std::uint16_t>& tile,
                      std::vector<std::uint16_t>& saved,
                      std::vector<std::uint16_t>& defaults,
                      std::vector<entropy::avg_cdf::CdfTable>& tables,
                      std::string* error) {
  const std::size_t words = cdf_image_words_impl();
  if (frame.size() != words || tile.size() != words ||
      saved.size() != words || defaults.size() != words) {
    if (error != nullptr) {
      *error = "CDF images must all have cdf_image_words() entries";
    }
    return false;
  }
  tables.clear();
  for (const CdfArrayDesc& desc : cdf_arrays()) {
    const std::size_t entries = static_cast<std::size_t>(desc.symbols) + 1U;
    if (entries == 0U || desc.words % entries != 0U) {
      if (error != nullptr) *error = "CDF array size is not a multiple of N+1";
      return false;
    }
    const std::size_t instances = desc.words / entries;
    for (std::size_t i = 0; i < instances; ++i) {
      entropy::avg_cdf::CdfTable table;
      table.symbols = desc.symbols;
      table.frame = frame.data() + desc.packed_offset + i * entries;
      table.tile = tile.data() + desc.packed_offset + i * entries;
      table.saved = saved.data() + desc.packed_offset + i * entries;
      table.defaults = defaults.data() + desc.packed_offset + i * entries;
      tables.push_back(table);
    }
  }
  return true;
}

int y_mode_joint(int av2_mode, int angle_delta) {
  return y_mode_code(av2_mode, angle_delta).joint;
}

void build_scan(int w, int h, std::span<std::uint16_t> scan) {
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[static_cast<std::size_t>(c)] =
        static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
}

void export_inter_rate_cdfs(int base_q_idx,
                            std::span<const std::uint16_t> cdf_image,
                            InterRateCdfRows* out) {
  if (out == nullptr) return;
  detail::CdfState state;
  detail::init_cdfs(state,
                    ::av2enc::still::qctx::slice_for_base_q_idx(base_q_idx),
                    /*reduced_tx_part_set=*/false);
  if (!cdf_image.empty()) {
    detail::import_cdf_image(state, cdf_image.data());
  }
  std::memcpy(out->is_inter, state.is_inter, sizeof(out->is_inter));
  std::memcpy(out->skip, state.skip, sizeof(out->skip));
  std::memcpy(out->comp_mode, state.comp_mode, sizeof(out->comp_mode));
  std::memcpy(out->single_ref, state.single_ref, sizeof(out->single_ref));
  std::memcpy(out->single_mode, state.single_mode, sizeof(out->single_mode));
  std::memcpy(out->drl_mode, state.drl_mode, sizeof(out->drl_mode));
  std::memcpy(out->use_bawp, state.use_bawp, sizeof(out->use_bawp));
  std::memcpy(out->explicit_bawp, state.explicit_bawp,
              sizeof(out->explicit_bawp));
  std::memcpy(out->use_bawp_chroma, state.use_bawp_chroma,
              sizeof(out->use_bawp_chroma));
  std::memcpy(out->shell_set, state.shell_set, sizeof(out->shell_set));
  std::memcpy(out->shell6_class0, state.shell6_class0,
              sizeof(out->shell6_class0));
  std::memcpy(out->shell6_class1, state.shell6_class1,
              sizeof(out->shell6_class1));
  std::memcpy(out->shell_last_two, state.shell_last_two,
              sizeof(out->shell_last_two));
  std::memcpy(out->offset_low, state.offset_low, sizeof(out->offset_low));
  std::memcpy(out->offset_class2, state.offset_class2,
              sizeof(out->offset_class2));
  std::memcpy(out->offset_other, state.offset_other,
              sizeof(out->offset_other));
  std::memcpy(out->col_mv_greater, state.col_mv_greater,
              sizeof(out->col_mv_greater));
  std::memcpy(out->col_mv_index, state.col_mv_index,
              sizeof(out->col_mv_index));
}

}  // namespace av2enc::video::syntax

namespace av2enc::video::rate_interval {

// R19: the §8.2 image → scratch store bridge.  `image` is the packed word
// array the tile writer imports (frame-start live state); empty selects the
// §9.3 defaults for `base_q_idx`.
bool load_store_from_image(std::span<const std::uint16_t> image,
                           int base_q_idx, CdfStore& out) {
  if (image.empty()) {
    load_defaults(out, base_q_idx);
    return true;
  }
  if (image.size() != syntax::cdf_image_words_impl()) return false;
  syntax::detail::CdfState state{};
  syntax::detail::import_cdf_image(state, image.data());
  load_from_state(out, state);
  return true;
}

}  // namespace av2enc::video::rate_interval
