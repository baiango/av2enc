// AV2 M2 P-frame tile syntax writer: residual syntax.  EOB/coefficient coding
// (§5.20.6), luma transform type (§5.20.8.2) and the TX partition symbols.
#include "pipeline/video/video_syntax_writer.h"

namespace av2enc::video::syntax {
namespace detail {

bool Writer::write_eob(int eob, int plane, int coded_w, int coded_h, bool is_inter) {
  RawScope raw_scope(this, kFineEob);
  const double eob_bits_before =
      eob_dump_active_ ? cost_meter_.fine_bits[kFineEob] : 0.0;
  int eob_pt = 1;
  if (eob == 2) {
    eob_pt = 2;
  } else if (eob >= 3) {
    eob_pt = floor_log2(eob - 1) + 2;
  }
  const int multisize = std::min(5, floor_log2(coded_w)) +
                        std::min(5, floor_log2(coded_h)) - 4;
  std::uint16_t* cdf = nullptr;
  int escape_bits = 0;
  int cdf_n = 8;
  const int cdf_ctx = plane > 0 ? 2 : (is_inter ? 1 : 0);
  switch (multisize) {
    case 0:
      // §5.20.7.27 / §8.3.2: eobMultisize 0 (TX_4X4) codes eob_pt_16,
      // whose default CDF has five symbols (eobPt 1..5) and no escape.
      cdf = &cdfs_.eob_pt_16[cdf_ctx][0];
      escape_bits = 0;
      cdf_n = 5;
      break;
    case 1:
      // eobMultisize 1 (TX_4X8 / TX_8X4) codes eob_pt_32: six symbols
      // (eobPt 1..6), no escape.
      cdf = &cdfs_.eob_pt_32[cdf_ctx][0];
      escape_bits = 0;
      cdf_n = 6;
      break;
    case 2:
      cdf = &cdfs_.eob_pt_64[cdf_ctx][0];
      escape_bits = 0;
      cdf_n = 7;
      break;
    case 3:
      cdf = &cdfs_.eob_pt_128[cdf_ctx][0];
      break;
    case 4:
      cdf = &cdfs_.eob_pt_256[cdf_ctx][0];
      escape_bits = 1;
      break;
    case 5:
      cdf = &cdfs_.eob_pt_512[cdf_ctx][0];
      escape_bits = 2;
      break;
    default:
      cdf = &cdfs_.eob_pt_1024[cdf_ctx][0];
      escape_bits = 2;
      break;
  }
  int symbol = 0;
  int escape = 0;
  if (eob_pt <= cdf_n) {
    symbol = eob_pt - 1;
  } else {
    // The last CDF symbol escapes to eobPt = cdf_n + escape.
    symbol = cdf_n - 1;
    escape = eob_pt - cdf_n;
  }
  if (!sym(cdf, cdf_n, symbol)) return false;
  if (symbol == cdf_n - 1 && escape_bits > 0) {
    if (!write_literal(escape_bits, static_cast<std::uint32_t>(escape))) {
      return false;
    }
  }
  if (eob_pt >= 3) {
    const int base = (1 << (eob_pt - 2)) + 1;
    const int extra = eob - base;
    const int eob_extra = (extra >> (eob_pt - 3)) & 1;
    if (!sym(&cdfs_.eob_extra[0], 2, eob_extra)) return false;
    for (int i = eob_pt - 4; i >= 0; --i) {
      if (!raw_bit(((extra >> i) & 1) != 0)) return false;
    }
  }
  // AV2ENC_EOB_DUMP: record the TXB at the granularity the 'eob' census bucket
  // is actually charged at (see the header).  Diagnosis only.
  if (eob_dump_active_) {
    EobRecord rec;
    rec.txb_index = static_cast<int>(eob_dump_written_);
    rec.plane = plane;
    rec.eob = eob;
    rec.eob_pt = eob_pt;
    rec.cdf_n = cdf_n;
    rec.multisize = multisize;
    rec.extra_symbols = eob_pt >= 3 ? 1 : 0;
    rec.raw_extra_bits = eob_pt >= 3 ? std::max(0, eob_pt - 4) + 1 : 0;
    rec.escape_bits = (symbol == cdf_n - 1) ? escape_bits : 0;
    rec.priced_bits = cost_meter_.fine_bits[kFineEob] - eob_bits_before;
    record_eob_txb(rec);
  }
  return true;
}

bool Writer::write_quant_escape(int x, int& hr_level_avg) {
  const int pred_level = hr_level_avg;
  const int m = clip3(1, 6, pred_level <= 0 ? 0 : floor_log2(pred_level));
  const int k = m + 1;
  const int c_max = std::min(m + 4, 6);
  if (x < (c_max << m)) {
    const int q = x >> m;
    for (int i = 0; i < q; ++i) {
      if (!raw_bit(false)) return false;
    }
    if (!raw_bit(true)) return false;
    if (!write_literal(m, static_cast<std::uint32_t>(x & ((1 << m) - 1)))) {
      return false;
    }
  } else {
    for (int i = 0; i < c_max; ++i) {
      if (!raw_bit(false)) return false;
    }
    int g = 0;
    while (true) {
      const int length = k + g;
      const int x_base = (c_max << m) + (1 << length) - (1 << k);
      if (x < x_base + (1 << length)) {
        for (int i = 0; i < g; ++i) {
          if (!raw_bit(false)) return false;
        }
        if (!raw_bit(true)) return false;
        if (!write_literal(length,
                           static_cast<std::uint32_t>(x - x_base))) {
          return false;
        }
        break;
      }
      ++g;
    }
  }
  hr_level_avg = (x + hr_level_avg) >> 1;
  return true;
}

int Writer::dc_sign_context(int x4, int y4, int w4, int h4) const {
  if (const char* forced = std::getenv("AV2ENC_STILL_DCSIGN_CTX")) {
    return std::atoi(forced);
  }
  int dc_sign = 0;
  for (int k = 0; k < w4; ++k) {
    if (x4 + k < mi_cols_) {
      const int sign = above_dc_[0][static_cast<std::size_t>(x4 + k)];
      if (sign == 1) {
        --dc_sign;
      } else if (sign == 2) {
        ++dc_sign;
      }
    }
  }
  for (int k = 0; k < h4; ++k) {
    if (y4 + k < mi_rows_) {
      const int sign = left_dc_[0][static_cast<std::size_t>(y4 + k)];
      if (sign == 1) {
        --dc_sign;
      } else if (sign == 2) {
        ++dc_sign;
      }
    }
  }
  if (dc_sign < 0) return 1;
  if (dc_sign > 0) return 2;
  return 0;
}

// Luma inter transform-type syntax (§5.20.8.2).  §5.20.8.2 codes a TX_TYPE
// for every inter TX block whose set is not TX_SET_DCTONLY, including
// eob == 1 (the eob <= 1 shortcut is intra-only).  `tx_type` is a TX_TYPE
// value from the TX search; with the search off it is always DCT_DCT and the
// emitted symbols are the historical ones.
bool Writer::write_inter_tx_type(int tx_size, int tx_type, int eob, int coded_w,
                         int coded_h) {
  const auto ts = static_cast<txv::TxSize>(tx_size);
  const residual::InterTxSet set =
      residual::inter_tx_set(ts, plan_.reduced_tx_set);
  if (set == residual::InterTxSet::kDctOnly) return true;
  residual::InterTxCandidate cand;
  if (!residual::inter_tx_symbol(ts, plan_.reduced_tx_set,
                                 /*enable_inter_ddt=*/false,
                                 static_cast<txv::TxType>(tx_type), &cand)) {
    error_ = "inter tx_type is not codable for this size";
    return false;
  }
  const int ctx = inter_tx_type_context(eob, coded_w, coded_h, true);
  const int sqr = kTxSizeSqr[tx_size];
  switch (set) {
    case residual::InterTxSet::kWide64:
    case residual::InterTxSet::kHigh64:
      // WIDE_64/HIGH_64 imply is_long_side_dct = 1; only the type is coded.
      return sym(&cdfs_.inter_tx_type_long[ctx][sqr][0], 4, cand.symbol);
    case residual::InterTxSet::kWide32:
    case residual::InterTxSet::kHigh32:
      if (!sym(&cdfs_.is_long_side_dct[1][0], 2,
               cand.is_long_side_dct ? 1 : 0)) {
        return false;
      }
      return sym(&cdfs_.inter_tx_type_long[ctx][sqr][0], 4, cand.symbol);
    case residual::InterTxSet::kInter1: {
      if (!sym(&cdfs_.inter_tx_type_set1[ctx][sqr][0], 2, cand.symbol)) {
        return false;
      }
      const int off_ctx =
          inter_tx_type_context(eob, coded_w, coded_h, false);
      if (cand.symbol == 0) {
        return sym(&cdfs_.inter_tx_type_index_set1[off_ctx][0], 8,
                   cand.offset_symbol);
      }
      return sym(&cdfs_.inter_tx_type_offset_set1[off_ctx][0], 8,
                 cand.offset_symbol);
    }
    case residual::InterTxSet::kInter2: {
      if (!sym(&cdfs_.inter_tx_type_set2[ctx][0], 2, cand.symbol)) {
        return false;
      }
      const int off_ctx =
          inter_tx_type_context(eob, coded_w, coded_h, false);
      if (cand.symbol == 0) {
        return sym(&cdfs_.inter_tx_type_index_set2[off_ctx][0], 8,
                   cand.offset_symbol);
      }
      return sym(&cdfs_.inter_tx_type_offset_set2[off_ctx][0], 4,
                 cand.offset_symbol);
    }
    case residual::InterTxSet::kDctIdtx:
      return sym(&cdfs_.inter_tx_type_set3[ctx][sqr][0], 2, cand.symbol);
    case residual::InterTxSet::kDctIdtxIddct:
      error_ = "TX_SET_DCT_IDTX_IDDCT is not signalled by this writer";
      return false;
    case residual::InterTxSet::kDctOnly:
      return true;
  }
  error_ = "unknown inter TX set";
  return false;
}

// `level_base` is the index of this plane's first scan-order level.
bool Writer::code_tx_view(const BlockSyntax& node, int plane, const TxView& view,
                  std::size_t level_base, int bsize_w, int bsize_h,
                  std::size_t levels_end, bool is_inter,
                  std::string& error) {
  RawScope raw_scope(this, kFineCoeffMag);
  const int tx_size = view.tx_size;
  if (tx_size >= 25) {
    error = "invalid tx_size " + std::to_string(tx_size) + " on plane " +
            std::to_string(plane);
    return false;
  }
  // §5.20.7.27/.30: the one-axis 1-D classes need the class-specific scan,
  // get_lf_limits() contexts and the dc_sign_horz_vert element.  This writer
  // (and the TX search's 1-D opt-in) does not signal them; pinned dav2d also
  // rejects any negative H_DCT/V_DCT coefficient under --strict 1.  Fail
  // loudly instead of coding a 2-D scan for a 1-D transform.
  if (view.tx_type >= 10 && view.tx_type <= 15) {
    error = "1-D TX types (V/H_DCT, V/H_ADST, V/H_FLIPADST) are not "
            "signalled: dc_sign_horz_vert (§5.20.7.27) needs the "
            "class-specific scan, and pinned dav2d rejects negative "
            "H_DCT/V_DCT coefficients under --strict 1";
    return false;
  }
  const auto params_opt =
      txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                          txv::TxType::kDctDct);
  if (!params_opt.has_value()) {
    error = "unsupported transform size";
    return false;
  }
  const txv::TxParams& params = *params_opt;
  const int coded_w = params.coded_w;
  const int coded_h = params.coded_h;
  const int adj_w = params.w;
  const int adj_h = params.h;
  const int area = adj_w * adj_h;
  const int eob = view.eob;
  if (eob < 0 || eob > area) {
    error = "eob " + std::to_string(eob) + " outside the plane " +
            std::to_string(plane) + " transform area";
    return false;
  }
  if (level_base + static_cast<std::size_t>(eob) > levels_end) {
    error = "coefficient levels overflow the flat level array";
    return false;
  }
  const int w4 = coded_w >> 2;
  const int h4 = coded_h >> 2;
  const int sub_x = plane_sub_x(plane);
  const int sub_y = plane_sub_y(plane);
  const int x4 = view.mi_col >> sub_x;
  const int y4 = view.mi_row >> sub_y;
  const int max_x4 = plane == 0 ? mi_cols_ : chroma_mi_cols();
  const int max_y4 = plane == 0 ? mi_rows_ : chroma_mi_rows();

  int ctx = 0;
  if (plane == 0) {
    int top = 0;
    int left = 0;
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < max_x4) top |= above_level(0, x4 + k);
    }
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < max_y4) left |= left_level(0, y4 + k);
    }
    top = std::min(top, 4);
    left = std::min(left, 4);
    if (bsize_w == coded_w && bsize_h == coded_h) {
      ctx = 0;
    } else {
      ctx = (top + left + 3) >> 1;
    }
  } else {
    int above = 0;
    int left = 0;
    for (int i = 0; i < w4; ++i) {
      if (x4 + i < max_x4) {
        above |= above_level(plane, x4 + i);
        above |= above_dc(plane, x4 + i);
      }
    }
    for (int i = 0; i < h4; ++i) {
      if (y4 + i < max_y4) {
        left |= left_level(plane, y4 + i);
        left |= left_dc(plane, y4 + i);
      }
    }
    ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
    if (plane == 2) {
      if (bsize_w * bsize_h > coded_w * coded_h) ctx += 3;
      if (eob_u_ != 0) ctx += 6;
    } else {
      ctx += 6;
    }
  }

  const int tsc = tx_sz_ctx(tx_size);
  std::uint16_t* skip_cdf =
      plane == 2 ? &cdfs_.v_txb_skip[ctx][0]
                 : &cdfs_.txb_skip[is_inter ? 1 : 0][tsc][ctx][0];
  if (!sym(skip_cdf, 2, eob == 0 ? 1 : 0)) return false;
  if (eob == 0) {
    for (int i = 0; i < w4; ++i) {
      above_level(plane, x4 + i) = 0;
      above_dc(plane, x4 + i) = 0;
    }
    for (int i = 0; i < h4; ++i) {
      left_level(plane, y4 + i) = 0;
      left_dc(plane, y4 + i) = 0;
    }
    if (plane == 1) eob_u_ = 0;
    return true;
  }

  if (!write_eob(eob, plane, coded_w, coded_h, is_inter)) {
    error = error_;
    return false;
  }
  if (plane == 1 && plan_.enable_cctx) {
    // §5.20.7.27: cctx_type follows the U EOB and precedes the coefficients.
    // The decoder reads the symbol when (is_inter || eob != 1); it is absent
    // (inferred 0) for an intra block with U eob == 1.
    if (is_inter || eob != 1) {
      if (!sym(&cdfs_.cctx_type[0], av2enc::tcq::kCctxTypeCount,
               node.cctx_type)) {
        return false;
      }
    } else if (node.cctx_type != 0) {
      error = "cctx_type must be 0 when the U plane has eob == 1";
      return false;
    }
  }
  if (plane == 0) {
    if (is_inter) {
      if (!write_inter_tx_type(tx_size, view.tx_type, eob, coded_w,
                               coded_h)) {
        return false;
      }
    } else {
      switch (luma_tx_type_syntax(tx_size, eob)) {
      case TxTypeSyntax::kNone:
        break;
      case TxTypeSyntax::kIntraSet1:
      case TxTypeSyntax::kLong32:
      case TxTypeSyntax::kLong64: {
        // §5.20.7.29 / §5.20.8.2: invert the signalled set to the symbol for
        // this TX_TYPE.  DCT_DCT is symbol 0 in every table, so the
        // TX_MODE_LARGEST path emits the historical bits.
        txtype::TxTypeQuery query;
        query.coded_w = coded_w;
        query.coded_h = coded_h;
        const int mapped = mrl_stream::mapped_intra_mode_for_mrl(
            node.y_mode, node.y_angle_delta, node.mrl_index, coded_w,
            coded_h);
        query.mapped_mode = mapped < 0 ? 0 : mapped;
        query.reduced_tx_set = plan_.reduced_tx_set;
        query.dct_only = false;
        int symbol = -1;
        bool is_long_side_dct = false;
        if (!txtype::tx_type_symbol(
                query, static_cast<txtype::TxType>(view.tx_type), &symbol,
                &is_long_side_dct)) {
          error_ = "intra tx_type is not codable for this size/mode";
          return false;
        }
        if (luma_tx_type_syntax(tx_size, eob) == TxTypeSyntax::kIntraSet1) {
          if (!sym(&cdfs_.intra_tx_type_set1[kTxSizeSqr[tx_size]][0], 7,
                   symbol)) {
            return false;
          }
        } else if (luma_tx_type_syntax(tx_size, eob) ==
                   TxTypeSyntax::kLong32) {
          if (!sym(&cdfs_.is_long_side_dct[0][0], 2,
                   is_long_side_dct ? 1 : 0)) {
            return false;
          }
          if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4,
                   symbol)) {
            return false;
          }
        } else {
          if (!sym(&cdfs_.intra_tx_type_long[kTxSizeSqr[tx_size]][0], 4,
                   symbol)) {
            return false;
          }
        }
        break;
      }
      }
    }
  }

  if (plane == 0 && !is_inter && config_.enable_intra_ist) {
    // §5.20.8.2: sec_tx_type follows transform_type() and precedes the
    // coefficients.  Luma only; chroma never codes it.
    txsec::SecondaryTransformConfig stx_config;
    stx_config.enable_intra_ist = true;
    txsec::SecondarySyntax stx;
    stx.plane_tx_type = static_cast<txv::TxType>(view.tx_type);
    stx.coded_w = coded_w;
    stx.coded_h = coded_h;
    stx.y_mode = node.y_mode;
    stx.angle_delta = node.y_angle_delta;
    stx.mrl_index = node.mrl_index;
    stx.most_probable_stx_set = view.stx_set;
    stx.sec_tx_type = view.sec_tx_type;
    stx.is_inter = false;
    stx.lossless = false;
    stx.bit_depth = 8;
    if (txsec::secondary_syntax_applicable(0, eob, stx, stx_config)) {
      if (!sym(&cdfs_.sec_tx_type[0][kTxSizeSqr[tx_size]][0], 4,
               view.sec_tx_type)) {
        return false;
      }
      if (view.sec_tx_type != 0) {
        if (!txsec::secondary_syntax_valid(0, eob, stx, stx_config)) {
          error_ = "sec_tx_type is not valid for this block";
          return false;
        }
        const int set_count = txsec::most_probable_stx_set_count(
            stx.plane_tx_type, coded_w, coded_h);
        std::uint16_t* set_cdf = set_count == 4
                                     ? cdfs_.most_probable_stx_set_adst
                                     : cdfs_.most_probable_stx_set;
        if (!sym(set_cdf, set_count, view.stx_set)) return false;
      }
    } else if (view.sec_tx_type != 0) {
      error_ = "sec_tx_type set on a block where IST is not applicable";
      return false;
    }
  }

  static constexpr int kSigRef[5][2] = {
      {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
  static constexpr int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  build_scan(adj_w, adj_h, scan);
  std::vector<int> level_pre(static_cast<std::size_t>(area), 0);
  // Levels arrive in scan order: levels_[level_base + c] is scan position c.
  const auto level_at = [&](int c) {
    return static_cast<int>(
        levels_[level_base + static_cast<std::size_t>(c)]);
  };
  // §5.20.7.27 useTcq: luma-only, TX_CLASS_2D-only, lossy and non-FSC.  A
  // 1-D TX type (10..15) is rejected above, IDTX (9) is TX_CLASS_2D.
  const bool use_tcq = plan_.allow_tcq && plane == 0 && view.tx_type <= 9;
  int tcq_state = 0;
  for (int c = eob - 1; c >= 0; --c) {
    const int pos = scan[static_cast<std::size_t>(c)];
    const int row = pos / adj_w;
    const int col = pos % adj_w;
    const int abs_value = std::abs(level_at(c));
    const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
    const int base_levels = is_lf ? 4 : 2;
    const int base_max = is_lf ? 5 : 3;
    const int br_max = (is_lf && plane > 0) ? 0 : 3;
    const int pre_max = base_max + br_max;
    // Under TCQ the writer levels are full levels `q`; the syntax codes
    // syntax_level_for(q, pre_max) and read_quant carries the escape
    // increment x = (q - l) / 2.
    const int syntax_level =
        use_tcq ? av2enc::tcq::syntax_level_for(abs_value, pre_max)
                : abs_value;
    int base_level = 0;
    int br_level = 0;
    if (syntax_level <= base_levels) {
      base_level = syntax_level;
    } else {
      base_level = std::min(base_levels + 1, base_max);
      const int pre = std::min(syntax_level, pre_max);
      br_level = pre - base_level;
    }

    int ctx_eob = 0;
    if (c != 0) {
      if (c <= area / 8) {
        ctx_eob = 1;
      } else if (c <= area / 4) {
        ctx_eob = 2;
      } else {
        ctx_eob = 3;
      }
    }

    if (c == eob - 1) {
      const int symbol = base_level - 1;
      if (plane == 0) {
        if (is_lf) {
          if (!sym(&cdfs_.coeff_base_lf_eob[tsc][ctx_eob][0], 5, symbol)) {
            return false;
          }
        } else {
          if (!sym(&cdfs_.coeff_base_eob[tsc][ctx_eob][0], 3, symbol)) {
            return false;
          }
        }
      } else {
        if (is_lf) {
          if (!sym(&cdfs_.coeff_base_lf_eob_uv[ctx_eob][0], 5, symbol)) {
            return false;
          }
        } else {
          if (!sym(&cdfs_.coeff_base_eob_uv[ctx_eob][0], 3, symbol)) {
            return false;
          }
        }
      }
    } else {
      int mag = 0;
      const int num_ref = plane == 0 ? 5 : 3;
      for (int idx = 0; idx < num_ref; ++idx) {
        const int rr = row + kSigRef[idx][0];
        const int cc = col + kSigRef[idx][1];
        if (rr < adj_h && cc < adj_w) {
          const int limit = is_lf ? 5 : 3;
          const std::size_t level_pos =
              static_cast<std::size_t>(rr) * static_cast<std::size_t>(adj_w) +
              static_cast<std::size_t>(cc);
          mag += std::min(level_pre[level_pos], limit);
        }
      }
      const int mctx = (mag + 1) >> 1;
      // §8.3.2: coeff_base / coeff_base_lf select their CDF with
      // (tcqState >> 1) & 1.  The state is 0 when TCQ is off.
      const int tcq_ctx = use_tcq ? ((tcq_state >> 1) & 1) : 0;
      if (plane == 0) {
        if (is_lf) {
          int lf_ctx = 0;
          if (c == 0) {
            lf_ctx = std::min(mctx, 8);
          } else if (row + col < 2) {
            lf_ctx = std::min(mctx, 6) + 9;
          } else {
            lf_ctx = std::min(mctx, 4) + 16;
          }
          if (!sym(&cdfs_.coeff_base_lf[tsc][lf_ctx][tcq_ctx][0], 6,
                   base_level)) {
            return false;
          }
        } else {
          const int ctx2 = std::min(mctx, 4);
          int hf_ctx = ctx2;
          if (row + col >= 6 && row + col < 8) {
            hf_ctx = ctx2 + 5;
          } else if (row + col >= 8) {
            hf_ctx = ctx2 + 10;
          }
          if (!sym(&cdfs_.coeff_base[tsc][hf_ctx][tcq_ctx][0], 4,
                   base_level)) {
            return false;
          }
        }
      } else {
        const int ctx2 = std::min(mctx, 3);
        const int uv_ctx = plane == 1 ? ctx2 : ctx2 + 4;
        if (is_lf) {
          if (!sym(&cdfs_.coeff_base_lf_uv[uv_ctx][0], 6, base_level)) {
            return false;
          }
        } else {
          if (!sym(&cdfs_.coeff_base_uv[uv_ctx][0], 4, base_level)) {
            return false;
          }
        }
      }
    }

    const int pre_level = base_level + br_level;
    if (pre_level > base_levels && !(is_lf && plane > 0)) {
      int mag = 0;
      for (int idx = 0; idx < 3; ++idx) {
        const int rr = row + kMagRef[idx][0];
        const int cc = col + kMagRef[idx][1];
        if (rr < adj_h && cc < adj_w) {
          const std::size_t level_pos =
              static_cast<std::size_t>(rr) * static_cast<std::size_t>(adj_w) +
              static_cast<std::size_t>(cc);
          mag += std::min(level_pre[level_pos], 5);
        }
      }
      mag = std::min((mag + 1) >> 1, 6);
      if (plane == 0) {
        int br_ctx = mag;
        if (is_lf) {
          if (pos != 0) br_ctx = mag + 7;
          if (!sym(&cdfs_.coeff_br_lf[br_ctx][0], 4, br_level)) {
            return false;
          }
        } else {
          if (!sym(&cdfs_.coeff_br[mag][0], 4, br_level)) return false;
        }
      } else {
        if (!sym(&cdfs_.coeff_br_uv[std::min(mag, 3)][0], 4, br_level)) {
          return false;
        }
      }
    }
    if (use_tcq) {
      // First pass: Tcq_Next_State[state][syntax_level & 1].
      tcq_state =
          av2enc::tcq::next_state(tcq_state, syntax_level & 1);
    }
    level_pre[static_cast<std::size_t>(pos)] = pre_level;
  }

  int cul_level = 0;
  int dc_category = 0;
  int hr_level_avg = 0;
  for (int c = eob - 1; c >= 0; --c) {
    const int pos = scan[static_cast<std::size_t>(c)];
    const int row = pos / adj_w;
    const int col = pos % adj_w;
    const int pre_level = level_pre[static_cast<std::size_t>(pos)];
    const int level = level_at(c);
    const int abs_value = std::abs(level);
    const bool negative = level < 0;
    if (pre_level != 0) {
      if (row == 0 && col == 0 && plane == 0) {
        const int sign_ctx = dc_sign_context(x4, y4, w4, h4);
        if (trace_) {
          std::cerr << "  dc_sign ctx=" << sign_ctx
                    << " sign=" << (negative ? 1 : 0) << "\n";
        }
        if (!sym(&cdfs_.dc_sign[0][0][sign_ctx][0], 2, negative ? 1 : 0)) {
          return false;
        }
      } else {
        if (!raw_bit(negative)) return false;
      }
      const bool is_lf = plane == 0 ? (row + col) < 4 : (row + col) < 1;
      const int max_level = is_lf ? (plane == 0 ? 8 : 5) : 6;
      if (use_tcq) {
        // §5.20.7.27 read_quant under TCQ escapes at `maxLevel - 1` and adds
        // `x << 1`, where x = (full_level - syntax_level) / 2.  `pre_level`
        // is the syntax level the first pass coded.
        if (pre_level >= max_level - 1) {
          if (!write_quant_escape((abs_value - pre_level) / 2,
                                  hr_level_avg)) {
            return false;
          }
        }
      } else if (abs_value >= max_level) {
        if (!write_quant_escape(abs_value - pre_level, hr_level_avg)) {
          return false;
        }
      }
    }
    if (pos == 0 && abs_value > 0) dc_category = negative ? 1 : 2;
    cul_level = std::min(4, cul_level + abs_value);
  }

  for (int i = 0; i < w4; ++i) {
    above_level(plane, x4 + i) = static_cast<std::uint8_t>(cul_level);
    above_dc(plane, x4 + i) = static_cast<std::uint8_t>(dc_category);
  }
  for (int i = 0; i < h4; ++i) {
    left_level(plane, y4 + i) = static_cast<std::uint8_t>(cul_level);
    left_dc(plane, y4 + i) = static_cast<std::uint8_t>(dc_category);
  }
  if (plane == 1) eob_u_ = 1;
  if (trace_ && plane == 0) {
    std::cerr << "  tx p0 eob=" << eob << " cul=" << cul_level
              << " dcCat=" << dc_category << " dcLevel="
              << level_at(0) << " x4=" << x4 << " y4=" << y4 << " w4=" << w4
              << " h4=" << h4 << "\n";
  }
  return true;
}

// reset_block_context(bw4, bh4) from decode_block(): a skip block resets
// the coefficient level/DC contexts over its area. Inter blocks in this
// milestone are always skip blocks, so the next intra block must see zero
// contexts even if an earlier intra block populated them.
void Writer::reset_coeff_contexts(int r, int c, int w4, int h4) {
  const int x4 = c;
  const int y4 = r;
  for (int i = 0; i < w4; ++i) {
    above_level(0, x4 + i) = 0;
    above_dc(0, x4 + i) = 0;
  }
  for (int i = 0; i < h4; ++i) {
    left_level(0, y4 + i) = 0;
    left_dc(0, y4 + i) = 0;
  }
  if (!has_chroma_) return;
  const int cx4 = c >> sub_x_;
  const int cy4 = r >> sub_y_;
  const int cw4 = w4 >> sub_x_;
  const int ch4 = h4 >> sub_y_;
  for (int plane = 1; plane < kPlaneCount; ++plane) {
    for (int i = 0; i < cw4; ++i) {
      above_level(plane, cx4 + i) = 0;
      above_dc(plane, cx4 + i) = 0;
    }
    for (int i = 0; i < ch4; ++i) {
      left_level(plane, cy4 + i) = 0;
      left_dc(plane, cy4 + i) = 0;
    }
  }
}

// Total number of luma levels the leaf's plan contributed.
int Writer::luma_level_count(const BlockSyntax& node) const {
  if (!plan_.tx_mode_select || node.tx_part_count == 0) {
    return static_cast<int>(node.eob[0]);
  }
  int total = 0;
  for (int i = 0; i < node.tx_part_count; ++i) {
    total += static_cast<int>(node.tx_part_eob[i]);
  }
  return total;
}

// Codes the luma residual of one leaf.  In TX_MODE_LARGEST this is the
// historical single `code_tx()` call; with TX_MODE_SELECT it first emits the
// §5.20.6.3 partition (including the "NONE" do_partition symbol) and then one
// transform block per returned part, using the per-part TX size/type/EOB and
// the part's own 4x4 origin for the §8.3.2 contexts.
bool Writer::code_luma_residual(const BlockSyntax& node, int w, int h,
                        std::size_t levels_end, bool is_inter,
                        std::size_t level_base, std::string& error) {
  if (!plan_.tx_mode_select) {
    const TxView view{node.tx_size[0], node.tx_type[0],
                      static_cast<int>(node.eob[0]), node.mi_row,
                      node.mi_col,
                      static_cast<int>(node.sec_tx_type[0]),
                      static_cast<int>(node.most_probable_stx_set[0])};
    return code_tx_view(node, 0, view, level_base, w, h, levels_end,
                        is_inter, error);
  }
  const int part_count = node.tx_part_count != 0 ? node.tx_part_count : 1;
  if (part_count < 1 || part_count > 4) {
    error = "luma TX part count out of range";
    return false;
  }
  // BLOCK_4X4 codes no partition syntax; every other M2 coding block does.
  if (node.bsize != static_cast<std::uint8_t>(txpartition::kBlock4x4)) {
    txpartition::BlockTxContext ctx;
    ctx.mi_size = node.bsize;
    ctx.is_inter = is_inter;
    ctx.fsc_mode = false;
    ctx.reduced_tx_part_set = plan_.reduced_tx_part_set;
    TxPartitionSink sink;
    sink.writer = this;
    txpartition::SplitGeometry relative;
    if (!txpartition::write_tx_partition(
            sink, cdfs_.tx_partition, ctx,
            static_cast<txpartition::TxPartition>(node.tx_partition),
            &relative, error)) {
      return false;
    }
    if (relative.count != part_count) {
      error = "luma TX part count does not match the signalled partition";
      return false;
    }
    for (int i = 0; i < part_count; ++i) {
      if (relative.parts[i].tx_size != node.tx_part_tx_size[i]) {
        error = "luma TX part size does not match the signalled geometry";
        return false;
      }
    }
  } else if (part_count != 1) {
    error = "BLOCK_4X4 cannot signal more than one transform";
    return false;
  }

  std::size_t base = level_base;
  for (int i = 0; i < part_count; ++i) {
    const TxView view{
        node.tx_part_tx_size[i], node.tx_part_tx_type[i],
        static_cast<int>(node.tx_part_eob[i]),
        static_cast<int>(node.mi_row) + static_cast<int>(node.tx_part_mi_row[i]),
        static_cast<int>(node.mi_col) + static_cast<int>(node.tx_part_mi_col[i]),
        static_cast<int>(node.tx_part_sec_tx_type[i]),
        static_cast<int>(node.tx_part_stx_set[i])};
    if (!code_tx_view(node, 0, view, base, w, h, levels_end, is_inter,
                      error)) {
      return false;
    }
    base += node.tx_part_eob[i];
  }
  return true;
}

// Inter residual syntax: transform coefficients for a non-skip inter leaf.
// Mirrors write_intra_leaf's residual half; prediction is already applied by
// the producer.  Chroma never signals a transform type and M2 does not
// enable CCTX/parity hiding/TCQ, so the plane order is luma, U, V.
bool Writer::write_inter_residual(const BlockSyntax& node, int w, int h, int cw,
                          int ch, std::string& error) {
  const int coded_w[kPlaneCount] = {w, cw, cw};
  const int coded_h[kPlaneCount] = {h, ch, ch};
  const int plane_count = has_chroma_ ? kPlaneCount : 1;
  for (int plane = plan_.tx_mode_select ? 1 : 0; plane < plane_count;
       ++plane) {
    const auto params_opt = txv::make_tx_params(
        static_cast<txv::TxSize>(node.tx_size[plane]),
        txv::TxType::kDctDct);
    if (!params_opt.has_value() ||
        params_opt->coded_w != coded_w[plane] ||
        params_opt->coded_h != coded_h[plane]) {
      error = "inter tx_size on plane " + std::to_string(plane) +
              " does not match the TX_MODE_LARGEST block size";
      return false;
    }
  }

  const std::size_t levels_end = levels_.size();
  std::size_t level_base = node.levels_offset;
  if (!code_luma_residual(node, w, h, levels_end, true, level_base, error)) {
    return false;
  }
  if (!has_chroma_) {
    if (error.empty() && !error_.empty()) error = error_;
    return true;
  }
  level_base += static_cast<std::size_t>(luma_level_count(node));
  const TxView u_view{node.tx_size[1], node.tx_type[1],
                      static_cast<int>(node.eob[1]), node.mi_row,
                      node.mi_col};
  if (!code_tx_view(node, 1, u_view, level_base, cw, ch, levels_end, true,
                    error)) {
    return false;
  }
  level_base += node.eob[1];
  const TxView v_view{node.tx_size[2], node.tx_type[2],
                      static_cast<int>(node.eob[2]), node.mi_row,
                      node.mi_col};
  if (!code_tx_view(node, 2, v_view, level_base, cw, ch, levels_end, true,
                    error)) {
    return false;
  }
  if (error.empty() && !error_.empty()) error = error_;
  return true;
}

}  // namespace detail
}  // namespace av2enc::video::syntax
