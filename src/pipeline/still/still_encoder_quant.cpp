// Still-picture encoder: quantisation, dequantisation, RDOQ and the
// coefficient level stream.  Split verbatim out of still_encoder.cpp;
// see still_encoder_internal.h for the shared declaration.

#include "pipeline/still/still_encoder_internal.h"

namespace av2enc::still {
namespace detail {

void Encoder::apply_qindex_step(txv::TxParams& params) const {
    if (quant_step_ == txv::kQuantStep) return;
    params.quant_factor *=
        static_cast<double>(txv::kQuantStep) / static_cast<double>(quant_step_);
  }

int Encoder::coefficient_q(int plane, int index) const {
    if (delta_quant_ == nullptr) return quant_step_;
    const dcdeltaq::PlaneQuant& q = delta_quant_->plane[plane];
    return index == 0 ? q.dc : q.ac;
  }

bool Encoder::dequant_inverse_qindex(const std::int32_t* levels,
                            const txv::TxParams& params, int plane, int eob,
                            int y_mode, int y_angle_delta, int sec_tx_type,
                            int stx_set, int mrl_index,
                            std::int32_t* dequant,
                            std::int16_t* residual) {
    const int n = params.w * params.h;
    // §7.14.4: TCQ doubles dqDenom for luma 2-D transform classes.  IDTX is
    // TX_CLASS_2D too (qm_core.h's `< IDTX` predicate is only for useQm).
    const bool tcq_shift = options_.tcq && plane == 0 && params.tx_type <= 9;
    const int dq_denom =
        tcq_shift ? (params.dq_denom << 1) : params.dq_denom;
    if (options_.delta_q) {
      for (int i = 0; i < n; ++i) {
        const std::int32_t qc = levels[i];
        const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
        const std::int64_t high =
            (aq * coefficient_q(plane, i)) & 0xFFFFFF;
        const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                            txv::kQuantTableBits) /
                                dq_denom;
        const std::int32_t value = qc < 0 ? -dq : dq;
        dequant[i] = std::max(-(1 << 15), std::min((1 << 15) - 1, value));
      }
    } else if (options_.qm) {
      qm::QmDequantParams qp;
      qp.state = &qm_state_;
      qp.tx_size = params.tx_size;
      qp.plane = plane;
      qp.tx_type = params.tx_type;
      qp.dc_quant = quant_step_;
      qp.ac_quant = quant_step_;
      qp.bit_depth = 8;
      if (!qm::dequantize_block(levels, qp, dequant)) {
        error_ = "qm dequantize failed";
        return false;
      }
    } else {
      for (int i = 0; i < n; ++i) {
        const std::int32_t qc = levels[i];
        const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
        const std::int64_t high = (aq * quant_step_) & 0xFFFFFF;
        const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                            txv::kQuantTableBits) /
                                dq_denom;
        const std::int32_t value = qc < 0 ? -dq : dq;
        dequant[i] = std::max(-(1 << 15), std::min((1 << 15) - 1, value));
      }
    }

    // §7.15.3: the decoder applies the secondary inverse to the dequantized
    // luma coefficients before the primary 2-D inverse.
    if (plane == 0 && sec_tx_type != 0) {
      txsec::SecondaryTransformConfig cfg;
      cfg.enable_intra_ist = true;
      txsec::SecondarySyntax stx;
      stx.plane_tx_type = static_cast<txv::TxType>(params.tx_type);
      stx.coded_w = params.coded_w;
      stx.coded_h = params.coded_h;
      stx.y_mode = y_mode;
      stx.angle_delta = y_angle_delta;
      stx.mrl_index = mrl_index;
      stx.most_probable_stx_set = stx_set;
      stx.sec_tx_type = sec_tx_type;
      stx.is_inter = false;
      stx.lossless = false;
      stx.bit_depth = 8;
      const txsec::SecondaryStatus status = txsec::apply_inverse_secondary(
          params, plane, eob, stx, cfg, dequant,
          static_cast<std::size_t>(n));
      if (status != txsec::SecondaryStatus::kApplied) {
        error_ = "secondary transform inverse failed";
        return false;
      }
    }

    txv::inverse_block(dequant, params, residual);
    return true;
  }

bool Encoder::compute_tx(int plane, int x0, int y0, int cw, int ch,
                const std::vector<std::uint8_t>& pred, TxPlan& plan,
                bool choose_tx_type, int y_mode,
                int y_angle_delta, int mrl_index) {
    std::optional<txv::TxParams> params_opt =
        txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error_ = "unsupported transform size";
      return false;
    }
    apply_qindex_step(*params_opt);
    txv::TxType tx_type = txv::TxType::kDctDct;
    const int coded_pels = cw * ch;
    std::vector<std::int16_t> residual_coded(
        static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        residual_coded[static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(cw) +
                       static_cast<std::size_t>(x)] =
            static_cast<std::int16_t>(src - p);
      }
    }

    // Intra luma TX_TYPE selection.  The chooser runs the full candidate set
    // (SSE + lambda * rate); we then re-run the chosen type through the common
    // quantise/RDOQ/inverse path below so both features compose.
    if (choose_tx_type) {
      int mapped_mode = mrl_stream::mapped_intra_mode_for_mrl(
          y_mode, y_angle_delta, mrl_index, cw, ch);
      if (mapped_mode < 0) mapped_mode = 0;
      txtype::TxTypeQuery query;
      query.coded_w = cw;
      query.coded_h = ch;
      query.mapped_mode = mapped_mode;
      query.reduced_tx_set = 0;
      query.dct_only = false;
      txtype::TxTypeCostConfig cost;
      cost.lambda = av2enc::rdo::lambda_sse_from_quant_step(
          1.0 / params_opt->quant_factor, options_.tx_type_sse_scale);
      txtype::TxTypeChoice choice;
      // §5.4.8: M1 signals enable_idtx_intra = 0, so identity-based intra
      // transforms (IDTX / V_DCT / H_DCT) are not allowed.  The long-side sets
      // list them, so filter those candidates before the decision.
      txtype::TxTypeSet candidates;
      if (!txtype::intra_tx_candidates(query, &candidates)) {
        error_ = "tx_type candidate enumeration failed";
        return false;
      }
      txtype::TxTypeCandidate usable[txtype::kMaxTxTypeCandidates];
      int usable_count = 0;
      for (int i = 0; i < candidates.count; ++i) {
        const txtype::TxTypeCandidate& candidate = candidates.candidates[i];
        // enable_idtx_intra = 0 forbids every identity 1D transform.  With the
        // flag on, only the pure 2-D identity type (IDTX) is enabled; the
        // one-axis types (V/H_DCT, V/H_ADST, V/H_FLIPADST) still need the
        // class-specific scans and contexts and remain out of scope.
        txv::Tx1dType row_type = txv::Tx1dType::kDct;
        txv::Tx1dType col_type = txv::Tx1dType::kDct;
        txv::tx_1d_types(candidate.type, &row_type, &col_type);
        const bool is_idtx =
            row_type == txv::Tx1dType::kIdt && col_type == txv::Tx1dType::kIdt;
        const bool mixed_identity =
            (row_type == txv::Tx1dType::kIdt) !=
            (col_type == txv::Tx1dType::kIdt);
        if (mixed_identity) continue;
        if (is_idtx && !options_.idtx) continue;
        usable[usable_count++] = candidate;
      }
      if (usable_count == 0) {
        error_ = "no legal non-identity intra tx_type for this size/mode";
        return false;
      }
      if (!txtype::choose_tx_type_from_candidates(
              residual_coded.data(), query, usable, usable_count, cost,
              &choice, /*levels_out=*/nullptr, /*recon_out=*/nullptr)) {
        error_ = "tx_type selection failed";
        return false;
      }
      tx_type = choice.candidate.type;
      const auto chosen =
          txv::make_tx_params(cw, ch, tx_type);
      if (!chosen.has_value()) {
        error_ = "unsupported transform size for the chosen tx_type";
        return false;
      }
      params_opt = chosen;
      apply_qindex_step(*params_opt);
    }
    plan.plane = plane;
    plan.coded_w = cw;
    plan.coded_h = ch;
    plan.pred = pred;

    // Quantise/inverse parameterised by the effective transform.  Runs once
    // for the chosen TX type, and a second time as DCT_DCT when TX_TYPE
    // selection plus RDOQ leaves eob <= 1 (the decoder infers DCT there).
    const auto run_quant = [&](const txv::TxParams& p) -> bool {
      std::vector<std::int16_t> residual_adj(
          static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h));
      txv::downsample_residual(residual_coded.data(), p, residual_adj.data());
      std::vector<double> coeff(
          static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h));
      plan.levels.assign(coeff.size(), 0);
      plan.dequant_levels.clear();
      if (options_.delta_q) {
        // DC/AC delta-Q: scale the scalar qf by qindex_step / q_coefficient.
        txv::forward_block(residual_adj.data(), p, coeff.data());
        for (std::size_t i = 0; i < coeff.size(); ++i) {
          const double factor =
              static_cast<double>(p.quant_factor) *
              static_cast<double>(quant_step_) /
              static_cast<double>(coefficient_q(
                  plane, static_cast<int>(i)));
          plan.levels[i] = static_cast<std::int32_t>(
              std::lround(coeff[i] * factor));
        }
      } else if (options_.qm) {
        // QM changes the per-coefficient quantizer to q2 = Round2(q * m, 5).
        // The forward target scales the scalar qf by q / q2 so the returned
        // levels reconstruct to the same residual under QM dequant.
        txv::forward_block(residual_adj.data(), p, coeff.data());
        qm::QmDequantParams qp;
        qp.state = &qm_state_;
        qp.tx_size = p.tx_size;
        qp.plane = plane;
        qp.tx_type = p.tx_type;
        qp.dc_quant = quant_step_;
        qp.ac_quant = quant_step_;
        qp.bit_depth = 8;
        std::vector<std::int32_t> q2(coeff.size());
        if (!qm::build_q2_matrix(qp, q2.data())) {
          error_ = "qm q2 matrix build failed";
          return false;
        }
        for (std::size_t i = 0; i < coeff.size(); ++i) {
          const double factor =
              static_cast<double>(p.quant_factor) *
              static_cast<double>(quant_step_) /
              static_cast<double>(q2[i]);
          plan.levels[i] = static_cast<std::int32_t>(
              std::lround(coeff[i] * factor));
        }
      } else {
        txv::forward_transform(residual_adj.data(), p, coeff.data(),
                               plan.levels.data());
      }

      const int area = p.w * p.h;
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      syntax::build_scan(p.w, p.h, scan);

      // Full-trellis RDOQ: same syntax, better coefficient levels.  The
      // trellis rate model is the static qctx-0 AV2 model; the syntax writer
      // and the reconstruction both consume the returned levels, so the
      // stream stays decoder-exact.
      if (options_.rdoq) {
        std::vector<std::int32_t> scalar_scan(
            static_cast<std::size_t>(area));
        std::vector<av2enc::rdoq::Coefficient> input(
            static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          scalar_scan[static_cast<std::size_t>(c)] = plan.levels[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].original = coeff[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].scalar_level =
              plan.levels[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].level_context = 0;
        }
        av2enc::rdoq::av2::BlockInfo info;
        info.plane = plane;
        info.params = p;
        info.txb_skip_context = 0;
        info.dc_sign_context = 0;
        info.qctx = qctx_slice_;
        av2enc::rdoq::av2::Model model;
        std::string rdoq_error;
        if (!model.init(info, scalar_scan, &rdoq_error)) {
          error_ = "rdoq model init failed: " + rdoq_error;
          return false;
        }
        av2enc::rdoq::Config rdoq_config;
        rdoq_config.quant_step = 1.0 / p.quant_factor;
        rdoq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
            rdoq_config.quant_step, options_.rdoq_sse_scale);
        rdoq_config.max_coefficients = area;
        av2enc::rdoq::RdoqResult rdoq_result;
        if (!av2enc::rdoq::rdoq_quantize(input, rdoq_config,
                                         model.rate_model(), rdoq_result,
                                         &rdoq_error)) {
          error_ = "rdoq quantize failed: " + rdoq_error;
          return false;
        }
        std::fill(plan.levels.begin(), plan.levels.end(), 0);
        for (int c = 0; c < rdoq_result.eob; ++c) {
          plan.levels[scan[static_cast<std::size_t>(c)]] =
              rdoq_result.levels[static_cast<std::size_t>(c)];
        }
      }

      // §5.20.7.27 trellis-coded quantisation.  Luma TX_CLASS_2D only.  The
      // DP returns the syntax base levels (what coeff_base/br and the
      // read_quant escape code) and the mapped Quant magnitudes that §7.14.4
      // dequantises after the extra dqDenom shift.
      if (options_.tcq && plane == 0 && p.tx_type <= 9) {
        std::vector<av2enc::tcq::Coefficient> input(
            static_cast<std::size_t>(area));
        std::vector<std::int32_t> reference(static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          const int row = pos / p.w;
          const int col = pos % p.w;
          input[static_cast<std::size_t>(c)].original = coeff[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].level_context = 0;
          input[static_cast<std::size_t>(c)].pre_max =
              (row + col) < 4 ? 8 : 6;
          // Frozen neighbour/escape reference: the scalar level at half the
          // syntax step, exactly like the component's RD harness.
          const std::int32_t scalar = plan.levels[static_cast<std::size_t>(pos)];
          const std::int32_t magnitude = std::abs(scalar);
          const std::int32_t base = (magnitude + 1) / 2;
          reference[static_cast<std::size_t>(c)] =
              scalar < 0 ? -base : base;
        }
        av2enc::tcq::av2::BlockInfo info;
        info.plane = 0;
        info.params = p;
        info.txb_skip_context = 0;
        info.dc_sign_context = 0;
        info.qctx = qctx_slice_;
        info.is_inter = false;
        info.tx_class_2d = true;
        av2enc::tcq::av2::Model model;
        std::string tcq_error;
        if (!model.init(info, reference, &tcq_error)) {
          error_ = "tcq model init failed: " + tcq_error;
          return false;
        }
        av2enc::tcq::Config tcq_config;
        tcq_config.quant_step = 1.0 / p.quant_factor;
        tcq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
            tcq_config.quant_step, options_.tcq_sse_scale);
        tcq_config.max_coefficients = area;
        av2enc::tcq::Result tcq_result;
        if (!av2enc::tcq::quantize(input, tcq_config, model.rate_model(),
                                   tcq_result, &tcq_error)) {
          error_ = "tcq quantize failed: " + tcq_error;
          return false;
        }
        std::fill(plan.levels.begin(), plan.levels.end(), 0);
        plan.dequant_levels.assign(static_cast<std::size_t>(area), 0);
        for (int c = 0; c < tcq_result.eob; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          plan.levels[static_cast<std::size_t>(pos)] =
              tcq_result.base_levels[static_cast<std::size_t>(c)];
          plan.dequant_levels[static_cast<std::size_t>(pos)] =
              tcq_result.quant_levels[static_cast<std::size_t>(c)];
        }
      }

      if (plane == 0 && std::getenv("AV2ENC_STILL_ZERO_DC") != nullptr) {
        plan.levels[0] = 0;
      }
      plan.eob = 0;
      for (int c = area - 1; c >= 0; --c) {
        if (plan.levels[scan[static_cast<std::size_t>(c)]] != 0) {
          plan.eob = c + 1;
          break;
        }
      }
      return true;
    };

    if (!run_quant(*params_opt)) return false;
    if (plane == 0 && tx_type != txv::TxType::kDctDct && plan.eob <= 1) {
      const auto dct = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
      if (!dct.has_value()) {
        error_ = "unsupported transform size for the DCT fallback";
        return false;
      }
      tx_type = txv::TxType::kDctDct;
      params_opt = dct;
      apply_qindex_step(*params_opt);
      if (!run_quant(*params_opt)) return false;
    }

    const txv::TxParams& params = *params_opt;
    plan.tx_size = params.tx_size;
    plan.tx_type = static_cast<int>(tx_type);
    plan.adj_w = params.w;
    plan.adj_h = params.h;

    // §5.20.8.2 / §7.15.3 intra IST.  The spec defines only the decoder
    // inverse; the encoder uses the component's analysis projection on the
    // quantised primary coefficients (documented as non-normative) and picks
    // the candidate that minimises reconstructed SSE + lambda * symbol bits.
    if (options_.ist && plane == 0) {
      txsec::SecondaryTransformConfig ist_cfg;
      ist_cfg.enable_intra_ist = true;
      txsec::SecondarySyntax stx;
      stx.plane_tx_type = tx_type;
      stx.coded_w = params.coded_w;
      stx.coded_h = params.coded_h;
      stx.y_mode = y_mode;
      stx.angle_delta = y_angle_delta;
      stx.mrl_index = mrl_index;
      stx.is_inter = false;
      stx.lossless = false;
      stx.bit_depth = 8;
      if (plan.eob >= 2 &&
          txsec::secondary_syntax_applicable(0, plan.eob, stx, ist_cfg)) {
        txsec::SecondaryCandidate candidates[32];
        const int count = txsec::enumerate_secondary_candidates(
            stx, ist_cfg, plan.eob, candidates, 32);
        if (count > 0) {
          const int area = params.w * params.h;
          std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
          syntax::build_scan(params.w, params.h, scan);
          const auto eob_of = [&](const std::vector<std::int32_t>& lv) {
            for (int c = area - 1; c >= 0; --c) {
              if (lv[scan[static_cast<std::size_t>(c)]] != 0) return c + 1;
            }
            return 0;
          };
          const auto distortion =
              [&](const std::vector<std::int32_t>& lv, int ev, int sec,
                  int set, bool* ok) -> double {
            std::vector<std::int32_t> dq(static_cast<std::size_t>(area));
            std::vector<std::int16_t> rr(static_cast<std::size_t>(coded_pels));
            if (!dequant_inverse_qindex(lv.data(), params, 0, ev, y_mode,
                                        y_angle_delta, sec, set, mrl_index,
                                        dq.data(), rr.data())) {
              *ok = false;
              return 0.0;
            }
            double sse = 0.0;
            for (int y = 0; y < ch; ++y) {
              for (int x = 0; x < cw; ++x) {
                const int p = pred[static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(cw) +
                                   static_cast<std::size_t>(x)];
                const int r = rr[static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(cw) +
                                 static_cast<std::size_t>(x)];
                const int v = clip1(p + r);
                const double d =
                    static_cast<double>(source_sample(plane, x0 + x, y0 + y)) -
                    static_cast<double>(v);
                sse += d * d;
              }
            }
            return sse;
          };
          bool ok = true;
          double best_cost = distortion(plan.levels, plan.eob, 0, 0, &ok);
          if (!ok) return false;
          double lambda = av2enc::rdo::lambda_sse_from_quant_step(
              1.0 / params.quant_factor, options_.ist_sse_scale);
          std::vector<std::int32_t> best_levels;
          int best_eob = plan.eob;
          int best_sec = 0;
          int best_set = 0;
          for (int i = 0; i < count; ++i) {
            const txsec::SecondaryCandidate& candidate = candidates[i];
            std::vector<std::int32_t> lv = plan.levels;
            if (!project_ist_levels(params, tx_type, candidate.kernel,
                                    candidate.sec_tx_type,
                                    candidate.transpose, lv)) {
              continue;
            }
            const int ev = eob_of(lv);
            if (ev < 2) continue;
            txsec::SecondarySyntax cs = stx;
            cs.sec_tx_type = candidate.sec_tx_type;
            cs.most_probable_stx_set = candidate.most_probable_stx_set;
            if (!txsec::secondary_syntax_valid(0, ev, cs, ist_cfg)) continue;
            ok = true;
            const double sse =
                distortion(lv, ev, candidate.sec_tx_type,
                           candidate.most_probable_stx_set, &ok);
            if (!ok) continue;
            const double bits =
                txsec::sec_tx_type_bits(candidate.sec_tx_type, params.tx_size,
                                        false) +
                txsec::most_probable_stx_set_bits(
                    candidate.most_probable_stx_set, tx_type, cw, ch);
            const double cost = sse + lambda * bits;
            if (cost < best_cost) {
              best_cost = cost;
              best_levels = std::move(lv);
              best_eob = ev;
              best_sec = candidate.sec_tx_type;
              best_set = candidate.most_probable_stx_set;
            }
          }
          if (best_sec != 0) {
            plan.levels = std::move(best_levels);
            plan.eob = best_eob;
            plan.sec_tx_type = best_sec;
            plan.most_probable_stx_set = best_set;
          }
        }
      }
    }

    std::vector<std::int32_t> dequant(
        static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
    std::vector<std::int16_t> recon_residual(
        static_cast<std::size_t>(coded_pels));
    const std::int32_t* recon_levels =
        plan.dequant_levels.empty() ? plan.levels.data()
                                    : plan.dequant_levels.data();
    if (!dequant_inverse_qindex(recon_levels, params, plane, plan.eob,
                                y_mode, y_angle_delta, plan.sec_tx_type,
                                plan.most_probable_stx_set, mrl_index,
                                dequant.data(), recon_residual.data())) {
      return false;
    }
    plan.recon.resize(static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        const int r =
            recon_residual[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(cw) +
                           static_cast<std::size_t>(x)];
        plan.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(clip1(p + r));
      }
    }
    return true;
  }

bool Encoder::compute_chroma_cctx(int x0, int y0, int cw, int ch,
                         const std::vector<std::uint8_t>& u_pred,
                         const std::vector<std::uint8_t>& v_pred,
                         TxPlan& u_plan, TxPlan& v_plan) {
    const std::optional<txv::TxParams> params_opt =
        txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error_ = "unsupported chroma transform size for CCTX";
      return false;
    }
    txv::TxParams params = *params_opt;
    apply_qindex_step(params);
    const int area = params.w * params.h;
    const int coded_pels = cw * ch;

    const auto residual_adj = [&](int plane,
                                  const std::vector<std::uint8_t>& pred) {
      std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_pels));
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const int src = source_sample(plane, x0 + x, y0 + y);
          const int p =
              pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)];
          coded[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] =
              static_cast<std::int16_t>(src - p);
        }
      }
      std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
      txv::downsample_residual(coded.data(), params, adj.data());
      return adj;
    };
    const std::vector<std::int16_t> u_adj = residual_adj(1, u_pred);
    const std::vector<std::int16_t> v_adj = residual_adj(2, v_pred);
    std::vector<double> u_coeff(static_cast<std::size_t>(area));
    std::vector<double> v_coeff(static_cast<std::size_t>(area));
    txv::forward_block(u_adj.data(), params, u_coeff.data());
    txv::forward_block(v_adj.data(), params, v_coeff.data());

    av2enc::tcq::CctxConfig config;
    config.enabled = true;
    config.quant_step_u = 1.0 / params.quant_factor;
    config.quant_step_v = config.quant_step_u;
    config.bitdepth = 8;
    std::vector<av2enc::tcq::CctxResult> candidates;
    std::string cctx_error;
    if (!av2enc::tcq::cctx_enumerate_scalar(u_coeff, v_coeff, params.w,
                                            params.h, config, &candidates,
                                            &cctx_error)) {
      error_ = "cctx enumerate failed: " + cctx_error;
      return false;
    }

    // §7.14.4 dequant for one chroma plane (no QM, no delta-Q, no TCQ shift).
    const auto dequant_plane = [&](const std::vector<std::int32_t>& levels,
                                   std::vector<std::int32_t>& dequant) {
      dequant.assign(static_cast<std::size_t>(area), 0);
      for (int i = 0; i < area; ++i) {
        const std::int32_t qc = levels[static_cast<std::size_t>(i)];
        const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
        const std::int64_t high = (aq * quant_step_) & 0xFFFFFF;
        const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                            txv::kQuantTableBits) /
                                params.dq_denom;
        dequant[static_cast<std::size_t>(i)] =
            std::max(-(1 << 15), std::min((1 << 15) - 1,
                                          qc < 0 ? -dq : dq));
      }
    };
    const auto reconstruct = [&](const av2enc::tcq::CctxResult& candidate,
                                 std::vector<std::int16_t>& u_res,
                                 std::vector<std::int16_t>& v_res) {
      std::vector<std::int32_t> du;
      std::vector<std::int32_t> dv;
      dequant_plane(candidate.u_levels, du);
      dequant_plane(candidate.v_levels, dv);
      std::vector<std::int32_t> pu(static_cast<std::size_t>(area));
      std::vector<std::int32_t> pv(static_cast<std::size_t>(area));
      for (int i = 0; i < area; ++i) {
        const av2enc::tcq::CctxPair pair = av2enc::tcq::cctx_dequant_pair(
            du[static_cast<std::size_t>(i)], dv[static_cast<std::size_t>(i)],
            candidate.type, 8);
        pu[static_cast<std::size_t>(i)] = pair.u;
        pv[static_cast<std::size_t>(i)] = pair.v;
      }
      u_res.assign(static_cast<std::size_t>(coded_pels), 0);
      v_res.assign(static_cast<std::size_t>(coded_pels), 0);
      txv::inverse_block(pu.data(), params, u_res.data());
      txv::inverse_block(pv.data(), params, v_res.data());
    };
    const auto distortion = [&](const av2enc::tcq::CctxResult& candidate,
                                const std::vector<std::int16_t>& u_res,
                                const std::vector<std::int16_t>& v_res) {
      double sse = 0.0;
      for (int plane = 1; plane <= 2; ++plane) {
        const std::vector<std::uint8_t>& pred = plane == 1 ? u_pred : v_pred;
        const std::vector<std::int16_t>& res = plane == 1 ? u_res : v_res;
        for (int y = 0; y < ch; ++y) {
          for (int x = 0; x < cw; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x);
            const int v = clip1(static_cast<int>(pred[index]) +
                                static_cast<int>(res[index]));
            const double d =
                static_cast<double>(source_sample(plane, x0 + x, y0 + y)) -
                static_cast<double>(v);
            sse += d * d;
          }
        }
      }
      (void)candidate;
      return sse;
    };

    const double lambda = av2enc::rdo::lambda_sse_from_quant_step(
        1.0 / params.quant_factor, 0.125);
    // §5.20.7.30 scan order: eob is the last non-zero *scan* position, not the
    // raster index.  The component returns raster-order eobs; the syntax needs
    // scan order.
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(params.w, params.h, scan);
    const auto scan_eob = [&](const std::vector<std::int32_t>& levels) {
      for (int c = area - 1; c >= 0; --c) {
        if (levels[static_cast<std::size_t>(
                scan[static_cast<std::size_t>(c)])] != 0) {
          return c + 1;
        }
      }
      return 0;
    };
    // RDOQ is a per-plane trellis over the quantised coefficients.  CCTX
    // quantises the *rotated* U/V pair with the scalar step, so run the same
    // trellis on the rotated coefficients of each candidate type; the decoder
    // dequantises the levels and applies the §7.14.6 inverse rotation, so the
    // resulting stream is exact.  Without this the --still-rdoq flag was
    // silently ignored on CCTX chroma.
    const auto rdoq_tune = [&](av2enc::tcq::CctxResult& candidate) -> bool {
      std::vector<double> rot_u(static_cast<std::size_t>(area));
      std::vector<double> rot_v(static_cast<std::size_t>(area));
      for (int i = 0; i < area; ++i) {
        av2enc::tcq::cctx_forward_rotate(
            u_coeff[static_cast<std::size_t>(i)],
            v_coeff[static_cast<std::size_t>(i)], candidate.type,
            &rot_u[static_cast<std::size_t>(i)],
            &rot_v[static_cast<std::size_t>(i)]);
      }
      for (int plane = 1; plane <= 2; ++plane) {
        const std::vector<double>& rotated = plane == 1 ? rot_u : rot_v;
        std::vector<std::int32_t>& levels =
            plane == 1 ? candidate.u_levels : candidate.v_levels;
        std::vector<std::int32_t> scalar_scan(
            static_cast<std::size_t>(area));
        std::vector<av2enc::rdoq::Coefficient> input(
            static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          scalar_scan[static_cast<std::size_t>(c)] =
              levels[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].original =
              rotated[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].scalar_level =
              levels[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].level_context = 0;
        }
        av2enc::rdoq::av2::BlockInfo info;
        info.plane = plane;
        info.params = params;
        info.txb_skip_context = 0;
        info.dc_sign_context = 0;
        info.qctx = qctx_slice_;
        av2enc::rdoq::av2::Model model;
        std::string rdoq_error;
        if (!model.init(info, scalar_scan, &rdoq_error)) {
          error_ = "cctx rdoq model init failed: " + rdoq_error;
          return false;
        }
        av2enc::rdoq::Config rdoq_config;
        rdoq_config.quant_step = 1.0 / params.quant_factor;
        rdoq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
            rdoq_config.quant_step, options_.rdoq_sse_scale);
        rdoq_config.max_coefficients = area;
        av2enc::rdoq::RdoqResult rdoq_result;
        if (!av2enc::rdoq::rdoq_quantize(input, rdoq_config,
                                         model.rate_model(), rdoq_result,
                                         &rdoq_error)) {
          error_ = "cctx rdoq quantize failed: " + rdoq_error;
          return false;
        }
        std::fill(levels.begin(), levels.end(), 0);
        for (int c = 0; c < rdoq_result.eob; ++c) {
          levels[static_cast<std::size_t>(
              scan[static_cast<std::size_t>(c)])] =
              rdoq_result.levels[static_cast<std::size_t>(c)];
        }
      }
      return true;
    };
    if (options_.rdoq) {
      for (av2enc::tcq::CctxResult& candidate : candidates) {
        if (!rdoq_tune(candidate)) return false;
      }
    }
    int best = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    int best_eob_u = 0;
    int best_eob_v = 0;
    std::vector<std::int16_t> best_u_res;
    std::vector<std::int16_t> best_v_res;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const av2enc::tcq::CctxResult& candidate = candidates[i];
      const int eob_u = scan_eob(candidate.u_levels);
      // The decoder infers cctx_type = 0 and reads no symbol when the U EOB
      // is 0 (all_zero) or 1.  A rotation with eob_u <= 1 is therefore
      // unsignalable.
      if (candidate.type != av2enc::tcq::CctxType::kNone && eob_u <= 1) {
        continue;
      }
      std::vector<std::int16_t> u_res;
      std::vector<std::int16_t> v_res;
      reconstruct(candidate, u_res, v_res);
      const double sse = distortion(candidate, u_res, v_res);
      const double bits =
          eob_u > 1 ? av2enc::tcq::cctx_type_bits(candidate.type) : 0.0;
      const double cost = sse + lambda * bits;
      if (cost < best_cost) {
        best_cost = cost;
        best = static_cast<int>(i);
        best_eob_u = eob_u;
        best_eob_v = scan_eob(candidate.v_levels);
        best_u_res = std::move(u_res);
        best_v_res = std::move(v_res);
      }
    }
    if (best < 0) {
      error_ = "no signallable CCTX candidate";
      return false;
    }
    const av2enc::tcq::CctxResult& chosen =
        candidates[static_cast<std::size_t>(best)];

    const auto fill_plan = [&](TxPlan& plan, int plane,
                               const std::vector<std::int32_t>& levels,
                               std::int32_t eob,
                               const std::vector<std::uint8_t>& pred,
                               const std::vector<std::int16_t>& res) {
      plan.plane = plane;
      plan.tx_size = params.tx_size;
      plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
      plan.coded_w = cw;
      plan.coded_h = ch;
      plan.adj_w = params.w;
      plan.adj_h = params.h;
      plan.eob = eob;
      plan.levels = levels;
      plan.dequant_levels.clear();
      plan.pred = pred;
      plan.recon.resize(static_cast<std::size_t>(coded_pels));
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const std::size_t index =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x);
          plan.recon[index] = static_cast<std::uint8_t>(
              clip1(static_cast<int>(pred[index]) +
                    static_cast<int>(res[index])));
        }
      }
    };
    fill_plan(u_plan, 1, chosen.u_levels, best_eob_u, u_pred, best_u_res);
    fill_plan(v_plan, 2, chosen.v_levels, best_eob_v, v_pred, best_v_res);
    cctx_type_ = static_cast<int>(chosen.type);
    return true;
  }

void Encoder::append_levels(const TxPlan& plan) {
    const int area = plan.adj_w * plan.adj_h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(plan.adj_w, plan.adj_h, scan);
    for (int c = 0; c < plan.eob; ++c) {
      levels_.push_back(plan.levels[scan[static_cast<std::size_t>(c)]]);
    }
  }

}  // namespace detail
}  // namespace av2enc::still
