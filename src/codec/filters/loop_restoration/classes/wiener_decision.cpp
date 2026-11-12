#include "codec/filters/loop_restoration/classes/wiener_decision.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::restoration::classes {
namespace {

constexpr int kScratchUnitBytes = 1 << 16;
constexpr int kScratchTileBytes = 1 << 24;

void zero_coeffs(std::int16_t* coeffs) {
  for (int i = 0; i < kWienerNsCoeffStride; ++i) {
    coeffs[i] = 0;
  }
}

int unit_size_from_shift(int plane, int shift, int sub_x, int sub_y) {
  const int sub = std::max(sub_x, sub_y);
  const int base = (plane == 0) ? kRestorationTileSizeMax
                                : (kRestorationTileSizeMax >> sub);
  return base >> shift;
}

int shift_from_unit_size(int plane, int unit_size, int sub_x, int sub_y) {
  const int sub = std::max(sub_x, sub_y);
  const int base = (plane == 0) ? kRestorationTileSizeMax
                                : (kRestorationTileSizeMax >> sub);
  int shift = 0;
  int v = base;
  while (v > unit_size && shift < 8) {
    v >>= 1;
    ++shift;
  }
  return shift;
}

int max_shift_for_sb(int sb_size) {
  if (sb_size >= 256) {
    return 1;
  }
  if (sb_size >= 128) {
    return 2;
  }
  return 3;
}

int unit_count(int unit_size, int frame_size) {
  return lr_count_units(unit_size, frame_size);
}

WienerUnitSyntax to_syntax(const WienerUnitPlan& plan) {
  WienerUnitSyntax unit;
  unit.type = plan.type;
  unit.merged = plan.merged;
  unit.use_bank_k = plan.use_bank_k;
  unit.subset = plan.subset;
  unit.uv_sym = plan.uv_sym;
  for (int j = 0; j < kWienerNsCoeffStride; ++j) {
    unit.coeffs[j] = plan.coeffs[j];
  }
  return unit;
}

bool measure_unit_bits(const WienerTileState& sim, int plane, int row,
                       int col, const WienerUnitPlan& plan, int* bits,
                       std::string* error) {
  WienerTileState copy = sim;
  std::vector<std::uint8_t> buffer(kScratchUnitBytes);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return false;
  }
  const WienerUnitSyntax unit = to_syntax(plan);
  if (!wiener_write_unit(encoder, copy, plane, row, col, unit, error)) {
    return false;
  }
  *bits = static_cast<int>(encoder.prefix_bits);
  return true;
}

bool measure_header_bits(const WienerLrHeaderConfig& config,
                         const WienerLrHeaderParams& params,
                         const WienerReferenceSet& refs, int* bits,
                         std::string* error) {
  BitWriter writer;
  WienerLrHeaderParams copy = params;
  if (!wiener_write_lr_params(writer, config, &copy, refs, error)) {
    return false;
  }
  *bits = static_cast<int>(writer.bit_size());
  return true;
}

WienerLrHeaderConfig make_config(const WienerDecisionInput& input,
                                 const WienerDecisionOptions& options,
                                 int sb_size) {
  WienerLrHeaderConfig config;
  config.coded_lossless = false;
  config.enable_restoration = true;
  config.frame_is_intra =
      input.refs != nullptr ? input.refs->frame_is_intra : true;
  config.is_switch_frame =
      input.refs != nullptr ? input.refs->is_switch_frame : false;
  config.num_total_refs = input.refs != nullptr ? input.refs->num_total_refs : 0;
  config.num_planes = 3;
  config.sb_size = sb_size;
  config.sub_x = input.core.sub_x;
  config.sub_y = input.core.sub_y;
  config.base_q_idx = input.base_q_idx;
  config.lr_tools_disable[0][kRestoreNone] = false;
  config.lr_tools_disable[0][kRestorePcWiener] = true;
  config.lr_tools_disable[0][kRestoreWienerNs] = !options.allow_luma;
  config.lr_tools_disable[0][kRestoreSwitchable] = false;
  config.lr_tools_disable[1][kRestoreNone] = false;
  config.lr_tools_disable[1][kRestorePcWiener] = true;
  config.lr_tools_disable[1][kRestoreWienerNs] = !options.allow_chroma;
  config.lr_tools_disable[1][kRestoreSwitchable] = false;
  return config;
}

struct WorkCore {
  LrCoreFrame core;
  std::vector<std::uint8_t> type_grid;
  std::vector<std::int16_t> coeff_grid;
};

bool build_work_core(const WienerDecisionInput& input, int plane,
                     int unit_size, int unit_rows, int unit_cols, bool frame,
                     int num_classes, const std::uint8_t* subclass_grid,
                     const std::uint8_t* filter_class_grid,
                     int filter_class_count,
                     const std::int16_t* frame_coeffs,
                     WorkCore* work, std::string* error) {
  work->core = input.core;
  work->core.frame_type[plane] = kRestoreWienerNs;
  work->core.frame_filters_on[plane] = frame ? 1 : 0;
  work->core.num_filter_classes = num_classes;
  work->core.subclass_grid = subclass_grid;
  work->core.filter_class_grid = filter_class_grid;
  work->core.filter_class_count = filter_class_count;
  work->core.unit_size[plane] = unit_size;
  work->core.plane[plane].unit_rows = unit_rows;
  work->core.plane[plane].unit_cols = unit_cols;
  const std::size_t units =
      static_cast<std::size_t>(unit_rows) * static_cast<std::size_t>(unit_cols);
  work->type_grid.assign(units, static_cast<std::uint8_t>(kRestoreWienerNs));
  work->coeff_grid.assign(units * kWienerNsCoeffStride, 0);
  work->core.plane[plane].unit_type = work->type_grid.data();
  work->core.plane[plane].unit_coeffs = work->coeff_grid.data();
  work->core.plane[plane].frame_coeffs = frame_coeffs;
  if (work->core.plane[plane].cdef == nullptr ||
      work->core.plane[plane].curr == nullptr) {
    return false;
  }
  (void)error;
  return true;
}

double identity_plane_sse(const WorkCore& work, int plane,
                          const std::uint8_t* target, int target_stride) {
  std::int16_t zero[kWienerNsCoeffStride] = {};
  return wiener_filter_sse(work.core, plane, -1, -1, target, target_stride,
                           zero);
}

bool plan_better(const WienerUnitPlan& a, double a_cost,
                 const WienerUnitPlan& b, double b_cost) {
  if (a_cost != b_cost) {
    return a_cost < b_cost;
  }
  if (a.type != b.type) {
    return a.type < b.type;
  }
  if (a.merged != b.merged) {
    return a.merged;  // prefer merging on an exact tie
  }
  if (a.subset != b.subset) {
    return a.subset < b.subset;
  }
  if (a.use_bank_k != b.use_bank_k) {
    return a.use_bank_k < b.use_bank_k;
  }
  if (a.bits != b.bits) {
    return a.bits < b.bits;
  }
  for (int j = 0; j < kWienerNsCoeffStride; ++j) {
    if (a.coeffs[j] != b.coeffs[j]) {
      return a.coeffs[j] < b.coeffs[j];
    }
  }
  return false;
}

bool choose_unit_plan(const WienerDecisionInput& input, const WorkCore& work,
                      WienerTileState& sim, int plane, int row, int col,
                      const WienerFitResult* fit, double lambda,
                      WienerUnitPlan* chosen, std::string* error) {
  const std::uint8_t* target = input.target[plane];
  const int stride = input.target_stride[plane];
  std::int16_t zero[kWienerNsCoeffStride] = {};
  const double identity_sse =
      wiener_filter_sse(work.core, plane, row, col, target, stride, zero);

  bool have = false;
  WienerUnitPlan best;
  double best_cost = 0.0;

  auto consider = [&](const WienerUnitPlan& plan, double sse) -> bool {
    int bits = 0;
    if (!measure_unit_bits(sim, plane, row, col, plan, &bits, error)) {
      return false;
    }
    WienerUnitPlan scored = plan;
    scored.sse = sse;
    scored.bits = bits;
    const double cost = sse + lambda * static_cast<double>(bits);
    if (!have || plan_better(scored, cost, best, best_cost)) {
      best = scored;
      best_cost = cost;
      have = true;
    }
    return true;
  };

  WienerUnitPlan none;
  none.type = kRestoreNone;
  if (!consider(none, identity_sse)) {
    return false;
  }

  const int cls = 0;
  const int old_size = sim.bank_size[plane][cls];
  const int old_ptr = sim.bank_ptr[plane][cls];
  const int n_banks = old_size <= 0 ? 1 : old_size;

  if (fit != nullptr) {
    for (const WienerFitCandidate& cand : fit->candidates) {
      for (int k = 0; k < n_banks; ++k) {
        WienerUnitPlan plan;
        plan.type = kRestoreWienerNs;
        plan.merged = false;
        plan.use_bank_k = k;
        plan.subset = cand.subset;
        plan.uv_sym = cand.uv_sym;
        for (int j = 0; j < kWienerNsCoeffStride; ++j) {
          plan.coeffs[j] = cand.coeffs[j];
        }
        if (!consider(plan, cand.sse)) {
          return false;
        }
      }
    }
  }

  for (int k = 0; k < n_banks; ++k) {
    const int ref_slot = (old_ptr - k + kLrBankSize) & (kLrBankSize - 1);
    const double sse = wiener_filter_sse(
        work.core, plane, row, col, target, stride,
        sim.ref_bank[plane][cls][ref_slot]);
    WienerUnitPlan plan;
    plan.type = kRestoreWienerNs;
    plan.merged = true;
    plan.use_bank_k = k;
    if (!consider(plan, sse)) {
      return false;
    }
  }

  *chosen = best;
  return true;
}

bool commit_unit(WienerTileState& sim, int plane, int row, int col,
                 const WienerUnitPlan& plan, std::string* error) {
  std::vector<std::uint8_t> buffer(kScratchUnitBytes);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return false;
  }
  WienerUnitSyntax unit;
  unit.type = plan.type;
  unit.merged = plan.merged;
  unit.use_bank_k = plan.use_bank_k;
  unit.subset = plan.subset;
  unit.uv_sym = plan.uv_sym;
  for (int j = 0; j < kWienerNsCoeffStride; ++j) {
    unit.coeffs[j] = plan.coeffs[j];
  }
  return wiener_write_unit(encoder, sim, plane, row, col, unit, error);
}

struct PlaneCandidate {
  int mode = 0;
  int shift = 3;
  int num_classes = 1;
  double sse = 0.0;
  int header_bits = 0;
  int tile_bits = 0;
  double cost = 0.0;
  WienerPlaneDecision detail;
};

bool plane_better(const PlaneCandidate& a, const PlaneCandidate& b) {
  if (a.cost != b.cost) {
    return a.cost < b.cost;
  }
  if (a.mode != b.mode) {
    return a.mode < b.mode;
  }
  if (a.num_classes != b.num_classes) {
    return a.num_classes < b.num_classes;
  }
  if (a.detail.unit_size != b.detail.unit_size) {
    return a.detail.unit_size > b.detail.unit_size;
  }
  return a.shift < b.shift;
}

bool evaluate_none(const WienerDecisionInput& input,
                   const WienerDecisionOptions& options,
                   const WienerLrHeaderConfig& config, int plane,
                   int shift, PlaneCandidate* out, std::string* error) {
  const int default_size = (plane == 0) ? 64 : (64 >> input.core.sub_y);
  const int unit_size =
      shift >= 0
          ? unit_size_from_shift(plane, shift, input.core.sub_x,
                                 input.core.sub_y)
          : default_size;
  const int unit_rows =
      unit_count(unit_size, lr_plane_height(input.core, plane));
  const int unit_cols =
      unit_count(unit_size, lr_plane_width(input.core, plane));
  WorkCore work;
  if (!build_work_core(input, plane, unit_size, unit_rows, unit_cols, false,
                       1, nullptr, nullptr, kWienerNsClasses, nullptr, &work,
                       error)) {
    return false;
  }
  out->mode = 0;
  out->shift = shift >= 0
                   ? shift
                   : shift_from_unit_size(plane, unit_size, input.core.sub_x,
                                          input.core.sub_y);
  out->num_classes = 1;
  out->sse = identity_plane_sse(work, plane, input.target[plane],
                                input.target_stride[plane]);
  out->detail = WienerPlaneDecision{};
  out->detail.mode = 0;
  out->detail.frame_type = kRestoreNone;
  out->detail.frame_filters_on = false;
  out->detail.num_filter_classes = 1;
  out->detail.num_filter_classes_idx = 0;
  out->detail.shift = out->shift;
  out->detail.unit_size = unit_size;
  out->detail.unit_rows = unit_rows;
  out->detail.unit_cols = unit_cols;
  const std::size_t units = static_cast<std::size_t>(unit_rows) *
                            static_cast<std::size_t>(unit_cols);
  out->detail.unit_type.assign(units, static_cast<std::uint8_t>(kRestoreNone));
  out->detail.unit_coeffs.assign(units * kWienerNsCoeffStride, 0);
  out->detail.units.assign(units, WienerUnitPlan{});
  for (WienerUnitPlan& p : out->detail.units) {
    p.type = kRestoreNone;
  }

  WienerLrHeaderParams params;
  params.frame_type[plane] = kRestoreNone;
  params.luma_shift = 3;
  params.chroma_shift = 3;
  if (!measure_header_bits(config, params, input.refs != nullptr
                                                ? *input.refs
                                                : WienerReferenceSet{},
                           &out->header_bits, error)) {
    return false;
  }
  out->tile_bits = 0;
  out->cost = out->sse + options.lambda_bits * out->header_bits;
  out->detail.sse = out->sse;
  out->detail.header_bits = out->header_bits;
  out->detail.tile_bits = 0;
  out->detail.cost = out->cost;
  return true;
}

bool evaluate_per_unit(const WienerDecisionInput& input,
                       const WienerDecisionOptions& options,
                       const WienerLrHeaderConfig& config, int plane,
                       int shift, PlaneCandidate* out, std::string* error) {
  const int unit_size = unit_size_from_shift(plane, shift, input.core.sub_x,
                                             input.core.sub_y);
  const int unit_rows =
      unit_count(unit_size, lr_plane_height(input.core, plane));
  const int unit_cols =
      unit_count(unit_size, lr_plane_width(input.core, plane));
  WorkCore work;
  if (!build_work_core(input, plane, unit_size, unit_rows, unit_cols, false,
                       1, nullptr, nullptr, kWienerNsClasses, nullptr, &work,
                       error)) {
    return false;
  }

  WienerLrHeaderParams params;
  params.frame_type[plane] = kRestoreWienerNs;
  params.frame_filters_on[plane] = false;
  params.luma_shift =
      (plane == 0) ? shift : 3;
  params.chroma_shift =
      (plane == 0) ? 3 : shift;

  const WienerReferenceSet empty_refs;
  const WienerReferenceSet& refs =
      input.refs != nullptr ? *input.refs : empty_refs;
  WienerTileState sim;
  if (!wiener_tile_init(&sim, params, input.core.mi_cols, input.core.mi_rows,
                        input.core.sub_x, input.core.sub_y, error)) {
    return false;
  }

  const double lambda = options.lambda_bits;
  double total_sse = 0.0;
  int total_bits = 0;
  out->detail = WienerPlaneDecision{};
  out->detail.mode = 1;
  out->detail.frame_type = kRestoreWienerNs;
  out->detail.frame_filters_on = false;
  out->detail.num_filter_classes = 1;
  out->detail.num_filter_classes_idx = 0;
  out->detail.shift = shift;
  out->detail.unit_size = unit_size;
  out->detail.unit_rows = unit_rows;
  out->detail.unit_cols = unit_cols;
  const std::size_t units = static_cast<std::size_t>(unit_rows) *
                            static_cast<std::size_t>(unit_cols);
  out->detail.units.assign(units, WienerUnitPlan{});

  WienerFitOptions fit_options;
  for (int row = 0; row < unit_rows; ++row) {
    for (int col = 0; col < unit_cols; ++col) {
      WienerFitResult fit;
      const WienerFitStatus fit_status = wiener_fit_candidates(
          work.core, plane, row, col, -1, nullptr, input.target[plane],
          input.target_stride[plane], fit_options, &fit);
      const WienerFitResult* fit_ptr =
          fit_status == WienerFitStatus::kOk ? &fit : nullptr;
      WienerUnitPlan plan;
      if (!choose_unit_plan(input, work, sim, plane, row, col, fit_ptr,
                            lambda, &plan, error)) {
        return false;
      }
      if (!commit_unit(sim, plane, row, col, plan, error)) {
        return false;
      }
      const std::size_t index =
          static_cast<std::size_t>(row) *
              static_cast<std::size_t>(unit_cols) +
          static_cast<std::size_t>(col);
      out->detail.units[index] = plan;
      total_sse += plan.sse;
      total_bits += plan.bits;
    }
  }
  out->sse = total_sse;
  out->tile_bits = total_bits;
  out->header_bits = 0;
  if (!measure_header_bits(config, params, refs, &out->header_bits, error)) {
    return false;
  }
  out->mode = 1;
  out->shift = shift;
  out->num_classes = 1;
  out->detail.unit_type = sim.unit_type[plane];
  out->detail.unit_coeffs = sim.unit_coeffs[plane];
  out->cost = out->sse + options.lambda_bits *
                             static_cast<double>(out->header_bits +
                                                 out->tile_bits);
  out->detail.sse = out->sse;
  out->detail.header_bits = out->header_bits;
  out->detail.tile_bits = out->tile_bits;
  out->detail.cost = out->cost;
  return true;
}

void resolve_frame_filter(
    int num_classes, int m,
    const std::int16_t final_coeffs[kWienerNsClasses][kWienerNsCoeffStride],
    const WienerSearchResult& search, const WienerReferenceSet& refs,
    std::int16_t out[kWienerNsCoeffStride]) {
  zero_coeffs(out);
  if (m == 0) {
    return;
  }
  if (m < num_classes) {
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      out[j] = final_coeffs[m - 1][j];
    }
    return;
  }
  const int ref_index = m - num_classes;
  if (ref_index < 0 ||
      ref_index >= static_cast<int>(search.ref_filters.size())) {
    return;
  }
  const WienerRefFilterEntry& e =
      search.ref_filters[static_cast<std::size_t>(ref_index)];
  for (int j = 0; j < kWienerNsCoeffStride; ++j) {
    out[j] = refs.ref[e.ref_idx].coeffs[e.ref_plane][e.ref_class][j];
  }
}

bool evaluate_frame_filters(const WienerDecisionInput& input,
                            const WienerDecisionOptions& options,
                            const WienerLrHeaderConfig& config, int plane,
                            int shift, int num_classes,
                            const PcWienerTables* tables,
                            PlaneCandidate* out, std::string* error,
                            std::vector<std::string>* notes) {
  const int unit_size = unit_size_from_shift(plane, shift, input.core.sub_x,
                                             input.core.sub_y);
  const int unit_rows =
      unit_count(unit_size, lr_plane_height(input.core, plane));
  const int unit_cols =
      unit_count(unit_size, lr_plane_width(input.core, plane));
  if (plane > 0 && num_classes != 1) {
    return false;
  }
  std::vector<std::uint8_t> filter_class_grid;
  std::vector<std::uint8_t> subclass_grid;
  std::vector<std::uint8_t> subclass_lookup;
  const std::uint8_t* class_ptr = nullptr;
  const std::uint8_t* subclass_ptr = nullptr;
  int filter_class_count = kWienerNsClasses;

  WienerLrHeaderParams params;
  params.frame_type[plane] = kRestoreWienerNs;
  params.frame_filters_on[plane] = true;
  params.num_filter_classes = (plane == 0) ? num_classes : 1;
  params.num_filter_classes_idx =
      encode_num_filter_classes_idx(params.num_filter_classes);
  if (params.num_filter_classes_idx < 0) {
    return false;
  }
  params.luma_shift = (plane == 0) ? shift : 3;
  params.chroma_shift = (plane == 0) ? 3 : shift;

  WorkCore work;
  if (num_classes > 1) {
    if (tables == nullptr || !tables->has_lut_to_class() ||
        (input.filter_class_grid == nullptr && input.lr_tx_skip == nullptr) ||
        !tables->has_sub_classify2()) {
      if (notes != nullptr) {
        notes->push_back(
            "frame multi-class candidate skipped: Pc_Wiener_Lut_To_Class / "
            "Pc_Wiener_Sub_Classify2 are not in the spec text");
      }
      return false;
    }
    if (!build_work_core(input, plane, unit_size, unit_rows, unit_cols, true,
                         num_classes, nullptr, nullptr, kWienerNsClasses,
                         nullptr, &work, error)) {
      return false;
    }
    const std::size_t cells = static_cast<std::size_t>(input.core.mi_cols) *
                              static_cast<std::size_t>(input.core.mi_rows);
    if (input.filter_class_grid != nullptr) {
      filter_class_grid.assign(input.filter_class_grid,
                               input.filter_class_grid + cells);
    } else {
      WienerClassifierContext ctx;
      ctx.core = &work.core;
      ctx.lr_tx_skip = input.lr_tx_skip;
      ctx.base_q_idx = input.base_q_idx;
      ctx.bit_depth = input.core.bit_depth;
      ctx.mi_row_start = 0;
      ctx.mi_row_end = input.core.mi_rows;
      ctx.mi_col_end = input.core.mi_cols;
      const WienerTableStatus st = wiener_derive_filter_class_grid(
          work.core, *tables, ctx, &filter_class_grid);
      if (st != WienerTableStatus::kOk) {
        if (notes != nullptr) {
          notes->push_back(std::string("frame multi-class candidate skipped: ") +
                           wiener_table_status_string(st));
        }
        return false;
      }
    }
    const int filter_set = lr_filter_set_index(input.base_q_idx);
    const WienerTableStatus st = wiener_build_subclass_grid(
        filter_class_grid.data(), input.core.mi_cols, input.core.mi_rows,
        filter_set, num_classes, params.num_filter_classes_idx, *tables,
        &subclass_grid);
    if (st != WienerTableStatus::kOk) {
      if (notes != nullptr) {
        notes->push_back(std::string("frame multi-class candidate skipped: ") +
                         wiener_table_status_string(st));
      }
      return false;
    }
    subclass_lookup.assign(kNumPcWienerLutClasses, 0);
    for (int cls = 0; cls < kNumPcWienerLutClasses; ++cls) {
      subclass_lookup[static_cast<std::size_t>(cls)] = tables->subclass(
          filter_set, params.num_filter_classes_idx, cls);
    }
    class_ptr = filter_class_grid.data();
    subclass_ptr = subclass_grid.data();
    filter_class_count = kNumPcWienerLutClasses;
  }

  if (subclass_ptr == nullptr) {
    if (!build_work_core(input, plane, unit_size, unit_rows, unit_cols, true,
                         1, nullptr, nullptr, kWienerNsClasses, nullptr,
                         &work, error)) {
      return false;
    }
  } else {
    // Rebuild with the class map so lr_block_context and the fit see it.
    if (!build_work_core(input, plane, unit_size, unit_rows, unit_cols, true,
                         num_classes, subclass_ptr, class_ptr,
                         filter_class_count, nullptr, &work, error)) {
      return false;
    }
  }

  const int n_classes = (plane == 0) ? num_classes : 1;
  std::int16_t fitted[kWienerNsClasses][kWienerNsCoeffStride] = {};
  double fitted_sse[kWienerNsClasses] = {};
  for (int c = 0; c < n_classes; ++c) {
    WienerFitOptions fit_options;
    WienerFitResult fit;
    const WienerFitStatus st = wiener_fit_candidates(
        work.core, plane, -1, -1, c, subclass_ptr, input.target[plane],
        input.target_stride[plane], fit_options, &fit);
    const WienerFitCandidate* best = nullptr;
    if (st == WienerFitStatus::kOk) {
      best = wiener_best_fit_candidate(fit);
    }
    if (best != nullptr) {
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        fitted[c][j] = best->coeffs[j];
      }
      fitted_sse[c] = best->sse;
    } else {
      std::int16_t zero[kWienerNsCoeffStride] = {};
      fitted_sse[c] = wiener_filter_sse_class(
          work.core, plane, c, subclass_ptr, input.target[plane],
          input.target_stride[plane], zero);
    }
  }

  const WienerReferenceSet empty_refs;
  const WienerReferenceSet& refs =
      input.refs != nullptr ? *input.refs : empty_refs;
  const bool nopcw = true;
  const WienerSearchResult search =
      wiener_search_frame_filters(plane, -1, params, refs, nopcw);

  std::int16_t final_coeffs[kWienerNsClasses][kWienerNsCoeffStride] = {};
  std::int16_t resolved[kWienerNsCoeffStride];
  for (int c = 0; c < n_classes; ++c) {
    std::int16_t zero[kWienerNsCoeffStride] = {};
    const double identity_class = wiener_filter_sse_class(
        work.core, plane, c, subclass_ptr, input.target[plane],
        input.target_stride[plane], zero);
    int match = 0;
    bool merged = false;
    if (c == 0) {
      match = 0;
      merged = fitted_sse[c] >= identity_class;
      if (merged) {
        zero_coeffs(final_coeffs[c]);
      } else {
        for (int j = 0; j < kWienerNsCoeffStride; ++j) {
          final_coeffs[c][j] = fitted[c][j];
        }
      }
    } else {
      int best_match = 0;
      double best_sse = identity_class;
      for (int m = 1; m <= c; ++m) {
        resolve_frame_filter(n_classes, m, final_coeffs, search, refs,
                             resolved);
        const double sse = wiener_filter_sse_class(
            work.core, plane, c, subclass_ptr, input.target[plane],
            input.target_stride[plane], resolved);
        if (sse < best_sse) {
          best_sse = sse;
          best_match = m;
        }
      }
      for (int m = n_classes;
           m < n_classes + search.num_ref_filters; ++m) {
        resolve_frame_filter(n_classes, m, final_coeffs, search, refs,
                             resolved);
        const double sse = wiener_filter_sse_class(
            work.core, plane, c, subclass_ptr, input.target[plane],
            input.target_stride[plane], resolved);
        if (sse < best_sse) {
          best_sse = sse;
          best_match = m;
        }
      }
      if (best_sse <= fitted_sse[c]) {
        merged = true;
        match = best_match;
        resolve_frame_filter(n_classes, match, final_coeffs, search,
                             refs, final_coeffs[c]);
      } else {
        merged = false;
        match = best_match;
        for (int j = 0; j < kWienerNsCoeffStride; ++j) {
          final_coeffs[c][j] = fitted[c][j];
        }
      }
    }
    WienerFrameFilterSyntax& f = params.frame_filter[plane][c];
    f.match_index = match;
    f.merged = merged;
    if (!merged) {
      f.subset = 0;
      f.uv_sym = false;
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        f.coeffs[j] = fitted[c][j];
      }
      // Recover the subset/uv_sym of the chosen candidate.
      WienerFitOptions fit_options;
      WienerFitResult fit;
      if (wiener_fit_candidates(work.core, plane, -1, -1, c, subclass_ptr,
                                input.target[plane],
                                input.target_stride[plane], fit_options,
                                &fit) == WienerFitStatus::kOk) {
        const WienerFitCandidate* best = wiener_best_fit_candidate(fit);
        if (best != nullptr) {
          f.subset = best->subset;
          f.uv_sym = best->uv_sym;
        }
      }
    }
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      params.frame_coeffs[plane][c][j] = final_coeffs[c][j];
    }
  }

  if (!measure_header_bits(config, params, refs, &out->header_bits, error)) {
    return false;
  }

  WienerTileState sim;
  if (!wiener_tile_init(&sim, params, input.core.mi_cols, input.core.mi_rows,
                        input.core.sub_x, input.core.sub_y, error)) {
    return false;
  }
  out->detail = WienerPlaneDecision{};
  out->detail.mode = 2;
  out->detail.frame_type = kRestoreWienerNs;
  out->detail.frame_filters_on = true;
  out->detail.num_filter_classes = n_classes;
  out->detail.num_filter_classes_idx = params.num_filter_classes_idx;
  out->detail.shift = shift;
  out->detail.unit_size = unit_size;
  out->detail.unit_rows = unit_rows;
  out->detail.unit_cols = unit_cols;
  out->detail.filter_class_grid = filter_class_grid;
  out->detail.subclass_grid = subclass_grid;
  out->detail.subclass_lookup = subclass_lookup;
  out->detail.filter_class_count = filter_class_count;
  const std::size_t units = static_cast<std::size_t>(unit_rows) *
                            static_cast<std::size_t>(unit_cols);
  out->detail.units.assign(units, WienerUnitPlan{});
  for (int c = 0; c < n_classes; ++c) {
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      out->detail.frame_coeffs[c][j] = final_coeffs[c][j];
    }
    out->detail.frame_filter[c] = params.frame_filter[plane][c];
  }

  double total_sse = 0.0;
  int total_bits = 0;
  for (int row = 0; row < unit_rows; ++row) {
    for (int col = 0; col < unit_cols; ++col) {
      const double frame_sse = wiener_frame_filter_sse(
          work.core, plane, row, col, n_classes, subclass_ptr,
          input.target[plane], input.target_stride[plane], final_coeffs);
      std::int16_t zero[kWienerNsCoeffStride] = {};
      const double identity_sse = wiener_filter_sse(
          work.core, plane, row, col, input.target[plane],
          input.target_stride[plane], zero);
      int best_bits = 0;
      WienerUnitPlan best;
      best.type = kRestoreNone;
      best.sse = identity_sse;
      if (!measure_unit_bits(sim, plane, row, col, best, &best_bits, error)) {
        return false;
      }
      best.bits = best_bits;
      double best_cost = identity_sse + options.lambda_bits * best_bits;
      WienerUnitPlan wiener;
      wiener.type = kRestoreWienerNs;
      int wiener_bits = 0;
      if (!measure_unit_bits(sim, plane, row, col, wiener, &wiener_bits,
                             error)) {
        return false;
      }
      const double wiener_cost =
          frame_sse + options.lambda_bits * wiener_bits;
      if (wiener_cost < best_cost ||
          (wiener_cost == best_cost && wiener.type < best.type)) {
        best = wiener;
        best.sse = frame_sse;
        best.bits = wiener_bits;
        best_cost = wiener_cost;
      }
      if (!commit_unit(sim, plane, row, col, best, error)) {
        return false;
      }
      const std::size_t index =
          static_cast<std::size_t>(row) *
              static_cast<std::size_t>(unit_cols) +
          static_cast<std::size_t>(col);
      out->detail.units[index] = best;
      total_sse += best.sse;
      total_bits += best.bits;
    }
  }
  out->sse = total_sse;
  out->tile_bits = total_bits;
  out->mode = 2;
  out->shift = shift;
  out->num_classes = n_classes;
  out->detail.unit_type = sim.unit_type[plane];
  out->detail.unit_coeffs = sim.unit_coeffs[plane];
  out->cost = out->sse + options.lambda_bits *
                             static_cast<double>(out->header_bits +
                                                 out->tile_bits);
  out->detail.sse = out->sse;
  out->detail.header_bits = out->header_bits;
  out->detail.tile_bits = out->tile_bits;
  out->detail.cost = out->cost;
  return true;
}

}  // namespace

const char* wiener_decision_status_string(WienerDecisionStatus status) {
  switch (status) {
    case WienerDecisionStatus::kOk:
      return "ok";
    case WienerDecisionStatus::kInvalidArgument:
      return "invalid argument";
    case WienerDecisionStatus::kMissingNormativeTable:
      return "missing normative Pc_Wiener_* table";
    case WienerDecisionStatus::kNoCandidates:
      return "no usable restoration candidate";
    case WienerDecisionStatus::kSyntaxError:
      return "syntax writer rejected the decision";
  }
  return "unknown";
}

WienerDecisionStatus wiener_decide(const WienerDecisionInput& input,
                                   const WienerDecisionOptions& options,
                                   WienerDecision* decision) {
  if (decision == nullptr || input.core.mi_cols <= 0 ||
      input.core.mi_rows <= 0 || input.core.bit_depth != 8) {
    return WienerDecisionStatus::kInvalidArgument;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (input.target[plane] == nullptr ||
        input.target_stride[plane] < lr_plane_width(input.core, plane)) {
      return WienerDecisionStatus::kInvalidArgument;
    }
  }
  const int sb_size = kMiSize << input.core.sb_shift;
  const int max_shift = max_shift_for_sb(sb_size);
  const WienerLrHeaderConfig config = make_config(input, options, sb_size);
  const WienerReferenceSet empty_refs;
  const WienerReferenceSet& refs =
      input.refs != nullptr ? *input.refs : empty_refs;

  std::array<PlaneCandidate, 3> best;
  bool have[3] = {false, false, false};
  std::vector<std::string> notes;

  auto consider = [&](int plane, const PlaneCandidate& cand) {
    if (!have[plane] || plane_better(cand, best[static_cast<std::size_t>(plane)])) {
      best[static_cast<std::size_t>(plane)] = cand;
      have[plane] = true;
    }
  };

  for (int plane = 0; plane < 3; ++plane) {
    const bool allowed =
        (plane == 0) ? options.allow_luma : options.allow_chroma;
    PlaneCandidate none;
    const int forced_none = (plane == 0) ? options.forced_luma_shift
                                         : options.forced_chroma_shift;
    if (!evaluate_none(input, options, config, plane, forced_none, &none,
                       nullptr)) {
      return WienerDecisionStatus::kSyntaxError;
    }
    consider(plane, none);
    if (!allowed) {
      continue;
    }
    const int forced = (plane == 0) ? options.forced_luma_shift
                                    : options.forced_chroma_shift;
    int shift_begin = forced >= 0 ? forced : 0;
    int shift_end = forced >= 0 ? forced : max_shift;
    if (plane == 2 && have[1]) {
      // §5.18.7.11 sets LoopRestorationSize[2] = LoopRestorationSize[1] and
      // codes one chroma size flag group.  Keep the search conformant by
      // evaluating plane 2 at the shift already chosen for plane 1.
      shift_begin = best[static_cast<std::size_t>(1)].shift;
      shift_end = shift_begin;
    }
    if (shift_begin < 0 || shift_end > max_shift || shift_begin > shift_end) {
      return WienerDecisionStatus::kInvalidArgument;
    }
    for (int shift = shift_begin; shift <= shift_end; ++shift) {
      const int unit_size =
          unit_size_from_shift(plane, shift, input.core.sub_x,
                               input.core.sub_y);
      if (unit_size <= 0) {
        continue;
      }
      if (options.allow_per_unit) {
        PlaneCandidate cand;
        std::string err;
        if (evaluate_per_unit(input, options, config, plane, shift, &cand,
                              &err)) {
          consider(plane, cand);
        } else if (!err.empty()) {
          return WienerDecisionStatus::kSyntaxError;
        }
      }
      if (options.allow_frame_filters) {
        for (int nclasses = 1;
             nclasses <= options.max_num_filter_classes; ++nclasses) {
          if (plane > 0 && nclasses != 1) {
            continue;
          }
          if (nclasses != 1 && !valid_num_filter_classes(nclasses)) {
            continue;
          }
          PlaneCandidate cand;
          std::string err;
          if (evaluate_frame_filters(input, options, config, plane, shift,
                                     nclasses, input.tables, &cand, &err,
                                     &notes)) {
            consider(plane, cand);
          } else if (!err.empty()) {
            return WienerDecisionStatus::kSyntaxError;
          }
        }
      }
    }
  }

  if (!have[0] || !have[1] || !have[2]) {
    return WienerDecisionStatus::kNoCandidates;
  }

  *decision = WienerDecision{};
  decision->notes = notes;
  for (int plane = 0; plane < 3; ++plane) {
    const WienerPlaneDecision& pd = best[static_cast<std::size_t>(plane)].detail;
    decision->unit_size[plane] = pd.unit_size;
    decision->unit_rows[plane] = pd.unit_rows;
    decision->unit_cols[plane] = pd.unit_cols;
    decision->unit_type[plane] = pd.unit_type;
    decision->unit_coeffs[plane] = pd.unit_coeffs;
    decision->plane[plane] = pd;
    decision->params.frame_type[plane] = pd.frame_type;
    decision->params.frame_filters_on[plane] = pd.frame_filters_on;
    decision->params.temporal_pred_flag[plane] = false;
    decision->params.rst_ref_pic_idx[plane] = 0;
    for (int c = 0; c < kWienerNsClasses; ++c) {
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        decision->params.frame_coeffs[plane][c][j] =
            pd.frame_coeffs[c][j];
      }
      decision->params.frame_filter[plane][c] = pd.frame_filter[c];
    }
    if (plane == 0) {
      decision->params.num_filter_classes = pd.num_filter_classes;
      decision->params.num_filter_classes_idx = pd.num_filter_classes_idx;
      if (pd.num_filter_classes > 1) {
        decision->filter_class_grid = pd.filter_class_grid;
        decision->subclass_grid = pd.subclass_grid;
        decision->subclass_lookup = pd.subclass_lookup;
        decision->filter_class_count = pd.filter_class_count;
        decision->num_filter_classes = pd.num_filter_classes;
      }
    }
    decision->params.luma_shift = pd.shift;
    decision->params.chroma_shift = pd.shift;
  }
  // Shifts are per plane; luma and chroma can pick different sizes.
  decision->params.luma_shift = best[0].detail.shift;
  decision->params.chroma_shift = best[1].detail.shift;

  // A RESTORE_NONE plane has no syntax plan, but it still owns a unit grid in
  // the tile state.  Re-derive it from the final chroma/luma shift so the
  // read_lr() walk and the plan vectors agree.
  for (int plane = 0; plane < 3; ++plane) {
    const int final_size = unit_size_from_shift(
        plane, (plane == 0) ? decision->params.luma_shift
                            : decision->params.chroma_shift,
        input.core.sub_x, input.core.sub_y);
    const int final_rows =
        unit_count(final_size, lr_plane_height(input.core, plane));
    const int final_cols =
        unit_count(final_size, lr_plane_width(input.core, plane));
    WienerPlaneDecision& pd = best[static_cast<std::size_t>(plane)].detail;
    const std::size_t units = static_cast<std::size_t>(final_rows) *
                              static_cast<std::size_t>(final_cols);
    if (pd.mode == 0) {
      pd.unit_size = final_size;
      pd.unit_rows = final_rows;
      pd.unit_cols = final_cols;
      pd.unit_type.assign(units, static_cast<std::uint8_t>(kRestoreNone));
      pd.unit_coeffs.assign(units * kWienerNsCoeffStride, 0);
      pd.units.assign(units, WienerUnitPlan{});
      for (WienerUnitPlan& u : pd.units) {
        u.type = kRestoreNone;
      }
    } else if (pd.unit_size != final_size || pd.unit_rows != final_rows ||
               pd.unit_cols != final_cols) {
      return WienerDecisionStatus::kInvalidArgument;
    }
    decision->unit_size[plane] = final_size;
    decision->unit_rows[plane] = final_rows;
    decision->unit_cols[plane] = final_cols;
  }

  std::string err;
  if (!measure_header_bits(config, decision->params, refs,
                           &decision->header_bits, &err)) {
    return WienerDecisionStatus::kSyntaxError;
  }

  // Replay the tile in read_lr() order and capture the final grids.
  WienerTileState sim;
  if (!wiener_tile_init(&sim, decision->params, input.core.mi_cols,
                        input.core.mi_rows, input.core.sub_x,
                        input.core.sub_y, &err)) {
    return WienerDecisionStatus::kSyntaxError;
  }
  std::vector<WienerUnitSyntax> plans[3];
  for (int plane = 0; plane < 3; ++plane) {
    plans[static_cast<std::size_t>(plane)].resize(best[static_cast<std::size_t>(plane)].detail.units.size());
    for (std::size_t i = 0; i < plans[static_cast<std::size_t>(plane)].size(); ++i) {
      const WienerUnitPlan& p = best[static_cast<std::size_t>(plane)].detail.units[i];
      WienerUnitSyntax u;
      u.type = p.type;
      u.merged = p.merged;
      u.use_bank_k = p.use_bank_k;
      u.subset = p.subset;
      u.uv_sym = p.uv_sym;
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        u.coeffs[j] = p.coeffs[j];
      }
      plans[static_cast<std::size_t>(plane)][i] = u;
    }
  }
  std::vector<std::uint8_t> buffer(kScratchTileBytes);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return WienerDecisionStatus::kSyntaxError;
  }
  if (!wiener_write_tile(encoder, sim, plans, sb_size, 0, 0, &err)) {
    return WienerDecisionStatus::kSyntaxError;
  }
  decision->tile_bits = static_cast<int>(encoder.prefix_bits);
  for (int plane = 0; plane < 3; ++plane) {
    decision->unit_type[plane] = sim.unit_type[plane];
    decision->unit_coeffs[plane] = sim.unit_coeffs[plane];
    decision->plane[plane].unit_type = sim.unit_type[plane];
    decision->plane[plane].unit_coeffs = sim.unit_coeffs[plane];
  }
  double total_sse = 0.0;
  for (int plane = 0; plane < 3; ++plane) {
    total_sse += best[static_cast<std::size_t>(plane)].sse;
  }
  decision->sse = total_sse;
  decision->cost = decision->sse +
                   options.lambda_bits *
                       static_cast<double>(decision->header_bits +
                                           decision->tile_bits);
  return WienerDecisionStatus::kOk;
}

void wiener_decision_to_loop_params(const WienerDecision& decision,
                                    LoopRestorationParams* params) {
  if (params == nullptr) {
    return;
  }
  *params = LoopRestorationParams{};
  for (int p = 0; p < 3; ++p) {
    params->frame_type[p] =
        static_cast<RestorationType>(decision.params.frame_type[p]);
    params->unit_size[p] = decision.unit_size[p];
    params->unit_rows[p] = decision.unit_rows[p];
    params->unit_cols[p] = decision.unit_cols[p];
    params->unit_type[p] =
        decision.unit_type[p].empty() ? nullptr : decision.unit_type[p].data();
    params->unit_coeffs[p] = decision.unit_coeffs[p].empty()
                                 ? nullptr
                                 : decision.unit_coeffs[p].data();
    params->frame_filters_on[p] =
        decision.params.frame_filters_on[p] ? 1 : 0;
    params->frame_coeffs[p] =
        decision.params.frame_filters_on[p]
            ? &decision.params.frame_coeffs[p][0][0]
            : nullptr;
  }
  params->num_filter_classes = decision.num_filter_classes;
  params->filter_class_count = decision.filter_class_count;
  params->filter_class_grid =
      decision.filter_class_grid.empty() ? nullptr
                                         : decision.filter_class_grid.data();
  params->subclass_lookup =
      decision.subclass_lookup.empty() ? nullptr
                                       : decision.subclass_lookup.data();
  params->subclass_grid =
      decision.subclass_grid.empty() ? nullptr : decision.subclass_grid.data();
}

}  // namespace av2enc::restoration::classes
