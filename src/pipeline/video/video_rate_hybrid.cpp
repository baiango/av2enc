// R09 rate model: exact inter syntax probe + prediction-error report.
// See video_rate_hybrid.h for the model and the gate contract.
#include "pipeline/video/video_rate_hybrid.h"

#include <inttypes.h>

#include "codec/entropy/av2/av2_entropy.h"
#include "codec/inter/mv_precision/mv_precision.h"
#include "codec/partition/tree/partition.h"
#include "pipeline/video/video_interp_filter.h"

namespace av2enc::video::rate_hybrid {

namespace {

constexpr std::int8_t kIntraRef = 8;  // §6.19.7.9 INTRA_FRAME
constexpr double kRawBit = 1.0;

int floor_log2_int(int value) {
  int r = 0;
  while (value > 1) {
    value >>= 1;
    ++r;
  }
  return r;
}

// §5.20.7.20 / write_ns_bits(): number of literal bits the writer emits.
int ns_bits(std::uint32_t n, std::uint32_t value) {
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  return value < m ? static_cast<int>(w - 1U) : static_cast<int>(w);
}

bool has_newmv_mode(int y_mode) { return y_mode == 16; }

}  // namespace

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

namespace report {

namespace {

struct Collector {
  std::vector<Sample> samples;
  std::uint64_t fallbacks = 0;
  std::uint64_t decisions = 0;
  std::uint64_t skip_changed = 0;
  std::uint64_t intra_changed = 0;
  std::uint64_t candidate_changed = 0;
};

Collector& collector() {
  static Collector value;
  return value;
}

double percentile(std::vector<double>& values, double q) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::min<double>(static_cast<double>(values.size() - 1),
                       q * static_cast<double>(values.size() - 1)));
  return values[index];
}

struct Stats {
  std::size_t n = 0;
  double median = 0.0;
  double p90 = 0.0;       // p90 of |error|
  double bias = 0.0;      // mean signed error (pred - actual)
  double mean_pred = 0.0;
  double mean_actual = 0.0;
};

Stats stats(const std::vector<double>& errors, double pred_sum,
            double actual_sum) {
  Stats s;
  s.n = errors.size();
  if (errors.empty()) return s;
  std::vector<double> sorted = errors;
  std::sort(sorted.begin(), sorted.end());
  s.median = sorted[sorted.size() / 2];
  std::vector<double> abs_errors(errors.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < errors.size(); ++i) {
    abs_errors[i] = std::fabs(errors[i]);
    sum += errors[i];
  }
  s.p90 = percentile(abs_errors, 0.9);
  s.bias = sum / static_cast<double>(errors.size());
  s.mean_pred = pred_sum / static_cast<double>(errors.size());
  s.mean_actual = actual_sum / static_cast<double>(errors.size());
  return s;
}

template <typename Pred, typename Actual>
void print_group(const char* label, const char* value, const char* term,
                 const std::vector<const Sample*>& group, Pred pred,
                 Actual actual) {
  if (group.empty()) return;
  std::vector<double> errors;
  errors.reserve(group.size());
  double pred_sum = 0.0;
  double actual_sum = 0.0;
  for (const Sample* s : group) {
    const double p = pred(*s);
    const double a = actual(*s);
    errors.push_back(p - a);
    pred_sum += p;
    actual_sum += a;
  }
  const Stats st = stats(errors, pred_sum, actual_sum);
  std::fprintf(stderr,
               "rate-hybrid-s: %s=%s term=%s n=%zu pred=%.3f actual=%.3f "
               "med=%+.2f p90abs=%.2f bias=%+.2f\n",
               label, value, term, st.n, st.mean_pred, st.mean_actual,
               st.median, st.p90, st.bias);
}

}  // namespace

void add(const Sample& sample) {
  Collector& c = collector();
  if (c.samples.size() >= 2000000) return;
  c.samples.push_back(sample);
}

void init() {
  static const bool once = [] {
    collector();
    std::atexit(&print);
    return true;
  }();
  (void)once;
}

void count_fallback() { ++collector().fallbacks; }

void count_decision(const char* name) {
  Collector& c = collector();
  ++c.decisions;
  if (std::strcmp(name, "skip") == 0) ++c.skip_changed;
  if (std::strcmp(name, "intra") == 0) ++c.intra_changed;
  if (std::strcmp(name, "candidate") == 0) ++c.candidate_changed;
}

void print() {
  const Collector& c = collector();
  std::fprintf(stderr,
               "rate-hybrid-s: gate=%d measure=%d syntax=%d coeff=%d "
               "leaves=%zu fallbacks=%" PRIu64
               " decisions=%" PRIu64 " (skip=%" PRIu64 " intra=%" PRIu64
               " candidate=%" PRIu64 ")\n",
               enabled() ? 1 : 0, measure_enabled() ? 1 : 0,
               syntax_enabled() ? 1 : 0, coeff_enabled() ? 1 : 0,
               c.samples.size(), c.fallbacks, c.decisions, c.skip_changed,
               c.intra_changed, c.candidate_changed);
  if (c.samples.empty()) return;

  std::vector<const Sample*> inter;
  std::vector<const Sample*> residual;
  std::vector<const Sample*> skipped;
  inter.reserve(c.samples.size());
  for (const Sample& s : c.samples) {
    if (!s.is_inter) continue;
    inter.push_back(&s);
    if (s.skip == 0) {
      residual.push_back(&s);
    } else {
      skipped.push_back(&s);
    }
  }
  // Aggregate: syntax and coefficient models against the measured coder.
  print_group("scope", "inter", "syntax_old", inter,
              [](const Sample& s) { return s.pred_syntax_old; },
              [](const Sample& s) {
                return s.act_is_inter_skip + s.act_mode_info;
              });
  print_group("scope", "inter", "syntax_new", inter,
              [](const Sample& s) { return s.pred_syntax_new; },
              [](const Sample& s) {
                return s.act_is_inter_skip + s.act_mode_info;
              });
  print_group("scope", "inter", "coeff_old", inter,
              [](const Sample& s) { return s.pred_coeff_old; },
              [](const Sample& s) { return s.act_residual; });
  print_group("scope", "inter", "coeff_new", inter,
              [](const Sample& s) { return s.pred_coeff_new; },
              [](const Sample& s) { return s.act_residual; });
  // The coefficient model only ever prices residual-coded leaves.
  print_group("scope", "residual", "coeff_old", residual,
              [](const Sample& s) { return s.pred_coeff_old; },
              [](const Sample& s) { return s.act_residual; });
  print_group("scope", "residual", "coeff_new", residual,
              [](const Sample& s) { return s.pred_coeff_new; },
              [](const Sample& s) { return s.act_residual; });
  // Decisive for "can adaptivity solve the coefficient pricing": the writer's
  // own write-time -log2(p) over the residual group, under the real adaptive
  // contexts, against the range coder.  If these agree, a context-priced
  // decision is exact and only the analytic surrogate is wrong.
  print_group("scope", "residual", "coeff_exact", residual,
              [](const Sample& s) { return s.exact[kTagResid]; },
              [](const Sample& s) { return s.act_residual; });
  print_group("scope", "skip", "syntax_new", skipped,
              [](const Sample& s) { return s.pred_syntax_new; },
              [](const Sample& s) {
                return s.act_is_inter_skip + s.act_mode_info;
              });
  if (!residual.empty()) {
    double eob = 0.0, nz = 0.0, abs_sum = 0.0, bits = 0.0;
    for (const Sample* s : residual) {
      eob += s->eob;
      nz += s->nonzero;
      abs_sum += s->sum_abs;
      bits += s->act_residual;
    }
    const double n = static_cast<double>(residual.size());
    std::fprintf(stderr,
                 "rate-hybrid-s: coeff-stats n=%.0f eob=%.2f nz=%.2f "
                 "meanabs=%.2f actual_bits=%.2f mag_b=%.2f\n",
                 n, eob / n, nz / n, nz > 0 ? abs_sum / nz : 0.0, bits / n,
                 nz > 0 ? abs_sum / nz : 0.0);
  }

  // Term split.  The decision prediction is compared with the write-time
  // -log2(p) of the same symbol group (the coder's own model cost), because
  // the range coder only exposes one aggregate per leaf.
  print_group("term", "is_inter", "probe", inter,
              [](const Sample& s) { return s.pred_term[0]; },
              [](const Sample& s) { return s.exact[kTagIsInter]; });
  print_group("term", "skip", "probe", inter,
              [](const Sample& s) { return s.pred_term[1]; },
              [](const Sample& s) { return s.exact[kTagSkip]; });
  print_group("term", "mode", "probe", inter,
              [](const Sample& s) { return s.pred_term[2]; },
              [](const Sample& s) { return s.exact[kTagSingleMode]; });
  print_group("term", "ref", "probe", inter,
              [](const Sample& s) {
                return s.pred_term[3];
              },
              [](const Sample& s) {
                return s.exact[kTagSingleRef] + s.exact[kTagCompMode];
              });
  print_group("term", "drl", "probe", inter,
              [](const Sample& s) { return s.pred_term[4]; },
              [](const Sample& s) { return s.exact[kTagDrl]; });
  print_group("term", "mvd", "probe", inter,
              [](const Sample& s) { return s.pred_term[5]; },
              [](const Sample& s) { return s.exact[kTagMvd]; });
  print_group("term", "bawp", "probe", inter,
              [](const Sample& s) { return s.pred_bawp; },
              [](const Sample& s) { return s.exact[kTagBawp]; });

  // Per mode and block size: the protocol's required breakdown.
  for (int mode = 0; mode < 3; ++mode) {
    for (int size = 0; size < 29; ++size) {
      std::vector<const Sample*> group;
      for (const Sample* s : inter) {
        if (static_cast<int>(s->inter_mode) == mode &&
            static_cast<int>(s->bsize) == size) {
          group.push_back(s);
        }
      }
      if (group.size() < 4) continue;
      std::vector<double> errors;
      errors.reserve(group.size());
      double pred_sum = 0.0;
      double actual_sum = 0.0;
      for (const Sample* s : group) {
        const double p = s->pred_syntax_new;
        const double a = s->act_is_inter_skip + s->act_mode_info;
        errors.push_back(p - a);
        pred_sum += p;
        actual_sum += a;
      }
      const Stats st = stats(errors, pred_sum, actual_sum);
      const auto bs = static_cast<partition::BlockSize>(size);
      std::fprintf(stderr,
                   "rate-hybrid-s: split mode=%d bsize=%dx%d n=%zu pred=%.3f "
                   "actual=%.3f med=%+.2f p90abs=%.2f bias=%+.2f\n",
                   mode, partition::block_width(bs),
                   partition::block_height(bs), st.n, st.mean_pred,
                   st.mean_actual, st.median, st.p90, st.bias);
    }
  }
}

}  // namespace report

// ---------------------------------------------------------------------------
// SyntaxProbe
// ---------------------------------------------------------------------------

void SyntaxProbe::reset(const ProbeConfig& config,
                        std::span<const std::uint16_t> cdf_image) {
  config_ = config;
  degraded_ = false;
  syntax::export_inter_rate_cdfs(config_.base_q_idx, cdf_image, &cdf_);
  const std::size_t cells =
      static_cast<std::size_t>(std::max(0, config_.mi_cols)) *
      static_cast<std::size_t>(std::max(0, config_.mi_rows));
  mi_.assign(cells, inter::mc::MiInfo{});
  skip_grid_.assign(cells, 0);
  ymode_grid_.assign(cells, 0);
  comp_group_grid_.assign(cells, 0);
  filter_grid_.assign(cells, 3);
}

double SyntaxProbe::cost(const std::uint16_t* cdf, int n, int symbol) const {
  if (cdf == nullptr || n < 2 || symbol < 0 || symbol >= n) return 0.0;
  // §8.2 inverse CDF: cdf[0..n-2] are boundaries, cdf[n-1] the adaptation
  // rate index, cdf[n] the use count.  P(s) = (cdf[s] - cdf[s-1]) / 32768
  // with the implicit boundaries 0 and 32768.
  const int lo = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
  const int hi = symbol == n - 1 ? static_cast<int>(av2::kCdfScale)
                                 : static_cast<int>(cdf[symbol]);
  const int freq = hi - lo;
  if (freq <= 0) return 0.0;
  return std::log2(static_cast<double>(av2::kCdfScale) /
                   static_cast<double>(freq));
}

void SyntaxProbe::adapt(std::uint16_t* cdf, int n, int symbol) {
  if (config_.disable_cdf_update) return;
  av2::adapt_cdf(cdf, n, symbol);
}

int SyntaxProbe::neighbor_positions(int pos[2][2], int r, int c, int w4,
                                    int h4) const {
  int n = 0;
  const auto add = [&](int nr, int nc) {
    if (n < 2 && nr >= 0 && nc >= 0 && nr < config_.mi_rows &&
        nc < config_.mi_cols) {
      pos[n][0] = nr;
      pos[n][1] = nc;
      ++n;
    }
  };
  add(r + h4 - 1, c - 1);
  add(r - 1, c + w4 - 1);
  add(r, c - 1);
  add(r - 1, c);
  return n;
}

int SyntaxProbe::is_inter_context(int r, int c, int w4, int h4) const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos, r, c, w4, h4);
  const auto intra_at = [&](int i) {
    return mi_[cell_index(pos[i][0], pos[i][1])].is_inter == 0;
  };
  if (n == 2) {
    return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
  }
  if (n == 1) {
    return 2 * (intra_at(0) ? 1 : 0);
  }
  return 0;
}

int SyntaxProbe::skip_context(int r, int c, int w4, int h4) const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos, r, c, w4, h4);
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    ctx += skip_grid_[cell_index(pos[i][0], pos[i][1])] != 0 ? 1 : 0;
  }
  return ctx;
}

int SyntaxProbe::count_refs(int r, int c, int w4, int h4, int ref) const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos, r, c, w4, h4);
  int count = 0;
  for (int i = 0; i < n; ++i) {
    const inter::mc::MiInfo& mi = mi_[cell_index(pos[i][0], pos[i][1])];
    if (mi.ref[0] == ref) ++count;
    if (mi.ref[1] == ref) ++count;
  }
  return count;
}

bool SyntaxProbe::ref_backward(int ref) const {
  return ref >= 0 && ref < 8 && ref < config_.num_refs &&
         config_.frame_distance[ref] < 0;
}

int SyntaxProbe::is_joint_context(int ref0, int ref1) const {
  const int d0 = std::abs(config_.frame_distance[ref0]);
  const int d1 = std::abs(config_.frame_distance[ref1]);
  const bool same_side =
      (config_.frame_distance[ref0] >= 0) ==
      (config_.frame_distance[ref1] >= 0);
  return (same_side || d0 != d1) ? 1 : 0;
}

int SyntaxProbe::comp_group_context(int r, int c, int w4, int h4, int ref0,
                                    int ref1) const {
  const int bck = std::abs(config_.frame_distance[ref1]);
  const int fwd = std::abs(config_.frame_distance[ref0]);
  const int offset = (fwd == bck) ? 1 : 0;
  int pos[2][2] = {};
  const int n = neighbor_positions(pos, r, c, w4, h4);
  int ctxs[2] = {0, 0};
  int furthest = -1;
  for (int ref = 0; ref < config_.num_refs; ++ref) {
    if (config_.frame_distance[ref] >= 0) continue;
    if (furthest < 0 ||
        config_.frame_distance[ref] > config_.frame_distance[furthest]) {
      furthest = ref;
    }
  }
  for (int i = 0; i < n && i < 2; ++i) {
    const std::size_t idx = cell_index(pos[i][0], pos[i][1]);
    const inter::mc::MiInfo& mi = mi_[idx];
    if (mi.ref[1] >= 0 && mi.ref[1] < 8) {
      ctxs[i] = comp_group_grid_[idx] ? 1 : 0;
    } else if (furthest >= 0 && mi.ref[0] == furthest) {
      ctxs[i] = 2;
    }
  }
  return ctxs[1] + ctxs[0] + (std::min(ctxs[1], ctxs[0]) > 0 ? 1 : 0) +
         offset * 6;
}

int SyntaxProbe::interp_filter_context(int r, int c, int w4, int h4, int ref0,
                                       bool is_compound) const {
  int ctx = is_compound ? 4 : 0;
  int left_type = 3;
  int above_type = 3;
  const struct {
    int row;
    int col;
  } positions[2] = {{r + h4 - 1, c - 1}, {r - 1, c + w4 - 1}};
  for (int i = 0; i < 2; ++i) {
    const int row = positions[i].row;
    const int col = positions[i].col;
    if (row < 0 || col < 0 || row >= config_.mi_rows ||
        col >= config_.mi_cols) {
      continue;
    }
    if ((r >> 4) != (row >> 4)) continue;  // NPos: same superblock row
    const std::size_t idx = cell_index(row, col);
    const inter::mc::MiInfo& mi = mi_[idx];
    if (!mi.is_inter) continue;
    if (mi.ref[0] == ref0 || mi.ref[1] == ref0) {
      const int filter = static_cast<int>(filter_grid_[idx]);
      if (i == 0) {
        left_type = filter;
      } else {
        above_type = filter;
      }
    }
  }
  if (left_type == above_type) {
    ctx += left_type;
  } else if (left_type == 3) {
    ctx += above_type;
  } else if (above_type == 3) {
    ctx += left_type;
  } else {
    ctx += 3;
  }
  return std::clamp(ctx, 0, 15);
}

int SyntaxProbe::comp_mode_context(int r, int c, int w4, int h4) const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos, r, c, w4, h4);
  const auto single = [&](int i) {
    const int ref1 = mi_[cell_index(pos[i][0], pos[i][1])].ref[1];
    return !(ref1 >= 0 && ref1 < 8);
  };
  const auto ref0 = [&](int i) {
    return static_cast<int>(mi_[cell_index(pos[i][0], pos[i][1])].ref[0]);
  };
  const auto intra = [&](int i) {
    return mi_[cell_index(pos[i][0], pos[i][1])].is_inter == 0;
  };
  if (n == 2) {
    if (single(0) && single(1)) {
      return (ref_backward(ref0(0)) ? 1 : 0) ^
             (ref_backward(ref0(1)) ? 1 : 0);
    }
    if (single(0)) return 2 + ((ref_backward(ref0(0)) || intra(0)) ? 1 : 0);
    if (single(1)) return 2 + ((ref_backward(ref0(1)) || intra(1)) ? 1 : 0);
    return 4;
  }
  if (n == 1) {
    if (single(0)) return ref_backward(ref0(0)) ? 1 : 0;
    return 3;
  }
  return 1;
}

int SyntaxProbe::new_mv_context(int r, int c, int w4, int h4) const {
  int new_mv_count = 0;
  const bool sb_border = (r & 15) == 0;
  const auto scan = [&](int dr, int dc) -> int {
    const int mr = r + dr;
    const int mc = c + dc;
    if (mr < 0 || mc < 0 || mr >= config_.mi_rows || mc >= config_.mi_cols) {
      return 0;
    }
    const std::size_t idx = cell_index(mr, mc);
    if (mi_[idx].is_inter == 0) return 0;
    if (has_newmv_mode(ymode_grid_[idx])) {
      new_mv_count = std::min(3, new_mv_count + 1);
    }
    return 1;
  };
  const int left_a = scan(h4 - 1, -1);
  const int above_a = scan(-1, w4 - 1);
  const int left_b = scan(0, -1);
  const int above_b = (w4 >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
  const int nearest = ((above_a || above_b) ? 1 : 0) +
                      ((left_a || left_b) ? 1 : 0);
  return nearest + (new_mv_count > 0 ? 2 : 0);
}

bool SyntaxProbe::walk_flags(const syntax::BlockSyntax& node, Terms* out,
                             bool commit) {
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  const int w4 = partition::block_width(size) >> 2;
  const int h4 = partition::block_height(size) >> 2;
  if (w4 <= 0 || h4 <= 0) return false;
  const int r = node.mi_row;
  const int c = node.mi_col;
  const int inter_ctx = is_inter_context(r, c, w4, h4);
  const int inter_sym = node.is_inter ? 1 : 0;
  if (out != nullptr) out->is_inter = cost(cdf_.is_inter[inter_ctx], 2, inter_sym);
  if (commit) adapt(cdf_.is_inter[inter_ctx], 2, inter_sym);
  if (!node.is_inter) return true;
  const int skip_ctx = skip_context(r, c, w4, h4);
  const int skip_sym = node.skip_flag ? 1 : 0;
  if (out != nullptr) out->skip = cost(cdf_.skip[skip_ctx], 2, skip_sym);
  if (commit) adapt(cdf_.skip[skip_ctx], 2, skip_sym);
  return true;
}

bool SyntaxProbe::walk_compound_mode_info(const syntax::BlockSyntax& node,
                                          Terms* out, bool commit) {
  // The exact model covers the signalled (0,1) compound pair with the plain
  // 1/8-pel MVD path.  Unsupported carriers fall back to the historical
  // estimate and stop the probe (the CDF state would no longer match).
  if (!config_.compound_exact || config_.num_refs != 2 ||
      config_.mv_precision != 0) {
    return false;
  }
  if (node.ref_index != 0 || node.ref_index1 != 1) return false;
  if (node.use_optflow || node.use_refinemv || node.compound_masked) {
    return false;
  }
  const bool joint = node.compound_joint != 0;
  if (!joint && node.compound_mode > 3) return false;
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  const int w4 = partition::block_width(size) >> 2;
  const int h4 = partition::block_height(size) >> 2;
  if (w4 <= 0 || h4 <= 0) return false;
  const int r = node.mi_row;
  const int c = node.mi_col;
  const int bw = w4 * 4;
  const int bh = h4 * 4;
  if (bw < 8 || bh < 8) return false;

  // §5.20.7.10 comp_mode.  This is a compound block, so the symbol is 1.
  const int comp_ctx = comp_mode_context(r, c, w4, h4);
  if (out != nullptr) out->comp_mode += cost(cdf_.comp_mode[comp_ctx], 2, 1);
  if (commit) adapt(cdf_.comp_mode[comp_ctx], 2, 1);
  // read_compound_ref(): NumTotalRefs == 2 and the (0,1) pair is implied.
  // §5.20.7.6 compound mode context from the same grid the writer uses.
  merge::NewMvContextParams nparams;
  nparams.mi_row = r;
  nparams.mi_col = c;
  nparams.bw4 = w4;
  nparams.bh4 = h4;
  nparams.sb_h4 = 16;
  nparams.mi_cols = config_.mi_cols;
  nparams.mi_rows = config_.mi_rows;
  nparams.mi = mi_.data();
  nparams.y_mode = ymode_grid_.data();
  nparams.is_compound = true;
  nparams.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
  nparams.ref_frame[1] = static_cast<std::int8_t>(node.ref_index1);
  int ctx = 0;
  if (merge::derive_new_mv_context(nparams, &ctx) != merge::Status::kOk) {
    return false;
  }
  const int jctx = is_joint_context(0, 1);
  const int jsym = joint ? 1 : 0;
  if (out != nullptr) out->mode += cost(cdf_.is_joint[jctx], 2, jsym);
  if (commit) adapt(cdf_.is_joint[jctx], 2, jsym);
  if (!joint) {
    if (out != nullptr) {
      out->mode += cost(cdf_.compound_mode_non_joint[ctx], 5,
                        node.compound_mode);
    }
    if (commit) {
      adapt(cdf_.compound_mode_non_joint[ctx], 5, node.compound_mode);
    }
  } else {
    // §5.20.7.6 jmvd_scale_mode precedes the DRL index.  The carried
    // candidates always use scale mode 0.
    if (out != nullptr) out->mvd += cost(cdf_.jmvd_scale_mode, 5, 0);
    if (commit) adapt(cdf_.jmvd_scale_mode, 5, 0);
  }
  // §5.20.7.8 read_drl_idx(): two indices for NEAR_NEARMV / NEAR_NEWMV,
  // one for NEW_NEARMV / JOINT_NEWMV, none for GLOBAL_GLOBALMV.
  const int m = config_.max_drl_bits_minus_1 + 1;
  const auto drl_one = [&](int index, int start) -> bool {
    if (index < start || index > m) return false;
    for (int idx = start; idx < m; ++idx) {
      const int mode_bit = idx < index ? 1 : 0;
      std::uint16_t* row = cdf_.drl_mode[std::min(idx, 2)][ctx];
      if (out != nullptr) out->drl += cost(row, 2, mode_bit);
      if (commit) adapt(row, 2, mode_bit);
      if (mode_bit == 0) break;
    }
    return true;
  };
  if (!joint && (node.compound_mode == 0 || node.compound_mode == 1)) {
    if (!drl_one(node.drl_index, 0)) return false;
    if (!drl_one(node.drl_index1, 0)) return false;
  } else if (joint || node.compound_mode == 2) {
    if (!drl_one(node.drl_index, 0)) return false;
  }
  // §5.20.7.13 assign_mv(): one MVD on the NEWMV list.
  if (joint || node.compound_mode == 1 || node.compound_mode == 2) {
    if (!walk_shell(node.mvd_y, node.mvd_x, out, commit)) return false;
  }
  // §5.20.7.16 read_compound_type(): comp_group_idx = 0 (AVERAGE).
  if (config_.enable_masked_compound) {
    const int cg = comp_group_context(r, c, w4, h4, 0, 1);
    if (out != nullptr) out->mode += cost(cdf_.comp_group_idx[cg], 2, 0);
    if (commit) adapt(cdf_.comp_group_idx[cg], 2, 0);
  }
  // §5.20.7.6 cwp_idx is signalled for NEAR_NEARMV and JOINT_NEWMV (scale 0).
  const bool cwp_eligible = joint || node.compound_mode == 0;
  if (config_.enable_cwp && cwp_eligible) {
    if (node.cwp_index < 0 || node.cwp_index >= 5) return false;
    for (int idx = 0; idx < 4; ++idx) {
      const int bit = idx < node.cwp_index ? 1 : 0;
      if (out != nullptr) out->mode += cost(cdf_.cwp_idx[idx], 2, bit);
      if (commit) adapt(cdf_.cwp_idx[idx], 2, bit);
      if (bit == 0) break;
    }
  } else if (node.cwp_index != 0) {
    return false;
  }
  // §5.20.7.6 / §5.20.7.17 compound interp_filter symbol.
  if (interp_block_enabled()) {
    const bool large = bw >= 8 && bh >= 8;
    const int y_mode =
        joint ? inter::mvprec::kYModeJointNewMv
              : inter::mvprec::kYModeNearNearMv + node.compound_mode;
    if (inter::mvprec::needs_interp_filter(large, y_mode,
                                           /*motion_mode=*/0)) {
      if (node.mrl_index < 0 || node.mrl_index > 2) return false;
      const int fctx =
          interp_filter_context(r, c, w4, h4, node.ref_index, true);
      if (out != nullptr) {
        out->mode += cost(cdf_.interp_filter[fctx], 3, node.mrl_index);
      }
      if (commit) adapt(cdf_.interp_filter[fctx], 3, node.mrl_index);
    } else if (node.mrl_index != 0) {
      return false;
    }
  } else if (node.mrl_index != 0) {
    return false;
  }
  return true;
}

bool SyntaxProbe::walk_mode_info(const syntax::BlockSyntax& node, Terms* out,
                                 bool include_bawp, bool commit) {
  if (!node.is_inter) return true;
  if (node.tip_mode) return false;
  if (node.compound) {
    return walk_compound_mode_info(node, out, commit);
  }
  if (node.inter_mode > syntax::kInterNewMv) return false;
  if (config_.enable_interintra) return false;
  const bool new_mv = node.inter_mode == syntax::kInterNewMv;
  if (new_mv && (node.use_amvd || node.block_mv_precision >= 0 ||
                 node.mvd_sign_derived || config_.enable_adaptive_mvd ||
                 config_.enable_flex_mvres || config_.enable_mvd_sign_derive ||
                 config_.mv_precision != 0)) {
    return false;
  }
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  const int w4 = partition::block_width(size) >> 2;
  const int h4 = partition::block_height(size) >> 2;
  const int bw = w4 * 4;
  const int bh = h4 * 4;
  const int r = node.mi_row;
  const int c = node.mi_col;
  if (node.ref_index < 0 || node.ref_index >= config_.num_refs) return false;

  // §5.20.7.10 comp_mode: coded when reference_select is on and the block is
  // large enough for compound.  This writer's single-prediction leaf codes 0.
  if (config_.num_refs > 1 && config_.reference_select && bw >= 8 &&
      bh >= 8) {
    const int ctx = comp_mode_context(r, c, w4, h4);
    if (out != nullptr) out->comp_mode = cost(cdf_.comp_mode[ctx], 2, 0);
    if (commit) adapt(cdf_.comp_mode[ctx], 2, 0);
  }

  // §5.20.7.10 read_single_ref(): one binary per reference in order.
  if (config_.num_refs > 1) {
    const int want = node.ref_index;
    bool found = false;
    for (int ref = 0; ref < config_.num_refs - 1; ++ref) {
      const int this_count = count_refs(r, c, w4, h4, ref);
      int next_count = 0;
      for (int i = ref + 1; i < config_.num_refs; ++i) {
        next_count += count_refs(r, c, w4, h4, i);
      }
      const int ctx = this_count == next_count
                          ? 1
                          : (this_count < next_count ? 0 : 2);
      const int bit = ref == want ? 1 : 0;
      if (out != nullptr) out->ref += cost(cdf_.single_ref[ctx][ref], 2, bit);
      if (commit) adapt(cdf_.single_ref[ctx][ref], 2, bit);
      if (bit) {
        found = true;
        break;
      }
    }
    if (!found && want != config_.num_refs - 1) return false;
  }

  int ctx = 0;
  if (config_.num_refs > 1) {
    merge::NewMvContextParams params;
    params.mi_row = r;
    params.mi_col = c;
    params.bw4 = w4;
    params.bh4 = h4;
    params.sb_h4 = 16;
    params.mi_cols = config_.mi_cols;
    params.mi_rows = config_.mi_rows;
    params.mi = mi_.data();
    params.y_mode = ymode_grid_.data();
    params.is_compound = false;
    params.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
    if (merge::derive_new_mv_context(params, &ctx) != merge::Status::kOk) {
      return false;
    }
  } else {
    ctx = new_mv_context(r, c, w4, h4);
  }
  const int single_mode = node.inter_mode == syntax::kInterGlobalMv
                              ? 1
                              : (node.inter_mode == syntax::kInterNearMv ? 0 : 2);
  if (out != nullptr) out->mode = cost(cdf_.single_mode[ctx], 3, single_mode);
  if (commit) adapt(cdf_.single_mode[ctx], 3, single_mode);

  // §5.20.7.6 use_bawp + §7.13.3.25 sub-flags.  The encoder codes the fitted
  // form (explicit_bawp = 0); the explicit form is not modelled.
  if (include_bawp && config_.allow_bawp && bw >= 8 && bh >= 8 &&
      node.inter_mode != syntax::kInterGlobalMv) {
    const int single_ctx = node.inter_mode == syntax::kInterNearMv
                               ? 0
                               : (node.inter_mode == syntax::kInterNewMv ? 2 : 0);
    if (node.explicit_bawp || node.explicit_bawp_scale) return false;
    const int bawp = node.use_bawp ? 1 : 0;
    if (out != nullptr) out->bawp += cost(cdf_.use_bawp, 2, bawp);
    if (commit) adapt(cdf_.use_bawp, 2, bawp);
    if (bawp) {
      if (out != nullptr) out->bawp += cost(cdf_.explicit_bawp[single_ctx], 2, 0);
      if (commit) adapt(cdf_.explicit_bawp[single_ctx], 2, 0);
      const int chroma = node.use_bawp_chroma ? 1 : 0;
      if (out != nullptr) out->bawp += cost(cdf_.use_bawp_chroma, 2, chroma);
      if (commit) adapt(cdf_.use_bawp_chroma, 2, chroma);
    }
  }

  // §5.20.7.6 read_drl_idx(): NEARMV and NEWMV both code the index chain.
  if (node.inter_mode != syntax::kInterGlobalMv) {
    const int index = new_mv ? 0 : node.drl_index;
    const int m = config_.max_drl_bits_minus_1 + 1;
    if (index < 0 || index > m) return false;
    for (int idx = 0; idx < m; ++idx) {
      const int mode_bit = idx < index ? 1 : 0;
      std::uint16_t* row = cdf_.drl_mode[std::min(idx, 2)][ctx];
      if (out != nullptr) out->drl += cost(row, 2, mode_bit);
      if (commit) adapt(row, 2, mode_bit);
      if (mode_bit == 0) break;
    }
  }

  if (new_mv && !walk_shell(node.mvd_y, node.mvd_x, out, commit)) return false;
  return true;
}

bool SyntaxProbe::walk_shell(int dy, int dx, Terms* out, bool commit) {
  const std::int32_t ay = std::abs(dy);
  const std::int32_t ax = std::abs(dx);
  const std::int32_t shell = ay + ax;
  if (shell >= (1 << 17)) return false;
  const int cls = shell == 0 ? 0 : floor_log2_int(static_cast<int>(shell));
  const std::int32_t base = cls == 0 ? 0 : (1 << cls);
  const std::int32_t off = shell - base;
  const int setsym = cls >= 8 ? 1 : 0;
  const int class_sym = cls >= 8 ? cls - 8 : cls;
  const auto add = [&](double bits) {
    if (out != nullptr) out->mvd += bits;
  };
  add(cost(cdf_.shell_set, 2, setsym));
  if (commit) adapt(cdf_.shell_set, 2, setsym);
  std::uint16_t* class_cdf = setsym ? cdf_.shell6_class1 : cdf_.shell6_class0;
  add(cost(class_cdf, 8, class_sym));
  if (commit) adapt(class_cdf, 8, class_sym);
  if (setsym && class_sym == 7) {
    add(cost(cdf_.shell_last_two, 2, cls - 15));
    if (commit) adapt(cdf_.shell_last_two, 2, cls - 15);
  }
  if (cls < 2) {
    add(cost(cdf_.offset_low[cls], 2, static_cast<int>(off)));
    if (commit) adapt(cdf_.offset_low[cls], 2, static_cast<int>(off));
  } else if (cls == 2) {
    if (off == 0) {
      add(cost(cdf_.offset_class2, 2, 0));
      if (commit) adapt(cdf_.offset_class2, 2, 0);
    } else {
      add(cost(cdf_.offset_class2, 2, 1));
      if (commit) adapt(cdf_.offset_class2, 2, 1);
      add(off == 1 ? 1.0 : 2.0);
    }
  } else {
    for (int i = 0; i < cls; ++i) {
      const int bit = static_cast<int>((off >> i) & 1);
      add(cost(cdf_.offset_other[i], 2, bit));
      if (commit) adapt(cdf_.offset_other[i], 2, bit);
    }
  }
  if (shell > 0) {
    const std::int32_t max_pair = shell >> 1;
    bool skip_col_bit = false;
    std::int32_t col = 0;
    if (ax == max_pair && ((shell & 1) == 0)) {
      skip_col_bit = true;
      col = max_pair;
    } else if (ax <= max_pair) {
      col = ax;
    } else {
      col = ay;
    }
    if (max_pair > 0) {
      const int max_bits = std::min<std::int32_t>(max_pair, 2);
      const std::int32_t prefix_col = std::min<std::int32_t>(col, 2);
      for (int i = 0; i < max_bits; ++i) {
        const int greater = prefix_col > i ? 1 : 0;
        add(cost(cdf_.col_mv_greater[i], 2, greater));
        if (commit) adapt(cdf_.col_mv_greater[i], 2, greater);
        if (greater == 0) break;
      }
      if (max_pair > 2 && prefix_col == 2) {
        add(static_cast<double>(ns_bits(
            static_cast<std::uint32_t>(max_pair - 1),
            static_cast<std::uint32_t>(col - 2))));
      }
    }
    if (!skip_col_bit) {
      const int index = ax <= max_pair ? 0 : 1;
      const int row = std::min(cls, 3);
      add(cost(cdf_.col_mv_index[row], 2, index));
      if (commit) adapt(cdf_.col_mv_index[row], 2, index);
    }
  }
  if (dy != 0) add(kRawBit);
  if (dx != 0) add(kRawBit);
  return true;
}

bool SyntaxProbe::price_mode(const syntax::BlockSyntax& node,
                             Terms* out) {
  if (out == nullptr) return false;
  *out = Terms{};
  if (degraded_ || !node.is_inter) return false;
  Terms scratch;
  if (!walk_mode_info(node, &scratch, /*include_bawp=*/false,
                      /*commit=*/false)) {
    return false;
  }
  *out = scratch;
  return true;
}

bool SyntaxProbe::price_full(const syntax::BlockSyntax& node, Terms* out) {
  if (out == nullptr) return false;
  *out = Terms{};
  if (degraded_) return false;
  Terms scratch;
  if (!walk_flags(node, &scratch, /*commit=*/false)) return false;
  if (!walk_mode_info(node, &scratch, /*include_bawp=*/true,
                      /*commit=*/false)) {
    return false;
  }
  *out = scratch;
  return true;
}

bool SyntaxProbe::price_leaf_flags(const syntax::BlockSyntax& node,
                                   double* is_inter, double* skip) {
  if (degraded_) return false;
  Terms scratch;
  if (!walk_flags(node, &scratch, /*commit=*/false)) return false;
  if (is_inter != nullptr) *is_inter = scratch.is_inter;
  if (skip != nullptr) *skip = scratch.skip;
  return true;
}

bool SyntaxProbe::skip_delta(const syntax::BlockSyntax& node,
                             double* delta) const {
  if (delta == nullptr || degraded_ || !node.is_inter) return false;
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  const int w4 = partition::block_width(size) >> 2;
  const int h4 = partition::block_height(size) >> 2;
  if (w4 <= 0 || h4 <= 0) return false;
  const int ctx = skip_context(node.mi_row, node.mi_col, w4, h4);
  *delta = cost(cdf_.skip[ctx], 2, 1) - cost(cdf_.skip[ctx], 2, 0);
  return true;
}

void SyntaxProbe::update_grids(const syntax::BlockSyntax& node) {
  const auto size = static_cast<partition::BlockSize>(node.bsize);
  const int w4 = partition::block_width(size) >> 2;
  const int h4 = partition::block_height(size) >> 2;
  const bool tip = node.is_inter && node.tip_mode != 0;
  std::uint8_t ymode = node.y_mode;
  if (node.is_inter) {
    if (tip) {
      ymode = node.tip_pred_mode ? 16 : 14;
    } else if (node.compound) {
      ymode = static_cast<std::uint8_t>(
          node.compound_joint ? 24 : 19 + node.compound_mode);
    } else if (node.inter_mode == syntax::kInterGlobalMv) {
      ymode = 15;
    } else if (node.inter_mode == syntax::kInterNearMv) {
      ymode = 14;
    } else {
      ymode = 16;
    }
  }
  const int r = node.mi_row;
  const int c = node.mi_col;
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = r + yy;
    if (rr >= config_.mi_rows) break;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = c + xx;
      if (cc >= config_.mi_cols) break;
      const std::size_t idx = cell_index(rr, cc);
      inter::mc::MiInfo& mi = mi_[idx];
      mi = inter::mc::MiInfo{};
      mi.ref[0] = node.is_inter
                      ? (tip ? merge::kTipFrame
                             : static_cast<std::int8_t>(node.ref_index))
                      : kIntraRef;
      mi.ref[1] = tip
                      ? merge::kRefNone
                      : (node.compound
                             ? static_cast<std::int8_t>(node.ref_index1)
                             : (node.is_inter && node.motion_mode == 1
                                    ? kIntraRef
                                    : merge::kRefNone));
      mi.is_inter = node.is_inter ? 1 : 0;
      mi.w4 = static_cast<std::int8_t>(w4);
      mi.h4 = static_cast<std::int8_t>(h4);
      mi.col_base = static_cast<std::int16_t>(c);
      mi.row_base = static_cast<std::int16_t>(r);
      skip_grid_[idx] = node.skip_flag ? 1 : 0;
      ymode_grid_[idx] = ymode;
      comp_group_grid_[idx] = (node.compound && node.compound_masked) ? 1 : 0;
      filter_grid_[idx] =
          node.is_inter
              ? static_cast<std::uint8_t>(interp_block_enabled()
                                              ? node.mrl_index
                                              : 3)
              : 3;
    }
  }
}

void SyntaxProbe::commit(const syntax::BlockSyntax& node) {
  if (degraded_) {
    update_grids(node);
    return;
  }
  Terms scratch;
  if (!walk_flags(node, &scratch, /*commit=*/true)) {
    degraded_ = true;
    update_grids(node);
    return;
  }
  if (node.is_inter &&
      !walk_mode_info(node, &scratch, /*include_bawp=*/true,
                      /*commit=*/true)) {
    if (std::getenv("AV2ENC_RATE_HYBRID_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "probe degrade mi=%d,%d cmpd=%d mode=%d cmode=%d cj=%d "
                   "tip=%d cwp=%d mrl=%d opt=%d ref=%d ref1=%d\n",
                   node.mi_row, node.mi_col, node.compound,
                   node.inter_mode, node.compound_mode, node.compound_joint,
                   node.tip_mode, node.cwp_index, node.mrl_index,
                   node.use_optflow, node.ref_index, node.ref_index1);
    }
    degraded_ = true;
  }
  update_grids(node);
}

}  // namespace av2enc::video::rate_hybrid
