// Greedy TX partition / transform-type selection.  CPU reference only.
#include "codec/transform/variants/tx_select.h"

#include <cstddef>
#include <vector>

namespace av2enc::txv {
namespace {

constexpr TxType kSearchTypes[] = {
    TxType::kDctDct,       TxType::kAdstAdst,  TxType::kDctAdst,
    TxType::kAdstDct,      TxType::kFlipadstFlipadst, TxType::kIdtx,
};

bool run_block(const std::int16_t* residual_coded, int block_w, int block_h,
               int x0, int y0, const TxParams& p, double* sse,
               std::int32_t* nonzero) {
  if (p.coded_w <= 0 || p.coded_h <= 0 || x0 < 0 || y0 < 0 ||
      x0 + p.coded_w > block_w || y0 + p.coded_h > block_h) {
    return false;
  }
  const int coded_n = p.coded_w * p.coded_h;
  const int adj_n = p.w * p.h;

  std::vector<std::int16_t> sub(static_cast<std::size_t>(coded_n));
  for (int y = 0; y < p.coded_h; ++y) {
    const std::int16_t* src = residual_coded +
                             static_cast<std::size_t>((y0 + y) * block_w + x0);
    std::int16_t* dst = sub.data() +
                        static_cast<std::size_t>(y) * static_cast<std::size_t>(p.coded_w);
    for (int x = 0; x < p.coded_w; ++x) {
      dst[x] = src[x];
    }
  }

  std::vector<std::int16_t> adj(static_cast<std::size_t>(adj_n));
  downsample_residual(sub.data(), p, adj.data());

  std::vector<double> coeff(static_cast<std::size_t>(adj_n));
  std::vector<std::int32_t> levels(static_cast<std::size_t>(adj_n));
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(adj_n));
  std::vector<std::int16_t> recon(static_cast<std::size_t>(coded_n));

  forward_transform(adj.data(), p, coeff.data(), levels.data());
  inverse_transform(levels.data(), p, dequant.data(), recon.data());

  std::int64_t count = 0;
  const std::int32_t* lv = levels.data();
  for (int i = 0; i < adj_n; ++i) {
    if (lv[i] != 0) {
      ++count;
    }
  }
  std::int64_t error = 0;
  const std::int16_t* rc = recon.data();
  const std::int16_t* sb = sub.data();
  for (int i = 0; i < coded_n; ++i) {
    const std::int64_t d = static_cast<std::int64_t>(rc[i]) - sb[i];
    error += d * d;
  }
  *sse = static_cast<double>(error);
  *nonzero = static_cast<std::int32_t>(count);
  return true;
}

struct NodeResult {
  double cost = 0.0;
  double sse = 0.0;
  std::int32_t nonzero = 0;
  int leaf_count = 0;
  TxPartition root_partition = TxPartition::kNone;
  TxLeaf leaves[16];
};

double best_type_for_node(const std::int16_t* residual_coded, int block_w,
                          int block_h, int x0, int y0, int coded_w,
                          int coded_h, const TxSelectOptions& options,
                          double* out_sse, std::int32_t* out_nonzero,
                          TxParams* out_params) {
  double best = 0.0;
  bool have = false;
  TxParams best_p;
  double best_sse = 0.0;
  std::int32_t best_nz = 0;

  auto try_type = [&](TxType type) {
    const auto params = make_tx_params(coded_w, coded_h, type);
    if (!params.has_value()) {
      return;
    }
    double sse = 0.0;
    std::int32_t nz = 0;
    if (!run_block(residual_coded, block_w, block_h, x0, y0, *params, &sse,
                   &nz)) {
      return;
    }
    const double total = sse + options.lambda * static_cast<double>(nz);
    if (!have || total < best) {
      have = true;
      best = total;
      best_sse = sse;
      best_nz = nz;
      best_p = *params;
    }
  };

  try_type(options.tx_type);
  if (options.search_tx_type) {
    for (TxType type : kSearchTypes) {
      if (type != options.tx_type) {
        try_type(type);
      }
    }
  }

  if (!have) {
    return -1.0;
  }
  *out_sse = best_sse;
  *out_nonzero = best_nz;
  *out_params = best_p;
  return best;
}

bool evaluate_node(const std::int16_t* residual_coded, int root_w,
                   int root_h, int x0, int y0, int coded_w, int coded_h,
                   int depth_left, const TxSelectOptions& options,
                   NodeResult* result) {
  TxParams best_p;
  double sse = 0.0;
  std::int32_t nz = 0;
  const double none_cost = best_type_for_node(
      residual_coded, root_w, root_h, x0, y0, coded_w, coded_h, options, &sse,
      &nz, &best_p);
  if (none_cost < 0.0) {
    return false;
  }

  result->cost = none_cost;
  result->sse = sse;
  result->nonzero = nz;
  result->leaf_count = 1;
  result->leaves[0].x = static_cast<std::int16_t>(x0);
  result->leaves[0].y = static_cast<std::int16_t>(y0);
  result->leaves[0].params = best_p;

  if (depth_left <= 0) {
    return true;
  }

  static constexpr TxPartition kParts[3] = {
      TxPartition::kSplit, TxPartition::kHorz, TxPartition::kVert};
  for (TxPartition part : kParts) {
    int xs[4];
    int ys[4];
    int ws[4];
    int hs[4];
    const int count =
        tx_partition_geometry(coded_w, coded_h, part, xs, ys, ws, hs);
    if (count <= 0) {
      continue;
    }
    NodeResult trial;
    trial.cost = 0.0;
    trial.sse = 0.0;
    trial.nonzero = 0;
    trial.leaf_count = 0;
    bool ok = true;
    for (int k = 0; k < count; ++k) {
      NodeResult child;
      if (!evaluate_node(residual_coded, root_w, root_h, x0 + xs[k],
                         y0 + ys[k], ws[k], hs[k], depth_left - 1, options,
                         &child)) {
        ok = false;
        break;
      }
      trial.cost += child.cost;
      trial.sse += child.sse;
      trial.nonzero += child.nonzero;
      for (int l = 0; l < child.leaf_count; ++l) {
        if (trial.leaf_count < 16) {
          trial.leaves[trial.leaf_count++] = child.leaves[l];
        }
      }
    }
    if (!ok) {
      continue;
    }
    if (trial.cost < result->cost) {
      trial.root_partition = part;
      *result = trial;
    }
  }
  return true;
}

}  // namespace

bool tx_block_cost(const std::int16_t* residual_coded, int block_w,
                   int block_h, int x0, int y0, const TxParams& params,
                   double* cost, double* sse, std::int32_t* nonzero) {
  if (residual_coded == nullptr || cost == nullptr || sse == nullptr ||
      nonzero == nullptr) {
    return false;
  }
  double block_sse = 0.0;
  std::int32_t block_nz = 0;
  if (!run_block(residual_coded, block_w, block_h, x0, y0, params, &block_sse,
                 &block_nz)) {
    return false;
  }
  *sse = block_sse;
  *nonzero = block_nz;
  *cost = block_sse;
  return true;
}

bool select_tx_partition(const std::int16_t* residual_coded, int block_w,
                         int block_h, const TxSelectOptions& options,
                         TxSelection* out) {
  if (residual_coded == nullptr || out == nullptr) {
    return false;
  }
  if (!tx_size_supported(block_w, block_h)) {
    return false;
  }
  if (options.max_depth < 0 || options.max_depth > 2) {
    return false;
  }
  NodeResult result;
  if (!evaluate_node(residual_coded, block_w, block_h, 0, 0, block_w, block_h,
                     options.max_depth, options, &result)) {
    return false;
  }
  out->count = result.leaf_count;
  for (int i = 0; i < result.leaf_count; ++i) {
    out->leaves[i] = result.leaves[i];
  }
  out->cost = result.cost;
  out->sse = result.sse;
  out->nonzero = result.nonzero;
  out->root_partition = result.root_partition;
  return true;
}

}  // namespace av2enc::txv
