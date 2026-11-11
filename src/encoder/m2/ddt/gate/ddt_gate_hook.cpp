#include "encoder/m2/ddt/gate/ddt_gate_hook.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "encoder/m2/ddt/ddt.h"

namespace av2enc::m2::ddt::gate {
namespace {

std::uint64_t g_blocks = 0;
std::uint64_t g_directions = 0;
std::uint64_t g_ddtx = 0;
std::uint64_t g_fddt = 0;

struct Reporter {
  ~Reporter() {
    std::fprintf(stderr,
                 "DDT_GATE blocks=%llu directions=%llu ddtx=%llu fddt=%llu\n",
                 static_cast<unsigned long long>(g_blocks),
                 static_cast<unsigned long long>(g_directions),
                 static_cast<unsigned long long>(g_ddtx),
                 static_cast<unsigned long long>(g_fddt));
  }
};
const Reporter kReporter;

}  // namespace

bool enable_inter_ddt() {
  static const bool enabled = [] {
    const char* value = std::getenv("DDT_GATE_ENABLE");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

void inverse_transform(const std::int32_t* levels, const txv::TxParams& params,
                       std::int32_t* dequant_scratch,
                       std::int16_t* residual_coded) {
  txv::dequantize(levels, params, dequant_scratch);
  Context ctx;
  ctx.enable_inter_ddt = true;
  ctx.is_inter = true;
  ctx.use_intrabc = false;
  ctx.lossless = false;
  ctx.bit_depth = 8;
  const Plan plan = make_plan(params, ctx);
  if (plan_has_ddt(params, ctx)) {
    ++g_blocks;
    if (plan.row_ddt && (params.w == 8 || params.w == 16)) {
      ++g_directions;
      (plan.row == Effective1d::kFddt ? g_fddt : g_ddtx) += 1;
    }
    if (plan.col_ddt && (params.h == 8 || params.h == 16)) {
      ++g_directions;
      (plan.col == Effective1d::kFddt ? g_fddt : g_ddtx) += 1;
    }
  }
  if (!inverse_block(dequant_scratch, params, ctx, residual_coded)) {
    std::fprintf(stderr, "DDT_GATE inverse_block failed\n");
    std::abort();
  }
}

std::uint64_t ddt_blocks() { return g_blocks; }
std::uint64_t ddt_directions() { return g_directions; }

}  // namespace av2enc::m2::ddt::gate
