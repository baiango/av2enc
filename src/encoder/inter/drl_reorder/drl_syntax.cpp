#include "encoder/inter/drl_reorder/drl_syntax.h"

#include <algorithm>

#include "encoder/inter/drl_reorder/drl_reorder.h"

namespace av2enc::inter::drl {
namespace {

// Default_Drl_Mode_Cdf [3][5][3] from §9.3 (spec/tables/default_cdf.json).
constexpr std::uint16_t kDefaultDrlMode[kDrlModeRows][kDrlNewMvContexts][3] = {
    {{15721, 90, 0}, {21115, 0, 0}, {19567, 1, 0}, {17602, 93, 0},
     {13319, 93, 0}},
    {{18692, 93, 0}, {19343, 90, 0}, {18207, 90, 0}, {17908, 93, 0},
     {18304, 93, 0}},
    {{22157, 90, 0}, {23233, 90, 0}, {22782, 90, 0}, {22353, 93, 0},
     {22457, 93, 0}},
};

// Default_Skip_Drl_Mode_Cdf [3][3].
constexpr std::uint16_t kDefaultSkipDrlMode[kDrlModeRows][3] = {
    {21634, 0, 0}, {17376, 0, 0}, {18432, 75, 0}};

// Default_Tip_Drl_Mode_Cdf [3][3].
constexpr std::uint16_t kDefaultTipDrlMode[kDrlModeRows][3] = {
    {30662, 0, 0}, {23823, 6, 0}, {21676, 6, 0}};

std::uint16_t* select_cdf(DrlCdfState* cdfs, const DrlSymbolContext& ctx,
                          int idx) {
  const int row = drl_cdf_row(idx);
  switch (ctx.source) {
    case DrlCdfSource::kInter: {
      const int mvc = std::min(std::max(ctx.new_mv_context, 0),
                               kDrlNewMvContexts - 1);
      return &cdfs->drl_mode[row][mvc][0];
    }
    case DrlCdfSource::kSkipMode:
      return &cdfs->skip_drl_mode[row][0];
    case DrlCdfSource::kTip:
      return &cdfs->tip_drl_mode[row][0];
  }
  return nullptr;
}

}  // namespace

const char* status_name(Status status) {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kInvalidArgument:
      return "invalid argument";
    case Status::kInvalidIndex:
      return "invalid DRL index";
    case Status::kCoderError:
      return "entropy coder error";
  }
  return "unknown";
}

void init_default_drl_cdfs(DrlCdfState* state) {
  if (state == nullptr) {
    return;
  }
  std::copy(&kDefaultDrlMode[0][0][0], &kDefaultDrlMode[0][0][0] +
                                          sizeof(kDefaultDrlMode) /
                                              sizeof(std::uint16_t),
            &state->drl_mode[0][0][0]);
  std::copy(&kDefaultSkipDrlMode[0][0],
            &kDefaultSkipDrlMode[0][0] +
                sizeof(kDefaultSkipDrlMode) / sizeof(std::uint16_t),
            &state->skip_drl_mode[0][0]);
  std::copy(&kDefaultTipDrlMode[0][0],
            &kDefaultTipDrlMode[0][0] +
                sizeof(kDefaultTipDrlMode) / sizeof(std::uint16_t),
            &state->tip_drl_mode[0][0]);
}

int drl_cdf_row(int idx) { return std::min(std::max(idx, 0), 2); }

int drl_symbol_count(int index, int start, int m) {
  if (m < 0 || start < 0 || start > m || index < start || index > m) {
    return 0;
  }
  // Symbols are emitted for idx = start .. index-1 (all 1) plus the
  // terminating 0 when index < m.
  return (index - start) + (index < m ? 1 : 0);
}

Status write_drl_index(av2::Encoder* enc, DrlCdfState* cdfs,
                       const DrlSymbolContext& ctx, int index, int start,
                       int m) {
  if (enc == nullptr || cdfs == nullptr) {
    return Status::kInvalidArgument;
  }
  if (m < 0 || m > kMaxDrlIndex || start < 0 || start > m || index < start ||
      index > m) {
    return Status::kInvalidIndex;
  }
  for (int idx = start; idx < m; ++idx) {
    const int mode = idx < index ? 1 : 0;
    std::uint16_t* cdf = select_cdf(cdfs, ctx, idx);
    if (cdf == nullptr) {
      return Status::kInvalidArgument;
    }
    const av2::Status status = enc->encode_symbol(cdf, 2, mode);
    if (status != av2::Status::kOk) {
      return Status::kCoderError;
    }
    if (mode == 0) {
      // Defensive: the loop would stop here in read_drl_idx().  index > idx
      // cannot happen for a valid index, so this is unreachable.
      return index == idx ? Status::kOk : Status::kInvalidIndex;
    }
  }
  return Status::kOk;
}

Status read_drl_index(av2::Decoder* dec, DrlCdfState* cdfs,
                      const DrlSymbolContext& ctx, int start, int m,
                      int* index) {
  if (dec == nullptr || cdfs == nullptr || index == nullptr) {
    return Status::kInvalidArgument;
  }
  if (dec->in == nullptr && dec->payload_bytes == 0) {
    return Status::kInvalidArgument;  // init_symbol() was never called
  }
  if (m < 0 || m > kMaxDrlIndex || start < 0 || start > m) {
    return Status::kInvalidIndex;
  }
  for (int idx = start; idx < m; ++idx) {
    std::uint16_t* cdf = select_cdf(cdfs, ctx, idx);
    if (cdf == nullptr) {
      return Status::kInvalidArgument;
    }
    int mode = 0;
    const av2::Status status = dec->decode_symbol(cdf, 2, &mode);
    if (status != av2::Status::kOk) {
      return Status::kCoderError;
    }
    if (mode == 0) {
      *index = idx;
      return Status::kOk;
    }
  }
  *index = m;
  return Status::kOk;
}

}  // namespace av2enc::inter::drl
