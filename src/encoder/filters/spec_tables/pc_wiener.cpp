#include "encoder/filters/spec_tables/pc_wiener.h"

#include <algorithm>
#include <cstddef>

#include "encoder/filters/spec_tables/pc_wiener_tables.h"
#include "encoder/filters/spec_tables/spec_math.h"

namespace av2enc::filters::spec {
namespace {

using namespace av2enc::filters::spec_tables;

// §7.14.2 get_q(qindex, delta).
int GetQ(int qindex, int delta, int bit_depth) {
  if (qindex == 0 && delta <= 0) return kAcQlookup[0];
  const int q_clamped = Clip3(1, MaxQ(bit_depth), qindex + delta);
  return kQlookup[q_clamped];
}

// §7.20.4 get_qval_given_tskip(qindex, tskip, i).
std::int64_t GetQvalGivenTskip(int qindex, std::int64_t tskip, int i,
                               int bit_depth) {
  std::int64_t qstep = GetQ(qindex, 0, bit_depth);
  const int qstep_shift = 3 + 10;  // QUANT_TABLE_BITS + 10
  qstep = Round2(qstep, bit_depth - 8);
  const int diff_shift = qstep_shift - 8;
  const std::int64_t prod = Round2(tskip * qstep, 8);
  const std::int64_t qval =
      kPcWienerModeWeights[i][0] * (tskip << diff_shift) +
      kPcWienerModeWeights[i][1] * qstep + kPcWienerModeWeights[i][2] * prod;
  return 255 * (kPcWienerModeOffsets[i] + Round2Signed(qval, qstep_shift));
}

// §7.20.4 get_tx_skip(x, y).
int GetTxSkip(const PcWienerFrame& frame, const PcWienerConfig& cfg,
              int block_start_x, int block_end_x, int x, int y) {
  x = std::min(block_end_x, x);
  x = std::max(block_start_x, x);
  y = Clip3(frame.src.luma_stripe_start_y, frame.src.luma_stripe_end_y, y);
  const int tile_start_y = cfg.mi_row_start * 4;
  const int tile_end_y = cfg.mi_row_end * 4 - 1;
  y = Clip3(tile_start_y, tile_end_y, y);
  return cfg.lr_tx_skip[static_cast<std::ptrdiff_t>(y >> 2) *
                           cfg.lr_tx_skip_stride +
                       (x >> 2)];
}

// §7.20.4 get_features(x, y).
void GetFeatures(const PcWienerFrame& frame, const PcWienerConfig& cfg,
                 int block_start_x, int block_end_x, int x, int y,
                 std::int64_t f[kPcWienerNumFeatures], int* skip_out) {
  x = std::min(block_end_x + 2, x);
  const int m = luma_source_sample(frame.src, x, y);
  const int up = luma_source_sample(frame.src, x, y - 1);
  const int down = luma_source_sample(frame.src, x, y + 1);
  const int vert = up - 2 * m + down;
  const int upright = luma_source_sample(frame.src, x + 1, y - 1);
  const int downleft = luma_source_sample(frame.src, x - 1, y + 1);
  const int anti_diag = upright - 2 * m + downleft;
  const int downright = luma_source_sample(frame.src, x + 1, y + 1);
  const int upleft = luma_source_sample(frame.src, x - 1, y - 1);
  const int diag = upleft - 2 * m + downright;

  f[0] = 0;
  f[1] = Abs(vert);
  f[2] = Abs(anti_diag);
  f[3] = Abs(diag);
  *skip_out = GetTxSkip(frame, cfg, block_start_x, block_end_x, x, y);
}

// §7.20.4 get_box_features(x, y).
void GetBoxFeatures(const PcWienerFrame& frame, const PcWienerConfig& cfg,
                    int block_start_x, int block_end_x, int x, int y,
                    std::int64_t f[kPcWienerNumFeatures],
                    std::int64_t* tskip_out) {
  for (int i = 0; i < kPcWienerNumFeatures; ++i) f[i] = 0;
  std::int64_t s = 0;
  for (int dy = -kPcWienerLead; dy <= kPcWienerLag; ++dy) {
    for (int dx = -kPcWienerLead; dx <= kPcWienerLag; ++dx) {
      std::int64_t tf[kPcWienerNumFeatures];
      int skip = 0;
      GetFeatures(frame, cfg, block_start_x, block_end_x, x + dx, y + dy, tf,
                  &skip);
      for (int i = 0; i < kPcWienerNumFeatures; ++i) f[i] += tf[i];
      s += skip;
    }
  }
  for (int i = 0; i < kPcWienerNumFeatures; ++i) {
    f[i] = Round2(f[i] * kPcWienerNormalizer[i], cfg.bit_depth - 8);
  }
  *tskip_out = s * kPcWienerNormalizer[kPcWienerNumFeatures];
}

}  // namespace

int PcWienerClass(const PcWienerFrame& frame, const PcWienerConfig& cfg, int x,
                  int y) {
  const int block_start_x = (x >> 6) << 6;
  const int block_end_x = std::min(cfg.mi_col_end * 4 - 1, block_start_x + 63);
  std::int64_t f[kPcWienerNumFeatures];
  std::int64_t tskip = 0;
  GetBoxFeatures(frame, cfg, block_start_x, block_end_x, x, y, f, &tskip);

  int lut_input = 0;
  for (int i = 0; i < kPcWienerNumFeatures; ++i) {
    std::int64_t qval =
        Round2Signed(f[i] + GetQvalGivenTskip(cfg.base_q_idx, tskip, i,
                                              cfg.bit_depth),
                     kPcWienerPrecFeature);
    qval = qval < 0 ? 0 : (qval > 255 ? 255 : qval);
    qval >>= 5;
    lut_input += static_cast<int>(qval) << (3 * (3 - i));
  }
  return kPcWienerLutToClass[lut_input];
}

void PcWienerClassifyBlock(const PcWienerFrame& frame, const PcWienerConfig& cfg,
                           int x, int y) {
  const int cls = PcWienerClass(frame, cfg, x, y);
  frame.filter_class[static_cast<std::ptrdiff_t>(y >> 2) *
                         frame.filter_class_stride +
                     (x >> 2)] = static_cast<std::uint8_t>(cls);
}

void PcWienerFilterBlock(const PcWienerFrame& frame, const PcWienerConfig& cfg,
                         int x, int y, int w, int h) {
  const int index = PcWienerFilterSetIndex(cfg.base_q_idx);
  const int cls = PcWienerClass(frame, cfg, x, y);
  const int filt = kPcWienerSubClassify[index][cls];

  for (int r2 = 0; r2 < h; ++r2) {
    for (int c2 = 0; c2 < w; ++c2) {
      const int m = luma_source_sample(frame.src, x + c2, y + r2);
      std::int64_t s = static_cast<std::int64_t>(m) << kPcWienerPrecBits;
      for (int i = 0; i < kPcWienerTaps; ++i) {
        const std::int64_t coeff = kPcWienerFilters[index][filt][i >> 1];
        const int sample = luma_source_sample(
            frame.src, x + c2 + kPcWienerConfig[i].dx,
            y + r2 + kPcWienerConfig[i].dy);
        s += static_cast<std::int64_t>(sample) * coeff;
      }
      const int v = static_cast<int>(Round2(s, kPcWienerPrecBits));
      frame.lr[static_cast<std::ptrdiff_t>(y + r2) * frame.lr_stride +
               (x + c2)] =
          static_cast<std::uint16_t>(Clip1(cfg.bit_depth, v));
    }
  }
}

void PcWienerProcess(const PcWienerFrame& frame, const PcWienerConfig& cfg, int x,
                     int y, int w, int h, bool skip_filter) {
  if (skip_filter) {
    PcWienerClassifyBlock(frame, cfg, x, y);
  } else {
    PcWienerFilterBlock(frame, cfg, x, y, w, h);
  }
}

}  // namespace av2enc::filters::spec
