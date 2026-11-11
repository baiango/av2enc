// Deterministic tests for the M2 post layer: QP selection, tile geometry and
// splitting, 10-bit checklist, filter order, and byte-exact header parity with
// the untouched M2 core for the one-tile/filters-off case.
//
// Build/run with ./build_and_test.sh.  This test does not use CUDA or CMake.
#include "encoder/m2/post/m2_post.h"

#include "encoder/m2/m2_headers.h"
#include "encoder/m2/post/bit_depth_checklist.h"
#include "encoder/m2/post/loop_filters.h"
#include "encoder/m2/post/rc_adapter.h"
#include "encoder/m2/post/tiled_frame.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++g_checks;                                                              \
    const auto va = (a);                                                     \
    const auto vb = (b);                                                     \
    if (!(va == vb)) {                                                       \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s (%lld) != %s (%lld)\n", __FILE__, \
                   __LINE__, #a, static_cast<long long>(va), #b,             \
                   static_cast<long long>(vb));                              \
    }                                                                        \
  } while (0)

av2enc::rc::FrameComplexity make_complexity(int w, int h, double mad,
                                            double mse) {
  av2enc::rc::FrameComplexity c;
  c.width = w;
  c.height = h;
  c.residual_mad = mad;
  c.residual_mse = mse;
  return c;
}

void test_rate_control() {
  using namespace av2enc::m2::post;
  using av2enc::rc::RateControlMode;

  M2RcConfig cfg;
  cfg.depth = av2enc::fmt::BitDepth::k12;
  std::string error;
  CHECK(!m2_rc_config_valid(cfg, &error));

  cfg = M2RcConfig{};
  cfg.depth = av2enc::fmt::BitDepth::k10;
  cfg.rc.mode = RateControlMode::kCqp;
  cfg.rc.cqp.base_q_idx = 90;
  CHECK(m2_rc_config_valid(cfg, &error));

  cfg = M2RcConfig{};
  cfg.rc.mode = RateControlMode::kCqp;
  cfg.rc.cqp.base_q_idx = 90;
  M2RateControl rc;
  CHECK(rc.configure(cfg, &error));
  M2FrameQp qp;
  CHECK(rc.begin_frame(make_complexity(64, 64, 4.0, 16.0), qp, &error));
  CHECK_EQ(qp.base_q_idx, 90);
  CHECK_EQ(qp.coeff_cdf_qctx, 0);
  CHECK_EQ(qp.qp_slice, 0);
  CHECK_EQ(qp.base_q_idx_bits, 8);
  CHECK_EQ(qp.max_quantizer, 255);
  CHECK(!qp.clamped_for_m2);
  CHECK(qp.lambda_sse > 0.0);
  CHECK(rc.end_frame(1000, &error));
  CHECK(rc.begin_frame(make_complexity(64, 64, 4.0, 16.0), qp, &error));
  CHECK(rc.end_frame(1000, &error));

  // Above the shipped coefficient-CDF slice: clamped to 90, flagged.
  cfg = M2RcConfig{};
  cfg.rc.mode = RateControlMode::kCqp;
  cfg.rc.cqp.base_q_idx = 130;
  M2RateControl clamp_rc;
  CHECK(clamp_rc.configure(cfg, &error));
  CHECK(clamp_rc.begin_frame(make_complexity(64, 64, 4.0, 16.0), qp, &error));
  CHECK_EQ(qp.base_q_idx, 90);
  CHECK(qp.clamped_for_m2);
  CHECK(clamp_rc.end_frame(1000, &error));

  // With no clamping, the same decision is rejected.
  cfg.policy.clamp_to_policy = false;
  M2RateControl strict_rc;
  CHECK(strict_rc.configure(cfg, &error));
  CHECK(!strict_rc.begin_frame(make_complexity(64, 64, 4.0, 16.0), qp,
                               &error));
  CHECK(error.find("outside") != std::string::npos);

  // 10-bit header fields: f(9), MaxQ 303.
  cfg = M2RcConfig{};
  cfg.depth = av2enc::fmt::BitDepth::k10;
  cfg.rc.mode = RateControlMode::kCqp;
  cfg.rc.cqp.base_q_idx = 90;
  M2RateControl hbd_rc;
  CHECK(hbd_rc.configure(cfg, &error));
  CHECK(hbd_rc.begin_frame(make_complexity(64, 64, 4.0, 16.0), qp, &error));
  CHECK_EQ(qp.base_q_idx_bits, 9);
  CHECK_EQ(qp.max_quantizer, 303);
  CHECK(hbd_rc.end_frame(1000, &error));

  // CRF is monotone: a larger crf cannot lower qindex.
  int previous = 1;
  for (double crf = 0.0; crf <= 63.0; crf += 7.0) {
    M2RcConfig c;
    c.rc.mode = RateControlMode::kCrf;
    c.rc.crf.crf = crf;
    c.policy.max_base_q_idx = 90;
    M2RateControl r;
    CHECK(r.configure(c, &error));
    M2FrameQp d;
    CHECK(r.begin_frame(make_complexity(64, 64, 4.0, 16.0), d, &error));
    CHECK(d.base_q_idx >= previous);
    previous = d.base_q_idx;
    CHECK(r.end_frame(1000, &error));
  }

  // VBR: a higher target bitrate selects a no-larger qindex, and the result is
  // deterministic across two independent controllers.
  const auto run_vbr = [&](double bps, int frames, std::vector<int>* out) {
    M2RcConfig c;
    c.rc.mode = RateControlMode::kVbr;
    c.rc.vbr.target_bitrate_bps = bps;
    c.rc.vbr.frame_rate_num = 30;
    c.rc.vbr.max_qindex_step = 8;
    M2RateControl r;
    std::string e;
    if (!r.configure(c, &e)) return false;
    for (int i = 0; i < frames; ++i) {
      M2FrameQp d;
      if (!r.begin_frame(make_complexity(64, 64, 8.0, 64.0), d, &e)) {
        return false;
      }
      if (out != nullptr) out->push_back(d.base_q_idx);
      if (!r.end_frame(600, &e)) return false;
    }
    return true;
  };
  std::vector<int> low;
  std::vector<int> high;
  CHECK(run_vbr(100000.0, 6, &low));
  CHECK(run_vbr(2000000.0, 6, &high));
  CHECK(!low.empty() && !high.empty());
  for (std::size_t i = 0; i < std::min(low.size(), high.size()); ++i) {
    CHECK(low[i] >= high[i]);
  }
  std::vector<int> again;
  CHECK(run_vbr(100000.0, 6, &again));
  CHECK(low == again);

  // Slice boundaries are the §8.3.1 thresholds; the M2 adapter only ships 0.
  CHECK_EQ(m2_qp_slice(1), 0);
  CHECK_EQ(m2_qp_slice(90), 0);
  CHECK_EQ(m2_qp_slice(91), 1);
  CHECK_EQ(m2_qp_slice(140), 1);
  CHECK_EQ(m2_qp_slice(141), 2);
  CHECK_EQ(m2_qp_slice(191), 3);
  CHECK(m2_coeff_cdf_slice_available(0));
  CHECK(!m2_coeff_cdf_slice_available(1));
}

av2enc::m2::syntax::BlockSyntax make_root(int mi_row, int mi_col,
                                         bool inter) {
  av2enc::m2::syntax::BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = 12;  // BLOCK_64X64
  node.partition = 0;
  node.allowed_mask = 1U << 0;  // PARTITION_NONE
  node.flags = av2enc::m2::syntax::kFlagSbRoot;
  node.subtree = 0;
  node.is_inter = inter ? 1 : 0;
  node.skip_flag = inter ? 1 : 0;
  node.inter_mode = av2enc::m2::syntax::kInterGlobalMv;
  node.y_mode = 0;
  node.uv_mode = 0;
  node.tx_size[0] = 4;  // 64x64
  node.tx_size[1] = 3;  // 32x32 chroma
  node.tx_size[2] = 3;
  node.eob[0] = node.eob[1] = node.eob[2] = 0;
  node.levels_offset = 0;
  return node;
}

void test_tiles_and_headers() {
  using namespace av2enc::m2::post;

  M2TilePlan plan = plan_m2_tiles(128, 64, 2, 1);
  CHECK(plan.ok);
  CHECK_EQ(plan.layout.tile_count(), 2);
  CHECK_EQ(plan.layout.cols_log2, 1);
  CHECK_EQ(plan.layout.tiles[0].x0, 0);
  CHECK_EQ(plan.layout.tiles[0].x1, 64);
  CHECK_EQ(plan.layout.tiles[1].x0, 64);
  CHECK_EQ(plan.layout.tiles[1].x1, 128);

  plan = plan_m2_tiles(256, 128, 2, 2);
  CHECK(plan.ok);
  CHECK_EQ(plan.layout.tile_count(), 4);
  CHECK_EQ(plan.layout.cols_log2, 1);
  CHECK_EQ(plan.layout.rows_log2, 1);

  CHECK(!plan_m2_tiles(64, 64, 2, 1).ok);   // too small for 2 tiles
  CHECK(!plan_m2_tiles(130, 64, 1, 1).ok);  // not divisible by 8
  CHECK(!plan_m2_tiles(128, 64, 3, 1).ok);  // non-power-of-two grid

  const auto layout_result = plan_m2_tiles(128, 64, 2, 1);
  CHECK(layout_result.ok);
  const auto& layout = layout_result.layout;

  std::vector<av2enc::m2::syntax::BlockSyntax> blocks;
  blocks.push_back(make_root(0, 0, false));
  blocks.push_back(make_root(0, 16, false));
  M2TileBlocks tile0;
  M2TileBlocks tile1;
  std::string error;
  CHECK(split_m2_blocks_for_tile(layout, 0, blocks, &tile0, &error));
  CHECK(split_m2_blocks_for_tile(layout, 1, blocks, &tile1, &error));
  CHECK_EQ(tile0.width, 64);
  CHECK_EQ(tile0.height, 64);
  CHECK_EQ(tile1.width, 64);
  CHECK_EQ(tile0.blocks.size(), 1U);
  CHECK_EQ(tile1.blocks.size(), 1U);
  CHECK_EQ(tile0.blocks[0].mi_col, 0);
  CHECK_EQ(tile1.blocks[0].mi_col, 0);

  // Byte-exact parity with the untouched M2 core for the one-tile, filters-off
  // path: sequence header and frame OBU.
  av2enc::m2::M2Config config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.base_q_idx = 90;
  M2PostSequenceOptions sequence;
  const auto ours_seq = make_m2_sequence_header_obu(config, sequence, &error);
  const auto core_seq = av2enc::m2::make_sequence_header_obu(config);
  CHECK(!ours_seq.empty());
  CHECK(ours_seq == core_seq);

  const auto one = plan_m2_tiles(64, 64, 1, 1);
  CHECK(one.ok);
  std::vector<av2enc::m2::syntax::BlockSyntax> one_blocks;
  one_blocks.push_back(make_root(0, 0, false));
  std::vector<std::int32_t> levels;
  M2TiledFrame ours;
  M2TiledFrameHeader header;
  header.keyframe = true;
  header.order_hint = 0;
  CHECK(assemble_m2_tiled_frame(config, sequence, header, one.layout,
                                one_blocks, levels, &ours, &error));

  av2enc::m2::M2FrameHeader core_header;
  core_header.keyframe = true;
  core_header.order_hint = 0;
  core_header.ref_slot = 0;
  core_header.refresh_frame_flags = 1;
  const auto core_obu =
      av2enc::m2::make_frame_obu(config, core_header, ours.tile_payloads[0]);
  CHECK(ours.obu == core_obu);

  // Multi-tile assembly: two tiles, raster order, TileSizeBytes present.
  const auto two = plan_m2_tiles(128, 64, 2, 1);
  CHECK(two.ok);
  std::vector<av2enc::m2::syntax::BlockSyntax> two_blocks;
  two_blocks.push_back(make_root(0, 0, false));
  two_blocks.push_back(make_root(0, 16, false));
  M2TiledFrame two_frame;
  av2enc::m2::M2Config two_config = config;
  two_config.max_frame_width = 128;
  two_config.max_frame_height = 64;
  CHECK(assemble_m2_tiled_frame(two_config, sequence, header, two.layout,
                                two_blocks, levels, &two_frame, &error));
  CHECK_EQ(two_frame.tile_payloads.size(), 2U);
  CHECK_EQ(two_frame.tile_bytes.size(), 2U);
  CHECK(two_frame.tile_size_bytes >= 1 && two_frame.tile_size_bytes <= 4);
  CHECK(!two_frame.obu.empty());
  CHECK(two_frame.frame_header_bits > 0);

  // tile_info() bit counts: one tile is just uniform_tile_spacing_flag; 2x1
  // adds one increment bit plus context_update_tile_id and
  // tile_size_bytes_minus_1.
  {
    av2enc::BitWriter w1;
    CHECK(write_m2_tile_info(w1, one.layout, 1, &error));
    CHECK_EQ(w1.bit_size(), 1U);
    av2enc::BitWriter w2;
    CHECK(write_m2_tile_info(w2, two.layout, 1, &error));
    CHECK_EQ(w2.bit_size(), 5U);
  }

  // The payload escape hatch selects a wider TileSizeBytes from the largest
  // tile and does not depend on the M2 syntax writer.
  {
    const std::vector<std::uint8_t> big(300, 0x5A);
    const std::vector<std::uint8_t> small(10, 0x33);
    std::vector<std::vector<std::uint8_t>> payloads{big, small};
    M2TiledFrame wide;
    CHECK(assemble_m2_tile_group_from_payloads(two_config, sequence, header,
                                               two.layout, payloads, &wide,
                                               &error));
    CHECK_EQ(wide.tile_size_bytes, 2);
    CHECK_EQ(wide.largest_tile_bytes, 300U);
    CHECK_EQ(wide.tile_bytes[0], 300U);
  }

  // Adopting a verified pipeline plan copies the layout and frame qindex, and
  // rejects per-tile QP that this header writer cannot signal.
  {
    av2enc::pipeline::FramePipelinePlan pipeline_plan;
    pipeline_plan.valid = true;
    pipeline_plan.layout = two.layout;
    pipeline_plan.base_q_idx = 88;
    pipeline_plan.tiles.resize(2);
    pipeline_plan.tiles[0].qindex = 88;
    pipeline_plan.tiles[1].qindex = 88;
    M2TiledFrameHeader adopted_header;
    M2TilePlan adopted_tiles;
    CHECK(adopt_pipeline_frame_plan(pipeline_plan, &adopted_header,
                                    &adopted_tiles, &error));
    CHECK_EQ(adopted_header.base_q_idx, 88);
    CHECK_EQ(adopted_tiles.layout.tile_count(), 2);
    pipeline_plan.tiles[1].qindex = 90;
    CHECK(!adopt_pipeline_frame_plan(pipeline_plan, &adopted_header,
                                     &adopted_tiles, &error));
  }

  // Loop-filter header rows: CDEF and CCSO flags are emitted, and a frame
  // request that the sequence header disabled is rejected rather than dropped.
  {
    M2PostSequenceOptions cdef_seq;
    cdef_seq.filters.enable_cdef = true;
    CHECK(!make_m2_sequence_header_obu(config, cdef_seq, &error).empty());
    M2TiledFrameHeader cdef_header;
    cdef_header.keyframe = true;
    cdef_header.cdef_frame_enable = true;
    cdef_header.cdef.frame_enable = true;
    cdef_header.cdef.damping_minus_3 = 1;
    cdef_header.cdef.num_strengths = 1;
    cdef_header.cdef.strengths_minus_1 = 0;
    cdef_header.cdef.set[0].y_pri = 4;
    cdef_header.cdef.set[0].y_sec = 2;
    cdef_header.cdef.set[0].uv_pri = 2;
    cdef_header.cdef.set[0].uv_sec = 1;
    av2enc::BitWriter cdef_writer;
    std::size_t cdef_bits = 0;
    CHECK(write_m2_frame_header_syntax(cdef_writer, config, cdef_seq,
                                       cdef_header, one.layout, 1, &cdef_bits,
                                       &error));
    CHECK(cdef_bits > 0);
    M2TiledFrameHeader bad_cdef = cdef_header;
    CHECK(!write_m2_frame_header_syntax(cdef_writer, config, sequence,
                                        bad_cdef, one.layout, 1, &cdef_bits,
                                        &error));

    M2PostSequenceOptions ccso_seq;
    ccso_seq.filters.enable_ccso = true;
    ccso_seq.filters.ccso_unit_matches_sb_size = true;
    CHECK(!make_m2_sequence_header_obu(config, ccso_seq, &error).empty());
    M2TiledFrameHeader ccso_header;
    ccso_header.keyframe = false;
    ccso_header.order_hint = 1;
    ccso_header.ccso_frame_flag = true;
    av2enc::BitWriter ccso_writer;
    std::size_t ccso_bits = 0;
    CHECK(write_m2_frame_header_syntax(ccso_writer, config, ccso_seq,
                                       ccso_header, one.layout, 1, &ccso_bits,
                                       &error));
    CHECK(ccso_bits > 0);

    const bool apply_all[4] = {true, true, true, true};
    const bool apply_none[4] = {false, false, false, false};
    const bool df_none[4] = {false, false, false, false};
    av2enc::BitWriter dw;
    CHECK_EQ(write_m2_deblock_params(dw, apply_none, df_none, &error), 2U);
    av2enc::BitWriter dw2;
    CHECK_EQ(write_m2_deblock_params(dw2, apply_all, df_none, &error), 8U);
  }

  // A layout whose level-derived tile limits do not match the sequence level
  // would emit a different number of increment bits; reject it instead of
  // desynchronising.
  {
    av2enc::m2::M2Config level_config;
    level_config.max_frame_width = 256;
    level_config.max_frame_height = 128;
    level_config.base_q_idx = 90;
    const auto planned = plan_m2_tiles(256, 128, 2, 2);
    CHECK(planned.ok);
    M2TilePlan wrong_level = planned;
    wrong_level.layout.min_log2_tile_cols += 1;
    av2enc::BitWriter w;
    std::size_t bits = 0;
    CHECK(!write_m2_frame_header_syntax(w, level_config, sequence, header,
                                        wrong_level.layout, 1, &bits,
                                        &error));
    CHECK(error.find("level") != std::string::npos);
  }

  // End-to-end with the verified pipeline planner: its layout and frame QP
  // are accepted by the M2 header writer.
  {
    av2enc::pipeline::PipelineConfig pc;
    pc.width = 256;
    pc.height = 128;
    pc.seq_level_idx = 7;
    pc.layout.mode = av2enc::pipeline::LayoutPolicyMode::kExplicit;
    pc.layout.explicit_preset = av2enc::tiles::TileLayoutPreset::k2x2;
    pc.tile_qp.mode = av2enc::pipeline::TileQpMode::kUniform;
    pc.rate_control.mode = av2enc::rc::RateControlMode::kCqp;
    pc.rate_control.cqp.base_q_idx = 90;
    pc.min_base_q_idx = 90;
    pc.max_base_q_idx = 90;
    pc.gpu.gpu_count = 1;
    pc.device.total_bytes = 8ull << 30;
    pc.device.free_bytes = 7ull << 30;
    av2enc::pipeline::PipelinePlanner planner;
    CHECK(planner.configure(pc, &error));
    av2enc::pipeline::FrameStats stats;
    stats.complexity.width = 256;
    stats.complexity.height = 128;
    stats.complexity.residual_mad = 4.0;
    stats.complexity.residual_mse = 16.0;
    av2enc::pipeline::FramePipelinePlan frame_plan;
    CHECK(planner.plan_frame(stats, frame_plan, &error));
    M2TiledFrameHeader pipeline_header;
    M2TilePlan pipeline_tiles;
    CHECK(adopt_pipeline_frame_plan(frame_plan, &pipeline_header,
                                    &pipeline_tiles, &error));
    CHECK_EQ(pipeline_tiles.layout.tile_count(), 4);
    CHECK_EQ(pipeline_header.base_q_idx, 90);
    CHECK(planner.end_frame(1000, &error));
  }

  // Multi-tile + adaptive CDFs is rejected instead of desynchronising.
  av2enc::m2::M2Config adaptive = two_config;
  adaptive.disable_cdf_update = false;
  M2TiledFrame rejected;
  CHECK(!assemble_m2_tiled_frame(adaptive, sequence, header, two.layout,
                                 two_blocks, levels, &rejected, &error));
  CHECK(error.find("adaptive CDF") != std::string::npos);
}

void test_bit_depth_checklist() {
  using namespace av2enc::m2::post;
  using av2enc::fmt::BitDepth;

  M2BitDepthChecklist c8 = make_m2_bit_depth_checklist(640, 360, 64,
                                                       BitDepth::k8, 90);
  CHECK(c8.ok_for_header());
  CHECK(c8.ok_for_8bit_m2_tile_path());
  CHECK_EQ(c8.bit_depth_idc, 1);
  CHECK_EQ(c8.base_q_idx_bits, 8);
  CHECK_EQ(c8.max_quantizer, 255);
  CHECK_EQ(c8.mi_cols, 160);
  CHECK_EQ(c8.mi_rows, 90);

  M2BitDepthChecklist c10 = make_m2_bit_depth_checklist(640, 360, 64,
                                                        BitDepth::k10, 90);
  CHECK(c10.ok_for_header());
  CHECK(!c10.ok_for_8bit_m2_tile_path());
  CHECK_EQ(c10.bit_depth_idc, 0);
  CHECK_EQ(c10.base_q_idx_bits, 9);
  CHECK_EQ(c10.max_quantizer, 303);

  M2BitDepthChecklist c12 = make_m2_bit_depth_checklist(64, 64, 64,
                                                        BitDepth::k12, 90);
  CHECK(!c12.ok_for_header());
  CHECK(!c12.ok_for_8bit_m2_tile_path());

  M2BitDepthChecklist bad_q =
      make_m2_bit_depth_checklist(64, 64, 64, BitDepth::k10, 304);
  CHECK(!bad_q.ok_for_header());

  std::string reason;
  CHECK(!m2_geometry_supported(130, 64, 64, &reason));
  CHECK(!m2_geometry_supported(64, 64, 32, &reason));
  CHECK(m2_geometry_supported(1920, 1080, 64, &reason));
}

std::vector<std::uint8_t> make_blocky_recon(int width, int height) {
  // A gentle 4-code step.  Hard steps and alternating texture are deliberately
  // rejected by the deblock side mask (see deblock_test_common.h), so they
  // would make an order test vacuous.
  std::vector<std::uint8_t> y(static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height));
  for (int yy = 0; yy < height; ++yy) {
    for (int xx = 0; xx < width; ++xx) {
      const int block = ((xx / 16) + (yy / 16)) & 1;
      const int value = block ? 126 : 130;
      y[static_cast<std::size_t>(yy) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(xx)] =
          static_cast<std::uint8_t>(value);
    }
  }
  return y;
}

av2enc::m2::M2ReconFrame make_recon(int width, int height) {
  av2enc::m2::M2ReconFrame recon;
  recon.width = static_cast<std::uint32_t>(width);
  recon.height = static_cast<std::uint32_t>(height);
  recon.y = make_blocky_recon(width, height);
  recon.u.assign(static_cast<std::size_t>(width / 2) *
                       static_cast<std::size_t>(height / 2),
                   110);
  recon.v.assign(static_cast<std::size_t>(width / 2) *
                       static_cast<std::size_t>(height / 2),
                   140);
  return recon;
}

bool same_planes(const std::vector<std::uint8_t>& a,
                 const std::vector<std::uint8_t>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size()) == 0;
}

void test_filter_order() {
  using namespace av2enc::m2::post;
  std::string error;

  M2PostFilterFrame frame;
  CHECK(frame.init_from_recon(make_recon(64, 64), 90, true, &error));
  frame.fill_deblock_rect(0, 0, 16, 16, false, true, 90);

  M2PostFilterConfig config;
  config.enable_deblock = true;
  config.deblock.apply_deblocking_filter[0] = true;
  config.deblock.apply_deblocking_filter[1] = true;
  config.deblock.apply_deblocking_filter[2] = true;
  config.deblock.apply_deblocking_filter[3] = true;
  config.deblock.frame_is_intra = true;
  config.enable_cdef = true;
  config.cdef.damping = 3;
  config.cdef.num_strengths = 1;
  config.cdef.strengths[0].y_pri = 4;
  config.cdef.strengths[0].y_sec = 2;
  config.cdef.strengths[0].uv_pri = 2;
  config.cdef.strengths[0].uv_sec = 1;
  config.cdef.on_skip_txfm = true;
  config.enable_ccso = true;
  config.ccso.ccso_luma_size_log2 = 6;
  for (int plane = 0; plane < 3; ++plane) {
    config.ccso.plane[plane].enabled = true;
    config.ccso.plane[plane].bo_only = true;
    config.ccso.plane[plane].scale_idx = 0;
    config.ccso.plane[plane].quant_idx = 0;
    config.ccso.plane[plane].max_band_log2 = 1;
    config.ccso.plane[plane].offsets[0] = 1;
    config.ccso.plane[plane].offsets[1] = 1;
  }

  M2PostFilterReport report;
  CHECK(run_m2_post_filters(frame, config, report, &error));
  CHECK(report.deblock_ran);
  CHECK(report.cdef_ran);
  CHECK(report.ccso_ran);
  CHECK_EQ(static_cast<int>(report.order.size()), 3);
  CHECK(report.order[0] == M2PostFilterStage::kDeblock);
  CHECK(report.order[1] == M2PostFilterStage::kCdef);
  CHECK(report.order[2] == M2PostFilterStage::kCcso);
  // The chain output must differ from the raw reconstruction.
  M2PostFilterFrame copy;
  CHECK(copy.init_from_recon(make_recon(64, 64), 90, true, &error));
  std::vector<std::uint8_t> original(copy.plane_data(0),
                                     copy.plane_data(0) + 64 * 64);
  CHECK(!same_planes(frame.output_plane(0), original));

  // CDEF must consume the post-deblock frame.  Running CDEF on the raw frame
  // first produces a different intermediate; the chain output follows the
  // deblocked source.
  M2PostFilterFrame manual;
  CHECK(manual.init_from_recon(make_recon(64, 64), 90, true, &error));
  manual.fill_deblock_rect(0, 0, 16, 16, false, true, 90);
  M2PostFilterConfig deblock_only;
  deblock_only.enable_deblock = true;
  deblock_only.deblock = config.deblock;
  M2PostFilterReport r1;
  CHECK(run_m2_post_filters(manual, deblock_only, r1, &error));
  CHECK(r1.deblock_ran);
  CHECK(!same_planes(manual.output_plane(0), original));

  // The all-three output must not equal the deblock-only output (CDEF/CCSO ran
  // after it).
  CHECK(!same_planes(frame.output_plane(0), manual.output_plane(0)));

  // Manual chain: deblock-only output fed back as the source of a CDEF+CCSO
  // run must match the single all-three call.  This pins the stage order and
  // the fact that CDEF consumes post-deblock samples, not the raw recon.
  av2enc::m2::M2ReconFrame deblocked_recon;
  deblocked_recon.width = 64;
  deblocked_recon.height = 64;
  deblocked_recon.y = manual.output_plane(0);
  deblocked_recon.u = manual.output_plane(1);
  deblocked_recon.v = manual.output_plane(2);
  M2PostFilterFrame chained;
  CHECK(chained.init_from_recon(deblocked_recon, 90, true, &error));
  M2PostFilterConfig cdef_ccso = config;
  cdef_ccso.enable_deblock = false;
  M2PostFilterReport r2;
  CHECK(run_m2_post_filters(chained, cdef_ccso, r2, &error));
  CHECK(!r2.deblock_ran);
  CHECK(r2.cdef_ran);
  CHECK(r2.ccso_ran);
  CHECK(same_planes(frame.output_plane(0), chained.output_plane(0)));
  CHECK(same_planes(frame.output_plane(1), chained.output_plane(1)));
  CHECK(same_planes(frame.output_plane(2), chained.output_plane(2)));

  // Wrong order: CDEF+CCSO on the raw reconstruction must differ.
  M2PostFilterFrame wrong;
  CHECK(wrong.init_from_recon(make_recon(64, 64), 90, true, &error));
  M2PostFilterReport r3;
  CHECK(run_m2_post_filters(wrong, cdef_ccso, r3, &error));
  CHECK(!same_planes(frame.output_plane(0), wrong.output_plane(0)));

  // No filters: the output is a copy of the reconstruction.
  M2PostFilterFrame none;
  CHECK(none.init_from_recon(make_recon(32, 32), 90, true, &error));
  M2PostFilterConfig empty;
  M2PostFilterReport rnone;
  CHECK(run_m2_post_filters(none, empty, rnone, &error));
  CHECK(rnone.order.empty());
  std::vector<std::uint8_t> none_in(none.plane_data(0),
                                    none.plane_data(0) + 32 * 32);
  CHECK(same_planes(none.output_plane(0), none_in));

  // 10-bit is rejected: the underlying filters are 8-bit only.
  M2PostFilterFrame hbd;
  CHECK(hbd.init_from_recon(make_recon(32, 32), 90, true, &error));
  hbd.set_bit_depth(10);
  M2PostFilterReport rhbd;
  CHECK(!run_m2_post_filters(hbd, empty, rhbd, &error));
  CHECK(error.find("8-bit") != std::string::npos);

  // Tile-aware clipping metadata.
  M2PostFilterFrame tiled;
  CHECK(tiled.init_from_recon(make_recon(128, 64), 90, true, &error));
  const auto layout_result = plan_m2_tiles(128, 64, 2, 1);
  CHECK(layout_result.ok);
  CHECK(tiled.set_tile_layout(layout_result.layout, &error));
  CHECK_EQ(tiled.tile_col_starts().size(), 2U);
  CHECK_EQ(tiled.tile_col_starts()[1], 16);
}

}  // namespace

int main() {
  test_rate_control();
  test_tiles_and_headers();
  test_bit_depth_checklist();
  test_filter_order();

  std::cout << "m2_post_test: " << g_checks << " checks, " << g_failures
            << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
