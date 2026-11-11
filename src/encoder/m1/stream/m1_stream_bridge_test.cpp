// Unit test for the M1 GPU -> syntax bridge (src/encoder/m1/stream/).
//
// Covers three things:
//   1. Byte identity: real CPU decisions/coefficients -> m1gpu PODs -> bridge ->
//      syntax writer must reproduce encode_m1_still_picture()'s frame bytes.
//   2. Tree packing: a hand-written multi-level partition tree packs into the
//      same BlockSyntax stream as the equivalent hand-written entries, and the
//      writer produces identical bytes from both.
//   3. Validation: malformed GPU results are rejected instead of writing a
//      stream that would desynchronize the CDF state.
//
// The CPU encoder keeps its decision loop in an anonymous namespace, so this
// test includes the translation unit (the same approach as
// src/cuda/m1/recon/recon_test.cu).  Everything is host-only; no CUDA needed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_syntax.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/m1/stream/m1_stream_bridge.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"

#include <cstdlib>
#include <sstream>

// Pulls in the CPU decision/reconstruction path and, through it, the syntax
// writer (m1_syntax.cpp).  Do not link m1_still.cpp / m1_syntax.cpp separately.
#include "encoder/m1/m1_still.cpp"

namespace {

using av2enc::m1::M1Config;
using av2enc::m1::M1Options;
using av2enc::m1::M1Result;
using av2enc::m1::StillPicture;
using av2enc::m1::SyntaxStream;
using av2enc::m1::syntax::BlockSyntax;
using av2enc::m1::syntax::FramePlan;
using av2enc::m1::stream::GpuCoefficients;
using av2enc::m1::stream::PackedFrame;
using av2enc::m1::stream::TreeNode;
using av2enc::partition::BlockSize;
using av2enc::partition::FrameGeometry;
using av2enc::partition::PartitionType;
using av2enc::txv::TxParams;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,  \
                   #cond);                                                   \
    }                                                                        \
  } while (0)

#define CHECK_MSG(cond, msg)                                                 \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s (%s)\n", __FILE__,       \
                   __LINE__, #cond, (msg));                                  \
    }                                                                        \
  } while (0)

// Deterministic synthetic 4:2:0 picture (no ffmpeg / test media needed).
StillPicture make_picture(int width, int height,
                          std::vector<std::uint8_t>& y,
                          std::vector<std::uint8_t>& u,
                          std::vector<std::uint8_t>& v) {
  const int cw = (width + 1) / 2;
  const int ch = (height + 1) / 2;
  y.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  u.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  v.resize(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch));
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int base = (col * 3 + row * 5) & 0x7F;
      const int detail = ((col ^ row) & 7) * 6;
      y[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
        static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(base + detail);
    }
  }
  for (int row = 0; row < ch; ++row) {
    for (int col = 0; col < cw; ++col) {
      u[static_cast<std::size_t>(row) * static_cast<std::size_t>(cw) +
        static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(64 + ((col * 2 + row) & 0x3F));
      v[static_cast<std::size_t>(row) * static_cast<std::size_t>(cw) +
        static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(160 + ((col + row * 3) & 0x3F));
    }
  }
  StillPicture picture;
  picture.width = static_cast<std::uint32_t>(width);
  picture.height = static_cast<std::uint32_t>(height);
  picture.y = y;
  picture.u = u;
  picture.v = v;
  return picture;
}

// M1Config exactly as encode_m1_still_picture() builds it for these options.
M1Config make_config(const StillPicture& picture, const M1Options& options) {
  M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_intra_edge_filter = options.enable_intra_edge_filter;
  config.enable_cfl_intra = options.enable_cfl;
  config.enable_chroma_dctonly = true;
  config.tx_mode_select = false;
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u = options.enable_deblock;
  config.apply_deblock_v = options.enable_deblock;
  return config;
}

// Everything the GPU path would hand the bridge: plan + leaves + decisions +
// the per-plane coefficient buffers in raster (transform output) order.
struct GpuFrame {
  av2enc::cuda::m1gpu::FramePlan plan;
  std::vector<av2enc::cuda::m1gpu::BlockPlan> blocks;
  std::vector<av2enc::cuda::m1gpu::BlockDecision> decisions;
  std::vector<std::int32_t> coeff_y;
  std::vector<std::int32_t> coeff_u;
  std::vector<std::int32_t> coeff_v;
};

// Convert the CPU decision stream (raster/single-block plan) into GPU PODs.
// Levels arrive scan-ordered from the CPU and must be written raster-ordered
// here, which is the inverse of what the bridge does.
bool build_raster_gpu_frame(const StillPicture& picture, const SyntaxStream& stream,
                            GpuFrame& frame, std::string& error) {
  FrameGeometry geometry;
  geometry.width = static_cast<int>(picture.width);
  geometry.height = static_cast<int>(picture.height);
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  frame.plan.width = picture.width;
  frame.plan.height = picture.height;
  frame.plan.mi_cols =
      static_cast<std::uint16_t>(av2enc::partition::mi_cols(geometry));
  frame.plan.mi_rows =
      static_cast<std::uint16_t>(av2enc::partition::mi_rows(geometry));
  frame.plan.sb_cols = static_cast<std::uint16_t>(
      (frame.plan.mi_cols + 15) / 16);
  frame.plan.sb_rows = static_cast<std::uint16_t>(
      (frame.plan.mi_rows + 15) / 16);
  frame.plan.sb_size = 64;
  frame.plan.is_placeholder = 1;

  frame.blocks.clear();
  frame.decisions.clear();
  frame.coeff_y.clear();
  frame.coeff_u.clear();
  frame.coeff_v.clear();

  for (const BlockSyntax& block : stream.blocks) {
    if ((block.flags & av2enc::m1::syntax::kFlagSbRoot) == 0) {
      error = "CPU stream node is not a superblock root";
      return false;
    }
    const bool is_leaf = block.partition ==
                         static_cast<std::uint8_t>(PartitionType::kNone);
    if (!is_leaf) {
      error = "raster conversion expects one PARTITION_NONE per superblock";
      return false;
    }
    const auto bsize = static_cast<BlockSize>(block.bsize);
    const BlockSize chroma_size =
        av2enc::partition::chroma_plane_residual_size(geometry, bsize);
    const int luma_w = av2enc::partition::block_width(bsize);
    const int luma_h = av2enc::partition::block_height(bsize);
    const int chroma_w = av2enc::partition::block_width(chroma_size);
    const int chroma_h = av2enc::partition::block_height(chroma_size);
    const auto luma_tx = av2enc::txv::make_tx_params(luma_w, luma_h);
    const auto chroma_tx = av2enc::txv::make_tx_params(chroma_w, chroma_h);
    if (!luma_tx.has_value() || !chroma_tx.has_value()) {
      error = "bad CPU transform size";
      return false;
    }

    av2enc::cuda::m1gpu::BlockPlan plan_block;
    plan_block.mi_row = block.mi_row;
    plan_block.mi_col = block.mi_col;
    plan_block.luma_w = static_cast<std::uint8_t>(luma_w);
    plan_block.luma_h = static_cast<std::uint8_t>(luma_h);
    plan_block.chroma_w = static_cast<std::uint8_t>(chroma_w);
    plan_block.chroma_h = static_cast<std::uint8_t>(chroma_h);
    plan_block.tx_w = static_cast<std::uint8_t>(luma_tx->w);
    plan_block.tx_h = static_cast<std::uint8_t>(luma_tx->h);
    plan_block.chroma_tx_w = static_cast<std::uint8_t>(chroma_tx->w);
    plan_block.chroma_tx_h = static_cast<std::uint8_t>(chroma_tx->h);
    plan_block.has_chroma = 1;
    plan_block.tx_upscale_w =
        static_cast<std::uint8_t>(luma_w > luma_tx->w ? 2 : 1);
    plan_block.tx_upscale_h =
        static_cast<std::uint8_t>(luma_h > luma_tx->h ? 2 : 1);
    plan_block.coeff_offset_y = static_cast<std::uint32_t>(frame.coeff_y.size());
    plan_block.coeff_offset_u = static_cast<std::uint32_t>(frame.coeff_u.size());
    plan_block.coeff_offset_v = static_cast<std::uint32_t>(frame.coeff_v.size());

    av2enc::cuda::m1gpu::BlockDecision decision;
    decision.luma_mode = block.y_mode;
    decision.luma_angle_delta = block.y_angle_delta;
    decision.uv_mode = block.uv_mode;
    decision.is_cfl =
        (block.flags & av2enc::m1::syntax::kFlagIsCfl) != 0 ? 1 : 0;
    decision.eob_y = static_cast<std::uint16_t>(block.eob[0]);
    decision.eob_u = static_cast<std::uint16_t>(block.eob[1]);
    decision.eob_v = static_cast<std::uint16_t>(block.eob[2]);

    const TxParams* params[3] = {&*luma_tx, &*chroma_tx, &*chroma_tx};
    std::vector<std::int32_t>* planes[3] = {&frame.coeff_y, &frame.coeff_u,
                                            &frame.coeff_v};
    std::size_t level_base = block.levels_offset;
    for (int plane = 0; plane < 3; ++plane) {
      const TxParams& tx = *params[plane];
      const int area = tx.w * tx.h;
      std::vector<std::int32_t>& out = *planes[plane];
      const std::size_t offset = out.size();
      out.resize(offset + static_cast<std::size_t>(area), 0);
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      av2enc::m1::syntax::build_scan(tx.w, tx.h, scan);
      for (std::uint32_t c = 0; c < block.eob[plane]; ++c) {
        out[offset + scan[c]] = stream.levels[level_base + c];
      }
      level_base += block.eob[plane];
    }

    frame.blocks.push_back(plan_block);
    frame.decisions.push_back(decision);
  }

  frame.plan.block_count = static_cast<std::uint32_t>(frame.blocks.size());
  frame.plan.leaf_count = static_cast<std::uint32_t>(frame.blocks.size());
  frame.plan.coeff_count_y = static_cast<std::uint32_t>(frame.coeff_y.size());
  frame.plan.coeff_count_u = static_cast<std::uint32_t>(frame.coeff_u.size());
  frame.plan.coeff_count_v = static_cast<std::uint32_t>(frame.coeff_v.size());
  return true;
}

GpuCoefficients coeffs_of(const GpuFrame& frame) {
  GpuCoefficients coeffs;
  coeffs.y = std::span<const std::int32_t>(frame.coeff_y);
  coeffs.u = std::span<const std::int32_t>(frame.coeff_u);
  coeffs.v = std::span<const std::int32_t>(frame.coeff_v);
  return coeffs;
}

// 1. Real CPU decisions -> GPU PODs -> bridge -> identical frame bytes.
void test_raster_round_trip(int width, int height, bool full_intra) {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  const StillPicture picture = make_picture(width, height, y, u, v);

  M1Options options;
  options.force_single_block = true;
  options.luma_dc_only = !full_intra;  // full intra exercises directional modes

  M1Result reference;
  std::string error;
  CHECK_MSG(av2enc::m1::encode_m1_still_picture(picture, options, reference,
                                                error),
            error.c_str());

  av2enc::m1::Encoder encoder(picture, options);
  SyntaxStream stream;
  CHECK_MSG(encoder.encode(&stream, error), error.c_str());

  GpuFrame gpu;
  CHECK_MSG(build_raster_gpu_frame(picture, stream, gpu, error), error.c_str());

  PackedFrame packed;
  CHECK_MSG(av2enc::m1::stream::pack_gpu_raster_frame(
                gpu.plan, gpu.blocks, gpu.decisions, coeffs_of(gpu), packed,
                error),
            error.c_str());

  std::vector<std::uint8_t> bytes;
  std::size_t tile_bytes = 0;
  const M1Config config = make_config(picture, options);
  CHECK_MSG(av2enc::m1::syntax::assemble_m1_frame(
                packed.plan, packed.blocks, packed.levels, config, bytes, error,
                &tile_bytes),
            error.c_str());

  CHECK_MSG(bytes == reference.frame, "bridge bytes differ from the CPU frame");
  CHECK(tile_bytes == reference.tile_bytes);
  CHECK(packed.blocks.size() == gpu.plan.sb_cols * gpu.plan.sb_rows);
  CHECK(packed.levels.size() == stream.levels.size());
}

// 2. Hand-written multi-level tree: bridge output == hand-written BlockSyntax.
void test_tree_pack_matches_block_syntax() {
  // 64x64 superblock split HORZ into two 64x32 blocks, all planes skipped.
  const BlockSize root_size = BlockSize::k64x64;
  const BlockSize leaf_size = BlockSize::k64x32;
  const auto leaf_luma_tx = av2enc::txv::make_tx_params(64, 32);
  const auto leaf_chroma_tx = av2enc::txv::make_tx_params(32, 16);
  CHECK(leaf_luma_tx.has_value());
  CHECK(leaf_chroma_tx.has_value());
  if (!leaf_luma_tx.has_value() || !leaf_chroma_tx.has_value()) return;

  av2enc::cuda::m1gpu::FramePlan plan;
  plan.width = 64;
  plan.height = 64;
  plan.mi_cols = 16;
  plan.mi_rows = 16;
  plan.sb_cols = 1;
  plan.sb_rows = 1;
  plan.sb_size = 64;
  plan.is_placeholder = 0;

  std::vector<TreeNode> tree(3);
  tree[0].mi_row = 0;
  tree[0].mi_col = 0;
  tree[0].bsize = static_cast<std::uint8_t>(root_size);
  tree[0].partition = static_cast<std::uint8_t>(PartitionType::kHorz);
  tree[0].allowed_mask = 0x0F;
  tree[0].flags = 0;
  tree[0].child_count = 2;
  for (int child = 0; child < 2; ++child) {
    TreeNode& node = tree[static_cast<std::size_t>(1 + child)];
    node.mi_row = static_cast<std::uint16_t>(child * 8);
    node.mi_col = 0;
    node.bsize = static_cast<std::uint8_t>(leaf_size);
    node.partition = static_cast<std::uint8_t>(PartitionType::kNone);
    node.allowed_mask = 0x01;
    node.flags = av2enc::m1::syntax::kFlagForced;
    node.child_count = 0;
    node.leaf_index = static_cast<std::uint32_t>(child);
  }

  std::vector<av2enc::cuda::m1gpu::BlockPlan> blocks(2);
  std::vector<av2enc::cuda::m1gpu::BlockDecision> decisions(2);
  for (int child = 0; child < 2; ++child) {
    av2enc::cuda::m1gpu::BlockPlan& block =
        blocks[static_cast<std::size_t>(child)];
    block.mi_row = static_cast<std::uint16_t>(child * 8);
    block.mi_col = 0;
    block.luma_w = 64;
    block.luma_h = 32;
    block.chroma_w = 32;
    block.chroma_h = 16;
    block.tx_w = static_cast<std::uint8_t>(leaf_luma_tx->w);
    block.tx_h = static_cast<std::uint8_t>(leaf_luma_tx->h);
    block.chroma_tx_w = static_cast<std::uint8_t>(leaf_chroma_tx->w);
    block.chroma_tx_h = static_cast<std::uint8_t>(leaf_chroma_tx->h);
    block.has_chroma = 1;
    // All-skip leaves never touch the coefficient planes.
    decisions[static_cast<std::size_t>(child)].luma_mode = 0;
    decisions[static_cast<std::size_t>(child)].uv_mode = 0;
    decisions[static_cast<std::size_t>(child)].is_cfl = 0;
  }

  PackedFrame packed;
  std::string error;
  const GpuCoefficients empty_coeffs{};
  CHECK_MSG(av2enc::m1::stream::pack_gpu_frame(plan, tree, blocks, decisions,
                                               empty_coeffs, packed, error),
            error.c_str());
  CHECK(packed.blocks.size() == 3);
  CHECK(packed.levels.empty());
  if (packed.blocks.size() != 3) return;
  CHECK(packed.blocks[0].subtree == 2);
  CHECK(packed.blocks[1].subtree == 0);
  CHECK(packed.blocks[2].subtree == 0);
  CHECK((packed.blocks[0].flags & av2enc::m1::syntax::kFlagSbRoot) != 0);
  CHECK((packed.blocks[1].flags & av2enc::m1::syntax::kFlagSbRoot) == 0);
  CHECK(packed.blocks[1].tx_size[0] ==
        static_cast<std::uint8_t>(leaf_luma_tx->tx_size));
  CHECK(packed.blocks[1].tx_size[1] ==
        static_cast<std::uint8_t>(leaf_chroma_tx->tx_size));

  // The same stream written by hand.
  FramePlan hand_plan;
  hand_plan.width = 64;
  hand_plan.height = 64;
  hand_plan.sb_size = 64;
  std::vector<BlockSyntax> hand(3);
  hand[0].mi_row = 0;
  hand[0].mi_col = 0;
  hand[0].bsize = static_cast<std::uint8_t>(root_size);
  hand[0].partition = static_cast<std::uint8_t>(PartitionType::kHorz);
  hand[0].allowed_mask = 0x0F;
  hand[0].flags = av2enc::m1::syntax::kFlagSbRoot;
  hand[0].subtree = 2;
  for (int child = 0; child < 2; ++child) {
    BlockSyntax& leaf = hand[static_cast<std::size_t>(1 + child)];
    leaf.mi_row = static_cast<std::uint16_t>(child * 8);
    leaf.mi_col = 0;
    leaf.bsize = static_cast<std::uint8_t>(leaf_size);
    leaf.partition = static_cast<std::uint8_t>(PartitionType::kNone);
    leaf.allowed_mask = 0x01;
    leaf.flags = av2enc::m1::syntax::kFlagForced;
    leaf.subtree = 0;
    leaf.y_mode = 0;
    leaf.y_angle_delta = 0;
    leaf.uv_mode = 0;
    leaf.tx_size[0] = static_cast<std::uint8_t>(leaf_luma_tx->tx_size);
    leaf.tx_size[1] = static_cast<std::uint8_t>(leaf_chroma_tx->tx_size);
    leaf.tx_size[2] = static_cast<std::uint8_t>(leaf_chroma_tx->tx_size);
    leaf.eob[0] = 0;
    leaf.eob[1] = 0;
    leaf.eob[2] = 0;
    leaf.levels_offset = 0;
  }

  M1Config config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  std::vector<std::uint8_t> from_bridge;
  std::vector<std::uint8_t> from_hand;
  CHECK_MSG(av2enc::m1::syntax::assemble_m1_frame(
                packed.plan, packed.blocks, packed.levels, config, from_bridge,
                error),
            error.c_str());
  CHECK_MSG(av2enc::m1::syntax::assemble_m1_frame(hand_plan, hand, {}, config,
                                                  from_hand, error),
            error.c_str());
  CHECK(!from_bridge.empty());
  CHECK(from_bridge == from_hand);

  // The writer rejects an inconsistent subtree count.
  {
    std::vector<BlockSyntax> bad = hand;
    bad[0].subtree = 3;  // past the end of a 3-entry stream
    std::vector<std::uint8_t> bytes;
    std::string subtree_error;
    CHECK(!av2enc::m1::syntax::assemble_m1_frame(
        hand_plan, bad, {}, config, bytes, subtree_error));
    CHECK(!subtree_error.empty());
  }
}

// 3. Per-plane GPU coefficient buffers -> one flat Y/U/V-interleaved array.
//
// The GPU keeps three independent plane buffers with their own offsets; the
// syntax writer consumes one array where each block contributes
// [luma eob][U eob][V eob] in scan order.  This test builds synthetic blocks
// whose per-plane offsets are deliberately unequal (and padded) so a naive
// "index by raster position" or "index by one shared offset" implementation
// cannot pass, then checks the packed array entry for entry.
void test_coefficient_layout_conversion() {
  struct LeafSpec {
    int mi_row = 0;
    int mi_col = 0;
    BlockSize size = BlockSize::kInvalid;
    std::uint32_t eob[3] = {0, 0, 0};
    std::vector<std::int32_t> raster[3];  // adjusted size, raster order
    std::uint32_t offset[3] = {0, 0, 0};
  };

  const auto add_leaf = [](int mi_row, int mi_col, BlockSize size,
                           bool full_eob) {
    LeafSpec leaf;
    leaf.mi_row = mi_row;
    leaf.mi_col = mi_col;
    leaf.size = size;
    const int luma_w = av2enc::partition::block_width(size);
    const int luma_h = av2enc::partition::block_height(size);
    const auto luma_tx = av2enc::txv::make_tx_params(luma_w, luma_h);
    const auto chroma_tx = av2enc::txv::make_tx_params(luma_w / 2, luma_h / 2);
    CHECK(luma_tx.has_value());
    CHECK(chroma_tx.has_value());
    if (!luma_tx.has_value() || !chroma_tx.has_value()) return leaf;
    const int areas[3] = {luma_tx->w * luma_tx->h, chroma_tx->w * chroma_tx->h,
                          chroma_tx->w * chroma_tx->h};
    for (int plane = 0; plane < 3; ++plane) {
      const int area = areas[plane];
      leaf.eob[plane] = full_eob ? static_cast<std::uint32_t>(area)
                                 : static_cast<std::uint32_t>(area / 3 + 1);
      leaf.raster[plane].resize(static_cast<std::size_t>(area));
    }
    return leaf;
  };

  // Frame A: one PARTITION_NONE 64x64 block (coded 64x64 -> adjusted 32x32).
  // Frame B: HORZ(64x64) / HORZ(64x32) / VERT(64x16) / HORZ(32x16) ends in two
  // 32x8 luma blocks.  (VERT/HORZ are not defined at 64x8, so 32x8 is reached
  // through 32x16 — the same restriction the syntax writer validates.)
  struct FrameSpec {
    std::vector<LeafSpec> leaves;
    std::vector<TreeNode> tree;
    int mi_rows = 0;
    int mi_cols = 0;
  };

  std::vector<FrameSpec> frames(2);
  {
    FrameSpec& frame = frames[0];
    frame.mi_rows = 16;
    frame.mi_cols = 16;
    frame.leaves.push_back(add_leaf(0, 0, BlockSize::k64x64, true));
    TreeNode root;
    root.mi_row = 0;
    root.mi_col = 0;
    root.bsize = static_cast<std::uint8_t>(BlockSize::k64x64);
    root.partition = static_cast<std::uint8_t>(PartitionType::kNone);
    root.allowed_mask = 0x0F;
    root.child_count = 0;
    root.leaf_index = 0;
    frame.tree.push_back(root);
  }
  {
    FrameSpec& frame = frames[1];
    frame.mi_rows = 16;
    frame.mi_cols = 16;
    // Leaves in preorder (leaf_index == index).
    frame.leaves.push_back(add_leaf(0, 0, BlockSize::k32x8, true));
    frame.leaves.push_back(add_leaf(2, 0, BlockSize::k32x8, true));
    frame.leaves.push_back(add_leaf(0, 8, BlockSize::k32x16, true));
    frame.leaves.push_back(add_leaf(4, 0, BlockSize::k64x16, true));
    frame.leaves.push_back(add_leaf(8, 0, BlockSize::k64x32, false));
    const auto node = [](int mi_row, int mi_col, BlockSize size,
                         PartitionType partition, std::uint8_t children,
                         std::uint32_t leaf_index) {
      TreeNode out;
      out.mi_row = static_cast<std::uint16_t>(mi_row);
      out.mi_col = static_cast<std::uint16_t>(mi_col);
      out.bsize = static_cast<std::uint8_t>(size);
      out.partition = static_cast<std::uint8_t>(partition);
      out.allowed_mask = 0x0F;
      out.child_count = children;
      out.leaf_index = leaf_index;
      out.flags = children == 0 ? av2enc::m1::syntax::kFlagForced : 0;
      return out;
    };
    frame.tree.push_back(node(0, 0, BlockSize::k64x64, PartitionType::kHorz, 2, 0));
    frame.tree.push_back(node(0, 0, BlockSize::k64x32, PartitionType::kHorz, 2, 0));
    frame.tree.push_back(node(0, 0, BlockSize::k64x16, PartitionType::kVert, 2, 0));
    frame.tree.push_back(node(0, 0, BlockSize::k32x16, PartitionType::kHorz, 2, 0));
    frame.tree.push_back(node(0, 0, BlockSize::k32x8, PartitionType::kNone, 0, 0));
    frame.tree.push_back(node(2, 0, BlockSize::k32x8, PartitionType::kNone, 0, 1));
    frame.tree.push_back(node(0, 8, BlockSize::k32x16, PartitionType::kNone, 0, 2));
    frame.tree.push_back(node(4, 0, BlockSize::k64x16, PartitionType::kNone, 0, 3));
    frame.tree.push_back(node(8, 0, BlockSize::k64x32, PartitionType::kNone, 0, 4));
  }

  for (std::size_t frame_index = 0; frame_index < frames.size();
       ++frame_index) {
    FrameSpec& frame = frames[frame_index];
    // Distinct, padded per-plane offsets: no shared base, no plane alignment.
    std::vector<std::int32_t> planes[3];
    const std::uint32_t plane_padding[3] = {3, 11, 5};
    std::uint32_t cursor[3] = {0, 0, 0};
    for (int plane = 0; plane < 3; ++plane) {
      planes[plane].assign(plane_padding[plane], -777);  // poison padding
      cursor[plane] = plane_padding[plane];
    }
    std::vector<std::int32_t> expected;
    for (std::size_t leaf_index = 0; leaf_index < frame.leaves.size();
         ++leaf_index) {
      LeafSpec& leaf = frame.leaves[leaf_index];
      const int luma_w = av2enc::partition::block_width(leaf.size);
      const int luma_h = av2enc::partition::block_height(leaf.size);
      const auto luma_tx = av2enc::txv::make_tx_params(luma_w, luma_h);
      const auto chroma_tx =
          av2enc::txv::make_tx_params(luma_w / 2, luma_h / 2);
      const TxParams* params[3] = {&*luma_tx, &*chroma_tx, &*chroma_tx};
      for (int plane = 0; plane < 3; ++plane) {
        const TxParams& tx = *params[plane];
        const int area = tx.w * tx.h;
        leaf.offset[plane] = cursor[plane];
        cursor[plane] += static_cast<std::uint32_t>(area);
        planes[plane].resize(cursor[plane]);
        for (int i = 0; i < area; ++i) {
          // Unique per (frame, leaf, plane, raster index), sign-alternating.
          const int magnitude =
              1 + static_cast<int>((frame_index * 1000 + leaf_index * 100 +
                                    static_cast<std::size_t>(plane) * 10 +
                                    static_cast<std::size_t>(i)) %
                                   29);
          const int value = (i % 2 == 0) ? magnitude : -magnitude;
          leaf.raster[plane][static_cast<std::size_t>(i)] = value;
          planes[plane][leaf.offset[plane] + static_cast<std::uint32_t>(i)] =
              value;
        }
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        av2enc::m1::syntax::build_scan(tx.w, tx.h, scan);
        for (std::uint32_t c = 0; c < leaf.eob[plane]; ++c) {
          expected.push_back(leaf.raster[plane][scan[c]]);
        }
      }
    }

    // GPU-side PODs.
    av2enc::cuda::m1gpu::FramePlan plan;
    plan.width = 64;
    plan.height = 64;
    plan.mi_cols = static_cast<std::uint16_t>(frame.mi_cols);
    plan.mi_rows = static_cast<std::uint16_t>(frame.mi_rows);
    plan.sb_cols = 1;
    plan.sb_rows = 1;
    plan.sb_size = 64;
    plan.is_placeholder = 0;
    plan.coeff_count_y = static_cast<std::uint32_t>(planes[0].size());
    plan.coeff_count_u = static_cast<std::uint32_t>(planes[1].size());
    plan.coeff_count_v = static_cast<std::uint32_t>(planes[2].size());

    std::vector<av2enc::cuda::m1gpu::BlockPlan> blocks(frame.leaves.size());
    std::vector<av2enc::cuda::m1gpu::BlockDecision> decisions(
        frame.leaves.size());
    for (std::size_t i = 0; i < frame.leaves.size(); ++i) {
      const LeafSpec& leaf = frame.leaves[i];
      const int luma_w = av2enc::partition::block_width(leaf.size);
      const int luma_h = av2enc::partition::block_height(leaf.size);
      const auto luma_tx = av2enc::txv::make_tx_params(luma_w, luma_h);
      const auto chroma_tx =
          av2enc::txv::make_tx_params(luma_w / 2, luma_h / 2);
      av2enc::cuda::m1gpu::BlockPlan& block = blocks[i];
      block.mi_row = static_cast<std::uint16_t>(leaf.mi_row);
      block.mi_col = static_cast<std::uint16_t>(leaf.mi_col);
      block.luma_w = static_cast<std::uint8_t>(luma_w);
      block.luma_h = static_cast<std::uint8_t>(luma_h);
      block.chroma_w = static_cast<std::uint8_t>(luma_w / 2);
      block.chroma_h = static_cast<std::uint8_t>(luma_h / 2);
      block.tx_w = static_cast<std::uint8_t>(luma_tx->w);
      block.tx_h = static_cast<std::uint8_t>(luma_tx->h);
      block.chroma_tx_w = static_cast<std::uint8_t>(chroma_tx->w);
      block.chroma_tx_h = static_cast<std::uint8_t>(chroma_tx->h);
      block.has_chroma = 1;
      block.coeff_offset_y = leaf.offset[0];
      block.coeff_offset_u = leaf.offset[1];
      block.coeff_offset_v = leaf.offset[2];
      av2enc::cuda::m1gpu::BlockDecision& decision = decisions[i];
      decision.luma_mode = 0;
      decision.uv_mode = 0;
      decision.eob_y = static_cast<std::uint16_t>(leaf.eob[0]);
      decision.eob_u = static_cast<std::uint16_t>(leaf.eob[1]);
      decision.eob_v = static_cast<std::uint16_t>(leaf.eob[2]);
    }

    GpuCoefficients coeffs;
    coeffs.y = std::span<const std::int32_t>(planes[0]);
    coeffs.u = std::span<const std::int32_t>(planes[1]);
    coeffs.v = std::span<const std::int32_t>(planes[2]);

    PackedFrame packed;
    std::string error;
    CHECK_MSG(av2enc::m1::stream::pack_gpu_frame(plan, frame.tree, blocks,
                                                 decisions, coeffs, packed,
                                                 error),
              error.c_str());
    CHECK_MSG(packed.levels == expected,
              "flat level array is not the interleaved scan-order stream");

    // Per-block slices must line up with levels_offset in visit order.
    std::size_t offset = 0;
    for (std::size_t i = 0; i < frame.leaves.size(); ++i) {
      // Tree entries interleave internal nodes, so find the leaf's BlockSyntax
      // by its (mi_row, mi_col) position.
      const BlockSyntax* block = nullptr;
      for (const BlockSyntax& candidate : packed.blocks) {
        if (candidate.mi_row == blocks[i].mi_row &&
            candidate.mi_col == blocks[i].mi_col &&
            candidate.partition ==
                static_cast<std::uint8_t>(PartitionType::kNone)) {
          block = &candidate;
          break;
        }
      }
      CHECK(block != nullptr);
      if (block == nullptr) continue;
      CHECK(block->levels_offset == offset);
      offset += block->eob[0] + block->eob[1] + block->eob[2];
    }
    CHECK(offset == expected.size());

    // The writer must accept the block-interleaved stream.
    M1Config config;
    config.max_frame_width = 64;
    config.max_frame_height = 64;
    std::vector<std::uint8_t> bytes;
    CHECK_MSG(av2enc::m1::syntax::assemble_m1_frame(
                  packed.plan, packed.blocks, packed.levels, config, bytes,
                  error),
              error.c_str());
    CHECK(!bytes.empty());
  }

  // 64x64 coded blocks are 32x32 adjusted; the synthetic tree must contain a
  // real 32x8 luma transform as well.
  {
    const auto big = av2enc::txv::make_tx_params(64, 64);
    const auto narrow = av2enc::txv::make_tx_params(32, 8);
    CHECK(big.has_value() && big->w == 32 && big->h == 32);
    CHECK(narrow.has_value() && narrow->w == 32 && narrow->h == 8);
    CHECK(static_cast<std::uint8_t>(BlockSize::k32x8) == 22);
  }

  // Overlapping per-plane coefficient ranges are rejected: two blocks must not
  // code the same coefficients.
  {
    std::vector<TreeNode> tree(3);
    tree[0].mi_row = 0;
    tree[0].mi_col = 0;
    tree[0].bsize = static_cast<std::uint8_t>(BlockSize::k64x64);
    tree[0].partition = static_cast<std::uint8_t>(PartitionType::kHorz);
    tree[0].allowed_mask = 0x0F;
    tree[0].child_count = 2;
    for (int child = 0; child < 2; ++child) {
      TreeNode& leaf = tree[static_cast<std::size_t>(1 + child)];
      leaf.mi_row = static_cast<std::uint16_t>(child * 8);
      leaf.mi_col = 0;
      leaf.bsize = static_cast<std::uint8_t>(BlockSize::k64x32);
      leaf.partition = static_cast<std::uint8_t>(PartitionType::kNone);
      leaf.allowed_mask = 0x01;
      leaf.flags = av2enc::m1::syntax::kFlagForced;
      leaf.child_count = 0;
      leaf.leaf_index = static_cast<std::uint32_t>(child);
    }

    av2enc::cuda::m1gpu::FramePlan plan;
    plan.width = 64;
    plan.height = 64;
    plan.mi_cols = 16;
    plan.mi_rows = 16;
    plan.sb_cols = 1;
    plan.sb_rows = 1;
    plan.is_placeholder = 0;
    std::vector<av2enc::cuda::m1gpu::BlockPlan> blocks(2);
    std::vector<av2enc::cuda::m1gpu::BlockDecision> decisions(2);
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      av2enc::cuda::m1gpu::BlockPlan& block = blocks[i];
      block.mi_row = static_cast<std::uint16_t>(i * 8);
      block.luma_w = 64;
      block.luma_h = 32;
      block.chroma_w = 32;
      block.chroma_h = 16;
      block.tx_w = 32;
      block.tx_h = 32;
      block.chroma_tx_w = 32;
      block.chroma_tx_h = 16;
      block.has_chroma = 1;
      block.coeff_offset_y = 0;  // both leaves claim the same luma range
      decisions[i].eob_y = 4;
    }
    std::vector<std::int32_t> levels(4096, 1);
    GpuCoefficients coeffs;
    coeffs.y = std::span<const std::int32_t>(levels);
    PackedFrame packed;
    std::string error;
    CHECK(!av2enc::m1::stream::pack_gpu_frame(plan, tree, blocks, decisions,
                                              coeffs, packed, error));
    CHECK(error.find("overlap") != std::string::npos);
  }
}

// 4. Real host plan (plan::build_frame_plan) -> syntax stream.
//
// The plan module is the GPU pipeline's block list; it carries leaves only, so
// the bridge derives the preorder tree from that list.  This test builds real
// 64x64 and 640x360 plans, synthesizes decisions + coefficients, packs them,
// and checks the node/leaf order against `plan_to_tsv()` and `plan.nodes`.
void test_plan_frame_ordering(int width, int height) {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  make_picture(width, height, y, u, v);

  av2enc::m1::plan::PlanOptions plan_options;
  av2enc::m1::plan::FramePlan plan;
  std::string error;
  CHECK_MSG(av2enc::m1::plan::build_frame_plan(width, height, y, plan_options,
                                               plan, error),
            error.c_str());
  CHECK_MSG(av2enc::m1::plan::validate_frame_plan(plan, error), error.c_str());
  CHECK(!plan.blocks.empty());
  CHECK(plan.leaves == static_cast<int>(plan.blocks.size()));
  if (plan.blocks.empty()) return;

  // Synthetic decisions: DC / DC, no CfL, a deterministic EOB pattern.
  std::vector<av2enc::cuda::m1gpu::BlockDecision> decisions(plan.blocks.size());
  std::size_t expected_levels = 0;
  for (std::size_t i = 0; i < plan.blocks.size(); ++i) {
    const av2enc::m1::plan::BlockPlan& block = plan.blocks[i];
    av2enc::cuda::m1gpu::BlockDecision& decision = decisions[i];
    decision.luma_mode = 0;
    decision.uv_mode = 0;
    decision.is_cfl = 0;
    const int areas[3] = {block.luma.adj_w * block.luma.adj_h,
                          block.u.adj_w * block.u.adj_h,
                          block.v.adj_w * block.v.adj_h};
    std::uint16_t eobs[3];
    for (int plane = 0; plane < 3; ++plane) {
      eobs[plane] = static_cast<std::uint16_t>(
          1 + ((i * 7U + static_cast<std::size_t>(plane) * 3U) %
               static_cast<std::size_t>(areas[plane])));
    }
    // Exercise the skip path too.
    if (i % 5U == 4U) eobs[2] = 0;
    decision.eob_y = eobs[0];
    decision.eob_u = eobs[1];
    decision.eob_v = eobs[2];
    expected_levels += eobs[0] + eobs[1] + eobs[2];
  }

  // Per-plane coefficient buffers in the plan's layout (raster order, offsets
  // from PlaneTxPlan::coeff_offset).
  std::vector<std::int32_t> planes[3];
  planes[0].assign(plan.luma_coeff_count, 0);
  planes[1].assign(plan.u_coeff_count, 0);
  planes[2].assign(plan.v_coeff_count, 0);
  for (std::size_t i = 0; i < plan.blocks.size(); ++i) {
    const av2enc::m1::plan::BlockPlan& block = plan.blocks[i];
    const av2enc::m1::plan::PlaneTxPlan* txs[3] = {&block.luma, &block.u,
                                                   &block.v};
    for (int plane = 0; plane < 3; ++plane) {
      const av2enc::m1::plan::PlaneTxPlan& tx = *txs[plane];
      const std::size_t area = static_cast<std::size_t>(tx.adj_w * tx.adj_h);
      CHECK(tx.coeff_offset + area <= planes[plane].size());
      if (tx.coeff_offset + area > planes[plane].size()) continue;
      for (std::size_t e = 0; e < area; ++e) {
        const int magnitude =
            1 + static_cast<int>((i * 31 + e * 7 +
                                  static_cast<std::size_t>(plane)) %
                                 23);
        planes[plane][tx.coeff_offset + e] =
            (e % 2 == 0) ? magnitude : -magnitude;
      }
    }
  }
  GpuCoefficients coeffs;
  coeffs.y = std::span<const std::int32_t>(planes[0]);
  coeffs.u = std::span<const std::int32_t>(planes[1]);
  coeffs.v = std::span<const std::int32_t>(planes[2]);

  // The plan's exported tree converts 1:1 and matches its own accounting.
  std::vector<TreeNode> tree;
  CHECK_MSG(av2enc::m1::stream::plan_tree_to_nodes(plan, tree, error),
            error.c_str());
  CHECK(tree.size() == plan.tree.size());
  CHECK(static_cast<int>(tree.size()) == plan.nodes);

  // Tree order: leaves appear in plan order, each at its own geometry.
  std::size_t leaf_cursor = 0;
  for (const TreeNode& node : tree) {
    if (node.child_count != 0) continue;
    CHECK(node.leaf_index == leaf_cursor);
    if (node.leaf_index != leaf_cursor) continue;
    const av2enc::m1::plan::BlockPlan& block = plan.blocks[leaf_cursor];
    CHECK(node.mi_row == block.mi_row);
    CHECK(node.mi_col == block.mi_col);
    CHECK(node.bsize == block.block_size);
    ++leaf_cursor;
  }
  CHECK(leaf_cursor == plan.blocks.size());

  // Superblock roots: raster order, SB-aligned, marked.
  int sb_index = 0;
  for (std::size_t i = 0; i < tree.size();) {
    const TreeNode& root = tree[i];
    CHECK((root.flags & av2enc::m1::syntax::kFlagSbRoot) != 0);
    CHECK(static_cast<int>(root.mi_row) ==
          (sb_index / plan.sb_cols) * (plan.sb_size / 4));
    CHECK(static_cast<int>(root.mi_col) ==
          (sb_index % plan.sb_cols) * (plan.sb_size / 4));
    // Skip the whole subtree with an explicit child-count stack.
    std::size_t cursor = i + 1;
    std::vector<int> stack;
    if (root.child_count > 0) stack.push_back(root.child_count);
    while (!stack.empty() && cursor < tree.size()) {
      const TreeNode& node = tree[cursor];
      --stack.back();
      if (node.child_count > 0) stack.push_back(node.child_count);
      ++cursor;
      while (!stack.empty() && stack.back() == 0) stack.pop_back();
    }
    CHECK(cursor > i);
    i = cursor;
    ++sb_index;
  }
  CHECK(sb_index == plan.sb_cols * plan.sb_rows);

  PackedFrame packed;
  CHECK_MSG(av2enc::m1::stream::pack_plan_frame(plan, plan.blocks, decisions,
                                                coeffs, packed, error),
            error.c_str());
  CHECK(packed.levels.size() == expected_levels);

  // Packed leaf order must match plan.blocks and the TSV dump.
  std::vector<const BlockSyntax*> packed_leaves;
  for (const BlockSyntax& block : packed.blocks) {
    if (block.partition == static_cast<std::uint8_t>(PartitionType::kNone)) {
      packed_leaves.push_back(&block);
    }
  }
  CHECK(packed_leaves.size() == plan.blocks.size());

  const std::string tsv = av2enc::m1::plan::plan_to_tsv(plan);
  std::istringstream lines(tsv);
  std::string line;
  std::size_t row = 0;
  while (std::getline(lines, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (line.compare(0, 5, "index") == 0) continue;  // column header
    std::vector<std::string> columns;
    std::size_t start = 0;
    while (true) {
      const std::size_t tab = line.find('\t', start);
      columns.push_back(
          line.substr(start, tab == std::string::npos ? std::string::npos
                                                      : tab - start));
      if (tab == std::string::npos) break;
      start = tab + 1;
    }
    CHECK(columns.size() >= 7);
    if (columns.size() < 7 || row >= packed_leaves.size()) {
      ++row;
      continue;
    }
    CHECK_MSG(static_cast<std::size_t>(std::atoi(columns[0].c_str())) == row,
              "plan dump index is not sequential");
    CHECK(static_cast<int>(packed_leaves[row]->mi_row) ==
          std::atoi(columns[4].c_str()));
    CHECK(static_cast<int>(packed_leaves[row]->mi_col) ==
          std::atoi(columns[5].c_str()));
    CHECK(std::string(av2enc::partition::block_size_name(
              static_cast<BlockSize>(packed_leaves[row]->bsize))) ==
          columns[6]);
    ++row;
  }
  CHECK(row == plan.blocks.size());

  M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  std::vector<std::uint8_t> bytes;
  std::size_t tile_bytes = 0;
  CHECK_MSG(av2enc::m1::stream::assemble_plan_frame(
                plan, plan.blocks, decisions, coeffs, config, bytes, error,
                &tile_bytes),
            error.c_str());
  CHECK(!bytes.empty());
  CHECK(tile_bytes > 0);

  // End-to-end: the CPU reference's own decisions + coefficients, routed
  // through the plan tree and the bridge, must reproduce the CPU frame bytes.
  // (Both default M1Options and default PlanOptions are min 8 / depth 4 /
  // rect-only, so the two trees must agree; a mismatch means the plan's cost
  // model drifted from m1_still.cpp.)
  StillPicture picture;
  picture.width = static_cast<std::uint32_t>(width);
  picture.height = static_cast<std::uint32_t>(height);
  picture.y = y;
  picture.u = u;
  picture.v = v;
  M1Options options;
  M1Result reference;
  CHECK_MSG(av2enc::m1::encode_m1_still_picture(picture, options, reference,
                                                error),
            error.c_str());
  av2enc::m1::Encoder encoder(picture, options);
  SyntaxStream cpu_stream;
  CHECK_MSG(encoder.encode(&cpu_stream, error), error.c_str());
  std::vector<const BlockSyntax*> cpu_leaves;
  for (const BlockSyntax& block : cpu_stream.blocks) {
    if (block.partition == static_cast<std::uint8_t>(PartitionType::kNone)) {
      cpu_leaves.push_back(&block);
    }
  }
  CHECK_MSG(cpu_leaves.size() == plan.blocks.size(),
            "plan leaf count differs from the CPU M1 encoder");
  if (cpu_leaves.size() != plan.blocks.size()) return;

  std::vector<av2enc::cuda::m1gpu::BlockDecision> cpu_decisions(plan.blocks.size());
  std::vector<std::int32_t> cpu_planes[3];
  cpu_planes[0].assign(plan.luma_coeff_count, 0);
  cpu_planes[1].assign(plan.u_coeff_count, 0);
  cpu_planes[2].assign(plan.v_coeff_count, 0);
  for (std::size_t i = 0; i < cpu_leaves.size(); ++i) {
    const BlockSyntax& leaf = *cpu_leaves[i];
    CHECK_MSG(leaf.mi_row == plan.blocks[i].mi_row &&
                  leaf.mi_col == plan.blocks[i].mi_col &&
                  leaf.bsize == plan.blocks[i].block_size,
              "plan tree leaf order differs from the CPU M1 encoder");
    av2enc::cuda::m1gpu::BlockDecision& decision = cpu_decisions[i];
    decision.luma_mode = leaf.y_mode;
    decision.luma_angle_delta = leaf.y_angle_delta;
    decision.uv_mode = leaf.uv_mode;
    decision.is_cfl = (leaf.flags & av2enc::m1::syntax::kFlagIsCfl) != 0 ? 1 : 0;
    decision.eob_y = static_cast<std::uint16_t>(leaf.eob[0]);
    decision.eob_u = static_cast<std::uint16_t>(leaf.eob[1]);
    decision.eob_v = static_cast<std::uint16_t>(leaf.eob[2]);
    const av2enc::m1::plan::PlaneTxPlan* txs[3] = {
        &plan.blocks[i].luma, &plan.blocks[i].u, &plan.blocks[i].v};
    std::size_t level_base = leaf.levels_offset;
    for (int plane = 0; plane < 3; ++plane) {
      const av2enc::m1::plan::PlaneTxPlan& tx = *txs[plane];
      const int area = tx.adj_w * tx.adj_h;
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      av2enc::m1::syntax::build_scan(tx.adj_w, tx.adj_h, scan);
      for (std::uint32_t c = 0; c < leaf.eob[plane]; ++c) {
        cpu_planes[plane][tx.coeff_offset + scan[c]] =
            cpu_stream.levels[level_base + c];
      }
      level_base += leaf.eob[plane];
    }
  }
  GpuCoefficients cpu_coeffs;
  cpu_coeffs.y = std::span<const std::int32_t>(cpu_planes[0]);
  cpu_coeffs.u = std::span<const std::int32_t>(cpu_planes[1]);
  cpu_coeffs.v = std::span<const std::int32_t>(cpu_planes[2]);
  std::vector<std::uint8_t> cpu_bytes;
  CHECK_MSG(av2enc::m1::stream::assemble_plan_frame(
                plan, plan.blocks, cpu_decisions, cpu_coeffs, config,
                cpu_bytes, error),
            error.c_str());
  CHECK_MSG(cpu_bytes == reference.frame,
            "plan+bridge bytes differ from the CPU M1 frame");
}

void test_validation() {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  const StillPicture picture = make_picture(64, 64, y, u, v);

  M1Options options;
  options.force_single_block = true;
  M1Result reference;
  std::string error;
  CHECK(av2enc::m1::encode_m1_still_picture(picture, options, reference, error));

  av2enc::m1::Encoder encoder(picture, options);
  SyntaxStream stream;
  CHECK_MSG(encoder.encode(&stream, error), error.c_str());

  GpuFrame gpu;
  CHECK_MSG(build_raster_gpu_frame(picture, stream, gpu, error), error.c_str());

  PackedFrame packed;
  // Baseline must succeed.
  CHECK_MSG(av2enc::m1::stream::pack_gpu_raster_frame(
                gpu.plan, gpu.blocks, gpu.decisions, coeffs_of(gpu), packed,
                error),
            error.c_str());

  // Wrong block count for a raster plan.
  {
    std::vector<av2enc::cuda::m1gpu::BlockPlan> blocks(gpu.blocks.begin(),
                                                       gpu.blocks.end() - 1);
    std::vector<av2enc::cuda::m1gpu::BlockDecision> decisions(
        gpu.decisions.begin(), gpu.decisions.end() - 1);
    PackedFrame out;
    std::string err;
    CHECK(!av2enc::m1::stream::pack_gpu_raster_frame(
        gpu.plan, blocks, decisions, coeffs_of(gpu), out, err));
    CHECK(!err.empty());
  }
  // eob outside the transform area.
  {
    GpuFrame bad = gpu;
    bad.decisions[0].eob_y = 4096;
    PackedFrame out;
    std::string err;
    CHECK(!av2enc::m1::stream::pack_gpu_raster_frame(
        bad.plan, bad.blocks, bad.decisions, coeffs_of(bad), out, err));
    CHECK(!err.empty());
  }
  // Non-DCT transform type (M1 codes no tx_type syntax).
  {
    GpuFrame bad = gpu;
    bad.decisions[0].tx_type_y = 1;
    PackedFrame out;
    std::string err;
    CHECK(!av2enc::m1::stream::pack_gpu_raster_frame(
        bad.plan, bad.blocks, bad.decisions, coeffs_of(bad), out, err));
    CHECK(!err.empty());
  }
  // Truncated coefficient plane while the block still has levels.
  {
    GpuFrame bad = gpu;
    if (bad.decisions[0].eob_u != 0) {
      bad.coeff_u.resize(bad.coeff_u.size() / 2);
      PackedFrame out;
      std::string err;
      CHECK(!av2enc::m1::stream::pack_gpu_raster_frame(
          bad.plan, bad.blocks, bad.decisions, coeffs_of(bad), out, err));
      CHECK(!err.empty());
    }
  }
  // Tree node with a bad block size / child count.
  {
    std::vector<TreeNode> tree(1);
    tree[0].mi_row = 0;
    tree[0].mi_col = 0;
    tree[0].bsize = static_cast<std::uint8_t>(BlockSize::k64x64);
    tree[0].partition = static_cast<std::uint8_t>(PartitionType::kHorz);
    tree[0].allowed_mask = 0x0F;
    tree[0].child_count = 0;  // non-leaf without children
    PackedFrame out;
    std::string err;
    CHECK(!av2enc::m1::stream::pack_gpu_frame(gpu.plan, tree, gpu.blocks,
                                              gpu.decisions, coeffs_of(gpu),
                                              out, err));
    CHECK(!err.empty());
  }
  // Frame-plan / header size mismatch.
  {
    std::vector<std::uint8_t> bytes;
    std::string err;
    M1Config config = make_config(picture, options);
    config.max_frame_width = 128;
    CHECK(!av2enc::m1::syntax::assemble_m1_frame(
        packed.plan, packed.blocks, packed.levels, config, bytes, err));
    CHECK(!err.empty());
  }
}

}  // namespace

int main() {
  test_raster_round_trip(64, 64, /*full_intra=*/false);
  test_raster_round_trip(64, 64, /*full_intra=*/true);
  test_raster_round_trip(640, 360, /*full_intra=*/false);
  test_raster_round_trip(640, 360, /*full_intra=*/true);
  test_tree_pack_matches_block_syntax();
  test_coefficient_layout_conversion();
  test_plan_frame_ordering(64, 64);
  test_plan_frame_ordering(640, 360);
  test_validation();

  if (g_failures != 0) {
    std::fprintf(stderr, "m1_stream_bridge_test: %d/%d checks failed\n",
                 g_failures, g_checks);
    return 1;
  }
  std::printf("m1_stream_bridge_test: PASS (%d checks)\n", g_checks);
  return 0;
}
