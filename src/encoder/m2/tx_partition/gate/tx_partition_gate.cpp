// dav2d gate for AV2 transform-partition signalling (TX_MODE_SELECT).
//
// Builds a deterministic two-frame 64x64 4:2:0 stream:
//   * frame 0: intra keyframe, TX_MODE_LARGEST, DC prediction, all-zero
//     residual -> reconstruction is 128 everywhere;
//   * frame 1: P frame with tx_mode_select = 1.  One non-skip inter block
//     (zero GLOBALMV, prediction = 128) signals a real transform partition
//     and codes non-zero coefficients in every resulting transform block.
//     All other blocks are skip.
//
// The gate reconstructs frame 1 from the same transform coefficients and
// requires dav2d (--strict 1 --threads 1) to reproduce every sample.
//
// Cases (--case):
//   A  one 64x64 coding block, TX SPLIT -> 4x 32x32   (default)
//   B  16x16 coding blocks, TX SPLIT -> 4x 8x8
//   C   8x8 coding blocks, TX SPLIT -> 4x 4x4
//   D  16x16 coding blocks, TX HORZ  -> 2x 8x16
//   E  16x16 coding blocks, TX VERT  -> 2x 16x8
//   F  16x16 coding blocks, TX NONE  (control: no smaller transform)
//
// Exit code 0 only when the stream decodes and every sample matches.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "encoder/m2/m2_headers.h"
#include "encoder/m2/m2_syntax.h"
#include "encoder/m2/tx_partition/gate/tx_partition_gate_hook.h"
#include "encoder/m2/tx_partition/tx_partition.h"
#include "encoder/m2/tx_partition/tx_scan.h"
#include "encoder/partition/partition.h"
#include "encoder/tx_variants/tx_variants.h"
#include "io/ivf.h"
#include "io/y4m.h"

namespace {

using av2enc::m2::M2Config;
using av2enc::m2::M2FrameHeader;
using av2enc::m2::syntax::BlockSyntax;
using av2enc::m2::syntax::FramePlan;
using av2enc::m2::txpartition::BlockTxContext;
using av2enc::m2::txpartition::SplitGeometry;
using av2enc::m2::txpartition::TxPartition;
using av2enc::partition::BlockSize;
using av2enc::partition::FrameGeometry;
using av2enc::partition::PartitionNode;

constexpr int kWidth = 64;
constexpr int kHeight = 64;
constexpr int kChromaWidth = 32;
constexpr int kChromaHeight = 32;
constexpr int kBaseQIdx = 90;  // qlookup(90) == 528 == txv::kQuantStep

struct CaseConfig {
  char id = 'A';
  const char* name = "64x64 -> 4x 32x32";
  int leaf = 64;
  TxPartition partition = TxPartition::kSplit;
  // Negative control: keep the split syntax in the tile but signal
  // tx_mode_select = 0, so the decoder parses those bits as coefficients.
  bool negative = false;
};

struct FramePlanes {
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

struct Stream {
  std::vector<std::vector<std::uint8_t>> units;
  FramePlanes recon0;
  FramePlanes recon1;
  std::size_t tile0_bytes = 0;
  std::size_t tile1_bytes = 0;
  std::uint32_t split_blocks = 0;
  std::uint32_t split_parts = 0;
};

int clip1(int value) { return std::clamp(value, 0, 255); }

// Splits a square superblock down to exactly `leaf`-sized squares using
// HORZ/VERT (PARTITION_SPLIT is not legal at 64x64 superblocks in this
// encoder).  The cost function is spec-independent: it only steers the search
// to a deterministic tree, and decide_frame() derives every allowed/forced
// flag from §5.20.3.2.
av2enc::partition::FrameDecision decide_tree(int leaf_px) {
  FrameGeometry geometry;
  geometry.width = kWidth;
  geometry.height = kHeight;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  const auto cost_fn = [leaf_px](
                           const av2enc::partition::PartitionNodeInfo& info) {
    av2enc::partition::NodeCosts costs;
    const int bw = av2enc::partition::block_width(info.size);
    const int bh = av2enc::partition::block_height(info.size);
    costs.leaf = (bw <= leaf_px && bh <= leaf_px) ? 0.0 : 1.0e9;
    costs.none_signal = 0.0;
    // Prefer the split that moves a rectangle towards square.
    const bool square = bw == bh;
    costs.horz_signal = square ? 100.0 : (bh > bw ? 1.0 : 100.0);
    costs.vert_signal = square ? 100.0 : (bw > bh ? 1.0 : 100.0);
    costs.split_signal = 1.0e9;  // never used (allow_split = false)
    return costs;
  };

  av2enc::partition::DecisionOptions options;
  options.min_block_size_px = 4;
  options.max_depth = 32;
  options.allow_rect = true;
  options.allow_split = false;
  return av2enc::partition::decide_frame(geometry, cost_fn, options);
}

FrameGeometry frame_geometry() {
  FrameGeometry geometry;
  geometry.width = kWidth;
  geometry.height = kHeight;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  return geometry;
}

int tx_size_for(int w, int h) {
  const int tx = av2enc::txv::tx_size_index(w, h);
  return tx;
}

int chroma_width(BlockSize size) {
  const BlockSize chroma =
      av2enc::partition::chroma_plane_residual_size(frame_geometry(), size);
  return av2enc::partition::block_width(chroma);
}
int chroma_height(BlockSize size) {
  const BlockSize chroma =
      av2enc::partition::chroma_plane_residual_size(frame_geometry(), size);
  return av2enc::partition::block_height(chroma);
}

// Flattens a PartitionNode tree into BlockSyntax, preserving decode order.
void collect(const PartitionNode& node, std::vector<BlockSyntax>& blocks) {
  const std::size_t index = blocks.size();
  BlockSyntax entry;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  std::uint8_t mask = 0;
  for (int p = 0; p < av2enc::partition::kPartitionTypeCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint8_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  if (node.forced) entry.flags |= av2enc::m2::syntax::kFlagForced;
  if (node.depth == 0) entry.flags |= av2enc::m2::syntax::kFlagSbRoot;
  blocks.push_back(entry);
  if (node.partition != av2enc::partition::PartitionType::kNone) {
    for (const PartitionNode& child : node.children) {
      collect(child, blocks);
    }
  }
  blocks[index].subtree =
      static_cast<std::uint16_t>(blocks.size() - index - 1);
}

// Forward transform + quantise + inverse for one transform block of the
// synthesised residual.  Returns the coded-size reconstruction residual.
std::vector<std::int16_t> code_residual(const av2enc::txv::TxParams& params,
                                        int seed,
                                        std::vector<std::int32_t>* levels,
                                        std::uint32_t* eob) {
  const int cw = params.coded_w;
  const int ch = params.coded_h;
  std::vector<std::int16_t> residual_coded(
      static_cast<std::size_t>(cw * ch));
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      int value = ((x * 17 + y * 9 + seed * 5) % 41) - 20;
      // Keep the pattern asymmetric so all three classes of coefficient
      // (DC/AC, row/column) see energy.
      if (((x + y + seed) & 3) == 0) value += 6;
      residual_coded[static_cast<std::size_t>(y * cw + x)] =
          static_cast<std::int16_t>(value);
    }
  }
  std::vector<std::int16_t> residual_adj(
      static_cast<std::size_t>(params.w * params.h));
  av2enc::txv::downsample_residual(residual_coded.data(), params,
                                   residual_adj.data());
  std::vector<double> coeff(
      static_cast<std::size_t>(params.w * params.h));
  levels->assign(coeff.size(), 0);
  av2enc::txv::forward_transform(residual_adj.data(), params, coeff.data(),
                                 levels->data());

  const int area = params.w * params.h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::m2::txpartition::get_scan(
      params.tx_size, av2enc::m2::txpartition::TxScanClass::k2D, scan);
  *eob = 0;
  for (int c = area - 1; c >= 0; --c) {
    const std::size_t pos =
        scan[static_cast<std::size_t>(c)];
    if ((*levels)[pos] != 0) {
      *eob = static_cast<std::uint32_t>(c + 1);
      break;
    }
  }
  // The gate uses base_q_idx 90, so qlookup(90) == kQuantStep and the default
  // txv inverse path is the normative §7.14.4 dequant + §7.15.4 inverse.
  std::vector<std::int32_t> dequant(
      static_cast<std::size_t>(params.w * params.h));
  std::vector<std::int16_t> recon(static_cast<std::size_t>(cw * ch));
  av2enc::txv::inverse_transform(levels->data(), params, dequant.data(),
                                 recon.data());
  return recon;
}

struct PlanInfo {
  av2enc::m2::txpartition::gate::GateSplitPlan plan;
  std::vector<std::int32_t> levels;
  int split_leaf_bsize = 0;
  int split_leaf_mi_row = 0;
  int split_leaf_mi_col = 0;
};

bool build_plan(const CaseConfig& config, const PartitionNode& root,
                PlanInfo* info, std::string& error) {
  // Find the first leaf in decode order.  The cost function above makes it a
  // `leaf x leaf` square at (0,0).
  const PartitionNode* first = &root;
  while (first->partition != av2enc::partition::PartitionType::kNone) {
    if (first->children.empty()) {
      error = "partition tree has an empty internal node";
      return false;
    }
    first = &first->children.front();
  }
  const int leaf_w = av2enc::partition::block_width(first->size);
  const int leaf_h = av2enc::partition::block_height(first->size);
  if (leaf_w != config.leaf || leaf_h != config.leaf) {
    error = "unexpected first leaf size " + std::to_string(leaf_w) + "x" +
            std::to_string(leaf_h);
    return false;
  }

  info->split_leaf_bsize = static_cast<int>(first->size);
  info->split_leaf_mi_row = first->mi_row;
  info->split_leaf_mi_col = first->mi_col;

  SplitGeometry geometry;
  error.clear();
  if (!av2enc::m2::txpartition::tx_partition_geometry(
          info->split_leaf_bsize, first->mi_row, first->mi_col,
          config.partition, &geometry, &error)) {
    return false;
  }

  info->plan = av2enc::m2::txpartition::gate::GateSplitPlan{};
  info->plan.active = true;
  info->plan.mi_row = static_cast<std::uint16_t>(first->mi_row);
  info->plan.mi_col = static_cast<std::uint16_t>(first->mi_col);
  info->plan.mi_size = static_cast<std::uint8_t>(first->size);
  info->plan.partition = static_cast<std::uint8_t>(config.partition);
  info->plan.count = geometry.count;

  for (int i = 0; i < geometry.count; ++i) {
    const auto& part = geometry.parts[i];
    const auto params = av2enc::txv::make_tx_params(
        static_cast<av2enc::txv::TxSize>(part.tx_size),
        av2enc::txv::TxType::kDctDct);
    if (!params.has_value()) {
      error = "unsupported transform size in the split plan";
      return false;
    }
    std::vector<std::int32_t> block_levels;
    std::uint32_t eob = 0;
    const std::vector<std::int16_t> recon =
        code_residual(*params, i, &block_levels, &eob);

    auto& gate_sub = info->plan.subs[i];
    gate_sub.mi_row = static_cast<std::uint16_t>(part.mi_row);
    gate_sub.mi_col = static_cast<std::uint16_t>(part.mi_col);
    gate_sub.tx_size = static_cast<std::uint8_t>(part.tx_size);
    gate_sub.eob = eob;
    gate_sub.levels_offset = static_cast<std::uint32_t>(info->levels.size());
    if (eob > 0) {
      std::vector<std::uint16_t> scan(
          static_cast<std::size_t>(params->w * params->h));
      av2enc::m2::txpartition::get_scan(
          part.tx_size, av2enc::m2::txpartition::TxScanClass::k2D, scan);
      for (std::uint32_t c = 0; c < eob; ++c) {
        info->levels.push_back(
            block_levels[static_cast<std::size_t>(
                scan[static_cast<std::size_t>(c)])]);
      }
    }
    (void)recon;
  }
  return true;
}

void fill_keyframe_block(BlockSyntax* node) {
  const auto size = static_cast<BlockSize>(node->bsize);
  node->is_inter = 0;
  node->skip_flag = 0;
  node->y_mode = 0;   // DC_PRED
  node->uv_mode = 0;  // DC_PRED
  node->y_angle_delta = 0;
  node->tx_size[0] = static_cast<std::uint8_t>(
      tx_size_for(av2enc::partition::block_width(size),
                  av2enc::partition::block_height(size)));
  node->tx_size[1] = static_cast<std::uint8_t>(
      tx_size_for(chroma_width(size), chroma_height(size)));
  node->tx_size[2] = node->tx_size[1];
  node->eob[0] = node->eob[1] = node->eob[2] = 0;
  node->levels_offset = 0;
}

void fill_inter_skip_block(BlockSyntax* node) {
  const auto size = static_cast<BlockSize>(node->bsize);
  node->is_inter = 1;
  node->skip_flag = 1;
  node->inter_mode = av2enc::m2::syntax::kInterGlobalMv;
  node->drl_index = 0;
  node->mv_y = node->mv_x = 0;
  node->mvd_y = node->mvd_x = 0;
  node->tx_size[0] = static_cast<std::uint8_t>(
      tx_size_for(av2enc::partition::block_width(size),
                  av2enc::partition::block_height(size)));
  node->tx_size[1] = static_cast<std::uint8_t>(
      tx_size_for(chroma_width(size), chroma_height(size)));
  node->tx_size[2] = node->tx_size[1];
  node->eob[0] = node->eob[1] = node->eob[2] = 0;
  node->levels_offset = 0;
}

void fill_inter_split_block(BlockSyntax* node) {
  fill_inter_skip_block(node);
  node->skip_flag = 0;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

bool read_decoded(const std::string& path, std::vector<FramePlanes>* frames,
                  std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    FramePlanes planes;
    planes.y.assign(frame.y().begin(), frame.y().end());
    planes.u.assign(frame.u().begin(), frame.u().end());
    planes.v.assign(frame.v().begin(), frame.v().end());
    frames->push_back(std::move(planes));
  }
  return true;
}

bool compare_plane(const std::vector<std::uint8_t>& got,
                   const std::vector<std::uint8_t>& want, int plane,
                   int frame_index, long long* mismatches, int* max_diff) {
  if (got.size() != want.size()) {
    std::cerr << "  plane " << plane << " size mismatch\n";
    return false;
  }
  bool ok = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const int diff = std::abs(static_cast<int>(got[i]) -
                              static_cast<int>(want[i]));
    if (diff != 0) {
      if (*mismatches < 5) {
        std::cerr << "  MISMATCH frame " << frame_index << " plane " << plane
                  << " sample " << i << ": got " << static_cast<int>(got[i])
                  << " want " << static_cast<int>(want[i]) << "\n";
      }
      ++*mismatches;
      ok = false;
    }
    *max_diff = std::max(*max_diff, diff);
  }
  return ok;
}

bool run_case(const CaseConfig& config, const std::string& out_path,
              const std::string& dav2d) {
  std::cout << "== case " << config.id << ": " << config.name << " ==\n";
  const av2enc::partition::FrameDecision decision = decide_tree(config.leaf);
  if (decision.status != av2enc::partition::DecisionStatus::kOk) {
    std::cerr << "partition decision failed: " << decision.error << "\n";
    return false;
  }
  std::string error;
  if (!av2enc::partition::validate_frame(frame_geometry(), decision, error)) {
    std::cerr << "partition tree invalid: " << error << "\n";
    return false;
  }

  M2Config m2config;
  m2config.max_frame_width = kWidth;
  m2config.max_frame_height = kHeight;
  m2config.base_q_idx = kBaseQIdx;
  m2config.order_hint_bits = 4;
  m2config.num_ref_frames = 2;
  m2config.num_total_refs = 1;
  m2config.max_drl_bits_minus_1 = 4;
  m2config.disable_cdf_update = true;
  m2config.enable_cfl_intra = true;

  FramePlan plan0;
  plan0.width = kWidth;
  plan0.height = kHeight;
  plan0.sb_size = 64;
  plan0.inter_frame = false;

  std::vector<BlockSyntax> blocks0;
  for (const PartitionNode& sb : decision.superblocks) collect(sb, blocks0);
  for (BlockSyntax& node : blocks0) {
    if ((node.flags & av2enc::m2::syntax::kFlagForced) == 0 &&
        node.partition == 0) {
      fill_keyframe_block(&node);
    } else if (node.partition == 0) {
      fill_keyframe_block(&node);
    }
  }

  std::vector<std::int32_t> levels0;
  std::vector<std::uint8_t> tile0;
  av2enc::m2::txpartition::gate::reset(nullptr, false);
  if (!av2enc::m2::syntax::assemble_m2_frame(plan0, blocks0, levels0, m2config,
                                             tile0, error)) {
    std::cerr << "keyframe assemble failed: " << error << "\n";
    return false;
  }

  PlanInfo plan_info;
  if (!build_plan(config, decision.superblocks.front(), &plan_info, error)) {
    std::cerr << "split plan failed: " << error << "\n";
    return false;
  }

  FramePlan plan1;
  plan1.width = kWidth;
  plan1.height = kHeight;
  plan1.sb_size = 64;
  plan1.inter_frame = true;
  std::vector<BlockSyntax> blocks1;
  for (const PartitionNode& sb : decision.superblocks) collect(sb, blocks1);
  int target_index = -1;
  for (std::size_t i = 0; i < blocks1.size(); ++i) {
    if (blocks1[i].partition != 0) continue;
    fill_inter_skip_block(&blocks1[i]);
    if (target_index < 0 && blocks1[i].mi_row == plan_info.plan.mi_row &&
        blocks1[i].mi_col == plan_info.plan.mi_col &&
        blocks1[i].bsize == plan_info.plan.mi_size) {
      target_index = static_cast<int>(i);
    }
  }
  if (target_index < 0) {
    std::cerr << "target leaf not found in the block list\n";
    return false;
  }
  if (config.partition != TxPartition::kNone) {
    fill_inter_split_block(&blocks1[static_cast<std::size_t>(target_index)]);
  }

  av2enc::m2::txpartition::gate::reset(&plan_info.plan, !config.negative);
  std::vector<std::uint8_t> tile1;
  if (!av2enc::m2::syntax::assemble_m2_frame(
          plan1, blocks1, plan_info.levels, m2config, tile1, error)) {
    std::cerr << "P-frame assemble failed: " << error << "\n";
    return false;
  }

  // --- reconstruction ------------------------------------------------------
  FramePlanes recon0;
  recon0.y.assign(static_cast<std::size_t>(kWidth * kHeight), 128);
  recon0.u.assign(static_cast<std::size_t>(kChromaWidth * kChromaHeight), 128);
  recon0.v.assign(static_cast<std::size_t>(kChromaWidth * kChromaHeight), 128);
  FramePlanes recon1 = recon0;

  if (config.partition != TxPartition::kNone) {
    for (int i = 0; i < plan_info.plan.count; ++i) {
      const auto& sub = plan_info.plan.subs[i];
      const auto params = av2enc::txv::make_tx_params(
          static_cast<av2enc::txv::TxSize>(sub.tx_size),
          av2enc::txv::TxType::kDctDct);
      if (!params.has_value()) {
        std::cerr << "invalid split tx size\n";
        return false;
      }
      // Re-run the same forward/quantise/inverse the writer used.
      std::vector<std::int32_t> levels;
      std::uint32_t eob = 0;
      const std::vector<std::int16_t> recon =
          code_residual(*params, i, &levels, &eob);
      const int x0 = sub.mi_col * 4;
      const int y0 = sub.mi_row * 4;
      for (int y = 0; y < params->coded_h; ++y) {
        for (int x = 0; x < params->coded_w; ++x) {
          const int value =
              clip1(128 + recon[static_cast<std::size_t>(
                                y * params->coded_w + x)]);
          recon1.y[static_cast<std::size_t>((y0 + y) * kWidth + x0 + x)] =
              static_cast<std::uint8_t>(value);
        }
      }
    }
  }

  // --- container -----------------------------------------------------------
  M2FrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.ref_slot = 0;
  key_header.refresh_frame_flags =
      (1U << m2config.num_ref_frames) - 1U;
  const std::vector<std::uint8_t> key_obu =
      av2enc::m2::make_frame_obu(m2config, key_header, tile0);

  M2FrameHeader p_header;
  p_header.keyframe = false;
  p_header.order_hint = 1;
  p_header.ref_slot = 0;
  p_header.refresh_frame_flags = 1;
  // reset() above already enabled the per-frame tx_mode_select bit.
  const std::vector<std::uint8_t> p_obu =
      av2enc::m2::make_frame_obu(m2config, p_header, tile1);

  std::vector<std::vector<std::uint8_t>> units(2);
  {
    const std::vector<std::uint8_t> td =
        av2enc::m2::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq =
        av2enc::m2::make_sequence_header_obu(m2config);
    units[0].insert(units[0].end(), td.begin(), td.end());
    units[0].insert(units[0].end(), seq.begin(), seq.end());
    units[0].insert(units[0].end(), key_obu.begin(), key_obu.end());
  }
  {
    const std::vector<std::uint8_t> td =
        av2enc::m2::make_temporal_delimiter_obu();
    units[1].insert(units[1].end(), td.begin(), td.end());
    units[1].insert(units[1].end(), p_obu.begin(), p_obu.end());
  }

  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = kWidth;
  header.height = kHeight;
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 2;
  if (!writer.open(out_path, header, error)) {
    std::cerr << "IVF open failed: " << error << "\n";
    return false;
  }
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) {
      std::cerr << "IVF write failed: " << error << "\n";
      return false;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "IVF close failed: " << error << "\n";
    return false;
  }

  std::cout << "  tile0=" << tile0.size() << "B tile1=" << tile1.size()
            << "B frame0=" << units[0].size() << "B frame1="
            << units[1].size() << "B\n";
  std::cout << "  split block: " << plan_info.plan.count
            << " transform block(s), tx sizes";
  int nonzero_parts = 0;
  for (int i = 0; i < plan_info.plan.count; ++i) {
    std::cout << " " << av2enc::txv::tx_size_name(
        static_cast<av2enc::txv::TxSize>(plan_info.plan.subs[i].tx_size));
    if (plan_info.plan.subs[i].eob > 0) ++nonzero_parts;
  }
  std::cout << " (eob>0 in " << nonzero_parts << "/" << plan_info.plan.count
            << ")\n";
  if (nonzero_parts != plan_info.plan.count) {
    std::cerr << "GATE FAIL: a transform block coded no coefficients; the "
                 "split would not be reconstructed from real data\n";
    return false;
  }

  if (dav2d.empty()) {
    std::cerr << "DECODE SKIP: no dav2d binary found (set DAV2D)\n";
    return false;
  }
  const std::string decoded = out_path + ".decoded.y4m";
  const std::string command =
      dav2d + " -q -i " + out_path + " -o " + decoded +
      " --strict 1 --threads 1";
  std::cout << "  decode: " << command << "\n";
  const int decode_rc = std::system(command.c_str());
  if (config.negative) {
    if (decode_rc != 0) {
      std::cout << "  NEGATIVE CONTROL PASS: decoder rejected the stream "
                   "(tx_mode_select = 0, split syntax present)\n";
      return true;
    }
    std::vector<FramePlanes> neg_frames;
    if (!read_decoded(decoded, &neg_frames, error) || neg_frames.size() != 2) {
      std::cout << "  NEGATIVE CONTROL PASS: decoder rejected the P frame "
                   "(" << neg_frames.size() << " frame(s) decoded)\n";
      return true;
    }
    long long neg_mismatches = 0;
    int neg_max_diff = 0;
    bool same = compare_plane(neg_frames[0].y, recon0.y, 0, 0,
                              &neg_mismatches, &neg_max_diff);
    same &= compare_plane(neg_frames[0].u, recon0.u, 1, 0, &neg_mismatches,
                          &neg_max_diff);
    same &= compare_plane(neg_frames[0].v, recon0.v, 2, 0, &neg_mismatches,
                          &neg_max_diff);
    same &= compare_plane(neg_frames[1].y, recon1.y, 0, 1, &neg_mismatches,
                          &neg_max_diff);
    same &= compare_plane(neg_frames[1].u, recon1.u, 1, 1, &neg_mismatches,
                          &neg_max_diff);
    same &= compare_plane(neg_frames[1].v, recon1.v, 2, 1, &neg_mismatches,
                          &neg_max_diff);
    if (!same) {
      std::cout << "  NEGATIVE CONTROL PASS: recon diverges (mismatches="
                << neg_mismatches << " max_diff=" << neg_max_diff << ")\n";
      return true;
    }
    std::cerr << "NEGATIVE CONTROL FAIL: stream with tx_mode_select=0 still "
                 "reconstructed exactly\n";
    return false;
  }
  if (decode_rc != 0) {
    std::cerr << "DECODE FAIL\n";
    return false;
  }

  std::vector<FramePlanes> frames;
  if (!read_decoded(decoded, &frames, error)) {
    std::cerr << "decoded Y4M read failed: " << error << "\n";
    return false;
  }
  if (frames.size() != 2) {
    std::cerr << "expected 2 decoded frames, got " << frames.size() << "\n";
    return false;
  }
  long long mismatches = 0;
  int max_diff = 0;
  bool ok = true;
  ok &= compare_plane(frames[0].y, recon0.y, 0, 0, &mismatches, &max_diff);
  ok &= compare_plane(frames[0].u, recon0.u, 1, 0, &mismatches, &max_diff);
  ok &= compare_plane(frames[0].v, recon0.v, 2, 0, &mismatches, &max_diff);
  ok &= compare_plane(frames[1].y, recon1.y, 0, 1, &mismatches, &max_diff);
  ok &= compare_plane(frames[1].u, recon1.u, 1, 1, &mismatches, &max_diff);
  ok &= compare_plane(frames[1].v, recon1.v, 2, 1, &mismatches, &max_diff);
  if (!ok) {
    std::cerr << "RECON MISMATCH: mismatches=" << mismatches
              << " max_diff=" << max_diff << "\n";
    return false;
  }
  std::cout << "  PASS: dav2d --strict 1 --threads 1, 2 frames, recon exact\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_dir = ".";
  std::string only;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--case" && i + 1 < argc) {
      only = argv[++i];
    } else if (arg == "--help") {
      std::printf("usage: %s [--out-dir DIR] [--case A|B|C|D|E|F]\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  std::filesystem::create_directories(out_dir);

  const CaseConfig cases[] = {
      {'A', "64x64 block, SPLIT -> 4x 32x32", 64, TxPartition::kSplit},
      {'B', "16x16 block, SPLIT -> 4x 8x8", 16, TxPartition::kSplit},
      {'C', "8x8 block, SPLIT -> 4x 4x4", 8, TxPartition::kSplit},
      {'D', "16x16 block, HORZ -> 2x 16x8", 16, TxPartition::kHorz},
      {'E', "16x16 block, VERT -> 2x 8x16", 16, TxPartition::kVert},
      {'F', "16x16 block, NONE (control)", 16, TxPartition::kNone},
      {'G', "16x16 SPLIT with tx_mode_select=0 (negative control)", 16,
       TxPartition::kSplit, true},
  };

  const std::string dav2d = find_dav2d();
  int failures = 0;
  int ran = 0;
  for (const CaseConfig& config : cases) {
    if (!only.empty() && only[0] != config.id) continue;
    ++ran;
    const std::string path =
        (std::filesystem::path(out_dir) /
         (std::string("tx_partition_") + config.id + ".ivf"))
            .string();
    if (!run_case(config, path, dav2d)) ++failures;
  }
  if (ran == 0) {
    std::cerr << "no cases matched --case " << only << "\n";
    return 2;
  }
  std::printf("\n%s: %d/%d case(s) passed\n",
              failures == 0 ? "TX PARTITION GATE PASS"
                            : "TX PARTITION GATE FAIL",
              ran - failures, ran);
  return failures == 0 ? 0 : 1;
}
