// Unit tests for src/encoder/m2/deltaq/ (M2 frame/block glue).
//
// The normative syntax is tested in src/encoder/m1/deltaq/.  Here we test the
// ordering the M2 writer must use and the FrameQpContext wiring.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "codec/entropy/deltaq/still/deltaq.h"
#include "codec/entropy/deltaq/video/video_deltaq.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "codec/partition/tree/partition.h"
#include "codec/rate/rc/rc.h"

namespace {

using namespace av2enc::video::deltaq;
namespace core = av2enc::deltaq;
namespace rc = av2enc::rc;
namespace syntax = av2enc::video::syntax;
namespace partition = av2enc::partition;
using av2enc::video::VideoFrameHeader;

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

#define CHECK_EQ(a, b)                                                       \
  do {                                                                       \
    ++g_checks;                                                              \
    const auto va = (a);                                                     \
    const auto vb = (b);                                                     \
    if (!(va == vb)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, \
                   __LINE__, #a, #b, static_cast<long long>(va),              \
                   static_cast<long long>(vb));                              \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

std::vector<std::uint8_t> finish(av2enc::BitWriter& writer) {
  writer.align_to_byte();
  const std::span<const std::uint8_t> bytes = writer.bytes();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

syntax::BlockSyntax make_node(int mi_row, int mi_col,
                              partition::BlockSize size) {
  syntax::BlockSyntax node;
  node.mi_row = static_cast<std::uint16_t>(mi_row);
  node.mi_col = static_cast<std::uint16_t>(mi_col);
  node.bsize = static_cast<std::uint8_t>(size);
  node.partition = static_cast<std::uint8_t>(partition::PartitionType::kNone);
  node.flags = syntax::kFlagSbRoot | syntax::kFlagForced;
  return node;
}

FrameQpPlan make_plan(bool inter, bool temporal) {
  FrameQpPlan plan;
  plan.base_q_idx = 90;
  plan.delta_q.present = true;
  plan.delta_q.res = 1;
  std::string error;
  CHECK(rc::set_segment_alt_q(plan.segmentation, 1, 12, &error));
  CHECK(error.empty());
  plan.segmentation.update_map = true;
  plan.segmentation.temporal_update = temporal;
  // A real M2 P frame referencing a key frame has DerivedPrimaryRefFrame ==
  // PRIMARY_REF_NONE; tests that exercise the coded update_map bits pass
  // `inter == true` and set `primary_ref_is_none = false` explicitly.
  plan.primary_ref_is_none = !inter;
  return plan;
}

void test_frame_qp_header() {
  CHECK(primary_ref_is_none_for(false));
  CHECK(!primary_ref_is_none_for(true));

  FrameQpPlan key_plan;
  key_plan.primary_ref_is_none = true;
  const core::FrameQpHeader key_header = frame_qp_header(key_plan);
  CHECK(key_header.primary_ref_is_none);
  CHECK(!key_header.enable_ext_seg);
  CHECK(!key_header.frame_type_switch);

  FrameQpPlan inter_plan;
  inter_plan.primary_ref_is_none = false;
  inter_plan.enable_ext_seg = true;
  const core::FrameQpHeader inter_header = frame_qp_header(inter_plan);
  CHECK(!inter_header.primary_ref_is_none);
  CHECK(inter_header.enable_ext_seg);
}

void test_begin_frame_validation() {
  VideoFrameHeader header;
  header.keyframe = false;
  FrameQpPlan plan = make_plan(true, false);
  // P frame referencing a key frame: primary reference is NONE.
  plan.primary_ref_is_none = true;
  FrameQpContext context;
  std::string error;
  CHECK(begin_frame(context, plan, header, 16, 16, {}, &error));
  CHECK(error.empty());
  CHECK(context.tile_qp.current_q_index == 90);
  CHECK(context.segmentation.enabled);
  CHECK_EQ(context.segmentation.num_segments, 8);
  CHECK_EQ(context.grid.mi_cols, 16);

  plan.base_q_idx = 0;
  CHECK(!begin_frame(context, plan, header, 16, 16, {}, &error));
  CHECK(!error.empty());
  error.clear();

  plan = make_plan(true, false);
  plan.enable_ext_seg = false;
  // Activate segment 9 while MaxSegments is 8.
  CHECK(rc::set_segment_alt_q(plan.segmentation, 9, 1, &error));
  CHECK(!begin_frame(context, plan, header, 16, 16, {}, &error));
  CHECK(!error.empty());
}

void test_frame_syntax_matches_core() {
  VideoFrameHeader key;
  key.keyframe = true;
  FrameQpPlan plan = make_plan(false, false);
  av2enc::BitWriter writer;
  std::string error;
  const std::size_t bits = write_frame_qp_syntax(writer, plan, key, &error);
  CHECK(error.empty());
  const std::vector<std::uint8_t> bytes = finish(writer);

  const core::FrameQpHeader header = frame_qp_header(plan);
  CHECK(header.primary_ref_is_none);
  av2enc::BitWriter reference;
  const std::size_t ref_bits = core::write_segmentation_and_delta_q(
      reference, plan.segmentation, plan.delta_q, plan.base_q_idx, header,
      &error);
  CHECK(error.empty());
  CHECK_EQ(bits, ref_bits);
  CHECK(bytes == finish(reference));

  core::BitReader reader(bytes);
  rc::SegmentationParams read_seg;
  rc::DeltaQParams read_delta;
  bool using_qmatrix = true;
  CHECK(core::read_segmentation_and_delta_q(reader, plan.base_q_idx, header,
                                            read_seg, read_delta,
                                            using_qmatrix, &error));
  CHECK(error.empty());
  CHECK(read_seg.enabled);
  CHECK(read_delta.present);
  CHECK_EQ(read_delta.res, 1);
  CHECK(read_seg.segments[1].enabled[rc::kSegLvlAltQ]);
  CHECK_EQ(read_seg.segments[1].data[rc::kSegLvlAltQ], 12);

  // Key frame: update_map inferred 1, temporal 0.
  core::FrameSegmentation derived;
  CHECK(core::derive_frame_segmentation(read_seg, read_delta, plan.base_q_idx,
                                        header, derived, &error));
  CHECK(derived.update_map);
  CHECK(!derived.temporal_update);

  // Inter frame: update_map and temporal_update bits are present.
  VideoFrameHeader inter;
  inter.keyframe = false;
  FrameQpPlan temporal_plan = make_plan(true, true);
  temporal_plan.primary_ref_is_none = false;
  av2enc::BitWriter inter_writer;
  const std::size_t inter_bits =
      write_frame_qp_syntax(inter_writer, temporal_plan, inter, &error);
  CHECK(error.empty());
  CHECK_EQ(inter_bits, bits + 2U);

  // A P frame whose primary reference is a key frame: DerivedPrimaryRefFrame
  // is PRIMARY_REF_NONE, so the update_map/temporal_update bits are absent.
  // The header is identical to the key-frame form for the same features.
  FrameQpPlan p_key_ref = make_plan(true, false);
  p_key_ref.primary_ref_is_none = true;
  av2enc::BitWriter p_writer;
  const std::size_t p_bits =
      write_frame_qp_syntax(p_writer, p_key_ref, inter, &error);
  CHECK(error.empty());
  CHECK_EQ(p_bits, bits);
  CHECK(finish(p_writer) == bytes);

  // A key frame cannot have a primary reference.
  FrameQpPlan bad = make_plan(false, false);
  bad.primary_ref_is_none = false;
  FrameQpContext bad_context;
  CHECK(!begin_frame(bad_context, bad, key, 16, 16, {}, &error));
  CHECK(!error.empty());
}

void test_inter_block_order() {
  VideoFrameHeader header;
  header.keyframe = false;
  FrameQpPlan plan = make_plan(true, false);
  // P frame referencing a key frame: primary reference is NONE.
  plan.primary_ref_is_none = true;
  FrameQpContext context;
  std::string error;
  CHECK(begin_frame(context, plan, header, 16, 16, {}, &error));

  std::vector<std::uint8_t> buffer(1U << 14, 0U);
  av2enc::av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
        av2enc::av2::Status::kOk);

  const syntax::BlockSyntax node =
      make_node(0, 0, partition::BlockSize::k64x64);
  CHECK(mi_size_is_sb(node, 64));
  CHECK(!mi_size_is_sb(make_node(0, 0, partition::BlockSize::k32x32), 64));

  begin_superblock(context, 0);
  CHECK(context.tile_qp.read_deltas);
  // §5.20.7.1 order: pre-skip id (no-op), is_inter/skip, post-skip id, delta.
  CHECK(write_inter_segment_id_pre(context, encoder, node, 0, false, &error));
  CHECK(error.empty());
  CHECK(context.tile_qp.read_deltas);  // segment id did not touch ReadDeltas
  CHECK(write_inter_segment_id_post(context, encoder, node, 1, false, &error));
  CHECK(error.empty());
  CHECK(write_delta_qindex(context, encoder, node, false, 2, &error));
  CHECK(error.empty());
  CHECK_EQ(context.tile_qp.current_q_index, 94);  // 90 + (2 << 1)
  CHECK(!context.tile_qp.read_deltas);
  CHECK(commit_segment_block(context, node, 1, &error));
  CHECK(error.empty());

  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk);
  buffer.resize(payload_bytes);

  // Independent decoder pass with the same ordering.
  FrameQpContext dec;
  CHECK(begin_frame(dec, plan, header, 16, 16, {}, &error));
  av2enc::av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2enc::av2::Status::kOk);
  begin_superblock(dec, 0);
  int id = -1;
  bool coded = true;
  CHECK(core::read_inter_segment_id(decoder, dec.cdfs, dec.segmentation,
                                    dec.grid, core::SegmentBlock{0, 0, 16, 16},
                                    true, false, id, coded, &error));
  CHECK(!coded);
  CHECK(core::read_inter_segment_id(decoder, dec.cdfs, dec.segmentation,
                                    dec.grid, core::SegmentBlock{0, 0, 16, 16},
                                    false, false, id, coded, &error));
  CHECK(error.empty());
  CHECK(coded);
  CHECK_EQ(id, 1);
  int reduced = 0;
  CHECK(core::read_delta_qindex(decoder, dec.cdfs.delta_q, dec.tile_qp, true,
                                false, reduced, &error));
  CHECK_EQ(reduced, 2);
  CHECK_EQ(dec.tile_qp.current_q_index, 94);
  CHECK(core::commit_segment_block(dec.grid, core::SegmentBlock{0, 0, 16, 16},
                                   id, &error));
  CHECK(decoder.exit_symbol() == av2enc::av2::Status::kOk);
  CHECK(dec.grid.current == context.grid.current);
}

void test_intra_block_order() {
  VideoFrameHeader header;
  header.keyframe = true;
  FrameQpPlan plan = make_plan(false, false);
  plan.delta_q.present = false;
  plan.delta_q.res = 0;
  FrameQpContext context;
  std::string error;
  CHECK(begin_frame(context, plan, header, 16, 16, {}, &error));

  std::vector<std::uint8_t> buffer(1U << 14, 0U);
  av2enc::av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
        av2enc::av2::Status::kOk);
  const syntax::BlockSyntax node =
      make_node(0, 0, partition::BlockSize::k64x64);
  begin_superblock(context, 0);
  // §5.20.5.2: intra_segment_id() then read_delta_qindex() (a no-op here
  // because delta_q_present == 0).
  CHECK(write_intra_segment_id(context, encoder, node, 1, false, &error));
  CHECK(error.empty());
  CHECK(write_delta_qindex(context, encoder, node, false, 7, &error));
  CHECK(error.empty());
  CHECK_EQ(context.tile_qp.current_q_index, 90);
  CHECK(commit_segment_block(context, node, 1, &error));

  std::size_t payload_bytes = 0;
  CHECK(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk);
  buffer.resize(payload_bytes);

  FrameQpContext dec;
  CHECK(begin_frame(dec, plan, header, 16, 16, {}, &error));
  av2enc::av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
        av2enc::av2::Status::kOk);
  begin_superblock(dec, 0);
  int id = -1;
  CHECK(core::read_intra_segment_id(decoder, dec.cdfs, dec.segmentation,
                                    dec.grid, core::SegmentBlock{0, 0, 16, 16},
                                    false, id, &error));
  CHECK_EQ(id, 1);
  int reduced = 0;
  CHECK(core::read_delta_qindex(decoder, dec.cdfs.delta_q, dec.tile_qp, true,
                                false, reduced, &error));
  CHECK_EQ(reduced, 0);
  CHECK(decoder.exit_symbol() == av2enc::av2::Status::kOk);
}

void test_superblock_plan_step() {
  VideoFrameHeader header;
  header.keyframe = false;
  FrameQpPlan plan = make_plan(true, false);
  plan.primary_ref_is_none = true;
  plan.segmentation = rc::SegmentationParams{};
  FrameQpContext context;
  std::string error;
  CHECK(begin_frame(context, plan, header, 16, 16, {}, &error));

  const rc::DeltaQPlanStep step{/*sb_x=*/0, /*sb_y=*/0,
                                /*reduced_delta=*/3,
                                /*qindex_before=*/90,
                                /*qindex_after=*/96};
  std::vector<std::uint8_t> buffer(4096U, 0U);
  av2enc::av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) ==
        av2enc::av2::Status::kOk);
  const syntax::BlockSyntax node =
      make_node(0, 0, partition::BlockSize::k64x64);
  begin_superblock(context, 0);
  CHECK(write_superblock_delta_q(context, encoder, step, node, false, &error));
  CHECK(error.empty());
  CHECK_EQ(context.tile_qp.current_q_index, 96);

  // A step whose qindex_before does not match the tile state is rejected.
  CHECK(encoder.exit_symbol(nullptr) == av2enc::av2::Status::kOk);
  begin_superblock(context, 0);
  const rc::DeltaQPlanStep stale{0, 0, 1, 90, 92};
  CHECK(!write_superblock_delta_q(context, encoder, stale, node, false,
                                  &error));
  CHECK(!error.empty());
}

void test_commit_validation() {
  VideoFrameHeader header;
  header.keyframe = false;
  FrameQpPlan plan = make_plan(true, false);
  // P frame referencing a key frame: primary reference is NONE.
  plan.primary_ref_is_none = true;
  FrameQpContext context;
  std::string error;
  CHECK(begin_frame(context, plan, header, 16, 16, {}, &error));
  const syntax::BlockSyntax node =
      make_node(0, 0, partition::BlockSize::k64x64);
  CHECK(!commit_segment_block(context, node, 8, &error));  // id >= MaxSegments
  CHECK(!error.empty());
  error.clear();
  CHECK(commit_segment_block(context, node, 1, &error));
  CHECK(context.grid.current[0] == 1);
  CHECK(context.grid.current[15 * 16 + 15] == 1);
}

}  // namespace

int main() {
  test_frame_qp_header();
  test_begin_frame_validation();
  test_frame_syntax_matches_core();
  test_inter_block_order();
  test_intra_block_order();
  test_superblock_plan_step();
  test_commit_validation();

  if (g_failures != 0) {
    std::fprintf(stderr, "%d/%d checks failed\n", g_failures, g_checks);
    return 1;
  }
  std::printf("ok: %d checks\n", g_checks);
  return 0;
}
