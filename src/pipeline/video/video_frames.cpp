#include "pipeline/video/video_frames.h"

#include "codec/inter/leading/leading.h"
#include "codec/inter/longterm/longterm.h"
#include "codec/inter/longterm/longterm_stream.h"
#include "codec/inter/random_access/random_access.h"
#include "codec/inter/sef/sef.h"
#include "pipeline/grain/grain_stream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::video::frames {
namespace {

namespace ra = inter::random_access;
namespace sef = inter::sef;

namespace lt = inter::longterm;

namespace gr = filters::grain;

// §5.14 film grain: the wrappers rebuild sequence/frame headers, so they plan
// one model per *output* picture (coding order) and insert the type-23 OBU
// immediately before that picture's frame OBU.  Non-output pictures (the
// hidden last picture of --video-sef, leading pictures, the OLK) code no
// film_grain_config() and carry no model OBU.
gr::GrainChromaIdc grain_chroma_idc_of(fmt::ChromaFormat chroma) {
  switch (chroma) {
    case fmt::ChromaFormat::k400:
      return gr::GrainChromaIdc::k400;
    case fmt::ChromaFormat::k422:
      return gr::GrainChromaIdc::k422;
    case fmt::ChromaFormat::k444:
      return gr::GrainChromaIdc::k444;
    case fmt::ChromaFormat::k420:
    default:
      return gr::GrainChromaIdc::k420;
  }
}

gr::GrainFrameView make_grain_view(const VideoInputFrame& frame) {
  gr::GrainFrameView view;
  const bool packed16 = !frame.y16.empty();
  const int sub_x = fmt::subsampling_x(frame.format.chroma);
  const bool has_chroma = fmt::has_chroma(frame.format.chroma);
  if (packed16) {
    view.y16 = frame.y16.data();
    if (has_chroma) {
      view.u16 = frame.u16.data();
      view.v16 = frame.v16.data();
    }
    view.bit_depth = frame.format.depth == fmt::BitDepth::k10 ? 10 : 8;
  } else {
    view.y = frame.y.data();
    if (has_chroma) {
      view.u = frame.u.data();
      view.v = frame.v.data();
    }
    view.bit_depth = 8;
  }
  view.width = static_cast<int>(frame.width);
  view.height = static_cast<int>(frame.height);
  view.y_stride = static_cast<int>(frame.width);
  view.c_stride = static_cast<int>(frame.width >> sub_x);
  return view;
}

// One planner per wrapper sequence, with the grain chroma idc of the coded
// format (§6.17.10.1 compares fgm_chroma_idc against chroma_format_idc).
gr::GrainStreamPlanner make_grain_planner(const VideoOptions& options) {
  gr::GrainPlanOptions grain_options;
  grain_options.chroma_idc = grain_chroma_idc_of(options.format.chroma);
  return gr::GrainStreamPlanner(grain_options);
}

bool plan_grain(const VideoInputFrame& frame, std::uint32_t index,
                gr::GrainStreamPlanner& planner, gr::GrainFramePlan& plan,
                std::string& error) {
  return planner.PlanFrame(make_grain_view(frame), index, /*tlayer_id=*/0,
                           /*mlayer_id=*/0, &plan, &error);
}

// Copies `grain` into a wrapper header's film_grain_config() fields.
template <typename Header>
void set_grain(Header& header, const gr::GrainFramePlan& plan) {
  header.grain.apply_grain = plan.apply_grain;
  header.grain.fgm_id = plan.fgm_id;
  header.grain.grain_seed = plan.grain_seed;
}

// Emits this picture's model OBU (if the planner allocated a new slot) before
// the `frame_ordinal`-th coded-frame OBU of `unit`.
bool insert_grain_obu(const gr::GrainFramePlan& plan, std::size_t frame_ordinal,
                      std::vector<std::uint8_t>& unit, std::string& error) {
  if (!plan.emit_model) return true;
  std::string grain_error;
  const std::vector<std::uint8_t> obu =
      gr::MakeFilmGrainObu(plan.obu, &grain_error);
  if (obu.empty()) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  std::vector<std::uint8_t> with_grain;
  if (!gr::InsertFilmGrainObuBeforeFrame(unit, frame_ordinal, obu, &with_grain,
                                         &grain_error)) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  unit.swap(with_grain);
  return true;
}

// One plain (flag-free) sub-encode: tile payloads in decode order plus the
// reconstruction of every picture the sub-stream would output.
struct PlainEncode {
  std::vector<std::vector<std::uint8_t>> tiles;
  std::vector<VideoReconFrame> recon;
  std::vector<std::uint32_t> tile_bytes;
  std::vector<std::uint32_t> inter_blocks;
  std::vector<std::uint32_t> intra_blocks;
};

VideoOptions plain_options(const VideoOptions& options) {
  VideoOptions plain = options;
  plain.long_term = false;
  plain.olk = false;
  plain.leading_frames = 0;
  plain.sef = false;
  plain.ra_switch = false;
  plain.ras = false;
  plain.hide_last_picture = false;
  // The wrappers keep only the tile payloads from these sub-encodes: their
  // sequence/frame headers are rebuilt by the component writers, which code
  // film_grain_params_present and film_grain_config() themselves.  Running the
  // §5.14 planner inside the sub-encode would emit OBUs that are immediately
  // discarded (and plan models for pictures the wrapper never outputs).
  plain.film_grain = false;
  return plain;
}

bool encode_plain(std::span<const VideoInputFrame> frames,
                  const VideoOptions& options, PlainEncode& out,
                  std::string& error) {
  VideoSequenceResult sequence;
  if (!encode_video_sequence(frames, plain_options(options), sequence,
                             error)) {
    return false;
  }
  out.tiles.clear();
  out.tiles.reserve(sequence.units.size());
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    const std::size_t count = sequence.tile_bytes[i];
    if (count == 0 || count > sequence.units[i].size()) {
      error = "plain sub-encode produced an invalid tile size";
      return false;
    }
    out.tiles.emplace_back(sequence.units[i].end() -
                               static_cast<std::ptrdiff_t>(count),
                           sequence.units[i].end());
  }
  out.recon = std::move(sequence.recon);
  out.tile_bytes = std::move(sequence.tile_bytes);
  out.inter_blocks = std::move(sequence.inter_blocks);
  out.intra_blocks = std::move(sequence.intra_blocks);
  return true;
}

lt::SequenceConfig long_term_config(const VideoOptions& options,
                                    std::uint32_t width,
                                    std::uint32_t height,
                                    std::uint8_t num_ref_frames) {
  lt::SequenceConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = num_ref_frames;
  config.long_term_frame_id_bits = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.enable_cfl_intra = true;
  config.film_grain_params_present = options.film_grain;
  // Pinned dav2d f9dafbc does not parse the §5.18.2 OLK long-term list; the
  // retained slot still survives because §5.18.2 only invalidates slots whose
  // RefLongTermId == -1.  See docs/negative-results.md.
  config.olk_long_term_list = lt::LongTermListSyntax::kOmitPinnedDav2d;
  return config;
}

void fill_long_term_result(const lt::AssembledSequence& assembled,
                           std::vector<std::uint32_t> tile_bytes,
                           std::vector<std::uint32_t> inter_blocks,
                           std::vector<std::uint32_t> intra_blocks,
                           std::vector<std::int32_t> output_index,
                           std::vector<VideoReconFrame> recon,
                           VideoSequenceResult& result) {
  result.units = assembled.units;
  result.recon = std::move(recon);
  result.frame_output_index = std::move(output_index);
  result.frame_bytes.clear();
  result.frame_bytes.reserve(assembled.frames.size());
  for (const lt::FrameEvent& frame : assembled.frames) {
    result.frame_bytes.push_back(static_cast<std::uint32_t>(frame.obu_bytes));
  }
  result.tile_bytes = std::move(tile_bytes);
  result.inter_blocks = std::move(inter_blocks);
  result.intra_blocks = std::move(intra_blocks);
}

// Closed-GOP long-term references: the default two-slot P chain, but the CLK
// carries LongTermId 0 and the sequence header signals
// long_term_frame_id_bits = 1.  Tile payloads are the plain stream's, so only
// the headers move.
bool encode_long_term_closed(std::span<const VideoInputFrame> frames,
                             const VideoOptions& options,
                             VideoSequenceResult& result, std::string& error) {
  // long_term_config() pins order_hint_bits = 4 and this path codes the frame
  // index as OrderHint (unlike the plain path, which masks with & 15). Reject
  // 17+ frames up front instead of failing on frame 17 after 16 were encoded.
  constexpr std::size_t kMaxOrderHintFrames = std::size_t{1} << 4;
  if (frames.size() > kMaxOrderHintFrames) {
    error = "--video-long-term codes OrderHint with OrderHintBits = 4, so it "
            "supports at most 16 coded frames (got " +
            std::to_string(frames.size()) + ")";
    return false;
  }
  PlainEncode plain;
  if (!encode_plain(frames, options, plain, error)) return false;
  const lt::SequenceConfig config =
      long_term_config(options, frames[0].width, frames[0].height,
                       /*num_ref_frames=*/2);
  std::vector<lt::TemporalUnitPlan> units;
  units.reserve(frames.size());
  gr::GrainStreamPlanner grain_planner = make_grain_planner(options);
  std::vector<gr::GrainFramePlan> grain_plans(frames.size());
  lt::FrameHeader key;
  if (!lt::plan_closed_loop_key(config, 0, /*long_term_id=*/0, &key, &error)) {
    return false;
  }
  if (options.film_grain &&
      !plan_grain(frames[0], 0, grain_planner, grain_plans[0], error)) {
    return false;
  }
  if (options.film_grain) set_grain(key, grain_plans[0]);
  units.emplace_back();
  units.back().frames.push_back(lt::PlannedFrame{key, plain.tiles[0]});
  for (std::size_t i = 1; i < frames.size(); ++i) {
    lt::FrameHeader header;
    const std::uint8_t ref_slot =
        static_cast<std::uint8_t>((i - 1U) % 2U);
    const std::uint8_t refresh_slot = static_cast<std::uint8_t>(i % 2U);
    const std::vector<std::uint8_t> refs{ref_slot};
    if (!lt::plan_inter_frame(config, lt::FrameRole::kRegular,
                              static_cast<std::uint32_t>(i), refs,
                              refresh_slot, true, &header, &error)) {
      return false;
    }
    if (options.film_grain &&
        !plan_grain(frames[i], static_cast<std::uint32_t>(i), grain_planner,
                    grain_plans[i], error)) {
      return false;
    }
    if (options.film_grain) set_grain(header, grain_plans[i]);
    units.emplace_back();
    units.back().frames.push_back(lt::PlannedFrame{header, plain.tiles[i]});
  }
  lt::AssembledSequence assembled;
  if (!lt::assemble_sequence(config, units, &assembled, &error)) return false;
  if (options.film_grain) {
    // One coded frame per unit, so each model OBU goes immediately before it.
    for (std::size_t i = 0; i < assembled.units.size(); ++i) {
      if (!insert_grain_obu(grain_plans[i], 0, assembled.units[i], error)) {
        return false;
      }
    }
  }
  std::vector<std::int32_t> output_index;
  output_index.reserve(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    output_index.push_back(static_cast<std::int32_t>(i));
  }
  fill_long_term_result(assembled, plain.tile_bytes, plain.inter_blocks,
                        plain.intra_blocks, std::move(output_index),
                        std::move(plain.recon), result);
  return true;
}

// Open random access (open-loop key).  Stream shape, four coded pictures:
//
//   TU0  CLK f0   LongTermId 0, refreshes all slots
//   TU1  OLK f1   slot 1 only, non-output; P2 f2 predicts from the OLK
//   TU2  P3  f3   references slot 0, the long-term frame, across the RAP
//
// §7.4.4 keeps slot 0 because RefLongTermId != -1 when the first Regular frame
// after the OLK arrives.  The decoder must output f0, f2, f3 in that order.
bool encode_olk_sequence(std::span<const VideoInputFrame> frames,
                         const VideoOptions& options,
                         VideoSequenceResult& result, std::string& error) {
  if (frames.size() != 4) {
    error = "--video-olk requires exactly 4 input frames "
            "(CLK, OLK, P2, P3)";
    return false;
  }
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  PlainEncode key0;
  PlainEncode olk_key;
  PlainEncode olk_p2;
  PlainEncode p3;
  if (!encode_plain(frames.subspan(0, 1), options, key0, error) ||
      !encode_plain(frames.subspan(1, 1), options, olk_key, error) ||
      !encode_plain(frames.subspan(1, 2), options, olk_p2, error)) {
    return false;
  }
  const VideoInputFrame p3_inputs[2] = {frames[0], frames[3]};
  if (!encode_plain(std::span<const VideoInputFrame>(p3_inputs, 2), options,
                    p3, error)) {
    return false;
  }
  const lt::SequenceConfig config =
      long_term_config(options, width, height, /*num_ref_frames=*/4);
  lt::PoolState pool;
  lt::reset_pool(config, &pool);

  gr::GrainStreamPlanner grain_planner = make_grain_planner(options);
  gr::GrainFramePlan grain_clk;
  gr::GrainFramePlan grain_p2;
  gr::GrainFramePlan grain_p3;
  lt::FrameHeader clk;
  if (!lt::plan_closed_loop_key(config, 0, /*long_term_id=*/0, &clk,
                                &error)) {
    return false;
  }
  if (options.film_grain &&
      !plan_grain(frames[0], 0, grain_planner, grain_clk, error)) {
    return false;
  }
  if (options.film_grain) set_grain(clk, grain_clk);
  lt::PoolEvents events;
  if (!lt::apply_frame(config, clk, /*first_picture_in_tu=*/true, &pool,
                       &events, &error)) {
    return false;
  }
  lt::FrameHeader olk;
  if (!lt::plan_open_loop_key(config, pool, /*order_hint=*/1, /*olk_slot=*/1,
                              /*olk_long_term_id=*/-1, {}, &olk, &error)) {
    return false;
  }
  // §7.3.8.2: the OLK is a random access point, so a decoder that starts there
  // cannot rely on a film grain model first sent in the CLK unit.  Plan the
  // post-RAP pictures with a fresh planner so P2 always re-emits its model in
  // the OLK temporal unit (the OLK itself is non-output and codes no config).
  gr::GrainStreamPlanner rap_grain_planner = make_grain_planner(options);
  lt::FrameHeader p2;
  {
    const std::vector<std::uint8_t> refs{1};
    if (!lt::plan_inter_frame(config, lt::FrameRole::kRegular,
                              /*order_hint=*/2, refs, /*refresh_slot=*/2,
                              true, &p2, &error)) {
      return false;
    }
    // Coding order is CLK, OLK (hidden), P2, P3: the planner sees the output
    // pictures in that order.  The OLK is not output, so it codes no config.
    if (options.film_grain &&
        !plan_grain(frames[2], 2, rap_grain_planner, grain_p2, error)) {
      return false;
    }
    if (options.film_grain) set_grain(p2, grain_p2);
  }
  lt::FrameHeader p3_header;
  {
    const std::vector<std::uint8_t> refs{0};
    if (!lt::plan_inter_frame(config, lt::FrameRole::kRegular,
                              /*order_hint=*/3, refs, /*refresh_slot=*/3,
                              true, &p3_header, &error)) {
      return false;
    }
    if (options.film_grain &&
        !plan_grain(frames[3], 3, rap_grain_planner, grain_p3, error)) {
      return false;
    }
    if (options.film_grain) set_grain(p3_header, grain_p3);
  }
  std::vector<lt::TemporalUnitPlan> units(3);
  units[0].frames.push_back(lt::PlannedFrame{clk, key0.tiles[0]});
  units[1].frames.push_back(lt::PlannedFrame{olk, olk_key.tiles[0]});
  units[1].frames.push_back(lt::PlannedFrame{p2, olk_p2.tiles[1]});
  units[2].frames.push_back(lt::PlannedFrame{p3_header, p3.tiles[1]});
  lt::AssembledSequence assembled;
  if (!lt::assemble_sequence(config, units, &assembled, &error)) return false;
  if (options.film_grain) {
    if (assembled.units.size() != 3U) {
      error = "OLK sequence assembled an unexpected unit count";
      return false;
    }
    // Unit 1 carries the hidden OLK first and P2 second (§7.3.3/.4).
    if (!insert_grain_obu(grain_clk, 0, assembled.units[0], error) ||
        !insert_grain_obu(grain_p2, 1, assembled.units[1], error) ||
        !insert_grain_obu(grain_p3, 0, assembled.units[2], error)) {
      return false;
    }
  }

  // Coded order: CLK, OLK (hidden), P2, P3.  Output order: f0, f2, f3.
  std::vector<std::int32_t> output_index{0, -1, 1, 2};
  std::vector<std::uint32_t> tile_bytes{key0.tile_bytes[0],
                                        olk_key.tile_bytes[0],
                                        olk_p2.tile_bytes[1],
                                        p3.tile_bytes[1]};
  std::vector<std::uint32_t> inter_blocks{key0.inter_blocks[0],
                                          olk_key.inter_blocks[0],
                                          olk_p2.inter_blocks[1],
                                          p3.inter_blocks[1]};
  std::vector<std::uint32_t> intra_blocks{key0.intra_blocks[0],
                                          olk_key.intra_blocks[0],
                                          olk_p2.intra_blocks[1],
                                          p3.intra_blocks[1]};
  std::vector<VideoReconFrame> recon;
  recon.push_back(std::move(key0.recon[0]));
  recon.push_back(std::move(olk_p2.recon[1]));
  recon.push_back(std::move(p3.recon[1]));
  fill_long_term_result(assembled, std::move(tile_bytes),
                        std::move(inter_blocks), std::move(intra_blocks),
                        std::move(output_index), std::move(recon), result);
  return true;
}

namespace ld = inter::leading;

// Open-GOP leading pictures, decode order:
//
//   CLK    f0        output
//   OLK    f(L+1)    output (implicit_output_frame = 1), RAP
//   LEAD   f1..fL    coded non-output (§7.3.4), reference the OLK
//   REG    f(L+2)..  output, reference the OLK
//
// monotonic_output_order_flag = 0 is required so the OLK can be output without
// a second output picture in its temporal unit.  Tile payloads are encoded as
// `[OLK, picture]` P sub-encodes, so every tile predicts from the slot its
// header names.
bool encode_leading_sequence(std::span<const VideoInputFrame> frames,
                             const VideoOptions& options,
                             VideoSequenceResult& result, std::string& error) {
  const int leading = options.leading_frames;
  if (leading < 1 || leading > 7) {
    error = "--video-leading must be in [1, 7]";
    return false;
  }
  const std::size_t lead = static_cast<std::size_t>(leading);
  if (frames.size() < lead + 2U) {
    error = "--video-leading " + std::to_string(leading) +
            " needs at least " + std::to_string(leading + 2) +
            " input frames";
    return false;
  }
  const std::size_t regular = frames.size() - lead - 2U;
  // Slots: OLK 0, leading 1..L, regular L+1..L+R.  order hints go to L+1+R.
  if (lead + regular > 7U) {
    error = "--video-leading: leading + regular frames exceed the 8 slots "
            "this profile signals";
    return false;
  }
  if (lead + 1U + regular > 15U) {
    error = "--video-leading: order hints exceed OrderHintBits = 4";
    return false;
  }
  const std::size_t olk_index = lead + 1U;

  PlainEncode clk;
  PlainEncode olk;
  if (!encode_plain(frames.subspan(0, 1), options, clk, error) ||
      !encode_plain(frames.subspan(olk_index, 1), options, olk, error)) {
    return false;
  }
  std::vector<PlainEncode> leading_enc(lead);
  for (std::size_t j = 0; j < lead; ++j) {
    const VideoInputFrame inputs[2] = {frames[olk_index], frames[1U + j]};
    if (!encode_plain(std::span<const VideoInputFrame>(inputs, 2), options,
                      leading_enc[j], error)) {
      return false;
    }
  }
  std::vector<PlainEncode> regular_enc(regular);
  for (std::size_t r = 0; r < regular; ++r) {
    const VideoInputFrame inputs[2] = {frames[olk_index],
                                       frames[lead + 2U + r]};
    if (!encode_plain(std::span<const VideoInputFrame>(inputs, 2), options,
                      regular_enc[r], error)) {
      return false;
    }
  }

  ld::SequenceConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(frames[0].width);
  config.max_frame_height = static_cast<std::uint16_t>(frames[0].height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = 8;
  config.max_drl_bits_minus_1 = 4;
  config.monotonic_output_order_flag = false;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.film_grain_params_present = options.film_grain;

  ld::Plan plan;
  ld::FrameHeader header;
  header.kind = ld::FrameKind::kClosedLoopKey;
  header.order_hint = 0;
  header.refresh_frame_flags = 0xFFU;
  header.num_total_refs = 0;
  plan.frames.push_back(header);
  header = ld::FrameHeader{};
  header.kind = ld::FrameKind::kOpenLoopKey;
  header.order_hint = static_cast<std::uint32_t>(olk_index);
  header.refresh_frame_flags = 1U;
  plan.frames.push_back(header);
  for (std::size_t j = 0; j < lead; ++j) {
    header = ld::FrameHeader{};
    header.kind = ld::FrameKind::kLeadingTileGroup;
    header.order_hint = static_cast<std::uint32_t>(1U + j);
    header.refresh_frame_flags = 1U << (1U + j);
    header.num_total_refs = 1;
    header.ref_frame_idx[0] = 0;
    header.leading_output_frame = false;
    plan.frames.push_back(header);
  }
  for (std::size_t r = 0; r < regular; ++r) {
    header = ld::FrameHeader{};
    header.kind = ld::FrameKind::kRegularTileGroup;
    header.order_hint = static_cast<std::uint32_t>(lead + 2U + r);
    header.refresh_frame_flags = 1U << (lead + 1U + r);
    header.num_total_refs = 1;
    header.ref_frame_idx[0] = 0;
    plan.frames.push_back(header);
  }
  // §5.14 film grain: the CLK, the OLK (implicit output) and every regular
  // frame are output pictures; the leading pictures are coded non-output and
  // code no film_grain_config().  Plan in coding order so the fgm_id slots the
  // decoder sees match the encoder's planner state.  §7.3.8.2 makes the OLK a
  // random access point: restart the planner there so the OLK always re-emits
  // its model in its own unit (models first sent with the CLK are not
  // available to a decoder that starts at the OLK).
  std::vector<gr::GrainFramePlan> grain_plans(plan.frames.size());
  gr::GrainStreamPlanner clk_grain_planner = make_grain_planner(options);
  gr::GrainStreamPlanner rap_grain_planner = make_grain_planner(options);
  if (options.film_grain) {
    if (!plan_grain(frames[0], 0, clk_grain_planner, grain_plans[0], error) ||
        !plan_grain(frames[olk_index],
                    static_cast<std::uint32_t>(olk_index),
                    rap_grain_planner, grain_plans[1], error)) {
      return false;
    }
    for (std::size_t r = 0; r < regular; ++r) {
      const std::size_t plan_index = lead + 2U + r;
      const std::size_t input_index = lead + 2U + r;
      if (!plan_grain(frames[input_index],
                      static_cast<std::uint32_t>(input_index),
                      rap_grain_planner, grain_plans[plan_index], error)) {
        return false;
      }
    }
    set_grain(plan.frames[0], grain_plans[0]);
    set_grain(plan.frames[1], grain_plans[1]);
    for (std::size_t r = 0; r < regular; ++r) {
      const std::size_t plan_index = lead + 2U + r;
      set_grain(plan.frames[plan_index], grain_plans[plan_index]);
    }
  }
  if (!ld::validate_plan(config, plan, error)) return false;

  std::vector<std::vector<std::uint8_t>> tiles;
  tiles.reserve(plan.frames.size());
  tiles.push_back(clk.tiles[0]);
  tiles.push_back(olk.tiles[0]);
  for (std::size_t j = 0; j < lead; ++j) tiles.push_back(leading_enc[j].tiles[1]);
  for (std::size_t r = 0; r < regular; ++r) tiles.push_back(regular_enc[r].tiles[1]);

  result.units = ld::assemble_temporal_units(config, plan, tiles);
  if (options.film_grain) {
    if (result.units.size() != plan.frames.size()) {
      error = "leading plan assembled an unexpected unit count";
      return false;
    }
    for (std::size_t i = 0; i < result.units.size(); ++i) {
      if (!insert_grain_obu(grain_plans[i], 0, result.units[i], error)) {
        return false;
      }
    }
  }
  result.frame_output_index.clear();
  result.recon.clear();
  result.recon.push_back(std::move(clk.recon[0]));
  result.recon.push_back(std::move(olk.recon[0]));
  result.frame_output_index.push_back(0);
  result.frame_output_index.push_back(1);
  for (std::size_t j = 0; j < lead; ++j) {
    result.frame_output_index.push_back(-1);
  }
  for (std::size_t r = 0; r < regular; ++r) {
    result.frame_output_index.push_back(static_cast<std::int32_t>(2U + r));
    result.recon.push_back(std::move(regular_enc[r].recon[1]));
  }
  result.frame_bytes.clear();
  result.tile_bytes.clear();
  result.inter_blocks.clear();
  result.intra_blocks.clear();
  const std::size_t td_size = ld::make_temporal_delimiter_obu().size();
  const std::size_t seq_size = ld::make_sequence_header_obu(config).size();
  const auto push_stats = [&result, td_size,
                           seq_size](const PlainEncode& enc, std::size_t unit,
                                     std::size_t tile_index,
                                     std::size_t plan_index) {
    const std::size_t overhead =
        td_size + ((plan_index <= 1U) ? seq_size : 0U);
    const std::size_t unit_bytes = result.units[plan_index].size();
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(unit_bytes - overhead));
    result.tile_bytes.push_back(enc.tile_bytes[tile_index]);
    result.inter_blocks.push_back(enc.inter_blocks[unit]);
    result.intra_blocks.push_back(enc.intra_blocks[unit]);
  };
  push_stats(clk, 0, 0, 0);
  push_stats(olk, 0, 0, 1);
  for (std::size_t j = 0; j < lead; ++j) {
    push_stats(leading_enc[j], 1, 1, 2U + j);
  }
  for (std::size_t r = 0; r < regular; ++r) {
    push_stats(regular_enc[r], 1, 1, lead + 2U + r);
  }
  return true;
}

ra::SequenceConfig random_access_config(const VideoOptions& options,
                                        std::uint32_t width,
                                        std::uint32_t height,
                                        std::uint8_t num_ref_frames,
                                        std::uint8_t long_term_id_bits,
                                        bool ras_codes_refresh) {
  ra::SequenceConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = 4;
  config.num_ref_frames = num_ref_frames;
  config.max_drl_bits_minus_1 = 4;
  config.long_term_frame_id_bits = long_term_id_bits;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_avg_cdf = options.avg_cdf;
  config.avg_cdf_type = true;
  config.enable_cfl_intra = true;
  // Pinned dav2d f9dafbc reads explicit refresh bits for OBU_RAS_FRAME.
  config.ras_codes_refresh_frame_flags = ras_codes_refresh;
  config.film_grain_params_present = options.film_grain;
  return config;
}

void fill_random_access_result(
    const std::vector<std::vector<std::uint8_t>>& units,
    std::vector<std::uint32_t> frame_bytes,
    std::vector<std::uint32_t> tile_bytes,
    std::vector<std::uint32_t> inter_blocks,
    std::vector<std::uint32_t> intra_blocks,
    std::vector<VideoReconFrame> recon, VideoSequenceResult& result) {
  result.units = units;
  result.recon = std::move(recon);
  result.frame_output_index.clear();
  result.frame_output_index.reserve(frame_bytes.size());
  for (std::size_t i = 0; i < frame_bytes.size(); ++i) {
    result.frame_output_index.push_back(static_cast<std::int32_t>(i));
  }
  result.frame_bytes = std::move(frame_bytes);
  result.tile_bytes = std::move(tile_bytes);
  result.inter_blocks = std::move(inter_blocks);
  result.intra_blocks = std::move(intra_blocks);
}

// Two-picture random-access switch: CLK f0, then OBU_SWITCH f1 with
// restricted_prediction_switch = 1.  The P payload encoded against f0 is
// reused; §5.18.2 needs only the header and OBU type to change.
bool encode_switch_sequence(std::span<const VideoInputFrame> frames,
                            const VideoOptions& options,
                            VideoSequenceResult& result, std::string& error) {
  if (frames.size() != 2) {
    error = "--video-switch requires exactly 2 input frames (CLK, SWITCH)";
    return false;
  }
  PlainEncode plain;
  if (!encode_plain(frames, options, plain, error)) return false;
  const ra::SequenceConfig config =
      random_access_config(options, frames[0].width, frames[0].height,
                           /*num_ref_frames=*/2, /*long_term_id_bits=*/0,
                           /*ras_codes_refresh=*/false);
  ra::RefState refs;
  ra::FrameHeader key;
  key.obu_type = ra::ObuType::kClosedLoopKey;
  key.order_hint = 0;
  key.long_term_id_plus_1 = 0;
  key.immediate_output_frame = true;
  gr::GrainStreamPlanner grain_planner = make_grain_planner(options);
  gr::GrainFramePlan grain_key;
  gr::GrainFramePlan grain_switch;
  if (options.film_grain && !plan_grain(frames[0], 0, grain_planner,
                                        grain_key, error)) {
    return false;
  }
  if (options.film_grain) set_grain(key, grain_key);
  ra::ValidationResult check = ra::validate_header(config, key, refs, 0);
  if (!check.ok) {
    error = "SWITCH key header: " + check.violations.front().message;
    return false;
  }
  const std::uint16_t key_refresh =
      static_cast<std::uint16_t>((1U << config.num_ref_frames) - 1U);
  ra::update_reference_state(config, key, key_refresh, refs);

  ra::PFrameCandidate candidate;
  candidate.reference_slot = 0;
  candidate.refresh_frame_flags = 2;  // slot 1; keep slot 0
  candidate.order_hint = 1;
  candidate.has_inter_blocks = plain.inter_blocks[1] > 0;
  const ra::PayloadPlan plan = ra::plan_switch_frame(
      config, refs, candidate, /*restricted_prediction_switch=*/true);
  if (plan.decision != ra::PayloadDecision::kReusePFrame) {
    error = "SWITCH re-encode required: " + plan.reason;
    return false;
  }
  ra::FrameHeader switch_header = plan.header;
  if (options.film_grain &&
      !plan_grain(frames[1], 1, grain_planner, grain_switch, error)) {
    return false;
  }
  if (options.film_grain) set_grain(switch_header, grain_switch);
  check = ra::validate_header(config, switch_header, refs, 0);
  if (!check.ok) {
    error = "SWITCH header: " + check.violations.front().message;
    return false;
  }

  const std::vector<std::uint8_t> td = ra::make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = ra::make_sequence_header_obu(config);
  const std::vector<std::uint8_t> key_obu =
      ra::make_frame_obu(config, key, plain.tiles[0]);
  const std::vector<std::uint8_t> switch_obu =
      ra::make_frame_obu(config, switch_header, plain.tiles[1]);
  std::vector<std::vector<std::uint8_t>> units(2);
  const auto append = [](std::vector<std::uint8_t>& out,
                         const std::vector<std::uint8_t>& bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
  };
  append(units[0], td);
  append(units[0], seq);
  append(units[0], key_obu);
  append(units[1], td);
  append(units[1], switch_obu);
  if (options.film_grain) {
    if (!insert_grain_obu(grain_key, 0, units[0], error) ||
        !insert_grain_obu(grain_switch, 0, units[1], error)) {
      return false;
    }
  }

  std::vector<VideoReconFrame> recon;
  recon.push_back(std::move(plain.recon[0]));
  recon.push_back(std::move(plain.recon[1]));
  fill_random_access_result(
      units,
      {static_cast<std::uint32_t>(key_obu.size()),
       static_cast<std::uint32_t>(switch_obu.size())},
      {plain.tile_bytes[0], plain.tile_bytes[1]},
      {plain.inter_blocks[0], plain.inter_blocks[1]},
      {plain.intra_blocks[0], plain.intra_blocks[1]}, std::move(recon), result);
  return true;
}

// Three-picture RAS stream: CLK (LongTermId 0), OBU_SWITCH, then
// OBU_RAS_FRAME referencing the named long-term slot across the switch.  The
// RAS payload is the separate [f0, f2] P encode, i.e. predicted from f0 rather
// than from the switch frame (§7.4.5).
bool encode_ras_sequence(std::span<const VideoInputFrame> frames,
                         const VideoOptions& options, VideoSequenceResult& result,
                         std::string& error) {
  if (frames.size() != 3) {
    error = "--video-ras requires exactly 3 input frames (CLK, SWITCH, RAS)";
    return false;
  }
  PlainEncode all;
  PlainEncode ras_payload;
  if (!encode_plain(frames, options, all, error)) return false;
  const VideoInputFrame ras_inputs[2] = {frames[0], frames[2]};
  if (!encode_plain(std::span<const VideoInputFrame>(ras_inputs, 2), options,
                    ras_payload, error)) {
    return false;
  }
  const ra::SequenceConfig config =
      random_access_config(options, frames[0].width, frames[0].height,
                           /*num_ref_frames=*/2, /*long_term_id_bits=*/1,
                           /*ras_codes_refresh=*/true);
  ra::RefState refs;
  ra::FrameHeader key;
  key.obu_type = ra::ObuType::kClosedLoopKey;
  key.order_hint = 0;
  key.long_term_id_plus_1 = 1;  // LongTermId 0
  key.immediate_output_frame = true;
  gr::GrainStreamPlanner grain_planner = make_grain_planner(options);
  gr::GrainFramePlan grain_key;
  gr::GrainFramePlan grain_switch;
  gr::GrainFramePlan grain_ras;
  if (options.film_grain && !plan_grain(frames[0], 0, grain_planner,
                                        grain_key, error)) {
    return false;
  }
  if (options.film_grain) set_grain(key, grain_key);
  ra::ValidationResult check = ra::validate_header(config, key, refs, 0);
  if (!check.ok) {
    error = "RAS key header: " + check.violations.front().message;
    return false;
  }
  const std::uint16_t key_refresh =
      static_cast<std::uint16_t>((1U << config.num_ref_frames) - 1U);
  ra::update_reference_state(config, key, key_refresh, refs);

  ra::PFrameCandidate switch_candidate;
  switch_candidate.reference_slot = 0;
  switch_candidate.refresh_frame_flags = 2;
  switch_candidate.order_hint = 1;
  switch_candidate.has_inter_blocks = all.inter_blocks[1] > 0;
  const ra::PayloadPlan switch_plan = ra::plan_switch_frame(
      config, refs, switch_candidate, /*restricted_prediction_switch=*/true);
  if (switch_plan.decision != ra::PayloadDecision::kReusePFrame) {
    error = "RAS/SWITCH re-encode required: " + switch_plan.reason;
    return false;
  }
  ra::FrameHeader switch_header = switch_plan.header;
  if (options.film_grain &&
      !plan_grain(frames[1], 1, grain_planner, grain_switch, error)) {
    return false;
  }
  if (options.film_grain) set_grain(switch_header, grain_switch);
  check = ra::validate_header(config, switch_header, refs, 0);
  if (!check.ok) {
    error = "RAS/SWITCH header: " + check.violations.front().message;
    return false;
  }
  ra::update_reference_state(config, switch_header,
                             switch_header.refresh_frame_flags, refs);

  const std::uint8_t ras_ids[] = {0};
  ra::PFrameCandidate ras_candidate;
  ras_candidate.reference_slot = 0;
  ras_candidate.refresh_frame_flags = 0;  // derived for RAS
  ras_candidate.order_hint = 2;
  ras_candidate.has_inter_blocks = ras_payload.inter_blocks[1] > 0;
  const ra::PayloadPlan ras_plan =
      ra::plan_ras_frame(config, refs, ras_candidate, ras_ids);
  if (ras_plan.decision != ra::PayloadDecision::kReusePFrame) {
    error = "RAS re-encode required: " + ras_plan.reason;
    return false;
  }
  ra::FrameHeader ras_header = ras_plan.header;
  if (options.film_grain &&
      !plan_grain(frames[2], 2, grain_planner, grain_ras, error)) {
    return false;
  }
  if (options.film_grain) set_grain(ras_header, grain_ras);
  check = ra::validate_header(config, ras_header, refs, 0);
  if (!check.ok) {
    error = "RAS header: " + check.violations.front().message;
    return false;
  }

  const std::vector<std::uint8_t> td = ra::make_temporal_delimiter_obu();
  const std::vector<std::uint8_t> seq = ra::make_sequence_header_obu(config);
  const std::vector<std::uint8_t> key_obu =
      ra::make_frame_obu(config, key, all.tiles[0]);
  const std::vector<std::uint8_t> switch_obu =
      ra::make_frame_obu(config, switch_header, all.tiles[1]);
  const std::vector<std::uint8_t> ras_obu =
      ra::make_frame_obu(config, ras_header, ras_payload.tiles[1]);
  std::vector<std::vector<std::uint8_t>> units(3);
  const auto append = [](std::vector<std::uint8_t>& out,
                         const std::vector<std::uint8_t>& bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
  };
  append(units[0], td);
  append(units[0], seq);
  append(units[0], key_obu);
  append(units[1], td);
  append(units[1], switch_obu);
  append(units[2], td);
  append(units[2], ras_obu);
  if (options.film_grain) {
    if (!insert_grain_obu(grain_key, 0, units[0], error) ||
        !insert_grain_obu(grain_switch, 0, units[1], error) ||
        !insert_grain_obu(grain_ras, 0, units[2], error)) {
      return false;
    }
  }

  std::vector<VideoReconFrame> recon;
  recon.push_back(std::move(all.recon[0]));
  recon.push_back(std::move(all.recon[1]));
  recon.push_back(std::move(ras_payload.recon[1]));
  fill_random_access_result(
      units,
      {static_cast<std::uint32_t>(key_obu.size()),
       static_cast<std::uint32_t>(switch_obu.size()),
       static_cast<std::uint32_t>(ras_obu.size())},
      {all.tile_bytes[0], all.tile_bytes[1], ras_payload.tile_bytes[1]},
      {all.inter_blocks[0], all.inter_blocks[1], ras_payload.inter_blocks[1]},
      {all.intra_blocks[0], all.intra_blocks[1], ras_payload.intra_blocks[1]},
      std::move(recon), result);
  return true;
}

// Show-existing-frame: re-encode the plain sequence with the last picture
// hidden, then append one TD + OBU_REGULAR_SEF unit that re-shows it.
bool encode_sef_sequence(std::span<const VideoInputFrame> frames,
                         const VideoOptions& options, VideoSequenceResult& result,
                         std::string& error) {
  if (frames.size() < 2) {
    error = "--video-sef needs at least 2 input frames";
    return false;
  }
  VideoOptions plain = plain_options(options);
  plain.hide_last_picture = true;
  // Unlike the other wrappers this function keeps the plain temporal units
  // verbatim (only the SEF unit is appended), so the base pictures must carry
  // the §5.14 sequence bit / config / model OBUs themselves.
  plain.film_grain = options.film_grain;
  VideoSequenceResult base;
  if (!encode_video_sequence(frames, plain, base, error)) return false;

  const int num_ref_frames = resolved_num_ref_frames(options);
  const int last_slot = static_cast<int>((frames.size() - 1U) %
                                         static_cast<std::size_t>(num_ref_frames));
  // §5.14: the SEF is an output picture with its own film_grain_config() and
  // (when the model is new) one model OBU immediately before its OBU.  A
  // dedicated planner is safe: the OBU overwrites whichever slot it names
  // before the SEF header is parsed, so the referenced model is self-contained
  // (§7.3.8.8).
  gr::GrainStreamPlanner grain_planner = make_grain_planner(options);
  gr::GrainFramePlan grain_plan;
  if (options.film_grain &&
      !plan_grain(frames.back(), static_cast<std::uint32_t>(frames.size() - 1U),
                  grain_planner, grain_plan, error)) {
    return false;
  }
  sef::SefParams params;
  params.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  params.order_hint_bits = 4;
  params.film_grain_params_present = options.film_grain;
  sef::SlotMap slots_map;
  slots_map.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  sef::RefSlotState& shown =
      slots_map.slots[static_cast<std::size_t>(last_slot)];
  shown.valid = true;
  // The last order hint is the coded frame index (masked to OrderHintBits).
  shown.order_hint = static_cast<std::uint32_t>(frames.size() - 1U) & 0xFU;
  shown.immediate_output_frame = false;
  shown.implicit_output_frame = false;
  shown.shown = false;
  sef::SefHeader sef_header;
  sef::SefError sef_error = sef::SefError::kOk;
  if (!sef::make_show_header(slots_map, params,
                             static_cast<std::uint8_t>(last_slot),
                             sef::SefKind::kRegular, sef_header, &sef_error)) {
    error = std::string("SEF header: ") + sef::sef_error_name(sef_error);
    return false;
  }
  if (options.film_grain) set_grain(sef_header, grain_plan);
  if (!sef::validate_dav2d_compat(slots_map, params, sef_header, &sef_error)) {
    error = std::string("SEF pinned-decoder check: ") +
            sef::sef_error_name(sef_error);
    return false;
  }
  std::vector<std::uint8_t> unit = sef::make_sef_temporal_unit(params, sef_header);
  if (options.film_grain &&
      !insert_grain_obu(grain_plan, 0, unit, error)) {
    return false;
  }
  base.units.push_back(std::move(unit));
  base.frame_bytes.push_back(
      static_cast<std::uint32_t>(base.units.back().size()));
  base.tile_bytes.push_back(0);
  base.inter_blocks.push_back(0);
  base.intra_blocks.push_back(0);
  base.frame_output_index.push_back(
      static_cast<std::int32_t>(frames.size() - 1U));
  result = std::move(base);
  return true;
}

}  // namespace

bool encode_special_sequence(std::span<const VideoInputFrame> frames,
                             const VideoOptions& options,
                             VideoSequenceResult& result, std::string& error) {
  const int modes =
      ((options.long_term || options.olk) ? 1 : 0) +
      (options.leading_frames > 0 ? 1 : 0) + (options.sef ? 1 : 0) +
      (options.ra_switch ? 1 : 0) + (options.ras ? 1 : 0);
  if (modes != 1) {
    error = "select exactly one frame-type mode";
    return false;
  }
  if (options.ra_switch || options.ras) {
    if (options.num_total_refs != 1 || options.mv_precision != 0) {
      error = "--video-switch/--video-ras need --video-refs 1 and eighth-pel "
              "MV precision";
      return false;
    }
    if (options.ras) return encode_ras_sequence(frames, options, result, error);
    return encode_switch_sequence(frames, options, result, error);
  }
  if (options.sef) return encode_sef_sequence(frames, options, result, error);
  if (options.long_term || options.olk) {
    if (options.num_total_refs != 1 || options.mv_precision != 0) {
      error = "--video-long-term/--video-olk need --video-refs 1 and "
              "eighth-pel MV precision";
      return false;
    }
    if (options.olk) return encode_olk_sequence(frames, options, result, error);
    return encode_long_term_closed(frames, options, result, error);
  }
  if (options.num_total_refs != 1 || options.mv_precision != 0) {
    error = "--video-leading needs --video-refs 1 and eighth-pel MV precision";
    return false;
  }
  return encode_leading_sequence(frames, options, result, error);
}

}  // namespace av2enc::video::frames
