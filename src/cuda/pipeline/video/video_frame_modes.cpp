// GPU frame-type wrappers (long-term/OLK, leading, SEF, SWITCH/RAS).
//
// This mirrors src/pipeline/video/video_frames.cpp exactly.  The wrappers only
// change headers/OBU assembly; every tile payload still comes from the GPU
// plain P-chain encoder (`encode_video_sequence_gpu()` with the wrapper flags
// cleared), so the result is byte-identical to the CPU reference whenever the
// plain path is.
#include "video_sequence_internal.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "codec/inter/leading/leading.h"
#include "codec/inter/longterm/longterm.h"
#include "codec/inter/longterm/longterm_stream.h"
#include "codec/inter/random_access/random_access.h"
#include "codec/inter/sef/sef.h"
#include "pipeline/video/video_frame.h"
#include "video_gpu.h"

namespace av2enc::cuda::video::detail {
namespace {

namespace lt = ::av2enc::inter::longterm;
namespace ld = ::av2enc::inter::leading;
namespace ra = ::av2enc::inter::random_access;
namespace sef = ::av2enc::inter::sef;

using av2enc::video::VideoInputFrame;
using av2enc::video::VideoOptions;
using av2enc::video::VideoReconFrame;
using av2enc::video::VideoSequenceResult;

// One plain (flag-free) sub-encode: tile payloads in decode order, the
// reconstruction of every picture the sub-stream outputs, and its stats.
struct PlainEncode {
  std::vector<std::vector<std::uint8_t>> tiles;
  std::vector<VideoReconFrame> recon;
  std::vector<std::uint32_t> tile_bytes;
  std::vector<std::uint32_t> inter_blocks;
  std::vector<std::uint32_t> intra_blocks;
  VideoGpuStats stats;
};

void accumulate(VideoGpuStats& total, const VideoGpuStats& add) {
  total.variance_launches += add.variance_launches;
  total.me_launches += add.me_launches;
  total.mc_launches += add.mc_launches;
  total.sse_launches += add.sse_launches;
  total.intra_launches += add.intra_launches;
  total.tx_launches += add.tx_launches;
  total.residual_launches += add.residual_launches;
  total.recon_launches += add.recon_launches;
  total.syncs += add.syncs;
  total.bytes_uploaded += add.bytes_uploaded;
  total.bytes_downloaded += add.bytes_downloaded;
  total.me_jobs += add.me_jobs;
  total.inter_candidates += add.inter_candidates;
  total.device_ms += add.device_ms;
  total.frame_ms.insert(total.frame_ms.end(), add.frame_ms.begin(),
                        add.frame_ms.end());
}

VideoOptions plain_options(const VideoOptions& options) {
  VideoOptions plain = options;
  plain.long_term = false;
  plain.olk = false;
  plain.leading_frames = 0;
  plain.sef = false;
  plain.ra_switch = false;
  plain.ras = false;
  plain.hide_last_picture = false;
  return plain;
}

bool encode_plain(std::span<const VideoInputFrame> frames,
                  const VideoOptions& options, PlainEncode& out,
                  std::string& error) {
  VideoSequenceResult sequence;
  if (!encode_video_sequence_gpu(frames, plain_options(options), sequence,
                                 error, &out.stats)) {
    return false;
  }
  out.tiles.clear();
  out.tiles.reserve(sequence.units.size());
  for (std::size_t i = 0; i < sequence.units.size(); ++i) {
    const std::size_t count = sequence.tile_bytes[i];
    if (count == 0 || count > sequence.units[i].size()) {
      error = "plain GPU sub-encode produced an invalid tile size";
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
                             VideoSequenceResult& result, std::string& error,
                             VideoGpuStats& stats) {
  constexpr std::size_t kMaxOrderHintFrames = std::size_t{1} << 4;
  if (frames.size() > kMaxOrderHintFrames) {
    error = "--video-long-term codes OrderHint with OrderHintBits = 4, so it "
            "supports at most 16 coded frames (got " +
            std::to_string(frames.size()) + ")";
    return false;
  }
  PlainEncode plain;
  if (!encode_plain(frames, options, plain, error)) return false;
  accumulate(stats, plain.stats);
  const lt::SequenceConfig config =
      long_term_config(options, frames[0].width, frames[0].height,
                       /*num_ref_frames=*/2);
  std::vector<lt::TemporalUnitPlan> units;
  units.reserve(frames.size());
  lt::FrameHeader key;
  if (!lt::plan_closed_loop_key(config, 0, /*long_term_id=*/0, &key, &error)) {
    return false;
  }
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
    units.emplace_back();
    units.back().frames.push_back(lt::PlannedFrame{header, plain.tiles[i]});
  }
  lt::AssembledSequence assembled;
  if (!lt::assemble_sequence(config, units, &assembled, &error)) return false;
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
bool encode_olk_sequence(std::span<const VideoInputFrame> frames,
                         const VideoOptions& options,
                         VideoSequenceResult& result, std::string& error,
                         VideoGpuStats& stats) {
  if (frames.size() != 4) {
    error = "--video-olk requires exactly 4 input frames (CLK, OLK, P2, P3)";
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
  for (const PlainEncode* p : {&key0, &olk_key, &olk_p2, &p3}) {
    accumulate(stats, p->stats);
  }
  const lt::SequenceConfig config =
      long_term_config(options, width, height, /*num_ref_frames=*/4);
  lt::PoolState pool;
  lt::reset_pool(config, &pool);

  lt::FrameHeader clk;
  if (!lt::plan_closed_loop_key(config, 0, /*long_term_id=*/0, &clk,
                                &error)) {
    return false;
  }
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
  lt::FrameHeader p2;
  {
    const std::vector<std::uint8_t> refs{1};
    if (!lt::plan_inter_frame(config, lt::FrameRole::kRegular,
                              /*order_hint=*/2, refs, /*refresh_slot=*/2,
                              true, &p2, &error)) {
      return false;
    }
  }
  lt::FrameHeader p3_header;
  {
    const std::vector<std::uint8_t> refs{0};
    if (!lt::plan_inter_frame(config, lt::FrameRole::kRegular,
                              /*order_hint=*/3, refs, /*refresh_slot=*/3,
                              true, &p3_header, &error)) {
      return false;
    }
  }
  std::vector<lt::TemporalUnitPlan> units(3);
  units[0].frames.push_back(lt::PlannedFrame{clk, key0.tiles[0]});
  units[1].frames.push_back(lt::PlannedFrame{olk, olk_key.tiles[0]});
  units[1].frames.push_back(lt::PlannedFrame{p2, olk_p2.tiles[1]});
  units[2].frames.push_back(lt::PlannedFrame{p3_header, p3.tiles[1]});
  lt::AssembledSequence assembled;
  if (!lt::assemble_sequence(config, units, &assembled, &error)) return false;

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

// Open-GOP leading pictures, decode order:
//
//   CLK    f0        output
//   OLK    f(L+1)    output (implicit_output_frame = 1), RAP
//   LEAD   f1..fL    coded non-output (§7.3.4), reference the OLK
//   REG    f(L+2)..  output, reference the OLK
bool encode_leading_sequence(std::span<const VideoInputFrame> frames,
                             const VideoOptions& options,
                             VideoSequenceResult& result, std::string& error,
                             VideoGpuStats& stats) {
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
  accumulate(stats, clk.stats);
  accumulate(stats, olk.stats);
  for (const PlainEncode& e : leading_enc) accumulate(stats, e.stats);
  for (const PlainEncode& e : regular_enc) accumulate(stats, e.stats);

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
  if (!ld::validate_plan(config, plan, error)) return false;

  std::vector<std::vector<std::uint8_t>> tiles;
  tiles.reserve(plan.frames.size());
  tiles.push_back(clk.tiles[0]);
  tiles.push_back(olk.tiles[0]);
  for (std::size_t j = 0; j < lead; ++j) tiles.push_back(leading_enc[j].tiles[1]);
  for (std::size_t r = 0; r < regular; ++r) tiles.push_back(regular_enc[r].tiles[1]);

  result.units = ld::assemble_temporal_units(config, plan, tiles);
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
  config.ras_codes_refresh_frame_flags = ras_codes_refresh;
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
                            VideoSequenceResult& result, std::string& error,
                            VideoGpuStats& stats) {
  if (frames.size() != 2) {
    error = "--video-switch requires exactly 2 input frames (CLK, SWITCH)";
    return false;
  }
  PlainEncode plain;
  if (!encode_plain(frames, options, plain, error)) return false;
  accumulate(stats, plain.stats);
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
  candidate.refresh_frame_flags = 2;
  candidate.order_hint = 1;
  candidate.has_inter_blocks = plain.inter_blocks[1] > 0;
  const ra::PayloadPlan plan = ra::plan_switch_frame(
      config, refs, candidate, /*restricted_prediction_switch=*/true);
  if (plan.decision != ra::PayloadDecision::kReusePFrame) {
    error = "SWITCH re-encode required: " + plan.reason;
    return false;
  }
  ra::FrameHeader switch_header = plan.header;
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
// OBU_RAS_FRAME referencing the named long-term slot across the switch.
bool encode_ras_sequence(std::span<const VideoInputFrame> frames,
                         const VideoOptions& options, VideoSequenceResult& result,
                         std::string& error, VideoGpuStats& stats) {
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
  accumulate(stats, all.stats);
  accumulate(stats, ras_payload.stats);
  const ra::SequenceConfig config =
      random_access_config(options, frames[0].width, frames[0].height,
                           /*num_ref_frames=*/2, /*long_term_id_bits=*/1,
                           /*ras_codes_refresh=*/true);
  ra::RefState refs;
  ra::FrameHeader key;
  key.obu_type = ra::ObuType::kClosedLoopKey;
  key.order_hint = 0;
  key.long_term_id_plus_1 = 1;
  key.immediate_output_frame = true;
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
  ras_candidate.refresh_frame_flags = 0;
  ras_candidate.order_hint = 2;
  ras_candidate.has_inter_blocks = ras_payload.inter_blocks[1] > 0;
  const ra::PayloadPlan ras_plan =
      ra::plan_ras_frame(config, refs, ras_candidate, ras_ids);
  if (ras_plan.decision != ra::PayloadDecision::kReusePFrame) {
    error = "RAS re-encode required: " + ras_plan.reason;
    return false;
  }
  ra::FrameHeader ras_header = ras_plan.header;
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
                         std::string& error, VideoGpuStats& stats) {
  if (frames.size() < 2) {
    error = "--video-sef needs at least 2 input frames";
    return false;
  }
  VideoOptions plain = plain_options(options);
  plain.hide_last_picture = true;
  VideoSequenceResult base;
  if (!encode_video_sequence_gpu(frames, plain, base, error, &stats)) {
    return false;
  }

  const int num_ref_frames = options.num_total_refs > 1 ? 8 : 2;
  const int last_slot = static_cast<int>(
      (frames.size() - 1U) % static_cast<std::size_t>(num_ref_frames));
  sef::SefParams params;
  params.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  params.order_hint_bits = 4;
  sef::SlotMap slots_map;
  slots_map.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  sef::RefSlotState& shown =
      slots_map.slots[static_cast<std::size_t>(last_slot)];
  shown.valid = true;
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
  if (!sef::validate_dav2d_compat(slots_map, params, sef_header, &sef_error)) {
    error = std::string("SEF pinned-decoder check: ") +
            sef::sef_error_name(sef_error);
    return false;
  }
  std::vector<std::uint8_t> unit = sef::make_sef_temporal_unit(params, sef_header);
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

bool is_frame_type_mode(const VideoOptions& options) {
  return options.long_term || options.olk || options.leading_frames > 0 ||
         options.sef || options.ra_switch || options.ras;
}

bool encode_special_sequence_gpu(
    std::span<const VideoInputFrame> frames, const VideoOptions& options,
    VideoSequenceResult& result, std::string& error, VideoGpuStats* stats) {
  error.clear();
  result = VideoSequenceResult{};
  VideoGpuStats total;
  const int modes =
      ((options.long_term || options.olk) ? 1 : 0) +
      (options.leading_frames > 0 ? 1 : 0) + (options.sef ? 1 : 0) +
      (options.ra_switch ? 1 : 0) + (options.ras ? 1 : 0);
  if (modes != 1) {
    error = "select exactly one frame-type mode";
    return false;
  }
  bool ok = false;
  if (options.ra_switch || options.ras) {
    if (options.num_total_refs != 1 || options.mv_precision != 0) {
      error = "--video-switch/--video-ras need --video-refs 1 and eighth-pel "
              "MV precision";
      return false;
    }
    ok = options.ras ? encode_ras_sequence(frames, options, result, error, total)
                     : encode_switch_sequence(frames, options, result, error,
                                              total);
  } else if (options.sef) {
    ok = encode_sef_sequence(frames, options, result, error, total);
  } else if (options.long_term || options.olk) {
    if (options.num_total_refs != 1 || options.mv_precision != 0) {
      error = "--video-long-term/--video-olk need --video-refs 1 and "
              "eighth-pel MV precision";
      return false;
    }
    ok = options.olk ? encode_olk_sequence(frames, options, result, error, total)
                     : encode_long_term_closed(frames, options, result, error,
                                               total);
  } else {
    if (options.num_total_refs != 1 || options.mv_precision != 0) {
      error = "--video-leading needs --video-refs 1 and eighth-pel MV precision";
      return false;
    }
    ok = encode_leading_sequence(frames, options, result, error, total);
  }
  if (ok && stats != nullptr) *stats = total;
  return ok;
}

}  // namespace av2enc::cuda::video::detail
