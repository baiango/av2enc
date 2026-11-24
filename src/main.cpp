#include <exception>
#include <new>

#include "core/bitstream/basic_headers.h"
#include "codec/entropy/lossless/lossless.h"
#include "codec/entropy/lossless/inter/lossless_inter.h"
#include "formats/chroma/chroma_still.h"
#include "formats/format_y4m.h"
#include "formats/format_encode.h"
#include "formats/format_y4m.h"
#include "pipeline/still/basic_encoder.h"
#include "pipeline/still/still_encoder.h"
#include "pipeline/svc/svc_stream.h"
#include "pipeline/tiles/stream/tiles_stream.h"
#include "pipeline/video/post/rc_adapter.h"
#include "pipeline/dispatch/planner.h"
#include "pipeline/planner/effort.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"
#include "pipeline/video/video_frame.h"
#include "codec/inter/bframes/bframes.h"
#include "codec/inter/tlayers/tlayers.h"
#include "codec/rate/rc/hrd/hrd.h"
#include "frame/metadata/cli/metadata_cli.h"
#include "frame/metadata/types/metadata_types.h"
#include "cli/options.h"

#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if AV2ENC_HAVE_CUDA
#include "cuda/arena.h"
#include "cuda/context.h"

// Still GPU pipeline (owner: orchestrator, src/cuda/pipeline/still/).  Include
// the contract header when it is visible.  The call itself is only linked when
// CMake also found the orchestrator entry point; until then --still-gpu reports
// itself as unavailable at runtime.
#if __has_include("cuda/pipeline/still/lossless/still_lossless_gpu.h")
#include "cuda/pipeline/still/lossless/still_lossless_gpu.h"
#define AV2ENC_STILL_LOSSLESS_HEADER 1
#else
#define AV2ENC_STILL_LOSSLESS_HEADER 0
#endif

#if __has_include("cuda/pipeline/still/still_gpu.h")
#include "cuda/pipeline/still/still_gpu.h"
#define AV2ENC_STILL_GPU_HEADER 1
#else
#define AV2ENC_STILL_GPU_HEADER 0
// TODO(agent 1/agent 9): replace this mirror with the real header once
// src/cuda/pipeline/still/still_gpu.h lands in the tree.  Field names follow
// docs/m1-gpu.md (GpuOptions) and the orchestrator header.
namespace av2enc::cuda::still_gpu {

struct GpuOptions {
  std::uint8_t base_q_idx = 90;
  bool enable_cfl = true;
  bool enable_deblock = true;
  bool enable_edge_filter = true;
  bool luma_dc_only = true;
  int min_block_size = 8;
  int max_partition_depth = 4;
  bool adaptive_cdf = false;
  bool force_single_block = false;
  bool force_dc = false;
  int device = 0;
  std::uint32_t vram_cap_mb = 0;
  bool allow_cpu_fallback = true;
  bool verify_against_cpu = true;
  bool verbose = false;
};

struct GpuResult {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  int ibc_blocks = 0;
  std::vector<std::uint8_t> frame;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  bool frame_bytes_from_cpu = true;
};

bool encode_still_gpu(const av2enc::still::StillPicture& picture,
                   const GpuOptions& options, GpuResult& result,
                   std::string& error);

}  // namespace av2enc::cuda::still_gpu
#endif

#if AV2ENC_STILL_GPU_HEADER && defined(AV2ENC_HAVE_STILL_GPU) && AV2ENC_HAVE_STILL_GPU
#define AV2ENC_STILL_GPU_CALLABLE 1
#else
#define AV2ENC_STILL_GPU_CALLABLE 0
#endif

// Video GPU pipeline (owner: orchestrator, src/cuda/pipeline/video/).  Same
// pattern as the Still path: include the contract header when visible and only
// treat the call as linkable when CMake built the M2 target.
#if __has_include("cuda/pipeline/video/video_gpu.h")
#include "cuda/pipeline/video/video_gpu.h"
#define AV2ENC_VIDEO_GPU_HEADER 1
#else
#define AV2ENC_VIDEO_GPU_HEADER 0
#endif
#if AV2ENC_VIDEO_GPU_HEADER && defined(AV2ENC_HAVE_VIDEO_GPU) && AV2ENC_HAVE_VIDEO_GPU
#define AV2ENC_VIDEO_GPU_CALLABLE 1
#else
#define AV2ENC_VIDEO_GPU_CALLABLE 0
#endif
#else
#define AV2ENC_STILL_GPU_CALLABLE 0
#define AV2ENC_VIDEO_GPU_CALLABLE 0
#endif

namespace {

// Appends the opt-in metadata OBUs as §5.17.2 suffix units at the end of one
// temporal unit.  The payloads are the minimal spec-valid forms (no invented
// timing/hash policy); pinned dav2d ignores OBU types 8/9, so the gate is
// decode invariance.
bool append_neutral_metadata_obus(
    std::vector<std::uint8_t>& unit,
    const av2enc::metadata::cli::Options& metadata,
    const av2enc::metadata::cli::Options::FramePlanes* planes, bool timecode,
    bool scan_type, std::string& error) {
  if (!metadata.any() && !timecode && !scan_type) return true;
  if (!av2enc::metadata::cli::append_suffix_metadata(unit, metadata, planes,
                                                     error)) {
    return false;
  }
  av2enc::hdr::MetadataObuConfig config;
  config.suffix = true;
  config.layer = av2enc::hdr::MetadataLayer::kCurrent;
  config.persistence = av2enc::hdr::MetadataPersistence::kGlobal;
  if (timecode) {
    av2enc::metadata::Timecode value;  // all-zero minimal 23-bit unit
    const auto payload = av2enc::metadata::make_timecode_unit(value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        av2enc::metadata::make_short_obu(*payload, config, error);
    if (obu.empty()) return false;
    unit.insert(unit.end(), obu.begin(), obu.end());
  }
  if (scan_type) {
    av2enc::metadata::ScanType value;
    value.mps_pic_struct_type = 0;
    value.mps_source_scan_type_idc = 1;  // Table 6.18: progressive
    value.mps_duplicate_flag = false;
    const auto payload = av2enc::metadata::make_scan_type_unit(value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        av2enc::metadata::make_short_obu(*payload, config, error);
    if (obu.empty()) return false;
    unit.insert(unit.end(), obu.begin(), obu.end());
  }
  return true;
}

// Maps the CLI flags to the metadata carrier options.  Everything stays
// unsignalled unless its flag is set.
av2enc::metadata::cli::Options metadata_cli_options(const Options& options) {
  av2enc::metadata::cli::Options metadata;
  metadata.cll = options.metadata_cll;
  metadata.cll_value.max_cll = options.metadata_max_cll;
  metadata.cll_value.max_fall = options.metadata_max_fall;
  metadata.mdcv = options.metadata_mdcv;
  metadata.mdcv_value = options.metadata_mdcv_value;
  metadata.icc = options.metadata_icc;
  metadata.icc_data = options.metadata_icc_data;
  metadata.temporal_point = options.metadata_temporal_point;
  metadata.frame_presentation_time = options.metadata_frame_presentation_time;
  metadata.frame_hash = options.metadata_frame_hash;
  metadata.frame_hash_per_plane = options.metadata_frame_hash_per_plane;
  metadata.banding = options.metadata_banding;
  metadata.banding_value = options.metadata_banding_value;
  metadata.user_data = options.metadata_user_data;
  metadata.user_data_value = options.metadata_user_data_value;
  return metadata;
}

// 8-bit 4:2:0 reconstruction planes for the §6.16.13 hash.
av2enc::metadata::cli::Options::FramePlanes make_plane_view(
    const std::vector<std::uint8_t>& y, const std::vector<std::uint8_t>& u,
    const std::vector<std::uint8_t>& v, std::uint32_t width,
    std::uint32_t height) {
  av2enc::metadata::cli::Options::FramePlanes planes;
  planes.y = std::span<const std::uint8_t>(y.data(), y.size());
  planes.u = std::span<const std::uint8_t>(u.data(), u.size());
  planes.v = std::span<const std::uint8_t>(v.data(), v.size());
  planes.width = width;
  planes.height = height;
  return planes;
}

av2enc::metadata::cli::Options::FramePlanes make_plane_view(
    std::span<const std::uint8_t> y, std::span<const std::uint8_t> u,
    std::span<const std::uint8_t> v, std::uint32_t width,
    std::uint32_t height) {
  av2enc::metadata::cli::Options::FramePlanes planes;
  planes.y = y;
  planes.u = u;
  planes.v = v;
  planes.width = width;
  planes.height = height;
  return planes;
}

// §6.11 buffer removal timing: one non-dependent BRT OBU with br_time in the
// rg(4) range.  It belongs before the frame OBU inside the coded output frame
// unit (§7.3.3), so it is spliced in rather than appended.
bool make_brt_obu(std::uint32_t br_time, std::vector<std::uint8_t>& obu,
                  std::string& error) {
  av2enc::hrd::BufferRemovalTiming brt;
  brt.br_ops_dependent_flag = false;
  brt.br_time = br_time;
  return av2enc::hrd::build_buffer_removal_timing_obu(brt, obu, error);
}

// Inserts `obu` immediately before the first frame OBU of a temporal unit.
bool insert_before_first_frame_obu(std::vector<std::uint8_t>& unit,
                                   const std::vector<std::uint8_t>& obu,
                                   std::string& error) {
  std::size_t pos = 0;
  while (pos < unit.size()) {
    std::size_t cursor = pos;
    std::uint64_t size = 0;
    unsigned shift = 0;
    while (true) {
      if (cursor >= unit.size()) {
        error = "temporal unit truncated inside an OBU size";
        return false;
      }
      const std::uint8_t byte = unit[cursor++];
      size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
      if ((byte & 0x80U) == 0U) break;
      shift += 7U;
      if (shift > 28U) {
        error = "temporal unit has an oversized OBU size";
        return false;
      }
    }
    if (cursor >= unit.size() || cursor + size > unit.size()) {
      error = "temporal unit truncated inside an OBU payload";
      return false;
    }
    const unsigned type = (unit[cursor] >> 2U) & 0x1FU;
    const bool is_frame =
        type == 4U || type == 5U || type == 6U || type == 7U || type == 10U ||
        type == 11U || type == 12U || type == 13U || type == 14U ||
        type == 19U || type == 21U;
    if (is_frame) {
      unit.insert(unit.begin() + static_cast<std::ptrdiff_t>(pos), obu.begin(),
                  obu.end());
      return true;
    }
    pos = cursor + static_cast<std::size_t>(size);
  }
  error = "temporal unit has no frame OBU";
  return false;
}

av2enc::tiles::TileLayoutPreset tile_preset_from_layout(std::string_view canonical) {
  using av2enc::tiles::TileLayoutPreset;
  if (canonical == "2x1") return TileLayoutPreset::k2x1;
  if (canonical == "2x2") return TileLayoutPreset::k2x2;
  if (canonical == "4x2") return TileLayoutPreset::k4x2;
  if (canonical == "8x2") return TileLayoutPreset::k8x2;
  if (canonical == "16x2") return TileLayoutPreset::k16x2;
  return TileLayoutPreset::k1x1;
}

#if AV2ENC_HAVE_CUDA
// Nothing to query here: the pipeline sizes the arena from the device at first
// use. This only records the optional upper bound.
bool apply_vram_cap(std::uint64_t cap_mib, std::string& error) {
  if (cap_mib > std::numeric_limits<std::uint64_t>::max() /
                   av2enc::cuda::Arena::kMiB) {
    error = "--vram-cap is too large";
    return false;
  }
  av2enc::cuda::Arena::set_capacity_cap_bytes(
      cap_mib * av2enc::cuda::Arena::kMiB);
  return true;
}
#endif

struct OwnedVideoFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

// One-frame Y4M dump for the tiled still path (8-bit 4:2:0). Comparing this
// with the decoded frame is how the --tiles gate proves sample-exact
// reconstruction; the CLI itself only writes the bitstream.
bool write_still_recon_y4m(const std::filesystem::path& path,
                           const av2enc::Y4mHeader& header, int width, int height,
                           const std::vector<std::uint8_t>& y,
                           const std::vector<std::uint8_t>& u,
                           const std::vector<std::uint8_t>& v, std::string& error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error = "cannot open --recon output '" + path.string() + "'";
    return false;
  }
  out << "YUV4MPEG2 W" << width << " H" << height << " F" << header.fps_num << ":"
      << header.fps_den << " Ip A0:0 C420\nFRAME\n";
  out.write(reinterpret_cast<const char*>(y.data()),
            static_cast<std::streamsize>(y.size()));
  out.write(reinterpret_cast<const char*>(u.data()),
            static_cast<std::streamsize>(u.size()));
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size()));
  if (!out) {
    error = "failed while writing --recon output";
    return false;
  }
  return true;
}


bool write_video_recon_y4m(const std::filesystem::path& path,
                           const av2enc::Y4mHeader& header,
                           const std::vector<av2enc::video::VideoReconFrame>& recon,
                           std::string& error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error = "cannot open --video-recon output '" + path.string() + "'";
    return false;
  }
  out << "YUV4MPEG2 W" << header.width << " H" << header.height << " F"
      << header.fps_num << ":" << header.fps_den << " Ip A1:1 C420\n";
  for (const auto& frame : recon) {
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(frame.y.data()),
              static_cast<std::streamsize>(frame.y.size()));
    out.write(reinterpret_cast<const char*>(frame.u.data()),
              static_cast<std::streamsize>(frame.u.size()));
    out.write(reinterpret_cast<const char*>(frame.v.data()),
              static_cast<std::streamsize>(frame.v.size()));
  }
  if (!out) {
    error = "failed while writing --video-recon output";
    return false;
  }
  return true;
}

// Multi-frame reconstruction dump for the format video core.  The Y4M body is
// 16-bit little-endian for 10-bit and 8-bit bytes for the 8-bit formats, and
// the C tag matches the signalled format.
bool write_video_recon_y4m_format(
    const std::filesystem::path& path, const av2enc::fmt::Format& format,
    std::uint32_t fps_num, std::uint32_t fps_den,
    const std::vector<av2enc::video::VideoReconFrame>& recon,
    std::string& error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error = "cannot open --video-recon output '" + path.string() + "'";
    return false;
  }
  const int bps = format.depth == av2enc::fmt::BitDepth::k8 ? 1 : 2;
  out << "YUV4MPEG2 W" << format.width << " H" << format.height << " F"
      << fps_num << ":" << fps_den << " Ip A1:1 "
      << av2enc::fmt::y4m_colorspace_tag(format.chroma, format.depth) << "\n";
  const int sub_x = av2enc::fmt::subsampling_x(format.chroma);
  const int sub_y = av2enc::fmt::subsampling_y(format.chroma);
  const bool has_chroma = av2enc::fmt::has_chroma(format.chroma);
  const std::uint32_t widths[3] = {format.width, format.width >> sub_x,
                                   format.width >> sub_x};
  const std::uint32_t heights[3] = {format.height, format.height >> sub_y,
                                    format.height >> sub_y};
  const int plane_count = av2enc::fmt::num_planes(format.chroma);
  std::vector<std::uint8_t> row;
  for (const av2enc::video::VideoReconFrame& frame : recon) {
    out << "FRAME\n";
    for (int plane = 0; plane < plane_count; ++plane) {
      const std::vector<std::uint16_t>& src =
          plane == 0 ? frame.y16 : (plane == 1 ? frame.u16 : frame.v16);
      if (src.empty()) {
        error = "reconstruction plane is empty";
        return false;
      }
      row.resize(static_cast<std::size_t>(widths[plane]) *
                 static_cast<std::size_t>(bps));
      for (std::uint32_t y = 0; y < heights[plane]; ++y) {
        for (std::uint32_t x = 0; x < widths[plane]; ++x) {
          const std::uint16_t sample =
              src[static_cast<std::size_t>(y) * widths[plane] + x];
          if (bps == 1) {
            row[x] = static_cast<std::uint8_t>(sample);
          } else {
            row[static_cast<std::size_t>(x) * 2U] =
                static_cast<std::uint8_t>(sample & 0xFFU);
            row[static_cast<std::size_t>(x) * 2U + 1U] =
                static_cast<std::uint8_t>((sample >> 8) & 0x03U);
          }
        }
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size()));
      }
    }
  }
  (void)has_chroma;
  if (!out) {
    error = "failed while writing --video-recon output";
    return false;
  }
  return true;
}

// Two-pass calibration hook for --rc vbr: scales the built-in placeholder
// bitrate model so that its estimate at the pass-1 QP matches the measured
// pass-1 payload bits. The controller then bisects the calibrated model. This
// corrects the model's absolute scale, which is documented as a placeholder.
struct ScaledBitrateModel {
  double scale = 1.0;
};

double scaled_bitrate_estimate(void* context,
                               const av2enc::rc::FrameComplexity& complexity,
                               int base_q_idx) {
  const auto* model = static_cast<const ScaledBitrateModel*>(context);
  return av2enc::rc::estimate_frame_bits_model(complexity, base_q_idx) *
         model->scale;
}

// Encoder coding order for the reordering schedules.  The rate-control loop
// must walk the pictures in the order they are actually coded: the B and
// hierarchical-P schedules emit units in coding order, and the complexity
// proxy for a picture is the source of its primary reference, not the
// previous display frame.  `primary_ref[display]` is -1 for an intra key.
struct VideoCodingSchedule {
  std::vector<int> order;        // display indices in coding order
  std::vector<int> primary_ref;  // primary reference display, -1 for intra
};

bool video_coding_schedule(const Options& options, int frame_count,
                           VideoCodingSchedule& out, std::string& error) {
  out = VideoCodingSchedule{};
  if (options.b_frames != 0) {
    namespace bf = av2enc::inter::bframes;
    const int gop_size = static_cast<int>(options.b_frames) + 2;
    const int num_gops = std::max(1, options.video_gops);
    bf::SequenceConfig sc;
    sc.num_gops = num_gops;
    sc.gop_size = gop_size;
    sc.pattern = bf::GopPattern::kIpbb;
    int order_hint_bits = 4;
    while ((1 << order_hint_bits) < frame_count) ++order_hint_bits;
    sc.order_hint_bits = order_hint_bits;
    sc.num_slots = 16;
    sc.base_q_idx = static_cast<int>(options.qp);
    bf::Plan plan;
    bf::PlanError plan_error = bf::PlanError::kOk;
    if (!bf::build_sequence(sc, plan, &plan_error)) {
      error = std::string("B-frame rate-control schedule: ") +
              bf::plan_error_name(plan_error);
      return false;
    }
    out.order = plan.coding_order;
    out.primary_ref.assign(static_cast<std::size_t>(plan.display_count()), -1);
    for (const bf::FramePlan& fp : plan.frames) {
      if (!fp.is_key) {
        out.primary_ref[static_cast<std::size_t>(fp.display_index)] =
            fp.ref_display[0];
      }
    }
    return true;
  }
  // --svc-temporal N is the same hierarchical-P closed GOP with 1 << N
  // pictures (the SVC writer expands it into temporal units).
  const bool svc_temporal =
      options.svc_given && options.svc_temporal > 1;
  if (options.video_tlayers > 0 || svc_temporal) {
    namespace tl = av2enc::inter::tlayers;
    const int gop = svc_temporal ? (1 << options.svc_temporal)
                                 : options.video_tlayers;
    tl::PlanConfig pc;
    pc.gop_size = gop;
    pc.schedule = tl::Schedule::kHierarchicalP;
    pc.ref_mode = tl::RefMode::kTwo;
    pc.dependency = tl::DependencyMode::kAllLower;
    pc.explicit_dependency_map = false;
    pc.order_hint_bits = gop >= 16 ? 5 : 4;
    pc.num_slots = gop;
    pc.base_q_idx = static_cast<std::uint8_t>(options.qp);
    tl::Plan plan;
    tl::PlanError plan_error = tl::PlanError::kOk;
    if (!tl::build_plan(pc, plan, &plan_error)) {
      error = std::string("temporal-layer rate-control schedule: ") +
              tl::plan_error_name(plan_error);
      return false;
    }
    out.order = plan.coding_order;
    out.primary_ref.assign(static_cast<std::size_t>(plan.display_count()), -1);
    for (const tl::PlannedFrame& pf : plan.frames) {
      if (!pf.is_key) {
        out.primary_ref[static_cast<std::size_t>(pf.display_index)] =
            pf.ref_display[0];
      }
    }
    return true;
  }
  return true;  // flat key+P: display order, previous display frame
}

// Per-frame base_q_idx planning for --video. Empty output means the historical
// fixed --qp stream (the default). CRF and VBR go through the shared rate
// controller and the per-frame RC adapter so the frame header and the coefficient-CDF
// slice always agree. VBR runs a statistics pass at --qp first: those payload
// bits calibrate the placeholder bitrate model, which is this encoder's
// two-pass step.
bool plan_video_rate_control(
    const Options& options, const av2enc::Y4mHeader& y4m,
    const std::vector<av2enc::video::VideoInputFrame>& inputs,
    const std::vector<std::uint32_t>* pass1_payload_bits,
    std::vector<std::uint8_t>* frame_qp, std::string& error) {
  frame_qp->clear();
  if (options.rc_mode == "cqp") {
    return true;
  }
  if (options.rc_mode == "vbr" && (y4m.fps_num == 0 || y4m.fps_den == 0)) {
    error = "--rc vbr needs a non-zero frame rate in the Y4M header";
    return false;
  }

  const int width = static_cast<int>(inputs[0].width);
  const int height = static_cast<int>(inputs[0].height);
  const std::vector<std::uint8_t> zero_plane(
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);

  // Reordering schedules: plan each picture in coding order using its actual
  // primary reference as the motion-free inter proxy.  The flat key+P chain
  // keeps the historical display-order/previous-frame model.
  VideoCodingSchedule schedule;
  if (!video_coding_schedule(options, static_cast<int>(inputs.size()),
                             schedule, error)) {
    return false;
  }
  if (schedule.order.empty()) {
    schedule.order.resize(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      schedule.order[i] = static_cast<int>(i);
    }
  }
  // `primary_ref` is only populated by video_coding_schedule for the
  // reordering paths; an empty vector means the historical proxy.
  const bool has_primary_ref = !schedule.primary_ref.empty();
  if (has_primary_ref &&
      schedule.primary_ref.size() != inputs.size()) {
    error = "rate-control schedule does not cover every input picture";
    return false;
  }

  std::vector<av2enc::rc::FrameComplexity> complexity(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    std::span<const std::uint8_t> predicted(zero_plane);
    int ref = i > 0 ? static_cast<int>(i) - 1 : -1;
    if (has_primary_ref) ref = schedule.primary_ref[i];
    if (ref >= 0 && ref < static_cast<int>(inputs.size())) {
      predicted = inputs[static_cast<std::size_t>(ref)].y;
    }
    complexity[i] = av2enc::rc::complexity_from_u8_planes(
        inputs[i].y, predicted, width, height, width);
    complexity[i].intra_fraction = ref < 0 ? 1.0 : 0.0;
  }

  av2enc::video::post::VideoRcConfig rc_config;
  rc_config.depth = av2enc::fmt::BitDepth::k8;
  rc_config.policy.min_base_q_idx = av2enc::rc::kMinQIndex;
  rc_config.policy.max_base_q_idx = av2enc::rc::kMaxQIndex;
  ScaledBitrateModel model_scale;
  if (options.rc_mode == "crf") {
    rc_config.rc.mode = av2enc::rc::RateControlMode::kCrf;
    rc_config.rc.crf.crf = options.rc_crf;
  } else {
    rc_config.rc.mode = av2enc::rc::RateControlMode::kVbr;
    rc_config.rc.vbr.target_bitrate_bps = options.rc_bitrate_kbps * 1000.0;
    rc_config.rc.vbr.frame_rate_num = static_cast<std::int32_t>(y4m.fps_num);
    rc_config.rc.vbr.frame_rate_den = static_cast<std::int32_t>(y4m.fps_den);
    if (pass1_payload_bits != nullptr) {
      // Calibrate against the pass-1 payload at the fixed --qp.  The encoder
      // emitted tile_bytes in coding order, so pair each payload with the
      // complexity of the picture at that coding position.
      double measured = 0.0;
      double modelled = 0.0;
      for (std::size_t c = 0; c < schedule.order.size(); ++c) {
        const std::size_t display =
            static_cast<std::size_t>(schedule.order[c]);
        if (c >= pass1_payload_bits->size()) break;
        measured += static_cast<double>((*pass1_payload_bits)[c]);
        modelled += av2enc::rc::estimate_frame_bits_model(
            complexity[display], static_cast<int>(options.qp));
      }
      if (modelled > 0.0 && std::isfinite(measured)) {
        model_scale.scale =
            std::clamp(measured / modelled, 1.0e-3, 1.0e3);
      }
      rc_config.rc.estimator.context = &model_scale;
      rc_config.rc.estimator.estimate = &scaled_bitrate_estimate;
    }
  }

  av2enc::video::post::VideoRateControl rc;
  if (!rc.configure(rc_config, error)) {
    return false;
  }

  frame_qp->resize(inputs.size());
  // The virtual buffer walks the actual coding order.  frame_qp itself stays
  // indexed by display order because that is how the encoder's per-frame plan
  // looks up each picture's base_q_idx.
  for (const int display_index : schedule.order) {
    const std::size_t display = static_cast<std::size_t>(display_index);
    av2enc::video::post::VideoFrameQp qp;
    if (!rc.begin_frame(complexity[display], qp, error)) {
      return false;
    }
    (*frame_qp)[display] = static_cast<std::uint8_t>(qp.base_q_idx);
    // Feed the controller's own (calibrated) estimate back into the virtual
    // buffer: the encoding pass is open-loop, so there is no later measurement.
    const std::int64_t actual_bits =
        static_cast<std::int64_t>(std::llround(qp.estimated_bits));
    if (!rc.end_frame(actual_bits, error)) {
      return false;
    }
  }
  return true;
}

// True when two input pictures carry identical source planes.  The reference-
// intro QP plan uses this to keep a duplicated frame pair on one base_q_idx;
// the comparison is exact and deterministic (no tolerance, no downscaling).
bool video_sources_identical(const av2enc::video::VideoInputFrame& a,
                             const av2enc::video::VideoInputFrame& b) {
  if (a.width != b.width || a.height != b.height) return false;
  if (!a.y.empty() || !b.y.empty()) {
    return a.y.size() == b.y.size() && a.u.size() == b.u.size() &&
           a.v.size() == b.v.size() &&
           std::equal(a.y.begin(), a.y.end(), b.y.begin()) &&
           std::equal(a.u.begin(), a.u.end(), b.u.begin()) &&
           std::equal(a.v.begin(), a.v.end(), b.v.begin());
  }
  return a.y16.size() == b.y16.size() && a.u16.size() == b.u16.size() &&
         a.v16.size() == b.v16.size() &&
         std::equal(a.y16.begin(), a.y16.end(), b.y16.begin()) &&
         std::equal(a.u16.begin(), a.u16.end(), b.u16.begin()) &&
         std::equal(a.v16.begin(), a.v16.end(), b.v16.begin());
}

// Deterministic reference-intro QP plan for --video-key-qp-offset.  The flat
// key+P chain codes in display order: the key frame takes --qp + offset and a
// picture whose source is byte-identical to its predecessor inherits the
// predecessor's QP, so a duplicated opening pair (f1 == f0) is coded at one
// QP instead of splitting the pair.  Every other frame keeps --qp.  The result
// feeds the same frame_base_q_idx plumbing as --rc (frame header and
// coefficient-CDF qctx follow it); without the option the plan stays empty, so
// the fixed-QP stream is untouched.
bool plan_video_frame_qp(const Options& options,
                         std::span<const av2enc::video::VideoInputFrame> inputs,
                         std::vector<std::uint8_t>* frame_qp,
                         std::string& error) {
  frame_qp->clear();
  if (!options.video_key_qp_offset_given) return true;
  if (inputs.empty()) {
    error = "the reference-intro QP plan needs at least one frame";
    return false;
  }
  const int key_qp =
      static_cast<int>(options.qp) + options.video_key_qp_offset;
  if (key_qp < 1 || key_qp > 255) {
    error = "the key frame base_q_idx " + std::to_string(key_qp) +
            " is outside [1, 255]";
    return false;
  }
  frame_qp->resize(inputs.size());
  (*frame_qp)[0] = static_cast<std::uint8_t>(key_qp);
  for (std::size_t i = 1; i < inputs.size(); ++i) {
    (*frame_qp)[i] = video_sources_identical(inputs[i - 1], inputs[i])
                         ? (*frame_qp)[i - 1]
                         : static_cast<std::uint8_t>(options.qp);
  }
  return true;
}

// Host-only multi-GPU planning dump (--gpu-plan). It reads the Y4M header for
// geometry/frame rate, plans the tile layout and the deterministic per-GPU
// assignment, prints the contract and exits. No CUDA context, allocation,
// kernel, stream or device query happens here, and no IVF is written.
int run_gpu_plan(const Options& options) {
  std::string error;
  av2enc::Y4mReader reader(options.input);
  if (!reader.open(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }
  const av2enc::Y4mHeader& y4m = reader.header();
  if (y4m.width % 2U != 0U || y4m.height % 2U != 0U) {
    std::cerr << "av2enc: error: --gpu-plan requires even dimensions, got "
              << y4m.width << "x" << y4m.height << "\n";
    return 2;
  }

  // An explicit --tiles layout is a promise: never replace it with a default.
  // Without one, use the GPU-count default and fall back through the verified
  // presets when the preferred grid is not representable at this frame size.
  const bool explicit_tiles = flag_given(options, "--tiles");
  using av2enc::tiles::TileLayoutPreset;
  const TileLayoutPreset preferred =
      explicit_tiles ? tile_preset_from_layout(options.tiles)
                     : av2enc::multigpu::default_layout_for_gpus(
                           options.gpu_plan);
  std::vector<TileLayoutPreset> candidates;
  if (explicit_tiles) {
    candidates.push_back(preferred);
  } else {
    candidates.push_back(preferred);
    static constexpr TileLayoutPreset kFallbacks[] = {
        TileLayoutPreset::k16x2, TileLayoutPreset::k8x2,
        TileLayoutPreset::k4x2,  TileLayoutPreset::k2x2,
        TileLayoutPreset::k2x1,  TileLayoutPreset::k1x1};
    for (const TileLayoutPreset fallback : kFallbacks) {
      if (fallback != preferred) candidates.push_back(fallback);
    }
  }

  TileLayoutPreset preset = preferred;
  av2enc::tiles::LayoutResult planned;
  std::string first_detail;
  bool chosen = false;
  for (const TileLayoutPreset candidate : candidates) {
    const av2enc::tiles::TileLayoutDims dims =
        av2enc::tiles::preset_dims(candidate);
    // Level 5.1 (idx 7) is the M1 default, but Table A.9 MaxTileCols only
    // admits 16 columns from level 6.0. Plan (and report) a level that carries
    // the grid.
    const int plan_level = std::max(
        7, av2enc::tiles::minimum_level_for_grid(dims.cols, dims.rows));
    av2enc::tiles::LayoutResult attempt = av2enc::tiles::plan_tile_layout(
        static_cast<int>(y4m.width), static_cast<int>(y4m.height), dims.cols,
        dims.rows, av2enc::tiles::kSuperblockPixels64, plan_level, 0);
    if (!attempt.ok() || attempt.layout.tile_count() < options.gpu_plan) {
      if (first_detail.empty()) {
        first_detail = attempt.ok() ? "only " +
                                          std::to_string(
                                              attempt.layout.tile_count()) +
                                          " tile(s) for " +
                                          std::to_string(options.gpu_plan) +
                                          " GPU(s)"
                                    : attempt.error;
      }
      continue;
    }
    preset = candidate;
    planned = std::move(attempt);
    chosen = true;
    break;
  }
  if (!chosen) {
    std::cerr << "av2enc: error: --gpu-plan: ";
    if (explicit_tiles) {
      std::cerr << "layout " << av2enc::tiles::preset_name(preferred)
                << " rejected: " << first_detail;
    } else {
      std::cerr << "no tile layout fits " << y4m.width << "x" << y4m.height
                << " for " << options.gpu_plan << " GPU(s); preferred "
                << av2enc::tiles::preset_name(preferred) << ": "
                << first_detail;
    }
    std::cerr << "\n";
    return 2;
  }

  av2enc::multigpu::PlannerConfig config;
  config.gpu_count = options.gpu_plan;
  config.bit_depth = 8;
  // The plan has no device query (CPU-only host contract). Use --vram-cap when
  // the caller has one; otherwise the planner's compat8g emulation cap applies.
  // This is an estimate, never an allocation.
  if (options.vram_cap_mib != 0) {
    config.vram_cap_mb = static_cast<double>(options.vram_cap_mib);
  }
  config.target_fps = y4m.fps_den != 0
                          ? static_cast<double>(y4m.fps_num) /
                                static_cast<double>(y4m.fps_den)
                          : 1.0;

  av2enc::multigpu::MultiGpuPlan plan;
  if (!av2enc::multigpu::plan_multi_gpu(planned.layout, config, plan, error)) {
    std::cerr << "av2enc: error: --gpu-plan: " << error << "\n";
    return 2;
  }

  std::cout << "av2enc: host-only multi-GPU tile plan (no device allocation, "
               "no stream)\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.fps_num << "/" << y4m.fps_den
            << " fps)\n"
            << "  layout:  " << av2enc::tiles::preset_name(preset) << " ("
            << plan.tile_count << " tiles), sb 64x64\n"
            << "  gpus:    " << config.gpu_count << ", "
            << av2enc::multigpu::assignment_policy_name(config.assignment)
            << "\n";
  if (!explicit_tiles && preset != preferred) {
    std::cout << "  note:    preferred layout "
              << av2enc::tiles::preset_name(preferred)
              << " does not fit; using " << av2enc::tiles::preset_name(preset)
              << "\n";
  }
  for (const av2enc::multigpu::GpuPlan& gpu : plan.gpus) {
    std::cout << "  gpu " << gpu.gpu_index << ":  tiles";
    for (const int tile : gpu.tile_indices) {
      std::cout << " " << tile;
    }
    std::cout << " (" << gpu.tile_count << ", " << gpu.tile_area_pixels
              << " px), vram " << gpu.vram.live_mb << " / " << gpu.vram.cap_mb
              << " MB fits=" << (gpu.vram.fits ? "yes" : "no") << "\n";
  }
  std::cout << "  dispatch:";
  for (const av2enc::multigpu::TileDispatchEntry& entry : plan.dispatch) {
    std::cout << " " << entry.raster_slot << ":" << entry.gpu_index;
  }
  std::cout << " (raster merge order; each tile is one work item)\n";
  std::cout << "  host:    transfer " << plan.transfer.total_mb << " MB, "
            << plan.transfer.total_ms << " ms ("
            << plan.transfer.pct_of_frame_budget << "% of the frame budget, "
            << plan.transfer.phases << " phases)\n"
            << "  scale:   model only (measured=" << (plan.measured ? 1 : 0)
            << ", evidence="
            << av2enc::multigpu::evidence_name(plan.evidence)
            << "), estimated speedup " << plan.scaling.speedup
            << "x; not a measurement\n"
            << "  note:    " << plan.disclaimer << "\n";
  return 0;
}

// Frame-count and order-hint preflight for the frame-type modes. The
// sequence encoders discover these after IvfWriter::open, so an invalid
// command used to truncate an existing -o stream and exit 1. Validate once the
// input length is known and before the writer is opened; the caller reports
// the message as a usage error (exit 2).
bool validate_video_sequence_length(const Options& options,
                                    std::size_t frame_count,
                                    std::string& error) {
  if (options.video_olk && frame_count != 4) {
    error = "--video-olk requires exactly 4 input frames (CLK, OLK, P2, P3); "
            "got " + std::to_string(frame_count);
    return false;
  }
  if (options.video_switch && frame_count != 2) {
    error = "--video-switch requires exactly 2 input frames (CLK, SWITCH); "
            "got " + std::to_string(frame_count);
    return false;
  }
  if (options.video_ras && frame_count != 3) {
    error = "--video-ras requires exactly 3 input frames (CLK, SWITCH, RAS); "
            "got " + std::to_string(frame_count);
    return false;
  }
  if (options.video_sef && frame_count < 2) {
    error = "--video-sef needs at least 2 input frames; got " +
            std::to_string(frame_count);
    return false;
  }
  if (options.video_leading != 0) {
    const std::size_t lead = static_cast<std::size_t>(options.video_leading);
    if (frame_count < lead + 2U) {
      error = "--video-leading " + std::to_string(options.video_leading) +
              " needs at least " + std::to_string(lead + 2U) +
              " input frames; got " + std::to_string(frame_count);
      return false;
    }
    const std::size_t regular = frame_count - lead - 2U;
    if (lead + regular > 7U) {
      error = "--video-leading: leading + regular frames exceed the 8 slots "
              "this profile signals (pass at most 9 input frames)";
      return false;
    }
  }
  if (options.video_tlayers != 0) {
    const std::size_t gop = static_cast<std::size_t>(options.video_tlayers);
    // `--video-gops > 1` is the chained form.  A single picture is legal with
    // or without it (--frames 1): video_frame_sequence.cpp reduces the schedule
    // to a lone key, which is what a one-picture stream is.  Without this the
    // chained presets (--nightly-b) could not be used on a still.
    if (tlayers_chain_enabled(options) &&
        (options.video_gops > 1 || frame_count == 1)) {
      // `--video-gops M` is the maximum number of 8-frame blocks; any frame
      // count up to M*gop is legal and the remainder becomes a dyadic tail
      // block (video_frame_sequence.cpp does the decomposition and names an
      // unrepresentable remainder).  This check only rejects a count that
      // cannot fit at all.
      const std::size_t full =
          gop * static_cast<std::size_t>(options.video_gops);
      // One tail block of up to 4 pictures past the full blocks, so the
      // historical 60 = 7*8 + 4 spelling still fits --video-gops 7.
      if (frame_count > full + 4U) {
        error = "--video-tlayers --video-gops " +
                std::to_string(options.video_gops) + " codes at most " +
                std::to_string(full + 4U) + " frames (" +
                std::to_string(options.video_gops) + " blocks of " +
                std::to_string(gop) + " plus a tail of up to 4); got " +
                std::to_string(frame_count) +
                " (raise --video-gops to fit more)";
        return false;
      }
    } else if (frame_count != gop) {
      error = "--video-tlayers codes one closed GOP: pass exactly " +
              std::to_string(options.video_tlayers) + " input frames; got " +
              std::to_string(frame_count);
      return false;
    }
  }
  if (options.b_frames != 0) {
    const std::size_t per_gop = static_cast<std::size_t>(options.b_frames) + 2U;
    const std::size_t gops = static_cast<std::size_t>(std::max(1, options.video_gops));
    if (frame_count != per_gop * gops) {
      error = "--b-frames --video-gops " + std::to_string(gops) +
              " codes " + std::to_string(gops) + " closed GOP(s): pass exactly " +
              std::to_string(per_gop * gops) +
              " frames (N B frames + key + P anchor per GOP); got " +
              std::to_string(frame_count);
      return false;
    }
  }
  if (options.video_long_term && frame_count > 16U) {
    error = "--video-long-term supports at most 16 input frames "
            "(OrderHintBits = 4); got " + std::to_string(frame_count);
    return false;
  }
  return true;
}

// Video path for the non-default pixel formats (10-bit 4:2:0/4:2:2/4:4:4/4:0:0
// and 8-bit 4:2:2/4:4:4/4:0:0).  The input reader is the strict format Y4M
// reader; the requested --bit-depth/--chroma must match the stream tag exactly
// (no implicit resampling or depth conversion).
int run_video_format_encode(const Options& options) {
  // Tool flags that are 8-bit 4:2:0 only.  Reject by name instead of silently
  // ignoring them on the format path.
  const std::pair<bool, const char*> unsupported[] = {
      {options.tiles != "1x1", "--tiles"},
      {options.video_tx_search, "--video-tx-search"},
      {options.video_key_intra_txpart, "--video-key-intra-txpart"},
      {options.video_partition_rd, "--video-partition-rd"},
      {options.video_tcq, "--video-tcq"},
      {options.video_rdoq, "--video-rdoq"},
      {options.video_cctx, "--video-cctx"},
      {options.video_intra_full, "--video-intra-full"},
      {options.video_ibp, "--video-ibp"},
      {options.video_gme, "--video-gme"},
      {options.video_warp, "--video-warp"},
      {options.video_compound, "--video-compound"},
      {options.video_compound_near_near, "--video-compound-near-near"},
      {options.video_opfl, "--video-opfl"},
      {options.video_refine_mv, "--video-refine-mv"},
      {options.video_cwp_weighted, "--video-cwp-weighted"},
      {options.video_imp_mask, "--video-imp-mask"},
      {options.video_df_sub_pu, "--video-df-sub-pu"},
      {options.video_drl_reorder || options.video_drl_reorder_always,
       "--video-drl-reorder"},
      {options.b_frames != 0, "--b-frames"},
      {options.video_tlayers != 0, "--video-tlayers"},
      {options.video_long_term, "--video-long-term"},
      {options.video_olk, "--video-olk"},
      {options.video_leading != 0, "--video-leading"},
      {options.video_sef, "--video-sef"},
      {options.video_switch, "--video-switch"},
      {options.video_ras, "--video-ras"},
      {options.avg_cdf, "--avg-cdf"},
      {options.adaptive_cdf, "--adaptive-cdf"},
      {options.cdf_continuity, "--cdf-continuity"},
      {options.video_mfh, "--video-mfh"},
      {options.video_timing, "--video-timing"},
      {options.video_decoder_model, "--video-decoder-model"},
      {options.video_brt, "--video-brt"},
      {options.metadata_timecode, "--metadata-timecode"},
      {options.metadata_scan_type, "--metadata-scan-type"},
      {options.metadata_cll, "--metadata-cll"},
      {options.metadata_mdcv, "--metadata-mdcv"},
      {options.metadata_icc, "--metadata-icc"},
      {options.metadata_temporal_point, "--metadata-temporal-point"},
      {options.metadata_frame_hash, "--metadata-frame-hash"},
      {options.metadata_banding, "--metadata-banding"},
      {options.metadata_user_data, "--metadata-user-data"},
      {options.rc_mode != "cqp", "--rc"},
  };
  for (const auto& [set, flag] : unsupported) {
    if (set) {
      std::cerr << "av2enc: error: " << flag
                << " is not wired for the --bit-depth/--chroma video formats "
                   "(8-bit 4:2:0 only)\n";
      return 2;
    }
  }

  av2enc::fmt::Format format;
  format.chroma = options.chroma == "422"   ? av2enc::fmt::ChromaFormat::k422
                  : options.chroma == "444" ? av2enc::fmt::ChromaFormat::k444
                  : options.chroma == "400" ? av2enc::fmt::ChromaFormat::k400
                                            : av2enc::fmt::ChromaFormat::k420;
  format.depth = options.bit_depth == 10 ? av2enc::fmt::BitDepth::k10
                                         : av2enc::fmt::BitDepth::k8;

  std::size_t limit = 0;
  if (options.max_frames != 0) {
    limit = static_cast<std::size_t>(options.max_frames);
  } else if (options.p_frames != 0) {
    limit = static_cast<std::size_t>(options.p_frames) + 1U;
  }

  std::vector<av2enc::fmt::Picture> pictures;
  av2enc::fmt::Y4mFormatInfo info;
  std::string error;
  if (!av2enc::fmt::read_y4m_sequence(options.input, &pictures, &info, limit,
                                      error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  if (info.chroma != format.chroma || info.depth != format.depth) {
    std::cerr << "av2enc: error: --bit-depth/--chroma request "
              << av2enc::fmt::bit_depth_name(format.depth) << " "
              << av2enc::fmt::chroma_format_name(format.chroma)
              << " does not match the input coloursapce "
              << av2enc::fmt::chroma_format_name(info.chroma) << " "
              << av2enc::fmt::bit_depth_name(info.depth)
              << "; the format paths never convert implicitly\n";
    return 2;
  }
  format.width = info.width;
  format.height = info.height;
  // The format core codes the §5.18.4.4 MI-aligned picture (MiCols x MiRows
  // rounded up to 8 pixels) and crops the output back to the real size.  Only
  // the chroma layout constrains the input: 4:2:0 needs even width/height,
  // 4:2:2 needs an even width, and 4:4:4/4:0:0 accept any size.
  if (av2enc::fmt::has_chroma(format.chroma) &&
      ((av2enc::fmt::subsampling_x(format.chroma) != 0 &&
        (format.width & 1U) != 0) ||
       (av2enc::fmt::subsampling_y(format.chroma) != 0 &&
        (format.height & 1U) != 0))) {
    std::cerr << "av2enc: error: "
              << av2enc::fmt::chroma_format_name(format.chroma)
              << " cannot represent " << format.width << "x" << format.height
              << " (chroma subsampling requires even coded dimensions)\n";
    return 2;
  }
  if (!av2enc::fmt::is_signalable(format)) {
    std::cerr << "av2enc: error: the requested format is not signalable in "
                 "AV2 v1.0.0\n";
    return 2;
  }

  const int sub_x = av2enc::fmt::subsampling_x(format.chroma);
  const int sub_y = av2enc::fmt::subsampling_y(format.chroma);
  const bool has_chroma = av2enc::fmt::has_chroma(format.chroma);
  std::vector<std::vector<std::uint16_t>> owned_y;
  std::vector<std::vector<std::uint16_t>> owned_u;
  std::vector<std::vector<std::uint16_t>> owned_v;
  owned_y.reserve(pictures.size());
  owned_u.reserve(pictures.size());
  owned_v.reserve(pictures.size());
  for (const av2enc::fmt::Picture& picture : pictures) {
    const std::uint32_t cw = format.width >> sub_x;
    const std::uint32_t ch = format.height >> sub_y;
    std::vector<std::uint16_t> y(
        static_cast<std::size_t>(format.width) * format.height);
    for (std::uint32_t r = 0; r < format.height; ++r) {
      std::memcpy(y.data() + static_cast<std::size_t>(r) * format.width,
                  picture.planes[0].samples.data() +
                      static_cast<std::size_t>(r) * picture.planes[0].stride,
                  static_cast<std::size_t>(format.width) *
                      sizeof(std::uint16_t));
    }
    owned_y.push_back(std::move(y));
    std::vector<std::uint16_t> u;
    std::vector<std::uint16_t> v;
    if (has_chroma) {
      u.assign(static_cast<std::size_t>(cw) * ch, 0);
      v.assign(static_cast<std::size_t>(cw) * ch, 0);
      for (std::uint32_t r = 0; r < ch; ++r) {
        std::memcpy(u.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[1].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[1].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
        std::memcpy(v.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[2].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[2].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
      }
    }
    owned_u.push_back(std::move(u));
    owned_v.push_back(std::move(v));
  }

  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(pictures.size());
  for (std::size_t i = 0; i < pictures.size(); ++i) {
    av2enc::video::VideoInputFrame input;
    input.width = format.width;
    input.height = format.height;
    input.format = format;
    input.y16 = std::span<const std::uint16_t>(owned_y[i].data(),
                                               owned_y[i].size());
    input.u16 = std::span<const std::uint16_t>(owned_u[i].data(),
                                               owned_u[i].size());
    input.v16 = std::span<const std::uint16_t>(owned_v[i].data(),
                                               owned_v[i].size());
    inputs.push_back(input);
  }

  if (!validate_video_sequence_length(options, inputs.size(), error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(format.width);
  ivf.height = static_cast<std::uint16_t>(format.height);
  ivf.fps_num = info.fps_num;
  ivf.fps_den = info.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  av2enc::video::VideoOptions video_options;
  video_options.format = format;
  video_options.base_q_idx = static_cast<std::uint16_t>(options.qp);
  video_options.num_total_refs = options.video_refs;
  video_options.num_ref_frames = options.video_num_ref_frames;
  video_options.implicit_ref_map = options.video_implicit_ref_map;
  video_options.enable_inter_residual = options.video_inter_residual;
  video_options.enable_edge_inter = options.video_edge_inter;
  video_options.enable_edge_mask = options.video_edge_mask;
  video_options.enable_leaf_rd = options.video_leaf_rd;
  video_options.me_subpel_eighth = options.video_subpel_eighth;
  video_options.mv_precision = options.video_mv_precision;
  video_options.enable_adaptive_mvd = options.video_adaptive_mvd;
  video_options.enable_mvd_sign_derive = options.video_mvd_sign_derive;
  video_options.enable_flex_mvres = options.video_flex_mvres;
  video_options.enable_ref_frame_mvs = options.video_ref_mvs;
  video_options.ref_mvs_stress = options.video_ref_mvs_stress;
  video_options.force_single_block = options.video_force_single;
  video_options.skip_collapse = options.video_skip_collapse;
  video_options.sb128 = options.video_sb128;
  video_options.sb256 = options.video_sb256;
  video_options.ext_partitions = options.video_ext_partitions;
  video_options.min_block_size = options.video_min_block;
  video_options.max_partition_depth = options.video_max_depth;
  if (format.chroma == av2enc::fmt::ChromaFormat::k422 &&
      !flag_given(options, "--video-max-depth")) {
    // Historical 4:2:2 default: 32x32 roots.  16-pixel 4:2:2 leaves are now
    // exact (partition models 4:2:2 chroma), so an explicit
    // --video-max-depth 3/4 is accepted; the default stream stays unchanged.
    video_options.max_partition_depth = 2;
  }
  video_options.me_range = options.video_me_range;
  video_options.trace = options.video_trace;
  video_options.enable_deblock = options.video_deblock;
  video_options.loop_filters.cdef = options.video_cdef;
  video_options.loop_filters.cdef_num_strengths = options.video_cdef_strength;
  video_options.loop_filters.cdef_damping = options.video_cdef_damping;
  video_options.loop_filters.cdef_rd = options.video_cdef_rd;
  video_options.loop_filters.cdef_rd_lambda = options.video_cdef_rd_lambda;
  video_options.loop_filters.cdef_search_mode =
      static_cast<av2enc::video::post::VideoLoopFilterOptions::CdefSearchMode>(
          options.video_cdef_search);
  video_options.loop_filters.ccso = options.video_ccso;
  video_options.loop_filters.ccso_bands_log2 = options.video_ccso_bands;
  video_options.loop_filters.ccso_rd = options.video_ccso_rd;
  video_options.loop_filters.ccso_lambda = options.video_ccso_lambda;
  video_options.loop_filters.ccso_refit_passes =
      options.video_ccso_refit_passes;
  video_options.loop_filters.ccso_unit_bits = options.video_ccso_unit_bits;
  video_options.loop_filters.ccso_header_rd = options.video_ccso_header_rd;
  video_options.loop_filters.cfl_ds_filter_index = options.video_cfl_ds_filter;
  video_options.loop_filters.wiener = options.video_wiener;
  video_options.loop_filters.pc_wiener = options.video_pc_wiener;
  video_options.loop_filters.chroma_lr = options.video_chroma_lr;
  video_options.loop_filters.gdf = options.video_gdf;
  video_options.film_grain = options.video_grain;

  if (options.video_gpu &&
      (format.width % 8U != 0U || format.height % 8U != 0U)) {
    std::cerr << "av2enc: error: --video-gpu is CPU-only for dimensions not "
                 "divisible by 8 (the CUDA kernels assume an MI-aligned "
                 "picture); rerun without --video-gpu, got "
              << format.width << "x" << format.height << "\n";
    return 2;
  }
  if (options.video_gpu &&
      (video_options.enable_deblock || video_options.loop_filters.any())) {
    std::cerr << "av2enc: error: the format video loop-filter stage is CPU-only; "
                 "use --video (the 8-bit 4:2:0 --video-gpu path has its own "
                 "device filters)\n";
    return 2;
  }
  av2enc::video::VideoSequenceResult result;
  bool ok = false;
  if (options.video_gpu) {
#if AV2ENC_VIDEO_GPU_CALLABLE
    ok = av2enc::cuda::video::encode_video_sequence_gpu(inputs, video_options,
                                                        result, error);
    if (ok) {
      std::cerr << "av2enc: --video-gpu backend="
                << av2enc::cuda::video::video_backend_name()
                << " frame_bytes_from_cpu=0\n";
    }
#else
    error = "--video-gpu is not linked in this build";
#endif
  } else {
    ok = av2enc::video::encode_video_sequence(inputs, video_options, result,
                                              error);
  }
  if (!ok) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  for (std::size_t i = 0; i < result.units.size(); ++i) {
    if (!writer.write_frame(result.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    if (!write_video_recon_y4m_format(options.video_recon, format,
                                      info.fps_num, info.fps_den, result.recon,
                                      error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t total_tile = 0;
  std::uint64_t total_inter = 0;
  std::uint64_t total_intra = 0;
  for (std::size_t i = 0; i < result.tile_bytes.size(); ++i) {
    total_tile += result.tile_bytes[i];
    total_inter += result.inter_blocks[i];
    total_intra += result.intra_blocks[i];
  }
  std::cout << "av2enc: Video P-frame encode complete\n"
            << "  input:   " << options.input.string() << " (" << format.width
            << "x" << format.height << ", " << info.colorspace << ", "
            << info.fps_num << "/" << info.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << result.units.size() << " temporal unit(s), "
            << result.tile_bytes.size() << " coded frame(s), "
            << result.recon.size() << " output frame(s), base_q_idx "
            << static_cast<int>(video_options.base_q_idx) << ", tile "
            << total_tile << " bytes, inter blocks " << total_inter
            << ", intra blocks " << total_intra << "\n";
  return 0;
}

// --video-lossless: the plain multi-frame sequence is coded by the lossless
// inter writer (`src/codec/entropy/lossless/inter/`).  That module owns its
// sequence/frame headers (base_q_idx = 0, CodedLossless = 1, TxMode =
// ONLY_4X4), the lossless DPCM/WHT/IDTX block syntax and FSC coefficients, so
// the CLI only supplies the pictures and writes the temporal units/recon.
int run_video_lossless_encode(const Options& options,
                              const av2enc::Y4mHeader& y4m,
                              const std::vector<OwnedVideoFrame>& owned) {
  std::string error;
  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(y4m.width);
  ivf.height = static_cast<std::uint16_t>(y4m.height);
  ivf.fps_num = y4m.fps_num;
  ivf.fps_den = y4m.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::vector<av2enc::lossless::inter::Picture> pictures;
  pictures.reserve(owned.size());
  for (const OwnedVideoFrame& frame : owned) {
    av2enc::lossless::inter::Picture picture;
    picture.width = static_cast<int>(frame.width);
    picture.height = static_cast<int>(frame.height);
    picture.format = av2enc::lossless::inter::ChromaFormat::k420;
    picture.bit_depth = 8;
    picture.y.assign(frame.y.begin(), frame.y.end());
    picture.u.assign(frame.u.begin(), frame.u.end());
    picture.v.assign(frame.v.begin(), frame.v.end());
    pictures.push_back(std::move(picture));
  }

  av2enc::lossless::inter::Options lossless;
  lossless.bit_depth = 8;
  lossless.temporal_layers = options.video_tlayers;
  lossless.b_schedule = options.video_lossless_b;
  lossless.adaptive_cdf = options.adaptive_cdf;
  lossless.enable_fsc = true;
  lossless.enable_dpcm = true;
  lossless.force_intra_fsc = options.lossless_force_fsc;
  lossless.intra_period = options.video_lossless_intra_period;
  if (flag_given(options, "--video-min-block")) {
    lossless.block_size = options.video_min_block;
  } else {
    lossless.block_size = 8;  // the module's default coding block size
  }
  av2enc::lossless::inter::EncodeResult encoded;
  if (!av2enc::lossless::inter::encode_video(pictures, lossless, &encoded,
                                             error)) {
    std::cerr << "av2enc: error: --video-lossless: " << error << "\n";
    return 1;
  }
  if (!encoded.exact) {
    std::cerr << "av2enc: error: --video-lossless reconstruction is not exact "
                 "(max |err| "
              << encoded.max_abs_recon_error << ")\n";
    return 1;
  }
  for (std::size_t i = 0; i < encoded.units.size(); ++i) {
    if (!writer.write_frame(encoded.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    std::vector<av2enc::video::VideoReconFrame> recon;
    recon.reserve(encoded.recon.size());
    for (const av2enc::lossless::inter::Picture& picture : encoded.recon) {
      av2enc::video::VideoReconFrame frame;
      frame.width = static_cast<std::uint32_t>(picture.width);
      frame.height = static_cast<std::uint32_t>(picture.height);
      frame.y.assign(picture.y.begin(), picture.y.end());
      frame.u.assign(picture.u.begin(), picture.u.end());
      frame.v.assign(picture.v.begin(), picture.v.end());
      recon.push_back(std::move(frame));
    }
    if (!write_video_recon_y4m(options.video_recon, y4m, recon, error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t inter_blocks = 0;
  std::uint64_t intra_blocks = 0;
  for (const av2enc::lossless::inter::FrameStats& stats : encoded.stats) {
    inter_blocks += static_cast<std::uint64_t>(stats.inter_blocks);
    intra_blocks += static_cast<std::uint64_t>(stats.intra_blocks);
  }
  std::cout << "av2enc: Video lossless encode complete\n"
            << "  input:   " << options.input.string() << " ("
            << y4m.width << "x" << y4m.height << ", " << y4m.fps_num << "/"
            << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << encoded.units.size() << " temporal unit(s), "
            << encoded.recon.size() << " output frame(s), base_q_idx 0, "
            << writer.bytes_written() << " IVF bytes, inter blocks "
            << inter_blocks << ", intra blocks " << intra_blocks << "\n";
  return 0;
}

// --video-lossless on the format video cores: 10-bit and 8-bit
// 4:2:2/4:4:4/4:0:0.  The lossless writer is depth/chroma generic; the CLI
// only converts the fmt::Picture planes into its uint16_t pictures.
// Entry point for --video-lossless on a non-4:2:0 or 10-bit input.  It reads
// the input with the format Y4M reader (the 8-bit 4:2:0 path uses the plain
// reader) and hands the pictures to the shared lossless writer.
int run_video_lossless_format_encode(
    const Options& options, const av2enc::fmt::Format& format,
    const av2enc::fmt::Y4mFormatInfo& info,
    const std::vector<av2enc::fmt::Picture>& pictures);

int run_video_lossless_any(const Options& options) {
  av2enc::fmt::Format format;
  format.chroma = options.chroma == "422"   ? av2enc::fmt::ChromaFormat::k422
                  : options.chroma == "444" ? av2enc::fmt::ChromaFormat::k444
                  : options.chroma == "400" ? av2enc::fmt::ChromaFormat::k400
                                            : av2enc::fmt::ChromaFormat::k420;
  format.depth = options.bit_depth == 10 ? av2enc::fmt::BitDepth::k10
                                         : av2enc::fmt::BitDepth::k8;

  std::size_t limit = 0;
  if (options.max_frames != 0) {
    limit = static_cast<std::size_t>(options.max_frames);
  } else if (options.p_frames != 0) {
    limit = static_cast<std::size_t>(options.p_frames) + 1U;
  }
  std::vector<av2enc::fmt::Picture> pictures;
  av2enc::fmt::Y4mFormatInfo info;
  std::string error;
  if (!av2enc::fmt::read_y4m_sequence(options.input, &pictures, &info, limit,
                                      error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  if (info.chroma != format.chroma || info.depth != format.depth) {
    std::cerr << "av2enc: error: --bit-depth/--chroma request "
              << av2enc::fmt::bit_depth_name(format.depth) << " "
              << av2enc::fmt::chroma_format_name(format.chroma)
              << " does not match the input coloursapce "
              << av2enc::fmt::chroma_format_name(info.chroma) << " "
              << av2enc::fmt::bit_depth_name(info.depth)
              << "; the format paths never convert implicitly\n";
    return 2;
  }
  format.width = info.width;
  format.height = info.height;
  if (format.width % 8U != 0U || format.height % 8U != 0U) {
    std::cerr << "av2enc: error: --video-lossless requires dimensions "
                 "divisible by 8 (the lossless tile writer codes whole "
                 "superblocks), got "
              << format.width << "x" << format.height << "\n";
    return 2;
  }
  if (!av2enc::fmt::is_signalable(format)) {
    std::cerr << "av2enc: error: the requested format is not signalable in "
                 "AV2 v1.0.0\n";
    return 2;
  }
  if (!validate_video_sequence_length(options, pictures.size(), error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }
  return run_video_lossless_format_encode(options, format, info, pictures);
}

int run_video_lossless_format_encode(
    const Options& options, const av2enc::fmt::Format& format,
    const av2enc::fmt::Y4mFormatInfo& info,
    const std::vector<av2enc::fmt::Picture>& pictures) {
  std::string error;
  const int sub_x = av2enc::fmt::subsampling_x(format.chroma);
  const int sub_y = av2enc::fmt::subsampling_y(format.chroma);
  const bool has_chroma = av2enc::fmt::has_chroma(format.chroma);
  const std::uint32_t cw = has_chroma ? (format.width >> sub_x) : 0;
  const std::uint32_t ch = has_chroma ? (format.height >> sub_y) : 0;

  av2enc::lossless::inter::ChromaFormat chroma =
      av2enc::lossless::inter::ChromaFormat::k420;
  if (format.chroma == av2enc::fmt::ChromaFormat::k422) {
    chroma = av2enc::lossless::inter::ChromaFormat::k422;
  } else if (format.chroma == av2enc::fmt::ChromaFormat::k444) {
    chroma = av2enc::lossless::inter::ChromaFormat::k444;
  } else if (format.chroma == av2enc::fmt::ChromaFormat::k400) {
    chroma = av2enc::lossless::inter::ChromaFormat::k400;
  }
  if (!has_chroma && format.chroma != av2enc::fmt::ChromaFormat::k400) {
    std::cerr << "av2enc: error: --video-lossless: inconsistent monochrome "
                 "format\n";
    return 2;
  }

  std::vector<av2enc::lossless::inter::Picture> lossless_pictures;
  lossless_pictures.reserve(pictures.size());
  for (const av2enc::fmt::Picture& picture : pictures) {
    av2enc::lossless::inter::Picture out;
    out.width = static_cast<int>(format.width);
    out.height = static_cast<int>(format.height);
    out.format = chroma;
    out.bit_depth = format.depth == av2enc::fmt::BitDepth::k10 ? 10 : 8;
    out.y.assign(static_cast<std::size_t>(format.width) * format.height, 0);
    for (std::uint32_t r = 0; r < format.height; ++r) {
      std::memcpy(out.y.data() + static_cast<std::size_t>(r) * format.width,
                  picture.planes[0].samples.data() +
                      static_cast<std::size_t>(r) * picture.planes[0].stride,
                  static_cast<std::size_t>(format.width) *
                      sizeof(std::uint16_t));
    }
    out.u.assign(static_cast<std::size_t>(cw) * ch, 0);
    out.v.assign(static_cast<std::size_t>(cw) * ch, 0);
    if (has_chroma) {
      for (std::uint32_t r = 0; r < ch; ++r) {
        std::memcpy(out.u.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[1].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[1].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
        std::memcpy(out.v.data() + static_cast<std::size_t>(r) * cw,
                    picture.planes[2].samples.data() +
                        static_cast<std::size_t>(r) * picture.planes[2].stride,
                    static_cast<std::size_t>(cw) * sizeof(std::uint16_t));
      }
    }
    lossless_pictures.push_back(std::move(out));
  }

  av2enc::lossless::inter::Options lossless;
  lossless.bit_depth = format.depth == av2enc::fmt::BitDepth::k10 ? 10 : 8;
  lossless.temporal_layers = options.video_tlayers;
  lossless.b_schedule = options.video_lossless_b;
  lossless.adaptive_cdf = options.adaptive_cdf;
  lossless.enable_fsc = true;
  lossless.enable_dpcm = true;
  lossless.force_intra_fsc = options.lossless_force_fsc;
  lossless.intra_period = options.video_lossless_intra_period;
  lossless.block_size =
      flag_given(options, "--video-min-block") ? options.video_min_block : 8;
  av2enc::lossless::inter::EncodeResult encoded;
  if (!av2enc::lossless::inter::encode_video(lossless_pictures, lossless,
                                             &encoded, error)) {
    std::cerr << "av2enc: error: --video-lossless: " << error << "\n";
    return 1;
  }
  if (!encoded.exact) {
    std::cerr << "av2enc: error: --video-lossless reconstruction is not exact "
                 "(max |err| "
              << encoded.max_abs_recon_error << ")\n";
    return 1;
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(format.width);
  ivf.height = static_cast<std::uint16_t>(format.height);
  ivf.fps_num = info.fps_num;
  ivf.fps_den = info.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  for (std::size_t i = 0; i < encoded.units.size(); ++i) {
    if (!writer.write_frame(encoded.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    std::vector<av2enc::video::VideoReconFrame> recon;
    recon.reserve(encoded.recon.size());
    for (const av2enc::lossless::inter::Picture& picture : encoded.recon) {
      av2enc::video::VideoReconFrame frame;
      frame.width = static_cast<std::uint32_t>(picture.width);
      frame.height = static_cast<std::uint32_t>(picture.height);
      frame.format = format;
      frame.y16 = picture.y;
      frame.u16 = picture.u;
      frame.v16 = picture.v;
      recon.push_back(std::move(frame));
    }
    if (!write_video_recon_y4m_format(options.video_recon, format,
                                      info.fps_num, info.fps_den, recon,
                                      error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t inter_blocks = 0;
  std::uint64_t intra_blocks = 0;
  for (const av2enc::lossless::inter::FrameStats& stats : encoded.stats) {
    inter_blocks += static_cast<std::uint64_t>(stats.inter_blocks);
    intra_blocks += static_cast<std::uint64_t>(stats.intra_blocks);
  }
  std::cout << "av2enc: Video lossless encode complete\n"
            << "  input:   " << options.input.string() << " ("
            << format.width << "x" << format.height << ", "
            << info.colorspace << ", " << info.fps_num << "/" << info.fps_den
            << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << encoded.units.size() << " temporal unit(s), "
            << encoded.recon.size() << " output frame(s), base_q_idx 0, "
            << "inter blocks " << inter_blocks << ", intra blocks "
            << intra_blocks << "\n";
  return 0;
}

int run_video_encode(const Options& options, av2enc::Y4mReader& reader,
                     const av2enc::Y4mHeader& y4m) {
  std::vector<OwnedVideoFrame> owned;
  std::string error;
  std::size_t limit = 0;
  if (options.max_frames != 0) {
    limit = static_cast<std::size_t>(options.max_frames);
  } else if (options.p_frames != 0) {
    limit = static_cast<std::size_t>(options.p_frames) + 1U;
  }
  av2enc::Y4mFrame frame;
  while (limit == 0 || owned.size() < limit) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      std::cerr << "av2enc: " << error << "\n";
      return 1;
    }
    OwnedVideoFrame copy;
    copy.width = frame.width();
    copy.height = frame.height();
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    owned.push_back(std::move(copy));
  }
  if (owned.empty()) {
    std::cerr << "av2enc: error: input has no frames\n";
    return 1;
  }

  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(owned.size());
  for (const OwnedVideoFrame& f : owned) {
    av2enc::video::VideoInputFrame input;
    input.width = f.width;
    input.height = f.height;
    input.y = std::span<const std::uint8_t>(f.y.data(), f.y.size());
    input.u = std::span<const std::uint8_t>(f.u.data(), f.u.size());
    input.v = std::span<const std::uint8_t>(f.v.data(), f.v.size());
    inputs.push_back(input);
  }

  if (!validate_video_sequence_length(options, inputs.size(), error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }

  if (options.video_lossless) {
    return run_video_lossless_encode(options, y4m, owned);
  }

  // The output is opened only after every sequence-level usage check above, so
  // an invalid combination never truncates an existing -o stream.
  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(y4m.width);
  ivf.height = static_cast<std::uint16_t>(y4m.height);
  ivf.fps_num = y4m.fps_num;
  ivf.fps_den = y4m.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  av2enc::video::VideoOptions video_options;
  video_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
  video_options.avg_cdf = options.avg_cdf;
  video_options.adaptive_cdf = options.adaptive_cdf;
  video_options.cdf_continuity = options.cdf_continuity;
  video_options.cdf_per_frame_qp = options.video_cdf_per_frame_qp;
  video_options.mfh = options.video_mfh;
  video_options.high_tier = options.video_tier_high;
  video_options.multi_sequence = options.video_multi_sequence;
  video_options.lcr_full = options.video_lcr_full;
  if (options.video_timing) {
    if (y4m.fps_num == 0 || y4m.fps_den == 0) {
      std::cerr << "av2enc: error: --video-timing needs a non-zero Y4M frame "
                   "rate (num_units_in_display_tick/time_scale)\n";
      return 1;
    }
    video_options.video_timing = true;
    video_options.video_decoder_model = options.video_decoder_model;
    video_options.decoder_buffer_delay = options.decoder_buffer_delay;
    video_options.encoder_buffer_delay = options.encoder_buffer_delay;
    video_options.low_delay_mode = options.low_delay_mode;
    video_options.timing_num_units_in_display_tick = y4m.fps_den;
    video_options.timing_time_scale = y4m.fps_num;
  }
  video_options.num_total_refs = options.video_refs;
  video_options.num_ref_frames = options.video_num_ref_frames;
  video_options.implicit_ref_map = options.video_implicit_ref_map;
  video_options.compound = options.video_compound;
  video_options.compound_wedge = options.video_compound_wedge;
  video_options.compound_diffwtd = options.video_compound_diffwtd;
  video_options.compound_near_near = options.video_compound_near_near ||
                                     options.video_opfl || options.video_refine_mv;
  video_options.opfl_refine_type = options.video_opfl ? 1 : 0;
  video_options.enable_refinemv = options.video_refine_mv;
  video_options.enable_cwp_weighted = options.video_cwp_weighted;
  video_options.enable_imp_msk_bld = options.video_imp_mask;
  video_options.enable_df_sub_pu = options.video_df_sub_pu;
  video_options.b_frames = static_cast<int>(options.b_frames);
  video_options.video_gops = options.video_gops;
  video_options.enable_inter_residual = options.video_inter_residual;
  video_options.enable_edge_inter = options.video_edge_inter;
  video_options.enable_edge_mask = options.video_edge_mask;
  // --video-tile-flush-min (implementation, then docs).
  video_options.minimal_tile_trailer = options.video_tile_flush_min;
  // --video-tile-flush-min block end.
  // --video-drift-refresh (implementation, then docs).
  video_options.drift_refresh = options.video_drift_refresh;
  video_options.drift_refresh_threshold_db =
      options.video_drift_refresh_threshold;
  video_options.drift_refresh_qp = options.video_drift_refresh_qp;
  video_options.drift_refresh_budget = options.video_drift_refresh_budget;
  // --video-drift-refresh block end.
  video_options.enable_leaf_rd = options.video_leaf_rd;
  video_options.enable_sdp = options.video_sdp;
  video_options.enable_inter_ddt = options.video_ddt;
  video_options.enable_tx_search = options.video_tx_search;
  video_options.enable_key_intra_tx_part = options.video_key_intra_txpart;
  video_options.tx_context_rate = options.video_tx_context_rate;
  video_options.tx_rd_lambda = options.video_tx_lambda;
  video_options.key_tx_lambda = options.video_key_tx_lambda;
  video_options.leaf_rd_lambda_ref = options.video_leaf_rd_lambda;
  video_options.qp_lambda_scale = options.video_qp_lambda;
  video_options.leaf_rd_lambda_explicit =
      flag_given(options, "--video-leaf-rd-lambda");
  video_options.tx_rd_lambda_explicit =
      flag_given(options, "--video-tx-lambda");
  video_options.key_tx_lambda_explicit =
      flag_given(options, "--video-key-tx-lambda");
  video_options.rdoq_sse_scale_explicit =
      flag_given(options, "--video-rdoq-scale");
  video_options.partition_lambda_explicit =
      flag_given(options, "--video-partition-rd-lambda");
  video_options.partition_rd = options.video_partition_rd;
  video_options.partition_probe_lambda = options.video_partition_lambda;
  video_options.partition_probe_depth = options.video_partition_depth;
  video_options.enable_deblock = options.video_deblock;
  video_options.enable_segmentation = options.video_segmentation;
  video_options.enable_delta_q = options.video_delta_q || options.video_aq;
  video_options.delta_q_res = options.video_delta_q_res;
  video_options.enable_aq = options.video_aq;
  video_options.aq_strength = options.video_aq_strength;
  video_options.aq_clamp = options.video_aq_clamp;
  video_options.aq_bias = options.video_aq_bias;
  video_options.aq_bias_auto = !options.video_aq_bias_given;
  video_options.per_segment_q = options.video_per_segment_q;
  video_options.seg_q_delta = options.video_seg_delta;
  // enable_uv_delta_q also switches on the segmentation/delta-Q planner, which
  // the symmetric offset does not need: it is a constant sequence field.
  video_options.enable_uv_delta_q = options.video_uv_delta_q;
  video_options.uv_delta_q = options.video_uv_delta_q_value;
  if (!options.video_uv_qp_offset_given && !options.video_uv_delta_q) {
    // --nightly-b sets video_uv_qp_offset without video_uv_qp_offset_given:
    // the preset default (see apply_video_nightly_b).  An explicit
    // --video-uv-qp-offset 0 sets _given and leaves the value 0, which is the
    // opt-out: it must stay a TRUE no-op, because coding the §5.4.8 group with
    // base 0 costs the enable bit and changes the bytes while leaving the step
    // identical.
    if (options.video_uv_qp_offset != 0) {
      video_options.seq_delta_q = av2enc::dcdeltaq::SequenceDeltaQ{};
      video_options.seq_delta_q.equal_ac_dc_q = true;
      video_options.seq_delta_q.separate_uv_delta_q = false;
      video_options.seq_delta_q.uv_ac_delta_q_enabled = true;
      video_options.seq_delta_q.base_uv_ac_delta_q = options.video_uv_qp_offset;
      video_options.frame_delta_q = av2enc::dcdeltaq::FrameDeltaQ{};
      video_options.enable_uv_qp_offset = true;
    }
  } else if (options.video_uv_qp_offset_given && options.video_uv_qp_offset == 0) {
    // Explicit --video-uv-qp-offset 0: the default quantizer, no §5.4.8 group.
  } else if (options.video_uv_qp_offset_given) {
    // §5.4.8 symmetric chroma offset: BOTH planes take the SAME delta, so a
    // negative value codes U and V finer than luma at the same luma QP.
    // equal_ac_dc_q = 1 keeps the group to one base field plus one enable bit,
    // and normalized_sequence_delta_q() then infers
    // BaseUVDcDeltaQ = BaseUVAcDeltaQ, so both DC and AC land on the offset
    // step.  equal_ac_dc_q = 0 would additionally code base_y_dc_delta_q and a
    // per-frame Y-DC field that this path does not need.  The per-plane FRAME
    // deltas stay 0: the sequence base carries the whole offset.
    video_options.seq_delta_q = av2enc::dcdeltaq::SequenceDeltaQ{};
    video_options.seq_delta_q.equal_ac_dc_q = true;
    video_options.seq_delta_q.separate_uv_delta_q = false;
    video_options.seq_delta_q.uv_ac_delta_q_enabled = true;
    video_options.seq_delta_q.base_uv_ac_delta_q = options.video_uv_qp_offset;
    video_options.frame_delta_q = av2enc::dcdeltaq::FrameDeltaQ{};
    // enable_uv_delta_q is deliberately NOT set: it also switches on the
    // per-superblock segmentation/delta-Q planner, which bars every chained and
    // CDF-carrying schedule.  A constant sequence field needs none of that; the
    // frame encoder resolves the per-plane steps when either flag is set.
    video_options.enable_uv_qp_offset = true;
  } else if (options.video_uv_delta_q) {
    // §5.4.8: separate U/V deltas with the DC fields enabled.  equal_ac_dc_q=0
    // is required to code base_uv_dc_delta_q / uv_dc_delta_q_enabled.  U uses
    // +D and V uses -D so diff_uv_delta=1 exercises both chroma planes.
    video_options.seq_delta_q = av2enc::dcdeltaq::SequenceDeltaQ{};
    video_options.seq_delta_q.equal_ac_dc_q = false;
    video_options.seq_delta_q.separate_uv_delta_q = true;
    video_options.seq_delta_q.uv_dc_delta_q_enabled = true;
    video_options.seq_delta_q.uv_ac_delta_q_enabled = true;
    video_options.seq_delta_q.base_uv_dc_delta_q = 0;
    video_options.seq_delta_q.base_uv_ac_delta_q = 0;
    video_options.frame_delta_q = av2enc::dcdeltaq::FrameDeltaQ{};
    video_options.frame_delta_q.delta_q_u_dc = options.video_uv_delta_q_value;
    video_options.frame_delta_q.delta_q_v_dc = -options.video_uv_delta_q_value;
    video_options.frame_delta_q.delta_q_u_ac = options.video_uv_delta_q_value;
    video_options.frame_delta_q.delta_q_v_ac = -options.video_uv_delta_q_value;
  }
  video_options.enable_ext_seg =
      options.video_segments > 8;
  video_options.num_segments =
      options.video_segments == 0 ? 2 : options.video_segments;
  video_options.enable_seg_skip = options.video_seg_skip;
  video_options.enable_seg_globalmv = options.video_seg_globalmv;
  video_options.enable_seg_temporal = options.video_seg_temporal;
  video_options.enable_gme = options.video_gme;
  video_options.enable_warp = options.video_warp;
  video_options.enable_bawp = options.video_bawp;
  video_options.bawp_carrier = options.video_bawp_carrier;
  video_options.enable_morph = options.video_morph;
  video_options.enable_palette = options.video_palette;
  video_options.enable_ibc = options.video_ibc;
  video_options.seq_force_screen_content_tools = options.video_scc_force;
  video_options.seq_force_integer_mv = options.video_force_integer_mv;
  video_options.reduce_pb_aspect_ratio = options.video_reduce_pb_aspect;
  video_options.max_pb_aspect_ratio_log2_minus_1 =
      options.video_pb_aspect_log2_minus_1;
  video_options.seq_seg_info_present = options.video_seq_seg_info;
  video_options.seq_allow_seg_info_change = !options.video_seq_seg_frozen;
  video_options.seq_tile_info_present = options.video_tile_group_compact;
  video_options.enable_interintra = options.video_compound_interintra;
  video_options.enable_ref_frame_mvs = options.video_ref_mvs;
  video_options.ref_mvs_stress = options.video_ref_mvs_stress;
  video_options.short_refresh = options.video_short_refresh;
  video_options.ref_signal_compact = options.video_ref_signal_compact;
  video_options.refmvbank = options.video_refmvbank;
  video_options.mv_traj = options.video_mv_traj;
  video_options.same_ref_compound = options.video_same_ref_compound;
  video_options.enable_tip = options.video_tip;
  video_options.tip_hole_fill = !options.video_tip_no_hole_fill;
  video_options.tip_block_ref = options.video_tip_block_ref;
  video_options.enable_intra_full = options.video_intra_full;
  video_options.enable_cfl = options.video_cfl;
  video_options.enable_mhccp = options.video_mhccp;
  video_options.enable_qm = options.video_qm;
  video_options.user_qm = options.video_user_qm;
  video_options.enable_tcq = options.video_tcq;
  video_options.enable_rdoq = options.video_rdoq;
  video_options.rdoq_sse_scale = options.video_rdoq_scale;
  video_options.enable_cctx = options.video_cctx;
  video_options.film_grain = options.video_grain;
  video_options.enable_idtx_intra = options.video_idtx;
  video_options.enable_mrl = options.video_mrl;
  video_options.enable_ibp = options.video_ibp;
  video_options.enable_dip = options.video_dip;
  video_options.enable_ist = options.video_ist;
  video_options.me_subpel_eighth = options.video_subpel_eighth;
  video_options.mv_precision = options.video_mv_precision;
  video_options.enable_adaptive_mvd = options.video_adaptive_mvd;
  video_options.enable_mvd_sign_derive = options.video_mvd_sign_derive;
  video_options.enable_flex_mvres = options.video_flex_mvres;
  video_options.drl_reorder = options.video_drl_reorder;
  video_options.drl_reorder_always = options.video_drl_reorder_always;
  video_options.long_term = options.video_long_term;
  video_options.olk = options.video_olk;
  video_options.leading_frames = options.video_leading;
  video_options.sef = options.video_sef;
  video_options.ra_switch = options.video_switch;
  video_options.ras = options.video_ras;
  video_options.temporal_layers = options.video_tlayers;
  video_options.tlayer_drop = options.video_tlayer_drop;
  video_options.tlayer_chain = tlayers_chain_enabled(options);
  video_options.tlayer_b = tlayers_b_enabled(options);
  video_options.rd_driver = options.video_rd_driver;
  video_options.eff_perc = options.video_eff_perc;
  video_options.tail_tx = options.video_tail_tx;
  video_options.head_tx = options.video_head_tx;
  video_options.tlayer_qp_active = options.video_tlayer_qp_given;
  video_options.tlayer_qp_offset = options.video_tlayer_qp_offset;
  video_options.tlayer_anchor_residual = options.video_tlayer_anchor_residual;
  video_options.tlayer_refresh_threshold_db =
      options.video_tlayer_refresh_threshold;
  video_options.tlayer_refresh_budget = options.video_tlayer_refresh_budget;
  video_options.force_single_block = options.video_force_single;
  video_options.skip_collapse = options.video_skip_collapse;
  video_options.sb128 = options.video_sb128;
  video_options.sb256 = options.video_sb256;
  video_options.ext_partitions = options.video_ext_partitions;
  video_options.min_block_size = options.video_min_block;
  video_options.max_partition_depth = options.video_max_depth;
  video_options.me_range = options.video_me_range;
  video_options.loop_filters.cdef = options.video_cdef;
  video_options.loop_filters.cdef_num_strengths = options.video_cdef_strength;
  video_options.loop_filters.cdef_damping = options.video_cdef_damping;
  video_options.loop_filters.cdef_rd = options.video_cdef_rd;
  video_options.loop_filters.cdef_rd_lambda = options.video_cdef_rd_lambda;
  video_options.loop_filters.cdef_search_mode =
      static_cast<av2enc::video::post::VideoLoopFilterOptions::CdefSearchMode>(
          options.video_cdef_search);
  video_options.loop_filters.ccso = options.video_ccso;
  video_options.loop_filters.ccso_bands_log2 = options.video_ccso_bands;
  video_options.loop_filters.ccso_rd = options.video_ccso_rd;
  video_options.loop_filters.ccso_lambda = options.video_ccso_lambda;
  video_options.loop_filters.ccso_refit_passes =
      options.video_ccso_refit_passes;
  video_options.loop_filters.ccso_unit_bits = options.video_ccso_unit_bits;
  video_options.loop_filters.ccso_header_rd = options.video_ccso_header_rd;
  video_options.loop_filters.wiener = options.video_wiener;
  video_options.loop_filters.pc_wiener = options.video_pc_wiener;
  video_options.loop_filters.chroma_lr = options.video_chroma_lr;
  video_options.loop_filters.gdf = options.video_gdf;
  video_options.trace = options.video_trace;
  video_options.cpu_threads = options.cpu_threads;
  if (options.tiles != "1x1") {
    const av2enc::tiles::TileLayoutDims dims =
        av2enc::tiles::preset_dims(tile_preset_from_layout(options.tiles));
    video_options.tile_cols = dims.cols;
    video_options.tile_rows = dims.rows;
  }

  // Opt-in per-frame rate control. CQP (default) leaves frame_base_q_idx empty,
  // so encode_video_sequence runs the historical fixed-QP path unchanged.
  // The RC statistics / probe passes must use the same backend as the final
  // encode; --video-gpu never falls back to the CPU encoder.  For CQP the
  // historical single call below is unchanged.
  const auto encode_rc_pass =
      [&error, &options](const std::vector<av2enc::video::VideoInputFrame>& seq,
                         const av2enc::video::VideoOptions& opts,
                         av2enc::video::VideoSequenceResult& out) -> bool {
#if AV2ENC_VIDEO_GPU_CALLABLE
    if (options.video_gpu) {
      return av2enc::cuda::video::encode_video_sequence_gpu(seq, opts, out,
                                                            error);
    }
#else
    (void)options;  // --video-gpu is not callable in this build
#endif
    return av2enc::video::encode_video_sequence(seq, opts, out, error);
  };
  std::vector<std::uint8_t> frame_qp;
  // The rate controller's per-frame plan.  VBR keeps this shape and searches a
  // single global offset; it never replaces the plan with one uniform QP.
  std::vector<std::uint8_t> planned_qp;
  if (options.rc_mode != "cqp") {
    std::vector<std::uint32_t> pass1_payload_bits;
    if (options.rc_mode == "vbr") {
      // Statistics pass at the fixed --qp: payload bits feed the VBR virtual
      // buffer, the same role as a first pass in a conventional encoder.
      av2enc::video::VideoSequenceResult stats;
      if (!encode_rc_pass(inputs, video_options, stats)) {
        std::cerr << "av2enc: error: --rc vbr statistics pass: " << error
                  << "\n";
        return 1;
      }
      pass1_payload_bits.reserve(stats.tile_bytes.size());
      for (const std::uint32_t bytes : stats.tile_bytes) {
        pass1_payload_bits.push_back(bytes * 8U);
      }
    }
    if (!plan_video_rate_control(
            options, y4m, inputs,
            options.rc_mode == "vbr" ? &pass1_payload_bits : nullptr, &frame_qp,
            error)) {
      std::cerr << "av2enc: error: --rc " << options.rc_mode << ": " << error
                << "\n";
      return 1;
    }
    video_options.frame_base_q_idx = frame_qp;
    video_options.base_q_idx = frame_qp.front();
    planned_qp = frame_qp;
  } else if (options.video_key_qp_offset_given) {
    if (!plan_video_frame_qp(options, inputs, &frame_qp, error)) {
      std::cerr << "av2enc: error: --video-key-qp-offset: " << error << "\n";
      return 1;
    }
    video_options.frame_base_q_idx = frame_qp;
    video_options.base_q_idx = frame_qp.front();
  }
  // AV2ENC_FRAME_QP_PLAN=<head>:<tail> adds a per-frame QP offset to the head
  // (frames 1-32) and the tail (frames 33-59) on the fixed-QP path.  This is
  // the head/tail allocation probe: a per-frame base_q_idx is legal with the
  // carried CDF image (--cdf-continuity, which --fast-decode enables), unlike
  // delta-Q/segmentation.  Unset leaves every pin byte-identical.
  if (options.rc_mode == "cqp" && !options.video_gpu) {
    if (const char* env_plan = std::getenv("AV2ENC_FRAME_QP_PLAN")) {
      char* end = nullptr;
      const long head_off = std::strtol(env_plan, &end, 10);
      long tail_off = 0;
      bool parsed = end != env_plan && end != nullptr && *end == ':';
      if (parsed) {
        const char* tail = end + 1;
        char* tail_end = nullptr;
        tail_off = std::strtol(tail, &tail_end, 10);
        parsed = tail_end != tail;
      }
      if (parsed) {
        if (frame_qp.empty()) {
          frame_qp.assign(inputs.size(),
                          static_cast<std::uint8_t>(video_options.base_q_idx));
        }
        const std::size_t head_end = std::min<std::size_t>(33, frame_qp.size());
        for (std::size_t i = 1; i < frame_qp.size(); ++i) {
          const long off = i < head_end ? head_off : tail_off;
          frame_qp[i] = static_cast<std::uint8_t>(std::clamp<long>(
              static_cast<long>(frame_qp[i]) + off, 1, 255));
        }
        video_options.frame_base_q_idx = frame_qp;
        video_options.base_q_idx = frame_qp.front();
      } else {
        std::cerr << "av2enc: error: AV2ENC_FRAME_QP_PLAN expects <head>:<tail>\n";
        return 1;
      }
    }
  }

  av2enc::video::VideoSequenceResult result;
  int vbr_iterations = 0;
  double vbr_target_kbps = 0.0;
  if (options.rc_mode == "vbr") {
    // Closed-loop two-pass VBR.  The statistics pass calibrates the controller
    // and the controller plans a base_q_idx per frame.  Every probe is that
    // plan shifted by one global offset (the frame-to-frame shape is kept, so
    // flat and busy frames do not collapse onto one QP); the measured payload
    // bits drive a bisection over the offset.  The measured payload is in the
    // same units as the target, so the search is deterministic.
    const double duration =
        static_cast<double>(inputs.size()) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    vbr_target_kbps = options.rc_bitrate_kbps;
    const double target_bits = vbr_target_kbps * 1000.0 * duration;
    const auto apply_offset = [&](int offset) {
      frame_qp.resize(planned_qp.size());
      for (std::size_t i = 0; i < planned_qp.size(); ++i) {
        frame_qp[i] = static_cast<std::uint8_t>(
            std::clamp(static_cast<int>(planned_qp[i]) + offset, 1, 255));
      }
      video_options.frame_base_q_idx = frame_qp;
      video_options.base_q_idx = frame_qp.front();
    };
    int low = -254;
    int high = 254;
    int offset = 0;
    double best_error = std::numeric_limits<double>::infinity();
    av2enc::video::VideoSequenceResult best_result;
    std::vector<std::uint8_t> best_qp = planned_qp;
    while (low <= high && vbr_iterations < 12) {
      ++vbr_iterations;
      apply_offset(offset);
      av2enc::video::VideoSequenceResult candidate;
      if (!encode_rc_pass(inputs, video_options, candidate)) {
        std::cerr << "av2enc: error: --rc vbr pass " << vbr_iterations << ": "
                  << error << "\n";
        return 1;
      }
      std::uint64_t measured = 0;
      for (const std::uint32_t bytes : candidate.tile_bytes) {
        measured += static_cast<std::uint64_t>(bytes) * 8U;
      }
      const double error_ratio =
          target_bits > 0.0
              ? std::fabs(static_cast<double>(measured) - target_bits) /
                    target_bits
              : 0.0;
      if (error_ratio < best_error) {
        best_error = error_ratio;
        best_result = candidate;
        best_qp = frame_qp;
      }
      if (error_ratio <= 0.03) break;
      if (static_cast<double>(measured) > target_bits) {
        low = offset + 1;  // too many bits: raise every frame's base_q_idx
      } else {
        high = offset - 1;
      }
      offset = low + (high - low) / 2;
    }
    result = std::move(best_result);
    frame_qp = best_qp;
    video_options.frame_base_q_idx = frame_qp;
    video_options.base_q_idx = frame_qp.front();
  } else if (options.video_gpu) {
#if AV2ENC_VIDEO_GPU_CALLABLE
    av2enc::cuda::video::VideoGpuStats gpu_stats;
    if (!av2enc::cuda::video::encode_video_sequence_gpu(
            inputs, video_options, result, error, &gpu_stats)) {
      std::cerr << "av2enc: error: --video-gpu: " << error << "\n";
      return 1;
    }
#if AV2ENC_HAVE_CUDA
    if (options.video_trace) {
      std::cerr << "av2enc: video GPU stats: launches="
                << gpu_stats.total_launches() << " syncs=" << gpu_stats.syncs
                << " uploaded=" << gpu_stats.bytes_uploaded
                << " downloaded=" << gpu_stats.bytes_downloaded
                << " device_ms=" << gpu_stats.device_ms << "\n";
    }
#endif
#else
    std::cerr << "av2enc: error: --video-gpu is not linked in this build\n";
    return 1;
#endif
  } else if (!av2enc::video::encode_video_sequence(inputs, video_options,
                                                   result, error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }
  if (options.video_gpu) {
#if AV2ENC_VIDEO_GPU_CALLABLE
    // Honest provenance: neither the CQP dispatch nor an --rc crf/vbr probe
    // re-enters the CPU encoder, so frame_bytes_from_cpu stays 0.  Name the
    // linked backend explicitly for every --video-gpu mode.
    std::cerr << "av2enc: --video-gpu backend="
              << av2enc::cuda::video::video_backend_name()
              << " frame_bytes_from_cpu=0\n";
#endif
  }
  for (std::size_t i = 0; i < result.units.size(); ++i) {
    if (options.video_brt) {
      std::vector<std::uint8_t> brt;
      if (!make_brt_obu(options.brt_time, brt, error) ||
          !insert_before_first_frame_obu(result.units[i], brt, error)) {
        std::cerr << "av2enc: error: " << error << "\n";
        return 1;
      }
    }
    // Neutral metadata is placed once, as suffix units of the first temporal
    // unit.  It is decoder-invariant (§5.17.2 preserves coded output order).
    if (i == 0 &&
        !append_neutral_metadata_obus(result.units[i],
                                      metadata_cli_options(options), nullptr,
                                      options.metadata_timecode,
                                      options.metadata_scan_type, error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
    if (!writer.write_frame(result.units[i],
                            static_cast<std::uint64_t>(i), error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!options.video_recon.empty()) {
    if (!write_video_recon_y4m(options.video_recon, y4m, result.recon, error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  std::uint64_t total_tile = 0;
  std::uint64_t total_inter = 0;
  std::uint64_t total_intra = 0;
  for (std::size_t i = 0; i < result.tile_bytes.size(); ++i) {
    total_tile += result.tile_bytes[i];
    total_inter += result.inter_blocks[i];
    total_intra += result.intra_blocks[i];
  }
  std::cout << "av2enc: Video P-frame encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.colorspace << ", "
            << y4m.fps_num << "/" << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  " << result.units.size() << " temporal unit(s), "
            << result.tile_bytes.size() << " coded frame(s), "
            << result.recon.size() << " output frame(s), base_q_idx ";
  if (frame_qp.empty()) {
    std::cout << static_cast<int>(video_options.base_q_idx);
  } else {
    std::cout << "per-frame";
  }
  std::cout << ", tile " << total_tile << " bytes, inter blocks " << total_inter
            << ", intra blocks " << total_intra;
  if (result.temporal_candidates != 0) {
    std::cout << ", temporal MV candidates " << result.temporal_candidates;
  }
  if (result.temporal_selected != 0) {
    std::cout << ", temporal-selected leaves " << result.temporal_selected;
  }
  if (result.same_ref_blocks != 0) {
    std::cout << ", same-ref compound blocks " << result.same_ref_blocks;
  }
  if (result.refmvbank_candidates != 0) {
    std::cout << ", refmvbank candidates " << result.refmvbank_candidates;
  }
  if (result.refmvbank_selected != 0) {
    std::cout << ", refmvbank-selected leaves " << result.refmvbank_selected;
  }
  if (result.mv_traj_derived != 0) {
    std::cout << ", mv-traj derived candidates " << result.mv_traj_derived;
  }
  if (result.short_refresh_frames != 0) {
    std::cout << ", short refresh frames " << result.short_refresh_frames;
  }
  // --video-drift-refresh evidence (implementation, then docs).
  if (options.video_drift_refresh) {
    std::cout << ", drift refresh frames " << result.drift_refresh_frames
              << " (" << result.drift_refresh_bytes << " extra payload bytes)";
  }
  if (options.video_tlayer_anchor_residual) {
    std::cout << ", anchor refresh " << result.tlayer_anchor_refresh << "/"
              << result.tlayer_anchor_total << " (" << result.drift_refresh_bytes
              << " extra payload bytes)";
  }
  // --video-drift-refresh block end.
  if (result.amvd_selected != 0) {
    std::cout << ", AMVD leaves " << result.amvd_selected;
  }
  if (result.flex_precision_selected != 0) {
    std::cout << ", flex-precision leaves " << result.flex_precision_selected;
  }
  if (result.derived_sign_selected != 0) {
    std::cout << ", derived-sign leaves " << result.derived_sign_selected;
  }
  if (result.ref_mvs_warp_cells != 0) {
    std::cout << ", §7.22 warp 8x8 cells " << result.ref_mvs_warp_cells;
  }
  if (result.ref_mvs_wedge_dropped != 0) {
    std::cout << ", §7.22 wedge allow-list cells "
              << result.ref_mvs_wedge_dropped;
  }
  std::cout << "\n";
  if (!frame_qp.empty()) {
    // The achieved payload is reported over the displayed duration, which is
    // the same denominator the VBR target used.  --video-sef appends a
    // show-existing-frame unit, so result.units.size() over-counts the coded
    // frames; result.recon.size() is the displayed picture count.
    const std::size_t displayed_frames =
        result.recon.empty() ? result.units.size() : result.recon.size();
    const double duration =
        static_cast<double>(displayed_frames) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    const double kbps = duration > 0.0
                            ? static_cast<double>(total_tile) * 8.0 / duration /
                                  1000.0
                            : 0.0;
    std::cout << "  rc:      " << options.rc_mode;
    if (options.rc_mode == "crf") {
      std::cout << " " << options.rc_crf;
    } else if (options.rc_mode == "vbr") {
      const double error_pct =
          vbr_target_kbps > 0.0
              ? (kbps - vbr_target_kbps) / vbr_target_kbps * 100.0
              : 0.0;
      std::cout << " target " << vbr_target_kbps << " kbps (closed-loop "
                << vbr_iterations << " pass(es))";
      std::cout << ", achieved " << kbps << " kbps ("
                << (error_pct >= 0.0 ? "+" : "") << error_pct
                << "% error, payload)\n";
    }
    if (options.rc_mode == "crf") {
      std::cout << ", achieved " << kbps << " kbps (payload)\n";
    }
    std::cout << "  frame QP:";
    for (const std::uint8_t qp : frame_qp) {
      std::cout << " " << static_cast<int>(qp);
    }
    std::cout << "\n  frame bits:";
    for (const std::uint32_t bytes : result.tile_bytes) {
      std::cout << " " << (static_cast<std::uint64_t>(bytes) * 8U);
    }
    std::cout << "\n";
  }
  return 0;
}

// "x0=WxH,x1=WxH" -> SpatialLayerSpec list.  Ids must start at 0 and be
// contiguous ascending; build_layer_plan() re-validates the same rule.
bool parse_svc_layer_specs(
    std::string_view text,
    std::vector<av2enc::svc::SpatialLayerSpec>& layers, std::string& error) {
  layers.clear();
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::string_view item =
        text.substr(start, comma == std::string_view::npos
                               ? std::string_view::npos
                               : comma - start);
    if (item.empty()) {
      error = "--svc-layers has an empty layer entry";
      return false;
    }
    const std::size_t eq = item.find('=');
    if (eq == std::string_view::npos) {
      error = "--svc-layers entries must be xID=WxH";
      return false;
    }
    std::string_view id_text = item.substr(0, eq);
    if (!id_text.empty() && (id_text.front() == 'x' || id_text.front() == 'X')) {
      id_text.remove_prefix(1);
    }
    int id = 0;
    auto id_result = std::from_chars(id_text.data(),
                                     id_text.data() + id_text.size(), id);
    if (id_result.ec != std::errc{} ||
        id_result.ptr != id_text.data() + id_text.size() || id < 0 || id > 30) {
      error = "--svc-layers xlayer id must be an integer in [0, 30]";
      return false;
    }
    const std::size_t xpos = item.find('x', eq + 1);
    if (xpos == std::string_view::npos) {
      error = "--svc-layers entries must be xID=WxH";
      return false;
    }
    int width = 0;
    int height = 0;
    auto w_result = std::from_chars(item.data() + eq + 1, item.data() + xpos,
                                   width);
    auto h_result = std::from_chars(item.data() + xpos + 1,
                                    item.data() + item.size(), height);
    if (w_result.ec != std::errc{} ||
        w_result.ptr != item.data() + xpos || h_result.ec != std::errc{} ||
        h_result.ptr != item.data() + item.size() || width <= 0 || height <= 0) {
      error = "--svc-layers WxH must be positive integers";
      return false;
    }
    av2enc::svc::SpatialLayerSpec spec;
    spec.xlayer_id = static_cast<std::uint8_t>(id);
    spec.width = width;
    spec.height = height;
    layers.push_back(spec);
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return true;
}

int run_svc_encode(const Options& options, av2enc::Y4mReader& reader,
                   const av2enc::Y4mHeader& y4m) {
  std::string error;
  if (options.metadata_timecode || options.metadata_scan_type ||
      options.metadata_cll || options.metadata_mdcv || options.metadata_icc ||
      options.metadata_temporal_point || options.metadata_frame_hash ||
      options.metadata_banding || options.metadata_user_data ||
      options.video_brt) {
    std::cerr << "av2enc: error: --metadata-* is not wired into the --svc-layers "
                 "layered writer\n";
    return 2;
  }
  av2enc::svc::SvcEncodeOptions svc;
  if (!parse_svc_layer_specs(options.svc_layers, svc.config.spatial_layers,
                             error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 2;
  }
  svc.config.temporal_layers = static_cast<std::uint8_t>(options.svc_temporal);
  svc.config.embedded_layers = static_cast<std::uint8_t>(options.svc_embedded);
  svc.config.single_picture_header = false;
  svc.config.still_picture = false;
  svc.config.seq_lcr_id = static_cast<std::uint8_t>(options.svc_lcr_id);
  svc.all_layers = !(options.svc_oppoint >= 0 ||
                     options.svc_spatial_layers >= 1);
  svc.oppoint = options.svc_oppoint;
  svc.spatial_layers = options.svc_spatial_layers;
  svc.video.base_q_idx = static_cast<std::uint8_t>(options.qp);
  svc.video.num_total_refs = 1;
  svc.video.enable_inter_residual = options.video_inter_residual;
  svc.video.me_subpel_eighth = options.video_subpel_eighth;
  svc.video.mv_precision = options.video_mv_precision;
  svc.video.force_single_block = options.video_force_single;
  svc.video.min_block_size = options.video_min_block;
  svc.video.max_partition_depth = options.video_max_depth;
  svc.video.me_range = options.video_me_range;
  svc.video.trace = options.video_trace;

  std::vector<OwnedVideoFrame> owned;
  std::size_t limit = 0;
  if (options.max_frames != 0) {
    limit = static_cast<std::size_t>(options.max_frames);
  } else if (options.p_frames != 0) {
    limit = static_cast<std::size_t>(options.p_frames) + 1U;
  }
  av2enc::Y4mFrame frame;
  while (limit == 0 || owned.size() < limit) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) {
      std::cerr << "av2enc: " << error << "\n";
      return 1;
    }
    OwnedVideoFrame copy;
    copy.width = frame.width();
    copy.height = frame.height();
    copy.y.assign(frame.y().begin(), frame.y().end());
    copy.u.assign(frame.u().begin(), frame.u().end());
    copy.v.assign(frame.v().begin(), frame.v().end());
    owned.push_back(std::move(copy));
  }
  if (owned.empty()) {
    std::cerr << "av2enc: error: input has no frames\n";
    return 1;
  }

  std::vector<av2enc::video::VideoInputFrame> inputs;
  inputs.reserve(owned.size());
  for (const OwnedVideoFrame& f : owned) {
    av2enc::video::VideoInputFrame input;
    input.width = f.width;
    input.height = f.height;
    input.y = std::span<const std::uint8_t>(f.y.data(), f.y.size());
    input.u = std::span<const std::uint8_t>(f.u.data(), f.u.size());
    input.v = std::span<const std::uint8_t>(f.v.data(), f.v.size());
    inputs.push_back(input);
  }

  // Per-frame rate control for the layered writer.  The plan is indexed by
  // display order; svc_stream shifts it by the embedded-layer quality step so
  // every mlayer keeps its own quality rung.  VBR reuses the same two-pass
  // calibration + global-offset bisection as the flat --video path, measuring
  // the summed tile payload of every emitted temporal unit.
  std::vector<std::uint8_t> frame_qp;
  std::vector<std::uint8_t> planned_qp;
  int svc_vbr_iterations = 0;
  double svc_vbr_target_kbps = 0.0;
  if (options.rc_mode != "cqp") {
    std::vector<std::uint32_t> pass1_payload_bits;
    if (options.rc_mode == "vbr") {
      av2enc::svc::SvcEncodeOptions stats_options = svc;
      stats_options.video.base_q_idx = static_cast<std::uint8_t>(options.qp);
      av2enc::svc::SvcStreamResult stats;
      if (!av2enc::svc::encode_svc_sequence(inputs, stats_options, stats,
                                            error)) {
        std::cerr << "av2enc: error: --rc vbr statistics pass: " << error
                  << "\n";
        return 1;
      }
      pass1_payload_bits.reserve(stats.unit_payload_bytes.size());
      for (const std::uint32_t bytes : stats.unit_payload_bytes) {
        pass1_payload_bits.push_back(bytes * 8U);
      }
    }
    if (!plan_video_rate_control(
            options, y4m, inputs,
            options.rc_mode == "vbr" ? &pass1_payload_bits : nullptr, &frame_qp,
            error)) {
      std::cerr << "av2enc: error: --rc " << options.rc_mode << ": " << error
                << "\n";
      return 1;
    }
    planned_qp = frame_qp;
    svc.video.frame_base_q_idx = frame_qp;
    svc.video.base_q_idx = frame_qp.front();
  }

  av2enc::svc::SvcStreamResult result;
  if (options.rc_mode == "vbr") {
    const double duration =
        static_cast<double>(inputs.size()) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    svc_vbr_target_kbps = options.rc_bitrate_kbps;
    const double target_bits = svc_vbr_target_kbps * 1000.0 * duration;
    const auto apply_offset = [&](int offset) {
      frame_qp.resize(planned_qp.size());
      for (std::size_t i = 0; i < planned_qp.size(); ++i) {
        frame_qp[i] = static_cast<std::uint8_t>(
            std::clamp(static_cast<int>(planned_qp[i]) + offset, 1, 255));
      }
      svc.video.frame_base_q_idx = frame_qp;
      svc.video.base_q_idx = frame_qp.front();
    };
    int low = -254;
    int high = 254;
    int offset = 0;
    double best_error = std::numeric_limits<double>::infinity();
    av2enc::svc::SvcStreamResult best_result;
    std::vector<std::uint8_t> best_qp = planned_qp;
    while (low <= high && svc_vbr_iterations < 12) {
      ++svc_vbr_iterations;
      apply_offset(offset);
      av2enc::svc::SvcStreamResult candidate;
      if (!av2enc::svc::encode_svc_sequence(inputs, svc, candidate, error)) {
        std::cerr << "av2enc: error: --rc vbr pass " << svc_vbr_iterations
                  << ": " << error << "\n";
        return 1;
      }
      std::uint64_t measured = 0;
      for (const std::uint32_t bytes : candidate.unit_payload_bytes) {
        measured += static_cast<std::uint64_t>(bytes) * 8U;
      }
      const double error_ratio =
          target_bits > 0.0
              ? std::fabs(static_cast<double>(measured) - target_bits) /
                    target_bits
              : 0.0;
      if (error_ratio < best_error) {
        best_error = error_ratio;
        best_result = candidate;
        best_qp = frame_qp;
      }
      if (error_ratio <= 0.03) break;
      if (static_cast<double>(measured) > target_bits) {
        low = offset + 1;
      } else {
        high = offset - 1;
      }
      offset = low + (high - low) / 2;
    }
    result = std::move(best_result);
    frame_qp = best_qp;
    svc.video.frame_base_q_idx = frame_qp;
    svc.video.base_q_idx = frame_qp.front();
  } else if (!av2enc::svc::encode_svc_sequence(inputs, svc, result, error)) {
    std::cerr << "av2enc: error: --svc-layers: " << error << "\n";
    return 1;
  }

  // The IVF header stays the base-layer size (Annex F.3.2 extraction keeps the
  // container header); timestamps are temporal-unit indices.
  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(svc.config.spatial_layers.front().width);
  ivf.height =
      static_cast<std::uint16_t>(svc.config.spatial_layers.front().height);
  ivf.fps_num = y4m.fps_num;
  ivf.fps_den = y4m.fps_den;
  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }
  for (std::size_t i = 0; i < result.units.size(); ++i) {
    if (!writer.write_frame(result.units[i], static_cast<std::uint64_t>(i),
                            error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "av2enc: error: " << error << "\n";
    return 1;
  }

  // Per-layer reconstruction dump.  A single xlayer/m0 stream writes exactly
  // --video-recon; a layered stream writes <stem>_x<X>m<M><ext> next to it so
  // the decoder gate can compare every layer/mlayer sample-exactly.
  if (!options.video_recon.empty()) {
    const bool single = result.layers.size() == 1 &&
                        result.layers.front().mlayer_recon.size() == 1;
    for (const av2enc::svc::SvcLayerOutput& layer : result.layers) {
      for (std::size_t m = 0; m < layer.mlayer_recon.size(); ++m) {
        std::filesystem::path path = options.video_recon;
        if (!single) {
          std::ostringstream suffix;
          suffix << "_x" << layer.xlayer_id << "m" << m;
          path = options.video_recon.parent_path() /
                 (options.video_recon.stem().string() + suffix.str() +
                  options.video_recon.extension().string());
        }
        av2enc::Y4mHeader recon_header = y4m;
        recon_header.width = static_cast<std::uint32_t>(layer.width);
        recon_header.height = static_cast<std::uint32_t>(layer.height);
        const std::vector<av2enc::video::VideoReconFrame>& recon =
            layer.mlayer_recon[m].empty() && m + 1 == layer.mlayer_recon.size()
                ? layer.recon
                : layer.mlayer_recon[m];
        if (!write_video_recon_y4m(path, recon_header, recon, error)) {
          std::cerr << "av2enc: error: " << error << "\n";
          return 1;
        }
      }
    }
  }

  std::uint64_t total_payload = 0;
  for (const std::uint32_t bytes : result.unit_payload_bytes) {
    total_payload += bytes;
  }
  std::cout << "av2enc: SVC layered encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.fps_num << "/" << y4m.fps_den
            << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF, base "
            << ivf.width << "x" << ivf.height << ")\n"
            << "  layers:  " << result.layers.size() << "\n";
  for (const av2enc::svc::SvcLayerOutput& layer : result.layers) {
    std::cout << "    x" << layer.xlayer_id << ": " << layer.width << "x"
              << layer.height << ", " << layer.recon.size() << " frame(s)\n";
  }
  std::cout << "  units:   " << result.units.size() << " temporal unit(s), "
            << result.selected_layers << " selected layer(s), "
            << result.selection_label << "\n"
            << "  payload: " << total_payload << " bytes\n";
  if (!frame_qp.empty()) {
    const double duration =
        static_cast<double>(inputs.size()) *
        static_cast<double>(y4m.fps_den) / static_cast<double>(y4m.fps_num);
    const double kbps = duration > 0.0
                            ? static_cast<double>(total_payload) * 8.0 /
                                  duration / 1000.0
                            : 0.0;
    std::cout << "  rc:      " << options.rc_mode;
    if (options.rc_mode == "crf") {
      std::cout << " " << options.rc_crf << ", achieved " << kbps
                << " kbps (payload)\n";
    } else if (options.rc_mode == "vbr") {
      const double error_pct =
          svc_vbr_target_kbps > 0.0
              ? (kbps - svc_vbr_target_kbps) / svc_vbr_target_kbps * 100.0
              : 0.0;
      std::cout << " target " << svc_vbr_target_kbps << " kbps (closed-loop "
                << svc_vbr_iterations << " pass(es)), achieved " << kbps
                << " kbps (" << (error_pct >= 0.0 ? "+" : "") << error_pct
                << "% error, payload)\n";
    }
    std::cout << "  frame QP:";
    for (const std::uint8_t qp : frame_qp) {
      std::cout << " " << static_cast<int>(qp);
    }
    std::cout << "\n";
  }
  return 0;
}

}  // namespace

static int av2enc_main(int argc, char** argv) {
  Options options;
  std::string error;
  ParseStatus status = parse_arguments(argc, argv, options, error);
  if (status == ParseStatus::kOk) {
    status = validate_options(options, error);
  }
  if (status == ParseStatus::kHelp) {
    print_usage(std::cout);
    return 0;
  }
  if (status == ParseStatus::kError) {
    std::cerr << "av2enc: " << error << "\n\n";
    print_usage(std::cerr);
    return 2;
  }

  // The effort report is printed for every --effort/--cpu-used run (the
  // contract asks for the selected flags and the used budget), and
  // --effort-plan stops before the input or output is touched.
  if (options.effort_active) {
    std::cerr << options.effort_report;
  }
  if (options.effort_plan) {
    return 0;
  }

  if (options.vram_cap_mib != 0) {
#if AV2ENC_HAVE_CUDA
    if (!apply_vram_cap(options.vram_cap_mib, error)) {
      std::cerr << "av2enc: " << error << "\n";
      return 2;
    }
#else
    std::cerr << "av2enc: note: --vram-cap has no effect in this "
                 "CPU-only build\n";
#endif
  }

  if (options.gpu_plan != 0) {
    return run_gpu_plan(options);
  }

  if (options.still_ibc && !options.m1 && !options.still_gpu &&
      !options.lossless) {
    std::cerr << "av2enc: error: --still-ibc is an M1 still-picture tool; "
                 "combine it with --still or --still-gpu\n";
    return 2;
  }

  if (options.still_palette && !options.m1 && !options.still_gpu &&
      !options.lossless) {
    std::cerr << "av2enc: error: --still-palette is an M1 still-picture tool; "
                 "combine it with --still or --still-gpu\n";
    return 2;
  }

  if (options.still_ibc && (options.still_sb128 || options.still_sb256)) {
    std::cerr << "av2enc: error: --still-ibc requires 64x64 superblocks "
                 "(drop --still-sb128/--still-sb256)\n";
    return 2;
  }

  if (options.still_ibc_search_step != 8 && !options.still_ibc) {
    std::cerr << "av2enc: error: --still-ibc-search requires --still-ibc\n";
    return 2;
  }

#if !AV2ENC_STILL_GPU_CALLABLE
  // Accept the flag in every build but fail before touching the input/output
  // so the caller gets a precise reason instead of a link-time or decode-time
  // surprise.
  if (options.still_gpu) {
#if AV2ENC_HAVE_CUDA
    std::cerr << "av2enc: error: --still-gpu is not available in this build: "
                 "the Still GPU entry point (src/cuda/pipeline/still/still_gpu.cu) has not "
                 "landed yet. Use --still for the CPU Still path.\n";
#else
    std::cerr << "av2enc: error: --still-gpu requires a CUDA build; this is a "
                 "CPU-only reference build. Use --still for the CPU Still path.\n";
#endif
    return 2;
  }
#endif

  if (options.still_seg_delta != 8 && !options.still_per_segment_q) {
    std::cerr << "av2enc: error: --still-seg-delta requires "
                 "--still-per-segment-q\n";
    return 2;
  }
  if (options.still_per_segment_q) {
    if (!options.m1 && !options.still_gpu) {
      std::cerr << "av2enc: error: --still-per-segment-q requires --still\n";
      return 2;
    }
    if (options.still_segmentation || options.still_delta_q) {
      std::cerr << "av2enc: error: --still-per-segment-q replaces the neutral "
                   "--still-segmentation/--still-delta-q table; use one form\n";
      return 2;
    }
  }

  if (options.lossless &&
      (options.still_content_interpretation || options.still_segmentation ||
       options.still_delta_q || options.still_per_segment_q ||
       options.still_palette || options.still_ibc)) {
    std::cerr << "av2enc: error: --lossless uses its own header writer; "
                 "--still-content-interpretation, the delta-Q/segmentation "
                 "flags and the screen-content tools are not applied there\n";
    return 2;
  }

  if (options.still_gpu) {
    const auto reject_cpu_only = [](const char* flag) {
      std::cerr << "av2enc: error: --still-gpu: " << flag
                << " is not ported to the GPU pipeline yet; use --still (CPU)\n";
      return 2;
    };
    const bool still_gpu_format =
        options.bit_depth != 8 || options.chroma != "420";
    if (still_gpu_format && (options.still_tcq || options.still_rdoq)) {
      // The 10-bit / non-4:2:0 product writer runs the `hbd` GPU block stage,
      // which has no §7.16 decision stage; reject instead of dropping the flag.
      std::cerr << "av2enc: error: --still-gpu with --bit-depth 10 / "
                   "--chroma 422|444|400 does not support --still-tcq or "
                   "--still-rdoq yet; use --still (CPU)\n";
      return 2;
    }
    if (options.still_user_qm) return reject_cpu_only("--still-user-qm");
    // --still-tcq now has a GPU stage (device §7.16 rate model + §5.4.8
    // trellis); gpu_m3_options_supported() rejects only the combinations the
    // CPU reference itself refuses.
    if (options.still_tx_partition) {
      return reject_cpu_only("--still-tx-partition");
    }
    if (options.still_ddt) return reject_cpu_only("--still-ddt");
    // CDEF, CCSO, loop restoration (Wiener NS, chroma LR, PC-Wiener, GDF),
    // film-grain signalling (including the adaptive model), content
    // interpretation, TCQ and RDOQ now have GPU stages.  The remaining tools
    // are rejected by name.
    if (options.still_rdo_intra) return reject_cpu_only("--still-rdo-intra");
  }

  // Multi-tile stills (--tiles).  The tiled writer is a distinct composition
  // root: it reuses the Still reference per tile but pins the tool set to the
  // verified form, so every option that would change the tile payload is
  // rejected by name instead of being ignored.
  const bool multi_tile = options.tiles != "1x1";
  if (options.cdf_continuity && !options.video) {
    std::cerr << "av2enc: error: --cdf-continuity is wired for --video only\n";
    return 2;
  }
  if (options.video_timing && !options.video) {
    std::cerr << "av2enc: error: --video-timing needs --video: a "
                 "single-picture still header cannot carry the timing block\n";
    return 2;
  }
  if (options.video_mfh && !options.video) {
    std::cerr << "av2enc: error: --video-mfh needs --video: the pinned "
                 "decoder rejects MFH on the single-picture still path "
                 "(docs/negative-results.md)\n";
    return 2;
  }
  if ((options.video_tier_high || options.video_multi_sequence ||
       options.video_lcr_full) &&
      !options.video) {
    std::cerr << "av2enc: error: --video-tier/--video-multi-sequence/"
                 "--video-lcr-full are wired into the non-single-picture "
                 "--video sequence header only\n";
    return 2;
  }
  if (options.video_multi_sequence &&
      (options.video_mfh || options.video_timing)) {
    std::cerr << "av2enc: error: --video-multi-sequence needs the direct "
                 "seq_header_id_in_frame_header field; --video-mfh and "
                 "--video-timing take the single-header form\n";
    return 2;
  }
  if ((options.video_multi_sequence || options.video_lcr_full) &&
      (options.b_frames > 0 || options.video_tlayers > 0 || options.video_tip ||
       options.video_long_term || options.video_olk ||
       options.video_leading > 0 || options.video_sef || options.video_switch ||
       options.video_ras || options.tiles != "1x1" ||
       options.video_lossless || options.bit_depth != 8 ||
       options.chroma != "420")) {
    std::cerr << "av2enc: error: --video-multi-sequence/--video-lcr-full are "
                 "wired into the plain 8-bit 4:2:0 single-tile --video chain; "
                 "the B/temporal/TIP/random-access schedules and the format "
                 "video core write their own headers\n";
    return 2;
  }
  if (options.video_tier_high &&
      (options.b_frames > 0 || options.video_tlayers > 0 || options.video_tip ||
       options.video_long_term || options.video_olk ||
       options.video_leading > 0 || options.video_sef || options.video_switch ||
       options.video_ras || options.tiles != "1x1" ||
       options.video_lossless || options.bit_depth != 8 ||
       options.chroma != "420")) {
    std::cerr << "av2enc: error: --video-tier high is wired into the plain "
                 "8-bit 4:2:0 --video sequence header only; the B/temporal/"
                 "TIP/random-access schedules, --tiles and the format video "
                 "core write their own sequence headers\n";
    return 2;
  }
  if (options.avg_cdf) {
    if (options.lossless) {
      std::cerr << "av2enc: error: --avg-cdf is not implemented by the "
                   "--lossless writer\n";
      return 2;
    }
    if (!options.video && !multi_tile) {
      std::cerr << "av2enc: error: --avg-cdf needs a multi-tile still "
                   "(--tiles > 1) or --video; a single-tile still already "
                   "infers enable_avg_cdf = 1 and has no frame end to average\n";
      return 2;
    }
  }
  if (multi_tile) {
    if (options.video) {
      // Multi-tile video: the per-frame splitter re-emits the whole-frame
      // block list through post::assemble_video_tiled_frame().  Layouts and
      // header fields are pinned to what that writer signals.
      if (options.tiles == "16x2") {
        std::cerr << "av2enc: error: --tiles 16x2 is not gate-verified for "
                     "--video; use 2x1/2x2/4x2/8x2\n";
        return 2;
      }
      if (options.video_refs != 1) {
        std::cerr << "av2enc: error: multi-tile --video requires --video-refs "
                     "1 (the tiled frame header signals one reference slot)\n";
        return 2;
      }
      if (options.video_mv_precision != 0) {
        std::cerr << "av2enc: error: multi-tile --video requires eighth-pel MV "
                     "precision (the default)\n";
        return 2;
      }
      if (options.video_long_term || options.video_olk ||
          options.video_leading != 0 || options.video_sef ||
          options.video_switch || options.video_ras ||
          options.video_drl_reorder || options.video_drl_reorder_always) {
        std::cerr << "av2enc: error: multi-tile --video cannot be combined with "
                     "a frame-type mode or DRL reordering\n";
        return 2;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        std::cerr << "av2enc: error: multi-tile --video is 8-bit 4:2:0 only\n";
        return 2;
      }
    } else {
      if (options.still_gpu) {
        if (options.qp != 90) {
          std::cerr << "av2enc: error: multi-tile --still-gpu pins base_q_idx "
                       "90 (the device quantiser step); use --qp 90 or --still "
                       "(CPU)\n";
          return 2;
        }
        if (options.metadata_timecode || options.metadata_scan_type ||
            options.metadata_cll || options.metadata_mdcv ||
            options.metadata_icc || options.metadata_temporal_point ||
            options.metadata_frame_hash || options.metadata_banding ||
            options.metadata_user_data) {
          std::cerr << "av2enc: error: multi-tile --still-gpu does not apply "
                       "the neutral metadata OBUs\n";
          return 2;
        }
      }
      if (!options.m1 && !options.still_gpu) {
        std::cerr << "av2enc: error: --tiles > 1 requires --still or --video\n";
        return 2;
      }
      if (options.lossless) {
        std::cerr << "av2enc: error: --tiles is not available in the --lossless "
                     "writer\n";
        return 2;
      }
      if (options.bit_depth != 8 || options.chroma != "420") {
        std::cerr << "av2enc: error: the tiled still writer is 8-bit 4:2:0 only; "
                     "--tiles cannot be combined with --bit-depth 10 or --chroma\n";
        return 2;
      }
      if (options.adaptive_cdf) {
        std::cerr << "av2enc: error: --tiles resets frozen CDFs per tile and does "
                     "not signal --adaptive-cdf\n";
        return 2;
      }
      if (options.still_deblock) {
        std::cerr << "av2enc: error: --tiles requires --no-deblock: the pinned "
                     "dav2d filters across interior tile edges, so multi-tile "
                     "reconstruction would not be sample-exact\n";
        return 2;
      }
      const std::pair<bool, const char*> unsupported[] = {
          {options.still_single, "--still-single"},
          {options.still_dc, "--still-dc"},
          {options.still_full_intra, "--still-full-intra"},
          {options.still_uv_full, "--still-uv-full"},
          {options.still_rdo_intra, "--still-rdo-intra"},
          {options.still_rdoq, "--still-rdoq"},
          {options.still_tx_type, "--still-tx-type"},
          {options.still_cdef, "--still-cdef"},
          {options.still_ccso, "--still-ccso"},
          {options.still_sb128, "--still-sb128"},
          {options.still_sb256, "--still-sb256"},
          {options.still_ext_partitions, "--still-ext-partitions"},
          {options.still_sub8, "--still-sub8"},
          {options.still_sdp, "--still-sdp"},
          {options.still_segmentation, "--still-segmentation"},
          {options.still_delta_q, "--still-delta-q"},
          {options.still_per_segment_q, "--still-per-segment-q"},
          {options.still_content_interpretation,
           "--still-content-interpretation"},
          {options.still_qm, "--still-qm"},
          {options.still_user_qm, "--still-user-qm"},
          {options.still_ist, "--still-ist"},
          {options.still_tcq, "--still-tcq"},
          {options.still_cctx, "--still-cctx"},
          {options.still_tx_partition, "--still-tx-partition"},
          {options.still_idtx, "--still-idtx"},
          {options.still_ddt, "--still-ddt"},
          {options.still_dc_ac_delta_q, "--still-dc-ac-delta-q"},
          {options.still_ibp, "--still-ibp"},
          {options.still_mrl, "--still-mrl"},
          {options.still_cfl_ds != 0, "--still-cfl-ds"},
          {options.still_cfl_explicit, "--still-cfl-explicit"},
          {options.still_dip, "--still-dip"},
          {options.still_palette, "--still-palette"},
          {options.still_ibc, "--still-ibc"},
          {options.still_mhccp, "--still-mhccp"},
          {options.still_ccso_edge, "--still-ccso-edge"},
          {options.still_wiener, "--still-wiener"},
          {options.still_chroma_lr, "--still-chroma-lr"},
          {options.still_pc_wiener, "--still-pc-wiener"},
          {options.still_gdf, "--still-gdf"},
          {options.still_grain, "--still-grain"},
          {options.still_grain_adaptive, "--still-grain-adaptive"},
          {options.still_cdef_on_skip_txfm, "--still-cdef-on-skip-txfm"},
          {options.metadata_timecode, "--metadata-timecode"},
          {options.metadata_scan_type, "--metadata-scan-type"},
          {options.metadata_cll, "--metadata-cll"},
          {options.metadata_mdcv, "--metadata-mdcv"},
          {options.metadata_icc, "--metadata-icc"},
          {options.metadata_temporal_point, "--metadata-temporal-point"},
          {options.metadata_frame_hash, "--metadata-frame-hash"},
          {options.metadata_banding, "--metadata-banding"},
          {options.metadata_user_data, "--metadata-user-data"},
          {options.still_depth >= 0, "--still-depth"},
      };
      for (const auto& [set, flag] : unsupported) {
        if (!set) continue;
        std::cerr << "av2enc: error: --tiles uses the tiled still writer, which "
                     "does not implement "
                  << flag << "; drop the flag or use a single-tile --still "
                           "encode\n";
        return 2;
      }
    }
  }

  // Optical-format still paths (10-bit, 4:2:2/4:4:4/4:0:0).  Every flag
  // defaults to today's 8-bit 4:2:0 stream, which never enters this branch.
  const bool format_10bit = options.bit_depth == 10;
  const bool format_chroma = options.chroma != "420";
  if ((format_10bit || format_chroma) && options.video) {
    if (options.video_lossless) {
      return run_video_lossless_any(options);
    }
    if (options.video_user_qm) {
      std::cerr << "av2enc: error: --video-user-qm is wired for the 8-bit "
                   "4:2:0 video core only\n";
      return 2;
    }
    if (options.video_segmentation || options.video_delta_q ||
        options.video_per_segment_q || options.video_uv_delta_q ||
        options.video_aq) {
      std::cerr << "av2enc: error: segmentation/delta-Q is only wired on the "
                   "8-bit 4:2:0 video core; use --bit-depth 8 --chroma 420\n";
      return 2;
    }
    if (!video_format_wired(options.bit_depth, options.chroma)) {
      std::cerr << "av2enc: error: --video supports 8-bit 4:2:0 and 10-bit "
                   "4:2:0; the remaining --bit-depth/--chroma combinations "
                   "are not verified on the video path yet\n";
      return 2;
    }
    // The format video core reads its own strict Y4M stream and never falls
    // back to the 8-bit 4:2:0 reader below.
    return run_video_format_encode(options);
  }
  if (format_10bit || format_chroma) {
    // Product GPU formats with a parity-gated block stage.  Every product
    // format the CPU format writers support is wired (10-bit
    // 4:2:0/4:2:2/4:4:4/4:0:0 and 8-bit 4:2:2/4:4:4/4:0:0); anything else is
    // rejected by name instead of silently running the CPU encoder under the
    // GPU flag.
    const bool gpu_format_wired =
        format_10bit ||
        (!format_10bit && (options.chroma == "422" ||
                           options.chroma == "444" ||
                           options.chroma == "400"));
    if (options.still_gpu && !gpu_format_wired) {
      std::cerr << "av2enc: error: --still-gpu does not accept this "
                   "--bit-depth/--chroma combination\n";
      return 2;
    }
    if (!options.m1 && !options.still_gpu) {
      std::cerr << "av2enc: error: --bit-depth/--chroma require --still or "
                   "--still-gpu\n";
      return 2;
    }
    if (options.lossless) {
      std::cerr << "av2enc: error: --lossless has its own 8-bit 4:2:0 writer "
                   "and cannot be combined with --bit-depth/--chroma\n";
      return 2;
    }
    if (options.adaptive_cdf) {
      std::cerr << "av2enc: error: --adaptive-cdf is not signalled by the "
                   "format still writers (frozen CDFs)\n";
      return 2;
    }
    if (options.metadata_timecode || options.metadata_scan_type ||
        options.metadata_cll || options.metadata_mdcv ||
        options.metadata_icc || options.metadata_temporal_point ||
        options.metadata_frame_hash || options.metadata_banding ||
        options.metadata_user_data || options.video_brt) {
      std::cerr << "av2enc: error: --metadata-* is not wired into the "
                   "--bit-depth/--chroma format still writers\n";
      return 2;
    }
    // Name the offending flag: the format writer signals none of these tools.
    // 10-bit CDEF/CCSO/LR/GDF are wired; the remaining tools are not.
    const std::pair<bool, const char*> unsupported_by_format[] = {
        {options.still_single, "--still-single"},
        {options.still_dc, "--still-dc"},
        {options.still_full_intra, "--still-full-intra"},
        {options.still_uv_full, "--still-uv-full"},
        {options.still_rdo_intra, "--still-rdo-intra"},
        {options.still_rdoq, "--still-rdoq"},
        {options.still_tx_type, "--still-tx-type"},
        {flag_given(options, "--still-ccso-bands"), "--still-ccso-bands"},
        {options.still_sb128, "--still-sb128"},
        {options.still_sb256, "--still-sb256"},
        {options.still_ext_partitions, "--still-ext-partitions"},
        {options.still_segmentation, "--still-segmentation"},
        {options.still_delta_q, "--still-delta-q"},
        {options.still_per_segment_q, "--still-per-segment-q"},
        {options.still_content_interpretation, "--still-content-interpretation"},
        {options.still_qm, "--still-qm"},
        {options.still_user_qm, "--still-user-qm"},
        {options.still_ist, "--still-ist"},
        {options.still_tcq, "--still-tcq"},
        {options.still_cctx, "--still-cctx"},
        {options.still_tx_partition, "--still-tx-partition"},
        {options.still_idtx, "--still-idtx"},
        {options.still_ddt, "--still-ddt"},
        {options.still_dc_ac_delta_q, "--still-dc-ac-delta-q"},
        {options.still_ibp, "--still-ibp"},
        {options.still_mrl, "--still-mrl"},
        {options.still_sub8, "--still-sub8"},
        {options.still_sdp, "--still-sdp"},
        {flag_given(options, "--still-cfl-ds"), "--still-cfl-ds"},
        {options.still_cfl_explicit, "--still-cfl-explicit"},
        {options.still_dip, "--still-dip"},
        {options.still_palette, "--still-palette"},
        {options.still_ibc, "--still-ibc"},
        {options.still_mhccp, "--still-mhccp"},
        {options.still_grain, "--still-grain"},
        {options.still_depth >= 0, "--still-depth"},
    };
    for (const auto& [set, flag] : unsupported_by_format) {
      if (!set) continue;
      std::cerr << "av2enc: error: " << flag
                << " is not applied by the --bit-depth/--chroma format "
                   "writer: format paths use their own minimal still writer "
                   "(CDEF/CCSO/LR at 10-bit are not implemented)\n";
      return 2;
    }
    if (format_10bit && (options.qp < 1 || options.qp > 303)) {
      std::cerr << "av2enc: error: 10-bit encoding supports base_q_idx 1..303 "
                   "(MaxQ at 10-bit); got --qp "
                << options.qp << "\n";
      return 2;
    }
    if (!format_10bit && options.qp != 90) {
      std::cerr << "av2enc: error: 8-bit 4:2:2/4:4:4/4:0:0 pins --qp 90; got "
                << options.qp << "\n";
      return 2;
    }

    av2enc::formats::StillFormatOptions format_options;
    format_options.depth = format_10bit ? av2enc::fmt::BitDepth::k10
                                        : av2enc::fmt::BitDepth::k8;
    if (options.chroma == "422") {
      format_options.chroma = av2enc::fmt::ChromaFormat::k422;
    } else if (options.chroma == "444") {
      format_options.chroma = av2enc::fmt::ChromaFormat::k444;
    } else if (options.chroma == "400") {
      format_options.chroma = av2enc::fmt::ChromaFormat::k400;
    } else {
      format_options.chroma = av2enc::fmt::ChromaFormat::k420;
    }
    format_options.base_q_idx = static_cast<std::uint16_t>(options.qp);
    format_options.recon_out = options.recon;
    // §7.17 deblocking is opt-in on this path: existing 10-bit streams signal
    // it off and must stay byte-identical.
    if (flag_given(options, "--still-deblock")) {
      if (flag_given(options, "--no-deblock")) {
        std::cerr << "av2enc: error: --still-deblock and --no-deblock are "
                     "mutually exclusive\n";
        return 2;
      }
      format_options.enable_deblock = true;
    }
    // §7.18 CDEF is wired on both format writers.
    if (options.still_cdef) {
      format_options.enable_cdef = true;
      format_options.cdef_num_strengths = options.still_cdef_strengths;
    }
    // §7.19 CCSO is wired on both format writers (band-offset and
    // edge-classified forms are both searched).
    if (options.still_ccso) {
      format_options.enable_ccso = true;
      format_options.ccso_edge_only = options.still_ccso_edge;
    }
    // §7.20 loop restoration and §7.20.5 GDF on both format writers.
    if (options.still_wiener || options.still_chroma_lr ||
        options.still_pc_wiener || options.still_gdf) {
      if (options.still_pc_wiener && options.still_wiener) {
        std::cerr << "av2enc: error: --still-pc-wiener and --still-wiener "
                     "select two luma restoration filters\n";
        return 2;
      }
      format_options.enable_wiener = options.still_wiener;
      format_options.enable_chroma_lr = options.still_chroma_lr;
      format_options.enable_pc_wiener = options.still_pc_wiener;
      format_options.enable_gdf = options.still_gdf;
    }
    std::unique_ptr<av2enc::chroma::ChromaGpuExecutor> chroma_gpu;
#if AV2ENC_STILL_GPU_CALLABLE
    if (options.still_gpu && format_10bit) {
      format_options.high_bitdepth_gpu =
          &av2enc::cuda::still_gpu::encode_format_gpu;
    } else if (options.still_gpu && !format_10bit) {
      chroma_gpu = av2enc::cuda::still_gpu::make_chroma_gpu_executor();
      format_options.chroma_gpu = chroma_gpu.get();
    }
#endif

    av2enc::formats::StillFormatStats stats;
    if (!av2enc::formats::encode_still_format(options.input, options.output,
                                              format_options, &stats, error)) {
      std::cerr << "av2enc: error: " << error << "\n";
      return 1;
    }
    std::cout << "av2enc: Format still-picture encode complete\n"
              << "  input:   " << options.input.string() << " (" << stats.width
              << "x" << stats.height << ", "
              << av2enc::fmt::y4m_colorspace_tag(stats.chroma, stats.depth)
              << ", " << stats.fps_num << "/" << stats.fps_den << " fps)\n"
              << "  output:  " << options.output.string() << " ("
              << stats.ivf_bytes << " bytes IVF)\n"
              << "  format:  " << stats.width << "x" << stats.height << " "
              << av2enc::fmt::bit_depth_name(stats.depth) << ", "
              << av2enc::fmt::chroma_format_name(stats.chroma) << "\n"
              << "  frames:  1, base_q_idx " << options.qp << ", tile "
              << stats.tile_bytes << " bytes, temporal unit "
              << stats.frame_bytes << " bytes\n";
    if (!options.recon.empty()) {
      std::cout << "  recon:   " << options.recon.string() << "\n";
    }
    return 0;
  }

  if (!options.recon.empty() && !multi_tile && !options.m1) {
    std::cerr << "av2enc: error: --recon is wired for the --bit-depth 10 and "
                 "--chroma format paths, --still, and multi-tile --tiles "
                 "stills only\n";
    return 2;
  }
  if (options.qp > 255) {
    std::cerr << "av2enc: error: --qp " << options.qp
              << " exceeds MaxQ = 255 at 8-bit; base_q_idx 256..303 requires "
                 "--still --bit-depth 10\n";
    return 2;
  }

  av2enc::Y4mReader reader(options.input);
  if (!reader.open(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  const av2enc::Y4mHeader& y4m = reader.header();
  const std::uint16_t max_dimension = std::numeric_limits<std::uint16_t>::max();
  if (y4m.width > max_dimension || y4m.height > max_dimension) {
    std::cerr << "av2enc: error: Y4M dimensions " << y4m.width << "x" << y4m.height
              << " exceed the IVF limit of 65535x65535\n";
    return 1;
  }
  // Geometry constraints of the selected mode are usage errors: report them
  // before the output file is opened.
  if (options.video) {
    // The video path codes the §5.18.4.4 MI-aligned picture and crops the
    // output/reconstruction back to the input size.  This reader supplies
    // 8-bit 4:2:0, so only the chroma layout is a hard dimension constraint.
    if ((y4m.width & 1U) != 0 || (y4m.height & 1U) != 0) {
      std::cerr << "av2enc: error: 4:2:0 video requires even dimensions, got "
                << y4m.width << "x" << y4m.height << "\n";
      return 2;
    }
    if (options.video_gpu &&
        (y4m.width % 8U != 0U || y4m.height % 8U != 0U)) {
      std::cerr << "av2enc: error: --video-gpu is CPU-only for dimensions not "
                   "divisible by 8 (the CUDA kernels assume an MI-aligned "
                   "picture); rerun without --video-gpu, got "
                << y4m.width << "x" << y4m.height << "\n";
      return 2;
    }
    if (options.metadata_frame_hash) {
      std::cerr << "av2enc: error: --metadata-frame-hash is wired into the "
                   "still/basic writers only; the --video frame hash needs a "
                   "per-unit carrier\n";
      return 2;
    }
    const int declared_frames =
        options.max_frames != 0
            ? static_cast<int>(options.max_frames)
            : (options.p_frames != 0
                   ? static_cast<int>(options.p_frames + 1)
                   : 0);
    if (options.video_long_term && declared_frames > 16) {
      std::cerr << "av2enc: error: --video-long-term codes OrderHint with "
                   "OrderHintBits = 4, so it supports at most 16 coded frames "
                   "(requested "
                << declared_frames << ")\n";
      return 2;
    }
  } else if (y4m.width % 2U != 0U || y4m.height % 2U != 0U) {
    std::cerr << "av2enc: error: 4:2:0 requires even dimensions, got "
              << y4m.width << "x" << y4m.height << "\n";
    return 2;
  }
  if (options.still_dc_ac_delta_q &&
      (y4m.width % 8U != 0U || y4m.height % 8U != 0U)) {
    std::cerr << "av2enc: error: --still-dc-ac-delta-q requires width and "
                 "height multiples of 8, got "
              << y4m.width << "x" << y4m.height << "\n";
    return 2;
  }

  if (options.svc_given) {
    return run_svc_encode(options, reader, y4m);
  }

  if (options.video) {
    // All video option/sequence validation (including the frame-count and
    // rate-control combinations) happens inside run_video_encode before it
    // opens the output, so an invalid command exits 2 without truncating -o.
    return run_video_encode(options, reader, y4m);
  }

  av2enc::IvfHeader ivf;
  ivf.width = static_cast<std::uint16_t>(y4m.width);
  ivf.height = static_cast<std::uint16_t>(y4m.height);
  ivf.fps_num = y4m.fps_num;
  ivf.fps_den = y4m.fps_den;

  av2enc::IvfWriter writer;
  if (!writer.open(options.output, ivf, error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  const auto fail_encode = [&](const std::string& message) {
    std::string close_error;
    writer.close(close_error);
    std::cerr << "av2enc: error: " << message << "\n";
    if (!close_error.empty()) {
      std::cerr << "av2enc: " << close_error << "\n";
    }
    return 1;
  };
  const auto write_unit = [&](std::vector<std::uint8_t>& unit,
                              std::uint64_t timestamp, std::string& err,
                              const av2enc::metadata::cli::Options::FramePlanes*
                                  planes = nullptr) {
    if (options.video_brt) {
      std::vector<std::uint8_t> brt;
      if (!make_brt_obu(options.brt_time, brt, err) ||
          !insert_before_first_frame_obu(unit, brt, err)) {
        return false;
      }
    }
    if (!append_neutral_metadata_obus(unit, metadata_cli_options(options),
                                      planes, options.metadata_timecode,
                                      options.metadata_scan_type, err)) {
      return false;
    }
    return writer.write_frame(unit, timestamp, err);
  };

  av2enc::Y4mFrame frame;
  const av2enc::Y4mReadResult read_result = reader.read_frame(frame, error);
  if (read_result == av2enc::Y4mReadResult::kEof) {
    return fail_encode("input has no frames");
  }
  if (read_result == av2enc::Y4mReadResult::kError) {
    return fail_encode(error);
  }

  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  int ibc_blocks = 0;

  // Basic's edge recursion cannot code blocks below 32x32 in one dimension
  // (it would need 16x16 / 16x8 / 8x16 / 8x32 / 8x8 coders). Rather than emit
  // a desynchronised stream, use the Still path for those geometries and say
  // so on stderr. Explicit --still / --still-gpu are unaffected.
  bool basic_fallback_to_m1 = false;
  if (!options.m1 && !options.still_gpu && !options.lossless) {
    std::string reason;
    if (!av2enc::basic::geometry_supported(frame.width(), frame.height(),
                                        reason)) {
      std::cerr << "av2enc: note: Basic cannot represent " << frame.width()
                << "x" << frame.height() << " (" << reason
                << "); falling back to the Still path\n";
      basic_fallback_to_m1 = true;
    }
  }

  if (options.lossless) {
    av2enc::lossless::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

    av2enc::lossless::Options lossless_options;
    lossless_options.adaptive_cdf = options.adaptive_cdf;
    lossless_options.fsc = options.lossless_force_fsc
                               ? av2enc::lossless::FscMode::kForce
                               : av2enc::lossless::FscMode::kOff;

    // --still-gpu: the DPCM/FSC decision and the reversible transforms run on
    // the device; the CPU lossless module keeps the §5.20 syntax, the §8.2
    // entropy writer and the decoder-side exactness check.
    std::vector<av2enc::lossless::BlockCoding> gpu_blocks;
#if AV2ENC_STILL_GPU_CALLABLE && AV2ENC_STILL_LOSSLESS_HEADER
    if (options.still_gpu) {
      std::string gpu_error;
      if (!av2enc::cuda::still_lossless::compute_block_coding(
              picture, lossless_options, &gpu_blocks, gpu_error)) {
        return fail_encode(gpu_error);
      }
      lossless_options.precomputed = gpu_blocks.data();
    }
#else
    if (options.still_gpu) {
      return fail_encode(
          "--still-gpu: the lossless DPCM/FSC device stage is not built");
    }
#endif

    av2enc::lossless::EncodeResult encoded;
    if (!av2enc::lossless::encode_still(picture, lossless_options, encoded,
                                        error)) {
      return fail_encode(error);
    }
    if (!encoded.stats.exact) {
      return fail_encode("lossless reconstruction is not exact (max |err| " +
                         std::to_string(encoded.stats.max_abs_recon_error) +
                         ")");
    }
    const auto planes = make_plane_view(
        picture.y, picture.u, picture.v, picture.width, picture.height);
    if (!write_unit(encoded.temporal_unit, 0, error, &planes)) {
      return fail_encode(error);
    }
    tile_bytes = encoded.stats.tile_bytes;
    frame_bytes = encoded.stats.frame_bytes;
  } else if (options.m1 || options.still_gpu || basic_fallback_to_m1) {
    av2enc::still::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

#if AV2ENC_STILL_GPU_CALLABLE
    // GpuOptions mirrors StillOptions and adds orchestrator controls; see
    // src/cuda/pipeline/still/still_gpu.h and docs/m1-gpu.md.  One builder is
    // shared by the single-tile and multi-tile (per-tile) GPU paths so a parsed
    // flag can never reach one branch and not the other.
    const auto build_gpu_options = [&]() {
      av2enc::cuda::still_gpu::GpuOptions gpu_options;
      gpu_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
      gpu_options.enable_cfl = options.still_cfl;
      gpu_options.cfl_ds_filter = options.still_cfl_ds;
      gpu_options.enable_deblock = options.still_deblock;
      gpu_options.enable_edge_filter = true;
      gpu_options.luma_dc_only = !options.still_full_intra &&
                                 !options.still_rdo_intra && !options.still_mrl;
      gpu_options.min_block_size = 8;
      gpu_options.sb_size = options.still_sb256 ? 256
                            : options.still_sb128 ? 128
                                                  : 64;
      gpu_options.enable_ibp = options.still_ibp;
      gpu_options.enable_mrl = options.still_mrl;
      gpu_options.enable_dip = options.still_dip;
      gpu_options.cfl_explicit = options.still_cfl_explicit;
      gpu_options.mhccp = options.still_mhccp;
      gpu_options.palette = options.still_palette;
      gpu_options.ibc = options.still_ibc;
      gpu_options.ibc_search_step = options.still_ibc_search_step;
      gpu_options.ext_partitions = options.still_ext_partitions;
      gpu_options.sdp = options.still_sdp;
      gpu_options.sub8 = options.still_sub8;
      if (options.still_depth >= 0) {
        gpu_options.max_partition_depth = options.still_depth;
      }
      gpu_options.adaptive_cdf = options.adaptive_cdf;
      gpu_options.force_single_block = options.still_single;
      gpu_options.force_dc = options.still_dc;
      gpu_options.uv_full = options.still_uv_full;
      gpu_options.enable_cdef = options.still_cdef;
      gpu_options.cdef_num_strengths = options.still_cdef_strengths;
      gpu_options.cdef_on_skip_txfm = options.still_cdef_on_skip_txfm;
      gpu_options.enable_ccso = options.still_ccso;
      gpu_options.ccso_bands_log2 = options.still_ccso_bands;
      gpu_options.ccso_edge = options.still_ccso_edge;
      gpu_options.enable_wiener = options.still_wiener;
      gpu_options.enable_chroma_lr = options.still_chroma_lr;
      gpu_options.enable_pc_wiener = options.still_pc_wiener;
      gpu_options.enable_gdf = options.still_gdf;
      gpu_options.enable_grain = options.still_grain;
      gpu_options.grain_adaptive = options.still_grain_adaptive;
      gpu_options.content_interpretation =
          options.still_content_interpretation;
      gpu_options.delta_q = options.still_dc_ac_delta_q;
      gpu_options.enable_segmentation = options.still_segmentation;
      gpu_options.enable_delta_q = options.still_delta_q;
      gpu_options.delta_q_res = options.still_delta_q_res;
      gpu_options.per_segment_q = options.still_per_segment_q;
      gpu_options.per_segment_q_delta = options.still_seg_delta;
      gpu_options.qm = options.still_qm;
      gpu_options.qm_level = options.still_qm_level;
      gpu_options.rdoq = options.still_rdoq;
      gpu_options.rdoq_sse_scale = options.still_rdoq_scale;
      gpu_options.tx_type = options.still_tx_type;
      gpu_options.tx_type_sse_scale = options.still_tx_type_scale;
      gpu_options.idtx = options.still_idtx;
      gpu_options.ist = options.still_ist;
      gpu_options.ist_sse_scale = options.still_ist_scale;
      gpu_options.tcq = options.still_tcq;
      gpu_options.tcq_sse_scale = options.still_tcq_scale;
      gpu_options.cctx = options.still_cctx;
      gpu_options.tx_partition = options.still_tx_partition;
      gpu_options.rdo_intra = options.still_rdo_intra;
      gpu_options.rdo_sse_scale = options.still_rdo_scale;
      gpu_options.vram_cap_mb = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(options.vram_cap_mib,
                                  std::numeric_limits<std::uint32_t>::max()));
      return gpu_options;
    };
#endif

    if (options.still_gpu && !multi_tile) {
#if AV2ENC_STILL_GPU_CALLABLE
      av2enc::cuda::still_gpu::GpuOptions gpu_options = build_gpu_options();
      av2enc::cuda::still_gpu::GpuResult encoded;
      if (!av2enc::cuda::still_gpu::encode_still_gpu(picture, gpu_options, encoded,
                                              error)) {
        return fail_encode(error);
      }
      if (encoded.frame_bytes_from_cpu) {
        std::cerr << "av2enc: note: frame_bytes_from_cpu=1; frame bytes came "
                     "from the CPU Still reference (linked stage hooks: "
                  << av2enc::cuda::still_gpu::linked_stage_count() << ")\n";
      }
      const auto planes =
          make_plane_view(encoded.recon_y, encoded.recon_u, encoded.recon_v,
                          picture.width, picture.height);
      if (!write_unit(encoded.frame, 0, error, &planes)) {
        return fail_encode(error);
      }
      tile_bytes = encoded.tile_bytes;
      frame_bytes = encoded.frame_bytes;
#else
      return fail_encode("--still-gpu is not available in this build");
#endif
    } else if (multi_tile) {
      // One independent §8.2 entropy stream per tile; every tile is encoded
      // from its own local origin through the Still reference, so CDFs, above/left
      // contexts and prediction availability reset at the tile boundary
      // (§5.20.2.1, §5.20.9.1).  Deblocking is off (checked above): the pinned
      // decoder filters shared tile edges and would break exactness.
      av2enc::still::tiles_stream::Config tile_config;
      tile_config.width = static_cast<int>(frame.width());
      tile_config.height = static_cast<int>(frame.height());
      tile_config.preset = tile_preset_from_layout(options.tiles);
      tile_config.base_q_idx = static_cast<std::uint8_t>(options.qp);
      tile_config.enable_deblock = false;
      tile_config.disable_loopfilters_across_tiles = true;
      tile_config.seq_level_idx = 7;
      tile_config.enable_cfl_intra = options.still_cfl;
      tile_config.enable_intra_edge_filter = true;
      tile_config.enable_chroma_dctonly = true;
      tile_config.adaptive_cdf = options.avg_cdf;

      av2enc::still::tiles_stream::Frame tiled;
      std::string tile_error;
      bool tiled_ok = false;
#if AV2ENC_STILL_GPU_CALLABLE
      if (options.still_gpu) {
        // Per-tile plan/search/TX/recon on the device.  The CPU tile-group
        // writer still owns the tiled headers, the §8.2.2 seed and the §8.2.4
        // average-CDF bank; this callback only replaces the per-tile encoder.
        const auto gpu_tile_encoder =
            [&](const av2enc::still::StillPicture& tile_picture,
                const av2enc::still::StillOptions& tile_options,
                std::span<const std::uint16_t> cdf_seed,
                av2enc::still::tiles_stream::TileEncodeOutput* out,
                std::string& tile_encode_error) -> bool {
          av2enc::cuda::still_gpu::GpuOptions gpu_options = build_gpu_options();
          gpu_options.base_q_idx = tile_options.base_q_idx;
          gpu_options.enable_cfl = tile_options.enable_cfl;
          gpu_options.enable_deblock = tile_options.enable_deblock;
          gpu_options.adaptive_cdf = tile_options.adaptive_cdf;
          gpu_options.cdf_seed.assign(cdf_seed.begin(), cdf_seed.end());
          gpu_options.min_block_size = tile_options.min_block_size;
          gpu_options.max_partition_depth = tile_options.max_partition_depth;
          gpu_options.luma_dc_only = tile_options.luma_dc_only;
          gpu_options.verify_against_cpu = true;
          av2enc::cuda::still_gpu::GpuResult encoded;
          if (!av2enc::cuda::still_gpu::encode_still_gpu(
                  tile_picture, gpu_options, encoded, tile_encode_error)) {
            return false;
          }
          out->tile_bytes = encoded.tile_bytes;
          out->frame = std::move(encoded.frame);
          out->recon_y = std::move(encoded.recon_y);
          out->recon_u = std::move(encoded.recon_u);
          out->recon_v = std::move(encoded.recon_v);
          out->cdf_final = std::move(encoded.cdf_final);
          return true;
        };
        tiled_ok = av2enc::still::tiles_stream::encode(
            picture, tile_config, &tiled, tile_error, gpu_tile_encoder);
      } else
#endif
      {
        tiled_ok = av2enc::still::tiles_stream::encode(
            picture, tile_config, &tiled, tile_error);
      }
      if (!tiled_ok) {
        return fail_encode("--tiles " + options.tiles + ": " + tile_error);
      }
      if (!write_unit(tiled.frame, 0, error)) {
        return fail_encode(error);
      }
      std::uint32_t tile_payload_bytes = 0;
      for (const auto& tile : tiled.tiles) {
        tile_payload_bytes += static_cast<std::uint32_t>(tile.payload.size());
      }
      tile_bytes = tile_payload_bytes;
      frame_bytes = static_cast<std::uint32_t>(tiled.frame.size());

      if (!options.recon.empty()) {
        av2enc::still::tiles_stream::Reconstruction recon;
        std::string recon_error;
        if (!av2enc::still::tiles_stream::assemble_reconstruction(
                tiled, &recon, recon_error)) {
          return fail_encode(recon_error);
        }
        if (!write_still_recon_y4m(options.recon, y4m, recon.width,
                                   recon.height, recon.y, recon.u, recon.v,
                                   error)) {
          return fail_encode(error);
        }
      }
      std::cout << "  tiles:   " << tiled.tiles.size() << " (" << options.tiles
                << "), tile group " << tiled.clk_payload.size() << " bytes ("
                << tiled.tile_size_bytes << "-byte size fields)\n";
    } else {
      av2enc::still::StillOptions still_options;
      still_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
      still_options.enable_cfl = options.still_cfl;
      still_options.enable_deblock = options.still_deblock;
      still_options.adaptive_cdf = options.adaptive_cdf;
      still_options.force_single_block = options.still_single;
      still_options.force_dc = options.still_dc;
      still_options.luma_dc_only =
          !options.still_full_intra && !options.still_rdo_intra &&
          !options.still_mrl;
      still_options.uv_full = options.still_uv_full;
      still_options.enable_mrl = options.still_mrl;
      still_options.cfl_ds_filter = options.still_cfl_ds;
      still_options.cfl_explicit = options.still_cfl_explicit;
      still_options.enable_dip = options.still_dip;
      still_options.palette = options.still_palette;
      still_options.ibc = options.still_ibc;
      still_options.ibc_search_step = options.still_ibc_search_step;
      still_options.mhccp = options.still_mhccp;
      still_options.rdo_intra = options.still_rdo_intra;
      still_options.rdo_sse_scale = options.still_rdo_scale;
      still_options.rdoq = options.still_rdoq;
      still_options.rdoq_sse_scale = options.still_rdoq_scale;
      still_options.tx_type = options.still_tx_type;
      still_options.tx_type_sse_scale = options.still_tx_type_scale;
      still_options.enable_cdef = options.still_cdef;
      still_options.cdef_num_strengths = options.still_cdef_strengths;
      still_options.cdef_on_skip_txfm = options.still_cdef_on_skip_txfm;
      still_options.enable_ccso = options.still_ccso;
      still_options.ccso_bands_log2 = options.still_ccso_bands;
      still_options.sb128 = options.still_sb128;
      still_options.sb256 = options.still_sb256;
      still_options.ext_partitions = options.still_ext_partitions;
      still_options.sub8 = options.still_sub8;
      still_options.sdp = options.still_sdp;
      still_options.enable_segmentation = options.still_segmentation;
      still_options.enable_delta_q = options.still_delta_q;
      still_options.delta_q_res = options.still_delta_q_res;
      still_options.per_segment_q = options.still_per_segment_q;
      still_options.per_segment_q_delta = options.still_seg_delta;
      still_options.content_interpretation =
          options.still_content_interpretation;
      still_options.qm = options.still_qm;
      still_options.qm_level = options.still_qm_level;
      still_options.user_qm = options.still_user_qm;
      still_options.ist = options.still_ist;
      still_options.ist_sse_scale = options.still_ist_scale;
      still_options.tcq = options.still_tcq;
      still_options.tcq_sse_scale = options.still_tcq_scale;
      still_options.cctx = options.still_cctx;
      still_options.tx_partition = options.still_tx_partition;
      still_options.idtx = options.still_idtx;
      still_options.ddt = options.still_ddt;
      still_options.delta_q = options.still_dc_ac_delta_q;
      still_options.enable_ibp = options.still_ibp;
      still_options.ccso_edge = options.still_ccso_edge;
      still_options.enable_wiener = options.still_wiener;
      still_options.enable_chroma_lr = options.still_chroma_lr;
      still_options.enable_pc_wiener = options.still_pc_wiener;
      still_options.enable_gdf = options.still_gdf;
      still_options.enable_grain = options.still_grain;
      still_options.grain_adaptive = options.still_grain_adaptive;
      if (options.still_depth >= 0) {
        still_options.max_partition_depth = options.still_depth;
      }

      av2enc::still::StillResult encoded;
      if (!av2enc::still::encode_still_picture(picture, still_options, encoded,
                                               error)) {
        return fail_encode(error);
      }
      const auto planes =
          make_plane_view(encoded.recon_y, encoded.recon_u, encoded.recon_v,
                          picture.width, picture.height);
      if (!write_unit(encoded.frame, 0, error, &planes)) {
        return fail_encode(error);
      }
      if (!options.recon.empty()) {
        if (!write_still_recon_y4m(options.recon, y4m,
                                   static_cast<int>(y4m.width),
                                   static_cast<int>(y4m.height),
                                   encoded.recon_y, encoded.recon_u,
                                   encoded.recon_v, error)) {
          return fail_encode(error);
        }
      }
      tile_bytes = encoded.tile_bytes;
      frame_bytes = encoded.frame_bytes;
      ibc_blocks = encoded.ibc_blocks;
    }
  } else {
    av2enc::basic::StillPicture picture;
    picture.width = frame.width();
    picture.height = frame.height();
    picture.y = frame.y();
    picture.u = frame.u();
    picture.v = frame.v();

    av2enc::basic::StillPictureConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(y4m.width);
    config.max_frame_height = static_cast<std::uint16_t>(y4m.height);
    config.base_q_idx = static_cast<std::uint8_t>(options.qp);
    // Frozen CDFs are the default Basic bitstream. Adaptive coding is an opt-in
    // normative §8.2.6 mode used to test the CDF-update path.
    config.enable_cdf_adaptation = options.adaptive_cdf;

    av2enc::basic::StillEncodeResult encoded;
    if (!av2enc::basic::encode_still_picture(picture, config, encoded, error)) {
      return fail_encode(error);
    }
    const auto planes =
        make_plane_view(encoded.recon_y, encoded.recon_u, encoded.recon_v,
                        picture.width, picture.height);
    if (!write_unit(encoded.frame, 0, error, &planes)) {
      return fail_encode(error);
    }
    tile_bytes = encoded.tile_bytes;
    frame_bytes = encoded.frame_bytes;
  }

  if (!writer.close(error)) {
    std::cerr << "av2enc: " << error << "\n";
    return 1;
  }

  const char* path_name =
      options.lossless
          ? "Lossless"
          : (options.still_gpu
                 ? "Still GPU"
                 : (options.m1 ? (multi_tile ? "Still (tiled)" : "Still")
                               : (basic_fallback_to_m1 ? "Still (Basic fallback)"
                                                       : "Basic")));
  std::cout << "av2enc: " << path_name
            << " still-picture encode complete\n"
            << "  input:   " << options.input.string() << " (" << y4m.width
            << "x" << y4m.height << ", " << y4m.colorspace << ", "
            << y4m.fps_num << "/" << y4m.fps_den << " fps)\n"
            << "  output:  " << options.output.string() << " ("
            << writer.bytes_written() << " bytes IVF)\n"
            << "  frames:  1, base_q_idx "
            << (options.lossless ? 0 : options.qp) << ", tile "
            << tile_bytes << " bytes, temporal unit " << frame_bytes
            << " bytes\n";
  if (ibc_blocks > 0) {
    std::cout << "  ibc:     " << ibc_blocks << " exact-copy block(s)\n";
  }
  return 0;
}

// Entry point wrapper: untrusted input (a Y4M header) can still ask for an
// allocation larger than the machine has even after the pixel budget, and a
// malformed CLI value can throw from the standard library.  Report either as a
// normal error instead of letting std::terminate print a SIGABRT.
int main(int argc, char** argv) {
  try {
    return av2enc_main(argc, argv);
  } catch (const std::bad_alloc&) {
    std::cerr << "av2enc: error: out of memory\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "av2enc: error: " << e.what() << "\n";
    return 1;
  }
}
