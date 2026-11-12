#include "codec/inter/leading/leading_encode.h"

#include "codec/inter/leading/leading_reader.h"

#include <span>

namespace av2enc::inter::leading {
namespace {

av2enc::video::VideoInputFrame to_video_input(const Picture& picture) {
  av2enc::video::VideoInputFrame input;
  input.width = picture.width;
  input.height = picture.height;
  input.y = std::span<const std::uint8_t>(picture.y);
  input.u = std::span<const std::uint8_t>(picture.u);
  input.v = std::span<const std::uint8_t>(picture.v);
  return input;
}

Picture to_picture(const av2enc::video::VideoReconFrame& recon) {
  Picture picture;
  picture.width = recon.width;
  picture.height = recon.height;
  picture.y = recon.y;
  picture.u = recon.u;
  picture.v = recon.v;
  return picture;
}

bool extract_last_tile_payload(const std::vector<std::uint8_t>& unit,
                               const SequenceConfig& config,
                               std::vector<std::uint8_t>& tile,
                               std::string& error) {
  std::vector<ParsedObu> obus;
  if (!parse_obu_sequence(unit, obus, error)) return false;
  for (auto it = obus.rbegin(); it != obus.rend(); ++it) {
    if (!is_tile_group_obu(it->type)) continue;
    ParsedFrameHeader header;
    if (!parse_frame_header(it->payload, it->type, config, header, error)) {
      return false;
    }
    if (header.header_bytes >= it->payload.size()) {
      error = "tile payload is empty";
      return false;
    }
    tile.assign(it->payload.begin() +
                    static_cast<std::ptrdiff_t>(header.header_bytes),
                it->payload.end());
    return true;
  }
  error = "frame OBU not found";
  return false;
}

SequenceConfig config_for(const av2enc::video::VideoOptions& options,
                          const Picture& picture) {
  SequenceConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.base_q_idx = options.base_q_idx;
  // M2 writes these sequence fields unconditionally.
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.max_drl_bits_minus_1 = 4;
  config.monotonic_output_order_flag = true;
  return config;
}

}  // namespace

bool encode_intra_picture(const Picture& source, const av2enc::video::VideoOptions& options,
                          EncodedPicture& out, std::string& error) {
  error.clear();
  out = EncodedPicture{};
  const av2enc::video::VideoInputFrame input = to_video_input(source);
  av2enc::video::VideoSequenceResult sequence;
  if (!av2enc::video::encode_video_sequence(std::span<const av2enc::video::VideoInputFrame>(&input, 1),
                              options, sequence, error)) {
    error = "intra picture: " + error;
    return false;
  }
  if (sequence.units.empty() || sequence.recon.empty()) {
    error = "intra picture: M2 produced no output";
    return false;
  }
  if (!extract_last_tile_payload(sequence.units.front(),
                                 config_for(options, source),
                                 out.tile_payload, error)) {
    error = "intra picture: " + error;
    return false;
  }
  out.recon = to_picture(sequence.recon.front());
  return true;
}

bool encode_inter_picture(const Picture& reference, const Picture& source,
                          const av2enc::video::VideoOptions& options, EncodedPicture& out,
                          std::string& error) {
  error.clear();
  out = EncodedPicture{};
  if (reference.width != source.width || reference.height != source.height) {
    error = "inter picture: reference and source dimensions differ";
    return false;
  }
  const av2enc::video::VideoInputFrame inputs[2] = {to_video_input(reference),
                                      to_video_input(source)};
  av2enc::video::VideoSequenceResult sequence;
  if (!av2enc::video::encode_video_sequence(std::span<const av2enc::video::VideoInputFrame>(inputs, 2),
                              options, sequence, error)) {
    error = "inter picture: " + error;
    return false;
  }
  if (sequence.units.size() != 2U || sequence.recon.size() != 2U) {
    error = "inter picture: M2 did not produce a P frame";
    return false;
  }
  if (!extract_last_tile_payload(sequence.units[1], config_for(options, source),
                                 out.tile_payload, error)) {
    error = "inter picture: " + error;
    return false;
  }
  out.recon = to_picture(sequence.recon[1]);
  return true;
}

}  // namespace av2enc::inter::leading
