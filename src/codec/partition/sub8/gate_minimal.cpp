// Writes the minimal IVF streams for the sub-8 dav2d gate.
//
// Usage: gate_minimal <output-dir>
//
// Every stream is a complete AV2 still picture (TD + sequence header + CLK)
// assembled with the production M1 header writer.  The tile payload is built
// by sub8_syntax.cpp.  The tree names describe the partition decisions:
//
//   baseline_none   8x8   NONE                      (no sub-8 block; control)
//   vert_leaf4x8    8x8   VERT -> 4x8,4x8           (4x8 leaves)
//   vert_first4x4   8x8   VERT -> [HORZ -> 4x4,4x4], 4x8
//   vert_last4x4    8x8   VERT -> 4x8, [HORZ -> 4x4,4x4]
//   vert_all4x4     8x8   VERT -> [HORZ]x2          (four 4x4 leaves)
//   horz_leaf8x4    8x8   HORZ -> 8x4,8x4           (8x4 leaves)
//   horz_first4x4   8x8   HORZ -> [VERT -> 4x4,4x4], 8x4
//   horz_last4x4    8x8   HORZ -> 8x4, [VERT -> 4x4,4x4]
//   horz_all4x4     8x8   HORZ -> [VERT]x2
//   edge16_all4x4   16x16 16x8->VERT 8x8->HORZ 8x4->VERT, whole frame
//   multi_sb_128x64 128x64 two full 64x64 SBs, each fully decomposed to 4x4
//
// The 16x16 stream puts sub-8 leaves at the right and bottom coded-plane edge;
// the 128x64 stream exercises two superblocks and 512 sub-8 leaves.
#include "codec/partition/sub8/sub8_minimal.h"

#include "core/io/ivf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using av2enc::still::sub8::BlockSize;
using av2enc::still::sub8::ChromaInfo;
using av2enc::still::sub8::DecodedLeaf;
using av2enc::still::sub8::FrameGeometry;
using av2enc::still::sub8::Node;
using av2enc::still::sub8::PartitionPlan;
using av2enc::still::sub8::PartitionType;
using av2enc::still::sub8::SyntaxOptions;

struct Shape {
  const char* name;
  int width;
  int height;
  const char* description;
};

enum class Policy { kMode8, kEdge16, kDeep };

PartitionType choose_8x8(const Node& node, const PartitionPlan&, const char* mode) {
  const bool first = mode[1] == 'F' || mode[1] == 'S';
  const bool last = mode[1] == 'L' || mode[1] == 'S';
  if (node.size == BlockSize::k8x8) {
    if (mode[0] == 'V') return PartitionType::kVert;
    if (mode[0] == 'H') return PartitionType::kHorz;
    return PartitionType::kNone;
  }
  if (node.size == BlockSize::k4x8 && first) {
    if (node.mi_col == 0) return PartitionType::kHorz;
  }
  if (node.size == BlockSize::k4x8 && last) {
    if (node.mi_col != 0) return PartitionType::kHorz;
  }
  if (node.size == BlockSize::k8x4 && first) {
    if (node.mi_row == 0) return PartitionType::kVert;
  }
  if (node.size == BlockSize::k8x4 && last) {
    if (node.mi_row != 0) return PartitionType::kVert;
  }
  return PartitionType::kNone;
}

PartitionType choose_deep(const Node& node, const PartitionPlan&) {
  if (node.size == BlockSize::k64x64) return PartitionType::kHorz;
  if (node.size == BlockSize::k64x32) return PartitionType::kVert;
  if (node.size == BlockSize::k32x32) return PartitionType::kHorz;
  if (node.size == BlockSize::k32x16) return PartitionType::kVert;
  if (node.size == BlockSize::k16x16) return PartitionType::kHorz;
  if (node.size == BlockSize::k16x8) return PartitionType::kVert;
  if (node.size == BlockSize::k8x8) return PartitionType::kHorz;
  if (node.size == BlockSize::k8x4) return PartitionType::kVert;
  return PartitionType::kNone;
}

PartitionType choose_edge16(const Node& node, const PartitionPlan&) {
  if (node.size == BlockSize::k16x16) return PartitionType::kHorz;  // -> 16x8
  if (node.size == BlockSize::k16x8) return PartitionType::kVert;  // -> 8x8
  if (node.size == BlockSize::k8x8) return PartitionType::kHorz;    // -> 8x4
  if (node.size == BlockSize::k8x4) return PartitionType::kVert;    // -> 4x4
  return PartitionType::kNone;
}

bool write_ivf(const std::string& path, const FrameGeometry& geometry,
               const std::vector<std::uint8_t>& frame, std::string& error) {
  av2enc::IvfWriter writer;
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(geometry.width);
  header.height = static_cast<std::uint16_t>(geometry.height);
  header.fps_num = 30;
  header.fps_den = 1;
  header.frame_count = 1;
  if (!writer.open(path, header, error)) return false;
  if (!writer.write_frame(frame, 0, error)) return false;
  return writer.close(error);
}

bool emit(const std::string& dir, const Shape& shape, const char* mode,
          Policy policy, std::string& error) {
  FrameGeometry geometry;
  geometry.width = shape.width;
  geometry.height = shape.height;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  const auto chooser = [mode, policy](const Node& node,
                                      const PartitionPlan& plan) {
    if (policy == Policy::kEdge16) return choose_edge16(node, plan);
    if (policy == Policy::kDeep) return choose_deep(node, plan);
    return choose_8x8(node, plan, mode);
  };

  const int sb_mi = 64 / 4;
  const int mi_rows = av2enc::partition::mi_rows(geometry);
  const int mi_cols = av2enc::partition::mi_cols(geometry);
  std::vector<Node> superblocks;
  for (int row = 0; row * sb_mi < mi_rows; ++row) {
    for (int col = 0; col * sb_mi < mi_cols; ++col) {
      Node sb;
      if (!av2enc::still::sub8::build_superblock(
              geometry, row * sb_mi, col * sb_mi, chooser, sb, error)) {
        return false;
      }
      superblocks.push_back(std::move(sb));
    }
  }

  SyntaxOptions options;
  std::vector<std::uint8_t> frame;
  std::vector<DecodedLeaf> leaves;
  if (!av2enc::still::sub8::make_minimal_frame(
          geometry, superblocks, options, frame, &leaves, error)) {
    return false;
  }
  const std::string path = dir + "/" + shape.name + ".ivf";
  if (!write_ivf(path, geometry, frame, error)) return false;

  int sub8_leaves = 0;
  for (const DecodedLeaf& leaf : leaves) {
    if (av2enc::still::sub8::is_sub8(leaf.size)) ++sub8_leaves;
  }
  std::printf("%s %d %d %d %zu %s\n", shape.name, shape.width, shape.height,
              sub8_leaves, frame.size(), shape.description);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  std::string error;

  struct Case {
    Shape shape;
    const char* mode;
    Policy policy;
  };
  const Case cases[] = {
      {{"baseline_none", 8, 8, "8x8 NONE"}, "N", Policy::kMode8},
      {{"vert_leaf4x8", 8, 8, "VERT -> 4x8,4x8"}, "V", Policy::kMode8},
      {{"vert_first4x4", 8, 8, "VERT -> 4x4,4x4,4x8"}, "VF", Policy::kMode8},
      {{"vert_last4x4", 8, 8, "VERT -> 4x8,4x4,4x4"}, "VL", Policy::kMode8},
      {{"vert_all4x4", 8, 8, "VERT -> 4x4 x4"}, "VS", Policy::kMode8},
      {{"horz_leaf8x4", 8, 8, "HORZ -> 8x4,8x4"}, "H", Policy::kMode8},
      {{"horz_first4x4", 8, 8, "HORZ -> 4x4,4x4,8x4"}, "HF", Policy::kMode8},
      {{"horz_last4x4", 8, 8, "HORZ -> 8x4,4x4,4x4"}, "HL", Policy::kMode8},
      {{"horz_all4x4", 8, 8, "HORZ -> 4x4 x4"}, "HS", Policy::kMode8},
      {{"edge16_all4x4", 16, 16, "16x8 VERT, 8x8 HORZ, 8x4 VERT"}, "",
       Policy::kEdge16},
      {{"multi_sb_128x64", 128, 64, "two 64x64 SBs, 4x4 x512"}, "",
       Policy::kDeep},
  };

  for (const Case& item : cases) {
    if (!emit(dir, item.shape, item.mode, item.policy, error)) {
      std::fprintf(stderr, "gate: %s: %s\n", item.shape.name, error.c_str());
      return 1;
    }
  }
  return 0;
}
